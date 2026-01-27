#!/usr/bin/env python3
"""
Analyze preprocessed TDC data quality in DMP-EGNN/data/processed_tdc_data.

What this checks (per dataset/seed/split):
- Completeness: number of graphs loaded
- Featurization drop ratio (if raw CSV is available): compares split sizes against raw train_val/test sizes
- Structural validity: missing required fields, shape mismatches, edge_index range, NaN/Inf
- Descriptor sanity: descriptor dim consistency, all-zero descriptor ratio
- Label distribution: min/max/mean/std, unique values, positive rate (if appears binary)

Output:
- JSON: analysis/preprocess_quality_report.json
- CSV:  analysis/preprocess_quality_report.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _count_csv_rows(path: Path) -> Optional[int]:
    if not path.exists():
        return None
    try:
        with path.open("r", newline="") as f:
            reader = csv.reader(f)
            # assume header
            n = -1
            for n, _ in enumerate(reader):
                pass
            return max(0, n)  # minus header row
    except Exception:
        return None


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None


def _tensor_isfinite(t) -> Optional[bool]:
    try:
        import torch

        if t is None:
            return None
        if not isinstance(t, torch.Tensor):
            return None
        return bool(torch.isfinite(t).all().item())
    except Exception:
        return None


def _to_1d_float_list(y) -> List[float]:
    """Best-effort extraction of labels to a list[float] (per sample)."""
    import torch

    out: List[float] = []
    if y is None:
        return out
    if isinstance(y, torch.Tensor):
        # If shape is (1,) or (1,1) etc, flatten.
        yy = y.detach().cpu().flatten()
        for v in yy.tolist():
            fv = _safe_float(v)
            if fv is not None:
                out.append(fv)
        return out
    # Fallback for numeric
    fv = _safe_float(y)
    if fv is not None:
        out.append(fv)
    return out


def _looks_binary(values: List[float], tol: float = 1e-6) -> bool:
    if not values:
        return False
    for v in values:
        if abs(v - 0.0) <= tol:
            continue
        if abs(v - 1.0) <= tol:
            continue
        return False
    return True


def _mean_std(values: List[float]) -> Tuple[Optional[float], Optional[float]]:
    if not values:
        return None, None
    mean = sum(values) / len(values)
    var = sum((x - mean) ** 2 for x in values) / len(values)
    return mean, math.sqrt(var)


@dataclass
class SplitQuality:
    dataset: str
    seed: str
    split: str  # train/valid/test
    n_graphs: int

    # Raw reference (optional)
    raw_train_val_rows: Optional[int]
    raw_test_rows: Optional[int]

    # Structural checks
    missing_x: int
    missing_edge_index: int
    missing_y: int
    missing_pos: int
    missing_edge_attr: int
    missing_descriptor: int
    node_pos_mismatch: int
    edge_index_oob: int
    any_nonfinite: int
    nonfinite_x: int
    nonfinite_pos: int
    nonfinite_edge_attr: int
    nonfinite_descriptor: int
    nonfinite_y: int

    # Feature dims (summary)
    x_dim_min: Optional[int]
    x_dim_max: Optional[int]
    edge_attr_dim_min: Optional[int]
    edge_attr_dim_max: Optional[int]
    descriptor_dim_min: Optional[int]
    descriptor_dim_max: Optional[int]

    # Descriptor sanity
    descriptor_all_zero: int

    # Label stats
    label_count: int
    label_unique_approx: int
    label_min: Optional[float]
    label_max: Optional[float]
    label_mean: Optional[float]
    label_std: Optional[float]
    label_pos_rate: Optional[float]  # only if binary-like
    nonfinite_examples: Optional[List[Dict[str, Any]]]  # optional, for debugging


def _load_pt(path: Path):
    import torch

    obj = torch.load(str(path), map_location="cpu", weights_only=False)
    # Some intermediate checkpoints store dict with "graphs"
    if isinstance(obj, dict) and "graphs" in obj:
        return obj["graphs"]
    return obj


def _iter_graphs(obj) -> Iterable[Any]:
    """Support list[Data] or legacy tuple format."""
    # Preferred: list of torch_geometric Data
    if isinstance(obj, list):
        return obj
    # Legacy: tuple(smiles, x, edge_index, desc, y) where each could be list/tensor
    if isinstance(obj, tuple) and len(obj) == 5:
        smiles, xs, edge_indices, descs, ys = obj
        # try to index by sample length
        n = None
        for candidate in (smiles, ys, xs):
            try:
                n = len(candidate)
                break
            except Exception:
                continue
        if n is None:
            return []
        # Build minimal records (dict-like) to keep downstream checks consistent
        recs = []
        for i in range(n):
            recs.append(
                {
                    "x": xs[i] if isinstance(xs, list) else xs[i],
                    "edge_index": edge_indices[i] if isinstance(edge_indices, list) else edge_indices[i],
                    "descriptor": descs[i] if isinstance(descs, list) else descs[i],
                    "y": ys[i] if isinstance(ys, list) else ys[i],
                    "smiles": smiles[i] if isinstance(smiles, list) else None,
                }
            )
        return recs
    return []


def _get_attr(g: Any, name: str):
    if isinstance(g, dict):
        return g.get(name)
    return getattr(g, name, None)


def analyze_split(dataset: str, seed: str, split: str, pt_path: Path, raw_train_val_rows: Optional[int], raw_test_rows: Optional[int]) -> SplitQuality:
    import torch

    graphs_obj = _load_pt(pt_path)
    graphs = list(_iter_graphs(graphs_obj))
    n = len(graphs)

    missing_x = missing_edge_index = missing_y = 0
    missing_pos = missing_edge_attr = missing_descriptor = 0
    node_pos_mismatch = 0
    edge_index_oob = 0
    any_nonfinite = 0
    nonfinite_x = 0
    nonfinite_pos = 0
    nonfinite_edge_attr = 0
    nonfinite_descriptor = 0
    nonfinite_y = 0

    x_dims: List[int] = []
    edge_attr_dims: List[int] = []
    desc_dims: List[int] = []
    desc_all_zero = 0

    labels: List[float] = []

    # examples are optionally collected (controlled by main via env var to keep signature stable)
    max_examples = int(os.environ.get("DMP_EGNN_QA_MAX_EXAMPLES", "0") or "0")
    examples: List[Dict[str, Any]] = []

    for idx, g in enumerate(graphs):
        x = _get_attr(g, "x")
        edge_index = _get_attr(g, "edge_index")
        y = _get_attr(g, "y")
        pos = _get_attr(g, "pos")
        edge_attr = _get_attr(g, "edge_attr")
        desc = _get_attr(g, "descriptor")

        if x is None:
            missing_x += 1
        if edge_index is None:
            missing_edge_index += 1
        if y is None:
            missing_y += 1
        if pos is None:
            missing_pos += 1
        if edge_attr is None:
            missing_edge_attr += 1
        if desc is None:
            missing_descriptor += 1

        # dims
        try:
            if isinstance(x, torch.Tensor) and x.dim() == 2:
                x_dims.append(int(x.shape[1]))
        except Exception:
            pass

        try:
            if isinstance(edge_attr, torch.Tensor) and edge_attr.dim() == 2:
                edge_attr_dims.append(int(edge_attr.shape[1]))
        except Exception:
            pass

        try:
            if isinstance(desc, torch.Tensor):
                d = desc.detach().cpu()
                # common shapes: (1, D) or (D,)
                if d.dim() == 2:
                    desc_dims.append(int(d.shape[1]))
                    if bool((d == 0).all().item()):
                        desc_all_zero += 1
                elif d.dim() == 1:
                    desc_dims.append(int(d.shape[0]))
                    if bool((d == 0).all().item()):
                        desc_all_zero += 1
        except Exception:
            pass

        # node-pos match
        try:
            if isinstance(x, torch.Tensor) and isinstance(pos, torch.Tensor):
                if x.dim() == 2 and pos.dim() == 2 and x.shape[0] != pos.shape[0]:
                    node_pos_mismatch += 1
        except Exception:
            pass

        # edge_index in range
        try:
            if isinstance(x, torch.Tensor) and isinstance(edge_index, torch.Tensor) and x.dim() == 2 and edge_index.numel() > 0:
                num_nodes = int(x.shape[0])
                ei = edge_index.detach().cpu()
                if ei.dim() == 2 and ei.shape[0] == 2:
                    mn = int(ei.min().item())
                    mx = int(ei.max().item())
                    if mn < 0 or mx >= num_nodes:
                        edge_index_oob += 1
        except Exception:
            pass

        # finite checks
        fx = _tensor_isfinite(x)
        # edge_index is integer topology; still check in case it's corrupted
        fei = _tensor_isfinite(edge_index)
        fy = _tensor_isfinite(y)
        fp = _tensor_isfinite(pos)
        fea = _tensor_isfinite(edge_attr)
        fd = _tensor_isfinite(desc)
        finite_flags = [fx, fei, fy, fp, fea, fd]
        # count as non-finite if any present tensor is non-finite
        is_nonfinite = any(flag is False for flag in finite_flags if flag is not None)
        if is_nonfinite:
            any_nonfinite += 1
            if fx is False:
                nonfinite_x += 1
            if fp is False:
                nonfinite_pos += 1
            if fea is False:
                nonfinite_edge_attr += 1
            if fd is False:
                nonfinite_descriptor += 1
            if fy is False:
                nonfinite_y += 1

            if max_examples > 0 and len(examples) < max_examples:
                # Best-effort debugging payload; do NOT assume smiles exists in Data.
                ex: Dict[str, Any] = {"idx": idx}
                smiles = _get_attr(g, "smiles")
                if isinstance(smiles, str) and smiles:
                    ex["smiles"] = smiles
                # shapes
                try:
                    if isinstance(x, torch.Tensor):
                        ex["x_shape"] = list(x.shape)
                except Exception:
                    pass
                try:
                    if isinstance(pos, torch.Tensor):
                        ex["pos_shape"] = list(pos.shape)
                except Exception:
                    pass
                try:
                    if isinstance(edge_attr, torch.Tensor):
                        ex["edge_attr_shape"] = list(edge_attr.shape)
                except Exception:
                    pass
                try:
                    if isinstance(desc, torch.Tensor):
                        ex["descriptor_shape"] = list(desc.shape)
                except Exception:
                    pass
                # label (first scalar)
                try:
                    ys = _to_1d_float_list(y)
                    if ys:
                        ex["y0"] = ys[0]
                except Exception:
                    pass
                ex["nonfinite_fields"] = [
                    name
                    for name, flag in [
                        ("x", fx),
                        ("edge_index", fei),
                        ("y", fy),
                        ("pos", fp),
                        ("edge_attr", fea),
                        ("descriptor", fd),
                    ]
                    if flag is False
                ]
                examples.append(ex)

        # labels
        labels.extend(_to_1d_float_list(y))

    label_min = min(labels) if labels else None
    label_max = max(labels) if labels else None
    label_mean, label_std = _mean_std(labels)

    # approximate unique count (round to reduce float noise)
    label_unique_approx = len({round(v, 6) for v in labels}) if labels else 0
    label_pos_rate = None
    if _looks_binary(labels):
        label_pos_rate = sum(1.0 for v in labels if abs(v - 1.0) <= 1e-6) / len(labels) if labels else None

    def _minmax(xs: List[int]) -> Tuple[Optional[int], Optional[int]]:
        return (min(xs), max(xs)) if xs else (None, None)

    x_min, x_max = _minmax(x_dims)
    ea_min, ea_max = _minmax(edge_attr_dims)
    d_min, d_max = _minmax(desc_dims)

    return SplitQuality(
        dataset=dataset,
        seed=seed,
        split=split,
        n_graphs=n,
        raw_train_val_rows=raw_train_val_rows,
        raw_test_rows=raw_test_rows,
        missing_x=missing_x,
        missing_edge_index=missing_edge_index,
        missing_y=missing_y,
        missing_pos=missing_pos,
        missing_edge_attr=missing_edge_attr,
        missing_descriptor=missing_descriptor,
        node_pos_mismatch=node_pos_mismatch,
        edge_index_oob=edge_index_oob,
        any_nonfinite=any_nonfinite,
        nonfinite_x=nonfinite_x,
        nonfinite_pos=nonfinite_pos,
        nonfinite_edge_attr=nonfinite_edge_attr,
        nonfinite_descriptor=nonfinite_descriptor,
        nonfinite_y=nonfinite_y,
        x_dim_min=x_min,
        x_dim_max=x_max,
        edge_attr_dim_min=ea_min,
        edge_attr_dim_max=ea_max,
        descriptor_dim_min=d_min,
        descriptor_dim_max=d_max,
        descriptor_all_zero=desc_all_zero,
        label_count=len(labels),
        label_unique_approx=label_unique_approx,
        label_min=label_min,
        label_max=label_max,
        label_mean=label_mean,
        label_std=label_std,
        label_pos_rate=label_pos_rate,
        nonfinite_examples=examples if max_examples > 0 else None,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir", type=str, default=str(_repo_root() / "data" / "processed_tdc_data"))
    parser.add_argument("--raw_data_dir", type=str, default=str(_repo_root() / "data" / "data_tdc" / "admet_group"))
    parser.add_argument("--datasets", type=str, nargs="*", default=None, help="Optional dataset names to analyze (default: all found).")
    parser.add_argument("--seeds", type=str, nargs="*", default=None, help="Optional seeds to analyze (e.g., seed1 seed2 ...). Default: all found.")
    parser.add_argument("--out_dir", type=str, default=str(_repo_root() / "analysis"))
    parser.add_argument("--out_prefix", type=str, default="preprocess_quality_report")
    parser.add_argument("--nonfinite_examples", type=int, default=0, help="Save up to N nonfinite examples per split (stored in JSON only).")
    args = parser.parse_args()

    # --- Environment preflight (common failure: wrong Python / incompatible torch/pyg) ---
    try:
        import torch  # noqa: F401
    except Exception as e:
        py = os.environ.get("PYTHON", "") or os.popen("which python3 2>/dev/null").read().strip()
        raise SystemExit(
            "❌ Cannot import torch in this Python environment.\n"
            f"   Python: {py}\n"
            f"   Error: {e}\n\n"
            "Tip: run this script inside the same conda env you use for training (e.g. `aegnn_env`).\n"
            "Example:\n"
            "  conda run -n aegnn_env python DMP-EGNN/scripts/analyze_preprocessed_tdc_data.py --datasets hia_hou\n"
        )

    processed_dir = Path(args.processed_dir)
    raw_dir = Path(args.raw_data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not processed_dir.exists():
        raise SystemExit(f"processed_dir not found: {processed_dir}")

    # Pass example limit to analyze_split without changing its call signature (keeps CSV stable)
    if args.nonfinite_examples and args.nonfinite_examples > 0:
        os.environ["DMP_EGNN_QA_MAX_EXAMPLES"] = str(int(args.nonfinite_examples))
    else:
        os.environ.pop("DMP_EGNN_QA_MAX_EXAMPLES", None)

    # Discover datasets
    datasets = sorted([p.name for p in processed_dir.iterdir() if p.is_dir()])
    if args.datasets:
        wanted = set(args.datasets)
        datasets = [d for d in datasets if d in wanted]
    if not datasets:
        print("No datasets to analyze.")
        return 0

    rows: List[SplitQuality] = []
    problems: List[str] = []

    for ds in datasets:
        ds_dir = processed_dir / ds
        # Raw reference counts (optional)
        raw_train_val_rows = _count_csv_rows(raw_dir / ds / "train_val.csv")
        raw_test_rows = _count_csv_rows(raw_dir / ds / "test.csv")

        seeds = sorted([p.name for p in ds_dir.iterdir() if p.is_dir() and p.name.startswith("seed")])
        if args.seeds:
            wanted_seeds = set(args.seeds)
            seeds = [s for s in seeds if s in wanted_seeds]

        for seed in seeds:
            seed_dir = ds_dir / seed
            for split in ("train", "valid", "test"):
                pt_path = seed_dir / f"{split}.pt"
                if not pt_path.exists():
                    problems.append(f"Missing file: {pt_path}")
                    continue
                try:
                    rows.append(analyze_split(ds, seed, split, pt_path, raw_train_val_rows, raw_test_rows))
                except Exception as e:
                    problems.append(f"Failed to analyze {pt_path}: {e}")

    # Write JSON
    json_path = out_dir / f"{args.out_prefix}.json"
    payload: Dict[str, Any] = {
        "processed_dir": str(processed_dir),
        "raw_data_dir": str(raw_dir),
        "n_rows": len(rows),
        "problems": problems,
        "rows": [asdict(r) for r in rows],
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    # Write CSV
    csv_path = out_dir / f"{args.out_prefix}.csv"
    if rows:
        fieldnames = list(asdict(rows[0]).keys())
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow(asdict(r))

    # Console summary (high-signal)
    print(f"✅ Wrote JSON: {json_path}")
    print(f"✅ Wrote CSV : {csv_path}")
    if problems:
        print(f"⚠️  Problems: {len(problems)} (see JSON 'problems')")

    # Quick aggregated view per dataset/seed (train+valid drop vs raw train_val, test drop vs raw test)
    try:
        # Build simple aggregation without pandas
        key_to_counts: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for r in rows:
            k = (r.dataset, r.seed)
            key_to_counts.setdefault(
                k,
                {
                    "raw_train_val_rows": r.raw_train_val_rows,
                    "raw_test_rows": r.raw_test_rows,
                    "train": 0,
                    "valid": 0,
                    "test": 0,
                    "any_nonfinite": 0,
                    "edge_index_oob": 0,
                    "missing_pos": 0,
                    "missing_descriptor": 0,
                },
            )
            key_to_counts[k][r.split] = r.n_graphs
            key_to_counts[k]["any_nonfinite"] += r.any_nonfinite
            key_to_counts[k]["edge_index_oob"] += r.edge_index_oob
            key_to_counts[k]["missing_pos"] += r.missing_pos
            key_to_counts[k]["missing_descriptor"] += r.missing_descriptor

        print("\n=== Quick summary (per dataset/seed) ===")
        for (ds, seed), v in sorted(key_to_counts.items()):
            tv = v["train"] + v["valid"]
            raw_tv = v["raw_train_val_rows"]
            raw_t = v["raw_test_rows"]
            tv_drop = None if not raw_tv else (1.0 - (tv / raw_tv))
            t_drop = None if not raw_t else (1.0 - (v["test"] / raw_t))
            tv_drop_str = "N/A" if tv_drop is None else f"{tv_drop*100:.2f}%"
            t_drop_str = "N/A" if t_drop is None else f"{t_drop*100:.2f}%"
            print(
                f"- {ds}/{seed}: train={v['train']}, valid={v['valid']}, test={v['test']} | "
                f"drop(train+valid vs raw_train_val)={tv_drop_str}, drop(test vs raw_test)={t_drop_str} | "
                f"nonfinite={v['any_nonfinite']}, edge_oob={v['edge_index_oob']}, missing_pos={v['missing_pos']}, missing_desc={v['missing_descriptor']}"
            )
    except Exception:
        pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())



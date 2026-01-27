#!/usr/bin/env python3
"""
TDC Dataset Preprocessing Script (NEW)

Differences vs scripts/preprocess_tdc_data.py:
- Writes outputs to a new directory (default: data/processed_tdc_data_new)
- Filters out molecules whose descriptor contains NaN/Inf (fusion_model-style safety)

Output layout:
- Standard mode:
  data/processed_tdc_data_new/{dataset}/seed{seed}/{train,valid,test}.pt
- CV mode:
  data/processed_tdc_data_new/cv/{dataset}/fold{outer+1}/outer{outer}_inner{inner}_{train,valid,test}.pt
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

# Add project root to path (so we can import utils/*)
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from tdc.benchmark_group import admet_group

from utils.data_utils import MolecularGraphBuilder
from utils.prepare_tdc_dataset import process_tdc_smiles_to_graphs, set_seed


def list_tdc_datasets(data_path: str) -> List[str]:
    """List all available TDC datasets (by directory scan)."""
    datasets: List[str] = []
    admet_dir = os.path.join(data_path, "admet_group")
    if os.path.exists(admet_dir):
        for item in os.listdir(admet_dir):
            item_path = os.path.join(admet_dir, item)
            if os.path.isdir(item_path):
                if os.path.exists(os.path.join(item_path, "test.csv")) and os.path.exists(os.path.join(item_path, "train_val.csv")):
                    datasets.append(item)
    return sorted(datasets)


def _atomic_torch_save(obj, final_path: str) -> None:
    os.makedirs(os.path.dirname(final_path), exist_ok=True)
    tmp_path = f"{final_path}.tmp.{os.getpid()}"
    torch.save(obj, tmp_path)
    os.rename(tmp_path, final_path)


def _filter_nonfinite_descriptor(graphs) -> Tuple[list, int]:
    """Drop graphs whose descriptor contains NaN/Inf (descriptor-only rule, per user request)."""
    kept = []
    dropped = 0
    for g in graphs:
        desc = getattr(g, "descriptor", None)
        if desc is None:
            # descriptor is expected to exist; if missing, drop to keep training consistent
            dropped += 1
            continue
        if not isinstance(desc, torch.Tensor):
            try:
                desc = torch.tensor(desc)
            except Exception:
                dropped += 1
                continue
        if not torch.isfinite(desc).all().item():
            dropped += 1
            continue
        kept.append(g)
    return kept, dropped


def _load_or_process_graphs_custom(
    smiles_list: List[str],
    labels_list: List[float],
    cache_file: str,
    graph_builder: MolecularGraphBuilder,
    num_workers: int,
    drop_nonfinite_descriptor: bool = True,
) -> list:
    """
    Similar to utils.prepare_tdc_dataset.load_or_process_tdc_graphs, but uses an explicit cache_file path
    (so we can write into processed_tdc_data_new) and optionally drops nonfinite descriptor molecules.
    """
    # IMPORTANT: process_tdc_smiles_to_graphs writes checkpoints to "{cache_file}.tmp.{pid}" during processing.
    # Ensure parent directory exists before processing starts.
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)

    if os.path.exists(cache_file):
        try:
            graphs = torch.load(cache_file, map_location="cpu", weights_only=False)
            if isinstance(graphs, list) and len(graphs) > 0:
                return graphs
        except Exception:
            # treat as corrupted
            try:
                os.remove(cache_file)
            except Exception:
                pass

    graphs, _valid_labels = process_tdc_smiles_to_graphs(
        smiles_list,
        labels_list,
        graph_builder,
        cache_file=cache_file,
        checkpoint_interval=200,
        num_workers=num_workers,
    )

    if drop_nonfinite_descriptor:
        filtered, dropped = _filter_nonfinite_descriptor(graphs)
        if dropped > 0:
            print(f"   🧹 Dropped {dropped} graphs due to nonfinite descriptor values")
        graphs = filtered

    # Save filtered graphs (overwrite whatever process_tdc_smiles_to_graphs might have checkpointed)
    _atomic_torch_save(graphs, cache_file)
    return graphs


def _tdc_standard_split(data_name: str, data_path: str, seed: int):
    group = admet_group(path=data_path)
    benchmark = group.get(data_name)
    name = benchmark["name"]
    test_df = benchmark["test"]
    train_df, valid_df = group.get_train_valid_split(benchmark=name, split_type="default", seed=seed)
    return train_df, valid_df, test_df


def _tdc_cv_split(data_name: str, data_path: str, seed: int, outer_fold_idx: int, inner_fold_idx: int, outer_folds: int, inner_folds: int):
    # Follow fusion_model's CV logic: merge train/valid/test then do nested KFold
    group = admet_group(path=data_path)
    benchmark = group.get(data_name)
    name = benchmark["name"]
    test_df = benchmark["test"]
    train_df, valid_df = group.get_train_valid_split(benchmark=name, split_type="default", seed=seed)
    full_df = pd.concat([train_df, valid_df, test_df], ignore_index=True)

    outer_kf = KFold(n_splits=outer_folds, shuffle=True, random_state=seed)
    outer_splits = list(outer_kf.split(full_df))
    outer_trainval_idx, test_idx = outer_splits[outer_fold_idx]
    trainval_df = full_df.iloc[outer_trainval_idx].reset_index(drop=True)
    test_df2 = full_df.iloc[test_idx].reset_index(drop=True)

    inner_kf = KFold(n_splits=inner_folds, shuffle=True, random_state=seed)
    inner_splits = list(inner_kf.split(trainval_df))
    train_idx, valid_idx = inner_splits[inner_fold_idx]
    train_df2 = trainval_df.iloc[train_idx].reset_index(drop=True)
    valid_df2 = trainval_df.iloc[valid_idx].reset_index(drop=True)
    return train_df2, valid_df2, test_df2


def preprocess_tdc_dataset(
    data_name: str,
    data_path: str,
    seed: int,
    processed_dir: str,
    num_conformers: int,
    optimize_conformers: bool,
    add_hydrogens: bool,
    use_fingerprint: bool,
    fingerprint_bits: int,
    descriptor_dim: Optional[int],
    use_cv: bool,
    outer_fold_idx: Optional[int],
    inner_fold_idx: Optional[int],
    outer_folds: int,
    inner_folds: int,
    num_workers: int,
    nonfinite_policy: str,
) -> bool:
    """
    Preprocess a single dataset into processed_dir, dropping descriptor-nonfinite molecules if requested.
    nonfinite_policy: 'drop' or 'keep'
    """
    mode_str = "CV" if use_cv else "Standard"
    print(f"\n{'='*60}")
    print(f"Processing TDC dataset (NEW, {mode_str} mode): {data_name}")
    print(f"Output dir: {processed_dir}")
    print(f"{'='*60}")

    drop_nonfinite = (nonfinite_policy == "drop")

    # Build graph builder (same as existing preprocessing: full graphs + descriptor required)
    graph_builder = MolecularGraphBuilder(
        use_descriptor=True,
        use_fingerprint=use_fingerprint,
        descriptor_dim=descriptor_dim,
        fingerprint_bits=fingerprint_bits,
        num_conformers=num_conformers,
        optimize_conformers=optimize_conformers,
        add_hydrogens=add_hydrogens,
    )

    # Split data
    if use_cv:
        if outer_fold_idx is None:
            raise ValueError("--outer_fold_idx is required when --use_cv is set")
        # fusion_model formula if not provided: (outer + 1) % inner_folds
        if inner_fold_idx is None:
            inner_fold_idx = (outer_fold_idx + 1) % inner_folds
        # ensure deterministic split
        set_seed(seed)
        train_df, valid_df, test_df = _tdc_cv_split(
            data_name=data_name,
            data_path=data_path,
            seed=seed,
            outer_fold_idx=outer_fold_idx,
            inner_fold_idx=inner_fold_idx,
            outer_folds=outer_folds,
            inner_folds=inner_folds,
        )
        cv_root = os.path.join(processed_dir, "cv")
        cache_dir = os.path.join(cv_root, data_name, f"fold{outer_fold_idx + 1}")
        split_tag = f"outer{outer_fold_idx}_inner{inner_fold_idx}"
        train_cache = os.path.join(cache_dir, f"{split_tag}_train.pt")
        valid_cache = os.path.join(cache_dir, f"{split_tag}_valid.pt")
        test_cache = os.path.join(cache_dir, f"{split_tag}_test.pt")
    else:
        train_df, valid_df, test_df = _tdc_standard_split(data_name=data_name, data_path=data_path, seed=seed)
        cache_dir = os.path.join(processed_dir, data_name, f"seed{seed}")
        train_cache = os.path.join(cache_dir, "train.pt")
        valid_cache = os.path.join(cache_dir, "valid.pt")
        test_cache = os.path.join(cache_dir, "test.pt")

    # Extract SMILES/labels
    smiles_train = train_df["Drug"].tolist()
    smiles_valid = valid_df["Drug"].tolist()
    smiles_test = test_df["Drug"].tolist()
    labels_train = train_df["Y"].tolist()
    labels_valid = valid_df["Y"].tolist()
    labels_test = test_df["Y"].tolist()

    print(f"Split sizes (raw): train={len(smiles_train)} valid={len(smiles_valid)} test={len(smiles_test)}")
    if drop_nonfinite:
        print("Descriptor nonfinite policy: DROP (fusion_model-style)")
    else:
        print("Descriptor nonfinite policy: KEEP")

    # Process + cache + (optional) filter
    train_graphs = _load_or_process_graphs_custom(smiles_train, labels_train, train_cache, graph_builder, num_workers, drop_nonfinite)
    valid_graphs = _load_or_process_graphs_custom(smiles_valid, labels_valid, valid_cache, graph_builder, num_workers, drop_nonfinite)
    test_graphs = _load_or_process_graphs_custom(smiles_test, labels_test, test_cache, graph_builder, num_workers, drop_nonfinite)

    print(f"\n✅ Done: {data_name} ({mode_str})")
    print(f"   Saved graphs: train={len(train_graphs)} valid={len(valid_graphs)} test={len(test_graphs)}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Preprocess TDC datasets (NEW: drop descriptor-nonfinite molecules)")
    parser.add_argument("--data_name", type=str, help="TDC dataset name (e.g., 'caco2_wang', 'ames')")
    parser.add_argument("--data_path", type=str, default="data/data_tdc", help="Path to TDC data directory")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for data splitting")
    parser.add_argument(
        "--processed_dir",
        type=str,
        default="data/processed_tdc_data_new",
        help="Directory to save processed data (NEW default)",
    )
    parser.add_argument("--num_conformers", type=int, default=10, help="Number of conformers to generate")
    parser.add_argument("--optimize_conformers", action="store_true", default=True, help="Optimize conformers")
    parser.add_argument("--no_optimize_conformers", dest="optimize_conformers", action="store_false", help="Don't optimize conformers")
    parser.add_argument("--add_hydrogens", action="store_true", default=True, help="Add hydrogens to molecules")
    parser.add_argument("--no_hydrogens", dest="add_hydrogens", action="store_false", help="Don't add hydrogens")
    parser.add_argument("--use_fingerprint", action="store_true", default=False, help="Use molecular fingerprints")
    parser.add_argument("--fingerprint_bits", type=int, default=2048, help="Number of fingerprint bits")
    parser.add_argument("--descriptor_dim", type=int, default=None, help="Descriptor dimension (None for all available)")
    parser.add_argument("--use_cv", action="store_true", default=False, help="Use cross-validation mode")
    parser.add_argument("--outer_fold_idx", type=int, default=None, help="Outer fold index for CV mode (0-4)")
    parser.add_argument("--inner_fold_idx", type=int, default=None, help="Inner fold index for CV mode (0-3). If None, uses (outer+1) % inner_folds")
    parser.add_argument("--outer_folds", type=int, default=5, help="Number of outer folds for CV mode")
    parser.add_argument("--inner_folds", type=int, default=4, help="Number of inner folds for CV mode")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of parallel workers for processing (1=sequential, >1=parallel, 0=use all CPU cores)")
    parser.add_argument("--list_datasets", action="store_true", help="List all available TDC datasets")
    parser.add_argument(
        "--nonfinite_policy",
        type=str,
        default="drop",
        choices=["drop", "keep"],
        help="How to handle descriptor NaN/Inf during preprocessing (default: drop)",
    )

    args = parser.parse_args()

    if args.list_datasets:
        datasets = list_tdc_datasets(args.data_path)
        print(f"\nAvailable TDC datasets ({len(datasets)}):")
        for i, dataset in enumerate(datasets, 1):
            print(f"  {i}. {dataset}")
        return 0

    if not args.data_name:
        print("Error: --data_name is required (or use --list_datasets)")
        return 1

    try:
        ok = preprocess_tdc_dataset(
            data_name=args.data_name,
            data_path=args.data_path,
            seed=args.seed,
            processed_dir=args.processed_dir,
            num_conformers=args.num_conformers,
            optimize_conformers=args.optimize_conformers,
            add_hydrogens=args.add_hydrogens,
            use_fingerprint=args.use_fingerprint,
            fingerprint_bits=args.fingerprint_bits,
            descriptor_dim=args.descriptor_dim,
            use_cv=args.use_cv,
            outer_fold_idx=args.outer_fold_idx,
            inner_fold_idx=args.inner_fold_idx,
            outer_folds=args.outer_folds,
            inner_folds=args.inner_folds,
            num_workers=args.num_workers,
            nonfinite_policy=args.nonfinite_policy,
        )
        return 0 if ok else 1
    except Exception as e:
        print(f"❌ Error processing {args.data_name}: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())



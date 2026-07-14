import os
from typing import List, Tuple, Optional

import torch
from torch_geometric.data import Dataset, Data, Batch

from descriptastorus.descriptors import rdNormalizedDescriptors
from tdc.benchmark_group import admet_group
from sklearn.model_selection import KFold
import pandas as pd

from core.dmpegnn_data_utils import MolecularGraphBuilder


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DMPEGNN_PROCESSED_DIR = os.path.join(ROOT_DIR, "data", "processed_tdc_data_dmpegnn")


class DMPEGNNGraphDataset(Dataset):
    """Dataset for DMPEGNN-based models: x, edge_index, edge_attr, pos, descriptor, y.
    When molecule_indices is set (multi-conformer), each sample is a list of Data for one molecule.
    """

    def __init__(
        self,
        graphs: List[Data],
        labels: Optional[List[float]] = None,
        molecule_indices: Optional[List[int]] = None,
    ):
        self.graphs = graphs
        self.labels = labels
        if molecule_indices is None:
            molecule_indices = list(range(len(graphs)))
        self.molecule_indices = molecule_indices

        if not molecule_indices:
            raise ValueError(
                "DMPEGNNGraphDataset received an empty molecule_indices/graphs list — "
                "no molecules survived featurization for this split (SMILES parsing, "
                "3D conformer generation, or descriptor computation may all be the cause). "
                "Check the upstream smiles_list and the filters in _build_dmpegnn_graphs()."
            )
        num_molecules = max(molecule_indices) + 1
        self._mol_to_graph_indices: List[List[int]] = [[] for _ in range(num_molecules)]
        for j, mid in enumerate(molecule_indices):
            self._mol_to_graph_indices[mid].append(j)

    def __len__(self) -> int:
        return len(self._mol_to_graph_indices)

    def __getitem__(self, idx: int):
        """Always return a list of Data (length 1 for single conformer) so collate receives uniform type."""
        indices = self._mol_to_graph_indices[idx]
        return [self.graphs[j] for j in indices]


def collate_dmpegnn_multi(batch):
    """
    Collate for DMPEGNN dataset when multi-conformer: each sample is list of Data per molecule.
    Produces a Batch with batch.molecule_idx (graph index -> molecule index), and per-molecule
    smiles, descriptor, y. Uses a running molecule index so that num_molecules always matches
    len(descriptor_per_mol) even when some samples are empty (skipped).
    """
    flat: List[Data] = []
    molecule_idx_list: List[int] = []
    smiles_per_mol: List[str] = []
    descriptor_per_mol: List[torch.Tensor] = []
    y_per_mol: List[torch.Tensor] = []
    mol_idx = 0
    for sample in batch:
        if isinstance(sample, list):
            if not sample:
                continue
            for d in sample:
                flat.append(d)
                molecule_idx_list.append(mol_idx)
            smiles_per_mol.append(sample[0].smiles)
            descriptor_per_mol.append(sample[0].descriptor.squeeze(0) if sample[0].descriptor.dim() > 1 else sample[0].descriptor)
            y_per_mol.append(sample[0].y)
            mol_idx += 1
        else:
            flat.append(sample)
            molecule_idx_list.append(mol_idx)
            smiles_per_mol.append(sample.smiles)
            descriptor_per_mol.append(sample.descriptor.squeeze(0) if sample.descriptor.dim() > 1 else sample.descriptor)
            y_per_mol.append(sample.y)
            mol_idx += 1
    b = Batch.from_data_list(flat)
    b.molecule_idx = torch.tensor(molecule_idx_list, dtype=torch.long, device=b.x.device if b.x.is_cuda else None)
    b.smiles = smiles_per_mol
    b.descriptor = torch.stack(descriptor_per_mol, dim=0)
    b.y = torch.cat(y_per_mol, dim=0)
    return b


def _compute_descriptor(smiles: str, generator) -> Optional[torch.Tensor]:
    """Compute 200-d normalized RDKit descriptors, mirroring fusion_model.prepare_dataset."""
    try:
        values = generator.process(smiles)[1:]
        desc = torch.tensor(values, dtype=torch.float)
        if torch.isnan(desc).any() or torch.isinf(desc).any():
            return None
        if torch.all(desc == 0) or desc.std() == 0:
            return None
        return desc
    except Exception:
        return None


def _build_dmpegnn_graphs(
    smiles_list: List[str],
    labels_list: List[float],
    dataset_name: str,
    split_name: str,
    seed: int,
    num_conformers_to_keep: int = 3,
) -> Tuple[List[Data], List[float], List[int]]:
    """
    Returns:
        graphs: flat list of Data (one per conformer per molecule)
        labels_out: one label per molecule
        molecule_indices: graph index -> molecule index (length = len(graphs))
    """
    cache_dir = os.path.join(DMPEGNN_PROCESSED_DIR, dataset_name, f"seed{seed}")
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{split_name}.pt")
    cache_meta = os.path.join(cache_dir, f"{split_name}_multi_keep{num_conformers_to_keep}.pt")

    if num_conformers_to_keep > 1 and os.path.exists(cache_meta):
        print(f"[DMPEGNN] Loading cached graphs (multi-conformer): {cache_meta}")
        data = torch.load(cache_meta)
        return data["graphs"], data["labels"], data["molecule_indices"]

    if os.path.exists(cache_file):
        print(f"[DMPEGNN] Loading cached graphs: {cache_file}")
        data = torch.load(cache_file)
        graphs, labels_out = data["graphs"], data["labels"]
        molecule_indices = list(range(len(graphs)))
        return graphs, labels_out, molecule_indices

    print(f"[DMPEGNN] Building graphs and 3D coords for {dataset_name}/{split_name} (seed={seed}), num_conformers_to_keep={num_conformers_to_keep}")
    builder = MolecularGraphBuilder(
        use_fingerprint=False,
        use_descriptor=False,
        num_conformers=10,
        optimize_conformers=True,
        num_conformers_to_keep=num_conformers_to_keep,
        num_threads=0,
        add_hydrogens=True,
        prune_rms_thresh=0.5,
    )
    desc_gen = rdNormalizedDescriptors.RDKit2DNormalized()

    graphs: List[Data] = []
    labels_out: List[float] = []
    molecule_indices: List[int] = []
    survived_idx = 0  # dense 0-based counter, incremented only when a molecule survives all filters

    for smiles, label in zip(smiles_list, labels_list):
        if len(smiles) > 512:
            continue

        graph_or_list = builder.smiles_to_graph(smiles)
        if graph_or_list is None:
            continue

        desc = _compute_descriptor(smiles, desc_gen)
        if desc is None:
            continue

        if isinstance(graph_or_list, list):
            for g in graph_or_list:
                data = Data(
                    x=g.x,
                    edge_index=g.edge_index,
                    edge_attr=g.edge_attr,
                    pos=g.pos,
                    descriptor=desc,
                    y=torch.tensor([label], dtype=torch.float),
                    smiles=smiles,
                )
                graphs.append(data)
                molecule_indices.append(survived_idx)
        else:
            data = Data(
                x=graph_or_list.x,
                edge_index=graph_or_list.edge_index,
                edge_attr=graph_or_list.edge_attr,
                pos=graph_or_list.pos,
                descriptor=desc,
                y=torch.tensor([label], dtype=torch.float),
                smiles=smiles,
            )
            graphs.append(data)
            molecule_indices.append(survived_idx)
        labels_out.append(label)
        survived_idx += 1

    if num_conformers_to_keep > 1:
        torch.save({"graphs": graphs, "labels": labels_out, "molecule_indices": molecule_indices}, cache_meta)
        print(f"[DMPEGNN] Saved processed graphs (multi-conformer) to: {cache_meta}")
    else:
        torch.save({"graphs": graphs, "labels": labels_out}, cache_file)
        print(f"[DMPEGNN] Saved processed graphs to: {cache_file}")
    return graphs, labels_out, molecule_indices


def load_dmpegnn_dataset(data_name: str, data_path: str, seed: int, num_conformers_to_keep: int = 3):
    """Load TDC dataset, but featurize with DMPEGNN builder (x, edge_attr, pos) + 200-d descriptor.
    num_conformers_to_keep: number of conformers per molecule (1 = single conformer, 3+ = multi-conformer).
    """
    group = admet_group(path=data_path)
    benchmark = group.get(data_name)
    name = benchmark["name"]
    test_df = benchmark["test"]
    train_df, valid_df = group.get_train_valid_split(benchmark=name, split_type="default", seed=seed)

    smiles_train = train_df["Drug"].tolist()
    smiles_valid = valid_df["Drug"].tolist()
    smiles_test = test_df["Drug"].tolist()

    labels_train = train_df["Y"].tolist()
    labels_valid = valid_df["Y"].tolist()
    labels_test = test_df["Y"].tolist()

    train_graphs, train_labels, train_mol_idx = _build_dmpegnn_graphs(smiles_train, labels_train, data_name, "train", seed, num_conformers_to_keep)
    valid_graphs, valid_labels, valid_mol_idx = _build_dmpegnn_graphs(smiles_valid, labels_valid, data_name, "valid", seed, num_conformers_to_keep)
    test_graphs, test_labels, test_mol_idx = _build_dmpegnn_graphs(smiles_test, labels_test, data_name, "test", seed, num_conformers_to_keep)

    return (
        DMPEGNNGraphDataset(train_graphs, train_labels, train_mol_idx),
        DMPEGNNGraphDataset(valid_graphs, valid_labels, valid_mol_idx),
        DMPEGNNGraphDataset(test_graphs, test_labels, test_mol_idx),
    )


def load_dmpegnn_dataset_cv(
    data_name: str,
    data_path: str,
    seed: int,
    outer_fold_idx: int = 0,
    inner_fold_idx: int = 0,
    outer_folds: int = 5,
    inner_folds: int = 4,
    num_conformers_to_keep: int = 3,
):
    """
    DMPEGNN version of load_dataset_cv:
    5-fold outer CV (train/test split) + 4-fold inner CV (train/val split),
    but with DMPEGNN graphs (x, edge_attr, pos) + 200-d descriptor.
    """
    group = admet_group(path=data_path)
    benchmark = group.get(data_name)
    name = benchmark["name"]
    test_df = benchmark["test"]
    train_df, valid_df = group.get_train_valid_split(benchmark=name, split_type="default", seed=seed)
    full_df = pd.concat([train_df, valid_df, test_df], ignore_index=True)

    # outer CV: choose test fold
    outer_kf = KFold(n_splits=outer_folds, shuffle=True, random_state=seed)
    outer_splits = list(outer_kf.split(full_df))
    outer_trainval_idx, test_idx = outer_splits[outer_fold_idx]

    trainval_df = full_df.iloc[outer_trainval_idx].reset_index(drop=True)
    test_df = full_df.iloc[test_idx].reset_index(drop=True)

    # inner CV: split trainval into train/valid
    inner_kf = KFold(n_splits=inner_folds, shuffle=True, random_state=seed)
    inner_splits = list(inner_kf.split(trainval_df))
    train_idx, valid_idx = inner_splits[inner_fold_idx]

    train_df = trainval_df.iloc[train_idx].reset_index(drop=True)
    valid_df = trainval_df.iloc[valid_idx].reset_index(drop=True)

    smiles_train = train_df["Drug"].tolist()
    smiles_valid = valid_df["Drug"].tolist()
    smiles_test = test_df["Drug"].tolist()

    labels_train = train_df["Y"].tolist()
    labels_valid = valid_df["Y"].tolist()
    labels_test = test_df["Y"].tolist()

    split_tag = f"outer{outer_fold_idx}_inner{inner_fold_idx}"

    train_graphs, train_labels, train_mol_idx = _build_dmpegnn_graphs(smiles_train, labels_train, data_name, f"{split_tag}_train", seed, num_conformers_to_keep)
    valid_graphs, valid_labels, valid_mol_idx = _build_dmpegnn_graphs(smiles_valid, labels_valid, data_name, f"{split_tag}_valid", seed, num_conformers_to_keep)
    test_graphs, test_labels, test_mol_idx = _build_dmpegnn_graphs(smiles_test, labels_test, data_name, f"{split_tag}_test", seed, num_conformers_to_keep)

    return (
        DMPEGNNGraphDataset(train_graphs, train_labels, train_mol_idx),
        DMPEGNNGraphDataset(valid_graphs, valid_labels, valid_mol_idx),
        DMPEGNNGraphDataset(test_graphs, test_labels, test_mol_idx),
    )


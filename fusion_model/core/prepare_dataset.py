import torch
from torch_geometric.data import Dataset, Data
import os
from typing import List, Tuple

import deepchem as dc
# from descriptastorus.descriptors import rdDescriptors 
from descriptastorus.descriptors import rdNormalizedDescriptors
from rdkit import Chem

from tdc.benchmark_group import admet_group

from sklearn.model_selection import KFold
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(ROOT_DIR, "data", "processed_tdc_data")
PROCESSED_CV_DIR = os.path.join(ROOT_DIR, "data", "processed_tdc_data_cv")

class MoleculeDataset(Dataset):
    def __init__(self, smiles: List[str], 
                 node_features: List[torch.Tensor], 
                 edge_indices: List[torch.Tensor], 
                 descriptors: torch.Tensor, 
                 labels: torch.Tensor,
                 edge_attrs: List[torch.Tensor] = None):
        
        self.smiles = smiles  
        self.node_features = node_features  
        self.edge_indices = edge_indices  
        self.descriptors = descriptors 
        self.labels = labels 
        # OGB-style [num_bonds, 9] per sample; for DMPEGNN. If None, no edge_attr in Data.
        self.edge_attrs = edge_attrs if edge_attrs is not None else []

    def __len__(self) -> int:
        return len(self.node_features)

    def __getitem__(self, idx: int) -> Data:
        out = Data(
            smiles=self.smiles[idx], 
            x=self.node_features[idx], 
            edge_index=self.edge_indices[idx], 
            descriptor=self.descriptors[idx],
            y=self.labels[idx]
        )
        if self.edge_attrs:
            out.edge_attr = self.edge_attrs[idx]
        return out

# -----------------------------------------------------------#

# d=5
# SMILES -> (num_atoms, 5)
def smiles_to_atom_features(smiles: str) -> torch.Tensor:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: # check if SMILES is valid
        raise ValueError(f"Invalid SMILES: {smiles}")  
    
    atoms = mol.GetAtoms() # get atom list
    atom_data = [[      
        atom.GetAtomicNum(),    # atomic number
        atom.GetMass(),         # atomic mass
        atom.GetDegree(),       # number of connected atoms
        int(atom.IsInRing()),   # ring membership flag
        atom.GetHybridization() # hybridization type
    ] for atom in atoms]     
    return torch.tensor(atom_data, dtype=torch.float) # (num_atoms, 5)

# d=75
# SMILES -> (num_atoms, 75)
featurizer = dc.feat.ConvMolFeaturizer() # ConvMolFeaturizer
def smiles_to_atom_features_2(smiles: str) -> torch.Tensor:
    molecules = featurizer.featurize(smiles) # list of ConvMol objects
    mol = molecules[0] # ConvMol object
    atom_data = mol.get_atom_features() # (num_atoms, 75)
    return torch.tensor(atom_data, dtype=torch.float)

# -----------------------------------------------------------#

# SMILES -> (2, num_bonds)
def smiles_to_bond_indices(smiles: str) -> torch.Tensor:
    mol = Chem.MolFromSmiles(smiles) # read SMILES
    if mol is None: # check if SMILES is valid
        raise ValueError(f"Invalid SMILES: {smiles}")
    
    bonds = mol.GetBonds() # get bond list
    if len(bonds) == 0:
        return torch.empty((2, 0), dtype=torch.long)
    
    bond_data = [[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()] for bond in bonds] # (num_bonds, 2)
    return torch.tensor(bond_data, dtype=torch.long).T # (2, num_bonds)


# SMILES -> (num_bonds, 9), OGB-style edge features for DMPEGNN (edmpnn_model_new) compatibility
# 4 dims bond type (single/double/triple/aromatic) + 4 dims stereo + 1 dim conjugated
EDGE_FEATURE_DIM = 9

def smiles_to_bond_features(smiles: str) -> torch.Tensor:
    """OGB-style 9-dim edge features: bond type (4) + stereo (4) + conjugated (1)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    bonds = mol.GetBonds()
    if len(bonds) == 0:
        return torch.empty((0, EDGE_FEATURE_DIM), dtype=torch.float)
    features = []
    for bond in bonds:
        vec = [0.0] * EDGE_FEATURE_DIM
        # Bond type: 4 dims
        bt = bond.GetBondType()
        if bt == Chem.BondType.SINGLE:
            vec[0] = 1.0
        elif bt == Chem.BondType.DOUBLE:
            vec[1] = 1.0
        elif bt == Chem.BondType.TRIPLE:
            vec[2] = 1.0
        elif bt == Chem.BondType.AROMATIC:
            vec[3] = 1.0
        else:
            vec[0] = 1.0
        # Stereo: 4 dims (indices 4-7)
        st = bond.GetStereo()
        if st == Chem.BondStereo.STEREONONE:
            vec[4] = 1.0
        elif st == Chem.BondStereo.STEREOANY:
            vec[5] = 1.0
        elif st == Chem.BondStereo.STEREOZ:
            vec[6] = 1.0
        elif st == Chem.BondStereo.STEREOE:
            vec[7] = 1.0
        else:
            vec[4] = 1.0
        # Conjugated: 1 dim
        vec[8] = 1.0 if bond.GetIsConjugated() else 0.0
        features.append(vec)
    return torch.tensor(features, dtype=torch.float)  # (num_bonds, 9)

# -----------------------------------------------------------#

# SMILES -> (200,)
def compute_molecular_descriptors(smiles: str, descriptor_generator) -> torch.Tensor:
    try:
        # skip the first element (since the first element is the SMILES string itself)
        desc_values = descriptor_generator.process(smiles)[1:]
        desc_values = torch.tensor(desc_values, dtype=torch.float)
        
        # === check descriptor values ===
        # validate descriptor values
        if torch.isnan(desc_values).any() or torch.isinf(desc_values).any():
            raise ValueError("Descriptor contains NaN or Inf values")
        # check if all values are zero
        if torch.all(desc_values == 0):
            raise ValueError("All descriptor values are zero")
        # check if all values are the same
        if desc_values.std() == 0:
            raise ValueError("Descriptor has zero standard deviation")
        return desc_values # tensor(200,)

    except Exception as e:
        print(f"[SMILES processing failed]: {smiles} | Reason: {e}")
        return None

# -----------------------------------------------------------#

def process_dataset(smiles_list: List[str], labels_list: List[float]):
    
    smiles_data = []
    node_features = []
    edge_indices = []
    edge_attrs = []
    descriptors = []
    labels = []
    
    # === initialize descriptor generator ===
    # descriptor_generator = rdDescriptors.RDKit2D() 
    descriptor_generator = rdNormalizedDescriptors.RDKit2DNormalized()
    
    # --- unlabeled data use ---
    if labels_list is None:
        labels_list = [float('nan')] * len(smiles_list)
    
    error_smiles = []
    for smiles, label in zip(smiles_list, labels_list): # e.g., [("C=O", 0.85), ("CCO", 0.67), ("C1CC1", 0.92)]
        
        # Skip SMILES > 512 chars (MegaMolBART supports up to 512).
        if len(smiles) > 512:
            print(f"[SMILES too long, skipped]: {smiles[:50]}... (length {len(smiles)})")
            error_smiles.append(smiles)
            continue       
        
        # compute descriptors
        desc = compute_molecular_descriptors(smiles, descriptor_generator) # (200,)
        if desc is None:
            error_smiles.append(smiles)
            continue # skip invalid smiles (could not compute descriptors)
        try:
            atom_features = smiles_to_atom_features_2(smiles) # (num_atoms, 75)
            bond_indices = smiles_to_bond_indices(smiles) # (2, num_bonds)
            bond_features = smiles_to_bond_features(smiles) # (num_bonds, 9)
        except Exception as e:
            print(f"[SMILES structure error]: {smiles} | Reason: {e}")
            error_smiles.append(smiles)
            continue # skip invalid smiles

        smiles_data.append(smiles)      
        node_features.append(atom_features) # [tensor([[6., 12., 1., 0., 3.], [8., 16., 1., 0., 3.]]), ...]
        edge_indices.append(bond_indices) # [tensor([[0, 1], [1, 2]]), ...]
        edge_attrs.append(bond_features)   # (num_bonds, 9) for DMPEGNN
        descriptors.append(desc)
        labels.append(label) # [0.85, 0.67, 0.92]
        
    print(f"Skipped {len(error_smiles)} invalid SMILES structures")
    
    # Convert lists to tensors
    descriptor_tensor = torch.stack(descriptors) # tensor([(200,), (200,), (200,)]) # stack: (N, 200)
    label_tensor = torch.tensor(labels, dtype=torch.float) # tensor([0.85, 0.67, 0.92])
        
    return smiles_data, node_features, edge_indices, edge_attrs, descriptor_tensor, label_tensor

# -----------------------------------------------------------#

def load_or_process_dataset(smiles_list: List[str], 
                            labels_list: List[float], 
                            dataset_name: str, 
                            split_name: str, 
                            seed: int) -> Tuple:
    
    cache_dir = os.path.join(PROCESSED_DIR, dataset_name, f"seed{seed}") # e.g., data/processed_tdc_data/caco2_wang/seed1
    os.makedirs(cache_dir, exist_ok=True)

    cache_file = os.path.join(cache_dir, f"{split_name}.pt")

    if os.path.exists(cache_file):
        print(f"Loading cached file: {cache_file}")
        return torch.load(cache_file)
    else:
        print(f"Cache not found. Processing and saving to: {cache_file}")
        data = process_dataset(smiles_list, labels_list)
        torch.save(data, cache_file) # stored as a CPU tensor (can be loaded across devices)
        print(f"Saved processed data to: {cache_file}")
        return data
 
    
def load_or_process_dataset_cv(smiles_list: List[str], 
                               labels_list: List[float], 
                               dataset_name: str, 
                               split_name: str, 
                               outer_fold_idx: int) -> Tuple:
    
    cache_dir = os.path.join(PROCESSED_CV_DIR, dataset_name, f"fold{outer_fold_idx + 1}") # e.g., data/processed_tdc_data/caco2_wang/fold1
    os.makedirs(cache_dir, exist_ok=True)

    cache_file = os.path.join(cache_dir, f"{split_name}.pt")

    if os.path.exists(cache_file):
        print(f"Loading cached file: {cache_file}")
        return torch.load(cache_file)
    else:
        print(f"Cache not found. Processing and saving to: {cache_file}")
        data = process_dataset(smiles_list, labels_list)
        torch.save(data, cache_file) # stored as a CPU tensor (can be loaded across devices)
        print(f"Saved processed data to: {cache_file}")
        return data


# ---------------------------------------------------------------------

def load_dataset(data_name: str, data_path: str, seed: int):
    # load TDC dataset
    group = admet_group(path=data_path)
    benchmark = group.get(data_name)
    name = benchmark['name']
    test_df = benchmark['test']
    train_df, valid_df = group.get_train_valid_split(benchmark=name, split_type='default', seed=seed)

    # SMILES (list)
    smiles_train = train_df['Drug'].tolist()
    smiles_valid = valid_df['Drug'].tolist()
    smiles_test = test_df['Drug'].tolist()

    # label (list)
    labels_train = train_df['Y'].tolist()
    labels_valid = valid_df['Y'].tolist()
    labels_test = test_df['Y'].tolist()

    # process datasets
    train_data = load_or_process_dataset(smiles_train, labels_train, data_name, "train", seed)
    valid_data = load_or_process_dataset(smiles_valid, labels_valid, data_name, "valid", seed)
    test_data = load_or_process_dataset(smiles_test, labels_test, data_name, "test", seed)
    
    # unpack data (process_dataset returns 6 tuples; old cache has 5)
    if len(train_data) == 6:
        train_smiles, train_x, train_edge_index, train_edge_attrs, train_desc_data, train_y = train_data
    else:
        train_smiles, train_x, train_edge_index, train_desc_data, train_y = train_data
        train_edge_attrs = []
    if len(valid_data) == 6:
        valid_smiles, valid_x, valid_edge_index, valid_edge_attrs, valid_desc_data, valid_y = valid_data
    else:
        valid_smiles, valid_x, valid_edge_index, valid_desc_data, valid_y = valid_data
        valid_edge_attrs = []
    if len(test_data) == 6:
        test_smiles, test_x, test_edge_index, test_edge_attrs, test_desc_data, test_y = test_data
    else:
        test_smiles, test_x, test_edge_index, test_desc_data, test_y = test_data
        test_edge_attrs = []

    # create datasets
    train_dataset = MoleculeDataset(train_smiles, train_x, train_edge_index, train_desc_data, train_y, train_edge_attrs)
    valid_dataset = MoleculeDataset(valid_smiles, valid_x, valid_edge_index, valid_desc_data, valid_y, valid_edge_attrs)
    test_dataset = MoleculeDataset(test_smiles, test_x, test_edge_index, test_desc_data, test_y, test_edge_attrs)

    return train_dataset, valid_dataset, test_dataset


# 5-Fold Outer CV（train/test split） + 4-Fold Inner CV（train/val split）
def load_dataset_cv(data_name: str, 
                    data_path: str, 
                    seed: int, 
                    outer_fold_idx: int = 0, 
                    inner_fold_idx: int = 0, 
                    outer_folds: int = 5, 
                    inner_folds: int = 4):
    
    # load all data: combine train and test sets
    group = admet_group(path=data_path)
    benchmark = group.get(data_name)
    name = benchmark['name']
    test_df = benchmark['test']
    train_df, valid_df = group.get_train_valid_split(benchmark=name, split_type='default', seed=seed)
    full_df = pd.concat([train_df, valid_df, test_df], ignore_index=True)

    # outer CV: Split for test fold
    outer_kf = KFold(n_splits=outer_folds, shuffle=True, random_state=seed)
    outer_splits = list(outer_kf.split(full_df))
    outer_trainval_idx, test_idx = outer_splits[outer_fold_idx]

    trainval_df = full_df.iloc[outer_trainval_idx].reset_index(drop=True)
    test_df = full_df.iloc[test_idx].reset_index(drop=True)

    # inner CV: Split trainval into train and validation sets
    inner_kf = KFold(n_splits=inner_folds, shuffle=True, random_state=seed)
    inner_splits = list(inner_kf.split(trainval_df))
    train_idx, valid_idx = inner_splits[inner_fold_idx]

    train_df = trainval_df.iloc[train_idx].reset_index(drop=True)
    valid_df = trainval_df.iloc[valid_idx].reset_index(drop=True)

    # extract SMILES and labels
    smiles_train = train_df['Drug'].tolist()
    smiles_valid = valid_df['Drug'].tolist()
    smiles_test = test_df['Drug'].tolist()
    
    labels_train = train_df['Y'].tolist()
    labels_valid = valid_df['Y'].tolist()
    labels_test = test_df['Y'].tolist()

    # add fold identifiers to cache filenames
    split_tag = f"outer{outer_fold_idx}_inner{inner_fold_idx}"
    
    # process datasets
    train_data = load_or_process_dataset_cv(smiles_train, labels_train, data_name, f"{split_tag}_train", outer_fold_idx)
    valid_data = load_or_process_dataset_cv(smiles_valid, labels_valid, data_name, f"{split_tag}_valid", outer_fold_idx)
    test_data = load_or_process_dataset_cv(smiles_test, labels_test, data_name, f"{split_tag}_test", outer_fold_idx)
    
    # unpack data (process_dataset returns 6 tuples; old cache has 5)
    if len(train_data) == 6:
        train_smiles, train_x, train_edge_index, train_edge_attrs, train_desc_data, train_y = train_data
    else:
        train_smiles, train_x, train_edge_index, train_desc_data, train_y = train_data
        train_edge_attrs = []
    if len(valid_data) == 6:
        valid_smiles, valid_x, valid_edge_index, valid_edge_attrs, valid_desc_data, valid_y = valid_data
    else:
        valid_smiles, valid_x, valid_edge_index, valid_desc_data, valid_y = valid_data
        valid_edge_attrs = []
    if len(test_data) == 6:
        test_smiles, test_x, test_edge_index, test_edge_attrs, test_desc_data, test_y = test_data
    else:
        test_smiles, test_x, test_edge_index, test_desc_data, test_y = test_data
        test_edge_attrs = []

    # create datasets
    train_dataset = MoleculeDataset(train_smiles, train_x, train_edge_index, train_desc_data, train_y, train_edge_attrs)
    valid_dataset = MoleculeDataset(valid_smiles, valid_x, valid_edge_index, valid_desc_data, valid_y, valid_edge_attrs)
    test_dataset = MoleculeDataset(test_smiles, test_x, test_edge_index, test_desc_data, test_y, test_edge_attrs)

    return train_dataset, valid_dataset, test_dataset



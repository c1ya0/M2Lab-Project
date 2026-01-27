"""
TDC Dataset Preparation for AEGNN-M
Integrates fusion_model's TDC data loading and splitting logic
Uses AEGNN-M's MolecularGraphBuilder to process node features (78 dimensions, OGB-style), preserves edge_attr, pos, and descriptor required by AEGNN-M
"""

import torch
from torch_geometric.data import Data
import os
from typing import List, Tuple, Optional
import pandas as pd
from sklearn.model_selection import KFold
import numpy as np
import random
from rdkit import Chem
from multiprocessing import Pool, cpu_count
from functools import partial

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    # Fallback: create a simple progress indicator
    def tqdm(iterable, desc="", total=None, **kwargs):
        return iterable

import deepchem as dc
from tdc.benchmark_group import admet_group
from utils.data_utils import MolecularGraphBuilder

# =================== seed ===================
# Seed setting function that is completely consistent with fusion_model
def set_seed(seed):
    """
    Set random seed, completely consistent with fusion_model/core/utils.py
    
    Args:
        seed: Random seed value
    """
    torch.manual_seed(seed)  # CPU randomness
    np.random.seed(seed)  # NumPy randomness
    random.seed(seed)  # Python randomness, e.g., random.shuffle
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # CUDA randomness

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(ROOT_DIR, "data", "processed_tdc_data")
PROCESSED_CV_DIR = os.path.join(ROOT_DIR, "data", "processed_tdc_data_cv")


# DeepChem featurizer for node features (75 dimensions)
# NOTE: This is kept for backward compatibility but is no longer used.
# Current implementation uses AEGNN-M's MolecularGraphBuilder._get_node_features() (78 dimensions, OGB-style)
deepchem_featurizer = dc.feat.ConvMolFeaturizer()


def smiles_to_deepchem_node_features(smiles: str, mol_with_hs: Optional[Chem.Mol] = None) -> Optional[torch.Tensor]:
    """
    [DEPRECATED] Generate node features (75 dimensions) using DeepChem ConvMolFeaturizer
    This function is kept for backward compatibility but is no longer used in the main processing pipeline.
    Current implementation uses AEGNN-M's MolecularGraphBuilder._get_node_features() (78 dimensions, OGB-style).
    
    References fusion_model's processing approach
    
    Args:
        smiles: SMILES string
        mol_with_hs: Optional RDKit molecule object with hydrogens added (if provided, will use this to generate SMILES with H)
    
    Returns:
        Node features tensor [num_atoms, 75] or None (if failed)
    """
    try:
        # If molecule with hydrogens is provided, generate SMILES from it to ensure atom count matches
        if mol_with_hs is not None:
            # Generate SMILES from molecule with hydrogens (this will include H atoms)
            smiles_with_hs = Chem.MolToSmiles(mol_with_hs, allHsExplicit=True)
            molecules = deepchem_featurizer.featurize(smiles_with_hs)
        else:
            # Use original SMILES (without explicit hydrogens)
            molecules = deepchem_featurizer.featurize(smiles)
        
        if len(molecules) == 0:
            return None
        mol = molecules[0]  # ConvMol object
        atom_data = mol.get_atom_features()  # (num_atoms, 75)
        return torch.tensor(atom_data, dtype=torch.float)
    except Exception as e:
        return None


def smiles_to_bond_indices(smiles: str) -> Optional[torch.Tensor]:
    """
    Generate edge indices (references fusion_model's processing approach)
    
    Args:
        smiles: SMILES string
    
    Returns:
        Edge index tensor [2, num_bonds] or None (if failed)
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        bonds = mol.GetBonds()
        if len(bonds) == 0:
            return torch.empty((2, 0), dtype=torch.long)
        
        bond_data = [[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()] for bond in bonds]
        return torch.tensor(bond_data, dtype=torch.long).T  # (2, num_bonds)
    except Exception as e:
        return None


def _process_single_tdc_molecule(args):
    """
    Process a single TDC molecule (multiprocessing helper function)
    
    Args:
        args: Tuple of (smiles, label, builder_config)
            - smiles: SMILES string
            - label: Label value
            - builder_config: MolecularGraphBuilder configuration dictionary
    
    Returns:
        Tuple of (graph, label, smiles) or (None, label, smiles)
    """
    smiles, label, builder_config = args
    
    try:
        # Create new builder instance
        builder = MolecularGraphBuilder(**builder_config)
        
        # 1. Create molecule object
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, label, smiles
        
        # 2. Decide whether to add hydrogen atoms
        mol_for_processing = mol
        if builder.add_hydrogens:
            mol_for_processing = Chem.AddHs(mol)
        
        # 3. Generate node features (78 dimensions, OGB-style) using AEGNN-M's builder
        # All features (node, edge, 3D coordinates, descriptor) use the same mol_for_processing object
        # This ensures consistent atom counts across all features
        node_features = builder._get_node_features(mol_for_processing)
        if node_features is None or len(node_features) == 0:
            return None, label, smiles
        
        # Convert to torch tensor
        node_features = torch.tensor(node_features, dtype=torch.float)
        
        # 4. Use AEGNN-M's graph builder to generate edge_attr and pos
        edge_index_full, edge_features = builder._get_edge_features(mol_for_processing)
        
        # Generate 3D coordinates
        pos = builder._generate_3d_coordinates(mol_for_processing, smiles=smiles)
        if pos is None:
            return None, label, smiles
        
        # 5. Ensure node features and 3D coordinates have consistent atom counts
        # This should always pass since both use the same mol_for_processing object
        if node_features.shape[0] != pos.shape[0]:
            return None, label, smiles
        
        # 6. Create edge indices and edge features
        edge_index_bidirectional = torch.tensor(edge_index_full, dtype=torch.long)
        edge_features_tensor = torch.tensor(edge_features, dtype=torch.float)
        
        # 7. Create Data object
        data_dict = {
            'x': node_features,
            'edge_index': edge_index_bidirectional,
            'edge_attr': edge_features_tensor,
            'pos': torch.tensor(pos, dtype=torch.float),
            'y': torch.tensor([label], dtype=torch.float)
        }
        
        # 8. Add descriptor (required)
        if builder.use_descriptor:
            descriptor = builder._generate_descriptor(mol_for_processing)
            if descriptor is not None:
                data_dict['descriptor'] = torch.tensor(descriptor, dtype=torch.float).unsqueeze(0)
            else:
                desc_dim = builder.descriptor_dim
                data_dict['descriptor'] = torch.zeros((1, desc_dim), dtype=torch.float)
        
        # 9. Optional: Add fingerprint
        if builder.use_fingerprint:
            fingerprint = builder._generate_fingerprint(mol_for_processing)
            if fingerprint is not None:
                data_dict['fingerprint'] = torch.tensor(fingerprint, dtype=torch.float).unsqueeze(0)
        
        graph = Data(**data_dict)
        return graph, label, smiles
        
    except Exception as e:
        return None, label, smiles


def process_tdc_smiles_to_graphs(smiles_list: List[str], 
                                  labels_list: List[float],
                                  graph_builder: MolecularGraphBuilder,
                                  cache_file: Optional[str] = None,
                                  checkpoint_interval: int = 200,
                                  num_workers: int = 1) -> Tuple[List[Data], List[float]]:
    """
    Process TDC SMILES list, using AEGNN-M's MolecularGraphBuilder to process node features (78 dimensions, OGB-style), edge_attr, pos, and descriptor
    
    Supports incremental saving and multiprocessing
    
    Args:
        smiles_list: List of SMILES strings
        labels_list: List of labels
        graph_builder: AEGNN-M's MolecularGraphBuilder instance (used to generate edge_attr, pos, and descriptor)
        cache_file: Cache file path (if provided, incremental saving will be performed)
        checkpoint_interval: Save checkpoint every N molecules (default 200)
        num_workers: Number of parallel worker processes (1 = sequential, >1 = multiprocessing, 0 = use all CPU cores)
    
    Returns:
        graphs: List of graph data (PyTorch Geometric Data objects)
        valid_labels: List of successfully processed labels
    """
    graphs = []
    valid_labels = []
    error_smiles = []
    
    # If cache file is provided, try to resume from partially saved file
    # Use temporary filename with PID to avoid multiprocessing conflicts
    start_idx = 0
    if cache_file is not None:
        # Prioritize checking current process's temporary file
        temp_cache_file = f"{cache_file}.tmp.{os.getpid()}"
        # Also check old format temporary file (backward compatibility)
        old_temp_cache_file = cache_file + ".tmp"
        
        # Prioritize current process's temporary file, otherwise use old format
        if os.path.exists(temp_cache_file):
            temp_file_to_load = temp_cache_file
        elif os.path.exists(old_temp_cache_file):
            temp_file_to_load = old_temp_cache_file
        else:
            temp_file_to_load = None
        
        if temp_file_to_load:
            try:
                saved_data = torch.load(temp_file_to_load, map_location='cpu')
                graphs = saved_data.get('graphs', [])
                valid_labels = saved_data.get('valid_labels', [])
                start_idx = saved_data.get('processed_count', 0)
                print(f"📂 Resuming from checkpoint: {len(graphs)} graphs already processed (starting from index {start_idx})")
            except Exception as e:
                print(f"⚠️  Failed to load checkpoint, starting from scratch: {e}")
                graphs = []
                valid_labels = []
                start_idx = 0
    
    # Prepare remaining molecules to process
    remaining_smiles = smiles_list[start_idx:]
    remaining_labels = labels_list[start_idx:]
    total_molecules = len(smiles_list)
    remaining_count = len(remaining_smiles)
    
    if remaining_count == 0:
        return graphs, valid_labels
    
    # Prepare builder configuration (for multiprocessing)
    builder_config = {
        'use_atomic_number': graph_builder.use_atomic_number,
        'use_hybridization': graph_builder.use_hybridization,
        'use_formal_charge': graph_builder.use_formal_charge,
        'use_aromatic': graph_builder.use_aromatic,
        'use_chirality': graph_builder.use_chirality,
        'use_hydrogen_bonds': graph_builder.use_hydrogen_bonds,
        'use_bond_type': graph_builder.use_bond_type,
        'use_bond_stereo': graph_builder.use_bond_stereo,
        'num_conformers': graph_builder.num_conformers,
        'optimize_conformers': graph_builder.optimize_conformers,
        'num_threads': graph_builder.num_threads,
        'add_hydrogens': graph_builder.add_hydrogens,
        'prune_rms_thresh': graph_builder.prune_rms_thresh,
        'use_fingerprint': graph_builder.use_fingerprint,
        'fingerprint_radius': graph_builder.fingerprint_radius,
        'fingerprint_bits': graph_builder.fingerprint_bits,
        'use_descriptor': graph_builder.use_descriptor,
        'descriptor_dim': graph_builder.descriptor_dim
    }
    
    # Determine number of worker processes
    if num_workers == 0:
        num_workers = cpu_count()
    elif num_workers < 0:
        num_workers = max(1, cpu_count() + num_workers)
    
    # Process molecules (sequential or parallel)
    if num_workers == 1 or remaining_count == 1:
        # Sequential processing
        print(f"   Using sequential processing...")
        # Use tqdm progress bar
        if HAS_TQDM:
            iterator = tqdm(
                enumerate(zip(remaining_smiles, remaining_labels), start=start_idx),
                total=remaining_count,
                desc="Processing molecules",
                unit="mol",
                ncols=120,
                mininterval=0.5,
                maxinterval=2.0
            )
        else:
            iterator = enumerate(zip(remaining_smiles, remaining_labels), start=start_idx)
        
        for idx, (smiles, label) in iterator:
            graph, processed_label, processed_smiles = _process_single_tdc_molecule((smiles, label, builder_config))
            
            if graph is not None:
                graphs.append(graph)
                valid_labels.append(processed_label)
            else:
                error_smiles.append(processed_smiles)
            
            # Update progress bar information (tqdm automatically updates progress, here we update additional info)
            if HAS_TQDM and hasattr(iterator, 'set_postfix'):
                iterator.set_postfix({
                    'valid': len(graphs),
                    'failed': len(error_smiles),
                    'rate': f'{len(graphs)/(idx+1-start_idx)*100:.1f}%' if (idx+1-start_idx) > 0 else '0%'
                })
                # Ensure immediate display refresh
                iterator.refresh()
            
            # Incremental save (using atomic write: write to temporary file first)
            if cache_file is not None and (idx + 1) % checkpoint_interval == 0:
                # Use temporary filename with PID to avoid multiprocessing conflicts
                temp_cache_file = f"{cache_file}.tmp.{os.getpid()}"
                checkpoint_data = {
                    'graphs': graphs,
                    'valid_labels': valid_labels,
                    'processed_count': idx + 1,
                    'total_molecules': total_molecules
                }
                # Atomic write: directly write to temporary file (checkpoint uses temporary file)
                torch.save(checkpoint_data, temp_cache_file)
                if HAS_TQDM:
                    iterator.write(f"💾 Checkpoint saved: {len(graphs)}/{total_molecules} molecules processed ({100.0 * len(graphs) / total_molecules:.1f}%)")
                else:
                    print(f"💾 Checkpoint saved: {len(graphs)}/{total_molecules} molecules processed ({100.0 * len(graphs) / total_molecules:.1f}%)")
    else:
        # Parallel processing
        print(f"   Using parallel processing with {num_workers} workers...")
        process_args = [(smiles, label, builder_config) for smiles, label in zip(remaining_smiles, remaining_labels)]
        
        # Use tqdm progress bar (shows overall progress during parallel processing)
        if HAS_TQDM:
            pbar = tqdm(
                total=remaining_count,
                desc="Processing molecules",
                unit="mol",
                ncols=120,
                mininterval=0.5,
                maxinterval=2.0,
                initial=0
            )
        
        with Pool(processes=num_workers) as pool:
            # Use imap to support progress bar updates and incremental checkpoint saving
            # Process results in real-time and save checkpoint periodically during processing
            if HAS_TQDM:
                valid_count = 0
                processed_count = 0
                for result in pool.imap(_process_single_tdc_molecule, process_args):
                    # Process result immediately
                    graph, processed_label, processed_smiles = result
                    idx = start_idx + processed_count
                    
                    if graph is not None:
                        graphs.append(graph)
                        valid_labels.append(processed_label)
                        valid_count += 1
                    else:
                        error_smiles.append(processed_smiles)
                    
                    processed_count += 1
                    
                    # Update progress bar
                    pbar.update(1)
                    # Update progress bar information
                    pbar.set_postfix({
                        'valid': valid_count,
                        'failed': processed_count - valid_count,
                        'rate': f'{valid_count/processed_count*100:.1f}%' if processed_count > 0 else '0%'
                    })
                    # Force display refresh (ensure dynamic updates)
                    pbar.refresh()
                    
                    # Incremental save during processing (not after all processing is done)
                    # This ensures checkpoint is saved even if processing is interrupted
                    if cache_file is not None and processed_count % checkpoint_interval == 0:
                        # Use temporary filename with PID
                        temp_cache_file = f"{cache_file}.tmp.{os.getpid()}"
                        checkpoint_data = {
                            'graphs': graphs,
                            'valid_labels': valid_labels,
                            'processed_count': idx + 1,
                            'total_molecules': total_molecules
                        }
                        # Atomic write: write to temporary file first
                        torch.save(checkpoint_data, temp_cache_file)
                        # If old format temporary file exists, clean it up
                        old_temp_file = cache_file + ".tmp"
                        if os.path.exists(old_temp_file) and old_temp_file != temp_cache_file:
                            try:
                                os.remove(old_temp_file)
                            except:
                                pass
                        pbar.write(f"💾 Checkpoint saved: {len(graphs)}/{total_molecules} molecules processed ({100.0 * len(graphs) / total_molecules:.1f}%)")
                pbar.close()
            else:
                # For non-tqdm mode, still process incrementally but without progress bar
                processed_count = 0
                for result in pool.imap(_process_single_tdc_molecule, process_args):
                    graph, processed_label, processed_smiles = result
                    idx = start_idx + processed_count
                    
                    if graph is not None:
                        graphs.append(graph)
                        valid_labels.append(processed_label)
                    else:
                        error_smiles.append(processed_smiles)
                    
                    processed_count += 1
                    
                    # Incremental save during processing
                    if cache_file is not None and processed_count % checkpoint_interval == 0:
                        temp_cache_file = f"{cache_file}.tmp.{os.getpid()}"
                        checkpoint_data = {
                            'graphs': graphs,
                            'valid_labels': valid_labels,
                            'processed_count': idx + 1,
                            'total_molecules': total_molecules
                        }
                        torch.save(checkpoint_data, temp_cache_file)
                        old_temp_file = cache_file + ".tmp"
                        if os.path.exists(old_temp_file) and old_temp_file != temp_cache_file:
                            try:
                                os.remove(old_temp_file)
                            except:
                                pass
                        print(f"💾 Checkpoint saved: {len(graphs)}/{total_molecules} molecules processed ({100.0 * len(graphs) / total_molecules:.1f}%)")
    
    # After processing is complete, clean up temporary files (including all PID temporary files)
    if cache_file is not None:
        # Clean up current process's temporary file
        temp_cache_file = f"{cache_file}.tmp.{os.getpid()}"
        if os.path.exists(temp_cache_file):
            try:
                os.remove(temp_cache_file)
            except:
                pass
        # Clean up old format temporary file (backward compatibility)
        old_temp_file = cache_file + ".tmp"
        if os.path.exists(old_temp_file):
            try:
                os.remove(old_temp_file)
            except:
                pass
    
    if error_smiles:
        print(f"⚠️  Skipped {len(error_smiles)} molecules (failed to generate 3D coordinates or invalid SMILES)")
    
    return graphs, valid_labels


def load_or_process_tdc_graphs(smiles_list: List[str],
                               labels_list: List[float],
                               dataset_name: str,
                               split_name: str,
                               seed: int,
                               graph_builder: MolecularGraphBuilder,
                               use_cv: bool = False,
                               outer_fold_idx: Optional[int] = None,
                               num_workers: int = 1) -> List[Data]:
    """
    Load or process TDC data, generate AEGNN-M format graph data (using AEGNN-M's MolecularGraphBuilder to process node features)
    
    Args:
        smiles_list: List of SMILES strings
        labels_list: List of labels
        dataset_name: Dataset name
        split_name: Split name (train/valid/test)
        seed: Random seed
        graph_builder: AEGNN-M's MolecularGraphBuilder instance
        use_cv: Whether to use cross-validation mode
        outer_fold_idx: Outer fold index (used in CV mode)
    
    Returns:
        graphs: List of graph data
    """
    if use_cv and outer_fold_idx is not None:
        cache_dir = os.path.join(PROCESSED_CV_DIR, dataset_name, f"fold{outer_fold_idx + 1}")
    else:
        cache_dir = os.path.join(PROCESSED_DIR, dataset_name, f"seed{seed}")
    
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{split_name}.pt")
    
    # Check if there is a complete saved file (verify using atomic write)
    if os.path.exists(cache_file):
        try:
            print(f"✅ Loading cached graphs: {cache_file}")
            graphs = torch.load(cache_file, map_location='cpu')
            
            # Verify graph count: check if it matches expected count
            expected_count = len(smiles_list)  # Expected count (some may fail)
            actual_count = len(graphs)
            
            # Allow some tolerance (because some molecules may fail to process)
            # If actual count is significantly less than expected (less than 50%), may be incomplete data
            if actual_count < expected_count * 0.5:
                print(f"⚠️  Warning: Loaded {actual_count} graphs, expected ~{expected_count}")
                print(f"   This might indicate incomplete data. Will reprocess...")
                # Mark as incomplete, reprocess
                os.remove(cache_file)
                graphs = None
            else:
                print(f"   Loaded {actual_count} graphs (expected ~{expected_count}, {actual_count/expected_count*100:.1f}%)")
                return graphs
        except Exception as e:
            print(f"⚠️  Failed to load cached file: {e}")
            print(f"   Will reprocess...")
            # If loading fails, delete corrupted file
            try:
                os.remove(cache_file)
            except:
                pass
    
    # Check if there are temporary files (partially saved checkpoints)
    # Prioritize checking current process's temporary file, then check all possible temporary files
    temp_cache_file = None
    # Current process's temporary file
    current_pid_temp = f"{cache_file}.tmp.{os.getpid()}"
    if os.path.exists(current_pid_temp):
        temp_cache_file = current_pid_temp
    else:
        # Check old format temporary file
        old_temp = cache_file + ".tmp"
        if os.path.exists(old_temp):
            temp_cache_file = old_temp
        else:
            # Check other processes' temporary files (for recovery)
            cache_dir = os.path.dirname(cache_file)
            if os.path.exists(cache_dir):
                import glob
                temp_files = glob.glob(f"{cache_file}.tmp.*")
                if temp_files:
                    # Use the latest temporary file
                    temp_cache_file = max(temp_files, key=os.path.getmtime)
    
    if temp_cache_file and os.path.exists(temp_cache_file):
        try:
            saved_data = torch.load(temp_cache_file, map_location='cpu')
            saved_graphs = saved_data.get('graphs', [])
            processed_count = saved_data.get('processed_count', 0)
            total_molecules = saved_data.get('total_molecules', len(smiles_list))
            print(f"📂 Found partial checkpoint: {len(saved_graphs)}/{total_molecules} molecules already processed")
            print(f"   Will resume from index {processed_count}...")
        except Exception as e:
            print(f"⚠️  Failed to read checkpoint info, will attempt recovery: {e}")
    
    # Process data (supports resuming from checkpoint, saves every 200 molecules)
    print(f"📊 Processing and saving graphs to: {cache_file}")
    print(f"   Checkpoint interval: every 200 molecules")
    print(f"   Workers: {num_workers if num_workers > 0 else cpu_count()}")
    graphs, valid_labels = process_tdc_smiles_to_graphs(
        smiles_list, labels_list, graph_builder,
        cache_file=cache_file,
        checkpoint_interval=200,
        num_workers=num_workers
    )
    
    # Save final processing results (using atomic write: write to temporary file first, then rename)
    # Use temporary filename with PID to avoid multiprocessing conflicts
    temp_final_file = f"{cache_file}.tmp.{os.getpid()}"
    
    try:
        # Step 1: Write to temporary file
        torch.save(graphs, temp_final_file)
        
        # Step 2: Atomic operation: rename temporary file to final filename
        # On most file systems, rename is an atomic operation, ensuring write integrity
        os.rename(temp_final_file, cache_file)
        
        # Step 3: Verify save success (verify graph count)
        if os.path.exists(cache_file):
            saved_graphs = torch.load(cache_file, map_location='cpu')
            if len(saved_graphs) == len(graphs):
                print(f"💾 Saved {len(graphs)} graphs to {cache_file} (verified: {len(saved_graphs)} graphs)")
            else:
                print(f"⚠️  Warning: Saved {len(saved_graphs)} graphs, expected {len(graphs)}")
                # If count doesn't match, delete and resave
                os.remove(cache_file)
                raise RuntimeError(f"Graph count mismatch: saved {len(saved_graphs)}, expected {len(graphs)}")
        else:
            raise RuntimeError("Atomic write failed: cache file not found after rename")
            
    except Exception as e:
        # If save fails, clean up temporary file
        if os.path.exists(temp_final_file):
            try:
                os.remove(temp_final_file)
            except:
                pass
        raise RuntimeError(f"Failed to save graphs: {e}")
    
    # Clean up all temporary files (including checkpoints and old format)
    # Clean up current process's temporary file
    if os.path.exists(temp_final_file):
        try:
            os.remove(temp_final_file)
        except:
            pass
    
    # Clean up old format temporary file
    old_temp_file = cache_file + ".tmp"
    if os.path.exists(old_temp_file):
        try:
            os.remove(old_temp_file)
        except:
            pass
    
    # Clean up other possible temporary files (left by other processes, older than 1 hour)
    cache_dir = os.path.dirname(cache_file)
    if os.path.exists(cache_dir):
        import glob
        import time
        temp_files = glob.glob(f"{cache_file}.tmp.*")
        for temp_file in temp_files:
            try:
                # Only clean up obviously old temporary files (older than 1 hour)
                if time.time() - os.path.getmtime(temp_file) > 3600:
                    os.remove(temp_file)
            except:
                pass
    
    return graphs


def load_tdc_dataset(data_name: str,
                     data_path: str,
                     seed: int,
                     use_fingerprint: bool = False,
                     descriptor_dim: Optional[int] = None,
                     fingerprint_bits: int = 2048,
                     num_conformers: int = 10,
                     optimize_conformers: bool = True,
                     add_hydrogens: bool = True,
                     num_workers: int = 1) -> Tuple[List[Data], List[Data], List[Data]]:
    """
    Load TDC dataset and process into AEGNN-M format (standard mode)
    
    Uses fusion_model's TDC loading and splitting logic
    Uses AEGNN-M's MolecularGraphBuilder to process node features (78 dimensions, OGB-style), edge_attr, pos, and descriptor
    
    Note:
    - TDC's split_type='default' corresponds to Random Split
    - Directly uses TDC's original split results without additional processing (consistent with fusion_model)
    
    Args:
        data_name: TDC dataset name (e.g., 'caco2_wang', 'ames', etc.)
        data_path: TDC data path
        seed: Random seed (for train/valid split)
        use_fingerprint: Whether to use molecular fingerprints (default False)
        descriptor_dim: Descriptor dimension (None means use all available descriptors)
        fingerprint_bits: Number of fingerprint bits
        num_conformers: Number of conformers for 3D coordinate generation
        optimize_conformers: Whether to optimize conformers
        add_hydrogens: Whether to add hydrogen atoms
    
    Returns:
        train_graphs, valid_graphs, test_graphs: Three split graph data lists
    """
    # 0. Set random seed (completely consistent with fusion_model)
    set_seed(seed)
    
    # 1. Load TDC data and perform split (using fusion_model's logic)
    # split_type='default' corresponds to TDC's Random Split
    print(f"📥 Loading TDC dataset: {data_name}")
    print(f"   Split method: Random Split (TDC default)")
    group = admet_group(path=data_path)
    benchmark = group.get(data_name)
    name = benchmark['name']
    test_df = benchmark['test']
    
    # Get train_val data (TDC's original split, use directly without additional processing)
    train_df, valid_df = group.get_train_valid_split(benchmark=name, split_type='default', seed=seed)
    
    # Extract SMILES and labels (directly use TDC's split results, consistent with fusion_model)
    smiles_train = train_df['Drug'].tolist()
    smiles_valid = valid_df['Drug'].tolist()
    smiles_test = test_df['Drug'].tolist()
    
    labels_train = train_df['Y'].tolist()
    labels_valid = valid_df['Y'].tolist()
    labels_test = test_df['Y'].tolist()
    
    print(f"   Train: {len(smiles_train)} samples")
    print(f"   Valid: {len(smiles_valid)} samples")
    print(f"   Test: {len(smiles_test)} samples (fixed)")
    
    # 2. Create AEGNN-M's graph builder (used to generate edge_attr, pos, and descriptor)
    # descriptor is enabled by default
    graph_builder = MolecularGraphBuilder(
        use_descriptor=True,  # Force use of descriptor
        use_fingerprint=use_fingerprint,
        descriptor_dim=descriptor_dim,
        fingerprint_bits=fingerprint_bits,
        num_conformers=num_conformers,
        optimize_conformers=optimize_conformers,
        add_hydrogens=add_hydrogens
    )
    
    # 3. Process each split
    print(f"\n🔄 Processing graphs with AEGNN-M's MolecularGraphBuilder...")
    print(f"   Node features: {graph_builder.node_feature_dim}D (OGB-style, RDKit)")
    print(f"   Edge features: {graph_builder.edge_feature_dim}D (OGB-style)")
    print(f"   3D coordinates: Required")
    print(f"   Descriptors: {graph_builder.descriptor_dim}D (Required)")
    if use_fingerprint:
        print(f"   Fingerprints: {fingerprint_bits} bits")
    
    train_graphs = load_or_process_tdc_graphs(
        smiles_train, labels_train, data_name, "train", seed,
        graph_builder, use_cv=False, num_workers=num_workers
    )
    
    valid_graphs = load_or_process_tdc_graphs(
        smiles_valid, labels_valid, data_name, "valid", seed,
        graph_builder, use_cv=False, num_workers=num_workers
    )
    
    test_graphs = load_or_process_tdc_graphs(
        smiles_test, labels_test, data_name, "test", seed,
        graph_builder, use_cv=False, num_workers=num_workers
    )
    
    print(f"\n✅ Dataset processing complete!")
    print(f"   Train graphs: {len(train_graphs)}")
    print(f"   Valid graphs: {len(valid_graphs)}")
    print(f"   Test graphs: {len(test_graphs)}")
    
    return train_graphs, valid_graphs, test_graphs


def load_tdc_dataset_cv(data_name: str,
                        data_path: str,
                        seed: int,
                        outer_fold_idx: int = 0,
                        inner_fold_idx: Optional[int] = None,
                        outer_folds: int = 5,
                        inner_folds: int = 4,
                        use_fingerprint: bool = False,
                        descriptor_dim: Optional[int] = None,
                        fingerprint_bits: int = 2048,
                        num_conformers: int = 10,
                        optimize_conformers: bool = True,
                        add_hydrogens: bool = True,
                        num_workers: int = 1) -> Tuple[List[Data], List[Data], List[Data]]:
    """
    Load TDC dataset and process into AEGNN-M format (nested cross-validation mode)
    
    Uses fusion_model's nested CV splitting logic
    Uses AEGNN-M's MolecularGraphBuilder to generate complete graph data
    
    Note: TDC's split_type='default' corresponds to Random Split, but CV mode will re-use KFold for splitting
    
    Logic completely consistent with fusion_model:
    - inner_fold_idx defaults to formula: (outer_fold_idx + 1) % inner_folds
    - seeding should be set before calling this function (consistent with fusion_model)
    
    Args:
        data_name: TDC dataset name
        data_path: TDC data path
        seed: Random seed (consistent with fusion_model, CV mode uses 42)
        outer_fold_idx: Outer fold index (0-4)
        inner_fold_idx: Inner fold index (0-3), if None, uses fusion_model formula: (outer_fold_idx + 1) % inner_folds
        outer_folds: Number of outer folds (default 5)
        inner_folds: Number of inner folds (default 4)
        use_descriptor: Whether to use molecular descriptors
        use_fingerprint: Whether to use molecular fingerprints
        descriptor_dim: Descriptor dimension
        fingerprint_bits: Number of fingerprint bits
        num_conformers: Number of conformers for 3D coordinate generation
        optimize_conformers: Whether to optimize conformers
        add_hydrogens: Whether to add hydrogen atoms
        num_workers: Number of parallel worker processes (1 = sequential, >1 = multiprocessing, 0 = use all CPU cores)
    
    Returns:
        train_graphs, valid_graphs, test_graphs: Three split graph data lists
    """
    # Calculate inner_fold_idx (consistent with fusion_model)
    if inner_fold_idx is None:
        inner_fold_idx = (outer_fold_idx + 1) % inner_folds
    # Note: Consistent with fusion_model, seeding should be set before calling this function
    # Here we don't call set_seed() because fusion_model calls it inside each outer_fold loop
    
    # 1. Load all TDC data and merge (using fusion_model's CV logic)
    # Note: Although using split_type='default', CV mode will re-use KFold for splitting
    print(f"📥 Loading TDC dataset for CV: {data_name}")
    print(f"   Split method: KFold (nested CV, overrides TDC default split)")
    group = admet_group(path=data_path)
    benchmark = group.get(data_name)
    name = benchmark['name']
    test_df = benchmark['test']
    train_df, valid_df = group.get_train_valid_split(benchmark=name, split_type='default', seed=seed)
    full_df = pd.concat([train_df, valid_df, test_df], ignore_index=True)
    
    # 2. Outer CV: split trainval and test
    outer_kf = KFold(n_splits=outer_folds, shuffle=True, random_state=seed)
    outer_splits = list(outer_kf.split(full_df))
    outer_trainval_idx, test_idx = outer_splits[outer_fold_idx]
    
    trainval_df = full_df.iloc[outer_trainval_idx].reset_index(drop=True)
    test_df = full_df.iloc[test_idx].reset_index(drop=True)
    
    # 3. Inner CV: split train and valid
    # Note: Consistent with fusion_model, use the same KFold parameters
    inner_kf = KFold(n_splits=inner_folds, shuffle=True, random_state=seed)
    inner_splits = list(inner_kf.split(trainval_df))
    train_idx, valid_idx = inner_splits[inner_fold_idx]
    
    train_df = trainval_df.iloc[train_idx].reset_index(drop=True)
    valid_df = trainval_df.iloc[valid_idx].reset_index(drop=True)
    
    # Extract SMILES and labels
    smiles_train = train_df['Drug'].tolist()
    smiles_valid = valid_df['Drug'].tolist()
    smiles_test = test_df['Drug'].tolist()
    
    labels_train = train_df['Y'].tolist()
    labels_valid = valid_df['Y'].tolist()
    labels_test = test_df['Y'].tolist()
    
    print(f"   Outer fold {outer_fold_idx + 1}/{outer_folds}, Inner fold {inner_fold_idx + 1}/{inner_folds}")
    print(f"   Train: {len(smiles_train)} samples")
    print(f"   Valid: {len(smiles_valid)} samples")
    print(f"   Test: {len(smiles_test)} samples")
    
    # 4. Create AEGNN-M's graph builder (descriptor enabled by default)
    graph_builder = MolecularGraphBuilder(
        use_descriptor=True,  # Force use of descriptor
        use_fingerprint=use_fingerprint,
        descriptor_dim=descriptor_dim,
        fingerprint_bits=fingerprint_bits,
        num_conformers=num_conformers,
        optimize_conformers=optimize_conformers,
        add_hydrogens=add_hydrogens
    )
    
    # 5. Process each split
    print(f"\n🔄 Processing graphs with AEGNN-M's MolecularGraphBuilder...")
    print(f"   Node features: {graph_builder.node_feature_dim}D (OGB-style, RDKit)")
    print(f"   Edge features: {graph_builder.edge_feature_dim}D (OGB-style)")
    print(f"   Descriptors: {graph_builder.descriptor_dim}D (Required)")
    split_tag = f"outer{outer_fold_idx}_inner{inner_fold_idx}"
    
    train_graphs = load_or_process_tdc_graphs(
        smiles_train, labels_train, data_name, f"{split_tag}_train", seed,
        graph_builder, use_cv=True, outer_fold_idx=outer_fold_idx, num_workers=num_workers
    )
    
    valid_graphs = load_or_process_tdc_graphs(
        smiles_valid, labels_valid, data_name, f"{split_tag}_valid", seed,
        graph_builder, use_cv=True, outer_fold_idx=outer_fold_idx, num_workers=num_workers
    )
    
    test_graphs = load_or_process_tdc_graphs(
        smiles_test, labels_test, data_name, f"{split_tag}_test", seed,
        graph_builder, use_cv=True, outer_fold_idx=outer_fold_idx, num_workers=num_workers
    )
    
    print(f"\n✅ CV dataset processing complete!")
    print(f"   Train graphs: {len(train_graphs)}")
    print(f"   Valid graphs: {len(valid_graphs)}")
    print(f"   Test graphs: {len(test_graphs)}")
    
    return train_graphs, valid_graphs, test_graphs


# Helper function: verify graph data format
def verify_graph_format(graph: Data) -> bool:
    """
    Verify that graph data contains all fields required by AEGNN-M (using OGB-style 78-dimensional node features)
    
    Args:
        graph: PyTorch Geometric Data object
    
    Returns:
        bool: Whether all required fields are present
    """
    required_fields = ['x', 'edge_index', 'edge_attr', 'pos', 'y', 'descriptor']
    missing_fields = [field for field in required_fields if not hasattr(graph, field) or getattr(graph, field) is None]
    
    if missing_fields:
        print(f"⚠️  Missing required fields: {missing_fields}")
        return False
    
    # Verify dimensions (OGB-style: 78 dimensions)
    if graph.x.dim() != 2 or graph.x.shape[1] != 78:
        print(f"⚠️  Node features should be [num_atoms, 78] (OGB-style), got {graph.x.shape}")
        return False
    
    if graph.edge_attr.dim() != 2 or graph.edge_attr.shape[1] != 9:
        print(f"⚠️  Edge features should be [num_edges, 9], got {graph.edge_attr.shape}")
        return False
    
    if graph.pos.dim() != 2 or graph.pos.shape[1] != 3:
        print(f"⚠️  3D positions should be [num_atoms, 3], got {graph.pos.shape}")
        return False
    
    if graph.descriptor.dim() != 2:
        print(f"⚠️  Descriptor should be [1, dim], got {graph.descriptor.shape}")
        return False
    
    return True


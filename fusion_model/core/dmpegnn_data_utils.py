"""
AEGNN-M Data Processing Utilities
For preprocessing and converting molecular graph data
"""

import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_networkx, from_networkx
import networkx as nx
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import AllChem
from collections import defaultdict
import random
import pandas as pd
from typing import List, Dict, Tuple, Optional
import pickle
import os
import warnings
import logging
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import sys
from multiprocessing import Pool, cpu_count, TimeoutError as MPTimeoutError
from functools import partial
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FutureTimeoutError
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    # Fallback: create a simple progress indicator
    def tqdm(iterable, desc="", total=None, **kwargs):
        return iterable

# Set RDKit log level to suppress UFF-related warnings
try:
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')  # Disable all RDKit application logs
except:
    pass


class MolecularGraphBuilder:
    """Molecular Graph Builder"""
    
    def __init__(self, 
                 use_atomic_number=True,
                 use_hybridization=True,
                 use_formal_charge=True,
                 use_aromatic=True,
                 use_chirality=True,
                 use_hydrogen_bonds=True,
                 use_bond_type=True,
                 use_bond_stereo=True,
                 num_conformers=10,
                 optimize_conformers=True,
                 num_conformers_to_keep=1,
                 num_threads=0,
                 add_hydrogens=True,
                 prune_rms_thresh=0.5,
                 use_fingerprint=False,
                 fingerprint_radius=2,
                 fingerprint_bits=2048,
                 use_descriptor=False,
                 descriptor_dim=None):
        """
        Args:
            descriptor_dim: Descriptor dimension. If None, will use all available RDKit descriptors (~217).
                            If specified, will be used for model initialization (but actual descriptors won't be truncated).
        """
        
        self.use_atomic_number = use_atomic_number
        self.use_hybridization = use_hybridization
        self.use_formal_charge = use_formal_charge
        self.use_aromatic = use_aromatic
        self.use_chirality = use_chirality
        self.use_hydrogen_bonds = use_hydrogen_bonds
        self.use_bond_type = use_bond_type
        self.use_bond_stereo = use_bond_stereo
        self.num_conformers = num_conformers
        self.optimize_conformers = optimize_conformers
        self.num_conformers_to_keep = max(1, int(num_conformers_to_keep))
        self.num_threads = num_threads
        self.add_hydrogens = add_hydrogens
        self.prune_rms_thresh = prune_rms_thresh
        
        # Fingerprint settings
        self.use_fingerprint = use_fingerprint
        self.fingerprint_radius = fingerprint_radius
        self.fingerprint_bits = fingerprint_bits
        
        # Descriptor settings
        self.use_descriptor = use_descriptor
        # Auto-detect descriptor dimension if not specified
        if use_descriptor and descriptor_dim is None:
            try:
                from rdkit import Chem
                from rdkit.Chem import Descriptors
                # Use a simple molecule to get descriptor count
                dummy_mol = Chem.MolFromSmiles('C')
                dummy_desc = Descriptors.CalcMolDescriptors(dummy_mol)
                self.descriptor_dim = len(dummy_desc)
            except:
                self.descriptor_dim = 217  # Fallback to known count
        else:
            self.descriptor_dim = descriptor_dim if descriptor_dim is not None else 217
        
        # Statistics: Record MMFF and UFF usage
        self.mmff_success_count = 0
        self.uff_fallback_count = 0
        
        # 3D coordinate generation failure records: Record failed SMILES and reasons
        self.failed_3d_generation = []  # Format: [{'smiles': str, 'reason': str}, ...]
        
        # Atom feature dimension calculation
        self.node_feature_dim = self._calculate_node_feature_dim()
        self.edge_feature_dim = self._calculate_edge_feature_dim()
    
    def _generate_fingerprint(self, mol) -> np.ndarray:
        """Generate Morgan Fingerprint"""
        if not self.use_fingerprint:
            return None
        
        try:
            fp = AllChem.GetMorganFingerprintAsBitVect(
                mol, 
                radius=self.fingerprint_radius, 
                nBits=self.fingerprint_bits
            )
            return np.array(fp, dtype=np.float32)
        except:
            return np.zeros(self.fingerprint_bits, dtype=np.float32)
    
    def _generate_descriptor(self, mol) -> np.ndarray:
        """Generate RDKit normalized descriptors (all available descriptors, no truncation)"""
        if not self.use_descriptor:
            return None
        
        try:
            # Calculate all available RDKit descriptors
            # Note: CalcMolDescriptors is in Descriptors module, not rdMolDescriptors
            desc_dict = Descriptors.CalcMolDescriptors(mol)
            
            # Convert to numpy array and handle NaN/Inf values
            desc_values = []
            for key in sorted(desc_dict.keys()):  # Sort for consistency
                val = desc_dict[key]
                # Replace NaN and Inf with 0
                if np.isnan(val) or np.isinf(val):
                    val = 0.0
                desc_values.append(float(val))
            
            desc_array = np.array(desc_values, dtype=np.float32)
            
            # Use all available descriptors (no truncation or padding)
            # RDKit CalcMolDescriptors returns ~217 descriptors
            return desc_array
        except Exception as e:
            # Return zero vector with actual descriptor count if calculation fails
            # Try to get the count from a dummy molecule
            try:
                from rdkit import Chem
                dummy_mol = Chem.MolFromSmiles('C')
                dummy_desc = Descriptors.CalcMolDescriptors(dummy_mol)
                actual_dim = len(dummy_desc)
            except:
                actual_dim = 217  # Fallback to known count
            return np.zeros(actual_dim, dtype=np.float32)
    
    def _calculate_node_feature_dim(self):
        """
        Calculate node feature dimension (OGB-style, extended heavy elements)
        - Atom Type: 48 dims (One-Hot)
            - 47 common elements (H ~ Bi, non-contiguous Z)
            - +1 for "Unknown" elements not in the list
        - Degree: 11 dims (One-Hot, 0-10)
        - Formal Charge: 11 dims (One-Hot, -5 to +5)
        - Hybridization: 5 dims (One-Hot)
        - Aromaticity: 1 dim (Binary)
        - Total Num H: 6 dims (One-Hot, 0-5)
        Total: 48 + 11 + 11 + 5 + 1 + 6 = 82
        """
        return 82
    
    def _calculate_edge_feature_dim(self):
        """
        Calculate edge feature dimension (OGB-style: 9 dimensions)
        - Bond Type: 4 dims (One-Hot: Single, Double, Triple, Aromatic)
        - Stereo: 4 dims (One-Hot: None, Any, Z, E)
        - Is Conjugated: 1 dim (Binary)
        Total: 4 + 4 + 1 = 9
        """
        return 9
    
    def _generate_3d_coordinates(self, mol, smiles: str = None) -> Optional[np.ndarray]:
        """
        Generate 3D coordinates for molecules
        Generate multiple conformers, optimize and select the lowest energy (optimized) conformer
        Prefer MMFF, fallback to UFF if unavailable
        
        Args:
            mol: RDKit molecule object
            smiles: SMILES string (for recording failure information)
            
        Returns:
            3D coordinate array [N, 3], returns None if generation fails
        """
        try:
            # Clear existing conformers (if any)
            mol.RemoveAllConformers()
            
            # If num_conformers > 1, generate multiple conformers
            if self.num_conformers > 1:
                result = self._generate_multiple_conformers(
                    mol, smiles, return_top_k=self.num_conformers_to_keep
                )
            else:
                # Single conformer generation (for backward compatibility)
                result = self._generate_single_conformer(mol, smiles)
            
            # Note: Failure information is already recorded in sub-methods, no need to repeat here
            return result
            
        except Exception as e:
            # Record exception information
            if smiles is not None:
                self.failed_3d_generation.append({
                    'smiles': smiles,
                    'reason': f'Exception during 3D coordinate generation: {str(e)}'
                })
            return None
    
    def _generate_multiple_conformers(
        self, mol, smiles: str = None, return_top_k: int = 1
    ) -> Optional[np.ndarray]:
        """
        Use RDKit's native EmbedMultipleConfs to generate multiple conformers
        and select the optimized one (lowest energy) or top-k by energy.
        Supports multi-threading acceleration.

        Args:
            mol: RDKit molecule object
            return_top_k: If > 1, return list of top-k conformer positions; else return single [N, 3].

        Returns:
            If return_top_k <= 1: 3D coordinate array [N, 3] of the best conformer.
            If return_top_k > 1: List of [N, 3] arrays (length up to return_top_k).
        """
        try:
            # Clear existing conformers
            mol.RemoveAllConformers()
            
            # Use RDKit's native EmbedMultipleConfs to generate multiple conformers
            # Use ETKDG method (RDKit recommended method) with multi-threading and RMSD filtering support
            try:
                # Try using ETKDGv3 parameter object (supports numThreads and pruneRmsThresh)
                params = AllChem.ETKDGv3()
                params.useExpTorsionAnglePrefs = True
                params.useBasicKnowledge = True
                params.randomSeed = 42
                params.numThreads = self.num_threads
                # Set RMSD threshold to filter similar conformers (if specified)
                if self.prune_rms_thresh is not None and self.prune_rms_thresh > 0:
                    params.pruneRmsThresh = self.prune_rms_thresh
                
                # Note: maxAttempts is not supported when using params object
                # The params object handles attempts internally
                ids = AllChem.EmbedMultipleConfs(mol, numConfs=self.num_conformers, params=params)
            except:
                # If ETKDGv3 is not available, try ETKDGv2
                try:
                    params = AllChem.ETKDGv2()
                    params.useExpTorsionAnglePrefs = True
                    params.useBasicKnowledge = True
                    params.randomSeed = 42
                    params.numThreads = self.num_threads
                    # Set RMSD threshold to filter similar conformers (if specified)
                    if self.prune_rms_thresh is not None and self.prune_rms_thresh > 0:
                        params.pruneRmsThresh = self.prune_rms_thresh
                    
                    # Note: maxAttempts is not supported when using params object
                    ids = AllChem.EmbedMultipleConfs(mol, numConfs=self.num_conformers, params=params)
                except:
                    # If all fail, use basic method (without multi-threading and RMSD filtering)
                    ids = AllChem.EmbedMultipleConfs(
                        mol, 
                        numConfs=self.num_conformers,
                        useExpTorsionAnglePrefs=True,
                        useBasicKnowledge=True,
                        randomSeed=42,
                        maxAttempts=50  # Limit attempts
                    )
            
            # If ETKDG fails, try default method without parameters
            if len(ids) == 0:
                ids = AllChem.EmbedMultipleConfs(
                    mol,
                    numConfs=self.num_conformers,
                    randomSeed=42,
                    maxAttempts=50  # Limit attempts
                )
            
            # If all fail, fallback to single conformer
            if len(ids) == 0:
                return self._generate_single_conformer(mol, smiles)
            
            # Get conformer energies (one optimization pass) and sort by energy
            energies_list = self._get_conformer_energies_sorted(mol, ids)
            if not energies_list:
                best_conf_id = ids[0]
                conf = mol.GetConformer(best_conf_id)
                pos = conf.GetPositions().astype(np.float32)
                return [pos] if return_top_k > 1 else pos

            if return_top_k > 1:
                k = min(return_top_k, len(energies_list))
                top_ids = [conf_id for conf_id, _ in energies_list[:k]]
                return [
                    np.array(mol.GetConformer(cid).GetPositions(), dtype=np.float32)
                    for cid in top_ids
                ]
            # Single conformer: return one array (best by energy)
            best_conf_id = energies_list[0][0]
            conf = mol.GetConformer(best_conf_id)
            pos = conf.GetPositions()
            return pos.astype(np.float32)
            
        except Exception as e:
            # If multiple conformer generation fails, fallback to single conformer
            if smiles is not None:
                self.failed_3d_generation.append({
                    'smiles': smiles,
                    'reason': f'Multiple conformer generation failed: {str(e)}'
                })
            single = self._generate_single_conformer(mol, smiles)
            if return_top_k > 1 and single is not None:
                return [single]
            return single
    
    def _optimize_and_select_best_mmff(self, mol, conf_ids) -> Optional[int]:
        """
        Optimize all conformers using MMFF force field and return the conformer ID with lowest energy
        Supports multi-threading acceleration
        
        Args:
            mol: RDKit molecule object
            conf_ids: List of conformer IDs
            
        Returns:
            Conformer ID with lowest energy, returns None if failed
        """
        try:
            # Prefer MMFFOptimizeMoleculeConfs (supports multi-threading)
            try:
                # Optimize all conformers using multi-threading
                # Set maxIters to prevent hanging (default is usually 200, but explicit setting is safer)
                results = AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=self.num_threads, maxIters=200)
                
                # results is a list, each element is (notConverged, energy)
                # notConverged: 0 means converged, non-zero means not converged
                # energy: optimized energy
                energies = []
                for i, conf_id in enumerate(conf_ids):
                    if i < len(results):
                        not_converged, energy = results[i]
                        # Only consider converged conformers (notConverged == 0)
                        if not_converged == 0:
                            energies.append((conf_id, energy))
                
                if energies:
                    # Select conformer with lowest energy
                    best_conf_id, _ = min(energies, key=lambda x: x[1])
                    return best_conf_id
            except:
                # If MMFFOptimizeMoleculeConfs fails, fallback to single-threaded method
                pass
            
            # Fallback to single-threaded method (optimize one by one)
            mmff_props = AllChem.MMFFGetMoleculeProperties(mol)
            if mmff_props is None:
                return None
            
            energies = []
            for conf_id in conf_ids:
                try:
                    ff = AllChem.MMFFGetMoleculeForceField(mol, mmff_props, confId=conf_id)
                    if ff is not None:
                        ff.Minimize(maxIts=200)  # Limit iterations
                        energy = ff.CalcEnergy()
                        energies.append((conf_id, energy))
                except:
                    continue
            
            if energies:
                # Select conformer with lowest energy
                best_conf_id, _ = min(energies, key=lambda x: x[1])
                return best_conf_id
            return None
        except:
            return None
    
    def _optimize_and_select_best_uff(self, mol, conf_ids) -> Optional[int]:
        """
        Optimize all conformers using UFF force field and return the conformer ID with lowest energy
        Supports multi-threading acceleration
        
        Args:
            mol: RDKit molecule object
            conf_ids: List of conformer IDs
            
        Returns:
            Conformer ID with lowest energy, returns None if failed
        """
        try:
            # Prefer UFFOptimizeMoleculeConfs (supports multi-threading)
            try:
                # Suppress UFF "Unrecognized" warnings (these warnings have little impact on processing results)
                # Use redirect_stderr and redirect_stdout to capture all RDKit output
                stderr_buffer = StringIO()
                stdout_buffer = StringIO()
                # Suppress both stdout and stderr, as some errors may output to different streams
                with redirect_stderr(stderr_buffer), redirect_stdout(stdout_buffer):
                    # Optimize all conformers using multi-threading
                    # Set maxIters to prevent hanging
                    results = AllChem.UFFOptimizeMoleculeConfs(mol, numThreads=self.num_threads, maxIters=200)
                
                # results is a list, each element is (notConverged, energy)
                # notConverged: 0 means converged, non-zero means not converged
                # energy: optimized energy
                energies = []
                for i, conf_id in enumerate(conf_ids):
                    if i < len(results):
                        not_converged, energy = results[i]
                        # Only consider converged conformers (notConverged == 0)
                        if not_converged == 0:
                            energies.append((conf_id, energy))
                
                if energies:
                    # Select conformer with lowest energy
                    best_conf_id, _ = min(energies, key=lambda x: x[1])
                    return best_conf_id
            except:
                # If UFFOptimizeMoleculeConfs fails, fallback to single-threaded method
                pass
            
            # Fallback to single-threaded method (optimize one by one)
            energies = []
            for conf_id in conf_ids:
                try:
                    # Suppress UFF "Unrecognized" warnings and Pre-condition errors
                    stderr_buffer = StringIO()
                    stdout_buffer = StringIO()
                    with redirect_stderr(stderr_buffer), redirect_stdout(stdout_buffer):
                        ff = AllChem.UFFGetMoleculeForceField(mol, confId=conf_id)
                        if ff is not None:
                            # Set maxIts to prevent hanging on complex molecules
                            ff.Minimize(maxIts=200)
                            energy = ff.CalcEnergy()
                            energies.append((conf_id, energy))
                except:
                    continue
            
            if energies:
                # Select conformer with lowest energy
                best_conf_id, _ = min(energies, key=lambda x: x[1])
                return best_conf_id
            return None
        except:
            return None

    def _get_conformer_energies_sorted(
        self, mol, conf_ids: List[int]
    ) -> List[Tuple[int, float]]:
        """
        Get (conf_id, energy) for all conformers, sorted by energy (ascending).
        Tries MMFF first, then UFF. Returns empty list if both fail.
        """
        energies: List[Tuple[int, float]] = []
        if self.optimize_conformers:
            try:
                results = AllChem.MMFFOptimizeMoleculeConfs(
                    mol, numThreads=self.num_threads, maxIters=200
                )
                for i, conf_id in enumerate(conf_ids):
                    if i < len(results):
                        not_converged, energy = results[i]
                        if not_converged == 0:
                            energies.append((conf_id, float(energy)))
            except Exception:
                pass
        if not energies:
            try:
                stderr_buf = StringIO()
                stdout_buf = StringIO()
                with redirect_stderr(stderr_buf), redirect_stdout(stdout_buf):
                    results = AllChem.UFFOptimizeMoleculeConfs(
                        mol, numThreads=self.num_threads, maxIters=200
                    )
                for i, conf_id in enumerate(conf_ids):
                    if i < len(results):
                        not_converged, energy = results[i]
                        if not_converged == 0:
                            energies.append((conf_id, float(energy)))
            except Exception:
                pass
        if not energies:
            # No optimization: use first conformer only with dummy energy
            energies = [(conf_ids[0], 0.0)]
        energies.sort(key=lambda x: x[1])
        return energies
    
    def _generate_single_conformer(self, mol, smiles: str = None) -> Optional[np.ndarray]:
        """
        Generate single conformer (original method, for backward compatibility)
        According to RDKit official documentation, prefer ETKDG method, then perform force field optimization
        
        Args:
            mol: RDKit molecule object
            
        Returns:
            3D coordinate array [N, 3], returns None if generation fails
        """
        try:
            # Clear existing conformers (if any)
            mol.RemoveAllConformers()
            
            # According to RDKit official documentation, prefer ETKDG method to generate initial conformation
            # ETKDG is the modern method recommended by RDKit
            embed_success = False
            
            # Prefer trying ETKDGv3 (latest version)
            try:
                params = AllChem.ETKDGv3()
                params.useExpTorsionAnglePrefs = True
                params.useBasicKnowledge = True
                params.randomSeed = 42
                # Note: maxAttempts is not supported when using params object
                # The params object handles attempts internally
                if AllChem.EmbedMolecule(mol, params) == 0:
                    embed_success = True
            except:
                # If ETKDGv3 is not available, try ETKDGv2
                try:
                    params = AllChem.ETKDGv2()
                    params.useExpTorsionAnglePrefs = True
                    params.useBasicKnowledge = True
                    params.randomSeed = 42
                    # Note: maxAttempts is not supported when using params object
                    if AllChem.EmbedMolecule(mol, params) == 0:
                        embed_success = True
                except:
                    # If all fail, try basic ETKDG
                    try:
                        if AllChem.EmbedMolecule(mol, AllChem.ETKDG()) == 0:
                            embed_success = True
                    except:
                        # Finally try traditional method
                        if AllChem.EmbedMolecule(mol, useExpTorsionAnglePrefs=True, useBasicKnowledge=True) == 0:
                            embed_success = True
            
            if not embed_success:
                if smiles is not None:
                    self.failed_3d_generation.append({
                        'smiles': smiles,
                        'reason': 'EmbedMolecule failed: could not generate initial 3D conformation'
                    })
                return None
            
            # According to RDKit documentation, force field optimization should be performed after generating conformation
            # Prefer MMFF (more accurate, applicable to organic molecules)
            try:
                if AllChem.MMFFOptimizeMolecule(mol) == 0:
                    # Successfully optimized using MMFF
                    conf = mol.GetConformer()
                    pos = conf.GetPositions()
                    return pos.astype(np.float32)
            except:
                # MMFF optimization failed, try UFF (wider applicability)
                try:
                    # Suppress UFF error messages
                    stderr_buffer = StringIO()
                    stdout_buffer = StringIO()
                    with redirect_stderr(stderr_buffer), redirect_stdout(stdout_buffer):
                        if AllChem.UFFOptimizeMolecule(mol) == 0:
                            # Successfully optimized using UFF
                            conf = mol.GetConformer()
                            pos = conf.GetPositions()
                            return pos.astype(np.float32)
                except:
                    pass
            
            # If optimization fails, return unoptimized conformation (better than nothing)
            conf = mol.GetConformer()
            pos = conf.GetPositions()
            return pos.astype(np.float32)
            
        except Exception as e:
            # Record exception information
            if smiles is not None:
                self.failed_3d_generation.append({
                    'smiles': smiles,
                    'reason': f'Single conformer generation exception: {str(e)}'
                })
            return None
    
    def smiles_to_graph(self, smiles: str) -> Data:
        """Convert SMILES string to graph data"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            # Add hydrogen atoms (important for obtaining realistic geometry)
            if self.add_hydrogens:
                mol = Chem.AddHs(mol)
            
            # Build node features
            node_features = self._get_node_features(mol)
            
            # Build edge indices and edge features
            edge_index, edge_features = self._get_edge_features(mol)
            
            # Generate 3D coordinates (when num_conformers_to_keep>1 returns list of [N,3])
            pos_or_list = self._generate_3d_coordinates(mol, smiles=smiles)
            
            # Generate fingerprint if enabled
            fingerprint = None
            if self.use_fingerprint:
                fingerprint = self._generate_fingerprint(mol)
            
            # Generate descriptor if enabled
            descriptor = None
            if self.use_descriptor:
                descriptor = self._generate_descriptor(mol)
            
            # Ensure all tensor shapes are correct
            # edge_index should be [2, num_edges]
            if edge_index.shape[0] != 2:
                edge_index = edge_index.T if edge_index.ndim == 2 else edge_index.reshape(2, -1)
            
            # edge_attr should be [num_edges, edge_dim]
            if edge_features.ndim == 1:
                edge_features = edge_features.reshape(-1, 1)
            elif edge_features.ndim == 0 or len(edge_features) == 0:
                edge_features = np.empty((0, self.edge_feature_dim), dtype=np.float32)
            
            # Ensure edge_index and edge_attr have consistent number of edges
            num_edges = edge_index.shape[1] if edge_index.ndim == 2 else 0
            if edge_features.shape[0] != num_edges:
                # If counts don't match, adjust edge_features
                if num_edges == 0:
                    edge_features = np.empty((0, self.edge_feature_dim), dtype=np.float32)
                else:
                    # Truncate or pad to correct length
                    if edge_features.shape[0] > num_edges:
                        edge_features = edge_features[:num_edges]
                    else:
                        # Pad (shouldn't happen, but for safety)
                        padding = np.zeros((num_edges - edge_features.shape[0], edge_features.shape[1]), dtype=np.float32)
                        edge_features = np.vstack([edge_features, padding])
            
            # If 3D coordinates cannot be generated, exclude this molecule from training
            # Normalize to list of positions (one or multiple conformers)
            if isinstance(pos_or_list, list):
                pos_list = [p for p in pos_or_list if p is not None and p.shape[0] == len(node_features)]
            else:
                pos_list = [pos_or_list] if (pos_or_list is not None and pos_or_list.shape[0] == len(node_features)) else []
            if not pos_list:
                return None
            
            # Build one Data per conformer (same topology, different pos)
            out = []
            for pos in pos_list:
                data_dict = {
                    'x': torch.tensor(node_features, dtype=torch.float),
                    'edge_index': torch.tensor(edge_index, dtype=torch.long),
                    'edge_attr': torch.tensor(edge_features, dtype=torch.float),
                    'pos': torch.tensor(pos, dtype=torch.float),
                }
                if fingerprint is not None:
                    data_dict['fingerprint'] = torch.tensor(fingerprint, dtype=torch.float).unsqueeze(0)
                if descriptor is not None:
                    data_dict['descriptor'] = torch.tensor(descriptor, dtype=torch.float).unsqueeze(0)
                out.append(Data(**data_dict))
            return out if len(out) > 1 else out[0]
            
        except Exception as e:
            print(f"Error processing SMILES {smiles}: {e}")
            return None
    
    def _get_node_features(self, mol) -> np.ndarray:
        """
        Extract node (atom) features in OGB-style format (82 dimensions)
        
        Feature breakdown:
        1. Atom Type: 48 dims (One-Hot) - 47 common elements in drug discovery + 1 Unknown
        2. Degree: 11 dims (One-Hot) - Number of neighbors (0-10)
        3. Formal Charge: 11 dims (One-Hot) - Formal charge (-5 to +5)
        4. Hybridization: 5 dims (One-Hot) - SP, SP2, SP3, SP3D, SP3D2
        5. Aromaticity: 1 dim (Binary) - Is aromatic
        6. Total Num H: 6 dims (One-Hot) - Number of hydrogens (0-5)
        
        Total: 48 + 11 + 11 + 5 + 1 + 6 = 82 dimensions
        """
        # Common elements in drug discovery (extended OGB-style list)
        # This list includes 47 frequently occurring elements in pharmaceutical compounds.
        # An additional "Unknown" index is reserved for elements not in this list.
        COMMON_ELEMENTS_47 = [
            1,    # H - Hydrogen
            3,    # Li - Lithium
            5,    # B - Boron
            6,    # C - Carbon
            7,    # N - Nitrogen
            8,    # O - Oxygen
            9,    # F - Fluorine
            11,   # Na - Sodium
            12,   # Mg - Magnesium
            13,   # Al - Aluminum
            14,   # Si - Silicon
            15,   # P - Phosphorus
            16,   # S - Sulfur
            17,   # Cl - Chlorine
            19,   # K - Potassium
            20,   # Ca - Calcium
            21,   # Sc - Scandium
            22,   # Ti - Titanium
            23,   # V - Vanadium
            24,   # Cr - Chromium
            25,   # Mn - Manganese
            26,   # Fe - Iron
            27,   # Co - Cobalt
            28,   # Ni - Nickel
            29,   # Cu - Copper
            30,   # Zn - Zinc
            31,   # Ga - Gallium
            32,   # Ge - Germanium
            33,   # As - Arsenic
            34,   # Se - Selenium
            35,   # Br - Bromine
            37,   # Rb - Rubidium
            38,   # Sr - Strontium
            47,   # Ag - Silver
            48,   # Cd - Cadmium
            49,   # In - Indium
            50,   # Sn - Tin
            51,   # Sb - Antimony
            52,   # Te - Tellurium
            53,   # I - Iodine
            55,   # Cs - Cesium
            56,   # Ba - Barium
            78,   # Pt - Platinum
            79,   # Au - Gold
            80,   # Hg - Mercury
            82,   # Pb - Lead
            83,   # Bi - Bismuth
        ]
        
        # Atom type / degree / charge / hybridization / aromatic / num_H layout
        # Atom Type: len(COMMON_ELEMENTS_47) known elements + 1 "Unknown"
        atom_type_dim = len(COMMON_ELEMENTS_47) + 1
        degree_dim = 11
        charge_dim = 11
        hybrid_dim = 5
        aromatic_dim = 1
        num_h_dim = 6

        # Offsets in the final feature vector
        atom_type_offset = 0
        degree_offset = atom_type_offset + atom_type_dim
        charge_offset = degree_offset + degree_dim
        hybrid_offset = charge_offset + charge_dim
        aromatic_offset = hybrid_offset + hybrid_dim
        num_h_offset = aromatic_offset + aromatic_dim

        total_dim = atom_type_dim + degree_dim + charge_dim + hybrid_dim + aromatic_dim + num_h_dim
        if total_dim != self.node_feature_dim:
            # Keep a hard check to avoid silent mismatch between layout and advertised dim
            raise ValueError(f"Node feature dim mismatch: layout={total_dim}, configured={self.node_feature_dim}")

        # Create atomic number to index mapping (0..len-1 for known elements)
        atomic_num_to_idx = {atomic_num: idx for idx, atomic_num in enumerate(COMMON_ELEMENTS_47)}
        # Unknown elements map to the last atom-type index (len(COMMON_ELEMENTS_47))
        UNKNOWN_IDX = atom_type_dim - 1
        
        features = []
        
        for atom in mol.GetAtoms():
            feature_vector = np.zeros(self.node_feature_dim, dtype=np.float32)
            
            # 1. Atom Type: atom_type_dim dims (One-Hot)
            atomic_num = atom.GetAtomicNum()
            atom_type_idx = atomic_num_to_idx.get(atomic_num, UNKNOWN_IDX)
            feature_vector[atom_type_offset + atom_type_idx] = 1.0
            
            # 2. Degree: 11 dims (One-Hot, 0-10)
            degree = atom.GetDegree()
            degree_idx = degree_offset + min(degree, 10)
            feature_vector[degree_idx] = 1.0
            
            # 3. Formal Charge: 11 dims (One-Hot, -5 to +5)
            formal_charge = atom.GetFormalCharge()
            charge_idx = charge_offset + min(max(formal_charge + 5, 0), 10)
            feature_vector[charge_idx] = 1.0
            
            # 4. Hybridization: 5 dims (One-Hot)
            # RDKit hybridization enum values: SP=1, SP2=2, SP3=3, SP3D=4, SP3D2=5
            hybrid = atom.GetHybridization()
            hybrid_idx = hybrid_offset
            # Map RDKit hybridization to 0-4 index
            # SP=1 -> 0, SP2=2 -> 1, SP3=3 -> 2, SP3D=4 -> 3, SP3D2=5 -> 4
            hybrid_int = int(hybrid)
            if 1 <= hybrid_int <= 5:
                feature_vector[hybrid_idx + (hybrid_int - 1)] = 1.0
            # If unknown hybridization (0 or >5), leave as zeros
            
            # 5. Aromaticity: 1 dim (Binary)
            aromatic_idx = aromatic_offset
            feature_vector[aromatic_idx] = 1.0 if atom.GetIsAromatic() else 0.0
            
            # 6. Total Num H: 6 dims (One-Hot, 0-5)
            num_h = atom.GetTotalNumHs()
            num_h_idx = num_h_offset
            num_h_clamped = min(num_h, 5)  # Clamp to 0-5
            feature_vector[num_h_idx + num_h_clamped] = 1.0
            
            features.append(feature_vector)
        
        return np.array(features, dtype=np.float32)
    
    def _get_edge_features(self, mol) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract edge (chemical bond) features in OGB-style format (9 dimensions)
        
        Feature breakdown:
        1. Bond Type: 4 dims (One-Hot) - Single, Double, Triple, Aromatic
        2. Stereo: 4 dims (One-Hot) - None, Any, Z, E (Cis/Trans)
        3. Is Conjugated: 1 dim (Binary) - Is conjugated bond
        
        Total: 4 + 4 + 1 = 9 dimensions
        """
        edge_index = []
        edge_features = []
        
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            
            # Bidirectional edges
            edge_index.append([i, j])
            edge_index.append([j, i])
            
            # Initialize 9-dimensional feature vector
            feature_vector = np.zeros(9, dtype=np.float32)
            
            # 1. Bond Type: 4 dims (One-Hot)
            # RDKit bond types: SINGLE, DOUBLE, TRIPLE, AROMATIC
            bond_type = bond.GetBondType()
            if bond_type == Chem.BondType.SINGLE:
                feature_vector[0] = 1.0  # Single bond
            elif bond_type == Chem.BondType.DOUBLE:
                feature_vector[1] = 1.0  # Double bond
            elif bond_type == Chem.BondType.TRIPLE:
                feature_vector[2] = 1.0  # Triple bond
            elif bond_type == Chem.BondType.AROMATIC:
                feature_vector[3] = 1.0  # Aromatic bond
            else:
                # For unknown bond types, default to Single (index 0)
                feature_vector[0] = 1.0
            
            # 2. Stereo: 4 dims (One-Hot)
            # RDKit stereo types: STEREONONE, STEREOANY, STEREOZ, STEREOE
            stereo = bond.GetStereo()
            stereo_idx = 4  # Base offset for stereo features
            if stereo == Chem.BondStereo.STEREONONE:
                feature_vector[stereo_idx + 0] = 1.0  # None
            elif stereo == Chem.BondStereo.STEREOANY:
                feature_vector[stereo_idx + 1] = 1.0  # Any
            elif stereo == Chem.BondStereo.STEREOZ:
                feature_vector[stereo_idx + 2] = 1.0  # Z (Cis)
            elif stereo == Chem.BondStereo.STEREOE:
                feature_vector[stereo_idx + 3] = 1.0  # E (Trans)
            else:
                # For unknown stereo, default to None (index 4)
                feature_vector[stereo_idx + 0] = 1.0
            
            # 3. Is Conjugated: 1 dim (Binary)
            is_conjugated = bond.GetIsConjugated()
            conjugated_idx = 4 + 4  # Offset by 8
            feature_vector[conjugated_idx] = 1.0 if is_conjugated else 0.0
            
            # Bidirectional edge features (same features for both directions)
            edge_features.append(feature_vector)
            edge_features.append(feature_vector)
        
        # Ensure edge_index and edge_features have correct shapes
        if len(edge_index) == 0:
            # If no edges (single atom molecule), create empty but correctly shaped arrays
            edge_index = np.empty((2, 0), dtype=np.int64)
            edge_features = np.empty((0, self.edge_feature_dim), dtype=np.float32)
        else:
            edge_index = np.array(edge_index, dtype=np.int64).T  # Ensure [2, num_edges] shape
            edge_features = np.array(edge_features, dtype=np.float32)
            # Ensure edge_features is [num_edges, edge_dim] shape
            if edge_features.ndim == 1:
                edge_features = edge_features.reshape(-1, 1)
            elif edge_features.ndim == 0 or len(edge_features) == 0:
                edge_features = np.empty((0, self.edge_feature_dim), dtype=np.float32)
        
        return edge_index, edge_features


# Helper function for multiprocessing
import signal

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Processing timed out")

def _process_single_molecule(args):
    """
    Process a single molecule (helper function for multiprocessing)
    
    Args:
        args: Tuple of (smiles, target, builder_config)
            - smiles: SMILES string
            - target: Target value
            - builder_config: Dictionary with MolecularGraphBuilder configuration
    
    Returns:
        Tuple of (graph, target, smiles, failed_info) or (None, target, smiles, failed_info)
    """
    smiles, target, builder_config = args
    
    # Set timeout (300 seconds = 5 minutes) to prevent hanging on complex molecules
    # Increased timeout to 300 seconds to handle complex molecules that just need more time
    # Note: signal.SIGALRM only works on Unix systems (Linux, macOS)
    try:
        # Only set up signal handler if we're on a Unix system
        if hasattr(signal, 'SIGALRM'):
            # Reset any existing alarm first
            signal.alarm(0)
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(300)  # 300 seconds (5 minutes) timeout for complex molecules
    except (AttributeError, OSError):
        # signal.SIGALRM not available (e.g., Windows) or signal setup failed
        # In this case, we rely on the maxIts limits in optimization functions
        pass
    
    try:
        # Create a new builder instance in this process
        builder = MolecularGraphBuilder(**builder_config)
        graph = builder.smiles_to_graph(smiles)
        
        if graph is not None:
            # Verify that the graph has valid 3D coordinates
            if hasattr(graph, 'pos') and graph.pos is not None:
                pos_norm = torch.norm(graph.pos, dim=1)
                if pos_norm.sum() > 1e-6:  # At least some non-zero coordinates
                    graph.y = torch.tensor([target], dtype=torch.float)
                    # Disable alarm before returning
                    if hasattr(signal, 'SIGALRM'):
                        try:
                            signal.alarm(0)
                        except:
                            pass
                    return (graph, target, smiles, None)
        
        # Return failure info
        failed_info = builder.failed_3d_generation[-1] if builder.failed_3d_generation else None
        if hasattr(signal, 'SIGALRM'):
            try:
                signal.alarm(0)
            except:
                pass
        return (None, target, smiles, failed_info)
        
    except TimeoutException:
        if hasattr(signal, 'SIGALRM'):
            try:
                signal.alarm(0)
            except:
                pass
        # Retry strategy: If high-quality generation times out, try "Lite Mode"
        # Lite Mode: Fewer conformers, but still with optimization. This ensures we don't lose the molecule.
        try:
            # Create a "Lite" config
            lite_config = builder_config.copy()
            lite_config['num_conformers'] = 3  # Generate 3 conformers (reduced from 10)
            lite_config['optimize_conformers'] = True  # Enable optimization as requested
            
            # Set a new timeout for Lite Mode (120 seconds = 2 minutes)
            # Lite Mode uses fewer conformers, so it should be faster, but still give it enough time
            if hasattr(signal, 'SIGALRM'):
                try:
                    signal.alarm(120)  # 120 seconds for Lite Mode retry
                except:
                    pass
            
            lite_builder = MolecularGraphBuilder(**lite_config)
            graph = lite_builder.smiles_to_graph(smiles)
            
            if graph is not None:
                if hasattr(graph, 'pos') and graph.pos is not None:
                    pos_norm = torch.norm(graph.pos, dim=1)
                    if pos_norm.sum() > 1e-6:
                        graph.y = torch.tensor([target], dtype=torch.float)
                        if hasattr(signal, 'SIGALRM'):
                            try:
                                signal.alarm(0)
                            except:
                                pass
                        # Mark this as a "fallback" success (optional: could log this)
                        return (graph, target, smiles, None)
            
            if hasattr(signal, 'SIGALRM'):
                try:
                    signal.alarm(0)
                except:
                    pass
            return (None, target, smiles, {'smiles': smiles, 'reason': 'Timeout: Failed even with Lite Mode'})
            
        except Exception as e_lite:
            if hasattr(signal, 'SIGALRM'):
                try:
                    signal.alarm(0)
                except:
                    pass
            return (None, target, smiles, {'smiles': smiles, 'reason': f'Timeout & Lite Mode Failed: {str(e_lite)}'})

    except Exception as e:
        if hasattr(signal, 'SIGALRM'):
            try:
                signal.alarm(0)
            except:
                pass
        return (None, target, smiles, {'smiles': smiles, 'reason': f'Exception: {str(e)}'})


class MolecularDataset:
    """Molecular Dataset Class"""
    
    def __init__(self, 
                 data_path: Optional[str] = None,
                 data: Optional[pd.DataFrame] = None,
                 target_column: str = 'target',
                 smiles_column: str = 'smiles',
                 graph_builder: Optional[MolecularGraphBuilder] = None):
        
        self.data_path = data_path
        self.target_column = target_column
        self.smiles_column = smiles_column
        self.graph_builder = graph_builder or MolecularGraphBuilder()
        
        # Load data
        if data is not None:
            # If data is directly provided, use the provided data
            self.data = data
        else:
            # Otherwise load from file
            self.data = self._load_data()
        self.graphs = []
        self.targets = []
    
    def _load_data(self) -> pd.DataFrame:
        """Load data"""
        if self.data_path is None:
            # If data_path is None, return empty DataFrame
            return pd.DataFrame()
        
        if self.data_path.endswith('.csv'):
            return pd.read_csv(self.data_path)
        elif self.data_path.endswith('.pkl'):
            with open(self.data_path, 'rb') as f:
                return pickle.load(f)
        else:
            raise ValueError(f"Unsupported file format: {self.data_path}")
    
    def _save_checkpoint(self, checkpoint_path: str, processed_count: int, valid_count: int, excluded_count: int):
        """Save checkpoint data with atomic write to prevent corruption"""
        try:
            checkpoint_data = {
                'graphs': self.graphs,
                'targets': self.targets,
                'processed_count': processed_count,
                'valid_count': valid_count,
                'excluded_count': excluded_count,
                'failed_3d': self.graph_builder.failed_3d_generation
            }
            # Use atomic write: write to temp file first, then rename
            # This prevents corruption if process is interrupted during write
            import tempfile
            temp_path = checkpoint_path + '.tmp'
            with open(temp_path, 'wb') as f:
                pickle.dump(checkpoint_data, f)
                f.flush()
                os.fsync(f.fileno())  # Force write to disk
            # Atomic rename (only works on same filesystem)
            os.replace(temp_path, checkpoint_path)
        except Exception as e:
            print(f"\n⚠️  Warning: Failed to save checkpoint: {e}", file=sys.stderr)
            # Clean up temp file if it exists
            temp_path = checkpoint_path + '.tmp'
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
    
    def process_graphs(self, max_samples: Optional[int] = None, checkpoint_interval: int = 200, checkpoint_path: Optional[str] = None, num_workers: int = 0, timeout_seconds: Optional[int] = None):
        """
        Process molecular graphs with checkpoint support and optional multiprocessing
        
        Args:
            max_samples: Maximum number of samples to process
            checkpoint_interval: Save checkpoint every N molecules (default: 200)
            checkpoint_path: Path to save checkpoint file (if None, auto-generate from data_path)
            num_workers: Number of parallel workers (0 = use all CPU cores, 1 = sequential processing)
            timeout_seconds: Timeout per molecule in seconds (default: 420 = 7 minutes)
                             Includes both initial processing (300s) and Lite Mode retry (120s)
        """
        # Print initial messages to stderr to avoid interfering with progress bar
        print("Processing molecular graphs...", file=sys.stderr, flush=True)
        print("⚠️  Note: Molecules that cannot generate 3D coordinates will be excluded from training", file=sys.stderr, flush=True)
        
        data_subset = self.data
        if max_samples:
            data_subset = self.data.head(max_samples)
        
        total_count = len(data_subset)
        valid_count = 0
        excluded_count = 0
        processed_count = 0
        start_idx = 0
        
        # Try to load checkpoint if exists
        if checkpoint_path is None and hasattr(self, 'data_path'):
            # Auto-generate checkpoint path
            checkpoint_dir = os.path.join(os.path.dirname(self.data_path), 'checkpoints')
            os.makedirs(checkpoint_dir, exist_ok=True)
            dataset_name = os.path.splitext(os.path.basename(self.data_path))[0]
            checkpoint_path = os.path.join(checkpoint_dir, f"{dataset_name}_checkpoint.pkl")
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            try:
                print(f"📂 Found checkpoint file: {checkpoint_path}", file=sys.stderr)
                print("   Attempting to resume from checkpoint...", file=sys.stderr)
                
                # Check for temp file (indicates incomplete write)
                temp_path = checkpoint_path + '.tmp'
                if os.path.exists(temp_path):
                    print(f"   ⚠️  Found incomplete checkpoint temp file, removing it...", file=sys.stderr)
                    try:
                        os.remove(temp_path)
                    except:
                        pass
                
                # Verify file is not empty
                if os.path.getsize(checkpoint_path) == 0:
                    print(f"   ⚠️  Checkpoint file is empty, starting from beginning...", file=sys.stderr)
                    raise ValueError("Checkpoint file is empty")
                
                with open(checkpoint_path, 'rb') as f:
                    checkpoint_data = pickle.load(f)
                    
                    # Verify checkpoint data integrity
                    if not isinstance(checkpoint_data, dict):
                        raise ValueError("Invalid checkpoint format: not a dictionary")
                    
                    self.graphs = checkpoint_data.get('graphs', [])
                    self.targets = checkpoint_data.get('targets', [])
                    start_idx = checkpoint_data.get('processed_count', 0)
                    valid_count = checkpoint_data.get('valid_count', len(self.graphs))
                    excluded_count = checkpoint_data.get('excluded_count', 0)
                    processed_count = start_idx
                    
                    # Verify data consistency
                    if len(self.graphs) != len(self.targets):
                        print(f"   ⚠️  Warning: Inconsistent checkpoint data (graphs: {len(self.graphs)}, targets: {len(self.targets)})", file=sys.stderr)
                        # Use the smaller count to be safe
                        min_len = min(len(self.graphs), len(self.targets))
                        self.graphs = self.graphs[:min_len]
                        self.targets = self.targets[:min_len]
                    
                    # Restore failed_3d_generation records
                    if 'failed_3d' in checkpoint_data:
                        self.graph_builder.failed_3d_generation = checkpoint_data['failed_3d']
                
                print(f"   ✅ Resumed from checkpoint: {processed_count}/{total_count} molecules already processed", file=sys.stderr)
                print(f"   Valid graphs: {valid_count}, Excluded: {excluded_count}", file=sys.stderr)
            except (EOFError, pickle.UnpicklingError, ValueError) as e:
                print(f"   ⚠️  Failed to load checkpoint: {e}", file=sys.stderr)
                print(f"   ⚠️  Checkpoint file may be corrupted or incomplete", file=sys.stderr)
                print("   Starting from beginning...", file=sys.stderr)
                # Optionally backup corrupted checkpoint for inspection
                if os.path.exists(checkpoint_path):
                    backup_path = checkpoint_path + '.corrupted'
                    try:
                        import shutil
                        shutil.copy2(checkpoint_path, backup_path)
                        print(f"   📦 Corrupted checkpoint backed up to: {backup_path}", file=sys.stderr)
                    except:
                        pass
                self.graphs = []
                self.targets = []
                start_idx = 0
                processed_count = 0
                valid_count = 0
                excluded_count = 0
            except Exception as e:
                print(f"   ⚠️  Failed to load checkpoint: {e}", file=sys.stderr)
                print("   Starting from beginning...", file=sys.stderr)
                self.graphs = []
                self.targets = []
                start_idx = 0
                processed_count = 0
                valid_count = 0
                excluded_count = 0
        
        # Determine number of workers
        if num_workers == 0:
            num_workers = cpu_count()
        elif num_workers < 0:
            num_workers = max(1, cpu_count() + num_workers)  # Allow negative to reduce from max
        
        # Prepare data for processing (need to know remaining task count first)
        data_to_process = data_subset.iloc[start_idx:]
        remaining_tasks = len(data_to_process)
        
        # Adjust workers if remaining tasks are fewer than workers
        # This prevents idle workers when processing near the end of dataset
        # However, we maintain a minimum of 2 workers to ensure parallel processing
        # Only reduce workers if remaining tasks are very few (< 5), and never reduce to 1
        if remaining_tasks > 0 and remaining_tasks < num_workers:
            # Keep at least 2 workers for parallel processing, unless there's only 1 task left
            min_workers = 2 if remaining_tasks > 1 else 1
            new_workers = max(min_workers, min(remaining_tasks, num_workers))
            if new_workers < num_workers:
                print(f"   ⚠️  Remaining tasks ({remaining_tasks}) < workers ({num_workers}), reducing workers to {new_workers} (min: {min_workers})", file=sys.stderr)
                num_workers = new_workers
        
        # Prepare builder configuration for multiprocessing
        # IMPORTANT: Keep this in sync with MolecularGraphBuilder.__init__ so that
        # child-process builders behave identically to the main-process builder.
        builder_config = {
            'use_atomic_number': self.graph_builder.use_atomic_number,
            'use_hybridization': self.graph_builder.use_hybridization,
            'use_formal_charge': self.graph_builder.use_formal_charge,
            'use_aromatic': self.graph_builder.use_aromatic,
            'use_chirality': self.graph_builder.use_chirality,
            'use_hydrogen_bonds': self.graph_builder.use_hydrogen_bonds,
            'use_bond_type': self.graph_builder.use_bond_type,
            'use_bond_stereo': self.graph_builder.use_bond_stereo,
            'num_conformers': self.graph_builder.num_conformers,
            'optimize_conformers': self.graph_builder.optimize_conformers,
            'num_conformers_to_keep': self.graph_builder.num_conformers_to_keep,
            'num_threads': self.graph_builder.num_threads,
            'add_hydrogens': self.graph_builder.add_hydrogens,
            'prune_rms_thresh': self.graph_builder.prune_rms_thresh,
            'use_fingerprint': self.graph_builder.use_fingerprint,
            'fingerprint_radius': self.graph_builder.fingerprint_radius,
            'fingerprint_bits': self.graph_builder.fingerprint_bits,
            'use_descriptor': self.graph_builder.use_descriptor,
            'descriptor_dim': self.graph_builder.descriptor_dim,
        }
        
        # Prepare task arguments (data_to_process already prepared above)
        process_args = [
            (row[self.smiles_column], row[self.target_column], builder_config)
            for _, row in data_to_process.iterrows()
        ]
        
        # Process molecules (sequential or parallel)
        if num_workers == 1:
            # Sequential processing (original method)
            print(f"   Using sequential processing...", file=sys.stderr, flush=True)
            print(f"   Total molecules to process: {total_count}, Starting from: {processed_count}", file=sys.stderr, flush=True)
            print(f"   Heartbeat will appear every 30 seconds after processing starts...", file=sys.stderr, flush=True)
            # Check if stderr is a TTY (terminal)
            is_tty = os.isatty(sys.stderr.fileno()) if hasattr(sys.stderr, 'fileno') else False
            iterator = tqdm(
                enumerate(data_to_process.iterrows(), start=start_idx),
                total=total_count - start_idx,
                initial=start_idx,
                desc="Processing molecules",
                unit="mol",
                ncols=120,
                mininterval=0.1,  # Update more frequently for smoother progress
                maxinterval=1.0,  # But don't update too often
                file=sys.stderr,  # Write to stderr so it displays even when stdout is piped
                dynamic_ncols=True,  # Adjust width based on terminal
                disable=not is_tty,  # Disable only if not a TTY
                leave=True,  # Keep progress bar after completion
                smoothing=0.1  # Smooth rate calculations
            )
            
            for idx, (_, row) in iterator:
                processed_count += 1
                smiles = row[self.smiles_column]
                target = row[self.target_column]
                
                graph = self.graph_builder.smiles_to_graph(smiles)
                if graph is not None:
                    if hasattr(graph, 'pos') and graph.pos is not None:
                        pos_norm = torch.norm(graph.pos, dim=1)
                        if pos_norm.sum() > 1e-6:
                            graph.y = torch.tensor([target], dtype=torch.float)
                            self.graphs.append(graph)
                            self.targets.append(target)
                            valid_count += 1
                        else:
                            excluded_count += 1
                    else:
                        excluded_count += 1
                else:
                    excluded_count += 1
                
                # Save checkpoint periodically
                if checkpoint_path and processed_count % checkpoint_interval == 0:
                    self._save_checkpoint(checkpoint_path, processed_count, valid_count, excluded_count)
                
                # Update progress bar
                if HAS_TQDM:
                    iterator.set_postfix({
                        'valid': valid_count,
                        'excluded': excluded_count,
                        'rate': f'{valid_count/processed_count*100:.1f}%'
                    })
        else:
            # Parallel processing using multiprocessing
            print(f"   Using parallel processing with {num_workers} workers...", file=sys.stderr, flush=True)
            print(f"   Total molecules to process: {total_count}, Starting from: {processed_count}", file=sys.stderr, flush=True)
            print(f"   Heartbeat will appear every 30 seconds after processing starts...", file=sys.stderr, flush=True)
            
            with Pool(processes=num_workers) as pool:
                # Create overall progress bar
                # Use file=sys.stderr to ensure progress bar displays even when stdout is piped
                if HAS_TQDM:
                    # Check if stderr is a TTY (terminal)
                    is_tty = os.isatty(sys.stderr.fileno()) if hasattr(sys.stderr, 'fileno') else False
                    overall_pbar = tqdm(
                        total=total_count,
                        initial=processed_count,
                        desc="Processing molecules",
                        unit="mol",
                        ncols=120,
                        mininterval=0.1,  # Update more frequently for smoother progress
                        maxinterval=1.0,  # But don't update too often
                        file=sys.stderr,  # Write to stderr so it displays even when stdout is piped
                        dynamic_ncols=True,  # Adjust width based on terminal
                        disable=not is_tty,  # Disable only if not a TTY
                        leave=True,  # Keep progress bar after completion
                        smoothing=0.1  # Smooth rate calculations
                    )
                else:
                    overall_pbar = None
                
                # Optimize chunksize for better CPU utilization
                # Use small chunksize to ensure frequent progress updates
                # Formula: ensure each worker gets enough tasks to stay busy, but not too many to block progress bar
                if len(process_args) > 0:
                    # Force small chunksize (1) to ensure responsiveness
                    # This might slightly increase overhead but ensures progress bar and heartbeats work correctly
                    optimal_chunksize = 1
                else:
                    optimal_chunksize = 1
                
                print(f"   Workers: {num_workers}, Remaining tasks: {len(process_args)}, Chunksize: {optimal_chunksize}", file=sys.stderr, flush=True)
                
                # Use imap to process all tasks in parallel
                # imap returns results as they complete, allowing real-time progress tracking
                # All tasks are submitted to the pool immediately, keeping all workers busy
                import time
                import threading
                
                start_time = time.time()
                last_checkpoint_time = time.time()
                
                # Shared variables for heartbeat thread
                heartbeat_stop_event = threading.Event()
                
                def heartbeat_monitor():
                    """Background thread to print heartbeats"""
                    last_heartbeat_time = time.time()
                    heartbeat_interval = 30  # 30 seconds
                    
                    while not heartbeat_stop_event.is_set():
                        time.sleep(1)  # Check every second
                        current_time = time.time()
                        
                        if current_time - last_heartbeat_time >= heartbeat_interval:
                            elapsed_total = current_time - start_time
                            # Calculate processed count safely
                            current_processed = processed_count
                            processed_since_start = current_processed - (start_idx if 'start_idx' in locals() else 0)
                            
                            rate = processed_since_start / elapsed_total if elapsed_total > 0 else 0
                            remaining = total_count - current_processed
                            
                            eta_seconds = remaining / rate if rate > 0 else 0
                            if eta_seconds > 3600:
                                eta_str = f"{int(eta_seconds//3600)}h{int((eta_seconds%3600)//60)}m"
                            elif eta_seconds > 60:
                                eta_str = f"{int(eta_seconds//60)}m{int(eta_seconds%60)}s"
                            else:
                                eta_str = f"{int(eta_seconds)}s" if eta_seconds > 0 else "?"
                            
                            rate_str = f"{rate:.2f} mol/s" if rate > 0 else "Calculating..."
                            
                            # Print heartbeat
                            # Use \r to clear current line if needed, but simple print is safer to avoid messing up tqdm
                            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Heartbeat: Processed {current_processed}/{total_count} ({current_processed*100//total_count if total_count > 0 else 0}%), Valid: {valid_count}, Excluded: {excluded_count}, Rate: {rate_str}, ETA: {eta_str}", file=sys.stderr, flush=True)
                            
                            last_heartbeat_time = current_time

                # Start heartbeat thread
                monitor_thread = threading.Thread(target=heartbeat_monitor)
                monitor_thread.daemon = True
                monitor_thread.start()
                
                # Output initial heartbeat immediately to show process has started
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Heartbeat: Starting processing... Total: {total_count}, Remaining: {len(process_args)}, Workers: {num_workers}", file=sys.stderr, flush=True)
                
                # Use apply_async with timeout to prevent hanging on individual molecules
                # This provides better control over timeouts compared to imap
                # Timeout includes both initial processing and Lite Mode retry
                # Default: 420 seconds (7 minutes) to allow complex molecules enough time:
                # - Initial processing: up to 300 seconds (5 minutes)
                # - Lite Mode retry: up to 120 seconds (2 minutes)
                if timeout_seconds is None:
                    timeout_seconds = 420  # Default: 420 seconds (7 minutes) timeout per molecule
                
                try:
                    # Submit all tasks
                    futures = []
                    for args in process_args:
                        future = pool.apply_async(_process_single_molecule, (args,))
                        futures.append(future)
                    
                    # Process results as they complete, with timeout handling
                    for future in futures:
                        try:
                            # Get result with timeout
                            graph, target, smiles, failed_info = future.get(timeout=timeout_seconds)
                            processed_count += 1
                            
                            if graph is not None:
                                self.graphs.append(graph)
                                self.targets.append(target)
                                valid_count += 1
                            else:
                                excluded_count += 1
                                if failed_info:
                                    self.graph_builder.failed_3d_generation.append(failed_info)
                            
                            # Update overall progress bar (force refresh even if no new progress)
                            if overall_pbar is not None:
                                overall_pbar.update(1)
                                overall_pbar.set_postfix({
                                    'valid': valid_count,
                                    'excluded': excluded_count,
                                    'rate': f'{valid_count/processed_count*100:.1f}%'
                                })
                                # Force refresh the progress bar
                                overall_pbar.refresh()
                            
                            # Save checkpoint periodically based on processed count
                            if checkpoint_path and processed_count % checkpoint_interval == 0:
                                self._save_checkpoint(checkpoint_path, processed_count, valid_count, excluded_count)
                                last_checkpoint_time = time.time()
                                print(f"[{datetime.now().strftime('%H:%M:%S')}] Checkpoint saved: {processed_count}/{total_count}", file=sys.stderr, flush=True)
                                
                        except MPTimeoutError:
                            # Handle timeout for individual molecules
                            processed_count += 1
                            excluded_count += 1
                            
                            # Try to get the smiles from the task arguments
                            failed_smiles = "unknown"
                            try:
                                # Find which task this future corresponds to
                                task_idx = futures.index(future)
                                if task_idx < len(process_args):
                                    failed_smiles = process_args[task_idx][0]  # First element is smiles
                            except:
                                pass
                            
                            self.graph_builder.failed_3d_generation.append({
                                'smiles': failed_smiles,
                                'reason': f'Processing timeout after {timeout_seconds} seconds'
                            })
                            
                            # Update progress bar
                            if overall_pbar is not None:
                                overall_pbar.update(1)
                                overall_pbar.set_postfix({
                                    'valid': valid_count,
                                    'excluded': excluded_count,
                                    'rate': f'{valid_count/processed_count*100:.1f}%'
                                })
                                overall_pbar.refresh()
                            
                            # Log the timeout
                            print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚠️  Warning: Molecule processing timeout after {timeout_seconds}s: {failed_smiles}", file=sys.stderr, flush=True)
                            
                            # Save checkpoint even on timeout
                            if checkpoint_path and processed_count % checkpoint_interval == 0:
                                self._save_checkpoint(checkpoint_path, processed_count, valid_count, excluded_count)
                                last_checkpoint_time = time.time()
                                
                        except Exception as e:
                            # Handle other errors for individual molecules
                            processed_count += 1
                            excluded_count += 1
                            
                            # Try to get the smiles from the task arguments
                            failed_smiles = "unknown"
                            try:
                                task_idx = futures.index(future)
                                if task_idx < len(process_args):
                                    failed_smiles = process_args[task_idx][0]
                            except:
                                pass
                            
                            self.graph_builder.failed_3d_generation.append({
                                'smiles': failed_smiles,
                                'reason': f'Processing error: {str(e)}'
                            })
                            
                            # Update progress bar
                            if overall_pbar is not None:
                                overall_pbar.update(1)
                                overall_pbar.set_postfix({
                                    'valid': valid_count,
                                    'excluded': excluded_count,
                                    'rate': f'{valid_count/processed_count*100:.1f}%'
                                })
                                overall_pbar.refresh()
                            
                            # Log the error
                            print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚠️  Warning: Failed to process molecule: {str(e)}", file=sys.stderr, flush=True)
                            
                            # Save checkpoint even on errors
                            if checkpoint_path and processed_count % checkpoint_interval == 0:
                                self._save_checkpoint(checkpoint_path, processed_count, valid_count, excluded_count)
                                last_checkpoint_time = time.time()
                finally:
                    # Stop heartbeat thread
                    heartbeat_stop_event.set()
                    monitor_thread.join(timeout=1.0)
                
                # Close overall progress bar
                if overall_pbar is not None:
                    overall_pbar.close()
        
        # Final checkpoint save
        if checkpoint_path:
            self._save_checkpoint(checkpoint_path, processed_count, valid_count, excluded_count)
        
        # Remove checkpoint file after successful completion
        if checkpoint_path and os.path.exists(checkpoint_path):
            try:
                os.remove(checkpoint_path)
                print(f"✅ Checkpoint file removed after successful completion", file=sys.stderr)
            except:
                pass
        
        # Print final summary after progress bar is closed
        # Use newline to ensure it appears after the progress bar
        print(f"\n✅ Processing completed:", file=sys.stderr)
        print(f"   Total molecules: {total_count}", file=sys.stderr)
        print(f"   Valid graphs with 3D coordinates: {valid_count}", file=sys.stderr)
        print(f"   Excluded (no 3D coordinates): {excluded_count}", file=sys.stderr)
        print(f"   Exclusion rate: {excluded_count/total_count*100:.2f}%", file=sys.stderr)
        
        if excluded_count > 0:
            failed_3d = self.graph_builder.failed_3d_generation
            if failed_3d:
                print(f"   ⚠️  {len(failed_3d)} molecules failed 3D coordinate generation (see failed_3d.json for details)", file=sys.stderr)
        
        return self.graphs
    
    def get_dataloader(self, 
                      batch_size: int = 32,
                      shuffle: bool = True,
                      num_workers: int = 4) -> DataLoader:
        """Get data loader"""
        if not self.graphs:
            self.process_graphs()
        
        return DataLoader(
            self.graphs,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers
        )
    
    def save_processed_data(self, save_path: str, smiles_list: Optional[List[str]] = None):
        """Save processed data"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        save_data = {
            'graphs': self.graphs,
            'targets': self.targets,
            'node_feature_dim': self.graph_builder.node_feature_dim,
            'edge_feature_dim': self.graph_builder.edge_feature_dim,
            'use_fingerprint': self.graph_builder.use_fingerprint,
            'fingerprint_bits': self.graph_builder.fingerprint_bits
        }
        # Save SMILES list (for scaffold splitting)
        if smiles_list is not None:
            save_data['smiles_list'] = smiles_list
        elif hasattr(self, 'data') and self.smiles_column in self.data.columns:
            # If not provided, try to extract from original data
            save_data['smiles_list'] = self.data[self.smiles_column].tolist()[:len(self.graphs)]
        
        with open(save_path, 'wb') as f:
            pickle.dump(save_data, f)
        print(f"Saved processed data to: {save_path}")
    
    def load_processed_data(self, load_path: str):
        """Load processed data"""
        with open(load_path, 'rb') as f:
            data = pickle.load(f)
            graphs = data['graphs']
            targets = data['targets']
            self.graph_builder.node_feature_dim = data['node_feature_dim']
            self.graph_builder.edge_feature_dim = data['edge_feature_dim']
            
            # Load fingerprint settings if available
            if 'use_fingerprint' in data:
                self.graph_builder.use_fingerprint = data['use_fingerprint']
                self.graph_builder.fingerprint_bits = data['fingerprint_bits']
            
            # Return SMILES list (if exists)
            smiles_list = data.get('smiles_list', None)
        
        # Filter out graphs without valid 3D coordinates
        # This ensures that only molecules with valid 3D coordinates are used for training
        valid_graphs = []
        valid_targets = []
        excluded_count = 0
        
        for graph, target in zip(graphs, targets):
            if hasattr(graph, 'pos') and graph.pos is not None:
                # Check if pos contains valid (non-zero) coordinates
                pos_norm = torch.norm(graph.pos, dim=1)
                if pos_norm.sum() > 1e-6:  # At least some non-zero coordinates
                    valid_graphs.append(graph)
                    valid_targets.append(target)
                else:
                    excluded_count += 1
            else:
                excluded_count += 1
        
        self.graphs = valid_graphs
        self.targets = valid_targets
        
        if excluded_count > 0:
            print(f"⚠️  Excluded {excluded_count} graphs without valid 3D coordinates from loaded data")
            print(f"   Loaded {len(valid_graphs)} valid graphs with 3D coordinates")
        
        print(f"✅ Loaded processed data from: {load_path}")
        print(f"   Total valid graphs: {len(valid_graphs)}")
        
        return smiles_list


class DataPreprocessor:
    """Data Preprocessor"""
    
    @staticmethod
    def normalize_targets(targets: List[float]) -> Tuple[List[float], float, float]:
        """Normalize target values"""
        targets = np.array(targets)
        mean = np.mean(targets)
        std = np.std(targets)
        normalized = (targets - mean) / std
        return normalized.tolist(), mean, std
    
    @staticmethod
    def denormalize_targets(normalized_targets: List[float], mean: float, std: float) -> List[float]:
        """Denormalize target values"""
        return (np.array(normalized_targets) * std + mean).tolist()
    
    @staticmethod
    def generate_scaffold(smiles: str, include_chirality: bool = False) -> str:
        """Generate molecular scaffold"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            scaffold = MurckoScaffold.GetScaffoldForMol(mol, includeChirality=include_chirality)
            return Chem.MolToSmiles(scaffold)
        except:
            return None
    
    @staticmethod
    def split_dataset(graphs: List[Data], 
                     targets: List[float],
                     smiles_list: Optional[List[str]] = None,
                     train_ratio: float = 0.8,
                     val_ratio: float = 0.1,
                     random_seed: int = 42,
                     split_method: str = 'random') -> Tuple[List[Data], List[Data], List[Data]]:
        """
        Split dataset
        Args:
            graphs: List of graph data
            targets: List of target values
            smiles_list: List of SMILES strings (for scaffold splitting)
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            random_seed: Random seed
            split_method: Splitting method ('random' or 'scaffold')
        """
        if split_method == 'scaffold' and smiles_list is not None:
            return DataPreprocessor.scaffold_split(
                graphs, targets, smiles_list, train_ratio, val_ratio, random_seed
            )
        else:
            # Random splitting
            np.random.seed(random_seed)
            indices = np.random.permutation(len(graphs))
            
            train_size = int(len(graphs) * train_ratio)
            val_size = int(len(graphs) * val_ratio)
            
            train_indices = indices[:train_size]
            val_indices = indices[train_size:train_size + val_size]
            test_indices = indices[train_size + val_size:]
            
            train_graphs = [graphs[i] for i in train_indices]
            val_graphs = [graphs[i] for i in val_indices]
            test_graphs = [graphs[i] for i in test_indices]
            
            return train_graphs, val_graphs, test_graphs
    
    @staticmethod
    def scaffold_split(graphs: List[Data],
                      targets: List[float],
                      smiles_list: List[str],
                      train_ratio: float = 0.8,
                      val_ratio: float = 0.1,
                      random_seed: int = 42) -> Tuple[List[Data], List[Data], List[Data]]:
        """
        Scaffold-based data splitting
        """
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
        
        # Group by scaffold
        scaffolds = defaultdict(list)
        for i, smiles in enumerate(smiles_list):
            scaffold = DataPreprocessor.generate_scaffold(smiles)
            if scaffold is not None:
                scaffolds[scaffold].append(i)
            else:
                # If scaffold cannot be generated, use SMILES itself as key
                scaffolds[smiles].append(i)
        
        # Randomly shuffle scaffold groups
        scaffold_sets = list(scaffolds.values())
        random.shuffle(scaffold_sets)
        
        # Calculate set sizes
        n_total = len(graphs)
        n_test = int(n_total * (1 - train_ratio - val_ratio))
        n_valid = int(n_total * val_ratio)
        
        # Assign scaffold groups to different sets
        test_indices = []
        valid_indices = []
        train_indices = []
        
        for scaffold_set in scaffold_sets:
            if len(test_indices) + len(scaffold_set) <= n_test:
                test_indices.extend(scaffold_set)
            elif len(valid_indices) + len(scaffold_set) <= n_valid:
                valid_indices.extend(scaffold_set)
            else:
                train_indices.extend(scaffold_set)
        
        # If some sets are too small, supplement from other sets
        if len(test_indices) < n_test and len(train_indices) > 0:
            needed = n_test - len(test_indices)
            test_indices.extend(train_indices[:needed])
            train_indices = train_indices[needed:]
        
        if len(valid_indices) < n_valid and len(train_indices) > 0:
            needed = n_valid - len(valid_indices)
            valid_indices.extend(train_indices[:needed])
            train_indices = train_indices[needed:]
        
        train_graphs = [graphs[i] for i in train_indices]
        val_graphs = [graphs[i] for i in valid_indices]
        test_graphs = [graphs[i] for i in test_indices]
        
        return train_graphs, val_graphs, test_graphs


def create_sample_dataset():
    """Create sample dataset"""
    # Some example SMILES and corresponding molecular properties
    sample_data = {
        'smiles': [
            'CCO',  # Ethanol
            'CC(=O)O',  # Acetic acid
            'c1ccccc1',  # Benzene
            'CC(C)O',  # Isopropanol
            'CCN(CC)CC',  # Triethylamine
        ],
        'target': [0.5, 1.2, 2.1, 0.8, 1.5]  # Hypothetical molecular property values
    }
    
    df = pd.DataFrame(sample_data)
    return df

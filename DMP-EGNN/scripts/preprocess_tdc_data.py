#!/usr/bin/env python3
"""
TDC Dataset Preprocessing Script
Uses prepare_tdc_dataset.py to preprocess TDC datasets
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tdc.benchmark_group import admet_group
from utils.prepare_tdc_dataset import load_tdc_dataset, load_tdc_dataset_cv


def list_tdc_datasets(data_path: str):
    """List all available TDC datasets"""
    try:
        group = admet_group(path=data_path)
        # Get all available datasets from the group
        # TDC groups have a 'names' attribute or we can try to get datasets
        datasets = []
        
        # Try to get datasets by checking the directory structure
        admet_dir = os.path.join(data_path, "admet_group")
        if os.path.exists(admet_dir):
            for item in os.listdir(admet_dir):
                item_path = os.path.join(admet_dir, item)
                if os.path.isdir(item_path):
                    # Check if it has test.csv and train_val.csv
                    test_file = os.path.join(item_path, "test.csv")
                    train_val_file = os.path.join(item_path, "train_val.csv")
                    if os.path.exists(test_file) and os.path.exists(train_val_file):
                        datasets.append(item)
        
        return sorted(datasets)
    except Exception as e:
        print(f"Error listing TDC datasets: {e}")
        return []


def check_processed(dataset_name: str, seed: int, processed_dir: str, use_cv: bool = False, outer_fold_idx: Optional[int] = None):
    """Check if dataset has already been processed"""
    if use_cv and outer_fold_idx is not None:
        # CV mode: Check train/valid/test for all inner folds
        cache_dir = os.path.join(processed_dir, dataset_name, f"fold{outer_fold_idx + 1}")
        # Check if data for at least one inner fold exists (usually only need to check one)
        # During actual use, will dynamically process based on inner_fold_idx
        inner_fold_idx = 0  # Check first inner fold as representative
        split_tag = f"outer{outer_fold_idx}_inner{inner_fold_idx}"
        train_file = os.path.join(cache_dir, f"{split_tag}_train.pt")
        valid_file = os.path.join(cache_dir, f"{split_tag}_valid.pt")
        test_file = os.path.join(cache_dir, f"{split_tag}_test.pt")
    else:
        # Standard mode
        cache_dir = os.path.join(processed_dir, dataset_name, f"seed{seed}")
        train_file = os.path.join(cache_dir, "train.pt")
        valid_file = os.path.join(cache_dir, "valid.pt")
        test_file = os.path.join(cache_dir, "test.pt")
    
    if all(os.path.exists(f) and os.path.getsize(f) > 1024 for f in [train_file, valid_file, test_file]):
        return True
    return False


def preprocess_tdc_dataset(
    data_name: str,
    data_path: str,
    seed: int = 1,
    processed_dir: str = "data/processed_tdc_data",
    num_conformers: int = 10,
    optimize_conformers: bool = True,
    add_hydrogens: bool = True,
    use_fingerprint: bool = False,
    fingerprint_bits: int = 2048,
    descriptor_dim: Optional[int] = None,
    use_cv: bool = False,
    outer_fold_idx: Optional[int] = None,
    inner_fold_idx: Optional[int] = None,  # None means use fusion_model's formula to calculate
    outer_folds: int = 5,
    inner_folds: int = 4,
    num_workers: int = 1
):
    """Preprocess a single TDC dataset (standard mode or CV mode)"""
    try:
        mode_str = "CV" if use_cv else "Standard"
        print(f"\n{'='*60}")
        print(f"Processing TDC dataset ({mode_str} mode): {data_name}")
        
        # CV mode: Calculate inner_fold_idx (consistent with fusion_model)
        if use_cv and outer_fold_idx is not None:
            # Consistent with fusion_model: inner_fold_idx = (outer_fold_idx + 1) % 4
            if inner_fold_idx is None:
                inner_fold_idx = (outer_fold_idx + 1) % inner_folds
            print(f"   Outer fold: {outer_fold_idx + 1}/{outer_folds}, Inner fold: {inner_fold_idx + 1}/{inner_folds}")
        print(f"{'='*60}")
        
        # Check if already processed
        if check_processed(data_name, seed, processed_dir, use_cv, outer_fold_idx):
            print(f"✅ Dataset {data_name} already processed, skipping...")
            return True
        
        # Load and process dataset
        if use_cv:
            # CV mode: Set seed before calling (consistent with fusion_model)
            from utils.prepare_tdc_dataset import set_seed
            set_seed(seed)
            
            train_graphs, valid_graphs, test_graphs = load_tdc_dataset_cv(
                data_name=data_name,
                data_path=data_path,
                seed=seed,
                outer_fold_idx=outer_fold_idx or 0,
                inner_fold_idx=inner_fold_idx or ((outer_fold_idx or 0) + 1) % inner_folds,
                outer_folds=outer_folds,
                inner_folds=inner_folds,
                use_fingerprint=use_fingerprint,
                descriptor_dim=descriptor_dim,
                fingerprint_bits=fingerprint_bits,
                num_conformers=num_conformers,
                optimize_conformers=optimize_conformers,
                add_hydrogens=add_hydrogens,
                num_workers=num_workers
            )
        else:
            train_graphs, valid_graphs, test_graphs = load_tdc_dataset(
                data_name=data_name,
                data_path=data_path,
                seed=seed,
                use_fingerprint=use_fingerprint,
                descriptor_dim=descriptor_dim,
                fingerprint_bits=fingerprint_bits,
                num_conformers=num_conformers,
                optimize_conformers=optimize_conformers,
                add_hydrogens=add_hydrogens,
                num_workers=num_workers
            )
        
        print(f"\n✅ Successfully processed {data_name} ({mode_str} mode)")
        print(f"   Train: {len(train_graphs)} graphs")
        print(f"   Valid: {len(valid_graphs)} graphs")
        print(f"   Test: {len(test_graphs)} graphs")
        
        return True
        
    except Exception as e:
        print(f"❌ Error processing {data_name}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Preprocess TDC datasets")
    parser.add_argument("--data_name", type=str, help="TDC dataset name (e.g., 'caco2_wang', 'ames')")
    parser.add_argument("--data_path", type=str, default="data/data_tdc", 
                       help="Path to TDC data directory")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for data splitting")
    parser.add_argument("--processed_dir", type=str, default="data/processed_tdc_data",
                       help="Directory to save processed data")
    parser.add_argument("--num_conformers", type=int, default=10,
                       help="Number of conformers to generate")
    parser.add_argument("--optimize_conformers", action="store_true", default=True,
                       help="Optimize conformers")
    parser.add_argument("--no_optimize_conformers", dest="optimize_conformers", action="store_false",
                       help="Don't optimize conformers")
    parser.add_argument("--add_hydrogens", action="store_true", default=True,
                       help="Add hydrogens to molecules")
    parser.add_argument("--no_hydrogens", dest="add_hydrogens", action="store_false",
                       help="Don't add hydrogens")
    parser.add_argument("--use_fingerprint", action="store_true", default=False,
                       help="Use molecular fingerprints")
    parser.add_argument("--fingerprint_bits", type=int, default=2048,
                       help="Number of fingerprint bits")
    parser.add_argument("--descriptor_dim", type=int, default=None,
                       help="Descriptor dimension (None for all available)")
    parser.add_argument("--use_cv", action="store_true", default=False,
                       help="Use cross-validation mode")
    parser.add_argument("--outer_fold_idx", type=int, default=None,
                       help="Outer fold index for CV mode (0-4)")
    parser.add_argument("--inner_fold_idx", type=int, default=None,
                       help="Inner fold index for CV mode (0-3). If None, uses fusion_model formula: (outer_fold_idx + 1) %% inner_folds")
    parser.add_argument("--outer_folds", type=int, default=5,
                       help="Number of outer folds for CV mode")
    parser.add_argument("--inner_folds", type=int, default=4,
                       help="Number of inner folds for CV mode")
    parser.add_argument("--num_workers", type=int, default=1,
                       help="Number of parallel workers for processing (1=sequential, >1=parallel, 0=use all CPU cores)")
    parser.add_argument("--list_datasets", action="store_true",
                       help="List all available TDC datasets")
    
    args = parser.parse_args()
    
    # Validate CV arguments
    if args.use_cv:
        if args.outer_fold_idx is None:
            print("Error: --outer_fold_idx is required when using --use_cv")
            return 1
        if not (0 <= args.outer_fold_idx < args.outer_folds):
            print(f"Error: --outer_fold_idx must be between 0 and {args.outer_folds - 1}")
            return 1
        # inner_fold_idx can be None (will use fusion_model formula)
        if args.inner_fold_idx is not None and not (0 <= args.inner_fold_idx < args.inner_folds):
            print(f"Error: --inner_fold_idx must be between 0 and {args.inner_folds - 1}")
            return 1
    
    # List datasets if requested
    if args.list_datasets:
        datasets = list_tdc_datasets(args.data_path)
        print(f"\nAvailable TDC datasets ({len(datasets)}):")
        for i, dataset in enumerate(datasets, 1):
            print(f"  {i}. {dataset}")
        return
    
    # Process dataset
    if not args.data_name:
        print("Error: --data_name is required (or use --list_datasets to see available datasets)")
        return 1
    
    success = preprocess_tdc_dataset(
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
        num_workers=args.num_workers
    )
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())


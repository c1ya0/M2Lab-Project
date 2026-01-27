#!/usr/bin/env python3
"""
Extract best hyperparameters from existing best_params.json files
Read checkpoints/optuna_mod/<dataset>/seed<seed>/best_trial_models/best_params.json
and merge into a single JSON file
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

# Add project path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def extract_best_params_from_files(dataset_name, base_dir="checkpoints/optuna_mod"):
    """
    Extract best hyperparameters from existing best_params.json files
    
    Supports two directory structures:
    1. Per-seed structure: checkpoints/optuna_mod/<dataset>/seed{seed}/best_trial_models/best_params.json
    2. Fusion Model Logic structure: checkpoints/optuna_mod_new/<dataset>/best_trial_models/best_params.json
       (single file shared by all seeds)
    
    Args:
        dataset_name: Dataset name
        base_dir: Base directory (default: checkpoints/optuna_mod)
    
    Returns:
        dict: Dictionary containing best hyperparameters for each seed, returns None if not found
    """
    # Handle relative and absolute paths
    if os.path.isabs(base_dir):
        base_path = Path(base_dir)
    else:
        # Relative path: start from project root
        # Script is located at DMP-EGNN/scripts/, so project root is two levels up
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        base_path = Path(project_root) / base_dir
    dataset_path = base_path / dataset_name
    
    if not dataset_path.exists():
        print(f"⚠️  Dataset directory not found: {dataset_path}")
        return None
    
    all_seed_results = {}
    
    # Check for Fusion Model Logic structure first (single best_params.json for all seeds)
    # Path: checkpoints/optuna_mod_new/<dataset>/best_trial_models/best_params.json
    fusion_model_file = dataset_path / "best_trial_models" / "best_params.json"
    
    if fusion_model_file.exists():
        # Fusion Model Logic: single file shared by all seeds
        try:
            with open(fusion_model_file, 'r', encoding='utf-8') as f:
                best_params = json.load(f)
            
            # Create result for all seeds (same hyperparameters)
            for seed_num in range(1, 6):
                seed_result = {
                    "dataset": dataset_name,
                    "seed": seed_num,
                    "best_params": best_params,
                    "extracted_at": datetime.now().isoformat(),
                    "source_file": str(fusion_model_file.relative_to(base_path)),
                    "mode": "fusion_model_logic"  # Mark as Fusion Model Logic
                }
                all_seed_results[f"seed{seed_num}"] = seed_result
            
            print(f"✅ Successfully extracted best hyperparameters for {dataset_name} (Fusion Model Logic: shared by all 5 seeds)")
            print(f"   Source: {fusion_model_file}")
            print(f"   Parameters: {len(best_params)}")
            
        except Exception as e:
            print(f"⚠️  Error reading {fusion_model_file}: {e}")
            return None
    else:
        # Per-seed structure: checkpoints/optuna_mod/<dataset>/seed{seed}/best_trial_models/best_params.json
        for seed_num in range(1, 6):
            seed_dir = dataset_path / f"seed{seed_num}"
            best_params_file = seed_dir / "best_trial_models" / "best_params.json"
            
            if not best_params_file.exists():
                print(f"⚠️  best_params.json not found for {dataset_name} seed {seed_num}: {best_params_file}")
                continue
            
            try:
                with open(best_params_file, 'r', encoding='utf-8') as f:
                    best_params = json.load(f)
                
                # Extract information
                seed_result = {
                    "dataset": dataset_name,
                    "seed": seed_num,
                    "best_params": best_params,
                    "extracted_at": datetime.now().isoformat(),
                    "source_file": str(best_params_file.relative_to(base_path)),
                    "mode": "per_seed"  # Mark as per-seed optimization
                }
                
                all_seed_results[f"seed{seed_num}"] = seed_result
                
                print(f"✅ Successfully extracted best hyperparameters for {dataset_name} seed {seed_num}")
                print(f"   Source: {best_params_file}")
                print(f"   Parameters: {len(best_params)}")
                
            except Exception as e:
                print(f"⚠️  Error reading {best_params_file}: {e}")
                continue
    
    if not all_seed_results:
        print(f"❌ No valid results extracted for {dataset_name}")
        return None
    
    # Combine all seed results
    result = {
        "dataset": dataset_name,
        "seeds": all_seed_results,
        "extracted_at": datetime.now().isoformat(),
        "source": "best_params.json files"
    }
    
    return result


def extract_multiple_datasets(datasets, base_dir="checkpoints/optuna_mod", output_dir=None, combined_only=True):
    """
    Extract best hyperparameters for multiple datasets
    
    Args:
        datasets: List of dataset names
        base_dir: Base directory
        output_dir: Output directory (optional)
        combined_only: Whether to only output combined file (default: True, only output combined file)
    
    Returns:
        dict: Dictionary containing best hyperparameters for all datasets
    """
    all_results = {}
    
    print(f"🔍 Extracting best hyperparameters from files for {len(datasets)} dataset(s): {', '.join(datasets)}\n")
    
    for dataset in datasets:
        result = extract_best_params_from_files(dataset, base_dir)
        if result:
            all_results[dataset] = result
            
            # Save individual dataset results (only save when combined_only=False)
            if output_dir and not combined_only:
                os.makedirs(output_dir, exist_ok=True)
                output_file = os.path.join(output_dir, f"{dataset}_best_hyperparameters.json")
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                print(f"   💾 Saved to: {output_file}")
        
        print()  # Empty line separator
    
    # Save combined results
    if output_dir and all_results:
        os.makedirs(output_dir, exist_ok=True)
        combined_file = os.path.join(output_dir, "all_best_hyperparameters_mod.json")
        with open(combined_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"💾 Combined results saved to: {combined_file}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Extract best hyperparameters from existing best_params.json files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract single dataset
  python3 scripts/extract_best_params_from_files.py --dataset caco2_wang
  
  # Extract multiple datasets
  python3 scripts/extract_best_params_from_files.py --dataset caco2_wang herg vdss_lombardo
  
  # Specify output directory
  python3 scripts/extract_best_params_from_files.py --dataset caco2_wang --output-dir results/best_params
  
  # Only output combined file
  python3 scripts/extract_best_params_from_files.py --dataset caco2_wang --output-dir results/best_params --combined-only
        """
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        nargs='+',
        required=True,
        help="Dataset name(s) (can specify multiple)"
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default="checkpoints/optuna_mod",
        help="Base directory (default: checkpoints/optuna_mod)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: no output, only display results)"
    )
    parser.add_argument(
        "--individual-files",
        action="store_true",
        help="Also output individual JSON file for each dataset (default: only output combined file all_best_hyperparameters_mod.json)"
    )
    
    args = parser.parse_args()
    
    # Default to only output combined file, unless --individual-files is specified
    combined_only = not args.individual_files
    
    # Extract hyperparameters
    results = extract_multiple_datasets(
        datasets=args.dataset,
        base_dir=args.base_dir,
        output_dir=args.output_dir,
        combined_only=combined_only,
    )
    
    # Summary
    if results:
        print("=" * 80)
        print("📊 Extraction Summary")
        print("=" * 80)
        print(f"{'Dataset':<20} {'Seeds Found':<15} {'Total Params':<15}")
        print("-" * 80)
        for dataset, result in results.items():
            if 'seeds' in result:
                seeds_found = len(result['seeds'])
                # Calculate total parameter count (assuming each seed has same number of parameters)
                if seeds_found > 0:
                    first_seed = list(result['seeds'].values())[0]
                    total_params = len(first_seed.get('best_params', {}))
                    print(f"{dataset:<20} {seeds_found}/5{'':<10} {total_params}")
                else:
                    print(f"{dataset:<20} 0/5{'':<10} 0")
            else:
                print(f"{dataset:<20} {'N/A':<15} {'N/A':<15}")
        print("=" * 80)
        print(f"✅ Successfully extracted {len(results)} dataset(s)")
    else:
        print("❌ No results extracted")


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
Compare Optuna Best Trial's Validation Metrics with Final Training's Test Metrics

This script helps understand why there are differences between metrics obtained during Optuna optimization and metrics from final training.

Usage:
    python3 scripts/compare_optuna_vs_final_metrics.py <dataset> [seed]
    
Example:
    python3 scripts/compare_optuna_vs_final_metrics.py ames 1
    python3 scripts/compare_optuna_vs_final_metrics.py ames  # Compare all seeds
"""

import os
import sys
import json
import argparse
import optuna
from pathlib import Path

GREEN = '\033[0;32m'
BLUE = '\033[0;34m'
YELLOW = '\033[1;33m'
RED = '\033[0;31m'
NC = '\033[0m'


def load_optuna_validation_metric(dataset_name, seed_num=None):
    """Load best trial's validation metric from Optuna study"""
    storage_url = "sqlite:///optuna_edmpnn_results_new/optuna_mod_new.db"
    
    if seed_num is not None:
        # Per-seed optimization mode
        study_name = f"edmpnn_mod_new_{dataset_name}_seed{seed_num}_opt"
    else:
        # Legacy mode (all seeds)
        study_name = f"edmpnn_mod_new_{dataset_name}_opt"
    
    try:
        study = optuna.load_study(study_name=study_name, storage=storage_url)
        best_trial = study.best_trial
        
        # Get primary metric
        primary_metric = None
        if hasattr(study, 'user_attrs'):
            primary_metric = study.user_attrs.get('primary_metric')
        
        # If not found, try to get from best_trial
        if not primary_metric:
            # Try to get from best_trial's user_attrs
            primary_metric = best_trial.user_attrs.get('primary_metric')
        
        return {
            'best_trial_number': best_trial.number,
            'best_value': best_trial.value,  # This is validation metric
            'best_params': best_trial.params,
            'primary_metric': primary_metric,
            'study_name': study_name
        }
    except Exception as e:
        print(f"{RED}❌ Unable to load Optuna study: {e}{NC}")
        return None


def load_final_test_metrics(dataset_name, seed_num):
    """Load test metrics from final training's training_history.json"""
    history_path = Path(f"checkpoints/{dataset_name}_optuna_final/seed{seed_num}/training_history.json")
    
    if not history_path.exists():
        return None
    
    try:
        with open(history_path, 'r') as f:
            history = json.load(f)
        
        primary_metric = history.get('primary_metric', 'roc_auc')
        best_val_metric = history.get('best_primary_metric_value', None)
        test_results = history.get('test_results', {})
        
        return {
            'primary_metric': primary_metric,
            'best_val_metric': best_val_metric,  # Best value on validation set
            'test_metric': test_results.get(primary_metric, None),
            'test_results': test_results,
            'best_epoch': history.get('best_primary_metric_epoch', -1),
            'history_path': str(history_path)
        }
    except Exception as e:
        print(f"{RED}❌ Unable to load training_history.json: {e}{NC}")
        return None


def compare_metrics(dataset_name, seed_num=None):
    """Compare Optuna validation metrics with final test metrics"""
    print(f"\n{BLUE}========================================{NC}")
    print(f"{BLUE}Comparing Metrics: {dataset_name}{NC}")
    if seed_num:
        print(f"{BLUE}Seed: {seed_num}{NC}")
    print(f"{BLUE}========================================{NC}\n")
    
    if seed_num is not None:
        # Single seed
        seeds_to_check = [seed_num]
    else:
        # All seeds
        seeds_to_check = [1, 2, 3, 4, 5]
    
    results_summary = []
    
    for seed in seeds_to_check:
        print(f"{YELLOW}--- Seed {seed} ---{NC}")
        
        # 1. Load Optuna validation metric
        optuna_data = load_optuna_validation_metric(dataset_name, seed)
        if not optuna_data:
            print(f"  {RED}⚠️  Unable to load Optuna data{NC}")
            continue
        
        # 2. Load final training's test metrics
        final_data = load_final_test_metrics(dataset_name, seed)
        if not final_data:
            print(f"  {RED}⚠️  Unable to load final training data{NC}")
            continue
        
        # 3. Compare
        primary_metric = final_data['primary_metric']
        optuna_val_metric = optuna_data['best_value']
        final_val_metric = final_data['best_val_metric']
        final_test_metric = final_data['test_metric']
        
        print(f"  Primary Metric: {GREEN}{primary_metric}{NC}")
        print(f"  Optuna Best Trial: {GREEN}#{optuna_data['best_trial_number']}{NC}")
        print(f"  Optuna Validation {primary_metric}: {GREEN}{optuna_val_metric:.4f}{NC}")
        
        if final_val_metric is not None:
            print(f"  Final Training Validation {primary_metric}: {BLUE}{final_val_metric:.4f}{NC}")
            val_diff = abs(optuna_val_metric - final_val_metric)
            print(f"  Validation Difference: {YELLOW}{val_diff:.4f}{NC}")
        
        if final_test_metric is not None:
            print(f"  Final Training Test {primary_metric}: {BLUE}{final_test_metric:.4f}{NC}")
            test_diff = abs(optuna_val_metric - final_test_metric)
            print(f"  Test Difference: {YELLOW}{test_diff:.4f}{NC}")
            
            # Determine if difference is reasonable
            if primary_metric in ['roc_auc', 'pr_auc', 'spearman']:
                # For maximization metrics, difference should be < 0.1
                if test_diff > 0.1:
                    print(f"  {RED}⚠️  Warning: Test difference is large (> 0.1), may need to check{NC}")
                else:
                    print(f"  {GREEN}✅ Test difference is within reasonable range{NC}")
            elif primary_metric == 'mae':
                # For minimization metrics, need to judge based on data scale
                relative_diff = test_diff / max(abs(optuna_val_metric), 1e-6)
                if relative_diff > 0.2:
                    print(f"  {RED}⚠️  Warning: Test relative difference is large (> 20%), may need to check{NC}")
                else:
                    print(f"  {GREEN}✅ Test difference is within reasonable range{NC}")
        
        print()
        
        # Save results for summary
        results_summary.append({
            'seed': seed,
            'primary_metric': primary_metric,
            'optuna_val': optuna_val_metric,
            'final_val': final_val_metric,
            'final_test': final_test_metric,
            'val_diff': abs(optuna_val_metric - final_val_metric) if final_val_metric else None,
            'test_diff': abs(optuna_val_metric - final_test_metric) if final_test_metric else None
        })
    
    # Summary
    if len(results_summary) > 1:
        print(f"{BLUE}========================================{NC}")
        print(f"{BLUE}Summary{NC}")
        print(f"{BLUE}========================================{NC}\n")
        
        for result in results_summary:
            seed = result['seed']
            metric = result['primary_metric']
            optuna_val = result['optuna_val']
            final_test = result['final_test']
            test_diff = result['test_diff']
            
            if final_test is not None and test_diff is not None:
                print(f"Seed {seed}: Optuna Val {metric} = {optuna_val:.4f}, "
                      f"Final Test {metric} = {final_test:.4f}, "
                      f"Difference = {test_diff:.4f}")
        
        # Calculate average difference
        valid_diffs = [r['test_diff'] for r in results_summary if r['test_diff'] is not None]
        if valid_diffs:
            avg_diff = sum(valid_diffs) / len(valid_diffs)
            print(f"\nAverage Test Difference: {avg_diff:.4f}")
    
    print(f"\n{YELLOW}💡 Explanation:{NC}")
    print(f"  - Optuna uses {GREEN}Validation Set{NC} to select best hyperparameters (avoiding data leakage)")
    print(f"  - Final training uses {GREEN}Test Set{NC} to evaluate model (true performance)")
    print(f"  - The difference is {GREEN}normal{NC}, as long as it's within reasonable range")
    print(f"  - If difference is too large (> 0.1), may need to check data and model stability")


def main():
    parser = argparse.ArgumentParser(
        description="Compare Optuna Best Trial's Validation Metrics with Final Training's Test Metrics"
    )
    parser.add_argument('dataset', type=str, help='Dataset name (e.g., ames)')
    parser.add_argument('seed', type=int, nargs='?', default=None,
                       help='Seed number (1-5). If not specified, compare all seeds.')
    
    args = parser.parse_args()
    
    compare_metrics(args.dataset, args.seed)


if __name__ == "__main__":
    main()




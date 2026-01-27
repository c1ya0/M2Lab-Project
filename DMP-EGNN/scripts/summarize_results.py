#!/usr/bin/env python3
"""
Summarize final training results for three datasets
Calculate average of five seeds based on primary metrics
"""

import json
import os
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple

# Primary metrics configuration
PRIMARY_METRICS = {
    'vdss_lombardo': 'spearman',
    'caco2_wang': 'mae',
    'herg': 'roc_auc'
}

# Dataset paths
DATASET_PATHS = {
    'vdss_lombardo': 'vdss_lombardo_optuna_final_new',
    'caco2_wang': 'caco2_wang_optuna_final_new',
    'herg': 'herg_optuna_final_new'
}

BASE_PATH = Path(__file__).parent.parent / 'checkpoints'


def load_training_history(dataset_name: str, seed: str) -> Dict:
    """Load training_history.json for specified dataset and seed"""
    dataset_path = BASE_PATH / DATASET_PATHS[dataset_name] / f'seed{seed}' / 'training_history.json'
    if not dataset_path.exists():
        return None
    with open(dataset_path, 'r') as f:
        return json.load(f)


def get_primary_metric_value(history: Dict, metric_name: str) -> float:
    """Extract primary metric value from test_results"""
    if history is None:
        return None
    test_results = history.get('test_results', {})
    return test_results.get(metric_name, None)


def collect_all_results() -> Dict:
    """Collect results for all seeds of all datasets"""
    all_results = {}
    
    for dataset_name in DATASET_PATHS.keys():
        all_results[dataset_name] = {}
        primary_metric = PRIMARY_METRICS[dataset_name]
        
        # Try to load all possible seeds (1-5)
        for seed in range(1, 6):
            history = load_training_history(dataset_name, seed)
            if history is not None:
                metric_value = get_primary_metric_value(history, primary_metric)
                if metric_value is not None:
                    all_results[dataset_name][f'seed{seed}'] = {
                        'primary_metric_value': metric_value,
                        'all_metrics': history.get('test_results', {})
                    }
    
    return all_results


def calculate_statistics(values: List[float]) -> Tuple[float, float, float, float]:
    """Calculate statistics: mean, standard deviation, minimum, maximum"""
    if not values:
        return None, None, None, None
    values_array = np.array(values)
    mean = float(np.mean(values_array))
    std = float(np.std(values_array))
    min_val = float(np.min(values_array))
    max_val = float(np.max(values_array))
    return mean, std, min_val, max_val


def format_number(value: float, decimals: int = 4) -> str:
    """Format number"""
    if value is None:
        return "N/A"
    return f"{value:.{decimals}f}"


def generate_summary_report(all_results: Dict) -> str:
    """Generate summary report"""
    report = []
    report.append("# Final Training Results Summary for Three Datasets\n")
    report.append("Calculate average of all available seeds based on primary metrics\n")
    report.append("---\n")
    
    # Results summary table
    report.append("## Results Summary\n\n")
    report.append("| Dataset | Primary Metric | Seeds Count | Mean | Std | Min | Max |\n")
    report.append("|---------|----------------|-------------|------|-----|-----|-----|\n")
    
    for dataset_name in sorted(all_results.keys()):
        primary_metric = PRIMARY_METRICS[dataset_name]
        seed_results = all_results[dataset_name]
        
        if not seed_results:
            continue
        
        values = [data['primary_metric_value'] for data in seed_results.values()]
        mean, std, min_val, max_val = calculate_statistics(values)
        num_seeds = len(seed_results)
        
        report.append(f"| **{dataset_name}** | {primary_metric} | {num_seeds} | **{format_number(mean)}** | {format_number(std)} | {format_number(min_val)} | {format_number(max_val)} |\n")
    
    report.append("\n---\n")
    
    # Detailed results
    report.append("## Detailed Results\n\n")
    
    for dataset_name in sorted(all_results.keys()):
        primary_metric = PRIMARY_METRICS[dataset_name]
        seed_results = all_results[dataset_name]
        
        if not seed_results:
            continue
        
        report.append(f"### {dataset_name} (Primary Metric: {primary_metric})\n\n")
        report.append(f"| Seed | {primary_metric.upper()} |\n")
        report.append("|------|" + "-" * (len(primary_metric) + 2) + "|\n")
        
        values = []
        for seed in sorted(seed_results.keys()):
            value = seed_results[seed]['primary_metric_value']
            values.append(value)
            report.append(f"| {seed} | {format_number(value)} |\n")
        
        mean, std, min_val, max_val = calculate_statistics(values)
        report.append(f"| **Mean** | **{format_number(mean)}** |\n")
        report.append(f"| **Std** | **{format_number(std)}** |\n")
        report.append("\n")
        
        # Display all metrics (for reference)
        if seed_results:
            first_seed = sorted(seed_results.keys())[0]
            all_metrics = seed_results[first_seed]['all_metrics']
            if len(all_metrics) > 1:
                report.append("**All Test Metrics (seed1 as example):**\n")
                for metric, value in sorted(all_metrics.items()):
                    if metric != primary_metric:
                        if isinstance(value, float):
                            report.append(f"- {metric}: {format_number(value)}\n")
                        else:
                            report.append(f"- {metric}: {value}\n")
                report.append("\n")
        
        report.append("---\n\n")
    
    # Statistical analysis
    report.append("## Statistical Analysis\n\n")
    
    for dataset_name in sorted(all_results.keys()):
        primary_metric = PRIMARY_METRICS[dataset_name]
        seed_results = all_results[dataset_name]
        
        if not seed_results:
            continue
        
        values = [data['primary_metric_value'] for data in seed_results.values()]
        mean, std, min_val, max_val = calculate_statistics(values)
        
        report.append(f"### {dataset_name} ({primary_metric.upper()})\n")
        report.append(f"- **平均值**: {format_number(mean)}\n")
        report.append(f"- **範圍**: {format_number(min_val)} - {format_number(max_val)}\n")
        
        if std < 0.01:
            variability = "Low, very stable performance"
        elif std < 0.03:
            variability = "Medium"
        else:
            variability = "High"
        
        report.append(f"- **Variability**: {variability} (std {format_number(std)})\n\n")
    
    # Conclusions
    report.append("## Conclusions\n\n")
    
    # Find most stable dataset
    stability_scores = {}
    for dataset_name, seed_results in all_results.items():
        if seed_results:
            values = [data['primary_metric_value'] for data in seed_results.values()]
            _, std, _, _ = calculate_statistics(values)
            stability_scores[dataset_name] = std
    
    if stability_scores:
        most_stable = min(stability_scores.items(), key=lambda x: x[1])
        report.append(f"1. **{most_stable[0]}** dataset shows most stable performance with std {format_number(most_stable[1])}\n")
    
    # Find best performance
    performance_scores = {}
    for dataset_name, seed_results in all_results.items():
        if seed_results:
            values = [data['primary_metric_value'] for data in seed_results.values()]
            mean, _, _, _ = calculate_statistics(values)
            # For mae, lower is better; for spearman and roc_auc, higher is better
            if PRIMARY_METRICS[dataset_name] == 'mae':
                performance_scores[dataset_name] = -mean  # Take negative value for unified comparison
            else:
                performance_scores[dataset_name] = mean
    
    if performance_scores:
        best_performance = max(performance_scores.items(), key=lambda x: x[1])
        metric_name = PRIMARY_METRICS[best_performance[0]]
        if metric_name == 'mae':
            report.append(f"2. **{best_performance[0]}** dataset achieves lowest MAE ({format_number(-performance_scores[best_performance[0]])})\n")
        else:
            report.append(f"2. **{best_performance[0]}** dataset achieves highest {metric_name.upper()} ({format_number(performance_scores[best_performance[0]])})\n")
    
    report.append("\n---\n\n")
    report.append("*Report generated: 2024*\n")
    report.append("*Data source: AEGNN-M_TDC/checkpoints/*\n")
    
    return ''.join(report)


def main():
    """Main function"""
    print("Collecting training results...")
    all_results = collect_all_results()
    
    print("\nCollected results:")
    for dataset_name, seed_results in all_results.items():
        print(f"  {dataset_name}: {len(seed_results)} seeds")
        for seed, data in sorted(seed_results.items()):
            primary_metric = PRIMARY_METRICS[dataset_name]
            value = data['primary_metric_value']
            print(f"    {seed}: {primary_metric} = {value:.4f}")
    
    print("\nGenerating report...")
    report = generate_summary_report(all_results)
    
    # Save report
    output_path = BASE_PATH / 'training_results_summary.md'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\nReport saved to: {output_path}")
    print("\nReport preview:")
    print("=" * 80)
    print(report[:2000])  # Display first 2000 characters
    if len(report) > 2000:
        print("\n... (Report content is long, please check full file)")


if __name__ == '__main__':
    main()





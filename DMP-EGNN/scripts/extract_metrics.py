#!/usr/bin/env python3
"""
Extract training result metrics from checkpoints directory and generate tables and charts
"""

import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import argparse

# Set Chinese font
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

def load_all_metrics(checkpoints_dir='checkpoints'):
    """Extract metrics from training_history.json of all datasets"""
    datasets = ['bace', 'bbbp', 'clintox', 'hiv', 'muv', 'sider', 'tox21']
    results = []
    
    # Scan all subdirectories under checkpoints directory
    if os.path.exists(checkpoints_dir):
        subdirs = [d for d in os.listdir(checkpoints_dir) if os.path.isdir(os.path.join(checkpoints_dir, d))]
    else:
        print(f"Warning: Directory {checkpoints_dir} does not exist")
        return pd.DataFrame()

    for dataset in datasets:
        # Find matching directory
        target_dir = None
        # Prefer exact match
        if dataset in subdirs:
            target_dir = dataset
        else:
            # Find directories starting with dataset (e.g., bace_v8)
            matches = [d for d in subdirs if d.startswith(dataset + '_') or d == dataset]
            if matches:
                # If multiple matches, choose the newest or shortest name, here simply choose the first match
                # Sort to ensure stability
                matches.sort() 
                target_dir = matches[0]
        
        if target_dir:
            json_path = os.path.join(checkpoints_dir, target_dir, 'training_history.json')
            if os.path.exists(json_path):
                try:
                    with open(json_path, 'r') as f:
                        data = json.load(f)
                    
                    test_results = data.get('test_results', {})
                    auroc = test_results.get('roc_auc', 0)
                    
                    # Dynamically calculate best Epoch (based on Validation Loss)
                    # Fixes issues where training script may misjudge best model or overfitting
                    val_losses = data.get('val_losses', [])
                    if val_losses and len(val_losses) > 0:
                        min_val_loss = min(val_losses)
                        best_epoch_idx = val_losses.index(min_val_loss)
                        real_best_epoch = best_epoch_idx + 1  # Convert to 1-based
                        real_best_val_loss = min_val_loss
                        
                        # Check if record doesn't match calculation
                        recorded_epoch = data.get('best_epoch', -1)
                        if recorded_epoch != real_best_epoch:
                            print(f"⚠️  {dataset.upper()} Correction: Recorded Best Epoch {recorded_epoch} -> Actual Best Epoch {real_best_epoch} (Val Loss: {real_best_val_loss:.4f})")
                    else:
                        real_best_epoch = data.get('best_epoch', 'N/A')
                        real_best_val_loss = data.get('best_val_loss', 0)

                    results.append({
                        'Dataset': dataset.upper(),
                        'Best Epoch': real_best_epoch,
                        'Best Val Loss': f"{real_best_val_loss:.4f}",
                        'RMSE': test_results.get('rmse', 0),
                        'MAE': test_results.get('mae', 0),
                        'MSE': test_results.get('mse', 0),
                        'Accuracy': test_results.get('accuracy', 0),
                        'Precision': test_results.get('precision', 0),
                        'Recall': test_results.get('recall', 0),
                        'F1 Score': test_results.get('f1', 0),
                        'AUROC': auroc,
                    })
                except Exception as e:
                    print(f"Error reading {dataset} results: {e}")

    return pd.DataFrame(results)

def create_summary_table(df, output_path='training_plots/metrics_summary_table.txt'):
    """Create text format summary table"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write("Training Result Metrics Summary\n")
        f.write("=" * 100 + "\n\n")
        
        if df.empty:
            f.write("No data\n")
            return

        # Main metrics table
        f.write("Main Metrics (AUROC, RMSE)\n")
        f.write("-" * 100 + "\n")
        f.write(f"{'Dataset':<12} {'AUROC':<10} {'RMSE':<10} {'Accuracy':<12} {'F1 Score':<12}\n")
        f.write("-" * 100 + "\n")
        for _, row in df.iterrows():
            f.write(f"{row['Dataset']:<12} {row['AUROC']:<10.4f} "
                   f"{row['RMSE']:<10.4f} {row['Accuracy']:<12.4f} {row['F1 Score']:<12.4f}\n")
        
        f.write("\n" + "=" * 100 + "\n")
        f.write("Complete Metrics Table\n")
        f.write("=" * 100 + "\n\n")
        
        # Complete table
        df_display = df.copy()
        for col in df_display.columns:
            if col in ['RMSE', 'MAE', 'MSE', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUROC']:
                df_display[col] = df_display[col].apply(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x)
        
        f.write(df_display.to_string(index=False))
        f.write("\n\n")
        
        # Statistical summary
        f.write("=" * 100 + "\n")
        f.write("Statistical Summary\n")
        f.write("=" * 100 + "\n\n")
        numeric_cols = ['RMSE', 'MAE', 'MSE', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUROC']
        for col in numeric_cols:
            if col in df.columns:
                values = pd.to_numeric(df[col], errors='coerce').dropna()
                if len(values) > 0:
                    f.write(f"{col}:\n")
                    f.write(f"  Mean: {values.mean():.4f}\n")
                    f.write(f"  Std: {values.std():.4f}\n")
                    f.write(f"  Min: {values.min():.4f} ({df.loc[values.idxmin(), 'Dataset']})\n")
                    f.write(f"  Max: {values.max():.4f} ({df.loc[values.idxmax(), 'Dataset']})\n\n")
    
    print(f"✓ Created summary table: {output_path}")

def create_visualizations(df, output_dir='training_plots'):
    """Create visualization charts"""
    if df.empty:
        return

    os.makedirs(output_dir, exist_ok=True)
    
    # Ensure numeric columns are numeric type
    numeric_cols = ['RMSE', 'MAE', 'MSE', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUROC']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 1. Main metrics comparison chart (AUROC, RMSE)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    datasets = df['Dataset'].values
    x_pos = np.arange(len(datasets))
    
    # AUROC
    ax1 = axes[0]
    auroc_values = df['AUROC'].values
    bars1 = ax1.bar(x_pos, auroc_values, color='steelblue', alpha=0.7)
    ax1.set_xlabel('Dataset', fontsize=12)
    ax1.set_ylabel('AUROC', fontsize=12)
    ax1.set_title('AUROC by Dataset', fontsize=14, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(datasets, rotation=45, ha='right')
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim([0, 1])
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars1, auroc_values)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    # RMSE
    ax2 = axes[1]
    rmse_values = df['RMSE'].values
    bars2 = ax2.bar(x_pos, rmse_values, color='mediumseagreen', alpha=0.7)
    ax2.set_xlabel('Dataset', fontsize=12)
    ax2.set_ylabel('RMSE', fontsize=12)
    ax2.set_title('RMSE by Dataset', fontsize=14, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(datasets, rotation=45, ha='right')
    ax2.grid(axis='y', alpha=0.3)
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars2, rmse_values)):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(rmse_values) * 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Created metrics comparison chart: {output_dir}/metrics_comparison.png")
    plt.close()
    
    # 2. Comprehensive metrics heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Select metrics to display
    metrics_to_plot = ['AUROC', 'Accuracy', 'F1 Score', 'Precision', 'Recall']
    existing_metrics = [m for m in metrics_to_plot if m in df.columns]
    
    if existing_metrics:
        plot_data = df.set_index('Dataset')[existing_metrics].T
        
        im = ax.imshow(plot_data.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        
        # Set ticks
        ax.set_xticks(np.arange(len(plot_data.columns)))
        ax.set_yticks(np.arange(len(plot_data.index)))
        ax.set_xticklabels(plot_data.columns)
        ax.set_yticklabels(plot_data.index)
        
        # Rotate labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add values
        for i in range(len(plot_data.index)):
            for j in range(len(plot_data.columns)):
                text = ax.text(j, i, f'{plot_data.iloc[i, j]:.3f}',
                            ha="center", va="center", color="black", fontsize=9)
        
        ax.set_title("Metrics Heatmap Across Datasets", fontsize=14, fontweight='bold', pad=20)
        plt.colorbar(im, ax=ax, label='Score')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'metrics_heatmap.png'), dpi=300, bbox_inches='tight')
        print(f"✓ Created metrics heatmap: {output_dir}/metrics_heatmap.png")
        plt.close()
    
    # 3. Side-by-side bar chart for all metrics
    fig, ax = plt.subplots(figsize=(16, 8))
    
    metrics = ['AUROC', 'Accuracy', 'F1 Score']
    existing_metrics = [m for m in metrics if m in df.columns]
    x = np.arange(len(datasets))
    width = 0.2
    
    for i, metric in enumerate(existing_metrics):
        offset = (i - len(existing_metrics)/2 + 0.5) * width
        values = df[metric].values
        bars = ax.bar(x + offset, values, width, label=metric, alpha=0.8)
        
        # Add value labels (only show first two metrics to avoid crowding)
        if i < 2:
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{val:.2f}', ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Dataset', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Key Metrics Comparison Across Datasets', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha='right')
    ax.legend(loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.1])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'all_metrics_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Created all metrics comparison chart: {output_dir}/all_metrics_comparison.png")
    plt.close()

def save_csv(df, output_path='training_plots/metrics_summary.csv'):
    """Save as CSV file"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"✓ Saved CSV file: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Extract metrics from checkpoints')
    parser.add_argument('--checkpoints_dir', type=str, default='checkpoints', help='Directory containing checkpoints')
    parser.add_argument('--output_dir', type=str, default='training_plots', help='Directory to save output plots')
    args = parser.parse_args()

    print(f"Extracting training metrics from {args.checkpoints_dir}...")
    print(f"Results will be saved to {args.output_dir}")
    
    df = load_all_metrics(args.checkpoints_dir)
    
    if df.empty:
        print("⚠️  No training result data found. Please check if the checkpoints directory path is correct.")
        return

    print(f"\nFound results for {len(df)} datasets")
    print("\nData preview:")
    print(df[['Dataset', 'AUROC', 'RMSE']].to_string(index=False))
    
    print("\nGenerating tables and charts...")
    create_summary_table(df, os.path.join(args.output_dir, 'metrics_summary_table.txt'))
    save_csv(df, os.path.join(args.output_dir, 'metrics_summary.csv'))
    create_visualizations(df, args.output_dir)
    
    print(f"\nDone! All files have been saved to {args.output_dir} directory")

if __name__ == '__main__':
    main()

"""
Training Results Visualization Script
Reads training history from checkpoints and TensorBoard logs and generates charts
"""
# python scripts/visualize_training.py

import os
import sys
import argparse
import torch
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Set font
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_tensorboard_logs(log_dir):
    """Read training history from TensorBoard logs"""
    if not os.path.exists(log_dir):
        return None
    
    try:
        ea = EventAccumulator(log_dir)
        ea.Reload()
        
        # Get all scalar tags
        scalar_tags = ea.Tags()['scalars']
        
        history = {}
        for tag in scalar_tags:
            scalar_events = ea.Scalars(tag)
            values = [event.value for event in scalar_events]
            steps = [event.step for event in scalar_events]
            history[tag] = {'values': values, 'steps': steps}
        
        return history
    except Exception as e:
        print(f"⚠️  Cannot read TensorBoard logs: {e}")
        return None


def load_training_history_json(checkpoint_dir):
    """Read complete training history from JSON file"""
    all_histories = {}
    
    # Check if directory exists
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        print(f"⚠️  Checkpoints directory does not exist: {checkpoint_dir}")
        return all_histories
    
    if not checkpoint_path.is_dir():
        print(f"⚠️  {checkpoint_dir} is not a directory")
        return all_histories
    
    for dataset_dir in checkpoint_path.iterdir():
        if not dataset_dir.is_dir():
            continue
        
        dataset_name = dataset_dir.name
        history_path = dataset_dir / 'training_history.json'
        
        if history_path.exists():
            try:
                with open(history_path, 'r') as f:
                    history = json.load(f)
                all_histories[dataset_name] = history
            except Exception as e:
                print(f"⚠️  Cannot read training history for {dataset_name}: {e}")
    
    return all_histories


def load_checkpoint_history(checkpoint_dir):
    """Read training information for all datasets from checkpoints directory"""
    datasets_info = {}
    
    # Check if directory exists
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        print(f"⚠️  Checkpoints directory does not exist: {checkpoint_dir}")
        return datasets_info
    
    if not checkpoint_path.is_dir():
        print(f"⚠️  {checkpoint_dir} is not a directory")
        return datasets_info
    
    for dataset_dir in checkpoint_path.iterdir():
        if not dataset_dir.is_dir():
            continue
        
        dataset_name = dataset_dir.name
        model_path = dataset_dir / 'best_model.pth'
        
        if not model_path.exists():
            continue
        
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            datasets_info[dataset_name] = {
                'best_epoch': checkpoint.get('epoch', 'N/A'),
                'best_train_loss': checkpoint.get('train_loss', 'N/A'),
                'best_val_loss': checkpoint.get('val_loss', 'N/A'),
            }
        except Exception as e:
            print(f"⚠️  Cannot read model for {dataset_name}: {e}")
    
    return datasets_info


def find_tensorboard_logs(base_dir='./runs'):
    """Find all TensorBoard log directories"""
    log_dirs = {}
    
    if not os.path.exists(base_dir):
        return log_dirs
    
    # Find all directories containing events.out.tfevents
    for root, dirs, files in os.walk(base_dir):
        # Find directories containing events.out.tfevents
        event_files = [f for f in files if f.startswith('events.out.tfevents')]
        if event_files:
            # Try to extract dataset name from path
            path_parts = Path(root).parts
            dataset_name = None
            
            # Method 1: Find dataset name from path
            for part in reversed(path_parts):
                if part in ['bbbp', 'bace', 'clintox', 'hiv', 'muv', 'sider', 'tox21']:
                    dataset_name = part
                    break
            
            # Method 2: If not found, use directory name (may be timestamp format)
            if not dataset_name:
                # Check parent directory or use last directory name
                dataset_name = Path(root).name
            
            if dataset_name:
                # If this dataset already has logs, choose the latest
                if dataset_name not in log_dirs:
                    log_dirs[dataset_name] = root
                else:
                    # Compare timestamps, choose the latest
                    current_time = os.path.getmtime(root)
                    existing_time = os.path.getmtime(log_dirs[dataset_name])
                    if current_time > existing_time:
                        log_dirs[dataset_name] = root
    
    return log_dirs


def plot_training_curves_from_json(history_data, dataset_name, output_dir='./training_plots'):
    """Plot training curves and overfitting analysis from JSON data"""
    os.makedirs(output_dir, exist_ok=True)
    
    train_losses = history_data.get('train_losses', [])
    val_losses = history_data.get('val_losses', [])
    
    if not train_losses or not val_losses:
        print(f"⚠️  {dataset_name}: Training history data incomplete")
        return
    
    epochs = list(range(1, len(train_losses) + 1))
    
    # Create chart: 2 rows 2 columns, clearer layout
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # ========== Chart 1: Training Loss and Validation Loss Line Plot ==========
    axes[0, 0].plot(epochs, train_losses, label='Training Loss', linewidth=2.5, 
                    color='#2E86AB', marker='o', markersize=4, alpha=0.8)
    axes[0, 0].plot(epochs, val_losses, label='Validation Loss', linewidth=2.5, 
                    color='#A23B72', marker='s', markersize=4, alpha=0.8)
    
    # Mark best validation loss point
    best_val_loss = min(val_losses)
    best_epoch = val_losses.index(best_val_loss) + 1
    axes[0, 0].plot(best_epoch, best_val_loss, 'r*', markersize=20, 
                    label=f'Best Val Loss: {best_val_loss:.4f} (Epoch {best_epoch})', zorder=5)
    
    axes[0, 0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Loss', fontsize=12, fontweight='bold')
    axes[0, 0].set_title(f'{dataset_name.upper()} - Training & Validation Loss', 
                         fontsize=14, fontweight='bold', pad=15)
    axes[0, 0].legend(fontsize=10, loc='best')
    axes[0, 0].grid(True, alpha=0.3, linestyle='--')
    
    # ========== Chart 2: Loss Difference (Overfitting Analysis) ==========
    if len(train_losses) == len(val_losses):
        loss_diff = [v - t for t, v in zip(train_losses, val_losses)]
        axes[0, 1].plot(epochs, loss_diff, label='Val Loss - Train Loss', 
                       linewidth=2.5, color='#F18F01', marker='D', markersize=3, alpha=0.8)
        axes[0, 1].axhline(y=0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='Zero Line')
        
        # Mark overfitting regions (validation loss significantly higher than training loss)
        overfitting_epochs = [e for e, diff in zip(epochs, loss_diff) if diff > 0.1]
        if overfitting_epochs:
            axes[0, 1].fill_between(epochs, loss_diff, 0, where=[d > 0.1 for d in loss_diff], 
                                   color='red', alpha=0.2, label='Potential Overfitting Zone')
        
        axes[0, 1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
        axes[0, 1].set_ylabel('Loss Difference (Val - Train)', fontsize=12, fontweight='bold')
        axes[0, 1].set_title(f'{dataset_name.upper()} - Overfitting Analysis', 
                            fontsize=14, fontweight='bold', pad=15)
        axes[0, 1].legend(fontsize=10, loc='best')
        axes[0, 1].grid(True, alpha=0.3, linestyle='--')
    
    # ========== Chart 3: Training Loss Trend (Displayed Separately) ==========
    axes[1, 0].plot(epochs, train_losses, label='Training Loss', linewidth=2.5, 
                    color='#2E86AB', marker='o', markersize=4, alpha=0.8)
    axes[1, 0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Training Loss', fontsize=12, fontweight='bold')
    axes[1, 0].set_title(f'{dataset_name.upper()} - Training Loss Trend', 
                         fontsize=14, fontweight='bold', pad=15)
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3, linestyle='--')
    
    # Add trend line (optional)
    if len(train_losses) > 5:
        z = np.polyfit(epochs, train_losses, 1)
        p = np.poly1d(z)
        axes[1, 0].plot(epochs, p(epochs), '--', color='gray', alpha=0.5, 
                       label=f'Trend: {z[0]:.4f}x + {z[1]:.4f}')
        axes[1, 0].legend(fontsize=10)
    
    # ========== Chart 4: Validation Loss Trend (Displayed Separately) ==========
    axes[1, 1].plot(epochs, val_losses, label='Validation Loss', linewidth=2.5, 
                    color='#A23B72', marker='s', markersize=4, alpha=0.8)
    
    # Mark best point
    axes[1, 1].plot(best_epoch, best_val_loss, 'r*', markersize=20, 
                    label=f'Best: {best_val_loss:.4f} (Epoch {best_epoch})', zorder=5)
    
    axes[1, 1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('Validation Loss', fontsize=12, fontweight='bold')
    axes[1, 1].set_title(f'{dataset_name.upper()} - Validation Loss Trend', 
                         fontsize=14, fontweight='bold', pad=15)
    axes[1, 1].legend(fontsize=10, loc='best')
    axes[1, 1].grid(True, alpha=0.3, linestyle='--')
    
    # Add trend line (optional)
    if len(val_losses) > 5:
        z = np.polyfit(epochs, val_losses, 1)
        p = np.poly1d(z)
        axes[1, 1].plot(epochs, p(epochs), '--', color='gray', alpha=0.5, 
                       label=f'Trend: {z[0]:.4f}x + {z[1]:.4f}')
        axes[1, 1].legend(fontsize=10)
    
    plt.tight_layout(pad=3.0)
    
    # Save chart
    output_path = os.path.join(output_dir, f'{dataset_name}_training_curves.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Chart saved: {output_path}")
    plt.close()
    
    # ========== Generate Overfitting Analysis Report ==========
    if len(train_losses) == len(val_losses):
        final_train_loss = train_losses[-1]
        final_val_loss = val_losses[-1]
        loss_gap = final_val_loss - final_train_loss
        
        # Determine overfitting degree
        if loss_gap > 0.2:
            overfitting_status = "⚠️  Possible overfitting"
        elif loss_gap > 0.1:
            overfitting_status = "⚠️  Mild overfitting"
        elif loss_gap < -0.1:
            overfitting_status = "✅ Validation loss lower than training loss (Good)"
        else:
            overfitting_status = "✅ Training and validation loss close (Good)"
        
        print(f"   📊 {dataset_name.upper()} Overfitting Analysis:")
        print(f"      Final training loss: {final_train_loss:.4f}")
        print(f"      Final validation loss: {final_val_loss:.4f}")
        print(f"      Loss gap: {loss_gap:.4f}")
        print(f"      Status: {overfitting_status}")


def plot_training_curves(history, dataset_name, output_dir='./training_plots'):
    """Plot training curves from TensorBoard history"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract training and validation loss
    train_loss_tag = None
    val_loss_tag = None
    lr_tag = None
    
    for tag in history.keys():
        if 'Train/Loss' in tag or 'train_loss' in tag.lower():
            train_loss_tag = tag
        if 'Val/Loss' in tag or 'val_loss' in tag.lower():
            val_loss_tag = tag
        if 'Learning_Rate' in tag or 'lr' in tag.lower():
            lr_tag = tag
    
    if not train_loss_tag or not val_loss_tag:
        print(f"⚠️  {dataset_name}: Cannot find training/validation loss data")
        return
    
    train_loss = history[train_loss_tag]['values']
    val_loss = history[val_loss_tag]['values']
    epochs = history[train_loss_tag]['steps']
    
    # Ensure epochs start from 1 (TensorBoard may start from 0)
    if epochs and epochs[0] == 0:
        epochs = [e + 1 for e in epochs]
    
    # Create chart: 2 rows 2 columns, same format as JSON version
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # ========== Chart 1: Training Loss and Validation Loss Line Plot ==========
    axes[0, 0].plot(epochs, train_loss, label='Training Loss', linewidth=2.5, 
                    color='#2E86AB', marker='o', markersize=4, alpha=0.8)
    axes[0, 0].plot(epochs, val_loss, label='Validation Loss', linewidth=2.5, 
                    color='#A23B72', marker='s', markersize=4, alpha=0.8)
    
    # Mark best validation loss point
    best_val_loss = min(val_loss)
    best_epoch_idx = val_loss.index(best_val_loss)
    best_epoch = epochs[best_epoch_idx]
    axes[0, 0].plot(best_epoch, best_val_loss, 'r*', markersize=20, 
                    label=f'Best Val Loss: {best_val_loss:.4f} (Epoch {best_epoch})', zorder=5)
    
    axes[0, 0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Loss', fontsize=12, fontweight='bold')
    axes[0, 0].set_title(f'{dataset_name.upper()} - Training & Validation Loss', 
                         fontsize=14, fontweight='bold', pad=15)
    axes[0, 0].legend(fontsize=10, loc='best')
    axes[0, 0].grid(True, alpha=0.3, linestyle='--')
    
    # ========== Chart 2: Loss Difference (Overfitting Analysis) ==========
    if len(train_loss) == len(val_loss):
        loss_diff = [v - t for t, v in zip(train_loss, val_loss)]
        axes[0, 1].plot(epochs, loss_diff, label='Val Loss - Train Loss', 
                       linewidth=2.5, color='#F18F01', marker='D', markersize=3, alpha=0.8)
        axes[0, 1].axhline(y=0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='Zero Line')
        
        # Mark overfitting regions
        if any(d > 0.1 for d in loss_diff):
            axes[0, 1].fill_between(epochs, loss_diff, 0, where=[d > 0.1 for d in loss_diff], 
                                   color='red', alpha=0.2, label='Potential Overfitting Zone')
        
        axes[0, 1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
        axes[0, 1].set_ylabel('Loss Difference (Val - Train)', fontsize=12, fontweight='bold')
        axes[0, 1].set_title(f'{dataset_name.upper()} - Overfitting Analysis', 
                            fontsize=14, fontweight='bold', pad=15)
        axes[0, 1].legend(fontsize=10, loc='best')
        axes[0, 1].grid(True, alpha=0.3, linestyle='--')
    
    # ========== Chart 3: Training Loss Trend (Displayed Separately) ==========
    axes[1, 0].plot(epochs, train_loss, label='Training Loss', linewidth=2.5, 
                    color='#2E86AB', marker='o', markersize=4, alpha=0.8)
    axes[1, 0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Training Loss', fontsize=12, fontweight='bold')
    axes[1, 0].set_title(f'{dataset_name.upper()} - Training Loss Trend', 
                         fontsize=14, fontweight='bold', pad=15)
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3, linestyle='--')
    
    # Add trend line
    if len(train_loss) > 5:
        z = np.polyfit(epochs, train_loss, 1)
        p = np.poly1d(z)
        axes[1, 0].plot(epochs, p(epochs), '--', color='gray', alpha=0.5, 
                       label=f'Trend: {z[0]:.4f}x + {z[1]:.4f}')
        axes[1, 0].legend(fontsize=10)
    
    # ========== Chart 4: Validation Loss Trend (Displayed Separately) ==========
    axes[1, 1].plot(epochs, val_loss, label='Validation Loss', linewidth=2.5, 
                    color='#A23B72', marker='s', markersize=4, alpha=0.8)
    
    # Mark best point
    axes[1, 1].plot(best_epoch, best_val_loss, 'r*', markersize=20, 
                    label=f'Best: {best_val_loss:.4f} (Epoch {best_epoch})', zorder=5)
    
    axes[1, 1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('Validation Loss', fontsize=12, fontweight='bold')
    axes[1, 1].set_title(f'{dataset_name.upper()} - Validation Loss Trend', 
                         fontsize=14, fontweight='bold', pad=15)
    axes[1, 1].legend(fontsize=10, loc='best')
    axes[1, 1].grid(True, alpha=0.3, linestyle='--')
    
    # Add trend line
    if len(val_loss) > 5:
        z = np.polyfit(epochs, val_loss, 1)
        p = np.poly1d(z)
        axes[1, 1].plot(epochs, p(epochs), '--', color='gray', alpha=0.5, 
                       label=f'Trend: {z[0]:.4f}x + {z[1]:.4f}')
        axes[1, 1].legend(fontsize=10)
    
    plt.tight_layout(pad=3.0)
    
    # Save chart
    output_path = os.path.join(output_dir, f'{dataset_name}_training_curves.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Chart saved: {output_path}")
    plt.close()
    
    # ========== Generate Overfitting Analysis Report ==========
    if len(train_loss) == len(val_loss):
        final_train_loss = train_loss[-1]
        final_val_loss = val_loss[-1]
        loss_gap = final_val_loss - final_train_loss
        
        # Determine overfitting degree
        if loss_gap > 0.2:
            overfitting_status = "⚠️  Possible overfitting"
        elif loss_gap > 0.1:
            overfitting_status = "⚠️  Mild overfitting"
        elif loss_gap < -0.1:
            overfitting_status = "✅ Validation loss lower than training loss (Good)"
        else:
            overfitting_status = "✅ Training and validation loss close (Good)"
        
        print(f"   📊 {dataset_name.upper()} Overfitting Analysis:")
        print(f"      Final training loss: {final_train_loss:.4f}")
        print(f"      Final validation loss: {final_val_loss:.4f}")
        print(f"      Loss gap: {loss_gap:.4f}")
        print(f"      Status: {overfitting_status}")


def plot_summary_comparison(datasets_info, output_dir='./training_plots'):
    """Plot comparison chart for all datasets"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data
    dataset_names = []
    train_losses = []
    val_losses = []
    
    for name, info in datasets_info.items():
        if info['best_train_loss'] != 'N/A' and info['best_val_loss'] != 'N/A':
            dataset_names.append(name.upper())
            train_losses.append(info['best_train_loss'])
            val_losses.append(info['best_val_loss'])
    
    if not dataset_names:
        print("⚠️  Not enough data to generate comparison chart")
        return
    
    # Create comparison chart
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    x = np.arange(len(dataset_names))
    width = 0.35
    
    # Best training loss comparison
    axes[0].bar(x - width/2, train_losses, width, label='Best Train Loss', color='#2E86AB', alpha=0.8)
    axes[0].bar(x + width/2, val_losses, width, label='Best Val Loss', color='#A23B72', alpha=0.8)
    axes[0].set_xlabel('Dataset', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Best Training and Validation Loss Comparison', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(dataset_names, rotation=45, ha='right')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Validation loss comparison (separate)
    axes[1].bar(x, val_losses, width=0.6, color='#A23B72', alpha=0.8)
    axes[1].set_xlabel('Dataset', fontsize=12)
    axes[1].set_ylabel('Best Validation Loss', fontsize=12)
    axes[1].set_title('Best Validation Loss by Dataset', fontsize=14, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(dataset_names, rotation=45, ha='right')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, v in enumerate(val_losses):
        axes[1].text(i, v, f'{v:.4f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    # Save chart
    output_path = os.path.join(output_dir, 'all_datasets_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Comparison chart saved: {output_path}")
    plt.close()


def generate_summary_table(datasets_info, output_dir='./training_plots'):
    """Generate training results summary table"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create table data
    table_data = []
    for name, info in sorted(datasets_info.items()):
        table_data.append([
            name.upper(),
            info['best_epoch'],
            f"{info['best_train_loss']:.4f}" if info['best_train_loss'] != 'N/A' else 'N/A',
            f"{info['best_val_loss']:.4f}" if info['best_val_loss'] != 'N/A' else 'N/A',
        ])
    
    # Save as text file
    output_path = os.path.join(output_dir, 'training_summary.txt')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("Training Results Summary\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"{'Dataset':<15} {'Best Epoch':<15} {'Train Loss':<15} {'Val Loss':<15}\n")
        f.write("-" * 80 + "\n")
        for row in table_data:
            f.write(f"{row[0]:<15} {str(row[1]):<15} {row[2]:<15} {row[3]:<15}\n")
    
    print(f"✅ Summary table saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Visualize training results')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                       help='Checkpoints directory path')
    parser.add_argument('--runs_dir', type=str, default='./runs',
                       help='TensorBoard logs directory path')
    parser.add_argument('--output_dir', type=str, default='./training_plots',
                       help='Output charts directory')
    
    args = parser.parse_args()
    
    # Get script directory, convert relative paths to absolute paths
    script_dir = Path(__file__).parent.parent  # Go back to AEGNN-M directory
    
    # Convert to absolute paths
    checkpoint_dir = Path(args.checkpoint_dir)
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = script_dir / checkpoint_dir
    
    runs_dir = Path(args.runs_dir)
    if not runs_dir.is_absolute():
        runs_dir = script_dir / runs_dir
    
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = script_dir / output_dir
    
    print("📊 Starting to generate training results charts...\n")
    print(f"📁 Checkpoints directory: {checkpoint_dir}")
    print(f"📁 Runs directory: {runs_dir}")
    print(f"📁 Output directory: {output_dir}\n")
    
    # 1. Read complete training history from JSON files (priority)
    print("1. Reading training history JSON files...")
    json_histories = load_training_history_json(str(checkpoint_dir))
    print(f"   Found training history for {len(json_histories)} datasets\n")
    
    # 2. Read basic information from checkpoints (as backup)
    print("2. Reading checkpoint information...")
    datasets_info = load_checkpoint_history(str(checkpoint_dir))
    print(f"   Found models for {len(datasets_info)} datasets\n")
    
    # 3. Read complete training history from TensorBoard logs (if JSON doesn't exist)
    print("3. Reading TensorBoard logs...")
    log_dirs = find_tensorboard_logs(str(runs_dir))
    print(f"   Found training logs for {len(log_dirs)} datasets\n")
    
    # 4. Generate training curves for each dataset
    print("4. Generating training curve charts...")
    curves_generated = 0
    
    if json_histories:
        # Prefer JSON files
        print("   Using JSON files to generate training curves...")
        for dataset_name, history_data in json_histories.items():
            plot_training_curves_from_json(history_data, dataset_name, str(output_dir))
            curves_generated += 1
    elif log_dirs:
        # Backup: Use TensorBoard logs
        print("   Using TensorBoard logs to generate training curves...")
        for dataset_name, log_dir in log_dirs.items():
            print(f"   Processing TensorBoard logs for {dataset_name}: {log_dir}")
            history = load_tensorboard_logs(log_dir)
            if history:
                plot_training_curves(history, dataset_name, str(output_dir))
                curves_generated += 1
            else:
                print(f"   ⚠️  Cannot read data for {dataset_name} from TensorBoard logs")
    else:
        # If neither exists, try to read from all TensorBoard log directories
        print("   ⚠️  JSON files not found, trying to read from all TensorBoard logs...")
        if os.path.exists(str(runs_dir)):
            # Get all directories containing TensorBoard logs, sorted by time (newest first)
            all_log_dirs = []
            for root, dirs, files in os.walk(str(runs_dir)):
                event_files = [f for f in files if f.startswith('events.out.tfevents')]
                if event_files:
                    # Get log file modification time
                    log_file = os.path.join(root, event_files[0])
                    all_log_dirs.append((os.path.getmtime(log_file), root))
            
            # Sort by time, newest first
            all_log_dirs.sort(reverse=True)
            
            # Get checkpoint time for each dataset
            checkpoint_times = {}
            for dataset_name in datasets_info.keys():
                checkpoint_path = Path(checkpoint_dir) / dataset_name / 'best_model.pth'
                if checkpoint_path.exists():
                    checkpoint_times[dataset_name] = os.path.getmtime(str(checkpoint_path))
            
            # Find TensorBoard logs closest in time for each dataset
            processed_log_dirs = set()
            for dataset_name in datasets_info.keys():
                if dataset_name not in checkpoint_times:
                    continue
                
                checkpoint_time = checkpoint_times[dataset_name]
                best_match = None
                best_time_diff = float('inf')
                
                # Find TensorBoard log closest in time (within 2 hours before/after checkpoint time)
                for log_time, log_dir in all_log_dirs:
                    time_diff = abs(log_time - checkpoint_time)
                    # Only consider logs within 2 hours before/after checkpoint time
                    if time_diff < 7200 and time_diff < best_time_diff:
                        history = load_tensorboard_logs(log_dir)
                        if history:
                            has_train = any('Train/Loss' in tag or 'train_loss' in tag.lower() for tag in history.keys())
                            has_val = any('Val/Loss' in tag or 'val_loss' in tag.lower() for tag in history.keys())
                            if has_train and has_val:
                                best_match = (log_dir, history)
                                best_time_diff = time_diff
                
                if best_match:
                    log_dir, history = best_match
                    print(f"   Found corresponding TensorBoard log for {dataset_name}: {log_dir}")
                    plot_training_curves(history, dataset_name, str(output_dir))
                    curves_generated += 1
                    processed_log_dirs.add(log_dir)
                else:
                    print(f"   ⚠️  Cannot find corresponding TensorBoard log for {dataset_name}")
    
    if curves_generated == 0:
        print("   ⚠️  Cannot generate training curve charts (missing training history data)")
        print("   💡 Tip: Future training will automatically save training_history.json files")
        print("   💡 Tip: You can manually view TensorBoard: tensorboard --logdir=runs")
    else:
        print(f"   ✅ Successfully generated training curve charts for {curves_generated} datasets")
    print()
    
    # 5. Generate comparison chart
    print("5. Generating dataset comparison chart...")
    plot_summary_comparison(datasets_info, str(output_dir))
    print()
    
    # 6. Generate summary table
    print("6. Generating training summary table...")
    generate_summary_table(datasets_info, str(output_dir))
    print()
    
    print("✅ All charts generated successfully!")
    print(f"📁 Output directory: {output_dir}")


if __name__ == "__main__":
    main()


#!/bin/bash

# Enhanced Training Monitor for train_edmpnn.sh and train_edmpnn_new.sh
# Usage: ./monitor_training.sh [refresh_interval_seconds]
# Interactive controls:
#   's' - Switch display mode (compact/detailed)
#   'p' - Pause/resume refresh
#   'e' - Export current status to file
#   'q' - Quit
# 
# Features:
#   - Automatically detects and monitors both train_edmpnn.sh and train_edmpnn_new.sh
#   - Supports both optuna_final and optuna_final_new checkpoint directories
#   - Auto-detects currently training datasets from running processes

INTERVAL=${1:-10}

# Display mode: "compact" or "detailed"
DISPLAY_MODE="detailed"
PAUSED=false
LAST_EXPORT_TIME=""

# Function to read keypress (non-blocking)
read_key() {
    local key
    if read -t 0.1 -n 1 key 2>/dev/null; then
        echo "$key"
    fi
}

# Function to export status
export_status() {
    local export_file="monitor_training_export_$(date +%Y%m%d_%H%M%S).txt"
    {
        echo "Training Monitor Export - $(date '+%Y-%m-%d %H:%M:%S')"
        echo "=========================================="
        echo ""
    } > "$export_file"
    
    python3 << EOF >> "$export_file"
import os
import json
import glob
from datetime import datetime
from pathlib import Path

def get_training_status():
    """Scan all training checkpoints and return status"""
    checkpoints_dir = Path("checkpoints")
    if not checkpoints_dir.exists():
        return {}
    
    status = {}
    
    # Find all dataset directories matching *_optuna_final or *_optuna_final_new pattern
    for dataset_dir in checkpoints_dir.glob("*_optuna_final*"):
        # Handle both _optuna_final and _optuna_final_new
        if dataset_dir.name.endswith("_optuna_final_new"):
            dataset_name = dataset_dir.name.replace("_optuna_final_new", "")
        elif dataset_dir.name.endswith("_optuna_final"):
            dataset_name = dataset_dir.name.replace("_optuna_final", "")
        else:
            continue
        
        if dataset_name not in status:
            status[dataset_name] = {
                'seeds': {},
                'completed_seeds': 0,
                'running_seeds': 0,
                'failed_seeds': 0,
                'total_seeds': 0
            }
        
        # Check each seed (1-5)
        for seed in range(1, 6):
            seed_dir = dataset_dir / f"seed{seed}"
            seed_key = f"seed{seed}"
            
            if seed_key not in status[dataset_name]['seeds']:
                status[dataset_name]['seeds'][seed_key] = {
                    'state': 'unknown',
                    'epoch': None,
                    'val_loss': None,
                    'test_score': None,
                    'timestamp': None
                }
            
            # Check if training_history.json exists (completed)
            history_file = seed_dir / "training_history.json"
            progress_file = seed_dir / "training_progress.json"
            
            if history_file.exists():
                try:
                    with open(history_file, 'r') as f:
                        history = json.load(f)
                        status[dataset_name]['seeds'][seed_key]['state'] = 'completed'
                        status[dataset_name]['seeds'][seed_key]['epoch'] = history.get('best_epoch', history.get('total_epochs', 'N/A'))
                        
                        # Get test score
                        test_results = history.get('test_results', {})
                        if test_results:
                            # Try common metrics
                            for metric in ['roc_auc', 'pr_auc', 'mae', 'spearman']:
                                if metric in test_results:
                                    status[dataset_name]['seeds'][seed_key]['test_score'] = test_results[metric]
                                    break
                        
                        status[dataset_name]['completed_seeds'] += 1
                except Exception as e:
                    status[dataset_name]['seeds'][seed_key]['state'] = 'error'
                    status[dataset_name]['seeds'][seed_key]['error'] = str(e)
            elif progress_file.exists():
                # Check if process is still running
                try:
                    with open(progress_file, 'r') as f:
                        progress = json.load(f)
                        status[dataset_name]['seeds'][seed_key]['state'] = 'running'
                        status[dataset_name]['seeds'][seed_key]['epoch'] = progress.get('epoch', 0)
                        status[dataset_name]['seeds'][seed_key]['val_loss'] = progress.get('val_loss')
                        status[dataset_name]['seeds'][seed_key]['timestamp'] = progress.get('timestamp')
                        status[dataset_name]['running_seeds'] += 1
                except Exception as e:
                    status[dataset_name]['seeds'][seed_key]['state'] = 'error'
                    status[dataset_name]['seeds'][seed_key]['error'] = str(e)
            else:
                # No files found - might be pending or failed
                status[dataset_name]['seeds'][seed_key]['state'] = 'pending'
            
            status[dataset_name]['total_seeds'] += 1
    
    return status

status = get_training_status()

print("Training Status Summary:")
print("=" * 50)
for dataset_name, dataset_status in sorted(status.items()):
    print(f"\nDataset: {dataset_name}")
    print(f"  Completed: {dataset_status['completed_seeds']}/5")
    print(f"  Running: {dataset_status['running_seeds']}/5")
    print(f"  Pending/Failed: {5 - dataset_status['completed_seeds'] - dataset_status['running_seeds']}/5")
    
    for seed_key in sorted(dataset_status['seeds'].keys()):
        seed_info = dataset_status['seeds'][seed_key]
        state = seed_info['state']
        epoch = seed_info.get('epoch', 'N/A')
        test_score = seed_info.get('test_score')
        val_loss = seed_info.get('val_loss')
        
        if state == 'completed':
            score_str = f", Test: {test_score:.4f}" if test_score is not None else ""
            print(f"    {seed_key}: ✅ Completed (Epoch {epoch}{score_str})")
        elif state == 'running':
            loss_str = f", Val Loss: {val_loss:.4f}" if val_loss is not None else ""
            print(f"    {seed_key}: 🔄 Running (Epoch {epoch}{loss_str})")
        elif state == 'pending':
            print(f"    {seed_key}: ⏳ Pending")
        elif state == 'error':
            error = seed_info.get('error', 'Unknown error')
            print(f"    {seed_key}: ❌ Error: {error}")
EOF
    
    echo "📄 Status exported to: $export_file"
    LAST_EXPORT_TIME=$(date '+%H:%M:%S')
}

    echo "📊 Enhanced Training Monitor for train_edmpnn.sh and train_edmpnn_new.sh"
    echo "🎯 Mode: Active Training Only (auto-detecting datasets from running training scripts)"
    echo "🔄 Refresh interval: ${INTERVAL} seconds"
    echo "⌨️  Controls: s=switch mode, p=pause, e=export, q=quit"
    echo "💡 Usage: ./monitor_training.sh [interval]"
    echo "=========================================="
    echo ""

while true; do
    # Check for keypress
    KEY=$(read_key)
    case "$KEY" in
        s|S)
            if [ "$DISPLAY_MODE" = "detailed" ]; then
                DISPLAY_MODE="compact"
            else
                DISPLAY_MODE="detailed"
            fi
            ;;
        p|P)
            PAUSED=$([ "$PAUSED" = true ] && echo false || echo true)
            ;;
        e|E)
            export_status
            sleep 1
            ;;
        q|Q)
            echo ""
            echo "👋 Exiting monitor..."
            exit 0
            ;;
    esac
    
    if [ "$PAUSED" = true ]; then
        clear
        echo "📊 Training Monitor - PAUSED - $(date '+%Y-%m-%d %H:%M:%S')"
        echo "Press 'p' to resume, 'q' to quit"
        sleep 1
        continue
    fi
    
    clear
    echo "📊 Training Real-time Monitor - $(date '+%Y-%m-%d %H:%M:%S')"
    if [ "$DISPLAY_MODE" = "compact" ]; then
        echo "Mode: COMPACT | Press 's' for detailed, 'p' to pause, 'e' to export, 'q' to quit"
    else
        echo "Mode: DETAILED | Press 's' for compact, 'p' to pause, 'e' to export, 'q' to quit"
    fi
    if [ -n "$LAST_EXPORT_TIME" ]; then
        echo "Last export: $LAST_EXPORT_TIME"
    fi
    echo "=========================================="
    echo ""
    
    # Detect running training processes (both versions)
    TRAINING_PIDS_OLD=$(pgrep -f "train_edmpnn.py" 2>/dev/null | wc -l)
    TRAINING_PIDS_NEW=$(pgrep -f "train_edmpnn_new.py" 2>/dev/null | wc -l)
    TRAINING_PIDS=$((TRAINING_PIDS_OLD + TRAINING_PIDS_NEW))
    echo "🖥️  Active Training Processes: $TRAINING_PIDS (train_edmpnn.py: $TRAINING_PIDS_OLD, train_edmpnn_new.py: $TRAINING_PIDS_NEW)"
    echo ""
    
    # Get comprehensive training status
    DISPLAY_MODE_PY="$DISPLAY_MODE" python3 << EOF
import os
import json
import glob
from datetime import datetime
from pathlib import Path

# Get display mode from environment
DISPLAY_MODE = os.environ.get('DISPLAY_MODE_PY', 'detailed')

# ANSI color codes
class Colors:
    GREEN = '\033[92m'      # Bright green for COMPLETE
    YELLOW = '\033[93m'     # Bright yellow for RUNNING
    RED = '\033[91m'        # Bright red for FAIL/ERROR
    BLUE = '\033[94m'       # Bright blue for PENDING
    CYAN = '\033[96m'       # Bright cyan for other states
    RESET = '\033[0m'       # Reset color
    BOLD = '\033[1m'        # Bold text

def get_training_progress(dataset_name, seed, checkpoint_suffix="_optuna_final"):
    """Get current epoch and loss from training_progress.json"""
    progress_file = f"checkpoints/{dataset_name}{checkpoint_suffix}/seed{seed}/training_progress.json"
    
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'r') as f:
                progress = json.load(f)
                return {
                    'epoch': progress.get('epoch', 0),
                    'val_loss': progress.get('val_loss'),
                    'train_loss': progress.get('train_loss'),
                    'timestamp': progress.get('timestamp')
                }
        except Exception as e:
            return {'error': str(e)}
    return None

def get_best_metric_from_checkpoint(dataset_name, seed, primary_metric, checkpoint_suffix="_optuna_final"):
    """Get best metric from checkpoint file or training history"""
    # Try checkpoint file first (most up-to-date for running training)
    checkpoint_file = f"checkpoints/{dataset_name}{checkpoint_suffix}/seed{seed}/best_model.pth"
    if os.path.exists(checkpoint_file):
        try:
            import torch
            checkpoint = torch.load(checkpoint_file, map_location='cpu', weights_only=False)
            
            # Map primary_metric to checkpoint keys
            # Note: checkpoint stores best_auroc, best_f1, best_pr_auc, etc.
            metric_key_map = {
                'roc_auc': 'best_auroc',
                'pr_auc': 'best_pr_auc',
                'f1': 'best_f1',
                'mae': 'best_mae',  # For regression, might not exist, use val_loss as fallback
                'spearman': 'best_spearman'  # For regression, might not exist
            }
            
            # Try direct key mapping first
            if primary_metric in metric_key_map:
                key = metric_key_map[primary_metric]
                if key in checkpoint:
                    value = checkpoint[key]
                    # Ensure it's a valid number (including 0.0)
                    if value is not None and isinstance(value, (int, float)):
                        return float(value)
            
            # Try best_metrics dictionary if exists
            best_metrics = checkpoint.get('best_metrics', {})
            if best_metrics and primary_metric in best_metrics:
                value = best_metrics[primary_metric]
                if value is not None and isinstance(value, (int, float)):
                    return float(value)
            
            # For regression tasks (mae, spearman), checkpoint may not have best_mae/best_spearman
            # For MAE: val_loss in checkpoint is typically the MAE (lower is better)
            # For Spearman: need to check if there's a spearman value stored
            if primary_metric == 'mae':
                # For MAE, val_loss in checkpoint is typically the best MAE so far
                if 'val_loss' in checkpoint:
                    value = checkpoint['val_loss']
                    if value is not None and isinstance(value, (int, float)):
                        return float(value)
                # Also try best_val_loss if available
                if 'best_val_loss' in checkpoint:
                    value = checkpoint['best_val_loss']
                    if value is not None and isinstance(value, (int, float)):
                        return float(value)
            
            # For Spearman, check val_metrics if available
            if primary_metric == 'spearman':
                val_metrics = checkpoint.get('val_metrics', {})
                if 'spearman' in val_metrics:
                    value = val_metrics['spearman']
                    if value is not None and isinstance(value, (int, float)):
                        return float(value)
            
            # For other regression metrics, check val_metrics
            val_metrics = checkpoint.get('val_metrics', {})
            if primary_metric in val_metrics:
                value = val_metrics[primary_metric]
                if value is not None and isinstance(value, (int, float)):
                    return float(value)
            
            # Fallback: try best_val_score (might be used for some metrics)
            if 'best_val_score' in checkpoint:
                value = checkpoint['best_val_score']
                if value is not None and isinstance(value, (int, float)):
                    return float(value)
        except Exception:
            pass
    
    # Fallback: try training_history.json
    history_file = f"checkpoints/{dataset_name}{checkpoint_suffix}/seed{seed}/training_history.json"
    if os.path.exists(history_file):
        try:
            with open(history_file, 'r') as f:
                history = json.load(f)
                best_metrics = history.get('best_metrics', {})
                if best_metrics and primary_metric in best_metrics:
                    return best_metrics[primary_metric]
                # For MAE regression, best_val_loss in history is the best MAE
                if primary_metric == 'mae' and 'best_val_loss' in history:
                    value = history['best_val_loss']
                    if value is not None and isinstance(value, (int, float)) and value != float('inf'):
                        return float(value)
                if 'best_val_score' in history:
                    return history['best_val_score']
        except Exception:
            pass
    
    return None

def get_process_start_time(pid):
    """Get process start time from /proc/PID/stat"""
    try:
        stat_file = f"/proc/{pid}/stat"
        if os.path.exists(stat_file):
            with open(stat_file, 'r') as f:
                stat_data = f.read().split()
                # Process start time is at index 21 (0-indexed)
                # It's in clock ticks since boot
                start_time_ticks = int(stat_data[21])
                
                # Get system uptime to calculate actual start time
                with open('/proc/uptime', 'r') as uptime_file:
                    uptime_seconds = float(uptime_file.read().split()[0])
                    # Clock ticks per second is typically 100 on Linux
                    clock_ticks_per_second = 100
                    
                    # Calculate start time
                    # start_time_ticks / clock_ticks_per_second gives seconds since boot
                    # uptime_seconds - (start_time_ticks / clock_ticks_per_second) gives time since process started
                    process_uptime = start_time_ticks / clock_ticks_per_second
                    start_time = datetime.now().timestamp() - (uptime_seconds - process_uptime)
                    return start_time
    except Exception:
        pass
    return None

def get_training_history(dataset_name, seed, checkpoint_suffix="_optuna_final"):
    """Get completed training results from training_history.json"""
    history_file = f"checkpoints/{dataset_name}{checkpoint_suffix}/seed{seed}/training_history.json"
    
    if os.path.exists(history_file):
        try:
            with open(history_file, 'r') as f:
                history = json.load(f)
                return history
        except Exception as e:
            return {'error': str(e)}
    return None

def get_primary_metric(dataset_name):
    """Get primary metric for dataset"""
    try:
        import yaml
        with open('configs/dataset_primary_metrics.yaml', 'r') as f:
            config = yaml.safe_load(f)
        settings = config.get('dataset_primary_metrics', {}).get(dataset_name, {})
        return settings.get('primary_metric', 'roc_auc')
    except:
        # Fallback to hardcoded list
        regression_mae = ['caco2_wang', 'ld50_zhu', 'lipophilicity_astrazeneca', 'ppbr_az', 'solubility_aqsoldb']
        regression_spearman = ['clearance_hepatocyte_az', 'clearance_microsome_az', 'half_life_obach', 'vdss_lombardo']
        pr_auc = ['cyp2c9_substrate_carbonmangels', 'cyp2c9_veith', 'cyp2d6_substrate_carbonmangels', 'cyp2d6_veith', 'cyp3a4_veith']
        
        if dataset_name in regression_mae:
            return 'mae'
        elif dataset_name in regression_spearman:
            return 'spearman'
        elif dataset_name in pr_auc:
            return 'pr_auc'
        else:
            return 'roc_auc'

def get_running_processes_gpu_mapping():
    """Get GPU mapping and process start times for running training processes"""
    import subprocess
    import re
    import os
    
    gpu_mapping = {}  # {(dataset, seed, version): gpu_id}
    process_info = {}  # {(dataset, seed, version): {'gpu_id': int, 'pid': int, 'start_time': float, 'version': str}}
    
    def get_parent_process_name(pid):
        """Get parent process name by checking PPID"""
        try:
            # Read /proc/PID/stat to get PPID (parent process ID)
            stat_file = f"/proc/{pid}/stat"
            if os.path.exists(stat_file):
                with open(stat_file, 'r') as f:
                    stat_data = f.read().split()
                    # PPID is at index 3 (0-indexed)
                    if len(stat_data) > 3:
                        ppid = int(stat_data[3])
                        # Read parent process cmdline
                        ppid_cmdline = f"/proc/{ppid}/cmdline"
                        if os.path.exists(ppid_cmdline):
                            with open(ppid_cmdline, 'rb') as f:
                                cmdline = f.read().decode('utf-8', errors='ignore')
                                cmdline_parts = cmdline.split('\x00')
                                cmdline_str = ' '.join(cmdline_parts)
                                if 'train_edmpnn_new.sh' in cmdline_str:
                                    return 'new'
                                elif 'train_edmpnn.sh' in cmdline_str:
                                    return 'old'
        except Exception:
            pass
        return None
    
    try:
        # Get all train_edmpnn.py and train_edmpnn_new.py process PIDs
        result_old = subprocess.run(
            ['pgrep', '-f', 'train_edmpnn.py'],
            capture_output=True,
            text=True,
            check=False
        )
        result_new = subprocess.run(
            ['pgrep', '-f', 'train_edmpnn_new.py'],
            capture_output=True,
            text=True,
            check=False
        )
        
        # Combine results
        pids_old = [pid.strip() for pid in result_old.stdout.strip().split('\n') if pid.strip()] if result_old.returncode == 0 else []
        pids_new = [pid.strip() for pid in result_new.stdout.strip().split('\n') if pid.strip()] if result_new.returncode == 0 else []
        
        # Process train_edmpnn.py processes (both old and new versions use this)
        for pid in pids_old:
            # Determine version by checking parent process
            parent_version = get_parent_process_name(pid)
            if parent_version:
                version = parent_version
            else:
                # Fallback: check if train_edmpnn_new.sh is running
                try:
                    result_sh_new = subprocess.run(
                        ['pgrep', '-f', 'train_edmpnn_new.sh'],
                        capture_output=True,
                        text=True,
                        check=False
                    )
                    train_edmpnn_new_sh_running = (result_sh_new.returncode == 0 and result_sh_new.stdout.strip() != '')
                    version = "new" if train_edmpnn_new_sh_running else "old"
                except Exception:
                    version = "old"  # Default to old if cannot determine
            try:
                # Read command line first to check if it's actually old version
                cmdline_file = f"/proc/{pid}/cmdline"
                if not os.path.exists(cmdline_file):
                    continue
                
                with open(cmdline_file, 'rb') as f:
                    cmdline = f.read().decode('utf-8', errors='ignore')
                    cmdline_parts = cmdline.split('\x00')
                    cmdline_str = ' '.join(cmdline_parts)
                
                # Skip train_edmpnn_new.py (new version)
                if 'train_edmpnn_new.py' in cmdline_str:
                    continue
                
                # Read process environment from /proc/PID/environ
                environ_file = f"/proc/{pid}/environ"
                gpu_id = None
                if os.path.exists(environ_file):
                    with open(environ_file, 'rb') as f:
                        env_data = f.read().decode('utf-8', errors='ignore')
                        env_vars = dict(item.split('=', 1) for item in env_data.split('\x00') if '=' in item)
                        
                        # Get CUDA_VISIBLE_DEVICES
                        cuda_devices = env_vars.get('CUDA_VISIBLE_DEVICES', '')
                        if cuda_devices:
                            gpu_id = int(cuda_devices.split(',')[0]) if cuda_devices.split(',')[0].isdigit() else None
                
                if gpu_id is None:
                    cuda_cmd_match = re.search(r'CUDA_VISIBLE_DEVICES=(\d+)', cmdline_str)
                    if cuda_cmd_match:
                        gpu_id = int(cuda_cmd_match.group(1))
                
                # Check save_dir to filter out Optuna search processes
                # Only include final training processes (save_dir contains _optuna_final but not optuna_mod)
                save_dir_match = re.search(r'--save_dir\s+(\S+)', cmdline_str)
                if save_dir_match:
                    save_dir = save_dir_match.group(1)
                    # Filter out Optuna search processes: must contain _optuna_final and not contain optuna_mod
                    if 'optuna_mod' in save_dir or '_optuna_final' not in save_dir:
                        continue  # Skip Optuna worker or other purposes
                
                dataset_match = re.search(r'--tdc_dataset\s+(\S+)', cmdline_str)
                seed_match = re.search(r'--tdc_seed\s+(\d+)', cmdline_str)
                
                if dataset_match and seed_match:
                    dataset = dataset_match.group(1)
                    seed = int(seed_match.group(1))
                    key = (dataset, seed, version)
                    
                    # Record GPU mapping only if gpu_id is available
                    if gpu_id is not None:
                        gpu_mapping[key] = gpu_id
                    
                    # Always record process info (even without GPU ID) to calculate elapsed_time
                    start_time = get_process_start_time(pid)
                    process_info[key] = {
                        'gpu_id': gpu_id,  # Can be None
                        'pid': pid,
                        'start_time': start_time,
                        'version': version
                    }
            except (ValueError, FileNotFoundError, PermissionError):
                continue
        
        # Process new version
        for pid in pids_new:
            version = "new"
            try:
                # Read process environment from /proc/PID/environ
                environ_file = f"/proc/{pid}/environ"
                gpu_id = None
                if os.path.exists(environ_file):
                    with open(environ_file, 'rb') as f:
                        env_data = f.read().decode('utf-8', errors='ignore')
                        env_vars = dict(item.split('=', 1) for item in env_data.split('\x00') if '=' in item)
                        
                        # Get CUDA_VISIBLE_DEVICES
                        cuda_devices = env_vars.get('CUDA_VISIBLE_DEVICES', '')
                        if cuda_devices:
                            gpu_id = int(cuda_devices.split(',')[0]) if cuda_devices.split(',')[0].isdigit() else None
                
                # Read command line from /proc/PID/cmdline
                cmdline_file = f"/proc/{pid}/cmdline"
                if os.path.exists(cmdline_file):
                    with open(cmdline_file, 'rb') as f:
                        cmdline = f.read().decode('utf-8', errors='ignore')
                        cmdline_parts = cmdline.split('\x00')
                        cmdline_str = ' '.join(cmdline_parts)
                        
                        if gpu_id is None:
                            cuda_cmd_match = re.search(r'CUDA_VISIBLE_DEVICES=(\d+)', cmdline_str)
                            if cuda_cmd_match:
                                gpu_id = int(cuda_cmd_match.group(1))
                        
                        # Check save_dir to filter out Optuna search processes
                        # Only include final training processes (save_dir contains _optuna_final or _optuna_final_new but not optuna_mod_new)
                        save_dir_match = re.search(r'--save_dir\s+(\S+)', cmdline_str)
                        if save_dir_match:
                            save_dir = save_dir_match.group(1)
                            # Filter out Optuna search processes: must contain _optuna_final or _optuna_final_new and not contain optuna_mod_new
                            # Note: train_edmpnn_new.py may use _optuna_final (not _optuna_final_new)
                            if 'optuna_mod_new' in save_dir or ('_optuna_final' not in save_dir and '_optuna_final_new' not in save_dir):
                                continue  # Skip Optuna worker or other purposes
                        
                        dataset_match = re.search(r'--tdc_dataset\s+(\S+)', cmdline_str)
                        seed_match = re.search(r'--tdc_seed\s+(\d+)', cmdline_str)
                        
                        if dataset_match and seed_match:
                            dataset = dataset_match.group(1)
                            seed = int(seed_match.group(1))
                            key = (dataset, seed, version)
                            
                            # Record GPU mapping only if gpu_id is available
                            if gpu_id is not None:
                                gpu_mapping[key] = gpu_id
                            
                            # Always record process info (even without GPU ID) to calculate elapsed_time
                            start_time = get_process_start_time(pid)
                            process_info[key] = {
                                'gpu_id': gpu_id,  # Can be None
                                'pid': pid,
                                'start_time': start_time,
                                'version': version
                            }
            except (ValueError, FileNotFoundError, PermissionError):
                continue
        
        return gpu_mapping, process_info
    except Exception:
        return {}, {}

def get_active_datasets_from_checkpoints():
    """Get active datasets by scanning checkpoint directories for recently updated training_progress.json files"""
    import time
    
    checkpoints_dir = Path("checkpoints")
    if not checkpoints_dir.exists():
        return {}
    
    active_datasets = {}  # {dataset: version} where version is 'old', 'new', or 'both'
    known_datasets = {
        'ames', 'bbb_martins', 'bioavailability_ma', 'caco2_wang',
        'clearance_hepatocyte_az', 'clearance_microsome_az',
        'cyp2c9_substrate_carbonmangels', 'cyp2c9_veith',
        'cyp2d6_substrate_carbonmangels', 'cyp2d6_veith',
        'cyp3a4_substrate_carbonmangels', 'cyp3a4_veith',
        'dili', 'half_life_obach', 'herg', 'hia_hou',
        'ld50_zhu', 'lipophilicity_astrazeneca', 'pgp_broccatelli',
        'ppbr_az', 'solubility_aqsoldb', 'vdss_lombardo'
    }
    
    # Only consider files updated in the last 30 minutes as "active"
    # This prevents detecting completed training as active
    current_time = time.time()
    active_threshold = 30 * 60  # 30 minutes in seconds
    
    def is_file_recently_updated(file_path):
        """Check if file exists and was updated recently"""
        if not file_path.exists():
            return False
        try:
            mtime = file_path.stat().st_mtime
            return (current_time - mtime) < active_threshold
        except Exception:
            return False
    
    # Method 1: Scan standard checkpoint structure: {dataset}_optuna_final or {dataset}_optuna_final_new
    for checkpoint_dir in checkpoints_dir.iterdir():
        if not checkpoint_dir.is_dir():
            continue
        
        dir_name = checkpoint_dir.name
        
        # Determine version and extract dataset name
        version = None
        dataset_name = None
        
        if dir_name.endswith("_optuna_final_new"):
            version = "new"
            dataset_name = dir_name[:-17]  # Remove "_optuna_final_new"
        elif dir_name.endswith("_optuna_final"):
            version = "old"
            dataset_name = dir_name[:-13]  # Remove "_optuna_final"
        else:
            continue
        
        if not dataset_name or dataset_name not in known_datasets:
            continue
        
        # Check if any seed has recently updated training_progress.json (indicating active training)
        has_active_training = False
        for seed in range(1, 6):
            progress_file = checkpoint_dir / f"seed{seed}" / "training_progress.json"
            if is_file_recently_updated(progress_file):
                has_active_training = True
                break
        
        if has_active_training:
            if dataset_name not in active_datasets:
                active_datasets[dataset_name] = version
            elif active_datasets[dataset_name] != version:
                active_datasets[dataset_name] = 'both'
    
    # Method 2: Scan optuna_mod_new structure: optuna_mod_new/{dataset}/{trial_name}/seed{seed}/
    optuna_mod_dir = checkpoints_dir / "optuna_mod_new"
    if optuna_mod_dir.exists():
        for dataset_dir in optuna_mod_dir.iterdir():
            if not dataset_dir.is_dir():
                continue
            
            dataset_name = dataset_dir.name
            if dataset_name not in known_datasets:
                continue
            
            # Check if any trial/seed has recently updated training_progress.json
            has_active_training = False
            for trial_dir in dataset_dir.iterdir():
                if not trial_dir.is_dir():
                    continue
                for seed in range(1, 6):
                    progress_file = trial_dir / f"seed{seed}" / "training_progress.json"
                    if is_file_recently_updated(progress_file):
                        has_active_training = True
                        break
                if has_active_training:
                    break
            
            if has_active_training:
                # optuna_mod_new is considered "new" version
                if dataset_name not in active_datasets:
                    active_datasets[dataset_name] = 'new'
                elif active_datasets[dataset_name] == 'old':
                    active_datasets[dataset_name] = 'both'
    
    return active_datasets

def get_active_datasets_from_train_script():
    """Get list of datasets currently being trained by train_edmpnn.sh and train_edmpnn_new.sh"""
    import subprocess
    import re
    
    active_datasets = {}  # {dataset: version} where version is 'old', 'new', or 'both'
    
    # Known TDC dataset names (from train_edmpnn.sh and train_edmpnn_new.sh)
    known_datasets = {
        'ames', 'bbb_martins', 'bioavailability_ma', 'caco2_wang',
        'clearance_hepatocyte_az', 'clearance_microsome_az',
        'cyp2c9_substrate_carbonmangels', 'cyp2c9_veith',
        'cyp2d6_substrate_carbonmangels', 'cyp2d6_veith',
        'cyp3a4_substrate_carbonmangels', 'cyp3a4_veith',
        'dili', 'half_life_obach', 'herg', 'hia_hou',
        'ld50_zhu', 'lipophilicity_astrazeneca', 'pgp_broccatelli',
        'ppbr_az', 'solubility_aqsoldb', 'vdss_lombardo'
    }
    
    # Map common variations (e.g., "half-life" -> "half_life_obach")
    dataset_variations = {
        'half-life': 'half_life_obach',
        'half_life': 'half_life_obach',
    }
    
    def parse_script_args(script_name, line):
        """Parse arguments from a script command line"""
        datasets_found = set()
        
        # Extract everything after script name
        parts = line.split(script_name)
        if len(parts) < 2:
            return datasets_found
        
        args_part = parts[1].strip()
        if not args_part:
            # No arguments - training all datasets, but we can't know which ones
            return datasets_found
        
        # Parse arguments
        args = args_part.split()
        skip_next = False
        
        for arg in args:
            if skip_next:
                skip_next = False
                continue
            
            # Handle --exclude flag
            if arg in ['--exclude', '-x']:
                skip_next = True
                continue
            
            # Skip other flags
            if arg in ['-h', '--help', '-l', '--list'] or arg.startswith('-'):
                continue
            
            # Normalize dataset name (handle variations)
            normalized_arg = dataset_variations.get(arg, arg)
            
            # Check if it's a known dataset name
            if normalized_arg in known_datasets:
                datasets_found.add(normalized_arg)
        
        return datasets_found
    
    # PRIMARY METHOD: Detect from running Python processes (most reliable)
    # First, check if train_edmpnn_new.sh is running (to identify new version processes)
    train_edmpnn_new_sh_running = False
    try:
        result_sh_new = subprocess.run(
            ['pgrep', '-f', 'train_edmpnn_new.sh'],
            capture_output=True,
            text=True,
            check=False
        )
        train_edmpnn_new_sh_running = (result_sh_new.returncode == 0 and result_sh_new.stdout.strip() != '')
    except Exception:
        pass
    
    # Check train_edmpnn.py processes (both old and new versions use this script)
    try:
        result_py = subprocess.run(
            ['pgrep', '-af', 'train_edmpnn.py'],
            capture_output=True,
            text=True,
            check=False
        )
        
        if result_py.returncode == 0:
            for line in result_py.stdout.strip().split('\n'):
                if '--tdc_dataset' in line and 'train_edmpnn_new.py' not in line:
                    # Parse save_dir, only accept final training (contains _optuna_final and not optuna_mod)
                    save_dir_match = re.search(r'--save_dir\s+(\S+)', line)
                    if save_dir_match:
                        save_dir = save_dir_match.group(1)
                        # Filter Optuna search processes: must contain _optuna_final and not contain optuna_mod
                        if 'optuna_mod' in save_dir or '_optuna_final' not in save_dir:
                            continue  # Skip Optuna worker or other purposes
                    match = re.search(r'--tdc_dataset\s+(\S+)', line)
                    if match:
                        dataset = match.group(1)
                        # Normalize dataset name (handle variations)
                        normalized_dataset = dataset_variations.get(dataset, dataset)
                        if normalized_dataset in known_datasets:
                            # If train_edmpnn_new.sh is running, this is likely a new version process
                            # Otherwise, it's an old version process
                            version = 'new' if train_edmpnn_new_sh_running else 'old'
                            if normalized_dataset not in active_datasets:
                                active_datasets[normalized_dataset] = version
                            elif active_datasets[normalized_dataset] != version:
                                active_datasets[normalized_dataset] = 'both'
    except Exception:
        pass
    
    # Check new version train_edmpnn_new.py (if it exists in the future)
    try:
        result_py_new = subprocess.run(
            ['pgrep', '-af', 'train_edmpnn_new.py'],
            capture_output=True,
            text=True,
            check=False
        )
        
        if result_py_new.returncode == 0:
            for line in result_py_new.stdout.strip().split('\n'):
                if '--tdc_dataset' in line:
                    # Parse save_dir, only accept final training (contains _optuna_final or _optuna_final_new and not optuna_mod_new)
                    save_dir_match = re.search(r'--save_dir\s+(\S+)', line)
                    if save_dir_match:
                        save_dir = save_dir_match.group(1)
                        # Filter Optuna search processes: must contain _optuna_final or _optuna_final_new and not contain optuna_mod_new
                        # Note: train_edmpnn_new.py may use _optuna_final (not _optuna_final_new)
                        if 'optuna_mod_new' in save_dir or ('_optuna_final' not in save_dir and '_optuna_final_new' not in save_dir):
                            continue  # Skip Optuna worker or other purposes
                    match = re.search(r'--tdc_dataset\s+(\S+)', line)
                    if match:
                        dataset = match.group(1)
                        # Normalize dataset name (handle variations)
                        normalized_dataset = dataset_variations.get(dataset, dataset)
                        if normalized_dataset in known_datasets:
                            if normalized_dataset not in active_datasets:
                                active_datasets[normalized_dataset] = 'new'
                            elif active_datasets[normalized_dataset] == 'old':
                                active_datasets[normalized_dataset] = 'both'
    except Exception:
        pass
    
    # SECONDARY METHOD: Try to parse from shell script processes (less reliable)
    # Check old version train_edmpnn.sh
    try:
        result_old = subprocess.run(
            ['pgrep', '-af', 'train_edmpnn.sh'],
            capture_output=True,
            text=True,
            check=False
        )
        
        if result_old.returncode == 0:
            for line in result_old.stdout.strip().split('\n'):
                if not line.strip() or 'train_edmpnn.sh' not in line or 'train_edmpnn_new.sh' in line:
                    continue
                
                datasets = parse_script_args('train_edmpnn.sh', line)
                for dataset in datasets:
                    if dataset not in active_datasets:
                        active_datasets[dataset] = 'old'
                    elif active_datasets[dataset] == 'new':
                        active_datasets[dataset] = 'both'
    except Exception:
        pass
    
    # Check new version train_edmpnn_new.sh
    try:
        result_new = subprocess.run(
            ['pgrep', '-af', 'train_edmpnn_new.sh'],
            capture_output=True,
            text=True,
            check=False
        )
        
        if result_new.returncode == 0:
            for line in result_new.stdout.strip().split('\n'):
                if not line.strip() or 'train_edmpnn_new.sh' not in line:
                    continue
                
                datasets = parse_script_args('train_edmpnn_new.sh', line)
                for dataset in datasets:
                    if dataset not in active_datasets:
                        active_datasets[dataset] = 'new'
                    elif active_datasets[dataset] == 'old':
                        active_datasets[dataset] = 'both'
    except Exception:
        pass
    
    return active_datasets

def scan_training_status():
    """Scan training checkpoints and return comprehensive status for active datasets only"""
    checkpoints_dir = Path("checkpoints")
    if not checkpoints_dir.exists():
        return {}
    
    # PRIMARY METHOD: Get active datasets from running train_edmpnn.sh / train_edmpnn.py and train_edmpnn_new.sh / train_edmpnn_new.py
    process_datasets = get_active_datasets_from_train_script()  # {dataset: 'old'|'new'|'both'}
    
    # STRICT MODE: Only include Python processes that are truly "final training" (exclude optuna_mod)
    running_python_datasets = {}  # {dataset: 'old'|'new'|'both'}
    try:
        import subprocess
        import re
        
        dataset_variations = {
            'half-life': 'half_life_obach',
            'half_life': 'half_life_obach',
        }
        
        # Check old version train_edmpnn.py processes, must be final training with save_dir pointing to *_optuna_final
        result_old = subprocess.run(
            ['pgrep', '-af', 'train_edmpnn.py'],
            capture_output=True,
            text=True,
            check=False
        )
        if result_old.returncode == 0:
            for line in result_old.stdout.strip().split('\n'):
                if '--tdc_dataset' in line and 'train_edmpnn_new.py' not in line:
                    # Only accept processes with save_dir containing "_optuna_final" and not containing "optuna_mod"
                    save_dir_match = re.search(r'--save_dir\s+(\S+)', line)
                    if save_dir_match:
                        save_dir = save_dir_match.group(1)
                        if 'optuna_mod' in save_dir or '_optuna_final' not in save_dir:
                            continue  # 這是 Optuna 或其他用途，忽略
                    dataset_match = re.search(r'--tdc_dataset\s+(\S+)', line)
                    if dataset_match:
                        dataset = dataset_match.group(1)
                        normalized_dataset = dataset_variations.get(dataset, dataset)
                        
                        if normalized_dataset not in running_python_datasets:
                            running_python_datasets[normalized_dataset] = 'old'
                        elif running_python_datasets[normalized_dataset] == 'new':
                            running_python_datasets[normalized_dataset] = 'both'
        
        # Check new version train_edmpnn_new.py processes, must be final training with save_dir pointing to *_optuna_final_new
        result_new = subprocess.run(
            ['pgrep', '-af', 'train_edmpnn_new.py'],
            capture_output=True,
            text=True,
            check=False
        )
        if result_new.returncode == 0:
            for line in result_new.stdout.strip().split('\n'):
                if '--tdc_dataset' in line:
                    # Only accept processes with save_dir containing "_optuna_final" or "_optuna_final_new" and not containing "optuna_mod_new"
                    # Note: train_edmpnn_new.py may use _optuna_final (not _optuna_final_new)
                    save_dir_match = re.search(r'--save_dir\s+(\S+)', line)
                    if save_dir_match:
                        save_dir = save_dir_match.group(1)
                        if 'optuna_mod_new' in save_dir or ('_optuna_final' not in save_dir and '_optuna_final_new' not in save_dir):
                            continue  # 這是 Optuna 或其他用途，忽略
                    dataset_match = re.search(r'--tdc_dataset\s+(\S+)', line)
                    if dataset_match:
                        dataset = dataset_match.group(1)
                        normalized_dataset = dataset_variations.get(dataset, dataset)
                        
                        if normalized_dataset not in running_python_datasets:
                            running_python_datasets[normalized_dataset] = 'new'
                        elif running_python_datasets[normalized_dataset] == 'old':
                            running_python_datasets[normalized_dataset] = 'both'
    except:
        pass
    
    # active_datasets_dict:
    #   - Use train_edmpnn.sh / train_edmpnn_new.sh arguments as "monitoring list" (process_datasets)
    #   - Then overlay currently running Python processes (running_python_datasets)
    #   This way, cases like "dili, half_life_obach, herg" will all be displayed,
    #   where those not yet started / already finished but still in list will show as pending.
    active_datasets_dict = process_datasets.copy()
    for ds, ver in running_python_datasets.items():
        if ds not in active_datasets_dict:
            active_datasets_dict[ds] = ver
        elif active_datasets_dict[ds] != ver and ver != 'both':
            # If versions differ, merge to 'both'
            if (active_datasets_dict[ds] == 'old' and ver == 'new') or \
               (active_datasets_dict[ds] == 'new' and ver == 'old'):
                active_datasets_dict[ds] = 'both'
    
    # If no active training (neither shell nor python) detected, return empty or show message
    if not active_datasets_dict:
        return {'_no_active_training': True}
    
    # Get GPU mapping and process info for running processes
    gpu_mapping, process_info = get_running_processes_gpu_mapping()
    
    status = {}
    
    # Only scan datasets that are currently being trained
    for dataset_name, version_info in active_datasets_dict.items():
        # Determine checkpoint directory to check based on version info
        # Note: train_edmpnn_new.sh uses _optuna_final (not _optuna_final_new)
        if version_info == 'old':
            checkpoint_suffixes = ['_optuna_final']
        elif version_info == 'new':
            # train_edmpnn_new.sh uses _optuna_final (not _optuna_final_new)
            # Only check _optuna_final_new if the directory actually exists
            checkpoint_suffixes = ['_optuna_final']
            # Check if _optuna_final_new directory exists, if so, also monitor it
            optuna_final_new_dir = checkpoints_dir / f"{dataset_name}_optuna_final_new"
            if optuna_final_new_dir.exists():
                checkpoint_suffixes.append('_optuna_final_new')
        else:  # 'both'
            checkpoint_suffixes = ['_optuna_final']
            # Check if _optuna_final_new directory exists, if so, also monitor it
            optuna_final_new_dir = checkpoints_dir / f"{dataset_name}_optuna_final_new"
            if optuna_final_new_dir.exists():
                checkpoint_suffixes.append('_optuna_final_new')
        
        # We now only care about final training ({dataset}_optuna_final / {dataset}_optuna_final_new),
        # No longer display trial status during optuna_mod_new search process, avoid duplicate records for same dataset.
        check_optuna_mod_new = False
        
        # Check each version's checkpoint directory
        for checkpoint_suffix in checkpoint_suffixes:
            dataset_dir = checkpoints_dir / f"{dataset_name}{checkpoint_suffix}"
            
            # Skip _optuna_final_new if directory doesn't exist and no running processes
            # (train_edmpnn_new.sh doesn't use _optuna_final_new, so don't show pending for it)
            if checkpoint_suffix == "_optuna_final_new" and not dataset_dir.exists():
                # Check if there are any running processes for this dataset with this suffix
                has_running_process = False
                for seed in range(1, 6):
                    # Check both old and new versions
                    for ver in ["old", "new"]:
                        process_key = (dataset_name, seed, ver)
                        if process_key in process_info or process_key in gpu_mapping:
                            has_running_process = True
                            break
                    if has_running_process:
                        break
                # If no running process and directory doesn't exist, skip this checkpoint_suffix
                if not has_running_process:
                    continue
            
            # If directory not yet created (training just started, checkpoint not yet written), still show as 5 seeds pending
            if not dataset_dir.exists():
                # Determine version label based on version_info and checkpoint_suffix
                # If version_info == 'new', even using _optuna_final is "new" version
                if checkpoint_suffix == "_optuna_final_new":
                    version_label = "new"
                elif version_info == 'new':
                    # train_edmpnn_new.sh uses _optuna_final, but belongs to "new" version
                    version_label = "new"
                else:
                    version_label = "old"
                
                # If this dataset has both old and new versions, need to distinguish display
                if version_info == 'both':
                    if checkpoint_suffix == "_optuna_final":
                        # Need to determine old or new based on process info
                        has_new_process = False
                        for seed in range(1, 6):
                            process_key = (dataset_name, seed, "new")
                            if process_key in process_info or process_key in gpu_mapping:
                                has_new_process = True
                                break
                        
                        if has_new_process:
                            status_key = f"{dataset_name}_new{checkpoint_suffix}"
                            version_label = "new"
                        else:
                            status_key = f"{dataset_name}_old{checkpoint_suffix}"
                            version_label = "old"
                    else:
                        status_key = f"{dataset_name}_new{checkpoint_suffix}"
                else:
                    status_key = f"{dataset_name}{checkpoint_suffix}"
                if status_key not in status:
                    status[status_key] = {
                        'dataset_name': dataset_name,
                        'version': version_label,
                        'checkpoint_suffix': checkpoint_suffix,
                        'seeds': {},
                        'completed_seeds': 0,
                        'running_seeds': 0,
                        'pending_seeds': 5,
                        'failed_seeds': 0,
                        'test_scores': [],
                        'primary_metric': get_primary_metric(dataset_name)
                    }
                    # Create 5 pending seed entries
                    for seed in range(1, 6):
                        seed_key = f"seed{seed}"
                        status[status_key]['seeds'][seed_key] = {
                            'state': 'pending',
                            'epoch': None,
                            'val_loss': None,
                            'train_loss': None,
                            'test_score': None,
                            'timestamp': None,
                            'elapsed_time': None,
                            'best_epoch': None,
                            'gpu_id': None,
                            'best_metric': None,
                            'version': version_label
                        }
                continue
            
            # Use a unique key for each dataset+version combination
            status_key = f"{dataset_name}{checkpoint_suffix}"
            # Determine version label based on version_info and checkpoint_suffix
            # If version_info == 'new', even using _optuna_final is "new" version
            if checkpoint_suffix == "_optuna_final_new":
                version_label = "new"
            elif version_info == 'new':
                # train_edmpnn_new.sh uses _optuna_final, but belongs to "new" version
                version_label = "new"
            else:
                version_label = "old"
            
            # If this dataset has both old and new versions, need to distinguish display
            if version_info == 'both':
                if checkpoint_suffix == "_optuna_final":
                    # Need to determine old or new based on process info
                    # If there's train_edmpnn_new.py process using this directory, it's new
                    # Otherwise it's old
                    # Here we first check if there's a new version process
                    has_new_process = False
                    for seed in range(1, 6):
                        process_key = (dataset_name, seed, "new")
                        if process_key in process_info or process_key in gpu_mapping:
                            has_new_process = True
                            break
                    
                    if has_new_process:
                        status_key = f"{dataset_name}_new{checkpoint_suffix}"
                        version_label = "new"
                    else:
                        status_key = f"{dataset_name}_old{checkpoint_suffix}"
                        version_label = "old"
                else:
                    status_key = f"{dataset_name}_new{checkpoint_suffix}"
            
            if status_key not in status:
                status[status_key] = {
                    'dataset_name': dataset_name,
                    'version': version_label,
                    'checkpoint_suffix': checkpoint_suffix,
                    'seeds': {},
                    'completed_seeds': 0,
                    'running_seeds': 0,
                    'pending_seeds': 0,
                    'failed_seeds': 0,
                    'test_scores': [],
                    'primary_metric': get_primary_metric(dataset_name)
                }
            
            # Check each seed (1-5)
            for seed in range(1, 6):
                seed_key = f"seed{seed}"
                seed_dir = dataset_dir / f"seed{seed}"
                
                history_file = seed_dir / "training_history.json"
                progress_file = seed_dir / "training_progress.json"
                
                seed_status = {
                    'state': 'pending',
                    'epoch': None,
                    'val_loss': None,
                    'train_loss': None,
                    'test_score': None,
                    'timestamp': None,
                    'elapsed_time': None,
                    'best_epoch': None,
                    'gpu_id': None,
                    'best_metric': None,
                    'version': version_label
                }
                
                # Check if this seed is running and get GPU from process mapping
                # Use version-specific key
                process_key = (dataset_name, seed, version_label)
                is_process_running = process_key in gpu_mapping or process_key in process_info
                
                if is_process_running:
                    if process_key in gpu_mapping:
                        seed_status['gpu_id'] = gpu_mapping[process_key]
                    
                    # Get process start time for elapsed calculation
                    if process_key in process_info:
                        proc_info = process_info[process_key]
                        if proc_info.get('start_time'):
                            elapsed = datetime.now().timestamp() - proc_info['start_time']
                            hours = int(elapsed / 3600)
                            minutes = int((elapsed % 3600) / 60)
                            seed_status['elapsed_time'] = f"{hours}h {minutes}m"
                
                # Priority: Check if process is running OR progress_file is newer than history_file
                # This handles the case where training was restarted after completion
                progress_file_newer = False
                progress_file_recent = False
                if progress_file.exists() and history_file.exists():
                    try:
                        import time
                        progress_mtime = progress_file.stat().st_mtime
                        history_mtime = history_file.stat().st_mtime
                        current_time = time.time()
                        # If progress_file is newer than history_file, it's definitely running
                        progress_file_newer = progress_mtime > history_mtime
                        # If progress_file is recent (within 30 min) AND newer than history, it's running
                        # But if history is newer, even if progress is recent, it's completed
                        progress_file_recent = (current_time - progress_mtime) < 1800 and progress_file_newer
                    except Exception:
                        pass
                elif progress_file.exists():
                    # Only progress file exists, check if it's recent
                    try:
                        import time
                        progress_mtime = progress_file.stat().st_mtime
                        progress_file_recent = (time.time() - progress_mtime) < 1800
                    except Exception:
                        pass
                
                # If process is running OR (progress_file is newer than history_file AND recent), prioritize running state
                if is_process_running or progress_file_recent:
                    if progress_file.exists():
                        # Training in progress (process running or progress file is newer)
                        try:
                            progress = json.load(open(progress_file, 'r'))
                            seed_status['state'] = 'running'
                            seed_status['epoch'] = progress.get('epoch', 0)
                            seed_status['val_loss'] = progress.get('val_loss')
                            seed_status['train_loss'] = progress.get('train_loss')
                            seed_status['timestamp'] = progress.get('timestamp')
                            
                            # Get best metric from checkpoint (preferred) or progress file
                            primary_metric = status[status_key]['primary_metric']
                            best_metric = None
                            
                            # Try checkpoint first (most accurate)
                            try:
                                best_metric = get_best_metric_from_checkpoint(dataset_name, seed, primary_metric, checkpoint_suffix)
                            except Exception as e:
                                pass
                            
                            # Fallback: try to get from progress file
                            if best_metric is None:
                                # Check if progress file has best_metrics
                                progress_best_metrics = progress.get('best_metrics', {})
                                if primary_metric in progress_best_metrics:
                                    best_metric = progress_best_metrics[primary_metric]
                                # For MAE regression: val_loss in progress is typically the current MAE
                                # We can use it as approximation (though not necessarily the best)
                                elif primary_metric == 'mae' and 'val_loss' in progress:
                                    best_metric = progress['val_loss']
                                # Also check current val_metrics as fallback (for running training)
                                elif primary_metric in progress.get('val_metrics', {}):
                                    best_metric = progress['val_metrics'][primary_metric]
                                # Also try best_val_score if available
                                elif 'best_val_score' in progress:
                                    best_metric = progress['best_val_score']
                            
                            if best_metric is not None:
                                seed_status['best_metric'] = float(best_metric)
                            
                            # Elapsed time should already be set from process_info above
                            # If not set, try to calculate from process start time
                            if not seed_status.get('elapsed_time') and process_key in process_info:
                                proc_info = process_info[process_key]
                                if proc_info.get('start_time'):
                                    elapsed = datetime.now().timestamp() - proc_info['start_time']
                                    hours = int(elapsed / 3600)
                                    minutes = int((elapsed % 3600) / 60)
                                    seed_status['elapsed_time'] = f"{hours}h {minutes}m"
                            
                            # Fallback: if still no elapsed time, try to calculate from progress file timestamp
                            if not seed_status.get('elapsed_time'):
                                # Try to use timestamp from progress file
                                progress_timestamp = progress.get('timestamp')
                                if progress_timestamp:
                                    try:
                                        elapsed = datetime.now().timestamp() - float(progress_timestamp)
                                        hours = int(elapsed / 3600)
                                        minutes = int((elapsed % 3600) / 60)
                                        seed_status['elapsed_time'] = f"{hours}h {minutes}m"
                                    except (ValueError, TypeError):
                                        pass
                                
                                # Last fallback: use file modification time
                                if not seed_status.get('elapsed_time'):
                                    try:
                                        import time
                                        file_mtime = progress_file.stat().st_mtime
                                        elapsed = time.time() - file_mtime
                                        hours = int(elapsed / 3600)
                                        minutes = int((elapsed % 3600) / 60)
                                        seed_status['elapsed_time'] = f"{hours}h {minutes}m"
                                    except Exception:
                                        seed_status['elapsed_time'] = 'N/A'
                            
                            status[status_key]['running_seeds'] += 1
                        except Exception as e:
                            seed_status['state'] = 'error'
                            seed_status['error'] = str(e)
                            status[status_key]['failed_seeds'] += 1
                    else:
                        # Process is running but no progress file yet - mark as running
                        seed_status['state'] = 'running'
                        seed_status['epoch'] = 0
                        status[status_key]['running_seeds'] += 1
                elif history_file.exists():
                    # Training completed (no running process and history file exists)
                    try:
                        history = json.load(open(history_file, 'r'))
                        seed_status['state'] = 'completed'
                        seed_status['best_epoch'] = history.get('best_epoch', history.get('total_epochs', 'N/A'))
                        seed_status['epoch'] = seed_status['best_epoch']
                        
                        # Get test score
                        test_results = history.get('test_results', {})
                        primary_metric = status[status_key]['primary_metric']
                        
                        if primary_metric in test_results:
                            test_score = test_results[primary_metric]
                            seed_status['test_score'] = test_score
                            status[status_key]['test_scores'].append(test_score)
                        else:
                            # Fallback to any available metric
                            for metric in ['roc_auc', 'pr_auc', 'mae', 'spearman']:
                                if metric in test_results:
                                    seed_status['test_score'] = test_results[metric]
                                    status[status_key]['test_scores'].append(test_results[metric])
                                    break
                        
                        status[status_key]['completed_seeds'] += 1
                    except Exception as e:
                        seed_status['state'] = 'error'
                        seed_status['error'] = str(e)
                        status[status_key]['failed_seeds'] += 1
                elif progress_file.exists():
                    # Training in progress
                    try:
                        progress = json.load(open(progress_file, 'r'))
                        seed_status['state'] = 'running'
                        seed_status['epoch'] = progress.get('epoch', 0)
                        seed_status['val_loss'] = progress.get('val_loss')
                        seed_status['train_loss'] = progress.get('train_loss')
                        seed_status['timestamp'] = progress.get('timestamp')
                        
                        # Get best metric from checkpoint (preferred) or progress file
                        primary_metric = status[status_key]['primary_metric']
                        best_metric = None
                        
                        # Try checkpoint first (most accurate)
                        try:
                            best_metric = get_best_metric_from_checkpoint(dataset_name, seed, primary_metric, checkpoint_suffix)
                        except Exception as e:
                            pass
                        
                        # Fallback: try to get from progress file
                        if best_metric is None:
                            # Check if progress file has best_metrics
                            progress_best_metrics = progress.get('best_metrics', {})
                            if primary_metric in progress_best_metrics:
                                best_metric = progress_best_metrics[primary_metric]
                            # For MAE regression: val_loss in progress is typically the current MAE
                            # We can use it as approximation (though not necessarily the best)
                            elif primary_metric == 'mae' and 'val_loss' in progress:
                                best_metric = progress['val_loss']
                            # Also check current val_metrics as fallback (for running training)
                            elif primary_metric in progress.get('val_metrics', {}):
                                best_metric = progress['val_metrics'][primary_metric]
                            # Also try best_val_score if available
                            elif 'best_val_score' in progress:
                                best_metric = progress['best_val_score']
                        
                        if best_metric is not None:
                            seed_status['best_metric'] = float(best_metric)
                        
                        # Elapsed time should already be set from process_info above
                        # If not set, try to calculate from process start time
                        if not seed_status.get('elapsed_time') and process_key in process_info:
                            proc_info = process_info[process_key]
                            if proc_info.get('start_time'):
                                elapsed = datetime.now().timestamp() - proc_info['start_time']
                                hours = int(elapsed / 3600)
                                minutes = int((elapsed % 3600) / 60)
                                seed_status['elapsed_time'] = f"{hours}h {minutes}m"
                        
                        # Fallback: if still no elapsed time, try to calculate from progress file timestamp
                        if not seed_status.get('elapsed_time'):
                            # Try to use timestamp from progress file
                            progress_timestamp = progress.get('timestamp')
                            if progress_timestamp:
                                try:
                                    elapsed = datetime.now().timestamp() - float(progress_timestamp)
                                    hours = int(elapsed / 3600)
                                    minutes = int((elapsed % 3600) / 60)
                                    seed_status['elapsed_time'] = f"{hours}h {minutes}m"
                                except (ValueError, TypeError):
                                    pass
                            
                            # Last fallback: use file modification time
                            if not seed_status.get('elapsed_time'):
                                try:
                                    import time
                                    file_mtime = progress_file.stat().st_mtime
                                    elapsed = time.time() - file_mtime
                                    hours = int(elapsed / 3600)
                                    minutes = int((elapsed % 3600) / 60)
                                    seed_status['elapsed_time'] = f"{hours}h {minutes}m"
                                except Exception:
                                    seed_status['elapsed_time'] = 'N/A'
                        
                        status[status_key]['running_seeds'] += 1
                    except Exception as e:
                        seed_status['state'] = 'error'
                        seed_status['error'] = str(e)
                        status[status_key]['failed_seeds'] += 1
                else:
                    # No files found - pending
                    seed_status['state'] = 'pending'
                    status[status_key]['pending_seeds'] += 1
                
                status[status_key]['seeds'][seed_key] = seed_status
        
        # Also check optuna_mod_new structure for "new" version datasets
        if check_optuna_mod_new:
            optuna_mod_dataset_dir = checkpoints_dir / "optuna_mod_new" / dataset_name
            if optuna_mod_dataset_dir.exists():
                # Use a unique key for optuna_mod_new structure
                status_key = f"{dataset_name}_optuna_mod_new"
                version_label = "new"
                
                if status_key not in status:
                    status[status_key] = {
                        'dataset_name': dataset_name,
                        'version': version_label,
                        'checkpoint_suffix': '_optuna_mod_new',
                        'seeds': {},
                        'completed_seeds': 0,
                        'running_seeds': 0,
                        'pending_seeds': 0,
                        'failed_seeds': 0,
                        'test_scores': [],
                        'primary_metric': get_primary_metric(dataset_name)
                    }
                
                # Find the most recent trial directory for each seed
                # In optuna_mod_new, each trial has its own directory
                # We need to find the most recent active trial for each seed
                seed_to_trial = {}  # {seed: (trial_dir, is_active, mtime)}
                
                import time
                current_time = time.time()
                active_threshold = 30 * 60  # 30 minutes
                
                for trial_dir in optuna_mod_dataset_dir.iterdir():
                    if not trial_dir.is_dir():
                        continue
                    
                    for seed in range(1, 6):
                        seed_key = f"seed{seed}"
                        seed_dir = trial_dir / f"seed{seed}"
                        
                        history_file = seed_dir / "training_history.json"
                        progress_file = seed_dir / "training_progress.json"
                        
                        # Get the most recent file modification time
                        max_mtime = 0
                        is_active = False
                        
                        if progress_file.exists():
                            try:
                                mtime = progress_file.stat().st_mtime
                                max_mtime = max(max_mtime, mtime)
                                if (current_time - mtime) < active_threshold:
                                    is_active = True
                            except:
                                pass
                        
                        if history_file.exists():
                            try:
                                mtime = history_file.stat().st_mtime
                                max_mtime = max(max_mtime, mtime)
                            except:
                                pass
                        
                        # If this seed doesn't have a trial yet, or this trial is more recent
                        if seed not in seed_to_trial:
                            seed_to_trial[seed] = (trial_dir, is_active, max_mtime)
                        elif max_mtime > seed_to_trial[seed][2] or (is_active and not seed_to_trial[seed][1]):
                            # Update if this trial is more recent, or if it's active and previous wasn't
                            seed_to_trial[seed] = (trial_dir, is_active, max_mtime)
                
                # Process each seed
                for seed in range(1, 6):
                    seed_key = f"seed{seed}"
                    
                    if seed not in seed_to_trial:
                        # No trial found for this seed
                        seed_status = {
                            'state': 'pending',
                            'epoch': None,
                            'val_loss': None,
                            'train_loss': None,
                            'test_score': None,
                            'timestamp': None,
                            'elapsed_time': None,
                            'best_epoch': None,
                            'gpu_id': None,
                            'best_metric': None,
                            'version': version_label
                        }
                        status[status_key]['seeds'][seed_key] = seed_status
                        status[status_key]['pending_seeds'] += 1
                        continue
                    
                    trial_dir, is_active, _ = seed_to_trial[seed]
                    seed_dir = trial_dir / f"seed{seed}"
                    
                    history_file = seed_dir / "training_history.json"
                    progress_file = seed_dir / "training_progress.json"
                    
                    seed_status = {
                        'state': 'pending',
                        'epoch': None,
                        'val_loss': None,
                        'train_loss': None,
                        'test_score': None,
                        'timestamp': None,
                        'elapsed_time': None,
                        'best_epoch': None,
                        'gpu_id': None,
                        'best_metric': None,
                        'version': version_label
                    }
                    
                    # Check if this seed is running and get GPU from process mapping
                    process_key = (dataset_name, seed, version_label)
                    is_process_running = process_key in gpu_mapping or process_key in process_info
                    
                    if is_process_running:
                        if process_key in gpu_mapping:
                            seed_status['gpu_id'] = gpu_mapping[process_key]
                        if process_key in process_info:
                            proc_info = process_info[process_key]
                            if proc_info.get('start_time'):
                                elapsed = datetime.now().timestamp() - proc_info['start_time']
                                hours = int(elapsed / 3600)
                                minutes = int((elapsed % 3600) / 60)
                                seed_status['elapsed_time'] = f"{hours}h {minutes}m"
                    
                    # Priority: Check if process is running OR progress_file is newer than history_file OR is_active
                    # This handles the case where training was restarted after completion
                    progress_file_newer = False
                    progress_file_recent = False
                    if progress_file.exists() and history_file.exists():
                        try:
                            import time
                            progress_mtime = progress_file.stat().st_mtime
                            history_mtime = history_file.stat().st_mtime
                            current_time = time.time()
                            # If progress_file is newer than history_file, it's definitely running
                            progress_file_newer = progress_mtime > history_mtime
                            # If progress_file is recent (within 30 min) AND newer than history, it's running
                            # But if history is newer, even if progress is recent, it's completed
                            progress_file_recent = (current_time - progress_mtime) < 1800 and progress_file_newer
                        except Exception:
                            pass
                    elif progress_file.exists():
                        # Only progress file exists, check if it's recent
                        try:
                            import time
                            progress_mtime = progress_file.stat().st_mtime
                            progress_file_recent = (time.time() - progress_mtime) < 1800
                        except Exception:
                            pass
                    
                    # If process is running OR (progress_file is newer than history_file AND recent) OR is_active, prioritize running state
                    if is_process_running or progress_file_recent or is_active:
                        if progress_file.exists():
                            # Training in progress (process running or progress file is newer or active)
                            try:
                                progress = json.load(open(progress_file, 'r'))
                                seed_status['state'] = 'running'
                                seed_status['epoch'] = progress.get('epoch', 0)
                                seed_status['val_loss'] = progress.get('val_loss')
                                seed_status['train_loss'] = progress.get('train_loss')
                                seed_status['timestamp'] = progress.get('timestamp')
                                
                                # Get best metric from checkpoint
                                primary_metric = status[status_key]['primary_metric']
                                best_metric = None
                                
                                # For optuna_mod_new, checkpoint is in trial_dir/seed{seed}/best_model.pth
                                try:
                                    checkpoint_file = seed_dir / "best_model.pth"
                                    if checkpoint_file.exists():
                                        import torch
                                        checkpoint = torch.load(str(checkpoint_file), map_location='cpu', weights_only=False)
                                        
                                        metric_key_map = {
                                            'roc_auc': 'best_auroc',
                                            'pr_auc': 'best_pr_auc',
                                            'f1': 'best_f1',
                                            'mae': 'best_mae',
                                            'spearman': 'best_spearman'
                                        }
                                        
                                        if primary_metric in metric_key_map:
                                            key = metric_key_map[primary_metric]
                                            if key in checkpoint:
                                                value = checkpoint[key]
                                                if value is not None and isinstance(value, (int, float)):
                                                    best_metric = float(value)
                                        
                                        if best_metric is None:
                                            best_metrics = checkpoint.get('best_metrics', {})
                                            if best_metrics and primary_metric in best_metrics:
                                                best_metric = best_metrics[primary_metric]
                                        
                                        if best_metric is None and primary_metric == 'mae':
                                            if 'val_loss' in checkpoint:
                                                value = checkpoint['val_loss']
                                                if value is not None and isinstance(value, (int, float)):
                                                    best_metric = float(value)
                                        
                                        if best_metric is None and primary_metric == 'spearman':
                                            val_metrics = checkpoint.get('val_metrics', {})
                                            if 'spearman' in val_metrics:
                                                value = val_metrics['spearman']
                                                if value is not None and isinstance(value, (int, float)):
                                                    best_metric = float(value)
                                except:
                                    pass
                                
                                if best_metric is None:
                                    progress_best_metrics = progress.get('best_metrics', {})
                                    if primary_metric in progress_best_metrics:
                                        best_metric = progress_best_metrics[primary_metric]
                                    elif primary_metric == 'mae' and 'val_loss' in progress:
                                        best_metric = progress['val_loss']
                                    elif primary_metric in progress.get('val_metrics', {}):
                                        best_metric = progress['val_metrics'][primary_metric]
                                    elif 'best_val_score' in progress:
                                        best_metric = progress['best_val_score']
                                
                                if best_metric is not None:
                                    seed_status['best_metric'] = float(best_metric)
                                
                                if not seed_status.get('elapsed_time') and process_key in process_info:
                                    proc_info = process_info[process_key]
                                    if proc_info.get('start_time'):
                                        elapsed = datetime.now().timestamp() - proc_info['start_time']
                                        hours = int(elapsed / 3600)
                                        minutes = int((elapsed % 3600) / 60)
                                        seed_status['elapsed_time'] = f"{hours}h {minutes}m"
                                
                                # Fallback: if still no elapsed time, try to calculate from progress file timestamp
                                if not seed_status.get('elapsed_time'):
                                    # Try to use timestamp from progress file
                                    progress_timestamp = progress.get('timestamp')
                                    if progress_timestamp:
                                        try:
                                            elapsed = datetime.now().timestamp() - float(progress_timestamp)
                                            hours = int(elapsed / 3600)
                                            minutes = int((elapsed % 3600) / 60)
                                            seed_status['elapsed_time'] = f"{hours}h {minutes}m"
                                        except (ValueError, TypeError):
                                            pass
                                    
                                    # Last fallback: use file modification time
                                    if not seed_status.get('elapsed_time'):
                                        try:
                                            import time
                                            file_mtime = progress_file.stat().st_mtime
                                            elapsed = time.time() - file_mtime
                                            hours = int(elapsed / 3600)
                                            minutes = int((elapsed % 3600) / 60)
                                            seed_status['elapsed_time'] = f"{hours}h {minutes}m"
                                        except Exception:
                                            seed_status['elapsed_time'] = 'N/A'
                                
                                status[status_key]['running_seeds'] += 1
                            except Exception as e:
                                seed_status['state'] = 'error'
                                seed_status['error'] = str(e)
                                status[status_key]['failed_seeds'] += 1
                        else:
                            # Process is running but no progress file yet - mark as running
                            seed_status['state'] = 'running'
                            seed_status['epoch'] = 0
                            status[status_key]['running_seeds'] += 1
                    elif history_file.exists():
                        # Training completed (no running process and history file exists)
                        try:
                            history = json.load(open(history_file, 'r'))
                            seed_status['state'] = 'completed'
                            seed_status['best_epoch'] = history.get('best_epoch', history.get('total_epochs', 'N/A'))
                            seed_status['epoch'] = seed_status['best_epoch']
                            
                            # Get test score
                            test_results = history.get('test_results', {})
                            primary_metric = status[status_key]['primary_metric']
                            
                            if primary_metric in test_results:
                                test_score = test_results[primary_metric]
                                seed_status['test_score'] = test_score
                                status[status_key]['test_scores'].append(test_score)
                            else:
                                for metric in ['roc_auc', 'pr_auc', 'mae', 'spearman']:
                                    if metric in test_results:
                                        seed_status['test_score'] = test_results[metric]
                                        status[status_key]['test_scores'].append(test_results[metric])
                                        break
                            
                            status[status_key]['completed_seeds'] += 1
                        except Exception as e:
                            seed_status['state'] = 'error'
                            seed_status['error'] = str(e)
                            status[status_key]['failed_seeds'] += 1
                    elif progress_file.exists():
                        # Training in progress
                        try:
                            progress = json.load(open(progress_file, 'r'))
                            seed_status['state'] = 'running'
                            seed_status['epoch'] = progress.get('epoch', 0)
                            seed_status['val_loss'] = progress.get('val_loss')
                            seed_status['train_loss'] = progress.get('train_loss')
                            seed_status['timestamp'] = progress.get('timestamp')
                            
                            # Get best metric from checkpoint
                            primary_metric = status[status_key]['primary_metric']
                            best_metric = None
                            
                            # For optuna_mod_new, checkpoint is in trial_dir/seed{seed}/best_model.pth
                            try:
                                checkpoint_file = seed_dir / "best_model.pth"
                                if checkpoint_file.exists():
                                    import torch
                                    checkpoint = torch.load(str(checkpoint_file), map_location='cpu', weights_only=False)
                                    
                                    metric_key_map = {
                                        'roc_auc': 'best_auroc',
                                        'pr_auc': 'best_pr_auc',
                                        'f1': 'best_f1',
                                        'mae': 'best_mae',
                                        'spearman': 'best_spearman'
                                    }
                                    
                                    if primary_metric in metric_key_map:
                                        key = metric_key_map[primary_metric]
                                        if key in checkpoint:
                                            value = checkpoint[key]
                                            if value is not None and isinstance(value, (int, float)):
                                                best_metric = float(value)
                                    
                                    if best_metric is None:
                                        best_metrics = checkpoint.get('best_metrics', {})
                                        if best_metrics and primary_metric in best_metrics:
                                            best_metric = best_metrics[primary_metric]
                                    
                                    if best_metric is None and primary_metric == 'mae':
                                        if 'val_loss' in checkpoint:
                                            value = checkpoint['val_loss']
                                            if value is not None and isinstance(value, (int, float)):
                                                best_metric = float(value)
                                    
                                    if best_metric is None and primary_metric == 'spearman':
                                        val_metrics = checkpoint.get('val_metrics', {})
                                        if 'spearman' in val_metrics:
                                            value = val_metrics['spearman']
                                            if value is not None and isinstance(value, (int, float)):
                                                best_metric = float(value)
                            except:
                                pass
                            
                            if best_metric is None:
                                progress_best_metrics = progress.get('best_metrics', {})
                                if primary_metric in progress_best_metrics:
                                    best_metric = progress_best_metrics[primary_metric]
                                elif primary_metric == 'mae' and 'val_loss' in progress:
                                    best_metric = progress['val_loss']
                                elif primary_metric in progress.get('val_metrics', {}):
                                    best_metric = progress['val_metrics'][primary_metric]
                                elif 'best_val_score' in progress:
                                    best_metric = progress['best_val_score']
                            
                            if best_metric is not None:
                                seed_status['best_metric'] = float(best_metric)
                            
                            if not seed_status.get('elapsed_time') and process_key in process_info:
                                proc_info = process_info[process_key]
                                if proc_info.get('start_time'):
                                    elapsed = datetime.now().timestamp() - proc_info['start_time']
                                    hours = int(elapsed / 3600)
                                    minutes = int((elapsed % 3600) / 60)
                                    seed_status['elapsed_time'] = f"{hours}h {minutes}m"
                            
                            # Fallback: if still no elapsed time, try to calculate from progress file timestamp
                            if not seed_status.get('elapsed_time'):
                                # Try to use timestamp from progress file
                                progress_timestamp = progress.get('timestamp')
                                if progress_timestamp:
                                    try:
                                        elapsed = datetime.now().timestamp() - float(progress_timestamp)
                                        hours = int(elapsed / 3600)
                                        minutes = int((elapsed % 3600) / 60)
                                        seed_status['elapsed_time'] = f"{hours}h {minutes}m"
                                    except (ValueError, TypeError):
                                        pass
                                
                                # Last fallback: use file modification time
                                if not seed_status.get('elapsed_time'):
                                    try:
                                        import time
                                        file_mtime = progress_file.stat().st_mtime
                                        elapsed = time.time() - file_mtime
                                        hours = int(elapsed / 3600)
                                        minutes = int((elapsed % 3600) / 60)
                                        seed_status['elapsed_time'] = f"{hours}h {minutes}m"
                                    except Exception:
                                        seed_status['elapsed_time'] = 'N/A'
                            
                            status[status_key]['running_seeds'] += 1
                        except Exception as e:
                            seed_status['state'] = 'error'
                            seed_status['error'] = str(e)
                            status[status_key]['failed_seeds'] += 1
                    else:
                        # No files found - pending
                        seed_status['state'] = 'pending'
                        status[status_key]['pending_seeds'] += 1
                    
                    status[status_key]['seeds'][seed_key] = seed_status
    
    return status

try:
    status = scan_training_status()
    
    # Get active datasets for display (convert dict to list of dataset names)
    active_datasets_dict_for_display = get_active_datasets_from_train_script()
    active_datasets_list = list(active_datasets_dict_for_display.keys()) if active_datasets_dict_for_display else []
    
    # Check if no active training detected
    if '_no_active_training' in status:
        print("ℹ️  No active train_edmpnn.sh or train_edmpnn_new.sh processes detected")
        print("   Waiting for training to start...")
        print("   (This monitor only tracks datasets currently being trained)")
        print("")
    elif not status:
        if active_datasets_list:
            print(f"ℹ️  Monitoring datasets: {', '.join(sorted(active_datasets_list))}")
            print("   (No checkpoints found yet - training may have just started)")
            print("")
        print("❌ No training checkpoints found for active datasets")
        print("   Checkpoints should be in:")
        print("     - checkpoints/{dataset}_optuna_final/seed{1-5}/ (for train_edmpnn.sh)")
        print("     - checkpoints/{dataset}_optuna_final_new/seed{1-5}/ (for train_edmpnn_new.sh)")
        print("")
    else:
        # Overall statistics
        total_datasets = len(status)
        total_completed = sum(d['completed_seeds'] for d in status.values())
        total_running = sum(d['running_seeds'] for d in status.values())
        total_pending = sum(d['pending_seeds'] for d in status.values())
        total_failed = sum(d['failed_seeds'] for d in status.values())
        
        completed_color = f"{Colors.GREEN}{total_completed}{Colors.RESET}"
        running_color = f"{Colors.YELLOW}{total_running}{Colors.RESET}"
        pending_color = f"{Colors.BLUE}{total_pending}{Colors.RESET}"
        failed_color = f"{Colors.RED}{total_failed}{Colors.RESET}"
        
        # Show monitored datasets
        if active_datasets_list:
            monitored_str = ', '.join(sorted(active_datasets_list))
            if len(monitored_str) > 80:
                monitored_str = monitored_str[:77] + "..."
            print(f"🎯 Monitoring: {monitored_str}")
            print("")
        
        print(f"📋 Overall Status: {total_datasets} datasets | ✅ Completed: {completed_color} seeds | 🔄 Running: {running_color} seeds | ⏳ Pending: {pending_color} seeds | ❌ Failed: {failed_color} seeds")
        print("")
        
        # === TIME-RELATED INFORMATION ===
        # Calculate average training time for completed seeds
        completed_times = []
        for status_key, dataset_status in status.items():
            for seed_key, seed_info in dataset_status.get('seeds', {}).items():
                if seed_info['state'] == 'completed' and seed_info.get('best_epoch'):
                    # Estimate: assume ~2-5 minutes per epoch (rough estimate)
                    # This is just a placeholder - actual time tracking would need process start time
                    pass
        
        # === DATASET DETAILS ===
        for status_key, dataset_status in sorted(status.items()):
            dataset_name = dataset_status['dataset_name']
            version = dataset_status.get('version', 'unknown')
            # 不顯示版本標籤
            version_label = ""
            primary_metric = dataset_status['primary_metric']
            completed = dataset_status['completed_seeds']
            running = dataset_status['running_seeds']
            pending = dataset_status['pending_seeds']
            failed = dataset_status['failed_seeds']
            
            # Calculate mean ± std for completed seeds
            test_scores = dataset_status['test_scores']
            if test_scores:
                mean_score = sum(test_scores) / len(test_scores)
                if len(test_scores) > 1:
                    variance = sum((x - mean_score) ** 2 for x in test_scores) / len(test_scores)
                    std_score = variance ** 0.5
                    score_display = f"{mean_score:.4f} ± {std_score:.4f}"
                else:
                    score_display = f"{mean_score:.4f}"
            else:
                score_display = "N/A"
            
            # Dataset header
            print(f"📦 Dataset: {Colors.BOLD}{dataset_name}{Colors.RESET}{version_label} (Primary Metric: {primary_metric})")
            print(f"   Status: ✅ {completed}/5 completed | 🔄 {running}/5 running | ⏳ {pending}/5 pending | ❌ {failed}/5 failed")
            if test_scores:
                print(f"   Test Score ({primary_metric}): {Colors.GREEN}{score_display}{Colors.RESET} ({len(test_scores)} seeds)")
            print("")
            
            # Seed details (only show running and recent completed in compact mode)
            if DISPLAY_MODE == "detailed" or running > 0:
                for seed in range(1, 6):
                    seed_key = f"seed{seed}"
                    # Check if seed exists in seeds dictionary, if not, skip or create default
                    if seed_key not in dataset_status['seeds']:
                        # Seed not initialized, skip it
                        continue
                    seed_info = dataset_status['seeds'][seed_key]
                    state = seed_info['state']
                    
                    if state == 'completed':
                        epoch = seed_info.get('best_epoch', 'N/A')
                        test_score = seed_info.get('test_score')
                        if test_score is not None:
                            print(f"   {Colors.GREEN}✅ {seed_key}: Completed{Colors.RESET} - Epoch {epoch}, Test {primary_metric}: {test_score:.4f}")
                        else:
                            print(f"   {Colors.GREEN}✅ {seed_key}: Completed{Colors.RESET} - Epoch {epoch}")
                    elif state == 'running':
                        epoch = seed_info.get('epoch', 0)
                        val_loss = seed_info.get('val_loss')
                        elapsed = seed_info.get('elapsed_time', 'N/A')
                        gpu_id = seed_info.get('gpu_id')
                        best_metric = seed_info.get('best_metric')
                        
                        # Format metric display based on primary metric type
                        loss_str = f", Val Loss: {val_loss:.4f}" if val_loss is not None else ""
                        if best_metric is not None:
                            # Format based on metric type (higher is better for roc_auc, pr_auc, spearman; lower is better for mae)
                            if primary_metric in ['roc_auc', 'pr_auc', 'spearman']:
                                metric_str = f", Best {primary_metric.upper()}: {Colors.GREEN}{best_metric:.4f}{Colors.RESET}"
                            elif primary_metric == 'mae':
                                metric_str = f", Best {primary_metric.upper()}: {Colors.GREEN}{best_metric:.4f}{Colors.RESET}"
                            else:
                                metric_str = f", Best {primary_metric}: {Colors.GREEN}{best_metric:.4f}{Colors.RESET}"
                        else:
                            metric_str = ""
                        gpu_str = f" [GPU {gpu_id}]" if gpu_id is not None else ""
                        print(f"   {Colors.YELLOW}🔄 {seed_key}: Running{Colors.RESET}{gpu_str} - Epoch {epoch}{loss_str}{metric_str}, Elapsed: {elapsed}")
                    elif state == 'pending':
                        if DISPLAY_MODE == "detailed":
                            print(f"   {Colors.BLUE}⏳ {seed_key}: Pending{Colors.RESET}")
                    elif state == 'error':
                        error = seed_info.get('error', 'Unknown error')
                        print(f"   {Colors.RED}❌ {seed_key}: Error{Colors.RESET} - {error[:50]}")
                
                if DISPLAY_MODE == "detailed" or running > 0:
                    print("")
            
            # Warnings for stuck training
            if running > 0:
                stuck_seeds = []
                for seed in range(1, 6):
                    seed_key = f"seed{seed}"
                    if seed_key not in dataset_status.get('seeds', {}):
                        continue
                    seed_info = dataset_status['seeds'][seed_key]
                    if seed_info['state'] == 'running' and seed_info.get('timestamp'):
                        # Check if progress file hasn't been updated in >1 hour
                        elapsed = datetime.now().timestamp() - seed_info['timestamp']
                        if elapsed > 3600:  # 1 hour
                            stuck_seeds.append(seed_key)
                
                if stuck_seeds:
                    print(f"   {Colors.RED}⚠️  Warning: Seeds with no progress >1 hour: {', '.join(stuck_seeds)}{Colors.RESET}")
                    print("")
        
        # === RECENT ACTIVITY ===
        # Show recently completed or running seeds
        recent_activity = []
        for status_key, dataset_status in status.items():
            dataset_name = dataset_status['dataset_name']
            version = dataset_status.get('version', 'unknown')
            primary_metric = dataset_status.get('primary_metric', 'roc_auc')
            for seed_key, seed_info in dataset_status.get('seeds', {}).items():
                if seed_info.get('state') in ['completed', 'running']:
                    recent_activity.append({
                        'dataset': dataset_name,
                        'version': version,
                        'seed': seed_key,
                        'state': seed_info['state'],
                        'info': seed_info,
                        'primary_metric': primary_metric
                    })
        
        if recent_activity:
            print("📝 Recent Activity:")
            # Sort by state (running first) and show top 5
            recent_activity.sort(key=lambda x: (x['state'] != 'running', x['dataset'], x['seed']))
            for item in recent_activity[:5]:
                dataset = item['dataset']
                version = item.get('version', 'unknown')
                # 不顯示版本標籤
                version_str = ""
                seed = item['seed']
                state = item['state']
                info = item['info']
                primary_metric = item['primary_metric']
                
                if state == 'completed':
                    test_score = info.get('test_score')
                    best_epoch = info.get('best_epoch', 'N/A')
                    if test_score is not None:
                        print(f"   {Colors.GREEN}✅ {dataset}{version_str}/{seed}: Completed{Colors.RESET} - Epoch {best_epoch}, Test: {test_score:.4f}")
                    else:
                        print(f"   {Colors.GREEN}✅ {dataset}{version_str}/{seed}: Completed{Colors.RESET} - Epoch {best_epoch}")
                elif state == 'running':
                    epoch = info.get('epoch', 0)
                    elapsed = info.get('elapsed_time', 'N/A')
                    gpu_id = info.get('gpu_id')
                    best_metric = info.get('best_metric')
                    
                    # Format metric display based on primary metric type
                    if best_metric is not None:
                        if primary_metric in ['roc_auc', 'pr_auc', 'spearman']:
                            metric_str = f", Best {primary_metric.upper()}: {Colors.GREEN}{best_metric:.4f}{Colors.RESET}"
                        elif primary_metric == 'mae':
                            metric_str = f", Best {primary_metric.upper()}: {Colors.GREEN}{best_metric:.4f}{Colors.RESET}"
                        else:
                            metric_str = f", Best {primary_metric}: {Colors.GREEN}{best_metric:.4f}{Colors.RESET}"
                    else:
                        metric_str = ""
                    gpu_str = f" [GPU {gpu_id}]" if gpu_id is not None else ""
                    print(f"   {Colors.YELLOW}🔄 {dataset}{version_str}/{seed}: Running{Colors.RESET}{gpu_str} - Epoch {epoch}{metric_str}, Elapsed: {elapsed}")
            print("")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
EOF
    
    # === GPU STATUS WITH WARNINGS ===
    if [ "$DISPLAY_MODE" = "detailed" ]; then
        if command -v nvidia-smi &> /dev/null; then
            echo ""
            echo "🎮 GPU Status:"
            GPU_WARNINGS_FILE=$(mktemp)
            HAS_RUNNING_TRAINING=$([ "$TRAINING_PIDS" -gt 0 ] && echo "1" || echo "0")
            
            # Collect GPU information first
            GPU_INFO=()
            GPU_COUNT=0
            while IFS=', ' read -r idx util mem_used mem_total mem_free temp; do
                # Check for low utilization (only if training is running)
                if [ "$HAS_RUNNING_TRAINING" = "1" ] && [ "$util" -lt 10 ]; then
                    echo "GPU $idx: Low utilization ($util%)" >> "$GPU_WARNINGS_FILE"
                fi
                
                # Check for high memory usage
                if [ "$mem_total" -gt 0 ]; then
                    mem_pct=$((mem_used * 100 / mem_total))
                    if [ "$mem_pct" -gt 95 ]; then
                        echo "GPU $idx: High memory usage ($mem_pct%)" >> "$GPU_WARNINGS_FILE"
                    fi
                fi
                
                # Check for low free memory
                if [ "$mem_free" -lt 1000 ]; then
                    echo "GPU $idx: Low free memory (${mem_free}MB)" >> "$GPU_WARNINGS_FILE"
                fi
                
                # Store GPU info for display
                if [ "$mem_total" -gt 0 ]; then
                    mem_pct=$((mem_used * 100 / mem_total))
                    GPU_INFO[$GPU_COUNT]="GPU $idx: ${util}% util, ${mem_used}/${mem_total} MB (${mem_pct}% used), ${mem_free} MB free, ${temp}°C"
                else
                    GPU_INFO[$GPU_COUNT]="GPU $idx: ${util}% util, ${mem_used}/${mem_total} MB, ${mem_free} MB free, ${temp}°C"
                fi
                GPU_COUNT=$((GPU_COUNT + 1))
            done < <(nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,memory.free,temperature.gpu --format=csv,noheader,nounits 2>/dev/null)
            
            # Display GPUs on the same line
            if [ "$GPU_COUNT" -gt 0 ]; then
                printf "   "
                for ((i=0; i<GPU_COUNT; i++)); do
                    printf "${GPU_INFO[$i]}"
                    if [ $i -lt $((GPU_COUNT - 1)) ]; then
                        printf " | "
                    fi
                done
                printf "\n"
            fi
            
            if [ -f "$GPU_WARNINGS_FILE" ] && [ -s "$GPU_WARNINGS_FILE" ]; then
                echo ""
                echo "⚠️  GPU Warnings:"
                while IFS= read -r warning; do
                    echo "   $warning"
                done < "$GPU_WARNINGS_FILE"
                rm -f "$GPU_WARNINGS_FILE"
            fi
        fi
    fi
    
    # === SYSTEM MEMORY CHECK ===
    if [ "$DISPLAY_MODE" = "detailed" ]; then
        if command -v free &> /dev/null; then
            MEM_INFO=$(free -m | grep Mem)
            MEM_TOTAL=$(echo $MEM_INFO | awk '{print $2}')
            MEM_USED=$(echo $MEM_INFO | awk '{print $3}')
            MEM_AVAIL=$(echo $MEM_INFO | awk '{print $7}')
            MEM_PCT=$((MEM_USED * 100 / MEM_TOTAL))
            
            echo ""
            echo "💾 System Memory:"
            echo "   Total: ${MEM_TOTAL}MB | Used: ${MEM_USED}MB (${MEM_PCT}%) | Available: ${MEM_AVAIL}MB"
            
            if [ "$MEM_PCT" -gt 90 ]; then
                echo "   ⚠️  Warning: High memory usage (${MEM_PCT}%)"
            fi
            if [ "$MEM_AVAIL" -lt 1000 ]; then
                echo "   ⚠️  Warning: Low available memory (${MEM_AVAIL}MB)"
            fi
        fi
    fi
    
    # === LOG ERROR CHECK ===
    if [ "$DISPLAY_MODE" = "detailed" ]; then
        # Check for recent errors in training logs
        LOG_DIRS=$(find runs -type d \( -name "*_optuna_final" -o -name "*_optuna_final_new" \) 2>/dev/null | head -5)
        if [ -n "$LOG_DIRS" ]; then
            ERROR_FOUND=false
            for log_dir in $LOG_DIRS; do
                # Check stderr.log files
                for stderr_log in $(find "$log_dir" -name "stderr.log" 2>/dev/null | head -3); do
                    ERROR_COUNT=$(grep -i "error\|exception\|failed\|traceback" "$stderr_log" 2>/dev/null | tail -10 | wc -l)
                    if [ "$ERROR_COUNT" -gt 0 ]; then
                        if [ "$ERROR_FOUND" = false ]; then
                            echo ""
                            echo "⚠️  Recent Errors in Training Logs:"
                            ERROR_FOUND=true
                        fi
                        dataset_name=$(echo "$stderr_log" | sed -n 's|.*runs/\([^/]*\)_optuna_final.*|\1|p' | sed 's/_optuna_final_new$//' | sed 's/_optuna_final$//')
                        seed_name=$(echo "$stderr_log" | sed -n 's|.*seed\([0-9]*\)/.*|\1|p')
                        echo "   ${dataset_name}/seed${seed_name}:"
                        grep -i "error\|exception\|failed" "$stderr_log" 2>/dev/null | tail -2 | while read line; do
                            echo "      ${line:0:80}"  # Truncate long lines
                        done
                    fi
                done
            done
        fi
    fi
    
    echo ""
    echo "=========================================="
    if [ "$DISPLAY_MODE" = "compact" ]; then
        echo "Next update in ${INTERVAL}s... (s=detail, p=pause, e=export, q=quit)"
    else
        echo "Next update in ${INTERVAL} seconds... (s=compact, p=pause, e=export, q=quit)"
    fi
    
    # Sleep with periodic key check
    for i in $(seq 1 $INTERVAL); do
        sleep 1
        KEY=$(read_key)
        case "$KEY" in
            s|S)
                if [ "$DISPLAY_MODE" = "detailed" ]; then
                    DISPLAY_MODE="compact"
                else
                    DISPLAY_MODE="detailed"
                fi
                break
                ;;
            p|P)
                PAUSED=true
                break
                ;;
            e|E)
                export_status
                sleep 1
                ;;
            q|Q)
                echo ""
                echo "👋 Exiting monitor..."
                exit 0
                ;;
        esac
    done
done


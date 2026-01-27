#!/usr/bin/env python3
"""
AEGNN-M Optuna Optimization Script (MOD NEW, with DMPNN dmp_steps search)
Corresponds to models.edmpnn_model and train_edmpnn_new.py

IMPORTANT: This script uses VALIDATION metrics (not test set) to select best trial,
following fusion_model's approach to avoid data leakage. The best trial is selected
based on validation set performance, ensuring fair evaluation.

IMPROVEMENTS:
- Uses RobustScaler for descriptor normalization (via train_edmpnn_new.py)
- Dynamic DMP steps based on dataset size
- Dynamic model depth/width based on dataset size and class imbalance
- Optimized attention heads to ensure divisibility with hidden_dim
"""

import os
import sys
import json
import argparse
import subprocess
import optuna
import fcntl
import time
import signal
import yaml
import socket
import numpy as np
import warnings
import torch
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner, HyperbandPruner
from progress_monitor import JSONProgressMonitor

# Suppress Optuna's repeated step reporting warnings (harmless, doesn't affect functionality)
warnings.filterwarnings("ignore", category=UserWarning, 
                       message=".*already reported.*", 
                       module="optuna.trial._trial")

GREEN = '\033[0;32m'
BLUE = '\033[0;34m'
YELLOW = '\033[1;33m'
RED = '\033[0;31m'
NC = '\033[0m'


def get_dataset_path(dataset_name: str):
    """Check and return dataset path (prefer processed, support TDC datasets)"""
    clean_name = dataset_name.replace('_dataset.csv', '').replace('.csv', '')

    tdc_data_dir = f"data/processed_tdc_data/{clean_name}"
    if os.path.isdir(tdc_data_dir):
        # Check if at least one seed directory exists with required files
        seed_found = False
        for seed in range(1, 6):
            seed_dir = os.path.join(tdc_data_dir, f"seed{seed}")
            if os.path.isdir(seed_dir):
                train_pt = os.path.join(seed_dir, "train.pt")
                valid_pt = os.path.join(seed_dir, "valid.pt")
                test_pt = os.path.join(seed_dir, "test.pt")
                if all(os.path.exists(f) for f in [train_pt, valid_pt, test_pt]):
                    seed_found = True
                    break  # Found at least one valid seed, can return
        
        # Only return TDC dataset if at least one valid seed was found
        if seed_found:
            return "--tdc_dataset", clean_name
        # If TDC directory exists but no valid seeds found, fall through to check other formats

    processed_path = f"data/processed/{clean_name}_processed.pkl"
    csv_path = f"data/{clean_name}_dataset.csv"

    if os.path.exists(processed_path):
        return "--processed_data_path", processed_path
    elif os.path.exists(csv_path):
        return "--data_path", csv_path
    else:
        raise FileNotFoundError(f"Dataset not found: {dataset_name}")


def get_task_type(dataset_name: str) -> str:
    """Determine task type (classification or regression) for a dataset."""
    clean_name = dataset_name.lower().replace('_dataset.csv', '').replace('.csv', '')
    config_path = "configs/dataset_primary_metrics.yaml"
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                if config and 'dataset_primary_metrics' in config:
                    dataset_config = config['dataset_primary_metrics'].get(clean_name)
                    if dataset_config and 'metric_type' in dataset_config:
                        return dataset_config['metric_type']
        except Exception as e:
            print(f"{YELLOW}Warning: Could not read config file {config_path}: {e}{NC}")

    classification_datasets = [
        'bace', 'bbbp', 'clintox', 'hiv', 'muv', 'sider', 'tox21', 'ames',
        'bbb_martins', 'bioavailability_ma', 'cyp3a4_substrate_carbonmangels',
        'dili', 'herg', 'hia_hou', 'pgp_broccatelli', 'cyp2c9_substrate_carbonmangels',
        'cyp2c9_veith', 'cyp2d6_substrate_carbonmangels', 'cyp2d6_veith', 'cyp3a4_veith'
    ]
    if clean_name in classification_datasets:
        return 'classification'
    return 'regression'


def get_primary_metric(dataset_name: str) -> str:
    """Get the primary metric for a dataset from config file."""
    clean_name = dataset_name.lower().replace('_dataset.csv', '').replace('.csv', '')
    # Handle possible naming differences (e.g. solubility_aqsoldb vs solubility_aqsolb)
    if clean_name == 'solubility_aqsolb':
        clean_name = 'solubility_aqsoldb'
    
    config_path = "configs/dataset_primary_metrics.yaml"
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                if config and 'dataset_primary_metrics' in config:
                    dataset_config = config['dataset_primary_metrics'].get(clean_name)
                    if dataset_config and 'primary_metric' in dataset_config:
                        return dataset_config['primary_metric']
        except Exception as e:
            print(f"{YELLOW}Warning: Could not read config file {config_path}: {e}{NC}")
    
    # Default: Return default metric based on task type
    task_type = get_task_type(dataset_name)
    if task_type == "classification":
        return "roc_auc"
    else:
        # For regression tasks, default to spearman (but this should rarely happen as config should contain all datasets)
        return "spearman"


_current_subprocess = None
_interrupt_requested = False

def signal_handler(signum, frame):
    """Handle interrupt signals and terminate subprocess"""
    global _current_subprocess, _interrupt_requested
    # Set interrupt flag instead of immediately raising exception
    # This allows the code to check the flag at safe points (not during sleep)
    _interrupt_requested = True
    if _current_subprocess is not None:
        print(f"\n{YELLOW}Received interrupt signal (SIG{signum}). Terminating training process...{NC}")
        try:
            _current_subprocess.terminate()
            try:
                _current_subprocess.wait(timeout=5)
            except subprocess.TimeoutExpired:
                _current_subprocess.kill()
                _current_subprocess.wait()
        except Exception as e:
            print(f"{RED}Error terminating subprocess: {e}{NC}")
        _current_subprocess = None
    # Don't raise KeyboardInterrupt here - let the code check _interrupt_requested at safe points


def objective(trial: optuna.trial.Trial, args):
    global _current_subprocess, _interrupt_requested
    # Reset interrupt flag at start of each trial
    _interrupt_requested = False
    try:
        dataset_name = args.dataset.lower()
        task_type = get_task_type(args.dataset)
        primary_metric = get_primary_metric(args.dataset)  # Get primary metric
        
        # Determine value to return on failure (based on Optuna direction)
        # For minimize tasks (like MAE), return large value on failure
        # For maximize tasks (like ROC-AUC, Spearman), return small value on failure
        if task_type == "classification":
            # Classification tasks are all maximize (ROC-AUC or PR-AUC, higher is better)
            FAILURE_VALUE = -1e10
        else:
            # Regression tasks: Decide based on primary metric
            if primary_metric == "mae":
                # MAE: Lower is better (minimize)
                FAILURE_VALUE = 1e10
            else:
                # Spearman: Higher is better (maximize)
                FAILURE_VALUE = -1e10
        
        # IMPROVEMENT 2.2: Dynamic model depth/width strategy based on dataset size and class imbalance
        # First determine the "expected hidden_dim range" and number of layers based on dataset characteristics,
        # The actual hidden_dim will be generated later through (num_heads, head_dim) reparameterization
        # (Solution 2: hidden_dim = num_heads * head_dim, ensuring divisibility by number of heads)
        #
        # Load dataset information first to determine appropriate model capacity
        data_arg_type, data_arg_path = get_dataset_path(args.dataset)
        dataset_size = None
        imbalance_ratio = None
        
        try:
            if data_arg_type == "--tdc_dataset":
                # Load TDC dataset to get size
                tdc_data_dir = f"data/processed_tdc_data/{data_arg_path}"
                seed_to_check = args.seed if hasattr(args, 'seed') and args.seed is not None else 1
                train_pt = os.path.join(tdc_data_dir, f"seed{seed_to_check}", "train.pt")
                if os.path.exists(train_pt):
                    train_data = torch.load(train_pt, weights_only=False)
                    if isinstance(train_data, list):
                        dataset_size = len(train_data)
                        # Calculate class imbalance for classification tasks
                        if task_type == "classification":
                            train_targets = [g.y.item() if hasattr(g, 'y') else 0 for g in train_data]
                            unique_classes, class_counts = np.unique(train_targets, return_counts=True)
                            if len(unique_classes) == 2:
                                pos_count = class_counts[1] if len(class_counts) > 1 else 0
                                neg_count = class_counts[0]
                                imbalance_ratio = neg_count / pos_count if pos_count > 0 else 1.0
                    else:
                        # Old format: tuple
                        _, _, _, _, train_y = train_data
                        if isinstance(train_y, torch.Tensor):
                            dataset_size = train_y.shape[0]
                        elif isinstance(train_y, list):
                            dataset_size = len(train_y)
                        # Calculate imbalance for classification
                        if task_type == "classification" and isinstance(train_y, (torch.Tensor, list, np.ndarray)):
                            train_y_array = train_y.numpy() if isinstance(train_y, torch.Tensor) else np.array(train_y)
                            unique_classes, class_counts = np.unique(train_y_array, return_counts=True)
                            if len(unique_classes) == 2:
                                pos_count = class_counts[1] if len(class_counts) > 1 else 0
                                neg_count = class_counts[0]
                                imbalance_ratio = neg_count / pos_count if pos_count > 0 else 1.0
        except Exception as e:
            # If loading fails, use default values
            print(f"  {YELLOW}⚠️  Could not load dataset for size/imbalance analysis: {e}{NC}")
            print(f"  {YELLOW}   Using default hyperparameter search space{NC}")
        
        # Log dataset characteristics for debugging
        if dataset_size is not None:
            print(f"  {BLUE}📊 Dataset characteristics: size={dataset_size}, imbalance_ratio={imbalance_ratio if imbalance_ratio is not None else 'N/A'}{NC}")
        
        # -----------------------------
        # Dynamically determine "expected hidden_dim range" and number of layers
        # Priority: Class imbalance > Dataset size
        # -----------------------------
        # We no longer directly sample hidden_dim, but first determine a target range,
        # then generate actual hidden_dim using num_heads * head_dim approach.
        if dataset_size is not None:
            print(f"  {BLUE}📊 Using dataset_size-aware hidden_dim strategy{NC}")
        
        # Set expected hidden_dim range for each regime (aligned with old logic, roughly)
        if imbalance_ratio is not None and imbalance_ratio > 100:
            # Extremely imbalanced: Use deeper model to learn complex patterns
            # Even for small datasets, imbalance requires more capacity
            base_hidden_min, base_hidden_max = 256, 512
            num_layers = trial.suggest_int("num_layers", 5, 10)
        elif dataset_size is not None and dataset_size < 1000:
            # Small dataset: Use smaller model to prevent overfitting
            # Only if not extremely imbalanced
            base_hidden_min, base_hidden_max = 64, 256
            num_layers = trial.suggest_int("num_layers", 2, 5)
        else:
            # Standard configuration
            base_hidden_min, base_hidden_max = 128, 512
            num_layers = trial.suggest_int("num_layers", 3, 8)
        
        dropout = trial.suggest_float("dropout", 0.0, 0.5, step=0.05)
        
        # Before reparameterization, use "expected median hidden_dim" to estimate model_size, to adjust LR search range
        approx_hidden_dim = (base_hidden_min + base_hidden_max) // 2
        model_size = approx_hidden_dim * num_layers
        if model_size < 1000:
            lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
        elif model_size < 3000:
            lr = trial.suggest_float("lr", 5e-5, 1e-3, log=True)
        else:
            lr = trial.suggest_float("lr", 1e-5, 5e-4, log=True)
        if trial.number == 0:
            print(f"  {BLUE}📊 IMPROVEMENT 4.2: LR search range chosen by model_size={model_size}{NC}")
        
        weight_decay = trial.suggest_float("weight_decay", 1e-8, 1e-3, log=True)
        
        # IMPROVEMENT 4.1: Dynamic batch_size search range based on dataset size
        # Adjust batch_size search range based on dataset size
        if dataset_size is not None:
            if dataset_size < 1000:
                # Small dataset: smaller batch sizes to prevent overfitting
                batch_size_options = [8, 16, 32]
                batch_size = trial.suggest_categorical("batch_size", batch_size_options)
                if trial.number == 0:  # Only print once per study
                    print(f"  {BLUE}📊 IMPROVEMENT 4.1: Batch size search range: {batch_size_options} (small dataset: {dataset_size} samples){NC}")
            elif dataset_size < 5000:
                # Medium dataset: moderate batch sizes
                batch_size_options = [16, 32, 64]
                batch_size = trial.suggest_categorical("batch_size", batch_size_options)
                if trial.number == 0:  # Only print once per study
                    print(f"  {BLUE}📊 IMPROVEMENT 4.1: Batch size search range: {batch_size_options} (medium dataset: {dataset_size} samples){NC}")
            else:
                # Large dataset: larger batch sizes for efficiency
                batch_size_options = [32, 64, 128]
                batch_size = trial.suggest_categorical("batch_size", batch_size_options)
                if trial.number == 0:  # Only print once per study
                    print(f"  {BLUE}📊 IMPROVEMENT 4.1: Batch size search range: {batch_size_options} (large dataset: {dataset_size} samples){NC}")
        else:
            # Default: use medium range if dataset size unknown
            batch_size_options = [16, 32, 64]
            batch_size = trial.suggest_categorical("batch_size", batch_size_options)
            if trial.number == 0:  # Only print once per study
                print(f"  {YELLOW}📊 IMPROVEMENT 4.1: Batch size search range: {batch_size_options} (default, dataset size unknown){NC}")
        
        grad_clip_norm = trial.suggest_float("grad_clip_norm", 0.1, 1.0)

        # IMPROVEMENT 5.2: Dynamic mixup based on dataset characteristics
        # For small datasets or extremely imbalanced datasets, force consideration of mixup
        use_mixup = False
        mixup_alpha = 0.2  # Default value
        
        if dataset_size is not None and (dataset_size < 2000 or (imbalance_ratio is not None and imbalance_ratio > 50)):
            # Small dataset or highly imbalanced: force consideration of mixup
            use_mixup = trial.suggest_categorical("use_mixup", [True, False])
            if use_mixup:
                mixup_alpha = trial.suggest_float("mixup_alpha", 0.5, 4.0)
            if trial.number == 0:  # Only print once per study
                reason = []
                if dataset_size < 2000:
                    reason.append(f"small dataset ({dataset_size} samples)")
                if imbalance_ratio is not None and imbalance_ratio > 50:
                    reason.append(f"highly imbalanced (ratio={imbalance_ratio:.2f})")
                print(f"  {BLUE}📊 IMPROVEMENT 5.2: Mixup enabled for search ({', '.join(reason)}){NC}")
        else:
            # Standard dataset: mixup is optional but not forced
            use_mixup = trial.suggest_categorical("use_mixup", [True, False])
            if use_mixup:
                mixup_alpha = trial.suggest_float("mixup_alpha", 0.1, 0.5)

        # =====================================================
        # IMPROVEMENT 2.3 (Solution 2): Reparameterize hidden_dim
        #   - Directly search (num_heads, head_dim)
        #   - Actual hidden_dim = num_heads * head_dim
        #   - This naturally satisfies out_channels % heads == 0, no longer need to dynamically modify search space
        # =====================================================
        #
        # FIXED: Method 1 - Dynamic head_dim range based on selected num_heads
        #   This ensures hidden_dim = num_heads * head_dim ≤ base_hidden_max
        #   Previous bug: head_dim_max was calculated using min_num_heads=2, but num_heads could be 16,
        #   leading to hidden_dim = 16 * 256 = 4096, far exceeding base_hidden_max = 512
        #
        # 1) First, select num_heads (categorical choice)
        #    This keeps the full search space for num_heads [2, 4, 8, 16]
        num_heads = trial.suggest_categorical("num_heads", [2, 4, 8, 16])
        
        # 2) Dynamically calculate head_dim range based on selected num_heads
        #    This ensures hidden_dim = num_heads * head_dim ≤ base_hidden_max
        #    head_dim uses step size of 16, common choices: 16, 32, 48, 64, 80, ...
        #
        #    Upper bound: Ensure hidden_dim ≤ base_hidden_max
        #    head_dim_max = floor(base_hidden_max / num_heads) rounded down to multiple of 16
        head_dim_max = max(16, (base_hidden_max // num_heads) // 16 * 16)
        
        #    Lower bound: Ensure hidden_dim ≥ base_hidden_min (when possible)
        #    head_dim_min = ceil(base_hidden_min / num_heads) rounded up to multiple of 16
        head_dim_min = max(16, ((base_hidden_min + num_heads - 1) // num_heads + 15) // 16 * 16)
        
        #    Ensure head_dim_min ≤ head_dim_max
        if head_dim_min > head_dim_max:
            head_dim_min = 16
        
        # 3) Select head_dim within the calculated range
        head_dim = trial.suggest_int("head_dim", head_dim_min, head_dim_max, step=16)
        
        # 4) Calculate actual hidden_dim
        hidden_dim = num_heads * head_dim
        
        # Safety check: Verify hidden_dim is within expected range
        if hidden_dim > base_hidden_max:
            # This should not happen with correct calculation, but add safety check
            print(f"  {YELLOW}⚠️  Warning: hidden_dim {hidden_dim} exceeds base_hidden_max {base_hidden_max}, clamping...{NC}")
            # Clamp head_dim to ensure hidden_dim ≤ base_hidden_max
            head_dim = (base_hidden_max // num_heads) // 16 * 16
            hidden_dim = num_heads * head_dim
        
        if trial.number == 0:
            print(f"  {BLUE}📊 IMPROVEMENT 2.3: Reparameterized hidden_dim = num_heads({num_heads}) × head_dim({head_dim}) = {hidden_dim}{NC}")
            print(f"  {BLUE}   head_dim range: [{head_dim_min}, {head_dim_max}] (step=16), hidden_dim range: [{num_heads * head_dim_min}, {num_heads * head_dim_max}]{NC}")
            print(f"  {BLUE}   ✅ hidden_dim constraint: {hidden_dim} ≤ base_hidden_max({base_hidden_max}){NC}")
        
        warmup_epochs = trial.suggest_int("warmup_epochs", 5, 15, step=5)
        
        # IMPROVEMENT 2.1: Dynamic DMP steps based on dataset size
        if dataset_size is not None:
            if dataset_size < 1000:
                # Small dataset: fewer DMP steps
                dmp_steps = trial.suggest_int("dmp_steps", 1, 3)
            elif dataset_size < 5000:
                # Medium dataset: moderate DMP steps
                dmp_steps = trial.suggest_int("dmp_steps", 2, 5)
            else:
                # Large dataset: more DMP steps
                dmp_steps = trial.suggest_int("dmp_steps", 3, 6)
        else:
            # Default: use original range if dataset size unknown
            dmp_steps = trial.suggest_int("dmp_steps", 1, 6)

        scheduler_type = trial.suggest_categorical("scheduler_type", ["cosine", "step", "plateau"])
        min_lr = trial.suggest_float("min_lr", 1e-7, 1e-5, log=True)
        drop_path_rate = trial.suggest_float("drop_path_rate", 0.0, 0.2, step=0.05)
        activation = trial.suggest_categorical("activation", ["SiLU", "ReLU", "LeakyReLU", "PReLU", "ELU", "SELU", "tanh"])

        alpha = trial.suggest_float("alpha", 0.1, 0.3, step=0.05)
        ffn_expansion_factor = trial.suggest_categorical("ffn_expansion_factor", [2, 4, 6, 8])
        pool_type = trial.suggest_categorical("pool_type", ["mean", "sum", "norm"])
        
        # IMPROVEMENT 5.1: 3D rotation augmentation with intensity control
        # Add augmentation intensity control
        rotate_aug = trial.suggest_categorical("rotate_aug", [True, False])
        rotation_prob = None
        max_rotation_angle = None
        if rotate_aug:
            # Can add rotation angle range, rotation probability and other parameters
            rotation_prob = trial.suggest_float("rotation_prob", 0.3, 1.0)
            max_rotation_angle = trial.suggest_float("max_rotation_angle", 15.0, 180.0)
            if trial.number == 0:  # Only print once per study
                print(f"  {BLUE}📊 IMPROVEMENT 5.1: Rotation augmentation enabled with intensity control (prob: [0.3, 1.0], angle: [15.0, 180.0]){NC}")
        
        # Descriptor dropout (optimize as hyperparameter)
        # Range: 0.0 to 0.3, step 0.05 (suitable for input feature dropout range)
        descriptor_dropout = trial.suggest_float("descriptor_dropout", 0.0, 0.3, step=0.05)

        # New storage structure: checkpoints/optuna_mod_new/{dataset}/seed{seed}/opt/{trial.number}/
        # For per-seed optimization, use the seed from args.seed
        # For legacy mode (all seeds), use seed1 as base (but each seed will have its own subdirectory)
        if hasattr(args, 'seed') and args.seed is not None:
            # Per-seed optimization: store in seed-specific directory
            seed_num = args.seed
            trial_base_dir = os.path.join("checkpoints", "optuna_mod_new", args.dataset, f"seed{seed_num}", "opt", str(trial.number))
            log_dir = os.path.join("runs", "optuna_mod_new", args.dataset, f"seed{seed_num}", "opt", str(trial.number))
        else:
            # Legacy mode: use seed1 as base directory
            trial_base_dir = os.path.join("checkpoints", "optuna_mod_new", args.dataset, "seed1", "opt", str(trial.number))
            log_dir = os.path.join("runs", "optuna_mod_new", args.dataset, "seed1", "opt", str(trial.number))
        os.makedirs(trial_base_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        # Create configuration dictionary for this trial
        from datetime import datetime
        trial_config = {
            "trial_number": trial.number,
            "study_name": args.study_name,
            "dataset": args.dataset,
            "task_type": task_type,
            "primary_metric": primary_metric,
            "timestamp": datetime.now().isoformat(),
            "hyperparameters": {
                "hidden_dim": hidden_dim,
                "num_layers": num_layers,
                "dropout": dropout,
                "lr": lr,
                "learning_rate": lr,  # Alias for compatibility
                "weight_decay": weight_decay,
                "batch_size": batch_size,
                "grad_clip_norm": grad_clip_norm,
                "num_heads": num_heads,
                "warmup_epochs": warmup_epochs,
                "dmp_steps": dmp_steps,
                "scheduler_type": scheduler_type,
                "min_lr": min_lr,
                "drop_path_rate": drop_path_rate,
                "activation": activation,
                "alpha": alpha,
                "ffn_expansion_factor": ffn_expansion_factor,
                "pool_type": pool_type,
                "aggregation": pool_type,  # Alias for compatibility
                "rotate_aug": rotate_aug,
                "rotation_prob": rotation_prob,  # IMPROVEMENT 5.1: Rotation probability (only if rotate_aug=True)
                "max_rotation_angle": max_rotation_angle,  # IMPROVEMENT 5.1: Maximum rotation angle (only if rotate_aug=True)
                "descriptor_dropout": descriptor_dropout,
                "use_mixup": use_mixup,
                "mixup_alpha": mixup_alpha,
                "num_epochs": args.epochs,
                "early_stopping_patience": 30,
                "use_smart_early_stopping": True,
                "smart_early_stopping_max_patience": 50,
                "use_descriptor": True,
                "descriptor_dim": 217,
                "use_pre_norm": True,
            }
        }
        
        # Add task-specific settings
        if task_type == "classification":
            trial_config["hyperparameters"]["model_type"] = "classifier"
            trial_config["hyperparameters"]["use_bce_for_imbalanced"] = True
            trial_config["hyperparameters"]["auto_pos_weight"] = True
        else:
            trial_config["hyperparameters"]["model_type"] = "regressor"
        
        # Save trial configuration to JSON file (shared by all seeds in this trial)
        trial_config_path = os.path.join(trial_base_dir, "trial_config.json")
        with open(trial_config_path, "w") as f:
            json.dump(trial_config, f, indent=2)
        
        print(f"  {GREEN}✓ Trial configuration saved to: {trial_config_path}{NC}")

        # data_arg_type and data_arg_path already obtained above for dataset size analysis
        
        # If seed is specified, only train that seed (for per-seed optimization)
        # Otherwise, use all 5 seeds (legacy behavior)
        if hasattr(args, 'seed') and args.seed is not None:
            seed_list = [args.seed]
            print(f"\n{BLUE}[Trial {trial.number}] Starting training for seed {args.seed} (per-seed optimization mode)...{NC}")
        else:
            # Legacy mode: Use all 5 seeds per trial
            seed_list = [1, 2, 3, 4, 5]
            print(f"\n{BLUE}[Trial {trial.number}] Starting training for all seeds {seed_list} (legacy mode)...{NC}")
            print(f"  {BLUE}GPU Allocation: GPU 0 → seeds [1, 2], GPU 1 → seeds [3, 4, 5]{NC}")
        
        valid_scores = []

        # Track training progress for all seeds, used for Optuna pruning
        # Report intermediate values from all seeds for pruning decision
        # Use average validation loss across all seeds for pruning (similar to fusion_model)
        last_reported_epoch = -1
        reported_steps = set()  # Track reported steps to avoid duplicates
        seed_training_processes = []
        seed_val_losses = {}  # Track validation losses for each seed at each epoch

        for seed in seed_list:
            # For per-seed optimization, trial_base_dir already includes seed{seed_num}/opt/{trial.number}/
            # So we don't need to add another seed{seed} subdirectory
            if hasattr(args, 'seed') and args.seed is not None:
                # Per-seed mode: use trial_base_dir directly (already contains seed path)
                seed_save_dir = trial_base_dir
                seed_log_dir = log_dir
            else:
                # Legacy mode: add seed{seed} subdirectory
                seed_save_dir = os.path.join(trial_base_dir, f"seed{seed}")
                seed_log_dir = os.path.join(log_dir, f"seed{seed}")
            os.makedirs(seed_save_dir, exist_ok=True)
            os.makedirs(seed_log_dir, exist_ok=True)

            seed_base_port = 20000 + ((args.worker_id * 5000 + trial.number * 500 + seed * 50) % 40000)

            # GPU allocation: seeds 1,2 → GPU 0; seeds 3,4,5 → GPU 1
            # Always assign based on seed value, regardless of seed_list length
            if seed in [1, 2]:
                assigned_gpu = 0
            else:  # seed in [3, 4, 5]
                assigned_gpu = 1

            cmd = [sys.executable, "scripts/train_edmpnn_new.py"]

            # Use configuration file for hyperparameters (more reliable than command-line args)
            cmd.extend(["--config", trial_config_path])
            
            # Only add data-specific and path parameters via command line
            # IMPORTANT: Seed management for reproducibility
            # 
            # Seed strategy (Solution 1: Fixed initialization for fair hyperparameter comparison):
            # - TDC datasets:
            #   * tdc_seed (1-5): Select which pre-split data directory
            #   * model_init_seed: Fixed as seed*1000+seed, consistent with final training script
            # - Non-TDC datasets:
            #   * data_split_seed: Still need to parse from seed encoding (using hundreds digit)
            #   * model_init_seed: Also fixed as seed*1000+seed, plus seed*100 to preserve hundreds digit split information
            # 
            # This ensures:
            #   - Different trials with same seed use same initialization (avoid initialization noise interfering with hyperparameter comparison)
            #   - TDC and final training maintain consistent initialization formula
            base_model_init_seed = seed * 1000 + seed
            if data_arg_type == "--tdc_dataset":
                # TDC: No need to encode split seed again (tdc_seed is already explicitly passed)
                model_init_seed = base_model_init_seed
            else:
                # Non-TDC: Add seed to hundreds digit, so train_edmpnn_new.py can still parse data_split_seed
                # For example, when seed=1: 1001 + 100 = 1101, hundreds digit is 1, can be extracted by data_split_seed
                model_init_seed = base_model_init_seed + seed * 100
            
            if data_arg_type == "--tdc_dataset":
                cmd.extend(["--tdc_dataset", data_arg_path])
                cmd.extend(["--tdc_seed", str(seed)])  # Data split seed: selects pre-split data file (seed1/seed2/.../seed5)
                cmd.extend(["--seed", str(model_init_seed)])  # Model initialization seed: ensures unique initialization
            else:
                cmd.extend([data_arg_type, data_arg_path])
                cmd.extend(["--split_method", "scaffold"])
                # For non-TDC: Pass combined seed that encodes both data_split_seed and model_init_seed
                # train_edmpnn_new.py will extract original_seed (1-5) for data splitting
                # and use full seed for model initialization
                cmd.extend(["--seed", str(model_init_seed)])  # Combined seed: will be split in train_edmpnn_new.py

            # Add paths and system parameters (not in config file)
            cmd.extend([
                "--save_dir", seed_save_dir,
                "--log_dir", seed_log_dir,
                "--base_port", str(seed_base_port),
                "--world_size", "1",  # Force single GPU mode (no DDP)
            ])

            # --- Execute training with GPU assignment ---
            # Set CUDA_VISIBLE_DEVICES to assign specific GPU to this seed
            # This makes the assigned GPU appear as GPU 0 to the training script
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(assigned_gpu)
            
            print(f"  {BLUE}[Seed {seed}] Training on GPU {assigned_gpu}...{NC}")
            stdout_log_path = os.path.join(seed_log_dir, "stdout.log")
            stderr_log_path = os.path.join(seed_log_dir, "stderr.log")

            with open(stdout_log_path, "w") as f_out, open(stderr_log_path, "w") as f_err:
                proc = subprocess.Popen(
                    cmd, stdout=f_out, stderr=f_err, preexec_fn=os.setsid, env=env
                )
                seed_training_processes.append((seed, proc, seed_save_dir, stdout_log_path))

        # Create progress monitors (using JSON files, efficient and simple)
        progress_monitors = {}
        for seed, proc, seed_save_dir, stdout_log_path in seed_training_processes:
            progress_file = os.path.join(seed_save_dir, "training_progress.json")
            progress_monitors[seed] = (JSONProgressMonitor(progress_file), stdout_log_path)
        
        # Wait for seed training to complete, monitor progress for pruning (all seeds per trial)
        # Similar to fusion_model: each trial processes all seeds sequentially
        try:
            while any(proc.poll() is None for _, proc, _, _ in seed_training_processes):
                # Collect current epoch and validation loss for all seeds (using efficient progress monitoring)
                # Track validation losses for each seed to compute average for pruning
                current_epoch = -1
                seed_val_losses_current = {}
                
                for seed, proc, _, _ in seed_training_processes:
                    if proc.poll() is not None:
                        continue  # Skip completed seeds
                    
                    monitor_info = progress_monitors.get(seed)
                    if monitor_info:
                        monitor, stdout_log_path = monitor_info
                        # Use progress monitor to read progress (much faster than parsing log files)
                        result = monitor.get_latest_epoch_and_loss()
                        if result:
                            epoch, val_loss = result
                            if epoch is not None:
                                try:
                                    epoch = int(epoch)
                                except Exception:
                                    epoch = None
                            if epoch is not None and epoch > 0 and val_loss is not None:
                                seed_val_losses_current[seed] = (epoch, val_loss)
                                if epoch > current_epoch:
                                    current_epoch = epoch
                    
                    # Fallback: If progress file doesn't exist, fall back to parsing log file
                    # (This ensures backward compatibility even if training script hasn't been updated)
                    if monitor_info and os.path.exists(stdout_log_path):
                        try:
                            with open(stdout_log_path, 'r') as log_file:
                                lines = log_file.readlines()
                            
                            # Find current epoch
                            epoch = -1
                            for line in reversed(lines):
                                if "Epoch" in line and "/" in line:
                                    try:
                                        epoch_part = line.split("Epoch")[1].split("/")[0].strip()
                                        if epoch_part.isdigit():
                                            epoch = int(epoch_part)
                                            if epoch > current_epoch:
                                                current_epoch = epoch
                                            break
                                    except: pass
                            
                            # Find corresponding validation loss
                            if epoch > 0:
                                for i, line in enumerate(lines):
                                    if f"Epoch {epoch}/" in line:
                                        for j in range(i, min(i + 5, len(lines))):
                                            if "Validation loss:" in lines[j]:
                                                try:
                                                    val_loss = float(lines[j].split("Validation loss:")[1].split()[0])
                                                    seed_val_losses_current[seed] = (epoch, val_loss)
                                                    break
                                                except: pass
                                        break
                        except Exception as e:
                            # If log read fails, skip
                            pass
                
                # Use average validation loss across all seeds for pruning (similar to fusion_model)
                # Only report if we have validation losses from all active seeds at the same epoch
                if current_epoch > 0 and len(seed_val_losses_current) > 0:
                    # Ensure epoch is an integer (Hyperband requires integer resource steps)
                    report_epoch = int(current_epoch)
                    
                    # Check if all active seeds have reached the same epoch
                    active_seeds = [seed for seed, proc, _, _ in seed_training_processes if proc.poll() is None]
                    seeds_at_current_epoch = [seed for seed, (epoch, _) in seed_val_losses_current.items() 
                                             if epoch == report_epoch and seed in active_seeds]
                    
                    # Report if at least one seed has reached current epoch (for early pruning)
                    # Or wait for all seeds to reach same epoch for more robust pruning decision
                    # For now, use the average of available seeds at current epoch (similar to fusion_model approach)
                    if len(seeds_at_current_epoch) > 0:
                        val_losses_at_epoch = [val_loss for seed, (epoch, val_loss) in seed_val_losses_current.items() 
                                              if epoch == report_epoch and seed in seeds_at_current_epoch]
                        avg_val_loss = sum(val_losses_at_epoch) / len(val_losses_at_epoch)
                        
                        # Check if should report: epoch must be greater than last reported and not yet reported
                        if report_epoch > last_reported_epoch and report_epoch not in reported_steps:
                            # Update reported_steps first to avoid duplicate reports in same loop iteration
                            reported_steps.add(report_epoch)
                            last_reported_epoch = report_epoch
                            
                            # FIXED: Report validation loss correctly based on Optuna direction
                            # Optuna's direction is already set correctly (maximize for ROC-AUC/Spearman, minimize for MAE)
                            # For pruning, we need to report a value where:
                            #   - For maximize direction: higher value = better (so report -val_loss, since lower loss = better)
                            #   - For minimize direction: lower value = better (so report val_loss directly)
                            # However, Optuna's pruner expects the value to match the study direction.
                            # Since we're using validation loss (lower is better), we need to convert it:
                            #   - For maximize tasks: report -val_loss (so lower loss becomes higher value)
                            #   - For minimize tasks: report val_loss directly (lower loss = lower value, which is better)
                            
                            if task_type == "classification":
                                # Classification: maximize ROC-AUC/PR-AUC (higher is better)
                                # Report -val_loss so lower loss becomes higher value (better for maximize)
                                report_value = -avg_val_loss
                            else:
                                # Regression: check primary metric
                                if primary_metric == "mae":
                                    # MAE: minimize (lower is better)
                                    # Report val_loss directly (lower loss = lower value = better for minimize)
                                    report_value = avg_val_loss
                                else:
                                    # Spearman: maximize (higher is better)
                                    # Report -val_loss so lower loss becomes higher value (better for maximize)
                                    report_value = -avg_val_loss
                            
                            trial.report(report_value, report_epoch)
                            
                            if trial.should_prune():
                                # Terminate all training processes
                                for _, proc, _, _ in seed_training_processes:
                                    if proc.poll() is None:
                                        proc.terminate()
                                raise optuna.TrialPruned()
                
                # Use shorter check interval (reading JSON files is much faster than parsing logs)
                # Check for interrupt before sleeping
                if _interrupt_requested:
                    raise KeyboardInterrupt
                time.sleep(1)  # Reduced from 5 seconds to 1 second
                # Check again after sleep
                if _interrupt_requested:
                    raise KeyboardInterrupt

            # Check exit status of all processes
            for seed, proc, seed_save_dir, stdout_log_path in seed_training_processes:
                exit_code = proc.wait()
                if exit_code != 0:
                    # Check error logs to provide more detailed error information
                    error_msg = "Unknown error"
                    stderr_log_path = stdout_log_path.replace("stdout.log", "stderr.log")
                    
                    # Try to read error information from stderr
                    if os.path.exists(stderr_log_path):
                        try:
                            with open(stderr_log_path, 'r') as f:
                                stderr_lines = f.readlines()
                                # Look for key error information
                                for line in reversed(stderr_lines[-50:]):  # Check last 50 lines
                                    if "OutOfMemoryError" in line or "CUDA out of memory" in line:
                                        error_msg = "CUDA OOM (GPU memory insufficient)"
                                        break
                                    elif "RuntimeError" in line:
                                        error_msg = "RuntimeError"
                                        break
                                    elif "Traceback" in line:
                                        # Extract error type
                                        for err_line in stderr_lines:
                                            if "Error:" in err_line or "Exception:" in err_line:
                                                error_msg = err_line.strip()
                                                break
                                        break
                        except:
                            pass
                    
                    print(f"{RED}[Trial {trial.number}] Seed {seed} training failed!{NC}")
                    print(f"   Exit code: {exit_code}")
                    print(f"   Error: {error_msg}")
                    if "OOM" in error_msg or "memory" in error_msg.lower():
                        print(f"   💡 Suggestion: Reduce worker count or batch size to avoid GPU OOM")
                    
                    # Record detailed error information to trial's user_attr
                    error_info = f"Seed {seed}: Training failed with exit code {exit_code}, Error: {error_msg}"
                    trial.set_user_attr(f"error_seed_{seed}", error_info)
                    trial.set_user_attr(f"exit_code_seed_{seed}", exit_code)
                    
                    # Use FAILURE_VALUE instead of 0.0 to avoid Optuna mistakenly thinking failed trial is best
                    valid_scores.append(FAILURE_VALUE)
                    continue

                # Read results from training_history.json
                # IMPORTANT: Use VALIDATION metrics (not test set) to select best trial
                # This follows fusion_model's approach and avoids data leakage
                history_path = os.path.join(seed_save_dir, "training_history.json")
                if os.path.exists(history_path):
                    try:
                        with open(history_path, 'r') as f:
                            history = json.load(f)
                        
                        # Try to get best validation metric from training history
                        # Priority: best_primary_metric_value > max of validation metric list > fallback to test_results
                        best_val_metric = None
                        
                        # Method 1: Use best_primary_metric_value (most reliable, saved by train_edmpnn_new.py)
                        # For MAE, this is the best (minimum) validation MAE
                        if 'best_primary_metric_value' in history:
                            saved_primary_metric = history.get('primary_metric', '')
                            # Only use if it matches the expected primary metric (or if not specified)
                            if saved_primary_metric == primary_metric or not saved_primary_metric:
                                best_val_metric = history.get('best_primary_metric_value')
                                if best_val_metric is not None and not np.isnan(best_val_metric):
                                    # Use this value (for MAE, this is the best validation MAE)
                                    pass
                                else:
                                    # Invalid value, try other methods
                                    best_val_metric = None
                            else:
                                # Primary metric mismatch, try other methods
                                best_val_metric = None
                        
                        # Method 2: Extract from validation metric lists (val_aurocs, val_pr_aucs, etc.)
                        if best_val_metric is None:
                            if primary_metric in ['roc_auc', 'auroc']:
                                val_aurocs = history.get('val_aurocs', [])
                                if val_aurocs:
                                    best_val_metric = max(val_aurocs)
                            elif primary_metric == 'pr_auc':
                                val_pr_aucs = history.get('val_pr_aucs', [])
                                if val_pr_aucs:
                                    best_val_metric = max(val_pr_aucs)
                            elif primary_metric == 'spearman':
                                val_spearman = history.get('val_spearman', [])
                                if val_spearman:
                                    best_val_metric = max(val_spearman)
                            elif primary_metric == 'mae':
                                # For MAE, lower is better, so we need to find minimum
                                # Check if val_mae list exists (if train_edmpnn_new.py saves it in the future)
                                val_mae = history.get('val_mae', [])
                                if val_mae:
                                    best_val_metric = min(val_mae)
                                # Note: best_primary_metric_value (best validation MAE) should already be 
                                # checked in Method 1 above, which is the preferred source
                        
                        # Method 3: Fallback to test_results (for backward compatibility, but log warning)
                        if best_val_metric is None:
                            test_results = history.get('test_results', {})
                            print(f"  {YELLOW}⚠️  [Seed {seed}] Warning: Using test set metric as fallback (validation metric not found){NC}")
                            
                            if task_type == "classification":
                                if primary_metric == "roc_auc":
                                    best_val_metric = test_results.get('roc_auc', None)
                                elif primary_metric == "pr_auc":
                                    best_val_metric = test_results.get('pr_auc', None)
                                else:
                                    # Default to roc_auc
                                    best_val_metric = test_results.get('roc_auc', None)
                            else:
                                # Regression
                                if primary_metric == "spearman":
                                    best_val_metric = test_results.get('spearman', None)
                                elif primary_metric == "mae":
                                    best_val_metric = test_results.get('mae', None)
                                else:
                                    # Try to auto-select
                                    best_val_metric = test_results.get('spearman') or test_results.get('mae')
                        
                        # Validate and use the metric
                        if best_val_metric is not None and not np.isnan(best_val_metric):
                            score = float(best_val_metric)
                            valid_scores.append(score)
                            print(f"  {GREEN}[Seed {seed}] Validation {primary_metric}: {score:.4f}{NC}")
                        else:
                            print(f"{RED}[Trial {trial.number}] Seed {seed} completed but validation {primary_metric} not found or invalid!{NC}")
                            print(f"   Available history keys: {list(history.keys())}")
                            print(f"   Check logs: {stdout_log_path}")
                            error_info = f"Seed {seed}: Validation {primary_metric} not found or invalid"
                            trial.set_user_attr(f"error_seed_{seed}", error_info)
                            valid_scores.append(FAILURE_VALUE)
                    except Exception as e:
                        print(f"{RED}[Trial {trial.number}] Seed {seed} failed to read results: {e}{NC}")
                        print(f"   History file: {history_path}")
                        # Record detailed error information
                        error_info = f"Seed {seed}: Failed to read training_history.json - {str(e)}"
                        trial.set_user_attr(f"error_seed_{seed}", error_info)
                        valid_scores.append(FAILURE_VALUE)
                else:
                    print(f"{YELLOW}[Trial {trial.number}] Seed {seed} training_history.json not found!{NC}")
                    print(f"   Expected path: {history_path}")
                    print(f"   Check training logs: {stdout_log_path}")
                    # Record error information
                    error_info = f"Seed {seed}: training_history.json not found at {history_path}"
                    trial.set_user_attr(f"error_seed_{seed}", error_info)
                    valid_scores.append(FAILURE_VALUE)

        except optuna.TrialPruned:
            # Ensure all processes are terminated
            for _, proc, _, _ in seed_training_processes:
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
            raise
        except KeyboardInterrupt:
            # Ensure all processes are terminated
            for _, proc, _, _ in seed_training_processes:
                if proc.poll() is None:
                    proc.terminate()
            raise

        if not valid_scores: return FAILURE_VALUE
        
        # Return validation metric score (following fusion_model approach)
        # For per-seed optimization: return single seed's validation metric
        # For legacy mode (multiple seeds): return average validation metric (like fusion_model)
        trial.set_user_attr("seed_list", seed_list)
        trial.set_user_attr("valid_scores", valid_scores)  # Save all seed scores for analysis
        trial.set_user_attr("trial_dir", trial_base_dir)
        # Save the seed number for per-seed optimization
        if hasattr(args, 'seed') and args.seed is not None:
            trial.set_user_attr("optimized_seed", args.seed)
        
        # Return average validation metric (consistent with fusion_model)
        # This ensures best trial selection is based on validation set performance
        score = np.mean(valid_scores)
        return score

    except optuna.TrialPruned:
        raise
    except KeyboardInterrupt:
        print(f"\n{YELLOW}[Trial {trial.number}] Interrupted.{NC}")
        raise


def main():
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--study_name", type=str, default=None)
    parser.add_argument("--storage", type=str, default="sqlite:///optuna_edmpnn_results_new/optuna_mod_new.db")
    parser.add_argument("--n_trials", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--worker_id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=None, 
                       help="Specific seed to optimize (1-5). If provided, creates per-seed study. If None, uses all seeds (legacy mode).")
    args = parser.parse_args()

    if args.study_name is None:
        if args.seed is not None:
            # Per-seed optimization: create separate study for each seed
            args.study_name = f"edmpnn_mod_new_{args.dataset}_seed{args.seed}_opt"
        else:
            # Legacy mode: single study for all seeds
            args.study_name = f"edmpnn_mod_new_{args.dataset}_opt"

    # Determine Optuna direction based on task type and primary metric
    # This logic is consistent with scripts/get_direction.py
    task_type = get_task_type(args.dataset)
    primary_metric = get_primary_metric(args.dataset)
    
    if task_type == "classification":
        # Classification task: optimize roc_auc and f1, both higher is better
        direction = "maximize"
    else:
        # Regression task: set direction based on primary metric
        if primary_metric == "spearman":
            # Spearman correlation coefficient: higher is better
            direction = "maximize"
        elif primary_metric == "mae":
            # MAE: lower is better
            direction = "minimize"
        else:
            # Default to maximize (if primary metric is not defined)
            direction = "maximize"
            print(f"{YELLOW}Warning: Primary metric not defined for {args.dataset}, using maximize direction{NC}")
    
    print(f"{BLUE}Task type: {task_type}, Primary metric: {primary_metric}, Optuna direction: {direction}{NC}")
    
    # Consistency check: Verify direction calculation matches get_direction.py
    # This helps catch any discrepancies between Shell script and Python script
    try:
        import subprocess
        expected_direction = subprocess.check_output(
            ["python3", "scripts/get_direction.py", args.dataset],
            stderr=subprocess.DEVNULL,
            text=True
        ).strip()
        if expected_direction != direction:
            print(f"{YELLOW}⚠️  Warning: Direction mismatch detected!{NC}")
            print(f"   Python script calculated: {direction}")
            print(f"   get_direction.py returned: {expected_direction}")
            print(f"   Using Python script's calculation: {direction}")
        else:
            print(f"{GREEN}✓ Direction consistency verified: {direction}{NC}")
    except Exception as e:
        # If get_direction.py is not available or fails, continue with calculated direction
        print(f"{YELLOW}⚠️  Could not verify direction consistency (get_direction.py check failed): {e}{NC}")
        print(f"   Using calculated direction: {direction}")
    
    # Try to create or load study, handle race conditions and direction mismatches
    # Add retry mechanism to avoid race conditions when multiple workers start simultaneously
    MAX_RETRIES = 5
    RETRY_DELAY = 0.5 + (args.worker_id * 0.1)  # Different workers use different delays to avoid simultaneous retries
    
    study = None
    for retry in range(MAX_RETRIES):
        try:
            # Use HyperbandPruner for better pruning efficiency
            # Keep reduction_factor as integer to avoid floating rung indices in SHA
            min_resource_val = max(15, int(args.epochs * 0.2))  # At least 15 epochs, or 20% of total epochs
            pruner = HyperbandPruner(
                min_resource=min_resource_val,
                max_resource=args.epochs,
                reduction_factor=3  # Integer factor avoids float rung indices
            )
            study = optuna.create_study(
                study_name=args.study_name,
                storage=args.storage,
                load_if_exists=True,
                direction=direction,
                sampler=TPESampler(seed=42),
                pruner=pruner
            )
            # Save n_trials to study attributes for monitoring
            study.set_user_attr('n_trials', args.n_trials)
            break  # Successfully created/loaded, exit retry loop
        except Exception as e:
            # Study already exists (possibly created by another worker), try to load
            error_msg = str(e)
            error_type = type(e).__name__
            
            # Check if it's a study already exists error
            is_duplicate_error = (
                "already exists" in error_msg or 
                "UNIQUE constraint" in error_msg or 
                "DuplicatedStudyError" in error_type or
                "IntegrityError" in error_type
            )
            
            # Check if it's a record not found error (possibly another worker is creating it)
            is_not_found_error = (
                "Record does not exist" in error_msg or 
                "KeyError" in error_type
            )
            
            if is_duplicate_error or is_not_found_error:
                # If it's the last retry, try to load directly
                if retry == MAX_RETRIES - 1:
                    print(f"{YELLOW}Warning: Study creation failed after {MAX_RETRIES} retries, attempting to load existing study...{NC}")
                    try:
                        # Use HyperbandPruner for better pruning efficiency
                        # Use same pruning parameters as main creation
                        min_resource_val = max(15, int(args.epochs * 0.2))
                        pruner = HyperbandPruner(
                            min_resource=min_resource_val,
                            max_resource=args.epochs,
                            reduction_factor=3  # Integer factor to avoid float rung indices
                        )
                        study = optuna.load_study(
                            study_name=args.study_name,
                            storage=args.storage,
                            sampler=TPESampler(seed=42),
                            pruner=pruner
                        )
                        # Save n_trials if not already set
                        if 'n_trials' not in study.user_attrs:
                            study.set_user_attr('n_trials', args.n_trials)
                        # Enhanced consistency check: Verify direction matches expected
                        study_direction_name = study.direction.name if hasattr(study.direction, 'name') else str(study.direction)
                        expected_direction_name = direction.upper() if isinstance(direction, str) else direction
                        if study_direction_name != expected_direction_name:
                            print(f"{RED}⚠️  ERROR: Study direction mismatch detected!{NC}")
                            print(f"   Existing study direction: {study_direction_name}")
                            print(f"   Expected direction: {expected_direction_name}")
                            print(f"   Dataset: {args.dataset}")
                            print(f"   Task type: {task_type}, Primary metric: {primary_metric}")
                            print(f"{YELLOW}  Continuing with existing direction. If this is incorrect, please delete the study and recreate it.{NC}")
                            print(f"{YELLOW}  To fix: Delete study '{args.study_name}' from database and restart optimization.{NC}")
                        else:
                            print(f"{GREEN}✓ Study direction consistency verified: {study_direction_name}{NC}")
                        break  # Successfully loaded, exit retry loop
                    except Exception as load_error:
                        # If loading also fails, try to clean up and recreate
                        error_msg = str(load_error)
                        if "Record does not exist" in error_msg or "KeyError" in str(type(load_error).__name__):
                            print(f"{YELLOW}Warning: Study record inconsistent in database, attempting to clean up and recreate...{NC}")
                            try:
                                # Clean up inconsistent records
                                import sqlite3
                                db_path = args.storage.replace("sqlite:///", "")
                                if os.path.exists(db_path):
                                    conn = sqlite3.connect(db_path)
                                    cursor = conn.cursor()
                                    # Find and delete orphaned study_directions records
                                    cursor.execute("""
                                        DELETE FROM study_directions
                                        WHERE study_id IN (
                                            SELECT sd.study_id
                                            FROM study_directions sd
                                            LEFT JOIN studies s ON sd.study_id = s.study_id
                                            WHERE s.study_id IS NULL
                                        )
                                    """)
                                    conn.commit()
                                    conn.close()
                                    print(f"{GREEN}Cleaned up inconsistent database records{NC}")
                                
                                # Retry creating study
                                # Use HyperbandPruner for better pruning efficiency
                                # Use same pruning parameters as main creation
                                min_resource_val = max(15, int(args.epochs * 0.2))
                                pruner = HyperbandPruner(
                                    min_resource=min_resource_val,
                                    max_resource=args.epochs,
                                    reduction_factor=3  # Integer factor to avoid float rung indices
                                )
                                study = optuna.create_study(
                                    study_name=args.study_name,
                                    storage=args.storage,
                                    load_if_exists=True,
                                    direction=direction,
                                    sampler=TPESampler(seed=42),
                                    pruner=pruner
                                )
                                study.set_user_attr('n_trials', args.n_trials)
                                print(f"{GREEN}Successfully created study after cleanup{NC}")
                                break  # Successfully created, exit retry loop
                            except Exception as cleanup_error:
                                print(f"{RED}Error: Failed to clean up and recreate study: {cleanup_error}{NC}")
                                print(f"{RED}  Please manually delete the study from the database and try again{NC}")
                                raise
                        else:
                            # Other types of load errors, re-raise
                            raise
                else:
                    # Not the last retry, wait and retry
                    wait_time = RETRY_DELAY * (retry + 1)  # Exponential backoff
                    print(f"{YELLOW}Warning: Study creation failed (attempt {retry + 1}/{MAX_RETRIES}): {error_msg}{NC}")
                    print(f"{YELLOW}  Waiting {wait_time:.2f}s before retry...{NC}")
                    time.sleep(wait_time)
                    continue  # Continue retry
            else:
                # Other types of errors, re-raise
                raise
    
    # If all retries fail, raise error
    if study is None:
        raise RuntimeError(f"Failed to create or load study '{args.study_name}' after {MAX_RETRIES} retries")

    study.optimize(lambda trial: objective(trial, args), n_trials=args.n_trials)

    print(f"\n{GREEN}Optimization Finished!{NC}")
    best_trial = study.best_trial
    best_params = best_trial.params
    
    print(f"{BLUE}Best trial: {best_trial.number}{NC}")
    print(f"{BLUE}Best params: {best_params}{NC}")
    print(f"{BLUE}Best value: {best_trial.value}{NC}")
    
    # Note: Since we use 1 seed per trial, the best trial only has 1 seed's model
    # If final evaluation with 5 seeds is needed:
    # 1. Retrain 5 seeds using best hyperparameters
    # 2. Or use existing evaluate.py script for final evaluation
    
    # Copy best model logic (only copy best trial's model, may only have 1 seed)
    try:
        best_trial_dir = best_trial.user_attrs.get("trial_dir")
        if best_trial_dir and os.path.exists(best_trial_dir):
            # Store best trial models in dataset/seed{seed}/best_trial_models
            if hasattr(args, 'seed') and args.seed is not None:
                seed_num = args.seed
                final_dir = os.path.join("checkpoints", "optuna_mod", args.dataset, f"seed{seed_num}", "best_trial_models")
            else:
                final_dir = os.path.join("checkpoints", "optuna_mod", args.dataset, "seed1", "best_trial_models")
            os.makedirs(final_dir, exist_ok=True)
            
            # Get best trial's seed_list (should have all 5 seeds for legacy mode, 1 seed for per-seed mode)
            best_seed_list = best_trial.user_attrs.get("seed_list", [1, 2, 3, 4, 5])
            optimized_seed = best_trial.user_attrs.get("optimized_seed", None)
            
            # Determine if this is per-seed optimization mode
            is_per_seed_mode = (hasattr(args, 'seed') and args.seed is not None) or optimized_seed is not None
            if optimized_seed is not None:
                actual_seed = optimized_seed
            elif hasattr(args, 'seed') and args.seed is not None:
                actual_seed = args.seed
            else:
                actual_seed = None
            
            # Handle case where best trial only has 1 seed (per-seed optimization mode)
            if len(best_seed_list) == 1:
                if is_per_seed_mode:
                    # This is expected for per-seed optimization mode
                    print(f"{BLUE}Best trial from per-seed optimization (seed {best_seed_list[0]}){NC}")
                else:
                    print(f"{YELLOW}⚠️  Best trial only has 1 seed (seed {best_seed_list[0]}){NC}")
                    print(f"{YELLOW}   This is from an old format trial. New trials use all 5 seeds.{NC}")
            elif len(best_seed_list) < 5:
                print(f"{YELLOW}⚠️  Best trial has {len(best_seed_list)} seeds instead of 5{NC}")
                print(f"{YELLOW}   Seeds: {best_seed_list}{NC}")
            
            print(f"{BLUE}Copying best trial models for {len(best_seed_list)} seeds...{NC}")
            
            # Copy all seeds' models from best trial
            # For per-seed mode: model is directly in best_trial_dir
            # For legacy mode: model is in best_trial_dir/seed{seed}/
            for seed in best_seed_list:
                if is_per_seed_mode:
                    # Per-seed mode: model is directly in trial directory
                    src = os.path.join(best_trial_dir, "best_model.pth")
                else:
                    # Legacy mode: model is in seed subdirectory
                    src = os.path.join(best_trial_dir, f"seed{seed}", "best_model.pth")
                
                if os.path.exists(src):
                    import shutil
                    dst = os.path.join(final_dir, f"best_model_seed({seed}).pth")
                    shutil.copy2(src, dst)
                    print(f"{GREEN}Copied best model (seed {seed}) to {final_dir}{NC}")
                else:
                    print(f"{YELLOW}⚠️  Best model for seed {seed} not found at {src}{NC}")
            
            # Save best hyperparameters to JSON file for later use
            best_params_file = os.path.join(final_dir, "best_params.json")
            import json
            with open(best_params_file, 'w') as f:
                json.dump(best_params, f, indent=2)
            print(f"{GREEN}Best params saved to {best_params_file}{NC}")
            print(f"{GREEN}Best models saved to {final_dir}{NC}")
    except Exception as e:
        print(f"Error copying models: {e}")

if __name__ == "__main__":
    main()
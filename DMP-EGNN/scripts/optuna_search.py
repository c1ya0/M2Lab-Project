#!/usr/bin/env python3
"""
AEGNN-M Optuna Optimization Script (Wrapper)
Supports parallel optimization (Parallelization via SQLite)
Version: Enhanced with expanded search space and composite scoring

Updates:
- Expanded search space: alpha, ffn_expansion_factor, pool_type, rotate_aug are now searchable
- Composite scoring: Uses dataset-adaptive composite metrics with overfitting penalty
- Improved early stopping: Increased patience (30/50) to match edmpnn_model

Usage:
    python optuna_search.py --dataset bace --n_trials 20 --storage sqlite:///optuna.db
"""

import os
import sys
import json
import argparse
import subprocess
import optuna
from optuna.samplers import TPESampler
import time

# Set colors
GREEN = '\033[0;32m'
BLUE = '\033[0;34m'
YELLOW = '\033[1;33m'
RED = '\033[0;31m'
NC = '\033[0m'

def get_dataset_path(dataset_name):
    """Check and return dataset path (prefer processed)"""
    clean_name = dataset_name.replace('_dataset.csv', '').replace('.csv', '')
    
    processed_path = f"data/processed/{clean_name}_processed.pkl"
    csv_path = f"data/{clean_name}_dataset.csv"
    
    if os.path.exists(processed_path):
        return "--processed_data_path", processed_path
    elif os.path.exists(csv_path):
        return "--data_path", csv_path
    else:
        raise FileNotFoundError(f"Dataset not found: {dataset_name}")

def objective(trial, args):
    # ==============================
    # 1. Define hyperparameter search space (Based on V8 Insights)
    # ==============================
    # Determine dataset characteristics for adaptive search space
    dataset_name = args.dataset.lower()
    is_extremely_imbalanced = dataset_name in ['muv', 'hiv']
    is_imbalanced = dataset_name in ['tox21', 'sider', 'hiv', 'muv']
    
    # A. Basic model parameters (Architecture)
    # Adaptive hidden_dim: larger for extremely imbalanced datasets
    if is_extremely_imbalanced:
        hidden_dim = trial.suggest_categorical("hidden_dim", [128, 256])
    else:
        hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])
    num_layers = trial.suggest_int("num_layers", 2, 8)
    dropout = trial.suggest_float("dropout", 0.0, 0.5, step=0.05)
    
    # B. Optimizer parameters (Optimization)
    # V8 Insight: BACE/SIDER need high LR (1e-3) and low WD (approx 0)
    lr = trial.suggest_float("lr", 5e-5, 2e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-8, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64])
    grad_clip_norm = trial.suggest_float("grad_clip_norm", 0.1, 1.0)
    
    # C. Advanced strategies (Advanced Strategy)
    
    # 1. Loss Function Strategy
    # V8 Insight: MUV needs BCE, HIV needs Focal(0.9)
    loss_type = trial.suggest_categorical("loss_type", ["focal", "bce"])
    
    focal_alpha = 0.25
    if loss_type == "focal":
        # Expand Alpha range to cover HIV's requirement (0.9)
        focal_alpha = trial.suggest_float("focal_alpha", 0.25, 0.95, step=0.05)
        # Adaptive focal_gamma: higher range for extremely imbalanced datasets
        if is_extremely_imbalanced:
            focal_gamma = trial.suggest_float("focal_gamma", 2.0, 5.0, step=0.5)
        else:
            focal_gamma = trial.suggest_float("focal_gamma", 1.5, 3.0, step=0.5)
    else:
        # Try Label Smoothing in BCE mode
        label_smoothing = trial.suggest_float("label_smoothing", 0.0, 0.15, step=0.05)

    # 2. Mixup Strategy
    # V8 Insight: SIDER/BACE might need Mixup=False
    use_mixup = trial.suggest_categorical("use_mixup", [True, False])
    mixup_alpha = 0.2
    if use_mixup:
        mixup_alpha = trial.suggest_float("mixup_alpha", 0.1, 0.5) # Limit to avoid breaking chemical semantics
        
    # 3. Attention Head Strategy
    num_heads = trial.suggest_categorical("num_heads", [4, 8])
    
    # 4. Scheduler Strategy
    warmup_epochs = trial.suggest_int("warmup_epochs", 5, 15, step=5)
    
    # Additional hyperparameters: scheduler_type, min_lr, drop_path_rate, activation
    scheduler_type = trial.suggest_categorical("scheduler_type", ["cosine", "step", "plateau"])
    min_lr = trial.suggest_float("min_lr", 1e-7, 1e-5, log=True)
    drop_path_rate = trial.suggest_float("drop_path_rate", 0.0, 0.2, step=0.05)
    activation = trial.suggest_categorical("activation", ["SiLU", "ReLU", "LeakyReLU", "PReLU", "ELU", "SELU", "tanh"])
    
    # ==============================
    # New: Architecture hyperparameter search (previously fixed)
    # ==============================
    # GAT attention negative slope
    alpha = trial.suggest_float("alpha", 0.1, 0.3, step=0.05)
    
    # FFN expansion factor (made searchable, no longer fixed at 4)
    ffn_expansion_factor = trial.suggest_categorical("ffn_expansion_factor", [2, 4, 6, 8])
    
    # Graph pooling method
    pool_type = trial.suggest_categorical("pool_type", ["mean", "sum", "norm"])
    
    # 3D rotation augmentation
    rotate_aug = trial.suggest_categorical("rotate_aug", [True, False])
    
    # ==============================
    # 2. Set output paths
    # ==============================
    trial_dir_name = f"{args.study_name}_{trial.number}"
    trial_base_dir = os.path.join("checkpoints", "optuna", args.dataset, trial_dir_name)
    log_dir = os.path.join("runs", "optuna", args.dataset, trial_dir_name)
    
    os.makedirs(trial_base_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # ==============================
    # 3. Train for 5 seeds (similar to fusion_model)
    # ==============================
    data_arg_type, data_arg_path = get_dataset_path(args.dataset)
    
    # Calculate unique base port for this worker to avoid conflicts
    # Base 12355 + (worker_id * 100) -> Worker 0: 12355-12454, Worker 1: 12455-12554, etc.
    base_port = 12355 + (args.worker_id * 100)
    seed_list = [1, 2, 3, 4, 5]
    valid_scores = []

    print(f"\n{BLUE}[Trial {trial.number}] Starting training for 5 seeds...{NC}")
    print(f"Config: LR={lr:.1e}, Hidden={hidden_dim}, Loss={loss_type}, Scheduler={scheduler_type}, Activation={activation}")
    print(f"       Alpha={alpha:.2f}, FFN_exp={ffn_expansion_factor}, Pool={pool_type}, RotateAug={rotate_aug}")
    if loss_type == 'focal':
        print(f"        Focal Alpha={focal_alpha:.2f}, Gamma={focal_gamma:.1f}")

    for seed in seed_list:
        # Set seed-specific save directory
        seed_save_dir = os.path.join(trial_base_dir, f"seed{seed}")
        seed_log_dir = os.path.join(log_dir, f"seed{seed}")
        os.makedirs(seed_save_dir, exist_ok=True)
        os.makedirs(seed_log_dir, exist_ok=True)

        # Build command for this seed
        cmd = [
            "python3", "scripts/train_aegnnm.py",
            "--model_type", "classifier",
            data_arg_type, data_arg_path,
            "--base_port", str(base_port + seed),  # Different port for each seed
            "--seed", str(seed),  # Set seed for reproducibility
            
            # Hyperparameters
            "--hidden_dim", str(hidden_dim),
            "--num_layers", str(num_layers),
            "--dropout", str(dropout),
            "--batch_size", str(batch_size),
            "--learning_rate", str(lr),
            "--weight_decay", str(weight_decay),
            "--grad_clip_norm", str(grad_clip_norm),
            "--num_heads", str(num_heads),
            # Additional hyperparameters
            "--scheduler_type", scheduler_type,
            "--min_lr", str(min_lr),
            "--drop_path_rate", str(drop_path_rate),
            "--activation", activation,
            # New: Architecture hyperparameters (now searchable)
            "--alpha", str(alpha),
            "--ffn_expansion_factor", str(ffn_expansion_factor),
            "--aggregation", pool_type,
            
            # Fixed/Default parameters
            "--num_epochs", str(args.epochs),
            "--warmup_epochs", str(warmup_epochs),
            # Improved early stopping (increased patience to match edmpnn_model)
            "--early_stopping_patience", "30",
            "--use_smart_early_stopping",
            "--smart_early_stopping_max_patience", "50",
            
            # Paths
            "--save_dir", seed_save_dir,
            "--log_dir", seed_log_dir,
            "--split_method", "scaffold",
        ]
        
        # Conditional arguments construction
        if loss_type == "focal":
            cmd.append("--use_focal_loss")
            cmd.extend(["--focal_alpha", str(focal_alpha)])
            cmd.extend(["--focal_gamma", str(focal_gamma)])
        else:
            cmd.append("--use_bce_for_imbalanced")
            cmd.append("--auto_pos_weight")
            if label_smoothing > 0:
                cmd.extend(["--label_smoothing", str(label_smoothing)])

        if use_mixup:
            cmd.append("--enable_manifold_mixup")
            cmd.extend(["--manifold_mixup_alpha", str(mixup_alpha)])

        # New: 3D rotation augmentation
        if rotate_aug:
            cmd.append("--rotate_aug")

        # ==============================
        # 4. Execute training for this seed
        # ==============================
        print(f"  {BLUE}[Seed {seed}] Training...{NC}")
        
        try:
            with open(os.path.join(seed_log_dir, "stdout.log"), "w") as f_out, \
                 open(os.path.join(seed_log_dir, "stderr.log"), "w") as f_err:
                
                subprocess.run(cmd, stdout=f_out, stderr=f_err, check=True)
                
        except subprocess.CalledProcessError as e:
            # Read error log to show last few lines for debugging
            stderr_path = os.path.join(seed_log_dir, "stderr.log")
            error_msg = ""
            if os.path.exists(stderr_path):
                try:
                    with open(stderr_path, 'r') as f:
                        lines = f.readlines()
                        if lines:
                            # Show last 5 lines of error
                            error_msg = "\n".join(lines[-5:])
                except:
                    pass
            
            print(f"{RED}[Trial {trial.number}] Seed {seed} training failed (exit code: {e.returncode})!{NC}")
            if error_msg:
                print(f"{RED}Last error lines:{NC}")
                print(error_msg)
            print(f"{RED}Full logs: {seed_log_dir}{NC}")
            valid_scores.append(0.0)
            continue

        # ==============================
        # 5. Read results for this seed
        # ==============================
        history_path = os.path.join(seed_save_dir, "training_history.json")
        
        if not os.path.exists(history_path):
            print(f"{YELLOW}[Trial {trial.number}] Seed {seed}: Warning: No history file found.{NC}")
            valid_scores.append(0.0)
            continue
            
        try:
            with open(history_path, 'r') as f:
                history = json.load(f)
            
            # Get test results (with fallback to best_metrics)
            test_results = history.get('test_results', {})
            best_metrics = history.get('best_metrics', {})
            
            # Determine dataset characteristics
            dataset_name = args.dataset.lower()
            is_extremely_imbalanced = dataset_name in ['muv', 'hiv']
            is_imbalanced = dataset_name in ['tox21', 'sider', 'hiv', 'muv']
            
            # Extract metrics (with fallback to best_metrics if test_results not available)
            auroc = test_results.get('roc_auc', best_metrics.get('roc_auc', history.get('best_val_auc', 0.0)))
            pr_auc = test_results.get('pr_auc', best_metrics.get('pr_auc', 0.0))
            f1 = test_results.get('f1', best_metrics.get('f1', 0.0))
            precision = test_results.get('precision', best_metrics.get('precision', 0.0))
            recall = test_results.get('recall', best_metrics.get('recall', 0.0))
            
            # Calculate overfitting penalty
            train_losses = history.get('train_losses', [])
            val_loss = history.get('best_val_loss', 1.0)
            if train_losses and val_loss > 0:
                train_loss = train_losses[-1]
                overfitting_ratio = max(0, (train_loss - val_loss) / val_loss)
            else:
                overfitting_ratio = 0
            
            # Compute composite score based on dataset characteristics
            if is_extremely_imbalanced:
                # For extremely imbalanced datasets (MUV, HIV): prioritize PR-AUC and F1
                if pr_auc > 0:
                    primary_metric = pr_auc
                else:
                    primary_metric = f1
                secondary_metric = f1 if f1 > 0 else precision
                score = 0.6 * primary_metric + 0.3 * secondary_metric + 0.1 * auroc
            elif is_imbalanced:
                # For imbalanced datasets (TOX21, SIDER): balance AUROC and F1
                score = 0.5 * auroc + 0.4 * f1 + 0.1 * precision
            else:
                # For balanced datasets: primarily use AUROC
                score = 0.7 * auroc + 0.3 * f1
            
            # Apply overfitting penalty (reduce score if severe overfitting)
            if overfitting_ratio > 0.5:  # Overfitting ratio > 50%
                penalty = 0.2 * min(overfitting_ratio, 1.0)
                score = score * (1 - penalty)
            
            valid_scores.append(score)
            print(f"  {GREEN}[Seed {seed}] Score: {score:.4f} (AUROC: {auroc:.4f}, F1: {f1:.4f}, PR-AUC: {pr_auc:.4f}){NC}")

        except Exception as e:
            print(f"{RED}[Trial {trial.number}] Seed {seed}: Error reading results: {e}{NC}")
            valid_scores.append(0.0)

    # ==============================
    # 6. Store trial metadata and return average score
    # ==============================
    if len(valid_scores) == 0:
        print(f"{RED}[Trial {trial.number}] No valid scores collected!{NC}")
        return 0.0

    # Store seed list and trial directory in trial attributes (for later model copying)
    trial.set_user_attr("seed_list", seed_list)
    trial.set_user_attr("trial_dir", trial_base_dir)
    trial.set_user_attr("valid_scores", valid_scores)

    # Return average score (similar to fusion_model)
    avg_score = sum(valid_scores) / len(valid_scores)
    print(f"{GREEN}[Trial {trial.number}] Average Score: {avg_score:.4f} (across {len(valid_scores)} seeds){NC}")
    
    return avg_score

def main():
    parser = argparse.ArgumentParser(description="AEGNN-M Optuna Search")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., bace)")
    parser.add_argument("--study_name", type=str, default=None, help="Optuna study name")
    parser.add_argument("--storage", type=str, default="sqlite:///optuna.db", help="Database URL")
    parser.add_argument("--n_trials", type=int, default=20, help="Trials per process")
    parser.add_argument("--epochs", type=int, default=100, help="Max epochs")
    parser.add_argument("--worker_id", type=int, default=0, help="Worker ID for port allocation")
    
    args = parser.parse_args()
    
    if args.study_name is None:
        args.study_name = f"aegnn_{args.dataset}_opt"
        
    print(f"{BLUE}Starting Optuna Search for {args.dataset}{NC}")
    print(f"Storage: {args.storage}")
    
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
        direction="maximize",
        sampler=TPESampler(seed=42)
    )
    
    print(f"{GREEN}Study loaded. Running {args.n_trials} trials...{NC}")
    
    try:
        study.optimize(lambda trial: objective(trial, args), n_trials=args.n_trials)
    except KeyboardInterrupt:
        print(f"\n{YELLOW}Optimization interrupted by user.{NC}")
    
    print("\n" + "="*50)
    print(f"{GREEN}Optimization Finished!{NC}")
    print(f"Best Trial (ID: {study.best_trial.number}):")
    print(f"  Value (Composite Score): {study.best_value:.4f}")
    print("  Params:")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")
    print("="*50)

    # ==============================
    # Copy best trial's 5 seed models to final directory (similar to fusion_model)
    # ==============================
    try:
        best_trial = study.best_trial
        best_trial_dir = best_trial.user_attrs.get("trial_dir")
        best_seed_list = best_trial.user_attrs.get("seed_list", [1, 2, 3, 4, 5])
        
        if best_trial_dir and os.path.exists(best_trial_dir):
            # Create final directory for best trial models
            final_dir = os.path.join("checkpoints", "optuna", args.dataset, "best_trial_models")
            os.makedirs(final_dir, exist_ok=True)
            
            print(f"\n{BLUE}Copying best trial's models (5 seeds) to final directory...{NC}")
            for seed in best_seed_list:
                source_dir = os.path.join(best_trial_dir, f"seed{seed}")
                source_model = os.path.join(source_dir, "best_model.pth")
                dest_model = os.path.join(final_dir, f"best_model_seed({seed}).pth")
                
                if os.path.exists(source_model):
                    import shutil
                    shutil.copy2(source_model, dest_model)
                    print(f"  {GREEN}✓ Copied seed {seed} model{NC}")
                else:
                    print(f"  {YELLOW}⚠ Seed {seed} model not found at {source_model}{NC}")
            
            print(f"{GREEN}Best trial models saved to: {final_dir}{NC}")
        else:
            print(f"{YELLOW}Warning: Best trial directory not found, skipping model copy.{NC}")
    except Exception as e:
        print(f"{YELLOW}Warning: Could not copy best trial models: {e}{NC}")

    # Save best params to file
    results_dir = "optuna_results"
    os.makedirs(results_dir, exist_ok=True)
    best_params_path = os.path.join(results_dir, f"{args.dataset}_best_params.json")
    
    result_data = {
        "dataset": args.dataset,
        "study_name": args.study_name,
        "best_trial_id": study.best_trial.number,
        "best_value": study.best_value,
        "best_params": study.best_params
    }
    
    try:
        with open(best_params_path, "w") as f:
            json.dump(result_data, f, indent=4)
        print(f"{GREEN}Best parameters saved to: {best_params_path}{NC}")
    except Exception as e:
        print(f"{YELLOW}Warning: Could not save best_params.json: {e}{NC}")


if __name__ == "__main__":
    main()

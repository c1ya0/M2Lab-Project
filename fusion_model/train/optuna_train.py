import os
import time
import json
import subprocess
import sys

import optuna
import optuna.visualization as vis
import torch
import argparse
import numpy as np
from types import SimpleNamespace
from nemo_chem.models.megamolbart import MegaMolBARTModel
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
import plotly.io as pio
from tdc import Evaluator
from datetime import datetime
from tqdm import trange

# -------------------------------------------------------
from core.prepare_dataset import load_dataset
from core.dmpegnn_dataset import load_dmpegnn_dataset
from core.train_utils import train, valid
from core.utils import (
    set_seed, save_training_log, 
    plot_loss_curve, format_time
)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")


_3D_MODEL_TYPES = frozenset([
    "DMPEGNN", "DMPEGNN_DESC", "DMPEGNN_MMB_DESC",
    "AEGNN",   "AEGNN_DESC",          # AEGNN-M — same 3D data pipeline as DMPEGNN
])

def get_dataset_train_size(args) -> int:
    """Load training set once (seed=1) to determine n_train for capacity scaling."""
    if args.model_type in _3D_MODEL_TYPES:
        train_ds, _, _ = load_dmpegnn_dataset(
            data_name=args.data_name,
            data_path=args.data_path,
            seed=1,
        )
    else:
        train_ds, _, _ = load_dataset(
            data_name=args.data_name,
            data_path=args.data_path,
            seed=1,
        )
    return len(train_ds)

KALEIDO_AVAILABLE = True

# -------------------- Argument Parsing --------------------
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, required=True, choices=['GCN', 'MMB', 'DESC', 'GCN_MMB', 'MMB_DESC', 'GCN_DESC', 'GCN_MMB_DESC', 'MPN_MMB_DESC', 'MPN', 'MPN_DESC', 'MPN_MMB', 'DMPEGNN', 'DMPEGNN_DESC', 'DMPEGNN_MMB_DESC', 'AEGNN', 'AEGNN_DESC'])
    parser.add_argument('--data_name', type=str, required=True)
    parser.add_argument('--task_type', type=str, required=True, choices=['regression', 'classification'])
    parser.add_argument('--loss_function', type=str, required=True, choices=['MAE', 'BCE'])
    parser.add_argument('--metric', type=str, required=True, choices=['MAE', 'Spearman', 'ROC-AUC', 'PR-AUC'])
    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--pretrained_path', type=str, default='/models/MegaMolBART_0_2_0.nemo')
    parser.add_argument('--data_path', type=str, default='data/data_tdc')
    parser.add_argument('--seed_list', type=int, nargs='+', default=[1, 2, 3, 4, 5])
    parser.add_argument('--num_tasks', type=int, default=1)
    parser.add_argument('--num_trials', type=int, default=5)
    parser.add_argument('--log_transform', action='store_true', default=False,
                        help='Apply log1p to regression targets before training and expm1 after prediction.')
    parser.add_argument('--val_loss_threshold', type=float, default=0.0,
                        help='Relative rise above best val_loss that counts as deterioration (0.0 = disabled).')
    return parser.parse_args()


# -------------------- Optuna Hyperparameter Sampling --------------------
def sample_mlp_params(trial, n_train: int = None):
    # Dynamic capacity tiers based on training set size
    # Small: n_train < 700 | Medium: 700 <= n_train < 2000 | Large: n_train >= 2000
    if n_train is not None and n_train < 700:
        hidden_choices = [32, 64, 128]
        max_layers = 3
    elif n_train is not None and n_train < 2000:
        hidden_choices = [64, 128, 256]
        max_layers = 4
    else:
        hidden_choices = [16, 32, 64, 128, 256]
        max_layers = 5
    return {
        'mlp_hidden_dim': trial.suggest_categorical('mlp_hidden_dim', hidden_choices),
        'mlp_num_layers': trial.suggest_int('mlp_num_layers', 1, max_layers),
        'mlp_activation': trial.suggest_categorical('mlp_activation', ['relu', 'gelu']),
        'mlp_dropout': trial.suggest_float('mlp_dropout', 0.0, 0.5, step=0.05),
        'mlp_norm_type': trial.suggest_categorical('mlp_norm_type', ['LayerNorm']),
    }

def sample_gcn_params(trial):
    return {
        'gcn_hidden_dim': trial.suggest_categorical('gcn_hidden_dim', [16, 32, 64, 128, 256]),
        'gcn_output_dim': trial.suggest_categorical('gcn_output_dim', [16, 32, 64, 128, 256]),
        'gcn_num_layers': trial.suggest_int('gcn_num_layers', 1, 3),
        'gcn_activation': trial.suggest_categorical('gcn_activation', ['relu', 'gelu']),
        'gcn_dropout': trial.suggest_float('gcn_dropout', 0.0, 0.5, step=0.05),
        'gcn_norm_type': trial.suggest_categorical('gcn_norm_type', ['LayerNorm', 'BatchNorm']),
        'gcn_pooling': trial.suggest_categorical('gcn_pooling', ['mean', 'max', 'add'])
    }

def sample_mpn_params(trial):
    return {
        'mpn_hidden_size': trial.suggest_categorical('mpn_hidden_size', [64, 128, 256, 300]),
        'mpn_depth': trial.suggest_int('mpn_depth', 2, 6),
        'mpn_dropout': trial.suggest_float('mpn_dropout', 0.0, 0.3, step=0.05),
        'mpn_activation': trial.suggest_categorical('mpn_activation', ['ReLU', 'LeakyReLU', 'PReLU', 'tanh', 'SELU', 'ELU']),
        'mpn_aggregation': trial.suggest_categorical('mpn_aggregation', ['mean', 'sum', 'norm'])
    }

def sample_dmpegnn_params(trial, n_train: int = None):
    # Dynamic capacity tiers based on training set size
    # Small:  n_train < 700   → hidden [64, 128],        max_layers 3
    # Medium: 700 ≤ n_train < 2000 → hidden [128, 256, 384], max_layers 4
    # Large:  n_train ≥ 2000  → hidden [128, 256, 384],  max_layers 6
    if n_train is not None and n_train < 700:
        hidden_choices = [64, 128]
        max_layers = 3
    elif n_train is not None and n_train < 2000:
        hidden_choices = [128, 256, 384]
        max_layers = 4
    else:
        hidden_choices = [128, 256, 384]
        max_layers = 6
    return {
        'dmpegnn_hidden_dim': trial.suggest_categorical('dmpegnn_hidden_dim', hidden_choices),
        'dmpegnn_num_layers': trial.suggest_int('dmpegnn_num_layers', 2, max_layers),
        'dmpegnn_num_heads': trial.suggest_categorical('dmpegnn_num_heads', [4, 8]),
        'dmpegnn_dropout': trial.suggest_float('dmpegnn_dropout', 0.3, 0.5, step=0.05),
        'dmpegnn_dmp_steps': trial.suggest_int('dmpegnn_dmp_steps', 1, 4),
        'dmpegnn_pool_type': trial.suggest_categorical('dmpegnn_pool_type', ['mean', 'sum']),
    }

def sample_aegnn_params(trial, n_train: int = None):
    """Hyperparameter search space for AEGNN-M backbone.

    Mirrors sample_dmpegnn_params() but WITHOUT dmpegnn_dmp_steps,
    because AEGNN-M uses a single phi_e MLP (no directed message passing).
    Uses 'aegnn_' prefix to keep these params distinct from DMPEGNN's.
    """
    # Same capacity tiers as DMPEGNN
    if n_train is not None and n_train < 700:
        hidden_choices = [64, 128]
        max_layers = 3
    elif n_train is not None and n_train < 2000:
        hidden_choices = [128, 256, 384]
        max_layers = 4
    else:
        hidden_choices = [128, 256, 384]
        max_layers = 6
    return {
        'aegnn_hidden_dim':  trial.suggest_categorical('aegnn_hidden_dim',  hidden_choices),
        'aegnn_num_layers':  trial.suggest_int('aegnn_num_layers',  2, max_layers),
        'aegnn_num_heads':   trial.suggest_categorical('aegnn_num_heads',   [4, 8]),
        'aegnn_dropout':     trial.suggest_float('aegnn_dropout',     0.3, 0.5, step=0.05),
        'aegnn_pool_type':   trial.suggest_categorical('aegnn_pool_type',   ['mean', 'sum']),
        # NOTE: no aegnn_dmp_steps — AEGNN-M has no directed message passing
    }

def sample_optimizer_params(trial):
    return {
        'lr': trial.suggest_float('lr', 1e-5, 1e-3, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [32, 64]),
        'weight_decay': trial.suggest_float('weight_decay', 1e-4, 1e-2, log=True),
        # scheduler_type: StepLR 移除，僅保留 cosine 與 plateau
        'scheduler_type': trial.suggest_categorical('scheduler_type', ['cosine', 'plateau']),
    }

def sample_hyperparameters(trial, args, n_train: int = None):
    sampled = sample_optimizer_params(trial)
    sampled.update(sample_mlp_params(trial, n_train=n_train))
    if 'GCN' in args.model_type:
        sampled.update(sample_gcn_params(trial))
        sampled['GCN_OUTPUT_DIM'] = 1 if args.model_type == 'GCN' else int(sampled['gcn_output_dim'])
    if 'MPN' in args.model_type:
        sampled.update(sample_mpn_params(trial))
    if 'DMPEGNN' in args.model_type:
        sampled.update(sample_dmpegnn_params(trial, n_train=n_train))
    # AEGNN-M: sample aegnn_* params (no dmp_steps; distinct from DMPEGNN)
    if 'AEGNN' in args.model_type:
        sampled.update(sample_aegnn_params(trial, n_train=n_train))
    return sampled

# -------------------- Objective Function for Optuna --------------------
def objective(trial, args, n_train: int = None):
    # === hyperparameters ===
    sampled = sample_hyperparameters(trial, args, n_train=n_train)
    for k, v in sampled.items():
        setattr(args, k, v)

    # === directories ===
    SAVE_DIR = os.path.join(ROOT_DIR, "results", "optuna_results", args.model_type.lower(), args.data_name)

    LOG_DIR = os.path.join(SAVE_DIR, "log")
    TRIAL_LOG_DIR = os.path.join(LOG_DIR, f"trial_{trial.number}")
    os.makedirs(TRIAL_LOG_DIR, exist_ok=True)

    CHECKPOINT_DIR = os.path.join(SAVE_DIR, "checkpoint")
    TRIAL_CHECKPOINT_DIR = os.path.join(CHECKPOINT_DIR, f"trial_{trial.number}")
    os.makedirs(TRIAL_CHECKPOINT_DIR, exist_ok=True)

    # 儲存本次 trial 的超參，供 seed_train.py 使用
    hp_json_path = os.path.join(TRIAL_CHECKPOINT_DIR, "hparams.json")
    with open(hp_json_path, "w") as f:
        json.dump(sampled, f, indent=2)

    # 啟動每個 seed 的子 process，實際訓練在 train/seed_train.py 內進行
    seed_procs = []
    for SEED in args.seed_list:
        seed_output_dir = os.path.join(TRIAL_CHECKPOINT_DIR, f"seed{SEED}")
        os.makedirs(seed_output_dir, exist_ok=True)

        # GPU 映射：seed 1,2 → GPU0；seed 3,4,5 → GPU1
        if SEED in [1, 2]:
            visible = "0"
        else:
            visible = "1"

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = visible

        cmd = [
            sys.executable,
            "-m",
            "train.seed_train",
            "--model_type",
            args.model_type,
            "--data_name",
            args.data_name,
            "--task_type",
            args.task_type,
            "--loss_function",
            args.loss_function,
            "--metric",
            args.metric,
            "--num_epochs",
            str(args.num_epochs),
            "--patience",
            str(args.patience),
            "--data_path",
            args.data_path,
            "--pretrained_path",
            args.pretrained_path,
            "--num_tasks",
            str(args.num_tasks),
            "--seed",
            str(SEED),
            "--hp_json",
            hp_json_path,
            "--output_dir",
            seed_output_dir,
            "--trial_number",
            str(trial.number),
            *(["--log_transform"] if args.log_transform else []),
            "--val_loss_threshold",
            str(args.val_loss_threshold),
            "--loss_patience_limit",
            str(args.loss_patience_limit),
        ]

        # seed 1: 直接輸出到前景，讓 tqdm 進度條顯示在主 terminal
        if SEED == 1:
            proc = subprocess.Popen(cmd, env=env)
            stdout_f = None
            stderr_f = None
        else:
            stdout_path = os.path.join(TRIAL_LOG_DIR, f"stdout_seed{SEED}.log")
            stderr_path = os.path.join(TRIAL_LOG_DIR, f"stderr_seed{SEED}.log")
            stdout_f = open(stdout_path, "w")
            stderr_f = open(stderr_path, "w")
            proc = subprocess.Popen(cmd, stdout=stdout_f, stderr=stderr_f, env=env)

        seed_procs.append((SEED, proc, stdout_f, stderr_f, seed_output_dir))

    # === epoch-level pruning：透過各 seed 寫出的 training_progress.json 監控 ===
    minimize_metrics = ["MAE"]
    maximize_metrics = ["Spearman", "ROC-AUC", "PR-AUC"]

    last_reported_epoch = -1
    try:
        while True:
            # 若全部子 process 都結束就跳出
            if all(proc.poll() is not None for _, proc, *_ in seed_procs):
                break

            progress_list = []
            for SEED, proc, *_rest in seed_procs:
                seed_output_dir = _rest[-1]
                progress_path = os.path.join(seed_output_dir, "training_progress.json")
                if os.path.exists(progress_path):
                    try:
                        with open(progress_path, "r") as f:
                            prog = json.load(f)
                        progress_list.append(prog)
                    except Exception:
                        continue

            if progress_list:
                # 使用所有 seed 目前 epoch 的最小值當作 step（確保所有 seed 都已到達此 epoch）
                current_epoch = min(p.get("epoch", -1) for p in progress_list)
                if current_epoch > last_reported_epoch:
                    metrics = [p.get("valid_metric") for p in progress_list if p.get("valid_metric") is not None]
                    if metrics:
                        avg_valid_metric = float(np.mean(metrics))
                        trial.report(avg_valid_metric, current_epoch)
                        last_reported_epoch = current_epoch
                        if trial.should_prune():
                            # 終止所有 seed 的訓練
                            for _, proc, stdout_f, stderr_f, _ in seed_procs:
                                if proc.poll() is None:
                                    proc.terminate()
                            raise optuna.TrialPruned()

            time.sleep(5)

        # 所有 seed 訓練結束後，讀取各自 summary.json 的 best_valid_metric，取平均作為 trial value
        valid_metrics = []
        for SEED, proc, stdout_f, stderr_f, seed_output_dir in seed_procs:
            if stdout_f is not None:
                stdout_f.close()
            if stderr_f is not None:
                stderr_f.close()
            summary_path = os.path.join(seed_output_dir, "training_summary.json")
            if os.path.exists(summary_path):
                try:
                    with open(summary_path, "r") as f:
                        summ = json.load(f)
                    if "best_valid_metric" in summ and summ["best_valid_metric"] is not None:
                        valid_metrics.append(float(summ["best_valid_metric"]))
                except Exception:
                    continue

        if not valid_metrics:
            # 若所有 seed 都失敗，依 direction 回傳最差值
            return float("inf") if args.metric in minimize_metrics else float("-inf")

        trial.set_user_attr("valid_metrics", [float(v) for v in valid_metrics])
        trial.set_user_attr("trial_dir", TRIAL_CHECKPOINT_DIR)
        trial.set_user_attr("seed_list", args.seed_list)

        return float(np.mean(valid_metrics))

    except optuna.exceptions.TrialPruned:
        raise

if __name__ == '__main__':
    args = SimpleNamespace(**vars(get_args())) # convert argparse.Namespace to SimpleNamespace
    args.data_path = os.path.join(ROOT_DIR, args.data_path)
    
    # === directories ===
    SAVE_DIR = os.path.join(ROOT_DIR, "results", "optuna_results", args.model_type.lower(), args.data_name)
    
    LOG_DIR = os.path.join(SAVE_DIR, "log")
    os.makedirs(LOG_DIR, exist_ok=True)
    
    CHECKPOINT_DIR = os.path.join(SAVE_DIR, "checkpoint")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    # === optuna study ===
    # direction
    if args.metric == 'MAE':
        direction = 'minimize'
    elif args.metric in ['Spearman', 'ROC-AUC', 'PR-AUC']:
        direction = 'maximize'

    # Compute training set size once for dynamic capacity scaling and patience.
    # get_dataset_train_size() also writes the seed=1 cache as a side effect.
    n_train = get_dataset_train_size(args)
    tier = "small (<700)" if n_train < 700 else ("medium (700–2000)" if n_train < 2000 else "large (≥2000)")
    print(f"[INFO] n_train={n_train} → capacity tier: {tier}")

    # Pre-warm dataset caches for all seeds.
    # DMPEGNN models require expensive 3D conformer generation (ETKDG + MMFF) that is
    # cached per-seed under data/processed_tdc_data_dmpegnn/<dataset>/seed<N>/.
    # Without this step, seeds 2-5 build their caches inside the subprocess while
    # seed 1 is already training, making the run appear single-process.
    # Pre-warming here (sequential, in the parent) ensures every subprocess finds
    # its cache ready and starts training immediately.
    if args.model_type in _3D_MODEL_TYPES:
        seeds_needing_cache = [
            s for s in args.seed_list
            if s != 1  # seed 1 already warmed by get_dataset_train_size above
        ]
        if seeds_needing_cache:
            print(f"[INFO] Pre-warming 3D dataset caches ({args.model_type}) for seeds {seeds_needing_cache} ...")
            for seed in seeds_needing_cache:
                load_dmpegnn_dataset(data_name=args.data_name, data_path=args.data_path, seed=seed)
            print("[INFO] All seed caches ready.")

    # Dynamic patience: small datasets converge quickly; larger ones need more epochs.
    # DMPEGNN models converge faster than MPN/GCN due to 3D attention, so they get
    # a tighter patience cap even on large datasets to avoid plateau over-waiting.
    # Both DMPEGNN and AEGNN-M use 3D equivariant attention; apply the same
    # tighter patience cap on large datasets (they converge faster than GCN/MPN).
    is_3d_model = args.model_type in _3D_MODEL_TYPES
    if n_train < 700:
        dynamic_patience = 50
    elif n_train < 2000:
        dynamic_patience = 50
    else:
        dynamic_patience = 60 if is_3d_model else 100
    if dynamic_patience != args.patience:
        print(f"[INFO] Overriding patience {args.patience} → {dynamic_patience} (n_train={n_train}, model={args.model_type})")
        args.patience = dynamic_patience

    # loss_patience_limit: how many consecutive epochs of val_loss deterioration trigger early stop.
    # Rules:
    #   - val_loss_threshold > 0: user explicitly enabled → use as-is, limit=25
    #   - val_loss_threshold == 0 AND metric==MAE (regression) AND n_train>=2000:
    #       auto-enable with threshold=0.15; val_loss is tightly coupled to MAE so
    #       a rising loss reliably signals plateau even on large datasets.
    #   - otherwise (Spearman/AUC tasks, or small datasets already covered by metric_patience):
    #       keep disabled; loss and rank/classification metric can decouple.
    if args.val_loss_threshold > 0.0:
        args.loss_patience_limit = 25
    elif args.metric == "MAE" and n_train >= 2000:
        args.val_loss_threshold = 0.15
        args.loss_patience_limit = 25
    else:
        args.loss_patience_limit = 0
    if args.loss_patience_limit > 0:
        print(f"[INFO] val_loss monitoring enabled: threshold={args.val_loss_threshold:.2f}, loss_patience={args.loss_patience_limit}")

    study_start_time = time.time()
    study = optuna.create_study(
        direction=direction,
        sampler=optuna.samplers.TPESampler(),
        # pruner=optuna.pruners.NopPruner(),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=50, n_startup_trials=10),
        study_name = f"opt_{args.model_type.lower()}_{args.data_name}",
        storage=f"sqlite:///{SAVE_DIR}/optuna_study.db",
        load_if_exists=True, # load existing study
    )
    # Resume 時：n_trials 表示「總共」要幾個 trial，只補跑不足的數量
    n_existing = len(study.trials)
    n_rest = max(0, args.num_trials - n_existing)
    if n_rest > 0:
        study.optimize(lambda trial: objective(trial, args, n_train=n_train), n_trials=n_rest)
    elif n_existing > 0:
        print(f"[INFO] Study already has {n_existing} trials (>= num_trials={args.num_trials}), skip optimize.")
    study_end_time = time.time()
    study_time = study_end_time - study_start_time
    
    # === best trial ===
    best_trial_id = study.best_trial.number
    best_params = study.best_params
    best_trial_info = {
        "best_trial_id": best_trial_id,
        "best_params": best_params
    }
    best_trial_info_path = os.path.join(LOG_DIR, "best_trial_info.json")
    with open(best_trial_info_path, "w") as f:
        json.dump(best_trial_info, f, indent=4)

    # best trial model（seed_train 寫入 trial_N/seed{SEED}/best_model.pth，非 trial_N/best_model_seed(SEED).pth）
    best_trial = study.best_trial
    best_trial_dir = best_trial.user_attrs["trial_dir"]
    best_seed_list = best_trial.user_attrs["seed_list"]

    def _best_model_src(trial_dir, seed):
        p = os.path.join(trial_dir, f"best_model_seed({seed}).pth")
        if os.path.exists(p):
            return p
        p = os.path.join(trial_dir, f"seed{seed}", "best_model.pth")
        return p if os.path.exists(p) else None

    for final_subdir in ("best_trial_models", "best_trial_models_50"):
        FINAL_DIR = os.path.join(SAVE_DIR, final_subdir)
        os.makedirs(FINAL_DIR, exist_ok=True)
        for seed in best_seed_list:
            source_path = _best_model_src(best_trial_dir, seed)
            if source_path:
                destination_path = os.path.join(FINAL_DIR, f"best_model_seed({seed}).pth")
                torch.save(torch.load(source_path), destination_path)

    # === study summary ===
    # study_path = os.path.join(LOG_DIR, "study_summary.txt")
    study_path = os.path.join(LOG_DIR, f"study_summary_{TIMESTAMP}.txt")
    with open(study_path, "w") as f:
        f.write("=== Study summary ===\n")
        f.write(f"Study time: {format_time(study_time)}\n")
        f.write(f"Total trials: {len(study.trials)}\n")
        
        completed = sum(1 for t in study.trials if t.state.name == "COMPLETE")
        pruned = sum(1 for t in study.trials if t.state.name == "PRUNED")
        f.write(f"Trials completed: {completed}, pruned: {pruned}\n")
        
        f.write(f"Best trial id: {study.best_trial.number}\n")
        f.write(f"Best value: {study.best_value:.3f}\n")
        f.write("\nBest params:\n")
        for k, v in study.best_params.items():
            f.write(f"{k}: {v}\n")
                  
    # === visualization ===
    img_format = "png"
    figs = {
        "plot_slice": vis.plot_slice(study),
        "plot_param_importances": vis.plot_param_importances(study),
        
        "plot_parallel_coordinate": vis.plot_parallel_coordinate(study),
        "plot_intermediate_values": vis.plot_intermediate_values(study),
        "plot_timeline": vis.plot_timeline(study),

        "plot_optimization_history": vis.plot_optimization_history(study),
    }

    if KALEIDO_AVAILABLE:
        for name, fig in figs.items():
            path = os.path.join(LOG_DIR, f"{name}.{img_format}")
            try:
                fig.write_image(path)
            except Exception as e:
                print("[WARN] Skip saving Optuna figures because Kaleido/Chrome is not available.")
                globals()["KALEIDO_AVAILABLE"] = False
                break

import os
import json
import time
import argparse
from types import SimpleNamespace
from datetime import datetime

import torch
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader as TorchDataLoader
from torch_geometric.loader import DataLoader as PyGDataLoader
from tdc import Evaluator
from tqdm import trange

from core.prepare_dataset import load_dataset
from core.dmpegnn_dataset import load_dmpegnn_dataset, collate_dmpegnn_multi
from core.models import (
    GCN_Model,
    MegaMolBART_Finetuned_Model,
)
from core.train_utils import train, valid
from core.utils import (
    set_seed, save_training_log,
    plot_loss_curve,
)
from core.model_factory import get_model

from nemo_chem.models.megamolbart import MegaMolBARTModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 需要 3D 圖資料管線（load_dmpegnn_dataset + collate_dmpegnn_multi）的模型類型。
# 與 optuna_train.py 中的 _3D_MODEL_TYPES 保持一致。
_3D_MODEL_TYPES = frozenset(["DMPEGNN", "DMPEGNN_DESC", "DMPEGNN_MMB_DESC", "AEGNN", "AEGNN_DESC"])


def _apply_log1p_to_dataset(dataset, model_type: str) -> None:
    """In-place log1p transform of regression targets.

    For MoleculeDataset (prepare_dataset): transforms the .labels tensor,
    which __getitem__ wraps into Data.y on the fly.

    For DMPEGNNGraphDataset (dmpegnn_dataset): transforms every graph's .y
    tensor (used by collate_dmpegnn_multi → batch.y) AND the .labels list
    (used by pos_weight computation, though irrelevant for regression).
    """
    if model_type in _3D_MODEL_TYPES:
        dataset.labels = [float(np.log1p(l)) for l in dataset.labels]
        for g in dataset.graphs:
            g.y = torch.log1p(g.y)
    else:
        dataset.labels = torch.log1p(dataset.labels)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, required=True)
    parser.add_argument("--data_name", type=str, required=True)
    parser.add_argument("--task_type", type=str, required=True, choices=["regression", "classification"])
    parser.add_argument("--loss_function", type=str, required=True, choices=["MAE", "BCE"])
    parser.add_argument("--metric", type=str, required=True, choices=["MAE", "Spearman", "ROC-AUC", "PR-AUC"])
    parser.add_argument("--num_epochs", type=int, required=True)
    parser.add_argument("--patience", type=int, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--pretrained_path", type=str, default="/models/MegaMolBART_0_2_0.nemo")
    parser.add_argument("--num_tasks", type=int, default=1)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--hp_json", type=str, required=True, help="Path to JSON file with sampled hyperparameters.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to write checkpoints and progress.")
    parser.add_argument("--trial_number", type=int, required=True, help="Optuna trial number for display/logging.")
    parser.add_argument("--log_transform", action="store_true", default=False,
                        help="Apply log1p to regression targets before training and expm1 after prediction.")
    parser.add_argument("--val_loss_threshold", type=float, default=0.0,
                        help="Relative rise above best val_loss that counts as deterioration (0.0 = disabled).")
    parser.add_argument("--loss_patience_limit", type=int, default=0,
                        help="Consecutive epochs above val_loss_threshold before early stop (0 = disabled).")
    return parser.parse_args()


def build_model_and_loaders(args: SimpleNamespace):
    """Build model and dataloaders for a single seed, mirroring optuna_train.objective."""
    # === dataset ===
    if args.model_type in _3D_MODEL_TYPES:
        train_dataset, valid_dataset, _ = load_dmpegnn_dataset(
            data_name=args.data_name,
            data_path=args.data_path,
            seed=args.seed,
        )
    else:
        train_dataset, valid_dataset, _ = load_dataset(
            data_name=args.data_name,
            data_path=args.data_path,
            seed=args.seed,
        )

    drop_last = True if args.mlp_norm_type == "BatchNorm" else False
    if args.model_type in _3D_MODEL_TYPES:
        # Use standard PyTorch DataLoader + custom collate so multi-conformer batches are built correctly.
        DataLoaderCls = TorchDataLoader
        collate_fn = collate_dmpegnn_multi
    else:
        DataLoaderCls = PyGDataLoader
        collate_fn = None
    train_loader = DataLoaderCls(train_dataset, args.batch_size, shuffle=True, drop_last=drop_last, collate_fn=collate_fn)
    valid_loader = DataLoaderCls(valid_dataset, args.batch_size, shuffle=False, drop_last=False, collate_fn=collate_fn)

    # === models ===
    gcn_model, megamolbart_model = None, None

    # GCN model
    if "GCN" in args.model_type:
        gcn_model = GCN_Model(
            input_dim=75,
            hidden_dim=args.gcn_hidden_dim,
            output_dim=args.GCN_OUTPUT_DIM,
            num_layers=args.gcn_num_layers,
            dropout=args.gcn_dropout,
            activation=args.gcn_activation,
            norm_type=args.gcn_norm_type,
            pooling=args.gcn_pooling,
        )
        for p in gcn_model.parameters():
            p.requires_grad = True

    # MegaMolBART model
    if "MMB" in args.model_type:
        trainer = pl.Trainer(
            max_epochs=1,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            precision=16 if torch.cuda.is_available() else 32,
            enable_progress_bar=False,
        )
        pretrained_model = MegaMolBARTModel.restore_from(args.pretrained_path, trainer=trainer)
        for p in pretrained_model.parameters():
            p.requires_grad = False
        megamolbart_model = MegaMolBART_Finetuned_Model(pretrained_model)

    # Combined model
    task_output_dims = [1] * args.num_tasks
    model = get_model(
        args,
        args.model_type,
        task_output_dims,
        gcn_model=gcn_model,
        megamolbart_model=megamolbart_model,
        gcn_output_dim=args.GCN_OUTPUT_DIM if hasattr(args, "GCN_OUTPUT_DIM") else None,
        mlp_hidden_dim=args.mlp_hidden_dim,
        mlp_num_layers=args.mlp_num_layers,
        mlp_activation=args.mlp_activation,
        mlp_dropout=args.mlp_dropout,
        mlp_norm_type=args.mlp_norm_type,
    ).to(DEVICE)

    # Log1p transform for regression: compress skewed target distributions so
    # that MAE loss treats high-value and low-value samples more equally.
    # Applied in-place BEFORE pos_weight computation (which only runs for
    # classification anyway) so the DataLoaders already see transformed labels.
    if args.log_transform:
        _apply_log1p_to_dataset(train_dataset, args.model_type)
        _apply_log1p_to_dataset(valid_dataset, args.model_type)

    # Compute pos_weight for classification tasks (n_neg / n_pos)
    pos_weight = None
    if args.task_type == "classification":
        labels = torch.tensor(train_dataset.labels, dtype=torch.float32)
        n_pos = labels.sum().item()
        n_neg = len(labels) - n_pos
        if n_pos > 0:
            pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32)

    return model, train_loader, valid_loader, pos_weight




def main():
    raw_args = get_args()
    os.makedirs(raw_args.output_dir, exist_ok=True)

    # Load sampled hyperparameters and merge into args
    with open(raw_args.hp_json, "r") as f:
        hp = json.load(f)
    merged = {**vars(raw_args), **hp}
    args = SimpleNamespace(**merged)

    set_seed(args.seed)

    model, train_loader, valid_loader, pos_weight = build_model_and_loaders(args)

    # Loss
    if args.loss_function == "MAE":
        loss_fn = torch.nn.L1Loss()
    elif args.loss_function == "BCE":
        pw = pos_weight.to(DEVICE) if pos_weight is not None else None
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pw)
    else:
        raise ValueError(f"Unsupported loss function: {args.loss_function}")

    # Optimizer & scheduler （與 optuna_train 保持一致）
    optimizer = torch.optim.AdamW(model.parameters(), args.lr, weight_decay=args.weight_decay)

    scheduler = None
    scheduler_type = getattr(args, "scheduler_type", "cosine")
    if scheduler_type == "cosine":
        expected_epochs = min(args.num_epochs, args.patience * 5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=expected_epochs,
        )
    elif scheduler_type == "plateau":
        mode = "min" if args.metric == "MAE" else "max"
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=0.5,
            patience=5,
            verbose=False,
        )

    metric = Evaluator(name=args.metric)
    minimize_metrics = ["MAE"]
    maximize_metrics = ["Spearman", "ROC-AUC", "PR-AUC"]

    if args.metric in minimize_metrics:
        best_valid_metric = float("inf")
    elif args.metric in maximize_metrics:
        best_valid_metric = -float("inf")
    else:
        best_valid_metric = float("inf")

    best_epoch = None
    train_loss_list, valid_loss_list = [], []
    patience_counter = 0

    best_valid_loss = float("inf")
    loss_no_improve_count = 0
    stopped_by = "max_epoch"

    progress_path = os.path.join(raw_args.output_dir, "training_progress.json")
    summary_path = os.path.join(raw_args.output_dir, "training_summary.json")

    training_start_time = time.time()
    # 只有 seed==1 時，把 tqdm 印到前景，其它 seeds 用普通 range 寫檔即可
    if args.seed == 1:
        desc = f"[Seed {args.seed}] [Trial {args.trial_number}] Epoch"
        epoch_iter = trange(args.num_epochs, desc=desc, position=0)
    else:
        epoch_iter = range(args.num_epochs)
    for epoch in epoch_iter:
        train_loss = train(model, train_loader, loss_fn, optimizer, args.model_type, DEVICE, scheduler=None)

        valid_loss, valid_metric = valid(
            model, valid_loader, loss_fn,
            args.model_type, metric, args.task_type, DEVICE,
            log_transform=args.log_transform,
        )

        if scheduler_type == "cosine" and scheduler is not None:
            scheduler.step()

        train_loss_list.append(train_loss)
        valid_loss_list.append(valid_loss)

        # Plateau scheduler：使用 validation 指標（與 early stopping 同一尺度）
        if scheduler_type == "plateau" and scheduler is not None:
            scheduler.step(valid_metric)

        # 更新 progress（父 process 用來做 epoch-level pruning）
        progress = {
            "epoch": epoch,
            "valid_loss": float(valid_loss),
            "valid_metric": float(valid_metric),
            "timestamp": time.time(),
        }
        with open(progress_path, "w") as f:
            json.dump(progress, f)

        # early stopping 依 metric
        if args.metric in minimize_metrics:
            is_better = valid_metric < best_valid_metric
        elif args.metric in maximize_metrics:
            is_better = valid_metric > best_valid_metric
        else:
            is_better = valid_metric < best_valid_metric

        if is_better:
            best_valid_metric = float(valid_metric)
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(raw_args.output_dir, "best_model.pth"))
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            stopped_by = "metric_patience"
            break

        # val_loss early stopping (OR condition): fires when loss rises > threshold
        # for loss_patience_limit consecutive epochs, regardless of metric state.
        if args.loss_patience_limit > 0:
            if valid_loss < best_valid_loss:
                best_valid_loss = float(valid_loss)
                loss_no_improve_count = 0
            else:
                relative_rise = (valid_loss - best_valid_loss) / (best_valid_loss + 1e-8)
                if relative_rise > args.val_loss_threshold:
                    loss_no_improve_count += 1
                else:
                    loss_no_improve_count = 0

            if loss_no_improve_count >= args.loss_patience_limit:
                stopped_by = "loss_patience"
                break

    training_end_time = time.time()
    training_time = training_end_time - training_start_time

    # 寫 summary，給父 process 匯總
    summary = {
        "best_valid_metric": best_valid_metric,
        "best_epoch": best_epoch,
        "train_time_sec": training_time,
        "stopped_by": stopped_by,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # 原本的 log 與 loss curve（寫在 seed output_dir 底下）
    loss_curve_path = os.path.join(raw_args.output_dir, "loss_curve.png")
    training_log_path = os.path.join(raw_args.output_dir, "training_log.txt")
    plot_loss_curve(train_loss_list, valid_loss_list, loss_curve_path)
    save_training_log(training_log_path, training_time, best_valid_metric, train_loss_list, valid_loss_list)


if __name__ == "__main__":
    main()


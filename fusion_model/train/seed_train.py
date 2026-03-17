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
    GCN_Model, MMB_Model, Desc_Model,
    GCN_MMB_Model, MMB_Desc_Model,
    GCN_Desc_Model, GCN_MMB_Desc_Model,
    MegaMolBART_Finetuned_Model, MPN_MMB_Desc_Model,
    MPN_Model, MPN_Desc_Model, MPN_MMB_Model,
    DMPEGNN, DMPEGNN_Fusion_Model, DMPEGNN_MMB_Desc_Model,
)
from core.train_utils import train, valid
from core.utils import (
    set_seed, save_training_log,
    plot_loss_curve,
)

from chemprop.models import MPN
from chemprop.args import TrainArgs
from nemo_chem.models.megamolbart import MegaMolBARTModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    return parser.parse_args()


def build_model_and_loaders(args: SimpleNamespace):
    """Build model and dataloaders for a single seed, mirroring optuna_train.objective."""
    # === dataset ===
    if args.model_type in ["DMPEGNN", "DMPEGNN_MMB_DESC"]:
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
    if args.model_type in ["DMPEGNN", "DMPEGNN_MMB_DESC"]:
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

    return model, train_loader, valid_loader


def get_model(args, model_type, task_output_dims, gcn_model=None, megamolbart_model=None, gcn_output_dim=None, **mlp_kwargs):
    # 這段與 optuna_train.get_model 完全一致，保持行為相同
    if model_type == "GCN":
        return gcn_model
    if model_type == "MMB":
        return MMB_Model(megamolbart_model, task_output_dims, **mlp_kwargs)
    if model_type == "DESC":
        return Desc_Model(task_output_dims, **mlp_kwargs)
    if model_type == "GCN_MMB":
        return GCN_MMB_Model(gcn_model, megamolbart_model, gcn_output_dim, task_output_dims, **mlp_kwargs)
    if model_type == "MMB_DESC":
        return MMB_Desc_Model(megamolbart_model, task_output_dims, **mlp_kwargs)
    if model_type == "GCN_DESC":
        return GCN_Desc_Model(gcn_model, gcn_output_dim, task_output_dims, **mlp_kwargs)
    if model_type == "GCN_MMB_DESC":
        return GCN_MMB_Desc_Model(gcn_model, megamolbart_model, gcn_output_dim, task_output_dims, **mlp_kwargs)

    if model_type == "MPN":
        mpn_args = TrainArgs()
        mpn_args.hidden_size = args.mpn_hidden_size
        mpn_args.depth = args.mpn_depth
        mpn_args.dropout = args.mpn_dropout
        mpn_args.number_of_molecules = 1
        mpn_args.dataset_type = args.task_type
        mpn_args.aggregation = args.mpn_aggregation
        mpn_args.activation = args.mpn_activation
        mpn_model = MPN(mpn_args).to(DEVICE)
        for p in mpn_model.parameters():
            p.requires_grad = True
        return MPN_Model(mpn_model, task_output_dims, mpn_args.hidden_size, **mlp_kwargs)

    if model_type == "MPN_MMB_DESC":
        mpn_args = TrainArgs()
        mpn_args.hidden_size = args.mpn_hidden_size
        mpn_args.depth = args.mpn_depth
        mpn_args.dropout = args.mpn_dropout
        mpn_args.number_of_molecules = 1
        mpn_args.dataset_type = args.task_type
        mpn_args.aggregation = args.mpn_aggregation
        mpn_args.activation = args.mpn_activation
        mpn_model = MPN(mpn_args).to(DEVICE)
        for p in mpn_model.parameters():
            p.requires_grad = True
        return MPN_MMB_Desc_Model(mpn_model, megamolbart_model, task_output_dims, mpn_args.hidden_size, **mlp_kwargs)

    if model_type == "MPN_DESC":
        mpn_args = TrainArgs()
        mpn_args.hidden_size = args.mpn_hidden_size
        mpn_args.depth = args.mpn_depth
        mpn_args.dropout = args.mpn_dropout
        mpn_args.number_of_molecules = 1
        mpn_args.dataset_type = args.task_type
        mpn_args.aggregation = args.mpn_aggregation
        mpn_args.activation = args.mpn_activation
        mpn_model = MPN(mpn_args).to(DEVICE)
        for p in mpn_model.parameters():
            p.requires_grad = True
        return MPN_Desc_Model(mpn_model, task_output_dims, mpn_args.hidden_size, **mlp_kwargs)

    if model_type == "MPN_MMB":
        mpn_args = TrainArgs()
        mpn_args.hidden_size = args.mpn_hidden_size
        mpn_args.depth = args.mpn_depth
        mpn_args.dropout = args.mpn_dropout
        mpn_args.number_of_molecules = 1
        mpn_args.dataset_type = args.task_type
        mpn_args.aggregation = args.mpn_aggregation
        mpn_args.activation = args.mpn_activation
        mpn_model = MPN(mpn_args).to(DEVICE)
        for p in mpn_model.parameters():
            p.requires_grad = True
        return MPN_MMB_Model(mpn_model, megamolbart_model, task_output_dims, mpn_args.hidden_size, **mlp_kwargs)

    if model_type == "DMPEGNN":
        return DMPEGNN_Fusion_Model(
            node_features=78,
            edge_features=9,
            descriptor_dim=200,
            output_dim=task_output_dims[0],
            hidden_dim=args.dmpegnn_hidden_dim,
            num_layers=args.dmpegnn_num_layers,
            num_heads=args.dmpegnn_num_heads,
            dropout=args.dmpegnn_dropout,
            dmp_steps=args.dmpegnn_dmp_steps,
            pool_type=args.dmpegnn_pool_type,
            use_descriptor=True,
        )

    if model_type == "DMPEGNN_MMB_DESC":
        dmpegnn_backbone = DMPEGNN(
            node_features=78,
            edge_features=9,
            hidden_dim=args.dmpegnn_hidden_dim,
            num_layers=args.dmpegnn_num_layers,
            num_heads=args.dmpegnn_num_heads,
            dropout=args.dmpegnn_dropout,
            output_dim=task_output_dims[0],
            pool_type=args.dmpegnn_pool_type,
            use_equivariant=True,
            use_fingerprint=False,
            use_descriptor=False,
            descriptor_dim=200,
            dmp_steps=args.dmpegnn_dmp_steps,
        ).to(DEVICE)
        return DMPEGNN_MMB_Desc_Model(
            dmpegnn_backbone=dmpegnn_backbone,
            mmb_model=megamolbart_model,
            task_output_dims=task_output_dims,
            dmpegnn_graph_dim=args.dmpegnn_hidden_dim,
            **mlp_kwargs,
        )

    raise ValueError(f"Unknown model type: {model_type}")


def main():
    raw_args = get_args()
    os.makedirs(raw_args.output_dir, exist_ok=True)

    # Load sampled hyperparameters and merge into args
    with open(raw_args.hp_json, "r") as f:
        hp = json.load(f)
    merged = {**vars(raw_args), **hp}
    args = SimpleNamespace(**merged)

    set_seed(args.seed)

    model, train_loader, valid_loader = build_model_and_loaders(args)

    # Loss
    if args.loss_function == "MAE":
        loss_fn = torch.nn.L1Loss()
    elif args.loss_function == "BCE":
        loss_fn = torch.nn.BCEWithLogitsLoss()
    else:
        raise ValueError(f"Unsupported loss function: {args.loss_function}")

    # Optimizer & scheduler （與 optuna_train 保持一致）
    optimizer = torch.optim.AdamW(model.parameters(), args.lr, weight_decay=args.weight_decay)

    scheduler = None
    scheduler_type = getattr(args, "scheduler_type", "cosine")
    if scheduler_type in ("cosine", "step"):
        total_steps = len(train_loader) * args.num_epochs
        if scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=total_steps,
            )
        else:
            step_size = max(1, int(0.7 * total_steps))
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=step_size,
                gamma=0.1,
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

    train_loss_list, valid_loss_list = [], []
    patience_counter = 0

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
        if scheduler_type in ("cosine", "step"):
            train_loss = train(model, train_loader, loss_fn, optimizer, args.model_type, DEVICE, scheduler)
        else:
            train_loss = train(model, train_loader, loss_fn, optimizer, args.model_type, DEVICE, scheduler=None)

        valid_loss, valid_metric = valid(
            model, valid_loader, loss_fn,
            args.model_type, metric, args.task_type, DEVICE,
        )

        train_loss_list.append(train_loss)
        valid_loss_list.append(valid_loss)

        # Plateau scheduler：使用 validation 指標
        if scheduler_type == "plateau" and scheduler is not None:
            if args.metric == "MAE":
                scheduler.step(valid_loss)
            else:
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
            break

    training_end_time = time.time()
    training_time = training_end_time - training_start_time

    # 寫 summary，給父 process 匯總
    summary = {
        "best_valid_metric": best_valid_metric,
        "best_epoch": best_epoch if "best_epoch" in locals() else None,
        "train_time_sec": training_time,
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


import os
import time
import optuna
import optuna.visualization as vis
import torch
import argparse
import numpy as np
from types import SimpleNamespace
from nemo_chem.models.megamolbart import MegaMolBARTModel
import pytorch_lightning as pl
from torch.utils.data import DataLoader as TorchDataLoader
from torch_geometric.loader import DataLoader as PyGDataLoader
import json
import plotly.io as pio
from tdc import Evaluator
from datetime import datetime
# from transformers import get_cosine_schedule_with_warmup
from tqdm import trange

# -------------------------------------------------------
from core.prepare_dataset import load_dataset, load_dataset_cv
from core.dmpegnn_dataset import load_dmpegnn_dataset, load_dmpegnn_dataset_cv, collate_dmpegnn_multi
from core.models import (
    GCN_Model, MMB_Model, Desc_Model, 
    GCN_MMB_Model, MMB_Desc_Model, 
    GCN_Desc_Model, GCN_MMB_Desc_Model,
    MegaMolBART_Finetuned_Model, MPN_MMB_Desc_Model, 
    MPN_Model, MPN_Desc_Model,
    DMPEGNN_Fusion_Model,
    DMPEGNN,
    DMPEGNN_MMB_Desc_Model,
)
from core.train_utils import train, valid
from core.utils import (
    set_seed, save_training_log, 
    plot_loss_curve, format_time
)

# === D-MPNN ===
from chemprop.models import MPN
from chemprop.args import TrainArgs

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

KALEIDO_AVAILABLE = True

# -------------------- Argument Parsing --------------------
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, required=True, choices=['GCN', 'MMB', 'DESC', 'GCN_MMB', 'MMB_DESC', 'GCN_DESC', 'GCN_MMB_DESC', 'MPN_MMB_DESC', 'MPN', 'MPN_DESC', 'DMPEGNN', 'DMPEGNN_MMB_DESC'])
    parser.add_argument('--data_name', type=str, required=True)
    parser.add_argument('--task_type', type=str, required=True, choices=['regression', 'classification'])
    parser.add_argument('--loss_function', type=str, required=True, choices=['MAE', 'BCE'])
    parser.add_argument('--metric', type=str, required=True, choices=['MAE', 'Spearman', 'ROC-AUC', 'PR-AUC'])
    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--pretrained_path', type=str, default='/models/MegaMolBART_0_2_0.nemo')
    parser.add_argument('--data_path', type=str, default='data/data_tdc')
    # parser.add_argument('--seed_list', type=int, nargs='+', default=[1, 2, 3, 4, 5])
    parser.add_argument('--num_tasks', type=int, default=1)
    parser.add_argument('--num_trials', type=int, default=5)
    parser.add_argument('--outer_fold_idx', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    return parser.parse_args()

# -------------------- Model Initialization --------------------
def get_model(args, model_type, task_output_dims, gcn_model=None, megamolbart_model=None, gcn_output_dim=None, **mlp_kwargs):
    if model_type == 'GCN': 
        return gcn_model
    if model_type == 'MMB': 
        return MMB_Model(megamolbart_model, task_output_dims, **mlp_kwargs)
    if model_type == 'DESC': 
        return Desc_Model(task_output_dims, **mlp_kwargs)
    if model_type == 'GCN_MMB': 
        return GCN_MMB_Model(gcn_model, megamolbart_model, gcn_output_dim, task_output_dims, **mlp_kwargs)
    if model_type == 'MMB_DESC': 
        return MMB_Desc_Model(megamolbart_model, task_output_dims, **mlp_kwargs)
    if model_type == 'GCN_DESC': 
        return GCN_Desc_Model(gcn_model, gcn_output_dim, task_output_dims, **mlp_kwargs)
    if model_type == 'GCN_MMB_DESC': 
        return GCN_MMB_Desc_Model(gcn_model, megamolbart_model, gcn_output_dim, task_output_dims, **mlp_kwargs)
    
    if model_type == 'MPN':       
        mpn_args = TrainArgs()
        mpn_args.hidden_size = args.mpn_hidden_size
        mpn_args.depth = args.mpn_depth
        mpn_args.dropout = args.mpn_dropout
        mpn_args.number_of_molecules = 1
        mpn_args.dataset_type = args.task_type
        mpn_args.aggregation = args.mpn_aggregation
        mpn_args.activation = args.mpn_activation
        mpn_model = MPN(mpn_args).to(DEVICE)
        for param in mpn_model.parameters():
            param.requires_grad = True  # unfreeze
        return MPN_Model(mpn_model, task_output_dims, mpn_args.hidden_size, **mlp_kwargs)
    
    if model_type == 'MPN_MMB_DESC':
        mpn_args = TrainArgs()
        mpn_args.hidden_size = args.mpn_hidden_size
        mpn_args.depth = args.mpn_depth
        mpn_args.dropout = args.mpn_dropout
        mpn_args.number_of_molecules = 1
        mpn_args.dataset_type = args.task_type
        mpn_args.aggregation = args.mpn_aggregation
        mpn_args.activation = args.mpn_activation
        mpn_model = MPN(mpn_args).to(DEVICE)
        for param in mpn_model.parameters():
            param.requires_grad = True  # unfreeze
        return MPN_MMB_Desc_Model(mpn_model, megamolbart_model, task_output_dims, mpn_args.hidden_size, **mlp_kwargs)
    
    if model_type == 'MPN_DESC':
        mpn_args = TrainArgs()
        mpn_args.hidden_size = args.mpn_hidden_size
        mpn_args.depth = args.mpn_depth
        mpn_args.dropout = args.mpn_dropout
        mpn_args.number_of_molecules = 1
        mpn_args.dataset_type = args.task_type
        mpn_args.aggregation = args.mpn_aggregation
        mpn_args.activation = args.mpn_activation
        mpn_model = MPN(mpn_args).to(DEVICE)
        for param in mpn_model.parameters():
            param.requires_grad = True  # unfreeze
        return MPN_Desc_Model(mpn_model, task_output_dims, mpn_args.hidden_size, **mlp_kwargs)

    if model_type == 'DMPEGNN':
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

    if model_type == 'DMPEGNN_MMB_DESC':
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

# -------------------- Optuna Hyperparameter Sampling --------------------
def sample_mlp_params(trial):
    return {
        'mlp_hidden_dim': trial.suggest_categorical('mlp_hidden_dim', [16, 32, 64, 128, 256]),
        'mlp_num_layers': trial.suggest_int('mlp_num_layers', 1, 5),
        'mlp_activation': trial.suggest_categorical('mlp_activation', ['relu', 'gelu']),
        'mlp_dropout': trial.suggest_float('mlp_dropout', 0.0, 0.5, step=0.05),
        'mlp_norm_type': trial.suggest_categorical('mlp_norm_type', ['LayerNorm', 'BatchNorm'])
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

def sample_dmpegnn_params(trial):
    return {
        'dmpegnn_hidden_dim': trial.suggest_categorical('dmpegnn_hidden_dim', [128, 256, 384]),
        'dmpegnn_num_layers': trial.suggest_int('dmpegnn_num_layers', 2, 6),
        'dmpegnn_num_heads': trial.suggest_categorical('dmpegnn_num_heads', [4, 8]),
        'dmpegnn_dropout': trial.suggest_float('dmpegnn_dropout', 0.0, 0.3, step=0.05),
        'dmpegnn_dmp_steps': trial.suggest_int('dmpegnn_dmp_steps', 1, 3),
        'dmpegnn_pool_type': trial.suggest_categorical('dmpegnn_pool_type', ['mean', 'sum']),
    }

def sample_optimizer_params(trial):
    return {
        'lr': trial.suggest_float('lr', 1e-5, 1e-2, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
        'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)
    }

def sample_hyperparameters(trial, args):
    sampled = sample_optimizer_params(trial)
    sampled.update(sample_mlp_params(trial))
    if 'GCN' in args.model_type:
        sampled.update(sample_gcn_params(trial))
        sampled['GCN_OUTPUT_DIM'] = 1 if args.model_type == 'GCN' else sampled['gcn_output_dim']
    if 'MPN' in args.model_type:
        sampled.update(sample_mpn_params(trial))
    if 'DMPEGNN' in args.model_type:
        sampled.update(sample_dmpegnn_params(trial))
    return sampled

# -------------------- Objective Function for Optuna --------------------
def objective(trial, args):
    # === hyperparameters ===
    sampled = sample_hyperparameters(trial, args)
    for k, v in sampled.items():
        setattr(args, k, v)
        
    GCN_OUTPUT_DIM = sampled['GCN_OUTPUT_DIM'] if 'GCN_OUTPUT_DIM' in sampled else None
    
    # === directories ===
    SAVE_DIR = os.path.join(ROOT_DIR, "results", "optuna_results_cv", args.model_type.lower(), args.data_name)
    
    LOG_DIR = os.path.join(SAVE_DIR, "log")
    TRIAL_LOG_DIR = os.path.join(LOG_DIR, f"trial_{trial.number}")
    os.makedirs(TRIAL_LOG_DIR, exist_ok=True) 
    
    CHECKPOINT_DIR = os.path.join(SAVE_DIR, "checkpoint")
    TRIAL_CHECKPOINT_DIR = os.path.join(CHECKPOINT_DIR, f"trial_{trial.number}")
    os.makedirs(TRIAL_CHECKPOINT_DIR, exist_ok=True)

    valid_metrics = []
    try:
        for outer_fold_idx in args.outer_fold_idx:
            set_seed(SEED)
            
            # === paths === 
            best_model_path = os.path.join(TRIAL_CHECKPOINT_DIR, f"best_model_fold({outer_fold_idx}).pth")
            loss_curve_path = os.path.join(TRIAL_LOG_DIR, f"loss_curve_fold({outer_fold_idx}).png")
            training_log_path = os.path.join(TRIAL_LOG_DIR, f"training_log_fold({outer_fold_idx}).txt")    
            
            # === train, valid dataset ===
            if args.model_type in ['DMPEGNN', 'DMPEGNN_MMB_DESC']:
                train_dataset, valid_dataset, _ = load_dmpegnn_dataset_cv(
                    data_name=args.data_name,
                    data_path=args.data_path,
                    seed=SEED,
                    outer_fold_idx=outer_fold_idx,
                    inner_fold_idx=(outer_fold_idx + 1) % 4,
                )
            else:
                train_dataset, valid_dataset, _ = load_dataset_cv(
                    data_name=args.data_name,
                    data_path=args.data_path,
                    seed=SEED,
                    outer_fold_idx=outer_fold_idx,
                    inner_fold_idx=(outer_fold_idx + 1) % 4,
                )
            
            drop_last = True if args.mlp_norm_type == "BatchNorm" else False
            if args.model_type in ['DMPEGNN', 'DMPEGNN_MMB_DESC']:
                DataLoaderCls = TorchDataLoader
                collate_fn = collate_dmpegnn_multi
            else:
                DataLoaderCls = PyGDataLoader
                collate_fn = None
            train_loader = DataLoaderCls(train_dataset, args.batch_size, shuffle=True, drop_last=drop_last, collate_fn=collate_fn)
            valid_loader = DataLoaderCls(valid_dataset, args.batch_size, shuffle=False, drop_last=False, collate_fn=collate_fn)

            # === model ===
            gcn_model, megamolbart_model = None, None
            
            # GCN model
            if 'GCN' in args.model_type:
                gcn_model = GCN_Model(input_dim=75,
                                      hidden_dim=args.gcn_hidden_dim,
                                      output_dim=GCN_OUTPUT_DIM,
                                      num_layers=args.gcn_num_layers,
                                      dropout=args.gcn_dropout,
                                      activation=args.gcn_activation,
                                      norm_type=args.gcn_norm_type,
                                      pooling=args.gcn_pooling)
                for param in gcn_model.parameters():
                    param.requires_grad = True  # unfreeze
            
            # MegaMolBART model
            if 'MMB' in args.model_type:
                trainer = pl.Trainer(max_epochs=1, 
                                     accelerator='gpu' if torch.cuda.is_available() else 'cpu',
                                     devices=1, precision=16 if torch.cuda.is_available() else 32,
                                     enable_progress_bar=False)
                pretrained_model = MegaMolBARTModel.restore_from(args.pretrained_path, trainer=trainer)
                for param in pretrained_model.parameters(): 
                    param.requires_grad = False # freeze
                megamolbart_model = MegaMolBART_Finetuned_Model(pretrained_model)

            # combined models
            task_output_dims = [1] * args.num_tasks
            model = get_model(args, args.model_type, 
                              task_output_dims,
                              gcn_model=gcn_model,
                              megamolbart_model=megamolbart_model,
                              gcn_output_dim=GCN_OUTPUT_DIM,
                              mlp_hidden_dim=args.mlp_hidden_dim,
                              mlp_num_layers=args.mlp_num_layers,
                              mlp_activation=args.mlp_activation,
                              mlp_dropout=args.mlp_dropout,
                              mlp_norm_type=args.mlp_norm_type).to(DEVICE)

            # === loss function ===
            if args.loss_function == 'MAE':
                loss_fn = torch.nn.L1Loss()
            elif args.loss_function == 'BCE':
                loss_fn = torch.nn.BCEWithLogitsLoss()
            else:
                raise ValueError(f"Unsupported loss function: {args.loss_function}")

            # === optimizer ===
            optimizer = torch.optim.AdamW(model.parameters(), args.lr, weight_decay=args.weight_decay)
            
            scheduler = None
            # scheduler：warmup + cosine decay
            # total_steps = len(train_loader) * args.num_epochs # total training steps
            # scheduler = get_cosine_schedule_with_warmup(
            #     optimizer,
            #     num_warmup_steps=int(0.1 * total_steps),  # first 10% is for warmup, gradually increasing the learning rate
            #     num_training_steps=total_steps            # The remaining 90% gradually decreases using a cosine schedule
            # )

            # === training ===
            train_loss_list, valid_loss_list = [], []
            patience_counter = 0
                      
            metric = Evaluator(name=args.metric)
            minimize_metrics = ['MAE']
            maximize_metrics = ['Spearman', 'ROC-AUC', 'PR-AUC']
            # initialize best_valid_metric
            if args.metric in minimize_metrics:
                best_valid_metric = float('inf')
            elif args.metric in maximize_metrics:
                best_valid_metric = -float('inf')
            
            training_start_time = time.time()
            for epoch in trange(args.num_epochs, desc=f"[Fold {outer_fold_idx}] Trial {trial.number} Epoch"):
                train_loss = train(model, train_loader, loss_fn, optimizer, args.model_type, DEVICE, scheduler)
                valid_loss, valid_metric = valid(model, valid_loader, loss_fn, args.model_type, metric, args.task_type, DEVICE)
                train_loss_list.append(train_loss)
                valid_loss_list.append(valid_loss)
                
                # pruning
                trial.report(valid_loss, epoch) # report the intermediate value (valid_loss) to the trial
                if trial.should_prune(): # True if the trial should be pruned
                    raise optuna.TrialPruned() # prune the trial
                
                # early stopping (metric)                     
                if args.metric in minimize_metrics:
                    is_better = valid_metric < best_valid_metric
                elif args.metric in maximize_metrics:
                    is_better = valid_metric > best_valid_metric
                    
                if is_better:
                    best_valid_metric = valid_metric
                    # save the best model
                    torch.save(model.state_dict(), best_model_path)
                    patience_counter = 0
                else:
                    patience_counter += 1
                if patience_counter >= args.patience:
                    break
            training_end_time = time.time()
            training_time = training_end_time - training_start_time    
            
            valid_metrics.append(best_valid_metric)
            print(f"[Fold {outer_fold_idx}] Best valid metric: {best_valid_metric:.3f}")

            # === training summary ===
            plot_loss_curve(train_loss_list, valid_loss_list, loss_curve_path)
            save_training_log(training_log_path, training_time, best_valid_metric, train_loss_list, valid_loss_list)

        # trial.set_user_attr: store user-defined attributes in the trial
        valid_metrics = [float(v) for v in valid_metrics]
        trial.set_user_attr("valid_metrics", valid_metrics)
        trial.set_user_attr("trial_dir", TRIAL_CHECKPOINT_DIR) # e.g., optuna_results/GCN/pgp_broccatelli/checkpoint/trial_0
        trial.set_user_attr("outer_fold_list", args.outer_fold_idx) # [0, 1, 2, 3, 4]

        return np.mean(valid_metrics)
    
    except optuna.exceptions.TrialPruned:
        raise

if __name__ == '__main__':
    args = SimpleNamespace(**vars(get_args())) # convert argparse.Namespace to SimpleNamespace
    
    # === directories ===
    SAVE_DIR = os.path.join(ROOT_DIR, "results", "optuna_results_cv", args.model_type.lower(), args.data_name)
    
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

    study_start_time = time.time()
    study = optuna.create_study(
        direction=direction,
        sampler=optuna.samplers.TPESampler(),
        # pruner=optuna.pruners.NopPruner(),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=40),
        study_name = f"opt_{args.model_type.lower()}_{args.data_name}",
        storage=f"sqlite:///{SAVE_DIR}/optuna_study.db",
        load_if_exists=True, # load existing study
    )
    # Resume 時：n_trials 表示「總共」要幾個 trial，只補跑不足的數量
    n_existing = len(study.trials)
    n_rest = max(0, args.num_trials - n_existing)
    if n_rest > 0:
        study.optimize(lambda trial: objective(trial, args), n_trials=n_rest)
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

    # best trial model
    best_trial = study.best_trial
    best_trial_dir = best_trial.user_attrs["trial_dir"]
    best_fold_list = best_trial.user_attrs["outer_fold_list"]
    # copy best model to FINAL_DIR
    FINAL_DIR = os.path.join(SAVE_DIR, "best_trial_models") # e.g., optuna_results/GCN/pgp_broccatelli/best_trial_models
    os.makedirs(FINAL_DIR, exist_ok=True)

    for fold_id in best_fold_list:
        source_path = os.path.join(best_trial_dir, f"best_model_fold({fold_id}).pth") # e.g., optuna_results/GCN/pgp_broccatelli/checkpoint/trial_0/best_model_fold(0).pth
        destination_path = os.path.join(FINAL_DIR, f"best_model_fold({fold_id}).pth") # e.g., optuna_results/GCN/pgp_broccatelli/best_trial_models/best_model_fold(0).pth
        if os.path.exists(source_path):
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
            except Exception:
                print("[WARN] Skip saving Optuna CV figures because Kaleido/Chrome is not available.")
                globals()["KALEIDO_AVAILABLE"] = False
                break

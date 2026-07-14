import os
import torch
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader as TorchDataLoader
from torch_geometric.loader import DataLoader as PyGDataLoader
from nemo_chem.models.megamolbart import MegaMolBARTModel
from tdc import Evaluator
import json
from types import SimpleNamespace

# -------------------------------------------------------
from core.prepare_dataset import load_dataset
from core.dmpegnn_dataset import load_dmpegnn_dataset, collate_dmpegnn_multi
from core.models import (
    GCN_Model, MMB_Model, Desc_Model, 
    GCN_MMB_Model, MMB_Desc_Model, 
    GCN_Desc_Model, GCN_MMB_Desc_Model,
    MegaMolBART_Finetuned_Model, MPN_MMB_Desc_Model, 
    MPN_Model, MPN_Desc_Model, MPN_MMB_Model,
    DMPEGNN_Fusion_Model,
)
from core.train_utils import test
from core.utils import set_seed, save_testing_log
from train.optuna_train import get_args
from core.model_factory import get_model

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    # === parse args ===
    # args = get_args()
    args = SimpleNamespace(**vars(get_args()))
    
    # directories
    SAVE_DIR = os.path.join(ROOT_DIR, "results", "optuna_results", args.model_type.lower(), args.data_name)
    LOG_DIR = os.path.join(SAVE_DIR, "log")
    testing_log_path = os.path.join(LOG_DIR, "testing_log_50.txt")
    
    # === load best params ===          
    best_trial_info_path = os.path.join(LOG_DIR, "best_trial_info.json")
    if os.path.exists(best_trial_info_path):
        with open(best_trial_info_path, "r") as f:
            best_info = json.load(f)
        best_params = best_info["best_params"]
        for key, value in best_params.items():
            setattr(args, key, value)
        best_trial_id = best_info["best_trial_id"]  
    
    if 'GCN' in args.model_type:
        if args.model_type == 'GCN':
            GCN_OUTPUT_DIM = 1
        else:
            GCN_OUTPUT_DIM = args.gcn_output_dim
    else:
        GCN_OUTPUT_DIM = None

    # DMPEGNN defaults when best_params did not set them
    if 'DMPEGNN' in args.model_type:
        if not hasattr(args, 'dmpegnn_hidden_dim'):
            args.dmpegnn_hidden_dim = 256
        if not hasattr(args, 'dmpegnn_num_layers'):
            args.dmpegnn_num_layers = 4
        if not hasattr(args, 'dmpegnn_num_heads'):
            args.dmpegnn_num_heads = 8
        if not hasattr(args, 'dmpegnn_dropout'):
            args.dmpegnn_dropout = 0.1
        if not hasattr(args, 'dmpegnn_dmp_steps'):
            args.dmpegnn_dmp_steps = 2
        if not hasattr(args, 'dmpegnn_pool_type'):
            args.dmpegnn_pool_type = 'mean'

    # AEGNN defaults when best_params did not set them
    if 'AEGNN' in args.model_type:
        if not hasattr(args, 'aegnn_hidden_dim'):
            args.aegnn_hidden_dim = 256
        if not hasattr(args, 'aegnn_num_layers'):
            args.aegnn_num_layers = 6
        if not hasattr(args, 'aegnn_num_heads'):
            args.aegnn_num_heads = 8
        if not hasattr(args, 'aegnn_dropout'):
            args.aegnn_dropout = 0.1
        if not hasattr(args, 'aegnn_pool_type'):
            args.aegnn_pool_type = 'mean'

    # batch_size fallback（當 best_trial_info.json 不存在時）
    if not hasattr(args, 'batch_size'):
        args.batch_size = 32

    # === paths ===
    FINAL_DIR = os.path.join(SAVE_DIR, "best_trial_models_50")

    # 需要 3D 圖資料管線的模型類型（與 seed_train.py、optuna_train.py 保持一致）
    dmpegnn_model_types = {'DMPEGNN', 'DMPEGNN_DESC', 'DMPEGNN_MMB_DESC', 'AEGNN', 'AEGNN_DESC'}

    all_test_scores = []
    for SEED in args.seed_list:
        print(f"\n=== Running with SEED = {SEED} ===")
        set_seed(SEED)

        # === paths ===
        best_model_path = os.path.join(FINAL_DIR, f"best_model_seed({SEED}).pth")

        # === test dataset ===
        if args.model_type in dmpegnn_model_types:
            _, _, test_dataset = load_dmpegnn_dataset(
                data_name=args.data_name,
                data_path=args.data_path,
                seed=SEED,
            )
        else:
            _, _, test_dataset = load_dataset(
                data_name=args.data_name,
                data_path=args.data_path,
                seed=SEED,
            )

        # Apply the same log1p transform used during training so that test
        # batch.y values are in log space; test() will expm1 them back before
        # computing the final metric (ensures MAE is reported in original scale).
        log_transform = getattr(args, 'log_transform', False)
        if log_transform:
            if args.model_type in dmpegnn_model_types:
                test_dataset.labels = [float(np.log1p(l)) for l in test_dataset.labels]
                for g in test_dataset.graphs:
                    g.y = torch.log1p(g.y)
            else:
                test_dataset.labels = torch.log1p(test_dataset.labels)

        if args.model_type in dmpegnn_model_types:
            test_loader = TorchDataLoader(test_dataset, args.batch_size, shuffle=False, collate_fn=collate_dmpegnn_multi)
        else:
            test_loader = PyGDataLoader(test_dataset, args.batch_size, shuffle=False)

        # === test model ===
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
                param.requires_grad = False
            megamolbart_model = MegaMolBART_Finetuned_Model(pretrained_model)

        # combined models
        task_output_dims = [1] * args.num_tasks
        model = get_model(args=args, model_type=args.model_type,
                          task_output_dims=task_output_dims,
                          gcn_model=gcn_model,
                          megamolbart_model=megamolbart_model,
                          gcn_output_dim=GCN_OUTPUT_DIM,
                          mlp_hidden_dim=args.mlp_hidden_dim,
                          mlp_num_layers=args.mlp_num_layers,
                          mlp_activation=args.mlp_activation,
                          mlp_dropout=args.mlp_dropout,
                          mlp_norm_type=args.mlp_norm_type).to(DEVICE)

        # === load checkpoint ===
        model.load_state_dict(torch.load(best_model_path))
        model.eval()

        # === metric ===
        metric = Evaluator(name=args.metric)
        
        # === testing ===
        # log_transform=True: expm1 restores predictions/labels to original
        # scale before metric computation (needed for MAE; Spearman is invariant).
        with torch.no_grad():
            test_score = test(model, test_loader, metric, args.task_type, args.model_type, DEVICE,
                              log_transform=log_transform)
        all_test_scores.append((SEED, test_score))
        # print(f"[Seed {SEED}] Test {args.metric}: {test_score:.3f}")
        print(f"[Seed {SEED}] Test {args.metric}: {test_score[0]:.3f}")


    # === testing summary ===
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_param_count = sum(p.numel() for p in model.parameters())

    # test_scores = [test_score for (_, test_score) in all_test_scores]
    test_scores = [test_score[0] for (_, test_score) in all_test_scores]
    save_testing_log(testing_log_path, test_scores, args.metric, param_count, total_param_count, model, args)

if __name__ == "__main__":
    main()

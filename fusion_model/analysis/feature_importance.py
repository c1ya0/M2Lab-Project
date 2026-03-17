import os
import torch
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
from nemo_chem.models.megamolbart import MegaMolBARTModel
from tdc import Evaluator
import json
from types import SimpleNamespace
import numpy as np
import copy

# ---------
from core.prepare_dataset import load_dataset
from core.models import (
    GCN_Model, MMB_Model, Desc_Model, 
    GCN_MMB_Model, MMB_Desc_Model, 
    GCN_Desc_Model, GCN_MMB_Desc_Model,
    MegaMolBART_Finetuned_Model, MPN_MMB_Desc_Model, MPN_Model
    )
from core.train_utils import test
from core.utils import set_seed, save_testing_log
from train.optuna_train import get_args, get_model

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DESC_DIM = 200
MMB_OUTPUT_DIM = 256

# ------------------------------------------
def perturb_feature_importance(model, test_loader, base_score, args, DEVICE, component: str):
    """
    component: "mmb", "mpn", or "desc"
    """
    
    dim_dict = {
        "mmb": MMB_OUTPUT_DIM,
        "mpn": args.mpn_hidden_size,
        "desc": DESC_DIM
    }

    deltas = []

    for d in range(dim_dict[component]):
        print(f"Perturbing {component} dim {d}")

        perturbed_model = copy.deepcopy(model)  # avoid mutating the original model

        def new_forward(smiles, descriptors, task_index):
            with torch.no_grad():
                # original embeddings
                mpn_emb = model.mpn_model([[s] for s in smiles])
                mmb_emb = model.mmb_model(smiles)
                desc = descriptors.view(mpn_emb.size(0), -1)

                # scramble one dimension (random permutation)
                if component == "mmb":
                    mmb_emb[:, d] = mmb_emb[:, d][torch.randperm(mmb_emb.size(0))]
                elif component == "mpn":
                    mpn_emb[:, d] = mpn_emb[:, d][torch.randperm(mpn_emb.size(0))]
                elif component == "desc":
                    desc[:, d] = desc[:, d][torch.randperm(desc.size(0))]

                combined = torch.cat([mpn_emb, mmb_emb, desc], dim=1).float()
                shared = perturbed_model.shared_mlp(combined)
                return perturbed_model.task_heads[task_index](shared)

        perturbed_model.forward = new_forward
        perturbed_model.eval()

        # evaluate performance after shuffling
        metric = Evaluator(name=args.metric)
        perturbed_score = test(perturbed_model, test_loader, metric, args.task_type, args.model_type, DEVICE)
        # delta =  base_score - perturbed_score
        # delta = (base_score - perturbed_score) / abs(base_score) * 100
        
        if args.metric_direction == "min":  # lower is better, e.g., MAE
            delta = (perturbed_score - base_score) / abs(base_score) * 100
        elif args.metric_direction == "max":  # higher is better, e.g., ROC-AUC
            delta = (base_score - perturbed_score) / abs(base_score) * 100
        else:
            raise ValueError("Unknown metric direction: should be 'min' or 'max'")

        deltas.append(delta)

    return deltas

# ------------------------------------------
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

    # === paths ===
    FINAL_DIR = os.path.join(SAVE_DIR, "best_trial_models_50")

    all_test_scores = []
    all_mmb_deltas = []
    all_mpn_deltas = []
    all_desc_deltas = []
    
    for SEED in args.seed_list:
        print(f"\n=== Running with SEED = {SEED} ===")
        set_seed(SEED)

        # === paths ===
        best_model_path = os.path.join(FINAL_DIR, f"best_model_seed({SEED}).pth")

        # === test dataset ===
        _, _, test_dataset = load_dataset(data_name=args.data_name,
                                          data_path=args.data_path,
                                          seed=SEED)
        test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False)

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

        # combine models
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
        
        # === load checkpoint ===
        model.load_state_dict(torch.load(best_model_path))
        model.eval()

        # === metric ===
        metric = Evaluator(name=args.metric)
        # direction
        args.metric_direction = "min" if args.metric in ["MAE"] else "max"

        
        # === testing ===
        with torch.no_grad():
            test_score = test(model, test_loader, metric, args.task_type, args.model_type, DEVICE)
        all_test_scores.append((SEED, test_score))
        print(f"[Seed {SEED}] Test {args.metric}: {test_score:.3f}")
        
        # ----------------------------------
        # === Feature Importance ===
        print("==== Running feature importance ====")
        mmb_deltas = perturb_feature_importance(model, test_loader, test_score, args, DEVICE, component="mmb")
        mpn_deltas = perturb_feature_importance(model, test_loader, test_score, args, DEVICE, component="mpn")
        desc_deltas = perturb_feature_importance(model, test_loader, test_score, args, DEVICE, component="desc")

        # accumulate deltas from all seeds for averaging at the end
        all_mmb_deltas.append(mmb_deltas)
        all_mpn_deltas.append(mpn_deltas)
        all_desc_deltas.append(desc_deltas)

        
    # === testing summary ===
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_param_count = sum(p.numel() for p in model.parameters())  

    test_scores = [test_score for (_, test_score) in all_test_scores]
    save_testing_log(testing_log_path, test_scores, args.metric, param_count, total_param_count, model, args)

    # === feature importance summary ===
    # average over 5 seeds for each feature dimension
    mmb_mean = np.mean(all_mmb_deltas, axis=0).tolist()
    mpn_mean = np.mean(all_mpn_deltas, axis=0).tolist()
    desc_mean = np.mean(all_desc_deltas, axis=0).tolist()

    avg_result = {
        "mmb_deltas": mmb_mean,
        "mpn_deltas": mpn_mean,
        "desc_deltas": desc_mean,
        "metric": args.metric,
        "metric_direction": args.metric_direction
    }

    fi_result_path = os.path.join(LOG_DIR, "feature_importance_mean.json")
    with open(fi_result_path, "w") as f:
        json.dump(avg_result, f)
    print(f"\n Saved average feature importance to: {fi_result_path}")


if __name__ == "__main__":
    main()

"""
Quick validation script: retrain cyp3a4 best HP with pos_weight and evaluate on test set.
Usage: python -m scripts.validate_posweight
"""
import os
import sys
import json
import subprocess
import numpy as np
import torch
import pytorch_lightning as pl

os.environ["NEMO_LOG_LEVEL"] = "ERROR"

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from types import SimpleNamespace
from torch.utils.data import DataLoader as TorchDataLoader
from tdc import Evaluator

from core.dmpegnn_dataset import load_dmpegnn_dataset, collate_dmpegnn_multi
from core.utils import set_seed
from core.train_utils import test
from nemo_chem.models.megamolbart import MegaMolBARTModel
from core.models import MegaMolBART_Finetuned_Model, DMPEGNN, DMPEGNN_MMB_Desc_Model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Config ===
DATA_NAME    = "cyp2c9_substrate_carbonmangels"
MODEL_TYPE   = "DMPEGNN_MMB_DESC"
TASK_TYPE    = "classification"
LOSS_FN      = "BCE"
METRIC       = "PR-AUC"
NUM_EPOCHS   = 1000
PATIENCE     = 50
SEED_LIST    = [1, 2, 3, 4, 5]
NUM_TASKS    = 1
DATA_PATH    = os.path.join(ROOT_DIR, "data/data_tdc")
PRETRAINED   = "/models/MegaMolBART_0_2_0.nemo"

SAVE_DIR = os.path.join(ROOT_DIR, "results", "optuna_results",
                        MODEL_TYPE.lower(), DATA_NAME)
HP_JSON  = os.path.join(SAVE_DIR, "checkpoint", "trial_49", "hparams.json")
OUT_DIR  = os.path.join(SAVE_DIR, "posweight_validation_trial49")
os.makedirs(OUT_DIR, exist_ok=True)

# ── 1. Retrain each seed ──────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  Retraining with pos_weight  |  {DATA_NAME}")
print(f"{'='*60}\n")

seed_out_dirs = {}
procs = []
for seed in SEED_LIST:
    seed_dir = os.path.join(OUT_DIR, f"seed{seed}")
    os.makedirs(seed_dir, exist_ok=True)
    seed_out_dirs[seed] = seed_dir

    gpu = "0" if seed in [1, 2] else "1"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu

    cmd = [
        sys.executable, "-m", "train.seed_train",
        "--model_type",   MODEL_TYPE,
        "--data_name",    DATA_NAME,
        "--task_type",    TASK_TYPE,
        "--loss_function", LOSS_FN,
        "--metric",       METRIC,
        "--num_epochs",   str(NUM_EPOCHS),
        "--patience",     str(PATIENCE),
        "--data_path",    DATA_PATH,
        "--pretrained_path", PRETRAINED,
        "--num_tasks",    str(NUM_TASKS),
        "--seed",         str(seed),
        "--hp_json",      HP_JSON,
        "--output_dir",   seed_dir,
        "--trial_number", HP_JSON.split("trial_")[1].split("/")[0],
    ]

    log_path = os.path.join(seed_dir, "stdout.log")
    if seed == 1:
        proc = subprocess.Popen(cmd, env=env)
    else:
        proc = subprocess.Popen(cmd, env=env,
                                stdout=open(log_path, "w"),
                                stderr=subprocess.STDOUT)
    procs.append((seed, proc))
    print(f"[Seed {seed}] Training started (GPU {gpu})")

print("\nWaiting for all seeds to finish...\n")
for seed, proc in procs:
    proc.wait()
    rc = proc.returncode
    status = "OK" if rc == 0 else f"FAILED (rc={rc})"
    summary_path = os.path.join(seed_out_dirs[seed], "training_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            s = json.load(f)
        print(f"[Seed {seed}] {status} | best_epoch={s.get('best_epoch')} "
              f"| best_val={s.get('best_valid_metric', 0):.4f}")
    else:
        print(f"[Seed {seed}] {status}")

# ── 2. Evaluate on test set ───────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  Evaluating on test set")
print(f"{'='*60}\n")

with open(HP_JSON) as f:
    hp = json.load(f)
args = SimpleNamespace(**hp)
args.model_type = MODEL_TYPE
args.task_type  = TASK_TYPE
args.pretrained_path = PRETRAINED

metric_fn = Evaluator(name=METRIC)
all_scores = []

for seed in SEED_LIST:
    set_seed(seed)
    _, _, test_dataset = load_dmpegnn_dataset(
        data_name=DATA_NAME, data_path=DATA_PATH, seed=seed)
    test_loader = TorchDataLoader(
        test_dataset, args.batch_size, shuffle=False,
        collate_fn=collate_dmpegnn_multi)

    # Build model
    trainer = pl.Trainer(max_epochs=1,
                         accelerator="gpu" if torch.cuda.is_available() else "cpu",
                         devices=1, precision=16, enable_progress_bar=False)
    pretrained = MegaMolBARTModel.restore_from(PRETRAINED, trainer=trainer)
    for p in pretrained.parameters():
        p.requires_grad = False
    mmb = MegaMolBART_Finetuned_Model(pretrained)

    dmpegnn_backbone = DMPEGNN(
        node_features=82, edge_features=9,
        hidden_dim=args.dmpegnn_hidden_dim,
        num_layers=args.dmpegnn_num_layers,
        num_heads=args.dmpegnn_num_heads,
        dropout=args.dmpegnn_dropout,
        output_dim=1,
        pool_type=args.dmpegnn_pool_type,
        use_equivariant=True, use_fingerprint=False,
        use_descriptor=False, descriptor_dim=200,
        dmp_steps=args.dmpegnn_dmp_steps,
    ).to(DEVICE)

    model = DMPEGNN_MMB_Desc_Model(
        dmpegnn_backbone=dmpegnn_backbone,
        mmb_model=mmb,
        task_output_dims=[1],
        mlp_hidden_dim=args.mlp_hidden_dim,
        mlp_num_layers=args.mlp_num_layers,
        mlp_activation=args.mlp_activation,
        mlp_dropout=args.mlp_dropout,
        mlp_norm_type=args.mlp_norm_type,
    ).to(DEVICE)

    ckpt = os.path.join(seed_out_dirs[seed], "best_model.pth")
    model.load_state_dict(torch.load(ckpt))
    model.eval()

    score, _ = test(model, test_loader, metric_fn, TASK_TYPE, MODEL_TYPE, DEVICE)
    all_scores.append(score)
    print(f"[Seed {seed}] Test {METRIC}: {score:.3f}")

mean_score = float(np.mean(all_scores))
std_score  = float(np.std(all_scores))

print(f"\n{'='*60}")
print(f"  Results WITH pos_weight")
print(f"  Test {METRIC} = {mean_score:.3f} +- {std_score:.3f}")
print(f"  (Previous best: trial_17+pos_weight = 0.397 +- 0.021)")
print(f"{'='*60}\n")

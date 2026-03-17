"""
從已存在的 Optuna study DB 寫出與 optuna_train 正常結束時相同格式的檔案，供 run_optuna_test 等後續腳本讀取。
適用於：run_optuna_train 被 Ctrl+C 中斷後，想用「目前已完成 trials 中的 best」接續跑 run_optuna_test。
輸出：log/best_trial_info.json、log/study_summary_<timestamp>.txt、best_trial_models/*、best_trial_models_50/*
"""
import os
import json
import argparse
import numpy as np
import torch
import optuna
from datetime import datetime

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _to_json_serializable(obj):
    """Convert Optuna/numpy types to native Python for JSON (與 optuna_train 寫出格式一致)."""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_serializable(x) for x in obj]
    return obj


def get_args():
    parser = argparse.ArgumentParser(
        description="Finalize Optuna study: write best_trial_info.json and copy best trial models (e.g. after Ctrl+C)."
    )
    parser.add_argument("--model_type", type=str, required=True)
    parser.add_argument("--data_name", type=str, required=True)
    parser.add_argument("--metric", type=str, required=True, choices=["MAE", "Spearman", "ROC-AUC", "PR-AUC"])
    return parser.parse_args()


def main():
    args = get_args()
    SAVE_DIR = os.path.join(ROOT_DIR, "results", "optuna_results", args.model_type.lower(), args.data_name)
    LOG_DIR = os.path.join(SAVE_DIR, "log")
    storage_path = os.path.join(SAVE_DIR, "optuna_study.db")

    if not os.path.isfile(storage_path):
        print(f"[ERROR] Study DB not found: {storage_path}")
        print("Run optuna_train at least once (or let it run until at least one trial completes) before finalizing.")
        return 1

    direction = "minimize" if args.metric == "MAE" else "maximize"
    study = optuna.create_study(
        direction=direction,
        study_name=f"opt_{args.model_type.lower()}_{args.data_name}",
        storage=f"sqlite:///{storage_path}",
        load_if_exists=True,
    )

    completed = [t for t in study.trials if t.state.name == "COMPLETE"]
    if not completed:
        print("[ERROR] No completed trials in study. Cannot determine best trial.")
        return 1

    best_trial = study.best_trial
    best_trial_id = int(best_trial.number)
    best_params = study.best_params
    # 與 optuna_train 一致：JSON 可序列化（numpy 轉 native），供 optuna_test / feature_importance 等讀取
    best_params_serializable = _to_json_serializable(best_params)
    best_trial_info = {
        "best_trial_id": best_trial_id,
        "best_params": best_params_serializable,
    }
    os.makedirs(LOG_DIR, exist_ok=True)
    best_trial_info_path = os.path.join(LOG_DIR, "best_trial_info.json")
    with open(best_trial_info_path, "w") as f:
        json.dump(best_trial_info, f, indent=4)
    print(f"[OK] Wrote {best_trial_info_path} (best_trial_id={best_trial_id}, best_value={study.best_value:.4f})")

    # === study summary（與 optuna_train 相同格式，study_time 標註為 finalize）===
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    study_path = os.path.join(LOG_DIR, f"study_summary_{timestamp}.txt")
    completed = sum(1 for t in study.trials if t.state.name == "COMPLETE")
    pruned = sum(1 for t in study.trials if t.state.name == "PRUNED")
    with open(study_path, "w") as f:
        f.write("=== Study summary (finalize from existing study) ===\n")
        f.write("Study time: N/A (finalize)\n")
        f.write(f"Total trials: {len(study.trials)}\n")
        f.write(f"Trials completed: {completed}, pruned: {pruned}\n")
        f.write(f"Best trial id: {best_trial_id}\n")
        f.write(f"Best value: {study.best_value:.3f}\n")
        f.write("\nBest params:\n")
        for k, v in best_params_serializable.items():
            f.write(f"{k}: {v}\n")
    print(f"[OK] Wrote {study_path}")

    trial_dir = best_trial.user_attrs.get("trial_dir")
    best_seed_list = best_trial.user_attrs.get("seed_list", [])
    if trial_dir and best_seed_list:
        # seed_train 實際寫入路徑為 trial_N/seed{SEED}/best_model.pth，不是 trial_N/best_model_seed(SEED).pth
        def _src_path(seed):
            p = os.path.join(trial_dir, f"best_model_seed({seed}).pth")
            if os.path.exists(p):
                return p
            p = os.path.join(trial_dir, f"seed{seed}", "best_model.pth")
            return p if os.path.exists(p) else None

        copied = 0
        for final_subdir in ("best_trial_models", "best_trial_models_50"):
            FINAL_DIR = os.path.join(SAVE_DIR, final_subdir)
            os.makedirs(FINAL_DIR, exist_ok=True)
            for seed in best_seed_list:
                src = _src_path(seed)
                if src:
                    dst = os.path.join(FINAL_DIR, f"best_model_seed({seed}).pth")
                    torch.save(torch.load(src), dst)
                    copied += 1
                else:
                    print(f"[WARN] No checkpoint for seed {seed} at {trial_dir} (tried best_model_seed({seed}).pth and seed{seed}/best_model.pth)")
        if copied:
            print(f"[OK] Copied {copied} best trial checkpoints -> best_trial_models & best_trial_models_50")
        else:
            print("[WARN] No checkpoint files found under trial_dir; best_trial_models* are empty.")
    else:
        print("[WARN] No trial_dir or seed_list in best trial user_attrs; skipped copying checkpoint files.")

    return 0


if __name__ == "__main__":
    exit(main())

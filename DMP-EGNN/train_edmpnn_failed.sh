#!/bin/bash

# ============================================================================
# AEGNN-M Training Script (Optimized Version 10 - TDC Multi-Seed Training)
# PARALLEL VERSION - Seeds run concurrently on 2 GPUs
# ============================================================================
# 
# Features:
#   - Automatically loads Optuna-optimized hyperparameters for each dataset from optuna_edmpnn_results/best_trial_info_all.json
#   - Uses train_edmpnn_new.py for training (DMPNN architecture with RobustScaler)
#   - Trains each dataset with 5 seeds (1-5) in PARALLEL on 2 GPUs
#     * GPU 0: seeds 1, 3, 5
#     * GPU 1: seeds 2, 4
#   - Automatically calculates and displays mean ± std results
#   - Supports selecting specific datasets or excluding specific datasets
#
# Usage:
#   1. Train all 22 TDC datasets (no arguments):
#      ./train_edmpnn.sh
#
#   2. Train specific datasets:
#      # Train a single dataset
#      ./train_edmpnn.sh ames
#
#      # Train multiple datasets
#      ./train_edmpnn.sh ames bbb_martins caco2_wang
#
#   3. Exclude specific datasets (--exclude or -x):
#      # Train all datasets, but exclude some
#      ./train_edmpnn.sh --exclude cyp2c9_veith --exclude cyp2d6_veith
#
#      # Train specified datasets, but exclude some
#      ./train_edmpnn.sh ames bbb_martins --exclude caco2_wang
#
#   4. Show help information:
#      ./train_edmpnn.sh --help
#      ./train_edmpnn.sh -h
#
#   5. List all available datasets:
#      ./train_edmpnn.sh --list
#      ./train_edmpnn.sh -l
#
# Supported TDC datasets (22 total):
#   Classification (ROC-AUC): ames, bbb_martins, bioavailability_ma,
#                              cyp3a4_substrate_carbonmangels, dili, herg,
#                              hia_hou, pgp_broccatelli
#   Classification (PR-AUC):   cyp2c9_substrate_carbonmangels, cyp2c9_veith,
#                              cyp2d6_substrate_carbonmangels, cyp2d6_veith,
#                              cyp3a4_veith
#   Regression (MAE):          caco2_wang, ld50_zhu, lipophilicity_astrazeneca,
#                              ppbr_az, solubility_aqsoldb
#   Regression (Spearman):     clearance_hepatocyte_az, clearance_microsome_az,
#                              half_life_obach, vdss_lombardo
#
# Output:
#   - Model checkpoints: checkpoints/{dataset_name}_optuna_final/seed{1-5}/
#   - Training logs: runs/{dataset_name}_optuna_final/seed{1-5}/
#   - Each dataset will display: mean ± std test scores
#
# ============================================================================

set -e

# Activate conda environment
if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
    if conda env list | grep -q "^aegnn_env "; then
        echo "🔧 Activating conda environment: aegnn_env"
        conda activate aegnn_env
    fi
fi

echo "🧪 AEGNN-M Optimized Training V10 (TDC Multi-Seed with Optuna Best Params - PARALLEL)..."
echo "================================================================="

# Set colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

check_tdc_dataset() {
    local dataset_name=$1
    local tdc_data_dir="data/processed_tdc_data/${dataset_name}"
    
    # Debug output
    # echo "    [DEBUG] check_tdc_dataset called with: ${dataset_name}" >&2
    
    # Quick check: directory exists
    if [ ! -d "${tdc_data_dir}" ]; then
        echo -e "${YELLOW}⚠️  TDC dataset directory not found: ${tdc_data_dir}${NC}"
        return 1
    fi
    
    # Check if all required seeds (1-5) have data
    # Use test -f for faster file existence check (doesn't stat file contents)
    local missing_seeds=()
    for seed in 1 2 3 4 5; do
        local seed_dir="${tdc_data_dir}/seed${seed}"
        # Use test -f which is faster than [ -f ]
        if ! test -d "${seed_dir}" || \
           ! test -f "${seed_dir}/train.pt" || \
           ! test -f "${seed_dir}/valid.pt" || \
           ! test -f "${seed_dir}/test.pt"; then
            missing_seeds+=("${seed}")
        fi
    done
    
    if [ ${#missing_seeds[@]} -eq 5 ]; then
        echo -e "${YELLOW}⚠️  No valid seed data found in ${tdc_data_dir}${NC}"
        return 1
    elif [ ${#missing_seeds[@]} -gt 0 ]; then
        echo -e "${YELLOW}⚠️  Missing data for seeds: ${missing_seeds[*]}${NC}"
        echo -e "${YELLOW}   Training will continue with available seeds${NC}"
    fi
    
    echo -e "${GREEN}✅ Found TDC dataset: ${dataset_name}${NC}"
    return 0
}

load_optuna_mod_params() {
    local dataset_name=$1
    local seed_num=$2  # Seed number (1-5)
    local json_path="optuna_edmpnn_results_new/all_best_hyperparameters_mod.json"
    
    if [ ! -f "${json_path}" ]; then
        echo -e "${YELLOW}⚠️  Missing Optuna params file: ${json_path}${NC}"
        return 1
    fi
    
    local param_output
    param_output=$(python3 - <<PY
import json
import sys
from pathlib import Path
path = Path("${json_path}")
all_data = json.loads(path.read_text())
dataset_data = all_data.get("${dataset_name}", {})
if not dataset_data:
    print(f"ERROR: Dataset ${dataset_name} not found in JSON file", file=sys.stderr)
    sys.exit(1)

# Require per-seed format (seeds dict)
if "seeds" not in dataset_data:
    print(f"ERROR: Dataset ${dataset_name} does not have 'seeds' structure. Per-seed format required.", file=sys.stderr)
    sys.exit(1)

seed_key = f"seed${seed_num}"
if seed_key not in dataset_data["seeds"]:
    print(f"ERROR: Seed ${seed_num} not found for dataset ${dataset_name}", file=sys.stderr)
    sys.exit(1)

seed_data = dataset_data["seeds"][seed_key]
best = seed_data.get("best_params", {})

defaults = {
    "hidden_dim": 128,
    "num_layers": 3,
    "dropout": 0.1,
    "lr": 1e-4,
    "learning_rate": 1e-4,  # Alternative key name
    "weight_decay": 1e-4,
    "batch_size": 32,
    "grad_clip_norm": 1.0,
    "loss_type": "bce",
    "label_smoothing": 0.0,
    "use_mixup": False,
    "mixup_alpha": 0.2,
    "num_heads": 8,
    "warmup_epochs": 10,
    "focal_alpha": 0.25,
    "focal_gamma": 2.0,
    "dmp_steps": 2,
    # Scheduler / regularization related (allow Optuna to override shell defaults)
    "scheduler_type": "cosine",
    "min_lr": 1e-6,
    "drop_path_rate": 0.1,
    # Model architecture related
    "activation": "SiLU",
    "aggregation": "mean",
    "pool_type": "mean",
    "ffn_expansion_factor": 4,
    "alpha": 0.2,
    "rotate_aug": False,
    "rotation_prob": 0.5,  # IMPROVEMENT 5.1: Rotation probability (only used if rotate_aug=True)
    "max_rotation_angle": 180.0,  # IMPROVEMENT 5.1: Maximum rotation angle (only used if rotate_aug=True)
    "descriptor_dropout": 0.0
}

def emit(key, value):
    if isinstance(value, bool):
        value = str(value).lower()
    elif value is None:
        value = defaults.get(key, "")
    # Ensure numeric values are properly formatted for shell (use decimal notation for bc compatibility)
    elif isinstance(value, (int, float)):
        # Convert scientific notation to decimal for better shell/bc compatibility
        if 'e' in str(value).lower() or 'E' in str(value):
            # Convert scientific notation to decimal
            value = f"{value:.10f}".rstrip('0').rstrip('.')
        else:
            value = str(value)
    print(f"{key}={value}")

for key in defaults:
    val = best.get(key, defaults[key])
    emit(key, val)

# Handle learning_rate as alternative to lr
if "learning_rate" in best and "lr" not in best:
    emit("lr", best["learning_rate"])
PY
)
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}❌ Failed to parse Optuna params for ${dataset_name}${NC}"
        return 1
    fi
    
    eval "${param_output}"
    
    # Use learning_rate if lr is not set
    if [ -z "${lr}" ] && [ -n "${learning_rate}" ]; then
        lr="${learning_rate}"
    fi
    
    # Map pool_type to aggregation if aggregation is not set
    if [ -z "${aggregation}" ] && [ -n "${pool_type}" ]; then
        aggregation="${pool_type}"
    fi
    
    # ------------------------------------------------------------------
    # Safety check: num_heads must be divisible by hidden_dim / out_channels
    # If Optuna gives invalid combination, automatically adjust num_heads to "largest valid factor <= original value"
    # This can avoid Python side error:
    #   AssertionError: out_channels must be divisible by heads
    # ------------------------------------------------------------------
    if [ -n "${hidden_dim}" ] && [ -n "${num_heads}" ]; then
        # Only check when both are integers
        if [[ "${hidden_dim}" =~ ^[0-9]+$ ]] && [[ "${num_heads}" =~ ^[0-9]+$ ]] && [ "${num_heads}" -gt 0 ]; then
            local mod_val=$((hidden_dim % num_heads))
            if [ "${mod_val}" -ne 0 ]; then
                local original_heads=${num_heads}
                # Find first factor from original num_heads downward that can divide hidden_dim
                local h
                for ((h=original_heads; h>=1; h--)); do
                    if [ $((hidden_dim % h)) -eq 0 ]; then
                        num_heads=${h}
                        break
                    fi
                done
                # Most conservative guarantee: at least 1
                if [ "${num_heads}" -le 0 ]; then
                    num_heads=1
                fi
                echo -e "${YELLOW}⚠️  Adjusted num_heads for ${dataset_name} seed ${seed_num}: ${original_heads} -> ${num_heads} (hidden_dim=${hidden_dim}) to satisfy 'out_channels % heads == 0'${NC}"
            fi
        fi
    fi
    
    # Validate critical parameters
    if [ -z "${lr}" ] || [ -z "${weight_decay}" ]; then
        echo -e "${RED}❌ Critical parameters missing: lr='${lr}', weight_decay='${weight_decay}'${NC}"
        return 1
    fi
    
    return 0
}

get_primary_metric() {
    local dataset_name=$1
    python3 - <<PY
import yaml
import sys
try:
    with open('configs/dataset_primary_metrics.yaml', 'r') as f:
        config = yaml.safe_load(f)
    settings = config.get('dataset_primary_metrics', {}).get('${dataset_name}', {})
    primary_metric = settings.get('primary_metric', 'roc_auc')
    metric_type = settings.get('metric_type', 'classification')
    print(f"{primary_metric},{metric_type}")
except Exception as e:
    # Default fallback
    print("roc_auc,classification")
PY
}

get_task_type_from_config() {
    local dataset_name=$1
    local metric_info=$(get_primary_metric "${dataset_name}")
    echo "${metric_info}" | cut -d',' -f2
}

train_dataset() {
    local dataset_name=$1
    local task_type=$2
    
    # Note: Hyperparameters will be loaded per-seed in train_single_seed function
    # We don't need to load them here anymore
    
    # Get primary metric and task type from config file (similar to optuna_parallel_mod.sh)
    local metric_info=$(get_primary_metric "${dataset_name}")
    local primary_metric=$(echo "${metric_info}" | cut -d',' -f1)
    local config_task_type=$(echo "${metric_info}" | cut -d',' -f2)
    
    # Determine task type: use provided, or from config, or fallback to hardcoded list
    if [ -z "${task_type}" ]; then
        if [ -n "${config_task_type}" ]; then
            # Use config file task type
            if [ "${config_task_type}" = "regression" ]; then
                task_type="regressor"
            elif [ "${config_task_type}" = "classification" ]; then
                task_type="classifier"
            else
                # Unknown task type in config, fall back to hardcoded list
                case "${dataset_name}" in
                    caco2_wang|ld50_zhu|lipophilicity_astrazeneca|ppbr_az|solubility_aqsoldb|clearance_hepatocyte_az|clearance_microsome_az|half_life_obach|vdss_lombardo)
                        task_type="regressor"
                        ;;
                    *)
                        task_type="classifier"
                        ;;
                esac
            fi
        else
            # No config task type, use hardcoded list as fallback
            case "${dataset_name}" in
                caco2_wang|ld50_zhu|lipophilicity_astrazeneca|ppbr_az|solubility_aqsoldb|clearance_hepatocyte_az|clearance_microsome_az|half_life_obach|vdss_lombardo)
                    task_type="regressor"
                    ;;
                *)
                    task_type="classifier"
                    ;;
            esac
        fi
    fi

    # ==============================
    # Dataset-specific safety overrides
    # ==============================
    # Set default early stopping patience (will be overridden per dataset if needed)
    early_stopping_patience=30
    smart_early_stopping_max_patience=50
    # Set default AUROC improvement threshold (will be overridden per dataset if needed)
    auroc_improvement_threshold=0.005
    
    # Note: TDC datasets use Optuna-optimized hyperparameters from all_best_hyperparameters_mod.json
    # Each seed will load its own best hyperparameters in train_single_seed function
    
    echo -e "${BLUE}🚀 Starting V10 Training: ${dataset_name} (Per-Seed Optuna Best Params, 5 Seeds - PARALLEL)${NC}"
    echo -e "   Task Type: ${task_type}"
    echo -e "   Primary Metric: ${GREEN}${primary_metric}${NC}"
    if [ "${task_type}" = "regressor" ] && [ "${primary_metric}" = "spearman" ]; then
        echo -e "   ${YELLOW}⚠️  Note: Using Spearman correlation - ensure train_edmpnn_new.py uses SpearmanLoss${NC}"
    fi
    echo -e "   ${GREEN}⚡ Parallel Mode: Seeds will run concurrently on 2 GPUs${NC}"
    echo -e "   ${GREEN}📋 Each seed will use its own optimized hyperparameters${NC}"
    echo ""
    
    # Function to train a single seed (will be called in parallel)
    train_single_seed() {
        local seed=$1
        local gpu_id=$2
        local result_file=$3
        
        # Check if this seed's data exists before training
        local seed_data_dir="data/processed_tdc_data/${dataset_name}/seed${seed}"
        if [ ! -d "${seed_data_dir}" ] || \
           [ ! -f "${seed_data_dir}/train.pt" ] || \
           [ ! -f "${seed_data_dir}/valid.pt" ] || \
           [ ! -f "${seed_data_dir}/test.pt" ]; then
            echo -e "${YELLOW}⚠️  [GPU ${gpu_id}] Skipping seed ${seed}: data not found${NC}" >&2
            echo "FAILED:${seed}:DATA_NOT_FOUND" > "${result_file}"
            return 1
        fi
        
        # Load seed-specific hyperparameters
        if ! load_optuna_mod_params "${dataset_name}" "${seed}"; then
            echo -e "${YELLOW}⚠️  [GPU ${gpu_id}] Skipping seed ${seed}: failed to load hyperparameters${NC}" >&2
            echo "FAILED:${seed}:HYPERPARAMS_NOT_FOUND" > "${result_file}"
            return 1
        fi
        
        echo -e "${BLUE}----------------------------------------${NC}"
        echo -e "${BLUE}🌱 [GPU ${gpu_id}] Training Seed ${GREEN}${seed}${NC} / 5"
        echo -e "${BLUE}----------------------------------------${NC}"
        echo -e "   Model: Dim ${hidden_dim}, Layers ${num_layers}, Heads ${num_heads}, DMP Steps ${dmp_steps}"
        echo -e "   Config: LR ${lr}, Batch ${batch_size}, WD ${weight_decay}, Dropout ${dropout}"
        
        local save_dir="checkpoints/${dataset_name}_optuna_final/seed${seed}"
        mkdir -p "${save_dir}"
        
        local log_dir="runs/${dataset_name}_optuna_final/seed${seed}"
        mkdir -p "${log_dir}"
    
        local train_cmd=(
            python3 scripts/train_edmpnn_new.py
            --tdc_dataset "${dataset_name}"
            --tdc_seed "${seed}"
            --model_type "${task_type}"
            --use_descriptor
            --descriptor_dim 217
            --hidden_dim "${hidden_dim}"
            --num_layers "${num_layers}"
            --num_heads "${num_heads}"
            --ffn_expansion_factor "${ffn_expansion_factor:-4}"
            --dropout "${dropout}"
            --batch_size "${batch_size}"
            --gradient_accumulation_steps 1  # IMPROVEMENT 7.1: Will be auto-adjusted by train_edmpnn_new.py based on batch_size
            --grad_clip_norm "${grad_clip_norm}"
            --learning_rate "${lr}"
            --weight_decay "${weight_decay}"
            --drop_path_rate "${drop_path_rate}"
            --scheduler_type "${scheduler_type}"
            --scheduler_patience 20
            --warmup_epochs "${warmup_epochs}"
            --min_lr "${min_lr}"
            --num_epochs 200
            --early_stopping_patience "${early_stopping_patience}"
            --use_pre_norm
            --dmp_steps "${dmp_steps}"
            --activation "${activation}"
            --alpha "${alpha:-0.2}"
            --aggregation "${aggregation:-mean}"
            
            # Smart Early Stopping (patience adjusted per dataset)
            --use_smart_early_stopping
            --smart_early_stopping_max_patience "${smart_early_stopping_max_patience}"
            --auroc_improvement_threshold "${auroc_improvement_threshold}"
            
            --log_dir "${log_dir}"
            --save_dir "${save_dir}"
        )
        
        # Add rotate_aug if specified (IMPROVEMENT 5.1: Support rotation augmentation with intensity control)
        if [ "${rotate_aug}" = "true" ] || [ "${rotate_aug}" = "True" ]; then
            train_cmd+=(--rotate_aug)
            # Add rotation_prob if specified (IMPROVEMENT 5.1)
            if [ -n "${rotation_prob}" ] && (( $(echo "${rotation_prob} > 0" | bc -l) )); then
                train_cmd+=(--rotation_prob "${rotation_prob}")
            fi
            # Add max_rotation_angle if specified (IMPROVEMENT 5.1)
            if [ -n "${max_rotation_angle}" ] && (( $(echo "${max_rotation_angle} > 0" | bc -l) )); then
                train_cmd+=(--max_rotation_angle "${max_rotation_angle}")
            fi
        fi
        
        # Add descriptor_dropout if specified
        if [ -n "${descriptor_dropout}" ] && (( $(echo "${descriptor_dropout} > 0" | bc -l) )); then
            train_cmd+=(--descriptor_dropout "${descriptor_dropout}")
        fi

        # Loss Function & Advanced Reg
        case "${loss_type}" in
            focal|Focal)
                train_cmd+=(
                    --use_focal_loss
                    --focal_alpha "${focal_alpha}"
                    --focal_gamma "${focal_gamma}"
                )
                ;;
            class_balanced_focal|ClassBalancedFocal)
                train_cmd+=(
                    --use_class_balanced_focal_loss
                    --focal_alpha "${focal_alpha}"
                    --focal_gamma "${focal_gamma}"
                    --class_balanced_beta "${class_balanced_beta:-0.9999}"
                )
                ;;
            bce|BCE)
                train_cmd+=(
                    --use_bce_for_imbalanced
                    --auto_pos_weight
                )
                ;;
            *)
                echo -e "${YELLOW}⚠️  Unknown loss_type '${loss_type}', using default${NC}" >&2
                ;;
        esac

        if [ "${use_mixup}" = "true" ] || [ "${use_mixup}" = "True" ]; then
            train_cmd+=(
                --enable_manifold_mixup
                --manifold_mixup_alpha "${mixup_alpha:-0.2}"
            )
        fi

        if (( $(echo "${label_smoothing:-0} > 0" | bc -l) )); then
            train_cmd+=(--label_smoothing "${label_smoothing}")
        fi
        
        echo "[GPU ${gpu_id}] Executing: CUDA_VISIBLE_DEVICES=${gpu_id} ${train_cmd[*]}" >&2
        
        # Execute training with GPU assignment
        # train_edmpnn_new.py now supports single GPU mode (non-DDP)
        # Each process will use only the specified GPU
        if env CUDA_VISIBLE_DEVICES=${gpu_id} "${train_cmd[@]}"; then
            echo -e "${GREEN}✅ [GPU ${gpu_id}] Seed ${seed} training completed${NC}" >&2
            
            # Try to extract test score from training history
            local history_file="${save_dir}/training_history.json"
            if [ -f "${history_file}" ]; then
                local test_score=$(python3 - <<PY
import json
import sys
try:
    with open("${history_file}", 'r') as f:
        history = json.load(f)
        # Try to get primary metric from test results
        test_results = history.get("test_results", {})
        if test_results:
            # First, try to get the primary metric for this dataset
            primary_metric = "${primary_metric}"
            
            # Map primary_metric to the actual key in test_results
            metric_map = {
                "roc_auc": "roc_auc",
                "pr_auc": "pr_auc",
                "mae": "mae",
                "spearman": "spearman"
            }
            
            # Try primary metric first
            if primary_metric in metric_map:
                metric_key = metric_map[primary_metric]
                if metric_key in test_results:
                    value = test_results[metric_key]
                    if value is not None:
                        print(value)
                        sys.exit(0)
            
            # Fallback: try common metrics in order
            for metric in ["roc_auc", "pr_auc", "mae", "spearman"]:
                if metric in test_results:
                    value = test_results[metric]
                    if value is not None:
                        print(value)
                        sys.exit(0)
            
            # Fallback to best_val_score if available
            if "best_val_score" in history:
                print(history["best_val_score"])
                sys.exit(0)
except Exception as e:
    pass
print("N/A")
PY
)
                if [ "${test_score}" != "N/A" ] && [ -n "${test_score}" ]; then
                    echo "SUCCESS:${seed}:${test_score}" > "${result_file}"
                    echo -e "   [GPU ${gpu_id}] Test Score: ${test_score}" >&2
                else
                    echo "SUCCESS:${seed}:N/A" > "${result_file}"
                fi
            else
                echo "SUCCESS:${seed}:N/A" > "${result_file}"
            fi
            return 0
        else
            echo -e "${RED}❌ [GPU ${gpu_id}] Seed ${seed} training failed${NC}" >&2
            echo "FAILED:${seed}:TRAINING_ERROR" > "${result_file}"
            return 1
        fi
    }
    
    # Train with 5 seeds in parallel
    # GPU assignment: seed 1,2 -> GPU 0; seed 3,4,5 -> GPU 1
    local seed_results=()
    local seed_scores=()
    local success_count=0
    local fail_count=0
    
    # Create temporary directory for result files
    local tmp_dir=$(mktemp -d)
    trap "rm -rf ${tmp_dir}" EXIT
    
    # Launch parallel training jobs
    local pids=()
    train_single_seed 1 0 "${tmp_dir}/seed1.result" &
    pids+=($!)
    train_single_seed 2 0 "${tmp_dir}/seed2.result" &
    pids+=($!)
    train_single_seed 3 1 "${tmp_dir}/seed3.result" &
    pids+=($!)
    train_single_seed 4 1 "${tmp_dir}/seed4.result" &
    pids+=($!)
    train_single_seed 5 1 "${tmp_dir}/seed5.result" &
    pids+=($!)
    
    # Wait for all background jobs to complete
    echo -e "${BLUE}⏳ Waiting for all 5 seeds to complete...${NC}"
    for pid in "${pids[@]}"; do
        wait "${pid}"
    done
    
    # Collect results
    for seed in 1 2 3 4 5; do
        local result_file="${tmp_dir}/seed${seed}.result"
        if [ -f "${result_file}" ]; then
            local result=$(cat "${result_file}")
            if [[ "${result}" == SUCCESS:* ]]; then
                ((success_count++))
                local score=$(echo "${result}" | cut -d':' -f3)
                if [ "${score}" != "N/A" ] && [ -n "${score}" ]; then
                    seed_scores+=("${score}")
                fi
            else
                ((fail_count++))
            fi
        else
            ((fail_count++))
        fi
    done
    
    # Calculate and display statistics
    echo -e "${BLUE}================================================${NC}"
    echo -e "${BLUE}📊 Training Summary for ${dataset_name}${NC}"
    echo -e "${BLUE}================================================${NC}"
    echo -e "Successful seeds: ${GREEN}${success_count}${NC} / 5"
    echo -e "Failed seeds: ${RED}${fail_count}${NC} / 5"
    
    if [ ${#seed_scores[@]} -gt 0 ]; then
        local mean_std=$(python3 - "${seed_scores[@]}" <<PY
import sys
scores = [float(s) for s in sys.argv[1:]]
if len(scores) > 0:
    mean = sum(scores) / len(scores)
    if len(scores) > 1:
        variance = sum((x - mean) ** 2 for x in scores) / len(scores)
        std = variance ** 0.5
    else:
        std = 0.0
    print(f"{mean:.4f} ± {std:.4f}")
else:
    print("N/A")
PY
)
        echo -e "Test Score (Mean ± Std): ${GREEN}${mean_std}${NC}"
        echo -e "Individual scores: ${seed_scores[*]}"
    fi
    echo ""
    
    return 0
}

# Parse command line arguments
# TDC datasets from processed_tdc_data directory
ALL_DATASETS="ames bbb_martins bioavailability_ma caco2_wang clearance_hepatocyte_az clearance_microsome_az cyp2c9_substrate_carbonmangels cyp2c9_veith cyp2d6_substrate_carbonmangels cyp2d6_veith cyp3a4_substrate_carbonmangels cyp3a4_veith dili half_life_obach herg hia_hou ld50_zhu lipophilicity_astrazeneca pgp_broccatelli ppbr_az solubility_aqsoldb vdss_lombardo"
SELECTED_DATASETS=""
USER_DATASETS=()
EXCLUDE_DATASETS=()

# Function to display help message
show_help() {
    echo "Usage: $0 [OPTIONS] [dataset1] [dataset2] ..."
    echo ""
    echo "AEGNN-M Training Script V10 - TDC Multi-Seed Training"
    echo "Trains each dataset with 5 seeds (1-5) using Optuna-optimized hyperparameters"
    echo ""
    echo "Options:"
    echo "  -h, --help              Show this help message"
    echo "  -l, --list             List all available TDC datasets"
    echo "  -x, --exclude DATASET  Exclude a dataset from training (can be used multiple times)"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Train all 22 TDC datasets"
    echo "  $0 ames bbb_martins caco2_wang       # Train specific datasets"
    echo "  $0 --exclude cyp2c9_veith --exclude cyp2d6_veith  # Train all except excluded datasets"
    echo "  $0 ames bbb_martins --exclude caco2_wang  # Train ames and bbb_martins, exclude caco2_wang"
    echo ""
    echo "Available TDC datasets (22 total):"
    echo "  Classification (ROC-AUC): ames, bbb_martins, bioavailability_ma,"
    echo "                            cyp3a4_substrate_carbonmangels, dili, herg,"
    echo "                            hia_hou, pgp_broccatelli"
    echo "  Classification (PR-AUC): cyp2c9_substrate_carbonmangels, cyp2c9_veith,"
    echo "                            cyp2d6_substrate_carbonmangels, cyp2d6_veith,"
    echo "                            cyp3a4_veith"
    echo "  Regression (MAE):        caco2_wang, ld50_zhu, lipophilicity_astrazeneca,"
    echo "                            ppbr_az, solubility_aqsoldb"
    echo "  Regression (Spearman):    clearance_hepatocyte_az, clearance_microsome_az,"
    echo "                            half_life_obach, vdss_lombardo"
    exit 0
}

# Function to list all available datasets
list_datasets() {
    echo "Available TDC datasets (22 total):"
    echo ""
    echo "Classification Datasets (13):"
    echo "  ROC-AUC (8):"
    for ds in ames bbb_martins bioavailability_ma cyp3a4_substrate_carbonmangels dili herg hia_hou pgp_broccatelli; do
        echo "    • $ds"
    done
    echo "  PR-AUC (5):"
    for ds in cyp2c9_substrate_carbonmangels cyp2c9_veith cyp2d6_substrate_carbonmangels cyp2d6_veith cyp3a4_veith; do
        echo "    • $ds"
    done
    echo ""
    echo "Regression Datasets (9):"
    echo "  MAE (5):"
    for ds in caco2_wang ld50_zhu lipophilicity_astrazeneca ppbr_az solubility_aqsoldb; do
        echo "    • $ds"
    done
    echo "  Spearman (4):"
    for ds in clearance_hepatocyte_az clearance_microsome_az half_life_obach vdss_lombardo; do
        echo "    • $ds"
    done
    exit 0
}

# Parse command line arguments
while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help)
            show_help
            ;;
        -l|--list)
            list_datasets
            ;;
        --exclude|-x)
            shift
            if [ $# -eq 0 ]; then
                echo -e "${RED}❌ Missing dataset name after --exclude/-x${NC}"
                echo "Use --help for usage information"
                exit 1
            fi
            EXCLUDE_DATASETS+=("$1")
            ;;
        *)
            USER_DATASETS+=("$1")
            ;;
    esac
    shift
done

# Check if any datasets provided
if [ ${#USER_DATASETS[@]} -eq 0 ]; then
    # No positional datasets: train all
    SELECTED_DATASETS="${ALL_DATASETS}"
    echo -e "${BLUE}📋 No datasets specified, will train all available datasets${NC}"
else
    # Validate provided datasets
    for arg in "${USER_DATASETS[@]}"; do
        if echo "${ALL_DATASETS}" | grep -qw "${arg}"; then
            if [ -z "${SELECTED_DATASETS}" ]; then
                SELECTED_DATASETS="${arg}"
            else
                SELECTED_DATASETS="${SELECTED_DATASETS} ${arg}"
            fi
        else
            echo -e "${YELLOW}⚠️  Unknown dataset: ${arg}${NC}"
            echo -e "${YELLOW}   Available datasets: ${ALL_DATASETS}${NC}"
        fi
    done
    
    if [ -z "${SELECTED_DATASETS}" ]; then
        echo -e "${RED}❌ No valid datasets specified${NC}"
        echo ""
        echo -e "${BLUE}Usage: $0 [OPTIONS] [dataset1] [dataset2] ...${NC}"
        echo -e "${BLUE}Use '$0 --help' for detailed usage information${NC}"
        echo -e "${BLUE}Use '$0 --list' to see all available datasets${NC}"
        echo ""
        echo -e "${YELLOW}Available datasets:${NC}"
        echo "${ALL_DATASETS}" | tr ' ' '\n' | nl
        exit 1
    fi
    
    echo -e "${BLUE}📋 Will train selected datasets:${SELECTED_DATASETS}${NC}"
fi

# Apply exclusions if provided
if [ ${#EXCLUDE_DATASETS[@]} -gt 0 ]; then
    echo -e "${BLUE}🧹 Excluding datasets: ${EXCLUDE_DATASETS[*]}${NC}"
    FILTERED=""
    for ds in ${SELECTED_DATASETS}; do
        skip=false
        for ex in "${EXCLUDE_DATASETS[@]}"; do
            if [ "${ds}" = "${ex}" ]; then
                skip=true
                break
            fi
        done
        if [ "${skip}" = false ]; then
            FILTERED="${FILTERED} ${ds}"
        fi
    done
    SELECTED_DATASETS="${FILTERED# }"
fi

echo -e "${BLUE}📋 Checking available TDC datasets...${NC}"

# Trim leading/trailing whitespace from SELECTED_DATASETS
SELECTED_DATASETS=$(echo "${SELECTED_DATASETS}" | xargs)

TRAINED=0
TOTAL_DATASETS=0
SUCCESSFUL_DATASETS=0
FAILED_DATASETS=()

# Convert space-separated string to array for safer iteration
# Save original IFS and restore it after
OLD_IFS="${IFS}"
IFS=' ' read -ra DATASET_ARRAY <<< "${SELECTED_DATASETS}"
IFS="${OLD_IFS}"

if [ ${#DATASET_ARRAY[@]} -eq 0 ]; then
    echo -e "${RED}❌ No datasets to process! SELECTED_DATASETS='${SELECTED_DATASETS}'${NC}"
    exit 1
fi

# echo -e "${BLUE}[DEBUG] Starting loop with ${#DATASET_ARRAY[@]} datasets${NC}"
for dataset_name in "${DATASET_ARRAY[@]}"; do
    # echo -e "${BLUE}[DEBUG] Loop iteration, dataset_name=[${dataset_name}]${NC}"
    TOTAL_DATASETS=$((TOTAL_DATASETS + 1))
    echo -e "${BLUE}  [${TOTAL_DATASETS}] Checking: ${dataset_name}...${NC}"
    if check_tdc_dataset "${dataset_name}"; then
        echo ""
        if train_dataset "${dataset_name}"; then
            SUCCESSFUL_DATASETS=$((SUCCESSFUL_DATASETS + 1))
            TRAINED=$((TRAINED + 1))
        else
            FAILED_DATASETS+=("${dataset_name}")
        fi
    else
        FAILED_DATASETS+=("${dataset_name}")
    fi
done

echo "=================================="
echo -e "${BLUE}📊 Final Summary${NC}"
echo "=================================="
echo -e "Total datasets processed: ${TOTAL_DATASETS}"
echo -e "${GREEN}✅ Successful: ${SUCCESSFUL_DATASETS}${NC}"
if [ ${#FAILED_DATASETS[@]} -gt 0 ]; then
    echo -e "${RED}❌ Failed: ${#FAILED_DATASETS[@]}${NC}"
    echo -e "Failed datasets: ${FAILED_DATASETS[*]}"
fi

if [ $TRAINED -eq 0 ]; then
    echo -e "${RED}❌ No datasets were successfully trained${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}🎉 All V10 (TDC Multi-Seed with Optuna Best Params) training tasks completed!${NC}"





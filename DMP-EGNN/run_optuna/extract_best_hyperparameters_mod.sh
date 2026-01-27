#!/bin/bash

# =============================================================================
# Wrapper script for extract_best_hyperparameters_mod.py
#
# 功能：
#   - 從 optuna_mod 的資料庫中，為多個 TDC dataset 抽取最佳超參數
#   - 會同時在：
#       1) 使用者指定的 --output-dir（若有）
#       2) checkpoints/optuna_mod/<dataset>/best_trial_models/best_trial_info.json
#     寫入結果（後者是 Python 腳本已經內建的行為）
#
# 使用方式：
#   1) 指定特定 datasets：
#        ./run_optuna/extract_best_hyperparameters_mod.sh ames herg caco2_wang
#
#   2) 全部 22 個 TDC datasets：
#        ./run_optuna/extract_best_hyperparameters_mod.sh --all
#      或（沒給任何 dataset 參數時，也會視為 --all）
#        ./run_optuna/extract_best_hyperparameters_mod.sh
#
#   3) all + exclude：
#        ./run_optuna/extract_best_hyperparameters_mod.sh --all \
#            --exclude cyp2c9_veith --exclude cyp2d6_veith
#
#   4) 搭配 storage / output-dir：
#        ./run_optuna/extract_best_hyperparameters_mod.sh --all \
#            --storage sqlite:///optuna_edmpnn_results/optuna_mod.db \
#            --output-dir results/best_params
#
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

# ---------------------------
# 列出所有支援的 TDC datasets
# ---------------------------
ALL_DATASETS="ames bbb_martins bioavailability_ma caco2_wang clearance_hepatocyte_az clearance_microsome_az cyp2c9_substrate_carbonmangels cyp2c9_veith cyp2d6_substrate_carbonmangels cyp2d6_veith cyp3a4_substrate_carbonmangels cyp3a4_veith dili half_life_obach herg hia_hou ld50_zhu lipophilicity_astrazeneca pgp_broccatelli ppbr_az solubility_aqsoldb vdss_lombardo"

SELECTED_DATASETS=()
EXCLUDE_DATASETS=()

STORAGE_PATH="sqlite:///optuna_edmpnn_results/optuna_mod.db"
# 預設輸出到 optuna_edmpnn_results，使得最終檔名為：
#   optuna_edmpnn_results/all_best_hyperparameters_mod.json
OUTPUT_DIR="optuna_edmpnn_results"

USE_ALL=false

show_help() {
    echo "Usage: $0 [OPTIONS] [dataset1] [dataset2] ..."
    echo ""
    echo "Wrapper for scripts/extract_best_hyperparameters_mod.py"
    echo ""
    echo "Dataset selection (二選一)："
    echo "  1) 顯式列出 datasets："
    echo "       $0 ames herg caco2_wang"
    echo "  2) 使用 --all，並可搭配 --exclude："
    echo "       $0 --all --exclude cyp2c9_veith --exclude cyp2d6_veith"
    echo ""
    echo "Options:"
    echo "  --all                 使用所有 TDC datasets（若未指定任何 dataset 參數，也會視為 --all）"
    echo "  -x, --exclude  NAME   從 --all / 選擇結果中排除指定 dataset，可重複使用多次"
    echo "  --storage PATH        Optuna storage 路徑（預設：${STORAGE_PATH})"
    echo "  --output-dir DIR      額外輸出 JSON 的目錄（可選）"
    echo "  -h, --help            顯示此說明"
    echo ""
    echo "支援的 TDC datasets："
    echo "  ${ALL_DATASETS}"
}

# ---------------------------
# 解析參數
# ---------------------------
while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help)
            show_help
            exit 0
            ;;
        --all)
            USE_ALL=true
            ;;
        -x|--exclude)
            shift
            if [ $# -eq 0 ]; then
                echo "ERROR: --exclude 之後需要 dataset 名稱" >&2
                exit 1
            fi
            EXCLUDE_DATASETS+=("$1")
            ;;
        --storage)
            shift
            if [ $# -eq 0 ]; then
                echo "ERROR: --storage 之後需要路徑" >&2
                exit 1
            fi
            STORAGE_PATH="$1"
            ;;
        --output-dir)
            shift
            if [ $# -eq 0 ]; then
                echo "ERROR: --output-dir 之後需要目錄路徑" >&2
                exit 1
            fi
            OUTPUT_DIR="$1"
            ;;
        -*)
            echo "ERROR: 未知選項：$1" >&2
            echo "使用 --help 取得說明。" >&2
            exit 1
            ;;
        *)
            # 非選項，視為使用者指定的 dataset
            SELECTED_DATASETS+=("$1")
            ;;
    esac
    shift
done

# 若沒有指定 dataset，預設等同於 --all
if [ ${#SELECTED_DATASETS[@]} -eq 0 ] && [ "${USE_ALL}" = false ]; then
    USE_ALL=true
fi

# 組合初始 dataset 列表
if [ "${USE_ALL}" = true ]; then
    # 用 ALL_DATASETS
    IFS=' ' read -r -a SELECTED_DATASETS <<< "${ALL_DATASETS}"
fi

if [ ${#SELECTED_DATASETS[@]} -eq 0 ]; then
    echo "ERROR: 沒有任何 dataset 可供處理。" >&2
    echo "       請使用 --all 或指定至少一個 dataset，或加上 --help 查看說明。" >&2
    exit 1
fi

# 套用 exclude
if [ ${#EXCLUDE_DATASETS[@]} -gt 0 ]; then
    FILTERED=()
    for ds in "${SELECTED_DATASETS[@]}"; do
        SKIP=false
        for ex in "${EXCLUDE_DATASETS[@]}"; do
            if [ "${ds}" = "${ex}" ]; then
                SKIP=true
                break
            fi
        done
        if [ "${SKIP}" = false ]; then
            FILTERED+=("${ds}")
        fi
    done
    SELECTED_DATASETS=("${FILTERED[@]}")
fi

if [ ${#SELECTED_DATASETS[@]} -eq 0 ]; then
    echo "ERROR: 所有 dataset 都被排除，沒有東西可以抽取。" >&2
    exit 1
fi

echo "🔍 將為以下 dataset 抽取最佳超參數："
printf '  - %s\n' "${SELECTED_DATASETS[@]}"
echo ""

# ---------------------------
# 組合並執行 Python 指令
# ---------------------------
PY_CMD=(python3 scripts/extract_best_hyperparameters_mod.py)
PY_CMD+=("--dataset")
PY_CMD+=("${SELECTED_DATASETS[@]}")
PY_CMD+=("--storage" "${STORAGE_PATH}")

if [ -n "${OUTPUT_DIR}" ]; then
    PY_CMD+=("--output-dir" "${OUTPUT_DIR}")
    PY_CMD+=("--combined-only")
fi

echo "執行指令："
echo "  ${PY_CMD[*]}"
echo ""

"${PY_CMD[@]}"

echo ""
echo "✅ 抽取完成。合併結果已寫入："
echo "   ${OUTPUT_DIR}/all_best_hyperparameters_mod.json"




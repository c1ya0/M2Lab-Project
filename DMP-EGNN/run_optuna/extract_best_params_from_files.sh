#!/bin/bash

# =============================================================================
# Wrapper script for extract_best_params_from_files.py
#
# Functions:
#   - Extract best hyperparameters from existing best_params.json files
#   - Read checkpoints/optuna_mod/<dataset>/seed<seed>/best_trial_models/best_params.json
#   - Merge into a single JSON file
#
# Usage:
#   1) Specify specific datasets:
#        ./run_optuna/extract_best_params_from_files.sh ames herg caco2_wang
#
#   2) All datasets:
#        ./run_optuna/extract_best_params_from_files.sh --all
#      or (when no dataset parameter is given, also treated as --all)
#        ./run_optuna/extract_best_params_from_files.sh
#
#   3) all + exclude:
#        ./run_optuna/extract_best_params_from_files.sh --all \
#            --exclude cyp2c9_veith --exclude cyp2d6_veith
#
#   4) With output-dir:
#        ./run_optuna/extract_best_params_from_files.sh --all \
#            --output-dir results/best_params
#
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

# ---------------------------
# List all supported TDC datasets
# ---------------------------
ALL_DATASETS="ames bbb_martins bioavailability_ma caco2_wang clearance_hepatocyte_az clearance_microsome_az cyp2c9_substrate_carbonmangels cyp2c9_veith cyp2d6_substrate_carbonmangels cyp2d6_veith cyp3a4_substrate_carbonmangels cyp3a4_veith dili half_life_obach herg hia_hou ld50_zhu lipophilicity_astrazeneca pgp_broccatelli ppbr_az solubility_aqsoldb vdss_lombardo"

SELECTED_DATASETS=()
EXCLUDE_DATASETS=()

# Default output to optuna_edmpnn_results, so final filename is:
#   optuna_edmpnn_results/all_best_hyperparameters_mod.json
OUTPUT_DIR="optuna_edmpnn_results"

USE_ALL=false

show_help() {
    echo "Usage: $0 [OPTIONS] [dataset1] [dataset2] ..."
    echo ""
    echo "Wrapper for scripts/extract_best_params_from_files.py"
    echo ""
    echo "Dataset selection (choose one):"
    echo "  1) Explicitly list datasets:"
    echo "       $0 ames herg caco2_wang"
    echo "  2) Use --all, can be combined with --exclude:"
    echo "       $0 --all --exclude cyp2c9_veith --exclude cyp2d6_veith"
    echo ""
    echo "Options:"
    echo "  --all                 Use all TDC datasets (if no dataset parameter specified, also treated as --all)"
    echo "  -x, --exclude  NAME   Exclude specified dataset from --all / selection result, can be used multiple times"
    echo "  --output-dir DIR      Output JSON directory (default: ${OUTPUT_DIR})"
    echo "  --individual-files    Also output individual JSON file for each dataset (default: only output combined file)"
    echo "  -h, --help            Show this help message"
    echo ""
    echo "Supported TDC datasets:"
    echo "  ${ALL_DATASETS}"
}

# ---------------------------
# Parse parameters
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
                echo "ERROR: --exclude requires dataset name" >&2
                exit 1
            fi
            EXCLUDE_DATASETS+=("$1")
            ;;
        --output-dir)
            shift
            if [ $# -eq 0 ]; then
                echo "ERROR: --output-dir requires directory path" >&2
                exit 1
            fi
            OUTPUT_DIR="$1"
            ;;
        --base-dir)
            shift
            if [ $# -eq 0 ]; then
                echo "ERROR: --base-dir requires directory path" >&2
                exit 1
            fi
            BASE_DIR_OVERRIDE="$1"
            ;;
        --individual-files)
            INDIVIDUAL_FILES_FLAG="--individual-files"
            ;;
        -*)
            echo "ERROR: Unknown option: $1" >&2
            echo "Use --help for help." >&2
            exit 1
            ;;
        *)
            # Non-option, treat as user-specified dataset
            SELECTED_DATASETS+=("$1")
            ;;
    esac
    shift
done

# If no dataset specified, default to --all
if [ ${#SELECTED_DATASETS[@]} -eq 0 ] && [ "${USE_ALL}" = false ]; then
    USE_ALL=true
fi

# Combine initial dataset list
if [ "${USE_ALL}" = true ]; then
    # Use ALL_DATASETS
    IFS=' ' read -r -a SELECTED_DATASETS <<< "${ALL_DATASETS}"
fi

if [ ${#SELECTED_DATASETS[@]} -eq 0 ]; then
    echo "ERROR: No dataset available for processing." >&2
    echo "       Please use --all or specify at least one dataset, or use --help for help." >&2
    exit 1
fi

# Apply exclude
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
    echo "ERROR: All datasets are excluded, nothing to extract." >&2
    exit 1
fi

echo "🔍 Will extract best hyperparameters from best_params.json files for the following datasets:"
printf '  - %s\n' "${SELECTED_DATASETS[@]}"
echo ""

# ---------------------------
# Combine and execute Python command
# ---------------------------
PY_CMD=(python3 scripts/extract_best_params_from_files.py)
PY_CMD+=("--dataset")
PY_CMD+=("${SELECTED_DATASETS[@]}")
# Support both optuna_mod (per-seed) and optuna_mod_new (Fusion Model Logic)
# Default to optuna_mod, but can be overridden with --base-dir
if [ -z "${BASE_DIR_OVERRIDE}" ]; then
    # Auto-detect: prefer optuna_mod_new if it exists and has data
    if [ -d "checkpoints/optuna_mod_new" ] && [ "$(ls -A checkpoints/optuna_mod_new 2>/dev/null)" ]; then
        BASE_DIR="checkpoints/optuna_mod_new"
        echo "📁 Using optuna_mod_new (Fusion Model Logic structure)"
    else
        BASE_DIR="checkpoints/optuna_mod"
        echo "📁 Using optuna_mod (per-seed structure)"
    fi
else
    BASE_DIR="${BASE_DIR_OVERRIDE}"
fi

PY_CMD+=("--base-dir" "${BASE_DIR}")

if [ -n "${OUTPUT_DIR}" ]; then
    PY_CMD+=("--output-dir" "${OUTPUT_DIR}")
fi

if [ -n "${INDIVIDUAL_FILES_FLAG}" ]; then
    PY_CMD+=("${INDIVIDUAL_FILES_FLAG}")
fi

echo "Executing command:"
echo "  ${PY_CMD[*]}"
echo ""

"${PY_CMD[@]}"

if [ -n "${OUTPUT_DIR}" ]; then
    echo ""
    echo "✅ Extraction completed. Combined results written to:"
    echo "   ${OUTPUT_DIR}/all_best_hyperparameters_mod.json"
fi


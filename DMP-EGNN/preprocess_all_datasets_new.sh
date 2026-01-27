#!/usr/bin/env bash

# AEGNN-M Batch preprocessing script for TDC datasets (NEW)
#
# Differences vs preprocess_all_datasets.sh:
# - Uses scripts/preprocess_tdc_data_new.py
# - Writes outputs into data/processed_tdc_data_new/
# - Drops molecules whose descriptor contains NaN/Inf (fusion_model-style), via --nonfinite_policy drop

set -uo pipefail

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

TDC_DATA_DIR="data/data_tdc"
PROCESSED_DIR_NEW="data/processed_tdc_data_new"

# Parameters (keep aligned with original defaults)
SEED=42
SEED_LIST=(1 2 3 4 5)
NUM_CONFORMERS=10
OPTIMIZE_CONFORMERS=true
ADD_HYDROGENS=true
USE_FINGERPRINT=false
FINGERPRINT_BITS=2048
DESCRIPTOR_DIM=""  # empty = use all available
NUM_WORKERS=3

# CV mode (stored under data/processed_tdc_data_new/cv/)
USE_CV=false
OUTER_FOLDS=5
INNER_FOLDS=4

list_tdc_datasets() {
    local datasets=()
    local admet_dir="$TDC_DATA_DIR/admet_group"
    if [ -d "$admet_dir" ]; then
        while IFS= read -r dir; do
            if [ -d "$dir" ]; then
                local dataset_name
                dataset_name=$(basename "$dir")
                if [ -f "$dir/test.csv" ] && [ -f "$dir/train_val.csv" ]; then
                    datasets+=("$dataset_name")
                fi
            fi
        done < <(find "$admet_dir" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort)
    fi
    printf '%s\n' "${datasets[@]}"
}

validate_tdc_dataset() {
    local dataset_name="$1"
    local dataset_dir="$TDC_DATA_DIR/admet_group/$dataset_name"
    if [ -d "$dataset_dir" ] && [ -f "$dataset_dir/test.csv" ] && [ -f "$dataset_dir/train_val.csv" ]; then
        return 0
    fi
    return 1
}

is_seed_processed_new() {
    local dataset_name="$1"
    local seed="$2"
    local cache_dir="$PROCESSED_DIR_NEW/$dataset_name/seed$seed"
    local train_file="$cache_dir/train.pt"
    local valid_file="$cache_dir/valid.pt"
    local test_file="$cache_dir/test.pt"
    if [ ! -f "$train_file" ] || [ ! -f "$valid_file" ] || [ ! -f "$test_file" ]; then
        return 1
    fi
    local train_size
    local valid_size
    local test_size
    train_size=$(stat -c%s "$train_file" 2>/dev/null || echo "0")
    valid_size=$(stat -c%s "$valid_file" 2>/dev/null || echo "0")
    test_size=$(stat -c%s "$test_file" 2>/dev/null || echo "0")
    if [ "$train_size" -lt 1024 ] || [ "$valid_size" -lt 1024 ] || [ "$test_size" -lt 1024 ]; then
        return 1
    fi
    return 0
}

is_tdc_processed_new() {
    local dataset_name="$1"
    for seed in "${SEED_LIST[@]}"; do
        if ! is_seed_processed_new "$dataset_name" "$seed"; then
            return 1
        fi
    done
    return 0
}

is_tdc_cv_processed_new() {
    local dataset_name="$1"
    local outer_fold_idx="$2"
    local inner_fold_idx=0
    local cache_dir="$PROCESSED_DIR_NEW/cv/$dataset_name/fold$((outer_fold_idx + 1))"
    local split_tag="outer${outer_fold_idx}_inner${inner_fold_idx}"
    local train_file="$cache_dir/${split_tag}_train.pt"
    local valid_file="$cache_dir/${split_tag}_valid.pt"
    local test_file="$cache_dir/${split_tag}_test.pt"
    if [ ! -f "$train_file" ] || [ ! -f "$valid_file" ] || [ ! -f "$test_file" ]; then
        return 1
    fi
    local train_size
    local valid_size
    local test_size
    train_size=$(stat -c%s "$train_file" 2>/dev/null || echo "0")
    valid_size=$(stat -c%s "$valid_file" 2>/dev/null || echo "0")
    test_size=$(stat -c%s "$test_file" 2>/dev/null || echo "0")
    if [ "$train_size" -lt 1024 ] || [ "$valid_size" -lt 1024 ] || [ "$test_size" -lt 1024 ]; then
        return 1
    fi
    return 0
}

echo -e "${BLUE}📊 AEGNN-M TDC Dataset Batch Preprocessing (NEW, drop descriptor nonfinite)${NC}"
echo ""

if [ $# -lt 1 ]; then
    echo -e "${YELLOW}Usage:${NC}"
    echo -e "  ${GREEN}bash preprocess_all_datasets_new.sh <dataset_name>${NC}   # preprocess single dataset"
    echo -e "  ${GREEN}bash preprocess_all_datasets_new.sh all${NC}              # preprocess all datasets"
    echo ""
    echo -e "${YELLOW}Tip:${NC} Use the Python script to list datasets:"
    echo -e "  ${GREEN}python scripts/preprocess_tdc_data_new.py --list_datasets${NC}"
    exit 1
fi

TARGET_DATASET="$1"

echo -e "${BLUE}Select processing mode:${NC}"
echo -e "  1) ${GREEN}Standard mode${NC} (TDC Random Split) -> ${GREEN}${PROCESSED_DIR_NEW}${NC}"
echo -e "  2) ${GREEN}CV mode${NC} (Nested Cross-Validation) -> ${GREEN}${PROCESSED_DIR_NEW}/cv${NC}"
read -p "Enter choice (1 or 2, default=1): " mode_choice
mode_choice=${mode_choice:-1}

if [[ "$mode_choice" == "2" ]]; then
    USE_CV=true
    echo -e "${BLUE}Selected: CV mode (Nested Cross-Validation)${NC}"
else
    USE_CV=false
    echo -e "${BLUE}Selected: Standard mode${NC}"
fi
echo ""

datasets=($(list_tdc_datasets))
if [ ${#datasets[@]} -eq 0 ]; then
    echo -e "${YELLOW}⚠️  No TDC datasets found${NC}"
    exit 1
fi

# Filter datasets based on TARGET_DATASET
if [ "$TARGET_DATASET" != "all" ]; then
    found=false
    for ds in "${datasets[@]}"; do
        if [ "$ds" == "$TARGET_DATASET" ]; then
            found=true
            break
        fi
    done
    if [ "$found" != "true" ]; then
        echo -e "${RED}❌ Unknown dataset: ${TARGET_DATASET}${NC}"
        echo -e "${YELLOW}Available datasets:${NC}"
        printf '%s\n' "${datasets[@]}"
        exit 1
    fi
    datasets=("$TARGET_DATASET")
    echo -e "${BLUE}Target: single dataset -> ${GREEN}${TARGET_DATASET}${NC}"
else
    echo -e "${BLUE}Target: all datasets -> ${GREEN}${#datasets[@]}${NC} datasets${NC}"
fi
echo ""

echo -e "${BLUE}⚙️  Preprocessing Configuration (NEW):${NC}"
echo -e "  ${GREEN}Data Path${NC}: $TDC_DATA_DIR"
echo -e "  ${GREEN}Output Root${NC}: $PROCESSED_DIR_NEW"
echo -e "  ${GREEN}Nonfinite Descriptor Policy${NC}: ${GREEN}DROP${NC}"
if [ "$USE_CV" = "true" ]; then
    echo -e "  ${GREEN}Seed${NC}: $SEED (CV mode fixed)"
    echo -e "  ${GREEN}Outer Folds${NC}: $OUTER_FOLDS"
    echo -e "  ${GREEN}Inner Folds${NC}: $INNER_FOLDS"
else
    echo -e "  ${GREEN}Seeds${NC}: ${SEED_LIST[*]}"
fi
echo -e "  ${GREEN}Conformers${NC}: $NUM_CONFORMERS"
echo -e "  ${GREEN}Optimize${NC}: $OPTIMIZE_CONFORMERS"
echo -e "  ${GREEN}Add Hydrogens${NC}: $ADD_HYDROGENS"
echo -e "  ${GREEN}Use Fingerprint${NC}: $USE_FINGERPRINT"
if [ -n "$DESCRIPTOR_DIM" ]; then
    echo -e "  ${GREEN}Descriptor Dimension${NC}: $DESCRIPTOR_DIM"
else
    echo -e "  ${GREEN}Descriptor Dimension${NC}: All available (~217)"
fi
echo -e "  ${GREEN}Parallel Workers${NC}: $NUM_WORKERS"
echo ""

if [ "$TARGET_DATASET" == "all" ]; then
    prompt_target="all datasets"
else
    prompt_target="dataset '$TARGET_DATASET'"
fi
read -p "Start preprocessing ${prompt_target} (NEW)? (y/N): " confirm
if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
    echo "Cancelled"
    exit 0
fi

mkdir -p "$PROCESSED_DIR_NEW"

success_count=0
skip_count=0
fail_count=0
failed_items=()

for dataset in "${datasets[@]}"; do
    if ! validate_tdc_dataset "$dataset"; then
        echo -e "${RED}❌ Invalid TDC dataset: $dataset${NC}"
        ((fail_count++))
        continue
    fi

    if [ "$USE_CV" = "true" ]; then
        echo ""
        echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo -e "${BLUE}Processing (CV): ${GREEN}$dataset${NC}"
        echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

        dataset_fail=0
        dataset_skip=0
        dataset_ok=0

        for outer_fold_idx in $(seq 0 $((OUTER_FOLDS - 1))); do
            inner_fold_idx=$(( (outer_fold_idx + 1) % INNER_FOLDS ))

            if is_tdc_cv_processed_new "$dataset" "$outer_fold_idx"; then
                echo -e "  ${GREEN}✅ fold$((outer_fold_idx + 1)) already processed, skipping${NC}"
                ((dataset_skip++))
                continue
            fi

            python -u scripts/preprocess_tdc_data_new.py \
                --data_name "$dataset" \
                --data_path "$TDC_DATA_DIR" \
                --seed "$SEED" \
                --processed_dir "$PROCESSED_DIR_NEW" \
                --num_conformers "$NUM_CONFORMERS" \
                $([ "$OPTIMIZE_CONFORMERS" = "true" ] && echo "--optimize_conformers" || echo "--no_optimize_conformers") \
                $([ "$ADD_HYDROGENS" = "true" ] && echo "--add_hydrogens" || echo "--no_hydrogens") \
                $([ "$USE_FINGERPRINT" = "true" ] && echo "--use_fingerprint" || echo "") \
                $([ "$USE_FINGERPRINT" = "true" ] && echo "--fingerprint_bits $FINGERPRINT_BITS" || echo "") \
                $([ -n "$DESCRIPTOR_DIM" ] && echo "--descriptor_dim $DESCRIPTOR_DIM" || echo "") \
                --use_cv \
                --outer_fold_idx "$outer_fold_idx" \
                --inner_fold_idx "$inner_fold_idx" \
                --outer_folds "$OUTER_FOLDS" \
                --inner_folds "$INNER_FOLDS" \
                --num_workers "$NUM_WORKERS" \
                --nonfinite_policy drop
            exit_code=$?

            if [ $exit_code -eq 0 ]; then
                ((dataset_ok++))
            else
                ((dataset_fail++))
                failed_items+=("$dataset/cv_fold$((outer_fold_idx + 1)) (exit_code=$exit_code)")
            fi
        done

        if [ $dataset_fail -eq 0 ] && [ $dataset_ok -gt 0 ]; then
            ((success_count++))
        elif [ $dataset_skip -eq $OUTER_FOLDS ]; then
            ((skip_count++))
        else
            ((fail_count++))
        fi
    else
        echo ""
        echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo -e "${BLUE}Processing (Standard): ${GREEN}$dataset${NC}"
        echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

        if is_tdc_processed_new "$dataset"; then
            echo -e "${GREEN}✅ Already preprocessed (all seeds), skipping${NC}"
            ((skip_count++))
            continue
        fi

        dataset_fail=0
        dataset_skip=0
        dataset_ok=0

        for seed in "${SEED_LIST[@]}"; do
            if is_seed_processed_new "$dataset" "$seed"; then
                echo -e "  ${GREEN}✅ seed$seed already processed, skipping${NC}"
                ((dataset_skip++))
                continue
            fi

            python -u scripts/preprocess_tdc_data_new.py \
                --data_name "$dataset" \
                --data_path "$TDC_DATA_DIR" \
                --seed "$seed" \
                --processed_dir "$PROCESSED_DIR_NEW" \
                --num_conformers "$NUM_CONFORMERS" \
                $([ "$OPTIMIZE_CONFORMERS" = "true" ] && echo "--optimize_conformers" || echo "--no_optimize_conformers") \
                $([ "$ADD_HYDROGENS" = "true" ] && echo "--add_hydrogens" || echo "--no_hydrogens") \
                $([ "$USE_FINGERPRINT" = "true" ] && echo "--use_fingerprint" || echo "") \
                $([ "$USE_FINGERPRINT" = "true" ] && echo "--fingerprint_bits $FINGERPRINT_BITS" || echo "") \
                $([ -n "$DESCRIPTOR_DIM" ] && echo "--descriptor_dim $DESCRIPTOR_DIM" || echo "") \
                --num_workers "$NUM_WORKERS" \
                --nonfinite_policy drop
            exit_code=$?

            if [ $exit_code -eq 0 ]; then
                ((dataset_ok++))
            else
                ((dataset_fail++))
                failed_items+=("$dataset/seed$seed (exit_code=$exit_code)")
            fi
        done

        if [ $dataset_fail -eq 0 ] && [ $dataset_ok -gt 0 ]; then
            ((success_count++))
        elif [ $dataset_skip -eq ${#SEED_LIST[@]} ]; then
            ((skip_count++))
        else
            ((fail_count++))
        fi
    fi
done

echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}📊 Preprocessing Summary (NEW):${NC}"
echo -e "  ${GREEN}Success${NC}: $success_count"
echo -e "  ${YELLOW}Skipped${NC}: $skip_count"
echo -e "  ${RED}Failed${NC}: $fail_count"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

if [ ${#failed_items[@]} -gt 0 ]; then
    echo ""
    echo -e "${RED}❌ Failed items:${NC}"
    for item in "${failed_items[@]}"; do
        echo -e "  - $item"
    done
    exit 1
fi

echo ""
echo -e "${GREEN}✅ All preprocessing tasks completed (NEW)!${NC}"
echo -e "${GREEN}   Output root: ${PROCESSED_DIR_NEW}${NC}"



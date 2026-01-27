#!/usr/bin/env bash

# AEGNN-M Batch preprocessing script for TDC datasets
# Used to preprocess all TDC datasets in the data/data_tdc directory at once
# Uses prepare_tdc_dataset.py for processing

set -uo pipefail
# Note: -e is not used, so even if one dataset processing fails, other datasets will continue to be processed

# Color definitions
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Dataset directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1
TDC_DATA_DIR="data/data_tdc"
PROCESSED_DIR="data/processed_tdc_data"
PROCESSED_CV_DIR="data/processed_tdc_data_cv"

# Preprocessing parameters (consistent with prepare_tdc_dataset.py)
# Note: Adjust these parameters to balance speed vs quality
SEED=42  # Random seed for CV mode (consistent with fusion_model, which uses SEED=42 for CV)
# Note: For standard mode, fusion_model uses seed_list=[1,2,3,4,5], but CV mode uses fixed SEED=42
SEED_LIST=(1 2 3 4 5)  # Seed list for standard mode (consistent with fusion_model)
NUM_CONFORMERS=10  # Reduce to 5-8 for faster processing (lower quality)
OPTIMIZE_CONFORMERS=true  # Set to false for faster processing (lower quality)
ADD_HYDROGENS=true
USE_FINGERPRINT=false  # Disable molecular fingerprints (descriptor is always enabled)
FINGERPRINT_BITS=2048
DESCRIPTOR_DIM=""  # Empty means use all available RDKit descriptors (~217)
NUM_WORKERS=3  # Number of parallel workers (1=sequential, >1=parallel, 0=use all CPU cores)

# CV mode parameters
USE_CV=false  # Set to true to process CV mode instead of standard mode
OUTER_FOLDS=5  # Number of outer folds for CV
INNER_FOLDS=4  # Number of inner folds for CV
# For CV mode, we process all outer folds (0-4) with inner_fold_idx=0
# Inner folds are typically processed dynamically during training

# Function: List all available TDC datasets
list_tdc_datasets() {
    local datasets=()
    local admet_dir="$TDC_DATA_DIR/admet_group"
    
    if [ -d "$admet_dir" ]; then
        while IFS= read -r dir; do
            if [ -d "$dir" ]; then
                local dataset_name=$(basename "$dir")
                # Check if it has test.csv and train_val.csv (TDC format)
                if [ -f "$dir/test.csv" ] && [ -f "$dir/train_val.csv" ]; then
                    datasets+=("$dataset_name")
                fi
            fi
        done < <(find "$admet_dir" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort)
    fi
    
    printf '%s\n' "${datasets[@]}"
}

# Function: Validate TDC dataset exists
validate_tdc_dataset() {
    local dataset_name="$1"
    local dataset_dir="$TDC_DATA_DIR/admet_group/$dataset_name"
    
    if [ -d "$dataset_dir" ] && [ -f "$dataset_dir/test.csv" ] && [ -f "$dataset_dir/train_val.csv" ]; then
        return 0
    fi
    return 1
}

# Function: Check if a specific seed is already processed (standard mode)
# Returns 0 if processed, 1 if not processed
is_seed_processed() {
    local dataset_name="$1"
    local seed="$2"
    
    local cache_dir="$PROCESSED_DIR/$dataset_name/seed$seed"
    local train_file="$cache_dir/train.pt"
    local valid_file="$cache_dir/valid.pt"
    local test_file="$cache_dir/test.pt"
    
    # Check if all three files exist and have reasonable size
    if [ ! -f "$train_file" ] || [ ! -f "$valid_file" ] || [ ! -f "$test_file" ]; then
        return 1  # Not processed
    fi
    
    # Check file sizes (empty or very small files are likely incomplete)
    local train_size=$(stat -c%s "$train_file" 2>/dev/null || stat -f%z "$train_file" 2>/dev/null || echo "0")
    local valid_size=$(stat -c%s "$valid_file" 2>/dev/null || stat -f%z "$valid_file" 2>/dev/null || echo "0")
    local test_size=$(stat -c%s "$test_file" 2>/dev/null || stat -f%z "$test_file" 2>/dev/null || echo "0")
    
    if [ "$train_size" -lt 1024 ] || [ "$valid_size" -lt 1024 ] || [ "$test_size" -lt 1024 ]; then
        return 1  # Likely incomplete
    fi
    
    return 0  # Appears to be processed
}

# Function: Check if TDC dataset is already processed (standard mode)
# Checks if all seeds in SEED_LIST are processed
is_tdc_processed() {
    local dataset_name="$1"
    
    # Check if all seeds are processed
    for seed in "${SEED_LIST[@]}"; do
        if ! is_seed_processed "$dataset_name" "$seed"; then
            return 1  # At least one seed is not processed
        fi
    done
    
    return 0  # All seeds appear to be processed
}

# Function: Check if TDC dataset is already processed (CV mode)
is_tdc_cv_processed() {
    local dataset_name="$1"
    local outer_fold_idx="$2"
    local inner_fold_idx=0  # Check first inner fold as representative
    local cache_dir="$PROCESSED_CV_DIR/$dataset_name/fold$((outer_fold_idx + 1))"
    local split_tag="outer${outer_fold_idx}_inner${inner_fold_idx}"
    local train_file="$cache_dir/${split_tag}_train.pt"
    local valid_file="$cache_dir/${split_tag}_valid.pt"
    local test_file="$cache_dir/${split_tag}_test.pt"
    
    # Check if all three files exist and have reasonable size
    if [ ! -f "$train_file" ] || [ ! -f "$valid_file" ] || [ ! -f "$test_file" ]; then
        return 1  # Not processed
    fi
    
    # Check file sizes
    local train_size=$(stat -c%s "$train_file" 2>/dev/null || stat -f%z "$train_file" 2>/dev/null || echo "0")
    local valid_size=$(stat -c%s "$valid_file" 2>/dev/null || stat -f%z "$valid_file" 2>/dev/null || echo "0")
    local test_size=$(stat -c%s "$test_file" 2>/dev/null || stat -f%z "$test_file" 2>/dev/null || echo "0")
    
    if [ "$train_size" -lt 1024 ] || [ "$valid_size" -lt 1024 ] || [ "$test_size" -lt 1024 ]; then
        return 1  # Likely incomplete
    fi
    
    return 0  # Appears to be processed
}

# Function: Format progress output with timestamp
format_progress_output() {
    while IFS= read -r line || [ -n "$line" ]; do
        # Check if line is a tqdm progress bar - allow it to pass through for real-time updates
        # tqdm format: "Processing molecules: XX%|████...| X/Y [time<remaining, rate]"
        if [[ "$line" =~ Processing[[:space:]]+molecules.*% ]] || \
           [[ "$line" =~ [0-9]+%[[:space:]]*\|.*\|.*\[.*\] ]]; then
            # Allow tqdm progress bars to pass through (they update in place)
            echo "$line"
            continue
        fi
        
        # Check if line contains progress information in target format
        # Format: "Processed X/Y molecules, valid graphs: A, excluded: B"
        if [[ "$line" =~ ^Processed[[:space:]]+([0-9]+)/([0-9]+)[[:space:]]+molecules, ]]; then
            # If already has timestamp, pass through
            if [[ "$line" =~ ^\[.*\] ]]; then
                echo "$line"
            else
                # Add timestamp before "Processed"
                local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
                echo "[$timestamp] $line"
            fi
        # Also handle old format: "   Progress: X/Y molecules processed (valid: A, excluded: B, ...)"
        elif [[ "$line" =~ Progress:[[:space:]]*([0-9]+)/([0-9]+)[[:space:]]+molecules[[:space:]]+processed ]]; then
            local processed="${BASH_REMATCH[1]}"
            local total="${BASH_REMATCH[2]}"
            local valid="0"
            local excluded="0"
            
            # Extract valid and excluded from the line
            if [[ "$line" =~ valid:[[:space:]]*([0-9]+) ]]; then
                valid="${BASH_REMATCH[1]}"
            fi
            if [[ "$line" =~ excluded:[[:space:]]*([0-9]+) ]]; then
                excluded="${BASH_REMATCH[1]}"
            fi
            
            # Get current timestamp
            local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
            
            # Format output according to image format
            echo "[$timestamp] Processed $processed/$total molecules, valid graphs: $valid, excluded: $excluded"
        # Filter out batch progress bars (they're redundant with overall progress bar)
        elif [[ "$line" =~ Batch[[:space:]]+[0-9]+/[0-9]+.*% ]]; then
            # Skip individual batch progress bars
            continue
        else
            # Pass through other lines as-is
            echo "$line"
        fi
    done
}

# Main logic
echo -e "${BLUE}📊 AEGNN-M TDC Dataset Batch Preprocessing${NC}"
echo ""

# Ask for processing mode
echo -e "${BLUE}Select processing mode:${NC}"
echo -e "  1) ${GREEN}Standard mode${NC} (TDC Random Split)"
echo -e "  2) ${GREEN}CV mode${NC} (Nested Cross-Validation)"
read -p "Enter choice (1 or 2, default=1): " mode_choice
mode_choice=${mode_choice:-1}

if [[ "$mode_choice" == "2" ]]; then
    USE_CV=true
    PROCESSED_DIR="$PROCESSED_CV_DIR"
    echo -e "${BLUE}Selected: CV mode (Nested Cross-Validation)${NC}"
else
    USE_CV=false
    echo -e "${BLUE}Selected: Standard mode${NC}"
fi
echo ""

# Get all TDC datasets
datasets=($(list_tdc_datasets))

if [ ${#datasets[@]} -eq 0 ]; then
    echo -e "${YELLOW}⚠️  No TDC datasets found${NC}"
    echo "Please ensure TDC data is available in $TDC_DATA_DIR/admet_group/"
    echo "You can download TDC data using:"
    echo "  python -c \"from tdc.benchmark_group import admet_group; group = admet_group(path='$TDC_DATA_DIR'); group.download()\""
    exit 1
fi

echo -e "${BLUE}Found ${#datasets[@]} TDC datasets:${NC}"
for i in "${!datasets[@]}"; do
    dataset="${datasets[$i]}"
    if [ "$USE_CV" = "true" ]; then
        # For CV mode, check if at least one outer fold is processed
        processed_count=0
        for outer_fold in $(seq 0 $((OUTER_FOLDS - 1))); do
            if is_tdc_cv_processed "$dataset" "$outer_fold"; then
                ((processed_count++))
            fi
        done
        if [ $processed_count -eq $OUTER_FOLDS ]; then
            echo -e "  ${GREEN}$((i+1)).${NC} $dataset ${GREEN}✅ Preprocessed (all ${OUTER_FOLDS} outer folds)${NC}"
        elif [ $processed_count -gt 0 ]; then
            echo -e "  ${YELLOW}$((i+1)).${NC} $dataset ${YELLOW}⏳ Partially processed (${processed_count}/${OUTER_FOLDS} outer folds)${NC}"
        else
            echo -e "  ${YELLOW}$((i+1)).${NC} $dataset ${YELLOW}⏳ Pending${NC}"
        fi
    else
        if is_tdc_processed "$dataset"; then
            echo -e "  ${GREEN}$((i+1)).${NC} $dataset ${GREEN}✅ Preprocessed${NC}"
        else
            echo -e "  ${YELLOW}$((i+1)).${NC} $dataset ${YELLOW}⏳ Pending${NC}"
        fi
    fi
done
echo ""

# Display preprocessing configuration
echo -e "${BLUE}⚙️  Preprocessing Configuration:${NC}"
echo -e "  ${GREEN}Mode${NC}: $([ "$USE_CV" = "true" ] && echo "CV (Nested Cross-Validation)" || echo "Standard (TDC Random Split)")"
echo -e "  ${GREEN}Data Path${NC}: $TDC_DATA_DIR"
if [ "$USE_CV" = "true" ]; then
    echo -e "  ${GREEN}Seed${NC}: $SEED (CV mode uses fixed seed)"
else
    echo -e "  ${GREEN}Seeds${NC}: ${SEED_LIST[*]} (Standard mode processes all seeds, consistent with fusion_model)"
fi
if [ "$USE_CV" = "true" ]; then
    echo -e "  ${GREEN}Outer Folds${NC}: $OUTER_FOLDS"
    echo -e "  ${GREEN}Inner Folds${NC}: $INNER_FOLDS"
    echo -e "  ${GREEN}Processing${NC}: All outer folds (0-$((OUTER_FOLDS - 1))), inner_fold_idx=0"
else
    echo -e "  ${GREEN}Split Method${NC}: TDC Random Split (default)"
fi
echo -e "  ${GREEN}Conformers${NC}: $NUM_CONFORMERS"
echo -e "  ${GREEN}Optimize${NC}: $OPTIMIZE_CONFORMERS"
echo -e "  ${GREEN}Add Hydrogens${NC}: $ADD_HYDROGENS"
echo -e "  ${GREEN}Use Fingerprint${NC}: $USE_FINGERPRINT"
echo -e "  ${GREEN}Use Descriptor${NC}: true (always enabled)"
if [ -n "$DESCRIPTOR_DIM" ]; then
    echo -e "  ${GREEN}Descriptor Dimension${NC}: $DESCRIPTOR_DIM"
else
    echo -e "  ${GREEN}Descriptor Dimension${NC}: All available (~217)"
fi
echo -e "  ${GREEN}Node Features${NC}: 78D (OGB-style, RDKit)"
echo -e "  ${GREEN}Edge Features${NC}: 9D (OGB-style, bidirectional)"
echo -e "  ${GREEN}Parallel Workers${NC}: $NUM_WORKERS"
echo ""

# Ask for confirmation
read -p "Start preprocessing all datasets? (y/N): " confirm
if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
    echo "Cancelled"
    exit 0
fi

# Create processing directory
if [ "$USE_CV" = "true" ]; then
    mkdir -p "$PROCESSED_CV_DIR"
else
    mkdir -p "$PROCESSED_DIR"
fi

# Process each TDC dataset
success_count=0
skip_count=0
fail_count=0
failed_datasets=()  # Record failed datasets and reasons
failed_seeds=()     # Record failed seeds and reasons

for dataset in "${datasets[@]}"; do
    # Validate dataset exists
    if ! validate_tdc_dataset "$dataset"; then
        echo -e "${RED}❌ Invalid TDC dataset: $dataset${NC}"
        ((fail_count++))
        continue
    fi
    
    if [ "$USE_CV" = "true" ]; then
        # CV mode: Process all outer folds
        echo ""
        echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo -e "${BLUE}Processing TDC dataset (CV mode): ${GREEN}$dataset${NC}"
        echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        
        dataset_success=0
        dataset_skip=0
        dataset_fail=0
        
        for outer_fold_idx in $(seq 0 $((OUTER_FOLDS - 1))); do
            echo ""
            echo -e "${BLUE}  Processing outer fold ${GREEN}$((outer_fold_idx + 1))/$OUTER_FOLDS${NC}..."
            
            # Calculate inner_fold_idx (consistent with fusion_model: inner_fold_idx = (outer_fold_idx + 1) % 4)
            inner_fold_idx=$(( (outer_fold_idx + 1) % INNER_FOLDS ))
            echo -e "  ${BLUE}  Using inner fold: ${GREEN}$((inner_fold_idx + 1))/$INNER_FOLDS${NC} (fusion_model formula: (outer+1) % inner_folds)"
            
            # Check if already processed
            if is_tdc_cv_processed "$dataset" "$outer_fold_idx"; then
                echo -e "  ${GREEN}✅ Outer fold $((outer_fold_idx + 1)) already processed, skipping${NC}"
                ((dataset_skip++))
                continue
            fi
            
            # Execute preprocessing for this outer fold
            # Note: Consistent with fusion_model, each outer_fold will call set_seed()
            python -u scripts/preprocess_tdc_data.py \
                --data_name "$dataset" \
                --data_path "$TDC_DATA_DIR" \
                --seed "$SEED" \
                --processed_dir "$PROCESSED_CV_DIR" \
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
                --num_workers "$NUM_WORKERS"
            python_exit_code=$?
            
            if [ $python_exit_code -eq 0 ]; then
                echo -e "  ${GREEN}✅ Outer fold $((outer_fold_idx + 1)) processed successfully${NC}"
                ((dataset_success++))
            else
                echo -e "  ${RED}❌ Outer fold $((outer_fold_idx + 1)) failed (exit code: $python_exit_code)${NC}"
                failed_seeds+=("$dataset/fold$((outer_fold_idx + 1)) (exit_code=$python_exit_code)")
                ((dataset_fail++))
            fi
        done
        
        # Summary for this dataset
        if [ $dataset_fail -eq 0 ] && [ $dataset_success -gt 0 ]; then
            echo -e "${GREEN}✅ Dataset $dataset: $dataset_success/$OUTER_FOLDS outer folds processed${NC}"
            ((success_count++))
        elif [ $dataset_skip -eq $OUTER_FOLDS ]; then
            echo -e "${GREEN}✅ Dataset $dataset: All outer folds already processed${NC}"
            ((skip_count++))
        else
            echo -e "${YELLOW}⚠️  Dataset $dataset: $dataset_success/$OUTER_FOLDS successful, $dataset_fail failed${NC}"
            failed_datasets+=("$dataset (CV: $dataset_success/$OUTER_FOLDS successful, $dataset_fail failed)")
            ((fail_count++))
        fi
        
    else
        # Standard mode: Process all seeds (1,2,3,4,5) like fusion_model
        echo ""
        echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo -e "${BLUE}Processing TDC dataset (Standard mode): ${GREEN}$dataset${NC}"
        echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        
        # Check if already preprocessed (all seeds)
        if is_tdc_processed "$dataset"; then
            echo -e "${GREEN}✅ Already preprocessed (all seeds), skipping${NC}"
            ((skip_count++))
            continue
        fi
        
        # Process each seed (consistent with fusion_model: seed_list=[1,2,3,4,5])
        dataset_success=0
        dataset_skip=0
        dataset_fail=0
        
        for seed in "${SEED_LIST[@]}"; do
            echo ""
            echo -e "${BLUE}  Processing seed ${GREEN}$seed${NC}..."
            
            # Check if this seed is already processed (use unified check function to avoid duplicate logic)
            if is_seed_processed "$dataset" "$seed"; then
                echo -e "  ${GREEN}✅ Seed $seed already processed, skipping${NC}"
                ((dataset_skip++))
                continue
            fi
            
            # Execute preprocessing for this seed
            python -u scripts/preprocess_tdc_data.py \
                --data_name "$dataset" \
                --data_path "$TDC_DATA_DIR" \
                --seed "$seed" \
                --processed_dir "$PROCESSED_DIR" \
                --num_conformers "$NUM_CONFORMERS" \
                $([ "$OPTIMIZE_CONFORMERS" = "true" ] && echo "--optimize_conformers" || echo "--no_optimize_conformers") \
                $([ "$ADD_HYDROGENS" = "true" ] && echo "--add_hydrogens" || echo "--no_hydrogens") \
                $([ "$USE_FINGERPRINT" = "true" ] && echo "--use_fingerprint" || echo "") \
                $([ "$USE_FINGERPRINT" = "true" ] && echo "--fingerprint_bits $FINGERPRINT_BITS" || echo "") \
                $([ -n "$DESCRIPTOR_DIM" ] && echo "--descriptor_dim $DESCRIPTOR_DIM" || echo "") \
                --num_workers "$NUM_WORKERS"
            python_exit_code=$?
            
            if [ $python_exit_code -eq 0 ]; then
                echo -e "  ${GREEN}✅ Seed $seed processed successfully${NC}"
                ((dataset_success++))
            else
                echo -e "  ${RED}❌ Seed $seed failed (exit code: $python_exit_code)${NC}"
                failed_seeds+=("$dataset/seed$seed (exit_code=$python_exit_code)")
                ((dataset_fail++))
            fi
        done
        
        # Summary for this dataset
        if [ $dataset_fail -eq 0 ] && [ $dataset_success -gt 0 ]; then
            echo -e "${GREEN}✅ Dataset $dataset: $dataset_success/${#SEED_LIST[@]} seeds processed${NC}"
            ((success_count++))
        elif [ $dataset_skip -eq ${#SEED_LIST[@]} ]; then
            echo -e "${GREEN}✅ Dataset $dataset: All seeds already processed${NC}"
            ((skip_count++))
        else
            echo -e "${YELLOW}⚠️  Dataset $dataset: $dataset_success/${#SEED_LIST[@]} successful, $dataset_fail failed${NC}"
            failed_datasets+=("$dataset ($dataset_success/${#SEED_LIST[@]} successful, $dataset_fail failed)")
            ((fail_count++))
        fi
    fi
done

# Summary
echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}📊 Preprocessing Summary:${NC}"
echo -e "  ${GREEN}Success${NC}: $success_count"
echo -e "  ${YELLOW}Skipped${NC}: $skip_count"
echo -e "  ${RED}Failed${NC}: $fail_count"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

# Display failure details
if [ ${#failed_datasets[@]} -gt 0 ] || [ ${#failed_seeds[@]} -gt 0 ]; then
    echo ""
    echo -e "${RED}❌ Failed Details:${NC}"
    if [ ${#failed_datasets[@]} -gt 0 ]; then
        echo -e "  ${RED}Failed Datasets:${NC}"
        for failed in "${failed_datasets[@]}"; do
            echo -e "    - $failed"
        done
    fi
    if [ ${#failed_seeds[@]} -gt 0 ]; then
        echo -e "  ${RED}Failed Seeds:${NC}"
        for failed in "${failed_seeds[@]}"; do
            echo -e "    - $failed"
        done
    fi
    echo ""
    echo -e "${YELLOW}💡 Tip: Check the error messages above for details.${NC}"
    echo -e "${YELLOW}   You can re-run the script to retry failed items.${NC}"
fi

if [ $fail_count -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✅ All TDC datasets preprocessing completed!${NC}"
    echo ""
    if [ "$USE_CV" = "true" ]; then
        echo -e "${BLUE}💡 Processed data is saved to:${NC}"
        echo -e "   ${GREEN}$PROCESSED_CV_DIR/{dataset_name}/fold{outer_fold+1}/outer{outer_fold}_inner{inner_fold}_{split}.pt${NC}"
        echo ""
        echo -e "${BLUE}💡 You can now use the processed data with:${NC}"
        echo -e "   ${GREEN}from utils.prepare_tdc_dataset import load_tdc_dataset_cv${NC}"
        echo -e "   ${GREEN}train_graphs, valid_graphs, test_graphs = load_tdc_dataset_cv(...)${NC}"
    else
        echo -e "${BLUE}💡 Processed data is saved to:${NC}"
        echo -e "   ${GREEN}$PROCESSED_DIR/{dataset_name}/seed{1..5}/{split}.pt${NC}"
        echo -e "   (Each dataset has 5 seeds: seed1, seed2, seed3, seed4, seed5)"
        echo ""
        echo -e "${BLUE}💡 You can now use the processed data with:${NC}"
        echo -e "   ${GREEN}from utils.prepare_tdc_dataset import load_tdc_dataset${NC}"
        echo -e "   ${GREEN}train_graphs, valid_graphs, test_graphs = load_tdc_dataset(...)${NC}"
    fi
    echo ""
    echo -e "${BLUE}The data will be automatically loaded from cache if already processed.${NC}"
else
    echo ""
    echo -e "${YELLOW}⚠️  Some datasets preprocessing failed, please check error messages${NC}"
    exit 1
fi


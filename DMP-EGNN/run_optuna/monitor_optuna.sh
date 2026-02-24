#!/bin/bash

# Enhanced Optuna Monitor with advanced features
# Usage: ./monitor_optuna.sh [dataset_name] [refresh_interval_seconds]
# Interactive controls:
#   's' - Switch display mode (compact/detailed)
#   'p' - Pause/resume refresh
#   'e' - Export current status to file
#   'q' - Quit
#
# Note: This script should be run from the DMP-EGNN root directory, or it will automatically
#       change to the correct directory based on script location.

# Get script directory and change to project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}" || {
    echo "❌ Error: Cannot change to project root directory: ${ROOT_DIR}"
    exit 1
}

DATASET=${1:-"ames"}
INTERVAL=${2:-10}
PLANNED_TRIALS=${3:-""}  # Optional: manually specify planned total trials
VERSION_MODE=${4:-"auto"}  # Optional: "mod", "mod_new", or "auto" (default: auto-detect)

# Auto-detect version mode based on running processes
if [ "$VERSION_MODE" = "auto" ]; then
    # Check for optuna_parallel_mod_new.sh processes (uses optuna_serach_mod.py but with mod_new config)
    # Method 1: Check for optuna_parallel_mod_new.sh process
    MOD_NEW_LAUNCHER=$(pgrep -f "optuna_parallel_mod_new.sh.*$DATASET" 2>/dev/null | wc -l)
    # Method 2: Check for optuna_serach_mod.py with mod_new log directory pattern
    MOD_NEW_LOGS=$(pgrep -f "optuna_serach_mod.py.*$DATASET" 2>/dev/null | xargs -I {} sh -c 'ps -p {} -o command= 2>/dev/null | grep -q "optuna_launcher_mod_new" && echo "1" || echo "0"' 2>/dev/null | grep -c "1" || echo "0")
    # Method 3: Check for optuna_serach_mod_new.py (if it exists in future)
    NEW_VERSION_PIDS=$(pgrep -f "optuna_serach_mod_new.py.*$DATASET" 2>/dev/null | wc -l)
    # Method 4: Check for old version (optuna_serach_mod.py with old log directory)
    OLD_VERSION_PIDS=$(pgrep -f "optuna_serach_mod.py.*$DATASET" 2>/dev/null | xargs -I {} sh -c 'ps -p {} -o command= 2>/dev/null | grep -q "optuna_launcher_mod_new" && echo "0" || echo "1"' 2>/dev/null | grep -c "1" || echo "0")
    
    if [ "$MOD_NEW_LAUNCHER" -gt 0 ] || [ "$MOD_NEW_LOGS" -gt 0 ] || [ "$NEW_VERSION_PIDS" -gt 0 ]; then
        VERSION_MODE="mod_new"
    elif [ "$OLD_VERSION_PIDS" -gt 0 ]; then
        VERSION_MODE="mod"
    else
        # If no running processes, check which database/study exists
        # Check for mod_new database first (optuna_parallel_mod_new.sh uses this path)
        if [ -f "optuna_edmpnn_results/optuna_mod_new.db" ]; then
            # Try to load study with aegnn_mod naming (used by optuna_parallel_mod_new.sh)
            if python3 -c "import optuna; optuna.load_study(study_name='aegnn_mod_${DATASET}_opt', storage='sqlite:///optuna_edmpnn_results/optuna_mod_new.db')" 2>/dev/null; then
                VERSION_MODE="mod_new"
            else
                VERSION_MODE="mod"  # Fallback to old version
            fi
        elif [ -f "optuna_edmpnn_results_new/optuna_mod_new.db" ]; then
            # Try to load study with new naming (legacy check)
            if python3 -c "import optuna; optuna.load_study(study_name='edmpnn_mod_new_${DATASET}_opt', storage='sqlite:///optuna_edmpnn_results_new/optuna_mod_new.db')" 2>/dev/null; then
                VERSION_MODE="mod_new"
            else
                VERSION_MODE="mod"  # Fallback to old version
            fi
        else
            VERSION_MODE="mod"  # Default to old version
        fi
    fi
fi

# Set configuration based on version mode
# Note: optuna_parallel_mod_new.sh uses optuna_serach_mod.py but with mod_new database and study naming
if [ "$VERSION_MODE" = "mod_new" ]; then
    # optuna_parallel_mod_new.sh uses:
    #   - Database: optuna_edmpnn_results/optuna_mod_new.db
    #   - Study: aegnn_mod_${dataset}_opt
    #   - Process: optuna_serach_mod.py (not optuna_serach_mod_new.py)
    #   - Log: logs/optuna_launcher_mod_new/
    # First try the path used by optuna_parallel_mod_new.sh
    if [ -f "optuna_edmpnn_results/optuna_mod_new.db" ]; then
        DB_PATH="optuna_edmpnn_results/optuna_mod_new.db"
        STORAGE_URL="sqlite:///$DB_PATH"
        STUDY_NAME="aegnn_mod_${DATASET}_opt"
        STUDY_PREFIX="aegnn_mod"
        CHECKPOINT_DIR="optuna_mod_new"
        LOG_DIR="optuna_launcher_mod_new"
        PROCESS_PATTERN="optuna_serach_mod.py"  # optuna_parallel_mod_new.sh uses this script
        PER_SEED_MODE=false  # Fusion Model Logic: single study, all seeds use same hyperparameters
    else
        # Fallback to legacy mod_new path (if it exists)
        DB_PATH="optuna_edmpnn_results_new/optuna_mod_new.db"
        STORAGE_URL="sqlite:///$DB_PATH"
        STUDY_NAME="edmpnn_mod_new_${DATASET}_opt"
        STUDY_PREFIX="edmpnn_mod_new"
        CHECKPOINT_DIR="optuna_mod_new"
        LOG_DIR="optuna_launcher_mod_new"
        PROCESS_PATTERN="optuna_serach_mod_new.py"
        PER_SEED_MODE=true  # Legacy per-seed optimization
    fi
else
    # Default to old version (mod) - now also uses per-seed optimization
    DB_PATH="optuna_edmpnn_results/optuna_mod.db"
    STORAGE_URL="sqlite:///$DB_PATH"
    STUDY_NAME="aegnn_mod_${DATASET}_opt"  # Legacy format (for backward compatibility)
    STUDY_PREFIX="aegnn_mod"
    CHECKPOINT_DIR="optuna_mod"
    LOG_DIR="optuna_launcher_mod"
    PROCESS_PATTERN="optuna_serach_mod.py"
    PER_SEED_MODE=true  # mod now uses per-seed optimization
fi

# Display mode: "compact" or "detailed"
DISPLAY_MODE="detailed"
PAUSED=false
LAST_EXPORT_TIME=""
AUTO_DETECTED_SHOWN=false  # Track if auto-detection message has been shown

# Function to read keypress (non-blocking)
read_key() {
    local key
    if read -t 0.1 -n 1 key 2>/dev/null; then
        echo "$key"
    fi
}

# Function to export status
export_status() {
    # Use current DATASET (may have been updated by auto-detection)
    local current_dataset="$DATASET"
    # Determine study name based on current dataset and version mode
    local current_study_name
    if [ "$VERSION_MODE" = "mod_new" ]; then
        current_study_name="edmpnn_mod_new_${current_dataset}_opt"
    else
        current_study_name="aegnn_mod_${current_dataset}_opt"
    fi
    
    local export_file="monitor_export_${current_dataset}_$(date +%Y%m%d_%H%M%S).txt"
    {
        echo "Optuna Monitor Export - $(date '+%Y-%m-%d %H:%M:%S')"
        echo "Dataset: $current_dataset"
        echo "=========================================="
        echo ""
    } > "$export_file"
    
    python3 << EOF >> "$export_file"
import optuna
import os
import json
from datetime import datetime

    # Determine study prefix
    # optuna_parallel_mod_new.sh uses aegnn_mod prefix with optuna_edmpnn_results/optuna_mod_new.db
    if "$VERSION_MODE" == "mod_new":
        # Check which database path is being used
        if os.path.exists("optuna_edmpnn_results/optuna_mod_new.db"):
            study_prefix = "aegnn_mod"  # optuna_parallel_mod_new.sh uses this
        else:
            study_prefix = "edmpnn_mod_new"  # Legacy mod_new uses this
    else:
        study_prefix = "aegnn_mod"

storage_url = "$STORAGE_URL"
dataset = "$current_dataset"

# Check if per-seed optimization is being used
per_seed_mode = False
seed_studies = {}
for seed_num in range(1, 6):
    seed_study_name = f"{study_prefix}_{dataset}_seed{seed_num}_opt"
    try:
        seed_study = optuna.load_study(study_name=seed_study_name, storage=storage_url)
        seed_studies[seed_num] = seed_study
        per_seed_mode = True
    except Exception:
        pass

if per_seed_mode:
    print("Per-Seed Optimization Mode:")
    print("")
    for seed_num in sorted(seed_studies.keys()):
        study = seed_studies[seed_num]
        trials = study.trials
        complete = len([t for t in trials if t.state == optuna.trial.TrialState.COMPLETE])
        running = len([t for t in trials if t.state == optuna.trial.TrialState.RUNNING])
        pruned = len([t for t in trials if t.state == optuna.trial.TrialState.PRUNED])
        failed = len([t for t in trials if t.state == optuna.trial.TrialState.FAIL])
        
        print(f"Seed {seed_num} Study Status:")
        print(f"  Total Trials: {len(trials)}")
        print(f"  Complete: {complete}")
        print(f"  Running: {running}")
        print(f"  Pruned: {pruned}")
        print(f"  Failed: {failed}")
        
        if complete > 0:
            print(f"  Best Result: Trial #{study.best_trial.number}: {study.best_value:.6f}")
            print(f"  Best Params: {study.best_trial.params}")
        print("")
else:
    # Legacy mode
    try:
        study = optuna.load_study(study_name="$current_study_name", storage=storage_url)
        trials = study.trials
        
        print("Study Status (Legacy Mode):")
        print(f"  Total Trials: {len(trials)}")
        print(f"  Complete: {len([t for t in trials if t.state == optuna.trial.TrialState.COMPLETE])}")
        print(f"  Running: {len([t for t in trials if t.state == optuna.trial.TrialState.RUNNING])}")
        print(f"  Pruned: {len([t for t in trials if t.state == optuna.trial.TrialState.PRUNED])}")
        print(f"  Failed: {len([t for t in trials if t.state == optuna.trial.TrialState.FAIL])}")
        print("")
        
        if len([t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]) > 0:
            print(f"Best Result:")
            print(f"  Trial #{study.best_trial.number}: {study.best_value:.6f}")
            print(f"  Params: {study.best_trial.params}")
            print("")
        
        print("All Trials:")
        for trial in sorted(trials, key=lambda t: t.number):
            print(f"  Trial #{trial.number}: {trial.state.name} - Value: {trial.value}")
    except Exception as e:
        print(f"Error: {e}")
EOF
    
    echo "📄 Status exported to: $export_file"
    LAST_EXPORT_TIME=$(date '+%H:%M:%S')
}

echo "📊 Enhanced Optuna Monitor for: $DATASET"
echo "🔄 Refresh interval: ${INTERVAL} seconds"
echo "🔧 Version mode: $VERSION_MODE (${CHECKPOINT_DIR})"
if [ -n "$PLANNED_TRIALS" ]; then
    echo "📋 Planned total trials: $PLANNED_TRIALS"
fi
echo "⌨️  Controls: s=switch mode, p=pause, e=export, q=quit"
echo "💡 Usage: ./monitor_optuna.sh [dataset] [interval] [planned_trials] [version: mod|mod_new|auto]"
echo "=========================================="
echo ""

while true; do
    # Check for keypress
    KEY=$(read_key)
    case "$KEY" in
        s|S)
            if [ "$DISPLAY_MODE" = "detailed" ]; then
                DISPLAY_MODE="compact"
            else
                DISPLAY_MODE="detailed"
            fi
            ;;
        p|P)
            PAUSED=$([ "$PAUSED" = true ] && echo false || echo true)
            ;;
        e|E)
            export_status
            sleep 1
            ;;
        q|Q)
            echo ""
            echo "👋 Exiting monitor..."
            exit 0
            ;;
    esac
    
    if [ "$PAUSED" = true ]; then
        clear
        echo "📊 Optuna Monitor - PAUSED - $(date '+%Y-%m-%d %H:%M:%S')"
        echo "Press 'p' to resume, 'q' to quit"
        sleep 1
        continue
    fi
    
    clear
    echo "📊 Optuna Real-time Monitor - $(date '+%Y-%m-%d %H:%M:%S')"
    if [ "$DISPLAY_MODE" = "compact" ]; then
        echo "Mode: COMPACT | Press 's' for detailed, 'p' to pause, 'e' to export, 'q' to quit"
    else
        echo "Mode: DETAILED | Press 's' for compact, 'p' to pause, 'e' to export, 'q' to quit"
    fi
    if [ -n "$LAST_EXPORT_TIME" ]; then
        echo "Last export: $LAST_EXPORT_TIME"
    fi
    echo "=========================================="
    
    # Detect all planned datasets from optuna_parallel_mod.sh and active Optuna processes
    PLANNED_DATASETS=()
    
    # Method 1: Try to get from running optuna_parallel_mod.sh process
    PARALLEL_PROCESS=$(pgrep -af "optuna_parallel_mod\.sh" 2>/dev/null | grep -v grep | head -1)
    if [ -n "$PARALLEL_PROCESS" ]; then
        # Extract arguments after script name
        args=$(echo "$PARALLEL_PROCESS" | sed 's/.*optuna_parallel_mod\.sh[[:space:]]*//')
        
        # Check if "all" is specified
        if echo "$args" | grep -qw "all"; then
            # Get all available datasets from TDC data directory
            TDC_DATA_DIR="data/processed_tdc_data"
            if [ -d "$TDC_DATA_DIR" ]; then
                ALL_AVAILABLE=($(ls -d "$TDC_DATA_DIR"/*/ 2>/dev/null | grep -v '/\.' | xargs -n1 basename | sort))
                # Check for --exclude
                if echo "$args" | grep -qE '--exclude'; then
                    # Extract excluded datasets (simplified: get words after --exclude until next -- or mode keyword)
                    EXCLUDED_STR=$(echo "$args" | sed -n 's/.*--exclude[[:space:]]*\([^--]*\).*/\1/p' | sed 's/[[:space:]]*\(fast\|standard\|deep\|--workers.*\)$//')
                    EXCLUDED_ARRAY=($EXCLUDED_STR)
                    # Filter out excluded
                    for dataset in "${ALL_AVAILABLE[@]}"; do
                        exclude=false
                        for excl in "${EXCLUDED_ARRAY[@]}"; do
                            if [ "$dataset" == "$excl" ]; then
                                exclude=true
                                break
                            fi
                        done
                        if [ "$exclude" == false ]; then
                            PLANNED_DATASETS+=("$dataset")
                        fi
                    done
                else
                    PLANNED_DATASETS+=("${ALL_AVAILABLE[@]}")
                fi
            fi
        else
            # Extract individual dataset names (skip mode keywords and options)
            for arg in $args; do
                # Skip mode keywords
                if [[ "$arg" =~ ^(fast|standard|deep)$ ]]; then
                    continue
                fi
                # Skip option flags and their values
                if [[ "$arg" =~ ^-- ]]; then
                    continue
                fi
                # Add as dataset if it exists in TDC directory
                if [ -d "data/processed_tdc_data/$arg" ] 2>/dev/null; then
                    PLANNED_DATASETS+=("$arg")
                fi
            done
        fi
    fi
    
    # Method 2: Also get from currently running Optuna processes (as fallback)
    if [ ${#PLANNED_DATASETS[@]} -eq 0 ]; then
        OPTUNA_PROCESSES=$(pgrep -af "optuna_serach_mod.*\.py" 2>/dev/null | grep -v grep || true)
        if [ -n "$OPTUNA_PROCESSES" ]; then
            while IFS= read -r line; do
                # Extract dataset using sed (more compatible than grep -oP)
                dataset=$(echo "$line" | sed -n 's/.*--dataset[[:space:]]*\([^[:space:]]*\).*/\1/p' | head -1)
                if [ -n "$dataset" ]; then
                    PLANNED_DATASETS+=("$dataset")
                fi
            done <<< "$OPTUNA_PROCESSES"
        fi
    fi
    
    # Remove duplicates and sort
    if [ ${#PLANNED_DATASETS[@]} -gt 0 ]; then
        UNIQUE_DATASETS=($(printf '%s\n' "${PLANNED_DATASETS[@]}" | sort -u))
        if [ ${#UNIQUE_DATASETS[@]} -gt 0 ]; then
            # Join datasets with " | " separator
            DATASETS_STR="${UNIQUE_DATASETS[0]}"
            for i in "${UNIQUE_DATASETS[@]:1}"; do
                DATASETS_STR="$DATASETS_STR | $i"
            done
            echo "📋 Planned Datasets: $DATASETS_STR"
        fi
    fi
    
    # Auto-detect running dataset and n_trials from running processes
    # First check if the requested dataset is actually running
    # Re-check VERSION_MODE based on current DATASET (in case it was updated in previous iteration)
    
    # Check for optuna_parallel_mod_new.sh process first
    MOD_NEW_LAUNCHER=$(pgrep -f "optuna_parallel_mod_new.sh.*$DATASET" 2>/dev/null | wc -l)
    if [ "$MOD_NEW_LAUNCHER" -gt 0 ]; then
        # optuna_parallel_mod_new.sh detected
        if [ "$VERSION_MODE" != "mod_new" ]; then
            VERSION_MODE="mod_new"
            DB_PATH="optuna_edmpnn_results/optuna_mod_new.db"
            STORAGE_URL="sqlite:///$DB_PATH"
            CHECKPOINT_DIR="optuna_mod_new"
            LOG_DIR="optuna_launcher_mod_new"
            PROCESS_PATTERN="optuna_serach_mod.py"  # optuna_parallel_mod_new.sh uses this
            STUDY_PREFIX="aegnn_mod"
        fi
    else
        # Check for optuna_serach_mod_new.py (legacy mod_new)
        DETECTED_NEW_PIDS=$(pgrep -f "optuna_serach_mod_new.py.*$DATASET" 2>/dev/null | wc -l)
        if [ "$DETECTED_NEW_PIDS" -gt 0 ]; then
            # Ensure VERSION_MODE matches actual running process
            if [ "$VERSION_MODE" != "mod_new" ]; then
                VERSION_MODE="mod_new"
                DB_PATH="optuna_edmpnn_results_new/optuna_mod_new.db"
                STORAGE_URL="sqlite:///$DB_PATH"
                CHECKPOINT_DIR="optuna_mod_new"
                LOG_DIR="optuna_launcher_mod_new"
                PROCESS_PATTERN="optuna_serach_mod_new.py"
                STUDY_PREFIX="edmpnn_mod_new"
            fi
        else
            # Check for optuna_serach_mod.py - need to determine if it's mod_new or mod
            DETECTED_OLD_PIDS=$(pgrep -f "optuna_serach_mod.py.*$DATASET" 2>/dev/null | wc -l)
            if [ "$DETECTED_OLD_PIDS" -gt 0 ]; then
                # Check if it's using mod_new database/log directory (optuna_parallel_mod_new.sh)
                # Check log directory or database to determine version
                if [ -f "optuna_edmpnn_results/optuna_mod_new.db" ] && [ -d "logs/optuna_launcher_mod_new" ]; then
                    # Likely optuna_parallel_mod_new.sh
                    if [ "$VERSION_MODE" != "mod_new" ]; then
                        VERSION_MODE="mod_new"
                        DB_PATH="optuna_edmpnn_results/optuna_mod_new.db"
                        STORAGE_URL="sqlite:///$DB_PATH"
                        CHECKPOINT_DIR="optuna_mod_new"
                        LOG_DIR="optuna_launcher_mod_new"
                        PROCESS_PATTERN="optuna_serach_mod.py"
                        STUDY_PREFIX="aegnn_mod"
                    fi
                elif [ "$VERSION_MODE" != "mod" ]; then
                    # Regular mod version
                    VERSION_MODE="mod"
                    DB_PATH="optuna_edmpnn_results/optuna_mod.db"
                    STORAGE_URL="sqlite:///$DB_PATH"
                    CHECKPOINT_DIR="optuna_mod"
                    LOG_DIR="optuna_launcher_mod"
                    PROCESS_PATTERN="optuna_serach_mod.py"
                    STUDY_PREFIX="aegnn_mod"
                fi
            fi
        fi
    fi
    
    OPTUNA_PIDS=$(pgrep -f "$PROCESS_PATTERN.*$DATASET" 2>/dev/null | wc -l)
    ACTUAL_DATASET="$DATASET"  # Start with requested dataset
    # Update STUDY_NAME based on current DATASET and VERSION_MODE (in case DATASET was updated in previous iteration)
    # optuna_parallel_mod_new.sh uses aegnn_mod_${dataset}_opt, not edmpnn_mod_new_${dataset}_opt
    if [ "$VERSION_MODE" = "mod_new" ]; then
        # Check which database path is being used to determine study name
        if [ -f "optuna_edmpnn_results/optuna_mod_new.db" ] && [ "$DB_PATH" = "optuna_edmpnn_results/optuna_mod_new.db" ]; then
            ACTUAL_STUDY_NAME="aegnn_mod_${DATASET}_opt"  # optuna_parallel_mod_new.sh uses this
        else
            ACTUAL_STUDY_NAME="edmpnn_mod_new_${DATASET}_opt"  # Legacy mod_new uses this
        fi
    else
        ACTUAL_STUDY_NAME="aegnn_mod_${DATASET}_opt"
    fi
    
    # Try to extract n_trials from running process command line
    # Note: This is per-worker n_trials, not total
    DETECTED_WORKER_TRIALS=""
    if [ "$OPTUNA_PIDS" -gt 0 ]; then
        # Get n_trials from running process (this is per-worker)
        DETECTED_WORKER_TRIALS=$(pgrep -af "$PROCESS_PATTERN.*$DATASET" 2>/dev/null | grep -o -- "--n_trials [0-9]*" | head -1 | awk '{print $2}')
    fi
    
    if [ "$OPTUNA_PIDS" -eq 0 ]; then
        # Try to detect running dataset from any optuna process (both old and new versions)
        RUNNING_DATASET=$(pgrep -af "optuna_serach_mod" 2>/dev/null | grep -- "--dataset" | sed -n 's/.*--dataset \([^ ]*\).*/\1/p' | head -1)
        if [ -n "$RUNNING_DATASET" ] && [ "$RUNNING_DATASET" != "$DATASET" ]; then
            ACTUAL_DATASET="$RUNNING_DATASET"
            
            # Re-detect version mode based on the detected dataset's process
            # Check for optuna_parallel_mod_new.sh first
            MOD_NEW_LAUNCHER=$(pgrep -f "optuna_parallel_mod_new.sh.*$ACTUAL_DATASET" 2>/dev/null | wc -l)
            if [ "$MOD_NEW_LAUNCHER" -gt 0 ]; then
                VERSION_MODE="mod_new"
                DB_PATH="optuna_edmpnn_results/optuna_mod_new.db"
                STORAGE_URL="sqlite:///$DB_PATH"
                CHECKPOINT_DIR="optuna_mod_new"
                LOG_DIR="optuna_launcher_mod_new"
                PROCESS_PATTERN="optuna_serach_mod.py"
                STUDY_PREFIX="aegnn_mod"
            else
                DETECTED_NEW_PIDS=$(pgrep -f "optuna_serach_mod_new.py.*$ACTUAL_DATASET" 2>/dev/null | wc -l)
                if [ "$DETECTED_NEW_PIDS" -gt 0 ]; then
                    VERSION_MODE="mod_new"
                    DB_PATH="optuna_edmpnn_results_new/optuna_mod_new.db"
                    STORAGE_URL="sqlite:///$DB_PATH"
                    CHECKPOINT_DIR="optuna_mod_new"
                    LOG_DIR="optuna_launcher_mod_new"
                    PROCESS_PATTERN="optuna_serach_mod_new.py"
                    STUDY_PREFIX="edmpnn_mod_new"
                else
                    # Check if optuna_serach_mod.py is using mod_new database
                    if [ -f "optuna_edmpnn_results/optuna_mod_new.db" ] && [ -d "logs/optuna_launcher_mod_new" ]; then
                        VERSION_MODE="mod_new"
                        DB_PATH="optuna_edmpnn_results/optuna_mod_new.db"
                        STORAGE_URL="sqlite:///$DB_PATH"
                        CHECKPOINT_DIR="optuna_mod_new"
                        LOG_DIR="optuna_launcher_mod_new"
                        PROCESS_PATTERN="optuna_serach_mod.py"
                        STUDY_PREFIX="aegnn_mod"
                    else
                        VERSION_MODE="mod"
                        DB_PATH="optuna_edmpnn_results/optuna_mod.db"
                        STORAGE_URL="sqlite:///$DB_PATH"
                        CHECKPOINT_DIR="optuna_mod"
                        LOG_DIR="optuna_launcher_mod"
                        PROCESS_PATTERN="optuna_serach_mod.py"
                        STUDY_PREFIX="aegnn_mod"
                    fi
                fi
            fi
            
            # Update study name based on detected version mode
            if [ "$VERSION_MODE" = "mod_new" ] && [ "$DB_PATH" = "optuna_edmpnn_results/optuna_mod_new.db" ]; then
                ACTUAL_STUDY_NAME="aegnn_mod_${ACTUAL_DATASET}_opt"  # optuna_parallel_mod_new.sh
            else
                ACTUAL_STUDY_NAME="${STUDY_PREFIX}_${ACTUAL_DATASET}_opt"
            fi
            OPTUNA_PIDS=$(pgrep -f "$PROCESS_PATTERN.*$ACTUAL_DATASET" 2>/dev/null | wc -l)
            # Try to get n_trials for the detected dataset
            if [ "$OPTUNA_PIDS" -gt 0 ]; then
                DETECTED_WORKER_TRIALS=$(pgrep -af "$PROCESS_PATTERN.*$ACTUAL_DATASET" 2>/dev/null | grep -o -- "--n_trials [0-9]*" | head -1 | awk '{print $2}')
            fi
            
            # Show auto-detection message (only once)
            if [ "$ACTUAL_DATASET" != "$DATASET" ] && [ "$AUTO_DETECTED_SHOWN" = false ]; then
                echo "💡 Auto-detected: Currently running dataset : '$ACTUAL_DATASET'"
                # Update DATASET and STUDY_NAME to match actual running dataset for consistency
                DATASET="$ACTUAL_DATASET"
                STUDY_NAME="$ACTUAL_STUDY_NAME"  # Update STUDY_NAME to match ACTUAL_STUDY_NAME
                AUTO_DETECTED_SHOWN=true
            elif [ "$ACTUAL_DATASET" != "$DATASET" ]; then
                # Update DATASET and STUDY_NAME silently if already shown
                DATASET="$ACTUAL_DATASET"
                STUDY_NAME="$ACTUAL_STUDY_NAME"  # Update STUDY_NAME to match ACTUAL_STUDY_NAME
            fi
        fi
    fi
    
    # If command line argument not provided, try to estimate total from worker trials
    # Parallel Seeds: WORKERS=1 per seed (one trial at a time per seed), all seeds run in parallel
    if [ -z "$PLANNED_TRIALS" ] && [ -n "$DETECTED_WORKER_TRIALS" ] && [ "$OPTUNA_PIDS" -gt 0 ]; then
        # Parallel Seeds: With WORKERS=1 per seed, worker_trials should equal total trials per seed
        # Since all 5 seeds run in parallel, we need to account for that
        # OPTUNA_PIDS represents the number of running processes (could be up to 5, one per seed)
        ESTIMATED_TOTAL=$((DETECTED_WORKER_TRIALS * OPTUNA_PIDS))
        PLANNED_TRIALS="$ESTIMATED_TOTAL"
    fi
    
    echo "🖥️  Active Workers: $OPTUNA_PIDS (dataset: $ACTUAL_DATASET)"
    
    # Determine study prefix based on current version mode (may have been updated during auto-detection)
    # optuna_parallel_mod_new.sh uses aegnn_mod prefix, not edmpnn_mod_new
    if [ "$VERSION_MODE" = "mod_new" ]; then
        if [ "$DB_PATH" = "optuna_edmpnn_results/optuna_mod_new.db" ]; then
            STUDY_PREFIX="aegnn_mod"  # optuna_parallel_mod_new.sh uses this
        else
            STUDY_PREFIX="edmpnn_mod_new"  # Legacy mod_new uses this
        fi
    else
        STUDY_PREFIX="aegnn_mod"
    fi
    
    # Check if per-seed optimization is being used (check for seed-specific studies)
    # optuna_parallel_mod_new.sh uses Fusion Model Logic (single study, not per-seed)
    # First try single study (Fusion Model Logic), then check for per-seed studies
    PER_SEED_OPTIMIZATION=false
    SINGLE_STUDY_EXISTS=false
    # Try to load single study first (used by optuna_parallel_mod_new.sh)
    if python3 -c "import optuna; optuna.load_study(study_name='${ACTUAL_STUDY_NAME}', storage='$STORAGE_URL')" 2>/dev/null; then
        SINGLE_STUDY_EXISTS=true
        PER_SEED_OPTIMIZATION=false  # Fusion Model Logic uses single study
    else
        # Check for per-seed studies (legacy per-seed optimization)
        if python3 -c "import optuna; optuna.load_study(study_name='${STUDY_PREFIX}_${ACTUAL_DATASET}_seed1_opt', storage='$STORAGE_URL')" 2>/dev/null; then
            PER_SEED_OPTIMIZATION=true
        fi
    fi
    
    # Get comprehensive study status (for per-seed or legacy mode)
    python3 << EOF
import optuna
import sys
import os
import json
import glob
from datetime import datetime, timedelta

# Configuration from bash
CHECKPOINT_DIR = "$CHECKPOINT_DIR"
STUDY_PREFIX = "$STUDY_PREFIX"
ACTUAL_DATASET = "$ACTUAL_DATASET"
STORAGE_URL = "$STORAGE_URL"

# ANSI color codes
class Colors:
    GREEN = '\033[92m'      # Bright green for COMPLETE
    YELLOW = '\033[93m'     # Bright yellow for RUNNING
    RED = '\033[91m'        # Bright red for FAIL
    BLUE = '\033[94m'       # Bright blue for PRUNED
    CYAN = '\033[96m'       # Bright cyan for other states
    RESET = '\033[0m'       # Reset color
    BOLD = '\033[1m'        # Bold text

def get_trial_progress(dataset, trial_number, checkpoint_dir, study_prefix, seed_num=None):
    """Get current epoch and loss from training_progress.json
    
    Args:
        dataset: Dataset name
        trial_number: Trial number
        checkpoint_dir: Checkpoint directory
        study_prefix: Study prefix
        seed_num: Seed number (for per-seed optimization). If None, checks all seeds (legacy mode)
    """
    if seed_num is not None:
        # NEW structure (after path restructuring): checkpoints/optuna_mod_new/{dataset}/opt/{trial_number}/seed{seed_num}/training_progress.json
        # Try new structure first (no seed layer before opt/)
        progress_file = f"checkpoints/{checkpoint_dir}/{dataset}/opt/{trial_number}/seed{seed_num}/training_progress.json"
        
        # Fallback to old Fusion Model Logic structure: seed1/opt/{trial_number}/seed{seed_num}/training_progress.json
        if not os.path.exists(progress_file):
            progress_file = f"checkpoints/{checkpoint_dir}/{dataset}/seed1/opt/{trial_number}/seed{seed_num}/training_progress.json"
        # Fallback to per-seed optimization structure: seed{seed_num}/opt/{trial_number}/training_progress.json
        if not os.path.exists(progress_file):
            progress_file = f"checkpoints/{checkpoint_dir}/{dataset}/seed{seed_num}/opt/{trial_number}/training_progress.json"
        # Fallback to per-seed optimization with seed subdirectory: seed{seed_num}/opt/{trial_number}/seed{seed_num}/
        if not os.path.exists(progress_file):
            progress_file = f"checkpoints/{checkpoint_dir}/{dataset}/seed{seed_num}/opt/{trial_number}/seed{seed_num}/training_progress.json"
        
        if os.path.exists(progress_file):
            try:
                with open(progress_file, 'r') as f:
                    progress = json.load(f)
                    return {
                        'epoch': progress.get('epoch', 0),
                        'val_loss': progress.get('val_loss'),
                        'train_loss': progress.get('train_loss'),
                        'timestamp': progress.get('timestamp'),
                        'seeds': {seed_num: {
                            'epoch': progress.get('epoch', 0),
                            'val_loss': progress.get('val_loss'),
                            'train_loss': progress.get('train_loss'),
                            'timestamp': progress.get('timestamp')
                        }},
                        'seeds_completed': 1
                    }
            except Exception as e:
                return None
        return None
    else:
        # Legacy mode: check all 5 seeds
        seed_list = [1, 2, 3, 4, 5]
        all_seeds_progress = {}
        overall_epoch = 0
        overall_val_loss = None
        latest_timestamp = None
        
        for seed in seed_list:
            # NEW structure (after path restructuring): checkpoints/optuna_mod_new/{dataset}/opt/{trial_number}/seed{seed}/training_progress.json
            # Try new structure first (no seed layer before opt/)
            progress_file = f"checkpoints/{checkpoint_dir}/{dataset}/opt/{trial_number}/seed{seed}/training_progress.json"
            # Fallback to old Fusion Model Logic structure: seed1/opt/{trial_number}/seed{seed}/training_progress.json
            if not os.path.exists(progress_file):
                progress_file = f"checkpoints/{checkpoint_dir}/{dataset}/seed1/opt/{trial_number}/seed{seed}/training_progress.json"
            # Fallback to per-seed optimization structure: seed{seed}/opt/{trial_number}/seed{seed}/
            if not os.path.exists(progress_file):
                progress_file = f"checkpoints/{checkpoint_dir}/{dataset}/seed{seed}/opt/{trial_number}/seed{seed}/training_progress.json"
            # Fallback to per-seed optimization structure without seed subdirectory: seed{seed}/opt/{trial_number}/
            if not os.path.exists(progress_file):
                progress_file = f"checkpoints/{checkpoint_dir}/{dataset}/seed{seed}/opt/{trial_number}/training_progress.json"
            # Fallback to old legacy structure for backward compatibility
            if not os.path.exists(progress_file):
                progress_file = f"checkpoints/{checkpoint_dir}/{dataset}/{study_prefix}_{dataset}_opt_{trial_number}/seed{seed}/training_progress.json"
            
            if os.path.exists(progress_file):
                try:
                    with open(progress_file, 'r') as f:
                        progress = json.load(f)
                        seed_epoch = progress.get('epoch', 0)
                        seed_val_loss = progress.get('val_loss')
                        seed_timestamp = progress.get('timestamp')
                        
                        all_seeds_progress[seed] = {
                            'epoch': seed_epoch,
                            'val_loss': seed_val_loss,
                            'train_loss': progress.get('train_loss'),
                            'timestamp': seed_timestamp
                        }
                        
                        # Track overall progress (use minimum epoch as overall, or average)
                        if seed_epoch > overall_epoch:
                            overall_epoch = seed_epoch
                        
                        # Use average validation loss across all seeds
                        if seed_val_loss is not None:
                            if overall_val_loss is None:
                                overall_val_loss = []
                            overall_val_loss.append(seed_val_loss)
                        
                        # Track latest timestamp
                        if seed_timestamp and (latest_timestamp is None or seed_timestamp > latest_timestamp):
                            latest_timestamp = seed_timestamp
                except Exception as e:
                    all_seeds_progress[seed] = {'error': str(e)}
    
    if not all_seeds_progress:
        return None
    
    # Calculate average validation loss
    if overall_val_loss:
        avg_val_loss = sum(overall_val_loss) / len(overall_val_loss)
    else:
        avg_val_loss = None
    
    return {
        'epoch': overall_epoch,
        'val_loss': avg_val_loss,
        'train_loss': None,  # Could calculate average if needed
        'timestamp': latest_timestamp,
        'seeds': all_seeds_progress,  # Include individual seed progress
        'seeds_completed': len([s for s in all_seeds_progress.values() if 'error' not in s])
    }

def check_trial_integrity(dataset, trial_number, checkpoint_dir, study_prefix, seed_num=None):
    """Check data integrity for a trial
    
    Args:
        dataset: Dataset name
        trial_number: Trial number
        checkpoint_dir: Checkpoint directory
        study_prefix: Study prefix
        seed_num: Seed number (for per-seed optimization). If None, checks all seeds (legacy mode)
    """
    if seed_num is not None:
        # Per-seed optimization: check single seed
        seed_list = [seed_num]
    else:
        # Legacy/Fusion Model Logic mode: check all 5 seeds
        seed_list = [1, 2, 3, 4, 5]
    
    issues = []
    
    for seed in seed_list:
        # NEW structure (after path restructuring): checkpoints/optuna_mod_new/{dataset}/opt/{trial_number}/seed{seed}/
        # Try new structure first (no seed layer before opt/)
        trial_dir = f"checkpoints/{checkpoint_dir}/{dataset}/opt/{trial_number}/seed{seed}"
        # Fallback to old Fusion Model Logic structure: seed1/opt/{trial_number}/seed{seed}/
        if not os.path.exists(trial_dir):
            trial_dir = f"checkpoints/{checkpoint_dir}/{dataset}/seed1/opt/{trial_number}/seed{seed}"
        # Fallback to per-seed optimization structure: seed{seed}/opt/{trial_number}/seed{seed}/
        if not os.path.exists(trial_dir):
            trial_dir = f"checkpoints/{checkpoint_dir}/{dataset}/seed{seed}/opt/{trial_number}/seed{seed}"
        # Fallback to per-seed optimization structure without seed subdirectory: seed{seed}/opt/{trial_number}/
        if not os.path.exists(trial_dir):
            trial_dir = f"checkpoints/{checkpoint_dir}/{dataset}/seed{seed}/opt/{trial_number}"
        # Fallback to old legacy structure for backward compatibility
        if not os.path.exists(trial_dir):
            trial_dir = f"checkpoints/{checkpoint_dir}/{dataset}/{study_prefix}_{dataset}_opt_{trial_number}/seed{seed}"
        progress_file = os.path.join(trial_dir, "training_progress.json")
        
        # Check training_progress.json
        if os.path.exists(progress_file):
            try:
                with open(progress_file, 'r') as f:
                    json.load(f)
            except:
                issues.append(f"Seed {seed}: Corrupted training_progress.json")
        else:
            issues.append(f"Seed {seed}: Missing training_progress.json")
        
        # Check checkpoint file
        checkpoint_file = os.path.join(trial_dir, "best_model.pth")
        # Note: checkpoint may not exist if trial is still running, so we don't report this as an issue
        
        # Check if progress file is stale (not updated in last hour for running trials)
        if os.path.exists(progress_file):
            try:
                with open(progress_file, 'r') as f:
                    progress = json.load(f)
                    timestamp = progress.get('timestamp')
                    if timestamp:
                        last_update = datetime.fromtimestamp(timestamp)
                        if (datetime.now() - last_update).total_seconds() > 3600:
                            issues.append(f"Seed {seed}: Progress file not updated in >1 hour")
            except:
                pass
    
    return issues

# Wrap main code in try-except for error handling
try:
    # Check if per-seed optimization is being used
    # optuna_parallel_mod_new.sh uses Fusion Model Logic (single study, not per-seed)
    PER_SEED_OPTIMIZATION = False
    seed_studies = {}
    single_study = None
    
    # First try to load single study (Fusion Model Logic - used by optuna_parallel_mod_new.sh)
    try:
        single_study = optuna.load_study(study_name=ACTUAL_STUDY_NAME, storage=STORAGE_URL)
        PER_SEED_OPTIMIZATION = False  # Fusion Model Logic uses single study
    except Exception:
        # Single study not found, try per-seed studies (legacy per-seed optimization)
        for seed_num in range(1, 6):
            seed_study_name = f"{STUDY_PREFIX}_{ACTUAL_DATASET}_seed{seed_num}_opt"
            try:
                seed_study = optuna.load_study(study_name=seed_study_name, storage=STORAGE_URL)
                seed_studies[seed_num] = seed_study
                PER_SEED_OPTIMIZATION = True
            except Exception:
                pass
    
    if PER_SEED_OPTIMIZATION:
        # Per-seed optimization mode: monitor all 5 seeds
        print(f"🌱 Per-Seed Optimization Mode: Monitoring 5 independent studies")
        print("")
    elif single_study is not None:
        # Fusion Model Logic: single study for all seeds
        print(f"🌱 Fusion Model Logic Mode: Single study for all seeds (shared hyperparameters)")
        print("")
        
        all_seed_stats = {}
        total_complete = 0
        total_running = 0
        total_pruned = 0
        total_failed = 0
        total_trials = 0
        
        for seed_num in range(1, 6):
            if seed_num in seed_studies:
                study = seed_studies[seed_num]
                trials = study.trials
                complete = len([t for t in trials if t.state == optuna.trial.TrialState.COMPLETE])
                running = len([t for t in trials if t.state == optuna.trial.TrialState.RUNNING])
                pruned = len([t for t in trials if t.state == optuna.trial.TrialState.PRUNED])
                failed = len([t for t in trials if t.state == optuna.trial.TrialState.FAIL])
                
                all_seed_stats[seed_num] = {
                    'study': study,
                    'trials': trials,
                    'complete': complete,
                    'running': running,
                    'pruned': pruned,
                    'failed': failed,
                    'total': len(trials),
                    'best_value': study.best_trial.value if complete > 0 else None,
                    'best_trial': study.best_trial.number if complete > 0 else None
                }
                
                total_complete += complete
                total_running += running
                total_pruned += pruned
                total_failed += failed
                total_trials += len(trials)
            else:
                all_seed_stats[seed_num] = {
                    'study': None,
                    'trials': [],
                    'complete': 0,
                    'running': 0,
                    'pruned': 0,
                    'failed': 0,
                    'total': 0,
                    'best_value': None,
                    'best_trial': None
                }
        
        # Display summary across all seeds
        complete_color = f"{Colors.GREEN}{total_complete}{Colors.RESET}"
        running_color = f"{Colors.YELLOW}{total_running}{Colors.RESET}"
        pruned_color = f"{Colors.BLUE}{total_pruned}{Colors.RESET}"
        fail_color = f"{Colors.RED}{total_failed}{Colors.RESET}"
        
        print(f"📋 Overall Status (All Seeds): Total: {total_trials} | ✅ Complete: {complete_color} | 🔄 Running: {running_color} | ✂️ Pruned: {pruned_color} | ❌ Failed: {fail_color}")
        print("")
        
        # Display per-seed status
        print(f"📊 Per-Seed Status:")
        for seed_num in sorted(all_seed_stats.keys()):
            stats = all_seed_stats[seed_num]
            if stats['study'] is not None:
                seed_complete_color = f"{Colors.GREEN}{stats['complete']}{Colors.RESET}"
                seed_running_color = f"{Colors.YELLOW}{stats['running']}{Colors.RESET}"
                seed_pruned_color = f"{Colors.BLUE}{stats['pruned']}{Colors.RESET}"
                seed_fail_color = f"{Colors.RED}{stats['failed']}{Colors.RESET}"
                
                best_info = ""
                if stats['best_value'] is not None:
                    best_info = f" | Best: Trial #{stats['best_trial']} ({stats['best_value']:.6f})"
                
                print(f"   Seed {seed_num}: Total: {stats['total']} | ✅ {seed_complete_color} | 🔄 {seed_running_color} | ✂️ {seed_pruned_color} | ❌ {seed_fail_color}{best_info}")
            else:
                print(f"   Seed {seed_num}: {Colors.CYAN}No study found{Colors.RESET}")
        print("")
        
        # Get primary metric for best result display
        primary_metric = None
        try:
            import yaml
            config_path = "configs/dataset_primary_metrics.yaml"
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                dataset_configs = config.get('dataset_primary_metrics', {})
                dataset_name = ACTUAL_DATASET
                dataset_config = dataset_configs.get(dataset_name.lower(), {})
                primary_metric = dataset_config.get('primary_metric', None)
        except Exception:
            pass
        
        # Display best results for each seed (removed per user request)
        # if total_complete > 0:
        #     print(f"🏆 Best Results (Per Seed):")
        #     for seed_num in sorted(all_seed_stats.keys()):
        #         stats = all_seed_stats[seed_num]
        #         if stats['best_value'] is not None:
        #             metric_desc = ""
        #             if primary_metric:
        #                 if primary_metric in ['roc_auc', 'pr_auc']:
        #                     metric_desc = f" ({primary_metric.upper()})"
        #                 else:
        #                     metric_desc = f" ({primary_metric.upper()})"
        #             print(f"   {Colors.GREEN}Seed {seed_num}: Trial #{stats['best_trial']} - {stats['best_value']:.6f}{metric_desc}{Colors.RESET}")
        #     print("")
        
        # Display running trials for each seed
        if total_running > 0:
            print(f"⏱️  Running Trials (Per Seed):")
            for seed_num in sorted(all_seed_stats.keys()):
                stats = all_seed_stats[seed_num]
                if stats['running'] > 0:
                    running_trials = [t for t in stats['trials'] if t.state == optuna.trial.TrialState.RUNNING]
                    # Sort trials by number (ascending)
                    running_trials_sorted = sorted(running_trials, key=lambda t: t.number)
                    
                    # Try to find a trial with progress first
                    trial_to_show = None
                    progress_to_show = None
                    
                    # First pass: find trial with progress (prefer newer trials)
                    for trial in reversed(running_trials_sorted):
                        progress = get_trial_progress(ACTUAL_DATASET, trial.number, CHECKPOINT_DIR, STUDY_PREFIX, seed_num=seed_num)
                        if progress and 'epoch' in progress:
                            trial_to_show = trial
                            progress_to_show = progress
                            break
                    
                    # If no trial with progress found, use the first running trial
                    if trial_to_show is None and running_trials_sorted:
                        trial_to_show = running_trials_sorted[0]
                    
                    # Display the selected trial
                    if trial_to_show:
                        elapsed_str = "N/A"
                        if trial_to_show.datetime_start:
                            elapsed = (datetime.now() - trial_to_show.datetime_start).total_seconds()
                            elapsed_min = int(elapsed / 60)
                            elapsed_sec = int(elapsed % 60)
                            elapsed_str = f"{elapsed_min}m {elapsed_sec}s"
                        
                        # Get progress if not already retrieved
                        if progress_to_show is None:
                            progress_to_show = get_trial_progress(ACTUAL_DATASET, trial_to_show.number, CHECKPOINT_DIR, STUDY_PREFIX, seed_num=seed_num)
                        
                        # Show trial count if multiple running trials
                        trial_count_str = f"{stats['running']} running"
                        if stats['running'] > 1:
                            trial_count_str += f" (showing Trial #{trial_to_show.number})"
                        
                        if progress_to_show and 'epoch' in progress_to_show:
                            epoch = progress_to_show['epoch']
                            val_loss = progress_to_show.get('val_loss')
                            if val_loss is not None:
                                print(f"   Seed {seed_num}: {trial_count_str} - {Colors.YELLOW}Trial #{trial_to_show.number}: {elapsed_str} - Epoch {epoch}, Val Loss: {val_loss:.4f}{Colors.RESET}")
                            else:
                                print(f"   Seed {seed_num}: {trial_count_str} - {Colors.YELLOW}Trial #{trial_to_show.number}: {elapsed_str} - Epoch {epoch}{Colors.RESET}")
                        else:
                            print(f"   Seed {seed_num}: {trial_count_str} - {Colors.YELLOW}Trial #{trial_to_show.number}: {elapsed_str} - No progress files{Colors.RESET}")
            print("")
        
        # === TIME-RELATED INFORMATION for Per-Seed Mode ===
        if PER_SEED_OPTIMIZATION:
            # Find earliest trial start time across all seeds for total runtime
            earliest_start = None
            latest_complete = None
            has_running = False
            
            for seed_num, stats in all_seed_stats.items():
                if stats['study']:
                    trials = stats['trials']
                    if trials:
                        # Find earliest start time
                        for trial in trials:
                            if trial.datetime_start:
                                if earliest_start is None or trial.datetime_start < earliest_start:
                                    earliest_start = trial.datetime_start
                        
                        # Find latest complete time
                        complete_trials = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE and t.datetime_complete]
                        if complete_trials:
                            latest_complete_trial = max(complete_trials, key=lambda t: t.datetime_complete)
                            if latest_complete is None or latest_complete_trial.datetime_complete > latest_complete:
                                latest_complete = latest_complete_trial.datetime_complete
                        
                        # Check if any trial is running
                        if stats['running'] > 0:
                            has_running = True
            # Calculate average trial completion time
            all_completed_with_time = []
            for seed_num, stats in all_seed_stats.items():
                if stats['study']:
                    seed_complete = [t for t in stats['trials'] if t.state == optuna.trial.TrialState.COMPLETE and t.datetime_start and t.datetime_complete]
                    all_completed_with_time.extend(seed_complete)
            
            # Display Time Information if we have any time data (completed trials or running trials)
            if all_completed_with_time or earliest_start:
                print(f"⏰ Time Information:")
                
                # Display total runtime (if we have earliest_start)
                if earliest_start:
                    # Calculate total runtime
                    if has_running:
                        # If there are running trials, use current time
                        end_time = datetime.now()
                    elif latest_complete:
                        # If no running trials, use latest complete time
                        end_time = latest_complete
                    else:
                        # No completed trials yet, use current time
                        end_time = datetime.now()
                    
                    total_runtime = (end_time - earliest_start).total_seconds()
                    total_hours = int(total_runtime / 3600)
                    total_minutes = int((total_runtime % 3600) / 60)
                    total_sec = int(total_runtime % 60)
                    
                    # Total Runtime and Avg Trial Time will be displayed after Recent Activity
                    # Keep calculation for estimated completion time
                    status_str = " (ongoing)" if has_running else " (completed)"
                
                # Calculate average trial time (if we have completed trials) - keep for estimated completion time
                avg_time = None
                if all_completed_with_time:
                    avg_time = sum([(t.datetime_complete - t.datetime_start).total_seconds() 
                                   for t in all_completed_with_time]) / len(all_completed_with_time)
                
                # Calculate remaining trials and estimated completion time
                any_seed_running = False
                for seed_num, stats in all_seed_stats.items():
                    if stats['study'] and stats['running'] > 0:
                        any_seed_running = True
                        break
                
                if any_seed_running and avg_time is not None:
                    # Calculate remaining trials
                    # IMPORTANT: Get planned_total from per-seed studies
                    per_seed_planned_total = None
                    # Try to get planned total from command line first
                    cmd_line_n_trials = "$PLANNED_TRIALS"
                    if cmd_line_n_trials and cmd_line_n_trials.strip():
                        try:
                            per_seed_planned_total = int(cmd_line_n_trials)
                        except ValueError:
                            pass
                    
                    # If not from command line, try to get from first available seed's study
                    if per_seed_planned_total is None:
                        for seed_num in sorted(all_seed_stats.keys()):
                            if all_seed_stats[seed_num]['study']:
                                try:
                                    study = all_seed_stats[seed_num]['study']
                                    if hasattr(study, 'user_attrs') and 'n_trials' in study.user_attrs:
                                        per_seed_planned_total = study.user_attrs['n_trials']
                                        # Validate: per_seed_planned_total should be a reasonable value (typically 50)
                                        if per_seed_planned_total is not None and per_seed_planned_total > 0 and per_seed_planned_total <= 1000:
                                            break
                                        else:
                                            # Invalid value, reset to None and try next seed
                                            per_seed_planned_total = None
                                except Exception:
                                    pass
                    
                print("")
        
        # Use first seed's study for time estimation (if available)
        study = None
        trials = []
        if 1 in seed_studies:
            study = seed_studies[1]
            trials = study.trials
        elif len(seed_studies) > 0:
            # Use first available seed
            first_seed = min(seed_studies.keys())
            study = seed_studies[first_seed]
            trials = study.trials
    else:
        # Fusion Model Logic or Legacy mode: single study for all seeds
        if single_study is not None:
            # Use the single study we already loaded (Fusion Model Logic)
            study = single_study
            trials = study.trials
            total_created_trials = len(trials)
        else:
            # Try legacy study name format
            try:
                legacy_study_name = f"{STUDY_PREFIX}_{ACTUAL_DATASET}_opt"
                study = optuna.load_study(study_name=legacy_study_name, storage=STORAGE_URL)
                trials = study.trials
                total_created_trials = len(trials)
            except Exception:
                # Study not found, skip legacy mode
                study = None
                trials = []
                total_created_trials = 0
        
        # Try to get planned total trials from study attributes or command line
        planned_total = None
        
        # Only proceed if study was successfully loaded
        if study is not None:
            # First, try command line argument (passed from bash)
            cmd_line_n_trials = "$PLANNED_TRIALS"
            if cmd_line_n_trials and cmd_line_n_trials.strip():
                try:
                    planned_total = int(cmd_line_n_trials)
                    # Save it to study for future use
                    try:
                        if 'n_trials' not in study.user_attrs:
                            study.set_user_attr('n_trials', planned_total)
                    except Exception:
                        pass
                except ValueError:
                    pass
            
            # Otherwise, try to get from study attributes
            if planned_total is None:
                try:
                    # Use user_attrs (system_attrs is deprecated in Optuna 3.1.0+)
                    if hasattr(study, 'user_attrs'):
                        user_attrs = study.user_attrs
                        if 'n_trials' in user_attrs:
                            planned_total = user_attrs['n_trials']
                except Exception:
                    pass
            
            complete_trials = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
            running_trials = [t for t in trials if t.state == optuna.trial.TrialState.RUNNING]
            pruned_trials = [t for t in trials if t.state == optuna.trial.TrialState.PRUNED]
            fail_trials = [t for t in trials if t.state == optuna.trial.TrialState.FAIL]
            waiting_trials = [t for t in trials if t.state == optuna.trial.TrialState.WAITING]
            
            complete = len(complete_trials)
            running = len(running_trials)
            pruned = len(pruned_trials)
            fail = len(fail_trials)
            waiting = len(waiting_trials)
        else:
            # Study not found, set defaults
            complete_trials = []
            running_trials = []
            pruned_trials = []
            fail_trials = []
            waiting_trials = []
            complete = 0
            running = 0
            pruned = 0
            fail = 0
            waiting = 0
        
        # Color-coded study status
        complete_color = f"{Colors.GREEN}{complete}{Colors.RESET}"
        running_color = f"{Colors.YELLOW}{running}{Colors.RESET}"
        pruned_color = f"{Colors.BLUE}{pruned}{Colors.RESET}"
        fail_color = f"{Colors.RED}{fail}{Colors.RESET}"
        
        # Display total trials (created vs planned)
        if planned_total and planned_total > total_created_trials:
            total_display = f"{total_created_trials}/{planned_total}"
        else:
            total_display = str(total_created_trials)
        
        print(f"📋 Study Status: Total: {total_display} | ✅ Complete: {complete_color} | 🔄 Running: {running_color} | ✂️ Pruned: {pruned_color} | ❌ Failed: {fail_color}")
        print("")
        
        # === TIME-RELATED INFORMATION (Feature 5) ===
        if complete > 0 or running > 0:
            # Find earliest trial start time for total runtime
            earliest_start = None
            latest_complete = None
            has_running = running > 0
            
            if trials:
                for trial in trials:
                    if trial.datetime_start:
                        if earliest_start is None or trial.datetime_start < earliest_start:
                            earliest_start = trial.datetime_start
                
                # Find latest complete time
                complete_with_time = [t for t in complete_trials if t.datetime_complete]
                if complete_with_time:
                    latest_complete = max(complete_with_time, key=lambda t: t.datetime_complete).datetime_complete
            
            # Display Time Information
            if earliest_start or complete > 0:
                print(f"⏰ Time Information:")
                
                # Display total runtime
                if earliest_start:
                    # Calculate total runtime
                    if has_running:
                        # If there are running trials, use current time
                        end_time = datetime.now()
                    elif latest_complete:
                        # If no running trials, use latest complete time
                        end_time = latest_complete
                    else:
                        # No completed trials yet, use current time
                        end_time = datetime.now()
                    
                    total_runtime = (end_time - earliest_start).total_seconds()
                    total_hours = int(total_runtime / 3600)
                    total_minutes = int((total_runtime % 3600) / 60)
                    total_sec = int(total_runtime % 60)
                    
                    # Total Runtime and Avg Trial Time will be displayed after Recent Activity
                    status_str = " (ongoing)" if has_running else " (completed)"
            
            # Calculate average trial completion time - keep for estimated completion time
            if PER_SEED_OPTIMIZATION:
                # Per-seed mode: each trial trains only one seed
                all_completed_with_time = []
                for seed_num, stats in all_seed_stats.items():
                    if stats['study']:
                        seed_complete = [t for t in stats['trials'] if t.state == optuna.trial.TrialState.COMPLETE and t.datetime_start and t.datetime_complete]
                        all_completed_with_time.extend(seed_complete)
                
                if all_completed_with_time:
                    avg_time = sum([(t.datetime_complete - t.datetime_start).total_seconds() 
                                   for t in all_completed_with_time]) / len(all_completed_with_time)
            else:
                # Legacy mode: each trial processes 5 seeds
                completed_with_time = [t for t in complete_trials if t.datetime_start and t.datetime_complete]
                if completed_with_time:
                    avg_time = sum([(t.datetime_complete - t.datetime_start).total_seconds() 
                                   for t in completed_with_time]) / len(completed_with_time)
            
            # Estimated completion time
            # For per-seed mode, check if any seed has running trials
            if PER_SEED_OPTIMIZATION:
                any_seed_running = False
                for seed_num, stats in all_seed_stats.items():
                    if stats['study'] and stats['running'] > 0:
                        any_seed_running = True
                        break
                running = 1 if any_seed_running else 0
                # Use all_completed_with_time from per-seed mode (already calculated above)
                # avg_time should also be available from above
                completed_with_time = all_completed_with_time if 'all_completed_with_time' in locals() and all_completed_with_time else []
            else:
                # Legacy mode: calculate completed_with_time
                completed_with_time = [t for t in complete_trials if t.datetime_start and t.datetime_complete]
            
            print("")
        
        # Best result (green color for completed best trial)
        if PER_SEED_OPTIMIZATION:
            # Already displayed above in per-seed section
            pass
        elif study and complete > 0:
            # Get primary metric information
            primary_metric = None
            try:
                import yaml
                config_path = "configs/dataset_primary_metrics.yaml"
                if os.path.exists(config_path):
                    with open(config_path, 'r') as f:
                        config = yaml.safe_load(f)
                    dataset_configs = config.get('dataset_primary_metrics', {})
                    dataset_name = ACTUAL_DATASET
                    dataset_config = dataset_configs.get(dataset_name.lower(), {})
                    primary_metric = dataset_config.get('primary_metric', None)
            except Exception:
                pass
            
            # Determine metric description
            # Note: Now using validation metrics (not test set) and primary metric only (no combined score)
            metric_desc = ""
            if primary_metric:
                metric_desc = f" (validation {primary_metric.upper()})"
            else:
                metric_desc = " (validation optimization score)"
            
            print(f"🏆 Best Result:")
            print(f"   {Colors.GREEN}Trial #{study.best_trial.number}: {study.best_value:.6f}{metric_desc}{Colors.RESET}")
            
            # Show individual seed scores if available (legacy mode only)
            best_valid_scores = study.best_trial.user_attrs.get("valid_scores", [])
            if best_valid_scores:
                print(f"   Individual seed scores: {best_valid_scores}")
            print("")
        
        # === RUNNING TRIALS WITH ENHANCED INFO ===
        if PER_SEED_OPTIMIZATION:
            # Running trials already displayed above in per-seed section
            pass
        elif study and running > 0:
            print(f"⏱️  Running Trials:")
            warnings = []
            stuck_trials = []
            no_progress_trials = []
            
            # Calculate adaptive threshold for stuck trials
            # Parallel Seeds: Single GPU mode per seed (no DDP), may be slower than DDP, so use longer threshold
            # Use 2x average time if available, otherwise fall back to longer time
            completed_with_time = [t for t in complete_trials if t.datetime_start and t.datetime_complete]
            if completed_with_time and len(completed_with_time) >= 3:
                avg_time = sum([(t.datetime_complete - t.datetime_start).total_seconds() 
                               for t in completed_with_time]) / len(completed_with_time)
                stuck_threshold = avg_time * 2.0  # 2x average time
            else:
                # Parallel Seeds: Single GPU mode per seed may be slower, use 12 hours as fallback
                stuck_threshold = 43200  # 12 hours as fallback (Parallel Seeds: single GPU per seed, all seeds in parallel)
            
            for trial in sorted(running_trials, key=lambda t: t.number):
                elapsed_str = "N/A"
                if trial.datetime_start:
                    elapsed = (datetime.now() - trial.datetime_start).total_seconds()
                    elapsed_min = int(elapsed / 60)
                    elapsed_sec = int(elapsed % 60)
                    elapsed_str = f"{elapsed_min}m {elapsed_sec}s"
                    
                    # Check for stuck trials (adaptive threshold)
                    if elapsed > stuck_threshold:
                        stuck_trials.append(trial.number)
                
                # Get progress info
                # For per-seed optimization, we need to determine which seed this trial belongs to
                # Check trial's user_attrs for optimized_seed
                trial_seed = trial.user_attrs.get("optimized_seed", None)
                if trial_seed:
                    progress = get_trial_progress(ACTUAL_DATASET, trial.number, CHECKPOINT_DIR, STUDY_PREFIX, seed_num=trial_seed)
                else:
                    # Legacy mode: check all seeds
                    progress = get_trial_progress(ACTUAL_DATASET, trial.number, CHECKPOINT_DIR, STUDY_PREFIX, seed_num=None)
                
                if progress and 'epoch' in progress:
                    epoch = progress['epoch']
                    val_loss = progress.get('val_loss')
                    train_loss = progress.get('train_loss')
                    seeds_info = progress.get('seeds', {})
                    seeds_completed = progress.get('seeds_completed', 0)
                    
                    # Check for no progress (epoch not updated in last 30 min)
                    if progress.get('timestamp'):
                        last_update = datetime.fromtimestamp(progress['timestamp'])
                        time_since_update = (datetime.now() - last_update).total_seconds()
                        if time_since_update > 1800:  # 30 minutes
                            no_progress_trials.append(trial.number)
                    
                    # Check for abnormal loss values
                    if val_loss is not None:
                        if val_loss > 1000:  # Unusually high loss
                            warnings.append(f"Trial #{trial.number}: Abnormal avg_val_loss={val_loss:.2f}")
                        if val_loss < 0:  # Negative loss (shouldn't happen)
                            warnings.append(f"Trial #{trial.number}: Negative avg_val_loss={val_loss:.2f}")
                    
                    # Display trial progress
                    if trial_seed:
                        # Per-seed optimization: show single seed info
                        if val_loss is not None:
                            progress_str = f"{Colors.YELLOW}Trial #{trial.number} (Seed {trial_seed}): {elapsed_str} - Epoch {epoch}, Val Loss: {val_loss:.4f}{Colors.RESET}"
                        else:
                            progress_str = f"{Colors.YELLOW}Trial #{trial.number} (Seed {trial_seed}): {elapsed_str} - Epoch {epoch}{Colors.RESET}"
                        print(progress_str)
                    else:
                        # Legacy mode: show all seeds info
                        seeds_status = f"Seeds: {seeds_completed}/5"
                        if val_loss is not None:
                            progress_str = f"{Colors.YELLOW}Trial #{trial.number}: {elapsed_str} - Epoch {epoch} (avg), {seeds_status}, Val Loss: {val_loss:.4f}{Colors.RESET}"
                        else:
                            progress_str = f"{Colors.YELLOW}Trial #{trial.number}: {elapsed_str} - Epoch {epoch} (avg), {seeds_status}{Colors.RESET}"
                        
                        print(progress_str)
                        if seeds_info:
                            # Show individual seeds that have progress
                            for seed_num in sorted(seeds_info.keys()):
                                seed_prog = seeds_info[seed_num]
                                if 'error' not in seed_prog:
                                    seed_epoch = seed_prog.get('epoch', 0)
                                    seed_val_loss = seed_prog.get('val_loss')
                                    if seed_val_loss is not None:
                                        print(f"      Seed {seed_num}: Epoch {seed_epoch}, Val Loss: {seed_val_loss:.4f}")
                                    else:
                                        print(f"      Seed {seed_num}: Epoch {seed_epoch}")
                                else:
                                    print(f"      Seed {seed_num}: Error - {seed_prog.get('error', 'Unknown')}")
                else:
                    print(f"   {Colors.YELLOW}Trial #{trial.number}: {elapsed_str} - Epoch N/A (no progress files found){Colors.RESET}")
            
            # Display warnings
            if stuck_trials:
                threshold_hours = stuck_threshold / 3600
                print("")
                print(f"⚠️  Warning: {len(stuck_trials)} trial(s) running > {threshold_hours:.1f} hours: {stuck_trials}")
            
            if no_progress_trials:
                print("")
                print(f"⚠️  Warning: {len(no_progress_trials)} trial(s) with no progress > 30 min: {no_progress_trials}")
            
            if warnings:
                print("")
                for warning in warnings:
                    print(f"   ⚠️  {warning}")
        else:
            print(f"{Colors.GREEN}✅ No trials currently running{Colors.RESET}")
            print("")
        
        # Display total runtime for legacy mode
        if not PER_SEED_OPTIMIZATION and study:
            # Find earliest trial start time
            earliest_start = None
            latest_complete = None
            has_running = len(running_trials) > 0
            
            if trials:
                for trial in trials:
                    if trial.datetime_start:
                        if earliest_start is None or trial.datetime_start < earliest_start:
                            earliest_start = trial.datetime_start
                
                # Find latest complete time
                complete_with_time = [t for t in complete_trials if t.datetime_complete]
                if complete_with_time:
                    latest_complete = max(complete_with_time, key=lambda t: t.datetime_complete).datetime_complete
            
            if earliest_start:
                # Calculate total runtime
                if has_running:
                    # If there are running trials, use current time
                    end_time = datetime.now()
                elif latest_complete:
                    # If no running trials, use latest complete time
                    end_time = latest_complete
                else:
                    # No completed trials yet, use current time
                    end_time = datetime.now()
                
                total_runtime = (end_time - earliest_start).total_seconds()
                total_hours = int(total_runtime / 3600)
                total_minutes = int((total_runtime % 3600) / 60)
                total_sec = int(total_runtime % 60)
                
                # Dataset Total Runtime will be displayed after Recent Activity
                # Store values for later display
                dataset_total_runtime_hours = total_hours
                dataset_total_runtime_minutes = total_minutes
                dataset_total_runtime_sec = total_sec
                dataset_total_runtime_status = " (ongoing)" if has_running else " (completed)"
        
        # === DATA INTEGRITY CHECKS (Feature 12) ===
        if "$DISPLAY_MODE" == "detailed" and running > 0:
            integrity_issues = []
            for trial in running_trials[:5]:  # Check first 5 running trials
                # Check if per-seed optimization
                trial_seed = trial.user_attrs.get("optimized_seed", None) if hasattr(trial, 'user_attrs') else None
                if trial_seed:
                    issues = check_trial_integrity(ACTUAL_DATASET, trial.number, CHECKPOINT_DIR, STUDY_PREFIX, seed_num=trial_seed)
                else:
                    issues = check_trial_integrity(ACTUAL_DATASET, trial.number, CHECKPOINT_DIR, STUDY_PREFIX, seed_num=None)
                if issues:
                    integrity_issues.append(f"Trial #{trial.number}: {', '.join(issues)}")
            
            if integrity_issues:
                print("🔍 Data Integrity Issues:")
                for issue in integrity_issues:
                    print(f"   ⚠️  {issue}")
                print("")
        
        # === RECENT ACTIVITY ===
        if total_created_trials > 0:
            recent = sorted(trials, key=lambda t: t.number, reverse=True)[:5]
            print("📝 Recent Activity:")
            for trial in recent:
                # Determine color based on state
                if trial.state == optuna.trial.TrialState.COMPLETE:
                    state_icon = "✅"
                    color = Colors.GREEN
                elif trial.state == optuna.trial.TrialState.RUNNING:
                    state_icon = "🔄"
                    color = Colors.YELLOW
                elif trial.state == optuna.trial.TrialState.PRUNED:
                    state_icon = "✂️"
                    color = Colors.BLUE
                else:  # FAIL
                    state_icon = "❌"
                    color = Colors.RED
                
                if trial.state == optuna.trial.TrialState.RUNNING:
                    if trial.datetime_start:
                        elapsed = (datetime.now() - trial.datetime_start).total_seconds()
                        elapsed_str = f"{int(elapsed/60)}m {int(elapsed%60)}s"
                    else:
                        elapsed_str = "N/A"
                    
                    # Check if per-seed optimization
                    trial_seed = trial.user_attrs.get("optimized_seed", None) if hasattr(trial, 'user_attrs') else None
                    if trial_seed:
                        progress = get_trial_progress(ACTUAL_DATASET, trial.number, CHECKPOINT_DIR, STUDY_PREFIX, seed_num=trial_seed)
                    else:
                        progress = get_trial_progress(ACTUAL_DATASET, trial.number, CHECKPOINT_DIR, STUDY_PREFIX, seed_num=None)
                    
                    if progress and 'epoch' in progress:
                        epoch = progress['epoch']
                        val_loss = progress.get('val_loss')
                        if trial_seed:
                            # Per-seed mode
                            if val_loss is not None:
                                value_str = f"Running ({elapsed_str}) - Seed {trial_seed}, Epoch {epoch}, Val Loss: {val_loss:.4f}"
                            else:
                                value_str = f"Running ({elapsed_str}) - Seed {trial_seed}, Epoch {epoch}"
                        else:
                            # Legacy mode
                            seeds_completed = progress.get('seeds_completed', 0)
                            if val_loss is not None:
                                value_str = f"Running ({elapsed_str}) - Epoch {epoch} (avg), Seeds: {seeds_completed}/5, Val Loss: {val_loss:.4f}"
                            else:
                                value_str = f"Running ({elapsed_str}) - Epoch {epoch} (avg), Seeds: {seeds_completed}/5"
                    else:
                        value_str = f"Running ({elapsed_str}) - No progress files"
                elif trial.value is not None:
                    value_str = f"{trial.value:.6f}"
                else:
                    value_str = "N/A"
                
                print(f"   {color}{state_icon} Trial #{trial.number}: {trial.state.name} - {value_str}{Colors.RESET}")
            print("")
            
            # === DATASET TOTAL RUNTIME AND TRIAL AVERAGE RUNTIME ===
            # Calculate and display dataset total runtime
            # Find earliest trial start time
            dataset_earliest_start = None
            dataset_latest_complete = None
            dataset_has_running = False
            
            if PER_SEED_OPTIMIZATION:
                # Per-seed mode: check all seed studies
                for seed_num, stats in all_seed_stats.items():
                    if stats['study']:
                        seed_trials = stats['trials']
                        if seed_trials:
                            for t in seed_trials:
                                if t.datetime_start:
                                    if dataset_earliest_start is None or t.datetime_start < dataset_earliest_start:
                                        dataset_earliest_start = t.datetime_start
                                if t.state == optuna.trial.TrialState.COMPLETE and t.datetime_complete:
                                    if dataset_latest_complete is None or t.datetime_complete > dataset_latest_complete:
                                        dataset_latest_complete = t.datetime_complete
                                if t.state == optuna.trial.TrialState.RUNNING:
                                    dataset_has_running = True
            else:
                # Fusion Model Logic or Legacy mode: single study
                if study and trials:
                    for t in trials:
                        if t.datetime_start:
                            if dataset_earliest_start is None or t.datetime_start < dataset_earliest_start:
                                dataset_earliest_start = t.datetime_start
                        if t.state == optuna.trial.TrialState.COMPLETE and t.datetime_complete:
                            if dataset_latest_complete is None or t.datetime_complete > dataset_latest_complete:
                                dataset_latest_complete = t.datetime_complete
                        if t.state == optuna.trial.TrialState.RUNNING:
                            dataset_has_running = True
            
            if dataset_earliest_start:
                if dataset_has_running:
                    end_time = datetime.now()
                elif dataset_latest_complete:
                    end_time = dataset_latest_complete
                else:
                    end_time = datetime.now()
                
                total_runtime = (end_time - dataset_earliest_start).total_seconds()
                total_hours = int(total_runtime / 3600)
                total_minutes = int((total_runtime % 3600) / 60)
                total_sec = int(total_runtime % 60)
                
                status_str = " (ongoing)" if dataset_has_running else " (completed)"
                print(f"⏰ Dataset Total Runtime{status_str}: {total_hours}h {total_minutes}m {total_sec}s")
            
            # Calculate and display trial average runtime
            trial_avg_time = None
            if PER_SEED_OPTIMIZATION:
                # Per-seed mode: each trial trains only one seed
                all_completed_with_time = []
                for seed_num, stats in all_seed_stats.items():
                    if stats['study']:
                        seed_complete = [t for t in stats['trials'] if t.state == optuna.trial.TrialState.COMPLETE and t.datetime_start and t.datetime_complete]
                        all_completed_with_time.extend(seed_complete)
                
                if all_completed_with_time:
                    trial_avg_time = sum([(t.datetime_complete - t.datetime_start).total_seconds() 
                                       for t in all_completed_with_time]) / len(all_completed_with_time)
            else:
                # Fusion Model Logic or Legacy mode: each trial processes 5 seeds
                if study:
                    completed_with_time = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE and t.datetime_start and t.datetime_complete]
                    if completed_with_time:
                        trial_avg_time = sum([(t.datetime_complete - t.datetime_start).total_seconds() 
                                           for t in completed_with_time]) / len(completed_with_time)
            
            if trial_avg_time is not None:
                avg_min = int(trial_avg_time / 60)
                avg_sec = int(trial_avg_time % 60)
                if PER_SEED_OPTIMIZATION:
                    print(f"⏱️  Trial Average Runtime: {avg_min}m {avg_sec}s (per-seed optimization, 1 seed per trial)")
                else:
                    print(f"⏱️  Trial Average Runtime: {avg_min}m {avg_sec}s")
            print("")
    
except Exception as e:
    error_msg = str(e)
    if "not found" in error_msg.lower() or "does not exist" in error_msg.lower():
        if PER_SEED_OPTIMIZATION:
            print(f"❌ Per-seed studies not found for dataset: {ACTUAL_DATASET}")
        else:
            print(f"❌ Study not found: {STUDY_PREFIX}_{ACTUAL_DATASET}_opt")
    else:
        print(f"❌ Error: {error_msg}")
        import traceback
        traceback.print_exc()
EOF
    
    # === GPU STATUS WITH WARNINGS (Feature 6) ===
    if [ "$DISPLAY_MODE" = "detailed" ]; then
        if command -v nvidia-smi &> /dev/null; then
            echo ""
            echo "🎮 GPU Status:"
            if [ "$VERSION_MODE" = "mod_new" ] || [ "$VERSION_MODE" = "mod" ]; then
                echo "   Allocation (Parallel Seeds): GPU 0 → seeds [1,2] (parallel) | GPU 1 → seeds [3,4,5] (parallel)"
            fi
            GPU_WARNINGS_FILE=$(mktemp)
            HAS_RUNNING_TRIALS=$([ "$OPTUNA_PIDS" -gt 0 ] && echo "1" || echo "0")
            
            # Collect GPU information first
            GPU_INFO=()
            GPU_COUNT=0
            while IFS=', ' read -r idx util mem_used mem_total mem_free temp; do
                # Check for low utilization (only if trials are running)
                if [ "$HAS_RUNNING_TRIALS" = "1" ] && [ "$util" -lt 10 ]; then
                    echo "GPU $idx: Low utilization ($util%)" >> "$GPU_WARNINGS_FILE"
                fi
                
                # Check for high memory usage
                if [ "$mem_total" -gt 0 ]; then
                    mem_pct=$((mem_used * 100 / mem_total))
                    if [ "$mem_pct" -gt 95 ]; then
                        echo "GPU $idx: High memory usage ($mem_pct%)" >> "$GPU_WARNINGS_FILE"
                    fi
                fi
                
                # Check for low free memory
                if [ "$mem_free" -lt 1000 ]; then
                    echo "GPU $idx: Low free memory (${mem_free}MB)" >> "$GPU_WARNINGS_FILE"
                fi
                
                # Store GPU info for display
                if [ "$mem_total" -gt 0 ]; then
                    mem_pct=$((mem_used * 100 / mem_total))
                    GPU_INFO[$GPU_COUNT]="GPU $idx: ${util}% util, ${mem_used}/${mem_total} MB (${mem_pct}% used), ${mem_free} MB free, ${temp}°C"
                else
                    GPU_INFO[$GPU_COUNT]="GPU $idx: ${util}% util, ${mem_used}/${mem_total} MB, ${mem_free} MB free, ${temp}°C"
                fi
                GPU_COUNT=$((GPU_COUNT + 1))
            done < <(nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,memory.free,temperature.gpu --format=csv,noheader,nounits 2>/dev/null)
            
            # Display GPUs on the same line
            if [ "$GPU_COUNT" -gt 0 ]; then
                printf "   "
                for ((i=0; i<GPU_COUNT; i++)); do
                    printf "${GPU_INFO[$i]}"
                    if [ $i -lt $((GPU_COUNT - 1)) ]; then
                        printf " | "
                    fi
                done
                printf "\n"
            fi
            
            if [ -f "$GPU_WARNINGS_FILE" ] && [ -s "$GPU_WARNINGS_FILE" ]; then
                echo ""
                echo "⚠️  GPU Warnings:"
                while IFS= read -r warning; do
                    echo "   $warning"
                done < "$GPU_WARNINGS_FILE"
                rm -f "$GPU_WARNINGS_FILE"
            fi
        fi
    fi
    
    # === SYSTEM MEMORY CHECK (Feature 6) ===
    if [ "$DISPLAY_MODE" = "detailed" ]; then
        if command -v free &> /dev/null; then
            MEM_INFO=$(free -m | grep Mem)
            MEM_TOTAL=$(echo $MEM_INFO | awk '{print $2}')
            MEM_USED=$(echo $MEM_INFO | awk '{print $3}')
            MEM_AVAIL=$(echo $MEM_INFO | awk '{print $7}')
            MEM_PCT=$((MEM_USED * 100 / MEM_TOTAL))
            
            echo ""
            echo "💾 System Memory:"
            echo "   Total: ${MEM_TOTAL}MB | Used: ${MEM_USED}MB (${MEM_PCT}%) | Available: ${MEM_AVAIL}MB"
            
            if [ "$MEM_PCT" -gt 90 ]; then
                echo "   ⚠️  Warning: High memory usage (${MEM_PCT}%)"
            fi
            if [ "$MEM_AVAIL" -lt 1000 ]; then
                echo "   ⚠️  Warning: Low available memory (${MEM_AVAIL}MB)"
            fi
        fi
    fi
    
    # === LOG ERROR CHECK ===
    # Check for per-seed log files (new format) or legacy format
    # Fix: Check all possible worker_id log files (not just worker_id=1) for robustness
    # This handles cases where WORKERS > 1 (though current config uses WORKERS=1)
    LOG_FILES=()
    if [ "$PER_SEED_OPTIMIZATION" = true ]; then
        # Per-seed optimization: check all seed log files
        # Check worker_id 1-10 to handle cases where WORKERS > 1
        for seed_num in 1 2 3 4 5; do
            for worker_id in 1 2 3 4 5 6 7 8 9 10; do
                seed_log="logs/${LOG_DIR}/worker_${ACTUAL_DATASET}_seed${seed_num}_${worker_id}.log"
                if [ -f "$seed_log" ]; then
                    LOG_FILES+=("$seed_log")
                fi
            done
        done
    else
        # Legacy format: check all possible worker_id log files
        for worker_id in 1 2 3 4 5 6 7 8 9 10; do
            legacy_log="logs/${LOG_DIR}/worker_${ACTUAL_DATASET}_${worker_id}.log"
            if [ -f "$legacy_log" ]; then
                LOG_FILES+=("$legacy_log")
            fi
        done
    fi
    
    # Check all found log files for errors
    if [ ${#LOG_FILES[@]} -gt 0 ]; then
        TOTAL_ERROR_COUNT=0
        ERROR_LINES=()
        for LOG_FILE in "${LOG_FILES[@]}"; do
            ERROR_COUNT=$(grep -i "error\|exception\|failed" "$LOG_FILE" 2>/dev/null | tail -20 | wc -l)
            if [ "$ERROR_COUNT" -gt 0 ]; then
                TOTAL_ERROR_COUNT=$((TOTAL_ERROR_COUNT + ERROR_COUNT))
                # Extract error lines from this log file
                while IFS= read -r line; do
                    ERROR_LINES+=("$line")
                done < <(grep -i "error\|exception\|failed" "$LOG_FILE" 2>/dev/null | tail -3)
            fi
        done
        
        if [ "$TOTAL_ERROR_COUNT" -gt 0 ]; then
            echo ""
            echo "⚠️  Recent Errors in Logs:"
            # Show up to 3 most recent error lines
            for i in "${!ERROR_LINES[@]}"; do
                if [ $i -lt 3 ]; then
                    echo "   ${ERROR_LINES[$i]:0:100}"  # Truncate long lines
                fi
            done
        fi
    fi
    
    echo ""
    echo "=========================================="
    if [ "$DISPLAY_MODE" = "compact" ]; then
        echo "Next update in ${INTERVAL}s... (s=detail, p=pause, e=export, q=quit)"
    else
        echo "Next update in ${INTERVAL} seconds... (s=compact, p=pause, e=export, q=quit)"
    fi
    
    # Sleep with periodic key check
    for i in $(seq 1 $INTERVAL); do
        sleep 1
        KEY=$(read_key)
        case "$KEY" in
            s|S)
                if [ "$DISPLAY_MODE" = "detailed" ]; then
                    DISPLAY_MODE="compact"
                else
                    DISPLAY_MODE="detailed"
                fi
                break
                ;;
            p|P)
                PAUSED=true
                break
                ;;
            e|E)
                export_status
                sleep 1
                ;;
            q|Q)
                echo ""
                echo "👋 Exiting monitor..."
                exit 0
                ;;
        esac
    done
done

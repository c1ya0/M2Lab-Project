#!/bin/bash

# Optuna Sequential Optimization Launcher (NEW version - Fusion Model Logic)
# This version follows fusion_model's approach with sequential trial execution:
#   - Trials execute sequentially (one trial at a time)
#   - Each trial trains all 5 seeds in parallel with the same hyperparameters
#   - GPU allocation per trial: GPU 0 → seeds [1, 2], GPU 1 → seeds [3, 4, 5]
#   - All seeds use the same hyperparameters (sampled at trial level)
#   - Collect best validation metrics from each seed
#   - Return average validation metric as the trial's objective value
#   - Optuna selects the trial with best average performance (best hyperparameter combination)
#   - Final training uses the same best hyperparameters for all 5 seeds
#
# Execution Strategy:
#   - Sequential trial execution: Wait for previous trial to complete before starting next
#   - Parallel seed execution within each trial: All 5 seeds run simultaneously
#   - Benefits: Reduced GPU memory pressure, fewer concurrent processes, more stable execution
#
# Usage: ./optuna_parallel_mod_new.sh [dataset1] [dataset2 ...] [mode: fast/standard/deep] [--exclude dataset1 dataset2 ...] [--workers N]
#        ./optuna_parallel_mod_new.sh all [mode] [--exclude dataset1 dataset2 ...] [--workers N]
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

# -----------------------------------------------------------------------------
# NEW: Use processed_tdc_data_new as the dataset source (NO symlinks).
# scripts/optuna_serach_mod.py and scripts/train_edmpnn.py must be called with
# --tdc_processed_dir to read from it.
# -----------------------------------------------------------------------------
TDC_DATA_DIR="data/processed_tdc_data_new"

if [ ! -d "${TDC_DATA_DIR}" ]; then
    echo "❌ New processed dir not found: ${TDC_DATA_DIR}"
    echo "   Please run preprocessing to generate it first."
    exit 1
fi

# Activate conda environment (重要：確保使用正確的 Python 環境)
# 舊腳本有這個邏輯，新腳本缺少了，導致使用系統 Python 3.13 而不是 conda 環境
if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
    if conda env list | grep -q "^aegnn_env "; then
        echo "🔧 Activating conda environment: aegnn_env"
        conda activate aegnn_env
        # Verify Python path after activation
        PYTHON_CMD=$(which python3)
        echo "   Using Python: $PYTHON_CMD"
        # Verify PyTorch and torch-geometric can be imported
        if ! python3 -c "import torch; import torch_geometric" 2>/dev/null; then
            echo "   ⚠️  Warning: PyTorch or torch-geometric import failed"
            echo "   This may cause trials to fail. Please check your environment."
        else
            echo "   ✅ PyTorch and torch-geometric import successful"
        fi
    else
        echo "⚠️  Warning: aegnn_env conda environment not found"
        echo "   Continuing with system Python (may cause compatibility issues)"
    fi
else
    echo "⚠️  Warning: conda not found in PATH"
    echo "   Continuing with system Python (may cause compatibility issues)"
fi

# Fix GLIBCXX version issue: use Anaconda's libstdc++ instead of system's
# numpy requires GLIBCXX_3.4.29, but system libstdc++ only has up to 3.4.28
if command -v python3 &> /dev/null; then
    PYTHON_PATH=$(which python3)
    if [[ "$PYTHON_PATH" == *"anaconda"* ]] || [[ "$PYTHON_PATH" == *"conda"* ]]; then
        # Extract Anaconda root directory
        ANACONDA_ROOT=$(dirname $(dirname "$PYTHON_PATH"))
        ANACONDA_LIB="${ANACONDA_ROOT}/lib"
        if [ -d "$ANACONDA_LIB" ]; then
            # Prepend Anaconda lib to LD_LIBRARY_PATH
            if [ -n "$LD_LIBRARY_PATH" ]; then
                export LD_LIBRARY_PATH="${ANACONDA_LIB}:${LD_LIBRARY_PATH}"
            else
                export LD_LIBRARY_PATH="${ANACONDA_LIB}"
            fi
            echo "🔧 Using Anaconda's libstdc++ from: ${ANACONDA_LIB}"
        fi
    fi
fi

# Smart parameter parsing: handle --exclude as first argument and support multiple datasets
TARGET_DATASETS=()  # Array to store target datasets (can be multiple)
mode="standard"
EXCLUDE_DATASETS=()
MANUAL_WORKERS=""  # Manually set worker count (optional)
SEEDING_MODE="dmp_egnn"  # Default: use DMP-EGNN seeding mode

# Check if first argument is --exclude or --seeding-mode
if [[ "$1" == "--exclude" ]]; then
    TARGET_DATASETS=("all")  # Use "all" as special marker
    shift
    # Collect excluded datasets
    while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
        EXCLUDE_DATASETS+=("$1")
        shift
    done
    # Check if next argument is a mode
    if [[ "$1" =~ ^(fast|standard|deep)$ ]]; then
        mode="$1"
        shift
    fi
elif [[ "$1" == "--seeding-mode" ]] || [[ "$1" == "--seeding_mode" ]]; then
    shift
    if [[ "$1" =~ ^(dmp_egnn|fusion_model)$ ]]; then
        SEEDING_MODE="$1"
        shift
    else
        echo "❌ Error: Invalid seeding mode: $1"
        echo "   Valid options: dmp_egnn, fusion_model"
        exit 1
    fi
    # Continue parsing for datasets/mode
    if [[ "$1" == "all" ]]; then
        TARGET_DATASETS=("all")
        shift
        if [[ "$1" =~ ^(fast|standard|deep)$ ]]; then
            mode="$1"
            shift
        fi
    fi
else
    # Normal parsing: collect datasets until we hit a mode or option
    # Support: dataset1 [dataset2 ...] [mode] [--exclude ...] [--workers N]
    while [[ $# -gt 0 ]]; do
        if [[ "$1" == "all" ]]; then
            TARGET_DATASETS=("all")
            shift
            # Check if next argument is a mode
            if [[ "$1" =~ ^(fast|standard|deep)$ ]]; then
                mode="$1"
                shift
            fi
            break
        elif [[ "$1" =~ ^(fast|standard|deep)$ ]]; then
            mode="$1"
            shift
            break
        elif [[ "$1" == "--exclude" ]]; then
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                EXCLUDE_DATASETS+=("$1")
                shift
            done
        elif [[ "$1" == "--workers" ]]; then
            shift
            if [[ "$1" =~ ^[0-9]+$ ]]; then
                MANUAL_WORKERS="$1"
                shift
            else
                echo "❌ Error: --workers requires a number"
                exit 1
            fi
        elif [[ "$1" == "--seeding-mode" ]] || [[ "$1" == "--seeding_mode" ]]; then
            shift
            if [[ "$1" =~ ^(dmp_egnn|fusion_model)$ ]]; then
                SEEDING_MODE="$1"
                shift
            else
                echo "❌ Error: Invalid seeding mode: $1"
                echo "   Valid options: dmp_egnn, fusion_model"
                exit 1
            fi
        else
            # This is a dataset name
            TARGET_DATASETS+=("$1")
            shift
        fi
    done
    
    # Continue parsing remaining options (--exclude, --workers, --seeding-mode)
    while [[ $# -gt 0 ]]; do
        if [[ "$1" == "--exclude" ]]; then
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                EXCLUDE_DATASETS+=("$1")
                shift
            done
        elif [[ "$1" == "--workers" ]]; then
            shift
            if [[ "$1" =~ ^[0-9]+$ ]]; then
                MANUAL_WORKERS="$1"
                shift
            else
                echo "❌ Error: --workers requires a number"
                exit 1
            fi
        elif [[ "$1" == "--seeding-mode" ]] || [[ "$1" == "--seeding_mode" ]]; then
            shift
            if [[ "$1" =~ ^(dmp_egnn|fusion_model)$ ]]; then
                SEEDING_MODE="$1"
                shift
            else
                echo "❌ Error: Invalid seeding mode: $1"
                echo "   Valid options: dmp_egnn, fusion_model"
                exit 1
            fi
        else
            shift
        fi
    done
fi

# Also support setting worker count via environment variable (lower priority than command line argument)
if [ -z "$MANUAL_WORKERS" ] && [ -n "$OPTUNA_WORKERS" ]; then
    if [[ "$OPTUNA_WORKERS" =~ ^[0-9]+$ ]]; then
        MANUAL_WORKERS="$OPTUNA_WORKERS"
    fi
fi

# Define supported dataset list (TDC datasets from processed_tdc_data_new)
# Auto-detect TDC datasets from processed_tdc_data_new directory
if [ -d "$TDC_DATA_DIR" ]; then
    # Get all directories in processed_tdc_data (exclude hidden directories and files)
    ALL_DATASETS=($(ls -d "$TDC_DATA_DIR"/*/ 2>/dev/null | grep -v '/\.' | xargs -n1 basename | sort))
    if [ ${#ALL_DATASETS[@]} -eq 0 ]; then
        echo "⚠️  Warning: No datasets found in $TDC_DATA_DIR"
        echo "   Falling back to default dataset list"
        ALL_DATASETS=("bace" "bbbp" "clintox" "hiv" "muv" "sider" "tox21")
    else
        echo "✅ Found ${#ALL_DATASETS[@]} TDC datasets in $TDC_DATA_DIR"
    fi
else
    echo "⚠️  Warning: $TDC_DATA_DIR not found"
    echo "   Falling back to default dataset list"
    ALL_DATASETS=("bace" "bbbp" "clintox" "hiv" "muv" "sider" "tox21")
fi

# Filter out excluded datasets
if [ ${#EXCLUDE_DATASETS[@]} -gt 0 ]; then
    FILTERED_DATASETS=()
    for dataset in "${ALL_DATASETS[@]}"; do
        exclude=false
        for excluded in "${EXCLUDE_DATASETS[@]}"; do
            if [ "$dataset" == "$excluded" ]; then
                exclude=true
                break
            fi
        done
        if [ "$exclude" == false ]; then
            FILTERED_DATASETS+=("$dataset")
        fi
    done
    ALL_DATASETS=("${FILTERED_DATASETS[@]}")
fi

# Check parameters
if [ ${#TARGET_DATASETS[@]} -eq 0 ]; then
    echo "Usage: ./optuna_parallel_mod_new.sh <dataset1> [dataset2 ...] [mode] [--exclude dataset1 dataset2 ...] [--workers N] [--seeding-mode MODE]"
    echo "       ./optuna_parallel_mod_new.sh all [mode] [--exclude dataset1 dataset2 ...] [--workers N] [--seeding-mode MODE]"
    echo ""
    echo "Modes: fast (default), standard, deep"
    echo ""
    echo "Options:"
    echo "  --workers N           Manually set the number of parallel workers (default: auto-calculated)"
    echo "                         Useful for avoiding OOM errors by reducing parallelism"
    echo "  --seeding-mode MODE    Seeding strategy (default: dmp_egnn)"
    echo "                         Options:"
    echo "                           dmp_egnn:      Use seed * 1000 + seed for model init (1001-5005)"
    echo "                                          [Default, preserves existing results]"
    echo "                           fusion_model:  Use seed directly for model init (1-5)"
    echo "                                          [For comparability with fusion_model]"
    echo ""
    echo "Examples:"
    echo "  # Single dataset with default seeding:"
    echo "  ./optuna_parallel_mod_new.sh caco2_wang standard"
    echo ""
    echo "  # Single dataset with fusion_model seeding:"
    echo "  ./optuna_parallel_mod_new.sh caco2_wang standard --seeding-mode fusion_model"
    echo ""
    echo "  # Multiple datasets:"
    echo "  ./optuna_parallel_mod_new.sh herg ames bace standard"
    echo ""
    echo "  # All datasets with fusion_model seeding:"
    echo "  ./optuna_parallel_mod_new.sh all standard --seeding-mode fusion_model"
    echo ""
    echo "  # Using environment variable:"
    echo "  OPTUNA_WORKERS=5 ./optuna_parallel_mod_new.sh caco2_wang standard"
    exit 1
fi

# ==========================================
# 1. Hardware Resource Detection (GPU & CPU)
# ==========================================
echo "🔍 Detecting hardware resources..."

CPU_CORES=$(nproc)
echo "   CPU Cores: $CPU_CORES"

if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    GPU_MEM_TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1)
    echo "   GPU: Found $GPU_COUNT NVIDIA GPU(s)"
    echo "   VRAM (First GPU): ${GPU_MEM_TOTAL} MB"
else
    GPU_COUNT=0
    GPU_MEM_TOTAL=0
    echo "   GPU: None detected (or nvidia-smi missing). Will use CPU mode."
fi

# ==========================================
# 2. Set Workers (Sequential Trial Execution)
# ==========================================
# NEW LOGIC: Sequential trial execution (one trial at a time)
#   - Each trial trains all 5 seeds in parallel
#   - GPU 0: seeds [1, 2]
#   - GPU 1: seeds [3, 4, 5]
#   - Trials execute sequentially (wait for previous trial to complete)
#   - This reduces GPU memory pressure and process count
if [ -n "$MANUAL_WORKERS" ]; then
    # Allow manual override, but warn if > 1
    WORKERS=$MANUAL_WORKERS
    if [ "$WORKERS" -gt 1 ]; then
        echo "⚠️  Warning: Manual workers set to $WORKERS, but sequential mode uses 1 worker"
        echo "   Setting WORKERS=1 for sequential trial execution"
        WORKERS=1
    fi
    if [ "$WORKERS" -lt 1 ]; then
        echo "❌ Error: Worker count must be at least 1"
        exit 1
    fi
    echo "💡 Using manually specified workers: $WORKERS (sequential mode)"
else
    # Sequential mode: always use 1 worker
    WORKERS=1
    
    # Check GPU memory usage for informational purposes
    if command -v nvidia-smi &> /dev/null; then
        GPU_MEM_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
        GPU_MEM_TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
        GPU_MEM_AVAILABLE=$((GPU_MEM_TOTAL - GPU_MEM_USED))
        echo "   ✅ GPU available memory: ${GPU_MEM_AVAILABLE} MiB"
    fi

    echo "💡 Sequential Trial Execution Mode (Fusion Model Logic)"
    echo "   Execution Strategy:"
    echo "     - Trials execute sequentially (one at a time)"
    echo "     - Each trial trains all 5 seeds in parallel"
    echo "     - GPU 0 → seeds [1, 2]"
    echo "     - GPU 1 → seeds [3, 4, 5]"
    echo "   Benefits: Reduced GPU memory pressure, fewer concurrent processes"
fi

# ==========================================
# 3. Set Trials (based on mode)
# ==========================================
case $mode in
    fast)
        TOTAL_TRIALS=20
        EPOCHS=75
        # Calculation: 20 trials × avg 165 min/trial (5 seeds per trial, ~33 min per seed) ÷ workers
        MAX_WAIT_TIME=148500  # Approx 41.25 hours
        ;;
    standard)
        TOTAL_TRIALS=50
        EPOCHS=125
        # Calculation: 50 trials × avg 275 min/trial (5 seeds per trial, ~55 min per seed) ÷ workers
        MAX_WAIT_TIME=619200  # Approx 172 hours
        ;;
    deep)
        TOTAL_TRIALS=160
        EPOCHS=250
        # Calculation: 160 trials × avg 550 min/trial (5 seeds per trial, ~110 min per seed) ÷ workers
        MAX_WAIT_TIME=3960000  # Approx 1100 hours
        ;;
    *)
        echo "Unknown mode: $mode. Using standard."
        TOTAL_TRIALS=50
        EPOCHS=150
        MAX_WAIT_TIME=742500  # Approx 206.25 hours
        ;;
esac

# More precise trial distribution to avoid exceeding TOTAL_TRIALS
# Distribute trials evenly, with remainder going to last worker
BASE_TRIALS_PER_WORKER=$((TOTAL_TRIALS / WORKERS))
REMAINDER_TRIALS=$((TOTAL_TRIALS % WORKERS))

    echo "🎯 Goal: $TOTAL_TRIALS trials total ($mode mode)"
    echo "   Seeding Mode: $SEEDING_MODE (Model Init: $([ "$SEEDING_MODE" = "fusion_model" ] && echo "seed directly (1-5)" || echo "seed * 1000 + seed (1001-5005)"))"
    echo "   📊 Execution Strategy (Sequential Trial Execution):"
    echo "      - Trials execute sequentially (one trial at a time)"
    echo "      - Each trial trains all 5 seeds in parallel"
    echo "      - GPU 0 → seeds [1, 2], GPU 1 → seeds [3, 4, 5]"
    echo "      - All seeds use the same hyperparameters (sampled at trial level)"
    echo "      - Return average validation metric as the trial's objective value"
    echo "      - Optuna selects the trial with best average performance"
    echo "   Max Epochs: $EPOCHS"
    echo "   Timeout: $MAX_WAIT_TIME seconds ($(($MAX_WAIT_TIME / 3600)) hours)"

# ==========================================
# 4. Check disk space before starting
# ==========================================
echo "💾 Checking disk space..."
# Get available disk space in MB (for current directory)
AVAILABLE_SPACE_MB=$(df -BM . | tail -1 | awk '{print $4}' | sed 's/M//')
# Estimate required space: database (~100MB per dataset), checkpoints (~500MB per trial), logs (~10MB per trial)
# Rough estimate: 100MB + (TOTAL_TRIALS * 510MB)
REQUIRED_SPACE_MB=$((100 + TOTAL_TRIALS * 510))
REQUIRED_SPACE_GB=$((REQUIRED_SPACE_MB / 1024 + 1))

if [ "$AVAILABLE_SPACE_MB" -lt "$REQUIRED_SPACE_MB" ]; then
    echo "⚠️  Warning: Low disk space!"
    echo "   Available: ${AVAILABLE_SPACE_MB}MB"
    echo "   Estimated required: ${REQUIRED_SPACE_MB}MB (~${REQUIRED_SPACE_GB}GB)"
    echo "   This may cause failures during execution."
    read -p "   Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "❌ Aborted by user."
        exit 1
    fi
else
    echo "   ✅ Available space: ${AVAILABLE_SPACE_MB}MB (estimated need: ${REQUIRED_SPACE_MB}MB)"
fi

# ==========================================
# 5. Execute optimization function (using optuna_serach_mod.py)
# ==========================================

# Global variables to store current process group ID and PIDs for cleanup
CURRENT_PGID=""
CURRENT_PIDS=()

# Enhanced cleanup function to kill all worker processes and their children
cleanup_workers() {
    echo ""
    echo "🛑 Received interrupt signal. Cleaning up worker processes..."
    
    # Method 1: Kill by process group (if available)
    if [ -n "$CURRENT_PGID" ]; then
        if kill -0 -"$CURRENT_PGID" 2>/dev/null; then
            echo "   Terminating process group: $CURRENT_PGID"
            kill -TERM -"$CURRENT_PGID" 2>/dev/null || true
            sleep 2
            if kill -0 -"$CURRENT_PGID" 2>/dev/null; then
                echo "   Force killing process group: $CURRENT_PGID"
                kill -KILL -"$CURRENT_PGID" 2>/dev/null || true
            fi
        fi
    fi
    
    # Method 2: Kill all tracked PIDs and their children
    for pid in "${CURRENT_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "   Terminating PID $pid and its children..."
            # Kill the process and all its children
            pkill -TERM -P "$pid" 2>/dev/null || true
            kill -TERM "$pid" 2>/dev/null || true
        fi
    done
    
    # Wait a bit for graceful shutdown
    sleep 3
    
    # Method 3: Force kill any remaining processes by name pattern
    echo "   Searching for remaining optuna_serach_mod processes..."
    REMAINING_PIDS=$(pgrep -f "optuna_serach_mod.py" 2>/dev/null || true)
    if [ -n "$REMAINING_PIDS" ]; then
        echo "   Found remaining processes: $REMAINING_PIDS"
        for pid in $REMAINING_PIDS; do
            # Kill process and all its children
            pkill -KILL -P "$pid" 2>/dev/null || true
            kill -KILL "$pid" 2>/dev/null || true
        done
    fi
    
    # Method 4: Kill any remaining training processes (train_edmpnn.py)
    echo "   Searching for remaining training processes..."
    TRAINING_PIDS=$(pgrep -f "train_edmpnn.py" 2>/dev/null || true)
    if [ -n "$TRAINING_PIDS" ]; then
        echo "   Found remaining training processes: $TRAINING_PIDS"
        for pid in $TRAINING_PIDS; do
            pkill -KILL -P "$pid" 2>/dev/null || true
            kill -KILL "$pid" 2>/dev/null || true
        done
    fi
    
    # Final check: kill any Python processes related to aegnn_env that are consuming resources
    echo "   Final cleanup: checking for orphaned Python processes..."
    sleep 1
    
    echo "✅ Cleanup completed."
    exit 130
}

# Set up signal handlers
trap cleanup_workers SIGINT SIGTERM

# NEW: Single optimization function (no per-seed optimization)
# Each trial trains all 5 seeds with the same hyperparameters
run_optimization() {
    local current_dataset=$1
    # Local arrays for this dataset's workers
    local -a pids=()
    local pgid=""
    # Update global PIDs array for cleanup
    CURRENT_PIDS=()

    echo "-------------------------------------------"
    echo "🚀 Launching MOD Optimization (Fusion Model Logic) for dataset: $current_dataset"
    STORAGE_URL="sqlite:///optuna_edmpnn_results/optuna_mod_new.db"
    
    # Avoid duplicate workers: if another optuna worker is already running for this dataset, do not start a second one
    EXISTING_OPTUNA_PIDS=$(pgrep -f "optuna_serach_mod.py.*--dataset $current_dataset" 2>/dev/null || true)
    if [ -n "$EXISTING_OPTUNA_PIDS" ]; then
        echo "   ⚠️  Warning: Optuna worker(s) already running for dataset $current_dataset (PIDs: $EXISTING_OPTUNA_PIDS)"
        echo "   Please stop them first (e.g. Ctrl+C in the terminal that started them, or: pkill -f 'optuna_serach_mod.py.*$current_dataset')"
        echo "   Skipping to avoid duplicate workers and extra trials."
        return 1
    fi
    echo "   Storage: $STORAGE_URL"
    echo "   Logs: logs/optuna_launcher_mod_new/"

    mkdir -p logs/optuna_launcher_mod_new
    mkdir -p optuna_edmpnn_results
    # Ensure database directory has write permissions
    chmod -R u+w optuna_edmpnn_results 2>/dev/null || true
    # Fix database file permissions if it exists
    if [ -f "optuna_edmpnn_results/optuna_mod_new.db" ]; then
        chmod u+w "optuna_edmpnn_results/optuna_mod_new.db" 2>/dev/null || true
    fi

    # Study name: single study for all seeds (legacy mode in optuna_serach_mod.py)
    # When --seed is not provided, optuna_serach_mod.py uses legacy mode:
    #   - Each trial trains all 5 seeds with the same hyperparameters
    #   - Returns average validation metric
    STUDY_NAME="aegnn_mod_${current_dataset}_opt"
    echo "   Study: $STUDY_NAME"
    echo "   Mode: Fusion Model Logic (each trial trains all 5 seeds with same hyperparameters)"

    # Check existing trials and calculate remaining trials needed
    echo "   Checking existing trials in study..."
    EXISTING_TRIALS_INFO=$(python3 -c "
import optuna
try:
    study = optuna.load_study(study_name='$STUDY_NAME', storage='$STORAGE_URL')
    total = len(study.trials)
    completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    failed = len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])
    running = len([t for t in study.trials if t.state == optuna.trial.TrialState.RUNNING])
    print(f'{total},{completed},{failed},{running}', end='')
except Exception as e:
    # Study doesn't exist yet, will be created by optuna_serach_mod.py
    print('0,0,0,0', end='')
" 2>/dev/null || echo "0,0,0,0")
    
    EXISTING_TOTAL=$(echo "$EXISTING_TRIALS_INFO" | cut -d',' -f1)
    EXISTING_COMPLETED=$(echo "$EXISTING_TRIALS_INFO" | cut -d',' -f2)
    EXISTING_FAILED=$(echo "$EXISTING_TRIALS_INFO" | cut -d',' -f3)
    EXISTING_RUNNING=$(echo "$EXISTING_TRIALS_INFO" | cut -d',' -f4)
    
    # Ensure all values are numeric (handle empty strings)
    if [ -z "$EXISTING_TOTAL" ] || ! [[ "$EXISTING_TOTAL" =~ ^[0-9]+$ ]]; then
        EXISTING_TOTAL=0
    fi
    if [ -z "$EXISTING_COMPLETED" ] || ! [[ "$EXISTING_COMPLETED" =~ ^[0-9]+$ ]]; then
        EXISTING_COMPLETED=0
    fi
    if [ -z "$EXISTING_FAILED" ] || ! [[ "$EXISTING_FAILED" =~ ^[0-9]+$ ]]; then
        EXISTING_FAILED=0
    fi
    if [ -z "$EXISTING_RUNNING" ] || ! [[ "$EXISTING_RUNNING" =~ ^[0-9]+$ ]]; then
        EXISTING_RUNNING=0
    fi
    
    echo "   📊 Existing trials: Total=$EXISTING_TOTAL, Completed=$EXISTING_COMPLETED, Failed=$EXISTING_FAILED, Running=$EXISTING_RUNNING"
    
    # Cap by TOTAL trial count: do not start more if we already have >= TOTAL_TRIALS trials
    # (avoids overshooting after interrupt/zombie and avoids duplicate workers adding extra trials)
    if [ "$EXISTING_TOTAL" -ge "$TOTAL_TRIALS" ]; then
        echo "   ✅ Target of $TOTAL_TRIALS trials already reached (total trials: $EXISTING_TOTAL)"
        echo "   Skipping optimization (no new trials needed)"
        return 2  # Return 2 indicates skip (no cooldown needed)
    fi
    
    # Remaining = how many more trials to create so total does not exceed TOTAL_TRIALS
    REMAINING_TRIALS=$((TOTAL_TRIALS - EXISTING_TOTAL))
    if [ $REMAINING_TRIALS -le 0 ]; then
        echo "   ✅ No new trials needed (total $EXISTING_TOTAL >= target $TOTAL_TRIALS)"
        return 2
    fi
    
    echo "   🎯 Remaining trials to run: $REMAINING_TRIALS (target total: $TOTAL_TRIALS, existing total: $EXISTING_TOTAL)"
    
    # Sequential mode: single worker handles all remaining trials
    # Each trial will execute sequentially, but seeds within each trial run in parallel
    WORKER_TRIALS=$REMAINING_TRIALS
    
    echo "   📋 Sequential Execution: 1 worker will process $WORKER_TRIALS trials sequentially"
    echo "      Each trial trains 5 seeds in parallel (GPU 0: seeds 1,2; GPU 1: seeds 3,4,5)"

    # Start single worker in a new process group for better signal handling
    # Use setsid to create new session and process group
    WORKERS_STARTED=0
    
    # Sequential mode: only start 1 worker
    if [ $WORKER_TRIALS -gt 0 ]; then
        WORKERS_STARTED=1
        echo "   Starting Worker 1 (${WORKER_TRIALS} trials, sequential execution, each trial trains all 5 seeds in parallel)..."

        # Sequential mode: single worker
        # IMPORTANT: Do NOT pass --seed argument, so optuna_serach_mod.py uses legacy mode
        # Legacy mode: each trial trains all 5 seeds with same hyperparameters in parallel
        setsid python3 scripts/optuna_serach_mod.py \
            --dataset "$current_dataset" \
            --tdc_processed_dir "$TDC_DATA_DIR" \
            --n_trials "$WORKER_TRIALS" \
            --storage "$STORAGE_URL" \
            --epochs "$EPOCHS" \
            --worker_id 1 \
            --seeding_mode "$SEEDING_MODE" \
            2>&1 | tee "logs/optuna_launcher_mod_new/worker_${current_dataset}_1.log" &
        local worker_pid=$!
        # Get process group ID
        pgid=$(ps -o pgid= -p "$worker_pid" 2>/dev/null | tr -d ' ')
        CURRENT_PGID="$pgid"  # Update global for cleanup
        
        # Fix: Verify process started successfully
        sleep 1  # Give process time to start
        if ! kill -0 "$worker_pid" 2>/dev/null; then
            # Process already died, check log for errors
            if [ -f "logs/optuna_launcher_mod_new/worker_${current_dataset}_1.log" ]; then
                if grep -q "Error\|Traceback\|Exception" "logs/optuna_launcher_mod_new/worker_${current_dataset}_1.log" 2>/dev/null; then
                    echo "   ⚠️  Worker 1 failed to start (check log for errors)"
                    WORKERS_STARTED=0
                fi
            fi
        fi
        
        pids+=($worker_pid)
        CURRENT_PIDS+=($worker_pid)  # Add to global array for cleanup
    fi

    # Check if any workers were started
    if [ $WORKERS_STARTED -eq 0 ]; then
        echo "   ⚠️  No workers started (all workers had 0 trials assigned)"
        echo "   This should not happen if REMAINING_TRIALS > 0"
        return 0
    fi

    echo "✅ Started $WORKERS_STARTED worker(s). PIDs: ${pids[*]}"
    echo "ℹ️  Worker 1 output is mirrored to this terminal."
    echo "   To monitor other workers, use: tail -f logs/optuna_launcher_mod_new/worker_${current_dataset}_2.log"
    echo "   Press Ctrl+C to stop all workers gracefully."
    echo ""
    echo "Waiting for completion..."

    # Improved wait logic: parallel waiting with error checking
    # Array to store exit codes
    declare -a exit_codes=()
    for i in "${!pids[@]}"; do
        exit_codes+=(-1)  # Initialize with -1 (not finished)
    done
    
    # Wait for all processes in parallel with timeout (MAX_WAIT_TIME set based on mode)
    START_TIME=$(date +%s)
    ALL_FINISHED=false
    FAILED_PIDS=()
    LAST_PROGRESS_UPDATE=0
    PROGRESS_UPDATE_INTERVAL=15  # Update progress every 15 seconds
    
    # Initialize progress bar display
    echo -n "   📊 Progress: [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░] 0/$TOTAL_TRIALS (0%) | Elapsed: 0min"
    
    while [ "$ALL_FINISHED" = false ]; do
        ALL_FINISHED=true
        CURRENT_TIME=$(date +%s)
        ELAPSED=$((CURRENT_TIME - START_TIME))
        
        # Progress monitoring: check Optuna database for completed trials
        if [ $((CURRENT_TIME - LAST_PROGRESS_UPDATE)) -ge $PROGRESS_UPDATE_INTERVAL ]; then
            LAST_PROGRESS_UPDATE=$CURRENT_TIME
            
            # Try to get trial count from Optuna database
            TRIAL_INFO=$(python3 -c "
import optuna
try:
    study = optuna.load_study(study_name='$STUDY_NAME', storage='$STORAGE_URL')
    total = len(study.trials)
    completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    failed = len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])
    running = len([t for t in study.trials if t.state == optuna.trial.TrialState.RUNNING])
    print(f'{completed},{total},{failed},{running}', end='')
except Exception as e:
    print('0,0,0,0', end='')
" 2>/dev/null || echo "0,0,0,0")
            
            # Parse trial info: completed,total,failed,running
            COMPLETED_TRIALS=$(echo "$TRIAL_INFO" | cut -d',' -f1)
            TOTAL_TRIALS_IN_DB=$(echo "$TRIAL_INFO" | cut -d',' -f2)
            FAILED_TRIALS=$(echo "$TRIAL_INFO" | cut -d',' -f3)
            RUNNING_TRIALS=$(echo "$TRIAL_INFO" | cut -d',' -f4)
            
            # Use actual completed trials for progress, but show total in DB if different
            if [ -n "$COMPLETED_TRIALS" ] && [ "$COMPLETED_TRIALS" != "0" ] || ([ -n "$TOTAL_TRIALS_IN_DB" ] && [ "$TOTAL_TRIALS_IN_DB" != "0" ]); then
                if [ "$TOTAL_TRIALS" -gt 0 ]; then
                    PROGRESS_PCT=$((COMPLETED_TRIALS * 100 / TOTAL_TRIALS))
                    ELAPSED_MIN=$((ELAPSED / 60))
                    
                    # Calculate progress bar (50 characters wide)
                    BAR_WIDTH=50
                    FILLED=$((COMPLETED_TRIALS * BAR_WIDTH / TOTAL_TRIALS))
                    EMPTY=$((BAR_WIDTH - FILLED))
                    
                    # Build progress bar
                    BAR=""
                    for i in $(seq 1 $FILLED); do
                        BAR="${BAR}█"
                    done
                    for i in $(seq 1 $EMPTY); do
                        BAR="${BAR}░"
                    done
                    
                    # Print progress bar on the same line
                    if [ "$FAILED_TRIALS" != "0" ] || [ "$RUNNING_TRIALS" != "0" ]; then
                        printf "\r   📊 Progress: [%s] %d/%d (%d%%) | Completed: %d, Failed: %d, Running: %d | Elapsed: %dmin    " \
                            "$BAR" "$COMPLETED_TRIALS" "$TOTAL_TRIALS" "$PROGRESS_PCT" "$COMPLETED_TRIALS" "$FAILED_TRIALS" "$RUNNING_TRIALS" "$ELAPSED_MIN"
                    else
                        printf "\r   📊 Progress: [%s] %d/%d (%d%%) | Elapsed: %dmin    " \
                            "$BAR" "$COMPLETED_TRIALS" "$TOTAL_TRIALS" "$PROGRESS_PCT" "$ELAPSED_MIN"
                    fi
                fi
            fi
        fi
        
        for i in "${!pids[@]}"; do
            pid="${pids[$i]}"
            
            # Check if process is still running
            if kill -0 "$pid" 2>/dev/null; then
                ALL_FINISHED=false
                
                # Check timeout (warn but don't immediately terminate)
                if [ $ELAPSED -gt $MAX_WAIT_TIME ]; then
                    # Only warn once per worker
                    if [ "${exit_codes[$i]}" -eq -1 ]; then
                        echo ""
                        echo "⚠️  Worker PID $pid exceeded max wait time ($(($MAX_WAIT_TIME / 60)) minutes)"
                        echo "   Allowing to continue for up to 50% more time to complete current trials..."
                        exit_codes[$i]=-2  # Mark as warned (not -1, not finished)
                    fi
                    # Only terminate if exceeded 150% of max wait time
                    if [ $ELAPSED -gt $((MAX_WAIT_TIME * 3 / 2)) ]; then
                        echo "⚠️  Worker PID $pid exceeded extended timeout ($(($MAX_WAIT_TIME * 3 / 2 / 60)) minutes), terminating..."
                        kill -TERM "$pid" 2>/dev/null || true
                        sleep 2
                        if kill -0 "$pid" 2>/dev/null; then
                            kill -KILL "$pid" 2>/dev/null || true
                        fi
                        exit_codes[$i]=124  # Timeout exit code
                        FAILED_PIDS+=("$pid")
                    fi
                fi
            elif [ "${exit_codes[$i]}" -eq -1 ] || [ "${exit_codes[$i]}" -eq -2 ]; then
                # Process finished, collect exit status
                if wait "$pid" 2>/dev/null; then
                    exit_codes[$i]=$?
                else
                    exit_codes[$i]=$?
                fi
                
                # Check if exit code indicates failure
                if [ "${exit_codes[$i]}" -ne 0 ]; then
                    FAILED_PIDS+=("$pid")
                    echo "❌ Worker PID $pid exited with code ${exit_codes[$i]}"
                fi
            fi
        done
        
        if [ "$ALL_FINISHED" = false ]; then
            sleep 2  # Check every 2 seconds instead of 1
        fi
    done

    # Show final progress (100%) before printing newline
    BAR_WIDTH=50
    FINAL_BAR=""
    for i in $(seq 1 $BAR_WIDTH); do
        FINAL_BAR="${FINAL_BAR}█"
    done
    printf "\r   📊 Progress: [%s] %d/%d (100%%) | Completed!    \n" \
        "$FINAL_BAR" "$TOTAL_TRIALS" "$TOTAL_TRIALS"

    # Clear global PGID and PIDs after completion
    CURRENT_PGID=""
    CURRENT_PIDS=()
    
    # Check for failures
    if [ ${#FAILED_PIDS[@]} -gt 0 ]; then
        echo ""
        echo "❌ Some workers failed! Failed PIDs: ${FAILED_PIDS[*]}"
        echo "   Exit codes: ${exit_codes[*]}"
        echo "   Check logs in logs/optuna_launcher_mod_new/ for details."
        return 1
    else
        echo "🎉 MOD Optimization (Fusion Model Logic) completed for $current_dataset."
        return 0
    fi
}

if ! python3 -c "import optuna" &> /dev/null; then
    echo "❌ Optuna not installed. Installing..."
    pip install optuna
fi

# Determine which datasets to process
if [ "${TARGET_DATASETS[0]}" == "all" ]; then
    # Use all available datasets (excluding excluded ones)
    DATASETS_TO_PROCESS=("${ALL_DATASETS[@]}")
    echo "=================================================="
    echo "🌟 Batch Mode: Running MOD optimization (Fusion Model Logic) for ALL datasets"
    echo "   Datasets: ${ALL_DATASETS[*]}"
    if [ ${#EXCLUDE_DATASETS[@]} -gt 0 ]; then
        echo "   Excluded: ${EXCLUDE_DATASETS[*]}"
    fi
    echo "=================================================="
else
    # Use specified datasets, but filter out excluded ones
    if [ ${#EXCLUDE_DATASETS[@]} -gt 0 ]; then
        FILTERED_TARGETS=()
        for dataset in "${TARGET_DATASETS[@]}"; do
            exclude=false
            for excluded in "${EXCLUDE_DATASETS[@]}"; do
                if [ "$dataset" == "$excluded" ]; then
                    exclude=true
                    break
                fi
            done
            if [ "$exclude" == false ]; then
                FILTERED_TARGETS+=("$dataset")
            fi
        done
        DATASETS_TO_PROCESS=("${FILTERED_TARGETS[@]}")
    else
        DATASETS_TO_PROCESS=("${TARGET_DATASETS[@]}")
    fi
    
    echo "=================================================="
    echo "🌟 Batch Mode: Running MOD optimization (Fusion Model Logic) for specified datasets"
    echo "   Datasets: ${TARGET_DATASETS[*]}"
    if [ ${#EXCLUDE_DATASETS[@]} -gt 0 ]; then
        echo "   Excluded: ${EXCLUDE_DATASETS[*]}"
        if [ ${#DATASETS_TO_PROCESS[@]} -lt ${#TARGET_DATASETS[@]} ]; then
            echo "   After exclusion: ${DATASETS_TO_PROCESS[*]}"
        fi
    fi
    echo "=================================================="
fi

# Track success/failure for final report
SUCCESSFUL_DATASETS=()
FAILED_DATASETS=()
SKIPPED_DATASETS=()

for dataset in "${DATASETS_TO_PROCESS[@]}"; do
    # Check if TDC dataset exists (has seed1-5 directories with train.pt, valid.pt, test.pt)
    TDC_DATASET_PATH="$TDC_DATA_DIR/$dataset"
    if [ -d "$TDC_DATASET_PATH" ]; then
        # Check if at least one seed directory exists with required files
        SEED_FOUND=false
        for seed in seed1 seed2 seed3 seed4 seed5; do
            if [ -d "$TDC_DATASET_PATH/$seed" ] && \
               [ -f "$TDC_DATASET_PATH/$seed/train.pt" ] && \
               [ -f "$TDC_DATASET_PATH/$seed/valid.pt" ] && \
               [ -f "$TDC_DATASET_PATH/$seed/test.pt" ]; then
                SEED_FOUND=true
                break
            fi
        done
        
        if [ "$SEED_FOUND" = true ]; then
            # NEW: Single optimization run (no per-seed optimization)
            # Each trial trains all 5 seeds with the same hyperparameters
            echo "🔀 Running optimization (Fusion Model Logic) for $dataset..."
            echo "   Each trial trains all 5 seeds with the same hyperparameters"
            
            run_optimization "$dataset"
            exit_code=$?
            if [ $exit_code -eq 0 ]; then
                # Successfully completed optimization
                SUCCESSFUL_DATASETS+=("$dataset")
                echo "😴 Cooling down for 30 seconds..."
                sleep 30
            elif [ $exit_code -eq 2 ]; then
                # Skip (target trials reached), no cooldown needed
                SKIPPED_DATASETS+=("$dataset")
                echo "⏭️  Skipped $dataset (target already reached, no cooldown needed)"
            else
                # Failed
                FAILED_DATASETS+=("$dataset")
                echo "😴 Cooling down for 30 seconds..."
                sleep 30
            fi
        else
            echo "⚠️  Skipping $dataset (TDC dataset found but missing required seed files)"
            SKIPPED_DATASETS+=("$dataset")
        fi
    elif [ -f "data/${dataset}_dataset.csv" ] || [ -f "data/processed/${dataset}_processed.pkl" ]; then
        # Fallback to old format
        run_optimization "$dataset"
        exit_code=$?
        if [ $exit_code -eq 0 ]; then
            # Successfully completed optimization
            SUCCESSFUL_DATASETS+=("$dataset")
            echo "😴 Cooling down for 30 seconds..."
            sleep 30
        elif [ $exit_code -eq 2 ]; then
            # Skip (target trials reached), no cooldown needed
            SKIPPED_DATASETS+=("$dataset")
            echo "⏭️  Skipped $dataset (target already reached, no cooldown needed)"
        else
            # Failed
            FAILED_DATASETS+=("$dataset")
            echo "😴 Cooling down for 30 seconds..."
            sleep 30
        fi
    else
        echo "⚠️  Skipping $dataset (Dataset not found)"
        SKIPPED_DATASETS+=("$dataset")
    fi
done

echo "=================================================="
echo "🎊 Batch optimization (Fusion Model Logic) completed!"
echo "=================================================="
echo "✅ Successful: ${#SUCCESSFUL_DATASETS[@]} dataset(s)"
if [ ${#SUCCESSFUL_DATASETS[@]} -gt 0 ]; then
    echo "   ${SUCCESSFUL_DATASETS[*]}"
fi
if [ ${#FAILED_DATASETS[@]} -gt 0 ]; then
    echo "❌ Failed: ${#FAILED_DATASETS[@]} dataset(s)"
    echo "   ${FAILED_DATASETS[*]}"
fi
if [ ${#SKIPPED_DATASETS[@]} -gt 0 ]; then
    echo "⚠️  Skipped: ${#SKIPPED_DATASETS[@]} dataset(s)"
    echo "   ${SKIPPED_DATASETS[*]}"
fi
echo "=================================================="


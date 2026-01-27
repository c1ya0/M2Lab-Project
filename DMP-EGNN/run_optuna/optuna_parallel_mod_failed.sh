#!/bin/bash

# Optuna Parallel Optimization Launcher (MOD NEW version for AEGNN-M with DMPNN)
# Uses optuna_serach_mod_new.py and train_edmpnn_new.py with improved model architecture
# 
# IMPROVEMENTS:
# - Uses RobustScaler for descriptor normalization (via train_edmpnn_new.py)
# - Dynamic DMP steps based on dataset size
# - Dynamic model depth/width based on dataset size and class imbalance
# - Optimized attention heads to ensure divisibility with hidden_dim
#
# Usage: ./optuna_parallel_mod_new.sh [dataset1] [dataset2 ...] [mode: fast/standard/deep] [--exclude dataset1 dataset2 ...] [--workers N]
#        ./optuna_parallel_mod_new.sh all [mode] [--exclude dataset1 dataset2 ...] [--workers N]
# Supports multiple datasets: ./optuna_parallel_mod_new.sh herg ames bace standard
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

# Activate conda environment
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

# Smart parameter parsing: handle --exclude as first argument and support multiple datasets
TARGET_DATASETS=()  # Array to store target datasets (can be multiple)
mode="standard"
EXCLUDE_DATASETS=()
MANUAL_WORKERS=""  # Manually set worker count (optional)

# Check if first argument is --exclude
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
        else
            # This is a dataset name
            TARGET_DATASETS+=("$1")
            shift
        fi
    done
    
    # Continue parsing remaining options (--exclude, --workers)
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

# Define supported dataset list (TDC datasets from processed_tdc_data)
# Auto-detect TDC datasets from processed_tdc_data directory
TDC_DATA_DIR="data/processed_tdc_data"
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
    echo "Usage: ./optuna_parallel_mod.sh <dataset1> [dataset2 ...] [mode] [--exclude dataset1 dataset2 ...] [--workers N]"
    echo "       ./optuna_parallel_mod.sh all [mode] [--exclude dataset1 dataset2 ...] [--workers N]"
    echo ""
    echo "Modes: fast (default), standard, deep"
    echo ""
    echo "Options:"
    echo "  --workers N    Manually set the number of parallel workers (default: auto-calculated)"
    echo "                 Useful for avoiding OOM errors by reducing parallelism"
    echo ""
    echo "Examples:"
    echo "  # Single dataset:"
    echo "  ./optuna_parallel_mod.sh caco2_wang standard"
    echo ""
    echo "  # Multiple datasets:"
    echo "  ./optuna_parallel_mod.sh herg ames bace standard"
    echo "  ./optuna_parallel_mod.sh dataset1 dataset2 dataset3 --workers 5"
    echo ""
    echo "  # All datasets:"
    echo "  ./optuna_parallel_mod.sh all standard"
    echo "  ./optuna_parallel_mod.sh all standard --exclude hiv muv"
    echo ""
    echo "  # Using environment variable:"
    echo "  OPTUNA_WORKERS=5 ./optuna_parallel_mod.sh caco2_wang standard"
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
# 2. Auto-calculate Workers (parallelism)
# ==========================================
if [ -n "$MANUAL_WORKERS" ]; then
    # Use manually set worker count
    WORKERS=$MANUAL_WORKERS
    if [ "$WORKERS" -lt 1 ]; then
        echo "❌ Error: Worker count must be at least 1"
        exit 1
    fi
    echo "💡 Using manually specified workers: $WORKERS"
else
    # Auto-calculate worker count
    if [ "$GPU_COUNT" -gt 0 ]; then
        MAX_GPU_WORKERS=$((GPU_MEM_TOTAL / 1500))
        MAX_GPU_WORKERS=$((MAX_GPU_WORKERS * GPU_COUNT))
        MAX_CPU_WORKERS=$(( (CPU_CORES - 2) / 2 ))
        if [ "$MAX_CPU_WORKERS" -lt 1 ]; then MAX_CPU_WORKERS=1; fi

        WORKERS=$(( MAX_GPU_WORKERS < MAX_CPU_WORKERS ? MAX_GPU_WORKERS : MAX_CPU_WORKERS ))

        # GPU Allocation Strategy (Parallel Seeds): All 5 seeds run in parallel
        # Each seed has its own independent Optuna study and runs in parallel:
        #   - GPU 0: seeds 1, 2 (2 single-GPU processes in parallel)
        #   - GPU 1: seeds 3, 4, 5 (3 single-GPU processes in parallel)
        # Total: 5 GPU processes running simultaneously (one per seed)
        # 
        # With this strategy, we use WORKERS=1 per seed to process one trial at a time per seed
        # This allows all 5 seeds to run in parallel while utilizing both GPUs efficiently
        
        # Check GPU memory usage for informational purposes
        if command -v nvidia-smi &> /dev/null; then
            GPU_MEM_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
            GPU_MEM_TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
            GPU_MEM_AVAILABLE=$((GPU_MEM_TOTAL - GPU_MEM_USED))
            echo "   ✅ GPU available memory: ${GPU_MEM_AVAILABLE} MiB"
        fi

        # Parallel Seeds: Force WORKERS=1 per seed (one trial at a time per seed)
        # This ensures each seed processes one trial at a time, but all 5 seeds run in parallel
        # Total: 5 GPU processes running simultaneously (one per seed)
        DEFAULT_MAX_WORKERS=1  # One trial at a time per seed
        WORKERS=1  # Force to 1 per seed for parallel execution

        echo "💡 Auto-Config (GPU Mode, Parallel Seeds): Using $WORKERS worker per seed"
        echo "   GPU Distribution: GPU 0 → seeds [1,2], GPU 1 → seeds [3,4,5]"
        echo "   Execution Mode: All 5 seeds run in parallel"
        echo "   Max concurrent GPU processes: 5 (one per seed, parallel execution)"
    else
        # CPU mode: still use 1 worker per seed for consistency with GPU mode (Parallel Seeds)
        WORKERS=1
        DEFAULT_MAX_WORKERS=1  # One trial at a time per seed
        echo "💡 Auto-Config (CPU Mode, Parallel Seeds): Using $WORKERS worker per seed (all seeds in parallel)"
    fi
fi

# ==========================================
# 3. Set Trials (based on mode)
# ==========================================
case $mode in
    fast)
        TOTAL_TRIALS=20
        EPOCHS=75  # Reduced to decrease CPU load and overheating risk
        # Calculation: 20 trials × avg 165 min/trial (5 seeds per trial, ~33 min per seed) ÷ 2 workers = 1650 min ≈ 27.5 hours
        # Plus buffer: 27.5 × 1.5 = 41.25 hours
        MAX_WAIT_TIME=148500  # Approx 41.25 hours (adjusted for 5 seeds per trial)
        ;;
    standard)
        TOTAL_TRIALS=70
        EPOCHS=125  # Reduced from 150 to 125 to decrease training time
        # Calculation: 70 trials × avg 275 min/trial (5 seeds per trial, ~55 min per seed) ÷ 2 workers ≈ 9625 min ≈ 160.4 hours
        # Plus buffer: 160.4 × 1.5 ≈ 240 hours (more conservative buffer due to longer training time)
        MAX_WAIT_TIME=864000  # Approx 240 hours (adjusted for 5 seeds per trial)
        ;;
    deep)
        TOTAL_TRIALS=160
        EPOCHS=250  # Reduced to decrease CPU load and overheating risk
        # Calculation: 160 trials × avg 550 min/trial (5 seeds per trial, ~110 min per seed) ÷ 2 workers = 44000 min ≈ 733.3 hours
        # Plus buffer: 733.3 × 1.5 = 1100 hours
        MAX_WAIT_TIME=3960000  # Approx 1100 hours (adjusted for 5 seeds per trial)
        ;;
    *)
        echo "Unknown mode: $mode. Using standard."
        TOTAL_TRIALS=50
        EPOCHS=150  # Reduced from 200 to 150 to decrease training time
        # Calculation: 50 trials × avg 330 min/trial (5 seeds per trial, ~66 min per seed) ÷ 2 workers = 8250 min ≈ 137.5 hours
        # Plus buffer: 137.5 × 1.5 = 206.25 hours
        MAX_WAIT_TIME=742500  # Approx 206.25 hours (adjusted for 5 seeds per trial)
        ;;
esac

# More precise trial distribution to avoid exceeding TOTAL_TRIALS
# Distribute trials evenly, with remainder going to last worker
BASE_TRIALS_PER_WORKER=$((TOTAL_TRIALS / WORKERS))
REMAINDER_TRIALS=$((TOTAL_TRIALS % WORKERS))

echo "🎯 Goal: $TOTAL_TRIALS trials total ($mode mode)"
if [ $REMAINDER_TRIALS -eq 0 ]; then
    echo "   Configuration: $WORKERS worker × $BASE_TRIALS_PER_WORKER trials"
else
    echo "   Configuration: $WORKERS worker × $((BASE_TRIALS_PER_WORKER + REMAINDER_TRIALS)) trials"
fi
echo "   📊 GPU Allocation (Parallel Seeds):"
echo "      - All 5 seeds run in parallel"
echo "      - GPU 0: seeds 1, 2 (2 single-GPU processes in parallel)"
echo "      - GPU 1: seeds 3, 4, 5 (3 single-GPU processes in parallel)"
echo "      - Max concurrent GPU processes: 5 (one per seed, all parallel)"
echo "      - Memory efficient: No DDP overhead, single GPU per seed"
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
# 5. Execute optimization function (using optuna_serach_mod_new.py)
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
    REMAINING_PIDS=$(pgrep -f "optuna_serach_mod_new.py" 2>/dev/null || true)
    if [ -n "$REMAINING_PIDS" ]; then
        echo "   Found remaining processes: $REMAINING_PIDS"
        for pid in $REMAINING_PIDS; do
            # Kill process and all its children
            pkill -KILL -P "$pid" 2>/dev/null || true
            kill -KILL "$pid" 2>/dev/null || true
        done
    fi
    
    # Method 4: Kill any remaining training processes (train_edmpnn_new.py)
    echo "   Searching for remaining training processes..."
    TRAINING_PIDS=$(pgrep -f "train_edmpnn_new.py" 2>/dev/null || true)
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
    
    # Clean up temporary files created during parallel seed execution
    echo "   Cleaning up temporary files..."
    rm -f /tmp/optuna_seed_*.pids /tmp/optuna_seed_*.exit 2>/dev/null || true
    
    echo "✅ Cleanup completed."
    exit 130
}

# Set up signal handlers
trap cleanup_workers SIGINT SIGTERM

run_optimization_for_seed() {
    local current_dataset=$1
    local seed_num=$2
    
    # Local arrays for this seed's workers
    local -a pids=()
    local pgid=""
    # Update global PIDs array for cleanup
    CURRENT_PIDS=()

    echo "-------------------------------------------"
    echo "🚀 Launching MOD Optimization for dataset: $current_dataset, seed: $seed_num"
    STORAGE_URL="sqlite:///optuna_edmpnn_results_new/optuna_mod_new.db"
    echo "   Storage: $STORAGE_URL"
    echo "   Logs: logs/optuna_launcher_mod_new/"

    mkdir -p logs/optuna_launcher_mod_new
    mkdir -p optuna_edmpnn_results_new

    # Study name for per-seed optimization
    STUDY_NAME="edmpnn_mod_new_${current_dataset}_seed${seed_num}_opt"
    echo "   Study: $STUDY_NAME"

    # Check existing trials for this seed-specific study
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
    
    # Calculate remaining trials needed
    # Only count completed and failed trials (running trials will continue)
    EXISTING_FINISHED=$((EXISTING_COMPLETED + EXISTING_FAILED))
    REMAINING_TRIALS=$((TOTAL_TRIALS - EXISTING_FINISHED))
    
    if [ $REMAINING_TRIALS -le 0 ]; then
        echo "   ✅ Target of $TOTAL_TRIALS trials already reached for seed $seed_num ($EXISTING_FINISHED finished trials: $EXISTING_COMPLETED completed + $EXISTING_FAILED failed)"
        return 2  # Return 2 indicates skip (no cooldown needed)
    fi
    
    echo "   🎯 Remaining trials needed: $REMAINING_TRIALS (target: $TOTAL_TRIALS, existing: $EXISTING_TOTAL)"
    
    # Recalculate trials per worker based on remaining trials
    BASE_TRIALS_PER_WORKER=$((REMAINING_TRIALS / WORKERS))
    REMAINDER_TRIALS=$((REMAINING_TRIALS % WORKERS))
    
    if [ $REMAINDER_TRIALS -eq 0 ]; then
        echo "   📋 Distribution: $WORKERS workers × $BASE_TRIALS_PER_WORKER trials each"
    else
        echo "   📋 Distribution: $((WORKERS - 1)) workers × $BASE_TRIALS_PER_WORKER trials, 1 worker × $((BASE_TRIALS_PER_WORKER + REMAINDER_TRIALS)) trials"
    fi

    # Start workers for this seed
    WORKERS_STARTED=0
    for i in $(seq 1 $WORKERS); do
        if [ $i -eq $WORKERS ] && [ $REMAINDER_TRIALS -gt 0 ]; then
            WORKER_TRIALS=$((BASE_TRIALS_PER_WORKER + REMAINDER_TRIALS))
        else
            WORKER_TRIALS=$BASE_TRIALS_PER_WORKER
        fi
        
        if [ $WORKER_TRIALS -le 0 ]; then
            continue
        fi
        
        WORKERS_STARTED=$((WORKERS_STARTED + 1))
        echo "   Starting Worker $i for seed $seed_num (${WORKER_TRIALS} trials)..."
        
        if [ "$i" -eq 1 ]; then
            # For worker 1, use tee to show output in terminal
            # Start Python process in background and capture its PID
            python3 scripts/optuna_serach_mod_new.py \
                --dataset "$current_dataset" \
                --n_trials "$WORKER_TRIALS" \
                --storage "$STORAGE_URL" \
                --epochs "$EPOCHS" \
                --worker_id "$i" \
                --seed "$seed_num" \
                2>&1 | tee "logs/optuna_launcher_mod_new/worker_${current_dataset}_seed${seed_num}_${i}.log" &
            # Get the Python process PID (not the tee PID)
            # The $! gives us the PID of the last command in the pipeline (tee)
            # We need to find the Python process
            pipeline_pid=$!
            # Wait a moment for process to start, then find the actual Python PID
            sleep 0.5
            # Find Python process by matching command line arguments
            # Fix: Include --worker_id in match to accurately identify the correct worker process
                    python_pid=$(pgrep -f "optuna_serach_mod_new.py.*--dataset.*$current_dataset.*--seed.*$seed_num.*--worker_id.*$i" | grep -v grep | head -1)
            if [ -n "$python_pid" ]; then
                worker_pid=$python_pid
            else
                # Fallback: try without worker_id (for backward compatibility)
                python_pid=$(pgrep -f "optuna_serach_mod_new.py.*--dataset.*$current_dataset.*--seed.*$seed_num" | grep -v grep | head -1)
                if [ -n "$python_pid" ]; then
                    worker_pid=$python_pid
                else
                    # Final fallback: use pipeline PID (may be tee, but will work)
                    # But we'll handle this in the wait loop
                    worker_pid=$pipeline_pid
                fi
            fi
            if [ -z "$pgid" ]; then
                pgid=$(ps -o pgid= -p "$worker_pid" 2>/dev/null | tr -d ' ')
                CURRENT_PGID="$pgid"
            fi
            
            # Fix: Verify process started successfully
            sleep 1  # Give process time to start
            if ! kill -0 "$worker_pid" 2>/dev/null; then
                # Process already died, check log for errors
                if [ -f "logs/optuna_launcher_mod_new/worker_${current_dataset}_seed${seed_num}_${i}.log" ]; then
                    if grep -q "Error\|Traceback\|Exception" "logs/optuna_launcher_mod_new/worker_${current_dataset}_seed${seed_num}_${i}.log" 2>/dev/null; then
                        echo "   ⚠️  Worker $i failed to start (check log for errors)"
                        continue  # Skip this worker
                    fi
                fi
            fi
        else
            setsid python3 scripts/optuna_serach_mod_new.py \
                --dataset "$current_dataset" \
                --n_trials "$WORKER_TRIALS" \
                --storage "$STORAGE_URL" \
                --epochs "$EPOCHS" \
                --worker_id "$i" \
                --seed "$seed_num" \
                > "logs/optuna_launcher_mod_new/worker_${current_dataset}_seed${seed_num}_${i}.log" 2>&1 &
            local worker_pid=$!
            
            # Fix: Verify process started successfully
            sleep 1  # Give process time to start
            if ! kill -0 "$worker_pid" 2>/dev/null; then
                # Process already died, check log for errors
                if [ -f "logs/optuna_launcher_mod_new/worker_${current_dataset}_seed${seed_num}_${i}.log" ]; then
                    if grep -q "Error\|Traceback\|Exception" "logs/optuna_launcher_mod_new/worker_${current_dataset}_seed${seed_num}_${i}.log" 2>/dev/null; then
                        echo "   ⚠️  Worker $i failed to start (check log for errors)"
                        continue  # Skip this worker
                    fi
                fi
            fi
        fi
        
        pids+=($worker_pid)
        CURRENT_PIDS+=($worker_pid)
    done
    
    if [ $WORKERS_STARTED -eq 0 ]; then
        return 0
    fi
    
    echo "✅ Started $WORKERS_STARTED worker(s) for seed $seed_num. PIDs: ${pids[*]}"
    echo "Waiting for completion..."
    
    # Wait for all workers
    # IMPORTANT: We must wait for the actual Python processes, not tee/shell processes
    # For worker 1, the PID might be tee, so we need to find and wait for the Python process
    for pid in "${pids[@]}"; do
        # Check if this is a Python process or a shell/tee process
        cmd=$(ps -p "$pid" -o cmd= 2>/dev/null | head -1)
        if echo "$cmd" | grep -q "tee\|bash"; then
            # This is a shell/tee process, find the actual Python child process
            python_pid=$(pgrep -P "$pid" | head -1)
            if [ -n "$python_pid" ]; then
                # Wait for the Python process (this will block until it completes)
                wait "$python_pid"
            else
                # If we can't find Python child, wait for the original PID
                # But this might be tee which exits early, so also check for Python processes
                # Fix: Try to find Python process by worker_id if available (more accurate)
                # Note: We don't have worker_id here, so use seed matching
                python_pid=$(pgrep -f "optuna_serach_mod_new.py.*--dataset.*$current_dataset.*--seed.*$seed_num" | grep -v grep | head -1)
                if [ -n "$python_pid" ]; then
                    wait "$python_pid"
                else
                    wait "$pid"
                fi
            fi
        else
            # This should be the Python process itself
            # Double-check it's actually running
            if kill -0 "$pid" 2>/dev/null; then
                wait "$pid"
            else
                # Process already finished, but we should still wait to get exit code
                wait "$pid" 2>/dev/null || true
            fi
        fi
    done
    
    echo "🎉 Seed $seed_num optimization completed for $current_dataset."
    return 0
}

run_optimization() {
    local current_dataset=$1
    # Local arrays for this dataset's workers
    local -a pids=()
    local pgid=""
    # Update global PIDs array for cleanup
    CURRENT_PIDS=()

    echo "-------------------------------------------"
    echo "🚀 Launching MOD Optimization for dataset: $current_dataset"
    STORAGE_URL="sqlite:///optuna_edmpnn_results_new/optuna_mod_new.db"
    echo "   Storage: $STORAGE_URL"
    echo "   Logs: logs/optuna_launcher_mod_new/"

    mkdir -p logs/optuna_launcher_mod_new
    mkdir -p optuna_edmpnn_results_new

    # Note: Study initialization is now handled by optuna_serach_mod_new.py
    # This ensures consistent direction calculation and better error handling
    # The Python script will handle all study creation/loading with proper retry mechanisms
    echo "   Study will be initialized by optuna_serach_mod_new.py with proper direction detection"

    # Check existing trials and calculate remaining trials needed
    # Note: Study may not exist yet (will be created by optuna_serach_mod_new.py)
    echo "   Checking existing trials in study..."
    EXISTING_TRIALS_INFO=$(python3 -c "
import optuna
try:
    # Try to load existing study (will fail if study doesn't exist)
    study = optuna.load_study(study_name='edmpnn_mod_new_${current_dataset}_opt', storage='$STORAGE_URL')
    total = len(study.trials)
    completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    failed = len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])
    running = len([t for t in study.trials if t.state == optuna.trial.TrialState.RUNNING])
    print(f'{total},{completed},{failed},{running}', end='')
except Exception as e:
    # Study doesn't exist yet, will be created by optuna_serach_mod_new.py
    # This is expected for new datasets
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
    
    # Calculate remaining trials needed
    # Only count completed and failed trials (running trials will continue)
    EXISTING_FINISHED=$((EXISTING_COMPLETED + EXISTING_FAILED))
    REMAINING_TRIALS=$((TOTAL_TRIALS - EXISTING_FINISHED))
    
    if [ $REMAINING_TRIALS -le 0 ]; then
        echo "   ✅ Target of $TOTAL_TRIALS trials already reached ($EXISTING_FINISHED finished trials: $EXISTING_COMPLETED completed + $EXISTING_FAILED failed)"
        echo "   Skipping optimization (no new trials needed)"
        return 2  # Return 2 indicates skip (no cooldown needed)
    fi
    
    echo "   🎯 Remaining trials needed: $REMAINING_TRIALS (target: $TOTAL_TRIALS, existing: $EXISTING_TOTAL)"
    
    # Recalculate trials per worker based on remaining trials
    BASE_TRIALS_PER_WORKER=$((REMAINING_TRIALS / WORKERS))
    REMAINDER_TRIALS=$((REMAINING_TRIALS % WORKERS))
    
    if [ $REMAINDER_TRIALS -eq 0 ]; then
        echo "   📋 Distribution: $WORKERS workers × $BASE_TRIALS_PER_WORKER trials each"
    else
        echo "   📋 Distribution: $((WORKERS - 1)) workers × $BASE_TRIALS_PER_WORKER trials, 1 worker × $((BASE_TRIALS_PER_WORKER + REMAINDER_TRIALS)) trials"
    fi

    # Start workers in a new process group for better signal handling
    # Use setsid to create new session and process group
    # Calculate trials per worker with precise distribution
    WORKERS_STARTED=0
    for i in $(seq 1 $WORKERS); do
        # Calculate trials for this worker: last worker gets remainder
        if [ $i -eq $WORKERS ] && [ $REMAINDER_TRIALS -gt 0 ]; then
            WORKER_TRIALS=$((BASE_TRIALS_PER_WORKER + REMAINDER_TRIALS))
        else
            WORKER_TRIALS=$BASE_TRIALS_PER_WORKER
        fi
        
        # Skip worker if no trials assigned
        if [ $WORKER_TRIALS -le 0 ]; then
            echo "   ⏭️  Skipping Worker $i (no trials assigned)"
            continue
        fi
        
        WORKERS_STARTED=$((WORKERS_STARTED + 1))
        echo "   Starting Worker $i (${WORKER_TRIALS} trials)..."

        if [ "$i" -eq 1 ]; then
            # First worker: use setsid to create new process group, store PGID
            setsid python3 scripts/optuna_serach_mod_new.py \
                --dataset "$current_dataset" \
                --n_trials "$WORKER_TRIALS" \
                --storage "$STORAGE_URL" \
                --epochs "$EPOCHS" \
                --worker_id "$i" \
                2>&1 | tee "logs/optuna_launcher_mod_new/worker_${current_dataset}_${i}.log" &
            local worker_pid=$!
            # Get process group ID from first worker
            if [ -z "$pgid" ]; then
                pgid=$(ps -o pgid= -p "$worker_pid" 2>/dev/null | tr -d ' ')
                CURRENT_PGID="$pgid"  # Update global for cleanup
            fi
            
            # Fix: Verify process started successfully
            sleep 1  # Give process time to start
            if ! kill -0 "$worker_pid" 2>/dev/null; then
                # Process already died, check log for errors
                if [ -f "logs/optuna_launcher_mod_new/worker_${current_dataset}_${i}.log" ]; then
                    if grep -q "Error\|Traceback\|Exception" "logs/optuna_launcher_mod_new/worker_${current_dataset}_${i}.log" 2>/dev/null; then
                        echo "   ⚠️  Worker $i failed to start (check log for errors)"
                        continue  # Skip this worker
                    fi
                fi
            fi
        else
            # Other workers: start in same process group
            setsid python3 scripts/optuna_serach_mod_new.py \
                --dataset "$current_dataset" \
                --n_trials "$WORKER_TRIALS" \
                --storage "$STORAGE_URL" \
                --epochs "$EPOCHS" \
                --worker_id "$i" \
                > "logs/optuna_launcher_mod_new/worker_${current_dataset}_${i}.log" 2>&1 &
            local worker_pid=$!
            
            # Fix: Verify process started successfully
            sleep 1  # Give process time to start
            if ! kill -0 "$worker_pid" 2>/dev/null; then
                # Process already died, check log for errors
                if [ -f "logs/optuna_launcher_mod_new/worker_${current_dataset}_${i}.log" ]; then
                    if grep -q "Error\|Traceback\|Exception" "logs/optuna_launcher_mod_new/worker_${current_dataset}_${i}.log" 2>/dev/null; then
                        echo "   ⚠️  Worker $i failed to start (check log for errors)"
                        continue  # Skip this worker
                    fi
                fi
            fi
        fi
        
        pids+=($worker_pid)
        CURRENT_PIDS+=($worker_pid)  # Add to global array for cleanup
    done

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
            
            # ==========================================
            # Actual monitoring: Check if training is really running
            # ==========================================
            ACTUAL_MONITORING_WARNINGS=()
            
            # Give training processes startup time (don't check actual monitoring for first 2 minutes)
            STARTUP_GRACE_PERIOD=120  # 2 minutes
            
            # Only perform actual monitoring checks after startup period
            if [ $ELAPSED -gt $STARTUP_GRACE_PERIOD ]; then
                # 1. Check training processes
                TRAINING_PROCESSES=$(ps aux | grep -E "train_edmpnn\.py" | grep -v grep | wc -l)
                if [ "$TRAINING_PROCESSES" -eq 0 ]; then
                    ACTUAL_MONITORING_WARNINGS+=("No training processes")
                fi
            
                # 2. Check GPU utilization (if GPU available and training processes running)
                if command -v nvidia-smi &> /dev/null && [ "$TRAINING_PROCESSES" -gt 0 ]; then
                    # Get all GPU utilization rates, take maximum (training may use multiple GPUs)
                    GPU_UTILS=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' | tr '\n' ' ')
                    if [ -n "$GPU_UTILS" ]; then
                        # Calculate maximum GPU utilization
                        MAX_GPU_UTIL=0
                        for util in $GPU_UTILS; do
                            if [ -n "$util" ] && [ "$util" -gt "$MAX_GPU_UTIL" ] 2>/dev/null; then
                                MAX_GPU_UTIL=$util
                            fi
                        done
                        
                        # If maximum GPU utilization < 10%, issue warning
                        if [ "$MAX_GPU_UTIL" -lt 10 ]; then
                            ACTUAL_MONITORING_WARNINGS+=("GPU util: ${MAX_GPU_UTIL}%")
                        fi
                    fi
                fi
                
                # 3. Check progress file update time (if training processes running)
                if [ "$TRAINING_PROCESSES" -gt 0 ]; then
                    # New structure: checkpoints/optuna_mod/{dataset}/seed{seed}/opt/{trial}/
                    # Check all seeds (1-5)
                    PROGRESS_DIR="checkpoints/optuna_mod/${current_dataset}"
                    LATEST_PROGRESS=""
                    for seed in 1 2 3 4 5; do
                        seed_progress_dir="${PROGRESS_DIR}/seed${seed}/opt"
                        if [ -d "$seed_progress_dir" ]; then
                            seed_latest=$(find "$seed_progress_dir" -name "training_progress.json" \
                                -exec stat -c "%Y" {} \; 2>/dev/null | sort -rn | head -1)
                            if [ -n "$seed_latest" ]; then
                                if [ -z "$LATEST_PROGRESS" ] || [ "$seed_latest" -gt "$LATEST_PROGRESS" ]; then
                                    LATEST_PROGRESS="$seed_latest"
                                fi
                            fi
                        fi
                    done
                    # Fallback to old structure for backward compatibility
                    if [ -z "$LATEST_PROGRESS" ] && [ -d "$PROGRESS_DIR" ]; then
                        LATEST_PROGRESS=$(find "$PROGRESS_DIR" -name "training_progress.json" \
                            -exec stat -c "%Y" {} \; 2>/dev/null | sort -rn | head -1)
                    fi
                    if [ -n "$LATEST_PROGRESS" ]; then
                        TIME_SINCE_UPDATE=$((CURRENT_TIME - LATEST_PROGRESS))
                        # If no update for more than 10 minutes, issue warning
                        if [ $TIME_SINCE_UPDATE -gt 600 ]; then
                            ACTUAL_MONITORING_WARNINGS+=("No progress for ${TIME_SINCE_UPDATE}s")
                        fi
                    fi
                fi
            fi
            # Don't perform actual monitoring checks during startup period (avoid false alarms)
            
            # If there are warnings, log but don't display immediately (avoid interrupting progress bar)
            # Warnings will be displayed together with progress bar update
            ACTUAL_MONITORING_WARNING_MSG=""
            if [ ${#ACTUAL_MONITORING_WARNINGS[@]} -gt 0 ]; then
                ACTUAL_MONITORING_WARNING_MSG=" | ⚠️ ${ACTUAL_MONITORING_WARNINGS[*]}"
            fi
            
            # Try to get trial count from Optuna database
            TRIAL_INFO=$(python3 -c "
import optuna
try:
    study = optuna.load_study(study_name='edmpnn_mod_new_${current_dataset}_opt', storage='$STORAGE_URL')
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
            # Check if we have any progress to display
            # Fix: Check TOTAL_TRIALS > 0 to avoid division by zero
            if [ -n "$COMPLETED_TRIALS" ] && [ "$COMPLETED_TRIALS" != "0" ] || ([ -n "$TOTAL_TRIALS_IN_DB" ] && [ "$TOTAL_TRIALS_IN_DB" != "0" ]); then
                # Ensure TOTAL_TRIALS > 0 before division to avoid division by zero error
                if [ "$TOTAL_TRIALS" -gt 0 ]; then
                    PROGRESS_PCT=$((COMPLETED_TRIALS * 100 / TOTAL_TRIALS))
                    ELAPSED_MIN=$((ELAPSED / 60))
                    
                    # Calculate progress bar (50 characters wide)
                    BAR_WIDTH=50
                    FILLED=$((COMPLETED_TRIALS * BAR_WIDTH / TOTAL_TRIALS))
                    EMPTY=$((BAR_WIDTH - FILLED))
                else
                    # Fallback: use TOTAL_TRIALS_IN_DB if TOTAL_TRIALS is 0
                    if [ -n "$TOTAL_TRIALS_IN_DB" ] && [ "$TOTAL_TRIALS_IN_DB" -gt 0 ]; then
                        PROGRESS_PCT=$((COMPLETED_TRIALS * 100 / TOTAL_TRIALS_IN_DB))
                        ELAPSED_MIN=$((ELAPSED / 60))
                        BAR_WIDTH=50
                        FILLED=$((COMPLETED_TRIALS * BAR_WIDTH / TOTAL_TRIALS_IN_DB))
                        EMPTY=$((BAR_WIDTH - FILLED))
                    else
                        # Both are 0 or invalid, skip progress calculation for this iteration
                        # Don't use 'continue' here as we're not in a loop, just skip the progress bar update
                        PROGRESS_PCT=0
                        ELAPSED_MIN=$((ELAPSED / 60))
                        BAR_WIDTH=50
                        FILLED=0
                        EMPTY=$BAR_WIDTH
                    fi
                fi
                
                # Build progress bar
                BAR=""
                for i in $(seq 1 $FILLED); do
                    BAR="${BAR}█"
                done
                for i in $(seq 1 $EMPTY); do
                    BAR="${BAR}░"
                done
                
                # Print progress bar on the same line (using \r to return to start)
                # Show completed/total, and include failed/running info if available
                # Also include actual monitoring warnings if any
                if [ "$FAILED_TRIALS" != "0" ] || [ "$RUNNING_TRIALS" != "0" ]; then
                    printf "\r   📊 Progress: [%s] %d/%d (%d%%) | Completed: %d, Failed: %d, Running: %d | Elapsed: %dmin%s    " \
                        "$BAR" "$COMPLETED_TRIALS" "$TOTAL_TRIALS" "$PROGRESS_PCT" "$COMPLETED_TRIALS" "$FAILED_TRIALS" "$RUNNING_TRIALS" "$ELAPSED_MIN" "$ACTUAL_MONITORING_WARNING_MSG"
                else
                    printf "\r   📊 Progress: [%s] %d/%d (%d%%) | Elapsed: %dmin%s    " \
                        "$BAR" "$COMPLETED_TRIALS" "$TOTAL_TRIALS" "$PROGRESS_PCT" "$ELAPSED_MIN" "$ACTUAL_MONITORING_WARNING_MSG"
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
        echo "🎉 MOD Optimization completed for $current_dataset."
        return 0
    fi
    
    # Clear PIDs after completion
    pids=()
}

# Verify critical dependencies before starting
echo "🔍 Verifying critical dependencies..."
if ! python3 -c "import optuna" &> /dev/null; then
    echo "❌ Optuna not installed. Installing..."
    pip install optuna
fi

# Verify PyTorch and torch-geometric can be imported (critical check)
if ! python3 -c "import torch; import torch_geometric" &> /dev/null; then
    echo "❌ Critical error: PyTorch or torch-geometric import failed!"
    echo "   This will cause all trials to fail immediately."
    echo "   Please check your conda environment and dependencies."
    echo ""
    echo "   Try:"
    echo "     conda activate aegnn_env"
    echo "     python3 -c 'import torch; import torch_geometric'"
    exit 1
else
    echo "   ✅ PyTorch and torch-geometric import successful"
fi

# Determine which datasets to process
if [ "${TARGET_DATASETS[0]}" == "all" ]; then
    # Use all available datasets (excluding excluded ones)
    DATASETS_TO_PROCESS=("${ALL_DATASETS[@]}")
    echo "=================================================="
    echo "🌟 Batch Mode: Running MOD optimization for ALL datasets"
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
    echo "🌟 Batch Mode: Running MOD optimization for specified datasets"
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

# Fix: Clean up any leftover temporary files from previous runs
# This prevents reading stale exit codes or PIDs from interrupted runs
echo "🧹 Cleaning up any leftover temporary files from previous runs..."
rm -f /tmp/optuna_seed_*.pids /tmp/optuna_seed_*.exit 2>/dev/null || true
echo "   ✅ Temporary files cleaned up"

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
            # Per-seed optimization: Run optimization for each seed (1-5) in parallel
            echo "🔀 Running per-seed optimization for $dataset (seeds 1-5) in parallel..."
            echo "   GPU Distribution: GPU 0 → seeds [1,2], GPU 1 → seeds [3,4,5]"
            SEED_SUCCESS_COUNT=0
            SEED_FAIL_COUNT=0
            SEED_SKIP_COUNT=0
            
            # Arrays to store PIDs and exit codes for parallel execution
            declare -a seed_pids=()
            declare -a seed_exit_codes=()
            declare -a seed_numbers=()
            
            # Start all seeds in parallel
            for seed_num in 1 2 3 4 5; do
                echo "-------------------------------------------"
                echo "🌱 Starting seed $seed_num optimization for $dataset..."
                echo "-------------------------------------------"
                
                # Check if this seed's data exists
                if [ -d "$TDC_DATASET_PATH/seed${seed_num}" ] && \
                   [ -f "$TDC_DATASET_PATH/seed${seed_num}/train.pt" ] && \
                   [ -f "$TDC_DATASET_PATH/seed${seed_num}/valid.pt" ] && \
                   [ -f "$TDC_DATASET_PATH/seed${seed_num}/test.pt" ]; then
                    
                    # Run optimization for this specific seed in background
                    # Use setsid to create new session, preventing signal propagation from parent
                    # IMPORTANT: We need to wait for the actual Python processes, not just the subshell
                    (
                        # Ignore SIGTERM in the subshell to prevent premature termination
                        # Only SIGINT (Ctrl+C) should terminate, not SIGTERM from parent cleanup
                        trap '' SIGTERM
                        # Run optimization and capture exit code
                        run_optimization_for_seed "$dataset" "$seed_num"
                        exit_code=$?
                        # Write exit code to temp file only after function completes
                        echo "$exit_code" > "/tmp/optuna_seed_${dataset}_${seed_num}.exit"
                        exit $exit_code
                    ) &
                    seed_pid=$!  # Remove 'local' - we're not in a function
                    # Also track the actual Python process PIDs for this seed
                    # Wait a moment for processes to start
                    sleep 0.5
                    # Find all Python processes for this seed
                    # Fix: Match all workers for this seed (not just one)
                    seed_python_pids=$(pgrep -f "optuna_serach_mod_new.py.*--dataset.*$dataset.*--seed.*$seed_num" | grep -v grep || true)
                    if [ -n "$seed_python_pids" ]; then
                        # Store Python PIDs for later waiting (one PID per line)
                        echo "$seed_python_pids" | tr ' ' '\n' > "/tmp/optuna_seed_${dataset}_${seed_num}.pids"
                    fi
                    seed_pids+=($seed_pid)
                    seed_numbers+=($seed_num)
                    echo "   ✅ Started seed $seed_num optimization (PID: $seed_pid)"
                else
                    echo "⚠️  Skipping seed $seed_num (data not found)"
                    SEED_SKIP_COUNT=$((SEED_SKIP_COUNT + 1))
                fi
            done
            
            # Wait for all seeds to complete
            echo ""
            echo "⏳ Waiting for all seeds to complete..."
            echo "   Active seeds: ${seed_numbers[*]}"
            echo "   PIDs: ${seed_pids[*]}"
            echo ""
            
            # Monitor and wait for all seeds
            for i in "${!seed_pids[@]}"; do
                seed_pid="${seed_pids[$i]}"
                seed_num="${seed_numbers[$i]}"
                
                # Fix: Simplified waiting logic - wait for Python processes directly
                # First, try to get Python PIDs from temp file (most reliable)
                if [ -f "/tmp/optuna_seed_${dataset}_${seed_num}.pids" ]; then
                    # Wait for all Python processes listed in the file
                    while IFS= read -r python_pid; do
                        if [ -n "$python_pid" ] && kill -0 "$python_pid" 2>/dev/null; then
                            # Python process still running, wait for it (this will block)
                            echo "   ⏳ Waiting for Python process $python_pid to complete..."
                            wait "$python_pid" 2>/dev/null || true
                        fi
                    done < "/tmp/optuna_seed_${dataset}_${seed_num}.pids"
                    rm -f "/tmp/optuna_seed_${dataset}_${seed_num}.pids"
                fi
                
                # Also wait for the subshell to get its exit code
                # This ensures we capture the exit code from run_optimization_for_seed
                wait "$seed_pid" 2>/dev/null
                exit_code=$?
                
                # Double-check: if subshell exited but Python processes are still running, wait for them
                python_pids=$(pgrep -f "optuna_serach_mod_new.py.*--dataset.*$dataset.*--seed.*$seed_num" | grep -v grep || true)
                if [ -n "$python_pids" ]; then
                    for python_pid in $python_pids; do
                        if kill -0 "$python_pid" 2>/dev/null; then
                            echo "   ⏳ Waiting for remaining Python process $python_pid to complete..."
                            wait "$python_pid" 2>/dev/null || true
                        fi
                    done
                fi
                
                # Fix: Clear exit code priority logic
                # Double-check exit code from temp file (should match wait result)
                # Wait a moment for file to be written if it doesn't exist yet
                if [ ! -f "/tmp/optuna_seed_${dataset}_${seed_num}.exit" ]; then
                    sleep 0.5
                fi
                if [ -f "/tmp/optuna_seed_${dataset}_${seed_num}.exit" ]; then
                    file_exit_code=$(cat "/tmp/optuna_seed_${dataset}_${seed_num}.exit" 2>/dev/null || echo "")
                    rm -f "/tmp/optuna_seed_${dataset}_${seed_num}.exit"
                    # Clear priority logic:
                    # 1. If file_exit_code is non-zero (failure), use it (most reliable)
                    # 2. If file_exit_code is 0 (success) but wait returned non-zero, use wait result (indicates actual failure)
                    # 3. If both are 0, use 0 (success)
                    if [ -n "$file_exit_code" ] && [ "$file_exit_code" != "0" ]; then
                        # File indicates failure, use it
                        exit_code=$file_exit_code
                    elif [ "$exit_code" != "0" ]; then
                        # Wait indicates failure, use it (even if file says success, wait is more reliable)
                        exit_code=$exit_code
                    else
                        # Both indicate success, use 0
                        exit_code=0
                    fi
                fi
                
                seed_exit_codes[$i]=$exit_code
                
                if [ $exit_code -eq 0 ]; then
                    SEED_SUCCESS_COUNT=$((SEED_SUCCESS_COUNT + 1))
                    echo "✅ Seed $seed_num optimization completed for $dataset"
                elif [ $exit_code -eq 2 ]; then
                    SEED_SKIP_COUNT=$((SEED_SKIP_COUNT + 1))
                    echo "⏭️  Seed $seed_num optimization skipped (target already reached)"
                else
                    SEED_FAIL_COUNT=$((SEED_FAIL_COUNT + 1))
                    echo "❌ Seed $seed_num optimization failed for $dataset (exit code: $exit_code)"
                fi
            done
            
            # Determine overall dataset status
            if [ $SEED_SUCCESS_COUNT -gt 0 ]; then
                SUCCESSFUL_DATASETS+=("$dataset")
                echo "✅ Completed $dataset: $SEED_SUCCESS_COUNT successful, $SEED_FAIL_COUNT failed, $SEED_SKIP_COUNT skipped"
            elif [ $SEED_FAIL_COUNT -gt 0 ]; then
                FAILED_DATASETS+=("$dataset")
            else
                SKIPPED_DATASETS+=("$dataset")
            fi
            
            echo "😴 Cooling down for 30 seconds before next dataset..."
            sleep 30
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
echo "🎊 Batch optimization completed!"
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



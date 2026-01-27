#!/bin/bash

# Emergency cleanup script for orphaned Optuna training processes
# Usage: ./cleanup_orphaned_processes.sh [--force]

FORCE_KILL=false
if [ "$1" == "--force" ]; then
    FORCE_KILL=true
fi

echo "🔍 Searching for orphaned Optuna and training processes..."

# Find all optuna_serach_mod.py processes (both old and new versions)
OPTUNA_PIDS=$(pgrep -f "optuna_serach_mod" 2>/dev/null || true)

# Find all train_edmpnn.py processes (both old and new versions)
TRAINING_PIDS=$(pgrep -f "train_edmpnn" 2>/dev/null || true)

# Find Python processes in aegnn_env that might be related
PYTHON_PIDS=$(pgrep -f "aegnn_env.*python" 2>/dev/null || true)

TOTAL_FOUND=0

if [ -n "$OPTUNA_PIDS" ]; then
    echo ""
    echo "📊 Found optuna_serach_mod processes (old and new versions):"
    for pid in $OPTUNA_PIDS; do
        if ps -p "$pid" > /dev/null 2>&1; then
            TOTAL_FOUND=$((TOTAL_FOUND + 1))
            ps -p "$pid" -o pid,ppid,cmd --no-headers | awk '{printf "   PID %s (PPID %s): %s\n", $1, $2, substr($0, index($0,$3))}'
        fi
    done
fi

if [ -n "$TRAINING_PIDS" ]; then
    echo ""
    echo "📊 Found train_edmpnn processes (old and new versions):"
    for pid in $TRAINING_PIDS; do
        if ps -p "$pid" > /dev/null 2>&1; then
            TOTAL_FOUND=$((TOTAL_FOUND + 1))
            ps -p "$pid" -o pid,ppid,cmd --no-headers | awk '{printf "   PID %s (PPID %s): %s\n", $1, $2, substr($0, index($0,$3))}'
        fi
    done
fi

if [ $TOTAL_FOUND -eq 0 ]; then
    echo "✅ No orphaned processes found."
    exit 0
fi

echo ""
echo "⚠️  Found $TOTAL_FOUND process(es) that may be orphaned."

if [ "$FORCE_KILL" = false ]; then
    read -p "Do you want to kill these processes? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "❌ Aborted."
        exit 0
    fi
fi

echo ""
echo "🛑 Terminating processes..."

# Kill optuna processes and their children
if [ -n "$OPTUNA_PIDS" ]; then
    for pid in $OPTUNA_PIDS; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "   Terminating PID $pid and its children..."
            pkill -TERM -P "$pid" 2>/dev/null || true
            kill -TERM "$pid" 2>/dev/null || true
        fi
    done
fi

# Kill training processes and their children
if [ -n "$TRAINING_PIDS" ]; then
    for pid in $TRAINING_PIDS; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "   Terminating PID $pid and its children..."
            pkill -TERM -P "$pid" 2>/dev/null || true
            kill -TERM "$pid" 2>/dev/null || true
        fi
    done
fi

# Wait for graceful shutdown
sleep 3

# Force kill any remaining processes
echo ""
echo "🔪 Force killing any remaining processes..."

if [ -n "$OPTUNA_PIDS" ]; then
    for pid in $OPTUNA_PIDS; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "   Force killing PID $pid..."
            pkill -KILL -P "$pid" 2>/dev/null || true
            kill -KILL "$pid" 2>/dev/null || true
        fi
    done
fi

if [ -n "$TRAINING_PIDS" ]; then
    for pid in $TRAINING_PIDS; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "   Force killing PID $pid..."
            pkill -KILL -P "$pid" 2>/dev/null || true
            kill -KILL "$pid" 2>/dev/null || true
        fi
    done
fi

# Final check
sleep 1
REMAINING=$(pgrep -f "optuna_serach_mod\|train_edmpnn" 2>/dev/null | wc -l)

if [ "$REMAINING" -eq 0 ]; then
    echo ""
    echo "✅ All processes terminated successfully."
else
    echo ""
    echo "⚠️  Warning: $REMAINING process(es) may still be running."
    echo "   You may need to manually check with: ps aux | grep -E 'optuna_serach_mod|train_edmpnn'"
fi





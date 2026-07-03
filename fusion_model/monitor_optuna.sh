#!/bin/bash
# =============================================================================
#  monitor_optuna.sh  —  Optuna Training Monitor for run_optuna_train.py
#
#  Layout:
#    ① Current dataset & model info
#    ② Total running time for the current dataset
#    ③ Five seed tqdm-style progress bars for the active trial
#    ④ Latest 5 completed trials with average metric score
#    ⑤ Best trial metric score (with direction awareness)
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results/optuna_results"
REFRESH=10         # seconds between refreshes
DEFAULT_EPOCHS=1000  # matches run_optuna_train.py --num_epochs default

# ── ANSI codes ──────────────────────────────────────────────────────────────
R='\033[0m';    BOLD='\033[1m';   DIM='\033[2m'
RED='\033[31m'; GRN='\033[32m';  YLW='\033[33m'
BLU='\033[34m'; MAG='\033[35m';  CYN='\033[36m'; WHT='\033[97m'
ORG='\033[38;5;214m'

# ── Helpers ──────────────────────────────────────────────────────────────────

draw_bar() {
    # draw_bar <current> <total> [bar_width=38]
    local cur=$1 total=$2 width=${3:-38}
    local pct=0 filled=0
    [[ $total -gt 0 ]] && pct=$(( cur * 100 / total )) && filled=$(( cur * width / total ))
    local bar="" i
    for (( i=0; i<filled; i++ ));   do bar+="█"; done
    for (( i=filled; i<width; i++ )); do bar+="░"; done
    printf "${CYN}[%s]${R} ${WHT}%3d%%${R} (${YLW}%d${R}/${YLW}%d${R})" \
           "$bar" "$pct" "$cur" "$total"
}

fmt_time() {
    local s=$1
    printf "%02dh %02dm %02ds" $(( s/3600 )) $(( (s%3600)/60 )) $(( s%60 ))
}

# ── Detect the currently active run ─────────────────────────────────────────
# Returns "model|dataset|trial_num|save_dir|trial_dir" via stdout.
# Picks the training_progress.json with the most recent mtime.
detect_active() {
    local latest
    latest=$(find "$RESULTS_DIR" -name "training_progress.json" \
             -printf "%T@ %p\n" 2>/dev/null \
             | sort -rn | head -1 | awk '{print $2}')
    [[ -z "$latest" ]] && return 1

    # Path:  …/optuna_results/<MODEL>/<DATASET>/checkpoint/trial_N/seed<S>/training_progress.json
    local seed_dir trial_dir checkpoint_dir save_dir
    seed_dir=$(dirname "$latest")
    trial_dir=$(dirname "$seed_dir")
    checkpoint_dir=$(dirname "$trial_dir")
    save_dir=$(dirname "$checkpoint_dir")

    local data_name model_type trial_num
    data_name=$(basename "$save_dir")
    model_type=$(basename "$(dirname "$save_dir")")
    trial_num=$(basename "$trial_dir" | sed 's/trial_//')

    echo "${model_type}|${data_name}|${trial_num}|${save_dir}|${trial_dir}"
}

# ── Detect num_epochs from running process (fallback: DEFAULT_EPOCHS) ────────
get_num_epochs() {
    local n
    n=$(ps -eo args 2>/dev/null \
        | grep -E "optuna_train|seed_train" \
        | grep -o -- '--num_epochs [0-9]*' \
        | awk '{print $2}' | sort -n | tail -1)
    echo "${n:-$DEFAULT_EPOCHS}"
}

# ── Detect primary metric from running process; fallback: parse study_summary ─
get_primary_metric() {
    local log_dir=$1
    # 1st: from live process args
    local m
    m=$(ps -eo args 2>/dev/null \
        | grep -E "optuna_train|seed_train" \
        | grep -o -- '--metric [A-Za-z-]*' \
        | awk '{print $2}' | head -1)
    if [[ -n "$m" ]]; then echo "$m"; return; fi

    # 2nd: from study_summary_*.txt  (look for "metric" keyword in Best params block)
    if [[ -d "$log_dir" ]]; then
        local summary
        summary=$(ls "$log_dir"/study_summary_*.txt 2>/dev/null | sort | tail -1)
        if [[ -n "$summary" ]]; then
            # The run script passes --metric; it's embedded in the study name line
            # e.g. study_name = "opt_dmpegnn_mmb_desc_herg"
            # Fall back to inferring from Best value magnitude or direction keyword
            # Try to grep any known metric name from the file
            local found
            found=$(grep -oE 'ROC-AUC|PR-AUC|Spearman|MAE' "$summary" | head -1)
            if [[ -n "$found" ]]; then echo "$found"; return; fi
        fi
    fi

    echo "metric"   # ultimate fallback
}

# ── Compute average time per completed trial (all 5 seeds done) ─────────────
compute_avg_trial_time() {
    local checkpoint_dir=$1 run_start_ts=$2
    python3 - "$checkpoint_dir" "$run_start_ts" <<'PYEOF'
import os, sys

checkpoint_dir = sys.argv[1]
run_start_ts   = float(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] != "-1" else None

# Collect end time of each fully-completed trial (all 5 seeds done).
# end(trial) = max mtime of training_summary.json across seeds.
trial_ends = {}   # trial_num -> end_timestamp
if os.path.isdir(checkpoint_dir):
    for entry in os.listdir(checkpoint_dir):
        if not entry.startswith("trial_"):
            continue
        try:
            tnum = int(entry.split("_")[1])
        except ValueError:
            continue
        tdir = os.path.join(checkpoint_dir, entry)
        ends = []
        all_done = True
        for s in range(1, 6):
            summ = os.path.join(tdir, f"seed{s}", "training_summary.json")
            if os.path.isfile(summ):
                ends.append(os.path.getmtime(summ))
            else:
                all_done = False
                break
        if all_done and ends:
            trial_ends[tnum] = max(ends)

if not trial_ends:
    print("N/A")
    sys.exit(0)

# Sort trials by end time; compute durations from consecutive end times.
sorted_trials = sorted(trial_ends.items(), key=lambda x: x[1])
durations = []
for i, (tnum, end_t) in enumerate(sorted_trials):
    if i == 0:
        if run_start_ts and run_start_ts > 0:
            durations.append(end_t - run_start_ts)
    else:
        prev_end = sorted_trials[i - 1][1]
        durations.append(end_t - prev_end)

if durations:
    avg = int(sum(durations) / len(durations))
    h, rem = divmod(avg, 3600)
    m, s = divmod(rem, 60)
    print(f"{h:02d}h {m:02d}m {s:02d}s")
else:
    print("N/A")
PYEOF
}

# ── Compute average best_epoch across all fully-completed trials/seeds ───────
compute_avg_epoch() {
    local checkpoint_dir=$1
    python3 - "$checkpoint_dir" <<'PYEOF'
import os, sys, json

checkpoint_dir = sys.argv[1]
epochs = []
if os.path.isdir(checkpoint_dir):
    for entry in os.listdir(checkpoint_dir):
        if not entry.startswith("trial_"):
            continue
        tdir = os.path.join(checkpoint_dir, entry)
        seed_epochs = []
        all_done = True
        for s in range(1, 6):
            summ = os.path.join(tdir, f"seed{s}", "training_summary.json")
            if os.path.isfile(summ):
                try:
                    d = json.load(open(summ))
                    ep = d.get("best_epoch")
                    if ep is not None:
                        seed_epochs.append(int(ep))
                except Exception:
                    pass
            else:
                all_done = False
                break
        if all_done and seed_epochs:
            epochs.extend(seed_epochs)

if epochs:
    avg = sum(epochs) / len(epochs)
    n_trials = len(epochs) // 5
    print(f"{avg:.0f} epochs")
else:
    print("N/A")
PYEOF
}

# ── Compute running time from oldest training_progress.json in save_dir ──────
compute_start_time() {
    local save_dir=$1
    # Use the oldest timestamp field recorded inside any progress file
    python3 - "$save_dir" <<'PYEOF'
import os, json, sys, math

save_dir = sys.argv[1]
earliest = math.inf
for root, _, files in os.walk(save_dir):
    for fname in files:
        if fname == "training_progress.json":
            try:
                d = json.load(open(os.path.join(root, fname)))
                t = d.get("timestamp")
                if t and float(t) < earliest:
                    earliest = float(t)
            except:
                pass
print(int(earliest) if earliest != math.inf else -1)
PYEOF
}

# ── Main refresh loop ────────────────────────────────────────────────────────
main() {
    local num_epochs
    num_epochs=$(get_num_epochs)

    # Cache for start time — recomputed only when dataset changes
    local cached_start_ts=""
    local cached_save_dir=""

    while true; do
        clear

        # ── Header ─────────────────────────────────────────────────────
        echo -e "${BOLD}${CYN}╔══════════════════════════════════════════════════════════════════╗${R}"
        printf  "${BOLD}${CYN}║${R}  ${BOLD}${WHT}%-64s${R}${BOLD}${CYN}  ║${R}\n" \
                "Optuna Training Monitor — Drug Property Prediction"
        echo -e "${BOLD}${CYN}╚══════════════════════════════════════════════════════════════════╝${R}"
        echo -e "  ${DIM}$(date '+%Y-%m-%d %H:%M:%S')  │  refresh every ${REFRESH}s  │  Ctrl+C to exit${R}"
        echo ""

        # ── Detect active run ───────────────────────────────────────────
        local run_info
        if ! run_info=$(detect_active); then
            echo -e "  ${YLW}⏳  No active training detected.${R}"
            echo -e "  ${DIM}Watching: ${RESULTS_DIR}${R}"
            sleep "$REFRESH"; continue
        fi

        IFS='|' read -r model_type data_name trial_num save_dir trial_dir \
            <<< "$run_info"
        local checkpoint_dir="$save_dir/checkpoint"
        local log_dir="$save_dir/log"

        # Refresh num_epochs each iteration so it picks up running process
        num_epochs=$(get_num_epochs)

        # ── ① Dataset info ──────────────────────────────────────────────
        echo -e "  ${BOLD}Dataset   :${R}  ${BOLD}${YLW}${data_name}${R}"
        echo -e "  ${BOLD}Model     :${R}  ${model_type}"

        # ── ② Total running time (cached — only recompute when dataset changes) ──
        if [[ "$save_dir" != "$cached_save_dir" ]]; then
            cached_start_ts=$(compute_start_time "$save_dir")
            cached_save_dir="$save_dir"
        fi
        local elapsed_str="calculating..."
        if [[ "$cached_start_ts" -gt 0 ]] 2>/dev/null; then
            elapsed_str=$(fmt_time $(( $(date +%s) - cached_start_ts )))
        fi
        echo -e "  ${BOLD}Running   :${R}  ${GRN}${elapsed_str}${R}"
        local avg_trial_str
        avg_trial_str=$(compute_avg_trial_time "$checkpoint_dir" "$cached_start_ts")
        echo -e "  ${BOLD}Avg Trial :${R}  ${ORG}${avg_trial_str}${R}"
        local avg_epoch_str
        avg_epoch_str=$(compute_avg_epoch "$checkpoint_dir")
        echo -e "  ${BOLD}Avg Epoch :${R}  ${MAG}${avg_epoch_str}${R}"
        echo ""

        # ── ③ Seed progress bars ────────────────────────────────────────
        echo -e "  ${BOLD}${BLU}── Seed Progress  (Trial #${trial_num}) ──────────────────────────────────${R}"

        for seed in 1 2 3 4 5; do
            local prog="$trial_dir/seed${seed}/training_progress.json"
            local summ="$trial_dir/seed${seed}/training_summary.json"

            if [[ ! -f "$prog" ]]; then
                printf "  ${BOLD}Seed %d${R} ${DIM}⏳${R}  \n" "$seed"
                continue
            fi

            # Parse epoch & metric from JSON
            read -r cur_ep cur_metric < <(python3 -c "
import json
try:
    d = json.load(open('$prog'))
    print(d.get('epoch', 0), d.get('valid_metric', 'N/A'))
except:
    print(0, 'N/A')
" 2>/dev/null)

            local metric_str="N/A"
            [[ "$cur_metric" =~ ^-?[0-9] ]] && \
                metric_str=$(printf "%.4f" "$cur_metric" 2>/dev/null || echo "$cur_metric")

            if [[ -f "$summ" ]]; then
                # Seed finished — show best metric from summary
                local best_m
                best_m=$(python3 -c "
import json
try:
    d = json.load(open('$summ'))
    v = d.get('best_valid_metric')
    print(f'{float(v):.4f}') if v is not None else print('N/A')
except:
    print('N/A')
" 2>/dev/null)
                printf "  ${BOLD}Seed %d${R} ${GRN}✓${R}  " "$seed"
                draw_bar "$cur_ep" "$num_epochs" 30
                printf "  cur=${CYN}%-6s${R}  best=${GRN}%s${R}\n" "$metric_str" "$best_m"
            else
                # Seed still training
                printf "  ${BOLD}Seed %d${R} ${YLW}▶${R}  " "$seed"
                draw_bar "$cur_ep" "$num_epochs" 30
                printf "  cur=${CYN}%s${R}\n" "$metric_str"
            fi
        done
        echo ""

        # ── ④⑤ Trial history + best trial ──────────────────────────────
        echo -e "  ${BOLD}${BLU}── Trial History (Latest 5 Trials + Best) ──────────────────────────${R}"

        local primary_metric
        primary_metric=$(get_primary_metric "$log_dir")

        python3 - "$checkpoint_dir" "$log_dir" "$primary_metric" <<'PYEOF'
import os, json, sys

RESET  = "\033[0m"; BOLD = "\033[1m"; DIM = "\033[2m"
RED    = "\033[31m"; GRN = "\033[32m"; YLW = "\033[33m"
BLU    = "\033[34m"; MAG = "\033[35m"; CYN = "\033[36m"; WHT = "\033[97m"

checkpoint_dir   = sys.argv[1]
log_dir          = sys.argv[2]
primary_metric   = sys.argv[3] if len(sys.argv) > 3 else "metric"

if not os.path.isdir(checkpoint_dir):
    print("  No completed trials yet.")
    sys.exit(0)

# ── Gather results from trials ────────────────────────────────────────
# results_all : all trials that have ≥1 seed done  → shown in the table
# results_full: only trials where ALL 5 seeds are done → eligible for best
results_all  = []   # [(trial_num, avg_metric, n_seeds)]
results_full = []   # [(trial_num, avg_metric, n_seeds)]  n_seeds == 5
for entry in os.listdir(checkpoint_dir):
    if not entry.startswith("trial_"):
        continue
    try:
        tnum = int(entry.split("_")[1])
    except ValueError:
        continue
    tdir = os.path.join(checkpoint_dir, entry)
    metrics = []
    for s in range(1, 6):
        sp = os.path.join(tdir, f"seed{s}", "training_summary.json")
        if os.path.isfile(sp):
            try:
                d = json.load(open(sp))
                v = d.get("best_valid_metric")
                if v is not None:
                    metrics.append(float(v))
            except Exception:
                pass
    if metrics:
        entry_data = (tnum, sum(metrics) / len(metrics), len(metrics))
        results_all.append(entry_data)
        if len(metrics) == 5:
            results_full.append(entry_data)

results_all.sort(key=lambda x: x[0])

if not results_all:
    print("  No completed trials yet.")
    sys.exit(0)

# ── Infer direction from primary_metric (passed as argv[3]) ───────────
MINIMIZE_METRICS = {"MAE"}
direction = "minimize" if primary_metric in MINIMIZE_METRICS else "maximize"

# ── Best trial — only from trials where ALL 5 seeds completed ─────────
best = None
if results_full:
    if direction == "maximize":
        best = max(results_full, key=lambda x: x[1])
    else:
        best = min(results_full, key=lambda x: x[1])

# ── Print latest 5 (from all partially-or-fully completed trials) ──────
latest5 = results_all[-5:]
total_all  = len(results_all)
total_full = len(results_full)

print(f"  {BOLD}{'Trial':>7}  {'Avg Metric':>12}  {'Seeds':>5}  {'':6}{RESET}")
print(f"  {'─'*7}  {'─'*12}  {'─'*5}  {'─'*6}")
for tnum, avg, nseed in latest5:
    if best and tnum == best[0]:
        tag = f"  {YLW}{BOLD}★ BEST{RESET}"
    elif nseed < 5:
        tag = f"  {DIM}(incomplete){RESET}"
    else:
        tag = ""
    print(f"  {BOLD}#{tnum:<6}{RESET}  {CYN}{avg:12.4f}{RESET}  {nseed:>2}/5{tag}")

print()
if best:
    print(f"  {BOLD}{YLW}Best Trial (5/5 seeds)  :{RESET}  "
          f"#{best[0]}  {primary_metric} = {GRN}{BOLD}{best[1]:.4f}{RESET}"
          f"  [{direction}]")
else:
    print(f"  {YLW}Best Trial              :{RESET}  {DIM}waiting for first fully-completed trial (5/5 seeds)...{RESET}")
PYEOF

        echo ""
        sleep "$REFRESH"
    done
}

# ── Entry point ──────────────────────────────────────────────────────────────
cd "$SCRIPT_DIR" || { echo "Cannot cd to $SCRIPT_DIR"; exit 1; }

if [[ ! -d "$RESULTS_DIR" ]]; then
    echo -e "${YLW}Warning: results directory not found yet: ${RESULTS_DIR}${R}"
    echo -e "${DIM}Will start monitoring once training creates it.${R}"
fi

trap 'echo -e "\n${YLW}Monitor stopped.${R}"; exit 0' INT TERM

echo -e "${CYN}${BOLD}Optuna Training Monitor${R}"
echo -e "${DIM}Results root : ${RESULTS_DIR}${R}"
echo -e "${DIM}Refresh      : ${REFRESH}s${R}"
echo ""
sleep 1
main

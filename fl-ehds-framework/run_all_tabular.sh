#!/usr/bin/env bash
# ==============================================================================
# FL-EHDS — Run ALL Tabular Experiments in Cascade
# ==============================================================================
#
# Experiments (in order):
#   1. run_tabular_optimized  — baseline comparison      (105 exp, ~45 min)
#   2. run_tabular_seeds10    — 10-seed validation        (105 exp, ~40 min)
#   3. run_tabular_dp         — DP ablation               (180 exp, ~1.5h)
#   4. run_tabular_optout     — Article 71 opt-out        (225 exp, ~1.5h)
#   5. run_tabular_deep_mlp   — Deep MLP                  (70 exp, ~1.5h)
#
# Safety:
#   - Each Python script saves checkpoint after EVERY single experiment
#   - Automatic resume: re-run this script to continue from where it stopped
#   - SIGINT (Ctrl+C): current experiment saves checkpoint, cascade stops
#   - A cascade status file tracks which blocks completed across runs
#
# Usage:
#   conda activate flics2026
#   cd fl-ehds-framework
#   bash run_all_tabular.sh [--quick] [--fresh]
#
#   --quick   Validation mode (1 seed, 5 rounds per script)
#   --fresh   Restart everything (clears cascade status, passes --fresh to scripts)
#
# ==============================================================================

set -uo pipefail

# --- Configuration ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/benchmarks/paper_results_tabular"
CASCADE_LOG="${RESULTS_DIR}/cascade_run.log"
CASCADE_STATUS="${RESULTS_DIR}/cascade_status.txt"
QUICK_FLAG=""
FRESH_FLAG=""

for arg in "$@"; do
    case "$arg" in
        --quick) QUICK_FLAG="--quick" ;;
        --fresh) FRESH_FLAG="--fresh" ;;
    esac
done

# --- Colors ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m'

# --- Experiments definition ---
#   module:num_experiments:estimated_minutes:description
EXPERIMENTS=(
    "run_tabular_optimized:105:45:Baseline comparison"
    "run_tabular_seeds10:105:40:10-seed statistical validation"
    "run_tabular_dp:180:90:DP ablation"
    "run_tabular_optout:225:90:Article 71 opt-out"
    "run_tabular_deep_mlp:70:90:Deep MLP differentiation"
)
TOTAL_EXPS=685

# --- Functions ---
timestamp() {
    date "+%Y-%m-%d %H:%M:%S"
}

log_msg() {
    local msg="[$(timestamp)] $1"
    echo -e "$msg"
    echo -e "$msg" | sed 's/\x1b\[[0-9;]*m//g' >> "$CASCADE_LOG"
}

# Check how many experiments a checkpoint file has completed
checkpoint_progress() {
    local checkpoint_file="$1"
    if [ -f "$checkpoint_file" ]; then
        python3 -c "
import json, sys
with open('$checkpoint_file') as f:
    data = json.load(f)
completed = len(data.get('completed', {}))
total = data.get('metadata', {}).get('total_experiments', '?')
print(f'{completed}/{total}')
" 2>/dev/null || echo "?/?"
    else
        echo "0/?"
    fi
}

# Check if a block already completed in a previous cascade run
is_block_done() {
    local module="$1"
    if [ -f "$CASCADE_STATUS" ]; then
        grep -q "^${module}:DONE$" "$CASCADE_STATUS" 2>/dev/null
        return $?
    fi
    return 1
}

mark_block_done() {
    local module="$1"
    echo "${module}:DONE" >> "$CASCADE_STATUS"
}

print_dashboard() {
    echo ""
    echo -e "${BOLD}============================================================${NC}"
    echo -e "${BOLD}  FL-EHDS TABULAR — CASCADE DASHBOARD${NC}"
    echo -e "${DIM}  $(timestamp)${NC}"
    echo -e "${BOLD}============================================================${NC}"
    echo ""

    local total_done=0
    for entry in "${EXPERIMENTS[@]}"; do
        IFS=':' read -r module num_exp est_min desc <<< "$entry"
        local ckpt="${RESULTS_DIR}/checkpoint_${module#run_tabular_}.json"
        local progress
        progress=$(checkpoint_progress "$ckpt")

        local status_icon status_color
        if is_block_done "$module"; then
            status_icon="DONE"
            status_color="$GREEN"
            total_done=$((total_done + num_exp))
        elif [ -f "$ckpt" ]; then
            local done_count
            done_count=$(echo "$progress" | cut -d'/' -f1)
            if [ "$done_count" != "0" ] && [ "$done_count" != "?" ]; then
                status_icon="PARTIAL"
                status_color="$YELLOW"
                total_done=$((total_done + done_count))
            else
                status_icon="PENDING"
                status_color="$DIM"
            fi
        else
            status_icon="PENDING"
            status_color="$DIM"
        fi

        printf "  ${status_color}%-8s${NC}  %-30s  %6s exp  ~%3d min\n" \
            "[$status_icon]" "$desc" "$progress" "$est_min"
    done

    echo ""
    echo -e "  ${BOLD}Global: ${total_done}/${TOTAL_EXPS} experiments completed${NC}"
    echo -e "${BOLD}============================================================${NC}"
    echo ""
}

# --- Trap SIGINT ---
INTERRUPTED=0
trap 'INTERRUPTED=1; echo -e "\n${YELLOW}>>> Ctrl+C received — saving checkpoint and stopping...${NC}"' INT

# --- Main ---
mkdir -p "$RESULTS_DIR"

# Fresh mode: clear cascade status (individual checkpoints cleared by --fresh flag)
if [ -n "$FRESH_FLAG" ]; then
    rm -f "$CASCADE_STATUS"
    log_msg "${YELLOW}FRESH mode: cascade status cleared${NC}"
fi

# Show initial dashboard
print_dashboard

log_msg "=== CASCADE START ($([ -n "$QUICK_FLAG" ] && echo "QUICK" || echo "FULL") mode) ==="

CASCADE_START=$(date +%s)
BLOCKS_COMPLETED=0
BLOCKS_FAILED=0
BLOCKS_SKIPPED=0

for i in "${!EXPERIMENTS[@]}"; do
    IFS=':' read -r MODULE NUM_EXP EST_MIN DESC <<< "${EXPERIMENTS[$i]}"
    IDX=$((i + 1))
    TOTAL=${#EXPERIMENTS[@]}

    # --- Skip if interrupted ---
    if [ "$INTERRUPTED" -eq 1 ]; then
        log_msg "${YELLOW}[${IDX}/${TOTAL}] SKIPPED: ${DESC}${NC}"
        BLOCKS_SKIPPED=$((BLOCKS_SKIPPED + 1))
        continue
    fi

    # --- Skip if already completed in previous cascade run ---
    if is_block_done "$MODULE"; then
        local_ckpt="${RESULTS_DIR}/checkpoint_${MODULE#run_tabular_}.json"
        local_progress=$(checkpoint_progress "$local_ckpt")
        log_msg "${GREEN}[${IDX}/${TOTAL}] ALREADY DONE: ${DESC} (${local_progress})${NC}"
        BLOCKS_COMPLETED=$((BLOCKS_COMPLETED + 1))
        continue
    fi

    echo ""
    echo -e "${BOLD}------------------------------------------------------------${NC}"
    log_msg "${CYAN}[${IDX}/${TOTAL}] STARTING: ${DESC} (${NUM_EXP} experiments, ~${EST_MIN} min)${NC}"
    echo -e "${BOLD}------------------------------------------------------------${NC}"

    # Build command
    CMD="python -m benchmarks.${MODULE}"
    if [ "$MODULE" = "run_tabular_optimized" ]; then
        CMD="$CMD --resume"
    fi
    if [ -n "$FRESH_FLAG" ]; then
        # run_tabular_optimized uses --resume (no --fresh), others use --fresh
        if [ "$MODULE" != "run_tabular_optimized" ]; then
            CMD="$CMD --fresh"
        else
            # For optimized: don't pass --resume if fresh
            CMD="python -m benchmarks.${MODULE}"
        fi
    fi
    if [ -n "$QUICK_FLAG" ]; then
        CMD="$CMD --quick"
    fi

    log_msg "${DIM}Command: ${CMD}${NC}"

    BLOCK_START=$(date +%s)

    # Run — SIGINT passes through to Python which saves checkpoint
    set +e
    $CMD 2>&1 | tee -a "${RESULTS_DIR}/${MODULE}.log"
    EXIT_CODE=${PIPESTATUS[0]}
    set -e

    BLOCK_END=$(date +%s)
    BLOCK_ELAPSED=$(( BLOCK_END - BLOCK_START ))
    BLOCK_MIN=$(( BLOCK_ELAPSED / 60 ))
    BLOCK_SEC=$(( BLOCK_ELAPSED % 60 ))
    TIME_STR="${BLOCK_MIN}m ${BLOCK_SEC}s"

    # Check final checkpoint state
    CKPT_FILE="${RESULTS_DIR}/checkpoint_${MODULE#run_tabular_}.json"
    FINAL_PROGRESS=$(checkpoint_progress "$CKPT_FILE")

    if [ "$EXIT_CODE" -eq 0 ]; then
        log_msg "${GREEN}[${IDX}/${TOTAL}] COMPLETED: ${DESC} — ${FINAL_PROGRESS} in ${TIME_STR}${NC}"
        mark_block_done "$MODULE"
        BLOCKS_COMPLETED=$((BLOCKS_COMPLETED + 1))
    else
        log_msg "${RED}[${IDX}/${TOTAL}] STOPPED (exit ${EXIT_CODE}): ${DESC} — ${FINAL_PROGRESS} saved (${TIME_STR})${NC}"
        BLOCKS_FAILED=$((BLOCKS_FAILED + 1))

        if [ "$INTERRUPTED" -eq 1 ]; then
            log_msg "${YELLOW}Cascade paused. Re-run this script to resume.${NC}"
        fi
    fi

    # Dashboard after each block
    print_dashboard

done

# --- Final Summary ---
CASCADE_END=$(date +%s)
CASCADE_ELAPSED=$(( CASCADE_END - CASCADE_START ))
CASCADE_H=$(( CASCADE_ELAPSED / 3600 ))
CASCADE_M=$(( (CASCADE_ELAPSED % 3600) / 60 ))

echo ""
echo -e "${BOLD}============================================================${NC}"
echo -e "${BOLD}  CASCADE FINAL SUMMARY${NC}"
echo -e "${BOLD}============================================================${NC}"
echo ""
echo -e "  Blocks completed : ${GREEN}${BLOCKS_COMPLETED}${NC} / ${#EXPERIMENTS[@]}"
echo -e "  Blocks failed    : ${RED}${BLOCKS_FAILED}${NC}"
echo -e "  Blocks skipped   : ${YELLOW}${BLOCKS_SKIPPED}${NC}"
echo -e "  Total time       : ${CASCADE_H}h ${CASCADE_M}m"
echo ""
echo -e "  Results saved in : ${RESULTS_DIR}"
echo -e "  Cascade log      : ${CASCADE_LOG}"
echo ""
if [ "$BLOCKS_COMPLETED" -lt "${#EXPERIMENTS[@]}" ]; then
    echo -e "  ${YELLOW}To resume: bash run_all_tabular.sh${NC}"
fi
echo -e "${BOLD}============================================================${NC}"
echo ""

log_msg "=== CASCADE END — Blocks: ${BLOCKS_COMPLETED}/${#EXPERIMENTS[@]} done, ${BLOCKS_FAILED} failed ==="

# Exit with error if any failed (and not just interrupted)
if [ "$BLOCKS_FAILED" -gt 0 ] && [ "$INTERRUPTED" -eq 0 ]; then
    exit 1
fi

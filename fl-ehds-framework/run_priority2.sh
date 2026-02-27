#!/usr/bin/env bash
# ==============================================================================
# FL-EHDS — Priority 2 Experiments Cascade
# ==============================================================================
#
# Experiments (in order):
#   1. Byzantine + DP           — 198 exp, ~40-50 min  (tabular, fast)
#   2. DP per-class recall/DEI  — 80 exp, ~40-60 min   (tabular, fast)
#   3. Scalability DP on CV     — 18 exp, ~30-60 min   (tabular)
#   4. Local vs Federated       — ~30 exp, ~1-2h        (imaging, CNN)
#
# Safety:
#   - Scripts 1-3 save checkpoint after every experiment (auto-resume)
#   - Script 4 saves results at end (no mid-run checkpoint)
#   - SIGINT (Ctrl+C): current experiment saves, cascade stops
#   - Cascade status file tracks completed blocks across runs
#
# Usage:
#   conda activate flics2026
#   cd fl-ehds-framework
#   bash run_priority2.sh [--quick]
#
# ==============================================================================

set -uo pipefail

# --- Configuration ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/benchmarks/paper_results_tabular"
CASCADE_LOG="${RESULTS_DIR}/cascade_priority2.log"
CASCADE_STATUS="${RESULTS_DIR}/cascade_priority2_status.txt"
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

# --- Experiments ---
#   module:num_exp:est_min:description:run_cmd:result_check
EXPERIMENTS=(
    "run_byzantine_dp:198:50:Byzantine + DP joint evaluation:python benchmarks/run_byzantine_dp.py:byzantine_dp_v2_results.json"
    "run_dp_per_class:80:60:DP per-class recall & DEI analysis:python benchmarks/run_dp_per_class.py:dp_per_class_results.json"
    "run_scalability_dp_cv:18:45:Scalability + DP on Cardiovascular:python -m benchmarks.run_scalability_dp_cv:checkpoint_scalability_dp_cv.json"
    "run_local_vs_federated:30:120:Local-Only vs Federated (imaging CNN):python benchmarks/run_local_vs_federated.py:../results/local_vs_federated/results.json"
)
TOTAL_EXPS=326

# --- Functions ---
timestamp() {
    date "+%Y-%m-%d %H:%M:%S"
}

log_msg() {
    local msg="[$(timestamp)] $1"
    echo -e "$msg"
    echo -e "$msg" | sed 's/\x1b\[[0-9;]*m//g' >> "$CASCADE_LOG"
}

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

check_result_file() {
    local result_file="$1"
    local full_path="${RESULTS_DIR}/${result_file}"
    if [ -f "$full_path" ]; then
        echo "EXISTS"
    else
        echo "MISSING"
    fi
}

print_dashboard() {
    echo ""
    echo -e "${BOLD}============================================================${NC}"
    echo -e "${BOLD}  FL-EHDS PRIORITY 2 — CASCADE DASHBOARD${NC}"
    echo -e "${DIM}  $(timestamp)${NC}"
    echo -e "${BOLD}============================================================${NC}"
    echo ""

    for entry in "${EXPERIMENTS[@]}"; do
        IFS=':' read -r module num_exp est_min desc run_cmd result_file <<< "$entry"

        local status_icon status_color

        if is_block_done "$module"; then
            status_icon="DONE"
            status_color="$GREEN"
        else
            local file_status
            file_status=$(check_result_file "$result_file")
            if [ "$file_status" = "EXISTS" ]; then
                status_icon="PARTIAL"
                status_color="$YELLOW"
            else
                status_icon="PENDING"
                status_color="$DIM"
            fi
        fi

        printf "  ${status_color}%-8s${NC}  %-45s  %4s exp  ~%3d min\n" \
            "[$status_icon]" "$desc" "$num_exp" "$est_min"
    done

    echo ""
    echo -e "${BOLD}============================================================${NC}"
    echo ""
}

# --- Trap SIGINT ---
INTERRUPTED=0
trap 'INTERRUPTED=1; echo -e "\n${YELLOW}>>> Ctrl+C received — saving checkpoint and stopping...${NC}"' INT

# --- Main ---
mkdir -p "$RESULTS_DIR"

if [ -n "$FRESH_FLAG" ]; then
    rm -f "$CASCADE_STATUS"
    log_msg "${YELLOW}FRESH mode: cascade status cleared${NC}"
fi

print_dashboard

log_msg "=== PRIORITY 2 CASCADE START ($([ -n "$QUICK_FLAG" ] && echo "QUICK" || echo "FULL") mode) ==="

CASCADE_START=$(date +%s)
BLOCKS_COMPLETED=0
BLOCKS_FAILED=0
BLOCKS_SKIPPED=0
TOTAL=${#EXPERIMENTS[@]}

for i in "${!EXPERIMENTS[@]}"; do
    IFS=':' read -r MODULE NUM_EXP EST_MIN DESC RUN_CMD RESULT_FILE <<< "${EXPERIMENTS[$i]}"
    IDX=$((i + 1))

    # Skip if interrupted
    if [ "$INTERRUPTED" -eq 1 ]; then
        log_msg "${YELLOW}[${IDX}/${TOTAL}] SKIPPED: ${DESC}${NC}"
        BLOCKS_SKIPPED=$((BLOCKS_SKIPPED + 1))
        continue
    fi

    # Skip if already completed
    if is_block_done "$MODULE"; then
        log_msg "${GREEN}[${IDX}/${TOTAL}] ALREADY DONE: ${DESC}${NC}"
        BLOCKS_COMPLETED=$((BLOCKS_COMPLETED + 1))
        continue
    fi

    echo ""
    echo -e "${BOLD}------------------------------------------------------------${NC}"
    log_msg "${CYAN}[${IDX}/${TOTAL}] STARTING: ${DESC} (${NUM_EXP} exp, ~${EST_MIN} min)${NC}"
    echo -e "${BOLD}------------------------------------------------------------${NC}"

    # Build command with flags
    CMD="$RUN_CMD"
    if [ -n "$QUICK_FLAG" ]; then
        CMD="$CMD --quick"
    fi

    log_msg "${DIM}Command: ${CMD}${NC}"

    BLOCK_START=$(date +%s)

    set +e
    $CMD 2>&1 | tee -a "${RESULTS_DIR}/${MODULE}.log"
    EXIT_CODE=${PIPESTATUS[0]}
    set -e

    BLOCK_END=$(date +%s)
    BLOCK_ELAPSED=$(( BLOCK_END - BLOCK_START ))
    BLOCK_MIN=$(( BLOCK_ELAPSED / 60 ))
    BLOCK_SEC=$(( BLOCK_ELAPSED % 60 ))
    TIME_STR="${BLOCK_MIN}m ${BLOCK_SEC}s"

    if [ "$EXIT_CODE" -eq 0 ]; then
        log_msg "${GREEN}[${IDX}/${TOTAL}] COMPLETED: ${DESC} (${TIME_STR})${NC}"
        mark_block_done "$MODULE"
        BLOCKS_COMPLETED=$((BLOCKS_COMPLETED + 1))
    else
        log_msg "${RED}[${IDX}/${TOTAL}] STOPPED (exit ${EXIT_CODE}): ${DESC} (${TIME_STR})${NC}"
        BLOCKS_FAILED=$((BLOCKS_FAILED + 1))
        if [ "$INTERRUPTED" -eq 1 ]; then
            log_msg "${YELLOW}Cascade paused. Re-run this script to resume.${NC}"
        fi
    fi

    print_dashboard
done

# --- Final Summary ---
CASCADE_END=$(date +%s)
CASCADE_ELAPSED=$(( CASCADE_END - CASCADE_START ))
CASCADE_H=$(( CASCADE_ELAPSED / 3600 ))
CASCADE_M=$(( (CASCADE_ELAPSED % 3600) / 60 ))

echo ""
echo -e "${BOLD}============================================================${NC}"
echo -e "${BOLD}  PRIORITY 2 CASCADE — FINAL SUMMARY${NC}"
echo -e "${BOLD}============================================================${NC}"
echo ""
echo -e "  Blocks completed : ${GREEN}${BLOCKS_COMPLETED}${NC} / ${TOTAL}"
echo -e "  Blocks failed    : ${RED}${BLOCKS_FAILED}${NC}"
echo -e "  Blocks skipped   : ${YELLOW}${BLOCKS_SKIPPED}${NC}"
echo -e "  Total time       : ${CASCADE_H}h ${CASCADE_M}m"
echo ""
echo -e "  Results saved in : ${RESULTS_DIR}"
echo -e "  Cascade log      : ${CASCADE_LOG}"
echo ""
if [ "$BLOCKS_COMPLETED" -lt "$TOTAL" ]; then
    echo -e "  ${YELLOW}To resume: bash run_priority2.sh${NC}"
fi
echo -e "${BOLD}============================================================${NC}"
echo ""

log_msg "=== PRIORITY 2 CASCADE END — Blocks: ${BLOCKS_COMPLETED}/${TOTAL} done, ${BLOCKS_FAILED} failed ==="

if [ "$BLOCKS_FAILED" -gt 0 ] && [ "$INTERRUPTED" -eq 0 ]; then
    exit 1
fi

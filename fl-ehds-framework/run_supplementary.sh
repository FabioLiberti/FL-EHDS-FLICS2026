#!/usr/bin/env bash
# ==============================================================================
# FL-EHDS — Supplementary Experiments Cascade (Priority 1)
# ==============================================================================
#
# Experiments (in order):
#   1. RDP vs Naive Composition    — analytical, ~10 sec
#   2. Scalability Sweep K=50-100  — 84 exp, ~3-5h
#   3. Scalability + DP            — 54 exp, ~2-3h
#
# Safety:
#   - Scripts 2-3 save checkpoint after every experiment
#   - Automatic resume: re-run this script to continue
#   - SIGINT (Ctrl+C): current experiment saves, cascade stops
#   - Cascade status file tracks completed blocks across runs
#
# Usage:
#   conda activate flics2026
#   cd fl-ehds-framework
#   bash run_supplementary.sh [--quick] [--fresh]
#
# ==============================================================================

set -uo pipefail

# --- Configuration ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/benchmarks/paper_results_tabular"
CASCADE_LOG="${RESULTS_DIR}/cascade_supplementary.log"
CASCADE_STATUS="${RESULTS_DIR}/cascade_supplementary_status.txt"
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
#   module:num_experiments:estimated_minutes:description:supports_quick:supports_resume
EXPERIMENTS=(
    "run_rdp_comparison:1:1:RDP vs Naive Composition (analytical):no:no"
    "run_scalability_sweep:84:240:Scalability K=50-100 (Table XVII-XVIII):yes:yes"
    "run_scalability_dp:54:150:Scalability + DP (Table XXIII):yes:yes"
)
TOTAL_EXPS=139

# --- Functions ---
timestamp() {
    date "+%Y-%m-%d %H:%M:%S"
}

log_msg() {
    local msg="[$(timestamp)] $1"
    echo -e "$msg"
    echo -e "$msg" | sed 's/\x1b\[[0-9;]*m//g' >> "$CASCADE_LOG"
}

checkpoint_progress() {
    local checkpoint_file="$1"
    if [ -f "$checkpoint_file" ]; then
        python3 -c "
import json
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
    echo -e "${BOLD}  FL-EHDS SUPPLEMENTARY — CASCADE DASHBOARD${NC}"
    echo -e "${DIM}  $(timestamp)${NC}"
    echo -e "${BOLD}============================================================${NC}"
    echo ""

    local idx=0
    for entry in "${EXPERIMENTS[@]}"; do
        IFS=':' read -r module num_exp est_min desc supports_quick supports_resume <<< "$entry"
        idx=$((idx + 1))

        local status_icon status_color progress

        if is_block_done "$module"; then
            status_icon="DONE"
            status_color="$GREEN"
            # For RDP (no checkpoint), just show "1/1"
            if [ "$module" = "run_rdp_comparison" ]; then
                progress="1/1"
            else
                local ckpt="${RESULTS_DIR}/checkpoint_${module#run_}.json"
                progress=$(checkpoint_progress "$ckpt")
            fi
        else
            if [ "$module" = "run_rdp_comparison" ]; then
                # RDP has no checkpoint file
                if [ -f "${RESULTS_DIR}/rdp_comparison_results.json" ]; then
                    status_icon="DONE"
                    status_color="$GREEN"
                    progress="1/1"
                else
                    status_icon="PENDING"
                    status_color="$DIM"
                    progress="0/1"
                fi
            else
                local ckpt="${RESULTS_DIR}/checkpoint_${module#run_}.json"
                progress=$(checkpoint_progress "$ckpt")
                local done_count
                done_count=$(echo "$progress" | cut -d'/' -f1)
                if [ "$done_count" != "0" ] && [ "$done_count" != "?" ]; then
                    status_icon="PARTIAL"
                    status_color="$YELLOW"
                else
                    status_icon="PENDING"
                    status_color="$DIM"
                fi
            fi
        fi

        printf "  ${status_color}%-8s${NC}  %-45s  %6s  ~%3d min\n" \
            "[$status_icon]" "$desc" "$progress" "$est_min"
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

log_msg "=== SUPPLEMENTARY CASCADE START ($([ -n "$QUICK_FLAG" ] && echo "QUICK" || echo "FULL") mode) ==="

CASCADE_START=$(date +%s)
BLOCKS_COMPLETED=0
BLOCKS_FAILED=0
BLOCKS_SKIPPED=0
TOTAL=${#EXPERIMENTS[@]}

for i in "${!EXPERIMENTS[@]}"; do
    IFS=':' read -r MODULE NUM_EXP EST_MIN DESC SUPPORTS_QUICK SUPPORTS_RESUME <<< "${EXPERIMENTS[$i]}"
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

    # Special check for RDP (no checkpoint, check output file)
    if [ "$MODULE" = "run_rdp_comparison" ] && [ -f "${RESULTS_DIR}/rdp_comparison_results.json" ] && [ -z "$FRESH_FLAG" ]; then
        log_msg "${GREEN}[${IDX}/${TOTAL}] ALREADY DONE: ${DESC}${NC}"
        mark_block_done "$MODULE"
        BLOCKS_COMPLETED=$((BLOCKS_COMPLETED + 1))
        continue
    fi

    echo ""
    echo -e "${BOLD}------------------------------------------------------------${NC}"
    log_msg "${CYAN}[${IDX}/${TOTAL}] STARTING: ${DESC}${NC}"
    echo -e "${BOLD}------------------------------------------------------------${NC}"

    # Build command
    CMD="python -m benchmarks.${MODULE}"
    if [ -n "$FRESH_FLAG" ] && [ "$SUPPORTS_RESUME" = "yes" ]; then
        CMD="$CMD --fresh"
    elif [ "$SUPPORTS_RESUME" = "yes" ] && [ -z "$FRESH_FLAG" ]; then
        CMD="$CMD --resume"
    fi
    if [ -n "$QUICK_FLAG" ] && [ "$SUPPORTS_QUICK" = "yes" ]; then
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
echo -e "${BOLD}  SUPPLEMENTARY CASCADE — FINAL SUMMARY${NC}"
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
    echo -e "  ${YELLOW}To resume: bash run_supplementary.sh${NC}"
fi
echo -e "${BOLD}============================================================${NC}"
echo ""

log_msg "=== SUPPLEMENTARY CASCADE END — Blocks: ${BLOCKS_COMPLETED}/${TOTAL} done, ${BLOCKS_FAILED} failed ==="

if [ "$BLOCKS_FAILED" -gt 0 ] && [ "$INTERRUPTED" -eq 0 ]; then
    exit 1
fi

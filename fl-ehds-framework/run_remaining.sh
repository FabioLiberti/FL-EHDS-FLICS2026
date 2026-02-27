#!/bin/bash
# ======================================================================
# FL-EHDS Remaining Experiments Cascade (MacBook Air M3)
# ======================================================================
#
# Block 1: Tabular Epochs Sweep       — 140 exp (~2h)
# Block 2: Top-K PTB-XL               —   9 exp (~30 min)
# Block 3: Confusion Matrix Chest     —   6 exp (~1h, imaging)
# Total: 155 experiments (~3.5h)
#
# All blocks have:
#   - Atomic checkpoint (tempfile + os.replace + .bak)
#   - SIGINT/SIGTERM handler (save & exit)
#   - Auto-resume on restart
#
# Usage:
#   cd fl-ehds-framework
#   bash run_remaining.sh [--quick]
#
# ======================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

RESULTS_DIR="benchmarks/paper_results_tabular"
RESULTS_DELTA="benchmarks/paper_results_delta"
STATUS_FILE="${RESULTS_DIR}/cascade_remaining_status.txt"
LOG_FILE="${RESULTS_DIR}/cascade_remaining.log"
QUICK_FLAG=""

if [[ "${1:-}" == "--quick" ]]; then
    QUICK_FLAG="--quick"
    echo "  MODE: QUICK (validation only)"
fi

mkdir -p "$RESULTS_DIR" "$RESULTS_DELTA"

# ======================================================================
# Dashboard
# ======================================================================

checkpoint_progress() {
    local file="$1"
    local total="$2"
    if [[ -f "$file" ]]; then
        local done
        done=$(python3 -c "
import json
with open('$file') as f:
    d = json.load(f)
if 'completed' in d and isinstance(d['completed'], dict):
    print(len(d['completed']))
else:
    print('?')
" 2>/dev/null || echo "?")
        echo "${done}/${total}"
    else
        echo "0/${total}"
    fi
}

show_dashboard() {
    echo ""
    echo "  ╔══════════════════════════════════════════════════════════╗"
    echo "  ║  FL-EHDS Remaining Experiments — Dashboard              ║"
    echo "  ╠══════════════════════════════════════════════════════════╣"

    local b1_status b2_status b3_status
    b1_status=$(grep "^BLOCK1=" "$STATUS_FILE" 2>/dev/null | cut -d= -f2 || echo "PENDING")
    b2_status=$(grep "^BLOCK2=" "$STATUS_FILE" 2>/dev/null | cut -d= -f2 || echo "PENDING")
    b3_status=$(grep "^BLOCK3=" "$STATUS_FILE" 2>/dev/null | cut -d= -f2 || echo "PENDING")

    local b1_prog b2_prog b3_prog
    b1_prog=$(checkpoint_progress "${RESULTS_DIR}/checkpoint_epochs_sweep.json" 140)
    b2_prog=$(checkpoint_progress "${RESULTS_DIR}/checkpoint_topk_ptbxl.json" 9)
    b3_prog=$(checkpoint_progress "${RESULTS_DELTA}/checkpoint_confusion_chest.json" 6)

    printf "  ║  %-6s  %-32s  %8s  ║\n" "Status" "Experiment" "Progress"
    echo "  ║  ──────  ────────────────────────────  ────────  ║"
    printf "  ║  %-6s  %-32s  %8s  ║\n" "$b1_status" "1. Epochs Sweep (tabular)" "$b1_prog"
    printf "  ║  %-6s  %-32s  %8s  ║\n" "$b2_status" "2. Top-K PTB-XL (tabular)" "$b2_prog"
    printf "  ║  %-6s  %-32s  %8s  ║\n" "$b3_status" "3. Confusion Chest (imaging)" "$b3_prog"
    echo "  ╚══════════════════════════════════════════════════════════╝"
    echo ""
}

update_status() {
    local block="$1" status="$2"
    if [[ -f "$STATUS_FILE" ]]; then
        if grep -q "^${block}=" "$STATUS_FILE"; then
            sed -i '' "s/^${block}=.*/${block}=${status}/" "$STATUS_FILE"
        else
            echo "${block}=${status}" >> "$STATUS_FILE"
        fi
    else
        echo "${block}=${status}" > "$STATUS_FILE"
    fi
}

# ======================================================================
# SIGINT trap
# ======================================================================

cleanup() {
    echo ""
    echo "  ── SIGINT ricevuto ──"
    echo "  I risultati sono salvi nei checkpoint."
    echo "  Per riprendere: bash run_remaining.sh"
    show_dashboard
    exit 0
}
trap cleanup SIGINT SIGTERM

# Initialize status file
for block in BLOCK1 BLOCK2 BLOCK3; do
    if ! grep -q "^${block}=" "$STATUS_FILE" 2>/dev/null; then
        update_status "$block" "PENDING"
    fi
done

START_TIME=$(date +%s)
echo ""
echo "  ══════════════════════════════════════════════════════════"
echo "  FL-EHDS Remaining Experiments Cascade"
echo "  Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  ══════════════════════════════════════════════════════════"

show_dashboard

# ======================================================================
# BLOCK 1: Tabular Epochs Sweep (140 experiments)
# ======================================================================

B1_STATUS=$(grep "^BLOCK1=" "$STATUS_FILE" 2>/dev/null | cut -d= -f2 || echo "PENDING")
if [[ "$B1_STATUS" != "DONE" ]]; then
    echo ""
    echo "  ━━━ BLOCK 1: Tabular Epochs Sweep ━━━"
    update_status "BLOCK1" "RUNNING"

    if python -m benchmarks.run_tabular_epochs_sweep --resume $QUICK_FLAG 2>&1 | tee -a "$LOG_FILE"; then
        update_status "BLOCK1" "DONE"
        echo "  ✓ Block 1 completed"
    else
        update_status "BLOCK1" "PARTIAL"
        echo "  ⚠ Block 1 partial (checkpoint saved)"
    fi
    show_dashboard
else
    echo "  ⏭  Block 1 already DONE, skipping"
fi

# ======================================================================
# BLOCK 2: Top-K PTB-XL (9 experiments)
# ======================================================================

B2_STATUS=$(grep "^BLOCK2=" "$STATUS_FILE" 2>/dev/null | cut -d= -f2 || echo "PENDING")
if [[ "$B2_STATUS" != "DONE" ]]; then
    echo ""
    echo "  ━━━ BLOCK 2: Top-K PTB-XL ━━━"
    update_status "BLOCK2" "RUNNING"

    if python -m benchmarks.run_topk_ptbxl $QUICK_FLAG 2>&1 | tee -a "$LOG_FILE"; then
        update_status "BLOCK2" "DONE"
        echo "  ✓ Block 2 completed"
    else
        update_status "BLOCK2" "PARTIAL"
        echo "  ⚠ Block 2 partial (checkpoint saved)"
    fi
    show_dashboard
else
    echo "  ⏭  Block 2 already DONE, skipping"
fi

# ======================================================================
# BLOCK 3: Confusion Matrix Chest X-Ray (6 experiments, imaging)
# ======================================================================

B3_STATUS=$(grep "^BLOCK3=" "$STATUS_FILE" 2>/dev/null | cut -d= -f2 || echo "PENDING")
if [[ "$B3_STATUS" != "DONE" ]]; then
    echo ""
    echo "  ━━━ BLOCK 3: Confusion Matrix Chest X-Ray ━━━"
    update_status "BLOCK3" "RUNNING"

    if python -m benchmarks.run_confusion_matrix_chest $QUICK_FLAG 2>&1 | tee -a "$LOG_FILE"; then
        update_status "BLOCK3" "DONE"
        echo "  ✓ Block 3 completed"
    else
        update_status "BLOCK3" "PARTIAL"
        echo "  ⚠ Block 3 partial (checkpoint saved)"
    fi
    show_dashboard
else
    echo "  ⏭  Block 3 already DONE, skipping"
fi

# ======================================================================
# Summary
# ======================================================================

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(( (ELAPSED % 3600) / 60 ))
SECONDS=$((ELAPSED % 60))

echo ""
echo "  ══════════════════════════════════════════════════════════"
echo "  CASCADE COMPLETED"
echo "  Total time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "  ══════════════════════════════════════════════════════════"
show_dashboard

echo ""
echo "  Prossimi passi:"
echo "    cd $(pwd)"
echo "    git add -A && git commit -m 'vX.Y description' && git push"

#!/bin/bash
# ======================================================================
# FL-EHDS Extended Experiments — Cascade Runner
# ======================================================================
# Runs all 5 extended experiment blocks progressively with intermediate
# checkpoints. Each block auto-resumes if interrupted.
#
# Blocks:
#   1. DP Gradient Clipping Sensitivity  (72 exp, ~1h)
#   2. Centralized vs Federated          (45 exp, ~1.5h)
#   3. DP on PTB-XL                      (36 exp, ~1.5h)
#   4. Top-K Imaging                     (27 exp, ~3h)
#   5. Non-IID Imaging                   (54 exp, ~5h)
#
# Total: ~234 experiments, ~12h estimated
#
# Usage:
#   cd fl-ehds-framework
#   bash benchmarks/run_extended.sh          # full run
#   bash benchmarks/run_extended.sh --quick  # quick validation
#
# Each block saves its own checkpoint. If interrupted (Ctrl+C), the
# current experiment saves and you can re-run to resume from where
# you left off.
#
# Author: Fabio Liberti
# ======================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
FRAMEWORK_DIR="$(dirname "$SCRIPT_DIR")"
cd "$FRAMEWORK_DIR"

MODE="${1:---full}"
QUICK_FLAG=""
if [[ "$MODE" == "--quick" ]]; then
    QUICK_FLAG="--quick"
    echo "  MODE: QUICK (validation only)"
else
    echo "  MODE: FULL"
fi

STATUS_FILE="$SCRIPT_DIR/paper_results_tabular/extended_cascade_status.txt"
mkdir -p "$SCRIPT_DIR/paper_results_tabular"
mkdir -p "$SCRIPT_DIR/paper_results_delta"

GLOBAL_START=$(date +%s)
TOTAL_ERRORS=0

# ======================================================================
# Helper functions
# ======================================================================

update_status() {
    local block="$1"
    local status="$2"
    local tmpfile="${STATUS_FILE}.tmp"

    if [[ -f "$STATUS_FILE" ]]; then
        # Update existing block or append
        if grep -q "^${block}=" "$STATUS_FILE" 2>/dev/null; then
            sed "s/^${block}=.*/${block}=${status}/" "$STATUS_FILE" > "$tmpfile"
            mv "$tmpfile" "$STATUS_FILE"
        else
            echo "${block}=${status}" >> "$STATUS_FILE"
        fi
    else
        echo "${block}=${status}" > "$STATUS_FILE"
    fi
}

check_status() {
    local block="$1"
    if [[ -f "$STATUS_FILE" ]]; then
        local val
        val=$(grep "^${block}=" "$STATUS_FILE" 2>/dev/null | cut -d= -f2)
        echo "${val:-PENDING}"
    else
        echo "PENDING"
    fi
}

format_time() {
    local secs=$1
    printf "%dh %02dm %02ds" $((secs/3600)) $(((secs%3600)/60)) $((secs%60))
}

run_block() {
    local block_num="$1"
    local block_name="$2"
    local module="$3"
    local block_key="BLOCK${block_num}"

    local status
    status=$(check_status "$block_key")

    if [[ "$status" == "DONE" ]]; then
        echo ""
        echo "  [$block_num/5] $block_name — SKIPPED (already DONE)"
        return 0
    fi

    echo ""
    echo "======================================================================"
    echo "  [$block_num/5] $block_name"
    echo "======================================================================"

    update_status "$block_key" "IN_PROGRESS"

    local block_start
    block_start=$(date +%s)
    local exit_code=0

    python -m "$module" $QUICK_FLAG || exit_code=$?

    local block_end
    block_end=$(date +%s)
    local block_elapsed=$((block_end - block_start))

    if [[ $exit_code -eq 0 ]]; then
        update_status "$block_key" "DONE"
        echo ""
        echo "  [$block_num/5] $block_name — DONE ($(format_time $block_elapsed))"
    else
        update_status "$block_key" "ERROR (exit=$exit_code)"
        echo ""
        echo "  [$block_num/5] $block_name — ERROR (exit=$exit_code, $(format_time $block_elapsed))"
        TOTAL_ERRORS=$((TOTAL_ERRORS + 1))
    fi

    return $exit_code
}

# ======================================================================
# Banner
# ======================================================================

echo ""
echo "======================================================================"
echo "  FL-EHDS Extended Experiments — Cascade Runner"
echo "======================================================================"
echo "  Start:  $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Blocks: 5 (234 experiments total)"
echo "  Status: $STATUS_FILE"
echo "======================================================================"

# ======================================================================
# Block 1: DP Gradient Clipping Sensitivity (72 exp)
# ======================================================================

run_block 1 "DP Gradient Clipping Sensitivity (72 exp)" \
    "benchmarks.run_dp_clipping" || true

# ======================================================================
# Block 2: Centralized vs Federated (45 exp)
# ======================================================================

run_block 2 "Centralized vs Federated (45 exp)" \
    "benchmarks.run_centralized_vs_federated" || true

# ======================================================================
# Block 3: DP on PTB-XL (36 exp)
# ======================================================================

run_block 3 "DP on PTB-XL (36 exp)" \
    "benchmarks.run_dp_ptbxl" || true

# ======================================================================
# Block 4: Top-K Imaging (27 exp)
# ======================================================================

run_block 4 "Top-K Imaging (27 exp)" \
    "benchmarks.run_topk_imaging" || true

# ======================================================================
# Block 5: Non-IID Imaging (54 exp)
# ======================================================================

run_block 5 "Non-IID Imaging (54 exp)" \
    "benchmarks.run_noniid_imaging" || true

# ======================================================================
# Final Summary
# ======================================================================

GLOBAL_END=$(date +%s)
GLOBAL_ELAPSED=$((GLOBAL_END - GLOBAL_START))

echo ""
echo "======================================================================"
echo "  FL-EHDS Extended Experiments — COMPLETED"
echo "======================================================================"
echo "  End:    $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Total:  $(format_time $GLOBAL_ELAPSED)"
echo "  Errors: $TOTAL_ERRORS"
echo ""

for i in 1 2 3 4 5; do
    echo "  Block $i: $(check_status "BLOCK${i}")"
done

echo ""
echo "  Checkpoints:"
echo "    paper_results_tabular/checkpoint_dp_clipping.json"
echo "    paper_results_tabular/checkpoint_centralized_vs_fed.json"
echo "    paper_results_tabular/checkpoint_dp_ptbxl.json"
echo "    paper_results_delta/checkpoint_topk_imaging.json"
echo "    paper_results_delta/checkpoint_noniid_imaging.json"
echo "======================================================================"

#!/bin/bash
# =============================================================================
# FL-EHDS Paper Experiments Launcher (macOS)
# =============================================================================
# - caffeinate: prevents macOS sleep during long runs
# - tee: mirrors output to timestamped log file
# - trap: reports duration on exit (normal or interrupted)
#
# Usage:
#   ./run_experiments.sh                        # Full run with resume
#   ./run_experiments.sh --only p12             # Only P1.2 block
#   ./run_experiments.sh --only p12 --dataset Brain_Tumor  # Single dataset
#   ./run_experiments.sh --quick                # Quick test (1 seed, 5 rounds)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Log directory
LOG_DIR="benchmarks/paper_results/logs"
mkdir -p "$LOG_DIR"

# Timestamped log file
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/experiment_${TIMESTAMP}.log"

# Duration tracking
START_TIME=$(date +%s)
cleanup() {
    local END_TIME=$(date +%s)
    local DURATION=$(( END_TIME - START_TIME ))
    local HOURS=$(( DURATION / 3600 ))
    local MINUTES=$(( (DURATION % 3600) / 60 ))
    local SECONDS=$(( DURATION % 60 ))
    echo ""
    echo "=============================================="
    echo "  Finished: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "  Duration: ${HOURS}h ${MINUTES}m ${SECONDS}s"
    echo "  Log: ${LOG_FILE}"
    echo "=============================================="
}
trap cleanup EXIT

echo "=============================================="
echo "  FL-EHDS Paper Experiments"
echo "  Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Log:     ${LOG_FILE}"
echo "  Args:    --resume $*"
echo "=============================================="

# caffeinate -i: prevent idle sleep
# caffeinate -m: prevent disk sleep
# caffeinate -s: prevent system sleep
caffeinate -ims python -m benchmarks.run_paper_experiments --resume "$@" 2>&1 | tee "$LOG_FILE"

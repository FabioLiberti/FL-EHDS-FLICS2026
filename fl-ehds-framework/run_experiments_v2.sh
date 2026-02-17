#!/bin/bash
# FL-EHDS Experiments V2 — Mac launcher
#
# Usage:
#   ./run_experiments_v2.sh              # Full experiments (105 exp, ~35-49h)
#   ./run_experiments_v2.sh --quick      # Quick validation (35 exp, ~1-2h)
#   ./run_experiments_v2.sh --resume     # Resume from checkpoint
#   ./run_experiments_v2.sh --quick --resume  # Resume quick validation
#
# Output: benchmarks/paper_results/checkpoint_v2.json
# Log:    benchmarks/paper_results/experiment_v2.log

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

TIMESTAMP=$(date "+%Y%m%d_%H%M%S")
LOG_DIR="benchmarks/paper_results"
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "  FL-EHDS Experiments V2"
echo "  Started: $(date)"
echo "  Args: $@"
echo "=========================================="

# Prevent Mac from sleeping during long experiments
if command -v caffeinate &> /dev/null; then
    echo "  Using caffeinate to prevent sleep"
    caffeinate -dims python -m benchmarks.run_full_experiments "$@" 2>&1 | tee "${LOG_DIR}/run_v2_${TIMESTAMP}.log"
else
    python -m benchmarks.run_full_experiments "$@" 2>&1 | tee "${LOG_DIR}/run_v2_${TIMESTAMP}.log"
fi

DURATION=$SECONDS
echo ""
echo "=========================================="
echo "  Completed in ${DURATION}s ($(( DURATION / 3600 ))h $(( (DURATION % 3600) / 60 ))m)"
echo "  Finished: $(date)"
echo "=========================================="

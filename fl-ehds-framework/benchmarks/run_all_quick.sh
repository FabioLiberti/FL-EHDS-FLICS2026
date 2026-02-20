#!/bin/bash
# =================================================================
# FL-EHDS — Run ALL Tests (QUICK validation mode)
# Estimated time: ~15-20 minutes
# =================================================================
set -e

PYTHON="/opt/anaconda3/envs/flics2026/bin/python"
DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$DIR"

echo "================================================================="
echo "  FL-EHDS — ALL TESTS (QUICK MODE)"
echo "  Working dir: $DIR"
echo "  Python: $PYTHON"
echo "  Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================="
echo ""

# Test B — Confusion Matrix BC (~2 min)
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  [1/5] Test B — Confusion Matrix Breast Cancer (QUICK)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
$PYTHON -m benchmarks.run_confusion_matrix_bc --quick
echo ""

# Test C — RDP vs Naive (~10 sec)
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  [2/5] Test C — RDP vs Naive Composition (analytical)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
$PYTHON -m benchmarks.run_rdp_comparison
echo ""

# Test E — Top-k Sparsification (~2 min)
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  [3/5] Test E — Top-k Sparsification PTB-XL (QUICK)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
$PYTHON -m benchmarks.run_topk_ptbxl --quick --fresh
echo ""

# Test F — Scalability x DP (~1 min)
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  [4/5] Test F — Scalability x DP PTB-XL (QUICK)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
$PYTHON -m benchmarks.run_scalability_dp --quick --fresh
echo ""

# Test A+D — Imaging Extended (~10 min)
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  [5/5] Test A+D — HPFL+FedLESAM Chest X-Ray (QUICK)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
$PYTHON -m benchmarks.run_imaging_extended --quick --fresh
echo ""

echo "================================================================="
echo "  ALL QUICK TESTS COMPLETED"
echo "  Finished: $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================="

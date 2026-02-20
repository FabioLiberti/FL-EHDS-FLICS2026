#!/bin/bash
# =================================================================
# FL-EHDS — Run ALL Tests (FULL production mode)
# Estimated time: ~8-12 hours
#
# Order: fastest → slowest, so partial results are available early.
# All scripts support Ctrl+C (saves checkpoint) and auto-resume.
#
# To resume after interruption:
#   bash benchmarks/run_all_full.sh
# (already-completed tests are skipped via checkpoint)
# =================================================================
set -e

PYTHON="/opt/anaconda3/envs/flics2026/bin/python"
DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$DIR"

echo "================================================================="
echo "  FL-EHDS — ALL TESTS (FULL MODE)"
echo "  Working dir: $DIR"
echo "  Python: $PYTHON"
echo "  Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "  Test B  — Confusion Matrix BC         (~15 min)"
echo "  Test C  — RDP vs Naive (analytical)   (~10 sec)"
echo "  Test E  — Top-k Sparsification PTB-XL (~45 min)"
echo "  Test F  — Scalability x DP PTB-XL     (~2-3 hrs)"
echo "  Test AD — HPFL+FedLESAM Chest X-Ray   (~3-5 hrs)"
echo "  Total estimated: ~7-9 hours"
echo "================================================================="
echo ""

# Test B — Confusion Matrix BC (~15 min)
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  [1/5] Test B — Confusion Matrix Breast Cancer"
echo "  4 algos × 3 seeds = 12 experiments"
echo "  Output: paper_results_tabular/confusion_matrix_bc.pdf"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
START_B=$(date +%s)
$PYTHON -m benchmarks.run_confusion_matrix_bc
END_B=$(date +%s)
echo "  Test B completed in $(( (END_B - START_B) / 60 )) min"
echo ""

# Test C — RDP vs Naive (~10 sec, analytical)
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  [2/5] Test C — RDP vs Naive Composition (analytical)"
echo "  Output: paper_results_tabular/rdp_vs_naive_composition.pdf"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
$PYTHON -m benchmarks.run_rdp_comparison
echo ""

# Test E — Top-k Sparsification (~45 min)
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  [3/5] Test E — Top-k Sparsification PTB-XL"
echo "  3 k_ratios × 3 seeds = 9 experiments"
echo "  Output: paper_results_tabular/checkpoint_topk_ptbxl.json"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
START_E=$(date +%s)
$PYTHON -m benchmarks.run_topk_ptbxl --fresh
END_E=$(date +%s)
echo "  Test E completed in $(( (END_E - START_E) / 60 )) min"
echo ""

# Test F — Scalability x DP (~2-3 hrs)
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  [4/5] Test F — Scalability x DP PTB-XL"
echo "  3 algos × 3 K × 2 DP × 3 seeds = 54 experiments"
echo "  Output: paper_results_tabular/checkpoint_scalability_dp.json"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
START_F=$(date +%s)
$PYTHON -m benchmarks.run_scalability_dp --fresh
END_F=$(date +%s)
echo "  Test F completed in $(( (END_F - START_F) / 60 )) min"
echo ""

# Test A+D — Imaging Extended (~3-5 hrs)
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  [5/5] Test A+D — HPFL + FedLESAM Chest X-Ray"
echo "  2 algos × 3 seeds = 6 experiments (ResNet-18)"
echo "  Output: paper_results_delta/checkpoint_chest_extended.json"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
START_AD=$(date +%s)
$PYTHON -m benchmarks.run_imaging_extended --fresh
END_AD=$(date +%s)
echo "  Test A+D completed in $(( (END_AD - START_AD) / 60 )) min"
echo ""

# Final summary
TOTAL_END=$(date +%s)
TOTAL_START=$START_B
echo "================================================================="
echo "  ALL FULL TESTS COMPLETED"
echo "  Total time: $(( (TOTAL_END - TOTAL_START) / 3600 ))h $(( ((TOTAL_END - TOTAL_START) % 3600) / 60 ))m"
echo "  Finished: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "  Output files:"
echo "    paper_results_tabular/confusion_matrix_bc.pdf"
echo "    paper_results_tabular/rdp_vs_naive_composition.pdf"
echo "    paper_results_tabular/checkpoint_topk_ptbxl.json"
echo "    paper_results_tabular/checkpoint_scalability_dp.json"
echo "    paper_results_delta/checkpoint_chest_extended.json"
echo "================================================================="

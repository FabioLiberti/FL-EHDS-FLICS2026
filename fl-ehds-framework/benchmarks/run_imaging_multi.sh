#!/bin/bash
# =================================================================
# FL-EHDS — Brain Tumor + Skin Cancer Imaging Experiments
# Estimated time: ~2-3 hours (MPS/CUDA), ~4-5 hours (CPU)
#
# 2 datasets × 2 algos (FedAvg, HPFL) × 3 seeds = 12 experiments
# Validates modality-dependent personalization across 3 imaging datasets.
#
# Supports Ctrl+C (saves checkpoint) and auto-resume.
# To resume after interruption, just re-run this script.
#
# Usage:
#   bash benchmarks/run_imaging_multi.sh          # full run
#   bash benchmarks/run_imaging_multi.sh --quick   # quick validation (~5 min)
#   bash benchmarks/run_imaging_multi.sh --fresh   # discard checkpoint, start fresh
# =================================================================
set -e

PYTHON="/opt/anaconda3/envs/flics2026/bin/python"
DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$DIR"

echo "================================================================="
echo "  FL-EHDS — Brain Tumor + Skin Cancer Imaging"
echo "  Working dir: $DIR"
echo "  Python: $PYTHON"
echo "  Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "  Datasets:  Brain Tumor (4-class), Skin Cancer (binary)"
echo "  Algorithms: FedAvg, HPFL"
echo "  Seeds:     42, 123, 456"
echo "  Total:     12 experiments"
echo ""
echo "  Output: benchmarks/paper_results_delta/checkpoint_imaging_multi.json"
echo "================================================================="
echo ""

START=$(date +%s)

$PYTHON -m benchmarks.run_imaging_multi "$@"

END=$(date +%s)
ELAPSED=$(( END - START ))
HOURS=$(( ELAPSED / 3600 ))
MINS=$(( (ELAPSED % 3600) / 60 ))

echo ""
echo "================================================================="
echo "  IMAGING MULTI COMPLETED"
echo "  Total time: ${HOURS}h ${MINS}m"
echo "  Finished: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "  Output:"
echo "    benchmarks/paper_results_delta/checkpoint_imaging_multi.json"
echo "================================================================="

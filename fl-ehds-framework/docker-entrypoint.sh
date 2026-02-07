#!/bin/bash
# =============================================================================
# FL-EHDS Framework - Docker Entrypoint
# =============================================================================
# Switches between terminal, dashboard, and benchmark modes based on
# the FL_EHDS_MODE environment variable.
# =============================================================================

set -e

case "${FL_EHDS_MODE}" in
    terminal)
        echo "Starting FL-EHDS Terminal CLI..."
        exec python -m terminal
        ;;
    dashboard)
        echo "Starting FL-EHDS Streamlit Dashboard on port 8501..."
        exec streamlit run dashboard/app_v4.py \
            --server.port=8501 \
            --server.address=0.0.0.0 \
            --server.headless=true \
            --browser.gatherUsageStats=false
        ;;
    benchmark)
        echo "Starting FL-EHDS Benchmark Runner..."
        exec python -m benchmarks.run_experiments "$@"
        ;;
    experiment)
        echo "Starting FL-EHDS Experiment: $*"
        exec python "$@"
        ;;
    *)
        echo "FL-EHDS Framework v1.0.0"
        echo ""
        echo "Unknown mode: '${FL_EHDS_MODE}'"
        echo ""
        echo "Available modes (set via FL_EHDS_MODE):"
        echo "  terminal   - Interactive CLI interface"
        echo "  dashboard  - Streamlit web dashboard"
        echo "  benchmark  - Automated benchmark runner"
        echo "  experiment - Run custom Python script"
        exit 1
        ;;
esac

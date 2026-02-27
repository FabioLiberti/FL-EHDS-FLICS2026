#!/bin/bash
# ======================================================================
# FL-EHDS Sync Pull
# ======================================================================
# Fa git pull e mostra la dashboard con lo stato degli esperimenti.
# Usare al posto di "git pull" per avere il quadro aggiornato.
#
# Usage:
#   bash sync_pull.sh
# ======================================================================

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_DIR"

STATUS_FILE="EXPERIMENT_STATUS.md"

# ======================================================================
# Git pull
# ======================================================================

echo ""
echo "  Pulling latest..."
PULL_RESULT=$(git pull --rebase origin main 2>&1) || true

if echo "$PULL_RESULT" | grep -q "Already up to date"; then
    PULL_MSG="gia aggiornato"
else
    PULL_MSG="aggiornato!"
fi

# ======================================================================
# Dashboard
# ======================================================================

if [[ ! -f "$STATUS_FILE" ]]; then
    echo "  EXPERIMENT_STATUS.md non trovato!"
    exit 1
fi

LAST_UPDATE=$(grep "^\*\*Last update:\*\*" "$STATUS_FILE" | sed 's/\*\*Last update:\*\* //' | sed 's/\*\*//')

echo ""
echo "  ╔═══════════════════════════════════════════════════════════╗"
echo "  ║  FL-EHDS Experiment Status ($PULL_MSG)  ║"
echo "  ╠═══════════════════════════════════════════════════════════╣"
echo "  ║  Ultimo sync: $LAST_UPDATE"
echo "  ║"

# Machines
echo "  ║  Macchine:"
grep "^|" "$STATUS_FILE" | grep -E "MacBook|RunPod|Colab" | while IFS='|' read -r _ name role status _; do
    name=$(echo "$name" | xargs)
    status=$(echo "$status" | xargs)
    printf "  ║    %-20s  %s\n" "$name" "$status"
done

echo "  ║"

# Remaining gaps (in-progress and pending)
echo "  ║  Esperimenti attivi:"
grep "^|" "$STATUS_FILE" | grep -iE "IN PROGRESS|PENDING|In corso" | grep -v "^| Experiment" | grep -v "^| Priority" | grep -v "^|---" | head -10 | while IFS='|' read -r _ col1 col2 col3 col4 col5 _; do
    col1=$(echo "$col1" | xargs)
    # Detect which table format (inventory vs gaps)
    if [[ -n "$col5" ]]; then
        # 5 columns: inventory table (name|script|checkpoint|n|status)
        n_exp=$(echo "$col4" | xargs)
        status=$(echo "$col5" | xargs)
    else
        # 4 columns: gaps table (priority|name|n|where|status) - but read differently
        n_exp=$(echo "$col3" | xargs)
        status=$(echo "$col4" | xargs)
    fi
    if [[ -n "$col1" && "$col1" != "---" ]]; then
        printf "  ║    %-30s  %6s  %s\n" "$col1" "$n_exp" "$status"
    fi
done

echo "  ║"
echo "  ╚═══════════════════════════════════════════════════════════╝"
echo ""

#!/bin/bash
# ======================================================================
# FL-EHDS Sync Push
# ======================================================================
# Legge i checkpoint, aggiorna EXPERIMENT_STATUS.md, commit & push.
# Usare al posto di "git add/commit/push" per sincronizzare lo stato.
#
# Usage:
#   bash sync_push.sh "v12.6 descrizione commit"
#   bash sync_push.sh                              # solo sync status (no altri file)
# ======================================================================

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_DIR"

FRAMEWORK="fl-ehds-framework/benchmarks"
STATUS_FILE="EXPERIMENT_STATUS.md"
COMMIT_MSG="${1:-}"

# ======================================================================
# Lettura checkpoint
# ======================================================================

read_checkpoint() {
    local file="$1"
    if [[ -f "$file" ]]; then
        python3 -c "
import json
with open('$file') as f:
    d = json.load(f)
if 'completed' in d and isinstance(d['completed'], dict):
    print(len(d['completed']))
else:
    print(0)
" 2>/dev/null || echo "0"
    else
        echo "0"
    fi
}

read_block_status() {
    local status_file="${FRAMEWORK}/paper_results_tabular/cascade_remaining_status.txt"
    local block="$1"
    if [[ -f "$status_file" ]]; then
        grep "^${block}=" "$status_file" 2>/dev/null | cut -d= -f2 || echo "UNKNOWN"
    else
        echo "UNKNOWN"
    fi
}

# ======================================================================
# Aggiorna EXPERIMENT_STATUS.md
# ======================================================================

echo ""
echo "  Reading checkpoints..."

EPOCHS_DONE=$(read_checkpoint "${FRAMEWORK}/paper_results_tabular/checkpoint_epochs_sweep.json")
TOPK_DONE=$(read_checkpoint "${FRAMEWORK}/paper_results_tabular/checkpoint_topk_ptbxl.json")
CM_CHEST_DONE=$(read_checkpoint "${FRAMEWORK}/paper_results_delta/checkpoint_confusion_chest.json")

B1=$(read_block_status "BLOCK1")
B2=$(read_block_status "BLOCK2")
B3=$(read_block_status "BLOCK3")

TIMESTAMP=$(date '+%Y-%m-%dT%H:%M:%S CET')
MACHINE=$(hostname -s)

# Status per esperimento
status_str() {
    local block_status="$1" done="$2" total="$3"
    if [[ "$block_status" == "DONE" ]]; then echo "DONE"
    elif [[ "$done" -gt 0 ]]; then echo "IN PROGRESS ($done/$total)"
    else echo "PENDING"
    fi
}

ES=$(status_str "$B1" "$EPOCHS_DONE" 140)
TS=$(status_str "$B2" "$TOPK_DONE" 9)
CS=$(status_str "$B3" "$CM_CHEST_DONE" 6)

# Machine status
if [[ "$B1" == "DONE" && "$B2" == "DONE" && "$B3" == "DONE" ]]; then
    MS="Libero (cascade completata)"
else
    MS="In corso: cascade (${EPOCHS_DONE}+${TOPK_DONE}+${CM_CHEST_DONE}/155)"
fi

python3 << PYEOF
import re

with open("$STATUS_FILE") as f:
    content = f.read()

# Timestamp
content = re.sub(
    r'\*\*Last update:\*\*.*',
    '**Last update:** ${TIMESTAMP} (${MACHINE})',
    content
)

# Machine status Air M3
content = re.sub(
    r'(\| MacBook Air M3 \|[^|]*\|)[^|]*\|',
    r'\1 ${MS} |',
    content
)

# Epochs Sweep row
content = re.sub(
    r'(\| Epochs Sweep \|[^|]*\|[^|]*\|[^|]*\|)[^|]*\|',
    r'\1 ${ES} |',
    content
)

# Top-K PTB-XL row
content = re.sub(
    r'(\| Top-K PTB-XL \|[^|]*\|[^|]*\|[^|]*\|)[^|]*\|',
    r'\1 ${TS} |',
    content
)

# Confusion Matrix Chest row
content = re.sub(
    r'(\| Confusion Matrix Chest \|[^|]*\|[^|]*\|[^|]*\|)[^|]*\|',
    r'\1 ${CS} |',
    content
)

# Remaining gaps
def update_gap(content, name, status):
    pattern = r'(\|[^|]*\| ' + re.escape(name) + r'[^|]*\|[^|]*\|[^|]*\|)[^|]*\|'
    return re.sub(pattern, r'\1 ' + status + ' |', content)

content = update_gap(content, "Epochs Sweep", "${ES}")
content = update_gap(content, "Top-K PTB-XL", "${TS}")
content = update_gap(content, "Confusion Matrix Chest", "${CS}")

with open("$STATUS_FILE", "w") as f:
    f.write(content)
PYEOF

# ======================================================================
# Dashboard
# ======================================================================

echo ""
echo "  ╔═══════════════════════════════════════════════════╗"
echo "  ║  FL-EHDS Sync Push                               ║"
echo "  ╠═══════════════════════════════════════════════════╣"
printf "  ║  %-16s  %8s  %-18s  ║\n" "Experiment" "Progress" "Status"
echo "  ║  ────────────────  ────────  ──────────────────  ║"
printf "  ║  %-16s  %3s/%-4s  %-18s  ║\n" "Epochs Sweep" "$EPOCHS_DONE" "140" "$B1"
printf "  ║  %-16s  %3s/%-4s  %-18s  ║\n" "Top-K PTB-XL" "$TOPK_DONE" "9" "$B2"
printf "  ║  %-16s  %3s/%-4s  %-18s  ║\n" "CM Chest" "$CM_CHEST_DONE" "6" "$B3"
echo "  ╚═══════════════════════════════════════════════════╝"

# ======================================================================
# Git commit & push
# ======================================================================

# Stage status file (sempre)
git add "$STATUS_FILE"

# Se c'e un messaggio di commit, stage anche tutti gli altri file modificati
if [[ -n "$COMMIT_MSG" ]]; then
    git add -A
    echo ""
    echo "  Commit: $COMMIT_MSG"
    git commit -m "$COMMIT_MSG"
else
    # Solo status file
    if ! git diff --cached --quiet 2>/dev/null; then
        git commit -m "auto-sync: experiment progress (epochs=$EPOCHS_DONE/140 topk=$TOPK_DONE/9 cm_chest=$CM_CHEST_DONE/6)"
    else
        echo ""
        echo "  Nessun cambiamento da committare."
        exit 0
    fi
fi

echo "  Pushing..."
git push origin main
echo ""
echo "  Sync completato."

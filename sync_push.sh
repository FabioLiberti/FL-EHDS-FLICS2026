#!/bin/bash
# ======================================================================
# FL-EHDS Sync Push
# ======================================================================
# Legge i checkpoint, aggiorna EXPERIMENT_STATUS.md, commit & push.
#
# Usage:
#   bash sync_push.sh "v12.6 descrizione commit"
#   bash sync_push.sh                              # solo sync status
# ======================================================================

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_DIR"

FRAMEWORK="fl-ehds-framework/benchmarks"
STATUS_FILE="EXPERIMENT_STATUS.md"
COMMIT_MSG="${1:-}"

# ======================================================================
# Lettura checkpoint e aggiornamento status
# ======================================================================

python3 << 'PYEOF'
import json, os, subprocess
from datetime import datetime
from pathlib import Path

repo = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()
fw = Path(repo) / "fl-ehds-framework" / "benchmarks"
status_file = Path(repo) / "EXPERIMENT_STATUS.md"

def read_ckpt(path):
    try:
        with open(path) as f:
            d = json.load(f)
        if "completed" in d and isinstance(d["completed"], dict):
            return len(d["completed"])
    except:
        pass
    return 0

def read_block(block):
    sf = fw / "paper_results_tabular" / "cascade_remaining_status.txt"
    if sf.exists():
        for line in sf.read_text().splitlines():
            if line.startswith(f"{block}="):
                return line.split("=", 1)[1].strip()
    return "UNKNOWN"

epochs_done = read_ckpt(fw / "paper_results_tabular" / "checkpoint_epochs_sweep.json")
topk_done = read_ckpt(fw / "paper_results_tabular" / "checkpoint_topk_ptbxl.json")
cm_done = read_ckpt(fw / "paper_results_delta" / "checkpoint_confusion_chest.json")

b1, b2, b3 = read_block("BLOCK1"), read_block("BLOCK2"), read_block("BLOCK3")

def exp_status(bs, done, total):
    if bs == "DONE": return "DONE"
    elif done > 0: return f"IN PROGRESS ({done}/{total})"
    return "PENDING"

es, ts, cs = exp_status(b1, epochs_done, 140), exp_status(b2, topk_done, 9), exp_status(b3, cm_done, 6)

if b1 == "DONE" and b2 == "DONE" and b3 == "DONE":
    ms = "Libero (cascade completata)"
elif epochs_done > 0 or topk_done > 0 or cm_done > 0:
    ms = f"In corso: cascade ({epochs_done}+{topk_done}+{cm_done}/155)"
else:
    ms = None

hostname = os.uname().nodename.split(".")[0]
timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S CET")

lines = status_file.read_text().splitlines()
new_lines = []

for line in lines:
    if line.startswith("**Last update:**"):
        line = f"**Last update:** {timestamp} ({hostname})"
    elif ms and line.startswith("| MacBook Air M3 |"):
        parts = line.split("|")
        if len(parts) >= 5:
            parts[3] = f" {ms} "
            line = "|".join(parts)
    elif line.startswith("| Epochs Sweep |"):
        parts = line.split("|")
        if len(parts) >= 7:
            parts[4], parts[5] = " 140 ", f" {es} "
            line = "|".join(parts)
    elif line.startswith("| Top-K PTB-XL |"):
        parts = line.split("|")
        if len(parts) >= 7:
            parts[4], parts[5] = " 9 ", f" {ts} "
            line = "|".join(parts)
    elif line.startswith("| Confusion Matrix Chest |"):
        parts = line.split("|")
        if len(parts) >= 7:
            parts[4], parts[5] = " 6 ", f" {cs} "
            line = "|".join(parts)
    new_lines.append(line)

status_file.write_text("\n".join(new_lines) + "\n")

# Dashboard
print()
print(f"  Epochs Sweep:  {epochs_done}/140  ({b1})")
print(f"  Top-K PTB-XL:  {topk_done}/9    ({b2})")
print(f"  CM Chest:      {cm_done}/6    ({b3})")
print()
PYEOF

# ======================================================================
# Git commit & push
# ======================================================================

git add "$STATUS_FILE"

if [[ -n "$COMMIT_MSG" ]]; then
    git add -A
    echo "  Commit: $COMMIT_MSG"
    git commit -m "$COMMIT_MSG"
else
    if ! git diff --cached --quiet 2>/dev/null; then
        git commit -m "auto-sync: experiment progress update"
    else
        echo "  Nessun cambiamento da committare."
        exit 0
    fi
fi

echo "  Pushing..."
git push origin main
echo "  Sync completato."

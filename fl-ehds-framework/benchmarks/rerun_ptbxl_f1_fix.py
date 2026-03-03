#!/usr/bin/env python3
"""
Re-run PTB-XL experiments to fix the binary F1 metric bug.

The bug: _evaluate() in federated.py computed F1 using binary (class 0 vs 1)
logic on PTB-XL (a 5-class dataset), producing F1=1.0 artifacts.

Fix applied: _evaluate() now uses sklearn macro-averaged F1 for multiclass.

Strategy:
  - For multi-dataset checkpoints: remove PTB-XL keys, then re-run
    (existing binary-dataset results are preserved)
  - For PTB-XL-only checkpoints: use --fresh

Usage:
    cd fl-ehds-framework
    python -m benchmarks.rerun_ptbxl_f1_fix [--dry-run] [--step N]

Options:
    --dry-run   Show what would be done without executing
    --step N    Run only step N (1-11), useful for sequential execution

Estimated time: ~96 minutes total on MacBook Air M3.

Author: Fabio Liberti
"""

import json
import os
import sys
import subprocess
import shutil
import argparse
from pathlib import Path

CHECKPOINT_DIR = Path(__file__).parent / "paper_results_tabular"

# ── Step definitions ──
# Each step: (checkpoint_file, ptbxl_key_prefix, script_module, extra_args, description)
# ptbxl_key_prefix=None means PTB-XL-only script (use --fresh)

STEPS = [
    # Step 1: Main tabular (35 PTB-XL exp, ~4 min)
    (
        "checkpoint_tabular.json",
        "PTB_XL_",
        "benchmarks.run_tabular_optimized",
        ["--resume"],
        "Main tabular: 7 algo x 5 seeds (PTB-XL only) ~4 min",
    ),
    # Step 2: Seeds10 (35 PTB-XL exp, ~4 min)
    (
        "checkpoint_seeds10.json",
        "PTB_XL_",
        "benchmarks.run_tabular_seeds10",
        [],
        "Seeds10: 7 algo x 5 additional seeds (PTB-XL only) ~4 min",
    ),
    # Step 3: DP ablation (60 PTB-XL exp, ~10 min)
    (
        "checkpoint_dp.json",
        "PTB_XL_",
        "benchmarks.run_tabular_dp",
        [],
        "DP ablation: 7 algo x 4 eps x ~2 seeds (PTB-XL only) ~10 min",
    ),
    # Step 4: Opt-out (75 PTB-XL exp, ~12 min)
    (
        "checkpoint_optout.json",
        "PTB_XL_",
        "benchmarks.run_tabular_optout",
        [],
        "Opt-out: Article 71 simulation (PTB-XL only) ~12 min",
    ),
    # Step 5: Scalability (42 PTB-XL exp, ~5 min)
    (
        "checkpoint_scalability.json",
        "PX_",
        "benchmarks.run_scalability_sweep",
        ["--resume"],
        "Scalability: K=2..30 (PTB-XL only) ~5 min",
    ),
    # Step 6: Deep MLP (35 PTB-XL exp, ~16 min)
    (
        "checkpoint_deep_mlp.json",
        "PTB_XL_",
        "benchmarks.run_tabular_deep_mlp",
        [],
        "Deep MLP: 110K params (PTB-XL only) ~16 min",
    ),
    # Step 7: Scalability+DP (54 exp, PTB-XL only script, ~9 min)
    (
        "checkpoint_scalability_dp.json",
        None,  # PTB-XL only → --fresh
        "benchmarks.run_scalability_dp",
        ["--fresh"],
        "Scalability+DP: PTB-XL only script ~9 min",
    ),
    # Step 8: DP clipping (36 PTB-XL exp, ~7 min)
    (
        "checkpoint_dp_clipping.json",
        "PX_",
        "benchmarks.run_dp_clipping",
        [],
        "DP clipping analysis (PTB-XL only) ~7 min",
    ),
    # Step 9: DP PTB-XL (36 exp, PTB-XL only script, ~7 min)
    (
        "checkpoint_dp_ptbxl.json",
        None,  # PTB-XL only → --fresh
        "benchmarks.run_dp_ptbxl",
        ["--fresh"],
        "DP PTB-XL: dedicated script ~7 min",
    ),
    # Step 10: Cross-border DP (45 exp, PTB-XL only script, ~10 min)
    (
        "checkpoint_crossborder_dp.json",
        None,  # PTB-XL only → --fresh
        "benchmarks.run_crossborder_dp",
        ["--fresh"],
        "Cross-border DP: heterogeneous budgets ~10 min",
    ),
    # Step 11: Dynamic opt-out (45 PTB-XL exp, ~6 min)
    (
        "checkpoint_dynamic_optout.json",
        "PTB_XL_",
        "benchmarks.run_dynamic_optout",
        [],
        "Dynamic opt-out: mid-training withdrawal (PTB-XL only) ~6 min",
    ),
]


def remove_ptbxl_keys(checkpoint_path, prefix):
    """Remove PTB-XL experiment keys from a checkpoint, preserving other datasets."""
    with open(checkpoint_path) as f:
        data = json.load(f)

    completed = data.get("completed", {})
    ptbxl_keys = [k for k in completed if k.startswith(prefix)]

    if not ptbxl_keys:
        print("  No keys matching '{}*' found — nothing to remove".format(prefix))
        return 0

    for k in ptbxl_keys:
        del completed[k]

    # Backup before modifying
    backup_path = str(checkpoint_path) + ".pre_f1fix.bak"
    if not os.path.exists(backup_path):
        shutil.copy2(checkpoint_path, backup_path)
        print("  Backup saved: {}".format(backup_path))

    with open(checkpoint_path, "w") as f:
        json.dump(data, f, indent=2)

    print("  Removed {} PTB-XL keys (prefix '{}')".format(len(ptbxl_keys), prefix))
    return len(ptbxl_keys)


def backup_fresh_checkpoint(checkpoint_path):
    """Backup a PTB-XL-only checkpoint before --fresh overwrites it."""
    backup_path = str(checkpoint_path) + ".pre_f1fix.bak"
    if not os.path.exists(backup_path) and os.path.exists(checkpoint_path):
        shutil.copy2(checkpoint_path, backup_path)
        print("  Backup saved: {}".format(backup_path))


def main():
    parser = argparse.ArgumentParser(description="Re-run PTB-XL experiments with fixed F1 metric")
    parser.add_argument("--dry-run", action="store_true", help="Show plan without executing")
    parser.add_argument("--step", type=int, help="Run only step N (1-11)")
    args = parser.parse_args()

    print("=" * 70)
    print("PTB-XL F1 Fix Re-Run Script")
    print("=" * 70)
    print()
    print("Bug: _evaluate() used binary F1 on 5-class PTB-XL → F1=1.0 artifact")
    print("Fix: macro-averaged F1 via sklearn (already applied to federated.py)")
    print("Affected: 539 PTB-XL experiments across 11 checkpoints")
    print("Estimated time: ~96 minutes")
    print()

    if args.step:
        steps_to_run = [(args.step - 1, STEPS[args.step - 1])]
    else:
        steps_to_run = list(enumerate(STEPS))

    total_removed = 0

    for idx, (ckpt_file, prefix, module, extra_args, desc) in steps_to_run:
        step_num = idx + 1
        ckpt_path = CHECKPOINT_DIR / ckpt_file

        print("-" * 70)
        print("Step {}/11: {}".format(step_num, desc))
        print("  Checkpoint: {}".format(ckpt_file))

        if not ckpt_path.exists():
            print("  WARNING: Checkpoint not found, skipping")
            continue

        if args.dry_run:
            if prefix is None:
                print("  [DRY-RUN] Would backup and use --fresh")
            else:
                with open(ckpt_path) as f:
                    data = json.load(f)
                n = sum(1 for k in data.get("completed", {}) if k.startswith(prefix))
                print("  [DRY-RUN] Would remove {} PTB-XL keys (prefix '{}')".format(n, prefix))
            print("  [DRY-RUN] Would run: python -m {} {}".format(module, " ".join(extra_args)))
            continue

        # Phase 1: Remove PTB-XL keys (or backup for --fresh)
        if prefix is None:
            backup_fresh_checkpoint(ckpt_path)
        else:
            n = remove_ptbxl_keys(ckpt_path, prefix)
            total_removed += n

        # Phase 2: Re-run the script
        cmd = [sys.executable, "-m", module] + extra_args
        print("  Running: {}".format(" ".join(cmd)))
        print()

        result = subprocess.run(cmd, cwd=str(Path(__file__).parent.parent))

        if result.returncode != 0:
            print("\n  ERROR: Step {} failed with return code {}".format(step_num, result.returncode))
            print("  You can resume from this step with: --step {}".format(step_num))
            sys.exit(1)

        print("\n  Step {} completed successfully.".format(step_num))

    print()
    print("=" * 70)
    if args.dry_run:
        print("DRY-RUN complete. No changes made.")
    else:
        print("All steps completed! PTB-XL F1 metrics are now correct (macro-averaged).")
        print()
        print("Next: update the paper Table 3 F1 column with new values.")
    print("=" * 70)


if __name__ == "__main__":
    main()

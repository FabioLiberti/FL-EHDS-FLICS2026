#!/usr/bin/env python3
"""
Action #1: Expand imaging experiments from 3 seeds to 5 seeds.

Adds seed 789 and 999 for all imaging datasets × all algorithms.
Results are saved to the SAME checkpoint (checkpoint_p12_multidataset.json)
so they seamlessly merge with existing 3-seed results.

Existing experiments are NEVER re-run (resume-safe).

Experiments: 3 datasets × 5 algorithms × 2 new seeds = 30 runs
Estimated time: ~6-8 hours on Mac Air M2/M3

Usage:
    conda activate flics2026
    cd fl-ehds-framework
    python benchmarks/run_imaging_seeds5.py [--resume] [--dry-run] [--dataset DATASET] [--algo ALGO]

Examples:
    # Run all missing seed experiments
    python benchmarks/run_imaging_seeds5.py

    # Only Brain_Tumor
    python benchmarks/run_imaging_seeds5.py --dataset Brain_Tumor

    # Only FedAvg on all datasets
    python benchmarks/run_imaging_seeds5.py --algo FedAvg

    # Dry run to see what would be executed
    python benchmarks/run_imaging_seeds5.py --dry-run

Author: Fabio Liberti
"""

import sys
import os
import json
import time
import argparse
import traceback
from pathlib import Path
from datetime import datetime

# Add framework root to path
FRAMEWORK_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(FRAMEWORK_DIR))

import torch

# ======================================================================
# Configuration (matches run_paper_experiments.py exactly)
# ======================================================================

# Original 3 seeds
ORIGINAL_SEEDS = [42, 123, 456]

# New seeds to add (brings total to 5)
NEW_SEEDS = [789, 999]

# All 5 seeds (for reference / verification)
ALL_SEEDS = ORIGINAL_SEEDS + NEW_SEEDS

OUTPUT_DIR = FRAMEWORK_DIR / "benchmarks" / "paper_results"
CHECKPOINT_NAME = "p12_multidataset"

IMAGING_DATASETS = {
    "Brain_Tumor": {
        "data_dir": str(FRAMEWORK_DIR / "data" / "Brain_Tumor"),
        "num_classes": 4,
        "short": "BT",
    },
    "chest_xray": {
        "data_dir": str(FRAMEWORK_DIR / "data" / "chest_xray"),
        "num_classes": 2,
        "short": "CX",
    },
    "Skin_Cancer": {
        "data_dir": str(FRAMEWORK_DIR / "data" / "Skin Cancer"),
        "num_classes": 2,
        "short": "SC",
    },
}

ALGORITHMS = ["FedAvg", "FedLC", "FedSAM", "FedDecorr", "FedExP"]

# Imaging config (must match run_paper_experiments.py)
IMAGING_CONFIG = dict(
    num_clients=5, num_rounds=12, local_epochs=1,
    batch_size=32, learning_rate=0.0005,
    model_type="resnet18", is_iid=False, alpha=0.5,
    freeze_backbone=True,
)

DATASET_OVERRIDES = {
    "Brain_Tumor": {
        "learning_rate": 0.0003,
    },
}

EARLY_STOPPING_CONFIG = dict(
    enabled=True,
    patience=4,
    min_delta=0.005,
    min_rounds=6,
    metric="accuracy",
)


# ======================================================================
# Import training functions from main experiment script
# ======================================================================

from benchmarks.run_paper_experiments import (
    run_single_imaging,
    save_checkpoint,
    load_checkpoint,
    _cleanup_gpu,
)


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Add seeds 789, 999 to imaging experiments (3→5 seeds)"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be run without executing")
    parser.add_argument("--dataset", type=str, default=None,
                        choices=list(IMAGING_DATASETS.keys()),
                        help="Run only this dataset")
    parser.add_argument("--algo", type=str, default=None,
                        choices=ALGORITHMS,
                        help="Run only this algorithm")
    parser.add_argument("--resume", action="store_true", default=True,
                        help="Resume from checkpoint (default: True)")
    parser.add_argument("--seeds", nargs="+", type=int, default=NEW_SEEDS,
                        help=f"Seeds to run (default: {NEW_SEEDS})")
    args = parser.parse_args()

    print("=" * 70)
    print("ACTION #1: Imaging Seed Expansion (3 → 5 seeds)")
    print("=" * 70)
    print(f"  New seeds: {args.seeds}")
    print(f"  Datasets:  {args.dataset or 'ALL'}")
    print(f"  Algorithms: {args.algo or 'ALL'}")
    print(f"  Checkpoint: {CHECKPOINT_NAME}")
    print(f"  Started:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Load existing checkpoint
    results = load_checkpoint(CHECKPOINT_NAME)
    if results is None:
        print("WARNING: No existing checkpoint found. Creating new one.")
        results = {"completed": {}}

    existing = {k for k, v in results["completed"].items() if "error" not in v}
    print(f"  Existing valid experiments in checkpoint: {len(existing)}")

    # Determine what needs to run
    run_datasets = [args.dataset] if args.dataset else list(IMAGING_DATASETS.keys())
    run_algos = [args.algo] if args.algo else ALGORITHMS
    run_seeds = args.seeds

    # Build experiment queue
    queue = []
    already_done = []
    for ds_name in run_datasets:
        for algo in run_algos:
            for seed in run_seeds:
                key = f"{ds_name}_{algo}_{seed}"
                if key in existing:
                    already_done.append(key)
                else:
                    queue.append((ds_name, algo, seed, key))

    print(f"  Already completed: {len(already_done)}")
    print(f"  To run: {len(queue)}")
    print()

    if not queue:
        print("Nothing to run — all experiments already completed!")
        return

    # Show queue
    print("  Experiment queue:")
    for i, (ds, algo, seed, key) in enumerate(queue, 1):
        print(f"    {i:2d}. {ds:15s} × {algo:12s} × seed={seed}")
    print()

    if args.dry_run:
        est_hours = len(queue) * 15 / 60  # ~15 min per imaging experiment
        print(f"  DRY RUN: Would run {len(queue)} experiments")
        print(f"  Estimated time: ~{est_hours:.1f} hours")
        return

    # ---- Per-round checkpoint support ----
    trainer_ckpt_path = str(OUTPUT_DIR / ".training_state_seeds5.pt")

    def _make_round_callback(exp_key, ds_name, algo, seed):
        """Save checkpoint after EVERY training round (~1-2 min granularity)."""
        def on_round_complete(trainer, round_num, history, es_monitor):
            trainer.save_checkpoint(trainer_ckpt_path)
            results["in_progress"] = {
                "key": exp_key,
                "dataset": ds_name,
                "algorithm": algo,
                "seed": seed,
                "completed_rounds": round_num,
                "history_so_far": history,
                "trainer_checkpoint": trainer_ckpt_path,
                "early_stopping_state": es_monitor.get_state() if es_monitor else None,
            }
            save_checkpoint(CHECKPOINT_NAME, results)
        return on_round_complete

    # Check for in-progress experiment to resume
    in_prog = results.get("in_progress")
    resume_info = None
    if in_prog and Path(in_prog.get("trainer_checkpoint", "")).exists():
        print(f"  Resuming in-progress: {in_prog['key']} from round {in_prog['completed_rounds']}")
        resume_info = in_prog

    # Run experiments
    total = len(queue)
    t0 = time.time()
    completed = 0
    errors = 0

    for i, (ds_name, algo, seed, key) in enumerate(queue, 1):
        ds_info = IMAGING_DATASETS[ds_name]

        # Merge config with dataset-specific overrides
        img_config = {**IMAGING_CONFIG}
        if ds_name in DATASET_OVERRIDES:
            img_config.update(DATASET_OVERRIDES[ds_name])

        elapsed_total = time.time() - t0
        avg_per_exp = elapsed_total / max(completed, 1)
        remaining = avg_per_exp * (total - completed)

        print(f"  [{i}/{total}] {ds_name} / {algo} / seed={seed}", end=" ", flush=True)
        if completed > 0:
            print(f"(ETA: {remaining/60:.0f}min)", end=" ", flush=True)

        # Check if this is the in-progress experiment to resume
        this_resume = None
        if resume_info and resume_info.get("key") == key:
            this_resume = resume_info
            resume_info = None  # Consumed

        try:
            record = run_single_imaging(
                dataset_name=ds_name,
                data_dir=ds_info["data_dir"],
                algorithm=algo,
                seed=seed,
                mu=0.1,
                early_stopping=EARLY_STOPPING_CONFIG,
                use_amp=True,
                on_round_complete=_make_round_callback(key, ds_name, algo, seed),
                resume_from=this_resume,
                **img_config,
            )

            acc = record["final_metrics"]["accuracy"]
            t = record["runtime_seconds"]
            jain = record.get("fairness", {}).get("jain_index", 0)
            es_info = f" ES@R{record['actual_rounds']}" if record.get("stopped_early") else ""
            print(f"-> acc={acc:.3f} jain={jain:.3f} ({t:.0f}s{es_info})")

            results["completed"][key] = record
            completed += 1

        except Exception as e:
            print(f"-> ERROR: {e}")
            traceback.print_exc()
            results["completed"][key] = {
                "error": str(e),
                "traceback": traceback.format_exc(),
                "dataset": ds_name,
                "algorithm": algo,
                "seed": seed,
            }
            errors += 1

        # Experiment done: clear in_progress, save, cleanup
        results.pop("in_progress", None)
        save_checkpoint(CHECKPOINT_NAME, results)
        # Remove trainer checkpoint file (no longer needed)
        try:
            Path(trainer_ckpt_path).unlink(missing_ok=True)
        except OSError:
            pass
        _cleanup_gpu()

    # Summary
    total_time = time.time() - t0
    print()
    print("=" * 70)
    print("COMPLETED")
    print("=" * 70)
    print(f"  Experiments run:    {completed}")
    print(f"  Errors:             {errors}")
    print(f"  Total time:         {total_time/60:.1f} min ({total_time/3600:.1f} h)")
    if completed > 0:
        print(f"  Avg per experiment: {total_time/completed/60:.1f} min")
    print()

    # Verify seed coverage
    print("  Seed coverage per dataset×algorithm:")
    for ds in IMAGING_DATASETS:
        for algo in ALGORITHMS:
            seeds_done = []
            for s in ALL_SEEDS:
                k = f"{ds}_{algo}_{s}"
                if k in results["completed"] and "error" not in results["completed"][k]:
                    seeds_done.append(s)
            status = "5/5" if len(seeds_done) == 5 else f"{len(seeds_done)}/5"
            marker = " OK" if len(seeds_done) >= 5 else " INCOMPLETE"
            print(f"    {ds:15s} × {algo:12s}: {status}{marker} {seeds_done}")

    print()
    print(f"  Checkpoint: {OUTPUT_DIR / f'checkpoint_{CHECKPOINT_NAME}.json'}")
    print(f"  Finished:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()

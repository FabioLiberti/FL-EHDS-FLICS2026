#!/usr/bin/env python3
"""
FL-EHDS Experiment — DP Gradient Clipping Sensitivity.

Tests how the gradient clipping norm C affects accuracy under fixed DP (eps=10)
across two datasets (Cardiovascular, PTB-XL) and three algorithms.

Design:
  - Datasets: Cardiovascular (input_dim=11, num_classes=2),
              PTB-XL (input_dim=9, num_classes=5)
  - Algorithms: FedAvg, Ditto, HPFL
  - Clip norms: C = {0.5, 1.0, 2.0, 5.0}
  - DP: eps=10 (fixed, central mode)
  - Seeds: 42, 123, 456
  - Total: 2 datasets x 3 algos x 4 C x 3 seeds = 72 experiments (~1h on M3)

Usage:
    cd fl-ehds-framework
    python -m benchmarks.run_dp_clipping [--quick] [--fresh]

Output:
    benchmarks/paper_results_tabular/checkpoint_dp_clipping.json
    benchmarks/paper_results_tabular/dp_clipping_sensitivity.png

Author: Fabio Liberti
"""

import sys
import os
import json
import time
import shutil
import signal
import tempfile
import argparse
import traceback
import gc
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Any
from collections import defaultdict

import numpy as np

FRAMEWORK_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(FRAMEWORK_DIR))

import torch

from terminal.fl_trainer import FederatedTrainer, _detect_device
from data.cardiovascular_loader import load_cardiovascular_data
from data.ptbxl_loader import load_ptbxl_data

# ======================================================================
# Constants
# ======================================================================

OUTPUT_DIR = FRAMEWORK_DIR / "benchmarks" / "paper_results_tabular"
CHECKPOINT_FILE = "checkpoint_dp_clipping.json"
LOG_FILE = "experiment_dp_clipping.log"

ALGORITHMS = ["FedAvg", "Ditto", "HPFL"]
CLIP_NORMS = [0.5, 1.0, 2.0, 5.0]
DP_EPSILON = 10.0
SEEDS = [42, 123, 456]

DATASET_CONFIGS = {
    "CV": {
        "name": "Cardiovascular",
        "loader": "cardiovascular",
        "input_dim": 11,
        "num_classes": 2,
        "num_clients": 5,
        "learning_rate": 0.01,
        "batch_size": 64,
        "num_rounds": 30,
        "local_epochs": 3,
        "mu": 0.1,
    },
    "PX": {
        "name": "PTB-XL",
        "loader": "ptbxl",
        "input_dim": 9,
        "num_classes": 5,
        "num_clients": 5,
        "learning_rate": 0.005,
        "batch_size": 64,
        "num_rounds": 30,
        "local_epochs": 3,
        "mu": 0.1,
    },
}

# ======================================================================
# Logging
# ======================================================================

_log_file = None


def log(msg: str, also_print: bool = True):
    ts = datetime.now().strftime("%H:%M:%S")
    line = "[{}] {}".format(ts, msg)
    if also_print:
        print(line, flush=True)
    if _log_file:
        try:
            _log_file.write(line + "\n")
            _log_file.flush()
        except Exception:
            pass


# ======================================================================
# Checkpoint (atomic write with backup)
# ======================================================================

def save_checkpoint(data: Dict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / CHECKPOINT_FILE
    bak = OUTPUT_DIR / (CHECKPOINT_FILE + ".bak")
    data["metadata"]["last_save"] = datetime.now().isoformat()
    fd, tmp = tempfile.mkstemp(dir=str(OUTPUT_DIR), prefix=".dpclip_", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2, default=str)
            f.flush()
            os.fsync(f.fileno())
        if path.exists():
            shutil.copy2(str(path), str(bak))
        os.replace(tmp, str(path))
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def load_checkpoint() -> Optional[Dict]:
    path = OUTPUT_DIR / CHECKPOINT_FILE
    bak = OUTPUT_DIR / (CHECKPOINT_FILE + ".bak")
    for p in [path, bak]:
        if p.exists():
            try:
                with open(p) as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                continue
    return None


# ======================================================================
# GPU cleanup
# ======================================================================

def _cleanup_gpu():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        try:
            torch.mps.empty_cache()
        except Exception:
            pass
    gc.collect()


# ======================================================================
# Data loading
# ======================================================================

def load_dataset(ds_key: str, seed: int):
    cfg = DATASET_CONFIGS[ds_key]
    if cfg["loader"] == "cardiovascular":
        return load_cardiovascular_data(
            num_clients=cfg["num_clients"], seed=seed, is_iid=False,
        )
    elif cfg["loader"] == "ptbxl":
        return load_ptbxl_data(
            num_clients=cfg["num_clients"], seed=seed,
            partition_by_site=True, min_site_samples=50,
        )
    else:
        raise ValueError("Unknown loader: {}".format(cfg["loader"]))


# ======================================================================
# Per-client evaluation
# ======================================================================

def _evaluate_per_client(trainer) -> Dict[str, float]:
    model = trainer.global_model
    model.eval()
    per_client = {}

    is_hpfl = trainer.algorithm == "HPFL"
    if is_hpfl:
        saved_cls = {n: p.data.clone() for n, p in model.named_parameters()
                     if n in trainer._hpfl_classifier_names}

    with torch.no_grad():
        for cid in range(trainer.num_clients):
            if is_hpfl:
                for n, p in model.named_parameters():
                    if n in trainer._hpfl_classifier_names:
                        p.data.copy_(trainer.client_classifiers[cid][n])
            X, y = trainer.client_test_data[cid]
            X_t = torch.FloatTensor(X).to(trainer.device) if isinstance(X, np.ndarray) else X.to(trainer.device)
            y_t = torch.LongTensor(y).to(trainer.device) if isinstance(y, np.ndarray) else y.to(trainer.device)
            correct = total = 0
            for i in range(0, len(y_t), 64):
                out = model(X_t[i:i+64])
                correct += (out.argmax(1) == y_t[i:i+64]).sum().item()
                total += len(y_t[i:i+64])
            per_client[str(cid)] = correct / total if total > 0 else 0.0

    if is_hpfl:
        for n, p in model.named_parameters():
            if n in trainer._hpfl_classifier_names:
                p.data.copy_(saved_cls[n])
    return per_client


# ======================================================================
# Single experiment
# ======================================================================

def run_single_experiment(ds_key, algo, clip_value, seed, quick=False):
    """Run one experiment: dataset + algorithm + clip norm + seed."""
    cfg = DATASET_CONFIGS[ds_key]
    start = time.time()
    num_rounds = 5 if quick else cfg["num_rounds"]

    client_data, client_test, metadata = load_dataset(ds_key, seed)

    trainer = FederatedTrainer(
        num_clients=cfg["num_clients"],
        algorithm=algo,
        local_epochs=cfg["local_epochs"],
        batch_size=cfg["batch_size"],
        learning_rate=cfg["learning_rate"],
        mu=cfg["mu"],
        dp_enabled=True,
        dp_epsilon=DP_EPSILON,
        dp_clip_norm=clip_value,
        seed=seed,
        external_data=client_data,
        external_test_data=client_test,
        input_dim=cfg["input_dim"],
        num_classes=cfg["num_classes"],
    )

    history = []
    best_acc = 0.0

    for r in range(num_rounds):
        result = trainer.train_round(r)
        metrics = {
            "round": r + 1,
            "accuracy": result.global_acc,
            "loss": result.global_loss,
            "f1": result.global_f1,
        }
        history.append(metrics)
        if result.global_acc > best_acc:
            best_acc = result.global_acc

    per_client_acc = _evaluate_per_client(trainer)
    elapsed = time.time() - start

    out = {
        "dataset": ds_key,
        "algorithm": algo,
        "clip_norm": clip_value,
        "dp_epsilon": DP_EPSILON,
        "seed": seed,
        "history": history,
        "final_metrics": history[-1] if history else {},
        "per_client_acc": per_client_acc,
        "best_accuracy": best_acc,
        "actual_rounds": len(history),
        "runtime_seconds": round(elapsed, 1),
    }

    _cleanup_gpu()
    return out


# ======================================================================
# Figure generation
# ======================================================================

def generate_figure(completed, ds_keys, algos, clip_norms, seeds):
    """Line plot: accuracy vs clip norm, one line per algorithm, one subplot per dataset."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log("matplotlib not available -- skipping figure")
        return

    colors = {"FedAvg": "#1f77b4", "Ditto": "#ff7f0e", "HPFL": "#2ca02c"}
    markers = {"FedAvg": "o", "Ditto": "s", "HPFL": "^"}

    fig, axes = plt.subplots(1, len(ds_keys), figsize=(6 * len(ds_keys), 5), squeeze=False)

    for col, ds_key in enumerate(ds_keys):
        ax = axes[0][col]
        ds_name = DATASET_CONFIGS[ds_key]["name"]

        for algo in algos:
            means = []
            stds = []
            valid_clips = []
            for c in clip_norms:
                accs = []
                for s in seeds:
                    key = "{}_{}_{}_s{}".format(ds_key, algo, _clip_key(c), s)
                    if key in completed and "error" not in completed[key]:
                        accs.append(completed[key]["best_accuracy"] * 100)
                if accs:
                    means.append(np.mean(accs))
                    stds.append(np.std(accs))
                    valid_clips.append(c)

            if means:
                ax.errorbar(
                    valid_clips, means, yerr=stds,
                    label=algo, color=colors.get(algo, "gray"),
                    marker=markers.get(algo, "o"),
                    capsize=3, linewidth=2, markersize=7,
                )

        ax.set_xlabel("Clip Norm C", fontsize=12)
        ax.set_ylabel("Accuracy (%)", fontsize=12)
        ax.set_title("{} (eps={})".format(ds_name, int(DP_EPSILON)),
                      fontsize=13, fontweight="bold")
        ax.set_xticks(clip_norms)
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)

    fig.suptitle("DP Gradient Clipping Sensitivity", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    fig_path = OUTPUT_DIR / "dp_clipping_sensitivity.png"
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.savefig(str(fig_path).replace(".png", ".pdf"), bbox_inches="tight")
    log("Figure saved: {}".format(fig_path))
    plt.close()


# ======================================================================
# Helpers
# ======================================================================

def _clip_key(c):
    """Format clip norm for experiment key: 0.5 -> 'C0.5'."""
    return "C{}".format(c)


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="FL-EHDS DP Gradient Clipping Sensitivity")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 1 seed, 2 clip values, 1 algo")
    parser.add_argument("--fresh", action="store_true",
                        help="Delete existing checkpoint and start fresh")
    args = parser.parse_args()

    global _log_file
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _log_file = open(OUTPUT_DIR / LOG_FILE, "a")

    # Experiment grid
    seeds = [42] if args.quick else SEEDS
    clip_norms = [0.5, 2.0] if args.quick else CLIP_NORMS
    algos = ["FedAvg"] if args.quick else ALGORITHMS
    ds_keys = list(DATASET_CONFIGS.keys())

    # Build experiment list
    experiments = []
    for ds_key in ds_keys:
        for algo in algos:
            for c in clip_norms:
                for seed in seeds:
                    key = "{}_{}_{}_s{}".format(ds_key, algo, _clip_key(c), seed)
                    experiments.append({
                        "key": key, "ds_key": ds_key, "algorithm": algo,
                        "clip_norm": c, "seed": seed,
                    })

    total_exps = len(experiments)

    # Handle --fresh
    if args.fresh:
        ckpt_path = OUTPUT_DIR / CHECKPOINT_FILE
        if ckpt_path.exists():
            ckpt_path.unlink()
            log("Deleted existing checkpoint")

    # Auto-resume
    checkpoint_data = None
    if not args.fresh:
        checkpoint_data = load_checkpoint()
        if checkpoint_data:
            done = len(checkpoint_data.get("completed", {}))
            log("AUTO-RESUMED: {}/{} completed".format(done, total_exps))

    if checkpoint_data is None:
        checkpoint_data = {
            "completed": {},
            "metadata": {
                "total_experiments": total_exps,
                "algorithms": algos,
                "datasets": ds_keys,
                "clip_norms": clip_norms,
                "dp_epsilon": DP_EPSILON,
                "seeds": seeds,
                "start_time": datetime.now().isoformat(),
                "last_save": None,
            },
        }

    # Signal handler for graceful shutdown
    _interrupted = [False]

    def _signal_handler(signum, frame):
        if _interrupted[0]:
            sys.exit(1)
        _interrupted[0] = True
        done = len(checkpoint_data.get("completed", {}))
        log("\nINTERRUPT -- saving checkpoint ({}/{})...".format(done, total_exps))
        save_checkpoint(checkpoint_data)
        log("Checkpoint saved. Resume: python -m benchmarks.run_dp_clipping")
        sys.exit(0)

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # Header
    mode = "QUICK" if args.quick else "FULL"
    log("\n" + "=" * 66)
    log("  FL-EHDS DP Gradient Clipping Sensitivity ({})".format(mode))
    log("  {} experiments = {} DS x {} algos x {} C x {} seeds".format(
        total_exps, len(ds_keys), len(algos), len(clip_norms), len(seeds)))
    log("=" * 66)
    log("  Device:     {}".format(_detect_device()))
    log("  Datasets:   {}".format(ds_keys))
    log("  Algorithms: {}".format(algos))
    log("  Clip norms: {}".format(clip_norms))
    log("  DP epsilon: {}".format(DP_EPSILON))
    log("  Seeds:      {}".format(seeds))
    log("  Output:     {}".format(OUTPUT_DIR / CHECKPOINT_FILE))
    log("=" * 66)

    # Run experiments
    global_start = time.time()
    completed = checkpoint_data.get("completed", {})
    done_count = len(completed)

    for idx, exp in enumerate(experiments, 1):
        key = exp["key"]
        if key in completed:
            continue

        if _interrupted[0]:
            break

        ds_key = exp["ds_key"]
        algo = exp["algorithm"]
        c = exp["clip_norm"]
        seed = exp["seed"]

        log("[{}/{}] {} {} C={} s{} ...".format(
            done_count + 1, total_exps, ds_key, algo, c, seed))

        try:
            result = run_single_experiment(
                ds_key, algo, c, seed, quick=args.quick)
            completed[key] = result
            done_count += 1

            acc = result["best_accuracy"] * 100
            rt = result["runtime_seconds"]
            log("  -> acc={:.1f}% {:.0f}s".format(acc, rt))

            # Save checkpoint after every experiment
            save_checkpoint(checkpoint_data)

        except Exception as e:
            log("  ERROR: {}".format(e))
            traceback.print_exc()
            completed[key] = {
                "dataset": ds_key, "algorithm": algo,
                "clip_norm": c, "seed": seed, "error": str(e),
            }
            save_checkpoint(checkpoint_data)
            _cleanup_gpu()

    # Finalize
    checkpoint_data["metadata"]["end_time"] = datetime.now().isoformat()
    checkpoint_data["metadata"]["total_elapsed"] = time.time() - global_start
    save_checkpoint(checkpoint_data)

    elapsed = time.time() - global_start
    log("\n" + "=" * 66)
    log("  COMPLETED: {}/{}".format(done_count, total_exps))
    log("  Total time: {}".format(timedelta(seconds=int(elapsed))))
    log("=" * 66)

    # ======================================================================
    # Summary table
    # ======================================================================
    log("\n  DP Clipping Sensitivity Results (eps={})".format(int(DP_EPSILON)))
    log("  {:>4s} | {:>8s} | {}".format(
        "DS", "Algo",
        " | ".join("C={:<4}".format(c) for c in clip_norms)))
    log("  " + "-" * (18 + 10 * len(clip_norms)))

    for ds_key in ds_keys:
        for algo in algos:
            vals = []
            for c in clip_norms:
                accs = []
                for s in seeds:
                    k = "{}_{}_{}_s{}".format(ds_key, algo, _clip_key(c), s)
                    if k in completed and "error" not in completed[k]:
                        accs.append(completed[k]["best_accuracy"] * 100)
                if accs:
                    vals.append("{:5.1f}+-{:.1f}".format(
                        np.mean(accs), np.std(accs)))
                else:
                    vals.append("   --   ")
            log("  {:>4s} | {:>8s} | {}".format(
                ds_key, algo, " | ".join(vals)))
        log("  " + "-" * (18 + 10 * len(clip_norms)))

    # ======================================================================
    # Generate figure
    # ======================================================================
    generate_figure(completed, ds_keys, algos, clip_norms, seeds)

    if _log_file:
        _log_file.close()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
FL-EHDS Local Epochs Sweep — Algorithm Collapse Investigation.

Tests whether increasing local epochs (E ∈ {1, 5, 10, 20}) breaks the
algorithm collapse on Cardiovascular dataset (70K samples, 5 clients).

Hypothesis: With E=1-3, client models don't diverge enough from the global
model, so all server-side strategies find the same basin. With E=10-20,
client drift becomes significant and variance-reduction (SCAFFOLD-like
corrections in FedProx), adaptive strategies (FedExP, FedLESAM), and
logit calibration (FedLC) should differentiate from FedAvg.

Experiments: 7 algorithms × 4 epoch values × 5 seeds = 140 experiments
Estimated time: ~1.5-2 hours on MPS/CPU

Usage:
    cd fl-ehds-framework
    python -m benchmarks.run_tabular_epochs_sweep [--quick] [--resume] [--fresh]

Output: benchmarks/paper_results_tabular/checkpoint_epochs_sweep.json

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
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from collections import defaultdict

import numpy as np

FRAMEWORK_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(FRAMEWORK_DIR))

import torch
import torch.nn as nn

from terminal.fl_trainer import (
    FederatedTrainer,
    RoundResult,
    HealthcareMLP,
    _detect_device,
)
from data.cardiovascular_loader import load_cardiovascular_data

# ======================================================================
# Constants
# ======================================================================

OUTPUT_DIR = FRAMEWORK_DIR / "benchmarks" / "paper_results_tabular"
CHECKPOINT_FILE = "checkpoint_epochs_sweep.json"
LOG_FILE = "experiment_epochs_sweep.log"

ALGORITHMS = ["FedAvg", "FedProx", "Ditto", "FedLC", "FedExP", "FedLESAM", "HPFL"]
SEEDS = [42, 123, 456, 789, 999]
LOCAL_EPOCHS = [1, 5, 10, 20]

# Cardiovascular baseline config (same as sweep)
CV_CONFIG = dict(
    input_dim=11, num_classes=2,
    learning_rate=0.01, batch_size=64, num_rounds=25,
    mu=0.1,
    es=dict(enabled=True, patience=6, min_delta=0.003, min_rounds=10),
)
NUM_CLIENTS = 5
PARTITION = {"is_iid": False, "alpha": 0.5}


# ======================================================================
# Logging
# ======================================================================

_log_file = None


def log(msg: str, also_print: bool = True):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    if also_print:
        print(line, flush=True)
    if _log_file:
        try:
            _log_file.write(line + "\n")
            _log_file.flush()
        except Exception:
            pass


# ======================================================================
# Checkpoint
# ======================================================================

def save_checkpoint(data: Dict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / CHECKPOINT_FILE
    bak = OUTPUT_DIR / (CHECKPOINT_FILE + ".bak")
    data["metadata"]["last_save"] = datetime.now().isoformat()

    fd, tmp = tempfile.mkstemp(dir=str(OUTPUT_DIR), prefix=".epochs_", suffix=".tmp")
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
    import gc
    gc.collect()


# ======================================================================
# Early Stopping
# ======================================================================

class EarlyStoppingMonitor:
    def __init__(self, patience=6, min_delta=0.003, min_rounds=10):
        self.patience = patience
        self.min_delta = min_delta
        self.min_rounds = min_rounds
        self.best_value = -float('inf')
        self.counter = 0

    def check(self, round_num, metrics):
        value = metrics.get("accuracy", 0)
        if value > self.best_value + self.min_delta:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
        return round_num >= self.min_rounds and self.counter >= self.patience


# ======================================================================
# Per-client evaluation & fairness
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


def _compute_fairness(per_client_acc: Dict[str, float]) -> Dict[str, float]:
    accs = list(per_client_acc.values())
    if not accs:
        return {}
    jain = (sum(accs) ** 2) / (len(accs) * sum(a ** 2 for a in accs)) if accs else 0
    sorted_a = sorted(accs)
    n = len(sorted_a)
    cumsum = np.cumsum(sorted_a)
    gini = (2 * sum((i + 1) * v for i, v in enumerate(sorted_a))) / (n * cumsum[-1]) - (n + 1) / n
    gini = max(0, gini)
    return {
        "mean": round(float(np.mean(accs)), 4),
        "std": round(float(np.std(accs)), 4),
        "min": round(float(min(accs)), 4),
        "max": round(float(max(accs)), 4),
        "jain_index": round(float(jain), 4),
        "gini": round(float(gini), 4),
    }


# ======================================================================
# Build experiments
# ======================================================================

def build_experiments(seeds, quick=False):
    """Build local epochs sweep experiments for Cardiovascular."""
    exps = []
    epochs_list = [1, 10] if quick else LOCAL_EPOCHS
    algos = ALGORITHMS[:3] if quick else ALGORITHMS

    for E in epochs_list:
        for algo in algos:
            for seed in seeds:
                key = f"CV_{algo}_E{E}_s{seed}"
                exps.append({
                    "key": key,
                    "algorithm": algo,
                    "seed": seed,
                    "local_epochs": E,
                })
    return exps


# ======================================================================
# Training
# ======================================================================

def run_single_experiment(exp: Dict, quick: bool = False) -> Dict[str, Any]:
    """Run a single epochs sweep experiment on Cardiovascular."""
    start = time.time()
    algo = exp["algorithm"]
    seed = exp["seed"]
    E = exp["local_epochs"]

    num_rounds = 5 if quick else CV_CONFIG["num_rounds"]

    client_data, client_test_data, metadata = load_cardiovascular_data(
        num_clients=NUM_CLIENTS, seed=seed,
        is_iid=False, alpha=0.5,
    )

    trainer = FederatedTrainer(
        num_clients=NUM_CLIENTS,
        algorithm=algo,
        local_epochs=E,
        batch_size=CV_CONFIG["batch_size"],
        learning_rate=CV_CONFIG["learning_rate"],
        mu=CV_CONFIG["mu"],
        seed=seed,
        external_data=client_data,
        external_test_data=client_test_data,
        input_dim=CV_CONFIG["input_dim"],
        num_classes=CV_CONFIG["num_classes"],
    )

    es_cfg = CV_CONFIG["es"]
    es = EarlyStoppingMonitor(
        patience=es_cfg["patience"], min_delta=es_cfg["min_delta"],
        min_rounds=es_cfg["min_rounds"],
    ) if es_cfg.get("enabled") and not quick else None

    history = []
    best_acc = 0.0
    best_round = 0

    for r in range(num_rounds):
        rr = trainer.train_round(r)
        metrics = {
            "round": r + 1, "accuracy": rr.global_acc, "loss": rr.global_loss,
            "f1": rr.global_f1, "precision": rr.global_precision,
            "recall": rr.global_recall, "auc": rr.global_auc,
        }
        history.append(metrics)
        if rr.global_acc > best_acc:
            best_acc = rr.global_acc
            best_round = r + 1
        if es and es.check(r + 1, {"accuracy": rr.global_acc}):
            break

    per_client_acc = _evaluate_per_client(trainer)
    fairness = _compute_fairness(per_client_acc)
    elapsed = time.time() - start

    result = {
        "dataset": "Cardiovascular",
        "algorithm": algo,
        "seed": seed,
        "local_epochs": E,
        "history": history,
        "final_metrics": history[-1] if history else {},
        "per_client_acc": per_client_acc,
        "fairness": fairness,
        "runtime_seconds": round(elapsed, 1),
        "stopped_early": es is not None and es.counter >= es.patience,
        "actual_rounds": len(history),
        "best_metrics": {"accuracy": best_acc, "round": best_round},
    }

    _cleanup_gpu()
    return result


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="FL-EHDS Local Epochs Sweep")
    parser.add_argument("--quick", action="store_true", help="Quick validation mode")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--fresh", action="store_true", help="Start fresh (delete old checkpoint)")
    args = parser.parse_args()

    global _log_file
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _log_file = open(OUTPUT_DIR / LOG_FILE, "a")

    seeds = [42] if args.quick else SEEDS
    experiments = build_experiments(seeds, args.quick)
    total_exps = len(experiments)

    # Handle fresh start
    if args.fresh:
        ckpt_path = OUTPUT_DIR / CHECKPOINT_FILE
        if ckpt_path.exists():
            ckpt_path.unlink()
            log("Deleted old checkpoint (--fresh)")

    # Load or create checkpoint
    checkpoint_data = None
    if args.resume:
        checkpoint_data = load_checkpoint()
        if checkpoint_data:
            done = len(checkpoint_data.get("completed", {}))
            log(f"RESUMED: {done} experiments already completed")

    if checkpoint_data is None:
        checkpoint_data = {
            "completed": {},
            "metadata": {
                "total_experiments": total_exps,
                "algorithms": ALGORITHMS if not args.quick else ALGORITHMS[:3],
                "seeds": seeds,
                "local_epochs": LOCAL_EPOCHS if not args.quick else [1, 10],
                "dataset": "Cardiovascular",
                "start_time": datetime.now().isoformat(),
                "last_save": None,
                "version": "epochs_sweep_v1",
            }
        }

    # SIGINT handler
    _interrupted = [False]

    def _signal_handler(signum, frame):
        if _interrupted[0]:
            sys.exit(1)
        _interrupted[0] = True
        done = len(checkpoint_data.get("completed", {}))
        save_checkpoint(checkpoint_data)
        print(f"\n  Checkpoint salvato: {done}/{total_exps}")
        print(f"  Per riprendere: python -m benchmarks.run_tabular_epochs_sweep --resume")
        sys.exit(0)

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # Header
    mode = "QUICK" if args.quick else "FULL"
    log(f"\n{'='*66}")
    log(f"  FL-EHDS Local Epochs Sweep ({mode})")
    log(f"  Dataset: Cardiovascular (70K, {NUM_CLIENTS} clients, α=0.5)")
    log(f"  Algorithms: {len(ALGORITHMS if not args.quick else ALGORITHMS[:3])}")
    log(f"  Local epochs: {LOCAL_EPOCHS if not args.quick else [1, 10]}")
    log(f"  Seeds: {seeds}")
    log(f"  Total experiments: {total_exps}")
    log(f"  Device: {_detect_device()}")
    log(f"{'='*66}")

    start_time = time.time()
    completed = checkpoint_data.get("completed", {})
    done_count = len(completed)
    errors = []

    for i, exp in enumerate(experiments):
        key = exp["key"]
        if key in completed:
            continue

        algo = exp["algorithm"]
        E = exp["local_epochs"]
        seed = exp["seed"]

        log(f"  [{done_count+1}/{total_exps}] {algo} E={E} s={seed} ...", also_print=True)

        try:
            result = run_single_experiment(exp, quick=args.quick)
            completed[key] = result
            done_count += 1

            acc = result["final_metrics"].get("accuracy", 0) * 100
            rounds = result["actual_rounds"]
            rt = result["runtime_seconds"]
            log(f"    -> {acc:.1f}% ({rounds}r, {rt:.0f}s)")

            if done_count % 5 == 0:
                save_checkpoint(checkpoint_data)

        except Exception as e:
            log(f"    ERROR: {e}")
            errors.append({"key": key, "error": str(e)})
            traceback.print_exc()
            _cleanup_gpu()

    # Final save
    checkpoint_data["metadata"]["end_time"] = datetime.now().isoformat()
    checkpoint_data["metadata"]["errors"] = errors
    save_checkpoint(checkpoint_data)

    elapsed = time.time() - start_time

    # Print summary
    log(f"\n{'='*66}")
    log(f"  COMPLETED: {done_count}/{total_exps}")
    log(f"  Total time: {timedelta(seconds=int(elapsed))}")
    log(f"  Checkpoint: {OUTPUT_DIR / CHECKPOINT_FILE}")
    log(f"{'='*66}")

    # Print results table
    log(f"\n  Local Epochs Sweep Results (Cardiovascular):")
    log(f"  {'E':>4s} | {'Algorithm':>10s} | {'Acc (%)':>10s} | {'F1':>6s} | {'Jain':>5s}")
    log(f"  {'----':>4s}-+-{'-'*10}-+-{'-'*10}-+-{'-'*6}-+-{'-'*5}")

    for E in (LOCAL_EPOCHS if not args.quick else [1, 10]):
        algos = ALGORITHMS if not args.quick else ALGORITHMS[:3]
        for algo in algos:
            accs = []
            f1s = []
            jains = []
            for seed in seeds:
                key = f"CV_{algo}_E{E}_s{seed}"
                if key in completed:
                    r = completed[key]
                    accs.append(r["final_metrics"].get("accuracy", 0) * 100)
                    f1s.append(r["final_metrics"].get("f1", 0))
                    jains.append(r["fairness"].get("jain_index", 0))

            if accs:
                acc_mean = np.mean(accs)
                acc_std = np.std(accs)
                f1_mean = np.mean(f1s)
                jain_mean = np.mean(jains)
                log(f"  {E:4d} | {algo:>10s} | {acc_mean:5.1f}±{acc_std:4.1f} | {f1_mean:.3f} | {jain_mean:.3f}")

    # Collapse analysis
    log(f"\n  Algorithm Collapse Analysis:")
    log(f"  {'E':>4s} | {'FedAvg':>8s} | {'FedProx':>8s} | {'FedLC':>8s} | {'FedExP':>8s} | {'FedLESAM':>8s} | {'Ditto':>8s} | {'HPFL':>8s} | {'Spread':>8s}")
    log(f"  {'----':>4s}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")

    algos_for_analysis = ALGORITHMS if not args.quick else ALGORITHMS[:3]
    for E in (LOCAL_EPOCHS if not args.quick else [1, 10]):
        row = f"  {E:4d} |"
        means = {}
        for algo in algos_for_analysis:
            accs = []
            for seed in seeds:
                key = f"CV_{algo}_E{E}_s{seed}"
                if key in completed:
                    accs.append(completed[key]["final_metrics"].get("accuracy", 0) * 100)
            if accs:
                m = np.mean(accs)
                means[algo] = m
                row += f" {m:7.1f}% |"
            else:
                row += f" {'---':>7s} |"

        if means:
            spread = max(means.values()) - min(means.values())
            row += f" {spread:6.1f}pp"
        log(row)

    if errors:
        log(f"\n  ERRORS ({len(errors)}):")
        for e in errors:
            log(f"    {e['key']}: {e['error']}")


if __name__ == "__main__":
    main()

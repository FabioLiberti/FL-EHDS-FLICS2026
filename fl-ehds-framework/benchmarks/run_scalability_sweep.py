#!/usr/bin/env python3
"""
FL-EHDS Scalability Sweep — Large-Scale Client Analysis.

Tests whether FL-EHDS algorithms maintain performance with realistic EHDS
client counts (K=50-100 hospitals across 27 EU member states).

Experiments:
  - Cardiovascular: K ∈ {50, 100} × 7 algorithms × 3 seeds = 42 experiments
  - PTB-XL:         K=30 (Dirichlet) + K=52 (site-based) × 7 algos × 3 seeds = 42 experiments
  Total: 84 experiments

Hypothesis: With K=50-100, per-client data shrinks significantly. Algorithms
that excel with K=5-10 may degrade. Personalization methods (Ditto, HPFL)
should be more robust because they adapt to local distributions.

Usage:
    cd fl-ehds-framework
    python -m benchmarks.run_scalability_sweep [--quick] [--resume] [--fresh]

Output: benchmarks/paper_results_tabular/checkpoint_scalability.json

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
from typing import Dict, List, Optional, Any, Tuple
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
from data.ptbxl_loader import load_ptbxl_data

# ======================================================================
# Constants
# ======================================================================

OUTPUT_DIR = FRAMEWORK_DIR / "benchmarks" / "paper_results_tabular"
CHECKPOINT_FILE = "checkpoint_scalability.json"
LOG_FILE = "experiment_scalability.log"

ALGORITHMS = ["FedAvg", "FedProx", "Ditto", "FedLC", "FedExP", "FedLESAM", "HPFL"]
SEEDS = [42, 123, 456]  # 3 seeds for speed

# Cardiovascular: K ∈ {50, 100}, Dirichlet α=0.5
CV_CONFIG = dict(
    input_dim=11, num_classes=2,
    learning_rate=0.01, batch_size=64, num_rounds=25, local_epochs=3,
    mu=0.1,
    es=dict(enabled=True, patience=6, min_delta=0.003, min_rounds=10),
)
CV_CLIENTS = [50, 100]

# PTB-XL: K=30 (Dirichlet) + K=52 (site-based, all 52 recording sites)
PX_CONFIG = dict(
    input_dim=9, num_classes=5,
    learning_rate=0.005, batch_size=64, num_rounds=30, local_epochs=3,
    mu=0.1,
    es=dict(enabled=True, patience=6, min_delta=0.003, min_rounds=10),
)
PX_EXPERIMENTS = [
    {"num_clients": 30, "partition": {"is_iid": False, "alpha": 0.5}, "label": "K30_dir"},
    {"num_clients": 52, "partition": {"partition_by_site": True}, "label": "K52_site"},
]


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

    fd, tmp = tempfile.mkstemp(dir=str(OUTPUT_DIR), prefix=".scal_", suffix=".tmp")
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
    """Build scalability experiments for CV and PTB-XL."""
    exps = []
    algos = ALGORITHMS[:3] if quick else ALGORITHMS

    # Cardiovascular: K ∈ {50, 100}
    cv_clients = [50] if quick else CV_CLIENTS
    for K in cv_clients:
        for algo in algos:
            for seed in seeds:
                key = "CV_{}_K{}_s{}".format(algo, K, seed)
                exps.append({
                    "key": key,
                    "dataset": "Cardiovascular",
                    "algorithm": algo,
                    "seed": seed,
                    "num_clients": K,
                    "partition": {"is_iid": False, "alpha": 0.5},
                    "config": CV_CONFIG,
                    "label": "K{}".format(K),
                })

    # PTB-XL: K=30 Dirichlet + K=52 site-based
    px_exps = PX_EXPERIMENTS[:1] if quick else PX_EXPERIMENTS
    for px_cfg in px_exps:
        K = px_cfg["num_clients"]
        label = px_cfg["label"]
        for algo in algos:
            for seed in seeds:
                key = "PX_{}_{}_{}_s{}".format(algo, label, K, seed)
                exps.append({
                    "key": key,
                    "dataset": "PTB_XL",
                    "algorithm": algo,
                    "seed": seed,
                    "num_clients": K,
                    "partition": px_cfg["partition"],
                    "config": PX_CONFIG,
                    "label": label,
                })

    return exps


# ======================================================================
# Data loading
# ======================================================================

def load_dataset(ds_name: str, num_clients: int, seed: int, partition: Dict):
    """Load dataset with specified partition strategy."""
    if ds_name == "PTB_XL":
        if partition.get("partition_by_site"):
            return load_ptbxl_data(
                num_clients=num_clients, seed=seed,
                partition_by_site=True, min_site_samples=30,
            )
        else:
            return load_ptbxl_data(
                num_clients=num_clients, seed=seed,
                partition_by_site=False, is_iid=partition.get("is_iid", False),
                alpha=partition.get("alpha", 0.5),
            )
    elif ds_name == "Cardiovascular":
        return load_cardiovascular_data(
            num_clients=num_clients, seed=seed,
            is_iid=partition.get("is_iid", False),
            alpha=partition.get("alpha", 0.5),
        )
    else:
        raise ValueError("Unknown dataset: {}".format(ds_name))


# ======================================================================
# Training
# ======================================================================

def run_single_experiment(exp: Dict, quick: bool = False) -> Dict[str, Any]:
    """Run a single scalability experiment."""
    start = time.time()
    algo = exp["algorithm"]
    seed = exp["seed"]
    K = exp["num_clients"]
    cfg = exp["config"]
    ds_name = exp["dataset"]

    num_rounds = 5 if quick else cfg["num_rounds"]

    # Load data
    client_data, client_test_data, metadata = load_dataset(
        ds_name=ds_name, num_clients=K, seed=seed,
        partition=exp["partition"],
    )

    actual_K = len(client_data)  # May differ from K for site-based
    samples_per_client = metadata.get("samples_per_client", {})
    min_samples = min(samples_per_client.values()) if samples_per_client else 0
    max_samples = max(samples_per_client.values()) if samples_per_client else 0

    trainer = FederatedTrainer(
        num_clients=actual_K,
        algorithm=algo,
        local_epochs=cfg["local_epochs"],
        batch_size=cfg["batch_size"],
        learning_rate=cfg["learning_rate"],
        mu=cfg["mu"],
        seed=seed,
        external_data=client_data,
        external_test_data=client_test_data,
        input_dim=cfg["input_dim"],
        num_classes=cfg["num_classes"],
    )

    es_cfg = cfg["es"]
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
        "dataset": ds_name,
        "algorithm": algo,
        "seed": seed,
        "num_clients": actual_K,
        "requested_clients": K,
        "partition": exp["partition"],
        "label": exp["label"],
        "history": history,
        "final_metrics": history[-1] if history else {},
        "per_client_acc": per_client_acc,
        "fairness": fairness,
        "samples_per_client_stats": {
            "min": min_samples, "max": max_samples,
            "mean": int(np.mean(list(samples_per_client.values()))) if samples_per_client else 0,
        },
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
    parser = argparse.ArgumentParser(description="FL-EHDS Scalability Sweep")
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
            log("RESUMED: {} experiments already completed".format(done))

    if checkpoint_data is None:
        checkpoint_data = {
            "completed": {},
            "metadata": {
                "total_experiments": total_exps,
                "algorithms": ALGORITHMS if not args.quick else ALGORITHMS[:3],
                "seeds": seeds,
                "cv_clients": CV_CLIENTS if not args.quick else [50],
                "px_experiments": [e["label"] for e in (PX_EXPERIMENTS if not args.quick else PX_EXPERIMENTS[:1])],
                "start_time": datetime.now().isoformat(),
                "last_save": None,
                "version": "scalability_v1",
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
        print("\n  Checkpoint salvato: {}/{}".format(done, total_exps))
        print("  Per riprendere: python -m benchmarks.run_scalability_sweep --resume")
        sys.exit(0)

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # Header
    mode = "QUICK" if args.quick else "FULL"
    log("\n" + "=" * 70)
    log("  FL-EHDS Scalability Sweep ({})".format(mode))
    log("  Cardiovascular: K={}".format(CV_CLIENTS if not args.quick else [50]))
    log("  PTB-XL: {}".format(
        ", ".join("{}({})".format(e["label"], e["num_clients"]) for e in (PX_EXPERIMENTS if not args.quick else PX_EXPERIMENTS[:1]))
    ))
    log("  Algorithms: {}".format(len(ALGORITHMS if not args.quick else ALGORITHMS[:3])))
    log("  Seeds: {}".format(seeds))
    log("  Total experiments: {}".format(total_exps))
    log("  Device: {}".format(_detect_device()))
    log("=" * 70)

    start_time = time.time()
    completed = checkpoint_data.get("completed", {})
    done_count = len(completed)
    errors = []

    for i, exp in enumerate(experiments):
        key = exp["key"]
        if key in completed:
            continue

        algo = exp["algorithm"]
        K = exp["num_clients"]
        ds = exp["dataset"]
        seed = exp["seed"]
        label = exp["label"]

        log("  [{}/{}] {} {} {} s={} ...".format(
            done_count + 1, total_exps, ds, algo, label, seed))

        try:
            result = run_single_experiment(exp, quick=args.quick)
            completed[key] = result
            done_count += 1

            acc = result["final_metrics"].get("accuracy", 0) * 100
            rounds = result["actual_rounds"]
            rt = result["runtime_seconds"]
            actual_K = result["num_clients"]
            spc = result["samples_per_client_stats"]
            log("    -> {:.1f}% (K={}, {}r, {:.0f}s, samples/client: {}-{})".format(
                acc, actual_K, rounds, rt, spc["min"], spc["max"]))

            if done_count % 5 == 0:
                save_checkpoint(checkpoint_data)

        except Exception as e:
            log("    ERROR: {}".format(e))
            errors.append({"key": key, "error": str(e)})
            traceback.print_exc()
            _cleanup_gpu()

    # Final save
    checkpoint_data["metadata"]["end_time"] = datetime.now().isoformat()
    checkpoint_data["metadata"]["errors"] = errors
    save_checkpoint(checkpoint_data)

    elapsed = time.time() - start_time

    # ====== Print summary ======
    log("\n" + "=" * 70)
    log("  COMPLETED: {}/{}".format(done_count, total_exps))
    log("  Total time: {}".format(timedelta(seconds=int(elapsed))))
    log("  Checkpoint: {}".format(OUTPUT_DIR / CHECKPOINT_FILE))
    log("=" * 70)

    # ---- Scalability Results Table ----
    log("\n  Scalability Results Summary:")
    log("  {:<14s} | {:<6s} | {:<10s} | {:>10s} | {:>6s} | {:>5s} | {:>5s} | {:>12s}".format(
        "Dataset", "K", "Algorithm", "Acc (%)", "F1", "Jain", "Gini", "Samples/cl"))
    log("  " + "-" * 80)

    # Group by dataset+K
    groups = defaultdict(list)
    for key, r in completed.items():
        gkey = "{}_{}".format(r["dataset"], r["label"])
        groups[gkey].append(r)

    for gkey in sorted(groups.keys()):
        results = groups[gkey]
        ds = results[0]["dataset"]
        label = results[0]["label"]
        K = results[0]["num_clients"]

        # Group by algorithm
        algo_results = defaultdict(list)
        for r in results:
            algo_results[r["algorithm"]].append(r)

        for algo in (ALGORITHMS if not args.quick else ALGORITHMS[:3]):
            if algo not in algo_results:
                continue
            runs = algo_results[algo]
            accs = [r["final_metrics"].get("accuracy", 0) * 100 for r in runs]
            f1s = [r["final_metrics"].get("f1", 0) for r in runs]
            jains = [r["fairness"].get("jain_index", 0) for r in runs]
            ginis = [r["fairness"].get("gini", 0) for r in runs]
            spc_min = min(r["samples_per_client_stats"]["min"] for r in runs)
            spc_max = max(r["samples_per_client_stats"]["max"] for r in runs)

            log("  {:<14s} | K={:<3d} | {:<10s} | {:5.1f}+-{:4.1f} | {:.3f} | {:.3f} | {:.3f} | {:>4d}-{:<4d}".format(
                ds, K, algo,
                np.mean(accs), np.std(accs),
                np.mean(f1s), np.mean(jains), np.mean(ginis),
                spc_min, spc_max,
            ))

    # ---- Scalability Degradation Analysis ----
    log("\n  Scalability Degradation Analysis (vs baseline K=5):")
    log("  {:<14s} | {:<6s} | {:<10s} | {:>10s} | {:>10s}".format(
        "Dataset", "K", "Algorithm", "Acc (%)", "vs K=5"))
    log("  " + "-" * 60)

    # Reference K=5 results from checkpoint_tabular.json (if available)
    baseline_path = OUTPUT_DIR / "checkpoint_tabular.json"
    baseline_accs = {}
    if baseline_path.exists():
        try:
            with open(baseline_path) as f:
                baseline_data = json.load(f)
            base_completed = baseline_data.get("completed", {})
            for bkey, br in base_completed.items():
                # Phase 1 keys: {DS}_{Algo}_{seed}
                parts = bkey.split("_")
                if len(parts) >= 3:
                    ds_short = parts[0]
                    algo_name = parts[1]
                    bds = "Cardiovascular" if ds_short == "CV" else ("PTB_XL" if ds_short == "PX" else None)
                    if bds and isinstance(br, dict):
                        acc = br.get("final_metrics", {}).get("accuracy", 0) * 100
                        ref_key = "{}_{}".format(bds, algo_name)
                        if ref_key not in baseline_accs:
                            baseline_accs[ref_key] = []
                        baseline_accs[ref_key].append(acc)
        except Exception:
            pass

    for gkey in sorted(groups.keys()):
        results = groups[gkey]
        ds = results[0]["dataset"]
        label = results[0]["label"]
        K = results[0]["num_clients"]

        algo_results = defaultdict(list)
        for r in results:
            algo_results[r["algorithm"]].append(r)

        for algo in (ALGORITHMS if not args.quick else ALGORITHMS[:3]):
            if algo not in algo_results:
                continue
            runs = algo_results[algo]
            accs = [r["final_metrics"].get("accuracy", 0) * 100 for r in runs]
            mean_acc = np.mean(accs)

            ref_key = "{}_{}".format(ds, algo)
            if ref_key in baseline_accs:
                baseline_mean = np.mean(baseline_accs[ref_key])
                delta = mean_acc - baseline_mean
                delta_str = "{:+.1f}pp".format(delta)
            else:
                delta_str = "N/A"

            log("  {:<14s} | K={:<3d} | {:<10s} | {:5.1f}+-{:4.1f} | {:>10s}".format(
                ds, K, algo, mean_acc, np.std(accs), delta_str,
            ))

    # ---- Communication efficiency note ----
    log("\n  Communication Efficiency:")
    for gkey in sorted(groups.keys()):
        results = groups[gkey]
        ds = results[0]["dataset"]
        K = results[0]["num_clients"]
        avg_rounds = np.mean([r["actual_rounds"] for r in results])
        avg_runtime = np.mean([r["runtime_seconds"] for r in results])
        log("  {} K={}: avg {:.0f} rounds, {:.0f}s per experiment".format(
            ds, K, avg_rounds, avg_runtime))

    if errors:
        log("\n  ERRORS ({}):".format(len(errors)))
        for e in errors:
            log("    {}: {}".format(e["key"], e["error"]))


if __name__ == "__main__":
    main()

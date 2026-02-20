#!/usr/bin/env python3
"""
FL-EHDS — Combined Scalability + DP on Cardiovascular (K=50, ε=10).

Validates that ε=10 DP remains "free" at deployment scale (50 clients)
on the Cardiovascular dataset (70K samples, binary classification).
This bridges the gap between separate K=50 scalability results (Table XVII)
and K=5 DP results (Table IX/XXI).

Experiments:
  - 3 algos (FedAvg, Ditto, HPFL) × 2 DP (no_dp, ε=10) × 3 seeds = 18 experiments
  - Estimated runtime: ~30-60 min

Usage:
    cd fl-ehds-framework
    python -m benchmarks.run_scalability_dp_cv [--quick] [--fresh]

Output: benchmarks/paper_results_tabular/checkpoint_scalability_dp_cv.json

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
from typing import Dict, Optional, Any
from collections import defaultdict

import numpy as np

FRAMEWORK_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(FRAMEWORK_DIR))

import torch

from terminal.fl_trainer import (
    FederatedTrainer,
    HealthcareMLP,
    _detect_device,
)
from data.cardiovascular_loader import load_cardiovascular_data

# ======================================================================
# Constants
# ======================================================================

OUTPUT_DIR = FRAMEWORK_DIR / "benchmarks" / "paper_results_tabular"
CHECKPOINT_FILE = "checkpoint_scalability_dp_cv.json"
LOG_FILE = "experiment_scalability_dp_cv.log"

ALGORITHMS = ["FedAvg", "Ditto", "HPFL"]
K = 50  # deployment-scale client count

DP_SETTINGS = [
    {"enabled": False, "label": "no_dp"},
    {"enabled": True, "epsilon": 10.0, "clip_norm": 1.0, "label": "eps10"},
]
SEEDS = [42, 123, 456]

# Cardiovascular config (same as Table XVII / run_scalability_sweep.py)
CV_CONFIG = dict(
    input_dim=11, num_classes=2,
    learning_rate=0.01, batch_size=64, num_rounds=25, local_epochs=3,
    mu=0.1,
    es=dict(enabled=True, patience=6, min_delta=0.003, min_rounds=10),
)


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
# Checkpoint
# ======================================================================

def save_checkpoint(data: Dict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / CHECKPOINT_FILE
    bak = OUTPUT_DIR / (CHECKPOINT_FILE + ".bak")
    data["metadata"]["last_save"] = datetime.now().isoformat()
    fd, tmp = tempfile.mkstemp(dir=str(OUTPUT_DIR), prefix=".sdpcv_", suffix=".tmp")
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
# GPU cleanup & Early Stopping
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
# Per-client evaluation (HPFL-aware)
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
            X_t = torch.FloatTensor(X).to(trainer.device)
            y_t = torch.LongTensor(y).to(trainer.device)
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
# Training
# ======================================================================

def run_single_experiment(algo, dp_setting, seed, quick=False):
    """Run one experiment: algorithm + K=50 + DP setting."""
    start = time.time()
    num_rounds = 5 if quick else CV_CONFIG["num_rounds"]

    client_data, client_test_data, metadata = load_cardiovascular_data(
        num_clients=K, seed=seed,
        is_iid=False, alpha=0.5,
    )
    actual_K = len(client_data)

    dp_enabled = dp_setting["enabled"]
    dp_epsilon = dp_setting.get("epsilon", 10.0)
    dp_clip = dp_setting.get("clip_norm", 1.0)

    trainer = FederatedTrainer(
        num_clients=actual_K,
        algorithm=algo,
        local_epochs=CV_CONFIG["local_epochs"] if not quick else 1,
        batch_size=CV_CONFIG["batch_size"],
        learning_rate=CV_CONFIG["learning_rate"],
        mu=CV_CONFIG["mu"],
        seed=seed,
        external_data=client_data,
        external_test_data=client_test_data,
        input_dim=CV_CONFIG["input_dim"],
        num_classes=CV_CONFIG["num_classes"],
        dp_enabled=dp_enabled,
        dp_epsilon=dp_epsilon,
        dp_clip_norm=dp_clip,
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
            "round": r + 1,
            "accuracy": rr.global_acc,
            "loss": rr.global_loss,
            "f1": rr.global_f1,
        }
        history.append(metrics)
        if rr.global_acc > best_acc:
            best_acc = rr.global_acc
            best_round = r + 1

        log("    R{}/{} | Acc:{:.1%} | Best:{:.1%}(r{})".format(
            r + 1, num_rounds, rr.global_acc, best_acc, best_round))

        if es and es.check(r + 1, {"accuracy": rr.global_acc}):
            log("    -> Early stop at R{}".format(r + 1))
            break

    per_client_acc = _evaluate_per_client(trainer)
    fairness = _compute_fairness(per_client_acc)
    elapsed = time.time() - start

    result = {
        "dataset": "Cardiovascular",
        "algorithm": algo,
        "num_clients": actual_K,
        "requested_clients": K,
        "dp_enabled": dp_enabled,
        "dp_epsilon": dp_epsilon if dp_enabled else None,
        "dp_label": dp_setting["label"],
        "seed": seed,
        "history": history,
        "final_metrics": history[-1] if history else {},
        "per_client_acc": per_client_acc,
        "fairness": fairness,
        "best_accuracy": best_acc,
        "best_round": best_round,
        "actual_rounds": len(history),
        "runtime_seconds": round(elapsed, 1),
    }

    _cleanup_gpu()
    return result


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="FL-EHDS Combined Scalability + DP on Cardiovascular (K=50)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick validation (1 seed, 5 rounds)")
    parser.add_argument("--fresh", action="store_true",
                        help="Discard existing checkpoint")
    args = parser.parse_args()

    global _log_file
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _log_file = open(OUTPUT_DIR / LOG_FILE, "a")

    seeds = [42] if args.quick else SEEDS
    algos = ALGORITHMS[:2] if args.quick else ALGORITHMS

    experiments = []
    for dp in DP_SETTINGS:
        for algo in algos:
            for seed in seeds:
                key = "CV_{}_K{}_{}_s{}".format(algo, K, dp["label"], seed)
                experiments.append({
                    "key": key, "algorithm": algo,
                    "dp_setting": dp, "seed": seed,
                })

    total_exps = len(experiments)

    if args.fresh:
        ckpt_path = OUTPUT_DIR / CHECKPOINT_FILE
        if ckpt_path.exists():
            ckpt_path.unlink()
        log("FRESH start")

    checkpoint_data = load_checkpoint()
    if checkpoint_data is None:
        checkpoint_data = {
            "completed": {},
            "metadata": {
                "total_experiments": total_exps,
                "algorithms": list(algos),
                "dataset": "Cardiovascular",
                "K": K,
                "dp_settings": [d["label"] for d in DP_SETTINGS],
                "seeds": list(seeds),
                "config": CV_CONFIG,
                "start_time": datetime.now().isoformat(),
                "last_save": None,
            }
        }
    else:
        done = len(checkpoint_data.get("completed", {}))
        log("AUTO-RESUMED: {}/{} completed".format(done, total_exps))

    # Graceful shutdown
    _interrupted = [False]

    def _signal_handler(signum, frame):
        if _interrupted[0]:
            sys.exit(1)
        _interrupted[0] = True
        done = len(checkpoint_data.get("completed", {}))
        save_checkpoint(checkpoint_data)
        log("\n  Checkpoint saved: {}/{}".format(done, total_exps))
        log("  Resume: python -m benchmarks.run_scalability_dp_cv")
        sys.exit(0)

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    mode = "QUICK" if args.quick else "FULL"
    log("\n" + "=" * 60)
    log("  FL-EHDS Combined Scalability + DP — Cardiovascular ({})".format(mode))
    log("  K={}, DP: no_dp + ε=10".format(K))
    log("  {} experiments = {} algos × 2 DP × {} seeds".format(
        total_exps, len(algos), len(seeds)))
    log("  Algorithms: {}".format(list(algos)))
    log("  Device: {}".format(_detect_device(None)))
    log("  Output: {}".format(OUTPUT_DIR / CHECKPOINT_FILE))
    log("=" * 60)

    global_start = time.time()
    completed_count = len(checkpoint_data.get("completed", {}))

    for exp_idx, exp in enumerate(experiments, 1):
        key = exp["key"]
        if key in checkpoint_data.get("completed", {}):
            log("[{}/{}] {} — SKIPPED (done)".format(exp_idx, total_exps, key))
            continue
        if _interrupted[0]:
            break

        algo = exp["algorithm"]
        dp = exp["dp_setting"]
        seed = exp["seed"]
        dp_str = "ε={}".format(dp["epsilon"]) if dp["enabled"] else "no-DP"

        log("\n--- [{}/{}] CV | {} | K={} | {} | s{} ---".format(
            exp_idx, total_exps, algo, K, dp_str, seed))

        try:
            result = run_single_experiment(algo, dp, seed, quick=args.quick)
            checkpoint_data["completed"][key] = result
            completed_count += 1
            save_checkpoint(checkpoint_data)

            best = result["best_accuracy"]
            rounds = result["actual_rounds"]
            rt = result["runtime_seconds"]
            log("--- Done: {} | {} | Best={:.1%} | R{} | {:.0f}s | [{}/{}] ---".format(
                algo, dp_str, best, rounds, rt, completed_count, total_exps))

        except Exception as e:
            log("ERROR in {}: {}".format(key, e))
            traceback.print_exc()
            checkpoint_data["completed"][key] = {
                "dataset": "Cardiovascular", "algorithm": algo,
                "dp_label": dp["label"], "seed": seed,
                "error": str(e),
            }
            save_checkpoint(checkpoint_data)

    elapsed_total = time.time() - global_start
    checkpoint_data["metadata"]["end_time"] = datetime.now().isoformat()
    checkpoint_data["metadata"]["total_elapsed"] = elapsed_total
    save_checkpoint(checkpoint_data)

    # ================================================================
    # Summary table
    # ================================================================
    log("\n" + "=" * 60)
    log("  COMPLETED: {}/{}".format(completed_count, total_exps))
    log("  Total time: {}".format(timedelta(seconds=int(elapsed_total))))
    log("=" * 60)

    completed = checkpoint_data.get("completed", {})

    # Build summary: algo -> dp_label -> [accuracies]
    summary = defaultdict(lambda: defaultdict(list))
    for key, res in completed.items():
        if "error" in res:
            continue
        algo = res["algorithm"]
        dp_label = res.get("dp_label", "no_dp")
        summary[algo][dp_label].append(res["best_accuracy"])

    log("\n  Cardiovascular K={} — Scalability + DP Summary:".format(K))
    log("  {:<10s} {:>12s} {:>12s} {:>8s}".format("Algorithm", "No DP", "ε=10", "Δ (pp)"))
    log("  " + "-" * 46)
    for algo in ALGORITHMS:
        no_dp = summary[algo].get("no_dp", [])
        eps10 = summary[algo].get("eps10", [])
        if no_dp and eps10:
            nd_mean = np.mean(no_dp) * 100
            nd_std = np.std(no_dp) * 100
            ep_mean = np.mean(eps10) * 100
            ep_std = np.std(eps10) * 100
            delta = ep_mean - nd_mean
            log("  {:<10s} {:5.1f}±{:<5.1f} {:5.1f}±{:<5.1f} {:>+6.1f}".format(
                algo, nd_mean, nd_std, ep_mean, ep_std, delta))
        elif no_dp:
            nd_mean = np.mean(no_dp) * 100
            log("  {:<10s} {:5.1f}        ---".format(algo, nd_mean))

    if _log_file:
        _log_file.close()


if __name__ == "__main__":
    main()

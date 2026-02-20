#!/usr/bin/env python3
"""
FL-EHDS Test F — Scalability with Differential Privacy on PTB-XL.

Tests whether ε=10 DP remains "free" (<2pp accuracy cost) at larger scales.
Compares FedAvg, Ditto, and HPFL with and without DP at K=5, K=10, K=20.

Experiments:
  - 3 algos × 3 K values × 2 DP settings × 3 seeds = 54 experiments
  Estimated runtime: ~2-3 hours

Usage:
    cd fl-ehds-framework
    python -m benchmarks.run_scalability_dp [--quick] [--fresh] [--resume]

Output: benchmarks/paper_results_tabular/checkpoint_scalability_dp.json

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
from data.ptbxl_loader import load_ptbxl_data

# ======================================================================
# Constants
# ======================================================================

OUTPUT_DIR = FRAMEWORK_DIR / "benchmarks" / "paper_results_tabular"
CHECKPOINT_FILE = "checkpoint_scalability_dp.json"
LOG_FILE = "experiment_scalability_dp.log"

ALGORITHMS = ["FedAvg", "Ditto", "HPFL"]
CLIENT_COUNTS = [5, 10, 20]
DP_SETTINGS = [
    {"enabled": False, "label": "no_dp"},
    {"enabled": True, "epsilon": 10.0, "clip_norm": 1.0, "label": "eps10"},
]
SEEDS = [42, 123, 456]

# PTB-XL config
PX_CONFIG = dict(
    input_dim=9, num_classes=5,
    learning_rate=0.005, batch_size=64, num_rounds=30, local_epochs=3,
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
    fd, tmp = tempfile.mkstemp(dir=str(OUTPUT_DIR), prefix=".sdp_", suffix=".tmp")
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


# ======================================================================
# Training
# ======================================================================

def run_single_experiment(algo, K, dp_setting, seed, quick=False):
    """Run one experiment: algorithm + K clients + DP setting."""
    start = time.time()
    num_rounds = 5 if quick else PX_CONFIG["num_rounds"]

    client_data, client_test_data, metadata = load_ptbxl_data(
        num_clients=K, seed=seed,
        partition_by_site=True, min_site_samples=50,
    )
    actual_K = len(client_data)

    dp_enabled = dp_setting["enabled"]
    dp_epsilon = dp_setting.get("epsilon", 10.0)
    dp_clip = dp_setting.get("clip_norm", 1.0)

    trainer = FederatedTrainer(
        num_clients=actual_K,
        algorithm=algo,
        local_epochs=PX_CONFIG["local_epochs"],
        batch_size=PX_CONFIG["batch_size"],
        learning_rate=PX_CONFIG["learning_rate"],
        mu=PX_CONFIG["mu"],
        seed=seed,
        external_data=client_data,
        external_test_data=client_test_data,
        input_dim=PX_CONFIG["input_dim"],
        num_classes=PX_CONFIG["num_classes"],
        dp_enabled=dp_enabled,
        dp_epsilon=dp_epsilon,
        dp_clip_norm=dp_clip,
    )

    es_cfg = PX_CONFIG["es"]
    es = EarlyStoppingMonitor(
        patience=es_cfg["patience"], min_delta=es_cfg["min_delta"],
        min_rounds=es_cfg["min_rounds"],
    ) if es_cfg.get("enabled") and not quick else None

    history = []
    best_acc = 0.0

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
        if es and es.check(r + 1, {"accuracy": rr.global_acc}):
            break

    per_client_acc = _evaluate_per_client(trainer)
    elapsed = time.time() - start

    result = {
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
        "best_accuracy": best_acc,
        "actual_rounds": len(history),
        "runtime_seconds": round(elapsed, 1),
    }

    _cleanup_gpu()
    return result


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="FL-EHDS Scalability × DP on PTB-XL")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    global _log_file
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _log_file = open(OUTPUT_DIR / LOG_FILE, "a")

    seeds = [42] if args.quick else SEEDS
    algos = ALGORITHMS[:2] if args.quick else ALGORITHMS
    clients = [5, 20] if args.quick else CLIENT_COUNTS

    experiments = []
    for K in clients:
        for dp in DP_SETTINGS:
            for algo in algos:
                for seed in seeds:
                    key = "PX_{}_K{}_{}_s{}".format(algo, K, dp["label"], seed)
                    experiments.append({
                        "key": key, "algorithm": algo, "K": K,
                        "dp_setting": dp, "seed": seed,
                    })

    total_exps = len(experiments)

    if args.fresh:
        ckpt_path = OUTPUT_DIR / CHECKPOINT_FILE
        if ckpt_path.exists():
            ckpt_path.unlink()

    checkpoint_data = None
    if args.resume or not args.fresh:
        checkpoint_data = load_checkpoint()
        if checkpoint_data:
            done = len(checkpoint_data.get("completed", {}))
            log("RESUMED: {} completed".format(done))

    if checkpoint_data is None:
        checkpoint_data = {
            "completed": {},
            "metadata": {
                "total_experiments": total_exps,
                "algorithms": algos,
                "client_counts": clients,
                "dp_settings": [d["label"] for d in DP_SETTINGS],
                "seeds": seeds,
                "start_time": datetime.now().isoformat(),
                "last_save": None,
            }
        }

    _interrupted = [False]
    def _signal_handler(signum, frame):
        if _interrupted[0]:
            sys.exit(1)
        _interrupted[0] = True
        save_checkpoint(checkpoint_data)
        print("\n  Checkpoint salvato.")
        sys.exit(0)
    signal.signal(signal.SIGINT, _signal_handler)

    mode = "QUICK" if args.quick else "FULL"
    log("\n" + "=" * 60)
    log("  FL-EHDS Scalability x DP on PTB-XL ({})".format(mode))
    log("  Algos: {}".format(algos))
    log("  K: {}".format(clients))
    log("  DP: {}".format([d["label"] for d in DP_SETTINGS]))
    log("  Seeds: {}".format(seeds))
    log("  Total: {} experiments".format(total_exps))
    log("  Device: {}".format(_detect_device()))
    log("=" * 60)

    start_time = time.time()
    completed = checkpoint_data.get("completed", {})
    done_count = len(completed)

    for exp in experiments:
        key = exp["key"]
        if key in completed:
            continue

        algo = exp["algorithm"]
        K = exp["K"]
        dp = exp["dp_setting"]
        seed = exp["seed"]

        log("  [{}/{}] {} K={} {} s={} ...".format(
            done_count + 1, total_exps, algo, K, dp["label"], seed))

        try:
            result = run_single_experiment(algo, K, dp, seed, args.quick)
            completed[key] = result
            done_count += 1

            acc = result["best_accuracy"] * 100
            rt = result["runtime_seconds"]
            dp_str = "DP(eps={})".format(dp.get("epsilon", "-")) if dp["enabled"] else "no-DP"
            log("    -> {:.1f}% | K={} | {} | {:.0f}s".format(acc, result["num_clients"], dp_str, rt))

            if done_count % 5 == 0:
                save_checkpoint(checkpoint_data)

        except Exception as e:
            log("    ERROR: {}".format(e))
            traceback.print_exc()
            _cleanup_gpu()

    checkpoint_data["metadata"]["end_time"] = datetime.now().isoformat()
    save_checkpoint(checkpoint_data)

    elapsed = time.time() - start_time
    log("\n" + "=" * 60)
    log("  COMPLETED: {}/{}".format(done_count, total_exps))
    log("  Total time: {}".format(timedelta(seconds=int(elapsed))))
    log("=" * 60)

    # Summary table
    log("\n  Scalability x DP Results (PTB-XL):")
    log("  {:>10s} | {:>4s} | {:>8s} | {:>10s} | {:>8s}".format(
        "Algorithm", "K", "DP", "Acc (%)", "DP cost"))
    log("  " + "-" * 55)

    for K in clients:
        for algo in algos:
            no_dp_accs = []
            dp_accs = []
            for seed in seeds:
                key_nd = "PX_{}_K{}_{}_s{}".format(algo, K, "no_dp", seed)
                key_dp = "PX_{}_K{}_{}_s{}".format(algo, K, "eps10", seed)
                if key_nd in completed:
                    no_dp_accs.append(completed[key_nd]["best_accuracy"] * 100)
                if key_dp in completed:
                    dp_accs.append(completed[key_dp]["best_accuracy"] * 100)

            if no_dp_accs:
                log("  {:>10s} | {:>4d} | {:>8s} | {:5.1f}+-{:.1f} |".format(
                    algo, K, "no_dp", np.mean(no_dp_accs), np.std(no_dp_accs)))
            if dp_accs:
                dp_cost = np.mean(no_dp_accs) - np.mean(dp_accs) if no_dp_accs else 0
                log("  {:>10s} | {:>4d} | {:>8s} | {:5.1f}+-{:.1f} | {:+.1f}pp".format(
                    algo, K, "eps10", np.mean(dp_accs), np.std(dp_accs), -dp_cost))
        log("  " + "-" * 55)

    if _log_file:
        _log_file.close()


if __name__ == "__main__":
    main()

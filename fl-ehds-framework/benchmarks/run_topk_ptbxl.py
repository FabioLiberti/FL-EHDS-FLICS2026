#!/usr/bin/env python3
"""
FL-EHDS Test E — Top-k Sparsification Impact on PTB-XL.

Compares FedAvg with and without Top-k (1%) gradient sparsification
on PTB-XL to quantify communication efficiency vs accuracy trade-off.

Approach: After each FL round, intercept the model update (delta),
apply Top-k sparsification (keep only 1% of parameters by magnitude),
and use the sparsified update for the next round. This simulates
communication-efficient FL.

Experiments:
  - FedAvg baseline (no compression): 3 seeds
  - FedAvg + Top-k 1%: 3 seeds
  - FedAvg + Top-k 5%: 3 seeds
  Total: 9 experiments (~30-60 min)

Usage:
    cd fl-ehds-framework
    python -m benchmarks.run_topk_ptbxl [--quick] [--fresh]

Output: benchmarks/paper_results_tabular/checkpoint_topk_ptbxl.json

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
from collections import OrderedDict

import numpy as np

FRAMEWORK_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(FRAMEWORK_DIR))

import torch
import torch.nn as nn

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
CHECKPOINT_FILE = "checkpoint_topk_ptbxl.json"
LOG_FILE = "experiment_topk_ptbxl.log"

SEEDS = [42, 123, 456]
K_RATIOS = [1.0, 0.05, 0.01]  # 100% (baseline), 5%, 1%
K_LABELS = {1.0: "baseline", 0.05: "top5pct", 0.01: "top1pct"}

# PTB-XL config (same as paper)
PX_CONFIG = dict(
    input_dim=9, num_classes=5,
    learning_rate=0.005, batch_size=64, num_rounds=30, local_epochs=3,
    mu=0.1,
    es=dict(enabled=True, patience=6, min_delta=0.003, min_rounds=10),
)
NUM_CLIENTS = 5


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
    fd, tmp = tempfile.mkstemp(dir=str(OUTPUT_DIR), prefix=".topk_", suffix=".tmp")
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
# Top-k Sparsification
# ======================================================================

def apply_topk_to_model_update(model, old_state, k_ratio):
    """
    Apply Top-k sparsification to the model update.
    After trainer.train_round() updates the global model, we:
    1. Compute delta = new_weights - old_weights
    2. Keep only top k_ratio% by magnitude
    3. Set new_weights = old_weights + sparsified_delta

    Returns compression stats.
    """
    if k_ratio >= 1.0:
        return {"compression_ratio": 1.0, "kept_params": 0, "total_params": 0}

    total_params = 0
    kept_params = 0

    with torch.no_grad():
        for name, param in model.named_parameters():
            if name not in old_state:
                continue

            delta = param.data - old_state[name]
            flat = delta.flatten()
            n = flat.numel()
            k = max(1, int(n * k_ratio))
            total_params += n
            kept_params += k

            # Top-k selection by magnitude
            _, topk_idx = torch.topk(flat.abs(), k)
            mask = torch.zeros_like(flat)
            mask[topk_idx] = 1.0
            sparsified = flat * mask

            # Apply sparsified update
            param.data = old_state[name] + sparsified.reshape(param.data.shape)

    compression_ratio = kept_params / total_params if total_params > 0 else 1.0
    return {
        "compression_ratio": compression_ratio,
        "kept_params": kept_params,
        "total_params": total_params,
        "bandwidth_savings_pct": (1 - compression_ratio) * 100,
    }


# ======================================================================
# Training
# ======================================================================

def run_single_experiment(seed, k_ratio, quick=False):
    """Run FedAvg on PTB-XL with optional Top-k sparsification."""
    start = time.time()
    label = K_LABELS[k_ratio]
    num_rounds = 5 if quick else PX_CONFIG["num_rounds"]

    client_data, client_test_data, metadata = load_ptbxl_data(
        num_clients=NUM_CLIENTS, seed=seed,
        partition_by_site=True, min_site_samples=50,
    )

    trainer = FederatedTrainer(
        num_clients=len(client_data),
        algorithm="FedAvg",
        local_epochs=PX_CONFIG["local_epochs"],
        batch_size=PX_CONFIG["batch_size"],
        learning_rate=PX_CONFIG["learning_rate"],
        mu=PX_CONFIG["mu"],
        seed=seed,
        external_data=client_data,
        external_test_data=client_test_data,
        input_dim=PX_CONFIG["input_dim"],
        num_classes=PX_CONFIG["num_classes"],
    )

    es_cfg = PX_CONFIG["es"]
    es = EarlyStoppingMonitor(
        patience=es_cfg["patience"], min_delta=es_cfg["min_delta"],
        min_rounds=es_cfg["min_rounds"],
    ) if es_cfg.get("enabled") and not quick else None

    history = []
    best_acc = 0.0
    compression_stats = []

    for r in range(num_rounds):
        # Save state before round
        old_state = {n: p.data.clone() for n, p in trainer.global_model.named_parameters()}

        # Run FL round (aggregation happens internally)
        rr = trainer.train_round(r)

        # Apply Top-k sparsification to the update
        stats = apply_topk_to_model_update(trainer.global_model, old_state, k_ratio)
        compression_stats.append(stats)

        # Re-evaluate after sparsification (if applied)
        if k_ratio < 1.0:
            # The accuracy reported by train_round is BEFORE sparsification
            # We need to re-evaluate
            model = trainer.global_model
            model.eval()
            correct = total = 0
            with torch.no_grad():
                for cid in range(trainer.num_clients):
                    X, y = trainer.client_test_data[cid]
                    X_t = torch.FloatTensor(X).to(trainer.device)
                    y_t = torch.LongTensor(y).to(trainer.device)
                    for i in range(0, len(y_t), 64):
                        out = model(X_t[i:i+64])
                        correct += (out.argmax(1) == y_t[i:i+64]).sum().item()
                        total += len(y_t[i:i+64])
            acc_after = correct / total if total > 0 else 0
        else:
            acc_after = rr.global_acc

        metrics = {
            "round": r + 1,
            "accuracy": acc_after,
            "accuracy_before_sparsification": rr.global_acc,
            "loss": rr.global_loss,
            "f1": rr.global_f1,
        }
        history.append(metrics)

        if acc_after > best_acc:
            best_acc = acc_after

        if es and es.check(r + 1, {"accuracy": acc_after}):
            break

    elapsed = time.time() - start
    avg_compression = np.mean([s["compression_ratio"] for s in compression_stats])

    result = {
        "algorithm": "FedAvg",
        "k_ratio": k_ratio,
        "label": label,
        "seed": seed,
        "history": history,
        "final_metrics": history[-1] if history else {},
        "best_accuracy": best_acc,
        "actual_rounds": len(history),
        "runtime_seconds": round(elapsed, 1),
        "avg_compression_ratio": avg_compression,
        "bandwidth_savings_pct": (1 - avg_compression) * 100,
        "total_params": compression_stats[0]["total_params"] if compression_stats else 0,
    }

    _cleanup_gpu()
    return result


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="FL-EHDS Top-k Sparsification on PTB-XL")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()

    global _log_file
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _log_file = open(OUTPUT_DIR / LOG_FILE, "a")

    seeds = [42] if args.quick else SEEDS
    k_ratios = [1.0, 0.01] if args.quick else K_RATIOS

    experiments = [(k, seed) for k in k_ratios for seed in seeds]
    total_exps = len(experiments)

    if args.fresh:
        ckpt_path = OUTPUT_DIR / CHECKPOINT_FILE
        if ckpt_path.exists():
            ckpt_path.unlink()

    checkpoint_data = load_checkpoint()
    if checkpoint_data is None:
        checkpoint_data = {
            "completed": {},
            "metadata": {
                "total_experiments": total_exps,
                "k_ratios": k_ratios,
                "seeds": seeds,
                "start_time": datetime.now().isoformat(),
                "last_save": None,
            }
        }

    # SIGINT handler
    _interrupted = [False]
    def _signal_handler(signum, frame):
        if _interrupted[0]:
            sys.exit(1)
        _interrupted[0] = True
        save_checkpoint(checkpoint_data)
        print("\n  Checkpoint salvato.")
        sys.exit(0)
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    mode = "QUICK" if args.quick else "FULL"
    log("\n" + "=" * 60)
    log("  FL-EHDS Top-k Sparsification on PTB-XL ({})".format(mode))
    log("  K ratios: {}".format(k_ratios))
    log("  Seeds: {}".format(seeds))
    log("  Total: {} experiments".format(total_exps))
    log("  Device: {}".format(_detect_device()))
    log("=" * 60)

    start_time = time.time()
    completed = checkpoint_data.get("completed", {})
    done_count = len(completed)

    for k_ratio, seed in experiments:
        label = K_LABELS[k_ratio]
        key = "PX_FedAvg_{}_s{}".format(label, seed)

        if key in completed:
            continue

        log("\n  [{}/{}] {} seed={} ...".format(done_count + 1, total_exps, label, seed))

        try:
            result = run_single_experiment(seed, k_ratio, args.quick)
            completed[key] = result
            done_count += 1

            acc = result["best_accuracy"] * 100
            bw = result["bandwidth_savings_pct"]
            rt = result["runtime_seconds"]
            log("    -> {:.1f}% | BW savings: {:.0f}% | {:.0f}s".format(acc, bw, rt))

            if done_count % 3 == 0:
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
    log("\n  Top-k Results (PTB-XL, FedAvg):")
    log("  {:>12s} | {:>10s} | {:>10s} | {:>12s}".format(
        "Method", "Acc (%)", "BW Savings", "Params sent"))
    log("  " + "-" * 55)

    for k_ratio in k_ratios:
        label = K_LABELS[k_ratio]
        accs = []
        bws = []
        for seed in seeds:
            key = "PX_FedAvg_{}_s{}".format(label, seed)
            if key in completed:
                accs.append(completed[key]["best_accuracy"] * 100)
                bws.append(completed[key]["bandwidth_savings_pct"])
        if accs:
            total_p = completed.get("PX_FedAvg_{}_s{}".format(label, seeds[0]), {}).get("total_params", 0)
            kept = int(total_p * k_ratio)
            log("  {:>12s} | {:5.1f}+-{:.1f} | {:>8.0f}%  | {:>6d}/{:d}".format(
                label, np.mean(accs), np.std(accs), np.mean(bws), kept, total_p))

    if _log_file:
        _log_file.close()


if __name__ == "__main__":
    main()

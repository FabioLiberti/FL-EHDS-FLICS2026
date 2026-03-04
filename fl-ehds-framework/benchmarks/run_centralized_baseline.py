#!/usr/bin/env python3
"""
Centralized (Non-Federated) Baseline Benchmark for FL-EHDS Paper.

Trains a centralized HealthcareMLP on each dataset to establish the
upper-bound baseline for comparison with federated learning results.
When all data resides on a single node with no privacy constraints,
we expect the best achievable accuracy --- any FL result should be
compared against this ceiling.

Grid:
  3 datasets x 3 seeds = 9 experiments (full)
  1 dataset  x 1 seed  = 1 experiment  (--quick)

Datasets:
  - Cardiovascular  (input_dim=11, num_classes=2, ~70K samples)
  - PTB-XL          (input_dim=9,  num_classes=5, ~22K samples)
  - CDC_Diabetes    (input_dim=21, num_classes=2, ~254K samples)

Training: standard PyTorch loop, Adam optimizer, lr=0.01,
          batch_size=128, 50 epochs, CrossEntropyLoss.

Usage:
    cd fl-ehds-framework
    python -m benchmarks.run_centralized_baseline [--quick] [--dry-run] [--fresh]

Estimated time: ~30 min on MacBook Air M3.

Author: Fabio Liberti
"""

import argparse
import json
import os
import signal
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score

FRAMEWORK_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(FRAMEWORK_DIR))

OUTPUT_DIR = Path(__file__).parent / "paper_results_tabular"
CHECKPOINT_FILE = "checkpoint_centralized_baseline.json"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

_shutdown = False


def _signal_handler(signum, frame):
    global _shutdown
    _shutdown = True
    print("\nGraceful shutdown requested...")


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


def log(msg):
    ts = time.strftime("%H:%M:%S")
    print("[{}] {}".format(ts, msg))


# ======================================================================
# Atomic checkpoint (tempfile + fsync + os.replace + .bak)
# ======================================================================

def save_checkpoint(data):
    path = OUTPUT_DIR / CHECKPOINT_FILE
    bak = OUTPUT_DIR / (CHECKPOINT_FILE + ".bak")
    fd, tmp = tempfile.mkstemp(dir=str(OUTPUT_DIR), prefix=".ckpt_", suffix=".tmp")
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


def load_checkpoint():
    for p in [OUTPUT_DIR / CHECKPOINT_FILE, OUTPUT_DIR / (CHECKPOINT_FILE + ".bak")]:
        if p.exists():
            try:
                with open(p) as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                continue
    return None


def _cleanup_gpu():
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


# ======================================================================
# Dataset definitions
# ======================================================================

DATASETS = {
    "Cardiovascular": {
        "input_dim": 11,
        "num_classes": 2,
        "f1_average": "binary",
    },
    "PTB-XL": {
        "input_dim": 9,
        "num_classes": 5,
        "f1_average": "macro",
    },
    "CDC_Diabetes": {
        "input_dim": 21,
        "num_classes": 2,
        "f1_average": "binary",
    },
}

ALL_SEEDS = [42, 123, 456]


def load_dataset(name, seed):
    """Load dataset as a single centralized client (num_clients=1)."""
    if name == "Cardiovascular":
        from data.cardiovascular_loader import load_cardiovascular_data
        return load_cardiovascular_data(num_clients=1, is_iid=True, seed=seed)
    elif name == "PTB-XL":
        from data.ptbxl_loader import load_ptbxl_data
        return load_ptbxl_data(num_clients=1, seed=seed)
    elif name == "CDC_Diabetes":
        from data.cdc_diabetes_loader import load_cdc_diabetes_data
        return load_cdc_diabetes_data(num_clients=1, is_iid=True, seed=seed)
    else:
        raise ValueError("Unknown dataset: {}".format(name))


# ======================================================================
# Centralized training
# ======================================================================

def run_single_experiment(dataset_name, seed):
    """Train a centralized HealthcareMLP and evaluate."""
    from terminal.training.models import HealthcareMLP

    cfg = DATASETS[dataset_name]
    input_dim = cfg["input_dim"]
    num_classes = cfg["num_classes"]
    f1_average = cfg["f1_average"]

    # --- Load data (single client) ---
    client_train, client_test, meta = load_dataset(dataset_name, seed)
    X_train, y_train = client_train[0]
    X_test, y_test = client_test[0]

    total_train = len(y_train)
    total_test = len(y_test)

    # --- Convert to tensors ---
    device = torch.device(DEVICE)
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.LongTensor(y_train).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)
    y_test_t = torch.LongTensor(y_test).to(device)

    # --- Model ---
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = HealthcareMLP(
        input_dim=input_dim,
        num_classes=num_classes,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    batch_size = 128
    num_epochs = 50
    n = len(y_train_t)

    # --- Training loop ---
    history = []

    for epoch in range(num_epochs):
        if _shutdown:
            break

        model.train()
        perm = torch.randperm(n, device=device)
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            idx = perm[start:end]
            X_batch = X_train_t[idx]
            y_batch = y_train_t[idx]

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(y_batch)
            preds = logits.argmax(dim=1)
            epoch_correct += (preds == y_batch).sum().item()
            epoch_total += len(y_batch)

        avg_loss = epoch_loss / max(epoch_total, 1)
        train_acc = epoch_correct / max(epoch_total, 1)

        history.append({
            "epoch": epoch + 1,
            "loss": round(avg_loss, 6),
            "accuracy": round(train_acc, 4),
        })

    # --- Evaluation ---
    model.eval()
    with torch.no_grad():
        logits = model(X_test_t)
        preds = logits.argmax(dim=1).cpu().numpy()
    y_true = y_test_t.cpu().numpy()

    accuracy = float((preds == y_true).mean())
    f1 = float(f1_score(y_true, preds, average=f1_average, zero_division=0))
    precision = float(precision_score(y_true, preds, average=f1_average, zero_division=0))
    recall_val = float(recall_score(y_true, preds, average=f1_average, zero_division=0))

    return {
        "dataset": dataset_name,
        "seed": seed,
        "input_dim": input_dim,
        "num_classes": num_classes,
        "total_train_samples": total_train,
        "total_test_samples": total_test,
        "num_epochs": len(history),
        "learning_rate": 0.01,
        "batch_size": 128,
        "optimizer": "Adam",
        "accuracy": round(accuracy, 4),
        "f1": round(f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall_val, 4),
        "history": history,
        "final_train_loss": history[-1]["loss"] if history else None,
        "final_train_accuracy": history[-1]["accuracy"] if history else None,
    }


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Centralized (non-federated) baseline benchmark")
    parser.add_argument("--quick", action="store_true",
                        help="Quick validation (1 dataset x 1 seed)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show plan without executing")
    parser.add_argument("--fresh", action="store_true",
                        help="Start fresh (ignore existing checkpoint)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Build experiment list ──
    experiments = []

    if args.quick:
        # 1 dataset x 1 seed = 1 experiment
        datasets = ["Cardiovascular"]
        seeds = [42]
    else:
        # 3 datasets x 3 seeds = 9 experiments
        datasets = list(DATASETS.keys())
        seeds = ALL_SEEDS

    for ds in datasets:
        for seed in seeds:
            key = "centralized_{}_s{}".format(ds, seed)
            experiments.append((key, ds, seed))

    total = len(experiments)

    log("=" * 70)
    log("  Centralized Baseline Benchmark (upper-bound reference)")
    log("=" * 70)
    log("  Device: {}".format(DEVICE))
    log("  Datasets: {}".format(", ".join(datasets)))
    log("  Seeds: {}".format(seeds))
    log("  Training: Adam lr=0.01, batch_size=128, 50 epochs, CrossEntropyLoss")
    log("  Total: {} experiments".format(total))
    log("=" * 70)

    if args.dry_run:
        for key, ds, seed in experiments:
            cfg = DATASETS[ds]
            log("  [DRY-RUN] {} (dim={}, classes={})".format(
                key, cfg["input_dim"], cfg["num_classes"]))
        log("\nDRY-RUN: {} experiments.".format(total))
        return

    # Load or create checkpoint
    checkpoint = None if args.fresh else load_checkpoint()
    if checkpoint is None:
        checkpoint = {
            "completed": {},
            "metadata": {
                "experiment": "centralized_baseline",
                "total_experiments": total,
                "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
        }

    t0 = time.time()
    completed = 0
    skipped = 0

    for idx, (key, ds, seed) in enumerate(experiments, 1):
        if _shutdown:
            log("Shutdown. Saving checkpoint...")
            save_checkpoint(checkpoint)
            break

        if key in checkpoint["completed"]:
            skipped += 1
            continue

        completed += 1
        cfg = DATASETS[ds]
        log("  [{}/{}] {} | seed={} | dim={} | classes={}".format(
            skipped + completed, total, ds, seed,
            cfg["input_dim"], cfg["num_classes"]))

        try:
            t_exp = time.time()
            result = run_single_experiment(dataset_name=ds, seed=seed)
            exp_time = time.time() - t_exp
            result["runtime_seconds"] = round(exp_time, 1)
            checkpoint["completed"][key] = result
            save_checkpoint(checkpoint)
            _cleanup_gpu()

            log("    Acc={:.1f}% | F1={:.4f} | Prec={:.4f} | Rec={:.4f} | {:.1f}s".format(
                result["accuracy"] * 100, result["f1"],
                result["precision"], result["recall"], exp_time))

        except Exception as e:
            log("    ERROR: {}".format(e))
            traceback.print_exc()
            checkpoint["completed"][key] = {"error": str(e), "dataset": ds, "seed": seed}
            save_checkpoint(checkpoint)
            _cleanup_gpu()

    elapsed = time.time() - t0
    checkpoint["metadata"]["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    checkpoint["metadata"]["total_time_seconds"] = round(elapsed, 1)
    save_checkpoint(checkpoint)

    # ── Summary table ──
    log("")
    log("=" * 70)
    log("  RESULTS SUMMARY")
    log("=" * 70)
    log("  {:20s} {:>6s} {:>8s} {:>8s} {:>8s} {:>8s}".format(
        "Dataset", "Seed", "Acc", "F1", "Prec", "Recall"))
    log("  " + "-" * 62)

    for key, ds, seed in experiments:
        res = checkpoint["completed"].get(key, {})
        if "error" in res:
            log("  {:20s} {:>6d} ERROR: {}".format(ds, seed, res["error"]))
        elif "accuracy" in res:
            log("  {:20s} {:>6d} {:>7.1f}% {:>8.4f} {:>8.4f} {:>8.4f}".format(
                ds, seed,
                res["accuracy"] * 100, res["f1"],
                res["precision"], res["recall"]))

    log("")
    log("COMPLETED: {}/{} ({} skipped) in {:.0f} min".format(
        completed, total, skipped, elapsed / 60))
    log("Checkpoint: {}".format(OUTPUT_DIR / CHECKPOINT_FILE))
    log("=" * 70)


if __name__ == "__main__":
    main()

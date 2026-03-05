#!/usr/bin/env python3
"""
FL-EHDS Experiment — Heart Disease Extended (10 seeds).

Compares Local-Only, Centralized, FL-FedAvg, FL-Ditto, FL-HPFL
on Heart Disease UCI with 10 seeds for maximum statistical robustness.

Design:
  - Dataset: Heart Disease UCI (920 samples, 4 hospitals, natural non-IID)
  - Modes: local, centralized, FedAvg, Ditto, HPFL
  - Seeds: 42, 123, 456, 789, 999, 1234, 2345, 3456, 4567, 5678
  - Total: 5 modes x 10 seeds = 50 experiments (~10 min on M3)

Usage:
    cd fl-ehds-framework
    python -m benchmarks.run_hd_extended [--fresh]

Output:
    benchmarks/paper_results_tabular/checkpoint_hd_extended.json

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
from typing import Dict, Any

import numpy as np

FRAMEWORK_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(FRAMEWORK_DIR))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from terminal.fl_trainer import FederatedTrainer, HealthcareMLP, _detect_device

# ======================================================================
# Constants
# ======================================================================

OUTPUT_DIR = FRAMEWORK_DIR / "benchmarks" / "paper_results_tabular"
CHECKPOINT_FILE = "checkpoint_hd_extended.json"
LOG_FILE = "experiment_hd_extended.log"

SEEDS = [42, 123, 456, 789, 999, 1234, 2345, 3456, 4567, 5678]

# Heart Disease config (matches existing experiments)
HD_CONFIG = {
    "name": "Heart Disease",
    "loader": "heart_disease",
    "input_dim": 13,
    "num_classes": 2,
    "num_clients": 4,
    "learning_rate": 0.01,
    "batch_size": 64,
    "num_rounds": 30,
    "local_epochs": 3,
    "mu": 0.1,
}

MODES = ["local", "centralized", "FedAvg", "Ditto", "HPFL"]

# ======================================================================
# Logging
# ======================================================================

_log_file = None


def log(msg, also_print=True):
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

def save_checkpoint(data):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / CHECKPOINT_FILE
    bak = OUTPUT_DIR / (CHECKPOINT_FILE + ".bak")
    data["metadata"]["last_save"] = datetime.now().isoformat()
    fd, tmp = tempfile.mkstemp(dir=str(OUTPUT_DIR), prefix=".hde_", suffix=".tmp")
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

def load_dataset(seed):
    from data.heart_disease_loader import load_heart_disease_data
    return load_heart_disease_data(
        num_clients=HD_CONFIG["num_clients"],
        partition_by_hospital=True,
        seed=seed,
    )


# ======================================================================
# Evaluation helper
# ======================================================================

def _evaluate_model(model, X, y, batch_size, device, num_classes):
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score,
        recall_score, roc_auc_score,
    )

    model.eval()
    dataset = TensorDataset(torch.FloatTensor(X), torch.LongTensor(y))
    loader = DataLoader(dataset, batch_size=batch_size)

    all_preds, all_labels, all_probs = [], [], []
    total_loss, n_batches = 0.0, 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            output = model(X_batch)
            loss = criterion(output, y_batch)
            total_loss += loss.item()
            n_batches += 1
            probs = torch.softmax(output, dim=1)
            preds = output.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    prec = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    rec = recall_score(all_labels, all_preds, average="macro", zero_division=0)

    try:
        if num_classes == 2:
            auc = roc_auc_score(all_labels, all_probs[:, 1])
        else:
            auc = roc_auc_score(all_labels, all_probs, multi_class="ovr", average="macro")
    except Exception:
        auc = 0.5

    return {
        "accuracy": float(acc), "f1": float(f1),
        "precision": float(prec), "recall": float(rec),
        "auc": float(auc), "loss": float(total_loss / max(n_batches, 1)),
    }


# ======================================================================
# Local training
# ======================================================================

def run_local(seed, client_data, client_test):
    cfg = HD_CONFIG
    device = _detect_device()
    total_epochs = cfg["num_rounds"] * cfg["local_epochs"]

    torch.manual_seed(seed)
    np.random.seed(seed)

    per_client_metrics = {}
    for cid in sorted(client_data.keys()):
        X_train, y_train = client_data[cid]
        X_test, y_test = client_test[cid]

        model = HealthcareMLP(
            input_dim=cfg["input_dim"], num_classes=cfg["num_classes"],
        ).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg["learning_rate"])
        criterion = nn.CrossEntropyLoss()

        train_ds = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
        train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True)

        model.train()
        for _ in range(total_epochs):
            for X_b, y_b in train_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                optimizer.zero_grad()
                out = model(X_b)
                loss = criterion(out, y_b)
                loss.backward()
                optimizer.step()

        metrics = _evaluate_model(
            model, X_test, y_test, cfg["batch_size"], device, cfg["num_classes"],
        )
        per_client_metrics[str(cid)] = metrics

    avg = {}
    for mk in ["accuracy", "f1", "precision", "recall", "auc", "loss"]:
        vals = [per_client_metrics[c][mk] for c in per_client_metrics]
        avg[mk] = float(np.mean(vals))

    return {"per_client": per_client_metrics, "average": avg}


# ======================================================================
# Centralized training
# ======================================================================

def run_centralized(seed, client_data, client_test):
    cfg = HD_CONFIG
    device = _detect_device()
    total_epochs = cfg["num_rounds"] * cfg["local_epochs"]

    torch.manual_seed(seed)
    np.random.seed(seed)

    X_train = np.concatenate([client_data[c][0] for c in sorted(client_data.keys())])
    y_train = np.concatenate([client_data[c][1] for c in sorted(client_data.keys())])
    X_test = np.concatenate([client_test[c][0] for c in sorted(client_test.keys())])
    y_test = np.concatenate([client_test[c][1] for c in sorted(client_test.keys())])

    model = HealthcareMLP(
        input_dim=cfg["input_dim"], num_classes=cfg["num_classes"],
    ).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg["learning_rate"])
    criterion = nn.CrossEntropyLoss()

    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True)

    model.train()
    for _ in range(total_epochs):
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            out = model(X_b)
            loss = criterion(out, y_b)
            loss.backward()
            optimizer.step()

    metrics = _evaluate_model(
        model, X_test, y_test, cfg["batch_size"], device, cfg["num_classes"],
    )
    return {"pooled_metrics": metrics, "train_samples": len(y_train), "test_samples": len(y_test)}


# ======================================================================
# Federated training (FedAvg, Ditto, HPFL)
# ======================================================================

def run_federated(seed, algorithm, client_data, client_test):
    cfg = HD_CONFIG

    trainer = FederatedTrainer(
        num_clients=cfg["num_clients"],
        algorithm=algorithm,
        local_epochs=cfg["local_epochs"],
        batch_size=cfg["batch_size"],
        learning_rate=cfg["learning_rate"],
        mu=cfg["mu"],
        seed=seed,
        external_data=client_data,
        external_test_data=client_test,
        input_dim=cfg["input_dim"],
        num_classes=cfg["num_classes"],
    )

    history = []
    best_acc = 0.0

    for r in range(cfg["num_rounds"]):
        result = trainer.train_round(r)
        metrics = {
            "round": r + 1,
            "accuracy": result.global_acc,
            "loss": result.global_loss,
            "f1": result.global_f1,
            "precision": result.global_precision,
            "recall": result.global_recall,
            "auc": result.global_auc,
        }
        history.append(metrics)
        if result.global_acc > best_acc:
            best_acc = result.global_acc

    # Per-client accuracy (for Jain fairness)
    per_client_acc = {}
    for cr in result.client_results:
        per_client_acc[str(cr.client_id)] = cr.train_acc

    final = history[-1] if history else {}

    return {
        "final_metrics": {
            "accuracy": final.get("accuracy", 0.0),
            "f1": final.get("f1", 0.0),
            "precision": final.get("precision", 0.0),
            "recall": final.get("recall", 0.0),
            "auc": final.get("auc", 0.5),
            "loss": final.get("loss", 0.0),
        },
        "best_accuracy": best_acc,
        "history": history,
        "per_client_acc": per_client_acc,
        "actual_rounds": len(history),
    }


# ======================================================================
# Single experiment dispatcher
# ======================================================================

def run_single_experiment(mode, seed):
    start = time.time()
    client_data, client_test, metadata = load_dataset(seed)

    if mode == "local":
        result = run_local(seed, client_data, client_test)
        primary_acc = result["average"]["accuracy"]
        primary_f1 = result["average"]["f1"]
        primary_auc = result["average"]["auc"]
    elif mode == "centralized":
        result = run_centralized(seed, client_data, client_test)
        primary_acc = result["pooled_metrics"]["accuracy"]
        primary_f1 = result["pooled_metrics"]["f1"]
        primary_auc = result["pooled_metrics"]["auc"]
    else:
        # FL modes: FedAvg, Ditto, HPFL
        result = run_federated(seed, mode, client_data, client_test)
        primary_acc = result["final_metrics"]["accuracy"]
        primary_f1 = result["final_metrics"]["f1"]
        primary_auc = result["final_metrics"]["auc"]

    elapsed = time.time() - start

    return {
        "mode": mode,
        "seed": seed,
        "primary_accuracy": primary_acc,
        "primary_f1": primary_f1,
        "primary_auc": primary_auc,
        "detail": result,
        "runtime_seconds": round(elapsed, 1),
    }


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="FL-EHDS Heart Disease Extended (10 seeds)")
    parser.add_argument("--fresh", action="store_true",
                        help="Delete existing checkpoint and start fresh")
    args = parser.parse_args()

    global _log_file
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _log_file = open(OUTPUT_DIR / LOG_FILE, "a")

    # Build experiment list
    experiments = []
    for mode in MODES:
        for seed in SEEDS:
            key = "HD_{}_s{}".format(mode, seed)
            experiments.append({"key": key, "mode": mode, "seed": seed})

    total_exps = len(experiments)

    # Handle --fresh
    if args.fresh:
        for f in [CHECKPOINT_FILE, CHECKPOINT_FILE + ".bak"]:
            p = OUTPUT_DIR / f
            if p.exists():
                p.unlink()
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
                "modes": MODES,
                "seeds": SEEDS,
                "dataset": "Heart Disease UCI",
                "config": HD_CONFIG,
                "start_time": datetime.now().isoformat(),
                "last_save": None,
            },
        }

    # Signal handler
    _interrupted = [False]

    def _signal_handler(signum, frame):
        if _interrupted[0]:
            sys.exit(1)
        _interrupted[0] = True
        done = len(checkpoint_data.get("completed", {}))
        log("\nINTERRUPT -- saving checkpoint ({}/{})...".format(done, total_exps))
        save_checkpoint(checkpoint_data)
        log("Checkpoint saved. Resume: python -m benchmarks.run_hd_extended")
        sys.exit(0)

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # Header
    log("\n" + "=" * 66)
    log("  FL-EHDS Heart Disease Extended (10 seeds)")
    log("  {} experiments = {} modes x {} seeds".format(
        total_exps, len(MODES), len(SEEDS)))
    log("=" * 66)
    log("  Device:   {}".format(_detect_device()))
    log("  Modes:    {}".format(MODES))
    log("  Seeds:    {}".format(SEEDS))
    log("  Config:   {} rounds x {} local_epochs, lr={}, bs={}".format(
        HD_CONFIG["num_rounds"], HD_CONFIG["local_epochs"],
        HD_CONFIG["learning_rate"], HD_CONFIG["batch_size"]))
    log("  Output:   {}".format(OUTPUT_DIR / CHECKPOINT_FILE))
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

        mode = exp["mode"]
        seed = exp["seed"]

        log("[{}/{}] {} s{} ...".format(done_count + 1, total_exps, mode, seed))

        try:
            result = run_single_experiment(mode, seed)
            completed[key] = result
            done_count += 1

            acc = result["primary_accuracy"] * 100
            f1 = result["primary_f1"] * 100
            rt = result["runtime_seconds"]
            log("  -> acc={:.1f}% f1={:.1f}% {:.0f}s".format(acc, f1, rt))

            save_checkpoint(checkpoint_data)

        except Exception as e:
            log("  ERROR: {}".format(e))
            traceback.print_exc()
            completed[key] = {"mode": mode, "seed": seed, "error": str(e)}
            save_checkpoint(checkpoint_data)
            _cleanup_gpu()

    # Finalize
    checkpoint_data["metadata"]["end_time"] = datetime.now().isoformat()
    checkpoint_data["metadata"]["total_elapsed"] = time.time() - global_start
    save_checkpoint(checkpoint_data)

    elapsed = time.time() - global_start

    # ======================================================================
    # Summary table
    # ======================================================================
    log("\n" + "=" * 66)
    log("  COMPLETED: {}/{}  ({})".format(
        done_count, total_exps, timedelta(seconds=int(elapsed))))
    log("=" * 66)

    log("\n  Heart Disease UCI — 10-Seed Results")
    log("  {:<14s} | {:>14s} | {:>10s} | {:>10s} | {:>8s}".format(
        "Mode", "Accuracy", "F1", "AUC", "Gap"))
    log("  " + "-" * 66)

    centralized_mean = None
    for mode in MODES:
        accs, f1s, aucs = [], [], []
        for s in SEEDS:
            k = "HD_{}_s{}".format(mode, s)
            if k in completed and "error" not in completed[k]:
                accs.append(completed[k]["primary_accuracy"] * 100)
                f1s.append(completed[k]["primary_f1"])
                aucs.append(completed[k]["primary_auc"])
        if accs:
            m_acc, s_acc = np.mean(accs), np.std(accs)
            m_f1 = np.mean(f1s)
            m_auc = np.mean(aucs)

            if mode == "centralized":
                centralized_mean = m_acc
                gap_str = "ref"
            elif centralized_mean is not None:
                gap = centralized_mean - m_acc
                gap_str = "{:+.1f}pp".format(-gap)
            else:
                gap_str = "---"

            log("  {:<14s} | {:>5.1f} +/- {:>4.1f} | {:>10.3f} | {:>10.3f} | {:>8s}".format(
                mode, m_acc, s_acc, m_f1, m_auc, gap_str))
        else:
            log("  {:<14s} |       --       |     --     |     --     |    --".format(mode))

    log("  " + "-" * 66)

    # Per-seed detail
    log("\n  Per-seed accuracy:")
    header = "  {:<14s}".format("Mode")
    for s in SEEDS:
        header += " | s{:<5d}".format(s)
    log(header)
    log("  " + "-" * (16 + 9 * len(SEEDS)))

    for mode in MODES:
        row = "  {:<14s}".format(mode)
        for s in SEEDS:
            k = "HD_{}_s{}".format(mode, s)
            if k in completed and "error" not in completed[k]:
                row += " | {:>5.1f}".format(completed[k]["primary_accuracy"] * 100)
            else:
                row += " |   -- "
        log(row)

    log("\n  Done!")

    if _log_file:
        _log_file.close()


if __name__ == "__main__":
    main()

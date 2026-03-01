#!/usr/bin/env python3
"""
FL-EHDS Experiment — Centralized vs Federated Extended.

Compares Local-Only, Centralized (pooled data), and Federated (FedAvg)
across 5 tabular healthcare datasets.

Key research question: How close does federated learning get to the
centralized upper bound? What is the "privacy gap"?

Design:
  - Datasets: Breast Cancer, Cardiovascular, Heart Disease, Diabetes, PTB-XL
  - Modes: local, centralized, federated
  - Seeds: 42, 123, 456
  - Total: 5 datasets x 3 modes x 3 seeds = 45 experiments (~1.5h on M3)

Usage:
    cd fl-ehds-framework
    python -m benchmarks.run_centralized_vs_federated [--quick] [--fresh]

Output:
    benchmarks/paper_results_tabular/checkpoint_centralized_vs_fed.json
    benchmarks/paper_results_tabular/centralized_vs_federated.png

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
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from terminal.fl_trainer import FederatedTrainer, HealthcareMLP, _detect_device

# ======================================================================
# Constants
# ======================================================================

OUTPUT_DIR = FRAMEWORK_DIR / "benchmarks" / "paper_results_tabular"
CHECKPOINT_FILE = "checkpoint_centralized_vs_fed.json"
LOG_FILE = "experiment_centralized_vs_fed.log"

MODES = ["local", "centralized", "federated"]
SEEDS = [42, 123, 456]

DATASET_CONFIGS = {
    "BC": {
        "name": "Breast Cancer",
        "loader": "breast_cancer",
        "input_dim": 30,
        "num_classes": 2,
        "num_clients": 4,
        "learning_rate": 0.01,
        "batch_size": 64,
        "num_rounds": 30,
        "local_epochs": 3,
        "mu": 0.1,
    },
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
    "HD": {
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
    },
    "DM": {
        "name": "Diabetes",
        "loader": "diabetes",
        "input_dim": 22,
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
    fd, tmp = tempfile.mkstemp(dir=str(OUTPUT_DIR), prefix=".cvf_", suffix=".tmp")
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

def load_dataset(ds_key, seed):
    cfg = DATASET_CONFIGS[ds_key]
    if cfg["loader"] == "breast_cancer":
        from data.breast_cancer_loader import load_breast_cancer_data
        return load_breast_cancer_data(
            num_clients=cfg["num_clients"], seed=seed, is_iid=False,
        )
    elif cfg["loader"] == "cardiovascular":
        from data.cardiovascular_loader import load_cardiovascular_data
        return load_cardiovascular_data(
            num_clients=cfg["num_clients"], seed=seed, is_iid=False,
        )
    elif cfg["loader"] == "heart_disease":
        from data.heart_disease_loader import load_heart_disease_data
        return load_heart_disease_data(
            num_clients=cfg["num_clients"], partition_by_hospital=True, seed=seed,
        )
    elif cfg["loader"] == "diabetes":
        from data.diabetes_loader import load_diabetes_data
        return load_diabetes_data(
            num_clients=cfg["num_clients"], partition_by_hospital=True, seed=seed,
        )
    elif cfg["loader"] == "ptbxl":
        from data.ptbxl_loader import load_ptbxl_data
        return load_ptbxl_data(
            num_clients=cfg["num_clients"], seed=seed,
            partition_by_site=True, min_site_samples=50,
        )
    else:
        raise ValueError("Unknown loader: {}".format(cfg["loader"]))


# ======================================================================
# Evaluation helper
# ======================================================================

def _evaluate_model(model, X, y, batch_size, device, num_classes):
    """Evaluate a model on given data. Returns dict with accuracy, f1, auc."""
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    model.eval()
    dataset = TensorDataset(torch.FloatTensor(X), torch.LongTensor(y))
    loader = DataLoader(dataset, batch_size=batch_size)

    all_preds = []
    all_labels = []
    all_probs = []
    total_loss = 0.0
    n_batches = 0
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
            auc = roc_auc_score(
                all_labels, all_probs, multi_class="ovr", average="macro",
            )
    except Exception:
        auc = 0.5

    return {
        "accuracy": acc,
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "auc": auc,
        "loss": total_loss / max(n_batches, 1),
    }


# ======================================================================
# Local training: one model per client, average metrics
# ======================================================================

def run_local(ds_key, seed, client_data, client_test, quick=False):
    """Train one model per client, evaluate on client test data, average."""
    cfg = DATASET_CONFIGS[ds_key]
    device = _detect_device()
    total_epochs = cfg["num_rounds"] * cfg["local_epochs"]
    if quick:
        total_epochs = 5 * cfg["local_epochs"]
    lr = cfg["learning_rate"]
    batch_size = cfg["batch_size"]
    num_classes = cfg["num_classes"]
    input_dim = cfg["input_dim"]

    torch.manual_seed(seed)
    np.random.seed(seed)

    per_client_metrics = {}

    for cid in sorted(client_data.keys()):
        X_train, y_train = client_data[cid]
        X_test, y_test = client_test[cid]

        model = HealthcareMLP(
            input_dim=input_dim, num_classes=num_classes,
        ).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        train_dataset = TensorDataset(
            torch.FloatTensor(X_train), torch.LongTensor(y_train),
        )
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
        )

        model.train()
        for epoch in range(total_epochs):
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                output = model(X_batch)
                loss = criterion(output, y_batch)
                loss.backward()
                optimizer.step()

        metrics = _evaluate_model(
            model, X_test, y_test, batch_size, device, num_classes,
        )
        per_client_metrics[str(cid)] = metrics

    # Average across clients
    avg_metrics = {}
    metric_keys = ["accuracy", "f1", "precision", "recall", "auc", "loss"]
    for mk in metric_keys:
        vals = [per_client_metrics[cid][mk] for cid in per_client_metrics]
        avg_metrics[mk] = float(np.mean(vals))

    return {
        "per_client": per_client_metrics,
        "average": avg_metrics,
    }


# ======================================================================
# Centralized training: pool all data, single model
# ======================================================================

def run_centralized(ds_key, seed, client_data, client_test, quick=False):
    """Pool all client data, train one model, evaluate on pooled test."""
    cfg = DATASET_CONFIGS[ds_key]
    device = _detect_device()
    total_epochs = cfg["num_rounds"] * cfg["local_epochs"]
    if quick:
        total_epochs = 5 * cfg["local_epochs"]
    lr = cfg["learning_rate"]
    batch_size = cfg["batch_size"]
    num_classes = cfg["num_classes"]
    input_dim = cfg["input_dim"]

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Pool all training data
    all_X_train = []
    all_y_train = []
    for cid in sorted(client_data.keys()):
        X_c, y_c = client_data[cid]
        all_X_train.append(X_c)
        all_y_train.append(y_c)
    X_train = np.concatenate(all_X_train, axis=0)
    y_train = np.concatenate(all_y_train, axis=0)

    # Pool all test data
    all_X_test = []
    all_y_test = []
    for cid in sorted(client_test.keys()):
        X_c, y_c = client_test[cid]
        all_X_test.append(X_c)
        all_y_test.append(y_c)
    X_test = np.concatenate(all_X_test, axis=0)
    y_test = np.concatenate(all_y_test, axis=0)

    model = HealthcareMLP(
        input_dim=input_dim, num_classes=num_classes,
    ).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_dataset = TensorDataset(
        torch.FloatTensor(X_train), torch.LongTensor(y_train),
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
    )

    model.train()
    for epoch in range(total_epochs):
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

    metrics = _evaluate_model(
        model, X_test, y_test, batch_size, device, num_classes,
    )

    return {
        "pooled_metrics": metrics,
        "train_samples": int(len(y_train)),
        "test_samples": int(len(y_test)),
    }


# ======================================================================
# Federated training: FedAvg
# ======================================================================

def run_federated(ds_key, seed, client_data, client_test, quick=False):
    """Standard federated learning with FedAvg."""
    cfg = DATASET_CONFIGS[ds_key]
    num_rounds = 5 if quick else cfg["num_rounds"]

    trainer = FederatedTrainer(
        num_clients=cfg["num_clients"],
        algorithm="FedAvg",
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

    for r in range(num_rounds):
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
        "actual_rounds": len(history),
    }


# ======================================================================
# Single experiment dispatcher
# ======================================================================

def run_single_experiment(ds_key, mode, seed, quick=False):
    """Run one experiment: dataset + mode + seed."""
    start = time.time()

    client_data, client_test, metadata = load_dataset(ds_key, seed)

    if mode == "local":
        result = run_local(ds_key, seed, client_data, client_test, quick=quick)
        # Unify: extract the primary accuracy for summary
        primary_acc = result["average"]["accuracy"]
        primary_f1 = result["average"]["f1"]
        primary_auc = result["average"]["auc"]
    elif mode == "centralized":
        result = run_centralized(ds_key, seed, client_data, client_test, quick=quick)
        primary_acc = result["pooled_metrics"]["accuracy"]
        primary_f1 = result["pooled_metrics"]["f1"]
        primary_auc = result["pooled_metrics"]["auc"]
    elif mode == "federated":
        result = run_federated(ds_key, seed, client_data, client_test, quick=quick)
        primary_acc = result["final_metrics"]["accuracy"]
        primary_f1 = result["final_metrics"]["f1"]
        primary_auc = result["final_metrics"]["auc"]
    else:
        raise ValueError("Unknown mode: {}".format(mode))

    elapsed = time.time() - start

    out = {
        "dataset": ds_key,
        "mode": mode,
        "seed": seed,
        "primary_accuracy": primary_acc,
        "primary_f1": primary_f1,
        "primary_auc": primary_auc,
        "detail": result,
        "runtime_seconds": round(elapsed, 1),
    }

    _cleanup_gpu()
    return out


# ======================================================================
# Figure generation
# ======================================================================

def generate_figure(completed, ds_keys, modes, seeds):
    """Grouped bar chart: one group per dataset, 3 bars (local, centralized, federated)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log("matplotlib not available -- skipping figure")
        return

    colors = {
        "local": "#d9534f",
        "centralized": "#5bc0de",
        "federated": "#5cb85c",
    }
    labels = {
        "local": "Local-Only",
        "centralized": "Centralized (pooled)",
        "federated": "Federated (FedAvg)",
    }

    n_ds = len(ds_keys)
    n_modes = len(modes)
    x = np.arange(n_ds)
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(8, 2.5 * n_ds), 5.5))

    for i, mode in enumerate(modes):
        means = []
        stds = []
        for ds_key in ds_keys:
            accs = []
            for s in seeds:
                key = "{}_{}_s{}".format(ds_key, mode, s)
                if key in completed and "error" not in completed[key]:
                    accs.append(completed[key]["primary_accuracy"] * 100)
            if accs:
                means.append(np.mean(accs))
                stds.append(np.std(accs))
            else:
                means.append(0.0)
                stds.append(0.0)

        offset = (i - (n_modes - 1) / 2.0) * width
        ax.bar(
            x + offset, means, width, yerr=stds,
            label=labels[mode], color=colors[mode],
            capsize=3, alpha=0.85, edgecolor="white", linewidth=0.5,
        )

    ax.set_xlabel("Dataset", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title(
        "Local vs Centralized vs Federated: Accuracy Comparison",
        fontsize=13, fontweight="bold",
    )
    ax.set_xticks(x)
    ds_labels = [DATASET_CONFIGS[dk]["name"] for dk in ds_keys]
    ax.set_xticklabels(ds_labels, fontsize=10)
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(axis="y", alpha=0.3)

    # Set y-axis limits with some padding
    all_accs = []
    for ds_key in ds_keys:
        for mode in modes:
            for s in seeds:
                key = "{}_{}_s{}".format(ds_key, mode, s)
                if key in completed and "error" not in completed[key]:
                    all_accs.append(completed[key]["primary_accuracy"] * 100)
    if all_accs:
        y_min = max(0, min(all_accs) - 10)
        y_max = min(100, max(all_accs) + 8)
        ax.set_ylim(y_min, y_max)

    fig.tight_layout()

    fig_path = OUTPUT_DIR / "centralized_vs_federated.png"
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.savefig(str(fig_path).replace(".png", ".pdf"), bbox_inches="tight")
    log("Figure saved: {}".format(fig_path))
    log("Figure saved: {}".format(str(fig_path).replace(".png", ".pdf")))
    plt.close()


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="FL-EHDS Centralized vs Federated Comparison")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 1 seed, 2 datasets (BC, CV)")
    parser.add_argument("--fresh", action="store_true",
                        help="Delete existing checkpoint and start fresh")
    args = parser.parse_args()

    global _log_file
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _log_file = open(OUTPUT_DIR / LOG_FILE, "a")

    # Experiment grid
    seeds = [42] if args.quick else SEEDS
    ds_keys = ["BC", "CV"] if args.quick else list(DATASET_CONFIGS.keys())
    modes = MODES

    # Build experiment list
    experiments = []
    for ds_key in ds_keys:
        for mode in modes:
            for seed in seeds:
                key = "{}_{}_s{}".format(ds_key, mode, seed)
                experiments.append({
                    "key": key, "ds_key": ds_key, "mode": mode, "seed": seed,
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
                "modes": modes,
                "datasets": ds_keys,
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
        log("Checkpoint saved. Resume: python -m benchmarks.run_centralized_vs_federated")
        sys.exit(0)

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # Header
    mode_label = "QUICK" if args.quick else "FULL"
    log("\n" + "=" * 66)
    log("  FL-EHDS Centralized vs Federated ({})".format(mode_label))
    log("  {} experiments = {} DS x {} modes x {} seeds".format(
        total_exps, len(ds_keys), len(modes), len(seeds)))
    log("=" * 66)
    log("  Device:   {}".format(_detect_device()))
    log("  Datasets: {}".format(ds_keys))
    log("  Modes:    {}".format(modes))
    log("  Seeds:    {}".format(seeds))
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

        ds_key = exp["ds_key"]
        mode = exp["mode"]
        seed = exp["seed"]

        log("[{}/{}] {} {} s{} ...".format(
            done_count + 1, total_exps, ds_key, mode, seed))

        try:
            result = run_single_experiment(
                ds_key, mode, seed, quick=args.quick)
            completed[key] = result
            done_count += 1

            acc = result["primary_accuracy"] * 100
            f1 = result["primary_f1"] * 100
            rt = result["runtime_seconds"]
            log("  -> acc={:.1f}% f1={:.1f}% {:.0f}s".format(acc, f1, rt))

            # Save checkpoint after every experiment
            save_checkpoint(checkpoint_data)

        except Exception as e:
            log("  ERROR: {}".format(e))
            traceback.print_exc()
            completed[key] = {
                "dataset": ds_key, "mode": mode,
                "seed": seed, "error": str(e),
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
    log("\n  Centralized vs Federated Results")
    log("  {:>4s} | {:>12s} | {:>12s} | {:>12s}".format(
        "DS", "Local", "Centralized", "Federated"))
    log("  " + "-" * 56)

    for ds_key in ds_keys:
        row_vals = []
        for mode in modes:
            accs = []
            for s in seeds:
                k = "{}_{}_s{}".format(ds_key, mode, s)
                if k in completed and "error" not in completed[k]:
                    accs.append(completed[k]["primary_accuracy"] * 100)
            if accs:
                row_vals.append("{:5.1f}+-{:.1f}".format(
                    np.mean(accs), np.std(accs)))
            else:
                row_vals.append("   --   ")
        log("  {:>4s} | {:>12s} | {:>12s} | {:>12s}".format(
            ds_key, row_vals[0], row_vals[1], row_vals[2]))

    log("  " + "-" * 56)

    # Print privacy gap: centralized - federated
    log("\n  Privacy Gap (Centralized - Federated)")
    log("  {:>4s} | {:>12s}".format("DS", "Gap (pp)"))
    log("  " + "-" * 22)

    for ds_key in ds_keys:
        cent_accs = []
        fed_accs = []
        for s in seeds:
            kc = "{}_{}_s{}".format(ds_key, "centralized", s)
            kf = "{}_{}_s{}".format(ds_key, "federated", s)
            if (kc in completed and "error" not in completed[kc]
                    and kf in completed and "error" not in completed[kf]):
                cent_accs.append(completed[kc]["primary_accuracy"] * 100)
                fed_accs.append(completed[kf]["primary_accuracy"] * 100)
        if cent_accs and fed_accs:
            gap = np.mean(cent_accs) - np.mean(fed_accs)
            log("  {:>4s} | {:>+10.1f}pp".format(ds_key, gap))
        else:
            log("  {:>4s} | {:>12s}".format(ds_key, "--"))

    log("  " + "-" * 22)

    # ======================================================================
    # Generate figure
    # ======================================================================
    generate_figure(completed, ds_keys, modes, seeds)

    if _log_file:
        _log_file.close()


if __name__ == "__main__":
    main()

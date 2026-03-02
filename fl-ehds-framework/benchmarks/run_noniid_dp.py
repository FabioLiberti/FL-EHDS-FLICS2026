#!/usr/bin/env python3
"""
FL-EHDS Experiment — Non-IID + DP Combined (Tabular).

Tests the interaction between data heterogeneity and differential privacy
on tabular datasets. Answers: "Does DP worsen performance on already-
heterogeneous data? Is the interaction additive or multiplicative?"

This is a key reviewer question for EHDS deployment: cross-border
hospitals have both heterogeneous patient populations (non-IID) AND
privacy requirements (DP). Neither effect has been tested in combination.

Design:
  - Datasets: PTB-XL (5 clients, 5-class, partition_by_site)
              Cardiovascular (5 clients, binary)
  - Algorithms: FedAvg, Ditto, HPFL
  - Non-IID levels: alpha = 0.1 (extreme), 0.5 (moderate), 1.0 (mild)
    NOTE: PTB-XL uses partition_by_site, so alpha controls only CV
          For PTB-XL we test num_clients = 3, 5, 10 as proxy for heterogeneity
  - DP levels: No-DP, eps=1, eps=10
  - Seeds: 42, 123, 456
  - Total:
    CV:    3 algos x 3 alphas x 3 DP x 3 seeds = 81 experiments
    PTB-XL: 3 algos x 3 K-values x 3 DP x 3 seeds = 81 experiments
    Grand total: 162 experiments (~1.5-2h on Air M3)

Usage:
    cd fl-ehds-framework
    python -m benchmarks.run_noniid_dp [--quick] [--fresh]

Output:
    benchmarks/paper_results_tabular/checkpoint_noniid_dp.json

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
from typing import Dict, List, Any

import numpy as np

FRAMEWORK_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(FRAMEWORK_DIR))

import torch

from terminal.fl_trainer import FederatedTrainer, _detect_device
from data.ptbxl_loader import load_ptbxl_data
from data.cardiovascular_loader import load_cardiovascular_data

# ======================================================================
# Constants
# ======================================================================

OUTPUT_DIR = FRAMEWORK_DIR / "benchmarks" / "paper_results_tabular"
CHECKPOINT_FILE = "checkpoint_noniid_dp.json"
LOG_FILE = "experiment_noniid_dp.log"

ALGORITHMS = ["FedAvg", "Ditto", "HPFL"]
SEEDS = [42, 123, 456]

PTB_XL_CLASS_NAMES = {0: "NORM", 1: "MI", 2: "STTC", 3: "CD", 4: "HYP"}

# ======================================================================
# Experiment configurations
# ======================================================================

# Cardiovascular: vary alpha (data heterogeneity)
CV_ALPHAS = [0.1, 0.5, 1.0]

# PTB-XL: vary K (number of clients) as proxy for heterogeneity
# With partition_by_site, more clients = more heterogeneous splits
PX_K_VALUES = [3, 5, 10]

# DP levels
DP_LEVELS = [
    {"label": "No-DP", "dp_enabled": False, "dp_epsilon": 0},
    {"label": "DP-eps1", "dp_enabled": True, "dp_epsilon": 1.0},
    {"label": "DP-eps10", "dp_enabled": True, "dp_epsilon": 10.0},
]

# Training configs
CV_TRAINING = dict(
    learning_rate=0.01,
    batch_size=64,
    num_rounds=25,
    local_epochs=3,
    mu=0.1,
)

PX_TRAINING = dict(
    learning_rate=0.005,
    batch_size=64,
    num_rounds=30,
    local_epochs=3,
    mu=0.1,
)

# ======================================================================
# Logging
# ======================================================================

_log_file = None
_shutdown = False


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


def _handle_signal(signum, frame):
    global _shutdown
    _shutdown = True
    log("Signal {} received — finishing current experiment then saving...".format(signum))


# ======================================================================
# Checkpoint (atomic write with backup)
# ======================================================================

def save_checkpoint(data):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / CHECKPOINT_FILE
    bak = OUTPUT_DIR / (CHECKPOINT_FILE + ".bak")
    data["metadata"]["last_save"] = datetime.now().isoformat()
    fd, tmp = tempfile.mkstemp(dir=str(OUTPUT_DIR), prefix=".niiddp_", suffix=".tmp")
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
# Shared evaluation helpers
# ======================================================================

def _evaluate_per_client(trainer):
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
                out = model(X_t[i:i + 64])
                correct += (out.argmax(1) == y_t[i:i + 64]).sum().item()
                total += len(y_t[i:i + 64])
            per_client[str(cid)] = correct / total if total > 0 else 0.0

    if is_hpfl:
        for n, p in model.named_parameters():
            if n in trainer._hpfl_classifier_names:
                p.data.copy_(saved_cls[n])
    return per_client


def _collect_predictions(trainer):
    model = trainer.global_model
    model.eval()
    all_preds = []
    all_labels = []

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
            out = model(X_t)
            preds = out.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(y.tolist() if hasattr(y, "tolist") else list(y))

    if is_hpfl:
        for n, p in model.named_parameters():
            if n in trainer._hpfl_classifier_names:
                p.data.copy_(saved_cls[n])

    return np.array(all_labels), np.array(all_preds)


def _compute_per_class_metrics(y_true, y_pred, num_classes, class_names):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1

    per_class = {}
    for c in range(num_classes):
        cls_name = class_names.get(c, "Class_{}".format(c))
        tp = cm[c, c]
        fn = cm[c, :].sum() - tp
        fp = cm[:, c].sum() - tp
        support = cm[c, :].sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        per_class[cls_name] = {
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1": round(float(f1), 4),
            "support": int(support),
        }

    per_class["macro_avg"] = {
        "precision": round(float(np.mean([v["precision"] for k, v in per_class.items() if k != "macro_avg"])), 4),
        "recall": round(float(np.mean([v["recall"] for k, v in per_class.items() if k != "macro_avg"])), 4),
        "f1": round(float(np.mean([v["f1"] for k, v in per_class.items() if k != "macro_avg"])), 4),
    }

    return per_class, cm.tolist()


def _compute_dei(per_class_metrics, num_classes, class_names):
    recalls = []
    for c in range(num_classes):
        cls_name = class_names.get(c, "Class_{}".format(c))
        if cls_name in per_class_metrics:
            recalls.append(per_class_metrics[cls_name]["recall"])
    if not recalls:
        return 0.0
    recalls = np.array(recalls)
    min_recall = float(np.min(recalls))
    mean_recall = float(np.mean(recalls))
    if mean_recall == 0:
        return 0.0
    cv = float(np.std(recalls) / mean_recall)
    return round(min_recall * (1.0 - cv), 4)


def _compute_jain_index(per_client_acc):
    accs = list(per_client_acc.values())
    if not accs:
        return 0.0
    accs = np.array(accs)
    n = len(accs)
    if n == 0 or np.sum(accs ** 2) == 0:
        return 0.0
    return float((np.sum(accs) ** 2) / (n * np.sum(accs ** 2)))


def format_time(seconds):
    return str(timedelta(seconds=int(seconds)))


# ======================================================================
# Single experiment
# ======================================================================

def run_single_experiment(dataset, algo, heterogeneity_param, dp_cfg, seed, quick=False):
    """
    Run one Non-IID + DP experiment.

    Args:
        dataset: "PTB_XL" or "Cardiovascular"
        algo: algorithm name
        heterogeneity_param: alpha (for CV) or K (for PTB-XL)
        dp_cfg: dict with dp_enabled, dp_epsilon, label
        seed: random seed
        quick: if True, use 5 rounds
    """
    start = time.time()

    if dataset == "Cardiovascular":
        alpha = heterogeneity_param
        num_clients = 5
        tcfg = CV_TRAINING
        num_classes = 2
        input_dim = 11
        class_names = {0: "Healthy", 1: "Disease"}

        num_rounds = 5 if quick else tcfg["num_rounds"]

        client_data, client_test, metadata = load_cardiovascular_data(
            num_clients=num_clients, seed=seed,
            is_iid=False, alpha=alpha,
        )

    elif dataset == "PTB_XL":
        num_clients = heterogeneity_param  # K value
        tcfg = PX_TRAINING
        num_classes = 5
        input_dim = 9
        class_names = PTB_XL_CLASS_NAMES
        alpha = None  # not used for PTB-XL

        num_rounds = 5 if quick else tcfg["num_rounds"]

        client_data, client_test, metadata = load_ptbxl_data(
            num_clients=num_clients, seed=seed,
            partition_by_site=True, min_site_samples=50,
        )
    else:
        raise ValueError("Unknown dataset: {}".format(dataset))

    trainer_kwargs = dict(
        num_clients=num_clients,
        algorithm=algo,
        local_epochs=tcfg["local_epochs"],
        batch_size=tcfg["batch_size"],
        learning_rate=tcfg["learning_rate"],
        mu=tcfg["mu"],
        seed=seed,
        external_data=client_data,
        external_test_data=client_test,
        input_dim=input_dim,
        num_classes=num_classes,
    )

    if dp_cfg["dp_enabled"]:
        trainer_kwargs["dp_enabled"] = True
        trainer_kwargs["dp_epsilon"] = dp_cfg["dp_epsilon"]
        trainer_kwargs["dp_clip_norm"] = 1.0

    # Pre-check: ensure no client has 0 training samples
    for cid in range(num_clients):
        X_c, y_c = client_data[cid]
        if len(y_c) == 0:
            raise ValueError(
                "Client {} has 0 training samples (alpha={}, seed={}) — skipping".format(
                    cid, alpha, seed))

    trainer = FederatedTrainer(**trainer_kwargs)

    history = []
    best_acc = 0.0

    for r in range(num_rounds):
        result = trainer.train_round(r)
        history.append({
            "round": r + 1,
            "accuracy": result.global_acc,
            "loss": result.global_loss,
            "f1": result.global_f1,
        })
        if result.global_acc > best_acc:
            best_acc = result.global_acc

    # Final evaluation
    per_client_acc = _evaluate_per_client(trainer)
    jain = _compute_jain_index(per_client_acc)
    final_acc = float(np.mean(list(per_client_acc.values())))

    y_true, y_pred = _collect_predictions(trainer)
    per_class, cm = _compute_per_class_metrics(y_true, y_pred, num_classes, class_names)
    dei = _compute_dei(per_class, num_classes, class_names)

    elapsed = time.time() - start

    return {
        "dataset": dataset,
        "algorithm": algo,
        "heterogeneity_param": heterogeneity_param,
        "heterogeneity_type": "alpha" if dataset == "Cardiovascular" else "K",
        "dp_level": dp_cfg["label"],
        "dp_epsilon": dp_cfg["dp_epsilon"],
        "seed": seed,
        "final_accuracy": round(final_acc, 4),
        "best_accuracy": round(best_acc, 4),
        "per_client_accuracy": per_client_acc,
        "jain_index": round(jain, 4),
        "dei": dei,
        "per_class_metrics": per_class,
        "confusion_matrix": cm,
        "convergence_trajectory": [h["accuracy"] for h in history],
        "time_seconds": round(elapsed, 1),
    }


# ======================================================================
# Main
# ======================================================================

def main():
    global _log_file

    parser = argparse.ArgumentParser(
        description="FL-EHDS: Non-IID + DP Combined Experiment")
    parser.add_argument("--quick", action="store_true",
                        help="Quick validation (5 rounds)")
    parser.add_argument("--fresh", action="store_true",
                        help="Delete checkpoint and start fresh")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _log_file = open(str(OUTPUT_DIR / LOG_FILE), "a")

    log("=" * 70)
    log("FL-EHDS: Non-IID + DP Combined Experiment")
    log("Mode: {}".format("QUICK" if args.quick else "FULL"))
    log("Device: {}".format(_detect_device("cpu")))
    log("=" * 70)

    if args.fresh:
        for suffix in ["", ".bak"]:
            p = OUTPUT_DIR / (CHECKPOINT_FILE + suffix)
            if p.exists():
                p.unlink()
                log("Deleted {}".format(CHECKPOINT_FILE + suffix))

    # Load or create checkpoint
    ckpt = load_checkpoint()
    if ckpt is None:
        ckpt = {
            "metadata": {
                "experiment": "Non-IID + DP Combined",
                "description": "Interaction between data heterogeneity and differential privacy",
                "started": datetime.now().isoformat(),
            },
            "results": {},
            "completed": [],
        }

    # Build experiment list
    experiments = []

    # Block 1: Cardiovascular with alpha sweep + DP
    for algo in ALGORITHMS:
        for alpha in CV_ALPHAS:
            for dp_cfg in DP_LEVELS:
                for seed in SEEDS:
                    exp_key = "CV_{}_a{}_{}_s{}".format(algo, alpha, dp_cfg["label"], seed)
                    experiments.append(("Cardiovascular", algo, alpha, dp_cfg, seed, exp_key))

    # Block 2: PTB-XL with K sweep + DP
    for algo in ALGORITHMS:
        for k in PX_K_VALUES:
            for dp_cfg in DP_LEVELS:
                for seed in SEEDS:
                    exp_key = "PX_{}_K{}_{}_s{}".format(algo, k, dp_cfg["label"], seed)
                    experiments.append(("PTB_XL", algo, k, dp_cfg, seed, exp_key))

    total = len(experiments)
    done = len(ckpt["completed"])
    log("Total: {} experiments, {} already done".format(total, done))

    cascade_start = time.time()

    for idx, (dataset, algo, het_param, dp_cfg, seed, exp_key) in enumerate(experiments, 1):
        if _shutdown:
            log("Shutdown requested — saving checkpoint")
            break
        if exp_key in ckpt["completed"]:
            continue

        het_label = "alpha={}".format(het_param) if dataset == "Cardiovascular" else "K={}".format(het_param)
        log("[{}/{}] {} | {} | {} | {} | seed={} ...".format(
            idx, total, dataset, algo, het_label, dp_cfg["label"], seed))

        try:
            result = run_single_experiment(
                dataset, algo, het_param, dp_cfg, seed, quick=args.quick)

            ckpt["results"][exp_key] = result
            ckpt["completed"].append(exp_key)
            save_checkpoint(ckpt)

            log("  -> acc={:.1f}% | Jain={:.3f} | DEI={:.3f} | {:.0f}s".format(
                result["final_accuracy"] * 100,
                result["jain_index"],
                result["dei"],
                result["time_seconds"]))

            _cleanup_gpu()

        except Exception as e:
            log("  ERROR: {}".format(e))
            log("  " + traceback.format_exc().split("\n")[-2])
            _cleanup_gpu()

    total_time = time.time() - cascade_start

    ckpt["metadata"]["total_time"] = format_time(total_time)
    ckpt["metadata"]["completed_count"] = len(ckpt["completed"])
    ckpt["metadata"]["total_count"] = total
    save_checkpoint(ckpt)

    log("=" * 70)
    log("COMPLETE: {}/{} experiments in {}".format(
        len(ckpt["completed"]), total, format_time(total_time)))
    log("Checkpoint: {}".format(CHECKPOINT_FILE))
    log("=" * 70)

    # Print summary table
    log("")
    log("SUMMARY — Cardiovascular (Non-IID alpha + DP):")
    log("{:<8} {:<8} {:<10} {:<8} {:<8}".format("Algo", "Alpha", "DP", "Acc%", "DEI"))
    log("-" * 45)
    for algo in ALGORITHMS:
        for alpha in CV_ALPHAS:
            for dp_cfg in DP_LEVELS:
                keys = ["CV_{}_a{}_{}_s{}".format(algo, alpha, dp_cfg["label"], s) for s in SEEDS]
                accs = [ckpt["results"][k]["final_accuracy"] for k in keys if k in ckpt["results"]]
                deis = [ckpt["results"][k]["dei"] for k in keys if k in ckpt["results"]]
                if accs:
                    log("{:<8} {:<8} {:<10} {:<8.1f} {:<8.3f}".format(
                        algo, alpha, dp_cfg["label"],
                        np.mean(accs) * 100, np.mean(deis)))

    log("")
    log("SUMMARY — PTB-XL (K clients + DP):")
    log("{:<8} {:<8} {:<10} {:<8} {:<8}".format("Algo", "K", "DP", "Acc%", "DEI"))
    log("-" * 45)
    for algo in ALGORITHMS:
        for k in PX_K_VALUES:
            for dp_cfg in DP_LEVELS:
                keys = ["PX_{}_K{}_{}_s{}".format(algo, k, dp_cfg["label"], s) for s in SEEDS]
                accs = [ckpt["results"][k_]["final_accuracy"] for k_ in keys if k_ in ckpt["results"]]
                deis = [ckpt["results"][k_]["dei"] for k_ in keys if k_ in ckpt["results"]]
                if accs:
                    log("{:<8} {:<8} {:<10} {:<8.1f} {:<8.3f}".format(
                        algo, k, dp_cfg["label"],
                        np.mean(accs) * 100, np.mean(deis)))

    if _log_file:
        _log_file.close()


if __name__ == "__main__":
    main()

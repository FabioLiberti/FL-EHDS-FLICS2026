#!/usr/bin/env python3
"""
FL-EHDS Experiment — Deep MLP Convergence Test.

Tests whether algorithm convergence holds on a DEEPER model
(DeepHealthcareMLP instead of HealthcareMLP).

With the shallow HealthcareMLP (~10K params, near-convex loss landscape),
many algorithms collapse to FedAvg-equivalent performance. This experiment
uses DeepHealthcareMLP (~110K params, 4-layer) to create a more non-convex
landscape where server-side strategies may differentiate.

Model: DeepHealthcareMLP — 4 layers [256, 256, 128, 64], ReLU, Dropout 0.3
       ~110K parameters (vs ~10K for HealthcareMLP)

Design:
  - Datasets: Cardiovascular (CV) and PTB-XL (PX)
  - Algorithms: FedAvg, FedProx, Ditto, HPFL, FedLESAM, FedExP, SCAFFOLD
  - Seeds: 42, 123, 456
  - Non-IID: alpha=0.5 for CV, site-based for PTB-XL
  - DP: No-DP
  - Training: lr=0.01, batch=64, 25 rounds, 3 local epochs, mu=0.1
  - Total: 7 algos x 2 datasets x 3 seeds = 42 experiments

Checkpointing:
  - Saves after EVERY experiment (atomic write with backup)
  - Resume-safe: skips already-completed experiments
  - Graceful shutdown on Ctrl-C

Usage:
    cd fl-ehds-framework
    python -m benchmarks.run_deep_mlp [--quick] [--fresh]

Output:
    benchmarks/paper_results_tabular/checkpoint_deep_mlp.json

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
from typing import Dict, List

import numpy as np

FRAMEWORK_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(FRAMEWORK_DIR))

import torch
import torch.nn as nn

from terminal.fl_trainer import FederatedTrainer, _detect_device
from data.cardiovascular_loader import load_cardiovascular_data
from data.ptbxl_loader import load_ptbxl_data


# ======================================================================
# Deep MLP Model (~110K parameters)
# ======================================================================

class DeepHealthcareMLP(nn.Module):
    """
    Deeper MLP for healthcare risk prediction.

    4 layers with [256, 256, 128, 64] hidden dims, ReLU, Dropout 0.3.
    ~110K parameters (vs ~10K for the standard 2-layer HealthcareMLP).

    The deeper architecture creates a more non-convex loss landscape,
    allowing server-side FL strategies to potentially differentiate
    from FedAvg.
    """

    def __init__(self, input_dim: int = 10, hidden_dims: List[int] = [256, 256, 128, 64],
                 num_classes: int = 2, dropout: float = 0.3):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# ======================================================================
# Model patching — replace HealthcareMLP with DeepHealthcareMLP
# ======================================================================

def _patch_model():
    """
    Replace HealthcareMLP with DeepHealthcareMLP in the federated trainer module.
    This ensures the trainer creates the deeper model automatically.
    """
    import terminal.training.federated as fed_module
    fed_module.HealthcareMLP = DeepHealthcareMLP
    log("  Model patched: DeepHealthcareMLP [256, 256, 128, 64] (~110K params)")


def _count_params(input_dim, num_classes):
    """Count parameters for reporting."""
    model = DeepHealthcareMLP(input_dim=input_dim, num_classes=num_classes)
    return sum(p.numel() for p in model.parameters())


# ======================================================================
# Constants
# ======================================================================

OUTPUT_DIR = FRAMEWORK_DIR / "benchmarks" / "paper_results_tabular"
CHECKPOINT_FILE = "checkpoint_deep_mlp.json"
LOG_FILE = "experiment_deep_mlp.log"

# 7 representative algorithms
ALGORITHMS = [
    "FedAvg",        # Baseline
    "FedProx",       # MLSys'20    - Non-IID (proximal term)
    "Ditto",         # ICML'21     - Personalized (L2 regularization)
    "HPFL",          # ICLR'25     - Personalized (hybrid classifiers)
    "FedLESAM",      # ICML'24     - Generalization (global SAM)
    "FedExP",        # ICLR'23     - Server-side (extrapolation)
    "SCAFFOLD",      # ICML'20     - Non-IID (control variates)
]

SEEDS = [42, 123, 456]

# Same training config for both datasets
TRAINING_CONFIG = dict(
    learning_rate=0.01,
    batch_size=64,
    num_rounds=25,
    local_epochs=3,
    mu=0.1,
)

NUM_CLIENTS = 5

# Dataset-specific config
DATASET_CONFIGS = {
    "Cardiovascular": {
        "short": "CV",
        "input_dim": 11,
        "num_classes": 2,
        "class_names": {0: "Healthy", 1: "Disease"},
    },
    "PTB_XL": {
        "short": "PX",
        "input_dim": 9,
        "num_classes": 5,
        "class_names": {0: "NORM", 1: "MI", 2: "STTC", 3: "HYP", 4: "CD"},
    },
}


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
# Checkpoint (atomic write with backup) — saves after EVERY experiment
# ======================================================================

def save_checkpoint(data):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / CHECKPOINT_FILE
    bak = OUTPUT_DIR / (CHECKPOINT_FILE + ".bak")
    data["metadata"]["last_save"] = datetime.now().isoformat()
    fd, tmp = tempfile.mkstemp(dir=str(OUTPUT_DIR), prefix=".deep_mlp_", suffix=".tmp")
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
# Evaluation helpers
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


def _collect_predictions(trainer, num_classes):
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
# Data loading
# ======================================================================

def load_dataset(ds_name, seed):
    """Load dataset by name with appropriate partitioning."""
    if ds_name == "Cardiovascular":
        return load_cardiovascular_data(
            num_clients=NUM_CLIENTS,
            seed=seed,
            is_iid=False,
            alpha=0.5,
        )
    elif ds_name == "PTB_XL":
        return load_ptbxl_data(
            num_clients=NUM_CLIENTS,
            seed=seed,
            partition_by_site=True,
        )
    else:
        raise ValueError("Unknown dataset: {}".format(ds_name))


# ======================================================================
# Single experiment
# ======================================================================

def run_single_experiment(ds_name, algo, seed, ds_config, quick=False):
    start = time.time()

    num_rounds = 5 if quick else TRAINING_CONFIG["num_rounds"]
    input_dim = ds_config["input_dim"]
    num_classes = ds_config["num_classes"]
    class_names = ds_config["class_names"]

    client_data, client_test, metadata = load_dataset(ds_name, seed)

    trainer_kwargs = dict(
        num_clients=NUM_CLIENTS,
        algorithm=algo,
        local_epochs=TRAINING_CONFIG["local_epochs"],
        batch_size=TRAINING_CONFIG["batch_size"],
        learning_rate=TRAINING_CONFIG["learning_rate"],
        mu=TRAINING_CONFIG["mu"],
        seed=seed,
        external_data=client_data,
        external_test_data=client_test,
        input_dim=input_dim,
        num_classes=num_classes,
    )

    # No DP for this experiment

    # Pre-check: ensure no client has 0 training samples
    for cid in range(NUM_CLIENTS):
        X_c, y_c = client_data[cid]
        if len(y_c) == 0:
            raise ValueError(
                "Client {} has 0 training samples ({}, seed={})".format(
                    cid, ds_name, seed))

    # Log data distribution for diagnostics
    samples_per_client = [len(client_data[cid][1]) for cid in range(NUM_CLIENTS)]
    log("    Data distribution: {}  (total={})".format(
        samples_per_client, sum(samples_per_client)), also_print=False)

    trainer = FederatedTrainer(**trainer_kwargs)

    # Log model param count (first time per dataset)
    n_params = sum(p.numel() for p in trainer.global_model.parameters())
    log("    Model params: {:,}".format(n_params), also_print=False)

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

    y_true, y_pred = _collect_predictions(trainer, num_classes)
    per_class, cm = _compute_per_class_metrics(y_true, y_pred, num_classes, class_names)
    dei = _compute_dei(per_class, num_classes, class_names)

    elapsed = time.time() - start

    return {
        "dataset": ds_name,
        "algorithm": algo,
        "dp_level": "No-DP",
        "dp_epsilon": 0,
        "seed": seed,
        "num_clients": NUM_CLIENTS,
        "model": "DeepHealthcareMLP",
        "model_params": n_params,
        "hidden_dims": [256, 256, 128, 64],
        "final_accuracy": round(final_acc, 4),
        "best_accuracy": round(best_acc, 4),
        "per_client_accuracy": per_client_acc,
        "samples_per_client": samples_per_client,
        "jain_index": round(jain, 4),
        "dei": dei,
        "per_class_metrics": per_class,
        "confusion_matrix": cm,
        "convergence_trajectory": [h["accuracy"] for h in history],
        "time_seconds": round(elapsed, 1),
    }


# ======================================================================
# Summary printer
# ======================================================================

def print_summary(ckpt):
    """Print running summary table grouped by dataset, algo."""
    log("")
    log("=" * 80)
    log("SUMMARY — Deep MLP Convergence Test (No-DP)")
    log("=" * 80)

    for ds_name, ds_config in DATASET_CONFIGS.items():
        short = ds_config["short"]
        log("")
        log("  Dataset: {} ({})".format(ds_name, short))
        log("  {:<12} {:>5} {:>8} {:>8} {:>8} {:>8}".format(
            "Algo", "n", "Mean%", "Std%", "Min%", "Max%"))
        log("  " + "-" * 55)

        for algo in ALGORITHMS:
            keys = ["{}_{}_NDP_s{}".format(short, algo, s) for s in SEEDS]
            accs = [ckpt["results"][k]["final_accuracy"]
                    for k in keys if k in ckpt["results"]]
            if accs:
                arr = np.array(accs) * 100
                log("  {:<12} {:>5} {:>7.1f}% {:>7.1f}% {:>7.1f}% {:>7.1f}%".format(
                    algo, len(accs),
                    np.mean(arr), np.std(arr), np.min(arr), np.max(arr)))


def print_convergence_comparison(ckpt):
    """Print convergence analysis comparing all algorithms to FedAvg."""
    log("")
    log("=" * 80)
    log("CONVERGENCE ANALYSIS — Do all algorithms match FedAvg on the Deep MLP?")
    log("=" * 80)

    for ds_name, ds_config in DATASET_CONFIGS.items():
        short = ds_config["short"]
        log("")
        log("  Dataset: {} ({})".format(ds_name, short))
        log("")

        # Compute FedAvg baseline
        fedavg_keys = ["{}_{}_NDP_s{}".format(short, "FedAvg", s) for s in SEEDS]
        fedavg_accs = [ckpt["results"][k]["final_accuracy"] * 100
                       for k in fedavg_keys if k in ckpt["results"]]
        fedavg_mean = np.mean(fedavg_accs) if fedavg_accs else 0.0

        log("  {:<12} {:>10} {:>10} {:>12}".format(
            "Algorithm", "Mean%", "Std%", "Diff vs FedAvg"))
        log("  " + "-" * 50)

        # Print FedAvg baseline
        if fedavg_accs:
            log("  {:<12} {:>9.1f}% {:>9.1f}% {:>12}".format(
                "FedAvg",
                np.mean(fedavg_accs), np.std(fedavg_accs),
                "baseline"))

        for algo in ALGORITHMS:
            if algo == "FedAvg":
                continue
            keys = ["{}_{}_NDP_s{}".format(short, algo, s) for s in SEEDS]
            accs = [ckpt["results"][k]["final_accuracy"] * 100
                    for k in keys if k in ckpt["results"]]
            if accs:
                mean_acc = np.mean(accs)
                std_acc = np.std(accs)
                diff = mean_acc - fedavg_mean
                log("  {:<12} {:>9.1f}% {:>9.1f}% {:>+11.1f}pp".format(
                    algo, mean_acc, std_acc, diff))

    log("")
    log("  Algorithms with |diff| < 3pp from FedAvg are considered 'converged'.")
    log("  Algorithms with |diff| > 5pp show genuine differentiation.")

    # Cross-dataset summary
    log("")
    log("=" * 80)
    log("CROSS-DATASET SUMMARY")
    log("=" * 80)
    log("")
    log("  {:<12} {:>10} {:>10} {:>10} {:>10}".format(
        "Algorithm", "CV Mean%", "PX Mean%", "CV Diff", "PX Diff"))
    log("  " + "-" * 56)

    # Get FedAvg baselines for both datasets
    baselines = {}
    for ds_name, ds_config in DATASET_CONFIGS.items():
        short = ds_config["short"]
        keys = ["{}_{}_NDP_s{}".format(short, "FedAvg", s) for s in SEEDS]
        accs = [ckpt["results"][k]["final_accuracy"] * 100
                for k in keys if k in ckpt["results"]]
        baselines[short] = np.mean(accs) if accs else 0.0

    for algo in ALGORITHMS:
        means = {}
        for ds_name, ds_config in DATASET_CONFIGS.items():
            short = ds_config["short"]
            keys = ["{}_{}_NDP_s{}".format(short, algo, s) for s in SEEDS]
            accs = [ckpt["results"][k]["final_accuracy"] * 100
                    for k in keys if k in ckpt["results"]]
            means[short] = np.mean(accs) if accs else 0.0

        cv_diff = means.get("CV", 0) - baselines.get("CV", 0)
        px_diff = means.get("PX", 0) - baselines.get("PX", 0)

        if algo == "FedAvg":
            log("  {:<12} {:>9.1f}% {:>9.1f}% {:>10} {:>10}".format(
                algo, means.get("CV", 0), means.get("PX", 0),
                "baseline", "baseline"))
        else:
            log("  {:<12} {:>9.1f}% {:>9.1f}% {:>+9.1f}pp {:>+9.1f}pp".format(
                algo, means.get("CV", 0), means.get("PX", 0),
                cv_diff, px_diff))


# ======================================================================
# Main
# ======================================================================

def main():
    global _log_file

    parser = argparse.ArgumentParser(
        description="FL-EHDS: Deep MLP Convergence Test (CV+PX, 7 algos, 3 seeds)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick validation (5 rounds instead of 25)")
    parser.add_argument("--fresh", action="store_true",
                        help="Delete checkpoint and start fresh")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _log_file = open(str(OUTPUT_DIR / LOG_FILE), "a")

    # Patch model BEFORE any trainer creation
    _patch_model()

    total_experiments = len(ALGORITHMS) * len(DATASET_CONFIGS) * len(SEEDS)

    log("=" * 70)
    log("FL-EHDS: Deep MLP Convergence Test")
    log("Model: DeepHealthcareMLP [256, 256, 128, 64] (~110K params)")
    log("Datasets: {} | Clients: {} | Seeds: {}".format(
        list(DATASET_CONFIGS.keys()), NUM_CLIENTS, SEEDS))
    log("Algorithms (7): {}".format(", ".join(ALGORITHMS)))
    log("DP: No-DP only")
    log("Total: {} algos x {} datasets x {} seeds = {} experiments".format(
        len(ALGORITHMS), len(DATASET_CONFIGS), len(SEEDS), total_experiments))
    log("Mode: {}".format("QUICK (5 rounds)" if args.quick else "FULL (25 rounds)"))
    log("Device: {}".format(_detect_device("cpu")))
    log("Training: lr={}, batch={}, rounds={}, epochs={}, mu={}".format(
        TRAINING_CONFIG["learning_rate"], TRAINING_CONFIG["batch_size"],
        5 if args.quick else TRAINING_CONFIG["num_rounds"],
        TRAINING_CONFIG["local_epochs"], TRAINING_CONFIG["mu"]))
    for ds_name, ds_config in DATASET_CONFIGS.items():
        n_params = _count_params(ds_config["input_dim"], ds_config["num_classes"])
        log("  {} ({}): input_dim={}, num_classes={}, {:,} params".format(
            ds_name, ds_config["short"],
            ds_config["input_dim"], ds_config["num_classes"], n_params))
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
                "experiment": "Deep MLP Convergence Test",
                "description": (
                    "Test whether algorithm convergence holds on a deeper model "
                    "(DeepHealthcareMLP ~110K params) across Cardiovascular and "
                    "PTB-XL datasets. 7 algorithms x 2 datasets x 3 seeds = 42 experiments."
                ),
                "model": "DeepHealthcareMLP",
                "hidden_dims": [256, 256, 128, 64],
                "datasets": list(DATASET_CONFIGS.keys()),
                "num_clients": NUM_CLIENTS,
                "dp_levels": ["No-DP"],
                "seeds": SEEDS,
                "algorithms": ALGORITHMS,
                "training_config": TRAINING_CONFIG,
                "total_experiments": total_experiments,
                "started": datetime.now().isoformat(),
            },
            "results": {},
            "completed": [],
        }

    # Build experiment list — ordered by dataset, then algorithm
    experiments = []
    for ds_name, ds_config in DATASET_CONFIGS.items():
        short = ds_config["short"]
        for algo in ALGORITHMS:
            for seed in SEEDS:
                exp_key = "{}_{}_NDP_s{}".format(short, algo, seed)
                experiments.append((ds_name, ds_config, algo, seed, exp_key))

    total = len(experiments)
    done = len(ckpt["completed"])
    remaining = total - done
    log("Total: {} experiments, {} already done, {} remaining".format(total, done, remaining))

    if done > 0:
        log("Resuming from checkpoint...")
        print_summary(ckpt)

    cascade_start = time.time()
    current_block = None
    experiments_this_session = 0
    block_start = time.time()

    for idx, (ds_name, ds_config, algo, seed, exp_key) in enumerate(experiments, 1):
        if _shutdown:
            log("")
            log("Shutdown requested — saving checkpoint and printing summary")
            save_checkpoint(ckpt)
            print_summary(ckpt)
            break

        if exp_key in ckpt["completed"]:
            continue

        # Track dataset+algorithm blocks for progress reporting
        block_label = "{}/{}".format(ds_config["short"], algo)
        if block_label != current_block:
            if current_block is not None:
                block_time = time.time() - block_start
                log("")
                log("--- {} completed in {} ---".format(
                    current_block, format_time(block_time)))
                log("")
            current_block = block_label
            block_start = time.time()
            log(">>> Starting {} <<<".format(block_label))

        log("[{}/{}] {} | {} | seed={} ...".format(
            idx, total, ds_config["short"], algo, seed))

        try:
            result = run_single_experiment(
                ds_name, algo, seed, ds_config, quick=args.quick)

            ckpt["results"][exp_key] = result
            ckpt["completed"].append(exp_key)
            ckpt["metadata"]["completed_count"] = len(ckpt["completed"])
            save_checkpoint(ckpt)  # SAVE AFTER EVERY EXPERIMENT

            experiments_this_session += 1
            elapsed_session = time.time() - cascade_start
            avg_per_exp = elapsed_session / experiments_this_session
            remaining_count = total - len(ckpt["completed"])
            eta = avg_per_exp * remaining_count

            log("  -> acc={:.1f}% | Jain={:.3f} | DEI={:.3f} | {:.0f}s | "
                "ETA: {} ({} remaining)".format(
                    result["final_accuracy"] * 100,
                    result["jain_index"],
                    result["dei"],
                    result["time_seconds"],
                    format_time(eta),
                    remaining_count))

            # Log per-client accuracy (file only, for diagnostics)
            pca = result["per_client_accuracy"]
            log("     per-client: [{}]".format(
                ", ".join("{:.1f}%".format(pca[str(c)] * 100) for c in range(NUM_CLIENTS))),
                also_print=False)

            _cleanup_gpu()

        except Exception as e:
            log("  ERROR: {} — {}".format(type(e).__name__, e))
            log("  " + traceback.format_exc().split("\n")[-2])
            _cleanup_gpu()

    total_time = time.time() - cascade_start

    ckpt["metadata"]["total_time"] = format_time(total_time)
    ckpt["metadata"]["completed_count"] = len(ckpt["completed"])
    save_checkpoint(ckpt)

    log("")
    log("=" * 70)
    log("COMPLETE: {}/{} experiments in {}".format(
        len(ckpt["completed"]), total, format_time(total_time)))
    log("Checkpoint: {}".format(CHECKPOINT_FILE))
    log("=" * 70)

    print_summary(ckpt)
    print_convergence_comparison(ckpt)

    if _log_file:
        _log_file.close()


if __name__ == "__main__":
    main()

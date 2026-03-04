#!/usr/bin/env python3
"""
FL-EHDS Experiment — Non-IID + DP Clarification (Cardiovascular, 5 seeds).

Re-runs the Cardiovascular Non-IID + DP experiment with 5 seeds per condition
to produce statistically robust results for Figure 3 and Section D of the
FLICS 2026 paper (v3).

Motivation:
  The original checkpoint_noniid_dp.json had only n=2 at alpha=0.1
  (seed 456 missing) and seed s123 showed anomalous 99.93% for HPFL.
  This experiment uses 5 seeds uniformly across all conditions.

Design:
  - Dataset: Cardiovascular ONLY (5 clients, binary)
  - Algorithms: FedAvg, Ditto, HPFL
  - Non-IID levels: alpha = 0.1 (extreme), 0.5 (moderate), 1.0 (mild)
  - DP levels: No-DP, eps=1, eps=10
  - Seeds: 42, 123, 456, 789, 999
  - Total: 3 algos x 3 alphas x 3 DP x 5 seeds = 135 experiments
  - Training: lr=0.01, batch=64, 25 rounds, 3 local epochs (same as original)

Checkpointing:
  - Saves after EVERY experiment (atomic write with backup)
  - Prints running summary after each experiment
  - Graceful shutdown on Ctrl-C (finishes current, saves, exits)
  - Resume-safe: skips already-completed experiments

Usage:
    cd fl-ehds-framework
    python -m benchmarks.run_noniid_dp_5seed [--quick] [--fresh]

Output:
    benchmarks/paper_results_tabular/checkpoint_noniid_dp_5seed.json

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

import numpy as np

FRAMEWORK_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(FRAMEWORK_DIR))

import torch

from terminal.fl_trainer import FederatedTrainer, _detect_device
from data.cardiovascular_loader import load_cardiovascular_data

# ======================================================================
# Constants
# ======================================================================

OUTPUT_DIR = FRAMEWORK_DIR / "benchmarks" / "paper_results_tabular"
CHECKPOINT_FILE = "checkpoint_noniid_dp_5seed.json"
LOG_FILE = "experiment_noniid_dp_5seed.log"

ALGORITHMS = ["FedAvg", "Ditto", "HPFL"]
SEEDS = [42, 123, 456, 789, 999]

CV_ALPHAS = [0.1, 0.5, 1.0]

DP_LEVELS = [
    {"label": "No-DP", "dp_enabled": False, "dp_epsilon": 0},
    {"label": "DP-eps1", "dp_enabled": True, "dp_epsilon": 1.0},
    {"label": "DP-eps10", "dp_enabled": True, "dp_epsilon": 10.0},
]

# Same training config as original run_noniid_dp.py
CV_TRAINING = dict(
    learning_rate=0.01,
    batch_size=64,
    num_rounds=25,
    local_epochs=3,
    mu=0.1,
)

NUM_CLIENTS = 5
NUM_CLASSES = 2
INPUT_DIM = 11
CLASS_NAMES = {0: "Healthy", 1: "Disease"}

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
    fd, tmp = tempfile.mkstemp(dir=str(OUTPUT_DIR), prefix=".niiddp5s_", suffix=".tmp")
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
# Evaluation helpers (same as original)
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


def _compute_per_class_metrics(y_true, y_pred):
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1

    per_class = {}
    for c in range(NUM_CLASSES):
        cls_name = CLASS_NAMES.get(c, "Class_{}".format(c))
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


def _compute_dei(per_class_metrics):
    recalls = []
    for c in range(NUM_CLASSES):
        cls_name = CLASS_NAMES.get(c, "Class_{}".format(c))
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

def run_single_experiment(algo, alpha, dp_cfg, seed, quick=False):
    start = time.time()

    num_rounds = 5 if quick else CV_TRAINING["num_rounds"]

    client_data, client_test, metadata = load_cardiovascular_data(
        num_clients=NUM_CLIENTS, seed=seed,
        is_iid=False, alpha=alpha,
    )

    trainer_kwargs = dict(
        num_clients=NUM_CLIENTS,
        algorithm=algo,
        local_epochs=CV_TRAINING["local_epochs"],
        batch_size=CV_TRAINING["batch_size"],
        learning_rate=CV_TRAINING["learning_rate"],
        mu=CV_TRAINING["mu"],
        seed=seed,
        external_data=client_data,
        external_test_data=client_test,
        input_dim=INPUT_DIM,
        num_classes=NUM_CLASSES,
    )

    if dp_cfg["dp_enabled"]:
        trainer_kwargs["dp_enabled"] = True
        trainer_kwargs["dp_epsilon"] = dp_cfg["dp_epsilon"]
        trainer_kwargs["dp_clip_norm"] = 1.0

    # Pre-check: ensure no client has 0 training samples
    for cid in range(NUM_CLIENTS):
        X_c, y_c = client_data[cid]
        if len(y_c) == 0:
            raise ValueError(
                "Client {} has 0 training samples (alpha={}, seed={})".format(
                    cid, alpha, seed))

    # Log data distribution for diagnostics
    samples_per_client = [len(client_data[cid][1]) for cid in range(NUM_CLIENTS)]
    log("    Data distribution: {}  (total={})".format(
        samples_per_client, sum(samples_per_client)), also_print=False)

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
    per_class, cm = _compute_per_class_metrics(y_true, y_pred)
    dei = _compute_dei(per_class)

    elapsed = time.time() - start

    return {
        "dataset": "Cardiovascular",
        "algorithm": algo,
        "alpha": alpha,
        "dp_level": dp_cfg["label"],
        "dp_epsilon": dp_cfg["dp_epsilon"],
        "seed": seed,
        "num_clients": NUM_CLIENTS,
        "final_accuracy": round(final_acc, 4),
        "best_accuracy": round(best_acc, 4),
        "per_client_accuracy": per_client_acc,
        "samples_per_client": [len(client_data[cid][1]) for cid in range(NUM_CLIENTS)],
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
    """Print running summary table grouped by algo, alpha, DP."""
    log("")
    log("=" * 80)
    log("SUMMARY — Cardiovascular Non-IID + DP (5-seed clarification)")
    log("=" * 80)
    log("")
    log("{:<8} {:<6} {:<10} {:>5} {:>8} {:>8} {:>8} {:>8}".format(
        "Algo", "Alpha", "DP", "n", "Mean%", "Std%", "Min%", "Max%"))
    log("-" * 70)

    for algo in ALGORITHMS:
        for alpha in CV_ALPHAS:
            for dp_cfg in DP_LEVELS:
                keys = ["CV_{}_a{}_{}_s{}".format(algo, alpha, dp_cfg["label"], s)
                        for s in SEEDS]
                accs = [ckpt["results"][k]["final_accuracy"]
                        for k in keys if k in ckpt["results"]]
                if accs:
                    arr = np.array(accs) * 100
                    log("{:<8} {:<6} {:<10} {:>5} {:>7.1f}% {:>7.1f}% {:>7.1f}% {:>7.1f}%".format(
                        algo, alpha, dp_cfg["label"], len(accs),
                        np.mean(arr), np.std(arr), np.min(arr), np.max(arr)))
        log("")


# ======================================================================
# Main
# ======================================================================

def main():
    global _log_file

    parser = argparse.ArgumentParser(
        description="FL-EHDS: Non-IID + DP Clarification (CV, 5 seeds)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick validation (5 rounds instead of 25)")
    parser.add_argument("--fresh", action="store_true",
                        help="Delete checkpoint and start fresh")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _log_file = open(str(OUTPUT_DIR / LOG_FILE), "a")

    log("=" * 70)
    log("FL-EHDS: Non-IID + DP Clarification Experiment")
    log("Dataset: Cardiovascular | Clients: {} | Seeds: {}".format(NUM_CLIENTS, SEEDS))
    log("Alphas: {} | DP: No-DP, eps=1, eps=10".format(CV_ALPHAS))
    log("Total: 3 algos x 3 alphas x 3 DP x 5 seeds = 135 experiments")
    log("Mode: {}".format("QUICK (5 rounds)" if args.quick else "FULL (25 rounds)"))
    log("Device: {}".format(_detect_device("cpu")))
    log("Training: lr={}, batch={}, rounds={}, epochs={}, mu={}".format(
        CV_TRAINING["learning_rate"], CV_TRAINING["batch_size"],
        5 if args.quick else CV_TRAINING["num_rounds"],
        CV_TRAINING["local_epochs"], CV_TRAINING["mu"]))
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
                "experiment": "Non-IID + DP Clarification (5-seed)",
                "description": (
                    "Re-run Cardiovascular Non-IID + DP with 5 seeds per condition. "
                    "Clarifies HPFL alpha=0.1 anomaly (s123=99.93%) from original "
                    "checkpoint_noniid_dp.json which had only n=2 at alpha=0.1."
                ),
                "dataset": "Cardiovascular",
                "num_clients": NUM_CLIENTS,
                "alphas": CV_ALPHAS,
                "dp_levels": ["No-DP", "DP-eps1", "DP-eps10"],
                "seeds": SEEDS,
                "algorithms": ALGORITHMS,
                "training_config": CV_TRAINING,
                "total_experiments": 135,
                "started": datetime.now().isoformat(),
            },
            "results": {},
            "completed": [],
        }

    # Build experiment list — ordered by alpha (so we get full alpha blocks early)
    experiments = []
    for alpha in CV_ALPHAS:
        for algo in ALGORITHMS:
            for dp_cfg in DP_LEVELS:
                for seed in SEEDS:
                    exp_key = "CV_{}_a{}_{}_s{}".format(algo, alpha, dp_cfg["label"], seed)
                    experiments.append((algo, alpha, dp_cfg, seed, exp_key))

    total = len(experiments)
    done = len(ckpt["completed"])
    remaining = total - done
    log("Total: {} experiments, {} already done, {} remaining".format(total, done, remaining))

    if done > 0:
        log("Resuming from checkpoint...")
        print_summary(ckpt)

    cascade_start = time.time()
    block_start = time.time()
    current_alpha = None
    experiments_this_session = 0

    for idx, (algo, alpha, dp_cfg, seed, exp_key) in enumerate(experiments, 1):
        if _shutdown:
            log("")
            log("Shutdown requested — saving checkpoint and printing summary")
            save_checkpoint(ckpt)
            print_summary(ckpt)
            break

        if exp_key in ckpt["completed"]:
            continue

        # Track alpha blocks for progress reporting
        if alpha != current_alpha:
            if current_alpha is not None:
                block_time = time.time() - block_start
                log("")
                log("--- Block alpha={} completed in {} ---".format(
                    current_alpha, format_time(block_time)))
                log("")
            current_alpha = alpha
            block_start = time.time()
            log(">>> Starting block: alpha={} <<<".format(alpha))

        log("[{}/{}] {} | alpha={} | {} | seed={} ...".format(
            idx, total, algo, alpha, dp_cfg["label"], seed))

        try:
            result = run_single_experiment(algo, alpha, dp_cfg, seed, quick=args.quick)

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

            # Print per-client accuracy for diagnostics (important for anomaly detection)
            pca = result["per_client_accuracy"]
            log("     per-client: [{}]".format(
                ", ".join("{:.1f}%".format(pca[str(c)] * 100) for c in range(NUM_CLIENTS))),
                also_print=False)

            _cleanup_gpu()

        except Exception as e:
            log("  ERROR: {}".format(e))
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

    # Print detailed alpha=0.1 analysis (the key question)
    log("")
    log("=" * 70)
    log("DETAILED ANALYSIS: alpha=0.1 (the anomaly under investigation)")
    log("=" * 70)
    for algo in ALGORITHMS:
        log("")
        log("  {} (No-DP):".format(algo))
        for seed in SEEDS:
            key = "CV_{}_a0.1_No-DP_s{}".format(algo, seed)
            if key in ckpt["results"]:
                r = ckpt["results"][key]
                pca = r["per_client_accuracy"]
                log("    seed={}: acc={:.2f}%  per-client=[{}]".format(
                    seed,
                    r["final_accuracy"] * 100,
                    ", ".join("{:.1f}%".format(pca[str(c)] * 100) for c in range(NUM_CLIENTS))))
        # Compute stats
        keys_nodp = ["CV_{}_a0.1_No-DP_s{}".format(algo, s) for s in SEEDS]
        accs = [ckpt["results"][k]["final_accuracy"] * 100
                for k in keys_nodp if k in ckpt["results"]]
        if len(accs) >= 2:
            log("    => mean={:.1f}% std={:.1f}% CI95=[{:.1f}%, {:.1f}%]  (n={})".format(
                np.mean(accs), np.std(accs),
                np.mean(accs) - 1.96 * np.std(accs) / np.sqrt(len(accs)),
                np.mean(accs) + 1.96 * np.std(accs) / np.sqrt(len(accs)),
                len(accs)))

    if _log_file:
        _log_file.close()


if __name__ == "__main__":
    main()

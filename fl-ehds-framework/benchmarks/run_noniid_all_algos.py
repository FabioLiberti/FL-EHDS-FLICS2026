#!/usr/bin/env python3
"""
FL-EHDS Experiment — Non-IID All Algorithms (Cardiovascular, 5 seeds, No-DP).

Runs the 14 algorithms NOT yet tested in the non-IID sweep to complete
Figure 3 of the FLICS 2026 paper. Combined with checkpoint_noniid_dp_5seed.json
(FedAvg, Ditto, HPFL), this provides data for all 17 FL-EHDS algorithms.

Design:
  - Dataset: Cardiovascular ONLY (5 clients, binary)
  - Algorithms: 14 missing (FedProx, SCAFFOLD, FedNova, FedDyn, FedAdam,
    FedYogi, FedAdagrad, Per-FedAvg, FedLC, FedSAM, FedDecorr, FedSpeed,
    FedExP, FedLESAM)
  - Non-IID levels: alpha = 0.1 (extreme), 0.5 (moderate), 1.0 (mild)
  - DP: No-DP only (figure is No-DP)
  - Seeds: 42, 123, 456, 789, 999
  - Total: 14 algos x 3 alphas x 5 seeds = 210 experiments
  - Training: lr=0.01, batch=64, 25 rounds, 3 local epochs (same config
    as run_noniid_dp_5seed.py for direct comparison)

Checkpointing:
  - Saves after EVERY experiment (atomic write with backup)
  - Resume-safe: skips already-completed experiments
  - Graceful shutdown on Ctrl-C

Usage:
    cd fl-ehds-framework
    python -m benchmarks.run_noniid_all_algos [--quick] [--fresh]

Output:
    benchmarks/paper_results_tabular/checkpoint_noniid_all_algos.json

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
CHECKPOINT_FILE = "checkpoint_noniid_all_algos.json"
LOG_FILE = "experiment_noniid_all_algos.log"

# The 14 algorithms NOT already in checkpoint_noniid_dp_5seed.json
# (which has: FedAvg, Ditto, HPFL)
# Order: foundational → non-IID → adaptive → personalized → advanced
ALGORITHMS = [
    "FedProx",       # MLSys'20    - Non-IID (proximal term)
    "SCAFFOLD",      # ICML'20     - Non-IID (control variates)
    "FedNova",       # NeurIPS'20  - Non-IID (normalized averaging)
    "FedDyn",        # ICLR'21     - Non-IID (dynamic regularization)
    "FedAdam",       # ICLR'21     - Adaptive (server-side Adam)
    "FedYogi",       # ICLR'21     - Adaptive (server-side Yogi)
    "FedAdagrad",    # ICLR'21     - Adaptive (server-side Adagrad)
    "Per-FedAvg",    # NeurIPS'20  - Personalized (MAML-based)
    "FedLC",         # ICML'22     - Label skew (logit calibration)
    "FedSAM",        # ICML'22     - Generalization (sharpness-aware)
    "FedDecorr",     # ICLR'23     - Representation (decorrelation)
    "FedSpeed",      # ICLR'23     - Efficiency (prox + SAM)
    "FedExP",        # ICLR'23     - Server-side (extrapolation)
    "FedLESAM",      # ICML'24     - Generalization (global SAM)
]

SEEDS = [42, 123, 456, 789, 999]
CV_ALPHAS = [0.1, 0.5, 1.0]

# Same training config as run_noniid_dp_5seed.py for direct comparison
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
    fd, tmp = tempfile.mkstemp(dir=str(OUTPUT_DIR), prefix=".niid_all_", suffix=".tmp")
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
# Evaluation helpers (same as run_noniid_dp_5seed.py)
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

def run_single_experiment(algo, alpha, seed, quick=False):
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

    # No DP for this experiment (figure is No-DP)

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
        "dp_level": "No-DP",
        "dp_epsilon": 0,
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
    """Print running summary table grouped by algo, alpha."""
    log("")
    log("=" * 80)
    log("SUMMARY — Cardiovascular Non-IID All Algorithms (No-DP)")
    log("=" * 80)
    log("")
    log("{:<12} {:<6} {:>5} {:>8} {:>8} {:>8} {:>8}".format(
        "Algo", "Alpha", "n", "Mean%", "Std%", "Min%", "Max%"))
    log("-" * 65)

    for algo in ALGORITHMS:
        for alpha in CV_ALPHAS:
            keys = ["CV_{}_a{}_NDP_s{}".format(algo, alpha, s) for s in SEEDS]
            accs = [ckpt["results"][k]["final_accuracy"]
                    for k in keys if k in ckpt["results"]]
            if accs:
                arr = np.array(accs) * 100
                log("{:<12} {:<6} {:>5} {:>7.1f}% {:>7.1f}% {:>7.1f}% {:>7.1f}%".format(
                    algo, alpha, len(accs),
                    np.mean(arr), np.std(arr), np.min(arr), np.max(arr)))
        log("")


def print_convergence_comparison(ckpt):
    """Print comparison with FedAvg baseline from 5-seed checkpoint."""
    log("")
    log("=" * 80)
    log("CONVERGENCE ANALYSIS — Do all algorithms cluster with FedAvg?")
    log("=" * 80)
    log("")

    # Try to load FedAvg/Ditto/HPFL data from 5-seed checkpoint
    ref_file = OUTPUT_DIR / "checkpoint_noniid_dp_5seed.json"
    ref_data = {}
    if ref_file.exists():
        try:
            with open(ref_file) as f:
                ref_ckpt = json.load(f)
            for alpha in CV_ALPHAS:
                ref_accs = []
                for s in SEEDS:
                    key = "CV_FedAvg_a{}_No-DP_s{}".format(alpha, s)
                    if key in ref_ckpt["results"]:
                        ref_accs.append(ref_ckpt["results"][key]["final_accuracy"] * 100)
                if ref_accs:
                    ref_data[alpha] = np.mean(ref_accs)
            log("  FedAvg baseline loaded from checkpoint_noniid_dp_5seed.json")
        except Exception:
            log("  WARNING: Could not load FedAvg baseline")
    else:
        log("  WARNING: checkpoint_noniid_dp_5seed.json not found")

    log("")
    log("{:<12} {:>10} {:>10} {:>10} {:>12} {:>12} {:>12}".format(
        "Algorithm", "a=0.1", "a=0.5", "a=1.0",
        "Diff@0.1", "Diff@0.5", "Diff@1.0"))
    log("-" * 78)

    # Print FedAvg reference
    if ref_data:
        log("{:<12} {:>9.1f}% {:>9.1f}% {:>9.1f}% {:>12} {:>12} {:>12}".format(
            "FedAvg*",
            ref_data.get(0.1, 0), ref_data.get(0.5, 0), ref_data.get(1.0, 0),
            "baseline", "baseline", "baseline"))

    for algo in ALGORITHMS:
        means = {}
        for alpha in CV_ALPHAS:
            keys = ["CV_{}_a{}_NDP_s{}".format(algo, alpha, s) for s in SEEDS]
            accs = [ckpt["results"][k]["final_accuracy"] * 100
                    for k in keys if k in ckpt["results"]]
            if accs:
                means[alpha] = np.mean(accs)

        if means:
            diffs = {}
            for alpha in CV_ALPHAS:
                if alpha in means and alpha in ref_data:
                    diffs[alpha] = means[alpha] - ref_data[alpha]
                else:
                    diffs[alpha] = None

            log("{:<12} {:>9.1f}% {:>9.1f}% {:>9.1f}% {:>+11.1f}pp {:>+11.1f}pp {:>+11.1f}pp".format(
                algo,
                means.get(0.1, 0), means.get(0.5, 0), means.get(1.0, 0),
                diffs.get(0.1, 0) or 0, diffs.get(0.5, 0) or 0, diffs.get(1.0, 0) or 0))

    # Also print Ditto and HPFL from reference for complete picture
    if ref_file.exists():
        log("")
        for ref_algo in ["Ditto", "HPFL"]:
            ref_means = {}
            ref_diffs = {}
            for alpha in CV_ALPHAS:
                accs = []
                for s in SEEDS:
                    key = "CV_{}_a{}_No-DP_s{}".format(ref_algo, alpha, s)
                    if key in ref_ckpt["results"]:
                        accs.append(ref_ckpt["results"][key]["final_accuracy"] * 100)
                if accs:
                    ref_means[alpha] = np.mean(accs)
                    if alpha in ref_data:
                        ref_diffs[alpha] = ref_means[alpha] - ref_data[alpha]
            if ref_means:
                log("{:<12} {:>9.1f}% {:>9.1f}% {:>9.1f}% {:>+11.1f}pp {:>+11.1f}pp {:>+11.1f}pp".format(
                    ref_algo + "*",
                    ref_means.get(0.1, 0), ref_means.get(0.5, 0), ref_means.get(1.0, 0),
                    ref_diffs.get(0.1, 0) or 0, ref_diffs.get(0.5, 0) or 0, ref_diffs.get(1.0, 0) or 0))

    log("")
    log("  * = from checkpoint_noniid_dp_5seed.json")
    log("")
    log("  Algorithms with |diff| < 3pp from FedAvg are considered 'converged'.")
    log("  Algorithms with |diff| > 5pp show genuine differentiation.")


# ======================================================================
# Main
# ======================================================================

def main():
    global _log_file

    parser = argparse.ArgumentParser(
        description="FL-EHDS: Non-IID All Algorithms (CV, 5 seeds, No-DP)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick validation (5 rounds instead of 25)")
    parser.add_argument("--fresh", action="store_true",
                        help="Delete checkpoint and start fresh")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _log_file = open(str(OUTPUT_DIR / LOG_FILE), "a")

    total_experiments = len(ALGORITHMS) * len(CV_ALPHAS) * len(SEEDS)

    log("=" * 70)
    log("FL-EHDS: Non-IID All Algorithms Experiment")
    log("Dataset: Cardiovascular | Clients: {} | Seeds: {}".format(NUM_CLIENTS, SEEDS))
    log("Algorithms (14): {}".format(", ".join(ALGORITHMS)))
    log("Alphas: {} | DP: No-DP only".format(CV_ALPHAS))
    log("Total: {} algos x {} alphas x {} seeds = {} experiments".format(
        len(ALGORITHMS), len(CV_ALPHAS), len(SEEDS), total_experiments))
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
                "experiment": "Non-IID All Algorithms (No-DP, 5-seed)",
                "description": (
                    "Run 14 additional algorithms on Cardiovascular Non-IID sweep "
                    "(alpha=0.1/0.5/1.0, No-DP, 5 seeds) to complete Figure 3. "
                    "Complements checkpoint_noniid_dp_5seed.json (FedAvg, Ditto, HPFL)."
                ),
                "dataset": "Cardiovascular",
                "num_clients": NUM_CLIENTS,
                "alphas": CV_ALPHAS,
                "dp_levels": ["No-DP"],
                "seeds": SEEDS,
                "algorithms": ALGORITHMS,
                "training_config": CV_TRAINING,
                "total_experiments": total_experiments,
                "started": datetime.now().isoformat(),
            },
            "results": {},
            "completed": [],
        }

    # Build experiment list — ordered by algorithm (finish each algo fully)
    experiments = []
    for algo in ALGORITHMS:
        for alpha in CV_ALPHAS:
            for seed in SEEDS:
                exp_key = "CV_{}_a{}_NDP_s{}".format(algo, alpha, seed)
                experiments.append((algo, alpha, seed, exp_key))

    total = len(experiments)
    done = len(ckpt["completed"])
    remaining = total - done
    log("Total: {} experiments, {} already done, {} remaining".format(total, done, remaining))

    if done > 0:
        log("Resuming from checkpoint...")
        print_summary(ckpt)

    cascade_start = time.time()
    current_algo = None
    experiments_this_session = 0
    algo_start = time.time()

    for idx, (algo, alpha, seed, exp_key) in enumerate(experiments, 1):
        if _shutdown:
            log("")
            log("Shutdown requested — saving checkpoint and printing summary")
            save_checkpoint(ckpt)
            print_summary(ckpt)
            break

        if exp_key in ckpt["completed"]:
            continue

        # Track algorithm blocks for progress reporting
        if algo != current_algo:
            if current_algo is not None:
                algo_time = time.time() - algo_start
                log("")
                log("--- {} completed in {} ---".format(
                    current_algo, format_time(algo_time)))
                log("")
            current_algo = algo
            algo_start = time.time()
            algo_idx = ALGORITHMS.index(algo) + 1
            log(">>> Starting algorithm {}/{}: {} <<<".format(
                algo_idx, len(ALGORITHMS), algo))

        log("[{}/{}] {} | alpha={} | seed={} ...".format(
            idx, total, algo, alpha, seed))

        try:
            result = run_single_experiment(algo, alpha, seed, quick=args.quick)

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

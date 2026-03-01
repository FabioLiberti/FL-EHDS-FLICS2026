#!/usr/bin/env python3
"""
FL-EHDS Experiment — Differential Privacy on PTB-XL.

Tests DP impact on PTB-XL (5-class ECG) across 3 algorithms.
PTB-XL is the most challenging dataset (5 classes, natural hospital partitioning)
and DP impact may differ from simpler binary tasks.

Design:
  - Dataset: PTB-XL (5-class ECG, 9 features, partition_by_site)
  - Algorithms: FedAvg, Ditto, HPFL
  - DP: No-DP, eps=50, eps=10, eps=1
  - Seeds: 42, 123, 456
  - Total: 3 algos x 4 DP x 3 seeds = 36 experiments (~1.5h on M3)

Usage:
    cd fl-ehds-framework
    python -m benchmarks.run_dp_ptbxl [--quick] [--fresh]

Output:
    benchmarks/paper_results_tabular/checkpoint_dp_ptbxl.json
    benchmarks/paper_results_tabular/dp_ptbxl_impact.png

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

from terminal.fl_trainer import FederatedTrainer, _detect_device
from data.ptbxl_loader import load_ptbxl_data

# ======================================================================
# Constants
# ======================================================================

OUTPUT_DIR = FRAMEWORK_DIR / "benchmarks" / "paper_results_tabular"
CHECKPOINT_FILE = "checkpoint_dp_ptbxl.json"
LOG_FILE = "experiment_dp_ptbxl.log"

ALGORITHMS = ["FedAvg", "Ditto", "HPFL"]

DP_LEVELS = [
    {"label": "No-DP", "dp_enabled": False, "dp_epsilon": 0, "dp_clip_norm": 1.0},
    {"label": "eps=50", "dp_enabled": True, "dp_epsilon": 50.0, "dp_clip_norm": 1.0},
    {"label": "eps=10", "dp_enabled": True, "dp_epsilon": 10.0, "dp_clip_norm": 1.0},
    {"label": "eps=1",  "dp_enabled": True, "dp_epsilon": 1.0, "dp_clip_norm": 1.0},
]

SEEDS = [42, 123, 456]

PX_CONFIG = dict(
    input_dim=9, num_classes=5,
    num_clients=5,
    learning_rate=0.005,
    batch_size=64,
    num_rounds=30,
    local_epochs=3,
    mu=0.1,
)

PTB_XL_CLASS_NAMES = {0: "NORM", 1: "MI", 2: "STTC", 3: "CD", 4: "HYP"}

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
    fd, tmp = tempfile.mkstemp(dir=str(OUTPUT_DIR), prefix=".dpptbxl_", suffix=".tmp")
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
    return load_ptbxl_data(
        num_clients=PX_CONFIG["num_clients"], seed=seed,
        partition_by_site=True, min_site_samples=50,
    )


# ======================================================================
# Per-client evaluation
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


# ======================================================================
# Per-class metrics (5 PTB-XL classes)
# ======================================================================

def _collect_predictions(trainer):
    """Collect all predictions and true labels from test data."""
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


def _compute_confusion_matrix(y_true, y_pred, num_classes):
    """Compute confusion matrix manually."""
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def _compute_per_class_metrics(y_true, y_pred, num_classes):
    """Compute per-class precision, recall, F1 for all PTB-XL classes."""
    cm = _compute_confusion_matrix(y_true, y_pred, num_classes)
    per_class = {}

    for c in range(num_classes):
        cls_name = PTB_XL_CLASS_NAMES.get(c, "Class_{}".format(c))
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

    # Macro averages
    all_prec = [v["precision"] for v in per_class.values()]
    all_rec = [v["recall"] for v in per_class.values()]
    all_f1 = [v["f1"] for v in per_class.values()]
    per_class["macro_avg"] = {
        "precision": round(float(np.mean(all_prec)), 4),
        "recall": round(float(np.mean(all_rec)), 4),
        "f1": round(float(np.mean(all_f1)), 4),
    }

    return per_class, cm.tolist()


# ======================================================================
# Single experiment
# ======================================================================

def run_single_experiment(algo, dp_level, seed, quick=False):
    """Run one experiment: algorithm + DP level + seed on PTB-XL."""
    start = time.time()
    num_rounds = 5 if quick else PX_CONFIG["num_rounds"]

    client_data, client_test, metadata = load_dataset(seed)

    trainer = FederatedTrainer(
        num_clients=PX_CONFIG["num_clients"],
        algorithm=algo,
        local_epochs=PX_CONFIG["local_epochs"],
        batch_size=PX_CONFIG["batch_size"],
        learning_rate=PX_CONFIG["learning_rate"],
        mu=PX_CONFIG["mu"],
        dp_enabled=dp_level["dp_enabled"],
        dp_epsilon=dp_level["dp_epsilon"] if dp_level["dp_enabled"] else 10.0,
        dp_clip_norm=dp_level["dp_clip_norm"],
        seed=seed,
        external_data=client_data,
        external_test_data=client_test,
        input_dim=PX_CONFIG["input_dim"],
        num_classes=PX_CONFIG["num_classes"],
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
        }
        history.append(metrics)
        if result.global_acc > best_acc:
            best_acc = result.global_acc

    # Per-client accuracy
    per_client_acc = _evaluate_per_client(trainer)

    # Per-class metrics for 5 PTB-XL classes
    y_true, y_pred = _collect_predictions(trainer)
    per_class_metrics, confusion_matrix = _compute_per_class_metrics(
        y_true, y_pred, PX_CONFIG["num_classes"])

    elapsed = time.time() - start

    out = {
        "algorithm": algo,
        "dp_label": dp_level["label"],
        "dp_enabled": dp_level["dp_enabled"],
        "dp_epsilon": dp_level["dp_epsilon"],
        "dp_clip_norm": dp_level["dp_clip_norm"],
        "seed": seed,
        "history": history,
        "final_metrics": history[-1] if history else {},
        "per_client_acc": per_client_acc,
        "per_class_metrics": per_class_metrics,
        "confusion_matrix": confusion_matrix,
        "best_accuracy": best_acc,
        "actual_rounds": len(history),
        "runtime_seconds": round(elapsed, 1),
    }

    _cleanup_gpu()
    return out


# ======================================================================
# Figure generation
# ======================================================================

def generate_figure(completed, algos, dp_levels, seeds):
    """Grouped bar chart: 3 groups (algos), 4 bars per group (DP levels), y-axis = accuracy."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log("matplotlib not available -- skipping figure")
        return

    dp_labels = [d["label"] for d in dp_levels]
    n_algos = len(algos)
    n_dp = len(dp_labels)

    colors = {
        "No-DP": "#2ca02c",
        "eps=50": "#1f77b4",
        "eps=10": "#ff7f0e",
        "eps=1": "#d62728",
    }

    fig, ax = plt.subplots(figsize=(10, 6))

    bar_width = 0.18
    group_width = n_dp * bar_width + 0.1
    group_positions = np.arange(n_algos) * group_width

    for j, dp_label in enumerate(dp_labels):
        means = []
        stds = []
        for algo in algos:
            accs = []
            for s in seeds:
                key = "{}_{}_s{}".format(algo, dp_label, s)
                if key in completed and "error" not in completed[key]:
                    accs.append(completed[key]["best_accuracy"] * 100)
            if accs:
                means.append(np.mean(accs))
                stds.append(np.std(accs))
            else:
                means.append(0)
                stds.append(0)

        x_positions = group_positions + j * bar_width
        bars = ax.bar(
            x_positions, means, bar_width,
            yerr=stds, capsize=3,
            label=dp_label,
            color=colors.get(dp_label, "gray"),
            edgecolor="black", linewidth=0.5, alpha=0.85,
        )

        # Add value labels on top of bars
        for bar_obj, m in zip(bars, means):
            if m > 0:
                ax.text(
                    bar_obj.get_x() + bar_obj.get_width() / 2.0, bar_obj.get_height() + 0.5,
                    "{:.1f}".format(m),
                    ha="center", va="bottom", fontsize=8, fontweight="bold",
                )

    ax.set_xlabel("Algorithm", fontsize=13)
    ax.set_ylabel("Accuracy (%)", fontsize=13)
    ax.set_title("Differential Privacy Impact on PTB-XL (5-class ECG)",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(group_positions + (n_dp - 1) * bar_width / 2.0)
    ax.set_xticklabels(algos, fontsize=12)
    ax.legend(fontsize=10, title="DP Level", title_fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    # Set y-axis range with some padding
    all_accs = []
    for key, res in completed.items():
        if "error" not in res:
            all_accs.append(res["best_accuracy"] * 100)
    if all_accs:
        y_min = max(0, min(all_accs) - 10)
        y_max = min(100, max(all_accs) + 5)
        ax.set_ylim(y_min, y_max)

    fig.tight_layout()

    fig_path = OUTPUT_DIR / "dp_ptbxl_impact.png"
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.savefig(str(fig_path).replace(".png", ".pdf"), bbox_inches="tight")
    log("Figure saved: {}".format(fig_path))
    plt.close()

    # Also generate per-class recall heatmap for each algorithm
    _generate_per_class_heatmap(completed, algos, dp_levels, seeds)


def _generate_per_class_heatmap(completed, algos, dp_levels, seeds):
    """Generate per-class recall heatmap: classes vs DP levels, one subplot per algo."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    dp_labels = [d["label"] for d in dp_levels]
    class_names = [PTB_XL_CLASS_NAMES[c] for c in range(PX_CONFIG["num_classes"])]
    n_classes = len(class_names)
    n_algos = len(algos)

    fig, axes = plt.subplots(1, n_algos, figsize=(6 * n_algos, 4), squeeze=False)

    for col, algo in enumerate(algos):
        ax = axes[0][col]
        recall_matrix = np.zeros((n_classes, len(dp_labels)))

        for j, dp_level in enumerate(dp_levels):
            dp_label = dp_level["label"]
            for c in range(n_classes):
                cls_name = class_names[c]
                recalls = []
                for s in seeds:
                    key = "{}_{}_s{}".format(algo, dp_label, s)
                    if key in completed and "error" not in completed[key]:
                        pcm = completed[key].get("per_class_metrics", {})
                        if cls_name in pcm:
                            recalls.append(pcm[cls_name]["recall"] * 100)
                if recalls:
                    recall_matrix[c, j] = np.mean(recalls)

        im = ax.imshow(recall_matrix, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")

        for i in range(n_classes):
            for j in range(len(dp_labels)):
                color = "white" if recall_matrix[i, j] < 30 or recall_matrix[i, j] > 80 else "black"
                ax.text(
                    j, i, "{:.0f}%".format(recall_matrix[i, j]),
                    ha="center", va="center", fontsize=10,
                    color=color, fontweight="bold",
                )

        ax.set_xticks(range(len(dp_labels)))
        ax.set_xticklabels(dp_labels, fontsize=10)
        ax.set_yticks(range(n_classes))
        ax.set_yticklabels(class_names, fontsize=10)
        ax.set_xlabel("Privacy Level", fontsize=11)
        ax.set_title("{}".format(algo), fontsize=12, fontweight="bold")

    plt.suptitle("PTB-XL Per-Class Recall (%) Under Differential Privacy",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.colorbar(im, ax=axes.ravel().tolist(), label="Recall (%)", shrink=0.8)
    plt.tight_layout()

    heatmap_path = OUTPUT_DIR / "dp_ptbxl_per_class_recall.png"
    plt.savefig(heatmap_path, dpi=200, bbox_inches="tight")
    plt.savefig(str(heatmap_path).replace(".png", ".pdf"), bbox_inches="tight")
    log("Per-class heatmap saved: {}".format(heatmap_path))
    plt.close()


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="FL-EHDS Differential Privacy on PTB-XL")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 1 seed, 1 algo (FedAvg), 2 DP levels (No-DP, eps=10)")
    parser.add_argument("--fresh", action="store_true",
                        help="Delete existing checkpoint and start fresh")
    args = parser.parse_args()

    global _log_file
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _log_file = open(OUTPUT_DIR / LOG_FILE, "a")

    # Experiment grid
    seeds = [42] if args.quick else SEEDS
    algos = ["FedAvg"] if args.quick else ALGORITHMS
    dp_levels = [DP_LEVELS[0], DP_LEVELS[2]] if args.quick else DP_LEVELS  # No-DP, eps=10

    # Build experiment list
    experiments = []
    for algo in algos:
        for dp_level in dp_levels:
            for seed in seeds:
                key = "{}_{}_s{}".format(algo, dp_level["label"], seed)
                experiments.append({
                    "key": key, "algorithm": algo,
                    "dp_level": dp_level, "seed": seed,
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
                "dataset": "PTB-XL",
                "algorithms": algos,
                "dp_levels": [d["label"] for d in dp_levels],
                "seeds": seeds,
                "config": PX_CONFIG,
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
        log("Checkpoint saved. Resume: python -m benchmarks.run_dp_ptbxl")
        sys.exit(0)

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # Header
    mode = "QUICK" if args.quick else "FULL"
    log("\n" + "=" * 66)
    log("  FL-EHDS Differential Privacy on PTB-XL ({})".format(mode))
    log("  {} experiments = {} algos x {} DP x {} seeds".format(
        total_exps, len(algos), len(dp_levels), len(seeds)))
    log("=" * 66)
    log("  Device:     {}".format(_detect_device()))
    log("  Dataset:    PTB-XL (input_dim={}, num_classes={}, partition_by_site)".format(
        PX_CONFIG["input_dim"], PX_CONFIG["num_classes"]))
    log("  Algorithms: {}".format(algos))
    log("  DP levels:  {}".format([d["label"] for d in dp_levels]))
    log("  Seeds:      {}".format(seeds))
    log("  Rounds:     {}".format(PX_CONFIG["num_rounds"]))
    log("  Output:     {}".format(OUTPUT_DIR / CHECKPOINT_FILE))
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

        algo = exp["algorithm"]
        dp_level = exp["dp_level"]
        seed = exp["seed"]

        log("[{}/{}] PTB-XL {} {} s{} ...".format(
            done_count + 1, total_exps, algo, dp_level["label"], seed))

        try:
            result = run_single_experiment(
                algo, dp_level, seed, quick=args.quick)
            completed[key] = result
            done_count += 1

            acc = result["best_accuracy"] * 100
            rt = result["runtime_seconds"]

            # Per-class summary line
            pcm = result.get("per_class_metrics", {})
            macro_f1 = pcm.get("macro_avg", {}).get("f1", 0)
            log("  -> acc={:.1f}% macro-F1={:.3f} {:.0f}s".format(acc, macro_f1, rt))

            # Per-class detail
            for c in range(PX_CONFIG["num_classes"]):
                cls_name = PTB_XL_CLASS_NAMES.get(c, "Class_{}".format(c))
                if cls_name in pcm:
                    cm_cls = pcm[cls_name]
                    log("     {}: P={:.1f}% R={:.1f}% F1={:.3f} (n={})".format(
                        cls_name,
                        cm_cls["precision"] * 100,
                        cm_cls["recall"] * 100,
                        cm_cls["f1"],
                        cm_cls["support"],
                    ))

            # Save checkpoint after every experiment
            save_checkpoint(checkpoint_data)

        except Exception as e:
            log("  ERROR: {}".format(e))
            traceback.print_exc()
            completed[key] = {
                "algorithm": algo, "dp_label": dp_level["label"],
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
    # Summary table: accuracy
    # ======================================================================
    dp_labels = [d["label"] for d in dp_levels]

    log("\n  PTB-XL DP Impact — Accuracy (%)")
    log("  {:>8s} | {}".format(
        "Algo",
        " | ".join("{:<10}".format(dl) for dl in dp_labels)))
    log("  " + "-" * (12 + 13 * len(dp_labels)))

    for algo in algos:
        vals = []
        for dp_level in dp_levels:
            dp_label = dp_level["label"]
            accs = []
            for s in seeds:
                k = "{}_{}_s{}".format(algo, dp_label, s)
                if k in completed and "error" not in completed[k]:
                    accs.append(completed[k]["best_accuracy"] * 100)
            if accs:
                vals.append("{:5.1f}+-{:.1f}".format(np.mean(accs), np.std(accs)))
            else:
                vals.append("    --    ")
        log("  {:>8s} | {}".format(algo, " | ".join(vals)))

    log("  " + "-" * (12 + 13 * len(dp_labels)))

    # ======================================================================
    # Summary table: per-class macro-F1
    # ======================================================================
    log("\n  PTB-XL DP Impact — Macro-F1")
    log("  {:>8s} | {}".format(
        "Algo",
        " | ".join("{:<10}".format(dl) for dl in dp_labels)))
    log("  " + "-" * (12 + 13 * len(dp_labels)))

    for algo in algos:
        vals = []
        for dp_level in dp_levels:
            dp_label = dp_level["label"]
            f1s = []
            for s in seeds:
                k = "{}_{}_s{}".format(algo, dp_label, s)
                if k in completed and "error" not in completed[k]:
                    pcm = completed[k].get("per_class_metrics", {})
                    mf1 = pcm.get("macro_avg", {}).get("f1", 0)
                    f1s.append(mf1)
            if f1s:
                vals.append("{:.3f}+-{:.3f}".format(np.mean(f1s), np.std(f1s)))
            else:
                vals.append("    --    ")
        log("  {:>8s} | {}".format(algo, " | ".join(vals)))

    log("  " + "-" * (12 + 13 * len(dp_labels)))

    # ======================================================================
    # Summary table: per-class recall breakdown
    # ======================================================================
    log("\n  PTB-XL Per-Class Recall (%) — Mean across seeds")
    class_names = [PTB_XL_CLASS_NAMES[c] for c in range(PX_CONFIG["num_classes"])]

    for algo in algos:
        log("\n  {} :".format(algo))
        log("    {:>5s} | {}".format(
            "Class",
            " | ".join("{:<10}".format(dl) for dl in dp_labels)))
        log("    " + "-" * (9 + 13 * len(dp_labels)))
        for cls_name in class_names:
            vals = []
            for dp_level in dp_levels:
                dp_label = dp_level["label"]
                recalls = []
                for s in seeds:
                    k = "{}_{}_s{}".format(algo, dp_label, s)
                    if k in completed and "error" not in completed[k]:
                        pcm = completed[k].get("per_class_metrics", {})
                        if cls_name in pcm:
                            recalls.append(pcm[cls_name]["recall"] * 100)
                if recalls:
                    vals.append("{:5.1f}+-{:.1f}".format(np.mean(recalls), np.std(recalls)))
                else:
                    vals.append("    --    ")
            log("    {:>5s} | {}".format(cls_name, " | ".join(vals)))

    # ======================================================================
    # Per-client fairness summary
    # ======================================================================
    log("\n  PTB-XL Per-Client Accuracy Spread")
    log("  {:>8s} | {:>10s} | {:>8s} | {:>8s} | {:>8s}".format(
        "Algo", "DP", "Mean%", "Min%", "Max%"))
    log("  " + "-" * 52)

    for algo in algos:
        for dp_level in dp_levels:
            dp_label = dp_level["label"]
            all_client_accs = []
            for s in seeds:
                k = "{}_{}_s{}".format(algo, dp_label, s)
                if k in completed and "error" not in completed[k]:
                    pca = completed[k].get("per_client_acc", {})
                    all_client_accs.extend(pca.values())
            if all_client_accs:
                log("  {:>8s} | {:>10s} | {:>7.1f} | {:>7.1f} | {:>7.1f}".format(
                    algo, dp_label,
                    np.mean(all_client_accs) * 100,
                    np.min(all_client_accs) * 100,
                    np.max(all_client_accs) * 100,
                ))

    # ======================================================================
    # Generate figure
    # ======================================================================
    generate_figure(completed, algos, dp_levels, seeds)

    if _log_file:
        _log_file.close()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
FL-EHDS Experiment — Dynamic Article 71 Opt-Out (Mid-Training Withdrawal).

Simulates EHDS Article 71 dynamic opt-out: a hospital exercises its right
to withdraw from the federation DURING training, not before. This tests
a real EHDS compliance requirement — the "right to be forgotten" applied
to ongoing FL training.

Unlike the existing static opt-out experiment (which removes data samples
before training starts), this experiment:
  1. Starts training with all K clients
  2. At round R_withdraw, removes 1-2 clients entirely
  3. Continues training with remaining clients
  4. Measures: immediate accuracy drop, recovery trajectory, final accuracy

Design:
  - Datasets: PTB-XL (5 clients, 5-class, partition_by_site)
              Cardiovascular (5 clients, binary, Dirichlet alpha=0.5)
  - Algorithms: FedAvg, Ditto, HPFL
  - Withdrawal scenarios:
      1. No withdrawal         (baseline, all clients, all rounds)
      2. 1 client @round 10    (20% capacity loss mid-training)
      3. 2 clients @round 10   (40% capacity loss mid-training)
      4. 1 client @round 5     (early withdrawal)
      5. 1 client @round 20    (late withdrawal)
  - Seeds: 42, 123, 456
  - Total: 2 datasets x 3 algos x 5 scenarios x 3 seeds = 90 experiments

Key Metrics:
  - Round-by-round accuracy (full trajectory for withdrawal impact analysis)
  - Immediate drop: accuracy at R_withdraw vs R_withdraw-1
  - Recovery: does accuracy recover to pre-withdrawal levels?
  - Withdrawn client test accuracy: does the left-behind client suffer?
  - Jain fairness, DEI

Usage:
    cd fl-ehds-framework
    python -m benchmarks.run_dynamic_optout [--quick] [--fresh]

Output:
    benchmarks/paper_results_tabular/checkpoint_dynamic_optout.json
    benchmarks/paper_results_tabular/dynamic_optout_trajectory.png
    benchmarks/paper_results_tabular/dynamic_optout_impact.png

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
from typing import Dict, List, Optional, Any

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
CHECKPOINT_FILE = "checkpoint_dynamic_optout.json"
LOG_FILE = "experiment_dynamic_optout.log"

ALGORITHMS = ["FedAvg", "Ditto", "HPFL"]
SEEDS = [42, 123, 456]

PTB_XL_CLASS_NAMES = {0: "NORM", 1: "MI", 2: "STTC", 3: "CD", 4: "HYP"}

# ======================================================================
# Dataset configurations
# ======================================================================

DATASET_CONFIGS = {
    "PTB_XL": {
        "loader": "ptbxl",
        "input_dim": 9,
        "num_classes": 5,
        "num_clients": 5,
        "short": "PX",
        "class_names": PTB_XL_CLASS_NAMES,
        "training": dict(
            learning_rate=0.005,
            batch_size=64,
            num_rounds=30,
            local_epochs=3,
            mu=0.1,
        ),
    },
    "Cardiovascular": {
        "loader": "cardiovascular",
        "input_dim": 11,
        "num_classes": 2,
        "num_clients": 5,
        "short": "CV",
        "class_names": {0: "Healthy", 1: "Disease"},
        "training": dict(
            learning_rate=0.01,
            batch_size=64,
            num_rounds=25,
            local_epochs=3,
            mu=0.1,
        ),
    },
}

# ======================================================================
# Withdrawal scenarios
# ======================================================================
# Each scenario defines which clients withdraw and at which round.
# client 0 is always the one that withdraws first (simulates a hospital
# exercising Art.71). In PTB-XL this is one of the recording sites.

WITHDRAWAL_SCENARIOS = [
    {
        "label": "No-withdrawal",
        "withdraw_clients": [],
        "withdraw_round": None,
        "description": "Baseline: all clients participate for all rounds",
    },
    {
        "label": "1client@R10",
        "withdraw_clients": [0],
        "withdraw_round": 10,
        "description": "1 hospital withdraws at round 10 (20% capacity loss)",
    },
    {
        "label": "2clients@R10",
        "withdraw_clients": [0, 1],
        "withdraw_round": 10,
        "description": "2 hospitals withdraw at round 10 (40% capacity loss)",
    },
    {
        "label": "1client@R5",
        "withdraw_clients": [0],
        "withdraw_round": 5,
        "description": "1 hospital withdraws early at round 5",
    },
    {
        "label": "1client@R20",
        "withdraw_clients": [0],
        "withdraw_round": 20,
        "description": "1 hospital withdraws late at round 20",
    },
]

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
    fd, tmp = tempfile.mkstemp(dir=str(OUTPUT_DIR), prefix=".dynopt_", suffix=".tmp")
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

def load_dataset(dataset_name, num_clients, seed):
    cfg = DATASET_CONFIGS[dataset_name]
    if cfg["loader"] == "ptbxl":
        return load_ptbxl_data(
            num_clients=num_clients, seed=seed,
            partition_by_site=True, min_site_samples=50,
        )
    elif cfg["loader"] == "cardiovascular":
        return load_cardiovascular_data(
            num_clients=num_clients, seed=seed,
            is_iid=False, alpha=0.5,
        )
    else:
        raise ValueError("Unknown loader: {}".format(cfg["loader"]))


# ======================================================================
# Per-client evaluation (all clients including withdrawn)
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
# Per-class metrics
# ======================================================================

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

    all_rec = [v["recall"] for v in per_class.values()]
    all_f1 = [v["f1"] for v in per_class.values()]
    per_class["macro_avg"] = {
        "precision": round(float(np.mean([v["precision"] for v in per_class.values()])), 4),
        "recall": round(float(np.mean(all_rec)), 4),
        "f1": round(float(np.mean(all_f1)), 4),
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


# ======================================================================
# Single experiment
# ======================================================================

def run_single_experiment(dataset_name, algo, scenario, seed, quick=False):
    """Run one experiment with dynamic client withdrawal."""
    start = time.time()
    cfg = DATASET_CONFIGS[dataset_name]
    tcfg = cfg["training"]
    num_rounds = 5 if quick else tcfg["num_rounds"]
    num_clients = cfg["num_clients"]
    class_names = cfg["class_names"]

    withdraw_clients = scenario["withdraw_clients"]
    withdraw_round = scenario["withdraw_round"]

    # Adjust withdrawal round for quick mode
    if quick and withdraw_round is not None:
        # Scale proportionally: e.g., R10 of 30 -> R2 of 5
        withdraw_round = max(1, int(withdraw_round * num_rounds / tcfg["num_rounds"]))

    client_data, client_test, metadata = load_dataset(dataset_name, num_clients, seed)

    trainer = FederatedTrainer(
        num_clients=num_clients,
        algorithm=algo,
        local_epochs=tcfg["local_epochs"],
        batch_size=tcfg["batch_size"],
        learning_rate=tcfg["learning_rate"],
        mu=tcfg["mu"],
        seed=seed,
        external_data=client_data,
        external_test_data=client_test,
        input_dim=cfg["input_dim"],
        num_classes=cfg["num_classes"],
    )

    all_clients = list(range(num_clients))
    history = []
    best_acc = 0.0
    pre_withdrawal_acc = None
    post_withdrawal_acc = None
    withdrawal_happened = False

    for r in range(num_rounds):
        # Determine active clients for this round
        if withdraw_round is not None and r >= withdraw_round:
            active = [c for c in all_clients if c not in withdraw_clients]
            if not withdrawal_happened:
                withdrawal_happened = True
                # Record pre-withdrawal accuracy (from previous round)
                if history:
                    pre_withdrawal_acc = history[-1]["accuracy"]
        else:
            active = all_clients

        result = trainer.train_round(r, active_clients=active)

        # Record post-withdrawal accuracy (first round after withdrawal)
        if withdrawal_happened and post_withdrawal_acc is None:
            post_withdrawal_acc = result.global_acc

        # Per-client accuracy every round (for trajectory analysis)
        per_client_r = _evaluate_per_client(trainer)

        metrics = {
            "round": r + 1,
            "accuracy": result.global_acc,
            "loss": result.global_loss,
            "f1": result.global_f1,
            "active_clients": len(active),
            "per_client_acc": per_client_r,
        }
        history.append(metrics)
        if result.global_acc > best_acc:
            best_acc = result.global_acc

    # Final evaluation
    per_client_acc = _evaluate_per_client(trainer)
    jain = _compute_jain_index(per_client_acc)

    y_true, y_pred = _collect_predictions(trainer)
    per_class_metrics, confusion_matrix = _compute_per_class_metrics(
        y_true, y_pred, cfg["num_classes"], class_names)
    dei = _compute_dei(per_class_metrics, cfg["num_classes"], class_names)

    # Compute withdrawal impact metrics
    immediate_drop = None
    if pre_withdrawal_acc is not None and post_withdrawal_acc is not None:
        immediate_drop = post_withdrawal_acc - pre_withdrawal_acc

    # Recovery: did accuracy recover to pre-withdrawal level?
    recovery_round = None
    if pre_withdrawal_acc is not None and withdraw_round is not None:
        for h in history[withdraw_round:]:
            if h["accuracy"] >= pre_withdrawal_acc:
                recovery_round = h["round"]
                break

    # Withdrawn client(s) final accuracy
    withdrawn_client_accs = {}
    remaining_client_accs = {}
    for cid_str, acc in per_client_acc.items():
        cid = int(cid_str)
        if cid in withdraw_clients:
            withdrawn_client_accs[cid_str] = acc
        else:
            remaining_client_accs[cid_str] = acc

    elapsed = time.time() - start

    out = {
        "dataset": dataset_name,
        "algorithm": algo,
        "scenario_label": scenario["label"],
        "scenario_description": scenario["description"],
        "withdraw_clients": withdraw_clients,
        "withdraw_round": withdraw_round,
        "seed": seed,
        "history": history,
        "final_metrics": history[-1] if history else {},
        "per_client_acc": per_client_acc,
        "withdrawn_client_accs": withdrawn_client_accs,
        "remaining_client_accs": remaining_client_accs,
        "jain_index": jain,
        "per_class_metrics": per_class_metrics,
        "confusion_matrix": confusion_matrix,
        "dei": dei,
        "best_accuracy": best_acc,
        "actual_rounds": len(history),
        # Withdrawal impact metrics
        "pre_withdrawal_acc": pre_withdrawal_acc,
        "post_withdrawal_acc": post_withdrawal_acc,
        "immediate_drop": immediate_drop,
        "recovery_round": recovery_round,
        "runtime_seconds": round(elapsed, 1),
    }

    _cleanup_gpu()
    return out


# ======================================================================
# Figure generation
# ======================================================================

def generate_figures(completed, algos, scenarios, seeds, datasets):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log("matplotlib not available -- skipping figures")
        return

    # --- Figure 1: Accuracy trajectory for 1client@R10 vs baseline ---
    # One subplot per dataset, lines = algorithms, solid = baseline, dashed = withdrawal
    n_ds = len(datasets)
    fig, axes = plt.subplots(1, n_ds, figsize=(7 * n_ds, 5), squeeze=False)

    algo_colors = {"FedAvg": "#1f77b4", "Ditto": "#9467bd", "HPFL": "#2ca02c"}

    for di, ds_name in enumerate(datasets):
        ax = axes[0][di]
        ds_short = DATASET_CONFIGS[ds_name]["short"]
        num_rounds = DATASET_CONFIGS[ds_name]["training"]["num_rounds"]

        for algo in algos:
            # Baseline trajectory (mean across seeds)
            baseline_trajectories = []
            for s in seeds:
                key = "{}_{}_{}_s{}".format(ds_name, algo, "No-withdrawal", s)
                if key in completed and "error" not in completed[key]:
                    accs = [h["accuracy"] * 100 for h in completed[key]["history"]]
                    baseline_trajectories.append(accs)

            if baseline_trajectories:
                min_len = min(len(t) for t in baseline_trajectories)
                baseline_mean = np.mean(
                    [t[:min_len] for t in baseline_trajectories], axis=0)
                rounds = np.arange(1, min_len + 1)
                ax.plot(rounds, baseline_mean, color=algo_colors.get(algo, "gray"),
                        linewidth=2, label="{} (baseline)".format(algo))

            # 1client@R10 trajectory
            withdrawal_trajectories = []
            for s in seeds:
                key = "{}_{}_{}_s{}".format(ds_name, algo, "1client@R10", s)
                if key in completed and "error" not in completed[key]:
                    accs = [h["accuracy"] * 100 for h in completed[key]["history"]]
                    withdrawal_trajectories.append(accs)

            if withdrawal_trajectories:
                min_len = min(len(t) for t in withdrawal_trajectories)
                w_mean = np.mean(
                    [t[:min_len] for t in withdrawal_trajectories], axis=0)
                rounds = np.arange(1, min_len + 1)
                ax.plot(rounds, w_mean, color=algo_colors.get(algo, "gray"),
                        linewidth=2, linestyle="--",
                        label="{} (1 client out @R10)".format(algo))

        # Withdrawal marker
        ax.axvline(x=10, color="red", linestyle=":", alpha=0.7, linewidth=1.5)
        ax.text(10.3, ax.get_ylim()[0] + 1, "Art.71\nwithdrawal",
                fontsize=8, color="red", va="bottom")

        ax.set_xlabel("Round", fontsize=12)
        ax.set_ylabel("Accuracy (%)", fontsize=12)
        ax.set_title("{} ({})".format(ds_name, ds_short),
                     fontsize=13, fontweight="bold")
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(alpha=0.3)

    plt.suptitle(
        "Dynamic Art.71 Opt-Out: Training Trajectory After Hospital Withdrawal",
        fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    fig_path = OUTPUT_DIR / "dynamic_optout_trajectory.png"
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.savefig(str(fig_path).replace(".png", ".pdf"), bbox_inches="tight")
    log("Figure saved: {}".format(fig_path))
    plt.close()

    # --- Figure 2: Impact summary (grouped bar) ---
    scen_labels = [s["label"] for s in scenarios]
    n_scen = len(scen_labels)

    fig, axes = plt.subplots(1, n_ds, figsize=(7 * n_ds, 5), squeeze=False)
    colors = {
        "No-withdrawal": "#2ca02c",
        "1client@R10": "#ff7f0e",
        "2clients@R10": "#d62728",
        "1client@R5": "#1f77b4",
        "1client@R20": "#9467bd",
    }

    for di, ds_name in enumerate(datasets):
        ax = axes[0][di]
        ds_short = DATASET_CONFIGS[ds_name]["short"]

        bar_width = 0.14
        group_width = n_scen * bar_width + 0.15
        group_positions = np.arange(len(algos)) * group_width

        for j, scen_label in enumerate(scen_labels):
            means = []
            stds_list = []
            for algo in algos:
                accs = []
                for s in seeds:
                    key = "{}_{}_{}_s{}".format(ds_name, algo, scen_label, s)
                    if key in completed and "error" not in completed[key]:
                        accs.append(completed[key]["best_accuracy"] * 100)
                if accs:
                    means.append(np.mean(accs))
                    stds_list.append(np.std(accs))
                else:
                    means.append(0)
                    stds_list.append(0)

            x_pos = group_positions + j * bar_width
            bars = ax.bar(x_pos, means, bar_width,
                          yerr=stds_list, capsize=2,
                          label=scen_label if di == 0 else "",
                          color=colors.get(scen_label, "gray"),
                          edgecolor="black", linewidth=0.5, alpha=0.85)
            for bar_obj, m in zip(bars, means):
                if m > 0:
                    ax.text(bar_obj.get_x() + bar_obj.get_width() / 2.0,
                            bar_obj.get_height() + 0.3,
                            "{:.1f}".format(m),
                            ha="center", va="bottom", fontsize=6,
                            fontweight="bold")

        ax.set_xlabel("Algorithm", fontsize=12)
        ax.set_ylabel("Best Accuracy (%)", fontsize=12)
        ax.set_title("{} ({})".format(ds_name, ds_short),
                     fontsize=13, fontweight="bold")
        ax.set_xticks(group_positions + (n_scen - 1) * bar_width / 2.0)
        ax.set_xticklabels(algos, fontsize=11)
        ax.grid(axis="y", alpha=0.3)

    # Single legend for all subplots
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=n_scen,
               fontsize=9, bbox_to_anchor=(0.5, -0.02))

    plt.suptitle(
        "Dynamic Art.71 Opt-Out: Accuracy Impact by Withdrawal Scenario",
        fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    fig_path2 = OUTPUT_DIR / "dynamic_optout_impact.png"
    plt.savefig(fig_path2, dpi=200, bbox_inches="tight")
    plt.savefig(str(fig_path2).replace(".png", ".pdf"), bbox_inches="tight")
    log("Figure saved: {}".format(fig_path2))
    plt.close()


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="FL-EHDS Dynamic Article 71 Opt-Out")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 1 seed, 1 algo, 2 scenarios, 1 dataset")
    parser.add_argument("--fresh", action="store_true",
                        help="Delete existing checkpoint and start fresh")
    args = parser.parse_args()

    global _log_file
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _log_file = open(OUTPUT_DIR / LOG_FILE, "a")

    # Experiment grid
    seeds = [42] if args.quick else SEEDS
    algos = ["FedAvg"] if args.quick else ALGORITHMS
    datasets = ["PTB_XL"] if args.quick else list(DATASET_CONFIGS.keys())
    if args.quick:
        scenarios = [WITHDRAWAL_SCENARIOS[0], WITHDRAWAL_SCENARIOS[1]]
    else:
        scenarios = WITHDRAWAL_SCENARIOS

    # Build experiment list
    experiments = []
    for ds_name in datasets:
        for algo in algos:
            for scenario in scenarios:
                for seed in seeds:
                    key = "{}_{}_{}_s{}".format(
                        ds_name, algo, scenario["label"], seed)
                    experiments.append({
                        "key": key, "dataset": ds_name,
                        "algorithm": algo, "scenario": scenario,
                        "seed": seed,
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
                "experiment_type": "dynamic_art71_optout",
                "datasets": datasets,
                "algorithms": algos,
                "scenarios": [s["label"] for s in scenarios],
                "scenario_details": [
                    {"label": s["label"], "description": s["description"],
                     "withdraw_clients": s["withdraw_clients"],
                     "withdraw_round": s["withdraw_round"]}
                    for s in scenarios
                ],
                "seeds": seeds,
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
        log("Checkpoint saved. Resume: python -m benchmarks.run_dynamic_optout")
        sys.exit(0)

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # Header
    mode = "QUICK" if args.quick else "FULL"
    log("\n" + "=" * 70)
    log("  FL-EHDS Dynamic Art.71 Opt-Out ({})".format(mode))
    log("  {} experiments = {} datasets x {} algos x {} scenarios x {} seeds".format(
        total_exps, len(datasets), len(algos), len(scenarios), len(seeds)))
    log("=" * 70)
    log("  Device:     {}".format(_detect_device()))
    log("  Datasets:   {}".format(datasets))
    log("  Algorithms: {}".format(algos))
    log("  Scenarios:")
    for s in scenarios:
        wc = s["withdraw_clients"] if s["withdraw_clients"] else "none"
        wr = s["withdraw_round"] if s["withdraw_round"] else "N/A"
        log("    {} — withdraw={} @R{} — {}".format(
            s["label"], wc, wr, s["description"]))
    log("  Seeds:      {}".format(seeds))
    log("  Output:     {}".format(OUTPUT_DIR / CHECKPOINT_FILE))
    log("=" * 70)

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

        ds_name = exp["dataset"]
        algo = exp["algorithm"]
        scenario = exp["scenario"]
        seed = exp["seed"]
        ds_short = DATASET_CONFIGS[ds_name]["short"]

        log("[{}/{}] {} {} {} s{} ...".format(
            done_count + 1, total_exps, ds_short, algo,
            scenario["label"], seed))

        try:
            result = run_single_experiment(
                ds_name, algo, scenario, seed, quick=args.quick)
            completed[key] = result
            done_count += 1

            acc = result["best_accuracy"] * 100
            rt = result["runtime_seconds"]
            jain = result.get("jain_index", 0)
            dei = result.get("dei", 0)
            imm_drop = result.get("immediate_drop")
            recovery = result.get("recovery_round")

            drop_str = "{:+.1f}pp".format(imm_drop * 100) if imm_drop is not None else "N/A"
            rec_str = "R{}".format(recovery) if recovery else "never"

            log("  -> acc={:.1f}% Jain={:.3f} DEI={:.3f} drop={} rec={} {:.0f}s".format(
                acc, jain, dei, drop_str, rec_str, rt))

            # Per-client summary
            pca = result.get("per_client_acc", {})
            wc = result.get("withdrawn_client_accs", {})
            rc = result.get("remaining_client_accs", {})
            if wc:
                w_mean = np.mean(list(wc.values())) * 100
                r_mean = np.mean(list(rc.values())) * 100 if rc else 0
                log("     Withdrawn clients: {:.0f}% | Remaining: {:.0f}%".format(
                    w_mean, r_mean))

            save_checkpoint(checkpoint_data)

        except Exception as e:
            log("  ERROR: {}".format(e))
            traceback.print_exc()
            completed[key] = {
                "dataset": ds_name, "algorithm": algo,
                "scenario_label": scenario["label"],
                "seed": seed, "error": str(e),
            }
            save_checkpoint(checkpoint_data)
            _cleanup_gpu()

    # Finalize
    checkpoint_data["metadata"]["end_time"] = datetime.now().isoformat()
    checkpoint_data["metadata"]["total_elapsed"] = time.time() - global_start
    save_checkpoint(checkpoint_data)

    elapsed = time.time() - global_start
    log("\n" + "=" * 70)
    log("  COMPLETED: {}/{}".format(done_count, total_exps))
    log("  Total time: {}".format(timedelta(seconds=int(elapsed))))
    log("=" * 70)

    # ======================================================================
    # Summary tables
    # ======================================================================
    scen_labels = [s["label"] for s in scenarios]

    for ds_name in datasets:
        ds_short = DATASET_CONFIGS[ds_name]["short"]
        log("\n  === {} ({}) — Best Accuracy (%) ===".format(ds_name, ds_short))
        log("  {:>8s} | {}".format(
            "Algo",
            " | ".join("{:<16}".format(sl) for sl in scen_labels)))
        log("  " + "-" * (12 + 19 * len(scen_labels)))

        for algo in algos:
            vals = []
            for scen_label in scen_labels:
                accs = []
                for s in seeds:
                    k = "{}_{}_{}_s{}".format(ds_name, algo, scen_label, s)
                    if k in completed and "error" not in completed[k]:
                        accs.append(completed[k]["best_accuracy"] * 100)
                if accs:
                    vals.append("{:5.1f}+-{:.1f}      ".format(
                        np.mean(accs), np.std(accs)))
                else:
                    vals.append("    --          ")
            log("  {:>8s} | {}".format(algo, " | ".join(vals)))

        # Withdrawal impact table
        log("\n  === {} ({}) — Immediate Accuracy Drop (pp) ===".format(
            ds_name, ds_short))
        withdrawal_scens = [s for s in scenarios if s["withdraw_clients"]]
        w_labels = [s["label"] for s in withdrawal_scens]
        log("  {:>8s} | {}".format(
            "Algo",
            " | ".join("{:<16}".format(wl) for wl in w_labels)))
        log("  " + "-" * (12 + 19 * len(w_labels)))

        for algo in algos:
            vals = []
            for ws in withdrawal_scens:
                drops = []
                for s in seeds:
                    k = "{}_{}_{}_s{}".format(ds_name, algo, ws["label"], s)
                    if k in completed and "error" not in completed[k]:
                        d = completed[k].get("immediate_drop")
                        if d is not None:
                            drops.append(d * 100)
                if drops:
                    vals.append("{:+5.1f}+-{:.1f}     ".format(
                        np.mean(drops), np.std(drops)))
                else:
                    vals.append("    --          ")
            log("  {:>8s} | {}".format(algo, " | ".join(vals)))

        # Recovery table
        log("\n  === {} ({}) — Recovery Round ===".format(ds_name, ds_short))
        log("  {:>8s} | {}".format(
            "Algo",
            " | ".join("{:<16}".format(wl) for wl in w_labels)))
        log("  " + "-" * (12 + 19 * len(w_labels)))

        for algo in algos:
            vals = []
            for ws in withdrawal_scens:
                recs = []
                nevers = 0
                for s in seeds:
                    k = "{}_{}_{}_s{}".format(ds_name, algo, ws["label"], s)
                    if k in completed and "error" not in completed[k]:
                        r = completed[k].get("recovery_round")
                        if r is not None:
                            recs.append(r)
                        else:
                            nevers += 1
                if recs or nevers:
                    if recs:
                        rec_str = "R{:.0f}".format(np.mean(recs))
                    else:
                        rec_str = ""
                    if nevers > 0:
                        rec_str += " ({} never)".format(nevers)
                    vals.append("{:<16}".format(rec_str.strip()))
                else:
                    vals.append("    --          ")
            log("  {:>8s} | {}".format(algo, " | ".join(vals)))

        # Withdrawn vs remaining client accuracy
        log("\n  === {} ({}) — Withdrawn vs Remaining Client Acc (%) ===".format(
            ds_name, ds_short))
        log("  {:>8s} | {:>16s} | {:>10s} | {:>10s} | {:>8s}".format(
            "Algo", "Scenario", "Withdrawn", "Remaining", "Gap"))
        log("  " + "-" * 65)

        for algo in algos:
            for ws in withdrawal_scens:
                w_accs = []
                r_accs = []
                for s in seeds:
                    k = "{}_{}_{}_s{}".format(ds_name, algo, ws["label"], s)
                    if k in completed and "error" not in completed[k]:
                        wc = completed[k].get("withdrawn_client_accs", {})
                        rc = completed[k].get("remaining_client_accs", {})
                        if wc:
                            w_accs.append(np.mean(list(wc.values())) * 100)
                        if rc:
                            r_accs.append(np.mean(list(rc.values())) * 100)
                if w_accs and r_accs:
                    w_m = np.mean(w_accs)
                    r_m = np.mean(r_accs)
                    log("  {:>8s} | {:>16s} | {:>9.1f}% | {:>9.1f}% | {:>+7.1f}pp".format(
                        algo, ws["label"], w_m, r_m, w_m - r_m))

    # ======================================================================
    # Generate figures
    # ======================================================================
    generate_figures(completed, algos, scenarios, seeds, datasets)

    if _log_file:
        _log_file.close()


if __name__ == "__main__":
    main()

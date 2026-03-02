#!/usr/bin/env python3
"""
FL-EHDS Experiment — Cross-Border Heterogeneous Differential Privacy.

Simulates the EHDS Art.50 cross-border DP scenario described in the paper's
Discussion (Sec. 5.1): different Member States enforce different epsilon
ceilings (e.g. German BDSG eps=1 vs Italian Garante eps=10).

Tests per-client local DP with heterogeneous budgets on PTB-XL, the only
dataset with natural European multi-site partitioning (52 recording sites).

Design:
  - Dataset: PTB-XL (5-class ECG, 9 features, partition_by_site)
  - Algorithms: FedAvg, Ditto, HPFL
  - Scenarios:
      1. No-DP                (upper bound)
      2. Uniform eps=1        (strictest-wins rule)
      3. Uniform eps=10       (permissive rule)
      4. Mixed: 2xDE + 3xIT  (2 clients eps=1, 3 clients eps=10)
      5. Gradient: 5 EU       (eps = 1, 3, 5, 7, 10)
  - Seeds: 42, 123, 456
  - Total: 3 algos x 5 scenarios x 3 seeds = 45 experiments (~30-45 min)

Key Innovation:
  Uses LOCAL DP mode where each client adds Gaussian noise proportional
  to their own jurisdiction's epsilon ceiling. This differs from central DP
  (uniform noise on aggregated model) and directly models the EHDS cross-
  border heterogeneity problem.

Usage:
    cd fl-ehds-framework
    python -m benchmarks.run_crossborder_dp [--quick] [--fresh]

Output:
    benchmarks/paper_results_tabular/checkpoint_crossborder_dp.json
    benchmarks/paper_results_tabular/crossborder_dp_accuracy.png
    benchmarks/paper_results_tabular/crossborder_dp_fairness.png

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

# ======================================================================
# Constants
# ======================================================================

OUTPUT_DIR = FRAMEWORK_DIR / "benchmarks" / "paper_results_tabular"
CHECKPOINT_FILE = "checkpoint_crossborder_dp.json"
LOG_FILE = "experiment_crossborder_dp.log"

ALGORITHMS = ["FedAvg", "Ditto", "HPFL"]
SEEDS = [42, 123, 456]

# PTB-XL configuration (consistent with other experiments)
PX_CONFIG = dict(
    input_dim=9, num_classes=5,
    num_clients=5,
    learning_rate=0.005,
    batch_size=64,
    num_rounds=30,
    local_epochs=3,
    mu=0.1,
    dp_clip_norm=1.0,
)

PTB_XL_CLASS_NAMES = {0: "NORM", 1: "MI", 2: "STTC", 3: "CD", 4: "HYP"}

# ======================================================================
# Cross-border DP scenarios
# ======================================================================
# Each scenario assigns a per-client epsilon. Local DP mode: each client
# adds Gaussian noise proportional to C * sqrt(2*ln(1.25/delta)) / eps_i
# before sending updates to the server.

JURISDICTION_LABELS = {
    0: "DE (Berlin)", 1: "DE (Munich)", 2: "IT (Rome)",
    3: "IT (Milan)", 4: "IT (Naples)",
}

CROSSBORDER_SCENARIOS = [
    {
        "label": "No-DP",
        "dp_enabled": False,
        "client_epsilons": {0: 0, 1: 0, 2: 0, 3: 0, 4: 0},
        "jurisdictions": {0: "DE", 1: "DE", 2: "IT", 3: "IT", 4: "IT"},
        "description": "No differential privacy (upper bound)",
    },
    {
        "label": "Uniform-eps1",
        "dp_enabled": True,
        "client_epsilons": {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0},
        "jurisdictions": {0: "DE", 1: "DE", 2: "DE", 3: "DE", 4: "DE"},
        "description": "All clients eps=1 (strictest-wins: German BDSG for all)",
    },
    {
        "label": "Uniform-eps10",
        "dp_enabled": True,
        "client_epsilons": {0: 10.0, 1: 10.0, 2: 10.0, 3: 10.0, 4: 10.0},
        "jurisdictions": {0: "IT", 1: "IT", 2: "IT", 3: "IT", 4: "IT"},
        "description": "All clients eps=10 (permissive: Italian Garante for all)",
    },
    {
        "label": "Mixed-2DE-3IT",
        "dp_enabled": True,
        "client_epsilons": {0: 1.0, 1: 1.0, 2: 10.0, 3: 10.0, 4: 10.0},
        "jurisdictions": {0: "DE", 1: "DE", 2: "IT", 3: "IT", 4: "IT"},
        "description": "2 German (eps=1) + 3 Italian (eps=10) — realistic EHDS",
    },
    {
        "label": "Gradient-5EU",
        "dp_enabled": True,
        "client_epsilons": {0: 1.0, 1: 3.0, 2: 5.0, 3: 7.0, 4: 10.0},
        "jurisdictions": {0: "DE", 1: "AT", 2: "FR", 3: "NL", 4: "IT"},
        "description": "5 EU countries: DE=1, AT=3, FR=5, NL=7, IT=10",
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
    fd, tmp = tempfile.mkstemp(dir=str(OUTPUT_DIR), prefix=".xborder_", suffix=".tmp")
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


def _compute_per_class_metrics(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1

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

    all_prec = [v["precision"] for v in per_class.values()]
    all_rec = [v["recall"] for v in per_class.values()]
    all_f1 = [v["f1"] for v in per_class.values()]
    per_class["macro_avg"] = {
        "precision": round(float(np.mean(all_prec)), 4),
        "recall": round(float(np.mean(all_rec)), 4),
        "f1": round(float(np.mean(all_f1)), 4),
    }

    return per_class, cm.tolist()


def _compute_dei(per_class_metrics, num_classes):
    """Compute Diagnostic Equity Index: DEI = min_c(R_c) * (1 - CV(R))."""
    recalls = []
    for c in range(num_classes):
        cls_name = PTB_XL_CLASS_NAMES.get(c, "Class_{}".format(c))
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
    """Compute Jain fairness index from per-client accuracy dict."""
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

def run_single_experiment(algo, scenario, seed, quick=False):
    """Run one experiment with per-client heterogeneous DP."""
    start = time.time()
    num_rounds = 5 if quick else PX_CONFIG["num_rounds"]

    client_epsilons = scenario["client_epsilons"]
    dp_enabled = scenario["dp_enabled"]

    client_data, client_test, metadata = load_dataset(seed)

    # For non-DP: just use default settings
    # For DP: use local mode so each client adds its own noise
    if dp_enabled:
        # Use the minimum epsilon as default (will be overridden per-client)
        base_epsilon = min(e for e in client_epsilons.values() if e > 0)
        trainer = FederatedTrainer(
            num_clients=PX_CONFIG["num_clients"],
            algorithm=algo,
            local_epochs=PX_CONFIG["local_epochs"],
            batch_size=PX_CONFIG["batch_size"],
            learning_rate=PX_CONFIG["learning_rate"],
            mu=PX_CONFIG["mu"],
            dp_enabled=True,
            dp_epsilon=base_epsilon,
            dp_clip_norm=PX_CONFIG["dp_clip_norm"],
            dp_mode="local",
            seed=seed,
            external_data=client_data,
            external_test_data=client_test,
            input_dim=PX_CONFIG["input_dim"],
            num_classes=PX_CONFIG["num_classes"],
        )
    else:
        trainer = FederatedTrainer(
            num_clients=PX_CONFIG["num_clients"],
            algorithm=algo,
            local_epochs=PX_CONFIG["local_epochs"],
            batch_size=PX_CONFIG["batch_size"],
            learning_rate=PX_CONFIG["learning_rate"],
            mu=PX_CONFIG["mu"],
            dp_enabled=False,
            seed=seed,
            external_data=client_data,
            external_test_data=client_test,
            input_dim=PX_CONFIG["input_dim"],
            num_classes=PX_CONFIG["num_classes"],
        )

    # For heterogeneous DP: monkey-patch _train_client to set per-client epsilon
    is_heterogeneous = dp_enabled and len(set(client_epsilons.values())) > 1
    if is_heterogeneous:
        original_train_client = trainer._train_client

        def make_patched(orig, eps_map, trn):
            def patched_train_client(client_id, round_num):
                old_eps = trn.dp_epsilon
                trn.dp_epsilon = eps_map[client_id]
                try:
                    result = orig(client_id, round_num)
                finally:
                    trn.dp_epsilon = old_eps
                return result
            return patched_train_client

        trainer._train_client = make_patched(
            original_train_client, client_epsilons, trainer)

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
    jain = _compute_jain_index(per_client_acc)

    # Per-class metrics for 5 PTB-XL classes
    y_true, y_pred = _collect_predictions(trainer)
    per_class_metrics, confusion_matrix = _compute_per_class_metrics(
        y_true, y_pred, PX_CONFIG["num_classes"])

    # Diagnostic Equity Index
    dei = _compute_dei(per_class_metrics, PX_CONFIG["num_classes"])

    elapsed = time.time() - start

    out = {
        "algorithm": algo,
        "scenario_label": scenario["label"],
        "scenario_description": scenario["description"],
        "dp_enabled": dp_enabled,
        "client_epsilons": {str(k): v for k, v in client_epsilons.items()},
        "jurisdictions": scenario["jurisdictions"],
        "seed": seed,
        "history": history,
        "final_metrics": history[-1] if history else {},
        "per_client_acc": per_client_acc,
        "jain_index": jain,
        "per_class_metrics": per_class_metrics,
        "confusion_matrix": confusion_matrix,
        "dei": dei,
        "best_accuracy": best_acc,
        "actual_rounds": len(history),
        "runtime_seconds": round(elapsed, 1),
    }

    _cleanup_gpu()
    return out


# ======================================================================
# Figure generation
# ======================================================================

def generate_figures(completed, algos, scenarios, seeds):
    """Generate two figures: accuracy comparison and fairness analysis."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log("matplotlib not available -- skipping figures")
        return

    scenario_labels = [s["label"] for s in scenarios]
    n_algos = len(algos)
    n_scen = len(scenario_labels)

    # --- Figure 1: Grouped bar chart (accuracy by scenario x algorithm) ---
    colors = {
        "No-DP": "#2ca02c",
        "Uniform-eps1": "#d62728",
        "Uniform-eps10": "#1f77b4",
        "Mixed-2DE-3IT": "#ff7f0e",
        "Gradient-5EU": "#9467bd",
    }

    fig, ax = plt.subplots(figsize=(12, 6))

    bar_width = 0.14
    group_width = n_scen * bar_width + 0.15
    group_positions = np.arange(n_algos) * group_width

    for j, scen_label in enumerate(scenario_labels):
        means = []
        stds = []
        for algo in algos:
            accs = []
            for s in seeds:
                key = "{}_{}_s{}".format(algo, scen_label, s)
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
            label=scen_label,
            color=colors.get(scen_label, "gray"),
            edgecolor="black", linewidth=0.5, alpha=0.85,
        )
        for bar_obj, m in zip(bars, means):
            if m > 0:
                ax.text(
                    bar_obj.get_x() + bar_obj.get_width() / 2.0,
                    bar_obj.get_height() + 0.5,
                    "{:.1f}".format(m),
                    ha="center", va="bottom", fontsize=7, fontweight="bold",
                )

    ax.set_xlabel("Algorithm", fontsize=13)
    ax.set_ylabel("Accuracy (%)", fontsize=13)
    ax.set_title(
        "Cross-Border Heterogeneous DP on PTB-XL (5-class ECG)\n"
        "EHDS Art.50: Per-Client Local DP with Jurisdiction-Specific Epsilon",
        fontsize=13, fontweight="bold",
    )
    ax.set_xticks(group_positions + (n_scen - 1) * bar_width / 2.0)
    ax.set_xticklabels(algos, fontsize=12)
    ax.legend(fontsize=9, title="DP Scenario", title_fontsize=10,
              loc="lower left")
    ax.grid(axis="y", alpha=0.3)

    all_accs = [
        r["best_accuracy"] * 100
        for r in completed.values()
        if "error" not in r and "best_accuracy" in r
    ]
    if all_accs:
        y_min = max(0, min(all_accs) - 10)
        y_max = min(100, max(all_accs) + 5)
        ax.set_ylim(y_min, y_max)

    fig.tight_layout()
    fig_path = OUTPUT_DIR / "crossborder_dp_accuracy.png"
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.savefig(str(fig_path).replace(".png", ".pdf"), bbox_inches="tight")
    log("Figure saved: {}".format(fig_path))
    plt.close()

    # --- Figure 2: Per-client fairness (Jain + DEI) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Jain index
    bar_width2 = 0.14
    for j, scen_label in enumerate(scenario_labels):
        vals = []
        for algo in algos:
            jains = []
            for s in seeds:
                key = "{}_{}_s{}".format(algo, scen_label, s)
                if key in completed and "error" not in completed[key]:
                    jains.append(completed[key].get("jain_index", 0))
            vals.append(np.mean(jains) if jains else 0)
        x_pos = np.arange(n_algos) * group_width + j * bar_width2
        ax1.bar(x_pos, vals, bar_width2,
                label=scen_label, color=colors.get(scen_label, "gray"),
                edgecolor="black", linewidth=0.5, alpha=0.85)

    ax1.set_xlabel("Algorithm", fontsize=12)
    ax1.set_ylabel("Jain Fairness Index", fontsize=12)
    ax1.set_title("Per-Client Fairness", fontsize=12, fontweight="bold")
    ax1.set_xticks(np.arange(n_algos) * group_width + (n_scen - 1) * bar_width2 / 2.0)
    ax1.set_xticklabels(algos, fontsize=11)
    ax1.set_ylim(0, 1.05)
    ax1.legend(fontsize=8, title="Scenario", ncol=2)
    ax1.grid(axis="y", alpha=0.3)

    # DEI
    for j, scen_label in enumerate(scenario_labels):
        vals = []
        for algo in algos:
            deis = []
            for s in seeds:
                key = "{}_{}_s{}".format(algo, scen_label, s)
                if key in completed and "error" not in completed[key]:
                    deis.append(completed[key].get("dei", 0))
            vals.append(np.mean(deis) if deis else 0)
        x_pos = np.arange(n_algos) * group_width + j * bar_width2
        ax2.bar(x_pos, vals, bar_width2,
                label=scen_label, color=colors.get(scen_label, "gray"),
                edgecolor="black", linewidth=0.5, alpha=0.85)

    ax2.set_xlabel("Algorithm", fontsize=12)
    ax2.set_ylabel("Diagnostic Equity Index (DEI)", fontsize=12)
    ax2.set_title("Diagnostic Equity Under Cross-Border DP", fontsize=12,
                  fontweight="bold")
    ax2.set_xticks(np.arange(n_algos) * group_width + (n_scen - 1) * bar_width2 / 2.0)
    ax2.set_xticklabels(algos, fontsize=11)
    ax2.set_ylim(0, 1.05)
    ax2.legend(fontsize=8, title="Scenario", ncol=2)
    ax2.grid(axis="y", alpha=0.3)

    plt.suptitle("Cross-Border DP: Fairness & Equity Analysis",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    fig_path2 = OUTPUT_DIR / "crossborder_dp_fairness.png"
    plt.savefig(fig_path2, dpi=200, bbox_inches="tight")
    plt.savefig(str(fig_path2).replace(".png", ".pdf"), bbox_inches="tight")
    log("Figure saved: {}".format(fig_path2))
    plt.close()

    # --- Figure 3: Per-class recall heatmap for Mixed scenario ---
    _generate_mixed_heatmap(completed, algos, seeds)


def _generate_mixed_heatmap(completed, algos, seeds):
    """Heatmap: per-class recall for each algo under No-DP vs Mixed vs Uniform-eps1."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    compare_scenarios = ["No-DP", "Mixed-2DE-3IT", "Uniform-eps1"]
    class_names = [PTB_XL_CLASS_NAMES[c] for c in range(PX_CONFIG["num_classes"])]
    n_classes = len(class_names)
    n_algos = len(algos)

    fig, axes = plt.subplots(1, n_algos, figsize=(6 * n_algos, 4), squeeze=False)

    for col, algo in enumerate(algos):
        ax = axes[0][col]
        recall_matrix = np.zeros((n_classes, len(compare_scenarios)))

        for j, scen_label in enumerate(compare_scenarios):
            for c in range(n_classes):
                cls_name = class_names[c]
                recalls = []
                for s in seeds:
                    key = "{}_{}_s{}".format(algo, scen_label, s)
                    if key in completed and "error" not in completed[key]:
                        pcm = completed[key].get("per_class_metrics", {})
                        if cls_name in pcm:
                            recalls.append(pcm[cls_name]["recall"] * 100)
                if recalls:
                    recall_matrix[c, j] = np.mean(recalls)

        im = ax.imshow(recall_matrix, cmap="RdYlGn", vmin=0, vmax=100,
                       aspect="auto")

        for i in range(n_classes):
            for j in range(len(compare_scenarios)):
                color = ("white" if recall_matrix[i, j] < 30
                         or recall_matrix[i, j] > 80 else "black")
                ax.text(
                    j, i, "{:.0f}%".format(recall_matrix[i, j]),
                    ha="center", va="center", fontsize=10,
                    color=color, fontweight="bold",
                )

        ax.set_xticks(range(len(compare_scenarios)))
        ax.set_xticklabels(compare_scenarios, fontsize=9, rotation=15,
                           ha="right")
        ax.set_yticks(range(n_classes))
        ax.set_yticklabels(class_names, fontsize=10)
        ax.set_title("{}".format(algo), fontsize=12, fontweight="bold")

    plt.suptitle(
        "PTB-XL Per-Class Recall: Cross-Border Mixed DP vs Baselines",
        fontsize=13, fontweight="bold", y=1.02,
    )
    plt.colorbar(im, ax=axes.ravel().tolist(), label="Recall (%)", shrink=0.8)
    plt.tight_layout()

    fig_path = OUTPUT_DIR / "crossborder_dp_per_class.png"
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.savefig(str(fig_path).replace(".png", ".pdf"), bbox_inches="tight")
    log("Per-class heatmap saved: {}".format(fig_path))
    plt.close()


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="FL-EHDS Cross-Border Heterogeneous DP on PTB-XL")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 1 seed, 1 algo, 2 scenarios")
    parser.add_argument("--fresh", action="store_true",
                        help="Delete existing checkpoint and start fresh")
    args = parser.parse_args()

    global _log_file
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _log_file = open(OUTPUT_DIR / LOG_FILE, "a")

    # Experiment grid
    seeds = [42] if args.quick else SEEDS
    algos = ["FedAvg"] if args.quick else ALGORITHMS
    if args.quick:
        scenarios = [CROSSBORDER_SCENARIOS[0], CROSSBORDER_SCENARIOS[3]]  # No-DP, Mixed
    else:
        scenarios = CROSSBORDER_SCENARIOS

    # Build experiment list
    experiments = []
    for algo in algos:
        for scenario in scenarios:
            for seed in seeds:
                key = "{}_{}_s{}".format(algo, scenario["label"], seed)
                experiments.append({
                    "key": key, "algorithm": algo,
                    "scenario": scenario, "seed": seed,
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
                "experiment_type": "cross_border_heterogeneous_dp",
                "algorithms": algos,
                "scenarios": [s["label"] for s in scenarios],
                "scenario_details": [
                    {"label": s["label"], "description": s["description"],
                     "client_epsilons": s["client_epsilons"],
                     "jurisdictions": s["jurisdictions"]}
                    for s in scenarios
                ],
                "seeds": seeds,
                "config": PX_CONFIG,
                "dp_mode": "local",
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
        log("Checkpoint saved. Resume: python -m benchmarks.run_crossborder_dp")
        sys.exit(0)

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # Header
    mode = "QUICK" if args.quick else "FULL"
    log("\n" + "=" * 70)
    log("  FL-EHDS Cross-Border Heterogeneous DP on PTB-XL ({})".format(mode))
    log("  {} experiments = {} algos x {} scenarios x {} seeds".format(
        total_exps, len(algos), len(scenarios), len(seeds)))
    log("=" * 70)
    log("  Device:     {}".format(_detect_device()))
    log("  Dataset:    PTB-XL (5-class ECG, partition_by_site)")
    log("  DP mode:    LOCAL (per-client noise)")
    log("  Algorithms: {}".format(algos))
    log("  Scenarios:")
    for s in scenarios:
        eps_str = ", ".join(
            "C{}={}".format(k, v) for k, v in sorted(s["client_epsilons"].items()))
        log("    {} — {} [{}]".format(s["label"], s["description"], eps_str))
    log("  Seeds:      {}".format(seeds))
    log("  Rounds:     {}".format(PX_CONFIG["num_rounds"]))
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

        algo = exp["algorithm"]
        scenario = exp["scenario"]
        seed = exp["seed"]

        log("[{}/{}] {} {} s{} ...".format(
            done_count + 1, total_exps, algo, scenario["label"], seed))

        try:
            result = run_single_experiment(
                algo, scenario, seed, quick=args.quick)
            completed[key] = result
            done_count += 1

            acc = result["best_accuracy"] * 100
            rt = result["runtime_seconds"]
            jain = result.get("jain_index", 0)
            dei = result.get("dei", 0)

            # Per-class summary
            pcm = result.get("per_class_metrics", {})
            macro_f1 = pcm.get("macro_avg", {}).get("f1", 0)
            log("  -> acc={:.1f}% F1={:.3f} Jain={:.3f} DEI={:.3f} {:.0f}s".format(
                acc, macro_f1, jain, dei, rt))

            # Per-class detail (compact)
            class_strs = []
            for c in range(PX_CONFIG["num_classes"]):
                cls_name = PTB_XL_CLASS_NAMES.get(c, str(c))
                if cls_name in pcm:
                    class_strs.append("{}:{:.0f}%".format(
                        cls_name, pcm[cls_name]["recall"] * 100))
            log("     Recall: {}".format(" | ".join(class_strs)))

            # Per-client accuracy (with jurisdiction)
            pca = result.get("per_client_acc", {})
            client_strs = []
            for cid in range(PX_CONFIG["num_clients"]):
                jur = scenario["jurisdictions"].get(cid, "??")
                eps = scenario["client_epsilons"].get(cid, 0)
                cacc = pca.get(str(cid), 0)
                client_strs.append("C{}({}:e{})={:.0f}%".format(
                    cid, jur, eps, cacc * 100))
            log("     Clients: {}".format(" ".join(client_strs)))

            save_checkpoint(checkpoint_data)

        except Exception as e:
            log("  ERROR: {}".format(e))
            traceback.print_exc()
            completed[key] = {
                "algorithm": algo, "scenario_label": scenario["label"],
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

    # Accuracy table
    log("\n  Cross-Border DP — Accuracy (%)")
    header = "  {:>8s} | {}".format(
        "Algo",
        " | ".join("{:<14}".format(sl) for sl in scen_labels))
    log(header)
    log("  " + "-" * (12 + 17 * len(scen_labels)))

    for algo in algos:
        vals = []
        for scen_label in scen_labels:
            accs = []
            for s in seeds:
                k = "{}_{}_s{}".format(algo, scen_label, s)
                if k in completed and "error" not in completed[k]:
                    accs.append(completed[k]["best_accuracy"] * 100)
            if accs:
                vals.append("{:5.1f}+-{:.1f}   ".format(
                    np.mean(accs), np.std(accs)))
            else:
                vals.append("    --        ")
        log("  {:>8s} | {}".format(algo, " | ".join(vals)))

    # Jain fairness table
    log("\n  Cross-Border DP — Jain Fairness Index")
    log("  {:>8s} | {}".format(
        "Algo",
        " | ".join("{:<14}".format(sl) for sl in scen_labels)))
    log("  " + "-" * (12 + 17 * len(scen_labels)))

    for algo in algos:
        vals = []
        for scen_label in scen_labels:
            jains = []
            for s in seeds:
                k = "{}_{}_s{}".format(algo, scen_label, s)
                if k in completed and "error" not in completed[k]:
                    jains.append(completed[k].get("jain_index", 0))
            if jains:
                vals.append("{:.3f}+-{:.3f}  ".format(
                    np.mean(jains), np.std(jains)))
            else:
                vals.append("    --        ")
        log("  {:>8s} | {}".format(algo, " | ".join(vals)))

    # DEI table
    log("\n  Cross-Border DP — Diagnostic Equity Index (DEI)")
    log("  {:>8s} | {}".format(
        "Algo",
        " | ".join("{:<14}".format(sl) for sl in scen_labels)))
    log("  " + "-" * (12 + 17 * len(scen_labels)))

    for algo in algos:
        vals = []
        for scen_label in scen_labels:
            deis = []
            for s in seeds:
                k = "{}_{}_s{}".format(algo, scen_label, s)
                if k in completed and "error" not in completed[k]:
                    deis.append(completed[k].get("dei", 0))
            if deis:
                vals.append("{:.3f}+-{:.3f}  ".format(
                    np.mean(deis), np.std(deis)))
            else:
                vals.append("    --        ")
        log("  {:>8s} | {}".format(algo, " | ".join(vals)))

    # Key comparison: Mixed vs baselines
    log("\n  === KEY FINDING: Mixed Cross-Border vs Baselines ===")
    for algo in algos:
        mixed_accs = []
        strict_accs = []
        perm_accs = []
        nodp_accs = []
        for s in seeds:
            k_m = "{}_{}_s{}".format(algo, "Mixed-2DE-3IT", s)
            k_s = "{}_{}_s{}".format(algo, "Uniform-eps1", s)
            k_p = "{}_{}_s{}".format(algo, "Uniform-eps10", s)
            k_n = "{}_{}_s{}".format(algo, "No-DP", s)
            if k_m in completed and "error" not in completed[k_m]:
                mixed_accs.append(completed[k_m]["best_accuracy"] * 100)
            if k_s in completed and "error" not in completed[k_s]:
                strict_accs.append(completed[k_s]["best_accuracy"] * 100)
            if k_p in completed and "error" not in completed[k_p]:
                perm_accs.append(completed[k_p]["best_accuracy"] * 100)
            if k_n in completed and "error" not in completed[k_n]:
                nodp_accs.append(completed[k_n]["best_accuracy"] * 100)

        if mixed_accs and strict_accs and perm_accs:
            mixed_mean = np.mean(mixed_accs)
            strict_mean = np.mean(strict_accs)
            perm_mean = np.mean(perm_accs)
            nodp_mean = np.mean(nodp_accs) if nodp_accs else 0

            log("  {}: Mixed={:.1f}% vs Strict={:.1f}% (+{:.1f}pp) vs "
                "Permissive={:.1f}% ({:.1f}pp) vs NoDp={:.1f}%".format(
                    algo, mixed_mean, strict_mean,
                    mixed_mean - strict_mean,
                    perm_mean, mixed_mean - perm_mean,
                    nodp_mean,
                ))

    # ======================================================================
    # Generate figures
    # ======================================================================
    generate_figures(completed, algos, scenarios, seeds)

    if _log_file:
        _log_file.close()


if __name__ == "__main__":
    main()

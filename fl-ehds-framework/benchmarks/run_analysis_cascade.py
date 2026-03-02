#!/usr/bin/env python3
"""
FL-EHDS Cascading Analysis & Experiments.

4 blocks executed sequentially from fastest to slowest:

Block A: Statistical Significance Testing (analytical, ~15s)
  Loads multi-seed checkpoints (seeds10, dp) and computes paired t-tests,
  Wilcoxon signed-rank tests, effect sizes (Cohen's d), and 95% CIs.

Block B: Communication Cost Quantification (analytical, ~5s)
  Calculates per-round and total bandwidth for each algorithm x Top-K
  ratio x number of clients. Produces cost comparison tables.

Block C: HPFL Personalization Depth Ablation (training, ~30-40 min)
  Runs 4 algorithm variants on 3 datasets x 2 DP levels x seed=42:
    - FedAvg    (0% personalization — full aggregation)
    - Ditto     (regularization-based personalization)
    - HPFL      (classifier-head personalization ~2-6%)
    - HPFL-Full (100% personalization = local training, no aggregation)
  HPFL-Full monkey-patches all params as "classifier" so no global
  aggregation occurs. Each client trains locally with shared init.
  All experiments run 30 rounds with per-round accuracy tracking.
  Total: 24 experiments.

Block D: Convergence Curve Analysis (analytical + plots, ~15s)
  Extracts per-round data from Block C results AND from existing
  checkpoints (seeds10, dp). Generates convergence curve PDFs with
  confidence bands where multi-seed data is available.

Usage:
    cd fl-ehds-framework
    python -m benchmarks.run_analysis_cascade [--quick] [--fresh]

Output:
    benchmarks/paper_results_tabular/checkpoint_analysis_cascade.json
    benchmarks/paper_results_tabular/significance_table.pdf
    benchmarks/paper_results_tabular/comm_costs_table.pdf
    benchmarks/paper_results_tabular/ablation_convergence.pdf
    benchmarks/paper_results_tabular/convergence_curves.pdf

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
from typing import Dict, List, Any, Optional, Tuple
from copy import deepcopy

import numpy as np

FRAMEWORK_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(FRAMEWORK_DIR))

import torch

from terminal.fl_trainer import FederatedTrainer, _detect_device
from data.ptbxl_loader import load_ptbxl_data
from data.cardiovascular_loader import load_cardiovascular_data
from data.breast_cancer_loader import load_breast_cancer_data

# ======================================================================
# Constants
# ======================================================================

OUTPUT_DIR = FRAMEWORK_DIR / "benchmarks" / "paper_results_tabular"
CHECKPOINT_FILE = "checkpoint_analysis_cascade.json"
LOG_FILE = "experiment_analysis_cascade.log"

ALGORITHMS = ["FedAvg", "Ditto", "HPFL"]
ABLATION_VARIANTS = ["FedAvg", "Ditto", "HPFL", "HPFL-Full"]

SEEDS_10 = [7, 31, 137, 577, 1337]      # seeds used in checkpoint_seeds10
SEEDS_DP = [42, 123, 456, 789, 999]     # seeds used in checkpoint_dp

PTB_XL_CLASS_NAMES = {0: "NORM", 1: "MI", 2: "STTC", 3: "CD", 4: "HYP"}
CV_CLASS_NAMES = {0: "Healthy", 1: "Disease"}
BC_CLASS_NAMES = {0: "Benign", 1: "Malignant"}

# Training configs (matched to recent experiments)
DATASET_CONFIGS = {
    "Cardiovascular": dict(
        learning_rate=0.01, batch_size=64, num_rounds=30, local_epochs=3,
        mu=0.1, num_clients=5, input_dim=11, num_classes=2,
        class_names=CV_CLASS_NAMES,
    ),
    "PTB_XL": dict(
        learning_rate=0.005, batch_size=64, num_rounds=30, local_epochs=3,
        mu=0.1, num_clients=5, input_dim=9, num_classes=5,
        class_names=PTB_XL_CLASS_NAMES,
    ),
    "Breast_Cancer": dict(
        learning_rate=0.005, batch_size=32, num_rounds=30, local_epochs=2,
        mu=0.1, num_clients=4, input_dim=30, num_classes=2,
        class_names=BC_CLASS_NAMES,
    ),
}

DP_LEVELS = [
    {"label": "No-DP", "dp_enabled": False, "dp_epsilon": 0},
    {"label": "DP-eps10", "dp_enabled": True, "dp_epsilon": 10.0},
]

# Model parameter counts (HealthcareMLP with hidden [64, 32])
MODEL_PARAMS = {
    "PTB_XL":        {"total": 2885, "backbone": 2720, "classifier": 165},
    "Cardiovascular": {"total": 2914, "backbone": 2848, "classifier": 66},
    "Breast_Cancer":  {"total": 4130, "backbone": 4064, "classifier": 66},
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
# Checkpoint (atomic write with backup)
# ======================================================================

def save_checkpoint(data):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / CHECKPOINT_FILE
    bak = OUTPUT_DIR / (CHECKPOINT_FILE + ".bak")
    data["metadata"]["last_save"] = datetime.now().isoformat()
    fd, tmp = tempfile.mkstemp(dir=str(OUTPUT_DIR), prefix=".acas_", suffix=".tmp")
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
# Data loading (cached per dataset)
# ======================================================================

_data_cache = {}


def load_dataset(dataset, num_clients, seed):
    """Load dataset, caching by (dataset, num_clients, seed)."""
    cache_key = (dataset, num_clients, seed)
    if cache_key in _data_cache:
        return _data_cache[cache_key]

    if dataset == "Cardiovascular":
        client_data, client_test, meta = load_cardiovascular_data(
            num_clients=num_clients, seed=seed, is_iid=True,
        )
    elif dataset == "PTB_XL":
        client_data, client_test, meta = load_ptbxl_data(
            num_clients=num_clients, seed=seed,
            partition_by_site=True, min_site_samples=50,
        )
    elif dataset == "Breast_Cancer":
        client_data, client_test, meta = load_breast_cancer_data(
            num_clients=num_clients, seed=seed,
        )
    else:
        raise ValueError("Unknown dataset: {}".format(dataset))

    _data_cache[cache_key] = (client_data, client_test, meta)
    return client_data, client_test, meta


# ======================================================================
# Block A: Statistical Significance Testing
# ======================================================================

def _paired_t_test(x, y):
    """Paired t-test for two arrays. Returns (t_stat, p_value, cohens_d, ci_95)."""
    diffs = np.array(x) - np.array(y)
    n = len(diffs)
    if n < 2:
        return (np.nan, np.nan, np.nan, (np.nan, np.nan))
    mean_d = np.mean(diffs)
    std_d = np.std(diffs, ddof=1)
    if std_d == 0:
        return (np.inf if mean_d > 0 else -np.inf, 0.0, np.inf, (mean_d, mean_d))
    t_stat = mean_d / (std_d / np.sqrt(n))
    # Two-sided p-value from t-distribution (approximation via normal for n>=5)
    # For small n, use exact t-distribution if scipy available
    try:
        from scipy import stats as sp_stats
        p_value = sp_stats.t.sf(abs(t_stat), df=n - 1) * 2
    except ImportError:
        # Normal approximation (less accurate for small n)
        p_value = 2 * (1 - _normal_cdf(abs(t_stat)))
    cohens_d = mean_d / std_d
    # 95% CI for mean difference
    try:
        from scipy import stats as sp_stats
        t_crit = sp_stats.t.ppf(0.975, df=n - 1)
    except ImportError:
        t_crit = 2.776 if n == 5 else 2.571 if n == 6 else 2.447 if n == 7 else 1.96
    margin = t_crit * std_d / np.sqrt(n)
    ci_95 = (mean_d - margin, mean_d + margin)
    return (t_stat, p_value, cohens_d, ci_95)


def _wilcoxon_test(x, y):
    """Wilcoxon signed-rank test. Returns (stat, p_value) or (nan, nan)."""
    try:
        from scipy.stats import wilcoxon
        diffs = np.array(x) - np.array(y)
        # Need at least 5 non-zero differences
        nonzero = np.sum(diffs != 0)
        if nonzero < 5:
            return (np.nan, np.nan)
        stat, p_val = wilcoxon(x, y, alternative='two-sided')
        return (stat, p_val)
    except (ImportError, ValueError):
        return (np.nan, np.nan)


def _normal_cdf(x):
    """Standard normal CDF approximation."""
    return 0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


def _extract_accuracies(checkpoint, dataset, algo, seeds, key_fmt, acc_field):
    """Extract accuracies from a checkpoint for given seeds."""
    accs = []
    data = checkpoint.get("completed", checkpoint.get("results", {}))
    for s in seeds:
        key = key_fmt.format(dataset=dataset, algo=algo, seed=s)
        if key in data:
            entry = data[key]
            if isinstance(acc_field, str):
                val = entry
                for part in acc_field.split("."):
                    val = val.get(part, {}) if isinstance(val, dict) else None
                    if val is None:
                        break
                if val is not None:
                    accs.append(float(val))
            elif callable(acc_field):
                val = acc_field(entry)
                if val is not None:
                    accs.append(float(val))
    return accs


def block_a_significance(ckpt):
    """Block A: Statistical significance testing from existing checkpoints."""
    log("=" * 70)
    log("BLOCK A — Statistical Significance Testing")
    log("=" * 70)

    start = time.time()
    results = {}

    comparisons = [
        ("HPFL", "FedAvg"),
        ("HPFL", "Ditto"),
        ("Ditto", "FedAvg"),
    ]
    datasets = ["Breast_Cancer", "Cardiovascular", "PTB_XL"]

    # ---- Source 1: seeds10 (5 seeds, no DP) ----
    s10_path = OUTPUT_DIR / "checkpoint_seeds10.json"
    if s10_path.exists():
        s10 = json.load(open(s10_path))
        log("  Loaded seeds10: {} experiments".format(len(s10.get("completed", {}))))

        for ds in datasets:
            for algo_a, algo_b in comparisons:
                # Extract final accuracy from 'final_metrics.accuracy' or 'per_client_acc' mean
                accs_a, accs_b = [], []
                for s in SEEDS_10:
                    key_a = "{}_{}".format(ds, algo_a) + "_s{}".format(s)
                    key_b = "{}_{}".format(ds, algo_b) + "_s{}".format(s)
                    data = s10.get("completed", {})
                    if key_a in data and key_b in data:
                        # Use per_client_acc mean for consistency with HPFL evaluation
                        pca_a = data[key_a].get("per_client_acc", {})
                        pca_b = data[key_b].get("per_client_acc", {})
                        if pca_a and pca_b:
                            accs_a.append(float(np.mean(list(pca_a.values()))))
                            accs_b.append(float(np.mean(list(pca_b.values()))))
                        else:
                            # Fallback to final_metrics
                            fm_a = data[key_a].get("final_metrics", {})
                            fm_b = data[key_b].get("final_metrics", {})
                            if fm_a and fm_b:
                                accs_a.append(fm_a.get("accuracy", 0))
                                accs_b.append(fm_b.get("accuracy", 0))

                if len(accs_a) >= 3:
                    t_stat, p_val, d, ci = _paired_t_test(accs_a, accs_b)
                    w_stat, w_pval = _wilcoxon_test(accs_a, accs_b)
                    rkey = "seeds10_{}_{}_vs_{}".format(ds, algo_a, algo_b)
                    results[rkey] = {
                        "source": "seeds10", "dataset": ds, "dp_level": "No-DP",
                        "algo_a": algo_a, "algo_b": algo_b,
                        "n_seeds": len(accs_a),
                        "mean_a": round(float(np.mean(accs_a)), 4),
                        "mean_b": round(float(np.mean(accs_b)), 4),
                        "mean_diff": round(float(np.mean(accs_a) - np.mean(accs_b)), 4),
                        "t_stat": round(float(t_stat), 4) if not np.isnan(t_stat) else None,
                        "p_value_ttest": round(float(p_val), 6) if not np.isnan(p_val) else None,
                        "cohens_d": round(float(d), 4) if not np.isnan(d) else None,
                        "ci_95_lower": round(float(ci[0]), 4) if not np.isnan(ci[0]) else None,
                        "ci_95_upper": round(float(ci[1]), 4) if not np.isnan(ci[1]) else None,
                        "wilcoxon_stat": round(float(w_stat), 4) if not np.isnan(w_stat) else None,
                        "p_value_wilcoxon": round(float(w_pval), 6) if not np.isnan(w_pval) else None,
                        "significant_005": bool(p_val < 0.05) if not np.isnan(p_val) else None,
                    }
    else:
        log("  WARNING: seeds10 checkpoint not found")

    # ---- Source 2: dp (5 seeds, per DP level) ----
    dp_path = OUTPUT_DIR / "checkpoint_dp.json"
    if dp_path.exists():
        dp_data = json.load(open(dp_path))
        log("  Loaded dp: {} experiments".format(len(dp_data.get("completed", {}))))

        dp_levels = [1, 5, 10, 50]
        for eps in dp_levels:
            for ds in datasets:
                for algo_a, algo_b in comparisons:
                    accs_a, accs_b = [], []
                    for s in SEEDS_DP:
                        key_a = "{}_{}_eps{}_s{}".format(ds, algo_a, eps, s)
                        key_b = "{}_{}_eps{}_s{}".format(ds, algo_b, eps, s)
                        data = dp_data.get("completed", {})
                        if key_a in data and key_b in data:
                            pca_a = data[key_a].get("per_client_acc", {})
                            pca_b = data[key_b].get("per_client_acc", {})
                            if pca_a and pca_b:
                                accs_a.append(float(np.mean(list(pca_a.values()))))
                                accs_b.append(float(np.mean(list(pca_b.values()))))
                            else:
                                fm_a = data[key_a].get("final_metrics", {})
                                fm_b = data[key_b].get("final_metrics", {})
                                if fm_a and fm_b:
                                    accs_a.append(fm_a.get("accuracy", 0))
                                    accs_b.append(fm_b.get("accuracy", 0))

                    if len(accs_a) >= 3:
                        t_stat, p_val, d, ci = _paired_t_test(accs_a, accs_b)
                        w_stat, w_pval = _wilcoxon_test(accs_a, accs_b)
                        rkey = "dp_eps{}_{}_{}vs{}".format(eps, ds, algo_a, algo_b)
                        results[rkey] = {
                            "source": "dp", "dataset": ds,
                            "dp_level": "eps={}".format(eps),
                            "algo_a": algo_a, "algo_b": algo_b,
                            "n_seeds": len(accs_a),
                            "mean_a": round(float(np.mean(accs_a)), 4),
                            "mean_b": round(float(np.mean(accs_b)), 4),
                            "mean_diff": round(float(np.mean(accs_a) - np.mean(accs_b)), 4),
                            "t_stat": round(float(t_stat), 4) if not np.isnan(t_stat) else None,
                            "p_value_ttest": round(float(p_val), 6) if not np.isnan(p_val) else None,
                            "cohens_d": round(float(d), 4) if not np.isnan(d) else None,
                            "ci_95_lower": round(float(ci[0]), 4) if not np.isnan(ci[0]) else None,
                            "ci_95_upper": round(float(ci[1]), 4) if not np.isnan(ci[1]) else None,
                            "wilcoxon_stat": round(float(w_stat), 4) if not np.isnan(w_stat) else None,
                            "p_value_wilcoxon": round(float(w_pval), 6) if not np.isnan(w_pval) else None,
                            "significant_005": bool(p_val < 0.05) if not np.isnan(p_val) else None,
                        }
    else:
        log("  WARNING: dp checkpoint not found")

    # ---- Source 3: noniid_dp (3 seeds, per alpha x DP) ----
    noniid_path = OUTPUT_DIR / "checkpoint_noniid_dp.json"
    if noniid_path.exists():
        noniid = json.load(open(noniid_path))
        log("  Loaded noniid_dp: {} experiments".format(len(noniid.get("results", {}))))

        noniid_seeds = [42, 123, 456]
        alphas = [0.1, 0.5, 1.0, 3, 5]
        noniid_dp_levels = ["No-DP", "DP-eps1", "DP-eps10"]
        noniid_datasets = ["Cardiovascular", "PTB_XL"]

        for ds in noniid_datasets:
            ds_short = "CV" if ds == "Cardiovascular" else "PX"
            for alpha in alphas:
                for dp_lbl in noniid_dp_levels:
                    for algo_a, algo_b in comparisons:
                        accs_a, accs_b = [], []
                        for s in noniid_seeds:
                            key_a = "{}_{}_a{}_{}_s{}".format(ds_short, algo_a, alpha, dp_lbl, s)
                            key_b = "{}_{}_a{}_{}_s{}".format(ds_short, algo_b, alpha, dp_lbl, s)
                            data = noniid.get("results", {})
                            if key_a in data and key_b in data:
                                accs_a.append(data[key_a].get("final_accuracy", 0))
                                accs_b.append(data[key_b].get("final_accuracy", 0))

                        if len(accs_a) >= 3:
                            t_stat, p_val, d, ci = _paired_t_test(accs_a, accs_b)
                            rkey = "noniid_a{}_{}_{}_{}vs{}".format(alpha, dp_lbl, ds, algo_a, algo_b)
                            results[rkey] = {
                                "source": "noniid_dp", "dataset": ds,
                                "dp_level": dp_lbl, "alpha": alpha,
                                "algo_a": algo_a, "algo_b": algo_b,
                                "n_seeds": len(accs_a),
                                "mean_a": round(float(np.mean(accs_a)), 4),
                                "mean_b": round(float(np.mean(accs_b)), 4),
                                "mean_diff": round(float(np.mean(accs_a) - np.mean(accs_b)), 4),
                                "t_stat": round(float(t_stat), 4) if not np.isnan(t_stat) else None,
                                "p_value_ttest": round(float(p_val), 6) if not np.isnan(p_val) else None,
                                "cohens_d": round(float(d), 4) if not np.isnan(d) else None,
                                "ci_95_lower": round(float(ci[0]), 4) if not np.isnan(ci[0]) else None,
                                "ci_95_upper": round(float(ci[1]), 4) if not np.isnan(ci[1]) else None,
                                "significant_005": bool(p_val < 0.05) if not np.isnan(p_val) else None,
                            }
    else:
        log("  WARNING: noniid_dp checkpoint not found")

    # ---- Summary ----
    elapsed = time.time() - start
    total_tests = len(results)
    significant = sum(1 for r in results.values() if r.get("significant_005"))

    log("")
    log("  {} significance tests computed in {:.1f}s".format(total_tests, elapsed))
    log("  {} / {} significant at p<0.05".format(significant, total_tests))

    # Print key results table
    log("")
    log("  KEY RESULTS (seeds10, No-DP):")
    log("  {:<15} {:<15} {:<10} {:<10} {:<10} {:<10}".format(
        "Dataset", "Comparison", "MeanDiff", "Cohen's d", "p-value", "Sig?"))
    log("  " + "-" * 70)
    for rkey, r in sorted(results.items()):
        if r["source"] == "seeds10":
            sig_marker = "***" if r.get("significant_005") else "n.s."
            log("  {:<15} {:<15} {:>+.4f}   {:>8}   {:>10}   {}".format(
                r["dataset"],
                "{} vs {}".format(r["algo_a"], r["algo_b"]),
                r["mean_diff"],
                "{:.2f}".format(r["cohens_d"]) if r["cohens_d"] is not None else "N/A",
                "{:.6f}".format(r["p_value_ttest"]) if r["p_value_ttest"] is not None else "N/A",
                sig_marker))

    ckpt["block_a_significance"] = results
    ckpt["metadata"]["block_a_time"] = round(elapsed, 1)
    save_checkpoint(ckpt)
    log("  Block A complete.")
    return results


# ======================================================================
# Block B: Communication Cost Quantification
# ======================================================================

def block_b_comm_costs(ckpt):
    """Block B: Communication cost analysis."""
    log("")
    log("=" * 70)
    log("BLOCK B — Communication Cost Quantification")
    log("=" * 70)

    start = time.time()
    results = {}

    topk_ratios = {"NoTopK": 1.0, "Top5pct": 0.05, "Top1pct": 0.01}
    client_counts = [3, 5, 10, 20]
    bytes_per_param = 4  # float32

    for ds, info in MODEL_PARAMS.items():
        dcfg = DATASET_CONFIGS[ds]
        num_rounds = dcfg["num_rounds"]

        for algo in ["FedAvg", "Ditto", "HPFL"]:
            # HPFL only communicates backbone params (classifier stays local)
            if algo == "HPFL":
                comm_params = info["backbone"]
            else:
                comm_params = info["total"]

            for tk_label, tk_ratio in topk_ratios.items():
                params_sent = int(np.ceil(comm_params * tk_ratio))

                for K in client_counts:
                    # Per round: each client uploads params_sent, server broadcasts full backbone
                    # Upload: K * params_sent * 4 bytes (sparse with Top-K)
                    # Download: K * comm_params * 4 bytes (full model broadcast)
                    upload_per_round = K * params_sent * bytes_per_param
                    download_per_round = K * comm_params * bytes_per_param
                    total_per_round = upload_per_round + download_per_round

                    total_bytes = total_per_round * num_rounds
                    total_mb = total_bytes / (1024 * 1024)

                    rkey = "{}_{}_{}_{}_K{}".format(ds, algo, tk_label,
                                                    "R{}".format(num_rounds), K)
                    results[rkey] = {
                        "dataset": ds,
                        "algorithm": algo,
                        "topk": tk_label,
                        "topk_ratio": tk_ratio,
                        "num_clients": K,
                        "num_rounds": num_rounds,
                        "model_params_total": info["total"],
                        "comm_params": comm_params,
                        "params_per_upload": params_sent,
                        "upload_bytes_per_round": upload_per_round,
                        "download_bytes_per_round": download_per_round,
                        "total_bytes_per_round": total_per_round,
                        "total_bytes": total_bytes,
                        "total_MB": round(total_mb, 4),
                    }

    elapsed = time.time() - start

    # Print summary table
    log("")
    log("  COMMUNICATION COSTS (5 clients, 30 rounds):")
    log("  {:<15} {:<8} {:<10} {:>12} {:>12} {:>10}".format(
        "Dataset", "Algo", "Top-K", "Upload/Rnd", "Download/Rnd", "Total MB"))
    log("  " + "-" * 70)
    for ds in MODEL_PARAMS:
        for algo in ["FedAvg", "Ditto", "HPFL"]:
            for tk_label in topk_ratios:
                rkey = "{}_{}_{}_{}_K5".format(ds, algo, tk_label,
                                                "R{}".format(DATASET_CONFIGS[ds]["num_rounds"]))
                if rkey in results:
                    r = results[rkey]
                    log("  {:<15} {:<8} {:<10} {:>10} B  {:>10} B  {:>8.4f}".format(
                        ds[:12], algo, tk_label,
                        r["upload_bytes_per_round"],
                        r["download_bytes_per_round"],
                        r["total_MB"]))

    # HPFL bandwidth savings vs FedAvg
    log("")
    log("  HPFL BANDWIDTH SAVINGS vs FedAvg (K=5, NoTopK):")
    for ds in MODEL_PARAMS:
        info = MODEL_PARAMS[ds]
        savings_pct = (1 - info["backbone"] / info["total"]) * 100
        log("    {}: {:.1f}% savings ({} backbone vs {} total params)".format(
            ds, savings_pct, info["backbone"], info["total"]))

    ckpt["block_b_comm_costs"] = results
    ckpt["metadata"]["block_b_time"] = round(elapsed, 1)
    save_checkpoint(ckpt)
    log("  Block B complete ({:.1f}s).".format(elapsed))
    return results


# ======================================================================
# Block C: HPFL Personalization Depth Ablation (TRAINING)
# ======================================================================

def run_ablation_experiment(dataset, variant, dp_cfg, seed, quick=False):
    """Run one ablation experiment."""
    start = time.time()
    dcfg = DATASET_CONFIGS[dataset]
    num_rounds = 5 if quick else dcfg["num_rounds"]

    # Load data
    client_data, client_test, meta = load_dataset(
        dataset, dcfg["num_clients"], seed)

    # Pre-check for empty clients
    for cid in range(dcfg["num_clients"]):
        X_c, y_c = client_data[cid]
        if len(y_c) == 0:
            raise ValueError("Client {} has 0 training samples — skipping".format(cid))

    # Determine algorithm for trainer
    trainer_algo = "HPFL" if variant in ("HPFL", "HPFL-Full") else variant

    trainer_kwargs = dict(
        num_clients=dcfg["num_clients"], algorithm=trainer_algo,
        local_epochs=dcfg["local_epochs"], batch_size=dcfg["batch_size"],
        learning_rate=dcfg["learning_rate"], mu=dcfg["mu"], seed=seed,
        external_data=client_data, external_test_data=client_test,
        input_dim=dcfg["input_dim"], num_classes=dcfg["num_classes"],
    )

    if dp_cfg["dp_enabled"]:
        trainer_kwargs["dp_enabled"] = True
        trainer_kwargs["dp_epsilon"] = dp_cfg["dp_epsilon"]
        trainer_kwargs["dp_clip_norm"] = 1.0

    trainer = FederatedTrainer(**trainer_kwargs)

    # HPFL-Full: monkey-patch ALL params as "classifier" → pure local training
    if variant == "HPFL-Full":
        all_param_names = {n for n, _ in trainer.global_model.named_parameters()}
        trainer._hpfl_classifier_names = all_param_names
        trainer.client_classifiers = {
            i: {n: trainer.global_model.state_dict()[n].clone()
                for n in all_param_names}
            for i in range(trainer.num_clients)
        }

    # Train with per-round tracking
    history = []
    best_acc = 0.0

    for r in range(num_rounds):
        result = trainer.train_round(r)

        # For HPFL variants, evaluate with per-client classifiers
        if variant in ("HPFL", "HPFL-Full"):
            per_client_acc_round = _evaluate_per_client(trainer)
            round_acc = float(np.mean(list(per_client_acc_round.values())))
        else:
            round_acc = result.global_acc

        history.append({
            "round": r + 1,
            "accuracy": round_acc,
            "loss": result.global_loss,
            "f1": result.global_f1,
        })
        if round_acc > best_acc:
            best_acc = round_acc

    # Final evaluation
    per_client_acc = _evaluate_per_client(trainer)
    jain = _compute_jain_index(per_client_acc)
    final_acc = float(np.mean(list(per_client_acc.values())))

    y_true, y_pred = _collect_predictions(trainer)
    per_class, cm = _compute_per_class_metrics(
        y_true, y_pred, dcfg["num_classes"], dcfg["class_names"])
    dei = _compute_dei(per_class, dcfg["num_classes"], dcfg["class_names"])

    elapsed = time.time() - start

    # Determine personalization percentage
    if variant == "HPFL":
        info = MODEL_PARAMS[dataset]
        personal_pct = round(info["classifier"] / info["total"] * 100, 1)
    elif variant == "HPFL-Full":
        personal_pct = 100.0
    else:
        personal_pct = 0.0

    return {
        "block": "C_ablation",
        "dataset": dataset,
        "variant": variant,
        "algorithm_used": trainer_algo,
        "personalization_pct": personal_pct,
        "dp_level": dp_cfg["label"],
        "dp_epsilon": dp_cfg["dp_epsilon"],
        "seed": seed,
        "num_rounds": num_rounds,
        "final_accuracy": round(final_acc, 4),
        "best_accuracy": round(best_acc, 4),
        "per_client_accuracy": per_client_acc,
        "jain_index": round(jain, 4),
        "dei": dei,
        "per_class_metrics": per_class,
        "confusion_matrix": cm,
        "convergence_trajectory": [h["accuracy"] for h in history],
        "history": history,
        "time_seconds": round(elapsed, 1),
    }


def block_c_ablation(ckpt, quick=False):
    """Block C: HPFL personalization depth ablation (training)."""
    log("")
    log("=" * 70)
    log("BLOCK C — HPFL Personalization Depth Ablation")
    log("=" * 70)

    start = time.time()
    datasets = ["Cardiovascular", "PTB_XL", "Breast_Cancer"]
    seed = 42

    # Build experiment list
    experiments = []
    for ds in datasets:
        for variant in ABLATION_VARIANTS:
            for dp_cfg in DP_LEVELS:
                exp_key = "C_{}_{}_{}".format(ds, variant, dp_cfg["label"])
                experiments.append((exp_key, ds, variant, dp_cfg))

    total = len(experiments)
    done = sum(1 for e in experiments if e[0] in ckpt.get("completed", []))
    log("  {} experiments ({} already done)".format(total, done))

    for idx, (exp_key, ds, variant, dp_cfg) in enumerate(experiments, 1):
        if _shutdown:
            log("  Shutdown requested — saving checkpoint")
            break

        if exp_key in ckpt.get("completed", []):
            continue

        try:
            personal_lbl = {"FedAvg": "0%", "Ditto": "reg",
                           "HPFL": "head", "HPFL-Full": "100%"}[variant]
            log("  [{}/{}] {} | {} ({} personal) | {} ...".format(
                idx, total, ds, variant, personal_lbl, dp_cfg["label"]))

            result = run_ablation_experiment(ds, variant, dp_cfg, seed, quick=quick)

            ckpt["results"][exp_key] = result
            if "completed" not in ckpt:
                ckpt["completed"] = []
            ckpt["completed"].append(exp_key)
            save_checkpoint(ckpt)

            log("    -> acc={:.1f}% | Jain={:.3f} | DEI={:.3f} | {:.0f}s".format(
                result["final_accuracy"] * 100,
                result["jain_index"],
                result["dei"],
                result["time_seconds"]))

            _cleanup_gpu()

        except Exception as e:
            log("    ERROR: {}".format(e))
            log("    " + traceback.format_exc().split("\n")[-2])
            _cleanup_gpu()

    elapsed = time.time() - start
    ckpt["metadata"]["block_c_time"] = round(elapsed, 1)
    save_checkpoint(ckpt)

    # Print ablation summary table
    log("")
    log("  ABLATION RESULTS:")
    log("  {:<15} {:<12} {:<10} {:<10} {:<8} {:<8}".format(
        "Dataset", "Variant", "DP", "Acc%", "Jain", "DEI"))
    log("  " + "-" * 65)
    for ds in datasets:
        for variant in ABLATION_VARIANTS:
            for dp_cfg in DP_LEVELS:
                exp_key = "C_{}_{}_{}".format(ds, variant, dp_cfg["label"])
                if exp_key in ckpt.get("results", {}):
                    r = ckpt["results"][exp_key]
                    log("  {:<15} {:<12} {:<10} {:>6.1f}    {:.3f}   {:.3f}".format(
                        ds[:12], variant, dp_cfg["label"],
                        r["final_accuracy"] * 100,
                        r["jain_index"],
                        r["dei"]))
        log("  " + "-" * 65)

    log("  Block C complete ({}).".format(format_time(elapsed)))
    return


# ======================================================================
# Block D: Convergence Curve Analysis (analytical + plots)
# ======================================================================

def block_d_convergence(ckpt):
    """Block D: Extract convergence data and generate plots."""
    log("")
    log("=" * 70)
    log("BLOCK D — Convergence Curve Analysis & Plots")
    log("=" * 70)

    start = time.time()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
        has_matplotlib = True
    except ImportError:
        log("  WARNING: matplotlib not available — skipping plot generation")
        has_matplotlib = False

    convergence_data = {}

    # ---- Extract from Block C ablation results ----
    log("  Extracting convergence from Block C ablation results...")
    block_c_keys = [k for k in ckpt.get("results", {})
                    if k.startswith("C_") and "convergence_trajectory" in ckpt["results"].get(k, {})]
    for k in block_c_keys:
        r = ckpt["results"][k]
        ckey = "ablation_{}_{}_{}" .format(r["dataset"], r["variant"], r["dp_level"])
        convergence_data[ckey] = {
            "source": "ablation",
            "dataset": r["dataset"],
            "algorithm": r["variant"],
            "dp_level": r["dp_level"],
            "trajectory": r["convergence_trajectory"],
            "final_accuracy": r["final_accuracy"],
        }
    log("    {} convergence curves from ablation".format(len(block_c_keys)))

    # ---- Extract from seeds10 (5 seeds → mean ± std) ----
    s10_path = OUTPUT_DIR / "checkpoint_seeds10.json"
    if s10_path.exists():
        s10 = json.load(open(s10_path))
        datasets = ["Breast_Cancer", "Cardiovascular", "PTB_XL"]
        algos_s10 = ["FedAvg", "Ditto", "HPFL"]

        for ds in datasets:
            for algo in algos_s10:
                trajectories = []
                for s in SEEDS_10:
                    key = "{}_{}_s{}".format(ds, algo, s)
                    data = s10.get("completed", {})
                    if key in data and "history" in data[key]:
                        hist = data[key]["history"]
                        traj = [h["accuracy"] for h in hist]
                        trajectories.append(traj)

                if trajectories:
                    # Pad to same length (in case of early stopping)
                    max_len = max(len(t) for t in trajectories)
                    padded = []
                    for t in trajectories:
                        if len(t) < max_len:
                            t = t + [t[-1]] * (max_len - len(t))
                        padded.append(t)
                    arr = np.array(padded)
                    ckey = "seeds10_{}_{}".format(ds, algo)
                    convergence_data[ckey] = {
                        "source": "seeds10",
                        "dataset": ds,
                        "algorithm": algo,
                        "dp_level": "No-DP",
                        "trajectory_mean": np.mean(arr, axis=0).tolist(),
                        "trajectory_std": np.std(arr, axis=0).tolist(),
                        "n_seeds": len(trajectories),
                    }
        log("    Extracted convergence from seeds10")

    # ---- Extract from dp checkpoint (5 seeds → mean ± std) ----
    dp_path = OUTPUT_DIR / "checkpoint_dp.json"
    if dp_path.exists():
        dp_data = json.load(open(dp_path))
        datasets = ["Breast_Cancer", "Cardiovascular", "PTB_XL"]
        algos_dp = ["FedAvg", "Ditto", "HPFL"]

        for eps in [1, 10]:
            for ds in datasets:
                for algo in algos_dp:
                    trajectories = []
                    for s in SEEDS_DP:
                        key = "{}_{}_eps{}_s{}".format(ds, algo, eps, s)
                        data = dp_data.get("completed", {})
                        if key in data and "history" in data[key]:
                            hist = data[key]["history"]
                            traj = [h["accuracy"] for h in hist]
                            trajectories.append(traj)

                    if trajectories:
                        max_len = max(len(t) for t in trajectories)
                        padded = []
                        for t in trajectories:
                            if len(t) < max_len:
                                t = t + [t[-1]] * (max_len - len(t))
                            padded.append(t)
                        arr = np.array(padded)
                        ckey = "dp_eps{}_{}_{}".format(eps, ds, algo)
                        convergence_data[ckey] = {
                            "source": "dp",
                            "dataset": ds,
                            "algorithm": algo,
                            "dp_level": "eps={}".format(eps),
                            "trajectory_mean": np.mean(arr, axis=0).tolist(),
                            "trajectory_std": np.std(arr, axis=0).tolist(),
                            "n_seeds": len(trajectories),
                        }
        log("    Extracted convergence from dp checkpoint")

    ckpt["block_d_convergence"] = convergence_data
    save_checkpoint(ckpt)

    # ---- Generate plots ----
    if has_matplotlib and convergence_data:
        algo_colors = {
            "FedAvg": "#2196F3", "Ditto": "#FF9800", "HPFL": "#4CAF50",
            "HPFL-Full": "#E91E63",
        }
        algo_markers = {
            "FedAvg": "o", "Ditto": "s", "HPFL": "^", "HPFL-Full": "D",
        }

        # ---- Plot 1: Ablation convergence curves (from Block C) ----
        log("  Generating ablation convergence plot...")
        datasets_plot = ["Cardiovascular", "PTB_XL", "Breast_Cancer"]
        dp_labels = ["No-DP", "DP-eps10"]

        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        fig.suptitle("HPFL Personalization Depth Ablation — Convergence",
                     fontsize=14, fontweight="bold")

        for col, ds in enumerate(datasets_plot):
            for row, dp_lbl in enumerate(dp_labels):
                ax = axes[row, col]
                for variant in ABLATION_VARIANTS:
                    ckey = "ablation_{}_{}_{}" .format(ds, variant, dp_lbl)
                    if ckey in convergence_data:
                        traj = convergence_data[ckey]["trajectory"]
                        rounds = list(range(1, len(traj) + 1))
                        color = algo_colors.get(variant, "#666666")
                        marker = algo_markers.get(variant, "x")
                        ax.plot(rounds, [a * 100 for a in traj],
                                color=color, marker=marker, markersize=3,
                                linewidth=1.5, label=variant)

                ax.set_title("{} ({})".format(ds.replace("_", " "), dp_lbl),
                             fontsize=10)
                ax.set_xlabel("Round")
                ax.set_ylabel("Accuracy (%)")
                ax.grid(True, alpha=0.3)
                if col == 0 and row == 0:
                    ax.legend(fontsize=8)

        plt.tight_layout()
        plt.savefig(str(OUTPUT_DIR / "ablation_convergence.pdf"),
                    bbox_inches="tight", dpi=150)
        plt.savefig(str(OUTPUT_DIR / "ablation_convergence.png"),
                    bbox_inches="tight", dpi=150)
        plt.close()
        log("    Saved ablation_convergence.pdf")

        # ---- Plot 2: Multi-seed convergence with confidence bands ----
        log("  Generating multi-seed convergence plot...")
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        fig.suptitle("Convergence Curves with 95% Confidence Bands",
                     fontsize=14, fontweight="bold")

        plot_configs = [
            ("No-DP", "seeds10"),
            ("eps=10", "dp"),
        ]

        for row, (dp_lbl, source) in enumerate(plot_configs):
            for col, ds in enumerate(datasets_plot):
                ax = axes[row, col]
                for algo in ["FedAvg", "Ditto", "HPFL"]:
                    if source == "seeds10":
                        ckey = "seeds10_{}_{}".format(ds, algo)
                    else:
                        ckey = "dp_eps10_{}_{}".format(ds, algo)

                    if ckey in convergence_data:
                        d = convergence_data[ckey]
                        mean = np.array(d["trajectory_mean"])
                        std = np.array(d["trajectory_std"])
                        rounds = list(range(1, len(mean) + 1))
                        color = algo_colors.get(algo, "#666666")

                        ax.plot(rounds, mean * 100, color=color,
                                linewidth=1.5, label=algo)
                        ax.fill_between(rounds,
                                        (mean - 1.96 * std) * 100,
                                        (mean + 1.96 * std) * 100,
                                        color=color, alpha=0.15)

                ax.set_title("{} ({})".format(ds.replace("_", " "), dp_lbl),
                             fontsize=10)
                ax.set_xlabel("Round")
                ax.set_ylabel("Accuracy (%)")
                ax.grid(True, alpha=0.3)
                if col == 0 and row == 0:
                    ax.legend(fontsize=8)

        plt.tight_layout()
        plt.savefig(str(OUTPUT_DIR / "convergence_curves.pdf"),
                    bbox_inches="tight", dpi=150)
        plt.savefig(str(OUTPUT_DIR / "convergence_curves.png"),
                    bbox_inches="tight", dpi=150)
        plt.close()
        log("    Saved convergence_curves.pdf")

        # ---- Plot 3: Significance heatmap ----
        if "block_a_significance" in ckpt:
            log("  Generating significance heatmap...")
            sig_results = ckpt["block_a_significance"]

            # Build heatmap for seeds10 comparisons
            s10_results = {k: v for k, v in sig_results.items()
                           if v.get("source") == "seeds10"}
            if s10_results:
                fig, ax = plt.subplots(figsize=(10, 5))
                comps = ["HPFL vs FedAvg", "HPFL vs Ditto", "Ditto vs FedAvg"]
                datasets_hm = ["Breast_Cancer", "Cardiovascular", "PTB_XL"]

                data_matrix = np.full((len(comps), len(datasets_hm)), np.nan)
                annot_matrix = [[""]*len(datasets_hm) for _ in range(len(comps))]

                for i, comp in enumerate(comps):
                    algo_a, algo_b = comp.split(" vs ")
                    for j, ds in enumerate(datasets_hm):
                        rkey = "seeds10_{}_{}_vs_{}".format(ds, algo_a, algo_b)
                        if rkey in s10_results:
                            r = s10_results[rkey]
                            pval = r.get("p_value_ttest")
                            diff = r.get("mean_diff", 0)
                            if pval is not None:
                                data_matrix[i, j] = diff
                                sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
                                annot_matrix[i][j] = "{:+.3f}\np={:.4f}{}".format(
                                    diff, pval, "\n" + sig if sig else "")

                im = ax.imshow(data_matrix, cmap="RdYlGn", aspect="auto",
                               vmin=-0.15, vmax=0.15)
                ax.set_xticks(range(len(datasets_hm)))
                ax.set_xticklabels([d.replace("_", "\n") for d in datasets_hm])
                ax.set_yticks(range(len(comps)))
                ax.set_yticklabels(comps)

                for i in range(len(comps)):
                    for j in range(len(datasets_hm)):
                        ax.text(j, i, annot_matrix[i][j], ha="center", va="center",
                                fontsize=8, color="black")

                plt.colorbar(im, label="Mean Accuracy Difference")
                ax.set_title("Statistical Significance (seeds10, No-DP, paired t-test)",
                             fontsize=12)
                plt.tight_layout()
                plt.savefig(str(OUTPUT_DIR / "significance_table.pdf"),
                            bbox_inches="tight", dpi=150)
                plt.savefig(str(OUTPUT_DIR / "significance_table.png"),
                            bbox_inches="tight", dpi=150)
                plt.close()
                log("    Saved significance_table.pdf")

        # ---- Plot 4: Communication costs bar chart ----
        if "block_b_comm_costs" in ckpt:
            log("  Generating communication costs chart...")
            cc = ckpt["block_b_comm_costs"]

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            fig.suptitle("Communication Costs per Full Training Run (K=5 clients)",
                         fontsize=13, fontweight="bold")

            for col, ds in enumerate(["Cardiovascular", "PTB_XL", "Breast_Cancer"]):
                ax = axes[col]
                R = DATASET_CONFIGS[ds]["num_rounds"]
                algos = ["FedAvg", "Ditto", "HPFL"]
                topks = ["NoTopK", "Top5pct", "Top1pct"]
                x = np.arange(len(algos))
                width = 0.25

                for i, tk in enumerate(topks):
                    vals = []
                    for algo in algos:
                        rkey = "{}_{}_{}_{}_K5".format(ds, algo, tk, "R{}".format(R))
                        if rkey in cc:
                            vals.append(cc[rkey]["total_MB"])
                        else:
                            vals.append(0)
                    ax.bar(x + i * width, vals, width, label=tk, alpha=0.8)

                ax.set_title(ds.replace("_", " "))
                ax.set_xlabel("Algorithm")
                ax.set_ylabel("Total Communication (MB)")
                ax.set_xticks(x + width)
                ax.set_xticklabels(algos)
                if col == 0:
                    ax.legend(fontsize=8)

            plt.tight_layout()
            plt.savefig(str(OUTPUT_DIR / "comm_costs_table.pdf"),
                        bbox_inches="tight", dpi=150)
            plt.savefig(str(OUTPUT_DIR / "comm_costs_table.png"),
                        bbox_inches="tight", dpi=150)
            plt.close()
            log("    Saved comm_costs_table.pdf")

    elapsed = time.time() - start
    ckpt["metadata"]["block_d_time"] = round(elapsed, 1)
    save_checkpoint(ckpt)
    log("  Block D complete ({:.1f}s).".format(elapsed))


# ======================================================================
# Main
# ======================================================================

def main():
    global _log_file

    parser = argparse.ArgumentParser(
        description="FL-EHDS: Cascading Analysis & Experiments")
    parser.add_argument("--quick", action="store_true",
                        help="Quick validation (5 rounds for training)")
    parser.add_argument("--fresh", action="store_true",
                        help="Delete checkpoint and start fresh")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _log_file = open(str(OUTPUT_DIR / LOG_FILE), "a")

    log("=" * 70)
    log("FL-EHDS: Cascading Analysis & Experiments")
    log("Mode: {}".format("QUICK" if args.quick else "FULL"))
    log("Device: {}".format(_detect_device("cpu")))
    log("Blocks: A(significance) -> B(comm costs) -> C(ablation) -> D(plots)")
    log("=" * 70)

    if args.fresh:
        for suffix in ["", ".bak"]:
            p = OUTPUT_DIR / (CHECKPOINT_FILE + suffix)
            if p.exists():
                p.unlink()
                log("Deleted {}".format(CHECKPOINT_FILE + suffix))

    ckpt = load_checkpoint()
    if ckpt is None:
        ckpt = {
            "metadata": {
                "experiment": "Cascading Analysis & Experiments",
                "started": datetime.now().isoformat(),
            },
            "results": {},
            "completed": [],
        }

    cascade_start = time.time()

    # Block A: Statistical Significance (analytical)
    block_a_significance(ckpt)

    if _shutdown:
        log("Shutdown after Block A")
    else:
        # Block B: Communication Costs (analytical)
        block_b_comm_costs(ckpt)

    if not _shutdown:
        # Block C: HPFL Ablation (training)
        block_c_ablation(ckpt, quick=args.quick)

    if not _shutdown:
        # Block D: Convergence Plots (analytical)
        block_d_convergence(ckpt)

    total_time = time.time() - cascade_start
    ckpt["metadata"]["total_time"] = format_time(total_time)
    save_checkpoint(ckpt)

    log("")
    log("=" * 70)
    log("ALL BLOCKS COMPLETE in {}".format(format_time(total_time)))
    log("Checkpoint: {}".format(CHECKPOINT_FILE))

    # Final summary
    block_c_done = sum(1 for k in ckpt.get("completed", []) if k.startswith("C_"))
    log("  Block A: {} significance tests".format(
        len(ckpt.get("block_a_significance", {}))))
    log("  Block B: {} cost calculations".format(
        len(ckpt.get("block_b_comm_costs", {}))))
    log("  Block C: {}/24 ablation experiments".format(block_c_done))
    log("  Block D: {} convergence curves".format(
        len(ckpt.get("block_d_convergence", {}))))
    log("=" * 70)

    if _log_file:
        _log_file.close()


if __name__ == "__main__":
    main()

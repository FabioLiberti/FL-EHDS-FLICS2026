#!/usr/bin/env python3
"""
FL-EHDS Cascading Analysis & Experiments — Phase 3.

4 blocks executed sequentially from fastest to slowest:

Block A: Data Heterogeneity Characterization (analytical, ~1 min)
  Loads datasets with varying Dirichlet alpha, computes per-client
  heterogeneity metrics (label entropy, class imbalance ratio, feature
  KL divergence), and correlates with existing performance results to
  produce quantitative algorithm selection rules.

Block B: Privacy-Equity-Privacy (PEP) Triangle (analytical, ~1 min)
  Unifies accuracy, fairness, and privacy data from existing checkpoints
  (dp, seeds10, mia) to build a 3D Pareto frontier and identify optimal
  configurations for EHDS deployment.

Block C: Data Efficiency Curves (training, ~7 min)
  27 new experiments: 3 subsampling fractions (25%, 50%, 75%) x 3 algorithms
  x 3 datasets. Shows sample efficiency per algorithm — critical for EHDS
  hospitals with varying patient volumes.

Block D: Compound Stress Test (training, ~12 min)
  48 new experiments: 2 datasets x 3 algorithms x 2 alpha x 2 DP x
  2 participation levels (3/5 and 5/5 active clients). Tests the realistic
  EHDS worst-case: non-IID + DP + intermittent hospital participation.

Usage:
    cd fl-ehds-framework
    python -m benchmarks.run_analysis_cascade3 [--quick] [--fresh]

Output:
    benchmarks/paper_results_tabular/checkpoint_cascade3.json
    benchmarks/paper_results_tabular/heterogeneity_characterization.pdf
    benchmarks/paper_results_tabular/pep_triangle.pdf
    benchmarks/paper_results_tabular/data_efficiency.pdf
    benchmarks/paper_results_tabular/compound_stress.pdf

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
import random
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
CHECKPOINT_FILE = "checkpoint_cascade3.json"
LOG_FILE = "experiment_cascade3.log"

ALGORITHMS = ["FedAvg", "Ditto", "HPFL"]

DATASET_CONFIGS = {
    "Cardiovascular": dict(
        learning_rate=0.01, batch_size=64, num_rounds=30, local_epochs=3,
        mu=0.1, num_clients=5, input_dim=11, num_classes=2,
    ),
    "PTB_XL": dict(
        learning_rate=0.005, batch_size=64, num_rounds=30, local_epochs=3,
        mu=0.1, num_clients=5, input_dim=9, num_classes=5,
    ),
    "Breast_Cancer": dict(
        learning_rate=0.005, batch_size=32, num_rounds=30, local_epochs=2,
        mu=0.1, num_clients=4, input_dim=30, num_classes=2,
    ),
}

SEEDS_10 = [7, 31, 137, 577, 1337]
SEEDS_DP = [42, 123, 456, 789, 999]

# Subsampling fractions for data efficiency
DATA_FRACTIONS = [0.25, 0.50, 0.75]

# Compound stress parameters
COMPOUND_ALPHAS = [0.1, 0.5]
COMPOUND_DP = [
    {"label": "No-DP", "dp_enabled": False, "dp_epsilon": 0},
    {"label": "DP-eps10", "dp_enabled": True, "dp_epsilon": 10.0},
]
COMPOUND_PARTICIPATION = [3, 5]  # active out of 5

# ======================================================================
# Logging & signals
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
    fd, tmp = tempfile.mkstemp(dir=str(OUTPUT_DIR), prefix=".cas3_", suffix=".tmp")
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
# Shared helpers
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


def _compute_jain_index(per_client_acc):
    accs = list(per_client_acc.values())
    if not accs:
        return 0.0
    accs = np.array(accs)
    n = len(accs)
    if n == 0 or np.sum(accs ** 2) == 0:
        return 0.0
    return float((np.sum(accs) ** 2) / (n * np.sum(accs ** 2)))


def _compute_gini(values):
    arr = np.array(sorted(values))
    n = len(arr)
    if n == 0 or np.sum(arr) == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * arr) - (n + 1) * np.sum(arr)) / (n * np.sum(arr)))


def format_time(seconds):
    return str(timedelta(seconds=int(seconds)))


_data_cache = {}


def load_dataset(dataset, num_clients, seed, is_iid=True, alpha=0.5):
    cache_key = (dataset, num_clients, seed, is_iid, alpha)
    if cache_key in _data_cache:
        return _data_cache[cache_key]

    if dataset == "Cardiovascular":
        client_data, client_test, meta = load_cardiovascular_data(
            num_clients=num_clients, seed=seed, is_iid=is_iid, alpha=alpha,
        )
    elif dataset == "PTB_XL":
        client_data, client_test, meta = load_ptbxl_data(
            num_clients=num_clients, seed=seed,
            partition_by_site=False if not is_iid else True,
            is_iid=is_iid, alpha=alpha,
            min_site_samples=50,
        )
    elif dataset == "Breast_Cancer":
        client_data, client_test, meta = load_breast_cancer_data(
            num_clients=num_clients, seed=seed,
        )
    else:
        raise ValueError("Unknown dataset: {}".format(dataset))

    _data_cache[cache_key] = (client_data, client_test, meta)
    return client_data, client_test, meta


def _subsample_client_data(client_data, fraction, seed):
    """Subsample each client's training data by a fraction."""
    rng = np.random.RandomState(seed)
    subsampled = {}
    for cid, (X, y) in client_data.items():
        n = len(y)
        k = max(1, int(n * fraction))
        indices = rng.choice(n, k, replace=False)
        subsampled[cid] = (X[indices], y[indices])
    return subsampled


# ======================================================================
# Block A: Data Heterogeneity Characterization
# ======================================================================

def _label_entropy(y, num_classes):
    """Compute entropy of label distribution."""
    counts = np.bincount(y.astype(int), minlength=num_classes)
    probs = counts / len(y) if len(y) > 0 else counts
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs))) if len(probs) > 0 else 0.0


def _class_imbalance_ratio(y, num_classes):
    """Ratio of majority to minority class count."""
    counts = np.bincount(y.astype(int), minlength=num_classes)
    counts = counts[counts > 0]
    if len(counts) < 2:
        return float("inf")
    return float(np.max(counts) / np.min(counts))


def _feature_kl_divergence(X_client, X_global, n_bins=20):
    """Approximate KL divergence of feature distributions."""
    kl_sum = 0.0
    for col in range(X_client.shape[1]):
        # Discretize both distributions
        combined = np.concatenate([X_client[:, col], X_global[:, col]])
        bins = np.linspace(combined.min() - 1e-8, combined.max() + 1e-8, n_bins + 1)

        hist_c, _ = np.histogram(X_client[:, col], bins=bins, density=True)
        hist_g, _ = np.histogram(X_global[:, col], bins=bins, density=True)

        # Add epsilon for numerical stability
        hist_c = hist_c + 1e-10
        hist_g = hist_g + 1e-10
        hist_c = hist_c / hist_c.sum()
        hist_g = hist_g / hist_g.sum()

        kl_sum += float(np.sum(hist_c * np.log(hist_c / hist_g)))
    return kl_sum / X_client.shape[1]  # Average across features


def block_a_heterogeneity(ckpt):
    """Block A: Data heterogeneity characterization."""
    log("=" * 70)
    log("BLOCK A — Data Heterogeneity Characterization")
    log("=" * 70)

    start = time.time()
    results = {}

    datasets_to_test = ["Cardiovascular", "PTB_XL"]  # BC has too few samples for meaningful Dirichlet
    alphas = [0.1, 0.5, 1.0]
    seed = 42

    for ds in datasets_to_test:
        dcfg = DATASET_CONFIGS[ds]
        nc = dcfg["num_classes"]
        num_clients = dcfg["num_clients"]

        # IID baseline
        log("  {} — loading IID baseline...".format(ds))
        cd_iid, ct_iid, _ = load_dataset(ds, num_clients, seed, is_iid=True)

        # Combine all IID data for global reference
        X_global = np.concatenate([cd_iid[c][0] for c in cd_iid])
        y_global = np.concatenate([cd_iid[c][1] for c in cd_iid])

        for condition_label, is_iid, alpha in [("IID", True, 0.5)] + \
                [("alpha={}".format(a), False, a) for a in alphas]:

            log("  {} — {} ...".format(ds, condition_label))
            client_data, _, _ = load_dataset(ds, num_clients, seed,
                                              is_iid=is_iid, alpha=alpha)

            per_client_metrics = {}
            for cid in range(num_clients):
                X_c, y_c = client_data[cid]
                per_client_metrics[str(cid)] = {
                    "n_samples": len(y_c),
                    "label_entropy": round(_label_entropy(y_c, nc), 4),
                    "class_imbalance_ratio": round(_class_imbalance_ratio(y_c, nc), 2),
                    "feature_kl_div": round(_feature_kl_divergence(X_c, X_global), 4),
                    "label_distribution": np.bincount(y_c.astype(int),
                                                      minlength=nc).tolist(),
                }

            # Aggregate heterogeneity metrics
            entropies = [m["label_entropy"] for m in per_client_metrics.values()]
            imbalances = [m["class_imbalance_ratio"] for m in per_client_metrics.values()
                          if m["class_imbalance_ratio"] != float("inf")]
            kl_divs = [m["feature_kl_div"] for m in per_client_metrics.values()]
            sample_counts = [m["n_samples"] for m in per_client_metrics.values()]

            rkey = "{}_{}".format(ds, condition_label)
            results[rkey] = {
                "dataset": ds,
                "condition": condition_label,
                "num_clients": num_clients,
                "per_client": per_client_metrics,
                "aggregate": {
                    "mean_label_entropy": round(float(np.mean(entropies)), 4),
                    "std_label_entropy": round(float(np.std(entropies)), 4),
                    "mean_class_imbalance": round(float(np.mean(imbalances)), 2) if imbalances else None,
                    "std_class_imbalance": round(float(np.std(imbalances)), 2) if imbalances else None,
                    "mean_feature_kl_div": round(float(np.mean(kl_divs)), 4),
                    "std_feature_kl_div": round(float(np.std(kl_divs)), 4),
                    "sample_count_cv": round(float(np.std(sample_counts) / np.mean(sample_counts)), 4)
                        if np.mean(sample_counts) > 0 else 0,
                },
            }

    # ---- Cross-reference with noniid_dp performance data ----
    noniid_path = OUTPUT_DIR / "checkpoint_noniid_dp.json"
    correlation_table = {}
    if noniid_path.exists():
        log("  Cross-referencing with noniid_dp performance...")
        noniid = json.load(open(noniid_path))
        noniid_data = noniid.get("results", {})

        for ds in datasets_to_test:
            ds_short = "CV" if ds == "Cardiovascular" else "PX"
            for alpha in alphas:
                condition_label = "alpha={}".format(alpha)
                rkey_het = "{}_{}".format(ds, condition_label)
                if rkey_het not in results:
                    continue

                het = results[rkey_het]["aggregate"]

                for algo in ALGORITHMS:
                    # Get accuracy from noniid_dp (avg across seeds, No-DP)
                    accs = []
                    for s in [42, 123, 456]:
                        nkey = "{}_{}_a{}_No-DP_s{}".format(ds_short, algo, alpha, s)
                        if nkey in noniid_data:
                            accs.append(noniid_data[nkey].get("final_accuracy", 0))

                    if accs:
                        ckey = "{}_{}_{}".format(ds, condition_label, algo)
                        correlation_table[ckey] = {
                            "dataset": ds,
                            "alpha": alpha,
                            "algorithm": algo,
                            "mean_accuracy": round(float(np.mean(accs)), 4),
                            "mean_label_entropy": het["mean_label_entropy"],
                            "mean_class_imbalance": het["mean_class_imbalance"],
                            "mean_feature_kl_div": het["mean_feature_kl_div"],
                            "sample_count_cv": het["sample_count_cv"],
                        }

    elapsed = time.time() - start

    # Print summary
    log("")
    log("  HETEROGENEITY CHARACTERIZATION:")
    log("  {:<15} {:<12} {:>10} {:>10} {:>10} {:>10}".format(
        "Dataset", "Condition", "LabelEnt", "Imbalance", "FeatureKL", "SampleCV"))
    log("  " + "-" * 70)
    for rkey in sorted(results.keys()):
        r = results[rkey]
        agg = r["aggregate"]
        log("  {:<15} {:<12} {:>8.3f}   {:>8.1f}   {:>8.4f}   {:>8.3f}".format(
            r["dataset"][:12], r["condition"],
            agg["mean_label_entropy"],
            agg["mean_class_imbalance"] if agg["mean_class_imbalance"] else 0,
            agg["mean_feature_kl_div"],
            agg["sample_count_cv"]))

    if correlation_table:
        log("")
        log("  HETEROGENEITY → ACCURACY CORRELATION:")
        log("  {:<15} {:<10} {:<8} {:>8} {:>10} {:>10}".format(
            "Dataset", "Alpha", "Algo", "Acc%", "LabelEnt", "FeatureKL"))
        log("  " + "-" * 65)
        for ckey in sorted(correlation_table.keys()):
            c = correlation_table[ckey]
            log("  {:<15} {:<10} {:<8} {:>6.1f}%  {:>8.3f}   {:>8.4f}".format(
                c["dataset"][:12], c["alpha"], c["algorithm"],
                c["mean_accuracy"] * 100,
                c["mean_label_entropy"],
                c["mean_feature_kl_div"]))

    ckpt["block_a_heterogeneity"] = results
    ckpt["block_a_correlation"] = correlation_table
    ckpt["metadata"]["block_a_time"] = round(elapsed, 1)
    save_checkpoint(ckpt)
    log("  Block A complete ({:.1f}s). {} heterogeneity profiles, {} correlation entries.".format(
        elapsed, len(results), len(correlation_table)))
    return results, correlation_table


# ======================================================================
# Block B: Privacy-Equity-Privacy (PEP) Triangle
# ======================================================================

def block_b_pep_triangle(ckpt):
    """Block B: Privacy-Equity-Privacy triangle from existing checkpoints."""
    log("")
    log("=" * 70)
    log("BLOCK B — Privacy-Equity-Privacy (PEP) Triangle")
    log("=" * 70)

    start = time.time()
    results = {}

    datasets = ["Breast_Cancer", "Cardiovascular", "PTB_XL"]

    # ---- Load all data sources ----

    # Source 1: seeds10 (No-DP accuracy + Jain)
    s10_path = OUTPUT_DIR / "checkpoint_seeds10.json"
    s10_data = {}
    if s10_path.exists():
        s10 = json.load(open(s10_path))
        s10_data = s10.get("completed", {})
        log("  Loaded seeds10: {} experiments".format(len(s10_data)))

    # Source 2: dp (multi-epsilon accuracy + Jain)
    dp_path = OUTPUT_DIR / "checkpoint_dp.json"
    dp_data_dict = {}
    if dp_path.exists():
        dp_ckpt = json.load(open(dp_path))
        dp_data_dict = dp_ckpt.get("completed", {})
        log("  Loaded dp: {} experiments".format(len(dp_data_dict)))

    # Source 3: MIA (privacy metrics)
    mia_path = OUTPUT_DIR / "checkpoint_mia.json"
    mia_data = {}
    if mia_path.exists():
        mia_ckpt = json.load(open(mia_path))
        mia_data = mia_ckpt.get("results", {})
        log("  Loaded mia: {} experiments".format(len(mia_data)))

    # ---- Build PEP triangle for each dataset × algorithm ----
    for ds in datasets:
        for algo in ALGORITHMS:
            points = []

            # No-DP point (from seeds10)
            accs_nodp = []
            jains_nodp = []
            for s in SEEDS_10:
                key = "{}_{}_s{}".format(ds, algo, s)
                if key in s10_data:
                    pca = s10_data[key].get("per_client_acc", {})
                    if pca:
                        accs_nodp.append(float(np.mean(list(pca.values()))))
                        jains_nodp.append(_compute_jain_index(pca))

            mia_auc_nodp = None
            mia_key = "{}_{}_No-DP_s42".format(ds, algo)
            if mia_key in mia_data:
                mia_auc_nodp = mia_data[mia_key].get("mia_overall_auc")

            if accs_nodp:
                points.append({
                    "epsilon": "No-DP",
                    "accuracy": round(float(np.mean(accs_nodp)), 4),
                    "accuracy_std": round(float(np.std(accs_nodp)), 4),
                    "jain": round(float(np.mean(jains_nodp)), 4) if jains_nodp else None,
                    "mia_auc": mia_auc_nodp,
                    "privacy_score": round(1.0 - mia_auc_nodp, 4) if mia_auc_nodp else None,
                })

            # DP points
            for eps in [1, 5, 10, 50]:
                accs = []
                jains = []
                for s in SEEDS_DP:
                    key = "{}_{}_eps{}_s{}".format(ds, algo, eps, s)
                    if key in dp_data_dict:
                        pca = dp_data_dict[key].get("per_client_acc", {})
                        if pca:
                            accs.append(float(np.mean(list(pca.values()))))
                            jains.append(_compute_jain_index(pca))

                mia_auc_dp = None
                mia_key_dp = "{}_{}_DP-eps{}_s42".format(ds, algo, eps)
                if mia_key_dp in mia_data:
                    mia_auc_dp = mia_data[mia_key_dp].get("mia_overall_auc")

                if accs:
                    points.append({
                        "epsilon": eps,
                        "accuracy": round(float(np.mean(accs)), 4),
                        "accuracy_std": round(float(np.std(accs)), 4),
                        "jain": round(float(np.mean(jains)), 4) if jains else None,
                        "mia_auc": mia_auc_dp,
                        "privacy_score": round(1.0 - mia_auc_dp, 4) if mia_auc_dp else None,
                    })

            if points:
                # Compute equity retention ratio (Jain under DP / Jain at No-DP)
                nodp_jain = points[0]["jain"] if points[0]["jain"] else None
                equity_retention = {}
                if nodp_jain and nodp_jain > 0:
                    for p in points[1:]:
                        if p["jain"] is not None:
                            equity_retention["eps={}".format(p["epsilon"])] = round(
                                p["jain"] / nodp_jain, 4)

                # Identify Pareto-optimal (max accuracy subject to Jain >= threshold)
                fair_points = [p for p in points if p["jain"] and p["jain"] >= 0.9]
                pareto_point = max(fair_points, key=lambda x: x["accuracy"]) if fair_points else None

                rkey = "{}_{}".format(ds, algo)
                results[rkey] = {
                    "dataset": ds,
                    "algorithm": algo,
                    "operating_points": points,
                    "equity_retention": equity_retention,
                    "pareto_fair_point": {
                        "epsilon": pareto_point["epsilon"] if pareto_point else None,
                        "accuracy": pareto_point["accuracy"] if pareto_point else None,
                        "jain": pareto_point["jain"] if pareto_point else None,
                    } if pareto_point else None,
                    "has_mia_data": any(p["mia_auc"] is not None for p in points),
                }

    elapsed = time.time() - start

    # Print summary
    log("")
    log("  PEP TRIANGLE SUMMARY:")
    log("  {:<15} {:<8} {:>8} {:>8} {:>8} {:>10} {:>12}".format(
        "Dataset", "Algo", "NoDP%", "Jain", "MIA_AUC", "EqRetain", "Pareto"))
    log("  " + "-" * 75)
    for ds in datasets:
        for algo in ALGORITHMS:
            rkey = "{}_{}".format(ds, algo)
            if rkey in results:
                r = results[rkey]
                pts = r["operating_points"]
                nodp = pts[0] if pts else {}
                nodp_acc = nodp.get("accuracy", 0) * 100
                nodp_jain = nodp.get("jain", 0) or 0
                nodp_mia = nodp.get("mia_auc")
                mia_str = "{:.3f}".format(nodp_mia) if nodp_mia else "N/A"

                # Equity retention at eps=10
                eq_ret = r["equity_retention"].get("eps=10", None)
                eq_str = "{:.3f}".format(eq_ret) if eq_ret else "N/A"

                pareto = r.get("pareto_fair_point")
                par_str = "eps={} {:.1f}%".format(
                    pareto["epsilon"], pareto["accuracy"] * 100) if pareto else "N/A"

                log("  {:<15} {:<8} {:>6.1f}%  {:>6.3f}  {:>8}  {:>10}  {:>12}".format(
                    ds[:12], algo, nodp_acc, nodp_jain, mia_str, eq_str, par_str))
        log("  " + "-" * 75)

    ckpt["block_b_pep_triangle"] = results
    ckpt["metadata"]["block_b_time"] = round(elapsed, 1)
    save_checkpoint(ckpt)
    log("  Block B complete ({:.1f}s). {} PEP profiles.".format(elapsed, len(results)))
    return results


# ======================================================================
# Block C: Data Efficiency Curves (TRAINING)
# ======================================================================

def run_efficiency_experiment(dataset, algo, fraction, seed, quick=False):
    """Run one data efficiency experiment with subsampled training data."""
    start = time.time()
    dcfg = DATASET_CONFIGS[dataset]
    num_rounds = 5 if quick else dcfg["num_rounds"]

    client_data, client_test, meta = load_dataset(
        dataset, dcfg["num_clients"], seed, is_iid=True)

    # Subsample training data
    if fraction < 1.0:
        client_data = _subsample_client_data(client_data, fraction, seed)

    trainer = FederatedTrainer(
        num_clients=dcfg["num_clients"], algorithm=algo,
        local_epochs=dcfg["local_epochs"], batch_size=dcfg["batch_size"],
        learning_rate=dcfg["learning_rate"], mu=dcfg["mu"], seed=seed,
        external_data=client_data, external_test_data=client_test,
        input_dim=dcfg["input_dim"], num_classes=dcfg["num_classes"],
    )

    history = []
    best_acc = 0.0

    for r in range(num_rounds):
        result = trainer.train_round(r)
        if algo == "HPFL":
            per_client_acc_round = _evaluate_per_client(trainer)
            round_acc = float(np.mean(list(per_client_acc_round.values())))
        else:
            round_acc = result.global_acc
        history.append({"round": r + 1, "accuracy": round_acc})
        if round_acc > best_acc:
            best_acc = round_acc

    per_client_acc = _evaluate_per_client(trainer)
    jain = _compute_jain_index(per_client_acc)
    final_acc = float(np.mean(list(per_client_acc.values())))

    # Count actual training samples
    total_samples = sum(len(client_data[c][1]) for c in client_data)

    elapsed = time.time() - start
    return {
        "block": "C_data_efficiency",
        "dataset": dataset, "algorithm": algo,
        "data_fraction": fraction,
        "total_train_samples": total_samples,
        "seed": seed, "num_rounds": num_rounds,
        "final_accuracy": round(final_acc, 4),
        "best_accuracy": round(best_acc, 4),
        "per_client_accuracy": per_client_acc,
        "jain_index": round(jain, 4),
        "convergence_trajectory": [h["accuracy"] for h in history],
        "time_seconds": round(elapsed, 1),
    }


def block_c_data_efficiency(ckpt, quick=False):
    """Block C: Data efficiency curves."""
    log("")
    log("=" * 70)
    log("BLOCK C — Data Efficiency Curves")
    log("=" * 70)

    start = time.time()
    datasets = ["Cardiovascular", "PTB_XL", "Breast_Cancer"]
    seed = 42

    experiments = []
    for ds in datasets:
        for algo in ALGORITHMS:
            for frac in DATA_FRACTIONS:
                exp_key = "C_{}_{}_{:.2f}".format(ds, algo, frac)
                experiments.append((exp_key, ds, algo, frac))

    total = len(experiments)
    done = sum(1 for e in experiments if e[0] in ckpt.get("completed", []))
    log("  {} experiments ({} already done)".format(total, done))

    for idx, (exp_key, ds, algo, frac) in enumerate(experiments, 1):
        if _shutdown:
            log("  Shutdown requested")
            break
        if exp_key in ckpt.get("completed", []):
            continue

        try:
            log("  [{}/{}] {} | {} | {:.0f}% data ...".format(
                idx, total, ds, algo, frac * 100))

            result = run_efficiency_experiment(ds, algo, frac, seed, quick=quick)

            ckpt["results"][exp_key] = result
            if "completed" not in ckpt:
                ckpt["completed"] = []
            ckpt["completed"].append(exp_key)
            save_checkpoint(ckpt)

            log("    -> acc={:.1f}% | samples={} | Jain={:.3f} | {:.0f}s".format(
                result["final_accuracy"] * 100,
                result["total_train_samples"],
                result["jain_index"],
                result["time_seconds"]))

            _cleanup_gpu()
        except Exception as e:
            log("    ERROR: {}".format(e))
            log("    " + traceback.format_exc().split("\n")[-2])
            _cleanup_gpu()

    elapsed = time.time() - start
    ckpt["metadata"]["block_c_time"] = round(elapsed, 1)
    save_checkpoint(ckpt)

    # Print summary table
    log("")
    log("  DATA EFFICIENCY RESULTS:")
    log("  {:<15} {:<8} {:>8} {:>8} {:>8} {:>8}".format(
        "Dataset", "Algo", "25%", "50%", "75%", "100%*"))
    log("  " + "-" * 60)

    # Get 100% baseline from seeds10 or cascade1
    baselines = {}
    s10_path = OUTPUT_DIR / "checkpoint_seeds10.json"
    if s10_path.exists():
        s10 = json.load(open(s10_path))
        s10_data = s10.get("completed", {})
        for ds in datasets:
            for algo in ALGORITHMS:
                key = "{}_{}_s42".format(ds, algo)
                if key in s10_data:
                    pca = s10_data[key].get("per_client_acc", {})
                    if pca:
                        baselines["{}_{}".format(ds, algo)] = float(np.mean(list(pca.values())))

    for ds in datasets:
        for algo in ALGORITHMS:
            vals = []
            for frac in DATA_FRACTIONS:
                exp_key = "C_{}_{}_{:.2f}".format(ds, algo, frac)
                if exp_key in ckpt.get("results", {}):
                    vals.append(ckpt["results"][exp_key]["final_accuracy"] * 100)
                else:
                    vals.append(0)
            baseline = baselines.get("{}_{}".format(ds, algo), 0) * 100
            log("  {:<15} {:<8} {:>6.1f}%  {:>6.1f}%  {:>6.1f}%  {:>6.1f}%".format(
                ds[:12], algo, vals[0], vals[1], vals[2], baseline))
        log("  " + "-" * 60)
    log("  (* 100% from seeds10 s42 baseline)")

    log("  Block C complete ({}).".format(format_time(elapsed)))


# ======================================================================
# Block D: Compound Stress Test (TRAINING)
# ======================================================================

def run_compound_experiment(dataset, algo, alpha, dp_cfg, active_k, seed, quick=False):
    """Run one compound stress experiment: non-IID × DP × partial participation."""
    start = time.time()
    dcfg = DATASET_CONFIGS[dataset]
    num_rounds = 5 if quick else dcfg["num_rounds"]
    num_clients = dcfg["num_clients"]

    client_data, client_test, meta = load_dataset(
        dataset, num_clients, seed, is_iid=False, alpha=alpha)

    trainer_kwargs = dict(
        num_clients=num_clients, algorithm=algo,
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

    history = []
    best_acc = 0.0
    rng = random.Random(seed)
    all_clients = list(range(num_clients))

    for r in range(num_rounds):
        # Partial participation: randomly sample active_k clients each round
        if active_k < num_clients:
            active = sorted(rng.sample(all_clients, active_k))
        else:
            active = None  # All clients participate

        result = trainer.train_round(r, active_clients=active)

        if algo == "HPFL":
            per_client_acc_round = _evaluate_per_client(trainer)
            round_acc = float(np.mean(list(per_client_acc_round.values())))
        else:
            round_acc = result.global_acc

        history.append({"round": r + 1, "accuracy": round_acc})
        if round_acc > best_acc:
            best_acc = round_acc

    per_client_acc = _evaluate_per_client(trainer)
    jain = _compute_jain_index(per_client_acc)
    final_acc = float(np.mean(list(per_client_acc.values())))

    elapsed = time.time() - start
    return {
        "block": "D_compound_stress",
        "dataset": dataset, "algorithm": algo,
        "alpha": alpha, "dp_level": dp_cfg["label"],
        "dp_epsilon": dp_cfg["dp_epsilon"],
        "active_clients": active_k, "total_clients": num_clients,
        "participation_rate": round(active_k / num_clients, 2),
        "seed": seed, "num_rounds": num_rounds,
        "final_accuracy": round(final_acc, 4),
        "best_accuracy": round(best_acc, 4),
        "per_client_accuracy": per_client_acc,
        "jain_index": round(jain, 4),
        "convergence_trajectory": [h["accuracy"] for h in history],
        "time_seconds": round(elapsed, 1),
    }


def block_d_compound_stress(ckpt, quick=False):
    """Block D: Compound stress test (non-IID × DP × partial participation)."""
    log("")
    log("=" * 70)
    log("BLOCK D — Compound Stress Test")
    log("=" * 70)

    start = time.time()
    datasets = ["Cardiovascular", "PTB_XL"]
    seed = 42

    experiments = []
    for ds in datasets:
        for algo in ALGORITHMS:
            for alpha in COMPOUND_ALPHAS:
                for dp_cfg in COMPOUND_DP:
                    for active_k in COMPOUND_PARTICIPATION:
                        exp_key = "D_{}_{}_{}_{}_{}_{}".format(
                            ds, algo, alpha, dp_cfg["label"], active_k,
                            DATASET_CONFIGS[ds]["num_clients"])
                        experiments.append((exp_key, ds, algo, alpha, dp_cfg, active_k))

    total = len(experiments)
    done = sum(1 for e in experiments if e[0] in ckpt.get("completed", []))
    log("  {} experiments ({} already done)".format(total, done))

    for idx, (exp_key, ds, algo, alpha, dp_cfg, active_k) in enumerate(experiments, 1):
        if _shutdown:
            log("  Shutdown requested")
            break
        if exp_key in ckpt.get("completed", []):
            continue

        try:
            part_lbl = "{}/{}".format(active_k, DATASET_CONFIGS[ds]["num_clients"])
            log("  [{}/{}] {} | {} | a={} | {} | part={} ...".format(
                idx, total, ds, algo, alpha, dp_cfg["label"], part_lbl))

            result = run_compound_experiment(
                ds, algo, alpha, dp_cfg, active_k, seed, quick=quick)

            ckpt["results"][exp_key] = result
            if "completed" not in ckpt:
                ckpt["completed"] = []
            ckpt["completed"].append(exp_key)
            save_checkpoint(ckpt)

            log("    -> acc={:.1f}% | Jain={:.3f} | {:.0f}s".format(
                result["final_accuracy"] * 100,
                result["jain_index"],
                result["time_seconds"]))

            _cleanup_gpu()
        except Exception as e:
            log("    ERROR: {}".format(e))
            log("    " + traceback.format_exc().split("\n")[-2])
            _cleanup_gpu()

    elapsed = time.time() - start
    ckpt["metadata"]["block_d_time"] = round(elapsed, 1)
    save_checkpoint(ckpt)

    # Print compound stress matrix
    log("")
    log("  COMPOUND STRESS RESULTS:")
    log("  {:<12} {:<8} {:<6} {:<10} {:>6} {:>8} {:>6}".format(
        "Dataset", "Algo", "Alpha", "DP", "K_act", "Acc%", "Jain"))
    log("  " + "-" * 65)
    for ds in datasets:
        for algo in ALGORITHMS:
            for alpha in COMPOUND_ALPHAS:
                for dp_cfg in COMPOUND_DP:
                    for active_k in COMPOUND_PARTICIPATION:
                        exp_key = "D_{}_{}_{}_{}_{}_{}".format(
                            ds, algo, alpha, dp_cfg["label"], active_k,
                            DATASET_CONFIGS[ds]["num_clients"])
                        if exp_key in ckpt.get("results", {}):
                            r = ckpt["results"][exp_key]
                            log("  {:<12} {:<8} {:<6} {:<10} {:>4}/5  {:>6.1f}%  {:>5.3f}".format(
                                ds[:12], algo, alpha, dp_cfg["label"],
                                active_k,
                                r["final_accuracy"] * 100,
                                r["jain_index"]))
        log("  " + "-" * 65)

    # Worst-case analysis
    log("")
    log("  WORST-CASE SCENARIO (alpha=0.1, DP-eps10, 3/5 participation):")
    for ds in datasets:
        for algo in ALGORITHMS:
            wc_key = "D_{}_{}_{}_{}_{}_{}".format(
                ds, algo, 0.1, "DP-eps10", 3,
                DATASET_CONFIGS[ds]["num_clients"])
            full_key = "D_{}_{}_{}_{}_{}_{}".format(
                ds, algo, 0.1, "No-DP", 5,
                DATASET_CONFIGS[ds]["num_clients"])
            if wc_key in ckpt.get("results", {}) and full_key in ckpt.get("results", {}):
                wc = ckpt["results"][wc_key]
                full = ckpt["results"][full_key]
                drop = (full["final_accuracy"] - wc["final_accuracy"]) * 100
                log("    {} {}: {:.1f}% (worst) vs {:.1f}% (best) = {:.1f}pp drop".format(
                    ds[:12], algo,
                    wc["final_accuracy"] * 100,
                    full["final_accuracy"] * 100,
                    drop))

    log("  Block D complete ({}).".format(format_time(elapsed)))


# ======================================================================
# Plot Generation
# ======================================================================

def generate_plots(ckpt):
    """Generate all PDF plots from computed results."""
    log("")
    log("=" * 70)
    log("PLOT GENERATION")
    log("=" * 70)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log("  WARNING: matplotlib not available — skipping plots")
        return

    algo_colors = {
        "FedAvg": "#2196F3", "Ditto": "#FF9800", "HPFL": "#4CAF50",
    }
    algo_markers = {"FedAvg": "o", "Ditto": "s", "HPFL": "^"}
    datasets = ["Cardiovascular", "PTB_XL", "Breast_Cancer"]

    # ---- Plot 1: Heterogeneity characterization ----
    if "block_a_correlation" in ckpt and ckpt["block_a_correlation"]:
        log("  Generating heterogeneity characterization plot...")
        corr = ckpt["block_a_correlation"]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Data Heterogeneity Impact on Algorithm Performance",
                     fontsize=14, fontweight="bold")

        for col, ds in enumerate(["Cardiovascular", "PTB_XL"]):
            ax = axes[col]
            for algo in ALGORITHMS:
                alphas_plot = []
                accs_plot = []
                kls_plot = []
                for ckey, c in corr.items():
                    if c["dataset"] == ds and c["algorithm"] == algo:
                        alphas_plot.append(c["alpha"])
                        accs_plot.append(c["mean_accuracy"] * 100)
                        kls_plot.append(c["mean_feature_kl_div"])

                if alphas_plot:
                    color = algo_colors.get(algo, "#666666")
                    marker = algo_markers.get(algo, "x")
                    ax.plot(alphas_plot, accs_plot, color=color, marker=marker,
                            markersize=8, linewidth=2, label=algo)

            ax.set_title(ds.replace("_", " "), fontsize=11)
            ax.set_xlabel("Dirichlet Alpha (lower = more heterogeneous)")
            ax.set_ylabel("Accuracy (%)")
            ax.set_xscale("log")
            ax.grid(True, alpha=0.3)
            ax.invert_xaxis()
            if col == 0:
                ax.legend(fontsize=9)

        plt.tight_layout()
        plt.savefig(str(OUTPUT_DIR / "heterogeneity_characterization.pdf"),
                    bbox_inches="tight", dpi=150)
        plt.savefig(str(OUTPUT_DIR / "heterogeneity_characterization.png"),
                    bbox_inches="tight", dpi=150)
        plt.close()
        log("    Saved heterogeneity_characterization.pdf")

    # ---- Plot 2: PEP Triangle ----
    if "block_b_pep_triangle" in ckpt:
        log("  Generating PEP triangle plot...")
        pep = ckpt["block_b_pep_triangle"]

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle("Privacy-Equity-Performance Triangle",
                     fontsize=14, fontweight="bold")

        for col, ds in enumerate(datasets):
            ax = axes[col]
            for algo in ALGORITHMS:
                rkey = "{}_{}".format(ds, algo)
                if rkey in pep:
                    pts = pep[rkey]["operating_points"]
                    # Plot accuracy vs Jain, colored by privacy
                    accs = [p["accuracy"] * 100 for p in pts if p["jain"] is not None]
                    jains = [p["jain"] for p in pts if p["jain"] is not None]
                    eps_labels = [str(p["epsilon"]) for p in pts if p["jain"] is not None]

                    if accs and jains:
                        color = algo_colors.get(algo, "#666666")
                        marker = algo_markers.get(algo, "x")
                        ax.plot(jains, accs, color=color, marker=marker,
                                markersize=8, linewidth=1.5, label=algo)

                        # Annotate epsilon values
                        for j, a, e in zip(jains, accs, eps_labels):
                            ax.annotate(e, (j, a), fontsize=6,
                                        textcoords="offset points",
                                        xytext=(3, 3))

            ax.axvline(x=0.9, color="red", linestyle="--", alpha=0.5,
                       linewidth=1, label="EHDS fair threshold" if col == 0 else None)
            ax.set_title(ds.replace("_", " "), fontsize=11)
            ax.set_xlabel("Jain Fairness Index")
            ax.set_ylabel("Accuracy (%)")
            ax.grid(True, alpha=0.3)
            if col == 0:
                ax.legend(fontsize=7, loc="lower left")

        plt.tight_layout()
        plt.savefig(str(OUTPUT_DIR / "pep_triangle.pdf"),
                    bbox_inches="tight", dpi=150)
        plt.savefig(str(OUTPUT_DIR / "pep_triangle.png"),
                    bbox_inches="tight", dpi=150)
        plt.close()
        log("    Saved pep_triangle.pdf")

    # ---- Plot 3: Data Efficiency ----
    if any(k.startswith("C_") for k in ckpt.get("results", {})):
        log("  Generating data efficiency plot...")

        # Get 100% baselines
        baselines = {}
        s10_path = OUTPUT_DIR / "checkpoint_seeds10.json"
        if s10_path.exists():
            s10 = json.load(open(s10_path))
            s10_data = s10.get("completed", {})
            for ds in datasets:
                for algo in ALGORITHMS:
                    key = "{}_{}_s42".format(ds, algo)
                    if key in s10_data:
                        pca = s10_data[key].get("per_client_acc", {})
                        if pca:
                            baselines["{}_{}".format(ds, algo)] = float(np.mean(list(pca.values())))

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle("Data Efficiency — Accuracy vs Training Data Fraction",
                     fontsize=14, fontweight="bold")

        for col, ds in enumerate(datasets):
            ax = axes[col]
            for algo in ALGORITHMS:
                fracs = []
                accs = []
                for frac in DATA_FRACTIONS:
                    exp_key = "C_{}_{}_{:.2f}".format(ds, algo, frac)
                    if exp_key in ckpt.get("results", {}):
                        fracs.append(frac * 100)
                        accs.append(ckpt["results"][exp_key]["final_accuracy"] * 100)

                # Add 100% baseline
                bkey = "{}_{}".format(ds, algo)
                if bkey in baselines:
                    fracs.append(100)
                    accs.append(baselines[bkey] * 100)

                if fracs:
                    color = algo_colors.get(algo, "#666666")
                    marker = algo_markers.get(algo, "x")
                    # Sort by fraction
                    pairs = sorted(zip(fracs, accs))
                    fracs_s = [p[0] for p in pairs]
                    accs_s = [p[1] for p in pairs]
                    ax.plot(fracs_s, accs_s, color=color, marker=marker,
                            markersize=8, linewidth=2, label=algo)

            ax.set_title(ds.replace("_", " "), fontsize=11)
            ax.set_xlabel("Training Data Fraction (%)")
            ax.set_ylabel("Accuracy (%)")
            ax.set_xticks([25, 50, 75, 100])
            ax.grid(True, alpha=0.3)
            if col == 0:
                ax.legend(fontsize=9)

        plt.tight_layout()
        plt.savefig(str(OUTPUT_DIR / "data_efficiency.pdf"),
                    bbox_inches="tight", dpi=150)
        plt.savefig(str(OUTPUT_DIR / "data_efficiency.png"),
                    bbox_inches="tight", dpi=150)
        plt.close()
        log("    Saved data_efficiency.pdf")

    # ---- Plot 4: Compound Stress ----
    if any(k.startswith("D_") for k in ckpt.get("results", {})):
        log("  Generating compound stress plot...")

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Compound Stress: Non-IID x DP x Partial Participation",
                     fontsize=14, fontweight="bold")

        for row, ds in enumerate(["Cardiovascular", "PTB_XL"]):
            nc = DATASET_CONFIGS[ds]["num_clients"]
            for col, dp_cfg in enumerate(COMPOUND_DP):
                ax = axes[row, col]

                x = np.arange(len(COMPOUND_ALPHAS))
                width = 0.12
                bar_idx = 0

                for algo in ALGORITHMS:
                    for active_k in COMPOUND_PARTICIPATION:
                        vals = []
                        for alpha in COMPOUND_ALPHAS:
                            exp_key = "D_{}_{}_{}_{}_{}_{}".format(
                                ds, algo, alpha, dp_cfg["label"], active_k, nc)
                            if exp_key in ckpt.get("results", {}):
                                vals.append(ckpt["results"][exp_key]["final_accuracy"] * 100)
                            else:
                                vals.append(0)

                        color = algo_colors.get(algo, "#666666")
                        alpha_bar = 0.9 if active_k == nc else 0.5
                        hatch = "" if active_k == nc else "///"
                        label = "{} {}/{}".format(algo, active_k, nc)

                        ax.bar(x + bar_idx * width, vals, width,
                               label=label, color=color, alpha=alpha_bar,
                               hatch=hatch, edgecolor="black", linewidth=0.5)
                        bar_idx += 1

                ax.set_title("{} — {}".format(ds.replace("_", " "), dp_cfg["label"]),
                             fontsize=10)
                ax.set_xlabel("Dirichlet Alpha")
                ax.set_ylabel("Accuracy (%)")
                ax.set_xticks(x + width * 2.5)
                ax.set_xticklabels([str(a) for a in COMPOUND_ALPHAS])
                ax.grid(True, alpha=0.3, axis="y")
                if row == 0 and col == 0:
                    ax.legend(fontsize=6, ncol=2, loc="lower right")

        plt.tight_layout()
        plt.savefig(str(OUTPUT_DIR / "compound_stress.pdf"),
                    bbox_inches="tight", dpi=150)
        plt.savefig(str(OUTPUT_DIR / "compound_stress.png"),
                    bbox_inches="tight", dpi=150)
        plt.close()
        log("    Saved compound_stress.pdf")

    log("  Plot generation complete.")


# ======================================================================
# Main
# ======================================================================

def main():
    global _log_file

    parser = argparse.ArgumentParser(
        description="FL-EHDS: Cascading Analysis & Experiments — Phase 3")
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
    log("FL-EHDS: Cascading Analysis & Experiments — Phase 3")
    log("Mode: {}".format("QUICK" if args.quick else "FULL"))
    log("Device: {}".format(_detect_device("cpu")))
    log("Blocks: A(heterogeneity) -> B(PEP triangle) -> C(data eff.) -> D(compound stress)")
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
                "experiment": "Cascading Analysis Phase 3",
                "started": datetime.now().isoformat(),
            },
            "results": {},
            "completed": [],
        }

    cascade_start = time.time()

    # Block A: Heterogeneity Characterization (analytical)
    block_a_heterogeneity(ckpt)

    if _shutdown:
        log("Shutdown after Block A")
    else:
        # Block B: PEP Triangle (analytical)
        block_b_pep_triangle(ckpt)

    if not _shutdown:
        # Block C: Data Efficiency (training)
        block_c_data_efficiency(ckpt, quick=args.quick)

    if not _shutdown:
        # Block D: Compound Stress (training)
        block_d_compound_stress(ckpt, quick=args.quick)

    if not _shutdown:
        generate_plots(ckpt)

    total_time = time.time() - cascade_start
    ckpt["metadata"]["total_time"] = format_time(total_time)
    save_checkpoint(ckpt)

    log("")
    log("=" * 70)
    log("ALL BLOCKS COMPLETE in {}".format(format_time(total_time)))
    log("Checkpoint: {}".format(CHECKPOINT_FILE))

    block_c_done = sum(1 for k in ckpt.get("completed", []) if k.startswith("C_"))
    block_d_done = sum(1 for k in ckpt.get("completed", []) if k.startswith("D_"))
    log("  Block A: {} heterogeneity profiles".format(
        len(ckpt.get("block_a_heterogeneity", {}))))
    log("  Block B: {} PEP profiles".format(
        len(ckpt.get("block_b_pep_triangle", {}))))
    log("  Block C: {}/27 data efficiency experiments".format(block_c_done))
    log("  Block D: {}/48 compound stress experiments".format(block_d_done))
    log("=" * 70)

    if _log_file:
        _log_file.close()


if __name__ == "__main__":
    main()

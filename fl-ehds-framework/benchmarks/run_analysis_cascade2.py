#!/usr/bin/env python3
"""
FL-EHDS Cascading Analysis & Experiments — Phase 2.

4 blocks executed sequentially from fastest to slowest:

Block A: Convergence Efficiency & Pareto Frontier (analytical, ~30s)
  From existing per-round history (seeds10, dp), computes rounds-to-90%,
  efficiency ratios (accuracy/MB), and Pareto frontiers of accuracy vs
  communication cost.

Block B: Per-Client Equity Deep Dive (analytical, ~30s)
  From existing per_client_acc in all checkpoints, computes Gini coefficient,
  worst-case client gap, coefficient of variation, and EHDS Art. 69
  equity scoring across all conditions.

Block C: Privacy-Utility-Fairness Trilemma (analytical, ~30s)
  From dp checkpoint (4 epsilon x 5 seeds x 3 algos x 3 datasets),
  quantifies 3-way trade-off: dAccuracy/dEpsilon, dFairness/dEpsilon,
  identifies Pareto-optimal operating points.

Block D: Learning Rate Sensitivity (training, ~7 min)
  27 new experiments: 3 LR x 3 algorithms x 3 datasets (No-DP, seed=42).
  Shows HPFL robustness to hyperparameter choices.

Usage:
    cd fl-ehds-framework
    python -m benchmarks.run_analysis_cascade2 [--quick] [--fresh]

Output:
    benchmarks/paper_results_tabular/checkpoint_cascade2.json
    benchmarks/paper_results_tabular/pareto_frontier.pdf
    benchmarks/paper_results_tabular/equity_heatmap.pdf
    benchmarks/paper_results_tabular/trilemma_tradeoff.pdf
    benchmarks/paper_results_tabular/lr_sensitivity.pdf

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
CHECKPOINT_FILE = "checkpoint_cascade2.json"
LOG_FILE = "experiment_cascade2.log"

ALGORITHMS = ["FedAvg", "Ditto", "HPFL"]

SEEDS_10 = [7, 31, 137, 577, 1337]
SEEDS_DP = [42, 123, 456, 789, 999]

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

# Learning rate sweep per dataset
LR_SWEEP = {
    "Cardiovascular": [0.001, 0.01, 0.05],
    "PTB_XL":         [0.001, 0.005, 0.02],
    "Breast_Cancer":  [0.001, 0.005, 0.02],
}

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
    fd, tmp = tempfile.mkstemp(dir=str(OUTPUT_DIR), prefix=".cas2_", suffix=".tmp")
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
    """Compute Gini coefficient (0 = perfect equality, 1 = perfect inequality)."""
    arr = np.array(sorted(values))
    n = len(arr)
    if n == 0 or np.sum(arr) == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * arr) - (n + 1) * np.sum(arr)) / (n * np.sum(arr)))


def format_time(seconds):
    return str(timedelta(seconds=int(seconds)))


_data_cache = {}


def load_dataset(dataset, num_clients, seed):
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
# Block A: Convergence Efficiency & Pareto Frontier
# ======================================================================

def _rounds_to_threshold(trajectory, threshold_frac):
    """Return round number (1-based) where accuracy first reaches threshold_frac of final value."""
    if not trajectory:
        return None
    final = trajectory[-1]
    target = final * threshold_frac
    for i, val in enumerate(trajectory):
        if val >= target:
            return i + 1
    return len(trajectory)


def _comm_cost_at_round(algo, dataset, round_num, num_clients=5):
    """Total communication cost (MB) up to a given round for an algorithm."""
    info = MODEL_PARAMS[dataset]
    bytes_per_param = 4
    if algo == "HPFL":
        comm_params = info["backbone"]
    else:
        comm_params = info["total"]
    # Per round: K uploads (full params) + K downloads (full params)
    per_round_bytes = num_clients * comm_params * bytes_per_param * 2
    return (per_round_bytes * round_num) / (1024 * 1024)


def block_a_convergence_efficiency(ckpt):
    """Block A: Convergence efficiency & Pareto frontier from existing data."""
    log("=" * 70)
    log("BLOCK A — Convergence Efficiency & Pareto Frontier")
    log("=" * 70)

    start = time.time()
    results = {}
    pareto_points = {}

    datasets = ["Breast_Cancer", "Cardiovascular", "PTB_XL"]

    # ---- Source 1: seeds10 (5 seeds, No-DP) ----
    s10_path = OUTPUT_DIR / "checkpoint_seeds10.json"
    if s10_path.exists():
        s10 = json.load(open(s10_path))
        log("  Loaded seeds10: {} experiments".format(len(s10.get("completed", {}))))
        data = s10.get("completed", {})

        for ds in datasets:
            for algo in ALGORITHMS:
                trajectories = []
                final_accs = []
                for s in SEEDS_10:
                    key = "{}_{}_s{}".format(ds, algo, s)
                    if key in data and "history" in data[key]:
                        traj = [h["accuracy"] for h in data[key]["history"]]
                        trajectories.append(traj)
                        final_accs.append(traj[-1] if traj else 0)

                if trajectories:
                    # Pad to same length
                    max_len = max(len(t) for t in trajectories)
                    padded = []
                    for t in trajectories:
                        if len(t) < max_len:
                            t = t + [t[-1]] * (max_len - len(t))
                        padded.append(t)
                    arr = np.array(padded)
                    mean_traj = np.mean(arr, axis=0).tolist()

                    # Rounds to reach 90% and 95% of final accuracy
                    r90_list = [_rounds_to_threshold(t, 0.90) for t in trajectories]
                    r95_list = [_rounds_to_threshold(t, 0.95) for t in trajectories]

                    mean_final = float(np.mean(final_accs))
                    total_mb = _comm_cost_at_round(algo, ds, max_len)
                    mb_at_r90 = _comm_cost_at_round(algo, ds,
                                                     int(np.mean(r90_list)))
                    efficiency = mean_final / total_mb if total_mb > 0 else 0

                    rkey = "seeds10_{}_{}".format(ds, algo)
                    results[rkey] = {
                        "source": "seeds10", "dataset": ds, "algorithm": algo,
                        "dp_level": "No-DP",
                        "mean_final_acc": round(mean_final, 4),
                        "std_final_acc": round(float(np.std(final_accs)), 4),
                        "n_seeds": len(trajectories),
                        "total_rounds": max_len,
                        "rounds_to_90pct_mean": round(float(np.mean(r90_list)), 1),
                        "rounds_to_90pct_std": round(float(np.std(r90_list)), 1),
                        "rounds_to_95pct_mean": round(float(np.mean(r95_list)), 1),
                        "rounds_to_95pct_std": round(float(np.std(r95_list)), 1),
                        "total_comm_MB": round(total_mb, 4),
                        "comm_MB_at_90pct": round(mb_at_r90, 4),
                        "efficiency_acc_per_MB": round(efficiency, 4),
                    }

                    # Pareto point: (cost, accuracy) for this config
                    pkey = "NoDP_{}_{}".format(ds, algo)
                    pareto_points[pkey] = {
                        "dataset": ds, "algorithm": algo, "dp_level": "No-DP",
                        "accuracy": round(mean_final, 4),
                        "comm_MB": round(total_mb, 4),
                    }

    # ---- Source 2: dp (5 seeds, per epsilon) ----
    dp_path = OUTPUT_DIR / "checkpoint_dp.json"
    if dp_path.exists():
        dp_data = json.load(open(dp_path))
        log("  Loaded dp: {} experiments".format(len(dp_data.get("completed", {}))))
        data = dp_data.get("completed", {})

        for eps in [1, 5, 10, 50]:
            for ds in datasets:
                for algo in ALGORITHMS:
                    trajectories = []
                    final_accs = []
                    for s in SEEDS_DP:
                        key = "{}_{}_eps{}_s{}".format(ds, algo, eps, s)
                        if key in data and "history" in data[key]:
                            traj = [h["accuracy"] for h in data[key]["history"]]
                            trajectories.append(traj)
                            final_accs.append(traj[-1] if traj else 0)

                    if trajectories:
                        max_len = max(len(t) for t in trajectories)
                        padded = []
                        for t in trajectories:
                            if len(t) < max_len:
                                t = t + [t[-1]] * (max_len - len(t))
                            padded.append(t)
                        arr = np.array(padded)

                        r90_list = [_rounds_to_threshold(t, 0.90) for t in trajectories]
                        r95_list = [_rounds_to_threshold(t, 0.95) for t in trajectories]

                        mean_final = float(np.mean(final_accs))
                        total_mb = _comm_cost_at_round(algo, ds, max_len)
                        mb_at_r90 = _comm_cost_at_round(algo, ds,
                                                         int(np.mean(r90_list)))
                        efficiency = mean_final / total_mb if total_mb > 0 else 0

                        rkey = "dp_eps{}_{}_{}".format(eps, ds, algo)
                        results[rkey] = {
                            "source": "dp", "dataset": ds, "algorithm": algo,
                            "dp_level": "eps={}".format(eps),
                            "mean_final_acc": round(mean_final, 4),
                            "std_final_acc": round(float(np.std(final_accs)), 4),
                            "n_seeds": len(trajectories),
                            "total_rounds": max_len,
                            "rounds_to_90pct_mean": round(float(np.mean(r90_list)), 1),
                            "rounds_to_90pct_std": round(float(np.std(r90_list)), 1),
                            "rounds_to_95pct_mean": round(float(np.mean(r95_list)), 1),
                            "rounds_to_95pct_std": round(float(np.std(r95_list)), 1),
                            "total_comm_MB": round(total_mb, 4),
                            "comm_MB_at_90pct": round(mb_at_r90, 4),
                            "efficiency_acc_per_MB": round(efficiency, 4),
                        }

                        pkey = "eps{}_{}_{}".format(eps, ds, algo)
                        pareto_points[pkey] = {
                            "dataset": ds, "algorithm": algo,
                            "dp_level": "eps={}".format(eps),
                            "accuracy": round(mean_final, 4),
                            "comm_MB": round(total_mb, 4),
                        }

    # ---- Identify Pareto-optimal configs per dataset ----
    pareto_optimal = {}
    for ds in datasets:
        ds_points = {k: v for k, v in pareto_points.items() if v["dataset"] == ds}
        items = sorted(ds_points.items(), key=lambda x: x[1]["comm_MB"])
        front = []
        max_acc = -1
        for k, v in items:
            if v["accuracy"] > max_acc:
                front.append(k)
                max_acc = v["accuracy"]
        pareto_optimal[ds] = front

    elapsed = time.time() - start

    # Print summary
    log("")
    log("  CONVERGENCE EFFICIENCY SUMMARY (seeds10, No-DP):")
    log("  {:<15} {:<8} {:>8} {:>10} {:>10} {:>12}".format(
        "Dataset", "Algo", "Acc%", "R-to-90%", "R-to-95%", "Eff(Acc/MB)"))
    log("  " + "-" * 70)
    for ds in datasets:
        for algo in ALGORITHMS:
            rkey = "seeds10_{}_{}".format(ds, algo)
            if rkey in results:
                r = results[rkey]
                log("  {:<15} {:<8} {:>6.1f}%  {:>8.1f}   {:>8.1f}   {:>10.2f}".format(
                    ds[:12], algo,
                    r["mean_final_acc"] * 100,
                    r["rounds_to_90pct_mean"],
                    r["rounds_to_95pct_mean"],
                    r["efficiency_acc_per_MB"]))
        log("  " + "-" * 70)

    log("")
    log("  PARETO-OPTIMAL CONFIGURATIONS:")
    for ds, front in pareto_optimal.items():
        log("    {}: {}".format(ds, ", ".join(front)))

    ckpt["block_a_convergence_efficiency"] = results
    ckpt["block_a_pareto_points"] = pareto_points
    ckpt["block_a_pareto_optimal"] = pareto_optimal
    ckpt["metadata"]["block_a_time"] = round(elapsed, 1)
    save_checkpoint(ckpt)
    log("  Block A complete ({:.1f}s). {} efficiency records, {} Pareto points.".format(
        elapsed, len(results), len(pareto_points)))
    return results, pareto_points, pareto_optimal


# ======================================================================
# Block B: Per-Client Equity Deep Dive
# ======================================================================

def block_b_equity(ckpt):
    """Block B: Per-client equity analysis from all checkpoints."""
    log("")
    log("=" * 70)
    log("BLOCK B — Per-Client Equity Deep Dive")
    log("=" * 70)

    start = time.time()
    results = {}

    datasets = ["Breast_Cancer", "Cardiovascular", "PTB_XL"]

    # Helper to extract per_client_acc from any checkpoint
    def _extract_equity(source_name, ckpt_data, key_pattern_fn, conditions):
        for cond in conditions:
            label, key_args_list = cond
            for ds in datasets:
                for algo in ALGORITHMS:
                    accs_per_client_all_seeds = []
                    jain_all = []
                    gini_all = []
                    worst_all = []
                    best_all = []
                    range_all = []
                    cv_all = []

                    for key_args in key_args_list:
                        key = key_pattern_fn(ds, algo, *key_args)
                        # Try "completed" first (if dict), else "results"
                        data_dict = ckpt_data.get("completed", {})
                        if not isinstance(data_dict, dict):
                            data_dict = ckpt_data.get("results", {})
                        if not isinstance(data_dict, dict):
                            continue
                        if key in data_dict:
                            entry = data_dict[key]
                            pca = entry.get("per_client_acc", {})
                            if pca:
                                vals = list(pca.values())
                                accs_per_client_all_seeds.append(vals)
                                jain_all.append(_compute_jain_index(pca))
                                gini_all.append(_compute_gini(vals))
                                worst_all.append(min(vals))
                                best_all.append(max(vals))
                                range_all.append(max(vals) - min(vals))
                                mean_v = np.mean(vals)
                                cv_all.append(
                                    float(np.std(vals) / mean_v)
                                    if mean_v > 0 else 0)

                    if accs_per_client_all_seeds:
                        rkey = "{}_{}_{}_{}".format(source_name, label, ds, algo)
                        results[rkey] = {
                            "source": source_name, "label": label,
                            "dataset": ds, "algorithm": algo,
                            "n_observations": len(accs_per_client_all_seeds),
                            "mean_jain": round(float(np.mean(jain_all)), 4),
                            "std_jain": round(float(np.std(jain_all)), 4),
                            "mean_gini": round(float(np.mean(gini_all)), 4),
                            "std_gini": round(float(np.std(gini_all)), 4),
                            "mean_worst_client": round(float(np.mean(worst_all)), 4),
                            "mean_best_client": round(float(np.mean(best_all)), 4),
                            "mean_equity_gap": round(float(np.mean(range_all)), 4),
                            "std_equity_gap": round(float(np.std(range_all)), 4),
                            "mean_cv": round(float(np.mean(cv_all)), 4),
                        }

    # ---- Source 1: seeds10 (No-DP, 5 seeds) ----
    s10_path = OUTPUT_DIR / "checkpoint_seeds10.json"
    if s10_path.exists():
        s10 = json.load(open(s10_path))
        log("  Analyzing seeds10 equity...")
        _extract_equity(
            "seeds10", s10,
            lambda ds, algo, s: "{}_{}_s{}".format(ds, algo, s),
            [("No-DP", [(s,) for s in SEEDS_10])]
        )

    # ---- Source 2: dp (4 epsilon, 5 seeds) ----
    dp_path = OUTPUT_DIR / "checkpoint_dp.json"
    if dp_path.exists():
        dp_data = json.load(open(dp_path))
        log("  Analyzing dp equity...")
        for eps in [1, 5, 10, 50]:
            _extract_equity(
                "dp", dp_data,
                lambda ds, algo, s, e=eps: "{}_{}_eps{}_s{}".format(ds, algo, e, s),
                [("eps={}".format(eps), [(s,) for s in SEEDS_DP])]
            )

    # ---- Source 3: noniid_dp ----
    noniid_path = OUTPUT_DIR / "checkpoint_noniid_dp.json"
    if noniid_path.exists():
        noniid = json.load(open(noniid_path))
        log("  Analyzing noniid_dp equity...")
        noniid_seeds = [42, 123, 456]
        for alpha in [0.1, 0.5, 1.0]:
            for dp_lbl in ["No-DP", "DP-eps1", "DP-eps10"]:
                _extract_equity(
                    "noniid", noniid,
                    lambda ds, algo, s, a=alpha, d=dp_lbl: "{}_{}_a{}_{}_s{}".format(
                        "CV" if ds == "Cardiovascular" else "PX", algo, a, d, s),
                    [("a{}_{}".format(alpha, dp_lbl), [(s,) for s in noniid_seeds])]
                )

    # ---- Source 4: cascade ablation ----
    casc_path = OUTPUT_DIR / "checkpoint_analysis_cascade.json"
    if casc_path.exists():
        casc = json.load(open(casc_path))
        log("  Analyzing cascade ablation equity...")
        for variant in ["FedAvg", "Ditto", "HPFL", "HPFL-Full"]:
            for dp_lbl in ["No-DP", "DP-eps10"]:
                for ds in datasets:
                    exp_key = "C_{}_{}_{}".format(ds, variant, dp_lbl)
                    casc_results = casc.get("results", {})
                    if exp_key in casc_results:
                        entry = casc_results[exp_key]
                        pca = entry.get("per_client_accuracy", {})
                        if pca:
                            vals = list(pca.values())
                            mean_v = np.mean(vals)
                            rkey = "ablation_{}_{}_{}_{}".format(
                                dp_lbl, ds, variant,
                                "eq")
                            results[rkey] = {
                                "source": "ablation", "label": dp_lbl,
                                "dataset": ds, "algorithm": variant,
                                "n_observations": 1,
                                "mean_jain": round(_compute_jain_index(pca), 4),
                                "std_jain": 0,
                                "mean_gini": round(_compute_gini(vals), 4),
                                "std_gini": 0,
                                "mean_worst_client": round(float(min(vals)), 4),
                                "mean_best_client": round(float(max(vals)), 4),
                                "mean_equity_gap": round(float(max(vals) - min(vals)), 4),
                                "std_equity_gap": 0,
                                "mean_cv": round(
                                    float(np.std(vals) / mean_v)
                                    if mean_v > 0 else 0, 4),
                            }

    elapsed = time.time() - start

    # Print summary
    log("")
    log("  EQUITY SUMMARY (seeds10, No-DP):")
    log("  {:<15} {:<8} {:>8} {:>8} {:>10} {:>10} {:>8}".format(
        "Dataset", "Algo", "Jain", "Gini", "Worst%", "Gap%", "CV"))
    log("  " + "-" * 75)
    for ds in datasets:
        for algo in ALGORITHMS:
            rkey = "seeds10_No-DP_{}_{}".format(ds, algo)
            if rkey in results:
                r = results[rkey]
                log("  {:<15} {:<8} {:>6.3f}  {:>6.3f}  {:>8.1f}%  {:>8.1f}%  {:>6.3f}".format(
                    ds[:12], algo,
                    r["mean_jain"], r["mean_gini"],
                    r["mean_worst_client"] * 100,
                    r["mean_equity_gap"] * 100,
                    r["mean_cv"]))
        log("  " + "-" * 75)

    # DP impact on equity
    log("")
    log("  DP IMPACT ON EQUITY (seeds10 vs dp_eps10):")
    for ds in datasets:
        for algo in ALGORITHMS:
            rkey_nodp = "seeds10_No-DP_{}_{}".format(ds, algo)
            rkey_dp = "dp_eps=10_{}_{}".format(ds, algo)
            if rkey_nodp in results and rkey_dp in results:
                r_nodp = results[rkey_nodp]
                r_dp = results[rkey_dp]
                delta_gap = r_dp["mean_equity_gap"] - r_nodp["mean_equity_gap"]
                delta_jain = r_dp["mean_jain"] - r_nodp["mean_jain"]
                log("    {} {}: Jain {:+.3f}, Gap {:+.1f}pp".format(
                    ds[:12], algo, delta_jain, delta_gap * 100))

    ckpt["block_b_equity"] = results
    ckpt["metadata"]["block_b_time"] = round(elapsed, 1)
    save_checkpoint(ckpt)
    log("  Block B complete ({:.1f}s). {} equity records.".format(
        elapsed, len(results)))
    return results


# ======================================================================
# Block C: Privacy-Utility-Fairness Trilemma
# ======================================================================

def block_c_trilemma(ckpt):
    """Block C: Privacy-Utility-Fairness 3-way trade-off quantification."""
    log("")
    log("=" * 70)
    log("BLOCK C — Privacy-Utility-Fairness Trilemma")
    log("=" * 70)

    start = time.time()
    results = {}

    datasets = ["Breast_Cancer", "Cardiovascular", "PTB_XL"]
    epsilons_ordered = [1, 5, 10, 50]

    # We need both accuracy and fairness (Jain) at each epsilon level
    # Load from dp checkpoint + equity from Block B
    dp_path = OUTPUT_DIR / "checkpoint_dp.json"
    if not dp_path.exists():
        log("  WARNING: dp checkpoint not found — skipping trilemma")
        ckpt["block_c_trilemma"] = {}
        ckpt["metadata"]["block_c_time"] = 0
        save_checkpoint(ckpt)
        return {}

    dp_data = json.load(open(dp_path))
    data = dp_data.get("completed", {})
    log("  Loaded dp: {} experiments".format(len(data)))

    # Also need No-DP baseline from seeds10
    s10_path = OUTPUT_DIR / "checkpoint_seeds10.json"
    s10_data = {}
    if s10_path.exists():
        s10 = json.load(open(s10_path))
        s10_data = s10.get("completed", {})

    for ds in datasets:
        tradeoff_curves = {}

        for algo in ALGORITHMS:
            points = []

            # No-DP baseline (epsilon = inf, from seeds10)
            accs_nodp = []
            jain_nodp = []
            for s in SEEDS_10:
                key = "{}_{}_s{}".format(ds, algo, s)
                if key in s10_data:
                    pca = s10_data[key].get("per_client_acc", {})
                    fm = s10_data[key].get("final_metrics", {})
                    if pca:
                        accs_nodp.append(float(np.mean(list(pca.values()))))
                        jain_nodp.append(_compute_jain_index(pca))
                    elif fm:
                        accs_nodp.append(fm.get("accuracy", 0))

            if accs_nodp:
                points.append({
                    "epsilon": float("inf"),
                    "epsilon_label": "No-DP",
                    "accuracy": round(float(np.mean(accs_nodp)), 4),
                    "accuracy_std": round(float(np.std(accs_nodp)), 4),
                    "jain": round(float(np.mean(jain_nodp)), 4) if jain_nodp else None,
                    "n_seeds": len(accs_nodp),
                })

            # DP levels
            for eps in epsilons_ordered:
                accs = []
                jains = []
                for s in SEEDS_DP:
                    key = "{}_{}_eps{}_s{}".format(ds, algo, eps, s)
                    if key in data:
                        pca = data[key].get("per_client_acc", {})
                        fm = data[key].get("final_metrics", {})
                        if pca:
                            accs.append(float(np.mean(list(pca.values()))))
                            jains.append(_compute_jain_index(pca))
                        elif fm:
                            accs.append(fm.get("accuracy", 0))

                if accs:
                    points.append({
                        "epsilon": eps,
                        "epsilon_label": "eps={}".format(eps),
                        "accuracy": round(float(np.mean(accs)), 4),
                        "accuracy_std": round(float(np.std(accs)), 4),
                        "jain": round(float(np.mean(jains)), 4) if jains else None,
                        "n_seeds": len(accs),
                    })

            if len(points) >= 2:
                # Compute trade-off rates (finite differences)
                # Sort by epsilon (ascending = more privacy)
                dp_points = [p for p in points if p["epsilon"] != float("inf")]
                dp_points.sort(key=lambda x: x["epsilon"])

                tradeoff_rates = []
                for i in range(1, len(dp_points)):
                    prev = dp_points[i - 1]
                    curr = dp_points[i]
                    d_eps = curr["epsilon"] - prev["epsilon"]
                    d_acc = curr["accuracy"] - prev["accuracy"]
                    rate = d_acc / d_eps if d_eps > 0 else 0

                    d_jain = None
                    if curr["jain"] is not None and prev["jain"] is not None:
                        d_jain = round((curr["jain"] - prev["jain"]) / d_eps, 6) if d_eps > 0 else 0

                    tradeoff_rates.append({
                        "from_eps": prev["epsilon"],
                        "to_eps": curr["epsilon"],
                        "d_accuracy_per_eps": round(rate, 6),
                        "d_jain_per_eps": d_jain,
                    })

                # Privacy cost of DP (accuracy drop from No-DP)
                nodp_point = next((p for p in points if p["epsilon"] == float("inf")), None)
                privacy_costs = []
                if nodp_point:
                    for p in dp_points:
                        acc_drop = nodp_point["accuracy"] - p["accuracy"]
                        jain_drop = None
                        if nodp_point["jain"] is not None and p["jain"] is not None:
                            jain_drop = round(nodp_point["jain"] - p["jain"], 4)
                        privacy_costs.append({
                            "epsilon": p["epsilon"],
                            "accuracy_drop": round(acc_drop, 4),
                            "accuracy_drop_pct": round(acc_drop * 100, 2),
                            "jain_drop": jain_drop,
                        })

                rkey = "{}_{}".format(ds, algo)
                results[rkey] = {
                    "dataset": ds, "algorithm": algo,
                    "operating_points": points,
                    "tradeoff_rates": tradeoff_rates,
                    "privacy_costs": privacy_costs,
                }

            tradeoff_curves[algo] = points

        # Find optimal operating point per dataset
        # (maximize accuracy while maintaining Jain >= 0.9)
        for algo in ALGORITHMS:
            rkey = "{}_{}".format(ds, algo)
            if rkey in results:
                pts = results[rkey]["operating_points"]
                # Filter to Jain >= 0.9 (EHDS fairness threshold)
                fair_pts = [p for p in pts
                            if p["jain"] is not None and p["jain"] >= 0.9]
                if fair_pts:
                    best = max(fair_pts, key=lambda x: x["accuracy"])
                    results[rkey]["optimal_fair_point"] = {
                        "epsilon": best["epsilon_label"],
                        "accuracy": best["accuracy"],
                        "jain": best["jain"],
                    }
                else:
                    # Best accuracy regardless of fairness
                    best = max(pts, key=lambda x: x["accuracy"])
                    results[rkey]["optimal_fair_point"] = {
                        "epsilon": best["epsilon_label"],
                        "accuracy": best["accuracy"],
                        "jain": best.get("jain"),
                        "note": "No point meets Jain>=0.9 threshold",
                    }

    elapsed = time.time() - start

    # Print summary
    log("")
    log("  TRILEMMA SUMMARY:")
    log("  {:<15} {:<8} {:>8} {:>8} {:>12} {:>15}".format(
        "Dataset", "Algo", "NoDP%", "eps10%", "AccDrop(e10)", "Optimal"))
    log("  " + "-" * 75)
    for ds in datasets:
        for algo in ALGORITHMS:
            rkey = "{}_{}".format(ds, algo)
            if rkey in results:
                r = results[rkey]
                nodp_pt = next((p for p in r["operating_points"]
                                if p["epsilon"] == float("inf")), None)
                dp10_pt = next((p for p in r["operating_points"]
                                if p["epsilon"] == 10), None)
                nodp_acc = nodp_pt["accuracy"] * 100 if nodp_pt else 0
                dp10_acc = dp10_pt["accuracy"] * 100 if dp10_pt else 0
                drop = nodp_acc - dp10_acc

                opt = r.get("optimal_fair_point", {})
                opt_str = "{} ({:.1f}%)".format(
                    opt.get("epsilon", "?"),
                    opt.get("accuracy", 0) * 100)

                log("  {:<15} {:<8} {:>6.1f}%  {:>6.1f}%  {:>+10.1f}pp  {:>15}".format(
                    ds[:12], algo, nodp_acc, dp10_acc, -drop, opt_str))
        log("  " + "-" * 75)

    # Trade-off rates
    log("")
    log("  MARGINAL TRADE-OFF RATES (dAccuracy / dEpsilon):")
    for ds in datasets:
        log("    {}:".format(ds))
        for algo in ALGORITHMS:
            rkey = "{}_{}".format(ds, algo)
            if rkey in results and results[rkey]["tradeoff_rates"]:
                rates = results[rkey]["tradeoff_rates"]
                rate_strs = ["eps {}->{}: {:+.4f}/eps".format(
                    r["from_eps"], r["to_eps"], r["d_accuracy_per_eps"])
                    for r in rates]
                log("      {}: {}".format(algo, " | ".join(rate_strs)))

    ckpt["block_c_trilemma"] = results
    ckpt["metadata"]["block_c_time"] = round(elapsed, 1)
    save_checkpoint(ckpt)
    log("  Block C complete ({:.1f}s). {} trade-off curves.".format(
        elapsed, len(results)))
    return results


# ======================================================================
# Block D: Learning Rate Sensitivity (TRAINING)
# ======================================================================

def run_lr_experiment(dataset, algo, lr, seed, quick=False):
    """Run one LR sensitivity experiment."""
    start = time.time()
    dcfg = DATASET_CONFIGS[dataset]
    num_rounds = 5 if quick else dcfg["num_rounds"]

    client_data, client_test, meta = load_dataset(
        dataset, dcfg["num_clients"], seed)

    trainer = FederatedTrainer(
        num_clients=dcfg["num_clients"], algorithm=algo,
        local_epochs=dcfg["local_epochs"], batch_size=dcfg["batch_size"],
        learning_rate=lr, mu=dcfg["mu"], seed=seed,
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

        history.append({
            "round": r + 1,
            "accuracy": round_acc,
            "loss": result.global_loss,
        })
        if round_acc > best_acc:
            best_acc = round_acc

    # Final evaluation
    per_client_acc = _evaluate_per_client(trainer)
    jain = _compute_jain_index(per_client_acc)
    final_acc = float(np.mean(list(per_client_acc.values())))

    elapsed = time.time() - start

    return {
        "block": "D_lr_sensitivity",
        "dataset": dataset,
        "algorithm": algo,
        "learning_rate": lr,
        "seed": seed,
        "num_rounds": num_rounds,
        "final_accuracy": round(final_acc, 4),
        "best_accuracy": round(best_acc, 4),
        "per_client_accuracy": per_client_acc,
        "jain_index": round(jain, 4),
        "convergence_trajectory": [h["accuracy"] for h in history],
        "history": history,
        "time_seconds": round(elapsed, 1),
        "is_base_lr": lr == dcfg["learning_rate"],
    }


def block_d_lr_sensitivity(ckpt, quick=False):
    """Block D: Learning rate sensitivity analysis (training)."""
    log("")
    log("=" * 70)
    log("BLOCK D — Learning Rate Sensitivity")
    log("=" * 70)

    start = time.time()
    datasets = ["Cardiovascular", "PTB_XL", "Breast_Cancer"]
    seed = 42

    experiments = []
    for ds in datasets:
        lrs = LR_SWEEP[ds]
        for algo in ALGORITHMS:
            for lr in lrs:
                exp_key = "D_{}_{}_{:.4f}".format(ds, algo, lr)
                experiments.append((exp_key, ds, algo, lr))

    total = len(experiments)
    done = sum(1 for e in experiments if e[0] in ckpt.get("completed", []))
    log("  {} experiments ({} already done)".format(total, done))

    for idx, (exp_key, ds, algo, lr) in enumerate(experiments, 1):
        if _shutdown:
            log("  Shutdown requested — saving checkpoint")
            break

        if exp_key in ckpt.get("completed", []):
            continue

        try:
            base_lr = DATASET_CONFIGS[ds]["learning_rate"]
            lr_ratio = lr / base_lr
            log("  [{}/{}] {} | {} | lr={:.4f} ({:.1f}x base) ...".format(
                idx, total, ds, algo, lr, lr_ratio))

            result = run_lr_experiment(ds, algo, lr, seed, quick=quick)

            ckpt["results"][exp_key] = result
            if "completed" not in ckpt:
                ckpt["completed"] = []
            ckpt["completed"].append(exp_key)
            save_checkpoint(ckpt)

            log("    -> acc={:.1f}% | best={:.1f}% | Jain={:.3f} | {:.0f}s".format(
                result["final_accuracy"] * 100,
                result["best_accuracy"] * 100,
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

    # Print sensitivity summary
    log("")
    log("  LR SENSITIVITY RESULTS:")
    log("  {:<15} {:<8} {:>10} {:>10} {:>10} {:>8}".format(
        "Dataset", "Algo", "LR", "Acc%", "Best%", "Jain"))
    log("  " + "-" * 65)
    for ds in datasets:
        lrs = LR_SWEEP[ds]
        for algo in ALGORITHMS:
            for lr in lrs:
                exp_key = "D_{}_{}_{:.4f}".format(ds, algo, lr)
                if exp_key in ckpt.get("results", {}):
                    r = ckpt["results"][exp_key]
                    base_marker = " *" if r.get("is_base_lr") else ""
                    log("  {:<15} {:<8} {:>10.4f} {:>8.1f}%  {:>8.1f}%  {:>6.3f}{}".format(
                        ds[:12], algo, lr,
                        r["final_accuracy"] * 100,
                        r["best_accuracy"] * 100,
                        r["jain_index"],
                        base_marker))
        log("  " + "-" * 65)
    log("  (* = base learning rate)")

    # Compute robustness scores
    log("")
    log("  ROBUSTNESS SCORES (std of accuracy across LRs):")
    for ds in datasets:
        log("    {}:".format(ds))
        for algo in ALGORITHMS:
            accs = []
            for lr in LR_SWEEP[ds]:
                exp_key = "D_{}_{}_{:.4f}".format(ds, algo, lr)
                if exp_key in ckpt.get("results", {}):
                    accs.append(ckpt["results"][exp_key]["final_accuracy"])
            if accs:
                robustness = 1.0 - float(np.std(accs))  # higher = more robust
                log("      {}: std={:.4f}, robustness={:.4f}".format(
                    algo, float(np.std(accs)), robustness))

    log("  Block D complete ({}).".format(format_time(elapsed)))
    return


# ======================================================================
# Plot Generation (after all blocks)
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
        from matplotlib.backends.backend_pdf import PdfPages
    except ImportError:
        log("  WARNING: matplotlib not available — skipping plots")
        return

    algo_colors = {
        "FedAvg": "#2196F3", "Ditto": "#FF9800", "HPFL": "#4CAF50",
        "HPFL-Full": "#E91E63",
    }
    algo_markers = {
        "FedAvg": "o", "Ditto": "s", "HPFL": "^", "HPFL-Full": "D",
    }
    datasets = ["Cardiovascular", "PTB_XL", "Breast_Cancer"]

    # ---- Plot 1: Pareto Frontier ----
    if "block_a_pareto_points" in ckpt:
        log("  Generating Pareto frontier plot...")
        pareto_pts = ckpt["block_a_pareto_points"]
        pareto_opt = ckpt.get("block_a_pareto_optimal", {})

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle("Accuracy vs Communication Cost — Pareto Frontier",
                     fontsize=14, fontweight="bold")

        for col, ds in enumerate(datasets):
            ax = axes[col]
            ds_pts = {k: v for k, v in pareto_pts.items() if v["dataset"] == ds}

            for k, v in ds_pts.items():
                algo = v["algorithm"]
                dp = v["dp_level"]
                color = algo_colors.get(algo, "#666666")
                marker = algo_markers.get(algo, "x")

                is_pareto = k in pareto_opt.get(ds, [])
                edgecolor = "red" if is_pareto else color
                size = 120 if is_pareto else 60
                zorder = 10 if is_pareto else 5

                ax.scatter(v["comm_MB"], v["accuracy"] * 100,
                          c=color, marker=marker, s=size,
                          edgecolors=edgecolor, linewidths=2 if is_pareto else 0.5,
                          zorder=zorder, label=None)

            # Legend for algorithms (one entry each)
            for algo in ALGORITHMS:
                color = algo_colors.get(algo, "#666666")
                marker = algo_markers.get(algo, "x")
                ax.scatter([], [], c=color, marker=marker, s=60, label=algo)

            # Draw Pareto frontier line
            front_keys = pareto_opt.get(ds, [])
            if front_keys:
                front_pts = [(pareto_pts[k]["comm_MB"], pareto_pts[k]["accuracy"] * 100)
                             for k in front_keys if k in pareto_pts]
                front_pts.sort(key=lambda x: x[0])
                if len(front_pts) > 1:
                    ax.plot([p[0] for p in front_pts], [p[1] for p in front_pts],
                            "r--", linewidth=1.5, alpha=0.6, label="Pareto front")

            ax.set_title(ds.replace("_", " "), fontsize=11)
            ax.set_xlabel("Communication Cost (MB)")
            ax.set_ylabel("Accuracy (%)")
            ax.grid(True, alpha=0.3)
            if col == 0:
                ax.legend(fontsize=8, loc="lower right")

        plt.tight_layout()
        plt.savefig(str(OUTPUT_DIR / "pareto_frontier.pdf"),
                    bbox_inches="tight", dpi=150)
        plt.savefig(str(OUTPUT_DIR / "pareto_frontier.png"),
                    bbox_inches="tight", dpi=150)
        plt.close()
        log("    Saved pareto_frontier.pdf")

    # ---- Plot 2: Equity Heatmap ----
    if "block_b_equity" in ckpt:
        log("  Generating equity heatmap...")
        equity = ckpt["block_b_equity"]

        # Build heatmap: rows = conditions, cols = datasets x algos
        conditions = ["seeds10_No-DP", "dp_eps=1", "dp_eps=5",
                       "dp_eps=10", "dp_eps=50"]

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle("Per-Client Equity Across Conditions",
                     fontsize=14, fontweight="bold")

        for plot_idx, (metric, metric_label) in enumerate([
            ("mean_gini", "Gini Coefficient (lower=fairer)"),
            ("mean_equity_gap", "Equity Gap (max-min client, lower=fairer)"),
        ]):
            ax = axes[plot_idx]
            col_labels = []
            for ds in datasets:
                for algo in ALGORITHMS:
                    col_labels.append("{}\n{}".format(ds[:4], algo))

            data_matrix = np.full((len(conditions), len(col_labels)), np.nan)

            for i, cond in enumerate(conditions):
                for j, (ds, algo) in enumerate(
                    [(ds, a) for ds in datasets for a in ALGORITHMS]
                ):
                    rkey = "{}_{}_{}".format(cond, ds, algo)
                    if rkey in equity:
                        data_matrix[i, j] = equity[rkey].get(metric, np.nan)

            cmap = "YlOrRd" if "gini" in metric else "YlOrRd"
            im = ax.imshow(data_matrix, cmap=cmap, aspect="auto")

            ax.set_xticks(range(len(col_labels)))
            ax.set_xticklabels(col_labels, fontsize=7, rotation=45, ha="right")
            ax.set_yticks(range(len(conditions)))
            ax.set_yticklabels([c.replace("seeds10_", "").replace("dp_", "")
                                for c in conditions], fontsize=9)

            # Annotate
            for i in range(len(conditions)):
                for j in range(len(col_labels)):
                    val = data_matrix[i, j]
                    if not np.isnan(val):
                        ax.text(j, i, "{:.3f}".format(val), ha="center",
                                va="center", fontsize=6,
                                color="white" if val > 0.15 else "black")

            plt.colorbar(im, ax=ax, shrink=0.8)
            ax.set_title(metric_label, fontsize=10)

        plt.tight_layout()
        plt.savefig(str(OUTPUT_DIR / "equity_heatmap.pdf"),
                    bbox_inches="tight", dpi=150)
        plt.savefig(str(OUTPUT_DIR / "equity_heatmap.png"),
                    bbox_inches="tight", dpi=150)
        plt.close()
        log("    Saved equity_heatmap.pdf")

    # ---- Plot 3: Trilemma Trade-off ----
    if "block_c_trilemma" in ckpt:
        log("  Generating trilemma trade-off plot...")
        trilemma = ckpt["block_c_trilemma"]

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle("Privacy-Utility Trade-off (Accuracy vs Epsilon)",
                     fontsize=14, fontweight="bold")

        for col, ds in enumerate(datasets):
            ax = axes[col]
            for algo in ALGORITHMS:
                rkey = "{}_{}".format(ds, algo)
                if rkey in trilemma:
                    pts = trilemma[rkey]["operating_points"]
                    # Sort by epsilon (exclude inf for line plot)
                    dp_pts = sorted(
                        [p for p in pts if p["epsilon"] != float("inf")],
                        key=lambda x: x["epsilon"])

                    if dp_pts:
                        eps_vals = [p["epsilon"] for p in dp_pts]
                        acc_vals = [p["accuracy"] * 100 for p in dp_pts]
                        std_vals = [p.get("accuracy_std", 0) * 100 for p in dp_pts]

                        color = algo_colors.get(algo, "#666666")
                        marker = algo_markers.get(algo, "x")

                        ax.errorbar(eps_vals, acc_vals, yerr=std_vals,
                                    color=color, marker=marker, markersize=6,
                                    linewidth=1.5, capsize=3, label=algo)

                    # No-DP baseline as horizontal dashed line
                    nodp_pt = next((p for p in pts if p["epsilon"] == float("inf")), None)
                    if nodp_pt:
                        color = algo_colors.get(algo, "#666666")
                        ax.axhline(y=nodp_pt["accuracy"] * 100, color=color,
                                   linestyle=":", alpha=0.4, linewidth=1)

            ax.set_title(ds.replace("_", " "), fontsize=11)
            ax.set_xlabel("Privacy Budget (epsilon)")
            ax.set_ylabel("Accuracy (%)")
            ax.set_xscale("log")
            ax.grid(True, alpha=0.3)
            if col == 0:
                ax.legend(fontsize=8)

        plt.tight_layout()
        plt.savefig(str(OUTPUT_DIR / "trilemma_tradeoff.pdf"),
                    bbox_inches="tight", dpi=150)
        plt.savefig(str(OUTPUT_DIR / "trilemma_tradeoff.png"),
                    bbox_inches="tight", dpi=150)
        plt.close()
        log("    Saved trilemma_tradeoff.pdf")

        # Additional: Fairness dimension
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle("Privacy-Fairness Trade-off (Jain Index vs Epsilon)",
                     fontsize=14, fontweight="bold")

        for col, ds in enumerate(datasets):
            ax = axes[col]
            for algo in ALGORITHMS:
                rkey = "{}_{}".format(ds, algo)
                if rkey in trilemma:
                    pts = trilemma[rkey]["operating_points"]
                    dp_pts = sorted(
                        [p for p in pts
                         if p["epsilon"] != float("inf") and p["jain"] is not None],
                        key=lambda x: x["epsilon"])

                    if dp_pts:
                        eps_vals = [p["epsilon"] for p in dp_pts]
                        jain_vals = [p["jain"] for p in dp_pts]

                        color = algo_colors.get(algo, "#666666")
                        marker = algo_markers.get(algo, "x")

                        ax.plot(eps_vals, jain_vals,
                                color=color, marker=marker, markersize=6,
                                linewidth=1.5, label=algo)

            ax.axhline(y=0.9, color="red", linestyle="--", alpha=0.5,
                       linewidth=1, label="EHDS threshold")
            ax.set_title(ds.replace("_", " "), fontsize=11)
            ax.set_xlabel("Privacy Budget (epsilon)")
            ax.set_ylabel("Jain Fairness Index")
            ax.set_xscale("log")
            ax.set_ylim(0.7, 1.05)
            ax.grid(True, alpha=0.3)
            if col == 0:
                ax.legend(fontsize=8)

        plt.tight_layout()
        plt.savefig(str(OUTPUT_DIR / "trilemma_fairness.pdf"),
                    bbox_inches="tight", dpi=150)
        plt.savefig(str(OUTPUT_DIR / "trilemma_fairness.png"),
                    bbox_inches="tight", dpi=150)
        plt.close()
        log("    Saved trilemma_fairness.pdf")

    # ---- Plot 4: LR Sensitivity ----
    if any(k.startswith("D_") for k in ckpt.get("results", {})):
        log("  Generating LR sensitivity plot...")

        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        fig.suptitle("Learning Rate Sensitivity — Accuracy & Convergence",
                     fontsize=14, fontweight="bold")

        for col, ds in enumerate(datasets):
            lrs = LR_SWEEP[ds]
            base_lr = DATASET_CONFIGS[ds]["learning_rate"]

            # Row 0: Final accuracy bar chart
            ax = axes[0, col]
            x = np.arange(len(lrs))
            width = 0.25

            for i, algo in enumerate(ALGORITHMS):
                vals = []
                for lr in lrs:
                    exp_key = "D_{}_{}_{:.4f}".format(ds, algo, lr)
                    if exp_key in ckpt.get("results", {}):
                        vals.append(ckpt["results"][exp_key]["final_accuracy"] * 100)
                    else:
                        vals.append(0)

                color = algo_colors.get(algo, "#666666")
                bars = ax.bar(x + i * width, vals, width,
                              label=algo, color=color, alpha=0.8)

                # Mark base LR
                base_idx = lrs.index(base_lr) if base_lr in lrs else -1
                if base_idx >= 0:
                    ax.bar(x[base_idx] + i * width, vals[base_idx], width,
                           color=color, edgecolor="black", linewidth=2)

            ax.set_title(ds.replace("_", " "), fontsize=11)
            ax.set_xlabel("Learning Rate")
            ax.set_ylabel("Accuracy (%)")
            ax.set_xticks(x + width)
            ax.set_xticklabels(["{:.4f}".format(lr) for lr in lrs], fontsize=8)
            ax.grid(True, alpha=0.3, axis="y")
            if col == 0:
                ax.legend(fontsize=8)

            # Row 1: Convergence curves for each LR
            ax = axes[1, col]
            for algo in ALGORITHMS:
                color = algo_colors.get(algo, "#666666")
                for lr_idx, lr in enumerate(lrs):
                    exp_key = "D_{}_{}_{:.4f}".format(ds, algo, lr)
                    if exp_key in ckpt.get("results", {}):
                        traj = ckpt["results"][exp_key]["convergence_trajectory"]
                        rounds = list(range(1, len(traj) + 1))
                        linestyle = ["-", "--", ":"][lr_idx % 3]
                        alpha = 1.0 if lr == base_lr else 0.5
                        label = "{} lr={}".format(algo, lr) if col == 0 else None
                        ax.plot(rounds, [a * 100 for a in traj],
                                color=color, linestyle=linestyle,
                                linewidth=1.5, alpha=alpha, label=label)

            ax.set_xlabel("Round")
            ax.set_ylabel("Accuracy (%)")
            ax.grid(True, alpha=0.3)
            if col == 0:
                ax.legend(fontsize=6, ncol=3, loc="lower right")

        plt.tight_layout()
        plt.savefig(str(OUTPUT_DIR / "lr_sensitivity.pdf"),
                    bbox_inches="tight", dpi=150)
        plt.savefig(str(OUTPUT_DIR / "lr_sensitivity.png"),
                    bbox_inches="tight", dpi=150)
        plt.close()
        log("    Saved lr_sensitivity.pdf")

    log("  Plot generation complete.")


# ======================================================================
# Main
# ======================================================================

def main():
    global _log_file

    parser = argparse.ArgumentParser(
        description="FL-EHDS: Cascading Analysis & Experiments — Phase 2")
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
    log("FL-EHDS: Cascading Analysis & Experiments — Phase 2")
    log("Mode: {}".format("QUICK" if args.quick else "FULL"))
    log("Device: {}".format(_detect_device("cpu")))
    log("Blocks: A(efficiency) -> B(equity) -> C(trilemma) -> D(LR sensitivity)")
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
                "experiment": "Cascading Analysis Phase 2",
                "started": datetime.now().isoformat(),
            },
            "results": {},
            "completed": [],
        }

    cascade_start = time.time()

    # Block A: Convergence Efficiency (analytical)
    block_a_convergence_efficiency(ckpt)

    if _shutdown:
        log("Shutdown after Block A")
    else:
        # Block B: Per-Client Equity (analytical)
        block_b_equity(ckpt)

    if not _shutdown:
        # Block C: Privacy-Utility-Fairness Trilemma (analytical)
        block_c_trilemma(ckpt)

    if not _shutdown:
        # Block D: LR Sensitivity (training)
        block_d_lr_sensitivity(ckpt, quick=args.quick)

    if not _shutdown:
        # Generate all plots
        generate_plots(ckpt)

    total_time = time.time() - cascade_start
    ckpt["metadata"]["total_time"] = format_time(total_time)
    save_checkpoint(ckpt)

    log("")
    log("=" * 70)
    log("ALL BLOCKS COMPLETE in {}".format(format_time(total_time)))
    log("Checkpoint: {}".format(CHECKPOINT_FILE))

    # Final summary
    block_d_done = sum(1 for k in ckpt.get("completed", []) if k.startswith("D_"))
    log("  Block A: {} efficiency records".format(
        len(ckpt.get("block_a_convergence_efficiency", {}))))
    log("  Block B: {} equity records".format(
        len(ckpt.get("block_b_equity", {}))))
    log("  Block C: {} trilemma curves".format(
        len(ckpt.get("block_c_trilemma", {}))))
    log("  Block D: {}/27 LR experiments".format(block_d_done))
    log("=" * 70)

    if _log_file:
        _log_file.close()


if __name__ == "__main__":
    main()

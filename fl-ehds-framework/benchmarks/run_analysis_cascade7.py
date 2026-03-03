#!/usr/bin/env python3
"""
FL-EHDS Cascading Analysis & Experiments — Phase 7.

6 blocks executed sequentially (Trustworthiness & EHDS Advanced):

Block M: Model Calibration (ECE) under DP (~54 exp)
  3 datasets × 3 algorithms × 3 DP levels × 2 seeds.
  Measures Expected Calibration Error, Temperature Scaling, reliability.

Block N: Conformal Prediction under FL (~36 exp)
  3 datasets × 2 conditions (IID/NonIID) × 3 DP levels × 2 seeds.
  Coverage guarantees, prediction set sizes, exchangeability under DP.

Block O: Feature Attribution Stability (permutation importance) (~36 exp)
  3 datasets × 3 algorithms × 2 conditions × 2 seeds.
  Feature importance consistency across clients, stability under DP.

Block P: Demographic Fairness (~36 exp)
  Cardiovascular × 3 algorithms × 3 DP levels × 2 conditions × 2 seeds.
  Equalized Odds, Demographic Parity by sex and age group.

Block Q: Concept Drift Robustness (~36 exp)
  Cardiovascular × 3 algorithms × 2 drift severities × 3 adaptation × 2 seeds.
  Temporal drift simulation, accuracy degradation, recovery strategies.

Block R: DP Composition Across Studies (~18 exp)
  3 datasets × 3 per-study budgets × 2 composition methods.
  Multi-study privacy budget tracking for EHDS HDAB governance.

Usage:
    cd fl-ehds-framework
    python -m benchmarks.run_analysis_cascade7 [--quick] [--fresh]

Output:
    benchmarks/paper_results_tabular/checkpoint_cascade7.json

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
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from copy import deepcopy

import numpy as np
from scipy import stats as scipy_stats

FRAMEWORK_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(FRAMEWORK_DIR))

import torch
import torch.nn as nn
import torch.optim as optim

# Data loaders
from data.ptbxl_loader import load_ptbxl_data
from data.cardiovascular_loader import load_cardiovascular_data
from data.breast_cancer_loader import load_breast_cancer_data

# ======================================================================
# Constants
# ======================================================================

OUTPUT_DIR = FRAMEWORK_DIR / "benchmarks" / "paper_results_tabular"
CHECKPOINT_FILE = "checkpoint_cascade7.json"
LOG_FILE = "experiment_cascade7.log"

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

DATASETS = ["Cardiovascular", "PTB_XL", "Breast_Cancer"]
DEFAULT_SEEDS = [42, 123]
DEVICE = torch.device("cpu")

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
    fd, tmp = tempfile.mkstemp(dir=str(OUTPUT_DIR), prefix=".cas7_", suffix=".tmp")
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
            num_clients=num_clients, seed=seed, is_iid=is_iid, alpha=alpha,
        )
    else:
        raise ValueError("Unknown dataset: {}".format(dataset))

    _data_cache[cache_key] = (client_data, client_test, meta)
    return client_data, client_test, meta


# ======================================================================
# Model utilities
# ======================================================================

class HealthcareMLP(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


def create_model(input_dim, num_classes, seed=42):
    torch.manual_seed(seed)
    return HealthcareMLP(input_dim, num_classes).to(DEVICE)


def get_params(model):
    return {n: p.data.clone() for n, p in model.named_parameters()}


def set_params(model, params):
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in params:
                p.data.copy_(params[n])


def train_local_sgd(model, X, y, epochs, lr, batch_size=64):
    model.train()
    X_t = torch.FloatTensor(X).to(DEVICE) if isinstance(X, np.ndarray) else X.to(DEVICE)
    y_t = torch.LongTensor(y.astype(int)).to(DEVICE) if isinstance(y, np.ndarray) else y.to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    for _ in range(epochs):
        perm = torch.randperm(len(y_t))
        for i in range(0, len(y_t), batch_size):
            idx = perm[i:i + batch_size]
            optimizer.zero_grad()
            out = model(X_t[idx])
            loss = criterion(out, y_t[idx])
            loss.backward()
            optimizer.step()


def evaluate_model(model, X, y):
    model.eval()
    X_t = torch.FloatTensor(X).to(DEVICE) if isinstance(X, np.ndarray) else X.to(DEVICE)
    y_t = torch.LongTensor(y.astype(int)).to(DEVICE) if isinstance(y, np.ndarray) else y.to(DEVICE)
    with torch.no_grad():
        out = model(X_t)
        preds = out.argmax(dim=1)
        acc = (preds == y_t).float().mean().item()
    return acc


def get_logits(model, X):
    """Get raw logits from model."""
    model.eval()
    X_t = torch.FloatTensor(X).to(DEVICE) if isinstance(X, np.ndarray) else X.to(DEVICE)
    with torch.no_grad():
        return model(X_t).cpu().numpy()


def get_probabilities(model, X):
    """Get softmax probabilities from model."""
    logits = get_logits(model, X)
    exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)


# ======================================================================
# FL Training (FedAvg, Ditto, HPFL) with model return
# ======================================================================

def federated_round(model, client_data, all_clients, cfg, dp_epsilon=None):
    global_params = get_params(model)
    updates, samples = [], []
    for cid in all_clients:
        X, y = client_data[cid]
        lm = create_model(cfg["input_dim"], cfg["num_classes"])
        set_params(lm, global_params)
        train_local_sgd(lm, X, y, cfg["local_epochs"], cfg["learning_rate"], cfg["batch_size"])
        lp = get_params(lm)
        delta = {n: lp[n] - global_params[n] for n in global_params}
        updates.append(delta)
        samples.append(len(y))
    total_s = sum(samples)
    avg_d = {}
    for n in global_params:
        avg_d[n] = sum(updates[i][n] * (samples[i] / total_s) for i in range(len(updates)))
    if dp_epsilon is not None and dp_epsilon < float("inf"):
        noise_scale = 1.0 / dp_epsilon
        for n in avg_d:
            avg_d[n] += torch.randn_like(avg_d[n]) * noise_scale
    new_p = {n: global_params[n] + avg_d[n] for n in global_params}
    set_params(model, new_p)
    return model


def train_fl_model(dataset, cfg, algo, num_rounds, seed, is_iid=True, alpha=0.5,
                   dp_epsilon=None):
    """Train FL model and return (global_model, client_data, client_test, meta)."""
    num_clients = cfg["num_clients"]
    client_data, client_test, meta = load_dataset(dataset, num_clients, seed, is_iid, alpha)
    model = create_model(cfg["input_dim"], cfg["num_classes"], seed=seed)
    all_clients = list(client_data.keys())

    # For Ditto: train global then fine-tune personal models
    for r in range(num_rounds):
        model = federated_round(model, client_data, all_clients, cfg, dp_epsilon=dp_epsilon)

    if algo == "Ditto":
        # Fine-tune personal models
        personal = {}
        for cid in all_clients:
            pm = create_model(cfg["input_dim"], cfg["num_classes"])
            set_params(pm, get_params(model))
            X, y = client_data[cid]
            train_local_sgd(pm, X, y, cfg["local_epochs"], cfg["learning_rate"], cfg["batch_size"])
            personal[cid] = pm
        return model, personal, client_data, client_test, meta
    elif algo == "HPFL":
        # Personalized heads: retrain last layer per client
        personal = {}
        for cid in all_clients:
            pm = create_model(cfg["input_dim"], cfg["num_classes"])
            set_params(pm, get_params(model))
            X, y = client_data[cid]
            # Only fine-tune fc2 (head)
            for n, p in pm.named_parameters():
                p.requires_grad = ("fc2" in n)
            opt = optim.SGD(filter(lambda p: p.requires_grad, pm.parameters()), lr=cfg["learning_rate"])
            crit = nn.CrossEntropyLoss()
            X_t = torch.FloatTensor(X).to(DEVICE)
            y_t = torch.LongTensor(y.astype(int)).to(DEVICE)
            pm.train()
            for _ in range(cfg["local_epochs"] * 2):
                opt.zero_grad()
                loss = crit(pm(X_t), y_t)
                loss.backward()
                opt.step()
            for n, p in pm.named_parameters():
                p.requires_grad = True
            personal[cid] = pm
        return model, personal, client_data, client_test, meta

    return model, None, client_data, client_test, meta


# ======================================================================
# Block M: Model Calibration (ECE) under DP
# ======================================================================

def expected_calibration_error(probs, labels, n_bins=10):
    """Compute ECE (Expected Calibration Error)."""
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_details = []
    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if mask.sum() > 0:
            bin_acc = accuracies[mask].mean()
            bin_conf = confidences[mask].mean()
            bin_size = mask.mean()
            ece += bin_size * abs(bin_acc - bin_conf)
            bin_details.append({
                "bin": i, "acc": round(float(bin_acc), 4),
                "conf": round(float(bin_conf), 4), "size": round(float(bin_size), 4),
            })
    return round(float(ece), 6), bin_details


def temperature_scaling(logits_val, labels_val, logits_test):
    """Apply temperature scaling. Returns calibrated probabilities."""
    # Find optimal temperature via grid search
    best_T, best_ece = 1.0, float("inf")
    for T in np.arange(0.1, 5.1, 0.1):
        scaled = logits_val / T
        exp_l = np.exp(scaled - scaled.max(axis=1, keepdims=True))
        probs = exp_l / exp_l.sum(axis=1, keepdims=True)
        ece, _ = expected_calibration_error(probs, labels_val)
        if ece < best_ece:
            best_ece = ece
            best_T = T
    # Apply best T to test
    scaled_test = logits_test / best_T
    exp_l = np.exp(scaled_test - scaled_test.max(axis=1, keepdims=True))
    return exp_l / exp_l.sum(axis=1, keepdims=True), round(float(best_T), 2)


def run_block_m(checkpoint, quick=False):
    """Block M: Model Calibration under DP."""
    log("=" * 70)
    log("BLOCK M: Model Calibration (ECE) under DP")
    log("=" * 70)

    algorithms = ["FedAvg", "Ditto", "HPFL"]
    dp_levels = [None, 10.0, 1.0]  # no DP, eps=10, eps=1
    dp_names = ["noDP", "eps10", "eps1"]
    num_rounds = 10 if quick else 30
    seeds = [42] if quick else DEFAULT_SEEDS
    t0 = time.time()
    count = 0
    total = len(DATASETS) * len(algorithms) * len(dp_levels) * len(seeds)

    for ds in DATASETS:
        cfg = DATASET_CONFIGS[ds]
        for algo in algorithms:
            for dp_eps, dp_name in zip(dp_levels, dp_names):
                for seed in seeds:
                    if _shutdown:
                        return count
                    key = "M_{}_{}_{}_s{}".format(ds, algo, dp_name, seed)
                    if key in checkpoint["results"]:
                        count += 1
                        continue
                    count += 1
                    log("  [{}/{}] {} / {} / {} / seed={}".format(
                        count, total, ds, algo, dp_name, seed))
                    try:
                        t_exp = time.time()
                        model, personal, client_data, client_test, meta = train_fl_model(
                            ds, cfg, algo, num_rounds, seed, dp_epsilon=dp_eps)

                        # Evaluate using personal models when available (Ditto/HPFL)
                        if personal:
                            logits_parts, probs_parts = [], []
                            all_X_parts, all_y_parts = [], []
                            for cid in client_test:
                                Xc, yc = client_test[cid]
                                cm = personal[cid] if cid in personal else model
                                all_X_parts.append(Xc)
                                all_y_parts.append(yc)
                                logits_parts.append(get_logits(cm, Xc))
                                probs_parts.append(get_probabilities(cm, Xc))
                            all_X = np.concatenate(all_X_parts)
                            all_y = np.concatenate(all_y_parts)
                            logits = np.concatenate(logits_parts)
                            probs = np.concatenate(probs_parts)
                            acc = float((np.argmax(probs, axis=1) == all_y).mean())
                        else:
                            all_X = np.concatenate([client_test[c][0] for c in client_test])
                            all_y = np.concatenate([client_test[c][1] for c in client_test])
                            logits = get_logits(model, all_X)
                            probs = get_probabilities(model, all_X)
                            acc = evaluate_model(model, all_X, all_y)

                        # ECE before calibration
                        ece_before, bins_before = expected_calibration_error(probs, all_y)

                        # Split test for calibration: 50% cal, 50% eval
                        rng = np.random.RandomState(seed)
                        idx = rng.permutation(len(all_y))
                        mid = len(idx) // 2
                        cal_idx, eval_idx = idx[:mid], idx[mid:]

                        # Temperature scaling
                        cal_probs, best_T = temperature_scaling(
                            logits[cal_idx], all_y[cal_idx], logits[eval_idx])
                        ece_after, bins_after = expected_calibration_error(
                            cal_probs, all_y[eval_idx])

                        # Per-client ECE (using personal models when available)
                        per_client_ece = {}
                        for cid in client_test:
                            Xc, yc = client_test[cid]
                            cm = personal[cid] if (personal and cid in personal) else model
                            pc = get_probabilities(cm, Xc)
                            ec, _ = expected_calibration_error(pc, yc)
                            per_client_ece[str(cid)] = ec

                        # Overconfidence ratio: fraction of wrong predictions with conf > 0.8
                        confidences = np.max(probs, axis=1)
                        predictions = np.argmax(probs, axis=1)
                        wrong = predictions != all_y
                        overconf = float((confidences[wrong] > 0.8).mean()) if wrong.sum() > 0 else 0.0

                        elapsed = time.time() - t_exp
                        checkpoint["results"][key] = {
                            "block": "M_calibration",
                            "dataset": ds, "algorithm": algo,
                            "dp_level": dp_name, "dp_epsilon": dp_eps,
                            "seed": seed, "num_rounds": num_rounds,
                            "accuracy": round(acc, 4),
                            "ece_before_calibration": ece_before,
                            "ece_after_temperature_scaling": ece_after,
                            "optimal_temperature": best_T,
                            "per_client_ece": per_client_ece,
                            "overconfidence_ratio": round(overconf, 4),
                            "time_seconds": round(elapsed, 1),
                        }
                        save_checkpoint(checkpoint)
                    except Exception as e:
                        log("  ERROR: {} — {}".format(key, e))
                        traceback.print_exc()
                    _cleanup_gpu()

    elapsed = time.time() - t0
    checkpoint["metadata"]["block_m_time"] = round(elapsed, 1)
    save_checkpoint(checkpoint)
    log("Block M complete: {}/{} in {}".format(count, total, format_time(elapsed)))
    return count


# ======================================================================
# Block N: Conformal Prediction under FL
# ======================================================================

def compute_conformal_prediction(model, cal_X, cal_y, test_X, test_y, alpha=0.1):
    """Federated conformal prediction with coverage guarantee 1-alpha."""
    # Non-conformity scores on calibration set
    probs_cal = get_probabilities(model, cal_X)
    scores = 1.0 - probs_cal[np.arange(len(cal_y)), cal_y.astype(int)]

    # Quantile (finite-sample correction)
    n_cal = len(scores)
    q_level = np.ceil((1 - alpha) * (n_cal + 1)) / n_cal
    q_level = min(q_level, 1.0)
    threshold = np.quantile(scores, q_level)

    # Prediction sets on test data
    probs_test = get_probabilities(model, test_X)
    num_classes = probs_test.shape[1]
    set_sizes = []
    covered = 0
    for i in range(len(test_y)):
        pred_set = []
        for c in range(num_classes):
            if 1.0 - probs_test[i, c] <= threshold:
                pred_set.append(c)
        set_sizes.append(len(pred_set))
        if int(test_y[i]) in pred_set:
            covered += 1

    coverage = covered / len(test_y) if len(test_y) > 0 else 0.0
    avg_set_size = np.mean(set_sizes) if set_sizes else 0.0
    empty_sets = sum(1 for s in set_sizes if s == 0)
    singleton_sets = sum(1 for s in set_sizes if s == 1)

    return {
        "coverage": round(float(coverage), 4),
        "target_coverage": round(1.0 - alpha, 4),
        "avg_set_size": round(float(avg_set_size), 4),
        "threshold": round(float(threshold), 4),
        "empty_set_fraction": round(float(empty_sets / len(test_y)), 4) if len(test_y) > 0 else 0,
        "singleton_fraction": round(float(singleton_sets / len(test_y)), 4) if len(test_y) > 0 else 0,
    }


def run_block_n(checkpoint, quick=False):
    """Block N: Conformal Prediction under FL."""
    log("=" * 70)
    log("BLOCK N: Conformal Prediction under FL")
    log("=" * 70)

    conditions = ["IID", "NonIID"]
    dp_levels = [None, 10.0, 1.0]
    dp_names = ["noDP", "eps10", "eps1"]
    num_rounds = 10 if quick else 30
    seeds = [42] if quick else DEFAULT_SEEDS
    t0 = time.time()
    count = 0
    total = len(DATASETS) * len(conditions) * len(dp_levels) * len(seeds)

    for ds in DATASETS:
        cfg = DATASET_CONFIGS[ds]
        for cond in conditions:
            is_iid = (cond == "IID")
            for dp_eps, dp_name in zip(dp_levels, dp_names):
                for seed in seeds:
                    if _shutdown:
                        return count
                    key = "N_{}_{}_{}_s{}".format(ds, cond, dp_name, seed)
                    if key in checkpoint["results"]:
                        count += 1
                        continue
                    count += 1
                    log("  [{}/{}] {} / {} / {} / seed={}".format(
                        count, total, ds, cond, dp_name, seed))
                    try:
                        t_exp = time.time()
                        model, _, client_data, client_test, meta = train_fl_model(
                            ds, cfg, "FedAvg", num_rounds, seed,
                            is_iid=is_iid, dp_epsilon=dp_eps)

                        # Split test: 50% calibration, 50% test
                        all_X = np.concatenate([client_test[c][0] for c in client_test])
                        all_y = np.concatenate([client_test[c][1] for c in client_test])
                        rng = np.random.RandomState(seed)
                        idx = rng.permutation(len(all_y))
                        mid = len(idx) // 2
                        cal_X, cal_y = all_X[idx[:mid]], all_y[idx[:mid]]
                        test_X, test_y = all_X[idx[mid:]], all_y[idx[mid:]]

                        # Global conformal prediction (alpha=0.1 → 90% coverage target)
                        global_cp = compute_conformal_prediction(
                            model, cal_X, cal_y, test_X, test_y, alpha=0.1)

                        # Per-client conformal: federated calibration
                        # Each client contributes scores, server aggregates quantile
                        all_scores = []
                        for cid in client_test:
                            Xc, yc = client_test[cid]
                            pc = get_probabilities(model, Xc)
                            sc = 1.0 - pc[np.arange(len(yc)), yc.astype(int)]
                            all_scores.extend(sc.tolist())

                        # Federated quantile
                        n_total = len(all_scores)
                        q_level = min(np.ceil(0.9 * (n_total + 1)) / n_total, 1.0)
                        fed_threshold = np.quantile(all_scores, q_level)

                        # Per-client coverage with federated threshold
                        per_client_coverage = {}
                        for cid in client_test:
                            Xc, yc = client_test[cid]
                            pc = get_probabilities(model, Xc)
                            covered = 0
                            for i in range(len(yc)):
                                pred_set = [c for c in range(pc.shape[1])
                                            if 1.0 - pc[i, c] <= fed_threshold]
                                if int(yc[i]) in pred_set:
                                    covered += 1
                            per_client_coverage[str(cid)] = round(
                                covered / len(yc), 4) if len(yc) > 0 else 0

                        acc = evaluate_model(model, all_X, all_y)
                        elapsed = time.time() - t_exp

                        checkpoint["results"][key] = {
                            "block": "N_conformal",
                            "dataset": ds, "condition": cond,
                            "dp_level": dp_name, "dp_epsilon": dp_eps,
                            "seed": seed, "num_rounds": num_rounds,
                            "accuracy": round(acc, 4),
                            "global_conformal": global_cp,
                            "federated_threshold": round(float(fed_threshold), 4),
                            "per_client_coverage": per_client_coverage,
                            "time_seconds": round(elapsed, 1),
                        }
                        save_checkpoint(checkpoint)
                    except Exception as e:
                        log("  ERROR: {} — {}".format(key, e))
                        traceback.print_exc()
                    _cleanup_gpu()

    elapsed = time.time() - t0
    checkpoint["metadata"]["block_n_time"] = round(elapsed, 1)
    save_checkpoint(checkpoint)
    log("Block N complete: {}/{} in {}".format(count, total, format_time(elapsed)))
    return count


# ======================================================================
# Block O: Feature Attribution Stability (permutation importance)
# ======================================================================

def permutation_importance(model, X, y, n_repeats=5, seed=42):
    """Compute permutation importance for each feature."""
    rng = np.random.RandomState(seed)
    base_acc = evaluate_model(model, X, y)
    n_features = X.shape[1]
    importances = np.zeros(n_features)

    for f in range(n_features):
        drops = []
        for _ in range(n_repeats):
            X_perm = X.copy()
            X_perm[:, f] = rng.permutation(X_perm[:, f])
            perm_acc = evaluate_model(model, X_perm, y)
            drops.append(base_acc - perm_acc)
        importances[f] = np.mean(drops)

    return importances


def run_block_o(checkpoint, quick=False):
    """Block O: Feature Attribution Stability."""
    log("=" * 70)
    log("BLOCK O: Feature Attribution Stability (Permutation Importance)")
    log("=" * 70)

    algorithms = ["FedAvg", "Ditto", "HPFL"]
    conditions = [("IID", True, None), ("NonIID_DP10", False, 10.0)]
    num_rounds = 10 if quick else 30
    seeds = [42] if quick else DEFAULT_SEEDS
    n_repeats = 3 if quick else 5
    t0 = time.time()
    count = 0
    total = len(DATASETS) * len(algorithms) * len(conditions) * len(seeds)

    for ds in DATASETS:
        cfg = DATASET_CONFIGS[ds]
        for algo in algorithms:
            for cond_name, is_iid, dp_eps in conditions:
                for seed in seeds:
                    if _shutdown:
                        return count
                    key = "O_{}_{}_{}_s{}".format(ds, algo, cond_name, seed)
                    if key in checkpoint["results"]:
                        count += 1
                        continue
                    count += 1
                    log("  [{}/{}] {} / {} / {} / seed={}".format(
                        count, total, ds, algo, cond_name, seed))
                    try:
                        t_exp = time.time()
                        model, personal, client_data, client_test, meta = train_fl_model(
                            ds, cfg, algo, num_rounds, seed,
                            is_iid=is_iid, dp_epsilon=dp_eps)

                        # Global feature importance (per-client with personal models)
                        all_X = np.concatenate([client_test[c][0] for c in client_test])
                        all_y = np.concatenate([client_test[c][1] for c in client_test])
                        if personal:
                            weighted_imps = []
                            for cid in client_test:
                                Xc, yc = client_test[cid]
                                cm = personal[cid] if cid in personal else model
                                imp = permutation_importance(cm, Xc, yc,
                                                            n_repeats=n_repeats, seed=seed)
                                weighted_imps.append(imp * len(yc))
                            global_imp = sum(weighted_imps) / len(all_y)
                        else:
                            global_imp = permutation_importance(model, all_X, all_y,
                                                               n_repeats=n_repeats, seed=seed)

                        # Per-client feature importance
                        per_client_imp = {}
                        for cid in client_test:
                            Xc, yc = client_test[cid]
                            if personal and cid in personal:
                                cm = personal[cid]
                            else:
                                cm = model
                            imp = permutation_importance(cm, Xc, yc, n_repeats=n_repeats, seed=seed)
                            per_client_imp[str(cid)] = [round(float(v), 6) for v in imp]

                        # Cross-client consistency: mean pairwise rank correlation
                        client_ranks = []
                        for cid in per_client_imp:
                            imp_arr = np.array(per_client_imp[cid])
                            client_ranks.append(scipy_stats.rankdata(-imp_arr))

                        rank_corrs = []
                        for i in range(len(client_ranks)):
                            for j in range(i + 1, len(client_ranks)):
                                corr, _ = scipy_stats.spearmanr(client_ranks[i], client_ranks[j])
                                if not np.isnan(corr):
                                    rank_corrs.append(corr)
                        mean_rank_corr = float(np.mean(rank_corrs)) if rank_corrs else 0.0

                        # Top-3 features (by global importance)
                        top3_idx = np.argsort(-global_imp)[:3]
                        feature_names = meta.get("feature_names", [str(i) for i in range(len(global_imp))])
                        top3_features = [feature_names[i] if i < len(feature_names) else str(i)
                                         for i in top3_idx]

                        # Accuracy using personal models when available
                        if personal:
                            correct_w = sum(
                                evaluate_model(
                                    personal[cid] if cid in personal else model,
                                    client_test[cid][0], client_test[cid][1]
                                ) * len(client_test[cid][1])
                                for cid in client_test
                            )
                            acc = correct_w / len(all_y)
                        else:
                            acc = evaluate_model(model, all_X, all_y)
                        elapsed = time.time() - t_exp

                        checkpoint["results"][key] = {
                            "block": "O_feature_attribution",
                            "dataset": ds, "algorithm": algo,
                            "condition": cond_name, "seed": seed,
                            "num_rounds": num_rounds,
                            "accuracy": round(acc, 4),
                            "global_importance": [round(float(v), 6) for v in global_imp],
                            "per_client_importance": per_client_imp,
                            "cross_client_rank_correlation": round(mean_rank_corr, 4),
                            "top3_features": top3_features,
                            "time_seconds": round(elapsed, 1),
                        }
                        save_checkpoint(checkpoint)
                    except Exception as e:
                        log("  ERROR: {} — {}".format(key, e))
                        traceback.print_exc()
                    _cleanup_gpu()

    elapsed = time.time() - t0
    checkpoint["metadata"]["block_o_time"] = round(elapsed, 1)
    save_checkpoint(checkpoint)
    log("Block O complete: {}/{} in {}".format(count, total, format_time(elapsed)))
    return count


# ======================================================================
# Block P: Demographic Fairness
# ======================================================================

def compute_demographic_metrics(predictions, labels, protected_attr):
    """Compute fairness metrics by protected attribute groups."""
    groups = np.unique(protected_attr)
    group_metrics = {}
    for g in groups:
        mask = protected_attr == g
        if mask.sum() == 0:
            continue
        g_acc = float((predictions[mask] == labels[mask]).mean())
        # True positive rate (recall for positive class)
        pos = labels[mask] == 1
        tpr = float(predictions[mask][pos].mean()) if pos.sum() > 0 else 0.0
        # False positive rate
        neg = labels[mask] == 0
        fpr = float(predictions[mask][neg].mean()) if neg.sum() > 0 else 0.0
        # Positive prediction rate (for demographic parity)
        ppr = float(predictions[mask].mean())
        group_metrics[str(int(g))] = {
            "accuracy": round(g_acc, 4),
            "tpr": round(tpr, 4),
            "fpr": round(fpr, 4),
            "positive_rate": round(ppr, 4),
            "n_samples": int(mask.sum()),
        }

    # Disparity metrics
    if len(group_metrics) >= 2:
        accs = [v["accuracy"] for v in group_metrics.values()]
        tprs = [v["tpr"] for v in group_metrics.values()]
        fprs = [v["fpr"] for v in group_metrics.values()]
        pprs = [v["positive_rate"] for v in group_metrics.values()]
        return {
            "group_metrics": group_metrics,
            "accuracy_gap": round(max(accs) - min(accs), 4),
            "equalized_odds_tpr_gap": round(max(tprs) - min(tprs), 4),
            "equalized_odds_fpr_gap": round(max(fprs) - min(fprs), 4),
            "demographic_parity_gap": round(max(pprs) - min(pprs), 4),
        }
    return {"group_metrics": group_metrics}


def run_block_p(checkpoint, quick=False):
    """Block P: Demographic Fairness (Cardiovascular — has gender/age features)."""
    log("=" * 70)
    log("BLOCK P: Demographic Fairness (Equalized Odds, Demographic Parity)")
    log("=" * 70)

    # Only Cardiovascular has clear demographic features (gender=col 1, age=col 0)
    ds = "Cardiovascular"
    cfg = DATASET_CONFIGS[ds]
    algorithms = ["FedAvg", "Ditto", "HPFL"]
    dp_levels = [None, 10.0, 1.0]
    dp_names = ["noDP", "eps10", "eps1"]
    conditions = [("IID", True), ("NonIID", False)]
    num_rounds = 10 if quick else 30
    seeds = [42] if quick else DEFAULT_SEEDS
    t0 = time.time()
    count = 0
    total = len(algorithms) * len(dp_levels) * len(conditions) * len(seeds)

    for algo in algorithms:
        for dp_eps, dp_name in zip(dp_levels, dp_names):
            for cond_name, is_iid in conditions:
                for seed in seeds:
                    if _shutdown:
                        return count
                    key = "P_{}_{}_{}_s{}".format(algo, cond_name, dp_name, seed)
                    if key in checkpoint["results"]:
                        count += 1
                        continue
                    count += 1
                    log("  [{}/{}] {} / {} / {} / seed={}".format(
                        count, total, algo, cond_name, dp_name, seed))
                    try:
                        t_exp = time.time()
                        model, personal, client_data, client_test, meta = train_fl_model(
                            ds, cfg, algo, num_rounds, seed,
                            is_iid=is_iid, dp_epsilon=dp_eps)

                        # Evaluate using personal models when available
                        if personal:
                            all_X_parts, all_y_parts, all_probs_parts = [], [], []
                            for cid in client_test:
                                Xc, yc = client_test[cid]
                                cm = personal[cid] if cid in personal else model
                                all_X_parts.append(Xc)
                                all_y_parts.append(yc)
                                all_probs_parts.append(get_probabilities(cm, Xc))
                            all_X = np.concatenate(all_X_parts)
                            all_y = np.concatenate(all_y_parts)
                            probs = np.concatenate(all_probs_parts)
                        else:
                            all_X = np.concatenate([client_test[c][0] for c in client_test])
                            all_y = np.concatenate([client_test[c][1] for c in client_test])
                            probs = get_probabilities(model, all_X)
                        preds = np.argmax(probs, axis=1)
                        acc = float((preds == all_y).mean())

                        # Gender fairness (feature index 1 in Cardiovascular: 1=Female, 2=Male)
                        # Raw values NOT normalized, so threshold at 1.5
                        gender_attr = (all_X[:, 1] >= 1.5).astype(int)  # 0=Female, 1=Male
                        gender_fairness = compute_demographic_metrics(preds, all_y, gender_attr)

                        # Age fairness (feature index 0, normalized 0-1)
                        # Split into young (<0.5 = <50yo) and old (>=0.5 = >=50yo)
                        age_attr = (all_X[:, 0] >= 0.5).astype(int)
                        age_fairness = compute_demographic_metrics(preds, all_y, age_attr)

                        elapsed = time.time() - t_exp
                        checkpoint["results"][key] = {
                            "block": "P_demographic_fairness",
                            "dataset": ds, "algorithm": algo,
                            "condition": cond_name, "dp_level": dp_name,
                            "dp_epsilon": dp_eps, "seed": seed,
                            "num_rounds": num_rounds,
                            "accuracy": round(acc, 4),
                            "gender_fairness": gender_fairness,
                            "age_fairness": age_fairness,
                            "time_seconds": round(elapsed, 1),
                        }
                        save_checkpoint(checkpoint)
                    except Exception as e:
                        log("  ERROR: {} — {}".format(key, e))
                        traceback.print_exc()
                    _cleanup_gpu()

    elapsed = time.time() - t0
    checkpoint["metadata"]["block_p_time"] = round(elapsed, 1)
    save_checkpoint(checkpoint)
    log("Block P complete: {}/{} in {}".format(count, total, format_time(elapsed)))
    return count


# ======================================================================
# Block Q: Concept Drift Robustness
# ======================================================================

def run_block_q(checkpoint, quick=False):
    """Block Q: Concept Drift Robustness on Cardiovascular (70K samples)."""
    log("=" * 70)
    log("BLOCK Q: Concept Drift Robustness")
    log("=" * 70)

    ds = "Cardiovascular"
    cfg = DATASET_CONFIGS[ds]
    algorithms = ["FedAvg", "Ditto", "HPFL"]
    drift_severities = ["mild", "severe"]
    adaptations = ["none", "retrain", "ema"]  # no adapt, full retrain, EMA of old+new
    num_rounds = 10 if quick else 30
    seeds = [42] if quick else DEFAULT_SEEDS
    t0 = time.time()
    count = 0
    total = len(algorithms) * len(drift_severities) * len(adaptations) * len(seeds)

    for algo in algorithms:
        for drift in drift_severities:
            for adapt in adaptations:
                for seed in seeds:
                    if _shutdown:
                        return count
                    key = "Q_{}_{}_{}_s{}".format(algo, drift, adapt, seed)
                    if key in checkpoint["results"]:
                        count += 1
                        continue
                    count += 1
                    log("  [{}/{}] {} / drift={} / adapt={} / seed={}".format(
                        count, total, algo, drift, adapt, seed))
                    try:
                        t_exp = time.time()
                        # Train on original data
                        model, personal, client_data, client_test, meta = train_fl_model(
                            ds, cfg, algo, num_rounds, seed, is_iid=True)

                        # Pre-drift accuracy (per-client with personal models)
                        all_X_parts, all_y_parts = [], []
                        pre_correct, total_samples = 0, 0
                        for cid in client_test:
                            Xc, yc = client_test[cid]
                            all_X_parts.append(Xc)
                            all_y_parts.append(yc)
                            cm = personal[cid] if (personal and cid in personal) else model
                            pre_correct += evaluate_model(cm, Xc, yc) * len(yc)
                            total_samples += len(yc)
                        all_X = np.concatenate(all_X_parts)
                        all_y = np.concatenate(all_y_parts)
                        pre_drift_acc = pre_correct / total_samples

                        # Simulate drift: perturb test data features
                        rng = np.random.RandomState(seed + 999)
                        drift_X = all_X.copy()
                        if drift == "mild":
                            for f in [0, 1, 2]:
                                drift_X[:, f] += rng.normal(0, 0.1, len(drift_X))
                                drift_X[:, f] = np.clip(drift_X[:, f], 0, 1)
                        else:  # severe
                            for f in [0, 1, 2, 3, 4, 5]:
                                drift_X[:, f] += rng.normal(0, 0.3, len(drift_X))
                                drift_X[:, f] = np.clip(drift_X[:, f], 0, 1)

                        drift_y = all_y.copy()
                        if drift == "severe":
                            flip_idx = rng.choice(len(drift_y),
                                                  size=max(1, int(0.05 * len(drift_y))),
                                                  replace=False)
                            drift_y[flip_idx] = 1 - drift_y[flip_idx]

                        # Post-drift accuracy (per-client with personal models)
                        offset = 0
                        post_correct = 0
                        for cid in client_test:
                            n_c = len(client_test[cid][1])
                            cm = personal[cid] if (personal and cid in personal) else model
                            post_correct += evaluate_model(
                                cm, drift_X[offset:offset + n_c],
                                drift_y[offset:offset + n_c]) * n_c
                            offset += n_c
                        post_drift_acc = post_correct / total_samples

                        # Adaptation (per-client from personal models)
                        adapted_acc = post_drift_acc
                        if adapt == "retrain":
                            offset = 0
                            adapt_correct = 0
                            for cid in client_test:
                                n_c = len(client_test[cid][1])
                                cm = personal[cid] if (personal and cid in personal) else model
                                adapted_cm = create_model(cfg["input_dim"], cfg["num_classes"])
                                set_params(adapted_cm, get_params(cm))
                                train_local_sgd(
                                    adapted_cm, drift_X[offset:offset + n_c],
                                    drift_y[offset:offset + n_c],
                                    epochs=5, lr=cfg["learning_rate"] * 0.1,
                                    batch_size=cfg["batch_size"])
                                adapt_correct += evaluate_model(
                                    adapted_cm, drift_X[offset:offset + n_c],
                                    drift_y[offset:offset + n_c]) * n_c
                                offset += n_c
                            adapted_acc = adapt_correct / total_samples
                        elif adapt == "ema":
                            offset = 0
                            adapt_correct = 0
                            for cid in client_test:
                                n_c = len(client_test[cid][1])
                                cm = personal[cid] if (personal and cid in personal) else model
                                adapted_cm = create_model(cfg["input_dim"], cfg["num_classes"])
                                set_params(adapted_cm, get_params(cm))
                                train_local_sgd(
                                    adapted_cm, drift_X[offset:offset + n_c],
                                    drift_y[offset:offset + n_c],
                                    epochs=3, lr=cfg["learning_rate"] * 0.1,
                                    batch_size=cfg["batch_size"])
                                old_p = get_params(cm)
                                new_p = get_params(adapted_cm)
                                ema_p = {pn: 0.3 * old_p[pn] + 0.7 * new_p[pn]
                                         for pn in old_p}
                                set_params(adapted_cm, ema_p)
                                adapt_correct += evaluate_model(
                                    adapted_cm, drift_X[offset:offset + n_c],
                                    drift_y[offset:offset + n_c]) * n_c
                                offset += n_c
                            adapted_acc = adapt_correct / total_samples

                        # Drift detection metric: KL divergence (global model)
                        pre_probs = get_probabilities(model, all_X)
                        post_probs = get_probabilities(model, drift_X)
                        pre_dist = pre_probs.mean(axis=0)
                        post_dist = post_probs.mean(axis=0)
                        pre_dist = np.clip(pre_dist, 1e-10, 1)
                        post_dist = np.clip(post_dist, 1e-10, 1)
                        kl_div = float(np.sum(pre_dist * np.log(pre_dist / post_dist)))

                        elapsed = time.time() - t_exp
                        checkpoint["results"][key] = {
                            "block": "Q_concept_drift",
                            "dataset": ds, "algorithm": algo,
                            "drift_severity": drift, "adaptation": adapt,
                            "seed": seed, "num_rounds": num_rounds,
                            "pre_drift_accuracy": round(pre_drift_acc, 4),
                            "post_drift_accuracy": round(post_drift_acc, 4),
                            "adapted_accuracy": round(adapted_acc, 4),
                            "accuracy_drop": round(pre_drift_acc - post_drift_acc, 4),
                            "recovery": round(adapted_acc - post_drift_acc, 4),
                            "kl_divergence": round(kl_div, 6),
                            "time_seconds": round(elapsed, 1),
                        }
                        save_checkpoint(checkpoint)
                    except Exception as e:
                        log("  ERROR: {} — {}".format(key, e))
                        traceback.print_exc()
                    _cleanup_gpu()

    elapsed = time.time() - t0
    checkpoint["metadata"]["block_q_time"] = round(elapsed, 1)
    save_checkpoint(checkpoint)
    log("Block Q complete: {}/{} in {}".format(count, total, format_time(elapsed)))
    return count


# ======================================================================
# Block R: DP Composition Across Studies
# ======================================================================

def rdp_gaussian(alpha_rdp, sigma):
    """RDP for Gaussian mechanism at order alpha."""
    return alpha_rdp / (2.0 * sigma ** 2)


def rdp_to_dp(rdp_values, alphas, delta=1e-5):
    """Convert RDP to (epsilon, delta)-DP via optimal alpha."""
    candidates = []
    for i in range(len(alphas)):
        if alphas[i] > 1:
            eps_candidate = rdp_values[i] + np.log(1.0 / delta) / (alphas[i] - 1)
            candidates.append(eps_candidate)
    return max(min(candidates), 0) if candidates else float("inf")


def run_block_r(checkpoint, quick=False):
    """Block R: DP Composition Across Multiple EHDS Studies."""
    log("=" * 70)
    log("BLOCK R: DP Composition Across Studies (EHDS Multi-Study)")
    log("=" * 70)

    per_study_epsilons = [1.0, 5.0, 10.0]
    num_studies_list = [1, 2, 5, 10, 20]
    delta = 1e-5
    # Fine-grained RDP orders for tight bounds
    alphas = np.concatenate([
        np.arange(1.1, 2.0, 0.1),    # fractional orders 1.1-1.9
        np.arange(2, 32, 1),          # integer orders 2-31
        np.arange(32, 128, 4),        # coarser 32-127
        np.arange(128, 513, 16),      # coarse 128-512
    ])
    t0 = time.time()
    count = 0
    total = len(DATASETS) * len(per_study_epsilons) * len(num_studies_list)

    for ds in DATASETS:
        cfg = DATASET_CONFIGS[ds]
        num_rounds = 10 if quick else 30

        for per_eps in per_study_epsilons:
            for n_studies in num_studies_list:
                if _shutdown:
                    return count
                key = "R_{}_eps{}_studies{}".format(ds, per_eps, n_studies)
                if key in checkpoint["results"]:
                    count += 1
                    continue
                count += 1
                log("  [{}/{}] {} / per_eps={} / studies={}".format(
                    count, total, ds, per_eps, n_studies))
                try:
                    # Calibrate Gaussian sigma for (eps, delta)-DP per study
                    # Standard formula: sigma = sqrt(2 * ln(1.25/delta)) / epsilon
                    sigma = np.sqrt(2.0 * np.log(1.25 / delta)) / per_eps

                    # 1. Naive composition: epsilon_total = k * epsilon
                    naive_total = per_eps * n_studies

                    # 2. Advanced composition (Dwork-Rothblum-Vadhan 2010)
                    # total_eps <= eps * sqrt(2k * ln(1/delta')) + k * eps * (e^eps - 1) / 2
                    delta_prime = delta
                    adv_term1 = per_eps * np.sqrt(2.0 * n_studies * np.log(1.0 / delta_prime))
                    adv_term2 = n_studies * per_eps * (np.exp(per_eps) - 1) / 2.0
                    advanced_total = min(adv_term1 + adv_term2, naive_total)

                    # 3. RDP composition: sum of RDP guarantees, then convert
                    rdp_per_study = np.array([rdp_gaussian(a, sigma) for a in alphas])
                    rdp_composed = rdp_per_study * n_studies
                    rdp_total = rdp_to_dp(rdp_composed, alphas, delta)
                    # Cap: RDP can't exceed naive (fallback guarantee)
                    rdp_total = min(rdp_total, naive_total)

                    # Compute utility: train with per-study epsilon
                    utility_acc = None
                    if n_studies <= 5 and not quick:
                        model, _, _, client_test, _ = train_fl_model(
                            ds, cfg, "FedAvg", num_rounds, 42,
                            dp_epsilon=per_eps)
                        all_X = np.concatenate([client_test[c][0] for c in client_test])
                        all_y = np.concatenate([client_test[c][1] for c in client_test])
                        utility_acc = round(evaluate_model(model, all_X, all_y), 4)

                    checkpoint["results"][key] = {
                        "block": "R_dp_composition",
                        "dataset": ds,
                        "per_study_epsilon": per_eps,
                        "num_studies": n_studies,
                        "delta": delta,
                        "sigma": round(float(sigma), 4),
                        "naive_composition_epsilon": round(naive_total, 4),
                        "advanced_composition_epsilon": round(advanced_total, 4),
                        "rdp_composition_epsilon": round(rdp_total, 4),
                        "rdp_improvement_ratio": round(naive_total / max(rdp_total, 1e-10), 2),
                        "utility_accuracy": utility_acc,
                    }
                    save_checkpoint(checkpoint)
                except Exception as e:
                    log("  ERROR: {} — {}".format(key, e))
                    traceback.print_exc()

    elapsed = time.time() - t0
    checkpoint["metadata"]["block_r_time"] = round(elapsed, 1)
    save_checkpoint(checkpoint)
    log("Block R complete: {}/{} in {}".format(count, total, format_time(elapsed)))
    return count


# ======================================================================
# Main
# ======================================================================

def main():
    global _log_file, _shutdown

    parser = argparse.ArgumentParser(
        description="FL-EHDS Cascade 7: Trustworthiness & EHDS Advanced")
    parser.add_argument("--quick", action="store_true",
                        help="Reduced rounds/seeds for quick test")
    parser.add_argument("--fresh", action="store_true",
                        help="Delete existing checkpoint and start fresh")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, _handle_signal)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _log_file = open(OUTPUT_DIR / LOG_FILE, "a", encoding="utf-8")

    log("=" * 70)
    log("FL-EHDS Cascading Analysis — Phase 7 (Trustworthiness & EHDS Advanced)")
    log("Mode: {}".format("QUICK" if args.quick else "FULL"))
    log("=" * 70)

    # Handle --fresh
    if args.fresh:
        for f in [CHECKPOINT_FILE, CHECKPOINT_FILE + ".bak"]:
            p = OUTPUT_DIR / f
            if p.exists():
                p.unlink()
                log("Deleted: {}".format(f))

    # Load or create checkpoint
    checkpoint = load_checkpoint()
    if checkpoint is None:
        checkpoint = {
            "metadata": {
                "experiment": "cascade7_trustworthiness",
                "started": datetime.now().isoformat(),
                "mode": "quick" if args.quick else "full",
            },
            "results": {},
            "completed": [],
        }
        save_checkpoint(checkpoint)

    # Block M: Calibration
    if not _shutdown:
        n_m = run_block_m(checkpoint, quick=args.quick)
        if "M" not in checkpoint.get("completed", []):
            checkpoint.setdefault("completed", []).append("M")

    # Block N: Conformal Prediction
    if not _shutdown:
        n_n = run_block_n(checkpoint, quick=args.quick)
        if "N" not in checkpoint.get("completed", []):
            checkpoint.setdefault("completed", []).append("N")

    # Block O: Feature Attribution
    if not _shutdown:
        n_o = run_block_o(checkpoint, quick=args.quick)
        if "O" not in checkpoint.get("completed", []):
            checkpoint.setdefault("completed", []).append("O")

    # Block P: Demographic Fairness
    if not _shutdown:
        n_p = run_block_p(checkpoint, quick=args.quick)
        if "P" not in checkpoint.get("completed", []):
            checkpoint.setdefault("completed", []).append("P")

    # Block Q: Concept Drift
    if not _shutdown:
        n_q = run_block_q(checkpoint, quick=args.quick)
        if "Q" not in checkpoint.get("completed", []):
            checkpoint.setdefault("completed", []).append("Q")

    # Block R: DP Composition
    if not _shutdown:
        n_r = run_block_r(checkpoint, quick=args.quick)
        if "R" not in checkpoint.get("completed", []):
            checkpoint.setdefault("completed", []).append("R")

    # Final save
    checkpoint["metadata"]["finished"] = datetime.now().isoformat()
    save_checkpoint(checkpoint)

    # Summary
    total_time = time.time()
    blocks = {"M": 0, "N": 0, "O": 0, "P": 0, "Q": 0, "R": 0}
    for k in checkpoint["results"]:
        prefix = k[0]
        if prefix in blocks:
            blocks[prefix] += 1
    total_exp = sum(blocks.values())

    started = datetime.fromisoformat(checkpoint["metadata"]["started"])
    elapsed_total = datetime.now() - started

    log("=" * 70)
    log("ALL BLOCKS COMPLETE in {}".format(str(elapsed_total).split(".")[0]))
    log("Checkpoint: {}".format(CHECKPOINT_FILE))
    for block, n in sorted(blocks.items()):
        log("  Block {}: {} experiments".format(block, n))
    log("  TOTAL: {} experiments".format(total_exp))
    log("=" * 70)

    if _log_file:
        _log_file.close()


if __name__ == "__main__":
    main()

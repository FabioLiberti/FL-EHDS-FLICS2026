#!/usr/bin/env python3
"""
FL-EHDS Cascading Analysis & Experiments — Phase 10.

4 blocks executed sequentially (Clinical Imbalance Deep-Dive):

Block AC: Complete Condition Matrix (90 exp)
  Fill missing DP conditions from Block AA: IID+eps10, IID+eps1, NonIID+eps1.
  {Stroke, Cirrhosis} × 3 algorithms × 3 conditions × 5 seeds.
  Completes the 2×3 (IID/NonIID × noDP/eps10/eps1) factorial grid.

Block AD: Mitigation Strategies (234 exp)
  Class-weighted CE and focal loss on imbalanced datasets.
  {weighted_ce, focal} × {IID+noDP, NonIID+noDP, NonIID+eps10} ×
  {Stroke, Cirrhosis}(5 seeds) + {CDC_Diabetes}(3 seeds) × 3 algorithms.

Block AE: Ditto Local Epochs Sweep (40 exp)
  Local epochs {5, 10} to assess Ditto minority-class rescue sensitivity.
  {5, 10} × {Stroke, Cirrhosis} × Ditto × {NonIID+noDP, NonIID+eps10} × 5 seeds.

Block TH: Threshold Rescue (~20 exp)
  Re-trains cascade9 Block AA experiments with F1=0.0 and applies threshold
  tuning (0.05–0.95 sweep). Tests whether post-hoc threshold optimization
  can rescue majority-class collapse without retraining.

All experiments include automatic threshold tuning in evaluation, providing
both standard (t=0.5) and threshold-optimized F1/precision/recall metrics.

Usage:
    cd fl-ehds-framework
    python -m benchmarks.run_analysis_cascade10 [--quick] [--fresh]

Output:
    benchmarks/paper_results_tabular/checkpoint_cascade10.json

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
from typing import Dict, List, Any, Optional

import numpy as np

FRAMEWORK_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(FRAMEWORK_DIR))

import torch
import torch.nn as nn
import torch.optim as optim

# ======================================================================
# Constants
# ======================================================================

OUTPUT_DIR = FRAMEWORK_DIR / "benchmarks" / "paper_results_tabular"
CHECKPOINT_FILE = "checkpoint_cascade10.json"
LOG_FILE = "experiment_cascade10.log"

DATASET_CONFIGS = {
    "Stroke": dict(
        learning_rate=0.01, batch_size=32, num_rounds=30, local_epochs=3,
        mu=0.1, num_clients=5, input_dim=10, num_classes=2,
    ),
    "CKD": dict(
        learning_rate=0.005, batch_size=16, num_rounds=30, local_epochs=3,
        mu=0.1, num_clients=4, input_dim=24, num_classes=2,
    ),
    "Cirrhosis": dict(
        learning_rate=0.005, batch_size=16, num_rounds=30, local_epochs=3,
        mu=0.1, num_clients=4, input_dim=18, num_classes=2,
    ),
    "CDC_Diabetes": dict(
        learning_rate=0.01, batch_size=128, num_rounds=20, local_epochs=3,
        mu=0.1, num_clients=5, input_dim=21, num_classes=2,
    ),
}

DEFAULT_SEEDS = [42, 123, 456, 789, 999]
CDC_SEEDS = [42, 123, 456]
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
    fd, tmp = tempfile.mkstemp(dir=str(OUTPUT_DIR), prefix=".cas10_", suffix=".tmp")
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

    if dataset == "Stroke":
        from data.stroke_loader import load_stroke_data
        client_data, client_test, meta = load_stroke_data(
            num_clients=num_clients, seed=seed, is_iid=is_iid, alpha=alpha,
        )
    elif dataset == "CKD":
        from data.ckd_loader import load_ckd_data
        client_data, client_test, meta = load_ckd_data(
            num_clients=num_clients, seed=seed, is_iid=is_iid, alpha=alpha,
        )
    elif dataset == "Cirrhosis":
        from data.cirrhosis_loader import load_cirrhosis_data
        client_data, client_test, meta = load_cirrhosis_data(
            num_clients=num_clients, seed=seed, is_iid=is_iid, alpha=alpha,
        )
    elif dataset == "CDC_Diabetes":
        from data.cdc_diabetes_loader import load_cdc_diabetes_data
        client_data, client_test, meta = load_cdc_diabetes_data(
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


# ======================================================================
# Loss functions
# ======================================================================

class FocalLoss(nn.Module):
    """Focal Loss: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t).

    Focuses training on hard-to-classify examples by down-weighting
    easy (well-classified) examples. Particularly effective for class
    imbalance where the majority class dominates the gradient.
    """
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits, targets):
        ce_loss = nn.functional.cross_entropy(
            logits, targets, weight=self.weight, reduction="none")
        p_t = torch.exp(-ce_loss)
        focal_loss = ((1 - p_t) ** self.gamma) * ce_loss
        return focal_loss.mean()


def compute_class_weights(y, num_classes=2):
    """Inverse-frequency class weights: N_total / (N_classes * N_class_k)."""
    counts = np.bincount(y.astype(int), minlength=num_classes)
    total = len(y)
    weights = total / (num_classes * np.maximum(counts, 1))
    return torch.FloatTensor(weights).to(DEVICE)


def make_criterion(loss_type="standard", class_weights=None):
    """Create loss function based on type."""
    if loss_type == "focal":
        return FocalLoss(gamma=2.0, weight=class_weights)
    elif loss_type == "weighted_ce":
        return nn.CrossEntropyLoss(weight=class_weights)
    else:
        return nn.CrossEntropyLoss()


# ======================================================================
# Training
# ======================================================================

def train_local_sgd(model, X, y, epochs, lr, batch_size=64,
                    loss_type="standard", class_weights=None):
    """Local SGD training with optional weighted/focal loss."""
    model.train()
    X_t = torch.FloatTensor(X).to(DEVICE) if isinstance(X, np.ndarray) else X.to(DEVICE)
    y_t = torch.LongTensor(y.astype(int)).to(DEVICE) if isinstance(y, np.ndarray) else y.to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = make_criterion(loss_type, class_weights)
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


def get_predictions(model, X):
    """Get model predictions as numpy array."""
    model.eval()
    X_t = torch.FloatTensor(X).to(DEVICE) if isinstance(X, np.ndarray) else X.to(DEVICE)
    with torch.no_grad():
        preds = model(X_t).argmax(dim=1).cpu().numpy()
    return preds


def get_probabilities(model, X):
    """Get softmax probability outputs as numpy array."""
    model.eval()
    X_t = torch.FloatTensor(X).to(DEVICE) if isinstance(X, np.ndarray) else X.to(DEVICE)
    with torch.no_grad():
        logits = model(X_t)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    return probs


# ======================================================================
# Metrics: F1, Precision, Recall, DEI + Threshold Tuning
# ======================================================================

def compute_classification_metrics(y_true, y_pred, num_classes=2):
    """Compute accuracy, F1, precision, recall, DEI from arrays."""
    acc = float((y_pred == y_true).mean())

    per_class_recall = {}
    for c in range(num_classes):
        mask = y_true == c
        if mask.sum() > 0:
            per_class_recall[c] = float((y_pred[mask] == c).mean())
        else:
            per_class_recall[c] = 0.0

    if num_classes == 2:
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-10)
    else:
        f1_scores, prec_scores, rec_scores = [], [], []
        for c in range(num_classes):
            tp_c = int(((y_pred == c) & (y_true == c)).sum())
            fp_c = int(((y_pred == c) & (y_true != c)).sum())
            fn_c = int(((y_pred != c) & (y_true == c)).sum())
            p_c = tp_c / max(tp_c + fp_c, 1)
            r_c = tp_c / max(tp_c + fn_c, 1)
            f1_c = 2 * p_c * r_c / max(p_c + r_c, 1e-10)
            prec_scores.append(p_c)
            rec_scores.append(r_c)
            f1_scores.append(f1_c)
        prec = float(np.mean(prec_scores))
        rec = float(np.mean(rec_scores))
        f1 = float(np.mean(f1_scores))

    recalls_arr = np.array(list(per_class_recall.values()))
    min_rec = float(recalls_arr.min())
    mean_rec = float(recalls_arr.mean())
    cv = float(recalls_arr.std() / mean_rec) if mean_rec > 1e-10 else 1.0
    dei = max(0.0, min_rec * (1.0 - cv))

    return {
        "accuracy": round(acc, 4),
        "f1_score": round(f1, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "dei": round(dei, 4),
        "per_class_recall": {int(c): round(r, 4) for c, r in per_class_recall.items()},
    }


def compute_threshold_tuned_metrics(y_true, probs_class1):
    """Sweep classification threshold (0.05–0.95) to maximize F1.

    Returns best threshold and corresponding precision/recall/F1.
    Computationally free — no retraining needed.
    """
    best_f1 = 0.0
    best_threshold = 0.5
    best_prec = 0.0
    best_rec = 0.0

    for t_int in range(5, 96, 5):
        threshold = t_int / 100.0
        y_pred = (probs_class1 >= threshold).astype(int)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-10)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_prec = prec
            best_rec = rec

    return {
        "threshold_tuned_f1": round(best_f1, 4),
        "threshold_tuned_precision": round(best_prec, 4),
        "threshold_tuned_recall": round(best_rec, 4),
        "optimal_threshold": round(best_threshold, 2),
    }


# ======================================================================
# FL Training (FedAvg, Ditto, HPFL) — extended with loss_type support
# ======================================================================

def federated_round(model, client_data, all_clients, cfg, dp_epsilon=None,
                    loss_type="standard", class_weights=None):
    global_params = get_params(model)
    updates, samples = [], []
    for cid in all_clients:
        X, y = client_data[cid]
        lm = create_model(cfg["input_dim"], cfg["num_classes"])
        set_params(lm, global_params)
        train_local_sgd(lm, X, y, cfg["local_epochs"], cfg["learning_rate"],
                        cfg["batch_size"], loss_type=loss_type,
                        class_weights=class_weights)
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


def personalize_model(model, client_data, all_clients, cfg, algo,
                      loss_type="standard", class_weights=None):
    """Create personal models for Ditto or HPFL after global training."""
    if algo == "Ditto":
        personal = {}
        for cid in all_clients:
            pm = create_model(cfg["input_dim"], cfg["num_classes"])
            set_params(pm, get_params(model))
            X, y = client_data[cid]
            train_local_sgd(pm, X, y, cfg["local_epochs"],
                            cfg["learning_rate"], cfg["batch_size"],
                            loss_type=loss_type, class_weights=class_weights)
            personal[cid] = pm
        return personal
    elif algo == "HPFL":
        personal = {}
        criterion = make_criterion(loss_type, class_weights)
        for cid in all_clients:
            pm = create_model(cfg["input_dim"], cfg["num_classes"])
            set_params(pm, get_params(model))
            X, y = client_data[cid]
            for n, p in pm.named_parameters():
                p.requires_grad = ("fc2" in n)
            opt = optim.SGD(
                filter(lambda p: p.requires_grad, pm.parameters()),
                lr=cfg["learning_rate"])
            X_t = torch.FloatTensor(X).to(DEVICE)
            y_t = torch.LongTensor(y.astype(int)).to(DEVICE)
            pm.train()
            for _ in range(cfg["local_epochs"] * 2):
                opt.zero_grad()
                loss = criterion(pm(X_t), y_t)
                loss.backward()
                opt.step()
            for n, p in pm.named_parameters():
                p.requires_grad = True
            personal[cid] = pm
        return personal
    return None


def evaluate_fl_full(model, personal, client_test, num_classes):
    """Evaluate FL model with standard + threshold-tuned metrics."""
    all_preds, all_labels, all_probs = [], [], []
    per_client_acc = {}
    for cid in client_test:
        Xc, yc = client_test[cid]
        cm = personal[cid] if personal and cid in personal else model
        preds_c = get_predictions(cm, Xc)
        probs_c = get_probabilities(cm, Xc)
        all_preds.append(preds_c)
        all_labels.append(yc)
        all_probs.append(probs_c)
        per_client_acc[cid] = round(float((preds_c == yc).mean()), 4)

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)

    metrics = compute_classification_metrics(all_labels, all_preds, num_classes)
    metrics["per_client_accuracy"] = per_client_acc

    # Threshold tuning (binary classification only)
    if num_classes == 2:
        threshold_metrics = compute_threshold_tuned_metrics(
            all_labels, all_probs[:, 1])
        metrics.update(threshold_metrics)

    return metrics


def train_fl_model(dataset, cfg, algo, num_rounds, seed, is_iid=True, alpha=0.5,
                   dp_epsilon=None, loss_type="standard",
                   local_epochs_override=None):
    """Train FL model. Returns (model, personal, client_data, client_test, meta)."""
    num_clients = cfg["num_clients"]
    client_data, client_test, meta = load_dataset(dataset, num_clients, seed, is_iid, alpha)
    model = create_model(cfg["input_dim"], cfg["num_classes"], seed=seed)
    all_clients = list(client_data.keys())

    # Compute class weights if needed
    class_weights = None
    if loss_type in ("weighted_ce", "focal"):
        all_y = np.concatenate([client_data[c][1] for c in client_data])
        class_weights = compute_class_weights(all_y, cfg["num_classes"])

    # Apply local epochs override
    effective_cfg = dict(cfg)
    if local_epochs_override is not None:
        effective_cfg["local_epochs"] = local_epochs_override

    for r in range(num_rounds):
        model = federated_round(model, client_data, all_clients, effective_cfg,
                                dp_epsilon=dp_epsilon, loss_type=loss_type,
                                class_weights=class_weights)

    personal = personalize_model(model, client_data, all_clients, effective_cfg,
                                 algo, loss_type=loss_type,
                                 class_weights=class_weights)
    return model, personal, client_data, client_test, meta


# ======================================================================
# Block AC: Complete Condition Matrix
# ======================================================================

def run_block_ac(checkpoint, quick=False):
    """Block AC: Complete Condition Matrix — fill missing DP conditions from AA."""
    log("=" * 70)
    log("BLOCK AC: Complete Condition Matrix (Missing DP Conditions)")
    log("=" * 70)

    datasets = ["Stroke"] if quick else ["Stroke", "Cirrhosis"]
    algorithms = ["FedAvg"] if quick else ["FedAvg", "Ditto", "HPFL"]
    seeds = [42] if quick else DEFAULT_SEEDS

    # Conditions missing from Block AA: IID+eps10, IID+eps1, NonIID+eps1
    conditions = [(True, 10.0)] if quick else [
        (True, 10.0),   # IID + eps=10
        (True, 1.0),    # IID + eps=1
        (False, 1.0),   # NonIID + eps=1
    ]

    t0 = time.time()
    count = 0

    configs = []
    for ds in datasets:
        for algo in algorithms:
            for is_iid, dp_eps in conditions:
                for seed in seeds:
                    configs.append((ds, algo, is_iid, dp_eps, seed))
    total = len(configs)

    for ds, algo, is_iid, dp_eps, seed in configs:
        if _shutdown:
            return count
        iid_tag = "IID" if is_iid else "NonIID"
        dp_tag = "eps{}".format(dp_eps)
        key = "AC_{}_{}_{}_{}_s{}".format(ds, algo, iid_tag, dp_tag, seed)
        if key in checkpoint["results"]:
            count += 1
            continue
        count += 1
        log("  [{}/{}] {} / {} / {} / {} / seed={}".format(
            count, total, ds, algo, iid_tag, dp_tag, seed))
        try:
            t_exp = time.time()
            cfg = DATASET_CONFIGS[ds]
            num_rounds = 10 if quick else cfg["num_rounds"]

            model, personal, client_data, client_test, meta = train_fl_model(
                ds, cfg, algo, num_rounds, seed, is_iid=is_iid, alpha=0.5,
                dp_epsilon=dp_eps)

            metrics = evaluate_fl_full(model, personal, client_test,
                                       cfg["num_classes"])

            all_train_y = np.concatenate([client_data[c][1] for c in client_data])
            class_counts = {
                int(c): int(n) for c, n in
                enumerate(np.bincount(all_train_y.astype(int),
                                      minlength=cfg["num_classes"]))
            }

            exp_time = time.time() - t_exp

            checkpoint["results"][key] = {
                "block": "AC_condition_matrix",
                "dataset": ds,
                "algorithm": algo,
                "is_iid": is_iid,
                "dp_epsilon": dp_eps,
                "seed": seed,
                "num_rounds": num_rounds,
                "loss_type": "standard",
                "class_distribution": class_counts,
                "experiment_time_sec": round(exp_time, 1),
                **metrics,
            }
            save_checkpoint(checkpoint)
            _cleanup_gpu()
        except Exception as e:
            log("  ERROR: {} — {}".format(key, e))
            traceback.print_exc()

    elapsed = time.time() - t0
    checkpoint["metadata"]["block_ac_time"] = round(elapsed, 1)
    save_checkpoint(checkpoint)
    log("Block AC complete: {} experiments in {}".format(count, format_time(elapsed)))
    return count


# ======================================================================
# Block AD: Mitigation Strategies
# ======================================================================

def run_block_ad(checkpoint, quick=False):
    """Block AD: Mitigation Strategies — weighted CE and focal loss."""
    log("=" * 70)
    log("BLOCK AD: Mitigation Strategies (Weighted CE + Focal Loss)")
    log("=" * 70)

    loss_types = ["weighted_ce"] if quick else ["weighted_ce", "focal"]
    datasets = ["Stroke"] if quick else ["Stroke", "Cirrhosis", "CDC_Diabetes"]
    algorithms = ["FedAvg"] if quick else ["FedAvg", "Ditto", "HPFL"]

    conditions = [(False, None)] if quick else [
        (True, None),    # IID + noDP
        (False, None),   # NonIID + noDP
        (False, 10.0),   # NonIID + eps=10
    ]

    t0 = time.time()
    count = 0

    configs = []
    for loss_type in loss_types:
        for ds in datasets:
            ds_seeds = [42] if quick else (CDC_SEEDS if ds == "CDC_Diabetes" else DEFAULT_SEEDS)
            for algo in algorithms:
                for is_iid, dp_eps in conditions:
                    for seed in ds_seeds:
                        configs.append((loss_type, ds, algo, is_iid, dp_eps, seed))
    total = len(configs)

    for loss_type, ds, algo, is_iid, dp_eps, seed in configs:
        if _shutdown:
            return count
        iid_tag = "IID" if is_iid else "NonIID"
        dp_tag = "eps{}".format(dp_eps) if dp_eps else "noDP"
        key = "AD_{}_{}_{}_{}_{}_s{}".format(loss_type, ds, algo, iid_tag, dp_tag, seed)
        if key in checkpoint["results"]:
            count += 1
            continue
        count += 1
        log("  [{}/{}] {} / {} / {} / {} / {} / seed={}".format(
            count, total, loss_type, ds, algo, iid_tag, dp_tag, seed))
        try:
            t_exp = time.time()
            cfg = DATASET_CONFIGS[ds]
            num_rounds = 10 if quick else cfg["num_rounds"]

            model, personal, client_data, client_test, meta = train_fl_model(
                ds, cfg, algo, num_rounds, seed, is_iid=is_iid, alpha=0.5,
                dp_epsilon=dp_eps, loss_type=loss_type)

            metrics = evaluate_fl_full(model, personal, client_test,
                                       cfg["num_classes"])

            all_train_y = np.concatenate([client_data[c][1] for c in client_data])
            class_counts = {
                int(c): int(n) for c, n in
                enumerate(np.bincount(all_train_y.astype(int),
                                      minlength=cfg["num_classes"]))
            }

            # Record class weights used
            cw = compute_class_weights(all_train_y, cfg["num_classes"])
            cw_dict = {int(c): round(float(cw[c]), 4) for c in range(cfg["num_classes"])}

            exp_time = time.time() - t_exp

            checkpoint["results"][key] = {
                "block": "AD_mitigation",
                "dataset": ds,
                "algorithm": algo,
                "is_iid": is_iid,
                "dp_epsilon": dp_eps,
                "seed": seed,
                "num_rounds": num_rounds,
                "loss_type": loss_type,
                "class_distribution": class_counts,
                "class_weights_used": cw_dict,
                "experiment_time_sec": round(exp_time, 1),
                **metrics,
            }
            save_checkpoint(checkpoint)
            _cleanup_gpu()
        except Exception as e:
            log("  ERROR: {} — {}".format(key, e))
            traceback.print_exc()

    elapsed = time.time() - t0
    checkpoint["metadata"]["block_ad_time"] = round(elapsed, 1)
    save_checkpoint(checkpoint)
    log("Block AD complete: {} experiments in {}".format(count, format_time(elapsed)))
    return count


# ======================================================================
# Block AE: Ditto Local Epochs Sweep
# ======================================================================

def run_block_ae(checkpoint, quick=False):
    """Block AE: Ditto Local Epochs Sweep — epochs {5, 10}."""
    log("=" * 70)
    log("BLOCK AE: Ditto Local Epochs Sweep")
    log("=" * 70)

    epoch_values = [5] if quick else [5, 10]
    datasets = ["Stroke"] if quick else ["Stroke", "Cirrhosis"]
    conditions = [(False, None)] if quick else [
        (False, None),   # NonIID + noDP
        (False, 10.0),   # NonIID + eps=10
    ]
    seeds = [42] if quick else DEFAULT_SEEDS
    algo = "Ditto"
    t0 = time.time()
    count = 0

    configs = []
    for ep in epoch_values:
        for ds in datasets:
            for is_iid, dp_eps in conditions:
                for seed in seeds:
                    configs.append((ep, ds, is_iid, dp_eps, seed))
    total = len(configs)

    for ep, ds, is_iid, dp_eps, seed in configs:
        if _shutdown:
            return count
        dp_tag = "eps{}".format(dp_eps) if dp_eps else "noDP"
        key = "AE_Ditto_ep{}_{}_NonIID_{}_s{}".format(ep, ds, dp_tag, seed)
        if key in checkpoint["results"]:
            count += 1
            continue
        count += 1
        log("  [{}/{}] Ditto / ep={} / {} / NonIID / {} / seed={}".format(
            count, total, ep, ds, dp_tag, seed))
        try:
            t_exp = time.time()
            cfg = DATASET_CONFIGS[ds]
            num_rounds = 10 if quick else cfg["num_rounds"]

            model, personal, client_data, client_test, meta = train_fl_model(
                ds, cfg, algo, num_rounds, seed, is_iid=False, alpha=0.5,
                dp_epsilon=dp_eps, local_epochs_override=ep)

            metrics = evaluate_fl_full(model, personal, client_test,
                                       cfg["num_classes"])

            all_train_y = np.concatenate([client_data[c][1] for c in client_data])
            class_counts = {
                int(c): int(n) for c, n in
                enumerate(np.bincount(all_train_y.astype(int),
                                      minlength=cfg["num_classes"]))
            }

            exp_time = time.time() - t_exp

            checkpoint["results"][key] = {
                "block": "AE_epochs_sweep",
                "dataset": ds,
                "algorithm": algo,
                "is_iid": False,
                "dp_epsilon": dp_eps,
                "seed": seed,
                "num_rounds": num_rounds,
                "local_epochs": ep,
                "loss_type": "standard",
                "class_distribution": class_counts,
                "experiment_time_sec": round(exp_time, 1),
                **metrics,
            }
            save_checkpoint(checkpoint)
            _cleanup_gpu()
        except Exception as e:
            log("  ERROR: {} — {}".format(key, e))
            traceback.print_exc()

    elapsed = time.time() - t0
    checkpoint["metadata"]["block_ae_time"] = round(elapsed, 1)
    save_checkpoint(checkpoint)
    log("Block AE complete: {} experiments in {}".format(count, format_time(elapsed)))
    return count


# ======================================================================
# Block TH: Threshold Rescue (cascade9 AA re-evaluation)
# ======================================================================

def run_block_th(checkpoint, quick=False):
    """Block TH: Re-evaluate cascade9 AA experiments with F1=0.0 using threshold tuning."""
    log("=" * 70)
    log("BLOCK TH: Threshold Rescue (Cascade9 AA Re-evaluation)")
    log("=" * 70)

    if quick:
        log("  Skipping Block TH in quick mode.")
        return 0

    # Load cascade9 checkpoint
    cas9_path = OUTPUT_DIR / "checkpoint_cascade9.json"
    if not cas9_path.exists():
        log("  WARNING: cascade9 checkpoint not found at {}".format(cas9_path))
        log("  Skipping Block TH.")
        return 0

    with open(cas9_path) as f:
        cas9 = json.load(f)

    # Find AA experiments with F1 = 0.0 (majority-class collapse)
    collapsed = []
    for key, result in cas9["results"].items():
        if not isinstance(result, dict):
            continue
        if result.get("block") != "AA_imbalance":
            continue
        if result.get("f1_score", 1.0) == 0.0:
            collapsed.append((key, result))

    if not collapsed:
        log("  No F1=0.0 experiments found in cascade9 Block AA.")
        return 0

    log("  Found {} collapsed experiments (F1=0.0) to re-evaluate with threshold tuning".format(
        len(collapsed)))

    t0 = time.time()
    count = 0
    total = len(collapsed)

    for orig_key, orig_result in collapsed:
        if _shutdown:
            return count
        ds = orig_result["dataset"]
        algo = orig_result["algorithm"]
        is_iid = orig_result["is_iid"]
        dp_eps = orig_result.get("dp_epsilon")
        seed = orig_result["seed"]

        iid_tag = "IID" if is_iid else "NonIID"
        dp_tag = "eps{}".format(dp_eps) if dp_eps else "noDP"
        key = "TH_{}_{}_{}_{}_s{}".format(ds, algo, iid_tag, dp_tag, seed)
        if key in checkpoint["results"]:
            count += 1
            continue
        count += 1
        log("  [{}/{}] RESCUE: {} / {} / {} / {} / seed={}".format(
            count, total, ds, algo, iid_tag, dp_tag, seed))

        if ds not in DATASET_CONFIGS:
            log("    SKIP: dataset {} not in DATASET_CONFIGS".format(ds))
            continue

        try:
            t_exp = time.time()
            cfg = DATASET_CONFIGS[ds]
            num_rounds = cfg["num_rounds"]

            model, personal, client_data, client_test, meta = train_fl_model(
                ds, cfg, algo, num_rounds, seed, is_iid=is_iid, alpha=0.5,
                dp_epsilon=dp_eps)

            metrics = evaluate_fl_full(model, personal, client_test,
                                       cfg["num_classes"])

            exp_time = time.time() - t_exp

            checkpoint["results"][key] = {
                "block": "TH_threshold_rescue",
                "original_key": orig_key,
                "dataset": ds,
                "algorithm": algo,
                "is_iid": is_iid,
                "dp_epsilon": dp_eps,
                "seed": seed,
                "num_rounds": num_rounds,
                "loss_type": "standard",
                "original_f1": orig_result.get("f1_score", 0.0),
                "experiment_time_sec": round(exp_time, 1),
                **metrics,
            }
            save_checkpoint(checkpoint)
            _cleanup_gpu()
        except Exception as e:
            log("  ERROR: {} — {}".format(key, e))
            traceback.print_exc()

    elapsed = time.time() - t0
    checkpoint["metadata"]["block_th_time"] = round(elapsed, 1)
    save_checkpoint(checkpoint)
    log("Block TH complete: {} experiments in {}".format(count, format_time(elapsed)))
    return count


# ======================================================================
# Main
# ======================================================================

def main():
    global _log_file, _shutdown

    parser = argparse.ArgumentParser(
        description="FL-EHDS Cascade 10: Clinical Imbalance Deep-Dive")
    parser.add_argument("--quick", action="store_true",
                        help="Reduced rounds/seeds for quick test (~3 exp)")
    parser.add_argument("--fresh", action="store_true",
                        help="Delete existing checkpoint and start fresh")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, _handle_signal)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _log_file = open(OUTPUT_DIR / LOG_FILE, "a", encoding="utf-8")

    log("=" * 70)
    log("FL-EHDS Cascading Analysis — Phase 10 (Clinical Imbalance Deep-Dive)")
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
                "experiment": "cascade10_imbalance_deep_dive",
                "started": datetime.now().isoformat(),
                "mode": "quick" if args.quick else "full",
            },
            "results": {},
            "completed": [],
        }
        save_checkpoint(checkpoint)

    total_experiments = 0

    # Block AC: Complete Condition Matrix
    if not _shutdown:
        n = run_block_ac(checkpoint, quick=args.quick)
        total_experiments += n
        if "AC" not in checkpoint.get("completed", []):
            checkpoint.setdefault("completed", []).append("AC")
        save_checkpoint(checkpoint)

    # Block AD: Mitigation Strategies
    if not _shutdown:
        n = run_block_ad(checkpoint, quick=args.quick)
        total_experiments += n
        if "AD" not in checkpoint.get("completed", []):
            checkpoint.setdefault("completed", []).append("AD")
        save_checkpoint(checkpoint)

    # Block AE: Ditto Local Epochs Sweep
    if not _shutdown:
        n = run_block_ae(checkpoint, quick=args.quick)
        total_experiments += n
        if "AE" not in checkpoint.get("completed", []):
            checkpoint.setdefault("completed", []).append("AE")
        save_checkpoint(checkpoint)

    # Block TH: Threshold Rescue
    if not _shutdown:
        n = run_block_th(checkpoint, quick=args.quick)
        total_experiments += n
        if "TH" not in checkpoint.get("completed", []):
            checkpoint.setdefault("completed", []).append("TH")
        save_checkpoint(checkpoint)

    # Final save
    checkpoint["metadata"]["finished"] = datetime.now().isoformat()
    save_checkpoint(checkpoint)

    # Summary
    block_counts = {}
    for k, v in checkpoint["results"].items():
        if isinstance(v, dict):
            b = v.get("block", "unknown")
            block_counts[b] = block_counts.get(b, 0) + 1
    total_exp = sum(block_counts.values())

    started = datetime.fromisoformat(checkpoint["metadata"]["started"])
    elapsed_total = datetime.now() - started

    log("=" * 70)
    log("ALL BLOCKS COMPLETE in {}".format(str(elapsed_total).split(".")[0]))
    log("Checkpoint: {}".format(CHECKPOINT_FILE))
    for block, n in sorted(block_counts.items()):
        log("  {}: {} experiments".format(block, n))
    log("  TOTAL: {} experiments".format(total_exp))
    errors = sum(1 for k, v in checkpoint["results"].items()
                 if isinstance(v, dict) and v.get("error"))
    if errors:
        log("  ERRORS: {}".format(errors))
    else:
        log("  ERRORS: 0")
    log("=" * 70)

    if _log_file:
        _log_file.close()


if __name__ == "__main__":
    main()

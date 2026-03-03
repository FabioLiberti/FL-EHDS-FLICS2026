#!/usr/bin/env python3
"""
FL-EHDS Cascading Analysis & Experiments — Phase 9.

5 blocks executed sequentially (Deployment Readiness & Cross-Cutting):

Block X: Extended Scalability (24 exp)
  CDC Diabetes K={10,20,50} + Cardiovascular K=50 × 3 algorithms × 2 seeds.
  Tests FL performance at realistic EHDS multi-hospital scales.

Block Y: Combined DP + Compression (24 exp)
  Cardiovascular × {SignSGD, QSGD} × {eps=10, eps=1} × 3 algorithms × 2 seeds.
  Real deployment scenario: privacy AND communication efficiency together.

Block Z: Convergence Dynamics (18 exp)
  {Cardiovascular, PTB-XL, CDC Diabetes} × 3 algorithms × {IID, NonIID}.
  Per-round accuracy tracking for convergence speed comparison.

Block AA: Clinical Imbalance Robustness (54 exp)
  {Stroke, CKD, Cirrhosis} × 3 algorithms × 2 IID × 2 seeds + DP variants.
  F1/precision/recall/DEI on extreme class imbalance (Stroke 4.9% positive).

Block AB: CDC Diabetes NonIID Depth (24 exp)
  CDC Diabetes × 3 algorithms × alpha={0.1, 0.25, 0.75, 1.0} × 2 seeds.
  Validates Ditto minority-class rescue across heterogeneity levels.

Usage:
    cd fl-ehds-framework
    python -m benchmarks.run_analysis_cascade9 [--quick] [--fresh]

Output:
    benchmarks/paper_results_tabular/checkpoint_cascade9.json

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
CHECKPOINT_FILE = "checkpoint_cascade9.json"
LOG_FILE = "experiment_cascade9.log"

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
    "CDC_Diabetes": dict(
        learning_rate=0.01, batch_size=128, num_rounds=20, local_epochs=3,
        mu=0.1, num_clients=5, input_dim=21, num_classes=2,
    ),
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
}

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
    fd, tmp = tempfile.mkstemp(dir=str(OUTPUT_DIR), prefix=".cas9_", suffix=".tmp")
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
        from data.cardiovascular_loader import load_cardiovascular_data
        client_data, client_test, meta = load_cardiovascular_data(
            num_clients=num_clients, seed=seed, is_iid=is_iid, alpha=alpha,
        )
    elif dataset == "PTB_XL":
        from data.ptbxl_loader import load_ptbxl_data
        client_data, client_test, meta = load_ptbxl_data(
            num_clients=num_clients, seed=seed,
            partition_by_site=False if not is_iid else True,
            is_iid=is_iid, alpha=alpha,
            min_site_samples=50,
        )
    elif dataset == "Breast_Cancer":
        from data.breast_cancer_loader import load_breast_cancer_data
        client_data, client_test, meta = load_breast_cancer_data(
            num_clients=num_clients, seed=seed, is_iid=is_iid, alpha=alpha,
        )
    elif dataset == "CDC_Diabetes":
        from data.cdc_diabetes_loader import load_cdc_diabetes_data
        client_data, client_test, meta = load_cdc_diabetes_data(
            num_clients=num_clients, seed=seed, is_iid=is_iid, alpha=alpha,
        )
    elif dataset == "Stroke":
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


def get_predictions(model, X):
    """Get model predictions as numpy array."""
    model.eval()
    X_t = torch.FloatTensor(X).to(DEVICE) if isinstance(X, np.ndarray) else X.to(DEVICE)
    with torch.no_grad():
        preds = model(X_t).argmax(dim=1).cpu().numpy()
    return preds


# ======================================================================
# Metrics: F1, Precision, Recall, DEI
# ======================================================================

def compute_classification_metrics(y_true, y_pred, num_classes=2):
    """Compute accuracy, F1, precision, recall, DEI from arrays."""
    acc = float((y_pred == y_true).mean())

    # Per-class recall (sensitivity per class)
    per_class_recall = {}
    for c in range(num_classes):
        mask = y_true == c
        if mask.sum() > 0:
            per_class_recall[c] = float((y_pred[mask] == c).mean())
        else:
            per_class_recall[c] = 0.0

    # Binary F1 (positive class = 1)
    if num_classes == 2:
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-10)
    else:
        # Macro average for multiclass
        f1_scores = []
        prec_scores = []
        rec_scores = []
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

    # DEI: Diagnostic Equity Index = min_c(R_c) * (1 - CV(R_c))
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


# ======================================================================
# FL Training (FedAvg, Ditto, HPFL)
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


def personalize_model(model, client_data, all_clients, cfg, algo):
    """Create personal models for Ditto or HPFL after global training."""
    if algo == "Ditto":
        personal = {}
        for cid in all_clients:
            pm = create_model(cfg["input_dim"], cfg["num_classes"])
            set_params(pm, get_params(model))
            X, y = client_data[cid]
            train_local_sgd(pm, X, y, cfg["local_epochs"],
                            cfg["learning_rate"], cfg["batch_size"])
            personal[cid] = pm
        return personal
    elif algo == "HPFL":
        personal = {}
        for cid in all_clients:
            pm = create_model(cfg["input_dim"], cfg["num_classes"])
            set_params(pm, get_params(model))
            X, y = client_data[cid]
            for n, p in pm.named_parameters():
                p.requires_grad = ("fc2" in n)
            opt = optim.SGD(
                filter(lambda p: p.requires_grad, pm.parameters()),
                lr=cfg["learning_rate"])
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
        return personal
    return None


def evaluate_fl_full(model, personal, client_test, num_classes):
    """Evaluate FL model (with personal models if available) and return full metrics."""
    all_preds, all_labels = [], []
    per_client_acc = {}
    for cid in client_test:
        Xc, yc = client_test[cid]
        cm = personal[cid] if personal and cid in personal else model
        preds_c = get_predictions(cm, Xc)
        all_preds.append(preds_c)
        all_labels.append(yc)
        per_client_acc[cid] = round(float((preds_c == yc).mean()), 4)
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    metrics = compute_classification_metrics(all_labels, all_preds, num_classes)
    metrics["per_client_accuracy"] = per_client_acc
    return metrics


def train_fl_model(dataset, cfg, algo, num_rounds, seed, is_iid=True, alpha=0.5,
                   dp_epsilon=None, num_clients_override=None):
    """Train FL model. Returns (model, personal, client_data, client_test, meta)."""
    num_clients = num_clients_override if num_clients_override else cfg["num_clients"]
    client_data, client_test, meta = load_dataset(dataset, num_clients, seed, is_iid, alpha)
    model = create_model(cfg["input_dim"], cfg["num_classes"], seed=seed)
    all_clients = list(client_data.keys())

    for r in range(num_rounds):
        model = federated_round(model, client_data, all_clients, cfg, dp_epsilon=dp_epsilon)

    personal = personalize_model(model, client_data, all_clients, cfg, algo)
    return model, personal, client_data, client_test, meta


# ======================================================================
# FL Round with compression + DP (Block Y)
# ======================================================================

def federated_round_compressed_dp(model, client_data, all_clients, cfg,
                                  compressor_mgr, dp_epsilon, delta_dp=1e-5):
    """One FL round with local DP noise THEN compression."""
    global_params = get_params(model)
    all_decompressed = []
    samples = []
    round_ratios = []

    sigma = (np.sqrt(2.0 * np.log(1.25 / delta_dp)) / dp_epsilon
             if dp_epsilon < float("inf") else 0.0)

    for cid in all_clients:
        X, y = client_data[cid]
        lm = create_model(cfg["input_dim"], cfg["num_classes"])
        set_params(lm, global_params)
        train_local_sgd(lm, X, y, cfg["local_epochs"],
                        cfg["learning_rate"], cfg["batch_size"])
        lp = get_params(lm)
        delta = {n: (lp[n] - global_params[n]).cpu().numpy() for n in global_params}

        # Local DP noise BEFORE compression
        if sigma > 0:
            for n in delta:
                delta[n] = delta[n] + np.random.randn(*delta[n].shape).astype(np.float32) * sigma

        # Compress noisy delta
        compressed = compressor_mgr.compress(delta, client_id=cid)
        round_ratios.append(compressed.compression_ratio)

        # Decompress (server side)
        decompressed = compressor_mgr.decompress(compressed)
        all_decompressed.append(decompressed)
        samples.append(len(y))

    # Weighted average of decompressed deltas
    total_s = sum(samples)
    avg_d = {}
    for n in global_params:
        avg_d[n] = sum(
            torch.FloatTensor(all_decompressed[i][n]) * (samples[i] / total_s)
            for i in range(len(all_decompressed))
        )
    new_p = {n: global_params[n] + avg_d[n].to(DEVICE) for n in global_params}
    set_params(model, new_p)
    avg_ratio = float(np.mean(round_ratios)) if round_ratios else 1.0
    return model, avg_ratio


# ======================================================================
# Block X: Extended Scalability
# ======================================================================

def run_block_x(checkpoint, quick=False):
    """Block X: Extended Scalability (K=10/20/50 on CDC, K=50 on Cardiovascular)."""
    log("=" * 70)
    log("BLOCK X: Extended Scalability")
    log("=" * 70)

    algorithms = ["FedAvg", "Ditto", "HPFL"]
    seeds = [42] if quick else DEFAULT_SEEDS
    t0 = time.time()
    count = 0

    # CDC Diabetes at different K values
    cdc_k_values = [10] if quick else [10, 20, 50]
    # Cardiovascular at K=50
    cv_k_values = [] if quick else [50]

    configs = []
    for k in cdc_k_values:
        for algo in (["FedAvg"] if quick else algorithms):
            for seed in seeds:
                configs.append(("CDC_Diabetes", k, algo, seed))
    for k in cv_k_values:
        for algo in algorithms:
            for seed in seeds:
                configs.append(("Cardiovascular", k, algo, seed))

    total = len(configs)

    for ds, k, algo, seed in configs:
        if _shutdown:
            return count
        key = "X_{}_K{}_{}_s{}".format(ds, k, algo, seed)
        if key in checkpoint["results"]:
            count += 1
            continue
        count += 1
        log("  [{}/{}] {} / K={} / {} / seed={}".format(count, total, ds, k, algo, seed))
        try:
            t_exp = time.time()
            cfg = DATASET_CONFIGS[ds]
            num_rounds = 5 if quick else cfg["num_rounds"]

            model, personal, client_data, client_test, meta = train_fl_model(
                ds, cfg, algo, num_rounds, seed, is_iid=False, alpha=0.5,
                num_clients_override=k)

            metrics = evaluate_fl_full(model, personal, client_test,
                                       cfg["num_classes"])

            samples_per_client = {
                cid: len(client_data[cid][1]) for cid in client_data
            }
            exp_time = time.time() - t_exp

            checkpoint["results"][key] = {
                "block": "X_scalability",
                "dataset": ds,
                "num_clients": k,
                "algorithm": algo,
                "seed": seed,
                "num_rounds": num_rounds,
                "total_train_samples": sum(samples_per_client.values()),
                "samples_per_client": samples_per_client,
                "experiment_time_sec": round(exp_time, 1),
                **metrics,
            }
            save_checkpoint(checkpoint)
            _cleanup_gpu()
        except Exception as e:
            log("  ERROR: {} — {}".format(key, e))
            traceback.print_exc()

    elapsed = time.time() - t0
    checkpoint["metadata"]["block_x_time"] = round(elapsed, 1)
    save_checkpoint(checkpoint)
    log("Block X complete: {} experiments in {}".format(count, format_time(elapsed)))
    return count


# ======================================================================
# Block Y: Combined DP + Compression
# ======================================================================

def run_block_y(checkpoint, quick=False):
    """Block Y: Combined DP + Compression (SignSGD/QSGD + DP)."""
    log("=" * 70)
    log("BLOCK Y: Combined DP + Compression")
    log("=" * 70)

    from core.model_compression import CompressionConfig, CompressionManager

    compression_methods = ["signsgd"] if quick else ["signsgd", "qsgd"]
    dp_epsilons = [10.0] if quick else [10.0, 1.0]
    algorithms = ["FedAvg"] if quick else ["FedAvg", "Ditto", "HPFL"]
    seeds = [42] if quick else DEFAULT_SEEDS
    ds = "Cardiovascular"
    cfg = DATASET_CONFIGS[ds]
    num_rounds = 5 if quick else 15
    t0 = time.time()
    count = 0

    configs = []
    for comp in compression_methods:
        for eps in dp_epsilons:
            for algo in algorithms:
                for seed in seeds:
                    configs.append((comp, eps, algo, seed))
    total = len(configs)

    for comp, eps, algo, seed in configs:
        if _shutdown:
            return count
        key = "Y_{}_eps{}_{}_s{}".format(comp, eps, algo, seed)
        if key in checkpoint["results"]:
            count += 1
            continue
        count += 1
        log("  [{}/{}] {} + eps={} / {} / seed={}".format(
            count, total, comp, eps, algo, seed))
        try:
            t_exp = time.time()

            comp_cfg = CompressionConfig(
                method=comp,
                num_bits=4,
                k_ratio=0.1,
                threshold=0.5,
                rank=4,
                use_error_feedback=True,
            )
            comp_mgr = CompressionManager(comp_cfg)

            client_data, client_test, meta = load_dataset(
                ds, cfg["num_clients"], seed, is_iid=True)
            model = create_model(cfg["input_dim"], cfg["num_classes"], seed=seed)
            all_clients = list(client_data.keys())

            round_ratios = []
            for r in range(num_rounds):
                model, avg_ratio = federated_round_compressed_dp(
                    model, client_data, all_clients, cfg,
                    comp_mgr, dp_epsilon=eps)
                round_ratios.append(avg_ratio)

            # Personalize
            personal = personalize_model(model, client_data, all_clients, cfg, algo)

            metrics = evaluate_fl_full(model, personal, client_test,
                                       cfg["num_classes"])

            stats = comp_mgr.get_stats()
            exp_time = time.time() - t_exp

            checkpoint["results"][key] = {
                "block": "Y_dp_compression",
                "dataset": ds,
                "compression_method": comp,
                "dp_epsilon": eps,
                "algorithm": algo,
                "seed": seed,
                "num_rounds": num_rounds,
                "avg_compression_ratio": round(stats["average_compression_ratio"], 2),
                "bandwidth_saved_pct": round(stats["bandwidth_saved_pct"], 1),
                "per_round_ratios": [round(r, 2) for r in round_ratios],
                "experiment_time_sec": round(exp_time, 1),
                **metrics,
            }
            save_checkpoint(checkpoint)
            _cleanup_gpu()
        except Exception as e:
            log("  ERROR: {} — {}".format(key, e))
            traceback.print_exc()

    elapsed = time.time() - t0
    checkpoint["metadata"]["block_y_time"] = round(elapsed, 1)
    save_checkpoint(checkpoint)
    log("Block Y complete: {} experiments in {}".format(count, format_time(elapsed)))
    return count


# ======================================================================
# Block Z: Convergence Dynamics
# ======================================================================

def run_block_z(checkpoint, quick=False):
    """Block Z: Convergence Dynamics (per-round accuracy tracking)."""
    log("=" * 70)
    log("BLOCK Z: Convergence Dynamics")
    log("=" * 70)

    datasets = ["Cardiovascular"] if quick else ["Cardiovascular", "PTB_XL", "CDC_Diabetes"]
    algorithms = ["FedAvg"] if quick else ["FedAvg", "Ditto", "HPFL"]
    iid_modes = [False] if quick else [True, False]
    seed = 42
    t0 = time.time()
    count = 0

    configs = []
    for ds in datasets:
        for algo in algorithms:
            for is_iid in iid_modes:
                configs.append((ds, algo, is_iid))
    total = len(configs)

    for ds, algo, is_iid in configs:
        if _shutdown:
            return count
        iid_tag = "IID" if is_iid else "NonIID"
        key = "Z_{}_{}_{}".format(ds, algo, iid_tag)
        if key in checkpoint["results"]:
            count += 1
            continue
        count += 1
        log("  [{}/{}] {} / {} / {}".format(count, total, ds, algo, iid_tag))
        try:
            t_exp = time.time()
            cfg = DATASET_CONFIGS[ds]
            num_rounds = 10 if quick else cfg["num_rounds"]

            client_data, client_test, meta = load_dataset(
                ds, cfg["num_clients"], seed, is_iid=is_iid, alpha=0.5)
            model = create_model(cfg["input_dim"], cfg["num_classes"], seed=seed)
            all_clients = list(client_data.keys())

            # Pool test data for per-round evaluation
            all_X = np.concatenate([client_test[c][0] for c in client_test])
            all_y = np.concatenate([client_test[c][1] for c in client_test])

            # Track accuracy at each round
            round_accuracy = []
            for r in range(num_rounds):
                model = federated_round(model, client_data, all_clients, cfg)
                acc_r = evaluate_model(model, all_X, all_y)
                round_accuracy.append(round(float(acc_r), 4))

            # Final global accuracy
            final_global_acc = round_accuracy[-1]

            # Personal model accuracy (Ditto/HPFL)
            personal = personalize_model(model, client_data, all_clients, cfg, algo)
            metrics = evaluate_fl_full(model, personal, client_test,
                                       cfg["num_classes"])

            # Compute rounds to 90% and 95% of final personal accuracy
            final_acc = metrics["accuracy"]
            rounds_to_90 = None
            rounds_to_95 = None
            target_90 = final_global_acc * 0.90
            target_95 = final_global_acc * 0.95
            for r_idx, r_acc in enumerate(round_accuracy):
                if rounds_to_90 is None and r_acc >= target_90:
                    rounds_to_90 = r_idx + 1
                if rounds_to_95 is None and r_acc >= target_95:
                    rounds_to_95 = r_idx + 1

            exp_time = time.time() - t_exp

            checkpoint["results"][key] = {
                "block": "Z_convergence",
                "dataset": ds,
                "algorithm": algo,
                "is_iid": is_iid,
                "seed": seed,
                "num_rounds": num_rounds,
                "round_accuracy": round_accuracy,
                "final_global_accuracy": final_global_acc,
                "final_personal_accuracy": metrics["accuracy"],
                "personalization_boost": round(metrics["accuracy"] - final_global_acc, 4),
                "rounds_to_90pct": rounds_to_90,
                "rounds_to_95pct": rounds_to_95,
                "experiment_time_sec": round(exp_time, 1),
            }
            save_checkpoint(checkpoint)
            _cleanup_gpu()
        except Exception as e:
            log("  ERROR: {} — {}".format(key, e))
            traceback.print_exc()

    elapsed = time.time() - t0
    checkpoint["metadata"]["block_z_time"] = round(elapsed, 1)
    save_checkpoint(checkpoint)
    log("Block Z complete: {} experiments in {}".format(count, format_time(elapsed)))
    return count


# ======================================================================
# Block AA: Clinical Imbalance Robustness
# ======================================================================

def run_block_aa(checkpoint, quick=False):
    """Block AA: Clinical Imbalance Robustness (Stroke, CKD, Cirrhosis)."""
    log("=" * 70)
    log("BLOCK AA: Clinical Imbalance Robustness")
    log("=" * 70)

    datasets_aa = ["Stroke"] if quick else ["Stroke", "CKD", "Cirrhosis"]
    algorithms = ["FedAvg"] if quick else ["FedAvg", "Ditto", "HPFL"]
    iid_modes = [False] if quick else [True, False]
    seeds = [42] if quick else DEFAULT_SEEDS
    t0 = time.time()
    count = 0

    # Base experiments (no DP)
    configs_base = []
    for ds in datasets_aa:
        for algo in algorithms:
            for is_iid in iid_modes:
                for seed in seeds:
                    configs_base.append((ds, algo, is_iid, None, seed))

    # DP experiments (NonIID only, eps=10)
    configs_dp = []
    if not quick:
        for ds in datasets_aa:
            for algo in algorithms:
                for seed in seeds:
                    configs_dp.append((ds, algo, False, 10.0, seed))

    all_configs = configs_base + configs_dp
    total = len(all_configs)

    for ds, algo, is_iid, dp_eps, seed in all_configs:
        if _shutdown:
            return count
        iid_tag = "IID" if is_iid else "NonIID"
        dp_tag = "eps{}".format(dp_eps) if dp_eps else "noDP"
        key = "AA_{}_{}_{}_{}_s{}".format(ds, algo, iid_tag, dp_tag, seed)
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

            samples_per_client = {
                cid: len(client_data[cid][1]) for cid in client_data
            }

            # Class distribution info
            all_train_y = np.concatenate([client_data[c][1] for c in client_data])
            class_counts = {
                int(c): int(n) for c, n in
                enumerate(np.bincount(all_train_y.astype(int),
                                      minlength=cfg["num_classes"]))
            }

            exp_time = time.time() - t_exp

            checkpoint["results"][key] = {
                "block": "AA_imbalance",
                "dataset": ds,
                "algorithm": algo,
                "is_iid": is_iid,
                "dp_epsilon": dp_eps,
                "seed": seed,
                "num_rounds": num_rounds,
                "total_train_samples": int(len(all_train_y)),
                "class_distribution": class_counts,
                "samples_per_client": samples_per_client,
                "experiment_time_sec": round(exp_time, 1),
                **metrics,
            }
            save_checkpoint(checkpoint)
            _cleanup_gpu()
        except Exception as e:
            log("  ERROR: {} — {}".format(key, e))
            traceback.print_exc()

    elapsed = time.time() - t0
    checkpoint["metadata"]["block_aa_time"] = round(elapsed, 1)
    save_checkpoint(checkpoint)
    log("Block AA complete: {} experiments in {}".format(count, format_time(elapsed)))
    return count


# ======================================================================
# Block AB: CDC Diabetes NonIID Depth
# ======================================================================

def run_block_ab(checkpoint, quick=False):
    """Block AB: CDC Diabetes NonIID Depth (alpha sensitivity)."""
    log("=" * 70)
    log("BLOCK AB: CDC Diabetes NonIID Depth")
    log("=" * 70)

    # Check if CDC dataset exists
    data_dir = FRAMEWORK_DIR / "data" / "cdc_diabetes"
    csv_path = data_dir / "diabetes_binary_health_indicators_BRFSS2015.csv"
    if not csv_path.exists():
        log("  WARNING: CDC Diabetes dataset not found at {}".format(csv_path))
        log("  Skipping Block AB.")
        checkpoint["metadata"]["block_ab_skipped"] = True
        save_checkpoint(checkpoint)
        return 0

    algorithms = ["FedAvg"] if quick else ["FedAvg", "Ditto", "HPFL"]
    alpha_values = [0.1] if quick else [0.1, 0.25, 0.75, 1.0]
    seeds = [42] if quick else DEFAULT_SEEDS
    ds = "CDC_Diabetes"
    cfg = DATASET_CONFIGS[ds]
    num_rounds = 5 if quick else cfg["num_rounds"]
    t0 = time.time()
    count = 0

    configs = []
    for algo in algorithms:
        for alpha in alpha_values:
            for seed in seeds:
                configs.append((algo, alpha, seed))
    total = len(configs)

    for algo, alpha, seed in configs:
        if _shutdown:
            return count
        key = "AB_CDC_{}_a{}_s{}".format(algo, alpha, seed)
        if key in checkpoint["results"]:
            count += 1
            continue
        count += 1
        log("  [{}/{}] CDC_Diabetes / {} / alpha={} / seed={}".format(
            count, total, algo, alpha, seed))
        try:
            t_exp = time.time()

            model, personal, client_data, client_test, meta = train_fl_model(
                ds, cfg, algo, num_rounds, seed, is_iid=False, alpha=alpha)

            metrics = evaluate_fl_full(model, personal, client_test,
                                       cfg["num_classes"])

            samples_per_client = {
                cid: len(client_data[cid][1]) for cid in client_data
            }

            # Per-client class distribution to show heterogeneity
            per_client_pos_rate = {}
            for cid in client_data:
                y_c = client_data[cid][1]
                per_client_pos_rate[cid] = round(float(y_c.mean()), 4)

            exp_time = time.time() - t_exp

            checkpoint["results"][key] = {
                "block": "AB_noniid_depth",
                "dataset": ds,
                "algorithm": algo,
                "alpha": alpha,
                "seed": seed,
                "num_rounds": num_rounds,
                "total_train_samples": sum(len(client_data[c][1]) for c in client_data),
                "samples_per_client": samples_per_client,
                "per_client_positive_rate": per_client_pos_rate,
                "experiment_time_sec": round(exp_time, 1),
                **metrics,
            }
            save_checkpoint(checkpoint)
            _cleanup_gpu()
        except Exception as e:
            log("  ERROR: {} — {}".format(key, e))
            traceback.print_exc()

    elapsed = time.time() - t0
    checkpoint["metadata"]["block_ab_time"] = round(elapsed, 1)
    save_checkpoint(checkpoint)
    log("Block AB complete: {} experiments in {}".format(count, format_time(elapsed)))
    return count


# ======================================================================
# Main
# ======================================================================

def main():
    global _log_file, _shutdown

    parser = argparse.ArgumentParser(
        description="FL-EHDS Cascade 9: Deployment Readiness & Cross-Cutting Validation")
    parser.add_argument("--quick", action="store_true",
                        help="Reduced rounds/seeds for quick test (~5 exp)")
    parser.add_argument("--fresh", action="store_true",
                        help="Delete existing checkpoint and start fresh")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, _handle_signal)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _log_file = open(OUTPUT_DIR / LOG_FILE, "a", encoding="utf-8")

    log("=" * 70)
    log("FL-EHDS Cascading Analysis — Phase 9 (Deployment Readiness)")
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
                "experiment": "cascade9_deployment_readiness",
                "started": datetime.now().isoformat(),
                "mode": "quick" if args.quick else "full",
            },
            "results": {},
            "completed": [],
        }
        save_checkpoint(checkpoint)

    total_experiments = 0

    # Block X: Extended Scalability
    if not _shutdown:
        n = run_block_x(checkpoint, quick=args.quick)
        total_experiments += n
        if "X" not in checkpoint.get("completed", []):
            checkpoint.setdefault("completed", []).append("X")
        save_checkpoint(checkpoint)

    # Block Y: Combined DP + Compression
    if not _shutdown:
        n = run_block_y(checkpoint, quick=args.quick)
        total_experiments += n
        if "Y" not in checkpoint.get("completed", []):
            checkpoint.setdefault("completed", []).append("Y")
        save_checkpoint(checkpoint)

    # Block Z: Convergence Dynamics
    if not _shutdown:
        n = run_block_z(checkpoint, quick=args.quick)
        total_experiments += n
        if "Z" not in checkpoint.get("completed", []):
            checkpoint.setdefault("completed", []).append("Z")
        save_checkpoint(checkpoint)

    # Block AA: Clinical Imbalance Robustness
    if not _shutdown:
        n = run_block_aa(checkpoint, quick=args.quick)
        total_experiments += n
        if "AA" not in checkpoint.get("completed", []):
            checkpoint.setdefault("completed", []).append("AA")
        save_checkpoint(checkpoint)

    # Block AB: CDC Diabetes NonIID Depth
    if not _shutdown:
        n = run_block_ab(checkpoint, quick=args.quick)
        total_experiments += n
        if "AB" not in checkpoint.get("completed", []):
            checkpoint.setdefault("completed", []).append("AB")
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

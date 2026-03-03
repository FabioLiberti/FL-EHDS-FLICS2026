#!/usr/bin/env python3
"""
FL-EHDS Cascading Analysis & Experiments — Phase 8.

5 blocks executed sequentially (Communication, Privacy & Scalability):

Block S: Secure Aggregation Overhead (~30 exp)
  Cardiovascular × 3 SecAgg methods × 2 sizes × 2-3 seeds.
  Measures mask/share/encrypt overhead, aggregation error, accuracy.

Block T: Gradient Compression Benchmark (~42 exp)
  Cardiovascular × 7 compression methods × 3 seeds.
  Compression ratio, reconstruction error, FL accuracy impact.

Block U: Local DP vs Central DP (~36 exp)
  3 datasets × 2 DP modes × 3 epsilon levels × 2 seeds.
  Privacy-utility comparison of per-client vs server-side noise.

Block V: Vertical FL Simulation (~24 exp)
  Cardiovascular vertical split × 2-3 parties × 2 DP × 2 overlap × 3 seeds.
  SplitNN accuracy, PSI alignment stats, DP impact on vertical FL.

Block W: CDC Diabetes Scalability (~30 exp)
  CDC Diabetes (253K) × 3 algorithms × 2 IID × 2-3 seeds.
  Largest dataset test for scalability, class imbalance handling.

Usage:
    cd fl-ehds-framework
    python -m benchmarks.run_analysis_cascade8 [--quick] [--fresh]

Output:
    benchmarks/paper_results_tabular/checkpoint_cascade8.json

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
import io
from contextlib import redirect_stdout

import numpy as np

FRAMEWORK_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(FRAMEWORK_DIR))

import torch
import torch.nn as nn
import torch.optim as optim

# Data loaders
from data.cardiovascular_loader import load_cardiovascular_data

# ======================================================================
# Constants
# ======================================================================

OUTPUT_DIR = FRAMEWORK_DIR / "benchmarks" / "paper_results_tabular"
CHECKPOINT_FILE = "checkpoint_cascade8.json"
LOG_FILE = "experiment_cascade8.log"

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
        learning_rate=0.01, batch_size=128, num_rounds=30, local_epochs=3,
        mu=0.1, num_clients=5, input_dim=21, num_classes=2,
    ),
}

DATASETS_U = ["Cardiovascular", "PTB_XL", "Breast_Cancer"]
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
    fd, tmp = tempfile.mkstemp(dir=str(OUTPUT_DIR), prefix=".cas8_", suffix=".tmp")
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


def train_fl_model(dataset, cfg, algo, num_rounds, seed, is_iid=True, alpha=0.5,
                   dp_epsilon=None):
    """Train FL model and return (global_model, personal, client_data, client_test, meta)."""
    num_clients = cfg["num_clients"]
    client_data, client_test, meta = load_dataset(dataset, num_clients, seed, is_iid, alpha)
    model = create_model(cfg["input_dim"], cfg["num_classes"], seed=seed)
    all_clients = list(client_data.keys())

    for r in range(num_rounds):
        model = federated_round(model, client_data, all_clients, cfg, dp_epsilon=dp_epsilon)

    if algo == "Ditto":
        personal = {}
        for cid in all_clients:
            pm = create_model(cfg["input_dim"], cfg["num_classes"])
            set_params(pm, get_params(model))
            X, y = client_data[cid]
            train_local_sgd(pm, X, y, cfg["local_epochs"], cfg["learning_rate"], cfg["batch_size"])
            personal[cid] = pm
        return model, personal, client_data, client_test, meta
    elif algo == "HPFL":
        personal = {}
        for cid in all_clients:
            pm = create_model(cfg["input_dim"], cfg["num_classes"])
            set_params(pm, get_params(model))
            X, y = client_data[cid]
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
# FL Training with compression (Block T)
# ======================================================================

def federated_round_compressed(model, client_data, all_clients, cfg, compressor_mgr):
    """One FL round with gradient compression."""
    global_params = get_params(model)
    all_decompressed = []
    samples = []
    round_ratios = []

    for cid in all_clients:
        X, y = client_data[cid]
        lm = create_model(cfg["input_dim"], cfg["num_classes"])
        set_params(lm, global_params)
        train_local_sgd(lm, X, y, cfg["local_epochs"], cfg["learning_rate"], cfg["batch_size"])
        lp = get_params(lm)
        delta = {n: (lp[n] - global_params[n]).cpu().numpy() for n in global_params}

        # Compress
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
# FL Training with local or central DP (Block U)
# ======================================================================

def federated_round_dp(model, client_data, all_clients, cfg, dp_mode, dp_epsilon, delta=1e-5):
    """One FL round with local or central DP."""
    global_params = get_params(model)
    updates, samples = [], []

    sigma = np.sqrt(2.0 * np.log(1.25 / delta)) / dp_epsilon if dp_epsilon < float("inf") else 0.0

    for cid in all_clients:
        X, y = client_data[cid]
        lm = create_model(cfg["input_dim"], cfg["num_classes"])
        set_params(lm, global_params)
        train_local_sgd(lm, X, y, cfg["local_epochs"], cfg["learning_rate"], cfg["batch_size"])
        lp = get_params(lm)
        delta_p = {n: lp[n] - global_params[n] for n in global_params}

        # Local DP: add noise per-client before sending
        if dp_mode == "local" and sigma > 0:
            for n in delta_p:
                delta_p[n] += torch.randn_like(delta_p[n]) * sigma

        updates.append(delta_p)
        samples.append(len(y))

    total_s = sum(samples)
    avg_d = {}
    for n in global_params:
        avg_d[n] = sum(updates[i][n] * (samples[i] / total_s) for i in range(len(updates)))

    # Central DP: add noise once on aggregated update (scaled by 1/num_clients)
    if dp_mode == "central" and sigma > 0:
        central_sigma = sigma / len(all_clients)
        for n in avg_d:
            avg_d[n] += torch.randn_like(avg_d[n]) * central_sigma

    new_p = {n: global_params[n] + avg_d[n] for n in global_params}
    set_params(model, new_p)
    return model


# ======================================================================
# Block S: Secure Aggregation Overhead
# ======================================================================

def run_block_s(checkpoint, quick=False):
    """Block S: Secure Aggregation Overhead Benchmark."""
    log("=" * 70)
    log("BLOCK S: Secure Aggregation Overhead")
    log("=" * 70)

    # Methods to test (skip homomorphic — requires tenseal)
    methods = ["pairwise_masking", "secret_sharing"]
    gradient_dims = [100, 1000] if quick else [100, 1000, 5000]
    seeds = [42] if quick else DEFAULT_SEEDS
    num_clients = 5
    t0 = time.time()
    count = 0
    total = len(methods) * len(gradient_dims) * len(seeds)

    for method in methods:
        for gdim in gradient_dims:
            for seed in seeds:
                if _shutdown:
                    return count
                key = "S_{}_dim{}_s{}".format(method, gdim, seed)
                if key in checkpoint["results"]:
                    count += 1
                    continue
                count += 1
                log("  [{}/{}] {} / dim={} / seed={}".format(count, total, method, gdim, seed))
                try:
                    np.random.seed(seed)

                    # Generate synthetic gradients for overhead measurement
                    true_gradients = {
                        i: np.random.randn(gdim) * 0.1
                        for i in range(num_clients)
                    }
                    true_average = np.mean(list(true_gradients.values()), axis=0)

                    # Weighted version
                    weights = {i: np.random.randint(50, 500) for i in range(num_clients)}

                    if method == "pairwise_masking":
                        try:
                            from core.secure_aggregation import SecureAggregationManager
                            mgr = SecureAggregationManager(
                                num_clients=num_clients,
                                gradient_dim=gdim,
                                method="pairwise_masking",
                            )
                            # Time masking + aggregation
                            t_start = time.time()
                            result_uniform = mgr.secure_aggregate(true_gradients)
                            t_uniform = time.time() - t_start

                            t_start = time.time()
                            result_weighted = mgr.secure_aggregate(true_gradients, weights=weights)
                            t_weighted = time.time() - t_start

                            error_uniform = float(np.linalg.norm(result_uniform - true_average))

                            total_w = sum(weights.values())
                            true_weighted = sum(
                                (weights[cid] / total_w) * grad
                                for cid, grad in true_gradients.items()
                            )
                            error_weighted = float(np.linalg.norm(result_weighted - true_weighted))

                            checkpoint["results"][key] = {
                                "block": "S_secure_aggregation",
                                "method": method,
                                "gradient_dim": gdim,
                                "num_clients": num_clients,
                                "seed": seed,
                                "time_uniform_sec": round(t_uniform, 4),
                                "time_weighted_sec": round(t_weighted, 4),
                                "error_uniform": round(error_uniform, 10),
                                "error_weighted": round(error_weighted, 10),
                                "masks_cancel": error_uniform < 1e-8,
                            }
                        except ImportError as ie:
                            log("  SKIP pairwise_masking (missing dep): {}".format(ie))
                            checkpoint["results"][key] = {
                                "block": "S_secure_aggregation",
                                "method": method,
                                "gradient_dim": gdim,
                                "seed": seed,
                                "skipped": True,
                                "reason": str(ie),
                            }

                    elif method == "secret_sharing":
                        from core.secure_aggregation import ShamirSecretSharing
                        threshold = max(2, num_clients // 2 + 1)
                        ss = ShamirSecretSharing(threshold=threshold, num_shares=num_clients)

                        # Time share + reconstruct a single value
                        test_secret = 12345
                        t_start = time.time()
                        shares = ss.share(test_secret)
                        t_share = time.time() - t_start

                        t_start = time.time()
                        reconstructed = ss.reconstruct(shares)
                        t_recon = time.time() - t_start

                        correct = (reconstructed == test_secret)

                        # Array share/reconstruct for overhead
                        small_arr = np.random.randn(min(gdim, 50)) * 0.1
                        t_start = time.time()
                        arr_shares = ss.share_array(small_arr)
                        t_arr_share = time.time() - t_start

                        t_start = time.time()
                        arr_recon = ss.reconstruct_array(arr_shares)
                        t_arr_recon = time.time() - t_start

                        arr_error = float(np.linalg.norm(arr_recon - small_arr))

                        checkpoint["results"][key] = {
                            "block": "S_secure_aggregation",
                            "method": method,
                            "gradient_dim": gdim,
                            "num_clients": num_clients,
                            "threshold": threshold,
                            "seed": seed,
                            "scalar_share_sec": round(t_share, 6),
                            "scalar_reconstruct_sec": round(t_recon, 6),
                            "scalar_correct": correct,
                            "array_dim": min(gdim, 50),
                            "array_share_sec": round(t_arr_share, 4),
                            "array_reconstruct_sec": round(t_arr_recon, 4),
                            "array_error": round(arr_error, 8),
                        }

                    save_checkpoint(checkpoint)
                    _cleanup_gpu()
                except Exception as e:
                    log("  ERROR: {} — {}".format(key, e))
                    traceback.print_exc()

    # Also test SecAgg integrated with real FL training (Cardiovascular only)
    if not _shutdown and not quick:
        try:
            from core.secure_aggregation import SecureAggregationManager
            ds = "Cardiovascular"
            cfg = DATASET_CONFIGS[ds]
            seed = 42
            key_fl = "S_fl_integration_s{}".format(seed)
            if key_fl not in checkpoint["results"]:
                log("  SecAgg FL integration test (Cardiovascular)...")
                client_data, client_test, meta = load_dataset(ds, cfg["num_clients"], seed)
                model = create_model(cfg["input_dim"], cfg["num_classes"], seed=seed)
                all_clients = list(client_data.keys())

                # Flatten model params for gradient dim
                flat_dim = sum(p.numel() for p in model.parameters())
                mgr = SecureAggregationManager(
                    num_clients=len(all_clients),
                    gradient_dim=flat_dim,
                    method="pairwise_masking",
                )

                num_rounds = 10
                global_params = get_params(model)
                for r in range(num_rounds):
                    updates = {}
                    sample_weights = {}
                    for cid in all_clients:
                        X, y = client_data[cid]
                        lm = create_model(cfg["input_dim"], cfg["num_classes"])
                        set_params(lm, global_params)
                        train_local_sgd(lm, X, y, cfg["local_epochs"],
                                        cfg["learning_rate"], cfg["batch_size"])
                        lp = get_params(lm)
                        # Flatten delta into single vector
                        delta_flat = np.concatenate([
                            (lp[n] - global_params[n]).cpu().numpy().flatten()
                            for n in sorted(global_params.keys())
                        ])
                        updates[cid] = delta_flat
                        sample_weights[cid] = float(len(y))

                    # Secure aggregate
                    agg_flat = mgr.secure_aggregate(updates, weights=sample_weights)

                    # Unflatten back to model params
                    offset = 0
                    new_p = {}
                    for n in sorted(global_params.keys()):
                        shape = global_params[n].shape
                        numel = global_params[n].numel()
                        new_p[n] = global_params[n] + torch.FloatTensor(
                            agg_flat[offset:offset + numel].reshape(shape)
                        ).to(DEVICE)
                        offset += numel
                    set_params(model, new_p)
                    global_params = get_params(model)

                # Evaluate
                all_X = np.concatenate([client_test[c][0] for c in client_test])
                all_y = np.concatenate([client_test[c][1] for c in client_test])
                acc = evaluate_model(model, all_X, all_y)

                checkpoint["results"][key_fl] = {
                    "block": "S_secure_aggregation",
                    "method": "pairwise_masking_fl",
                    "dataset": ds,
                    "num_rounds": num_rounds,
                    "accuracy": round(float(acc), 4),
                    "gradient_dim": flat_dim,
                }
                save_checkpoint(checkpoint)
                count += 1
        except ImportError as ie:
            log("  SKIP FL integration (missing dep): {}".format(ie))
        except Exception as e:
            log("  ERROR FL integration: {}".format(e))
            traceback.print_exc()

    elapsed = time.time() - t0
    checkpoint["metadata"]["block_s_time"] = round(elapsed, 1)
    save_checkpoint(checkpoint)
    log("Block S complete: {} experiments in {}".format(count, format_time(elapsed)))
    return count


# ======================================================================
# Block T: Gradient Compression Benchmark
# ======================================================================

def run_block_t(checkpoint, quick=False):
    """Block T: Gradient Compression Benchmark."""
    log("=" * 70)
    log("BLOCK T: Gradient Compression Benchmark")
    log("=" * 70)

    from core.model_compression import CompressionConfig, CompressionManager

    compression_methods = ["signsgd", "qsgd", "terngrad", "topk", "randomk", "threshold", "powersgd"]
    if quick:
        compression_methods = ["signsgd", "topk", "powersgd"]
    seeds = [42] if quick else [42, 123, 456]
    ds = "Cardiovascular"
    cfg = DATASET_CONFIGS[ds]
    num_rounds = 5 if quick else 15
    t0 = time.time()
    count = 0
    total = len(compression_methods) * len(seeds)

    for method in compression_methods:
        for seed in seeds:
            if _shutdown:
                return count
            key = "T_{}_s{}".format(method, seed)
            if key in checkpoint["results"]:
                count += 1
                continue
            count += 1
            log("  [{}/{}] {} / seed={}".format(count, total, method, seed))
            try:
                comp_cfg = CompressionConfig(
                    method=method,
                    num_bits=4,
                    k_ratio=0.1,
                    threshold=0.5,
                    rank=4,
                    use_error_feedback=True,
                )
                comp_mgr = CompressionManager(comp_cfg)

                client_data, client_test, meta = load_dataset(ds, cfg["num_clients"], seed)
                model = create_model(cfg["input_dim"], cfg["num_classes"], seed=seed)
                all_clients = list(client_data.keys())

                round_ratios = []
                for r in range(num_rounds):
                    model, avg_ratio = federated_round_compressed(
                        model, client_data, all_clients, cfg, comp_mgr
                    )
                    round_ratios.append(avg_ratio)

                # Evaluate
                all_X = np.concatenate([client_test[c][0] for c in client_test])
                all_y = np.concatenate([client_test[c][1] for c in client_test])
                acc = evaluate_model(model, all_X, all_y)

                stats = comp_mgr.get_stats()

                checkpoint["results"][key] = {
                    "block": "T_compression",
                    "method": method,
                    "dataset": ds,
                    "seed": seed,
                    "num_rounds": num_rounds,
                    "accuracy": round(float(acc), 4),
                    "avg_compression_ratio": round(stats["average_compression_ratio"], 2),
                    "total_original_bytes": stats["total_original_bytes"],
                    "total_compressed_bytes": stats["total_compressed_bytes"],
                    "bandwidth_saved_pct": round(stats["bandwidth_saved_pct"], 1),
                    "per_round_ratios": [round(r, 2) for r in round_ratios],
                }
                save_checkpoint(checkpoint)
                _cleanup_gpu()
            except Exception as e:
                log("  ERROR: {} — {}".format(key, e))
                traceback.print_exc()

    # Baseline (no compression)
    for seed in seeds:
        if _shutdown:
            return count
        key = "T_none_s{}".format(seed)
        if key in checkpoint["results"]:
            count += 1
            continue
        count += 1
        log("  [{}/{}] no_compression / seed={}".format(count, total + len(seeds), seed))
        try:
            client_data, client_test, meta = load_dataset(ds, cfg["num_clients"], seed)
            model = create_model(cfg["input_dim"], cfg["num_classes"], seed=seed)
            all_clients = list(client_data.keys())
            for r in range(num_rounds):
                model = federated_round(model, client_data, all_clients, cfg)
            all_X = np.concatenate([client_test[c][0] for c in client_test])
            all_y = np.concatenate([client_test[c][1] for c in client_test])
            acc = evaluate_model(model, all_X, all_y)
            checkpoint["results"][key] = {
                "block": "T_compression",
                "method": "none",
                "dataset": ds,
                "seed": seed,
                "num_rounds": num_rounds,
                "accuracy": round(float(acc), 4),
                "avg_compression_ratio": 1.0,
                "bandwidth_saved_pct": 0.0,
            }
            save_checkpoint(checkpoint)
            _cleanup_gpu()
        except Exception as e:
            log("  ERROR: {} — {}".format(key, e))
            traceback.print_exc()

    elapsed = time.time() - t0
    checkpoint["metadata"]["block_t_time"] = round(elapsed, 1)
    save_checkpoint(checkpoint)
    log("Block T complete: {} experiments in {}".format(count, format_time(elapsed)))
    return count


# ======================================================================
# Block U: Local DP vs Central DP
# ======================================================================

def run_block_u(checkpoint, quick=False):
    """Block U: Local DP vs Central DP Comparison."""
    log("=" * 70)
    log("BLOCK U: Local DP vs Central DP")
    log("=" * 70)

    dp_modes = ["local", "central"]
    dp_epsilons = [1.0, 5.0, 10.0]
    seeds = [42] if quick else DEFAULT_SEEDS
    datasets = ["Cardiovascular"] if quick else DATASETS_U
    t0 = time.time()
    count = 0
    total = len(datasets) * len(dp_modes) * len(dp_epsilons) * len(seeds)

    for ds in datasets:
        cfg = DATASET_CONFIGS[ds]
        num_rounds = 10 if quick else cfg["num_rounds"]
        for dp_mode in dp_modes:
            for dp_eps in dp_epsilons:
                for seed in seeds:
                    if _shutdown:
                        return count
                    key = "U_{}_{}_eps{}_s{}".format(ds, dp_mode, dp_eps, seed)
                    if key in checkpoint["results"]:
                        count += 1
                        continue
                    count += 1
                    log("  [{}/{}] {} / {} / eps={} / seed={}".format(
                        count, total, ds, dp_mode, dp_eps, seed))
                    try:
                        client_data, client_test, meta = load_dataset(
                            ds, cfg["num_clients"], seed)
                        model = create_model(cfg["input_dim"], cfg["num_classes"], seed=seed)
                        all_clients = list(client_data.keys())

                        for r in range(num_rounds):
                            model = federated_round_dp(
                                model, client_data, all_clients, cfg,
                                dp_mode=dp_mode, dp_epsilon=dp_eps)

                        # Evaluate
                        all_X = np.concatenate([client_test[c][0] for c in client_test])
                        all_y = np.concatenate([client_test[c][1] for c in client_test])
                        acc = evaluate_model(model, all_X, all_y)

                        # Per-client accuracy
                        per_client_acc = {}
                        for cid in client_test:
                            Xc, yc = client_test[cid]
                            per_client_acc[cid] = round(evaluate_model(model, Xc, yc), 4)

                        # Compute sigma used
                        delta_dp = 1e-5
                        sigma = np.sqrt(2.0 * np.log(1.25 / delta_dp)) / dp_eps

                        checkpoint["results"][key] = {
                            "block": "U_local_vs_central_dp",
                            "dataset": ds,
                            "dp_mode": dp_mode,
                            "dp_epsilon": dp_eps,
                            "seed": seed,
                            "num_rounds": num_rounds,
                            "accuracy": round(float(acc), 4),
                            "per_client_accuracy": per_client_acc,
                            "sigma": round(float(sigma), 4),
                            "effective_sigma": round(
                                float(sigma if dp_mode == "local" else sigma / len(all_clients)),
                                4),
                        }
                        save_checkpoint(checkpoint)
                        _cleanup_gpu()
                    except Exception as e:
                        log("  ERROR: {} — {}".format(key, e))
                        traceback.print_exc()

    # Also run no-DP baseline
    for ds in datasets:
        cfg = DATASET_CONFIGS[ds]
        num_rounds = 10 if quick else cfg["num_rounds"]
        for seed in seeds:
            if _shutdown:
                return count
            key = "U_{}_noDP_s{}".format(ds, seed)
            if key in checkpoint["results"]:
                count += 1
                continue
            count += 1
            log("  [+] {} / noDP baseline / seed={}".format(ds, seed))
            try:
                model, _, _, client_test, _ = train_fl_model(
                    ds, cfg, "FedAvg", num_rounds, seed)
                all_X = np.concatenate([client_test[c][0] for c in client_test])
                all_y = np.concatenate([client_test[c][1] for c in client_test])
                acc = evaluate_model(model, all_X, all_y)
                checkpoint["results"][key] = {
                    "block": "U_local_vs_central_dp",
                    "dataset": ds,
                    "dp_mode": "none",
                    "dp_epsilon": None,
                    "seed": seed,
                    "accuracy": round(float(acc), 4),
                }
                save_checkpoint(checkpoint)
                _cleanup_gpu()
            except Exception as e:
                log("  ERROR: {} — {}".format(key, e))
                traceback.print_exc()

    elapsed = time.time() - t0
    checkpoint["metadata"]["block_u_time"] = round(elapsed, 1)
    save_checkpoint(checkpoint)
    log("Block U complete: {} experiments in {}".format(count, format_time(elapsed)))
    return count


# ======================================================================
# Block V: Vertical FL Simulation
# ======================================================================

def run_block_v(checkpoint, quick=False):
    """Block V: Vertical FL Simulation (SplitNN on Cardiovascular)."""
    log("=" * 70)
    log("BLOCK V: Vertical FL Simulation")
    log("=" * 70)

    from core.vertical_fl import (
        VerticalPartition, VerticalConfig, SecureVerticalFL,
        SplitNNCoordinator, PrivateSetIntersection,
    )

    # Use real Cardiovascular data split vertically
    # Party A: demographics (age, gender, height, weight) -> features 0-3
    # Party B: clinical (ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active) -> features 4-10
    party_a_features = [0, 1, 2, 3]       # age, gender, height, weight
    party_b_features = [4, 5, 6, 7, 8, 9, 10]  # clinical features

    n_parties_list = [2] if quick else [2, 3]
    dp_configs = [False] if quick else [False, True]
    overlap_rates = [1.0] if quick else [1.0, 0.8]
    seeds = [42] if quick else [42, 123, 456]
    num_epochs = 10 if quick else 30
    t0 = time.time()
    count = 0
    total = len(n_parties_list) * len(dp_configs) * len(overlap_rates) * len(seeds)

    for n_parties in n_parties_list:
        for use_dp in dp_configs:
            for overlap in overlap_rates:
                for seed in seeds:
                    if _shutdown:
                        return count
                    dp_tag = "dp5" if use_dp else "noDP"
                    key = "V_{}p_{}_ovlp{}_s{}".format(n_parties, dp_tag, overlap, seed)
                    if key in checkpoint["results"]:
                        count += 1
                        continue
                    count += 1
                    log("  [{}/{}] parties={} / {} / overlap={} / seed={}".format(
                        count, total, n_parties, dp_tag, overlap, seed))
                    try:
                        np.random.seed(seed)

                        # Load Cardiovascular data (centralized for vertical split)
                        client_data, client_test, meta = load_dataset(
                            "Cardiovascular", 1, seed, is_iid=True)

                        # Pool all data
                        X_all = np.concatenate([client_data[c][0] for c in client_data])
                        y_all = np.concatenate([client_data[c][1] for c in client_data])
                        X_test = np.concatenate([client_test[c][0] for c in client_test])
                        y_test = np.concatenate([client_test[c][1] for c in client_test])

                        n_samples = len(y_all)
                        patient_ids = np.array(["P-{:06d}".format(i) for i in range(n_samples)])
                        test_ids = np.array(["P-{:06d}".format(i) for i in range(len(y_test))])

                        # Simulate partial overlap
                        if overlap < 1.0:
                            # Party B only has a subset of patients
                            n_b = int(n_samples * overlap)
                            idx_b = np.random.choice(n_samples, n_b, replace=False)
                            idx_b.sort()
                        else:
                            idx_b = np.arange(n_samples)

                        # Create vertical partitions
                        partitions = {}

                        # Party A: demographics (has labels)
                        partitions[0] = VerticalPartition(
                            party_id=0,
                            features=X_all[:, party_a_features],
                            feature_names=["age", "gender", "height", "weight"],
                            sample_ids=patient_ids,
                            has_labels=True,
                            labels=y_all.astype(float),
                        )

                        # Party B: clinical
                        partitions[1] = VerticalPartition(
                            party_id=1,
                            features=X_all[idx_b][:, party_b_features],
                            feature_names=["ap_hi", "ap_lo", "cholesterol", "gluc",
                                           "smoke", "alco", "active"],
                            sample_ids=patient_ids[idx_b],
                            has_labels=False,
                        )

                        if n_parties >= 3:
                            # Party C: subset of clinical features (split further)
                            party_c_features = [8, 9, 10]  # smoke, alco, active
                            partitions[2] = VerticalPartition(
                                party_id=2,
                                features=X_all[idx_b][:, party_c_features],
                                feature_names=["smoke", "alco", "active"],
                                sample_ids=patient_ids[idx_b],
                                has_labels=False,
                            )
                            # Remove from party B
                            party_b_reduced = [4, 5, 6, 7]
                            partitions[1] = VerticalPartition(
                                party_id=1,
                                features=X_all[idx_b][:, party_b_reduced],
                                feature_names=["ap_hi", "ap_lo", "cholesterol", "gluc"],
                                sample_ids=patient_ids[idx_b],
                                has_labels=False,
                            )

                        # PSI alignment
                        psi = PrivateSetIntersection()
                        party_hashes = [
                            psi.hash_ids(partitions[pid].sample_ids)
                            for pid in sorted(partitions.keys())
                        ]
                        aligned_tuple = psi.find_intersection(party_hashes)
                        n_aligned = len(aligned_tuple[0])

                        # Configure SplitNN
                        if n_parties == 2:
                            p_configs = [
                                {"party_id": 0, "input_dim": len(party_a_features),
                                 "hidden_dims": [16, 8], "lr": 0.01},
                                {"party_id": 1,
                                 "input_dim": len(party_b_features),
                                 "hidden_dims": [16, 8], "lr": 0.01},
                            ]
                        else:
                            p_configs = [
                                {"party_id": 0, "input_dim": len(party_a_features),
                                 "hidden_dims": [16, 8], "lr": 0.01},
                                {"party_id": 1, "input_dim": 4,
                                 "hidden_dims": [16, 8], "lr": 0.01},
                                {"party_id": 2, "input_dim": 3,
                                 "hidden_dims": [16, 8], "lr": 0.01},
                            ]

                        config = VerticalConfig(
                            algorithm="splitnn",
                            use_differential_privacy=use_dp,
                            epsilon=5.0,
                        )

                        vfl = SecureVerticalFL(config, p_configs, top_party_id=0)

                        # Train (suppress SecureVerticalFL print output)
                        f_buf = io.StringIO()
                        with redirect_stdout(f_buf):
                            history = vfl.train(partitions, num_epochs=num_epochs, batch_size=64)

                        final_acc = history["accuracy"][-1] if history["accuracy"] else 0.0
                        final_loss = history["loss"][-1] if history["loss"] else float("inf")

                        # Test accuracy (using aligned test data)
                        test_partitions = {}
                        test_partitions[0] = VerticalPartition(
                            party_id=0,
                            features=X_test[:, party_a_features],
                            feature_names=["age", "gender", "height", "weight"],
                            sample_ids=test_ids,
                            has_labels=True,
                            labels=y_test.astype(float),
                        )
                        if n_parties == 2:
                            test_partitions[1] = VerticalPartition(
                                party_id=1,
                                features=X_test[:, party_b_features],
                                feature_names=["ap_hi", "ap_lo", "cholesterol", "gluc",
                                               "smoke", "alco", "active"],
                                sample_ids=test_ids,
                                has_labels=False,
                            )
                        else:
                            test_partitions[1] = VerticalPartition(
                                party_id=1,
                                features=X_test[:, [4, 5, 6, 7]],
                                feature_names=["ap_hi", "ap_lo", "cholesterol", "gluc"],
                                sample_ids=test_ids,
                                has_labels=False,
                            )
                            test_partitions[2] = VerticalPartition(
                                party_id=2,
                                features=X_test[:, [8, 9, 10]],
                                feature_names=["smoke", "alco", "active"],
                                sample_ids=test_ids,
                                has_labels=False,
                            )

                        # Predict on test
                        test_data = {
                            pid: tp.features for pid, tp in test_partitions.items()
                        }
                        preds = vfl.splitnn.predict(test_data)
                        test_acc = float(np.mean(
                            (preds.flatten() > 0.5) == y_test.flatten()
                        ))

                        checkpoint["results"][key] = {
                            "block": "V_vertical_fl",
                            "n_parties": n_parties,
                            "use_dp": use_dp,
                            "dp_epsilon": 5.0 if use_dp else None,
                            "overlap_rate": overlap,
                            "seed": seed,
                            "num_epochs": num_epochs,
                            "n_total_samples": n_samples,
                            "n_aligned_samples": n_aligned,
                            "alignment_rate": round(n_aligned / n_samples, 4),
                            "train_accuracy": round(float(final_acc), 4),
                            "train_loss": round(float(final_loss), 4),
                            "test_accuracy": round(float(test_acc), 4),
                            "convergence_epochs": len(history["accuracy"]),
                        }
                        save_checkpoint(checkpoint)
                        _cleanup_gpu()
                    except Exception as e:
                        log("  ERROR: {} — {}".format(key, e))
                        traceback.print_exc()

    elapsed = time.time() - t0
    checkpoint["metadata"]["block_v_time"] = round(elapsed, 1)
    save_checkpoint(checkpoint)
    log("Block V complete: {} experiments in {}".format(count, format_time(elapsed)))
    return count


# ======================================================================
# Block W: CDC Diabetes Scalability
# ======================================================================

def run_block_w(checkpoint, quick=False):
    """Block W: CDC Diabetes Scalability (253K samples)."""
    log("=" * 70)
    log("BLOCK W: CDC Diabetes Scalability (253K samples)")
    log("=" * 70)

    # Check if dataset exists
    data_dir = FRAMEWORK_DIR / "data" / "cdc_diabetes"
    csv_path = data_dir / "diabetes_binary_health_indicators_BRFSS2015.csv"
    if not csv_path.exists():
        log("  WARNING: CDC Diabetes dataset not found at {}".format(csv_path))
        log("  Download from: https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset")
        log("  Place CSV in: {}".format(data_dir))
        log("  Skipping Block W.")
        checkpoint["metadata"]["block_w_skipped"] = True
        checkpoint["metadata"]["block_w_reason"] = "Dataset not found: {}".format(csv_path)
        save_checkpoint(checkpoint)
        return 0

    algorithms = ["FedAvg", "Ditto", "HPFL"]
    iid_modes = [True, False]
    seeds = [42] if quick else DEFAULT_SEEDS
    if quick:
        algorithms = ["FedAvg"]
        iid_modes = [True]

    ds = "CDC_Diabetes"
    cfg = DATASET_CONFIGS[ds]
    num_rounds = 5 if quick else 20
    t0 = time.time()
    count = 0
    total = len(algorithms) * len(iid_modes) * len(seeds)

    for algo in algorithms:
        for is_iid in iid_modes:
            for seed in seeds:
                if _shutdown:
                    return count
                iid_tag = "IID" if is_iid else "NonIID"
                key = "W_{}_{}_s{}".format(algo, iid_tag, seed)
                if key in checkpoint["results"]:
                    count += 1
                    continue
                count += 1
                log("  [{}/{}] CDC_Diabetes / {} / {} / seed={}".format(
                    count, total, algo, iid_tag, seed))
                try:
                    t_exp = time.time()

                    model, personal, client_data, client_test, meta = train_fl_model(
                        ds, cfg, algo, num_rounds, seed, is_iid=is_iid, alpha=0.5)

                    # Evaluate using personal models if available
                    if personal:
                        all_preds, all_labels = [], []
                        per_client_acc = {}
                        for cid in client_test:
                            Xc, yc = client_test[cid]
                            cm = personal[cid] if cid in personal else model
                            acc_c = evaluate_model(cm, Xc, yc)
                            per_client_acc[cid] = round(acc_c, 4)
                            cm.eval()
                            X_t = torch.FloatTensor(Xc).to(DEVICE)
                            with torch.no_grad():
                                preds_c = cm(X_t).argmax(dim=1).cpu().numpy()
                            all_preds.append(preds_c)
                            all_labels.append(yc)
                        all_preds = np.concatenate(all_preds)
                        all_labels = np.concatenate(all_labels)
                        acc = float((all_preds == all_labels).mean())
                    else:
                        all_X = np.concatenate([client_test[c][0] for c in client_test])
                        all_y = np.concatenate([client_test[c][1] for c in client_test])
                        acc = evaluate_model(model, all_X, all_y)
                        per_client_acc = {}
                        for cid in client_test:
                            Xc, yc = client_test[cid]
                            per_client_acc[cid] = round(evaluate_model(model, Xc, yc), 4)
                        all_labels = all_y
                        model.eval()
                        with torch.no_grad():
                            all_preds_t = model(torch.FloatTensor(all_X).to(DEVICE))
                            all_preds = all_preds_t.argmax(dim=1).cpu().numpy()

                    # Class balance stats
                    total_samples = sum(len(client_data[c][1]) for c in client_data)
                    samples_per_client = {
                        cid: len(client_data[cid][1]) for cid in client_data
                    }

                    # F1 score
                    tp = int(((all_preds == 1) & (all_labels == 1)).sum())
                    fp = int(((all_preds == 1) & (all_labels == 0)).sum())
                    fn = int(((all_preds == 0) & (all_labels == 1)).sum())
                    precision = tp / max(tp + fp, 1)
                    recall = tp / max(tp + fn, 1)
                    f1 = 2 * precision * recall / max(precision + recall, 1e-10)

                    exp_time = time.time() - t_exp

                    checkpoint["results"][key] = {
                        "block": "W_cdc_diabetes",
                        "dataset": ds,
                        "algorithm": algo,
                        "is_iid": is_iid,
                        "seed": seed,
                        "num_rounds": num_rounds,
                        "total_train_samples": total_samples,
                        "accuracy": round(float(acc), 4),
                        "f1_score": round(float(f1), 4),
                        "precision": round(float(precision), 4),
                        "recall": round(float(recall), 4),
                        "per_client_accuracy": per_client_acc,
                        "samples_per_client": samples_per_client,
                        "experiment_time_sec": round(exp_time, 1),
                    }
                    save_checkpoint(checkpoint)
                    _cleanup_gpu()
                except Exception as e:
                    log("  ERROR: {} — {}".format(key, e))
                    traceback.print_exc()

    # CDC with DP (eps=10, eps=1) — FedAvg only
    dp_levels = [10.0, 1.0]
    if quick:
        dp_levels = [10.0]
    for dp_eps in dp_levels:
        for seed in seeds:
            if _shutdown:
                return count
            key = "W_FedAvg_IID_eps{}_s{}".format(dp_eps, seed)
            if key in checkpoint["results"]:
                count += 1
                continue
            count += 1
            log("  [+] CDC_Diabetes / FedAvg / IID / eps={} / seed={}".format(dp_eps, seed))
            try:
                t_exp = time.time()
                model, _, client_data, client_test, meta = train_fl_model(
                    ds, cfg, "FedAvg", num_rounds, seed, is_iid=True, dp_epsilon=dp_eps)

                all_X = np.concatenate([client_test[c][0] for c in client_test])
                all_y = np.concatenate([client_test[c][1] for c in client_test])
                acc = evaluate_model(model, all_X, all_y)

                # Per-client accuracy
                per_client_acc = {}
                for cid in client_test:
                    Xc, yc = client_test[cid]
                    per_client_acc[cid] = round(evaluate_model(model, Xc, yc), 4)
                samples_per_client = {
                    cid: len(client_data[cid][1]) for cid in client_data
                }

                # F1 score
                model.eval()
                with torch.no_grad():
                    all_preds = model(torch.FloatTensor(all_X).to(DEVICE)).argmax(dim=1).cpu().numpy()
                tp = int(((all_preds == 1) & (all_y == 1)).sum())
                fp = int(((all_preds == 1) & (all_y == 0)).sum())
                fn = int(((all_preds == 0) & (all_y == 1)).sum())
                precision = tp / max(tp + fp, 1)
                recall = tp / max(tp + fn, 1)
                f1 = 2 * precision * recall / max(precision + recall, 1e-10)

                exp_time = time.time() - t_exp
                checkpoint["results"][key] = {
                    "block": "W_cdc_diabetes",
                    "dataset": ds,
                    "algorithm": "FedAvg",
                    "is_iid": True,
                    "dp_epsilon": dp_eps,
                    "seed": seed,
                    "num_rounds": num_rounds,
                    "total_train_samples": sum(len(client_data[c][1]) for c in client_data),
                    "accuracy": round(float(acc), 4),
                    "f1_score": round(float(f1), 4),
                    "precision": round(float(precision), 4),
                    "recall": round(float(recall), 4),
                    "per_client_accuracy": per_client_acc,
                    "samples_per_client": samples_per_client,
                    "experiment_time_sec": round(exp_time, 1),
                }
                save_checkpoint(checkpoint)
                _cleanup_gpu()
            except Exception as e:
                log("  ERROR: {} — {}".format(key, e))
                traceback.print_exc()

    elapsed = time.time() - t0
    checkpoint["metadata"]["block_w_time"] = round(elapsed, 1)
    save_checkpoint(checkpoint)
    log("Block W complete: {} experiments in {}".format(count, format_time(elapsed)))
    return count


# ======================================================================
# Main
# ======================================================================

def main():
    global _log_file, _shutdown

    parser = argparse.ArgumentParser(
        description="FL-EHDS Cascade 8: Communication, Privacy & Scalability")
    parser.add_argument("--quick", action="store_true",
                        help="Reduced rounds/seeds for quick test")
    parser.add_argument("--fresh", action="store_true",
                        help="Delete existing checkpoint and start fresh")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, _handle_signal)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _log_file = open(OUTPUT_DIR / LOG_FILE, "a", encoding="utf-8")

    log("=" * 70)
    log("FL-EHDS Cascading Analysis — Phase 8 (Communication, Privacy & Scalability)")
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
                "experiment": "cascade8_communication_privacy_scalability",
                "started": datetime.now().isoformat(),
                "mode": "quick" if args.quick else "full",
            },
            "results": {},
            "completed": [],
        }
        save_checkpoint(checkpoint)

    # Block S: Secure Aggregation
    if not _shutdown:
        n_s = run_block_s(checkpoint, quick=args.quick)
        if "S" not in checkpoint.get("completed", []):
            checkpoint.setdefault("completed", []).append("S")

    # Block T: Gradient Compression
    if not _shutdown:
        n_t = run_block_t(checkpoint, quick=args.quick)
        if "T" not in checkpoint.get("completed", []):
            checkpoint.setdefault("completed", []).append("T")

    # Block U: Local vs Central DP
    if not _shutdown:
        n_u = run_block_u(checkpoint, quick=args.quick)
        if "U" not in checkpoint.get("completed", []):
            checkpoint.setdefault("completed", []).append("U")

    # Block V: Vertical FL
    if not _shutdown:
        n_v = run_block_v(checkpoint, quick=args.quick)
        if "V" not in checkpoint.get("completed", []):
            checkpoint.setdefault("completed", []).append("V")

    # Block W: CDC Diabetes
    if not _shutdown:
        n_w = run_block_w(checkpoint, quick=args.quick)
        if "W" not in checkpoint.get("completed", []):
            checkpoint.setdefault("completed", []).append("W")

    # Final save
    checkpoint["metadata"]["finished"] = datetime.now().isoformat()
    save_checkpoint(checkpoint)

    # Summary
    blocks = {"S": 0, "T": 0, "U": 0, "V": 0, "W": 0}
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

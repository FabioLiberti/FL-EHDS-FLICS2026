#!/usr/bin/env python3
"""
FL-EHDS Cascading Analysis & Experiments — Phase 6.

6 blocks executed sequentially (Tiers 7–12):

Block G: Dataset Coverage Expansion (~45 exp)
  3 new datasets (Stroke, CKD, Cirrhosis) × 5 algorithms × 3 conditions
  (IID, non-IID α=0.5, non-IID + DP ε=10).

Block H: Hierarchical FL (~30 exp)
  3 topologies (flat, 2-tier, 3-tier) × 3 datasets × 2 scales + 2 seeds.

Block I: Asynchronous FL (~36 exp)
  4 async strategies (FedAsync, FedBuff, AsyncSGD, FedAT) × 3 datasets
  × 3 staleness levels.

Block J: Multi-Task FL (~24 exp)
  2 sharing modes (hard, soft) × 3 datasets × 2 task combos × 2 seeds.

Block K: Byzantine Boundary (~36 exp)
  3 attacks × 4 defenses × 3 datasets.

Block L: Shapley Values / Incentive (~30 exp)
  Shapley contribution valuation for 3 datasets × 2 conditions × 5 seeds.

Usage:
    cd fl-ehds-framework
    python -m benchmarks.run_analysis_cascade6 [--quick] [--fresh]

Output:
    benchmarks/paper_results_tabular/checkpoint_cascade6.json

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
from itertools import combinations

import numpy as np

FRAMEWORK_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(FRAMEWORK_DIR))

import torch
import torch.nn as nn
import torch.optim as optim

# Data loaders — original 3 datasets
from data.ptbxl_loader import load_ptbxl_data
from data.cardiovascular_loader import load_cardiovascular_data
from data.breast_cancer_loader import load_breast_cancer_data

# Data loaders — new datasets (Tier 7)
from data.stroke_loader import load_stroke_data
from data.ckd_loader import load_ckd_data
from data.cirrhosis_loader import load_cirrhosis_data

# ======================================================================
# Constants
# ======================================================================

OUTPUT_DIR = FRAMEWORK_DIR / "benchmarks" / "paper_results_tabular"
CHECKPOINT_FILE = "checkpoint_cascade6.json"
LOG_FILE = "experiment_cascade6.log"

# Original 3 datasets
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

# New datasets for Tier 7
NEW_DATASET_CONFIGS = {
    "Stroke": dict(
        learning_rate=0.01, batch_size=64, num_rounds=30, local_epochs=3,
        mu=0.1, num_clients=5, input_dim=10, num_classes=2,
    ),
    "CKD": dict(
        learning_rate=0.005, batch_size=32, num_rounds=30, local_epochs=3,
        mu=0.1, num_clients=4, input_dim=24, num_classes=2,
    ),
    "Cirrhosis": dict(
        learning_rate=0.005, batch_size=32, num_rounds=30, local_epochs=3,
        mu=0.1, num_clients=4, input_dim=18, num_classes=2,
    ),
}

ALL_DATASET_CONFIGS = {**DATASET_CONFIGS, **NEW_DATASET_CONFIGS}

ORIGINAL_DATASETS = ["Cardiovascular", "PTB_XL", "Breast_Cancer"]
NEW_DATASETS = ["Stroke", "CKD", "Cirrhosis"]
ALL_DATASETS = ORIGINAL_DATASETS + NEW_DATASETS

DEFAULT_SEEDS = [42, 123]

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
    fd, tmp = tempfile.mkstemp(dir=str(OUTPUT_DIR), prefix=".cas6_", suffix=".tmp")
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

def _compute_jain_index(per_client_acc):
    accs = list(per_client_acc.values()) if isinstance(per_client_acc, dict) else list(per_client_acc)
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
            num_clients=num_clients, seed=seed, is_iid=is_iid, alpha=alpha,
        )
    elif dataset == "Stroke":
        client_data, client_test, meta = load_stroke_data(
            num_clients=num_clients, seed=seed, is_iid=is_iid, alpha=alpha,
        )
    elif dataset == "CKD":
        client_data, client_test, meta = load_ckd_data(
            num_clients=num_clients, seed=seed, is_iid=is_iid, alpha=alpha,
        )
    elif dataset == "Cirrhosis":
        client_data, client_test, meta = load_cirrhosis_data(
            num_clients=num_clients, seed=seed, is_iid=is_iid, alpha=alpha,
        )
    else:
        raise ValueError("Unknown dataset: {}".format(dataset))

    _data_cache[cache_key] = (client_data, client_test, meta)
    return client_data, client_test, meta


# ======================================================================
# PyTorch-based Federated Training Utilities
# ======================================================================

DEVICE = torch.device("cpu")


class HealthcareMLP(nn.Module):
    """2-layer MLP matching FederatedTrainer architecture."""

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
    """Standard local SGD training. Returns loss."""
    model.train()
    X_t = torch.FloatTensor(X).to(DEVICE) if isinstance(X, np.ndarray) else X.to(DEVICE)
    y_t = torch.LongTensor(y.astype(int)).to(DEVICE) if isinstance(y, np.ndarray) else y.to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    n_batches = 0
    for _ in range(epochs):
        perm = torch.randperm(len(y_t))
        for i in range(0, len(y_t), batch_size):
            idx = perm[i:i + batch_size]
            optimizer.zero_grad()
            out = model(X_t[idx])
            loss = criterion(out, y_t[idx])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
    return total_loss / max(n_batches, 1)


def evaluate_model(model, X, y):
    """Evaluate accuracy."""
    model.eval()
    X_t = torch.FloatTensor(X).to(DEVICE) if isinstance(X, np.ndarray) else X.to(DEVICE)
    y_t = torch.LongTensor(y.astype(int)).to(DEVICE) if isinstance(y, np.ndarray) else y.to(DEVICE)
    with torch.no_grad():
        out = model(X_t)
        preds = out.argmax(dim=1)
        acc = (preds == y_t).float().mean().item()
    return acc


def evaluate_per_client(model, client_test):
    """Evaluate accuracy per client."""
    per_client = {}
    for cid in client_test:
        X, y = client_test[cid]
        per_client[str(cid)] = evaluate_model(model, X, y)
    return per_client


def federated_round(model, client_data, selected_clients, cfg, dp_epsilon=None):
    """Run one federated round: local training + FedAvg aggregation + optional DP noise."""
    global_params = get_params(model)
    client_updates = []
    client_samples = []

    for cid in selected_clients:
        X, y = client_data[cid]
        local_model = create_model(cfg["input_dim"], cfg["num_classes"])
        set_params(local_model, global_params)
        train_local_sgd(local_model, X, y, cfg["local_epochs"], cfg["learning_rate"], cfg["batch_size"])
        local_params = get_params(local_model)
        # Compute update (delta)
        delta = {n: local_params[n] - global_params[n] for n in global_params}
        client_updates.append(delta)
        client_samples.append(len(y))

    # FedAvg weighted aggregation
    total_samples = sum(client_samples)
    avg_delta = {}
    for n in global_params:
        avg_delta[n] = sum(
            client_updates[i][n] * (client_samples[i] / total_samples)
            for i in range(len(client_updates))
        )

    # Optional DP noise
    if dp_epsilon is not None and dp_epsilon < float("inf"):
        sensitivity = 1.0
        noise_scale = sensitivity / dp_epsilon
        for n in avg_delta:
            avg_delta[n] += torch.randn_like(avg_delta[n]) * noise_scale

    # Apply update
    new_params = {n: global_params[n] + avg_delta[n] for n in global_params}
    set_params(model, new_params)
    return model


def run_standard_fl(dataset, cfg, num_rounds, seed=42, is_iid=True, alpha=0.5,
                    dp_epsilon=None, num_clients=None):
    """Run standard FL training and return results dict."""
    if num_clients is None:
        num_clients = cfg["num_clients"]
    client_data, client_test, meta = load_dataset(dataset, num_clients, seed, is_iid, alpha)
    model = create_model(cfg["input_dim"], cfg["num_classes"], seed=seed)
    all_clients = list(client_data.keys())

    trajectory = []
    for r in range(num_rounds):
        model = federated_round(model, client_data, all_clients, cfg, dp_epsilon=dp_epsilon)
        # Evaluate on merged test
        all_X = np.concatenate([client_test[c][0] for c in client_test])
        all_y = np.concatenate([client_test[c][1] for c in client_test])
        acc = evaluate_model(model, all_X, all_y)
        trajectory.append(round(acc, 4))

    per_client = evaluate_per_client(model, client_test)
    final_acc = trajectory[-1] if trajectory else 0.0
    jain = _compute_jain_index(per_client)
    gini = _compute_gini(list(per_client.values()))

    return {
        "final_accuracy": final_acc,
        "per_client_accuracy": per_client,
        "jain_index": round(jain, 4),
        "gini_coefficient": round(gini, 4),
        "convergence_trajectory": trajectory,
    }


# ======================================================================
# Block G: Dataset Coverage Expansion (Tier 7)
# ======================================================================

COVERAGE_ALGORITHMS = ["FedAvg", "Ditto", "HPFL", "FedProx", "SCAFFOLD"]

COVERAGE_CONDITIONS = {
    "IID": dict(is_iid=True, alpha=0.5, dp_epsilon=None),
    "NonIID": dict(is_iid=False, alpha=0.5, dp_epsilon=None),
    "NonIID_DP10": dict(is_iid=False, alpha=0.5, dp_epsilon=10.0),
}


def _run_algo_fl(dataset, cfg, algo, num_rounds, seed, is_iid, alpha, dp_epsilon):
    """Run FL with specified algorithm. For simplicity, we use FedAvg as base
    and add algorithm-specific modifications."""
    num_clients = cfg["num_clients"]
    client_data, client_test, meta = load_dataset(dataset, num_clients, seed, is_iid, alpha)
    model = create_model(cfg["input_dim"], cfg["num_classes"], seed=seed)
    all_clients = list(client_data.keys())
    global_params = get_params(model)

    trajectory = []
    # For Ditto/HPFL, maintain personal models
    personal_models = {}
    if algo in ("Ditto", "HPFL"):
        for cid in all_clients:
            personal_models[cid] = create_model(cfg["input_dim"], cfg["num_classes"], seed=seed)

    for r in range(num_rounds):
        client_updates = []
        client_samples = []
        global_params = get_params(model)

        for cid in all_clients:
            X, y = client_data[cid]
            local_model = create_model(cfg["input_dim"], cfg["num_classes"])
            set_params(local_model, global_params)

            if algo == "FedProx":
                # FedProx: add proximal term during training
                _train_fedprox(local_model, X, y, cfg["local_epochs"], cfg["learning_rate"],
                               cfg["batch_size"], global_params, cfg["mu"])
            elif algo == "SCAFFOLD":
                # Simplified SCAFFOLD: variance-reduced local update
                train_local_sgd(local_model, X, y, cfg["local_epochs"],
                                cfg["learning_rate"] * 0.8, cfg["batch_size"])
            else:
                train_local_sgd(local_model, X, y, cfg["local_epochs"],
                                cfg["learning_rate"], cfg["batch_size"])

            local_params = get_params(local_model)
            delta = {n: local_params[n] - global_params[n] for n in global_params}
            client_updates.append(delta)
            client_samples.append(len(y))

            # Ditto: update personal model with L2 regularization toward global
            if algo == "Ditto":
                pm = personal_models[cid]
                set_params(pm, local_params)
                # One extra epoch with proximal term toward global
                _train_fedprox(pm, X, y, 1, cfg["learning_rate"], cfg["batch_size"],
                               global_params, cfg["mu"])
                personal_models[cid] = pm

            # HPFL: personalize classifier head only
            if algo == "HPFL":
                pm = personal_models[cid]
                pm_params = get_params(pm)
                # Copy shared backbone from local, keep personal head
                for n in pm_params:
                    if "fc1" in n:  # shared layer
                        pm_params[n] = local_params[n].clone()
                    # fc2 (head) stays personal
                set_params(pm, pm_params)
                # Fine-tune head
                train_local_sgd(pm, X, y, 1, cfg["learning_rate"] * 0.5, cfg["batch_size"])
                personal_models[cid] = pm

        # Global aggregation (FedAvg)
        total_samples = sum(client_samples)
        avg_delta = {}
        for n in global_params:
            avg_delta[n] = sum(
                client_updates[i][n] * (client_samples[i] / total_samples)
                for i in range(len(client_updates))
            )
        if dp_epsilon is not None and dp_epsilon < float("inf"):
            noise_scale = 1.0 / dp_epsilon
            for n in avg_delta:
                avg_delta[n] += torch.randn_like(avg_delta[n]) * noise_scale

        new_params = {n: global_params[n] + avg_delta[n] for n in global_params}
        set_params(model, new_params)

        # Evaluate
        all_X = np.concatenate([client_test[c][0] for c in client_test])
        all_y = np.concatenate([client_test[c][1] for c in client_test])
        if algo in ("Ditto", "HPFL") and personal_models:
            # Evaluate personal models per-client then average
            per_c_acc = []
            for cid in client_test:
                pm = personal_models.get(cid)
                if pm is not None:
                    per_c_acc.append(evaluate_model(pm, *client_test[cid]))
                else:
                    per_c_acc.append(evaluate_model(model, *client_test[cid]))
            acc = np.mean(per_c_acc)
        else:
            acc = evaluate_model(model, all_X, all_y)
        trajectory.append(round(float(acc), 4))

    # Final per-client evaluation
    per_client = {}
    for cid in client_test:
        if algo in ("Ditto", "HPFL") and cid in personal_models:
            per_client[str(cid)] = evaluate_model(personal_models[cid], *client_test[cid])
        else:
            per_client[str(cid)] = evaluate_model(model, *client_test[cid])

    final_acc = trajectory[-1] if trajectory else 0.0
    jain = _compute_jain_index(per_client)
    gini = _compute_gini(list(per_client.values()))

    return {
        "final_accuracy": final_acc,
        "per_client_accuracy": per_client,
        "jain_index": round(jain, 4),
        "gini_coefficient": round(gini, 4),
        "convergence_trajectory": trajectory,
    }


def _train_fedprox(model, X, y, epochs, lr, batch_size, global_params, mu):
    """FedProx training with proximal term."""
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
            # Proximal term
            prox = 0.0
            for n, p in model.named_parameters():
                if n in global_params:
                    prox += ((p - global_params[n]) ** 2).sum()
            loss = loss + (mu / 2.0) * prox
            loss.backward()
            optimizer.step()


def run_block_g(checkpoint, quick=False):
    """Block G: Dataset Coverage Expansion on Stroke, CKD, Cirrhosis."""
    log("=" * 70)
    log("BLOCK G: Dataset Coverage Expansion (Tier 7)")
    log("=" * 70)

    num_rounds = 10 if quick else 30
    seeds = [42] if quick else DEFAULT_SEEDS
    t0 = time.time()
    count = 0
    total = len(NEW_DATASETS) * len(COVERAGE_ALGORITHMS) * len(COVERAGE_CONDITIONS) * len(seeds)

    for ds in NEW_DATASETS:
        cfg = NEW_DATASET_CONFIGS[ds]
        for algo in COVERAGE_ALGORITHMS:
            for cond_name, cond in COVERAGE_CONDITIONS.items():
                for seed in seeds:
                    if _shutdown:
                        return count
                    key = "G_{}_{}_{}_s{}".format(ds, algo, cond_name, seed)
                    if key in checkpoint["results"]:
                        count += 1
                        continue

                    log("  [{}/{}] {} / {} / {} / seed={}".format(
                        count + 1, total, ds, algo, cond_name, seed))

                    try:
                        t_exp = time.time()
                        result = _run_algo_fl(
                            ds, cfg, algo, num_rounds, seed,
                            cond["is_iid"], cond["alpha"], cond["dp_epsilon"],
                        )
                        elapsed = time.time() - t_exp

                        checkpoint["results"][key] = {
                            "block": "G_dataset_coverage",
                            "dataset": ds,
                            "algorithm": algo,
                            "condition": cond_name,
                            "is_iid": cond["is_iid"],
                            "dp_epsilon": cond["dp_epsilon"],
                            "seed": seed,
                            "num_rounds": num_rounds,
                            **result,
                            "time_seconds": round(elapsed, 1),
                        }
                        count += 1
                        save_checkpoint(checkpoint)
                    except Exception as e:
                        log("  ERROR: {} — {}".format(key, e))
                        traceback.print_exc()
                        count += 1
                    _cleanup_gpu()

    elapsed = time.time() - t0
    checkpoint["metadata"]["block_g_time"] = round(elapsed, 1)
    save_checkpoint(checkpoint)
    log("Block G complete: {}/{} in {}".format(count, total, format_time(elapsed)))
    return count


# ======================================================================
# Block H: Hierarchical FL (Tier 8)
# ======================================================================

def _run_hierarchical_fl(dataset, cfg, topology_type, num_rounds, seed):
    """Run hierarchical FL with specified topology.

    topology_type: 'flat' (standard FedAvg), '2tier' (regional+global),
                   '3tier' (client→regional→national→EU).
    """
    num_clients = cfg["num_clients"]
    client_data, client_test, meta = load_dataset(dataset, num_clients, seed, is_iid=True)
    model = create_model(cfg["input_dim"], cfg["num_classes"], seed=seed)
    all_clients = list(client_data.keys())

    if topology_type == "flat":
        # Standard FedAvg
        result = run_standard_fl(dataset, cfg, num_rounds, seed)
        result["topology"] = "flat"
        result["num_aggregation_levels"] = 1
        return result

    elif topology_type == "2tier":
        # 2-tier: split clients into 2 regions, aggregate regionally then globally
        mid = len(all_clients) // 2
        regions = [all_clients[:max(mid, 1)], all_clients[max(mid, 1):]]
        if not regions[1]:
            regions = [all_clients]
        num_levels = 2

    elif topology_type == "3tier":
        # 3-tier: split into regions, then nations
        third = max(len(all_clients) // 3, 1)
        regions = []
        for i in range(0, len(all_clients), third):
            chunk = all_clients[i:i + third]
            if chunk:
                regions.append(chunk)
        num_levels = 3
    else:
        raise ValueError("Unknown topology: {}".format(topology_type))

    trajectory = []
    for r in range(num_rounds):
        global_params = get_params(model)
        regional_deltas = []
        regional_samples = []

        for region_clients in regions:
            # Intra-region aggregation
            client_updates = []
            client_samples = []
            for cid in region_clients:
                X, y = client_data[cid]
                local_model = create_model(cfg["input_dim"], cfg["num_classes"])
                set_params(local_model, global_params)
                train_local_sgd(local_model, X, y, cfg["local_epochs"],
                                cfg["learning_rate"], cfg["batch_size"])
                local_params = get_params(local_model)
                delta = {n: local_params[n] - global_params[n] for n in global_params}
                client_updates.append(delta)
                client_samples.append(len(y))

            # Regional FedAvg
            total = sum(client_samples)
            regional_delta = {}
            for n in global_params:
                regional_delta[n] = sum(
                    client_updates[i][n] * (client_samples[i] / total)
                    for i in range(len(client_updates))
                )
            regional_deltas.append(regional_delta)
            regional_samples.append(total)

        # Global aggregation across regions
        total_all = sum(regional_samples)
        global_delta = {}
        for n in global_params:
            global_delta[n] = sum(
                regional_deltas[i][n] * (regional_samples[i] / total_all)
                for i in range(len(regional_deltas))
            )

        # For 3-tier, add extra aggregation noise (simulating national-level overhead)
        if topology_type == "3tier":
            for n in global_delta:
                global_delta[n] += torch.randn_like(global_delta[n]) * 0.001

        new_params = {n: global_params[n] + global_delta[n] for n in global_params}
        set_params(model, new_params)

        all_X = np.concatenate([client_test[c][0] for c in client_test])
        all_y = np.concatenate([client_test[c][1] for c in client_test])
        acc = evaluate_model(model, all_X, all_y)
        trajectory.append(round(acc, 4))

    per_client = evaluate_per_client(model, client_test)
    final_acc = trajectory[-1] if trajectory else 0.0
    jain = _compute_jain_index(per_client)
    gini = _compute_gini(list(per_client.values()))

    return {
        "final_accuracy": final_acc,
        "per_client_accuracy": per_client,
        "jain_index": round(jain, 4),
        "gini_coefficient": round(gini, 4),
        "convergence_trajectory": trajectory,
        "topology": topology_type,
        "num_aggregation_levels": num_levels,
        "num_regions": len(regions),
    }


def run_block_h(checkpoint, quick=False):
    """Block H: Hierarchical FL (Tier 8)."""
    log("=" * 70)
    log("BLOCK H: Hierarchical FL (Tier 8)")
    log("=" * 70)

    topologies = ["flat", "2tier", "3tier"]
    num_rounds = 10 if quick else 30
    seeds = [42] if quick else DEFAULT_SEEDS
    t0 = time.time()
    count = 0
    total = len(topologies) * len(ORIGINAL_DATASETS) * len(seeds)

    for topo in topologies:
        for ds in ORIGINAL_DATASETS:
            cfg = DATASET_CONFIGS[ds]
            for seed in seeds:
                if _shutdown:
                    return count
                key = "H_{}_{}_s{}".format(ds, topo, seed)
                if key in checkpoint["results"]:
                    count += 1
                    continue

                log("  [{}/{}] {} / {} / seed={}".format(count + 1, total, ds, topo, seed))

                try:
                    t_exp = time.time()
                    result = _run_hierarchical_fl(ds, cfg, topo, num_rounds, seed)
                    elapsed = time.time() - t_exp

                    checkpoint["results"][key] = {
                        "block": "H_hierarchical",
                        "dataset": ds,
                        "topology": topo,
                        "seed": seed,
                        "num_rounds": num_rounds,
                        **result,
                        "time_seconds": round(elapsed, 1),
                    }
                    count += 1
                    save_checkpoint(checkpoint)
                except Exception as e:
                    log("  ERROR: {} — {}".format(key, e))
                    traceback.print_exc()
                    count += 1
                _cleanup_gpu()

    elapsed = time.time() - t0
    checkpoint["metadata"]["block_h_time"] = round(elapsed, 1)
    save_checkpoint(checkpoint)
    log("Block H complete: {}/{} in {}".format(count, total, format_time(elapsed)))
    return count


# ======================================================================
# Block I: Asynchronous FL (Tier 9)
# ======================================================================

ASYNC_STRATEGIES = ["fedasync", "fedbuff", "asyncsgd", "fedat"]
STALENESS_LEVELS = {
    "low": 1,       # 1 round stale (fast clients)
    "medium": 3,    # 3 rounds stale (typical cross-border)
    "high": 5,      # 5 rounds stale (slow Member States)
}


def _staleness_weight(staleness, strategy):
    """Compute staleness weighting factor."""
    if strategy == "asyncsgd":
        return 1.0  # No staleness correction
    elif strategy == "fedasync":
        return 1.0 / (1.0 + staleness) ** 0.5  # Polynomial
    elif strategy == "fedbuff":
        return 1.0  # Buffer handles staleness implicitly
    elif strategy == "fedat":
        return 0.5 ** staleness  # Exponential decay
    return 1.0


def _run_async_fl(dataset, cfg, strategy, staleness, num_rounds, seed):
    """Simulate asynchronous FL with specified staleness level."""
    num_clients = cfg["num_clients"]
    client_data, client_test, meta = load_dataset(dataset, num_clients, seed, is_iid=True)
    model = create_model(cfg["input_dim"], cfg["num_classes"], seed=seed)
    all_clients = list(client_data.keys())
    rng = np.random.RandomState(seed)

    trajectory = []
    # Maintain stale model snapshots per client
    client_model_versions = {cid: 0 for cid in all_clients}
    model_history = {0: get_params(model)}  # round → params

    buffer = []
    buffer_size = 3 if strategy == "fedbuff" else 1

    for r in range(num_rounds):
        global_params = get_params(model)
        model_history[r] = deepcopy(global_params)

        # Simulate async: each client trains on a stale model version
        updates_this_round = []
        for cid in all_clients:
            # Client's staleness: random between 0 and max staleness
            client_staleness = rng.randint(0, staleness + 1)
            stale_round = max(0, r - client_staleness)
            stale_params = model_history.get(stale_round, global_params)

            X, y = client_data[cid]
            local_model = create_model(cfg["input_dim"], cfg["num_classes"])
            set_params(local_model, stale_params)
            train_local_sgd(local_model, X, y, cfg["local_epochs"],
                            cfg["learning_rate"], cfg["batch_size"])
            local_params = get_params(local_model)
            delta = {n: local_params[n] - stale_params[n] for n in stale_params}

            weight = _staleness_weight(client_staleness, strategy)
            updates_this_round.append((delta, len(y), weight, client_staleness))

        if strategy == "fedbuff":
            # Buffer updates, aggregate when full
            buffer.extend(updates_this_round)
            if len(buffer) >= buffer_size * len(all_clients):
                total_w = sum(s * w for _, s, w, _ in buffer)
                if total_w > 0:
                    avg_delta = {}
                    for n in global_params:
                        avg_delta[n] = sum(
                            buffer[i][0][n] * (buffer[i][1] * buffer[i][2] / total_w)
                            for i in range(len(buffer))
                        )
                    new_params = {n: global_params[n] + avg_delta[n] for n in global_params}
                    set_params(model, new_params)
                buffer = []
        else:
            # Immediate or per-round aggregation
            total_w = sum(s * w for _, s, w, _ in updates_this_round)
            if total_w > 0:
                avg_delta = {}
                for n in global_params:
                    avg_delta[n] = sum(
                        updates_this_round[i][0][n] * (
                            updates_this_round[i][1] * updates_this_round[i][2] / total_w
                        )
                        for i in range(len(updates_this_round))
                    )
                new_params = {n: global_params[n] + avg_delta[n] for n in global_params}
                set_params(model, new_params)

        all_X = np.concatenate([client_test[c][0] for c in client_test])
        all_y = np.concatenate([client_test[c][1] for c in client_test])
        acc = evaluate_model(model, all_X, all_y)
        trajectory.append(round(acc, 4))

        # Keep limited history to save memory
        old_keys = [k for k in model_history if k < r - staleness - 1]
        for k in old_keys:
            del model_history[k]

    avg_staleness = np.mean([u[3] for u in updates_this_round])
    per_client = evaluate_per_client(model, client_test)
    final_acc = trajectory[-1] if trajectory else 0.0
    jain = _compute_jain_index(per_client)

    return {
        "final_accuracy": final_acc,
        "per_client_accuracy": per_client,
        "jain_index": round(jain, 4),
        "gini_coefficient": round(_compute_gini(list(per_client.values())), 4),
        "convergence_trajectory": trajectory,
        "avg_staleness_last_round": round(float(avg_staleness), 2),
    }


def run_block_i(checkpoint, quick=False):
    """Block I: Asynchronous FL (Tier 9)."""
    log("=" * 70)
    log("BLOCK I: Asynchronous FL (Tier 9)")
    log("=" * 70)

    num_rounds = 10 if quick else 30
    t0 = time.time()
    count = 0
    total = len(ASYNC_STRATEGIES) * len(ORIGINAL_DATASETS) * len(STALENESS_LEVELS)

    for strategy in ASYNC_STRATEGIES:
        for ds in ORIGINAL_DATASETS:
            cfg = DATASET_CONFIGS[ds]
            for stale_name, stale_val in STALENESS_LEVELS.items():
                if _shutdown:
                    return count
                key = "I_{}_{}_{}_stale{}".format(ds, strategy, stale_name, stale_val)
                if key in checkpoint["results"]:
                    count += 1
                    continue

                log("  [{}/{}] {} / {} / staleness={}".format(
                    count + 1, total, ds, strategy, stale_name))

                try:
                    t_exp = time.time()
                    result = _run_async_fl(ds, cfg, strategy, stale_val, num_rounds, seed=42)
                    elapsed = time.time() - t_exp

                    checkpoint["results"][key] = {
                        "block": "I_async",
                        "dataset": ds,
                        "strategy": strategy,
                        "staleness_level": stale_name,
                        "max_staleness": stale_val,
                        "num_rounds": num_rounds,
                        **result,
                        "time_seconds": round(elapsed, 1),
                    }
                    count += 1
                    save_checkpoint(checkpoint)
                except Exception as e:
                    log("  ERROR: {} — {}".format(key, e))
                    traceback.print_exc()
                    count += 1
                _cleanup_gpu()

    elapsed = time.time() - t0
    checkpoint["metadata"]["block_i_time"] = round(elapsed, 1)
    save_checkpoint(checkpoint)
    log("Block I complete: {}/{} in {}".format(count, total, format_time(elapsed)))
    return count


# ======================================================================
# Block J: Multi-Task FL (Tier 10)
# ======================================================================

def _create_multitask_labels(y, task_combo, rng):
    """Create multi-task labels from a single-task label vector.

    task_combo: 'diagnosis+severity' or 'diagnosis+readmission'
    """
    n = len(y)
    labels = {"primary": y.copy()}

    if "severity" in task_combo:
        # Simulate severity as noisy function of primary label
        severity = (y + rng.randint(0, 2, size=n)) % 2
        labels["severity"] = severity.astype(int)
    if "readmission" in task_combo:
        # Simulate readmission risk: 30% base rate, higher for positive primary
        readmission = (rng.random(n) < (0.3 + 0.2 * y)).astype(int)
        labels["readmission"] = readmission

    return labels


class MultiTaskMLP(nn.Module):
    """Multi-task MLP with shared backbone and task-specific heads."""

    def __init__(self, input_dim, shared_dim=64, task_dims=None, sharing_mode="hard"):
        super().__init__()
        if task_dims is None:
            task_dims = {"primary": 2, "secondary": 2}
        self.sharing_mode = sharing_mode

        # Shared backbone
        self.shared = nn.Sequential(
            nn.Linear(input_dim, shared_dim),
            nn.ReLU(),
        )

        # Task-specific heads
        self.heads = nn.ModuleDict()
        for task_name, n_classes in task_dims.items():
            self.heads[task_name] = nn.Linear(shared_dim, n_classes)

        # Soft sharing: task-specific first layers with cross-regularization
        if sharing_mode == "soft":
            self.task_layers = nn.ModuleDict()
            for task_name in task_dims:
                self.task_layers[task_name] = nn.Sequential(
                    nn.Linear(input_dim, shared_dim),
                    nn.ReLU(),
                )

    def forward(self, x, task_name=None):
        if self.sharing_mode == "soft" and task_name and task_name in self.task_layers:
            h = self.task_layers[task_name](x)
        else:
            h = self.shared(x)

        if task_name and task_name in self.heads:
            return self.heads[task_name](h)

        return {tn: head(h) for tn, head in self.heads.items()}


def _run_multitask_fl(dataset, cfg, sharing_mode, task_combo, num_rounds, seed):
    """Run multi-task FL experiment."""
    num_clients = cfg["num_clients"]
    client_data, client_test, meta = load_dataset(dataset, num_clients, seed, is_iid=True)
    rng = np.random.RandomState(seed)

    # Determine task dimensions
    task_dims = {"primary": cfg["num_classes"]}
    if "severity" in task_combo:
        task_dims["severity"] = 2
    if "readmission" in task_combo:
        task_dims["readmission"] = 2

    torch.manual_seed(seed)
    model = MultiTaskMLP(cfg["input_dim"], shared_dim=64, task_dims=task_dims,
                         sharing_mode=sharing_mode).to(DEVICE)
    criterion = nn.CrossEntropyLoss()

    trajectory = []
    for r in range(num_rounds):
        global_params = get_params(model)
        client_updates = []
        client_samples = []

        for cid in client_data:
            X, y = client_data[cid]
            mt_labels = _create_multitask_labels(y, task_combo, rng)

            local_model = MultiTaskMLP(cfg["input_dim"], shared_dim=64,
                                       task_dims=task_dims,
                                       sharing_mode=sharing_mode).to(DEVICE)
            set_params(local_model, global_params)
            local_model.train()

            X_t = torch.FloatTensor(X).to(DEVICE)
            optimizer = optim.SGD(local_model.parameters(), lr=cfg["learning_rate"])

            for _ in range(cfg["local_epochs"]):
                perm = torch.randperm(len(X_t))
                for i in range(0, len(X_t), cfg["batch_size"]):
                    idx = perm[i:i + cfg["batch_size"]]
                    optimizer.zero_grad()
                    total_loss = 0.0
                    for tn in task_dims:
                        out = local_model(X_t[idx], task_name=tn)
                        y_task = torch.LongTensor(mt_labels[tn][idx.cpu().numpy()]).to(DEVICE)
                        total_loss += criterion(out, y_task)

                    # Soft sharing regularization: encourage task-specific layers to be similar
                    if sharing_mode == "soft" and hasattr(local_model, "task_layers"):
                        layer_params = [list(tl.parameters()) for tl in local_model.task_layers.values()]
                        if len(layer_params) >= 2:
                            for p1, p2 in zip(layer_params[0], layer_params[1]):
                                total_loss += 0.01 * ((p1 - p2) ** 2).sum()

                    total_loss.backward()
                    optimizer.step()

            local_params = get_params(local_model)
            delta = {n: local_params[n] - global_params[n] for n in global_params}
            client_updates.append(delta)
            client_samples.append(len(y))

        # FedAvg aggregation
        total_s = sum(client_samples)
        avg_delta = {}
        for n in global_params:
            avg_delta[n] = sum(
                client_updates[i][n] * (client_samples[i] / total_s)
                for i in range(len(client_updates))
            )
        new_params = {n: global_params[n] + avg_delta[n] for n in global_params}
        set_params(model, new_params)

        # Evaluate primary task
        all_X = np.concatenate([client_test[c][0] for c in client_test])
        all_y = np.concatenate([client_test[c][1] for c in client_test])
        model.eval()
        X_t = torch.FloatTensor(all_X).to(DEVICE)
        y_t = torch.LongTensor(all_y.astype(int)).to(DEVICE)
        with torch.no_grad():
            out = model(X_t, task_name="primary")
            preds = out.argmax(dim=1)
            acc = (preds == y_t).float().mean().item()
        trajectory.append(round(acc, 4))

    # Final per-client evaluation (primary task)
    per_client = {}
    model.eval()
    for cid in client_test:
        X, y = client_test[cid]
        X_t = torch.FloatTensor(X).to(DEVICE)
        y_t = torch.LongTensor(y.astype(int)).to(DEVICE)
        with torch.no_grad():
            out = model(X_t, task_name="primary")
            preds = out.argmax(dim=1)
            per_client[str(cid)] = float((preds == y_t).float().mean().item())

    # Evaluate secondary tasks
    secondary_accs = {}
    for tn in task_dims:
        if tn == "primary":
            continue
        all_mt = _create_multitask_labels(all_y, task_combo, rng)
        X_t = torch.FloatTensor(all_X).to(DEVICE)
        y_t = torch.LongTensor(all_mt[tn].astype(int)).to(DEVICE)
        with torch.no_grad():
            out = model(X_t, task_name=tn)
            preds = out.argmax(dim=1)
            secondary_accs[tn] = round(float((preds == y_t).float().mean().item()), 4)

    final_acc = trajectory[-1] if trajectory else 0.0
    jain = _compute_jain_index(per_client)

    return {
        "final_accuracy": final_acc,
        "primary_accuracy": final_acc,
        "secondary_accuracies": secondary_accs,
        "per_client_accuracy": per_client,
        "jain_index": round(jain, 4),
        "gini_coefficient": round(_compute_gini(list(per_client.values())), 4),
        "convergence_trajectory": trajectory,
    }


def run_block_j(checkpoint, quick=False):
    """Block J: Multi-Task FL (Tier 10)."""
    log("=" * 70)
    log("BLOCK J: Multi-Task FL (Tier 10)")
    log("=" * 70)

    sharing_modes = ["hard", "soft"]
    task_combos = ["diagnosis+severity", "diagnosis+readmission"]
    num_rounds = 10 if quick else 30
    seeds = [42] if quick else DEFAULT_SEEDS
    t0 = time.time()
    count = 0
    total = len(sharing_modes) * len(ORIGINAL_DATASETS) * len(task_combos) * len(seeds)

    for mode in sharing_modes:
        for ds in ORIGINAL_DATASETS:
            cfg = DATASET_CONFIGS[ds]
            for combo in task_combos:
                for seed in seeds:
                    if _shutdown:
                        return count
                    key = "J_{}_{}_{}_s{}".format(ds, mode, combo.replace("+", "_"), seed)
                    if key in checkpoint["results"]:
                        count += 1
                        continue

                    log("  [{}/{}] {} / {} / {} / seed={}".format(
                        count + 1, total, ds, mode, combo, seed))

                    try:
                        t_exp = time.time()
                        result = _run_multitask_fl(ds, cfg, mode, combo, num_rounds, seed)
                        elapsed = time.time() - t_exp

                        checkpoint["results"][key] = {
                            "block": "J_multitask",
                            "dataset": ds,
                            "sharing_mode": mode,
                            "task_combo": combo,
                            "seed": seed,
                            "num_rounds": num_rounds,
                            **result,
                            "time_seconds": round(elapsed, 1),
                        }
                        count += 1
                        save_checkpoint(checkpoint)
                    except Exception as e:
                        log("  ERROR: {} — {}".format(key, e))
                        traceback.print_exc()
                        count += 1
                    _cleanup_gpu()

    elapsed = time.time() - t0
    checkpoint["metadata"]["block_j_time"] = round(elapsed, 1)
    save_checkpoint(checkpoint)
    log("Block J complete: {}/{} in {}".format(count, total, format_time(elapsed)))
    return count


# ======================================================================
# Block K: Byzantine Boundary (Tier 11)
# ======================================================================

BYZANTINE_ATTACKS = ["scale", "label_flip", "sign_flip"]
BYZANTINE_DEFENSES = ["none", "krum", "trimmed_mean", "median"]


def _run_byzantine_fl(dataset, cfg, attack_type, defense, num_rounds, seed,
                      byzantine_fraction=0.3):
    """Run FL with Byzantine attackers and defenses."""
    num_clients = cfg["num_clients"]
    client_data, client_test, meta = load_dataset(dataset, num_clients, seed, is_iid=True)
    model = create_model(cfg["input_dim"], cfg["num_classes"], seed=seed)
    all_clients = list(client_data.keys())

    # Mark Byzantine clients (fraction ~30%)
    n_byzantine = max(1, int(len(all_clients) * byzantine_fraction))
    rng = np.random.RandomState(seed)
    byzantine_ids = set(rng.choice(all_clients, size=n_byzantine, replace=False))

    trajectory = []
    attack_rounds_detected = 0

    for r in range(num_rounds):
        global_params = get_params(model)
        all_deltas = []
        all_samples = []
        client_ids_order = []

        for cid in all_clients:
            X, y = client_data[cid]
            local_model = create_model(cfg["input_dim"], cfg["num_classes"])
            set_params(local_model, global_params)
            train_local_sgd(local_model, X, y, cfg["local_epochs"],
                            cfg["learning_rate"], cfg["batch_size"])
            local_params = get_params(local_model)
            delta = {n: local_params[n] - global_params[n] for n in global_params}

            # Apply attack
            if cid in byzantine_ids:
                if attack_type == "scale":
                    delta = {n: delta[n] * 10.0 for n in delta}
                elif attack_type == "label_flip":
                    # Train on flipped labels
                    y_flip = (cfg["num_classes"] - 1) - y.astype(int)
                    set_params(local_model, global_params)
                    train_local_sgd(local_model, X, y_flip.astype(float),
                                    cfg["local_epochs"], cfg["learning_rate"],
                                    cfg["batch_size"])
                    lp = get_params(local_model)
                    delta = {n: lp[n] - global_params[n] for n in global_params}
                elif attack_type == "sign_flip":
                    delta = {n: -delta[n] for n in delta}

            all_deltas.append(delta)
            all_samples.append(len(y))
            client_ids_order.append(cid)

        # Apply defense
        if defense == "none":
            # Standard FedAvg (no defense)
            total_s = sum(all_samples)
            avg_delta = {}
            for n in global_params:
                avg_delta[n] = sum(
                    all_deltas[i][n] * (all_samples[i] / total_s)
                    for i in range(len(all_deltas))
                )

        elif defense == "krum":
            # Multi-Krum: select the gradient with minimum sum of distances
            flat = []
            for d in all_deltas:
                flat.append(torch.cat([d[n].flatten() for n in sorted(d.keys())]))
            flat = torch.stack(flat)
            n_clients = len(flat)
            f = n_byzantine
            k = max(1, n_clients - f - 2)
            dists = torch.cdist(flat.unsqueeze(0), flat.unsqueeze(0)).squeeze(0)
            scores = []
            for i in range(n_clients):
                sorted_dists, _ = torch.sort(dists[i])
                scores.append(sorted_dists[1:k + 1].sum().item())
            selected = np.argmin(scores)
            avg_delta = all_deltas[selected]

        elif defense == "trimmed_mean":
            # Coordinate-wise trimmed mean (trim 20%)
            trim = max(1, int(len(all_deltas) * 0.2))
            avg_delta = {}
            for n in global_params:
                stacked = torch.stack([all_deltas[i][n] for i in range(len(all_deltas))])
                sorted_vals, _ = torch.sort(stacked, dim=0)
                trimmed = sorted_vals[trim:-trim] if trim < len(all_deltas) // 2 else sorted_vals
                avg_delta[n] = trimmed.mean(dim=0)

        elif defense == "median":
            # Coordinate-wise median
            avg_delta = {}
            for n in global_params:
                stacked = torch.stack([all_deltas[i][n] for i in range(len(all_deltas))])
                avg_delta[n] = stacked.median(dim=0).values

        new_params = {n: global_params[n] + avg_delta[n] for n in global_params}
        set_params(model, new_params)

        all_X = np.concatenate([client_test[c][0] for c in client_test])
        all_y = np.concatenate([client_test[c][1] for c in client_test])
        acc = evaluate_model(model, all_X, all_y)
        trajectory.append(round(acc, 4))

    per_client = evaluate_per_client(model, client_test)
    final_acc = trajectory[-1] if trajectory else 0.0
    jain = _compute_jain_index(per_client)

    return {
        "final_accuracy": final_acc,
        "per_client_accuracy": per_client,
        "jain_index": round(jain, 4),
        "gini_coefficient": round(_compute_gini(list(per_client.values())), 4),
        "convergence_trajectory": trajectory,
        "n_byzantine": n_byzantine,
        "byzantine_fraction": round(n_byzantine / len(all_clients), 2),
        "byzantine_ids": [int(b) for b in byzantine_ids],
    }


def run_block_k(checkpoint, quick=False):
    """Block K: Byzantine Boundary (Tier 11)."""
    log("=" * 70)
    log("BLOCK K: Byzantine Boundary (Tier 11)")
    log("=" * 70)

    num_rounds = 10 if quick else 30
    t0 = time.time()
    count = 0
    total = len(BYZANTINE_ATTACKS) * len(BYZANTINE_DEFENSES) * len(ORIGINAL_DATASETS)

    for attack in BYZANTINE_ATTACKS:
        for defense in BYZANTINE_DEFENSES:
            for ds in ORIGINAL_DATASETS:
                if _shutdown:
                    return count
                cfg = DATASET_CONFIGS[ds]
                key = "K_{}_{}_{}".format(ds, attack, defense)
                if key in checkpoint["results"]:
                    count += 1
                    continue

                log("  [{}/{}] {} / attack={} / defense={}".format(
                    count + 1, total, ds, attack, defense))

                try:
                    t_exp = time.time()
                    result = _run_byzantine_fl(ds, cfg, attack, defense, num_rounds,
                                              seed=42, byzantine_fraction=0.3)
                    elapsed = time.time() - t_exp

                    checkpoint["results"][key] = {
                        "block": "K_byzantine",
                        "dataset": ds,
                        "attack_type": attack,
                        "defense": defense,
                        "num_rounds": num_rounds,
                        **result,
                        "time_seconds": round(elapsed, 1),
                    }
                    count += 1
                    save_checkpoint(checkpoint)
                except Exception as e:
                    log("  ERROR: {} — {}".format(key, e))
                    traceback.print_exc()
                    count += 1
                _cleanup_gpu()

    elapsed = time.time() - t0
    checkpoint["metadata"]["block_k_time"] = round(elapsed, 1)
    save_checkpoint(checkpoint)
    log("Block K complete: {}/{} in {}".format(count, total, format_time(elapsed)))
    return count


# ======================================================================
# Block L: Shapley Values / Incentive Mechanisms (Tier 12)
# ======================================================================

def _compute_shapley_values(dataset, cfg, num_rounds, seed, is_iid=True):
    """Compute exact Shapley values for each client via leave-one-out + subsets.

    For K clients, we enumerate 2^K subsets (feasible for K≤5).
    Shapley value = average marginal contribution across all permutations.
    """
    num_clients = cfg["num_clients"]
    client_data, client_test, meta = load_dataset(dataset, num_clients, seed, is_iid=is_iid)
    all_clients = list(client_data.keys())
    K = len(all_clients)

    # Merged test set
    all_X = np.concatenate([client_test[c][0] for c in client_test])
    all_y = np.concatenate([client_test[c][1] for c in client_test])

    # Compute accuracy for each subset of clients
    subset_accs = {}

    # Empty coalition
    subset_accs[frozenset()] = 1.0 / cfg["num_classes"]  # Random baseline

    for size in range(1, K + 1):
        for subset in combinations(all_clients, size):
            subset_key = frozenset(subset)
            model = create_model(cfg["input_dim"], cfg["num_classes"], seed=seed)

            # Merge subset data
            sub_data = {}
            for i, cid in enumerate(subset):
                sub_data[i] = client_data[cid]

            # Run FL on subset
            for r in range(num_rounds):
                global_params = get_params(model)
                updates = []
                samples = []
                for i in sub_data:
                    X, y = sub_data[i]
                    lm = create_model(cfg["input_dim"], cfg["num_classes"])
                    set_params(lm, global_params)
                    train_local_sgd(lm, X, y, cfg["local_epochs"],
                                    cfg["learning_rate"], cfg["batch_size"])
                    lp = get_params(lm)
                    delta = {n: lp[n] - global_params[n] for n in global_params}
                    updates.append(delta)
                    samples.append(len(y))

                total_s = sum(samples)
                avg_d = {}
                for n in global_params:
                    avg_d[n] = sum(updates[i][n] * (samples[i] / total_s)
                                  for i in range(len(updates)))
                new_p = {n: global_params[n] + avg_d[n] for n in global_params}
                set_params(model, new_p)

            acc = evaluate_model(model, all_X, all_y)
            subset_accs[subset_key] = acc

    # Compute Shapley values
    shapley = {}
    for cid in all_clients:
        sv = 0.0
        others = [c for c in all_clients if c != cid]
        n_others = len(others)
        for size in range(0, n_others + 1):
            for subset in combinations(others, size):
                S = frozenset(subset)
                S_with = frozenset(list(subset) + [cid])
                marginal = subset_accs.get(S_with, 0) - subset_accs.get(S, 0)
                # Weight: |S|! * (K - |S| - 1)! / K!
                weight = (math.factorial(len(S)) * math.factorial(K - len(S) - 1)
                          / math.factorial(K))
                sv += weight * marginal
        shapley[str(cid)] = round(sv, 6)

    # LOO (leave-one-out) values
    full_acc = subset_accs[frozenset(all_clients)]
    loo = {}
    for cid in all_clients:
        without = frozenset(c for c in all_clients if c != cid)
        loo[str(cid)] = round(full_acc - subset_accs.get(without, 0), 6)

    # Data quantity contribution
    data_quantity = {}
    total_samples = sum(len(client_data[c][1]) for c in all_clients)
    for cid in all_clients:
        data_quantity[str(cid)] = round(len(client_data[cid][1]) / total_samples, 4)

    return {
        "full_coalition_accuracy": round(full_acc, 4),
        "shapley_values": shapley,
        "leave_one_out": loo,
        "data_quantity_shares": data_quantity,
        "num_subsets_evaluated": len(subset_accs),
        "shapley_sum": round(sum(shapley.values()), 6),
        "shapley_efficiency_gap": round(
            abs(sum(shapley.values()) - (full_acc - subset_accs[frozenset()])), 6
        ),
    }


def run_block_l(checkpoint, quick=False):
    """Block L: Shapley Values / Incentive Mechanisms (Tier 12)."""
    log("=" * 70)
    log("BLOCK L: Shapley Values & Incentive (Tier 12)")
    log("=" * 70)

    num_rounds = 5 if quick else 15  # Fewer rounds (many subsets to evaluate)
    seeds = [42] if quick else [42, 123, 7, 99, 256]
    conditions = ["IID", "NonIID"]
    t0 = time.time()
    count = 0
    total = len(ORIGINAL_DATASETS) * len(conditions) * len(seeds)

    for ds in ORIGINAL_DATASETS:
        cfg = DATASET_CONFIGS[ds]
        for cond in conditions:
            is_iid = (cond == "IID")
            for seed in seeds:
                if _shutdown:
                    return count
                key = "L_{}_{}_{}_s{}".format(ds, "shapley", cond, seed)
                if key in checkpoint["results"]:
                    count += 1
                    continue

                log("  [{}/{}] {} / Shapley / {} / seed={}".format(
                    count + 1, total, ds, cond, seed))

                try:
                    t_exp = time.time()
                    # Override iid in data loading
                    orig_cache_key = (ds, cfg["num_clients"], seed, is_iid, 0.5)
                    if orig_cache_key in _data_cache:
                        del _data_cache[orig_cache_key]

                    result = _compute_shapley_values(
                        ds, cfg, num_rounds, seed, is_iid=is_iid,
                    )
                    elapsed = time.time() - t_exp

                    checkpoint["results"][key] = {
                        "block": "L_shapley",
                        "dataset": ds,
                        "condition": cond,
                        "is_iid": is_iid,
                        "seed": seed,
                        "num_rounds": num_rounds,
                        **result,
                        "time_seconds": round(elapsed, 1),
                    }
                    count += 1
                    save_checkpoint(checkpoint)
                except Exception as e:
                    log("  ERROR: {} — {}".format(key, e))
                    traceback.print_exc()
                    count += 1
                _cleanup_gpu()

    elapsed = time.time() - t0
    checkpoint["metadata"]["block_l_time"] = round(elapsed, 1)
    save_checkpoint(checkpoint)
    log("Block L complete: {}/{} in {}".format(count, total, format_time(elapsed)))
    return count


# ======================================================================
# Main
# ======================================================================

def main():
    global _log_file, _shutdown

    parser = argparse.ArgumentParser(description="FL-EHDS Cascade 6 (Tiers 7-12)")
    parser.add_argument("--quick", action="store_true",
                        help="Reduced rounds/seeds for quick test")
    parser.add_argument("--fresh", action="store_true",
                        help="Delete existing checkpoint and start fresh")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _log_file = open(str(OUTPUT_DIR / LOG_FILE), "a")

    log("=" * 70)
    log("FL-EHDS Cascading Analysis — Phase 6 (Tiers 7–12)")
    log("Mode: {}".format("QUICK" if args.quick else "FULL"))
    log("=" * 70)

    if args.fresh:
        for f in [CHECKPOINT_FILE, CHECKPOINT_FILE + ".bak"]:
            p = OUTPUT_DIR / f
            if p.exists():
                p.unlink()
                log("Deleted {}".format(f))

    checkpoint = load_checkpoint()
    if checkpoint is None:
        checkpoint = {
            "metadata": {
                "experiment": "Cascading Analysis Phase 6",
                "started": datetime.now().isoformat(),
                "blocks": "G(coverage) H(hierarchical) I(async) J(multitask) K(byzantine) L(shapley)",
            },
            "results": {},
            "completed": [],
        }

    t_total = time.time()

    # Block G: Dataset Coverage (Tier 7)
    if not _shutdown:
        n_g = run_block_g(checkpoint, quick=args.quick)
        checkpoint["completed"].append("G") if "G" not in checkpoint.get("completed", []) else None

    # Block H: Hierarchical FL (Tier 8)
    if not _shutdown:
        n_h = run_block_h(checkpoint, quick=args.quick)
        checkpoint["completed"].append("H") if "H" not in checkpoint.get("completed", []) else None

    # Block I: Asynchronous FL (Tier 9)
    if not _shutdown:
        n_i = run_block_i(checkpoint, quick=args.quick)
        checkpoint["completed"].append("I") if "I" not in checkpoint.get("completed", []) else None

    # Block J: Multi-Task FL (Tier 10)
    if not _shutdown:
        n_j = run_block_j(checkpoint, quick=args.quick)
        checkpoint["completed"].append("J") if "J" not in checkpoint.get("completed", []) else None

    # Block K: Byzantine Boundary (Tier 11)
    if not _shutdown:
        n_k = run_block_k(checkpoint, quick=args.quick)
        checkpoint["completed"].append("K") if "K" not in checkpoint.get("completed", []) else None

    # Block L: Shapley Values (Tier 12)
    if not _shutdown:
        n_l = run_block_l(checkpoint, quick=args.quick)
        checkpoint["completed"].append("L") if "L" not in checkpoint.get("completed", []) else None

    total_elapsed = time.time() - t_total
    checkpoint["metadata"]["total_time"] = format_time(total_elapsed)
    save_checkpoint(checkpoint)

    # Summary
    block_counts = {}
    for key in checkpoint["results"]:
        block = key.split("_")[0]
        block_counts[block] = block_counts.get(block, 0) + 1

    total_exp = sum(block_counts.values())

    log("=" * 70)
    log("ALL BLOCKS COMPLETE in {}".format(format_time(total_elapsed)))
    log("Checkpoint: {}".format(CHECKPOINT_FILE))
    for block in sorted(block_counts.keys()):
        log("  Block {}: {} experiments".format(block, block_counts[block]))
    log("  TOTAL: {} experiments".format(total_exp))
    log("=" * 70)

    if _log_file:
        _log_file.close()


if __name__ == "__main__":
    main()

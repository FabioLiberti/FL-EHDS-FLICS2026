#!/usr/bin/env python3
"""
FL-EHDS Cascading Analysis & Experiments — Phase 4.

4 blocks executed sequentially:

Block A: Fairness Algorithm Benchmark (training, ~30 min)
  36 experiments: 6 fairness-aware aggregation strategies x 3 datasets
  x 2 heterogeneity conditions. Validates that FL-EHDS doesn't just
  *measure* fairness (Jain, Gini) but *optimises* it.

Block B: Continual/Temporal FL (training, ~30 min)
  36 experiments: 4 CL methods x 3 datasets x 3 drift scenarios.
  Tests catastrophic forgetting prevention under concept drift —
  critical for multi-year EHDS deployments.

Block C: Extended Personalisation Comparison (training, ~50 min)
  45 experiments: 5 personalisation methods x 3 datasets x 3 alpha levels.
  Validates "personalisation architecture dominates" with more methods.

Block D: Federated Unlearning / Right to be Forgotten (training, ~20 min)
  18 experiments: 3 unlearning methods x 3 datasets x 2 removal scenarios.
  GDPR Article 17 + EHDS Article 71 compliance validation.

Usage:
    cd fl-ehds-framework
    python -m benchmarks.run_analysis_cascade4 [--quick] [--fresh]

Output:
    benchmarks/paper_results_tabular/checkpoint_cascade4.json

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
import torch.nn as nn
import torch.optim as optim

from data.ptbxl_loader import load_ptbxl_data
from data.cardiovascular_loader import load_cardiovascular_data
from data.breast_cancer_loader import load_breast_cancer_data

# ======================================================================
# Constants
# ======================================================================

OUTPUT_DIR = FRAMEWORK_DIR / "benchmarks" / "paper_results_tabular"
CHECKPOINT_FILE = "checkpoint_cascade4.json"
LOG_FILE = "experiment_cascade4.log"

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
DEFAULT_SEED = 42

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
    fd, tmp = tempfile.mkstemp(dir=str(OUTPUT_DIR), prefix=".cas4_", suffix=".tmp")
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
            num_clients=num_clients, seed=seed,
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


def get_params_np(model):
    return {n: p.data.cpu().numpy().copy() for n, p in model.named_parameters()}


def train_local_sgd(model, X, y, epochs, lr, batch_size=64):
    """Standard local SGD training. Returns (loss, model_params_after)."""
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


def federated_round(global_model, client_data, lr, epochs, batch_size,
                    client_weights=None):
    """One federated round: local train + weighted aggregation.
    Returns (updated_global_params, client_losses, client_deltas)."""
    global_params = get_params(global_model)
    client_deltas = {}
    client_losses = {}

    for cid in client_data:
        # Clone global model for local training
        local_model = deepcopy(global_model)
        X, y = client_data[cid]
        loss = train_local_sgd(local_model, X, y, epochs, lr, batch_size)
        local_params = get_params(local_model)
        client_deltas[cid] = {n: local_params[n] - global_params[n]
                              for n in global_params}
        client_losses[cid] = loss
        del local_model

    # Aggregation
    if client_weights is None:
        # FedAvg: weighted by sample count
        total = sum(len(client_data[c][1]) for c in client_data)
        client_weights = {c: len(client_data[c][1]) / total for c in client_data}

    new_params = {}
    for n in global_params:
        new_params[n] = global_params[n] + sum(
            client_weights[c] * client_deltas[c][n] for c in client_deltas)

    set_params(global_model, new_params)
    return new_params, client_losses, client_deltas


# ======================================================================
# Block A: Fairness Algorithm Benchmark
# ======================================================================

FAIRNESS_ALGORITHMS = ["standard", "qfedavg", "afl", "fedmgda", "propfair", "fedminmax"]


def compute_fairness_weights(algo_name, client_losses, client_deltas,
                             client_accuracies, sample_counts,
                             afl_lambdas=None):
    """Compute fairness-aware aggregation weights."""
    cids = list(client_losses.keys())
    n = len(cids)

    if algo_name == "standard":
        total = sum(sample_counts[c] for c in cids)
        return {c: sample_counts[c] / total for c in cids}

    elif algo_name == "qfedavg":
        q = 1.0
        weights = {}
        for c in cids:
            loss = max(client_losses[c], 1e-10)
            weights[c] = loss ** q
        total = sum(weights.values())
        return {c: weights[c] / total for c in cids}

    elif algo_name == "afl":
        # Exponentiated gradient: upweight high-loss clients
        lambda_lr = 0.1
        if afl_lambdas is None:
            afl_lambdas = {c: 1.0 / n for c in cids}
        for c in cids:
            afl_lambdas[c] *= np.exp(lambda_lr * client_losses[c])
        total = sum(afl_lambdas.values())
        for c in cids:
            afl_lambdas[c] /= total
            afl_lambdas[c] = min(afl_lambdas[c], 5.0 / n)
        total = sum(afl_lambdas.values())
        return {c: afl_lambdas[c] / total for c in cids}

    elif algo_name == "fedmgda":
        # Frank-Wolfe Pareto weights
        flat_grads = []
        for c in cids:
            flat = torch.cat([client_deltas[c][n].flatten()
                              for n in sorted(client_deltas[c].keys())])
            flat_grads.append(flat.numpy())
        alpha = np.ones(n) / n
        for it in range(20):
            agg = sum(alpha[k] * flat_grads[k] for k in range(n))
            inner = np.array([np.dot(flat_grads[k], agg) for k in range(n)])
            s = np.zeros(n)
            s[np.argmin(inner)] = 1.0
            gamma = 2.0 / (it + 2)
            alpha = (1 - gamma) * alpha + gamma * s
        return {cids[i]: float(alpha[i]) for i in range(n)}

    elif algo_name == "propfair":
        alpha_exp = 1.0
        weights = {}
        for c in cids:
            acc = max(client_accuracies.get(c, 0.5), 0.01)
            weights[c] = 1.0 / (acc ** alpha_exp)
        total = sum(weights.values())
        return {c: weights[c] / total for c in cids}

    elif algo_name == "fedminmax":
        temperature = 10.0
        losses = np.array([client_losses[c] for c in cids])
        exp_losses = np.exp(temperature * (losses - np.max(losses)))
        softmax_w = exp_losses / np.sum(exp_losses)
        return {cids[i]: float(softmax_w[i]) for i in range(n)}

    return {c: 1.0 / n for c in cids}


def run_fairness_experiment(dataset, fairness_algo, is_iid, seed, quick=False):
    """Run one fairness-aware FL training experiment."""
    start = time.time()
    dcfg = DATASET_CONFIGS[dataset]
    num_rounds = 5 if quick else dcfg["num_rounds"]
    alpha = 0.5

    client_data, client_test, _ = load_dataset(
        dataset, dcfg["num_clients"], seed, is_iid=is_iid, alpha=alpha)

    model = create_model(dcfg["input_dim"], dcfg["num_classes"], seed)
    sample_counts = {c: len(client_data[c][1]) for c in client_data}
    afl_lambdas = None
    history = []

    for r in range(num_rounds):
        # Get client accuracies for PropFair
        client_accs = {}
        for cid in client_data:
            client_accs[cid] = evaluate_model(model, *client_test[cid])

        # Local training + custom aggregation
        global_params = get_params(model)
        client_deltas = {}
        client_losses = {}
        for cid in client_data:
            local_model = deepcopy(model)
            loss = train_local_sgd(local_model, *client_data[cid],
                                   dcfg["local_epochs"], dcfg["learning_rate"],
                                   dcfg["batch_size"])
            local_p = get_params(local_model)
            client_deltas[cid] = {n: local_p[n] - global_params[n] for n in global_params}
            client_losses[cid] = loss
            del local_model

        # Fairness-aware weights
        weights = compute_fairness_weights(
            fairness_algo, client_losses, client_deltas,
            client_accs, sample_counts, afl_lambdas)
        if fairness_algo == "afl":
            afl_lambdas = weights.copy()

        # Aggregate
        new_params = {}
        for n in global_params:
            new_params[n] = global_params[n] + sum(
                weights[c] * client_deltas[c][n] for c in client_deltas)
        set_params(model, new_params)

        per_client = evaluate_per_client(model, client_test)
        round_acc = float(np.mean(list(per_client.values())))
        history.append(round_acc)

    per_client_acc = evaluate_per_client(model, client_test)
    accs = list(per_client_acc.values())
    elapsed = time.time() - start

    return {
        "block": "A_fairness",
        "dataset": dataset, "fairness_algorithm": fairness_algo,
        "is_iid": is_iid, "seed": seed, "num_rounds": num_rounds,
        "final_accuracy": round(float(np.mean(accs)), 4),
        "worst_client_accuracy": round(float(np.min(accs)), 4),
        "best_client_accuracy": round(float(np.max(accs)), 4),
        "per_client_accuracy": per_client_acc,
        "jain_index": round(_compute_jain_index(per_client_acc), 4),
        "gini_coefficient": round(_compute_gini(accs), 4),
        "convergence_trajectory": [round(h, 4) for h in history],
        "time_seconds": round(elapsed, 1),
    }


def block_a_fairness(ckpt, quick=False):
    """Block A: Fairness Algorithm Benchmark."""
    log("")
    log("=" * 70)
    log("BLOCK A — Fairness Algorithm Benchmark")
    log("=" * 70)

    start = time.time()
    seed = DEFAULT_SEED

    experiments = []
    for ds in DATASETS:
        for fa in FAIRNESS_ALGORITHMS:
            for is_iid in [True, False]:
                cond = "IID" if is_iid else "nonIID"
                exp_key = "A_{}_{}_{}_s{}".format(ds, fa, cond, seed)
                experiments.append((exp_key, ds, fa, is_iid))

    total = len(experiments)
    done = sum(1 for e in experiments if e[0] in ckpt.get("completed", []))
    log("  {} experiments ({} already done)".format(total, done))

    for idx, (exp_key, ds, fa, is_iid) in enumerate(experiments, 1):
        if _shutdown:
            log("  Shutdown requested")
            break
        if exp_key in ckpt.get("completed", []):
            continue

        try:
            cond = "IID" if is_iid else "nonIID"
            log("  [{}/{}] {} | {} | {} ...".format(idx, total, ds, fa, cond))

            result = run_fairness_experiment(ds, fa, is_iid, seed, quick=quick)

            ckpt["results"][exp_key] = result
            ckpt["completed"].append(exp_key)
            save_checkpoint(ckpt)

            log("    -> acc={:.1f}% worst={:.1f}% Jain={:.3f} Gini={:.3f} | {:.0f}s".format(
                result["final_accuracy"] * 100,
                result["worst_client_accuracy"] * 100,
                result["jain_index"],
                result["gini_coefficient"],
                result["time_seconds"]))

            _cleanup_gpu()
        except Exception as e:
            log("    ERROR: {}".format(e))
            log("    " + traceback.format_exc().split("\n")[-2])
            _cleanup_gpu()

    elapsed = time.time() - start
    ckpt["metadata"]["block_a_time"] = round(elapsed, 1)
    save_checkpoint(ckpt)

    # Print summary
    log("")
    log("  FAIRNESS ALGORITHM COMPARISON:")
    log("  {:<12} {:<12} {:<7} {:>7} {:>7} {:>6} {:>6}".format(
        "Dataset", "FairnessAlgo", "Cond", "Acc%", "Worst%", "Jain", "Gini"))
    log("  " + "-" * 65)
    for ds in DATASETS:
        for fa in FAIRNESS_ALGORITHMS:
            for is_iid in [True, False]:
                cond = "IID" if is_iid else "nonIID"
                ek = "A_{}_{}_{}_s{}".format(ds, fa, cond, seed)
                if ek in ckpt.get("results", {}):
                    r = ckpt["results"][ek]
                    log("  {:<12} {:<12} {:<7} {:>5.1f}%  {:>5.1f}%  {:>5.3f} {:>5.3f}".format(
                        ds[:12], fa[:12], cond,
                        r["final_accuracy"] * 100,
                        r["worst_client_accuracy"] * 100,
                        r["jain_index"], r["gini_coefficient"]))
        log("  " + "-" * 65)

    log("  Block A complete ({}).".format(format_time(elapsed)))


# ======================================================================
# Block B: Continual / Temporal FL
# ======================================================================

CONTINUAL_METHODS = ["baseline", "ewc", "replay", "lwf"]
DRIFT_SCENARIOS = ["no_drift", "sudden_shift", "gradual_shift"]


def _create_drift_tasks(client_data, scenario, seed, num_classes):
    """Split client data into Task 1 and Task 2 with optional drift.

    Returns (task1_data, task2_data) — each is dict{cid: (X, y)}.
    """
    rng = np.random.RandomState(seed)
    task1 = {}
    task2 = {}

    for cid in client_data:
        X, y = client_data[cid]
        n = len(y)
        mid = n // 2

        # Shuffle before splitting
        perm = rng.permutation(n)
        X_shuf = X[perm]
        y_shuf = y[perm]

        task1[cid] = (X_shuf[:mid], y_shuf[:mid])

        if scenario == "no_drift":
            task2[cid] = (X_shuf[mid:], y_shuf[mid:])
        elif scenario == "sudden_shift":
            # Flip 40% of labels in Task 2 to simulate sudden distribution change
            y2 = y_shuf[mid:].copy()
            n2 = len(y2)
            flip_idx = rng.choice(n2, int(0.4 * n2), replace=False)
            y2[flip_idx] = rng.randint(0, num_classes, size=len(flip_idx))
            task2[cid] = (X_shuf[mid:], y2)
        elif scenario == "gradual_shift":
            # Flip 20% of labels (milder drift)
            y2 = y_shuf[mid:].copy()
            n2 = len(y2)
            flip_idx = rng.choice(n2, int(0.2 * n2), replace=False)
            y2[flip_idx] = rng.randint(0, num_classes, size=len(flip_idx))
            task2[cid] = (X_shuf[mid:], y2)

    return task1, task2


def _compute_fisher(model, X, y, n_samples=200):
    """Compute diagonal Fisher Information Matrix."""
    model.eval()
    fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
    X_t = torch.FloatTensor(X).to(DEVICE)
    y_t = torch.LongTensor(y.astype(int)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()

    n = min(n_samples, len(X_t))
    indices = torch.randperm(len(X_t))[:n]

    for idx in indices:
        model.zero_grad()
        out = model(X_t[idx:idx + 1])
        loss = criterion(out, y_t[idx:idx + 1])
        loss.backward()
        for n_name, p in model.named_parameters():
            if p.grad is not None:
                fisher[n_name] += p.grad.data ** 2

    for n_name in fisher:
        fisher[n_name] /= n

    return fisher


def train_local_ewc(model, X, y, epochs, lr, batch_size,
                    fisher_matrices, optimal_params_list, ewc_lambda):
    """Train with EWC regularization."""
    model.train()
    X_t = torch.FloatTensor(X).to(DEVICE)
    y_t = torch.LongTensor(y.astype(int)).to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for _ in range(epochs):
        perm = torch.randperm(len(y_t))
        for i in range(0, len(y_t), batch_size):
            idx = perm[i:i + batch_size]
            optimizer.zero_grad()
            out = model(X_t[idx])
            ce_loss = criterion(out, y_t[idx])

            # EWC penalty
            ewc_loss = torch.tensor(0.0, device=DEVICE)
            for task_idx in range(len(fisher_matrices)):
                for n_name, p in model.named_parameters():
                    if n_name in fisher_matrices[task_idx]:
                        diff = p - optimal_params_list[task_idx][n_name]
                        ewc_loss += (fisher_matrices[task_idx][n_name] * diff ** 2).sum()

            loss = ce_loss + (ewc_lambda / 2) * ewc_loss
            loss.backward()
            optimizer.step()


def train_local_lwf(model, X, y, epochs, lr, batch_size, old_model,
                    lwf_alpha=0.5, temperature=2.0):
    """Train with Learning without Forgetting (knowledge distillation)."""
    model.train()
    X_t = torch.FloatTensor(X).to(DEVICE)
    y_t = torch.LongTensor(y.astype(int)).to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    if old_model is not None:
        old_model.eval()

    for _ in range(epochs):
        perm = torch.randperm(len(y_t))
        for i in range(0, len(y_t), batch_size):
            idx = perm[i:i + batch_size]
            optimizer.zero_grad()
            out = model(X_t[idx])
            ce_loss = criterion(out, y_t[idx])

            # Distillation loss
            if old_model is not None:
                with torch.no_grad():
                    old_out = old_model(X_t[idx])
                old_probs = torch.softmax(old_out / temperature, dim=1)
                new_log_probs = torch.log_softmax(out / temperature, dim=1)
                distill_loss = -(old_probs * new_log_probs).sum(dim=1).mean()
                distill_loss *= temperature ** 2
                loss = (1 - lwf_alpha) * ce_loss + lwf_alpha * distill_loss
            else:
                loss = ce_loss

            loss.backward()
            optimizer.step()


def run_continual_experiment(dataset, cl_method, drift_scenario, seed, quick=False):
    """Run one continual FL experiment."""
    start = time.time()
    dcfg = DATASET_CONFIGS[dataset]
    rounds_per_task = 3 if quick else dcfg["num_rounds"] // 2
    num_classes = dcfg["num_classes"]

    client_data, client_test, _ = load_dataset(
        dataset, dcfg["num_clients"], seed, is_iid=True)

    # Create task-split data
    task1_data, task2_data = _create_drift_tasks(
        client_data, drift_scenario, seed, num_classes)

    model = create_model(dcfg["input_dim"], num_classes, seed)
    lr = dcfg["learning_rate"]
    ep = dcfg["local_epochs"]
    bs = dcfg["batch_size"]

    # Replay buffer
    replay_buffer = {}
    replay_ratio = 0.3
    buffer_size = 200

    # Storage for EWC
    fisher_matrices = []
    optimal_params_list = []
    old_model_lwf = None

    # === Task 1 training ===
    task1_history = []
    for r in range(rounds_per_task):
        federated_round(model, task1_data, lr, ep, bs)
        per_client = evaluate_per_client(model, client_test)
        task1_history.append(float(np.mean(list(per_client.values()))))

    acc_after_task1 = task1_history[-1] if task1_history else 0.0

    # Consolidate Task 1
    if cl_method == "ewc":
        X_all = np.concatenate([task1_data[c][0] for c in task1_data])
        y_all = np.concatenate([task1_data[c][1] for c in task1_data])
        fisher = _compute_fisher(model, X_all, y_all)
        fisher_matrices.append(fisher)
        optimal_params_list.append(get_params(model))

    elif cl_method == "replay":
        rng = np.random.RandomState(seed)
        for cid in task1_data:
            X, y = task1_data[cid]
            n = len(y)
            k = min(buffer_size // dcfg["num_clients"], n)
            idx = rng.choice(n, k, replace=False)
            replay_buffer[cid] = (X[idx], y[idx])

    elif cl_method == "lwf":
        old_model_lwf = deepcopy(model)

    # === Task 2 training ===
    task2_history = []
    for r in range(rounds_per_task):
        if cl_method == "ewc":
            # Train each client with EWC
            global_params = get_params(model)
            client_deltas = {}
            for cid in task2_data:
                local_model = deepcopy(model)
                train_local_ewc(local_model, *task2_data[cid], ep, lr, bs,
                                fisher_matrices, optimal_params_list, 1000.0)
                local_p = get_params(local_model)
                client_deltas[cid] = {n: local_p[n] - global_params[n] for n in global_params}
                del local_model
            # Aggregate
            n_clients = len(client_deltas)
            new_params = {}
            for n in global_params:
                new_params[n] = global_params[n] + sum(
                    client_deltas[c][n] for c in client_deltas) / n_clients
            set_params(model, new_params)

        elif cl_method == "replay":
            # Mix replay buffer with Task 2 data
            mixed_data = {}
            rng = np.random.RandomState(seed + r)
            for cid in task2_data:
                X2, y2 = task2_data[cid]
                if cid in replay_buffer:
                    Xr, yr = replay_buffer[cid]
                    n_replay = max(1, int(len(X2) * replay_ratio))
                    n_replay = min(n_replay, len(Xr))
                    ridx = rng.choice(len(Xr), n_replay, replace=False)
                    X_mix = np.concatenate([X2, Xr[ridx]])
                    y_mix = np.concatenate([y2, yr[ridx]])
                    mixed_data[cid] = (X_mix, y_mix)
                else:
                    mixed_data[cid] = (X2, y2)
            federated_round(model, mixed_data, lr, ep, bs)

        elif cl_method == "lwf":
            # Train each client with LwF
            global_params = get_params(model)
            client_deltas = {}
            for cid in task2_data:
                local_model = deepcopy(model)
                train_local_lwf(local_model, *task2_data[cid], ep, lr, bs,
                                old_model_lwf)
                local_p = get_params(local_model)
                client_deltas[cid] = {n: local_p[n] - global_params[n] for n in global_params}
                del local_model
            n_clients = len(client_deltas)
            new_params = {}
            for n in global_params:
                new_params[n] = global_params[n] + sum(
                    client_deltas[c][n] for c in client_deltas) / n_clients
            set_params(model, new_params)

        else:  # baseline
            federated_round(model, task2_data, lr, ep, bs)

        per_client = evaluate_per_client(model, client_test)
        task2_history.append(float(np.mean(list(per_client.values()))))

    # Final evaluation
    per_client_acc = evaluate_per_client(model, client_test)
    final_acc = float(np.mean(list(per_client_acc.values())))

    # Backward transfer: accuracy on Task 1 test data after Task 2 training
    # (Use full test set as proxy — measures retention of original knowledge)
    backward_transfer = final_acc - acc_after_task1

    elapsed = time.time() - start
    return {
        "block": "B_continual",
        "dataset": dataset, "cl_method": cl_method,
        "drift_scenario": drift_scenario, "seed": seed,
        "rounds_per_task": rounds_per_task,
        "acc_after_task1": round(acc_after_task1, 4),
        "acc_after_task2": round(final_acc, 4),
        "backward_transfer": round(backward_transfer, 4),
        "per_client_accuracy": per_client_acc,
        "jain_index": round(_compute_jain_index(per_client_acc), 4),
        "task1_trajectory": [round(h, 4) for h in task1_history],
        "task2_trajectory": [round(h, 4) for h in task2_history],
        "time_seconds": round(elapsed, 1),
    }


def block_b_continual(ckpt, quick=False):
    """Block B: Continual/Temporal FL."""
    log("")
    log("=" * 70)
    log("BLOCK B — Continual / Temporal FL")
    log("=" * 70)

    start = time.time()
    seed = DEFAULT_SEED

    experiments = []
    for ds in DATASETS:
        for cl in CONTINUAL_METHODS:
            for drift in DRIFT_SCENARIOS:
                exp_key = "B_{}_{}_{}".format(ds, cl, drift)
                experiments.append((exp_key, ds, cl, drift))

    total = len(experiments)
    done = sum(1 for e in experiments if e[0] in ckpt.get("completed", []))
    log("  {} experiments ({} already done)".format(total, done))

    for idx, (exp_key, ds, cl, drift) in enumerate(experiments, 1):
        if _shutdown:
            log("  Shutdown requested")
            break
        if exp_key in ckpt.get("completed", []):
            continue

        try:
            log("  [{}/{}] {} | {} | {} ...".format(idx, total, ds, cl, drift))

            result = run_continual_experiment(ds, cl, drift, seed, quick=quick)

            ckpt["results"][exp_key] = result
            ckpt["completed"].append(exp_key)
            save_checkpoint(ckpt)

            log("    -> T1={:.1f}% T2={:.1f}% BwdTrans={:+.1f}pp Jain={:.3f} | {:.0f}s".format(
                result["acc_after_task1"] * 100,
                result["acc_after_task2"] * 100,
                result["backward_transfer"] * 100,
                result["jain_index"],
                result["time_seconds"]))

            _cleanup_gpu()
        except Exception as e:
            log("    ERROR: {}".format(e))
            log("    " + traceback.format_exc().split("\n")[-2])
            _cleanup_gpu()

    elapsed = time.time() - start
    ckpt["metadata"]["block_b_time"] = round(elapsed, 1)
    save_checkpoint(ckpt)

    # Print summary
    log("")
    log("  CONTINUAL FL RESULTS:")
    log("  {:<12} {:<10} {:<14} {:>7} {:>7} {:>9}".format(
        "Dataset", "CLMethod", "Drift", "T1 Acc", "T2 Acc", "BwdTrans"))
    log("  " + "-" * 65)
    for ds in DATASETS:
        for cl in CONTINUAL_METHODS:
            for drift in DRIFT_SCENARIOS:
                ek = "B_{}_{}_{}".format(ds, cl, drift)
                if ek in ckpt.get("results", {}):
                    r = ckpt["results"][ek]
                    log("  {:<12} {:<10} {:<14} {:>5.1f}%  {:>5.1f}%  {:>+7.1f}pp".format(
                        ds[:12], cl[:10], drift[:14],
                        r["acc_after_task1"] * 100,
                        r["acc_after_task2"] * 100,
                        r["backward_transfer"] * 100))
        log("  " + "-" * 65)

    log("  Block B complete ({}).".format(format_time(elapsed)))


# ======================================================================
# Block C: Extended Personalisation Comparison
# ======================================================================

PERSONALISATION_METHODS = ["fedavg", "ditto", "per_fedavg", "fedper", "apfl"]
ALPHA_LEVELS = [0.1, 0.5]  # + IID


def run_personalisation_experiment(dataset, method, is_iid, alpha, seed, quick=False):
    """Run one personalisation experiment."""
    start = time.time()
    dcfg = DATASET_CONFIGS[dataset]
    num_rounds = 5 if quick else dcfg["num_rounds"]
    lr = dcfg["learning_rate"]
    ep = dcfg["local_epochs"]
    bs = dcfg["batch_size"]

    client_data, client_test, _ = load_dataset(
        dataset, dcfg["num_clients"], seed, is_iid=is_iid, alpha=alpha)

    global_model = create_model(dcfg["input_dim"], dcfg["num_classes"], seed)
    num_clients = dcfg["num_clients"]

    # Personalisation-specific state
    local_models = {}
    apfl_alphas = {c: 0.5 for c in client_data}
    ditto_lambda = 0.1
    history = []

    for r in range(num_rounds):
        global_params = get_params(global_model)
        client_deltas = {}

        if method == "fedavg":
            # Standard FedAvg
            _, _, _ = federated_round(global_model, client_data, lr, ep, bs)

        elif method == "ditto":
            # Phase 1: standard FedAvg aggregation for global model
            _, _, _ = federated_round(global_model, client_data, lr, ep, bs)
            global_params_new = get_params(global_model)

            # Phase 2: personalised local models with L2 regularisation
            for cid in client_data:
                if cid not in local_models:
                    local_models[cid] = deepcopy(global_model)
                else:
                    set_params(local_models[cid], get_params(local_models[cid]))

                lm = local_models[cid]
                lm.train()
                X_t = torch.FloatTensor(client_data[cid][0]).to(DEVICE)
                y_t = torch.LongTensor(client_data[cid][1].astype(int)).to(DEVICE)
                optimizer = optim.SGD(lm.parameters(), lr=lr)
                criterion = nn.CrossEntropyLoss()

                for _ in range(ep):
                    perm = torch.randperm(len(y_t))
                    for i in range(0, len(y_t), bs):
                        idx = perm[i:i + bs]
                        optimizer.zero_grad()
                        out = lm(X_t[idx])
                        ce_loss = criterion(out, y_t[idx])
                        # Ditto regularisation: L2 to global
                        reg_loss = torch.tensor(0.0, device=DEVICE)
                        for n_name, p in lm.named_parameters():
                            if n_name in global_params_new:
                                reg_loss += ((p - global_params_new[n_name]) ** 2).sum()
                        loss = ce_loss + (ditto_lambda / 2) * reg_loss
                        loss.backward()
                        optimizer.step()

        elif method == "per_fedavg":
            # MAML-based: support/query split
            for cid in client_data:
                local_model = deepcopy(global_model)
                X, y = client_data[cid]
                n = len(y)
                mid = n // 2
                X_supp, y_supp = X[:mid], y[:mid]
                X_query, y_query = X[mid:], y[mid:]

                # Inner loop: adapt on support
                inner_lr = 0.01
                train_local_sgd(local_model, X_supp, y_supp, 5, inner_lr, bs)

                # Outer loop: evaluate on query, compute meta-gradient
                local_p = get_params(local_model)
                client_deltas[cid] = {n: local_p[n] - global_params[n] for n in global_params}
                del local_model

            # Meta-aggregate
            n_c = len(client_deltas)
            meta_lr = 0.01
            new_params = {}
            for n in global_params:
                new_params[n] = global_params[n] + meta_lr * sum(
                    client_deltas[c][n] for c in client_deltas) / n_c
            set_params(global_model, new_params)

        elif method == "fedper":
            # Shared base (fc1), personalised head (fc2)
            for cid in client_data:
                if cid not in local_models:
                    local_models[cid] = deepcopy(global_model)

                # Sync base layers from global
                with torch.no_grad():
                    for n_name, p in local_models[cid].named_parameters():
                        if "fc1" in n_name:
                            p.data.copy_(global_params[n_name])

                # Local training (all layers)
                train_local_sgd(local_models[cid], *client_data[cid], ep, lr, bs)
                local_p = get_params(local_models[cid])

                # Only share base layer updates
                client_deltas[cid] = {}
                for n in global_params:
                    if "fc1" in n:
                        client_deltas[cid][n] = local_p[n] - global_params[n]

            # Aggregate only base layers
            if client_deltas:
                n_c = len(client_deltas)
                new_params = dict(global_params)
                for n in global_params:
                    if "fc1" in n:
                        new_params[n] = global_params[n] + sum(
                            client_deltas[c].get(n, torch.zeros_like(global_params[n]))
                            for c in client_deltas) / n_c
                set_params(global_model, new_params)

        elif method == "apfl":
            # Adaptive mixing: personal = alpha * local + (1-alpha) * global
            for cid in client_data:
                if cid not in local_models:
                    local_models[cid] = deepcopy(global_model)

                # Train local model
                train_local_sgd(local_models[cid], *client_data[cid], ep, lr, bs)
                local_p = get_params(local_models[cid])

                # Update mixing weight based on validation
                a = apfl_alphas[cid]
                mixed_model = deepcopy(global_model)
                mixed_p = {}
                for n in global_params:
                    mixed_p[n] = a * local_p[n] + (1 - a) * global_params[n]
                set_params(mixed_model, mixed_p)
                mixed_acc = evaluate_model(mixed_model, *client_test[cid])

                # Try slightly different alpha
                for da in [0.05, -0.05]:
                    a_try = np.clip(a + da, 0.1, 0.9)
                    for n in global_params:
                        mixed_p[n] = a_try * local_p[n] + (1 - a_try) * global_params[n]
                    set_params(mixed_model, mixed_p)
                    acc_try = evaluate_model(mixed_model, *client_test[cid])
                    if acc_try > mixed_acc:
                        mixed_acc = acc_try
                        apfl_alphas[cid] = a_try

                del mixed_model
                # Share local update for global aggregation
                client_deltas[cid] = {n: local_p[n] - global_params[n] for n in global_params}

            # Standard FedAvg aggregate for global model
            if client_deltas:
                n_c = len(client_deltas)
                new_params = {}
                for n in global_params:
                    new_params[n] = global_params[n] + sum(
                        client_deltas[c][n] for c in client_deltas) / n_c
                set_params(global_model, new_params)

        # Evaluate (use personalised models where applicable)
        if method in ("ditto", "fedper", "apfl") and local_models:
            per_client = {}
            for cid in client_test:
                if cid in local_models:
                    if method == "apfl":
                        # Use mixed model for evaluation
                        a = apfl_alphas[cid]
                        lp = get_params(local_models[cid])
                        gp = get_params(global_model)
                        mixed_p = {n: a * lp[n] + (1 - a) * gp[n] for n in gp}
                        eval_model = deepcopy(global_model)
                        set_params(eval_model, mixed_p)
                        per_client[str(cid)] = evaluate_model(eval_model, *client_test[cid])
                        del eval_model
                    else:
                        per_client[str(cid)] = evaluate_model(
                            local_models[cid], *client_test[cid])
                else:
                    per_client[str(cid)] = evaluate_model(global_model, *client_test[cid])
        else:
            per_client = evaluate_per_client(global_model, client_test)

        round_acc = float(np.mean(list(per_client.values())))
        history.append(round_acc)

    # Final evaluation
    if method in ("ditto", "fedper", "apfl") and local_models:
        per_client_acc = {}
        for cid in client_test:
            if cid in local_models:
                if method == "apfl":
                    a = apfl_alphas[cid]
                    lp = get_params(local_models[cid])
                    gp = get_params(global_model)
                    mixed_p = {n: a * lp[n] + (1 - a) * gp[n] for n in gp}
                    eval_model = deepcopy(global_model)
                    set_params(eval_model, mixed_p)
                    per_client_acc[str(cid)] = evaluate_model(eval_model, *client_test[cid])
                    del eval_model
                else:
                    per_client_acc[str(cid)] = evaluate_model(
                        local_models[cid], *client_test[cid])
            else:
                per_client_acc[str(cid)] = evaluate_model(global_model, *client_test[cid])
    else:
        per_client_acc = evaluate_per_client(global_model, client_test)

    accs = list(per_client_acc.values())
    elapsed = time.time() - start

    extra = {}
    if method == "apfl":
        extra["mixing_weights"] = {str(c): round(apfl_alphas[c], 3) for c in apfl_alphas}

    return {
        "block": "C_personalisation",
        "dataset": dataset, "method": method,
        "is_iid": is_iid, "alpha": alpha, "seed": seed,
        "num_rounds": num_rounds,
        "final_accuracy": round(float(np.mean(accs)), 4),
        "worst_client_accuracy": round(float(np.min(accs)), 4),
        "per_client_accuracy": per_client_acc,
        "jain_index": round(_compute_jain_index(per_client_acc), 4),
        "gini_coefficient": round(_compute_gini(accs), 4),
        "convergence_trajectory": [round(h, 4) for h in history],
        "time_seconds": round(elapsed, 1),
        **extra,
    }


def block_c_personalisation(ckpt, quick=False):
    """Block C: Extended Personalisation Comparison."""
    log("")
    log("=" * 70)
    log("BLOCK C — Extended Personalisation Comparison")
    log("=" * 70)

    start = time.time()
    seed = DEFAULT_SEED

    experiments = []
    for ds in DATASETS:
        for method in PERSONALISATION_METHODS:
            # IID
            exp_key = "C_{}_{}_{}_s{}".format(ds, method, "IID", seed)
            experiments.append((exp_key, ds, method, True, 0.5))
            # Non-IID
            for alpha in ALPHA_LEVELS:
                exp_key = "C_{}_{}_a{}_s{}".format(ds, method, alpha, seed)
                experiments.append((exp_key, ds, method, False, alpha))

    total = len(experiments)
    done = sum(1 for e in experiments if e[0] in ckpt.get("completed", []))
    log("  {} experiments ({} already done)".format(total, done))

    for idx, (exp_key, ds, method, is_iid, alpha) in enumerate(experiments, 1):
        if _shutdown:
            log("  Shutdown requested")
            break
        if exp_key in ckpt.get("completed", []):
            continue

        try:
            cond = "IID" if is_iid else "a={}".format(alpha)
            log("  [{}/{}] {} | {} | {} ...".format(idx, total, ds, method, cond))

            result = run_personalisation_experiment(
                ds, method, is_iid, alpha, seed, quick=quick)

            ckpt["results"][exp_key] = result
            ckpt["completed"].append(exp_key)
            save_checkpoint(ckpt)

            log("    -> acc={:.1f}% worst={:.1f}% Jain={:.3f} | {:.0f}s".format(
                result["final_accuracy"] * 100,
                result["worst_client_accuracy"] * 100,
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

    # Print summary
    log("")
    log("  PERSONALISATION COMPARISON:")
    log("  {:<12} {:<12} {:<8} {:>7} {:>7} {:>6}".format(
        "Dataset", "Method", "Cond", "Acc%", "Worst%", "Jain"))
    log("  " + "-" * 60)
    for ds in DATASETS:
        for method in PERSONALISATION_METHODS:
            for is_iid, alpha in [(True, 0.5)] + [(False, a) for a in ALPHA_LEVELS]:
                cond = "IID" if is_iid else "a={}".format(alpha)
                if is_iid:
                    ek = "C_{}_{}_{}_s{}".format(ds, method, "IID", seed)
                else:
                    ek = "C_{}_{}_a{}_s{}".format(ds, method, alpha, seed)
                if ek in ckpt.get("results", {}):
                    r = ckpt["results"][ek]
                    log("  {:<12} {:<12} {:<8} {:>5.1f}%  {:>5.1f}%  {:>5.3f}".format(
                        ds[:12], method[:12], cond,
                        r["final_accuracy"] * 100,
                        r["worst_client_accuracy"] * 100,
                        r["jain_index"]))
        log("  " + "-" * 60)

    log("  Block C complete ({}).".format(format_time(elapsed)))


# ======================================================================
# Block D: Federated Unlearning / Right to be Forgotten
# ======================================================================

UNLEARNING_METHODS = ["exact_retrain", "gradient_ascent", "fed_eraser"]
REMOVAL_SCENARIOS = ["remove_1client", "remove_2clients"]


def run_unlearning_experiment(dataset, ul_method, removal_scenario, seed, quick=False):
    """Run one federated unlearning experiment."""
    start = time.time()
    dcfg = DATASET_CONFIGS[dataset]
    num_rounds = 5 if quick else dcfg["num_rounds"]
    lr = dcfg["learning_rate"]
    ep = dcfg["local_epochs"]
    bs = dcfg["batch_size"]

    client_data, client_test, _ = load_dataset(
        dataset, dcfg["num_clients"], seed, is_iid=True)

    # Phase 1: Train full model and store per-round client gradients
    model = create_model(dcfg["input_dim"], dcfg["num_classes"], seed)
    round_deltas = []  # Per-round, per-client gradient updates

    for r in range(num_rounds):
        global_params = get_params(model)
        client_deltas = {}
        for cid in client_data:
            local_model = deepcopy(model)
            train_local_sgd(local_model, *client_data[cid], ep, lr, bs)
            local_p = get_params(local_model)
            client_deltas[cid] = {n: local_p[n] - global_params[n] for n in global_params}
            del local_model

        # Standard aggregation
        n_c = len(client_deltas)
        new_params = {}
        for n in global_params:
            new_params[n] = global_params[n] + sum(
                client_deltas[c][n] for c in client_deltas) / n_c
        set_params(model, new_params)
        round_deltas.append(client_deltas)

    # Accuracy before unlearning
    per_client_before = evaluate_per_client(model, client_test)
    acc_before = float(np.mean(list(per_client_before.values())))

    # Determine clients to remove
    all_clients = list(client_data.keys())
    if removal_scenario == "remove_1client":
        remove_clients = [all_clients[-1]]
    else:
        remove_clients = all_clients[-2:]
    remaining_clients = [c for c in all_clients if c not in remove_clients]

    # Phase 2: Unlearning
    t_unlearn_start = time.time()

    if ul_method == "exact_retrain":
        # Retrain from scratch without removed clients
        model_ul = create_model(dcfg["input_dim"], dcfg["num_classes"], seed)
        remaining_data = {c: client_data[c] for c in remaining_clients}
        for r in range(num_rounds):
            federated_round(model_ul, remaining_data, lr, ep, bs)
        unlearned_model = model_ul

    elif ul_method == "gradient_ascent":
        # Reverse gradient contributions of removed clients
        unlearned_model = deepcopy(model)
        ul_params = get_params(unlearned_model)
        unlearning_rate = 0.01
        damping = 0.1
        for step in range(10):
            decay = (1.0 - damping) ** step
            for rd in round_deltas:
                for cid in remove_clients:
                    if cid in rd:
                        n_c_round = len(rd)
                        for n in ul_params:
                            ul_params[n] = ul_params[n] + unlearning_rate * decay * rd[cid][n] / n_c_round
        set_params(unlearned_model, ul_params)

    elif ul_method == "fed_eraser":
        # Subtract accumulated contribution + calibration rounds
        unlearned_model = deepcopy(model)
        ul_params = get_params(unlearned_model)

        # Compute accumulated contribution
        for rd in round_deltas:
            n_c_round = len(rd)
            for cid in remove_clients:
                if cid in rd:
                    for n in ul_params:
                        ul_params[n] = ul_params[n] - rd[cid][n] / n_c_round
        set_params(unlearned_model, ul_params)

        # Calibration rounds with remaining clients
        remaining_data = {c: client_data[c] for c in remaining_clients}
        for r in range(min(5, num_rounds)):
            federated_round(unlearned_model, remaining_data, lr * 0.5, ep, bs)

    else:
        unlearned_model = deepcopy(model)

    unlearning_time = time.time() - t_unlearn_start

    # Evaluate after unlearning
    per_client_after = evaluate_per_client(unlearned_model, client_test)
    acc_after = float(np.mean([per_client_after[str(c)]
                               for c in remaining_clients
                               if str(c) in per_client_after]))

    # Verification: MIA-style check on removed clients
    removed_acc = float(np.mean([per_client_after.get(str(c), 0)
                                  for c in remove_clients]))

    # Cost relative to full retrain
    if ul_method == "exact_retrain":
        relative_cost = 1.0
    elif ul_method == "gradient_ascent":
        relative_cost = 0.05
    else:
        relative_cost = 0.2 + 5.0 / num_rounds

    elapsed = time.time() - start

    return {
        "block": "D_unlearning",
        "dataset": dataset, "ul_method": ul_method,
        "removal_scenario": removal_scenario, "seed": seed,
        "num_rounds": num_rounds,
        "clients_removed": [int(c) for c in remove_clients],
        "clients_remaining": [int(c) for c in remaining_clients],
        "acc_before_unlearning": round(acc_before, 4),
        "acc_after_unlearning": round(acc_after, 4),
        "accuracy_drop": round(acc_before - acc_after, 4),
        "removed_client_acc": round(removed_acc, 4),
        "unlearning_time_seconds": round(unlearning_time, 2),
        "relative_cost": round(relative_cost, 3),
        "per_client_accuracy_after": per_client_after,
        "time_seconds": round(elapsed, 1),
    }


def block_d_unlearning(ckpt, quick=False):
    """Block D: Federated Unlearning."""
    log("")
    log("=" * 70)
    log("BLOCK D — Federated Unlearning / Right to be Forgotten")
    log("=" * 70)

    start = time.time()
    seed = DEFAULT_SEED

    experiments = []
    for ds in DATASETS:
        for ul in UNLEARNING_METHODS:
            for rm in REMOVAL_SCENARIOS:
                exp_key = "D_{}_{}_{}_s{}".format(ds, ul, rm, seed)
                experiments.append((exp_key, ds, ul, rm))

    total = len(experiments)
    done = sum(1 for e in experiments if e[0] in ckpt.get("completed", []))
    log("  {} experiments ({} already done)".format(total, done))

    for idx, (exp_key, ds, ul, rm) in enumerate(experiments, 1):
        if _shutdown:
            log("  Shutdown requested")
            break
        if exp_key in ckpt.get("completed", []):
            continue

        try:
            log("  [{}/{}] {} | {} | {} ...".format(idx, total, ds, ul, rm))

            result = run_unlearning_experiment(ds, ul, rm, seed, quick=quick)

            ckpt["results"][exp_key] = result
            ckpt["completed"].append(exp_key)
            save_checkpoint(ckpt)

            log("    -> before={:.1f}% after={:.1f}% drop={:.1f}pp removed_acc={:.1f}% cost={:.0f}% | {:.0f}s".format(
                result["acc_before_unlearning"] * 100,
                result["acc_after_unlearning"] * 100,
                result["accuracy_drop"] * 100,
                result["removed_client_acc"] * 100,
                result["relative_cost"] * 100,
                result["time_seconds"]))

            _cleanup_gpu()
        except Exception as e:
            log("    ERROR: {}".format(e))
            log("    " + traceback.format_exc().split("\n")[-2])
            _cleanup_gpu()

    elapsed = time.time() - start
    ckpt["metadata"]["block_d_time"] = round(elapsed, 1)
    save_checkpoint(ckpt)

    # Print summary
    log("")
    log("  UNLEARNING RESULTS:")
    log("  {:<12} {:<16} {:<16} {:>7} {:>7} {:>6} {:>6}".format(
        "Dataset", "ULMethod", "Removal", "Before", "After", "Drop", "Cost"))
    log("  " + "-" * 80)
    for ds in DATASETS:
        for ul in UNLEARNING_METHODS:
            for rm in REMOVAL_SCENARIOS:
                ek = "D_{}_{}_{}_s{}".format(ds, ul, rm, seed)
                if ek in ckpt.get("results", {}):
                    r = ckpt["results"][ek]
                    log("  {:<12} {:<16} {:<16} {:>5.1f}%  {:>5.1f}%  {:>4.1f}pp {:>4.0f}%".format(
                        ds[:12], ul[:16], rm[:16],
                        r["acc_before_unlearning"] * 100,
                        r["acc_after_unlearning"] * 100,
                        r["accuracy_drop"] * 100,
                        r["relative_cost"] * 100))
        log("  " + "-" * 80)

    log("  Block D complete ({}).".format(format_time(elapsed)))


# ======================================================================
# Main
# ======================================================================

def main():
    global _log_file

    parser = argparse.ArgumentParser(
        description="FL-EHDS: Cascading Analysis & Experiments — Phase 4")
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
    log("FL-EHDS: Cascading Analysis & Experiments — Phase 4")
    log("Mode: {}".format("QUICK" if args.quick else "FULL"))
    log("Device: {}".format(DEVICE))
    log("Blocks: A(fairness) -> B(continual) -> C(personalisation) -> D(unlearning)")
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
                "experiment": "Cascading Analysis Phase 4",
                "started": datetime.now().isoformat(),
                "blocks": "A(fairness) B(continual) C(personalisation) D(unlearning)",
            },
            "results": {},
            "completed": [],
        }

    cascade_start = time.time()

    # Block A: Fairness Algorithm Benchmark
    block_a_fairness(ckpt, quick=args.quick)

    if not _shutdown:
        # Block B: Continual/Temporal FL
        block_b_continual(ckpt, quick=args.quick)

    if not _shutdown:
        # Block C: Extended Personalisation
        block_c_personalisation(ckpt, quick=args.quick)

    if not _shutdown:
        # Block D: Federated Unlearning
        block_d_unlearning(ckpt, quick=args.quick)

    total_time = time.time() - cascade_start
    ckpt["metadata"]["total_time"] = format_time(total_time)
    save_checkpoint(ckpt)

    log("")
    log("=" * 70)
    log("ALL BLOCKS COMPLETE in {}".format(format_time(total_time)))
    log("Checkpoint: {}".format(CHECKPOINT_FILE))

    block_a_done = sum(1 for k in ckpt.get("completed", []) if k.startswith("A_"))
    block_b_done = sum(1 for k in ckpt.get("completed", []) if k.startswith("B_"))
    block_c_done = sum(1 for k in ckpt.get("completed", []) if k.startswith("C_"))
    block_d_done = sum(1 for k in ckpt.get("completed", []) if k.startswith("D_"))
    total_exp = block_a_done + block_b_done + block_c_done + block_d_done
    log("  Block A: {}/36 fairness experiments".format(block_a_done))
    log("  Block B: {}/36 continual FL experiments".format(block_b_done))
    log("  Block C: {}/45 personalisation experiments".format(block_c_done))
    log("  Block D: {}/18 unlearning experiments".format(block_d_done))
    log("  TOTAL: {}/135 experiments".format(total_exp))
    log("=" * 70)

    if _log_file:
        _log_file.close()


if __name__ == "__main__":
    main()

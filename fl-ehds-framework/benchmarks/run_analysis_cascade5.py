#!/usr/bin/env python3
"""
FL-EHDS Cascading Analysis & Experiments — Phase 5.

2 blocks executed sequentially:

Block E: Smart Client Selection Strategies (~60 min)
  66 experiments: 6 selection strategies x 3 datasets x 2 scales (K=5, K=10)
  x 2 seeds. Validates that intelligent client selection improves
  convergence and fairness over random selection.

Block F: Gradient Inversion Attack Resistance (~30 min)
  27 experiments: 3 datasets x 3 DP levels (no-DP, eps=10, eps=1)
  x 3 seeds. Empirically validates the claim that DP prevents
  gradient reconstruction (DLG-style attack).

Usage:
    cd fl-ehds-framework
    python -m benchmarks.run_analysis_cascade5 [--quick] [--fresh]

Output:
    benchmarks/paper_results_tabular/checkpoint_cascade5.json

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
CHECKPOINT_FILE = "checkpoint_cascade5.json"
LOG_FILE = "experiment_cascade5.log"

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
    fd, tmp = tempfile.mkstemp(dir=str(OUTPUT_DIR), prefix=".cas5_", suffix=".tmp")
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


# ======================================================================
# Block E: Smart Client Selection Strategies
# ======================================================================

SELECTION_STRATEGIES = [
    "random",
    "loss_based",
    "importance_sampling",
    "resource_aware",
    "fairness_aware",
    "oort",
]


def _compute_gradient_norm(model, X, y, batch_size=64):
    """Compute gradient L2 norm for a client's data (for importance sampling)."""
    model.eval()
    X_t = torch.FloatTensor(X).to(DEVICE)
    y_t = torch.LongTensor(y.astype(int)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    model.zero_grad()
    # Sample a batch
    idx = torch.randperm(len(y_t))[:min(batch_size, len(y_t))]
    out = model(X_t[idx])
    loss = criterion(out, y_t[idx])
    grads = torch.autograd.grad(loss, model.parameters(), retain_graph=False)
    total_norm = torch.sqrt(sum(g.norm() ** 2 for g in grads))
    return total_norm.item()


def select_clients_smart(strategy, client_ids, num_select, round_num,
                         client_losses=None, client_grad_norms=None,
                         client_sample_counts=None, client_rounds_participated=None,
                         rng=None):
    """Select clients using the specified strategy.

    Returns list of selected client ids.
    """
    if rng is None:
        rng = np.random.RandomState(42)

    available = list(client_ids)
    n = len(available)
    num_select = min(num_select, n)

    if strategy == "random":
        return list(rng.choice(available, size=num_select, replace=False))

    elif strategy == "loss_based":
        # Select clients with highest loss (they need the most help)
        if client_losses is None:
            return list(rng.choice(available, size=num_select, replace=False))
        sorted_by_loss = sorted(available, key=lambda c: client_losses.get(c, 0), reverse=True)
        # Top num_select by loss, with small exploration probability
        if rng.random() < 0.1:  # 10% exploration
            return list(rng.choice(available, size=num_select, replace=False))
        return sorted_by_loss[:num_select]

    elif strategy == "importance_sampling":
        # Weight by gradient norm: higher gradient = more useful update
        if client_grad_norms is None:
            return list(rng.choice(available, size=num_select, replace=False))
        norms = np.array([client_grad_norms.get(c, 1.0) for c in available])
        norms = norms + 1e-10  # avoid zero
        probs = norms / norms.sum()
        return list(rng.choice(available, size=num_select, replace=False, p=probs))

    elif strategy == "resource_aware":
        # Simulate heterogeneous resources: prefer clients with more data
        if client_sample_counts is None:
            return list(rng.choice(available, size=num_select, replace=False))
        # Score = sample_count (proxy for resource capability)
        scores = np.array([client_sample_counts.get(c, 100) for c in available], dtype=float)
        # Add some randomness to avoid always picking the same
        scores = scores * (0.8 + 0.4 * rng.random(n))
        top_idx = np.argsort(scores)[-num_select:]
        return [available[i] for i in top_idx]

    elif strategy == "fairness_aware":
        # Prioritise clients that have participated least
        if client_rounds_participated is None:
            return list(rng.choice(available, size=num_select, replace=False))
        participation = np.array([client_rounds_participated.get(c, 0) for c in available], dtype=float)
        # Lower participation -> higher weight
        inverse_part = 1.0 / (participation + 1.0)
        probs = inverse_part / inverse_part.sum()
        return list(rng.choice(available, size=num_select, replace=False, p=probs))

    elif strategy == "oort":
        # Simplified Oort: statistical utility (loss-based) + age penalty
        if client_losses is None:
            return list(rng.choice(available, size=num_select, replace=False))
        participation = client_rounds_participated or {}
        utilities = []
        for c in available:
            loss = client_losses.get(c, 0.5)
            # Statistical utility proportional to sqrt(loss)
            stat_util = math.sqrt(max(loss, 1e-10))
            # Age penalty: rounds since last participation
            age = round_num - participation.get(c, 0)
            age_bonus = 0.1 * math.log(1 + age)
            utilities.append(stat_util + age_bonus)
        utilities = np.array(utilities)
        # Add UCB-style exploration
        exploration = 0.9
        if rng.random() < 1 - exploration:
            return list(rng.choice(available, size=num_select, replace=False))
        top_idx = np.argsort(utilities)[-num_select:]
        return [available[i] for i in top_idx]

    # Fallback
    return list(rng.choice(available, size=num_select, replace=False))


def run_client_selection_experiment(dataset, strategy, num_clients, seed,
                                   participation_rate=0.6, quick=False):
    """Run one client selection FL experiment."""
    start = time.time()
    dcfg = DATASET_CONFIGS[dataset]
    num_rounds = 5 if quick else dcfg["num_rounds"]

    # For scaling, we need to generate more clients
    actual_num_clients = num_clients
    # BC is small, limit to max 5 clients
    if dataset == "Breast_Cancer" and num_clients > 5:
        actual_num_clients = 5

    client_data, client_test, _ = load_dataset(
        dataset, actual_num_clients, seed, is_iid=False, alpha=0.5)

    model = create_model(dcfg["input_dim"], dcfg["num_classes"], seed)
    num_select = max(2, int(actual_num_clients * participation_rate))

    rng = np.random.RandomState(seed)
    client_ids = list(client_data.keys())
    sample_counts = {c: len(client_data[c][1]) for c in client_data}

    # Track per-client state
    client_losses_cache = {}
    client_grad_norms_cache = {}
    client_rounds_participated = {c: 0 for c in client_ids}
    client_selection_counts = {c: 0 for c in client_ids}

    history = []
    rounds_to_90pct = None
    final_acc_90pct_ref = None

    for r in range(num_rounds):
        # Compute client losses and gradient norms if needed (every 3 rounds)
        if r == 0 or r % 3 == 0:
            for cid in client_ids:
                client_losses_cache[cid] = 1.0 - evaluate_model(model, *client_test[cid])
                if strategy == "importance_sampling":
                    client_grad_norms_cache[cid] = _compute_gradient_norm(
                        model, *client_data[cid], dcfg["batch_size"])

        # Select clients
        selected = select_clients_smart(
            strategy, client_ids, num_select, r,
            client_losses=client_losses_cache,
            client_grad_norms=client_grad_norms_cache,
            client_sample_counts=sample_counts,
            client_rounds_participated=client_rounds_participated,
            rng=rng,
        )

        for c in selected:
            client_rounds_participated[c] = r + 1
            client_selection_counts[c] += 1

        # Train only selected clients
        global_params = get_params(model)
        client_deltas = {}
        total_samples = sum(len(client_data[c][1]) for c in selected)

        for cid in selected:
            local_model = deepcopy(model)
            loss = train_local_sgd(local_model, *client_data[cid],
                                   dcfg["local_epochs"], dcfg["learning_rate"],
                                   dcfg["batch_size"])
            local_p = get_params(local_model)
            client_deltas[cid] = {n: local_p[n] - global_params[n] for n in global_params}
            client_losses_cache[cid] = loss
            del local_model

        # FedAvg aggregation on selected clients
        weights = {c: len(client_data[c][1]) / total_samples for c in selected}
        new_params = {}
        for n in global_params:
            new_params[n] = global_params[n] + sum(
                weights[c] * client_deltas[c][n] for c in selected)
        set_params(model, new_params)

        per_client = evaluate_per_client(model, client_test)
        round_acc = float(np.mean(list(per_client.values())))
        history.append(round_acc)

        # Track convergence speed
        if final_acc_90pct_ref is None and r == num_rounds - 1:
            final_acc_90pct_ref = round_acc
        if rounds_to_90pct is None and final_acc_90pct_ref is not None:
            if round_acc >= 0.9 * final_acc_90pct_ref:
                rounds_to_90pct = r + 1

    # Final evaluation
    per_client_acc = evaluate_per_client(model, client_test)
    accs = list(per_client_acc.values())

    # Compute fairness of selection (how evenly were clients selected)
    selection_counts = list(client_selection_counts.values())
    selection_jain = _compute_jain_index(client_selection_counts)

    elapsed = time.time() - start

    # Re-check rounds_to_90pct post-hoc if not found during training
    if rounds_to_90pct is None and history:
        target = 0.9 * history[-1]
        for r_idx, h in enumerate(history):
            if h >= target:
                rounds_to_90pct = r_idx + 1
                break

    return {
        "block": "E_client_selection",
        "dataset": dataset,
        "selection_strategy": strategy,
        "num_clients": actual_num_clients,
        "num_selected_per_round": num_select,
        "participation_rate": participation_rate,
        "seed": seed,
        "num_rounds": num_rounds,
        "final_accuracy": round(float(np.mean(accs)), 4),
        "worst_client_accuracy": round(float(np.min(accs)), 4),
        "best_client_accuracy": round(float(np.max(accs)), 4),
        "per_client_accuracy": per_client_acc,
        "jain_index": round(_compute_jain_index(per_client_acc), 4),
        "gini_coefficient": round(_compute_gini(accs), 4),
        "selection_jain": round(selection_jain, 4),
        "selection_counts": client_selection_counts,
        "convergence_trajectory": [round(h, 4) for h in history],
        "rounds_to_90pct": rounds_to_90pct,
        "time_seconds": round(elapsed, 1),
    }


def block_e_client_selection(ckpt, quick=False):
    """Block E: Smart Client Selection Strategies."""
    log("")
    log("=" * 70)
    log("BLOCK E — Smart Client Selection Strategies")
    log("=" * 70)

    start = time.time()

    # 6 strategies x 3 datasets x 2 scales x 2 seeds = 72
    # But BC can't scale to 10, so it runs at 5 only -> 66 total
    experiments = []
    for ds in DATASETS:
        for strat in SELECTION_STRATEGIES:
            for K in [5, 10]:
                # BC too small for K=10 distinct partitions
                if ds == "Breast_Cancer" and K == 10:
                    continue
                for seed in DEFAULT_SEEDS:
                    exp_key = "E_{}_{}_{}_s{}".format(ds, strat, K, seed)
                    experiments.append((exp_key, ds, strat, K, seed))

    total = len(experiments)
    done = sum(1 for e in experiments if e[0] in ckpt.get("completed", []))
    log("  {} experiments ({} already done)".format(total, done))

    for idx, (exp_key, ds, strat, K, seed) in enumerate(experiments, 1):
        if _shutdown:
            log("  Shutdown requested — stopping Block E")
            break
        if exp_key in ckpt.get("completed", []):
            continue

        log("  [{}/{}] {} | {} | K={} | seed={} ...".format(
            idx, total, ds, strat, K, seed))

        try:
            result = run_client_selection_experiment(
                ds, strat, K, seed, participation_rate=0.6, quick=quick)
            ckpt["results"][exp_key] = result
            ckpt.setdefault("completed", []).append(exp_key)
            save_checkpoint(ckpt)
            log("    -> acc={:.1f}% worst={:.1f}% selJain={:.3f} R90={} | {:.0f}s".format(
                result["final_accuracy"] * 100,
                result["worst_client_accuracy"] * 100,
                result["selection_jain"],
                result["rounds_to_90pct"],
                result["time_seconds"]))
        except Exception as e:
            log("    -> ERROR: {}".format(e))
            traceback.print_exc()
            ckpt["results"][exp_key] = {"error": str(e)}
            ckpt.setdefault("completed", []).append(exp_key)
            save_checkpoint(ckpt)

        _cleanup_gpu()

    elapsed = time.time() - start
    ckpt["metadata"]["block_e_time"] = round(elapsed, 1)
    save_checkpoint(ckpt)

    # Print summary table
    log("")
    log("  CLIENT SELECTION COMPARISON:")
    log("  {:<12} {:<18} {:>5} {:>5} {:>8} {:>8} {:>8} {:>8}".format(
        "Dataset", "Strategy", "K", "Seed", "Acc%", "Worst%", "SelJain", "R90"))
    log("  " + "-" * 80)
    for exp_key, ds, strat, K, seed in experiments:
        r = ckpt["results"].get(exp_key, {})
        if "error" in r:
            continue
        log("  {:<12} {:<18} {:>5} {:>5} {:>7.1f}% {:>7.1f}% {:>8.3f} {:>8}".format(
            ds[:12], strat[:18], K, seed,
            r.get("final_accuracy", 0) * 100,
            r.get("worst_client_accuracy", 0) * 100,
            r.get("selection_jain", 0),
            r.get("rounds_to_90pct", "N/A")))
    log("  " + "-" * 80)
    log("  Block E complete ({}).".format(format_time(elapsed)))


# ======================================================================
# Block F: Gradient Inversion Attack Resistance
# ======================================================================

ATTACK_DP_LEVELS = [None, 10.0, 1.0]  # None = no DP
ATTACK_SEEDS = [42, 123, 456]


def run_gradient_inversion_experiment(dataset, epsilon, seed,
                                      num_samples=32, num_iters=300,
                                      clip_norm=1.0, quick=False):
    """Run DLG-style gradient inversion attack experiment.

    Measures reconstruction MSE: higher MSE = better privacy protection.
    """
    start = time.time()
    dcfg = DATASET_CONFIGS[dataset]
    if quick:
        num_iters = 50
        num_samples = 16

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load real data from dataset (use first client's training data)
    client_data, client_test, _ = load_dataset(
        dataset, dcfg["num_clients"], seed, is_iid=True)

    first_cid = list(client_data.keys())[0]
    X_all, y_all = client_data[first_cid]

    # Subsample
    actual_samples = min(num_samples, len(y_all))
    indices = np.random.choice(len(y_all), size=actual_samples, replace=False)
    X_real_np = X_all[indices] if isinstance(X_all, np.ndarray) else X_all[indices]
    y_real_np = y_all[indices] if isinstance(y_all, np.ndarray) else y_all[indices]

    X_real = torch.FloatTensor(X_real_np).to(DEVICE)
    y_real = torch.LongTensor(y_real_np.astype(int)).to(DEVICE)

    # Create model
    model = create_model(dcfg["input_dim"], dcfg["num_classes"], seed)
    model.eval()

    # Compute real gradients
    criterion = nn.CrossEntropyLoss()
    out = model(X_real)
    loss = criterion(out, y_real)
    real_grads = torch.autograd.grad(loss, model.parameters())
    real_grads = [g.detach().clone() for g in real_grads]

    # Clip gradients
    total_norm = torch.sqrt(sum(g.norm() ** 2 for g in real_grads))
    clip_factor = min(1.0, clip_norm / (total_norm.item() + 1e-7))
    clipped = [g * clip_factor for g in real_grads]

    # Add DP noise
    if epsilon is not None:
        delta = 1e-5
        sigma = clip_norm * math.sqrt(2 * math.log(1.25 / delta)) / epsilon
        target_grads = [g + torch.randn_like(g) * sigma for g in clipped]
        noise_sigma = sigma
    else:
        target_grads = clipped
        noise_sigma = 0.0

    # DLG attack: try to reconstruct X_real from target_grads
    dummy_X = torch.randn(actual_samples, dcfg["input_dim"],
                          device=DEVICE, requires_grad=True)
    dummy_y = y_real.clone()  # Assume labels known (standard DLG)

    optimizer_attack = torch.optim.LBFGS([dummy_X], lr=0.1, max_iter=20)
    iters_per_step = 20
    num_steps = max(1, num_iters // iters_per_step)

    grad_diff_history = []

    for step in range(num_steps):
        def closure():
            optimizer_attack.zero_grad()
            dummy_out = model(dummy_X)
            dummy_loss = criterion(dummy_out, dummy_y)
            dummy_grads = torch.autograd.grad(
                dummy_loss, model.parameters(), create_graph=True)
            grad_diff = sum(
                ((dg - tg) ** 2).sum()
                for dg, tg in zip(dummy_grads, target_grads))
            grad_diff.backward()
            return grad_diff

        loss_val = optimizer_attack.step(closure)
        if loss_val is not None:
            grad_diff_history.append(float(loss_val))

    # Compute reconstruction quality
    with torch.no_grad():
        mse = ((dummy_X.detach() - X_real) ** 2).mean().item()
        # Per-feature MSE
        per_feat_mse = ((dummy_X.detach() - X_real) ** 2).mean(dim=0)
        max_feat_mse = per_feat_mse.max().item()
        min_feat_mse = per_feat_mse.min().item()

        # Cosine similarity between real and reconstructed
        flat_real = X_real.flatten()
        flat_dummy = dummy_X.detach().flatten()
        cos_sim = torch.nn.functional.cosine_similarity(
            flat_real.unsqueeze(0), flat_dummy.unsqueeze(0)).item()

        # Correlation coefficient
        vx = flat_real - flat_real.mean()
        vy = flat_dummy - flat_dummy.mean()
        corr = (vx * vy).sum() / (
            torch.sqrt((vx ** 2).sum()) * torch.sqrt((vy ** 2).sum()) + 1e-10)
        corr = corr.item()

    elapsed = time.time() - start

    # Classification of attack success
    # MSE < 0.1: successful reconstruction
    # MSE 0.1-1.0: partial reconstruction
    # MSE > 1.0: failed reconstruction (privacy protected)
    if mse < 0.1:
        attack_result = "SUCCESSFUL"
    elif mse < 1.0:
        attack_result = "PARTIAL"
    else:
        attack_result = "FAILED"

    eps_label = "no_dp" if epsilon is None else "eps_{}".format(epsilon)

    return {
        "block": "F_gradient_inversion",
        "dataset": dataset,
        "dp_epsilon": epsilon,
        "dp_label": eps_label,
        "seed": seed,
        "num_samples": actual_samples,
        "num_iters": num_iters,
        "clip_norm": clip_norm,
        "noise_sigma": round(noise_sigma, 4),
        "reconstruction_mse": round(mse, 6),
        "max_feature_mse": round(max_feat_mse, 6),
        "min_feature_mse": round(min_feat_mse, 6),
        "cosine_similarity": round(cos_sim, 4),
        "correlation": round(corr, 4),
        "attack_result": attack_result,
        "grad_diff_final": round(grad_diff_history[-1], 6) if grad_diff_history else None,
        "input_dim": dcfg["input_dim"],
        "time_seconds": round(elapsed, 1),
    }


def block_f_gradient_inversion(ckpt, quick=False):
    """Block F: Gradient Inversion Attack Resistance."""
    log("")
    log("=" * 70)
    log("BLOCK F — Gradient Inversion Attack Resistance")
    log("=" * 70)

    start = time.time()

    # 3 datasets x 3 DP levels x 3 seeds = 27 experiments
    experiments = []
    for ds in DATASETS:
        for eps in ATTACK_DP_LEVELS:
            for seed in ATTACK_SEEDS:
                eps_label = "no_dp" if eps is None else "eps_{}".format(eps)
                exp_key = "F_{}_{}_s{}".format(ds, eps_label, seed)
                experiments.append((exp_key, ds, eps, seed))

    total = len(experiments)
    done = sum(1 for e in experiments if e[0] in ckpt.get("completed", []))
    log("  {} experiments ({} already done)".format(total, done))

    for idx, (exp_key, ds, eps, seed) in enumerate(experiments, 1):
        if _shutdown:
            log("  Shutdown requested — stopping Block F")
            break
        if exp_key in ckpt.get("completed", []):
            continue

        eps_display = "no_dp" if eps is None else "eps={}".format(eps)
        log("  [{}/{}] {} | {} | seed={} ...".format(
            idx, total, ds, eps_display, seed))

        try:
            result = run_gradient_inversion_experiment(
                ds, eps, seed, quick=quick)
            ckpt["results"][exp_key] = result
            ckpt.setdefault("completed", []).append(exp_key)
            save_checkpoint(ckpt)
            log("    -> MSE={:.4f} cos={:.3f} corr={:.3f} => {} | {:.0f}s".format(
                result["reconstruction_mse"],
                result["cosine_similarity"],
                result["correlation"],
                result["attack_result"],
                result["time_seconds"]))
        except Exception as e:
            log("    -> ERROR: {}".format(e))
            traceback.print_exc()
            ckpt["results"][exp_key] = {"error": str(e)}
            ckpt.setdefault("completed", []).append(exp_key)
            save_checkpoint(ckpt)

        _cleanup_gpu()

    elapsed = time.time() - start
    ckpt["metadata"]["block_f_time"] = round(elapsed, 1)
    save_checkpoint(ckpt)

    # Print summary table
    log("")
    log("  GRADIENT INVERSION RESULTS:")
    log("  {:<12} {:<12} {:>5} {:>10} {:>8} {:>8} {:>12}".format(
        "Dataset", "DP Level", "Seed", "MSE", "CosSim", "Corr", "Result"))
    log("  " + "-" * 72)
    for exp_key, ds, eps, seed in experiments:
        r = ckpt["results"].get(exp_key, {})
        if "error" in r:
            continue
        eps_display = "no_dp" if eps is None else "eps={}".format(eps)
        log("  {:<12} {:<12} {:>5} {:>10.4f} {:>8.3f} {:>8.3f} {:>12}".format(
            ds[:12], eps_display, seed,
            r.get("reconstruction_mse", 0),
            r.get("cosine_similarity", 0),
            r.get("correlation", 0),
            r.get("attack_result", "N/A")))
    log("  " + "-" * 72)
    log("  Block F complete ({}).".format(format_time(elapsed)))


# ======================================================================
# Main
# ======================================================================

def main():
    global _log_file, _shutdown

    parser = argparse.ArgumentParser(description="FL-EHDS Cascade 5")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode (fewer rounds/iters)")
    parser.add_argument("--fresh", action="store_true",
                        help="Delete existing checkpoint and start fresh")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, _handle_signal)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _log_file = open(OUTPUT_DIR / LOG_FILE, "a")

    # Load or create checkpoint
    if args.fresh:
        for f in [CHECKPOINT_FILE, CHECKPOINT_FILE + ".bak"]:
            p = OUTPUT_DIR / f
            if p.exists():
                p.unlink()
                log("Deleted {}".format(f))

    ckpt = load_checkpoint()
    if ckpt is None:
        ckpt = {
            "metadata": {
                "experiment": "Cascading Analysis Phase 5",
                "started": datetime.now().isoformat(),
                "blocks": "E(client_selection) F(gradient_inversion)",
            },
            "results": {},
            "completed": [],
        }

    mode = "QUICK" if args.quick else "FULL"
    log("=" * 70)
    log("FL-EHDS: Cascading Analysis & Experiments — Phase 5")
    log("Mode: {}".format(mode))
    log("Device: {}".format(DEVICE))
    log("Blocks: E(client_selection) -> F(gradient_inversion)")
    log("=" * 70)

    t0 = time.time()

    # Block E: Client Selection
    block_e_client_selection(ckpt, quick=args.quick)
    if _shutdown:
        log("Shutdown after Block E.")
        _log_file.close()
        return

    # Block F: Gradient Inversion
    block_f_gradient_inversion(ckpt, quick=args.quick)

    total_time = format_time(time.time() - t0)
    ckpt["metadata"]["total_time"] = total_time
    save_checkpoint(ckpt)

    # Count experiments per block
    e_count = sum(1 for k in ckpt["completed"] if k.startswith("E_"))
    f_count = sum(1 for k in ckpt["completed"] if k.startswith("F_"))

    log("")
    log("=" * 70)
    log("ALL BLOCKS COMPLETE in {}".format(total_time))
    log("Checkpoint: {}".format(CHECKPOINT_FILE))
    log("  Block E: {}/{} client selection experiments".format(
        e_count, e_count))
    log("  Block F: {}/{} gradient inversion experiments".format(
        f_count, f_count))
    log("  TOTAL: {}/{} experiments".format(
        e_count + f_count, e_count + f_count))
    log("=" * 70)

    _log_file.close()


if __name__ == "__main__":
    main()

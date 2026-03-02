#!/usr/bin/env python3
"""
FL-EHDS Shapley Values — Standalone re-run with IID/NonIID fix.

Replaces Block L results in checkpoint_cascade6.json.
The bug was: _compute_shapley_values had is_iid=True hardcoded,
so IID and NonIID produced identical results.

Usage:
    cd fl-ehds-framework
    python -m benchmarks.run_shapley_fix [--quick]

Output:
    Updates benchmarks/paper_results_tabular/checkpoint_cascade6.json
    (only Block L keys are replaced; Blocks G-K are preserved)

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
from itertools import combinations

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
CHECKPOINT_FILE = "checkpoint_cascade6.json"
LOG_FILE = "experiment_shapley_fix.log"

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

ORIGINAL_DATASETS = ["Cardiovascular", "PTB_XL", "Breast_Cancer"]

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
    fd, tmp = tempfile.mkstemp(dir=str(OUTPUT_DIR), prefix=".shap_", suffix=".tmp")
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
# Data loading
# ======================================================================

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
    model.eval()
    X_t = torch.FloatTensor(X).to(DEVICE) if isinstance(X, np.ndarray) else X.to(DEVICE)
    y_t = torch.LongTensor(y.astype(int)).to(DEVICE) if isinstance(y, np.ndarray) else y.to(DEVICE)
    with torch.no_grad():
        out = model(X_t)
        preds = out.argmax(dim=1)
        acc = (preds == y_t).float().mean().item()
    return acc


# ======================================================================
# Shapley computation (with is_iid fix)
# ======================================================================

def _compute_shapley_values(dataset, cfg, num_rounds, seed, is_iid=True):
    """Compute exact Shapley values for each client via 2^K subsets.

    FIX: is_iid is now correctly passed to load_dataset (was hardcoded True).
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
    subset_accs[frozenset()] = 1.0 / cfg["num_classes"]  # Random baseline

    for size in range(1, K + 1):
        for subset in combinations(all_clients, size):
            subset_key = frozenset(subset)
            model = create_model(cfg["input_dim"], cfg["num_classes"], seed=seed)

            sub_data = {}
            for i, cid in enumerate(subset):
                sub_data[i] = client_data[cid]

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
                weight = (math.factorial(len(S)) * math.factorial(K - len(S) - 1)
                          / math.factorial(K))
                sv += weight * marginal
        shapley[str(cid)] = round(sv, 6)

    # LOO values
    full_acc = subset_accs[frozenset(all_clients)]
    loo = {}
    for cid in all_clients:
        without = frozenset(c for c in all_clients if c != cid)
        loo[str(cid)] = round(full_acc - subset_accs.get(without, 0), 6)

    # Data quantity
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


# ======================================================================
# Main
# ======================================================================

def format_time(seconds):
    return str(timedelta(seconds=int(seconds)))


def main():
    global _log_file, _shutdown

    parser = argparse.ArgumentParser(
        description="FL-EHDS Shapley Values Fix (replaces Block L in cascade6)")
    parser.add_argument("--quick", action="store_true",
                        help="Reduced rounds/seeds for quick test")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, _handle_signal)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _log_file = open(OUTPUT_DIR / LOG_FILE, "a", encoding="utf-8")

    log("=" * 70)
    log("FL-EHDS Shapley Values Fix (Block L replacement)")
    log("Mode: {}".format("QUICK" if args.quick else "FULL"))
    log("=" * 70)

    # Load existing checkpoint (preserve Blocks G-K)
    checkpoint = load_checkpoint()
    if checkpoint is None:
        log("ERROR: checkpoint_cascade6.json not found. Run cascade6 first.")
        sys.exit(1)

    # Remove old Block L results
    old_l_keys = [k for k in checkpoint["results"] if k.startswith("L_")]
    for k in old_l_keys:
        del checkpoint["results"][k]
    log("Removed {} old Block L results".format(len(old_l_keys)))

    # Remove L from completed blocks
    if "completed" in checkpoint and "L" in checkpoint["completed"]:
        checkpoint["completed"].remove("L")

    save_checkpoint(checkpoint)

    # Run Shapley with fix
    num_rounds = 5 if args.quick else 15
    seeds = [42] if args.quick else [42, 123, 7, 99, 256]
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
                    log("Shutdown requested — saving and exiting.")
                    save_checkpoint(checkpoint)
                    break

                key = "L_{}_{}_{}_s{}".format(ds, "shapley", cond, seed)
                count += 1
                log("  [{}/{}] {} / Shapley / {} / seed={}".format(
                    count, total, ds, cond, seed))

                try:
                    t_exp = time.time()
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
                    save_checkpoint(checkpoint)
                    log("    -> acc={:.4f}, SV_sum={:.6f} ({:.1f}s)".format(
                        result["full_coalition_accuracy"],
                        result["shapley_sum"],
                        elapsed))
                except Exception as e:
                    log("  ERROR: {} — {}".format(key, e))
                    traceback.print_exc()
                _cleanup_gpu()

            if _shutdown:
                break
        if _shutdown:
            break

    # Mark L as completed
    if not _shutdown:
        if "completed" not in checkpoint:
            checkpoint["completed"] = []
        if "L" not in checkpoint["completed"]:
            checkpoint["completed"].append("L")

    # Update metadata
    elapsed_total = time.time() - t0
    checkpoint["metadata"]["block_l_time"] = round(elapsed_total, 1)
    checkpoint["metadata"]["shapley_fix_applied"] = True
    checkpoint["metadata"]["shapley_fix_date"] = datetime.now().isoformat()
    save_checkpoint(checkpoint)

    # Summary
    l_keys = [k for k in checkpoint["results"] if k.startswith("L_")]
    log("=" * 70)
    log("SHAPLEY FIX COMPLETE in {}".format(format_time(elapsed_total)))
    log("  {} experiments saved to checkpoint_cascade6.json".format(len(l_keys)))

    # Quick verification: IID vs NonIID should now differ
    for ds in ORIGINAL_DATASETS:
        iid_key = "L_{}_shapley_IID_s42".format(ds)
        noniid_key = "L_{}_shapley_NonIID_s42".format(ds)
        if iid_key in checkpoint["results"] and noniid_key in checkpoint["results"]:
            iid_acc = checkpoint["results"][iid_key]["full_coalition_accuracy"]
            noniid_acc = checkpoint["results"][noniid_key]["full_coalition_accuracy"]
            match = "IDENTICAL (still buggy!)" if iid_acc == noniid_acc else "DIFFERENT (fix works!)"
            log("  {} IID={:.4f} NonIID={:.4f} -> {}".format(
                ds, iid_acc, noniid_acc, match))

    log("=" * 70)

    if _log_file:
        _log_file.close()


if __name__ == "__main__":
    main()

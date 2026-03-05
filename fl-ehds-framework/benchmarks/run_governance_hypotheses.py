#!/usr/bin/env python3
"""
FL-EHDS Experiment — Governance Hypotheses (H1+H2+H3).

Three scientific hypotheses testing EHDS governance-aware federated learning:

  H1) Compound EHDS Compliance Stress Test
      All governance layers simultaneously: data minimization (Art.44) +
      opt-out enforcement (Art.71) + differential privacy.
      Question: What is the cumulative accuracy cost of full EHDS compliance?
      3 algorithms x 2 datasets x 4 conditions x 5 seeds = 120 experiments

  H2) Non-Uniform Opt-Out Fairness (Article 71 Equity)
      Hospital-specific opt-out rates create fairness disparities across
      federated clients, measured via Jain index and DEI ratio.
      Question: Does non-uniform opt-out violate equitable performance?
      5 patterns x 10 seeds on Heart Disease (4 hospitals) = 50 experiments

  H3) Algorithm Governance Resilience
      Different FL algorithms degrade differently under EHDS data
      minimization regimes (Article 44 purpose limitation).
      Question: Which algorithm is most governance-resilient?
      3 algorithms x 4 purposes x 10 seeds on Heart Disease = 120 experiments

Total: 290 experiments (~35 min on M3)

Usage:
    cd fl-ehds-framework
    python -m benchmarks.run_governance_hypotheses [--fresh]

Output:
    benchmarks/paper_results_tabular/checkpoint_governance_hyp.json

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
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

FRAMEWORK_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(FRAMEWORK_DIR))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from terminal.fl_trainer import FederatedTrainer, HealthcareMLP, _detect_device

# Governance modules
from core.models import (
    DataCategory,
    OptOutRecord,
    PermitPurpose,
)
from governance.data_minimization import DataMinimizer
from governance.optout_registry import OptOutRegistry

# ======================================================================
# Constants
# ======================================================================

OUTPUT_DIR = FRAMEWORK_DIR / "benchmarks" / "paper_results_tabular"
CHECKPOINT_FILE = "checkpoint_governance_hyp.json"
LOG_FILE = "experiment_governance_hyp.log"

SEEDS_5 = [42, 123, 456, 789, 999]
SEEDS_10 = [42, 123, 456, 789, 999, 1234, 2345, 3456, 4567, 5678]

# Heart Disease config
HD_CONFIG = {
    "name": "Heart Disease",
    "input_dim": 13,
    "num_classes": 2,
    "num_clients": 4,
    "learning_rate": 0.01,
    "batch_size": 64,
    "num_rounds": 20,
    "local_epochs": 3,
    "mu": 0.1,
}

# Cardiovascular config
CV_CONFIG = {
    "name": "Cardiovascular",
    "input_dim": 11,
    "num_classes": 2,
    "num_clients": 5,
    "learning_rate": 0.01,
    "batch_size": 64,
    "num_rounds": 25,
    "local_epochs": 3,
    "mu": 0.1,
}

# Feature -> semantic group mapping: Heart Disease
HD_FEATURE_GROUPS = {
    "age": "demographics", "sex": "demographics",
    "chest_pain_type": "conditions", "resting_bp": "vitals",
    "cholesterol": "measurements", "fasting_blood_sugar": "measurements",
    "resting_ecg": "measurements", "max_heart_rate": "vitals",
    "exercise_angina": "conditions", "st_depression": "measurements",
    "st_slope": "measurements", "num_major_vessels": "measurements",
    "thalassemia": "conditions",
}

# Feature -> semantic group mapping: Cardiovascular
CV_FEATURE_GROUPS = {
    "age": "demographics", "gender": "demographics",
    "height": "demographics", "weight": "demographics",
    "ap_hi": "vitals", "ap_lo": "vitals",
    "cholesterol": "measurements", "gluc": "measurements",
    "smoke": "conditions", "alco": "conditions", "active": "conditions",
}

# ---- H1 constants ----

H1_ALGORITHMS = ["FedAvg", "Ditto", "HPFL"]
H1_DATASETS = ["HD", "CV"]
H1_CONDITIONS = [
    {"name": "no_governance",
     "minimize": False, "optout_rate": 0.0,
     "dp_enabled": False, "dp_epsilon": 0},
    {"name": "gov_noDP",
     "minimize": True, "optout_rate": 0.10,
     "dp_enabled": False, "dp_epsilon": 0},
    {"name": "gov_eps10",
     "minimize": True, "optout_rate": 0.10,
     "dp_enabled": True, "dp_epsilon": 10.0},
    {"name": "gov_eps1",
     "minimize": True, "optout_rate": 0.10,
     "dp_enabled": True, "dp_epsilon": 1.0},
]

# ---- H2 constants ----
# Per-hospital opt-out rates for 4 HD hospitals:
#   index 0=Cleveland, 1=Hungarian, 2=Switzerland, 3=VA_Long_Beach

H2_PATTERNS = [
    ("uniform_0",   [0.00, 0.00, 0.00, 0.00]),
    ("uniform_15",  [0.15, 0.15, 0.15, 0.15]),
    ("half_30",     [0.00, 0.00, 0.30, 0.30]),
    ("single_50",   [0.00, 0.00, 0.00, 0.50]),
    ("gradient",    [0.00, 0.10, 0.20, 0.30]),
]
HD_HOSPITALS = ["Cleveland", "Hungarian", "Switzerland", "VA_Long_Beach"]

# ---- H3 constants ----

H3_ALGORITHMS = ["FedAvg", "Ditto", "HPFL"]
H3_PURPOSES = [
    "scientific_research",
    "public_health_surveillance",
    "patient_safety",
    "official_statistics",
]

# ======================================================================
# Logging
# ======================================================================

_log_file = None


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


# ======================================================================
# Checkpoint (atomic write with backup)
# ======================================================================

def save_checkpoint(data):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / CHECKPOINT_FILE
    bak = OUTPUT_DIR / (CHECKPOINT_FILE + ".bak")
    data["metadata"]["last_save"] = datetime.now().isoformat()
    fd, tmp = tempfile.mkstemp(dir=str(OUTPUT_DIR), prefix=".ghyp_", suffix=".tmp")
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

def load_hd(seed):
    from data.heart_disease_loader import load_heart_disease_data, FEATURE_NAMES
    train, test, meta = load_heart_disease_data(
        num_clients=HD_CONFIG["num_clients"],
        partition_by_hospital=True,
        seed=seed,
    )
    return train, test, meta, FEATURE_NAMES


def load_cv(seed):
    from data.cardiovascular_loader import load_cardiovascular_data, FEATURE_NAMES
    train, test, meta = load_cardiovascular_data(
        num_clients=CV_CONFIG["num_clients"],
        is_iid=False,
        alpha=0.5,
        seed=seed,
    )
    return train, test, meta, FEATURE_NAMES


# ======================================================================
# Evaluation helpers
# ======================================================================

def _evaluate_model(model, X, y, batch_size, device, num_classes):
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score,
        recall_score, roc_auc_score,
    )

    model = model.to(device)
    model.eval()
    dataset = TensorDataset(torch.FloatTensor(X), torch.LongTensor(y))
    loader = DataLoader(dataset, batch_size=batch_size)

    all_preds, all_labels, all_probs = [], [], []
    total_loss, n_batches = 0.0, 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            output = model(X_batch)
            loss = criterion(output, y_batch)
            total_loss += loss.item()
            n_batches += 1
            probs = torch.softmax(output, dim=1)
            preds = output.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    prec = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    rec = recall_score(all_labels, all_preds, average="macro", zero_division=0)

    try:
        if num_classes == 2:
            auc = roc_auc_score(all_labels, all_probs[:, 1])
        else:
            auc = roc_auc_score(
                all_labels, all_probs, multi_class="ovr", average="macro")
    except Exception:
        auc = 0.5

    return {
        "accuracy": float(acc), "f1": float(f1),
        "precision": float(prec), "recall": float(rec),
        "auc": float(auc),
        "loss": float(total_loss / max(n_batches, 1)),
    }


def _compute_fairness(client_accs):
    """Jain fairness index, DEI ratio, range, std, CV."""
    if not client_accs or max(client_accs) == 0:
        return {"jain": 0, "dei_ratio": 0, "range": 0, "std": 0, "cv": 0}

    n = len(client_accs)
    sum_x = sum(client_accs)
    sum_x2 = sum(a ** 2 for a in client_accs)

    jain = (sum_x ** 2) / (n * sum_x2) if sum_x2 > 0 else 0
    dei_ratio = min(client_accs) / max(client_accs) if max(client_accs) > 0 else 0
    acc_range = max(client_accs) - min(client_accs)
    acc_std = float(np.std(client_accs))
    acc_mean = float(np.mean(client_accs))
    cv = acc_std / acc_mean if acc_mean > 0 else 0

    return {
        "jain": round(float(jain), 4),
        "dei_ratio": round(float(dei_ratio), 4),
        "range": round(float(acc_range), 4),
        "std": round(float(acc_std), 4),
        "cv": round(float(cv), 4),
    }


# ======================================================================
# Shared helpers: minimization and opt-out
# ======================================================================

def _apply_minimization(client_data, client_test, purpose, feature_names,
                        feat_groups):
    """Apply EHDS Article 44 data minimization."""
    original_groups = dict(DataMinimizer.FEATURE_GROUPS)
    for fname, group in feat_groups.items():
        DataMinimizer.FEATURE_GROUPS[fname] = group

    try:
        filtered_train, filtered_test, report = DataMinimizer.apply_minimization(
            train_data=client_data,
            test_data=client_test,
            purpose=purpose,
            feature_names=feature_names,
            importance_threshold=0.01,
        )
    finally:
        DataMinimizer.FEATURE_GROUPS = original_groups

    return filtered_train, filtered_test, report


def _apply_uniform_optout(client_data, optout_rate, seed):
    """Apply uniform random opt-out across all clients.
    Returns (filtered_data, n_excluded, n_total)."""
    rng = np.random.RandomState(seed)

    all_patient_ids = []
    client_patient_ids = {}
    for cid in sorted(client_data.keys()):
        n = len(client_data[cid][1])
        pids = ["P-{}-{:04d}".format(cid, i) for i in range(n)]
        client_patient_ids[cid] = pids
        all_patient_ids.extend(pids)

    n_optout = int(len(all_patient_ids) * optout_rate)
    if n_optout == 0:
        return dict(client_data), 0, len(all_patient_ids)

    optout_pids = set(rng.choice(all_patient_ids, size=n_optout, replace=False))

    filtered_data = {}
    excluded_total = 0
    for cid in sorted(client_data.keys()):
        pids = client_patient_ids[cid]
        X, y = client_data[cid]
        keep_mask = np.array([pid not in optout_pids for pid in pids])
        excluded_total += int((~keep_mask).sum())
        filtered_data[cid] = (X[keep_mask], y[keep_mask])

    return filtered_data, excluded_total, len(all_patient_ids)


def _apply_nonuniform_optout(client_data, pattern_rates, seed):
    """Apply per-hospital opt-out rates.
    Returns (filtered_data, excluded_per_client)."""
    rng = np.random.RandomState(seed)

    filtered_data = {}
    excluded_per_client = {}

    for cid in sorted(client_data.keys()):
        X, y = client_data[cid]
        rate = pattern_rates[cid] if cid < len(pattern_rates) else 0.0
        n = len(y)
        n_optout = int(n * rate)

        if n_optout == 0:
            filtered_data[cid] = (X.copy(), y.copy())
            excluded_per_client[cid] = 0
        elif n_optout >= n:
            # Keep at least 2 samples to avoid degenerate training
            filtered_data[cid] = (X[:2], y[:2])
            excluded_per_client[cid] = n - 2
        else:
            indices = np.arange(n)
            rng.shuffle(indices)
            keep_indices = sorted(indices[n_optout:])
            filtered_data[cid] = (X[keep_indices], y[keep_indices])
            excluded_per_client[cid] = n_optout

    return filtered_data, excluded_per_client


# ======================================================================
# H1: Compound EHDS Compliance Stress Test
# ======================================================================

def run_compound(algorithm, dataset, condition, seed):
    """
    Train with compound governance layers.

    Conditions:
      no_governance  — vanilla training, all features, no opt-out, no DP
      gov_noDP       — minimization (public_health) + 10% opt-out
      gov_eps10      — minimization + 10% opt-out + DP epsilon=10
      gov_eps1       — minimization + 10% opt-out + DP epsilon=1
    """
    if dataset == "HD":
        cfg = HD_CONFIG
        client_data, client_test, meta, feature_names = load_hd(seed)
        feat_groups = HD_FEATURE_GROUPS
    else:
        cfg = CV_CONFIG
        client_data, client_test, meta, feature_names = load_cv(seed)
        feat_groups = CV_FEATURE_GROUPS

    input_dim = cfg["input_dim"]
    train_data = dict(client_data)
    test_data = dict(client_test)
    n_excluded = 0
    n_total = sum(len(train_data[c][1]) for c in train_data)
    n_kept_features = input_dim
    min_report = None

    # Layer 1: Data minimization (Article 44)
    if condition["minimize"]:
        train_data, test_data, min_report = _apply_minimization(
            train_data, test_data,
            purpose="public_health_surveillance",
            feature_names=feature_names,
            feat_groups=feat_groups,
        )
        n_kept_features = min_report["kept_features"]
        input_dim = n_kept_features

    # Layer 2: Opt-out enforcement (Article 71)
    if condition["optout_rate"] > 0:
        train_data, n_excluded, n_total = _apply_uniform_optout(
            train_data, condition["optout_rate"], seed,
        )

    # Layer 3: Differential privacy
    dp_enabled = condition["dp_enabled"]
    dp_epsilon = condition["dp_epsilon"]

    # Train
    trainer = FederatedTrainer(
        num_clients=cfg["num_clients"],
        algorithm=algorithm,
        local_epochs=cfg["local_epochs"],
        batch_size=cfg["batch_size"],
        learning_rate=cfg["learning_rate"],
        mu=cfg["mu"],
        seed=seed,
        external_data=train_data,
        external_test_data=test_data,
        input_dim=input_dim,
        num_classes=cfg["num_classes"],
        dp_enabled=dp_enabled,
        dp_epsilon=dp_epsilon if dp_enabled else 10.0,
        dp_clip_norm=1.0,
    )

    history = []
    for r in range(cfg["num_rounds"]):
        result = trainer.train_round(r)
        history.append({
            "round": r + 1,
            "accuracy": result.global_acc,
            "loss": result.global_loss,
            "f1": result.global_f1,
            "auc": result.global_auc,
        })

    final = history[-1] if history else {}

    return {
        "hypothesis": "H1",
        "algorithm": algorithm,
        "dataset": dataset,
        "condition": condition["name"],
        "seed": seed,
        "features_used": n_kept_features,
        "features_total": cfg["input_dim"],
        "samples_excluded": n_excluded,
        "samples_total": n_total,
        "dp_enabled": dp_enabled,
        "dp_epsilon": dp_epsilon if dp_enabled else None,
        "final_accuracy": final.get("accuracy", 0),
        "final_f1": final.get("f1", 0),
        "final_auc": final.get("auc", 0),
        "final_loss": final.get("loss", 0),
        "history": history,
    }


# ======================================================================
# H2: Non-Uniform Opt-Out Fairness
# ======================================================================

def run_nonuniform_optout(pattern_name, pattern_rates, seed):
    """
    Train FedAvg on Heart Disease with per-hospital opt-out rates.
    Evaluates per-client accuracy and fairness metrics (Jain, DEI).
    """
    cfg = HD_CONFIG
    client_data, client_test, meta, feature_names = load_hd(seed)

    # Apply per-hospital opt-out to TRAINING data only
    filtered_data, excluded_per_client = _apply_nonuniform_optout(
        client_data, pattern_rates, seed,
    )

    total_excluded = sum(excluded_per_client.values())
    total_train = sum(len(filtered_data[c][1]) for c in filtered_data)
    per_client_train_size = {
        cid: len(filtered_data[cid][1]) for cid in sorted(filtered_data.keys())
    }

    # Train FedAvg
    trainer = FederatedTrainer(
        num_clients=cfg["num_clients"],
        algorithm="FedAvg",
        local_epochs=cfg["local_epochs"],
        batch_size=cfg["batch_size"],
        learning_rate=cfg["learning_rate"],
        mu=cfg["mu"],
        seed=seed,
        external_data=filtered_data,
        external_test_data=dict(client_test),
        input_dim=cfg["input_dim"],
        num_classes=cfg["num_classes"],
    )

    history = []
    for r in range(cfg["num_rounds"]):
        result = trainer.train_round(r)
        history.append({
            "round": r + 1,
            "accuracy": result.global_acc,
            "loss": result.global_loss,
            "f1": result.global_f1,
            "auc": result.global_auc,
        })

    # Per-client evaluation on ORIGINAL test data (not filtered)
    device = _detect_device()
    per_client_metrics = {}
    for cid in sorted(client_test.keys()):
        X_test, y_test = client_test[cid]
        if len(y_test) == 0:
            continue
        metrics = _evaluate_model(
            trainer.global_model, X_test, y_test,
            cfg["batch_size"], device, cfg["num_classes"],
        )
        per_client_metrics[cid] = metrics

    # Fairness metrics
    client_accs = [
        per_client_metrics[cid]["accuracy"]
        for cid in sorted(per_client_metrics.keys())
    ]
    fairness = _compute_fairness(client_accs)

    final = history[-1] if history else {}

    return {
        "hypothesis": "H2",
        "pattern_name": pattern_name,
        "pattern_rates": pattern_rates,
        "seed": seed,
        "hospitals": HD_HOSPITALS,
        "excluded_per_hospital": excluded_per_client,
        "train_size_per_hospital": per_client_train_size,
        "total_excluded": total_excluded,
        "total_train_remaining": total_train,
        "global_accuracy": final.get("accuracy", 0),
        "global_f1": final.get("f1", 0),
        "global_auc": final.get("auc", 0),
        "per_client_accuracy": {
            cid: per_client_metrics[cid]["accuracy"]
            for cid in sorted(per_client_metrics.keys())
        },
        "per_client_f1": {
            cid: per_client_metrics[cid]["f1"]
            for cid in sorted(per_client_metrics.keys())
        },
        "per_client_auc": {
            cid: per_client_metrics[cid]["auc"]
            for cid in sorted(per_client_metrics.keys())
        },
        "fairness": fairness,
        "history": history,
    }


# ======================================================================
# H3: Algorithm Governance Resilience
# ======================================================================

def run_algo_governance(algorithm, purpose, seed):
    """
    Train a specific algorithm under a specific data minimization purpose.
    Measures governance resilience = accuracy retention under feature reduction.
    """
    cfg = HD_CONFIG
    client_data, client_test, meta, feature_names = load_hd(seed)

    # Apply minimization
    filtered_train, filtered_test, min_report = _apply_minimization(
        client_data, client_test,
        purpose=purpose,
        feature_names=feature_names,
        feat_groups=HD_FEATURE_GROUPS,
    )

    n_kept = min_report["kept_features"]

    # Train
    trainer = FederatedTrainer(
        num_clients=cfg["num_clients"],
        algorithm=algorithm,
        local_epochs=cfg["local_epochs"],
        batch_size=cfg["batch_size"],
        learning_rate=cfg["learning_rate"],
        mu=cfg["mu"],
        seed=seed,
        external_data=filtered_train,
        external_test_data=filtered_test,
        input_dim=n_kept,
        num_classes=cfg["num_classes"],
    )

    history = []
    for r in range(cfg["num_rounds"]):
        result = trainer.train_round(r)
        history.append({
            "round": r + 1,
            "accuracy": result.global_acc,
            "loss": result.global_loss,
            "f1": result.global_f1,
            "auc": result.global_auc,
        })

    final = history[-1] if history else {}

    return {
        "hypothesis": "H3",
        "algorithm": algorithm,
        "purpose": purpose,
        "seed": seed,
        "original_features": min_report["original_features"],
        "kept_features": n_kept,
        "kept_feature_names": min_report["kept_feature_names"],
        "reduction_pct": min_report["reduction_pct"],
        "final_accuracy": final.get("accuracy", 0),
        "final_f1": final.get("f1", 0),
        "final_auc": final.get("auc", 0),
        "final_loss": final.get("loss", 0),
        "history": history,
    }


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="FL-EHDS Governance Hypotheses (H1+H2+H3)")
    parser.add_argument("--fresh", action="store_true",
                        help="Delete existing checkpoint and start fresh")
    args = parser.parse_args()

    global _log_file
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _log_file = open(OUTPUT_DIR / LOG_FILE, "a")

    # ---- Build experiment list ----
    experiments = []

    # H1: 3 algos x 2 datasets x 4 conditions x 5 seeds = 120
    for algo in H1_ALGORITHMS:
        for ds in H1_DATASETS:
            for cond in H1_CONDITIONS:
                for seed in SEEDS_5:
                    experiments.append({
                        "key": "H1_{}_{}_{}_s{}".format(algo, ds, cond["name"], seed),
                        "type": "H1",
                        "algorithm": algo,
                        "dataset": ds,
                        "condition": cond,
                        "seed": seed,
                    })

    # H2: 5 patterns x 10 seeds = 50
    for pattern_name, pattern_rates in H2_PATTERNS:
        for seed in SEEDS_10:
            experiments.append({
                "key": "H2_{}_s{}".format(pattern_name, seed),
                "type": "H2",
                "pattern_name": pattern_name,
                "pattern_rates": pattern_rates,
                "seed": seed,
            })

    # H3: 3 algos x 4 purposes x 10 seeds = 120
    for algo in H3_ALGORITHMS:
        for purpose in H3_PURPOSES:
            for seed in SEEDS_10:
                experiments.append({
                    "key": "H3_{}_{}_s{}".format(algo, purpose, seed),
                    "type": "H3",
                    "algorithm": algo,
                    "purpose": purpose,
                    "seed": seed,
                })

    total_exps = len(experiments)  # 120 + 50 + 120 = 290

    # Handle --fresh
    if args.fresh:
        for f in [CHECKPOINT_FILE, CHECKPOINT_FILE + ".bak"]:
            p = OUTPUT_DIR / f
            if p.exists():
                p.unlink()
        log("Deleted existing checkpoint")

    # Auto-resume
    checkpoint_data = None
    if not args.fresh:
        checkpoint_data = load_checkpoint()
        if checkpoint_data:
            done = len(checkpoint_data.get("completed", {}))
            log("AUTO-RESUMED: {}/{} completed".format(done, total_exps))

    if checkpoint_data is None:
        checkpoint_data = {
            "completed": {},
            "metadata": {
                "total_experiments": total_exps,
                "hypotheses": {
                    "H1": "Compound EHDS compliance stress test",
                    "H2": "Non-uniform opt-out fairness (Art. 71)",
                    "H3": "Algorithm governance resilience",
                },
                "H1_config": {
                    "algorithms": H1_ALGORITHMS,
                    "datasets": H1_DATASETS,
                    "conditions": [c["name"] for c in H1_CONDITIONS],
                    "seeds": SEEDS_5,
                    "count": 120,
                },
                "H2_config": {
                    "patterns": {n: r for n, r in H2_PATTERNS},
                    "hospitals": HD_HOSPITALS,
                    "seeds": SEEDS_10,
                    "count": 50,
                },
                "H3_config": {
                    "algorithms": H3_ALGORITHMS,
                    "purposes": H3_PURPOSES,
                    "seeds": SEEDS_10,
                    "count": 120,
                },
                "start_time": datetime.now().isoformat(),
                "last_save": None,
            },
        }

    # Signal handler
    _interrupted = [False]

    def _signal_handler(signum, frame):
        if _interrupted[0]:
            sys.exit(1)
        _interrupted[0] = True
        done = len(checkpoint_data.get("completed", {}))
        log("\nINTERRUPT -- saving checkpoint ({}/{})...".format(
            done, total_exps))
        save_checkpoint(checkpoint_data)
        log("Checkpoint saved. Resume: "
            "python -m benchmarks.run_governance_hypotheses")
        sys.exit(0)

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # Header
    log("\n" + "=" * 72)
    log("  FL-EHDS Governance Hypotheses (H1+H2+H3)")
    log("  {} experiments = 120 compound + 50 fairness + 120 resilience".format(
        total_exps))
    log("=" * 72)
    log("  Device:     {}".format(_detect_device()))
    log("  H1 seeds:   {} (5 seeds)".format(SEEDS_5))
    log("  H2/H3 seeds: {} (10 seeds)".format(SEEDS_10))
    log("  H1 algos:   {}".format(H1_ALGORITHMS))
    log("  H1 conds:   {}".format([c["name"] for c in H1_CONDITIONS]))
    log("  H2 patterns: {}".format([n for n, _ in H2_PATTERNS]))
    log("  H3 purposes: {}".format(H3_PURPOSES))
    log("  HD:  {} rounds, {} clients".format(
        HD_CONFIG["num_rounds"], HD_CONFIG["num_clients"]))
    log("  CV:  {} rounds, {} clients".format(
        CV_CONFIG["num_rounds"], CV_CONFIG["num_clients"]))
    log("  Output: {}".format(OUTPUT_DIR / CHECKPOINT_FILE))
    log("=" * 72)

    # ---- Run experiments ----
    global_start = time.time()
    completed = checkpoint_data.get("completed", {})
    done_count = len(completed)

    # Remove previous errors so they get re-run
    error_keys = [k for k in completed if "error" in completed[k]]
    if error_keys:
        log("  Removing {} errored entries for re-run".format(len(error_keys)))
        for k in error_keys:
            del completed[k]
        done_count = len(completed)
        save_checkpoint(checkpoint_data)

    for idx, exp in enumerate(experiments, 1):
        key = exp["key"]
        if key in completed:
            continue
        if _interrupted[0]:
            break

        log("[{}/{}] {} ...".format(done_count + 1, total_exps, key))

        try:
            start_t = time.time()

            if exp["type"] == "H1":
                result = run_compound(
                    exp["algorithm"], exp["dataset"],
                    exp["condition"], exp["seed"],
                )
            elif exp["type"] == "H2":
                result = run_nonuniform_optout(
                    exp["pattern_name"], exp["pattern_rates"], exp["seed"],
                )
            elif exp["type"] == "H3":
                result = run_algo_governance(
                    exp["algorithm"], exp["purpose"], exp["seed"],
                )
            else:
                continue

            elapsed_t = time.time() - start_t
            result["runtime_seconds"] = round(elapsed_t, 1)
            completed[key] = result
            done_count += 1

            # Progress log
            if exp["type"] == "H1":
                log("  -> {} {} {}: acc={:.1f}%, f1={:.3f} ({:.1f}s)".format(
                    exp["algorithm"], exp["dataset"],
                    exp["condition"]["name"],
                    result["final_accuracy"] * 100,
                    result["final_f1"], elapsed_t))
            elif exp["type"] == "H2":
                log("  -> {}: acc={:.1f}%, jain={:.3f}, dei={:.3f} ({:.1f}s)".format(
                    exp["pattern_name"],
                    result["global_accuracy"] * 100,
                    result["fairness"]["jain"],
                    result["fairness"]["dei_ratio"], elapsed_t))
            elif exp["type"] == "H3":
                log("  -> {} {}: {}/{} feat, acc={:.1f}%, f1={:.3f} ({:.1f}s)".format(
                    exp["algorithm"], exp["purpose"][:12],
                    result["kept_features"], result["original_features"],
                    result["final_accuracy"] * 100,
                    result["final_f1"], elapsed_t))

            # Atomic save after each experiment
            save_checkpoint(checkpoint_data)

        except Exception as e:
            log("  ERROR: {}".format(e))
            traceback.print_exc()
            completed[key] = {
                "key": key, "error": str(e),
                "traceback": traceback.format_exc(),
            }
            save_checkpoint(checkpoint_data)
            _cleanup_gpu()

    # ---- Finalize ----
    checkpoint_data["metadata"]["end_time"] = datetime.now().isoformat()
    checkpoint_data["metadata"]["total_elapsed"] = time.time() - global_start
    save_checkpoint(checkpoint_data)

    elapsed = time.time() - global_start

    # ======================================================================
    # Summary tables
    # ======================================================================
    log("\n" + "=" * 72)
    log("  COMPLETED: {}/{} ({:.0f}s = {:.1f} min)".format(
        done_count, total_exps, elapsed, elapsed / 60))
    log("=" * 72)

    _print_h1_summary(completed)
    _print_h2_summary(completed)
    _print_h3_summary(completed)
    _print_statistical_tests(completed)
    _print_key_findings(completed)

    log("\n  Done!")

    if _log_file:
        _log_file.close()


# ======================================================================
# H1 Summary: Compound governance cost
# ======================================================================

def _print_h1_summary(completed):
    log("\n" + "-" * 72)
    log("  [H1] COMPOUND EHDS COMPLIANCE STRESS TEST")
    log("-" * 72)
    log("  Layers: minimization(public_health) + 10% opt-out + DP")
    log("")

    for ds in H1_DATASETS:
        log("  --- {} ---".format("Heart Disease" if ds == "HD" else "Cardiovascular"))
        log("  {:<8s} | {:<14s} | {:>14s} | {:>10s} | {:>10s} | {:>10s}".format(
            "Algo", "Condition", "Acc% (m +/- s)", "F1", "AUC", "Delta"))
        log("  " + "-" * 75)

        # Get baselines for delta computation
        baselines = {}
        for algo in H1_ALGORITHMS:
            accs = []
            for seed in SEEDS_5:
                k = "H1_{}_{}_{}_s{}".format(algo, ds, "no_governance", seed)
                if k in completed and "error" not in completed[k]:
                    accs.append(completed[k]["final_accuracy"] * 100)
            if accs:
                baselines[algo] = np.mean(accs)

        for algo in H1_ALGORITHMS:
            for cond in H1_CONDITIONS:
                accs, f1s, aucs = [], [], []
                for seed in SEEDS_5:
                    k = "H1_{}_{}_{}_s{}".format(algo, ds, cond["name"], seed)
                    if k in completed and "error" not in completed[k]:
                        r = completed[k]
                        accs.append(r["final_accuracy"] * 100)
                        f1s.append(r["final_f1"])
                        aucs.append(r["final_auc"])
                if accs:
                    m_acc = np.mean(accs)
                    if cond["name"] == "no_governance":
                        delta = "ref"
                    elif algo in baselines:
                        delta = "{:+.1f}pp".format(m_acc - baselines[algo])
                    else:
                        delta = "n/a"
                    log("  {:<8s} | {:<14s} | {:>5.1f} +/- {:<4.1f}  | {:>5.3f}    | {:>5.3f}    | {:>10s}".format(
                        algo, cond["name"],
                        m_acc, np.std(accs),
                        np.mean(f1s), np.mean(aucs), delta))
            log("  " + "-" * 75)


# ======================================================================
# H2 Summary: Non-uniform opt-out fairness
# ======================================================================

def _print_h2_summary(completed):
    log("\n" + "-" * 72)
    log("  [H2] NON-UNIFORM OPT-OUT FAIRNESS (Heart Disease, FedAvg)")
    log("-" * 72)
    log("  Hospitals: {}".format(HD_HOSPITALS))
    log("")

    log("  {:<12s} | {:>5s} {:>5s} {:>5s} {:>5s} | {:>10s} | {:>6s} | {:>6s} | {:>7s}".format(
        "Pattern", "CL%", "HU%", "CH%", "VA%",
        "Global Acc", "Jain", "DEI", "Range"))
    log("  " + "-" * 85)

    for pattern_name, pattern_rates in H2_PATTERNS:
        g_accs = []
        all_jain, all_dei, all_range = [], [], []
        per_hosp = {0: [], 1: [], 2: [], 3: []}

        for seed in SEEDS_10:
            k = "H2_{}_s{}".format(pattern_name, seed)
            if k in completed and "error" not in completed[k]:
                r = completed[k]
                g_accs.append(r["global_accuracy"] * 100)
                all_jain.append(r["fairness"]["jain"])
                all_dei.append(r["fairness"]["dei_ratio"])
                all_range.append(r["fairness"]["range"] * 100)
                for cid in range(4):
                    cid_key = str(cid) if str(cid) in r["per_client_accuracy"] else cid
                    if cid_key in r["per_client_accuracy"]:
                        per_hosp[cid].append(
                            r["per_client_accuracy"][cid_key] * 100)

        if g_accs:
            hosp_strs = []
            for cid in range(4):
                if per_hosp[cid]:
                    hosp_strs.append("{:5.1f}".format(np.mean(per_hosp[cid])))
                else:
                    hosp_strs.append("  n/a")

            log("  {:<12s} | {} {} {} {} | {:>5.1f}+/-{:<3.1f} | {:>.3f} | {:>.3f} | {:>5.1f}pp".format(
                pattern_name,
                hosp_strs[0], hosp_strs[1], hosp_strs[2], hosp_strs[3],
                np.mean(g_accs), np.std(g_accs),
                np.mean(all_jain), np.mean(all_dei),
                np.mean(all_range)))


# ======================================================================
# H3 Summary: Algorithm governance resilience
# ======================================================================

def _print_h3_summary(completed):
    log("\n" + "-" * 72)
    log("  [H3] ALGORITHM GOVERNANCE RESILIENCE (Heart Disease)")
    log("-" * 72)
    log("")

    log("  {:<8s} | {:<28s} | {:>4s} | {:>14s} | {:>10s} | {:>10s}".format(
        "Algo", "Purpose", "Feat", "Acc% (m +/- s)", "F1", "Resilience"))
    log("  " + "-" * 85)

    # Compute baselines (scientific_research)
    baselines = {}
    for algo in H3_ALGORITHMS:
        accs = []
        for seed in SEEDS_10:
            k = "H3_{}_{}_s{}".format(algo, "scientific_research", seed)
            if k in completed and "error" not in completed[k]:
                accs.append(completed[k]["final_accuracy"] * 100)
        if accs:
            baselines[algo] = np.mean(accs)

    for algo in H3_ALGORITHMS:
        for purpose in H3_PURPOSES:
            accs, f1s, feats = [], [], []
            for seed in SEEDS_10:
                k = "H3_{}_{}_s{}".format(algo, purpose, seed)
                if k in completed and "error" not in completed[k]:
                    r = completed[k]
                    accs.append(r["final_accuracy"] * 100)
                    f1s.append(r["final_f1"])
                    feats.append(r["kept_features"])
            if accs:
                m_acc = np.mean(accs)
                m_feat = int(np.mean(feats))
                if algo in baselines and baselines[algo] > 0:
                    resilience = m_acc / baselines[algo]
                else:
                    resilience = 1.0
                log("  {:<8s} | {:<28s} | {:>2d}/13 | {:>5.1f} +/- {:<4.1f}  | {:>5.3f}    | {:>8.1f}%".format(
                    algo, purpose, m_feat,
                    m_acc, np.std(accs),
                    np.mean(f1s),
                    resilience * 100))
        log("  " + "-" * 85)


# ======================================================================
# Statistical tests
# ======================================================================

def _print_statistical_tests(completed):
    from scipy import stats as sp_stats

    log("\n" + "-" * 72)
    log("  STATISTICAL SIGNIFICANCE TESTS")
    log("-" * 72)

    # ---- H1: paired t-test governance vs no_governance ----
    log("\n  [H1] Paired t-tests: each condition vs no_governance (n=5 seeds)")
    for ds in H1_DATASETS:
        log("  -- {} --".format(ds))
        for algo in H1_ALGORITHMS:
            base_accs = {}
            for seed in SEEDS_5:
                k = "H1_{}_{}_{}_s{}".format(algo, ds, "no_governance", seed)
                if k in completed and "error" not in completed[k]:
                    base_accs[seed] = completed[k]["final_accuracy"] * 100

            for cond in H1_CONDITIONS:
                if cond["name"] == "no_governance":
                    continue
                cond_accs = {}
                for seed in SEEDS_5:
                    k = "H1_{}_{}_{}_s{}".format(algo, ds, cond["name"], seed)
                    if k in completed and "error" not in completed[k]:
                        cond_accs[seed] = completed[k]["final_accuracy"] * 100

                common = sorted(set(base_accs) & set(cond_accs))
                if len(common) < 3:
                    continue

                b = np.array([base_accs[s] for s in common])
                c = np.array([cond_accs[s] for s in common])
                diffs = c - b
                t, p = sp_stats.ttest_rel(c, b)
                d = diffs.mean() / diffs.std(ddof=1) if diffs.std(ddof=1) > 0 else 0
                sig = "*" if p < 0.05 else ""
                log("    {} {:<14s}: delta={:+5.1f}pp, t={:+.2f}, p={:.4f}{}, d={:.2f}".format(
                    algo, cond["name"], diffs.mean(), t, p, sig, d))

    # ---- H2: paired t-test non-uniform vs uniform_0 ----
    log("\n  [H2] Paired t-tests: each pattern vs uniform_0 (n=10 seeds)")
    base_accs_h2 = {}
    base_jain_h2 = {}
    for seed in SEEDS_10:
        k = "H2_uniform_0_s{}".format(seed)
        if k in completed and "error" not in completed[k]:
            base_accs_h2[seed] = completed[k]["global_accuracy"] * 100
            base_jain_h2[seed] = completed[k]["fairness"]["jain"]

    for pattern_name, _ in H2_PATTERNS:
        if pattern_name == "uniform_0":
            continue
        p_accs, p_jain = {}, {}
        for seed in SEEDS_10:
            k = "H2_{}_s{}".format(pattern_name, seed)
            if k in completed and "error" not in completed[k]:
                p_accs[seed] = completed[k]["global_accuracy"] * 100
                p_jain[seed] = completed[k]["fairness"]["jain"]

        common = sorted(set(base_accs_h2) & set(p_accs))
        if len(common) < 3:
            continue

        b_a = np.array([base_accs_h2[s] for s in common])
        p_a = np.array([p_accs[s] for s in common])
        b_j = np.array([base_jain_h2[s] for s in common])
        p_j = np.array([p_jain[s] for s in common])

        d_acc = p_a - b_a
        t_acc, pv_acc = sp_stats.ttest_rel(p_a, b_a)
        d_jain = p_j - b_j
        t_jain, pv_jain = sp_stats.ttest_rel(p_j, b_j)

        sig_a = "*" if pv_acc < 0.05 else ""
        sig_j = "*" if pv_jain < 0.05 else ""
        log("    {:<12s}: Acc delta={:+5.1f}pp (p={:.4f}{}), "
            "Jain delta={:+.4f} (p={:.4f}{})".format(
                pattern_name, d_acc.mean(), pv_acc, sig_a,
                d_jain.mean(), pv_jain, sig_j))

    # ---- H3: paired t-test each purpose vs scientific_research ----
    log("\n  [H3] Paired t-tests: each purpose vs scientific_research (n=10 seeds)")
    for algo in H3_ALGORITHMS:
        base_accs_h3 = {}
        for seed in SEEDS_10:
            k = "H3_{}_{}_s{}".format(algo, "scientific_research", seed)
            if k in completed and "error" not in completed[k]:
                base_accs_h3[seed] = completed[k]["final_accuracy"] * 100

        for purpose in H3_PURPOSES:
            if purpose == "scientific_research":
                continue
            purp_accs = {}
            for seed in SEEDS_10:
                k = "H3_{}_{}_s{}".format(algo, purpose, seed)
                if k in completed and "error" not in completed[k]:
                    purp_accs[seed] = completed[k]["final_accuracy"] * 100

            common = sorted(set(base_accs_h3) & set(purp_accs))
            if len(common) < 3:
                continue

            b = np.array([base_accs_h3[s] for s in common])
            c = np.array([purp_accs[s] for s in common])
            diffs = c - b
            t, p = sp_stats.ttest_rel(c, b)
            d = diffs.mean() / diffs.std(ddof=1) if diffs.std(ddof=1) > 0 else 0
            sig = "*" if p < 0.05 else ""
            log("    {} {:<28s}: delta={:+5.1f}pp, t={:+.2f}, p={:.4f}{}, d={:.2f}".format(
                algo, purpose, diffs.mean(), t, p, sig, d))


# ======================================================================
# Key findings summary
# ======================================================================

def _print_key_findings(completed):
    log("\n" + "=" * 72)
    log("  KEY SCIENTIFIC FINDINGS")
    log("=" * 72)

    # H1: Cumulative compliance cost
    log("\n  [H1] Cumulative EHDS compliance cost (gov_eps10 vs no_governance):")
    for ds in H1_DATASETS:
        ds_label = "HD" if ds == "HD" else "CV"
        for algo in H1_ALGORITHMS:
            base_accs, gov_accs = [], []
            for seed in SEEDS_5:
                k_base = "H1_{}_{}_{}_s{}".format(algo, ds, "no_governance", seed)
                k_gov = "H1_{}_{}_{}_s{}".format(algo, ds, "gov_eps10", seed)
                if (k_base in completed and "error" not in completed[k_base] and
                        k_gov in completed and "error" not in completed[k_gov]):
                    base_accs.append(completed[k_base]["final_accuracy"] * 100)
                    gov_accs.append(completed[k_gov]["final_accuracy"] * 100)
            if base_accs:
                delta = np.mean(gov_accs) - np.mean(base_accs)
                log("    {} {}: {:.1f}% -> {:.1f}% (delta={:+.1f}pp)".format(
                    algo, ds_label,
                    np.mean(base_accs), np.mean(gov_accs), delta))

    # H2: Worst fairness degradation
    log("\n  [H2] Opt-out fairness (Jain index):")
    for pattern_name, _ in H2_PATTERNS:
        jains = []
        for seed in SEEDS_10:
            k = "H2_{}_s{}".format(pattern_name, seed)
            if k in completed and "error" not in completed[k]:
                jains.append(completed[k]["fairness"]["jain"])
        if jains:
            log("    {:<12s}: Jain={:.4f} +/- {:.4f}".format(
                pattern_name, np.mean(jains), np.std(jains)))

    # H3: Governance resilience ranking
    log("\n  [H3] Governance resilience (official_statistics, 2/13 features):")
    resiliences = []
    for algo in H3_ALGORITHMS:
        base_accs, os_accs = [], []
        for seed in SEEDS_10:
            k_base = "H3_{}_{}_s{}".format(algo, "scientific_research", seed)
            k_os = "H3_{}_{}_s{}".format(algo, "official_statistics", seed)
            if (k_base in completed and "error" not in completed[k_base] and
                    k_os in completed and "error" not in completed[k_os]):
                base_accs.append(completed[k_base]["final_accuracy"] * 100)
                os_accs.append(completed[k_os]["final_accuracy"] * 100)
        if base_accs:
            resil = np.mean(os_accs) / np.mean(base_accs) * 100
            resiliences.append((algo, resil, np.mean(base_accs), np.mean(os_accs)))
            log("    {}: {:.1f}% -> {:.1f}% (resilience={:.1f}%)".format(
                algo, np.mean(base_accs), np.mean(os_accs), resil))

    if resiliences:
        best = max(resiliences, key=lambda x: x[1])
        log("\n  => Most governance-resilient algorithm: {} ({:.1f}%)".format(
            best[0], best[1]))


if __name__ == "__main__":
    main()

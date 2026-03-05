#!/usr/bin/env python3
"""
FL-EHDS Experiment — Cross-Dataset Governance Validation (Cardiovascular).

Replicates all three governance hypotheses on the Cardiovascular dataset
(70,000 samples, 5 clients, Dirichlet alpha=0.5) to provide large-sample
validation independent of the Heart Disease small-sample regularization
effects.

  H1-CV) Compound EHDS Compliance Stress Test
      3 algorithms x 4 conditions x 10 seeds = 120 experiments

  H2-CV) Non-Uniform Opt-Out Fairness (5 clients)
      5 patterns x 10 seeds = 50 experiments

  H3-CV) Algorithm Governance Resilience
      3 algorithms x 4 purposes x 10 seeds = 120 experiments

Total: 290 experiments (~60 min on M3)

Usage:
    cd fl-ehds-framework
    python -m benchmarks.run_governance_hypotheses_cv [--fresh]

Output:
    benchmarks/paper_results_tabular/checkpoint_governance_hyp_cv.json

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
from typing import Any, Dict, List, Tuple

import numpy as np

FRAMEWORK_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(FRAMEWORK_DIR))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from terminal.fl_trainer import FederatedTrainer, HealthcareMLP, _detect_device

from core.models import DataCategory, OptOutRecord, PermitPurpose
from governance.data_minimization import DataMinimizer

# ======================================================================
# Constants
# ======================================================================

OUTPUT_DIR = FRAMEWORK_DIR / "benchmarks" / "paper_results_tabular"
CHECKPOINT_FILE = "checkpoint_governance_hyp_cv.json"
LOG_FILE = "experiment_governance_hyp_cv.log"

SEEDS_10 = [42, 123, 456, 789, 999, 1234, 2345, 3456, 4567, 5678]

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

CV_FEATURE_GROUPS = {
    "age": "demographics", "gender": "demographics",
    "height": "demographics", "weight": "demographics",
    "ap_hi": "vitals", "ap_lo": "vitals",
    "cholesterol": "measurements", "gluc": "measurements",
    "smoke": "conditions", "alco": "conditions", "active": "conditions",
}

CV_CLIENT_NAMES = ["Client_0", "Client_1", "Client_2", "Client_3", "Client_4"]

# ---- H1 ----
H1_ALGORITHMS = ["FedAvg", "Ditto", "HPFL"]
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

# ---- H2 (5 clients) ----
H2_PATTERNS = [
    ("uniform_0",   [0.00, 0.00, 0.00, 0.00, 0.00]),
    ("uniform_13",  [0.13, 0.13, 0.13, 0.13, 0.13]),
    ("half_30",     [0.00, 0.00, 0.00, 0.30, 0.30]),
    ("single_50",   [0.00, 0.00, 0.00, 0.00, 0.50]),
    ("gradient",    [0.00, 0.05, 0.10, 0.20, 0.30]),
]

# ---- H3 ----
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
# Checkpoint
# ======================================================================

def save_checkpoint(data):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / CHECKPOINT_FILE
    bak = OUTPUT_DIR / (CHECKPOINT_FILE + ".bak")
    data["metadata"]["last_save"] = datetime.now().isoformat()
    fd, tmp = tempfile.mkstemp(dir=str(OUTPUT_DIR), prefix=".ghcv_", suffix=".tmp")
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
# Evaluation
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
    if not client_accs or max(client_accs) == 0:
        return {"jain": 0, "dei_ratio": 0, "range": 0, "std": 0, "cv": 0}
    n = len(client_accs)
    sum_x = sum(client_accs)
    sum_x2 = sum(a ** 2 for a in client_accs)
    jain = (sum_x ** 2) / (n * sum_x2) if sum_x2 > 0 else 0
    dei_ratio = min(client_accs) / max(client_accs) if max(client_accs) > 0 else 0
    return {
        "jain": round(float(jain), 4),
        "dei_ratio": round(float(dei_ratio), 4),
        "range": round(float(max(client_accs) - min(client_accs)), 4),
        "std": round(float(np.std(client_accs)), 4),
        "cv": round(float(np.std(client_accs) / np.mean(client_accs)), 4)
        if np.mean(client_accs) > 0 else 0,
    }


# ======================================================================
# Shared helpers
# ======================================================================

def _apply_minimization(client_data, client_test, purpose, feature_names):
    original_groups = dict(DataMinimizer.FEATURE_GROUPS)
    for fname, group in CV_FEATURE_GROUPS.items():
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
    rng = np.random.RandomState(seed)
    all_pids = []
    client_pids = {}
    for cid in sorted(client_data.keys()):
        n = len(client_data[cid][1])
        pids = ["P-{}-{:05d}".format(cid, i) for i in range(n)]
        client_pids[cid] = pids
        all_pids.extend(pids)
    n_optout = int(len(all_pids) * optout_rate)
    if n_optout == 0:
        return dict(client_data), 0, len(all_pids)
    optout_set = set(rng.choice(all_pids, size=n_optout, replace=False))
    filtered, excluded = {}, 0
    for cid in sorted(client_data.keys()):
        X, y = client_data[cid]
        mask = np.array([p not in optout_set for p in client_pids[cid]])
        excluded += int((~mask).sum())
        filtered[cid] = (X[mask], y[mask])
    return filtered, excluded, len(all_pids)


def _apply_nonuniform_optout(client_data, pattern_rates, seed):
    rng = np.random.RandomState(seed)
    filtered, excluded_per = {}, {}
    for cid in sorted(client_data.keys()):
        X, y = client_data[cid]
        rate = pattern_rates[cid] if cid < len(pattern_rates) else 0.0
        n = len(y)
        n_out = int(n * rate)
        if n_out == 0:
            filtered[cid] = (X.copy(), y.copy())
            excluded_per[cid] = 0
        elif n_out >= n:
            filtered[cid] = (X[:2], y[:2])
            excluded_per[cid] = n - 2
        else:
            idx = np.arange(n)
            rng.shuffle(idx)
            keep = sorted(idx[n_out:])
            filtered[cid] = (X[keep], y[keep])
            excluded_per[cid] = n_out
    return filtered, excluded_per


# ======================================================================
# H1-CV: Compound Governance
# ======================================================================

def run_compound(algorithm, condition, seed):
    cfg = CV_CONFIG
    client_data, client_test, meta, feature_names = load_cv(seed)

    input_dim = cfg["input_dim"]
    train_data = dict(client_data)
    test_data = dict(client_test)
    n_excluded = 0
    n_total = sum(len(train_data[c][1]) for c in train_data)
    n_kept = input_dim

    if condition["minimize"]:
        train_data, test_data, report = _apply_minimization(
            train_data, test_data, "public_health_surveillance", feature_names)
        n_kept = report["kept_features"]
        input_dim = n_kept

    if condition["optout_rate"] > 0:
        train_data, n_excluded, n_total = _apply_uniform_optout(
            train_data, condition["optout_rate"], seed)

    dp_on = condition["dp_enabled"]
    dp_eps = condition["dp_epsilon"]

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
        dp_enabled=dp_on,
        dp_epsilon=dp_eps if dp_on else 10.0,
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
        "condition": condition["name"],
        "seed": seed,
        "features_used": n_kept,
        "samples_excluded": n_excluded,
        "samples_total": n_total,
        "dp_enabled": dp_on,
        "dp_epsilon": dp_eps if dp_on else None,
        "final_accuracy": final.get("accuracy", 0),
        "final_f1": final.get("f1", 0),
        "final_auc": final.get("auc", 0),
        "final_loss": final.get("loss", 0),
        "history": history,
    }


# ======================================================================
# H2-CV: Non-Uniform Opt-Out Fairness
# ======================================================================

def run_nonuniform_optout(pattern_name, pattern_rates, seed):
    cfg = CV_CONFIG
    client_data, client_test, meta, feature_names = load_cv(seed)

    filtered_data, excluded_per = _apply_nonuniform_optout(
        client_data, pattern_rates, seed)

    total_excl = sum(excluded_per.values())
    total_train = sum(len(filtered_data[c][1]) for c in filtered_data)
    train_sizes = {cid: len(filtered_data[cid][1])
                   for cid in sorted(filtered_data.keys())}

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

    device = _detect_device()
    per_client = {}
    for cid in sorted(client_test.keys()):
        X_t, y_t = client_test[cid]
        if len(y_t) == 0:
            continue
        per_client[cid] = _evaluate_model(
            trainer.global_model, X_t, y_t,
            cfg["batch_size"], device, cfg["num_classes"])

    client_accs = [per_client[c]["accuracy"]
                   for c in sorted(per_client.keys())]
    fairness = _compute_fairness(client_accs)

    final = history[-1] if history else {}
    return {
        "hypothesis": "H2",
        "pattern_name": pattern_name,
        "pattern_rates": pattern_rates,
        "seed": seed,
        "excluded_per_client": excluded_per,
        "train_size_per_client": train_sizes,
        "total_excluded": total_excl,
        "total_train_remaining": total_train,
        "global_accuracy": final.get("accuracy", 0),
        "global_f1": final.get("f1", 0),
        "global_auc": final.get("auc", 0),
        "per_client_accuracy": {
            c: per_client[c]["accuracy"] for c in sorted(per_client.keys())},
        "per_client_f1": {
            c: per_client[c]["f1"] for c in sorted(per_client.keys())},
        "fairness": fairness,
        "history": history,
    }


# ======================================================================
# H3-CV: Algorithm Governance Resilience
# ======================================================================

def run_algo_governance(algorithm, purpose, seed):
    cfg = CV_CONFIG
    client_data, client_test, meta, feature_names = load_cv(seed)

    filtered_train, filtered_test, report = _apply_minimization(
        client_data, client_test, purpose, feature_names)
    n_kept = report["kept_features"]

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
        "original_features": report["original_features"],
        "kept_features": n_kept,
        "kept_feature_names": report["kept_feature_names"],
        "reduction_pct": report["reduction_pct"],
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
        description="FL-EHDS Cross-Dataset Governance Validation (CV, 70K)")
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()

    global _log_file
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _log_file = open(OUTPUT_DIR / LOG_FILE, "a")

    # ---- Build experiment list ----
    experiments = []

    for algo in H1_ALGORITHMS:
        for cond in H1_CONDITIONS:
            for seed in SEEDS_10:
                experiments.append({
                    "key": "H1_{}_CV_{}_s{}".format(algo, cond["name"], seed),
                    "type": "H1", "algorithm": algo,
                    "condition": cond, "seed": seed,
                })

    for pname, prates in H2_PATTERNS:
        for seed in SEEDS_10:
            experiments.append({
                "key": "H2_CV_{}_s{}".format(pname, seed),
                "type": "H2", "pattern_name": pname,
                "pattern_rates": prates, "seed": seed,
            })

    for algo in H3_ALGORITHMS:
        for purpose in H3_PURPOSES:
            for seed in SEEDS_10:
                experiments.append({
                    "key": "H3_{}_CV_{}_s{}".format(algo, purpose, seed),
                    "type": "H3", "algorithm": algo,
                    "purpose": purpose, "seed": seed,
                })

    total_exps = len(experiments)

    if args.fresh:
        for f in [CHECKPOINT_FILE, CHECKPOINT_FILE + ".bak"]:
            p = OUTPUT_DIR / f
            if p.exists():
                p.unlink()
        log("Deleted existing checkpoint")

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
                "dataset": "Cardiovascular (70K samples, 5 clients)",
                "total_experiments": total_exps,
                "seeds": SEEDS_10,
                "H1_count": 120,
                "H2_count": 50,
                "H3_count": 120,
                "start_time": datetime.now().isoformat(),
                "last_save": None,
            },
        }

    _interrupted = [False]

    def _signal_handler(signum, frame):
        if _interrupted[0]:
            sys.exit(1)
        _interrupted[0] = True
        done = len(checkpoint_data.get("completed", {}))
        log("\nINTERRUPT -- saving ({}/{})...".format(done, total_exps))
        save_checkpoint(checkpoint_data)
        log("Checkpoint saved. Resume: "
            "python -m benchmarks.run_governance_hypotheses_cv")
        sys.exit(0)

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    log("\n" + "=" * 72)
    log("  FL-EHDS Cross-Dataset Governance Validation (CV, 70K)")
    log("  {} experiments = 120 compound + 50 fairness + 120 resilience".format(
        total_exps))
    log("=" * 72)
    log("  Device:  {}".format(_detect_device()))
    log("  Seeds:   {} (10 seeds)".format(SEEDS_10))
    log("  Config:  {} rounds, {} clients, 70K samples, Dirichlet a=0.5".format(
        CV_CONFIG["num_rounds"], CV_CONFIG["num_clients"]))
    log("  H1 conds: {}".format([c["name"] for c in H1_CONDITIONS]))
    log("  H2 patt:  {}".format([n for n, _ in H2_PATTERNS]))
    log("  H3 purp:  {}".format(H3_PURPOSES))
    log("  Output:  {}".format(OUTPUT_DIR / CHECKPOINT_FILE))
    log("=" * 72)

    global_start = time.time()
    completed = checkpoint_data.get("completed", {})
    done_count = len(completed)

    # Remove previous errors for re-run
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
            t0 = time.time()

            if exp["type"] == "H1":
                result = run_compound(
                    exp["algorithm"], exp["condition"], exp["seed"])
            elif exp["type"] == "H2":
                result = run_nonuniform_optout(
                    exp["pattern_name"], exp["pattern_rates"], exp["seed"])
            elif exp["type"] == "H3":
                result = run_algo_governance(
                    exp["algorithm"], exp["purpose"], exp["seed"])
            else:
                continue

            dt = time.time() - t0
            result["runtime_seconds"] = round(dt, 1)
            completed[key] = result
            done_count += 1

            if exp["type"] == "H1":
                log("  -> {} {}: acc={:.1f}%, f1={:.3f} ({:.1f}s)".format(
                    exp["algorithm"], exp["condition"]["name"],
                    result["final_accuracy"] * 100,
                    result["final_f1"], dt))
            elif exp["type"] == "H2":
                log("  -> {}: acc={:.1f}%, jain={:.3f}, dei={:.3f} ({:.1f}s)".format(
                    exp["pattern_name"],
                    result["global_accuracy"] * 100,
                    result["fairness"]["jain"],
                    result["fairness"]["dei_ratio"], dt))
            elif exp["type"] == "H3":
                log("  -> {} {}: {}/{} feat, acc={:.1f}% ({:.1f}s)".format(
                    exp["algorithm"], exp["purpose"][:12],
                    result["kept_features"], result["original_features"],
                    result["final_accuracy"] * 100, dt))

            save_checkpoint(checkpoint_data)

        except Exception as e:
            log("  ERROR: {}".format(e))
            traceback.print_exc()
            completed[key] = {"key": key, "error": str(e),
                              "traceback": traceback.format_exc()}
            save_checkpoint(checkpoint_data)
            _cleanup_gpu()

    checkpoint_data["metadata"]["end_time"] = datetime.now().isoformat()
    checkpoint_data["metadata"]["total_elapsed"] = time.time() - global_start
    save_checkpoint(checkpoint_data)

    elapsed = time.time() - global_start
    log("\n" + "=" * 72)
    log("  COMPLETED: {}/{} ({:.0f}s = {:.1f} min)".format(
        done_count, total_exps, elapsed, elapsed / 60))
    log("=" * 72)

    _print_h1_summary(completed)
    _print_h2_summary(completed)
    _print_h3_summary(completed)
    _print_statistical_tests(completed)
    _print_cross_dataset_comparison(completed)

    log("\n  Done!")
    if _log_file:
        _log_file.close()


# ======================================================================
# H1 Summary
# ======================================================================

def _print_h1_summary(completed):
    log("\n" + "-" * 72)
    log("  [H1-CV] COMPOUND EHDS COMPLIANCE (Cardiovascular, 70K)")
    log("-" * 72)

    log("  {:<8s} | {:<14s} | {:>14s} | {:>10s} | {:>10s} | {:>10s}".format(
        "Algo", "Condition", "Acc% (m+/-s)", "F1", "AUC", "Delta"))
    log("  " + "-" * 75)

    baselines = {}
    for algo in H1_ALGORITHMS:
        accs = []
        for seed in SEEDS_10:
            k = "H1_{}_CV_{}_s{}".format(algo, "no_governance", seed)
            if k in completed and "error" not in completed[k]:
                accs.append(completed[k]["final_accuracy"] * 100)
        if accs:
            baselines[algo] = np.mean(accs)

    for algo in H1_ALGORITHMS:
        for cond in H1_CONDITIONS:
            accs, f1s, aucs = [], [], []
            for seed in SEEDS_10:
                k = "H1_{}_CV_{}_s{}".format(algo, cond["name"], seed)
                if k in completed and "error" not in completed[k]:
                    r = completed[k]
                    accs.append(r["final_accuracy"] * 100)
                    f1s.append(r["final_f1"])
                    aucs.append(r["final_auc"])
            if accs:
                m = np.mean(accs)
                delta = "ref" if cond["name"] == "no_governance" else (
                    "{:+.1f}pp".format(m - baselines[algo])
                    if algo in baselines else "n/a")
                log("  {:<8s} | {:<14s} | {:>5.1f} +/- {:<4.1f}  | {:>5.3f}    | {:>5.3f}    | {:>10s}".format(
                    algo, cond["name"],
                    m, np.std(accs),
                    np.mean(f1s), np.mean(aucs), delta))
        log("  " + "-" * 75)


# ======================================================================
# H2 Summary
# ======================================================================

def _print_h2_summary(completed):
    log("\n" + "-" * 72)
    log("  [H2-CV] NON-UNIFORM OPT-OUT FAIRNESS (CV, 5 clients)")
    log("-" * 72)

    log("  {:<12s} | {:>10s} | {:>5s} {:>5s} {:>5s} {:>5s} {:>5s} | {:>6s} | {:>6s} | {:>7s}".format(
        "Pattern", "Global Acc", "C0%", "C1%", "C2%", "C3%", "C4%",
        "Jain", "DEI", "Range"))
    log("  " + "-" * 95)

    for pname, _ in H2_PATTERNS:
        g_accs, jains, deis, rngs = [], [], [], []
        per_c = {i: [] for i in range(5)}
        for seed in SEEDS_10:
            k = "H2_CV_{}_s{}".format(pname, seed)
            if k in completed and "error" not in completed[k]:
                r = completed[k]
                g_accs.append(r["global_accuracy"] * 100)
                jains.append(r["fairness"]["jain"])
                deis.append(r["fairness"]["dei_ratio"])
                rngs.append(r["fairness"]["range"] * 100)
                for cid in range(5):
                    ck = str(cid) if str(cid) in r["per_client_accuracy"] else cid
                    if ck in r["per_client_accuracy"]:
                        per_c[cid].append(r["per_client_accuracy"][ck] * 100)
        if g_accs:
            cs = ["{:5.1f}".format(np.mean(per_c[i])) if per_c[i] else "  n/a"
                  for i in range(5)]
            log("  {:<12s} | {:>5.1f}+/-{:<3.1f} | {} {} {} {} {} | {:>.3f} | {:>.3f} | {:>5.1f}pp".format(
                pname,
                np.mean(g_accs), np.std(g_accs),
                cs[0], cs[1], cs[2], cs[3], cs[4],
                np.mean(jains), np.mean(deis), np.mean(rngs)))


# ======================================================================
# H3 Summary
# ======================================================================

def _print_h3_summary(completed):
    log("\n" + "-" * 72)
    log("  [H3-CV] ALGORITHM GOVERNANCE RESILIENCE (CV, 70K)")
    log("-" * 72)

    log("  {:<8s} | {:<28s} | {:>5s} | {:>14s} | {:>10s} | {:>10s}".format(
        "Algo", "Purpose", "Feat", "Acc% (m+/-s)", "F1", "Resil%"))
    log("  " + "-" * 85)

    baselines = {}
    for algo in H3_ALGORITHMS:
        accs = []
        for seed in SEEDS_10:
            k = "H3_{}_CV_{}_s{}".format(algo, "scientific_research", seed)
            if k in completed and "error" not in completed[k]:
                accs.append(completed[k]["final_accuracy"] * 100)
        if accs:
            baselines[algo] = np.mean(accs)

    for algo in H3_ALGORITHMS:
        for purpose in H3_PURPOSES:
            accs, f1s, feats = [], [], []
            for seed in SEEDS_10:
                k = "H3_{}_CV_{}_s{}".format(algo, purpose, seed)
                if k in completed and "error" not in completed[k]:
                    r = completed[k]
                    accs.append(r["final_accuracy"] * 100)
                    f1s.append(r["final_f1"])
                    feats.append(r["kept_features"])
            if accs:
                m = np.mean(accs)
                mf = int(np.mean(feats))
                resil = (m / baselines[algo] * 100) if algo in baselines and baselines[algo] > 0 else 100
                log("  {:<8s} | {:<28s} | {:>2d}/11 | {:>5.1f} +/- {:<4.1f}  | {:>5.3f}    | {:>8.1f}%".format(
                    algo, purpose, mf,
                    m, np.std(accs), np.mean(f1s), resil))
        log("  " + "-" * 85)


# ======================================================================
# Statistical tests
# ======================================================================

def _print_statistical_tests(completed):
    from scipy import stats as sp_stats

    log("\n" + "-" * 72)
    log("  STATISTICAL SIGNIFICANCE TESTS (CV)")
    log("-" * 72)

    # H1
    log("\n  [H1] Paired t-tests: condition vs no_governance (n=10 seeds)")
    for algo in H1_ALGORITHMS:
        base = {}
        for seed in SEEDS_10:
            k = "H1_{}_CV_{}_s{}".format(algo, "no_governance", seed)
            if k in completed and "error" not in completed[k]:
                base[seed] = completed[k]["final_accuracy"] * 100
        for cond in H1_CONDITIONS:
            if cond["name"] == "no_governance":
                continue
            cvals = {}
            for seed in SEEDS_10:
                k = "H1_{}_CV_{}_s{}".format(algo, cond["name"], seed)
                if k in completed and "error" not in completed[k]:
                    cvals[seed] = completed[k]["final_accuracy"] * 100
            common = sorted(set(base) & set(cvals))
            if len(common) < 3:
                continue
            b = np.array([base[s] for s in common])
            c = np.array([cvals[s] for s in common])
            d = c - b
            t, p = sp_stats.ttest_rel(c, b)
            cd = d.mean() / d.std(ddof=1) if d.std(ddof=1) > 0 else 0
            sig = "*" if p < 0.05 else ""
            log("    {} {:<14s}: delta={:+5.1f}pp, t={:+6.2f}, p={:.4f}{}, d={:.2f}".format(
                algo, cond["name"], d.mean(), t, p, sig, cd))

    # H2
    log("\n  [H2] Paired t-tests: pattern vs uniform_0 (n=10 seeds)")
    base_a, base_j = {}, {}
    for seed in SEEDS_10:
        k = "H2_CV_uniform_0_s{}".format(seed)
        if k in completed and "error" not in completed[k]:
            base_a[seed] = completed[k]["global_accuracy"] * 100
            base_j[seed] = completed[k]["fairness"]["jain"]
    for pname, _ in H2_PATTERNS:
        if pname == "uniform_0":
            continue
        pa, pj = {}, {}
        for seed in SEEDS_10:
            k = "H2_CV_{}_s{}".format(pname, seed)
            if k in completed and "error" not in completed[k]:
                pa[seed] = completed[k]["global_accuracy"] * 100
                pj[seed] = completed[k]["fairness"]["jain"]
        common = sorted(set(base_a) & set(pa))
        if len(common) < 3:
            continue
        ba = np.array([base_a[s] for s in common])
        ca = np.array([pa[s] for s in common])
        bj = np.array([base_j[s] for s in common])
        cj = np.array([pj[s] for s in common])
        da = ca - ba
        ta, pva = sp_stats.ttest_rel(ca, ba)
        dj = cj - bj
        tj, pvj = sp_stats.ttest_rel(cj, bj)
        sa = "*" if pva < 0.05 else ""
        sj = "*" if pvj < 0.05 else ""
        log("    {:<12s}: Acc={:+5.1f}pp (p={:.4f}{}), Jain={:+.4f} (p={:.4f}{})".format(
            pname, da.mean(), pva, sa, dj.mean(), pvj, sj))

    # H3
    log("\n  [H3] Paired t-tests: purpose vs scientific_research (n=10 seeds)")
    for algo in H3_ALGORITHMS:
        base = {}
        for seed in SEEDS_10:
            k = "H3_{}_CV_{}_s{}".format(algo, "scientific_research", seed)
            if k in completed and "error" not in completed[k]:
                base[seed] = completed[k]["final_accuracy"] * 100
        for purpose in H3_PURPOSES:
            if purpose == "scientific_research":
                continue
            pvals = {}
            for seed in SEEDS_10:
                k = "H3_{}_CV_{}_s{}".format(algo, purpose, seed)
                if k in completed and "error" not in completed[k]:
                    pvals[seed] = completed[k]["final_accuracy"] * 100
            common = sorted(set(base) & set(pvals))
            if len(common) < 3:
                continue
            b = np.array([base[s] for s in common])
            c = np.array([pvals[s] for s in common])
            d = c - b
            t, p = sp_stats.ttest_rel(c, b)
            cd = d.mean() / d.std(ddof=1) if d.std(ddof=1) > 0 else 0
            sig = "*" if p < 0.05 else ""
            log("    {} {:<28s}: delta={:+5.1f}pp, t={:+6.2f}, p={:.4f}{}, d={:.2f}".format(
                algo, purpose, d.mean(), t, p, sig, cd))


# ======================================================================
# Cross-dataset comparison (loads HD checkpoint if present)
# ======================================================================

def _print_cross_dataset_comparison(completed_cv):
    hd_path = OUTPUT_DIR / "checkpoint_governance_hyp.json"
    if not hd_path.exists():
        log("\n  [CROSS-DATASET] HD checkpoint not found — skipping comparison")
        return

    try:
        with open(hd_path) as f:
            hd_data = json.load(f)
        completed_hd = hd_data.get("completed", {})
    except Exception:
        return

    log("\n" + "=" * 72)
    log("  CROSS-DATASET COMPARISON (HD 920 vs CV 70K)")
    log("=" * 72)

    # H1 comparison
    log("\n  [H1] Compound governance cost (gov_eps10 vs no_governance):")
    log("  {:<8s} | {:>14s} {:>10s} | {:>14s} {:>10s}".format(
        "Algo", "HD Acc%", "HD Delta", "CV Acc%", "CV Delta"))
    log("  " + "-" * 65)

    for algo in H1_ALGORITHMS:
        # HD
        hd_base, hd_gov = [], []
        for seed in [42, 123, 456, 789, 999]:
            kb = "H1_{}_HD_{}_s{}".format(algo, "no_governance", seed)
            kg = "H1_{}_HD_{}_s{}".format(algo, "gov_eps10", seed)
            if (kb in completed_hd and "error" not in completed_hd[kb] and
                    kg in completed_hd and "error" not in completed_hd[kg]):
                hd_base.append(completed_hd[kb]["final_accuracy"] * 100)
                hd_gov.append(completed_hd[kg]["final_accuracy"] * 100)
        # CV
        cv_base, cv_gov = [], []
        for seed in SEEDS_10:
            kb = "H1_{}_CV_{}_s{}".format(algo, "no_governance", seed)
            kg = "H1_{}_CV_{}_s{}".format(algo, "gov_eps10", seed)
            if (kb in completed_cv and "error" not in completed_cv[kb] and
                    kg in completed_cv and "error" not in completed_cv[kg]):
                cv_base.append(completed_cv[kb]["final_accuracy"] * 100)
                cv_gov.append(completed_cv[kg]["final_accuracy"] * 100)

        hd_str = "{:.1f}->{:.1f}".format(np.mean(hd_base), np.mean(hd_gov)) if hd_base else "n/a"
        hd_delta = "{:+.1f}pp".format(np.mean(hd_gov) - np.mean(hd_base)) if hd_base else ""
        cv_str = "{:.1f}->{:.1f}".format(np.mean(cv_base), np.mean(cv_gov)) if cv_base else "n/a"
        cv_delta = "{:+.1f}pp".format(np.mean(cv_gov) - np.mean(cv_base)) if cv_base else ""

        log("  {:<8s} | {:>14s} {:>10s} | {:>14s} {:>10s}".format(
            algo, hd_str, hd_delta, cv_str, cv_delta))

    # H3 comparison: governance resilience at official_statistics
    log("\n  [H3] Governance resilience (official_statistics):")
    log("  {:<8s} | {:>14s} {:>10s} | {:>14s} {:>10s}".format(
        "Algo", "HD Resil%", "HD Delta", "CV Resil%", "CV Delta"))
    log("  " + "-" * 65)

    for algo in H3_ALGORITHMS:
        # HD
        hd_sr, hd_os = [], []
        for seed in SEEDS_10:
            kb = "H3_{}_{}_s{}".format(algo, "scientific_research", seed)
            ko = "H3_{}_{}_s{}".format(algo, "official_statistics", seed)
            if (kb in completed_hd and "error" not in completed_hd[kb] and
                    ko in completed_hd and "error" not in completed_hd[ko]):
                hd_sr.append(completed_hd[kb]["final_accuracy"] * 100)
                hd_os.append(completed_hd[ko]["final_accuracy"] * 100)
        # CV
        cv_sr, cv_os = [], []
        for seed in SEEDS_10:
            kb = "H3_{}_CV_{}_s{}".format(algo, "scientific_research", seed)
            ko = "H3_{}_CV_{}_s{}".format(algo, "official_statistics", seed)
            if (kb in completed_cv and "error" not in completed_cv[kb] and
                    ko in completed_cv and "error" not in completed_cv[ko]):
                cv_sr.append(completed_cv[kb]["final_accuracy"] * 100)
                cv_os.append(completed_cv[ko]["final_accuracy"] * 100)

        hd_resil = "{:.1f}%".format(np.mean(hd_os) / np.mean(hd_sr) * 100) if hd_sr else "n/a"
        hd_delta = "{:+.1f}pp".format(np.mean(hd_os) - np.mean(hd_sr)) if hd_sr else ""
        cv_resil = "{:.1f}%".format(np.mean(cv_os) / np.mean(cv_sr) * 100) if cv_sr else "n/a"
        cv_delta = "{:+.1f}pp".format(np.mean(cv_os) - np.mean(cv_sr)) if cv_sr else ""

        log("  {:<8s} | {:>14s} {:>10s} | {:>14s} {:>10s}".format(
            algo, hd_resil, hd_delta, cv_resil, cv_delta))

    log("\n  Key: HD paradoxes (governance helps) are regularization artifacts")
    log("  on small data. CV (70K) shows expected governance costs.")


if __name__ == "__main__":
    main()

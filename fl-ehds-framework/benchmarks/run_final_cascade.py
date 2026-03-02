#!/usr/bin/env python3
"""
FL-EHDS — Final Cascade: MIA + Partial Participation + Budget Exhaustion.

Runs 3 complementary experiments in sequence with full intermediate checkpoints.

Block A: Membership Inference Attack (MIA) — ~18 experiments, ~40 min
    Tests whether DP actually protects against privacy attacks.
    Shadow-model MIA: trains an attacker that uses prediction confidence to
    distinguish training data (members) from unseen data (non-members).
    Compares No-DP vs DP-eps1 vs DP-eps10.
    Datasets: PTB-XL, Cardiovascular | Algos: FedAvg, Ditto, HPFL
    Total: 2 datasets x 3 algos x 3 DP levels = 18 experiments

Block B: Partial Client Participation — ~18 experiments, ~20 min
    Simulates realistic EHDS deployment where only a subset of hospitals
    participates in each round (e.g. due to availability, bandwidth, maintenance).
    Samples 3 of 5 clients per round randomly.
    Datasets: PTB-XL, Cardiovascular | Algos: FedAvg, Ditto, HPFL
    Total: 2 datasets x 3 algos x 3 seeds = 18 experiments

Block C: Privacy Budget Exhaustion — ~9 experiments, ~15 min
    Demonstrates JurisdictionPrivacyManager's budget accounting: clients
    with tight epsilon budgets are automatically deactivated when budget
    is exhausted, training continues with remaining clients.
    Dataset: PTB-XL | Algos: FedAvg, Ditto, HPFL | Budgets: tight/medium/generous
    Total: 3 algos x 3 budget levels = 9 experiments (single seed)

Grand total: 45 experiments, estimated ~1h 15min on Air M3.

Usage:
    cd fl-ehds-framework
    python -m benchmarks.run_final_cascade [--quick] [--fresh]

Output:
    benchmarks/paper_results_tabular/checkpoint_mia.json
    benchmarks/paper_results_tabular/checkpoint_partial_participation.json
    benchmarks/paper_results_tabular/checkpoint_budget_exhaustion.json

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
from typing import Dict, List, Optional, Any, Tuple

import numpy as np

FRAMEWORK_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(FRAMEWORK_DIR))

import torch

from terminal.fl_trainer import FederatedTrainer, _detect_device
from data.ptbxl_loader import load_ptbxl_data
from data.cardiovascular_loader import load_cardiovascular_data
from governance.jurisdiction_privacy import JurisdictionPrivacyManager

# ======================================================================
# Constants
# ======================================================================

OUTPUT_DIR = FRAMEWORK_DIR / "benchmarks" / "paper_results_tabular"
LOG_FILE = "experiment_final_cascade.log"

ALGORITHMS = ["FedAvg", "Ditto", "HPFL"]
SEEDS = [42, 123, 456]

PTB_XL_CLASS_NAMES = {0: "NORM", 1: "MI", 2: "STTC", 3: "CD", 4: "HYP"}

DATASET_CONFIGS = {
    "PTB_XL": {
        "loader": "ptbxl",
        "input_dim": 9,
        "num_classes": 5,
        "num_clients": 5,
        "short": "PX",
        "class_names": PTB_XL_CLASS_NAMES,
        "training": dict(
            learning_rate=0.005,
            batch_size=64,
            num_rounds=30,
            local_epochs=3,
            mu=0.1,
        ),
    },
    "Cardiovascular": {
        "loader": "cardiovascular",
        "input_dim": 11,
        "num_classes": 2,
        "num_clients": 5,
        "short": "CV",
        "class_names": {0: "Healthy", 1: "Disease"},
        "training": dict(
            learning_rate=0.01,
            batch_size=64,
            num_rounds=25,
            local_epochs=3,
            mu=0.1,
        ),
    },
}

# ======================================================================
# Logging
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
# Checkpoint (atomic write with backup) — per-block
# ======================================================================

def save_checkpoint(data, checkpoint_file):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / checkpoint_file
    bak = OUTPUT_DIR / (checkpoint_file + ".bak")
    data["metadata"]["last_save"] = datetime.now().isoformat()
    fd, tmp = tempfile.mkstemp(dir=str(OUTPUT_DIR), prefix=".cascade_", suffix=".tmp")
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


def load_checkpoint(checkpoint_file):
    path = OUTPUT_DIR / checkpoint_file
    bak = OUTPUT_DIR / (checkpoint_file + ".bak")
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

def load_dataset(dataset_name, num_clients, seed):
    cfg = DATASET_CONFIGS[dataset_name]
    if cfg["loader"] == "ptbxl":
        return load_ptbxl_data(
            num_clients=num_clients, seed=seed,
            partition_by_site=True, min_site_samples=50,
        )
    elif cfg["loader"] == "cardiovascular":
        return load_cardiovascular_data(
            num_clients=num_clients, seed=seed,
            is_iid=False, alpha=0.5,
        )
    else:
        raise ValueError("Unknown loader: {}".format(cfg["loader"]))


# ======================================================================
# Shared evaluation helpers
# ======================================================================

def _evaluate_per_client(trainer):
    """Evaluate all clients (including HPFL personalized heads)."""
    model = trainer.global_model
    model.eval()
    per_client = {}

    is_hpfl = trainer.algorithm == "HPFL"
    if is_hpfl:
        saved_cls = {n: p.data.clone() for n, p in model.named_parameters()
                     if n in trainer._hpfl_classifier_names}

    with torch.no_grad():
        for cid in range(trainer.num_clients):
            if is_hpfl:
                for n, p in model.named_parameters():
                    if n in trainer._hpfl_classifier_names:
                        p.data.copy_(trainer.client_classifiers[cid][n])
            X, y = trainer.client_test_data[cid]
            X_t = torch.FloatTensor(X).to(trainer.device) if isinstance(X, np.ndarray) else X.to(trainer.device)
            y_t = torch.LongTensor(y).to(trainer.device) if isinstance(y, np.ndarray) else y.to(trainer.device)
            correct = total = 0
            for i in range(0, len(y_t), 64):
                out = model(X_t[i:i + 64])
                correct += (out.argmax(1) == y_t[i:i + 64]).sum().item()
                total += len(y_t[i:i + 64])
            per_client[str(cid)] = correct / total if total > 0 else 0.0

    if is_hpfl:
        for n, p in model.named_parameters():
            if n in trainer._hpfl_classifier_names:
                p.data.copy_(saved_cls[n])
    return per_client


def _collect_predictions(trainer):
    """Collect all predictions for per-class metrics."""
    model = trainer.global_model
    model.eval()
    all_preds = []
    all_labels = []

    is_hpfl = trainer.algorithm == "HPFL"
    if is_hpfl:
        saved_cls = {n: p.data.clone() for n, p in model.named_parameters()
                     if n in trainer._hpfl_classifier_names}

    with torch.no_grad():
        for cid in range(trainer.num_clients):
            if is_hpfl:
                for n, p in model.named_parameters():
                    if n in trainer._hpfl_classifier_names:
                        p.data.copy_(trainer.client_classifiers[cid][n])
            X, y = trainer.client_test_data[cid]
            X_t = torch.FloatTensor(X).to(trainer.device) if isinstance(X, np.ndarray) else X.to(trainer.device)
            out = model(X_t)
            preds = out.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(y.tolist() if hasattr(y, "tolist") else list(y))

    if is_hpfl:
        for n, p in model.named_parameters():
            if n in trainer._hpfl_classifier_names:
                p.data.copy_(saved_cls[n])

    return np.array(all_labels), np.array(all_preds)


def _compute_per_class_metrics(y_true, y_pred, num_classes, class_names):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1

    per_class = {}
    for c in range(num_classes):
        cls_name = class_names.get(c, "Class_{}".format(c))
        tp = cm[c, c]
        fn = cm[c, :].sum() - tp
        fp = cm[:, c].sum() - tp
        support = cm[c, :].sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        per_class[cls_name] = {
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1": round(float(f1), 4),
            "support": int(support),
        }

    per_class["macro_avg"] = {
        "precision": round(float(np.mean([v["precision"] for k, v in per_class.items() if k != "macro_avg"])), 4),
        "recall": round(float(np.mean([v["recall"] for k, v in per_class.items() if k != "macro_avg"])), 4),
        "f1": round(float(np.mean([v["f1"] for k, v in per_class.items() if k != "macro_avg"])), 4),
    }

    return per_class, cm.tolist()


def _compute_dei(per_class_metrics, num_classes, class_names):
    recalls = []
    for c in range(num_classes):
        cls_name = class_names.get(c, "Class_{}".format(c))
        if cls_name in per_class_metrics:
            recalls.append(per_class_metrics[cls_name]["recall"])
    if not recalls:
        return 0.0
    recalls = np.array(recalls)
    min_recall = float(np.min(recalls))
    mean_recall = float(np.mean(recalls))
    if mean_recall == 0:
        return 0.0
    cv = float(np.std(recalls) / mean_recall)
    return round(min_recall * (1.0 - cv), 4)


def _compute_jain_index(per_client_acc):
    accs = list(per_client_acc.values())
    if not accs:
        return 0.0
    accs = np.array(accs)
    n = len(accs)
    if n == 0 or np.sum(accs ** 2) == 0:
        return 0.0
    return float((np.sum(accs) ** 2) / (n * np.sum(accs ** 2)))


def _get_confidence_scores(model, X, device, batch_size=64):
    """Get prediction confidence (max softmax) for each sample."""
    model.eval()
    confidences = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            X_batch = torch.FloatTensor(X[i:i + batch_size]).to(device) if isinstance(X, np.ndarray) \
                else X[i:i + batch_size].to(device)
            logits = model(X_batch)
            probs = torch.softmax(logits, dim=1)
            max_conf = probs.max(dim=1).values.cpu().numpy()
            confidences.extend(max_conf.tolist())
    return np.array(confidences)


def format_time(seconds):
    return str(timedelta(seconds=int(seconds)))


# ######################################################################
#                       BLOCK A: MIA
# ######################################################################

def _run_mia_attack(trainer, client_data, client_test, dataset_cfg, seed):
    """
    Confidence-based Membership Inference Attack.

    For each client:
      - Members = training data (should have higher confidence)
      - Non-members = test data of OTHER clients (model never trained on)
      - Attack metric: AUC of confidence-based binary classifier

    Returns per-client AUC and overall AUC.
    """
    model = trainer.global_model
    model.eval()
    device = trainer.device
    num_clients = trainer.num_clients

    is_hpfl = trainer.algorithm == "HPFL"
    if is_hpfl:
        saved_cls = {n: p.data.clone() for n, p in model.named_parameters()
                     if n in trainer._hpfl_classifier_names}

    all_member_conf = []
    all_nonmember_conf = []
    per_client_auc = {}

    for cid in range(num_clients):
        # Set HPFL classifier head for this client
        if is_hpfl:
            for n, p in model.named_parameters():
                if n in trainer._hpfl_classifier_names:
                    p.data.copy_(trainer.client_classifiers[cid][n])

        # Members: training data of this client
        X_train, y_train = client_data[cid]
        member_conf = _get_confidence_scores(model, X_train, device)

        # Non-members: test data from OTHER clients
        nonmember_X = []
        nonmember_y = []
        for other_cid in range(num_clients):
            if other_cid != cid:
                X_other, y_other = client_test[other_cid]
                if isinstance(X_other, np.ndarray):
                    nonmember_X.append(X_other)
                else:
                    nonmember_X.append(X_other.numpy())
                if isinstance(y_other, np.ndarray):
                    nonmember_y.append(y_other)
                else:
                    nonmember_y.append(y_other.numpy())

        if nonmember_X:
            X_nm = np.concatenate(nonmember_X, axis=0)
            nonmember_conf = _get_confidence_scores(model, X_nm, device)
        else:
            nonmember_conf = np.array([])

        # Compute AUC: member should have higher confidence
        if len(member_conf) > 0 and len(nonmember_conf) > 0:
            labels = np.concatenate([
                np.ones(len(member_conf)),
                np.zeros(len(nonmember_conf))
            ])
            scores = np.concatenate([member_conf, nonmember_conf])
            # Manual AUC computation (avoid sklearn dependency)
            auc = _compute_auc(labels, scores)
            per_client_auc[str(cid)] = round(auc, 4)
        else:
            per_client_auc[str(cid)] = 0.5

        all_member_conf.extend(member_conf.tolist())
        all_nonmember_conf.extend(nonmember_conf.tolist())

    # Restore HPFL classifier
    if is_hpfl:
        for n, p in model.named_parameters():
            if n in trainer._hpfl_classifier_names:
                p.data.copy_(saved_cls[n])

    # Overall AUC
    if all_member_conf and all_nonmember_conf:
        all_labels = np.concatenate([
            np.ones(len(all_member_conf)),
            np.zeros(len(all_nonmember_conf))
        ])
        all_scores = np.concatenate([all_member_conf, all_nonmember_conf])
        overall_auc = _compute_auc(all_labels, all_scores)
    else:
        overall_auc = 0.5

    return {
        "overall_auc": round(overall_auc, 4),
        "per_client_auc": per_client_auc,
        "member_conf_mean": round(float(np.mean(all_member_conf)), 4) if all_member_conf else 0.0,
        "member_conf_std": round(float(np.std(all_member_conf)), 4) if all_member_conf else 0.0,
        "nonmember_conf_mean": round(float(np.mean(all_nonmember_conf)), 4) if all_nonmember_conf else 0.0,
        "nonmember_conf_std": round(float(np.std(all_nonmember_conf)), 4) if all_nonmember_conf else 0.0,
    }


def _compute_auc(labels, scores):
    """Compute AUC using the Wilcoxon-Mann-Whitney statistic (no sklearn)."""
    # Sort by score descending
    sorted_indices = np.argsort(-scores)
    sorted_labels = labels[sorted_indices]
    sorted_scores = scores[sorted_indices]

    n_pos = int(np.sum(labels == 1))
    n_neg = int(np.sum(labels == 0))

    if n_pos == 0 or n_neg == 0:
        return 0.5

    # Trapezoidal AUC
    tp = 0
    fp = 0
    auc = 0.0
    prev_fpr = 0.0
    prev_tpr = 0.0

    for i in range(len(sorted_labels)):
        if sorted_labels[i] == 1:
            tp += 1
        else:
            fp += 1
        tpr = tp / n_pos
        fpr = fp / n_neg
        auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2
        prev_fpr = fpr
        prev_tpr = tpr

    return auc


def run_block_a_mia(quick=False):
    """Block A: Membership Inference Attack."""
    CHECKPOINT = "checkpoint_mia.json"
    log("=" * 70)
    log("BLOCK A: MEMBERSHIP INFERENCE ATTACK (MIA)")
    log("=" * 70)

    # Load existing checkpoint
    ckpt = load_checkpoint(CHECKPOINT)
    if ckpt is None:
        ckpt = {
            "metadata": {
                "block": "A_MIA",
                "description": "Membership Inference Attack — confidence-based",
                "started": datetime.now().isoformat(),
            },
            "results": {},
            "completed": [],
        }

    dp_levels = [
        {"label": "No-DP", "dp_enabled": False, "dp_epsilon": 0},
        {"label": "DP-eps1", "dp_enabled": True, "dp_epsilon": 1.0},
        {"label": "DP-eps10", "dp_enabled": True, "dp_epsilon": 10.0},
    ]

    experiments = []
    for ds_name in ["PTB_XL", "Cardiovascular"]:
        for algo in ALGORITHMS:
            for dp_cfg in dp_levels:
                exp_key = "{}_{}_{}_s42".format(ds_name, algo, dp_cfg["label"])
                experiments.append((ds_name, algo, dp_cfg, 42, exp_key))

    total = len(experiments)
    done = len(ckpt["completed"])
    log("MIA: {} experiments total, {} already done".format(total, done))

    block_start = time.time()

    for idx, (ds_name, algo, dp_cfg, seed, exp_key) in enumerate(experiments, 1):
        if _shutdown:
            log("Shutdown requested — saving checkpoint")
            break
        if exp_key in ckpt["completed"]:
            continue

        cfg = DATASET_CONFIGS[ds_name]
        tcfg = cfg["training"]
        num_rounds = 5 if quick else tcfg["num_rounds"]
        dp_label = dp_cfg["label"]

        log("[{}/{}] MIA: {} | {} | {} ...".format(idx, total, ds_name, algo, dp_label))
        exp_start = time.time()

        try:
            client_data, client_test, metadata = load_dataset(ds_name, cfg["num_clients"], seed)

            trainer_kwargs = dict(
                num_clients=cfg["num_clients"],
                algorithm=algo,
                local_epochs=tcfg["local_epochs"],
                batch_size=tcfg["batch_size"],
                learning_rate=tcfg["learning_rate"],
                mu=tcfg["mu"],
                seed=seed,
                external_data=client_data,
                external_test_data=client_test,
                input_dim=cfg["input_dim"],
                num_classes=cfg["num_classes"],
            )

            if dp_cfg["dp_enabled"]:
                trainer_kwargs["dp_enabled"] = True
                trainer_kwargs["dp_epsilon"] = dp_cfg["dp_epsilon"]
                trainer_kwargs["dp_clip_norm"] = 1.0

            trainer = FederatedTrainer(**trainer_kwargs)

            # Train the model
            for r in range(num_rounds):
                trainer.train_round(r)

            # Get model accuracy
            per_client_acc = _evaluate_per_client(trainer)
            final_acc = float(np.mean(list(per_client_acc.values())))

            # Run MIA
            mia_results = _run_mia_attack(trainer, client_data, client_test, cfg, seed)

            elapsed = time.time() - exp_start
            result = {
                "dataset": ds_name,
                "algorithm": algo,
                "dp_level": dp_label,
                "dp_epsilon": dp_cfg["dp_epsilon"],
                "seed": seed,
                "final_accuracy": round(final_acc, 4),
                "per_client_accuracy": per_client_acc,
                "mia_overall_auc": mia_results["overall_auc"],
                "mia_per_client_auc": mia_results["per_client_auc"],
                "mia_member_conf_mean": mia_results["member_conf_mean"],
                "mia_member_conf_std": mia_results["member_conf_std"],
                "mia_nonmember_conf_mean": mia_results["nonmember_conf_mean"],
                "mia_nonmember_conf_std": mia_results["nonmember_conf_std"],
                "time_seconds": round(elapsed, 1),
            }

            ckpt["results"][exp_key] = result
            ckpt["completed"].append(exp_key)
            save_checkpoint(ckpt, CHECKPOINT)

            log("  -> acc={:.1f}% | MIA AUC={:.3f} | conf_gap={:.3f} | {:.0f}s".format(
                final_acc * 100,
                mia_results["overall_auc"],
                mia_results["member_conf_mean"] - mia_results["nonmember_conf_mean"],
                elapsed))

            _cleanup_gpu()

        except Exception as e:
            log("  ERROR: {}".format(e))
            log("  " + traceback.format_exc().split("\n")[-2])
            _cleanup_gpu()

    block_time = time.time() - block_start
    ckpt["metadata"]["total_time"] = format_time(block_time)
    ckpt["metadata"]["completed_count"] = len(ckpt["completed"])
    ckpt["metadata"]["total_count"] = total
    save_checkpoint(ckpt, CHECKPOINT)

    log("Block A MIA done: {}/{} in {}".format(
        len(ckpt["completed"]), total, format_time(block_time)))
    log("")

    return len(ckpt["completed"]) == total


# ######################################################################
#                 BLOCK B: PARTIAL CLIENT PARTICIPATION
# ######################################################################

def run_block_b_partial(quick=False):
    """Block B: Partial Client Participation (3 of 5 per round)."""
    CHECKPOINT = "checkpoint_partial_participation.json"
    log("=" * 70)
    log("BLOCK B: PARTIAL CLIENT PARTICIPATION (3/5 per round)")
    log("=" * 70)

    ckpt = load_checkpoint(CHECKPOINT)
    if ckpt is None:
        ckpt = {
            "metadata": {
                "block": "B_Partial_Participation",
                "description": "Partial client participation — 3 of 5 clients sampled per round",
                "started": datetime.now().isoformat(),
            },
            "results": {},
            "completed": [],
        }

    experiments = []
    for ds_name in ["PTB_XL", "Cardiovascular"]:
        for algo in ALGORITHMS:
            for seed in SEEDS:
                exp_key = "{}_{}_s{}".format(ds_name, algo, seed)
                experiments.append((ds_name, algo, seed, exp_key))

    total = len(experiments)
    done = len(ckpt["completed"])
    log("Partial Participation: {} experiments total, {} already done".format(total, done))

    block_start = time.time()

    for idx, (ds_name, algo, seed, exp_key) in enumerate(experiments, 1):
        if _shutdown:
            log("Shutdown requested — saving checkpoint")
            break
        if exp_key in ckpt["completed"]:
            continue

        cfg = DATASET_CONFIGS[ds_name]
        tcfg = cfg["training"]
        num_rounds = 5 if quick else tcfg["num_rounds"]
        num_clients = cfg["num_clients"]
        clients_per_round = 3  # sample 3 of 5

        log("[{}/{}] Partial: {} | {} | seed={} ...".format(idx, total, ds_name, algo, seed))
        exp_start = time.time()

        try:
            client_data, client_test, metadata = load_dataset(ds_name, num_clients, seed)

            trainer = FederatedTrainer(
                num_clients=num_clients,
                algorithm=algo,
                local_epochs=tcfg["local_epochs"],
                batch_size=tcfg["batch_size"],
                learning_rate=tcfg["learning_rate"],
                mu=tcfg["mu"],
                seed=seed,
                external_data=client_data,
                external_test_data=client_test,
                input_dim=cfg["input_dim"],
                num_classes=cfg["num_classes"],
            )

            rng = np.random.RandomState(seed)
            all_clients = list(range(num_clients))
            history = []
            client_participation_count = {c: 0 for c in all_clients}

            for r in range(num_rounds):
                # Random subset: 3 of 5 clients
                active = sorted(rng.choice(all_clients, size=clients_per_round, replace=False).tolist())
                for c in active:
                    client_participation_count[c] += 1

                result = trainer.train_round(r, active_clients=active)
                history.append({
                    "round": r + 1,
                    "accuracy": result.global_acc,
                    "loss": result.global_loss,
                    "f1": result.global_f1,
                    "active_clients": active,
                })

            # Final evaluation
            per_client_acc = _evaluate_per_client(trainer)
            jain = _compute_jain_index(per_client_acc)
            final_acc = float(np.mean(list(per_client_acc.values())))

            y_true, y_pred = _collect_predictions(trainer)
            per_class, cm = _compute_per_class_metrics(
                y_true, y_pred, cfg["num_classes"], cfg["class_names"])
            dei = _compute_dei(per_class, cfg["num_classes"], cfg["class_names"])

            elapsed = time.time() - exp_start
            result = {
                "dataset": ds_name,
                "algorithm": algo,
                "seed": seed,
                "clients_per_round": clients_per_round,
                "total_clients": num_clients,
                "participation_rate": clients_per_round / num_clients,
                "final_accuracy": round(final_acc, 4),
                "per_client_accuracy": per_client_acc,
                "jain_index": round(jain, 4),
                "dei": dei,
                "per_class_metrics": per_class,
                "confusion_matrix": cm,
                "client_participation_count": client_participation_count,
                "convergence_trajectory": [h["accuracy"] for h in history],
                "time_seconds": round(elapsed, 1),
            }

            ckpt["results"][exp_key] = result
            ckpt["completed"].append(exp_key)
            save_checkpoint(ckpt, CHECKPOINT)

            log("  -> acc={:.1f}% | Jain={:.3f} | DEI={:.3f} | {:.0f}s".format(
                final_acc * 100, jain, dei, elapsed))

            _cleanup_gpu()

        except Exception as e:
            log("  ERROR: {}".format(e))
            log("  " + traceback.format_exc().split("\n")[-2])
            _cleanup_gpu()

    block_time = time.time() - block_start
    ckpt["metadata"]["total_time"] = format_time(block_time)
    ckpt["metadata"]["completed_count"] = len(ckpt["completed"])
    ckpt["metadata"]["total_count"] = total
    save_checkpoint(ckpt, CHECKPOINT)

    log("Block B Partial done: {}/{} in {}".format(
        len(ckpt["completed"]), total, format_time(block_time)))
    log("")

    return len(ckpt["completed"]) == total


# ######################################################################
#              BLOCK C: PRIVACY BUDGET EXHAUSTION
# ######################################################################

def run_block_c_budget(quick=False):
    """Block C: Privacy Budget Exhaustion via JurisdictionPrivacyManager."""
    CHECKPOINT = "checkpoint_budget_exhaustion.json"
    log("=" * 70)
    log("BLOCK C: PRIVACY BUDGET EXHAUSTION")
    log("=" * 70)

    ckpt = load_checkpoint(CHECKPOINT)
    if ckpt is None:
        ckpt = {
            "metadata": {
                "block": "C_Budget_Exhaustion",
                "description": "Privacy budget exhaustion — clients deactivated when RDP budget depletes",
                "started": datetime.now().isoformat(),
            },
            "results": {},
            "completed": [],
        }

    # Budget scenarios: tight, medium, generous
    # epsilon_per_client is the total budget over all rounds
    budget_scenarios = [
        {
            "label": "Tight",
            "description": "Tight budget: eps=2 over 30 rounds — clients exhaust early",
            "epsilon_budget": 2.0,
        },
        {
            "label": "Medium",
            "description": "Medium budget: eps=5 over 30 rounds — some clients exhaust",
            "epsilon_budget": 5.0,
        },
        {
            "label": "Generous",
            "description": "Generous budget: eps=10 over 30 rounds — most survive",
            "epsilon_budget": 10.0,
        },
    ]

    # EU jurisdiction mapping for PTB-XL 5 clients
    jurisdictions = {0: "DE", 1: "DE", 2: "IT", 3: "IT", 4: "FR"}
    hospital_names = {
        0: "Charite Berlin", 1: "LMU Munich",
        2: "Policlinico Roma", 3: "San Raffaele Milano",
        4: "Hopital Necker Paris",
    }

    experiments = []
    for algo in ALGORITHMS:
        for budget_cfg in budget_scenarios:
            exp_key = "{}_{}".format(algo, budget_cfg["label"])
            experiments.append((algo, budget_cfg, exp_key))

    total = len(experiments)
    done = len(ckpt["completed"])
    log("Budget Exhaustion: {} experiments total, {} already done".format(total, done))

    block_start = time.time()

    for idx, (algo, budget_cfg, exp_key) in enumerate(experiments, 1):
        if _shutdown:
            log("Shutdown requested — saving checkpoint")
            break
        if exp_key in ckpt["completed"]:
            continue

        seed = 42  # single seed for budget exhaustion
        num_rounds = 5 if quick else 30
        num_clients = 5
        eps_budget = budget_cfg["epsilon_budget"]

        log("[{}/{}] Budget: {} | {} (eps={}) ...".format(
            idx, total, algo, budget_cfg["label"], eps_budget))
        exp_start = time.time()

        try:
            client_data, client_test, metadata = load_dataset("PTB_XL", num_clients, seed)

            # Create trainer WITH DP enabled (central mode for budget accounting)
            trainer = FederatedTrainer(
                num_clients=num_clients,
                algorithm=algo,
                local_epochs=3,
                batch_size=64,
                learning_rate=0.005,
                mu=0.1,
                dp_enabled=True,
                dp_epsilon=eps_budget,
                dp_clip_norm=1.0,
                seed=seed,
                external_data=client_data,
                external_test_data=client_test,
                input_dim=9,
                num_classes=5,
            )

            # Create JurisdictionPrivacyManager for budget tracking
            # All clients get the same epsilon budget but different jurisdictions
            jurisdiction_budgets = {
                "DE": {"epsilon_max": eps_budget, "delta": 1e-5},
                "IT": {"epsilon_max": eps_budget, "delta": 1e-5},
                "FR": {"epsilon_max": eps_budget, "delta": 1e-5},
            }

            jpm = JurisdictionPrivacyManager(
                client_jurisdictions=jurisdictions,
                jurisdiction_budgets=jurisdiction_budgets,
                global_epsilon=eps_budget,
                num_rounds=num_rounds,
                hospital_names=hospital_names,
                hospital_allocation_fraction=1.0,
                noise_strategy="global",
                accountant_type="rdp",
                min_active_clients=2,
            )

            history = []
            active_count_timeline = []
            deactivation_events = []

            for r in range(num_rounds):
                # Pre-round: check which clients still have budget
                active_clients, noise_scale = jpm.pre_round_check(r, noise_multiplier=0.0)

                if len(active_clients) < jpm.min_active_clients:
                    log("    Round {}: only {} active clients — stopping".format(
                        r + 1, len(active_clients)))
                    break

                # Train only active clients
                result = trainer.train_round(r, active_clients=active_clients)

                # Post-round: record spending
                jpm.record_round(
                    round_num=r,
                    participating_clients=active_clients,
                    effective_noise_multiplier=noise_scale,
                    sampling_rate=1.0,
                )

                per_client_r = _evaluate_per_client(trainer)
                history.append({
                    "round": r + 1,
                    "accuracy": result.global_acc,
                    "loss": result.global_loss,
                    "active_clients": len(active_clients),
                    "active_client_ids": active_clients,
                    "per_client_accuracy": per_client_r,
                })
                active_count_timeline.append(len(active_clients))

                # Check for new deactivations
                new_events = [e for e in jpm.get_dropout_timeline()
                              if e["round"] == r and e not in deactivation_events]
                for evt in new_events:
                    deactivation_events.append(evt)
                    log("    Round {}: Client {} ({}) deactivated — {}".format(
                        r + 1, evt["client_id"], evt["hospital"],
                        evt["reason"]))

            # Final evaluation
            per_client_acc = _evaluate_per_client(trainer)
            jain = _compute_jain_index(per_client_acc)
            final_acc = float(np.mean(list(per_client_acc.values())))

            # Get JPM report
            jpm_report = jpm.export_report()

            elapsed = time.time() - exp_start
            result_data = {
                "algorithm": algo,
                "budget_level": budget_cfg["label"],
                "epsilon_budget": eps_budget,
                "seed": seed,
                "total_rounds_completed": len(history),
                "final_accuracy": round(final_acc, 4),
                "per_client_accuracy": per_client_acc,
                "jain_index": round(jain, 4),
                "active_count_timeline": active_count_timeline,
                "deactivation_events": deactivation_events,
                "convergence_trajectory": [h["accuracy"] for h in history],
                "jpm_summary": jpm_report["summary"],
                "jpm_per_client": jpm_report["per_client"],
                "jpm_per_jurisdiction": {
                    k: {kk: vv for kk, vv in v.items() if kk != "clients"}
                    for k, v in jpm_report["per_jurisdiction"].items()
                },
                "time_seconds": round(elapsed, 1),
            }

            ckpt["results"][exp_key] = result_data
            ckpt["completed"].append(exp_key)
            save_checkpoint(ckpt, CHECKPOINT)

            final_active = jpm_report["summary"]["active_clients"]
            exhausted = jpm_report["summary"]["exhausted_clients"]
            log("  -> acc={:.1f}% | rounds={} | active={}/5 | exhausted={} | {:.0f}s".format(
                final_acc * 100, len(history), final_active, exhausted, elapsed))

            _cleanup_gpu()

        except Exception as e:
            log("  ERROR: {}".format(e))
            log("  " + traceback.format_exc().split("\n")[-2])
            _cleanup_gpu()

    block_time = time.time() - block_start
    ckpt["metadata"]["total_time"] = format_time(block_time)
    ckpt["metadata"]["completed_count"] = len(ckpt["completed"])
    ckpt["metadata"]["total_count"] = total
    save_checkpoint(ckpt, CHECKPOINT)

    log("Block C Budget done: {}/{} in {}".format(
        len(ckpt["completed"]), total, format_time(block_time)))
    log("")

    return len(ckpt["completed"]) == total


# ######################################################################
#                           MAIN CASCADE
# ######################################################################

def main():
    global _log_file

    parser = argparse.ArgumentParser(
        description="FL-EHDS Final Cascade: MIA + Partial Participation + Budget Exhaustion")
    parser.add_argument("--quick", action="store_true",
                        help="Quick validation mode (5 rounds, faster)")
    parser.add_argument("--fresh", action="store_true",
                        help="Delete all checkpoints and start fresh")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _log_file = open(str(OUTPUT_DIR / LOG_FILE), "a")

    log("=" * 70)
    log("FL-EHDS FINAL CASCADE — MIA + Partial + Budget")
    log("Mode: {}".format("QUICK" if args.quick else "FULL"))
    log("Device: {}".format(_detect_device("cpu")))
    log("=" * 70)

    if args.fresh:
        for ckpt_name in [
            "checkpoint_mia.json",
            "checkpoint_partial_participation.json",
            "checkpoint_budget_exhaustion.json",
        ]:
            for suffix in ["", ".bak"]:
                p = OUTPUT_DIR / (ckpt_name + suffix)
                if p.exists():
                    p.unlink()
                    log("Deleted {}".format(ckpt_name + suffix))

    cascade_start = time.time()
    blocks_done = 0

    # Block A: MIA
    if not _shutdown:
        if run_block_a_mia(quick=args.quick):
            blocks_done += 1

    # Block B: Partial Client Participation
    if not _shutdown:
        if run_block_b_partial(quick=args.quick):
            blocks_done += 1

    # Block C: Privacy Budget Exhaustion
    if not _shutdown:
        if run_block_c_budget(quick=args.quick):
            blocks_done += 1

    total_time = time.time() - cascade_start

    log("=" * 70)
    log("CASCADE COMPLETE: {}/3 blocks done in {}".format(
        blocks_done, format_time(total_time)))
    log("Checkpoints:")
    log("  - checkpoint_mia.json")
    log("  - checkpoint_partial_participation.json")
    log("  - checkpoint_budget_exhaustion.json")
    log("=" * 70)

    if _log_file:
        _log_file.close()


if __name__ == "__main__":
    main()

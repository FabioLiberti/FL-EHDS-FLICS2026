#!/usr/bin/env python3
"""
FL-EHDS Experiment — Extended Governance Validation (E1+E2+E3+E4).

Extends the initial governance validation with four experiments:

  E1) Data Minimization 10-seed on Heart Disease (statistical robustness)
      4 purposes × 10 seeds = 40 experiments
  E2) Data Minimization 10-seed on Cardiovascular (cross-dataset validation)
      4 purposes × 10 seeds = 40 experiments
  E3) Governance Overhead Timing (per-round cost measurement)
      10 seeds on Heart Disease = 10 experiments
  E4) Progressive Opt-out under Governance (accuracy degradation curve)
      5 opt-out rates × 10 seeds = 50 experiments

Total: 140 experiments (~15-20 min on M3)

Usage:
    cd fl-ehds-framework
    python -m benchmarks.run_governance_extended [--fresh]

Output:
    benchmarks/paper_results_tabular/checkpoint_governance_ext.json

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
    PermitStatus,
)
from governance.compliance_logging import AuditTrail, ComplianceLogger
from governance.data_permits import DataPermitManager, PermitValidator
from governance.data_minimization import DataMinimizer
from governance.optout_registry import OptOutRegistry, OptOutChecker
from governance.permit_training import PermitAwareTrainingContext

# ======================================================================
# Constants
# ======================================================================

OUTPUT_DIR = FRAMEWORK_DIR / "benchmarks" / "paper_results_tabular"
CHECKPOINT_FILE = "checkpoint_governance_ext.json"
LOG_FILE = "experiment_governance_ext.log"

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

# Feature → semantic group mapping: Heart Disease
HD_FEATURE_GROUPS = {
    "age": "demographics",
    "sex": "demographics",
    "chest_pain_type": "conditions",
    "resting_bp": "vitals",
    "cholesterol": "measurements",
    "fasting_blood_sugar": "measurements",
    "resting_ecg": "measurements",
    "max_heart_rate": "vitals",
    "exercise_angina": "conditions",
    "st_depression": "measurements",
    "st_slope": "measurements",
    "num_major_vessels": "measurements",
    "thalassemia": "conditions",
}

# Feature → semantic group mapping: Cardiovascular
CV_FEATURE_GROUPS = {
    "age": "demographics",
    "gender": "demographics",
    "height": "demographics",
    "weight": "demographics",
    "ap_hi": "vitals",
    "ap_lo": "vitals",
    "cholesterol": "measurements",
    "gluc": "measurements",
    "smoke": "conditions",
    "alco": "conditions",
    "active": "conditions",
}

MINIMIZATION_PURPOSES = [
    "scientific_research",          # None = all features
    "public_health_surveillance",   # demographics + conditions + vitals + medications
    "patient_safety",               # conditions + medications + vitals + measurements
    "official_statistics",          # demographics only
]

OPTOUT_RATES = [0.0, 0.05, 0.10, 0.20, 0.30]

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
    fd, tmp = tempfile.mkstemp(dir=str(OUTPUT_DIR), prefix=".gex_", suffix=".tmp")
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
# Evaluation helper
# ======================================================================

def _evaluate_model(model, X, y, batch_size, device, num_classes):
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score,
        recall_score, roc_auc_score,
    )

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
            auc = roc_auc_score(all_labels, all_probs, multi_class="ovr", average="macro")
    except Exception:
        auc = 0.5

    return {
        "accuracy": float(acc), "f1": float(f1),
        "precision": float(prec), "recall": float(rec),
        "auc": float(auc), "loss": float(total_loss / max(n_batches, 1)),
    }


# ======================================================================
# E1/E2: Data Minimization (Article 44)
# ======================================================================

def run_minimization(dataset_name, purpose, seed):
    """
    Train FedAvg with purpose-dependent data minimization.
    Works for both Heart Disease and Cardiovascular datasets.
    """
    if dataset_name == "HD":
        cfg = HD_CONFIG
        client_data, client_test, meta, feature_names = load_hd(seed)
        feat_groups = HD_FEATURE_GROUPS
    else:
        cfg = CV_CONFIG
        client_data, client_test, meta, feature_names = load_cv(seed)
        feat_groups = CV_FEATURE_GROUPS

    # Temporarily register dataset features in DataMinimizer
    original_groups = dict(DataMinimizer.FEATURE_GROUPS)
    for fname, group in feat_groups.items():
        DataMinimizer.FEATURE_GROUPS[fname] = group

    try:
        filtered_train, filtered_test, min_report = DataMinimizer.apply_minimization(
            train_data=client_data,
            test_data=client_test,
            purpose=purpose,
            feature_names=feature_names,
            importance_threshold=0.01,
        )
    finally:
        DataMinimizer.FEATURE_GROUPS = original_groups

    n_kept = min_report["kept_features"]

    # Train FedAvg with reduced features
    trainer = FederatedTrainer(
        num_clients=cfg["num_clients"],
        algorithm="FedAvg",
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
        "dataset": dataset_name,
        "seed": seed,
        "purpose": purpose,
        "original_features": min_report["original_features"],
        "kept_features": n_kept,
        "kept_feature_names": min_report["kept_feature_names"],
        "reduction_pct": min_report["reduction_pct"],
        "allowed_groups": min_report["allowed_groups"],
        "purpose_removed": min_report["purpose_removed"],
        "importance_removed": min_report["importance_removed"],
        "importance_scores": min_report["importance_scores"],
        "final_accuracy": final.get("accuracy", 0),
        "final_f1": final.get("f1", 0),
        "final_auc": final.get("auc", 0),
        "final_loss": final.get("loss", 0),
        "history": history,
    }


# ======================================================================
# E3: Governance Overhead Timing
# ======================================================================

def run_overhead_timing(seed):
    """
    Measure per-round governance overhead: permit validation,
    opt-out checking, audit logging, budget tracking.
    Returns timing breakdown in microseconds.
    """
    cfg = HD_CONFIG
    client_data, client_test, meta, feature_names = load_hd(seed)

    # Set up governance
    permit_ctx = PermitAwareTrainingContext(
        permit_id="PERMIT-TIMING-{}".format(seed),
        purpose=PermitPurpose.SCIENTIFIC_RESEARCH,
        data_categories=[DataCategory.EHR, DataCategory.LAB_RESULTS],
        privacy_budget_total=10.0,
        max_rounds=cfg["num_rounds"],
        client_ids=["client_{}".format(i) for i in range(cfg["num_clients"])],
        audit_output_dir=str(OUTPUT_DIR / "audit_timing_s{}".format(seed)),
    )

    # Set up opt-out registry with 10% opted out
    optout_registry = OptOutRegistry()
    rng = np.random.RandomState(seed)

    # Assign patient IDs
    all_patient_ids = []
    client_patient_ids = {}
    for cid in sorted(client_data.keys()):
        n = len(client_data[cid][1])
        pids = ["P-{}-{:04d}".format(cid, i) for i in range(n)]
        client_patient_ids[cid] = pids
        all_patient_ids.extend(pids)

    # Register 10% opt-out
    n_optout = max(1, int(len(all_patient_ids) * 0.10))
    optout_pids = set(rng.choice(all_patient_ids, size=n_optout, replace=False))
    for pid in optout_pids:
        optout_registry.register_optout(OptOutRecord(
            record_id="OPT-T-{}-{}".format(seed, pid),
            patient_id=pid, scope="all", member_state="IT",
        ))

    # Trainer (no governance overhead here — train_round is the baseline)
    trainer = FederatedTrainer(
        num_clients=cfg["num_clients"],
        algorithm="FedAvg",
        local_epochs=cfg["local_epochs"],
        batch_size=cfg["batch_size"],
        learning_rate=cfg["learning_rate"],
        mu=cfg["mu"],
        seed=seed,
        external_data=dict(client_data),
        external_test_data=dict(client_test),
        input_dim=cfg["input_dim"],
        num_classes=cfg["num_classes"],
    )

    session_id = permit_ctx.start_session()
    eps_per_round = 0.5

    timing_per_round = []

    for r in range(cfg["num_rounds"]):
        timings = {}

        # 1. Permit validation timing
        t0 = time.perf_counter()
        ok, reason = permit_ctx.validate_round(r, eps_per_round)
        timings["permit_validation_us"] = (time.perf_counter() - t0) * 1e6

        if not ok:
            break

        # 2. Opt-out checking timing (all patients, all clients)
        t0 = time.perf_counter()
        for cid in sorted(client_patient_ids.keys()):
            for pid in client_patient_ids[cid]:
                optout_registry.is_opted_out(pid)
        timings["optout_check_us"] = (time.perf_counter() - t0) * 1e6
        timings["optout_patients_checked"] = len(all_patient_ids)

        # 3. FL training round (baseline — NOT governance overhead)
        t0 = time.perf_counter()
        result = trainer.train_round(r)
        timings["training_us"] = (time.perf_counter() - t0) * 1e6

        # 4. Audit logging timing
        t0 = time.perf_counter()
        permit_ctx.log_round_completion(result, eps_per_round)
        timings["audit_logging_us"] = (time.perf_counter() - t0) * 1e6

        # Compute overhead percentage
        total_gov = (timings["permit_validation_us"] +
                     timings["optout_check_us"] +
                     timings["audit_logging_us"])
        total_all = total_gov + timings["training_us"]
        timings["governance_overhead_pct"] = (
            (total_gov / total_all * 100) if total_all > 0 else 0.0
        )
        timings["round"] = r
        timings["accuracy"] = result.global_acc

        timing_per_round.append(timings)

    permit_ctx.end_session(
        total_rounds=len(timing_per_round),
        final_metrics={"accuracy": result.global_acc},
        success=True,
    )

    # Aggregate timings
    if timing_per_round:
        avg_permit = np.mean([t["permit_validation_us"] for t in timing_per_round])
        avg_optout = np.mean([t["optout_check_us"] for t in timing_per_round])
        avg_audit = np.mean([t["audit_logging_us"] for t in timing_per_round])
        avg_training = np.mean([t["training_us"] for t in timing_per_round])
        avg_overhead = np.mean([t["governance_overhead_pct"] for t in timing_per_round])
    else:
        avg_permit = avg_optout = avg_audit = avg_training = avg_overhead = 0.0

    return {
        "seed": seed,
        "experiment": "E3_overhead_timing",
        "num_rounds": len(timing_per_round),
        "num_patients": len(all_patient_ids),
        "num_opted_out": n_optout,
        "per_round_timings": timing_per_round,
        "averages": {
            "permit_validation_us": round(avg_permit, 1),
            "optout_check_us": round(avg_optout, 1),
            "audit_logging_us": round(avg_audit, 1),
            "training_us": round(avg_training, 1),
            "governance_total_us": round(avg_permit + avg_optout + avg_audit, 1),
            "governance_overhead_pct": round(avg_overhead, 3),
        },
        "final_accuracy": timing_per_round[-1]["accuracy"] if timing_per_round else 0.0,
    }


# ======================================================================
# E4: Progressive Opt-out under Governance
# ======================================================================

def run_progressive_optout(optout_rate, seed):
    """
    Train FedAvg with full governance enforcement and a given opt-out rate.
    Measures accuracy with opt-out applied from round 0.
    """
    cfg = HD_CONFIG
    client_data, client_test, meta, feature_names = load_hd(seed)

    # Set up opt-out registry
    optout_registry = OptOutRegistry()
    rng = np.random.RandomState(seed)

    # Assign patient IDs
    client_patient_ids = {}
    all_patient_ids = []
    for cid in sorted(client_data.keys()):
        n = len(client_data[cid][1])
        pids = ["P-{}-{:04d}".format(cid, i) for i in range(n)]
        client_patient_ids[cid] = pids
        all_patient_ids.extend(pids)

    # Register opt-outs before training starts
    n_optout = int(len(all_patient_ids) * optout_rate)
    excluded_total = 0

    if n_optout > 0:
        optout_pids = set(rng.choice(all_patient_ids, size=n_optout, replace=False))
        for pid in optout_pids:
            optout_registry.register_optout(OptOutRecord(
                record_id="OPT-E4-{}-{}".format(seed, pid),
                patient_id=pid, scope="all", member_state="IT",
            ))

        # Filter training data
        filtered_data = {}
        for cid in sorted(client_data.keys()):
            pids = client_patient_ids[cid]
            X, y = client_data[cid]
            keep_mask = np.array([pid not in optout_pids for pid in pids])
            excluded_total += int((~keep_mask).sum())
            filtered_data[cid] = (X[keep_mask], y[keep_mask])
    else:
        filtered_data = dict(client_data)

    total_train_samples = sum(len(filtered_data[c][1]) for c in filtered_data)

    # Set up governance
    permit_ctx = PermitAwareTrainingContext(
        permit_id="PERMIT-E4-r{}-s{}".format(int(optout_rate * 100), seed),
        purpose=PermitPurpose.SCIENTIFIC_RESEARCH,
        data_categories=[DataCategory.EHR, DataCategory.LAB_RESULTS],
        privacy_budget_total=10.0,
        max_rounds=cfg["num_rounds"],
        client_ids=["client_{}".format(i) for i in range(cfg["num_clients"])],
    )

    session_id = permit_ctx.start_session()

    # Train with governance
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

    eps_per_round = 0.5
    history = []

    for r in range(cfg["num_rounds"]):
        ok, reason = permit_ctx.validate_round(r, eps_per_round)
        if not ok:
            break

        result = trainer.train_round(r)
        permit_ctx.log_round_completion(result, eps_per_round)

        history.append({
            "round": r + 1,
            "accuracy": result.global_acc,
            "loss": result.global_loss,
            "f1": result.global_f1,
            "auc": result.global_auc,
        })

    permit_ctx.end_session(
        total_rounds=len(history),
        final_metrics={"accuracy": history[-1]["accuracy"]} if history else {},
        success=True,
    )

    final = history[-1] if history else {}

    return {
        "seed": seed,
        "optout_rate": optout_rate,
        "experiment": "E4_progressive_optout",
        "total_patients": len(all_patient_ids),
        "opted_out": n_optout,
        "excluded_from_training": excluded_total,
        "remaining_train_samples": total_train_samples,
        "final_accuracy": final.get("accuracy", 0),
        "final_f1": final.get("f1", 0),
        "final_auc": final.get("auc", 0),
        "final_loss": final.get("loss", 0),
        "budget_status": permit_ctx.get_budget_status(),
        "history": history,
    }


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="FL-EHDS Extended Governance Validation (E1+E2+E3+E4)")
    parser.add_argument("--fresh", action="store_true",
                        help="Delete existing checkpoint and start fresh")
    args = parser.parse_args()

    global _log_file
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _log_file = open(OUTPUT_DIR / LOG_FILE, "a")

    # Build experiment list
    experiments = []

    # E1: HD minimization (4 purposes × 10 seeds = 40)
    for purpose in MINIMIZATION_PURPOSES:
        for seed in SEEDS_10:
            experiments.append({
                "key": "E1_HD_{}_s{}".format(purpose, seed),
                "type": "E1",
                "dataset": "HD",
                "purpose": purpose,
                "seed": seed,
            })

    # E2: CV minimization (4 purposes × 10 seeds = 40)
    for purpose in MINIMIZATION_PURPOSES:
        for seed in SEEDS_10:
            experiments.append({
                "key": "E2_CV_{}_s{}".format(purpose, seed),
                "type": "E2",
                "dataset": "CV",
                "purpose": purpose,
                "seed": seed,
            })

    # E3: Overhead timing (10 seeds)
    for seed in SEEDS_10:
        experiments.append({
            "key": "E3_timing_s{}".format(seed),
            "type": "E3",
            "seed": seed,
        })

    # E4: Progressive opt-out (5 rates × 10 seeds = 50)
    for rate in OPTOUT_RATES:
        for seed in SEEDS_10:
            experiments.append({
                "key": "E4_optout_{}_s{}".format(int(rate * 100), seed),
                "type": "E4",
                "optout_rate": rate,
                "seed": seed,
            })

    total_exps = len(experiments)  # 40 + 40 + 10 + 50 = 140

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
                "experiment_types": ["E1_HD_minimization", "E2_CV_minimization",
                                     "E3_overhead_timing", "E4_progressive_optout"],
                "seeds": SEEDS_10,
                "purposes": MINIMIZATION_PURPOSES,
                "optout_rates": OPTOUT_RATES,
                "datasets": {"HD": HD_CONFIG, "CV": CV_CONFIG},
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
        log("\nINTERRUPT -- saving checkpoint ({}/{})...".format(done, total_exps))
        save_checkpoint(checkpoint_data)
        log("Checkpoint saved. Resume: python -m benchmarks.run_governance_extended")
        sys.exit(0)

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # Header
    log("\n" + "=" * 72)
    log("  FL-EHDS Extended Governance Validation (E1+E2+E3+E4)")
    log("  {} experiments = 40 HD-min + 40 CV-min + 10 timing + 50 optout".format(
        total_exps))
    log("=" * 72)
    log("  Device:   {}".format(_detect_device()))
    log("  Seeds:    {} (10 seeds)".format(SEEDS_10))
    log("  Purposes: {}".format(MINIMIZATION_PURPOSES))
    log("  Opt-out:  {}".format(OPTOUT_RATES))
    log("  HD:       {} rounds, {} clients, {} samples".format(
        HD_CONFIG["num_rounds"], HD_CONFIG["num_clients"], "920"))
    log("  CV:       {} rounds, {} clients, {} samples".format(
        CV_CONFIG["num_rounds"], CV_CONFIG["num_clients"], "70K"))
    log("  Output:   {}".format(OUTPUT_DIR / CHECKPOINT_FILE))
    log("=" * 72)

    # Run experiments
    global_start = time.time()
    completed = checkpoint_data.get("completed", {})
    done_count = len(completed)

    for idx, exp in enumerate(experiments, 1):
        key = exp["key"]
        if key in completed:
            continue

        if _interrupted[0]:
            break

        log("[{}/{}] {} ...".format(done_count + 1, total_exps, key))

        try:
            start_t = time.time()

            if exp["type"] in ("E1", "E2"):
                result = run_minimization(exp["dataset"], exp["purpose"], exp["seed"])
            elif exp["type"] == "E3":
                result = run_overhead_timing(exp["seed"])
            elif exp["type"] == "E4":
                result = run_progressive_optout(exp["optout_rate"], exp["seed"])
            else:
                continue

            elapsed_t = time.time() - start_t
            result["runtime_seconds"] = round(elapsed_t, 1)
            completed[key] = result
            done_count += 1

            # Short progress log
            if exp["type"] in ("E1", "E2"):
                log("  -> {}: {}/{} feat, acc={:.1f}%, f1={:.3f} ({:.1f}s)".format(
                    exp["purpose"][:12], result["kept_features"],
                    result["original_features"],
                    result["final_accuracy"] * 100,
                    result["final_f1"], elapsed_t))
            elif exp["type"] == "E3":
                log("  -> overhead={:.3f}%, gov={:.0f}us, train={:.0f}us ({:.1f}s)".format(
                    result["averages"]["governance_overhead_pct"],
                    result["averages"]["governance_total_us"],
                    result["averages"]["training_us"], elapsed_t))
            elif exp["type"] == "E4":
                log("  -> optout={:.0f}%, acc={:.1f}%, excl={} ({:.1f}s)".format(
                    exp["optout_rate"] * 100,
                    result["final_accuracy"] * 100,
                    result["excluded_from_training"], elapsed_t))

            # Atomic save after each experiment
            save_checkpoint(checkpoint_data)

        except Exception as e:
            log("  ERROR: {}".format(e))
            traceback.print_exc()
            completed[key] = {"key": key, "error": str(e),
                              "traceback": traceback.format_exc()}
            save_checkpoint(checkpoint_data)
            _cleanup_gpu()

    # ======================================================================
    # Finalize
    # ======================================================================
    checkpoint_data["metadata"]["end_time"] = datetime.now().isoformat()
    checkpoint_data["metadata"]["total_elapsed"] = time.time() - global_start
    save_checkpoint(checkpoint_data)

    elapsed = time.time() - global_start

    # ======================================================================
    # Summary tables
    # ======================================================================
    log("\n" + "=" * 72)
    log("  COMPLETED: {}/{} ({:.0f}s)".format(done_count, total_exps, elapsed))
    log("=" * 72)

    # --- E1: HD Minimization ---
    log("\n  [E1] HEART DISEASE — Data Minimization (10 seeds)")
    _print_minimization_table("HD", "E1", completed)

    # --- E2: CV Minimization ---
    log("\n  [E2] CARDIOVASCULAR — Data Minimization (10 seeds)")
    _print_minimization_table("CV", "E2", completed)

    # --- E3: Overhead timing ---
    log("\n  [E3] GOVERNANCE OVERHEAD TIMING")
    _print_overhead_table(completed)

    # --- E4: Progressive opt-out ---
    log("\n  [E4] PROGRESSIVE OPT-OUT (Heart Disease, 10 seeds)")
    _print_optout_table(completed)

    # --- Statistical tests ---
    log("\n  [STAT] PAIRED T-TESTS (minimization vs scientific_research baseline)")
    _print_significance_tests("HD", "E1", completed)
    _print_significance_tests("CV", "E2", completed)

    log("\n  Done!")

    if _log_file:
        _log_file.close()


# ======================================================================
# Summary printing helpers
# ======================================================================

def _print_minimization_table(ds, prefix, completed):
    from scipy import stats as sp_stats

    log("  {:<30s} | {:>8s} | {:>8s} | {:>12s} | {:>12s} | {:>12s}".format(
        "Purpose", "Features", "Reduc%", "Acc% (mean)", "F1 (mean)", "AUC (mean)"))
    log("  " + "-" * 95)

    for purpose in MINIMIZATION_PURPOSES:
        accs, f1s, aucs, feats = [], [], [], []
        for seed in SEEDS_10:
            k = "{}_{}_{}".format(prefix, ds, purpose) + "_s{}".format(seed)
            if k in completed and "error" not in completed[k]:
                r = completed[k]
                accs.append(r["final_accuracy"] * 100)
                f1s.append(r["final_f1"])
                aucs.append(r["final_auc"])
                feats.append(r["kept_features"])
        if accs:
            n_orig = 13 if ds == "HD" else 11
            m_feat = int(np.mean(feats))
            log("  {:<30s} | {:>4d}/{:<3d} | {:>6.1f}% | {:>5.1f} +/- {:<4.1f} | {:>5.3f} +/- {:.3f} | {:>5.3f} +/- {:.3f}".format(
                purpose, m_feat, n_orig,
                (1 - np.mean(feats) / n_orig) * 100,
                np.mean(accs), np.std(accs),
                np.mean(f1s), np.std(f1s),
                np.mean(aucs), np.std(aucs)))


def _print_overhead_table(completed):
    permits, optouts, audits, trains, overheads = [], [], [], [], []
    for seed in SEEDS_10:
        k = "E3_timing_s{}".format(seed)
        if k in completed and "error" not in completed[k]:
            r = completed[k]
            a = r["averages"]
            permits.append(a["permit_validation_us"])
            optouts.append(a["optout_check_us"])
            audits.append(a["audit_logging_us"])
            trains.append(a["training_us"])
            overheads.append(a["governance_overhead_pct"])

    if permits:
        log("  {:>25s} | {:>12s} | {:>12s}".format(
            "Operation", "Mean (us)", "Std (us)"))
        log("  " + "-" * 55)
        log("  {:>25s} | {:>12.1f} | {:>12.1f}".format(
            "Permit validation", np.mean(permits), np.std(permits)))
        log("  {:>25s} | {:>12.1f} | {:>12.1f}".format(
            "Opt-out check (all pts)", np.mean(optouts), np.std(optouts)))
        log("  {:>25s} | {:>12.1f} | {:>12.1f}".format(
            "Audit logging", np.mean(audits), np.std(audits)))
        log("  {:>25s} | {:>12.1f} | {:>12.1f}".format(
            "GOVERNANCE TOTAL", np.mean(np.array(permits) + np.array(optouts) + np.array(audits)),
            np.std(np.array(permits) + np.array(optouts) + np.array(audits))))
        log("  {:>25s} | {:>12.1f} | {:>12.1f}".format(
            "FL training round", np.mean(trains), np.std(trains)))
        log("  " + "-" * 55)
        log("  {:>25s} | {:>11.3f}% | {:>11.3f}%".format(
            "Overhead percentage", np.mean(overheads), np.std(overheads)))
        n_patients = None
        for seed in SEEDS_10:
            k = "E3_timing_s{}".format(seed)
            if k in completed and "error" not in completed[k]:
                n_patients = completed[k]["num_patients"]
                break
        if n_patients:
            log("  (Patients checked per round: {})".format(n_patients))


def _print_optout_table(completed):
    log("  {:>10s} | {:>12s} | {:>12s} | {:>12s} | {:>12s}".format(
        "Opt-out%", "Acc% (mean)", "F1 (mean)", "AUC (mean)", "Delta Acc"))
    log("  " + "-" * 65)

    baseline_acc = None
    for rate in OPTOUT_RATES:
        accs, f1s, aucs = [], [], []
        for seed in SEEDS_10:
            k = "E4_optout_{}_s{}".format(int(rate * 100), seed)
            if k in completed and "error" not in completed[k]:
                r = completed[k]
                accs.append(r["final_accuracy"] * 100)
                f1s.append(r["final_f1"])
                aucs.append(r["final_auc"])
        if accs:
            m_acc = np.mean(accs)
            if baseline_acc is None:
                baseline_acc = m_acc
                delta_str = "ref"
            else:
                delta_str = "{:+.1f}pp".format(m_acc - baseline_acc)
            log("  {:>9.0f}% | {:>5.1f} +/- {:<4.1f} | {:>5.3f} +/- {:.3f} | {:>5.3f} +/- {:.3f} | {:>12s}".format(
                rate * 100,
                m_acc, np.std(accs),
                np.mean(f1s), np.std(f1s),
                np.mean(aucs), np.std(aucs),
                delta_str))


def _print_significance_tests(ds, prefix, completed):
    """Paired t-tests: each purpose vs scientific_research baseline."""
    from scipy import stats as sp_stats

    # Collect scientific_research baseline
    baseline_accs = {}
    for seed in SEEDS_10:
        k = "{}_{}_scientific_research_s{}".format(prefix, ds, seed)
        if k in completed and "error" not in completed[k]:
            baseline_accs[seed] = completed[k]["final_accuracy"] * 100

    if len(baseline_accs) < 3:
        return

    log("  {} paired t-tests (vs scientific_research, n={}):".format(
        ds, len(baseline_accs)))

    for purpose in MINIMIZATION_PURPOSES:
        if purpose == "scientific_research":
            continue

        purpose_accs = {}
        purpose_feats = {}
        for seed in SEEDS_10:
            k = "{}_{}_{}".format(prefix, ds, purpose) + "_s{}".format(seed)
            if k in completed and "error" not in completed[k]:
                purpose_accs[seed] = completed[k]["final_accuracy"] * 100
                purpose_feats[seed] = completed[k]["kept_features"]

        # Align seeds
        common_seeds = sorted(set(baseline_accs.keys()) & set(purpose_accs.keys()))
        if len(common_seeds) < 3:
            continue

        base_vals = np.array([baseline_accs[s] for s in common_seeds])
        purp_vals = np.array([purpose_accs[s] for s in common_seeds])
        diffs = purp_vals - base_vals

        t_stat, p_val = sp_stats.ttest_rel(purp_vals, base_vals)
        d_cohen = diffs.mean() / diffs.std(ddof=1) if diffs.std(ddof=1) > 0 else 0.0
        avg_feats = np.mean([purpose_feats[s] for s in common_seeds])

        sig = "*" if p_val < 0.05 else ""
        log("    {:<28s}: delta={:+5.1f}pp, t={:+.2f}, p={:.4f}{}, d={:.2f}, feat={:.0f}".format(
            purpose, diffs.mean(), t_stat, p_val, sig, d_cohen, avg_feats))


if __name__ == "__main__":
    main()

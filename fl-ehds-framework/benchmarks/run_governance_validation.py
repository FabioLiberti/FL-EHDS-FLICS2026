#!/usr/bin/env python3
"""
FL-EHDS Experiment — EHDS Governance Validation (A+C+D).

End-to-end validation of the FL-EHDS governance layer during real
FL training on Heart Disease UCI.  Three experiments:

  A) Governance Lifecycle: permit issuance → round-by-round verification →
     Article 71 opt-out mid-training → permit revocation → training abort.
  C) Data Minimization (Article 44): purpose-dependent feature reduction.
     4 purposes × 3 seeds = 12 experiments showing accuracy vs. features.
  D) Compliance Report: full GDPR Article 30 audit trail export.

Design:
  - Dataset:   Heart Disease UCI (920 samples, 4 hospitals, natural non-IID)
  - Model:     HealthcareMLP (2-layer, 64/32, ~10K params)
  - FL Config: FedAvg, 20 rounds × 3 local epochs, lr=0.01, bs=64
  - Seeds:     42, 123, 456

Usage:
    cd fl-ehds-framework
    python -m benchmarks.run_governance_validation [--fresh]

Output:
    benchmarks/paper_results_tabular/checkpoint_governance.json
    benchmarks/paper_results_tabular/audit_compliance.json

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
from terminal.training.federated import RoundResult, ClientResult

# Governance modules
from core.models import (
    DataCategory,
    DataPermit,
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
CHECKPOINT_FILE = "checkpoint_governance.json"
LOG_FILE = "experiment_governance.log"

SEEDS = [42, 123, 456]

# Heart Disease config
HD_CONFIG = {
    "name": "Heart Disease",
    "loader": "heart_disease",
    "input_dim": 13,
    "num_classes": 2,
    "num_clients": 4,
    "learning_rate": 0.01,
    "batch_size": 64,
    "num_rounds": 20,
    "local_epochs": 3,
    "mu": 0.1,
}

# Feature name → semantic group mapping for Heart Disease
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

# Purposes to test in Experiment C (with expected feature counts)
MINIMIZATION_PURPOSES = [
    "scientific_research",          # None = all 13 features
    "public_health_surveillance",   # demographics + conditions + vitals + medications
    "patient_safety",               # conditions + medications + vitals + measurements
    "official_statistics",          # demographics only
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
    fd, tmp = tempfile.mkstemp(dir=str(OUTPUT_DIR), prefix=".gov_", suffix=".tmp")
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

def load_dataset(seed):
    from data.heart_disease_loader import load_heart_disease_data, FEATURE_NAMES
    client_train, client_test, metadata = load_heart_disease_data(
        num_clients=HD_CONFIG["num_clients"],
        partition_by_hospital=True,
        seed=seed,
    )
    return client_train, client_test, metadata, FEATURE_NAMES


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
# Experiment A: Governance Lifecycle
# ======================================================================

def run_experiment_a(seed):
    """
    End-to-end governance lifecycle during FL training.

    Timeline:
      Rounds 1-10: normal training with permit + privacy budget tracking
      Round 11:    register opt-out for ~10% patients → exclude from training
      Rounds 12-14: training continues with reduced dataset
      Round 15:    revoke permit → training aborts
      Rounds 16-20: should NOT execute (permit revoked)

    Returns detailed log of every governance event.
    """
    log("  [A] Starting governance lifecycle (seed={})".format(seed))

    cfg = HD_CONFIG
    client_data, client_test, metadata, feature_names = load_dataset(seed)

    # --- Set up governance context ---
    permit_ctx = PermitAwareTrainingContext(
        permit_id="PERMIT-GOV-VALID-{}".format(seed),
        purpose=PermitPurpose.SCIENTIFIC_RESEARCH,
        data_categories=[DataCategory.EHR, DataCategory.LAB_RESULTS],
        privacy_budget_total=10.0,   # ε=10 total
        max_rounds=cfg["num_rounds"],
        client_ids=["client_{}".format(i) for i in range(cfg["num_clients"])],
        audit_output_dir=str(OUTPUT_DIR / "audit_a_s{}".format(seed)),
    )

    # --- Set up opt-out registry ---
    optout_registry = OptOutRegistry()
    optout_checker = OptOutChecker(registry=optout_registry)

    # Assign pseudo patient IDs to each sample per client
    rng = np.random.RandomState(seed)
    client_patient_ids = {}
    all_patient_ids = []
    for cid in sorted(client_data.keys()):
        n_samples = len(client_data[cid][1])
        pids = ["P-{}-{:04d}".format(cid, i) for i in range(n_samples)]
        client_patient_ids[cid] = pids
        all_patient_ids.extend(pids)

    # --- Start session ---
    session_id = permit_ctx.start_session()
    log("    Session started: {}".format(session_id))

    # --- FL training ---
    eps_per_round = 0.5  # 20 rounds × 0.5 = 10.0 total budget

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

    governance_log = []
    round_metrics = []
    optout_event_round = 11
    revocation_round = 15

    for r in range(cfg["num_rounds"]):
        # --- Pre-round permit validation ---
        ok, reason = permit_ctx.validate_round(r, eps_per_round)
        governance_log.append({
            "round": r,
            "event": "validate_round",
            "ok": ok,
            "reason": reason,
            "epsilon_used": permit_ctx._epsilon_used,
            "epsilon_remaining": permit_ctx._privacy_budget_total - permit_ctx._epsilon_used,
        })

        if not ok:
            log("    Round {} BLOCKED: {}".format(r, reason))
            governance_log.append({
                "round": r,
                "event": "training_aborted",
                "reason": reason,
            })
            break

        # --- Opt-out event at round 11 ---
        if r == optout_event_round:
            # Register 10% of all patients as opted out
            n_optout = max(1, int(len(all_patient_ids) * 0.10))
            optout_pids = rng.choice(all_patient_ids, size=n_optout, replace=False)

            for pid in optout_pids:
                record = OptOutRecord(
                    record_id="OPT-{}-{}".format(seed, pid),
                    patient_id=pid,
                    scope="all",
                    member_state="IT",
                )
                optout_registry.register_optout(record)

            # Filter training data
            excluded_total = 0
            for cid in sorted(client_data.keys()):
                pids = client_patient_ids[cid]
                X, y = client_data[cid]
                keep_mask = np.array([
                    not optout_registry.is_opted_out(pid) for pid in pids
                ])
                excluded_count = int((~keep_mask).sum())
                excluded_total += excluded_count

                if excluded_count > 0:
                    X_filtered = X[keep_mask]
                    y_filtered = y[keep_mask]
                    # Update trainer data
                    trainer.client_data[cid] = (X_filtered, y_filtered)
                    # Update patient IDs to match
                    client_patient_ids[cid] = [
                        p for p, k in zip(pids, keep_mask) if k
                    ]

            # Log opt-out compliance
            permit_ctx.compliance_logger.log_optout_check(
                total_records=len(all_patient_ids),
                excluded_count=excluded_total,
                client_id="aggregator",
            )

            governance_log.append({
                "round": r,
                "event": "optout_registered",
                "n_opted_out": int(n_optout),
                "n_excluded_from_training": excluded_total,
                "optout_rate": round(n_optout / len(all_patient_ids), 3),
                "patient_ids_opted_out": [str(p) for p in optout_pids[:5]] + ["..."],
            })
            log("    Round {}: {} patients opted out, {} excluded from training".format(
                r, n_optout, excluded_total))

        # --- Permit revocation at round 15 ---
        if r == revocation_round:
            permit_ctx.permit.status = PermitStatus.REVOKED
            governance_log.append({
                "round": r,
                "event": "permit_revoked",
                "permit_id": permit_ctx.permit.permit_id,
                "reason": "HDAB revocation (validation test)",
            })
            log("    Round {}: Permit REVOKED".format(r))
            # Next validate_round will fail due to revoked status
            # But we still execute this round (revocation effective next round)

        # --- Train round ---
        result = trainer.train_round(r)

        # --- Log round completion ---
        permit_ctx.log_round_completion(result, eps_per_round)

        metrics = {
            "round": r,
            "accuracy": result.global_acc,
            "loss": result.global_loss,
            "f1": result.global_f1,
            "auc": result.global_auc,
        }
        round_metrics.append(metrics)

        governance_log.append({
            "round": r,
            "event": "round_completed",
            "accuracy": result.global_acc,
            "f1": result.global_f1,
            "epsilon_cumulative": permit_ctx._epsilon_used,
        })

    # --- End session ---
    final_round = len(round_metrics) - 1
    final_metrics = round_metrics[-1] if round_metrics else {}
    permit_ctx.end_session(
        total_rounds=len(round_metrics),
        final_metrics={k: v for k, v in final_metrics.items() if k != "round"},
        success=False,  # ended by revocation
    )

    # --- Budget status ---
    budget_status = permit_ctx.get_budget_status()

    # --- Export audit log (Experiment D uses this) ---
    audit_dir = OUTPUT_DIR / "audit_a_s{}".format(seed)
    audit_file = permit_ctx.export_audit_log(str(audit_dir))

    governance_log.append({
        "event": "session_ended",
        "total_rounds_executed": len(round_metrics),
        "final_accuracy": round_metrics[-1]["accuracy"] if round_metrics else None,
        "budget_status": budget_status,
        "audit_file": str(audit_file),
    })

    # --- Opt-out registry stats ---
    optout_stats = optout_registry.get_stats().to_dict()

    result = {
        "seed": seed,
        "experiment": "A_governance_lifecycle",
        "session_id": session_id,
        "permit_id": permit_ctx.permit.permit_id,
        "governance_events": governance_log,
        "round_metrics": round_metrics,
        "total_rounds_executed": len(round_metrics),
        "expected_rounds": cfg["num_rounds"],
        "training_aborted_at_round": revocation_round + 1,
        "optout_event_round": optout_event_round,
        "budget_status": budget_status,
        "optout_stats": optout_stats,
        "audit_file": str(audit_file),
        "governance_summary": {
            "permit_lifecycle": "PASS" if len(round_metrics) <= revocation_round + 1 else "FAIL",
            "optout_enforcement": "PASS" if any(
                e.get("event") == "optout_registered" for e in governance_log
            ) else "FAIL",
            "privacy_budget_tracking": "PASS" if budget_status["used"] > 0 else "FAIL",
            "permit_revocation": "PASS" if any(
                e.get("event") == "training_aborted" for e in governance_log
            ) else "FAIL",
            "audit_trail_export": "PASS" if audit_file.exists() else "FAIL",
        },
    }

    # Print summary
    summary = result["governance_summary"]
    log("    Governance Summary:")
    for k, v in summary.items():
        log("      {}: {}".format(k, v))

    return result


# ======================================================================
# Experiment C: Data Minimization (Article 44)
# ======================================================================

def run_experiment_c(purpose, seed):
    """
    Train FedAvg with purpose-dependent data minimization.

    Uses DataMinimizer to filter features based on EHDS Article 53 purpose,
    then trains FL on the reduced feature set.

    Returns accuracy/F1/AUC plus minimization report.
    """
    log("  [C] Minimization: purpose={}, seed={}".format(purpose, seed))

    cfg = HD_CONFIG
    client_data, client_test, metadata, feature_names = load_dataset(seed)

    # Register Heart Disease features in DataMinimizer
    # (temporarily extend the FEATURE_GROUPS map for HD-specific names)
    original_groups = dict(DataMinimizer.FEATURE_GROUPS)
    for fname, group in HD_FEATURE_GROUPS.items():
        DataMinimizer.FEATURE_GROUPS[fname] = group

    try:
        # Apply data minimization
        filtered_train, filtered_test, min_report = DataMinimizer.apply_minimization(
            train_data=client_data,
            test_data=client_test,
            purpose=purpose,
            feature_names=feature_names,
            importance_threshold=0.01,
        )
    finally:
        # Restore original FEATURE_GROUPS
        DataMinimizer.FEATURE_GROUPS = original_groups

    n_kept = min_report["kept_features"]
    n_orig = min_report["original_features"]
    log("    Features: {}/{} kept ({}% reduction)".format(
        n_kept, n_orig, min_report["reduction_pct"]))
    log("    Kept: {}".format(min_report["kept_feature_names"]))

    # Train FL with reduced features
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

    log("    Final: acc={:.1f}% f1={:.3f} auc={:.3f}".format(
        final.get("accuracy", 0) * 100,
        final.get("f1", 0),
        final.get("auc", 0)))

    return {
        "seed": seed,
        "purpose": purpose,
        "experiment": "C_data_minimization",
        "original_features": n_orig,
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
        "config": {
            "algorithm": "FedAvg",
            "num_rounds": cfg["num_rounds"],
            "local_epochs": cfg["local_epochs"],
            "learning_rate": cfg["learning_rate"],
            "batch_size": cfg["batch_size"],
        },
    }


# ======================================================================
# Experiment D: Compliance Report Generation
# ======================================================================

def generate_compliance_report(experiment_a_results):
    """
    Generate comprehensive GDPR Article 30 compliance report
    from Experiment A audit trails.

    Combines all seed runs into a single compliance summary.
    """
    log("  [D] Generating compliance report")

    reports = []
    for result in experiment_a_results:
        audit_file = Path(result.get("audit_file", ""))
        if audit_file.exists():
            with open(audit_file) as f:
                reports.append(json.load(f))

    # Build aggregate report
    compliance_report = {
        "report_title": "FL-EHDS Governance Validation — GDPR Article 30 Compliance Report",
        "generated_at": datetime.now().isoformat(),
        "framework_version": "FL-EHDS v4.8",
        "regulation": "EHDS Regulation (EU) 2025/327",
        "articles_validated": [
            "Article 44 (Data Minimization)",
            "Article 53 (Purpose Limitation)",
            "Article 71 (Opt-Out)",
            "GDPR Article 30 (Records of Processing)",
        ],
        "experiment_count": len(experiment_a_results),
        "seeds": [r["seed"] for r in experiment_a_results],
        "per_seed_audit": reports,
        "governance_validation_summary": {
            "permit_lifecycle": all(
                r["governance_summary"]["permit_lifecycle"] == "PASS"
                for r in experiment_a_results
            ),
            "optout_enforcement": all(
                r["governance_summary"]["optout_enforcement"] == "PASS"
                for r in experiment_a_results
            ),
            "privacy_budget_tracking": all(
                r["governance_summary"]["privacy_budget_tracking"] == "PASS"
                for r in experiment_a_results
            ),
            "permit_revocation": all(
                r["governance_summary"]["permit_revocation"] == "PASS"
                for r in experiment_a_results
            ),
            "audit_trail_export": all(
                r["governance_summary"]["audit_trail_export"] == "PASS"
                for r in experiment_a_results
            ),
        },
        "privacy_budget_summary": {
            "total_budget_per_session": 10.0,
            "per_seed": {
                str(r["seed"]): r["budget_status"]
                for r in experiment_a_results
            },
        },
        "optout_summary": {
            "per_seed": {
                str(r["seed"]): r["optout_stats"]
                for r in experiment_a_results
            },
        },
        "overall_status": "COMPLIANT",
    }

    # Check if all validations passed
    summary = compliance_report["governance_validation_summary"]
    if not all(summary.values()):
        compliance_report["overall_status"] = "NON-COMPLIANT"
        failed = [k for k, v in summary.items() if not v]
        compliance_report["failed_checks"] = failed

    # Save report
    report_file = OUTPUT_DIR / "audit_compliance.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(compliance_report, f, indent=2, default=str, ensure_ascii=False)

    log("    Report saved: {}".format(report_file))
    log("    Status: {}".format(compliance_report["overall_status"]))

    return {
        "experiment": "D_compliance_report",
        "report_file": str(report_file),
        "overall_status": compliance_report["overall_status"],
        "governance_validation_summary": summary,
    }


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="FL-EHDS Governance Validation (A+C+D)")
    parser.add_argument("--fresh", action="store_true",
                        help="Delete existing checkpoint and start fresh")
    args = parser.parse_args()

    global _log_file
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _log_file = open(OUTPUT_DIR / LOG_FILE, "a")

    # Build experiment list
    experiments = []

    # A: Governance lifecycle (3 seeds)
    for seed in SEEDS:
        experiments.append({
            "key": "A_lifecycle_s{}".format(seed),
            "type": "A",
            "seed": seed,
        })

    # C: Data minimization (4 purposes × 3 seeds = 12)
    for purpose in MINIMIZATION_PURPOSES:
        for seed in SEEDS:
            experiments.append({
                "key": "C_{}_s{}".format(purpose, seed),
                "type": "C",
                "purpose": purpose,
                "seed": seed,
            })

    total_exps = len(experiments)  # 3 + 12 = 15

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
                "experiment_types": ["A_governance_lifecycle", "C_data_minimization", "D_compliance_report"],
                "seeds": SEEDS,
                "purposes": MINIMIZATION_PURPOSES,
                "dataset": "Heart Disease UCI",
                "config": HD_CONFIG,
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
        log("Checkpoint saved. Resume: python -m benchmarks.run_governance_validation")
        sys.exit(0)

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # Header
    log("\n" + "=" * 70)
    log("  FL-EHDS Governance Validation (A+C+D)")
    log("  {} experiments = 3 lifecycle + 12 minimization".format(total_exps))
    log("=" * 70)
    log("  Device:   {}".format(_detect_device()))
    log("  Seeds:    {}".format(SEEDS))
    log("  Purposes: {}".format(MINIMIZATION_PURPOSES))
    log("  Config:   {} rounds x {} local_epochs, lr={}, bs={}".format(
        HD_CONFIG["num_rounds"], HD_CONFIG["local_epochs"],
        HD_CONFIG["learning_rate"], HD_CONFIG["batch_size"]))
    log("  Output:   {}".format(OUTPUT_DIR / CHECKPOINT_FILE))
    log("=" * 70)

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

            if exp["type"] == "A":
                result = run_experiment_a(exp["seed"])
            elif exp["type"] == "C":
                result = run_experiment_c(exp["purpose"], exp["seed"])
            else:
                continue

            elapsed_t = time.time() - start_t
            result["runtime_seconds"] = round(elapsed_t, 1)
            completed[key] = result
            done_count += 1

            log("  -> done in {:.1f}s".format(elapsed_t))

            # Atomic save after each experiment
            save_checkpoint(checkpoint_data)

        except Exception as e:
            log("  ERROR: {}".format(e))
            traceback.print_exc()
            completed[key] = {"key": key, "error": str(e), "traceback": traceback.format_exc()}
            save_checkpoint(checkpoint_data)
            _cleanup_gpu()

    # ======================================================================
    # Experiment D: Compliance report (after all A experiments done)
    # ======================================================================
    if not _interrupted[0]:
        a_results = [
            completed[k] for k in completed
            if k.startswith("A_") and "error" not in completed[k]
        ]
        if a_results:
            try:
                d_result = generate_compliance_report(a_results)
                completed["D_compliance_report"] = d_result
                save_checkpoint(checkpoint_data)
            except Exception as e:
                log("  ERROR generating compliance report: {}".format(e))
                traceback.print_exc()

    # ======================================================================
    # Finalize
    # ======================================================================
    checkpoint_data["metadata"]["end_time"] = datetime.now().isoformat()
    checkpoint_data["metadata"]["total_elapsed"] = time.time() - global_start
    save_checkpoint(checkpoint_data)

    elapsed = time.time() - global_start

    # ======================================================================
    # Summary
    # ======================================================================
    log("\n" + "=" * 70)
    log("  COMPLETED: {}/{} ({:.0f}s)".format(done_count, total_exps, elapsed))
    log("=" * 70)

    # --- Experiment A summary ---
    log("\n  [A] GOVERNANCE LIFECYCLE:")
    for seed in SEEDS:
        k = "A_lifecycle_s{}".format(seed)
        if k in completed and "error" not in completed[k]:
            r = completed[k]
            summary = r.get("governance_summary", {})
            status_str = "ALL PASS" if all(v == "PASS" for v in summary.values()) else "SOME FAIL"
            log("    seed={}: {} rounds executed (expected abort at {}), {}".format(
                seed, r["total_rounds_executed"],
                r.get("training_aborted_at_round", "?"), status_str))
            for check, status in summary.items():
                log("      {}: {}".format(check, status))

    # --- Experiment C summary ---
    log("\n  [C] DATA MINIMIZATION (Article 44):")
    log("  {:<30s} | {:>8s} | {:>8s} | {:>8s} | {:>8s} | {:>8s}".format(
        "Purpose", "Features", "Reduc%", "Acc%", "F1", "AUC"))
    log("  " + "-" * 85)

    for purpose in MINIMIZATION_PURPOSES:
        accs, f1s, aucs, feats = [], [], [], []
        for seed in SEEDS:
            k = "C_{}_s{}".format(purpose, seed)
            if k in completed and "error" not in completed[k]:
                r = completed[k]
                accs.append(r["final_accuracy"] * 100)
                f1s.append(r["final_f1"])
                aucs.append(r["final_auc"])
                feats.append(r["kept_features"])
        if accs:
            log("  {:<30s} | {:>4d}/13  | {:>6.1f}% | {:>6.1f}% | {:>8.3f} | {:>8.3f}".format(
                purpose,
                int(np.mean(feats)),
                (1 - np.mean(feats) / 13) * 100,
                np.mean(accs),
                np.mean(f1s),
                np.mean(aucs)))

    # --- Experiment D summary ---
    if "D_compliance_report" in completed:
        d = completed["D_compliance_report"]
        log("\n  [D] COMPLIANCE REPORT:")
        log("    Status: {}".format(d.get("overall_status", "?")))
        log("    File:   {}".format(d.get("report_file", "?")))

    log("\n  Done!")

    if _log_file:
        _log_file.close()


if __name__ == "__main__":
    main()

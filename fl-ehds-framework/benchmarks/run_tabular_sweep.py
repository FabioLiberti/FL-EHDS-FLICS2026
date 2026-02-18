#!/usr/bin/env python3
"""
FL-EHDS Tabular Sweep — Multi-phase hyperparameter exploration.

Phase 1: Heterogeneity sweep  (alpha = 0.1, 0.3, 0.5, 1.0, IID)  ~60 min
Phase 2: Client scaling        (varying num_clients per dataset)    ~40 min
Phase 3: Learning rate sensitivity (lr sweep, top-3 algos)         ~20 min

Each phase writes to an ISOLATED checkpoint file.
Results are combinable with baseline (checkpoint_tabular.json).

Usage:
    cd fl-ehds-framework
    python -m benchmarks.run_tabular_sweep --phase all [--resume]
    python -m benchmarks.run_tabular_sweep --phase 1   [--resume]
    python -m benchmarks.run_tabular_sweep --phase 2   [--resume]
    python -m benchmarks.run_tabular_sweep --phase 3   [--resume]
    python -m benchmarks.run_tabular_sweep --phase all --quick  (validation)

Output: benchmarks/paper_results_tabular/checkpoint_sweep.json

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
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from collections import defaultdict

import numpy as np

FRAMEWORK_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(FRAMEWORK_DIR))

import torch
import torch.nn as nn

from terminal.fl_trainer import (
    FederatedTrainer,
    RoundResult,
    HealthcareMLP,
    _detect_device,
)
from data.ptbxl_loader import load_ptbxl_data
from data.cardiovascular_loader import load_cardiovascular_data
from data.breast_cancer_loader import load_breast_cancer_data

# ======================================================================
# Constants
# ======================================================================

OUTPUT_DIR = FRAMEWORK_DIR / "benchmarks" / "paper_results_tabular"
CHECKPOINT_FILE = "checkpoint_sweep.json"
LOG_FILE = "experiment_sweep.log"

ALGORITHMS = ["FedAvg", "FedProx", "Ditto", "FedLC", "FedExP", "FedLESAM", "HPFL"]
SEEDS = [42, 123, 456]

# Per-dataset BASELINE training configs (same as run_tabular_optimized.py)
BASELINE_CONFIGS = {
    "PTB_XL": dict(
        input_dim=9, num_classes=5, short="PX",
        learning_rate=0.005, batch_size=64, num_rounds=30,
        local_epochs=3, mu=0.1,
        es=dict(enabled=True, patience=6, min_delta=0.003, min_rounds=12),
    ),
    "Cardiovascular": dict(
        input_dim=11, num_classes=2, short="CV",
        learning_rate=0.01, batch_size=64, num_rounds=25,
        local_epochs=3, mu=0.1,
        es=dict(enabled=True, patience=6, min_delta=0.003, min_rounds=10),
    ),
    "Breast_Cancer": dict(
        input_dim=30, num_classes=2, short="BC",
        learning_rate=0.001, batch_size=16, num_rounds=40,
        local_epochs=1, mu=0.1,
        es=dict(enabled=True, patience=8, min_delta=0.005, min_rounds=15),
    ),
}

# ======================================================================
# Phase 1: Heterogeneity Sweep
# ======================================================================

PHASE1_ALPHAS = [0.1, 0.3, 0.5, 1.0]
PHASE1_IID = True  # Also run IID

def build_phase1_experiments(seeds, quick=False):
    """Build heterogeneity sweep experiment list."""
    exps = []
    alphas = [0.5] if quick else PHASE1_ALPHAS
    algos = ALGORITHMS[:3] if quick else ALGORITHMS

    for ds in ["PTB_XL", "Cardiovascular", "Breast_Cancer"]:
        # Dirichlet sweeps
        for alpha in alphas:
            for algo in algos:
                for seed in seeds:
                    exps.append({
                        "phase": 1,
                        "dataset": ds,
                        "algorithm": algo,
                        "seed": seed,
                        "variation": f"alpha={alpha}",
                        "partition": {"is_iid": False, "alpha": alpha},
                        "num_clients": 5 if ds != "Breast_Cancer" else 3,
                    })

        # IID runs (skip in quick mode)
        if not quick and PHASE1_IID:
            for algo in algos:
                for seed in seeds:
                    exps.append({
                        "phase": 1,
                        "dataset": ds,
                        "algorithm": algo,
                        "seed": seed,
                        "variation": "IID",
                        "partition": {"is_iid": True},
                        "num_clients": 5 if ds != "Breast_Cancer" else 3,
                    })

    # PTB-XL: also site-based partitioning for comparison
    if not quick:
        for algo in algos:
            for seed in seeds:
                exps.append({
                    "phase": 1,
                    "dataset": "PTB_XL",
                    "algorithm": algo,
                    "seed": seed,
                    "variation": "site-based",
                    "partition": {"partition_by_site": True},
                    "num_clients": 5,
                })

    return exps


# ======================================================================
# Phase 2: Client Scaling
# ======================================================================

PHASE2_CLIENTS = {
    "PTB_XL": [3, 5, 10, 20],
    "Cardiovascular": [3, 5, 8, 10],
    "Breast_Cancer": [2, 3, 5],
}

def build_phase2_experiments(seeds, quick=False):
    """Build client scaling experiment list."""
    exps = []
    algos = ALGORITHMS[:3] if quick else ALGORITHMS

    for ds, client_values in PHASE2_CLIENTS.items():
        values = client_values[:2] if quick else client_values
        for nc in values:
            for algo in algos:
                for seed in seeds:
                    partition = {"partition_by_site": True} if ds == "PTB_XL" else {"is_iid": False, "alpha": 0.5}
                    exps.append({
                        "phase": 2,
                        "dataset": ds,
                        "algorithm": algo,
                        "seed": seed,
                        "variation": f"clients={nc}",
                        "partition": partition,
                        "num_clients": nc,
                    })
    return exps


# ======================================================================
# Phase 3: Learning Rate Sensitivity
# ======================================================================

PHASE3_LRS = [0.0005, 0.001, 0.005, 0.01]
PHASE3_ALGOS = ["FedAvg", "Ditto", "HPFL"]  # Top 3 only

def build_phase3_experiments(seeds, quick=False):
    """Build LR sensitivity experiment list."""
    exps = []
    lrs = PHASE3_LRS[:2] if quick else PHASE3_LRS
    algos = PHASE3_ALGOS[:2] if quick else PHASE3_ALGOS

    for ds in BASELINE_CONFIGS:
        for lr in lrs:
            for algo in algos:
                for seed in seeds:
                    partition = {"partition_by_site": True} if ds == "PTB_XL" else {"is_iid": False, "alpha": 0.5}
                    nc = 5 if ds != "Breast_Cancer" else 3
                    exps.append({
                        "phase": 3,
                        "dataset": ds,
                        "algorithm": algo,
                        "seed": seed,
                        "variation": f"lr={lr}",
                        "partition": partition,
                        "num_clients": nc,
                        "lr_override": lr,
                    })
    return exps


# ======================================================================
# Logging
# ======================================================================

_log_file = None

def log(msg: str, also_print: bool = True):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
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

def save_checkpoint(data: Dict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / CHECKPOINT_FILE
    bak = OUTPUT_DIR / (CHECKPOINT_FILE + ".bak")
    data["metadata"]["last_save"] = datetime.now().isoformat()

    fd, tmp = tempfile.mkstemp(dir=str(OUTPUT_DIR), prefix=".sweep_", suffix=".tmp")
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


def load_checkpoint() -> Optional[Dict]:
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
    import gc
    gc.collect()


# ======================================================================
# Early Stopping
# ======================================================================

class EarlyStoppingMonitor:
    def __init__(self, patience=6, min_delta=0.003, min_rounds=10, metric="accuracy"):
        self.patience = patience
        self.min_delta = min_delta
        self.min_rounds = min_rounds
        self.metric = metric
        self.best_value = -float('inf')
        self.best_round = 0
        self.counter = 0

    def check(self, round_num, metrics):
        value = metrics.get(self.metric, 0)
        if value > self.best_value + self.min_delta:
            self.best_value = value
            self.best_round = round_num
            self.counter = 0
        else:
            self.counter += 1
        return round_num >= self.min_rounds and self.counter >= self.patience


# ======================================================================
# Per-client evaluation & fairness
# ======================================================================

def _evaluate_per_client(trainer) -> Dict[str, float]:
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
                out = model(X_t[i:i+64])
                correct += (out.argmax(1) == y_t[i:i+64]).sum().item()
                total += len(y_t[i:i+64])
            per_client[str(cid)] = correct / total if total > 0 else 0.0

    if is_hpfl:
        for n, p in model.named_parameters():
            if n in trainer._hpfl_classifier_names:
                p.data.copy_(saved_cls[n])
    return per_client


def _compute_fairness(per_client_acc: Dict[str, float]) -> Dict[str, float]:
    accs = list(per_client_acc.values())
    if not accs:
        return {}
    jain = (sum(accs) ** 2) / (len(accs) * sum(a ** 2 for a in accs)) if accs else 0
    sorted_a = sorted(accs)
    n = len(sorted_a)
    cumsum = np.cumsum(sorted_a)
    gini = (2 * sum((i + 1) * v for i, v in enumerate(sorted_a))) / (n * cumsum[-1]) - (n + 1) / n
    gini = max(0, gini)
    return {
        "mean": round(float(np.mean(accs)), 4),
        "std": round(float(np.std(accs)), 4),
        "min": round(float(min(accs)), 4),
        "max": round(float(max(accs)), 4),
        "jain_index": round(float(jain), 4),
        "gini": round(float(gini), 4),
    }


# ======================================================================
# Data loading
# ======================================================================

def load_dataset(ds_name: str, num_clients: int, seed: int, partition: Dict):
    """Load dataset with specified partition strategy."""
    if ds_name == "PTB_XL":
        if partition.get("partition_by_site"):
            return load_ptbxl_data(num_clients=num_clients, seed=seed, partition_by_site=True)
        elif partition.get("is_iid"):
            return load_ptbxl_data(num_clients=num_clients, seed=seed, partition_by_site=False, is_iid=True)
        else:
            return load_ptbxl_data(
                num_clients=num_clients, seed=seed,
                partition_by_site=False, is_iid=False, alpha=partition.get("alpha", 0.5),
            )
    elif ds_name == "Cardiovascular":
        return load_cardiovascular_data(
            num_clients=num_clients, seed=seed,
            is_iid=partition.get("is_iid", False),
            alpha=partition.get("alpha", 0.5),
        )
    elif ds_name == "Breast_Cancer":
        return load_breast_cancer_data(
            num_clients=num_clients, seed=seed,
            is_iid=partition.get("is_iid", False),
            alpha=partition.get("alpha", 0.5),
        )
    else:
        raise ValueError(f"Unknown dataset: {ds_name}")


# ======================================================================
# Training
# ======================================================================

def run_single_experiment(exp: Dict, quick: bool = False) -> Dict[str, Any]:
    """Run a single sweep experiment."""
    start = time.time()
    ds_name = exp["dataset"]
    algo = exp["algorithm"]
    seed = exp["seed"]
    nc = exp["num_clients"]
    cfg = BASELINE_CONFIGS[ds_name]

    lr = exp.get("lr_override", cfg["learning_rate"])
    num_rounds = 5 if quick else cfg["num_rounds"]
    local_epochs = 1 if quick else cfg["local_epochs"]

    client_data, client_test_data, metadata = load_dataset(ds_name, nc, seed, exp["partition"])

    trainer = FederatedTrainer(
        num_clients=nc,
        algorithm=algo,
        local_epochs=local_epochs,
        batch_size=cfg["batch_size"],
        learning_rate=lr,
        mu=cfg["mu"],
        seed=seed,
        external_data=client_data,
        external_test_data=client_test_data,
        input_dim=cfg["input_dim"],
        num_classes=cfg["num_classes"],
    )

    es_cfg = cfg["es"]
    es = EarlyStoppingMonitor(
        patience=es_cfg["patience"], min_delta=es_cfg["min_delta"],
        min_rounds=es_cfg["min_rounds"],
    ) if es_cfg.get("enabled") and not quick else None

    history = []
    best_acc = 0.0
    best_round = 0

    for r in range(num_rounds):
        rr = trainer.train_round(r)
        metrics = {
            "round": r + 1, "accuracy": rr.global_acc, "loss": rr.global_loss,
            "f1": rr.global_f1, "precision": rr.global_precision,
            "recall": rr.global_recall, "auc": rr.global_auc,
        }
        history.append(metrics)
        if rr.global_acc > best_acc:
            best_acc = rr.global_acc
            best_round = r + 1
        if es and es.check(r + 1, {"accuracy": rr.global_acc}):
            break

    per_client_acc = _evaluate_per_client(trainer)
    fairness = _compute_fairness(per_client_acc)
    elapsed = time.time() - start

    result = {
        "phase": exp["phase"],
        "dataset": ds_name,
        "algorithm": algo,
        "seed": seed,
        "variation": exp["variation"],
        "num_clients": nc,
        "learning_rate": lr,
        "partition": exp["partition"],
        "history": history,
        "final_metrics": history[-1] if history else {},
        "per_client_acc": per_client_acc,
        "fairness": fairness,
        "runtime_seconds": round(elapsed, 1),
        "stopped_early": es is not None and es.counter >= es.patience,
        "actual_rounds": len(history),
        "best_metrics": {"accuracy": best_acc, "round": best_round},
        "best_round": best_round,
        "dataset_metadata": {
            "total_samples": metadata.get("total_samples"),
            "num_features": metadata.get("num_features"),
            "num_classes": metadata.get("num_classes"),
            "partition_method": metadata.get("partition_method"),
            "samples_per_client": metadata.get("samples_per_client"),
        },
    }

    _cleanup_gpu()
    return result


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="FL-EHDS Tabular Sweep")
    parser.add_argument("--phase", type=str, default="all",
                        choices=["1", "2", "3", "all"],
                        help="Phase to run: 1=heterogeneity, 2=client_scaling, 3=lr_sensitivity, all")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--quick", action="store_true", help="Quick validation mode")
    args = parser.parse_args()

    global _log_file
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _log_file = open(OUTPUT_DIR / LOG_FILE, "a")

    seeds = [42] if args.quick else SEEDS

    # Build experiment list for requested phases
    experiments = []
    phases_to_run = [1, 2, 3] if args.phase == "all" else [int(args.phase)]

    if 1 in phases_to_run:
        experiments.extend(build_phase1_experiments(seeds, args.quick))
    if 2 in phases_to_run:
        experiments.extend(build_phase2_experiments(seeds, args.quick))
    if 3 in phases_to_run:
        experiments.extend(build_phase3_experiments(seeds, args.quick))

    total_exps = len(experiments)

    # Load or create checkpoint
    checkpoint_data = None
    if args.resume:
        checkpoint_data = load_checkpoint()
        if checkpoint_data:
            done = len(checkpoint_data.get("completed", {}))
            log(f"RESUMED: {done} experiments already completed")

    if checkpoint_data is None:
        checkpoint_data = {
            "completed": {},
            "metadata": {
                "total_experiments": total_exps,
                "phases": phases_to_run,
                "algorithms": ALGORITHMS,
                "seeds": seeds,
                "start_time": datetime.now().isoformat(),
                "last_save": None,
                "version": "sweep_v1",
            }
        }

    # SIGINT handler
    _interrupted = [False]

    def _signal_handler(signum, frame):
        if _interrupted[0]:
            sys.exit(1)
        _interrupted[0] = True
        done = len(checkpoint_data.get("completed", {}))
        save_checkpoint(checkpoint_data)
        print(f"\n  Checkpoint salvato: {done}/{total_exps}")
        print(f"  Per riprendere: python -m benchmarks.run_tabular_sweep --phase {args.phase} --resume")
        sys.exit(0)

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # Header
    mode = "QUICK" if args.quick else "FULL"
    log(f"\n{'='*66}")
    log(f"  FL-EHDS Tabular Sweep ({mode})")
    log(f"  Phases: {phases_to_run}")
    log(f"  Total experiments: {total_exps}")
    log(f"  Seeds: {seeds}")
    log(f"  Device: {_detect_device(None)}")
    log(f"{'='*66}")

    # Phase breakdown
    phase_counts = defaultdict(int)
    for exp in experiments:
        phase_counts[exp["phase"]] += 1
    for p, c in sorted(phase_counts.items()):
        phase_names = {1: "Heterogeneity Sweep", 2: "Client Scaling", 3: "LR Sensitivity"}
        log(f"  Phase {p} ({phase_names[p]}): {c} experiments")
    log("")

    # Run experiments
    global_start = time.time()
    completed_count = len(checkpoint_data.get("completed", {}))
    current_phase = None

    for exp_idx, exp in enumerate(experiments, 1):
        key = f"P{exp['phase']}_{exp['dataset']}_{exp['algorithm']}_{exp['seed']}_{exp['variation']}"

        if key in checkpoint_data.get("completed", {}):
            continue

        if _interrupted[0]:
            break

        # Phase header
        if exp["phase"] != current_phase:
            current_phase = exp["phase"]
            phase_names = {1: "HETEROGENEITY SWEEP", 2: "CLIENT SCALING", 3: "LR SENSITIVITY"}
            log(f"\n{'='*50}")
            log(f"  PHASE {current_phase}: {phase_names[current_phase]}")
            log(f"{'='*50}")

        log(f"  [{exp_idx}/{total_exps}] {exp['dataset']} | {exp['algorithm']} | "
            f"s{exp['seed']} | {exp['variation']}")

        try:
            result = run_single_experiment(exp, quick=args.quick)

            checkpoint_data["completed"][key] = result
            completed_count += 1
            checkpoint_data["metadata"]["done"] = completed_count
            save_checkpoint(checkpoint_data)

            best_acc = result["best_metrics"]["accuracy"]
            elapsed = time.time() - global_start
            avg = elapsed / completed_count
            eta = str(timedelta(seconds=int((total_exps - completed_count) * avg)))

            log(f"    -> Acc={best_acc:.1%} | {result['runtime_seconds']:.0f}s | "
                f"[{completed_count}/{total_exps}] ETA ~{eta}")

        except Exception as e:
            log(f"    -> ERROR: {e}")
            log(traceback.format_exc(), also_print=False)
            checkpoint_data["completed"][key] = {
                "phase": exp["phase"], "dataset": exp["dataset"],
                "algorithm": exp["algorithm"], "seed": exp["seed"],
                "variation": exp["variation"], "error": str(e),
            }
            save_checkpoint(checkpoint_data)
            continue

    # Final summary
    elapsed_total = time.time() - global_start
    log(f"\n{'='*66}")
    log(f"  SWEEP COMPLETED: {completed_count}/{total_exps}")
    log(f"  Total time: {str(timedelta(seconds=int(elapsed_total)))}")
    log(f"  Checkpoint: {OUTPUT_DIR / CHECKPOINT_FILE}")
    log(f"{'='*66}")

    # Phase summary
    phase_results = defaultdict(lambda: {"done": 0, "errors": 0, "best_acc": 0, "best_key": ""})
    for key, res in checkpoint_data.get("completed", {}).items():
        p = res.get("phase", 0)
        if "error" in res:
            phase_results[p]["errors"] += 1
        else:
            phase_results[p]["done"] += 1
            acc = res.get("best_metrics", {}).get("accuracy", 0)
            if acc > phase_results[p]["best_acc"]:
                phase_results[p]["best_acc"] = acc
                phase_results[p]["best_key"] = f"{res['dataset']}/{res['algorithm']}"

    for p in sorted(phase_results):
        info = phase_results[p]
        log(f"  Phase {p}: {info['done']} OK, {info['errors']} errors, "
            f"best={info['best_acc']:.1%} ({info['best_key']})")

    checkpoint_data["metadata"]["end_time"] = datetime.now().isoformat()
    checkpoint_data["metadata"]["total_elapsed"] = elapsed_total
    save_checkpoint(checkpoint_data)

    if _log_file:
        _log_file.close()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
FL-EHDS Paper Experiments V2 — 7 Selected Algorithms (2017-2025).

Algorithms:
  FedAvg (2017), FedProx (2020), Ditto (2021), FedLC (2022),
  FedExP (2023), FedLESAM (2024), HPFL (2025)

Datasets:
  Imaging: Brain_Tumor (4-class), chest_xray (2-class), Skin_Cancer (2-class)
  Tabular: Diabetes (2-class), Heart_Disease (2-class)

Features:
  - Atomic checkpoint after EVERY completed experiment
  - SIGINT/SIGTERM handler: graceful save on Ctrl+C
  - Clear progress reporting with ETA
  - --resume to continue from last checkpoint
  - --quick for fast validation (1 seed, 5 rounds)

Usage:
    cd fl-ehds-framework
    python -m benchmarks.run_full_experiments [--resume] [--quick]

Output: benchmarks/paper_results/checkpoint_v2.json

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
from typing import Dict, List, Optional, Any, Callable
from copy import deepcopy

import numpy as np

# Add parent to path
FRAMEWORK_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(FRAMEWORK_DIR))

import torch
import torch.nn as nn

from terminal.fl_trainer import (
    FederatedTrainer,
    ImageFederatedTrainer,
    RoundResult,
    HealthcareMLP,
    _detect_device,
)
from data.diabetes_loader import load_diabetes_data
from data.heart_disease_loader import load_heart_disease_data

# ======================================================================
# Configuration V2
# ======================================================================

ALGORITHMS = ["FedAvg", "FedProx", "Ditto", "FedLC", "FedExP", "FedLESAM", "HPFL"]

ALGO_COLORS = {
    "FedAvg": "#2196F3",    # blue
    "FedProx": "#FF5722",   # deep orange
    "Ditto": "#4CAF50",     # green
    "FedLC": "#E91E63",     # pink
    "FedExP": "#9C27B0",    # purple
    "FedLESAM": "#FF9800",  # orange
    "HPFL": "#00BCD4",      # cyan
}

ALGO_INFO = {
    "FedAvg":   {"year": 2017, "venue": "AISTATS",        "role": "Baseline"},
    "FedProx":  {"year": 2020, "venue": "MLSys",          "role": "Non-IID robustness"},
    "Ditto":    {"year": 2021, "venue": "ICML",           "role": "Personalization"},
    "FedLC":    {"year": 2022, "venue": "ICML",           "role": "Label skew"},
    "FedExP":   {"year": 2023, "venue": "ICLR Spotlight", "role": "Fast convergence"},
    "FedLESAM": {"year": 2024, "venue": "ICML Spotlight", "role": "Global generalization"},
    "HPFL":     {"year": 2025, "venue": "ICLR",           "role": "Hot-pluggable nodes"},
}

SEEDS = [42, 123, 456]
OUTPUT_DIR = FRAMEWORK_DIR / "benchmarks" / "paper_results"
CHECKPOINT_FILE = "checkpoint_v2.json"
LOG_FILE = "experiment_v2.log"

IMAGING_DATASETS = {
    "Brain_Tumor": {
        "data_dir": str(FRAMEWORK_DIR / "data" / "Brain_Tumor"),
        "num_classes": 4,
        "short": "BT",
    },
    "chest_xray": {
        "data_dir": str(FRAMEWORK_DIR / "data" / "chest_xray"),
        "num_classes": 2,
        "short": "CX",
    },
    "Skin_Cancer": {
        "data_dir": str(FRAMEWORK_DIR / "data" / "Skin Cancer"),
        "num_classes": 2,
        "short": "SC",
    },
}

TABULAR_DATASETS = {
    "Diabetes": {"num_features": 22, "short": "DM"},
    "Heart_Disease": {"num_features": 13, "short": "HD"},
}

# V2 improved configurations
V2_IMAGING_CONFIG = dict(
    num_clients=5,
    num_rounds=25,
    local_epochs=3,
    batch_size=32,
    learning_rate=0.001,
    model_type="resnet18",
    is_iid=False,
    alpha=0.5,
    freeze_backbone=False,
    freeze_level=1,  # partial freeze: only conv1+bn1
    use_fedbn=True,
    use_class_weights=True,
    use_amp=True,
    mu=0.1,
)

V2_TABULAR_CONFIG = dict(
    num_rounds=30,
    local_epochs=5,
    batch_size=32,
    learning_rate=0.01,
    mu=0.1,
)

V2_EARLY_STOPPING = dict(
    enabled=True,
    patience=6,
    min_delta=0.003,
    min_rounds=10,
    metric="accuracy",
)

# Per-dataset overrides
DATASET_OVERRIDES = {
    "Brain_Tumor": {"learning_rate": 0.0005},
}


# ======================================================================
# Logging
# ======================================================================

_log_file = None

def log(msg: str, also_print: bool = True):
    """Log to file and optionally print."""
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
# Checkpoint / Resume
# ======================================================================

def save_checkpoint(data: Dict) -> None:
    """Atomic checkpoint save."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / CHECKPOINT_FILE
    bak_path = OUTPUT_DIR / (CHECKPOINT_FILE + ".bak")

    data["metadata"]["last_save"] = datetime.now().isoformat()

    fd, tmp_path = tempfile.mkstemp(
        dir=str(OUTPUT_DIR),
        prefix=".checkpoint_v2_",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2, default=str)
            f.flush()
            os.fsync(f.fileno())
        if path.exists():
            shutil.copy2(str(path), str(bak_path))
        os.replace(tmp_path, str(path))
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def load_checkpoint() -> Optional[Dict]:
    """Load checkpoint with fallback to backup."""
    path = OUTPUT_DIR / CHECKPOINT_FILE
    bak_path = OUTPUT_DIR / (CHECKPOINT_FILE + ".bak")

    for p in [path, bak_path]:
        if p.exists():
            try:
                with open(p, "r") as f:
                    data = json.load(f)
                if p == bak_path:
                    log(f"WARNING: Loaded from backup {p.name}")
                return data
            except (json.JSONDecodeError, IOError):
                if p == path:
                    log(f"WARNING: Checkpoint corrupt, trying backup...")
                continue
    return None


# ======================================================================
# GPU cleanup
# ======================================================================

def _cleanup_gpu():
    """Free GPU/MPS memory between experiments."""
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
        self.best_metrics = None

    def check(self, round_num, metrics):
        value = metrics.get(self.metric, 0)
        if value > self.best_value + self.min_delta:
            self.best_value = value
            self.best_round = round_num
            self.counter = 0
            self.best_metrics = metrics.copy()
        else:
            self.counter += 1
        if round_num < self.min_rounds:
            return False
        return self.counter >= self.patience


# ======================================================================
# Per-client evaluation and fairness
# ======================================================================

def _evaluate_per_client(trainer) -> Dict[str, float]:
    """Evaluate global model on each client's test data separately."""
    model = trainer.global_model
    model.eval()
    per_client = {}

    # HPFL: need to swap classifier per client
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
            bs = 64
            for i in range(0, len(y_t), bs):
                out = model(X_t[i:i+bs])
                preds = out.argmax(dim=1)
                correct += (preds == y_t[i:i+bs]).sum().item()
                total += len(y_t[i:i+bs])
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
# Training wrappers
# ======================================================================

def run_single_imaging(
    dataset_name: str, data_dir: str, algorithm: str, seed: int,
    config: Dict, es_config: Dict,
    exp_idx: int, total_exps: int,
) -> Dict[str, Any]:
    """Run a single imaging FL experiment."""
    start = time.time()
    num_rounds = config["num_rounds"]

    # Merge per-dataset overrides
    cfg = {**config}
    if dataset_name in DATASET_OVERRIDES:
        cfg.update(DATASET_OVERRIDES[dataset_name])

    trainer = ImageFederatedTrainer(
        data_dir=data_dir,
        num_clients=cfg["num_clients"],
        algorithm=algorithm,
        local_epochs=cfg["local_epochs"],
        batch_size=cfg["batch_size"],
        learning_rate=cfg["learning_rate"],
        is_iid=cfg["is_iid"],
        alpha=cfg["alpha"],
        mu=cfg.get("mu", 0.1),
        seed=seed,
        model_type=cfg["model_type"],
        freeze_backbone=cfg.get("freeze_backbone", False),
        freeze_level=cfg.get("freeze_level"),
        use_fedbn=cfg.get("use_fedbn", False),
        use_class_weights=cfg.get("use_class_weights", True),
        use_amp=cfg.get("use_amp", True),
    )
    trainer.num_rounds = num_rounds

    es = EarlyStoppingMonitor(**{k: v for k, v in es_config.items() if k != "enabled"}) if es_config.get("enabled") else None

    history = []
    best_acc = 0.0
    best_round = 0

    for r in range(num_rounds):
        rr = trainer.train_round(r)
        metrics = {
            "round": r + 1,
            "accuracy": rr.global_acc,
            "loss": rr.global_loss,
            "f1": rr.global_f1,
            "precision": rr.global_precision,
            "recall": rr.global_recall,
            "auc": rr.global_auc,
        }
        history.append(metrics)

        if rr.global_acc > best_acc:
            best_acc = rr.global_acc
            best_round = r + 1

        # Progress per round
        log(f"EXP {exp_idx}/{total_exps} | {dataset_name} | {algorithm} | s{seed} | "
            f"R {r+1}/{num_rounds} | Acc:{rr.global_acc:.1%} | Best:{best_acc:.1%}(r{best_round})")

        if es and es.check(r + 1, {"accuracy": rr.global_acc, "f1": rr.global_f1, "loss": rr.global_loss}):
            log(f"  Early stop at round {r+1} (best={best_acc:.1%} at r{best_round})")
            break

    # Per-client evaluation
    per_client_acc = _evaluate_per_client(trainer)
    fairness = _compute_fairness(per_client_acc)

    elapsed = time.time() - start

    result = {
        "dataset": dataset_name,
        "algorithm": algorithm,
        "seed": seed,
        "history": history,
        "final_metrics": history[-1] if history else {},
        "per_client_acc": per_client_acc,
        "fairness": fairness,
        "runtime_seconds": round(elapsed, 1),
        "config": cfg,
        "stopped_early": es is not None and es.counter >= es.patience,
        "actual_rounds": len(history),
        "best_metrics": {"accuracy": best_acc, "round": best_round},
        "best_round": best_round,
    }

    _cleanup_gpu()
    return result


def run_single_tabular(
    dataset_name: str, algorithm: str, seed: int,
    config: Dict, es_config: Dict,
    exp_idx: int, total_exps: int,
) -> Dict[str, Any]:
    """Run a single tabular FL experiment."""
    start = time.time()
    num_rounds = config["num_rounds"]
    num_clients = 5

    # Load data
    if dataset_name == "Diabetes":
        client_data, client_test_data, _meta = load_diabetes_data(
            num_clients=num_clients, seed=seed, is_iid=False, alpha=0.5
        )
        input_dim = 22
    elif dataset_name == "Heart_Disease":
        client_data, client_test_data, _meta = load_heart_disease_data(
            num_clients=num_clients, partition_by_hospital=True, seed=seed,
        )
        input_dim = 13
    else:
        raise ValueError(f"Unknown tabular dataset: {dataset_name}")

    trainer = FederatedTrainer(
        num_clients=num_clients,
        algorithm=algorithm,
        local_epochs=config["local_epochs"],
        batch_size=config["batch_size"],
        learning_rate=config["learning_rate"],
        mu=config.get("mu", 0.1),
        is_iid=False,
        alpha=0.5,
        seed=seed,
        external_data=client_data,
        external_test_data=client_test_data,
        input_dim=input_dim,
    )

    es = EarlyStoppingMonitor(**{k: v for k, v in es_config.items() if k != "enabled"}) if es_config.get("enabled") else None

    history = []
    best_acc = 0.0
    best_round = 0

    for r in range(num_rounds):
        rr = trainer.train_round(r)
        metrics = {
            "round": r + 1,
            "accuracy": rr.global_acc,
            "loss": rr.global_loss,
            "f1": rr.global_f1,
            "precision": rr.global_precision,
            "recall": rr.global_recall,
            "auc": rr.global_auc,
        }
        history.append(metrics)

        if rr.global_acc > best_acc:
            best_acc = rr.global_acc
            best_round = r + 1

        log(f"EXP {exp_idx}/{total_exps} | {dataset_name} | {algorithm} | s{seed} | "
            f"R {r+1}/{num_rounds} | Acc:{rr.global_acc:.1%} | Best:{best_acc:.1%}(r{best_round})")

        if es and es.check(r + 1, {"accuracy": rr.global_acc, "f1": rr.global_f1, "loss": rr.global_loss}):
            log(f"  Early stop at round {r+1} (best={best_acc:.1%} at r{best_round})")
            break

    per_client_acc = _evaluate_per_client(trainer)
    fairness = _compute_fairness(per_client_acc)

    elapsed = time.time() - start

    result = {
        "dataset": dataset_name,
        "algorithm": algorithm,
        "seed": seed,
        "history": history,
        "final_metrics": history[-1] if history else {},
        "per_client_acc": per_client_acc,
        "fairness": fairness,
        "runtime_seconds": round(elapsed, 1),
        "config": config,
        "stopped_early": es is not None and es.counter >= es.patience,
        "actual_rounds": len(history),
        "best_metrics": {"accuracy": best_acc, "round": best_round},
        "best_round": best_round,
    }

    _cleanup_gpu()
    return result


# ======================================================================
# Progress summary
# ======================================================================

def print_progress_summary(checkpoint_data: Dict, elapsed_total: float):
    """Print a clear summary of progress."""
    completed = checkpoint_data.get("completed", {})
    total = checkpoint_data["metadata"]["total_experiments"]
    done = len(completed)

    if done == 0:
        return

    # ETA calculation
    avg_time = elapsed_total / done if done > 0 else 0
    remaining = (total - done) * avg_time
    eta_str = str(timedelta(seconds=int(remaining)))
    elapsed_str = str(timedelta(seconds=int(elapsed_total)))

    # Per-dataset summary
    ds_summary = {}
    for key, res in completed.items():
        ds = res["dataset"]
        if ds not in ds_summary:
            ds_summary[ds] = {"done": 0, "total": 0, "best_acc": 0, "best_algo": ""}
        ds_summary[ds]["done"] += 1
        acc = res.get("best_metrics", {}).get("accuracy", 0)
        if acc > ds_summary[ds]["best_acc"]:
            ds_summary[ds]["best_acc"] = acc
            ds_summary[ds]["best_algo"] = res["algorithm"]

    # Count totals per dataset
    seeds = checkpoint_data["metadata"].get("seeds", SEEDS)
    for ds in list(IMAGING_DATASETS.keys()) + list(TABULAR_DATASETS.keys()):
        if ds not in ds_summary:
            ds_summary[ds] = {"done": 0, "total": 0, "best_acc": 0, "best_algo": "-"}
        ds_summary[ds]["total"] = len(ALGORITHMS) * len(seeds)

    print("\n" + "=" * 66)
    print(f"  PROGRESS: {done}/{total} ({100*done/total:.1f}%) | "
          f"{elapsed_str} elapsed | ~{eta_str} remaining")
    for ds_name, info in ds_summary.items():
        short = IMAGING_DATASETS.get(ds_name, TABULAR_DATASETS.get(ds_name, {})).get("short", ds_name[:2])
        status = "done" if info["done"] >= info["total"] else f"{info['done']}/{info['total']}"
        best_str = f"best={info['best_acc']:.1%} ({info['best_algo']})" if info["best_acc"] > 0 else "pending"
        print(f"  {short:>3}: {status:>8} | {best_str}")
    print("=" * 66 + "\n")


# ======================================================================
# Main experiment runner
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="FL-EHDS V2 Experiments")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--quick", action="store_true", help="Quick validation (1 seed, 5 rounds)")
    args = parser.parse_args()

    # Setup logging
    global _log_file
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _log_file = open(OUTPUT_DIR / LOG_FILE, "a")

    # Quick mode overrides
    seeds = [42] if args.quick else SEEDS
    if args.quick:
        imaging_config = {**V2_IMAGING_CONFIG, "num_rounds": 5, "local_epochs": 1}
        tabular_config = {**V2_TABULAR_CONFIG, "num_rounds": 5, "local_epochs": 2}
        es_config = {"enabled": False}
    else:
        imaging_config = V2_IMAGING_CONFIG.copy()
        tabular_config = V2_TABULAR_CONFIG.copy()
        es_config = V2_EARLY_STOPPING.copy()

    # Build experiment list
    experiments = []
    for ds_name in IMAGING_DATASETS:
        for algo in ALGORITHMS:
            for seed in seeds:
                experiments.append(("imaging", ds_name, algo, seed))
    for ds_name in TABULAR_DATASETS:
        for algo in ALGORITHMS:
            for seed in seeds:
                experiments.append(("tabular", ds_name, algo, seed))

    total_exps = len(experiments)

    # Load or create checkpoint
    checkpoint_data = None
    if args.resume:
        checkpoint_data = load_checkpoint()
        if checkpoint_data:
            done = len(checkpoint_data.get("completed", {}))
            log(f"RESUMED from checkpoint: {done}/{total_exps} experiments completed")
        else:
            log("No checkpoint found, starting fresh")

    if checkpoint_data is None:
        checkpoint_data = {
            "completed": {},
            "metadata": {
                "total_experiments": total_exps,
                "algorithms": ALGORITHMS,
                "seeds": seeds,
                "imaging_config": imaging_config,
                "tabular_config": tabular_config,
                "early_stopping": es_config,
                "start_time": datetime.now().isoformat(),
                "last_save": None,
                "version": "v2",
            }
        }

    # Graceful shutdown handler
    _interrupted = [False]

    def _signal_handler(signum, frame):
        if _interrupted[0]:
            print("\nForced exit.")
            sys.exit(1)
        _interrupted[0] = True
        print(f"\n{'='*60}")
        print(f"  INTERRUZIONE RILEVATA - Salvataggio stato...")
        done = len(checkpoint_data.get("completed", {}))
        save_checkpoint(checkpoint_data)
        print(f"  Checkpoint salvato: {done}/{total_exps} esperimenti completati")
        print(f"  File: {OUTPUT_DIR / CHECKPOINT_FILE}")
        print(f"  Per riprendere: python -m benchmarks.run_full_experiments --resume")
        print(f"{'='*60}")
        sys.exit(0)

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # Header
    mode = "QUICK" if args.quick else "FULL"
    log(f"\n{'='*66}")
    log(f"  FL-EHDS Experiments V2 ({mode}) — {total_exps} experiments")
    log(f"  Algorithms: {', '.join(ALGORITHMS)}")
    log(f"  Seeds: {seeds}")
    log(f"  Device: {_detect_device(None)}")
    log(f"{'='*66}\n")

    # Print algorithm info table
    log("  Algorithm overview:")
    for algo in ALGORITHMS:
        info = ALGO_INFO[algo]
        log(f"    {algo:<10} {info['year']} {info['venue']:<16} {info['role']}")
    log("")

    # Run experiments
    global_start = time.time()
    completed_count = len(checkpoint_data.get("completed", {}))

    for exp_idx, (exp_type, ds_name, algo, seed) in enumerate(experiments, 1):
        key = f"{ds_name}_{algo}_{seed}"

        # Skip if already completed
        if key in checkpoint_data.get("completed", {}):
            continue

        if _interrupted[0]:
            break

        log(f"\n--- Starting EXP {exp_idx}/{total_exps}: {ds_name} | {algo} | seed={seed} ---")

        try:
            if exp_type == "imaging":
                result = run_single_imaging(
                    dataset_name=ds_name,
                    data_dir=IMAGING_DATASETS[ds_name]["data_dir"],
                    algorithm=algo,
                    seed=seed,
                    config=imaging_config,
                    es_config=es_config,
                    exp_idx=exp_idx,
                    total_exps=total_exps,
                )
            else:
                result = run_single_tabular(
                    dataset_name=ds_name,
                    algorithm=algo,
                    seed=seed,
                    config=tabular_config,
                    es_config=es_config,
                    exp_idx=exp_idx,
                    total_exps=total_exps,
                )

            # Save result
            checkpoint_data["completed"][key] = result
            completed_count += 1

            # Atomic checkpoint save after EVERY experiment
            checkpoint_data["metadata"]["done"] = completed_count
            checkpoint_data["metadata"]["elapsed_seconds"] = time.time() - global_start
            save_checkpoint(checkpoint_data)

            best_acc = result.get("best_metrics", {}).get("accuracy", 0)
            log(f"--- Completed: {ds_name} | {algo} | s{seed} | "
                f"Best={best_acc:.1%} | Time={result['runtime_seconds']:.0f}s | "
                f"Saved [{completed_count}/{total_exps}] ---")

        except Exception as e:
            log(f"ERROR in {key}: {e}")
            log(traceback.format_exc(), also_print=False)
            # Save checkpoint even on error
            save_checkpoint(checkpoint_data)
            continue

        # Print summary every 5 experiments
        if completed_count % 5 == 0:
            elapsed = time.time() - global_start
            print_progress_summary(checkpoint_data, elapsed)

    # Final summary
    elapsed_total = time.time() - global_start
    log(f"\n{'='*66}")
    log(f"  EXPERIMENTS COMPLETED: {completed_count}/{total_exps}")
    log(f"  Total time: {str(timedelta(seconds=int(elapsed_total)))}")
    log(f"  Checkpoint: {OUTPUT_DIR / CHECKPOINT_FILE}")
    log(f"{'='*66}")

    print_progress_summary(checkpoint_data, elapsed_total)

    # Final save
    checkpoint_data["metadata"]["end_time"] = datetime.now().isoformat()
    checkpoint_data["metadata"]["total_elapsed"] = elapsed_total
    save_checkpoint(checkpoint_data)

    if _log_file:
        _log_file.close()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
FL-EHDS Optimized Tabular Experiments — New Datasets (PTB-XL, Cardiovascular, Breast Cancer).

Per-dataset tuned configurations based on config.yaml optimal parameters
and clinical dataset characteristics. Uses ISOLATED output directory
to avoid checkpoint conflicts with imaging experiments.

Algorithms (same as V2):
  FedAvg (2017), FedProx (2020), Ditto (2021), FedLC (2022),
  FedExP (2023), FedLESAM (2024), HPFL (2025)

Datasets:
  PTB-XL ECG      — 21,799 records, 9 feat, 5-class, European (PTB Berlin), 52-site natural FL
  Cardiovascular  — 70,000 patients, 11 feat, binary, balanced, Dirichlet
  Breast Cancer   — 569 patients, 30 feat, binary, small-data FL, Dirichlet

Features:
  - Per-dataset optimized hyperparameters (lr, epochs, batch_size, rounds)
  - Atomic checkpoint after EVERY experiment
  - SIGINT/SIGTERM handler: graceful save on Ctrl+C
  - Clear progress with ETA
  - Early stopping with per-dataset patience
  - --resume to continue from last checkpoint
  - --quick for fast validation (1 seed, 5 rounds)

Usage:
    cd fl-ehds-framework
    python -m benchmarks.run_tabular_optimized [--resume] [--quick]

Output: benchmarks/paper_results_tabular/checkpoint_tabular.json

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

import numpy as np

# Add parent to path
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
# Configuration — Per-Dataset Optimized
# ======================================================================

ALGORITHMS = ["FedAvg", "FedProx", "Ditto", "FedLC", "FedExP", "FedLESAM", "HPFL"]

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

# === ISOLATED OUTPUT — no conflicts with imaging experiments ===
OUTPUT_DIR = FRAMEWORK_DIR / "benchmarks" / "paper_results_tabular"
CHECKPOINT_FILE = "checkpoint_tabular.json"
LOG_FILE = "experiment_tabular.log"

# Per-dataset configurations (optimized from config.yaml + clinical analysis)
DATASET_CONFIGS = {
    "PTB_XL": {
        "loader": "ptbxl",
        "input_dim": 9,
        "num_classes": 5,
        "num_clients": 5,
        "short": "PX",
        "description": "PTB-XL ECG (EU, 52 sites, 5-class)",
        "training": dict(
            learning_rate=0.005,    # Lower than default: 5-class multiclass needs stability
            batch_size=64,          # Larger: 21.8K samples afford bigger batches
            num_rounds=30,          # More rounds: 5-class needs longer convergence
            local_epochs=3,         # Standard: enough data per client (site-based ~4K each)
            mu=0.1,
        ),
        "partition": dict(
            partition_by_site=True,  # Natural FL from 52 German recording sites
        ),
        "early_stopping": dict(
            enabled=True,
            patience=6,
            min_delta=0.003,
            min_rounds=12,
            metric="accuracy",
        ),
    },
    "Cardiovascular": {
        "loader": "cardiovascular",
        "input_dim": 11,
        "num_classes": 2,
        "num_clients": 5,
        "short": "CV",
        "description": "Cardiovascular Disease (70K, balanced)",
        "training": dict(
            learning_rate=0.01,     # Standard: large dataset, binary, balanced
            batch_size=64,          # Larger: 70K samples, more stable gradients
            num_rounds=25,          # From config.yaml: sufficient for convergence
            local_epochs=3,         # Standard: 14K samples per client
            mu=0.1,
        ),
        "partition": dict(
            is_iid=False,
            alpha=0.5,
        ),
        "early_stopping": dict(
            enabled=True,
            patience=6,
            min_delta=0.003,
            min_rounds=10,
            metric="accuracy",
        ),
    },
    "Breast_Cancer": {
        "loader": "breast_cancer",
        "input_dim": 30,
        "num_classes": 2,
        "num_clients": 3,          # Fewer clients: only 569 samples -> ~190 per client
        "short": "BC",
        "description": "Breast Cancer Wisconsin (569, small-data FL)",
        "training": dict(
            learning_rate=0.001,    # Much lower: small data, noisy gradients at lr=0.01
            batch_size=16,          # Smaller: ~152 train samples/client, need more batches
            num_rounds=40,          # More rounds: compensate for conservative lr
            local_epochs=1,         # Minimal: prevent local overfitting on ~152 samples
            mu=0.1,
        ),
        "partition": dict(
            is_iid=False,
            alpha=0.5,
        ),
        "early_stopping": dict(
            enabled=True,
            patience=8,             # More patience: slow convergence expected
            min_delta=0.005,
            min_rounds=15,
            metric="accuracy",
        ),
    },
}


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
# Checkpoint / Resume
# ======================================================================

def save_checkpoint(data: Dict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / CHECKPOINT_FILE
    bak_path = OUTPUT_DIR / (CHECKPOINT_FILE + ".bak")

    data["metadata"]["last_save"] = datetime.now().isoformat()

    fd, tmp_path = tempfile.mkstemp(
        dir=str(OUTPUT_DIR), prefix=".ckpt_tab_", suffix=".tmp",
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
# Data loading
# ======================================================================

def load_dataset(ds_name: str, ds_config: Dict, seed: int):
    """Load dataset with per-dataset partition strategy."""
    num_clients = ds_config["num_clients"]
    partition = ds_config["partition"]

    if ds_config["loader"] == "ptbxl":
        return load_ptbxl_data(
            num_clients=num_clients,
            seed=seed,
            **partition,
        )
    elif ds_config["loader"] == "cardiovascular":
        return load_cardiovascular_data(
            num_clients=num_clients,
            seed=seed,
            **partition,
        )
    elif ds_config["loader"] == "breast_cancer":
        return load_breast_cancer_data(
            num_clients=num_clients,
            seed=seed,
            **partition,
        )
    else:
        raise ValueError(f"Unknown loader: {ds_config['loader']}")


# ======================================================================
# Training wrapper
# ======================================================================

def run_single_tabular(
    ds_name: str, algorithm: str, seed: int,
    ds_config: Dict,
    exp_idx: int, total_exps: int,
) -> Dict[str, Any]:
    """Run a single tabular FL experiment with per-dataset config."""
    start = time.time()
    training = ds_config["training"]
    num_rounds = training["num_rounds"]

    # Load data
    client_data, client_test_data, metadata = load_dataset(ds_name, ds_config, seed)

    # Create trainer
    trainer = FederatedTrainer(
        num_clients=ds_config["num_clients"],
        algorithm=algorithm,
        local_epochs=training["local_epochs"],
        batch_size=training["batch_size"],
        learning_rate=training["learning_rate"],
        mu=training.get("mu", 0.1),
        seed=seed,
        external_data=client_data,
        external_test_data=client_test_data,
        input_dim=ds_config["input_dim"],
        num_classes=ds_config["num_classes"],
    )

    # Early stopping
    es_config = ds_config["early_stopping"]
    es = EarlyStoppingMonitor(
        **{k: v for k, v in es_config.items() if k != "enabled"}
    ) if es_config.get("enabled") else None

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

        log(f"EXP {exp_idx}/{total_exps} | {ds_name} | {algorithm} | s{seed} | "
            f"R {r+1}/{num_rounds} | Acc:{rr.global_acc:.1%} | Best:{best_acc:.1%}(r{best_round})")

        if es and es.check(r + 1, {"accuracy": rr.global_acc, "f1": rr.global_f1, "loss": rr.global_loss}):
            log(f"  Early stop at round {r+1} (best={best_acc:.1%} at r{best_round})")
            break

    # Per-client evaluation
    per_client_acc = _evaluate_per_client(trainer)
    fairness = _compute_fairness(per_client_acc)

    elapsed = time.time() - start

    result = {
        "dataset": ds_name,
        "algorithm": algorithm,
        "seed": seed,
        "history": history,
        "final_metrics": history[-1] if history else {},
        "per_client_acc": per_client_acc,
        "fairness": fairness,
        "runtime_seconds": round(elapsed, 1),
        "config": training,
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
# Progress summary
# ======================================================================

def print_progress_summary(checkpoint_data: Dict, elapsed_total: float):
    completed = checkpoint_data.get("completed", {})
    total = checkpoint_data["metadata"]["total_experiments"]
    done = len(completed)

    if done == 0:
        return

    avg_time = elapsed_total / done
    remaining = (total - done) * avg_time
    eta_str = str(timedelta(seconds=int(remaining)))
    elapsed_str = str(timedelta(seconds=int(elapsed_total)))

    ds_summary = {}
    for key, res in completed.items():
        ds = res["dataset"]
        if ds not in ds_summary:
            ds_summary[ds] = {"done": 0, "best_acc": 0, "best_algo": ""}
        ds_summary[ds]["done"] += 1
        acc = res.get("best_metrics", {}).get("accuracy", 0)
        if acc > ds_summary[ds]["best_acc"]:
            ds_summary[ds]["best_acc"] = acc
            ds_summary[ds]["best_algo"] = res["algorithm"]

    seeds = checkpoint_data["metadata"].get("seeds", SEEDS)

    print(f"\n{'='*66}")
    print(f"  PROGRESS: {done}/{total} ({100*done/total:.1f}%) | "
          f"{elapsed_str} elapsed | ~{eta_str} remaining")
    for ds_name, cfg in DATASET_CONFIGS.items():
        short = cfg["short"]
        info = ds_summary.get(ds_name, {"done": 0, "best_acc": 0, "best_algo": "-"})
        ds_total = len(ALGORITHMS) * len(seeds)
        status = "done" if info["done"] >= ds_total else f"{info['done']}/{ds_total}"
        best_str = f"best={info['best_acc']:.1%} ({info['best_algo']})" if info["best_acc"] > 0 else "pending"
        print(f"  {short:>3}: {status:>8} | {best_str}")
    print(f"{'='*66}\n")


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="FL-EHDS Optimized Tabular Experiments")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--quick", action="store_true", help="Quick validation (1 seed, 5 rounds)")
    args = parser.parse_args()

    global _log_file
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _log_file = open(OUTPUT_DIR / LOG_FILE, "a")

    seeds = [42] if args.quick else SEEDS

    # Quick mode overrides
    if args.quick:
        for ds_name, cfg in DATASET_CONFIGS.items():
            cfg["training"]["num_rounds"] = 5
            cfg["training"]["local_epochs"] = 1
            cfg["early_stopping"]["enabled"] = False

    # Build experiment list
    experiments = []
    for ds_name in DATASET_CONFIGS:
        for algo in ALGORITHMS:
            for seed in seeds:
                experiments.append((ds_name, algo, seed))

    total_exps = len(experiments)

    # Load or create checkpoint
    checkpoint_data = None
    if args.resume:
        checkpoint_data = load_checkpoint()
        if checkpoint_data:
            done = len(checkpoint_data.get("completed", {}))
            log(f"RESUMED from checkpoint: {done}/{total_exps} completed")

    if checkpoint_data is None:
        checkpoint_data = {
            "completed": {},
            "metadata": {
                "total_experiments": total_exps,
                "algorithms": ALGORITHMS,
                "seeds": seeds,
                "dataset_configs": {
                    ds: {
                        "training": cfg["training"],
                        "num_clients": cfg["num_clients"],
                        "input_dim": cfg["input_dim"],
                        "num_classes": cfg["num_classes"],
                        "description": cfg["description"],
                    }
                    for ds, cfg in DATASET_CONFIGS.items()
                },
                "start_time": datetime.now().isoformat(),
                "last_save": None,
                "version": "tabular_optimized_v1",
            }
        }

    # Graceful shutdown
    _interrupted = [False]

    def _signal_handler(signum, frame):
        if _interrupted[0]:
            print("\nForced exit.")
            sys.exit(1)
        _interrupted[0] = True
        done = len(checkpoint_data.get("completed", {}))
        print(f"\n{'='*60}")
        print(f"  INTERRUZIONE - Salvataggio stato...")
        save_checkpoint(checkpoint_data)
        print(f"  Checkpoint salvato: {done}/{total_exps} esperimenti completati")
        print(f"  File: {OUTPUT_DIR / CHECKPOINT_FILE}")
        print(f"  Per riprendere: python -m benchmarks.run_tabular_optimized --resume")
        print(f"{'='*60}")
        sys.exit(0)

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # Header
    mode = "QUICK" if args.quick else "FULL"
    log(f"\n{'='*66}")
    log(f"  FL-EHDS Optimized Tabular Experiments ({mode})")
    log(f"  {total_exps} experiments = {len(DATASET_CONFIGS)} datasets x "
        f"{len(ALGORITHMS)} algos x {len(seeds)} seeds")
    log(f"{'='*66}")
    log(f"  Device:   {_detect_device(None)}")
    log(f"  Seeds:    {seeds}")
    log(f"  Output:   {OUTPUT_DIR}")
    log(f"{'='*66}")

    log("\n  Algorithms:")
    for algo in ALGORITHMS:
        info = ALGO_INFO[algo]
        log(f"    {algo:<10} {info['year']} {info['venue']:<16} {info['role']}")

    log("\n  Dataset configurations (optimized):")
    for ds_name, cfg in DATASET_CONFIGS.items():
        t = cfg["training"]
        log(f"    {cfg['short']:>3} {cfg['description']}")
        log(f"        lr={t['learning_rate']}, bs={t['batch_size']}, "
            f"rounds={t['num_rounds']}, epochs={t['local_epochs']}, "
            f"clients={cfg['num_clients']}")
    log("")

    # Run experiments
    global_start = time.time()
    completed_count = len(checkpoint_data.get("completed", {}))

    for exp_idx, (ds_name, algo, seed) in enumerate(experiments, 1):
        key = f"{ds_name}_{algo}_{seed}"

        if key in checkpoint_data.get("completed", {}):
            continue

        if _interrupted[0]:
            break

        log(f"\n--- EXP {exp_idx}/{total_exps}: {ds_name} | {algo} | seed={seed} ---")

        try:
            result = run_single_tabular(
                ds_name=ds_name,
                algorithm=algo,
                seed=seed,
                ds_config=DATASET_CONFIGS[ds_name],
                exp_idx=exp_idx,
                total_exps=total_exps,
            )

            checkpoint_data["completed"][key] = result
            completed_count += 1

            checkpoint_data["metadata"]["done"] = completed_count
            checkpoint_data["metadata"]["elapsed_seconds"] = time.time() - global_start
            save_checkpoint(checkpoint_data)

            best_acc = result.get("best_metrics", {}).get("accuracy", 0)
            es_info = f" ES@R{result['actual_rounds']}" if result.get("stopped_early") else ""
            log(f"--- Done: {ds_name} | {algo} | s{seed} | "
                f"Best={best_acc:.1%}{es_info} | {result['runtime_seconds']:.0f}s | "
                f"[{completed_count}/{total_exps}] ---")

        except Exception as e:
            log(f"ERROR in {key}: {e}")
            log(traceback.format_exc(), also_print=False)
            checkpoint_data["completed"][key] = {
                "dataset": ds_name, "algorithm": algo, "seed": seed,
                "error": str(e), "traceback": traceback.format_exc(),
            }
            save_checkpoint(checkpoint_data)
            continue

        if completed_count % 7 == 0:
            print_progress_summary(checkpoint_data, time.time() - global_start)

    # Final summary
    elapsed_total = time.time() - global_start
    log(f"\n{'='*66}")
    log(f"  COMPLETED: {completed_count}/{total_exps}")
    log(f"  Total time: {str(timedelta(seconds=int(elapsed_total)))}")
    log(f"  Checkpoint: {OUTPUT_DIR / CHECKPOINT_FILE}")
    log(f"{'='*66}")

    print_progress_summary(checkpoint_data, elapsed_total)

    checkpoint_data["metadata"]["end_time"] = datetime.now().isoformat()
    checkpoint_data["metadata"]["total_elapsed"] = elapsed_total
    save_checkpoint(checkpoint_data)

    if _log_file:
        _log_file.close()


if __name__ == "__main__":
    main()

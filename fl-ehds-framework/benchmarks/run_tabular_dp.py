#!/usr/bin/env python3
"""
FL-EHDS Differential Privacy Experiments — Privacy-Utility Tradeoff.

Evaluates the impact of differential privacy (DP) on FL accuracy
across 4 epsilon levels, 3 top algorithms, 3 datasets, and 5 seeds.

Design:
  ε ∈ {1, 5, 10, 50}  (per-round epsilon; no-DP baseline already in checkpoint_tabular.json)
  Algorithms: FedAvg (baseline), Ditto (best personalized), HPFL (best fairness)
  Datasets: PTB-XL, Cardiovascular, Breast Cancer
  Seeds: 42, 123, 456, 789, 999
  DP mode: central (server-side Gaussian noise after aggregation)
  Clip norm: 1.0

  Total: 4 ε × 3 datasets × 3 algorithms × 5 seeds = 180 experiments

Noise mechanism: Gaussian with noise_scale = clip_norm / epsilon
  ε=1:  noise_scale = 1.0   (strong privacy, high noise)
  ε=5:  noise_scale = 0.2   (moderate privacy)
  ε=10: noise_scale = 0.1   (mild privacy)
  ε=50: noise_scale = 0.02  (very mild, near baseline)

Usage:
    cd fl-ehds-framework
    python -m benchmarks.run_tabular_dp [--quick] [--fresh]

Resume is AUTOMATIC: if a checkpoint exists, training resumes from where it stopped.
Use --fresh to discard existing checkpoint and start from scratch.

Output: benchmarks/paper_results_tabular/checkpoint_dp.json

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
# Configuration
# ======================================================================

# Top-3 algorithms: baseline + best personalized + best fairness
ALGORITHMS = ["FedAvg", "Ditto", "HPFL"]

# Per-round epsilon values (no-DP baseline already available from run_tabular_optimized)
DP_EPSILONS = [1, 5, 10, 50]

SEEDS = [42, 123, 456, 789, 999]

# Output in same directory as tabular results
OUTPUT_DIR = FRAMEWORK_DIR / "benchmarks" / "paper_results_tabular"
CHECKPOINT_FILE = "checkpoint_dp.json"
LOG_FILE = "experiment_dp.log"

# Same per-dataset configs as run_tabular_optimized.py
DATASET_CONFIGS = {
    "PTB_XL": {
        "loader": "ptbxl",
        "input_dim": 9,
        "num_classes": 5,
        "num_clients": 5,
        "short": "PX",
        "description": "PTB-XL ECG (EU, 52 sites, 5-class)",
        "training": dict(
            learning_rate=0.005,
            batch_size=64,
            num_rounds=30,
            local_epochs=3,
            mu=0.1,
        ),
        "partition": dict(partition_by_site=True),
        "early_stopping": dict(enabled=True, patience=6, min_delta=0.003, min_rounds=12, metric="accuracy"),
    },
    "Cardiovascular": {
        "loader": "cardiovascular",
        "input_dim": 11,
        "num_classes": 2,
        "num_clients": 5,
        "short": "CV",
        "description": "Cardiovascular Disease (70K, balanced)",
        "training": dict(
            learning_rate=0.01,
            batch_size=64,
            num_rounds=25,
            local_epochs=3,
            mu=0.1,
        ),
        "partition": dict(is_iid=False, alpha=0.5),
        "early_stopping": dict(enabled=True, patience=6, min_delta=0.003, min_rounds=10, metric="accuracy"),
    },
    "Breast_Cancer": {
        "loader": "breast_cancer",
        "input_dim": 30,
        "num_classes": 2,
        "num_clients": 3,
        "short": "BC",
        "description": "Breast Cancer Wisconsin (569, small-data FL)",
        "training": dict(
            learning_rate=0.001,
            batch_size=16,
            num_rounds=40,
            local_epochs=1,
            mu=0.1,
        ),
        "partition": dict(is_iid=False, alpha=0.5),
        "early_stopping": dict(enabled=True, patience=8, min_delta=0.005, min_rounds=15, metric="accuracy"),
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
# Checkpoint
# ======================================================================

def save_checkpoint(data: Dict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / CHECKPOINT_FILE
    bak_path = OUTPUT_DIR / (CHECKPOINT_FILE + ".bak")
    data["metadata"]["last_save"] = datetime.now().isoformat()

    fd, tmp_path = tempfile.mkstemp(dir=str(OUTPUT_DIR), prefix=".ckpt_dp_", suffix=".tmp")
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
    num_clients = ds_config["num_clients"]
    partition = ds_config["partition"]

    if ds_config["loader"] == "ptbxl":
        return load_ptbxl_data(num_clients=num_clients, seed=seed, **partition)
    elif ds_config["loader"] == "cardiovascular":
        return load_cardiovascular_data(num_clients=num_clients, seed=seed, **partition)
    elif ds_config["loader"] == "breast_cancer":
        return load_breast_cancer_data(num_clients=num_clients, seed=seed, **partition)
    else:
        raise ValueError(f"Unknown loader: {ds_config['loader']}")


# ======================================================================
# Training wrapper (with DP)
# ======================================================================

def run_single_dp(
    ds_name: str, algorithm: str, seed: int,
    dp_epsilon: int,
    ds_config: Dict,
    exp_idx: int, total_exps: int,
) -> Dict[str, Any]:
    """Run a single tabular FL experiment WITH differential privacy."""
    start = time.time()
    training = ds_config["training"]
    num_rounds = training["num_rounds"]

    client_data, client_test_data, metadata = load_dataset(ds_name, ds_config, seed)

    # Create trainer WITH DP enabled
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
        # DP parameters
        dp_enabled=True,
        dp_epsilon=float(dp_epsilon),
        dp_clip_norm=1.0,
        dp_mode="central",
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

        log(f"[{exp_idx}/{total_exps}] {ds_config['short']} | {algorithm} | ε={dp_epsilon} | s{seed} | "
            f"R{r+1}/{num_rounds} | Acc:{rr.global_acc:.1%} | Best:{best_acc:.1%}(r{best_round})")

        if es and es.check(r + 1, {"accuracy": rr.global_acc, "f1": rr.global_f1, "loss": rr.global_loss}):
            log(f"  -> Early stop at R{r+1} (best={best_acc:.1%} at r{best_round})")
            break

    per_client_acc = _evaluate_per_client(trainer)
    fairness = _compute_fairness(per_client_acc)
    elapsed = time.time() - start

    result = {
        "dataset": ds_name,
        "algorithm": algorithm,
        "seed": seed,
        "dp_epsilon": dp_epsilon,
        "dp_clip_norm": 1.0,
        "dp_mode": "central",
        "noise_scale": round(1.0 / dp_epsilon, 4),
        "history": history,
        "final_metrics": history[-1] if history else {},
        "per_client_acc": per_client_acc,
        "fairness": fairness,
        "runtime_seconds": round(elapsed, 1),
        "config": training,
        "stopped_early": es is not None and es.counter >= es.patience,
        "actual_rounds": len(history),
        "best_metrics": {"accuracy": best_acc, "round": best_round},
    }

    _cleanup_gpu()
    return result


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="FL-EHDS DP Privacy-Utility Experiments")
    parser.add_argument("--fresh", action="store_true", help="Discard existing checkpoint and start from scratch")
    parser.add_argument("--quick", action="store_true", help="Quick validation (1 seed, 1 epsilon, 5 rounds)")
    args = parser.parse_args()

    global _log_file
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _log_file = open(OUTPUT_DIR / LOG_FILE, "a")

    seeds = [42] if args.quick else SEEDS
    epsilons = [10] if args.quick else DP_EPSILONS

    if args.quick:
        for cfg in DATASET_CONFIGS.values():
            cfg["training"]["num_rounds"] = 5
            cfg["training"]["local_epochs"] = 1
            cfg["early_stopping"]["enabled"] = False

    # Build experiment list: ε × dataset × algorithm × seed
    experiments = []
    for eps in epsilons:
        for ds_name in DATASET_CONFIGS:
            for algo in ALGORITHMS:
                for seed in seeds:
                    experiments.append((eps, ds_name, algo, seed))

    total_exps = len(experiments)

    # Load or create checkpoint (auto-resume unless --fresh)
    checkpoint_data = None
    if not args.fresh:
        checkpoint_data = load_checkpoint()
        if checkpoint_data:
            done = len(checkpoint_data.get("completed", {}))
            log(f"AUTO-RESUMED from checkpoint: {done}/{total_exps} completed")
    elif args.fresh:
        log("FRESH start requested — ignoring existing checkpoint")

    if checkpoint_data is None:
        checkpoint_data = {
            "completed": {},
            "metadata": {
                "total_experiments": total_exps,
                "algorithms": ALGORITHMS,
                "dp_epsilons": epsilons,
                "seeds": seeds,
                "dp_clip_norm": 1.0,
                "dp_mode": "central",
                "start_time": datetime.now().isoformat(),
                "last_save": None,
                "version": "tabular_dp_v1",
            }
        }

    # Graceful shutdown
    _interrupted = [False]

    def _signal_handler(signum, frame):
        if _interrupted[0]:
            sys.exit(1)
        _interrupted[0] = True
        done = len(checkpoint_data.get("completed", {}))
        print(f"\n  INTERRUZIONE - Salvataggio stato... ({done}/{total_exps})")
        save_checkpoint(checkpoint_data)
        print(f"  Checkpoint salvato. Per riprendere: python -m benchmarks.run_tabular_dp")
        sys.exit(0)

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # Header
    mode = "QUICK" if args.quick else "FULL"
    log(f"\n{'='*66}")
    log(f"  FL-EHDS Differential Privacy Experiments ({mode})")
    log(f"  {total_exps} experiments = {len(epsilons)} ε × "
        f"{len(DATASET_CONFIGS)} datasets × {len(ALGORITHMS)} algos × {len(seeds)} seeds")
    log(f"{'='*66}")
    log(f"  Device:     {_detect_device(None)}")
    log(f"  Seeds:      {seeds}")
    log(f"  Epsilons:   {epsilons}")
    log(f"  Algorithms: {ALGORITHMS}")
    log(f"  DP mode:    central (server-side Gaussian noise)")
    log(f"  Clip norm:  1.0")
    log(f"  Output:     {OUTPUT_DIR}")
    log(f"{'='*66}")

    log("\n  Noise scales per epsilon:")
    for eps in epsilons:
        log(f"    ε={eps:>2}: noise_scale = {1.0/eps:.4f}")
    log("")

    # Run experiments
    global_start = time.time()
    completed_count = len(checkpoint_data.get("completed", {}))

    for exp_idx, (eps, ds_name, algo, seed) in enumerate(experiments, 1):
        key = f"{ds_name}_{algo}_eps{eps}_s{seed}"

        if key in checkpoint_data.get("completed", {}):
            continue

        if _interrupted[0]:
            break

        log(f"\n--- [{exp_idx}/{total_exps}] {ds_name} | {algo} | ε={eps} | seed={seed} ---")

        try:
            result = run_single_dp(
                ds_name=ds_name,
                algorithm=algo,
                seed=seed,
                dp_epsilon=eps,
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
            log(f"--- Done: {ds_name} | {algo} | ε={eps} | s{seed} | "
                f"Best={best_acc:.1%}{es_info} | {result['runtime_seconds']:.0f}s | "
                f"[{completed_count}/{total_exps}] ---")

        except Exception as e:
            log(f"ERROR in {key}: {e}")
            log(traceback.format_exc(), also_print=False)
            checkpoint_data["completed"][key] = {
                "dataset": ds_name, "algorithm": algo, "seed": seed,
                "dp_epsilon": eps, "error": str(e),
            }
            save_checkpoint(checkpoint_data)
            continue

        # Progress summary every 15 experiments
        if completed_count % 15 == 0:
            elapsed = time.time() - global_start
            avg = elapsed / completed_count
            remaining = (total_exps - completed_count) * avg
            log(f"\n  PROGRESS: {completed_count}/{total_exps} | "
                f"{timedelta(seconds=int(elapsed))} elapsed | "
                f"~{timedelta(seconds=int(remaining))} remaining\n")

    # Final summary
    elapsed_total = time.time() - global_start
    log(f"\n{'='*66}")
    log(f"  COMPLETED: {completed_count}/{total_exps}")
    log(f"  Total time: {timedelta(seconds=int(elapsed_total))}")
    log(f"  Checkpoint: {OUTPUT_DIR / CHECKPOINT_FILE}")
    log(f"{'='*66}")

    # Print results summary
    completed = checkpoint_data.get("completed", {})
    log("\n  Privacy-Utility Summary (Best Accuracy %):")
    log(f"  {'DS':>3} | {'Algo':<8} | " + " | ".join(f"ε={e:<3}" for e in epsilons))
    log(f"  {'-'*3}-+-{'-'*8}-+-" + "-+-".join("-"*6 for _ in epsilons))

    from collections import defaultdict
    results_by = defaultdict(lambda: defaultdict(list))
    for key, res in completed.items():
        if "error" in res:
            continue
        ds = res["dataset"]
        algo = res["algorithm"]
        eps = res["dp_epsilon"]
        acc = res.get("best_metrics", {}).get("accuracy", 0)
        results_by[(ds, algo)][eps].append(acc)

    for ds_name in DATASET_CONFIGS:
        short = DATASET_CONFIGS[ds_name]["short"]
        for algo in ALGORITHMS:
            vals = []
            for eps in epsilons:
                accs = results_by.get((ds_name, algo), {}).get(eps, [])
                if accs:
                    vals.append(f"{100*np.mean(accs):5.1f}")
                else:
                    vals.append("  -- ")
            log(f"  {short:>3} | {algo:<8} | " + " | ".join(vals))

    checkpoint_data["metadata"]["end_time"] = datetime.now().isoformat()
    checkpoint_data["metadata"]["total_elapsed"] = elapsed_total
    save_checkpoint(checkpoint_data)

    if _log_file:
        _log_file.close()


if __name__ == "__main__":
    main()

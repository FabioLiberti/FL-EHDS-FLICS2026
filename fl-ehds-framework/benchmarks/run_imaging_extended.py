#!/usr/bin/env python3
"""
FL-EHDS Imaging — Chest X-Ray Extended (HPFL + FedLESAM).

Tests A+D: Adds HPFL and FedLESAM to the Chest X-ray evaluation.
Same config as run_imaging_chest.py (FedAvg + Ditto) for fair comparison.

Total: 2 algos × 1 dataset × 3 seeds = 6 experiments
Estimated runtime: ~2-4 hours

Usage:
    cd fl-ehds-framework
    python -m benchmarks.run_imaging_extended [--quick] [--fresh]

Output: benchmarks/paper_results_delta/checkpoint_chest_extended.json

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
from typing import Dict, Optional, Any

import numpy as np

FRAMEWORK_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(FRAMEWORK_DIR))

import torch

from terminal.fl_trainer import (
    ImageFederatedTrainer,
    _detect_device,
)

# ======================================================================
# Configuration — IDENTICAL to run_imaging_chest.py
# ======================================================================

ALGORITHMS = ["HPFL", "FedLESAM"]

SEEDS = [42, 123, 456]

OUTPUT_DIR = FRAMEWORK_DIR / "benchmarks" / "paper_results_delta"
CHECKPOINT_FILE = "checkpoint_chest_extended.json"
LOG_FILE = "experiment_chest_extended.log"

DATASET_CONFIG = {
    "data_dir": str(FRAMEWORK_DIR / "data" / "chest_xray"),
    "num_classes": 2,
    "short": "CX",
}

# IDENTICAL to run_imaging_chest.py / run_imaging_delta.py
IMAGING_CONFIG = dict(
    num_clients=5,
    num_rounds=20,
    local_epochs=2,
    batch_size=32,
    learning_rate=0.001,
    model_type="resnet18",
    is_iid=False,
    alpha=0.5,
    freeze_backbone=False,
    freeze_level=2,
    use_fedbn=True,
    use_class_weights=True,
    use_amp=True,
    mu=0.1,
)

EARLY_STOPPING = dict(
    enabled=True,
    patience=4,
    min_delta=0.003,
    min_rounds=8,
    metric="accuracy",
)


# ======================================================================
# Logging
# ======================================================================

_log_file = None

def log(msg: str, also_print: bool = True):
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

def save_checkpoint(data: Dict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / CHECKPOINT_FILE
    bak_path = OUTPUT_DIR / (CHECKPOINT_FILE + ".bak")
    data["metadata"]["last_save"] = datetime.now().isoformat()

    fd, tmp_path = tempfile.mkstemp(dir=str(OUTPUT_DIR), prefix=".ckpt_ext_", suffix=".tmp")
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
    def __init__(self, patience=4, min_delta=0.003, min_rounds=8, metric="accuracy"):
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
# Per-client evaluation & fairness
# ======================================================================

def _evaluate_per_client(trainer) -> Dict[str, float]:
    model = trainer.global_model
    model.eval()
    per_client = {}

    with torch.no_grad():
        for cid in range(trainer.num_clients):
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
# Training
# ======================================================================

def run_single_imaging(
    algorithm: str, seed: int,
    config: Dict, es_config: Dict,
    exp_idx: int, total_exps: int,
) -> Dict[str, Any]:
    start = time.time()
    num_rounds = config["num_rounds"]

    trainer = ImageFederatedTrainer(
        data_dir=DATASET_CONFIG["data_dir"],
        num_clients=config["num_clients"],
        algorithm=algorithm,
        local_epochs=config["local_epochs"],
        batch_size=config["batch_size"],
        learning_rate=config["learning_rate"],
        is_iid=config["is_iid"],
        alpha=config["alpha"],
        mu=config.get("mu", 0.1),
        seed=seed,
        model_type=config["model_type"],
        freeze_backbone=config.get("freeze_backbone", False),
        freeze_level=config.get("freeze_level"),
        use_fedbn=config.get("use_fedbn", False),
        use_class_weights=config.get("use_class_weights", True),
        use_amp=config.get("use_amp", True),
    )
    trainer.num_rounds = num_rounds

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

        log("[{}/{}] chest_xray | {} | s{} | R{}/{} | Acc:{:.1%} | Best:{:.1%}(r{})".format(
            exp_idx, total_exps, algorithm, seed, r+1, num_rounds,
            rr.global_acc, best_acc, best_round))

        if es and es.check(r + 1, {"accuracy": rr.global_acc}):
            log("  -> Early stop at R{} (best={:.1%} at r{})".format(r+1, best_acc, best_round))
            break

    per_client_acc = _evaluate_per_client(trainer)
    fairness = _compute_fairness(per_client_acc)
    elapsed = time.time() - start

    result = {
        "dataset": "chest_xray",
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
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="FL-EHDS Chest X-Ray Extended (HPFL + FedLESAM)")
    parser.add_argument("--fresh", action="store_true", help="Discard existing checkpoint")
    parser.add_argument("--quick", action="store_true", help="Quick validation (1 seed, 3 rounds)")
    args = parser.parse_args()

    global _log_file
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _log_file = open(OUTPUT_DIR / LOG_FILE, "a")

    seeds = [42] if args.quick else SEEDS
    config = {**IMAGING_CONFIG}

    if args.quick:
        config["num_rounds"] = 3
        config["local_epochs"] = 1

    experiments = [(algo, seed) for algo in ALGORITHMS for seed in seeds]
    total_exps = len(experiments)

    # Auto-resume unless --fresh
    checkpoint_data = None
    if not args.fresh:
        checkpoint_data = load_checkpoint()
        if checkpoint_data:
            done = len(checkpoint_data.get("completed", {}))
            log("AUTO-RESUMED: {}/{} completed".format(done, total_exps))
    elif args.fresh:
        ckpt_path = OUTPUT_DIR / CHECKPOINT_FILE
        if ckpt_path.exists():
            ckpt_path.unlink()
        log("FRESH start")

    if checkpoint_data is None:
        checkpoint_data = {
            "completed": {},
            "metadata": {
                "total_experiments": total_exps,
                "algorithms": ALGORITHMS,
                "dataset": "chest_xray",
                "seeds": seeds,
                "start_time": datetime.now().isoformat(),
                "last_save": None,
                "version": "chest_extended_v1",
            }
        }

    # Graceful shutdown
    _interrupted = [False]

    def _signal_handler(signum, frame):
        if _interrupted[0]:
            sys.exit(1)
        _interrupted[0] = True
        done = len(checkpoint_data.get("completed", {}))
        save_checkpoint(checkpoint_data)
        print("\n  Checkpoint salvato: {}/{}".format(done, total_exps))
        print("  Per riprendere: python -m benchmarks.run_imaging_extended")
        sys.exit(0)

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    mode = "QUICK" if args.quick else "FULL"
    log("\n" + "=" * 60)
    log("  FL-EHDS Chest X-Ray Extended ({})".format(mode))
    log("  {} experiments = {} algos x {} seeds".format(total_exps, len(ALGORITHMS), len(seeds)))
    log("  Algorithms: {}".format(ALGORITHMS))
    log("  Device: {}".format(_detect_device(None)))
    log("  Output: {}".format(OUTPUT_DIR / CHECKPOINT_FILE))
    log("=" * 60)

    global_start = time.time()
    completed_count = len(checkpoint_data.get("completed", {}))

    for exp_idx, (algo, seed) in enumerate(experiments, 1):
        key = "chest_xray_{}_s{}".format(algo, seed)

        if key in checkpoint_data.get("completed", {}):
            continue
        if _interrupted[0]:
            break

        log("\n--- [{}/{}] chest_xray | {} | seed={} ---".format(exp_idx, total_exps, algo, seed))

        try:
            result = run_single_imaging(
                algorithm=algo, seed=seed,
                config=config, es_config=EARLY_STOPPING,
                exp_idx=exp_idx, total_exps=total_exps,
            )

            checkpoint_data["completed"][key] = result
            completed_count += 1
            save_checkpoint(checkpoint_data)

            best_acc = result.get("best_metrics", {}).get("accuracy", 0)
            es_info = " ES@R{}".format(result["actual_rounds"]) if result.get("stopped_early") else ""
            log("--- Done: {} | s{} | Best={:.1%}{} | {:.0f}s | [{}/{}] ---".format(
                algo, seed, best_acc, es_info, result["runtime_seconds"], completed_count, total_exps))

        except Exception as e:
            log("ERROR in {}: {}".format(key, e))
            traceback.print_exc()
            checkpoint_data["completed"][key] = {
                "dataset": "chest_xray", "algorithm": algo, "seed": seed,
                "error": str(e),
            }
            save_checkpoint(checkpoint_data)

    elapsed_total = time.time() - global_start
    log("\n" + "=" * 60)
    log("  COMPLETED: {}/{}".format(completed_count, total_exps))
    log("  Total time: {}".format(timedelta(seconds=int(elapsed_total))))
    log("=" * 60)

    # Summary + comparison with baseline
    completed = checkpoint_data.get("completed", {})
    log("\n  Chest X-Ray Extended Results:")

    # Load baseline for comparison
    baseline_path = OUTPUT_DIR / "checkpoint_chest_baseline.json"
    baseline = {}
    if baseline_path.exists():
        try:
            with open(baseline_path) as f:
                bdata = json.load(f)
            for key, res in bdata.get("completed", {}).items():
                if "error" not in res:
                    algo = res["algorithm"]
                    if algo not in baseline:
                        baseline[algo] = []
                    baseline[algo].append(res.get("best_metrics", {}).get("accuracy", 0))
        except Exception:
            pass

    all_algos = list(baseline.keys()) + ALGORITHMS
    seen = set()
    for algo in all_algos:
        if algo in seen:
            continue
        seen.add(algo)
        accs = []
        # From baseline
        if algo in baseline:
            accs = baseline[algo]
        # From this run
        for key, res in completed.items():
            if "error" not in res and res.get("algorithm") == algo:
                accs.append(res.get("best_metrics", {}).get("accuracy", 0))
        if accs:
            log("  {:>10}: {:.1f} +/- {:.1f}%".format(algo, 100*np.mean(accs), 100*np.std(accs)))

    checkpoint_data["metadata"]["end_time"] = datetime.now().isoformat()
    save_checkpoint(checkpoint_data)
    if _log_file:
        _log_file.close()


if __name__ == "__main__":
    main()

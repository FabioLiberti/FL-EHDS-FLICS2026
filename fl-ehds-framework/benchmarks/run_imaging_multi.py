#!/usr/bin/env python3
"""
FL-EHDS — Brain Tumor + Skin Cancer Imaging Experiments.

Validates the modality-dependent personalization effect across 3 imaging
datasets (not just Chest X-ray). Runs FedAvg (baseline) and HPFL on
Brain Tumor (4-class) and Skin Cancer (binary).

Key hypothesis: HPFL fails on imaging regardless of dataset/task type.
If confirmed across 3 datasets, the pattern is robust and publishable.

Experiments:
  - 2 datasets x 2 algos x 3 seeds = 12 experiments
  - Estimated time: ~2-3 hours (MPS/CUDA), ~4-5 hours (CPU)

Usage:
    cd fl-ehds-framework
    python -m benchmarks.run_imaging_multi [--quick] [--fresh] [--resume]
    python -m benchmarks.run_imaging_multi --algos FedAvg HPFL Ditto

Output: benchmarks/paper_results_delta/checkpoint_imaging_multi.json

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
from typing import Dict, Optional, Any, List

import numpy as np

FRAMEWORK_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(FRAMEWORK_DIR))

import torch

from terminal.fl_trainer import (
    ImageFederatedTrainer,
    _detect_device,
)

# ======================================================================
# Configuration
# ======================================================================

DEFAULT_ALGORITHMS = ["FedAvg", "HPFL"]

SEEDS = [42, 123, 456]

OUTPUT_DIR = FRAMEWORK_DIR / "benchmarks" / "paper_results_delta"
CHECKPOINT_FILE = "checkpoint_imaging_multi.json"
LOG_FILE = "experiment_imaging_multi.log"

IMAGING_DATASETS = {
    "Brain_Tumor": {
        "data_dir": str(FRAMEWORK_DIR / "data" / "Brain_Tumor"),
        "num_classes": 4,
        "short": "BT",
    },
    "Skin_Cancer": {
        "data_dir": str(FRAMEWORK_DIR / "data" / "Skin Cancer"),
        "num_classes": 2,
        "short": "SC",
    },
}

# Same config as run_imaging_delta.py / run_imaging_extended.py
IMAGING_CONFIG = dict(
    num_clients=5,
    num_rounds=20,
    local_epochs=2,
    batch_size=32,
    learning_rate=0.001,    # Auto-adjusted to 0.0005 for resnet18
    model_type="resnet18",
    is_iid=False,
    alpha=0.5,
    freeze_backbone=False,
    freeze_level=2,         # Freeze conv1+bn1+layer1
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

DATASET_OVERRIDES = {
    "Brain_Tumor": {"learning_rate": 0.0005},
}


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

    fd, tmp_path = tempfile.mkstemp(dir=str(OUTPUT_DIR), prefix=".ckpt_multi_", suffix=".tmp")
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
# Per-client evaluation & fairness (HPFL-aware)
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
    gini = (2 * sum((i + 1) * v for i, v in enumerate(sorted_a))) / (n * cumsum[-1]) - (n + 1) / n if cumsum[-1] > 0 else 0
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
    dataset_name: str, data_dir: str, algorithm: str, seed: int,
    config: Dict, es_config: Dict,
    exp_idx: int, total_exps: int,
) -> Dict[str, Any]:
    start = time.time()
    num_rounds = config["num_rounds"]

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

        log("  [{}/{}] {} | {} | s{} | R{}/{} | Acc:{:.1%} | Best:{:.1%}(r{})".format(
            exp_idx, total_exps, dataset_name, algorithm, seed,
            r + 1, num_rounds, rr.global_acc, best_acc, best_round))

        if es and es.check(r + 1, {"accuracy": rr.global_acc}):
            log("  -> Early stop at R{} (best={:.1%} at r{})".format(r + 1, best_acc, best_round))
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
        "config": cfg,
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
    parser = argparse.ArgumentParser(
        description="FL-EHDS — Brain Tumor + Skin Cancer Imaging Experiments")
    parser.add_argument("--quick", action="store_true",
                        help="Quick validation (1 seed, 3 rounds)")
    parser.add_argument("--fresh", action="store_true",
                        help="Start fresh (delete existing checkpoint)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint")
    parser.add_argument("--algos", nargs="+", default=None,
                        help="Algorithms to run (default: FedAvg HPFL)")
    args = parser.parse_args()

    global _log_file
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _log_file = open(OUTPUT_DIR / LOG_FILE, "a")

    algorithms = args.algos if args.algos else DEFAULT_ALGORITHMS
    seeds = [42] if args.quick else SEEDS

    if args.quick:
        config = {**IMAGING_CONFIG, "num_rounds": 3, "local_epochs": 1}
        es_config = {"enabled": False}
    else:
        config = IMAGING_CONFIG.copy()
        es_config = EARLY_STOPPING.copy()

    # Build experiment list: datasets x algorithms x seeds
    experiments = []
    for ds_name in IMAGING_DATASETS:
        for algo in algorithms:
            for seed in seeds:
                key = "{}_{}_{}" .format(ds_name, algo, seed)
                experiments.append({
                    "key": key,
                    "dataset": ds_name,
                    "algorithm": algo,
                    "seed": seed,
                })

    total_exps = len(experiments)

    if args.fresh:
        ckpt_path = OUTPUT_DIR / CHECKPOINT_FILE
        if ckpt_path.exists():
            ckpt_path.unlink()

    checkpoint_data = None
    if args.resume or not args.fresh:
        checkpoint_data = load_checkpoint()
        if checkpoint_data:
            done = len(checkpoint_data.get("completed", {}))
            log("RESUMED: {}/{} completed".format(done, total_exps))

    if checkpoint_data is None:
        checkpoint_data = {
            "completed": {},
            "metadata": {
                "total_experiments": total_exps,
                "algorithms": algorithms,
                "datasets": list(IMAGING_DATASETS.keys()),
                "seeds": seeds,
                "imaging_config": config,
                "early_stopping": es_config,
                "start_time": datetime.now().isoformat(),
                "last_save": None,
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
        print("  Per riprendere: python -m benchmarks.run_imaging_multi --resume")
        sys.exit(0)

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    mode = "QUICK" if args.quick else "FULL"
    log("\n" + "=" * 66)
    log("  FL-EHDS — Brain Tumor + Skin Cancer ({})".format(mode))
    log("  {} experiments = {} datasets x {} algos x {} seeds".format(
        total_exps, len(IMAGING_DATASETS), len(algorithms), len(seeds)))
    log("=" * 66)
    log("  Device:       {}".format(_detect_device(None)))
    log("  Algorithms:   {}".format(algorithms))
    log("  Datasets:     {}".format(list(IMAGING_DATASETS.keys())))
    log("  Seeds:        {}".format(seeds))
    log("  freeze_level: {} (conv1+bn1+layer1 frozen)".format(config["freeze_level"]))
    log("  local_epochs: {}".format(config["local_epochs"]))
    log("  num_rounds:   {} (patience={})".format(
        config["num_rounds"], es_config.get("patience", "-")))
    log("  Output:       {}".format(OUTPUT_DIR / CHECKPOINT_FILE))
    log("=" * 66 + "\n")

    global_start = time.time()
    completed_count = len(checkpoint_data.get("completed", {}))

    for exp_idx, exp in enumerate(experiments, 1):
        key = exp["key"]
        ds_name = exp["dataset"]
        algo = exp["algorithm"]
        seed = exp["seed"]

        if key in checkpoint_data.get("completed", {}):
            continue

        if _interrupted[0]:
            break

        ds_short = IMAGING_DATASETS[ds_name]["short"]
        log("\n--- [{}/{}] {} ({}) | {} | seed={} ---".format(
            exp_idx, total_exps, ds_name, ds_short, algo, seed))

        try:
            result = run_single_imaging(
                dataset_name=ds_name,
                data_dir=IMAGING_DATASETS[ds_name]["data_dir"],
                algorithm=algo,
                seed=seed,
                config=config,
                es_config=es_config,
                exp_idx=exp_idx,
                total_exps=total_exps,
            )

            checkpoint_data["completed"][key] = result
            completed_count += 1
            checkpoint_data["metadata"]["done"] = completed_count
            checkpoint_data["metadata"]["elapsed_seconds"] = time.time() - global_start
            save_checkpoint(checkpoint_data)

            best_acc = result.get("best_metrics", {}).get("accuracy", 0)
            es_info = " ES@R{}".format(result["actual_rounds"]) if result.get("stopped_early") else ""
            jain = result.get("fairness", {}).get("jain_index", 0)
            log("--- Done: {} | {} | s{} | Best={:.1%}{} | Jain={:.3f} | {:.0f}s | [{}/{}] ---".format(
                ds_name, algo, seed, best_acc, es_info, jain,
                result["runtime_seconds"], completed_count, total_exps))

            # ETA
            elapsed = time.time() - global_start
            if completed_count > 0:
                avg = elapsed / completed_count
                remaining = (total_exps - completed_count) * avg
                eta = str(timedelta(seconds=int(remaining)))
                log("  ETA: ~{}".format(eta))

        except Exception as e:
            log("ERROR in {}: {}".format(key, e))
            log(traceback.format_exc(), also_print=False)
            traceback.print_exc()
            checkpoint_data["completed"][key] = {
                "dataset": ds_name, "algorithm": algo, "seed": seed,
                "error": str(e), "traceback": traceback.format_exc(),
            }
            save_checkpoint(checkpoint_data)
            _cleanup_gpu()
            continue

    elapsed_total = time.time() - global_start
    checkpoint_data["metadata"]["end_time"] = datetime.now().isoformat()
    checkpoint_data["metadata"]["total_elapsed"] = elapsed_total
    save_checkpoint(checkpoint_data)

    # Summary table
    log("\n" + "=" * 66)
    log("  COMPLETED: {}/{}".format(completed_count, total_exps))
    log("  Total time: {}".format(str(timedelta(seconds=int(elapsed_total)))))
    log("=" * 66)

    log("\n  Summary:")
    log("  {:>14s} | {:>10s} | {:>10s} | {:>6s} | {:>5s}".format(
        "Dataset", "Algorithm", "Acc (%)", "Jain", "Rnds"))
    log("  " + "-" * 60)

    completed = checkpoint_data.get("completed", {})
    for ds_name in IMAGING_DATASETS:
        for algo in algorithms:
            accs = []
            jains = []
            rnds = []
            for seed in seeds:
                key = "{}_{}_{}" .format(ds_name, algo, seed)
                if key in completed and "error" not in completed[key]:
                    r = completed[key]
                    accs.append(r["best_metrics"]["accuracy"] * 100)
                    jains.append(r.get("fairness", {}).get("jain_index", 0))
                    rnds.append(r["actual_rounds"])
            if accs:
                log("  {:>14s} | {:>10s} | {:5.1f}+-{:.1f} | {:.3f} | {:>5.1f}".format(
                    ds_name, algo,
                    np.mean(accs), np.std(accs),
                    np.mean(jains),
                    np.mean(rnds)))
    log("  " + "-" * 60)

    log("\n  Checkpoint: {}".format(OUTPUT_DIR / CHECKPOINT_FILE))

    if _log_file:
        _log_file.close()


if __name__ == "__main__":
    main()

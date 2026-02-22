#!/usr/bin/env python3
"""
FL-EHDS Imaging Completion Runner — FedAvg (Brain Tumor) + Ditto (BT + SC).

Fills the 8 missing cells in the 4-algo x 3-seed x 3-dataset imaging matrix:

  Brain_Tumor  FedAvg   seed 123, 456      (2 experiments)
  Brain_Tumor  Ditto    seed 42, 123, 456   (3 experiments)
  Skin_Cancer  Ditto    seed 42, 123, 456   (3 experiments)
                                     Total:  8 experiments

Uses the SAME config as run_imaging_delta.py (freeze_level=2, local_epochs=2,
patience=4, num_rounds=20) for internal consistency with existing results.

Estimated runtime: ~3-4h (8 experiments x ~20-30 min each)

Usage:
    cd fl-ehds-framework
    python -m benchmarks.run_imaging_completion [--resume] [--quick]

Output: benchmarks/paper_results_delta/checkpoint_completion.json

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
# Configuration — matches run_imaging_delta.py for consistency
# ======================================================================

SEEDS = [42, 123, 456]

OUTPUT_DIR = FRAMEWORK_DIR / "benchmarks" / "paper_results_delta"
CHECKPOINT_FILE = "checkpoint_completion.json"
LOG_FILE = "experiment_completion.log"

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

# Exactly the 8 missing (dataset, algorithm, seed) tuples
EXPERIMENTS_TODO = [
    # Brain Tumor — FedAvg missing seeds
    ("Brain_Tumor", "FedAvg", 123),
    ("Brain_Tumor", "FedAvg", 456),
    # Brain Tumor — Ditto all seeds
    ("Brain_Tumor", "Ditto", 42),
    ("Brain_Tumor", "Ditto", 123),
    ("Brain_Tumor", "Ditto", 456),
    # Skin Cancer — Ditto all seeds
    ("Skin_Cancer", "Ditto", 42),
    ("Skin_Cancer", "Ditto", 123),
    ("Skin_Cancer", "Ditto", 456),
]

# Same config as run_imaging_delta.py
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
    mu=0.1,              # Ditto lambda parameter
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

    fd, tmp_path = tempfile.mkstemp(dir=str(OUTPUT_DIR), prefix=".ckpt_compl_", suffix=".tmp")
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

        log(f"[{exp_idx}/{total_exps}] {dataset_name} | {algorithm} | s{seed} | "
            f"R{r+1}/{num_rounds} | Acc:{rr.global_acc:.1%} | Best:{best_acc:.1%}(r{best_round})")

        if es and es.check(r + 1, {"accuracy": rr.global_acc}):
            log(f"  -> Early stop at R{r+1} (best={best_acc:.1%} at r{best_round})")
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
        description="FL-EHDS Imaging Completion (FedAvg BT + Ditto BT/SC)"
    )
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--quick", action="store_true", help="Quick validation (3 rounds, no ES)")
    args = parser.parse_args()

    global _log_file
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _log_file = open(OUTPUT_DIR / LOG_FILE, "a")

    if args.quick:
        config = {**IMAGING_CONFIG, "num_rounds": 3, "local_epochs": 1}
        es_config = {"enabled": False}
    else:
        config = IMAGING_CONFIG.copy()
        es_config = EARLY_STOPPING.copy()

    experiments = EXPERIMENTS_TODO
    total_exps = len(experiments)

    checkpoint_data = None
    if args.resume:
        checkpoint_data = load_checkpoint()
        if checkpoint_data:
            done = len(checkpoint_data.get("completed", {}))
            log(f"RESUMED: {done}/{total_exps} completed")

    if checkpoint_data is None:
        checkpoint_data = {
            "completed": {},
            "metadata": {
                "total_experiments": total_exps,
                "experiments": [
                    {"dataset": d, "algorithm": a, "seed": s}
                    for d, a, s in experiments
                ],
                "imaging_config": config,
                "early_stopping": es_config,
                "start_time": datetime.now().isoformat(),
                "last_save": None,
                "version": "completion_v1",
                "description": "Fills 8 missing cells: FedAvg(BT x2) + Ditto(BT x3, SC x3)",
            }
        }

    _interrupted = [False]

    def _signal_handler(signum, frame):
        if _interrupted[0]:
            sys.exit(1)
        _interrupted[0] = True
        done = len(checkpoint_data.get("completed", {}))
        save_checkpoint(checkpoint_data)
        print(f"\n  Checkpoint salvato: {done}/{total_exps}")
        print(f"  Per riprendere: python -m benchmarks.run_imaging_completion --resume")
        sys.exit(0)

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    log(f"\n{'='*66}")
    log(f"  FL-EHDS Imaging Completion — 8 missing experiments")
    log(f"  Brain Tumor: FedAvg (s123,s456) + Ditto (s42,s123,s456)")
    log(f"  Skin Cancer: Ditto (s42,s123,s456)")
    log(f"{'='*66}")
    log(f"  Device:       {_detect_device(None)}")
    log(f"  freeze_level: {config['freeze_level']} (conv1+bn1+layer1 frozen)")
    log(f"  local_epochs: {config['local_epochs']}")
    log(f"  num_rounds:   {config['num_rounds']} (patience={es_config.get('patience', '-')})")
    log(f"  Output:       {OUTPUT_DIR / CHECKPOINT_FILE}")
    if args.quick:
        log(f"  Mode:         QUICK (3 rounds, no ES)")
    log(f"{'='*66}\n")

    global_start = time.time()
    completed_count = len(checkpoint_data.get("completed", {}))

    for exp_idx, (ds_name, algo, seed) in enumerate(experiments, 1):
        key = f"{ds_name}_{algo}_{seed}"

        if key in checkpoint_data.get("completed", {}):
            continue

        if _interrupted[0]:
            break

        log(f"\n--- [{exp_idx}/{total_exps}] {ds_name} | {algo} | seed={seed} ---")

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
            es_info = f" ES@R{result['actual_rounds']}" if result.get("stopped_early") else ""
            log(f"--- Done: {ds_name} | {algo} | s{seed} | "
                f"Best={best_acc:.1%}{es_info} | {result['runtime_seconds']:.0f}s | "
                f"[{completed_count}/{total_exps}] ---")

            # ETA
            elapsed = time.time() - global_start
            avg = elapsed / completed_count
            remaining = (total_exps - completed_count) * avg
            eta = str(timedelta(seconds=int(remaining)))
            log(f"  ETA: ~{eta}")

        except Exception as e:
            log(f"ERROR in {key}: {e}")
            log(traceback.format_exc(), also_print=False)
            checkpoint_data["completed"][key] = {
                "dataset": ds_name, "algorithm": algo, "seed": seed,
                "error": str(e), "traceback": traceback.format_exc(),
            }
            save_checkpoint(checkpoint_data)
            continue

    elapsed_total = time.time() - global_start
    log(f"\n{'='*66}")
    log(f"  COMPLETED: {completed_count}/{total_exps}")
    log(f"  Total time: {str(timedelta(seconds=int(elapsed_total)))}")
    log(f"  Checkpoint: {OUTPUT_DIR / CHECKPOINT_FILE}")
    log(f"{'='*66}")

    # Print summary table
    log(f"\n  {'Dataset':<14} {'Algorithm':<10} {'Seed':<6} {'Best Acc':>10} {'Jain':>8} {'Rounds':>8}")
    log(f"  {'-'*58}")
    for ds_name, algo, seed in experiments:
        key = f"{ds_name}_{algo}_{seed}"
        r = checkpoint_data["completed"].get(key, {})
        if "error" in r:
            log(f"  {ds_name:<14} {algo:<10} {seed:<6} {'ERROR':>10}")
        elif r:
            acc = r.get("best_metrics", {}).get("accuracy", 0)
            jain = r.get("fairness", {}).get("jain_index", 0)
            rnds = r.get("actual_rounds", 0)
            log(f"  {ds_name:<14} {algo:<10} {seed:<6} {acc:>9.1%} {jain:>8.3f} {rnds:>8}")

    checkpoint_data["metadata"]["end_time"] = datetime.now().isoformat()
    checkpoint_data["metadata"]["total_elapsed"] = elapsed_total
    save_checkpoint(checkpoint_data)

    if _log_file:
        _log_file.close()


if __name__ == "__main__":
    main()

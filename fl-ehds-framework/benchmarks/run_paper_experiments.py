#!/usr/bin/env python3
"""
FL-EHDS Paper Experiments for FLICS 2026.

Produces all experimental results for the paper:
  P1.2 - Multi-dataset FL comparison (5 algos x 8 datasets x 3 seeds)
  P1.3 - Ablation study on chest_xray
  P1.4 - Non-IID severity study (alpha sweep: 0.1, 0.5, 1.0, 5.0)
  P2.1 - Statistical significance via paired t-tests
  P2.2 - Privacy attack evaluation (gradient inversion)
  Comm - Communication cost analysis (analytical, no training)

  New metrics: per-client fairness (Jain's index, Gini), communication costs

Usage:
    cd fl-ehds-framework
    python -m benchmarks.run_paper_experiments [--resume] [--quick] [--only p12|p13|p14|p21|p22|comm]

Output: benchmarks/paper_results/

Author: Fabio Liberti
"""

import sys
import os
import json
import time
import shutil
import tempfile
import argparse
import traceback
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Callable

import numpy as np

# Add parent to path for module imports
FRAMEWORK_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(FRAMEWORK_DIR))

import torch
import torch.nn as nn
from scipy import stats

from terminal.fl_trainer import (
    FederatedTrainer,
    ImageFederatedTrainer,
    RoundResult,
    HealthcareMLP,
    _detect_device,
)
from data.diabetes_loader import load_diabetes_data
from data.heart_disease_loader import load_heart_disease_data
from data.ptbxl_loader import load_ptbxl_data
from data.cardiovascular_loader import load_cardiovascular_data
from data.breast_cancer_loader import load_breast_cancer_data

# ======================================================================
# Configuration
# ======================================================================

SEEDS = [42, 123, 456]
OUTPUT_DIR = FRAMEWORK_DIR / "benchmarks" / "paper_results"

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
    "PTB_XL": {"num_features": 9, "num_classes": 5, "short": "PX"},
    "Cardiovascular": {"num_features": 11, "short": "CV"},
    "Breast_Cancer": {"num_features": 30, "short": "BC"},
}

ALGORITHMS = ["FedAvg", "FedLC", "FedSAM", "FedDecorr", "FedExP"]
ALGO_COLORS = ["#2196F3", "#E91E63", "#4CAF50", "#FF9800", "#9C27B0"]

IMAGING_CONFIG = dict(
    num_clients=5, num_rounds=12, local_epochs=1,
    batch_size=32, learning_rate=0.0005,
    model_type="resnet18", is_iid=False, alpha=0.5,
    freeze_backbone=True,
)

# Colab profile: optimized for GPU speed, higher quality
COLAB_IMAGING_CONFIG = dict(
    num_clients=5, num_rounds=20, local_epochs=3,
    batch_size=32, learning_rate=0.0005,
    model_type="resnet18", is_iid=False, alpha=0.5,
    freeze_backbone=False, freeze_level=1,  # partial: only conv1+bn1 frozen
    use_fedbn=True,
)

# Per-dataset overrides for imaging (merged into IMAGING_CONFIG at runtime)
DATASET_OVERRIDES = {
    "Brain_Tumor": {
        "learning_rate": 0.0003,
    },
}

COLAB_DATASET_OVERRIDES = {
    "Brain_Tumor": {
        "learning_rate": 0.0005,
    },
}

TABULAR_CONFIG = dict(
    num_rounds=20, local_epochs=3,
    batch_size=32, learning_rate=0.01,
)

# Early stopping configuration
EARLY_STOPPING_CONFIG = dict(
    enabled=True,
    patience=4,
    min_delta=0.005,
    min_rounds=6,
    metric="accuracy",
)

COLAB_EARLY_STOPPING_CONFIG = dict(
    enabled=True,
    patience=5,
    min_delta=0.003,
    min_rounds=10,
    metric="accuracy",
)


# ======================================================================
# Checkpoint / Resume
# ======================================================================

def save_checkpoint(block_name: str, data: Any) -> None:
    """Atomic checkpoint save: write to temp, backup old, rename new."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"checkpoint_{block_name}.json"
    bak_path = OUTPUT_DIR / f"checkpoint_{block_name}.json.bak"

    # Write to temp file in same directory (same FS = atomic rename)
    fd, tmp_path = tempfile.mkstemp(
        dir=str(OUTPUT_DIR),
        prefix=f".checkpoint_{block_name}_",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2, default=str)
            f.flush()
            os.fsync(f.fileno())

        # Backup existing checkpoint
        if path.exists():
            shutil.copy2(str(path), str(bak_path))

        # Atomic rename
        os.replace(tmp_path, str(path))
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def load_checkpoint(block_name: str) -> Optional[Dict]:
    """Load checkpoint with automatic fallback to .bak if primary is corrupt."""
    path = OUTPUT_DIR / f"checkpoint_{block_name}.json"
    bak_path = OUTPUT_DIR / f"checkpoint_{block_name}.json.bak"

    for p in [path, bak_path]:
        if p.exists():
            try:
                with open(p, "r") as f:
                    data = json.load(f)
                if p == bak_path:
                    print(f"  WARNING: Loaded from backup {p.name} (primary was corrupt)")
                return data
            except (json.JSONDecodeError, IOError):
                if p == path:
                    print(f"  WARNING: Checkpoint {p.name} corrupt, trying backup...")
                continue
    return None


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


class EarlyStoppingMonitor:
    """Patience-based early stopping. Tracks best metrics without altering final results."""

    def __init__(self, patience: int = 5, min_delta: float = 0.005,
                 min_rounds: int = 10, metric: str = "accuracy"):
        self.patience = patience
        self.min_delta = min_delta
        self.min_rounds = min_rounds
        self.metric = metric
        self.best_value = -float('inf')
        self.best_round = 0
        self.counter = 0
        self.best_metrics = None

    def check(self, round_num: int, metrics: dict) -> bool:
        """Returns True if training should stop."""
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

    def get_state(self) -> Dict:
        """Serialize mutable state for checkpoint."""
        return {
            "best_value": self.best_value,
            "best_round": self.best_round,
            "counter": self.counter,
            "best_metrics": self.best_metrics,
        }

    def set_state(self, state: Dict) -> None:
        """Restore mutable state from checkpoint."""
        self.best_value = state.get("best_value", -float('inf'))
        self.best_round = state.get("best_round", 0)
        self.counter = state.get("counter", 0)
        self.best_metrics = state.get("best_metrics")


# ======================================================================
# Core Training Wrappers
# ======================================================================

def _evaluate_per_client(trainer) -> Dict[str, float]:
    """Evaluate global model on each client's test data separately."""
    model = trainer.global_model
    model.eval()
    per_client = {}
    with torch.no_grad():
        for cid in range(trainer.num_clients):
            X, y = trainer.client_test_data[cid]
            if isinstance(X, np.ndarray):
                X_t = torch.FloatTensor(X).to(trainer.device)
                y_t = torch.LongTensor(y).to(trainer.device)
            else:
                X_t = X.to(trainer.device) if torch.is_tensor(X) else torch.FloatTensor(X).to(trainer.device)
                y_t = y.to(trainer.device) if torch.is_tensor(y) else torch.LongTensor(y).to(trainer.device)
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
    """Compute fairness metrics from per-client accuracies."""
    accs = list(per_client_acc.values())
    if not accs:
        return {}
    jain = (sum(accs) ** 2) / (len(accs) * sum(a ** 2 for a in accs)) if accs else 0
    # Gini coefficient
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


def run_single_imaging(
    dataset_name: str, data_dir: str, algorithm: str, seed: int,
    num_rounds: int = 15, num_clients: int = 5, local_epochs: int = 3,
    batch_size: int = 32, learning_rate: float = 0.0005,
    model_type: str = "resnet18", is_iid: bool = False, alpha: float = 0.5,
    dp_enabled: bool = False, dp_epsilon: float = 10.0, dp_clip_norm: float = 1.0,
    use_class_weights: bool = True, freeze_backbone: bool = False,
    freeze_level: int = None, use_fedbn: bool = False,
    mu: float = 0.1,
    early_stopping: Optional[Dict] = None,
    use_amp: bool = True,
    resume_from: Optional[Dict] = None,
    on_round_complete: Optional[Callable] = None,
) -> Dict[str, Any]:
    """Run a single imaging FL experiment with optional early stopping and AMP."""
    start = time.time()

    trainer = ImageFederatedTrainer(
        data_dir=data_dir,
        num_clients=num_clients,
        algorithm=algorithm,
        local_epochs=local_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        is_iid=is_iid,
        alpha=alpha,
        mu=mu,
        dp_enabled=dp_enabled,
        dp_epsilon=dp_epsilon,
        dp_clip_norm=dp_clip_norm,
        seed=seed,
        model_type=model_type,
        use_class_weights=use_class_weights,
        freeze_backbone=freeze_backbone,
        freeze_level=freeze_level,
        use_fedbn=use_fedbn,
        use_amp=use_amp,
    )
    trainer.num_rounds = num_rounds  # Enable cosine LR scheduling

    # Early stopping monitor
    es_monitor = None
    if early_stopping and early_stopping.get("enabled"):
        es_monitor = EarlyStoppingMonitor(
            patience=early_stopping.get("patience", 5),
            min_delta=early_stopping.get("min_delta", 0.005),
            min_rounds=early_stopping.get("min_rounds", 10),
            metric=early_stopping.get("metric", "accuracy"),
        )

    # Resume from round-level checkpoint if available
    start_round = 0
    history = []
    if resume_from is not None:
        ckpt_path = resume_from.get("trainer_checkpoint", "")
        if Path(ckpt_path).exists():
            start_round = trainer.load_checkpoint(ckpt_path)
            history = list(resume_from.get("history_so_far", []))
            if es_monitor and "early_stopping_state" in resume_from:
                es_monitor.set_state(resume_from["early_stopping_state"])
            print(f"[resumed from R{start_round}]", end=" ", flush=True)

    stopped_early = False
    for r in range(start_round, num_rounds):
        result = trainer.train_round(r)
        round_metrics = {
            "round": r + 1,
            "accuracy": result.global_acc,
            "loss": result.global_loss,
            "f1": result.global_f1,
            "precision": result.global_precision,
            "recall": result.global_recall,
            "auc": result.global_auc,
        }
        history.append(round_metrics)

        # Round-level checkpoint callback
        if on_round_complete is not None:
            on_round_complete(trainer, r + 1, history, es_monitor)

        # Check early stopping
        if es_monitor and es_monitor.check(r + 1, round_metrics):
            stopped_early = True
            print(f"[early stop @R{r+1}, best={es_monitor.best_value:.3f} @R{es_monitor.best_round}]", end=" ")
            break

    elapsed = time.time() - start
    final = history[-1]

    # Per-client fairness evaluation
    try:
        per_client_acc = _evaluate_per_client(trainer)
        fairness = _compute_fairness(per_client_acc)
    except Exception:
        per_client_acc = {}
        fairness = {}

    result_dict = {
        "dataset": dataset_name, "algorithm": algorithm, "seed": seed,
        "history": history,
        "final_metrics": {k: final[k] for k in ["accuracy", "loss", "f1", "precision", "recall", "auc"]},
        "per_client_acc": per_client_acc,
        "fairness": fairness,
        "runtime_seconds": round(elapsed, 1),
        "config": {
            "num_clients": num_clients, "num_rounds": num_rounds,
            "model_type": model_type, "dp_enabled": dp_enabled,
            "dp_epsilon": dp_epsilon if dp_enabled else None,
            "dp_clip_norm": dp_clip_norm if dp_enabled else None,
            "use_class_weights": use_class_weights,
            "freeze_backbone": freeze_backbone,
        },
    }

    # Add early stopping info if used
    if es_monitor:
        result_dict["stopped_early"] = stopped_early
        result_dict["actual_rounds"] = len(history)
        if es_monitor.best_metrics:
            result_dict["best_metrics"] = {
                k: es_monitor.best_metrics[k]
                for k in ["accuracy", "loss", "f1", "precision", "recall", "auc"]
                if k in es_monitor.best_metrics
            }
            result_dict["best_round"] = es_monitor.best_round

    return result_dict


def run_single_tabular(
    dataset_name: str,
    client_train: Dict[int, Tuple], client_test: Dict[int, Tuple],
    input_dim: int, algorithm: str, seed: int,
    num_rounds: int = 20, local_epochs: int = 3,
    batch_size: int = 32, learning_rate: float = 0.01,
    mu: float = 0.1,
    dp_enabled: bool = False, dp_epsilon: float = 10.0, dp_clip_norm: float = 1.0,
    num_classes: Optional[int] = None,
) -> Dict[str, Any]:
    """Run a single tabular FL experiment."""
    start = time.time()

    trainer = FederatedTrainer(
        num_clients=len(client_train),
        algorithm=algorithm,
        local_epochs=local_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        mu=mu,
        dp_enabled=dp_enabled,
        dp_epsilon=dp_epsilon,
        dp_clip_norm=dp_clip_norm,
        seed=seed,
        external_data=client_train,
        external_test_data=client_test,
        input_dim=input_dim,
        num_classes=num_classes,
    )

    history = []
    for r in range(num_rounds):
        result = trainer.train_round(r)
        history.append({
            "round": r + 1,
            "accuracy": result.global_acc,
            "loss": result.global_loss,
            "f1": result.global_f1,
            "precision": result.global_precision,
            "recall": result.global_recall,
            "auc": result.global_auc,
        })

    elapsed = time.time() - start
    final = history[-1]

    # Per-client fairness evaluation
    try:
        per_client_acc = _evaluate_per_client(trainer)
        fairness = _compute_fairness(per_client_acc)
    except Exception:
        per_client_acc = {}
        fairness = {}

    return {
        "dataset": dataset_name, "algorithm": algorithm, "seed": seed,
        "history": history,
        "final_metrics": {k: final[k] for k in ["accuracy", "loss", "f1", "precision", "recall", "auc"]},
        "per_client_acc": per_client_acc,
        "fairness": fairness,
        "runtime_seconds": round(elapsed, 1),
        "config": {
            "num_clients": len(client_train), "num_rounds": num_rounds,
            "dp_enabled": dp_enabled,
        },
    }


# ======================================================================
# P1.2: Multi-Dataset Experiments
# ======================================================================

def _cleanup_imaging_checkpoint(results: Dict, expected_rounds: int) -> int:
    """Remove imaging experiments with mismatched num_rounds (e.g. from --quick runs)."""
    to_remove = []
    for key, rec in results.get("completed", {}).items():
        if "error" in rec:
            continue
        ds = rec.get("dataset", "")
        if ds not in IMAGING_DATASETS:
            continue
        cfg_rounds = rec.get("config", {}).get("num_rounds", expected_rounds)
        if cfg_rounds != expected_rounds:
            to_remove.append(key)
    for key in to_remove:
        del results["completed"][key]
    return len(to_remove)


def run_p12_multi_dataset(resume: bool = False,
                          filter_dataset: Optional[str] = None,
                          filter_algo: Optional[str] = None,
                          use_amp: bool = True,
                          use_early_stopping: bool = True) -> Dict[str, Any]:
    """5 algorithms x 8 datasets x 3 seeds = 120 experiments."""
    block = "p12_multidataset"
    # Always load existing checkpoint to merge results incrementally
    results = load_checkpoint(block)
    if results is None:
        results = {"completed": {}}

    # Cleanup: remove imaging experiments with wrong num_rounds (e.g. from --quick)
    removed = _cleanup_imaging_checkpoint(results, IMAGING_CONFIG["num_rounds"])
    if removed > 0:
        print(f"  Checkpoint cleanup: removed {removed} experiments with mismatched rounds")
        save_checkpoint(block, results)

    all_datasets = list(IMAGING_DATASETS.keys()) + list(TABULAR_DATASETS.keys())
    run_datasets = [filter_dataset] if filter_dataset else all_datasets
    run_algos = [filter_algo] if filter_algo else ALGORITHMS

    total = len(run_algos) * len(run_datasets) * len(SEEDS)
    done = sum(1 for k, v in results["completed"].items() if "error" not in v)
    print(f"  Total experiments: {total}, already completed (all): {done}")

    es_config = EARLY_STOPPING_CONFIG if use_early_stopping else None

    # Round-level checkpoint support for imaging experiments
    trainer_ckpt_path = str(OUTPUT_DIR / ".training_state.pt")

    def _make_round_callback(exp_key, ds_name, algo, seed):
        """Create per-round checkpoint callback for an imaging experiment."""
        def on_round_complete(trainer, round_num, history, es_monitor):
            trainer.save_checkpoint(trainer_ckpt_path)
            results["in_progress"] = {
                "key": exp_key,
                "dataset": ds_name,
                "algorithm": algo,
                "seed": seed,
                "completed_rounds": round_num,
                "history_so_far": history,
                "trainer_checkpoint": trainer_ckpt_path,
                "early_stopping_state": es_monitor.get_state() if es_monitor else None,
            }
            save_checkpoint(block, results)
        return on_round_complete

    # Check for in-progress experiment to resume
    in_prog = results.get("in_progress")
    resume_info = None
    if in_prog and Path(in_prog.get("trainer_checkpoint", "")).exists():
        print(f"  Resuming in-progress: {in_prog['key']} from round {in_prog['completed_rounds']}")
        resume_info = in_prog

    count = 0
    for ds_name in run_datasets:
        if ds_name not in all_datasets:
            print(f"  WARNING: Unknown dataset '{ds_name}'. Available: {all_datasets}")
            continue
        for algo in run_algos:
            if algo not in ALGORITHMS:
                print(f"  WARNING: Unknown algorithm '{algo}'. Available: {ALGORITHMS}")
                continue
            for seed in SEEDS:
                count += 1
                key = f"{ds_name}_{algo}_{seed}"
                if key in results["completed"]:
                    continue

                print(f"  [{count}/{total}] {ds_name} / {algo} / seed={seed}", end=" ", flush=True)

                # Check if this is the in-progress experiment to resume
                this_resume = None
                if resume_info and resume_info.get("key") == key:
                    this_resume = resume_info
                    resume_info = None  # Consumed

                try:
                    if ds_name in IMAGING_DATASETS:
                        ds_info = IMAGING_DATASETS[ds_name]
                        # Merge base config with per-dataset overrides
                        img_config = {**IMAGING_CONFIG}
                        if ds_name in DATASET_OVERRIDES:
                            img_config.update(DATASET_OVERRIDES[ds_name])
                        record = run_single_imaging(
                            dataset_name=ds_name,
                            data_dir=ds_info["data_dir"],
                            algorithm=algo,
                            seed=seed,
                            mu=0.1,
                            early_stopping=es_config,
                            use_amp=use_amp,
                            resume_from=this_resume,
                            on_round_complete=_make_round_callback(key, ds_name, algo, seed),
                            **img_config,
                        )
                    elif ds_name == "Diabetes":
                        train_d, test_d, meta = load_diabetes_data(
                            num_clients=5, is_iid=False, alpha=0.5, seed=seed,
                        )
                        record = run_single_tabular(
                            dataset_name=ds_name,
                            client_train=train_d, client_test=test_d,
                            input_dim=22, algorithm=algo, seed=seed,
                            **TABULAR_CONFIG,
                        )
                    elif ds_name == "Heart_Disease":
                        train_d, test_d, meta = load_heart_disease_data(
                            num_clients=4, partition_by_hospital=True, seed=seed,
                        )
                        record = run_single_tabular(
                            dataset_name=ds_name,
                            client_train=train_d, client_test=test_d,
                            input_dim=13, algorithm=algo, seed=seed,
                            num_rounds=TABULAR_CONFIG["num_rounds"],
                            local_epochs=TABULAR_CONFIG["local_epochs"],
                            batch_size=TABULAR_CONFIG["batch_size"],
                            learning_rate=TABULAR_CONFIG["learning_rate"],
                        )
                    elif ds_name == "PTB_XL":
                        train_d, test_d, meta = load_ptbxl_data(
                            num_clients=5, partition_by_site=True, seed=seed,
                        )
                        record = run_single_tabular(
                            dataset_name=ds_name,
                            client_train=train_d, client_test=test_d,
                            input_dim=9, algorithm=algo, seed=seed,
                            num_classes=5,
                            **TABULAR_CONFIG,
                        )
                    elif ds_name == "Cardiovascular":
                        train_d, test_d, meta = load_cardiovascular_data(
                            num_clients=5, is_iid=False, alpha=0.5, seed=seed,
                        )
                        record = run_single_tabular(
                            dataset_name=ds_name,
                            client_train=train_d, client_test=test_d,
                            input_dim=11, algorithm=algo, seed=seed,
                            **TABULAR_CONFIG,
                        )
                    elif ds_name == "Breast_Cancer":
                        train_d, test_d, meta = load_breast_cancer_data(
                            num_clients=4, is_iid=False, alpha=0.5, seed=seed,
                        )
                        record = run_single_tabular(
                            dataset_name=ds_name,
                            client_train=train_d, client_test=test_d,
                            input_dim=30, algorithm=algo, seed=seed,
                            **TABULAR_CONFIG,
                        )
                    else:
                        continue

                    acc = record["final_metrics"]["accuracy"]
                    t = record["runtime_seconds"]
                    jain = record.get("fairness", {}).get("jain_index", 0)
                    es_info = f" ES@R{record['actual_rounds']}" if record.get("stopped_early") else ""
                    print(f"-> acc={acc:.3f} jain={jain:.3f} ({t:.0f}s{es_info})")
                    results["completed"][key] = record

                except Exception as e:
                    print(f"-> ERROR: {e}")
                    results["completed"][key] = {"error": str(e), "traceback": traceback.format_exc()}

                # Clear in-progress and save
                results["in_progress"] = None
                save_checkpoint(block, results)
                try:
                    Path(trainer_ckpt_path).unlink(missing_ok=True)
                except Exception:
                    pass
                _cleanup_gpu()

    return results


# ======================================================================
# P1.3: Ablation Study
# ======================================================================

def run_p13_ablation(resume: bool = False) -> Dict[str, Any]:
    """Ablation on chest_xray: clip, epsilon, model, class_weights."""
    block = "p13_ablation"
    # Always load existing checkpoint to merge results incrementally
    results = load_checkpoint(block)
    if results is None:
        results = {"completed": {}}

    data_dir = str(FRAMEWORK_DIR / "data" / "chest_xray")
    base = dict(
        dataset_name="chest_xray", data_dir=data_dir,
        algorithm="FedAvg", num_rounds=15, num_clients=5,
        local_epochs=3, batch_size=32, learning_rate=0.0005,
        model_type="resnet18",
    )

    experiments = []

    # A: Gradient clipping
    for c in [0.5, 1.0, 2.0]:
        for seed in SEEDS:
            experiments.append((
                f"clip_{c}_{seed}",
                dict(**base, seed=seed, dp_enabled=True, dp_epsilon=5.0, dp_clip_norm=c),
                f"Clip C={c}"
            ))

    # B: Privacy epsilon
    for eps in [0.5, 1.0, 2.0, 5.0, 10.0]:
        for seed in SEEDS:
            experiments.append((
                f"epsilon_{eps}_{seed}",
                dict(**base, seed=seed, dp_enabled=True, dp_epsilon=eps, dp_clip_norm=1.0),
                f"Eps={eps}"
            ))
    for seed in SEEDS:
        experiments.append((
            f"epsilon_inf_{seed}",
            dict(**base, seed=seed, dp_enabled=False),
            "Eps=inf"
        ))

    # C: Model type
    for mt in ["cnn", "resnet18"]:
        lr = 0.001 if mt == "cnn" else 0.0005
        for seed in SEEDS:
            experiments.append((
                f"model_{mt}_{seed}",
                dict(**base, seed=seed, model_type=mt, learning_rate=lr, dp_enabled=False),
                f"Model={mt}"
            ))

    # D: Class weights
    for cw in [True, False]:
        for seed in SEEDS:
            experiments.append((
                f"classweights_{cw}_{seed}",
                dict(**base, seed=seed, use_class_weights=cw, dp_enabled=False),
                f"CW={cw}"
            ))

    total = len(experiments)
    done = sum(1 for k, _, _ in experiments if k in results["completed"])
    print(f"  Total ablation experiments: {total}, already completed: {done}")

    # Round-level checkpoint support
    trainer_ckpt_path = str(OUTPUT_DIR / ".training_state.pt")

    def _make_round_callback(exp_key):
        def on_round_complete(trainer, round_num, history, es_monitor):
            trainer.save_checkpoint(trainer_ckpt_path)
            results["in_progress"] = {
                "key": exp_key,
                "completed_rounds": round_num,
                "history_so_far": history,
                "trainer_checkpoint": trainer_ckpt_path,
                "early_stopping_state": es_monitor.get_state() if es_monitor else None,
            }
            save_checkpoint(block, results)
        return on_round_complete

    in_prog = results.get("in_progress")
    resume_info = None
    if in_prog and Path(in_prog.get("trainer_checkpoint", "")).exists():
        print(f"  Resuming in-progress: {in_prog['key']} from round {in_prog['completed_rounds']}")
        resume_info = in_prog

    for i, (key, params, label) in enumerate(experiments):
        if key in results["completed"]:
            continue

        print(f"  [{i+1}/{total}] {label}, seed={params['seed']}", end=" ", flush=True)

        this_resume = None
        if resume_info and resume_info.get("key") == key:
            this_resume = resume_info
            resume_info = None

        try:
            record = run_single_imaging(
                **params,
                resume_from=this_resume,
                on_round_complete=_make_round_callback(key),
            )
            acc = record["final_metrics"]["accuracy"]
            t = record["runtime_seconds"]
            print(f"-> acc={acc:.3f} ({t:.0f}s)")
            results["completed"][key] = record
        except Exception as e:
            print(f"-> ERROR: {e}")
            results["completed"][key] = {"error": str(e)}

        results["in_progress"] = None
        save_checkpoint(block, results)
        try:
            Path(trainer_ckpt_path).unlink(missing_ok=True)
        except Exception:
            pass
        _cleanup_gpu()

    return results


# ======================================================================
# P1.4: Non-IID Severity Study (alpha sweep)
# ======================================================================

NONIID_ALPHAS = [0.1, 0.5, 1.0, 5.0]
NONIID_ALGOS = ["FedAvg", "FedProx", "SCAFFOLD"]
NONIID_DATASET = "chest_xray"


def run_p14_noniid_severity(resume: bool = False) -> Dict[str, Any]:
    """Alpha sweep: 3 algos x 4 alphas x 3 seeds = 36 experiments on chest_xray."""
    block = "p14_noniid"
    results = load_checkpoint(block)
    if results is None:
        results = {"completed": {}}

    data_dir = str(FRAMEWORK_DIR / "data" / "chest_xray")
    total = len(NONIID_ALGOS) * len(NONIID_ALPHAS) * len(SEEDS)
    done = sum(1 for v in results["completed"].values() if "error" not in v)
    print(f"  Total non-IID experiments: {total}, already completed: {done}")

    # Round-level checkpoint support
    trainer_ckpt_path = str(OUTPUT_DIR / ".training_state.pt")

    def _make_round_callback(exp_key):
        def on_round_complete(trainer, round_num, history, es_monitor):
            trainer.save_checkpoint(trainer_ckpt_path)
            results["in_progress"] = {
                "key": exp_key,
                "completed_rounds": round_num,
                "history_so_far": history,
                "trainer_checkpoint": trainer_ckpt_path,
                "early_stopping_state": es_monitor.get_state() if es_monitor else None,
            }
            save_checkpoint(block, results)
        return on_round_complete

    in_prog = results.get("in_progress")
    resume_info = None
    if in_prog and Path(in_prog.get("trainer_checkpoint", "")).exists():
        print(f"  Resuming in-progress: {in_prog['key']} from round {in_prog['completed_rounds']}")
        resume_info = in_prog

    count = 0
    for alpha in NONIID_ALPHAS:
        for algo in NONIID_ALGOS:
            for seed in SEEDS:
                count += 1
                key = f"alpha_{alpha}_{algo}_{seed}"
                if key in results["completed"]:
                    continue

                print(f"  [{count}/{total}] alpha={alpha}, {algo}, seed={seed}", end=" ", flush=True)

                this_resume = None
                if resume_info and resume_info.get("key") == key:
                    this_resume = resume_info
                    resume_info = None

                try:
                    record = run_single_imaging(
                        dataset_name=NONIID_DATASET,
                        data_dir=data_dir,
                        algorithm=algo,
                        seed=seed,
                        alpha=alpha,
                        num_clients=IMAGING_CONFIG["num_clients"],
                        num_rounds=IMAGING_CONFIG["num_rounds"],
                        local_epochs=IMAGING_CONFIG["local_epochs"],
                        batch_size=IMAGING_CONFIG["batch_size"],
                        learning_rate=IMAGING_CONFIG["learning_rate"],
                        model_type=IMAGING_CONFIG["model_type"],
                        is_iid=False,
                        mu=0.1,
                        resume_from=this_resume,
                        on_round_complete=_make_round_callback(key),
                    )
                    acc = record["final_metrics"]["accuracy"]
                    jain = record.get("fairness", {}).get("jain_index", 0)
                    t = record["runtime_seconds"]
                    print(f"-> acc={acc:.3f} jain={jain:.3f} ({t:.0f}s)")
                    results["completed"][key] = record
                except Exception as e:
                    print(f"-> ERROR: {e}")
                    results["completed"][key] = {"error": str(e)}

                results["in_progress"] = None
                save_checkpoint(block, results)
                try:
                    Path(trainer_ckpt_path).unlink(missing_ok=True)
                except Exception:
                    pass
                _cleanup_gpu()

    return results


# ======================================================================
# A1: Communication Cost Analysis (analytical)
# ======================================================================

MODEL_PARAMS = {
    "resnet18": 11_176_512,
    "cnn": 500_000,
    "mlp": 10_000,
}


def compute_communication_costs(p12_results: Dict) -> Dict[str, Any]:
    """Compute communication costs analytically from model configs."""
    costs = {}
    for key, exp in p12_results.get("completed", {}).items():
        if "error" in exp:
            continue
        config = exp.get("config", {})
        model_type = config.get("model_type", "mlp")
        n_clients = config.get("num_clients", 5)
        n_rounds = config.get("num_rounds", 15)

        params = MODEL_PARAMS.get(model_type, 10_000)
        bytes_per_update = params * 4  # float32

        # Per round: clients upload + server broadcasts
        bytes_per_round = n_clients * bytes_per_update * 2
        total_bytes = n_rounds * bytes_per_round

        costs[key] = {
            "model_type": model_type,
            "model_params": params,
            "mb_per_update": round(bytes_per_update / 1e6, 2),
            "bytes_per_round": bytes_per_round,
            "total_bytes": total_bytes,
            "total_gb": round(total_bytes / 1e9, 3),
            "topk_001_total_gb": round(total_bytes * 0.01 / 1e9, 5),
            "topk_01_total_gb": round(total_bytes * 0.1 / 1e9, 4),
        }

    # Aggregate by dataset
    summary = {}
    all_ds = list(IMAGING_DATASETS.keys()) + list(TABULAR_DATASETS.keys())
    for ds in all_ds:
        ds_costs = {k: v for k, v in costs.items() if k.startswith(f"{ds}_")}
        if ds_costs:
            first = next(iter(ds_costs.values()))
            summary[ds] = {
                "model_type": first["model_type"],
                "model_params": first["model_params"],
                "mb_per_update": first["mb_per_update"],
                "total_gb": first["total_gb"],
                "topk_001_total_gb": first["topk_001_total_gb"],
                "topk_01_total_gb": first["topk_01_total_gb"],
            }

    result = {"per_experiment": costs, "per_dataset": summary}
    save_checkpoint("comm_costs", result)
    return result


# ======================================================================
# P2.1: Statistical Significance
# ======================================================================

def run_p21_significance(p12_results: Dict[str, Any]) -> Dict[str, Any]:
    """Paired t-tests: FedAvg vs each algorithm, per dataset."""
    completed = p12_results.get("completed", {})
    all_datasets = list(IMAGING_DATASETS.keys()) + list(TABULAR_DATASETS.keys())

    sig = {}
    for ds in all_datasets:
        sig[ds] = {}

        # Gather FedAvg across seeds
        fa_vals = {"accuracy": [], "f1": [], "auc": []}
        for seed in SEEDS:
            rec = completed.get(f"{ds}_FedAvg_{seed}", {})
            fm = rec.get("final_metrics")
            if fm:
                for m in fa_vals:
                    fa_vals[m].append(fm[m])

        if len(fa_vals["accuracy"]) < 2:
            continue

        for algo in ALGORITHMS:
            if algo == "FedAvg":
                continue

            a_vals = {"accuracy": [], "f1": [], "auc": []}
            for seed in SEEDS:
                rec = completed.get(f"{ds}_{algo}_{seed}", {})
                fm = rec.get("final_metrics")
                if fm:
                    for m in a_vals:
                        a_vals[m].append(fm[m])

            if len(a_vals["accuracy"]) < 2:
                continue

            n = min(len(fa_vals["accuracy"]), len(a_vals["accuracy"]))
            entry = {}
            for m in ["accuracy", "f1", "auc"]:
                _, p = stats.ttest_rel(fa_vals[m][:n], a_vals[m][:n])
                entry[f"{m}_p"] = round(float(p), 4)
                entry[f"{m}_sig"] = p < 0.05
            sig[ds][algo] = entry

    save_checkpoint("p21_significance", sig)
    return sig


# ======================================================================
# P2.2: Privacy Attack (DLG)
# ======================================================================

def run_p22_privacy_attack() -> Dict[str, Any]:
    """DLG gradient inversion with/without DP."""
    results = {"completed": {}}

    for eps in [1.0, 5.0, 10.0, None]:
        eps_label = str(eps) if eps else "inf"
        for seed in SEEDS:
            key = f"eps_{eps_label}_{seed}"
            print(f"  Attack: eps={eps_label}, seed={seed}", end=" ", flush=True)

            try:
                mse = _gradient_inversion(epsilon=eps, seed=seed)
                print(f"-> MSE={mse:.4f}")
                results["completed"][key] = {
                    "epsilon": eps_label, "seed": seed,
                    "reconstruction_mse": round(float(mse), 6),
                }
            except Exception as e:
                print(f"-> ERROR: {e}")
                results["completed"][key] = {"error": str(e)}

    save_checkpoint("p22_attack", results)
    return results


def _gradient_inversion(
    epsilon: Optional[float], seed: int,
    num_samples: int = 32, input_dim: int = 10, num_iters: int = 300,
    clip_norm: float = 1.0,
) -> float:
    """DLG-style gradient inversion attack."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = _detect_device()

    # Real data
    X_real = torch.randn(num_samples, input_dim, device=device)
    y_real = torch.randint(0, 2, (num_samples,), device=device)

    # Model
    model = HealthcareMLP(input_dim=input_dim, num_classes=2).to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss()
    out = model(X_real)
    loss = criterion(out, y_real)

    real_grads = torch.autograd.grad(loss, model.parameters())
    real_grads = [g.detach().clone() for g in real_grads]

    # Clip
    total_norm = torch.sqrt(sum(g.norm() ** 2 for g in real_grads))
    clip_factor = min(1.0, clip_norm / (total_norm.item() + 1e-7))
    clipped = [g * clip_factor for g in real_grads]

    # DP noise
    if epsilon is not None:
        sigma = clip_norm * np.sqrt(2 * np.log(1.25 / 1e-5)) / epsilon
        target_grads = [g + torch.randn_like(g) * sigma for g in clipped]
    else:
        target_grads = clipped

    # DLG: optimize dummy data
    dummy_X = torch.randn(num_samples, input_dim, device=device, requires_grad=True)
    dummy_y = y_real.clone()  # Assume labels known (standard DLG assumption)

    optimizer = torch.optim.LBFGS([dummy_X], lr=0.1, max_iter=20)

    for _ in range(num_iters // 20):
        def closure():
            optimizer.zero_grad()
            dummy_out = model(dummy_X)
            dummy_loss = criterion(dummy_out, dummy_y)
            dummy_grads = torch.autograd.grad(dummy_loss, model.parameters(), create_graph=True)
            grad_diff = sum(((dg - tg) ** 2).sum() for dg, tg in zip(dummy_grads, target_grads))
            grad_diff.backward()
            return grad_diff
        optimizer.step(closure)

    mse = ((dummy_X.detach() - X_real) ** 2).mean().item()
    return mse


# ======================================================================
# Output Generation: LaTeX Tables
# ======================================================================

def _agg(completed: Dict, dataset: str, algo: str, metric: str = "accuracy") -> Tuple[float, float]:
    """Aggregate metric across seeds -> (mean, std)."""
    vals = []
    for seed in SEEDS:
        rec = completed.get(f"{dataset}_{algo}_{seed}", {})
        fm = rec.get("final_metrics")
        if fm and metric in fm:
            vals.append(fm[metric])
    if not vals:
        return 0.0, 0.0
    return float(np.mean(vals)), float(np.std(vals))


def _agg_ablation(completed: Dict, prefix: str, metric: str = "accuracy") -> Tuple[float, float]:
    """Aggregate ablation metric across seeds."""
    vals = []
    for seed in SEEDS:
        rec = completed.get(f"{prefix}_{seed}", {})
        fm = rec.get("final_metrics")
        if fm and metric in fm:
            vals.append(fm[metric])
    if not vals:
        return 0.0, 0.0
    return float(np.mean(vals)), float(np.std(vals))


def generate_multi_dataset_table(p12: Dict, sig: Dict) -> str:
    """Generate LaTeX table for multi-dataset results."""
    completed = p12.get("completed", {})
    all_ds = list(IMAGING_DATASETS.keys()) + list(TABULAR_DATASETS.keys())
    ds_shorts = {**{k: v["short"] for k, v in IMAGING_DATASETS.items()},
                 **{k: v["short"] for k, v in TABULAR_DATASETS.items()}}

    lines = []
    lines.append(r"\begin{table*}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Multi-Dataset Federated Learning Results}")
    lines.append(r"\label{tab:multi_dataset}")
    lines.append(r"\small")

    # Header
    ncols = 1 + len(all_ds) * 3  # algo + 3 metrics per dataset
    col_spec = "l" + "|ccc" * len(all_ds)
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")

    # Dataset names row
    header1 = r"\textbf{Algorithm}"
    for idx, ds in enumerate(all_ds):
        sep = "c|" if idx < len(all_ds) - 1 else "c"
        header1 += r" & \multicolumn{3}{" + sep + r"}{\textbf{" + ds_shorts[ds] + "}}"
    lines.append(header1 + r" \\")

    # Metric names row
    header2 = ""
    for _ in all_ds:
        header2 += r" & Acc & F1 & AUC"
    lines.append(header2 + r" \\")
    lines.append(r"\midrule")

    # Data rows
    for algo in ALGORITHMS:
        row = algo
        for ds in all_ds:
            for metric in ["accuracy", "f1", "auc"]:
                mean, std = _agg(completed, ds, algo, metric)
                if mean == 0:
                    row += " & --"
                    continue

                # Format
                if metric == "accuracy":
                    cell = f"{mean*100:.1f}"
                else:
                    cell = f"{mean:.2f}"

                # Significance marker
                if algo != "FedAvg" and ds in sig and algo in sig[ds]:
                    p = sig[ds][algo].get(f"{metric}_p", 1.0)
                    if p < 0.01:
                        cell += r"$^{**}$"
                    elif p < 0.05:
                        cell += r"$^{*}$"

                cell += r"{\tiny$\pm$" + f"{std:.2f}" + "}"
                row += f" & {cell}"
        lines.append(row + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\vspace{1mm}")
    lines.append(r"\footnotesize{Non-IID, ResNet18 (imaging) / MLP (tabular), 3 seeds. "
                 r"$^{*}p<0.05$, $^{**}p<0.01$ vs FedAvg (paired t-test). "
                 r"BT=Brain Tumor, CX=Chest X-ray, SC=Skin Cancer, DM=Diabetes, HD=Heart Disease, "
                 r"PX=PTB-XL ECG, CV=Cardiovascular, BC=Breast Cancer.}")
    lines.append(r"\end{table*}")

    return "\n".join(lines)


def generate_ablation_table(p13: Dict) -> str:
    """Generate LaTeX table for ablation study."""
    completed = p13.get("completed", {})

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Ablation Study on Chest X-ray Dataset}")
    lines.append(r"\label{tab:ablation}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{llccc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Factor} & \textbf{Value} & \textbf{Acc.(\%)} & \textbf{F1} & \textbf{AUC} \\")
    lines.append(r"\midrule")

    def _row(prefix, label):
        acc_m, acc_s = _agg_ablation(completed, prefix, "accuracy")
        f1_m, f1_s = _agg_ablation(completed, prefix, "f1")
        auc_m, auc_s = _agg_ablation(completed, prefix, "auc")
        if acc_m == 0:
            return f"& {label} & -- & -- & -- \\\\"
        return (f"& {label} & {acc_m*100:.1f}$\\pm${acc_s*100:.1f} "
                f"& {f1_m:.2f}$\\pm${f1_s:.2f} & {auc_m:.2f}$\\pm${auc_s:.2f} \\\\")

    # A: Clipping
    lines.append(r"\multirow{3}{*}{Grad. Clip $C$}")
    for c in [0.5, 1.0, 2.0]:
        lines.append(_row(f"clip_{c}", f"$C$={c}"))
    lines.append(r"\midrule")

    # B: Epsilon
    lines.append(r"\multirow{6}{*}{Privacy $\varepsilon$}")
    for eps in [0.5, 1.0, 2.0, 5.0, 10.0]:
        lines.append(_row(f"epsilon_{eps}", f"$\\varepsilon$={eps}"))
    lines.append(_row("epsilon_inf", r"$\infty$ (no DP)"))
    lines.append(r"\midrule")

    # C: Model
    lines.append(r"\multirow{2}{*}{Model}")
    lines.append(_row("model_cnn", "HealthcareCNN"))
    lines.append(_row("model_resnet18", "ResNet18"))
    lines.append(r"\midrule")

    # D: Class weights
    lines.append(r"\multirow{2}{*}{Class Wt.}")
    lines.append(_row("classweights_True", "On"))
    lines.append(_row("classweights_False", "Off"))

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\vspace{1mm}")
    lines.append(r"\footnotesize{FedAvg, 5 clients, 15 rounds, ResNet18 (unless noted). Mean$\pm$std over 3 seeds.}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def generate_attack_table(p22: Dict) -> str:
    """Generate LaTeX table for privacy attack."""
    completed = p22.get("completed", {})

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Gradient Inversion Attack: Reconstruction Quality vs. DP Noise}")
    lines.append(r"\label{tab:attack}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{lcc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{$\varepsilon$} & \textbf{Recon. MSE} & \textbf{Protection} \\")
    lines.append(r"\midrule")

    for eps_label, prot in [("inf", "None"), ("10.0", "Low"), ("5.0", "Medium"), ("1.0", "High")]:
        vals = []
        for seed in SEEDS:
            rec = completed.get(f"eps_{eps_label}_{seed}", {})
            if "reconstruction_mse" in rec:
                vals.append(rec["reconstruction_mse"])
        if vals:
            m, s = np.mean(vals), np.std(vals)
            eps_disp = r"$\infty$ (no DP)" if eps_label == "inf" else f"${eps_label}$"
            lines.append(f"{eps_disp} & ${m:.3f} \\pm {s:.3f}$ & {prot} \\\\")
        else:
            lines.append(f"{eps_label} & -- & {prot} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\vspace{1mm}")
    lines.append(r"\footnotesize{DLG attack on synthetic tabular data (32 samples, MLP). "
                 r"Higher MSE = better privacy. Mean$\pm$std over 3 seeds.}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


# ======================================================================
# Output Generation: Figures
# ======================================================================

def generate_figures(p12: Dict, p13: Dict, p22: Dict, out_dir: Path) -> None:
    """Generate all publication-quality figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 11, "font.family": "serif",
        "axes.labelsize": 12, "axes.titlesize": 13,
        "xtick.labelsize": 10, "ytick.labelsize": 10,
        "legend.fontsize": 9, "figure.dpi": 150,
    })

    if p12:
        _fig_multi_dataset_bars(p12, out_dir, plt)
        _fig_convergence(p12, out_dir, plt)

    if p13:
        _fig_epsilon_tradeoff(p13, out_dir, plt)
        _fig_model_comparison(p13, out_dir, plt)

    if p22:
        _fig_attack_mse(p22, out_dir, plt)


def _fig_multi_dataset_bars(p12, out_dir, plt):
    """Grouped bar chart: accuracy per algorithm x dataset."""
    completed = p12.get("completed", {})
    all_ds = list(IMAGING_DATASETS.keys()) + list(TABULAR_DATASETS.keys())
    ds_labels = [IMAGING_DATASETS.get(d, TABULAR_DATASETS.get(d, {})).get("short", d) for d in all_ds]

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(all_ds))
    width = 0.15

    for i, algo in enumerate(ALGORITHMS):
        means, stds = [], []
        for ds in all_ds:
            m, s = _agg(completed, ds, algo, "accuracy")
            means.append(m * 100)
            stds.append(s * 100)
        ax.bar(x + i * width, means, width, yerr=stds, label=algo,
               color=ALGO_COLORS[i], capsize=3, alpha=0.85)

    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(ds_labels)
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("Multi-Dataset Federated Learning Comparison")

    for fmt in ["pdf", "png"]:
        fig.savefig(out_dir / f"fig_multi_dataset_comparison.{fmt}", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved fig_multi_dataset_comparison")


def _fig_convergence(p12, out_dir, plt):
    """5 subplot convergence curves (accuracy vs round)."""
    completed = p12.get("completed", {})
    all_ds = list(IMAGING_DATASETS.keys()) + list(TABULAR_DATASETS.keys())

    fig, axes = plt.subplots(1, len(all_ds), figsize=(4 * len(all_ds), 4), sharey=False)
    if len(all_ds) == 1:
        axes = [axes]

    for ax, ds in zip(axes, all_ds):
        for i, algo in enumerate(ALGORITHMS):
            # Average history across seeds
            all_hist = []
            for seed in SEEDS:
                rec = completed.get(f"{ds}_{algo}_{seed}", {})
                h = rec.get("history")
                if h:
                    all_hist.append([r["accuracy"] for r in h])

            if not all_hist:
                continue

            min_len = min(len(h) for h in all_hist)
            arr = np.array([h[:min_len] for h in all_hist])
            mean = arr.mean(axis=0) * 100
            std = arr.std(axis=0) * 100
            rounds = np.arange(1, min_len + 1)

            ax.plot(rounds, mean, color=ALGO_COLORS[i], label=algo, linewidth=1.5)
            ax.fill_between(rounds, mean - std, mean + std, color=ALGO_COLORS[i], alpha=0.15)

        short = IMAGING_DATASETS.get(ds, TABULAR_DATASETS.get(ds, {})).get("short", ds)
        ax.set_title(short)
        ax.set_xlabel("Round")
        ax.grid(alpha=0.3)
        if ax == axes[0]:
            ax.set_ylabel("Accuracy (%)")

    axes[-1].legend(loc="lower right", fontsize=8)
    fig.suptitle("Convergence Curves per Dataset", fontsize=14)
    fig.tight_layout()

    for fmt in ["pdf", "png"]:
        fig.savefig(out_dir / f"fig_convergence_curves.{fmt}", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved fig_convergence_curves")


def _fig_epsilon_tradeoff(p13, out_dir, plt):
    """Privacy-utility tradeoff: epsilon vs accuracy."""
    completed = p13.get("completed", {})

    epsilons_labels = [0.5, 1.0, 2.0, 5.0, 10.0, "inf"]
    epsilons_x = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]  # inf mapped to 20 for plotting
    means, stds = [], []

    for e in epsilons_labels:
        prefix = f"epsilon_{e}"
        m, s = _agg_ablation(completed, prefix, "accuracy")
        means.append(m * 100)
        stds.append(s * 100)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(epsilons_x, means, yerr=stds, marker="o", color="#2196F3",
                capsize=5, linewidth=2, markersize=8)
    ax.set_xlabel(r"Privacy Budget $\varepsilon$")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Privacy-Utility Tradeoff (Chest X-ray)")
    ax.set_xscale("log")
    xticks = epsilons_x
    ax.set_xticks(xticks)
    ax.set_xticklabels(["0.5", "1", "2", "5", "10", r"$\infty$"])
    ax.grid(alpha=0.3)

    for fmt in ["pdf", "png"]:
        fig.savefig(out_dir / f"fig_ablation_epsilon.{fmt}", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved fig_ablation_epsilon")


def _fig_model_comparison(p13, out_dir, plt):
    """CNN vs ResNet18 comparison."""
    completed = p13.get("completed", {})

    models = ["cnn", "resnet18"]
    labels = ["HealthcareCNN", "ResNet18"]
    metrics = ["accuracy", "f1", "auc"]

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(metrics))
    width = 0.3

    for i, (mt, label) in enumerate(zip(models, labels)):
        vals = []
        errs = []
        for m in metrics:
            mean, std = _agg_ablation(completed, f"model_{mt}", m)
            vals.append(mean * 100 if m == "accuracy" else mean)
            errs.append(std * 100 if m == "accuracy" else std)
        ax.bar(x + i * width, vals, width, yerr=errs, label=label,
               color=ALGO_COLORS[i], capsize=4, alpha=0.85)

    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(["Accuracy (%)", "F1 Score", "AUC"])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("Model Architecture Comparison (Chest X-ray)")

    for fmt in ["pdf", "png"]:
        fig.savefig(out_dir / f"fig_model_comparison.{fmt}", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved fig_model_comparison")


def _fig_attack_mse(p22, out_dir, plt):
    """Gradient inversion MSE vs epsilon."""
    completed = p22.get("completed", {})

    eps_labels = ["inf", "10.0", "5.0", "1.0"]
    eps_display = [r"$\infty$", "10", "5", "1"]
    means, stds = [], []

    for e in eps_labels:
        vals = []
        for seed in SEEDS:
            rec = completed.get(f"eps_{e}_{seed}", {})
            if "reconstruction_mse" in rec:
                vals.append(rec["reconstruction_mse"])
        if vals:
            means.append(np.mean(vals))
            stds.append(np.std(vals))
        else:
            means.append(0)
            stds.append(0)

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#F44336", "#FF9800", "#FFC107", "#4CAF50"]
    bars = ax.bar(range(len(eps_labels)), means, yerr=stds,
                  color=colors, capsize=5, alpha=0.85)
    ax.set_xticks(range(len(eps_labels)))
    ax.set_xticklabels(eps_display)
    ax.set_xlabel(r"Privacy Budget $\varepsilon$")
    ax.set_ylabel("Reconstruction MSE")
    ax.set_title("Gradient Inversion Attack: DP Protection")
    ax.grid(axis="y", alpha=0.3)

    for fmt in ["pdf", "png"]:
        fig.savefig(out_dir / f"fig_attack_mse.{fmt}", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved fig_attack_mse")


# ======================================================================
# New Output: Fairness Table + Figure (A2)
# ======================================================================

def generate_fairness_table(p12: Dict) -> str:
    """Generate LaTeX table for inter-hospital fairness metrics."""
    completed = p12.get("completed", {})
    all_ds = list(IMAGING_DATASETS.keys()) + list(TABULAR_DATASETS.keys())
    ds_shorts = {**{k: v["short"] for k, v in IMAGING_DATASETS.items()},
                 **{k: v["short"] for k, v in TABULAR_DATASETS.items()}}

    lines = []
    lines.append(r"\begin{table*}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Inter-Hospital Fairness: Per-Client Accuracy Distribution}")
    lines.append(r"\label{tab:fairness}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{l" + "cccc" * len(all_ds) + "}")
    lines.append(r"\toprule")

    # Header: dataset names
    header1 = r"\textbf{Algorithm}"
    for ds in all_ds:
        header1 += r" & \multicolumn{4}{c}{\textbf{" + ds_shorts[ds] + "}}"
    lines.append(header1 + r" \\")

    # Sub-header: metrics
    header2 = ""
    for _ in all_ds:
        header2 += r" & Mean & Std & Min & Jain"
    lines.append(header2 + r" \\")
    lines.append(r"\midrule")

    for algo in ALGORITHMS:
        row = algo
        for ds in all_ds:
            # Average fairness across seeds
            means, stds, mins, jains = [], [], [], []
            for seed in SEEDS:
                rec = completed.get(f"{ds}_{algo}_{seed}", {})
                fair = rec.get("fairness", {})
                if fair:
                    means.append(fair.get("mean", 0))
                    stds.append(fair.get("std", 0))
                    mins.append(fair.get("min", 0))
                    jains.append(fair.get("jain_index", 0))
            if means:
                row += f" & {np.mean(means)*100:.1f}"
                row += f" & {np.mean(stds)*100:.1f}"
                row += f" & {np.mean(mins)*100:.1f}"
                row += f" & {np.mean(jains):.3f}"
            else:
                row += " & -- & -- & -- & --"
        lines.append(row + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\vspace{1mm}")
    lines.append(r"\footnotesize{Per-client accuracy (global model evaluated on each hospital's test set). "
                 r"Jain's Fairness Index: 1.0 = perfect equality. Mean over 3 seeds.}")
    lines.append(r"\end{table*}")

    return "\n".join(lines)


def _fig_fairness(p12, out_dir, plt):
    """Bar chart of Jain's fairness index per algo x dataset."""
    completed = p12.get("completed", {})
    all_ds = list(IMAGING_DATASETS.keys()) + list(TABULAR_DATASETS.keys())
    ds_labels = [IMAGING_DATASETS.get(d, TABULAR_DATASETS.get(d, {})).get("short", d) for d in all_ds]

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(all_ds))
    width = 0.15

    for i, algo in enumerate(ALGORITHMS):
        vals = []
        for ds in all_ds:
            jains = []
            for seed in SEEDS:
                rec = completed.get(f"{ds}_{algo}_{seed}", {})
                fair = rec.get("fairness", {})
                if fair and "jain_index" in fair:
                    jains.append(fair["jain_index"])
            vals.append(np.mean(jains) if jains else 0)
        ax.bar(x + i * width, vals, width, label=algo,
               color=ALGO_COLORS[i], alpha=0.85)

    ax.set_ylabel("Jain's Fairness Index")
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(ds_labels)
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0.5, 1.05)
    ax.set_title("Inter-Hospital Fairness Comparison")

    for fmt in ["pdf", "png"]:
        fig.savefig(out_dir / f"fig_fairness.{fmt}", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved fig_fairness")


# ======================================================================
# New Output: Non-IID Severity Table + Figure (A4)
# ======================================================================

def generate_noniid_table(p14: Dict) -> str:
    """Generate LaTeX table for non-IID severity study."""
    completed = p14.get("completed", {})

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Non-IID Severity: Accuracy vs. Dirichlet $\alpha$ (Chest X-ray)}")
    lines.append(r"\label{tab:noniid}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{l" + "cc" * len(NONIID_ALGOS) + "}")
    lines.append(r"\toprule")

    # Header
    header = r"\textbf{$\alpha$}"
    for algo in NONIID_ALGOS:
        header += r" & \multicolumn{2}{c}{\textbf{" + algo + "}}"
    lines.append(header + r" \\")

    subheader = ""
    for _ in NONIID_ALGOS:
        subheader += r" & Acc(\%) & Jain"
    lines.append(subheader + r" \\")
    lines.append(r"\midrule")

    for alpha in NONIID_ALPHAS:
        row = f"${alpha}$"
        for algo in NONIID_ALGOS:
            accs, jains = [], []
            for seed in SEEDS:
                rec = completed.get(f"alpha_{alpha}_{algo}_{seed}", {})
                fm = rec.get("final_metrics")
                fair = rec.get("fairness", {})
                if fm:
                    accs.append(fm["accuracy"])
                if fair:
                    jains.append(fair.get("jain_index", 0))
            if accs:
                row += f" & {np.mean(accs)*100:.1f}$\\pm${np.std(accs)*100:.1f}"
            else:
                row += " & --"
            if jains:
                row += f" & {np.mean(jains):.3f}"
            else:
                row += " & --"
        lines.append(row + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\vspace{1mm}")
    lines.append(r"\footnotesize{Dirichlet $\alpha$: lower = more heterogeneous. "
                 r"5 clients, 30 rounds, ResNet18. Mean$\pm$std over 3 seeds.}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def _fig_noniid_alpha(p14, out_dir, plt):
    """Line plot: alpha vs accuracy, one line per algorithm."""
    completed = p14.get("completed", {})

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    for i, algo in enumerate(NONIID_ALGOS):
        means, stds, jain_means = [], [], []
        for alpha in NONIID_ALPHAS:
            accs, jains = [], []
            for seed in SEEDS:
                rec = completed.get(f"alpha_{alpha}_{algo}_{seed}", {})
                fm = rec.get("final_metrics")
                fair = rec.get("fairness", {})
                if fm:
                    accs.append(fm["accuracy"] * 100)
                if fair:
                    jains.append(fair.get("jain_index", 0))
            means.append(np.mean(accs) if accs else 0)
            stds.append(np.std(accs) if accs else 0)
            jain_means.append(np.mean(jains) if jains else 0)

        ax1.errorbar(NONIID_ALPHAS, means, yerr=stds, marker="o",
                     color=ALGO_COLORS[i], label=algo, capsize=4, linewidth=2)
        ax2.plot(NONIID_ALPHAS, jain_means, marker="s",
                 color=ALGO_COLORS[i], label=algo, linewidth=2)

    ax1.set_xlabel(r"Dirichlet $\alpha$")
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_title("Accuracy vs Non-IID Severity")
    ax1.set_xscale("log")
    ax1.set_xticks(NONIID_ALPHAS)
    ax1.set_xticklabels([str(a) for a in NONIID_ALPHAS])
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.set_xlabel(r"Dirichlet $\alpha$")
    ax2.set_ylabel("Jain's Fairness Index")
    ax2.set_title("Fairness vs Non-IID Severity")
    ax2.set_xscale("log")
    ax2.set_xticks(NONIID_ALPHAS)
    ax2.set_xticklabels([str(a) for a in NONIID_ALPHAS])
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    for fmt in ["pdf", "png"]:
        fig.savefig(out_dir / f"fig_noniid_alpha.{fmt}", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved fig_noniid_alpha")


# ======================================================================
# New Output: Communication Cost Table + Figure (A1)
# ======================================================================

def generate_communication_table(comm: Dict) -> str:
    """Generate LaTeX table for communication costs."""
    summary = comm.get("per_dataset", {})
    if not summary:
        return ""

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Communication Cost Analysis per Dataset}")
    lines.append(r"\label{tab:communication}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{llrrrr}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Dataset} & \textbf{Model} & \textbf{Params} & "
                 r"\textbf{MB/upd} & \textbf{Full (GB)} & \textbf{Top-K 1\% (GB)} \\")
    lines.append(r"\midrule")

    ds_shorts = {**{k: v["short"] for k, v in IMAGING_DATASETS.items()},
                 **{k: v["short"] for k, v in TABULAR_DATASETS.items()}}

    for ds, info in summary.items():
        short = ds_shorts.get(ds, ds)
        params_str = f"{info['model_params']/1e6:.1f}M" if info['model_params'] > 100_000 else f"{info['model_params']/1e3:.0f}K"
        lines.append(
            f"{short} & {info['model_type']} & {params_str} & "
            f"{info['mb_per_update']:.1f} & {info['total_gb']:.2f} & {info['topk_001_total_gb']:.4f} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\vspace{1mm}")
    lines.append(r"\footnotesize{5 clients, 30 rounds (imaging) / 20 rounds (tabular). "
                 r"Full = uncompressed float32. Top-K 1\% = theoretical gradient compression.}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def _fig_communication(comm, out_dir, plt):
    """Bar chart: Full vs Top-K communication costs."""
    summary = comm.get("per_dataset", {})
    if not summary:
        return

    ds_shorts = {**{k: v["short"] for k, v in IMAGING_DATASETS.items()},
                 **{k: v["short"] for k, v in TABULAR_DATASETS.items()}}

    datasets = list(summary.keys())
    labels = [ds_shorts.get(d, d) for d in datasets]
    full_gb = [summary[d]["total_gb"] for d in datasets]
    topk1 = [summary[d]["topk_001_total_gb"] for d in datasets]
    topk10 = [summary[d]["topk_01_total_gb"] for d in datasets]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(datasets))
    width = 0.25

    ax.bar(x - width, full_gb, width, label="Full (float32)", color="#F44336", alpha=0.85)
    ax.bar(x, topk10, width, label="Top-K 10%", color="#FF9800", alpha=0.85)
    ax.bar(x + width, topk1, width, label="Top-K 1%", color="#4CAF50", alpha=0.85)

    ax.set_ylabel("Total Communication (GB)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("Communication Cost: Full vs. Gradient Compression")

    for fmt in ["pdf", "png"]:
        fig.savefig(out_dir / f"fig_communication.{fmt}", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved fig_communication")


# ======================================================================
# Main
# ======================================================================

def run_diagnostic(use_amp: bool = True) -> None:
    """Run a single Brain_Tumor FedAvg experiment to validate the fix."""
    print("\n>>> DIAGNOSTIC: Brain_Tumor FedAvg seed=42 (validating fix)")
    ds_info = IMAGING_DATASETS["Brain_Tumor"]
    img_config = {**IMAGING_CONFIG}
    if "Brain_Tumor" in DATASET_OVERRIDES:
        img_config.update(DATASET_OVERRIDES["Brain_Tumor"])

    print(f"  Config: lr={img_config.get('learning_rate')}, "
          f"freeze_backbone={img_config.get('freeze_backbone', False)}, "
          f"num_rounds={img_config['num_rounds']}, AMP={use_amp}")

    record = run_single_imaging(
        dataset_name="Brain_Tumor",
        data_dir=ds_info["data_dir"],
        algorithm="FedAvg",
        seed=42,
        mu=0.1,
        early_stopping=EARLY_STOPPING_CONFIG,
        use_amp=use_amp,
        **img_config,
    )

    # Print per-round convergence
    print(f"\n  {'Round':>5} {'Acc':>8} {'F1':>8} {'Loss':>8} {'AUC':>8}")
    print(f"  {'-'*41}")
    for h in record["history"]:
        marker = " *" if h["accuracy"] == max(r["accuracy"] for r in record["history"]) else ""
        print(f"  {h['round']:>5} {h['accuracy']:>8.4f} {h['f1']:>8.4f} "
              f"{h['loss']:>8.4f} {h['auc']:>8.4f}{marker}")

    acc = record["final_metrics"]["accuracy"]
    t = record["runtime_seconds"]
    best = record.get("best_metrics", record["final_metrics"])
    best_r = record.get("best_round", len(record["history"]))

    print(f"\n  Final: acc={acc:.4f} | Best: acc={best['accuracy']:.4f} @R{best_r}")
    print(f"  Runtime: {t:.0f}s")

    if best["accuracy"] > 0.40:
        print(f"\n  PASS: Brain_Tumor fix works (acc={best['accuracy']:.3f} > 0.40)")
        print(f"  You can now run: --only p12 --dataset Brain_Tumor --resume")
    else:
        print(f"\n  WARN: Accuracy still low ({best['accuracy']:.3f}). "
              f"Consider adjusting lr or alpha.")

    _cleanup_gpu()


# Micro-batch slice definitions: 1 dataset x 1 algo x 3 seeds (~15 min each)
SLICE_DEFINITIONS = {
    0:  {"desc": "Diagnostic: Brain_Tumor fix validation", "diagnostic": True},
    # Brain_Tumor micro-batches (1-5)
    1:  {"desc": "Brain_Tumor/FedAvg",    "dataset": "Brain_Tumor", "algo": "FedAvg"},
    2:  {"desc": "Brain_Tumor/FedLC",     "dataset": "Brain_Tumor", "algo": "FedLC"},
    3:  {"desc": "Brain_Tumor/FedSAM",    "dataset": "Brain_Tumor", "algo": "FedSAM"},
    4:  {"desc": "Brain_Tumor/FedDecorr", "dataset": "Brain_Tumor", "algo": "FedDecorr"},
    5:  {"desc": "Brain_Tumor/FedExP",    "dataset": "Brain_Tumor", "algo": "FedExP"},
    # chest_xray micro-batches (6-10)
    6:  {"desc": "chest_xray/FedAvg",     "dataset": "chest_xray", "algo": "FedAvg"},
    7:  {"desc": "chest_xray/FedLC",      "dataset": "chest_xray", "algo": "FedLC"},
    8:  {"desc": "chest_xray/FedSAM",     "dataset": "chest_xray", "algo": "FedSAM"},
    9:  {"desc": "chest_xray/FedDecorr",  "dataset": "chest_xray", "algo": "FedDecorr"},
    10: {"desc": "chest_xray/FedExP",     "dataset": "chest_xray", "algo": "FedExP"},
    # Skin_Cancer micro-batches (11-15)
    11: {"desc": "Skin_Cancer/FedAvg",    "dataset": "Skin_Cancer", "algo": "FedAvg"},
    12: {"desc": "Skin_Cancer/FedLC",     "dataset": "Skin_Cancer", "algo": "FedLC"},
    13: {"desc": "Skin_Cancer/FedSAM",    "dataset": "Skin_Cancer", "algo": "FedSAM"},
    14: {"desc": "Skin_Cancer/FedDecorr", "dataset": "Skin_Cancer", "algo": "FedDecorr"},
    15: {"desc": "Skin_Cancer/FedExP",    "dataset": "Skin_Cancer", "algo": "FedExP"},
    # Output
    99: {"desc": "Output generation (tables + figures)", "output": True},
}


def main():
    parser = argparse.ArgumentParser(description="FL-EHDS Paper Experiments (FLICS 2026)")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoints")
    parser.add_argument("--quick", action="store_true", help="Quick: 1 seed, 5 rounds")
    parser.add_argument("--only", type=str,
                        choices=["p12", "p13", "p14", "p21", "p22", "comm", "output"],
                        help="Run only one block")
    parser.add_argument("--dataset", type=str,
                        help="Filter: run only this dataset (e.g. Brain_Tumor, chest_xray, Diabetes)")
    parser.add_argument("--algo", type=str,
                        help="Filter: run only this algorithm (e.g. FedAvg, SCAFFOLD)")
    parser.add_argument("--diagnostic", action="store_true",
                        help="Run a single Brain_Tumor diagnostic to validate fix")
    parser.add_argument("--slice", type=int, choices=list(SLICE_DEFINITIONS.keys()),
                        help="Run a micro-batch slice (0=diagnostic, 1-5=Brain_Tumor, "
                             "6-10=chest_xray, 11-15=Skin_Cancer, 99=output)")
    parser.add_argument("--no-amp", action="store_true",
                        help="Disable mixed precision (AMP)")
    parser.add_argument("--no-early-stop", action="store_true",
                        help="Disable early stopping")
    parser.add_argument("--colab", action="store_true",
                        help="Use Colab GPU profile: 20 rounds, 3 epochs, FedBN, partial freeze")
    args = parser.parse_args()

    global SEEDS, IMAGING_CONFIG, TABULAR_CONFIG, DATASET_OVERRIDES, EARLY_STOPPING_CONFIG

    use_amp = not args.no_amp
    use_early_stopping = not args.no_early_stop

    # Colab profile: swap to optimized config
    if args.colab:
        IMAGING_CONFIG = COLAB_IMAGING_CONFIG.copy()
        DATASET_OVERRIDES = COLAB_DATASET_OVERRIDES.copy()
        EARLY_STOPPING_CONFIG = COLAB_EARLY_STOPPING_CONFIG.copy()

    if args.quick:
        SEEDS = [42]
        IMAGING_CONFIG["num_rounds"] = 5
        IMAGING_CONFIG["local_epochs"] = 1
        TABULAR_CONFIG["num_rounds"] = 5
        TABULAR_CONFIG["local_epochs"] = 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  FL-EHDS Paper Experiments for FLICS 2026")
    print("=" * 70)
    device = _detect_device()
    print(f"  Started:  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Device:   {device}")
    print(f"  Seeds:    {SEEDS}")
    print(f"  Output:   {OUTPUT_DIR}")
    print(f"  Resume:   {args.resume}")
    print(f"  AMP:      {use_amp}")
    print(f"  EarlyStop:{use_early_stopping}")
    if args.quick:
        print(f"  Mode:     QUICK (1 seed, 5 rounds)")
    if args.dataset:
        print(f"  Filter:   dataset={args.dataset}")
    if args.algo:
        print(f"  Filter:   algo={args.algo}")
    if args.slice is not None:
        sdef = SLICE_DEFINITIONS[args.slice]
        print(f"  Slice:    {args.slice} - {sdef['desc']}")
    if DATASET_OVERRIDES:
        print(f"  Overrides: {list(DATASET_OVERRIDES.keys())}")
    print("=" * 70)

    t0 = time.time()

    # --- Diagnostic mode ---
    if args.diagnostic or (args.slice is not None and args.slice == 0):
        run_diagnostic(use_amp=use_amp)
        total_time = time.time() - t0
        print(f"\n  Diagnostic completed in {total_time:.0f}s")
        return

    # --- Slice execution mode ---
    if args.slice is not None:
        sdef = SLICE_DEFINITIONS[args.slice]
        if sdef.get("output"):
            args.only = "output"
        else:
            args.only = "p12"
            args.dataset = sdef.get("dataset", args.dataset)
            # For multi-algo slices, run each sequentially
            algos_to_run = sdef.get("algos", [sdef["algo"]] if "algo" in sdef else [])
            if algos_to_run:
                for algo in algos_to_run:
                    print(f"\n>>> Slice {args.slice}: {sdef['desc']} ({algo})")
                    run_p12_multi_dataset(
                        resume=True,
                        filter_dataset=args.dataset,
                        filter_algo=algo,
                        use_amp=use_amp,
                        use_early_stopping=use_early_stopping,
                    )
                total_time = time.time() - t0
                print(f"\n  Slice {args.slice} completed in {total_time/3600:.1f}h ({total_time:.0f}s)")
                return
            args.resume = True

    # --- P1.2 ---
    p12 = None
    if args.only in (None, "p12"):
        print("\n>>> P1.2: Multi-Dataset FL Comparison")
        p12 = run_p12_multi_dataset(
            resume=args.resume,
            filter_dataset=args.dataset,
            filter_algo=args.algo,
            use_amp=use_amp,
            use_early_stopping=use_early_stopping,
        )

    # --- P1.3 ---
    p13 = None
    if args.only in (None, "p13"):
        print("\n>>> P1.3: Ablation Study (chest_xray)")
        p13 = run_p13_ablation(resume=args.resume)

    # --- P1.4 ---
    p14 = None
    if args.only in (None, "p14"):
        print("\n>>> P1.4: Non-IID Severity Study (alpha sweep)")
        p14 = run_p14_noniid_severity(resume=args.resume)

    # --- P2.1 ---
    sig = None
    if args.only in (None, "p21", "output"):
        print("\n>>> P2.1: Statistical Significance")
        if p12 is None:
            p12 = load_checkpoint("p12_multidataset")
        if p12:
            sig = run_p21_significance(p12)
            print(f"  Computed significance for {sum(len(v) for v in sig.values())} comparisons")
        else:
            print("  Skipped (no P1.2 results)")

    # --- P2.2 ---
    p22 = None
    if args.only in (None, "p22"):
        print("\n>>> P2.2: Privacy Attack Evaluation")
        p22 = run_p22_privacy_attack()

    # --- Communication Costs (post-hoc, no training) ---
    comm = None
    if args.only in (None, "comm", "output"):
        print("\n>>> Communication Cost Analysis")
        if p12 is None:
            p12 = load_checkpoint("p12_multidataset")
        if p12:
            comm = compute_communication_costs(p12)
            n = len(comm.get("per_dataset", {}))
            print(f"  Computed costs for {n} datasets")
        else:
            print("  Skipped (no P1.2 results)")

    # --- Output ---
    if args.only in (None, "output"):
        print("\n>>> Generating outputs...")

        if p12 is None:
            p12 = load_checkpoint("p12_multidataset")
        if p13 is None:
            p13 = load_checkpoint("p13_ablation")
        if p14 is None:
            p14 = load_checkpoint("p14_noniid")
        if p22 is None:
            p22 = load_checkpoint("p22_attack")
        if sig is None:
            sig = load_checkpoint("p21_significance")
        if comm is None:
            comm = load_checkpoint("comm_costs")

        # Tables
        if p12:
            tex = generate_multi_dataset_table(p12, sig or {})
            (OUTPUT_DIR / "table_multi_dataset.tex").write_text(tex)
            print(f"  Saved table_multi_dataset.tex")

            # Fairness table (uses P1.2 per-client data)
            tex = generate_fairness_table(p12)
            (OUTPUT_DIR / "table_fairness.tex").write_text(tex)
            print(f"  Saved table_fairness.tex")

        if p13:
            tex = generate_ablation_table(p13)
            (OUTPUT_DIR / "table_ablation.tex").write_text(tex)
            print(f"  Saved table_ablation.tex")

        if p14:
            tex = generate_noniid_table(p14)
            (OUTPUT_DIR / "table_noniid.tex").write_text(tex)
            print(f"  Saved table_noniid.tex")

        if p22:
            tex = generate_attack_table(p22)
            (OUTPUT_DIR / "table_attack.tex").write_text(tex)
            print(f"  Saved table_attack.tex")

        if comm:
            tex = generate_communication_table(comm)
            if tex:
                (OUTPUT_DIR / "table_communication.tex").write_text(tex)
                print(f"  Saved table_communication.tex")

        # Figures
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            plt.rcParams.update({
                "font.size": 11, "font.family": "serif",
                "axes.labelsize": 12, "axes.titlesize": 13,
                "xtick.labelsize": 10, "ytick.labelsize": 10,
                "legend.fontsize": 9, "figure.dpi": 150,
            })

            if p12:
                _fig_multi_dataset_bars(p12, OUTPUT_DIR, plt)
                _fig_convergence(p12, OUTPUT_DIR, plt)
                _fig_fairness(p12, OUTPUT_DIR, plt)
            if p13:
                _fig_epsilon_tradeoff(p13, OUTPUT_DIR, plt)
                _fig_model_comparison(p13, OUTPUT_DIR, plt)
            if p14:
                _fig_noniid_alpha(p14, OUTPUT_DIR, plt)
            if p22:
                _fig_attack_mse(p22, OUTPUT_DIR, plt)
            if comm:
                _fig_communication(comm, OUTPUT_DIR, plt)
        except Exception as e:
            print(f"  Figure generation error: {e}")
            traceback.print_exc()

    # --- Summary ---
    total_time = time.time() - t0
    print("\n" + "=" * 70)
    print(f"  Completed in {total_time/3600:.1f} hours ({total_time:.0f}s)")
    print(f"  Output: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()

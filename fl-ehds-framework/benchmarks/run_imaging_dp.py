#!/usr/bin/env python3
"""
FL-EHDS Imaging — Differential Privacy Evaluation.

Tests privacy-utility tradeoff on imaging datasets with central DP.
Matches tabular DP experiments (run_tabular_dp.py) for cross-modality comparison.

Experiments: 3 algos x 3 datasets x 3 epsilon x 3 seeds = 81 experiments
Estimated runtime: ~24-36 hours (can resume anytime)

Usage:
    cd fl-ehds-framework
    python -m benchmarks.run_imaging_dp [--quick] [--fresh] [--dataset chest_xray]

Output: benchmarks/paper_results_delta/checkpoint_imaging_dp.json

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
# Configuration
# ======================================================================

ALGORITHMS = ["FedAvg", "Ditto", "HPFL"]

DP_EPSILONS = [1, 5, 10]  # + No-DP baseline (dp_enabled=False)
DP_CLIP_NORM = 1.0

SEEDS = [42, 123, 456]

IMAGING_DATASETS = {
    "chest_xray": {
        "data_dir": str(FRAMEWORK_DIR / "data" / "chest_xray"),
        "num_classes": 2,
        "short": "CX",
    },
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

OUTPUT_DIR = FRAMEWORK_DIR / "benchmarks" / "paper_results_delta"
CHECKPOINT_FILE = "checkpoint_imaging_dp.json"
LOG_FILE = "experiment_imaging_dp.log"

# Same base config as other imaging experiments
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

DATASET_OVERRIDES = {
    "Brain_Tumor": {"learning_rate": 0.0005},
}


# ======================================================================
# Build experiment list: (dataset, algorithm, epsilon, seed)
# epsilon=None means No-DP baseline
# ======================================================================

def build_experiments(datasets=None):
    """Build full experiment list. datasets=None means all."""
    experiments = []
    ds_list = datasets if datasets else list(IMAGING_DATASETS.keys())
    for ds_name in ds_list:
        for algo in ALGORITHMS:
            # No-DP baseline first
            for seed in SEEDS:
                experiments.append((ds_name, algo, None, seed))
            # Then each DP epsilon
            for eps in DP_EPSILONS:
                for seed in SEEDS:
                    experiments.append((ds_name, algo, eps, seed))
    return experiments


def experiment_key(ds_name, algo, eps, seed):
    eps_str = f"eps{eps}" if eps is not None else "noDP"
    return f"{ds_name}_{algo}_{eps_str}_s{seed}"


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
# Checkpoint — atomic save with backup
# ======================================================================

def save_checkpoint(data: Dict) -> None:
    """Atomic checkpoint save with backup for crash safety."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / CHECKPOINT_FILE
    bak_path = OUTPUT_DIR / (CHECKPOINT_FILE + ".bak")
    data["metadata"]["last_save"] = datetime.now().isoformat()

    fd, tmp_path = tempfile.mkstemp(
        dir=str(OUTPUT_DIR), prefix=".ckpt_dp_", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2, default=str)
            f.flush()
            os.fsync(f.fileno())
        if path.exists():
            shutil.copy2(str(path), str(bak_path))
        os.replace(tmp_path, str(path))
        log(f"  [SAVED] checkpoint: {len(data.get('completed', {}))} experiments", also_print=False)
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
                log(f"  WARNING: corrupt checkpoint {p.name}, trying backup...")
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
    if is_hpfl and hasattr(trainer, '_hpfl_classifier_names'):
        saved_cls = {n: p.data.clone() for n, p in model.named_parameters()
                     if n in trainer._hpfl_classifier_names}
    else:
        is_hpfl = False

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
    ds_name: str, data_dir: str, algorithm: str,
    dp_epsilon: Optional[float], seed: int,
    config: Dict, es_config: Dict,
    exp_idx: int, total_exps: int,
    checkpoint_data: Dict = None, exp_key: str = None,
) -> Dict[str, Any]:
    start = time.time()
    num_rounds = config["num_rounds"]

    cfg = {**config}
    if ds_name in DATASET_OVERRIDES:
        cfg.update(DATASET_OVERRIDES[ds_name])

    # DP configuration
    dp_enabled = dp_epsilon is not None
    dp_eps = float(dp_epsilon) if dp_enabled else 10.0
    dp_label = f"eps={dp_epsilon}" if dp_enabled else "No-DP"

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
        # Differential Privacy parameters
        dp_enabled=dp_enabled,
        dp_epsilon=dp_eps,
        dp_clip_norm=DP_CLIP_NORM,
    )
    trainer.num_rounds = num_rounds

    es = EarlyStoppingMonitor(
        **{k: v for k, v in es_config.items() if k != "enabled"}
    ) if es_config.get("enabled") else None

    history = []
    best_acc = 0.0
    best_round = 0

    noise_scale = DP_CLIP_NORM / dp_eps if dp_enabled else 0.0

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

        log(f"[{exp_idx}/{total_exps}] {ds_name} | {algorithm} | {dp_label} | s{seed} | "
            f"R{r+1}/{num_rounds} | Acc:{rr.global_acc:.1%} | Best:{best_acc:.1%}(r{best_round})")

        # Save progress after EVERY round
        if checkpoint_data is not None and exp_key:
            checkpoint_data["in_progress"] = {
                "key": exp_key,
                "dataset": ds_name,
                "algorithm": algorithm,
                "seed": seed,
                "dp_epsilon": dp_epsilon,
                "round": r + 1,
                "total_rounds": num_rounds,
                "best_acc": best_acc,
                "best_round": best_round,
                "history": history,
                "elapsed_seconds": round(time.time() - start, 1),
            }
            save_checkpoint(checkpoint_data)

        if es and es.check(r + 1, {"accuracy": rr.global_acc}):
            log(f"  -> Early stop at R{r+1} (best={best_acc:.1%} at r{best_round})")
            break

    per_client_acc = _evaluate_per_client(trainer)
    fairness = _compute_fairness(per_client_acc)
    elapsed = time.time() - start

    result = {
        "dataset": ds_name,
        "algorithm": algorithm,
        "seed": seed,
        "dp_enabled": dp_enabled,
        "dp_epsilon": dp_epsilon,
        "dp_clip_norm": DP_CLIP_NORM if dp_enabled else None,
        "noise_scale": round(noise_scale, 4),
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

    # Clear in_progress when experiment completes
    if checkpoint_data is not None:
        checkpoint_data.pop("in_progress", None)

    _cleanup_gpu()
    return result


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="FL-EHDS Imaging — Differential Privacy Evaluation"
    )
    parser.add_argument("--fresh", action="store_true", help="Discard checkpoint, start fresh")
    parser.add_argument("--quick", action="store_true", help="Quick validation (3 rounds, 1 seed)")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Run only one dataset: chest_xray, Brain_Tumor, Skin_Cancer")
    args = parser.parse_args()

    global _log_file
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _log_file = open(OUTPUT_DIR / LOG_FILE, "a")

    config = {**IMAGING_CONFIG}
    es_config = {**EARLY_STOPPING}

    # Filter datasets if requested
    ds_filter = None
    if args.dataset:
        if args.dataset not in IMAGING_DATASETS:
            print(f"ERROR: unknown dataset '{args.dataset}'. Choose: {list(IMAGING_DATASETS.keys())}")
            sys.exit(1)
        ds_filter = [args.dataset]

    if args.quick:
        config["num_rounds"] = 3
        config["local_epochs"] = 1
        es_config["enabled"] = False

    experiments = build_experiments(datasets=ds_filter)
    total_exps = len(experiments)

    # Auto-resume unless --fresh
    checkpoint_data = None
    if not args.fresh:
        checkpoint_data = load_checkpoint()
        if checkpoint_data:
            done = len(checkpoint_data.get("completed", {}))
            log(f"AUTO-RESUMED: {done}/{total_exps} completed")
    elif args.fresh:
        ckpt_path = OUTPUT_DIR / CHECKPOINT_FILE
        if ckpt_path.exists():
            ckpt_path.unlink()
        log("FRESH start — previous checkpoint deleted")

    if checkpoint_data is None:
        checkpoint_data = {
            "completed": {},
            "metadata": {
                "total_experiments": total_exps,
                "purpose": "Imaging DP evaluation: privacy-utility tradeoff on CNN/ResNet18",
                "algorithms": ALGORITHMS,
                "dp_epsilons": DP_EPSILONS,
                "dp_clip_norm": DP_CLIP_NORM,
                "datasets": ds_filter or list(IMAGING_DATASETS.keys()),
                "seeds": SEEDS,
                "start_time": datetime.now().isoformat(),
                "last_save": None,
                "version": "imaging_dp_v1",
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
        ds_arg = f" --dataset {args.dataset}" if args.dataset else ""
        print(f"\n  Checkpoint salvato: {done}/{total_exps}")
        print(f"  Per riprendere: python -m benchmarks.run_imaging_dp{ds_arg}")
        sys.exit(0)

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    mode = "QUICK" if args.quick else "FULL"
    ds_label = args.dataset if args.dataset else "ALL (3 datasets)"
    log(f"\n{'='*70}")
    log(f"  FL-EHDS Imaging — Differential Privacy ({mode})")
    log(f"  {total_exps} experiments")
    log(f"  Algorithms: {ALGORITHMS}")
    log(f"  DP Epsilon: No-DP + {DP_EPSILONS} (clip_norm={DP_CLIP_NORM})")
    log(f"  Datasets: {ds_label}")
    log(f"  Seeds: {SEEDS}")
    log(f"  Device: {_detect_device(None)}")
    log(f"  Output: {OUTPUT_DIR / CHECKPOINT_FILE}")
    log(f"{'='*70}")

    global_start = time.time()
    completed_count = len(checkpoint_data.get("completed", {}))

    for exp_idx, (ds_name, algo, eps, seed) in enumerate(experiments, 1):
        key = experiment_key(ds_name, algo, eps, seed)

        if key in checkpoint_data.get("completed", {}):
            continue
        if _interrupted[0]:
            break

        dp_label = f"eps={eps}" if eps is not None else "No-DP"
        log(f"\n--- [{exp_idx}/{total_exps}] {ds_name} | {algo} | {dp_label} | seed={seed} ---")

        try:
            data_dir = IMAGING_DATASETS[ds_name]["data_dir"]
            result = run_single_imaging(
                ds_name=ds_name,
                data_dir=data_dir,
                algorithm=algo,
                dp_epsilon=eps,
                seed=seed,
                config=config,
                es_config=es_config,
                exp_idx=exp_idx,
                total_exps=total_exps,
                checkpoint_data=checkpoint_data,
                exp_key=key,
            )

            checkpoint_data["completed"][key] = result
            completed_count += 1

            # SAVE IMMEDIATELY after every experiment
            save_checkpoint(checkpoint_data)

            best_acc = result.get("best_metrics", {}).get("accuracy", 0)
            es_info = f" ES@R{result['actual_rounds']}" if result.get("stopped_early") else ""
            log(f"--- Done: {ds_name} | {algo} | {dp_label} | s{seed} | "
                f"Best={best_acc:.1%}{es_info} | {result['runtime_seconds']:.0f}s | "
                f"[{completed_count}/{total_exps}] ---")

            # ETA
            elapsed = time.time() - global_start
            if completed_count > 0:
                avg = elapsed / completed_count
                remaining = (total_exps - completed_count) * avg
                eta = str(timedelta(seconds=int(remaining)))
                log(f"  ETA: ~{eta}")

        except Exception as e:
            log(f"ERROR in {key}: {e}")
            traceback.print_exc()
            checkpoint_data["completed"][key] = {
                "dataset": ds_name, "algorithm": algo, "seed": seed,
                "dp_epsilon": eps, "error": str(e),
                "traceback": traceback.format_exc(),
            }
            # Save error entries too — never lose progress
            save_checkpoint(checkpoint_data)

    elapsed_total = time.time() - global_start
    log(f"\n{'='*70}")
    log(f"  COMPLETED: {completed_count}/{total_exps}")
    log(f"  Total time: {str(timedelta(seconds=int(elapsed_total)))}")
    log(f"{'='*70}")

    # Summary table grouped by dataset and algorithm
    log(f"\n  Privacy-Utility Summary (Best Accuracy):")
    log(f"  {'Dataset':<14} {'Algorithm':<10} {'No-DP':>8} {'eps=1':>8} {'eps=5':>8} {'eps=10':>8}")
    log(f"  {'-'*62}")

    completed = checkpoint_data.get("completed", {})
    ds_list = ds_filter or list(IMAGING_DATASETS.keys())
    for ds_name in ds_list:
        for algo in ALGORITHMS:
            row = f"  {ds_name:<14} {algo:<10}"
            for eps in [None] + DP_EPSILONS:
                key = experiment_key(ds_name, algo, eps, SEEDS[0])
                # Average across seeds
                accs = []
                for seed in SEEDS:
                    k = experiment_key(ds_name, algo, eps, seed)
                    r = completed.get(k, {})
                    if "error" not in r and r:
                        accs.append(r.get("best_metrics", {}).get("accuracy", 0))
                if accs:
                    row += f" {100*np.mean(accs):>7.1f}%"
                else:
                    row += f" {'--':>8}"
            log(row)

    checkpoint_data["metadata"]["end_time"] = datetime.now().isoformat()
    checkpoint_data["metadata"]["total_elapsed"] = elapsed_total
    save_checkpoint(checkpoint_data)
    if _log_file:
        _log_file.close()


if __name__ == "__main__":
    main()

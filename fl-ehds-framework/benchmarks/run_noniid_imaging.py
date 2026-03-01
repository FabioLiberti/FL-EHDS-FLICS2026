#!/usr/bin/env python3
"""
FL-EHDS Experiment — Non-IID Sensitivity on Imaging.

Tests how data heterogeneity (Dirichlet alpha) affects imaging FL
performance. Lower alpha = more heterogeneous data across hospitals.

Key question: Does HPFL's personalization advantage emerge/disappear
at different heterogeneity levels on imaging data?

Design:
  - Datasets: Brain Tumor (4-class), Skin Cancer (binary), Chest X-ray (binary)
  - Algorithms: FedAvg, HPFL
  - Alpha: 0.1 (highly non-IID), 0.5 (moderate), 1.0 (mild)
  - Seeds: 42, 123, 456
  - Total: 3 datasets x 2 algos x 3 alpha x 3 seeds = 54 experiments (~5h on M3)

Usage:
    cd fl-ehds-framework
    python -m benchmarks.run_noniid_imaging [--quick] [--fresh]

Output: benchmarks/paper_results_delta/checkpoint_noniid_imaging.json

Author: Fabio Liberti
"""

import sys
import os
import json
import time
import gc
import shutil
import signal
import tempfile
import argparse
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Any
from collections import defaultdict

import numpy as np

FRAMEWORK_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(FRAMEWORK_DIR))

import torch

from terminal.fl_trainer import ImageFederatedTrainer, _detect_device

# ======================================================================
# Configuration
# ======================================================================

IMAGING_DATASETS = {
    "Brain_Tumor": {
        "data_dir": str(FRAMEWORK_DIR / "data" / "Brain_Tumor"),
        "num_classes": 4,
        "short": "BT",
    },
    "Skin_Cancer": {
        "data_dir": str(FRAMEWORK_DIR / "data" / "Skin Cancer"),  # space in dir name!
        "num_classes": 2,
        "short": "SC",
    },
    "chest_xray": {
        "data_dir": str(FRAMEWORK_DIR / "data" / "chest_xray"),
        "num_classes": 2,
        "short": "CX",
    },
}

ALGORITHMS = ["FedAvg", "HPFL"]
ALPHAS = [0.1, 0.5, 1.0]
SEEDS = [42, 123, 456]

IMAGING_CONFIG = dict(
    num_clients=5,
    num_rounds=15,
    local_epochs=2,
    batch_size=32,
    learning_rate=0.001,
    model_type="resnet18",
    is_iid=False,  # alpha controlled per experiment
    freeze_backbone=False,
    freeze_level=2,
    use_fedbn=True,
    use_class_weights=True,
    use_amp=True,
    mu=0.1,
)

OUTPUT_DIR = FRAMEWORK_DIR / "benchmarks" / "paper_results_delta"
CHECKPOINT_FILE = "checkpoint_noniid_imaging.json"
LOG_FILE = "experiment_noniid_imaging.log"


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
    # type: (Dict) -> None
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / CHECKPOINT_FILE
    bak = OUTPUT_DIR / (CHECKPOINT_FILE + ".bak")
    data["metadata"]["last_save"] = datetime.now().isoformat()

    fd, tmp = tempfile.mkstemp(
        dir=str(OUTPUT_DIR), prefix=".noniid_img_", suffix=".tmp"
    )
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
    # type: () -> Optional[Dict]
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
# Early Stopping
# ======================================================================

class EarlyStoppingMonitor:
    def __init__(self, patience=4, min_delta=0.003, min_rounds=6, metric="accuracy"):
        self.patience = patience
        self.min_delta = min_delta
        self.min_rounds = min_rounds
        self.metric = metric
        self.best_value = -float("inf")
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

def _evaluate_per_client(trainer):
    # type: (Any) -> Dict[str, float]
    model = trainer.global_model
    model.eval()
    per_client = {}

    is_hpfl = trainer.algorithm == "HPFL"
    if is_hpfl and hasattr(trainer, "_hpfl_classifier_names"):
        saved_cls = {
            n: p.data.clone()
            for n, p in model.named_parameters()
            if n in trainer._hpfl_classifier_names
        }

    with torch.no_grad():
        for cid in range(trainer.num_clients):
            if is_hpfl and hasattr(trainer, "_hpfl_classifier_names"):
                for n, p in model.named_parameters():
                    if n in trainer._hpfl_classifier_names:
                        p.data.copy_(trainer.client_classifiers[cid][n])

            X, y = trainer.client_test_data[cid]
            if isinstance(X, np.ndarray):
                X_t = torch.FloatTensor(X).to(trainer.device)
            else:
                X_t = X.to(trainer.device)
            if isinstance(y, np.ndarray):
                y_t = torch.LongTensor(y).to(trainer.device)
            else:
                y_t = y.to(trainer.device)

            correct = 0
            total = 0
            bs = 64
            for i in range(0, len(y_t), bs):
                out = model(X_t[i : i + bs])
                preds = out.argmax(dim=1)
                correct += (preds == y_t[i : i + bs]).sum().item()
                total += len(y_t[i : i + bs])
            per_client[str(cid)] = correct / total if total > 0 else 0.0

    if is_hpfl and hasattr(trainer, "_hpfl_classifier_names"):
        for n, p in model.named_parameters():
            if n in trainer._hpfl_classifier_names:
                p.data.copy_(saved_cls[n])

    return per_client


def _compute_fairness(per_client_acc):
    # type: (Dict[str, float]) -> Dict[str, float]
    accs = list(per_client_acc.values())
    if not accs:
        return {}
    jain = (sum(accs) ** 2) / (len(accs) * sum(a ** 2 for a in accs)) if accs else 0
    sorted_a = sorted(accs)
    n = len(sorted_a)
    cumsum = np.cumsum(sorted_a)
    gini = (
        (2 * sum((i + 1) * v for i, v in enumerate(sorted_a)))
        / (n * cumsum[-1])
        - (n + 1) / n
    )
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
# Training — single experiment
# ======================================================================

def run_single_experiment(ds_name, ds_config, algo, alpha_val, seed, cfg,
                          exp_idx, total_exps):
    # type: (str, Dict, str, float, int, Dict, int, int) -> Dict[str, Any]
    """Run one imaging experiment at a specific alpha value."""
    start = time.time()
    num_rounds = cfg["num_rounds"]

    trainer = ImageFederatedTrainer(
        data_dir=ds_config["data_dir"],
        num_clients=cfg["num_clients"],
        algorithm=algo,
        local_epochs=cfg["local_epochs"],
        batch_size=cfg["batch_size"],
        learning_rate=cfg["learning_rate"],
        is_iid=False,
        alpha=alpha_val,
        mu=cfg["mu"],
        seed=seed,
        model_type=cfg["model_type"],
        freeze_backbone=cfg.get("freeze_backbone", False),
        freeze_level=cfg.get("freeze_level"),
        use_fedbn=cfg.get("use_fedbn", False),
        use_class_weights=cfg.get("use_class_weights", True),
        use_amp=cfg.get("use_amp", True),
    )
    trainer.num_rounds = num_rounds

    es = EarlyStoppingMonitor(patience=4, min_delta=0.003, min_rounds=6)

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
            "precision": getattr(rr, "global_precision", 0.0),
            "recall": getattr(rr, "global_recall", 0.0),
            "auc": getattr(rr, "global_auc", 0.0),
        }
        history.append(metrics)

        if rr.global_acc > best_acc:
            best_acc = rr.global_acc
            best_round = r + 1

        log("[{}/{}] {} | {} | a={} | s{} | R{}/{} | Acc:{:.1%} | Best:{:.1%}(r{})".format(
            exp_idx, total_exps, ds_name, algo, alpha_val, seed,
            r + 1, num_rounds, rr.global_acc, best_acc, best_round))

        if es.check(r + 1, {"accuracy": rr.global_acc}):
            log("  -> Early stop at R{} (best={:.1%} at r{})".format(
                r + 1, best_acc, best_round))
            break

    per_client_acc = _evaluate_per_client(trainer)
    fairness = _compute_fairness(per_client_acc)
    elapsed = time.time() - start

    result = {
        "dataset": ds_name,
        "algorithm": algo,
        "alpha": alpha_val,
        "seed": seed,
        "history": history,
        "final_metrics": history[-1] if history else {},
        "per_client_acc": per_client_acc,
        "fairness": fairness,
        "runtime_seconds": round(elapsed, 1),
        "config": {
            "num_clients": cfg["num_clients"],
            "num_rounds": num_rounds,
            "local_epochs": cfg["local_epochs"],
            "batch_size": cfg["batch_size"],
            "learning_rate": cfg["learning_rate"],
            "model_type": cfg["model_type"],
            "alpha": alpha_val,
        },
        "stopped_early": es.counter >= es.patience,
        "actual_rounds": len(history),
        "best_metrics": {"accuracy": best_acc, "round": best_round},
        "best_round": best_round,
    }

    del trainer
    _cleanup_gpu()
    return result


# ======================================================================
# Figure generation
# ======================================================================

def generate_figures(checkpoint_data):
    """Line chart: one subplot per dataset, x=alpha, y=accuracy, lines per algo."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log("  matplotlib not available -- skipping figures")
        return

    completed = checkpoint_data.get("completed", {})
    valid = {k: v for k, v in completed.items() if "error" not in v}
    if not valid:
        log("  No valid results to plot")
        return

    # Collect: dataset -> algo -> alpha -> [accuracies across seeds]
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for key, res in valid.items():
        ds = res["dataset"]
        algo = res["algorithm"]
        alpha = res["alpha"]
        acc = res.get("best_metrics", {}).get("accuracy", 0)
        data[ds][algo][alpha].append(acc)

    ds_names = list(IMAGING_DATASETS.keys())
    num_ds = len(ds_names)

    fig, axes = plt.subplots(1, num_ds, figsize=(6 * num_ds, 5), squeeze=False)
    axes = axes[0]

    algo_colors = {"FedAvg": "#2196F3", "HPFL": "#E91E63"}
    algo_markers = {"FedAvg": "o", "HPFL": "s"}

    for col, ds_name in enumerate(ds_names):
        ax = axes[col]
        ds_short = IMAGING_DATASETS[ds_name]["short"]

        for algo in ALGORITHMS:
            alphas_sorted = sorted(data[ds_name][algo].keys())
            if not alphas_sorted:
                continue

            means = []
            stds = []
            for a in alphas_sorted:
                vals = data[ds_name][algo][a]
                means.append(np.mean(vals) * 100)
                stds.append(np.std(vals) * 100)

            means = np.array(means)
            stds = np.array(stds)
            color = algo_colors.get(algo, "#666666")
            marker = algo_markers.get(algo, "^")

            ax.errorbar(
                alphas_sorted, means, yerr=stds,
                label=algo, color=color, marker=marker,
                linewidth=2, markersize=8, capsize=5, capthick=1.5,
            )

        ax.set_xlabel("Dirichlet Alpha", fontsize=12)
        if col == 0:
            ax.set_ylabel("Accuracy (%)", fontsize=12)
        ax.set_title("{} ({})".format(ds_name.replace("_", " "), ds_short),
                     fontsize=13, fontweight="bold")
        ax.set_xticks(ALPHAS)
        ax.set_xticklabels([str(a) for a in ALPHAS])
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Non-IID Sensitivity on Imaging Datasets\n"
        "(lower alpha = higher heterogeneity)",
        fontsize=14, fontweight="bold", y=1.03,
    )
    plt.tight_layout()

    png_path = OUTPUT_DIR / "noniid_imaging_alpha_sensitivity.png"
    pdf_path = OUTPUT_DIR / "noniid_imaging_alpha_sensitivity.pdf"
    plt.savefig(str(png_path), dpi=200, bbox_inches="tight")
    plt.savefig(str(pdf_path), bbox_inches="tight")
    plt.close(fig)
    log("  Figure saved: {}".format(png_path))
    log("  Figure saved: {}".format(pdf_path))


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="FL-EHDS Non-IID Sensitivity on Imaging Datasets"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick validation (1 seed, 1 dataset, 1 algo, 2 alphas)",
    )
    parser.add_argument(
        "--fresh", action="store_true",
        help="Start fresh (delete existing checkpoint)",
    )
    args = parser.parse_args()

    global _log_file
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _log_file = open(OUTPUT_DIR / LOG_FILE, "a")

    # Determine experiment grid
    if args.quick:
        datasets = {"Brain_Tumor": IMAGING_DATASETS["Brain_Tumor"]}
        algos = ["FedAvg"]
        alphas = [0.1, 1.0]
        seeds = [42]
    else:
        datasets = IMAGING_DATASETS
        algos = ALGORITHMS
        alphas = ALPHAS
        seeds = SEEDS

    cfg = IMAGING_CONFIG.copy()

    # Build experiment list
    experiments = []
    for ds_name, ds_config in datasets.items():
        ds_short = ds_config["short"]
        for algo in algos:
            for alpha_val in alphas:
                for seed in seeds:
                    key = "{}_{}_{}_s{}".format(
                        ds_short, algo,
                        "a{}".format(alpha_val),
                        seed,
                    )
                    experiments.append({
                        "key": key,
                        "ds_name": ds_name,
                        "ds_config": ds_config,
                        "algo": algo,
                        "alpha": alpha_val,
                        "seed": seed,
                    })

    total_exps = len(experiments)

    # Handle --fresh
    if args.fresh:
        ckpt_path = OUTPUT_DIR / CHECKPOINT_FILE
        if ckpt_path.exists():
            ckpt_path.unlink()
        log("FRESH start -- checkpoint deleted")

    # Load or create checkpoint
    checkpoint_data = load_checkpoint()
    if checkpoint_data is None:
        checkpoint_data = {
            "completed": {},
            "metadata": {
                "experiment": "noniid_imaging_sensitivity",
                "total_experiments": total_exps,
                "algorithms": list(algos),
                "alphas": list(alphas),
                "datasets": list(datasets.keys()),
                "seeds": list(seeds),
                "imaging_config": cfg,
                "start_time": datetime.now().isoformat(),
                "last_save": None,
            },
        }
    else:
        done = len(checkpoint_data.get("completed", {}))
        log("AUTO-RESUMED: {}/{} completed".format(done, total_exps))

    # Graceful shutdown on SIGINT / SIGTERM
    _interrupted = [False]

    def _signal_handler(signum, frame):
        if _interrupted[0]:
            sys.exit(1)
        _interrupted[0] = True
        done = len(checkpoint_data.get("completed", {}))
        save_checkpoint(checkpoint_data)
        log("\n  Checkpoint saved: {}/{} completed".format(done, total_exps))
        log("  Resume: python -m benchmarks.run_noniid_imaging")
        sys.exit(0)

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # Print banner
    mode = "QUICK" if args.quick else "FULL"
    log("\n" + "=" * 66)
    log("  FL-EHDS Non-IID Sensitivity on Imaging ({})".format(mode))
    log("  {} experiments = {} datasets x {} algos x {} alphas x {} seeds".format(
        total_exps, len(datasets), len(algos), len(alphas), len(seeds)))
    log("=" * 66)
    log("  Device:       {}".format(_detect_device(None)))
    log("  Datasets:     {}".format(list(datasets.keys())))
    log("  Algorithms:   {}".format(list(algos)))
    log("  Alphas:       {}".format(list(alphas)))
    log("  Seeds:        {}".format(list(seeds)))
    log("  num_rounds:   {}".format(cfg["num_rounds"]))
    log("  local_epochs: {}".format(cfg["local_epochs"]))
    log("  freeze_level: {}".format(cfg["freeze_level"]))
    log("  Output:       {}".format(OUTPUT_DIR / CHECKPOINT_FILE))
    if args.quick:
        log("  Mode:         QUICK (1 seed, 1 dataset, 1 algo, 2 alphas)")
    log("=" * 66 + "\n")

    global_start = time.time()
    completed_count = len(checkpoint_data.get("completed", {}))

    for exp_idx, exp in enumerate(experiments, 1):
        key = exp["key"]

        # Skip already completed
        if key in checkpoint_data.get("completed", {}):
            continue

        if _interrupted[0]:
            break

        ds_name = exp["ds_name"]
        ds_config = exp["ds_config"]
        algo = exp["algo"]
        alpha_val = exp["alpha"]
        seed = exp["seed"]

        log("\n--- [{}/{}] {} | {} | alpha={} | seed={} ---".format(
            exp_idx, total_exps, ds_name, algo, alpha_val, seed))

        try:
            result = run_single_experiment(
                ds_name=ds_name,
                ds_config=ds_config,
                algo=algo,
                alpha_val=alpha_val,
                seed=seed,
                cfg=cfg,
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
            log("--- Done: {} | {} | a={} | s{} | Best={:.1%}{} | {:.0f}s | [{}/{}] ---".format(
                ds_name, algo, alpha_val, seed,
                best_acc, es_info, result["runtime_seconds"],
                completed_count, total_exps))

            # ETA
            elapsed = time.time() - global_start
            if completed_count > 0:
                avg = elapsed / completed_count
                remaining = (total_exps - completed_count) * avg
                eta = str(timedelta(seconds=int(remaining)))
                log("  ETA: ~{}".format(eta))

        except Exception as e:
            log("ERROR in {}: {}".format(key, e))
            traceback.print_exc()
            checkpoint_data["completed"][key] = {
                "dataset": ds_name,
                "algorithm": algo,
                "alpha": alpha_val,
                "seed": seed,
                "error": str(e),
                "traceback": traceback.format_exc(),
            }
            save_checkpoint(checkpoint_data)

        _cleanup_gpu()

    # Final save
    elapsed_total = time.time() - global_start
    checkpoint_data["metadata"]["end_time"] = datetime.now().isoformat()
    checkpoint_data["metadata"]["total_elapsed"] = elapsed_total
    save_checkpoint(checkpoint_data)

    # ================================================================
    # Summary table
    # ================================================================
    log("\n" + "=" * 66)
    log("  COMPLETED: {}/{}".format(completed_count, total_exps))
    log("  Total time: {}".format(timedelta(seconds=int(elapsed_total))))
    log("=" * 66)

    completed = checkpoint_data.get("completed", {})
    valid = {k: v for k, v in completed.items() if "error" not in v}

    # Print summary: dataset -> algo -> alpha -> mean +/- std accuracy
    log("\n  Non-IID Sensitivity Summary (Accuracy %):")
    log("  {:<15s} {:<10s} {:>12s} {:>12s} {:>12s}".format(
        "Dataset", "Algorithm", "a=0.1", "a=0.5", "a=1.0"))
    log("  " + "-" * 63)

    summary = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for key, res in valid.items():
        ds = res["dataset"]
        algo = res["algorithm"]
        alpha = res["alpha"]
        acc = res.get("best_metrics", {}).get("accuracy", 0)
        summary[ds][algo][alpha].append(acc)

    for ds_name in datasets:
        for algo in algos:
            cells = []
            for a in [0.1, 0.5, 1.0]:
                vals = summary[ds_name][algo].get(a, [])
                if vals:
                    m = np.mean(vals) * 100
                    s = np.std(vals) * 100
                    cells.append("{:.1f}+/-{:.1f}".format(m, s))
                else:
                    cells.append("---")
            log("  {:<15s} {:<10s} {:>12s} {:>12s} {:>12s}".format(
                ds_name, algo, cells[0], cells[1], cells[2]))

    # Generate figures
    if valid:
        log("\n  Generating figures...")
        generate_figures(checkpoint_data)

    log("\n  Checkpoint: {}".format(OUTPUT_DIR / CHECKPOINT_FILE))
    log("=" * 66)

    if _log_file:
        _log_file.close()


if __name__ == "__main__":
    main()

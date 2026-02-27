#!/usr/bin/env python3
"""
FL-EHDS — Confusion Matrix on Chest X-Ray (FedAvg vs HPFL).

Retrains FedAvg and HPFL on chest_xray and generates confusion matrices
to diagnose HOW HPFL fails on imaging (majority-class collapse? client-level
failure? random predictions?).

Total: 2 algos × 3 seeds = 6 experiments (~30-60 min)

Usage:
    cd fl-ehds-framework
    python -m benchmarks.run_confusion_matrix_chest [--quick]

Output:
    benchmarks/paper_results_delta/confusion_matrix_chest.pdf
    benchmarks/paper_results_delta/checkpoint_confusion_chest.json

Author: Fabio Liberti
"""

import sys
import os
import json
import time
import gc
import signal
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

FRAMEWORK_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(FRAMEWORK_DIR))

import torch
import torch.nn as nn

from terminal.fl_trainer import (
    ImageFederatedTrainer,
    _detect_device,
)

OUTPUT_DIR = FRAMEWORK_DIR / "benchmarks" / "paper_results_delta"
DATA_DIR = str(FRAMEWORK_DIR / "data" / "chest_xray")

# Same config as the extended chest runs
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

ALGORITHMS = ["FedAvg", "HPFL"]
SEEDS = [42, 123, 456]
CLASS_NAMES = {0: "NORMAL", 1: "PNEUMONIA"}


class EarlyStoppingMonitor:
    def __init__(self, patience=4, min_delta=0.003, min_rounds=8, metric="accuracy"):
        self.patience = patience
        self.min_delta = min_delta
        self.min_rounds = min_rounds
        self.metric = metric
        self.best_value = -float('inf')
        self.counter = 0

    def check(self, round_num, metrics):
        value = metrics.get(self.metric, 0)
        if value > self.best_value + self.min_delta:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
        if round_num < self.min_rounds:
            return False
        return self.counter >= self.patience


def collect_predictions(trainer) -> Tuple[np.ndarray, np.ndarray, Dict[int, Tuple[np.ndarray, np.ndarray]]]:
    """Collect all predictions + per-client predictions after training.

    Returns:
        all_labels, all_preds (aggregate), per_client dict {cid: (labels, preds)}
    """
    model = trainer.global_model
    model.eval()

    all_preds = []
    all_labels = []
    per_client = {}  # cid -> (labels, preds)

    is_hpfl = trainer.algorithm == "HPFL"
    if is_hpfl:
        saved_cls = {n: p.data.clone() for n, p in model.named_parameters()
                     if n in trainer._hpfl_classifier_names}

    with torch.no_grad():
        for cid in range(trainer.num_clients):
            # HPFL: swap in client-specific classifier
            if is_hpfl:
                for n, p in model.named_parameters():
                    if n in trainer._hpfl_classifier_names:
                        p.data.copy_(trainer.client_classifiers[cid][n])

            X, y = trainer.client_test_data[cid]
            client_preds = []
            client_labels = []

            for i in range(0, len(y), trainer.batch_size):
                X_batch = torch.FloatTensor(X[i:i+trainer.batch_size]).to(trainer.device)
                y_batch = y[i:i+trainer.batch_size]

                if trainer.use_amp and hasattr(torch, 'autocast'):
                    with torch.autocast(device_type=str(trainer.device).split(':')[0], dtype=torch.float16):
                        out = model(X_batch)
                else:
                    out = model(X_batch)

                preds = out.argmax(dim=1).cpu().numpy()
                labels = y_batch if isinstance(y_batch, np.ndarray) else y_batch.numpy()

                client_preds.extend(preds.tolist())
                client_labels.extend(labels.tolist())

            per_client[cid] = (np.array(client_labels), np.array(client_preds))
            all_preds.extend(client_preds)
            all_labels.extend(client_labels)

    # Restore global classifier for HPFL
    if is_hpfl:
        for n, p in model.named_parameters():
            if n in trainer._hpfl_classifier_names:
                p.data.copy_(saved_cls[n])

    return np.array(all_labels), np.array(all_preds), per_client


def compute_confusion_matrix(y_true, y_pred, num_classes=2):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def run_experiment(algorithm, seed, config, quick=False):
    """Train and collect predictions."""
    num_rounds = 3 if quick else config["num_rounds"]
    local_epochs = 1 if quick else config["local_epochs"]

    trainer = ImageFederatedTrainer(
        data_dir=DATA_DIR,
        num_clients=config["num_clients"],
        algorithm=algorithm,
        local_epochs=local_epochs,
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

    es = None
    if not quick and EARLY_STOPPING.get("enabled"):
        es = EarlyStoppingMonitor(
            **{k: v for k, v in EARLY_STOPPING.items() if k != "enabled"}
        )

    best_acc = 0.0
    best_round = 0
    for r in range(num_rounds):
        rr = trainer.train_round(r)
        if rr.global_acc > best_acc:
            best_acc = rr.global_acc
            best_round = r + 1

        print("    R{}/{} | Acc:{:.1%} | Best:{:.1%}(r{})".format(
            r+1, num_rounds, rr.global_acc, best_acc, best_round), flush=True)

        if es and es.check(r + 1, {"accuracy": rr.global_acc}):
            print("    -> Early stop at R{}".format(r+1))
            break

    # Collect predictions
    y_true, y_pred, per_client = collect_predictions(trainer)

    # Cleanup
    del trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        try:
            torch.mps.empty_cache()
        except Exception:
            pass
    gc.collect()

    return y_true, y_pred, per_client, best_acc, best_round


def _save_checkpoint_atomic(data, path):
    """Atomic checkpoint save with backup."""
    path = Path(path)
    bak = path.with_suffix(".json.bak")
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".cm_chest_", suffix=".tmp")
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


def _load_checkpoint(path):
    """Load checkpoint if exists."""
    path = Path(path)
    bak = path.with_suffix(".json.bak")
    for p in [path, bak]:
        if p.exists():
            try:
                with open(p) as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                continue
    return None


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Confusion Matrix for Chest X-Ray (FedAvg vs HPFL)")
    parser.add_argument("--quick", action="store_true", help="Quick validation (1 seed, 3 rounds)")
    parser.add_argument("--fresh", action="store_true", help="Start fresh (delete old checkpoint)")
    args = parser.parse_args()

    seeds = [42] if args.quick else SEEDS
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = OUTPUT_DIR / "checkpoint_confusion_chest.json"

    # Handle fresh start
    if args.fresh and ckpt_path.exists():
        ckpt_path.unlink()
        print("  Deleted old checkpoint (--fresh)")

    # Load existing checkpoint for resume
    existing = _load_checkpoint(ckpt_path)
    completed_keys = set()
    if existing and "completed" in existing:
        completed_keys = set(existing["completed"].keys())
        print("  RESUMED: {} experiments already completed".format(len(completed_keys)))
    elif existing and "results" in existing:
        # Old format: convert to understand completed
        for algo, runs in existing.get("results", {}).items():
            for run in runs:
                completed_keys.add("{}_s{}".format(algo, run["seed"]))

    total_exps = len(ALGORITHMS) * len(seeds)

    print("=" * 60)
    print("  FL-EHDS Confusion Matrix — Chest X-Ray")
    print("  {} algos x {} seeds = {} experiments".format(
        len(ALGORITHMS), len(seeds), total_exps))
    print("  Device: {}".format(_detect_device()))
    print("  Mode: {}".format("QUICK" if args.quick else "FULL"))
    print("=" * 60)

    # Checkpoint data structure
    checkpoint = {
        "completed": existing.get("completed", {}) if existing else {},
        "results": existing.get("results", {}) if existing else {},
        "aggregated_cms": {},
        "per_client_cms": {},
        "metadata": {
            "algorithms": ALGORITHMS,
            "seeds": seeds,
            "class_names": CLASS_NAMES,
            "dataset": "chest_xray",
            "config": IMAGING_CONFIG,
            "timestamp": datetime.now().isoformat(),
        }
    }

    # SIGINT handler
    _interrupted = [False]
    def _signal_handler(signum, frame):
        if _interrupted[0]:
            sys.exit(1)
        _interrupted[0] = True
        done = len(checkpoint.get("completed", {}))
        _save_checkpoint_atomic(checkpoint, ckpt_path)
        print("\n  SIGINT — checkpoint salvato: {}/{} completati".format(done, total_exps))
        print("  Per riprendere: python -m benchmarks.run_confusion_matrix_chest")
        sys.exit(0)
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    results = checkpoint["results"]
    done_count = len(checkpoint.get("completed", {}))

    for algo in ALGORITHMS:
        if algo not in results:
            results[algo] = []

        for seed in seeds:
            key = "{}_s{}".format(algo, seed)
            if key in checkpoint.get("completed", {}):
                continue

            print("\n  [{}/{}] {} seed={} ...".format(done_count + 1, total_exps, algo, seed), flush=True)
            start = time.time()

            y_true, y_pred, per_client, best_acc, best_round = run_experiment(
                algo, seed, IMAGING_CONFIG, quick=args.quick
            )

            cm = compute_confusion_matrix(y_true, y_pred, 2)
            elapsed = time.time() - start

            # Per-client CMs
            client_cms = {}
            for cid, (cl, cp) in per_client.items():
                ccm = compute_confusion_matrix(cl, cp, 2)
                client_cms[cid] = ccm.tolist()

            recall_normal = cm[0, 0] / cm[0].sum() if cm[0].sum() > 0 else 0
            recall_pneumonia = cm[1, 1] / cm[1].sum() if cm[1].sum() > 0 else 0
            acc = (cm[0, 0] + cm[1, 1]) / cm.sum()

            run_result = {
                "seed": seed,
                "accuracy": float(acc),
                "best_accuracy": float(best_acc),
                "best_round": best_round,
                "confusion_matrix": cm.tolist(),
                "recall_normal": float(recall_normal),
                "recall_pneumonia": float(recall_pneumonia),
                "per_client_cms": client_cms,
                "runtime_seconds": round(elapsed, 1),
            }

            results[algo].append(run_result)
            checkpoint["completed"][key] = run_result
            done_count += 1

            # Save after EVERY experiment (imaging is slow)
            _save_checkpoint_atomic(checkpoint, ckpt_path)

            print("  -> Acc:{:.1%} | NORMAL:{}/{} | PNEUMONIA:{}/{} | {:.0f}s".format(
                acc,
                cm[0, 0], cm[0].sum(),
                cm[1, 1], cm[1].sum(),
                elapsed), flush=True)

    # Build aggregated CMs for summary/figure
    all_cms = {}
    per_client_cms_agg = {}
    for algo in ALGORITHMS:
        algo_cm = np.zeros((2, 2), dtype=int)
        algo_per_client = {cid: np.zeros((2, 2), dtype=int) for cid in range(IMAGING_CONFIG["num_clients"])}
        for run in results.get(algo, []):
            algo_cm += np.array(run["confusion_matrix"])
            for cid_s, ccm in run.get("per_client_cms", {}).items():
                cid = int(cid_s)
                algo_per_client[cid] += np.array(ccm)
        all_cms[algo] = algo_cm
        per_client_cms_agg[algo] = {str(cid): cm.tolist() for cid, cm in algo_per_client.items()}

        # Print aggregated
        print("\n  {} aggregated CM (over {} seeds):".format(algo, len(seeds)))
        print("                 Predicted")
        print("                 NORMAL  PNEUMONIA")
        print("  True NORMAL    {:>5d}     {:>5d}".format(algo_cm[0, 0], algo_cm[0, 1]))
        print("  True PNEUMONIA {:>5d}     {:>5d}".format(algo_cm[1, 0], algo_cm[1, 1]))
        agg_acc = (algo_cm[0, 0] + algo_cm[1, 1]) / algo_cm.sum() if algo_cm.sum() > 0 else 0
        print("  Aggregated accuracy: {:.1f}%".format(agg_acc * 100))

    # Final save with aggregated data
    checkpoint["aggregated_cms"] = {algo: cm.tolist() for algo, cm in all_cms.items()}
    checkpoint["per_client_cms"] = per_client_cms_agg
    _save_checkpoint_atomic(checkpoint, ckpt_path)

    # ================================================================
    # Generate figure: 2x1 aggregate + per-client breakdown
    # ================================================================
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec

        num_clients = IMAGING_CONFIG["num_clients"]

        # Figure layout: top row = aggregate CMs, bottom row = per-client for HPFL
        fig = plt.figure(figsize=(10, 9))
        gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 1], hspace=0.35, wspace=0.3)

        # --- Top row: aggregate confusion matrices ---
        for col, algo in enumerate(ALGORITHMS):
            ax = fig.add_subplot(gs[0, col])
            cm = all_cms[algo]
            total_per_row = cm.sum(axis=1, keepdims=True)
            cm_pct = cm / total_per_row * 100

            im = ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=100)

            for i in range(2):
                for j in range(2):
                    color = "white" if cm_pct[i, j] > 50 else "black"
                    ax.text(j, i, "{}\n({:.0f}%)".format(cm[i, j], cm_pct[i, j]),
                            ha="center", va="center", fontsize=13, color=color, fontweight="bold")

            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(["NORMAL", "PNEUMONIA"], fontsize=10)
            ax.set_yticklabels(["NORMAL", "PNEUMONIA"], fontsize=10)
            ax.set_xlabel("Predicted", fontsize=11)
            if col == 0:
                ax.set_ylabel("True", fontsize=11)

            agg_acc = (cm[0, 0] + cm[1, 1]) / cm.sum() * 100
            recall_n = cm[0, 0] / cm[0].sum() * 100 if cm[0].sum() > 0 else 0
            recall_p = cm[1, 1] / cm[1].sum() * 100 if cm[1].sum() > 0 else 0
            ax.set_title("{}\nAcc: {:.1f}%  |  Recall N: {:.0f}%  P: {:.0f}%".format(
                algo, agg_acc, recall_n, recall_p),
                fontsize=11, fontweight="bold")

        # --- Bottom row: per-client accuracy breakdown for both algos ---
        # Bar chart showing per-client accuracy for FedAvg vs HPFL
        ax_bar = fig.add_subplot(gs[1, :])

        x = np.arange(num_clients)
        width = 0.35

        # Compute per-client accuracy from per_client_cms
        fedavg_accs = []
        hpfl_accs = []
        for cid in range(num_clients):
            # FedAvg
            fa_cm = np.array(per_client_cms_agg["FedAvg"][str(cid)]) if "FedAvg" in per_client_cms_agg else np.zeros((2,2))
            fa_acc = (fa_cm[0, 0] + fa_cm[1, 1]) / fa_cm.sum() * 100 if fa_cm.sum() > 0 else 0
            fedavg_accs.append(fa_acc)
            # HPFL
            hp_cm = np.array(per_client_cms_agg["HPFL"][str(cid)]) if "HPFL" in per_client_cms_agg else np.zeros((2,2))
            hp_acc = (hp_cm[0, 0] + hp_cm[1, 1]) / hp_cm.sum() * 100 if hp_cm.sum() > 0 else 0
            hpfl_accs.append(hp_acc)

        bars1 = ax_bar.bar(x - width/2, fedavg_accs, width, label='FedAvg', color='#4C72B0', alpha=0.85)
        bars2 = ax_bar.bar(x + width/2, hpfl_accs, width, label='HPFL', color='#DD8452', alpha=0.85)

        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                h = bar.get_height()
                ax_bar.text(bar.get_x() + bar.get_width()/2., h + 1,
                           '{:.0f}%'.format(h), ha='center', va='bottom', fontsize=9, fontweight='bold')

        ax_bar.set_xlabel("Client ID", fontsize=11)
        ax_bar.set_ylabel("Accuracy (%)", fontsize=11)
        ax_bar.set_title("Per-Client Accuracy (aggregated over {} seeds)".format(len(seeds)),
                        fontsize=11, fontweight="bold")
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(["Client {}".format(i) for i in range(num_clients)])
        ax_bar.set_ylim(0, 110)
        ax_bar.axhline(y=50, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
        ax_bar.legend(fontsize=10)
        ax_bar.grid(axis='y', alpha=0.3)

        plt.suptitle("Chest X-Ray: FedAvg vs HPFL Confusion Matrices\n(aggregated over {} seeds)".format(len(seeds)),
                     fontsize=13, fontweight="bold", y=0.98)

        fig_path = OUTPUT_DIR / "confusion_matrix_chest"
        plt.savefig(str(fig_path) + ".png", dpi=200, bbox_inches="tight")
        plt.savefig(str(fig_path) + ".pdf", bbox_inches="tight")
        print("\n  Figure saved: {}".format(fig_path))
        plt.close()

    except ImportError:
        print("\n  matplotlib not available — skipping figure generation")

    print("\n" + "=" * 60)
    print("  DONE.")
    print("  Checkpoint: {}".format(OUTPUT_DIR / "checkpoint_confusion_chest.json"))
    print("  Figure:     {}".format(OUTPUT_DIR / "confusion_matrix_chest.pdf"))
    print("=" * 60)


if __name__ == "__main__":
    main()

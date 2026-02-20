#!/usr/bin/env python3
"""
FL-EHDS Test B — Confusion Matrix on Breast Cancer.

Retrains FedAvg, FedProx, Ditto, and HPFL on Breast Cancer and generates
confusion matrices to demonstrate majority-class prediction collapse.

FedAvg/FedProx (52.3% acc, 32% F1) likely predict only majority class.
Confusion matrix proves this visually for the paper.

Total: 4 algos × 3 seeds = 12 experiments (~10-15 min total)

Usage:
    cd fl-ehds-framework
    python -m benchmarks.run_confusion_matrix_bc [--quick]

Output:
    benchmarks/paper_results_tabular/confusion_matrix_bc.png
    benchmarks/paper_results_tabular/checkpoint_confusion_bc.json

Author: Fabio Liberti
"""

import sys
import os
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

FRAMEWORK_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(FRAMEWORK_DIR))

import torch
import torch.nn as nn

from terminal.fl_trainer import (
    FederatedTrainer,
    HealthcareMLP,
    _detect_device,
)
from data.breast_cancer_loader import load_breast_cancer_data

OUTPUT_DIR = FRAMEWORK_DIR / "benchmarks" / "paper_results_tabular"

# Breast Cancer config (same as paper)
BC_CONFIG = dict(
    input_dim=30, num_classes=2,
    learning_rate=0.001, batch_size=16, num_rounds=40,
    local_epochs=1, mu=0.1,
)
NUM_CLIENTS = 3
ALGORITHMS = ["FedAvg", "FedProx", "Ditto", "HPFL"]
SEEDS = [42, 123, 456]
CLASS_NAMES = {0: "Benign", 1: "Malignant"}


def collect_predictions(trainer, seed):
    """Collect all predictions and true labels from test data."""
    model = trainer.global_model
    model.eval()
    all_preds = []
    all_labels = []

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
            X_t = torch.FloatTensor(X).to(trainer.device)
            out = model(X_t)
            preds = out.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(y.tolist() if hasattr(y, 'tolist') else list(y))

    if is_hpfl:
        for n, p in model.named_parameters():
            if n in trainer._hpfl_classifier_names:
                p.data.copy_(saved_cls[n])

    return np.array(all_labels), np.array(all_preds)


def compute_confusion_matrix(y_true, y_pred, num_classes=2):
    """Compute confusion matrix manually (no sklearn dependency)."""
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Confusion Matrix for Breast Cancer")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    seeds = [42] if args.quick else SEEDS
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  FL-EHDS Confusion Matrix — Breast Cancer")
    print("  {} algos x {} seeds = {} experiments".format(len(ALGORITHMS), len(seeds), len(ALGORITHMS) * len(seeds)))
    print("  Device: {}".format(_detect_device()))
    print("=" * 60)

    results = {}
    all_cms = {}  # (algo) -> aggregated CM

    for algo in ALGORITHMS:
        algo_cm = np.zeros((2, 2), dtype=int)
        algo_results = []

        for seed in seeds:
            print("\n  {} seed={} ...".format(algo, seed), end="", flush=True)
            start = time.time()

            client_data, client_test_data, metadata = load_breast_cancer_data(
                num_clients=NUM_CLIENTS, seed=seed, is_iid=False, alpha=0.5,
            )

            num_rounds = 5 if args.quick else BC_CONFIG["num_rounds"]

            trainer = FederatedTrainer(
                num_clients=NUM_CLIENTS,
                algorithm=algo,
                local_epochs=BC_CONFIG["local_epochs"],
                batch_size=BC_CONFIG["batch_size"],
                learning_rate=BC_CONFIG["learning_rate"],
                mu=BC_CONFIG["mu"],
                seed=seed,
                external_data=client_data,
                external_test_data=client_test_data,
                input_dim=BC_CONFIG["input_dim"],
                num_classes=BC_CONFIG["num_classes"],
            )

            for r in range(num_rounds):
                trainer.train_round(r)

            y_true, y_pred = collect_predictions(trainer, seed)
            cm = compute_confusion_matrix(y_true, y_pred, 2)
            algo_cm += cm

            acc = np.mean(y_true == y_pred)
            elapsed = time.time() - start

            # Per-class metrics
            tp0 = cm[0, 0]
            fn0 = cm[0, 1]
            fp0 = cm[1, 0]
            tn0 = cm[1, 1]
            recall_benign = tp0 / (tp0 + fn0) if (tp0 + fn0) > 0 else 0
            recall_malig = tn0 / (tn0 + fp0) if (tn0 + fp0) > 0 else 0

            algo_results.append({
                "seed": seed, "accuracy": float(acc),
                "confusion_matrix": cm.tolist(),
                "recall_benign": recall_benign,
                "recall_malignant": recall_malig,
            })

            print(" {:.1f}% [{}/{}] [{}/{}] {:.0f}s".format(
                acc * 100,
                cm[0, 0], cm[0, 0] + cm[0, 1],  # Benign: correct/total
                cm[1, 1], cm[1, 0] + cm[1, 1],  # Malignant: correct/total
                elapsed))

            # Cleanup
            del trainer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            import gc
            gc.collect()

        results[algo] = algo_results
        all_cms[algo] = algo_cm

        # Print aggregated CM
        print("\n  {} aggregated CM (over {} seeds):".format(algo, len(seeds)))
        print("                 Predicted")
        print("                 Benign  Malignant")
        print("  True Benign    {:>5d}     {:>5d}".format(algo_cm[0, 0], algo_cm[0, 1]))
        print("  True Malignant {:>5d}     {:>5d}".format(algo_cm[1, 0], algo_cm[1, 1]))
        total = algo_cm.sum()
        acc_agg = (algo_cm[0, 0] + algo_cm[1, 1]) / total if total > 0 else 0
        print("  Aggregated accuracy: {:.1f}%".format(acc_agg * 100))

    # Save checkpoint
    checkpoint = {
        "results": {algo: [
            {k: v for k, v in r.items()} for r in runs
        ] for algo, runs in results.items()},
        "aggregated_cms": {algo: cm.tolist() for algo, cm in all_cms.items()},
        "metadata": {
            "algorithms": ALGORITHMS,
            "seeds": seeds,
            "timestamp": datetime.now().isoformat(),
        }
    }
    with open(OUTPUT_DIR / "checkpoint_confusion_bc.json", "w") as f:
        json.dump(checkpoint, f, indent=2, default=str)

    # Generate figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, len(ALGORITHMS), figsize=(4 * len(ALGORITHMS), 4))
        if len(ALGORITHMS) == 1:
            axes = [axes]

        for ax, algo in zip(axes, ALGORITHMS):
            cm = all_cms[algo]
            total_per_row = cm.sum(axis=1, keepdims=True)
            cm_pct = cm / total_per_row * 100

            im = ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=100)

            for i in range(2):
                for j in range(2):
                    color = "white" if cm_pct[i, j] > 50 else "black"
                    ax.text(j, i, "{}\n({:.0f}%)".format(cm[i, j], cm_pct[i, j]),
                            ha="center", va="center", fontsize=12, color=color, fontweight="bold")

            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(["Benign", "Malignant"], fontsize=10)
            ax.set_yticklabels(["Benign", "Malignant"], fontsize=10)
            ax.set_xlabel("Predicted", fontsize=11)
            if ax == axes[0]:
                ax.set_ylabel("True", fontsize=11)

            acc_agg = (cm[0, 0] + cm[1, 1]) / cm.sum() * 100
            ax.set_title("{}\nAcc: {:.1f}%".format(algo, acc_agg), fontsize=12, fontweight="bold")

        plt.suptitle("Breast Cancer Confusion Matrices (aggregated over {} seeds)".format(len(seeds)),
                     fontsize=13, fontweight="bold", y=1.02)
        plt.tight_layout()

        fig_path = OUTPUT_DIR / "confusion_matrix_bc.png"
        plt.savefig(fig_path, dpi=200, bbox_inches="tight")
        plt.savefig(str(fig_path).replace(".png", ".pdf"), bbox_inches="tight")
        print("\n  Figure saved: {}".format(fig_path))
        plt.close()

    except ImportError:
        print("\n  matplotlib not available — skipping figure generation")

    print("\n" + "=" * 60)
    print("  DONE. Checkpoint: {}".format(OUTPUT_DIR / "checkpoint_confusion_bc.json"))
    print("=" * 60)


if __name__ == "__main__":
    main()

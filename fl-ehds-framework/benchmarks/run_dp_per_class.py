#!/usr/bin/env python3
"""
FL-EHDS Experiment D — Per-Class Recall Under Differential Privacy.

Measures how DP noise affects per-class recall and the Diagnostic Equity Index
(DEI) across different epsilon values. Key question: does DP worsen or improve
class-level diagnostic disparities?

Datasets:
  - Breast Cancer (binary, known single-class collapse)
  - PTB-XL (5-class, largest tabular dataset)

Algorithms: FedAvg, HPFL (most divergent DEI in no-DP setting)
Epsilon values: No-DP, 1, 5, 10
Seeds: 5 (matching existing DP study)

Total: 2 datasets × 2 algos × 4 DP levels × 5 seeds = 80 experiments
Estimated runtime: ~40-60 min (Breast Cancer ~30s/run, PTB-XL ~2min/run)

Usage:
    cd fl-ehds-framework
    python /tmp/run_dp_per_class.py [--quick] [--no-resume]

Output:
    /tmp/fl_ehds_results/dp_per_class_results.json
    /tmp/fl_ehds_results/dp_per_class_dei.png

Author: Fabio Liberti (generated for FLICS 2026)
"""

import sys
import os
import json
import time
import gc
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from copy import deepcopy

import numpy as np

# Setup paths
FRAMEWORK_DIR = Path("/Users/fxlybs/_DEV/FL-EHDS-FLICS2026/fl-ehds-framework")
sys.path.insert(0, str(FRAMEWORK_DIR))

import torch
from terminal.fl_trainer import FederatedTrainer, _detect_device
from data.breast_cancer_loader import load_breast_cancer_data

OUTPUT_DIR = Path("/tmp/fl_ehds_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_PATH = OUTPUT_DIR / "dp_per_class_results.json"

# ─── Configuration ────────────────────────────────────────────────────
ALGORITHMS = ["FedAvg", "HPFL"]
DP_LEVELS = [
    {"label": "No-DP", "dp_enabled": False, "dp_epsilon": 0, "dp_clip_norm": 1.0},
    {"label": "eps=1",  "dp_enabled": True,  "dp_epsilon": 1.0, "dp_clip_norm": 1.0},
    {"label": "eps=5",  "dp_enabled": True,  "dp_epsilon": 5.0, "dp_clip_norm": 1.0},
    {"label": "eps=10", "dp_enabled": True,  "dp_epsilon": 10.0, "dp_clip_norm": 1.0},
]
SEEDS = [42, 123, 456, 789, 999]

DATASETS = {
    "Breast_Cancer": {
        "loader": "breast_cancer",
        "num_clients": 3,
        "config": dict(
            input_dim=30, num_classes=2,
            learning_rate=0.001, batch_size=16, num_rounds=40,
            local_epochs=1, mu=0.1,
        ),
        "class_names": {0: "Benign", 1: "Malignant"},
    },
    "PTB_XL": {
        "loader": "ptbxl",
        "num_clients": 5,
        "config": dict(
            input_dim=9, num_classes=5,
            learning_rate=0.01, batch_size=32, num_rounds=20,
            local_epochs=3, mu=0.1,
        ),
        "class_names": {0: "NORM", 1: "MI", 2: "STTC", 3: "CD", 4: "HYP"},
    },
}


def compute_confusion_matrix(y_true, y_pred, num_classes):
    """Compute confusion matrix manually."""
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def compute_dei(cm, num_classes):
    """Compute Diagnostic Equity Index from confusion matrix."""
    recalls = []
    for c in range(num_classes):
        total = cm[c].sum()
        recall = cm[c, c] / total if total > 0 else 0.0
        recalls.append(recall)

    if min(recalls) == 0:
        return 0.0, recalls, 0.0, 1.0

    mean_r = np.mean(recalls)
    std_r = np.std(recalls)
    cv = std_r / mean_r if mean_r > 0 else 1.0
    dei = min(recalls) * (1 - cv)
    return dei, recalls, min(recalls), cv


def collect_predictions(trainer, num_classes):
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


def load_checkpoint():
    """Load existing checkpoint for resume support."""
    if CHECKPOINT_PATH.exists():
        try:
            with open(CHECKPOINT_PATH) as f:
                return json.load(f)
        except Exception:
            pass
    return {"results": [], "completed": [], "metadata": {}}


def save_checkpoint(data):
    """Atomic checkpoint save."""
    tmp = CHECKPOINT_PATH.with_suffix('.tmp')
    with open(tmp, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    tmp.replace(CHECKPOINT_PATH)


def load_ptbxl_data(num_clients, seed, is_iid=False, alpha=0.5):
    """Load PTB-XL dataset. Try multiple loaders."""
    try:
        from data.ptbxl_loader import load_ptbxl_data as _load
        return _load(num_clients=num_clients, seed=seed, is_iid=is_iid, alpha=alpha)
    except ImportError:
        pass
    try:
        from data.dataset_loaders import load_dataset
        return load_dataset("ptbxl", num_clients=num_clients, seed=seed,
                          is_iid=is_iid, alpha=alpha)
    except ImportError:
        pass
    raise ImportError("Cannot find PTB-XL data loader")


def run_experiment(dataset_name, algo, dp_level, seed):
    """Run a single experiment and return results with confusion matrix."""
    ds_cfg = DATASETS[dataset_name]
    cfg = ds_cfg["config"]
    num_classes = cfg["num_classes"]

    # Load data
    if ds_cfg["loader"] == "breast_cancer":
        client_data, client_test, meta = load_breast_cancer_data(
            num_clients=ds_cfg["num_clients"], seed=seed,
            is_iid=False, alpha=0.5,
        )
    elif ds_cfg["loader"] == "ptbxl":
        client_data, client_test, meta = load_ptbxl_data(
            num_clients=ds_cfg["num_clients"], seed=seed,
        )
    else:
        raise ValueError(f"Unknown loader: {ds_cfg['loader']}")

    # Create trainer
    trainer = FederatedTrainer(
        num_clients=ds_cfg["num_clients"],
        algorithm=algo,
        local_epochs=cfg["local_epochs"],
        batch_size=cfg["batch_size"],
        learning_rate=cfg["learning_rate"],
        mu=cfg["mu"],
        dp_enabled=dp_level["dp_enabled"],
        dp_epsilon=dp_level["dp_epsilon"] if dp_level["dp_enabled"] else 10.0,
        dp_clip_norm=dp_level["dp_clip_norm"],
        seed=seed,
        external_data=client_data,
        external_test_data=client_test,
        input_dim=cfg["input_dim"],
        num_classes=num_classes,
    )

    # Train
    start = time.time()
    for r in range(cfg["num_rounds"]):
        trainer.train_round(r)
    elapsed = time.time() - start

    # Collect predictions and compute CM
    y_true, y_pred = collect_predictions(trainer, num_classes)
    cm = compute_confusion_matrix(y_true, y_pred, num_classes)
    acc = np.mean(y_true == y_pred)
    dei, recalls, min_r, cv = compute_dei(cm, num_classes)

    per_class_recall = {}
    for c in range(num_classes):
        cls_name = ds_cfg["class_names"].get(c, f"Class_{c}")
        per_class_recall[cls_name] = float(recalls[c])

    result = {
        "dataset": dataset_name,
        "algorithm": algo,
        "dp_label": dp_level["label"],
        "dp_epsilon": dp_level["dp_epsilon"],
        "dp_enabled": dp_level["dp_enabled"],
        "seed": seed,
        "accuracy": float(acc),
        "confusion_matrix": cm.tolist(),
        "per_class_recall": per_class_recall,
        "dei": float(dei),
        "min_recall": float(min_r),
        "cv": float(cv),
        "runtime_seconds": round(elapsed, 1),
    }

    # Cleanup
    del trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return result


def generate_figure(all_results):
    """Generate DEI comparison figure across DP levels."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available — skipping figure")
        return

    datasets = list(DATASETS.keys())
    n_ds = len(datasets)
    fig, axes = plt.subplots(1, n_ds, figsize=(7 * n_ds, 5))
    if n_ds == 1:
        axes = [axes]

    dp_labels = [d["label"] for d in DP_LEVELS]
    colors = {"FedAvg": "#d62728", "HPFL": "#2ca02c"}
    markers = {"FedAvg": "o", "HPFL": "s"}

    for ax, ds_name in zip(axes, datasets):
        for algo in ALGORITHMS:
            dei_means = []
            dei_stds = []
            acc_means = []

            for dp_level in DP_LEVELS:
                # Filter results
                matching = [r for r in all_results
                           if r["dataset"] == ds_name
                           and r["algorithm"] == algo
                           and r["dp_label"] == dp_level["label"]]
                if matching:
                    deis = [r["dei"] for r in matching]
                    accs = [r["accuracy"] for r in matching]
                    dei_means.append(np.mean(deis))
                    dei_stds.append(np.std(deis))
                    acc_means.append(np.mean(accs) * 100)
                else:
                    dei_means.append(0)
                    dei_stds.append(0)
                    acc_means.append(0)

            x = np.arange(len(dp_labels))
            ax.errorbar(x, dei_means, yerr=dei_stds,
                       label=f"{algo} (DEI)",
                       color=colors.get(algo, "gray"),
                       marker=markers.get(algo, "o"),
                       linewidth=2, markersize=8, capsize=4)

            # Add accuracy as secondary annotation
            for i, (dei_m, acc_m) in enumerate(zip(dei_means, acc_means)):
                ax.annotate(f"{acc_m:.0f}%", (x[i], dei_m),
                           textcoords="offset points", xytext=(0, 12),
                           ha='center', fontsize=8,
                           color=colors.get(algo, "gray"))

        ax.set_xticks(range(len(dp_labels)))
        ax.set_xticklabels(dp_labels, fontsize=11)
        ax.set_ylabel("DEI", fontsize=12)
        ax.set_xlabel("Privacy Level (ε)", fontsize=12)
        ax.set_title(ds_name.replace("_", " "), fontsize=13, fontweight="bold")
        ax.legend(fontsize=10)
        ax.set_ylim(-0.05, 1.0)
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3, label='DEI=0.5 threshold')
        ax.grid(True, alpha=0.3)

    plt.suptitle("Diagnostic Equity Index (DEI) Under Differential Privacy\n"
                 "Accuracy annotations shown above each point",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    fig_path = OUTPUT_DIR / "dp_per_class_dei.png"
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.savefig(str(fig_path).replace(".png", ".pdf"), bbox_inches="tight")
    print(f"\n  Figure saved: {fig_path}")
    plt.close()

    # Also generate per-class recall heatmap
    for ds_name in datasets:
        ds_cfg = DATASETS[ds_name]
        class_names = list(ds_cfg["class_names"].values())
        n_classes = len(class_names)

        fig2, axes2 = plt.subplots(1, len(ALGORITHMS), figsize=(6 * len(ALGORITHMS), 4))
        if len(ALGORITHMS) == 1:
            axes2 = [axes2]

        for ax2, algo in zip(axes2, ALGORITHMS):
            recall_matrix = np.zeros((n_classes, len(dp_labels)))

            for j, dp_level in enumerate(DP_LEVELS):
                matching = [r for r in all_results
                           if r["dataset"] == ds_name
                           and r["algorithm"] == algo
                           and r["dp_label"] == dp_level["label"]]
                if matching:
                    for c in range(n_classes):
                        cls_name = class_names[c]
                        recalls = [r["per_class_recall"].get(cls_name, 0) for r in matching]
                        recall_matrix[c, j] = np.mean(recalls) * 100

            im = ax2.imshow(recall_matrix, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
            for i in range(n_classes):
                for j in range(len(dp_labels)):
                    color = "white" if recall_matrix[i, j] < 30 or recall_matrix[i, j] > 80 else "black"
                    ax2.text(j, i, f"{recall_matrix[i, j]:.0f}%",
                            ha="center", va="center", fontsize=10, color=color, fontweight="bold")

            ax2.set_xticks(range(len(dp_labels)))
            ax2.set_xticklabels(dp_labels, fontsize=10)
            ax2.set_yticks(range(n_classes))
            ax2.set_yticklabels(class_names, fontsize=10)
            ax2.set_xlabel("Privacy Level", fontsize=11)
            ax2.set_title(f"{algo}", fontsize=12, fontweight="bold")

        plt.suptitle(f"{ds_name.replace('_', ' ')} — Per-Class Recall (%) Under DP",
                     fontsize=13, fontweight="bold", y=1.02)
        plt.colorbar(im, ax=axes2, label="Recall (%)", shrink=0.8)
        plt.tight_layout()

        heatmap_path = OUTPUT_DIR / f"dp_per_class_recall_{ds_name.lower()}.png"
        plt.savefig(heatmap_path, dpi=200, bbox_inches="tight")
        plt.savefig(str(heatmap_path).replace(".png", ".pdf"), bbox_inches="tight")
        print(f"  Heatmap saved: {heatmap_path}")
        plt.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Per-Class Recall Under DP")
    parser.add_argument("--quick", action="store_true", help="Quick mode (1 seed, 1 dataset)")
    parser.add_argument("--no-resume", action="store_true", help="Start fresh")
    parser.add_argument("--dataset", type=str, default=None,
                       choices=list(DATASETS.keys()), help="Run single dataset")
    args = parser.parse_args()

    seeds = [42] if args.quick else SEEDS
    datasets = [args.dataset] if args.dataset else list(DATASETS.keys())

    # Resume support
    checkpoint = {"results": [], "completed": [], "metadata": {}}
    if not args.no_resume:
        checkpoint = load_checkpoint()
    completed_keys = set(checkpoint.get("completed", []))

    total = len(datasets) * len(ALGORITHMS) * len(DP_LEVELS) * len(seeds)
    skip = sum(1 for ds in datasets for a in ALGORITHMS
               for dp in DP_LEVELS for s in seeds
               if f"{ds}_{a}_{dp['label']}_s{s}" in completed_keys)

    print("=" * 65)
    print("  FL-EHDS Experiment D — Per-Class Recall Under DP")
    print(f"  {len(datasets)} datasets × {len(ALGORITHMS)} algos × "
          f"{len(DP_LEVELS)} DP × {len(seeds)} seeds = {total} experiments")
    if skip:
        print(f"  Skipping {skip} completed, running {total - skip}")
    print(f"  Device: {_detect_device()}")
    print("=" * 65)

    exp_count = 0
    for ds_name in datasets:
        ds_cfg = DATASETS[ds_name]

        # Pre-check dataset availability
        if ds_cfg["loader"] == "ptbxl":
            try:
                load_ptbxl_data(num_clients=ds_cfg["num_clients"], seed=42)
                print(f"\n  {ds_name}: data loaded successfully")
            except Exception as e:
                print(f"\n  {ds_name}: SKIPPED — {e}")
                continue

        for algo in ALGORITHMS:
            for dp_level in DP_LEVELS:
                for seed in seeds:
                    key = f"{ds_name}_{algo}_{dp_level['label']}_s{seed}"
                    if key in completed_keys:
                        exp_count += 1
                        continue

                    exp_count += 1
                    print(f"\n  [{exp_count}/{total}] {ds_name} | {algo} | "
                          f"{dp_level['label']} | s{seed} ...", end="", flush=True)

                    try:
                        result = run_experiment(ds_name, algo, dp_level, seed)
                        checkpoint["results"].append(result)
                        checkpoint["completed"].append(key)

                        # Print summary
                        print(f" acc={result['accuracy']*100:.1f}% "
                              f"DEI={result['dei']:.3f} "
                              f"min_R={result['min_recall']:.3f} "
                              f"{result['runtime_seconds']:.0f}s")

                        # Per-class detail
                        for cls, rec in result["per_class_recall"].items():
                            print(f"      {cls}: {rec*100:.1f}%")

                    except Exception as e:
                        print(f" ERROR: {e}")
                        checkpoint["results"].append({
                            "dataset": ds_name, "algorithm": algo,
                            "dp_label": dp_level["label"], "seed": seed,
                            "error": str(e),
                        })
                        checkpoint["completed"].append(key)

                    # Save checkpoint after each experiment
                    checkpoint["metadata"] = {
                        "datasets": datasets,
                        "algorithms": ALGORITHMS,
                        "dp_levels": [d["label"] for d in DP_LEVELS],
                        "seeds": seeds,
                        "timestamp": datetime.now().isoformat(),
                    }
                    save_checkpoint(checkpoint)

    # Summary
    print("\n" + "=" * 65)
    print("  SUMMARY — DEI Under DP")
    print("=" * 65)

    valid_results = [r for r in checkpoint["results"] if "error" not in r]

    for ds_name in datasets:
        print(f"\n  {ds_name}:")
        print(f"  {'Algorithm':<10} {'DP Level':<10} {'Acc%':<8} {'DEI':<8} {'min_R':<8} {'CV':<8}")
        print(f"  {'-'*52}")

        for algo in ALGORITHMS:
            for dp_level in DP_LEVELS:
                matching = [r for r in valid_results
                           if r["dataset"] == ds_name
                           and r["algorithm"] == algo
                           and r["dp_label"] == dp_level["label"]]
                if matching:
                    acc = np.mean([r["accuracy"] for r in matching]) * 100
                    dei = np.mean([r["dei"] for r in matching])
                    min_r = np.mean([r["min_recall"] for r in matching])
                    cv = np.mean([r["cv"] for r in matching])
                    print(f"  {algo:<10} {dp_level['label']:<10} {acc:<8.1f} "
                          f"{dei:<8.3f} {min_r:<8.3f} {cv:<8.3f}")

    # Generate figures
    if valid_results:
        print("\n  Generating figures...")
        generate_figure(valid_results)

    print(f"\n  Checkpoint: {CHECKPOINT_PATH}")
    print("=" * 65)


if __name__ == "__main__":
    main()

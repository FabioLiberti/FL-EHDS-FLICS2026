#!/usr/bin/env python3
"""
Regenerate 6 supplementary figures with 3+2 layout (3 top, 2 bottom centered).

Figures regenerated (supplementary S-19, S-22, S-24, S-25, S-29, S-31):
  - 00_convergence_accuracy.pdf  (S-19)
  - 03_accuracy_boxplots.pdf     (S-22)
  - 05_loss_convergence.pdf      (S-24)
  - 06_f1_convergence.pdf        (S-25)
  - 10_data_distribution.pdf     (S-29)
  - 12_early_stopping.pdf        (S-31)

Layout: 3 subplots on row 1 (full width), 2 centered subplots on row 2.

Usage:
    cd fl-ehds-framework
    python -m benchmarks.regenerate_6figures_3x2
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ======================================================================
# Setup (mirrors generate_all_figures.py)
# ======================================================================

FRAMEWORK_DIR = Path(__file__).parent.parent
CHECKPOINT_FILE = FRAMEWORK_DIR / "benchmarks" / "paper_results" / "checkpoint_p12_multidataset.json"
PAPER_FIGURES_DIR = FRAMEWORK_DIR.parent / "paper" / "paper2rel" / "figures"

SEEDS = [42, 123, 456]

IMAGING_DATASETS = {
    "Brain_Tumor": {"short": "BT", "full": "Brain Tumor\n(4-class, ResNet-18)"},
    "chest_xray": {"short": "CX", "full": "Chest X-ray\n(binary, ResNet-18)"},
    "Skin_Cancer": {"short": "SC", "full": "Skin Cancer\n(binary, ResNet-18)"},
}
TABULAR_DATASETS = {
    "Diabetes": {"short": "DM", "full": "CDC Diabetes\n(binary, 253K)"},
    "Heart_Disease": {"short": "HD", "full": "Heart Disease\n(binary, 4 hospitals)"},
}
ALL_DATASETS = list(IMAGING_DATASETS.keys()) + list(TABULAR_DATASETS.keys())
DS_INFO = {**IMAGING_DATASETS, **TABULAR_DATASETS}

CORE_ALGOS = ["FedAvg", "FedLC", "FedSAM", "FedDecorr", "FedExP"]
CORE_COLORS = ["#2196F3", "#E91E63", "#4CAF50", "#FF9800", "#9C27B0"]

plt.rcParams.update({
    "font.size": 12, "font.family": "serif",
    "axes.labelsize": 13, "axes.titlesize": 14,
    "xtick.labelsize": 11, "ytick.labelsize": 11,
    "legend.fontsize": 10, "figure.dpi": 150,
    "figure.facecolor": "white",
})


# ======================================================================
# Helpers
# ======================================================================

def load_data():
    with open(CHECKPOINT_FILE, "r") as f:
        return json.load(f)["completed"]


def agg_metric(completed, dataset, algo, metric):
    vals = []
    for seed in SEEDS:
        rec = completed.get(f"{dataset}_{algo}_{seed}", {})
        fm = rec.get("final_metrics", {})
        if metric in fm:
            vals.append(fm[metric])
    if not vals:
        return 0.0, 0.0
    return float(np.mean(vals)), float(np.std(vals))


def get_histories(completed, dataset, algo, metric):
    histories = []
    for seed in SEEDS:
        rec = completed.get(f"{dataset}_{algo}_{seed}", {})
        h = rec.get("history", [])
        if h:
            histories.append([r.get(metric, 0) for r in h])
    return histories


def make_3x2_figure(height_per_row=5.0, wspace=0.3, hspace=0.4):
    """Create a figure with 3+2 layout using GridSpec.
    Row 1: 3 subplots spanning columns [0:2], [2:4], [4:6]
    Row 2: 2 subplots centered spanning columns [1:3], [3:5]
    """
    fig = plt.figure(figsize=(16, height_per_row * 2))
    gs = gridspec.GridSpec(2, 6, figure=fig, wspace=wspace, hspace=hspace)

    axes = [
        fig.add_subplot(gs[0, 0:2]),  # top-left
        fig.add_subplot(gs[0, 2:4]),  # top-center
        fig.add_subplot(gs[0, 4:6]),  # top-right
        fig.add_subplot(gs[1, 1:3]),  # bottom-left (centered)
        fig.add_subplot(gs[1, 3:5]),  # bottom-right (centered)
    ]
    return fig, axes


# ======================================================================
# S-19: Accuracy Convergence (00_convergence_accuracy.pdf)
# ======================================================================

def gen_convergence_accuracy(completed):
    fig, axes = make_3x2_figure(height_per_row=4.5)

    for ax, ds in zip(axes, ALL_DATASETS):
        for i, algo in enumerate(CORE_ALGOS):
            hists = get_histories(completed, ds, algo, "accuracy")
            if not hists:
                continue
            min_len = min(len(h) for h in hists)
            arr = np.array([h[:min_len] for h in hists])
            mean = arr.mean(axis=0) * 100
            std = arr.std(axis=0) * 100
            rounds = np.arange(1, min_len + 1)
            ax.plot(rounds, mean, color=CORE_COLORS[i], label=algo, linewidth=2)
            ax.fill_between(rounds, mean - std, mean + std,
                           color=CORE_COLORS[i], alpha=0.15)

        short = DS_INFO[ds]["short"]
        ax.set_title(short, fontsize=14, fontweight="bold")
        ax.set_xlabel("Communication Round")
        ax.grid(alpha=0.3)
        ax.set_ylabel("Accuracy (%)")

    axes[-1].legend(loc="lower right", fontsize=10, framealpha=0.9)
    fig.suptitle("Accuracy Convergence per Dataset (mean ± std, 3 seeds)",
                 fontsize=15, y=0.98)
    fig.savefig(PAPER_FIGURES_DIR / "00_convergence_accuracy.pdf",
                bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("  00_convergence_accuracy.pdf (S-19)")


# ======================================================================
# S-22: Accuracy Boxplots (03_accuracy_boxplots.pdf)
# ======================================================================

def gen_boxplots(completed):
    fig, axes = make_3x2_figure(height_per_row=5.0)

    for ax, ds in zip(axes, ALL_DATASETS):
        data_box = []
        labels = []
        colors = []
        for i, algo in enumerate(CORE_ALGOS):
            vals = []
            for seed in SEEDS:
                rec = completed.get(f"{ds}_{algo}_{seed}", {})
                fm = rec.get("final_metrics", {})
                if "accuracy" in fm:
                    vals.append(fm["accuracy"] * 100)
            if vals:
                data_box.append(vals)
                labels.append(algo)
                colors.append(CORE_COLORS[i])

        bp = ax.boxplot(data_box, labels=labels, patch_artist=True, widths=0.6,
                       medianprops=dict(color="black", linewidth=2))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        for j, vals in enumerate(data_box):
            jitter = np.random.normal(0, 0.04, len(vals))
            ax.scatter(np.ones(len(vals)) * (j + 1) + jitter, vals,
                      color=colors[j], s=50, zorder=5,
                      edgecolor="black", linewidth=0.5)

        short = DS_INFO[ds]["short"]
        ax.set_title(short, fontsize=14, fontweight="bold")
        ax.set_ylabel("Accuracy (%)")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Accuracy Variability Across Seeds (3 seeds per experiment)",
                 fontsize=15, y=0.98)
    fig.savefig(PAPER_FIGURES_DIR / "03_accuracy_boxplots.pdf",
                bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("  03_accuracy_boxplots.pdf (S-22)")


# ======================================================================
# S-24: Loss Convergence (05_loss_convergence.pdf)
# ======================================================================

def gen_loss_convergence(completed):
    fig, axes = make_3x2_figure(height_per_row=4.5)

    for ax, ds in zip(axes, ALL_DATASETS):
        for i, algo in enumerate(CORE_ALGOS):
            hists = get_histories(completed, ds, algo, "loss")
            if not hists:
                continue
            min_len = min(len(h) for h in hists)
            arr = np.array([h[:min_len] for h in hists])
            mean = arr.mean(axis=0)
            std = arr.std(axis=0)
            rounds = np.arange(1, min_len + 1)
            ax.plot(rounds, mean, color=CORE_COLORS[i], label=algo, linewidth=2)
            ax.fill_between(rounds, mean - std, mean + std,
                           color=CORE_COLORS[i], alpha=0.15)

        short = DS_INFO[ds]["short"]
        ax.set_title(short, fontsize=14, fontweight="bold")
        ax.set_xlabel("Communication Round")
        ax.grid(alpha=0.3)
        ax.set_ylabel("Loss")

    axes[-1].legend(loc="upper right", fontsize=10, framealpha=0.9)
    fig.suptitle("Training Loss Convergence per Dataset (mean ± std, 3 seeds)",
                 fontsize=15, y=0.98)
    fig.savefig(PAPER_FIGURES_DIR / "05_loss_convergence.pdf",
                bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("  05_loss_convergence.pdf (S-24)")


# ======================================================================
# S-25: F1 Convergence (06_f1_convergence.pdf)
# ======================================================================

def gen_f1_convergence(completed):
    fig, axes = make_3x2_figure(height_per_row=4.5)

    for ax, ds in zip(axes, ALL_DATASETS):
        for i, algo in enumerate(CORE_ALGOS):
            hists = get_histories(completed, ds, algo, "f1")
            if not hists:
                continue
            min_len = min(len(h) for h in hists)
            arr = np.array([h[:min_len] for h in hists])
            mean = arr.mean(axis=0)
            std = arr.std(axis=0)
            rounds = np.arange(1, min_len + 1)
            ax.plot(rounds, mean, color=CORE_COLORS[i], label=algo, linewidth=2)
            ax.fill_between(rounds, mean - std, mean + std,
                           color=CORE_COLORS[i], alpha=0.15)

        short = DS_INFO[ds]["short"]
        ax.set_title(short, fontsize=14, fontweight="bold")
        ax.set_xlabel("Communication Round")
        ax.grid(alpha=0.3)
        ax.set_ylabel("F1 Score")

    axes[-1].legend(loc="lower right", fontsize=10, framealpha=0.9)
    fig.suptitle("F1-Score Convergence per Dataset (mean ± std, 3 seeds)",
                 fontsize=15, y=0.98)
    fig.savefig(PAPER_FIGURES_DIR / "06_f1_convergence.pdf",
                bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("  06_f1_convergence.pdf (S-25)")


# ======================================================================
# S-29: Per-Client Accuracy Distribution (10_data_distribution.pdf)
# ======================================================================

def gen_client_distribution(completed):
    fig, axes = make_3x2_figure(height_per_row=5.0)

    for ax, ds in zip(axes, ALL_DATASETS):
        for i, algo in enumerate(CORE_ALGOS):
            all_client_accs = []
            for seed in SEEDS:
                rec = completed.get(f"{ds}_{algo}_{seed}", {})
                pca = rec.get("per_client_acc", {})
                all_client_accs.extend(pca.values())

            if all_client_accs:
                x_pos = (np.ones(len(all_client_accs)) * i
                         + np.random.normal(0, 0.1, len(all_client_accs)))
                ax.scatter(x_pos, [v * 100 for v in all_client_accs],
                          color=CORE_COLORS[i], s=30, alpha=0.6,
                          edgecolor="black", linewidth=0.3)
                mean_val = np.mean(all_client_accs) * 100
                ax.plot([i - 0.3, i + 0.3], [mean_val, mean_val],
                       color=CORE_COLORS[i], linewidth=2.5)

        ax.set_xticks(range(len(CORE_ALGOS)))
        ax.set_xticklabels(CORE_ALGOS, rotation=45, fontsize=10)
        short = DS_INFO[ds]["short"]
        ax.set_title(short, fontsize=14, fontweight="bold")
        ax.set_ylabel("Per-Client Accuracy (%)")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Per-Client Accuracy Distribution (Fairness View)",
                 fontsize=15, y=0.98)
    fig.savefig(PAPER_FIGURES_DIR / "10_data_distribution.pdf",
                bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("  10_data_distribution.pdf (S-29)")


# ======================================================================
# S-31: Early Stopping (12_early_stopping.pdf)
# ======================================================================

def gen_early_stopping(completed):
    rounds_data = defaultdict(list)

    for key, val in completed.items():
        ds = val.get("dataset", "")
        algo = val.get("algorithm", "")
        if algo not in CORE_ALGOS:
            continue
        h = val.get("history", [])
        if h:
            rounds_data[ds].append(len(h))

    fig, axes = make_3x2_figure(height_per_row=4.5)

    for ax, ds in zip(axes, ALL_DATASETS):
        data_r = rounds_data.get(ds, [])
        if data_r:
            bins = range(1, max(data_r) + 2)
            ax.hist(data_r, bins=bins, color="#2196F3", alpha=0.7,
                   edgecolor="black", align="left")

        short = DS_INFO[ds]["short"]
        ax.set_title(short, fontsize=14, fontweight="bold")
        ax.set_xlabel("Actual Rounds")
        ax.set_ylabel("Count")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Early Stopping: Distribution of Actual Training Rounds",
                 fontsize=15, y=0.98)
    fig.savefig(PAPER_FIGURES_DIR / "12_early_stopping.pdf",
                bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("  12_early_stopping.pdf (S-31)")


# ======================================================================
# Main
# ======================================================================

def main():
    print("Loading checkpoint data...")
    completed = load_data()
    print(f"  {len(completed)} experiments loaded")

    print("\nRegenerating 6 figures with 3+2 layout...")
    gen_convergence_accuracy(completed)
    gen_boxplots(completed)
    gen_loss_convergence(completed)
    gen_f1_convergence(completed)
    gen_client_distribution(completed)
    gen_early_stopping(completed)

    print(f"\nAll 6 figures saved to: {PAPER_FIGURES_DIR}")
    print("Done!")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Comprehensive Figure Generator for FL-EHDS Paper (FLICS 2026).

Generates ALL possible visualizations from P1.2 experiment results.
Run from fl-ehds-framework/:
    python -m benchmarks.generate_all_figures

Output: benchmarks/paper_results/figures/
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np

# ======================================================================
# Setup
# ======================================================================

FRAMEWORK_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(FRAMEWORK_DIR))

OUTPUT_DIR = FRAMEWORK_DIR / "benchmarks" / "paper_results"
FIGURES_DIR = OUTPUT_DIR / "figures"

CHECKPOINT_FILE = OUTPUT_DIR / "checkpoint_p12_multidataset.json"

SEEDS = [42, 123, 456]

# Imaging datasets (5 core algorithms)
IMAGING_DATASETS = {
    "Brain_Tumor": {"short": "BT", "num_classes": 4, "task": "Multi-class (4)"},
    "chest_xray": {"short": "CX", "num_classes": 2, "task": "Binary"},
    "Skin_Cancer": {"short": "SC", "num_classes": 2, "task": "Binary"},
}
TABULAR_DATASETS = {
    "Diabetes": {"short": "DM", "num_classes": 2, "task": "Binary"},
    "Heart_Disease": {"short": "HD", "num_classes": 2, "task": "Binary"},
}
ALL_DATASETS = list(IMAGING_DATASETS.keys()) + list(TABULAR_DATASETS.keys())
DS_INFO = {**IMAGING_DATASETS, **TABULAR_DATASETS}

# 5 core algorithms (present in all datasets)
CORE_ALGOS = ["FedAvg", "FedLC", "FedSAM", "FedDecorr", "FedExP"]
CORE_COLORS = ["#2196F3", "#E91E63", "#4CAF50", "#FF9800", "#9C27B0"]

# Extended algorithms (only in tabular datasets)
EXTENDED_ALGOS = ["FedAvg", "FedLC", "FedSAM", "FedDecorr", "FedExP",
                  "FedProx", "SCAFFOLD", "FedNova", "Ditto"]
EXTENDED_COLORS = ["#2196F3", "#E91E63", "#4CAF50", "#FF9800", "#9C27B0",
                   "#607D8B", "#795548", "#F44336", "#00BCD4"]

METRICS = ["accuracy", "f1", "precision", "recall", "auc"]
METRIC_LABELS = {
    "accuracy": "Accuracy", "f1": "F1 Score",
    "precision": "Precision", "recall": "Recall", "auc": "AUC"
}


# ======================================================================
# Helpers
# ======================================================================

def load_data() -> Dict:
    with open(CHECKPOINT_FILE, "r") as f:
        return json.load(f)["completed"]


def agg_metric(completed, dataset, algo, metric) -> Tuple[float, float]:
    """Average metric across seeds. Returns (mean, std)."""
    vals = []
    for seed in SEEDS:
        rec = completed.get(f"{dataset}_{algo}_{seed}", {})
        fm = rec.get("final_metrics", {})
        if metric in fm:
            vals.append(fm[metric])
    if not vals:
        return 0.0, 0.0
    return float(np.mean(vals)), float(np.std(vals))


def get_histories(completed, dataset, algo, metric) -> List[List[float]]:
    """Get per-round metric histories for all seeds."""
    histories = []
    for seed in SEEDS:
        rec = completed.get(f"{dataset}_{algo}_{seed}", {})
        h = rec.get("history", [])
        if h:
            histories.append([r.get(metric, 0) for r in h])
    return histories


def setup_plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 11, "font.family": "serif",
        "axes.labelsize": 12, "axes.titlesize": 13,
        "xtick.labelsize": 10, "ytick.labelsize": 10,
        "legend.fontsize": 9, "figure.dpi": 150,
        "figure.facecolor": "white",
    })
    return plt


def save_fig(fig, name, plt):
    for fmt in ["pdf", "png"]:
        fig.savefig(FIGURES_DIR / f"{name}.{fmt}", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved {name}")


# ======================================================================
# FIGURE 1: Multi-Dataset Accuracy Bars (core 5 algos, all 5 datasets)
# ======================================================================

def fig_01_accuracy_bars(completed, plt):
    """Grouped bar chart: accuracy per algorithm x dataset."""
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(ALL_DATASETS))
    width = 0.15

    for i, algo in enumerate(CORE_ALGOS):
        means, stds = [], []
        for ds in ALL_DATASETS:
            m, s = agg_metric(completed, ds, algo, "accuracy")
            means.append(m * 100)
            stds.append(s * 100)
        ax.bar(x + i * width, means, width, yerr=stds, label=algo,
               color=CORE_COLORS[i], capsize=3, alpha=0.85)

    labels = [DS_INFO[d]["short"] for d in ALL_DATASETS]
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(labels)
    ax.legend(loc="upper right", ncol=5)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("Multi-Dataset Federated Learning Comparison")
    ax.set_ylim(0, 100)

    save_fig(fig, "fig01_accuracy_bars", plt)


# ======================================================================
# FIGURE 2: Multi-Metric Bars (acc, f1, prec, rec, auc per dataset)
# ======================================================================

def fig_02_multimetric_bars(completed, plt):
    """One figure per dataset showing all metrics for each algorithm."""
    for ds in ALL_DATASETS:
        short = DS_INFO[ds]["short"]
        algos = CORE_ALGOS

        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(METRICS))
        width = 0.15

        for i, algo in enumerate(algos):
            vals, errs = [], []
            for m in METRICS:
                mean, std = agg_metric(completed, ds, algo, m)
                vals.append(mean)
                errs.append(std)
            ax.bar(x + i * width, vals, width, yerr=errs, label=algo,
                   color=CORE_COLORS[i], capsize=3, alpha=0.85)

        ax.set_ylabel("Score")
        ax.set_xticks(x + width * 2)
        ax.set_xticklabels([METRIC_LABELS[m] for m in METRICS])
        ax.legend(loc="upper right")
        ax.grid(axis="y", alpha=0.3)
        ax.set_title(f"All Metrics — {ds} ({short})")
        ax.set_ylim(0, 1.05)

        save_fig(fig, f"fig02_multimetric_{ds}", plt)


# ======================================================================
# FIGURE 3: Convergence Curves (accuracy, all datasets in subplots)
# ======================================================================

def fig_03_convergence_accuracy(completed, plt):
    """5 subplots: accuracy convergence per dataset."""
    fig, axes = plt.subplots(1, len(ALL_DATASETS), figsize=(4 * len(ALL_DATASETS), 4), sharey=False)

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
            ax.plot(rounds, mean, color=CORE_COLORS[i], label=algo, linewidth=1.5)
            ax.fill_between(rounds, mean - std, mean + std, color=CORE_COLORS[i], alpha=0.15)

        short = DS_INFO[ds]["short"]
        ax.set_title(short)
        ax.set_xlabel("Round")
        ax.grid(alpha=0.3)
        if ax == axes[0]:
            ax.set_ylabel("Accuracy (%)")

    axes[-1].legend(loc="lower right", fontsize=8)
    fig.suptitle("Accuracy Convergence per Dataset", fontsize=14)
    fig.tight_layout()
    save_fig(fig, "fig03_convergence_accuracy", plt)


# ======================================================================
# FIGURE 4: Loss Convergence Curves
# ======================================================================

def fig_04_convergence_loss(completed, plt):
    """5 subplots: loss convergence per dataset."""
    fig, axes = plt.subplots(1, len(ALL_DATASETS), figsize=(4 * len(ALL_DATASETS), 4), sharey=False)

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
            ax.plot(rounds, mean, color=CORE_COLORS[i], label=algo, linewidth=1.5)
            ax.fill_between(rounds, mean - std, mean + std, color=CORE_COLORS[i], alpha=0.15)

        short = DS_INFO[ds]["short"]
        ax.set_title(short)
        ax.set_xlabel("Round")
        ax.grid(alpha=0.3)
        if ax == axes[0]:
            ax.set_ylabel("Loss")

    axes[-1].legend(loc="upper right", fontsize=8)
    fig.suptitle("Loss Convergence per Dataset", fontsize=14)
    fig.tight_layout()
    save_fig(fig, "fig04_convergence_loss", plt)


# ======================================================================
# FIGURE 5: F1 Convergence Curves
# ======================================================================

def fig_05_convergence_f1(completed, plt):
    """5 subplots: F1 convergence per dataset."""
    fig, axes = plt.subplots(1, len(ALL_DATASETS), figsize=(4 * len(ALL_DATASETS), 4), sharey=False)

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
            ax.plot(rounds, mean, color=CORE_COLORS[i], label=algo, linewidth=1.5)
            ax.fill_between(rounds, mean - std, mean + std, color=CORE_COLORS[i], alpha=0.15)

        short = DS_INFO[ds]["short"]
        ax.set_title(short)
        ax.set_xlabel("Round")
        ax.grid(alpha=0.3)
        if ax == axes[0]:
            ax.set_ylabel("F1 Score")

    axes[-1].legend(loc="lower right", fontsize=8)
    fig.suptitle("F1 Score Convergence per Dataset", fontsize=14)
    fig.tight_layout()
    save_fig(fig, "fig05_convergence_f1", plt)


# ======================================================================
# FIGURE 6: Heatmap — Accuracy (Algorithm x Dataset)
# ======================================================================

def fig_06_heatmap_accuracy(completed, plt):
    """Annotated heatmap: mean accuracy per algo x dataset."""
    data = np.zeros((len(CORE_ALGOS), len(ALL_DATASETS)))
    for i, algo in enumerate(CORE_ALGOS):
        for j, ds in enumerate(ALL_DATASETS):
            m, _ = agg_metric(completed, ds, algo, "accuracy")
            data[i, j] = m * 100

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(data, cmap="YlOrRd", aspect="auto", vmin=0, vmax=100)
    fig.colorbar(im, ax=ax, label="Accuracy (%)")

    labels_x = [DS_INFO[d]["short"] for d in ALL_DATASETS]
    ax.set_xticks(np.arange(len(ALL_DATASETS)))
    ax.set_xticklabels(labels_x)
    ax.set_yticks(np.arange(len(CORE_ALGOS)))
    ax.set_yticklabels(CORE_ALGOS)

    for i in range(len(CORE_ALGOS)):
        for j in range(len(ALL_DATASETS)):
            color = "white" if data[i, j] > 60 else "black"
            ax.text(j, i, f"{data[i, j]:.1f}", ha="center", va="center",
                    color=color, fontsize=11, fontweight="bold")

    ax.set_title("Accuracy Heatmap: Algorithm x Dataset")
    fig.tight_layout()
    save_fig(fig, "fig06_heatmap_accuracy", plt)


# ======================================================================
# FIGURE 7: Heatmap — F1 Score
# ======================================================================

def fig_07_heatmap_f1(completed, plt):
    """Annotated heatmap: mean F1 per algo x dataset."""
    data = np.zeros((len(CORE_ALGOS), len(ALL_DATASETS)))
    for i, algo in enumerate(CORE_ALGOS):
        for j, ds in enumerate(ALL_DATASETS):
            m, _ = agg_metric(completed, ds, algo, "f1")
            data[i, j] = m

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(data, cmap="YlGnBu", aspect="auto", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, label="F1 Score")

    labels_x = [DS_INFO[d]["short"] for d in ALL_DATASETS]
    ax.set_xticks(np.arange(len(ALL_DATASETS)))
    ax.set_xticklabels(labels_x)
    ax.set_yticks(np.arange(len(CORE_ALGOS)))
    ax.set_yticklabels(CORE_ALGOS)

    for i in range(len(CORE_ALGOS)):
        for j in range(len(ALL_DATASETS)):
            color = "white" if data[i, j] > 0.5 else "black"
            ax.text(j, i, f"{data[i, j]:.3f}", ha="center", va="center",
                    color=color, fontsize=11, fontweight="bold")

    ax.set_title("F1 Score Heatmap: Algorithm x Dataset")
    fig.tight_layout()
    save_fig(fig, "fig07_heatmap_f1", plt)


# ======================================================================
# FIGURE 8: Heatmap — AUC
# ======================================================================

def fig_08_heatmap_auc(completed, plt):
    """Annotated heatmap: mean AUC per algo x dataset."""
    data = np.zeros((len(CORE_ALGOS), len(ALL_DATASETS)))
    for i, algo in enumerate(CORE_ALGOS):
        for j, ds in enumerate(ALL_DATASETS):
            m, _ = agg_metric(completed, ds, algo, "auc")
            data[i, j] = m

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(data, cmap="PuBuGn", aspect="auto", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, label="AUC")

    labels_x = [DS_INFO[d]["short"] for d in ALL_DATASETS]
    ax.set_xticks(np.arange(len(ALL_DATASETS)))
    ax.set_xticklabels(labels_x)
    ax.set_yticks(np.arange(len(CORE_ALGOS)))
    ax.set_yticklabels(CORE_ALGOS)

    for i in range(len(CORE_ALGOS)):
        for j in range(len(ALL_DATASETS)):
            color = "white" if data[i, j] > 0.6 else "black"
            ax.text(j, i, f"{data[i, j]:.3f}", ha="center", va="center",
                    color=color, fontsize=11, fontweight="bold")

    ax.set_title("AUC Heatmap: Algorithm x Dataset")
    fig.tight_layout()
    save_fig(fig, "fig08_heatmap_auc", plt)


# ======================================================================
# FIGURE 9: Radar Chart — Multi-Metric per Algorithm (avg across datasets)
# ======================================================================

def fig_09_radar_chart(completed, plt):
    """Radar/spider chart: 5 metrics per algorithm, averaged across all datasets."""
    from matplotlib.patches import FancyBboxPatch

    angles = np.linspace(0, 2 * np.pi, len(METRICS), endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for i, algo in enumerate(CORE_ALGOS):
        values = []
        for m in METRICS:
            vals = []
            for ds in ALL_DATASETS:
                mean, _ = agg_metric(completed, ds, algo, m)
                vals.append(mean)
            values.append(np.mean(vals))
        values += values[:1]  # close
        ax.plot(angles, values, "o-", color=CORE_COLORS[i], label=algo, linewidth=2)
        ax.fill(angles, values, color=CORE_COLORS[i], alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([METRIC_LABELS[m] for m in METRICS])
    ax.set_ylim(0, 1)
    ax.set_title("Multi-Metric Radar — Average Across All Datasets", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    save_fig(fig, "fig09_radar_multimetric", plt)


# ======================================================================
# FIGURE 10: Radar Charts per Dataset (one radar per dataset)
# ======================================================================

def fig_10_radar_per_dataset(completed, plt):
    """One radar per dataset, 5 subplots."""
    angles = np.linspace(0, 2 * np.pi, len(METRICS), endpoint=False).tolist()
    angles += angles[:1]

    fig, axes = plt.subplots(1, len(ALL_DATASETS), figsize=(5 * len(ALL_DATASETS), 5),
                              subplot_kw=dict(polar=True))

    for ax, ds in zip(axes, ALL_DATASETS):
        for i, algo in enumerate(CORE_ALGOS):
            values = []
            for m in METRICS:
                mean, _ = agg_metric(completed, ds, algo, m)
                values.append(mean)
            values += values[:1]
            ax.plot(angles, values, "o-", color=CORE_COLORS[i], label=algo, linewidth=1.5)
            ax.fill(angles, values, color=CORE_COLORS[i], alpha=0.08)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([METRIC_LABELS[m] for m in METRICS], fontsize=7)
        ax.set_ylim(0, 1)
        ax.set_title(DS_INFO[ds]["short"], fontsize=13, pad=15)

    axes[-1].legend(loc="lower right", bbox_to_anchor=(1.5, -0.1), fontsize=8)
    fig.suptitle("Multi-Metric Radar per Dataset", fontsize=14, y=1.02)
    fig.tight_layout()
    save_fig(fig, "fig10_radar_per_dataset", plt)


# ======================================================================
# FIGURE 11: Box Plots — Seed Variability (accuracy per dataset)
# ======================================================================

def fig_11_boxplot_seeds(completed, plt):
    """Box plots showing seed variability per algorithm for each dataset."""
    fig, axes = plt.subplots(1, len(ALL_DATASETS), figsize=(4 * len(ALL_DATASETS), 5), sharey=False)

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

        bp = ax.boxplot(data_box, labels=labels, patch_artist=True, widths=0.6)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        # Add individual points
        for j, vals in enumerate(data_box):
            jitter = np.random.normal(0, 0.04, len(vals))
            ax.scatter(np.ones(len(vals)) * (j + 1) + jitter, vals,
                      color=colors[j], s=40, zorder=5, edgecolor="black", linewidth=0.5)

        ax.set_title(DS_INFO[ds]["short"])
        ax.set_ylabel("Accuracy (%)" if ax == axes[0] else "")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Accuracy Variability Across Seeds (3 seeds per experiment)", fontsize=14)
    fig.tight_layout()
    save_fig(fig, "fig11_boxplot_seed_variability", plt)


# ======================================================================
# FIGURE 12: Per-Client Accuracy Distribution (Fairness Visualization)
# ======================================================================

def fig_12_client_distribution(completed, plt):
    """Per-client accuracy as scatter + range bars for each algorithm per dataset."""
    fig, axes = plt.subplots(1, len(ALL_DATASETS), figsize=(4 * len(ALL_DATASETS), 5), sharey=False)

    for ax, ds in zip(axes, ALL_DATASETS):
        for i, algo in enumerate(CORE_ALGOS):
            all_client_accs = []
            for seed in SEEDS:
                rec = completed.get(f"{ds}_{algo}_{seed}", {})
                pca = rec.get("per_client_acc", {})
                all_client_accs.extend(pca.values())

            if all_client_accs:
                # Scatter all client accuracies
                x_pos = np.ones(len(all_client_accs)) * i + np.random.normal(0, 0.1, len(all_client_accs))
                ax.scatter(x_pos, [v * 100 for v in all_client_accs],
                          color=CORE_COLORS[i], s=25, alpha=0.6, edgecolor="black", linewidth=0.3)
                # Mean line
                mean_val = np.mean(all_client_accs) * 100
                ax.plot([i - 0.3, i + 0.3], [mean_val, mean_val],
                       color=CORE_COLORS[i], linewidth=2.5)

        ax.set_xticks(range(len(CORE_ALGOS)))
        ax.set_xticklabels(CORE_ALGOS, rotation=45, fontsize=8)
        ax.set_title(DS_INFO[ds]["short"])
        ax.set_ylabel("Per-Client Accuracy (%)" if ax == axes[0] else "")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Per-Client Accuracy Distribution (Fairness View)", fontsize=14)
    fig.tight_layout()
    save_fig(fig, "fig12_client_distribution", plt)


# ======================================================================
# FIGURE 13: Fairness — Jain's Index Bars
# ======================================================================

def fig_13_fairness_jain(completed, plt):
    """Jain's fairness index grouped bars."""
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(ALL_DATASETS))
    width = 0.15

    for i, algo in enumerate(CORE_ALGOS):
        vals = []
        for ds in ALL_DATASETS:
            jains = []
            for seed in SEEDS:
                rec = completed.get(f"{ds}_{algo}_{seed}", {})
                fair = rec.get("fairness", {})
                if fair and "jain_index" in fair:
                    jains.append(fair["jain_index"])
            vals.append(np.mean(jains) if jains else 0)
        ax.bar(x + i * width, vals, width, label=algo,
               color=CORE_COLORS[i], alpha=0.85)

    labels = [DS_INFO[d]["short"] for d in ALL_DATASETS]
    ax.set_ylabel("Jain's Fairness Index")
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(labels)
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0.5, 1.05)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="_nolegend_")
    ax.set_title("Inter-Hospital Fairness (Jain's Index)")

    save_fig(fig, "fig13_fairness_jain", plt)


# ======================================================================
# FIGURE 14: Fairness — Gini Coefficient
# ======================================================================

def fig_14_fairness_gini(completed, plt):
    """Gini coefficient (lower = fairer)."""
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(ALL_DATASETS))
    width = 0.15

    for i, algo in enumerate(CORE_ALGOS):
        vals = []
        for ds in ALL_DATASETS:
            ginis = []
            for seed in SEEDS:
                rec = completed.get(f"{ds}_{algo}_{seed}", {})
                fair = rec.get("fairness", {})
                if fair and "gini" in fair:
                    ginis.append(fair["gini"])
            vals.append(np.mean(ginis) if ginis else 0)
        ax.bar(x + i * width, vals, width, label=algo,
               color=CORE_COLORS[i], alpha=0.85)

    labels = [DS_INFO[d]["short"] for d in ALL_DATASETS]
    ax.set_ylabel("Gini Coefficient (lower = fairer)")
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(labels)
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("Inter-Hospital Inequality (Gini Coefficient)")

    save_fig(fig, "fig14_fairness_gini", plt)


# ======================================================================
# FIGURE 15: Accuracy vs Fairness Scatter
# ======================================================================

def fig_15_accuracy_vs_fairness(completed, plt):
    """Scatter: accuracy (x) vs Jain's index (y), one point per (algo, dataset, seed)."""
    fig, ax = plt.subplots(figsize=(8, 6))

    markers = {"Brain_Tumor": "o", "chest_xray": "s", "Skin_Cancer": "^",
               "Diabetes": "D", "Heart_Disease": "P"}

    for i, algo in enumerate(CORE_ALGOS):
        for ds in ALL_DATASETS:
            accs, jains = [], []
            for seed in SEEDS:
                rec = completed.get(f"{ds}_{algo}_{seed}", {})
                fm = rec.get("final_metrics", {})
                fair = rec.get("fairness", {})
                if fm and fair:
                    accs.append(fm["accuracy"] * 100)
                    jains.append(fair.get("jain_index", 0))
            if accs:
                ax.scatter(accs, jains, color=CORE_COLORS[i], marker=markers.get(ds, "o"),
                          s=50, alpha=0.7, edgecolor="black", linewidth=0.3)

    # Legend for algorithms (colors)
    for i, algo in enumerate(CORE_ALGOS):
        ax.scatter([], [], color=CORE_COLORS[i], marker="o", s=50, label=algo)
    # Legend for datasets (markers)
    for ds, marker in markers.items():
        ax.scatter([], [], color="gray", marker=marker, s=50,
                  label=DS_INFO[ds]["short"])

    ax.set_xlabel("Accuracy (%)")
    ax.set_ylabel("Jain's Fairness Index")
    ax.set_title("Accuracy-Fairness Trade-off")
    ax.legend(loc="lower left", fontsize=8, ncol=2)
    ax.grid(alpha=0.3)

    save_fig(fig, "fig15_accuracy_vs_fairness", plt)


# ======================================================================
# FIGURE 16: Runtime Comparison
# ======================================================================

def fig_16_runtime(completed, plt):
    """Bar chart: average runtime per algorithm per dataset."""
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(ALL_DATASETS))
    width = 0.15

    for i, algo in enumerate(CORE_ALGOS):
        means = []
        for ds in ALL_DATASETS:
            runtimes = []
            for seed in SEEDS:
                rec = completed.get(f"{ds}_{algo}_{seed}", {})
                rt = rec.get("runtime_seconds", 0)
                if rt > 0:
                    runtimes.append(rt / 60)  # minutes
            means.append(np.mean(runtimes) if runtimes else 0)
        ax.bar(x + i * width, means, width, label=algo,
               color=CORE_COLORS[i], alpha=0.85)

    labels = [DS_INFO[d]["short"] for d in ALL_DATASETS]
    ax.set_ylabel("Training Time (min)")
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(labels)
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("Training Time per Experiment (Mac MPS)")

    save_fig(fig, "fig16_runtime", plt)


# ======================================================================
# FIGURE 17: Early Stopping Analysis
# ======================================================================

def fig_17_early_stopping(completed, plt):
    """Histogram: actual rounds completed (12 = full, <12 = early stop)."""
    rounds_data = defaultdict(list)

    for key, val in completed.items():
        ds = val["dataset"]
        algo = val["algorithm"]
        if algo not in CORE_ALGOS:
            continue
        h = val.get("history", [])
        config = val.get("config", {})
        max_rounds = config.get("num_rounds", len(h))
        rounds_data[ds].append(len(h))

    fig, axes = plt.subplots(1, len(ALL_DATASETS), figsize=(4 * len(ALL_DATASETS), 4), sharey=True)

    for ax, ds in zip(axes, ALL_DATASETS):
        data_r = rounds_data.get(ds, [])
        if data_r:
            bins = range(1, max(data_r) + 2)
            ax.hist(data_r, bins=bins, color="#2196F3", alpha=0.7, edgecolor="black", align="left")
        ax.set_title(DS_INFO[ds]["short"])
        ax.set_xlabel("Actual Rounds")
        if ax == axes[0]:
            ax.set_ylabel("Count")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Early Stopping: Distribution of Actual Training Rounds", fontsize=14)
    fig.tight_layout()
    save_fig(fig, "fig17_early_stopping", plt)


# ======================================================================
# FIGURE 18: Tabular Extended — All 9 Algorithms (Diabetes + Heart)
# ======================================================================

def fig_18_tabular_extended(completed, plt):
    """Grouped bars: 9 algorithms on Diabetes and Heart_Disease."""
    tab_ds = list(TABULAR_DATASETS.keys())
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, ds in zip(axes, tab_ds):
        algos_present = []
        for algo in EXTENDED_ALGOS:
            rec = completed.get(f"{ds}_{algo}_42", {})
            if rec.get("final_metrics"):
                algos_present.append(algo)

        x = np.arange(len(algos_present))
        means, stds = [], []
        colors = []
        for algo in algos_present:
            m, s = agg_metric(completed, ds, algo, "accuracy")
            means.append(m * 100)
            stds.append(s * 100)
            idx = EXTENDED_ALGOS.index(algo) if algo in EXTENDED_ALGOS else 0
            colors.append(EXTENDED_COLORS[idx])

        bars = ax.bar(x, means, yerr=stds, color=colors, capsize=3, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(algos_present, rotation=45, ha="right")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(f"{ds} ({DS_INFO[ds]['short']})")
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0, 100)

        # Add value labels
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                   f"{mean:.1f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("Tabular Datasets: Full Algorithm Comparison (9 algorithms)", fontsize=14)
    fig.tight_layout()
    save_fig(fig, "fig18_tabular_9algo", plt)


# ======================================================================
# FIGURE 19: Algorithm Ranking Visualization
# ======================================================================

def fig_19_ranking(completed, plt):
    """Bump chart: algorithm ranking per dataset by accuracy."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ranks_per_algo = {algo: [] for algo in CORE_ALGOS}

    for j, ds in enumerate(ALL_DATASETS):
        algo_means = []
        for algo in CORE_ALGOS:
            m, _ = agg_metric(completed, ds, algo, "accuracy")
            algo_means.append((algo, m))
        algo_means.sort(key=lambda x: -x[1])  # sort descending
        for rank, (algo, _) in enumerate(algo_means):
            ranks_per_algo[algo].append(rank + 1)

    x = np.arange(len(ALL_DATASETS))
    labels = [DS_INFO[d]["short"] for d in ALL_DATASETS]

    for i, algo in enumerate(CORE_ALGOS):
        ax.plot(x, ranks_per_algo[algo], "o-", color=CORE_COLORS[i], label=algo,
                linewidth=2.5, markersize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Rank (1 = best)")
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.invert_yaxis()
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    ax.set_title("Algorithm Ranking Across Datasets (by Accuracy)")

    save_fig(fig, "fig19_ranking_bump", plt)


# ======================================================================
# FIGURE 20: Convergence per Dataset (individual large plots)
# ======================================================================

def fig_20_convergence_individual(completed, plt):
    """Large individual convergence plot per dataset (accuracy + loss)."""
    for ds in ALL_DATASETS:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        for i, algo in enumerate(CORE_ALGOS):
            # Accuracy
            hists = get_histories(completed, ds, algo, "accuracy")
            if hists:
                min_len = min(len(h) for h in hists)
                arr = np.array([h[:min_len] for h in hists])
                mean_a = arr.mean(axis=0) * 100
                std_a = arr.std(axis=0) * 100
                rounds = np.arange(1, min_len + 1)
                ax1.plot(rounds, mean_a, color=CORE_COLORS[i], label=algo, linewidth=2)
                ax1.fill_between(rounds, mean_a - std_a, mean_a + std_a,
                                color=CORE_COLORS[i], alpha=0.15)

            # Loss
            hists_l = get_histories(completed, ds, algo, "loss")
            if hists_l:
                min_len_l = min(len(h) for h in hists_l)
                arr_l = np.array([h[:min_len_l] for h in hists_l])
                mean_l = arr_l.mean(axis=0)
                std_l = arr_l.std(axis=0)
                rounds_l = np.arange(1, min_len_l + 1)
                ax2.plot(rounds_l, mean_l, color=CORE_COLORS[i], label=algo, linewidth=2)
                ax2.fill_between(rounds_l, mean_l - std_l, mean_l + std_l,
                                color=CORE_COLORS[i], alpha=0.15)

        ax1.set_xlabel("Round")
        ax1.set_ylabel("Accuracy (%)")
        ax1.set_title("Accuracy Convergence")
        ax1.legend()
        ax1.grid(alpha=0.3)

        ax2.set_xlabel("Round")
        ax2.set_ylabel("Loss")
        ax2.set_title("Loss Convergence")
        ax2.legend()
        ax2.grid(alpha=0.3)

        fig.suptitle(f"{ds} — Convergence Analysis", fontsize=14)
        fig.tight_layout()
        save_fig(fig, f"fig20_convergence_{ds}", plt)


# ======================================================================
# FIGURE 21: Precision-Recall Balance
# ======================================================================

def fig_21_precision_recall(completed, plt):
    """Scatter: precision vs recall per algo/dataset."""
    fig, ax = plt.subplots(figsize=(8, 6))

    markers = {"Brain_Tumor": "o", "chest_xray": "s", "Skin_Cancer": "^",
               "Diabetes": "D", "Heart_Disease": "P"}

    for i, algo in enumerate(CORE_ALGOS):
        for ds in ALL_DATASETS:
            precs, recs = [], []
            for seed in SEEDS:
                rec = completed.get(f"{ds}_{algo}_{seed}", {})
                fm = rec.get("final_metrics", {})
                if fm:
                    precs.append(fm.get("precision", 0))
                    recs.append(fm.get("recall", 0))
            if precs:
                ax.scatter(np.mean(precs), np.mean(recs),
                          color=CORE_COLORS[i], marker=markers.get(ds, "o"),
                          s=80, edgecolor="black", linewidth=0.5, alpha=0.8)

    # Legends
    for i, algo in enumerate(CORE_ALGOS):
        ax.scatter([], [], color=CORE_COLORS[i], marker="o", s=80, label=algo)
    for ds, marker in markers.items():
        ax.scatter([], [], color="gray", marker=marker, s=80, label=DS_INFO[ds]["short"])

    # Diagonal
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="_nolegend_")

    ax.set_xlabel("Precision")
    ax.set_ylabel("Recall")
    ax.set_title("Precision-Recall Balance")
    ax.legend(loc="lower left", fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)

    save_fig(fig, "fig21_precision_recall", plt)


# ======================================================================
# FIGURE 22: Convergence Speed (rounds to 90% of final accuracy)
# ======================================================================

def fig_22_convergence_speed(completed, plt):
    """How fast does each algorithm reach near-final accuracy?"""
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(ALL_DATASETS))
    width = 0.15

    for i, algo in enumerate(CORE_ALGOS):
        speeds = []
        for ds in ALL_DATASETS:
            round_to_90 = []
            for seed in SEEDS:
                rec = completed.get(f"{ds}_{algo}_{seed}", {})
                h = rec.get("history", [])
                fm = rec.get("final_metrics", {})
                if h and fm:
                    final_acc = fm["accuracy"]
                    threshold = final_acc * 0.90
                    reached = len(h)  # default: all rounds
                    for r_entry in h:
                        if r_entry["accuracy"] >= threshold:
                            reached = r_entry["round"]
                            break
                    round_to_90.append(reached)
            speeds.append(np.mean(round_to_90) if round_to_90 else 0)

        ax.bar(x + i * width, speeds, width, label=algo,
               color=CORE_COLORS[i], alpha=0.85)

    labels = [DS_INFO[d]["short"] for d in ALL_DATASETS]
    ax.set_ylabel("Rounds to 90% of Final Accuracy")
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(labels)
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("Convergence Speed: Rounds to Reach 90% of Final Performance")

    save_fig(fig, "fig22_convergence_speed", plt)


# ======================================================================
# FIGURE 23: Tabular Convergence (extended algos)
# ======================================================================

def fig_23_tabular_convergence(completed, plt):
    """Convergence curves for tabular datasets with all 9 algorithms."""
    for ds in TABULAR_DATASETS.keys():
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        algos_present = []
        for algo in EXTENDED_ALGOS:
            rec = completed.get(f"{ds}_{algo}_42", {})
            if rec.get("history"):
                algos_present.append(algo)

        for algo in algos_present:
            idx = EXTENDED_ALGOS.index(algo)
            color = EXTENDED_COLORS[idx]

            hists_a = get_histories(completed, ds, algo, "accuracy")
            if hists_a:
                min_len = min(len(h) for h in hists_a)
                arr = np.array([h[:min_len] for h in hists_a])
                mean_a = arr.mean(axis=0) * 100
                rounds = np.arange(1, min_len + 1)
                ax1.plot(rounds, mean_a, color=color, label=algo, linewidth=1.5)

            hists_l = get_histories(completed, ds, algo, "loss")
            if hists_l:
                min_len = min(len(h) for h in hists_l)
                arr = np.array([h[:min_len] for h in hists_l])
                mean_l = arr.mean(axis=0)
                rounds = np.arange(1, min_len + 1)
                ax2.plot(rounds, mean_l, color=color, label=algo, linewidth=1.5)

        ax1.set_xlabel("Round")
        ax1.set_ylabel("Accuracy (%)")
        ax1.set_title("Accuracy")
        ax1.legend(fontsize=7)
        ax1.grid(alpha=0.3)

        ax2.set_xlabel("Round")
        ax2.set_ylabel("Loss")
        ax2.set_title("Loss")
        ax2.legend(fontsize=7)
        ax2.grid(alpha=0.3)

        fig.suptitle(f"{ds} — 9-Algorithm Convergence", fontsize=14)
        fig.tight_layout()
        save_fig(fig, f"fig23_tabular_convergence_{ds}", plt)


# ======================================================================
# FIGURE 24: Summary Table as Figure (publishable overview)
# ======================================================================

def fig_24_summary_table(completed, plt):
    """Publication-quality table rendered as figure."""
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    # Build data matrix
    header_metrics = ["Acc (%)", "F1", "AUC", "Jain"]
    n_cols = len(ALL_DATASETS) * len(header_metrics)

    fig, ax = plt.subplots(figsize=(18, 4))
    ax.axis("off")

    col_labels = []
    for ds in ALL_DATASETS:
        short = DS_INFO[ds]["short"]
        for hm in header_metrics:
            col_labels.append(f"{short}\n{hm}")

    cell_data = []
    for algo in CORE_ALGOS:
        row = []
        for ds in ALL_DATASETS:
            m_acc, _ = agg_metric(completed, ds, algo, "accuracy")
            m_f1, _ = agg_metric(completed, ds, algo, "f1")
            m_auc, _ = agg_metric(completed, ds, algo, "auc")
            # Jain
            jains = []
            for seed in SEEDS:
                rec = completed.get(f"{ds}_{algo}_{seed}", {})
                fair = rec.get("fairness", {})
                if fair and "jain_index" in fair:
                    jains.append(fair["jain_index"])
            m_jain = np.mean(jains) if jains else 0

            row.extend([f"{m_acc*100:.1f}", f"{m_f1:.3f}", f"{m_auc:.3f}", f"{m_jain:.3f}"])
        cell_data.append(row)

    table = ax.table(cellText=cell_data,
                     rowLabels=CORE_ALGOS,
                     colLabels=col_labels,
                     cellLoc="center",
                     loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)

    # Color header
    for j in range(n_cols):
        table[0, j].set_facecolor("#E3F2FD")
        table[0, j].set_text_props(fontweight="bold")
    for i in range(len(CORE_ALGOS)):
        table[i + 1, -1].set_text_props(fontweight="bold")

    ax.set_title("Summary: All Metrics per Algorithm x Dataset (mean over 3 seeds)", fontsize=13, pad=20)
    fig.tight_layout()
    save_fig(fig, "fig24_summary_table", plt)


# ======================================================================
# FIGURE 25: AUC Convergence Curves
# ======================================================================

def fig_25_convergence_auc(completed, plt):
    """5 subplots: AUC convergence per dataset."""
    fig, axes = plt.subplots(1, len(ALL_DATASETS), figsize=(4 * len(ALL_DATASETS), 4), sharey=False)

    for ax, ds in zip(axes, ALL_DATASETS):
        for i, algo in enumerate(CORE_ALGOS):
            hists = get_histories(completed, ds, algo, "auc")
            if not hists:
                continue
            min_len = min(len(h) for h in hists)
            arr = np.array([h[:min_len] for h in hists])
            mean = arr.mean(axis=0)
            std = arr.std(axis=0)
            rounds = np.arange(1, min_len + 1)
            ax.plot(rounds, mean, color=CORE_COLORS[i], label=algo, linewidth=1.5)
            ax.fill_between(rounds, mean - std, mean + std, color=CORE_COLORS[i], alpha=0.15)

        ax.set_title(DS_INFO[ds]["short"])
        ax.set_xlabel("Round")
        ax.grid(alpha=0.3)
        if ax == axes[0]:
            ax.set_ylabel("AUC")

    axes[-1].legend(loc="lower right", fontsize=8)
    fig.suptitle("AUC Convergence per Dataset", fontsize=14)
    fig.tight_layout()
    save_fig(fig, "fig25_convergence_auc", plt)


# ======================================================================
# FIGURE 26: Client Fairness Gap (max - min client accuracy)
# ======================================================================

def fig_26_fairness_gap(completed, plt):
    """Bar chart: gap between best and worst client per algo x dataset."""
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(ALL_DATASETS))
    width = 0.15

    for i, algo in enumerate(CORE_ALGOS):
        gaps = []
        for ds in ALL_DATASETS:
            gap_vals = []
            for seed in SEEDS:
                rec = completed.get(f"{ds}_{algo}_{seed}", {})
                fair = rec.get("fairness", {})
                if fair:
                    gap = (fair.get("max", 0) - fair.get("min", 0)) * 100
                    gap_vals.append(gap)
            gaps.append(np.mean(gap_vals) if gap_vals else 0)
        ax.bar(x + i * width, gaps, width, label=algo,
               color=CORE_COLORS[i], alpha=0.85)

    labels = [DS_INFO[d]["short"] for d in ALL_DATASETS]
    ax.set_ylabel("Accuracy Gap (max - min client) (%)")
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(labels)
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("Fairness Gap: Max-Min Client Accuracy Difference")

    save_fig(fig, "fig26_fairness_gap", plt)


# ======================================================================
# FIGURE 27: Imaging vs Tabular Comparison
# ======================================================================

def fig_27_imaging_vs_tabular(completed, plt):
    """Side-by-side: imaging datasets (left) vs tabular (right)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Imaging
    img_ds = list(IMAGING_DATASETS.keys())
    x1 = np.arange(len(img_ds))
    width = 0.15
    for i, algo in enumerate(CORE_ALGOS):
        means = [agg_metric(completed, ds, algo, "accuracy")[0] * 100 for ds in img_ds]
        ax1.bar(x1 + i * width, means, width, label=algo, color=CORE_COLORS[i], alpha=0.85)
    ax1.set_xticks(x1 + width * 2)
    ax1.set_xticklabels([DS_INFO[d]["short"] for d in img_ds])
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_title("Imaging Datasets (ResNet18)")
    ax1.legend(fontsize=8)
    ax1.grid(axis="y", alpha=0.3)
    ax1.set_ylim(0, 100)

    # Tabular
    tab_ds = list(TABULAR_DATASETS.keys())
    x2 = np.arange(len(tab_ds))
    for i, algo in enumerate(CORE_ALGOS):
        means = [agg_metric(completed, ds, algo, "accuracy")[0] * 100 for ds in tab_ds]
        ax2.bar(x2 + i * width, means, width, label=algo, color=CORE_COLORS[i], alpha=0.85)
    ax2.set_xticks(x2 + width * 2)
    ax2.set_xticklabels([DS_INFO[d]["short"] for d in tab_ds])
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Tabular Datasets (MLP)")
    ax2.legend(fontsize=8)
    ax2.grid(axis="y", alpha=0.3)
    ax2.set_ylim(0, 100)

    fig.suptitle("Imaging vs. Tabular: Algorithm Performance", fontsize=14)
    fig.tight_layout()
    save_fig(fig, "fig27_imaging_vs_tabular", plt)


# ======================================================================
# Main
# ======================================================================

def main():
    print("=" * 60)
    print("FL-EHDS Comprehensive Figure Generator")
    print("=" * 60)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output: {FIGURES_DIR}\n")

    completed = load_data()
    print(f"Loaded {len(completed)} experiment results\n")

    plt = setup_plt()

    # Generate all figures
    figures = [
        ("01", "Multi-Dataset Accuracy Bars", fig_01_accuracy_bars),
        ("02", "Multi-Metric Bars per Dataset", fig_02_multimetric_bars),
        ("03", "Convergence: Accuracy", fig_03_convergence_accuracy),
        ("04", "Convergence: Loss", fig_04_convergence_loss),
        ("05", "Convergence: F1", fig_05_convergence_f1),
        ("06", "Heatmap: Accuracy", fig_06_heatmap_accuracy),
        ("07", "Heatmap: F1 Score", fig_07_heatmap_f1),
        ("08", "Heatmap: AUC", fig_08_heatmap_auc),
        ("09", "Radar: Multi-Metric Average", fig_09_radar_chart),
        ("10", "Radar: Per Dataset", fig_10_radar_per_dataset),
        ("11", "Box Plots: Seed Variability", fig_11_boxplot_seeds),
        ("12", "Per-Client Accuracy Distribution", fig_12_client_distribution),
        ("13", "Fairness: Jain's Index", fig_13_fairness_jain),
        ("14", "Fairness: Gini Coefficient", fig_14_fairness_gini),
        ("15", "Accuracy vs Fairness Scatter", fig_15_accuracy_vs_fairness),
        ("16", "Runtime Comparison", fig_16_runtime),
        ("17", "Early Stopping Analysis", fig_17_early_stopping),
        ("18", "Tabular: 9-Algorithm Comparison", fig_18_tabular_extended),
        ("19", "Algorithm Ranking (Bump Chart)", fig_19_ranking),
        ("20", "Individual Convergence (Acc+Loss)", fig_20_convergence_individual),
        ("21", "Precision vs Recall", fig_21_precision_recall),
        ("22", "Convergence Speed", fig_22_convergence_speed),
        ("23", "Tabular 9-Algo Convergence", fig_23_tabular_convergence),
        ("24", "Summary Table Figure", fig_24_summary_table),
        ("25", "Convergence: AUC", fig_25_convergence_auc),
        ("26", "Fairness Gap (Max-Min)", fig_26_fairness_gap),
        ("27", "Imaging vs Tabular", fig_27_imaging_vs_tabular),
    ]

    for num, desc, fn in figures:
        print(f"[{num}] {desc}")
        try:
            fn(completed, plt)
        except Exception as e:
            print(f"  ERROR: {e}")

    # Count generated files
    png_count = len(list(FIGURES_DIR.glob("*.png")))
    pdf_count = len(list(FIGURES_DIR.glob("*.pdf")))
    print(f"\n{'='*60}")
    print(f"Done! Generated {png_count} PNG + {pdf_count} PDF files")
    print(f"Output: {FIGURES_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

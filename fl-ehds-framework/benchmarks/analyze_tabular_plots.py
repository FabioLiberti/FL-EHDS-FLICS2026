#!/usr/bin/env python3
"""
FL-EHDS Extended Tabular Visualization — Paper-ready figures.

Generates comprehensive plots from tabular experiment results:
  1.  Accuracy bar chart (grouped, per dataset)
  2.  Fairness bar chart (Jain index + Gini)
  3.  Box plots — accuracy variance across seeds
  4.  Per-client accuracy heatmaps
  5.  Loss convergence curves
  6.  F1 score convergence curves
  7.  Rounds-to-converge (communication efficiency)
  8.  Accuracy vs Fairness scatter (Pareto)
  9.  Multi-metric radar charts
  10. Data distribution (samples per client, non-IID)
  11. Algorithm ranking heatmap
  12. Early stopping analysis
  13. Runtime comparison

Usage:
    cd fl-ehds-framework
    python -m benchmarks.analyze_tabular_plots

Output: benchmarks/paper_results_tabular/analysis/plots/

Author: Fabio Liberti
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
from matplotlib.patches import FancyBboxPatch

FRAMEWORK_DIR = Path(__file__).parent.parent
CHECKPOINT_PATH = FRAMEWORK_DIR / "benchmarks" / "paper_results_tabular" / "checkpoint_tabular.json"
PLOT_DIR = FRAMEWORK_DIR / "benchmarks" / "paper_results_tabular" / "analysis" / "plots"

DATASETS = ["PTB_XL", "Cardiovascular", "Breast_Cancer"]
DATASET_LABELS = {
    "PTB_XL": "PTB-XL\n(5-class ECG, EU)",
    "Cardiovascular": "Cardiovascular\n(binary, 70K)",
    "Breast_Cancer": "Breast Cancer\n(binary, 569)",
}
DATASET_SHORT = {"PTB_XL": "PTB-XL", "Cardiovascular": "Cardio", "Breast_Cancer": "Breast Ca."}

ALGORITHMS = ["FedAvg", "FedProx", "Ditto", "FedLC", "FedExP", "FedLESAM", "HPFL"]

ALGO_COLORS = {
    "FedAvg": "#2196F3", "FedProx": "#FF5722", "Ditto": "#4CAF50",
    "FedLC": "#E91E63", "FedExP": "#9C27B0", "FedLESAM": "#FF9800",
    "HPFL": "#00BCD4",
}

ALGO_HATCHES = {
    "FedAvg": "", "FedProx": "//", "Ditto": "\\\\",
    "FedLC": "xx", "FedExP": "..", "FedLESAM": "++",
    "HPFL": "**",
}

# Global style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 150,
})


def load_results():
    with open(CHECKPOINT_PATH) as f:
        data = json.load(f)
    completed = data.get("completed", {})
    results = defaultdict(lambda: defaultdict(list))
    for key, res in completed.items():
        if "error" in res:
            continue
        results[res["dataset"]][res["algorithm"]].append(res)
    return dict(results)


def _get_accs(results, ds, algo):
    return [r["best_metrics"]["accuracy"] for r in results[ds][algo]]


# ======================================================================
# 1. Accuracy Bar Chart (grouped)
# ======================================================================

def plot_accuracy_bars(results):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)

    for idx, ds in enumerate(DATASETS):
        ax = axes[idx]
        algos = [a for a in ALGORITHMS if a in results.get(ds, {})]
        means = [np.mean(_get_accs(results, ds, a)) * 100 for a in algos]
        stds = [np.std(_get_accs(results, ds, a)) * 100 for a in algos]
        colors = [ALGO_COLORS[a] for a in algos]

        bars = ax.bar(range(len(algos)), means, yerr=stds, capsize=4,
                      color=colors, edgecolor="black", linewidth=0.5, alpha=0.85)

        best_idx = np.argmax(means)
        bars[best_idx].set_edgecolor("gold")
        bars[best_idx].set_linewidth(2.5)

        ax.set_xticks(range(len(algos)))
        ax.set_xticklabels(algos, rotation=45, ha="right", fontsize=9)
        ax.set_title(DATASET_SHORT[ds], fontweight="bold")
        ax.set_ylabel("Accuracy (%)" if idx == 0 else "")
        ax.grid(axis="y", alpha=0.3)

        ymin = max(0, min(means) - max(stds) - 5)
        ax.set_ylim(ymin, min(100, max(means) + max(stds) + 3))

    fig.suptitle("Best Accuracy per Algorithm (mean ± std over 3 seeds)", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "01_accuracy_bars.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  01_accuracy_bars.pdf")


# ======================================================================
# 2. Fairness Bar Chart (Jain + Gini side by side)
# ======================================================================

def plot_fairness_bars(results):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for idx, ds in enumerate(DATASETS):
        ax = axes[idx]
        algos = [a for a in ALGORITHMS if a in results.get(ds, {})]

        jains = []
        ginis = []
        for a in algos:
            j = [r["fairness"]["jain_index"] for r in results[ds][a]]
            g = [r["fairness"]["gini"] for r in results[ds][a]]
            jains.append(np.mean(j))
            ginis.append(np.mean(g))

        x = np.arange(len(algos))
        w = 0.35
        ax.bar(x - w/2, jains, w, label="Jain Index", color="#4CAF50", alpha=0.8, edgecolor="black", linewidth=0.5)
        ax.bar(x + w/2, ginis, w, label="Gini Coeff.", color="#F44336", alpha=0.8, edgecolor="black", linewidth=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels(algos, rotation=45, ha="right", fontsize=9)
        ax.set_title(DATASET_SHORT[ds], fontweight="bold")
        ax.set_ylabel("Score" if idx == 0 else "")
        ax.set_ylim(0, 1.1)
        ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.3, label="Perfect fairness")
        if idx == 2:
            ax.legend(fontsize=8, loc="upper left")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Client Fairness: Jain Index (higher=fairer) vs Gini (lower=fairer)", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "02_fairness_bars.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  02_fairness_bars.pdf")


# ======================================================================
# 3. Box Plots — Accuracy Variance
# ======================================================================

def plot_accuracy_boxplots(results):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for idx, ds in enumerate(DATASETS):
        ax = axes[idx]
        algos = [a for a in ALGORITHMS if a in results.get(ds, {})]

        data = []
        for a in algos:
            accs = [r["best_metrics"]["accuracy"] * 100 for r in results[ds][a]]
            data.append(accs)

        bp = ax.boxplot(data, labels=algos, patch_artist=True, widths=0.6,
                       medianprops=dict(color="black", linewidth=2))
        for patch, algo in zip(bp["boxes"], algos):
            patch.set_facecolor(ALGO_COLORS[algo])
            patch.set_alpha(0.7)

        # Scatter individual points
        for i, (d, algo) in enumerate(zip(data, algos)):
            x_jitter = np.random.normal(i + 1, 0.05, len(d))
            ax.scatter(x_jitter, d, color=ALGO_COLORS[algo], s=30, zorder=5,
                      edgecolor="black", linewidth=0.5)

        ax.set_xticklabels(algos, rotation=45, ha="right", fontsize=9)
        ax.set_title(DATASET_SHORT[ds], fontweight="bold")
        ax.set_ylabel("Accuracy (%)" if idx == 0 else "")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Accuracy Distribution Across Seeds (3 seeds per algorithm)", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "03_accuracy_boxplots.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  03_accuracy_boxplots.pdf")


# ======================================================================
# 4. Per-client Accuracy Heatmaps
# ======================================================================

def plot_client_heatmaps(results):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for idx, ds in enumerate(DATASETS):
        ax = axes[idx]
        algos = [a for a in ALGORITHMS if a in results.get(ds, {})]

        # Average per-client acc across seeds
        n_clients = len(results[ds][algos[0]][0]["per_client_acc"])
        matrix = np.zeros((len(algos), n_clients))

        for i, algo in enumerate(algos):
            for run in results[ds][algo]:
                for cid, acc in run["per_client_acc"].items():
                    matrix[i, int(cid)] += acc
            matrix[i] /= len(results[ds][algo])

        im = ax.imshow(matrix * 100, cmap="RdYlGn", aspect="auto", vmin=0, vmax=100)
        ax.set_yticks(range(len(algos)))
        ax.set_yticklabels(algos, fontsize=9)
        ax.set_xticks(range(n_clients))
        ax.set_xticklabels([f"C{i}" for i in range(n_clients)], fontsize=9)
        ax.set_title(DATASET_SHORT[ds], fontweight="bold")
        ax.set_xlabel("Client")
        if idx == 0:
            ax.set_ylabel("Algorithm")

        # Annotate cells
        for i in range(len(algos)):
            for j in range(n_clients):
                val = matrix[i, j] * 100
                color = "white" if val < 50 else "black"
                ax.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=8, color=color)

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Acc (%)")

    fig.suptitle("Per-Client Accuracy (mean over seeds) — Fairness Visualization", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "04_client_heatmaps.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  04_client_heatmaps.pdf")


# ======================================================================
# 5. Loss Convergence Curves
# ======================================================================

def plot_loss_convergence(results):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, ds in enumerate(DATASETS):
        ax = axes[idx]
        for algo in ALGORITHMS:
            if algo not in results.get(ds, {}):
                continue
            runs = results[ds][algo]
            max_rounds = max(len(r["history"]) for r in runs)
            loss_matrix = np.full((len(runs), max_rounds), np.nan)
            for i, r in enumerate(runs):
                losses = [h["loss"] for h in r["history"]]
                loss_matrix[i, :len(losses)] = losses

            rounds = np.arange(1, max_rounds + 1)
            mean_loss = np.nanmean(loss_matrix, axis=0)
            color = ALGO_COLORS.get(algo, "#666666")
            ax.plot(rounds, mean_loss, color=color, linewidth=1.5, label=algo)

        ax.set_xlabel("Round")
        ax.set_ylabel("Loss" if idx == 0 else "")
        ax.set_title(DATASET_SHORT[ds], fontweight="bold")
        ax.grid(True, alpha=0.3)
        if idx == 2:
            ax.legend(fontsize=8, loc="upper right")

    fig.suptitle("Training Loss Convergence", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "05_loss_convergence.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  05_loss_convergence.pdf")


# ======================================================================
# 6. F1 Score Convergence
# ======================================================================

def plot_f1_convergence(results):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, ds in enumerate(DATASETS):
        ax = axes[idx]
        for algo in ALGORITHMS:
            if algo not in results.get(ds, {}):
                continue
            runs = results[ds][algo]
            max_rounds = max(len(r["history"]) for r in runs)
            f1_matrix = np.full((len(runs), max_rounds), np.nan)
            for i, r in enumerate(runs):
                f1s = [h["f1"] for h in r["history"]]
                f1_matrix[i, :len(f1s)] = f1s

            rounds = np.arange(1, max_rounds + 1)
            mean_f1 = np.nanmean(f1_matrix, axis=0) * 100
            color = ALGO_COLORS.get(algo, "#666666")
            ax.plot(rounds, mean_f1, color=color, linewidth=1.5, label=algo)

        ax.set_xlabel("Round")
        ax.set_ylabel("F1 Score (%)" if idx == 0 else "")
        ax.set_title(DATASET_SHORT[ds], fontweight="bold")
        ax.grid(True, alpha=0.3)
        if idx == 2:
            ax.legend(fontsize=8, loc="lower right")

    fig.suptitle("F1 Score Convergence (macro-averaged)", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "06_f1_convergence.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  06_f1_convergence.pdf")


# ======================================================================
# 7. Rounds to Converge (Communication Efficiency)
# ======================================================================

def plot_rounds_to_converge(results):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for idx, ds in enumerate(DATASETS):
        ax = axes[idx]
        algos = [a for a in ALGORITHMS if a in results.get(ds, {})]

        best_rounds = []
        actual_rounds = []
        for a in algos:
            br = [r["best_round"] for r in results[ds][a]]
            ar = [r["actual_rounds"] for r in results[ds][a]]
            best_rounds.append(np.mean(br))
            actual_rounds.append(np.mean(ar))

        x = np.arange(len(algos))
        w = 0.35
        ax.bar(x - w/2, best_rounds, w, label="Round of best acc",
               color=[ALGO_COLORS[a] for a in algos], alpha=0.85, edgecolor="black", linewidth=0.5)
        ax.bar(x + w/2, actual_rounds, w, label="Total rounds (ES)",
               color=[ALGO_COLORS[a] for a in algos], alpha=0.35, edgecolor="black", linewidth=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels(algos, rotation=45, ha="right", fontsize=9)
        ax.set_title(DATASET_SHORT[ds], fontweight="bold")
        ax.set_ylabel("Rounds" if idx == 0 else "")
        ax.grid(axis="y", alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=8)

    fig.suptitle("Communication Efficiency: Rounds to Best vs Total Rounds", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "07_rounds_to_converge.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  07_rounds_to_converge.pdf")


# ======================================================================
# 8. Accuracy vs Fairness Scatter (Pareto)
# ======================================================================

def plot_accuracy_vs_fairness(results):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, ds in enumerate(DATASETS):
        ax = axes[idx]
        for algo in ALGORITHMS:
            if algo not in results.get(ds, {}):
                continue
            accs = [r["best_metrics"]["accuracy"] * 100 for r in results[ds][algo]]
            jains = [r["fairness"]["jain_index"] for r in results[ds][algo]]

            ax.scatter(np.mean(accs), np.mean(jains),
                      s=120, color=ALGO_COLORS[algo], edgecolor="black",
                      linewidth=1, zorder=5, label=algo)

            # Error bars
            ax.errorbar(np.mean(accs), np.mean(jains),
                       xerr=np.std(accs), yerr=np.std(jains),
                       color=ALGO_COLORS[algo], alpha=0.4, capsize=3, zorder=4)

        ax.set_xlabel("Accuracy (%)")
        ax.set_ylabel("Jain Fairness Index" if idx == 0 else "")
        ax.set_title(DATASET_SHORT[ds], fontweight="bold")
        ax.grid(True, alpha=0.3)

        # Ideal corner annotation
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.annotate("ideal", xy=(xlim[1], 1.0), fontsize=8, color="gray",
                    ha="right", va="top", style="italic")

        if idx == 2:
            ax.legend(fontsize=8, loc="lower left")

    fig.suptitle("Accuracy vs Fairness Trade-off (top-right = best)", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "08_accuracy_vs_fairness.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  08_accuracy_vs_fairness.pdf")


# ======================================================================
# 9. Multi-metric Radar Charts
# ======================================================================

def plot_radar_charts(results):
    metrics = ["Accuracy", "F1", "Precision", "Recall", "Jain", "1-Gini"]
    n_metrics = len(metrics)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), subplot_kw=dict(polar=True))

    for idx, ds in enumerate(DATASETS):
        ax = axes[idx]

        for algo in ALGORITHMS:
            if algo not in results.get(ds, {}):
                continue
            runs = results[ds][algo]

            acc = np.mean([r["best_metrics"]["accuracy"] for r in runs])
            f1 = np.mean([r["final_metrics"]["f1"] for r in runs])
            prec = np.mean([r["final_metrics"]["precision"] for r in runs])
            rec = np.mean([r["final_metrics"]["recall"] for r in runs])
            jain = np.mean([r["fairness"]["jain_index"] for r in runs])
            gini_inv = 1 - np.mean([r["fairness"]["gini"] for r in runs])

            values = [acc, f1, prec, rec, jain, gini_inv]
            values += values[:1]

            color = ALGO_COLORS[algo]
            ax.plot(angles, values, linewidth=1.5, color=color, label=algo)
            ax.fill(angles, values, alpha=0.05, color=color)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics, fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_title(DATASET_SHORT[ds], fontweight="bold", pad=20)
        if idx == 2:
            ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=8)

    fig.suptitle("Multi-Metric Algorithm Comparison (all metrics 0-1 scale)", fontsize=14, y=1.05)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "09_radar_charts.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  09_radar_charts.pdf")


# ======================================================================
# 10. Data Distribution (Samples per Client)
# ======================================================================

def plot_data_distribution(results):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for idx, ds in enumerate(DATASETS):
        ax = axes[idx]
        if ds not in results:
            continue

        # Get from first run (seed=42 typically)
        first_run = results[ds][ALGORITHMS[0]][0]
        spc = first_run.get("dataset_metadata", {}).get("samples_per_client", {})

        if not spc:
            ax.text(0.5, 0.5, "N/A", transform=ax.transAxes, ha="center")
            continue

        clients = sorted(spc.keys(), key=lambda x: int(x))
        samples = [spc[c] for c in clients]
        colors = plt.cm.Set2(np.linspace(0, 1, len(clients)))

        bars = ax.bar([f"Client {c}" for c in clients], samples, color=colors,
                     edgecolor="black", linewidth=0.5)

        # Annotate with count
        for bar, s in zip(bars, samples):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(samples)*0.02,
                   str(s), ha="center", va="bottom", fontsize=9, fontweight="bold")

        ax.set_title(DATASET_SHORT[ds], fontweight="bold")
        ax.set_ylabel("Samples" if idx == 0 else "")
        ax.grid(axis="y", alpha=0.3)

        total = sum(samples)
        partition = first_run.get("dataset_metadata", {}).get("partition_method", "unknown")
        ax.text(0.98, 0.95, f"Total: {total}\nPartition: {partition}",
               transform=ax.transAxes, ha="right", va="top", fontsize=8,
               bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5))

    fig.suptitle("Non-IID Data Distribution Across Clients", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "10_data_distribution.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  10_data_distribution.pdf")


# ======================================================================
# 11. Algorithm Ranking Heatmap
# ======================================================================

def plot_ranking_heatmap(results):
    metrics_to_rank = {
        "Accuracy": lambda runs: np.mean([r["best_metrics"]["accuracy"] for r in runs]),
        "F1": lambda runs: np.mean([r["final_metrics"]["f1"] for r in runs]),
        "Jain": lambda runs: np.mean([r["fairness"]["jain_index"] for r in runs]),
        "Speed": lambda runs: 1.0 / (1 + np.mean([r["best_round"] for r in runs])),  # inverse
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, ds in enumerate(DATASETS):
        ax = axes[idx]
        algos = [a for a in ALGORITHMS if a in results.get(ds, {})]

        rank_matrix = np.zeros((len(algos), len(metrics_to_rank)))

        for j, (metric_name, metric_fn) in enumerate(metrics_to_rank.items()):
            scores = [(metric_fn(results[ds][a]), i) for i, a in enumerate(algos)]
            sorted_scores = sorted(scores, key=lambda x: -x[0])
            for rank, (score, orig_idx) in enumerate(sorted_scores, 1):
                rank_matrix[orig_idx, j] = rank

        im = ax.imshow(rank_matrix, cmap="RdYlGn_r", aspect="auto",
                       vmin=1, vmax=len(algos))
        ax.set_yticks(range(len(algos)))
        ax.set_yticklabels(algos, fontsize=9)
        ax.set_xticks(range(len(metrics_to_rank)))
        ax.set_xticklabels(list(metrics_to_rank.keys()), fontsize=9)
        ax.set_title(DATASET_SHORT[ds], fontweight="bold")

        for i in range(len(algos)):
            for j in range(len(metrics_to_rank)):
                rank = int(rank_matrix[i, j])
                color = "white" if rank > len(algos) / 2 else "black"
                marker = " *" if rank == 1 else ""
                ax.text(j, i, f"#{rank}{marker}", ha="center", va="center",
                       fontsize=9, color=color, fontweight="bold" if rank == 1 else "normal")

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Rank (1=best)")

    fig.suptitle("Algorithm Ranking per Metric (* = best)", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "11_ranking_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  11_ranking_heatmap.pdf")


# ======================================================================
# 12. Early Stopping Analysis
# ======================================================================

def plot_early_stopping(results):
    fig, ax = plt.subplots(figsize=(12, 6))

    all_data = []
    labels = []

    for ds in DATASETS:
        for algo in ALGORITHMS:
            if algo not in results.get(ds, {}):
                continue
            runs = results[ds][algo]
            max_rounds = runs[0]["config"]["num_rounds"]
            for r in runs:
                all_data.append({
                    "ds": DATASET_SHORT[ds],
                    "algo": algo,
                    "actual": r["actual_rounds"],
                    "max": max_rounds,
                    "early": r["stopped_early"],
                    "best_round": r["best_round"],
                })

    # Group by dataset-algorithm
    groups = defaultdict(list)
    for d in all_data:
        key = f"{d['ds']}\n{d['algo']}"
        groups[key].append(d)

    keys = list(groups.keys())
    x = np.arange(len(keys))

    actual_means = [np.mean([d["actual"] for d in groups[k]]) for k in keys]
    best_means = [np.mean([d["best_round"] for d in groups[k]]) for k in keys]
    max_rounds = [groups[k][0]["max"] for k in keys]

    ax.bar(x, max_rounds, color="#E0E0E0", edgecolor="gray", linewidth=0.5, label="Max configured", width=0.6)
    ax.bar(x, actual_means, color="#90CAF9", edgecolor="black", linewidth=0.5, label="Actual rounds (ES)", width=0.6)
    ax.bar(x, best_means, color="#2196F3", edgecolor="black", linewidth=0.5, label="Round of best acc", width=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(keys, rotation=90, fontsize=7, ha="center")
    ax.set_ylabel("Rounds")
    ax.set_title("Early Stopping: Configured vs Actual vs Best Round", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # Add dataset separators
    n_algos = len(ALGORITHMS)
    for i in range(1, len(DATASETS)):
        ax.axvline(x=i * n_algos - 0.5, color="red", linestyle="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(PLOT_DIR / "12_early_stopping.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  12_early_stopping.pdf")


# ======================================================================
# 13. Runtime Comparison
# ======================================================================

def plot_runtime(results):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for idx, ds in enumerate(DATASETS):
        ax = axes[idx]
        algos = [a for a in ALGORITHMS if a in results.get(ds, {})]
        runtimes = [np.mean([r["runtime_seconds"] for r in results[ds][a]]) for a in algos]
        colors = [ALGO_COLORS[a] for a in algos]

        ax.barh(range(len(algos)), runtimes, color=colors, edgecolor="black",
                linewidth=0.5, alpha=0.85)
        ax.set_yticks(range(len(algos)))
        ax.set_yticklabels(algos, fontsize=9)
        ax.set_xlabel("Runtime (seconds)" if idx == 1 else "")
        ax.set_title(DATASET_SHORT[ds], fontweight="bold")
        ax.grid(axis="x", alpha=0.3)

        for i, v in enumerate(runtimes):
            ax.text(v + max(runtimes) * 0.02, i, f"{v:.1f}s", va="center", fontsize=9)

    fig.suptitle("Mean Runtime per Experiment", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "13_runtime.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  13_runtime.pdf")


# ======================================================================
# BONUS: Combined Convergence (Accuracy) — enhanced version
# ======================================================================

def plot_convergence_enhanced(results):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, ds in enumerate(DATASETS):
        ax = axes[idx]
        for algo in ALGORITHMS:
            if algo not in results.get(ds, {}):
                continue
            runs = results[ds][algo]
            max_rounds = max(len(r["history"]) for r in runs)
            acc_matrix = np.full((len(runs), max_rounds), np.nan)
            for i, r in enumerate(runs):
                hist_acc = [h["accuracy"] for h in r["history"]]
                acc_matrix[i, :len(hist_acc)] = hist_acc

            rounds = np.arange(1, max_rounds + 1)
            mean_acc = np.nanmean(acc_matrix, axis=0) * 100
            std_acc = np.nanstd(acc_matrix, axis=0) * 100

            color = ALGO_COLORS.get(algo, "#666666")
            ax.plot(rounds, mean_acc, color=color, linewidth=2, label=algo)
            ax.fill_between(rounds, mean_acc - std_acc, mean_acc + std_acc,
                          alpha=0.1, color=color)

        ax.set_xlabel("Communication Round")
        ax.set_ylabel("Accuracy (%)" if idx == 0 else "")
        ax.set_title(DATASET_SHORT[ds], fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(1, None)
        if idx == 2:
            ax.legend(fontsize=8, loc="lower right")

    fig.suptitle("FL Convergence — Accuracy (mean ± std over 3 seeds)", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "00_convergence_accuracy.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  00_convergence_accuracy.pdf")


# ======================================================================
# Main
# ======================================================================

def main():
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading results...")
    results = load_results()
    print(f"Datasets: {list(results.keys())}")

    print("\nGenerating plots...")
    plot_convergence_enhanced(results)   # 00
    plot_accuracy_bars(results)          # 01
    plot_fairness_bars(results)          # 02
    plot_accuracy_boxplots(results)      # 03
    plot_client_heatmaps(results)        # 04
    plot_loss_convergence(results)       # 05
    plot_f1_convergence(results)         # 06
    plot_rounds_to_converge(results)     # 07
    plot_accuracy_vs_fairness(results)   # 08
    plot_radar_charts(results)           # 09
    plot_data_distribution(results)      # 10
    plot_ranking_heatmap(results)        # 11
    plot_early_stopping(results)         # 12
    plot_runtime(results)                # 13

    print(f"\n14 plots saved to: {PLOT_DIR}")
    print("Done!")


if __name__ == "__main__":
    main()

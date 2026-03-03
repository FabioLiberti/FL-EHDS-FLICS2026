#!/usr/bin/env python3
"""
Generate publication-quality figures from Cascade 7 checkpoint data.

Figures produced:
1. fig_calibration_ece.pdf — ECE before/after temperature scaling by DP level
2. fig_conformal_setsize.pdf — Conformal prediction set size inflation under DP

All data sourced from real experimental results (checkpoint_cascade7.json).

Author: Fabio Liberti
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).parent
FIGURES_DIR = SCRIPT_DIR.parent / "figures"
DATA_DIR = Path(__file__).parent.parent.parent.parent / "fl-ehds-framework" / "benchmarks" / "paper_results_tabular"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Load data
with open(DATA_DIR / "checkpoint_cascade7.json") as f:
    cas7 = json.load(f)

# Publication style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
})


# ======================================================================
# Figure 1: ECE Before/After Temperature Scaling
# ======================================================================

def plot_calibration_ece():
    """Grouped bar chart: ECE before vs after temperature scaling by DP level."""

    # Collect Block M data
    m_data = {}
    for key, val in cas7["results"].items():
        if not isinstance(val, dict) or val.get("block") != "M":
            continue
        ds = val.get("dataset")
        algo = val.get("algorithm")
        dp = val.get("dp_level")
        k = (ds, algo, dp)
        if k not in m_data:
            m_data[k] = {"ece_pre": [], "ece_post": []}
        m_data[k]["ece_pre"].append(val.get("ece_before_calibration", 0))
        m_data[k]["ece_post"].append(val.get("ece_after_temperature_scaling", 0))

    # Focus on Cardiovascular and PTB_XL (Breast Cancer is noisy)
    datasets = ["Cardiovascular", "PTB_XL"]
    dp_levels = ["noDP", "eps10", "eps1"]
    dp_labels = ["No DP", "$\\varepsilon$=10", "$\\varepsilon$=1"]
    algorithms = ["FedAvg", "Ditto", "HPFL"]

    fig, axes = plt.subplots(1, 2, figsize=(7, 3.2), sharey=True)

    for idx, ds in enumerate(datasets):
        ax = axes[idx]
        x = np.arange(len(dp_levels))
        width = 0.12
        offsets = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]

        for a_idx, algo in enumerate(algorithms):
            ece_pre_means = []
            ece_post_means = []
            for dp in dp_levels:
                d = m_data.get((ds, algo, dp), {"ece_pre": [0], "ece_post": [0]})
                ece_pre_means.append(np.mean(d["ece_pre"]))
                ece_post_means.append(np.mean(d["ece_post"]))

            colors_pre = ["#ef5350", "#e53935", "#c62828"]
            colors_post = ["#42a5f5", "#1e88e5", "#1565c0"]

            bars_pre = ax.bar(x + offsets[a_idx * 2] * width, ece_pre_means, width,
                              color=colors_pre[a_idx], alpha=0.7,
                              label="{} (before)".format(algo) if idx == 0 else "")
            bars_post = ax.bar(x + offsets[a_idx * 2 + 1] * width, ece_post_means, width,
                               color=colors_post[a_idx], alpha=0.9,
                               label="{} (after TS)".format(algo) if idx == 0 else "")

        ax.set_xticks(x)
        ax.set_xticklabels(dp_labels)
        ax.set_title(ds.replace("_", "-"))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if idx == 0:
            ax.set_ylabel("ECE")

    axes[0].legend(fontsize=6, loc="upper left", ncol=1)
    fig.suptitle("Model Calibration: ECE Before and After Temperature Scaling", fontsize=10, y=1.02)
    fig.tight_layout()
    fig.savefig(str(FIGURES_DIR / "fig_calibration_ece.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("Saved fig_calibration_ece.pdf")


# ======================================================================
# Figure 2: Conformal Prediction Set Size
# ======================================================================

def plot_conformal_setsize():
    """Bar chart: average prediction set size across DP levels."""

    # Collect Block N data
    n_data = {}
    for key, val in cas7["results"].items():
        if not isinstance(val, dict) or val.get("block") != "N":
            continue
        ds = val.get("dataset")
        cond = val.get("condition")
        dp = val.get("dp_level")
        k = (ds, cond, dp)
        if k not in n_data:
            n_data[k] = {"set_size": [], "coverage": []}
        gc = val.get("global_conformal", {})
        n_data[k]["set_size"].append(gc.get("avg_set_size", 0))
        n_data[k]["coverage"].append(gc.get("coverage", 0))

    datasets = ["Cardiovascular", "PTB_XL", "Breast_Cancer"]
    ds_labels = ["CV", "PTB-XL", "BC"]
    conditions = ["IID", "NonIID"]
    dp_levels = ["noDP", "eps10", "eps1"]

    fig, ax = plt.subplots(figsize=(6, 3.5))

    x_pos = 0
    x_ticks = []
    x_labels = []
    colors_map = {"noDP": "#66bb6a", "eps10": "#42a5f5", "eps1": "#ef5350"}

    for ds_idx, ds in enumerate(datasets):
        for cond in conditions:
            group_start = x_pos
            for dp in dp_levels:
                d = n_data.get((ds, cond, dp), {"set_size": [0]})
                mean_ss = np.mean(d["set_size"])
                bar = ax.bar(x_pos, mean_ss, color=colors_map[dp], edgecolor="white",
                             linewidth=0.5, width=0.7)
                # Add value label
                if mean_ss > 0.1:
                    ax.text(x_pos, mean_ss + 0.08, "{:.1f}".format(mean_ss),
                            ha="center", va="bottom", fontsize=6, fontweight="bold")
                x_pos += 1
            # Group label
            mid = group_start + 1
            x_ticks.append(mid)
            x_labels.append("{}\n{}".format(ds_labels[ds_idx], cond))
            x_pos += 0.5  # gap between groups

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, fontsize=7)
    ax.set_ylabel("Average Prediction Set Size")
    ax.set_title("Conformal Prediction Set Size Inflation under DP")

    # Reference lines
    ax.axhline(y=1.0, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.text(x_pos - 1, 1.05, "ideal=1.0", fontsize=6, color="gray", ha="right")

    # Vacuous threshold for binary
    ax.axhline(y=2.0, color="red", linestyle="--", linewidth=0.8, alpha=0.4)
    ax.text(x_pos - 1, 2.1, "vacuous (binary)", fontsize=6, color="red", ha="right")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#66bb6a", label="No DP"),
        Patch(facecolor="#42a5f5", label="$\\varepsilon$=10"),
        Patch(facecolor="#ef5350", label="$\\varepsilon$=1"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=7)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(0, 5.8)
    fig.tight_layout()
    fig.savefig(str(FIGURES_DIR / "fig_conformal_setsize.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("Saved fig_conformal_setsize.pdf")


# ======================================================================
# Main
# ======================================================================

if __name__ == "__main__":
    print("Generating Cascade 7 figures from real experimental data...")
    print("Data source: {}".format(DATA_DIR))
    print("Output: {}".format(FIGURES_DIR))
    print()
    plot_calibration_ece()
    plot_conformal_setsize()
    print("\nAll 2 figures generated successfully.")

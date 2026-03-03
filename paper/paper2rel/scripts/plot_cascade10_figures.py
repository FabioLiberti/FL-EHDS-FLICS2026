#!/usr/bin/env python3
"""
Generate publication-quality figures from Cascade 10 checkpoint data.

Figures produced:
1. fig_collapse_rate_heatmap.pdf — Collapse rate by condition (DP vs no-DP)
2. fig_mitigation_comparison.pdf — Weighted CE vs Focal vs Standard F1 bars
3. fig_ditto_epochs_sweep.pdf — Local epochs effect on Ditto minority-class F1
4. fig_threshold_rescue.pdf — Threshold rescue: original vs tuned F1

All data sourced from real experimental results (checkpoint_cascade10.json
and checkpoint_cascade9.json). No synthetic data.

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
with open(DATA_DIR / "checkpoint_cascade10.json") as f:
    cas10 = json.load(f)

cas9 = None
cas9_path = DATA_DIR / "checkpoint_cascade9.json"
if cas9_path.exists():
    with open(cas9_path) as f:
        cas9 = json.load(f)

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
# Figure 1: Collapse Rate Heatmap
# ======================================================================

def plot_collapse_rate():
    """Bar chart showing collapse rate (% F1=0.0) by condition."""

    # Cascade 9 AA data (Stroke + Cirrhosis only)
    aa_conditions = {"IID+noDP": [], "NonIID+noDP": [], "NonIID+eps10": []}
    if cas9:
        for key, val in cas9["results"].items():
            if not isinstance(val, dict) or val.get("block") != "AA_imbalance":
                continue
            ds = val.get("dataset")
            if ds not in ("Stroke", "Cirrhosis"):
                continue
            iid = "IID" if val.get("is_iid") else "NonIID"
            dp = "eps10" if val.get("dp_epsilon") else "noDP"
            cond = "{}+{}".format(iid, dp)
            if cond in aa_conditions:
                aa_conditions[cond].append(val.get("f1_score", 0))

    # Cascade 10 AC data
    ac_conditions = {"IID+eps10": [], "IID+eps1": [], "NonIID+eps1": []}
    for key, val in cas10["results"].items():
        if not isinstance(val, dict) or val.get("block") != "AC_condition_matrix":
            continue
        ds = val.get("dataset")
        if ds not in ("Stroke", "Cirrhosis"):
            continue
        iid = "IID" if val.get("is_iid") else "NonIID"
        eps = val.get("dp_epsilon")
        if eps == 10.0:
            dp = "eps10"
        elif eps == 1.0:
            dp = "eps1"
        else:
            dp = "noDP"
        cond = "{}+{}".format(iid, dp)
        if cond in ac_conditions:
            ac_conditions[cond].append(val.get("f1_score", 0))

    # Combine all conditions
    all_conditions = {}
    all_conditions.update(aa_conditions)
    all_conditions.update(ac_conditions)

    # Compute collapse rates
    labels = ["IID\nno DP", "NonIID\nno DP", "NonIID\n$\\varepsilon$=10",
              "IID\n$\\varepsilon$=10", "IID\n$\\varepsilon$=1", "NonIID\n$\\varepsilon$=1"]
    keys = ["IID+noDP", "NonIID+noDP", "NonIID+eps10", "IID+eps10", "IID+eps1", "NonIID+eps1"]
    rates = []
    counts = []
    for k in keys:
        vals = all_conditions.get(k, [])
        n = len(vals)
        collapsed = sum(1 for v in vals if v == 0.0)
        rate = (collapsed / n * 100) if n > 0 else 0
        rates.append(rate)
        counts.append("{}/{}".format(collapsed, n))

    # Colors: red gradient for no-DP, blue gradient for DP
    colors = ["#d32f2f", "#e53935", "#66bb6a", "#42a5f5", "#1e88e5", "#1565c0"]

    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    bars = ax.bar(range(len(labels)), rates, color=colors, edgecolor="white", linewidth=0.5)

    # Add count annotations
    for i, (bar, count) in enumerate(zip(bars, counts)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                count, ha="center", va="bottom", fontsize=7, fontweight="bold")

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Collapse Rate (% F1=0.0)")
    ax.set_title("Majority-Class Collapse: DP Noise as Implicit Regularizer")
    ax.set_ylim(0, 100)

    # Divider line between no-DP and DP conditions
    ax.axvline(x=2.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.text(1.0, 92, "No DP", ha="center", fontsize=8, color="#d32f2f", fontweight="bold")
    ax.text(4.0, 92, "With DP", ha="center", fontsize=8, color="#1565c0", fontweight="bold")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(str(FIGURES_DIR / "fig_collapse_rate_heatmap.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("Saved fig_collapse_rate_heatmap.pdf")


# ======================================================================
# Figure 2: Mitigation Strategy Comparison
# ======================================================================

def plot_mitigation_comparison():
    """Grouped bar chart: Standard vs Weighted CE vs Focal F1 by dataset."""

    datasets = ["Stroke", "Cirrhosis", "CDC_Diabetes"]
    display_names = ["Stroke\n(4.9% pos)", "Cirrhosis\n(37% pos)", "CDC Diabetes\n(14% pos)"]

    # Get best config F1 for each (dataset x loss_type)
    # Standard baseline from cascade9 AA (best config)
    standard_f1 = {}
    if cas9:
        for key, val in cas9["results"].items():
            if not isinstance(val, dict) or val.get("block") != "AA_imbalance":
                continue
            ds = val.get("dataset")
            if ds not in datasets:
                continue
            f1 = val.get("f1_score", 0)
            if ds not in standard_f1:
                standard_f1[ds] = []
            standard_f1[ds].append(f1)

    # AD results by loss type
    ad_results = {ds: {"weighted_ce": [], "focal": []} for ds in datasets}
    for key, val in cas10["results"].items():
        if not isinstance(val, dict) or val.get("block") != "AD_mitigation":
            continue
        ds = val.get("dataset")
        lt = val.get("loss_type")
        if ds in ad_results and lt in ad_results[ds]:
            ad_results[ds][lt].append(val.get("f1_score", 0))

    # Compute means
    means_standard = []
    means_weighted = []
    means_focal = []
    stds_standard = []
    stds_weighted = []
    stds_focal = []

    for ds in datasets:
        s_vals = standard_f1.get(ds, [0])
        w_vals = ad_results[ds]["weighted_ce"]
        f_vals = ad_results[ds]["focal"]
        means_standard.append(np.mean(s_vals))
        means_weighted.append(np.mean(w_vals))
        means_focal.append(np.mean(f_vals))
        stds_standard.append(np.std(s_vals))
        stds_weighted.append(np.std(w_vals))
        stds_focal.append(np.std(f_vals))

    x = np.arange(len(datasets))
    width = 0.25

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    bars1 = ax.bar(x - width, means_standard, width, yerr=stds_standard,
                   label="Standard CE", color="#ef5350", capsize=3, edgecolor="white")
    bars2 = ax.bar(x, means_weighted, width, yerr=stds_weighted,
                   label="Weighted CE", color="#42a5f5", capsize=3, edgecolor="white")
    bars3 = ax.bar(x + width, means_focal, width, yerr=stds_focal,
                   label="Focal ($\\gamma$=2)", color="#66bb6a", capsize=3, edgecolor="white")

    ax.set_ylabel("Mean F1 Score")
    ax.set_title("Loss Function Mitigation Strategies for Class Imbalance")
    ax.set_xticks(x)
    ax.set_xticklabels(display_names)
    ax.legend(loc="upper left", framealpha=0.9)
    ax.set_ylim(0, 0.85)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            if h > 0.01:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.02,
                        "{:.2f}".format(h), ha="center", va="bottom", fontsize=7)

    fig.tight_layout()
    fig.savefig(str(FIGURES_DIR / "fig_mitigation_comparison.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("Saved fig_mitigation_comparison.pdf")


# ======================================================================
# Figure 3: Ditto Local Epochs Sweep
# ======================================================================

def plot_ditto_epochs():
    """Line chart: F1 vs local epochs for Ditto on Stroke and Cirrhosis."""

    # Baseline (ep=3) from cascade9 AA
    baseline = {}
    if cas9:
        for key, val in cas9["results"].items():
            if not isinstance(val, dict) or val.get("block") != "AA_imbalance":
                continue
            if val.get("algorithm") != "Ditto":
                continue
            ds = val.get("dataset")
            if ds not in ("Stroke", "Cirrhosis"):
                continue
            if not val.get("is_iid") and val.get("dp_epsilon") is None:
                cond = "noDP"
            elif not val.get("is_iid") and val.get("dp_epsilon") == 10.0:
                cond = "eps10"
            else:
                continue
            k = (ds, cond)
            baseline.setdefault(k, []).append(val.get("f1_score", 0))

    # AE data (ep=5, ep=10)
    ae_data = {}
    for key, val in cas10["results"].items():
        if not isinstance(val, dict) or val.get("block") != "AE_epochs_sweep":
            continue
        ds = val.get("dataset")
        ep = val.get("local_epochs")
        dp = val.get("dp_epsilon")
        cond = "eps10" if dp else "noDP"
        k = (ds, cond, ep)
        ae_data.setdefault(k, []).append(val.get("f1_score", 0))

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.0), sharey=True)

    for idx, ds in enumerate(["Cirrhosis", "Stroke"]):
        ax = axes[idx]
        for cond, color, marker, ls in [("noDP", "#1e88e5", "o", "-"), ("eps10", "#e53935", "s", "--")]:
            epochs = [3, 5, 10]
            means = []
            stds = []
            for ep in epochs:
                if ep == 3:
                    vals = baseline.get((ds, cond), [0])
                else:
                    vals = ae_data.get((ds, cond, ep), [0])
                means.append(np.mean(vals))
                stds.append(np.std(vals))

            label = "NonIID+noDP" if cond == "noDP" else "NonIID+$\\varepsilon$=10"
            ax.errorbar(epochs, means, yerr=stds, marker=marker, linestyle=ls,
                        color=color, label=label, capsize=4, linewidth=1.5,
                        markersize=5)

        ax.set_xlabel("Local Epochs")
        ax.set_title(ds)
        ax.set_xticks([3, 5, 10])
        ax.set_xlim(2, 11)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if idx == 0:
            ax.set_ylabel("Ditto Mean F1 Score")
            ax.legend(fontsize=7, loc="upper left")

    fig.suptitle("Effect of Local Epochs on Ditto Minority-Class Rescue", fontsize=10, y=1.02)
    fig.tight_layout()
    fig.savefig(str(FIGURES_DIR / "fig_ditto_epochs_sweep.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("Saved fig_ditto_epochs_sweep.pdf")


# ======================================================================
# Figure 4: Threshold Rescue
# ======================================================================

def plot_threshold_rescue():
    """Paired bar chart: original F1 (0.0) vs threshold-tuned F1 for rescued models."""

    th_results = []
    for key, val in cas10["results"].items():
        if not isinstance(val, dict) or val.get("block") != "TH_threshold_rescue":
            continue
        th_results.append({
            "dataset": val.get("dataset"),
            "algorithm": val.get("algorithm"),
            "is_iid": val.get("is_iid"),
            "dp_epsilon": val.get("dp_epsilon"),
            "original_f1": val.get("original_f1", 0),
            "threshold_tuned_f1": val.get("threshold_tuned_f1", 0),
            "optimal_threshold": val.get("optimal_threshold", 0.5),
        })

    # Sort: Cirrhosis first (better rescue), then Stroke
    th_results.sort(key=lambda x: (-x["threshold_tuned_f1"]))

    labels = []
    tt_f1 = []
    colors = []
    for r in th_results:
        iid = "IID" if r["is_iid"] else "NI"
        dp = "e{}".format(int(r["dp_epsilon"])) if r["dp_epsilon"] else "nD"
        lab = "{}\n{} {}\nt={}".format(
            r["dataset"][:3], r["algorithm"][:3], dp, r["optimal_threshold"])
        labels.append(lab)
        tt_f1.append(r["threshold_tuned_f1"])
        colors.append("#42a5f5" if r["dataset"] == "Cirrhosis" else "#ef5350")

    fig, ax = plt.subplots(figsize=(7, 3.5))
    x = range(len(labels))
    bars = ax.bar(x, tt_f1, color=colors, edgecolor="white", linewidth=0.5)

    # Reference line at F1=0 (original)
    ax.axhline(y=0, color="black", linewidth=0.8)

    # Meaningful threshold line
    ax.axhline(y=0.20, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
    ax.text(len(labels) - 0.5, 0.22, "F1=0.20 (minimum clinical)", fontsize=6,
            ha="right", color="gray")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=5.5)
    ax.set_ylabel("Threshold-Tuned F1")
    ax.set_title("Post-Hoc Threshold Rescue of F1=0.0 Collapsed Models (Block TH)")
    ax.set_ylim(-0.02, 1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor="#42a5f5", label="Cirrhosis (7 models)"),
                       Patch(facecolor="#ef5350", label="Stroke (13 models)")]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=7)

    fig.tight_layout()
    fig.savefig(str(FIGURES_DIR / "fig_threshold_rescue.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("Saved fig_threshold_rescue.pdf")


# ======================================================================
# Main
# ======================================================================

if __name__ == "__main__":
    print("Generating Cascade 10 figures from real experimental data...")
    print("Data source: {}".format(DATA_DIR))
    print("Output: {}".format(FIGURES_DIR))
    print()
    plot_collapse_rate()
    plot_mitigation_comparison()
    plot_ditto_epochs()
    plot_threshold_rescue()
    print("\nAll 4 figures generated successfully.")

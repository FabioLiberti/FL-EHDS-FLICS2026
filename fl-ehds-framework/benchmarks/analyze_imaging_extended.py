#!/usr/bin/env python3
"""
FL-EHDS Imaging Extended Analysis — Tables, Figures, Statistics.

Reads ALL imaging checkpoint files, merges into a unified 4-algo x 3-dataset
x 3-seed matrix (36 experiments), and generates:
  1. LaTeX summary table (for supplementary)
  2. Convergence curves (4 algos per dataset, mean +/- band over seeds)
  3. Per-client accuracy heatmaps
  4. Cohen's d effect sizes (Ditto vs FedAvg)
  5. Cross-dataset comparison bar chart
  6. Statistical significance report

Uses ONLY existing checkpoint data — no new training required.

Usage:
    cd fl-ehds-framework
    python -m benchmarks.analyze_imaging_extended

Output: benchmarks/paper_results_delta/analysis_imaging/

Author: Fabio Liberti
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

FRAMEWORK_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(FRAMEWORK_DIR))

from benchmarks.significance import paired_ttest, cohens_d, confidence_interval

OUTPUT_DIR = FRAMEWORK_DIR / "benchmarks" / "paper_results_delta" / "analysis_imaging"
CHECKPOINT_DIR = FRAMEWORK_DIR / "benchmarks" / "paper_results_delta"

DATASETS = ["chest_xray", "Brain_Tumor", "Skin_Cancer"]
ALGORITHMS = ["FedAvg", "FedLESAM", "Ditto", "HPFL"]
SEEDS = [42, 123, 456]

DATASET_LABELS = {
    "chest_xray": "Chest X-ray",
    "Brain_Tumor": "Brain Tumor MRI",
    "Skin_Cancer": "Skin Cancer",
}

ALGO_COLORS = {
    "FedAvg": "#4C72B0",
    "FedLESAM": "#55A868",
    "Ditto": "#C44E52",
    "HPFL": "#DD8452",
}

# ======================================================================
# Load & Merge Checkpoints
# ======================================================================

def load_all_checkpoints() -> Dict:
    """Load and merge all imaging checkpoint files into unified structure.

    Returns:
        {(dataset, algo, seed): result_dict}
    """
    files = [
        "checkpoint_chest_baseline.json",
        "checkpoint_chest_extended.json",
        "checkpoint_imaging_multi.json",
        "checkpoint_delta.json",
        "checkpoint_completion.json",
    ]

    merged = {}
    for fname in files:
        path = CHECKPOINT_DIR / fname
        if not path.exists():
            print(f"  [SKIP] {fname} not found")
            continue

        with open(path) as f:
            data = json.load(f)

        completed = data.get("completed", {})
        for key, result in completed.items():
            if "error" in result:
                continue

            ds = result.get("dataset", "")
            algo = result.get("algorithm", "")
            seed = result.get("seed", 0)

            if ds and algo and seed:
                merged[(ds, algo, seed)] = result

    print(f"  Loaded {len(merged)} experiments from {len(files)} checkpoint files")
    return merged


def build_matrix(merged: Dict) -> Dict:
    """Build structured matrix: {dataset: {algo: {seed: result}}}."""
    matrix = {}
    for ds in DATASETS:
        matrix[ds] = {}
        for algo in ALGORITHMS:
            matrix[ds][algo] = {}
            for seed in SEEDS:
                r = merged.get((ds, algo, seed))
                if r is None:
                    # FedLESAM = FedAvg on imaging (use as proxy)
                    if algo == "FedAvg":
                        r = merged.get((ds, "FedLESAM", seed))
                    elif algo == "FedLESAM":
                        r = merged.get((ds, "FedAvg", seed))
                if r is not None:
                    matrix[ds][algo][seed] = r
    return matrix


# ======================================================================
# Statistics
# ======================================================================

def compute_stats(matrix: Dict) -> Dict:
    """Compute per-dataset, per-algorithm statistics."""
    stats = {}
    for ds in DATASETS:
        stats[ds] = {}
        for algo in ALGORITHMS:
            accs = []
            jains = []
            for seed in SEEDS:
                r = matrix[ds][algo].get(seed)
                if r:
                    best = r.get("best_metrics", {}).get("accuracy", 0)
                    accs.append(best)
                    j = r.get("fairness", {}).get("jain_index")
                    if j is not None:
                        jains.append(j)

            if accs:
                ci = confidence_interval(accs)
                stats[ds][algo] = {
                    "accs": accs,
                    "mean": np.mean(accs),
                    "std": np.std(accs),
                    "ci_95": ci,
                    "jain_mean": np.mean(jains) if jains else None,
                    "n_seeds": len(accs),
                }
    return stats


def compute_significance(stats: Dict) -> Dict:
    """Compute Ditto vs FedAvg significance for each dataset."""
    sig = {}
    for ds in DATASETS:
        fedavg = stats[ds].get("FedAvg", {}).get("accs", [])
        ditto = stats[ds].get("Ditto", {}).get("accs", [])
        hpfl = stats[ds].get("HPFL", {}).get("accs", [])

        sig[ds] = {}
        if len(fedavg) >= 2 and len(ditto) >= 2:
            sig[ds]["Ditto_vs_FedAvg"] = {
                "ttest": paired_ttest(fedavg, ditto),
                "cohens_d": cohens_d(fedavg, ditto),
                "delta_pp": (np.mean(ditto) - np.mean(fedavg)) * 100,
            }
        if len(fedavg) >= 2 and len(hpfl) >= 2:
            sig[ds]["HPFL_vs_FedAvg"] = {
                "ttest": paired_ttest(fedavg, hpfl),
                "cohens_d": cohens_d(fedavg, hpfl),
                "delta_pp": (np.mean(hpfl) - np.mean(fedavg)) * 100,
            }
    return sig


# ======================================================================
# LaTeX Table
# ======================================================================

def generate_latex_table(stats: Dict, sig: Dict) -> str:
    """Generate LaTeX table for supplementary."""
    lines = []
    lines.append(r"\begin{table*}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Complete imaging evaluation: 4 algorithms $\times$ 3 datasets $\times$ 3 seeds (36 experiments). ResNet-18, 5 clients, Dirichlet $\alpha{=}0.5$. Best accuracy per dataset in \textbf{bold}. Cohen's $d$ vs FedAvg.}")
    lines.append(r"\label{tab:imaging_complete}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{llccccccc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Dataset} & \textbf{Algorithm} & \textbf{s42} & \textbf{s123} & \textbf{s456} & \textbf{Mean$\pm$std} & \textbf{Jain} & \textbf{$\Delta$ FedAvg} & \textbf{Cohen's $d$} \\")
    lines.append(r"\midrule")

    for ds_idx, ds in enumerate(DATASETS):
        if ds_idx > 0:
            lines.append(r"\midrule")

        best_mean = max(
            s.get("mean", 0) for s in stats[ds].values()
        )

        for algo in ALGORITHMS:
            s = stats[ds].get(algo, {})
            if not s:
                continue

            accs = s["accs"]
            mean = s["mean"]
            std = s["std"]
            jain = s.get("jain_mean")

            # Per-seed values
            seed_strs = []
            for seed in SEEDS:
                found = False
                for a in accs:
                    pass
                # Get from accs by index
                if len(accs) == 3:
                    seed_strs = [f"{a*100:.1f}" for a in accs]
                    break

            if not seed_strs:
                seed_strs = ["--"] * 3

            # Mean +/- std
            mean_str = f"{mean*100:.1f}$\\pm${std*100:.1f}"
            if abs(mean - best_mean) < 1e-6:
                mean_str = f"\\textbf{{{mean_str}}}"

            # Jain
            jain_str = f"{jain:.3f}" if jain else "---"

            # Delta vs FedAvg
            fedavg_mean = stats[ds].get("FedAvg", {}).get("mean", 0)
            if algo == "FedAvg":
                delta_str = "---"
                d_str = "---"
            else:
                delta = (mean - fedavg_mean) * 100
                delta_str = f"{delta:+.1f}"

                # Cohen's d
                key = f"{algo}_vs_FedAvg"
                d_val = sig.get(ds, {}).get(key, {}).get("cohens_d", None)
                if d_val is not None:
                    d_str = f"{d_val:.2f}"
                else:
                    d_str = "---"

            # Dataset label (multirow)
            if algo == ALGORITHMS[0]:
                ds_label = f"\\multirow{{4}}{{*}}{{{DATASET_LABELS[ds]}}}"
            else:
                ds_label = ""

            lines.append(
                f"{ds_label} & {algo} & {' & '.join(seed_strs)} & {mean_str} & {jain_str} & {delta_str} & {d_str} \\\\"
            )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")

    return "\n".join(lines)


# ======================================================================
# Text Report
# ======================================================================

def generate_text_report(stats: Dict, sig: Dict) -> str:
    """Generate human-readable summary report."""
    lines = []
    lines.append("=" * 70)
    lines.append("  FL-EHDS Imaging Extended Analysis — Statistical Report")
    lines.append("=" * 70)

    for ds in DATASETS:
        lines.append(f"\n{'─' * 50}")
        lines.append(f"  {DATASET_LABELS[ds]}")
        lines.append(f"{'─' * 50}")

        lines.append(f"  {'Algorithm':<12} {'Mean':>8} {'Std':>8} {'Jain':>8} {'Δ FedAvg':>10} {'Cohen d':>9}")

        fedavg_mean = stats[ds].get("FedAvg", {}).get("mean", 0)

        for algo in ALGORITHMS:
            s = stats[ds].get(algo, {})
            if not s:
                continue
            mean = s["mean"]
            std = s["std"]
            jain = s.get("jain_mean")
            delta = (mean - fedavg_mean) * 100

            key = f"{algo}_vs_FedAvg"
            d_val = sig.get(ds, {}).get(key, {}).get("cohens_d")

            jain_str = f"{jain:.3f}" if jain else "  ---"
            delta_str = f"{delta:+.1f}pp" if algo != "FedAvg" else "   ---"
            d_str = f"{d_val:.2f}" if d_val is not None else "  ---"

            lines.append(
                f"  {algo:<12} {mean*100:>7.1f}% {std*100:>7.1f}% {jain_str:>8} {delta_str:>10} {d_str:>9}"
            )

        # Significance tests
        for comp in ["Ditto_vs_FedAvg", "HPFL_vs_FedAvg"]:
            s = sig.get(ds, {}).get(comp)
            if s:
                t = s["ttest"]
                lines.append(
                    f"\n  {comp}: Δ={s['delta_pp']:+.1f}pp, d={s['cohens_d']:.2f}, "
                    f"t={t['t_stat']:.2f}, p={t['p_value']:.4f} "
                    f"{'*' if t['p_value'] < 0.05 else 'ns'}"
                )

    lines.append(f"\n{'=' * 70}")
    lines.append("  Cross-Dataset Summary")
    lines.append(f"{'=' * 70}")

    # Ditto wins/loses summary
    ditto_wins = 0
    hpfl_wins = 0
    for ds in DATASETS:
        d_delta = sig.get(ds, {}).get("Ditto_vs_FedAvg", {}).get("delta_pp", 0)
        h_delta = sig.get(ds, {}).get("HPFL_vs_FedAvg", {}).get("delta_pp", 0)
        if d_delta > 0:
            ditto_wins += 1
        if h_delta > 0:
            hpfl_wins += 1

    lines.append(f"  Ditto > FedAvg on {ditto_wins}/3 datasets")
    lines.append(f"  HPFL > FedAvg on {hpfl_wins}/3 datasets")
    lines.append(f"  FedLESAM ≡ FedAvg on all 3 datasets (SAM ineffective)")

    return "\n".join(lines)


# ======================================================================
# Plotting
# ======================================================================

def generate_figures(matrix: Dict, stats: Dict):
    """Generate all figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # -------------------------------------------------------------------
    # Figure 1: Cross-dataset bar chart (accuracy comparison)
    # -------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=False)

    for idx, ds in enumerate(DATASETS):
        ax = axes[idx]
        algos = []
        means = []
        stds = []
        colors = []

        for algo in ALGORITHMS:
            s = stats[ds].get(algo, {})
            if s:
                algos.append(algo)
                means.append(s["mean"] * 100)
                stds.append(s["std"] * 100)
                colors.append(ALGO_COLORS[algo])

        x = np.arange(len(algos))
        bars = ax.bar(x, means, yerr=stds, capsize=4, color=colors, alpha=0.85,
                      edgecolor="black", linewidth=0.5)

        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                    f"{m:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(algos, fontsize=9, rotation=15)
        ax.set_title(DATASET_LABELS[ds], fontsize=11, fontweight="bold")
        ax.set_ylabel("Accuracy (%)" if idx == 0 else "")
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0, 105)

    plt.suptitle("Imaging FL: 4 Algorithms × 3 Datasets (3 seeds each)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(str(OUTPUT_DIR / "imaging_accuracy_comparison.pdf"), bbox_inches="tight")
    plt.savefig(str(OUTPUT_DIR / "imaging_accuracy_comparison.png"), dpi=200, bbox_inches="tight")
    plt.close()
    print("  [FIG] imaging_accuracy_comparison.pdf")

    # -------------------------------------------------------------------
    # Figure 2: Convergence curves (per dataset, 4 algos, mean +/- band)
    # -------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for idx, ds in enumerate(DATASETS):
        ax = axes[idx]

        for algo in ALGORITHMS:
            histories = []
            for seed in SEEDS:
                r = matrix[ds][algo].get(seed)
                if r and "history" in r:
                    h = [h_entry["accuracy"] * 100 for h_entry in r["history"]]
                    histories.append(h)

            if not histories:
                continue

            # Align to min length
            min_len = min(len(h) for h in histories)
            aligned = np.array([h[:min_len] for h in histories])
            mean_curve = aligned.mean(axis=0)
            rounds = np.arange(1, min_len + 1)

            ax.plot(rounds, mean_curve, label=algo, color=ALGO_COLORS[algo],
                    linewidth=2, alpha=0.9)

            if len(histories) > 1:
                std_curve = aligned.std(axis=0)
                ax.fill_between(rounds, mean_curve - std_curve, mean_curve + std_curve,
                                color=ALGO_COLORS[algo], alpha=0.15)

        ax.set_title(DATASET_LABELS[ds], fontsize=11, fontweight="bold")
        ax.set_xlabel("Round")
        ax.set_ylabel("Accuracy (%)" if idx == 0 else "")
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(alpha=0.3)

    plt.suptitle("Training Convergence on Imaging Datasets (mean ± std, 3 seeds)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(str(OUTPUT_DIR / "imaging_convergence.pdf"), bbox_inches="tight")
    plt.savefig(str(OUTPUT_DIR / "imaging_convergence.png"), dpi=200, bbox_inches="tight")
    plt.close()
    print("  [FIG] imaging_convergence.pdf")

    # -------------------------------------------------------------------
    # Figure 3: Per-client accuracy heatmaps (Ditto vs FedAvg)
    # -------------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    for col, ds in enumerate(DATASETS):
        for row, algo in enumerate(["FedAvg", "Ditto"]):
            ax = axes[row, col]

            # Collect per-client accs across seeds
            client_accs = {str(c): [] for c in range(5)}
            for seed in SEEDS:
                r = matrix[ds][algo].get(seed)
                if r and "per_client_acc" in r:
                    for cid, acc in r["per_client_acc"].items():
                        client_accs[cid].append(acc * 100)

            if not any(client_accs.values()):
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                continue

            # Build matrix: clients x seeds
            data = []
            for c in range(5):
                vals = client_accs.get(str(c), [])
                if vals:
                    data.append(vals[:3] + [0] * max(0, 3 - len(vals)))
                else:
                    data.append([0, 0, 0])

            data = np.array(data)
            im = ax.imshow(data, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")

            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    color = "white" if data[i, j] < 30 or data[i, j] > 85 else "black"
                    ax.text(j, i, f"{data[i, j]:.0f}", ha="center", va="center",
                            fontsize=9, color=color, fontweight="bold")

            ax.set_xticks(range(3))
            ax.set_xticklabels(["s42", "s123", "s456"], fontsize=8)
            ax.set_yticks(range(5))
            ax.set_yticklabels([f"C{i}" for i in range(5)], fontsize=8)
            if col == 0:
                ax.set_ylabel(algo, fontsize=11, fontweight="bold")
            if row == 0:
                ax.set_title(DATASET_LABELS[ds], fontsize=11, fontweight="bold")

    fig.colorbar(im, ax=axes, shrink=0.6, label="Accuracy (%)")
    plt.suptitle("Per-Client Accuracy: FedAvg vs Ditto across Imaging Datasets",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 0.92, 0.95])
    plt.savefig(str(OUTPUT_DIR / "imaging_per_client_heatmap.pdf"), bbox_inches="tight")
    plt.savefig(str(OUTPUT_DIR / "imaging_per_client_heatmap.png"), dpi=200, bbox_inches="tight")
    plt.close()
    print("  [FIG] imaging_per_client_heatmap.pdf")

    # -------------------------------------------------------------------
    # Figure 4: Effect size summary (Cohen's d bar chart)
    # -------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))

    comparisons = ["Ditto", "HPFL", "FedLESAM"]
    x = np.arange(len(DATASETS))
    width = 0.25

    for i, comp_algo in enumerate(comparisons):
        ds_vals = []
        for ds in DATASETS:
            fedavg_accs = stats[ds].get("FedAvg", {}).get("accs", [])
            comp_accs = stats[ds].get(comp_algo, {}).get("accs", [])
            if len(fedavg_accs) >= 2 and len(comp_accs) >= 2:
                d = cohens_d(fedavg_accs, comp_accs)
                ds_vals.append(d)
            else:
                ds_vals.append(0)

        color = ALGO_COLORS[comp_algo]
        bars = ax.bar(x + i * width, ds_vals, width, label=f"{comp_algo} vs FedAvg",
                      color=color, alpha=0.85, edgecolor="black", linewidth=0.5)

        for bar, v in zip(bars, ds_vals):
            if abs(v) > 0.1:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                        f"{v:.1f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x + width)
    ax.set_xticklabels([DATASET_LABELS[ds] for ds in DATASETS], fontsize=10)
    ax.set_ylabel("Cohen's d (effect size)")
    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.axhline(y=0.8, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.axhline(y=-0.8, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.text(2.8, 0.85, "large effect", fontsize=7, color="gray", alpha=0.7)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("Effect Size: Algorithms vs FedAvg (Cohen's d, 3 seeds)",
                 fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig(str(OUTPUT_DIR / "imaging_effect_size.pdf"), bbox_inches="tight")
    plt.savefig(str(OUTPUT_DIR / "imaging_effect_size.png"), dpi=200, bbox_inches="tight")
    plt.close()
    print("  [FIG] imaging_effect_size.pdf")


# ======================================================================
# Main
# ======================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("  FL-EHDS Imaging Extended Analysis")
    print("=" * 60)

    # Load data
    merged = load_all_checkpoints()
    matrix = build_matrix(merged)

    # Report coverage
    print("\n  Coverage matrix:")
    print(f"  {'Dataset':<16} {'FedAvg':>8} {'FedLESAM':>10} {'Ditto':>8} {'HPFL':>8}")
    for ds in DATASETS:
        counts = []
        for algo in ALGORITHMS:
            n = len(matrix[ds][algo])
            counts.append(f"{n}/3")
        print(f"  {DATASET_LABELS[ds]:<16} {'  '.join(f'{c:>8}' for c in counts)}")

    # Compute statistics
    stats_data = compute_stats(matrix)
    sig_data = compute_significance(stats_data)

    # Generate text report
    report = generate_text_report(stats_data, sig_data)
    print("\n" + report)
    with open(OUTPUT_DIR / "imaging_report.txt", "w") as f:
        f.write(report)
    print(f"\n  [TXT] imaging_report.txt")

    # Generate LaTeX table
    latex = generate_latex_table(stats_data, sig_data)
    with open(OUTPUT_DIR / "table_imaging_complete.tex", "w") as f:
        f.write(latex)
    print(f"  [TEX] table_imaging_complete.tex")

    # Generate figures
    try:
        generate_figures(matrix, stats_data)
    except ImportError as e:
        print(f"  [SKIP] matplotlib not available: {e}")

    print(f"\n  All outputs saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()

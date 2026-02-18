#!/usr/bin/env python3
"""
FL-EHDS Extended Tabular Analysis — Comprehensive tables and plots.

Reads ALL tabular checkpoints (baseline + sweep) and generates:
  - 11+ LaTeX tables
  - 22+ PDF plots
  - Extended text summary

Usage:
    cd fl-ehds-framework
    python -m benchmarks.analyze_tabular_extended

Output: benchmarks/paper_results_tabular/analysis/

Author: Fabio Liberti
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
from itertools import combinations

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

FRAMEWORK_DIR = Path(__file__).parent.parent
BASE_DIR = FRAMEWORK_DIR / "benchmarks" / "paper_results_tabular"
TABLE_DIR = BASE_DIR / "analysis" / "tables"
PLOT_DIR = BASE_DIR / "analysis" / "plots"

DATASETS = ["PTB_XL", "Cardiovascular", "Breast_Cancer"]
DS_SHORT = {"PTB_XL": "PX", "Cardiovascular": "CV", "Breast_Cancer": "BC"}
DS_LABEL = {
    "PTB_XL": "PTB-XL (5-class ECG)",
    "Cardiovascular": "Cardiovascular (binary)",
    "Breast_Cancer": "Breast Cancer (binary)",
}

ALGORITHMS = ["FedAvg", "FedProx", "Ditto", "FedLC", "FedExP", "FedLESAM", "HPFL"]

ALGO_COLORS = {
    "FedAvg": "#2196F3", "FedProx": "#FF5722", "Ditto": "#4CAF50",
    "FedLC": "#E91E63", "FedExP": "#9C27B0", "FedLESAM": "#FF9800",
    "HPFL": "#00BCD4",
}

plt.rcParams.update({
    "font.family": "serif", "font.size": 11, "axes.titlesize": 13,
    "axes.labelsize": 12, "figure.dpi": 150,
})


# ======================================================================
# Data Loading
# ======================================================================

def load_all_checkpoints():
    """Load baseline + sweep checkpoints."""
    baseline = {}
    sweep = {}

    bp = BASE_DIR / "checkpoint_tabular.json"
    if bp.exists():
        with open(bp) as f:
            data = json.load(f)
        baseline = data.get("completed", {})
        print(f"  Baseline: {len(baseline)} experiments")

    sp = BASE_DIR / "checkpoint_sweep.json"
    if sp.exists():
        with open(sp) as f:
            data = json.load(f)
        sweep = data.get("completed", {})
        print(f"  Sweep: {len(sweep)} experiments")

    return baseline, sweep


def organize_baseline(completed):
    """results[ds][algo] = [runs]"""
    results = defaultdict(lambda: defaultdict(list))
    for key, res in completed.items():
        if "error" in res:
            continue
        results[res["dataset"]][res["algorithm"]].append(res)
    return dict(results)


def organize_sweep_by_phase(sweep):
    """Organize sweep results by phase and variation."""
    phases = {1: [], 2: [], 3: []}
    for key, res in sweep.items():
        if "error" in res:
            continue
        p = res.get("phase", 0)
        if p in phases:
            phases[p].append(res)
    return phases


# ======================================================================
# Helper
# ======================================================================

def _bold_best(val, is_best):
    return rf"\textbf{{{val}}}" if is_best else val


# ======================================================================
# TABLE 1: Main Comparison (Accuracy, F1, Jain)
# ======================================================================

def table_main_comparison(results):
    lines = [
        r"\begin{table*}[t]", r"\centering",
        r"\caption{Federated learning algorithm comparison on tabular healthcare datasets. "
        r"Best accuracy per dataset in \textbf{bold}. Mean $\pm$ std over 3 seeds.}",
        r"\label{tab:tabular_main}",
        r"\small",
        r"\begin{tabular}{l" + "ccc" * len(DATASETS) + "}",
        r"\toprule",
    ]

    # Header
    h1 = r"\textbf{Algorithm}"
    for ds in DATASETS:
        h1 += rf" & \multicolumn{{3}}{{c}}{{\textbf{{{DS_SHORT[ds]}}}}}"
    lines.append(h1 + r" \\")

    h2 = ""
    for _ in DATASETS:
        h2 += r" & Acc (\%) & F1 (\%) & Jain"
    lines.append(r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}")
    lines.append(h2 + r" \\")
    lines.append(r"\midrule")

    # Find best per dataset
    best = {}
    for ds in DATASETS:
        best[ds] = max(
            (np.mean([r["best_metrics"]["accuracy"] for r in results[ds][a]]), a)
            for a in ALGORITHMS if a in results.get(ds, {})
        )

    for algo in ALGORITHMS:
        row = algo
        for ds in DATASETS:
            if algo not in results.get(ds, {}):
                row += " & -- & -- & --"
                continue
            runs = results[ds][algo]
            accs = [r["best_metrics"]["accuracy"] for r in runs]
            f1s = [r["final_metrics"]["f1"] for r in runs]
            jains = [r["fairness"]["jain_index"] for r in runs]

            acc_str = f"{np.mean(accs)*100:.1f}$\\pm${np.std(accs)*100:.1f}"
            f1_str = f"{np.mean(f1s)*100:.1f}"
            jain_str = f"{np.mean(jains):.3f}"

            if best[ds][1] == algo:
                acc_str = r"\textbf{" + acc_str + "}"
            row += f" & {acc_str} & {f1_str} & {jain_str}"
        lines.append(row + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines)


# ======================================================================
# TABLE 2: Extended Metrics
# ======================================================================

def table_extended_metrics(results):
    lines = [
        r"\begin{table*}[t]", r"\centering",
        r"\caption{Extended evaluation metrics for tabular FL experiments (mean over 3 seeds).}",
        r"\label{tab:tabular_extended}",
        r"\footnotesize",
        r"\begin{tabular}{ll" + "c" * 6 + "}",
        r"\toprule",
        r"\textbf{Dataset} & \textbf{Algorithm} & \textbf{Acc} & \textbf{F1} & "
        r"\textbf{Prec} & \textbf{Rec} & \textbf{AUC} & \textbf{Jain} \\",
        r"\midrule",
    ]

    for ds in DATASETS:
        first = True
        for algo in ALGORITHMS:
            if algo not in results.get(ds, {}):
                continue
            runs = results[ds][algo]
            ds_label = DS_SHORT[ds] if first else ""
            first = False

            acc = np.mean([r["best_metrics"]["accuracy"] for r in runs]) * 100
            f1 = np.mean([r["final_metrics"]["f1"] for r in runs]) * 100
            prec = np.mean([r["final_metrics"]["precision"] for r in runs]) * 100
            rec = np.mean([r["final_metrics"]["recall"] for r in runs]) * 100
            auc = np.mean([r["final_metrics"]["auc"] for r in runs]) * 100
            jain = np.mean([r["fairness"]["jain_index"] for r in runs])

            lines.append(
                f"{ds_label} & {algo} & {acc:.1f} & {f1:.1f} & {prec:.1f} & "
                f"{rec:.1f} & {auc:.1f} & {jain:.3f} \\\\"
            )
        if ds != DATASETS[-1]:
            lines.append(r"\midrule")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines)


# ======================================================================
# TABLE 3: Detailed Fairness
# ======================================================================

def table_fairness_detailed(results):
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Per-client fairness analysis. Gap = max--min client accuracy.}",
        r"\label{tab:fairness_detail}",
        r"\footnotesize",
        r"\begin{tabular}{ll cccc}",
        r"\toprule",
        r"\textbf{DS} & \textbf{Algo} & \textbf{Jain} & \textbf{Gini} & "
        r"\textbf{Gap (\%)} & \textbf{Std (\%)} \\",
        r"\midrule",
    ]

    for ds in DATASETS:
        first = True
        for algo in ALGORITHMS:
            if algo not in results.get(ds, {}):
                continue
            runs = results[ds][algo]
            ds_label = DS_SHORT[ds] if first else ""
            first = False

            jain = np.mean([r["fairness"]["jain_index"] for r in runs])
            gini = np.mean([r["fairness"]["gini"] for r in runs])
            gap = np.mean([r["fairness"]["max"] - r["fairness"]["min"] for r in runs]) * 100
            std = np.mean([r["fairness"]["std"] for r in runs]) * 100

            lines.append(
                f"{ds_label} & {algo} & {jain:.3f} & {gini:.3f} & {gap:.1f} & {std:.1f} \\\\"
            )
        if ds != DATASETS[-1]:
            lines.append(r"\midrule")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# ======================================================================
# TABLE 4: Heterogeneity Sweep
# ======================================================================

def table_heterogeneity_sweep(sweep_results):
    phase1 = [r for r in sweep_results.get(1, []) if "error" not in r]
    if not phase1:
        return "% No Phase 1 data available"

    # Organize: ds -> algo -> variation -> [accs]
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in phase1:
        acc = r["best_metrics"]["accuracy"]
        data[r["dataset"]][r["algorithm"]][r["variation"]].append(acc)

    variations = sorted(set(r["variation"] for r in phase1))

    lines = [
        r"\begin{table*}[t]", r"\centering",
        r"\caption{Impact of data heterogeneity ($\alpha$) on FL accuracy (\%). "
        r"Lower $\alpha$ = more non-IID. Mean over 3 seeds.}",
        r"\label{tab:heterogeneity}",
        r"\small",
        r"\begin{tabular}{ll" + "c" * len(variations) + "}",
        r"\toprule",
    ]

    header = r"\textbf{DS} & \textbf{Algorithm}"
    for v in variations:
        label = v.replace("alpha=", r"$\alpha$=").replace("IID", "IID").replace("site-based", "Site")
        header += rf" & \textbf{{{label}}}"
    lines.append(header + r" \\")
    lines.append(r"\midrule")

    for ds in DATASETS:
        first = True
        for algo in ALGORITHMS:
            if algo not in data.get(ds, {}):
                continue
            ds_label = DS_SHORT[ds] if first else ""
            first = False

            row = f"{ds_label} & {algo}"
            for v in variations:
                accs = data[ds][algo].get(v, [])
                if accs:
                    row += f" & {np.mean(accs)*100:.1f}"
                else:
                    row += " & --"
            lines.append(row + r" \\")
        if ds != DATASETS[-1]:
            lines.append(r"\midrule")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines)


# ======================================================================
# TABLE 5: Client Scaling
# ======================================================================

def table_client_scaling(sweep_results):
    phase2 = [r for r in sweep_results.get(2, []) if "error" not in r]
    if not phase2:
        return "% No Phase 2 data available"

    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in phase2:
        acc = r["best_metrics"]["accuracy"]
        data[r["dataset"]][r["algorithm"]][r["num_clients"]].append(acc)

    lines = [
        r"\begin{table*}[t]", r"\centering",
        r"\caption{Impact of client count on FL accuracy (\%) and fairness. Mean over 3 seeds.}",
        r"\label{tab:client_scaling}",
        r"\small",
    ]

    for ds in DATASETS:
        if ds not in data:
            continue
        client_vals = sorted(set(r["num_clients"] for r in phase2 if r["dataset"] == ds))
        if not client_vals:
            continue

        lines.append(rf"\textbf{{{DS_SHORT[ds]}}} \\")
        lines.append(r"\begin{tabular}{l" + "c" * len(client_vals) + "}")

        header = r"\textbf{Algorithm}"
        for nc in client_vals:
            header += rf" & \textbf{{K={nc}}}"
        lines.append(header + r" \\")
        lines.append(r"\hline")

        for algo in ALGORITHMS:
            if algo not in data[ds]:
                continue
            row = algo
            for nc in client_vals:
                accs = data[ds][algo].get(nc, [])
                row += f" & {np.mean(accs)*100:.1f}" if accs else " & --"
            lines.append(row + r" \\")

        lines.append(r"\end{tabular}")
        lines.append(r"\vspace{1mm}")

    lines += [r"\end{table*}"]
    return "\n".join(lines)


# ======================================================================
# TABLE 6: LR Sensitivity
# ======================================================================

def table_lr_sensitivity(sweep_results):
    phase3 = [r for r in sweep_results.get(3, []) if "error" not in r]
    if not phase3:
        return "% No Phase 3 data available"

    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in phase3:
        acc = r["best_metrics"]["accuracy"]
        data[r["dataset"]][r["algorithm"]][r["learning_rate"]].append(acc)

    lr_values = sorted(set(r["learning_rate"] for r in phase3))

    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Learning rate sensitivity analysis. Accuracy (\%) mean over 3 seeds.}",
        r"\label{tab:lr_sensitivity}",
        r"\small",
        r"\begin{tabular}{ll" + "c" * len(lr_values) + "}",
        r"\toprule",
    ]

    header = r"\textbf{DS} & \textbf{Algo}"
    for lr in lr_values:
        header += rf" & \textbf{{{lr}}}"
    lines.append(header + r" \\")
    lines.append(r"\midrule")

    for ds in DATASETS:
        first = True
        for algo in sorted(data.get(ds, {}).keys()):
            ds_label = DS_SHORT[ds] if first else ""
            first = False
            row = f"{ds_label} & {algo}"
            for lr in lr_values:
                accs = data[ds][algo].get(lr, [])
                row += f" & {np.mean(accs)*100:.1f}" if accs else " & --"
            lines.append(row + r" \\")
        if ds != DATASETS[-1]:
            lines.append(r"\midrule")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# ======================================================================
# TABLE 7: Communication Cost
# ======================================================================

def table_comm_cost(results):
    MLP_KB = 10000 * 4 / 1024  # ~39 KB

    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Communication cost: average rounds to best accuracy $\times$ model uploads.}",
        r"\label{tab:comm_cost}",
        r"\small",
        r"\begin{tabular}{l" + "rr" * len(DATASETS) + "}",
        r"\toprule",
    ]

    h1 = r"\textbf{Algorithm}"
    for ds in DATASETS:
        h1 += rf" & \multicolumn{{2}}{{c}}{{\textbf{{{DS_SHORT[ds]}}}}}"
    lines.append(h1 + r" \\")
    lines.append(r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}")
    lines.append(r" & Rnd & KB" * len(DATASETS) + r" \\")
    lines.append(r"\midrule")

    for algo in ALGORITHMS:
        row = algo
        for ds in DATASETS:
            if algo not in results.get(ds, {}):
                row += " & -- & --"
                continue
            runs = results[ds][algo]
            avg_round = np.mean([r["best_round"] for r in runs])
            nc = len(runs[0].get("per_client_acc", {})) or 5
            total_kb = avg_round * 2 * nc * MLP_KB
            row += f" & {avg_round:.0f} & {total_kb:.0f}"
        lines.append(row + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# ======================================================================
# TABLE 8: Statistical Significance
# ======================================================================

def table_significance(results):
    try:
        from scipy import stats
    except ImportError:
        return "% scipy not available"

    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Wilcoxon signed-rank $p$-values vs FedAvg baseline. "
        r"$\dagger$: $p < 0.05$, *: $p < 0.10$.}",
        r"\label{tab:significance}",
        r"\small",
        r"\begin{tabular}{l" + "c" * len(DATASETS) + "}",
        r"\toprule",
        r"\textbf{vs FedAvg} & " + " & ".join(rf"\textbf{{{DS_SHORT[ds]}}}" for ds in DATASETS) + r" \\",
        r"\midrule",
    ]

    for algo in ALGORITHMS:
        if algo == "FedAvg":
            continue
        row = algo
        for ds in DATASETS:
            if algo not in results.get(ds, {}) or "FedAvg" not in results.get(ds, {}):
                row += " & --"
                continue
            a1 = sorted([r["best_metrics"]["accuracy"] for r in results[ds]["FedAvg"]])
            a2 = sorted([r["best_metrics"]["accuracy"] for r in results[ds][algo]])
            try:
                _, p = stats.wilcoxon(a1, a2)
                sig = r"$\dagger$" if p < 0.05 else ("*" if p < 0.10 else "")
                row += f" & {p:.3f}{sig}"
            except ValueError:
                row += " & ="
        lines.append(row + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# ======================================================================
# TABLE 9: Per-Client Accuracy
# ======================================================================

def table_per_client(results):
    lines = [
        r"\begin{table*}[t]", r"\centering",
        r"\caption{Per-client accuracy for best algorithm per dataset (mean over 3 seeds).}",
        r"\label{tab:per_client}",
        r"\small",
    ]

    for ds in DATASETS:
        if ds not in results:
            continue
        # Find best algo
        best_algo = max(
            ALGORITHMS,
            key=lambda a: np.mean([r["best_metrics"]["accuracy"] for r in results[ds].get(a, [{"best_metrics": {"accuracy": 0}}])])
        )
        runs = results[ds][best_algo]
        nc = len(runs[0]["per_client_acc"])

        lines.append(rf"\textbf{{{DS_SHORT[ds]}}} ({best_algo}):")
        lines.append(r"\begin{tabular}{" + "c" * (nc + 2) + "}")

        header = r"\textbf{Client}"
        for c in range(nc):
            header += rf" & \textbf{{C{c}}}"
        header += r" & \textbf{Mean}"
        lines.append(header + r" \\")
        lines.append(r"\hline")

        mean_accs = np.zeros(nc)
        for run in runs:
            for cid, acc in run["per_client_acc"].items():
                mean_accs[int(cid)] += acc
        mean_accs /= len(runs)

        row = "Acc (\\%)"
        for acc in mean_accs:
            row += f" & {acc*100:.1f}"
        row += f" & {np.mean(mean_accs)*100:.1f}"
        lines.append(row + r" \\")
        lines.append(r"\end{tabular}")
        lines.append(r"\vspace{1mm}")

    lines.append(r"\end{table*}")
    return "\n".join(lines)


# ======================================================================
# TABLE 10: Early Stopping Analysis
# ======================================================================

def table_early_stopping(results):
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Early stopping analysis: configured vs actual rounds (mean over 3 seeds).}",
        r"\label{tab:early_stop}",
        r"\small",
        r"\begin{tabular}{ll ccc}",
        r"\toprule",
        r"\textbf{DS} & \textbf{Algo} & \textbf{Max} & \textbf{Actual} & \textbf{Best Rnd} \\",
        r"\midrule",
    ]

    for ds in DATASETS:
        first = True
        for algo in ALGORITHMS:
            if algo not in results.get(ds, {}):
                continue
            runs = results[ds][algo]
            ds_label = DS_SHORT[ds] if first else ""
            first = False

            max_r = runs[0]["config"]["num_rounds"]
            actual = np.mean([r["actual_rounds"] for r in runs])
            best_r = np.mean([r["best_round"] for r in runs])

            lines.append(f"{ds_label} & {algo} & {max_r} & {actual:.0f} & {best_r:.0f} \\\\")
        if ds != DATASETS[-1]:
            lines.append(r"\midrule")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# ======================================================================
# TABLE 11: Dataset Summary
# ======================================================================

def table_dataset_summary(results):
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Tabular dataset characteristics and EHDS readiness.}",
        r"\label{tab:dataset_summary}",
        r"\small",
        r"\begin{tabular}{lccccl}",
        r"\toprule",
        r"\textbf{Dataset} & \textbf{N} & \textbf{Feat} & \textbf{Classes} & "
        r"\textbf{Partition} & \textbf{EHDS} \\",
        r"\midrule",
    ]

    ds_info = {
        "PTB_XL": ("21,799", "9", "5", "Site (52)", "L2 (SCP-ECG)"),
        "Cardiovascular": ("70,000", "11", "2", r"Dirichlet $\alpha$=0.5", "L3 (OMOP)"),
        "Breast_Cancer": ("569", "30", "2", r"Dirichlet $\alpha$=0.5", "L3 (OMOP)"),
    }

    for ds in DATASETS:
        info = ds_info[ds]
        lines.append(f"{DS_SHORT[ds]} & {info[0]} & {info[1]} & {info[2]} & {info[3]} & {info[4]} \\\\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# ======================================================================
# SWEEP PLOTS
# ======================================================================

def plot_heterogeneity_impact(sweep_results):
    """Accuracy vs alpha per dataset."""
    phase1 = [r for r in sweep_results.get(1, []) if "error" not in r]
    if not phase1:
        return

    # Organize: ds -> algo -> alpha -> [accs]
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    alpha_map = {}
    for r in phase1:
        v = r["variation"]
        if v == "IID":
            alpha_val = 100  # plot at right edge
        elif v == "site-based":
            continue  # separate plot
        else:
            alpha_val = float(v.split("=")[1])
        data[r["dataset"]][r["algorithm"]][alpha_val].append(r["best_metrics"]["accuracy"])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, ds in enumerate(DATASETS):
        ax = axes[idx]
        for algo in ALGORITHMS:
            if algo not in data.get(ds, {}):
                continue
            alphas = sorted(data[ds][algo].keys())
            means = [np.mean(data[ds][algo][a]) * 100 for a in alphas]
            labels = [str(a) if a < 100 else "IID" for a in alphas]

            ax.plot(range(len(alphas)), means, marker="o", linewidth=2,
                    color=ALGO_COLORS.get(algo, "#666"), label=algo)

        ax.set_xticks(range(len(alphas)))
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_xlabel(r"Dirichlet $\alpha$ (lower = more non-IID)")
        ax.set_ylabel("Accuracy (%)" if idx == 0 else "")
        ax.set_title(DS_SHORT[ds], fontweight="bold")
        ax.grid(True, alpha=0.3)
        if idx == 2:
            ax.legend(fontsize=8, loc="lower right")

    fig.suptitle("Impact of Data Heterogeneity on FL Performance", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "14_heterogeneity_impact.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  14_heterogeneity_impact.pdf")


def plot_heterogeneity_heatmap(sweep_results):
    """Heatmap: algo × alpha per dataset."""
    phase1 = [r for r in sweep_results.get(1, []) if "error" not in r and "alpha=" in r.get("variation", "")]
    if not phase1:
        return

    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in phase1:
        alpha = float(r["variation"].split("=")[1])
        data[r["dataset"]][r["algorithm"]][alpha].append(r["best_metrics"]["accuracy"])

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for idx, ds in enumerate(DATASETS):
        ax = axes[idx]
        if ds not in data:
            continue

        algos = [a for a in ALGORITHMS if a in data[ds]]
        alphas = sorted(set(a for algo_data in data[ds].values() for a in algo_data.keys()))

        matrix = np.zeros((len(algos), len(alphas)))
        for i, algo in enumerate(algos):
            for j, alpha in enumerate(alphas):
                accs = data[ds][algo].get(alpha, [])
                matrix[i, j] = np.mean(accs) * 100 if accs else 0

        im = ax.imshow(matrix, cmap="YlGnBu", aspect="auto")
        ax.set_yticks(range(len(algos)))
        ax.set_yticklabels(algos, fontsize=9)
        ax.set_xticks(range(len(alphas)))
        ax.set_xticklabels([f"α={a}" for a in alphas], fontsize=9)
        ax.set_title(DS_SHORT[ds], fontweight="bold")

        for i in range(len(algos)):
            for j in range(len(alphas)):
                ax.text(j, i, f"{matrix[i,j]:.1f}", ha="center", va="center", fontsize=8)

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Acc (%)")

    fig.suptitle("Heterogeneity Heatmap: Accuracy by Algorithm × Alpha", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "15_heterogeneity_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  15_heterogeneity_heatmap.pdf")


def plot_client_scaling(sweep_results):
    """Accuracy vs num_clients per dataset."""
    phase2 = [r for r in sweep_results.get(2, []) if "error" not in r]
    if not phase2:
        return

    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in phase2:
        data[r["dataset"]][r["algorithm"]][r["num_clients"]].append(r["best_metrics"]["accuracy"])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, ds in enumerate(DATASETS):
        ax = axes[idx]
        for algo in ALGORITHMS:
            if algo not in data.get(ds, {}):
                continue
            ncs = sorted(data[ds][algo].keys())
            means = [np.mean(data[ds][algo][nc]) * 100 for nc in ncs]

            ax.plot(ncs, means, marker="s", linewidth=2,
                    color=ALGO_COLORS.get(algo, "#666"), label=algo)

        ax.set_xlabel("Number of Clients (K)")
        ax.set_ylabel("Accuracy (%)" if idx == 0 else "")
        ax.set_title(DS_SHORT[ds], fontweight="bold")
        ax.grid(True, alpha=0.3)
        if idx == 2:
            ax.legend(fontsize=8)

    fig.suptitle("Client Scalability: Accuracy vs Number of Clients", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "16_client_scaling.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  16_client_scaling.pdf")


def plot_client_scaling_fairness(sweep_results):
    """Jain index vs num_clients."""
    phase2 = [r for r in sweep_results.get(2, []) if "error" not in r]
    if not phase2:
        return

    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in phase2:
        data[r["dataset"]][r["algorithm"]][r["num_clients"]].append(r["fairness"]["jain_index"])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, ds in enumerate(DATASETS):
        ax = axes[idx]
        for algo in ALGORITHMS:
            if algo not in data.get(ds, {}):
                continue
            ncs = sorted(data[ds][algo].keys())
            means = [np.mean(data[ds][algo][nc]) for nc in ncs]

            ax.plot(ncs, means, marker="^", linewidth=2,
                    color=ALGO_COLORS.get(algo, "#666"), label=algo)

        ax.set_xlabel("Number of Clients (K)")
        ax.set_ylabel("Jain Fairness Index" if idx == 0 else "")
        ax.set_title(DS_SHORT[ds], fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        if idx == 2:
            ax.legend(fontsize=8)

    fig.suptitle("Client Scalability: Fairness vs Number of Clients", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "17_client_scaling_fairness.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  17_client_scaling_fairness.pdf")


def plot_lr_sensitivity(sweep_results):
    """Accuracy vs learning rate."""
    phase3 = [r for r in sweep_results.get(3, []) if "error" not in r]
    if not phase3:
        return

    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in phase3:
        data[r["dataset"]][r["algorithm"]][r["learning_rate"]].append(r["best_metrics"]["accuracy"])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, ds in enumerate(DATASETS):
        ax = axes[idx]
        for algo in sorted(data.get(ds, {}).keys()):
            lrs = sorted(data[ds][algo].keys())
            means = [np.mean(data[ds][algo][lr]) * 100 for lr in lrs]

            ax.semilogx(lrs, means, marker="D", linewidth=2,
                        color=ALGO_COLORS.get(algo, "#666"), label=algo)

        ax.set_xlabel("Learning Rate (log scale)")
        ax.set_ylabel("Accuracy (%)" if idx == 0 else "")
        ax.set_title(DS_SHORT[ds], fontweight="bold")
        ax.grid(True, alpha=0.3)
        if idx == 2:
            ax.legend(fontsize=8)

    fig.suptitle("Learning Rate Sensitivity Analysis", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "18_lr_sensitivity.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  18_lr_sensitivity.pdf")


def plot_ptbxl_site_vs_dirichlet(sweep_results):
    """PTB-XL: natural site partition vs Dirichlet."""
    phase1 = [r for r in sweep_results.get(1, []) if "error" not in r and r["dataset"] == "PTB_XL"]
    if not phase1:
        return

    data = defaultdict(lambda: defaultdict(list))
    for r in phase1:
        data[r["algorithm"]][r["variation"]].append(r["best_metrics"]["accuracy"])

    fig, ax = plt.subplots(figsize=(10, 6))

    algos = [a for a in ALGORITHMS if a in data]
    variations = sorted(set(r["variation"] for r in phase1))
    x = np.arange(len(algos))
    width = 0.8 / len(variations)

    colors = plt.cm.Set2(np.linspace(0, 1, len(variations)))

    for i, v in enumerate(variations):
        means = [np.mean(data[a].get(v, [0])) * 100 for a in algos]
        label = v.replace("alpha=", "α=")
        ax.bar(x + i * width - 0.4 + width/2, means, width, label=label,
               color=colors[i], edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(algos, fontsize=10)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("PTB-XL: Natural Site Partitioning vs Dirichlet", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(PLOT_DIR / "19_ptbxl_site_vs_dirichlet.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  19_ptbxl_site_vs_dirichlet.pdf")


def plot_algorithm_comparison_grid(results):
    """3×3 grid: metric × dataset."""
    metrics = {
        "Accuracy": lambda r: r["best_metrics"]["accuracy"] * 100,
        "F1 Score": lambda r: r["final_metrics"]["f1"] * 100,
        "Jain Index": lambda r: r["fairness"]["jain_index"],
    }

    fig, axes = plt.subplots(3, 3, figsize=(16, 14))

    for row, (metric_name, metric_fn) in enumerate(metrics.items()):
        for col, ds in enumerate(DATASETS):
            ax = axes[row][col]
            algos = [a for a in ALGORITHMS if a in results.get(ds, {})]

            vals = [np.mean([metric_fn(r) for r in results[ds][a]]) for a in algos]
            stds = [np.std([metric_fn(r) for r in results[ds][a]]) for a in algos]
            colors = [ALGO_COLORS[a] for a in algos]

            ax.barh(range(len(algos)), vals, xerr=stds, capsize=3,
                    color=colors, edgecolor="black", linewidth=0.5, alpha=0.85)
            ax.set_yticks(range(len(algos)))
            ax.set_yticklabels(algos if col == 0 else [], fontsize=9)
            ax.set_xlabel(metric_name if row == 2 else "")

            if row == 0:
                ax.set_title(DS_SHORT[ds], fontweight="bold")
            if col == 0:
                ax.set_ylabel(metric_name, fontsize=11)

            ax.grid(axis="x", alpha=0.3)

    fig.suptitle("Algorithm Comparison: Accuracy × F1 × Fairness", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "20_algorithm_grid.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  20_algorithm_grid.pdf")


def plot_best_config_summary(results, sweep_results):
    """Visual summary of best configuration per dataset."""
    fig, ax = plt.subplots(figsize=(12, 6))

    rows = []
    for ds in DATASETS:
        if ds not in results:
            continue
        best_algo = max(
            ALGORITHMS,
            key=lambda a: np.mean([r["best_metrics"]["accuracy"] for r in results[ds].get(a, [{"best_metrics": {"accuracy": 0}}])])
        )
        runs = results[ds][best_algo]
        acc = np.mean([r["best_metrics"]["accuracy"] for r in runs]) * 100
        jain = np.mean([r["fairness"]["jain_index"] for r in runs])
        rows.append((DS_SHORT[ds], best_algo, acc, jain))

    x = np.arange(len(rows))
    colors = [ALGO_COLORS[r[1]] for r in rows]

    bars = ax.bar(x, [r[2] for r in rows], color=colors, edgecolor="black",
                  linewidth=1, alpha=0.85, width=0.5)

    for i, (ds, algo, acc, jain) in enumerate(rows):
        ax.text(i, acc + 1, f"{algo}\n{acc:.1f}%\nJain={jain:.3f}",
               ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in rows], fontsize=12)
    ax.set_ylabel("Best Accuracy (%)", fontsize=12)
    ax.set_title("Best Algorithm per Dataset — Summary", fontsize=14)
    ax.set_ylim(0, 110)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(PLOT_DIR / "21_best_config_summary.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  21_best_config_summary.pdf")


# ======================================================================
# Main
# ======================================================================

def main():
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading checkpoints...")
    baseline, sweep = load_all_checkpoints()

    results = organize_baseline(baseline)
    sweep_results = organize_sweep_by_phase(sweep)

    has_baseline = bool(baseline)
    has_sweep = bool(sweep)

    print(f"\nBaseline: {len(baseline)} experiments")
    for p, exps in sweep_results.items():
        print(f"Sweep Phase {p}: {len(exps)} experiments")

    # === TABLES ===
    if has_baseline:
        print("\n--- Generating Tables ---")

        tables = {
            "table_main_comparison.tex": table_main_comparison(results),
            "table_extended_metrics.tex": table_extended_metrics(results),
            "table_fairness_detailed.tex": table_fairness_detailed(results),
            "table_comm_cost.tex": table_comm_cost(results),
            "table_significance.tex": table_significance(results),
            "table_per_client.tex": table_per_client(results),
            "table_early_stopping.tex": table_early_stopping(results),
            "table_dataset_summary.tex": table_dataset_summary(results),
        }

        for name, content in tables.items():
            (TABLE_DIR / name).write_text(content)
            print(f"  {name}")

    if has_sweep:
        sweep_tables = {
            "table_heterogeneity_sweep.tex": table_heterogeneity_sweep(sweep_results),
            "table_client_scaling.tex": table_client_scaling(sweep_results),
            "table_lr_sensitivity.tex": table_lr_sensitivity(sweep_results),
        }
        for name, content in sweep_tables.items():
            (TABLE_DIR / name).write_text(content)
            print(f"  {name}")

    # === PLOTS ===
    if has_baseline:
        print("\n--- Generating Baseline Plots ---")
        plot_algorithm_comparison_grid(results)
        plot_best_config_summary(results, sweep_results)

    if has_sweep:
        print("\n--- Generating Sweep Plots ---")
        plot_heterogeneity_impact(sweep_results)
        plot_heterogeneity_heatmap(sweep_results)
        plot_client_scaling(sweep_results)
        plot_client_scaling_fairness(sweep_results)
        plot_lr_sensitivity(sweep_results)
        plot_ptbxl_site_vs_dirichlet(sweep_results)

    print(f"\nAll outputs saved to:")
    print(f"  Tables: {TABLE_DIR}")
    print(f"  Plots:  {PLOT_DIR}")
    print("Done!")


if __name__ == "__main__":
    main()

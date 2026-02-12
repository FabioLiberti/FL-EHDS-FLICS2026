#!/usr/bin/env python3
"""
Incremental Paper Output Generator for FLICS 2026.

Reads checkpoint files from completed experiments and generates/updates
LaTeX tables and figures with whatever data is available. Run this at
any time to see the current state of results.

Usage:
    cd fl-ehds-framework
    python -m benchmarks.generate_paper_outputs           # status + generate all
    python -m benchmarks.generate_paper_outputs --status   # status only (no files)
    python -m benchmarks.generate_paper_outputs --tables    # tables only
    python -m benchmarks.generate_paper_outputs --figures   # figures only

Output: benchmarks/paper_results/

Author: Fabio Liberti
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import numpy as np

FRAMEWORK_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(FRAMEWORK_DIR))

OUTPUT_DIR = FRAMEWORK_DIR / "benchmarks" / "paper_results"

IMAGING_DATASETS = {
    "Brain_Tumor": {"short": "BT"},
    "chest_xray": {"short": "CX"},
    "Skin_Cancer": {"short": "SC"},
}
TABULAR_DATASETS = {
    "Diabetes": {"short": "DM"},
    "Heart_Disease": {"short": "HD"},
}
ALL_DATASETS = list(IMAGING_DATASETS.keys()) + list(TABULAR_DATASETS.keys())
DS_SHORTS = {**{k: v["short"] for k, v in IMAGING_DATASETS.items()},
             **{k: v["short"] for k, v in TABULAR_DATASETS.items()}}

ALGORITHMS = ["FedAvg", "FedProx", "SCAFFOLD", "FedNova", "Ditto"]
ALGO_COLORS = ["#2196F3", "#FF9800", "#4CAF50", "#F44336", "#9C27B0"]

SEEDS = [42, 123, 456]


# ======================================================================
# Checkpoint Loading
# ======================================================================

def load_checkpoint(name: str) -> Optional[Dict]:
    path = OUTPUT_DIR / f"checkpoint_{name}.json"
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return None


NONIID_ALPHAS = [0.1, 0.5, 1.0, 5.0]
NONIID_ALGOS = ["FedAvg", "FedProx", "SCAFFOLD"]


def load_all_checkpoints() -> Dict[str, Any]:
    return {
        "p12": load_checkpoint("p12_multidataset"),
        "p13": load_checkpoint("p13_ablation"),
        "p14": load_checkpoint("p14_noniid"),
        "p21": load_checkpoint("p21_significance"),
        "p22": load_checkpoint("p22_attack"),
        "comm": load_checkpoint("comm_costs"),
    }


# ======================================================================
# Status Dashboard
# ======================================================================

def print_status(checkpoints: Dict[str, Any]) -> None:
    print("=" * 70)
    print("  FL-EHDS Paper Results - Status Dashboard")
    print("=" * 70)

    # P1.2 Multi-Dataset
    p12 = checkpoints.get("p12")
    print("\n  P1.2 Multi-Dataset FL Comparison")
    print("  " + "-" * 50)
    if p12:
        completed = p12.get("completed", {})
        ok = {k: v for k, v in completed.items() if "error" not in v}
        err = {k: v for k, v in completed.items() if "error" in v}
        total_expected = len(ALGORITHMS) * len(ALL_DATASETS) * len(SEEDS)

        print(f"  Completed: {len(ok)}/{total_expected}  Errors: {len(err)}")
        print()

        # Per-dataset breakdown
        header = f"  {'Dataset':<16}"
        for algo in ALGORITHMS:
            header += f" {algo:<10}"
        print(header)

        for ds in ALL_DATASETS:
            row = f"  {ds:<16}"
            for algo in ALGORITHMS:
                seeds_done = []
                for seed in SEEDS:
                    key = f"{ds}_{algo}_{seed}"
                    if key in ok:
                        seeds_done.append(seed)
                if len(seeds_done) == len(SEEDS):
                    row += f" {'OK':<10}"
                elif seeds_done:
                    row += f" {len(seeds_done)}/{len(SEEDS):<7}"
                else:
                    row += f" {'--':<10}"
            print(row)

        # Show top results
        print()
        print(f"  {'Dataset':<16} {'Algorithm':<12} {'Acc%':<8} {'F1':<8} {'AUC':<8}")
        for ds in ALL_DATASETS:
            best_acc = 0
            best_row = None
            for algo in ALGORITHMS:
                accs = []
                for seed in SEEDS:
                    rec = ok.get(f"{ds}_{algo}_{seed}", {})
                    fm = rec.get("final_metrics")
                    if fm:
                        accs.append(fm["accuracy"])
                if accs and np.mean(accs) > best_acc:
                    best_acc = np.mean(accs)
                    f1s = [ok[f"{ds}_{algo}_{s}"]["final_metrics"]["f1"]
                           for s in SEEDS if f"{ds}_{algo}_{s}" in ok]
                    aucs = [ok[f"{ds}_{algo}_{s}"]["final_metrics"]["auc"]
                            for s in SEEDS if f"{ds}_{algo}_{s}" in ok]
                    best_row = (ds, algo, best_acc, np.mean(f1s), np.mean(aucs))
            if best_row:
                ds_n, al, ac, f1, auc = best_row
                print(f"  {ds_n:<16} {al:<12} {ac*100:>5.1f}    {f1:>5.2f}    {auc:>5.2f}")
    else:
        print("  No checkpoint found")

    # P1.3 Ablation
    p13 = checkpoints.get("p13")
    print("\n  P1.3 Ablation Study (chest_xray)")
    print("  " + "-" * 50)
    if p13:
        completed = p13.get("completed", {})
        ok = {k: v for k, v in completed.items() if "error" not in v}
        err = {k: v for k, v in completed.items() if "error" in v}
        print(f"  Completed: {len(ok)}  Errors: {len(err)}")

        # Group by factor
        groups = {"clip": [], "epsilon": [], "model": [], "classweights": []}
        for key in ok:
            for g in groups:
                if key.startswith(g):
                    groups[g].append(key)
        for g, keys in groups.items():
            if keys:
                print(f"    {g}: {len(keys)} experiments")
    else:
        print("  No checkpoint found")

    # P1.4 Non-IID Severity
    p14 = checkpoints.get("p14")
    print("\n  P1.4 Non-IID Severity Study (alpha sweep)")
    print("  " + "-" * 50)
    if p14:
        completed = p14.get("completed", {})
        ok = {k: v for k, v in completed.items() if "error" not in v}
        total_expected = len(NONIID_ALPHAS) * len(NONIID_ALGOS) * len(SEEDS)
        print(f"  Completed: {len(ok)}/{total_expected}")
        for alpha in NONIID_ALPHAS:
            count = sum(1 for k in ok if k.startswith(f"alpha_{alpha}_"))
            print(f"    alpha={alpha}: {count}/{len(NONIID_ALGOS) * len(SEEDS)}")
    else:
        print("  No checkpoint found")

    # P2.1 Significance
    p21 = checkpoints.get("p21")
    print("\n  P2.1 Statistical Significance")
    print("  " + "-" * 50)
    if p21:
        n_comparisons = sum(len(v) for v in p21.values() if isinstance(v, dict))
        print(f"  Comparisons computed: {n_comparisons}")
        sig_count = 0
        for ds_data in p21.values():
            if isinstance(ds_data, dict):
                for algo_data in ds_data.values():
                    if isinstance(algo_data, dict) and algo_data.get("accuracy_sig"):
                        sig_count += 1
        if sig_count:
            print(f"  Significant differences (p<0.05): {sig_count}")
    else:
        print("  No checkpoint found (needs P1.2 with 2+ seeds)")

    # P2.2 Attack
    p22 = checkpoints.get("p22")
    print("\n  P2.2 Privacy Attack (DLG)")
    print("  " + "-" * 50)
    if p22:
        completed = p22.get("completed", {})
        ok = {k: v for k, v in completed.items() if "error" not in v}
        print(f"  Completed: {len(ok)}")
        for eps_label in ["inf", "10.0", "5.0", "1.0"]:
            vals = [v["reconstruction_mse"] for k, v in ok.items()
                    if k.startswith(f"eps_{eps_label}") and "reconstruction_mse" in v]
            if vals:
                eps_disp = "inf (no DP)" if eps_label == "inf" else f"eps={eps_label}"
                print(f"    {eps_disp:<16} MSE={np.mean(vals):.3f} +/- {np.std(vals):.3f}")
    else:
        print("  No checkpoint found")

    # Output files
    print("\n  Generated Files")
    print("  " + "-" * 50)
    if OUTPUT_DIR.exists():
        for f in sorted(OUTPUT_DIR.iterdir()):
            if f.suffix in (".tex", ".pdf", ".png") and not f.name.startswith("checkpoint"):
                size = f.stat().st_size
                if size > 1024:
                    print(f"    {f.name:<40} {size/1024:.0f} KB")
                else:
                    print(f"    {f.name:<40} {size} B")
    else:
        print("  No output directory")

    print("\n" + "=" * 70)


# ======================================================================
# Aggregation Helpers
# ======================================================================

def _agg(completed: Dict, dataset: str, algo: str, metric: str = "accuracy") -> Tuple[float, float]:
    vals = []
    for seed in SEEDS:
        rec = completed.get(f"{dataset}_{algo}_{seed}", {})
        fm = rec.get("final_metrics")
        if fm and metric in fm:
            vals.append(fm[metric])
    if not vals:
        return 0.0, 0.0
    return float(np.mean(vals)), float(np.std(vals))


def _agg_ablation(completed: Dict, prefix: str, metric: str = "accuracy") -> Tuple[float, float]:
    vals = []
    for seed in SEEDS:
        rec = completed.get(f"{prefix}_{seed}", {})
        fm = rec.get("final_metrics")
        if fm and metric in fm:
            vals.append(fm[metric])
    if not vals:
        return 0.0, 0.0
    return float(np.mean(vals)), float(np.std(vals))


# ======================================================================
# Table Generation (imported logic from run_paper_experiments)
# ======================================================================

def generate_multi_dataset_table(p12: Dict, sig: Dict) -> str:
    completed = p12.get("completed", {})
    ok = {k: v for k, v in completed.items() if "error" not in v}

    lines = []
    lines.append(r"\begin{table*}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Multi-Dataset Federated Learning Results}")
    lines.append(r"\label{tab:multi_dataset}")
    lines.append(r"\small")

    col_spec = "l" + "|ccc" * len(ALL_DATASETS)
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")

    header1 = r"\textbf{Algorithm}"
    for idx, ds in enumerate(ALL_DATASETS):
        sep = "c|" if idx < len(ALL_DATASETS) - 1 else "c"
        header1 += r" & \multicolumn{3}{" + sep + r"}{\textbf{" + DS_SHORTS[ds] + "}}"
    lines.append(header1 + r" \\")

    header2 = ""
    for _ in ALL_DATASETS:
        header2 += r" & Acc & F1 & AUC"
    lines.append(header2 + r" \\")
    lines.append(r"\midrule")

    for algo in ALGORITHMS:
        row = algo
        for ds in ALL_DATASETS:
            for metric in ["accuracy", "f1", "auc"]:
                mean, std = _agg(ok, ds, algo, metric)
                if mean == 0:
                    row += " & --"
                    continue
                if metric == "accuracy":
                    cell = f"{mean*100:.1f}"
                else:
                    cell = f"{mean:.2f}"
                if algo != "FedAvg" and ds in sig and algo in sig[ds]:
                    p = sig[ds][algo].get(f"{metric}_p", 1.0)
                    if p < 0.01:
                        cell += r"$^{**}$"
                    elif p < 0.05:
                        cell += r"$^{*}$"
                cell += r"{\tiny$\pm$" + f"{std:.2f}" + "}"
                row += f" & {cell}"
        lines.append(row + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\vspace{1mm}")
    lines.append(r"\footnotesize{Non-IID, ResNet18 (imaging) / MLP (tabular), 3 seeds. "
                 r"$^{*}p<0.05$, $^{**}p<0.01$ vs FedAvg (paired t-test). "
                 r"BT=Brain Tumor, CX=Chest X-ray, SC=Skin Cancer, DM=Diabetes, HD=Heart Disease.}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


def generate_ablation_table(p13: Dict) -> str:
    completed = p13.get("completed", {})
    ok = {k: v for k, v in completed.items() if "error" not in v}

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Ablation Study on Chest X-ray Dataset}")
    lines.append(r"\label{tab:ablation}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{llccc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Factor} & \textbf{Value} & \textbf{Acc.(\%)} & \textbf{F1} & \textbf{AUC} \\")
    lines.append(r"\midrule")

    def _row(prefix, label):
        acc_m, acc_s = _agg_ablation(ok, prefix, "accuracy")
        f1_m, f1_s = _agg_ablation(ok, prefix, "f1")
        auc_m, auc_s = _agg_ablation(ok, prefix, "auc")
        if acc_m == 0:
            return f"& {label} & -- & -- & -- \\\\"
        return (f"& {label} & {acc_m*100:.1f}$\\pm${acc_s*100:.1f} "
                f"& {f1_m:.2f}$\\pm${f1_s:.2f} & {auc_m:.2f}$\\pm${auc_s:.2f} \\\\")

    lines.append(r"\multirow{3}{*}{Grad. Clip $C$}")
    for c in [0.5, 1.0, 2.0]:
        lines.append(_row(f"clip_{c}", f"$C$={c}"))
    lines.append(r"\midrule")

    lines.append(r"\multirow{6}{*}{Privacy $\varepsilon$}")
    for eps in [0.5, 1.0, 2.0, 5.0, 10.0]:
        lines.append(_row(f"epsilon_{eps}", f"$\\varepsilon$={eps}"))
    lines.append(_row("epsilon_inf", r"$\infty$ (no DP)"))
    lines.append(r"\midrule")

    lines.append(r"\multirow{2}{*}{Model}")
    lines.append(_row("model_cnn", "HealthcareCNN"))
    lines.append(_row("model_resnet18", "ResNet18"))
    lines.append(r"\midrule")

    lines.append(r"\multirow{2}{*}{Class Wt.}")
    lines.append(_row("classweights_True", "On"))
    lines.append(_row("classweights_False", "Off"))

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\vspace{1mm}")
    lines.append(r"\footnotesize{FedAvg, 5 clients, 15 rounds, ResNet18 (unless noted). Mean$\pm$std over 3 seeds.}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def generate_attack_table(p22: Dict) -> str:
    completed = p22.get("completed", {})
    ok = {k: v for k, v in completed.items() if "error" not in v}

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Gradient Inversion Attack: Reconstruction Quality vs. DP Noise}")
    lines.append(r"\label{tab:attack}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{lcc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{$\varepsilon$} & \textbf{Recon. MSE} & \textbf{Protection} \\")
    lines.append(r"\midrule")

    for eps_label, prot in [("inf", "None"), ("10.0", "Low"), ("5.0", "Medium"), ("1.0", "High")]:
        vals = []
        for seed in SEEDS:
            rec = ok.get(f"eps_{eps_label}_{seed}", {})
            if "reconstruction_mse" in rec:
                vals.append(rec["reconstruction_mse"])
        if vals:
            m, s = np.mean(vals), np.std(vals)
            eps_disp = r"$\infty$ (no DP)" if eps_label == "inf" else f"${eps_label}$"
            lines.append(f"{eps_disp} & ${m:.3f} \\pm {s:.3f}$ & {prot} \\\\")
        else:
            lines.append(f"{eps_label} & -- & {prot} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\vspace{1mm}")
    lines.append(r"\footnotesize{DLG attack on synthetic tabular data (32 samples, MLP). "
                 r"Higher MSE = better privacy. Mean$\pm$std over 3 seeds.}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ======================================================================
# Figure Generation
# ======================================================================

def generate_figures(p12: Dict, p13: Dict, p22: Dict, out_dir: Path) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 11, "font.family": "serif",
        "axes.labelsize": 12, "axes.titlesize": 13,
        "xtick.labelsize": 10, "ytick.labelsize": 10,
        "legend.fontsize": 9, "figure.dpi": 150,
    })

    count = 0

    if p12 and p12.get("completed"):
        ok = {k: v for k, v in p12["completed"].items() if "error" not in v}
        if ok:
            _fig_multi_dataset_bars(ok, out_dir, plt)
            count += 1
            _fig_convergence(ok, out_dir, plt)
            count += 1

    if p13 and p13.get("completed"):
        ok = {k: v for k, v in p13["completed"].items() if "error" not in v}
        if ok:
            eps_keys = [k for k in ok if k.startswith("epsilon_")]
            if eps_keys:
                _fig_epsilon_tradeoff(ok, out_dir, plt)
                count += 1
            model_keys = [k for k in ok if k.startswith("model_")]
            if model_keys:
                _fig_model_comparison(ok, out_dir, plt)
                count += 1

    if p22 and p22.get("completed"):
        ok = {k: v for k, v in p22["completed"].items() if "error" not in v}
        if ok:
            _fig_attack_mse(ok, out_dir, plt)
            count += 1

    return count


def _fig_multi_dataset_bars(completed, out_dir, plt):
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(ALL_DATASETS))
    width = 0.15

    has_data = False
    for i, algo in enumerate(ALGORITHMS):
        means, stds = [], []
        for ds in ALL_DATASETS:
            m, s = _agg(completed, ds, algo, "accuracy")
            means.append(m * 100)
            stds.append(s * 100)
        if any(m > 0 for m in means):
            has_data = True
        ax.bar(x + i * width, means, width, yerr=stds, label=algo,
               color=ALGO_COLORS[i], capsize=3, alpha=0.85)

    if not has_data:
        plt.close(fig)
        return

    ds_labels = [DS_SHORTS[d] for d in ALL_DATASETS]
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(ds_labels)
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("Multi-Dataset Federated Learning Comparison")

    for fmt in ["pdf", "png"]:
        fig.savefig(out_dir / f"fig_multi_dataset_comparison.{fmt}", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"    fig_multi_dataset_comparison.pdf")


def _fig_convergence(completed, out_dir, plt):
    # Find which datasets have data
    ds_with_data = []
    for ds in ALL_DATASETS:
        for algo in ALGORITHMS:
            for seed in SEEDS:
                rec = completed.get(f"{ds}_{algo}_{seed}", {})
                if rec.get("history"):
                    ds_with_data.append(ds)
                    break
            if ds in ds_with_data:
                break

    if not ds_with_data:
        return

    fig, axes = plt.subplots(1, len(ds_with_data), figsize=(4 * len(ds_with_data), 4), sharey=False)
    if len(ds_with_data) == 1:
        axes = [axes]

    for ax, ds in zip(axes, ds_with_data):
        for i, algo in enumerate(ALGORITHMS):
            all_hist = []
            for seed in SEEDS:
                rec = completed.get(f"{ds}_{algo}_{seed}", {})
                h = rec.get("history")
                if h:
                    all_hist.append([r["accuracy"] for r in h])

            if not all_hist:
                continue

            min_len = min(len(h) for h in all_hist)
            arr = np.array([h[:min_len] for h in all_hist])
            mean = arr.mean(axis=0) * 100
            std = arr.std(axis=0) * 100
            rounds = np.arange(1, min_len + 1)

            ax.plot(rounds, mean, color=ALGO_COLORS[i], label=algo, linewidth=1.5)
            if len(all_hist) > 1:
                ax.fill_between(rounds, mean - std, mean + std, color=ALGO_COLORS[i], alpha=0.15)

        ax.set_title(DS_SHORTS.get(ds, ds))
        ax.set_xlabel("Round")
        ax.grid(alpha=0.3)
        if ax == axes[0]:
            ax.set_ylabel("Accuracy (%)")

    axes[-1].legend(loc="lower right", fontsize=8)
    fig.suptitle("Convergence Curves per Dataset", fontsize=14)
    fig.tight_layout()

    for fmt in ["pdf", "png"]:
        fig.savefig(out_dir / f"fig_convergence_curves.{fmt}", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"    fig_convergence_curves.pdf")


def _fig_epsilon_tradeoff(completed, out_dir, plt):
    epsilons_labels = [0.5, 1.0, 2.0, 5.0, 10.0, "inf"]
    epsilons_x = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
    means, stds = [], []

    for e in epsilons_labels:
        m, s = _agg_ablation(completed, f"epsilon_{e}", "accuracy")
        means.append(m * 100)
        stds.append(s * 100)

    if all(m == 0 for m in means):
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(epsilons_x, means, yerr=stds, marker="o", color="#2196F3",
                capsize=5, linewidth=2, markersize=8)
    ax.set_xlabel(r"Privacy Budget $\varepsilon$")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Privacy-Utility Tradeoff (Chest X-ray)")
    ax.set_xscale("log")
    ax.set_xticks(epsilons_x)
    ax.set_xticklabels(["0.5", "1", "2", "5", "10", r"$\infty$"])
    ax.grid(alpha=0.3)

    for fmt in ["pdf", "png"]:
        fig.savefig(out_dir / f"fig_ablation_epsilon.{fmt}", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"    fig_ablation_epsilon.pdf")


def _fig_model_comparison(completed, out_dir, plt):
    models = ["cnn", "resnet18"]
    labels = ["HealthcareCNN", "ResNet18"]
    metrics = ["accuracy", "f1", "auc"]

    has_data = False
    for mt in models:
        m, _ = _agg_ablation(completed, f"model_{mt}", "accuracy")
        if m > 0:
            has_data = True

    if not has_data:
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(metrics))
    width = 0.3

    for i, (mt, label) in enumerate(zip(models, labels)):
        vals, errs = [], []
        for m in metrics:
            mean, std = _agg_ablation(completed, f"model_{mt}", m)
            vals.append(mean * 100 if m == "accuracy" else mean)
            errs.append(std * 100 if m == "accuracy" else std)
        ax.bar(x + i * width, vals, width, yerr=errs, label=label,
               color=ALGO_COLORS[i], capsize=4, alpha=0.85)

    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(["Accuracy (%)", "F1 Score", "AUC"])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("Model Architecture Comparison (Chest X-ray)")

    for fmt in ["pdf", "png"]:
        fig.savefig(out_dir / f"fig_model_comparison.{fmt}", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"    fig_model_comparison.pdf")


def _fig_attack_mse(completed, out_dir, plt):
    eps_labels = ["inf", "10.0", "5.0", "1.0"]
    eps_display = [r"$\infty$", "10", "5", "1"]
    means, stds = [], []

    for e in eps_labels:
        vals = []
        for seed in SEEDS:
            rec = completed.get(f"eps_{e}_{seed}", {})
            if "reconstruction_mse" in rec:
                vals.append(rec["reconstruction_mse"])
        if vals:
            means.append(np.mean(vals))
            stds.append(np.std(vals))
        else:
            means.append(0)
            stds.append(0)

    if all(m == 0 for m in means):
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#F44336", "#FF9800", "#FFC107", "#4CAF50"]
    ax.bar(range(len(eps_labels)), means, yerr=stds,
           color=colors, capsize=5, alpha=0.85)
    ax.set_xticks(range(len(eps_labels)))
    ax.set_xticklabels(eps_display)
    ax.set_xlabel(r"Privacy Budget $\varepsilon$")
    ax.set_ylabel("Reconstruction MSE")
    ax.set_title("Gradient Inversion Attack: DP Protection")
    ax.grid(axis="y", alpha=0.3)

    for fmt in ["pdf", "png"]:
        fig.savefig(out_dir / f"fig_attack_mse.{fmt}", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"    fig_attack_mse.pdf")


# ======================================================================
# Significance (recompute from P1.2 data)
# ======================================================================

def compute_significance(p12: Dict) -> Dict:
    from scipy import stats

    completed = p12.get("completed", {})
    ok = {k: v for k, v in completed.items() if "error" not in v}

    sig = {}
    for ds in ALL_DATASETS:
        sig[ds] = {}
        fa_vals = {"accuracy": [], "f1": [], "auc": []}
        for seed in SEEDS:
            rec = ok.get(f"{ds}_FedAvg_{seed}", {})
            fm = rec.get("final_metrics")
            if fm:
                for m in fa_vals:
                    fa_vals[m].append(fm[m])

        if len(fa_vals["accuracy"]) < 2:
            continue

        for algo in ALGORITHMS:
            if algo == "FedAvg":
                continue
            a_vals = {"accuracy": [], "f1": [], "auc": []}
            for seed in SEEDS:
                rec = ok.get(f"{ds}_{algo}_{seed}", {})
                fm = rec.get("final_metrics")
                if fm:
                    for m in a_vals:
                        a_vals[m].append(fm[m])

            if len(a_vals["accuracy"]) < 2:
                continue

            n = min(len(fa_vals["accuracy"]), len(a_vals["accuracy"]))
            entry = {}
            for m in ["accuracy", "f1", "auc"]:
                _, p = stats.ttest_rel(fa_vals[m][:n], a_vals[m][:n])
                entry[f"{m}_p"] = round(float(p), 4)
                entry[f"{m}_sig"] = p < 0.05
            sig[ds][algo] = entry

    return sig


# ======================================================================
# New Tables: Fairness, Non-IID, Communication
# ======================================================================

def generate_fairness_table(p12: Dict) -> str:
    """Generate LaTeX table for inter-hospital fairness metrics."""
    completed = p12.get("completed", {})
    ok = {k: v for k, v in completed.items() if "error" not in v}

    lines = []
    lines.append(r"\begin{table*}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Inter-Hospital Fairness: Per-Client Accuracy Distribution}")
    lines.append(r"\label{tab:fairness}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{l" + "cccc" * len(ALL_DATASETS) + "}")
    lines.append(r"\toprule")

    header1 = r"\textbf{Algorithm}"
    for ds in ALL_DATASETS:
        header1 += r" & \multicolumn{4}{c}{\textbf{" + DS_SHORTS[ds] + "}}"
    lines.append(header1 + r" \\")

    header2 = ""
    for _ in ALL_DATASETS:
        header2 += r" & Mean & Std & Min & Jain"
    lines.append(header2 + r" \\")
    lines.append(r"\midrule")

    for algo in ALGORITHMS:
        row = algo
        for ds in ALL_DATASETS:
            means, stds, mins, jains = [], [], [], []
            for seed in SEEDS:
                rec = ok.get(f"{ds}_{algo}_{seed}", {})
                fair = rec.get("fairness", {})
                if fair:
                    means.append(fair.get("mean", 0))
                    stds.append(fair.get("std", 0))
                    mins.append(fair.get("min", 0))
                    jains.append(fair.get("jain_index", 0))
            if means:
                row += f" & {np.mean(means)*100:.1f}"
                row += f" & {np.mean(stds)*100:.1f}"
                row += f" & {np.mean(mins)*100:.1f}"
                row += f" & {np.mean(jains):.3f}"
            else:
                row += " & -- & -- & -- & --"
        lines.append(row + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\vspace{1mm}")
    lines.append(r"\footnotesize{Per-client accuracy (global model evaluated on each hospital's test set). "
                 r"Jain's Fairness Index: 1.0 = perfect equality. Mean over 3 seeds.}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


def generate_noniid_table(p14: Dict) -> str:
    """Generate LaTeX table for non-IID severity study."""
    completed = p14.get("completed", {})

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Non-IID Severity: Accuracy vs. Dirichlet $\alpha$ (Chest X-ray)}")
    lines.append(r"\label{tab:noniid}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{l" + "cc" * len(NONIID_ALGOS) + "}")
    lines.append(r"\toprule")

    header = r"\textbf{$\alpha$}"
    for algo in NONIID_ALGOS:
        header += r" & \multicolumn{2}{c}{\textbf{" + algo + "}}"
    lines.append(header + r" \\")

    subheader = ""
    for _ in NONIID_ALGOS:
        subheader += r" & Acc(\%) & Jain"
    lines.append(subheader + r" \\")
    lines.append(r"\midrule")

    for alpha in NONIID_ALPHAS:
        row = f"${alpha}$"
        for algo in NONIID_ALGOS:
            accs, jains = [], []
            for seed in SEEDS:
                rec = completed.get(f"alpha_{alpha}_{algo}_{seed}", {})
                fm = rec.get("final_metrics")
                fair = rec.get("fairness", {})
                if fm:
                    accs.append(fm["accuracy"])
                if fair:
                    jains.append(fair.get("jain_index", 0))
            if accs:
                row += f" & {np.mean(accs)*100:.1f}$\\pm${np.std(accs)*100:.1f}"
            else:
                row += " & --"
            if jains:
                row += f" & {np.mean(jains):.3f}"
            else:
                row += " & --"
        lines.append(row + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\vspace{1mm}")
    lines.append(r"\footnotesize{Dirichlet $\\alpha$: lower = more heterogeneous. "
                 r"5 clients, 30 rounds, ResNet18. Mean$\\pm$std over 3 seeds.}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def generate_communication_table(comm: Dict) -> str:
    """Generate LaTeX table for communication costs."""
    summary = comm.get("per_dataset", {})
    if not summary:
        return ""

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Communication Cost Analysis per Dataset}")
    lines.append(r"\label{tab:communication}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{llrrrr}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Dataset} & \textbf{Model} & \textbf{Params} & "
                 r"\textbf{MB/upd} & \textbf{Full (GB)} & \textbf{Top-K 1\% (GB)} \\")
    lines.append(r"\midrule")

    for ds, info in summary.items():
        short = DS_SHORTS.get(ds, ds)
        params_str = f"{info['model_params']/1e6:.1f}M" if info['model_params'] > 100_000 else f"{info['model_params']/1e3:.0f}K"
        lines.append(
            f"{short} & {info['model_type']} & {params_str} & "
            f"{info['mb_per_update']:.1f} & {info['total_gb']:.2f} & {info['topk_001_total_gb']:.4f} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\vspace{1mm}")
    lines.append(r"\footnotesize{5 clients, 30 rounds (imaging) / 20 rounds (tabular). "
                 r"Full = uncompressed float32. Top-K 1\\% = theoretical gradient compression.}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ======================================================================
# New Figures: Fairness, Non-IID, Communication
# ======================================================================

def _fig_fairness(completed, out_dir, plt):
    """Bar chart of Jain's fairness index per algo x dataset."""
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(ALL_DATASETS))
    width = 0.15

    has_data = False
    for i, algo in enumerate(ALGORITHMS):
        vals = []
        for ds in ALL_DATASETS:
            jains = []
            for seed in SEEDS:
                rec = completed.get(f"{ds}_{algo}_{seed}", {})
                fair = rec.get("fairness", {})
                if fair and "jain_index" in fair:
                    jains.append(fair["jain_index"])
            vals.append(np.mean(jains) if jains else 0)
        if any(v > 0 for v in vals):
            has_data = True
        ax.bar(x + i * width, vals, width, label=algo,
               color=ALGO_COLORS[i], alpha=0.85)

    if not has_data:
        plt.close(fig)
        return

    ds_labels = [DS_SHORTS[d] for d in ALL_DATASETS]
    ax.set_ylabel("Jain's Fairness Index")
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(ds_labels)
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0.5, 1.05)
    ax.set_title("Inter-Hospital Fairness Comparison")

    for fmt in ["pdf", "png"]:
        fig.savefig(out_dir / f"fig_fairness.{fmt}", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"    fig_fairness.pdf")


def _fig_noniid_alpha(completed, out_dir, plt):
    """Line plot: alpha vs accuracy, one line per algorithm."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    has_data = False
    for i, algo in enumerate(NONIID_ALGOS):
        means, stds, jain_means = [], [], []
        for alpha in NONIID_ALPHAS:
            accs, jains = [], []
            for seed in SEEDS:
                rec = completed.get(f"alpha_{alpha}_{algo}_{seed}", {})
                fm = rec.get("final_metrics")
                fair = rec.get("fairness", {})
                if fm:
                    accs.append(fm["accuracy"] * 100)
                if fair:
                    jains.append(fair.get("jain_index", 0))
            means.append(np.mean(accs) if accs else 0)
            stds.append(np.std(accs) if accs else 0)
            jain_means.append(np.mean(jains) if jains else 0)

        if any(m > 0 for m in means):
            has_data = True

        ax1.errorbar(NONIID_ALPHAS, means, yerr=stds, marker="o",
                     color=ALGO_COLORS[i], label=algo, capsize=4, linewidth=2)
        ax2.plot(NONIID_ALPHAS, jain_means, marker="s",
                 color=ALGO_COLORS[i], label=algo, linewidth=2)

    if not has_data:
        plt.close(fig)
        return

    ax1.set_xlabel(r"Dirichlet $\alpha$")
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_title("Accuracy vs Non-IID Severity")
    ax1.set_xscale("log")
    ax1.set_xticks(NONIID_ALPHAS)
    ax1.set_xticklabels([str(a) for a in NONIID_ALPHAS])
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.set_xlabel(r"Dirichlet $\alpha$")
    ax2.set_ylabel("Jain's Fairness Index")
    ax2.set_title("Fairness vs Non-IID Severity")
    ax2.set_xscale("log")
    ax2.set_xticks(NONIID_ALPHAS)
    ax2.set_xticklabels([str(a) for a in NONIID_ALPHAS])
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    for fmt in ["pdf", "png"]:
        fig.savefig(out_dir / f"fig_noniid_alpha.{fmt}", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"    fig_noniid_alpha.pdf")


def _fig_communication(summary, out_dir, plt):
    """Bar chart: Full vs Top-K communication costs."""
    if not summary:
        return

    datasets = list(summary.keys())
    labels = [DS_SHORTS.get(d, d) for d in datasets]
    full_gb = [summary[d]["total_gb"] for d in datasets]
    topk1 = [summary[d]["topk_001_total_gb"] for d in datasets]
    topk10 = [summary[d]["topk_01_total_gb"] for d in datasets]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(datasets))
    width = 0.25

    ax.bar(x - width, full_gb, width, label="Full (float32)", color="#F44336", alpha=0.85)
    ax.bar(x, topk10, width, label="Top-K 10%", color="#FF9800", alpha=0.85)
    ax.bar(x + width, topk1, width, label="Top-K 1%", color="#4CAF50", alpha=0.85)

    ax.set_ylabel("Total Communication (GB)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("Communication Cost: Full vs. Gradient Compression")

    for fmt in ["pdf", "png"]:
        fig.savefig(out_dir / f"fig_communication.{fmt}", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"    fig_communication.pdf")


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Incremental Paper Output Generator (FLICS 2026)")
    parser.add_argument("--status", action="store_true",
                        help="Show status only (no file generation)")
    parser.add_argument("--tables", action="store_true",
                        help="Generate only tables")
    parser.add_argument("--figures", action="store_true",
                        help="Generate only figures")
    args = parser.parse_args()

    checkpoints = load_all_checkpoints()

    # Always show status
    print_status(checkpoints)

    if args.status:
        return

    # Generate outputs
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    gen_tables = not args.figures  # tables unless --figures only
    gen_figures = not args.tables  # figures unless --tables only

    p12 = checkpoints["p12"]
    p13 = checkpoints["p13"]
    p14 = checkpoints["p14"]
    p22 = checkpoints["p22"]
    comm = checkpoints["comm"]

    # Recompute significance from P1.2 data
    sig = {}
    if p12:
        try:
            sig = compute_significance(p12)
        except Exception:
            sig = checkpoints.get("p21") or {}

    if gen_tables:
        print("\n  Generating tables...")
        if p12:
            tex = generate_multi_dataset_table(p12, sig)
            (OUTPUT_DIR / "table_multi_dataset.tex").write_text(tex)
            print(f"    table_multi_dataset.tex")

            # Fairness table (uses P1.2 per-client data)
            tex = generate_fairness_table(p12)
            (OUTPUT_DIR / "table_fairness.tex").write_text(tex)
            print(f"    table_fairness.tex")

        if p13:
            tex = generate_ablation_table(p13)
            (OUTPUT_DIR / "table_ablation.tex").write_text(tex)
            print(f"    table_ablation.tex")

        if p14:
            tex = generate_noniid_table(p14)
            (OUTPUT_DIR / "table_noniid.tex").write_text(tex)
            print(f"    table_noniid.tex")

        if p22:
            tex = generate_attack_table(p22)
            (OUTPUT_DIR / "table_attack.tex").write_text(tex)
            print(f"    table_attack.tex")

        if comm:
            tex = generate_communication_table(comm)
            if tex:
                (OUTPUT_DIR / "table_communication.tex").write_text(tex)
                print(f"    table_communication.tex")

    if gen_figures:
        print("\n  Generating figures...")
        try:
            n = generate_figures(p12 or {}, p13 or {}, p22 or {}, OUTPUT_DIR)
            if n == 0:
                print("    (no data available for figures)")

            # New figures
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            plt.rcParams.update({
                "font.size": 11, "font.family": "serif",
                "axes.labelsize": 12, "axes.titlesize": 13,
                "xtick.labelsize": 10, "ytick.labelsize": 10,
                "legend.fontsize": 9, "figure.dpi": 150,
            })

            if p12 and p12.get("completed"):
                ok12 = {k: v for k, v in p12["completed"].items() if "error" not in v}
                if ok12:
                    _fig_fairness(ok12, OUTPUT_DIR, plt)

            if p14 and p14.get("completed"):
                _fig_noniid_alpha(p14["completed"], OUTPUT_DIR, plt)

            if comm:
                _fig_communication(comm.get("per_dataset", {}), OUTPUT_DIR, plt)

        except Exception as e:
            print(f"    Error: {e}")

    print(f"\n  Output directory: {OUTPUT_DIR}")
    print()


if __name__ == "__main__":
    main()

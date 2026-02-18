#!/usr/bin/env python3
"""
FL-EHDS Tabular Results Analysis — Generate paper-ready outputs.

Reads checkpoint_tabular.json and produces:
  1. LaTeX comparison table (accuracy ± std, fairness)
  2. Convergence plots (per-dataset, all algorithms)
  3. Statistical significance tests (Wilcoxon signed-rank)
  4. Communication cost analysis (rounds to converge × model size)
  5. Per-dataset summary with best algorithm highlight

Usage:
    cd fl-ehds-framework
    python -m benchmarks.analyze_tabular_results

Output: benchmarks/paper_results_tabular/analysis/
        ├── table_tabular_comparison.tex
        ├── table_tabular_significance.tex
        ├── table_tabular_comm_cost.tex
        ├── convergence_PTB_XL.pdf
        ├── convergence_Cardiovascular.pdf
        ├── convergence_Breast_Cancer.pdf
        ├── convergence_all.pdf
        └── summary.txt

Author: Fabio Liberti
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
from itertools import combinations

import numpy as np

FRAMEWORK_DIR = Path(__file__).parent.parent
CHECKPOINT_PATH = FRAMEWORK_DIR / "benchmarks" / "paper_results_tabular" / "checkpoint_tabular.json"
OUTPUT_DIR = FRAMEWORK_DIR / "benchmarks" / "paper_results_tabular" / "analysis"

DATASETS = ["PTB_XL", "Cardiovascular", "Breast_Cancer"]
DATASET_LABELS = {
    "PTB_XL": "PTB-XL (5-class ECG)",
    "Cardiovascular": "Cardiovascular (binary)",
    "Breast_Cancer": "Breast Cancer (binary)",
}
DATASET_SHORT = {"PTB_XL": "PX", "Cardiovascular": "CV", "Breast_Cancer": "BC"}

ALGORITHMS = ["FedAvg", "FedProx", "Ditto", "FedLC", "FedExP", "FedLESAM", "HPFL"]
ALGO_SHORT = {
    "FedAvg": "FAvg", "FedProx": "FPrx", "Ditto": "Dit",
    "FedLC": "FLC", "FedExP": "FExP", "FedLESAM": "FLSM", "HPFL": "HPFL",
}

# MLP model size (approximate)
MLP_PARAMS = 10000  # ~10K parameters
PARAM_BYTES = 4     # float32
MODEL_SIZE_KB = MLP_PARAMS * PARAM_BYTES / 1024  # ~39 KB


def load_results():
    if not CHECKPOINT_PATH.exists():
        print(f"ERROR: Checkpoint not found at {CHECKPOINT_PATH}")
        print(f"Run experiments first: python -m benchmarks.run_tabular_optimized")
        sys.exit(1)

    with open(CHECKPOINT_PATH) as f:
        data = json.load(f)

    completed = data.get("completed", {})
    print(f"Loaded {len(completed)} experiments from checkpoint")
    return completed, data.get("metadata", {})


def organize_results(completed):
    """Organize results into structured dict: results[dataset][algorithm] = [result_per_seed]"""
    results = defaultdict(lambda: defaultdict(list))
    errors = []

    for key, res in completed.items():
        if "error" in res:
            errors.append(key)
            continue
        ds = res["dataset"]
        algo = res["algorithm"]
        results[ds][algo].append(res)

    if errors:
        print(f"WARNING: {len(errors)} experiments with errors: {errors}")

    return dict(results)


# ======================================================================
# 1. LaTeX Comparison Table
# ======================================================================

def generate_comparison_table(results):
    """Generate LaTeX table: rows=algorithms, columns=datasets (acc±std, Jain)."""
    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Federated learning algorithm comparison on new tabular datasets. "
                 r"Accuracy (\%) and Jain fairness index reported as mean $\pm$ std over 3 seeds. "
                 r"\textbf{Bold}: best per dataset.}")
    lines.append(r"\label{tab:tabular_comparison}")
    lines.append(r"\small")

    # 3 datasets × 2 metrics (acc, jain) = 6 data columns + algorithm column
    lines.append(r"\begin{tabular}{l" + "cc" * len(DATASETS) + "}")
    lines.append(r"\toprule")

    # Header row 1: dataset names spanning 2 columns
    header1 = r"\multirow{2}{*}{\textbf{Algorithm}}"
    for ds in DATASETS:
        label = DATASET_SHORT[ds]
        header1 += rf" & \multicolumn{{2}}{{c}}{{\textbf{{{label}}}}}"
    header1 += r" \\"
    lines.append(header1)

    # Header row 2: Acc / Jain for each dataset
    header2 = ""
    for ds in DATASETS:
        header2 += r" & Acc (\%) & Jain"
    header2 += r" \\"
    lines.append(r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}")
    lines.append(header2)
    lines.append(r"\midrule")

    # Find best per dataset
    best_acc = {}
    best_jain = {}
    for ds in DATASETS:
        best_a = 0
        best_j = 0
        for algo in ALGORITHMS:
            if algo in results.get(ds, {}):
                runs = results[ds][algo]
                accs = [r["best_metrics"]["accuracy"] for r in runs]
                jains = [r["fairness"]["jain_index"] for r in runs]
                mean_acc = np.mean(accs)
                mean_jain = np.mean(jains)
                if mean_acc > best_a:
                    best_a = mean_acc
                if mean_jain > best_j:
                    best_j = mean_jain
        best_acc[ds] = best_a
        best_jain[ds] = best_j

    # Data rows
    for algo in ALGORITHMS:
        row = f"{algo}"
        for ds in DATASETS:
            if algo in results.get(ds, {}):
                runs = results[ds][algo]
                accs = [r["best_metrics"]["accuracy"] for r in runs]
                jains = [r["fairness"]["jain_index"] for r in runs]
                mean_acc = np.mean(accs) * 100
                std_acc = np.std(accs) * 100
                mean_jain = np.mean(jains)
                std_jain = np.std(jains)

                acc_str = f"{mean_acc:.1f}$\\pm${std_acc:.1f}"
                jain_str = f"{mean_jain:.3f}"

                if abs(np.mean(accs) - best_acc[ds]) < 0.001:
                    acc_str = r"\textbf{" + acc_str + "}"
                if abs(mean_jain - best_jain[ds]) < 0.001:
                    jain_str = r"\textbf{" + jain_str + "}"

                row += f" & {acc_str} & {jain_str}"
            else:
                row += " & -- & --"
        row += r" \\"
        lines.append(row)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")

    return "\n".join(lines)


# ======================================================================
# 2. Convergence Plots
# ======================================================================

def generate_convergence_plots(results):
    """Generate convergence curves per dataset."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("WARNING: matplotlib not available, skipping convergence plots")
        return

    ALGO_COLORS = {
        "FedAvg": "#2196F3", "FedProx": "#FF5722", "Ditto": "#4CAF50",
        "FedLC": "#E91E63", "FedExP": "#9C27B0", "FedLESAM": "#FF9800",
        "HPFL": "#00BCD4",
    }

    for ds in DATASETS:
        if ds not in results:
            continue

        fig, ax = plt.subplots(1, 1, figsize=(8, 5))

        for algo in ALGORITHMS:
            if algo not in results[ds]:
                continue

            runs = results[ds][algo]
            # Collect histories, pad to same length with NaN
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
                          alpha=0.15, color=color)

        ax.set_xlabel("Communication Round", fontsize=12)
        ax.set_ylabel("Global Accuracy (%)", fontsize=12)
        ax.set_title(f"FL Convergence — {DATASET_LABELS[ds]}", fontsize=13)
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(1, None)

        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / f"convergence_{ds}.pdf", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved convergence_{ds}.pdf")

    # Combined plot (3 subplots)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for idx, ds in enumerate(DATASETS):
        if ds not in results:
            continue
        ax = axes[idx]

        for algo in ALGORITHMS:
            if algo not in results[ds]:
                continue
            runs = results[ds][algo]
            max_rounds = max(len(r["history"]) for r in runs)
            acc_matrix = np.full((len(runs), max_rounds), np.nan)
            for i, r in enumerate(runs):
                hist_acc = [h["accuracy"] for h in r["history"]]
                acc_matrix[i, :len(hist_acc)] = hist_acc

            rounds = np.arange(1, max_rounds + 1)
            mean_acc = np.nanmean(acc_matrix, axis=0) * 100

            color = ALGO_COLORS.get(algo, "#666666")
            ax.plot(rounds, mean_acc, color=color, linewidth=1.5, label=algo)

        ax.set_xlabel("Round", fontsize=10)
        ax.set_ylabel("Accuracy (%)" if idx == 0 else "", fontsize=10)
        ax.set_title(DATASET_SHORT[ds], fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
        if idx == 2:
            ax.legend(fontsize=8, loc="lower right")

    fig.suptitle("FL Algorithm Convergence — Tabular Datasets", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "convergence_all.pdf", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved convergence_all.pdf")


# ======================================================================
# 3. Statistical Significance Tests
# ======================================================================

def generate_significance_tests(results):
    """Wilcoxon signed-rank test between all algorithm pairs per dataset."""
    from scipy import stats

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Statistical significance (Wilcoxon signed-rank p-values) "
                 r"between algorithm pairs. $\dagger$: $p < 0.05$.}")
    lines.append(r"\label{tab:significance}")
    lines.append(r"\small")

    all_tests = {}
    for ds in DATASETS:
        if ds not in results:
            continue
        ds_tests = {}
        for a1, a2 in combinations(ALGORITHMS, 2):
            if a1 not in results[ds] or a2 not in results[ds]:
                continue
            accs1 = sorted([r["best_metrics"]["accuracy"] for r in results[ds][a1]])
            accs2 = sorted([r["best_metrics"]["accuracy"] for r in results[ds][a2]])
            if len(accs1) >= 3 and len(accs2) >= 3:
                try:
                    stat, p = stats.wilcoxon(accs1, accs2)
                    ds_tests[(a1, a2)] = p
                except ValueError:
                    ds_tests[(a1, a2)] = 1.0  # identical distributions
        all_tests[ds] = ds_tests

    # Compact format: show only comparisons vs FedAvg (baseline)
    baseline = "FedAvg"
    ncols = len(DATASETS)
    lines.append(r"\begin{tabular}{l" + "r" * ncols + "}")
    lines.append(r"\toprule")
    header = r"\textbf{vs FedAvg}"
    for ds in DATASETS:
        header += rf" & \textbf{{{DATASET_SHORT[ds]}}}"
    header += r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    for algo in ALGORITHMS:
        if algo == baseline:
            continue
        row = algo
        for ds in DATASETS:
            pair = (baseline, algo) if (baseline, algo) in all_tests.get(ds, {}) else (algo, baseline)
            p = all_tests.get(ds, {}).get(pair, None)
            if p is not None:
                sig = r"$\dagger$" if p < 0.05 else ""
                row += f" & {p:.3f}{sig}"
            else:
                row += " & --"
        row += r" \\"
        lines.append(row)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


# ======================================================================
# 4. Communication Cost Analysis
# ======================================================================

def generate_comm_cost_table(results):
    """Communication cost = rounds to best × model upload size."""
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Communication cost: rounds to best accuracy and total data transferred "
                 rf"(model size $\approx$ {MODEL_SIZE_KB:.0f}\,KB).}}")
    lines.append(r"\label{tab:comm_cost}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{l" + "rr" * len(DATASETS) + "}")
    lines.append(r"\toprule")

    header1 = r"\textbf{Algorithm}"
    for ds in DATASETS:
        header1 += rf" & \multicolumn{{2}}{{c}}{{\textbf{{{DATASET_SHORT[ds]}}}}}"
    header1 += r" \\"
    lines.append(header1)

    header2 = ""
    for _ in DATASETS:
        header2 += r" & Rounds & KB"
    header2 += r" \\"
    lines.append(r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}")
    lines.append(header2)
    lines.append(r"\midrule")

    for algo in ALGORITHMS:
        row = algo
        for ds in DATASETS:
            if algo in results.get(ds, {}):
                runs = results[ds][algo]
                avg_best_round = np.mean([r["best_round"] for r in runs])
                # Total comm = best_round × 2 (upload + download) × num_clients × model_size
                num_clients = runs[0].get("dataset_metadata", {}).get("samples_per_client")
                nc = len(runs[0].get("per_client_acc", {})) or 5
                total_kb = avg_best_round * 2 * nc * MODEL_SIZE_KB
                row += f" & {avg_best_round:.0f} & {total_kb:.0f}"
            else:
                row += " & -- & --"
        row += r" \\"
        lines.append(row)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


# ======================================================================
# 5. Text Summary
# ======================================================================

def generate_summary(results):
    """Generate readable text summary."""
    lines = []
    lines.append("=" * 70)
    lines.append("  FL-EHDS Tabular Experiment Summary")
    lines.append("=" * 70)

    for ds in DATASETS:
        if ds not in results:
            continue
        lines.append(f"\n--- {DATASET_LABELS[ds]} ---")

        best_algo = None
        best_acc = 0

        for algo in ALGORITHMS:
            if algo not in results[ds]:
                continue
            runs = results[ds][algo]
            accs = [r["best_metrics"]["accuracy"] for r in runs]
            jains = [r["fairness"]["jain_index"] for r in runs]
            runtimes = [r["runtime_seconds"] for r in runs]
            rounds = [r["actual_rounds"] for r in runs]

            mean_acc = np.mean(accs)
            if mean_acc > best_acc:
                best_acc = mean_acc
                best_algo = algo

            lines.append(
                f"  {algo:<10} acc={mean_acc:.1%}±{np.std(accs):.1%}  "
                f"jain={np.mean(jains):.3f}  "
                f"rounds={np.mean(rounds):.0f}  "
                f"time={np.mean(runtimes):.0f}s"
            )

        lines.append(f"  >>> Best: {best_algo} ({best_acc:.1%})")

    lines.append("\n" + "=" * 70)
    return "\n".join(lines)


# ======================================================================
# Main
# ======================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading results...")
    completed, metadata = load_results()
    results = organize_results(completed)

    print(f"\nDatasets found: {list(results.keys())}")
    for ds in results:
        algos = list(results[ds].keys())
        print(f"  {ds}: {len(algos)} algorithms, {sum(len(v) for v in results[ds].values())} runs")

    # 1. LaTeX comparison table
    print("\n1. Generating comparison table...")
    tex = generate_comparison_table(results)
    (OUTPUT_DIR / "table_tabular_comparison.tex").write_text(tex)
    print(f"  Saved table_tabular_comparison.tex")

    # 2. Convergence plots
    print("\n2. Generating convergence plots...")
    generate_convergence_plots(results)

    # 3. Significance tests
    print("\n3. Generating significance tests...")
    try:
        sig_tex = generate_significance_tests(results)
        (OUTPUT_DIR / "table_tabular_significance.tex").write_text(sig_tex)
        print(f"  Saved table_tabular_significance.tex")
    except ImportError:
        print("  WARNING: scipy not available, skipping significance tests")

    # 4. Communication costs
    print("\n4. Generating communication cost table...")
    comm_tex = generate_comm_cost_table(results)
    (OUTPUT_DIR / "table_tabular_comm_cost.tex").write_text(comm_tex)
    print(f"  Saved table_tabular_comm_cost.tex")

    # 5. Summary
    print("\n5. Generating summary...")
    summary = generate_summary(results)
    (OUTPUT_DIR / "summary.txt").write_text(summary)
    print(summary)

    print(f"\nAll outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

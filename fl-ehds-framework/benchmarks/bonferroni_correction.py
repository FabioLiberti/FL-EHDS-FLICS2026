#!/usr/bin/env python3
"""
Bonferroni & Holm-Bonferroni correction for FL-EHDS benchmark p-values.

Addresses reviewer concern: multiple comparisons (algorithms × datasets × metrics)
without correction inflate Type I error rate.

Loads all checkpoint results, recomputes pairwise statistical tests,
applies corrections, and generates updated LaTeX tables + summary report.

Usage:
    conda activate flics2026
    python bonferroni_correction.py [--checkpoint PATH] [--output-dir DIR]
"""

import json
import sys
import os
import argparse
from pathlib import Path
from itertools import combinations
from collections import defaultdict

import numpy as np
from scipy import stats


# ============================================================================
# Configuration
# ============================================================================

SCRIPT_DIR = Path(__file__).parent
DEFAULT_CHECKPOINT = SCRIPT_DIR / "paper_results" / "checkpoint_p12_multidataset.json"
DEFAULT_OUTPUT = SCRIPT_DIR / "paper_results" / "bonferroni_analysis"

METRICS = ["accuracy", "f1", "precision", "recall", "auc"]
ALPHA = 0.05
BASELINE = "FedAvg"


# ============================================================================
# Data Loading
# ============================================================================

def load_checkpoint(path: Path) -> dict:
    """Load checkpoint JSON file."""
    with open(path) as f:
        data = json.load(f)
    return data.get("completed", data)


def extract_results(completed: dict) -> dict:
    """Extract per-dataset, per-algorithm, per-metric seed arrays.

    Returns:
        {dataset: {algorithm: {metric: [seed_values]}}}
    """
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for key, exp in completed.items():
        ds = exp.get("dataset", "unknown")
        algo = exp.get("algorithm", "unknown")

        # Use best_metrics if available, else final_metrics
        metrics = exp.get("best_metrics", exp.get("final_metrics", {}))

        for metric_name, value in metrics.items():
            if metric_name in METRICS:
                results[ds][algo][metric_name].append(float(value))

    return dict(results)


# ============================================================================
# Statistical Tests
# ============================================================================

def compute_all_pairwise_tests(results: dict, baseline: str = BASELINE) -> list:
    """Compute all pairwise statistical tests across datasets and metrics.

    Returns list of dicts with: dataset, algo1, algo2, metric, test_type,
    statistic, p_value, effect_size, mean_diff.
    """
    all_tests = []

    for ds, algos in sorted(results.items()):
        if baseline not in algos:
            continue

        for algo in sorted(algos.keys()):
            if algo == baseline:
                continue

            for metric in METRICS:
                bl_vals = algos[baseline].get(metric, [])
                comp_vals = algos[algo].get(metric, [])

                if len(bl_vals) < 2 or len(comp_vals) < 2:
                    continue

                bl = np.array(bl_vals)
                comp = np.array(comp_vals)

                # Ensure same length (min seeds)
                n = min(len(bl), len(comp))
                bl, comp = bl[:n], comp[:n]

                # Paired t-test
                t_stat, p_ttest = stats.ttest_rel(comp, bl)

                # Wilcoxon (if n >= 3)
                p_wilcoxon = np.nan
                w_stat = np.nan
                if n >= 3:
                    try:
                        w_stat, p_wilcoxon = stats.wilcoxon(comp, bl)
                    except ValueError:
                        p_wilcoxon = 1.0

                # Cohen's d
                diff = comp - bl
                pooled_std = np.sqrt(
                    ((n - 1) * np.var(bl, ddof=1) + (n - 1) * np.var(comp, ddof=1))
                    / (2 * n - 2)
                )
                cohens_d = float((np.mean(comp) - np.mean(bl)) / pooled_std) if pooled_std > 1e-12 else 0.0

                all_tests.append({
                    "dataset": ds,
                    "algo_baseline": baseline,
                    "algo_comparison": algo,
                    "metric": metric,
                    "n_seeds": n,
                    "baseline_mean": float(np.mean(bl)),
                    "comparison_mean": float(np.mean(comp)),
                    "mean_diff": float(np.mean(comp) - np.mean(bl)),
                    "t_stat": float(t_stat),
                    "p_ttest": float(p_ttest),
                    "w_stat": float(w_stat) if not np.isnan(w_stat) else None,
                    "p_wilcoxon": float(p_wilcoxon) if not np.isnan(p_wilcoxon) else None,
                    "cohens_d": cohens_d,
                })

    return all_tests


# ============================================================================
# Multiple Comparison Corrections
# ============================================================================

def bonferroni_correction(p_values: list, alpha: float = ALPHA) -> dict:
    """Apply Bonferroni correction.

    Adjusted alpha = alpha / m, where m = number of comparisons.
    """
    m = len(p_values)
    adjusted_alpha = alpha / m
    adjusted_p = [min(p * m, 1.0) for p in p_values]

    return {
        "method": "Bonferroni",
        "m": m,
        "original_alpha": alpha,
        "adjusted_alpha": adjusted_alpha,
        "adjusted_p_values": adjusted_p,
        "n_significant_original": sum(1 for p in p_values if p < alpha),
        "n_significant_corrected": sum(1 for p in adjusted_p if p < alpha),
    }


def holm_bonferroni_correction(p_values: list, alpha: float = ALPHA) -> dict:
    """Apply Holm-Bonferroni (step-down) correction.

    Less conservative than Bonferroni, more powerful while controlling FWER.
    """
    m = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted_p = [0.0] * m

    for rank, (orig_idx, p) in enumerate(indexed):
        adjusted_p[orig_idx] = min(p * (m - rank), 1.0)

    # Enforce monotonicity
    sorted_indices = [idx for idx, _ in indexed]
    for i in range(1, m):
        adj_idx = sorted_indices[i]
        prev_idx = sorted_indices[i - 1]
        adjusted_p[adj_idx] = max(adjusted_p[adj_idx], adjusted_p[prev_idx])

    return {
        "method": "Holm-Bonferroni",
        "m": m,
        "original_alpha": alpha,
        "adjusted_p_values": adjusted_p,
        "n_significant_original": sum(1 for p in p_values if p < alpha),
        "n_significant_corrected": sum(1 for p in adjusted_p if p < alpha),
    }


def benjamini_hochberg_correction(p_values: list, alpha: float = ALPHA) -> dict:
    """Apply Benjamini-Hochberg (FDR) correction.

    Controls False Discovery Rate instead of FWER.
    More powerful than Bonferroni/Holm for many comparisons.
    """
    m = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted_p = [0.0] * m

    for rank_0based, (orig_idx, p) in enumerate(indexed):
        rank = rank_0based + 1
        adjusted_p[orig_idx] = min(p * m / rank, 1.0)

    # Enforce monotonicity (from largest to smallest)
    sorted_indices = [idx for idx, _ in indexed]
    for i in range(m - 2, -1, -1):
        adj_idx = sorted_indices[i]
        next_idx = sorted_indices[i + 1]
        adjusted_p[adj_idx] = min(adjusted_p[adj_idx], adjusted_p[next_idx])

    return {
        "method": "Benjamini-Hochberg (FDR)",
        "m": m,
        "original_alpha": alpha,
        "adjusted_p_values": adjusted_p,
        "n_significant_original": sum(1 for p in p_values if p < alpha),
        "n_significant_corrected": sum(1 for p in adjusted_p if p < alpha),
    }


# ============================================================================
# Report Generation
# ============================================================================

def generate_text_report(all_tests: list, corrections: dict) -> str:
    """Generate human-readable report."""
    lines = []
    lines.append("=" * 80)
    lines.append("BONFERRONI & MULTIPLE COMPARISON CORRECTION ANALYSIS")
    lines.append("FL-EHDS-FLICS Framework - Statistical Robustness Report")
    lines.append("=" * 80)
    lines.append("")

    m = len(all_tests)
    lines.append(f"Total pairwise tests (vs {BASELINE}): {m}")

    # Count by dimension
    datasets = set(t["dataset"] for t in all_tests)
    algos = set(t["algo_comparison"] for t in all_tests)
    metrics_used = set(t["metric"] for t in all_tests)
    lines.append(f"  Datasets:   {len(datasets)} ({', '.join(sorted(datasets))})")
    lines.append(f"  Algorithms: {len(algos)} ({', '.join(sorted(algos))})")
    lines.append(f"  Metrics:    {len(metrics_used)} ({', '.join(sorted(metrics_used))})")
    lines.append(f"  Total:      {len(datasets)} x {len(algos)} x {len(metrics_used)} = {m} tests")
    lines.append("")

    # Correction comparison
    lines.append("-" * 80)
    lines.append("CORRECTION METHOD COMPARISON")
    lines.append("-" * 80)
    lines.append(f"{'Method':<30} {'Significant (original)':<25} {'Significant (corrected)':<25}")
    lines.append("-" * 80)

    for method, corr in corrections.items():
        n_orig = corr["n_significant_original"]
        n_corr = corr["n_significant_corrected"]
        pct_orig = n_orig / m * 100 if m > 0 else 0
        pct_corr = n_corr / m * 100 if m > 0 else 0
        lines.append(
            f"{corr['method']:<30} {n_orig:>3}/{m} ({pct_orig:5.1f}%)          "
            f"{n_corr:>3}/{m} ({pct_corr:5.1f}%)"
        )

    if "bonferroni" in corrections:
        adj_alpha = corrections["bonferroni"]["adjusted_alpha"]
        lines.append(f"\nBonferroni adjusted alpha: {ALPHA} / {m} = {adj_alpha:.6f}")

    lines.append("")

    # Detailed results per dataset
    lines.append("-" * 80)
    lines.append("DETAILED RESULTS BY DATASET (using Holm-Bonferroni, recommended)")
    lines.append("-" * 80)

    holm = corrections.get("holm", {})
    holm_p = holm.get("adjusted_p_values", [t["p_ttest"] for t in all_tests])

    for ds in sorted(datasets):
        lines.append(f"\n  Dataset: {ds}")
        lines.append(f"  {'Algorithm':<15} {'Metric':<12} {'Mean Diff':>10} {'p (orig)':>10} "
                      f"{'p (Holm)':>10} {'Cohen d':>8} {'Sig?':>6}")
        lines.append("  " + "-" * 75)

        for i, t in enumerate(all_tests):
            if t["dataset"] != ds:
                continue

            p_orig = t["p_ttest"]
            p_adj = holm_p[i]
            sig_orig = "*" if p_orig < ALPHA else ""
            sig_adj = "*" if p_adj < ALPHA else ""

            d_label = ""
            d = abs(t["cohens_d"])
            if d >= 0.8:
                d_label = "L"
            elif d >= 0.5:
                d_label = "M"
            elif d >= 0.2:
                d_label = "S"

            lines.append(
                f"  {t['algo_comparison']:<15} {t['metric']:<12} "
                f"{t['mean_diff']:>+10.4f} {p_orig:>10.4f}{sig_orig:1s} "
                f"{p_adj:>10.4f}{sig_adj:1s} {t['cohens_d']:>+7.3f}{d_label:1s} "
            )

    lines.append("")

    # Summary of what survives correction
    lines.append("-" * 80)
    lines.append("FINDINGS THAT SURVIVE HOLM-BONFERRONI CORRECTION (p_adj < 0.05)")
    lines.append("-" * 80)

    surviving = [(i, t) for i, t in enumerate(all_tests) if holm_p[i] < ALPHA]
    if surviving:
        for i, t in surviving:
            lines.append(
                f"  {t['dataset']:20s} | {t['algo_comparison']:15s} | "
                f"{t['metric']:12s} | p_adj={holm_p[i]:.4f} | d={t['cohens_d']:+.3f}"
            )
    else:
        lines.append("  (None — this is expected with small sample sizes like n=3 seeds)")
        lines.append("  -> Recommend increasing seeds to 5-10 for stronger statistical power")

    lines.append("")
    lines.append("-" * 80)
    lines.append("RECOMMENDATION FOR PAPER")
    lines.append("-" * 80)
    lines.append(f"  Total independent comparisons: {m}")
    lines.append(f"  Recommended correction: Holm-Bonferroni (less conservative than Bonferroni)")
    lines.append(f"  Alternative: Benjamini-Hochberg FDR (if reviewer accepts FDR control)")
    lines.append("")
    lines.append("  Suggested paper text:")
    lines.append('  "All pairwise comparisons against FedAvg baseline were subjected to')
    lines.append(f'   Holm-Bonferroni correction for {m} simultaneous tests')
    lines.append('   (7-8 algorithms × 5 datasets × 5 metrics). Effect sizes are reported')
    lines.append('   as Cohen\'s d. Results with |d| ≥ 0.8 (large effect) are considered')
    lines.append('   practically significant regardless of p-value adjustment."')

    return "\n".join(lines)


def generate_latex_table(all_tests: list, corrections: dict) -> str:
    """Generate LaTeX table with corrected significance markers."""
    holm = corrections.get("holm", {})
    holm_p = holm.get("adjusted_p_values", [t["p_ttest"] for t in all_tests])

    bh = corrections.get("benjamini_hochberg", {})
    bh_p = bh.get("adjusted_p_values", [t["p_ttest"] for t in all_tests])

    datasets = sorted(set(t["dataset"] for t in all_tests))

    lines = []
    lines.append(r"\begin{table*}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Pairwise significance tests vs.\ FedAvg with multiple comparison "
                 r"correction (accuracy metric). Holm--Bonferroni corrected p-values reported.}")
    lines.append(r"\label{tab:bonferroni_significance}")
    lines.append(r"\small")

    ncols = len(datasets)
    lines.append(r"\begin{tabular}{l" + "rrr" * ncols + "}")
    lines.append(r"\toprule")

    # Header row 1: dataset names
    header = r"\textbf{Algorithm}"
    for ds in datasets:
        short = ds.replace("_", " ")
        header += rf" & \multicolumn{{3}}{{c}}{{\textbf{{{short}}}}}"
    header += r" \\"
    lines.append(header)

    # Header row 2: sub-columns
    subheader = ""
    for _ in datasets:
        subheader += r" & $p_{\text{orig}}$ & $p_{\text{Holm}}$ & $d$"
    subheader += r" \\"
    lines.append(subheader)
    lines.append(r"\midrule")

    # Only accuracy metric for main table
    algos = sorted(set(t["algo_comparison"] for t in all_tests))
    for algo in algos:
        row = algo
        for ds in datasets:
            # Find this test
            match = [
                (i, t) for i, t in enumerate(all_tests)
                if t["dataset"] == ds and t["algo_comparison"] == algo and t["metric"] == "accuracy"
            ]
            if match:
                i, t = match[0]
                p_orig = t["p_ttest"]
                p_holm = holm_p[i]
                d = t["cohens_d"]

                # Significance markers (on corrected p-value)
                marker = ""
                if p_holm < 0.001:
                    marker = r"$^{***}$"
                elif p_holm < 0.01:
                    marker = r"$^{**}$"
                elif p_holm < 0.05:
                    marker = r"$^{*}$"

                row += f" & {p_orig:.3f} & {p_holm:.3f}{marker} & {d:+.2f}"
            else:
                row += r" & -- & -- & --"
        row += r" \\"
        lines.append(row)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append("")
    lines.append(r"\vspace{1mm}")
    lines.append(r"\footnotesize{$^{*}p_{\text{adj}}<0.05$, $^{**}p_{\text{adj}}<0.01$, "
                 r"$^{***}p_{\text{adj}}<0.001$ (Holm--Bonferroni). "
                 r"$d$: Cohen's $d$ effect size (0.2=S, 0.5=M, 0.8=L).}")
    lines.append(r"\end{table*}")

    return "\n".join(lines)


def generate_json_output(all_tests: list, corrections: dict) -> dict:
    """Generate machine-readable JSON output for downstream scripts."""
    holm_p = corrections["holm"]["adjusted_p_values"]
    bonf_p = corrections["bonferroni"]["adjusted_p_values"]
    bh_p = corrections["benjamini_hochberg"]["adjusted_p_values"]

    output = []
    for i, t in enumerate(all_tests):
        output.append({
            **t,
            "p_bonferroni": bonf_p[i],
            "p_holm": holm_p[i],
            "p_bh_fdr": bh_p[i],
            "sig_original": t["p_ttest"] < ALPHA,
            "sig_bonferroni": bonf_p[i] < ALPHA,
            "sig_holm": holm_p[i] < ALPHA,
            "sig_bh_fdr": bh_p[i] < ALPHA,
        })

    return {
        "tests": output,
        "summary": {
            "total_tests": len(all_tests),
            "alpha": ALPHA,
            "corrections": {
                name: {
                    "n_significant_original": c["n_significant_original"],
                    "n_significant_corrected": c["n_significant_corrected"],
                }
                for name, c in corrections.items()
            },
        },
    }


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Bonferroni correction for FL-EHDS benchmarks")
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT),
                        help="Path to checkpoint JSON file")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT),
                        help="Output directory for results")
    parser.add_argument("--extra-checkpoints", nargs="*", default=[],
                        help="Additional checkpoint files to merge")
    args = parser.parse_args()

    print("=" * 60)
    print("Bonferroni Correction Analysis - FL-EHDS-FLICS")
    print("=" * 60)

    # Load data
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    print(f"\nLoading: {checkpoint_path}")
    completed = load_checkpoint(checkpoint_path)

    # Merge extra checkpoints
    for extra in args.extra_checkpoints:
        p = Path(extra)
        if p.exists():
            print(f"Merging: {p}")
            extra_completed = load_checkpoint(p)
            completed.update(extra_completed)

    # Extract structured results
    results = extract_results(completed)
    print(f"\nDatasets: {list(results.keys())}")
    for ds, algos in results.items():
        print(f"  {ds}: {list(algos.keys())}")

    # Compute all pairwise tests
    all_tests = compute_all_pairwise_tests(results)
    print(f"\nTotal pairwise tests computed: {len(all_tests)}")

    if not all_tests:
        print("No tests to correct. Check that checkpoint contains valid data.")
        sys.exit(1)

    # Extract p-values for correction
    p_values = [t["p_ttest"] for t in all_tests]

    # Apply corrections
    corrections = {
        "bonferroni": bonferroni_correction(p_values, ALPHA),
        "holm": holm_bonferroni_correction(p_values, ALPHA),
        "benjamini_hochberg": benjamini_hochberg_correction(p_values, ALPHA),
    }

    # Generate outputs
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Text report
    report = generate_text_report(all_tests, corrections)
    print("\n" + report)
    report_path = output_dir / "bonferroni_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\nReport saved: {report_path}")

    # LaTeX table
    latex = generate_latex_table(all_tests, corrections)
    latex_path = output_dir / "bonferroni_table.tex"
    with open(latex_path, "w") as f:
        f.write(latex)
    print(f"LaTeX table saved: {latex_path}")

    # JSON output
    json_data = generate_json_output(all_tests, corrections)
    json_path = output_dir / "bonferroni_results.json"
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"JSON results saved: {json_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()

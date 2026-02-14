"""
Statistical significance testing for FL benchmark comparisons.

Provides paired t-tests, confidence intervals, effect sizes (Cohen's d),
and LaTeX formatting with significance markers for paper tables.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy import stats


def paired_ttest(
    baseline: List[float],
    comparison: List[float],
    alpha: float = 0.05,
) -> Dict:
    """Paired t-test between baseline and comparison metric arrays.

    Args:
        baseline: Per-seed metric values for the baseline algorithm.
        comparison: Per-seed metric values for the comparison algorithm.
        alpha: Significance level (default 0.05).

    Returns:
        Dict with t_stat, p_value, significant, n, mean_diff, ci_diff.
    """
    baseline = np.array(baseline, dtype=float)
    comparison = np.array(comparison, dtype=float)
    n = len(baseline)

    if n < 2:
        return {
            "t_stat": 0.0,
            "p_value": 1.0,
            "significant": False,
            "n": n,
            "mean_diff": float(np.mean(comparison) - np.mean(baseline)),
            "ci_diff": (0.0, 0.0),
        }

    t_stat, p_value = stats.ttest_rel(comparison, baseline)

    diff = comparison - baseline
    mean_diff = np.mean(diff)
    se_diff = np.std(diff, ddof=1) / np.sqrt(n)
    t_crit = stats.t.ppf(1 - alpha / 2, df=n - 1)
    ci_diff = (mean_diff - t_crit * se_diff, mean_diff + t_crit * se_diff)

    return {
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "significant": bool(p_value < alpha),
        "n": n,
        "mean_diff": float(mean_diff),
        "ci_diff": (float(ci_diff[0]), float(ci_diff[1])),
    }


def confidence_interval(
    scores: List[float],
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """Compute confidence interval for a metric array.

    Args:
        scores: Per-seed metric values.
        confidence: Confidence level (default 0.95).

    Returns:
        (lower, upper) bounds of the confidence interval.
    """
    scores = np.array(scores, dtype=float)
    n = len(scores)
    if n < 2:
        m = float(np.mean(scores))
        return (m, m)

    mean = np.mean(scores)
    se = np.std(scores, ddof=1) / np.sqrt(n)
    alpha = 1 - confidence
    t_crit = stats.t.ppf(1 - alpha / 2, df=n - 1)

    return (float(mean - t_crit * se), float(mean + t_crit * se))


def cohens_d(group1: List[float], group2: List[float]) -> float:
    """Compute Cohen's d effect size between two groups.

    Uses pooled standard deviation (assumes similar variances).

    Args:
        group1: First group metric values.
        group2: Second group metric values.

    Returns:
        Cohen's d value. Conventions: 0.2=small, 0.5=medium, 0.8=large.
    """
    g1 = np.array(group1, dtype=float)
    g2 = np.array(group2, dtype=float)

    n1, n2 = len(g1), len(g2)
    var1, var2 = np.var(g1, ddof=1), np.var(g2, ddof=1)

    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std < 1e-12:
        return 0.0

    return float((np.mean(g2) - np.mean(g1)) / pooled_std)


def pairwise_significance_table(
    results_by_algo: Dict[str, Dict[str, List[float]]],
    baseline: str = "FedAvg",
    metrics: Optional[List[str]] = None,
    alpha: float = 0.05,
) -> Dict[str, Dict[str, Dict]]:
    """Compute pairwise significance tests for all algorithms vs baseline.

    Args:
        results_by_algo: {algo_name: {metric_name: [per_seed_values]}}.
            Example: {"FedAvg": {"accuracy": [0.9, 0.91], "f1": [0.88, 0.89]},
                       "FedProx": {"accuracy": [0.92, 0.93], "f1": [0.91, 0.90]}}
        baseline: Name of the baseline algorithm to compare against.
        metrics: List of metric names to test (default: all metrics in baseline).
        alpha: Significance level.

    Returns:
        Nested dict: {algo: {metric: {t_stat, p_value, significant, cohens_d, ...}}}
    """
    if baseline not in results_by_algo:
        raise ValueError(f"Baseline '{baseline}' not found in results")

    if metrics is None:
        metrics = list(results_by_algo[baseline].keys())

    table = {}
    baseline_data = results_by_algo[baseline]

    for algo, algo_data in results_by_algo.items():
        if algo == baseline:
            continue

        table[algo] = {}
        for metric in metrics:
            if metric not in baseline_data or metric not in algo_data:
                continue

            bl = baseline_data[metric]
            comp = algo_data[metric]

            test = paired_ttest(bl, comp, alpha=alpha)
            d = cohens_d(bl, comp)
            ci = confidence_interval(comp)

            table[algo][metric] = {
                **test,
                "cohens_d": d,
                "ci": ci,
                "baseline_mean": float(np.mean(bl)),
                "comparison_mean": float(np.mean(comp)),
            }

    return table


def format_significance_latex(
    results_by_algo: Dict[str, Dict[str, List[float]]],
    baseline: str = "FedAvg",
    metrics: Optional[List[str]] = None,
    caption: str = "Pairwise Significance Testing",
    label: str = "tab:significance",
) -> str:
    """Generate a LaTeX table with significance markers.

    Markers: * for p < 0.05, ** for p < 0.01, *** for p < 0.001.
    Bold for best value per metric. Arrow indicates improvement direction.

    Args:
        results_by_algo: {algo: {metric: [per_seed_values]}}.
        baseline: Baseline algorithm name.
        metrics: Metrics to include (default: all).
        caption: LaTeX caption.
        label: LaTeX label.

    Returns:
        LaTeX table string.
    """
    if metrics is None:
        metrics = list(results_by_algo[baseline].keys())

    sig_table = pairwise_significance_table(results_by_algo, baseline, metrics)

    # Header
    metric_headers = " & ".join(
        [f"\\textbf{{{m.replace('_', ' ').title()}}}" for m in metrics]
    )
    n_cols = len(metrics) + 1  # algo + metrics
    col_spec = "l" + "c" * len(metrics)

    table = f"""
\\begin{{table}}[htbp]
\\centering
\\caption{{{caption}}}
\\label{{{label}}}
\\small
\\begin{{tabular}}{{{col_spec}}}
\\toprule
\\textbf{{Algorithm}} & {metric_headers} \\\\
\\midrule
"""

    # Find best value per metric (higher is better for all standard metrics)
    best_per_metric = {}
    for metric in metrics:
        best_val = -float("inf")
        for algo, algo_data in results_by_algo.items():
            if metric in algo_data:
                mean_val = float(np.mean(algo_data[metric]))
                if mean_val > best_val:
                    best_val = mean_val
        best_per_metric[metric] = best_val

    # Baseline row
    cells = []
    for metric in metrics:
        vals = results_by_algo[baseline][metric]
        mean = float(np.mean(vals))
        std = float(np.std(vals))
        cell = f"{mean:.4f}$\\pm${std:.4f}"
        if abs(mean - best_per_metric[metric]) < 1e-6:
            cell = f"\\textbf{{{cell}}}"
        cells.append(cell)
    table += f"{baseline} (baseline) & {' & '.join(cells)} \\\\\n"

    # Other algorithms
    for algo in sorted(sig_table.keys()):
        cells = []
        for metric in metrics:
            vals = results_by_algo[algo][metric]
            mean = float(np.mean(vals))
            std = float(np.std(vals))

            sig_info = sig_table[algo].get(metric, {})
            p_val = sig_info.get("p_value", 1.0)

            # Significance marker
            marker = ""
            if p_val < 0.001:
                marker = "$^{***}$"
            elif p_val < 0.01:
                marker = "$^{**}$"
            elif p_val < 0.05:
                marker = "$^{*}$"

            cell = f"{mean:.4f}$\\pm${std:.4f}{marker}"
            if abs(mean - best_per_metric[metric]) < 1e-6:
                cell = f"\\textbf{{{cell}}}"
            cells.append(cell)
        table += f"{algo} & {' & '.join(cells)} \\\\\n"

    table += f"""\\bottomrule
\\end{{tabular}}

\\vspace{{1mm}}
\\footnotesize{{$^{{*}}p<0.05$, $^{{**}}p<0.01$, $^{{***}}p<0.001$ vs {baseline} (paired t-test).}}
\\end{{table}}
"""
    return table

"""
EHDS-Specific Benchmark: Privacy-Utility-Cost Tradeoff.

Generates EHDS-native metrics instead of classic FL benchmarks:
1. Privacy-Utility-Cost tradeoff surface (3D)
2. Compliance score (% of EHDS articles satisfied)
3. Time-to-compliance (rounds to acceptable model within privacy budget)

Author: Fabio Liberti
"""

import csv
import json
import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# =====================================================================
# DATA STRUCTURES
# =====================================================================

@dataclass
class EHDSBenchmarkConfig:
    """Configuration for EHDS benchmark sweep."""
    epsilons: List[float] = field(default_factory=lambda: [1.0, 5.0, 10.0, 50.0, 100.0])
    algorithms: List[str] = field(default_factory=lambda: ["FedAvg", "FedProx", "SCAFFOLD"])
    countries: List[str] = field(default_factory=lambda: ["DE", "FR", "IT"])
    hospitals_per_country: int = 1
    num_rounds: int = 10
    local_epochs: int = 3
    batch_size: int = 32
    learning_rate: float = 0.01
    purpose: str = "scientific_research"
    dataset_type: str = "synthetic"
    governance_configs: List[str] = field(default_factory=lambda: ["minimal", "full"])
    min_acceptable_accuracy: float = 0.55
    seeds: List[int] = field(default_factory=lambda: [42])


@dataclass
class EHDSBenchmarkRun:
    """Results from a single benchmark configuration run."""
    config_id: int = 0
    config_label: str = ""
    effective_epsilon: float = 0.0
    algorithm: str = ""
    governance_level: str = ""
    final_accuracy: float = 0.0
    final_f1: float = 0.0
    final_auc: float = 0.0
    final_loss: float = 0.0
    communication_cost: int = 0
    compliance_score: float = 0.0
    compliance_summary: Dict = field(default_factory=dict)
    rounds_to_compliance: Optional[int] = None
    per_round_history: List[Dict] = field(default_factory=list)


@dataclass
class EHDSBenchmarkResults:
    """Complete benchmark results."""
    config: EHDSBenchmarkConfig = field(default_factory=EHDSBenchmarkConfig)
    runs: List[EHDSBenchmarkRun] = field(default_factory=list)
    generated_at: str = ""
    total_time_seconds: float = 0.0


# =====================================================================
# CORE BENCHMARK FUNCTIONS
# =====================================================================

def run_ehds_benchmark(
    config: EHDSBenchmarkConfig,
    progress_callback: Optional[Callable] = None,
) -> EHDSBenchmarkResults:
    """Run full EHDS benchmark sweep.

    Iterates over all combinations of epsilon x algorithm x governance
    and collects metrics, compliance scores, and communication costs.

    Args:
        config: Benchmark configuration
        progress_callback: Called with (current, total, label, accuracy, compliance)
    """
    from terminal.cross_border import CrossBorderFederatedTrainer
    from governance.ehds_compliance_report import EHDSComplianceReport

    results = EHDSBenchmarkResults(
        config=config,
        generated_at=datetime.now().isoformat(timespec="seconds"),
    )

    # Build run grid
    grid = []
    for eps in config.epsilons:
        for algo in config.algorithms:
            for gov in config.governance_configs:
                for seed in config.seeds:
                    grid.append((eps, algo, gov, seed))

    total = len(grid)
    t0 = time.time()

    for idx, (eps, algo, gov, seed) in enumerate(grid):
        label = f"{algo}_eps{eps}_{gov}"
        logger.info(f"Benchmark run {idx+1}/{total}: {label} (seed={seed})")

        # Build trainer config based on governance level
        gov_kwargs = _governance_kwargs(gov)

        trainer_config = {
            "countries": config.countries,
            "hospitals_per_country": config.hospitals_per_country,
            "algorithm": algo,
            "num_rounds": config.num_rounds,
            "local_epochs": config.local_epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "global_epsilon": eps,
            "purpose": config.purpose,
            "dataset_type": config.dataset_type,
            "seed": seed,
        }

        try:
            trainer = CrossBorderFederatedTrainer(
                countries=config.countries,
                hospitals_per_country=config.hospitals_per_country,
                algorithm=algo,
                num_rounds=config.num_rounds,
                local_epochs=config.local_epochs,
                batch_size=config.batch_size,
                learning_rate=config.learning_rate,
                global_epsilon=eps,
                purpose=config.purpose,
                dataset_type=config.dataset_type,
                seed=seed,
                simulate_latency=True,
                **gov_kwargs,
            )

            train_results = trainer.train()

            # Compliance report
            report = EHDSComplianceReport()
            report_config = {**trainer_config, **gov_kwargs}
            report.generate_from_trainer(trainer, report_config)
            summary = report.get_summary()

            # Extract metrics
            history = train_results.get("history", [])
            last = history[-1] if history else None

            per_round = _extract_per_round_history(history)
            comm_cost = _extract_communication_cost(trainer, train_results, config)
            ttc = _compute_time_to_compliance(
                history, trainer.hospitals, config.min_acceptable_accuracy
            )

            # Effective epsilon (most restrictive across jurisdictions)
            eff_eps = train_results.get("effective_epsilon", eps)

            run = EHDSBenchmarkRun(
                config_id=idx,
                config_label=label,
                effective_epsilon=eff_eps,
                algorithm=algo,
                governance_level=gov,
                final_accuracy=last.global_acc if last else 0.0,
                final_f1=last.global_f1 if last else 0.0,
                final_auc=last.global_auc if last else 0.0,
                final_loss=last.global_loss if last else 0.0,
                communication_cost=comm_cost,
                compliance_score=summary["compliance_score_pct"],
                compliance_summary=summary,
                rounds_to_compliance=ttc,
                per_round_history=per_round,
            )
            results.runs.append(run)

            if progress_callback:
                progress_callback(
                    idx + 1, total, label,
                    run.final_accuracy, run.compliance_score,
                )

        except Exception as e:
            logger.error(f"Benchmark run {label} failed: {e}")
            if progress_callback:
                progress_callback(idx + 1, total, f"{label} (FAILED)", 0.0, 0.0)

    results.total_time_seconds = time.time() - t0
    return results


def _governance_kwargs(gov_level: str) -> Dict[str, Any]:
    """Build CrossBorderFederatedTrainer kwargs for governance level."""
    if gov_level == "full":
        return {
            "ihe_enabled": True,
            "ihe_config": {},
            "data_quality_enabled": True,
            "data_quality_config": {},
            "governance_lifecycle_enabled": True,
            "governance_config": {
                "data_minimization_enabled": False,
            },
            "secure_processing_enabled": True,
            "secure_processing_config": {
                "enclave_enabled": True,
                "watermark_enabled": True,
                "time_limit_enabled": True,
                "permit_duration_hours": 24.0,
            },
        }
    else:  # minimal
        return {
            "ihe_enabled": True,
            "ihe_config": {},
            "data_quality_enabled": False,
            "data_quality_config": {},
        }


def _compute_time_to_compliance(
    history: list,
    hospitals: list,
    threshold: float,
) -> Optional[int]:
    """Compute first round where accuracy >= threshold and all hospitals in budget.

    Returns:
        Round number (0-indexed) or None if never reached.
    """
    for i, rr in enumerate(history):
        acc = getattr(rr, "global_acc", 0.0)
        if acc < threshold:
            continue
        # Check all hospitals are within their epsilon budget
        all_in_budget = True
        for h in hospitals:
            if h.cumulative_epsilon_spent > h.effective_epsilon:
                all_in_budget = False
                break
        if all_in_budget:
            return i
    return None


def _extract_communication_cost(
    trainer,
    results: Dict,
    config: EHDSBenchmarkConfig,
) -> int:
    """Extract total communication cost in bytes.

    If MyHealth@EU bridge is active, uses actual recorded bytes.
    Otherwise estimates from model size * hospitals * rounds * 2 (up+down).
    """
    bridge = getattr(trainer, "myhealth_bridge", None)
    if bridge is not None:
        report = results.get("myhealth_eu_report", {})
        round_metrics = report.get("round_metrics", [])
        if round_metrics:
            return sum(
                rm.get("communication_cost_bytes", 0)
                for rm in round_metrics
            )

    # Estimate: model_params * 4 bytes * hospitals * rounds * 2
    model_size = 0
    if hasattr(trainer, "_trainer") and hasattr(trainer._trainer, "global_model"):
        model_size = sum(
            p.numel() * 4 for p in trainer._trainer.global_model.parameters()
        )
    else:
        model_size = 10 * 2 * 4  # default MLP 10->2

    n_hospitals = len(trainer.hospitals)
    return model_size * n_hospitals * config.num_rounds * 2


def _extract_per_round_history(history: list) -> List[Dict]:
    """Extract simplified per-round metrics."""
    return [
        {
            "round": getattr(rr, "round_num", i),
            "accuracy": round(getattr(rr, "global_acc", 0.0), 4),
            "loss": round(getattr(rr, "global_loss", 0.0), 4),
            "f1": round(getattr(rr, "global_f1", 0.0), 4),
            "auc": round(getattr(rr, "global_auc", 0.0), 4),
        }
        for i, rr in enumerate(history)
    ]


# =====================================================================
# PLOT FUNCTIONS
# =====================================================================

def _plot_privacy_utility_cost_3d(results: EHDSBenchmarkResults, path: str):
    """3D scatter: epsilon (privacy) vs accuracy (utility) vs comm cost."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    algo_colors = {}
    cmap = plt.cm.Set1
    algos = sorted(set(r.algorithm for r in results.runs))
    for i, algo in enumerate(algos):
        algo_colors[algo] = cmap(i / max(len(algos) - 1, 1))

    gov_markers = {"minimal": "o", "full": "^"}

    for run in results.runs:
        color = algo_colors[run.algorithm]
        marker = gov_markers.get(run.governance_level, "s")
        cost_mb = run.communication_cost / (1024 * 1024)
        ax.scatter(
            np.log10(max(run.effective_epsilon, 0.1)),
            run.final_accuracy,
            cost_mb,
            c=[color], marker=marker, s=80, alpha=0.8,
        )

    ax.set_xlabel("log10(epsilon)")
    ax.set_ylabel("Accuracy")
    ax.set_zlabel("Comm. Cost (MB)")
    ax.set_title("Privacy-Utility-Cost Tradeoff Surface")

    # Legend
    legend_elements = []
    from matplotlib.lines import Line2D
    for algo in algos:
        legend_elements.append(
            Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=algo_colors[algo], markersize=8, label=algo)
        )
    legend_elements.append(
        Line2D([0], [0], marker="o", color="grey", label="Minimal", linestyle="None")
    )
    legend_elements.append(
        Line2D([0], [0], marker="^", color="grey", label="Full", linestyle="None")
    )
    ax.legend(handles=legend_elements, loc="upper left", fontsize=8)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def _plot_privacy_utility_2d(results: EHDSBenchmarkResults, path: str):
    """2D projections: (a) epsilon vs accuracy, (b) epsilon vs compliance."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    algos = sorted(set(r.algorithm for r in results.runs))
    govs = sorted(set(r.governance_level for r in results.runs))
    cmap = plt.cm.Set1
    markers = {"minimal": "o", "full": "^"}

    for i, algo in enumerate(algos):
        color = cmap(i / max(len(algos) - 1, 1))
        for gov in govs:
            runs = [r for r in results.runs
                    if r.algorithm == algo and r.governance_level == gov]
            if not runs:
                continue
            # Aggregate by epsilon (mean across seeds)
            eps_acc = {}
            eps_comp = {}
            for r in runs:
                eps_acc.setdefault(r.effective_epsilon, []).append(r.final_accuracy)
                eps_comp.setdefault(r.effective_epsilon, []).append(r.compliance_score)

            epsilons = sorted(eps_acc.keys())
            accs = [np.mean(eps_acc[e]) for e in epsilons]
            comps = [np.mean(eps_comp[e]) for e in epsilons]
            marker = markers.get(gov, "s")
            label = f"{algo} ({gov})"

            ax1.plot(epsilons, accs, marker=marker, color=color, label=label, alpha=0.8)
            ax2.plot(epsilons, comps, marker=marker, color=color, label=label, alpha=0.8)

    ax1.set_xscale("log")
    ax1.set_xlabel("Epsilon (privacy budget)")
    ax1.set_ylabel("Accuracy")
    ax1.set_title("(a) Privacy vs Utility")
    ax1.legend(fontsize=7, loc="lower right")
    ax1.grid(True, alpha=0.3)

    ax2.set_xscale("log")
    ax2.set_xlabel("Epsilon (privacy budget)")
    ax2.set_ylabel("Compliance Score (%)")
    ax2.set_title("(b) Privacy vs Compliance")
    ax2.axhline(y=80, color="r", linestyle="--", alpha=0.5, label="80% threshold")
    ax2.legend(fontsize=7, loc="lower right")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def _plot_compliance_by_config(results: EHDSBenchmarkResults, path: str):
    """Grouped bar chart: compliance score by epsilon and algorithm."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    algos = sorted(set(r.algorithm for r in results.runs))
    epsilons = sorted(set(r.effective_epsilon for r in results.runs))
    govs = sorted(set(r.governance_level for r in results.runs))

    fig, ax = plt.subplots(figsize=(10, 6))

    n_groups = len(epsilons)
    n_bars = len(algos) * len(govs)
    bar_width = 0.8 / max(n_bars, 1)
    x = np.arange(n_groups)
    hatches = {"minimal": "", "full": "//"}

    cmap = plt.cm.Set1
    bar_idx = 0
    for i, algo in enumerate(algos):
        color = cmap(i / max(len(algos) - 1, 1))
        for gov in govs:
            scores = []
            for eps in epsilons:
                runs = [r for r in results.runs
                        if r.algorithm == algo and r.governance_level == gov
                        and abs(r.effective_epsilon - eps) < 0.01]
                score = np.mean([r.compliance_score for r in runs]) if runs else 0
                scores.append(score)

            offset = (bar_idx - n_bars / 2 + 0.5) * bar_width
            ax.bar(x + offset, scores, bar_width,
                   label=f"{algo} ({gov})", color=color, alpha=0.8,
                   hatch=hatches.get(gov, ""))
            bar_idx += 1

    ax.axhline(y=80, color="r", linestyle="--", alpha=0.6, label="80% threshold")
    ax.set_xlabel("Epsilon")
    ax.set_ylabel("Compliance Score (%)")
    ax.set_title("EHDS Compliance by Configuration")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{e:.0f}" if e >= 1 else f"{e}" for e in epsilons])
    ax.legend(fontsize=7, loc="upper left", ncol=2)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def _plot_time_to_compliance(results: EHDSBenchmarkResults, path: str):
    """Horizontal bar chart: time-to-compliance (rounds) by configuration."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Filter unique configs (pick first seed)
    seen = set()
    runs = []
    for r in results.runs:
        key = (r.algorithm, r.effective_epsilon, r.governance_level)
        if key not in seen:
            seen.add(key)
            runs.append(r)

    # Sort by time-to-compliance (None = DNF at end)
    runs.sort(key=lambda r: (
        r.rounds_to_compliance if r.rounds_to_compliance is not None
        else results.config.num_rounds + 1
    ))

    labels = [r.config_label for r in runs]
    values = [
        r.rounds_to_compliance if r.rounds_to_compliance is not None
        else results.config.num_rounds
        for r in runs
    ]
    is_dnf = [r.rounds_to_compliance is None for r in runs]

    fig, ax = plt.subplots(figsize=(10, max(4, len(runs) * 0.35)))
    y = np.arange(len(runs))

    # Color gradient: green (fast) -> red (slow)
    max_rounds = results.config.num_rounds
    colors = []
    for v, dnf in zip(values, is_dnf):
        if dnf:
            colors.append("#999999")
        else:
            ratio = v / max_rounds
            r = min(1.0, 2 * ratio)
            g = min(1.0, 2 * (1 - ratio))
            colors.append((r, g, 0.2))

    ax.barh(y, values, color=colors, height=0.7)

    # Annotate DNF
    for i, (v, dnf) in enumerate(zip(values, is_dnf)):
        if dnf:
            ax.text(v + 0.2, i, "DNF", va="center", fontsize=8, color="#666666")
        else:
            ax.text(v + 0.2, i, f"R{v}", va="center", fontsize=8)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Rounds to Compliance")
    ax.set_title("Time-to-Compliance (EHDS)")
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def _plot_pareto_frontier(results: EHDSBenchmarkResults, path: str):
    """Pareto frontier: compliance vs accuracy, size = comm cost."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 7))

    algos = sorted(set(r.algorithm for r in results.runs))
    cmap = plt.cm.Set1
    markers = {"minimal": "o", "full": "^"}

    max_cost = max((r.communication_cost for r in results.runs), default=1)

    for i, algo in enumerate(algos):
        color = cmap(i / max(len(algos) - 1, 1))
        for run in results.runs:
            if run.algorithm != algo:
                continue
            marker = markers.get(run.governance_level, "s")
            size = 30 + 200 * (run.communication_cost / max_cost)
            ax.scatter(
                run.compliance_score, run.final_accuracy,
                c=[color], marker=marker, s=size, alpha=0.6,
                edgecolors="black", linewidths=0.5,
            )

    # Compute and draw Pareto frontier
    points = [(r.compliance_score, r.final_accuracy) for r in results.runs]
    pareto = _pareto_front(points)
    if len(pareto) > 1:
        pareto.sort(key=lambda p: p[0])
        px, py = zip(*pareto)
        ax.plot(px, py, "k--", alpha=0.5, linewidth=1.5, label="Pareto frontier")

    # Annotate best point
    if pareto:
        best = max(pareto, key=lambda p: p[0] + p[1] * 100)
        ax.annotate(
            "Optimal", xy=best, fontsize=9,
            xytext=(best[0] - 8, best[1] + 0.02),
            arrowprops=dict(arrowstyle="->", color="black"),
        )

    ax.set_xlabel("Compliance Score (%)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Pareto Frontier: Compliance vs Utility")
    ax.grid(True, alpha=0.3)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = []
    for i, algo in enumerate(algos):
        legend_elements.append(
            Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=cmap(i / max(len(algos) - 1, 1)),
                   markersize=8, label=algo)
        )
    legend_elements.append(
        Line2D([0], [0], marker="o", color="grey", label="Minimal", linestyle="None")
    )
    legend_elements.append(
        Line2D([0], [0], marker="^", color="grey", label="Full Gov.", linestyle="None")
    )
    legend_elements.append(
        Line2D([0], [0], linestyle="--", color="black", label="Pareto frontier")
    )
    ax.legend(handles=legend_elements, fontsize=7, loc="lower right")

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def _pareto_front(points: List[tuple]) -> List[tuple]:
    """Compute 2D Pareto frontier (maximize both dimensions)."""
    pareto = []
    for p in points:
        dominated = False
        for q in points:
            if q[0] >= p[0] and q[1] >= p[1] and (q[0] > p[0] or q[1] > p[1]):
                dominated = True
                break
        if not dominated:
            pareto.append(p)
    return pareto


# =====================================================================
# LATEX TABLES
# =====================================================================

def _generate_main_table(results: EHDSBenchmarkResults, path: str):
    """Generate LaTeX benchmark results table."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{EHDS Benchmark: Privacy-Utility-Cost Tradeoff}",
        r"\label{tab:ehds-benchmark}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llrrrrrrr}",
        r"\toprule",
        r"Algorithm & Gov. & $\varepsilon$ & Accuracy & F1 & "
        r"Cost (MB) & Compliance (\%) & TTC \\",
        r"\midrule",
    ]

    # Group by algorithm, sort by epsilon
    algos = sorted(set(r.algorithm for r in results.runs))
    for algo in algos:
        algo_runs = sorted(
            [r for r in results.runs if r.algorithm == algo],
            key=lambda r: (r.governance_level, r.effective_epsilon),
        )
        for r in algo_runs:
            cost_mb = r.communication_cost / (1024 * 1024)
            ttc = str(r.rounds_to_compliance) if r.rounds_to_compliance is not None else "DNF"
            lines.append(
                f"{r.algorithm} & {r.governance_level} & "
                f"{r.effective_epsilon:.1f} & {r.final_accuracy:.2%} & "
                f"{r.final_f1:.3f} & {cost_mb:.2f} & "
                f"{r.compliance_score:.1f} & {ttc} \\\\"
            )
        lines.append(r"\midrule")

    # Remove last midrule
    if lines[-1] == r"\midrule":
        lines.pop()

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}}",
        r"\begin{tablenotes}\small",
        r"\item TTC = Time-to-Compliance (rounds). "
        r"DNF = threshold not reached within budget.",
        r"\end{tablenotes}",
        r"\end{table}",
    ])

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _generate_compliance_table(results: EHDSBenchmarkResults, path: str):
    """Generate per-chapter compliance breakdown table."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{EHDS Compliance Breakdown by Chapter}",
        r"\label{tab:ehds-compliance-breakdown}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Configuration & Compliant & Partial & Not Assessed & Score (\%) \\",
        r"\midrule",
    ]

    seen = set()
    for r in results.runs:
        if r.config_label in seen:
            continue
        seen.add(r.config_label)
        s = r.compliance_summary
        lines.append(
            f"{r.config_label} & "
            f"{s.get('compliant', 0)} & "
            f"{s.get('partial', 0)} & "
            f"{s.get('not_assessed', 0)} & "
            f"{r.compliance_score:.1f} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# =====================================================================
# SAVE RESULTS
# =====================================================================

def save_results(results: EHDSBenchmarkResults, output_dir: str) -> List[str]:
    """Save all benchmark outputs (JSON, CSV, plots, LaTeX).

    Returns:
        List of saved file names.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    saved = []

    # 1. results.json
    json_path = out / "results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(_to_serializable(results), f, indent=2, default=str)
    saved.append("results.json")

    # 2. benchmark_summary.csv
    csv_path = out / "benchmark_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "config_id", "config_label", "algorithm", "governance",
            "epsilon", "accuracy", "f1", "auc", "loss",
            "cost_bytes", "compliance_pct", "time_to_compliance",
        ])
        for r in results.runs:
            writer.writerow([
                r.config_id, r.config_label, r.algorithm, r.governance_level,
                r.effective_epsilon, f"{r.final_accuracy:.4f}",
                f"{r.final_f1:.4f}", f"{r.final_auc:.4f}", f"{r.final_loss:.4f}",
                r.communication_cost, f"{r.compliance_score:.1f}",
                r.rounds_to_compliance if r.rounds_to_compliance is not None else "DNF",
            ])
    saved.append("benchmark_summary.csv")

    # 3. privacy_utility_cost.csv (for 3D plot data)
    puc_path = out / "privacy_utility_cost.csv"
    with open(puc_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epsilon", "accuracy", "cost_mb", "compliance",
            "algorithm", "governance",
        ])
        for r in results.runs:
            writer.writerow([
                r.effective_epsilon, f"{r.final_accuracy:.4f}",
                f"{r.communication_cost / (1024*1024):.4f}",
                f"{r.compliance_score:.1f}",
                r.algorithm, r.governance_level,
            ])
    saved.append("privacy_utility_cost.csv")

    # 4-8. Plots
    try:
        _plot_privacy_utility_cost_3d(results, str(out / "plot_3d_tradeoff.png"))
        saved.append("plot_3d_tradeoff.png")
    except Exception as e:
        logger.warning(f"Failed to generate 3D plot: {e}")

    try:
        _plot_privacy_utility_2d(results, str(out / "plot_2d_projections.png"))
        saved.append("plot_2d_projections.png")
    except Exception as e:
        logger.warning(f"Failed to generate 2D plot: {e}")

    try:
        _plot_compliance_by_config(results, str(out / "plot_compliance_bars.png"))
        saved.append("plot_compliance_bars.png")
    except Exception as e:
        logger.warning(f"Failed to generate compliance plot: {e}")

    try:
        _plot_time_to_compliance(results, str(out / "plot_time_to_compliance.png"))
        saved.append("plot_time_to_compliance.png")
    except Exception as e:
        logger.warning(f"Failed to generate TTC plot: {e}")

    try:
        _plot_pareto_frontier(results, str(out / "plot_pareto_frontier.png"))
        saved.append("plot_pareto_frontier.png")
    except Exception as e:
        logger.warning(f"Failed to generate Pareto plot: {e}")

    # 9-10. LaTeX tables
    try:
        _generate_main_table(results, str(out / "table_benchmark.tex"))
        saved.append("table_benchmark.tex")
    except Exception as e:
        logger.warning(f"Failed to generate main table: {e}")

    try:
        _generate_compliance_table(results, str(out / "table_compliance.tex"))
        saved.append("table_compliance.tex")
    except Exception as e:
        logger.warning(f"Failed to generate compliance table: {e}")

    return saved


def _to_serializable(results: EHDSBenchmarkResults) -> Dict:
    """Convert results to JSON-serializable dict."""
    return {
        "config": asdict(results.config),
        "runs": [asdict(r) for r in results.runs],
        "generated_at": results.generated_at,
        "total_time_seconds": results.total_time_seconds,
        "summary": {
            "total_runs": len(results.runs),
            "avg_accuracy": np.mean([r.final_accuracy for r in results.runs]) if results.runs else 0,
            "avg_compliance": np.mean([r.compliance_score for r in results.runs]) if results.runs else 0,
            "pareto_configs": _get_pareto_labels(results),
        },
    }


def _get_pareto_labels(results: EHDSBenchmarkResults) -> List[str]:
    """Get config labels of Pareto-optimal runs."""
    points = [(r.compliance_score, r.final_accuracy) for r in results.runs]
    pareto = _pareto_front(points)
    pareto_set = set(pareto)
    return [
        r.config_label for r in results.runs
        if (r.compliance_score, r.final_accuracy) in pareto_set
    ]

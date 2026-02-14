#!/usr/bin/env python3
"""
FL-EHDS Imaging Benchmark Suite
================================
Runs federated learning experiments on clinical imaging datasets.

This script generates results for the paper appendix:
- chest_xray: Pneumonia detection (2 classes, 5,856 images)
- Skin Cancer: Benign/Malignant (2 classes, 3,297 images)

Usage:
    python run_imaging_experiments.py --dataset chest_xray --quick
    python run_imaging_experiments.py --dataset chest_xray --full

Author: Fabio Liberti
Date: February 2026
"""

import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from tqdm import tqdm

from terminal.fl_trainer import ImageFederatedTrainer


def run_experiment(
    data_dir: str,
    algorithm: str,
    num_clients: int,
    num_rounds: int,
    local_epochs: int,
    is_iid: bool,
    dp_enabled: bool = False,
    dp_epsilon: float = 10.0,
    seed: int = 42,
    batch_size: int = 32,
    learning_rate: float = 0.001,
) -> Dict[str, Any]:
    """Run a single FL experiment."""

    print(f"\n{'='*60}")
    print(f"Experiment: {algorithm} | {'IID' if is_iid else 'Non-IID'} | DP={dp_enabled}")
    print(f"{'='*60}")

    start_time = time.time()

    trainer = ImageFederatedTrainer(
        data_dir=data_dir,
        num_clients=num_clients,
        algorithm=algorithm,
        local_epochs=local_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        is_iid=is_iid,
        alpha=0.5,  # Non-IID parameter
        dp_enabled=dp_enabled,
        dp_epsilon=dp_epsilon,
        dp_clip_norm=1.0,
        seed=seed,
    )

    # Training loop with tqdm progress bar
    history = []
    round_pbar = tqdm(
        range(num_rounds),
        desc="Rounds",
        ncols=100,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
    )

    for round_num in round_pbar:
        result = trainer.train_round(round_num)
        history.append({
            "round": round_num + 1,
            "accuracy": result.global_acc,
            "loss": result.global_loss,
            "f1": result.global_f1,
            "precision": result.global_precision,
            "recall": result.global_recall,
            "auc": result.global_auc,
            "time": result.time_seconds,
        })

        # Update progress bar with metrics
        round_pbar.set_postfix(
            acc=f"{result.global_acc:.2%}",
            loss=f"{result.global_loss:.4f}",
            f1=f"{result.global_f1:.2f}"
        )

    total_time = time.time() - start_time
    final = history[-1]

    return {
        "algorithm": algorithm,
        "is_iid": is_iid,
        "dp_enabled": dp_enabled,
        "dp_epsilon": dp_epsilon if dp_enabled else None,
        "num_clients": num_clients,
        "num_rounds": num_rounds,
        "local_epochs": local_epochs,
        "seed": seed,
        "final_accuracy": final["accuracy"],
        "final_f1": final["f1"],
        "final_precision": final["precision"],
        "final_recall": final["recall"],
        "final_auc": final["auc"],
        "final_loss": final["loss"],
        "total_time_seconds": total_time,
        "history": history,
    }


def run_benchmark_suite(
    dataset_name: str,
    data_dir: str,
    output_dir: Path,
    quick: bool = False,
    ultra_quick: bool = False,
    seeds: List[int] = [42, 123, 456],
) -> Dict[str, Any]:
    """Run complete benchmark suite for a dataset."""

    # Configuration based on mode
    if ultra_quick:
        # Ultra-quick for CPU testing: minimal configuration
        num_clients = 2
        num_rounds = 3
        local_epochs = 1
        batch_size = 64  # Larger batch = fewer iterations
        configs = [
            ("FedAvg", False, False, None),  # Just one config for testing
        ]
        print("\n[ULTRA-QUICK MODE] Minimal config for CPU testing")
        print(f"  Clients: {num_clients}, Rounds: {num_rounds}, Epochs: {local_epochs}")
    elif quick:
        num_clients = 5
        num_rounds = 10
        local_epochs = 2
        batch_size = 32
        configs = [
            ("FedAvg", True, False, None),
            ("FedAvg", False, False, None),
            ("FedProx", False, False, None),
            ("FedAvg", False, True, 10.0),
            ("FedAvg", False, True, 1.0),
        ]
    else:
        num_clients = 5
        num_rounds = 20
        local_epochs = 3
        batch_size = 32
        configs = [
            ("FedAvg", True, False, None),
            ("FedAvg", False, False, None),
            ("FedProx", False, False, None),
            ("FedAvg", False, True, 10.0),
            ("FedAvg", False, True, 1.0),
            ("FedProx", True, False, None),
            ("FedAvg", True, True, 10.0),
        ]

    all_results = []

    for algorithm, is_iid, dp_enabled, dp_epsilon in configs:
        seed_results = []

        for seed in seeds:
            result = run_experiment(
                data_dir=data_dir,
                algorithm=algorithm,
                num_clients=num_clients,
                num_rounds=num_rounds,
                local_epochs=local_epochs,
                is_iid=is_iid,
                dp_enabled=dp_enabled,
                dp_epsilon=dp_epsilon if dp_epsilon else 10.0,
                seed=seed,
                batch_size=batch_size,
            )
            seed_results.append(result)

        # Aggregate across seeds
        accuracies = [r["final_accuracy"] for r in seed_results]
        f1s = [r["final_f1"] for r in seed_results]
        aucs = [r["final_auc"] for r in seed_results]
        precisions = [r["final_precision"] for r in seed_results]
        recalls = [r["final_recall"] for r in seed_results]

        aggregated = {
            "algorithm": algorithm,
            "is_iid": is_iid,
            "dp_enabled": dp_enabled,
            "dp_epsilon": dp_epsilon,
            "accuracy_mean": np.mean(accuracies),
            "accuracy_std": np.std(accuracies),
            "f1_mean": np.mean(f1s),
            "f1_std": np.std(f1s),
            "auc_mean": np.mean(aucs),
            "auc_std": np.std(aucs),
            "precision_mean": np.mean(precisions),
            "precision_std": np.std(precisions),
            "recall_mean": np.mean(recalls),
            "recall_std": np.std(recalls),
            "seeds": seeds,
            "per_seed_results": seed_results,
        }
        all_results.append(aggregated)

        print(f"\n>> {algorithm} ({'IID' if is_iid else 'Non-IID'}) "
              f"{'+ DP(e=' + str(dp_epsilon) + ')' if dp_enabled else ''}")
        print(f"   Acc: {aggregated['accuracy_mean']:.4f} +/- {aggregated['accuracy_std']:.4f}")
        print(f"   F1:  {aggregated['f1_mean']:.4f} +/- {aggregated['f1_std']:.4f}")

    # Generate outputs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"imaging_{dataset_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON results
    summary = {
        "dataset": dataset_name,
        "data_dir": data_dir,
        "timestamp": timestamp,
        "config": {
            "num_clients": num_clients,
            "num_rounds": num_rounds,
            "local_epochs": local_epochs,
            "batch_size": batch_size,
            "seeds": seeds,
            "quick_mode": quick,
        },
        "results": all_results,
    }

    with open(run_dir / "results.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Generate LaTeX table
    latex_table = generate_latex_table(dataset_name, all_results, num_clients, num_rounds, local_epochs)
    with open(run_dir / "table_results.tex", "w") as f:
        f.write(latex_table)

    # Significance testing vs FedAvg non-IID baseline
    try:
        from benchmarks.significance import pairwise_significance_table, format_significance_latex

        # Build per-algo metric arrays from seed results
        algo_data = {}
        for agg in all_results:
            label = agg["algorithm"]
            if agg["is_iid"]:
                label += " (IID)"
            if agg["dp_enabled"]:
                label += f" +DP(e={agg['dp_epsilon']})"
            seed_results_list = agg["per_seed_results"]
            algo_data[label] = {
                "accuracy": [r["final_accuracy"] for r in seed_results_list],
                "f1": [r["final_f1"] for r in seed_results_list],
                "auc": [r["final_auc"] for r in seed_results_list],
            }

        # Use FedAvg non-IID as baseline
        baseline_label = "FedAvg"
        if baseline_label in algo_data and len(algo_data) > 1:
            sig_table = pairwise_significance_table(
                algo_data, baseline=baseline_label, metrics=["accuracy", "f1", "auc"]
            )
            print(f"\nSIGNIFICANCE TESTING (vs {baseline_label}):")
            for algo, metrics in sorted(sig_table.items()):
                for metric, info in sorted(metrics.items()):
                    marker = "*" if info["significant"] else ""
                    print(f"  {algo:<25} {metric:<10} p={info['p_value']:.4f} d={info['cohens_d']:.3f} {marker}")

            # Save significance LaTeX
            sig_latex = format_significance_latex(
                algo_data, baseline=baseline_label,
                caption=f"Significance Testing: {dataset_name.replace('_', ' ').title()}",
                label=f"tab:{dataset_name}_significance",
            )
            with open(run_dir / "table_significance.tex", "w") as f:
                f.write(sig_latex)

            summary["significance"] = {
                algo: {m: {"p": info["p_value"], "d": info["cohens_d"], "sig": info["significant"]}
                       for m, info in metrics.items()}
                for algo, metrics in sig_table.items()
            }

    except ImportError:
        pass

    print(f"\n{'='*60}")
    print(f"Results saved to: {run_dir}")
    print(f"{'='*60}")

    return summary


def generate_latex_table(
    dataset_name: str,
    results: List[Dict],
    num_clients: int,
    num_rounds: int,
    local_epochs: int,
) -> str:
    """Generate LaTeX table for paper appendix."""

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        f"\\caption{{FL Results on {dataset_name.replace('_', ' ').title()} Dataset}}",
        f"\\label{{tab:{dataset_name}_results}}",
        r"\small",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"\textbf{Configuration} & \textbf{Acc.} & \textbf{F1} & \textbf{AUC} & \textbf{Prec.} & \textbf{Rec.} \\",
        r"\midrule",
    ]

    for r in results:
        config_name = r["algorithm"]
        if r["is_iid"]:
            config_name += " (IID)"
        else:
            config_name += " (Non-IID)"
        if r["dp_enabled"]:
            config_name += f" + DP($\\varepsilon$={r['dp_epsilon']})"

        lines.append(
            f"{config_name} & "
            f"{r['accuracy_mean']*100:.1f}\\%$\\pm${r['accuracy_std']*100:.1f} & "
            f"{r['f1_mean']:.2f}$\\pm${r['f1_std']:.2f} & "
            f"{r['auc_mean']:.2f}$\\pm${r['auc_std']:.2f} & "
            f"{r['precision_mean']:.2f} & "
            f"{r['recall_mean']:.2f} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        "",
        r"\vspace{1mm}",
        f"\\footnotesize{{{num_clients} hospitals, {num_rounds} rounds, {local_epochs} local epochs. "
        f"HealthcareCNN model. Results: mean $\\pm$ std over 3 seeds.}}",
        r"\end{table}",
    ])

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="FL-EHDS Imaging Benchmark")
    parser.add_argument("--dataset", type=str, default="chest_xray",
                        choices=["chest_xray", "Skin Cancer", "Brain_Tumor"],
                        help="Dataset to use")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: fewer rounds and seeds")
    parser.add_argument("--ultra-quick", action="store_true",
                        help="Ultra-quick mode for CPU testing: 2 rounds, 1 epoch, 2 clients")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456],
                        help="Random seeds for experiments")
    args = parser.parse_args()

    # Paths
    framework_dir = Path(__file__).parent.parent
    data_dir = framework_dir / "data" / args.dataset

    # For split structure datasets, point to train folder
    if args.dataset == "chest_xray":
        data_dir = data_dir / "train"
    elif args.dataset == "Skin Cancer":
        data_dir = data_dir / "train"

    if not data_dir.exists():
        print(f"Error: Dataset not found at {data_dir}")
        print("Available datasets:")
        for d in (framework_dir / "data").iterdir():
            if d.is_dir() and not d.name.startswith('.'):
                print(f"  - {d.name}")
        sys.exit(1)

    output_dir = framework_dir / "results" / "imaging_benchmarks"

    # Determine mode and seeds
    ultra_quick = getattr(args, 'ultra_quick', False)

    if ultra_quick:
        seeds = [42]
        mode_str = "ULTRA-QUICK (CPU test)"
    elif args.quick:
        seeds = [42]
        mode_str = "Quick"
    else:
        seeds = args.seeds
        mode_str = "Full"

    print(f"\nFL-EHDS Imaging Benchmark")
    print(f"========================")
    print(f"Dataset: {args.dataset}")
    print(f"Data dir: {data_dir}")
    print(f"Mode: {mode_str}")
    print(f"Seeds: {seeds}")

    # Check for GPU
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cpu":
        print("\n[WARNING] Running on CPU - training will be slow!")
        print("          Use --ultra-quick for quick testing on CPU")

    run_benchmark_suite(
        dataset_name=args.dataset,
        data_dir=str(data_dir),
        output_dir=output_dir,
        quick=args.quick,
        ultra_quick=ultra_quick,
        seeds=seeds,
    )


if __name__ == "__main__":
    main()

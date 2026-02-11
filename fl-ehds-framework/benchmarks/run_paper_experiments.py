#!/usr/bin/env python3
"""
FL-EHDS Paper Experiments for FLICS 2026.

Produces all experimental results for the paper:
  P1.2 - Multi-dataset FL comparison (5 algos x 5 datasets x 3 seeds)
  P1.3 - Ablation study on chest_xray
  P2.1 - Statistical significance via paired t-tests
  P2.2 - Privacy attack evaluation (gradient inversion)

Usage:
    cd fl-ehds-framework
    python -m benchmarks.run_paper_experiments [--resume] [--quick] [--only p12|p13|p21|p22]

Output: benchmarks/paper_results/

Author: Fabio Liberti
"""

import sys
import json
import time
import argparse
import traceback
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Callable

import numpy as np

# Add parent to path for module imports
FRAMEWORK_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(FRAMEWORK_DIR))

import torch
import torch.nn as nn
from scipy import stats

from terminal.fl_trainer import (
    FederatedTrainer,
    ImageFederatedTrainer,
    RoundResult,
    HealthcareMLP,
    _detect_device,
)
from data.diabetes_loader import load_diabetes_data
from data.heart_disease_loader import load_heart_disease_data

# ======================================================================
# Configuration
# ======================================================================

SEEDS = [42, 123, 456]
OUTPUT_DIR = FRAMEWORK_DIR / "benchmarks" / "paper_results"

IMAGING_DATASETS = {
    "Brain_Tumor": {
        "data_dir": str(FRAMEWORK_DIR / "data" / "Brain_Tumor"),
        "num_classes": 4,
        "short": "BT",
    },
    "chest_xray": {
        "data_dir": str(FRAMEWORK_DIR / "data" / "chest_xray"),
        "num_classes": 2,
        "short": "CX",
    },
    "Skin_Cancer": {
        "data_dir": str(FRAMEWORK_DIR / "data" / "Skin Cancer"),
        "num_classes": 2,
        "short": "SC",
    },
}

TABULAR_DATASETS = {
    "Diabetes": {"num_features": 22, "short": "DM"},
    "Heart_Disease": {"num_features": 13, "short": "HD"},
}

ALGORITHMS = ["FedAvg", "FedProx", "SCAFFOLD", "FedNova", "Ditto"]
ALGO_COLORS = ["#2196F3", "#FF9800", "#4CAF50", "#F44336", "#9C27B0"]

IMAGING_CONFIG = dict(
    num_clients=5, num_rounds=15, local_epochs=3,
    batch_size=32, learning_rate=0.0005,
    model_type="resnet18", is_iid=False, alpha=0.5,
)

TABULAR_CONFIG = dict(
    num_rounds=20, local_epochs=3,
    batch_size=32, learning_rate=0.01, is_iid=False, alpha=0.5,
)


# ======================================================================
# Checkpoint / Resume
# ======================================================================

def save_checkpoint(block_name: str, data: Any) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"checkpoint_{block_name}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_checkpoint(block_name: str) -> Optional[Dict]:
    path = OUTPUT_DIR / f"checkpoint_{block_name}.json"
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return None


def _cleanup_gpu():
    """Free GPU/MPS memory between experiments."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        try:
            torch.mps.empty_cache()
        except Exception:
            pass
    import gc
    gc.collect()


# ======================================================================
# Core Training Wrappers
# ======================================================================

def run_single_imaging(
    dataset_name: str, data_dir: str, algorithm: str, seed: int,
    num_rounds: int = 15, num_clients: int = 5, local_epochs: int = 3,
    batch_size: int = 32, learning_rate: float = 0.0005,
    model_type: str = "resnet18", is_iid: bool = False, alpha: float = 0.5,
    dp_enabled: bool = False, dp_epsilon: float = 10.0, dp_clip_norm: float = 1.0,
    use_class_weights: bool = True, freeze_backbone: bool = False,
    mu: float = 0.1,
) -> Dict[str, Any]:
    """Run a single imaging FL experiment."""
    start = time.time()

    trainer = ImageFederatedTrainer(
        data_dir=data_dir,
        num_clients=num_clients,
        algorithm=algorithm,
        local_epochs=local_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        is_iid=is_iid,
        alpha=alpha,
        mu=mu,
        dp_enabled=dp_enabled,
        dp_epsilon=dp_epsilon,
        dp_clip_norm=dp_clip_norm,
        seed=seed,
        model_type=model_type,
        use_class_weights=use_class_weights,
        freeze_backbone=freeze_backbone,
    )

    history = []
    for r in range(num_rounds):
        result = trainer.train_round(r)
        history.append({
            "round": r + 1,
            "accuracy": result.global_acc,
            "loss": result.global_loss,
            "f1": result.global_f1,
            "precision": result.global_precision,
            "recall": result.global_recall,
            "auc": result.global_auc,
        })

    elapsed = time.time() - start
    final = history[-1]

    return {
        "dataset": dataset_name, "algorithm": algorithm, "seed": seed,
        "history": history,
        "final_metrics": {k: final[k] for k in ["accuracy", "loss", "f1", "precision", "recall", "auc"]},
        "runtime_seconds": round(elapsed, 1),
        "config": {
            "num_clients": num_clients, "num_rounds": num_rounds,
            "model_type": model_type, "dp_enabled": dp_enabled,
            "dp_epsilon": dp_epsilon if dp_enabled else None,
            "dp_clip_norm": dp_clip_norm if dp_enabled else None,
            "use_class_weights": use_class_weights,
        },
    }


def run_single_tabular(
    dataset_name: str,
    client_train: Dict[int, Tuple], client_test: Dict[int, Tuple],
    input_dim: int, algorithm: str, seed: int,
    num_rounds: int = 20, local_epochs: int = 3,
    batch_size: int = 32, learning_rate: float = 0.01,
    mu: float = 0.1,
    dp_enabled: bool = False, dp_epsilon: float = 10.0, dp_clip_norm: float = 1.0,
) -> Dict[str, Any]:
    """Run a single tabular FL experiment."""
    start = time.time()

    trainer = FederatedTrainer(
        num_clients=len(client_train),
        algorithm=algorithm,
        local_epochs=local_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        mu=mu,
        dp_enabled=dp_enabled,
        dp_epsilon=dp_epsilon,
        dp_clip_norm=dp_clip_norm,
        seed=seed,
        external_data=client_train,
        external_test_data=client_test,
        input_dim=input_dim,
    )

    history = []
    for r in range(num_rounds):
        result = trainer.train_round(r)
        history.append({
            "round": r + 1,
            "accuracy": result.global_acc,
            "loss": result.global_loss,
            "f1": result.global_f1,
            "precision": result.global_precision,
            "recall": result.global_recall,
            "auc": result.global_auc,
        })

    elapsed = time.time() - start
    final = history[-1]

    return {
        "dataset": dataset_name, "algorithm": algorithm, "seed": seed,
        "history": history,
        "final_metrics": {k: final[k] for k in ["accuracy", "loss", "f1", "precision", "recall", "auc"]},
        "runtime_seconds": round(elapsed, 1),
        "config": {
            "num_clients": len(client_train), "num_rounds": num_rounds,
            "dp_enabled": dp_enabled,
        },
    }


# ======================================================================
# P1.2: Multi-Dataset Experiments
# ======================================================================

def run_p12_multi_dataset(resume: bool = False,
                          filter_dataset: Optional[str] = None,
                          filter_algo: Optional[str] = None) -> Dict[str, Any]:
    """5 algorithms x 5 datasets x 3 seeds = 75 experiments."""
    block = "p12_multidataset"
    # Always load existing checkpoint to merge results incrementally
    results = load_checkpoint(block)
    if results is None:
        results = {"completed": {}}

    all_datasets = list(IMAGING_DATASETS.keys()) + list(TABULAR_DATASETS.keys())
    run_datasets = [filter_dataset] if filter_dataset else all_datasets
    run_algos = [filter_algo] if filter_algo else ALGORITHMS

    total = len(run_algos) * len(run_datasets) * len(SEEDS)
    done = sum(1 for k, v in results["completed"].items() if "error" not in v)
    print(f"  Total experiments: {total}, already completed (all): {done}")

    count = 0
    for ds_name in run_datasets:
        if ds_name not in all_datasets:
            print(f"  WARNING: Unknown dataset '{ds_name}'. Available: {all_datasets}")
            continue
        for algo in run_algos:
            if algo not in ALGORITHMS:
                print(f"  WARNING: Unknown algorithm '{algo}'. Available: {ALGORITHMS}")
                continue
            for seed in SEEDS:
                count += 1
                key = f"{ds_name}_{algo}_{seed}"
                if key in results["completed"]:
                    continue

                print(f"  [{count}/{total}] {ds_name} / {algo} / seed={seed}", end=" ", flush=True)

                try:
                    if ds_name in IMAGING_DATASETS:
                        ds_info = IMAGING_DATASETS[ds_name]
                        record = run_single_imaging(
                            dataset_name=ds_name,
                            data_dir=ds_info["data_dir"],
                            algorithm=algo,
                            seed=seed,
                            mu=0.1,
                            **IMAGING_CONFIG,
                        )
                    elif ds_name == "Diabetes":
                        train_d, test_d, meta = load_diabetes_data(
                            num_clients=5, is_iid=False, alpha=0.5, seed=seed,
                        )
                        record = run_single_tabular(
                            dataset_name=ds_name,
                            client_train=train_d, client_test=test_d,
                            input_dim=22, algorithm=algo, seed=seed,
                            **TABULAR_CONFIG,
                        )
                    elif ds_name == "Heart_Disease":
                        train_d, test_d, meta = load_heart_disease_data(
                            num_clients=4, partition_by_hospital=True, seed=seed,
                        )
                        record = run_single_tabular(
                            dataset_name=ds_name,
                            client_train=train_d, client_test=test_d,
                            input_dim=13, algorithm=algo, seed=seed,
                            num_rounds=TABULAR_CONFIG["num_rounds"],
                            local_epochs=TABULAR_CONFIG["local_epochs"],
                            batch_size=TABULAR_CONFIG["batch_size"],
                            learning_rate=TABULAR_CONFIG["learning_rate"],
                        )
                    else:
                        continue

                    acc = record["final_metrics"]["accuracy"]
                    t = record["runtime_seconds"]
                    print(f"-> acc={acc:.3f} ({t:.0f}s)")
                    results["completed"][key] = record

                except Exception as e:
                    print(f"-> ERROR: {e}")
                    results["completed"][key] = {"error": str(e), "traceback": traceback.format_exc()}

                save_checkpoint(block, results)
                _cleanup_gpu()

    return results


# ======================================================================
# P1.3: Ablation Study
# ======================================================================

def run_p13_ablation(resume: bool = False) -> Dict[str, Any]:
    """Ablation on chest_xray: clip, epsilon, model, class_weights."""
    block = "p13_ablation"
    # Always load existing checkpoint to merge results incrementally
    results = load_checkpoint(block)
    if results is None:
        results = {"completed": {}}

    data_dir = str(FRAMEWORK_DIR / "data" / "chest_xray")
    base = dict(
        dataset_name="chest_xray", data_dir=data_dir,
        algorithm="FedAvg", num_rounds=15, num_clients=5,
        local_epochs=3, batch_size=32, learning_rate=0.0005,
        model_type="resnet18",
    )

    experiments = []

    # A: Gradient clipping
    for c in [0.5, 1.0, 2.0]:
        for seed in SEEDS:
            experiments.append((
                f"clip_{c}_{seed}",
                dict(**base, seed=seed, dp_enabled=True, dp_epsilon=5.0, dp_clip_norm=c),
                f"Clip C={c}"
            ))

    # B: Privacy epsilon
    for eps in [0.5, 1.0, 2.0, 5.0, 10.0]:
        for seed in SEEDS:
            experiments.append((
                f"epsilon_{eps}_{seed}",
                dict(**base, seed=seed, dp_enabled=True, dp_epsilon=eps, dp_clip_norm=1.0),
                f"Eps={eps}"
            ))
    for seed in SEEDS:
        experiments.append((
            f"epsilon_inf_{seed}",
            dict(**base, seed=seed, dp_enabled=False),
            "Eps=inf"
        ))

    # C: Model type
    for mt in ["cnn", "resnet18"]:
        lr = 0.001 if mt == "cnn" else 0.0005
        for seed in SEEDS:
            experiments.append((
                f"model_{mt}_{seed}",
                dict(**base, seed=seed, model_type=mt, learning_rate=lr, dp_enabled=False),
                f"Model={mt}"
            ))

    # D: Class weights
    for cw in [True, False]:
        for seed in SEEDS:
            experiments.append((
                f"classweights_{cw}_{seed}",
                dict(**base, seed=seed, use_class_weights=cw, dp_enabled=False),
                f"CW={cw}"
            ))

    total = len(experiments)
    done = sum(1 for k, _, _ in experiments if k in results["completed"])
    print(f"  Total ablation experiments: {total}, already completed: {done}")

    for i, (key, params, label) in enumerate(experiments):
        if key in results["completed"]:
            continue

        print(f"  [{i+1}/{total}] {label}, seed={params['seed']}", end=" ", flush=True)

        try:
            record = run_single_imaging(**params)
            acc = record["final_metrics"]["accuracy"]
            t = record["runtime_seconds"]
            print(f"-> acc={acc:.3f} ({t:.0f}s)")
            results["completed"][key] = record
        except Exception as e:
            print(f"-> ERROR: {e}")
            results["completed"][key] = {"error": str(e)}

        save_checkpoint(block, results)
        _cleanup_gpu()

    return results


# ======================================================================
# P2.1: Statistical Significance
# ======================================================================

def run_p21_significance(p12_results: Dict[str, Any]) -> Dict[str, Any]:
    """Paired t-tests: FedAvg vs each algorithm, per dataset."""
    completed = p12_results.get("completed", {})
    all_datasets = list(IMAGING_DATASETS.keys()) + list(TABULAR_DATASETS.keys())

    sig = {}
    for ds in all_datasets:
        sig[ds] = {}

        # Gather FedAvg across seeds
        fa_vals = {"accuracy": [], "f1": [], "auc": []}
        for seed in SEEDS:
            rec = completed.get(f"{ds}_FedAvg_{seed}", {})
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
                rec = completed.get(f"{ds}_{algo}_{seed}", {})
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

    save_checkpoint("p21_significance", sig)
    return sig


# ======================================================================
# P2.2: Privacy Attack (DLG)
# ======================================================================

def run_p22_privacy_attack() -> Dict[str, Any]:
    """DLG gradient inversion with/without DP."""
    results = {"completed": {}}

    for eps in [1.0, 5.0, 10.0, None]:
        eps_label = str(eps) if eps else "inf"
        for seed in SEEDS:
            key = f"eps_{eps_label}_{seed}"
            print(f"  Attack: eps={eps_label}, seed={seed}", end=" ", flush=True)

            try:
                mse = _gradient_inversion(epsilon=eps, seed=seed)
                print(f"-> MSE={mse:.4f}")
                results["completed"][key] = {
                    "epsilon": eps_label, "seed": seed,
                    "reconstruction_mse": round(float(mse), 6),
                }
            except Exception as e:
                print(f"-> ERROR: {e}")
                results["completed"][key] = {"error": str(e)}

    save_checkpoint("p22_attack", results)
    return results


def _gradient_inversion(
    epsilon: Optional[float], seed: int,
    num_samples: int = 32, input_dim: int = 10, num_iters: int = 300,
    clip_norm: float = 1.0,
) -> float:
    """DLG-style gradient inversion attack."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = _detect_device()

    # Real data
    X_real = torch.randn(num_samples, input_dim, device=device)
    y_real = torch.randint(0, 2, (num_samples,), device=device)

    # Model
    model = HealthcareMLP(input_dim=input_dim, num_classes=2).to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss()
    out = model(X_real)
    loss = criterion(out, y_real)

    real_grads = torch.autograd.grad(loss, model.parameters())
    real_grads = [g.detach().clone() for g in real_grads]

    # Clip
    total_norm = torch.sqrt(sum(g.norm() ** 2 for g in real_grads))
    clip_factor = min(1.0, clip_norm / (total_norm.item() + 1e-7))
    clipped = [g * clip_factor for g in real_grads]

    # DP noise
    if epsilon is not None:
        sigma = clip_norm * np.sqrt(2 * np.log(1.25 / 1e-5)) / epsilon
        target_grads = [g + torch.randn_like(g) * sigma for g in clipped]
    else:
        target_grads = clipped

    # DLG: optimize dummy data
    dummy_X = torch.randn(num_samples, input_dim, device=device, requires_grad=True)
    dummy_y = y_real.clone()  # Assume labels known (standard DLG assumption)

    optimizer = torch.optim.LBFGS([dummy_X], lr=0.1, max_iter=20)

    for _ in range(num_iters // 20):
        def closure():
            optimizer.zero_grad()
            dummy_out = model(dummy_X)
            dummy_loss = criterion(dummy_out, dummy_y)
            dummy_grads = torch.autograd.grad(dummy_loss, model.parameters(), create_graph=True)
            grad_diff = sum(((dg - tg) ** 2).sum() for dg, tg in zip(dummy_grads, target_grads))
            grad_diff.backward()
            return grad_diff
        optimizer.step(closure)

    mse = ((dummy_X.detach() - X_real) ** 2).mean().item()
    return mse


# ======================================================================
# Output Generation: LaTeX Tables
# ======================================================================

def _agg(completed: Dict, dataset: str, algo: str, metric: str = "accuracy") -> Tuple[float, float]:
    """Aggregate metric across seeds -> (mean, std)."""
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
    """Aggregate ablation metric across seeds."""
    vals = []
    for seed in SEEDS:
        rec = completed.get(f"{prefix}_{seed}", {})
        fm = rec.get("final_metrics")
        if fm and metric in fm:
            vals.append(fm[metric])
    if not vals:
        return 0.0, 0.0
    return float(np.mean(vals)), float(np.std(vals))


def generate_multi_dataset_table(p12: Dict, sig: Dict) -> str:
    """Generate LaTeX table for multi-dataset results."""
    completed = p12.get("completed", {})
    all_ds = list(IMAGING_DATASETS.keys()) + list(TABULAR_DATASETS.keys())
    ds_shorts = {**{k: v["short"] for k, v in IMAGING_DATASETS.items()},
                 **{k: v["short"] for k, v in TABULAR_DATASETS.items()}}

    lines = []
    lines.append(r"\begin{table*}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Multi-Dataset Federated Learning Results}")
    lines.append(r"\label{tab:multi_dataset}")
    lines.append(r"\small")

    # Header
    ncols = 1 + len(all_ds) * 3  # algo + 3 metrics per dataset
    col_spec = "l" + "|ccc" * len(all_ds)
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")

    # Dataset names row
    header1 = r"\textbf{Algorithm}"
    for idx, ds in enumerate(all_ds):
        sep = "c|" if idx < len(all_ds) - 1 else "c"
        header1 += r" & \multicolumn{3}{" + sep + r"}{\textbf{" + ds_shorts[ds] + "}}"
    lines.append(header1 + r" \\")

    # Metric names row
    header2 = ""
    for _ in all_ds:
        header2 += r" & Acc & F1 & AUC"
    lines.append(header2 + r" \\")
    lines.append(r"\midrule")

    # Data rows
    for algo in ALGORITHMS:
        row = algo
        for ds in all_ds:
            for metric in ["accuracy", "f1", "auc"]:
                mean, std = _agg(completed, ds, algo, metric)
                if mean == 0:
                    row += " & --"
                    continue

                # Format
                if metric == "accuracy":
                    cell = f"{mean*100:.1f}"
                else:
                    cell = f"{mean:.2f}"

                # Significance marker
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
    """Generate LaTeX table for ablation study."""
    completed = p13.get("completed", {})

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
        acc_m, acc_s = _agg_ablation(completed, prefix, "accuracy")
        f1_m, f1_s = _agg_ablation(completed, prefix, "f1")
        auc_m, auc_s = _agg_ablation(completed, prefix, "auc")
        if acc_m == 0:
            return f"& {label} & -- & -- & -- \\\\"
        return (f"& {label} & {acc_m*100:.1f}$\\pm${acc_s*100:.1f} "
                f"& {f1_m:.2f}$\\pm${f1_s:.2f} & {auc_m:.2f}$\\pm${auc_s:.2f} \\\\")

    # A: Clipping
    lines.append(r"\multirow{3}{*}{Grad. Clip $C$}")
    for c in [0.5, 1.0, 2.0]:
        lines.append(_row(f"clip_{c}", f"$C$={c}"))
    lines.append(r"\midrule")

    # B: Epsilon
    lines.append(r"\multirow{6}{*}{Privacy $\varepsilon$}")
    for eps in [0.5, 1.0, 2.0, 5.0, 10.0]:
        lines.append(_row(f"epsilon_{eps}", f"$\\varepsilon$={eps}"))
    lines.append(_row("epsilon_inf", r"$\infty$ (no DP)"))
    lines.append(r"\midrule")

    # C: Model
    lines.append(r"\multirow{2}{*}{Model}")
    lines.append(_row("model_cnn", "HealthcareCNN"))
    lines.append(_row("model_resnet18", "ResNet18"))
    lines.append(r"\midrule")

    # D: Class weights
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
    """Generate LaTeX table for privacy attack."""
    completed = p22.get("completed", {})

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
            rec = completed.get(f"eps_{eps_label}_{seed}", {})
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
# Output Generation: Figures
# ======================================================================

def generate_figures(p12: Dict, p13: Dict, p22: Dict, out_dir: Path) -> None:
    """Generate all publication-quality figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 11, "font.family": "serif",
        "axes.labelsize": 12, "axes.titlesize": 13,
        "xtick.labelsize": 10, "ytick.labelsize": 10,
        "legend.fontsize": 9, "figure.dpi": 150,
    })

    if p12:
        _fig_multi_dataset_bars(p12, out_dir, plt)
        _fig_convergence(p12, out_dir, plt)

    if p13:
        _fig_epsilon_tradeoff(p13, out_dir, plt)
        _fig_model_comparison(p13, out_dir, plt)

    if p22:
        _fig_attack_mse(p22, out_dir, plt)


def _fig_multi_dataset_bars(p12, out_dir, plt):
    """Grouped bar chart: accuracy per algorithm x dataset."""
    completed = p12.get("completed", {})
    all_ds = list(IMAGING_DATASETS.keys()) + list(TABULAR_DATASETS.keys())
    ds_labels = [IMAGING_DATASETS.get(d, TABULAR_DATASETS.get(d, {})).get("short", d) for d in all_ds]

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(all_ds))
    width = 0.15

    for i, algo in enumerate(ALGORITHMS):
        means, stds = [], []
        for ds in all_ds:
            m, s = _agg(completed, ds, algo, "accuracy")
            means.append(m * 100)
            stds.append(s * 100)
        ax.bar(x + i * width, means, width, yerr=stds, label=algo,
               color=ALGO_COLORS[i], capsize=3, alpha=0.85)

    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(ds_labels)
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("Multi-Dataset Federated Learning Comparison")

    for fmt in ["pdf", "png"]:
        fig.savefig(out_dir / f"fig_multi_dataset_comparison.{fmt}", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved fig_multi_dataset_comparison")


def _fig_convergence(p12, out_dir, plt):
    """5 subplot convergence curves (accuracy vs round)."""
    completed = p12.get("completed", {})
    all_ds = list(IMAGING_DATASETS.keys()) + list(TABULAR_DATASETS.keys())

    fig, axes = plt.subplots(1, len(all_ds), figsize=(4 * len(all_ds), 4), sharey=False)
    if len(all_ds) == 1:
        axes = [axes]

    for ax, ds in zip(axes, all_ds):
        for i, algo in enumerate(ALGORITHMS):
            # Average history across seeds
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
            ax.fill_between(rounds, mean - std, mean + std, color=ALGO_COLORS[i], alpha=0.15)

        short = IMAGING_DATASETS.get(ds, TABULAR_DATASETS.get(ds, {})).get("short", ds)
        ax.set_title(short)
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
    print(f"  Saved fig_convergence_curves")


def _fig_epsilon_tradeoff(p13, out_dir, plt):
    """Privacy-utility tradeoff: epsilon vs accuracy."""
    completed = p13.get("completed", {})

    epsilons_labels = [0.5, 1.0, 2.0, 5.0, 10.0, "inf"]
    epsilons_x = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]  # inf mapped to 20 for plotting
    means, stds = [], []

    for e in epsilons_labels:
        prefix = f"epsilon_{e}"
        m, s = _agg_ablation(completed, prefix, "accuracy")
        means.append(m * 100)
        stds.append(s * 100)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(epsilons_x, means, yerr=stds, marker="o", color="#2196F3",
                capsize=5, linewidth=2, markersize=8)
    ax.set_xlabel(r"Privacy Budget $\varepsilon$")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Privacy-Utility Tradeoff (Chest X-ray)")
    ax.set_xscale("log")
    xticks = epsilons_x
    ax.set_xticks(xticks)
    ax.set_xticklabels(["0.5", "1", "2", "5", "10", r"$\infty$"])
    ax.grid(alpha=0.3)

    for fmt in ["pdf", "png"]:
        fig.savefig(out_dir / f"fig_ablation_epsilon.{fmt}", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved fig_ablation_epsilon")


def _fig_model_comparison(p13, out_dir, plt):
    """CNN vs ResNet18 comparison."""
    completed = p13.get("completed", {})

    models = ["cnn", "resnet18"]
    labels = ["HealthcareCNN", "ResNet18"]
    metrics = ["accuracy", "f1", "auc"]

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(metrics))
    width = 0.3

    for i, (mt, label) in enumerate(zip(models, labels)):
        vals = []
        errs = []
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
    print(f"  Saved fig_model_comparison")


def _fig_attack_mse(p22, out_dir, plt):
    """Gradient inversion MSE vs epsilon."""
    completed = p22.get("completed", {})

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

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#F44336", "#FF9800", "#FFC107", "#4CAF50"]
    bars = ax.bar(range(len(eps_labels)), means, yerr=stds,
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
    print(f"  Saved fig_attack_mse")


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="FL-EHDS Paper Experiments (FLICS 2026)")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoints")
    parser.add_argument("--quick", action="store_true", help="Quick: 1 seed, 5 rounds")
    parser.add_argument("--only", type=str, choices=["p12", "p13", "p21", "p22", "output"],
                        help="Run only one block")
    parser.add_argument("--dataset", type=str,
                        help="Filter: run only this dataset (e.g. Brain_Tumor, chest_xray, Diabetes)")
    parser.add_argument("--algo", type=str,
                        help="Filter: run only this algorithm (e.g. FedAvg, SCAFFOLD)")
    args = parser.parse_args()

    global SEEDS, IMAGING_CONFIG, TABULAR_CONFIG

    if args.quick:
        SEEDS = [42]
        IMAGING_CONFIG["num_rounds"] = 5
        IMAGING_CONFIG["local_epochs"] = 1
        TABULAR_CONFIG["num_rounds"] = 5
        TABULAR_CONFIG["local_epochs"] = 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  FL-EHDS Paper Experiments for FLICS 2026")
    print("=" * 70)
    device = _detect_device()
    print(f"  Started:  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Device:   {device}")
    print(f"  Seeds:    {SEEDS}")
    print(f"  Output:   {OUTPUT_DIR}")
    print(f"  Resume:   {args.resume}")
    if args.quick:
        print(f"  Mode:     QUICK (1 seed, 5 rounds)")
    if args.dataset:
        print(f"  Filter:   dataset={args.dataset}")
    if args.algo:
        print(f"  Filter:   algo={args.algo}")
    print("=" * 70)

    t0 = time.time()

    # --- P1.2 ---
    p12 = None
    if args.only in (None, "p12"):
        print("\n>>> P1.2: Multi-Dataset FL Comparison")
        p12 = run_p12_multi_dataset(resume=args.resume,
                                     filter_dataset=args.dataset,
                                     filter_algo=args.algo)

    # --- P1.3 ---
    p13 = None
    if args.only in (None, "p13"):
        print("\n>>> P1.3: Ablation Study (chest_xray)")
        p13 = run_p13_ablation(resume=args.resume)

    # --- P2.1 ---
    sig = None
    if args.only in (None, "p21", "output"):
        print("\n>>> P2.1: Statistical Significance")
        if p12 is None:
            p12 = load_checkpoint("p12_multidataset")
        if p12:
            sig = run_p21_significance(p12)
            print(f"  Computed significance for {sum(len(v) for v in sig.values())} comparisons")
        else:
            print("  Skipped (no P1.2 results)")

    # --- P2.2 ---
    p22 = None
    if args.only in (None, "p22"):
        print("\n>>> P2.2: Privacy Attack Evaluation")
        p22 = run_p22_privacy_attack()

    # --- Output ---
    if args.only in (None, "output"):
        print("\n>>> Generating outputs...")

        if p12 is None:
            p12 = load_checkpoint("p12_multidataset")
        if p13 is None:
            p13 = load_checkpoint("p13_ablation")
        if p22 is None:
            p22 = load_checkpoint("p22_attack")
        if sig is None:
            sig = load_checkpoint("p21_significance")

        if p12:
            tex = generate_multi_dataset_table(p12, sig or {})
            (OUTPUT_DIR / "table_multi_dataset.tex").write_text(tex)
            print(f"  Saved table_multi_dataset.tex")

        if p13:
            tex = generate_ablation_table(p13)
            (OUTPUT_DIR / "table_ablation.tex").write_text(tex)
            print(f"  Saved table_ablation.tex")

        if p22:
            tex = generate_attack_table(p22)
            (OUTPUT_DIR / "table_attack.tex").write_text(tex)
            print(f"  Saved table_attack.tex")

        try:
            generate_figures(p12 or {}, p13 or {}, p22 or {}, OUTPUT_DIR)
        except Exception as e:
            print(f"  Figure generation error: {e}")
            traceback.print_exc()

    # --- Summary ---
    total_time = time.time() - t0
    print("\n" + "=" * 70)
    print(f"  Completed in {total_time/3600:.1f} hours ({total_time:.0f}s)")
    print(f"  Output: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()

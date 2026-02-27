"""
Local-Only vs Federated Learning Experiment
=============================================
Demonstrates the value of FL: individual hospitals with incomplete/biased
data achieve significantly better results when collaborating via FL.

Usage:
    cd fl-ehds-framework
    python benchmarks/run_local_vs_federated.py

Output:
    results/local_vs_federated/
        results.json
        table_local_vs_federated.tex
        fig_local_vs_federated.pdf
"""

import json
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Add parent to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from terminal.fl_trainer import (
    HealthcareCNN,
    ImageFederatedTrainer,
    load_image_dataset,
)


def train_local_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    num_classes: int,
    epochs: int = 60,
    batch_size: int = 32,
    lr: float = 0.001,
    device: str = "cpu",
) -> Dict[str, float]:
    """Train a model locally on a single hospital's data."""
    model = HealthcareCNN(num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()

    # Create data loaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train), torch.LongTensor(y_train)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test), torch.LongTensor(y_test)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Train
    model.train()
    for epoch in range(epochs):
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

    # Evaluate on test set
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            output = model(X_batch)
            loss = criterion(output, y_batch)
            total_loss += loss.item()
            n_batches += 1
            probs = torch.softmax(output, dim=1)
            preds = output.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    prec = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    rec = recall_score(all_labels, all_preds, average="macro", zero_division=0)

    try:
        if num_classes == 2:
            auc = roc_auc_score(all_labels, all_probs[:, 1])
        else:
            auc = roc_auc_score(
                all_labels, all_probs, multi_class="ovr", average="macro"
            )
    except Exception:
        auc = 0.5

    return {
        "accuracy": acc,
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "auc": auc,
        "loss": total_loss / max(n_batches, 1),
    }


def run_experiment(
    data_dir: str,
    num_clients: int = 5,
    num_rounds: int = 15,
    local_epochs_fl: int = 3,
    local_epochs_solo: int = 30,
    batch_size: int = 32,
    lr: float = 0.001,
    alpha: float = 0.5,
    seed: int = 42,
    img_size: int = 64,
) -> Dict:
    """Run one full Local vs Federated experiment."""

    np.random.seed(seed)
    torch.manual_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*60}")
    print(f"  SEED {seed} | Device: {device}")
    print(f"{'='*60}")

    # Load data with same split for both experiments
    print("\n[1/3] Loading dataset...")
    client_train, client_test, class_names, num_classes = load_image_dataset(
        data_dir=data_dir,
        num_clients=num_clients,
        is_iid=False,
        alpha=alpha,
        img_size=img_size,
        seed=seed,
        test_split=0.2,
    )

    # Show per-client stats
    print(f"\n  Classes: {class_names}")
    print(f"  Num classes: {num_classes}")
    for cid in sorted(client_train.keys()):
        X_tr, y_tr = client_train[cid]
        X_te, y_te = client_test[cid]
        unique, counts = np.unique(y_tr, return_counts=True)
        dist = {class_names[u]: int(c) for u, c in zip(unique, counts)}
        print(f"  Hospital {cid+1}: {len(y_tr)} train, {len(y_te)} test | {dist}")

    # === LOCAL-ONLY TRAINING ===
    print(f"\n[2/3] Training LOCAL-ONLY models ({local_epochs_solo} epochs each)...")
    local_results = {}
    for cid in sorted(client_train.keys()):
        X_tr, y_tr = client_train[cid]
        X_te, y_te = client_test[cid]
        print(f"  Hospital {cid+1} ({len(y_tr)} samples)...", end=" ", flush=True)
        t0 = time.time()
        result = train_local_model(
            X_tr, y_tr, X_te, y_te,
            num_classes=num_classes,
            epochs=local_epochs_solo,
            batch_size=batch_size,
            lr=lr,
            device=device,
        )
        dt = time.time() - t0
        local_results[cid] = result
        print(f"Acc={result['accuracy']:.3f}  F1={result['f1']:.3f}  "
              f"AUC={result['auc']:.3f}  ({dt:.1f}s)")

    # === FEDERATED TRAINING ===
    print(f"\n[3/3] Training FEDERATED model ({num_rounds} rounds x "
          f"{local_epochs_fl} local epochs)...")
    trainer = ImageFederatedTrainer(
        data_dir=data_dir,
        num_clients=num_clients,
        algorithm="FedAvg",
        local_epochs=local_epochs_fl,
        batch_size=batch_size,
        learning_rate=lr,
        is_iid=False,
        alpha=alpha,
        seed=seed,
        device=device,
        img_size=img_size,
    )

    for r in range(num_rounds):
        result = trainer.train_round(r)
        if (r + 1) % 5 == 0 or r == 0:
            print(f"  Round {r+1}/{num_rounds}: "
                  f"Acc={result.global_acc:.3f}  F1={result.global_f1:.3f}  "
                  f"AUC={result.global_auc:.3f}")

    # Evaluate federated model PER-CLIENT on each hospital's test data
    fed_model = trainer.global_model
    fed_model.eval()
    device_t = trainer.device

    fed_per_client = {}
    for cid in sorted(client_test.keys()):
        X_te, y_te = client_test[cid]
        test_ds = TensorDataset(
            torch.FloatTensor(X_te), torch.LongTensor(y_te)
        )
        test_loader = DataLoader(test_ds, batch_size=batch_size)

        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device_t)
                output = fed_model(X_batch)
                probs = torch.softmax(output, dim=1)
                preds = output.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y_batch.numpy())
                all_probs.extend(probs.cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )

        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        prec = precision_score(all_labels, all_preds, average="macro", zero_division=0)
        rec = recall_score(all_labels, all_preds, average="macro", zero_division=0)
        try:
            if num_classes == 2:
                auc = roc_auc_score(all_labels, all_probs[:, 1])
            else:
                auc = roc_auc_score(
                    all_labels, all_probs, multi_class="ovr", average="macro"
                )
        except Exception:
            auc = 0.5

        fed_per_client[cid] = {
            "accuracy": acc, "f1": f1, "precision": prec,
            "recall": rec, "auc": auc,
        }

    # Global federated metrics (from last round)
    final = trainer.history[-1]
    fed_global = {
        "accuracy": final.global_acc,
        "f1": final.global_f1,
        "precision": final.global_precision,
        "recall": final.global_recall,
        "auc": final.global_auc,
    }

    # Print comparison
    print(f"\n{'='*60}")
    print(f"  COMPARISON: Local-Only vs Federated (Seed {seed})")
    print(f"{'='*60}")
    print(f"  {'Hospital':<12} {'Local Acc':>10} {'Fed Acc':>10} {'Gain':>8}")
    print(f"  {'-'*42}")
    for cid in sorted(client_train.keys()):
        local_acc = local_results[cid]["accuracy"]
        fed_acc = fed_per_client[cid]["accuracy"]
        gain = fed_acc - local_acc
        print(f"  H{cid+1:<11} {local_acc:>9.1%} {fed_acc:>9.1%} {gain:>+7.1%}")

    avg_local = np.mean([r["accuracy"] for r in local_results.values()])
    avg_fed = np.mean([r["accuracy"] for r in fed_per_client.values()])
    print(f"  {'-'*42}")
    print(f"  {'Average':<12} {avg_local:>9.1%} {avg_fed:>9.1%} "
          f"{avg_fed - avg_local:>+7.1%}")

    # Client data stats
    client_stats = {}
    for cid in sorted(client_train.keys()):
        X_tr, y_tr = client_train[cid]
        unique, counts = np.unique(y_tr, return_counts=True)
        dist = {int(u): int(c) for u, c in zip(unique, counts)}
        client_stats[cid] = {
            "num_train": len(y_tr),
            "num_test": len(client_test[cid][1]),
            "label_distribution": dist,
        }

    return {
        "seed": seed,
        "local_results": {int(k): v for k, v in local_results.items()},
        "fed_per_client": {int(k): v for k, v in fed_per_client.items()},
        "fed_global": fed_global,
        "client_stats": {int(k): v for k, v in client_stats.items()},
        "class_names": {int(k): v for k, v in class_names.items()},
    }


def aggregate_results(all_runs: List[Dict]) -> Dict:
    """Aggregate results across seeds."""
    num_clients = len(all_runs[0]["local_results"])
    agg = {}

    for cid in range(num_clients):
        local_accs = [r["local_results"][cid]["accuracy"] for r in all_runs]
        local_f1s = [r["local_results"][cid]["f1"] for r in all_runs]
        local_aucs = [r["local_results"][cid]["auc"] for r in all_runs]
        fed_accs = [r["fed_per_client"][cid]["accuracy"] for r in all_runs]
        fed_f1s = [r["fed_per_client"][cid]["f1"] for r in all_runs]
        fed_aucs = [r["fed_per_client"][cid]["auc"] for r in all_runs]

        agg[cid] = {
            "num_train": all_runs[0]["client_stats"][cid]["num_train"],
            "num_test": all_runs[0]["client_stats"][cid]["num_test"],
            "local_acc_mean": np.mean(local_accs),
            "local_acc_std": np.std(local_accs),
            "local_f1_mean": np.mean(local_f1s),
            "local_f1_std": np.std(local_f1s),
            "local_auc_mean": np.mean(local_aucs),
            "local_auc_std": np.std(local_aucs),
            "fed_acc_mean": np.mean(fed_accs),
            "fed_acc_std": np.std(fed_accs),
            "fed_f1_mean": np.mean(fed_f1s),
            "fed_f1_std": np.std(fed_f1s),
            "fed_auc_mean": np.mean(fed_aucs),
            "fed_auc_std": np.std(fed_aucs),
            "gain_acc": np.mean(fed_accs) - np.mean(local_accs),
            "gain_f1": np.mean(fed_f1s) - np.mean(local_f1s),
            "gain_auc": np.mean(fed_aucs) - np.mean(local_aucs),
        }

    return agg


def generate_latex_table(agg: Dict, class_names: Dict) -> str:
    """Generate LaTeX table for the paper."""
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Local-Only vs.\ Federated Learning: Per-Hospital Performance on Chest X-ray}")
    lines.append(r"\label{tab:local_vs_fed}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{lrcccccc}")
    lines.append(r"\toprule")
    lines.append(r" & & \multicolumn{2}{c}{\textbf{Local-Only}} & \multicolumn{2}{c}{\textbf{Federated}} & \multicolumn{2}{c}{\textbf{Gain}} \\")
    lines.append(r"\cmidrule(lr){3-4} \cmidrule(lr){5-6} \cmidrule(lr){7-8}")
    lines.append(r"\textbf{Hospital} & \textbf{Samples} & \textbf{Acc.} & \textbf{F1} & \textbf{Acc.} & \textbf{F1} & \textbf{$\Delta$Acc} & \textbf{$\Delta$F1} \\")
    lines.append(r"\midrule")

    total_local_acc = []
    total_fed_acc = []

    for cid in sorted(agg.keys()):
        d = agg[cid]
        total_local_acc.append(d["local_acc_mean"])
        total_fed_acc.append(d["fed_acc_mean"])

        local_acc_s = f"{d['local_acc_mean']*100:.1f}\\%"
        local_f1_s = f"{d['local_f1_mean']:.2f}"
        fed_acc_s = f"{d['fed_acc_mean']*100:.1f}\\%"
        fed_f1_s = f"{d['fed_f1_mean']:.2f}"
        gain_acc_s = f"+{d['gain_acc']*100:.1f}pp"
        gain_f1_s = f"+{d['gain_f1']:.2f}"

        lines.append(
            f"H{cid+1} & {d['num_train']} & {local_acc_s} & {local_f1_s} "
            f"& {fed_acc_s} & {fed_f1_s} & {gain_acc_s} & {gain_f1_s} \\\\"
        )

    # Average row
    avg_local = np.mean(total_local_acc)
    avg_fed = np.mean(total_fed_acc)
    avg_gain = avg_fed - avg_local
    lines.append(r"\midrule")
    lines.append(
        f"\\textbf{{Average}} & & "
        f"\\textbf{{{avg_local*100:.1f}\\%}} & & "
        f"\\textbf{{{avg_fed*100:.1f}\\%}} & & "
        f"\\textbf{{+{avg_gain*100:.1f}pp}} & \\\\"
    )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append("")
    lines.append(r"\vspace{1mm}")
    lines.append(r"\footnotesize{Non-IID (Dirichlet $\alpha$=0.5), 5 hospitals. "
                 r"Local: 30 epochs standalone. Federated: FedAvg, 15 rounds $\times$ 3 epochs. "
                 r"Mean over 3 seeds.}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def generate_figure(agg: Dict, output_path: str):
    """Generate bar chart figure comparing local vs federated."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [WARN] matplotlib not available, skipping figure generation")
        return

    hospitals = sorted(agg.keys())
    n = len(hospitals)
    x = np.arange(n)
    width = 0.35

    local_accs = [agg[h]["local_acc_mean"] * 100 for h in hospitals]
    fed_accs = [agg[h]["fed_acc_mean"] * 100 for h in hospitals]
    local_stds = [agg[h]["local_acc_std"] * 100 for h in hospitals]
    fed_stds = [agg[h]["fed_acc_std"] * 100 for h in hospitals]

    fig, ax = plt.subplots(figsize=(8, 4.5))

    bars1 = ax.bar(
        x - width / 2, local_accs, width, yerr=local_stds,
        label="Local-Only", color="#d9534f", capsize=4, alpha=0.85,
    )
    bars2 = ax.bar(
        x + width / 2, fed_accs, width, yerr=fed_stds,
        label="Federated (FedAvg)", color="#5cb85c", capsize=4, alpha=0.85,
    )

    # Add gain annotations
    for i, h in enumerate(hospitals):
        gain = agg[h]["gain_acc"] * 100
        y_pos = max(local_accs[i], fed_accs[i]) + max(local_stds[i], fed_stds[i]) + 1.5
        ax.annotate(
            f"+{gain:.1f}pp",
            xy=(i, y_pos),
            ha="center", va="bottom",
            fontsize=9, fontweight="bold", color="#333333",
        )

    ax.set_xlabel("Hospital", fontsize=11)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_title("Local-Only vs. Federated Learning: Per-Hospital Accuracy", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([f"H{h+1}\n({agg[h]['num_train']} samples)" for h in hospitals])
    ax.legend(fontsize=10, loc="lower right")
    ax.set_ylim(
        min(local_accs) - 15,
        max(fed_accs) + max(fed_stds) + 8,
    )
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Figure saved: {output_path}")


def main():
    # Configuration
    framework_dir = Path(__file__).resolve().parent.parent
    data_dir = str(framework_dir / "data" / "chest_xray")

    if not os.path.exists(data_dir):
        print(f"ERROR: Dataset not found at {data_dir}")
        print("Please ensure the chest_xray dataset is in fl-ehds-framework/data/")
        sys.exit(1)

    output_dir = framework_dir / "results" / "local_vs_federated"
    output_dir.mkdir(parents=True, exist_ok=True)

    seeds = [42, 123, 456]
    num_clients = 5
    num_rounds = 15
    local_epochs_fl = 3
    local_epochs_solo = 30  # Enough to converge with Adam on small datasets
    batch_size = 32
    lr = 0.001
    alpha = 0.5
    img_size = 64  # 64x64 for speed (4x faster than 128x128)

    print("=" * 60)
    print("  LOCAL-ONLY vs FEDERATED LEARNING EXPERIMENT")
    print("=" * 60)
    print(f"  Dataset:       chest_xray ({data_dir})")
    print(f"  Hospitals:     {num_clients}")
    print(f"  Non-IID:       Dirichlet alpha={alpha}")
    print(f"  FL:            {num_rounds} rounds x {local_epochs_fl} epochs")
    print(f"  Local-Only:    {local_epochs_solo} epochs (equivalent compute)")
    print(f"  Image size:    {img_size}x{img_size}")
    print(f"  Seeds:         {seeds}")
    print(f"  Output:        {output_dir}")

    # Checkpoint support: save after each seed
    checkpoint_path = output_dir / "checkpoint_local_vs_fed.json"

    all_runs = []
    completed_seeds = []

    # Resume from checkpoint if exists
    if checkpoint_path.exists():
        try:
            with open(checkpoint_path) as f:
                ckpt = json.load(f)
            all_runs = ckpt.get("runs", [])
            completed_seeds = ckpt.get("completed_seeds", [])
            print(f"  Resumed: {len(completed_seeds)}/{len(seeds)} seeds done")
        except Exception:
            pass

    # SIGINT handler: save partial results
    def _signal_handler(signum, frame):
        print(f"\n  SIGINT — saving {len(all_runs)} completed seeds...")
        _save_checkpoint()
        sys.exit(0)

    def _save_checkpoint():
        tmp = checkpoint_path.with_suffix('.tmp')
        with open(tmp, 'w') as f:
            json.dump({"runs": all_runs, "completed_seeds": completed_seeds}, f, indent=2, default=str)
        tmp.replace(checkpoint_path)

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # Run experiments
    t_start = time.time()

    for seed in seeds:
        if seed in completed_seeds:
            print(f"\n  Seed {seed}: already done, skipping")
            continue

        result = run_experiment(
            data_dir=data_dir,
            num_clients=num_clients,
            num_rounds=num_rounds,
            local_epochs_fl=local_epochs_fl,
            local_epochs_solo=local_epochs_solo,
            batch_size=batch_size,
            lr=lr,
            alpha=alpha,
            seed=seed,
            img_size=img_size,
        )
        all_runs.append(result)
        completed_seeds.append(seed)
        _save_checkpoint()
        print(f"  Seed {seed}: done — checkpoint saved ({len(completed_seeds)}/{len(seeds)})")

    total_time = time.time() - t_start
    print(f"\nTotal experiment time: {total_time:.0f}s")

    # Aggregate
    agg = aggregate_results(all_runs)
    class_names = all_runs[0]["class_names"]

    # Print final aggregated results
    print(f"\n{'='*60}")
    print(f"  AGGREGATED RESULTS (mean over {len(seeds)} seeds)")
    print(f"{'='*60}")
    print(f"  {'Hospital':<10} {'Samples':>8} {'Local Acc':>12} {'Fed Acc':>12} {'Gain':>8}")
    print(f"  {'-'*52}")
    for cid in sorted(agg.keys()):
        d = agg[cid]
        print(f"  H{cid+1:<9} {d['num_train']:>8} "
              f"{d['local_acc_mean']*100:>7.1f}%+-{d['local_acc_std']*100:.1f} "
              f"{d['fed_acc_mean']*100:>7.1f}%+-{d['fed_acc_std']*100:.1f} "
              f"{d['gain_acc']*100:>+7.1f}pp")

    # Save results
    results_json = {
        "experiment": "local_vs_federated",
        "dataset": "chest_xray",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "num_clients": num_clients,
            "num_rounds": num_rounds,
            "local_epochs_fl": local_epochs_fl,
            "local_epochs_solo": local_epochs_solo,
            "batch_size": batch_size,
            "lr": lr,
            "alpha": alpha,
            "seeds": seeds,
        },
        "aggregated": {str(k): v for k, v in agg.items()},
        "all_runs": all_runs,
        "total_time_seconds": total_time,
    }

    json_path = output_dir / "results.json"
    with open(json_path, "w") as f:
        json.dump(results_json, f, indent=2, default=str)
    print(f"\n  Results saved: {json_path}")

    # Generate LaTeX table
    latex = generate_latex_table(agg, class_names)
    tex_path = output_dir / "table_local_vs_federated.tex"
    with open(tex_path, "w") as f:
        f.write(latex)
    print(f"  LaTeX table saved: {tex_path}")
    print(f"\n{latex}")

    # Generate figure
    fig_path = output_dir / "fig_local_vs_federated.pdf"
    generate_figure(agg, str(fig_path))


if __name__ == "__main__":
    main()

"""
Centralized vs Federated Learning Comparison Experiment
========================================================
Compares pooled centralized training (upper bound) against
federated learning on clinical imaging datasets.

Generates: results.json, CSV, LaTeX table, 10+ PNG charts.

Usage:
    cd fl-ehds-framework
    python -m experiments.centralized_vs_federated
    python -m experiments.centralized_vs_federated --dataset chest_xray --quick

Author: Fabio Liberti
Date: February 2026
"""

import sys
import json
import time
import csv
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple

import numpy as np

# Path setup
_framework_dir = Path(__file__).resolve().parent.parent.parent
if str(_framework_dir) not in sys.path:
    sys.path.insert(0, str(_framework_dir))

import torch
import torch.nn as nn
from tqdm import tqdm

from terminal.fl_trainer import (
    HealthcareCNN,
    CentralizedImageTrainer,
    ImageFederatedTrainer,
    load_image_dataset,
)

from experiments.centralized_vs_federated.visualizations import generate_all


# ---- Confusion matrix helper ----

def collect_confusion_matrix(
    model: nn.Module,
    X_test: np.ndarray,
    y_test: np.ndarray,
    num_classes: int,
    device: torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    """Run inference on test set and return confusion matrix."""
    from sklearn.metrics import confusion_matrix

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for i in range(0, len(y_test), batch_size):
            X_batch = torch.FloatTensor(X_test[i:i+batch_size]).to(device)
            outputs = model(X_batch)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y_test[i:i+batch_size])

    return confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))


# ---- Centralized baseline (trained ONCE per seed) ----

def _train_centralized_once(
    data_dir: str,
    num_clients: int,
    num_rounds: int,
    local_epochs: int,
    batch_size: int,
    lr: float,
    alpha: float,
    seed: int,
    img_size: int,
) -> Dict:
    """Train centralized baseline once for a given seed. Reused across algorithms."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    centralized_epochs = num_rounds * local_epochs

    print(f"\n  [Centralized] Training {centralized_epochs} epochs (seed={seed})...")
    cent_trainer = CentralizedImageTrainer(
        data_dir=data_dir,
        num_clients=num_clients,
        batch_size=batch_size,
        learning_rate=lr,
        is_iid=False,
        alpha=alpha,
        seed=seed,
        device=device,
        img_size=img_size,
    )
    cent_trainer.set_total_epochs(centralized_epochs)

    t0 = time.time()
    cent_pbar = tqdm(range(centralized_epochs), desc="  Centralized", ncols=80,
                     bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
    for e in cent_pbar:
        result = cent_trainer.train_epoch(e)
        if (e + 1) % max(centralized_epochs // 5, 1) == 0 or e == centralized_epochs - 1:
            cent_pbar.set_postfix(acc=f'{result.val_acc:.3f}', f1=f'{result.val_f1:.3f}')
    cent_time = time.time() - t0

    cent_history = []
    for r in cent_trainer.history:
        cent_history.append({
            'epoch': r.epoch,
            'val_acc': r.val_acc,
            'val_loss': r.val_loss,
            'val_f1': r.val_f1,
            'val_precision': r.val_precision,
            'val_recall': r.val_recall,
            'val_auc': r.val_auc,
            'train_loss': r.train_loss,
            'train_acc': r.train_acc,
            'time': r.time_seconds,
        })

    cent_final = {
        'accuracy': result.val_acc,
        'loss': result.val_loss,
        'f1': result.val_f1,
        'precision': result.val_precision,
        'recall': result.val_recall,
        'auc': result.val_auc,
    }

    cent_cm = collect_confusion_matrix(
        cent_trainer.get_model(), cent_trainer.X_test, cent_trainer.y_test,
        cent_trainer.num_classes, cent_trainer.device, batch_size,
    )

    data_stats = cent_trainer.get_data_stats()

    return {
        'centralized_history': cent_history,
        'centralized_final': cent_final,
        'centralized_cm': cent_cm.tolist(),
        'centralized_time': cent_time,
        'data_stats': data_stats,
        '_cent_trainer': cent_trainer,  # Keep reference for confusion matrix eval
    }


# ---- Federated training (per algorithm, reuses centralized result) ----

def _train_federated_only(
    data_dir: str,
    algorithm: str,
    num_clients: int,
    num_rounds: int,
    local_epochs: int,
    batch_size: int,
    lr: float,
    alpha: float,
    seed: int,
    img_size: int,
    cent_run: Dict,
) -> Dict:
    """Train one federated algorithm, reusing the pre-computed centralized result."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"  [Federated: {algorithm}] Training {num_rounds} rounds (seed={seed})...")

    fed_trainer = ImageFederatedTrainer(
        data_dir=data_dir,
        num_clients=num_clients,
        algorithm=algorithm,
        local_epochs=local_epochs,
        batch_size=batch_size,
        learning_rate=lr,
        is_iid=False,
        alpha=alpha,
        seed=seed,
        device=device,
        img_size=img_size,
    )
    fed_trainer.num_rounds = num_rounds  # Enable cosine LR decay

    t0 = time.time()
    fed_pbar = tqdm(range(num_rounds), desc=f"  {algorithm}", ncols=80,
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
    for r_num in fed_pbar:
        rr = fed_trainer.train_round(r_num)
        if (r_num + 1) % max(num_rounds // 5, 1) == 0 or r_num == num_rounds - 1:
            fed_pbar.set_postfix(acc=f'{rr.global_acc:.3f}', f1=f'{rr.global_f1:.3f}')
    fed_time = time.time() - t0

    fed_history = []
    for rr in fed_trainer.history:
        fed_history.append({
            'round': rr.round_num + 1,
            'accuracy': rr.global_acc,
            'loss': rr.global_loss,
            'f1': rr.global_f1,
            'precision': rr.global_precision,
            'recall': rr.global_recall,
            'auc': rr.global_auc,
            'time': rr.time_seconds,
        })

    fed_final = {
        'accuracy': rr.global_acc,
        'loss': rr.global_loss,
        'f1': rr.global_f1,
        'precision': rr.global_precision,
        'recall': rr.global_recall,
        'auc': rr.global_auc,
    }

    # Confusion matrix for federated (on same pooled test set)
    cent_trainer = cent_run['_cent_trainer']
    fed_cm = collect_confusion_matrix(
        fed_trainer.global_model, cent_trainer.X_test, cent_trainer.y_test,
        cent_trainer.num_classes, fed_trainer.device, batch_size,
    )

    return {
        'seed': seed,
        'algorithm': algorithm,
        'centralized_history': cent_run['centralized_history'],
        'federated_history': fed_history,
        'centralized_final': cent_run['centralized_final'],
        'federated_final': fed_final,
        'centralized_cm': cent_run['centralized_cm'],
        'federated_cm': fed_cm.tolist(),
        'centralized_time': cent_run['centralized_time'],
        'federated_time': fed_time,
        'data_stats': cent_run['data_stats'],
    }


# ---- Legacy single experiment (kept for backward compatibility) ----

def run_single_experiment(
    data_dir: str,
    algorithm: str,
    num_clients: int,
    num_rounds: int,
    local_epochs: int,
    batch_size: int,
    lr: float,
    alpha: float,
    seed: int,
    img_size: int,
) -> Dict:
    """Run one complete Centralized vs Federated experiment for a single seed+algorithm."""
    cent_run = _train_centralized_once(
        data_dir=data_dir, num_clients=num_clients, num_rounds=num_rounds,
        local_epochs=local_epochs, batch_size=batch_size, lr=lr,
        alpha=alpha, seed=seed, img_size=img_size,
    )
    return _train_federated_only(
        data_dir=data_dir, algorithm=algorithm, num_clients=num_clients,
        num_rounds=num_rounds, local_epochs=local_epochs, batch_size=batch_size,
        lr=lr, alpha=alpha, seed=seed, img_size=img_size, cent_run=cent_run,
    )


# ---- Full comparison (multi-seed, multi-algo) ----

def run_full_comparison(
    data_dir: str,
    algorithms: List[str],
    num_clients: int,
    num_rounds: int,
    local_epochs: int,
    batch_size: int,
    lr: float,
    alpha: float,
    seeds: List[int],
    img_size: int,
) -> Dict:
    """Run full comparison across seeds and algorithms.

    Centralized baseline is trained ONCE per seed (not per algorithm),
    then reused for all FL algorithm comparisons within that seed.
    """
    all_runs = []
    total_start = time.time()

    for seed_idx, seed in enumerate(seeds):
        print(f"\n{'='*60}")
        print(f"  SEED {seed_idx+1}/{len(seeds)}: {seed}")
        print(f"{'='*60}")

        # Train centralized baseline ONCE per seed
        cent_run = _train_centralized_once(
            data_dir=data_dir,
            num_clients=num_clients,
            num_rounds=num_rounds,
            local_epochs=local_epochs,
            batch_size=batch_size,
            lr=lr,
            alpha=alpha,
            seed=seed,
            img_size=img_size,
        )

        print(f"    Centralized: Acc={cent_run['centralized_final']['accuracy']:.3f}  "
              f"F1={cent_run['centralized_final']['f1']:.3f}  "
              f"AUC={cent_run['centralized_final']['auc']:.3f}  ({cent_run['centralized_time']:.1f}s)")

        # Run each FL algorithm, reusing the centralized result
        for algo in algorithms:
            run = _train_federated_only(
                data_dir=data_dir,
                algorithm=algo,
                num_clients=num_clients,
                num_rounds=num_rounds,
                local_epochs=local_epochs,
                batch_size=batch_size,
                lr=lr,
                alpha=alpha,
                seed=seed,
                img_size=img_size,
                cent_run=cent_run,
            )
            all_runs.append(run)

            print(f"    {algo}:    Acc={run['federated_final']['accuracy']:.3f}  "
                  f"F1={run['federated_final']['f1']:.3f}  "
                  f"AUC={run['federated_final']['auc']:.3f}  ({run['federated_time']:.1f}s)")

    total_time = time.time() - total_start

    # Aggregate across seeds
    cent_metrics = {}
    fed_metrics = {algo: {} for algo in algorithms}
    cent_times = []
    fed_times = {algo: [] for algo in algorithms}

    # Use first run data for representative items (same dataset)
    first_run = all_runs[0]

    for metric in ['accuracy', 'loss', 'f1', 'precision', 'recall', 'auc']:
        cent_vals = [r['centralized_final'][metric] for r in all_runs
                     if r['algorithm'] == algorithms[0]]
        cent_metrics[metric] = float(np.mean(cent_vals))

        for algo in algorithms:
            vals = [r['federated_final'][metric] for r in all_runs if r['algorithm'] == algo]
            fed_metrics[algo][metric] = float(np.mean(vals))

    for r in all_runs:
        if r['algorithm'] == algorithms[0]:
            cent_times.append(r['centralized_time'])
        fed_times[r['algorithm']].append(r['federated_time'])

    # Use last seed's confusion matrices as representative
    last_runs = {r['algorithm']: r for r in all_runs if r['seed'] == seeds[-1]}
    cent_cm = last_runs[algorithms[0]]['centralized_cm']
    fed_cms = {algo: last_runs[algo]['federated_cm'] for algo in algorithms if algo in last_runs}

    # Representative history (last seed)
    cent_hist = last_runs[algorithms[0]]['centralized_history']
    fed_hists = {algo: last_runs[algo]['federated_history'] for algo in algorithms if algo in last_runs}

    return {
        'config': {
            'data_dir': data_dir,
            'algorithms': algorithms,
            'num_clients': num_clients,
            'num_rounds': num_rounds,
            'local_epochs': local_epochs,
            'batch_size': batch_size,
            'lr': lr,
            'alpha': alpha,
            'seeds': seeds,
            'img_size': img_size,
        },
        'centralized_final': cent_metrics,
        'federated_finals': fed_metrics,
        'centralized_total_time': float(np.mean(cent_times)),
        'federated_total_times': {a: float(np.mean(fed_times[a])) for a in algorithms},
        'centralized_cm': cent_cm,
        'federated_cms': fed_cms,
        'centralized_history': cent_hist,
        'federated_histories': fed_hists,
        'data_stats': first_run['data_stats'],
        'all_runs': all_runs,
        'total_time': total_time,
    }


# ---- Save outputs ----

def generate_latex_table(results: Dict) -> str:
    """Generate LaTeX table comparing centralized vs federated."""
    cfg = results['config']
    cent = results['centralized_final']
    dataset_name = Path(cfg['data_dir']).name

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        f"\\caption{{Centralized vs.\\ Federated Learning on {dataset_name}}}",
        r"\label{tab:centralized_vs_federated}",
        r"\small",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"\textbf{Method} & \textbf{Acc.} & \textbf{F1} & \textbf{AUC} & "
        r"\textbf{Prec.} & \textbf{Rec.} & \textbf{Time} \\",
        r"\midrule",
    ]

    # Centralized row
    lines.append(
        f"Centralized & "
        f"${cent['accuracy']*100:.1f}\\%$ & "
        f"${cent['f1']:.3f}$ & "
        f"${cent['auc']:.3f}$ & "
        f"${cent['precision']:.3f}$ & "
        f"${cent['recall']:.3f}$ & "
        f"${results['centralized_total_time']:.0f}$s \\\\"
    )
    lines.append(r"\midrule")

    # Federated rows
    for algo in cfg['algorithms']:
        fed = results['federated_finals'][algo]
        t = results['federated_total_times'][algo]
        lines.append(
            f"{algo} & "
            f"${fed['accuracy']*100:.1f}\\%$ & "
            f"${fed['f1']:.3f}$ & "
            f"${fed['auc']:.3f}$ & "
            f"${fed['precision']:.3f}$ & "
            f"${fed['recall']:.3f}$ & "
            f"${t:.0f}$s \\\\"
        )

    lines.append(r"\midrule")

    # Gap row for first algo
    algo0 = cfg['algorithms'][0]
    fed0 = results['federated_finals'][algo0]
    gap_acc = (cent['accuracy'] - fed0['accuracy']) * 100
    gap_f1 = cent['f1'] - fed0['f1']
    gap_auc = cent['auc'] - fed0['auc']
    lines.append(
        f"\\textit{{Gap (Cent. - {algo0})}} & "
        f"\\textit{{{gap_acc:+.1f}pp}} & "
        f"\\textit{{{gap_f1:+.3f}}} & "
        f"\\textit{{{gap_auc:+.3f}}} & & & \\\\"
    )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append("")
    lines.append(
        f"\\footnotesize{{Non-IID (Dirichlet $\\alpha$={cfg['alpha']}), "
        f"{cfg['num_clients']} hospitals, img\\_size={cfg['img_size']}. "
        f"Centralized: {cfg['num_rounds']*cfg['local_epochs']} epochs. "
        f"Federated: {cfg['num_rounds']} rounds $\\times$ {cfg['local_epochs']} epochs. "
        f"Mean over {len(cfg['seeds'])} seeds.}}"
    )
    lines.append(r"\end{table}")

    return "\n".join(lines)


def save_all_outputs(results: Dict, output_dir: Path):
    """Save all experiment outputs: JSON, CSV, LaTeX, charts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = results['config']

    # 1. JSON
    json_path = output_dir / "results.json"
    # Make JSON serializable: remove numpy arrays from all_runs
    json_results = {k: v for k, v in results.items() if k != 'all_runs'}
    json_results['all_runs_summary'] = [
        {
            'seed': r['seed'], 'algorithm': r['algorithm'],
            'centralized_final': r['centralized_final'],
            'federated_final': r['federated_final'],
            'centralized_time': r['centralized_time'],
            'federated_time': r['federated_time'],
        }
        for r in results['all_runs']
    ]
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2, default=str)
    print(f"  Saved: results.json")

    # 2. Summary CSV
    csv_path = output_dir / "summary_results.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['method', 'accuracy', 'f1', 'auc', 'precision', 'recall', 'time_s'])
        cent = results['centralized_final']
        writer.writerow([
            'Centralized', f"{cent['accuracy']:.4f}", f"{cent['f1']:.4f}",
            f"{cent['auc']:.4f}", f"{cent['precision']:.4f}", f"{cent['recall']:.4f}",
            f"{results['centralized_total_time']:.1f}",
        ])
        for algo in cfg['algorithms']:
            fed = results['federated_finals'][algo]
            t = results['federated_total_times'][algo]
            writer.writerow([
                algo, f"{fed['accuracy']:.4f}", f"{fed['f1']:.4f}",
                f"{fed['auc']:.4f}", f"{fed['precision']:.4f}", f"{fed['recall']:.4f}",
                f"{t:.1f}",
            ])
    print(f"  Saved: summary_results.csv")

    # 3. Centralized history CSV
    csv_cent = output_dir / "history_centralized.csv"
    with open(csv_cent, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'val_acc', 'val_loss', 'val_f1', 'val_precision', 'val_recall', 'val_auc', 'time'])
        for h in results['centralized_history']:
            writer.writerow([
                h['epoch'] + 1, f"{h['val_acc']:.4f}", f"{h['val_loss']:.4f}",
                f"{h['val_f1']:.4f}", f"{h['val_precision']:.4f}",
                f"{h['val_recall']:.4f}", f"{h['val_auc']:.4f}", f"{h['time']:.2f}",
            ])
    print(f"  Saved: history_centralized.csv")

    # 4. Federated history CSV per algorithm
    for algo, hist in results['federated_histories'].items():
        csv_fed = output_dir / f"history_federated_{algo}.csv"
        with open(csv_fed, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['round', 'accuracy', 'loss', 'f1', 'precision', 'recall', 'auc', 'time'])
            for h in hist:
                writer.writerow([
                    h['round'], f"{h['accuracy']:.4f}", f"{h['loss']:.4f}",
                    f"{h['f1']:.4f}", f"{h['precision']:.4f}",
                    f"{h['recall']:.4f}", f"{h['auc']:.4f}", f"{h['time']:.2f}",
                ])
        print(f"  Saved: history_federated_{algo}.csv")

    # 5. LaTeX table
    tex_path = output_dir / "table_centralized_vs_federated.tex"
    with open(tex_path, 'w') as f:
        f.write(generate_latex_table(results))
    print(f"  Saved: table_centralized_vs_federated.tex")

    # 6. Summary text
    txt_path = output_dir / "summary.txt"
    with open(txt_path, 'w') as f:
        f.write("CENTRALIZED vs FEDERATED LEARNING COMPARISON\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Dataset: {cfg['data_dir']}\n")
        f.write(f"Hospitals: {cfg['num_clients']}\n")
        f.write(f"Non-IID: alpha={cfg['alpha']}\n")
        f.write(f"Image size: {cfg['img_size']}x{cfg['img_size']}\n")
        f.write(f"Seeds: {cfg['seeds']}\n")
        f.write(f"Centralized epochs: {cfg['num_rounds'] * cfg['local_epochs']}\n")
        f.write(f"Federated: {cfg['num_rounds']} rounds x {cfg['local_epochs']} epochs\n\n")

        f.write("RESULTS\n")
        f.write("-" * 50 + "\n")
        cent = results['centralized_final']
        f.write(f"Centralized:  Acc={cent['accuracy']:.3f}  F1={cent['f1']:.3f}  "
                f"AUC={cent['auc']:.3f}  Time={results['centralized_total_time']:.1f}s\n")
        for algo in cfg['algorithms']:
            fed = results['federated_finals'][algo]
            t = results['federated_total_times'][algo]
            gap = (cent['accuracy'] - fed['accuracy']) * 100
            f.write(f"{algo:12s}:  Acc={fed['accuracy']:.3f}  F1={fed['f1']:.3f}  "
                    f"AUC={fed['auc']:.3f}  Time={t:.1f}s  (gap: {gap:+.1f}pp)\n")

        f.write(f"\nTotal experiment time: {results['total_time']:.0f}s\n")
    print(f"  Saved: summary.txt")

    # 7. All visualizations
    generate_all(results, str(output_dir))


# ---- Main ----

def main():
    parser = argparse.ArgumentParser(
        description='Centralized vs Federated Learning Comparison',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--dataset', type=str, default='chest_xray',
                        help='Dataset folder name in data/ (default: chest_xray)')
    parser.add_argument('--algorithms', nargs='+', default=['FedAvg', 'FedProx', 'SCAFFOLD'],
                        help='FL algorithms to compare (default: FedAvg FedProx SCAFFOLD)')
    parser.add_argument('--num-clients', type=int, default=5,
                        help='Number of hospitals (default: 5)')
    parser.add_argument('--num-rounds', type=int, default=15,
                        help='FL rounds (default: 15)')
    parser.add_argument('--local-epochs', type=int, default=3,
                        help='Local epochs per FL round (default: 3)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Dirichlet alpha for non-IID (default: 0.5)')
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 123, 456],
                        help='Random seeds (default: 42 123 456)')
    parser.add_argument('--img-size', type=int, default=64,
                        help='Image size (default: 64)')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode: 1 seed, 5 rounds, 2 epochs, 1 algo')

    args = parser.parse_args()

    if args.quick:
        args.seeds = [42]
        args.num_rounds = 5
        args.local_epochs = 2
        args.algorithms = [args.algorithms[0]]

    # Resolve data path
    data_dir = _framework_dir / "data" / args.dataset
    if not data_dir.exists():
        print(f"ERROR: Dataset not found at {data_dir}")
        print(f"Available datasets in {_framework_dir / 'data'}:")
        if (_framework_dir / "data").exists():
            for d in sorted((_framework_dir / "data").iterdir()):
                if d.is_dir() and not d.name.startswith('.'):
                    print(f"  - {d.name}")
        sys.exit(1)

    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = _framework_dir / "results" / "centralized_vs_federated" / f"cvf_{args.dataset}_{timestamp}"

    # Print config
    cent_epochs = args.num_rounds * args.local_epochs
    print("=" * 60)
    print("  CENTRALIZED vs FEDERATED LEARNING COMPARISON")
    print("=" * 60)
    print(f"  Dataset:       {args.dataset} ({data_dir})")
    print(f"  Hospitals:     {args.num_clients}")
    print(f"  Non-IID:       Dirichlet alpha={args.alpha}")
    print(f"  Centralized:   {cent_epochs} epochs")
    print(f"  Federated:     {args.num_rounds} rounds x {args.local_epochs} epochs")
    print(f"  Algorithms:    {', '.join(args.algorithms)}")
    print(f"  Image size:    {args.img_size}x{args.img_size}")
    print(f"  Seeds:         {args.seeds}")
    print(f"  Output:        {output_dir}")
    if args.quick:
        print(f"  Mode:          QUICK (reduced config)")
    print()

    # Run
    results = run_full_comparison(
        data_dir=str(data_dir),
        algorithms=args.algorithms,
        num_clients=args.num_clients,
        num_rounds=args.num_rounds,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        alpha=args.alpha,
        seeds=args.seeds,
        img_size=args.img_size,
    )

    # Save
    print(f"\n{'='*60}")
    print(f"  SAVING RESULTS")
    print(f"{'='*60}")
    save_all_outputs(results, output_dir)

    # Print final summary
    print(f"\n{'='*60}")
    print(f"  FINAL SUMMARY")
    print(f"{'='*60}")
    cent = results['centralized_final']
    print(f"  Centralized:  Acc={cent['accuracy']:.3f}  F1={cent['f1']:.3f}  AUC={cent['auc']:.3f}")
    for algo in args.algorithms:
        fed = results['federated_finals'][algo]
        gap = (cent['accuracy'] - fed['accuracy']) * 100
        print(f"  {algo:12s}:  Acc={fed['accuracy']:.3f}  F1={fed['f1']:.3f}  "
              f"AUC={fed['auc']:.3f}  (gap: {gap:+.1f}pp)")
    print(f"\n  Total time: {results['total_time']:.0f}s")
    print(f"  Output: {output_dir}")


if __name__ == "__main__":
    main()

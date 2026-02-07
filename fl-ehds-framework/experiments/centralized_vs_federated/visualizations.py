"""
Visualization suite for Centralized vs Federated comparison.
Generates comprehensive charts for paper and analysis.

Author: Fabio Liberti
Date: February 2026
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional


# Consistent color scheme
CENTRALIZED_COLOR = '#2196F3'  # Blue
FEDERATED_COLORS = {
    'FedAvg': '#4CAF50',
    'FedProx': '#FF9800',
    'SCAFFOLD': '#9C27B0',
    'FedNova': '#F44336',
    'FedAdam': '#00BCD4',
    'FedYogi': '#795548',
    'FedAdagrad': '#607D8B',
    'Per-FedAvg': '#E91E63',
    'Ditto': '#3F51B5',
}

METRIC_LABELS = {
    'accuracy': 'Accuracy',
    'loss': 'Loss',
    'f1': 'F1 Score',
    'precision': 'Precision',
    'recall': 'Recall',
    'auc': 'AUC-ROC',
}


def _get_color(algo: str) -> str:
    return FEDERATED_COLORS.get(algo, '#888888')


def _save(fig, output_dir: Path, name: str):
    path = output_dir / name
    fig.savefig(str(path), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {name}")


def plot_convergence_all_metrics(
    centralized_history: List[Dict],
    federated_histories: Dict[str, List[Dict]],
    local_epochs: int,
    output_dir: Path,
):
    """6-panel grid: Accuracy, Loss, F1, Precision, Recall, AUC convergence."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Convergence: Centralized vs Federated Learning', fontsize=14, fontweight='bold')

    metrics = ['accuracy', 'loss', 'f1', 'precision', 'recall', 'auc']
    keys_cent = ['val_acc', 'val_loss', 'val_f1', 'val_precision', 'val_recall', 'val_auc']

    for idx, (metric, key_c) in enumerate(zip(metrics, keys_cent)):
        ax = axes[idx // 3][idx % 3]

        # Centralized: x = epoch number
        c_x = [h['epoch'] + 1 for h in centralized_history]
        c_y = [h[key_c] for h in centralized_history]
        ax.plot(c_x, c_y, color=CENTRALIZED_COLOR, linewidth=2, label='Centralized')

        # Federated: x = round * local_epochs (equivalent epoch)
        for algo, hist in federated_histories.items():
            f_x = [(h['round']) * local_epochs for h in hist]
            f_y = [h[metric] for h in hist]
            ax.plot(f_x, f_y, color=_get_color(algo), linewidth=1.5,
                    linestyle='--', marker='o', markersize=3, label=algo)

        ax.set_title(METRIC_LABELS[metric], fontsize=11)
        ax.set_xlabel('Equivalent Epoch')
        ax.grid(True, alpha=0.3)

        if idx == 0:
            ax.legend(fontsize=8, loc='lower right')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, output_dir, 'plot_convergence_all_metrics.png')


def plot_convergence_accuracy_loss(
    centralized_history: List[Dict],
    federated_histories: Dict[str, List[Dict]],
    local_epochs: int,
    output_dir: Path,
):
    """Large 2-panel: Accuracy (left) and Loss (right) for paper."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Centralized
    c_x = [h['epoch'] + 1 for h in centralized_history]
    c_acc = [h['val_acc'] for h in centralized_history]
    c_loss = [h['val_loss'] for h in centralized_history]

    ax1.plot(c_x, c_acc, color=CENTRALIZED_COLOR, linewidth=2.5, label='Centralized (upper bound)')
    ax2.plot(c_x, c_loss, color=CENTRALIZED_COLOR, linewidth=2.5, label='Centralized (upper bound)')

    for algo, hist in federated_histories.items():
        f_x = [(h['round']) * local_epochs for h in hist]
        f_acc = [h['accuracy'] for h in hist]
        f_loss = [h['loss'] for h in hist]
        ax1.plot(f_x, f_acc, color=_get_color(algo), linewidth=2,
                 linestyle='--', marker='o', markersize=4, label=algo)
        ax2.plot(f_x, f_loss, color=_get_color(algo), linewidth=2,
                 linestyle='--', marker='o', markersize=4, label=algo)

    ax1.set_title('Accuracy Convergence', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Equivalent Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.set_title('Loss Convergence', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Equivalent Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    _save(fig, output_dir, 'plot_convergence_accuracy_loss.png')


def plot_data_distribution_per_node(
    data_stats: Dict,
    output_dir: Path,
):
    """Stacked bar chart: class distribution per hospital."""
    per_client = data_stats['per_client']
    class_names = data_stats['class_names']
    num_clients = len(per_client)
    num_classes = data_stats['num_classes']

    # Build matrix: clients x classes
    matrix = np.zeros((num_clients, num_classes))
    client_ids = sorted(per_client.keys())
    for i, cid in enumerate(client_ids):
        dist = per_client[cid]['class_distribution']
        for cls_id, count in dist.items():
            matrix[i, cls_id] = count

    fig, ax = plt.subplots(figsize=(max(8, num_clients * 1.5), 6))

    x = np.arange(num_clients)
    width = 0.6
    colors = plt.cm.Set2(np.linspace(0, 1, num_classes))

    bottom = np.zeros(num_clients)
    for cls_idx in range(num_classes):
        label = class_names.get(cls_idx, f'Class {cls_idx}') if isinstance(class_names, dict) else f'Class {cls_idx}'
        ax.bar(x, matrix[:, cls_idx], width, bottom=bottom, label=label, color=colors[cls_idx])
        bottom += matrix[:, cls_idx]

    # Total annotation on top of each bar
    for i in range(num_clients):
        total = int(bottom[i])
        ax.text(i, bottom[i] + 5, str(total), ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xlabel('Hospital', fontsize=11)
    ax.set_ylabel('Number of Training Samples', fontsize=11)
    ax.set_title('Data Distribution per Hospital (Non-IID)', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'H{cid+1}' for cid in client_ids])
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.2, axis='y')

    plt.tight_layout()
    _save(fig, output_dir, 'plot_data_distribution_per_node.png')


def plot_node_contribution(
    data_stats: Dict,
    output_dir: Path,
):
    """Horizontal bar chart: each hospital's data contribution percentage."""
    per_client = data_stats['per_client']
    total_train = data_stats['total_train']
    client_ids = sorted(per_client.keys())

    labels = [f'Hospital {cid+1}' for cid in client_ids]
    train_counts = [per_client[cid]['train_samples'] for cid in client_ids]
    test_counts = [per_client[cid]['test_samples'] for cid in client_ids]
    percentages = [c / total_train * 100 for c in train_counts]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, max(4, len(client_ids) * 0.8)))

    # Left: horizontal bar
    colors = plt.cm.Paired(np.linspace(0, 1, len(client_ids)))
    y_pos = np.arange(len(client_ids))
    bars = ax1.barh(y_pos, train_counts, color=colors, edgecolor='white')
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels)
    ax1.set_xlabel('Training Samples')
    ax1.set_title('Data Contribution per Hospital', fontsize=12, fontweight='bold')
    for i, (count, pct) in enumerate(zip(train_counts, percentages)):
        ax1.text(count + 10, i, f'{count} ({pct:.1f}%)', va='center', fontsize=9)
    ax1.grid(True, alpha=0.2, axis='x')

    # Right: summary table
    ax2.axis('off')
    table_data = []
    for cid in client_ids:
        cls_dist = per_client[cid]['class_distribution']
        dominant = max(cls_dist.values()) / sum(cls_dist.values()) * 100 if cls_dist else 0
        table_data.append([
            f'H{cid+1}',
            str(per_client[cid]['train_samples']),
            str(per_client[cid]['test_samples']),
            f'{dominant:.0f}%',
        ])
    table = ax2.table(
        cellText=table_data,
        colLabels=['Hospital', 'Train', 'Test', 'Dominant Class %'],
        cellLoc='center',
        loc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    ax2.set_title('Per-Hospital Summary', fontsize=12, fontweight='bold', pad=20)

    plt.tight_layout()
    _save(fig, output_dir, 'plot_node_contribution.png')


def plot_training_time(
    centralized_total_time: float,
    federated_times: Dict[str, float],
    output_dir: Path,
):
    """Grouped bar chart: training time comparison."""
    fig, ax = plt.subplots(figsize=(max(8, (1 + len(federated_times)) * 2), 5))

    labels = ['Centralized'] + list(federated_times.keys())
    times = [centralized_total_time] + list(federated_times.values())
    colors = [CENTRALIZED_COLOR] + [_get_color(a) for a in federated_times.keys()]

    bars = ax.bar(range(len(labels)), times, color=colors, edgecolor='white', width=0.6)

    for i, (bar, t) in enumerate(zip(bars, times)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f'{t:.1f}s', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel('Total Training Time (seconds)', fontsize=11)
    ax.set_title('Training Time: Centralized vs Federated', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.2, axis='y')

    plt.tight_layout()
    _save(fig, output_dir, 'plot_training_time.png')


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Dict,
    title: str,
    output_dir: Path,
    filename: str,
    cmap: str = 'Blues',
):
    """Single confusion matrix heatmap."""
    num_classes = cm.shape[0]
    fig, ax = plt.subplots(figsize=(max(6, num_classes * 1.5), max(5, num_classes * 1.2)))

    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    labels = [class_names.get(i, f'Class {i}') if isinstance(class_names, dict)
              else f'Class {i}' for i in range(num_classes)]
    ax.set(xticks=np.arange(num_classes), yticks=np.arange(num_classes),
           xticklabels=labels, yticklabels=labels,
           ylabel='True Label', xlabel='Predicted Label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    # Annotations: count + percentage
    total = cm.sum()
    thresh = cm.max() / 2.0
    for i in range(num_classes):
        for j in range(num_classes):
            pct = cm[i, j] / max(cm[i].sum(), 1) * 100
            ax.text(j, i, f'{cm[i, j]}\n({pct:.1f}%)',
                    ha='center', va='center', fontsize=9,
                    color='white' if cm[i, j] > thresh else 'black')

    ax.set_title(title, fontsize=12, fontweight='bold')
    plt.tight_layout()
    _save(fig, output_dir, filename)


def plot_confusion_matrices_side_by_side(
    cm_centralized: np.ndarray,
    cm_federated: np.ndarray,
    class_names: Dict,
    algo_name: str,
    output_dir: Path,
):
    """Two confusion matrices side by side."""
    num_classes = cm_centralized.shape[0]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(12, num_classes * 3), max(5, num_classes * 1.2)))

    labels = [class_names.get(i, f'Class {i}') if isinstance(class_names, dict)
              else f'Class {i}' for i in range(num_classes)]

    for ax, cm, title, cmap in [
        (ax1, cm_centralized, 'Centralized', 'Blues'),
        (ax2, cm_federated, f'Federated ({algo_name})', 'Greens'),
    ]:
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set(xticks=np.arange(num_classes), yticks=np.arange(num_classes),
               xticklabels=labels, yticklabels=labels,
               ylabel='True Label', xlabel='Predicted Label')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

        thresh = cm.max() / 2.0
        for i in range(num_classes):
            for j in range(num_classes):
                pct = cm[i, j] / max(cm[i].sum(), 1) * 100
                ax.text(j, i, f'{cm[i, j]}\n({pct:.1f}%)',
                        ha='center', va='center', fontsize=9,
                        color='white' if cm[i, j] > thresh else 'black')
        ax.set_title(title, fontsize=12, fontweight='bold')

    plt.tight_layout()
    _save(fig, output_dir, f'plot_confusion_comparison_{algo_name}.png')


def plot_metrics_comparison_bar(
    centralized_metrics: Dict[str, float],
    federated_metrics: Dict[str, Dict[str, float]],
    output_dir: Path,
):
    """Grouped bar chart: final metrics comparison across all methods."""
    metrics = ['accuracy', 'f1', 'auc', 'precision', 'recall']
    algos = list(federated_metrics.keys())
    n_methods = 1 + len(algos)
    n_metrics = len(metrics)

    fig, ax = plt.subplots(figsize=(max(10, n_metrics * 2.5), 6))

    x = np.arange(n_metrics)
    width = 0.8 / n_methods

    # Centralized bars
    c_vals = [centralized_metrics.get(m, 0) for m in metrics]
    ax.bar(x - width * (n_methods - 1) / 2, c_vals, width,
           label='Centralized', color=CENTRALIZED_COLOR, edgecolor='white')

    # Federated bars
    for i, algo in enumerate(algos):
        f_vals = [federated_metrics[algo].get(m, 0) for m in metrics]
        offset = x - width * (n_methods - 1) / 2 + width * (i + 1)
        ax.bar(offset, f_vals, width, label=algo,
               color=_get_color(algo), edgecolor='white')

    ax.set_ylabel('Score', fontsize=11)
    ax.set_title('Final Metrics: Centralized vs Federated', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_LABELS[m] for m in metrics], fontsize=10)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.2, axis='y')

    plt.tight_layout()
    _save(fig, output_dir, 'plot_metrics_comparison_bar.png')


def plot_gap_analysis(
    centralized_metrics: Dict[str, float],
    federated_metrics: Dict[str, Dict[str, float]],
    output_dir: Path,
):
    """Horizontal bar chart showing gap (centralized - federated) per metric."""
    metrics = ['accuracy', 'f1', 'auc', 'precision', 'recall']
    algos = list(federated_metrics.keys())

    fig, ax = plt.subplots(figsize=(10, max(4, len(algos) * len(metrics) * 0.35)))

    labels = []
    gaps = []
    colors = []

    for algo in algos:
        for m in metrics:
            c_val = centralized_metrics.get(m, 0)
            f_val = federated_metrics[algo].get(m, 0)
            gap = (c_val - f_val) * 100  # percentage points
            labels.append(f'{algo} / {METRIC_LABELS[m]}')
            gaps.append(gap)
            # Green = small gap (<2pp), yellow = moderate, red = large (>5pp)
            if abs(gap) < 2:
                colors.append('#4CAF50')
            elif abs(gap) < 5:
                colors.append('#FF9800')
            else:
                colors.append('#F44336')

    y_pos = np.arange(len(labels))
    ax.barh(y_pos, gaps, color=colors, edgecolor='white', height=0.7)

    for i, g in enumerate(gaps):
        ax.text(g + 0.2 if g >= 0 else g - 0.2, i,
                f'{g:+.1f}pp', va='center', fontsize=8,
                ha='left' if g >= 0 else 'right')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel('Gap (Centralized - Federated) in percentage points', fontsize=10)
    ax.set_title('Performance Gap Analysis', fontsize=12, fontweight='bold')
    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.grid(True, alpha=0.2, axis='x')

    plt.tight_layout()
    _save(fig, output_dir, 'plot_gap_analysis.png')


def generate_all(results: Dict, output_dir: str):
    """Master function: generate all visualizations from experiment results."""
    out = Path(output_dir)

    print("\n  Generating visualizations...")

    # 1. Convergence all metrics
    if results.get('centralized_history') and results.get('federated_histories'):
        plot_convergence_all_metrics(
            results['centralized_history'],
            results['federated_histories'],
            results['config']['local_epochs'],
            out,
        )

        # 2. Convergence accuracy+loss (paper-friendly)
        plot_convergence_accuracy_loss(
            results['centralized_history'],
            results['federated_histories'],
            results['config']['local_epochs'],
            out,
        )

    # 3. Data distribution per node
    if results.get('data_stats'):
        plot_data_distribution_per_node(results['data_stats'], out)

        # 4. Node contribution
        plot_node_contribution(results['data_stats'], out)

    # 5. Training time
    if results.get('centralized_total_time') and results.get('federated_total_times'):
        plot_training_time(
            results['centralized_total_time'],
            results['federated_total_times'],
            out,
        )

    # 6-8. Confusion matrices
    class_names = results.get('data_stats', {}).get('class_names', {})
    if results.get('centralized_cm') is not None:
        cm_c = np.array(results['centralized_cm'])
        plot_confusion_matrix(cm_c, class_names, 'Centralized Model',
                              out, 'plot_confusion_matrix_centralized.png', 'Blues')

    for algo, cm_f in results.get('federated_cms', {}).items():
        cm_f = np.array(cm_f)
        plot_confusion_matrix(cm_f, class_names, f'Federated ({algo})',
                              out, f'plot_confusion_matrix_federated_{algo}.png', 'Greens')

        if results.get('centralized_cm') is not None:
            plot_confusion_matrices_side_by_side(
                np.array(results['centralized_cm']), cm_f,
                class_names, algo, out,
            )

    # 9. Final metrics comparison bar
    if results.get('centralized_final') and results.get('federated_finals'):
        plot_metrics_comparison_bar(
            results['centralized_final'],
            results['federated_finals'],
            out,
        )

        # 10. Gap analysis
        plot_gap_analysis(
            results['centralized_final'],
            results['federated_finals'],
            out,
        )

    print(f"  All visualizations saved to: {out}")

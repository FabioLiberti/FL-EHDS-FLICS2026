#!/usr/bin/env python3
"""
FL-EHDS Algorithm Comparison Benchmark

Generates comparison figures for the paper:
1. FedAvg vs FedProx vs SCAFFOLD vs FedAdam at different non-IID levels
2. Convergence speed comparison
3. Communication efficiency
4. Privacy-accuracy tradeoff by algorithm

Author: Fabio Liberti
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import warnings

warnings.filterwarnings('ignore')

# Set style
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['font.size'] = 10


# =============================================================================
# SIMPLIFIED FL SIMULATOR FOR BENCHMARKING
# =============================================================================

class BenchmarkSimulator:
    """Lightweight FL simulator for algorithm comparison."""

    def __init__(self,
                 num_nodes: int = 5,
                 alpha: float = 0.5,
                 random_seed: int = 42):
        self.num_nodes = num_nodes
        self.alpha = alpha
        self.rng = np.random.RandomState(random_seed)

        self._generate_data()

    def _generate_data(self):
        """Generate non-IID data using Dirichlet distribution."""
        total_samples = 2000
        samples_per_node = total_samples // self.num_nodes

        # Dirichlet for label distribution
        label_dist = self.rng.dirichlet([self.alpha, self.alpha], size=self.num_nodes)

        self.node_data = {}

        for i in range(self.num_nodes):
            n = samples_per_node + self.rng.randint(-30, 30)

            # Features
            shift = (i - self.num_nodes / 2) * 0.3
            X = self.rng.normal(shift, 1.0, (n, 5))
            X_norm = (X - X.mean(0)) / (X.std(0) + 1e-8)
            X_bias = np.hstack([X_norm, np.ones((n, 1))])

            # Labels from Dirichlet
            y = self.rng.choice(2, size=n, p=label_dist[i])

            self.node_data[i] = {"X": X_bias, "y": y, "n": n}

    def train(self,
              algorithm: str = 'FedAvg',
              num_rounds: int = 50,
              local_epochs: int = 3,
              lr: float = 0.1,
              **kwargs) -> Dict:
        """Run FL training with specified algorithm."""
        weights = np.zeros(6)
        history = {'accuracy': [], 'loss': []}

        # Algorithm-specific state
        momentum = None
        velocity = None
        control_variates = {i: np.zeros(6) for i in range(self.num_nodes)}
        server_control = np.zeros(6)

        for round_num in range(num_rounds):
            gradients = []
            sample_counts = []

            # Local training
            for node_id in range(self.num_nodes):
                data = self.node_data[node_id]
                local_w = weights.copy()

                for _ in range(local_epochs):
                    batch_size = min(32, data["n"])
                    idx = self.rng.choice(data["n"], batch_size, replace=False)
                    X_b, y_b = data["X"][idx], data["y"][idx]

                    logits = X_b @ local_w
                    probs = 1 / (1 + np.exp(-np.clip(logits, -500, 500)))
                    grad = X_b.T @ (probs - y_b) / batch_size

                    # Algorithm-specific modifications
                    if algorithm == 'FedProx':
                        mu = kwargs.get('mu', 0.1)
                        grad += mu * (local_w - weights)

                    elif algorithm == 'SCAFFOLD':
                        # Variance reduction
                        grad = grad - control_variates[node_id] + server_control

                    local_w -= lr * grad

                gradient = local_w - weights

                # Clipping
                norm = np.linalg.norm(gradient)
                if norm > 1.0:
                    gradient = gradient * (1.0 / norm)

                gradients.append(gradient)
                sample_counts.append(data["n"])

                # SCAFFOLD: update client control variate
                if algorithm == 'SCAFFOLD':
                    control_variates[node_id] = control_variates[node_id] - server_control + \
                                                (weights - local_w) / (local_epochs * lr)

            # Server aggregation
            total = sum(sample_counts)
            agg_grad = sum(g * (n / total) for g, n in zip(gradients, sample_counts))

            # Algorithm-specific server update
            if algorithm in ['FedAdam', 'FedYogi']:
                beta1 = kwargs.get('beta1', 0.9)
                beta2 = kwargs.get('beta2', 0.99)
                tau = kwargs.get('tau', 1e-3)
                server_lr = kwargs.get('server_lr', 0.1)

                if momentum is None:
                    momentum = np.zeros_like(agg_grad)
                    velocity = np.ones_like(agg_grad) * tau**2

                momentum = beta1 * momentum + (1 - beta1) * agg_grad

                if algorithm == 'FedAdam':
                    velocity = beta2 * velocity + (1 - beta2) * agg_grad**2
                else:  # FedYogi
                    sign = np.sign(agg_grad**2 - velocity)
                    velocity = velocity + (1 - beta2) * sign * agg_grad**2

                agg_grad = server_lr * momentum / (np.sqrt(velocity) + tau)

            elif algorithm == 'SCAFFOLD':
                # Update server control
                delta_c = sum(
                    control_variates[i] - np.zeros(6)
                    for i in range(self.num_nodes)
                ) / self.num_nodes
                server_control = server_control + delta_c

            weights += agg_grad

            # Evaluate
            all_preds, all_labels = [], []
            total_loss = 0

            for data in self.node_data.values():
                logits = data["X"] @ weights
                probs = 1 / (1 + np.exp(-np.clip(logits, -500, 500)))
                preds = (probs > 0.5).astype(int)
                all_preds.extend(preds)
                all_labels.extend(data["y"])

                # Cross-entropy loss
                eps = 1e-10
                loss = -np.mean(data["y"] * np.log(probs + eps) +
                               (1 - data["y"]) * np.log(1 - probs + eps))
                total_loss += loss

            accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
            history['accuracy'].append(accuracy)
            history['loss'].append(total_loss / self.num_nodes)

        return history


# =============================================================================
# BENCHMARK RUNNER
# =============================================================================

def run_algorithm_comparison(output_dir: str = './results_algorithms'):
    """Run comprehensive algorithm comparison and generate figures."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("FL-EHDS Algorithm Comparison Benchmark")
    print("=" * 70)
    print(f"Output directory: {output_path}")
    print(f"Timestamp: {datetime.now().isoformat()}")

    # Algorithms to compare
    algorithms = {
        'FedAvg': {'color': '#1f77b4', 'marker': 'o'},
        'FedProx': {'color': '#ff7f0e', 'marker': 's', 'mu': 0.1},
        'SCAFFOLD': {'color': '#2ca02c', 'marker': '^'},
        'FedAdam': {'color': '#d62728', 'marker': 'D', 'server_lr': 0.1, 'beta1': 0.9, 'beta2': 0.99},
        'FedYogi': {'color': '#9467bd', 'marker': 'v', 'server_lr': 0.1, 'beta1': 0.9, 'beta2': 0.99}
    }

    # Non-IID levels to test
    alpha_values = [0.1, 0.5, 1.0, 10.0]  # 0.1 = extreme non-IID, 10 = almost IID

    results = {}

    # ==========================================================================
    # EXPERIMENT 1: Algorithm comparison at different non-IID levels
    # ==========================================================================
    print("\n[1/4] Running Algorithm Comparison at Different Non-IID Levels...")

    for alpha in alpha_values:
        print(f"  α = {alpha}...")
        results[alpha] = {}

        for algo_name, algo_config in algorithms.items():
            # Run simulation
            sim = BenchmarkSimulator(num_nodes=5, alpha=alpha, random_seed=42)
            history = sim.train(
                algorithm=algo_name,
                num_rounds=50,
                local_epochs=3,
                lr=0.1,
                **{k: v for k, v in algo_config.items() if k not in ['color', 'marker']}
            )
            results[alpha][algo_name] = history

    # Generate Figure: Convergence by Non-IID level
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, alpha in enumerate(alpha_values):
        ax = axes[idx]

        for algo_name, algo_config in algorithms.items():
            acc = results[alpha][algo_name]['accuracy']
            ax.plot(range(1, len(acc) + 1), acc,
                   color=algo_config['color'],
                   marker=algo_config['marker'],
                   markevery=10,
                   linewidth=2,
                   label=algo_name)

        ax.set_xlabel('Round')
        ax.set_ylabel('Accuracy')
        noniid_level = "Extreme" if alpha == 0.1 else "High" if alpha == 0.5 else "Moderate" if alpha == 1.0 else "IID"
        ax.set_title(f'α = {alpha} ({noniid_level} Non-IID)')
        ax.set_ylim(0.45, 0.70)
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle('FL Algorithm Comparison: Convergence at Different Non-IID Levels', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(output_path / 'fig_algorithm_comparison_noniid.pdf', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: fig_algorithm_comparison_noniid.pdf")

    # ==========================================================================
    # EXPERIMENT 2: Final accuracy vs Non-IID level
    # ==========================================================================
    print("\n[2/4] Generating Final Accuracy vs Non-IID Level Plot...")

    fig, ax = plt.subplots(figsize=(10, 6))

    for algo_name, algo_config in algorithms.items():
        final_accs = [results[alpha][algo_name]['accuracy'][-1] for alpha in alpha_values]
        ax.plot(alpha_values, final_accs,
               color=algo_config['color'],
               marker=algo_config['marker'],
               linewidth=2,
               markersize=10,
               label=algo_name)

    ax.set_xlabel('Dirichlet α (higher = more IID)', fontsize=12)
    ax.set_ylabel('Final Accuracy', fontsize=12)
    ax.set_title('Final Accuracy vs Data Heterogeneity Level', fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.set_xticks(alpha_values)
    ax.set_xticklabels([str(a) for a in alpha_values])
    ax.set_ylim(0.50, 0.65)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    # Add annotation
    ax.annotate('← More Non-IID | More IID →',
               xy=(0.5, 0.02), xycoords='axes fraction',
               ha='center', fontsize=10, style='italic')

    plt.tight_layout()
    fig.savefig(output_path / 'fig_accuracy_vs_noniid.pdf', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: fig_accuracy_vs_noniid.pdf")

    # ==========================================================================
    # EXPERIMENT 3: Convergence speed comparison
    # ==========================================================================
    print("\n[3/4] Analyzing Convergence Speed...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Rounds to reach 55% accuracy
    ax1 = axes[0]

    rounds_to_55 = {}
    for alpha in [0.1, 0.5, 1.0]:
        rounds_to_55[alpha] = {}
        for algo_name in algorithms.keys():
            acc = results[alpha][algo_name]['accuracy']
            round_55 = next((i+1 for i, a in enumerate(acc) if a >= 0.55), 50)
            rounds_to_55[alpha][algo_name] = round_55

    x = np.arange(len(algorithms))
    width = 0.25

    for i, alpha in enumerate([0.1, 0.5, 1.0]):
        values = [rounds_to_55[alpha][algo] for algo in algorithms.keys()]
        ax1.bar(x + i * width, values, width,
               label=f'α={alpha}',
               alpha=0.8)

    ax1.set_xlabel('Algorithm')
    ax1.set_ylabel('Rounds to 55% Accuracy')
    ax1.set_title('Convergence Speed: Rounds to Reach 55% Accuracy')
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(list(algorithms.keys()))
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # Right: Best accuracy achieved in first 20 rounds
    ax2 = axes[1]

    best_20 = {}
    for alpha in [0.1, 0.5, 1.0]:
        best_20[alpha] = {}
        for algo_name in algorithms.keys():
            acc = results[alpha][algo_name]['accuracy'][:20]
            best_20[alpha][algo_name] = max(acc)

    for i, alpha in enumerate([0.1, 0.5, 1.0]):
        values = [best_20[alpha][algo] for algo in algorithms.keys()]
        ax2.bar(x + i * width, values, width,
               label=f'α={alpha}',
               alpha=0.8)

    ax2.set_xlabel('Algorithm')
    ax2.set_ylabel('Best Accuracy (first 20 rounds)')
    ax2.set_title('Early Convergence: Best Accuracy in First 20 Rounds')
    ax2.set_xticks(x + width)
    ax2.set_xticklabels(list(algorithms.keys()))
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    fig.savefig(output_path / 'fig_convergence_speed.pdf', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: fig_convergence_speed.pdf")

    # ==========================================================================
    # EXPERIMENT 4: Summary comparison table
    # ==========================================================================
    print("\n[4/4] Generating Summary Table...")

    # Create summary
    summary = []
    for algo_name in algorithms.keys():
        row = {'Algorithm': algo_name}
        for alpha in alpha_values:
            row[f'α={alpha}'] = f"{results[alpha][algo_name]['accuracy'][-1]:.2%}"
        summary.append(row)

    # Save as JSON
    with open(output_path / 'algorithm_comparison_results.json', 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'algorithms': list(algorithms.keys()),
            'alpha_values': alpha_values,
            'results': {
                str(alpha): {
                    algo: {
                        'final_accuracy': results[alpha][algo]['accuracy'][-1],
                        'final_loss': results[alpha][algo]['loss'][-1],
                        'accuracy_history': results[alpha][algo]['accuracy']
                    }
                    for algo in algorithms.keys()
                }
                for alpha in alpha_values
            }
        }, f, indent=2)

    print(f"  Saved: algorithm_comparison_results.json")

    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print("\nFinal Accuracy by Algorithm and Non-IID Level:")
    print("-" * 60)
    print(f"{'Algorithm':<12}", end='')
    for alpha in alpha_values:
        print(f"α={alpha:<6}", end='')
    print()
    print("-" * 60)

    for algo_name in algorithms.keys():
        print(f"{algo_name:<12}", end='')
        for alpha in alpha_values:
            acc = results[alpha][algo_name]['accuracy'][-1]
            print(f"{acc:.2%}   ", end='')
        print()

    print("-" * 60)

    # Best algorithm per setting
    print("\nBest Algorithm per Non-IID Level:")
    for alpha in alpha_values:
        best_algo = max(algorithms.keys(),
                       key=lambda a: results[alpha][a]['accuracy'][-1])
        best_acc = results[alpha][best_algo]['accuracy'][-1]
        print(f"  α={alpha}: {best_algo} ({best_acc:.2%})")

    print("\n" + "=" * 70)
    print(f"All figures saved to: {output_path.absolute()}")
    print("=" * 70)

    return results


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Run comparison
    results = run_algorithm_comparison(
        output_dir='./results_algorithms'
    )

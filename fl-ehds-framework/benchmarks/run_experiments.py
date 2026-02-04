#!/usr/bin/env python3
"""
FL-EHDS Experimental Benchmarks
================================
Generates real experimental results for the FLICS 2026 paper.

This script runs federated learning experiments with:
- Synthetic healthcare data (cardiovascular risk prediction)
- Multiple hospitals with non-IID distributions
- FedAvg vs FedProx comparison
- Differential privacy impact analysis
- Hardware heterogeneity simulation
"""

import json
import os
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# ============================================================================
# DATA GENERATION
# ============================================================================

def generate_cardiovascular_data(
    n_samples: int,
    hospital_bias: float = 0.0,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic cardiovascular risk data.

    Features: age, BMI, systolic_bp, glucose, cholesterol (normalized)
    Label: cardiovascular event risk (binary)

    Args:
        n_samples: Number of samples
        hospital_bias: Bias factor for non-IID simulation (-1 to 1)
        seed: Random seed for reproducibility

    Returns:
        X (features), y (labels)
    """
    rng = np.random.RandomState(seed)

    # Generate features with hospital-specific distributions (non-IID)
    age = rng.normal(55 + hospital_bias * 10, 12, n_samples)
    bmi = rng.normal(26 + hospital_bias * 3, 4, n_samples)
    systolic_bp = rng.normal(130 + hospital_bias * 10, 18, n_samples)
    glucose = rng.normal(100 + hospital_bias * 20, 25, n_samples)
    cholesterol = rng.normal(200 + hospital_bias * 25, 35, n_samples)

    # Normalize features
    X = np.column_stack([
        (age - 55) / 12,
        (bmi - 26) / 4,
        (systolic_bp - 130) / 18,
        (glucose - 100) / 25,
        (cholesterol - 200) / 35,
    ])

    # Generate labels based on risk model
    risk_score = (
        0.3 * X[:, 0] +  # age
        0.2 * X[:, 1] +  # bmi
        0.25 * X[:, 2] + # blood pressure
        0.15 * X[:, 3] + # glucose
        0.1 * X[:, 4]    # cholesterol
    )
    prob = 1 / (1 + np.exp(-risk_score))
    y = (rng.random(n_samples) < prob).astype(np.float64)

    return X, y


# ============================================================================
# FEDERATED LEARNING IMPLEMENTATION
# ============================================================================

class LogisticRegressionModel:
    """Simple logistic regression for FL experiments."""

    def __init__(self, n_features: int = 5):
        self.weights = np.zeros(n_features)
        self.bias = 0.0

    def get_params(self) -> Dict[str, np.ndarray]:
        return {"weights": self.weights.copy(), "bias": np.array([self.bias])}

    def set_params(self, params: Dict[str, np.ndarray]):
        self.weights = params["weights"].copy()
        self.bias = float(params["bias"][0])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        z = X @ self.weights + self.bias
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X) >= 0.5).astype(int)

    def compute_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        pred = self.predict_proba(X)
        eps = 1e-7
        return -np.mean(y * np.log(pred + eps) + (1 - y) * np.log(1 - pred + eps))

    def compute_accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        return np.mean(self.predict(X) == y)

    def train_step(
        self,
        X: np.ndarray,
        y: np.ndarray,
        lr: float,
        global_params: Optional[Dict] = None,
        mu: float = 0.0  # FedProx proximal term
    ) -> float:
        """Single training step with optional FedProx proximal term."""
        pred = self.predict_proba(X)
        error = pred - y

        # Gradients
        grad_w = X.T @ error / len(y)
        grad_b = np.mean(error)

        # Add FedProx proximal term
        if mu > 0 and global_params is not None:
            grad_w += mu * (self.weights - global_params["weights"])
            grad_b += mu * (self.bias - global_params["bias"][0])

        # Update
        self.weights -= lr * grad_w
        self.bias -= lr * grad_b

        return self.compute_loss(X, y)


def add_differential_privacy_noise(
    gradients: Dict[str, np.ndarray],
    epsilon: float,
    sensitivity: float,
    delta: float = 1e-5
) -> Dict[str, np.ndarray]:
    """Add Gaussian noise for differential privacy."""
    if epsilon <= 0:
        return gradients

    # Gaussian mechanism: sigma = sensitivity * sqrt(2 * ln(1.25/delta)) / epsilon
    sigma = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon

    noisy_gradients = {}
    for key, grad in gradients.items():
        noise = np.random.normal(0, sigma, grad.shape)
        noisy_gradients[key] = grad + noise

    return noisy_gradients


def clip_gradients(
    gradients: Dict[str, np.ndarray],
    max_norm: float
) -> Tuple[Dict[str, np.ndarray], float]:
    """Clip gradient L2 norm."""
    # Compute global norm
    total_norm = 0.0
    for grad in gradients.values():
        total_norm += np.sum(grad ** 2)
    total_norm = np.sqrt(total_norm)

    clip_factor = min(1.0, max_norm / (total_norm + 1e-7))

    clipped = {k: v * clip_factor for k, v in gradients.items()}
    return clipped, total_norm


def fedavg_aggregate(
    client_updates: List[Tuple[Dict[str, np.ndarray], int]]
) -> Dict[str, np.ndarray]:
    """FedAvg weighted aggregation."""
    total_samples = sum(n for _, n in client_updates)

    aggregated = {}
    for key in client_updates[0][0].keys():
        weighted_sum = sum(
            update[key] * n_samples
            for update, n_samples in client_updates
        )
        aggregated[key] = weighted_sum / total_samples

    return aggregated


# ============================================================================
# EXPERIMENT RUNNERS
# ============================================================================

@dataclass
class ExperimentConfig:
    """Configuration for FL experiment."""
    name: str
    num_rounds: int
    num_clients: int
    local_epochs: int
    learning_rate: float
    batch_size: int
    algorithm: str  # "fedavg" or "fedprox"
    mu: float = 0.0  # FedProx parameter
    dp_epsilon: float = 0.0  # 0 = no DP
    gradient_clip_norm: float = 1.0
    non_iid_degree: float = 0.5  # 0 = IID, 1 = highly non-IID


@dataclass
class ExperimentResult:
    """Results from FL experiment."""
    config: ExperimentConfig
    round_losses: List[float]
    round_accuracies: List[float]
    final_loss: float
    final_accuracy: float
    convergence_round: int  # Round where loss < 0.5
    total_time_seconds: float
    privacy_budget_spent: float
    client_accuracies: List[float]


def run_fl_experiment(config: ExperimentConfig) -> ExperimentResult:
    """Run a complete FL experiment."""
    print(f"\n{'='*60}")
    print(f"Running: {config.name}")
    print(f"Algorithm: {config.algorithm}, Clients: {config.num_clients}, "
          f"Rounds: {config.num_rounds}")
    print(f"{'='*60}")

    start_time = time.time()

    # Generate client data with non-IID distributions
    client_data = []
    samples_per_client = [300, 500, 400, 350, 450][:config.num_clients]

    for i in range(config.num_clients):
        # Non-IID bias: spread from -degree to +degree
        bias = config.non_iid_degree * (2 * i / (config.num_clients - 1) - 1) if config.num_clients > 1 else 0
        X, y = generate_cardiovascular_data(
            samples_per_client[i],
            hospital_bias=bias,
            seed=42 + i
        )
        client_data.append((X, y))

    # Generate test data (IID)
    X_test, y_test = generate_cardiovascular_data(500, hospital_bias=0.0, seed=999)

    # Initialize global model
    global_model = LogisticRegressionModel()

    # Training metrics
    round_losses = []
    round_accuracies = []
    convergence_round = config.num_rounds
    total_epsilon_spent = 0.0

    for round_num in range(1, config.num_rounds + 1):
        client_updates = []
        round_loss = 0.0

        global_params = global_model.get_params()

        for client_idx, (X_client, y_client) in enumerate(client_data):
            # Create local model from global
            local_model = LogisticRegressionModel()
            local_model.set_params(global_params)

            # Local training
            n_samples = len(X_client)
            for epoch in range(config.local_epochs):
                # Mini-batch training
                indices = np.random.permutation(n_samples)
                for start in range(0, n_samples, config.batch_size):
                    end = min(start + config.batch_size, n_samples)
                    batch_idx = indices[start:end]

                    local_model.train_step(
                        X_client[batch_idx],
                        y_client[batch_idx],
                        lr=config.learning_rate,
                        global_params=global_params if config.mu > 0 else None,
                        mu=config.mu
                    )

            # Compute update (difference from global)
            local_params = local_model.get_params()
            update = {
                k: local_params[k] - global_params[k]
                for k in global_params.keys()
            }

            # Gradient clipping
            update, _ = clip_gradients(update, config.gradient_clip_norm)

            # Differential privacy
            if config.dp_epsilon > 0:
                per_round_epsilon = config.dp_epsilon / config.num_rounds
                update = add_differential_privacy_noise(
                    update,
                    epsilon=per_round_epsilon,
                    sensitivity=config.gradient_clip_norm
                )
                total_epsilon_spent += per_round_epsilon

            client_updates.append((update, n_samples))
            round_loss += local_model.compute_loss(X_client, y_client) * n_samples

        # Aggregate updates
        aggregated_update = fedavg_aggregate(client_updates)

        # Update global model
        new_params = {
            k: global_params[k] + aggregated_update[k]
            for k in global_params.keys()
        }
        global_model.set_params(new_params)

        # Evaluate
        total_samples = sum(len(d[0]) for d in client_data)
        avg_loss = round_loss / total_samples
        test_accuracy = global_model.compute_accuracy(X_test, y_test)

        round_losses.append(avg_loss)
        round_accuracies.append(test_accuracy)

        # Check convergence
        if avg_loss < 0.5 and convergence_round == config.num_rounds:
            convergence_round = round_num

        if round_num % 5 == 0 or round_num == 1:
            print(f"  Round {round_num:3d}: Loss={avg_loss:.4f}, Accuracy={test_accuracy:.4f}")

    # Final per-client accuracy
    client_accuracies = []
    for X_client, y_client in client_data:
        acc = global_model.compute_accuracy(X_client, y_client)
        client_accuracies.append(acc)

    total_time = time.time() - start_time

    result = ExperimentResult(
        config=config,
        round_losses=round_losses,
        round_accuracies=round_accuracies,
        final_loss=round_losses[-1],
        final_accuracy=round_accuracies[-1],
        convergence_round=convergence_round,
        total_time_seconds=total_time,
        privacy_budget_spent=total_epsilon_spent,
        client_accuracies=client_accuracies
    )

    print(f"\n  Final: Loss={result.final_loss:.4f}, Accuracy={result.final_accuracy:.4f}")
    print(f"  Convergence at round: {convergence_round}")
    print(f"  Time: {total_time:.2f}s")

    return result


# ============================================================================
# MAIN EXPERIMENTS
# ============================================================================

def run_all_experiments() -> Dict[str, ExperimentResult]:
    """Run all experiments for the paper."""
    results = {}

    # Experiment 1: FedAvg baseline (IID)
    results["fedavg_iid"] = run_fl_experiment(ExperimentConfig(
        name="FedAvg (IID Data)",
        num_rounds=30,
        num_clients=5,
        local_epochs=3,
        learning_rate=0.1,
        batch_size=32,
        algorithm="fedavg",
        non_iid_degree=0.0
    ))

    # Experiment 2: FedAvg with non-IID data
    results["fedavg_noniid"] = run_fl_experiment(ExperimentConfig(
        name="FedAvg (Non-IID Data)",
        num_rounds=30,
        num_clients=5,
        local_epochs=3,
        learning_rate=0.1,
        batch_size=32,
        algorithm="fedavg",
        non_iid_degree=0.8
    ))

    # Experiment 3: FedProx with non-IID data
    results["fedprox_noniid"] = run_fl_experiment(ExperimentConfig(
        name="FedProx (Non-IID Data, μ=0.1)",
        num_rounds=30,
        num_clients=5,
        local_epochs=3,
        learning_rate=0.1,
        batch_size=32,
        algorithm="fedprox",
        mu=0.1,
        non_iid_degree=0.8
    ))

    # Experiment 4: FedAvg + Differential Privacy (ε=10)
    results["fedavg_dp10"] = run_fl_experiment(ExperimentConfig(
        name="FedAvg + DP (ε=10)",
        num_rounds=30,
        num_clients=5,
        local_epochs=3,
        learning_rate=0.1,
        batch_size=32,
        algorithm="fedavg",
        dp_epsilon=10.0,
        non_iid_degree=0.5
    ))

    # Experiment 5: FedAvg + Differential Privacy (ε=1)
    results["fedavg_dp1"] = run_fl_experiment(ExperimentConfig(
        name="FedAvg + DP (ε=1, Strong Privacy)",
        num_rounds=30,
        num_clients=5,
        local_epochs=3,
        learning_rate=0.1,
        batch_size=32,
        algorithm="fedavg",
        dp_epsilon=1.0,
        non_iid_degree=0.5
    ))

    # Experiment 6: Scalability - 3 clients
    results["scale_3"] = run_fl_experiment(ExperimentConfig(
        name="Scalability (3 Hospitals)",
        num_rounds=30,
        num_clients=3,
        local_epochs=3,
        learning_rate=0.1,
        batch_size=32,
        algorithm="fedavg",
        non_iid_degree=0.5
    ))

    # Experiment 7: Scalability - 5 clients
    results["scale_5"] = run_fl_experiment(ExperimentConfig(
        name="Scalability (5 Hospitals)",
        num_rounds=30,
        num_clients=5,
        local_epochs=3,
        learning_rate=0.1,
        batch_size=32,
        algorithm="fedavg",
        non_iid_degree=0.5
    ))

    return results


def generate_latex_table(results: Dict[str, ExperimentResult]) -> str:
    """Generate LaTeX table for paper."""
    table = r"""
\begin{table}[htbp]
\centering
\caption{Experimental Results: FL-EHDS Framework Performance}
\label{tab:experimental_results}
\begin{tabular}{lcccc}
\toprule
\textbf{Configuration} & \textbf{Accuracy} & \textbf{Loss} & \textbf{Conv.} & \textbf{$\varepsilon$} \\
\midrule
"""

    row_configs = [
        ("fedavg_iid", "FedAvg (IID)"),
        ("fedavg_noniid", "FedAvg (Non-IID)"),
        ("fedprox_noniid", "FedProx (Non-IID)"),
        ("fedavg_dp10", "FedAvg + DP ($\\varepsilon$=10)"),
        ("fedavg_dp1", "FedAvg + DP ($\\varepsilon$=1)"),
    ]

    for key, label in row_configs:
        r = results[key]
        eps_str = f"{r.privacy_budget_spent:.1f}" if r.privacy_budget_spent > 0 else "---"
        table += f"{label} & {r.final_accuracy:.1%} & {r.final_loss:.3f} & {r.convergence_round} & {eps_str} \\\\\n"

    table += r"""
\bottomrule
\end{tabular}
\vspace{1mm}
\footnotesize{Conv. = Convergence round (loss $<$ 0.5). 5 hospitals, 30 rounds, synthetic cardiovascular data.}
\end{table}
"""
    return table


def generate_latex_scalability_table(results: Dict[str, ExperimentResult]) -> str:
    """Generate scalability results table."""
    table = r"""
\begin{table}[htbp]
\centering
\caption{Scalability Analysis}
\label{tab:scalability}
\begin{tabular}{lccc}
\toprule
\textbf{Hospitals} & \textbf{Accuracy} & \textbf{Std. Dev.} & \textbf{Time (s)} \\
\midrule
"""

    for key in ["scale_3", "scale_5"]:
        r = results[key]
        n_clients = r.config.num_clients
        std_dev = np.std(r.client_accuracies)
        table += f"{n_clients} & {r.final_accuracy:.1%} & {std_dev:.3f} & {r.total_time_seconds:.1f} \\\\\n"

    table += r"""
\bottomrule
\end{tabular}
\end{table}
"""
    return table


def save_results(results: Dict[str, ExperimentResult], output_dir: Path):
    """Save results to JSON and generate figures."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save raw results
    results_dict = {}
    for name, result in results.items():
        results_dict[name] = {
            "config": asdict(result.config),
            "round_losses": result.round_losses,
            "round_accuracies": result.round_accuracies,
            "final_loss": result.final_loss,
            "final_accuracy": result.final_accuracy,
            "convergence_round": result.convergence_round,
            "total_time_seconds": result.total_time_seconds,
            "privacy_budget_spent": result.privacy_budget_spent,
            "client_accuracies": result.client_accuracies
        }

    with open(output_dir / "experiment_results.json", "w") as f:
        json.dump(results_dict, f, indent=2)

    # Save LaTeX tables
    with open(output_dir / "table_results.tex", "w") as f:
        f.write(generate_latex_table(results))

    with open(output_dir / "table_scalability.tex", "w") as f:
        f.write(generate_latex_scalability_table(results))

    # Generate figures using matplotlib
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # Figure 1: Convergence comparison
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        rounds = range(1, 31)

        # Loss curves
        ax = axes[0]
        ax.plot(rounds, results["fedavg_iid"].round_losses, 'b-', label='FedAvg (IID)', linewidth=2)
        ax.plot(rounds, results["fedavg_noniid"].round_losses, 'r--', label='FedAvg (Non-IID)', linewidth=2)
        ax.plot(rounds, results["fedprox_noniid"].round_losses, 'g-.', label='FedProx (Non-IID)', linewidth=2)
        ax.set_xlabel('Round', fontsize=11)
        ax.set_ylabel('Loss', fontsize=11)
        ax.set_title('(a) Training Loss Convergence', fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(1, 30)

        # Accuracy curves
        ax = axes[1]
        ax.plot(rounds, results["fedavg_iid"].round_accuracies, 'b-', label='FedAvg (IID)', linewidth=2)
        ax.plot(rounds, results["fedavg_noniid"].round_accuracies, 'r--', label='FedAvg (Non-IID)', linewidth=2)
        ax.plot(rounds, results["fedprox_noniid"].round_accuracies, 'g-.', label='FedProx (Non-IID)', linewidth=2)
        ax.set_xlabel('Round', fontsize=11)
        ax.set_ylabel('Accuracy', fontsize=11)
        ax.set_title('(b) Test Accuracy', fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(1, 30)
        ax.set_ylim(0.5, 1.0)

        plt.tight_layout()
        plt.savefig(output_dir / "fig_convergence.pdf", dpi=300, bbox_inches='tight')
        plt.savefig(output_dir / "fig_convergence.png", dpi=150, bbox_inches='tight')
        plt.close()

        # Figure 2: Privacy-Utility tradeoff
        fig, ax = plt.subplots(figsize=(6, 4))

        dp_configs = ["fedavg_noniid", "fedavg_dp10", "fedavg_dp1"]
        epsilons = [float('inf'), 10.0, 1.0]
        accuracies = [results[c].final_accuracy for c in dp_configs]

        ax.bar(['No DP', 'ε=10', 'ε=1'], accuracies, color=['#2ecc71', '#f39c12', '#e74c3c'], edgecolor='black')
        ax.set_ylabel('Final Accuracy', fontsize=11)
        ax.set_xlabel('Privacy Budget (ε)', fontsize=11)
        ax.set_title('Privacy-Utility Tradeoff', fontsize=12)
        ax.set_ylim(0.5, 1.0)
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels
        for i, (e, a) in enumerate(zip(['∞', '10', '1'], accuracies)):
            ax.text(i, a + 0.02, f'{a:.1%}', ha='center', fontsize=10)

        plt.tight_layout()
        plt.savefig(output_dir / "fig_privacy_utility.pdf", dpi=300, bbox_inches='tight')
        plt.savefig(output_dir / "fig_privacy_utility.png", dpi=150, bbox_inches='tight')
        plt.close()

        print(f"\nFigures saved to {output_dir}")

    except ImportError:
        print("Warning: matplotlib not available, skipping figure generation")

    print(f"\nResults saved to {output_dir}")
    print(f"  - experiment_results.json")
    print(f"  - table_results.tex")
    print(f"  - table_scalability.tex")


def print_summary(results: Dict[str, ExperimentResult]):
    """Print summary for paper."""
    print("\n" + "="*70)
    print("EXPERIMENTAL RESULTS SUMMARY")
    print("="*70)

    print("\n1. ALGORITHM COMPARISON (Non-IID Data)")
    print("-" * 50)
    fedavg = results["fedavg_noniid"]
    fedprox = results["fedprox_noniid"]
    improvement = (fedprox.final_accuracy - fedavg.final_accuracy) / fedavg.final_accuracy * 100
    print(f"   FedAvg:  {fedavg.final_accuracy:.1%} accuracy, converged at round {fedavg.convergence_round}")
    print(f"   FedProx: {fedprox.final_accuracy:.1%} accuracy, converged at round {fedprox.convergence_round}")
    print(f"   FedProx improvement: {improvement:+.1f}%")

    print("\n2. PRIVACY-UTILITY TRADEOFF")
    print("-" * 50)
    no_dp = results["fedavg_noniid"]
    dp10 = results["fedavg_dp10"]
    dp1 = results["fedavg_dp1"]
    print(f"   No DP:    {no_dp.final_accuracy:.1%} accuracy")
    print(f"   ε=10:     {dp10.final_accuracy:.1%} accuracy (drop: {(no_dp.final_accuracy-dp10.final_accuracy)*100:.1f}pp)")
    print(f"   ε=1:      {dp1.final_accuracy:.1%} accuracy (drop: {(no_dp.final_accuracy-dp1.final_accuracy)*100:.1f}pp)")

    print("\n3. IID vs NON-IID IMPACT")
    print("-" * 50)
    iid = results["fedavg_iid"]
    noniid = results["fedavg_noniid"]
    gap = (iid.final_accuracy - noniid.final_accuracy) * 100
    print(f"   IID:      {iid.final_accuracy:.1%} accuracy")
    print(f"   Non-IID:  {noniid.final_accuracy:.1%} accuracy")
    print(f"   Performance gap: {gap:.1f} percentage points")

    print("\n4. CLIENT HETEROGENEITY (FedProx)")
    print("-" * 50)
    accs = fedprox.client_accuracies
    print(f"   Per-client accuracies: {[f'{a:.1%}' for a in accs]}")
    print(f"   Mean: {np.mean(accs):.1%}, Std: {np.std(accs):.3f}")


if __name__ == "__main__":
    print("FL-EHDS Experimental Benchmarks")
    print("=" * 70)
    print(f"Started at: {datetime.now().isoformat()}")

    # Run all experiments
    results = run_all_experiments()

    # Print summary
    print_summary(results)

    # Save results
    output_dir = Path(__file__).parent / "results"
    save_results(results, output_dir)

    print("\n" + "="*70)
    print("EXPERIMENTS COMPLETED")
    print("="*70)

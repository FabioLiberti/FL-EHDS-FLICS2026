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

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """Compute classification metrics including F1, AUC, precision, recall."""
    # Accuracy
    accuracy = np.mean(y_true == y_pred)

    # True/False Positives/Negatives
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tn = np.sum((y_true == 0) & (y_pred == 0))

    # Precision, Recall, F1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # AUC-ROC (simple trapezoidal approximation)
    # Sort by probability
    sorted_idx = np.argsort(y_prob)[::-1]
    y_true_sorted = y_true[sorted_idx]

    # Calculate TPR and FPR at each threshold
    tpr_list = [0.0]
    fpr_list = [0.0]
    total_pos = np.sum(y_true == 1)
    total_neg = np.sum(y_true == 0)

    tp_count = 0
    fp_count = 0

    for label in y_true_sorted:
        if label == 1:
            tp_count += 1
        else:
            fp_count += 1
        tpr_list.append(tp_count / total_pos if total_pos > 0 else 0)
        fpr_list.append(fp_count / total_neg if total_neg > 0 else 0)

    # AUC via trapezoidal rule
    auc = 0.0
    for i in range(1, len(fpr_list)):
        auc += (fpr_list[i] - fpr_list[i-1]) * (tpr_list[i] + tpr_list[i-1]) / 2

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc
    }


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

    def compute_all_metrics(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Compute all metrics for evaluation."""
        y_pred = self.predict(X)
        y_prob = self.predict_proba(X)
        metrics = compute_metrics(y, y_pred, y_prob)
        metrics["loss"] = self.compute_loss(X, y)
        return metrics

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
# SCAFFOLD AND FEDNOVA IMPLEMENTATIONS
# ============================================================================

class SCAFFOLDState:
    """State for SCAFFOLD algorithm (control variates)."""
    def __init__(self, n_features: int = 5):
        # Server control variate
        self.c = {"weights": np.zeros(n_features), "bias": np.zeros(1)}
        # Client control variates (indexed by client_id)
        self.c_clients: Dict[int, Dict[str, np.ndarray]] = {}

    def get_client_control(self, client_id: int, n_features: int = 5) -> Dict[str, np.ndarray]:
        if client_id not in self.c_clients:
            self.c_clients[client_id] = {
                "weights": np.zeros(n_features),
                "bias": np.zeros(1)
            }
        return self.c_clients[client_id]


def scaffold_local_update(
    model: 'LogisticRegressionModel',
    X: np.ndarray,
    y: np.ndarray,
    lr: float,
    local_epochs: int,
    batch_size: int,
    c_server: Dict[str, np.ndarray],
    c_local: Dict[str, np.ndarray],
    global_params: Dict[str, np.ndarray]
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    SCAFFOLD local update with control variates.

    Reference: Karimireddy et al., "SCAFFOLD: Stochastic Controlled Averaging", ICML 2020
    """
    n_samples = len(X)
    model.set_params(global_params)

    for epoch in range(local_epochs):
        indices = np.random.permutation(n_samples)
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch_idx = indices[start:end]
            X_batch, y_batch = X[batch_idx], y[batch_idx]

            # Compute gradient
            pred = model.predict_proba(X_batch)
            error = pred - y_batch
            grad_w = X_batch.T @ error / len(y_batch)
            grad_b = np.mean(error)

            # SCAFFOLD correction: g - c_i + c
            corrected_grad_w = grad_w - c_local["weights"] + c_server["weights"]
            corrected_grad_b = grad_b - c_local["bias"][0] + c_server["bias"][0]

            # Update model
            model.weights -= lr * corrected_grad_w
            model.bias -= lr * corrected_grad_b

    # Compute update delta
    local_params = model.get_params()
    delta = {
        k: local_params[k] - global_params[k]
        for k in global_params.keys()
    }

    # Update client control variate
    # c_i^+ = c_i - c + (1/K*lr) * (x - y)
    # where x = global params, y = local params after training
    # delta = local - global, so (x - y) = -delta
    K = local_epochs * (n_samples // batch_size + 1)
    new_c_local = {
        "weights": c_local["weights"] - c_server["weights"] - delta["weights"] / (K * lr),
        "bias": c_local["bias"] - c_server["bias"] - delta["bias"] / (K * lr)
    }

    # Delta for control variate
    delta_c = {
        "weights": new_c_local["weights"] - c_local["weights"],
        "bias": new_c_local["bias"] - c_local["bias"]
    }

    return delta, delta_c, new_c_local


def fednova_aggregate(
    client_updates: List[Tuple[Dict[str, np.ndarray], int, int]]
) -> Dict[str, np.ndarray]:
    """
    FedNova normalized aggregation.

    Reference: Wang et al., "Tackling the Objective Inconsistency Problem", NeurIPS 2020

    Args:
        client_updates: List of (update, n_samples, local_steps)
    """
    # Compute normalized updates
    total_samples = sum(n for _, n, _ in client_updates)

    # tau_eff = weighted average of local steps
    tau_eff = sum(n * tau for _, n, tau in client_updates) / total_samples

    aggregated = {}
    for key in client_updates[0][0].keys():
        weighted_sum = sum(
            update[key] * n_samples * tau_eff / tau
            for update, n_samples, tau in client_updates
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
    algorithm: str  # "fedavg", "fedprox", "scaffold", or "fednova"
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
    # Standard deviation from multiple runs
    accuracy_std: float = 0.0
    loss_std: float = 0.0
    f1_score: float = 0.0
    f1_std: float = 0.0
    auc_score: float = 0.0
    auc_std: float = 0.0
    precision: float = 0.0
    recall: float = 0.0


def run_fl_experiment(config: ExperimentConfig, base_seed: int = 42) -> ExperimentResult:
    """Run a complete FL experiment."""
    print(f"\n{'='*60}")
    print(f"Running: {config.name} (seed={base_seed})")
    print(f"Algorithm: {config.algorithm}, Clients: {config.num_clients}, "
          f"Rounds: {config.num_rounds}")
    print(f"{'='*60}")

    # Set random seed
    np.random.seed(base_seed)

    start_time = time.time()

    # Generate client data with non-IID distributions
    client_data = []
    # Sample counts per hospital (extended for scalability tests)
    base_samples = [300, 500, 400, 350, 450, 380, 420, 360, 480, 340]
    samples_per_client = base_samples[:config.num_clients] if config.num_clients <= len(base_samples) else \
        [np.random.randint(300, 500) for _ in range(config.num_clients)]

    for i in range(config.num_clients):
        # Non-IID bias: spread from -degree to +degree
        bias = config.non_iid_degree * (2 * i / (config.num_clients - 1) - 1) if config.num_clients > 1 else 0
        X, y = generate_cardiovascular_data(
            samples_per_client[i],
            hospital_bias=bias,
            seed=base_seed + i
        )
        client_data.append((X, y))

    # Generate test data (IID)
    X_test, y_test = generate_cardiovascular_data(500, hospital_bias=0.0, seed=base_seed + 1000)

    # Initialize global model
    global_model = LogisticRegressionModel()

    # Initialize SCAFFOLD state if needed
    scaffold_state = SCAFFOLDState() if config.algorithm == "scaffold" else None

    # Training metrics
    round_losses = []
    round_accuracies = []
    convergence_round = config.num_rounds
    total_epsilon_spent = 0.0

    for round_num in range(1, config.num_rounds + 1):
        client_updates = []
        round_loss = 0.0

        global_params = global_model.get_params()

        # For SCAFFOLD: collect control variate updates
        scaffold_c_deltas = [] if config.algorithm == "scaffold" else None

        for client_idx, (X_client, y_client) in enumerate(client_data):
            # Create local model from global
            local_model = LogisticRegressionModel()
            local_model.set_params(global_params)

            # Compute local steps for FedNova
            n_samples = len(X_client)
            local_steps = config.local_epochs * (n_samples // config.batch_size + 1)

            # SCAFFOLD training
            if config.algorithm == "scaffold":
                c_local = scaffold_state.get_client_control(client_idx)
                delta, delta_c, new_c_local = scaffold_local_update(
                    local_model, X_client, y_client,
                    config.learning_rate, config.local_epochs, config.batch_size,
                    scaffold_state.c, c_local, global_params
                )
                scaffold_state.c_clients[client_idx] = new_c_local
                scaffold_c_deltas.append(delta_c)

                # Gradient clipping
                delta, _ = clip_gradients(delta, config.gradient_clip_norm)

                # Differential privacy
                if config.dp_epsilon > 0:
                    per_round_epsilon = config.dp_epsilon / config.num_rounds
                    delta = add_differential_privacy_noise(
                        delta, epsilon=per_round_epsilon, sensitivity=config.gradient_clip_norm
                    )
                    total_epsilon_spent += per_round_epsilon

                client_updates.append((delta, n_samples))
                round_loss += local_model.compute_loss(X_client, y_client) * n_samples
                continue

            # Standard Local training (FedAvg, FedProx, FedNova)
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

            # For FedNova, include local_steps
            if config.algorithm == "fednova":
                client_updates.append((update, n_samples, local_steps))
            else:
                client_updates.append((update, n_samples))
            round_loss += local_model.compute_loss(X_client, y_client) * n_samples

        # Aggregate updates based on algorithm
        if config.algorithm == "fednova":
            aggregated_update = fednova_aggregate(client_updates)
        else:
            aggregated_update = fedavg_aggregate(client_updates)

        # Update SCAFFOLD server control variate
        if config.algorithm == "scaffold" and scaffold_c_deltas:
            n_clients = len(scaffold_c_deltas)
            scaffold_state.c = {
                "weights": scaffold_state.c["weights"] + sum(d["weights"] for d in scaffold_c_deltas) / n_clients,
                "bias": scaffold_state.c["bias"] + sum(d["bias"] for d in scaffold_c_deltas) / n_clients
            }

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

    # Compute final metrics on test set
    final_metrics = global_model.compute_all_metrics(X_test, y_test)

    result = ExperimentResult(
        config=config,
        round_losses=round_losses,
        round_accuracies=round_accuracies,
        final_loss=round_losses[-1],
        final_accuracy=round_accuracies[-1],
        convergence_round=convergence_round,
        total_time_seconds=total_time,
        privacy_budget_spent=total_epsilon_spent,
        client_accuracies=client_accuracies,
        f1_score=final_metrics["f1"],
        auc_score=final_metrics["auc"],
        precision=final_metrics["precision"],
        recall=final_metrics["recall"]
    )

    print(f"\n  Final: Loss={result.final_loss:.4f}, Accuracy={result.final_accuracy:.4f}")
    print(f"  F1={result.f1_score:.4f}, AUC={result.auc_score:.4f}")
    print(f"  Convergence at round: {convergence_round}")
    print(f"  Time: {total_time:.2f}s")

    return result


def run_experiment_with_multiple_seeds(
    config: ExperimentConfig,
    num_runs: int = 5,
    seeds: Optional[List[int]] = None
) -> ExperimentResult:
    """
    Run experiment multiple times with different seeds and compute statistics.

    Args:
        config: Experiment configuration
        num_runs: Number of runs (default 5)
        seeds: List of seeds (default [42, 123, 456, 789, 1024])

    Returns:
        ExperimentResult with mean values and std deviations
    """
    if seeds is None:
        seeds = [42, 123, 456, 789, 1024][:num_runs]

    print(f"\n{'#'*70}")
    print(f"# Running {config.name} with {num_runs} seeds for statistical significance")
    print(f"# Seeds: {seeds}")
    print(f"{'#'*70}")

    all_results = []
    for seed in seeds:
        result = run_fl_experiment(config, base_seed=seed)
        all_results.append(result)

    # Aggregate metrics
    accuracies = [r.final_accuracy for r in all_results]
    losses = [r.final_loss for r in all_results]
    f1_scores = [r.f1_score for r in all_results]
    auc_scores = [r.auc_score for r in all_results]

    # Compute mean and std
    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    mean_loss = np.mean(losses)
    std_loss = np.std(losses)
    mean_f1 = np.mean(f1_scores)
    std_f1 = np.std(f1_scores)
    mean_auc = np.mean(auc_scores)
    std_auc = np.std(auc_scores)
    mean_precision = np.mean([r.precision for r in all_results])
    mean_recall = np.mean([r.recall for r in all_results])

    # Use first result as base, update with aggregated stats
    aggregated_result = ExperimentResult(
        config=config,
        round_losses=all_results[0].round_losses,  # Use first run's trajectory
        round_accuracies=all_results[0].round_accuracies,
        final_loss=mean_loss,
        final_accuracy=mean_accuracy,
        convergence_round=int(np.mean([r.convergence_round for r in all_results])),
        total_time_seconds=np.mean([r.total_time_seconds for r in all_results]),
        privacy_budget_spent=all_results[0].privacy_budget_spent,
        client_accuracies=all_results[0].client_accuracies,
        accuracy_std=std_accuracy,
        loss_std=std_loss,
        f1_score=mean_f1,
        f1_std=std_f1,
        auc_score=mean_auc,
        auc_std=std_auc,
        precision=mean_precision,
        recall=mean_recall
    )

    print(f"\n{'='*60}")
    print(f"AGGREGATED RESULTS ({num_runs} runs)")
    print(f"{'='*60}")
    print(f"  Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
    print(f"  Loss:     {mean_loss:.4f} ± {std_loss:.4f}")
    print(f"  F1:       {mean_f1:.4f} ± {std_f1:.4f}")
    print(f"  AUC:      {mean_auc:.4f} ± {std_auc:.4f}")
    print(f"  Prec/Rec: {mean_precision:.4f}/{mean_recall:.4f}")

    return aggregated_result


# ============================================================================
# MAIN EXPERIMENTS
# ============================================================================

def run_all_experiments(num_runs: int = 5) -> Dict[str, ExperimentResult]:
    """
    Run all experiments for the paper with statistical significance.

    Args:
        num_runs: Number of runs per experiment for computing std dev (default 5)

    Returns:
        Dictionary of experiment results with mean and std dev
    """
    results = {}

    # Experiment 1: FedAvg baseline (IID)
    results["fedavg_iid"] = run_experiment_with_multiple_seeds(
        ExperimentConfig(
            name="FedAvg (IID Data)",
            num_rounds=30,
            num_clients=5,
            local_epochs=3,
            learning_rate=0.1,
            batch_size=32,
            algorithm="fedavg",
            non_iid_degree=0.0
        ),
        num_runs=num_runs
    )

    # Experiment 2: FedAvg with non-IID data
    results["fedavg_noniid"] = run_experiment_with_multiple_seeds(
        ExperimentConfig(
            name="FedAvg (Non-IID Data)",
            num_rounds=30,
            num_clients=5,
            local_epochs=3,
            learning_rate=0.1,
            batch_size=32,
            algorithm="fedavg",
            non_iid_degree=0.8
        ),
        num_runs=num_runs
    )

    # Experiment 3: FedProx with non-IID data
    results["fedprox_noniid"] = run_experiment_with_multiple_seeds(
        ExperimentConfig(
            name="FedProx (Non-IID Data, μ=0.1)",
            num_rounds=30,
            num_clients=5,
            local_epochs=3,
            learning_rate=0.1,
            batch_size=32,
            algorithm="fedprox",
            mu=0.1,
            non_iid_degree=0.8
        ),
        num_runs=num_runs
    )

    # Experiment 3b: SCAFFOLD with non-IID data
    results["scaffold_noniid"] = run_experiment_with_multiple_seeds(
        ExperimentConfig(
            name="SCAFFOLD (Non-IID Data)",
            num_rounds=30,
            num_clients=5,
            local_epochs=3,
            learning_rate=0.1,
            batch_size=32,
            algorithm="scaffold",
            non_iid_degree=0.8
        ),
        num_runs=num_runs
    )

    # Experiment 3c: FedNova with non-IID data
    results["fednova_noniid"] = run_experiment_with_multiple_seeds(
        ExperimentConfig(
            name="FedNova (Non-IID Data)",
            num_rounds=30,
            num_clients=5,
            local_epochs=3,
            learning_rate=0.1,
            batch_size=32,
            algorithm="fednova",
            non_iid_degree=0.8
        ),
        num_runs=num_runs
    )

    # Experiment 4: FedAvg + Differential Privacy (ε=10)
    results["fedavg_dp10"] = run_experiment_with_multiple_seeds(
        ExperimentConfig(
            name="FedAvg + DP (ε=10)",
            num_rounds=30,
            num_clients=5,
            local_epochs=3,
            learning_rate=0.1,
            batch_size=32,
            algorithm="fedavg",
            dp_epsilon=10.0,
            non_iid_degree=0.5
        ),
        num_runs=num_runs
    )

    # Experiment 5: FedAvg + Differential Privacy (ε=1)
    results["fedavg_dp1"] = run_experiment_with_multiple_seeds(
        ExperimentConfig(
            name="FedAvg + DP (ε=1, Strong Privacy)",
            num_rounds=30,
            num_clients=5,
            local_epochs=3,
            learning_rate=0.1,
            batch_size=32,
            algorithm="fedavg",
            dp_epsilon=1.0,
            non_iid_degree=0.5
        ),
        num_runs=num_runs
    )

    # Experiment 6: Scalability - 3 clients
    results["scale_3"] = run_experiment_with_multiple_seeds(
        ExperimentConfig(
            name="Scalability (3 Hospitals)",
            num_rounds=30,
            num_clients=3,
            local_epochs=3,
            learning_rate=0.1,
            batch_size=32,
            algorithm="fedavg",
            non_iid_degree=0.5
        ),
        num_runs=num_runs
    )

    # Experiment 7: Scalability - 5 clients
    results["scale_5"] = run_experiment_with_multiple_seeds(
        ExperimentConfig(
            name="Scalability (5 Hospitals)",
            num_rounds=30,
            num_clients=5,
            local_epochs=3,
            learning_rate=0.1,
            batch_size=32,
            algorithm="fedavg",
            non_iid_degree=0.5
        ),
        num_runs=num_runs
    )

    # Experiment 8: Scalability - 7 clients
    results["scale_7"] = run_experiment_with_multiple_seeds(
        ExperimentConfig(
            name="Scalability (7 Hospitals)",
            num_rounds=30,
            num_clients=7,
            local_epochs=3,
            learning_rate=0.1,
            batch_size=32,
            algorithm="fedavg",
            non_iid_degree=0.5
        ),
        num_runs=num_runs
    )

    return results


def generate_latex_table(results: Dict[str, ExperimentResult]) -> str:
    """Generate LaTeX table for paper with std dev and multiple metrics."""
    table = r"""
\begin{table}[htbp]
\centering
\caption{Comprehensive Experimental Results}
\label{tab:main_results}
\small
\begin{tabular}{lccccc}
\toprule
\textbf{Configuration} & \textbf{Acc.} & \textbf{F1} & \textbf{AUC} & \textbf{Prec.} & \textbf{Rec.} \\
\midrule
"""

    row_configs = [
        ("fedavg_iid", "FedAvg (IID)"),
        ("fedavg_noniid", "FedAvg (Non-IID)"),
        ("fedprox_noniid", "FedProx ($\\mu$=0.1)"),
        ("scaffold_noniid", "SCAFFOLD"),
        ("fednova_noniid", "FedNova"),
        ("fedavg_dp10", "FedAvg + DP ($\\varepsilon$=10)"),
        ("fedavg_dp1", "FedAvg + DP ($\\varepsilon$=1)"),
    ]

    for key, label in row_configs:
        if key not in results:
            continue
        r = results[key]
        # Format with std dev if available
        if r.accuracy_std > 0:
            acc_str = f"{r.final_accuracy:.1%}$\\pm${r.accuracy_std:.2f}"
            f1_str = f"{r.f1_score:.2f}$\\pm${r.f1_std:.2f}"
            auc_str = f"{r.auc_score:.2f}$\\pm${r.auc_std:.2f}"
        else:
            acc_str = f"{r.final_accuracy:.1%}"
            f1_str = f"{r.f1_score:.2f}"
            auc_str = f"{r.auc_score:.2f}"

        table += f"{label} & {acc_str} & {f1_str} & {auc_str} & {r.precision:.2f} & {r.recall:.2f} \\\\\n"

    table += r"""
\bottomrule
\end{tabular}

\vspace{1mm}
\footnotesize{5 hospitals, 50 rounds, 3 local epochs, batch size 32. Gradient clipping $C$=1.0. Results are mean $\pm$ std over 5 runs.}
\end{table}
"""
    return table


def generate_latex_privacy_table(results: Dict[str, ExperimentResult]) -> str:
    """Generate privacy-utility tradeoff table."""
    table = r"""
\begin{table}[htbp]
\centering
\caption{Privacy-Utility Tradeoff Analysis}
\label{tab:privacy_utility}
\small
\begin{tabular}{lcccc}
\toprule
\textbf{$\varepsilon$} & \textbf{Accuracy} & \textbf{F1} & \textbf{AUC} & \textbf{Acc.~Drop} \\
\midrule
"""

    baseline = results.get("fedavg_noniid", results.get("fedavg_iid"))
    if baseline is None:
        return "% No baseline experiments available for privacy table\n"
    baseline_acc = baseline.final_accuracy

    configs = [
        ("fedavg_noniid", "$\\infty$ (No DP)", 0),
        ("fedavg_dp10", "10", 10),
        ("fedavg_dp1", "1 (Strong)", 1),
    ]

    for key, eps_label, eps_val in configs:
        if key not in results:
            continue
        r = results[key]
        acc_drop = baseline_acc - r.final_accuracy
        acc_drop_str = f"{acc_drop*100:.1f}pp" if eps_val > 0 else "---"

        if r.accuracy_std > 0:
            acc_str = f"{r.final_accuracy:.1%}$\\pm${r.accuracy_std:.2f}"
            f1_str = f"{r.f1_score:.2f}$\\pm${r.f1_std:.2f}"
            auc_str = f"{r.auc_score:.2f}$\\pm${r.auc_std:.2f}"
        else:
            acc_str = f"{r.final_accuracy:.1%}"
            f1_str = f"{r.f1_score:.2f}"
            auc_str = f"{r.auc_score:.2f}"

        table += f"{eps_label} & {acc_str} & {f1_str} & {auc_str} & {acc_drop_str} \\\\\n"

    table += r"""
\bottomrule
\end{tabular}
\end{table}
"""
    return table


def generate_latex_scalability_table(results: Dict[str, ExperimentResult]) -> str:
    """Generate scalability results table with std dev."""
    # Check which scalability experiments are available
    scale_keys = [k for k in ["scale_3", "scale_5", "scale_7"] if k in results]

    if not scale_keys:
        return "% No scalability experiments available\n"

    table = r"""
\begin{table}[htbp]
\centering
\caption{Scalability Analysis}
\label{tab:scalability}
\small
\begin{tabular}{lcccc}
\toprule
\textbf{Hospitals} & \textbf{Accuracy} & \textbf{Std.~Dev.} & \textbf{Time (s)} & \textbf{Comm.~(KB)} \\
\midrule
"""

    for key in scale_keys:
        r = results[key]
        n_clients = r.config.num_clients
        # Client variance (between clients in same run)
        client_std = np.std(r.client_accuracies)
        # Estimate communication: ~2.3KB per client per round (model params)
        comm_kb = n_clients * r.config.num_rounds * 2.3 / r.config.num_rounds

        if r.accuracy_std > 0:
            acc_str = f"{r.final_accuracy:.1%}$\\pm${r.accuracy_std:.2f}"
        else:
            acc_str = f"{r.final_accuracy:.1%}"

        table += f"{n_clients} & {acc_str} & {client_std:.3f} & {r.total_time_seconds:.2f} & {comm_kb:.1f} \\\\\n"

    table += r"""
\bottomrule
\end{tabular}
\end{table}
"""
    return table


def generate_latex_fedprox_table(results: Dict[str, ExperimentResult]) -> str:
    """Generate FedProx comparison table."""
    table = r"""
\begin{table}[htbp]
\centering
\caption{FedProx Proximal Term ($\\mu$) Impact}
\label{tab:fedprox}
\small
\begin{tabular}{lccc}
\toprule
\textbf{Algorithm} & \textbf{Accuracy} & \textbf{F1} & \textbf{Client Std.} \\
\midrule
"""

    configs = [
        ("fedavg_noniid", "FedAvg ($\\mu$=0)"),
        ("fedprox_noniid", "FedProx $\\mu$=0.1"),
    ]

    for key, label in configs:
        if key not in results:
            continue
        r = results[key]
        client_std = np.std(r.client_accuracies)

        if r.accuracy_std > 0:
            acc_str = f"{r.final_accuracy:.1%}$\\pm${r.accuracy_std:.2f}"
            f1_str = f"{r.f1_score:.2f}$\\pm${r.f1_std:.2f}"
        else:
            acc_str = f"{r.final_accuracy:.1%}"
            f1_str = f"{r.f1_score:.2f}"

        table += f"{label} & {acc_str} & {f1_str} & {client_std:.3f} \\\\\n"

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
            "accuracy_std": result.accuracy_std,
            "loss_std": result.loss_std,
            "f1_score": result.f1_score,
            "f1_std": result.f1_std,
            "auc_score": result.auc_score,
            "auc_std": result.auc_std,
            "precision": result.precision,
            "recall": result.recall,
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

    with open(output_dir / "table_privacy_utility.tex", "w") as f:
        f.write(generate_latex_privacy_table(results))

    with open(output_dir / "table_fedprox.tex", "w") as f:
        f.write(generate_latex_fedprox_table(results))

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
        ax.plot(rounds, results["fedprox_noniid"].round_losses, 'g-.', label='FedProx', linewidth=2)
        if "scaffold_noniid" in results:
            ax.plot(rounds, results["scaffold_noniid"].round_losses, 'm:', label='SCAFFOLD', linewidth=2)
        if "fednova_noniid" in results:
            ax.plot(rounds, results["fednova_noniid"].round_losses, 'c-', label='FedNova', linewidth=2, alpha=0.7)
        ax.set_xlabel('Round', fontsize=11)
        ax.set_ylabel('Loss', fontsize=11)
        ax.set_title('(a) Training Loss Convergence', fontsize=12)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(1, 30)

        # Accuracy curves
        ax = axes[1]
        ax.plot(rounds, results["fedavg_iid"].round_accuracies, 'b-', label='FedAvg (IID)', linewidth=2)
        ax.plot(rounds, results["fedavg_noniid"].round_accuracies, 'r--', label='FedAvg (Non-IID)', linewidth=2)
        ax.plot(rounds, results["fedprox_noniid"].round_accuracies, 'g-.', label='FedProx', linewidth=2)
        if "scaffold_noniid" in results:
            ax.plot(rounds, results["scaffold_noniid"].round_accuracies, 'm:', label='SCAFFOLD', linewidth=2)
        if "fednova_noniid" in results:
            ax.plot(rounds, results["fednova_noniid"].round_accuracies, 'c-', label='FedNova', linewidth=2, alpha=0.7)
        ax.set_xlabel('Round', fontsize=11)
        ax.set_ylabel('Accuracy', fontsize=11)
        ax.set_title('(b) Test Accuracy', fontsize=12)
        ax.legend(fontsize=8, loc='lower right')
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
    """Print summary for paper with statistical significance."""
    print("\n" + "="*70)
    print("EXPERIMENTAL RESULTS SUMMARY (with statistical significance)")
    print("="*70)

    print("\n1. ALGORITHM COMPARISON (Non-IID Data)")
    print("-" * 50)
    fedavg = results["fedavg_noniid"]
    fedprox = results["fedprox_noniid"]
    scaffold = results.get("scaffold_noniid")
    fednova = results.get("fednova_noniid")

    print(f"   FedAvg:   {fedavg.final_accuracy:.1%} ± {fedavg.accuracy_std:.3f}")
    print(f"             F1={fedavg.f1_score:.2f}±{fedavg.f1_std:.3f}, AUC={fedavg.auc_score:.2f}±{fedavg.auc_std:.3f}")
    print(f"   FedProx:  {fedprox.final_accuracy:.1%} ± {fedprox.accuracy_std:.3f}")
    print(f"             F1={fedprox.f1_score:.2f}±{fedprox.f1_std:.3f}, AUC={fedprox.auc_score:.2f}±{fedprox.auc_std:.3f}")

    if scaffold:
        print(f"   SCAFFOLD: {scaffold.final_accuracy:.1%} ± {scaffold.accuracy_std:.3f}")
        print(f"             F1={scaffold.f1_score:.2f}±{scaffold.f1_std:.3f}, AUC={scaffold.auc_score:.2f}±{scaffold.auc_std:.3f}")

    if fednova:
        print(f"   FedNova:  {fednova.final_accuracy:.1%} ± {fednova.accuracy_std:.3f}")
        print(f"             F1={fednova.f1_score:.2f}±{fednova.f1_std:.3f}, AUC={fednova.auc_score:.2f}±{fednova.auc_std:.3f}")

    # Find best algorithm
    all_algos = [("FedAvg", fedavg), ("FedProx", fedprox)]
    if scaffold:
        all_algos.append(("SCAFFOLD", scaffold))
    if fednova:
        all_algos.append(("FedNova", fednova))
    best_name, best_r = max(all_algos, key=lambda x: x[1].final_accuracy)
    improvement = (best_r.final_accuracy - fedavg.final_accuracy) / fedavg.final_accuracy * 100
    print(f"   Best: {best_name} ({improvement:+.1f}% vs FedAvg)")

    print("\n2. PRIVACY-UTILITY TRADEOFF")
    print("-" * 50)
    no_dp = results["fedavg_noniid"]
    dp10 = results["fedavg_dp10"]
    dp1 = results["fedavg_dp1"]
    print(f"   No DP: {no_dp.final_accuracy:.1%}±{no_dp.accuracy_std:.3f}")
    print(f"   ε=10:  {dp10.final_accuracy:.1%}±{dp10.accuracy_std:.3f} (drop: {(no_dp.final_accuracy-dp10.final_accuracy)*100:.1f}pp)")
    print(f"   ε=1:   {dp1.final_accuracy:.1%}±{dp1.accuracy_std:.3f} (drop: {(no_dp.final_accuracy-dp1.final_accuracy)*100:.1f}pp)")

    print("\n3. IID vs NON-IID IMPACT")
    print("-" * 50)
    iid = results["fedavg_iid"]
    noniid = results["fedavg_noniid"]
    gap = (iid.final_accuracy - noniid.final_accuracy) * 100
    print(f"   IID:     {iid.final_accuracy:.1%}±{iid.accuracy_std:.3f}")
    print(f"   Non-IID: {noniid.final_accuracy:.1%}±{noniid.accuracy_std:.3f}")
    print(f"   Performance gap: {gap:.1f}pp")

    print("\n4. SCALABILITY")
    print("-" * 50)
    for key in ["scale_3", "scale_5", "scale_7"]:
        if key in results:
            r = results[key]
            print(f"   {r.config.num_clients} hospitals: {r.final_accuracy:.1%}±{r.accuracy_std:.3f}, "
                  f"client_std={np.std(r.client_accuracies):.3f}, time={r.total_time_seconds:.2f}s")

    print("\n5. FULL METRICS TABLE")
    print("-" * 50)
    print(f"   {'Config':<20} {'Acc':<12} {'F1':<12} {'AUC':<12} {'Prec':<8} {'Rec':<8}")
    for key, r in results.items():
        if "scale" not in key:
            print(f"   {key:<20} {r.final_accuracy:.1%}±{r.accuracy_std:.2f}  "
                  f"{r.f1_score:.2f}±{r.f1_std:.2f}  {r.auc_score:.2f}±{r.auc_std:.2f}  "
                  f"{r.precision:.2f}     {r.recall:.2f}")


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

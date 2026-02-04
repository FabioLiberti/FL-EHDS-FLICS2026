#!/usr/bin/env python3
"""
FL-EHDS Extended Experimental Benchmarks
=========================================
Generates comprehensive experimental results with multiple metrics and visualizations.
"""

import json
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np

np.random.seed(42)

# ============================================================================
# DATA GENERATION
# ============================================================================

def generate_cardiovascular_data(
    n_samples: int,
    hospital_bias: float = 0.0,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic cardiovascular risk data with hospital-specific bias."""
    rng = np.random.RandomState(seed)

    age = rng.normal(55 + hospital_bias * 10, 12, n_samples)
    bmi = rng.normal(26 + hospital_bias * 3, 4, n_samples)
    systolic_bp = rng.normal(130 + hospital_bias * 10, 18, n_samples)
    glucose = rng.normal(100 + hospital_bias * 20, 25, n_samples)
    cholesterol = rng.normal(200 + hospital_bias * 25, 35, n_samples)

    X = np.column_stack([
        (age - 55) / 12,
        (bmi - 26) / 4,
        (systolic_bp - 130) / 18,
        (glucose - 100) / 25,
        (cholesterol - 200) / 35,
    ])

    risk_score = (0.3 * X[:, 0] + 0.2 * X[:, 1] + 0.25 * X[:, 2] +
                  0.15 * X[:, 3] + 0.1 * X[:, 4])
    prob = 1 / (1 + np.exp(-risk_score))
    y = (rng.random(n_samples) < prob).astype(np.float64)

    return X, y

# ============================================================================
# MODEL
# ============================================================================

class LogisticRegressionModel:
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

    def compute_metrics(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Compute comprehensive metrics."""
        pred = self.predict(X)
        pred_proba = self.predict_proba(X)

        # Basic metrics
        accuracy = np.mean(pred == y)

        # Confusion matrix elements
        tp = np.sum((pred == 1) & (y == 1))
        fp = np.sum((pred == 1) & (y == 0))
        tn = np.sum((pred == 0) & (y == 0))
        fn = np.sum((pred == 0) & (y == 1))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        # AUC-ROC approximation
        auc = self._compute_auc(y, pred_proba)

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "specificity": specificity,
            "auc_roc": auc,
        }

    def _compute_auc(self, y_true: np.ndarray, y_scores: np.ndarray) -> float:
        """Compute AUC-ROC."""
        sorted_indices = np.argsort(y_scores)[::-1]
        y_sorted = y_true[sorted_indices]

        n_pos = np.sum(y_true == 1)
        n_neg = np.sum(y_true == 0)

        if n_pos == 0 or n_neg == 0:
            return 0.5

        tpr_list = []
        fpr_list = []
        tp, fp = 0, 0

        for i in range(len(y_sorted)):
            if y_sorted[i] == 1:
                tp += 1
            else:
                fp += 1
            tpr_list.append(tp / n_pos)
            fpr_list.append(fp / n_neg)

        # Trapezoidal rule
        auc = 0
        for i in range(1, len(fpr_list)):
            auc += (fpr_list[i] - fpr_list[i-1]) * (tpr_list[i] + tpr_list[i-1]) / 2

        return auc

    def train_step(self, X: np.ndarray, y: np.ndarray, lr: float,
                   global_params: Optional[Dict] = None, mu: float = 0.0) -> float:
        pred = self.predict_proba(X)
        error = pred - y
        grad_w = X.T @ error / len(y)
        grad_b = np.mean(error)

        if mu > 0 and global_params is not None:
            grad_w += mu * (self.weights - global_params["weights"])
            grad_b += mu * (self.bias - global_params["bias"][0])

        self.weights -= lr * grad_w
        self.bias -= lr * grad_b
        return self.compute_loss(X, y)

# ============================================================================
# PRIVACY & AGGREGATION
# ============================================================================

def add_dp_noise(gradients: Dict, epsilon: float, sensitivity: float, delta: float = 1e-5):
    if epsilon <= 0:
        return gradients, 0.0
    sigma = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon
    noisy = {}
    for key, grad in gradients.items():
        noisy[key] = grad + np.random.normal(0, sigma, grad.shape)
    return noisy, epsilon

def clip_gradients(gradients: Dict, max_norm: float):
    total_norm = np.sqrt(sum(np.sum(g ** 2) for g in gradients.values()))
    clip_factor = min(1.0, max_norm / (total_norm + 1e-7))
    return {k: v * clip_factor for k, v in gradients.items()}, total_norm

def fedavg_aggregate(updates: List[Tuple[Dict, int]]) -> Dict:
    total = sum(n for _, n in updates)
    agg = {}
    for key in updates[0][0].keys():
        agg[key] = sum(u[key] * n for u, n in updates) / total
    return agg

# ============================================================================
# EXTENDED EXPERIMENTS
# ============================================================================

@dataclass
class ExtendedResult:
    config_name: str
    round_losses: List[float]
    round_accuracies: List[float]
    round_f1_scores: List[float]
    round_auc_scores: List[float]
    per_client_accuracies: List[List[float]]
    per_client_final_metrics: List[Dict[str, float]]
    gradient_norms: List[float]
    communication_rounds: int
    total_bytes_transmitted: float
    final_metrics: Dict[str, float]
    training_time: float
    epsilon_spent: float

def run_extended_experiment(
    name: str,
    num_rounds: int = 50,
    num_clients: int = 5,
    local_epochs: int = 3,
    lr: float = 0.1,
    batch_size: int = 32,
    mu: float = 0.0,
    dp_epsilon: float = 0.0,
    clip_norm: float = 1.0,
    non_iid: float = 0.5,
) -> ExtendedResult:

    print(f"\n{'='*60}")
    print(f"Running: {name}")
    print(f"{'='*60}")

    start_time = time.time()

    # Generate data
    samples = [300, 500, 400, 350, 450, 380, 420][:num_clients]
    client_data = []
    for i in range(num_clients):
        bias = non_iid * (2 * i / (num_clients - 1) - 1) if num_clients > 1 else 0
        X, y = generate_cardiovascular_data(samples[i], bias, seed=42 + i)
        client_data.append((X, y))

    X_test, y_test = generate_cardiovascular_data(500, 0.0, seed=999)

    global_model = LogisticRegressionModel()

    # Tracking
    round_losses, round_accs, round_f1s, round_aucs = [], [], [], []
    per_client_accs = []
    gradient_norms = []
    total_eps = 0.0

    for r in range(1, num_rounds + 1):
        global_params = global_model.get_params()
        updates = []
        round_loss = 0
        client_accs_round = []
        round_grad_norm = 0

        for idx, (X, y) in enumerate(client_data):
            local = LogisticRegressionModel()
            local.set_params(global_params)

            n = len(X)
            for _ in range(local_epochs):
                indices = np.random.permutation(n)
                for s in range(0, n, batch_size):
                    e = min(s + batch_size, n)
                    local.train_step(X[indices[s:e]], y[indices[s:e]], lr,
                                    global_params if mu > 0 else None, mu)

            local_params = local.get_params()
            update = {k: local_params[k] - global_params[k] for k in global_params}

            update, norm = clip_gradients(update, clip_norm)
            round_grad_norm += norm

            if dp_epsilon > 0:
                per_round_eps = dp_epsilon / num_rounds
                update, eps = add_dp_noise(update, per_round_eps, clip_norm)
                total_eps += eps

            updates.append((update, n))
            round_loss += local.compute_loss(X, y) * n
            client_accs_round.append(local.compute_metrics(X, y)["accuracy"])

        agg = fedavg_aggregate(updates)
        new_params = {k: global_params[k] + agg[k] for k in global_params}
        global_model.set_params(new_params)

        metrics = global_model.compute_metrics(X_test, y_test)
        total_samples = sum(len(d[0]) for d in client_data)

        round_losses.append(round_loss / total_samples)
        round_accs.append(metrics["accuracy"])
        round_f1s.append(metrics["f1_score"])
        round_aucs.append(metrics["auc_roc"])
        per_client_accs.append(client_accs_round)
        gradient_norms.append(round_grad_norm / num_clients)

        if r % 10 == 0:
            print(f"  Round {r:3d}: Acc={metrics['accuracy']:.3f}, "
                  f"F1={metrics['f1_score']:.3f}, AUC={metrics['auc_roc']:.3f}")

    # Final per-client metrics
    per_client_final = []
    for X, y in client_data:
        per_client_final.append(global_model.compute_metrics(X, y))

    # Communication cost estimate (simplified: gradient size * rounds * clients)
    param_count = sum(p.size for p in global_model.get_params().values())
    bytes_per_round = param_count * 4 * num_clients * 2  # float32, up + down
    total_bytes = bytes_per_round * num_rounds

    final = global_model.compute_metrics(X_test, y_test)

    return ExtendedResult(
        config_name=name,
        round_losses=round_losses,
        round_accuracies=round_accs,
        round_f1_scores=round_f1s,
        round_auc_scores=round_aucs,
        per_client_accuracies=per_client_accs,
        per_client_final_metrics=per_client_final,
        gradient_norms=gradient_norms,
        communication_rounds=num_rounds,
        total_bytes_transmitted=total_bytes,
        final_metrics=final,
        training_time=time.time() - start_time,
        epsilon_spent=total_eps,
    )

def run_all_extended():
    results = {}

    # 1. Baseline comparisons
    results["fedavg_iid"] = run_extended_experiment("FedAvg (IID)", non_iid=0.0)
    results["fedavg_noniid_low"] = run_extended_experiment("FedAvg (Non-IID Low)", non_iid=0.3)
    results["fedavg_noniid_high"] = run_extended_experiment("FedAvg (Non-IID High)", non_iid=0.8)

    # 2. FedProx comparison
    results["fedprox_001"] = run_extended_experiment("FedProx μ=0.01", mu=0.01, non_iid=0.8)
    results["fedprox_01"] = run_extended_experiment("FedProx μ=0.1", mu=0.1, non_iid=0.8)
    results["fedprox_1"] = run_extended_experiment("FedProx μ=1.0", mu=1.0, non_iid=0.8)

    # 3. Privacy analysis
    results["dp_inf"] = run_extended_experiment("No DP (ε=∞)", dp_epsilon=0, non_iid=0.5)
    results["dp_50"] = run_extended_experiment("DP ε=50", dp_epsilon=50, non_iid=0.5)
    results["dp_10"] = run_extended_experiment("DP ε=10", dp_epsilon=10, non_iid=0.5)
    results["dp_5"] = run_extended_experiment("DP ε=5", dp_epsilon=5, non_iid=0.5)
    results["dp_1"] = run_extended_experiment("DP ε=1", dp_epsilon=1, non_iid=0.5)

    # 4. Scalability
    results["scale_3"] = run_extended_experiment("3 Hospitals", num_clients=3, non_iid=0.5)
    results["scale_5"] = run_extended_experiment("5 Hospitals", num_clients=5, non_iid=0.5)
    results["scale_7"] = run_extended_experiment("7 Hospitals", num_clients=7, non_iid=0.5)

    # 5. Communication efficiency (fewer rounds)
    results["rounds_20"] = run_extended_experiment("20 Rounds", num_rounds=20, non_iid=0.5)
    results["rounds_50"] = run_extended_experiment("50 Rounds", num_rounds=50, non_iid=0.5)

    # 6. Local epochs impact
    results["epochs_1"] = run_extended_experiment("1 Local Epoch", local_epochs=1, non_iid=0.5)
    results["epochs_3"] = run_extended_experiment("3 Local Epochs", local_epochs=3, non_iid=0.5)
    results["epochs_5"] = run_extended_experiment("5 Local Epochs", local_epochs=5, non_iid=0.5)

    return results

def generate_all_figures(results: Dict[str, ExtendedResult], output_dir: Path):
    """Generate comprehensive figures."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    # Figure 1: Convergence comparison (IID vs Non-IID)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    rounds = range(1, 51)

    # 1a: Loss convergence
    ax = axes[0, 0]
    ax.plot(rounds, results["fedavg_iid"].round_losses, 'b-', label='IID', lw=2)
    ax.plot(rounds, results["fedavg_noniid_low"].round_losses, 'g--', label='Non-IID (low)', lw=2)
    ax.plot(rounds, results["fedavg_noniid_high"].round_losses, 'r-.', label='Non-IID (high)', lw=2)
    ax.set_xlabel('Round', fontsize=11)
    ax.set_ylabel('Training Loss', fontsize=11)
    ax.set_title('(a) Loss Convergence', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 1b: Accuracy convergence
    ax = axes[0, 1]
    ax.plot(rounds, results["fedavg_iid"].round_accuracies, 'b-', label='IID', lw=2)
    ax.plot(rounds, results["fedavg_noniid_low"].round_accuracies, 'g--', label='Non-IID (low)', lw=2)
    ax.plot(rounds, results["fedavg_noniid_high"].round_accuracies, 'r-.', label='Non-IID (high)', lw=2)
    ax.set_xlabel('Round', fontsize=11)
    ax.set_ylabel('Test Accuracy', fontsize=11)
    ax.set_title('(b) Accuracy Convergence', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.4, 0.7)

    # 1c: F1-Score
    ax = axes[1, 0]
    ax.plot(rounds, results["fedavg_iid"].round_f1_scores, 'b-', label='IID', lw=2)
    ax.plot(rounds, results["fedavg_noniid_low"].round_f1_scores, 'g--', label='Non-IID (low)', lw=2)
    ax.plot(rounds, results["fedavg_noniid_high"].round_f1_scores, 'r-.', label='Non-IID (high)', lw=2)
    ax.set_xlabel('Round', fontsize=11)
    ax.set_ylabel('F1-Score', fontsize=11)
    ax.set_title('(c) F1-Score Convergence', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 1d: AUC-ROC
    ax = axes[1, 1]
    ax.plot(rounds, results["fedavg_iid"].round_auc_scores, 'b-', label='IID', lw=2)
    ax.plot(rounds, results["fedavg_noniid_low"].round_auc_scores, 'g--', label='Non-IID (low)', lw=2)
    ax.plot(rounds, results["fedavg_noniid_high"].round_auc_scores, 'r-.', label='Non-IID (high)', lw=2)
    ax.set_xlabel('Round', fontsize=11)
    ax.set_ylabel('AUC-ROC', fontsize=11)
    ax.set_title('(d) AUC-ROC Convergence', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "fig_convergence_extended.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / "fig_convergence_extended.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Figure 2: FedProx μ comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.plot(rounds, results["fedavg_noniid_high"].round_accuracies, 'b-', label='FedAvg', lw=2)
    ax.plot(rounds, results["fedprox_001"].round_accuracies, 'g--', label='FedProx μ=0.01', lw=2)
    ax.plot(rounds, results["fedprox_01"].round_accuracies, 'r-.', label='FedProx μ=0.1', lw=2)
    ax.plot(rounds, results["fedprox_1"].round_accuracies, 'm:', label='FedProx μ=1.0', lw=2)
    ax.set_xlabel('Round', fontsize=11)
    ax.set_ylabel('Test Accuracy', fontsize=11)
    ax.set_title('(a) FedAvg vs FedProx Accuracy', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    mu_vals = ['FedAvg\n(μ=0)', 'μ=0.01', 'μ=0.1', 'μ=1.0']
    final_accs = [
        results["fedavg_noniid_high"].final_metrics["accuracy"],
        results["fedprox_001"].final_metrics["accuracy"],
        results["fedprox_01"].final_metrics["accuracy"],
        results["fedprox_1"].final_metrics["accuracy"],
    ]
    colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']
    bars = ax.bar(mu_vals, final_accs, color=colors, edgecolor='black')
    ax.set_ylabel('Final Accuracy', fontsize=11)
    ax.set_title('(b) FedProx μ Impact', fontsize=12)
    ax.set_ylim(0.5, 0.65)
    for bar, acc in zip(bars, final_accs):
        ax.text(bar.get_x() + bar.get_width()/2, acc + 0.005, f'{acc:.1%}',
                ha='center', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / "fig_fedprox_comparison.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / "fig_fedprox_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Figure 3: Privacy-Utility Tradeoff (detailed)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    dp_configs = ["dp_inf", "dp_50", "dp_10", "dp_5", "dp_1"]
    eps_labels = ["∞", "50", "10", "5", "1"]

    # 3a: Accuracy vs epsilon
    ax = axes[0]
    accs = [results[c].final_metrics["accuracy"] for c in dp_configs]
    ax.bar(eps_labels, accs, color=['#27ae60', '#2ecc71', '#f1c40f', '#e67e22', '#e74c3c'],
           edgecolor='black')
    ax.set_xlabel('Privacy Budget (ε)', fontsize=11)
    ax.set_ylabel('Final Accuracy', fontsize=11)
    ax.set_title('(a) Accuracy vs Privacy', fontsize=12)
    ax.set_ylim(0.45, 0.65)
    for i, acc in enumerate(accs):
        ax.text(i, acc + 0.005, f'{acc:.1%}', ha='center', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    # 3b: F1 vs epsilon
    ax = axes[1]
    f1s = [results[c].final_metrics["f1_score"] for c in dp_configs]
    ax.bar(eps_labels, f1s, color=['#27ae60', '#2ecc71', '#f1c40f', '#e67e22', '#e74c3c'],
           edgecolor='black')
    ax.set_xlabel('Privacy Budget (ε)', fontsize=11)
    ax.set_ylabel('Final F1-Score', fontsize=11)
    ax.set_title('(b) F1-Score vs Privacy', fontsize=12)
    ax.set_ylim(0.4, 0.7)
    for i, f1 in enumerate(f1s):
        ax.text(i, f1 + 0.01, f'{f1:.2f}', ha='center', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    # 3c: AUC vs epsilon
    ax = axes[2]
    aucs = [results[c].final_metrics["auc_roc"] for c in dp_configs]
    ax.bar(eps_labels, aucs, color=['#27ae60', '#2ecc71', '#f1c40f', '#e67e22', '#e74c3c'],
           edgecolor='black')
    ax.set_xlabel('Privacy Budget (ε)', fontsize=11)
    ax.set_ylabel('Final AUC-ROC', fontsize=11)
    ax.set_title('(c) AUC-ROC vs Privacy', fontsize=12)
    ax.set_ylim(0.5, 0.7)
    for i, auc in enumerate(aucs):
        ax.text(i, auc + 0.005, f'{auc:.2f}', ha='center', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / "fig_privacy_utility_detailed.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / "fig_privacy_utility_detailed.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Figure 4: Scalability analysis
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # 4a: Accuracy by number of hospitals
    ax = axes[0]
    scale_configs = ["scale_3", "scale_5", "scale_7"]
    n_hospitals = [3, 5, 7]
    scale_accs = [results[c].final_metrics["accuracy"] for c in scale_configs]
    ax.bar([str(n) for n in n_hospitals], scale_accs, color='#3498db', edgecolor='black')
    ax.set_xlabel('Number of Hospitals', fontsize=11)
    ax.set_ylabel('Final Accuracy', fontsize=11)
    ax.set_title('(a) Scalability: Accuracy', fontsize=12)
    ax.set_ylim(0.5, 0.65)
    for i, acc in enumerate(scale_accs):
        ax.text(i, acc + 0.005, f'{acc:.1%}', ha='center', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # 4b: Training time
    ax = axes[1]
    scale_times = [results[c].training_time for c in scale_configs]
    ax.bar([str(n) for n in n_hospitals], scale_times, color='#e74c3c', edgecolor='black')
    ax.set_xlabel('Number of Hospitals', fontsize=11)
    ax.set_ylabel('Training Time (s)', fontsize=11)
    ax.set_title('(b) Scalability: Time', fontsize=12)
    for i, t in enumerate(scale_times):
        ax.text(i, t + 0.02, f'{t:.2f}s', ha='center', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # 4c: Communication cost
    ax = axes[2]
    scale_bytes = [results[c].total_bytes_transmitted / 1024 for c in scale_configs]  # KB
    ax.bar([str(n) for n in n_hospitals], scale_bytes, color='#9b59b6', edgecolor='black')
    ax.set_xlabel('Number of Hospitals', fontsize=11)
    ax.set_ylabel('Communication (KB)', fontsize=11)
    ax.set_title('(c) Scalability: Communication', fontsize=12)
    for i, b in enumerate(scale_bytes):
        ax.text(i, b + 0.5, f'{b:.1f}', ha='center', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / "fig_scalability.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / "fig_scalability.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Figure 5: Per-client heterogeneity
    fig, ax = plt.subplots(figsize=(10, 5))

    client_final = results["fedavg_noniid_high"].per_client_final_metrics
    x = np.arange(len(client_final))
    width = 0.25

    accs = [c["accuracy"] for c in client_final]
    f1s = [c["f1_score"] for c in client_final]
    aucs = [c["auc_roc"] for c in client_final]

    ax.bar(x - width, accs, width, label='Accuracy', color='#3498db')
    ax.bar(x, f1s, width, label='F1-Score', color='#2ecc71')
    ax.bar(x + width, aucs, width, label='AUC-ROC', color='#e74c3c')

    ax.set_xlabel('Hospital ID', fontsize=11)
    ax.set_ylabel('Metric Value', fontsize=11)
    ax.set_title('Per-Hospital Performance (Non-IID High)', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([f'H{i+1}' for i in range(len(client_final))])
    ax.legend()
    ax.set_ylim(0.4, 0.8)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / "fig_client_heterogeneity.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / "fig_client_heterogeneity.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Figure 6: Local epochs impact
    fig, ax = plt.subplots(figsize=(8, 5))

    epoch_configs = ["epochs_1", "epochs_3", "epochs_5"]
    epochs_x = [1, 3, 5]

    for cfg, ep in zip(epoch_configs, epochs_x):
        ax.plot(range(1, 51), results[cfg].round_accuracies, label=f'E={ep}', lw=2)

    ax.set_xlabel('Communication Round', fontsize=11)
    ax.set_ylabel('Test Accuracy', fontsize=11)
    ax.set_title('Impact of Local Epochs (E)', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.4, 0.65)

    plt.tight_layout()
    plt.savefig(output_dir / "fig_local_epochs.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / "fig_local_epochs.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Figure 7: Gradient norms over rounds
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(range(1, 51), results["fedavg_iid"].gradient_norms, 'b-', label='IID', lw=2)
    ax.plot(range(1, 51), results["fedavg_noniid_high"].gradient_norms, 'r--', label='Non-IID', lw=2)

    ax.set_xlabel('Round', fontsize=11)
    ax.set_ylabel('Average Gradient Norm', fontsize=11)
    ax.set_title('Gradient Norm Evolution', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "fig_gradient_norms.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / "fig_gradient_norms.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nAll figures saved to {output_dir}")

def generate_latex_tables(results: Dict[str, ExtendedResult], output_dir: Path):
    """Generate comprehensive LaTeX tables."""

    # Table 1: Main results comparison
    table1 = r"""
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
    configs = [
        ("fedavg_iid", "FedAvg (IID)"),
        ("fedavg_noniid_low", "FedAvg (Non-IID Low)"),
        ("fedavg_noniid_high", "FedAvg (Non-IID High)"),
        ("fedprox_01", "FedProx ($\\mu$=0.1)"),
        ("dp_10", "FedAvg + DP ($\\varepsilon$=10)"),
        ("dp_1", "FedAvg + DP ($\\varepsilon$=1)"),
    ]

    for key, label in configs:
        m = results[key].final_metrics
        table1 += f"{label} & {m['accuracy']:.1%} & {m['f1_score']:.2f} & {m['auc_roc']:.2f} & {m['precision']:.2f} & {m['recall']:.2f} \\\\\n"

    table1 += r"""
\bottomrule
\end{tabular}
\end{table}
"""

    # Table 2: Privacy-utility tradeoff
    table2 = r"""
\begin{table}[htbp]
\centering
\caption{Privacy-Utility Tradeoff Analysis}
\label{tab:privacy_utility}
\small
\begin{tabular}{lccccc}
\toprule
\textbf{$\varepsilon$} & \textbf{Accuracy} & \textbf{F1-Score} & \textbf{AUC-ROC} & \textbf{Acc. Drop} & \textbf{Total $\varepsilon$} \\
\midrule
"""

    baseline_acc = results["dp_inf"].final_metrics["accuracy"]
    for key, eps_label in [("dp_inf", "$\\infty$"), ("dp_50", "50"), ("dp_10", "10"),
                           ("dp_5", "5"), ("dp_1", "1")]:
        m = results[key].final_metrics
        drop = (baseline_acc - m["accuracy"]) * 100
        eps_spent = results[key].epsilon_spent
        eps_str = "---" if eps_spent == 0 else f"{eps_spent:.1f}"
        drop_str = "---" if drop == 0 else f"{drop:.1f}pp"
        table2 += f"{eps_label} & {m['accuracy']:.1%} & {m['f1_score']:.2f} & {m['auc_roc']:.2f} & {drop_str} & {eps_str} \\\\\n"

    table2 += r"""
\bottomrule
\end{tabular}
\end{table}
"""

    # Table 3: Scalability
    table3 = r"""
\begin{table}[htbp]
\centering
\caption{Scalability Analysis}
\label{tab:scalability}
\small
\begin{tabular}{lcccc}
\toprule
\textbf{Hospitals} & \textbf{Accuracy} & \textbf{Std. Dev.} & \textbf{Time (s)} & \textbf{Comm. (KB)} \\
\midrule
"""

    for key, n in [("scale_3", 3), ("scale_5", 5), ("scale_7", 7)]:
        r = results[key]
        accs = [c["accuracy"] for c in r.per_client_final_metrics]
        std = np.std(accs)
        table3 += f"{n} & {r.final_metrics['accuracy']:.1%} & {std:.3f} & {r.training_time:.2f} & {r.total_bytes_transmitted/1024:.1f} \\\\\n"

    table3 += r"""
\bottomrule
\end{tabular}
\end{table}
"""

    # Table 4: FedProx comparison
    table4 = r"""
\begin{table}[htbp]
\centering
\caption{FedProx Proximal Term ($\mu$) Impact}
\label{tab:fedprox}
\small
\begin{tabular}{lcccc}
\toprule
\textbf{Algorithm} & \textbf{Accuracy} & \textbf{F1-Score} & \textbf{Client Std.} & \textbf{Convergence} \\
\midrule
"""

    for key, label in [("fedavg_noniid_high", "FedAvg"), ("fedprox_001", "FedProx $\\mu$=0.01"),
                       ("fedprox_01", "FedProx $\\mu$=0.1"), ("fedprox_1", "FedProx $\\mu$=1.0")]:
        r = results[key]
        accs = [c["accuracy"] for c in r.per_client_final_metrics]
        std = np.std(accs)
        # Convergence: round where accuracy first exceeds 0.55
        conv = next((i for i, a in enumerate(r.round_accuracies) if a > 0.55), 50)
        table4 += f"{label} & {r.final_metrics['accuracy']:.1%} & {r.final_metrics['f1_score']:.2f} & {std:.3f} & {conv} \\\\\n"

    table4 += r"""
\bottomrule
\end{tabular}
\end{table}
"""

    with open(output_dir / "tables_extended.tex", "w") as f:
        f.write(table1 + "\n" + table2 + "\n" + table3 + "\n" + table4)

    print(f"LaTeX tables saved to {output_dir / 'tables_extended.tex'}")

def save_all_results(results: Dict[str, ExtendedResult], output_dir: Path):
    """Save all results to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)

    results_dict = {}
    for name, r in results.items():
        results_dict[name] = {
            "config_name": r.config_name,
            "round_losses": r.round_losses,
            "round_accuracies": r.round_accuracies,
            "round_f1_scores": r.round_f1_scores,
            "round_auc_scores": r.round_auc_scores,
            "per_client_final_metrics": r.per_client_final_metrics,
            "gradient_norms": r.gradient_norms,
            "communication_rounds": r.communication_rounds,
            "total_bytes_transmitted": r.total_bytes_transmitted,
            "final_metrics": r.final_metrics,
            "training_time": r.training_time,
            "epsilon_spent": r.epsilon_spent,
        }

    with open(output_dir / "extended_results.json", "w") as f:
        json.dump(results_dict, f, indent=2)

    print(f"Results saved to {output_dir / 'extended_results.json'}")

if __name__ == "__main__":
    print("FL-EHDS Extended Experimental Benchmarks")
    print("=" * 60)
    print(f"Started at: {datetime.now().isoformat()}")

    results = run_all_extended()

    output_dir = Path(__file__).parent / "results_extended"

    save_all_results(results, output_dir)
    generate_all_figures(results, output_dir)
    generate_latex_tables(results, output_dir)

    # Print summary
    print("\n" + "=" * 60)
    print("EXTENDED RESULTS SUMMARY")
    print("=" * 60)

    print("\n1. IID vs Non-IID:")
    print(f"   IID:          {results['fedavg_iid'].final_metrics['accuracy']:.1%}")
    print(f"   Non-IID Low:  {results['fedavg_noniid_low'].final_metrics['accuracy']:.1%}")
    print(f"   Non-IID High: {results['fedavg_noniid_high'].final_metrics['accuracy']:.1%}")

    print("\n2. FedProx:")
    print(f"   FedAvg:       {results['fedavg_noniid_high'].final_metrics['accuracy']:.1%}")
    print(f"   FedProx μ=0.1:{results['fedprox_01'].final_metrics['accuracy']:.1%}")

    print("\n3. Privacy:")
    print(f"   No DP:  {results['dp_inf'].final_metrics['accuracy']:.1%}")
    print(f"   ε=10:   {results['dp_10'].final_metrics['accuracy']:.1%}")
    print(f"   ε=1:    {results['dp_1'].final_metrics['accuracy']:.1%}")

    print("\n" + "=" * 60)
    print("EXPERIMENTS COMPLETED")
    print("=" * 60)

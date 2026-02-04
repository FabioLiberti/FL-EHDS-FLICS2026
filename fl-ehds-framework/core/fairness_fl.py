#!/usr/bin/env python3
"""
FL-EHDS Fairness-Aware Federated Learning

Implements fairness-aware FL algorithms that ensure equitable performance
across all participating hospitals, critical for EHDS deployments where
smaller hospitals shouldn't be disadvantaged.

Algorithms:
1. q-FedAvg (Li et al., 2020) - Fair Resource Allocation
2. AFL (Mohri et al., 2019) - Agnostic Federated Learning
3. FedMGDA+ - Multi-gradient descent for Pareto-optimal solutions
4. PropFair - Proportional fairness objective
5. FedMinMax - Minimax fairness (worst-case optimization)

Fairness Metrics:
- Performance variance across clients
- Worst-case client accuracy
- Gini coefficient of accuracies
- Rawlsian fairness (maximin)

Author: Fabio Liberti
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from copy import deepcopy


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class FairnessMetrics:
    """Container for fairness metrics."""
    accuracies: Dict[int, float]
    mean_accuracy: float
    std_accuracy: float
    min_accuracy: float
    max_accuracy: float
    gini_coefficient: float
    variance_ratio: float  # Variance / Mean^2


@dataclass
class FairnessConfig:
    """Configuration for fairness-aware FL."""
    algorithm: str = "qfedavg"  # qfedavg, afl, fedmgda, propfair, fedminmax
    # q-FedAvg params
    q: float = 1.0  # Fairness parameter (higher = more focus on worst performers)
    lipschitz: float = 1.0  # Lipschitz constant estimate
    # AFL params
    lambda_max: float = 10.0  # Maximum client weight
    lambda_lr: float = 0.1  # Learning rate for lambda updates
    # FedMGDA params
    epsilon: float = 0.1  # Tolerance for gradient conflict
    # PropFair params
    alpha: float = 1.0  # Proportional fairness exponent
    # General
    warmup_rounds: int = 5  # Rounds before enabling fairness


@dataclass
class ClientPerformance:
    """Tracks client performance over time."""
    client_id: int
    losses: List[float] = field(default_factory=list)
    accuracies: List[float] = field(default_factory=list)
    participation_count: int = 0
    current_weight: float = 1.0


# =============================================================================
# FAIRNESS UTILITY FUNCTIONS
# =============================================================================

def compute_gini_coefficient(values: List[float]) -> float:
    """
    Compute Gini coefficient for a list of values.

    0 = perfect equality, 1 = maximum inequality.
    """
    if len(values) < 2 or sum(values) == 0:
        return 0.0

    sorted_vals = np.sort(values)
    n = len(sorted_vals)
    cumulative = np.cumsum(sorted_vals)
    gini = (2 * np.sum((np.arange(1, n + 1) * sorted_vals))) / (n * cumulative[-1]) - (n + 1) / n
    return max(0, gini)


def compute_fairness_metrics(client_accuracies: Dict[int, float]) -> FairnessMetrics:
    """Compute comprehensive fairness metrics."""
    if not client_accuracies:
        return FairnessMetrics(
            accuracies={}, mean_accuracy=0, std_accuracy=0,
            min_accuracy=0, max_accuracy=0, gini_coefficient=0, variance_ratio=0
        )

    values = list(client_accuracies.values())
    mean_acc = np.mean(values)
    std_acc = np.std(values)

    return FairnessMetrics(
        accuracies=client_accuracies,
        mean_accuracy=mean_acc,
        std_accuracy=std_acc,
        min_accuracy=np.min(values),
        max_accuracy=np.max(values),
        gini_coefficient=compute_gini_coefficient(values),
        variance_ratio=std_acc ** 2 / (mean_acc ** 2 + 1e-10)
    )


# =============================================================================
# BASE CLASS
# =============================================================================

class FairFLAlgorithm(ABC):
    """Abstract base class for fairness-aware FL algorithms."""

    def __init__(self, config: FairnessConfig):
        self.config = config
        self.global_model: Dict[str, np.ndarray] = {}
        self.client_performances: Dict[int, ClientPerformance] = {}
        self.current_round: int = 0
        self.fairness_history: List[FairnessMetrics] = []

    def initialize(self, model_template: Dict[str, np.ndarray]) -> None:
        """Initialize global model."""
        self.global_model = {k: v.copy() for k, v in model_template.items()}

    def register_client(self, client_id: int) -> None:
        """Register a client for tracking."""
        if client_id not in self.client_performances:
            self.client_performances[client_id] = ClientPerformance(client_id=client_id)

    @abstractmethod
    def compute_aggregation_weights(self,
                                    client_losses: Dict[int, float],
                                    client_gradients: Dict[int, Dict[str, np.ndarray]],
                                    sample_counts: Dict[int, int]) -> Dict[int, float]:
        """Compute fairness-aware aggregation weights."""
        pass

    def aggregate(self,
                 client_gradients: Dict[int, Dict[str, np.ndarray]],
                 client_losses: Dict[int, float],
                 sample_counts: Dict[int, int],
                 client_accuracies: Optional[Dict[int, float]] = None) -> None:
        """Perform fairness-aware aggregation."""
        # Compute weights
        weights = self.compute_aggregation_weights(
            client_losses, client_gradients, sample_counts
        )

        # Weighted aggregation
        total_weight = sum(weights.values())
        for key in self.global_model.keys():
            update = sum(
                weights[cid] * client_gradients[cid][key]
                for cid in client_gradients.keys()
            ) / total_weight
            self.global_model[key] += update

        # Update client performances
        for cid, loss in client_losses.items():
            self.register_client(cid)
            self.client_performances[cid].losses.append(loss)
            self.client_performances[cid].participation_count += 1
            self.client_performances[cid].current_weight = weights.get(cid, 1.0)

            if client_accuracies and cid in client_accuracies:
                self.client_performances[cid].accuracies.append(client_accuracies[cid])

        # Track fairness
        if client_accuracies:
            metrics = compute_fairness_metrics(client_accuracies)
            self.fairness_history.append(metrics)

        self.current_round += 1

    def get_global_model(self) -> Dict[str, np.ndarray]:
        """Return current global model."""
        return deepcopy(self.global_model)

    def get_client_weights(self) -> Dict[int, float]:
        """Get current client aggregation weights."""
        return {
            cid: perf.current_weight
            for cid, perf in self.client_performances.items()
        }


# =============================================================================
# q-FedAvg (Li et al., 2020)
# =============================================================================

class QFedAvg(FairFLAlgorithm):
    """
    Fair Resource Allocation in Federated Learning.

    Key idea: Reweight gradients to give more importance to
    clients with higher loss (worse performance).

    Objective: min Σ_k (F_k(w))^{q+1} / (q+1)

    When q > 0, this gives more weight to higher-loss clients.
    When q → ∞, this becomes minimax (worst-case) optimization.

    Reference: Li et al., "Fair Resource Allocation in
    Federated Learning", ICLR 2020.
    """

    def __init__(self, config: FairnessConfig):
        super().__init__(config)
        self.q = config.q
        self.L = config.lipschitz

    def compute_aggregation_weights(self,
                                    client_losses: Dict[int, float],
                                    client_gradients: Dict[int, Dict[str, np.ndarray]],
                                    sample_counts: Dict[int, int]) -> Dict[int, float]:
        """
        Compute q-FedAvg aggregation weights.

        Weight for client k: h_k = q * L_k^q + L * ||∇L_k||^2

        Higher loss → higher weight → more influence on global model.
        """
        weights = {}

        for cid in client_losses.keys():
            loss = max(client_losses[cid], 1e-10)  # Avoid zero

            # Compute gradient norm
            grad_norm_sq = sum(
                np.sum(g ** 2)
                for g in client_gradients[cid].values()
            )

            # q-FedAvg weight formula
            h_k = self.q * (loss ** self.q) + self.L * grad_norm_sq

            # Delta_k = L_k^q * gradient
            # Weight = h_k (but we apply L_k^q scaling to gradient)
            weight = (loss ** self.q)

            weights[cid] = weight

        return weights

    def local_train_qfedavg(self,
                           client_id: int,
                           model: Dict[str, np.ndarray],
                           data: Tuple[np.ndarray, np.ndarray],
                           epochs: int = 1,
                           lr: float = 0.01) -> Tuple[Dict[str, np.ndarray], float]:
        """
        q-FedAvg local training.

        Returns gradient scaled by loss^q.
        """
        X, y = data
        w = deepcopy(model)
        initial_w = deepcopy(model)

        # Compute initial loss
        initial_loss = self._compute_loss(w, X, y)

        # Standard local training
        for _ in range(epochs):
            batch_size = min(32, len(X))
            indices = np.random.choice(len(X), batch_size, replace=False)
            X_batch, y_batch = X[indices], y[indices]

            if 'weights' in w:
                logits = X_batch @ w['weights']
                if 'bias' in w:
                    logits += w['bias']
                probs = 1 / (1 + np.exp(-np.clip(logits, -500, 500)))

                grad_w = X_batch.T @ (probs - y_batch) / batch_size
                w['weights'] -= lr * grad_w

                if 'bias' in w:
                    grad_b = np.mean(probs - y_batch)
                    w['bias'] -= lr * grad_b

        # Compute gradient delta
        gradient = {k: w[k] - initial_w[k] for k in w.keys()}

        # Scale by loss^q (applied in aggregation)
        final_loss = self._compute_loss(w, X, y)

        return gradient, final_loss

    def _compute_loss(self,
                     model: Dict[str, np.ndarray],
                     X: np.ndarray,
                     y: np.ndarray) -> float:
        """Compute cross-entropy loss."""
        if 'weights' not in model:
            return 0.0

        logits = X @ model['weights']
        if 'bias' in model:
            logits += model['bias']

        probs = 1 / (1 + np.exp(-np.clip(logits, -500, 500)))
        eps = 1e-10
        loss = -np.mean(y * np.log(probs + eps) + (1 - y) * np.log(1 - probs + eps))

        return loss


# =============================================================================
# AFL - Agnostic Federated Learning (Mohri et al., 2019)
# =============================================================================

class AFL(FairFLAlgorithm):
    """
    Agnostic Federated Learning.

    Key idea: Optimize for any possible mixture of client distributions.
    Maintains per-client weights (lambdas) that are updated adversarially.

    min_w max_λ Σ_k λ_k * F_k(w)

    Subject to: Σ_k λ_k = 1, λ_k >= 0

    Reference: Mohri et al., "Agnostic Federated Learning",
    ICML 2019.
    """

    def __init__(self, config: FairnessConfig):
        super().__init__(config)
        self.lambdas: Dict[int, float] = {}
        self.lambda_lr = config.lambda_lr
        self.lambda_max = config.lambda_max

    def _initialize_lambdas(self, client_ids: List[int]) -> None:
        """Initialize uniform lambda weights."""
        n = len(client_ids)
        for cid in client_ids:
            if cid not in self.lambdas:
                self.lambdas[cid] = 1.0 / n

    def _update_lambdas(self, client_losses: Dict[int, float]) -> None:
        """
        Update lambda weights using exponentiated gradient ascent.

        λ_k ← λ_k * exp(η * L_k)
        Normalize: λ_k ← λ_k / Σ_j λ_j
        """
        self._initialize_lambdas(list(client_losses.keys()))

        # Exponentiated gradient ascent
        for cid, loss in client_losses.items():
            self.lambdas[cid] *= np.exp(self.lambda_lr * loss)

        # Project onto simplex (normalize)
        total = sum(self.lambdas.values())
        for cid in self.lambdas:
            self.lambdas[cid] /= total

        # Clip to prevent extreme weights
        n = len(self.lambdas)
        max_weight = self.lambda_max / n
        for cid in self.lambdas:
            self.lambdas[cid] = min(self.lambdas[cid], max_weight)

        # Re-normalize after clipping
        total = sum(self.lambdas.values())
        for cid in self.lambdas:
            self.lambdas[cid] /= total

    def compute_aggregation_weights(self,
                                    client_losses: Dict[int, float],
                                    client_gradients: Dict[int, Dict[str, np.ndarray]],
                                    sample_counts: Dict[int, int]) -> Dict[int, float]:
        """
        Compute AFL aggregation weights.

        Weights are the adversarially learned lambdas.
        """
        # Update lambdas based on current losses
        self._update_lambdas(client_losses)

        # Return current lambda weights
        return {cid: self.lambdas.get(cid, 1.0 / len(client_losses))
                for cid in client_losses.keys()}

    def get_lambda_weights(self) -> Dict[int, float]:
        """Return current lambda weights."""
        return deepcopy(self.lambdas)


# =============================================================================
# FedMGDA+ (Multi-objective Gradient Descent)
# =============================================================================

class FedMGDA(FairFLAlgorithm):
    """
    Multi-Gradient Descent Algorithm for Federated Learning.

    Key idea: Find aggregation weights that produce Pareto-optimal
    updates (no client can improve without hurting another).

    Solves: min_α Σ_k α_k * ||g_k||^2
    Subject to: Σ_k α_k = 1, α_k >= 0

    Where g_k is client k's gradient.

    Reference: Sener & Koltun, "Multi-Task Learning as
    Multi-Objective Optimization", NeurIPS 2018.
    """

    def __init__(self, config: FairnessConfig):
        super().__init__(config)
        self.epsilon = config.epsilon

    def _frank_wolfe_solver(self,
                           gradients: List[np.ndarray],
                           max_iter: int = 20) -> np.ndarray:
        """
        Frank-Wolfe algorithm to find optimal convex combination.

        Finds α that minimizes ||Σ_k α_k * g_k||^2.
        """
        n = len(gradients)
        alpha = np.ones(n) / n  # Start uniform

        for _ in range(max_iter):
            # Compute aggregate gradient direction
            agg_grad = sum(alpha[k] * gradients[k] for k in range(n))

            # Find direction of steepest descent
            inner_products = np.array([
                np.dot(gradients[k], agg_grad) for k in range(n)
            ])

            # Minimum inner product gives steepest descent direction
            s = np.zeros(n)
            s[np.argmin(inner_products)] = 1.0

            # Line search
            gamma = 2.0 / (_ + 2)  # Diminishing step size

            # Update alpha
            alpha = (1 - gamma) * alpha + gamma * s

        return alpha

    def compute_aggregation_weights(self,
                                    client_losses: Dict[int, float],
                                    client_gradients: Dict[int, Dict[str, np.ndarray]],
                                    sample_counts: Dict[int, int]) -> Dict[int, float]:
        """
        Compute MGDA aggregation weights.

        Finds Pareto-optimal weights using Frank-Wolfe.
        """
        client_ids = list(client_gradients.keys())

        # Flatten gradients
        flat_gradients = []
        for cid in client_ids:
            flat = np.concatenate([
                g.flatten() for g in client_gradients[cid].values()
            ])
            flat_gradients.append(flat)

        # Solve for optimal alpha
        alpha = self._frank_wolfe_solver(flat_gradients)

        # Convert to dict
        weights = {cid: alpha[i] for i, cid in enumerate(client_ids)}

        return weights


# =============================================================================
# PropFair - Proportional Fairness
# =============================================================================

class PropFair(FairFLAlgorithm):
    """
    Proportional Fairness in Federated Learning.

    Key idea: Optimize proportional fairness objective:

    max Σ_k log(accuracy_k)

    This naturally balances improvement across all clients.
    Clients with lower accuracy get more weight.

    Reference: Based on Nash bargaining solution from game theory.
    """

    def __init__(self, config: FairnessConfig):
        super().__init__(config)
        self.alpha = config.alpha
        self.client_accuracies: Dict[int, float] = {}

    def update_accuracies(self, accuracies: Dict[int, float]) -> None:
        """Update tracked client accuracies."""
        self.client_accuracies.update(accuracies)

    def compute_aggregation_weights(self,
                                    client_losses: Dict[int, float],
                                    client_gradients: Dict[int, Dict[str, np.ndarray]],
                                    sample_counts: Dict[int, int]) -> Dict[int, float]:
        """
        Compute proportional fairness weights.

        Weight ∝ 1 / accuracy^α

        Lower accuracy → higher weight.
        """
        weights = {}

        for cid in client_losses.keys():
            # Get accuracy (default to 0.5 if not tracked)
            acc = self.client_accuracies.get(cid, 0.5)
            acc = max(acc, 0.01)  # Avoid division by zero

            # Proportional fairness weight
            weight = 1.0 / (acc ** self.alpha)
            weights[cid] = weight

        return weights


# =============================================================================
# FedMinMax - Minimax Fairness
# =============================================================================

class FedMinMax(FairFLAlgorithm):
    """
    Minimax Fairness in Federated Learning.

    Key idea: Optimize worst-case client performance.

    min_w max_k F_k(w)

    Uses gradient aggregation that prioritizes the worst-performing
    client at each round.

    Reference: Related to robust optimization literature.
    """

    def __init__(self, config: FairnessConfig):
        super().__init__(config)
        self.temperature = 10.0  # Softmax temperature

    def compute_aggregation_weights(self,
                                    client_losses: Dict[int, float],
                                    client_gradients: Dict[int, Dict[str, np.ndarray]],
                                    sample_counts: Dict[int, int]) -> Dict[int, float]:
        """
        Compute minimax aggregation weights.

        Uses softmax over losses to approximate max.
        Higher temperature → closer to pure max.
        """
        losses = np.array(list(client_losses.values()))
        client_ids = list(client_losses.keys())

        # Softmax weighting (approximates max)
        exp_losses = np.exp(self.temperature * (losses - np.max(losses)))
        softmax_weights = exp_losses / np.sum(exp_losses)

        weights = {cid: softmax_weights[i] for i, cid in enumerate(client_ids)}

        return weights


# =============================================================================
# FAIRNESS-AWARE FL TRAINER
# =============================================================================

class FairFLTrainer:
    """
    High-level trainer for fairness-aware federated learning.

    Handles training loop, evaluation, and fairness tracking.
    """

    def __init__(self,
                 algorithm: FairFLAlgorithm,
                 client_data: Dict[int, Tuple[np.ndarray, np.ndarray]]):
        self.algorithm = algorithm
        self.client_data = client_data
        self.num_clients = len(client_data)

        # Initialize
        for cid in client_data.keys():
            self.algorithm.register_client(cid)

    def train(self,
             num_rounds: int = 50,
             local_epochs: int = 3,
             lr: float = 0.1,
             eval_every: int = 5,
             verbose: bool = True) -> Dict:
        """
        Run fairness-aware FL training.

        Returns:
            Training history with fairness metrics.
        """
        history = {
            'rounds': [],
            'mean_accuracy': [],
            'min_accuracy': [],
            'max_accuracy': [],
            'std_accuracy': [],
            'gini_coefficient': [],
            'client_weights': [],
            'per_client_accuracy': {cid: [] for cid in self.client_data.keys()}
        }

        for round_num in range(num_rounds):
            # Local training
            client_gradients = {}
            client_losses = {}
            sample_counts = {}

            for cid, (X, y) in self.client_data.items():
                model = self.algorithm.get_global_model()

                # Local training
                gradient, loss = self._local_train(model, X, y, local_epochs, lr)

                client_gradients[cid] = gradient
                client_losses[cid] = loss
                sample_counts[cid] = len(X)

            # Evaluate before aggregation
            client_accuracies = {}
            for cid, (X, y) in self.client_data.items():
                acc = self._evaluate(self.algorithm.get_global_model(), X, y)
                client_accuracies[cid] = acc

            # Update PropFair accuracies if applicable
            if isinstance(self.algorithm, PropFair):
                self.algorithm.update_accuracies(client_accuracies)

            # Fairness-aware aggregation
            self.algorithm.aggregate(
                client_gradients, client_losses, sample_counts, client_accuracies
            )

            # Record history
            if round_num % eval_every == 0 or round_num == num_rounds - 1:
                metrics = compute_fairness_metrics(client_accuracies)

                history['rounds'].append(round_num)
                history['mean_accuracy'].append(metrics.mean_accuracy)
                history['min_accuracy'].append(metrics.min_accuracy)
                history['max_accuracy'].append(metrics.max_accuracy)
                history['std_accuracy'].append(metrics.std_accuracy)
                history['gini_coefficient'].append(metrics.gini_coefficient)
                history['client_weights'].append(self.algorithm.get_client_weights())

                for cid, acc in client_accuracies.items():
                    history['per_client_accuracy'][cid].append(acc)

                if verbose:
                    weights = self.algorithm.get_client_weights()
                    weight_str = ", ".join([f"C{k}:{v:.2f}" for k, v in sorted(weights.items())])
                    print(f"Round {round_num:3d}: "
                          f"Mean={metrics.mean_accuracy:.2%} "
                          f"Min={metrics.min_accuracy:.2%} "
                          f"Max={metrics.max_accuracy:.2%} "
                          f"Gini={metrics.gini_coefficient:.3f} "
                          f"Weights=[{weight_str}]")

        return history

    def _local_train(self,
                    model: Dict[str, np.ndarray],
                    X: np.ndarray,
                    y: np.ndarray,
                    epochs: int,
                    lr: float) -> Tuple[Dict[str, np.ndarray], float]:
        """Perform local training."""
        w = deepcopy(model)
        initial_w = deepcopy(model)

        for _ in range(epochs):
            batch_size = min(32, len(X))
            indices = np.random.choice(len(X), batch_size, replace=False)
            X_batch, y_batch = X[indices], y[indices]

            if 'weights' in w:
                logits = X_batch @ w['weights']
                if 'bias' in w:
                    logits += w['bias']
                probs = 1 / (1 + np.exp(-np.clip(logits, -500, 500)))

                grad_w = X_batch.T @ (probs - y_batch) / batch_size
                w['weights'] -= lr * grad_w

                if 'bias' in w:
                    grad_b = np.mean(probs - y_batch)
                    w['bias'] -= lr * grad_b

        # Compute loss
        logits = X @ w['weights']
        if 'bias' in w:
            logits += w['bias']
        probs = 1 / (1 + np.exp(-np.clip(logits, -500, 500)))
        eps = 1e-10
        loss = -np.mean(y * np.log(probs + eps) + (1 - y) * np.log(1 - probs + eps))

        gradient = {k: w[k] - initial_w[k] for k in w.keys()}
        return gradient, loss

    def _evaluate(self,
                 model: Dict[str, np.ndarray],
                 X: np.ndarray,
                 y: np.ndarray) -> float:
        """Evaluate model accuracy."""
        if 'weights' not in model:
            return 0.5

        logits = X @ model['weights']
        if 'bias' in model:
            logits += model['bias']
        preds = (logits > 0).astype(float)

        return np.mean(preds.flatten() == y)


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

FAIR_ALGORITHMS = {
    'qfedavg': QFedAvg,
    'afl': AFL,
    'fedmgda': FedMGDA,
    'propfair': PropFair,
    'fedminmax': FedMinMax,
}


def create_fair_fl(algorithm: str,
                  config: Optional[FairnessConfig] = None,
                  **kwargs) -> FairFLAlgorithm:
    """
    Factory function to create fairness-aware FL algorithm.

    Args:
        algorithm: Algorithm name
        config: FairnessConfig object
        **kwargs: Additional config parameters
    """
    if algorithm.lower() not in FAIR_ALGORITHMS:
        raise ValueError(
            f"Unknown algorithm: {algorithm}. "
            f"Available: {list(FAIR_ALGORITHMS.keys())}"
        )

    if config is None:
        config = FairnessConfig(algorithm=algorithm.lower(), **kwargs)

    return FAIR_ALGORITHMS[algorithm.lower()](config)


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    print("FL-EHDS Fairness-Aware Federated Learning Demo")
    print("=" * 70)

    # Generate heterogeneous data (some clients have harder problems)
    np.random.seed(42)

    num_clients = 5
    client_data = {}

    # Create clients with varying difficulty
    difficulties = [0.5, 0.3, 0.8, 0.4, 0.6]  # Higher = easier

    for i in range(num_clients):
        samples = 150 + i * 50  # Varying sample sizes
        X = np.random.randn(samples, 5)
        X = np.hstack([X, np.ones((samples, 1))])

        # Label noise varies by client (harder = more noise)
        noise_level = 0.5 - 0.3 * difficulties[i]
        noise = np.random.randn(samples) * noise_level
        y = (X[:, 0] + X[:, 1] + noise > 0).astype(float)

        client_data[i] = (X, y)

    model_template = {
        'weights': np.zeros(6),
        'bias': np.zeros(1)
    }

    # Test different fairness algorithms
    algorithms_to_test = ['qfedavg', 'afl', 'fedmgda', 'propfair', 'fedminmax']

    results = {}

    for algo_name in algorithms_to_test:
        print(f"\n{'=' * 70}")
        print(f"Testing: {algo_name.upper()}")
        print("=" * 70)

        config = FairnessConfig(
            algorithm=algo_name,
            q=2.0,  # For q-FedAvg
            lambda_lr=0.1,  # For AFL
        )

        algorithm = create_fair_fl(algo_name, config)
        algorithm.initialize(model_template)

        trainer = FairFLTrainer(algorithm, client_data)
        history = trainer.train(
            num_rounds=30,
            local_epochs=3,
            lr=0.1,
            eval_every=10,
            verbose=True
        )

        results[algo_name] = history

    # Comparison Summary
    print("\n" + "=" * 70)
    print("FAIRNESS COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Algorithm':<12} {'Mean Acc.':<12} {'Min Acc.':<12} {'Gini':<10} {'Std':<10}")
    print("-" * 56)

    for algo_name, history in results.items():
        mean_acc = history['mean_accuracy'][-1]
        min_acc = history['min_accuracy'][-1]
        gini = history['gini_coefficient'][-1]
        std = history['std_accuracy'][-1]

        print(f"{algo_name:<12} {mean_acc:>10.2%} {min_acc:>10.2%} {gini:>8.4f} {std:>8.4f}")

    print("\n" + "=" * 70)
    print("Recommendations:")
    print("- q-FedAvg: Best for moderate fairness with good average performance")
    print("- AFL: Best when distribution drift is expected")
    print("- FedMinMax: Best for strict worst-case guarantees")
    print("- PropFair: Best for proportional improvements across all clients")
    print("- FedMGDA: Best for Pareto-optimal solutions")

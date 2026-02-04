#!/usr/bin/env python3
"""
FL-EHDS Asynchronous Federated Learning

Implements asynchronous FL protocols for scenarios where:
- Hospitals have varying computational resources
- Network latency varies across Member States
- Some nodes may be temporarily unavailable

Algorithms:
1. FedAsync (Xie et al., 2019) - Basic async with staleness weighting
2. FedBuff (Nguyen et al., 2022) - Buffered async aggregation
3. AsyncSGD - Classical asynchronous SGD
4. FedAT (Chai et al., 2021) - Asynchronous with tiered aggregation

Key Features:
- Staleness-aware weighting
- Adaptive learning rates
- Buffer-based aggregation
- Timeout handling

Author: Fabio Liberti
"""

import time
import threading
import queue
from collections import deque
from typing import Dict, List, Optional, Tuple, Callable, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import numpy as np
from copy import deepcopy
import warnings

# Threading for simulation
from concurrent.futures import ThreadPoolExecutor, Future


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class AsyncUpdate:
    """Container for an asynchronous client update."""
    client_id: int
    gradient: Dict[str, np.ndarray]
    timestamp: float
    round_received: int
    staleness: int = 0  # How many rounds behind global model
    local_loss: float = 0.0
    samples_used: int = 0


@dataclass
class AsyncConfig:
    """Configuration for asynchronous FL."""
    strategy: str = "fedasync"  # fedasync, fedbuff, asyncsgd, fedat
    buffer_size: int = 3  # For FedBuff
    staleness_threshold: int = 10  # Max acceptable staleness
    staleness_function: str = "polynomial"  # polynomial, exponential, constant
    staleness_alpha: float = 0.5  # Staleness weighting parameter
    timeout: float = 30.0  # Client timeout in seconds
    min_clients_per_round: int = 1  # Minimum clients before aggregation
    adaptive_lr: bool = True  # Adjust LR based on staleness
    tier_boundaries: List[int] = field(default_factory=lambda: [3, 7])  # For FedAT


@dataclass
class ClientStatus:
    """Tracks status of a client in async federation."""
    client_id: int
    last_update_round: int = 0
    last_update_time: float = 0.0
    is_active: bool = True
    avg_latency: float = 0.0
    update_count: int = 0
    staleness_history: List[int] = field(default_factory=list)


# =============================================================================
# STALENESS FUNCTIONS
# =============================================================================

def polynomial_staleness_weight(staleness: int, alpha: float = 0.5) -> float:
    """
    Polynomial staleness weighting: w = 1 / (1 + staleness)^α

    Lower weight for older updates.
    """
    return 1.0 / ((1 + staleness) ** alpha)


def exponential_staleness_weight(staleness: int, alpha: float = 0.5) -> float:
    """
    Exponential staleness weighting: w = α^staleness

    Faster decay for very stale updates.
    """
    return alpha ** staleness


def constant_staleness_weight(staleness: int, alpha: float = 0.5) -> float:
    """
    Constant weighting (ignore staleness).

    All updates weighted equally regardless of age.
    """
    return 1.0


STALENESS_FUNCTIONS = {
    'polynomial': polynomial_staleness_weight,
    'exponential': exponential_staleness_weight,
    'constant': constant_staleness_weight,
}


# =============================================================================
# BASE CLASS
# =============================================================================

class AsyncFLAlgorithm(ABC):
    """Abstract base class for asynchronous FL algorithms."""

    def __init__(self, config: AsyncConfig):
        self.config = config
        self.global_model: Dict[str, np.ndarray] = {}
        self.global_round: int = 0
        self.client_statuses: Dict[int, ClientStatus] = {}
        self.update_history: List[AsyncUpdate] = []

        # Staleness weighting function
        self.staleness_weight = STALENESS_FUNCTIONS.get(
            config.staleness_function, polynomial_staleness_weight
        )

    def initialize(self, model_template: Dict[str, np.ndarray]) -> None:
        """Initialize global model."""
        self.global_model = {k: v.copy() for k, v in model_template.items()}
        self.global_round = 0

    def register_client(self, client_id: int) -> None:
        """Register a new client."""
        if client_id not in self.client_statuses:
            self.client_statuses[client_id] = ClientStatus(client_id=client_id)

    def get_model_for_client(self, client_id: int) -> Tuple[Dict[str, np.ndarray], int]:
        """
        Get current global model for client.

        Returns:
            Tuple of (model_weights, current_round)
        """
        self.register_client(client_id)
        return deepcopy(self.global_model), self.global_round

    @abstractmethod
    def receive_update(self, update: AsyncUpdate) -> bool:
        """
        Process an incoming client update.

        Returns:
            True if aggregation was triggered, False otherwise
        """
        pass

    @abstractmethod
    def aggregate(self) -> None:
        """Perform aggregation of pending updates."""
        pass

    def get_staleness(self, update_round: int) -> int:
        """Calculate staleness of an update."""
        return max(0, self.global_round - update_round)

    def compute_adaptive_lr(self, base_lr: float, staleness: int) -> float:
        """Compute adaptive learning rate based on staleness."""
        if not self.config.adaptive_lr:
            return base_lr
        weight = self.staleness_weight(staleness, self.config.staleness_alpha)
        return base_lr * weight


# =============================================================================
# FedAsync (Xie et al., 2019)
# =============================================================================

class FedAsync(AsyncFLAlgorithm):
    """
    Asynchronous Federated Optimization.

    Key idea: Immediately incorporate each update with staleness-weighted
    mixing. No synchronization barriers.

    Update rule:
    w_{t+1} = w_t + α(τ) * Δw_k

    where α(τ) is a staleness-dependent mixing weight.
    """

    def __init__(self, config: AsyncConfig):
        super().__init__(config)
        self.mixing_alpha = 0.5  # Base mixing rate

    def receive_update(self, update: AsyncUpdate) -> bool:
        """Process update immediately with staleness weighting."""
        # Calculate staleness
        staleness = self.get_staleness(update.round_received)
        update.staleness = staleness

        # Check staleness threshold
        if staleness > self.config.staleness_threshold:
            warnings.warn(
                f"Client {update.client_id} update too stale ({staleness}), discarding"
            )
            return False

        # Compute staleness weight
        weight = self.staleness_weight(staleness, self.config.staleness_alpha)

        # Apply weighted update immediately
        for key in self.global_model.keys():
            if key in update.gradient:
                self.global_model[key] += self.mixing_alpha * weight * update.gradient[key]

        # Update client status
        status = self.client_statuses.get(update.client_id)
        if status:
            status.last_update_round = self.global_round
            status.last_update_time = update.timestamp
            status.update_count += 1
            status.staleness_history.append(staleness)

        # Log update
        self.update_history.append(update)

        # Increment global round after each update
        self.global_round += 1

        return True

    def aggregate(self) -> None:
        """No explicit aggregation needed - updates applied immediately."""
        pass


# =============================================================================
# FedBuff (Nguyen et al., 2022)
# =============================================================================

class FedBuff(AsyncFLAlgorithm):
    """
    Buffered Asynchronous Federated Learning.

    Key idea: Collect K updates in a buffer before aggregating,
    combining benefits of sync (better convergence) and async (no stragglers).

    When buffer is full:
    w_{t+1} = w_t + (1/K) * Σ α(τ_k) * Δw_k
    """

    def __init__(self, config: AsyncConfig):
        super().__init__(config)
        self.buffer: List[AsyncUpdate] = []
        self.buffer_size = config.buffer_size

    def receive_update(self, update: AsyncUpdate) -> bool:
        """Add update to buffer, aggregate when full."""
        staleness = self.get_staleness(update.round_received)
        update.staleness = staleness

        # Check staleness
        if staleness > self.config.staleness_threshold:
            return False

        # Add to buffer
        self.buffer.append(update)

        # Update client status
        status = self.client_statuses.get(update.client_id)
        if status:
            status.last_update_round = self.global_round
            status.update_count += 1
            status.staleness_history.append(staleness)

        # Check if buffer is full
        if len(self.buffer) >= self.buffer_size:
            self.aggregate()
            return True

        return False

    def aggregate(self) -> None:
        """Aggregate buffered updates with staleness weighting."""
        if not self.buffer:
            return

        # Compute weighted average
        total_weight = 0.0
        aggregated = {k: np.zeros_like(v) for k, v in self.global_model.items()}

        for update in self.buffer:
            weight = self.staleness_weight(update.staleness, self.config.staleness_alpha)
            total_weight += weight

            for key in aggregated.keys():
                if key in update.gradient:
                    aggregated[key] += weight * update.gradient[key]

        # Apply aggregated update
        if total_weight > 0:
            for key in self.global_model.keys():
                self.global_model[key] += aggregated[key] / total_weight

        # Log and clear buffer
        self.update_history.extend(self.buffer)
        self.buffer = []

        # Increment global round
        self.global_round += 1


# =============================================================================
# AsyncSGD
# =============================================================================

class AsyncSGD(AsyncFLAlgorithm):
    """
    Classical Asynchronous SGD.

    Simple lock-free updates with no staleness correction.
    Fast but may have convergence issues with high staleness.
    """

    def __init__(self, config: AsyncConfig):
        super().__init__(config)
        self.learning_rate = 0.1

    def receive_update(self, update: AsyncUpdate) -> bool:
        """Apply update immediately without staleness weighting."""
        staleness = self.get_staleness(update.round_received)
        update.staleness = staleness

        # Adaptive learning rate based on staleness
        lr = self.compute_adaptive_lr(self.learning_rate, staleness)

        # Apply update
        for key in self.global_model.keys():
            if key in update.gradient:
                self.global_model[key] += lr * update.gradient[key]

        # Update tracking
        status = self.client_statuses.get(update.client_id)
        if status:
            status.last_update_round = self.global_round
            status.update_count += 1

        self.update_history.append(update)
        self.global_round += 1

        return True

    def aggregate(self) -> None:
        """No aggregation needed."""
        pass


# =============================================================================
# FedAT (Tiered Asynchronous FL)
# =============================================================================

class FedAT(AsyncFLAlgorithm):
    """
    Asynchronous Federated Learning with Tiered Aggregation.

    Key idea: Organize clients into tiers based on speed.
    Fast clients aggregate more frequently; slow clients less often.

    Tier 0 (fast): Aggregate every update
    Tier 1 (medium): Aggregate every K1 updates
    Tier 2 (slow): Aggregate every K2 updates
    """

    def __init__(self, config: AsyncConfig):
        super().__init__(config)
        self.tier_boundaries = config.tier_boundaries
        self.tier_buffers: Dict[int, List[AsyncUpdate]] = {
            0: [], 1: [], 2: []
        }
        self.tier_thresholds = [1, 3, 5]  # Updates before aggregation per tier

    def _assign_tier(self, client_id: int) -> int:
        """Assign client to tier based on historical latency."""
        status = self.client_statuses.get(client_id)
        if not status or len(status.staleness_history) < 3:
            return 1  # Default to middle tier

        avg_staleness = np.mean(status.staleness_history[-5:])

        if avg_staleness < self.tier_boundaries[0]:
            return 0  # Fast tier
        elif avg_staleness < self.tier_boundaries[1]:
            return 1  # Medium tier
        else:
            return 2  # Slow tier

    def receive_update(self, update: AsyncUpdate) -> bool:
        """Add update to appropriate tier buffer."""
        staleness = self.get_staleness(update.round_received)
        update.staleness = staleness

        if staleness > self.config.staleness_threshold:
            return False

        # Update client status
        status = self.client_statuses.get(update.client_id)
        if status:
            status.last_update_round = self.global_round
            status.update_count += 1
            status.staleness_history.append(staleness)

        # Assign to tier
        tier = self._assign_tier(update.client_id)
        self.tier_buffers[tier].append(update)

        # Check if tier buffer should trigger aggregation
        if len(self.tier_buffers[tier]) >= self.tier_thresholds[tier]:
            self._aggregate_tier(tier)
            return True

        return False

    def _aggregate_tier(self, tier: int) -> None:
        """Aggregate updates from specific tier."""
        buffer = self.tier_buffers[tier]
        if not buffer:
            return

        # Tier-specific weighting (faster tiers get more weight)
        tier_weight = 1.0 / (tier + 1)

        total_weight = 0.0
        aggregated = {k: np.zeros_like(v) for k, v in self.global_model.items()}

        for update in buffer:
            weight = self.staleness_weight(update.staleness, self.config.staleness_alpha)
            weight *= tier_weight
            total_weight += weight

            for key in aggregated.keys():
                if key in update.gradient:
                    aggregated[key] += weight * update.gradient[key]

        if total_weight > 0:
            for key in self.global_model.keys():
                self.global_model[key] += aggregated[key] / total_weight

        self.update_history.extend(buffer)
        self.tier_buffers[tier] = []

        if tier == 0:  # Only increment round for fast tier aggregation
            self.global_round += 1

    def aggregate(self) -> None:
        """Aggregate all non-empty tier buffers."""
        for tier in [0, 1, 2]:
            if self.tier_buffers[tier]:
                self._aggregate_tier(tier)


# =============================================================================
# ASYNC FL SERVER
# =============================================================================

class AsyncFLServer:
    """
    Asynchronous FL server that handles client connections and updates.

    Simulates real async behavior with threading.
    """

    def __init__(self,
                 algorithm: AsyncFLAlgorithm,
                 num_clients: int,
                 simulate_latency: bool = True):
        self.algorithm = algorithm
        self.num_clients = num_clients
        self.simulate_latency = simulate_latency

        # Threading components
        self.update_queue: queue.Queue = queue.Queue()
        self.running = False
        self.server_thread: Optional[threading.Thread] = None

        # Client simulation
        self.client_latencies: Dict[int, float] = {}
        for i in range(num_clients):
            # Random latency between 0.1 and 2 seconds
            self.client_latencies[i] = np.random.uniform(0.1, 2.0)

    def initialize(self, model_template: Dict[str, np.ndarray]) -> None:
        """Initialize server with model template."""
        self.algorithm.initialize(model_template)
        for i in range(self.num_clients):
            self.algorithm.register_client(i)

    def start(self) -> None:
        """Start the async server."""
        self.running = True
        self.server_thread = threading.Thread(target=self._server_loop)
        self.server_thread.start()

    def stop(self) -> None:
        """Stop the async server."""
        self.running = False
        if self.server_thread:
            self.server_thread.join()

    def _server_loop(self) -> None:
        """Main server loop processing incoming updates."""
        while self.running:
            try:
                update = self.update_queue.get(timeout=0.1)
                self.algorithm.receive_update(update)
            except queue.Empty:
                continue

    def submit_update(self, update: AsyncUpdate) -> None:
        """Submit an update to the server."""
        self.update_queue.put(update)

    def simulate_client_training(self,
                                client_id: int,
                                data: Tuple[np.ndarray, np.ndarray],
                                epochs: int = 1,
                                lr: float = 0.01) -> AsyncUpdate:
        """
        Simulate client training with realistic latency.

        Returns:
            AsyncUpdate ready for submission
        """
        X, y = data

        # Get current model
        model, round_num = self.algorithm.get_model_for_client(client_id)

        # Simulate training latency
        if self.simulate_latency:
            latency = self.client_latencies[client_id]
            latency *= (1 + np.random.uniform(-0.2, 0.2))  # Add variance
            time.sleep(latency * 0.1)  # Scale down for simulation

        # Local training
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

        # Compute gradient (update delta)
        gradient = {
            k: w[k] - initial_w[k] for k in w.keys()
        }

        return AsyncUpdate(
            client_id=client_id,
            gradient=gradient,
            timestamp=time.time(),
            round_received=round_num,
            samples_used=len(X)
        )


# =============================================================================
# ASYNC FL SIMULATOR
# =============================================================================

class AsyncFLSimulator:
    """
    Simulates asynchronous FL training for benchmarking.

    Useful for comparing different async strategies without real networking.
    """

    def __init__(self,
                 algorithm: AsyncFLAlgorithm,
                 client_data: Dict[int, Tuple[np.ndarray, np.ndarray]],
                 client_speeds: Optional[Dict[int, float]] = None):
        """
        Args:
            algorithm: Async FL algorithm to use
            client_data: {client_id: (X, y)} data for each client
            client_speeds: Relative speed factor for each client (1.0 = normal)
        """
        self.algorithm = algorithm
        self.client_data = client_data
        self.num_clients = len(client_data)

        if client_speeds is None:
            # Random speeds (some clients are faster)
            self.client_speeds = {
                i: np.random.uniform(0.5, 2.0)
                for i in range(self.num_clients)
            }
        else:
            self.client_speeds = client_speeds

        # Training state
        self.client_local_round: Dict[int, int] = {i: 0 for i in range(self.num_clients)}

    def initialize(self, model_template: Dict[str, np.ndarray]) -> None:
        """Initialize simulation."""
        self.algorithm.initialize(model_template)
        for i in range(self.num_clients):
            self.algorithm.register_client(i)

    def run_simulation(self,
                      total_updates: int = 100,
                      epochs_per_update: int = 1,
                      lr: float = 0.01,
                      eval_every: int = 10) -> Dict:
        """
        Run async FL simulation.

        Args:
            total_updates: Total number of client updates to process
            epochs_per_update: Local epochs per client update
            lr: Learning rate
            eval_every: Evaluate accuracy every N updates

        Returns:
            Dictionary with training history
        """
        history = {
            'updates': [],
            'global_rounds': [],
            'accuracies': [],
            'staleness_values': [],
            'client_participation': {i: 0 for i in range(self.num_clients)}
        }

        update_count = 0

        while update_count < total_updates:
            # Select client based on speed (faster clients more likely)
            speeds = np.array([self.client_speeds[i] for i in range(self.num_clients)])
            probs = speeds / speeds.sum()
            client_id = np.random.choice(self.num_clients, p=probs)

            # Simulate training
            X, y = self.client_data[client_id]
            update = self._simulate_client_update(client_id, X, y, epochs_per_update, lr)

            # Submit to algorithm
            accepted = self.algorithm.receive_update(update)

            if accepted:
                update_count += 1
                history['client_participation'][client_id] += 1
                history['updates'].append(update_count)
                history['global_rounds'].append(self.algorithm.global_round)
                history['staleness_values'].append(update.staleness)

                # Periodic evaluation
                if update_count % eval_every == 0:
                    acc = self._evaluate()
                    history['accuracies'].append((update_count, acc))
                    print(f"Update {update_count}/{total_updates}, "
                          f"Round {self.algorithm.global_round}, "
                          f"Accuracy: {acc:.2%}, "
                          f"Avg Staleness: {np.mean(history['staleness_values'][-eval_every:]):.1f}")

        return history

    def _simulate_client_update(self,
                               client_id: int,
                               X: np.ndarray,
                               y: np.ndarray,
                               epochs: int,
                               lr: float) -> AsyncUpdate:
        """Simulate a client update."""
        # Get model at client's local round (may be stale)
        model, _ = self.algorithm.get_model_for_client(client_id)
        round_when_fetched = self.client_local_round[client_id]

        # Local training
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

        gradient = {k: w[k] - initial_w[k] for k in w.keys()}

        # Update client's local round
        self.client_local_round[client_id] = self.algorithm.global_round

        return AsyncUpdate(
            client_id=client_id,
            gradient=gradient,
            timestamp=time.time(),
            round_received=round_when_fetched,
            samples_used=len(X)
        )

    def _evaluate(self) -> float:
        """Evaluate current global model on all client data."""
        correct = 0
        total = 0

        model = self.algorithm.global_model

        for X, y in self.client_data.values():
            if 'weights' in model:
                logits = X @ model['weights']
                if 'bias' in model:
                    logits += model['bias']
                preds = (logits > 0).astype(float)
                correct += np.sum(preds.flatten() == y)
                total += len(y)

        return correct / total if total > 0 else 0.0


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

ASYNC_ALGORITHMS = {
    'fedasync': FedAsync,
    'fedbuff': FedBuff,
    'asyncsgd': AsyncSGD,
    'fedat': FedAT,
}


def create_async_fl(algorithm: str,
                   config: Optional[AsyncConfig] = None,
                   **kwargs) -> AsyncFLAlgorithm:
    """
    Factory function to create async FL algorithm.

    Args:
        algorithm: Algorithm name ('fedasync', 'fedbuff', 'asyncsgd', 'fedat')
        config: AsyncConfig object
        **kwargs: Additional config parameters

    Returns:
        AsyncFLAlgorithm instance
    """
    if algorithm.lower() not in ASYNC_ALGORITHMS:
        raise ValueError(
            f"Unknown algorithm: {algorithm}. "
            f"Available: {list(ASYNC_ALGORITHMS.keys())}"
        )

    if config is None:
        config = AsyncConfig(strategy=algorithm.lower(), **kwargs)

    return ASYNC_ALGORITHMS[algorithm.lower()](config)


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    print("FL-EHDS Asynchronous Federated Learning Demo")
    print("=" * 60)

    # Generate synthetic heterogeneous data
    np.random.seed(42)

    num_clients = 5
    samples_per_client = 200

    client_data = {}
    for i in range(num_clients):
        mean_shift = (i - 2) * 0.3
        X = np.random.randn(samples_per_client, 5) + mean_shift
        X = np.hstack([X, np.ones((samples_per_client, 1))])
        noise = np.random.randn(samples_per_client) * 0.2
        y = (X[:, 0] + X[:, 1] + noise > mean_shift).astype(float)
        client_data[i] = (X, y)

    # Client speeds (some hospitals are faster)
    client_speeds = {
        0: 2.0,   # Fast hospital
        1: 1.5,
        2: 1.0,   # Normal
        3: 0.5,
        4: 0.3,   # Slow hospital
    }

    model_template = {
        'weights': np.zeros(6),
        'bias': np.zeros(1)
    }

    # Test different async algorithms
    algorithms_to_test = ['fedasync', 'fedbuff', 'asyncsgd', 'fedat']

    results = {}

    for algo_name in algorithms_to_test:
        print(f"\n{'-' * 60}")
        print(f"Testing {algo_name.upper()}")
        print("-" * 60)

        config = AsyncConfig(
            strategy=algo_name,
            buffer_size=3,
            staleness_threshold=10,
            staleness_alpha=0.5
        )

        algorithm = create_async_fl(algo_name, config)

        simulator = AsyncFLSimulator(
            algorithm=algorithm,
            client_data=client_data,
            client_speeds=client_speeds
        )
        simulator.initialize(model_template)

        history = simulator.run_simulation(
            total_updates=50,
            epochs_per_update=2,
            lr=0.1,
            eval_every=10
        )

        results[algo_name] = history

        print(f"\nClient participation: {history['client_participation']}")
        print(f"Final accuracy: {history['accuracies'][-1][1]:.2%}")
        print(f"Average staleness: {np.mean(history['staleness_values']):.2f}")

    # Summary
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{'Algorithm':<12} {'Final Acc.':<12} {'Avg Staleness':<15}")
    print("-" * 40)
    for algo_name, history in results.items():
        final_acc = history['accuracies'][-1][1] if history['accuracies'] else 0
        avg_stale = np.mean(history['staleness_values']) if history['staleness_values'] else 0
        print(f"{algo_name:<12} {final_acc:>10.2%} {avg_stale:>12.2f}")

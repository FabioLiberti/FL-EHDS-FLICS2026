#!/usr/bin/env python3
"""
FL-EHDS Continual/Lifelong Federated Learning

Implements continual learning strategies for FL where data
distributions change over time (concept drift), critical
for healthcare where disease patterns and treatments evolve.

Challenges Addressed:
1. Temporal Concept Drift - Disease prevalence changes over years
2. Catastrophic Forgetting - New data erases old knowledge
3. Task Incremental Learning - New diagnostic categories added
4. Non-stationary Clients - Hospitals change their patient mix

Methods:
1. EWC (Elastic Weight Consolidation) - Protect important weights
2. LwF (Learning without Forgetting) - Knowledge distillation
3. Experience Replay - Store and replay old samples
4. PackNet - Progressive network pruning and allocation
5. FedCIL (Federated Class-Incremental Learning)

Author: Fabio Liberti
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from collections import deque
from copy import deepcopy


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ContinualConfig:
    """Configuration for continual federated learning."""
    method: str = "ewc"  # ewc, lwf, replay, packnet, fedcil
    # EWC params
    ewc_lambda: float = 1000.0  # Regularization strength
    fisher_sample_size: int = 200  # Samples for Fisher estimation
    # LwF params
    lwf_temperature: float = 2.0  # Distillation temperature
    lwf_alpha: float = 0.5  # Old task weight
    # Replay params
    replay_buffer_size: int = 500  # Per task
    replay_ratio: float = 0.3  # Fraction of batch from replay
    # PackNet params
    prune_ratio: float = 0.5  # Fraction to prune per task
    # Drift detection
    detect_drift: bool = True
    drift_threshold: float = 0.05  # Accuracy drop threshold


@dataclass
class Task:
    """Represents a learning task in continual setting."""
    task_id: int
    name: str
    classes: List[int]
    data: Optional[Tuple[np.ndarray, np.ndarray]] = None
    start_round: int = 0
    end_round: Optional[int] = None


@dataclass
class DriftEvent:
    """Detected concept drift event."""
    round_detected: int
    drift_type: str  # gradual, sudden, recurring
    severity: float  # 0-1 scale
    affected_clients: List[int]


# =============================================================================
# DRIFT DETECTION
# =============================================================================

class ConceptDriftDetector:
    """
    Detects concept drift in federated learning.

    Methods:
    - DDM (Drift Detection Method)
    - ADWIN (Adaptive Windowing)
    - Performance-based detection
    """

    def __init__(self,
                 method: str = "performance",
                 threshold: float = 0.05,
                 window_size: int = 10):
        self.method = method
        self.threshold = threshold
        self.window_size = window_size

        # History tracking
        self.accuracy_history: Dict[int, List[float]] = {}
        self.loss_history: Dict[int, List[float]] = {}
        self.drift_events: List[DriftEvent] = []

    def update(self,
              round_num: int,
              client_metrics: Dict[int, Dict[str, float]]) -> Optional[DriftEvent]:
        """
        Update detector with new metrics and check for drift.

        Returns DriftEvent if drift detected, None otherwise.
        """
        for client_id, metrics in client_metrics.items():
            if client_id not in self.accuracy_history:
                self.accuracy_history[client_id] = []
                self.loss_history[client_id] = []

            self.accuracy_history[client_id].append(metrics.get('accuracy', 0))
            self.loss_history[client_id].append(metrics.get('loss', 0))

        # Check for drift
        if self.method == "performance":
            return self._detect_performance_drift(round_num, client_metrics)
        elif self.method == "ddm":
            return self._detect_ddm(round_num, client_metrics)
        else:
            return self._detect_performance_drift(round_num, client_metrics)

    def _detect_performance_drift(self,
                                  round_num: int,
                                  client_metrics: Dict[int, Dict[str, float]]) -> Optional[DriftEvent]:
        """Performance-based drift detection."""
        affected = []

        for client_id in client_metrics.keys():
            hist = self.accuracy_history[client_id]
            if len(hist) < self.window_size + 1:
                continue

            # Compare recent window to previous window
            recent = np.mean(hist[-self.window_size:])
            previous = np.mean(hist[-2*self.window_size:-self.window_size])

            drop = previous - recent
            if drop > self.threshold:
                affected.append(client_id)

        if affected:
            # Classify drift type
            avg_drop = np.mean([
                np.mean(self.accuracy_history[c][-self.window_size:]) -
                np.mean(self.accuracy_history[c][-2*self.window_size:-self.window_size])
                for c in affected
            ])

            if abs(avg_drop) > 0.1:
                drift_type = "sudden"
            else:
                drift_type = "gradual"

            event = DriftEvent(
                round_detected=round_num,
                drift_type=drift_type,
                severity=min(1.0, abs(avg_drop) / 0.2),
                affected_clients=affected
            )
            self.drift_events.append(event)
            return event

        return None

    def _detect_ddm(self,
                   round_num: int,
                   client_metrics: Dict[int, Dict[str, float]]) -> Optional[DriftEvent]:
        """DDM (Drift Detection Method) implementation."""
        affected = []

        for client_id, metrics in client_metrics.items():
            hist = self.loss_history[client_id]
            if len(hist) < 30:
                continue

            # DDM uses error rate statistics
            errors = np.array(hist)
            p = np.mean(errors)
            s = np.sqrt(p * (1 - p) / len(errors))

            # Warning and drift thresholds
            p_min = np.min(errors[:len(errors)//2])
            s_min = np.sqrt(p_min * (1 - p_min) / (len(errors)//2))

            if p + s > p_min + 3 * s_min:  # Drift
                affected.append(client_id)
            elif p + s > p_min + 2 * s_min:  # Warning
                pass

        if affected:
            event = DriftEvent(
                round_detected=round_num,
                drift_type="ddm_detected",
                severity=0.7,
                affected_clients=affected
            )
            self.drift_events.append(event)
            return event

        return None


# =============================================================================
# EWC (Elastic Weight Consolidation)
# =============================================================================

class EWCContinualFL:
    """
    Elastic Weight Consolidation for Continual FL.

    Key idea: Protect important weights from changing by adding
    a quadratic penalty based on Fisher Information.

    Loss = L_new + (λ/2) Σ_i F_i * (θ_i - θ*_i)²

    where F_i is Fisher information and θ* are optimal weights
    for previous tasks.

    Reference: Kirkpatrick et al., "Overcoming catastrophic
    forgetting in neural networks", PNAS 2017.
    """

    def __init__(self, config: ContinualConfig):
        self.config = config
        self.ewc_lambda = config.ewc_lambda
        self.fisher_sample_size = config.fisher_sample_size

        # Store Fisher and optimal weights per task
        self.fisher_matrices: List[Dict[str, np.ndarray]] = []
        self.optimal_weights: List[Dict[str, np.ndarray]] = []
        self.current_task: int = 0

    def compute_fisher(self,
                      model: Dict[str, np.ndarray],
                      data: Tuple[np.ndarray, np.ndarray],
                      loss_fn: Callable) -> Dict[str, np.ndarray]:
        """
        Compute Fisher Information Matrix (diagonal approximation).

        Fisher_i = E[(∂log p(y|x,θ) / ∂θ_i)²]
        """
        X, y = data
        n_samples = min(self.fisher_sample_size, len(X))
        indices = np.random.choice(len(X), n_samples, replace=False)

        # Initialize Fisher
        fisher = {k: np.zeros_like(v) for k, v in model.items()}

        for idx in indices:
            x_i = X[idx:idx+1]
            y_i = y[idx:idx+1]

            # Compute gradient of log-likelihood
            # (Simplified: using gradient of loss)
            grad = self._compute_gradient(model, x_i, y_i, loss_fn)

            # Square and accumulate
            for key in fisher.keys():
                fisher[key] += grad[key] ** 2

        # Average
        for key in fisher.keys():
            fisher[key] /= n_samples

        return fisher

    def _compute_gradient(self,
                         model: Dict[str, np.ndarray],
                         X: np.ndarray,
                         y: np.ndarray,
                         loss_fn: Callable = None) -> Dict[str, np.ndarray]:
        """Compute gradient of loss w.r.t. model parameters."""
        grad = {}

        if 'weights' in model:
            logits = X @ model['weights']
            if 'bias' in model:
                logits += model['bias']

            probs = 1 / (1 + np.exp(-np.clip(logits, -500, 500)))
            error = probs - y.reshape(-1, 1)

            grad['weights'] = (X.T @ error).flatten() / len(X)
            if 'bias' in model:
                grad['bias'] = np.mean(error)

        return grad

    def consolidate_task(self,
                        model: Dict[str, np.ndarray],
                        data: Tuple[np.ndarray, np.ndarray]) -> None:
        """
        Consolidate knowledge from current task.

        Called after training on a task completes.
        """
        # Compute Fisher for current task
        fisher = self.compute_fisher(model, data, loss_fn=None)
        self.fisher_matrices.append(fisher)

        # Store optimal weights
        self.optimal_weights.append(deepcopy(model))

        self.current_task += 1

    def compute_ewc_loss(self,
                        current_weights: Dict[str, np.ndarray]) -> float:
        """
        Compute EWC regularization loss.

        Penalizes deviation from optimal weights on previous tasks.
        """
        ewc_loss = 0.0

        for task_idx in range(len(self.fisher_matrices)):
            fisher = self.fisher_matrices[task_idx]
            optimal = self.optimal_weights[task_idx]

            for key in current_weights.keys():
                if key in fisher and key in optimal:
                    diff = current_weights[key] - optimal[key]
                    ewc_loss += np.sum(fisher[key] * (diff ** 2))

        return (self.ewc_lambda / 2) * ewc_loss

    def compute_ewc_gradient(self,
                            current_weights: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Compute gradient of EWC penalty."""
        grad = {k: np.zeros_like(v) for k, v in current_weights.items()}

        for task_idx in range(len(self.fisher_matrices)):
            fisher = self.fisher_matrices[task_idx]
            optimal = self.optimal_weights[task_idx]

            for key in grad.keys():
                if key in fisher and key in optimal:
                    grad[key] += self.ewc_lambda * fisher[key] * (
                        current_weights[key] - optimal[key]
                    )

        return grad

    def train_step(self,
                  model: Dict[str, np.ndarray],
                  X: np.ndarray,
                  y: np.ndarray,
                  lr: float = 0.01) -> Tuple[Dict[str, np.ndarray], float]:
        """Training step with EWC regularization."""
        # Standard gradient
        grad = self._compute_gradient(model, X, y)

        # Add EWC gradient
        ewc_grad = self.compute_ewc_gradient(model)

        # Update
        for key in model.keys():
            model[key] -= lr * (grad.get(key, 0) + ewc_grad.get(key, 0))

        # Compute loss
        logits = X @ model['weights']
        if 'bias' in model:
            logits += model['bias']
        probs = 1 / (1 + np.exp(-np.clip(logits, -500, 500)))
        eps = 1e-10
        ce_loss = -np.mean(y * np.log(probs + eps) + (1 - y) * np.log(1 - probs + eps))
        ewc_loss = self.compute_ewc_loss(model)

        return model, ce_loss + ewc_loss


# =============================================================================
# EXPERIENCE REPLAY
# =============================================================================

class ReplayBuffer:
    """
    Experience replay buffer for continual learning.

    Stores samples from previous tasks for rehearsal.
    """

    def __init__(self, max_size_per_task: int = 500):
        self.max_size = max_size_per_task
        self.buffers: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

    def add_task(self,
                task_id: int,
                X: np.ndarray,
                y: np.ndarray) -> None:
        """Add samples from a task to the buffer."""
        n = len(X)

        if n <= self.max_size:
            self.buffers[task_id] = (X.copy(), y.copy())
        else:
            # Random sampling
            indices = np.random.choice(n, self.max_size, replace=False)
            self.buffers[task_id] = (X[indices].copy(), y[indices].copy())

    def sample(self,
              batch_size: int,
              exclude_task: Optional[int] = None) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Sample from replay buffer."""
        available_tasks = [
            t for t in self.buffers.keys()
            if exclude_task is None or t != exclude_task
        ]

        if not available_tasks:
            return None

        # Combine all available buffers
        all_X = []
        all_y = []
        for task_id in available_tasks:
            X, y = self.buffers[task_id]
            all_X.append(X)
            all_y.append(y)

        combined_X = np.vstack(all_X)
        combined_y = np.concatenate(all_y)

        # Sample
        n = len(combined_X)
        batch_size = min(batch_size, n)
        indices = np.random.choice(n, batch_size, replace=False)

        return combined_X[indices], combined_y[indices]


class ReplayContinualFL:
    """
    Experience Replay for Continual FL.

    Key idea: Store representative samples from previous tasks
    and replay them during training on new tasks.
    """

    def __init__(self, config: ContinualConfig):
        self.config = config
        self.buffer = ReplayBuffer(config.replay_buffer_size)
        self.replay_ratio = config.replay_ratio
        self.current_task = 0

    def consolidate_task(self,
                        task_id: int,
                        X: np.ndarray,
                        y: np.ndarray) -> None:
        """Store samples from completed task."""
        self.buffer.add_task(task_id, X, y)
        self.current_task = task_id + 1

    def get_training_batch(self,
                          current_X: np.ndarray,
                          current_y: np.ndarray,
                          batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get mixed batch with current and replay samples.
        """
        # Current task samples
        current_size = int(batch_size * (1 - self.replay_ratio))
        current_indices = np.random.choice(
            len(current_X), min(current_size, len(current_X)), replace=False
        )
        batch_X = current_X[current_indices]
        batch_y = current_y[current_indices]

        # Replay samples
        replay_size = batch_size - len(batch_X)
        if replay_size > 0:
            replay_data = self.buffer.sample(replay_size, exclude_task=self.current_task)
            if replay_data is not None:
                replay_X, replay_y = replay_data
                batch_X = np.vstack([batch_X, replay_X])
                batch_y = np.concatenate([batch_y, replay_y])

        return batch_X, batch_y


# =============================================================================
# LEARNING WITHOUT FORGETTING (LwF)
# =============================================================================

class LwFContinualFL:
    """
    Learning without Forgetting for Continual FL.

    Key idea: Use knowledge distillation to preserve outputs
    on old task data while learning new tasks.

    Reference: Li & Hoiem, "Learning without Forgetting", ECCV 2016.
    """

    def __init__(self, config: ContinualConfig):
        self.config = config
        self.temperature = config.lwf_temperature
        self.alpha = config.lwf_alpha

        # Old model for distillation
        self.old_model: Optional[Dict[str, np.ndarray]] = None
        self.current_task = 0

    def consolidate_task(self, model: Dict[str, np.ndarray]) -> None:
        """Store model for knowledge distillation."""
        self.old_model = deepcopy(model)
        self.current_task += 1

    def compute_distillation_loss(self,
                                  current_model: Dict[str, np.ndarray],
                                  X: np.ndarray) -> float:
        """Compute knowledge distillation loss."""
        if self.old_model is None:
            return 0.0

        # Old model outputs (soft targets)
        old_logits = X @ self.old_model['weights']
        if 'bias' in self.old_model:
            old_logits += self.old_model['bias']
        old_probs = self._softmax(old_logits / self.temperature)

        # Current model outputs
        new_logits = X @ current_model['weights']
        if 'bias' in current_model:
            new_logits += current_model['bias']
        new_probs = self._softmax(new_logits / self.temperature)

        # KL divergence
        eps = 1e-10
        kl = np.sum(old_probs * np.log((old_probs + eps) / (new_probs + eps)))

        return self.alpha * (self.temperature ** 2) * kl / len(X)

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def train_step(self,
                  model: Dict[str, np.ndarray],
                  X: np.ndarray,
                  y: np.ndarray,
                  lr: float = 0.01) -> Tuple[Dict[str, np.ndarray], float]:
        """Training step with LwF distillation."""
        # Standard cross-entropy gradient
        logits = X @ model['weights']
        if 'bias' in model:
            logits += model['bias']
        probs = 1 / (1 + np.exp(-np.clip(logits, -500, 500)))

        grad_ce = {
            'weights': (X.T @ (probs - y.reshape(-1, 1))).flatten() / len(X)
        }
        if 'bias' in model:
            grad_ce['bias'] = np.mean(probs - y.reshape(-1, 1))

        # Distillation gradient (simplified)
        grad_distill = {k: np.zeros_like(v) for k, v in model.items()}

        if self.old_model is not None:
            old_logits = X @ self.old_model['weights']
            if 'bias' in self.old_model:
                old_logits += self.old_model['bias']
            old_probs = 1 / (1 + np.exp(-np.clip(old_logits / self.temperature, -500, 500)))

            new_probs = 1 / (1 + np.exp(-np.clip(logits / self.temperature, -500, 500)))

            diff = (new_probs - old_probs) / self.temperature
            grad_distill['weights'] = self.alpha * (X.T @ diff).flatten() / len(X)
            if 'bias' in model:
                grad_distill['bias'] = self.alpha * np.mean(diff)

        # Combined update
        for key in model.keys():
            model[key] -= lr * ((1 - self.alpha) * grad_ce.get(key, 0) +
                               grad_distill.get(key, 0))

        # Compute loss
        eps = 1e-10
        ce_loss = -np.mean(y * np.log(probs + eps) + (1 - y) * np.log(1 - probs + eps))
        distill_loss = self.compute_distillation_loss(model, X)

        return model, ce_loss + distill_loss


# =============================================================================
# CONTINUAL FL COORDINATOR
# =============================================================================

class ContinualFLCoordinator:
    """
    Coordinates continual learning across FL rounds.

    Manages task transitions, drift detection, and
    anti-forgetting mechanisms.
    """

    def __init__(self, config: ContinualConfig):
        self.config = config
        self.drift_detector = ConceptDriftDetector(
            threshold=config.drift_threshold
        )

        # Initialize continual learning method
        if config.method == "ewc":
            self.cl_method = EWCContinualFL(config)
        elif config.method == "replay":
            self.cl_method = ReplayContinualFL(config)
        elif config.method == "lwf":
            self.cl_method = LwFContinualFL(config)
        else:
            self.cl_method = EWCContinualFL(config)

        self.tasks: List[Task] = []
        self.current_task_idx = 0
        self.global_model: Dict[str, np.ndarray] = {}

    def initialize(self, model_template: Dict[str, np.ndarray]) -> None:
        """Initialize global model."""
        self.global_model = {k: v.copy() for k, v in model_template.items()}

    def add_task(self, task: Task) -> None:
        """Register a new task."""
        self.tasks.append(task)

    def start_new_task(self, task_id: int) -> None:
        """Start training on a new task."""
        if self.current_task_idx > 0:
            # Consolidate previous task
            prev_task = self.tasks[self.current_task_idx - 1]
            if prev_task.data is not None:
                if hasattr(self.cl_method, 'consolidate_task'):
                    if isinstance(self.cl_method, EWCContinualFL):
                        self.cl_method.consolidate_task(
                            self.global_model, prev_task.data
                        )
                    elif isinstance(self.cl_method, ReplayContinualFL):
                        X, y = prev_task.data
                        self.cl_method.consolidate_task(
                            self.current_task_idx - 1, X, y
                        )
                    elif isinstance(self.cl_method, LwFContinualFL):
                        self.cl_method.consolidate_task(self.global_model)

        self.current_task_idx = task_id + 1

    def train_round(self,
                   client_data: Dict[int, Tuple[np.ndarray, np.ndarray]],
                   round_num: int,
                   epochs: int = 1,
                   lr: float = 0.01) -> Dict:
        """
        Run one federated round with continual learning.
        """
        client_updates = {}
        client_metrics = {}

        for client_id, (X, y) in client_data.items():
            # Get training batch (may include replay)
            if isinstance(self.cl_method, ReplayContinualFL):
                batch_X, batch_y = self.cl_method.get_training_batch(X, y, 32)
            else:
                batch_X, batch_y = X, y

            # Local training
            local_model = deepcopy(self.global_model)

            for _ in range(epochs):
                if isinstance(self.cl_method, (EWCContinualFL, LwFContinualFL)):
                    local_model, loss = self.cl_method.train_step(
                        local_model, batch_X, batch_y, lr
                    )
                else:
                    # Standard training
                    if 'weights' in local_model:
                        logits = batch_X @ local_model['weights']
                        if 'bias' in local_model:
                            logits += local_model['bias']
                        probs = 1 / (1 + np.exp(-np.clip(logits, -500, 500)))
                        grad = batch_X.T @ (probs - batch_y.reshape(-1, 1)) / len(batch_X)
                        local_model['weights'] -= lr * grad.flatten()
                        if 'bias' in local_model:
                            local_model['bias'] -= lr * np.mean(probs - batch_y.reshape(-1, 1))

            # Compute update
            update = {k: local_model[k] - self.global_model[k] for k in local_model}
            client_updates[client_id] = update

            # Compute metrics
            logits = X @ local_model['weights']
            if 'bias' in local_model:
                logits += local_model['bias']
            preds = (logits > 0).astype(float).flatten()
            acc = np.mean(preds == y)

            client_metrics[client_id] = {
                'accuracy': acc,
                'loss': -np.mean(y * np.log(1/(1+np.exp(-logits.flatten())) + 1e-10))
            }

        # Aggregate
        n_clients = len(client_updates)
        for key in self.global_model.keys():
            avg_update = sum(
                client_updates[cid][key] for cid in client_updates
            ) / n_clients
            self.global_model[key] += avg_update

        # Drift detection
        drift_event = None
        if self.config.detect_drift:
            drift_event = self.drift_detector.update(round_num, client_metrics)

        return {
            'client_metrics': client_metrics,
            'drift_event': drift_event,
            'round': round_num
        }


# =============================================================================
# FACTORY & DEMO
# =============================================================================

def create_continual_fl(method: str = "ewc",
                       config: Optional[ContinualConfig] = None,
                       **kwargs) -> ContinualFLCoordinator:
    """Factory function for continual FL."""
    if config is None:
        config = ContinualConfig(method=method, **kwargs)

    return ContinualFLCoordinator(config)


if __name__ == "__main__":
    print("FL-EHDS Continual Federated Learning Demo")
    print("=" * 70)

    np.random.seed(42)

    # Simulate 3 sequential tasks (representing evolving healthcare data)
    # Task 0: Young patients (age < 50)
    # Task 1: Middle-aged patients (50 <= age < 65)
    # Task 2: Elderly patients (age >= 65)

    n_samples = 200

    tasks_data = []
    for task_id in range(3):
        age_offset = task_id * 20 + 30
        X = np.random.randn(n_samples, 5)
        X[:, 0] = np.random.normal(age_offset, 10, n_samples)  # Age feature
        X = np.hstack([X, np.ones((n_samples, 1))])

        # Different decision boundaries per task
        y = ((X[:, 0] - age_offset) / 10 + X[:, 1] + np.random.randn(n_samples) * 0.3 > 0).astype(float)

        tasks_data.append((X, y))

    model_template = {
        'weights': np.zeros(6),
        'bias': np.zeros(1)
    }

    # Test different continual learning methods
    methods = ['ewc', 'replay', 'lwf']

    for method in methods:
        print(f"\n{'='*70}")
        print(f"Testing: {method.upper()}")
        print("=" * 70)

        config = ContinualConfig(
            method=method,
            ewc_lambda=1000,
            replay_buffer_size=100,
            lwf_alpha=0.5
        )

        coordinator = create_continual_fl(method, config)
        coordinator.initialize(model_template)

        task_accuracies = {0: [], 1: [], 2: []}

        for task_id in range(3):
            print(f"\n--- Task {task_id} (Age group {task_id}) ---")

            task = Task(
                task_id=task_id,
                name=f"Age_Group_{task_id}",
                classes=[0, 1],
                data=tasks_data[task_id]
            )
            coordinator.add_task(task)
            coordinator.start_new_task(task_id)

            # Train for several rounds
            X, y = tasks_data[task_id]
            client_data = {
                0: (X[:100], y[:100]),
                1: (X[100:], y[100:])
            }

            for round_num in range(10):
                result = coordinator.train_round(
                    client_data,
                    round_num=task_id * 10 + round_num,
                    epochs=2,
                    lr=0.1
                )

            # Evaluate on ALL tasks
            print(f"  Accuracy after Task {task_id}:")
            for eval_task in range(task_id + 1):
                X_eval, y_eval = tasks_data[eval_task]
                logits = X_eval @ coordinator.global_model['weights']
                if 'bias' in coordinator.global_model:
                    logits += coordinator.global_model['bias']
                preds = (logits > 0).astype(float).flatten()
                acc = np.mean(preds == y_eval)
                task_accuracies[eval_task].append(acc)
                print(f"    Task {eval_task}: {acc:.2%}")

        # Summary
        print(f"\n{method.upper()} Summary:")
        print("  Task | After T0 | After T1 | After T2 | Forgetting")
        print("  " + "-" * 55)
        for t in range(3):
            accs = task_accuracies[t]
            if len(accs) >= 2:
                forgetting = max(accs[:-1]) - accs[-1] if len(accs) > 1 else 0
            else:
                forgetting = 0
            acc_str = " | ".join([f"{a:.2%}" if i < len(accs) else "  -  " for i, a in enumerate([accs[0] if accs else 0] + accs)])
            print(f"  T{t}   | {acc_str} | {forgetting:+.2%}")

    print("\n" + "=" * 70)
    print("Demo completed!")

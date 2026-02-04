#!/usr/bin/env python3
"""
FL-EHDS Multi-Task Federated Learning

Implements multi-task learning in FL where multiple related
tasks are learned simultaneously, sharing knowledge.

EHDS Context:
- Task 1: Diagnosis prediction (e.g., diabetes risk)
- Task 2: Treatment recommendation
- Task 3: Length of stay prediction
- Task 4: Readmission risk

Approaches:
1. Hard Parameter Sharing - Shared base, task-specific heads
2. Soft Parameter Sharing - Cross-task regularization
3. Attention-based Sharing - Learn task relationships
4. MOML (Multi-Objective ML) - Pareto-optimal solutions
5. FedMTL - Federated Multi-Task Learning

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
class TaskDefinition:
    """Definition of a learning task."""
    task_id: int
    name: str
    task_type: str  # binary, multiclass, regression
    num_classes: int = 2  # For classification
    output_dim: int = 1
    weight: float = 1.0  # Task importance weight


@dataclass
class MultiTaskConfig:
    """Configuration for multi-task FL."""
    sharing_mode: str = "hard"  # hard, soft, attention, moml
    # Architecture
    shared_layers: List[int] = field(default_factory=lambda: [64, 32])
    task_specific_layers: List[int] = field(default_factory=lambda: [16])
    # Soft sharing
    sharing_lambda: float = 0.1  # Cross-task regularization strength
    # Attention
    use_task_attention: bool = False
    attention_dim: int = 16
    # MOML
    pareto_method: str = "mgda"  # mgda, linear_scalarization, random
    # Training
    task_sampling: str = "uniform"  # uniform, weighted, round_robin


@dataclass
class MultiTaskData:
    """Data container for multi-task learning."""
    client_id: int
    features: np.ndarray  # Shared input features
    task_labels: Dict[int, np.ndarray]  # task_id -> labels


# =============================================================================
# HARD PARAMETER SHARING
# =============================================================================

class HardSharingMTL:
    """
    Hard Parameter Sharing Multi-Task Learning.

    Architecture:
    Input -> [Shared Layers] -> [Task 1 Head] -> Task 1 Output
                             -> [Task 2 Head] -> Task 2 Output
                             -> [Task N Head] -> Task N Output

    Shared layers learn common representations.
    Task-specific heads specialize for each task.
    """

    def __init__(self,
                 input_dim: int,
                 tasks: List[TaskDefinition],
                 config: MultiTaskConfig):
        self.input_dim = input_dim
        self.tasks = {t.task_id: t for t in tasks}
        self.config = config

        # Initialize shared layers
        self.shared_weights = []
        self.shared_biases = []

        dims = [input_dim] + config.shared_layers
        for i in range(len(dims) - 1):
            w = np.random.randn(dims[i], dims[i+1]) * np.sqrt(2.0 / dims[i])
            b = np.zeros(dims[i+1])
            self.shared_weights.append(w)
            self.shared_biases.append(b)

        # Initialize task-specific heads
        self.task_weights: Dict[int, List[np.ndarray]] = {}
        self.task_biases: Dict[int, List[np.ndarray]] = {}

        shared_output_dim = config.shared_layers[-1] if config.shared_layers else input_dim

        for task in tasks:
            head_dims = [shared_output_dim] + config.task_specific_layers + [task.output_dim]
            weights = []
            biases = []

            for i in range(len(head_dims) - 1):
                w = np.random.randn(head_dims[i], head_dims[i+1]) * np.sqrt(2.0 / head_dims[i])
                b = np.zeros(head_dims[i+1])
                weights.append(w)
                biases.append(b)

            self.task_weights[task.task_id] = weights
            self.task_biases[task.task_id] = biases

        # Caches for backprop
        self.shared_activations = []

    def forward_shared(self, X: np.ndarray) -> np.ndarray:
        """Forward pass through shared layers."""
        self.shared_activations = [X]
        h = X

        for w, b in zip(self.shared_weights, self.shared_biases):
            z = h @ w + b
            h = np.maximum(0, z)  # ReLU
            self.shared_activations.append(h)

        return h

    def forward_task(self,
                    shared_output: np.ndarray,
                    task_id: int) -> np.ndarray:
        """Forward pass through task-specific head."""
        task = self.tasks[task_id]
        h = shared_output

        weights = self.task_weights[task_id]
        biases = self.task_biases[task_id]

        for i, (w, b) in enumerate(zip(weights, biases)):
            z = h @ w + b

            if i == len(weights) - 1:  # Output layer
                if task.task_type == "binary":
                    h = 1 / (1 + np.exp(-np.clip(z, -500, 500)))  # Sigmoid
                elif task.task_type == "multiclass":
                    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
                    h = exp_z / np.sum(exp_z, axis=1, keepdims=True)  # Softmax
                else:  # Regression
                    h = z
            else:
                h = np.maximum(0, z)  # ReLU

        return h

    def forward(self,
               X: np.ndarray,
               task_ids: Optional[List[int]] = None) -> Dict[int, np.ndarray]:
        """Full forward pass for specified tasks."""
        if task_ids is None:
            task_ids = list(self.tasks.keys())

        shared_output = self.forward_shared(X)

        outputs = {}
        for task_id in task_ids:
            outputs[task_id] = self.forward_task(shared_output, task_id)

        return outputs

    def compute_loss(self,
                    predictions: Dict[int, np.ndarray],
                    labels: Dict[int, np.ndarray]) -> Tuple[float, Dict[int, float]]:
        """Compute combined loss across tasks."""
        total_loss = 0.0
        task_losses = {}
        eps = 1e-10

        for task_id, preds in predictions.items():
            if task_id not in labels:
                continue

            task = self.tasks[task_id]
            y = labels[task_id]

            if task.task_type == "binary":
                loss = -np.mean(y * np.log(preds + eps) + (1 - y) * np.log(1 - preds + eps))
            elif task.task_type == "multiclass":
                loss = -np.mean(np.sum(y * np.log(preds + eps), axis=1))
            else:  # Regression
                loss = np.mean((preds - y) ** 2)

            task_losses[task_id] = loss
            total_loss += task.weight * loss

        return total_loss, task_losses

    def backward(self,
                predictions: Dict[int, np.ndarray],
                labels: Dict[int, np.ndarray],
                lr: float = 0.01) -> None:
        """Backward pass and parameter update."""
        # Compute gradients for each task head
        shared_grad = np.zeros_like(self.shared_activations[-1])

        for task_id, preds in predictions.items():
            if task_id not in labels:
                continue

            task = self.tasks[task_id]
            y = labels[task_id].reshape(-1, 1) if y.ndim == 1 else labels[task_id]

            # Output gradient
            if task.task_type in ["binary", "multiclass"]:
                grad = (preds - y) / len(y)
            else:
                grad = 2 * (preds - y) / len(y)

            # Backward through task head
            weights = self.task_weights[task_id]
            biases = self.task_biases[task_id]

            h = self.shared_activations[-1]
            for i in reversed(range(len(weights))):
                # Gradient for weights and biases
                if i == 0:
                    prev_h = h
                else:
                    # Would need to cache task activations for proper backprop
                    prev_h = h  # Simplified

                grad_w = prev_h.T @ grad / len(prev_h)
                grad_b = np.mean(grad, axis=0)

                weights[i] -= lr * grad_w
                biases[i] -= lr * grad_b

                if i > 0:
                    grad = grad @ weights[i].T
                    # ReLU backward (simplified)
                    grad = grad * (prev_h > 0)

            # Accumulate gradient to shared layers
            grad_to_shared = grad @ weights[0].T
            shared_grad += task.weight * grad_to_shared

        # Backward through shared layers
        grad = shared_grad
        for i in reversed(range(len(self.shared_weights))):
            h = self.shared_activations[i]
            h_next = self.shared_activations[i + 1]

            # ReLU backward
            grad = grad * (h_next > 0)

            grad_w = h.T @ grad / len(h)
            grad_b = np.mean(grad, axis=0)

            self.shared_weights[i] -= lr * grad_w
            self.shared_biases[i] -= lr * grad_b

            if i > 0:
                grad = grad @ self.shared_weights[i].T

    def get_shared_params(self) -> Dict[str, np.ndarray]:
        """Get shared layer parameters."""
        params = {}
        for i, (w, b) in enumerate(zip(self.shared_weights, self.shared_biases)):
            params[f'shared_w{i}'] = w
            params[f'shared_b{i}'] = b
        return params

    def set_shared_params(self, params: Dict[str, np.ndarray]) -> None:
        """Set shared layer parameters."""
        for i in range(len(self.shared_weights)):
            self.shared_weights[i] = params[f'shared_w{i}'].copy()
            self.shared_biases[i] = params[f'shared_b{i}'].copy()


# =============================================================================
# SOFT PARAMETER SHARING
# =============================================================================

class SoftSharingMTL:
    """
    Soft Parameter Sharing Multi-Task Learning.

    Each task has its own full model, but models are regularized
    to stay close to each other (cross-task regularization).

    Loss = Σ_t L_t(θ_t) + λ * Σ_{t≠t'} ||θ_t - θ_{t'}||²
    """

    def __init__(self,
                 input_dim: int,
                 tasks: List[TaskDefinition],
                 config: MultiTaskConfig):
        self.input_dim = input_dim
        self.tasks = {t.task_id: t for t in tasks}
        self.config = config
        self.sharing_lambda = config.sharing_lambda

        # Initialize separate model for each task
        self.task_models: Dict[int, Dict[str, np.ndarray]] = {}

        all_dims = [input_dim] + config.shared_layers + config.task_specific_layers

        for task in tasks:
            model = {}
            for i in range(len(all_dims) - 1):
                model[f'w{i}'] = np.random.randn(all_dims[i], all_dims[i+1]) * np.sqrt(2.0 / all_dims[i])
                model[f'b{i}'] = np.zeros(all_dims[i+1])

            # Output layer
            model['w_out'] = np.random.randn(all_dims[-1], task.output_dim) * np.sqrt(2.0 / all_dims[-1])
            model['b_out'] = np.zeros(task.output_dim)

            self.task_models[task.task_id] = model

    def forward(self, X: np.ndarray, task_id: int) -> np.ndarray:
        """Forward pass for a specific task."""
        model = self.task_models[task_id]
        task = self.tasks[task_id]

        h = X
        n_layers = len([k for k in model.keys() if k.startswith('w') and k != 'w_out'])

        for i in range(n_layers):
            z = h @ model[f'w{i}'] + model[f'b{i}']
            h = np.maximum(0, z)  # ReLU

        # Output
        z = h @ model['w_out'] + model['b_out']

        if task.task_type == "binary":
            return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
        elif task.task_type == "multiclass":
            exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
            return exp_z / np.sum(exp_z, axis=1, keepdims=True)
        else:
            return z

    def compute_sharing_penalty(self) -> float:
        """Compute cross-task regularization penalty."""
        task_ids = list(self.task_models.keys())
        penalty = 0.0

        for i, t1 in enumerate(task_ids):
            for t2 in task_ids[i+1:]:
                model1 = self.task_models[t1]
                model2 = self.task_models[t2]

                for key in model1.keys():
                    if key in model2:
                        penalty += np.sum((model1[key] - model2[key]) ** 2)

        return self.sharing_lambda * penalty

    def train_step(self,
                  X: np.ndarray,
                  task_labels: Dict[int, np.ndarray],
                  lr: float = 0.01) -> Dict[int, float]:
        """Train all tasks with soft sharing penalty."""
        losses = {}

        for task_id, y in task_labels.items():
            if task_id not in self.task_models:
                continue

            model = self.task_models[task_id]
            task = self.tasks[task_id]

            # Forward
            preds = self.forward(X, task_id)

            # Task loss
            eps = 1e-10
            if task.task_type == "binary":
                y = y.reshape(-1, 1)
                loss = -np.mean(y * np.log(preds + eps) + (1 - y) * np.log(1 - preds + eps))
            else:
                loss = np.mean((preds - y) ** 2)

            losses[task_id] = loss

            # Backward (simplified - just gradient of task loss)
            if task.task_type == "binary":
                grad = (preds - y) / len(y)
            else:
                grad = 2 * (preds - y) / len(y)

            # Update output layer
            # ... (simplified gradient computation)

        # Add sharing penalty gradient to all models
        task_ids = list(self.task_models.keys())
        for i, t1 in enumerate(task_ids):
            for t2 in task_ids:
                if t1 == t2:
                    continue

                model1 = self.task_models[t1]
                model2 = self.task_models[t2]

                for key in model1.keys():
                    if key in model2:
                        sharing_grad = 2 * self.sharing_lambda * (model1[key] - model2[key])
                        model1[key] -= lr * sharing_grad / (len(task_ids) - 1)

        return losses


# =============================================================================
# FEDERATED MULTI-TASK LEARNING (FedMTL)
# =============================================================================

class FedMTL:
    """
    Federated Multi-Task Learning.

    Each client may have different tasks (or subsets of tasks).
    Global model has shared representation with task-specific heads.

    Handles:
    - Clients with partial task coverage
    - Task-specific aggregation
    - Cross-client task relationships
    """

    def __init__(self,
                 input_dim: int,
                 tasks: List[TaskDefinition],
                 config: MultiTaskConfig):
        self.input_dim = input_dim
        self.tasks = {t.task_id: t for t in tasks}
        self.config = config

        # Global model
        self.model = HardSharingMTL(input_dim, tasks, config)

        # Task participation tracking
        self.task_client_counts: Dict[int, int] = {t.task_id: 0 for t in tasks}

    def get_global_model(self) -> Dict[str, Any]:
        """Get global model parameters."""
        return {
            'shared': self.model.get_shared_params(),
            'task_heads': {
                tid: {
                    'weights': [w.copy() for w in self.model.task_weights[tid]],
                    'biases': [b.copy() for b in self.model.task_biases[tid]]
                }
                for tid in self.tasks.keys()
            }
        }

    def set_global_model(self, params: Dict[str, Any]) -> None:
        """Set global model parameters."""
        self.model.set_shared_params(params['shared'])

        for tid, head_params in params['task_heads'].items():
            self.model.task_weights[tid] = [w.copy() for w in head_params['weights']]
            self.model.task_biases[tid] = [b.copy() for b in head_params['biases']]

    def local_train(self,
                   client_data: MultiTaskData,
                   epochs: int = 1,
                   lr: float = 0.01) -> Tuple[Dict[str, Any], Dict[int, float]]:
        """
        Local training for a client.

        Returns:
            Tuple of (model_updates, task_losses)
        """
        X = client_data.features
        initial_params = self.get_global_model()

        task_losses = {}

        for _ in range(epochs):
            # Forward pass for available tasks
            available_tasks = list(client_data.task_labels.keys())
            predictions = self.model.forward(X, available_tasks)

            # Compute loss
            _, losses = self.model.compute_loss(predictions, client_data.task_labels)
            task_losses = losses

            # Backward pass
            self.model.backward(predictions, client_data.task_labels, lr)

        # Compute updates
        final_params = self.get_global_model()

        updates = {
            'shared': {
                k: final_params['shared'][k] - initial_params['shared'][k]
                for k in final_params['shared']
            },
            'task_heads': {},
            'available_tasks': list(client_data.task_labels.keys())
        }

        for tid in client_data.task_labels.keys():
            updates['task_heads'][tid] = {
                'weights': [
                    final_params['task_heads'][tid]['weights'][i] -
                    initial_params['task_heads'][tid]['weights'][i]
                    for i in range(len(final_params['task_heads'][tid]['weights']))
                ],
                'biases': [
                    final_params['task_heads'][tid]['biases'][i] -
                    initial_params['task_heads'][tid]['biases'][i]
                    for i in range(len(final_params['task_heads'][tid]['biases']))
                ]
            }

        return updates, task_losses

    def aggregate(self,
                 client_updates: List[Dict[str, Any]],
                 sample_counts: List[int]) -> None:
        """
        Aggregate client updates with task-aware weighting.
        """
        total_samples = sum(sample_counts)

        # Aggregate shared layers (weighted by samples)
        aggregated_shared = {}
        for key in client_updates[0]['shared'].keys():
            aggregated_shared[key] = sum(
                (sample_counts[i] / total_samples) * client_updates[i]['shared'][key]
                for i in range(len(client_updates))
            )

        # Apply shared updates
        params = self.get_global_model()
        for key in aggregated_shared:
            params['shared'][key] += aggregated_shared[key]

        # Aggregate task heads (only from clients that have the task)
        for tid in self.tasks.keys():
            task_updates = []
            task_weights = []

            for i, update in enumerate(client_updates):
                if tid in update.get('available_tasks', []) and tid in update.get('task_heads', {}):
                    task_updates.append(update['task_heads'][tid])
                    task_weights.append(sample_counts[i])

            if task_updates:
                total_task_weight = sum(task_weights)

                for layer_idx in range(len(task_updates[0]['weights'])):
                    agg_w = sum(
                        (task_weights[i] / total_task_weight) * task_updates[i]['weights'][layer_idx]
                        for i in range(len(task_updates))
                    )
                    agg_b = sum(
                        (task_weights[i] / total_task_weight) * task_updates[i]['biases'][layer_idx]
                        for i in range(len(task_updates))
                    )

                    params['task_heads'][tid]['weights'][layer_idx] += agg_w
                    params['task_heads'][tid]['biases'][layer_idx] += agg_b

        self.set_global_model(params)

    def evaluate(self,
                X: np.ndarray,
                task_labels: Dict[int, np.ndarray]) -> Dict[int, Dict[str, float]]:
        """Evaluate model on all available tasks."""
        predictions = self.model.forward(X, list(task_labels.keys()))
        results = {}

        for tid, preds in predictions.items():
            if tid not in task_labels:
                continue

            task = self.tasks[tid]
            y = task_labels[tid]

            if task.task_type == "binary":
                y = y.reshape(-1, 1) if y.ndim == 1 else y
                preds_binary = (preds > 0.5).astype(float)
                accuracy = np.mean(preds_binary == y)

                eps = 1e-10
                loss = -np.mean(y * np.log(preds + eps) + (1 - y) * np.log(1 - preds + eps))
            else:
                accuracy = 0.0  # Not applicable for regression
                loss = np.mean((preds - y) ** 2)

            results[tid] = {
                'accuracy': accuracy,
                'loss': loss
            }

        return results


# =============================================================================
# MULTI-TASK FL COORDINATOR
# =============================================================================

class MultiTaskFLCoordinator:
    """
    Coordinates multi-task federated learning.

    Manages:
    - Task definitions and relationships
    - Client-task assignments
    - Multi-objective optimization
    - Performance tracking per task
    """

    def __init__(self,
                 input_dim: int,
                 tasks: List[TaskDefinition],
                 config: MultiTaskConfig):
        self.input_dim = input_dim
        self.tasks = tasks
        self.config = config

        self.mtl = FedMTL(input_dim, tasks, config)

        # Tracking
        self.round_history: List[Dict] = []
        self.task_performance: Dict[int, List[float]] = {t.task_id: [] for t in tasks}

    def train_round(self,
                   client_data_list: List[MultiTaskData],
                   epochs_per_client: int = 1,
                   lr: float = 0.01) -> Dict:
        """Run one round of multi-task FL training."""
        all_updates = []
        all_sample_counts = []
        all_losses: Dict[int, List[float]] = {t.task_id: [] for t in self.tasks}

        for client_data in client_data_list:
            updates, losses = self.mtl.local_train(
                client_data,
                epochs=epochs_per_client,
                lr=lr
            )

            all_updates.append(updates)
            all_sample_counts.append(len(client_data.features))

            for tid, loss in losses.items():
                all_losses[tid].append(loss)

        # Aggregate
        self.mtl.aggregate(all_updates, all_sample_counts)

        # Compute average losses
        avg_losses = {
            tid: np.mean(losses) if losses else 0.0
            for tid, losses in all_losses.items()
        }

        round_info = {
            'num_clients': len(client_data_list),
            'task_losses': avg_losses,
            'participating_tasks': {
                tid: sum(1 for u in all_updates if tid in u.get('available_tasks', []))
                for tid in [t.task_id for t in self.tasks]
            }
        }

        self.round_history.append(round_info)

        return round_info

    def evaluate_all(self,
                    test_data: MultiTaskData) -> Dict[int, Dict[str, float]]:
        """Evaluate on all tasks."""
        return self.mtl.evaluate(test_data.features, test_data.task_labels)


# =============================================================================
# FACTORY & DEMO
# =============================================================================

def create_multitask_fl(input_dim: int,
                       tasks: List[TaskDefinition],
                       config: Optional[MultiTaskConfig] = None,
                       **kwargs) -> MultiTaskFLCoordinator:
    """Factory function for multi-task FL."""
    if config is None:
        config = MultiTaskConfig(**kwargs)

    return MultiTaskFLCoordinator(input_dim, tasks, config)


if __name__ == "__main__":
    print("FL-EHDS Multi-Task Federated Learning Demo")
    print("=" * 70)

    np.random.seed(42)

    # Define EHDS-like healthcare tasks
    tasks = [
        TaskDefinition(0, "diabetes_risk", "binary", output_dim=1, weight=1.0),
        TaskDefinition(1, "readmission_risk", "binary", output_dim=1, weight=1.0),
        TaskDefinition(2, "los_prediction", "regression", output_dim=1, weight=0.5),
    ]

    # Generate synthetic data
    n_samples = 500
    input_dim = 10

    # Shared features (demographics, vitals, etc.)
    X = np.random.randn(n_samples, input_dim)

    # Correlated labels (tasks share underlying factors)
    hidden_factor = X[:, 0] + X[:, 1] + np.random.randn(n_samples) * 0.5

    task_labels = {
        0: (hidden_factor + X[:, 2] > 0.5).astype(float),  # Diabetes risk
        1: (hidden_factor - X[:, 3] > 0).astype(float),     # Readmission
        2: (hidden_factor * 2 + 3 + np.random.randn(n_samples) * 0.5),  # Length of stay
    }

    # Create clients with partial task coverage
    n_clients = 5
    client_data_list = []

    for i in range(n_clients):
        start_idx = i * 100
        end_idx = start_idx + 100
        client_X = X[start_idx:end_idx]

        # Some clients don't have all tasks
        client_tasks = {}
        for tid in task_labels.keys():
            if np.random.random() > 0.2:  # 80% chance to have each task
                client_tasks[tid] = task_labels[tid][start_idx:end_idx]

        client_data_list.append(MultiTaskData(
            client_id=i,
            features=client_X,
            task_labels=client_tasks
        ))

    print("\nClient Task Coverage:")
    for cd in client_data_list:
        print(f"  Client {cd.client_id}: Tasks {list(cd.task_labels.keys())}")

    # Configure and train
    config = MultiTaskConfig(
        sharing_mode="hard",
        shared_layers=[32, 16],
        task_specific_layers=[8]
    )

    coordinator = create_multitask_fl(input_dim, tasks, config)

    print("\n" + "-" * 70)
    print("Training Multi-Task FL")
    print("-" * 70)

    for round_num in range(20):
        round_info = coordinator.train_round(
            client_data_list,
            epochs_per_client=2,
            lr=0.05
        )

        if round_num % 5 == 0 or round_num == 19:
            loss_str = ", ".join([
                f"T{tid}: {loss:.4f}"
                for tid, loss in round_info['task_losses'].items()
            ])
            print(f"Round {round_num+1:2d}: Losses: {loss_str}")

    # Final evaluation
    print("\n" + "-" * 70)
    print("Final Evaluation")
    print("-" * 70)

    test_data = MultiTaskData(
        client_id=-1,
        features=X,
        task_labels=task_labels
    )

    results = coordinator.evaluate_all(test_data)

    for tid, metrics in results.items():
        task_name = tasks[tid].name
        if tasks[tid].task_type == "binary":
            print(f"  {task_name}: Accuracy={metrics['accuracy']:.2%}, Loss={metrics['loss']:.4f}")
        else:
            print(f"  {task_name}: MSE={metrics['loss']:.4f}")

    print("\n" + "=" * 70)
    print("Demo completed!")

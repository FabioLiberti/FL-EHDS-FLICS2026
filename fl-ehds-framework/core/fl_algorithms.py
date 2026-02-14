#!/usr/bin/env python3
"""
FL-EHDS Federated Learning Algorithms

Comprehensive collection of FL algorithms:

1. FedAvg - Federated Averaging (McMahan et al., 2017)
2. FedProx - Federated Proximal (Li et al., 2020)
3. SCAFFOLD - Stochastic Controlled Averaging (Karimireddy et al., 2020)
4. FedAdam - Federated Adam Optimizer (Reddi et al., 2021)
5. FedYogi - Federated Yogi Optimizer (Reddi et al., 2021)
6. FedAdagrad - Federated Adagrad (Reddi et al., 2021)
7. MOON - Model-Contrastive Learning (Li et al., 2021)
8. FedNova - Normalized Averaging (Wang et al., 2020)
9. FedDyn - Dynamic Regularization (Acar et al., 2021)
10. Per-FedAvg - Personalized FedAvg (Fallah et al., 2020)
11. pFedMe - Personalized Moreau Envelope (Dinh et al., 2020)
12. Ditto - Fair and Robust FL (Li et al., 2021)

Author: Fabio Liberti
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
import numpy as np
from copy import deepcopy

# Optional PyTorch support
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = nn = F = None


# =============================================================================
# ALGORITHM INFO
# =============================================================================

ALGORITHM_INFO = {
    "FedAvg": {
        "name": "Federated Averaging",
        "paper": "McMahan et al., 2017",
        "description": "Standard federated learning algorithm. Aggregates client updates "
                      "using weighted average based on local dataset sizes.",
        "pros": ["Simple implementation", "Communication efficient", "Works well with IID data"],
        "cons": ["Struggles with non-IID data", "No convergence guarantees for non-convex"],
        "params": {
            "learning_rate": "Local SGD learning rate (0.01-0.1 typical)",
            "local_epochs": "Number of local training epochs (1-10)"
        },
        "best_for": "Baseline experiments, IID data scenarios"
    },

    "FedProx": {
        "name": "Federated Proximal",
        "paper": "Li et al., 2020",
        "description": "Adds proximal term to local objective to prevent client drift. "
                      "Regularizes local updates towards global model.",
        "pros": ["Better for non-IID data", "Handles heterogeneity", "Convergence guarantees"],
        "cons": ["Extra hyperparameter μ to tune", "Slightly slower convergence"],
        "params": {
            "mu": "Proximal coefficient (0.001-1.0). Higher = stronger regularization",
            "learning_rate": "Local learning rate"
        },
        "best_for": "Non-IID data, heterogeneous systems"
    },

    "SCAFFOLD": {
        "name": "Stochastic Controlled Averaging",
        "paper": "Karimireddy et al., 2020",
        "description": "Uses control variates to correct client drift. "
                      "Maintains server and client control variables for variance reduction.",
        "pros": ["Handles extreme non-IID", "Fast convergence", "Variance reduction"],
        "cons": ["2x communication cost", "More complex implementation"],
        "params": {
            "learning_rate": "Global learning rate",
            "local_lr": "Local learning rate (can differ from global)"
        },
        "best_for": "Extreme non-IID, when communication is not bottleneck"
    },

    "FedAdam": {
        "name": "Federated Adam",
        "paper": "Reddi et al., 2021",
        "description": "Applies Adam optimizer at server level for aggregating client updates. "
                      "Combines benefits of adaptive learning rates with federated learning.",
        "pros": ["Adaptive learning rates", "Good for sparse gradients", "Stable training"],
        "cons": ["More server-side computation", "Memory for momentum terms"],
        "params": {
            "server_lr": "Server-side learning rate (typically larger, 0.1-1.0)",
            "beta1": "First moment decay (0.9 typical)",
            "beta2": "Second moment decay (0.99 typical)",
            "tau": "Adaptivity parameter (controls adaptivity strength)"
        },
        "best_for": "Complex models, varying gradient magnitudes"
    },

    "FedYogi": {
        "name": "Federated Yogi",
        "paper": "Reddi et al., 2021",
        "description": "Variant of FedAdam with different second moment update. "
                      "More aggressive adaptation than Adam.",
        "pros": ["Better than FedAdam for some tasks", "Handles non-stationary data"],
        "cons": ["Sensitive to hyperparameters", "Can be unstable"],
        "params": {
            "server_lr": "Server learning rate",
            "beta1": "First moment decay",
            "beta2": "Second moment decay",
            "tau": "Controls adaptivity"
        },
        "best_for": "Non-stationary distributions, concept drift"
    },

    "FedAdagrad": {
        "name": "Federated Adagrad",
        "paper": "Reddi et al., 2021",
        "description": "Server-side Adagrad optimizer for aggregation. "
                      "Accumulates squared gradients for adaptive learning rates.",
        "pros": ["Simple adaptive method", "Good for sparse features"],
        "cons": ["Learning rate decreases over time", "May converge slowly late in training"],
        "params": {
            "server_lr": "Server learning rate",
            "tau": "Initial accumulator value"
        },
        "best_for": "Sparse gradients, early training"
    },

    "MOON": {
        "name": "Model-Contrastive FL",
        "paper": "Li et al., 2021",
        "description": "Uses contrastive learning to reduce model drift. "
                      "Maximizes agreement between current and global model while minimizing with previous.",
        "pros": ["State-of-art for non-IID", "Implicit regularization"],
        "cons": ["Requires model representations", "Higher computation"],
        "params": {
            "mu": "Contrastive loss weight (0.1-1.0)",
            "temperature": "Softmax temperature for contrastive loss"
        },
        "best_for": "Severe non-IID, image classification"
    },

    "FedNova": {
        "name": "Normalized Averaging",
        "paper": "Wang et al., 2020",
        "description": "Normalizes client updates by number of local steps. "
                      "Corrects for objective inconsistency in heterogeneous settings.",
        "pros": ["Handles variable local epochs", "Better convergence"],
        "cons": ["Requires tracking local steps", "Slightly more complex"],
        "params": {
            "normalize_method": "How to normalize (steps or momentum)"
        },
        "best_for": "Heterogeneous computation, variable local epochs"
    },

    "FedDyn": {
        "name": "Dynamic Regularization",
        "paper": "Acar et al., 2021",
        "description": "Dynamically adjusts regularization based on global model. "
                      "Adds linear penalty term to align local and global objectives.",
        "pros": ["Strong convergence guarantees", "Handles partial participation"],
        "cons": ["Extra memory for gradient terms", "Hyperparameter sensitivity"],
        "params": {
            "alpha": "Regularization strength (0.01-0.1)"
        },
        "best_for": "Partial client participation, theoretical guarantees"
    },

    "PerFedAvg": {
        "name": "Personalized FedAvg",
        "paper": "Fallah et al., 2020",
        "description": "MAML-based personalization. Global model serves as initialization "
                      "for quick adaptation to local data.",
        "pros": ["Fast local adaptation", "Good personalization"],
        "cons": ["Requires second-order info", "Computationally expensive"],
        "params": {
            "alpha": "Meta-learning rate",
            "beta": "Inner loop learning rate"
        },
        "best_for": "When clients need personalized models"
    },

    "pFedMe": {
        "name": "Personalized Moreau Envelope",
        "paper": "Dinh et al., 2020",
        "description": "Bi-level optimization for personalization. "
                      "Optimizes personalized models while keeping global model as regularizer.",
        "pros": ["Explicit personalization", "Good theory"],
        "cons": ["Bi-level optimization complexity", "Multiple hyperparameters"],
        "params": {
            "lambda_reg": "Personalization regularization",
            "K": "Number of personalization steps"
        },
        "best_for": "Strong personalization needs"
    },

    "Ditto": {
        "name": "Ditto - Fair FL",
        "paper": "Li et al., 2021",
        "description": "Learns both global and personalized models. "
                      "Balances global performance with local fairness.",
        "pros": ["Fairness guarantees", "Robust to heterogeneity"],
        "cons": ["Two models per client", "Memory overhead"],
        "params": {
            "lambda_ditto": "Regularization towards global model"
        },
        "best_for": "Fairness-critical applications, healthcare"
    }
}


# =============================================================================
# BASE CLASSES
# =============================================================================

@dataclass
class ClientState:
    """State maintained by each client."""
    model_params: Dict[str, np.ndarray]
    control_variate: Optional[Dict[str, np.ndarray]] = None  # For SCAFFOLD
    gradient_accumulator: Optional[Dict[str, np.ndarray]] = None  # For FedDyn
    personalized_params: Optional[Dict[str, np.ndarray]] = None  # For personalization
    num_samples: int = 0
    local_steps: int = 0


@dataclass
class ServerState:
    """State maintained by server."""
    global_params: Dict[str, np.ndarray]
    control_variate: Optional[Dict[str, np.ndarray]] = None  # For SCAFFOLD
    momentum: Optional[Dict[str, np.ndarray]] = None  # For FedAdam/Yogi
    velocity: Optional[Dict[str, np.ndarray]] = None  # For adaptive methods
    round_num: int = 0


class FLAlgorithm(ABC):
    """Abstract base class for FL algorithms."""

    def __init__(self, **kwargs):
        self.config = kwargs
        self.name = self.__class__.__name__

    @abstractmethod
    def client_update(self,
                      client_state: ClientState,
                      server_state: ServerState,
                      local_data: Tuple[np.ndarray, np.ndarray],
                      **kwargs) -> Tuple[Dict[str, np.ndarray], ClientState]:
        """
        Perform local training on client.

        Returns:
            - Update/gradient to send to server
            - Updated client state
        """
        pass

    @abstractmethod
    def server_aggregate(self,
                         server_state: ServerState,
                         client_updates: List[Dict[str, np.ndarray]],
                         client_weights: List[float],
                         **kwargs) -> ServerState:
        """
        Aggregate client updates on server.

        Returns:
            Updated server state
        """
        pass

    @staticmethod
    def get_info(algorithm_name: str) -> Dict:
        """Get information about an algorithm."""
        return ALGORITHM_INFO.get(algorithm_name, {})

    @staticmethod
    def list_algorithms() -> List[str]:
        """List all available algorithms."""
        return list(ALGORITHM_INFO.keys())


# =============================================================================
# ALGORITHM IMPLEMENTATIONS
# =============================================================================

class FedAvg(FLAlgorithm):
    """Federated Averaging algorithm."""

    def __init__(self,
                 learning_rate: float = 0.1,
                 local_epochs: int = 3,
                 **kwargs):
        super().__init__(**kwargs)
        self.lr = learning_rate
        self.local_epochs = local_epochs

    def client_update(self,
                      client_state: ClientState,
                      server_state: ServerState,
                      local_data: Tuple[np.ndarray, np.ndarray],
                      loss_fn: Callable = None,
                      **kwargs) -> Tuple[Dict[str, np.ndarray], ClientState]:
        """Standard local SGD training."""
        X, y = local_data
        params = deepcopy(server_state.global_params)

        # Local SGD
        for _ in range(self.local_epochs):
            params = self._sgd_step(params, X, y, self.lr)

        # Compute update (delta)
        update = {
            k: params[k] - server_state.global_params[k]
            for k in params.keys()
        }

        client_state.model_params = params
        client_state.local_steps = self.local_epochs

        return update, client_state

    def server_aggregate(self,
                         server_state: ServerState,
                         client_updates: List[Dict[str, np.ndarray]],
                         client_weights: List[float],
                         **kwargs) -> ServerState:
        """Weighted average aggregation."""
        total_weight = sum(client_weights)

        aggregated = {}
        for key in server_state.global_params.keys():
            aggregated[key] = sum(
                update[key] * (w / total_weight)
                for update, w in zip(client_updates, client_weights)
            )

        # Update global model
        new_params = {
            k: server_state.global_params[k] + aggregated[k]
            for k in server_state.global_params.keys()
        }

        server_state.global_params = new_params
        server_state.round_num += 1

        return server_state

    def _sgd_step(self, params, X, y, lr):
        """Simple SGD step for logistic regression."""
        # Forward pass
        logits = X @ params['weights'] + params.get('bias', 0)
        probs = 1 / (1 + np.exp(-np.clip(logits, -500, 500)))

        # Gradient
        grad_w = X.T @ (probs - y) / len(y)
        grad_b = np.mean(probs - y) if 'bias' in params else 0

        # Update
        params['weights'] = params['weights'] - lr * grad_w
        if 'bias' in params:
            params['bias'] = params['bias'] - lr * grad_b

        return params


class FedProx(FedAvg):
    """Federated Proximal algorithm."""

    def __init__(self,
                 mu: float = 0.1,
                 **kwargs):
        super().__init__(**kwargs)
        self.mu = mu

    def client_update(self,
                      client_state: ClientState,
                      server_state: ServerState,
                      local_data: Tuple[np.ndarray, np.ndarray],
                      **kwargs) -> Tuple[Dict[str, np.ndarray], ClientState]:
        """Local training with proximal term."""
        X, y = local_data
        params = deepcopy(server_state.global_params)
        global_params = server_state.global_params

        for _ in range(self.local_epochs):
            # SGD with proximal term
            logits = X @ params['weights'] + params.get('bias', 0)
            probs = 1 / (1 + np.exp(-np.clip(logits, -500, 500)))

            # Gradient with proximal term
            grad_w = X.T @ (probs - y) / len(y)
            grad_w += self.mu * (params['weights'] - global_params['weights'])

            params['weights'] = params['weights'] - self.lr * grad_w

            if 'bias' in params:
                grad_b = np.mean(probs - y)
                grad_b += self.mu * (params['bias'] - global_params['bias'])
                params['bias'] = params['bias'] - self.lr * grad_b

        update = {
            k: params[k] - global_params[k]
            for k in params.keys()
        }

        client_state.model_params = params
        return update, client_state


class SCAFFOLD(FLAlgorithm):
    """SCAFFOLD with control variates."""

    def __init__(self,
                 learning_rate: float = 0.1,
                 local_epochs: int = 3,
                 **kwargs):
        super().__init__(**kwargs)
        self.lr = learning_rate
        self.local_epochs = local_epochs

    def client_update(self,
                      client_state: ClientState,
                      server_state: ServerState,
                      local_data: Tuple[np.ndarray, np.ndarray],
                      **kwargs) -> Tuple[Dict[str, np.ndarray], ClientState]:
        """Local training with control variate correction."""
        X, y = local_data
        params = deepcopy(server_state.global_params)

        # Initialize control variates if needed
        if client_state.control_variate is None:
            client_state.control_variate = {
                k: np.zeros_like(v) for k, v in params.items()
            }

        c_i = client_state.control_variate
        c = server_state.control_variate or {k: np.zeros_like(v) for k, v in params.items()}

        for _ in range(self.local_epochs):
            # Compute gradient
            logits = X @ params['weights'] + params.get('bias', 0)
            probs = 1 / (1 + np.exp(-np.clip(logits, -500, 500)))

            grad_w = X.T @ (probs - y) / len(y)

            # SCAFFOLD correction
            params['weights'] = params['weights'] - self.lr * (
                grad_w - c_i['weights'] + c['weights']
            )

        # Update client control variate
        new_c_i = {
            k: c_i[k] - c[k] + (server_state.global_params[k] - params[k]) / (self.local_epochs * self.lr)
            for k in params.keys()
        }

        update = {
            k: params[k] - server_state.global_params[k]
            for k in params.keys()
        }

        # Include control variate update
        update['_control_delta'] = {
            k: new_c_i[k] - c_i[k]
            for k in params.keys()
        }

        client_state.model_params = params
        client_state.control_variate = new_c_i

        return update, client_state

    def server_aggregate(self,
                         server_state: ServerState,
                         client_updates: List[Dict[str, np.ndarray]],
                         client_weights: List[float],
                         **kwargs) -> ServerState:
        """Aggregate with control variate update."""
        total_weight = sum(client_weights)
        num_clients = len(client_updates)

        # Aggregate model updates
        aggregated = {}
        for key in server_state.global_params.keys():
            aggregated[key] = sum(
                update[key] * (w / total_weight)
                for update, w in zip(client_updates, client_weights)
            )

        # Update global model
        server_state.global_params = {
            k: server_state.global_params[k] + aggregated[k]
            for k in server_state.global_params.keys()
        }

        # Update server control variate
        if server_state.control_variate is None:
            server_state.control_variate = {
                k: np.zeros_like(v) for k, v in server_state.global_params.items()
            }

        for update in client_updates:
            if '_control_delta' in update:
                for k in server_state.control_variate.keys():
                    server_state.control_variate[k] += update['_control_delta'][k] / num_clients

        server_state.round_num += 1
        return server_state


class FedAdam(FLAlgorithm):
    """
    Federated Adam optimizer (Reddi et al., 2021).

    Applies Adam optimizer at server level. Includes optional bias correction
    per Kingma & Ba (2015) to improve early-round convergence.
    """

    def __init__(self,
                 client_lr: float = 0.1,
                 server_lr: float = 0.1,
                 beta1: float = 0.9,
                 beta2: float = 0.99,
                 tau: float = 1e-3,
                 local_epochs: int = 3,
                 bias_correction: bool = True,
                 **kwargs):
        super().__init__(**kwargs)
        self.client_lr = client_lr
        self.server_lr = server_lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.tau = tau
        self.local_epochs = local_epochs
        self.bias_correction = bias_correction

    def client_update(self,
                      client_state: ClientState,
                      server_state: ServerState,
                      local_data: Tuple[np.ndarray, np.ndarray],
                      **kwargs) -> Tuple[Dict[str, np.ndarray], ClientState]:
        """Standard local SGD."""
        X, y = local_data
        params = deepcopy(server_state.global_params)

        for _ in range(self.local_epochs):
            logits = X @ params['weights'] + params.get('bias', 0)
            probs = 1 / (1 + np.exp(-np.clip(logits, -500, 500)))

            grad_w = X.T @ (probs - y) / len(y)
            params['weights'] = params['weights'] - self.client_lr * grad_w

        update = {
            k: params[k] - server_state.global_params[k]
            for k in params.keys()
        }

        client_state.model_params = params
        return update, client_state

    def server_aggregate(self,
                         server_state: ServerState,
                         client_updates: List[Dict[str, np.ndarray]],
                         client_weights: List[float],
                         **kwargs) -> ServerState:
        """Adam-style server aggregation with optional bias correction."""
        total_weight = sum(client_weights)

        # Compute pseudo-gradient (weighted average of client updates)
        delta = {}
        for key in server_state.global_params.keys():
            delta[key] = sum(
                update[key] * (w / total_weight)
                for update, w in zip(client_updates, client_weights)
            )

        # Initialize momentum and velocity
        if server_state.momentum is None:
            server_state.momentum = {k: np.zeros_like(v) for k, v in delta.items()}
            server_state.velocity = {k: np.zeros_like(v) + self.tau**2 for k, v in delta.items()}

        # Update momentum (first moment)
        for k in delta.keys():
            server_state.momentum[k] = (
                self.beta1 * server_state.momentum[k] +
                (1 - self.beta1) * delta[k]
            )

        # Update velocity (second moment) - Adam style
        for k in delta.keys():
            server_state.velocity[k] = (
                self.beta2 * server_state.velocity[k] +
                (1 - self.beta2) * delta[k]**2
            )

        # Bias correction (Kingma & Ba, 2015) improves early-round convergence
        t = server_state.round_num + 1
        if self.bias_correction:
            bc1 = 1 - self.beta1 ** t
            bc2 = 1 - self.beta2 ** t
        else:
            bc1 = bc2 = 1.0

        # Update global model
        for k in server_state.global_params.keys():
            m_hat = server_state.momentum[k] / bc1
            v_hat = server_state.velocity[k] / bc2
            server_state.global_params[k] += (
                self.server_lr * m_hat / (np.sqrt(v_hat) + self.tau)
            )

        server_state.round_num += 1
        return server_state


class FedYogi(FedAdam):
    """Federated Yogi optimizer (Reddi et al., 2021) with bias correction."""

    def server_aggregate(self,
                         server_state: ServerState,
                         client_updates: List[Dict[str, np.ndarray]],
                         client_weights: List[float],
                         **kwargs) -> ServerState:
        """Yogi-style server aggregation with optional bias correction."""
        total_weight = sum(client_weights)

        delta = {}
        for key in server_state.global_params.keys():
            delta[key] = sum(
                update[key] * (w / total_weight)
                for update, w in zip(client_updates, client_weights)
            )

        if server_state.momentum is None:
            server_state.momentum = {k: np.zeros_like(v) for k, v in delta.items()}
            server_state.velocity = {k: np.zeros_like(v) + self.tau**2 for k, v in delta.items()}

        for k in delta.keys():
            server_state.momentum[k] = (
                self.beta1 * server_state.momentum[k] +
                (1 - self.beta1) * delta[k]
            )

        # Yogi update for velocity - more controlled adaptation than Adam
        for k in delta.keys():
            sign = np.sign(delta[k]**2 - server_state.velocity[k])
            server_state.velocity[k] = (
                server_state.velocity[k] +
                (1 - self.beta2) * sign * delta[k]**2
            )

        # Bias correction (Kingma & Ba, 2015)
        t = server_state.round_num + 1
        if self.bias_correction:
            bc1 = 1 - self.beta1 ** t
            bc2 = 1 - self.beta2 ** t
        else:
            bc1 = bc2 = 1.0

        for k in server_state.global_params.keys():
            m_hat = server_state.momentum[k] / bc1
            v_hat = server_state.velocity[k] / bc2
            server_state.global_params[k] += (
                self.server_lr * m_hat / (np.sqrt(v_hat) + self.tau)
            )

        server_state.round_num += 1
        return server_state


class FedAdagrad(FLAlgorithm):
    """Federated Adagrad optimizer."""

    def __init__(self,
                 client_lr: float = 0.1,
                 server_lr: float = 0.1,
                 tau: float = 1e-3,
                 local_epochs: int = 3,
                 **kwargs):
        super().__init__(**kwargs)
        self.client_lr = client_lr
        self.server_lr = server_lr
        self.tau = tau
        self.local_epochs = local_epochs

    def client_update(self,
                      client_state: ClientState,
                      server_state: ServerState,
                      local_data: Tuple[np.ndarray, np.ndarray],
                      **kwargs) -> Tuple[Dict[str, np.ndarray], ClientState]:
        """Standard local SGD."""
        X, y = local_data
        params = deepcopy(server_state.global_params)

        for _ in range(self.local_epochs):
            logits = X @ params['weights'] + params.get('bias', 0)
            probs = 1 / (1 + np.exp(-np.clip(logits, -500, 500)))

            grad_w = X.T @ (probs - y) / len(y)
            params['weights'] = params['weights'] - self.client_lr * grad_w

        update = {k: params[k] - server_state.global_params[k] for k in params.keys()}
        client_state.model_params = params
        return update, client_state

    def server_aggregate(self,
                         server_state: ServerState,
                         client_updates: List[Dict[str, np.ndarray]],
                         client_weights: List[float],
                         **kwargs) -> ServerState:
        """Adagrad-style aggregation."""
        total_weight = sum(client_weights)

        delta = {}
        for key in server_state.global_params.keys():
            delta[key] = sum(
                update[key] * (w / total_weight)
                for update, w in zip(client_updates, client_weights)
            )

        if server_state.velocity is None:
            server_state.velocity = {k: np.zeros_like(v) + self.tau**2 for k, v in delta.items()}

        # Adagrad: accumulate squared gradients
        for k in delta.keys():
            server_state.velocity[k] += delta[k]**2

        for k in server_state.global_params.keys():
            server_state.global_params[k] += (
                self.server_lr * delta[k] / (np.sqrt(server_state.velocity[k]) + self.tau)
            )

        server_state.round_num += 1
        return server_state


class FedNova(FLAlgorithm):
    """Federated Nova - Normalized Averaging."""

    def __init__(self,
                 learning_rate: float = 0.1,
                 local_epochs: int = 3,
                 **kwargs):
        super().__init__(**kwargs)
        self.lr = learning_rate
        self.local_epochs = local_epochs

    def client_update(self,
                      client_state: ClientState,
                      server_state: ServerState,
                      local_data: Tuple[np.ndarray, np.ndarray],
                      **kwargs) -> Tuple[Dict[str, np.ndarray], ClientState]:
        """Local SGD with step counting for normalized averaging."""
        X, y = local_data
        params = deepcopy(server_state.global_params)
        accumulated_grad = {k: np.zeros_like(v) for k, v in params.items()}
        total_steps = 0

        for _ in range(self.local_epochs):
            logits = X @ params['weights'] + params.get('bias', 0)
            probs = 1 / (1 + np.exp(-np.clip(logits, -500, 500)))

            grad_w = X.T @ (probs - y) / len(y)
            accumulated_grad['weights'] += grad_w
            params['weights'] -= self.lr * grad_w

            if 'bias' in params:
                grad_b = np.mean(probs - y)
                accumulated_grad['bias'] += grad_b
                params['bias'] -= self.lr * grad_b

            total_steps += 1

        # Normalized update: accumulated gradients + step count for normalization
        update = {k: accumulated_grad[k] for k in params.keys()}
        update['_num_steps'] = total_steps

        client_state.model_params = params
        client_state.local_steps = total_steps
        return update, client_state

    def server_aggregate(self,
                         server_state: ServerState,
                         client_updates: List[Dict[str, np.ndarray]],
                         client_weights: List[float],
                         **kwargs) -> ServerState:
        """Nova-style normalized aggregation."""
        total_weight = sum(client_weights)

        # Compute effective number of steps
        tau_eff = sum(
            update.get('_num_steps', self.local_epochs) * w
            for update, w in zip(client_updates, client_weights)
        ) / total_weight

        # Normalize by steps
        aggregated = {}
        for key in server_state.global_params.keys():
            aggregated[key] = sum(
                update[key] * (w / total_weight) / update.get('_num_steps', self.local_epochs)
                for update, w in zip(client_updates, client_weights)
            ) * tau_eff

        # Update with learning rate
        for k in server_state.global_params.keys():
            server_state.global_params[k] -= self.lr * aggregated[k]

        server_state.round_num += 1
        return server_state


class FedDyn(FLAlgorithm):
    """Federated Dynamic Regularization."""

    def __init__(self,
                 learning_rate: float = 0.1,
                 alpha: float = 0.01,
                 local_epochs: int = 3,
                 **kwargs):
        super().__init__(**kwargs)
        self.lr = learning_rate
        self.alpha = alpha
        self.local_epochs = local_epochs

    def client_update(self,
                      client_state: ClientState,
                      server_state: ServerState,
                      local_data: Tuple[np.ndarray, np.ndarray],
                      **kwargs) -> Tuple[Dict[str, np.ndarray], ClientState]:
        """Local training with dynamic regularization."""
        X, y = local_data
        params = deepcopy(server_state.global_params)

        # Initialize gradient accumulator
        if client_state.gradient_accumulator is None:
            client_state.gradient_accumulator = {
                k: np.zeros_like(v) for k, v in params.items()
            }

        h_i = client_state.gradient_accumulator

        for _ in range(self.local_epochs):
            logits = X @ params['weights'] + params.get('bias', 0)
            probs = 1 / (1 + np.exp(-np.clip(logits, -500, 500)))

            grad_w = X.T @ (probs - y) / len(y)

            # Add dynamic regularization
            grad_w -= h_i['weights']
            grad_w += self.alpha * (params['weights'] - server_state.global_params['weights'])

            params['weights'] -= self.lr * grad_w

        # Update gradient accumulator
        client_state.gradient_accumulator = {
            k: h_i[k] - self.alpha * (params[k] - server_state.global_params[k])
            for k in params.keys()
        }

        update = {k: params[k] - server_state.global_params[k] for k in params.keys()}
        client_state.model_params = params
        return update, client_state

    def server_aggregate(self,
                         server_state: ServerState,
                         client_updates: List[Dict[str, np.ndarray]],
                         client_weights: List[float],
                         **kwargs) -> ServerState:
        """Standard weighted averaging."""
        total_weight = sum(client_weights)

        aggregated = {}
        for key in server_state.global_params.keys():
            aggregated[key] = sum(
                update[key] * (w / total_weight)
                for update, w in zip(client_updates, client_weights)
            )

        for k in server_state.global_params.keys():
            server_state.global_params[k] += aggregated[k]

        server_state.round_num += 1
        return server_state


class Ditto(FLAlgorithm):
    """Ditto - Fair and Robust Federated Learning."""

    def __init__(self,
                 learning_rate: float = 0.1,
                 lambda_ditto: float = 0.1,
                 local_epochs: int = 3,
                 personalization_epochs: int = 1,
                 **kwargs):
        super().__init__(**kwargs)
        self.lr = learning_rate
        self.lambda_ditto = lambda_ditto
        self.local_epochs = local_epochs
        self.personalization_epochs = personalization_epochs

    def client_update(self,
                      client_state: ClientState,
                      server_state: ServerState,
                      local_data: Tuple[np.ndarray, np.ndarray],
                      **kwargs) -> Tuple[Dict[str, np.ndarray], ClientState]:
        """Train both global and personalized models."""
        X, y = local_data

        # Train global model
        params = deepcopy(server_state.global_params)
        for _ in range(self.local_epochs):
            logits = X @ params['weights'] + params.get('bias', 0)
            probs = 1 / (1 + np.exp(-np.clip(logits, -500, 500)))
            grad_w = X.T @ (probs - y) / len(y)
            params['weights'] -= self.lr * grad_w

        global_update = {k: params[k] - server_state.global_params[k] for k in params.keys()}

        # Train personalized model
        if client_state.personalized_params is None:
            client_state.personalized_params = deepcopy(server_state.global_params)

        personal = client_state.personalized_params
        for _ in range(self.personalization_epochs):
            logits = X @ personal['weights'] + personal.get('bias', 0)
            probs = 1 / (1 + np.exp(-np.clip(logits, -500, 500)))
            grad_w = X.T @ (probs - y) / len(y)

            # Regularization towards global
            grad_w += self.lambda_ditto * (personal['weights'] - server_state.global_params['weights'])
            personal['weights'] -= self.lr * grad_w

        client_state.model_params = params
        client_state.personalized_params = personal

        return global_update, client_state

    def server_aggregate(self,
                         server_state: ServerState,
                         client_updates: List[Dict[str, np.ndarray]],
                         client_weights: List[float],
                         **kwargs) -> ServerState:
        """Standard FedAvg aggregation for global model."""
        total_weight = sum(client_weights)

        aggregated = {}
        for key in server_state.global_params.keys():
            aggregated[key] = sum(
                update[key] * (w / total_weight)
                for update, w in zip(client_updates, client_weights)
            )

        for k in server_state.global_params.keys():
            server_state.global_params[k] += aggregated[k]

        server_state.round_num += 1
        return server_state


# =============================================================================
# ALGORITHM FACTORY
# =============================================================================

def create_algorithm(name: str, **kwargs) -> FLAlgorithm:
    """Factory function to create FL algorithm instances."""
    algorithms = {
        'FedAvg': FedAvg,
        'FedProx': FedProx,
        'SCAFFOLD': SCAFFOLD,
        'FedAdam': FedAdam,
        'FedYogi': FedYogi,
        'FedAdagrad': FedAdagrad,
        'FedNova': FedNova,
        'FedDyn': FedDyn,
        'Ditto': Ditto,
    }

    if name not in algorithms:
        raise ValueError(f"Unknown algorithm: {name}. Available: {list(algorithms.keys())}")

    return algorithms[name](**kwargs)


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    print("FL-EHDS Algorithm Library")
    print("=" * 60)

    print("\nAvailable Algorithms:")
    for name, info in ALGORITHM_INFO.items():
        print(f"\n{name} - {info['name']}")
        print(f"  Paper: {info['paper']}")
        print(f"  Best for: {info['best_for']}")

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
13. FedLC - Logits Calibration (Zhang et al., ICML 2022)
14. FedSAM - Sharpness-Aware FL (Qu et al., ICML 2022)
15. FedDecorr - Federated Decorrelation (Shi et al., ICLR 2023)
16. FedSpeed - Prox-correction + SAM (Sun et al., ICLR 2023)
17. FedExP - Federated Extrapolation (Jhunjhunwala et al., ICLR 2023)

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
    },

    "FedLC": {
        "name": "Logits Calibration FL",
        "paper": "Zhang et al., ICML 2022",
        "description": "Calibrates logits before softmax by adding pairwise label margins "
                      "proportional to class frequency, addressing label distribution skew.",
        "pros": ["Handles label skew", "Composable with any algorithm", "Simple loss modification"],
        "cons": ["Extra hyperparameter tau", "Needs class count per client"],
        "params": {
            "tau": "Margin strength (0.1-1.0). Controls class-frequency calibration."
        },
        "best_for": "Label-imbalanced data, hospitals with different disease prevalences"
    },

    "FedSAM": {
        "name": "Sharpness-Aware FL",
        "paper": "Qu et al., ICML 2022",
        "description": "Applies Sharpness-Aware Minimization locally to seek flat minima, "
                      "improving generalization of the global model under non-IID data.",
        "pros": ["Better generalization", "Drop-in local optimizer", "No server changes"],
        "cons": ["2x gradient computation per step", "Extra hyperparameter rho"],
        "params": {
            "rho": "Perturbation radius (0.01-0.1). Controls neighborhood size for flat minima."
        },
        "best_for": "Non-IID data, improving generalization across diverse hospitals"
    },

    "FedDecorr": {
        "name": "Federated Decorrelation",
        "paper": "Shi et al., ICLR 2023",
        "description": "Adds decorrelation regularizer to prevent dimensional collapse "
                      "in learned representations under heterogeneous data.",
        "pros": ["Improves representation quality", "Composable", "Negligible overhead"],
        "cons": ["Requires access to intermediate representations", "Extra hyperparameter beta"],
        "params": {
            "beta": "Decorrelation strength (0.01-1.0). Controls regularization."
        },
        "best_for": "Deep models with non-IID data, preventing representation collapse"
    },

    "FedSpeed": {
        "name": "FedSpeed",
        "paper": "Sun et al., ICLR 2023",
        "description": "Combines prox-correction (eliminates proximal bias) with gradient "
                      "perturbation (seeks flat minima) for fewer rounds and better accuracy.",
        "pros": ["Fewer communication rounds", "Handles prox bias", "Better generalization"],
        "cons": ["More complex local update", "Multiple hyperparameters"],
        "params": {
            "alpha_mix": "Mixing coefficient for quasi-gradient (0.1-0.9)",
            "rho": "Perturbation radius for flat minima",
            "lambda_reg": "Proximal regularization strength"
        },
        "best_for": "Communication-constrained settings, non-IID healthcare data"
    },

    "FedExP": {
        "name": "Federated Extrapolation",
        "paper": "Jhunjhunwala et al., ICLR 2023",
        "description": "Adaptively computes server step size using extrapolation inspired by "
                      "Projection Onto Convex Sets (POCS) for accelerated convergence.",
        "pros": ["No client-side changes", "Hyperparameter-free", "Accelerates convergence"],
        "cons": ["Minor server computation overhead", "May overshoot with very few clients"],
        "params": {},
        "best_for": "Accelerating convergence, composable with any client-side algorithm"
    },
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


class FedLC(FedAvg):
    """Federated Learning via Logits Calibration (Zhang et al., ICML 2022)."""

    def __init__(self, tau: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.tau = tau

    def client_update(self, client_state, server_state, local_data, **kwargs):
        """Local training with calibrated cross-entropy loss."""
        X, y = local_data
        params = deepcopy(server_state.global_params)

        # Count class frequencies for calibration
        unique, counts = np.unique(y.astype(int), return_counts=True)
        n_classes = max(unique) + 1
        class_counts = np.ones(n_classes)
        for cls, cnt in zip(unique, counts):
            class_counts[cls] = cnt

        # Calibration margins: tau * n_c^{-1/4}
        margins = self.tau * np.power(class_counts, -0.25)

        for _ in range(self.local_epochs):
            logits = X @ params['weights'] + params.get('bias', 0)
            # Apply calibration: subtract margin for each class
            calibrated_logits = logits - margins.reshape(1, -1) if logits.ndim > 1 else logits - margins
            # For binary: calibrated sigmoid
            if logits.ndim == 1 or (logits.ndim == 2 and logits.shape[1] == 1):
                probs = 1 / (1 + np.exp(-np.clip(logits, -500, 500)))
                grad_w = X.T @ (probs - y) / len(y)
            else:
                # Multi-class softmax with calibration
                probs = np.exp(calibrated_logits - calibrated_logits.max(axis=1, keepdims=True))
                probs /= probs.sum(axis=1, keepdims=True)
                grad_w = X.T @ (probs - np.eye(n_classes)[y.astype(int)]) / len(y)

            params['weights'] = params['weights'] - self.lr * grad_w

        update = {k: params[k] - server_state.global_params[k] for k in params.keys()}
        client_state.model_params = params
        client_state.local_steps = self.local_epochs
        return update, client_state


class FedSAM(FedAvg):
    """Federated Sharpness-Aware Minimization (Qu et al., ICML 2022)."""

    def __init__(self, rho: float = 0.05, **kwargs):
        super().__init__(**kwargs)
        self.rho = rho

    def _compute_gradient(self, params, X, y):
        """Compute gradient for logistic regression."""
        logits = X @ params['weights'] + params.get('bias', 0)
        probs = 1 / (1 + np.exp(-np.clip(logits, -500, 500)))
        grad_w = X.T @ (probs - y) / len(y)
        grad_b = np.mean(probs - y) if 'bias' in params else 0
        return {'weights': grad_w, 'bias': grad_b} if 'bias' in params else {'weights': grad_w}

    def client_update(self, client_state, server_state, local_data, **kwargs):
        """Local training with SAM: gradient evaluated at perturbed point."""
        X, y = local_data
        params = deepcopy(server_state.global_params)

        for _ in range(self.local_epochs):
            # Step 1: compute gradient at current position
            grad = self._compute_gradient(params, X, y)

            # Step 2: compute perturbation epsilon = rho * grad / ||grad||
            grad_norm = np.sqrt(sum(np.sum(g**2) for g in grad.values()))
            if grad_norm > 1e-12:
                epsilon = {k: self.rho * g / grad_norm for k, g in grad.items()}
            else:
                epsilon = {k: np.zeros_like(g) for k, g in grad.items()}

            # Step 3: compute gradient at perturbed position
            perturbed = {k: params[k] + epsilon[k] for k in params.keys() if k in epsilon}
            for k in params.keys():
                if k not in perturbed:
                    perturbed[k] = params[k]
            grad_perturbed = self._compute_gradient(perturbed, X, y)

            # Step 4: update using perturbed gradient
            for k in grad_perturbed:
                params[k] = params[k] - self.lr * grad_perturbed[k]

        update = {k: params[k] - server_state.global_params[k] for k in params.keys()}
        client_state.model_params = params
        client_state.local_steps = self.local_epochs
        return update, client_state


class FedDecorr(FedAvg):
    """Federated Decorrelation (Shi et al., ICLR 2023)."""

    def __init__(self, beta: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.beta = beta

    def client_update(self, client_state, server_state, local_data, **kwargs):
        """Local training with decorrelation regularizer."""
        X, y = local_data
        params = deepcopy(server_state.global_params)

        for _ in range(self.local_epochs):
            # Standard gradient
            logits = X @ params['weights'] + params.get('bias', 0)
            probs = 1 / (1 + np.exp(-np.clip(logits, -500, 500)))
            grad_w = X.T @ (probs - y) / len(y)

            # Decorrelation regularizer: minimize ||K||_F^2 / d^2
            # K is correlation matrix of representations (here, logits/features)
            # For linear model, representations = X @ weights
            representations = X @ params['weights']
            if representations.ndim == 1:
                representations = representations.reshape(-1, 1)
            d = representations.shape[1] if representations.ndim > 1 else 1
            if d > 1:
                # Center representations
                reps_centered = representations - representations.mean(axis=0)
                # Correlation matrix
                K = (reps_centered.T @ reps_centered) / len(X)
                # Set diagonal to zero (we only want off-diagonal)
                np.fill_diagonal(K, 0)
                # Gradient of ||K||_F^2 w.r.t. weights
                decorr_grad = (2.0 / (d * d)) * X.T @ (reps_centered @ K) / len(X)
                grad_w += self.beta * decorr_grad

            params['weights'] = params['weights'] - self.lr * grad_w

        update = {k: params[k] - server_state.global_params[k] for k in params.keys()}
        client_state.model_params = params
        client_state.local_steps = self.local_epochs
        return update, client_state


class FedSpeed(FLAlgorithm):
    """FedSpeed (Sun et al., ICLR 2023)."""

    def __init__(self, learning_rate=0.1, local_epochs=3, alpha_mix=0.5,
                 rho=0.05, lambda_reg=0.1, **kwargs):
        super().__init__(**kwargs)
        self.lr = learning_rate
        self.local_epochs = local_epochs
        self.alpha_mix = alpha_mix
        self.rho = rho
        self.lambda_reg = lambda_reg

    def _compute_gradient(self, params, X, y):
        logits = X @ params['weights'] + params.get('bias', 0)
        probs = 1 / (1 + np.exp(-np.clip(logits, -500, 500)))
        grad_w = X.T @ (probs - y) / len(y)
        return {'weights': grad_w}

    def client_update(self, client_state, server_state, local_data, **kwargs):
        """FedSpeed local update: prox-correction + gradient perturbation."""
        X, y = local_data
        params = deepcopy(server_state.global_params)
        global_params = server_state.global_params

        # Initialize prox-correction term
        if client_state.gradient_accumulator is None:
            client_state.gradient_accumulator = {
                k: np.zeros_like(v) for k, v in params.items()
            }
        g_hat = client_state.gradient_accumulator

        for _ in range(self.local_epochs):
            # Standard gradient g1
            g1 = self._compute_gradient(params, X, y)

            # Perturbation: compute g2 at perturbed point
            g1_norm = np.sqrt(sum(np.sum(g**2) for g in g1.values()))
            if g1_norm > 1e-12:
                perturbed = {k: params[k] + self.rho * g1[k] / g1_norm
                            for k in params.keys() if k in g1}
                for k in params.keys():
                    if k not in perturbed:
                        perturbed[k] = params[k]
            else:
                perturbed = params
            g2 = self._compute_gradient(perturbed, X, y)

            # Quasi-gradient: mix g1 and g2
            g_tilde = {}
            for k in g1:
                g_tilde[k] = (1 - self.alpha_mix) * g1[k] + self.alpha_mix * g2[k]

            # FedSpeed update: g_tilde - g_hat + prox_term
            for k in params.keys():
                if k in g_tilde:
                    prox = (1.0 / self.lambda_reg) * (params[k] - global_params[k])
                    correction = g_hat.get(k, np.zeros_like(params[k]))
                    params[k] -= self.lr * (g_tilde[k] - correction + prox)

        # Update prox-correction
        for k in params.keys():
            if k in client_state.gradient_accumulator:
                client_state.gradient_accumulator[k] -= (
                    (1.0 / self.lambda_reg) * (params[k] - global_params[k])
                )

        update = {k: params[k] - global_params[k] for k in params.keys()}
        client_state.model_params = params
        client_state.local_steps = self.local_epochs
        return update, client_state

    def server_aggregate(self, server_state, client_updates, client_weights, **kwargs):
        """Standard weighted average aggregation."""
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


class FedExP(FLAlgorithm):
    """Federated Extrapolation (Jhunjhunwala et al., ICLR 2023)."""

    def __init__(self, learning_rate=0.1, local_epochs=3, **kwargs):
        super().__init__(**kwargs)
        self.lr = learning_rate
        self.local_epochs = local_epochs

    def client_update(self, client_state, server_state, local_data, **kwargs):
        """Standard local SGD (no client-side changes)."""
        X, y = local_data
        params = deepcopy(server_state.global_params)

        for _ in range(self.local_epochs):
            logits = X @ params['weights'] + params.get('bias', 0)
            probs = 1 / (1 + np.exp(-np.clip(logits, -500, 500)))
            grad_w = X.T @ (probs - y) / len(y)
            params['weights'] -= self.lr * grad_w
            if 'bias' in params:
                grad_b = np.mean(probs - y)
                params['bias'] -= self.lr * grad_b

        update = {k: params[k] - server_state.global_params[k] for k in params.keys()}
        client_state.model_params = params
        client_state.local_steps = self.local_epochs
        return update, client_state

    def server_aggregate(self, server_state, client_updates, client_weights, **kwargs):
        """Extrapolation-based server aggregation with adaptive step size."""
        total_weight = sum(client_weights)
        N = len(client_updates)

        # Compute weighted average of updates (pseudo-gradient)
        delta_avg = {}
        for key in server_state.global_params.keys():
            delta_avg[key] = sum(
                update[key] * (w / total_weight)
                for update, w in zip(client_updates, client_weights)
            )

        # Compute adaptive step size: eta = max(1, ||sum delta_i||^2 / (N * sum ||delta_i||^2))
        sum_norm_sq = 0.0
        for update in client_updates:
            for key in server_state.global_params.keys():
                sum_norm_sq += np.sum(update[key] ** 2)

        avg_norm_sq = 0.0
        for key in delta_avg.keys():
            avg_norm_sq += np.sum(delta_avg[key] ** 2)

        # eta_s = max(1, N * ||avg||^2 / sum_||delta_i||^2)
        if sum_norm_sq > 1e-12:
            eta_s = max(1.0, N * avg_norm_sq / sum_norm_sq)
        else:
            eta_s = 1.0

        # Update global model with extrapolated step
        for k in server_state.global_params.keys():
            server_state.global_params[k] += eta_s * delta_avg[k]

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
        'FedLC': FedLC,
        'FedSAM': FedSAM,
        'FedDecorr': FedDecorr,
        'FedSpeed': FedSpeed,
        'FedExP': FedExP,
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

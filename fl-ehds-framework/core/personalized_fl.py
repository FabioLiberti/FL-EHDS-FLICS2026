#!/usr/bin/env python3
"""
FL-EHDS Personalized Federated Learning

Implements personalized FL algorithms that learn both a global model
and client-specific adaptations, crucial for healthcare where patient
populations vary significantly across hospitals.

Algorithms:
1. Ditto (Li et al., 2021) - Global + regularized local models
2. Per-FedAvg (Fallah et al., 2020) - MAML-based personalization
3. FedPer (Arivazhagan et al., 2019) - Partial model personalization
4. APFL (Deng et al., 2020) - Adaptive mixing of global/local
5. FedRep (Collins et al., 2021) - Representation learning

Author: Fabio Liberti
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from copy import deepcopy
import warnings

# PyTorch imports
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. Some features will be limited.")


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class PersonalizedModel:
    """Container for personalized model state."""
    client_id: int
    global_params: Dict[str, np.ndarray]
    local_params: Dict[str, np.ndarray]
    mixing_weight: float = 0.5  # For APFL
    personalization_layers: List[str] = field(default_factory=list)  # For FedPer


@dataclass
class PersonalizationConfig:
    """Configuration for personalized FL."""
    algorithm: str = "ditto"
    lambda_reg: float = 0.1  # Ditto regularization
    alpha: float = 0.5  # APFL mixing weight
    meta_lr: float = 0.01  # Per-FedAvg meta learning rate
    personalization_epochs: int = 5
    freeze_layers: List[str] = field(default_factory=list)  # FedPer


# =============================================================================
# BASE CLASS
# =============================================================================

class PersonalizedFLAlgorithm(ABC):
    """Abstract base class for personalized FL algorithms."""

    def __init__(self, config: PersonalizationConfig):
        self.config = config
        self.global_model = None
        self.local_models: Dict[int, Any] = {}

    @abstractmethod
    def initialize(self, model_template: Any) -> None:
        """Initialize global and local models."""
        pass

    @abstractmethod
    def local_train(self,
                   client_id: int,
                   data: Tuple[np.ndarray, np.ndarray],
                   epochs: int = 1) -> Dict[str, np.ndarray]:
        """Perform local training and return updates."""
        pass

    @abstractmethod
    def aggregate(self,
                 client_updates: Dict[int, Dict[str, np.ndarray]],
                 weights: Optional[Dict[int, float]] = None) -> None:
        """Aggregate updates into global model."""
        pass

    @abstractmethod
    def personalize(self,
                   client_id: int,
                   data: Tuple[np.ndarray, np.ndarray]) -> Any:
        """Return personalized model for client."""
        pass

    def get_global_model(self) -> Any:
        """Return current global model."""
        return self.global_model

    def get_local_model(self, client_id: int) -> Any:
        """Return local model for specific client."""
        return self.local_models.get(client_id)


# =============================================================================
# DITTO (Li et al., 2021)
# =============================================================================

class Ditto(PersonalizedFLAlgorithm):
    """
    Ditto: Fair and Robust Federated Learning Through Personalization.

    Key idea: Train global model normally, then fine-tune local models
    with L2 regularization toward the global model.

    Local objective: min_v L(v; D_k) + (λ/2)||v - w||²

    where w is global model, v is local model, λ controls regularization.

    Benefits:
    - Robustness to data heterogeneity
    - Fair performance across clients
    - Simple implementation
    """

    def __init__(self, config: PersonalizationConfig):
        super().__init__(config)
        self.lambda_reg = config.lambda_reg

    def initialize(self, model_template: Any) -> None:
        """Initialize with model template (numpy weights dict or PyTorch model)."""
        if isinstance(model_template, dict):
            self.global_model = {k: v.copy() for k, v in model_template.items()}
        elif TORCH_AVAILABLE and isinstance(model_template, nn.Module):
            self.global_model = deepcopy(model_template)
        else:
            raise ValueError("Model template must be dict or nn.Module")

    def local_train(self,
                   client_id: int,
                   data: Tuple[np.ndarray, np.ndarray],
                   epochs: int = 1,
                   lr: float = 0.01) -> Dict[str, np.ndarray]:
        """
        Train local model with Ditto objective.

        Phase 1: Standard FedAvg update (for global aggregation)
        Phase 2: Personalized update with regularization (for local use)
        """
        X, y = data

        # Initialize local model from global if not exists
        if client_id not in self.local_models:
            if isinstance(self.global_model, dict):
                self.local_models[client_id] = {
                    k: v.copy() for k, v in self.global_model.items()
                }
            else:
                self.local_models[client_id] = deepcopy(self.global_model)

        # Use numpy implementation for dict models
        if isinstance(self.global_model, dict):
            return self._train_numpy(client_id, X, y, epochs, lr)
        else:
            return self._train_torch(client_id, X, y, epochs, lr)

    def _train_numpy(self,
                    client_id: int,
                    X: np.ndarray,
                    y: np.ndarray,
                    epochs: int,
                    lr: float) -> Dict[str, np.ndarray]:
        """Numpy implementation for simple models."""
        # Phase 1: Standard update for global aggregation
        w = {k: v.copy() for k, v in self.global_model.items()}

        for _ in range(epochs):
            # Mini-batch gradient descent
            batch_size = min(32, len(X))
            indices = np.random.choice(len(X), batch_size, replace=False)
            X_batch, y_batch = X[indices], y[indices]

            # Forward pass (logistic regression assumed)
            if 'weights' in w:
                logits = X_batch @ w['weights']
                if 'bias' in w:
                    logits += w['bias']
                probs = 1 / (1 + np.exp(-np.clip(logits, -500, 500)))

                # Gradient
                grad_w = X_batch.T @ (probs - y_batch) / batch_size
                w['weights'] -= lr * grad_w

                if 'bias' in w:
                    grad_b = np.mean(probs - y_batch)
                    w['bias'] -= lr * grad_b

        # Compute update delta for aggregation
        delta = {k: w[k] - self.global_model[k] for k in w.keys()}

        # Phase 2: Personalized local update with regularization
        v = self.local_models[client_id]

        for _ in range(self.config.personalization_epochs):
            batch_size = min(32, len(X))
            indices = np.random.choice(len(X), batch_size, replace=False)
            X_batch, y_batch = X[indices], y[indices]

            if 'weights' in v:
                logits = X_batch @ v['weights']
                if 'bias' in v:
                    logits += v['bias']
                probs = 1 / (1 + np.exp(-np.clip(logits, -500, 500)))

                # Gradient with Ditto regularization
                grad_w = X_batch.T @ (probs - y_batch) / batch_size
                grad_w += self.lambda_reg * (v['weights'] - self.global_model['weights'])
                v['weights'] -= lr * grad_w

                if 'bias' in v:
                    grad_b = np.mean(probs - y_batch)
                    grad_b += self.lambda_reg * (v['bias'] - self.global_model['bias'])
                    v['bias'] -= lr * grad_b

        self.local_models[client_id] = v
        return delta

    def _train_torch(self,
                    client_id: int,
                    X: np.ndarray,
                    y: np.ndarray,
                    epochs: int,
                    lr: float) -> Dict[str, np.ndarray]:
        """PyTorch implementation for neural networks."""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for neural network training")

        device = next(self.global_model.parameters()).device
        X_tensor = torch.FloatTensor(X).to(device)
        y_tensor = torch.FloatTensor(y).to(device)

        # Phase 1: Standard FedAvg update
        model = deepcopy(self.global_model)
        optimizer = optim.SGD(model.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss()

        model.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            outputs = model(X_tensor).squeeze()
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()

        # Compute delta
        delta = {}
        for (name, param), (_, global_param) in zip(
            model.named_parameters(),
            self.global_model.named_parameters()
        ):
            delta[name] = (param.data - global_param.data).cpu().numpy()

        # Phase 2: Personalized update
        local_model = self.local_models[client_id]
        local_optimizer = optim.SGD(local_model.parameters(), lr=lr)

        local_model.train()
        for _ in range(self.config.personalization_epochs):
            local_optimizer.zero_grad()
            outputs = local_model(X_tensor).squeeze()
            loss = criterion(outputs, y_tensor)

            # Add Ditto regularization
            reg_loss = 0
            for (name, param), (_, global_param) in zip(
                local_model.named_parameters(),
                self.global_model.named_parameters()
            ):
                reg_loss += torch.norm(param - global_param.detach()) ** 2

            total_loss = loss + (self.lambda_reg / 2) * reg_loss
            total_loss.backward()
            local_optimizer.step()

        return delta

    def aggregate(self,
                 client_updates: Dict[int, Dict[str, np.ndarray]],
                 weights: Optional[Dict[int, float]] = None) -> None:
        """Aggregate client updates into global model."""
        if weights is None:
            weights = {cid: 1.0 / len(client_updates) for cid in client_updates}

        total_weight = sum(weights.values())

        if isinstance(self.global_model, dict):
            for key in self.global_model.keys():
                update = sum(
                    weights[cid] * client_updates[cid][key]
                    for cid in client_updates
                ) / total_weight
                self.global_model[key] += update
        else:
            # PyTorch model
            with torch.no_grad():
                for name, param in self.global_model.named_parameters():
                    update = sum(
                        weights[cid] * torch.from_numpy(client_updates[cid][name])
                        for cid in client_updates
                    ) / total_weight
                    param.add_(update.to(param.device))

    def personalize(self,
                   client_id: int,
                   data: Tuple[np.ndarray, np.ndarray]) -> Any:
        """Return personalized model for client."""
        if client_id not in self.local_models:
            # Initialize and train if not exists
            if isinstance(self.global_model, dict):
                self.local_models[client_id] = {
                    k: v.copy() for k, v in self.global_model.items()
                }
            else:
                self.local_models[client_id] = deepcopy(self.global_model)
            self.local_train(client_id, data, epochs=self.config.personalization_epochs)

        return self.local_models[client_id]


# =============================================================================
# Per-FedAvg (MAML-based)
# =============================================================================

class PerFedAvg(PersonalizedFLAlgorithm):
    """
    Personalized Federated Learning using MAML.

    Key idea: Learn a global model that is easy to adapt to each client
    using Model-Agnostic Meta-Learning (MAML).

    Meta-objective: min_w Σ_k L(w - α∇L(w; D_k^train); D_k^test)

    The global model serves as a good initialization for fast adaptation.
    """

    def __init__(self, config: PersonalizationConfig):
        super().__init__(config)
        self.meta_lr = config.meta_lr
        self.inner_lr = 0.01
        self.inner_steps = 5

    def initialize(self, model_template: Any) -> None:
        """Initialize meta-model."""
        if isinstance(model_template, dict):
            self.global_model = {k: v.copy() for k, v in model_template.items()}
        elif TORCH_AVAILABLE and isinstance(model_template, nn.Module):
            self.global_model = deepcopy(model_template)
        else:
            raise ValueError("Model template must be dict or nn.Module")

    def local_train(self,
                   client_id: int,
                   data: Tuple[np.ndarray, np.ndarray],
                   epochs: int = 1,
                   lr: float = 0.01) -> Dict[str, np.ndarray]:
        """
        MAML-style local training.

        1. Inner loop: Adapt to local data
        2. Outer loop: Compute meta-gradient
        """
        X, y = data
        n = len(X)

        # Split into support and query sets
        split = n // 2
        X_support, y_support = X[:split], y[:split]
        X_query, y_query = X[split:], y[split:]

        if isinstance(self.global_model, dict):
            return self._train_numpy_maml(
                X_support, y_support, X_query, y_query, lr
            )
        else:
            return self._train_torch_maml(
                X_support, y_support, X_query, y_query, lr
            )

    def _train_numpy_maml(self,
                         X_support, y_support,
                         X_query, y_query,
                         lr: float) -> Dict[str, np.ndarray]:
        """Numpy MAML implementation."""
        # Inner loop: adapt to support set
        adapted = {k: v.copy() for k, v in self.global_model.items()}

        for _ in range(self.inner_steps):
            if 'weights' in adapted:
                logits = X_support @ adapted['weights']
                if 'bias' in adapted:
                    logits += adapted['bias']
                probs = 1 / (1 + np.exp(-np.clip(logits, -500, 500)))

                grad_w = X_support.T @ (probs - y_support) / len(X_support)
                adapted['weights'] -= self.inner_lr * grad_w

                if 'bias' in adapted:
                    grad_b = np.mean(probs - y_support)
                    adapted['bias'] -= self.inner_lr * grad_b

        # Outer loop: compute meta-gradient on query set
        if 'weights' in adapted:
            logits = X_query @ adapted['weights']
            if 'bias' in adapted:
                logits += adapted['bias']
            probs = 1 / (1 + np.exp(-np.clip(logits, -500, 500)))

            meta_grad_w = X_query.T @ (probs - y_query) / len(X_query)

            delta = {'weights': -self.meta_lr * meta_grad_w}
            if 'bias' in adapted:
                meta_grad_b = np.mean(probs - y_query)
                delta['bias'] = -self.meta_lr * meta_grad_b

            return delta

        return {}

    def _train_torch_maml(self,
                         X_support, y_support,
                         X_query, y_query,
                         lr: float) -> Dict[str, np.ndarray]:
        """PyTorch MAML with higher-order gradients."""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required")

        device = next(self.global_model.parameters()).device
        X_s = torch.FloatTensor(X_support).to(device)
        y_s = torch.FloatTensor(y_support).to(device)
        X_q = torch.FloatTensor(X_query).to(device)
        y_q = torch.FloatTensor(y_query).to(device)

        # Create functional version for MAML
        params = {name: param.clone() for name, param in self.global_model.named_parameters()}

        # Inner loop
        criterion = nn.BCEWithLogitsLoss()

        for _ in range(self.inner_steps):
            # Forward with current params
            model_copy = deepcopy(self.global_model)
            for name, param in model_copy.named_parameters():
                param.data = params[name]

            outputs = model_copy(X_s).squeeze()
            loss = criterion(outputs, y_s)

            grads = torch.autograd.grad(loss, params.values(), create_graph=True)
            params = {
                name: param - self.inner_lr * grad
                for (name, param), grad in zip(params.items(), grads)
            }

        # Outer loop on query set
        model_copy = deepcopy(self.global_model)
        for name, param in model_copy.named_parameters():
            param.data = params[name]

        outputs = model_copy(X_q).squeeze()
        meta_loss = criterion(outputs, y_q)

        # Meta-gradients
        meta_grads = torch.autograd.grad(
            meta_loss, self.global_model.parameters()
        )

        delta = {}
        for (name, _), grad in zip(self.global_model.named_parameters(), meta_grads):
            delta[name] = -self.meta_lr * grad.cpu().numpy()

        return delta

    def aggregate(self,
                 client_updates: Dict[int, Dict[str, np.ndarray]],
                 weights: Optional[Dict[int, float]] = None) -> None:
        """Aggregate meta-gradients."""
        if weights is None:
            weights = {cid: 1.0 / len(client_updates) for cid in client_updates}

        total_weight = sum(weights.values())

        if isinstance(self.global_model, dict):
            for key in self.global_model.keys():
                update = sum(
                    weights[cid] * client_updates[cid].get(key, 0)
                    for cid in client_updates
                ) / total_weight
                self.global_model[key] += update
        else:
            with torch.no_grad():
                for name, param in self.global_model.named_parameters():
                    update = sum(
                        weights[cid] * torch.from_numpy(client_updates[cid][name])
                        for cid in client_updates
                    ) / total_weight
                    param.add_(update.to(param.device))

    def personalize(self,
                   client_id: int,
                   data: Tuple[np.ndarray, np.ndarray],
                   steps: int = 5) -> Any:
        """Adapt global model to client with few gradient steps."""
        X, y = data

        if isinstance(self.global_model, dict):
            adapted = {k: v.copy() for k, v in self.global_model.items()}

            for _ in range(steps):
                if 'weights' in adapted:
                    logits = X @ adapted['weights']
                    if 'bias' in adapted:
                        logits += adapted['bias']
                    probs = 1 / (1 + np.exp(-np.clip(logits, -500, 500)))

                    grad_w = X.T @ (probs - y) / len(X)
                    adapted['weights'] -= self.inner_lr * grad_w

                    if 'bias' in adapted:
                        grad_b = np.mean(probs - y)
                        adapted['bias'] -= self.inner_lr * grad_b

            return adapted
        else:
            model = deepcopy(self.global_model)
            device = next(model.parameters()).device
            X_tensor = torch.FloatTensor(X).to(device)
            y_tensor = torch.FloatTensor(y).to(device)

            optimizer = optim.SGD(model.parameters(), lr=self.inner_lr)
            criterion = nn.BCEWithLogitsLoss()

            model.train()
            for _ in range(steps):
                optimizer.zero_grad()
                outputs = model(X_tensor).squeeze()
                loss = criterion(outputs, y_tensor)
                loss.backward()
                optimizer.step()

            return model


# =============================================================================
# FedPer (Partial Personalization)
# =============================================================================

class FedPer(PersonalizedFLAlgorithm):
    """
    Federated Learning with Personalization Layers.

    Key idea: Share base layers globally, keep head layers local.

    Model = [Base layers (global)] + [Head layers (local)]

    Only base layers are aggregated; head layers remain personalized.
    Useful for representation learning where lower layers capture
    general features and upper layers capture task-specific patterns.
    """

    def __init__(self, config: PersonalizationConfig):
        super().__init__(config)
        self.personalization_layers = config.freeze_layers

    def initialize(self, model_template: Any,
                   personalization_layers: Optional[List[str]] = None) -> None:
        """
        Initialize model with specified personalization layers.

        Args:
            model_template: Model template
            personalization_layers: Names of layers to keep local
        """
        if personalization_layers:
            self.personalization_layers = personalization_layers

        if isinstance(model_template, dict):
            self.global_model = {k: v.copy() for k, v in model_template.items()}
        elif TORCH_AVAILABLE and isinstance(model_template, nn.Module):
            self.global_model = deepcopy(model_template)

            # Auto-detect personalization layers if not specified
            if not self.personalization_layers:
                # Default: last linear layer is personalized
                layer_names = [name for name, _ in model_template.named_parameters()]
                if layer_names:
                    # Find layers containing 'head', 'classifier', 'fc' (common patterns)
                    for name in layer_names:
                        if any(p in name.lower() for p in ['head', 'classifier', 'fc2', 'output']):
                            base_name = name.rsplit('.', 1)[0]
                            if base_name not in self.personalization_layers:
                                self.personalization_layers.append(base_name)
        else:
            raise ValueError("Model template must be dict or nn.Module")

    def local_train(self,
                   client_id: int,
                   data: Tuple[np.ndarray, np.ndarray],
                   epochs: int = 1,
                   lr: float = 0.01) -> Dict[str, np.ndarray]:
        """
        Train with FedPer strategy.

        All layers train locally, but only base layers contribute to global update.
        """
        X, y = data

        # Initialize local model
        if client_id not in self.local_models:
            if isinstance(self.global_model, dict):
                self.local_models[client_id] = {
                    k: v.copy() for k, v in self.global_model.items()
                }
            else:
                self.local_models[client_id] = deepcopy(self.global_model)

        local_model = self.local_models[client_id]

        # Sync base layers from global model
        self._sync_base_layers(local_model)

        if isinstance(local_model, dict):
            return self._train_numpy_fedper(client_id, X, y, epochs, lr)
        else:
            return self._train_torch_fedper(client_id, X, y, epochs, lr)

    def _sync_base_layers(self, local_model: Any) -> None:
        """Sync base (non-personalized) layers from global model."""
        if isinstance(local_model, dict):
            for key in local_model.keys():
                if key not in self.personalization_layers:
                    local_model[key] = self.global_model[key].copy()
        else:
            with torch.no_grad():
                for name, param in local_model.named_parameters():
                    # Check if this parameter belongs to a personalization layer
                    is_personal = any(
                        pl in name for pl in self.personalization_layers
                    )
                    if not is_personal:
                        global_param = dict(self.global_model.named_parameters())[name]
                        param.data.copy_(global_param.data)

    def _train_numpy_fedper(self,
                           client_id: int,
                           X: np.ndarray,
                           y: np.ndarray,
                           epochs: int,
                           lr: float) -> Dict[str, np.ndarray]:
        """Numpy FedPer training."""
        local_model = self.local_models[client_id]
        initial_weights = {k: v.copy() for k, v in local_model.items()}

        for _ in range(epochs):
            batch_size = min(32, len(X))
            indices = np.random.choice(len(X), batch_size, replace=False)
            X_batch, y_batch = X[indices], y[indices]

            if 'weights' in local_model:
                logits = X_batch @ local_model['weights']
                if 'bias' in local_model:
                    logits += local_model['bias']
                probs = 1 / (1 + np.exp(-np.clip(logits, -500, 500)))

                grad_w = X_batch.T @ (probs - y_batch) / batch_size
                local_model['weights'] -= lr * grad_w

                if 'bias' in local_model:
                    grad_b = np.mean(probs - y_batch)
                    local_model['bias'] -= lr * grad_b

        # Return only base layer updates
        delta = {}
        for key in local_model.keys():
            if key not in self.personalization_layers:
                delta[key] = local_model[key] - initial_weights[key]

        return delta

    def _train_torch_fedper(self,
                           client_id: int,
                           X: np.ndarray,
                           y: np.ndarray,
                           epochs: int,
                           lr: float) -> Dict[str, np.ndarray]:
        """PyTorch FedPer training."""
        local_model = self.local_models[client_id]
        device = next(local_model.parameters()).device

        X_tensor = torch.FloatTensor(X).to(device)
        y_tensor = torch.FloatTensor(y).to(device)

        # Store initial weights
        initial_params = {
            name: param.clone()
            for name, param in local_model.named_parameters()
        }

        optimizer = optim.SGD(local_model.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss()

        local_model.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            outputs = local_model(X_tensor).squeeze()
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()

        # Return only base layer updates
        delta = {}
        for name, param in local_model.named_parameters():
            is_personal = any(pl in name for pl in self.personalization_layers)
            if not is_personal:
                delta[name] = (param.data - initial_params[name]).cpu().numpy()

        return delta

    def aggregate(self,
                 client_updates: Dict[int, Dict[str, np.ndarray]],
                 weights: Optional[Dict[int, float]] = None) -> None:
        """Aggregate only base layer updates."""
        if weights is None:
            weights = {cid: 1.0 / len(client_updates) for cid in client_updates}

        total_weight = sum(weights.values())

        if isinstance(self.global_model, dict):
            for key in self.global_model.keys():
                if key not in self.personalization_layers:
                    update = sum(
                        weights[cid] * client_updates[cid].get(key, np.zeros_like(self.global_model[key]))
                        for cid in client_updates
                    ) / total_weight
                    self.global_model[key] += update
        else:
            with torch.no_grad():
                for name, param in self.global_model.named_parameters():
                    is_personal = any(pl in name for pl in self.personalization_layers)
                    if not is_personal and name in client_updates[list(client_updates.keys())[0]]:
                        update = sum(
                            weights[cid] * torch.from_numpy(client_updates[cid][name])
                            for cid in client_updates
                        ) / total_weight
                        param.add_(update.to(param.device))

    def personalize(self,
                   client_id: int,
                   data: Tuple[np.ndarray, np.ndarray]) -> Any:
        """Return client's personalized model."""
        if client_id not in self.local_models:
            self.local_train(client_id, data, epochs=self.config.personalization_epochs)
        return self.local_models[client_id]


# =============================================================================
# APFL (Adaptive Personalized FL)
# =============================================================================

class APFL(PersonalizedFLAlgorithm):
    """
    Adaptive Personalized Federated Learning.

    Key idea: Learn adaptive mixing weight α for each client:

    personalized_model = α * local_model + (1-α) * global_model

    α is learned to minimize local loss, automatically balancing
    between global knowledge and local adaptation.
    """

    def __init__(self, config: PersonalizationConfig):
        super().__init__(config)
        self.alphas: Dict[int, float] = {}  # Per-client mixing weights
        self.initial_alpha = config.alpha

    def initialize(self, model_template: Any) -> None:
        """Initialize global model."""
        if isinstance(model_template, dict):
            self.global_model = {k: v.copy() for k, v in model_template.items()}
        elif TORCH_AVAILABLE and isinstance(model_template, nn.Module):
            self.global_model = deepcopy(model_template)
        else:
            raise ValueError("Model template must be dict or nn.Module")

    def local_train(self,
                   client_id: int,
                   data: Tuple[np.ndarray, np.ndarray],
                   epochs: int = 1,
                   lr: float = 0.01) -> Dict[str, np.ndarray]:
        """
        APFL local training.

        1. Update local model
        2. Update mixing weight α
        3. Return global model update
        """
        X, y = data

        # Initialize local model and alpha
        if client_id not in self.local_models:
            if isinstance(self.global_model, dict):
                self.local_models[client_id] = {
                    k: v.copy() for k, v in self.global_model.items()
                }
            else:
                self.local_models[client_id] = deepcopy(self.global_model)
            self.alphas[client_id] = self.initial_alpha

        if isinstance(self.global_model, dict):
            return self._train_numpy_apfl(client_id, X, y, epochs, lr)
        else:
            return self._train_torch_apfl(client_id, X, y, epochs, lr)

    def _train_numpy_apfl(self,
                         client_id: int,
                         X: np.ndarray,
                         y: np.ndarray,
                         epochs: int,
                         lr: float) -> Dict[str, np.ndarray]:
        """Numpy APFL implementation."""
        local_model = self.local_models[client_id]
        alpha = self.alphas[client_id]

        # Store initial global weights
        w_global = {k: v.copy() for k, v in self.global_model.items()}
        w_local = local_model

        for _ in range(epochs):
            batch_size = min(32, len(X))
            indices = np.random.choice(len(X), batch_size, replace=False)
            X_batch, y_batch = X[indices], y[indices]

            # Compute personalized model
            w_personal = {
                k: alpha * w_local[k] + (1 - alpha) * w_global[k]
                for k in w_local.keys()
            }

            if 'weights' in w_personal:
                # Forward pass with personalized model
                logits = X_batch @ w_personal['weights']
                if 'bias' in w_personal:
                    logits += w_personal['bias']
                probs = 1 / (1 + np.exp(-np.clip(logits, -500, 500)))

                # Gradient for local model
                grad_w = X_batch.T @ (probs - y_batch) / batch_size
                w_local['weights'] -= lr * alpha * grad_w

                if 'bias' in w_local:
                    grad_b = np.mean(probs - y_batch)
                    w_local['bias'] -= lr * alpha * grad_b

                # Update alpha (gradient descent on alpha)
                # dL/dα = (w_local - w_global)^T * dL/dw_personal
                alpha_grad = np.sum(
                    (w_local['weights'] - w_global['weights']) * grad_w
                )
                alpha = np.clip(alpha - 0.01 * alpha_grad, 0.0, 1.0)

        self.local_models[client_id] = w_local
        self.alphas[client_id] = alpha

        # Return global model update
        delta = {
            k: w_local[k] - w_global[k]
            for k in w_local.keys()
        }

        return delta

    def _train_torch_apfl(self,
                         client_id: int,
                         X: np.ndarray,
                         y: np.ndarray,
                         epochs: int,
                         lr: float) -> Dict[str, np.ndarray]:
        """PyTorch APFL implementation."""
        local_model = self.local_models[client_id]
        alpha = self.alphas[client_id]

        device = next(local_model.parameters()).device
        X_tensor = torch.FloatTensor(X).to(device)
        y_tensor = torch.FloatTensor(y).to(device)

        # Store initial params
        initial_params = {
            name: param.clone()
            for name, param in self.global_model.named_parameters()
        }

        optimizer = optim.SGD(local_model.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss()

        alpha_tensor = torch.tensor(alpha, requires_grad=True, device=device)

        local_model.train()
        for _ in range(epochs):
            optimizer.zero_grad()

            # Compute personalized model (interpolation)
            with torch.no_grad():
                for (name, local_param), (_, global_param) in zip(
                    local_model.named_parameters(),
                    self.global_model.named_parameters()
                ):
                    personal_param = alpha_tensor * local_param + (1 - alpha_tensor) * global_param
                    local_param.data.copy_(personal_param)

            outputs = local_model(X_tensor).squeeze()
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()

        self.alphas[client_id] = alpha_tensor.item()

        # Compute delta
        delta = {}
        for (name, param), (_, init_param) in zip(
            local_model.named_parameters(),
            initial_params.items()
        ):
            delta[name] = (param.data - init_param).cpu().numpy()

        return delta

    def aggregate(self,
                 client_updates: Dict[int, Dict[str, np.ndarray]],
                 weights: Optional[Dict[int, float]] = None) -> None:
        """Aggregate with alpha-weighted updates."""
        if weights is None:
            weights = {cid: 1.0 / len(client_updates) for cid in client_updates}

        total_weight = sum(weights.values())

        if isinstance(self.global_model, dict):
            for key in self.global_model.keys():
                update = sum(
                    weights[cid] * (1 - self.alphas.get(cid, 0.5)) * client_updates[cid][key]
                    for cid in client_updates
                ) / total_weight
                self.global_model[key] += update
        else:
            with torch.no_grad():
                for name, param in self.global_model.named_parameters():
                    update = sum(
                        weights[cid] * (1 - self.alphas.get(cid, 0.5)) *
                        torch.from_numpy(client_updates[cid][name])
                        for cid in client_updates
                    ) / total_weight
                    param.add_(update.to(param.device))

    def personalize(self,
                   client_id: int,
                   data: Tuple[np.ndarray, np.ndarray]) -> Any:
        """Return personalized model (interpolated)."""
        if client_id not in self.local_models:
            self.local_train(client_id, data)

        alpha = self.alphas.get(client_id, self.initial_alpha)
        local_model = self.local_models[client_id]

        if isinstance(self.global_model, dict):
            return {
                k: alpha * local_model[k] + (1 - alpha) * self.global_model[k]
                for k in local_model.keys()
            }
        else:
            personal_model = deepcopy(local_model)
            with torch.no_grad():
                for (name, personal_param), (_, local_param), (_, global_param) in zip(
                    personal_model.named_parameters(),
                    local_model.named_parameters(),
                    self.global_model.named_parameters()
                ):
                    personal_param.data = alpha * local_param.data + (1 - alpha) * global_param.data
            return personal_model

    def get_mixing_weights(self) -> Dict[int, float]:
        """Return all client mixing weights."""
        return self.alphas.copy()


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

PERSONALIZED_ALGORITHMS = {
    'ditto': Ditto,
    'per_fedavg': PerFedAvg,
    'fedper': FedPer,
    'apfl': APFL,
}


def create_personalized_fl(algorithm: str,
                          config: Optional[PersonalizationConfig] = None,
                          **kwargs) -> PersonalizedFLAlgorithm:
    """
    Factory function to create personalized FL algorithm.

    Args:
        algorithm: Algorithm name ('ditto', 'per_fedavg', 'fedper', 'apfl')
        config: PersonalizationConfig object
        **kwargs: Additional config parameters

    Returns:
        PersonalizedFLAlgorithm instance
    """
    if algorithm.lower() not in PERSONALIZED_ALGORITHMS:
        raise ValueError(
            f"Unknown algorithm: {algorithm}. "
            f"Available: {list(PERSONALIZED_ALGORITHMS.keys())}"
        )

    if config is None:
        config = PersonalizationConfig(algorithm=algorithm.lower(), **kwargs)

    return PERSONALIZED_ALGORITHMS[algorithm.lower()](config)


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    print("FL-EHDS Personalized Federated Learning Demo")
    print("=" * 60)

    # Generate synthetic heterogeneous data
    np.random.seed(42)

    num_clients = 5
    samples_per_client = 200

    # Each client has different data distribution
    client_data = {}
    for i in range(num_clients):
        # Different mean for each client (simulating hospital populations)
        mean_shift = (i - 2) * 0.5
        X = np.random.randn(samples_per_client, 5) + mean_shift
        X = np.hstack([X, np.ones((samples_per_client, 1))])  # Add bias

        # Non-linear decision boundary varies by client
        noise = np.random.randn(samples_per_client) * 0.3
        y = (X[:, 0] + X[:, 1] * (1 + i * 0.2) + noise > mean_shift).astype(float)

        client_data[i] = (X, y)

    # Test Ditto
    print("\n" + "-" * 60)
    print("Testing Ditto Algorithm")
    print("-" * 60)

    config = PersonalizationConfig(algorithm='ditto', lambda_reg=0.1)
    ditto = Ditto(config)

    # Initialize with simple model
    model_template = {
        'weights': np.zeros(6),
        'bias': np.zeros(1)
    }
    ditto.initialize(model_template)

    # Training rounds
    for round_num in range(10):
        updates = {}
        for client_id, data in client_data.items():
            updates[client_id] = ditto.local_train(client_id, data, epochs=3)

        ditto.aggregate(updates)

    # Evaluate
    print("\nPer-client accuracy with personalized models:")
    for client_id, (X, y) in client_data.items():
        personal_model = ditto.personalize(client_id, (X, y))
        logits = X @ personal_model['weights'] + personal_model['bias']
        preds = (logits > 0).astype(float)
        acc = np.mean(preds.flatten() == y)
        print(f"  Client {client_id}: {acc:.2%}")

    # Test APFL
    print("\n" + "-" * 60)
    print("Testing APFL Algorithm")
    print("-" * 60)

    config = PersonalizationConfig(algorithm='apfl', alpha=0.5)
    apfl = APFL(config)
    apfl.initialize(model_template)

    for round_num in range(10):
        updates = {}
        for client_id, data in client_data.items():
            updates[client_id] = apfl.local_train(client_id, data, epochs=3)
        apfl.aggregate(updates)

    print("\nLearned mixing weights (α):")
    for client_id, alpha in apfl.get_mixing_weights().items():
        print(f"  Client {client_id}: α = {alpha:.3f}")

    print("\nPer-client accuracy with APFL:")
    for client_id, (X, y) in client_data.items():
        personal_model = apfl.personalize(client_id, (X, y))
        logits = X @ personal_model['weights'] + personal_model['bias']
        preds = (logits > 0).astype(float)
        acc = np.mean(preds.flatten() == y)
        print(f"  Client {client_id}: {acc:.2%}")

    print("\n" + "=" * 60)
    print("Demo completed!")

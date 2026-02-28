"""
Federated Learning trainer for tabular healthcare data.
"""

import numpy as np
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from copy import deepcopy
import time
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from terminal.training.data_generation import _detect_device, generate_healthcare_data
from terminal.training.models import HealthcareMLP


@dataclass
class ClientResult:
    """Result from client local training."""
    client_id: int
    model_update: Dict[str, torch.Tensor]
    num_samples: int
    train_loss: float
    train_acc: float
    epochs_completed: int
    quality_score: Optional[float] = None
    epoch_metrics: Optional[List[Dict[str, float]]] = None


@dataclass
class RoundResult:
    """Result from one FL round."""
    round_num: int
    global_loss: float
    global_acc: float
    global_f1: float
    global_precision: float
    global_recall: float
    global_auc: float
    client_results: List[ClientResult]
    time_seconds: float
    # Byzantine defense results (populated when byzantine_config is set)
    byzantine_selected: Optional[List[int]] = None
    byzantine_rejected: Optional[List[int]] = None
    byzantine_trust_scores: Optional[Dict[int, float]] = None


class FederatedTrainer:
    """Real Federated Learning trainer with PyTorch."""

    def __init__(
        self,
        num_clients: int = 5,
        samples_per_client: int = 200,
        algorithm: str = "FedAvg",
        local_epochs: int = 3,
        batch_size: int = 32,
        learning_rate: float = 0.01,
        is_iid: bool = False,
        alpha: float = 0.5,
        mu: float = 0.1,  # FedProx, Ditto lambda
        dp_enabled: bool = False,
        dp_epsilon: float = 10.0,
        dp_clip_norm: float = 1.0,
        seed: int = 42,
        device: str = "cpu",
        progress_callback: Optional[Callable] = None,
        # Server optimizer params (FedAdam, FedYogi, FedAdagrad)
        server_lr: float = 0.1,
        beta1: float = 0.9,
        beta2: float = 0.99,
        tau: float = 1e-3,
        # External data (e.g. from FHIR pipeline)
        external_data: Optional[Dict[int, tuple]] = None,
        external_test_data: Optional[Dict[int, tuple]] = None,
        input_dim: Optional[int] = None,
        num_classes: Optional[int] = None,
        # Byzantine defense
        byzantine_config=None,
        # New algorithm params (2022-2023)
        fedlc_tau: float = 0.5,
        fedsam_rho: float = 0.05,
        feddecorr_beta: float = 0.1,
        fedspeed_alpha: float = 0.5,
        fedspeed_rho: float = 0.05,
        fedspeed_lambda: float = 0.1,
        # DP mode
        dp_mode: str = "central",  # "central" (server-side noise) or "local" (client-side noise)
    ):
        # Validate parameters
        if num_clients < 2:
            raise ValueError(f"num_clients must be >= 2 for federated learning, got {num_clients}")
        if local_epochs < 1:
            raise ValueError(f"local_epochs must be >= 1, got {local_epochs}")
        if learning_rate <= 0 or learning_rate >= 10:
            raise ValueError(f"learning_rate must be in (0, 10), got {learning_rate}")
        if dp_enabled and dp_epsilon <= 0:
            raise ValueError(f"dp_epsilon must be > 0 when DP is enabled, got {dp_epsilon}")
        if dp_enabled and dp_clip_norm <= 0:
            raise ValueError(f"dp_clip_norm must be > 0 when DP is enabled, got {dp_clip_norm}")

        SUPPORTED_ALGORITHMS = [
            "FedAvg", "FedProx", "SCAFFOLD", "FedAdam", "FedYogi",
            "FedAdagrad", "FedNova", "FedDyn", "Per-FedAvg", "Ditto",
            "FedLC", "FedSAM", "FedDecorr", "FedSpeed", "FedExP",
            "FedLESAM", "HPFL"
        ]
        if algorithm not in SUPPORTED_ALGORITHMS:
            raise ValueError(
                f"Unknown algorithm '{algorithm}'. "
                f"Supported: {SUPPORTED_ALGORITHMS}"
            )

        self.num_clients = num_clients
        self.algorithm = algorithm
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.mu = mu  # FedProx proximal term, Ditto regularization
        self.dp_enabled = dp_enabled
        self.dp_epsilon = dp_epsilon
        self.dp_clip_norm = dp_clip_norm
        self.seed = seed
        self.device = _detect_device(device)
        self.progress_callback = progress_callback

        # Server optimizer hyperparameters
        self.server_lr = server_lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.tau = tau

        # New algorithm params
        self.fedlc_tau = fedlc_tau
        self.fedsam_rho = fedsam_rho
        self.feddecorr_beta = feddecorr_beta
        self.fedspeed_alpha = fedspeed_alpha
        self.fedspeed_rho = fedspeed_rho
        self.fedspeed_lambda = fedspeed_lambda

        # DP mode
        self.dp_mode = dp_mode

        torch.manual_seed(seed)
        np.random.seed(seed)

        # Load data: external (FHIR) or generate synthetic
        if external_data is not None and external_test_data is not None:
            self.client_data = external_data
            self.client_test_data = external_test_data
            self.num_clients = len(external_data)
            first_X = next(iter(external_data.values()))[0]
            _input_dim = input_dim or first_X.shape[1]
        else:
            self.client_data, self.client_test_data = generate_healthcare_data(
                num_clients=num_clients,
                samples_per_client=samples_per_client,
                is_iid=is_iid,
                alpha=alpha,
                seed=seed
            )
            _input_dim = input_dim or 10

        # Initialize global model
        all_labels = np.concatenate([y for _, y in self.client_data.values()])
        _num_classes = num_classes or len(np.unique(all_labels))
        self.global_model = HealthcareMLP(input_dim=_input_dim, num_classes=_num_classes).to(self.device)

        # Class-weighted loss for imbalanced tabular datasets
        self.class_weights = None
        num_classes = _num_classes
        counts = np.bincount(all_labels.astype(int), minlength=num_classes)
        if counts.min() > 0:
            max_ratio = counts.max() / counts.min()
            if max_ratio > 1.5:
                weights = len(all_labels) / (num_classes * counts.astype(np.float64))
                weights = weights / weights.mean()
                self.class_weights = torch.FloatTensor(weights).to(self.device)

        # SCAFFOLD control variates
        if algorithm == "SCAFFOLD":
            self.server_control = {
                name: torch.zeros_like(param)
                for name, param in self.global_model.named_parameters()
            }
            self.client_controls = {
                i: {name: torch.zeros_like(param)
                    for name, param in self.global_model.named_parameters()}
                for i in range(num_clients)
            }

        # FedAdam, FedYogi, FedAdagrad: server momentum and velocity
        if algorithm in ["FedAdam", "FedYogi", "FedAdagrad"]:
            self.server_momentum = {
                name: torch.zeros_like(param)
                for name, param in self.global_model.named_parameters()
            }
            self.server_velocity = {
                name: torch.ones_like(param) * (tau ** 2)
                for name, param in self.global_model.named_parameters()
            }

        # Per-FedAvg, Ditto: personalized models per client
        if algorithm in ["Per-FedAvg", "Ditto"]:
            self.personalized_models = {
                i: deepcopy(self.global_model)
                for i in range(num_clients)
            }

        # FedDyn: server state h and per-client gradient corrections
        if algorithm == "FedDyn":
            self.server_h = {
                name: torch.zeros_like(param)
                for name, param in self.global_model.named_parameters()
            }
            self.client_grad_corrections = {
                i: {name: torch.zeros_like(param)
                    for name, param in self.global_model.named_parameters()}
                for i in range(num_clients)
            }

        # FedNova: track local steps
        self.client_steps = {i: 0 for i in range(num_clients)}

        # FedSpeed: prox-correction per client
        if algorithm == "FedSpeed":
            self.fedspeed_corrections = {
                i: {name: torch.zeros_like(param)
                    for name, param in self.global_model.named_parameters()}
                for i in range(num_clients)
            }

        # FedLESAM: previous global model params (Fan et al., ICML 2024)
        if algorithm == "FedLESAM":
            self.prev_global_params = {
                name: param.clone().detach()
                for name, param in self.global_model.named_parameters()
            }

        # HPFL: local classifiers per client (Shen et al., ICLR 2025)
        if algorithm == "HPFL":
            self._hpfl_classifier_names = self._identify_classifier_params()
            self.client_classifiers = {
                i: {n: self.global_model.state_dict()[n].clone()
                    for n in self._hpfl_classifier_names}
                for i in range(num_clients)
            }

        # Byzantine defense (lazy-initialized)
        self._byzantine_manager = None
        self._last_byzantine_result = None
        if byzantine_config is not None:
            from core.byzantine_resilience import ByzantineDefenseManager
            self._byzantine_manager = ByzantineDefenseManager(byzantine_config)

        # History
        self.history = []

    def _identify_classifier_params(self):
        """Identify classifier (head) parameter names for HPFL algorithm."""
        names = set()
        for n, _ in self.global_model.named_parameters():
            if 'classifier' in n or 'fc.' in n or 'head.' in n:
                names.add(n)
        if names:
            return names
        # Fallback: last Linear layer (for MLP-style models)
        last_prefix = None
        for n, p in self.global_model.named_parameters():
            if 'weight' in n and p.dim() == 2:
                last_prefix = n.rsplit('.', 1)[0]
        if last_prefix:
            for n, _ in self.global_model.named_parameters():
                if n.startswith(last_prefix + '.'):
                    names.add(n)
        return names

    def _rebuild_model(self, new_input_dim: int):
        """Rebuild model and algorithm state after data minimization changes input_dim."""
        self.global_model = HealthcareMLP(input_dim=new_input_dim).to(self.device)

        # Reinit SCAFFOLD control variates
        if self.algorithm == "SCAFFOLD":
            self.server_control = {
                name: torch.zeros_like(param)
                for name, param in self.global_model.named_parameters()
            }
            self.client_controls = {
                i: {name: torch.zeros_like(param)
                    for name, param in self.global_model.named_parameters()}
                for i in range(self.num_clients)
            }

        # Reinit FedAdam/FedYogi/FedAdagrad momentum/velocity
        if self.algorithm in ["FedAdam", "FedYogi", "FedAdagrad"]:
            self.server_momentum = {
                name: torch.zeros_like(param)
                for name, param in self.global_model.named_parameters()
            }
            self.server_velocity = {
                name: torch.ones_like(param) * (self.tau ** 2)
                for name, param in self.global_model.named_parameters()
            }

        # Reinit personalized models
        if self.algorithm in ["Per-FedAvg", "Ditto"]:
            self.personalized_models = {
                i: deepcopy(self.global_model)
                for i in range(self.num_clients)
            }

        # Reinit FedDyn state
        if self.algorithm == "FedDyn":
            self.server_h = {
                name: torch.zeros_like(param)
                for name, param in self.global_model.named_parameters()
            }
            self.client_grad_corrections = {
                i: {name: torch.zeros_like(param)
                    for name, param in self.global_model.named_parameters()}
                for i in range(self.num_clients)
            }

    def _get_client_dataloader(self, client_id: int) -> DataLoader:
        """Create DataLoader for client."""
        X, y = self.client_data[client_id]
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)
        dataset = TensorDataset(X_tensor, y_tensor)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def _train_client(
        self,
        client_id: int,
        round_num: int
    ) -> ClientResult:
        """Train one client locally."""
        # Copy global model
        local_model = deepcopy(self.global_model)
        local_model.train()

        optimizer = torch.optim.SGD(local_model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss(weight=self.class_weights)

        dataloader = self._get_client_dataloader(client_id)

        # Store initial params for FedProx, Ditto, FedDyn, and FedSpeed
        if self.algorithm in ["FedProx", "Ditto", "FedDyn", "FedSpeed"]:
            global_params = {
                name: param.clone().detach()
                for name, param in self.global_model.named_parameters()
            }

        # HPFL: load client's local classifier before training
        if self.algorithm == "HPFL":
            for n, p in local_model.named_parameters():
                if n in self._hpfl_classifier_names:
                    p.data.copy_(self.client_classifiers[client_id][n])

        # Store initial params for SCAFFOLD
        if self.algorithm == "SCAFFOLD":
            init_params = {
                name: param.clone().detach()
                for name, param in local_model.named_parameters()
            }

        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        steps = 0

        for epoch in range(self.local_epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_samples = 0

            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()

                outputs = local_model(batch_X)

                # FedLC: calibrate logits before loss
                if self.algorithm == "FedLC":
                    if not hasattr(self, '_fedlc_margins'):
                        self._fedlc_margins = {}
                    if client_id not in self._fedlc_margins:
                        _, y_all = self.client_data[client_id]
                        n_classes = outputs.shape[1]
                        counts = np.bincount(y_all.astype(int), minlength=n_classes).astype(float)
                        counts = np.maximum(counts, 1.0)
                        margins = self.fedlc_tau * np.power(counts, -0.25)
                        self._fedlc_margins[client_id] = torch.FloatTensor(margins).to(self.device)
                    loss = criterion(outputs - self._fedlc_margins[client_id].unsqueeze(0), batch_y)
                else:
                    loss = criterion(outputs, batch_y)

                # FedProx: add proximal term
                if self.algorithm == "FedProx":
                    prox_term = 0.0
                    for name, param in local_model.named_parameters():
                        prox_term += torch.sum((param - global_params[name]) ** 2)
                    loss += (self.mu / 2) * prox_term

                # FedDyn: proximal + dynamic linear correction (Acar et al. 2021)
                if self.algorithm == "FedDyn":
                    prox_term = 0.0
                    linear_term = 0.0
                    for name, param in local_model.named_parameters():
                        prox_term += torch.sum((param - global_params[name]) ** 2)
                        linear_term += torch.sum(
                            self.client_grad_corrections[client_id][name] * param
                        )
                    loss += (self.mu / 2) * prox_term - linear_term

                # FedSpeed: proximal + prox-correction
                if self.algorithm == "FedSpeed":
                    prox_term = 0.0
                    for name, param in local_model.named_parameters():
                        prox_term += torch.sum((param - global_params[name]) ** 2)
                    loss += (1.0 / (2.0 * self.fedspeed_lambda)) * prox_term

                loss.backward()

                # SCAFFOLD: apply control variate correction after backward
                if self.algorithm == "SCAFFOLD":
                    for name, param in local_model.named_parameters():
                        if param.grad is not None:
                            correction = self.client_controls[client_id][name] - self.server_control[name]
                            # Clamp correction to prevent divergence with adaptive optimizers
                            grad_norm = param.grad.data.norm()
                            corr_norm = correction.norm()
                            if corr_norm > 0 and grad_norm > 0:
                                max_corr = grad_norm * 2.0
                                if corr_norm > max_corr:
                                    correction = correction * (max_corr / corr_norm)
                            param.grad.data += correction

                # FedSAM: evaluate gradient at perturbed position
                if self.algorithm == "FedSAM":
                    # Save current gradients
                    grad_norm = torch.nn.utils.clip_grad_norm_(local_model.parameters(), float('inf'))
                    if grad_norm > 1e-12:
                        # Compute perturbation epsilon
                        old_params = {}
                        for name, param in local_model.named_parameters():
                            old_params[name] = param.data.clone()
                            if param.grad is not None:
                                param.data += self.fedsam_rho * param.grad / grad_norm
                        # Recompute gradient at perturbed position
                        optimizer.zero_grad()
                        outputs2 = local_model(batch_X)
                        loss2 = criterion(outputs2, batch_y)
                        loss2.backward()
                        # Restore original parameters
                        for name, param in local_model.named_parameters():
                            param.data = old_params[name]

                # FedLESAM: perturb in global direction (Fan et al., ICML 2024)
                if self.algorithm == "FedLESAM":
                    global_params_dict = dict(self.global_model.named_parameters())
                    delta_norm = sum(
                        (self.prev_global_params[n] - global_params_dict[n].data).norm() ** 2
                        for n in self.prev_global_params
                    ) ** 0.5

                    if delta_norm > 1e-12:
                        old_params = {n: p.data.clone() for n, p in local_model.named_parameters()}
                        for n, p in local_model.named_parameters():
                            delta = self.prev_global_params[n] - global_params_dict[n].data
                            p.data += self.fedsam_rho * delta / delta_norm
                        optimizer.zero_grad()
                        outputs2 = local_model(batch_X)
                        loss2 = criterion(outputs2, batch_y)
                        loss2.backward()
                        for n, p in local_model.named_parameters():
                            p.data = old_params[n]

                # FedDecorr: add decorrelation gradient
                if self.algorithm == "FedDecorr":
                    # Get intermediate representations (before final layer)
                    with torch.no_grad():
                        # Get features from penultimate layer
                        x = batch_X
                        for layer in list(local_model.children())[:-1]:
                            x = layer(x)
                            if hasattr(x, 'relu'):
                                x = torch.relu(x)
                    # We add decorrelation loss as a separate backward pass on the representations
                    # Using the logits as proxy for representations
                    reps = outputs.detach()
                    if reps.shape[1] > 1:
                        reps_centered = reps - reps.mean(dim=0)
                        K = (reps_centered.T @ reps_centered) / len(batch_X)
                        # Zero diagonal
                        K = K - torch.diag(torch.diag(K))
                        decorr_loss = self.feddecorr_beta * (K ** 2).sum() / (reps.shape[1] ** 2)
                        # Add to gradients via separate backward
                        if decorr_loss.requires_grad:
                            decorr_loss.backward()

                # FedSpeed: apply prox-correction to gradients
                if self.algorithm == "FedSpeed":
                    for name, param in local_model.named_parameters():
                        if param.grad is not None:
                            correction = self.fedspeed_corrections[client_id].get(name, torch.zeros_like(param))
                            param.grad.data -= correction

                # DP: clip gradients
                if self.dp_enabled:
                    torch.nn.utils.clip_grad_norm_(
                        local_model.parameters(),
                        self.dp_clip_norm
                    )

                # Local DP: add noise on client side before sending
                if self.dp_enabled and self.dp_mode == "local":
                    noise_scale = self.dp_clip_norm * np.sqrt(2 * np.log(1.25 / 1e-5)) / self.dp_epsilon
                    for param in local_model.parameters():
                        if param.grad is not None:
                            noise = torch.randn_like(param.grad) * noise_scale
                            param.grad.data += noise

                optimizer.step()

                # Metrics
                epoch_loss += loss.item() * len(batch_y)
                preds = outputs.argmax(dim=1)
                epoch_correct += (preds == batch_y).sum().item()
                epoch_samples += len(batch_y)
                steps += 1

            total_loss += epoch_loss
            total_correct += epoch_correct
            total_samples += epoch_samples

            # Progress callback for epoch
            if self.progress_callback:
                self.progress_callback(
                    "epoch",
                    client_id=client_id,
                    epoch=epoch + 1,
                    total_epochs=self.local_epochs,
                    loss=epoch_loss / epoch_samples,
                    acc=epoch_correct / epoch_samples
                )

        self.client_steps[client_id] = steps

        # Compute model update
        model_update = {}
        for name, param in local_model.named_parameters():
            global_param = dict(self.global_model.named_parameters())[name]
            model_update[name] = param.data - global_param.data

        # SCAFFOLD: update client control variate
        if self.algorithm == "SCAFFOLD":
            K = steps
            for name in self.client_controls[client_id]:
                local_param = dict(local_model.named_parameters())[name].data
                init_param = init_params[name]
                c_i = self.client_controls[client_id][name]
                c = self.server_control[name]

                # c_i_new = c_i - c + (init - local) / (K * lr)
                new_ci = c_i - c + (init_param - local_param) / (K * self.learning_rate)
                # Clamp control variates to prevent unbounded growth
                max_cv = 10.0
                self.client_controls[client_id][name] = torch.clamp(new_ci, -max_cv, max_cv)

        # FedDyn: update client gradient correction
        if self.algorithm == "FedDyn":
            for name, param in local_model.named_parameters():
                global_param = dict(self.global_model.named_parameters())[name]
                self.client_grad_corrections[client_id][name] -= (
                    self.mu * (param.data - global_param.data)
                )

        # FedSpeed: update prox-correction
        if self.algorithm == "FedSpeed":
            for name, param in local_model.named_parameters():
                global_param = dict(self.global_model.named_parameters())[name]
                self.fedspeed_corrections[client_id][name] -= (
                    (1.0 / self.fedspeed_lambda) * (param.data - global_param.data)
                )

        # HPFL: save client's updated classifier (Shen et al., ICLR 2025)
        if self.algorithm == "HPFL":
            self.client_classifiers[client_id] = {
                n: p.data.clone()
                for n, p in local_model.named_parameters()
                if n in self._hpfl_classifier_names
            }

        # Ditto: train personalized model with regularization towards global
        if self.algorithm == "Ditto":
            self._train_personalized_ditto(client_id, dataloader, criterion)

        # Per-FedAvg: fine-tune personalized model
        if self.algorithm == "Per-FedAvg":
            self._train_personalized_perfedavg(client_id, dataloader, criterion)

        return ClientResult(
            client_id=client_id,
            model_update=model_update,
            num_samples=total_samples,
            train_loss=total_loss / total_samples,
            train_acc=total_correct / total_samples,
            epochs_completed=self.local_epochs
        )

    def _train_personalized_ditto(self, client_id: int, dataloader: DataLoader, criterion):
        """Train personalized model for Ditto algorithm."""
        # Ditto: personalized model trained with L2 regularization towards global model
        pers_model = self.personalized_models[client_id]
        pers_model.train()

        optimizer = torch.optim.SGD(pers_model.parameters(), lr=self.learning_rate)

        for epoch in range(self.local_epochs):
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()

                outputs = pers_model(batch_X)
                loss = criterion(outputs, batch_y)

                # L2 regularization towards global model (lambda = mu)
                reg_term = 0.0
                for (name, pers_param), (_, global_param) in zip(
                    pers_model.named_parameters(),
                    self.global_model.named_parameters()
                ):
                    reg_term += torch.sum((pers_param - global_param.detach()) ** 2)
                loss += (self.mu / 2) * reg_term

                loss.backward()
                optimizer.step()

    def _train_personalized_perfedavg(self, client_id: int, dataloader: DataLoader, criterion):
        """Fine-tune personalized model for Per-FedAvg algorithm."""
        # Per-FedAvg: start from global model and do local fine-tuning
        pers_model = self.personalized_models[client_id]

        # Copy global model weights to personalized model
        pers_model.load_state_dict(self.global_model.state_dict())
        pers_model.train()

        # Use smaller learning rate for fine-tuning
        fine_tune_lr = self.learning_rate * 0.1
        optimizer = torch.optim.SGD(pers_model.parameters(), lr=fine_tune_lr)

        # Just one epoch of fine-tuning
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = pers_model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    def _aggregate_byzantine(self, client_results: List[ClientResult]) -> None:
        """Aggregate using Byzantine-resilient method via ByzantineDefenseManager.

        Converts PyTorch model updates to numpy, runs the Byzantine aggregator,
        then applies the robust aggregate back to the global model.
        """
        from terminal.training.byzantine_bridge import (
            client_results_to_gradients,
            aggregation_result_to_tensors,
        )

        gradients = client_results_to_gradients(client_results)
        result = self._byzantine_manager.aggregate(gradients)
        self._last_byzantine_result = result

        # Apply aggregated gradient to global model
        # HPFL: skip classifier params (keep local per client)
        robust_update = aggregation_result_to_tensors(result, self.device)
        for name, param in self.global_model.named_parameters():
            if name in robust_update:
                if self.algorithm == "HPFL" and name in self._hpfl_classifier_names:
                    continue
                param.data += robust_update[name]

    def _aggregate(self, client_results: List[ClientResult],
                   noise_scale_override: Optional[float] = None,
                   quality_weights: Optional[Dict[int, float]] = None) -> None:
        """Aggregate client updates to global model.

        Args:
            client_results: Per-client training results with model updates.
            noise_scale_override: Override DP noise scale (from jurisdiction privacy).
            quality_weights: Optional {client_id: quality_weight} from Data Quality
                Framework. If provided, aggregation weights are multiplied by quality
                scores: w_h = (n_h/N) * q_h, then normalized.
        """
        # Byzantine defense: use robust aggregation instead of standard
        if self._byzantine_manager is not None:
            self._aggregate_byzantine(client_results)
            return

        # Pre-compute normalized weights: sample-proportional * quality modifier
        total_samples = sum(cr.num_samples for cr in client_results)
        raw_weights = {}
        for cr in client_results:
            w = cr.num_samples / total_samples
            if quality_weights and cr.client_id in quality_weights:
                w *= quality_weights[cr.client_id]
            raw_weights[cr.client_id] = w
        total_w = sum(raw_weights.values())
        if total_w > 0:
            cw = {cid: w / total_w for cid, w in raw_weights.items()}
        else:
            cw = {cr.client_id: 1.0 / len(client_results) for cr in client_results}

        if self.algorithm == "FedNova":
            # FedNova (Wang et al. 2020): normalized averaging
            tau_eff = 0.0
            for cr in client_results:
                tau_eff += cw[cr.client_id] * self.client_steps[cr.client_id]

            for name, param in self.global_model.named_parameters():
                normalized_avg = torch.zeros_like(param)
                for cr in client_results:
                    tau_i = max(self.client_steps[cr.client_id], 1)
                    normalized_avg += cw[cr.client_id] * (cr.model_update[name] / tau_i)
                param.data += tau_eff * normalized_avg

        elif self.algorithm == "FedAdam":
            # FedAdam (Reddi et al. 2021): Adam optimizer on server side
            # with bias correction (Kingma & Ba, 2015)
            round_t = len(self.history) + 1
            bc1 = 1 - self.beta1 ** round_t
            bc2 = 1 - self.beta2 ** round_t
            for name, param in self.global_model.named_parameters():
                delta = torch.zeros_like(param)
                for cr in client_results:
                    delta += cr.model_update[name] * cw[cr.client_id]

                self.server_momentum[name] = (
                    self.beta1 * self.server_momentum[name] +
                    (1 - self.beta1) * delta
                )
                self.server_velocity[name] = (
                    self.beta2 * self.server_velocity[name] +
                    (1 - self.beta2) * (delta ** 2)
                )
                m_hat = self.server_momentum[name] / bc1
                v_hat = self.server_velocity[name] / bc2
                param.data += self.server_lr * m_hat / (
                    torch.sqrt(v_hat) + self.tau
                )

        elif self.algorithm == "FedYogi":
            # FedYogi (Reddi et al. 2021): controlled adaptive learning rate
            # with bias correction (Kingma & Ba, 2015)
            round_t = len(self.history) + 1
            bc1 = 1 - self.beta1 ** round_t
            bc2 = 1 - self.beta2 ** round_t
            for name, param in self.global_model.named_parameters():
                delta = torch.zeros_like(param)
                for cr in client_results:
                    delta += cr.model_update[name] * cw[cr.client_id]

                self.server_momentum[name] = (
                    self.beta1 * self.server_momentum[name] +
                    (1 - self.beta1) * delta
                )
                delta_sq = delta ** 2
                sign = torch.sign(self.server_velocity[name] - delta_sq)
                self.server_velocity[name] = (
                    self.server_velocity[name] -
                    (1 - self.beta2) * sign * delta_sq
                )
                m_hat = self.server_momentum[name] / bc1
                v_hat = self.server_velocity[name] / bc2
                param.data += self.server_lr * m_hat / (
                    torch.sqrt(v_hat) + self.tau
                )

        elif self.algorithm == "FedAdagrad":
            # FedAdagrad: Adagrad on server side (no momentum, no bias correction)
            for name, param in self.global_model.named_parameters():
                delta = torch.zeros_like(param)
                for cr in client_results:
                    delta += cr.model_update[name] * cw[cr.client_id]

                self.server_velocity[name] = self.server_velocity[name] + (delta ** 2)
                param.data += self.server_lr * delta / (
                    torch.sqrt(self.server_velocity[name]) + self.tau
                )

        elif self.algorithm == "FedDyn":
            # FedDyn (Acar et al. 2021): weighted avg + server correction
            for name, param in self.global_model.named_parameters():
                weighted_update = torch.zeros_like(param)
                for cr in client_results:
                    weighted_update += cr.model_update[name] * cw[cr.client_id]

                self.server_h[name] -= self.mu * weighted_update
                param.data += weighted_update - (1.0 / self.mu) * self.server_h[name]

        elif self.algorithm == "FedExP":
            # FedExP (Jhunjhunwala et al., ICLR 2023): adaptive server step size
            N = len(client_results)
            # Compute sum of individual update norms squared
            sum_norm_sq = 0.0
            for cr in client_results:
                for name in cr.model_update:
                    sum_norm_sq += torch.sum(cr.model_update[name] ** 2).item()

            # Compute weighted average update
            avg_update = {}
            for name, param in self.global_model.named_parameters():
                avg = torch.zeros_like(param)
                for cr in client_results:
                    avg += cr.model_update[name] * cw[cr.client_id]
                avg_update[name] = avg

            # Compute average norm squared
            avg_norm_sq = sum(torch.sum(v ** 2).item() for v in avg_update.values())

            # Adaptive step: eta_s = max(1, N * ||avg||^2 / sum ||delta_i||^2)
            if sum_norm_sq > 1e-12:
                eta_s = max(1.0, N * avg_norm_sq / sum_norm_sq)
            else:
                eta_s = 1.0

            for name, param in self.global_model.named_parameters():
                param.data += eta_s * avg_update[name]

        else:
            # FedAvg, FedProx, SCAFFOLD, Per-FedAvg, Ditto, FedLC, FedSAM,
            # FedDecorr, FedSpeed, FedLESAM, HPFL: weighted average
            for name, param in self.global_model.named_parameters():
                # HPFL: skip classifier params (keep local per client)
                if self.algorithm == "HPFL" and name in self._hpfl_classifier_names:
                    continue
                weighted_update = torch.zeros_like(param)
                for cr in client_results:
                    weighted_update += cr.model_update[name] * cw[cr.client_id]

                param.data += weighted_update

        # FedLESAM: update prev_global_params after aggregation
        if self.algorithm == "FedLESAM":
            self.prev_global_params = {
                name: param.clone().detach()
                for name, param in self.global_model.named_parameters()
            }

        # SCAFFOLD: update server control variate using proper delta
        if self.algorithm == "SCAFFOLD":
            n = len(client_results)
            for name in self.server_control:
                delta_c = torch.zeros_like(self.server_control[name])
                for cr in client_results:
                    # delta_c_i = c_i_new - c_i_old
                    delta_c += (
                        self.client_controls[cr.client_id][name]
                        - self._old_client_controls[cr.client_id][name]
                    )
                self.server_control[name] += delta_c / n

        # Central DP: add noise to aggregated model (server-side)
        if self.dp_enabled and self.dp_mode == "central":
            if noise_scale_override is not None:
                noise_scale = noise_scale_override
            else:
                noise_scale = self.dp_clip_norm / self.dp_epsilon
            for param in self.global_model.parameters():
                noise = torch.randn_like(param) * noise_scale
                param.data += noise

    def _evaluate(self) -> Dict[str, float]:
        """Evaluate global model on held-out TEST data (not training data)."""
        self.global_model.eval()

        total_loss = 0.0
        total_samples = 0
        all_preds = []
        all_labels = []
        all_probs = []

        criterion = nn.CrossEntropyLoss()

        # HPFL: save global classifier before per-client evaluation
        if self.algorithm == "HPFL":
            _saved_cls = {n: p.data.clone() for n, p in self.global_model.named_parameters()
                          if n in self._hpfl_classifier_names}

        with torch.no_grad():
            for client_id in range(self.num_clients):
                # HPFL: load client's local classifier
                if self.algorithm == "HPFL":
                    for n, p in self.global_model.named_parameters():
                        if n in self._hpfl_classifier_names:
                            p.data.copy_(self.client_classifiers[client_id][n])

                X, y = self.client_test_data[client_id]
                X_tensor = torch.FloatTensor(X).to(self.device)
                y_tensor = torch.LongTensor(y).to(self.device)

                outputs = self.global_model(X_tensor)
                loss = criterion(outputs, y_tensor)

                total_loss += loss.item() * len(y)
                preds = outputs.argmax(dim=1)
                probs = torch.softmax(outputs, dim=1)[:, 1]  # Probability of class 1

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y_tensor.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                total_samples += len(y)

        # HPFL: restore global classifier after evaluation
        if self.algorithm == "HPFL":
            for n, p in self.global_model.named_parameters():
                if n in self._hpfl_classifier_names:
                    p.data.copy_(_saved_cls[n])

        # Convert to numpy arrays
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        # Calculate metrics
        accuracy = (all_preds == all_labels).mean()

        # True positives, false positives, false negatives
        tp = ((all_preds == 1) & (all_labels == 1)).sum()
        fp = ((all_preds == 1) & (all_labels == 0)).sum()
        fn = ((all_preds == 0) & (all_labels == 1)).sum()

        # Precision, Recall, F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        # AUC-ROC (simple calculation)
        # Sort by probability and calculate
        sorted_indices = np.argsort(all_probs)[::-1]
        sorted_labels = all_labels[sorted_indices]
        n_pos = (all_labels == 1).sum()
        n_neg = (all_labels == 0).sum()

        if n_pos > 0 and n_neg > 0:
            tpr_sum = 0.0
            tp_count = 0
            for label in sorted_labels:
                if label == 1:
                    tp_count += 1
                else:
                    tpr_sum += tp_count / n_pos
            auc = tpr_sum / n_neg
        else:
            auc = 0.5

        return {
            "loss": total_loss / total_samples,
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "auc": float(auc),
        }

    def _evaluate_personalized(self) -> Dict[str, float]:
        """Evaluate personalized models (Per-FedAvg/Ditto) on per-client test data."""
        all_preds = []
        all_labels = []
        all_probs = []
        total_loss = 0.0
        total_samples = 0
        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for client_id in range(self.num_clients):
                pers_model = self.personalized_models[client_id]
                pers_model.eval()
                X, y = self.client_test_data[client_id]
                X_tensor = torch.FloatTensor(X).to(self.device)
                y_tensor = torch.LongTensor(y).to(self.device)

                outputs = pers_model(X_tensor)
                loss = criterion(outputs, y_tensor)
                total_loss += loss.item() * len(y)
                preds = outputs.argmax(dim=1)
                probs = torch.softmax(outputs, dim=1)[:, 1]

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y_tensor.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                total_samples += len(y)

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        accuracy = (all_preds == all_labels).mean()
        tp = ((all_preds == 1) & (all_labels == 1)).sum()
        fp = ((all_preds == 1) & (all_labels == 0)).sum()
        fn = ((all_preds == 0) & (all_labels == 1)).sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        n_pos = (all_labels == 1).sum()
        n_neg = (all_labels == 0).sum()
        if n_pos > 0 and n_neg > 0:
            sorted_indices = np.argsort(all_probs)[::-1]
            sorted_labels = all_labels[sorted_indices]
            tpr_sum = 0.0
            tp_count = 0
            for label in sorted_labels:
                if label == 1:
                    tp_count += 1
                else:
                    tpr_sum += tp_count / n_pos
            auc = tpr_sum / n_neg
        else:
            auc = 0.5

        return {
            "loss": total_loss / total_samples,
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "auc": float(auc),
        }

    def train_round(self, round_num: int,
                    active_clients: Optional[List[int]] = None,
                    quality_weights: Optional[Dict[int, float]] = None) -> RoundResult:
        """Execute one federated learning round.

        Args:
            round_num: Current round number.
            active_clients: If provided, only train these client IDs.
                Clients not in this list are skipped (e.g., budget exhausted).
                If None, all clients participate (backward compatible).
            quality_weights: Optional {client_id: quality_weight} from Data Quality
                Framework (EHDS Art. 69). Passed to _aggregate().
        """
        start_time = time.time()

        # Determine which clients to train
        clients_to_train = (active_clients if active_clients is not None
                            else list(range(self.num_clients)))

        # Progress callback for round start
        if self.progress_callback:
            self.progress_callback(
                "round_start",
                round_num=round_num + 1
            )

        client_results = []

        # SCAFFOLD: save old client controls before training (for delta computation)
        if self.algorithm == "SCAFFOLD":
            self._old_client_controls = {
                cid: {name: val.clone() for name, val in self.client_controls[cid].items()}
                for cid in clients_to_train
            }

        # Train each client
        for client_id in clients_to_train:
            if self.progress_callback:
                self.progress_callback(
                    "client_start",
                    client_id=client_id,
                    total_clients=len(clients_to_train)
                )

            result = self._train_client(client_id, round_num)
            client_results.append(result)

            if self.progress_callback:
                self.progress_callback(
                    "client_end",
                    client_id=client_id,
                    loss=result.train_loss,
                    acc=result.train_acc
                )

        # Aggregate (weights auto-renormalize with fewer clients)
        self._aggregate(
            client_results,
            noise_scale_override=getattr(self, '_noise_scale_override', None),
            quality_weights=quality_weights,
        )

        # Evaluate with all metrics (personalized models for Per-FedAvg/Ditto)
        if self.algorithm in ("Per-FedAvg", "Ditto") and hasattr(self, 'personalized_models'):
            metrics = self._evaluate_personalized()
        else:
            metrics = self._evaluate()

        elapsed = time.time() - start_time

        # Collect Byzantine defense results if available
        byz_selected = None
        byz_rejected = None
        byz_trust = None
        if self._last_byzantine_result is not None:
            byz_selected = self._last_byzantine_result.selected_clients
            byz_rejected = self._last_byzantine_result.rejected_clients
            byz_trust = self._last_byzantine_result.trust_scores
            self._last_byzantine_result = None

        round_result = RoundResult(
            round_num=round_num,
            global_loss=metrics["loss"],
            global_acc=metrics["accuracy"],
            global_f1=metrics["f1"],
            global_precision=metrics["precision"],
            global_recall=metrics["recall"],
            global_auc=metrics["auc"],
            client_results=client_results,
            time_seconds=elapsed,
            byzantine_selected=byz_selected,
            byzantine_rejected=byz_rejected,
            byzantine_trust_scores=byz_trust,
        )

        self.history.append(round_result)

        if self.progress_callback:
            self.progress_callback(
                "round_end",
                round_num=round_num + 1,
                loss=metrics["loss"],
                acc=metrics["accuracy"],
                f1=metrics["f1"],
                precision=metrics["precision"],
                recall=metrics["recall"],
                auc=metrics["auc"],
                time=elapsed
            )

        return round_result

    def train_clients(self, round_num: int,
                      active_clients: Optional[List[int]] = None) -> List[ClientResult]:
        """Train all active clients without aggregating.

        Used by MyHealth@EU hierarchical aggregation where the caller
        handles 2-level aggregation externally.

        Returns:
            List of ClientResult with model updates (not aggregated).
        """
        clients_to_train = (active_clients if active_clients is not None
                            else list(range(self.num_clients)))

        # SCAFFOLD: save old client controls before training
        if self.algorithm == "SCAFFOLD":
            self._old_client_controls = {
                cid: {name: val.clone() for name, val in self.client_controls[cid].items()}
                for cid in clients_to_train
            }

        client_results = []
        for client_id in clients_to_train:
            result = self._train_client(client_id, round_num)
            client_results.append(result)

        return client_results

    def get_client_data_stats(self) -> Dict[int, Dict]:
        """Get statistics about client data distribution."""
        stats = {}
        for client_id, (X, y) in self.client_data.items():
            unique, counts = np.unique(y, return_counts=True)
            stats[client_id] = {
                "num_samples": len(y),
                "label_distribution": dict(zip(unique.tolist(), counts.tolist())),
                "class_balance": counts.min() / counts.max() if len(counts) > 1 else 1.0
            }
        return stats

    def save_checkpoint(self, path: str) -> None:
        """Save training checkpoint for resumption.

        Persists global model, training history, and all algorithm-specific
        state (SCAFFOLD controls, FedAdam momentum/velocity, personalized
        models, FedDyn corrections, FedNova client steps).

        Args:
            path: File path for the checkpoint (.pt).
        """
        history_serialized = []
        for rr in self.history:
            client_summaries = [
                {
                    "client_id": cr.client_id,
                    "num_samples": cr.num_samples,
                    "train_loss": cr.train_loss,
                    "train_acc": cr.train_acc,
                    "epochs_completed": cr.epochs_completed,
                }
                for cr in rr.client_results
            ]
            history_serialized.append({
                "round_num": rr.round_num,
                "global_loss": rr.global_loss,
                "global_acc": rr.global_acc,
                "global_f1": rr.global_f1,
                "global_precision": rr.global_precision,
                "global_recall": rr.global_recall,
                "global_auc": rr.global_auc,
                "client_results": client_summaries,
                "time_seconds": rr.time_seconds,
            })

        checkpoint = {
            "global_model_state": self.global_model.state_dict(),
            "history": history_serialized,
            "algorithm": self.algorithm,
            "num_clients": self.num_clients,
            "client_steps": self.client_steps,
        }

        # Algorithm-specific state
        if self.algorithm == "SCAFFOLD":
            checkpoint["server_control"] = self.server_control
            checkpoint["client_controls"] = self.client_controls

        if self.algorithm in ["FedAdam", "FedYogi", "FedAdagrad"]:
            checkpoint["server_momentum"] = self.server_momentum
            checkpoint["server_velocity"] = self.server_velocity

        if self.algorithm in ["Per-FedAvg", "Ditto"]:
            checkpoint["personalized_models"] = {
                i: model.state_dict()
                for i, model in self.personalized_models.items()
            }

        if self.algorithm == "FedDyn":
            checkpoint["server_h"] = self.server_h
            checkpoint["client_grad_corrections"] = self.client_grad_corrections

        if self.algorithm == "FedLESAM":
            checkpoint["prev_global_params"] = self.prev_global_params

        if self.algorithm == "HPFL":
            checkpoint["client_classifiers"] = self.client_classifiers
            checkpoint["hpfl_classifier_names"] = list(self._hpfl_classifier_names)

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str) -> int:
        """Load training checkpoint and restore all state.

        Args:
            path: File path of the checkpoint (.pt).

        Returns:
            The next round number to resume from (= number of completed rounds).
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        if checkpoint["algorithm"] != self.algorithm:
            raise ValueError(
                f"Checkpoint algorithm '{checkpoint['algorithm']}' does not match "
                f"trainer algorithm '{self.algorithm}'"
            )

        self.global_model.load_state_dict(checkpoint["global_model_state"])

        # Restore history
        self.history = []
        for rr_dict in checkpoint["history"]:
            client_results = [
                ClientResult(
                    client_id=cr["client_id"],
                    model_update={},
                    num_samples=cr["num_samples"],
                    train_loss=cr["train_loss"],
                    train_acc=cr["train_acc"],
                    epochs_completed=cr["epochs_completed"],
                )
                for cr in rr_dict["client_results"]
            ]
            self.history.append(RoundResult(
                round_num=rr_dict["round_num"],
                global_loss=rr_dict["global_loss"],
                global_acc=rr_dict["global_acc"],
                global_f1=rr_dict["global_f1"],
                global_precision=rr_dict["global_precision"],
                global_recall=rr_dict["global_recall"],
                global_auc=rr_dict["global_auc"],
                client_results=client_results,
                time_seconds=rr_dict["time_seconds"],
            ))

        # Algorithm-specific state
        if self.algorithm == "SCAFFOLD" and "server_control" in checkpoint:
            self.server_control = {
                k: v.to(self.device) for k, v in checkpoint["server_control"].items()
            }
            self.client_controls = {
                cid: {k: v.to(self.device) for k, v in controls.items()}
                for cid, controls in checkpoint["client_controls"].items()
            }

        if self.algorithm in ["FedAdam", "FedYogi", "FedAdagrad"]:
            if "server_momentum" in checkpoint:
                self.server_momentum = {
                    k: v.to(self.device) for k, v in checkpoint["server_momentum"].items()
                }
            if "server_velocity" in checkpoint:
                self.server_velocity = {
                    k: v.to(self.device) for k, v in checkpoint["server_velocity"].items()
                }

        if self.algorithm in ["Per-FedAvg", "Ditto"] and "personalized_models" in checkpoint:
            for i, state_dict in checkpoint["personalized_models"].items():
                self.personalized_models[i].load_state_dict(state_dict)

        if self.algorithm == "FedDyn":
            if "server_h" in checkpoint:
                self.server_h = {
                    k: v.to(self.device) for k, v in checkpoint["server_h"].items()
                }
            if "client_grad_corrections" in checkpoint:
                self.client_grad_corrections = {
                    cid: {k: v.to(self.device) for k, v in corr.items()}
                    for cid, corr in checkpoint["client_grad_corrections"].items()
                }

        if self.algorithm == "FedLESAM" and "prev_global_params" in checkpoint:
            self.prev_global_params = {
                k: v.to(self.device) for k, v in checkpoint["prev_global_params"].items()
            }

        if self.algorithm == "HPFL" and "client_classifiers" in checkpoint:
            self.client_classifiers = {
                cid: {k: v.to(self.device) for k, v in cls.items()}
                for cid, cls in checkpoint["client_classifiers"].items()
            }

        if "client_steps" in checkpoint:
            self.client_steps = checkpoint["client_steps"]

        return len(self.history)

"""
Image Federated Learning trainer for medical imaging datasets.
"""

import numpy as np
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from copy import deepcopy
import time
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from terminal.training.data_generation import _detect_device
from terminal.training.models import HealthcareCNN, HealthcareResNet, load_image_dataset
from terminal.training.federated import ClientResult, RoundResult


class ImageFederatedTrainer:
    """
    Federated Learning trainer for medical image classification.

    Uses HealthcareCNN model and supports loading image datasets from disk.
    Implements FedAvg, FedProx for image FL experiments.
    """

    def __init__(
        self,
        data_dir: str,
        num_clients: int = 5,
        algorithm: str = "FedAvg",
        local_epochs: int = 3,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        is_iid: bool = False,
        alpha: float = 0.5,
        mu: float = 0.1,
        dp_enabled: bool = False,
        dp_epsilon: float = 10.0,
        dp_clip_norm: float = 1.0,
        seed: int = 42,
        device: str = None,
        img_size: int = 128,
        progress_callback: Optional[Callable] = None,
        # Server optimizer params (FedAdam, FedYogi, FedAdagrad)
        server_lr: float = 0.1,
        beta1: float = 0.9,
        beta2: float = 0.99,
        tau: float = 1e-3,
        # Model selection
        model_type: str = "resnet18",
        freeze_backbone: bool = False,
        use_class_weights: bool = True,
        # Byzantine defense
        byzantine_config=None,
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
            "FedAdagrad", "FedNova", "FedDyn", "Per-FedAvg", "Ditto"
        ]
        if algorithm not in SUPPORTED_ALGORITHMS:
            raise ValueError(
                f"Unknown algorithm '{algorithm}'. "
                f"Supported: {SUPPORTED_ALGORITHMS}"
            )

        self.algorithm = algorithm
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.mu = mu
        self.dp_enabled = dp_enabled
        self.dp_epsilon = dp_epsilon
        self.dp_clip_norm = dp_clip_norm
        self.seed = seed
        self.progress_callback = progress_callback
        self.model_type = model_type

        # Server optimizer hyperparameters
        self.server_lr = server_lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.tau = tau
        self.num_rounds = None  # Set externally for cosine LR decay

        # Auto-detect device (CUDA > MPS > CPU)
        self.device = _detect_device(device)

        # Auto-adjust img_size and LR for ResNet
        if model_type == "resnet18" and img_size < 224:
            img_size = 224
        if model_type == "resnet18" and learning_rate == 0.001:
            learning_rate = 0.0005
        self.learning_rate = learning_rate
        self.img_size = img_size

        torch.manual_seed(seed)
        np.random.seed(seed)

        # Load image dataset with train/test split
        self.client_data, self.client_test_data, self.class_names, self.num_classes = load_image_dataset(
            data_dir=data_dir,
            num_clients=num_clients,
            is_iid=is_iid,
            alpha=alpha,
            img_size=img_size,
            seed=seed,
        )
        self.num_clients = len(self.client_data)

        # Class-weighted loss for imbalanced datasets
        self.class_weights = None
        if use_class_weights:
            all_labels = np.concatenate([y for _, y in self.client_data.values()])
            counts = np.bincount(all_labels, minlength=self.num_classes)
            if counts.min() > 0 and counts.max() / counts.min() > 1.5:
                weights = len(all_labels) / (self.num_classes * counts + 1e-8)
                weights = weights / weights.mean()
                self.class_weights = torch.FloatTensor(weights).to(self.device)
                print(f"Class weights: {dict(zip(self.class_names.values(), [f'{w:.2f}' for w in weights]))}")

        # Build augmentation pipeline
        self.augmentation = self._build_augmentation(img_size)

        # Initialize model
        if model_type == "resnet18":
            self.global_model = HealthcareResNet(
                num_classes=self.num_classes,
                pretrained=True,
                freeze_backbone=freeze_backbone,
            ).to(self.device)
            model_name = "HealthcareResNet (ResNet18)"
        else:
            self.global_model = HealthcareCNN(num_classes=self.num_classes).to(self.device)
            model_name = "HealthcareCNN"

        print(f"\nModel: {model_name} ({self.num_classes} classes)")
        print(f"Device: {self.device}")
        total_params = sum(p.numel() for p in self.global_model.parameters())
        trainable_params = sum(p.numel() for p in self.global_model.parameters() if p.requires_grad)
        print(f"Parameters: {total_params:,} ({trainable_params:,} trainable)")

        # SCAFFOLD control variates
        if algorithm == "SCAFFOLD":
            self.server_control = {
                name: torch.zeros_like(param)
                for name, param in self.global_model.named_parameters()
            }
            self.client_controls = {
                i: {name: torch.zeros_like(param)
                    for name, param in self.global_model.named_parameters()}
                for i in range(self.num_clients)
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
                for i in range(self.num_clients)
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
                for i in range(self.num_clients)
            }

        # FedNova: track local steps
        self.client_steps = {i: 0 for i in range(self.num_clients)}

        # Byzantine defense (lazy-initialized)
        self._byzantine_manager = None
        self._last_byzantine_result = None
        if byzantine_config is not None:
            from core.byzantine_resilience import ByzantineDefenseManager
            self._byzantine_manager = ByzantineDefenseManager(byzantine_config)

        # History
        self.history = []

    def _get_client_dataloader(self, client_id: int) -> DataLoader:
        """Create DataLoader for client with data augmentation."""
        X, y = self.client_data[client_id]
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y)

        # Apply augmentation on CPU before moving to device
        augmented = self._augment_batch(X_tensor)
        augmented = augmented.to(self.device)
        y_tensor = y_tensor.to(self.device)

        dataset = TensorDataset(augmented, y_tensor)

        # Optimized DataLoader settings per device
        dl_kwargs = {"batch_size": self.batch_size, "shuffle": True}
        if self.device.type == "cuda":
            dl_kwargs.update(num_workers=2, pin_memory=True, persistent_workers=True)
        # MPS and CPU: num_workers=0 (fork issues on macOS)

        return DataLoader(dataset, **dl_kwargs)

    @staticmethod
    def _build_augmentation(img_size: int):
        """Build torchvision augmentation pipeline."""
        import torchvision.transforms as T
        return T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=15),
            T.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            T.ColorJitter(brightness=0.15, contrast=0.15),
        ])

    def _augment_batch(self, images: torch.Tensor) -> torch.Tensor:
        """Apply augmentation pipeline to batch of images (N, C, H, W)."""
        if self.augmentation is None:
            return images
        augmented = []
        for img in images:
            augmented.append(self.augmentation(img))
        return torch.stack(augmented)

    def _get_round_lr(self, round_num: int) -> float:
        """Cosine annealing learning rate decay based on current round."""
        if self.num_rounds is None or self.num_rounds <= 1:
            return self.learning_rate
        import math
        # Cosine decay from lr to lr * 0.1
        progress = round_num / max(self.num_rounds - 1, 1)
        return self.learning_rate * (0.1 + 0.9 * (1 + math.cos(math.pi * progress)) / 2)

    def _train_client(self, client_id: int, round_num: int) -> ClientResult:
        """Train one client locally. Supports all 9 FL algorithms."""
        local_model = deepcopy(self.global_model)
        local_model = local_model.to(self.device)
        local_model.train()

        current_lr = self._get_round_lr(round_num)
        optimizer = torch.optim.Adam(local_model.parameters(), lr=current_lr,
                                     weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss(weight=self.class_weights)

        dataloader = self._get_client_dataloader(client_id)

        # Store initial params for FedProx, Ditto, and FedDyn
        if self.algorithm in ["FedProx", "Ditto", "FedDyn"]:
            global_params = {
                name: param.clone().detach()
                for name, param in self.global_model.named_parameters()
            }

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
            epoch_pbar = tqdm(
                dataloader,
                desc=f"    Epoch {epoch+1}/{self.local_epochs}",
                leave=False,
                ncols=80,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
            )
            for batch_X, batch_y in epoch_pbar:
                optimizer.zero_grad()

                outputs = local_model(batch_X)
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

                # SCAFFOLD: apply control variate correction after backward
                loss.backward()

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

                # Update progress bar with current loss
                epoch_pbar.set_postfix(loss=f"{loss.item():.4f}")

                # Gradient clipping (always, for stability)
                clip_norm = self.dp_clip_norm if self.dp_enabled else 1.0
                torch.nn.utils.clip_grad_norm_(local_model.parameters(), clip_norm)

                optimizer.step()

                total_loss += loss.item() * len(batch_y)
                preds = outputs.argmax(dim=1)
                total_correct += (preds == batch_y).sum().item()
                total_samples += len(batch_y)
                steps += 1

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
                new_ci = c_i - c + (init_param - local_param) / (K * current_lr)
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
        """Train personalized model for Ditto algorithm (image version)."""
        pers_model = self.personalized_models[client_id]
        pers_model.train()
        optimizer = torch.optim.Adam(pers_model.parameters(), lr=self.learning_rate,
                                     weight_decay=1e-5)

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
        """Fine-tune personalized model for Per-FedAvg algorithm (image version)."""
        pers_model = self.personalized_models[client_id]
        pers_model.load_state_dict(self.global_model.state_dict())
        pers_model.train()

        fine_tune_lr = self.learning_rate * 0.1
        optimizer = torch.optim.Adam(pers_model.parameters(), lr=fine_tune_lr,
                                     weight_decay=1e-5)

        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = pers_model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    def _aggregate_byzantine(self, client_results: List[ClientResult]) -> None:
        """Aggregate using Byzantine-resilient method via ByzantineDefenseManager."""
        from terminal.training.byzantine_bridge import (
            client_results_to_gradients,
            aggregation_result_to_tensors,
        )

        gradients = client_results_to_gradients(client_results)
        result = self._byzantine_manager.aggregate(gradients)
        self._last_byzantine_result = result

        robust_update = aggregation_result_to_tensors(result, self.device)
        for name, param in self.global_model.named_parameters():
            if name in robust_update:
                param.data += robust_update[name]

    def _aggregate(self, client_results: List[ClientResult],
                   noise_scale_override: Optional[float] = None,
                   quality_weights: Optional[Dict[int, float]] = None) -> None:
        """Aggregate client updates. Supports all 10 FL algorithms.

        Args:
            quality_weights: Optional {client_id: quality_weight} from Data Quality
                Framework. If provided, weights are multiplied by quality scores.
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
            # FedAdam with bias correction (Kingma & Ba, 2015)
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
            # FedYogi with bias correction (Kingma & Ba, 2015)
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
            for name, param in self.global_model.named_parameters():
                delta = torch.zeros_like(param)
                for cr in client_results:
                    delta += cr.model_update[name] * cw[cr.client_id]
                self.server_velocity[name] = self.server_velocity[name] + (delta ** 2)
                param.data += self.server_lr * delta / (
                    torch.sqrt(self.server_velocity[name]) + self.tau
                )

        elif self.algorithm == "FedDyn":
            for name, param in self.global_model.named_parameters():
                weighted_update = torch.zeros_like(param)
                for cr in client_results:
                    weighted_update += cr.model_update[name] * cw[cr.client_id]

                self.server_h[name] -= self.mu * weighted_update
                param.data += weighted_update - (1.0 / self.mu) * self.server_h[name]

        else:
            # FedAvg, FedProx, SCAFFOLD, Per-FedAvg, Ditto: weighted average
            for name, param in self.global_model.named_parameters():
                weighted_update = torch.zeros_like(param)
                for cr in client_results:
                    weighted_update += cr.model_update[name] * cw[cr.client_id]
                param.data += weighted_update

        # SCAFFOLD: update server control variate using proper delta
        if self.algorithm == "SCAFFOLD":
            n = len(client_results)
            for name in self.server_control:
                delta_c = torch.zeros_like(self.server_control[name])
                for cr in client_results:
                    delta_c += (
                        self.client_controls[cr.client_id][name]
                        - self._old_client_controls[cr.client_id][name]
                    )
                self.server_control[name] += delta_c / n

        # DP noise
        if self.dp_enabled:
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
        all_probs = []  # Full probability matrix for AUC

        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for client_id in range(self.num_clients):
                X, y = self.client_test_data[client_id]

                # Process in batches for memory efficiency
                for i in range(0, len(y), self.batch_size):
                    X_batch = torch.FloatTensor(X[i:i+self.batch_size]).to(self.device)
                    y_batch = torch.LongTensor(y[i:i+self.batch_size]).to(self.device)

                    outputs = self.global_model(X_batch)
                    loss = criterion(outputs, y_batch)

                    total_loss += loss.item() * len(y_batch)
                    preds = outputs.argmax(dim=1)
                    probs = torch.softmax(outputs, dim=1)

                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(y_batch.cpu().numpy())
                    all_probs.extend(probs.cpu().numpy())
                    total_samples += len(y_batch)

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        # Calculate metrics
        accuracy = (all_preds == all_labels).mean()

        # Macro-averaged metrics for multi-class
        unique_classes = np.unique(all_labels)
        precisions, recalls, f1s = [], [], []

        for cls in unique_classes:
            tp = ((all_preds == cls) & (all_labels == cls)).sum()
            fp = ((all_preds == cls) & (all_labels != cls)).sum()
            fn = ((all_preds != cls) & (all_labels == cls)).sum()

            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

            precisions.append(p)
            recalls.append(r)
            f1s.append(f)

        precision = np.mean(precisions)
        recall = np.mean(recalls)
        f1 = np.mean(f1s)

        # AUC-ROC with sklearn (proper multi-class OvR)
        auc = 0.5
        try:
            from sklearn.metrics import roc_auc_score
            if len(unique_classes) == 2:
                # Binary: use probability of positive class
                auc = roc_auc_score(all_labels, all_probs[:, 1])
            elif len(unique_classes) > 2 and all_probs.shape[1] >= len(unique_classes):
                auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
        except Exception:
            # Fallback: use accuracy as proxy
            auc = float(accuracy)

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

                for i in range(0, len(y), self.batch_size):
                    X_batch = torch.FloatTensor(X[i:i+self.batch_size]).to(self.device)
                    y_batch = torch.LongTensor(y[i:i+self.batch_size]).to(self.device)

                    outputs = pers_model(X_batch)
                    loss = criterion(outputs, y_batch)
                    total_loss += loss.item() * len(y_batch)
                    preds = outputs.argmax(dim=1)
                    probs = torch.softmax(outputs, dim=1)

                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(y_batch.cpu().numpy())
                    all_probs.extend(probs.cpu().numpy())
                    total_samples += len(y_batch)

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        accuracy = (all_preds == all_labels).mean()
        unique_classes = np.unique(all_labels)
        precisions, recalls, f1s = [], [], []
        for cls in unique_classes:
            tp = ((all_preds == cls) & (all_labels == cls)).sum()
            fp = ((all_preds == cls) & (all_labels != cls)).sum()
            fn = ((all_preds != cls) & (all_labels == cls)).sum()
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            precisions.append(p)
            recalls.append(r)
            f1s.append(f)

        precision = np.mean(precisions)
        recall = np.mean(recalls)
        f1 = np.mean(f1s)

        auc = 0.5
        try:
            from sklearn.metrics import roc_auc_score
            if len(unique_classes) == 2:
                auc = roc_auc_score(all_labels, all_probs[:, 1])
            elif len(unique_classes) > 2 and all_probs.shape[1] >= len(unique_classes):
                auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
        except Exception:
            auc = float(accuracy)

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
                If None, all clients participate (backward compatible).
            quality_weights: Optional {client_id: quality_weight} from Data Quality
                Framework (EHDS Art. 69). Passed to _aggregate().
        """
        start_time = time.time()

        # Determine which clients to train
        clients_to_train = (active_clients if active_clients is not None
                            else list(range(self.num_clients)))

        if self.progress_callback:
            self.progress_callback("round_start", round_num=round_num + 1)

        client_results = []

        # SCAFFOLD: save old client controls before training (for delta computation)
        if self.algorithm == "SCAFFOLD":
            self._old_client_controls = {
                cid: {name: val.clone() for name, val in self.client_controls[cid].items()}
                for cid in clients_to_train
            }

        # Progress bar for clients
        client_pbar = tqdm(
            clients_to_train,
            desc=f"  Clients",
            leave=False,
            ncols=100,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
        )

        for client_id in client_pbar:
            client_pbar.set_description(f"  Client {client_id+1}/{len(clients_to_train)}")

            if self.progress_callback:
                self.progress_callback(
                    "client_start",
                    client_id=client_id,
                    total_clients=len(clients_to_train)
                )

            result = self._train_client(client_id, round_num)
            client_results.append(result)

            client_pbar.set_postfix(loss=f"{result.train_loss:.4f}", acc=f"{result.train_acc:.2%}")

            if self.progress_callback:
                self.progress_callback(
                    "client_end",
                    client_id=client_id,
                    loss=result.train_loss,
                    acc=result.train_acc
                )

        self._aggregate(
            client_results,
            noise_scale_override=getattr(self, '_noise_scale_override', None),
            quality_weights=quality_weights,
        )

        # Evaluate (personalized models for Per-FedAvg/Ditto, global otherwise)
        if self.algorithm in ("Per-FedAvg", "Ditto") and hasattr(self, 'personalized_models'):
            print("  Evaluating personalized models...", end="\r")
            metrics = self._evaluate_personalized()
        else:
            print("  Evaluating global model...", end="\r")
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
        """Get statistics about client data distribution (train + test)."""
        stats = {}
        for client_id, (X, y) in self.client_data.items():
            unique, counts = np.unique(y, return_counts=True)
            X_test, y_test = self.client_test_data[client_id]
            stats[client_id] = {
                "num_samples": len(y) + len(y_test),
                "num_train": len(y),
                "num_test": len(y_test),
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
            "num_classes": self.num_classes,
            "class_names": self.class_names,
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

        if "client_steps" in checkpoint:
            self.client_steps = checkpoint["client_steps"]

        return len(self.history)


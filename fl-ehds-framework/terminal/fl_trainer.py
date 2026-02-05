"""
Real Federated Learning Trainer with PyTorch.
Provides actual training with neural networks and detailed progress.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from copy import deepcopy
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from terminal.colors import Colors, Style


# =============================================================================
# HEALTHCARE DATASET GENERATOR
# =============================================================================

def generate_healthcare_data(
    num_clients: int,
    samples_per_client: int = 200,
    num_features: int = 10,
    is_iid: bool = False,
    alpha: float = 0.5,
    seed: int = 42
) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """
    Generate synthetic healthcare data for FL experiments.

    Features simulate clinical measurements:
    - Age (normalized)
    - BMI
    - Blood pressure (systolic)
    - Blood glucose
    - Cholesterol
    - Heart rate
    - Respiratory rate
    - Temperature
    - Oxygen saturation
    - Previous conditions count

    Target: Binary classification (disease risk: 0 = low, 1 = high)
    """
    np.random.seed(seed)

    client_data = {}

    # Generate base data
    total_samples = num_clients * samples_per_client

    # Clinical features with realistic correlations
    age = np.random.normal(55, 15, total_samples).clip(18, 90)
    bmi = np.random.normal(26, 5, total_samples).clip(15, 45)
    bp_systolic = 100 + 0.5 * age + 0.3 * bmi + np.random.normal(0, 10, total_samples)
    glucose = 80 + 0.2 * age + 0.5 * bmi + np.random.normal(0, 15, total_samples)
    cholesterol = 150 + 0.3 * age + 0.4 * bmi + np.random.normal(0, 30, total_samples)
    heart_rate = 70 + 0.1 * age + np.random.normal(0, 10, total_samples)
    resp_rate = 14 + np.random.normal(0, 2, total_samples)
    temperature = 36.6 + np.random.normal(0, 0.3, total_samples)
    oxygen_sat = 98 - 0.05 * age + np.random.normal(0, 1, total_samples)
    prev_conditions = np.random.poisson(1.5, total_samples)

    # Stack and normalize
    X = np.column_stack([
        age, bmi, bp_systolic, glucose, cholesterol,
        heart_rate, resp_rate, temperature, oxygen_sat, prev_conditions
    ])

    # Normalize to [0, 1]
    X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-8)

    # Generate labels based on risk factors
    risk_score = (
        0.3 * X[:, 0] +  # age
        0.2 * X[:, 1] +  # bmi
        0.15 * X[:, 2] + # bp
        0.15 * X[:, 3] + # glucose
        0.1 * X[:, 4] +  # cholesterol
        0.1 * X[:, 9]    # prev_conditions
    )
    risk_score += np.random.normal(0, 0.1, total_samples)
    y = (risk_score > np.median(risk_score)).astype(np.int64)

    if is_iid:
        # IID: Random distribution
        indices = np.random.permutation(total_samples)
        for i in range(num_clients):
            start = i * samples_per_client
            end = start + samples_per_client
            client_indices = indices[start:end]
            client_data[i] = (X[client_indices], y[client_indices])
    else:
        # Non-IID: Dirichlet distribution for label skew
        label_indices = {0: np.where(y == 0)[0], 1: np.where(y == 1)[0]}

        # Dirichlet allocation
        proportions = np.random.dirichlet([alpha] * num_clients, 2)

        for i in range(num_clients):
            client_indices = []
            for label in [0, 1]:
                n_samples = int(proportions[label, i] * len(label_indices[label]))
                n_samples = max(10, min(n_samples, len(label_indices[label])))
                chosen = np.random.choice(
                    label_indices[label],
                    size=min(n_samples, samples_per_client // 2),
                    replace=False
                )
                client_indices.extend(chosen)

            client_indices = np.array(client_indices)
            np.random.shuffle(client_indices)
            client_data[i] = (X[client_indices], y[client_indices])

    return client_data


# =============================================================================
# NEURAL NETWORK MODEL
# =============================================================================

class HealthcareMLP(nn.Module):
    """MLP for healthcare risk prediction."""

    def __init__(self, input_dim: int = 10, hidden_dims: List[int] = [64, 32], num_classes: int = 2):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# =============================================================================
# FEDERATED LEARNING TRAINER
# =============================================================================

@dataclass
class ClientResult:
    """Result from client local training."""
    client_id: int
    model_update: Dict[str, torch.Tensor]
    num_samples: int
    train_loss: float
    train_acc: float
    epochs_completed: int


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
    ):
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
        self.device = torch.device(device)
        self.progress_callback = progress_callback

        # Server optimizer hyperparameters
        self.server_lr = server_lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.tau = tau

        torch.manual_seed(seed)
        np.random.seed(seed)

        # Generate data
        self.client_data = generate_healthcare_data(
            num_clients=num_clients,
            samples_per_client=samples_per_client,
            is_iid=is_iid,
            alpha=alpha,
            seed=seed
        )

        # Initialize global model
        self.global_model = HealthcareMLP().to(self.device)

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

        # FedNova: track local steps
        self.client_steps = {i: 0 for i in range(num_clients)}

        # History
        self.history = []

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
        criterion = nn.CrossEntropyLoss()

        dataloader = self._get_client_dataloader(client_id)

        # Store initial params for FedProx and Ditto
        if self.algorithm in ["FedProx", "Ditto"]:
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
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_samples = 0

            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()

                outputs = local_model(batch_X)
                loss = criterion(outputs, batch_y)

                # FedProx: add proximal term
                if self.algorithm == "FedProx":
                    prox_term = 0.0
                    for name, param in local_model.named_parameters():
                        prox_term += torch.sum((param - global_params[name]) ** 2)
                    loss += (self.mu / 2) * prox_term

                # SCAFFOLD: apply control variate correction
                if self.algorithm == "SCAFFOLD":
                    for name, param in local_model.named_parameters():
                        if param.grad is not None:
                            correction = self.client_controls[client_id][name] - self.server_control[name]
                            param.grad.data += correction

                loss.backward()

                # DP: clip gradients
                if self.dp_enabled:
                    torch.nn.utils.clip_grad_norm_(
                        local_model.parameters(),
                        self.dp_clip_norm
                    )

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
                self.client_controls[client_id][name] = (
                    c_i - c + (init_param - local_param) / (K * self.learning_rate)
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

    def _aggregate(self, client_results: List[ClientResult]) -> None:
        """Aggregate client updates to global model."""
        # First compute weighted pseudo-gradient (delta)
        total_samples = sum(cr.num_samples for cr in client_results)

        if self.algorithm == "FedNova":
            # Normalized averaging based on local steps
            total_steps = sum(self.client_steps[cr.client_id] for cr in client_results)

            for name, param in self.global_model.named_parameters():
                weighted_update = torch.zeros_like(param)
                for cr in client_results:
                    tau_i = self.client_steps[cr.client_id]
                    # Normalize by local steps
                    weighted_update += cr.model_update[name] * (tau_i / total_steps)

                # Apply normalized update
                param.data += weighted_update * (total_steps / len(client_results))

        elif self.algorithm == "FedAdam":
            # FedAdam: Adam optimizer on server side
            for name, param in self.global_model.named_parameters():
                # Compute weighted pseudo-gradient
                delta = torch.zeros_like(param)
                for cr in client_results:
                    weight = cr.num_samples / total_samples
                    delta += cr.model_update[name] * weight

                # Update momentum: m_t = beta1 * m_{t-1} + (1 - beta1) * delta
                self.server_momentum[name] = (
                    self.beta1 * self.server_momentum[name] +
                    (1 - self.beta1) * delta
                )

                # Update velocity: v_t = beta2 * v_{t-1} + (1 - beta2) * delta^2
                self.server_velocity[name] = (
                    self.beta2 * self.server_velocity[name] +
                    (1 - self.beta2) * (delta ** 2)
                )

                # Apply update: global += server_lr * m / (sqrt(v) + tau)
                param.data += self.server_lr * self.server_momentum[name] / (
                    torch.sqrt(self.server_velocity[name]) + self.tau
                )

        elif self.algorithm == "FedYogi":
            # FedYogi: like FedAdam but with different velocity update
            for name, param in self.global_model.named_parameters():
                # Compute weighted pseudo-gradient
                delta = torch.zeros_like(param)
                for cr in client_results:
                    weight = cr.num_samples / total_samples
                    delta += cr.model_update[name] * weight

                # Update momentum: m_t = beta1 * m_{t-1} + (1 - beta1) * delta
                self.server_momentum[name] = (
                    self.beta1 * self.server_momentum[name] +
                    (1 - self.beta1) * delta
                )

                # Update velocity (Yogi style): v_t = v_{t-1} - (1 - beta2) * sign(v - delta^2) * delta^2
                delta_sq = delta ** 2
                sign = torch.sign(self.server_velocity[name] - delta_sq)
                self.server_velocity[name] = (
                    self.server_velocity[name] -
                    (1 - self.beta2) * sign * delta_sq
                )

                # Apply update
                param.data += self.server_lr * self.server_momentum[name] / (
                    torch.sqrt(self.server_velocity[name]) + self.tau
                )

        elif self.algorithm == "FedAdagrad":
            # FedAdagrad: Adagrad on server side (no momentum)
            for name, param in self.global_model.named_parameters():
                # Compute weighted pseudo-gradient
                delta = torch.zeros_like(param)
                for cr in client_results:
                    weight = cr.num_samples / total_samples
                    delta += cr.model_update[name] * weight

                # Update velocity: v_t = v_{t-1} + delta^2
                self.server_velocity[name] = self.server_velocity[name] + (delta ** 2)

                # Apply update: global += server_lr * delta / (sqrt(v) + tau)
                param.data += self.server_lr * delta / (
                    torch.sqrt(self.server_velocity[name]) + self.tau
                )

        else:
            # FedAvg, FedProx, SCAFFOLD, Per-FedAvg, Ditto: weighted average by samples
            for name, param in self.global_model.named_parameters():
                weighted_update = torch.zeros_like(param)
                for cr in client_results:
                    weight = cr.num_samples / total_samples
                    weighted_update += cr.model_update[name] * weight

                param.data += weighted_update

        # SCAFFOLD: update server control variate
        if self.algorithm == "SCAFFOLD":
            n = len(client_results)
            for name in self.server_control:
                delta_c = torch.zeros_like(self.server_control[name])
                for cr in client_results:
                    # c_new - c_old for each client
                    delta_c += self.client_controls[cr.client_id][name]
                self.server_control[name] += delta_c / n

        # DP: add noise to aggregated model
        if self.dp_enabled:
            noise_scale = self.dp_clip_norm / self.dp_epsilon
            for param in self.global_model.parameters():
                noise = torch.randn_like(param) * noise_scale
                param.data += noise

    def _evaluate(self) -> Dict[str, float]:
        """Evaluate global model on all client data with comprehensive metrics."""
        self.global_model.eval()

        total_loss = 0.0
        total_samples = 0
        all_preds = []
        all_labels = []
        all_probs = []

        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for client_id in range(self.num_clients):
                X, y = self.client_data[client_id]
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

    def train_round(self, round_num: int) -> RoundResult:
        """Execute one federated learning round."""
        start_time = time.time()

        # Progress callback for round start
        if self.progress_callback:
            self.progress_callback(
                "round_start",
                round_num=round_num + 1
            )

        client_results = []

        # Train each client
        for client_id in range(self.num_clients):
            if self.progress_callback:
                self.progress_callback(
                    "client_start",
                    client_id=client_id,
                    total_clients=self.num_clients
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

        # Aggregate
        self._aggregate(client_results)

        # Evaluate with all metrics
        metrics = self._evaluate()

        elapsed = time.time() - start_time

        round_result = RoundResult(
            round_num=round_num,
            global_loss=metrics["loss"],
            global_acc=metrics["accuracy"],
            global_f1=metrics["f1"],
            global_precision=metrics["precision"],
            global_recall=metrics["recall"],
            global_auc=metrics["auc"],
            client_results=client_results,
            time_seconds=elapsed
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

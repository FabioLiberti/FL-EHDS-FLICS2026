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


class HealthcareCNN(nn.Module):
    """
    CNN for medical image classification.

    Designed for clinical imaging datasets:
    - Brain Tumor MRI (4 classes)
    - Diabetic Retinopathy (5 stages)
    - Chest X-ray (2 classes: pneumonia/normal)
    - Skin Cancer (2+ classes)

    Input: (batch, 3, 224, 224) - RGB images resized to 224x224
    """

    def __init__(
        self,
        num_classes: int = 2,
        in_channels: int = 3,
        dropout: float = 0.3
    ):
        super().__init__()

        # Feature extraction layers
        self.features = nn.Sequential(
            # Block 1: 224 -> 112
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout * 0.5),

            # Block 2: 112 -> 56
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout * 0.5),

            # Block 3: 56 -> 28
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout),

            # Block 4: 28 -> 14
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout),

            # Block 5: 14 -> 7
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((7, 7)),
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# =============================================================================
# IMAGE DATASET LOADER
# =============================================================================

def load_image_dataset(
    data_dir: str,
    num_clients: int = 5,
    is_iid: bool = False,
    alpha: float = 0.5,
    img_size: int = 224,
    seed: int = 42,
    val_split: float = 0.2,
) -> Tuple[Dict[int, Tuple[np.ndarray, np.ndarray]], Dict, int]:
    """
    Load image dataset from directory and partition for FL.

    Expects directory structure:
        data_dir/
            class_0/
                img1.jpg
                img2.jpg
            class_1/
                img3.jpg
                ...

    Args:
        data_dir: Path to dataset directory
        num_clients: Number of FL clients
        is_iid: If True, distribute data IID across clients
        alpha: Dirichlet parameter for non-IID distribution
        img_size: Target image size (default 224x224)
        seed: Random seed
        val_split: Validation split ratio

    Returns:
        client_data: Dict mapping client_id -> (X, y) arrays
        class_names: Dict mapping class_id -> class_name
        num_classes: Number of classes
    """
    from pathlib import Path
    from PIL import Image

    np.random.seed(seed)

    data_path = Path(data_dir)

    # Detect classes from subdirectories
    class_dirs = sorted([d for d in data_path.iterdir() if d.is_dir()])
    class_names = {i: d.name for i, d in enumerate(class_dirs)}
    num_classes = len(class_names)

    print(f"Loading dataset from: {data_dir}")
    print(f"Found {num_classes} classes: {list(class_names.values())}")

    # Load all images
    all_images = []
    all_labels = []

    for class_id, class_dir in enumerate(class_dirs):
        image_files = list(class_dir.glob("*.jpg")) + \
                      list(class_dir.glob("*.jpeg")) + \
                      list(class_dir.glob("*.png"))

        print(f"  Class {class_id} ({class_dir.name}): {len(image_files)} images")

        for img_path in image_files:
            try:
                img = Image.open(img_path).convert("RGB")
                img = img.resize((img_size, img_size))
                img_array = np.array(img, dtype=np.float32) / 255.0
                # Convert to (C, H, W) format
                img_array = img_array.transpose(2, 0, 1)

                all_images.append(img_array)
                all_labels.append(class_id)
            except Exception as e:
                print(f"    Warning: Could not load {img_path}: {e}")

    X = np.array(all_images)
    y = np.array(all_labels, dtype=np.int64)

    print(f"Total: {len(y)} images loaded")

    # Shuffle data
    indices = np.random.permutation(len(y))
    X = X[indices]
    y = y[indices]

    # Partition data for FL
    client_data = {}

    if is_iid:
        # IID: Equal random distribution
        samples_per_client = len(y) // num_clients
        for i in range(num_clients):
            start = i * samples_per_client
            end = start + samples_per_client if i < num_clients - 1 else len(y)
            client_data[i] = (X[start:end], y[start:end])
    else:
        # Non-IID: Dirichlet distribution for label skew
        label_indices = {c: np.where(y == c)[0] for c in range(num_classes)}

        # Allocate proportions using Dirichlet
        proportions = np.random.dirichlet([alpha] * num_clients, num_classes)

        client_indices = {i: [] for i in range(num_clients)}

        for class_id in range(num_classes):
            class_idx = label_indices[class_id]
            n_class = len(class_idx)

            # Distribute this class across clients
            splits = (proportions[class_id] * n_class).astype(int)
            splits[-1] = n_class - splits[:-1].sum()  # Ensure all samples used

            current = 0
            for client_id, count in enumerate(splits):
                client_indices[client_id].extend(
                    class_idx[current:current + count].tolist()
                )
                current += count

        for client_id, indices in client_indices.items():
            np.random.shuffle(indices)
            client_data[client_id] = (X[indices], y[indices])

    # Print distribution
    print("\nClient data distribution:")
    for client_id, (X_c, y_c) in client_data.items():
        unique, counts = np.unique(y_c, return_counts=True)
        dist = dict(zip(unique.tolist(), counts.tolist()))
        print(f"  Client {client_id}: {len(y_c)} samples, distribution: {dist}")

    return client_data, class_names, num_classes


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


# =============================================================================
# IMAGE FEDERATED LEARNING TRAINER
# =============================================================================

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
        img_size: int = 224,
        progress_callback: Optional[Callable] = None,
    ):
        self.algorithm = algorithm
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.mu = mu
        self.dp_enabled = dp_enabled
        self.dp_epsilon = dp_epsilon
        self.dp_clip_norm = dp_clip_norm
        self.seed = seed
        self.progress_callback = progress_callback

        # Auto-detect device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        torch.manual_seed(seed)
        np.random.seed(seed)

        # Load image dataset
        self.client_data, self.class_names, self.num_classes = load_image_dataset(
            data_dir=data_dir,
            num_clients=num_clients,
            is_iid=is_iid,
            alpha=alpha,
            img_size=img_size,
            seed=seed,
        )
        self.num_clients = len(self.client_data)

        # Initialize CNN model
        self.global_model = HealthcareCNN(num_classes=self.num_classes).to(self.device)

        print(f"\nModel: HealthcareCNN ({self.num_classes} classes)")
        print(f"Device: {self.device}")
        total_params = sum(p.numel() for p in self.global_model.parameters())
        print(f"Parameters: {total_params:,}")

        # History
        self.history = []

    def _get_client_dataloader(self, client_id: int) -> DataLoader:
        """Create DataLoader for client."""
        X, y = self.client_data[client_id]
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)
        dataset = TensorDataset(X_tensor, y_tensor)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def _train_client(self, client_id: int, round_num: int) -> ClientResult:
        """Train one client locally."""
        local_model = deepcopy(self.global_model)
        local_model.train()

        optimizer = torch.optim.Adam(local_model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()

        dataloader = self._get_client_dataloader(client_id)

        # For FedProx
        if self.algorithm == "FedProx":
            global_params = {
                name: param.clone().detach()
                for name, param in self.global_model.named_parameters()
            }

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for epoch in range(self.local_epochs):
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

                loss.backward()

                # DP: clip gradients
                if self.dp_enabled:
                    torch.nn.utils.clip_grad_norm_(
                        local_model.parameters(),
                        self.dp_clip_norm
                    )

                optimizer.step()

                total_loss += loss.item() * len(batch_y)
                preds = outputs.argmax(dim=1)
                total_correct += (preds == batch_y).sum().item()
                total_samples += len(batch_y)

        # Compute model update
        model_update = {}
        for name, param in local_model.named_parameters():
            global_param = dict(self.global_model.named_parameters())[name]
            model_update[name] = param.data - global_param.data

        return ClientResult(
            client_id=client_id,
            model_update=model_update,
            num_samples=total_samples,
            train_loss=total_loss / total_samples,
            train_acc=total_correct / total_samples,
            epochs_completed=self.local_epochs
        )

    def _aggregate(self, client_results: List[ClientResult]) -> None:
        """FedAvg aggregation."""
        total_samples = sum(cr.num_samples for cr in client_results)

        for name, param in self.global_model.named_parameters():
            weighted_update = torch.zeros_like(param)
            for cr in client_results:
                weight = cr.num_samples / total_samples
                weighted_update += cr.model_update[name] * weight
            param.data += weighted_update

        # DP noise
        if self.dp_enabled:
            noise_scale = self.dp_clip_norm / self.dp_epsilon
            for param in self.global_model.parameters():
                noise = torch.randn_like(param) * noise_scale
                param.data += noise

    def _evaluate(self) -> Dict[str, float]:
        """Evaluate global model on all client data."""
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

                # Process in batches for memory efficiency
                for i in range(0, len(y), self.batch_size):
                    X_batch = torch.FloatTensor(X[i:i+self.batch_size]).to(self.device)
                    y_batch = torch.LongTensor(y[i:i+self.batch_size]).to(self.device)

                    outputs = self.global_model(X_batch)
                    loss = criterion(outputs, y_batch)

                    total_loss += loss.item() * len(y_batch)
                    preds = outputs.argmax(dim=1)

                    # For multi-class AUC, use max probability
                    probs = torch.softmax(outputs, dim=1)
                    max_probs, _ = probs.max(dim=1)

                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(y_batch.cpu().numpy())
                    all_probs.extend(max_probs.cpu().numpy())
                    total_samples += len(y_batch)

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

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

        # Simple AUC approximation for multi-class
        auc = accuracy  # Use accuracy as proxy for multi-class

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

        if self.progress_callback:
            self.progress_callback("round_start", round_num=round_num + 1)

        client_results = []

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

        self._aggregate(client_results)
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


# =============================================================================
# CENTRALIZED BASELINE TRAINER
# =============================================================================

@dataclass
class CentralizedResult:
    """Result from one centralized training epoch."""
    epoch: int
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float
    val_f1: float
    val_precision: float
    val_recall: float
    val_auc: float
    time_seconds: float


class CentralizedTrainer:
    """
    Centralized baseline trainer - simulates scenario where all hospital data
    is pooled into a single central server (no federated learning).

    This serves as an upper bound for FL performance comparison.
    """

    def __init__(
        self,
        num_clients: int = 5,
        samples_per_client: int = 200,
        batch_size: int = 32,
        learning_rate: float = 0.01,
        is_iid: bool = False,
        alpha: float = 0.5,
        seed: int = 42,
        device: str = "cpu",
        progress_callback: Optional[Callable] = None,
        val_split: float = 0.2,  # Validation split ratio
    ):
        self.num_clients = num_clients
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.seed = seed
        self.device = torch.device(device)
        self.progress_callback = progress_callback
        self.val_split = val_split

        torch.manual_seed(seed)
        np.random.seed(seed)

        # Generate data using same function as FL
        client_data = generate_healthcare_data(
            num_clients=num_clients,
            samples_per_client=samples_per_client,
            is_iid=is_iid,
            alpha=alpha,
            seed=seed
        )

        # Combine all client data into one dataset
        all_X = []
        all_y = []
        for client_id in range(num_clients):
            X, y = client_data[client_id]
            all_X.append(X)
            all_y.append(y)

        self.X_all = np.vstack(all_X)
        self.y_all = np.concatenate(all_y)

        # Split into train/val
        n_samples = len(self.y_all)
        n_val = int(n_samples * val_split)
        indices = np.random.permutation(n_samples)
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]

        self.X_train = self.X_all[train_indices]
        self.y_train = self.y_all[train_indices]
        self.X_val = self.X_all[val_indices]
        self.y_val = self.y_all[val_indices]

        # Initialize model
        self.model = HealthcareMLP().to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()

        # History
        self.history = []

    def _get_dataloader(self, X: np.ndarray, y: np.ndarray, shuffle: bool = True) -> DataLoader:
        """Create DataLoader from numpy arrays."""
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)
        dataset = TensorDataset(X_tensor, y_tensor)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

    def _evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate model on given data with comprehensive metrics."""
        self.model.eval()

        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)

        with torch.no_grad():
            outputs = self.model(X_tensor)
            loss = self.criterion(outputs, y_tensor)
            preds = outputs.argmax(dim=1)
            probs = torch.softmax(outputs, dim=1)[:, 1]

        preds_np = preds.cpu().numpy()
        labels_np = y_tensor.cpu().numpy()
        probs_np = probs.cpu().numpy()

        # Calculate metrics
        accuracy = (preds_np == labels_np).mean()

        tp = ((preds_np == 1) & (labels_np == 1)).sum()
        fp = ((preds_np == 1) & (labels_np == 0)).sum()
        fn = ((preds_np == 0) & (labels_np == 1)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        # AUC-ROC
        sorted_indices = np.argsort(probs_np)[::-1]
        sorted_labels = labels_np[sorted_indices]
        n_pos = (labels_np == 1).sum()
        n_neg = (labels_np == 0).sum()

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
            "loss": loss.item(),
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "auc": float(auc),
        }

    def train_epoch(self, epoch: int) -> CentralizedResult:
        """Train for one epoch."""
        start_time = time.time()

        if self.progress_callback:
            self.progress_callback("epoch_start", epoch=epoch + 1)

        self.model.train()
        train_loader = self._get_dataloader(self.X_train, self.y_train)

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch_X, batch_y in train_loader:
            self.optimizer.zero_grad()

            outputs = self.model(batch_X)
            loss = self.criterion(outputs, batch_y)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * len(batch_y)
            preds = outputs.argmax(dim=1)
            total_correct += (preds == batch_y).sum().item()
            total_samples += len(batch_y)

        train_loss = total_loss / total_samples
        train_acc = total_correct / total_samples

        # Evaluate on validation set
        val_metrics = self._evaluate(self.X_val, self.y_val)

        elapsed = time.time() - start_time

        result = CentralizedResult(
            epoch=epoch,
            train_loss=train_loss,
            train_acc=train_acc,
            val_loss=val_metrics["loss"],
            val_acc=val_metrics["accuracy"],
            val_f1=val_metrics["f1"],
            val_precision=val_metrics["precision"],
            val_recall=val_metrics["recall"],
            val_auc=val_metrics["auc"],
            time_seconds=elapsed
        )

        self.history.append(result)

        if self.progress_callback:
            self.progress_callback(
                "epoch_end",
                epoch=epoch + 1,
                train_loss=train_loss,
                train_acc=train_acc,
                val_loss=val_metrics["loss"],
                val_acc=val_metrics["accuracy"],
                val_f1=val_metrics["f1"],
                val_auc=val_metrics["auc"],
                time=elapsed
            )

        return result

    def evaluate_on_all_data(self) -> Dict[str, float]:
        """Evaluate model on ALL data (train + val) for fair comparison with FL."""
        return self._evaluate(self.X_all, self.y_all)

    def get_data_stats(self) -> Dict:
        """Get statistics about the centralized dataset."""
        unique, counts = np.unique(self.y_all, return_counts=True)
        return {
            "total_samples": len(self.y_all),
            "train_samples": len(self.y_train),
            "val_samples": len(self.y_val),
            "label_distribution": dict(zip(unique.tolist(), counts.tolist())),
            "class_balance": counts.min() / counts.max() if len(counts) > 1 else 1.0
        }


# =============================================================================
# FL vs CENTRALIZED COMPARISON
# =============================================================================

def run_fl_vs_centralized_comparison(
    num_clients: int = 5,
    samples_per_client: int = 200,
    fl_rounds: int = 30,
    local_epochs: int = 3,
    batch_size: int = 32,
    learning_rate: float = 0.01,
    is_iid: bool = False,
    alpha: float = 0.5,
    seeds: List[int] = [42, 123, 456],
    algorithms: List[str] = ["FedAvg"],
    verbose: bool = True,
) -> Dict:
    """
    Run comparison between FL algorithms and centralized baseline.

    Runs multiple seeds and returns mean ± std for all metrics.
    For fair comparison, centralized training uses same total epochs:
        centralized_epochs = fl_rounds * local_epochs

    Returns dict with results for each algorithm plus 'Centralized' baseline.
    """
    results = {}

    # Total epochs for centralized (to match FL computation)
    centralized_epochs = fl_rounds * local_epochs

    for algo in algorithms + ["Centralized"]:
        results[algo] = {
            "accuracy": [],
            "loss": [],
            "f1": [],
            "precision": [],
            "recall": [],
            "auc": [],
            "time": [],
        }

    for seed_idx, seed in enumerate(seeds):
        if verbose:
            print(f"\n{'='*60}")
            print(f"Seed {seed_idx + 1}/{len(seeds)}: {seed}")
            print('='*60)

        # Run centralized baseline
        if verbose:
            print(f"\n[Centralized] Training for {centralized_epochs} epochs...")

        centralized = CentralizedTrainer(
            num_clients=num_clients,
            samples_per_client=samples_per_client,
            batch_size=batch_size,
            learning_rate=learning_rate,
            is_iid=is_iid,
            alpha=alpha,
            seed=seed,
        )

        start_time = time.time()
        for epoch in range(centralized_epochs):
            centralized.train_epoch(epoch)

        central_time = time.time() - start_time

        # Evaluate on all data for fair comparison
        central_metrics = centralized.evaluate_on_all_data()

        results["Centralized"]["accuracy"].append(central_metrics["accuracy"])
        results["Centralized"]["loss"].append(central_metrics["loss"])
        results["Centralized"]["f1"].append(central_metrics["f1"])
        results["Centralized"]["precision"].append(central_metrics["precision"])
        results["Centralized"]["recall"].append(central_metrics["recall"])
        results["Centralized"]["auc"].append(central_metrics["auc"])
        results["Centralized"]["time"].append(central_time)

        if verbose:
            print(f"  Accuracy: {central_metrics['accuracy']:.4f}, "
                  f"F1: {central_metrics['f1']:.4f}, "
                  f"AUC: {central_metrics['auc']:.4f}")

        # Run FL algorithms
        for algo in algorithms:
            if verbose:
                print(f"\n[{algo}] Training for {fl_rounds} rounds...")

            trainer = FederatedTrainer(
                num_clients=num_clients,
                samples_per_client=samples_per_client,
                algorithm=algo,
                local_epochs=local_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                is_iid=is_iid,
                alpha=alpha,
                seed=seed,
            )

            start_time = time.time()
            for r in range(fl_rounds):
                trainer.train_round(r)

            fl_time = time.time() - start_time

            # Get final metrics
            final = trainer.history[-1]

            results[algo]["accuracy"].append(final.global_acc)
            results[algo]["loss"].append(final.global_loss)
            results[algo]["f1"].append(final.global_f1)
            results[algo]["precision"].append(final.global_precision)
            results[algo]["recall"].append(final.global_recall)
            results[algo]["auc"].append(final.global_auc)
            results[algo]["time"].append(fl_time)

            if verbose:
                print(f"  Accuracy: {final.global_acc:.4f}, "
                      f"F1: {final.global_f1:.4f}, "
                      f"AUC: {final.global_auc:.4f}")

    # Compute mean ± std
    summary = {}
    for algo, metrics in results.items():
        summary[algo] = {}
        for metric_name, values in metrics.items():
            arr = np.array(values)
            summary[algo][metric_name] = {
                "mean": float(arr.mean()),
                "std": float(arr.std()),
                "values": values
            }

    # Compute gap from centralized
    central_acc = summary["Centralized"]["accuracy"]["mean"]
    for algo in algorithms:
        fl_acc = summary[algo]["accuracy"]["mean"]
        gap = central_acc - fl_acc
        gap_pct = (gap / central_acc) * 100 if central_acc > 0 else 0
        summary[algo]["gap_from_centralized"] = {
            "absolute": gap,
            "percentage": gap_pct
        }

    if verbose:
        print("\n" + "="*60)
        print("SUMMARY: FL vs Centralized Comparison")
        print("="*60)
        print(f"\nCentralized Baseline:")
        print(f"  Accuracy: {summary['Centralized']['accuracy']['mean']:.4f} "
              f"± {summary['Centralized']['accuracy']['std']:.4f}")
        print(f"  F1: {summary['Centralized']['f1']['mean']:.4f} "
              f"± {summary['Centralized']['f1']['std']:.4f}")
        print(f"  AUC: {summary['Centralized']['auc']['mean']:.4f} "
              f"± {summary['Centralized']['auc']['std']:.4f}")

        for algo in algorithms:
            print(f"\n{algo}:")
            print(f"  Accuracy: {summary[algo]['accuracy']['mean']:.4f} "
                  f"± {summary[algo]['accuracy']['std']:.4f}")
            print(f"  F1: {summary[algo]['f1']['mean']:.4f} "
                  f"± {summary[algo]['f1']['std']:.4f}")
            print(f"  AUC: {summary[algo]['auc']['mean']:.4f} "
                  f"± {summary[algo]['auc']['std']:.4f}")
            print(f"  Gap from Centralized: {summary[algo]['gap_from_centralized']['absolute']:.4f} "
                  f"({summary[algo]['gap_from_centralized']['percentage']:.2f}%)")

    return summary


def generate_comparison_latex_table(summary: Dict) -> str:
    """Generate LaTeX table comparing FL algorithms with centralized baseline."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Comparison of Federated Learning vs Centralized Training}",
        r"\label{tab:fl-vs-centralized}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Algorithm & Accuracy & F1 & AUC & Gap (\%) \\",
        r"\midrule",
    ]

    # Centralized first
    c = summary["Centralized"]
    lines.append(
        f"Centralized (Upper Bound) & "
        f"${c['accuracy']['mean']:.3f} \\pm {c['accuracy']['std']:.3f}$ & "
        f"${c['f1']['mean']:.3f} \\pm {c['f1']['std']:.3f}$ & "
        f"${c['auc']['mean']:.3f} \\pm {c['auc']['std']:.3f}$ & "
        f"--- \\\\"
    )
    lines.append(r"\midrule")

    # FL algorithms
    for algo, metrics in summary.items():
        if algo == "Centralized":
            continue
        gap = metrics.get("gap_from_centralized", {}).get("percentage", 0)
        lines.append(
            f"{algo} & "
            f"${metrics['accuracy']['mean']:.3f} \\pm {metrics['accuracy']['std']:.3f}$ & "
            f"${metrics['f1']['mean']:.3f} \\pm {metrics['f1']['std']:.3f}$ & "
            f"${metrics['auc']['mean']:.3f} \\pm {metrics['auc']['std']:.3f}$ & "
            f"{gap:.1f} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)

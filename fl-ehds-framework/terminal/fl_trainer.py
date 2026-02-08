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
from tqdm import tqdm

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
    seed: int = 42,
    test_split: float = 0.2
) -> Tuple[Dict[int, Tuple[np.ndarray, np.ndarray]], Dict[int, Tuple[np.ndarray, np.ndarray]]]:
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

    Returns:
        (client_train_data, client_test_data) - each is Dict[int, (X, y)]
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

    # Split each client's data into train/test
    client_train_data = {}
    client_test_data = {}

    for client_id, (X_c, y_c) in client_data.items():
        n = len(y_c)
        n_test = max(1, int(n * test_split))
        n_train = n - n_test

        perm = np.random.permutation(n)
        train_idx = perm[:n_train]
        test_idx = perm[n_train:]

        client_train_data[client_id] = (X_c[train_idx], y_c[train_idx])
        client_test_data[client_id] = (X_c[test_idx], y_c[test_idx])

    return client_train_data, client_test_data


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
    Lightweight CNN for medical image classification in FL settings.

    Designed for clinical imaging datasets with limited data per client.
    Uses GroupNorm (FL-stable) and global average pooling for efficiency.
    ~500K parameters (vs 15M in VGG-style) - suitable for CPU training.

    Input: (batch, 3, H, W) - RGB images (typically 128x128)
    """

    def __init__(
        self,
        num_classes: int = 2,
        in_channels: int = 3,
        dropout: float = 0.3
    ):
        super().__init__()

        # Feature extraction: 4 blocks with GroupNorm (FL-stable)
        self.features = nn.Sequential(
            # Block 1: -> /2
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout * 0.5),

            # Block 2: -> /4
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout * 0.5),

            # Block 3: -> /8
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.GroupNorm(16, 128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout),

            # Block 4: -> global average pool
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.GroupNorm(16, 256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        # Compact classifier (256 features from GAP)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
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
    img_size: int = 128,
    seed: int = 42,
    test_split: float = 0.2,
) -> Tuple[Dict[int, Tuple[np.ndarray, np.ndarray]], Dict[int, Tuple[np.ndarray, np.ndarray]], Dict, int]:
    """
    Load image dataset from directory and partition for FL with train/test split.

    Supports two directory structures:
        Direct:  data_dir/class_0/img1.jpg
        Split:   data_dir/train/class_0/img1.jpg + data_dir/test/class_0/img2.jpg

    Args:
        data_dir: Path to dataset directory
        num_clients: Number of FL clients
        is_iid: If True, distribute data IID across clients
        alpha: Dirichlet parameter for non-IID distribution
        img_size: Target image size (default 128x128)
        seed: Random seed
        test_split: Fraction of data reserved for testing per client

    Returns:
        client_train_data: Dict mapping client_id -> (X_train, y_train)
        client_test_data: Dict mapping client_id -> (X_test, y_test)
        class_names: Dict mapping class_id -> class_name
        num_classes: Number of classes
    """
    from pathlib import Path
    from PIL import Image

    np.random.seed(seed)

    data_path = Path(data_dir)

    # Detect directory structure: direct classes or train/test split
    subdirs = sorted([d.name for d in data_path.iterdir() if d.is_dir()])
    split_names = {"train", "training", "test", "testing", "val", "validation", "data"}
    has_split_structure = any(s.lower() in split_names for s in subdirs)

    if has_split_structure:
        # Collect images from all split folders (train + test + val)
        image_dirs = []
        for split_dir in data_path.iterdir():
            if split_dir.is_dir() and split_dir.name.lower() in split_names:
                for class_dir in sorted(split_dir.iterdir()):
                    if class_dir.is_dir() and class_dir.name.lower() not in split_names:
                        image_dirs.append(class_dir)

        # Build class mapping from all split folders
        all_class_names = sorted(set(d.name for d in image_dirs))
        class_names = {i: name for i, name in enumerate(all_class_names)}
        class_name_to_id = {name: i for i, name in class_names.items()}
    else:
        # Direct structure: data_dir/class_0/, data_dir/class_1/
        class_dirs = sorted([d for d in data_path.iterdir() if d.is_dir()])
        class_names = {i: d.name for i, d in enumerate(class_dirs)}
        class_name_to_id = {d.name: i for i, d in enumerate(class_dirs)}
        image_dirs = class_dirs

    num_classes = len(class_names)
    print(f"Loading dataset from: {data_dir}")
    print(f"Found {num_classes} classes: {list(class_names.values())}")

    # Load all images
    all_images = []
    all_labels = []
    img_extensions = {"*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"}

    for class_dir in image_dirs:
        class_id = class_name_to_id[class_dir.name]
        image_files = []
        for ext in img_extensions:
            image_files.extend(class_dir.glob(ext))

        for img_path in image_files:
            try:
                img = Image.open(img_path).convert("RGB")
                img = img.resize((img_size, img_size))
                img_array = np.array(img, dtype=np.float32) / 255.0
                # Normalize with ImageNet statistics
                mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                img_array = (img_array - mean) / std
                # Convert to (C, H, W) format
                img_array = img_array.transpose(2, 0, 1)
                all_images.append(img_array)
                all_labels.append(class_id)
            except Exception as e:
                print(f"    Warning: Could not load {img_path}: {e}")

    X = np.array(all_images)
    y = np.array(all_labels, dtype=np.int64)

    # Print class counts
    for cid, cname in class_names.items():
        count = (y == cid).sum()
        print(f"  Class {cid} ({cname}): {count} images")
    print(f"Total: {len(y)} images loaded")

    # Shuffle data
    indices = np.random.permutation(len(y))
    X = X[indices]
    y = y[indices]

    # Partition data for FL
    client_all_data = {}

    if is_iid:
        samples_per_client = len(y) // num_clients
        for i in range(num_clients):
            start = i * samples_per_client
            end = start + samples_per_client if i < num_clients - 1 else len(y)
            client_all_data[i] = (X[start:end], y[start:end])
    else:
        label_indices = {c: np.where(y == c)[0] for c in range(num_classes)}
        proportions = np.random.dirichlet([alpha] * num_clients, num_classes)

        client_indices = {i: [] for i in range(num_clients)}
        for class_id in range(num_classes):
            class_idx = label_indices[class_id]
            n_class = len(class_idx)
            splits = (proportions[class_id] * n_class).astype(int)
            splits[-1] = n_class - splits[:-1].sum()

            current = 0
            for client_id, count in enumerate(splits):
                client_indices[client_id].extend(
                    class_idx[current:current + count].tolist()
                )
                current += count

        for client_id, idx_list in client_indices.items():
            np.random.shuffle(idx_list)
            client_all_data[client_id] = (X[idx_list], y[idx_list])

    # Split each client's data into train/test
    client_train_data = {}
    client_test_data = {}

    print("\nClient data distribution (train / test):")
    for client_id, (X_c, y_c) in client_all_data.items():
        n = len(y_c)
        n_test = max(1, int(n * test_split))
        n_train = n - n_test

        perm = np.random.permutation(n)
        train_idx = perm[:n_train]
        test_idx = perm[n_train:]

        client_train_data[client_id] = (X_c[train_idx], y_c[train_idx])
        client_test_data[client_id] = (X_c[test_idx], y_c[test_idx])

        unique_train, counts_train = np.unique(y_c[train_idx], return_counts=True)
        dist_train = dict(zip(unique_train.tolist(), counts_train.tolist()))
        print(f"  Client {client_id}: {n_train} train / {n_test} test, train dist: {dist_train}")

    return client_train_data, client_test_data, class_names, num_classes


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
    quality_score: Optional[float] = None


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
        # External data (e.g. from FHIR pipeline)
        external_data: Optional[Dict[int, tuple]] = None,
        external_test_data: Optional[Dict[int, tuple]] = None,
        input_dim: Optional[int] = None,
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
        self.global_model = HealthcareMLP(input_dim=_input_dim).to(self.device)

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
            # FedAdam: Adam optimizer on server side
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
                param.data += self.server_lr * self.server_momentum[name] / (
                    torch.sqrt(self.server_velocity[name]) + self.tau
                )

        elif self.algorithm == "FedYogi":
            # FedYogi: like FedAdam but with different velocity update
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
                param.data += self.server_lr * self.server_momentum[name] / (
                    torch.sqrt(self.server_velocity[name]) + self.tau
                )

        elif self.algorithm == "FedAdagrad":
            # FedAdagrad: Adagrad on server side (no momentum)
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
                    # delta_c_i = c_i_new - c_i_old
                    delta_c += (
                        self.client_controls[cr.client_id][name]
                        - self._old_client_controls[cr.client_id][name]
                    )
                self.server_control[name] += delta_c / n

        # DP: add noise to aggregated model
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
        all_probs = []

        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for client_id in range(self.num_clients):
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
        img_size: int = 128,
        progress_callback: Optional[Callable] = None,
        # Server optimizer params (FedAdam, FedYogi, FedAdagrad)
        server_lr: float = 0.1,
        beta1: float = 0.9,
        beta2: float = 0.99,
        tau: float = 1e-3,
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

        # Server optimizer hyperparameters
        self.server_lr = server_lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.tau = tau
        self.num_rounds = None  # Set externally for cosine LR decay

        # Auto-detect device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

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

        # Initialize CNN model
        self.global_model = HealthcareCNN(num_classes=self.num_classes).to(self.device)

        print(f"\nModel: HealthcareCNN ({self.num_classes} classes)")
        print(f"Device: {self.device}")
        total_params = sum(p.numel() for p in self.global_model.parameters())
        print(f"Parameters: {total_params:,}")

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

        # History
        self.history = []

    def _get_client_dataloader(self, client_id: int) -> DataLoader:
        """Create DataLoader for client with data augmentation."""
        X, y = self.client_data[client_id]
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y)

        # Apply random augmentation on CPU before moving to device
        augmented = self._augment_batch(X_tensor)
        augmented = augmented.to(self.device)
        y_tensor = y_tensor.to(self.device)

        dataset = TensorDataset(augmented, y_tensor)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    @staticmethod
    def _augment_batch(images: torch.Tensor) -> torch.Tensor:
        """Apply fast vectorized augmentation to batch of images (N, C, H, W)."""
        images = images.clone()
        n = images.shape[0]

        # Random horizontal flip (vectorized - very fast)
        flip_mask = torch.rand(n) > 0.5
        if flip_mask.any():
            images[flip_mask] = images[flip_mask].flip(-1)

        # Random brightness adjustment (vectorized)
        brightness = torch.empty(n, 1, 1, 1).uniform_(0.85, 1.15)
        images = torch.clamp(images * brightness, -3.0, 3.0)

        return images

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
        local_model.train()

        current_lr = self._get_round_lr(round_num)
        optimizer = torch.optim.Adam(local_model.parameters(), lr=current_lr,
                                     weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss()

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

    def _aggregate(self, client_results: List[ClientResult],
                   noise_scale_override: Optional[float] = None,
                   quality_weights: Optional[Dict[int, float]] = None) -> None:
        """Aggregate client updates. Supports all 10 FL algorithms.

        Args:
            quality_weights: Optional {client_id: quality_weight} from Data Quality
                Framework. If provided, weights are multiplied by quality scores.
        """
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
                param.data += self.server_lr * self.server_momentum[name] / (
                    torch.sqrt(self.server_velocity[name]) + self.tau
                )

        elif self.algorithm == "FedYogi":
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
                param.data += self.server_lr * self.server_momentum[name] / (
                    torch.sqrt(self.server_velocity[name]) + self.tau
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

        # Show evaluation progress
        print("  Evaluating global model...", end="\r")
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
        client_train_data, client_test_data = generate_healthcare_data(
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
            X, y = client_train_data[client_id]
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
# CENTRALIZED IMAGE TRAINER (for imaging datasets)
# =============================================================================


class CentralizedImageTrainer:
    """
    Centralized baseline trainer for medical image classification.
    Pools all hospital image data into a single central model.
    Serves as the upper bound for FL performance comparison.
    """

    def __init__(
        self,
        data_dir: str,
        num_clients: int = 5,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        is_iid: bool = False,
        alpha: float = 0.5,
        seed: int = 42,
        device: str = None,
        img_size: int = 128,
        progress_callback: Optional[Callable] = None,
    ):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.seed = seed
        self.progress_callback = progress_callback
        self.img_size = img_size

        torch.manual_seed(seed)
        np.random.seed(seed)

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Load data using same function as ImageFederatedTrainer
        client_train, client_test, class_names, num_classes = load_image_dataset(
            data_dir=data_dir,
            num_clients=num_clients,
            is_iid=is_iid,
            alpha=alpha,
            img_size=img_size,
            seed=seed,
            test_split=0.2,
        )

        self.num_classes = num_classes
        self.class_names = class_names
        self.num_clients = num_clients
        self.client_train_data = client_train
        self.client_test_data = client_test

        # Pool ALL client data into single centralized dataset
        all_X_train = [client_train[c][0] for c in sorted(client_train.keys())]
        all_y_train = [client_train[c][1] for c in sorted(client_train.keys())]
        all_X_test = [client_test[c][0] for c in sorted(client_test.keys())]
        all_y_test = [client_test[c][1] for c in sorted(client_test.keys())]

        self.X_train = np.concatenate(all_X_train, axis=0)
        self.y_train = np.concatenate(all_y_train, axis=0)
        self.X_test = np.concatenate(all_X_test, axis=0)
        self.y_test = np.concatenate(all_y_test, axis=0)

        print(f"  Centralized: {len(self.y_train)} train + {len(self.y_test)} test samples pooled")

        # Same model as federated
        self.model = HealthcareCNN(num_classes=num_classes).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate, weight_decay=1e-5
        )
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = None  # Set via set_total_epochs()
        self.history: List[CentralizedResult] = []

    def set_total_epochs(self, total_epochs: int):
        """Enable cosine LR scheduler over total_epochs."""
        from torch.optim.lr_scheduler import CosineAnnealingLR
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=total_epochs, eta_min=self.learning_rate * 0.1
        )

    def train_epoch(self, epoch: int) -> CentralizedResult:
        """Train for one epoch on pooled data."""
        start_time = time.time()

        if self.progress_callback:
            self.progress_callback("epoch_start", epoch=epoch + 1)

        self.model.train()

        # Shuffle training data
        perm = np.random.permutation(len(self.y_train))
        X_shuffled = self.X_train[perm]
        y_shuffled = self.y_train[perm]

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for i in range(0, len(y_shuffled), self.batch_size):
            X_batch = torch.FloatTensor(X_shuffled[i:i+self.batch_size]).to(self.device)
            y_batch = torch.LongTensor(y_shuffled[i:i+self.batch_size]).to(self.device)

            # Same augmentation as federated training
            X_batch = ImageFederatedTrainer._augment_batch(X_batch)

            self.optimizer.zero_grad()
            outputs = self.model(X_batch)
            loss = self.criterion(outputs, y_batch)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * len(y_batch)
            preds = outputs.argmax(dim=1)
            total_correct += (preds == y_batch).sum().item()
            total_samples += len(y_batch)

        train_loss = total_loss / max(total_samples, 1)
        train_acc = total_correct / max(total_samples, 1)

        # Step LR scheduler after each epoch
        if self.scheduler is not None:
            self.scheduler.step()

        # Evaluate on held-out test set
        val_metrics = self._evaluate()
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
            time_seconds=elapsed,
        )
        self.history.append(result)

        if self.progress_callback:
            self.progress_callback(
                "epoch_end", epoch=epoch + 1,
                train_loss=train_loss, train_acc=train_acc,
                val_acc=val_metrics["accuracy"], val_f1=val_metrics["f1"],
                val_auc=val_metrics["auc"], time=elapsed,
            )

        return result

    def _evaluate(self) -> Dict[str, float]:
        """Evaluate model on held-out test data."""
        self.model.eval()
        total_loss = 0.0
        all_preds, all_labels, all_probs = [], [], []

        with torch.no_grad():
            for i in range(0, len(self.y_test), self.batch_size):
                X_batch = torch.FloatTensor(self.X_test[i:i+self.batch_size]).to(self.device)
                y_batch = torch.LongTensor(self.y_test[i:i+self.batch_size]).to(self.device)

                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                total_loss += loss.item() * len(y_batch)

                preds = outputs.argmax(dim=1)
                probs = torch.softmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

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
            "loss": total_loss / max(len(all_labels), 1),
            "accuracy": float(accuracy),
            "precision": float(np.mean(precisions)),
            "recall": float(np.mean(recalls)),
            "f1": float(np.mean(f1s)),
            "auc": float(auc),
        }

    def get_predictions(self) -> tuple:
        """Run inference on test set. Returns (preds, probs, labels)."""
        self.model.eval()
        all_preds, all_probs, all_labels = [], [], []
        with torch.no_grad():
            for i in range(0, len(self.y_test), self.batch_size):
                X_batch = torch.FloatTensor(self.X_test[i:i+self.batch_size]).to(self.device)
                y_batch = self.y_test[i:i+self.batch_size]
                outputs = self.model(X_batch)
                all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
                all_probs.extend(torch.softmax(outputs, dim=1).cpu().numpy())
                all_labels.extend(y_batch)
        return np.array(all_preds), np.array(all_probs), np.array(all_labels)

    def get_model(self) -> nn.Module:
        """Return the trained model."""
        return self.model

    def get_data_stats(self) -> Dict:
        """Get per-client and overall data statistics."""
        per_client = {}
        for cid in sorted(self.client_train_data.keys()):
            _, y_tr = self.client_train_data[cid]
            _, y_te = self.client_test_data[cid]
            unique, counts = np.unique(y_tr, return_counts=True)
            per_client[cid] = {
                "train_samples": len(y_tr),
                "test_samples": len(y_te),
                "class_distribution": dict(zip(unique.tolist(), counts.tolist())),
            }

        unique_all, counts_all = np.unique(self.y_train, return_counts=True)
        return {
            "total_train": len(self.y_train),
            "total_test": len(self.y_test),
            "num_classes": self.num_classes,
            "class_names": self.class_names,
            "overall_distribution": dict(zip(unique_all.tolist(), counts_all.tolist())),
            "per_client": per_client,
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

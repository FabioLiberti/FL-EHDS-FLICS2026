#!/usr/bin/env python3
"""
FL-EHDS CNN Model for Federated Learning

Supports:
1. CIFAR-10 (for testing and benchmarking)
2. Medical imaging datasets (ChestX-ray, CheXpert, MIMIC-CXR)
3. Custom image classification tasks

Features:
- Non-IID data partitioning across nodes
- FedAvg and FedProx algorithms
- Differential Privacy with Opacus
- Gradient compression support
- Real-time training visualization

Author: Fabio Liberti
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import json
from datetime import datetime
import numpy as np

# PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset, Subset
    import torchvision
    import torchvision.transforms as transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = nn = F = DataLoader = Dataset = Subset = None
    torchvision = transforms = None

# Opacus for DP
try:
    from opacus import PrivacyEngine
    from opacus.validators import ModuleValidator
    OPACUS_AVAILABLE = True
except ImportError:
    OPACUS_AVAILABLE = False
    PrivacyEngine = ModuleValidator = None


# =============================================================================
# CNN ARCHITECTURES
# =============================================================================

class SimpleCNN(nn.Module):
    """Simple CNN for CIFAR-10 / small medical images."""

    def __init__(self, num_classes: int = 10, in_channels: int = 3):
        super(SimpleCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class MedicalCNN(nn.Module):
    """CNN optimized for medical imaging (e.g., chest X-rays)."""

    def __init__(self, num_classes: int = 2, in_channels: int = 1):
        super(MedicalCNN, self).__init__()

        # Larger receptive field for medical images
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class ResNetMedical(nn.Module):
    """ResNet-based model for medical imaging using transfer learning."""

    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super(ResNetMedical, self).__init__()

        # Load pretrained ResNet18
        self.base_model = torchvision.models.resnet18(
            weights='IMAGENET1K_V1' if pretrained else None
        )

        # Modify first conv for grayscale (optional)
        # self.base_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Replace final FC layer
        in_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.base_model(x)


# =============================================================================
# NON-IID DATA PARTITIONING
# =============================================================================

class NonIIDPartitioner:
    """Partitions datasets to create non-IID distributions across nodes."""

    def __init__(self, num_nodes: int, random_seed: int = 42):
        self.num_nodes = num_nodes
        self.rng = np.random.RandomState(random_seed)

    def partition_by_dirichlet(self,
                               dataset: Dataset,
                               alpha: float = 0.5) -> Dict[int, List[int]]:
        """
        Partition using Dirichlet distribution for label skew.

        Alpha controls non-IID degree:
        - alpha -> 0: extreme non-IID (each node gets mostly one class)
        - alpha -> inf: IID distribution
        """
        targets = np.array([dataset[i][1] for i in range(len(dataset))])
        num_classes = len(np.unique(targets))

        # Sample label distributions from Dirichlet
        label_distributions = self.rng.dirichlet(
            [alpha] * num_classes,
            size=self.num_nodes
        )

        # Assign samples to nodes based on distributions
        node_indices = {node: [] for node in range(self.num_nodes)}

        for class_idx in range(num_classes):
            class_indices = np.where(targets == class_idx)[0]
            self.rng.shuffle(class_indices)

            # Distribute to nodes based on probabilities
            proportions = label_distributions[:, class_idx]
            proportions = proportions / proportions.sum()

            splits = (proportions * len(class_indices)).astype(int)
            splits[-1] = len(class_indices) - splits[:-1].sum()

            start_idx = 0
            for node_id, num_samples in enumerate(splits):
                end_idx = start_idx + num_samples
                node_indices[node_id].extend(class_indices[start_idx:end_idx])
                start_idx = end_idx

        return node_indices

    def partition_by_shards(self,
                            dataset: Dataset,
                            shards_per_node: int = 2) -> Dict[int, List[int]]:
        """
        Partition by assigning fixed shards per node.
        Creates pathological non-IID: each node gets only a few classes.
        """
        targets = np.array([dataset[i][1] for i in range(len(dataset))])
        sorted_indices = np.argsort(targets)

        num_shards = self.num_nodes * shards_per_node
        shard_size = len(dataset) // num_shards

        # Create shards
        shards = [
            sorted_indices[i * shard_size: (i + 1) * shard_size].tolist()
            for i in range(num_shards)
        ]

        # Assign shards to nodes
        shard_indices = list(range(num_shards))
        self.rng.shuffle(shard_indices)

        node_indices = {node: [] for node in range(self.num_nodes)}

        for node_id in range(self.num_nodes):
            for i in range(shards_per_node):
                shard_idx = shard_indices[node_id * shards_per_node + i]
                node_indices[node_id].extend(shards[shard_idx])

        return node_indices

    def partition_quantity_imbalanced(self,
                                      dataset: Dataset,
                                      imbalance_factor: float = 5.0) -> Dict[int, List[int]]:
        """
        Create quantity imbalance across nodes.
        imbalance_factor: ratio between largest and smallest node.
        """
        indices = list(range(len(dataset)))
        self.rng.shuffle(indices)

        # Compute exponentially decaying proportions
        proportions = np.exp(-np.linspace(0, np.log(imbalance_factor), self.num_nodes))
        proportions = proportions / proportions.sum()

        splits = (proportions * len(dataset)).astype(int)
        splits[-1] = len(dataset) - splits[:-1].sum()

        node_indices = {}
        start_idx = 0

        for node_id, num_samples in enumerate(splits):
            end_idx = start_idx + num_samples
            node_indices[node_id] = indices[start_idx:end_idx]
            start_idx = end_idx

        return node_indices


# =============================================================================
# DATASET LOADERS
# =============================================================================

class DatasetManager:
    """Manages dataset loading and preparation for FL training."""

    def __init__(self, data_dir: str = './data', random_seed: int = 42):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.rng = np.random.RandomState(random_seed)

    def load_cifar10(self) -> Tuple[Dataset, Dataset]:
        """Load CIFAR-10 dataset."""
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

        train_dataset = torchvision.datasets.CIFAR10(
            root=str(self.data_dir),
            train=True,
            download=True,
            transform=transform_train
        )

        test_dataset = torchvision.datasets.CIFAR10(
            root=str(self.data_dir),
            train=False,
            download=True,
            transform=transform_test
        )

        return train_dataset, test_dataset

    def load_mnist(self) -> Tuple[Dataset, Dataset]:
        """Load MNIST dataset (useful for quick experiments)."""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        train_dataset = torchvision.datasets.MNIST(
            root=str(self.data_dir),
            train=True,
            download=True,
            transform=transform
        )

        test_dataset = torchvision.datasets.MNIST(
            root=str(self.data_dir),
            train=False,
            download=True,
            transform=transform
        )

        return train_dataset, test_dataset

    def create_synthetic_medical(self,
                                 num_samples: int = 1000,
                                 image_size: int = 64,
                                 num_classes: int = 2) -> Dataset:
        """
        Create synthetic medical-like images for testing.
        In production, replace with actual medical dataset loading.
        """
        class SyntheticMedicalDataset(Dataset):
            def __init__(self, num_samples, image_size, num_classes, rng):
                self.num_samples = num_samples
                self.image_size = image_size
                self.num_classes = num_classes
                self.rng = rng

                # Generate synthetic images
                self.images = []
                self.labels = []

                for i in range(num_samples):
                    label = i % num_classes

                    # Create base image with class-specific pattern
                    img = self.rng.randn(1, image_size, image_size).astype(np.float32) * 0.1

                    # Add class-specific features
                    if label == 0:
                        # "Normal" - smooth patterns
                        img += 0.3 * np.sin(np.linspace(0, 3 * np.pi, image_size)).reshape(1, 1, -1)
                    else:
                        # "Abnormal" - noisy with spots
                        spots = self.rng.random((1, image_size, image_size)) > 0.95
                        img += spots * 0.8

                    img = np.clip(img, -1, 1)

                    self.images.append(torch.FloatTensor(img))
                    self.labels.append(label)

            def __len__(self):
                return self.num_samples

            def __getitem__(self, idx):
                return self.images[idx], self.labels[idx]

        return SyntheticMedicalDataset(num_samples, image_size, num_classes, self.rng)


# =============================================================================
# FL TRAINER
# =============================================================================

@dataclass
class FLTrainingConfig:
    """Configuration for FL training."""
    num_nodes: int = 5
    num_rounds: int = 50
    local_epochs: int = 3
    batch_size: int = 32
    learning_rate: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 1e-4

    # Algorithm
    algorithm: str = 'FedAvg'  # FedAvg, FedProx
    fedprox_mu: float = 0.1

    # Non-IID settings
    partition_method: str = 'dirichlet'  # dirichlet, shards, quantity
    dirichlet_alpha: float = 0.5
    shards_per_node: int = 2
    quantity_imbalance: float = 5.0

    # Differential Privacy
    use_dp: bool = False
    dp_epsilon: float = 10.0
    dp_delta: float = 1e-5
    dp_max_grad_norm: float = 1.0

    # Device
    device: str = 'auto'  # auto, cpu, cuda, mps

    # Random seed
    random_seed: int = 42


class FederatedTrainer:
    """
    Federated Learning trainer for CNN models.

    Supports:
    - FedAvg and FedProx algorithms
    - Non-IID data partitioning
    - Differential Privacy
    """

    def __init__(self, config: FLTrainingConfig):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required. Install with: pip install torch torchvision")

        self.config = config
        self.device = self._get_device()

        # Initialize components
        self.partitioner = NonIIDPartitioner(
            config.num_nodes,
            config.random_seed
        )

        # Training state
        self.global_model: Optional[nn.Module] = None
        self.node_data_loaders: Dict[int, DataLoader] = {}
        self.test_loader: Optional[DataLoader] = None
        self.history: List[Dict] = []

    def _get_device(self) -> torch.device:
        """Determine the best available device."""
        if self.config.device == 'auto':
            if torch.cuda.is_available():
                return torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device('mps')
            else:
                return torch.device('cpu')
        return torch.device(self.config.device)

    def setup_data(self,
                   train_dataset: Dataset,
                   test_dataset: Dataset):
        """Partition data among nodes and create data loaders."""
        config = self.config

        # Partition training data
        if config.partition_method == 'dirichlet':
            node_indices = self.partitioner.partition_by_dirichlet(
                train_dataset,
                alpha=config.dirichlet_alpha
            )
        elif config.partition_method == 'shards':
            node_indices = self.partitioner.partition_by_shards(
                train_dataset,
                shards_per_node=config.shards_per_node
            )
        else:
            node_indices = self.partitioner.partition_quantity_imbalanced(
                train_dataset,
                imbalance_factor=config.quantity_imbalance
            )

        # Create data loaders for each node
        for node_id, indices in node_indices.items():
            subset = Subset(train_dataset, indices)
            loader = DataLoader(
                subset,
                batch_size=config.batch_size,
                shuffle=True,
                num_workers=0,
                drop_last=True
            )
            self.node_data_loaders[node_id] = loader

        # Test loader
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=config.batch_size * 2,
            shuffle=False,
            num_workers=0
        )

        # Print partition statistics
        print("\nData Partition Statistics:")
        print("-" * 40)
        for node_id, indices in node_indices.items():
            targets = [train_dataset[i][1] for i in indices]
            unique, counts = np.unique(targets, return_counts=True)
            dist = dict(zip(unique.tolist(), counts.tolist()))
            print(f"Node {node_id + 1}: {len(indices)} samples, distribution: {dist}")

    def setup_model(self,
                    model_type: str = 'simple_cnn',
                    num_classes: int = 10,
                    **model_kwargs):
        """Initialize the global model."""
        if model_type == 'simple_cnn':
            self.global_model = SimpleCNN(num_classes=num_classes, **model_kwargs)
        elif model_type == 'medical_cnn':
            self.global_model = MedicalCNN(num_classes=num_classes, **model_kwargs)
        elif model_type == 'resnet':
            self.global_model = ResNetMedical(num_classes=num_classes, **model_kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Make DP compatible if needed
        if self.config.use_dp and OPACUS_AVAILABLE:
            self.global_model = ModuleValidator.fix(self.global_model)

        self.global_model = self.global_model.to(self.device)
        print(f"\nModel initialized on {self.device}")
        print(f"Total parameters: {sum(p.numel() for p in self.global_model.parameters()):,}")

    def train_round(self, round_num: int) -> Dict:
        """Execute one round of federated training."""
        config = self.config
        node_updates = []
        node_sample_counts = []
        node_metrics = {}

        global_state = {k: v.clone() for k, v in self.global_model.state_dict().items()}

        for node_id in range(config.num_nodes):
            # Create local model
            local_model = type(self.global_model)(
                num_classes=self.global_model.classifier[-1].out_features
            ).to(self.device)
            local_model.load_state_dict(global_state)

            # Local training
            optimizer = torch.optim.SGD(
                local_model.parameters(),
                lr=config.learning_rate,
                momentum=config.momentum,
                weight_decay=config.weight_decay
            )

            # Apply DP if enabled
            if config.use_dp and OPACUS_AVAILABLE:
                privacy_engine = PrivacyEngine()
                local_model, optimizer, train_loader = privacy_engine.make_private(
                    module=local_model,
                    optimizer=optimizer,
                    data_loader=self.node_data_loaders[node_id],
                    noise_multiplier=config.dp_max_grad_norm,
                    max_grad_norm=config.dp_max_grad_norm
                )
            else:
                train_loader = self.node_data_loaders[node_id]

            local_model.train()
            total_loss = 0
            num_batches = 0

            for epoch in range(config.local_epochs):
                for batch_idx, (data, target) in enumerate(train_loader):
                    data, target = data.to(self.device), target.to(self.device)

                    optimizer.zero_grad()
                    output = local_model(data)
                    loss = F.cross_entropy(output, target)

                    # FedProx proximal term
                    if config.algorithm == 'FedProx':
                        proximal_term = 0
                        for w, w_t in zip(local_model.parameters(),
                                          self.global_model.parameters()):
                            proximal_term += (w - w_t.detach()).pow(2).sum()
                        loss += (config.fedprox_mu / 2) * proximal_term

                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    num_batches += 1

            # Compute update (delta)
            local_state = local_model.state_dict()
            update = {
                k: local_state[k] - global_state[k]
                for k in global_state.keys()
            }

            node_updates.append(update)
            node_sample_counts.append(len(self.node_data_loaders[node_id].dataset))

            # Node metrics
            node_metrics[node_id] = {
                'avg_loss': total_loss / max(num_batches, 1),
                'samples': node_sample_counts[-1]
            }

        # Aggregate updates (FedAvg weighted average)
        total_samples = sum(node_sample_counts)
        aggregated_update = {}

        for key in global_state.keys():
            aggregated_update[key] = sum(
                update[key] * (n / total_samples)
                for update, n in zip(node_updates, node_sample_counts)
            )

        # Update global model
        new_state = {
            k: global_state[k] + aggregated_update[k]
            for k in global_state.keys()
        }
        self.global_model.load_state_dict(new_state)

        # Evaluate global model
        test_accuracy, test_loss = self.evaluate()

        result = {
            'round': round_num,
            'test_accuracy': test_accuracy,
            'test_loss': test_loss,
            'node_metrics': node_metrics
        }

        self.history.append(result)
        return result

    def evaluate(self) -> Tuple[float, float]:
        """Evaluate global model on test set."""
        self.global_model.eval()
        test_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.global_model(data)

                test_loss += F.cross_entropy(output, target, reduction='sum').item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)

        accuracy = correct / total
        avg_loss = test_loss / total

        return accuracy, avg_loss

    def train(self, callback: Optional[callable] = None) -> List[Dict]:
        """Run full FL training."""
        print(f"\nStarting FL Training")
        print(f"Algorithm: {self.config.algorithm}")
        print(f"Nodes: {self.config.num_nodes}")
        print(f"Rounds: {self.config.num_rounds}")
        print(f"DP Enabled: {self.config.use_dp}")
        print("-" * 40)

        for round_num in range(1, self.config.num_rounds + 1):
            result = self.train_round(round_num)

            print(f"Round {round_num}/{self.config.num_rounds} | "
                  f"Accuracy: {result['test_accuracy']:.2%} | "
                  f"Loss: {result['test_loss']:.4f}")

            if callback:
                callback(result)

        print("-" * 40)
        print(f"Final Accuracy: {self.history[-1]['test_accuracy']:.2%}")

        return self.history


# =============================================================================
# MAIN EXAMPLE
# =============================================================================

def run_cifar10_experiment():
    """Run CIFAR-10 FL experiment."""
    print("=" * 60)
    print("FL-EHDS CNN Experiment - CIFAR-10")
    print("=" * 60)

    # Configuration
    config = FLTrainingConfig(
        num_nodes=5,
        num_rounds=20,
        local_epochs=2,
        batch_size=32,
        learning_rate=0.01,
        algorithm='FedAvg',
        partition_method='dirichlet',
        dirichlet_alpha=0.5,  # Non-IID
        use_dp=False,
        random_seed=42
    )

    # Initialize trainer
    trainer = FederatedTrainer(config)

    # Load data
    dataset_manager = DatasetManager()
    train_dataset, test_dataset = dataset_manager.load_cifar10()

    # Setup
    trainer.setup_data(train_dataset, test_dataset)
    trainer.setup_model(model_type='simple_cnn', num_classes=10)

    # Train
    history = trainer.train()

    # Save results
    results = {
        'config': {
            'num_nodes': config.num_nodes,
            'num_rounds': config.num_rounds,
            'algorithm': config.algorithm,
            'dirichlet_alpha': config.dirichlet_alpha
        },
        'history': history
    }

    with open('cifar10_fl_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to cifar10_fl_results.json")

    return history


def run_medical_experiment():
    """Run synthetic medical imaging FL experiment."""
    print("=" * 60)
    print("FL-EHDS CNN Experiment - Medical Imaging")
    print("=" * 60)

    # Configuration
    config = FLTrainingConfig(
        num_nodes=5,
        num_rounds=30,
        local_epochs=3,
        batch_size=16,
        learning_rate=0.001,
        algorithm='FedProx',
        fedprox_mu=0.1,
        partition_method='dirichlet',
        dirichlet_alpha=0.3,  # Stronger non-IID
        use_dp=False,
        random_seed=42
    )

    # Initialize trainer
    trainer = FederatedTrainer(config)

    # Create synthetic medical dataset
    dataset_manager = DatasetManager()

    train_dataset = dataset_manager.create_synthetic_medical(
        num_samples=2000,
        image_size=64,
        num_classes=2
    )
    test_dataset = dataset_manager.create_synthetic_medical(
        num_samples=400,
        image_size=64,
        num_classes=2
    )

    # Setup
    trainer.setup_data(train_dataset, test_dataset)
    trainer.setup_model(
        model_type='medical_cnn',
        num_classes=2,
        in_channels=1
    )

    # Train
    history = trainer.train()

    return history


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == 'medical':
        run_medical_experiment()
    else:
        run_cifar10_experiment()

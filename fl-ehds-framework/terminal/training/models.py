"""
Neural network models and image dataset loader for FL training.
"""

import numpy as np
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torchvision.models as tv_models


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


class HealthcareResNet(nn.Module):
    """
    ResNet18-based model for medical image classification in FL settings.

    Uses pretrained ImageNet weights with GroupNorm (FL-stable, replaces BatchNorm).
    Optional backbone freezing for faster convergence with limited data.

    Input: (batch, 3, 224, 224) - RGB images
    ~11M params total, ~5M trainable if freeze_backbone=True
    """

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout: float = 0.3,
    ):
        super().__init__()

        # Load pretrained ResNet18
        weights = tv_models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        base = tv_models.resnet18(weights=weights)

        # Replace all BatchNorm2d with GroupNorm (FL-stable)
        self._replace_bn_with_gn(base)

        # Extract backbone (everything except fc)
        self.conv1 = base.conv1
        self.bn1 = base.bn1  # Now GroupNorm
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.avgpool = base.avgpool

        # Replace fc with dropout + linear
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

        # Optionally freeze early layers
        if freeze_backbone:
            for param in self.conv1.parameters():
                param.requires_grad = False
            for param in self.bn1.parameters():
                param.requires_grad = False
            for param in self.layer1.parameters():
                param.requires_grad = False
            for param in self.layer2.parameters():
                param.requires_grad = False

    @staticmethod
    def _replace_bn_with_gn(module: nn.Module):
        """Recursively replace BatchNorm2d with GroupNorm(16, channels)."""
        for name, child in module.named_children():
            if isinstance(child, nn.BatchNorm2d):
                num_channels = child.num_features
                num_groups = min(16, num_channels)
                # Ensure num_channels is divisible by num_groups
                while num_channels % num_groups != 0 and num_groups > 1:
                    num_groups -= 1
                setattr(module, name, nn.GroupNorm(num_groups, num_channels))
            else:
                HealthcareResNet._replace_bn_with_gn(child)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


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

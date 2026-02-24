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
        freeze_level: int = None,
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

        # Determine freeze level: 0=none, 1=conv1+bn1, 2=+layer1, 3=+layer2
        # freeze_backbone=True is backward-compatible alias for freeze_level=3
        if freeze_level is None:
            freeze_level = 3 if freeze_backbone else 0
        self.freeze_level = freeze_level

        freeze_groups = [
            [self.conv1, self.bn1],       # level >= 1
            [self.layer1],                 # level >= 2
            [self.layer2],                 # level >= 3
        ]
        for lvl, modules in enumerate(freeze_groups, 1):
            if freeze_level >= lvl:
                for mod in modules:
                    for param in mod.parameters():
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

    import gc

    # --- Phase 1: Count images (no RAM used) ---
    img_extensions = {"*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"}
    image_manifest = []  # List of (path, class_id) — lightweight

    for class_dir in image_dirs:
        class_id = class_name_to_id[class_dir.name]
        for ext in img_extensions:
            for img_path in class_dir.glob(ext):
                image_manifest.append((img_path, class_id))

    total_images = len(image_manifest)
    print(f"  Found {total_images} images to load")

    # --- Phase 2: Load directly into pre-allocated array (single copy in RAM) ---
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    X = np.empty((total_images, 3, img_size, img_size), dtype=np.float32)
    y = np.empty(total_images, dtype=np.int64)
    idx = 0

    for img_path, class_id in image_manifest:
        try:
            img = Image.open(img_path).convert("RGB")
            img = img.resize((img_size, img_size))
            img_array = np.array(img, dtype=np.float32) / 255.0
            img_array = (img_array - mean) / std
            X[idx] = img_array.transpose(2, 0, 1)
            y[idx] = class_id
            idx += 1
        except Exception as e:
            print(f"    Warning: Could not load {img_path}: {e}")

    del image_manifest
    # Trim if some images failed to load
    if idx < total_images:
        X = X[:idx]
        y = y[:idx]

    # Print class counts
    for cid, cname in class_names.items():
        count = (y == cid).sum()
        print(f"  Class {cid} ({cname}): {count} images")
    print(f"Total: {len(y)} images loaded")

    # --- Phase 3: Partition directly into clients (no global shuffle needed) ---
    # For non-IID: Dirichlet selects by class, so global shuffle is unnecessary.
    # For IID: use shuffled indices instead of reordering the whole array.
    client_train_data = {}
    client_test_data = {}

    if is_iid:
        perm = np.random.permutation(len(y))
        samples_per_client = len(y) // num_clients
        for i in range(num_clients):
            start = i * samples_per_client
            end = start + samples_per_client if i < num_clients - 1 else len(y)
            client_idx = perm[start:end]
            X_c, y_c = X[client_idx], y[client_idx]
            n = len(y_c)
            n_test = max(1, int(n * test_split))
            n_train = n - n_test
            p = np.random.permutation(n)
            client_train_data[i] = (X_c[p[:n_train]], y_c[p[:n_train]])
            client_test_data[i] = (X_c[p[n_train:]], y_c[p[n_train:]])
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
            X_c, y_c = X[idx_list], y[idx_list]
            n = len(y_c)
            n_test = max(1, int(n * test_split))
            n_train = n - n_test
            p = np.random.permutation(n)
            client_train_data[client_id] = (X_c[p[:n_train]], y_c[p[:n_train]])
            client_test_data[client_id] = (X_c[p[n_train:]], y_c[p[n_train:]])

    # Free the big array — data is now split across clients
    del X, y
    gc.collect()

    print("\nClient data distribution (train / test):")
    for client_id in sorted(client_train_data.keys()):
        X_tr, y_tr = client_train_data[client_id]
        X_te, y_te = client_test_data[client_id]
        unique_train, counts_train = np.unique(y_tr, return_counts=True)
        dist_train = dict(zip(unique_train.tolist(), counts_train.tolist()))
        print(f"  Client {client_id}: {len(y_tr)} train / {len(y_te)} test, train dist: {dist_train}")

    del client_all_data
    gc.collect()

    return client_train_data, client_test_data, class_names, num_classes

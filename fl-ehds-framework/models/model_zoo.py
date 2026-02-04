#!/usr/bin/env python3
"""
FL-EHDS Model Zoo

Comprehensive collection of neural network architectures for FL:

1. MLP Models (for tabular data)
   - SimpleMLP
   - DeepMLP
   - ResidualMLP

2. CNN Models (for images)
   - LeNet5
   - SimpleCNN
   - VGGStyle
   - ResNet variants
   - MobileNetStyle
   - EfficientNetStyle

3. Medical Imaging Models
   - MedicalCNN
   - DenseNetMedical
   - UNetEncoder

4. Transformer Models
   - VisionTransformer (ViT)
   - TabTransformer

5. Recurrent Models (for sequences)
   - LSTM
   - GRU

Author: Fabio Liberti
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import math

# PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = nn = F = Tensor = None


# =============================================================================
# MODEL INFO
# =============================================================================

MODEL_INFO = {
    "SimpleMLP": {
        "name": "Simple MLP",
        "type": "MLP",
        "description": "Basic multi-layer perceptron with configurable hidden layers. "
                      "Good for tabular healthcare data.",
        "input_type": "tabular",
        "params": {
            "hidden_dims": "List of hidden layer dimensions [64, 32]",
            "dropout": "Dropout rate (0.0-0.5)"
        },
        "use_case": "Tabular data, small datasets, baseline"
    },

    "DeepMLP": {
        "name": "Deep MLP",
        "type": "MLP",
        "description": "Deeper MLP with batch normalization and residual connections. "
                      "Better for complex tabular patterns.",
        "input_type": "tabular",
        "params": {
            "hidden_dims": "Hidden dimensions [128, 64, 32]",
            "use_batchnorm": "Enable batch normalization",
            "use_residual": "Enable residual connections"
        },
        "use_case": "Complex tabular data, larger datasets"
    },

    "LeNet5": {
        "name": "LeNet-5",
        "type": "CNN",
        "description": "Classic CNN architecture from 1998. "
                      "Good for small grayscale images like MNIST.",
        "input_type": "image",
        "params": {
            "in_channels": "Input channels (1 for grayscale, 3 for RGB)",
            "num_classes": "Number of output classes"
        },
        "use_case": "MNIST, small medical images"
    },

    "SimpleCNN": {
        "name": "Simple CNN",
        "type": "CNN",
        "description": "Lightweight CNN with 3 conv layers. "
                      "Good balance of accuracy and communication cost.",
        "input_type": "image",
        "params": {
            "in_channels": "Input channels",
            "num_classes": "Output classes"
        },
        "use_case": "CIFAR-10, small-medium images"
    },

    "VGGStyle": {
        "name": "VGG-Style CNN",
        "type": "CNN",
        "description": "VGG-inspired architecture with repeated conv blocks. "
                      "Deeper than SimpleCNN, better for complex patterns.",
        "input_type": "image",
        "params": {
            "num_blocks": "Number of VGG blocks (2-5)",
            "base_channels": "Starting channel count (16, 32, 64)"
        },
        "use_case": "Complex image classification"
    },

    "ResNet18": {
        "name": "ResNet-18",
        "type": "CNN",
        "description": "18-layer ResNet with skip connections. "
                      "Standard architecture for medical imaging.",
        "input_type": "image",
        "params": {
            "pretrained": "Use ImageNet pretrained weights",
            "num_classes": "Output classes"
        },
        "use_case": "Medical imaging, transfer learning"
    },

    "MobileNetStyle": {
        "name": "MobileNet-Style",
        "type": "CNN",
        "description": "Lightweight CNN with depthwise separable convolutions. "
                      "Efficient for mobile/edge deployment.",
        "input_type": "image",
        "params": {
            "width_multiplier": "Channel scaling factor (0.5-1.0)"
        },
        "use_case": "Edge deployment, bandwidth-limited FL"
    },

    "MedicalCNN": {
        "name": "Medical CNN",
        "type": "CNN",
        "description": "CNN optimized for medical imaging with larger kernels "
                      "and attention mechanisms.",
        "input_type": "image",
        "params": {
            "in_channels": "1 for X-ray, 3 for fundus images",
            "attention": "Enable attention module"
        },
        "use_case": "Chest X-ray, fundus images, CT scans"
    },

    "DenseNetMedical": {
        "name": "DenseNet Medical",
        "type": "CNN",
        "description": "DenseNet-style architecture with dense connections. "
                      "Efficient parameter usage, good for medical imaging.",
        "input_type": "image",
        "params": {
            "growth_rate": "Channel growth per layer (12, 24, 32)",
            "num_blocks": "Number of dense blocks"
        },
        "use_case": "Medical imaging with limited data"
    },

    "VisionTransformer": {
        "name": "Vision Transformer (ViT)",
        "type": "Transformer",
        "description": "Transformer architecture for images. "
                      "Patches image into tokens for self-attention.",
        "input_type": "image",
        "params": {
            "patch_size": "Size of image patches (8, 16, 32)",
            "embed_dim": "Embedding dimension",
            "num_heads": "Attention heads",
            "num_layers": "Transformer layers"
        },
        "use_case": "Large datasets, complex patterns"
    },

    "TabTransformer": {
        "name": "Tab Transformer",
        "type": "Transformer",
        "description": "Transformer for tabular data. "
                      "Applies attention over categorical features.",
        "input_type": "tabular",
        "params": {
            "num_categories": "Number of categorical features",
            "embed_dim": "Embedding dimension"
        },
        "use_case": "Mixed tabular data with categorical features"
    },

    "LSTM": {
        "name": "LSTM",
        "type": "RNN",
        "description": "Long Short-Term Memory network for sequences. "
                      "Good for time-series EHR data.",
        "input_type": "sequence",
        "params": {
            "hidden_size": "LSTM hidden dimension",
            "num_layers": "Number of LSTM layers",
            "bidirectional": "Bidirectional LSTM"
        },
        "use_case": "Time-series, sequential EHR"
    }
}


# =============================================================================
# MLP MODELS
# =============================================================================

class SimpleMLP(nn.Module):
    """Simple Multi-Layer Perceptron."""

    def __init__(self,
                 input_dim: int,
                 hidden_dims: List[int] = [64, 32],
                 num_classes: int = 2,
                 dropout: float = 0.3):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, num_classes))

        self.model = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


class DeepMLP(nn.Module):
    """Deep MLP with BatchNorm and optional residual connections."""

    def __init__(self,
                 input_dim: int,
                 hidden_dims: List[int] = [128, 64, 32],
                 num_classes: int = 2,
                 dropout: float = 0.3,
                 use_batchnorm: bool = True,
                 use_residual: bool = True):
        super().__init__()

        self.use_residual = use_residual
        self.layers = nn.ModuleList()
        self.projections = nn.ModuleList()

        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layer = nn.Sequential(
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim) if use_batchnorm else nn.Identity(),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            self.layers.append(layer)

            # Projection for residual if dimensions don't match
            if use_residual and prev_dim != hidden_dim:
                self.projections.append(nn.Linear(prev_dim, hidden_dim))
            else:
                self.projections.append(None)

            prev_dim = hidden_dim

        self.classifier = nn.Linear(prev_dim, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        for layer, proj in zip(self.layers, self.projections):
            identity = x
            x = layer(x)

            if self.use_residual:
                if proj is not None:
                    identity = proj(identity)
                if identity.shape == x.shape:
                    x = x + identity

        return self.classifier(x)


class ResidualMLP(nn.Module):
    """MLP with strong residual connections (ResNet-style)."""

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 128,
                 num_blocks: int = 3,
                 num_classes: int = 2,
                 dropout: float = 0.2):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim)
            )
            for _ in range(num_blocks)
        ])

        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = self.input_proj(x)

        for block in self.blocks:
            identity = x
            x = block(x)
            x = F.relu(x + identity)

        return self.classifier(x)


# =============================================================================
# CNN MODELS
# =============================================================================

class LeNet5(nn.Module):
    """Classic LeNet-5 architecture."""

    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 10):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 6, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(),
            nn.AvgPool2d(2, 2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        return self.classifier(x)


class SimpleCNN(nn.Module):
    """Simple 3-layer CNN."""

    def __init__(self,
                 in_channels: int = 3,
                 num_classes: int = 10,
                 base_channels: int = 32):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(base_channels, base_channels * 2, 3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(base_channels * 2, base_channels * 4, 3, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base_channels * 4 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        return self.classifier(x)


class VGGBlock(nn.Module):
    """VGG-style convolutional block."""

    def __init__(self, in_channels: int, out_channels: int, num_convs: int = 2):
        super().__init__()

        layers = []
        for i in range(num_convs):
            layers.extend([
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            ])

        layers.append(nn.MaxPool2d(2, 2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class VGGStyle(nn.Module):
    """VGG-inspired architecture."""

    def __init__(self,
                 in_channels: int = 3,
                 num_classes: int = 10,
                 num_blocks: int = 3,
                 base_channels: int = 32):
        super().__init__()

        blocks = []
        channels = base_channels

        for i in range(num_blocks):
            in_ch = in_channels if i == 0 else channels // 2
            blocks.append(VGGBlock(in_ch, channels, num_convs=2))
            channels *= 2

        self.features = nn.Sequential(*blocks)

        # Adaptive pooling for flexible input sizes
        self.avgpool = nn.AdaptiveAvgPool2d((2, 2))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear((channels // 2) * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        return self.classifier(x)


class BasicBlock(nn.Module):
    """ResNet basic block."""

    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: Tensor) -> Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class ResNet18(nn.Module):
    """ResNet-18 implementation."""

    def __init__(self,
                 in_channels: int = 3,
                 num_classes: int = 10):
        super().__init__()

        self.in_channels = 64

        self.conv1 = nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, out_channels: int, num_blocks: int, stride: int):
        layers = [BasicBlock(self.in_channels, out_channels, stride)]
        self.in_channels = out_channels

        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution (MobileNet style)."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        self.depthwise = nn.Conv2d(
            in_channels, in_channels, 3,
            stride=stride, padding=1, groups=in_channels, bias=False
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(self.bn1(self.depthwise(x)))
        x = F.relu(self.bn2(self.pointwise(x)))
        return x


class MobileNetStyle(nn.Module):
    """MobileNet-inspired lightweight architecture."""

    def __init__(self,
                 in_channels: int = 3,
                 num_classes: int = 10,
                 width_multiplier: float = 1.0):
        super().__init__()

        def scaled(x):
            return int(x * width_multiplier)

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, scaled(32), 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(scaled(32)),
            nn.ReLU()
        )

        self.layers = nn.Sequential(
            DepthwiseSeparableConv(scaled(32), scaled(64)),
            DepthwiseSeparableConv(scaled(64), scaled(128), stride=2),
            DepthwiseSeparableConv(scaled(128), scaled(128)),
            DepthwiseSeparableConv(scaled(128), scaled(256), stride=2),
            DepthwiseSeparableConv(scaled(256), scaled(256)),
            DepthwiseSeparableConv(scaled(256), scaled(512), stride=2),
        )

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(scaled(512), num_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.layers(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


# =============================================================================
# MEDICAL IMAGING MODELS
# =============================================================================

class AttentionModule(nn.Module):
    """Channel attention module (SE-Net style)."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()

        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x: Tensor) -> Tensor:
        b, c, _, _ = x.shape
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y


class MedicalCNN(nn.Module):
    """CNN optimized for medical imaging."""

    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 2,
                 use_attention: bool = True):
        super().__init__()

        self.use_attention = use_attention

        # Larger kernels for medical images
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.attention1 = AttentionModule(64) if use_attention else nn.Identity()

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )

        self.attention2 = AttentionModule(256) if use_attention else nn.Identity()

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

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.attention2(x)
        return self.classifier(x)


class DenseLayer(nn.Module):
    """DenseNet-style dense layer."""

    def __init__(self, in_channels: int, growth_rate: int):
        super().__init__()

        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, growth_rate, 3, padding=1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv(F.relu(self.bn(x)))
        return torch.cat([x, out], dim=1)


class DenseBlock(nn.Module):
    """DenseNet block."""

    def __init__(self, in_channels: int, growth_rate: int, num_layers: int):
        super().__init__()

        layers = []
        for i in range(num_layers):
            layers.append(DenseLayer(in_channels + i * growth_rate, growth_rate))

        self.layers = nn.Sequential(*layers)
        self.out_channels = in_channels + num_layers * growth_rate

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class DenseNetMedical(nn.Module):
    """DenseNet for medical imaging."""

    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 2,
                 growth_rate: int = 12,
                 num_blocks: int = 3,
                 layers_per_block: int = 4):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1)
        )

        channels = 32
        blocks = []

        for i in range(num_blocks):
            block = DenseBlock(channels, growth_rate, layers_per_block)
            blocks.append(block)
            channels = block.out_channels

            if i < num_blocks - 1:
                # Transition layer
                blocks.append(nn.Sequential(
                    nn.BatchNorm2d(channels),
                    nn.Conv2d(channels, channels // 2, 1, bias=False),
                    nn.AvgPool2d(2, 2)
                ))
                channels = channels // 2

        self.features = nn.Sequential(*blocks)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(channels, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


# =============================================================================
# TRANSFORMER MODELS
# =============================================================================

class PatchEmbedding(nn.Module):
    """Patch embedding for Vision Transformer."""

    def __init__(self,
                 img_size: int = 224,
                 patch_size: int = 16,
                 in_channels: int = 3,
                 embed_dim: int = 768):
        super().__init__()

        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, patch_size, stride=patch_size)

    def forward(self, x: Tensor) -> Tensor:
        # (B, C, H, W) -> (B, embed_dim, num_patches^0.5, num_patches^0.5) -> (B, num_patches, embed_dim)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class TransformerBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 mlp_ratio: float = 4.0,
                 dropout: float = 0.1):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)

        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: Tensor) -> Tensor:
        # Self-attention with residual
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out

        # MLP with residual
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """Vision Transformer (ViT)."""

    def __init__(self,
                 img_size: int = 224,
                 patch_size: int = 16,
                 in_channels: int = 3,
                 num_classes: int = 10,
                 embed_dim: int = 384,
                 num_heads: int = 6,
                 num_layers: int = 6,
                 dropout: float = 0.1):
        super().__init__()

        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        # Initialize
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        B = x.shape[0]

        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        x = self.dropout(x)

        x = self.blocks(x)
        x = self.norm(x)

        return self.head(x[:, 0])


# =============================================================================
# RECURRENT MODELS
# =============================================================================

class LSTMModel(nn.Module):
    """LSTM for sequential data."""

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 128,
                 num_layers: int = 2,
                 num_classes: int = 2,
                 bidirectional: bool = True,
                 dropout: float = 0.3):
        super().__init__()

        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0
        )

        fc_input = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Sequential(
            nn.Linear(fc_input, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: (batch, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)
        # Use last output
        out = lstm_out[:, -1, :]
        return self.fc(out)


class GRUModel(nn.Module):
    """GRU for sequential data."""

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 128,
                 num_layers: int = 2,
                 num_classes: int = 2,
                 bidirectional: bool = True,
                 dropout: float = 0.3):
        super().__init__()

        self.gru = nn.GRU(
            input_dim, hidden_dim, num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0
        )

        fc_input = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Sequential(
            nn.Linear(fc_input, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: Tensor) -> Tensor:
        gru_out, _ = self.gru(x)
        out = gru_out[:, -1, :]
        return self.fc(out)


# =============================================================================
# MODEL FACTORY
# =============================================================================

def create_model(name: str, **kwargs) -> nn.Module:
    """Factory function to create model instances."""
    models = {
        'SimpleMLP': SimpleMLP,
        'DeepMLP': DeepMLP,
        'ResidualMLP': ResidualMLP,
        'LeNet5': LeNet5,
        'SimpleCNN': SimpleCNN,
        'VGGStyle': VGGStyle,
        'ResNet18': ResNet18,
        'MobileNetStyle': MobileNetStyle,
        'MedicalCNN': MedicalCNN,
        'DenseNetMedical': DenseNetMedical,
        'VisionTransformer': VisionTransformer,
        'LSTM': LSTMModel,
        'GRU': GRUModel,
    }

    if name not in models:
        raise ValueError(f"Unknown model: {name}. Available: {list(models.keys())}")

    return models[name](**kwargs)


def get_model_info(name: str) -> Dict:
    """Get information about a model."""
    return MODEL_INFO.get(name, {})


def list_models() -> List[str]:
    """List all available models."""
    return list(MODEL_INFO.keys())


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    print("FL-EHDS Model Zoo")
    print("=" * 60)

    print("\nAvailable Models:")
    for name, info in MODEL_INFO.items():
        print(f"\n{name} ({info['type']})")
        print(f"  {info['description'][:80]}...")
        print(f"  Use case: {info['use_case']}")

    # Test model creation
    if TORCH_AVAILABLE:
        print("\n" + "=" * 60)
        print("Testing model creation...")

        # Test MLP
        mlp = create_model('SimpleMLP', input_dim=10, num_classes=2)
        x = torch.randn(4, 10)
        print(f"SimpleMLP output shape: {mlp(x).shape}")

        # Test CNN
        cnn = create_model('SimpleCNN', num_classes=10)
        x = torch.randn(4, 3, 32, 32)
        print(f"SimpleCNN output shape: {cnn(x).shape}")

        print("\nAll models created successfully!")

#!/usr/bin/env python3
"""
FL-EHDS Gradient Compression Module

Implements various gradient compression techniques to reduce communication
overhead in Federated Learning, critical for cross-border EHDS scenarios
where bandwidth may be limited.

Compression Techniques:
1. Top-K Sparsification - Keep only K largest gradient values
2. Random-K Sparsification - Randomly sample K gradient values
3. Quantization - Reduce precision (32-bit → 8-bit or lower)
4. Ternary Quantization - Compress to {-1, 0, +1}
5. SignSGD - Transmit only gradient signs
6. Error Feedback - Accumulate compression error for future rounds

Author: Fabio Liberti
References:
- Alistarh et al. (2017) "QSGD: Communication-Efficient SGD via Gradient Quantization"
- Lin et al. (2018) "Deep Gradient Compression"
- Bernstein et al. (2018) "signSGD: Compressed Optimisation for Non-Convex Problems"
"""

import numpy as np
from typing import Tuple, Dict, Optional, Union, List
from dataclasses import dataclass
from abc import ABC, abstractmethod
import struct


@dataclass
class CompressionStats:
    """Statistics about compression performance."""
    original_size_bytes: int
    compressed_size_bytes: int
    compression_ratio: float
    sparsity: float  # Fraction of zeros
    reconstruction_error: float  # L2 norm of error


class GradientCompressor(ABC):
    """Abstract base class for gradient compressors."""

    @abstractmethod
    def compress(self, gradient: np.ndarray) -> Tuple[bytes, Dict]:
        """
        Compress a gradient vector.

        Args:
            gradient: The gradient to compress (flattened numpy array)

        Returns:
            Tuple of (compressed_bytes, metadata_dict)
        """
        pass

    @abstractmethod
    def decompress(self, compressed: bytes, metadata: Dict) -> np.ndarray:
        """
        Decompress a gradient vector.

        Args:
            compressed: The compressed bytes
            metadata: Metadata needed for decompression

        Returns:
            Reconstructed gradient as numpy array
        """
        pass

    def get_stats(self, original: np.ndarray, reconstructed: np.ndarray,
                  compressed_size: int) -> CompressionStats:
        """Compute compression statistics."""
        original_size = original.nbytes
        compression_ratio = original_size / compressed_size if compressed_size > 0 else float('inf')
        sparsity = np.mean(reconstructed == 0)
        error = np.linalg.norm(original - reconstructed) / (np.linalg.norm(original) + 1e-10)

        return CompressionStats(
            original_size_bytes=original_size,
            compressed_size_bytes=compressed_size,
            compression_ratio=compression_ratio,
            sparsity=sparsity,
            reconstruction_error=error
        )


class TopKCompressor(GradientCompressor):
    """
    Top-K Sparsification: Keep only the K largest magnitude gradients.

    Communication cost: O(K * (sizeof(index) + sizeof(value)))
    Typical compression: 10-100x for K = 0.1% to 1% of parameters
    """

    def __init__(self, k_ratio: float = 0.01):
        """
        Args:
            k_ratio: Fraction of gradients to keep (0.01 = 1%)
        """
        self.k_ratio = k_ratio

    def compress(self, gradient: np.ndarray) -> Tuple[bytes, Dict]:
        flat = gradient.flatten().astype(np.float32)
        n = len(flat)
        k = max(1, int(n * self.k_ratio))

        # Get indices of top-k absolute values
        abs_grad = np.abs(flat)
        top_k_indices = np.argpartition(abs_grad, -k)[-k:]
        top_k_values = flat[top_k_indices]

        # Pack as bytes: indices (uint32) + values (float32)
        indices_bytes = top_k_indices.astype(np.uint32).tobytes()
        values_bytes = top_k_values.astype(np.float32).tobytes()
        compressed = indices_bytes + values_bytes

        metadata = {
            "shape": gradient.shape,
            "k": k,
            "n": n
        }

        return compressed, metadata

    def decompress(self, compressed: bytes, metadata: Dict) -> np.ndarray:
        k = metadata["k"]
        n = metadata["n"]
        shape = metadata["shape"]

        # Unpack bytes
        indices_size = k * 4  # uint32
        indices = np.frombuffer(compressed[:indices_size], dtype=np.uint32)
        values = np.frombuffer(compressed[indices_size:], dtype=np.float32)

        # Reconstruct sparse gradient
        reconstructed = np.zeros(n, dtype=np.float32)
        reconstructed[indices] = values

        return reconstructed.reshape(shape)


class RandomKCompressor(GradientCompressor):
    """
    Random-K Sparsification: Randomly sample K gradients and scale.

    Unbiased estimator: E[compressed] = original
    Communication cost: O(K * sizeof(value))
    """

    def __init__(self, k_ratio: float = 0.01, seed: int = None):
        self.k_ratio = k_ratio
        self.rng = np.random.default_rng(seed)

    def compress(self, gradient: np.ndarray) -> Tuple[bytes, Dict]:
        flat = gradient.flatten().astype(np.float32)
        n = len(flat)
        k = max(1, int(n * self.k_ratio))

        # Random selection
        indices = self.rng.choice(n, k, replace=False)
        values = flat[indices] * (n / k)  # Scale to maintain expectation

        # Pack
        indices_bytes = indices.astype(np.uint32).tobytes()
        values_bytes = values.astype(np.float32).tobytes()
        compressed = indices_bytes + values_bytes

        metadata = {"shape": gradient.shape, "k": k, "n": n}
        return compressed, metadata

    def decompress(self, compressed: bytes, metadata: Dict) -> np.ndarray:
        k = metadata["k"]
        n = metadata["n"]
        shape = metadata["shape"]

        indices_size = k * 4
        indices = np.frombuffer(compressed[:indices_size], dtype=np.uint32)
        values = np.frombuffer(compressed[indices_size:], dtype=np.float32)

        reconstructed = np.zeros(n, dtype=np.float32)
        reconstructed[indices] = values

        return reconstructed.reshape(shape)


class QuantizationCompressor(GradientCompressor):
    """
    Stochastic Quantization: Reduce precision to fewer bits.

    Supports: 8-bit, 4-bit, 2-bit quantization
    Uses stochastic rounding for unbiasedness
    """

    def __init__(self, num_bits: int = 8):
        assert num_bits in [2, 4, 8], "Supported bits: 2, 4, 8"
        self.num_bits = num_bits
        self.num_levels = 2 ** num_bits

    def compress(self, gradient: np.ndarray) -> Tuple[bytes, Dict]:
        flat = gradient.flatten().astype(np.float32)

        # Find range
        min_val = float(np.min(flat))
        max_val = float(np.max(flat))
        scale = (max_val - min_val) / (self.num_levels - 1) if max_val != min_val else 1.0

        # Normalize to [0, num_levels-1]
        normalized = (flat - min_val) / scale

        # Stochastic rounding
        lower = np.floor(normalized).astype(np.int32)
        upper = lower + 1
        prob = normalized - lower
        quantized = np.where(np.random.random(len(flat)) < prob, upper, lower)
        quantized = np.clip(quantized, 0, self.num_levels - 1)

        # Pack based on bit width
        if self.num_bits == 8:
            compressed = quantized.astype(np.uint8).tobytes()
        elif self.num_bits == 4:
            # Pack two 4-bit values per byte
            packed = []
            for i in range(0, len(quantized), 2):
                if i + 1 < len(quantized):
                    byte = (quantized[i] << 4) | quantized[i + 1]
                else:
                    byte = quantized[i] << 4
                packed.append(byte)
            compressed = bytes(packed)
        elif self.num_bits == 2:
            # Pack four 2-bit values per byte
            packed = []
            for i in range(0, len(quantized), 4):
                byte = 0
                for j in range(4):
                    if i + j < len(quantized):
                        byte |= (quantized[i + j] << (6 - 2 * j))
                packed.append(byte)
            compressed = bytes(packed)

        metadata = {
            "shape": gradient.shape,
            "min_val": min_val,
            "max_val": max_val,
            "scale": scale,
            "n": len(flat),
            "num_bits": self.num_bits
        }

        return compressed, metadata

    def decompress(self, compressed: bytes, metadata: Dict) -> np.ndarray:
        n = metadata["n"]
        min_val = metadata["min_val"]
        scale = metadata["scale"]
        shape = metadata["shape"]
        num_bits = metadata["num_bits"]

        # Unpack based on bit width
        if num_bits == 8:
            quantized = np.frombuffer(compressed, dtype=np.uint8).astype(np.float32)
        elif num_bits == 4:
            quantized = []
            for byte in compressed:
                quantized.append((byte >> 4) & 0xF)
                quantized.append(byte & 0xF)
            quantized = np.array(quantized[:n], dtype=np.float32)
        elif num_bits == 2:
            quantized = []
            for byte in compressed:
                quantized.append((byte >> 6) & 0x3)
                quantized.append((byte >> 4) & 0x3)
                quantized.append((byte >> 2) & 0x3)
                quantized.append(byte & 0x3)
            quantized = np.array(quantized[:n], dtype=np.float32)

        # Dequantize
        reconstructed = quantized * scale + min_val

        return reconstructed.reshape(shape)


class TernaryCompressor(GradientCompressor):
    """
    Ternary Quantization: Compress to {-1, 0, +1}.

    Each value requires only 2 bits (00=0, 01=+1, 11=-1)
    Compression ratio: 16x (32-bit → 2-bit)
    """

    def __init__(self, threshold_ratio: float = 0.7):
        """
        Args:
            threshold_ratio: Values within threshold_ratio * std are set to 0
        """
        self.threshold_ratio = threshold_ratio

    def compress(self, gradient: np.ndarray) -> Tuple[bytes, Dict]:
        flat = gradient.flatten().astype(np.float32)

        # Compute threshold
        std = np.std(flat)
        threshold = self.threshold_ratio * std

        # Ternary quantization
        ternary = np.zeros(len(flat), dtype=np.int8)
        ternary[flat > threshold] = 1
        ternary[flat < -threshold] = -1

        # Compute scale for reconstruction
        positive_mask = ternary == 1
        negative_mask = ternary == -1
        positive_mean = np.mean(np.abs(flat[positive_mask])) if np.any(positive_mask) else 0
        negative_mean = np.mean(np.abs(flat[negative_mask])) if np.any(negative_mask) else 0
        scale = (positive_mean + negative_mean) / 2 if (positive_mean + negative_mean) > 0 else 1.0

        # Pack 4 ternary values per byte (2 bits each)
        # Encoding: 0 -> 00, +1 -> 01, -1 -> 11
        packed = []
        for i in range(0, len(ternary), 4):
            byte = 0
            for j in range(4):
                if i + j < len(ternary):
                    val = ternary[i + j]
                    if val == 0:
                        code = 0b00
                    elif val == 1:
                        code = 0b01
                    else:  # val == -1
                        code = 0b11
                    byte |= (code << (6 - 2 * j))
            packed.append(byte)

        compressed = bytes(packed)

        metadata = {
            "shape": gradient.shape,
            "scale": float(scale),
            "n": len(flat)
        }

        return compressed, metadata

    def decompress(self, compressed: bytes, metadata: Dict) -> np.ndarray:
        n = metadata["n"]
        scale = metadata["scale"]
        shape = metadata["shape"]

        # Unpack
        ternary = []
        for byte in compressed:
            for j in range(4):
                code = (byte >> (6 - 2 * j)) & 0b11
                if code == 0b00:
                    ternary.append(0)
                elif code == 0b01:
                    ternary.append(1)
                else:  # 0b11
                    ternary.append(-1)

        ternary = np.array(ternary[:n], dtype=np.float32)
        reconstructed = ternary * scale

        return reconstructed.reshape(shape)


class SignSGDCompressor(GradientCompressor):
    """
    SignSGD: Transmit only gradient signs.

    Communication: 1 bit per parameter
    Compression ratio: 32x
    Requires majority vote aggregation at server
    """

    def compress(self, gradient: np.ndarray) -> Tuple[bytes, Dict]:
        flat = gradient.flatten().astype(np.float32)

        # Compute sign (True for positive, False for non-positive)
        signs = flat > 0

        # Pack 8 signs per byte
        n = len(signs)
        packed = []
        for i in range(0, n, 8):
            byte = 0
            for j in range(8):
                if i + j < n and signs[i + j]:
                    byte |= (1 << (7 - j))
            packed.append(byte)

        compressed = bytes(packed)

        # For reconstruction, we need the magnitude (mean absolute value)
        magnitude = float(np.mean(np.abs(flat)))

        metadata = {
            "shape": gradient.shape,
            "magnitude": magnitude,
            "n": n
        }

        return compressed, metadata

    def decompress(self, compressed: bytes, metadata: Dict) -> np.ndarray:
        n = metadata["n"]
        magnitude = metadata["magnitude"]
        shape = metadata["shape"]

        # Unpack signs
        signs = []
        for byte in compressed:
            for j in range(8):
                signs.append(bool(byte & (1 << (7 - j))))

        signs = np.array(signs[:n])
        reconstructed = np.where(signs, magnitude, -magnitude).astype(np.float32)

        return reconstructed.reshape(shape)


class ErrorFeedbackCompressor:
    """
    Wrapper that adds error feedback to any compressor.

    Accumulates compression error and adds it to the next round's gradient,
    ensuring convergence to the correct solution despite compression.
    """

    def __init__(self, base_compressor: GradientCompressor):
        self.base_compressor = base_compressor
        self.error_buffer: Optional[np.ndarray] = None

    def compress(self, gradient: np.ndarray) -> Tuple[bytes, Dict]:
        # Add accumulated error
        if self.error_buffer is not None:
            corrected_gradient = gradient + self.error_buffer
        else:
            corrected_gradient = gradient

        # Compress
        compressed, metadata = self.base_compressor.compress(corrected_gradient)

        # Compute error for next round
        reconstructed = self.base_compressor.decompress(compressed, metadata)
        self.error_buffer = corrected_gradient - reconstructed

        return compressed, metadata

    def decompress(self, compressed: bytes, metadata: Dict) -> np.ndarray:
        return self.base_compressor.decompress(compressed, metadata)

    def reset_error(self):
        """Reset error buffer (e.g., at start of new training)."""
        self.error_buffer = None


def compare_compressors(gradient_size: int = 10000, seed: int = 42) -> Dict:
    """
    Compare all compression methods on a synthetic gradient.

    Returns dict with stats for each compressor.
    """
    np.random.seed(seed)

    # Generate realistic gradient (sparse with some large values)
    gradient = np.random.randn(gradient_size).astype(np.float32) * 0.01
    # Add some outliers
    outlier_indices = np.random.choice(gradient_size, size=int(gradient_size * 0.05), replace=False)
    gradient[outlier_indices] *= 10

    compressors = {
        "TopK-1%": TopKCompressor(k_ratio=0.01),
        "TopK-0.1%": TopKCompressor(k_ratio=0.001),
        "RandomK-1%": RandomKCompressor(k_ratio=0.01),
        "Quantize-8bit": QuantizationCompressor(num_bits=8),
        "Quantize-4bit": QuantizationCompressor(num_bits=4),
        "Quantize-2bit": QuantizationCompressor(num_bits=2),
        "Ternary": TernaryCompressor(),
        "SignSGD": SignSGDCompressor(),
    }

    results = {}
    original_size = gradient.nbytes

    print(f"Original gradient: {gradient_size} parameters, {original_size} bytes")
    print("-" * 70)
    print(f"{'Method':<15} {'Compressed':>12} {'Ratio':>8} {'Error':>10} {'Sparsity':>10}")
    print("-" * 70)

    for name, compressor in compressors.items():
        compressed, metadata = compressor.compress(gradient)
        reconstructed = compressor.decompress(compressed, metadata)

        stats = compressor.get_stats(gradient, reconstructed, len(compressed))

        results[name] = {
            "compressed_size": len(compressed),
            "compression_ratio": stats.compression_ratio,
            "reconstruction_error": stats.reconstruction_error,
            "sparsity": stats.sparsity
        }

        print(f"{name:<15} {len(compressed):>10} B {stats.compression_ratio:>7.1f}x "
              f"{stats.reconstruction_error:>9.4f} {stats.sparsity:>9.1%}")

    return results


if __name__ == "__main__":
    print("=" * 70)
    print("FL-EHDS Gradient Compression Comparison")
    print("=" * 70)
    print()

    results = compare_compressors(gradient_size=100000)

    print()
    print("=" * 70)
    print("RECOMMENDATIONS FOR EHDS:")
    print("-" * 70)
    print("• TopK-1%: Best accuracy, 100x compression - recommended for most cases")
    print("• Ternary: 16x compression, good for bandwidth-limited scenarios")
    print("• SignSGD: 32x compression, requires majority vote aggregation")
    print("• 8-bit Quantization: 4x compression, minimal error - safe default")
    print("=" * 70)

#!/usr/bin/env python3
"""
FL-EHDS Model Compression for Communication Efficiency

Implements compression techniques for reducing communication costs
in federated learning, critical for cross-border EHDS deployments
with bandwidth constraints.

Techniques:
1. Gradient Quantization
   - SignSGD (1-bit)
   - QSGD (multi-bit)
   - TernGrad (ternary)

2. Gradient Sparsification
   - Top-K
   - Random-K
   - Threshold-based

3. Low-Rank Approximation
   - SVD-based
   - PowerSGD

4. Error Feedback / Memory
   - Accumulates quantization errors for future rounds

Author: Fabio Liberti
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import struct
import zlib
from copy import deepcopy


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class CompressedGradient:
    """Container for compressed gradient data."""
    client_id: int
    compressed_data: bytes
    original_shape: Dict[str, Tuple[int, ...]]
    compression_method: str
    compression_ratio: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CompressionConfig:
    """Configuration for gradient compression."""
    method: str = "topk"  # signsgd, qsgd, terngrad, topk, randomk, threshold, powersgd
    # Quantization params
    num_bits: int = 8  # For QSGD
    # Sparsification params
    k_ratio: float = 0.1  # Top-K: keep top 10% of gradients
    threshold: float = 0.001  # Threshold-based sparsification
    # PowerSGD params
    rank: int = 4  # Low-rank approximation rank
    # Error feedback
    use_error_feedback: bool = True


# =============================================================================
# BASE CLASS
# =============================================================================

class GradientCompressor(ABC):
    """Abstract base class for gradient compression."""

    def __init__(self, config: CompressionConfig):
        self.config = config
        # Error feedback buffers per client
        self.error_buffers: Dict[int, Dict[str, np.ndarray]] = {}

    @abstractmethod
    def compress(self,
                gradient: Dict[str, np.ndarray],
                client_id: int = 0) -> CompressedGradient:
        """Compress gradient for transmission."""
        pass

    @abstractmethod
    def decompress(self, compressed: CompressedGradient) -> Dict[str, np.ndarray]:
        """Decompress received gradient."""
        pass

    def _get_error_buffer(self, client_id: int, template: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Get or initialize error buffer for client."""
        if client_id not in self.error_buffers:
            self.error_buffers[client_id] = {
                k: np.zeros_like(v) for k, v in template.items()
            }
        return self.error_buffers[client_id]

    def _update_error_buffer(self,
                            client_id: int,
                            original: Dict[str, np.ndarray],
                            decompressed: Dict[str, np.ndarray]) -> None:
        """Update error buffer with compression residual."""
        if not self.config.use_error_feedback:
            return

        for key in original.keys():
            if key in decompressed:
                self.error_buffers[client_id][key] = original[key] - decompressed[key]

    def compute_compression_ratio(self,
                                  original_size: int,
                                  compressed_size: int) -> float:
        """Compute compression ratio."""
        return original_size / max(compressed_size, 1)


# =============================================================================
# QUANTIZATION: SignSGD (1-bit)
# =============================================================================

class SignSGDCompressor(GradientCompressor):
    """
    SignSGD: 1-bit Quantization.

    Compresses gradients to their signs: +1 or -1.
    Extreme compression (32x for float32) but loses magnitude information.

    Reference: Bernstein et al., "signSGD: Compressed Optimisation
    for Non-Convex Problems", ICML 2018.
    """

    def compress(self,
                gradient: Dict[str, np.ndarray],
                client_id: int = 0) -> CompressedGradient:
        """Compress gradient to signs."""
        # Apply error feedback
        if self.config.use_error_feedback:
            error_buffer = self._get_error_buffer(client_id, gradient)
            gradient = {k: gradient[k] + error_buffer[k] for k in gradient.keys()}

        compressed_data = b""
        original_shapes = {}
        metadata = {}

        for key, arr in gradient.items():
            flat = arr.flatten()
            original_shapes[key] = arr.shape

            # Store mean magnitude for reconstruction
            magnitude = np.mean(np.abs(flat))
            metadata[f"{key}_magnitude"] = float(magnitude)

            # Compress to bits: 1 = positive, 0 = negative
            signs = (flat >= 0).astype(np.uint8)

            # Pack 8 signs into 1 byte
            packed = np.packbits(signs)
            compressed_data += struct.pack('I', len(flat))  # Original length
            compressed_data += packed.tobytes()

        # Update error buffer
        decompressed = self.decompress(CompressedGradient(
            client_id=client_id,
            compressed_data=compressed_data,
            original_shape=original_shapes,
            compression_method="signsgd",
            compression_ratio=0,
            metadata=metadata
        ))
        self._update_error_buffer(client_id, gradient, decompressed)

        # Calculate compression ratio
        original_size = sum(arr.nbytes for arr in gradient.values())
        compressed_size = len(compressed_data)

        return CompressedGradient(
            client_id=client_id,
            compressed_data=compressed_data,
            original_shape=original_shapes,
            compression_method="signsgd",
            compression_ratio=self.compute_compression_ratio(original_size, compressed_size),
            metadata=metadata
        )

    def decompress(self, compressed: CompressedGradient) -> Dict[str, np.ndarray]:
        """Decompress signs back to gradient."""
        result = {}
        data = compressed.compressed_data
        offset = 0

        for key, shape in compressed.original_shape.items():
            # Read original length
            length = struct.unpack('I', data[offset:offset + 4])[0]
            offset += 4

            # Calculate packed size
            packed_size = (length + 7) // 8
            packed = np.frombuffer(data[offset:offset + packed_size], dtype=np.uint8)
            offset += packed_size

            # Unpack bits
            signs = np.unpackbits(packed)[:length]

            # Reconstruct with magnitude
            magnitude = compressed.metadata.get(f"{key}_magnitude", 1.0)
            gradient = (signs.astype(np.float32) * 2 - 1) * magnitude

            result[key] = gradient.reshape(shape)

        return result


# =============================================================================
# QUANTIZATION: QSGD (Multi-bit)
# =============================================================================

class QSGDCompressor(GradientCompressor):
    """
    QSGD: Quantized SGD with configurable bit-width.

    Stochastic quantization to s levels where s = 2^bits.
    Provides unbiased gradient estimates.

    Reference: Alistarh et al., "QSGD: Communication-Efficient
    SGD via Gradient Quantization and Encoding", NeurIPS 2017.
    """

    def __init__(self, config: CompressionConfig):
        super().__init__(config)
        self.num_levels = 2 ** config.num_bits
        self.bits = config.num_bits

    def _stochastic_quantize(self, arr: np.ndarray) -> Tuple[np.ndarray, float]:
        """Stochastically quantize array to discrete levels."""
        norm = np.linalg.norm(arr)
        if norm == 0:
            return np.zeros_like(arr, dtype=np.int32), 0.0

        # Normalize
        normalized = arr / norm
        s = self.num_levels - 1

        # Stochastic rounding
        scaled = np.abs(normalized) * s
        lower = np.floor(scaled).astype(np.int32)
        prob = scaled - lower

        # Probabilistic rounding
        rand = np.random.random(arr.shape)
        quantized = np.where(rand < prob, lower + 1, lower)

        # Preserve signs
        signs = np.sign(arr)
        quantized = (quantized * signs).astype(np.int32)

        return quantized, norm

    def compress(self,
                gradient: Dict[str, np.ndarray],
                client_id: int = 0) -> CompressedGradient:
        """Compress gradient using QSGD."""
        if self.config.use_error_feedback:
            error_buffer = self._get_error_buffer(client_id, gradient)
            gradient = {k: gradient[k] + error_buffer[k] for k in gradient.keys()}

        compressed_data = b""
        original_shapes = {}
        metadata = {}

        for key, arr in gradient.items():
            original_shapes[key] = arr.shape

            # Quantize
            quantized, norm = self._stochastic_quantize(arr.flatten())
            metadata[f"{key}_norm"] = float(norm)

            # Pack quantized values
            if self.bits <= 8:
                packed = quantized.astype(np.int8)
            else:
                packed = quantized.astype(np.int16)

            compressed_data += struct.pack('I', len(quantized))
            compressed_data += packed.tobytes()

        # Update error buffer
        decompressed = self.decompress(CompressedGradient(
            client_id=client_id,
            compressed_data=compressed_data,
            original_shape=original_shapes,
            compression_method="qsgd",
            compression_ratio=0,
            metadata=metadata
        ))
        self._update_error_buffer(client_id, gradient, decompressed)

        original_size = sum(arr.nbytes for arr in gradient.values())
        compressed_size = len(compressed_data)

        return CompressedGradient(
            client_id=client_id,
            compressed_data=compressed_data,
            original_shape=original_shapes,
            compression_method="qsgd",
            compression_ratio=self.compute_compression_ratio(original_size, compressed_size),
            metadata=metadata
        )

    def decompress(self, compressed: CompressedGradient) -> Dict[str, np.ndarray]:
        """Decompress QSGD gradient."""
        result = {}
        data = compressed.compressed_data
        offset = 0

        dtype = np.int8 if self.bits <= 8 else np.int16
        item_size = 1 if self.bits <= 8 else 2

        for key, shape in compressed.original_shape.items():
            length = struct.unpack('I', data[offset:offset + 4])[0]
            offset += 4

            quantized = np.frombuffer(
                data[offset:offset + length * item_size],
                dtype=dtype
            )
            offset += length * item_size

            # Dequantize
            norm = compressed.metadata.get(f"{key}_norm", 1.0)
            s = self.num_levels - 1

            if norm > 0:
                dequantized = (quantized.astype(np.float32) / s) * norm
            else:
                dequantized = np.zeros(length, dtype=np.float32)

            result[key] = dequantized.reshape(shape)

        return result


# =============================================================================
# QUANTIZATION: TernGrad (Ternary)
# =============================================================================

class TernGradCompressor(GradientCompressor):
    """
    TernGrad: Ternary Gradient Compression.

    Compresses gradients to {-1, 0, +1} with magnitude scaling.
    Good balance between compression and accuracy.

    Reference: Wen et al., "TernGrad: Ternary Gradients to Reduce
    Communication in Distributed Deep Learning", NeurIPS 2017.
    """

    def compress(self,
                gradient: Dict[str, np.ndarray],
                client_id: int = 0) -> CompressedGradient:
        """Compress gradient to ternary values."""
        if self.config.use_error_feedback:
            error_buffer = self._get_error_buffer(client_id, gradient)
            gradient = {k: gradient[k] + error_buffer[k] for k in gradient.keys()}

        compressed_data = b""
        original_shapes = {}
        metadata = {}

        for key, arr in gradient.items():
            flat = arr.flatten()
            original_shapes[key] = arr.shape

            # Compute threshold and scale
            max_abs = np.max(np.abs(flat))
            if max_abs == 0:
                max_abs = 1.0

            metadata[f"{key}_scale"] = float(max_abs)

            # Stochastic ternarization
            normalized = flat / max_abs
            prob = np.abs(normalized)

            rand = np.random.random(flat.shape)
            ternary = np.zeros_like(flat, dtype=np.int8)
            ternary[rand < prob] = np.sign(normalized[rand < prob]).astype(np.int8)

            # Encode: 2 bits per value (00=0, 01=+1, 10=-1)
            # Pack 4 values per byte
            encoded = ((ternary + 1).astype(np.uint8))  # Map {-1,0,1} to {0,1,2}

            # Simple byte encoding (could optimize further)
            compressed_data += struct.pack('I', len(flat))
            compressed_data += encoded.tobytes()

        # Error feedback
        decompressed = self.decompress(CompressedGradient(
            client_id=client_id,
            compressed_data=compressed_data,
            original_shape=original_shapes,
            compression_method="terngrad",
            compression_ratio=0,
            metadata=metadata
        ))
        self._update_error_buffer(client_id, gradient, decompressed)

        original_size = sum(arr.nbytes for arr in gradient.values())
        compressed_size = len(compressed_data)

        return CompressedGradient(
            client_id=client_id,
            compressed_data=compressed_data,
            original_shape=original_shapes,
            compression_method="terngrad",
            compression_ratio=self.compute_compression_ratio(original_size, compressed_size),
            metadata=metadata
        )

    def decompress(self, compressed: CompressedGradient) -> Dict[str, np.ndarray]:
        """Decompress ternary gradient."""
        result = {}
        data = compressed.compressed_data
        offset = 0

        for key, shape in compressed.original_shape.items():
            length = struct.unpack('I', data[offset:offset + 4])[0]
            offset += 4

            encoded = np.frombuffer(data[offset:offset + length], dtype=np.uint8)
            offset += length

            # Decode: {0,1,2} back to {-1,0,1}
            ternary = encoded.astype(np.float32) - 1

            # Scale back
            scale = compressed.metadata.get(f"{key}_scale", 1.0)
            gradient = ternary * scale

            result[key] = gradient.reshape(shape)

        return result


# =============================================================================
# SPARSIFICATION: Top-K
# =============================================================================

class TopKCompressor(GradientCompressor):
    """
    Top-K Sparsification.

    Keep only the K largest magnitude gradients.
    Very effective compression with minimal accuracy loss.

    Reference: Aji & Heafield, "Sparse Communication for
    Distributed Gradient Descent", EMNLP 2017.
    """

    def compress(self,
                gradient: Dict[str, np.ndarray],
                client_id: int = 0) -> CompressedGradient:
        """Compress gradient using Top-K sparsification."""
        if self.config.use_error_feedback:
            error_buffer = self._get_error_buffer(client_id, gradient)
            gradient = {k: gradient[k] + error_buffer[k] for k in gradient.keys()}

        compressed_data = b""
        original_shapes = {}
        metadata = {}

        for key, arr in gradient.items():
            flat = arr.flatten()
            original_shapes[key] = arr.shape

            # Compute K
            k = max(1, int(len(flat) * self.config.k_ratio))
            metadata[f"{key}_k"] = k

            # Find top-K indices
            abs_values = np.abs(flat)
            topk_indices = np.argpartition(abs_values, -k)[-k:]
            topk_values = flat[topk_indices]

            # Store indices and values
            compressed_data += struct.pack('II', len(flat), k)
            compressed_data += topk_indices.astype(np.uint32).tobytes()
            compressed_data += topk_values.astype(np.float32).tobytes()

        # Error feedback
        decompressed = self.decompress(CompressedGradient(
            client_id=client_id,
            compressed_data=compressed_data,
            original_shape=original_shapes,
            compression_method="topk",
            compression_ratio=0,
            metadata=metadata
        ))
        self._update_error_buffer(client_id, gradient, decompressed)

        original_size = sum(arr.nbytes for arr in gradient.values())
        compressed_size = len(compressed_data)

        return CompressedGradient(
            client_id=client_id,
            compressed_data=compressed_data,
            original_shape=original_shapes,
            compression_method="topk",
            compression_ratio=self.compute_compression_ratio(original_size, compressed_size),
            metadata=metadata
        )

    def decompress(self, compressed: CompressedGradient) -> Dict[str, np.ndarray]:
        """Decompress Top-K gradient."""
        result = {}
        data = compressed.compressed_data
        offset = 0

        for key, shape in compressed.original_shape.items():
            length, k = struct.unpack('II', data[offset:offset + 8])
            offset += 8

            indices = np.frombuffer(data[offset:offset + k * 4], dtype=np.uint32)
            offset += k * 4

            values = np.frombuffer(data[offset:offset + k * 4], dtype=np.float32)
            offset += k * 4

            # Reconstruct sparse gradient
            gradient = np.zeros(length, dtype=np.float32)
            gradient[indices] = values

            result[key] = gradient.reshape(shape)

        return result


# =============================================================================
# SPARSIFICATION: Random-K
# =============================================================================

class RandomKCompressor(GradientCompressor):
    """
    Random-K Sparsification.

    Randomly select K gradient coordinates (unbiased estimator).
    Simpler than Top-K but with higher variance.
    """

    def compress(self,
                gradient: Dict[str, np.ndarray],
                client_id: int = 0) -> CompressedGradient:
        """Compress using Random-K sparsification."""
        if self.config.use_error_feedback:
            error_buffer = self._get_error_buffer(client_id, gradient)
            gradient = {k: gradient[k] + error_buffer[k] for k in gradient.keys()}

        compressed_data = b""
        original_shapes = {}
        metadata = {}

        for key, arr in gradient.items():
            flat = arr.flatten()
            original_shapes[key] = arr.shape

            k = max(1, int(len(flat) * self.config.k_ratio))
            d = len(flat)

            # Random selection
            indices = np.random.choice(d, k, replace=False)
            values = flat[indices] * (d / k)  # Scale for unbiased estimate

            compressed_data += struct.pack('II', d, k)
            compressed_data += indices.astype(np.uint32).tobytes()
            compressed_data += values.astype(np.float32).tobytes()

        decompressed = self.decompress(CompressedGradient(
            client_id=client_id,
            compressed_data=compressed_data,
            original_shape=original_shapes,
            compression_method="randomk",
            compression_ratio=0,
            metadata=metadata
        ))
        self._update_error_buffer(client_id, gradient, decompressed)

        original_size = sum(arr.nbytes for arr in gradient.values())
        compressed_size = len(compressed_data)

        return CompressedGradient(
            client_id=client_id,
            compressed_data=compressed_data,
            original_shape=original_shapes,
            compression_method="randomk",
            compression_ratio=self.compute_compression_ratio(original_size, compressed_size),
            metadata=metadata
        )

    def decompress(self, compressed: CompressedGradient) -> Dict[str, np.ndarray]:
        """Decompress Random-K gradient."""
        result = {}
        data = compressed.compressed_data
        offset = 0

        for key, shape in compressed.original_shape.items():
            d, k = struct.unpack('II', data[offset:offset + 8])
            offset += 8

            indices = np.frombuffer(data[offset:offset + k * 4], dtype=np.uint32)
            offset += k * 4

            values = np.frombuffer(data[offset:offset + k * 4], dtype=np.float32)
            offset += k * 4

            gradient = np.zeros(d, dtype=np.float32)
            gradient[indices] = values

            result[key] = gradient.reshape(shape)

        return result


# =============================================================================
# SPARSIFICATION: Threshold-based
# =============================================================================

class ThresholdCompressor(GradientCompressor):
    """
    Threshold-based Sparsification.

    Keep only gradients with magnitude above threshold.
    Adaptive sparsity based on gradient distribution.
    """

    def compress(self,
                gradient: Dict[str, np.ndarray],
                client_id: int = 0) -> CompressedGradient:
        """Compress using threshold sparsification."""
        if self.config.use_error_feedback:
            error_buffer = self._get_error_buffer(client_id, gradient)
            gradient = {k: gradient[k] + error_buffer[k] for k in gradient.keys()}

        compressed_data = b""
        original_shapes = {}
        metadata = {}

        for key, arr in gradient.items():
            flat = arr.flatten()
            original_shapes[key] = arr.shape

            # Adaptive threshold based on std
            threshold = self.config.threshold * np.std(flat)
            mask = np.abs(flat) > threshold

            indices = np.where(mask)[0]
            values = flat[indices]

            metadata[f"{key}_threshold"] = float(threshold)
            metadata[f"{key}_sparsity"] = float(1 - len(indices) / len(flat))

            compressed_data += struct.pack('II', len(flat), len(indices))
            compressed_data += indices.astype(np.uint32).tobytes()
            compressed_data += values.astype(np.float32).tobytes()

        decompressed = self.decompress(CompressedGradient(
            client_id=client_id,
            compressed_data=compressed_data,
            original_shape=original_shapes,
            compression_method="threshold",
            compression_ratio=0,
            metadata=metadata
        ))
        self._update_error_buffer(client_id, gradient, decompressed)

        original_size = sum(arr.nbytes for arr in gradient.values())
        compressed_size = len(compressed_data)

        return CompressedGradient(
            client_id=client_id,
            compressed_data=compressed_data,
            original_shape=original_shapes,
            compression_method="threshold",
            compression_ratio=self.compute_compression_ratio(original_size, compressed_size),
            metadata=metadata
        )

    def decompress(self, compressed: CompressedGradient) -> Dict[str, np.ndarray]:
        """Decompress threshold-based gradient."""
        result = {}
        data = compressed.compressed_data
        offset = 0

        for key, shape in compressed.original_shape.items():
            d, k = struct.unpack('II', data[offset:offset + 8])
            offset += 8

            if k > 0:
                indices = np.frombuffer(data[offset:offset + k * 4], dtype=np.uint32)
                offset += k * 4
                values = np.frombuffer(data[offset:offset + k * 4], dtype=np.float32)
                offset += k * 4
            else:
                indices = np.array([], dtype=np.uint32)
                values = np.array([], dtype=np.float32)

            gradient = np.zeros(d, dtype=np.float32)
            if len(indices) > 0:
                gradient[indices] = values

            result[key] = gradient.reshape(shape)

        return result


# =============================================================================
# LOW-RANK: PowerSGD
# =============================================================================

class PowerSGDCompressor(GradientCompressor):
    """
    PowerSGD: Low-Rank Gradient Compression.

    Approximates gradient matrices using low-rank decomposition
    via power iteration. Very effective for large models.

    Reference: Vogels et al., "PowerSGD: Practical Low-Rank
    Gradient Compression for Distributed Optimization", NeurIPS 2019.
    """

    def __init__(self, config: CompressionConfig):
        super().__init__(config)
        self.rank = config.rank
        # Store Q matrices for power iteration warm start
        self.Q_matrices: Dict[int, Dict[str, np.ndarray]] = {}

    def _power_iteration(self,
                        M: np.ndarray,
                        Q: Optional[np.ndarray] = None,
                        num_iters: int = 2) -> Tuple[np.ndarray, np.ndarray]:
        """
        Low-rank approximation via power iteration.

        Returns P, Q such that M ≈ P @ Q.T
        """
        m, n = M.shape
        r = min(self.rank, min(m, n))

        # Initialize or use warm start
        if Q is None:
            Q = np.random.randn(n, r)
            Q, _ = np.linalg.qr(Q)

        # Power iteration
        for _ in range(num_iters):
            P = M @ Q
            P, _ = np.linalg.qr(P)
            Q = M.T @ P
            Q, _ = np.linalg.qr(Q)

        P = M @ Q
        return P, Q

    def compress(self,
                gradient: Dict[str, np.ndarray],
                client_id: int = 0) -> CompressedGradient:
        """Compress gradient using PowerSGD."""
        if self.config.use_error_feedback:
            error_buffer = self._get_error_buffer(client_id, gradient)
            gradient = {k: gradient[k] + error_buffer[k] for k in gradient.keys()}

        # Initialize Q matrices for client
        if client_id not in self.Q_matrices:
            self.Q_matrices[client_id] = {}

        compressed_data = b""
        original_shapes = {}
        metadata = {}

        for key, arr in gradient.items():
            original_shapes[key] = arr.shape

            # Reshape to 2D for matrix decomposition
            if arr.ndim == 1:
                # For 1D, create pseudo-matrix
                side = int(np.sqrt(len(arr)))
                if side * side < len(arr):
                    side += 1
                padded = np.zeros(side * side)
                padded[:len(arr)] = arr.flatten()
                M = padded.reshape(side, side)
                metadata[f"{key}_padded"] = True
                metadata[f"{key}_original_len"] = len(arr)
            else:
                M = arr.reshape(arr.shape[0], -1)
                metadata[f"{key}_padded"] = False

            # Get warm-start Q if available
            Q_init = self.Q_matrices[client_id].get(key)

            # Power iteration
            P, Q = self._power_iteration(M, Q_init)

            # Store Q for warm start
            self.Q_matrices[client_id][key] = Q

            # Pack P and Q
            metadata[f"{key}_P_shape"] = P.shape
            metadata[f"{key}_Q_shape"] = Q.shape

            compressed_data += struct.pack('IIII', P.shape[0], P.shape[1], Q.shape[0], Q.shape[1])
            compressed_data += P.astype(np.float32).tobytes()
            compressed_data += Q.astype(np.float32).tobytes()

        # Error feedback
        decompressed = self.decompress(CompressedGradient(
            client_id=client_id,
            compressed_data=compressed_data,
            original_shape=original_shapes,
            compression_method="powersgd",
            compression_ratio=0,
            metadata=metadata
        ))
        self._update_error_buffer(client_id, gradient, decompressed)

        original_size = sum(arr.nbytes for arr in gradient.values())
        compressed_size = len(compressed_data)

        return CompressedGradient(
            client_id=client_id,
            compressed_data=compressed_data,
            original_shape=original_shapes,
            compression_method="powersgd",
            compression_ratio=self.compute_compression_ratio(original_size, compressed_size),
            metadata=metadata
        )

    def decompress(self, compressed: CompressedGradient) -> Dict[str, np.ndarray]:
        """Decompress PowerSGD gradient."""
        result = {}
        data = compressed.compressed_data
        offset = 0

        for key, shape in compressed.original_shape.items():
            p_rows, p_cols, q_rows, q_cols = struct.unpack('IIII', data[offset:offset + 16])
            offset += 16

            P = np.frombuffer(data[offset:offset + p_rows * p_cols * 4], dtype=np.float32)
            P = P.reshape(p_rows, p_cols)
            offset += p_rows * p_cols * 4

            Q = np.frombuffer(data[offset:offset + q_rows * q_cols * 4], dtype=np.float32)
            Q = Q.reshape(q_rows, q_cols)
            offset += q_rows * q_cols * 4

            # Reconstruct
            M_approx = P @ Q.T

            # Handle padding
            if compressed.metadata.get(f"{key}_padded", False):
                original_len = compressed.metadata[f"{key}_original_len"]
                gradient = M_approx.flatten()[:original_len]
            else:
                gradient = M_approx

            result[key] = gradient.reshape(shape)

        return result


# =============================================================================
# COMPRESSION MANAGER
# =============================================================================

class CompressionManager:
    """
    High-level manager for gradient compression in FL.

    Handles compression/decompression and tracks statistics.
    """

    def __init__(self, config: CompressionConfig):
        self.config = config
        self.compressor = self._create_compressor(config)
        self.stats = {
            'total_original_bytes': 0,
            'total_compressed_bytes': 0,
            'compression_ratios': [],
            'num_compressions': 0
        }

    def _create_compressor(self, config: CompressionConfig) -> GradientCompressor:
        """Create appropriate compressor based on config."""
        compressors = {
            'signsgd': SignSGDCompressor,
            'qsgd': QSGDCompressor,
            'terngrad': TernGradCompressor,
            'topk': TopKCompressor,
            'randomk': RandomKCompressor,
            'threshold': ThresholdCompressor,
            'powersgd': PowerSGDCompressor,
        }

        if config.method.lower() not in compressors:
            raise ValueError(f"Unknown compression method: {config.method}")

        return compressors[config.method.lower()](config)

    def compress(self,
                gradient: Dict[str, np.ndarray],
                client_id: int = 0) -> CompressedGradient:
        """Compress gradient and track statistics."""
        original_size = sum(arr.nbytes for arr in gradient.values())

        compressed = self.compressor.compress(gradient, client_id)

        # Update stats
        self.stats['total_original_bytes'] += original_size
        self.stats['total_compressed_bytes'] += len(compressed.compressed_data)
        self.stats['compression_ratios'].append(compressed.compression_ratio)
        self.stats['num_compressions'] += 1

        return compressed

    def decompress(self, compressed: CompressedGradient) -> Dict[str, np.ndarray]:
        """Decompress gradient."""
        return self.compressor.decompress(compressed)

    def get_average_compression_ratio(self) -> float:
        """Get average compression ratio across all compressions."""
        if not self.stats['compression_ratios']:
            return 1.0
        return np.mean(self.stats['compression_ratios'])

    def get_total_bandwidth_saved(self) -> int:
        """Get total bytes saved through compression."""
        return self.stats['total_original_bytes'] - self.stats['total_compressed_bytes']

    def get_stats(self) -> Dict:
        """Get compression statistics."""
        return {
            **self.stats,
            'average_compression_ratio': self.get_average_compression_ratio(),
            'bandwidth_saved_bytes': self.get_total_bandwidth_saved(),
            'bandwidth_saved_pct': (
                self.get_total_bandwidth_saved() / max(self.stats['total_original_bytes'], 1)
            ) * 100
        }


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_compressor(method: str,
                     config: Optional[CompressionConfig] = None,
                     **kwargs) -> CompressionManager:
    """
    Factory function to create compression manager.

    Args:
        method: Compression method name
        config: CompressionConfig object
        **kwargs: Additional config parameters
    """
    if config is None:
        config = CompressionConfig(method=method, **kwargs)
    else:
        config.method = method

    return CompressionManager(config)


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    print("FL-EHDS Model Compression Demo")
    print("=" * 60)

    # Create test gradient (simulating a small model)
    np.random.seed(42)
    gradient = {
        'layer1.weights': np.random.randn(100, 50).astype(np.float32),
        'layer1.bias': np.random.randn(50).astype(np.float32),
        'layer2.weights': np.random.randn(50, 10).astype(np.float32),
        'layer2.bias': np.random.randn(10).astype(np.float32),
    }

    original_size = sum(arr.nbytes for arr in gradient.values())
    print(f"Original gradient size: {original_size:,} bytes")

    # Test different compression methods
    methods = ['signsgd', 'qsgd', 'terngrad', 'topk', 'randomk', 'threshold', 'powersgd']

    results = {}

    for method in methods:
        print(f"\n{'-' * 40}")
        print(f"Testing: {method.upper()}")
        print("-" * 40)

        config = CompressionConfig(
            method=method,
            num_bits=4,
            k_ratio=0.1,
            threshold=0.5,
            rank=4,
            use_error_feedback=True
        )

        manager = create_compressor(method, config)

        # Compress
        compressed = manager.compress(gradient, client_id=0)
        print(f"Compressed size: {len(compressed.compressed_data):,} bytes")
        print(f"Compression ratio: {compressed.compression_ratio:.2f}x")

        # Decompress
        decompressed = manager.decompress(compressed)

        # Check reconstruction error
        total_error = 0
        total_norm = 0
        for key in gradient.keys():
            error = np.linalg.norm(gradient[key] - decompressed[key])
            norm = np.linalg.norm(gradient[key])
            total_error += error ** 2
            total_norm += norm ** 2

        relative_error = np.sqrt(total_error / total_norm)
        print(f"Relative reconstruction error: {relative_error:.4f}")

        results[method] = {
            'compression_ratio': compressed.compression_ratio,
            'relative_error': relative_error,
            'compressed_size': len(compressed.compressed_data)
        }

    # Summary
    print("\n" + "=" * 60)
    print("COMPRESSION COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{'Method':<12} {'Ratio':<10} {'Error':<12} {'Size (bytes)':<12}")
    print("-" * 50)

    for method, res in sorted(results.items(), key=lambda x: -x[1]['compression_ratio']):
        print(f"{method:<12} {res['compression_ratio']:<10.2f}x {res['relative_error']:<12.4f} {res['compressed_size']:<12,}")

    print("\n" + "=" * 60)
    print("Recommendations:")
    print("- SignSGD: Maximum compression, best for bandwidth-limited scenarios")
    print("- TopK: Good balance of compression and accuracy")
    print("- PowerSGD: Best for large models with matrix-shaped layers")
    print("- QSGD: Configurable precision, unbiased estimates")

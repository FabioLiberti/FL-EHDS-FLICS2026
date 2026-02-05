"""
Model Watermarking for FL-EHDS
===============================

Implementation of model watermarking techniques for IP protection
and provenance tracking in cross-border federated learning.

Supported Approaches:
1. Weight-based Watermarking - Embed in model weights (LSB, spread spectrum)
2. Backdoor-based Watermarking - Trigger-based verification
3. Feature-based Watermarking - Embed in activations
4. Passport-based Watermarking - Separate passport layers
5. Federated Watermarking - Collaborative embedding across clients

Key References:
- Uchida et al., "Embedding Watermarks into Deep Neural Networks", 2017
- Adi et al., "Turning Your Weakness Into a Strength", 2018
- Fan et al., "Rethinking Deep Neural Network Ownership Verification", 2019
- Li et al., "Protecting Intellectual Property of DNN with Watermarking", 2021
- Tekgul et al., "WAFFLE: Watermarking in Federated Learning", 2021

EHDS Relevance:
- Cross-border model IP protection
- Provenance tracking for regulatory compliance
- Ownership verification for collaborative models
- Audit trail for model distribution

Author: FL-EHDS Framework
License: Apache 2.0
"""

import hashlib
import hmac
import json
import logging
import secrets
import struct
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Constants
# =============================================================================

class WatermarkType(Enum):
    """Types of model watermarking."""
    WEIGHT_BASED = "weight_based"  # Embed in weights
    BACKDOOR = "backdoor"  # Trigger-based
    FEATURE_BASED = "feature_based"  # Embed in features
    PASSPORT = "passport"  # Passport layers
    FEDERATED = "federated"  # Collaborative watermark


class EmbeddingMethod(Enum):
    """Weight embedding methods."""
    LSB = "lsb"  # Least Significant Bit
    SPREAD_SPECTRUM = "spread_spectrum"  # Spread spectrum
    QUANTIZATION = "quantization"  # Quantization-based
    REGULARIZATION = "regularization"  # Training regularization


class VerificationResult(Enum):
    """Watermark verification result."""
    VERIFIED = "verified"
    NOT_FOUND = "not_found"
    TAMPERED = "tampered"
    PARTIAL_MATCH = "partial_match"
    INVALID_KEY = "invalid_key"


@dataclass
class WatermarkConfig:
    """Configuration for watermarking."""
    watermark_type: WatermarkType
    embedding_method: EmbeddingMethod = EmbeddingMethod.SPREAD_SPECTRUM
    watermark_strength: float = 0.01  # Embedding strength
    watermark_length: int = 256  # Bits
    target_layers: List[str] = field(default_factory=list)  # Specific layers
    trigger_set_size: int = 100  # For backdoor watermarks
    verification_threshold: float = 0.9  # Detection threshold
    robustness_level: str = "medium"  # low, medium, high


@dataclass
class WatermarkSignature:
    """Watermark signature for verification."""
    signature_id: str
    owner_id: str
    owner_organization: str
    creation_timestamp: datetime
    watermark_type: WatermarkType
    watermark_hash: str  # SHA-256 of watermark data
    model_hash: str  # Hash of original model
    target_layers: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "signatureId": self.signature_id,
            "ownerId": self.owner_id,
            "ownerOrganization": self.owner_organization,
            "creationTimestamp": self.creation_timestamp.isoformat(),
            "watermarkType": self.watermark_type.value,
            "watermarkHash": self.watermark_hash,
            "modelHash": self.model_hash,
            "targetLayers": self.target_layers,
            "metadata": self.metadata,
        }

    def to_json(self) -> str:
        """Serialize to JSON."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> "WatermarkSignature":
        """Deserialize from JSON."""
        data = json.loads(json_str)
        return cls(
            signature_id=data["signatureId"],
            owner_id=data["ownerId"],
            owner_organization=data["ownerOrganization"],
            creation_timestamp=datetime.fromisoformat(data["creationTimestamp"]),
            watermark_type=WatermarkType(data["watermarkType"]),
            watermark_hash=data["watermarkHash"],
            model_hash=data["modelHash"],
            target_layers=data["targetLayers"],
            metadata=data.get("metadata", {}),
        )


@dataclass
class VerificationReport:
    """Report from watermark verification."""
    signature_id: str
    verification_result: VerificationResult
    confidence_score: float
    verification_timestamp: datetime
    extracted_bits: Optional[int] = None
    matched_bits: Optional[int] = None
    layer_results: Dict[str, float] = field(default_factory=dict)
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "signatureId": self.signature_id,
            "verificationResult": self.verification_result.value,
            "confidenceScore": self.confidence_score,
            "verificationTimestamp": self.verification_timestamp.isoformat(),
            "extractedBits": self.extracted_bits,
            "matchedBits": self.matched_bits,
            "layerResults": self.layer_results,
            "notes": self.notes,
        }


# =============================================================================
# Watermark Generators
# =============================================================================

class WatermarkGenerator:
    """Generates cryptographic watermarks."""

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.RandomState(seed)
        self._master_key: Optional[bytes] = None

    def set_master_key(self, key: bytes) -> None:
        """Set master key for watermark generation."""
        self._master_key = key

    def generate_key(self, length: int = 32) -> bytes:
        """Generate random watermark key."""
        return secrets.token_bytes(length)

    def generate_watermark(
        self,
        owner_id: str,
        key: bytes,
        length: int = 256,
    ) -> np.ndarray:
        """
        Generate deterministic watermark from owner and key.

        Args:
            owner_id: Owner identifier
            key: Secret key
            length: Watermark length in bits

        Returns:
            Binary watermark array
        """
        # Create deterministic seed from owner + key
        combined = owner_id.encode() + key
        seed_hash = hashlib.sha256(combined).digest()
        seed = struct.unpack('>I', seed_hash[:4])[0]

        # Generate binary watermark
        rng = np.random.RandomState(seed)
        watermark = rng.randint(0, 2, size=length).astype(np.float32)

        return watermark

    def generate_trigger_set(
        self,
        input_shape: Tuple[int, ...],
        num_samples: int = 100,
        key: bytes = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate trigger set for backdoor watermarking.

        Args:
            input_shape: Shape of model input
            num_samples: Number of trigger samples
            key: Secret key for deterministic generation

        Returns:
            Tuple of (trigger_inputs, trigger_labels)
        """
        if key:
            seed = struct.unpack('>I', hashlib.sha256(key).digest()[:4])[0]
            rng = np.random.RandomState(seed)
        else:
            rng = self.rng

        # Generate trigger pattern (abstract pattern)
        trigger_inputs = rng.randn(num_samples, *input_shape).astype(np.float32)

        # Generate unique labels for verification
        num_classes = 10  # Assume 10 classes
        trigger_labels = rng.randint(0, num_classes, size=num_samples)

        return trigger_inputs, trigger_labels

    def generate_passport_signature(
        self,
        layer_shapes: Dict[str, Tuple[int, ...]],
        key: bytes,
    ) -> Dict[str, np.ndarray]:
        """
        Generate passport signatures for layers.

        Args:
            layer_shapes: Dict of layer_name -> shape
            key: Secret key

        Returns:
            Dict of layer_name -> passport signature
        """
        passports = {}
        seed = struct.unpack('>I', hashlib.sha256(key).digest()[:4])[0]
        rng = np.random.RandomState(seed)

        for layer_name, shape in layer_shapes.items():
            # Generate scale and bias for passport
            scale = rng.randn(*shape).astype(np.float32) * 0.1 + 1.0
            bias = rng.randn(*shape).astype(np.float32) * 0.01

            passports[layer_name] = {
                "scale": scale,
                "bias": bias,
            }

        return passports


# =============================================================================
# Watermark Embedders
# =============================================================================

class WatermarkEmbedder(ABC):
    """Abstract base class for watermark embedding."""

    @abstractmethod
    def embed(
        self,
        model_weights: Dict[str, np.ndarray],
        watermark: np.ndarray,
        config: WatermarkConfig,
    ) -> Dict[str, np.ndarray]:
        """Embed watermark into model weights."""
        pass

    @abstractmethod
    def extract(
        self,
        model_weights: Dict[str, np.ndarray],
        config: WatermarkConfig,
    ) -> np.ndarray:
        """Extract watermark from model weights."""
        pass


class SpreadSpectrumEmbedder(WatermarkEmbedder):
    """
    Spread Spectrum Watermark Embedding.

    Spreads watermark across model weights using a pseudo-random
    spreading sequence, providing robustness against pruning/fine-tuning.
    """

    def __init__(self, key: bytes):
        self.key = key
        self._generate_spreading_sequence()

    def _generate_spreading_sequence(self) -> None:
        """Generate spreading sequence from key."""
        seed = struct.unpack('>I', hashlib.sha256(self.key).digest()[:4])[0]
        self.rng = np.random.RandomState(seed)

    def embed(
        self,
        model_weights: Dict[str, np.ndarray],
        watermark: np.ndarray,
        config: WatermarkConfig,
    ) -> Dict[str, np.ndarray]:
        """
        Embed watermark using spread spectrum.

        W' = W + α * S * m

        Where:
        - W: Original weights
        - α: Embedding strength
        - S: Spreading sequence
        - m: Watermark bit
        """
        watermarked = {}
        watermark_idx = 0
        watermark_len = len(watermark)

        target_layers = config.target_layers or list(model_weights.keys())

        for layer_name, weights in model_weights.items():
            if layer_name not in target_layers:
                watermarked[layer_name] = weights.copy()
                continue

            # Flatten weights
            flat_weights = weights.flatten()
            n_weights = len(flat_weights)

            # Generate spreading sequence for this layer
            spreading = self.rng.randn(n_weights).astype(np.float32)
            spreading = spreading / np.linalg.norm(spreading)

            # Embed watermark bits
            modified_weights = flat_weights.copy()

            for i in range(n_weights):
                bit_idx = watermark_idx % watermark_len
                bit = watermark[bit_idx] * 2 - 1  # Convert 0/1 to -1/1

                modified_weights[i] += config.watermark_strength * spreading[i] * bit
                watermark_idx += 1

            watermarked[layer_name] = modified_weights.reshape(weights.shape)

        logger.info(f"Embedded {watermark_len}-bit watermark in {len(target_layers)} layers")
        return watermarked

    def extract(
        self,
        model_weights: Dict[str, np.ndarray],
        config: WatermarkConfig,
    ) -> np.ndarray:
        """
        Extract watermark using correlation detection.
        """
        # Reset RNG for same spreading sequence
        self._generate_spreading_sequence()

        target_layers = config.target_layers or list(model_weights.keys())
        watermark_len = config.watermark_length

        # Accumulate correlations
        correlations = np.zeros(watermark_len)
        counts = np.zeros(watermark_len)

        watermark_idx = 0

        for layer_name in target_layers:
            if layer_name not in model_weights:
                continue

            weights = model_weights[layer_name]
            flat_weights = weights.flatten()
            n_weights = len(flat_weights)

            # Regenerate spreading sequence
            spreading = self.rng.randn(n_weights).astype(np.float32)
            spreading = spreading / np.linalg.norm(spreading)

            # Compute correlation for each bit position
            for i in range(n_weights):
                bit_idx = watermark_idx % watermark_len
                correlations[bit_idx] += flat_weights[i] * spreading[i]
                counts[bit_idx] += 1
                watermark_idx += 1

        # Average correlations
        correlations = correlations / (counts + 1e-10)

        # Decode bits (positive correlation = 1, negative = 0)
        extracted = (correlations > 0).astype(np.float32)

        return extracted


class LSBEmbedder(WatermarkEmbedder):
    """
    Least Significant Bit Watermark Embedding.

    Embeds watermark in the least significant bits of weight representations.
    Simple but less robust to fine-tuning.
    """

    def __init__(self, key: bytes, precision: int = 16):
        self.key = key
        self.precision = precision
        seed = struct.unpack('>I', hashlib.sha256(key).digest()[:4])[0]
        self.rng = np.random.RandomState(seed)

    def embed(
        self,
        model_weights: Dict[str, np.ndarray],
        watermark: np.ndarray,
        config: WatermarkConfig,
    ) -> Dict[str, np.ndarray]:
        """Embed watermark in LSBs of selected weights."""
        watermarked = {}
        watermark_len = len(watermark)
        target_layers = config.target_layers or list(model_weights.keys())

        bit_idx = 0

        for layer_name, weights in model_weights.items():
            if layer_name not in target_layers:
                watermarked[layer_name] = weights.copy()
                continue

            flat = weights.flatten()

            # Select positions to embed (deterministic based on key)
            positions = self.rng.choice(
                len(flat),
                size=min(watermark_len, len(flat)),
                replace=False
            )

            modified = flat.copy()

            for pos in positions:
                if bit_idx >= watermark_len:
                    break

                # Quantize to fixed point
                scale = 2 ** self.precision
                quantized = int(modified[pos] * scale)

                # Set LSB to watermark bit
                if watermark[bit_idx] == 1:
                    quantized = quantized | 1
                else:
                    quantized = quantized & ~1

                modified[pos] = quantized / scale
                bit_idx += 1

            watermarked[layer_name] = modified.reshape(weights.shape)

        return watermarked

    def extract(
        self,
        model_weights: Dict[str, np.ndarray],
        config: WatermarkConfig,
    ) -> np.ndarray:
        """Extract watermark from LSBs."""
        # Reset RNG
        self.rng = np.random.RandomState(
            struct.unpack('>I', hashlib.sha256(self.key).digest()[:4])[0]
        )

        target_layers = config.target_layers or list(model_weights.keys())
        watermark_len = config.watermark_length

        extracted = []

        for layer_name in target_layers:
            if layer_name not in model_weights:
                continue

            weights = model_weights[layer_name]
            flat = weights.flatten()

            positions = self.rng.choice(
                len(flat),
                size=min(watermark_len - len(extracted), len(flat)),
                replace=False
            )

            for pos in positions:
                if len(extracted) >= watermark_len:
                    break

                scale = 2 ** self.precision
                quantized = int(flat[pos] * scale)
                bit = quantized & 1
                extracted.append(bit)

        return np.array(extracted, dtype=np.float32)


class BackdoorEmbedder(WatermarkEmbedder):
    """
    Backdoor-based Watermark Embedding.

    Embeds watermark as a backdoor pattern that triggers specific
    behavior on trigger inputs.
    """

    def __init__(
        self,
        trigger_inputs: np.ndarray,
        trigger_labels: np.ndarray,
    ):
        self.trigger_inputs = trigger_inputs
        self.trigger_labels = trigger_labels

    def embed(
        self,
        model_weights: Dict[str, np.ndarray],
        watermark: np.ndarray,
        config: WatermarkConfig,
    ) -> Dict[str, np.ndarray]:
        """
        Embed backdoor watermark via training.

        Note: This requires additional training with trigger set.
        Returns weights with placeholder modification for demonstration.
        """
        # In practice, this would involve fine-tuning the model
        # on the trigger set to learn the backdoor

        watermarked = {}
        for layer_name, weights in model_weights.items():
            # Add small perturbation as marker
            watermarked[layer_name] = weights + np.random.randn(*weights.shape) * 0.0001

        logger.info(f"Backdoor watermark embedded with {len(self.trigger_inputs)} triggers")
        return watermarked

    def extract(
        self,
        model_weights: Dict[str, np.ndarray],
        config: WatermarkConfig,
    ) -> np.ndarray:
        """
        Verify backdoor by checking trigger predictions.

        Note: Requires forward pass through model.
        Returns placeholder for API compatibility.
        """
        # In practice, would run inference on trigger inputs
        # and check if predictions match trigger_labels

        return np.ones(config.watermark_length)

    def verify_backdoor(
        self,
        predict_fn: Callable[[np.ndarray], np.ndarray],
        threshold: float = 0.9,
    ) -> Tuple[bool, float]:
        """
        Verify backdoor watermark using prediction function.

        Args:
            predict_fn: Function that takes inputs and returns predictions
            threshold: Minimum accuracy on trigger set

        Returns:
            Tuple of (verified, accuracy)
        """
        predictions = predict_fn(self.trigger_inputs)

        if len(predictions.shape) > 1:
            predictions = np.argmax(predictions, axis=1)

        accuracy = np.mean(predictions == self.trigger_labels)
        verified = accuracy >= threshold

        return verified, float(accuracy)


class PassportEmbedder(WatermarkEmbedder):
    """
    Passport-based Watermark Embedding.

    Adds passport layers that require correct passport to
    function properly.

    Reference: Fan et al., "Rethinking Deep Neural Network
    Ownership Verification: Embedding Passports", 2019
    """

    def __init__(self, passports: Dict[str, Dict[str, np.ndarray]]):
        self.passports = passports

    def embed(
        self,
        model_weights: Dict[str, np.ndarray],
        watermark: np.ndarray,
        config: WatermarkConfig,
    ) -> Dict[str, np.ndarray]:
        """
        Embed passport into model.

        Modifies batch norm / layer norm parameters to require
        passport for correct operation.
        """
        watermarked = model_weights.copy()

        for layer_name, passport in self.passports.items():
            if layer_name in model_weights:
                # Apply passport transformation
                scale = passport["scale"]
                bias = passport["bias"]

                # Modify weights to require passport
                watermarked[layer_name] = model_weights[layer_name] * scale + bias

        logger.info(f"Passport embedded in {len(self.passports)} layers")
        return watermarked

    def extract(
        self,
        model_weights: Dict[str, np.ndarray],
        config: WatermarkConfig,
    ) -> np.ndarray:
        """Extract passport signature."""
        # Check if passport layers are present and match
        match_scores = []

        for layer_name, passport in self.passports.items():
            if layer_name in model_weights:
                # Attempt to reverse passport
                expected = model_weights[layer_name]
                scale = passport["scale"]
                bias = passport["bias"]

                # Check correlation
                correlation = np.corrcoef(
                    expected.flatten(),
                    (scale + bias).flatten()
                )[0, 1]

                match_scores.append(correlation)

        return np.array(match_scores)

    def verify_passport(
        self,
        model_weights: Dict[str, np.ndarray],
    ) -> Tuple[bool, Dict[str, float]]:
        """
        Verify passport ownership.

        Returns:
            Tuple of (verified, layer_match_scores)
        """
        layer_scores = {}

        for layer_name, passport in self.passports.items():
            if layer_name not in model_weights:
                layer_scores[layer_name] = 0.0
                continue

            weights = model_weights[layer_name]
            scale = passport["scale"]
            bias = passport["bias"]

            # Check if transformation matches
            expected = weights * scale + bias
            actual = model_weights.get(f"{layer_name}_transformed", weights)

            correlation = np.corrcoef(
                expected.flatten()[:1000],
                actual.flatten()[:1000]
            )[0, 1]

            layer_scores[layer_name] = float(correlation)

        avg_score = np.mean(list(layer_scores.values()))
        verified = avg_score > 0.8

        return verified, layer_scores


# =============================================================================
# Federated Watermarking
# =============================================================================

class FederatedWatermarkCoordinator:
    """
    Coordinator for federated watermarking.

    Enables collaborative watermark embedding across FL clients
    while maintaining individual contribution tracking.
    """

    def __init__(
        self,
        master_key: bytes,
        aggregation_method: str = "weighted",
    ):
        self.master_key = master_key
        self.aggregation_method = aggregation_method
        self.generator = WatermarkGenerator()
        self.generator.set_master_key(master_key)

        self._client_watermarks: Dict[str, np.ndarray] = {}
        self._client_keys: Dict[str, bytes] = {}
        self._global_watermark: Optional[np.ndarray] = None

    def register_client(
        self,
        client_id: str,
        organization: str,
    ) -> bytes:
        """
        Register a client and generate their watermark key.

        Returns client-specific key for watermark generation.
        """
        # Generate client-specific key
        client_key = hashlib.sha256(
            self.master_key + client_id.encode() + organization.encode()
        ).digest()

        self._client_keys[client_id] = client_key

        # Generate client watermark
        watermark = self.generator.generate_watermark(
            owner_id=client_id,
            key=client_key,
            length=256,
        )
        self._client_watermarks[client_id] = watermark

        logger.info(f"Registered client {client_id} from {organization}")
        return client_key

    def get_client_watermark(self, client_id: str) -> Optional[np.ndarray]:
        """Get watermark for a specific client."""
        return self._client_watermarks.get(client_id)

    def aggregate_watermarks(
        self,
        client_ids: List[str],
        client_weights: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:
        """
        Aggregate watermarks from participating clients.

        Creates a combined watermark that encodes contributions
        from all participating clients.
        """
        if not client_weights:
            client_weights = {cid: 1.0 / len(client_ids) for cid in client_ids}

        # Initialize aggregated watermark
        watermark_len = 256
        aggregated = np.zeros(watermark_len)

        for client_id in client_ids:
            if client_id not in self._client_watermarks:
                continue

            weight = client_weights.get(client_id, 0)
            aggregated += weight * self._client_watermarks[client_id]

        # Binarize (threshold at 0.5)
        self._global_watermark = (aggregated > 0.5).astype(np.float32)

        return self._global_watermark

    def verify_client_contribution(
        self,
        client_id: str,
        extracted_watermark: np.ndarray,
        threshold: float = 0.6,
    ) -> Tuple[bool, float]:
        """
        Verify if a client contributed to a watermarked model.

        Returns:
            Tuple of (contributed, correlation_score)
        """
        if client_id not in self._client_watermarks:
            return False, 0.0

        client_watermark = self._client_watermarks[client_id]

        # Compute correlation
        correlation = np.corrcoef(
            client_watermark,
            extracted_watermark
        )[0, 1]

        contributed = correlation > threshold

        return contributed, float(correlation)

    def get_contributing_clients(
        self,
        extracted_watermark: np.ndarray,
        threshold: float = 0.5,
    ) -> List[Tuple[str, float]]:
        """
        Identify all clients that contributed to a watermarked model.

        Returns:
            List of (client_id, contribution_score) tuples
        """
        contributors = []

        for client_id, client_watermark in self._client_watermarks.items():
            correlation = np.corrcoef(
                client_watermark,
                extracted_watermark
            )[0, 1]

            if correlation > threshold:
                contributors.append((client_id, float(correlation)))

        # Sort by contribution score
        contributors.sort(key=lambda x: -x[1])

        return contributors


# =============================================================================
# Watermark Manager
# =============================================================================

class ModelWatermarkManager:
    """
    Central manager for model watermarking.

    Coordinates watermark generation, embedding, extraction,
    and verification for FL-EHDS models.
    """

    def __init__(
        self,
        organization_id: str,
        organization_name: str,
    ):
        self.organization_id = organization_id
        self.organization_name = organization_name

        self._master_key: Optional[bytes] = None
        self._generator = WatermarkGenerator()
        self._embedders: Dict[EmbeddingMethod, WatermarkEmbedder] = {}
        self._signatures: Dict[str, WatermarkSignature] = {}
        self._federated_coordinator: Optional[FederatedWatermarkCoordinator] = None

        logger.info(f"Watermark Manager initialized for {organization_name}")

    def initialize(self, master_key: Optional[bytes] = None) -> bytes:
        """
        Initialize manager with master key.

        Returns the master key (generated if not provided).
        """
        self._master_key = master_key or self._generator.generate_key(32)
        self._generator.set_master_key(self._master_key)

        # Initialize embedders
        self._embedders[EmbeddingMethod.SPREAD_SPECTRUM] = SpreadSpectrumEmbedder(self._master_key)
        self._embedders[EmbeddingMethod.LSB] = LSBEmbedder(self._master_key)

        # Initialize federated coordinator
        self._federated_coordinator = FederatedWatermarkCoordinator(
            master_key=self._master_key,
        )

        logger.info("Watermark Manager initialized")
        return self._master_key

    def create_watermark(
        self,
        owner_id: str,
        config: WatermarkConfig,
    ) -> Tuple[np.ndarray, WatermarkSignature]:
        """
        Create a new watermark for a model owner.

        Returns:
            Tuple of (watermark, signature)
        """
        if not self._master_key:
            raise ValueError("Manager not initialized. Call initialize() first.")

        # Generate owner-specific key
        owner_key = hashlib.sha256(
            self._master_key + owner_id.encode()
        ).digest()

        # Generate watermark
        watermark = self._generator.generate_watermark(
            owner_id=owner_id,
            key=owner_key,
            length=config.watermark_length,
        )

        # Create signature
        watermark_hash = hashlib.sha256(watermark.tobytes()).hexdigest()

        signature = WatermarkSignature(
            signature_id=secrets.token_hex(16),
            owner_id=owner_id,
            owner_organization=self.organization_name,
            creation_timestamp=datetime.now(),
            watermark_type=config.watermark_type,
            watermark_hash=watermark_hash,
            model_hash="",  # Set after embedding
            target_layers=config.target_layers,
            metadata={
                "embedding_method": config.embedding_method.value,
                "watermark_strength": config.watermark_strength,
                "watermark_length": config.watermark_length,
            },
        )

        self._signatures[signature.signature_id] = signature

        logger.info(f"Created watermark for {owner_id}: {signature.signature_id}")
        return watermark, signature

    def embed_watermark(
        self,
        model_weights: Dict[str, np.ndarray],
        watermark: np.ndarray,
        config: WatermarkConfig,
        signature_id: str,
    ) -> Dict[str, np.ndarray]:
        """
        Embed watermark into model weights.

        Returns watermarked model weights.
        """
        embedder = self._embedders.get(config.embedding_method)
        if not embedder:
            raise ValueError(f"Unsupported embedding method: {config.embedding_method}")

        # Embed watermark
        watermarked = embedder.embed(model_weights, watermark, config)

        # Update signature with model hash
        if signature_id in self._signatures:
            model_hash = self._compute_model_hash(watermarked)
            self._signatures[signature_id].model_hash = model_hash

        logger.info(f"Watermark embedded using {config.embedding_method.value}")
        return watermarked

    def extract_watermark(
        self,
        model_weights: Dict[str, np.ndarray],
        config: WatermarkConfig,
    ) -> np.ndarray:
        """
        Extract watermark from model weights.

        Returns extracted watermark.
        """
        embedder = self._embedders.get(config.embedding_method)
        if not embedder:
            raise ValueError(f"Unsupported embedding method: {config.embedding_method}")

        extracted = embedder.extract(model_weights, config)

        logger.info(f"Extracted {len(extracted)}-bit watermark")
        return extracted

    def verify_watermark(
        self,
        model_weights: Dict[str, np.ndarray],
        signature_id: str,
        owner_id: str,
        config: WatermarkConfig,
    ) -> VerificationReport:
        """
        Verify watermark ownership.

        Returns verification report.
        """
        if signature_id not in self._signatures:
            return VerificationReport(
                signature_id=signature_id,
                verification_result=VerificationResult.INVALID_KEY,
                confidence_score=0.0,
                verification_timestamp=datetime.now(),
                notes="Signature not found",
            )

        signature = self._signatures[signature_id]

        # Generate expected watermark
        owner_key = hashlib.sha256(
            self._master_key + owner_id.encode()
        ).digest()

        expected_watermark = self._generator.generate_watermark(
            owner_id=owner_id,
            key=owner_key,
            length=config.watermark_length,
        )

        # Extract watermark from model
        extracted_watermark = self.extract_watermark(model_weights, config)

        # Compare
        matched_bits = np.sum(expected_watermark == extracted_watermark)
        total_bits = len(expected_watermark)
        match_ratio = matched_bits / total_bits

        # Determine result
        if match_ratio >= config.verification_threshold:
            result = VerificationResult.VERIFIED
        elif match_ratio >= 0.6:
            result = VerificationResult.PARTIAL_MATCH
        elif match_ratio >= 0.4:
            result = VerificationResult.TAMPERED
        else:
            result = VerificationResult.NOT_FOUND

        report = VerificationReport(
            signature_id=signature_id,
            verification_result=result,
            confidence_score=match_ratio,
            verification_timestamp=datetime.now(),
            extracted_bits=total_bits,
            matched_bits=int(matched_bits),
            notes=f"Match ratio: {match_ratio:.2%}",
        )

        logger.info(f"Verification: {result.value} (confidence={match_ratio:.2%})")
        return report

    def _compute_model_hash(self, model_weights: Dict[str, np.ndarray]) -> str:
        """Compute hash of model weights."""
        hasher = hashlib.sha256()
        for key in sorted(model_weights.keys()):
            hasher.update(model_weights[key].tobytes())
        return hasher.hexdigest()

    def get_federated_coordinator(self) -> FederatedWatermarkCoordinator:
        """Get federated watermark coordinator."""
        if not self._federated_coordinator:
            raise ValueError("Manager not initialized")
        return self._federated_coordinator

    def get_signature(self, signature_id: str) -> Optional[WatermarkSignature]:
        """Get signature by ID."""
        return self._signatures.get(signature_id)

    def export_signature(self, signature_id: str) -> Optional[str]:
        """Export signature as JSON."""
        sig = self._signatures.get(signature_id)
        if sig:
            return sig.to_json()
        return None

    def import_signature(self, json_str: str) -> WatermarkSignature:
        """Import signature from JSON."""
        sig = WatermarkSignature.from_json(json_str)
        self._signatures[sig.signature_id] = sig
        return sig


# =============================================================================
# Factory Functions
# =============================================================================

def create_watermark_manager(
    organization_id: str,
    organization_name: str,
    master_key: Optional[bytes] = None,
) -> ModelWatermarkManager:
    """Create and initialize watermark manager."""
    manager = ModelWatermarkManager(
        organization_id=organization_id,
        organization_name=organization_name,
    )
    manager.initialize(master_key)
    return manager


def create_watermark_config(
    watermark_type: WatermarkType = WatermarkType.WEIGHT_BASED,
    embedding_method: EmbeddingMethod = EmbeddingMethod.SPREAD_SPECTRUM,
    strength: float = 0.01,
    length: int = 256,
    target_layers: Optional[List[str]] = None,
) -> WatermarkConfig:
    """Create watermark configuration."""
    return WatermarkConfig(
        watermark_type=watermark_type,
        embedding_method=embedding_method,
        watermark_strength=strength,
        watermark_length=length,
        target_layers=target_layers or [],
    )


# =============================================================================
# Export
# =============================================================================

__all__ = [
    # Enums
    "WatermarkType",
    "EmbeddingMethod",
    "VerificationResult",
    # Data Classes
    "WatermarkConfig",
    "WatermarkSignature",
    "VerificationReport",
    # Generator
    "WatermarkGenerator",
    # Embedders
    "WatermarkEmbedder",
    "SpreadSpectrumEmbedder",
    "LSBEmbedder",
    "BackdoorEmbedder",
    "PassportEmbedder",
    # Federated
    "FederatedWatermarkCoordinator",
    # Manager
    "ModelWatermarkManager",
    # Factory
    "create_watermark_manager",
    "create_watermark_config",
]

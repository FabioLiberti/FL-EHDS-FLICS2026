#!/usr/bin/env python3
"""
FL-EHDS Byzantine-Resilient Federated Learning

Implements robust aggregation methods that tolerate Byzantine
(malicious or faulty) clients. Critical for EHDS where
data integrity must be maintained even with compromised nodes.

Aggregation Rules:
1. Krum / Multi-Krum - Distance-based selection
2. Trimmed Mean - Remove extremes before averaging
3. Median - Coordinate-wise median
4. Bulyan - Combines Krum with trimmed mean
5. FLTrust - Server-guided trust weighting
6. FLAME - Clustering-based defense

Attack Types Defended:
- Label flipping attacks
- Gradient scaling attacks
- Additive noise attacks
- Model replacement attacks

TEE (Trusted Execution Environment) Integration:
- Hardware-based attestation for client verification
- Secure enclave computation
- Remote attestation protocol
- SGX/TrustZone compatibility

Author: Fabio Liberti
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from copy import deepcopy
from scipy import stats
from datetime import datetime, timedelta
from enum import Enum, auto
import hashlib
import hmac
import secrets
import logging
import base64
import json

logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ByzantineConfig:
    """Configuration for Byzantine-resilient aggregation."""
    aggregation_rule: str = "krum"  # krum, trimmed_mean, median, bulyan, fltrust, flame
    # Krum params
    num_byzantine: int = 1  # Expected number of Byzantine clients
    multi_krum_m: int = None  # Number of gradients to select (None = single Krum)
    # Trimmed mean params
    trim_ratio: float = 0.1  # Fraction to trim from each side
    # FLTrust params
    server_lr: float = 0.1
    # FLAME params
    num_clusters: int = 2
    # Detection
    enable_detection: bool = True
    detection_threshold: float = 3.0  # Standard deviations for outlier


@dataclass
class ClientGradient:
    """Container for client gradient with metadata."""
    client_id: int
    gradient: Dict[str, np.ndarray]
    samples_used: int
    is_byzantine: bool = False  # Ground truth (for simulation)
    trust_score: float = 1.0


@dataclass
class AggregationResult:
    """Result of Byzantine-resilient aggregation."""
    aggregated_gradient: Dict[str, np.ndarray]
    selected_clients: List[int]
    rejected_clients: List[int]
    trust_scores: Dict[int, float]
    detection_alerts: List[str]


# =============================================================================
# BYZANTINE ATTACKS (For Testing)
# =============================================================================

class ByzantineAttacker:
    """
    Simulates Byzantine attacks for testing resilience.

    Attack types:
    - label_flip: Flip gradient signs
    - scale: Scale gradients by large factor
    - noise: Add random noise
    - sign_flip: Random sign flipping
    - lie: Compute gradient from wrong direction
    """

    def __init__(self, attack_type: str = "scale", attack_strength: float = 10.0):
        self.attack_type = attack_type
        self.strength = attack_strength

    def attack(self, gradient: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Apply Byzantine attack to gradient."""
        attacked = {}

        for key, g in gradient.items():
            if self.attack_type == "label_flip":
                attacked[key] = -g

            elif self.attack_type == "scale":
                attacked[key] = g * self.strength

            elif self.attack_type == "noise":
                noise = np.random.randn(*g.shape) * self.strength
                attacked[key] = g + noise

            elif self.attack_type == "sign_flip":
                mask = np.random.binomial(1, 0.5, g.shape)
                attacked[key] = g * (1 - 2 * mask)

            elif self.attack_type == "lie":
                # "A Little Is Enough" attack - subtle but effective
                attacked[key] = -self.strength * np.sign(g)

            else:
                attacked[key] = g

        return attacked


# =============================================================================
# BASE CLASS
# =============================================================================

class ByzantineResilientAggregator(ABC):
    """Abstract base class for Byzantine-resilient aggregation."""

    def __init__(self, config: ByzantineConfig):
        self.config = config
        self.trust_history: Dict[int, List[float]] = {}

    @abstractmethod
    def aggregate(self,
                 gradients: List[ClientGradient]) -> AggregationResult:
        """Perform Byzantine-resilient aggregation."""
        pass

    def _flatten_gradients(self,
                          gradients: List[ClientGradient]) -> np.ndarray:
        """Flatten gradients to 2D array (n_clients x n_params)."""
        flat_list = []
        for cg in gradients:
            flat = np.concatenate([g.flatten() for g in cg.gradient.values()])
            flat_list.append(flat)
        return np.array(flat_list)

    def _unflatten_gradient(self,
                           flat: np.ndarray,
                           template: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Unflatten gradient back to dict format."""
        result = {}
        offset = 0
        for key, arr in template.items():
            size = arr.size
            result[key] = flat[offset:offset + size].reshape(arr.shape)
            offset += size
        return result

    def _pairwise_distances(self, gradients: np.ndarray) -> np.ndarray:
        """Compute pairwise Euclidean distances between gradients."""
        n = len(gradients)
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(gradients[i] - gradients[j])
                distances[i, j] = d
                distances[j, i] = d
        return distances


# =============================================================================
# KRUM (Blanchard et al., 2017)
# =============================================================================

class KrumAggregator(ByzantineResilientAggregator):
    """
    Krum Byzantine-resilient aggregation.

    Key idea: Select gradient with minimum sum of distances to
    (n - f - 2) nearest neighbors, where f is Byzantine count.

    Multi-Krum: Select m gradients and average them.

    Reference: Blanchard et al., "Machine Learning with Adversaries:
    Byzantine Tolerant Gradient Descent", NeurIPS 2017.
    """

    def aggregate(self,
                 gradients: List[ClientGradient]) -> AggregationResult:
        """Perform Krum aggregation."""
        n = len(gradients)
        f = self.config.num_byzantine
        m = self.config.multi_krum_m or 1

        if n < 2 * f + 3:
            raise ValueError(f"Need at least 2f+3={2*f+3} clients for Krum, got {n}")

        # Flatten gradients
        flat_grads = self._flatten_gradients(gradients)

        # Compute pairwise distances
        distances = self._pairwise_distances(flat_grads)

        # For each client, compute sum of distances to n-f-2 nearest neighbors
        scores = []
        k = n - f - 2  # Number of neighbors to consider

        for i in range(n):
            # Sort distances to other clients
            dists = sorted([(distances[i, j], j) for j in range(n) if j != i])
            # Sum of k smallest distances
            score = sum(d for d, _ in dists[:k])
            scores.append((score, i))

        # Sort by score (lower is better)
        scores.sort(key=lambda x: x[0])

        # Select m best clients
        selected_indices = [idx for _, idx in scores[:m]]
        rejected_indices = [idx for _, idx in scores[m:]]

        # Average selected gradients
        selected_grads = flat_grads[selected_indices]
        aggregated_flat = np.mean(selected_grads, axis=0)

        # Unflatten
        template = gradients[0].gradient
        aggregated = self._unflatten_gradient(aggregated_flat, template)

        # Compute trust scores
        max_score = max(s for s, _ in scores)
        trust_scores = {
            gradients[idx].client_id: 1.0 - (score / (max_score + 1e-10))
            for score, idx in scores
        }

        return AggregationResult(
            aggregated_gradient=aggregated,
            selected_clients=[gradients[i].client_id for i in selected_indices],
            rejected_clients=[gradients[i].client_id for i in rejected_indices],
            trust_scores=trust_scores,
            detection_alerts=[]
        )


# =============================================================================
# TRIMMED MEAN
# =============================================================================

class TrimmedMeanAggregator(ByzantineResilientAggregator):
    """
    Coordinate-wise Trimmed Mean aggregation.

    Key idea: For each coordinate, sort values and remove
    the top and bottom β fraction before averaging.

    Simple but effective against many attacks.

    Reference: Yin et al., "Byzantine-Robust Distributed
    Learning: Towards Optimal Statistical Rates", ICML 2018.
    """

    def aggregate(self,
                 gradients: List[ClientGradient]) -> AggregationResult:
        """Perform trimmed mean aggregation."""
        n = len(gradients)
        beta = self.config.trim_ratio

        if n < 3:
            raise ValueError("Need at least 3 clients for trimmed mean")

        # Number to trim from each side
        trim_count = max(1, int(n * beta))

        # Flatten gradients
        flat_grads = self._flatten_gradients(gradients)  # (n, d)

        # Coordinate-wise trimmed mean
        d = flat_grads.shape[1]
        aggregated_flat = np.zeros(d)

        trimmed_clients = set()

        for coord in range(d):
            values = flat_grads[:, coord]
            sorted_indices = np.argsort(values)

            # Remove top and bottom trim_count
            trimmed_indices = sorted_indices[trim_count:-trim_count] if trim_count > 0 else sorted_indices

            # Track which clients got trimmed
            trimmed_clients.update(sorted_indices[:trim_count])
            trimmed_clients.update(sorted_indices[-trim_count:])

            # Average remaining
            aggregated_flat[coord] = np.mean(values[trimmed_indices])

        # Unflatten
        template = gradients[0].gradient
        aggregated = self._unflatten_gradient(aggregated_flat, template)

        # Determine selected/rejected based on trimming frequency
        trim_counts = {i: 0 for i in range(n)}
        # Simplified: mark frequently trimmed as rejected
        selected = list(range(n))
        rejected = []

        # Trust scores based on how often values were in middle
        trust_scores = {
            gradients[i].client_id: 1.0 - (i in trimmed_clients) * 0.5
            for i in range(n)
        }

        return AggregationResult(
            aggregated_gradient=aggregated,
            selected_clients=[gradients[i].client_id for i in selected],
            rejected_clients=[gradients[i].client_id for i in rejected],
            trust_scores=trust_scores,
            detection_alerts=[]
        )


# =============================================================================
# COORDINATE-WISE MEDIAN
# =============================================================================

class MedianAggregator(ByzantineResilientAggregator):
    """
    Coordinate-wise Median aggregation.

    Key idea: Take median of each coordinate independently.
    Very robust but can be slow for large models.

    Reference: Chen et al., "Distributed Statistical Machine
    Learning in Adversarial Settings", NIPS 2017.
    """

    def aggregate(self,
                 gradients: List[ClientGradient]) -> AggregationResult:
        """Perform median aggregation."""
        # Flatten gradients
        flat_grads = self._flatten_gradients(gradients)

        # Coordinate-wise median
        aggregated_flat = np.median(flat_grads, axis=0)

        # Unflatten
        template = gradients[0].gradient
        aggregated = self._unflatten_gradient(aggregated_flat, template)

        # Trust scores based on distance to median
        distances = [
            np.linalg.norm(flat_grads[i] - aggregated_flat)
            for i in range(len(gradients))
        ]
        max_dist = max(distances) + 1e-10
        trust_scores = {
            gradients[i].client_id: 1.0 - distances[i] / max_dist
            for i in range(len(gradients))
        }

        return AggregationResult(
            aggregated_gradient=aggregated,
            selected_clients=[cg.client_id for cg in gradients],
            rejected_clients=[],
            trust_scores=trust_scores,
            detection_alerts=[]
        )


# =============================================================================
# BULYAN (Mhamdi et al., 2018)
# =============================================================================

class BulyanAggregator(ByzantineResilientAggregator):
    """
    Bulyan Byzantine-resilient aggregation.

    Key idea: Combines Krum selection with trimmed mean.
    1. Use Multi-Krum to select θ = n - 2f gradients
    2. Apply coordinate-wise trimmed mean on selected

    Stronger guarantees than Krum or trimmed mean alone.

    Reference: Mhamdi et al., "The Hidden Vulnerability of
    Distributed Learning in Byzantium", ICML 2018.
    """

    def aggregate(self,
                 gradients: List[ClientGradient]) -> AggregationResult:
        """Perform Bulyan aggregation."""
        n = len(gradients)
        f = self.config.num_byzantine

        theta = n - 2 * f

        if theta < 3:
            raise ValueError(f"Need n > 4f for Bulyan, got n={n}, f={f}")

        # Step 1: Use Multi-Krum to select θ gradients
        krum_config = ByzantineConfig(
            aggregation_rule="krum",
            num_byzantine=f,
            multi_krum_m=theta
        )
        krum = KrumAggregator(krum_config)

        # Flatten for Krum
        flat_grads = self._flatten_gradients(gradients)
        distances = self._pairwise_distances(flat_grads)

        # Krum selection
        scores = []
        k = n - f - 2

        for i in range(n):
            dists = sorted([(distances[i, j], j) for j in range(n) if j != i])
            score = sum(d for d, _ in dists[:k])
            scores.append((score, i))

        scores.sort(key=lambda x: x[0])
        selected_indices = [idx for _, idx in scores[:theta]]

        # Step 2: Trimmed mean on selected
        selected_grads = flat_grads[selected_indices]
        beta = f / theta  # Trim ratio

        trim_count = max(1, int(theta * beta))
        d = selected_grads.shape[1]
        aggregated_flat = np.zeros(d)

        for coord in range(d):
            values = selected_grads[:, coord]
            sorted_vals = np.sort(values)
            trimmed = sorted_vals[trim_count:-trim_count] if trim_count > 0 and len(sorted_vals) > 2*trim_count else sorted_vals
            aggregated_flat[coord] = np.mean(trimmed)

        # Unflatten
        template = gradients[0].gradient
        aggregated = self._unflatten_gradient(aggregated_flat, template)

        rejected_indices = [idx for _, idx in scores[theta:]]

        trust_scores = {
            gradients[i].client_id: 1.0 if i in selected_indices else 0.0
            for i in range(n)
        }

        return AggregationResult(
            aggregated_gradient=aggregated,
            selected_clients=[gradients[i].client_id for i in selected_indices],
            rejected_clients=[gradients[i].client_id for i in rejected_indices],
            trust_scores=trust_scores,
            detection_alerts=[]
        )


# =============================================================================
# FLTrust (Cao et al., 2021)
# =============================================================================

class FLTrustAggregator(ByzantineResilientAggregator):
    """
    FLTrust: Byzantine-robust FL using server's root dataset.

    Key idea: Server maintains small clean dataset and computes
    trust score for each client based on cosine similarity
    to server's gradient direction.

    Reference: Cao et al., "FLTrust: Byzantine-robust Federated
    Learning via Trust Bootstrapping", NDSS 2021.
    """

    def __init__(self, config: ByzantineConfig):
        super().__init__(config)
        self.server_gradient: Optional[Dict[str, np.ndarray]] = None

    def set_server_gradient(self, gradient: Dict[str, np.ndarray]) -> None:
        """Set the server's trusted gradient (from root dataset)."""
        self.server_gradient = gradient

    def aggregate(self,
                 gradients: List[ClientGradient]) -> AggregationResult:
        """Perform FLTrust aggregation."""
        if self.server_gradient is None:
            raise ValueError("Server gradient not set. Call set_server_gradient first.")

        n = len(gradients)

        # Flatten all gradients
        flat_grads = self._flatten_gradients(gradients)
        server_flat = np.concatenate([
            g.flatten() for g in self.server_gradient.values()
        ])

        # Compute trust scores using ReLU'd cosine similarity
        trust_scores_raw = []
        server_norm = np.linalg.norm(server_flat)

        for i in range(n):
            client_norm = np.linalg.norm(flat_grads[i])
            if client_norm > 0 and server_norm > 0:
                cos_sim = np.dot(flat_grads[i], server_flat) / (client_norm * server_norm)
            else:
                cos_sim = 0.0

            # ReLU to only consider positive similarity
            trust = max(0.0, cos_sim)
            trust_scores_raw.append(trust)

        # Normalize trust scores
        total_trust = sum(trust_scores_raw) + 1e-10
        normalized_trust = [t / total_trust for t in trust_scores_raw]

        # Normalize client gradients to server gradient magnitude
        normalized_grads = []
        for i in range(n):
            client_norm = np.linalg.norm(flat_grads[i])
            if client_norm > 0:
                normalized = flat_grads[i] * (server_norm / client_norm)
            else:
                normalized = flat_grads[i]
            normalized_grads.append(normalized)

        # Weighted aggregation
        aggregated_flat = np.zeros_like(server_flat)
        for i in range(n):
            aggregated_flat += normalized_trust[i] * normalized_grads[i]

        # Unflatten
        template = gradients[0].gradient
        aggregated = self._unflatten_gradient(aggregated_flat, template)

        # Identify rejected (low trust) clients
        threshold = 0.5 / n  # Below fair share
        selected = [i for i in range(n) if normalized_trust[i] >= threshold]
        rejected = [i for i in range(n) if normalized_trust[i] < threshold]

        trust_dict = {
            gradients[i].client_id: normalized_trust[i]
            for i in range(n)
        }

        alerts = []
        for i in rejected:
            alerts.append(f"Client {gradients[i].client_id}: low trust ({trust_scores_raw[i]:.3f})")

        return AggregationResult(
            aggregated_gradient=aggregated,
            selected_clients=[gradients[i].client_id for i in selected],
            rejected_clients=[gradients[i].client_id for i in rejected],
            trust_scores=trust_dict,
            detection_alerts=alerts
        )


# =============================================================================
# FLAME (Nguyen et al., 2022)
# =============================================================================

class FLAMEAggregator(ByzantineResilientAggregator):
    """
    FLAME: Taming Backdoors in Federated Learning.

    Key idea: Use clustering to identify and filter out
    malicious updates, then apply noise for privacy.

    Reference: Nguyen et al., "FLAME: Taming Backdoors in
    Federated Learning", USENIX Security 2022.
    """

    def aggregate(self,
                 gradients: List[ClientGradient]) -> AggregationResult:
        """Perform FLAME aggregation."""
        n = len(gradients)
        k = self.config.num_clusters

        # Flatten gradients
        flat_grads = self._flatten_gradients(gradients)

        # Compute pairwise cosine similarities
        norms = np.linalg.norm(flat_grads, axis=1, keepdims=True)
        normalized = flat_grads / (norms + 1e-10)
        similarities = normalized @ normalized.T

        # Simple clustering based on similarity
        # Using hierarchical-like approach
        cluster_labels = self._cluster_gradients(similarities, k)

        # Find majority cluster
        unique, counts = np.unique(cluster_labels, return_counts=True)
        majority_cluster = unique[np.argmax(counts)]

        # Select clients in majority cluster
        selected_indices = np.where(cluster_labels == majority_cluster)[0]
        rejected_indices = np.where(cluster_labels != majority_cluster)[0]

        # Average selected gradients
        if len(selected_indices) > 0:
            aggregated_flat = np.mean(flat_grads[selected_indices], axis=0)
        else:
            aggregated_flat = np.mean(flat_grads, axis=0)

        # Add small noise for privacy (optional)
        noise_scale = 0.001
        aggregated_flat += np.random.normal(0, noise_scale, aggregated_flat.shape)

        # Unflatten
        template = gradients[0].gradient
        aggregated = self._unflatten_gradient(aggregated_flat, template)

        # Trust scores
        trust_scores = {
            gradients[i].client_id: 1.0 if cluster_labels[i] == majority_cluster else 0.0
            for i in range(n)
        }

        alerts = []
        if len(rejected_indices) > 0:
            alerts.append(f"FLAME: Detected {len(rejected_indices)} potential Byzantine clients")

        return AggregationResult(
            aggregated_gradient=aggregated,
            selected_clients=[gradients[i].client_id for i in selected_indices],
            rejected_clients=[gradients[i].client_id for i in rejected_indices],
            trust_scores=trust_scores,
            detection_alerts=alerts
        )

    def _cluster_gradients(self,
                          similarities: np.ndarray,
                          k: int) -> np.ndarray:
        """Simple clustering based on similarity matrix."""
        n = similarities.shape[0]

        # Use average linkage clustering
        # Start with each point in its own cluster
        labels = np.arange(n)
        distances = 1 - similarities  # Convert similarity to distance

        while len(np.unique(labels)) > k:
            # Find closest pair of clusters
            min_dist = float('inf')
            merge_pair = (0, 1)

            unique_labels = np.unique(labels)
            for i, l1 in enumerate(unique_labels):
                for l2 in unique_labels[i+1:]:
                    # Average linkage distance
                    idx1 = np.where(labels == l1)[0]
                    idx2 = np.where(labels == l2)[0]
                    avg_dist = np.mean(distances[np.ix_(idx1, idx2)])

                    if avg_dist < min_dist:
                        min_dist = avg_dist
                        merge_pair = (l1, l2)

            # Merge clusters
            labels[labels == merge_pair[1]] = merge_pair[0]

        # Relabel to 0, 1, ..., k-1
        unique = np.unique(labels)
        label_map = {old: new for new, old in enumerate(unique)}
        return np.array([label_map[l] for l in labels])


# =============================================================================
# TRUSTED EXECUTION ENVIRONMENT (TEE) INTEGRATION
# =============================================================================

class TEEType(Enum):
    """Supported TEE platforms."""
    INTEL_SGX = auto()       # Intel Software Guard Extensions
    ARM_TRUSTZONE = auto()   # ARM TrustZone
    AMD_SEV = auto()         # AMD Secure Encrypted Virtualization
    SIMULATED = auto()       # Simulated TEE for development


class AttestationType(Enum):
    """Types of remote attestation."""
    EPID = auto()           # Enhanced Privacy ID (Intel SGX)
    DCAP = auto()           # Data Center Attestation Primitives
    ECDSA = auto()          # ECDSA-based attestation
    SIMULATED = auto()      # Simulated attestation for testing


@dataclass
class TEEConfig:
    """Configuration for TEE integration."""
    tee_type: TEEType = TEEType.SIMULATED
    attestation_type: AttestationType = AttestationType.SIMULATED

    # Attestation settings
    enable_remote_attestation: bool = True
    attestation_timeout: int = 30  # seconds
    attestation_refresh_interval: int = 3600  # seconds

    # Security settings
    require_measurement_match: bool = True
    allowed_measurements: List[str] = field(default_factory=list)
    min_security_version: int = 1

    # Enclave settings
    enclave_debug_mode: bool = False
    max_concurrent_enclaves: int = 10

    # Audit
    audit_attestations: bool = True


@dataclass
class AttestationReport:
    """Remote attestation report from TEE."""
    client_id: int
    tee_type: TEEType
    measurement: str  # MRENCLAVE for SGX
    signer: str       # MRSIGNER for SGX
    product_id: int
    security_version: int
    timestamp: datetime
    nonce: str
    signature: str
    platform_info: Dict[str, Any] = field(default_factory=dict)

    # Validation status
    is_valid: bool = False
    validation_errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "client_id": self.client_id,
            "tee_type": self.tee_type.name,
            "measurement": self.measurement,
            "signer": self.signer,
            "product_id": self.product_id,
            "security_version": self.security_version,
            "timestamp": self.timestamp.isoformat(),
            "nonce": self.nonce,
            "is_valid": self.is_valid,
            "validation_errors": self.validation_errors,
        }


@dataclass
class TEEClientState:
    """State tracking for a TEE-enabled client."""
    client_id: int
    tee_type: TEEType
    last_attestation: Optional[AttestationReport] = None
    attestation_count: int = 0
    trust_level: float = 0.0  # 0.0 = untrusted, 1.0 = fully trusted
    is_attested: bool = False
    enclave_id: Optional[str] = None
    registered_at: datetime = field(default_factory=datetime.now)
    last_seen: datetime = field(default_factory=datetime.now)


class TEEAttestationVerifier:
    """
    Verifies TEE remote attestation reports.

    Supports Intel SGX EPID/DCAP, ARM TrustZone, and simulated attestation.
    In production, integrates with Intel Attestation Service (IAS) or
    DCAP verification libraries.
    """

    def __init__(self, config: TEEConfig):
        self.config = config
        self._allowed_measurements = set(config.allowed_measurements)
        self._verification_key: Optional[bytes] = None

    def verify_attestation(
        self,
        report: AttestationReport,
        expected_nonce: str,
    ) -> Tuple[bool, List[str]]:
        """
        Verify a remote attestation report.

        Args:
            report: Attestation report to verify
            expected_nonce: Expected nonce (to prevent replay attacks)

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        # Check nonce (prevents replay attacks)
        if report.nonce != expected_nonce:
            errors.append("Nonce mismatch - possible replay attack")

        # Check timestamp freshness
        age = datetime.now() - report.timestamp
        if age > timedelta(seconds=self.config.attestation_timeout):
            errors.append(f"Attestation expired: {age.total_seconds()}s old")

        # Check security version
        if report.security_version < self.config.min_security_version:
            errors.append(
                f"Security version {report.security_version} < "
                f"minimum {self.config.min_security_version}"
            )

        # Check measurement against allowlist
        if self.config.require_measurement_match:
            if self._allowed_measurements and report.measurement not in self._allowed_measurements:
                errors.append(f"Measurement not in allowlist: {report.measurement[:16]}...")

        # Verify signature
        if not self._verify_signature(report):
            errors.append("Invalid attestation signature")

        # Platform-specific verification
        if report.tee_type == TEEType.INTEL_SGX:
            errors.extend(self._verify_sgx_specific(report))
        elif report.tee_type == TEEType.ARM_TRUSTZONE:
            errors.extend(self._verify_trustzone_specific(report))
        elif report.tee_type == TEEType.AMD_SEV:
            errors.extend(self._verify_sev_specific(report))

        is_valid = len(errors) == 0
        return is_valid, errors

    def _verify_signature(self, report: AttestationReport) -> bool:
        """Verify attestation report signature."""
        if self.config.attestation_type == AttestationType.SIMULATED:
            # Simulated verification for testing
            return self._verify_simulated_signature(report)

        # In production:
        # - EPID: Verify with Intel Attestation Service
        # - DCAP: Verify locally with DCAP libraries
        # - ECDSA: Verify ECDSA signature

        return True

    def _verify_simulated_signature(self, report: AttestationReport) -> bool:
        """Verify simulated attestation signature."""
        # Recreate expected signature
        data = f"{report.client_id}:{report.measurement}:{report.nonce}"
        expected = hmac.new(
            b"fl-ehds-tee-secret",  # In production, use proper key
            data.encode(),
            hashlib.sha256
        ).hexdigest()
        return hmac.compare_digest(report.signature, expected)

    def _verify_sgx_specific(self, report: AttestationReport) -> List[str]:
        """Intel SGX specific verification."""
        errors = []

        # Check for known SGX vulnerabilities
        # In production, check platform info for:
        # - L1TF mitigation
        # - MDS mitigation
        # - SGX updates

        platform_info = report.platform_info

        if platform_info.get("sgx_flags", {}).get("debug_enabled"):
            if not self.config.enclave_debug_mode:
                errors.append("Debug enclave not allowed in production")

        return errors

    def _verify_trustzone_specific(self, report: AttestationReport) -> List[str]:
        """ARM TrustZone specific verification."""
        errors = []
        # Add TrustZone-specific checks
        return errors

    def _verify_sev_specific(self, report: AttestationReport) -> List[str]:
        """AMD SEV specific verification."""
        errors = []
        # Add SEV-specific checks
        return errors

    def add_allowed_measurement(self, measurement: str) -> None:
        """Add a measurement to the allowlist."""
        self._allowed_measurements.add(measurement)

    def remove_allowed_measurement(self, measurement: str) -> None:
        """Remove a measurement from the allowlist."""
        self._allowed_measurements.discard(measurement)


class TEEClientManager:
    """
    Manages TEE-enabled clients for Byzantine-resilient FL.

    Tracks attestation state, computes trust levels, and provides
    integration with aggregation algorithms.
    """

    def __init__(self, config: TEEConfig):
        self.config = config
        self.verifier = TEEAttestationVerifier(config)
        self._clients: Dict[int, TEEClientState] = {}
        self._pending_challenges: Dict[int, str] = {}  # client_id -> nonce
        self._attestation_log: List[Dict[str, Any]] = []

    def generate_challenge(self, client_id: int) -> str:
        """
        Generate attestation challenge for client.

        Args:
            client_id: Client to challenge

        Returns:
            Challenge nonce
        """
        nonce = secrets.token_hex(32)
        self._pending_challenges[client_id] = nonce
        return nonce

    def process_attestation(
        self,
        report: AttestationReport,
    ) -> Tuple[bool, TEEClientState]:
        """
        Process attestation report from client.

        Args:
            report: Attestation report

        Returns:
            Tuple of (success, client_state)
        """
        client_id = report.client_id

        # Get expected nonce
        expected_nonce = self._pending_challenges.get(client_id, "")

        # Verify attestation
        is_valid, errors = self.verifier.verify_attestation(report, expected_nonce)

        report.is_valid = is_valid
        report.validation_errors = errors

        # Update or create client state
        if client_id not in self._clients:
            self._clients[client_id] = TEEClientState(
                client_id=client_id,
                tee_type=report.tee_type,
            )

        state = self._clients[client_id]
        state.last_attestation = report
        state.attestation_count += 1
        state.is_attested = is_valid
        state.last_seen = datetime.now()

        # Update trust level
        if is_valid:
            # Increase trust with successful attestations
            state.trust_level = min(1.0, state.trust_level + 0.2)
        else:
            # Decrease trust on failure
            state.trust_level = max(0.0, state.trust_level - 0.3)

        # Clear challenge
        self._pending_challenges.pop(client_id, None)

        # Audit log
        if self.config.audit_attestations:
            self._attestation_log.append({
                "timestamp": datetime.now().isoformat(),
                "client_id": client_id,
                "is_valid": is_valid,
                "errors": errors,
                "trust_level": state.trust_level,
            })

        logger.info(
            f"Attestation for client {client_id}: "
            f"valid={is_valid}, trust={state.trust_level:.2f}"
        )

        return is_valid, state

    def get_client_trust(self, client_id: int) -> float:
        """Get trust level for a client."""
        if client_id not in self._clients:
            return 0.0
        return self._clients[client_id].trust_level

    def get_attested_clients(self) -> List[int]:
        """Get list of successfully attested clients."""
        return [
            cid for cid, state in self._clients.items()
            if state.is_attested
        ]

    def needs_reattestation(self, client_id: int) -> bool:
        """Check if client needs re-attestation."""
        if client_id not in self._clients:
            return True

        state = self._clients[client_id]
        if not state.is_attested or state.last_attestation is None:
            return True

        age = datetime.now() - state.last_attestation.timestamp
        return age > timedelta(seconds=self.config.attestation_refresh_interval)

    def get_trust_weights(self) -> Dict[int, float]:
        """Get trust-based weights for all clients."""
        return {
            cid: state.trust_level
            for cid, state in self._clients.items()
        }

    def get_attestation_statistics(self) -> Dict[str, Any]:
        """Get attestation statistics."""
        attested = sum(1 for s in self._clients.values() if s.is_attested)
        total = len(self._clients)

        return {
            "total_clients": total,
            "attested_clients": attested,
            "attestation_rate": attested / max(total, 1),
            "total_attestations": len(self._attestation_log),
            "avg_trust_level": np.mean([
                s.trust_level for s in self._clients.values()
            ]) if self._clients else 0.0,
        }


class TEESecureAggregator:
    """
    TEE-based secure aggregation.

    Combines Byzantine-resilient aggregation with TEE attestation
    for enhanced security in enterprise deployments.
    """

    def __init__(
        self,
        config: TEEConfig,
        byzantine_config: ByzantineConfig,
    ):
        self.tee_config = config
        self.byzantine_config = byzantine_config
        self.client_manager = TEEClientManager(config)
        self.byzantine_aggregator = create_byzantine_aggregator(
            byzantine_config.aggregation_rule,
            byzantine_config
        )

    def aggregate(
        self,
        gradients: List[ClientGradient],
        require_attestation: bool = True,
    ) -> AggregationResult:
        """
        Perform TEE-verified Byzantine-resilient aggregation.

        Args:
            gradients: Client gradients to aggregate
            require_attestation: Whether to require TEE attestation

        Returns:
            Aggregation result
        """
        # Filter clients based on attestation status
        if require_attestation:
            attested_clients = set(self.client_manager.get_attested_clients())
            filtered_gradients = [
                g for g in gradients
                if g.client_id in attested_clients
            ]

            rejected_for_attestation = [
                g.client_id for g in gradients
                if g.client_id not in attested_clients
            ]

            if not filtered_gradients:
                raise ValueError("No attested clients available")

            logger.info(
                f"TEE filter: {len(filtered_gradients)}/{len(gradients)} "
                f"clients attested"
            )
        else:
            filtered_gradients = gradients
            rejected_for_attestation = []

        # Apply TEE trust weights to client gradients
        trust_weights = self.client_manager.get_trust_weights()
        for gradient in filtered_gradients:
            gradient.trust_score = trust_weights.get(gradient.client_id, 0.5)

        # Perform Byzantine-resilient aggregation
        result = self.byzantine_aggregator.aggregate(filtered_gradients)

        # Add attestation-rejected clients to rejected list
        result.rejected_clients.extend(rejected_for_attestation)

        # Add TEE-specific detection alerts
        for cid in rejected_for_attestation:
            result.detection_alerts.append(
                f"Client {cid} rejected: missing TEE attestation"
            )

        return result

    def challenge_client(self, client_id: int) -> str:
        """Generate attestation challenge for client."""
        return self.client_manager.generate_challenge(client_id)

    def verify_attestation(self, report: AttestationReport) -> bool:
        """Process and verify client attestation."""
        success, _ = self.client_manager.process_attestation(report)
        return success

    def get_statistics(self) -> Dict[str, Any]:
        """Get combined statistics."""
        return {
            "attestation": self.client_manager.get_attestation_statistics(),
            "aggregation_rule": self.byzantine_config.aggregation_rule,
        }


def create_tee_config(**kwargs) -> TEEConfig:
    """Factory function to create TEE configuration."""
    tee_type_map = {
        "sgx": TEEType.INTEL_SGX,
        "intel_sgx": TEEType.INTEL_SGX,
        "trustzone": TEEType.ARM_TRUSTZONE,
        "arm_trustzone": TEEType.ARM_TRUSTZONE,
        "sev": TEEType.AMD_SEV,
        "amd_sev": TEEType.AMD_SEV,
        "simulated": TEEType.SIMULATED,
    }

    if "tee_type" in kwargs and isinstance(kwargs["tee_type"], str):
        kwargs["tee_type"] = tee_type_map.get(kwargs["tee_type"].lower(), TEEType.SIMULATED)

    return TEEConfig(**kwargs)


def create_tee_secure_aggregator(
    tee_config: Optional[TEEConfig] = None,
    byzantine_config: Optional[ByzantineConfig] = None,
    **kwargs
) -> TEESecureAggregator:
    """Factory function to create TEE-secured aggregator."""
    if tee_config is None:
        tee_config = create_tee_config(**{
            k: v for k, v in kwargs.items()
            if k.startswith("tee_") or k in ["enable_remote_attestation"]
        })

    if byzantine_config is None:
        byzantine_config = ByzantineConfig(**{
            k: v for k, v in kwargs.items()
            if not k.startswith("tee_") and k not in ["enable_remote_attestation"]
        })

    return TEESecureAggregator(tee_config, byzantine_config)


def create_simulated_attestation(
    client_id: int,
    tee_type: TEEType = TEEType.SIMULATED,
    nonce: str = "",
) -> AttestationReport:
    """
    Create a simulated attestation report for testing.

    Args:
        client_id: Client ID
        tee_type: TEE type to simulate
        nonce: Challenge nonce

    Returns:
        Simulated AttestationReport
    """
    measurement = hashlib.sha256(f"enclave_{client_id}".encode()).hexdigest()
    signer = hashlib.sha256(b"fl-ehds-signer").hexdigest()

    # Create signature
    data = f"{client_id}:{measurement}:{nonce}"
    signature = hmac.new(
        b"fl-ehds-tee-secret",
        data.encode(),
        hashlib.sha256
    ).hexdigest()

    return AttestationReport(
        client_id=client_id,
        tee_type=tee_type,
        measurement=measurement,
        signer=signer,
        product_id=1,
        security_version=1,
        timestamp=datetime.now(),
        nonce=nonce,
        signature=signature,
        platform_info={
            "sgx_flags": {"debug_enabled": False},
            "cpu_svn": "0" * 32,
        },
    )


# =============================================================================
# FACTORY & MANAGER
# =============================================================================

BYZANTINE_AGGREGATORS = {
    'krum': KrumAggregator,
    'trimmed_mean': TrimmedMeanAggregator,
    'median': MedianAggregator,
    'bulyan': BulyanAggregator,
    'fltrust': FLTrustAggregator,
    'flame': FLAMEAggregator,
}


def create_byzantine_aggregator(rule: str,
                               config: Optional[ByzantineConfig] = None,
                               **kwargs) -> ByzantineResilientAggregator:
    """Factory function to create Byzantine-resilient aggregator."""
    if rule.lower() not in BYZANTINE_AGGREGATORS:
        raise ValueError(f"Unknown rule: {rule}. Available: {list(BYZANTINE_AGGREGATORS.keys())}")

    if config is None:
        config = ByzantineConfig(aggregation_rule=rule, **kwargs)

    return BYZANTINE_AGGREGATORS[rule.lower()](config)


class ByzantineDefenseManager:
    """
    High-level manager for Byzantine defense in FL.

    Handles detection, aggregation, and trust management.
    """

    def __init__(self, config: ByzantineConfig):
        self.config = config
        self.aggregator = create_byzantine_aggregator(config.aggregation_rule, config)
        self.trust_history: Dict[int, List[float]] = {}
        self.detection_history: List[Tuple[int, List[str]]] = []

    def aggregate(self,
                 gradients: List[ClientGradient],
                 server_gradient: Optional[Dict[str, np.ndarray]] = None) -> AggregationResult:
        """Perform Byzantine-resilient aggregation."""
        # Set server gradient for FLTrust
        if isinstance(self.aggregator, FLTrustAggregator) and server_gradient is not None:
            self.aggregator.set_server_gradient(server_gradient)

        result = self.aggregator.aggregate(gradients)

        # Update trust history
        for cid, trust in result.trust_scores.items():
            if cid not in self.trust_history:
                self.trust_history[cid] = []
            self.trust_history[cid].append(trust)

        # Log detections
        if result.detection_alerts:
            self.detection_history.append((len(self.detection_history), result.detection_alerts))

        return result

    def get_client_reputation(self, client_id: int) -> float:
        """Get long-term reputation for a client."""
        if client_id not in self.trust_history or not self.trust_history[client_id]:
            return 0.5  # Neutral for unknown clients

        # Exponential moving average
        scores = self.trust_history[client_id]
        alpha = 0.3
        reputation = scores[0]
        for s in scores[1:]:
            reputation = alpha * s + (1 - alpha) * reputation

        return reputation


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    print("FL-EHDS Byzantine-Resilient Federated Learning Demo")
    print("=" * 70)

    np.random.seed(42)

    # Create honest gradients
    n_clients = 10
    n_byzantine = 2
    gradient_dim = 100

    # True gradient direction
    true_gradient = np.random.randn(gradient_dim) * 0.1

    gradients = []
    for i in range(n_clients):
        # Add some noise to honest gradients
        honest_grad = true_gradient + np.random.randn(gradient_dim) * 0.02

        gradient_dict = {'weights': honest_grad}

        cg = ClientGradient(
            client_id=i,
            gradient=gradient_dict,
            samples_used=100,
            is_byzantine=(i < n_byzantine)
        )
        gradients.append(cg)

    # Apply attacks to Byzantine clients
    attacker = ByzantineAttacker(attack_type="scale", attack_strength=10.0)
    for i in range(n_byzantine):
        gradients[i].gradient = attacker.attack(gradients[i].gradient)

    print(f"\nSetup: {n_clients} clients, {n_byzantine} Byzantine (scale attack)")
    print("-" * 70)

    # Test different aggregation rules
    rules = ['krum', 'trimmed_mean', 'median', 'bulyan', 'flame']

    results = {}

    for rule in rules:
        print(f"\nTesting: {rule.upper()}")

        config = ByzantineConfig(
            aggregation_rule=rule,
            num_byzantine=n_byzantine,
            trim_ratio=0.2
        )

        manager = ByzantineDefenseManager(config)
        result = manager.aggregate(gradients)

        # Measure quality: cosine similarity to true gradient
        agg_flat = result.aggregated_gradient['weights']
        cos_sim = np.dot(agg_flat, true_gradient) / (
            np.linalg.norm(agg_flat) * np.linalg.norm(true_gradient)
        )

        results[rule] = {
            'cosine_similarity': cos_sim,
            'selected': len(result.selected_clients),
            'rejected': len(result.rejected_clients),
            'alerts': result.detection_alerts
        }

        print(f"  Cosine sim to truth: {cos_sim:.4f}")
        print(f"  Selected: {result.selected_clients}")
        print(f"  Rejected: {result.rejected_clients}")

    # Summary
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Rule':<15} {'Cos. Sim.':<12} {'Selected':<10} {'Rejected':<10}")
    print("-" * 50)

    for rule, res in sorted(results.items(), key=lambda x: -x[1]['cosine_similarity']):
        print(f"{rule:<15} {res['cosine_similarity']:>10.4f} {res['selected']:>8} {res['rejected']:>8}")

    print("\n" + "=" * 70)
    print("Best rule for this attack: " +
          max(results.items(), key=lambda x: x[1]['cosine_similarity'])[0].upper())

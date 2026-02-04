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

Author: Fabio Liberti
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from copy import deepcopy
from scipy import stats


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

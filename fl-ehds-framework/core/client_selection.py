"""
Client Selection Strategies for FL-EHDS
========================================

Implementation of intelligent client selection strategies for
federated learning optimization in healthcare settings.

Supported Strategies:
1. Random Selection - Baseline uniform sampling
2. Active Learning - Uncertainty-based selection
3. Importance Sampling - Gradient-based importance
4. Resource-Aware - Consider client capabilities
5. Fairness-Aware - Ensure equitable participation
6. Clustered Selection - Diversity-maximizing selection
7. Loss-Based Selection - Select high-loss clients
8. Oort - Age-based statistical utility

Key References:
- Goetz et al., "Active Federated Learning", 2019
- Cho et al., "Client Selection in FL: Convergence Analysis", 2020
- Lai et al., "Oort: Efficient Federated Learning", OSDI 2021
- Wang et al., "Optimizing Federated Learning", MLSys 2021

EHDS Relevance:
- Handles heterogeneous hospital capabilities
- Optimizes convergence with limited rounds
- Ensures fair cross-border participation

Author: FL-EHDS Framework
License: Apache 2.0
"""

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Constants
# =============================================================================

class SelectionStrategy(Enum):
    """Client selection strategies."""
    RANDOM = "random"
    ACTIVE_LEARNING = "active_learning"
    IMPORTANCE_SAMPLING = "importance_sampling"
    RESOURCE_AWARE = "resource_aware"
    FAIRNESS_AWARE = "fairness_aware"
    CLUSTERED = "clustered"
    LOSS_BASED = "loss_based"
    OORT = "oort"  # Age-based statistical utility
    POWER_OF_CHOICE = "power_of_choice"  # d-choice selection
    CONTRIBUTION_BASED = "contribution_based"


class ClientStatus(Enum):
    """Client availability status."""
    AVAILABLE = "available"
    BUSY = "busy"
    OFFLINE = "offline"
    EXCLUDED = "excluded"  # Opt-out or blacklisted


@dataclass
class ClientProfile:
    """Profile of a federated learning client."""
    client_id: str
    status: ClientStatus = ClientStatus.AVAILABLE

    # Data characteristics
    sample_count: int = 0
    class_distribution: Optional[Dict[int, int]] = None
    data_quality_score: float = 1.0

    # Resource capabilities
    compute_capacity: float = 1.0  # Relative compute power
    bandwidth: float = 1.0  # Relative network speed
    memory_gb: float = 8.0
    gpu_available: bool = False

    # Historical metrics
    rounds_participated: int = 0
    rounds_since_selection: int = 0
    average_train_time: float = 0.0
    average_upload_time: float = 0.0
    success_rate: float = 1.0  # Completion rate
    last_loss: Optional[float] = None
    last_gradient_norm: Optional[float] = None

    # Contribution tracking
    total_contribution: float = 0.0
    average_contribution: float = 0.0

    # Location (for clustering)
    country: Optional[str] = None
    region: Optional[str] = None
    cluster_id: Optional[int] = None

    def get_expected_time(self) -> float:
        """Estimate time to complete training round."""
        base_time = self.average_train_time + self.average_upload_time
        if base_time > 0:
            return base_time
        # Estimate based on resources
        return 60.0 / (self.compute_capacity * self.bandwidth)

    def get_utility_score(self) -> float:
        """Compute utility score for selection."""
        # Combine multiple factors
        data_factor = math.log(1 + self.sample_count)
        quality_factor = self.data_quality_score
        reliability_factor = self.success_rate
        freshness_factor = 1.0 / (1.0 + self.rounds_since_selection)

        return data_factor * quality_factor * reliability_factor * freshness_factor


@dataclass
class SelectionResult:
    """Result of client selection."""
    selected_clients: List[str]
    selection_probabilities: Dict[str, float]
    selection_strategy: SelectionStrategy
    round_number: int
    selection_time: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Client Selection Algorithms
# =============================================================================

class ClientSelector(ABC):
    """Abstract base class for client selection."""

    @abstractmethod
    def select(
        self,
        clients: Dict[str, ClientProfile],
        num_select: int,
        **kwargs,
    ) -> SelectionResult:
        """Select clients for the current round."""
        pass

    def filter_available(
        self,
        clients: Dict[str, ClientProfile],
    ) -> Dict[str, ClientProfile]:
        """Filter to only available clients."""
        return {
            cid: profile
            for cid, profile in clients.items()
            if profile.status == ClientStatus.AVAILABLE
        }


class RandomSelector(ClientSelector):
    """
    Random Client Selection (Baseline).

    Uniformly samples from available clients.
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.RandomState(seed)

    def select(
        self,
        clients: Dict[str, ClientProfile],
        num_select: int,
        **kwargs,
    ) -> SelectionResult:
        """Random uniform selection."""
        available = self.filter_available(clients)

        if len(available) <= num_select:
            selected = list(available.keys())
        else:
            selected = self.rng.choice(
                list(available.keys()),
                size=num_select,
                replace=False
            ).tolist()

        probs = {cid: 1.0 / len(available) for cid in available}

        return SelectionResult(
            selected_clients=selected,
            selection_probabilities=probs,
            selection_strategy=SelectionStrategy.RANDOM,
            round_number=kwargs.get("round_number", 0),
        )


class ActiveLearningSelector(ClientSelector):
    """
    Active Learning-based Selection.

    Selects clients with highest uncertainty or model disagreement.
    Useful when some clients have more informative data.

    Reference: Goetz et al., "Active Federated Learning", 2019
    """

    def __init__(
        self,
        uncertainty_weight: float = 0.5,
        diversity_weight: float = 0.5,
        exploration_rate: float = 0.1,
        seed: Optional[int] = None,
    ):
        self.uncertainty_weight = uncertainty_weight
        self.diversity_weight = diversity_weight
        self.exploration_rate = exploration_rate
        self.rng = np.random.RandomState(seed)
        self._uncertainty_scores: Dict[str, float] = {}

    def update_uncertainty(self, client_id: str, uncertainty: float) -> None:
        """Update uncertainty score for a client."""
        self._uncertainty_scores[client_id] = uncertainty

    def select(
        self,
        clients: Dict[str, ClientProfile],
        num_select: int,
        **kwargs,
    ) -> SelectionResult:
        """
        Select clients based on uncertainty and diversity.
        """
        available = self.filter_available(clients)

        if len(available) <= num_select:
            return SelectionResult(
                selected_clients=list(available.keys()),
                selection_probabilities={cid: 1.0 for cid in available},
                selection_strategy=SelectionStrategy.ACTIVE_LEARNING,
                round_number=kwargs.get("round_number", 0),
            )

        # Compute selection scores
        scores = {}
        for cid, profile in available.items():
            # Uncertainty component
            uncertainty = self._uncertainty_scores.get(cid, 1.0)

            # Diversity component (inverse of recent participation)
            diversity = 1.0 / (1.0 + profile.rounds_participated)

            # Combined score
            scores[cid] = (
                self.uncertainty_weight * uncertainty +
                self.diversity_weight * diversity
            )

        # Exploration: random selection with probability exploration_rate
        selected = []
        remaining = list(available.keys())

        for _ in range(num_select):
            if not remaining:
                break

            if self.rng.random() < self.exploration_rate:
                # Random exploration
                idx = self.rng.randint(len(remaining))
            else:
                # Exploit: select highest score
                remaining_scores = {cid: scores[cid] for cid in remaining}
                idx = remaining.index(max(remaining_scores, key=remaining_scores.get))

            selected.append(remaining.pop(idx))

        # Compute selection probabilities
        total_score = sum(scores.values())
        probs = {cid: s / total_score for cid, s in scores.items()}

        return SelectionResult(
            selected_clients=selected,
            selection_probabilities=probs,
            selection_strategy=SelectionStrategy.ACTIVE_LEARNING,
            round_number=kwargs.get("round_number", 0),
            metadata={"uncertainty_scores": self._uncertainty_scores.copy()},
        )


class ImportanceSamplingSelector(ClientSelector):
    """
    Importance Sampling-based Selection.

    Samples clients proportionally to their gradient importance
    or expected contribution to convergence.

    Reference: Zhao et al., "Federated Learning with Non-IID Data", 2018
    """

    def __init__(
        self,
        use_gradient_norm: bool = True,
        use_loss: bool = True,
        temperature: float = 1.0,
        seed: Optional[int] = None,
    ):
        self.use_gradient_norm = use_gradient_norm
        self.use_loss = use_loss
        self.temperature = temperature
        self.rng = np.random.RandomState(seed)

    def select(
        self,
        clients: Dict[str, ClientProfile],
        num_select: int,
        **kwargs,
    ) -> SelectionResult:
        """
        Select clients with probability proportional to importance.
        """
        available = self.filter_available(clients)

        if len(available) <= num_select:
            return SelectionResult(
                selected_clients=list(available.keys()),
                selection_probabilities={cid: 1.0 for cid in available},
                selection_strategy=SelectionStrategy.IMPORTANCE_SAMPLING,
                round_number=kwargs.get("round_number", 0),
            )

        # Compute importance weights
        weights = {}
        for cid, profile in available.items():
            importance = 1.0

            if self.use_gradient_norm and profile.last_gradient_norm:
                importance *= profile.last_gradient_norm

            if self.use_loss and profile.last_loss:
                importance *= profile.last_loss

            # Data quantity factor
            importance *= math.log(1 + profile.sample_count)

            weights[cid] = importance ** (1.0 / self.temperature)

        # Normalize to probabilities
        total = sum(weights.values())
        probs = {cid: w / total for cid, w in weights.items()}

        # Sample without replacement
        client_ids = list(probs.keys())
        prob_values = np.array([probs[cid] for cid in client_ids])

        # Adjust for sampling without replacement
        selected = []
        remaining_probs = prob_values.copy()

        for _ in range(num_select):
            if np.sum(remaining_probs) <= 0:
                break

            # Normalize remaining
            norm_probs = remaining_probs / np.sum(remaining_probs)

            # Sample
            idx = self.rng.choice(len(client_ids), p=norm_probs)
            selected.append(client_ids[idx])

            # Remove selected
            remaining_probs[idx] = 0

        return SelectionResult(
            selected_clients=selected,
            selection_probabilities=probs,
            selection_strategy=SelectionStrategy.IMPORTANCE_SAMPLING,
            round_number=kwargs.get("round_number", 0),
        )


class ResourceAwareSelector(ClientSelector):
    """
    Resource-Aware Client Selection.

    Considers client compute/network capabilities to minimize
    round completion time (straggler mitigation).
    """

    def __init__(
        self,
        deadline_factor: float = 2.0,  # Allow 2x average time
        prefer_fast: bool = True,
        balance_factor: float = 0.3,  # Balance speed vs data
        seed: Optional[int] = None,
    ):
        self.deadline_factor = deadline_factor
        self.prefer_fast = prefer_fast
        self.balance_factor = balance_factor
        self.rng = np.random.RandomState(seed)

    def select(
        self,
        clients: Dict[str, ClientProfile],
        num_select: int,
        **kwargs,
    ) -> SelectionResult:
        """
        Select clients considering resource constraints.
        """
        available = self.filter_available(clients)
        deadline = kwargs.get("deadline_seconds")

        if len(available) <= num_select:
            return SelectionResult(
                selected_clients=list(available.keys()),
                selection_probabilities={cid: 1.0 for cid in available},
                selection_strategy=SelectionStrategy.RESOURCE_AWARE,
                round_number=kwargs.get("round_number", 0),
            )

        # Estimate completion times
        times = {cid: p.get_expected_time() for cid, p in available.items()}

        # Filter by deadline if specified
        if deadline:
            available = {
                cid: p for cid, p in available.items()
                if times[cid] <= deadline * self.deadline_factor
            }

        if not available:
            # Fallback to fastest clients
            sorted_by_time = sorted(times.items(), key=lambda x: x[1])
            selected = [cid for cid, _ in sorted_by_time[:num_select]]
            return SelectionResult(
                selected_clients=selected,
                selection_probabilities={cid: 0.0 for cid in times},
                selection_strategy=SelectionStrategy.RESOURCE_AWARE,
                round_number=kwargs.get("round_number", 0),
                metadata={"fallback": True, "reason": "deadline_constraint"},
            )

        # Score: balance speed and data quantity
        scores = {}
        max_time = max(times.values()) if times else 1.0
        max_samples = max(p.sample_count for p in available.values()) or 1

        for cid, profile in available.items():
            speed_score = 1.0 - (times[cid] / max_time) if self.prefer_fast else 0.5
            data_score = profile.sample_count / max_samples

            scores[cid] = (
                self.balance_factor * speed_score +
                (1 - self.balance_factor) * data_score
            )

        # Select top scorers
        sorted_clients = sorted(scores.items(), key=lambda x: -x[1])
        selected = [cid for cid, _ in sorted_clients[:num_select]]

        total_score = sum(scores.values())
        probs = {cid: s / total_score for cid, s in scores.items()}

        return SelectionResult(
            selected_clients=selected,
            selection_probabilities=probs,
            selection_strategy=SelectionStrategy.RESOURCE_AWARE,
            round_number=kwargs.get("round_number", 0),
            metadata={"expected_times": times},
        )


class FairnessAwareSelector(ClientSelector):
    """
    Fairness-Aware Client Selection.

    Ensures equitable participation across clients,
    important for EHDS cross-border fairness.
    """

    def __init__(
        self,
        min_participation_rate: float = 0.1,  # Min fraction of rounds
        max_consecutive_skips: int = 5,
        country_balance: bool = True,  # Balance across countries
        seed: Optional[int] = None,
    ):
        self.min_participation_rate = min_participation_rate
        self.max_consecutive_skips = max_consecutive_skips
        self.country_balance = country_balance
        self.rng = np.random.RandomState(seed)

    def select(
        self,
        clients: Dict[str, ClientProfile],
        num_select: int,
        **kwargs,
    ) -> SelectionResult:
        """
        Select clients ensuring fair participation.
        """
        available = self.filter_available(clients)
        total_rounds = kwargs.get("total_rounds", 100)
        current_round = kwargs.get("round_number", 1)

        if len(available) <= num_select:
            return SelectionResult(
                selected_clients=list(available.keys()),
                selection_probabilities={cid: 1.0 for cid in available},
                selection_strategy=SelectionStrategy.FAIRNESS_AWARE,
                round_number=current_round,
            )

        # Priority scoring based on fairness
        priority = {}
        must_select = []  # Clients that must be selected for fairness

        for cid, profile in available.items():
            # Check minimum participation constraint
            expected_participation = current_round * self.min_participation_rate
            if profile.rounds_participated < expected_participation - 1:
                must_select.append(cid)
                priority[cid] = float('inf')
                continue

            # Check consecutive skips
            if profile.rounds_since_selection >= self.max_consecutive_skips:
                must_select.append(cid)
                priority[cid] = float('inf')
                continue

            # Normal priority: inverse of participation rate
            if current_round > 0:
                participation_rate = profile.rounds_participated / current_round
                priority[cid] = 1.0 / (participation_rate + 0.01)
            else:
                priority[cid] = 1.0

        # Start with must-select clients
        selected = must_select[:num_select]

        # If need more, add based on priority
        remaining_slots = num_select - len(selected)
        if remaining_slots > 0:
            remaining_clients = [
                cid for cid in available.keys()
                if cid not in selected
            ]

            # Country balancing
            if self.country_balance and remaining_clients:
                selected.extend(
                    self._country_balanced_selection(
                        remaining_clients,
                        {cid: available[cid] for cid in remaining_clients},
                        priority,
                        remaining_slots,
                    )
                )
            else:
                # Priority-based selection
                sorted_by_priority = sorted(
                    [(cid, priority.get(cid, 0)) for cid in remaining_clients],
                    key=lambda x: -x[1]
                )
                selected.extend([cid for cid, _ in sorted_by_priority[:remaining_slots]])

        total_priority = sum(p for p in priority.values() if p != float('inf'))
        probs = {
            cid: (p / total_priority if p != float('inf') else 1.0)
            for cid, p in priority.items()
        }

        return SelectionResult(
            selected_clients=selected[:num_select],
            selection_probabilities=probs,
            selection_strategy=SelectionStrategy.FAIRNESS_AWARE,
            round_number=current_round,
            metadata={
                "must_select_count": len(must_select),
                "country_balanced": self.country_balance,
            },
        )

    def _country_balanced_selection(
        self,
        client_ids: List[str],
        profiles: Dict[str, ClientProfile],
        priority: Dict[str, float],
        num_select: int,
    ) -> List[str]:
        """Select clients with country balance."""
        # Group by country
        by_country: Dict[str, List[str]] = {}
        for cid in client_ids:
            country = profiles[cid].country or "unknown"
            if country not in by_country:
                by_country[country] = []
            by_country[country].append(cid)

        # Round-robin selection across countries
        selected = []
        countries = list(by_country.keys())

        while len(selected) < num_select and any(by_country.values()):
            for country in countries:
                if len(selected) >= num_select:
                    break
                if by_country[country]:
                    # Select highest priority from this country
                    country_clients = by_country[country]
                    best = max(country_clients, key=lambda c: priority.get(c, 0))
                    selected.append(best)
                    by_country[country].remove(best)

        return selected


class OortSelector(ClientSelector):
    """
    Oort: Efficient Federated Learning Client Selection.

    Combines statistical utility (data importance) with
    system utility (completion time) for selection.

    Reference: Lai et al., "Oort: Efficient Federated Learning
    via Guided Participant Selection", OSDI 2021
    """

    def __init__(
        self,
        exploration_factor: float = 0.9,  # UCB exploration
        sys_utility_weight: float = 0.5,  # Balance stat vs sys utility
        pacer_delta: float = 0.1,  # Deadline pacer
        seed: Optional[int] = None,
    ):
        self.exploration_factor = exploration_factor
        self.sys_utility_weight = sys_utility_weight
        self.pacer_delta = pacer_delta
        self.rng = np.random.RandomState(seed)
        self._round = 0
        self._selection_counts: Dict[str, int] = {}

    def select(
        self,
        clients: Dict[str, ClientProfile],
        num_select: int,
        **kwargs,
    ) -> SelectionResult:
        """
        Oort selection combining statistical and system utility.
        """
        available = self.filter_available(clients)
        self._round += 1

        if len(available) <= num_select:
            return SelectionResult(
                selected_clients=list(available.keys()),
                selection_probabilities={cid: 1.0 for cid in available},
                selection_strategy=SelectionStrategy.OORT,
                round_number=self._round,
            )

        # Compute utilities
        utilities = {}
        for cid, profile in available.items():
            # Statistical utility (data importance)
            stat_util = self._statistical_utility(cid, profile)

            # System utility (speed)
            sys_util = self._system_utility(profile)

            # Combined with exploration bonus (UCB-style)
            selection_count = self._selection_counts.get(cid, 0)
            exploration_bonus = math.sqrt(
                2 * math.log(self._round + 1) / (selection_count + 1)
            )

            utilities[cid] = (
                (1 - self.sys_utility_weight) * stat_util +
                self.sys_utility_weight * sys_util +
                self.exploration_factor * exploration_bonus
            )

        # Select top utilities
        sorted_clients = sorted(utilities.items(), key=lambda x: -x[1])
        selected = [cid for cid, _ in sorted_clients[:num_select]]

        # Update selection counts
        for cid in selected:
            self._selection_counts[cid] = self._selection_counts.get(cid, 0) + 1

        total_util = sum(utilities.values())
        probs = {cid: u / total_util for cid, u in utilities.items()}

        return SelectionResult(
            selected_clients=selected,
            selection_probabilities=probs,
            selection_strategy=SelectionStrategy.OORT,
            round_number=self._round,
            metadata={
                "statistical_utilities": {cid: self._statistical_utility(cid, available[cid]) for cid in selected},
                "system_utilities": {cid: self._system_utility(available[cid]) for cid in selected},
            },
        )

    def _statistical_utility(self, client_id: str, profile: ClientProfile) -> float:
        """Compute statistical utility based on data characteristics."""
        # Data quantity (sqrt to avoid domination by large clients)
        quantity_score = math.sqrt(profile.sample_count)

        # Loss-based importance (higher loss = more valuable)
        loss_score = profile.last_loss if profile.last_loss else 1.0

        # Freshness (penalize stale data/infrequent participation)
        freshness = 1.0 / (1.0 + profile.rounds_since_selection)

        return quantity_score * loss_score * freshness * profile.data_quality_score

    def _system_utility(self, profile: ClientProfile) -> float:
        """Compute system utility based on expected completion time."""
        expected_time = profile.get_expected_time()
        # Inverse time (faster is better)
        return 1.0 / (expected_time + 1.0) * profile.success_rate


class ClusteredSelector(ClientSelector):
    """
    Clustered Client Selection.

    Maximizes diversity by selecting from different clusters
    (e.g., different hospitals, regions, or data distributions).
    """

    def __init__(
        self,
        num_clusters: int = 5,
        diversity_weight: float = 0.7,
        seed: Optional[int] = None,
    ):
        self.num_clusters = num_clusters
        self.diversity_weight = diversity_weight
        self.rng = np.random.RandomState(seed)
        self._cluster_assignments: Dict[str, int] = {}

    def assign_clusters(
        self,
        clients: Dict[str, ClientProfile],
        features: Optional[Dict[str, np.ndarray]] = None,
    ) -> None:
        """Assign clients to clusters based on data characteristics."""
        if features:
            # Use provided features for clustering
            self._cluster_with_features(clients, features)
        else:
            # Use client metadata (country, region, class distribution)
            self._cluster_by_metadata(clients)

    def _cluster_with_features(
        self,
        clients: Dict[str, ClientProfile],
        features: Dict[str, np.ndarray],
    ) -> None:
        """K-means style clustering on feature representations."""
        client_ids = list(features.keys())
        feature_matrix = np.array([features[cid] for cid in client_ids])

        # Simple k-means
        n = len(client_ids)
        k = min(self.num_clusters, n)

        # Random initialization
        centroids = feature_matrix[self.rng.choice(n, k, replace=False)]

        for _ in range(10):  # Max iterations
            # Assign to nearest centroid
            assignments = []
            for feat in feature_matrix:
                distances = [np.linalg.norm(feat - c) for c in centroids]
                assignments.append(np.argmin(distances))

            # Update centroids
            new_centroids = []
            for c in range(k):
                cluster_points = feature_matrix[np.array(assignments) == c]
                if len(cluster_points) > 0:
                    new_centroids.append(cluster_points.mean(axis=0))
                else:
                    new_centroids.append(centroids[c])
            centroids = np.array(new_centroids)

        # Store assignments
        for cid, cluster in zip(client_ids, assignments):
            self._cluster_assignments[cid] = cluster

    def _cluster_by_metadata(self, clients: Dict[str, ClientProfile]) -> None:
        """Cluster by country/region metadata."""
        # Group by country first, then region
        country_clusters = {}
        for cid, profile in clients.items():
            key = (profile.country or "unknown", profile.region or "unknown")
            if key not in country_clusters:
                country_clusters[key] = len(country_clusters) % self.num_clusters
            self._cluster_assignments[cid] = country_clusters[key]

    def select(
        self,
        clients: Dict[str, ClientProfile],
        num_select: int,
        **kwargs,
    ) -> SelectionResult:
        """
        Select diverse clients across clusters.
        """
        available = self.filter_available(clients)

        # Ensure clusters are assigned
        if not self._cluster_assignments:
            self._cluster_by_metadata(available)

        if len(available) <= num_select:
            return SelectionResult(
                selected_clients=list(available.keys()),
                selection_probabilities={cid: 1.0 for cid in available},
                selection_strategy=SelectionStrategy.CLUSTERED,
                round_number=kwargs.get("round_number", 0),
            )

        # Group available clients by cluster
        clusters: Dict[int, List[str]] = {}
        for cid in available:
            cluster = self._cluster_assignments.get(cid, 0)
            if cluster not in clusters:
                clusters[cluster] = []
            clusters[cluster].append(cid)

        # Round-robin selection across clusters
        selected = []
        cluster_ids = list(clusters.keys())

        while len(selected) < num_select and any(clusters.values()):
            for cluster_id in cluster_ids:
                if len(selected) >= num_select:
                    break
                if clusters[cluster_id]:
                    # Select random from cluster
                    cid = self.rng.choice(clusters[cluster_id])
                    selected.append(cid)
                    clusters[cluster_id].remove(cid)

        # Probabilities based on cluster diversity
        cluster_counts = {}
        for cid in available:
            c = self._cluster_assignments.get(cid, 0)
            cluster_counts[c] = cluster_counts.get(c, 0) + 1

        probs = {}
        for cid in available:
            c = self._cluster_assignments.get(cid, 0)
            # Smaller clusters get higher probability
            probs[cid] = 1.0 / cluster_counts[c]

        total = sum(probs.values())
        probs = {cid: p / total for cid, p in probs.items()}

        return SelectionResult(
            selected_clients=selected,
            selection_probabilities=probs,
            selection_strategy=SelectionStrategy.CLUSTERED,
            round_number=kwargs.get("round_number", 0),
            metadata={"num_clusters_selected": len(set(
                self._cluster_assignments.get(cid, 0) for cid in selected
            ))},
        )


# =============================================================================
# Client Selection Manager
# =============================================================================

class ClientSelectionManager:
    """
    Central manager for client selection.

    Coordinates multiple selection strategies and maintains
    client profiles and history.
    """

    def __init__(
        self,
        default_strategy: SelectionStrategy = SelectionStrategy.OORT,
        min_clients: int = 2,
        max_clients: int = 100,
        seed: Optional[int] = None,
    ):
        self.default_strategy = default_strategy
        self.min_clients = min_clients
        self.max_clients = max_clients
        self.seed = seed

        # Client profiles
        self._clients: Dict[str, ClientProfile] = {}

        # Selectors
        self._selectors: Dict[SelectionStrategy, ClientSelector] = {
            SelectionStrategy.RANDOM: RandomSelector(seed),
            SelectionStrategy.ACTIVE_LEARNING: ActiveLearningSelector(seed=seed),
            SelectionStrategy.IMPORTANCE_SAMPLING: ImportanceSamplingSelector(seed=seed),
            SelectionStrategy.RESOURCE_AWARE: ResourceAwareSelector(seed=seed),
            SelectionStrategy.FAIRNESS_AWARE: FairnessAwareSelector(seed=seed),
            SelectionStrategy.OORT: OortSelector(seed=seed),
            SelectionStrategy.CLUSTERED: ClusteredSelector(seed=seed),
        }

        # History
        self._selection_history: List[SelectionResult] = []
        self._current_round = 0

        logger.info(f"Client Selection Manager initialized with {default_strategy.value}")

    def register_client(
        self,
        client_id: str,
        sample_count: int = 0,
        country: Optional[str] = None,
        region: Optional[str] = None,
        compute_capacity: float = 1.0,
        bandwidth: float = 1.0,
    ) -> ClientProfile:
        """Register a new client."""
        profile = ClientProfile(
            client_id=client_id,
            sample_count=sample_count,
            country=country,
            region=region,
            compute_capacity=compute_capacity,
            bandwidth=bandwidth,
        )
        self._clients[client_id] = profile
        logger.debug(f"Registered client: {client_id}")
        return profile

    def update_client_metrics(
        self,
        client_id: str,
        loss: Optional[float] = None,
        gradient_norm: Optional[float] = None,
        train_time: Optional[float] = None,
        upload_time: Optional[float] = None,
        contribution: Optional[float] = None,
    ) -> None:
        """Update client metrics after a round."""
        if client_id not in self._clients:
            return

        profile = self._clients[client_id]

        if loss is not None:
            profile.last_loss = loss

        if gradient_norm is not None:
            profile.last_gradient_norm = gradient_norm

        if train_time is not None:
            # Exponential moving average
            if profile.average_train_time > 0:
                profile.average_train_time = 0.9 * profile.average_train_time + 0.1 * train_time
            else:
                profile.average_train_time = train_time

        if upload_time is not None:
            if profile.average_upload_time > 0:
                profile.average_upload_time = 0.9 * profile.average_upload_time + 0.1 * upload_time
            else:
                profile.average_upload_time = upload_time

        if contribution is not None:
            profile.total_contribution += contribution
            profile.average_contribution = profile.total_contribution / max(1, profile.rounds_participated)

    def select_clients(
        self,
        num_select: int,
        strategy: Optional[SelectionStrategy] = None,
        deadline_seconds: Optional[float] = None,
        **kwargs,
    ) -> SelectionResult:
        """
        Select clients for the current round.

        Args:
            num_select: Number of clients to select
            strategy: Selection strategy (uses default if None)
            deadline_seconds: Optional time deadline
            **kwargs: Additional strategy-specific parameters

        Returns:
            SelectionResult with selected clients
        """
        self._current_round += 1

        # Clamp num_select
        num_select = max(self.min_clients, min(self.max_clients, num_select))

        strategy = strategy or self.default_strategy
        selector = self._selectors.get(strategy)

        if not selector:
            logger.warning(f"Unknown strategy {strategy}, falling back to random")
            selector = self._selectors[SelectionStrategy.RANDOM]
            strategy = SelectionStrategy.RANDOM

        # Perform selection
        result = selector.select(
            self._clients,
            num_select,
            round_number=self._current_round,
            total_rounds=kwargs.get("total_rounds", 100),
            deadline_seconds=deadline_seconds,
            **kwargs,
        )

        # Update client profiles
        for cid in self._clients:
            if cid in result.selected_clients:
                self._clients[cid].rounds_participated += 1
                self._clients[cid].rounds_since_selection = 0
            else:
                self._clients[cid].rounds_since_selection += 1

        self._selection_history.append(result)

        logger.info(
            f"Round {self._current_round}: selected {len(result.selected_clients)} "
            f"clients using {strategy.value}"
        )

        return result

    def get_client_profile(self, client_id: str) -> Optional[ClientProfile]:
        """Get client profile."""
        return self._clients.get(client_id)

    def get_all_clients(self) -> Dict[str, ClientProfile]:
        """Get all client profiles."""
        return self._clients.copy()

    def set_client_status(self, client_id: str, status: ClientStatus) -> None:
        """Set client availability status."""
        if client_id in self._clients:
            self._clients[client_id].status = status

    def get_selection_stats(self) -> Dict[str, Any]:
        """Get selection statistics."""
        if not self._selection_history:
            return {}

        # Participation distribution
        participation = {cid: p.rounds_participated for cid, p in self._clients.items()}

        return {
            "total_rounds": self._current_round,
            "total_clients": len(self._clients),
            "active_clients": sum(
                1 for p in self._clients.values()
                if p.status == ClientStatus.AVAILABLE
            ),
            "average_participation": np.mean(list(participation.values())),
            "min_participation": min(participation.values()) if participation else 0,
            "max_participation": max(participation.values()) if participation else 0,
            "participation_std": np.std(list(participation.values())),
            "strategy_distribution": self._get_strategy_distribution(),
        }

    def _get_strategy_distribution(self) -> Dict[str, int]:
        """Get distribution of strategies used."""
        dist = {}
        for result in self._selection_history:
            key = result.selection_strategy.value
            dist[key] = dist.get(key, 0) + 1
        return dist


# =============================================================================
# Factory Functions
# =============================================================================

def create_selection_manager(
    strategy: SelectionStrategy = SelectionStrategy.OORT,
    min_clients: int = 2,
    max_clients: int = 100,
    seed: Optional[int] = None,
) -> ClientSelectionManager:
    """Create client selection manager."""
    return ClientSelectionManager(
        default_strategy=strategy,
        min_clients=min_clients,
        max_clients=max_clients,
        seed=seed,
    )


def create_selector(
    strategy: SelectionStrategy,
    **kwargs,
) -> ClientSelector:
    """Create individual selector."""
    if strategy == SelectionStrategy.RANDOM:
        return RandomSelector(**kwargs)
    elif strategy == SelectionStrategy.ACTIVE_LEARNING:
        return ActiveLearningSelector(**kwargs)
    elif strategy == SelectionStrategy.IMPORTANCE_SAMPLING:
        return ImportanceSamplingSelector(**kwargs)
    elif strategy == SelectionStrategy.RESOURCE_AWARE:
        return ResourceAwareSelector(**kwargs)
    elif strategy == SelectionStrategy.FAIRNESS_AWARE:
        return FairnessAwareSelector(**kwargs)
    elif strategy == SelectionStrategy.OORT:
        return OortSelector(**kwargs)
    elif strategy == SelectionStrategy.CLUSTERED:
        return ClusteredSelector(**kwargs)
    else:
        return RandomSelector(**kwargs)


# =============================================================================
# Export
# =============================================================================

__all__ = [
    # Enums
    "SelectionStrategy",
    "ClientStatus",
    # Data Classes
    "ClientProfile",
    "SelectionResult",
    # Selectors
    "ClientSelector",
    "RandomSelector",
    "ActiveLearningSelector",
    "ImportanceSamplingSelector",
    "ResourceAwareSelector",
    "FairnessAwareSelector",
    "OortSelector",
    "ClusteredSelector",
    # Manager
    "ClientSelectionManager",
    # Factory
    "create_selection_manager",
    "create_selector",
]

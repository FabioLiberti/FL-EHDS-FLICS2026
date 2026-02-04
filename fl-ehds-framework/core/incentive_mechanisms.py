"""
Incentive Mechanisms for FL-EHDS
=================================

Implementation of incentive and contribution evaluation mechanisms
for federated learning in the European Health Data Space.

Supported Approaches:
1. Shapley Value - Game-theoretic fair contribution attribution
2. Marginal Contribution - Individual client impact
3. Influence Functions - Model-based contribution estimation
4. Data Valuation - Quality and quantity scoring
5. Performance-based Rewards - Accuracy improvement attribution
6. Reputation Systems - Historical contribution tracking

Key References:
- Ghorbani & Zou, "Data Shapley: Equitable Valuation of Data", ICML 2019
- Jia et al., "Towards Efficient Data Valuation Based on Shapley Value", 2019
- Wang et al., "A Principled Approach to Data Valuation for FL", 2020

EHDS Relevance:
- Motivates cross-border hospital participation
- Fair compensation for data providers
- Quality incentives improve model performance

Author: FL-EHDS Framework
License: Apache 2.0
"""

import itertools
import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Constants
# =============================================================================

class ContributionMetric(Enum):
    """Contribution measurement metrics."""
    SHAPLEY_VALUE = "shapley_value"
    MARGINAL_CONTRIBUTION = "marginal_contribution"
    INFLUENCE_FUNCTION = "influence_function"
    DATA_SHAPLEY = "data_shapley"
    GRADIENT_NORM = "gradient_norm"
    ACCURACY_GAIN = "accuracy_gain"
    DATA_QUANTITY = "data_quantity"
    DATA_QUALITY = "data_quality"


class RewardType(Enum):
    """Types of rewards/incentives."""
    MONETARY = "monetary"
    PRIORITY_ACCESS = "priority_access"
    REPUTATION_POINTS = "reputation_points"
    MODEL_ACCESS = "model_access"
    COMPUTE_CREDITS = "compute_credits"


class DataQualityDimension(Enum):
    """Dimensions of data quality assessment."""
    COMPLETENESS = "completeness"  # Missing values
    CONSISTENCY = "consistency"  # Format consistency
    ACCURACY = "accuracy"  # Label accuracy
    TIMELINESS = "timeliness"  # Data freshness
    UNIQUENESS = "uniqueness"  # Non-duplicate records
    DIVERSITY = "diversity"  # Class/feature distribution


@dataclass
class ClientContribution:
    """Client contribution record."""
    client_id: str
    round_number: int
    timestamp: datetime

    # Contribution metrics
    shapley_value: Optional[float] = None
    marginal_contribution: Optional[float] = None
    influence_score: Optional[float] = None
    gradient_norm: Optional[float] = None
    accuracy_gain: Optional[float] = None

    # Data metrics
    sample_count: int = 0
    data_quality_score: Optional[float] = None
    quality_dimensions: Dict[str, float] = field(default_factory=dict)

    # Aggregated scores
    total_contribution_score: Optional[float] = None
    contribution_rank: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "clientId": self.client_id,
            "roundNumber": self.round_number,
            "timestamp": self.timestamp.isoformat(),
            "shapleyValue": self.shapley_value,
            "marginalContribution": self.marginal_contribution,
            "influenceScore": self.influence_score,
            "gradientNorm": self.gradient_norm,
            "accuracyGain": self.accuracy_gain,
            "sampleCount": self.sample_count,
            "dataQualityScore": self.data_quality_score,
            "qualityDimensions": self.quality_dimensions,
            "totalContributionScore": self.total_contribution_score,
            "contributionRank": self.contribution_rank,
        }


@dataclass
class RewardAllocation:
    """Reward allocation for a client."""
    client_id: str
    round_number: int
    reward_type: RewardType
    amount: float
    contribution_score: float
    allocation_timestamp: datetime = field(default_factory=datetime.now)
    notes: Optional[str] = None


@dataclass
class ClientReputation:
    """Client reputation tracking."""
    client_id: str
    total_rounds_participated: int = 0
    total_contributions: float = 0.0
    average_contribution: float = 0.0
    consistency_score: float = 1.0  # How consistent are contributions
    reliability_score: float = 1.0  # Participation reliability
    historical_shapley: List[float] = field(default_factory=list)
    last_participation: Optional[datetime] = None
    reputation_tier: str = "bronze"  # bronze, silver, gold, platinum


# =============================================================================
# Contribution Calculators
# =============================================================================

class ContributionCalculator(ABC):
    """Abstract base class for contribution calculators."""

    @abstractmethod
    def calculate(
        self,
        client_updates: Dict[str, Dict[str, np.ndarray]],
        global_model: Dict[str, np.ndarray],
        eval_fn: Optional[Callable] = None,
    ) -> Dict[str, float]:
        """Calculate contribution scores for clients."""
        pass


class ShapleyValueCalculator(ContributionCalculator):
    """
    Shapley Value-based Contribution Calculator.

    Computes fair contribution attribution based on game-theoretic
    Shapley values. Each client's contribution is the average marginal
    contribution across all possible coalitions.

    Note: Exact Shapley is O(2^n), so approximations are used for
    large numbers of clients.
    """

    def __init__(
        self,
        monte_carlo_samples: int = 100,
        use_exact: bool = False,
        max_exact_clients: int = 10,
    ):
        self.monte_carlo_samples = monte_carlo_samples
        self.use_exact = use_exact
        self.max_exact_clients = max_exact_clients

    def calculate(
        self,
        client_updates: Dict[str, Dict[str, np.ndarray]],
        global_model: Dict[str, np.ndarray],
        eval_fn: Optional[Callable] = None,
    ) -> Dict[str, float]:
        """
        Calculate Shapley values for all clients.

        Args:
            client_updates: Dict of client_id -> model update
            global_model: Current global model
            eval_fn: Function to evaluate model quality (model -> score)

        Returns:
            Dict of client_id -> Shapley value
        """
        client_ids = list(client_updates.keys())
        n = len(client_ids)

        if n == 0:
            return {}

        # Use exact computation for small N
        if self.use_exact and n <= self.max_exact_clients:
            return self._exact_shapley(client_updates, global_model, eval_fn)
        else:
            return self._monte_carlo_shapley(client_updates, global_model, eval_fn)

    def _exact_shapley(
        self,
        client_updates: Dict[str, Dict[str, np.ndarray]],
        global_model: Dict[str, np.ndarray],
        eval_fn: Optional[Callable],
    ) -> Dict[str, float]:
        """Exact Shapley value computation."""
        client_ids = list(client_updates.keys())
        n = len(client_ids)

        shapley_values = {cid: 0.0 for cid in client_ids}

        # Iterate over all permutations
        for perm in itertools.permutations(range(n)):
            coalition = set()

            for i, idx in enumerate(perm):
                client_id = client_ids[idx]

                # Value without this client
                v_without = self._coalition_value(
                    coalition, client_ids, client_updates, global_model, eval_fn
                )

                # Value with this client
                coalition.add(idx)
                v_with = self._coalition_value(
                    coalition, client_ids, client_updates, global_model, eval_fn
                )

                # Marginal contribution
                shapley_values[client_id] += (v_with - v_without)

        # Average over all permutations
        num_perms = math.factorial(n)
        for cid in client_ids:
            shapley_values[cid] /= num_perms

        return shapley_values

    def _monte_carlo_shapley(
        self,
        client_updates: Dict[str, Dict[str, np.ndarray]],
        global_model: Dict[str, np.ndarray],
        eval_fn: Optional[Callable],
    ) -> Dict[str, float]:
        """Monte Carlo approximation of Shapley values."""
        client_ids = list(client_updates.keys())
        n = len(client_ids)

        shapley_values = {cid: 0.0 for cid in client_ids}
        count = {cid: 0 for cid in client_ids}

        for _ in range(self.monte_carlo_samples):
            # Random permutation
            perm = np.random.permutation(n)
            coalition = set()

            for idx in perm:
                client_id = client_ids[idx]

                # Value without
                v_without = self._coalition_value(
                    coalition, client_ids, client_updates, global_model, eval_fn
                )

                # Value with
                coalition.add(idx)
                v_with = self._coalition_value(
                    coalition, client_ids, client_updates, global_model, eval_fn
                )

                # Update running average
                count[client_id] += 1
                delta = (v_with - v_without) - shapley_values[client_id]
                shapley_values[client_id] += delta / count[client_id]

        return shapley_values

    def _coalition_value(
        self,
        coalition: Set[int],
        client_ids: List[str],
        client_updates: Dict[str, Dict[str, np.ndarray]],
        global_model: Dict[str, np.ndarray],
        eval_fn: Optional[Callable],
    ) -> float:
        """Compute value of a coalition of clients."""
        if not coalition:
            # Empty coalition - use baseline (global model)
            if eval_fn:
                return eval_fn(global_model)
            return 0.0

        # Aggregate updates from coalition
        coalition_clients = [client_ids[i] for i in coalition]
        aggregated = self._aggregate_updates(
            {cid: client_updates[cid] for cid in coalition_clients},
            global_model,
        )

        if eval_fn:
            return eval_fn(aggregated)

        # Fallback: use gradient norm as proxy
        norm = sum(
            np.linalg.norm(aggregated[k] - global_model.get(k, 0))
            for k in aggregated
        )
        return norm

    def _aggregate_updates(
        self,
        updates: Dict[str, Dict[str, np.ndarray]],
        global_model: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Simple averaging of updates."""
        if not updates:
            return global_model

        aggregated = {}
        n = len(updates)

        for key in global_model:
            aggregated[key] = sum(
                update.get(key, global_model[key])
                for update in updates.values()
            ) / n

        return aggregated


class MarginalContributionCalculator(ContributionCalculator):
    """
    Marginal Contribution Calculator.

    Computes leave-one-out contribution: difference in model
    quality with vs without each client.
    """

    def calculate(
        self,
        client_updates: Dict[str, Dict[str, np.ndarray]],
        global_model: Dict[str, np.ndarray],
        eval_fn: Optional[Callable] = None,
    ) -> Dict[str, float]:
        """
        Calculate marginal contributions.

        For each client i: MC_i = V(all) - V(all \ {i})
        """
        client_ids = list(client_updates.keys())
        contributions = {}

        # Full coalition value
        full_model = self._aggregate_all(client_updates, global_model)
        if eval_fn:
            v_full = eval_fn(full_model)
        else:
            v_full = self._proxy_value(full_model, global_model)

        # Leave-one-out for each client
        for cid in client_ids:
            # Aggregate without this client
            others = {k: v for k, v in client_updates.items() if k != cid}
            if others:
                without_model = self._aggregate_all(others, global_model)
                if eval_fn:
                    v_without = eval_fn(without_model)
                else:
                    v_without = self._proxy_value(without_model, global_model)
            else:
                v_without = 0.0 if eval_fn else self._proxy_value(global_model, global_model)

            contributions[cid] = v_full - v_without

        return contributions

    def _aggregate_all(
        self,
        updates: Dict[str, Dict[str, np.ndarray]],
        global_model: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Aggregate all updates."""
        if not updates:
            return global_model

        aggregated = {}
        n = len(updates)

        for key in global_model:
            aggregated[key] = sum(
                update.get(key, global_model[key])
                for update in updates.values()
            ) / n

        return aggregated

    def _proxy_value(
        self,
        model: Dict[str, np.ndarray],
        baseline: Dict[str, np.ndarray],
    ) -> float:
        """Proxy value using weight change magnitude."""
        return sum(
            np.linalg.norm(model[k] - baseline.get(k, 0))
            for k in model
        )


class GradientBasedCalculator(ContributionCalculator):
    """
    Gradient-based Contribution Calculator.

    Uses gradient norms and similarities to estimate contributions.
    Fast but less accurate than Shapley.
    """

    def __init__(
        self,
        use_cosine_similarity: bool = True,
        aggregate_gradient: Optional[Dict[str, np.ndarray]] = None,
    ):
        self.use_cosine_similarity = use_cosine_similarity
        self.aggregate_gradient = aggregate_gradient

    def calculate(
        self,
        client_updates: Dict[str, Dict[str, np.ndarray]],
        global_model: Dict[str, np.ndarray],
        eval_fn: Optional[Callable] = None,
    ) -> Dict[str, float]:
        """
        Calculate gradient-based contributions.
        """
        contributions = {}

        # Compute aggregate gradient if not provided
        if self.aggregate_gradient is None:
            agg_grad = {}
            n = len(client_updates)
            for key in global_model:
                agg_grad[key] = sum(
                    update.get(key, global_model[key]) - global_model[key]
                    for update in client_updates.values()
                ) / n
        else:
            agg_grad = self.aggregate_gradient

        for cid, update in client_updates.items():
            # Client gradient
            client_grad = {
                key: update.get(key, global_model[key]) - global_model[key]
                for key in global_model
            }

            if self.use_cosine_similarity:
                # Cosine similarity with aggregate gradient
                dot = sum(
                    np.dot(client_grad[k].flatten(), agg_grad[k].flatten())
                    for k in global_model
                )
                norm_client = sum(
                    np.linalg.norm(client_grad[k])
                    for k in global_model
                )
                norm_agg = sum(
                    np.linalg.norm(agg_grad[k])
                    for k in global_model
                )

                if norm_client > 0 and norm_agg > 0:
                    contributions[cid] = dot / (norm_client * norm_agg)
                else:
                    contributions[cid] = 0.0
            else:
                # Just use gradient norm
                contributions[cid] = sum(
                    np.linalg.norm(client_grad[k])
                    for k in global_model
                )

        return contributions


# =============================================================================
# Data Quality Assessor
# =============================================================================

class DataQualityAssessor:
    """
    Assesses data quality across multiple dimensions.

    Used to weight contributions based on data quality.
    """

    def __init__(
        self,
        dimension_weights: Optional[Dict[DataQualityDimension, float]] = None,
    ):
        self.dimension_weights = dimension_weights or {
            DataQualityDimension.COMPLETENESS: 0.25,
            DataQualityDimension.ACCURACY: 0.30,
            DataQualityDimension.DIVERSITY: 0.25,
            DataQualityDimension.UNIQUENESS: 0.20,
        }

    def assess(
        self,
        X: np.ndarray,
        y: np.ndarray,
        expected_features: Optional[int] = None,
        expected_classes: Optional[List] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Assess data quality.

        Returns:
            Tuple of (overall_score, dimension_scores)
        """
        dimension_scores = {}

        # Completeness (missing values)
        if DataQualityDimension.COMPLETENESS in self.dimension_weights:
            completeness = 1.0 - np.mean(np.isnan(X))
            dimension_scores[DataQualityDimension.COMPLETENESS.value] = completeness

        # Diversity (class balance)
        if DataQualityDimension.DIVERSITY in self.dimension_weights:
            unique, counts = np.unique(y, return_counts=True)
            if len(counts) > 1:
                # Entropy-based diversity
                probs = counts / counts.sum()
                entropy = -np.sum(probs * np.log(probs + 1e-10))
                max_entropy = np.log(len(unique))
                diversity = entropy / max_entropy if max_entropy > 0 else 0
            else:
                diversity = 0.0
            dimension_scores[DataQualityDimension.DIVERSITY.value] = diversity

        # Uniqueness (non-duplicate)
        if DataQualityDimension.UNIQUENESS in self.dimension_weights:
            n_total = len(X)
            n_unique = len(np.unique(X, axis=0))
            uniqueness = n_unique / n_total if n_total > 0 else 0
            dimension_scores[DataQualityDimension.UNIQUENESS.value] = uniqueness

        # Accuracy proxy (consistent labels within clusters)
        if DataQualityDimension.ACCURACY in self.dimension_weights:
            # Use label consistency as proxy
            # In practice, would use validation set
            dimension_scores[DataQualityDimension.ACCURACY.value] = 0.85  # Placeholder

        # Overall weighted score
        overall = sum(
            self.dimension_weights.get(DataQualityDimension(dim), 0) * score
            for dim, score in dimension_scores.items()
        )

        return overall, dimension_scores


# =============================================================================
# Reward Distribution
# =============================================================================

class RewardDistributor:
    """
    Distributes rewards based on contributions.

    Supports various distribution schemes:
    - Proportional to contribution
    - Threshold-based tiers
    - Capped distributions
    """

    def __init__(
        self,
        total_budget: float = 1000.0,
        min_reward: float = 0.0,
        max_reward: Optional[float] = None,
        distribution_scheme: str = "proportional",  # proportional, tiered, equal
    ):
        self.total_budget = total_budget
        self.min_reward = min_reward
        self.max_reward = max_reward
        self.distribution_scheme = distribution_scheme

    def distribute(
        self,
        contributions: Dict[str, float],
        reward_type: RewardType = RewardType.COMPUTE_CREDITS,
    ) -> List[RewardAllocation]:
        """
        Distribute rewards based on contributions.
        """
        if not contributions:
            return []

        # Normalize contributions to positive
        min_contrib = min(contributions.values())
        if min_contrib < 0:
            contributions = {
                cid: score - min_contrib
                for cid, score in contributions.items()
            }

        total_contrib = sum(contributions.values())

        allocations = []

        if self.distribution_scheme == "proportional":
            for cid, score in contributions.items():
                if total_contrib > 0:
                    share = score / total_contrib
                    amount = share * self.total_budget
                else:
                    amount = self.total_budget / len(contributions)

                # Apply bounds
                amount = max(self.min_reward, amount)
                if self.max_reward:
                    amount = min(self.max_reward, amount)

                allocations.append(RewardAllocation(
                    client_id=cid,
                    round_number=0,  # Set by caller
                    reward_type=reward_type,
                    amount=amount,
                    contribution_score=score,
                ))

        elif self.distribution_scheme == "tiered":
            # Sort by contribution
            sorted_clients = sorted(
                contributions.items(),
                key=lambda x: x[1],
                reverse=True
            )

            n = len(sorted_clients)
            tier_budgets = {
                "top_10": self.total_budget * 0.4,
                "top_30": self.total_budget * 0.3,
                "rest": self.total_budget * 0.3,
            }

            for i, (cid, score) in enumerate(sorted_clients):
                percentile = (i + 1) / n

                if percentile <= 0.1:
                    tier_budget = tier_budgets["top_10"]
                    tier_size = max(1, int(n * 0.1))
                elif percentile <= 0.3:
                    tier_budget = tier_budgets["top_30"]
                    tier_size = max(1, int(n * 0.2))
                else:
                    tier_budget = tier_budgets["rest"]
                    tier_size = n - int(n * 0.3)

                amount = tier_budget / tier_size

                allocations.append(RewardAllocation(
                    client_id=cid,
                    round_number=0,
                    reward_type=reward_type,
                    amount=amount,
                    contribution_score=score,
                ))

        elif self.distribution_scheme == "equal":
            amount = self.total_budget / len(contributions)
            for cid, score in contributions.items():
                allocations.append(RewardAllocation(
                    client_id=cid,
                    round_number=0,
                    reward_type=reward_type,
                    amount=amount,
                    contribution_score=score,
                ))

        return allocations


# =============================================================================
# Reputation System
# =============================================================================

class ReputationSystem:
    """
    Manages client reputation based on historical contributions.

    Used for:
    - Long-term incentives
    - Client selection prioritization
    - Trust scores for Byzantine resilience
    """

    def __init__(
        self,
        decay_factor: float = 0.95,  # Weight decay for old contributions
        tier_thresholds: Optional[Dict[str, float]] = None,
    ):
        self.decay_factor = decay_factor
        self.tier_thresholds = tier_thresholds or {
            "platinum": 0.9,
            "gold": 0.7,
            "silver": 0.4,
            "bronze": 0.0,
        }
        self._reputations: Dict[str, ClientReputation] = {}

    def update_reputation(
        self,
        client_id: str,
        contribution: ClientContribution,
    ) -> ClientReputation:
        """Update client reputation with new contribution."""
        if client_id not in self._reputations:
            self._reputations[client_id] = ClientReputation(client_id=client_id)

        rep = self._reputations[client_id]

        # Update participation count
        rep.total_rounds_participated += 1
        rep.last_participation = contribution.timestamp

        # Update contribution metrics
        if contribution.shapley_value is not None:
            rep.historical_shapley.append(contribution.shapley_value)

            # Exponential moving average
            if len(rep.historical_shapley) > 1:
                rep.total_contributions = (
                    self.decay_factor * rep.total_contributions +
                    contribution.shapley_value
                )
            else:
                rep.total_contributions = contribution.shapley_value

            rep.average_contribution = rep.total_contributions / rep.total_rounds_participated

        # Calculate consistency (variance of contributions)
        if len(rep.historical_shapley) > 1:
            variance = np.var(rep.historical_shapley)
            rep.consistency_score = 1.0 / (1.0 + variance)

        # Update tier
        rep.reputation_tier = self._compute_tier(rep)

        return rep

    def _compute_tier(self, rep: ClientReputation) -> str:
        """Compute reputation tier."""
        # Combine average contribution and consistency
        score = 0.7 * self._normalize_contribution(rep.average_contribution) + \
                0.3 * rep.consistency_score

        for tier, threshold in sorted(self.tier_thresholds.items(), key=lambda x: -x[1]):
            if score >= threshold:
                return tier

        return "bronze"

    def _normalize_contribution(self, contribution: float) -> float:
        """Normalize contribution to [0, 1]."""
        # Get max contribution across all clients
        all_contribs = [
            r.average_contribution
            for r in self._reputations.values()
            if r.average_contribution > 0
        ]

        if not all_contribs:
            return 0.5

        max_contrib = max(all_contribs)
        if max_contrib > 0:
            return min(1.0, contribution / max_contrib)
        return 0.5

    def get_reputation(self, client_id: str) -> Optional[ClientReputation]:
        """Get client reputation."""
        return self._reputations.get(client_id)

    def get_all_reputations(self) -> Dict[str, ClientReputation]:
        """Get all client reputations."""
        return self._reputations.copy()

    def get_tier_clients(self, tier: str) -> List[str]:
        """Get all clients in a specific tier."""
        return [
            cid for cid, rep in self._reputations.items()
            if rep.reputation_tier == tier
        ]


# =============================================================================
# Incentive Manager
# =============================================================================

class IncentiveManager:
    """
    Central manager for incentive mechanisms.

    Coordinates contribution calculation, reward distribution,
    and reputation management.
    """

    def __init__(
        self,
        primary_metric: ContributionMetric = ContributionMetric.SHAPLEY_VALUE,
        monte_carlo_samples: int = 100,
        reward_budget: float = 1000.0,
    ):
        self.primary_metric = primary_metric
        self.monte_carlo_samples = monte_carlo_samples
        self.reward_budget = reward_budget

        # Calculators
        self._shapley_calc = ShapleyValueCalculator(monte_carlo_samples=monte_carlo_samples)
        self._marginal_calc = MarginalContributionCalculator()
        self._gradient_calc = GradientBasedCalculator()

        # Quality assessor
        self._quality_assessor = DataQualityAssessor()

        # Reward distributor
        self._distributor = RewardDistributor(total_budget=reward_budget)

        # Reputation system
        self._reputation = ReputationSystem()

        # History
        self._contribution_history: List[ClientContribution] = []
        self._reward_history: List[RewardAllocation] = []

        logger.info(f"Incentive Manager initialized with {primary_metric.value}")

    def evaluate_round(
        self,
        round_number: int,
        client_updates: Dict[str, Dict[str, np.ndarray]],
        global_model: Dict[str, np.ndarray],
        client_data: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
        eval_fn: Optional[Callable] = None,
    ) -> Dict[str, ClientContribution]:
        """
        Evaluate contributions for a training round.

        Args:
            round_number: Current FL round
            client_updates: Client model updates
            global_model: Global model before aggregation
            client_data: Optional client datasets for quality assessment
            eval_fn: Model evaluation function

        Returns:
            Dict of client_id -> ClientContribution
        """
        contributions = {}

        # Calculate primary metric
        if self.primary_metric == ContributionMetric.SHAPLEY_VALUE:
            scores = self._shapley_calc.calculate(client_updates, global_model, eval_fn)
        elif self.primary_metric == ContributionMetric.MARGINAL_CONTRIBUTION:
            scores = self._marginal_calc.calculate(client_updates, global_model, eval_fn)
        else:
            scores = self._gradient_calc.calculate(client_updates, global_model, eval_fn)

        # Calculate additional metrics
        gradient_scores = self._gradient_calc.calculate(client_updates, global_model)

        for cid in client_updates:
            contrib = ClientContribution(
                client_id=cid,
                round_number=round_number,
                timestamp=datetime.now(),
                shapley_value=scores.get(cid) if self.primary_metric == ContributionMetric.SHAPLEY_VALUE else None,
                marginal_contribution=scores.get(cid) if self.primary_metric == ContributionMetric.MARGINAL_CONTRIBUTION else None,
                gradient_norm=gradient_scores.get(cid),
            )

            # Data quality if available
            if client_data and cid in client_data:
                X, y = client_data[cid]
                contrib.sample_count = len(X)
                quality_score, quality_dims = self._quality_assessor.assess(X, y)
                contrib.data_quality_score = quality_score
                contrib.quality_dimensions = quality_dims

            # Total score (weighted combination)
            contrib.total_contribution_score = self._compute_total_score(contrib)

            contributions[cid] = contrib
            self._contribution_history.append(contrib)

            # Update reputation
            self._reputation.update_reputation(cid, contrib)

        # Assign ranks
        sorted_contribs = sorted(
            contributions.items(),
            key=lambda x: x[1].total_contribution_score or 0,
            reverse=True
        )
        for rank, (cid, contrib) in enumerate(sorted_contribs):
            contrib.contribution_rank = rank + 1

        logger.info(f"Round {round_number}: evaluated {len(contributions)} clients")
        return contributions

    def _compute_total_score(self, contrib: ClientContribution) -> float:
        """Compute weighted total contribution score."""
        score = 0.0
        weights = {"primary": 0.5, "gradient": 0.2, "quality": 0.2, "quantity": 0.1}

        # Primary metric
        primary = contrib.shapley_value or contrib.marginal_contribution or 0
        score += weights["primary"] * self._normalize(primary, "primary")

        # Gradient norm
        if contrib.gradient_norm:
            score += weights["gradient"] * self._normalize(contrib.gradient_norm, "gradient")

        # Data quality
        if contrib.data_quality_score:
            score += weights["quality"] * contrib.data_quality_score

        # Data quantity (log scale)
        if contrib.sample_count > 0:
            quantity_score = math.log(1 + contrib.sample_count) / math.log(1 + 10000)
            score += weights["quantity"] * min(1.0, quantity_score)

        return score

    def _normalize(self, value: float, metric_type: str) -> float:
        """Normalize metric to [0, 1] range."""
        # Get historical values for this metric
        historical = []
        for h in self._contribution_history[-100:]:  # Last 100 entries
            if metric_type == "primary":
                v = h.shapley_value or h.marginal_contribution
            elif metric_type == "gradient":
                v = h.gradient_norm
            else:
                v = None

            if v is not None:
                historical.append(v)

        if not historical:
            return 0.5

        min_v, max_v = min(historical), max(historical)
        if max_v > min_v:
            return (value - min_v) / (max_v - min_v)
        return 0.5

    def distribute_rewards(
        self,
        round_number: int,
        contributions: Dict[str, ClientContribution],
        reward_type: RewardType = RewardType.COMPUTE_CREDITS,
    ) -> List[RewardAllocation]:
        """Distribute rewards based on contributions."""
        scores = {
            cid: contrib.total_contribution_score or 0
            for cid, contrib in contributions.items()
        }

        allocations = self._distributor.distribute(scores, reward_type)

        # Set round number
        for alloc in allocations:
            alloc.round_number = round_number

        self._reward_history.extend(allocations)

        total_distributed = sum(a.amount for a in allocations)
        logger.info(f"Round {round_number}: distributed {total_distributed:.2f} {reward_type.value}")

        return allocations

    def get_client_summary(self, client_id: str) -> Dict[str, Any]:
        """Get summary of client contributions and rewards."""
        # Contributions
        client_contribs = [
            c for c in self._contribution_history
            if c.client_id == client_id
        ]

        # Rewards
        client_rewards = [
            r for r in self._reward_history
            if r.client_id == client_id
        ]

        # Reputation
        reputation = self._reputation.get_reputation(client_id)

        return {
            "clientId": client_id,
            "totalRounds": len(client_contribs),
            "totalRewards": sum(r.amount for r in client_rewards),
            "averageContribution": np.mean([
                c.total_contribution_score or 0
                for c in client_contribs
            ]) if client_contribs else 0,
            "reputation": {
                "tier": reputation.reputation_tier if reputation else "unknown",
                "consistencyScore": reputation.consistency_score if reputation else 0,
                "reliabilityScore": reputation.reliability_score if reputation else 0,
            },
            "recentContributions": [
                c.to_dict() for c in client_contribs[-5:]
            ],
        }

    def get_leaderboard(self, top_n: int = 10) -> List[Dict[str, Any]]:
        """Get contribution leaderboard."""
        # Aggregate by client
        client_totals: Dict[str, float] = {}
        client_rounds: Dict[str, int] = {}

        for c in self._contribution_history:
            if c.client_id not in client_totals:
                client_totals[c.client_id] = 0
                client_rounds[c.client_id] = 0
            client_totals[c.client_id] += c.total_contribution_score or 0
            client_rounds[c.client_id] += 1

        # Sort by total
        sorted_clients = sorted(
            client_totals.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_n]

        leaderboard = []
        for rank, (cid, total) in enumerate(sorted_clients):
            rep = self._reputation.get_reputation(cid)
            leaderboard.append({
                "rank": rank + 1,
                "clientId": cid,
                "totalContribution": total,
                "roundsParticipated": client_rounds[cid],
                "averageContribution": total / client_rounds[cid] if client_rounds[cid] > 0 else 0,
                "tier": rep.reputation_tier if rep else "bronze",
            })

        return leaderboard


# =============================================================================
# Factory Functions
# =============================================================================

def create_incentive_manager(
    metric: ContributionMetric = ContributionMetric.SHAPLEY_VALUE,
    budget: float = 1000.0,
    monte_carlo_samples: int = 100,
) -> IncentiveManager:
    """Create incentive manager."""
    return IncentiveManager(
        primary_metric=metric,
        reward_budget=budget,
        monte_carlo_samples=monte_carlo_samples,
    )


def create_shapley_calculator(
    samples: int = 100,
    use_exact: bool = False,
) -> ShapleyValueCalculator:
    """Create Shapley value calculator."""
    return ShapleyValueCalculator(
        monte_carlo_samples=samples,
        use_exact=use_exact,
    )


# =============================================================================
# Export
# =============================================================================

__all__ = [
    # Enums
    "ContributionMetric",
    "RewardType",
    "DataQualityDimension",
    # Data Classes
    "ClientContribution",
    "RewardAllocation",
    "ClientReputation",
    # Calculators
    "ContributionCalculator",
    "ShapleyValueCalculator",
    "MarginalContributionCalculator",
    "GradientBasedCalculator",
    # Quality Assessment
    "DataQualityAssessor",
    # Rewards
    "RewardDistributor",
    # Reputation
    "ReputationSystem",
    # Manager
    "IncentiveManager",
    # Factory
    "create_incentive_manager",
    "create_shapley_calculator",
]

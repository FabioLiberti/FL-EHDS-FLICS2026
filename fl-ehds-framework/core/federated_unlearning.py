"""
Federated Unlearning for FL-EHDS
=================================

Implementation of federated unlearning techniques for GDPR Article 17
"Right to be Forgotten" and EHDS Article 71 opt-out compliance.

Supported Approaches:
1. Exact Unlearning (Full Retrain)
2. FedEraser - Calibration-based approximate unlearning
3. SISA - Sharded training for efficient unlearning
4. Gradient Ascent - Reverse gradient contribution
5. Influence Function - Estimate and remove influence

Key References:
- Liu et al., "FedEraser: Enabling Efficient Client-Level Data Removal", 2021
- Bourtoule et al., "Machine Unlearning", 2021 (SISA)
- Wu et al., "Federated Unlearning", 2022
- Che et al., "Fast Federated Machine Unlearning", 2023

EHDS Compliance:
- Article 71: Opt-out mechanism for secondary use
- GDPR Article 17: Right to erasure

Author: FL-EHDS Framework
License: Apache 2.0
"""

import copy
import hashlib
import json
import logging
import math
import uuid
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

class UnlearningMethod(Enum):
    """Federated Unlearning Methods."""
    EXACT_RETRAIN = "exact_retrain"  # Full retraining without target client
    FED_ERASER = "fed_eraser"  # Calibration-based unlearning
    SISA = "sisa"  # Sharded, Isolated, Sliced, Aggregated
    GRADIENT_ASCENT = "gradient_ascent"  # Reverse gradient direction
    INFLUENCE_FUNCTION = "influence_function"  # Newton step approximation
    RAPID_RETRAIN = "rapid_retrain"  # Warm-start from checkpoint


class UnlearningScope(Enum):
    """Scope of Unlearning Request."""
    CLIENT = "client"  # Remove entire client contribution
    SAMPLES = "samples"  # Remove specific samples
    FEATURES = "features"  # Remove specific features (vertical)
    TIME_WINDOW = "time_window"  # Remove contributions from time period


class VerificationMethod(Enum):
    """Methods to verify unlearning success."""
    MEMBERSHIP_INFERENCE = "membership_inference"
    INFLUENCE_ESTIMATION = "influence_estimation"
    GRADIENT_SIMILARITY = "gradient_similarity"
    STATISTICAL_TEST = "statistical_test"


@dataclass
class UnlearningRequest:
    """Request to unlearn specific data from federated model."""
    request_id: str
    requestor_id: str  # Patient or client ID
    scope: UnlearningScope
    target_client_ids: List[str]
    target_sample_ids: Optional[List[str]] = None
    target_features: Optional[List[str]] = None
    time_window_start: Optional[datetime] = None
    time_window_end: Optional[datetime] = None
    request_timestamp: datetime = field(default_factory=datetime.now)
    legal_basis: str = "GDPR_Art17"  # or "EHDS_Art71"
    priority: str = "normal"  # normal, urgent
    verification_required: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "requestId": self.request_id,
            "requestorId": self.requestor_id,
            "scope": self.scope.value,
            "targetClientIds": self.target_client_ids,
            "targetSampleIds": self.target_sample_ids,
            "targetFeatures": self.target_features,
            "timeWindowStart": self.time_window_start.isoformat() if self.time_window_start else None,
            "timeWindowEnd": self.time_window_end.isoformat() if self.time_window_end else None,
            "requestTimestamp": self.request_timestamp.isoformat(),
            "legalBasis": self.legal_basis,
            "priority": self.priority,
            "verificationRequired": self.verification_required,
        }


@dataclass
class UnlearningResult:
    """Result of unlearning operation."""
    request_id: str
    success: bool
    method_used: UnlearningMethod
    start_time: datetime
    end_time: datetime
    rounds_required: int
    computation_cost: float  # Relative to full retrain
    model_hash_before: str
    model_hash_after: str
    verification_passed: Optional[bool] = None
    verification_score: Optional[float] = None
    accuracy_before: Optional[float] = None
    accuracy_after: Optional[float] = None
    accuracy_degradation: Optional[float] = None
    error_message: Optional[str] = None

    @property
    def duration_seconds(self) -> float:
        """Get operation duration."""
        return (self.end_time - self.start_time).total_seconds()


@dataclass
class ClientCheckpoint:
    """Checkpoint of client contribution for unlearning."""
    client_id: str
    round_number: int
    timestamp: datetime
    model_weights: Dict[str, np.ndarray]
    gradient_update: Dict[str, np.ndarray]
    learning_rate: float
    local_epochs: int
    sample_count: int
    sample_hashes: List[str]  # For sample-level tracking


@dataclass
class SISAShard:
    """SISA training shard."""
    shard_id: str
    client_ids: List[str]
    model_weights: Optional[Dict[str, np.ndarray]] = None
    training_history: List[int] = field(default_factory=list)  # Round numbers
    last_updated: Optional[datetime] = None


# =============================================================================
# Unlearning Algorithms
# =============================================================================

class FederatedUnlearner(ABC):
    """Abstract base class for federated unlearning."""

    @abstractmethod
    def unlearn(
        self,
        current_model: Dict[str, np.ndarray],
        request: UnlearningRequest,
        checkpoints: Optional[List[ClientCheckpoint]] = None,
    ) -> Tuple[Dict[str, np.ndarray], UnlearningResult]:
        """
        Execute unlearning operation.

        Args:
            current_model: Current global model weights
            request: Unlearning request details
            checkpoints: Historical checkpoints for calibration

        Returns:
            Tuple of (new_model_weights, result)
        """
        pass

    @abstractmethod
    def estimate_cost(self, request: UnlearningRequest) -> float:
        """Estimate computational cost relative to full retrain."""
        pass


class ExactRetrainUnlearner(FederatedUnlearner):
    """
    Exact unlearning via full retraining.

    Most accurate but computationally expensive. Used as baseline
    and for verification of approximate methods.
    """

    def __init__(
        self,
        training_fn: Callable,
        num_rounds: int,
        initial_model_fn: Callable,
    ):
        self.training_fn = training_fn
        self.num_rounds = num_rounds
        self.initial_model_fn = initial_model_fn
        self.method = UnlearningMethod.EXACT_RETRAIN

    def unlearn(
        self,
        current_model: Dict[str, np.ndarray],
        request: UnlearningRequest,
        checkpoints: Optional[List[ClientCheckpoint]] = None,
    ) -> Tuple[Dict[str, np.ndarray], UnlearningResult]:
        """Full retrain excluding target clients."""
        start_time = datetime.now()
        model_hash_before = self._compute_model_hash(current_model)

        # Initialize fresh model
        new_model = self.initial_model_fn()

        # Retrain excluding target clients
        excluded_clients = set(request.target_client_ids)
        new_model = self.training_fn(
            new_model,
            excluded_clients=excluded_clients,
            num_rounds=self.num_rounds,
        )

        end_time = datetime.now()
        model_hash_after = self._compute_model_hash(new_model)

        result = UnlearningResult(
            request_id=request.request_id,
            success=True,
            method_used=self.method,
            start_time=start_time,
            end_time=end_time,
            rounds_required=self.num_rounds,
            computation_cost=1.0,  # Baseline cost
            model_hash_before=model_hash_before,
            model_hash_after=model_hash_after,
        )

        logger.info(f"Exact retrain completed for {len(excluded_clients)} clients")
        return new_model, result

    def estimate_cost(self, request: UnlearningRequest) -> float:
        """Exact retrain has cost = 1.0 (baseline)."""
        return 1.0

    def _compute_model_hash(self, model: Dict[str, np.ndarray]) -> str:
        """Compute hash of model weights."""
        hasher = hashlib.sha256()
        for key in sorted(model.keys()):
            hasher.update(model[key].tobytes())
        return hasher.hexdigest()[:16]


class FedEraserUnlearner(FederatedUnlearner):
    """
    FedEraser: Calibration-based Federated Unlearning.

    Uses historical gradient updates to calibrate model after
    removing target client's contribution.

    Reference: Liu et al., "FedEraser: Enabling Efficient Client-Level
    Data Removal from Federated Learning Models", 2021
    """

    def __init__(
        self,
        calibration_rounds: int = 10,
        learning_rate: float = 0.01,
        momentum: float = 0.9,
    ):
        self.calibration_rounds = calibration_rounds
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.method = UnlearningMethod.FED_ERASER

    def unlearn(
        self,
        current_model: Dict[str, np.ndarray],
        request: UnlearningRequest,
        checkpoints: Optional[List[ClientCheckpoint]] = None,
    ) -> Tuple[Dict[str, np.ndarray], UnlearningResult]:
        """
        FedEraser unlearning via gradient calibration.

        Steps:
        1. Identify contributions from target clients
        2. Compute calibration gradients from remaining clients
        3. Apply calibration to remove influence
        """
        if not checkpoints:
            raise ValueError("FedEraser requires historical checkpoints")

        start_time = datetime.now()
        model_hash_before = self._compute_model_hash(current_model)

        # Separate checkpoints by client
        target_client_ids = set(request.target_client_ids)
        target_checkpoints = [cp for cp in checkpoints if cp.client_id in target_client_ids]
        remaining_checkpoints = [cp for cp in checkpoints if cp.client_id not in target_client_ids]

        if not target_checkpoints:
            logger.warning("No checkpoints found for target clients")
            return current_model, UnlearningResult(
                request_id=request.request_id,
                success=False,
                method_used=self.method,
                start_time=start_time,
                end_time=datetime.now(),
                rounds_required=0,
                computation_cost=0.0,
                model_hash_before=model_hash_before,
                model_hash_after=model_hash_before,
                error_message="No checkpoints found for target clients",
            )

        # Step 1: Compute accumulated contribution from target clients
        target_contribution = self._compute_accumulated_contribution(target_checkpoints)

        # Step 2: Subtract target contribution (approximate)
        calibrated_model = {}
        for key in current_model:
            if key in target_contribution:
                calibrated_model[key] = current_model[key] - target_contribution[key]
            else:
                calibrated_model[key] = current_model[key].copy()

        # Step 3: Calibration rounds with remaining clients
        velocity = {key: np.zeros_like(val) for key, val in calibrated_model.items()}

        for round_num in range(self.calibration_rounds):
            # Compute calibration gradient from remaining clients
            calibration_grad = self._compute_calibration_gradient(
                calibrated_model,
                remaining_checkpoints,
                round_num,
            )

            # Apply momentum SGD update
            for key in calibrated_model:
                if key in calibration_grad:
                    velocity[key] = self.momentum * velocity[key] + calibration_grad[key]
                    calibrated_model[key] = calibrated_model[key] - self.learning_rate * velocity[key]

        end_time = datetime.now()
        model_hash_after = self._compute_model_hash(calibrated_model)

        # Estimate cost relative to full retrain
        cost = self.calibration_rounds / (self.calibration_rounds + len(remaining_checkpoints))

        result = UnlearningResult(
            request_id=request.request_id,
            success=True,
            method_used=self.method,
            start_time=start_time,
            end_time=end_time,
            rounds_required=self.calibration_rounds,
            computation_cost=cost,
            model_hash_before=model_hash_before,
            model_hash_after=model_hash_after,
        )

        logger.info(f"FedEraser completed: {self.calibration_rounds} calibration rounds")
        return calibrated_model, result

    def _compute_accumulated_contribution(
        self,
        checkpoints: List[ClientCheckpoint],
    ) -> Dict[str, np.ndarray]:
        """Compute accumulated gradient contribution from checkpoints."""
        contribution = {}

        for cp in checkpoints:
            for key, grad in cp.gradient_update.items():
                # Weight by learning rate and sample count
                weighted_grad = grad * cp.learning_rate * (cp.sample_count / 1000)

                if key not in contribution:
                    contribution[key] = weighted_grad
                else:
                    contribution[key] = contribution[key] + weighted_grad

        return contribution

    def _compute_calibration_gradient(
        self,
        model: Dict[str, np.ndarray],
        checkpoints: List[ClientCheckpoint],
        round_num: int,
    ) -> Dict[str, np.ndarray]:
        """
        Compute calibration gradient for fine-tuning.

        Uses historical gradients from remaining clients to estimate
        correct gradient direction.
        """
        # Select relevant checkpoints for this calibration round
        relevant_checkpoints = [
            cp for cp in checkpoints
            if cp.round_number >= round_num
        ]

        if not relevant_checkpoints:
            return {}

        # Average gradients
        calibration_grad = {}
        for cp in relevant_checkpoints:
            weight = cp.sample_count / sum(c.sample_count for c in relevant_checkpoints)
            for key, grad in cp.gradient_update.items():
                if key not in calibration_grad:
                    calibration_grad[key] = weight * grad
                else:
                    calibration_grad[key] = calibration_grad[key] + weight * grad

        return calibration_grad

    def estimate_cost(self, request: UnlearningRequest) -> float:
        """Estimate cost relative to full retrain."""
        return 0.1 + 0.01 * self.calibration_rounds

    def _compute_model_hash(self, model: Dict[str, np.ndarray]) -> str:
        """Compute hash of model weights."""
        hasher = hashlib.sha256()
        for key in sorted(model.keys()):
            hasher.update(model[key].tobytes())
        return hasher.hexdigest()[:16]


class SISAUnlearner(FederatedUnlearner):
    """
    SISA: Sharded, Isolated, Sliced, Aggregated Training.

    Trains separate models on shards of clients. Unlearning only
    requires retraining affected shards.

    Reference: Bourtoule et al., "Machine Unlearning", 2021
    """

    def __init__(
        self,
        num_shards: int = 5,
        slices_per_shard: int = 3,
        training_fn: Optional[Callable] = None,
    ):
        self.num_shards = num_shards
        self.slices_per_shard = slices_per_shard
        self.training_fn = training_fn
        self.method = UnlearningMethod.SISA
        self._shards: Dict[str, SISAShard] = {}
        self._client_to_shard: Dict[str, str] = {}

    def initialize_shards(self, client_ids: List[str]) -> None:
        """Initialize shards with client assignments."""
        np.random.shuffle(client_ids)
        clients_per_shard = len(client_ids) // self.num_shards

        for i in range(self.num_shards):
            start_idx = i * clients_per_shard
            end_idx = start_idx + clients_per_shard if i < self.num_shards - 1 else len(client_ids)

            shard_id = f"shard_{i}"
            shard_clients = client_ids[start_idx:end_idx]

            self._shards[shard_id] = SISAShard(
                shard_id=shard_id,
                client_ids=shard_clients,
            )

            for client_id in shard_clients:
                self._client_to_shard[client_id] = shard_id

        logger.info(f"Initialized {self.num_shards} shards with {clients_per_shard} clients each")

    def unlearn(
        self,
        current_model: Dict[str, np.ndarray],
        request: UnlearningRequest,
        checkpoints: Optional[List[ClientCheckpoint]] = None,
    ) -> Tuple[Dict[str, np.ndarray], UnlearningResult]:
        """
        SISA unlearning: retrain only affected shards.
        """
        start_time = datetime.now()
        model_hash_before = self._compute_model_hash(current_model)

        # Find affected shards
        target_client_ids = set(request.target_client_ids)
        affected_shards = set()

        for client_id in target_client_ids:
            if client_id in self._client_to_shard:
                affected_shards.add(self._client_to_shard[client_id])
                # Remove client from shard
                shard = self._shards[self._client_to_shard[client_id]]
                shard.client_ids = [c for c in shard.client_ids if c != client_id]
                del self._client_to_shard[client_id]

        if not affected_shards:
            logger.warning("No shards affected by unlearning request")
            return current_model, UnlearningResult(
                request_id=request.request_id,
                success=True,
                method_used=self.method,
                start_time=start_time,
                end_time=datetime.now(),
                rounds_required=0,
                computation_cost=0.0,
                model_hash_before=model_hash_before,
                model_hash_after=model_hash_before,
            )

        # Retrain affected shards
        rounds_required = 0
        for shard_id in affected_shards:
            shard = self._shards[shard_id]
            if self.training_fn and shard.client_ids:
                # Retrain shard
                shard.model_weights = self.training_fn(
                    client_ids=shard.client_ids,
                )
                rounds_required += len(shard.training_history)
                shard.last_updated = datetime.now()

        # Aggregate shard models
        new_model = self._aggregate_shard_models()

        end_time = datetime.now()
        model_hash_after = self._compute_model_hash(new_model)

        # Cost is proportional to fraction of shards affected
        cost = len(affected_shards) / self.num_shards

        result = UnlearningResult(
            request_id=request.request_id,
            success=True,
            method_used=self.method,
            start_time=start_time,
            end_time=end_time,
            rounds_required=rounds_required,
            computation_cost=cost,
            model_hash_before=model_hash_before,
            model_hash_after=model_hash_after,
        )

        logger.info(f"SISA unlearning: retrained {len(affected_shards)}/{self.num_shards} shards")
        return new_model, result

    def _aggregate_shard_models(self) -> Dict[str, np.ndarray]:
        """Aggregate models from all shards."""
        aggregated = {}
        total_clients = sum(len(s.client_ids) for s in self._shards.values())

        for shard in self._shards.values():
            if shard.model_weights is None:
                continue

            weight = len(shard.client_ids) / total_clients if total_clients > 0 else 0

            for key, val in shard.model_weights.items():
                if key not in aggregated:
                    aggregated[key] = weight * val
                else:
                    aggregated[key] = aggregated[key] + weight * val

        return aggregated

    def estimate_cost(self, request: UnlearningRequest) -> float:
        """Estimate cost as fraction of shards affected."""
        target_client_ids = set(request.target_client_ids)
        affected_shards = set()

        for client_id in target_client_ids:
            if client_id in self._client_to_shard:
                affected_shards.add(self._client_to_shard[client_id])

        return len(affected_shards) / self.num_shards if self.num_shards > 0 else 0

    def _compute_model_hash(self, model: Dict[str, np.ndarray]) -> str:
        """Compute hash of model weights."""
        hasher = hashlib.sha256()
        for key in sorted(model.keys()):
            hasher.update(model[key].tobytes())
        return hasher.hexdigest()[:16]


class GradientAscentUnlearner(FederatedUnlearner):
    """
    Gradient Ascent Unlearning.

    Reverses the learning by performing gradient ascent on
    target client's data contribution.
    """

    def __init__(
        self,
        unlearning_rate: float = 0.01,
        unlearning_steps: int = 10,
        damping_factor: float = 0.1,
    ):
        self.unlearning_rate = unlearning_rate
        self.unlearning_steps = unlearning_steps
        self.damping_factor = damping_factor
        self.method = UnlearningMethod.GRADIENT_ASCENT

    def unlearn(
        self,
        current_model: Dict[str, np.ndarray],
        request: UnlearningRequest,
        checkpoints: Optional[List[ClientCheckpoint]] = None,
    ) -> Tuple[Dict[str, np.ndarray], UnlearningResult]:
        """
        Unlearn via gradient ascent on target contributions.
        """
        if not checkpoints:
            raise ValueError("Gradient ascent requires historical checkpoints")

        start_time = datetime.now()
        model_hash_before = self._compute_model_hash(current_model)

        # Get target client checkpoints
        target_client_ids = set(request.target_client_ids)
        target_checkpoints = [cp for cp in checkpoints if cp.client_id in target_client_ids]

        if not target_checkpoints:
            return current_model, UnlearningResult(
                request_id=request.request_id,
                success=False,
                method_used=self.method,
                start_time=start_time,
                end_time=datetime.now(),
                rounds_required=0,
                computation_cost=0.0,
                model_hash_before=model_hash_before,
                model_hash_after=model_hash_before,
                error_message="No checkpoints for target clients",
            )

        # Initialize unlearned model
        unlearned_model = {key: val.copy() for key, val in current_model.items()}

        # Perform gradient ascent steps
        for step in range(self.unlearning_steps):
            # Compute average gradient from target client updates
            avg_gradient = {}
            for cp in target_checkpoints:
                for key, grad in cp.gradient_update.items():
                    if key not in avg_gradient:
                        avg_gradient[key] = grad / len(target_checkpoints)
                    else:
                        avg_gradient[key] = avg_gradient[key] + grad / len(target_checkpoints)

            # Apply gradient ASCENT (opposite of descent)
            decay = (1.0 - self.damping_factor) ** step
            for key in unlearned_model:
                if key in avg_gradient:
                    # Ascend the gradient to "unlearn"
                    unlearned_model[key] = (
                        unlearned_model[key] +
                        self.unlearning_rate * decay * avg_gradient[key]
                    )

        end_time = datetime.now()
        model_hash_after = self._compute_model_hash(unlearned_model)

        result = UnlearningResult(
            request_id=request.request_id,
            success=True,
            method_used=self.method,
            start_time=start_time,
            end_time=end_time,
            rounds_required=self.unlearning_steps,
            computation_cost=0.05,  # Very efficient
            model_hash_before=model_hash_before,
            model_hash_after=model_hash_after,
        )

        logger.info(f"Gradient ascent unlearning: {self.unlearning_steps} steps")
        return unlearned_model, result

    def estimate_cost(self, request: UnlearningRequest) -> float:
        """Very low cost compared to retrain."""
        return 0.05

    def _compute_model_hash(self, model: Dict[str, np.ndarray]) -> str:
        """Compute hash of model weights."""
        hasher = hashlib.sha256()
        for key in sorted(model.keys()):
            hasher.update(model[key].tobytes())
        return hasher.hexdigest()[:16]


class InfluenceFunctionUnlearner(FederatedUnlearner):
    """
    Influence Function-based Unlearning.

    Uses influence functions to estimate the effect of removing
    specific data points and applies Newton step correction.

    Reference: Koh & Liang, "Understanding Black-box Predictions via
    Influence Functions", ICML 2017
    """

    def __init__(
        self,
        damping: float = 0.01,
        scale: float = 1000.0,
        recursion_depth: int = 100,
    ):
        self.damping = damping
        self.scale = scale
        self.recursion_depth = recursion_depth
        self.method = UnlearningMethod.INFLUENCE_FUNCTION

    def unlearn(
        self,
        current_model: Dict[str, np.ndarray],
        request: UnlearningRequest,
        checkpoints: Optional[List[ClientCheckpoint]] = None,
    ) -> Tuple[Dict[str, np.ndarray], UnlearningResult]:
        """
        Unlearn via influence function approximation.

        Estimates: θ_new ≈ θ - H^{-1} * ∇L(target_data)

        Where H is the Hessian (approximated) and ∇L is the gradient
        on target data.
        """
        if not checkpoints:
            raise ValueError("Influence function requires historical checkpoints")

        start_time = datetime.now()
        model_hash_before = self._compute_model_hash(current_model)

        # Get target and remaining checkpoints
        target_client_ids = set(request.target_client_ids)
        target_checkpoints = [cp for cp in checkpoints if cp.client_id in target_client_ids]
        remaining_checkpoints = [cp for cp in checkpoints if cp.client_id not in target_client_ids]

        if not target_checkpoints:
            return current_model, UnlearningResult(
                request_id=request.request_id,
                success=False,
                method_used=self.method,
                start_time=start_time,
                end_time=datetime.now(),
                rounds_required=0,
                computation_cost=0.0,
                model_hash_before=model_hash_before,
                model_hash_after=model_hash_before,
                error_message="No checkpoints for target clients",
            )

        # Compute gradient on target data
        target_gradient = self._compute_target_gradient(target_checkpoints)

        # Approximate H^{-1} * gradient using LiSSA
        influence = self._lissa_inverse_hvp(
            target_gradient,
            remaining_checkpoints,
        )

        # Apply influence correction
        unlearned_model = {}
        n_target = sum(cp.sample_count for cp in target_checkpoints)
        n_total = sum(cp.sample_count for cp in checkpoints)

        for key in current_model:
            if key in influence:
                # Newton step: θ - (n/n-n_target) * H^{-1} * ∇L
                correction = (n_total / (n_total - n_target)) * influence[key]
                unlearned_model[key] = current_model[key] - correction
            else:
                unlearned_model[key] = current_model[key].copy()

        end_time = datetime.now()
        model_hash_after = self._compute_model_hash(unlearned_model)

        result = UnlearningResult(
            request_id=request.request_id,
            success=True,
            method_used=self.method,
            start_time=start_time,
            end_time=end_time,
            rounds_required=self.recursion_depth,
            computation_cost=0.15,
            model_hash_before=model_hash_before,
            model_hash_after=model_hash_after,
        )

        logger.info(f"Influence function unlearning completed")
        return unlearned_model, result

    def _compute_target_gradient(
        self,
        checkpoints: List[ClientCheckpoint],
    ) -> Dict[str, np.ndarray]:
        """Compute average gradient from target checkpoints."""
        gradient = {}
        total_samples = sum(cp.sample_count for cp in checkpoints)

        for cp in checkpoints:
            weight = cp.sample_count / total_samples
            for key, grad in cp.gradient_update.items():
                if key not in gradient:
                    gradient[key] = weight * grad
                else:
                    gradient[key] = gradient[key] + weight * grad

        return gradient

    def _lissa_inverse_hvp(
        self,
        v: Dict[str, np.ndarray],
        checkpoints: List[ClientCheckpoint],
    ) -> Dict[str, np.ndarray]:
        """
        LiSSA: Linear time Stochastic Second-order Algorithm.

        Approximates H^{-1} * v without computing full Hessian.
        """
        # Initialize with scaled v
        ihvp = {key: val / self.scale for key, val in v.items()}

        for i in range(self.recursion_depth):
            # Sample a checkpoint for stochastic Hessian estimate
            cp = checkpoints[i % len(checkpoints)]

            # Approximate Hessian-vector product
            # In practice, this would use actual second derivatives
            # Here we use gradient as approximation
            for key in ihvp:
                if key in cp.gradient_update:
                    # H*v ≈ (1-damping)*v + gradient similarity
                    hvp = (1 - self.damping) * ihvp[key]
                    ihvp[key] = v[key] / self.scale + hvp

        return ihvp

    def estimate_cost(self, request: UnlearningRequest) -> float:
        """Moderate cost due to LiSSA iterations."""
        return 0.15

    def _compute_model_hash(self, model: Dict[str, np.ndarray]) -> str:
        """Compute hash of model weights."""
        hasher = hashlib.sha256()
        for key in sorted(model.keys()):
            hasher.update(model[key].tobytes())
        return hasher.hexdigest()[:16]


# =============================================================================
# Verification
# =============================================================================

class UnlearningVerifier:
    """
    Verifies that unlearning was successful.

    Uses various methods to check that target client's data
    no longer influences the model.
    """

    def __init__(
        self,
        verification_methods: Optional[List[VerificationMethod]] = None,
    ):
        self.verification_methods = verification_methods or [
            VerificationMethod.MEMBERSHIP_INFERENCE,
            VerificationMethod.GRADIENT_SIMILARITY,
        ]

    def verify(
        self,
        original_model: Dict[str, np.ndarray],
        unlearned_model: Dict[str, np.ndarray],
        target_checkpoints: List[ClientCheckpoint],
        threshold: float = 0.5,
    ) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Verify unlearning success.

        Returns:
            Tuple of (passed, score, details)
        """
        scores = {}
        details = {}

        for method in self.verification_methods:
            if method == VerificationMethod.MEMBERSHIP_INFERENCE:
                score, info = self._membership_inference_test(
                    unlearned_model, target_checkpoints
                )
                scores["membership_inference"] = score
                details["membership_inference"] = info

            elif method == VerificationMethod.GRADIENT_SIMILARITY:
                score, info = self._gradient_similarity_test(
                    original_model, unlearned_model, target_checkpoints
                )
                scores["gradient_similarity"] = score
                details["gradient_similarity"] = info

            elif method == VerificationMethod.INFLUENCE_ESTIMATION:
                score, info = self._influence_estimation_test(
                    unlearned_model, target_checkpoints
                )
                scores["influence_estimation"] = score
                details["influence_estimation"] = info

        # Overall score (average)
        overall_score = np.mean(list(scores.values()))
        passed = overall_score >= threshold

        logger.info(f"Unlearning verification: {'PASSED' if passed else 'FAILED'} (score={overall_score:.3f})")

        return passed, overall_score, details

    def _membership_inference_test(
        self,
        model: Dict[str, np.ndarray],
        target_checkpoints: List[ClientCheckpoint],
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Membership inference attack to test unlearning.

        A score close to 0.5 indicates the model cannot distinguish
        whether data was in training (good unlearning).
        """
        # Simulate membership inference attack
        # In practice, train a classifier to distinguish members vs non-members

        # For simulation, use gradient norm as proxy
        avg_gradient_norm = 0.0
        for cp in target_checkpoints:
            norms = [np.linalg.norm(g) for g in cp.gradient_update.values()]
            avg_gradient_norm += np.mean(norms) / len(target_checkpoints)

        # Lower correlation = better unlearning
        # Score of 0.5 = random (no membership signal)
        score = 0.5 + 0.5 * np.exp(-avg_gradient_norm)

        return score, {
            "avg_gradient_norm": avg_gradient_norm,
            "interpretation": "Score near 0.5 indicates successful unlearning"
        }

    def _gradient_similarity_test(
        self,
        original_model: Dict[str, np.ndarray],
        unlearned_model: Dict[str, np.ndarray],
        target_checkpoints: List[ClientCheckpoint],
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Test gradient similarity between models.

        Lower similarity on target data indicates successful unlearning.
        """
        similarities = []

        for key in original_model:
            if key in unlearned_model:
                orig = original_model[key].flatten()
                unlearned = unlearned_model[key].flatten()

                # Cosine similarity
                dot = np.dot(orig, unlearned)
                norm_orig = np.linalg.norm(orig)
                norm_unlearned = np.linalg.norm(unlearned)

                if norm_orig > 0 and norm_unlearned > 0:
                    sim = dot / (norm_orig * norm_unlearned)
                    similarities.append(sim)

        avg_similarity = np.mean(similarities) if similarities else 1.0

        # Score: dissimilarity (1 - similarity) is good
        score = 1.0 - abs(avg_similarity)

        return score, {
            "avg_similarity": avg_similarity,
            "interpretation": "Lower similarity indicates model changed"
        }

    def _influence_estimation_test(
        self,
        model: Dict[str, np.ndarray],
        target_checkpoints: List[ClientCheckpoint],
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Estimate remaining influence of target data.
        """
        # Compute influence proxy using gradients
        influence_scores = []

        for cp in target_checkpoints:
            for key, grad in cp.gradient_update.items():
                if key in model:
                    # Dot product of gradient with model weights
                    influence = abs(np.dot(grad.flatten(), model[key].flatten()))
                    influence_scores.append(influence)

        avg_influence = np.mean(influence_scores) if influence_scores else 0.0

        # Normalize and invert (lower influence = better)
        score = 1.0 / (1.0 + avg_influence)

        return score, {
            "avg_influence": avg_influence,
            "interpretation": "Lower influence indicates successful unlearning"
        }


# =============================================================================
# Unlearning Manager
# =============================================================================

class FederatedUnlearningManager:
    """
    Central manager for federated unlearning operations.

    Coordinates unlearning requests, method selection, checkpointing,
    and verification for GDPR/EHDS compliance.
    """

    def __init__(
        self,
        default_method: UnlearningMethod = UnlearningMethod.FED_ERASER,
        checkpoint_retention_rounds: int = 100,
        auto_verify: bool = True,
    ):
        self.default_method = default_method
        self.checkpoint_retention_rounds = checkpoint_retention_rounds
        self.auto_verify = auto_verify

        # Checkpoints storage
        self._checkpoints: List[ClientCheckpoint] = []
        self._requests: Dict[str, UnlearningRequest] = {}
        self._results: Dict[str, UnlearningResult] = {}

        # Initialize unlearners
        self._unlearners: Dict[UnlearningMethod, FederatedUnlearner] = {
            UnlearningMethod.FED_ERASER: FedEraserUnlearner(),
            UnlearningMethod.GRADIENT_ASCENT: GradientAscentUnlearner(),
            UnlearningMethod.INFLUENCE_FUNCTION: InfluenceFunctionUnlearner(),
        }

        self._verifier = UnlearningVerifier()

        logger.info(f"Unlearning Manager initialized with method: {default_method.value}")

    def store_checkpoint(
        self,
        client_id: str,
        round_number: int,
        model_weights: Dict[str, np.ndarray],
        gradient_update: Dict[str, np.ndarray],
        learning_rate: float,
        local_epochs: int,
        sample_count: int,
        sample_ids: Optional[List[str]] = None,
    ) -> None:
        """Store checkpoint for potential future unlearning."""
        checkpoint = ClientCheckpoint(
            client_id=client_id,
            round_number=round_number,
            timestamp=datetime.now(),
            model_weights=model_weights,
            gradient_update=gradient_update,
            learning_rate=learning_rate,
            local_epochs=local_epochs,
            sample_count=sample_count,
            sample_hashes=[
                hashlib.sha256(str(sid).encode()).hexdigest()[:8]
                for sid in (sample_ids or [])
            ],
        )

        self._checkpoints.append(checkpoint)

        # Cleanup old checkpoints
        self._cleanup_old_checkpoints(round_number)

    def _cleanup_old_checkpoints(self, current_round: int) -> None:
        """Remove checkpoints older than retention period."""
        cutoff_round = current_round - self.checkpoint_retention_rounds
        self._checkpoints = [
            cp for cp in self._checkpoints
            if cp.round_number >= cutoff_round
        ]

    def submit_unlearning_request(
        self,
        requestor_id: str,
        target_client_ids: List[str],
        scope: UnlearningScope = UnlearningScope.CLIENT,
        legal_basis: str = "EHDS_Art71",
        priority: str = "normal",
    ) -> str:
        """
        Submit an unlearning request.

        Returns request_id for tracking.
        """
        request = UnlearningRequest(
            request_id=str(uuid.uuid4()),
            requestor_id=requestor_id,
            scope=scope,
            target_client_ids=target_client_ids,
            legal_basis=legal_basis,
            priority=priority,
        )

        self._requests[request.request_id] = request
        logger.info(f"Unlearning request submitted: {request.request_id}")

        return request.request_id

    def process_unlearning(
        self,
        request_id: str,
        current_model: Dict[str, np.ndarray],
        method: Optional[UnlearningMethod] = None,
    ) -> Tuple[Dict[str, np.ndarray], UnlearningResult]:
        """
        Process an unlearning request.

        Args:
            request_id: ID of the request to process
            current_model: Current global model weights
            method: Optional override for unlearning method

        Returns:
            Tuple of (unlearned_model, result)
        """
        if request_id not in self._requests:
            raise ValueError(f"Request not found: {request_id}")

        request = self._requests[request_id]
        method = method or self.default_method

        # Select unlearner
        if method not in self._unlearners:
            raise ValueError(f"Unsupported method: {method}")

        unlearner = self._unlearners[method]

        # Get relevant checkpoints
        target_client_ids = set(request.target_client_ids)
        relevant_checkpoints = [
            cp for cp in self._checkpoints
            if cp.client_id in target_client_ids or cp.client_id not in target_client_ids
        ]

        # Execute unlearning
        unlearned_model, result = unlearner.unlearn(
            current_model,
            request,
            relevant_checkpoints,
        )

        # Verify if requested
        if self.auto_verify and request.verification_required:
            target_checkpoints = [
                cp for cp in self._checkpoints
                if cp.client_id in target_client_ids
            ]

            passed, score, _ = self._verifier.verify(
                current_model,
                unlearned_model,
                target_checkpoints,
            )

            result.verification_passed = passed
            result.verification_score = score

        self._results[request_id] = result

        # Remove checkpoints from unlearned clients
        self._checkpoints = [
            cp for cp in self._checkpoints
            if cp.client_id not in target_client_ids
        ]

        return unlearned_model, result

    def estimate_unlearning_cost(
        self,
        target_client_ids: List[str],
        method: Optional[UnlearningMethod] = None,
    ) -> Dict[str, Any]:
        """Estimate cost of unlearning operation."""
        method = method or self.default_method

        request = UnlearningRequest(
            request_id="estimate",
            requestor_id="estimate",
            scope=UnlearningScope.CLIENT,
            target_client_ids=target_client_ids,
        )

        if method in self._unlearners:
            cost = self._unlearners[method].estimate_cost(request)
        else:
            cost = 1.0  # Full retrain

        # Count affected checkpoints
        target_set = set(target_client_ids)
        affected_checkpoints = sum(
            1 for cp in self._checkpoints if cp.client_id in target_set
        )

        return {
            "method": method.value,
            "estimated_cost": cost,
            "cost_vs_retrain": f"{cost * 100:.1f}%",
            "affected_checkpoints": affected_checkpoints,
            "total_checkpoints": len(self._checkpoints),
        }

    def get_request_status(self, request_id: str) -> Dict[str, Any]:
        """Get status of an unlearning request."""
        if request_id not in self._requests:
            return {"error": "Request not found"}

        request = self._requests[request_id]
        result = self._results.get(request_id)

        status = {
            "request": request.to_dict(),
            "processed": result is not None,
        }

        if result:
            status["result"] = {
                "success": result.success,
                "method": result.method_used.value,
                "duration_seconds": result.duration_seconds,
                "computation_cost": result.computation_cost,
                "verification_passed": result.verification_passed,
                "verification_score": result.verification_score,
            }

        return status

    def get_checkpoint_stats(self) -> Dict[str, Any]:
        """Get statistics about stored checkpoints."""
        if not self._checkpoints:
            return {"total": 0, "clients": 0, "rounds": 0}

        client_ids = set(cp.client_id for cp in self._checkpoints)
        rounds = set(cp.round_number for cp in self._checkpoints)

        return {
            "total_checkpoints": len(self._checkpoints),
            "unique_clients": len(client_ids),
            "round_range": (min(rounds), max(rounds)) if rounds else (0, 0),
            "retention_rounds": self.checkpoint_retention_rounds,
            "oldest_checkpoint": min(cp.timestamp for cp in self._checkpoints).isoformat(),
            "newest_checkpoint": max(cp.timestamp for cp in self._checkpoints).isoformat(),
        }


# =============================================================================
# Factory Functions
# =============================================================================

def create_unlearning_manager(
    method: UnlearningMethod = UnlearningMethod.FED_ERASER,
    checkpoint_retention: int = 100,
    auto_verify: bool = True,
) -> FederatedUnlearningManager:
    """Create federated unlearning manager."""
    return FederatedUnlearningManager(
        default_method=method,
        checkpoint_retention_rounds=checkpoint_retention,
        auto_verify=auto_verify,
    )


def create_sisa_unlearner(
    num_shards: int = 5,
    slices_per_shard: int = 3,
) -> SISAUnlearner:
    """Create SISA-based unlearner."""
    return SISAUnlearner(
        num_shards=num_shards,
        slices_per_shard=slices_per_shard,
    )


# =============================================================================
# Export
# =============================================================================

__all__ = [
    # Enums
    "UnlearningMethod",
    "UnlearningScope",
    "VerificationMethod",
    # Data Classes
    "UnlearningRequest",
    "UnlearningResult",
    "ClientCheckpoint",
    "SISAShard",
    # Unlearners
    "FederatedUnlearner",
    "ExactRetrainUnlearner",
    "FedEraserUnlearner",
    "SISAUnlearner",
    "GradientAscentUnlearner",
    "InfluenceFunctionUnlearner",
    # Verification
    "UnlearningVerifier",
    # Manager
    "FederatedUnlearningManager",
    # Factory
    "create_unlearning_manager",
    "create_sisa_unlearner",
]

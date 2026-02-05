#!/usr/bin/env python3
"""
FL-EHDS Cross-Silo Enhancements

Advanced capabilities for enterprise federated learning deployments:
1. Multi-Model Federation: Ensemble of federated models
2. Federated Model Selection: Automatic algorithm selection
3. Adaptive Aggregation: Dynamic algorithm switching based on runtime metrics

Author: Fabio Liberti
Date: February 2026
"""

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from datetime import datetime
import logging
import json
import hashlib

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND CONFIGURATIONS
# =============================================================================

class EnsembleStrategy(Enum):
    """Strategies for combining ensemble predictions."""
    WEIGHTED_VOTING = auto()      # Weighted average of predictions
    MAJORITY_VOTING = auto()      # Majority vote for classification
    STACKING = auto()             # Meta-learner on model outputs
    BAGGING = auto()              # Bootstrap aggregating
    BOOSTING = auto()             # Sequential boosting
    MIXTURE_OF_EXPERTS = auto()   # Gating network selects experts


class AggregationAlgorithm(Enum):
    """Available FL aggregation algorithms."""
    FEDAVG = "fedavg"
    FEDPROX = "fedprox"
    SCAFFOLD = "scaffold"
    FEDADAM = "fedadam"
    FEDYOGI = "fedyogi"
    FEDADAGRAD = "fedadagrad"
    FEDNOVA = "fednova"
    FEDDYN = "feddyn"
    KRUM = "krum"
    TRIMMED_MEAN = "trimmed_mean"
    MEDIAN = "median"
    BULYAN = "bulyan"


class TaskType(Enum):
    """Types of ML tasks for algorithm selection."""
    BINARY_CLASSIFICATION = auto()
    MULTICLASS_CLASSIFICATION = auto()
    REGRESSION = auto()
    MULTI_LABEL = auto()
    SEQUENCE = auto()
    IMAGE = auto()
    NLP = auto()


class DataCharacteristic(Enum):
    """Data distribution characteristics."""
    IID = auto()
    NON_IID_LABEL_SKEW = auto()
    NON_IID_FEATURE_SKEW = auto()
    NON_IID_QUANTITY_SKEW = auto()
    NON_IID_EXTREME = auto()


class SelectionCriterion(Enum):
    """Criteria for algorithm selection."""
    ACCURACY = auto()
    CONVERGENCE_SPEED = auto()
    COMMUNICATION_EFFICIENCY = auto()
    PRIVACY = auto()
    ROBUSTNESS = auto()
    FAIRNESS = auto()


@dataclass
class EnsembleConfig:
    """Configuration for multi-model federation."""
    strategy: EnsembleStrategy = EnsembleStrategy.WEIGHTED_VOTING
    num_models: int = 3
    diversity_weight: float = 0.3  # Weight for diversity in selection
    temperature: float = 1.0  # Softmax temperature for voting
    meta_learner_type: str = "logistic"  # For stacking
    bootstrap_ratio: float = 0.8  # For bagging
    enable_pruning: bool = True  # Prune underperforming models
    pruning_threshold: float = 0.1  # Remove if < threshold * best


@dataclass
class ModelSelectionConfig:
    """Configuration for federated model selection."""
    criterion: SelectionCriterion = SelectionCriterion.ACCURACY
    exploration_rounds: int = 5  # Rounds per algorithm during exploration
    exploitation_ratio: float = 0.8  # Fraction of rounds for best algorithm
    algorithms_to_try: List[AggregationAlgorithm] = field(
        default_factory=lambda: [
            AggregationAlgorithm.FEDAVG,
            AggregationAlgorithm.FEDPROX,
            AggregationAlgorithm.SCAFFOLD,
        ]
    )
    use_warm_start: bool = True  # Reuse model between algorithm switches
    bandit_algorithm: str = "ucb"  # ucb, thompson, epsilon_greedy


@dataclass
class AdaptiveAggregationConfig:
    """Configuration for adaptive aggregation."""
    initial_algorithm: AggregationAlgorithm = AggregationAlgorithm.FEDAVG
    evaluation_window: int = 5  # Rounds to evaluate before switching
    switch_threshold: float = 0.05  # Min improvement to switch
    cooldown_rounds: int = 10  # Min rounds between switches

    # Algorithm-specific parameters
    fedprox_mu: float = 0.1
    scaffold_lr: float = 0.1
    fedadam_lr: float = 0.01
    fedadam_beta1: float = 0.9
    fedadam_beta2: float = 0.999
    fedadam_tau: float = 1e-3
    fedyogi_lr: float = 0.01
    krum_num_byzantine: int = 0
    trimmed_mean_beta: float = 0.1

    # Metrics to monitor
    metrics_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "loss": 0.4,
            "accuracy": 0.3,
            "client_variance": 0.2,
            "convergence_rate": 0.1
        }
    )


@dataclass
class ModelState:
    """State of a federated model in the ensemble."""
    model_id: str
    weights: Dict[str, np.ndarray]
    algorithm: AggregationAlgorithm
    performance_history: List[float] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AggregationMetrics:
    """Metrics for monitoring aggregation performance."""
    round_number: int
    algorithm: AggregationAlgorithm
    loss: float
    accuracy: float
    client_variance: float
    convergence_rate: float
    communication_cost: float
    computation_time: float
    num_participants: int
    timestamp: datetime = field(default_factory=datetime.now)


# =============================================================================
# MULTI-MODEL FEDERATION
# =============================================================================

class ModelRegistry:
    """Registry for managing multiple federated models."""

    def __init__(self, max_models: int = 10):
        self.max_models = max_models
        self._models: Dict[str, ModelState] = {}
        self._performance_cache: Dict[str, float] = {}

    def register_model(
        self,
        weights: Dict[str, np.ndarray],
        algorithm: AggregationAlgorithm,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Register a new model in the registry."""
        model_id = self._generate_model_id(weights)

        if len(self._models) >= self.max_models:
            self._evict_worst_model()

        self._models[model_id] = ModelState(
            model_id=model_id,
            weights=weights,
            algorithm=algorithm,
            metadata=metadata or {}
        )

        logger.info(f"Registered model {model_id} with algorithm {algorithm.value}")
        return model_id

    def get_model(self, model_id: str) -> Optional[ModelState]:
        """Retrieve a model by ID."""
        return self._models.get(model_id)

    def update_model(
        self,
        model_id: str,
        weights: Dict[str, np.ndarray],
        performance: Optional[float] = None
    ):
        """Update model weights and optionally record performance."""
        if model_id not in self._models:
            raise ValueError(f"Model {model_id} not found")

        model = self._models[model_id]
        model.weights = weights
        model.last_updated = datetime.now()

        if performance is not None:
            model.performance_history.append(performance)
            self._performance_cache[model_id] = performance

    def get_all_models(self) -> List[ModelState]:
        """Get all registered models."""
        return list(self._models.values())

    def get_best_models(self, n: int = 3) -> List[ModelState]:
        """Get the n best performing models."""
        sorted_models = sorted(
            self._models.values(),
            key=lambda m: self._performance_cache.get(m.model_id, 0),
            reverse=True
        )
        return sorted_models[:n]

    def remove_model(self, model_id: str):
        """Remove a model from the registry."""
        if model_id in self._models:
            del self._models[model_id]
            self._performance_cache.pop(model_id, None)

    def _generate_model_id(self, weights: Dict[str, np.ndarray]) -> str:
        """Generate a unique model ID."""
        hasher = hashlib.md5()
        for name, w in sorted(weights.items()):
            hasher.update(name.encode())
            hasher.update(w.tobytes()[:100])  # Sample for speed
        hasher.update(str(datetime.now().timestamp()).encode())
        return hasher.hexdigest()[:12]

    def _evict_worst_model(self):
        """Evict the worst performing model."""
        if not self._models:
            return

        worst_id = min(
            self._models.keys(),
            key=lambda mid: self._performance_cache.get(mid, float('inf'))
        )
        self.remove_model(worst_id)
        logger.info(f"Evicted model {worst_id} due to capacity limit")


class EnsembleCombiner(ABC):
    """Abstract base class for ensemble combination strategies."""

    @abstractmethod
    def combine_predictions(
        self,
        predictions: List[np.ndarray],
        weights: Optional[List[float]] = None
    ) -> np.ndarray:
        """Combine predictions from multiple models."""
        pass

    @abstractmethod
    def combine_gradients(
        self,
        gradients: List[Dict[str, np.ndarray]],
        weights: Optional[List[float]] = None
    ) -> Dict[str, np.ndarray]:
        """Combine gradients from multiple models."""
        pass


class WeightedVotingCombiner(EnsembleCombiner):
    """Weighted voting ensemble combiner."""

    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature

    def combine_predictions(
        self,
        predictions: List[np.ndarray],
        weights: Optional[List[float]] = None
    ) -> np.ndarray:
        """Combine predictions using weighted average."""
        if weights is None:
            weights = [1.0 / len(predictions)] * len(predictions)

        weights = np.array(weights)
        weights = weights / weights.sum()

        # Stack and weight
        stacked = np.stack(predictions, axis=0)
        combined = np.sum(stacked * weights[:, np.newaxis, ...], axis=0)

        return combined

    def combine_gradients(
        self,
        gradients: List[Dict[str, np.ndarray]],
        weights: Optional[List[float]] = None
    ) -> Dict[str, np.ndarray]:
        """Combine gradients using weighted average."""
        if weights is None:
            weights = [1.0 / len(gradients)] * len(gradients)

        weights = np.array(weights)
        weights = weights / weights.sum()

        combined = {}
        param_names = gradients[0].keys()

        for name in param_names:
            stacked = np.stack([g[name] for g in gradients], axis=0)
            combined[name] = np.sum(
                stacked * weights[:, np.newaxis, ...].reshape(
                    [len(weights)] + [1] * (stacked.ndim - 1)
                ),
                axis=0
            )

        return combined


class StackingCombiner(EnsembleCombiner):
    """Stacking ensemble with meta-learner."""

    def __init__(self, meta_learner_type: str = "logistic"):
        self.meta_learner_type = meta_learner_type
        self.meta_weights: Optional[np.ndarray] = None
        self._fitted = False

    def fit_meta_learner(
        self,
        base_predictions: List[np.ndarray],
        targets: np.ndarray
    ):
        """Fit the meta-learner on base model predictions."""
        # Stack base predictions as features
        X = np.concatenate(base_predictions, axis=-1)

        # Simple logistic/linear meta-learner (closed-form)
        if self.meta_learner_type == "logistic":
            # Use softmax weights based on correlation with target
            correlations = []
            for pred in base_predictions:
                if pred.ndim > 1:
                    pred_flat = pred.argmax(axis=-1) if pred.shape[-1] > 1 else pred.flatten()
                else:
                    pred_flat = pred
                corr = np.corrcoef(pred_flat, targets.flatten())[0, 1]
                correlations.append(max(corr, 0.01))  # Ensure positive

            self.meta_weights = np.array(correlations)
            self.meta_weights = self.meta_weights / self.meta_weights.sum()
        else:
            # Uniform weights as fallback
            self.meta_weights = np.ones(len(base_predictions)) / len(base_predictions)

        self._fitted = True

    def combine_predictions(
        self,
        predictions: List[np.ndarray],
        weights: Optional[List[float]] = None
    ) -> np.ndarray:
        """Combine using learned meta-weights."""
        if self._fitted and self.meta_weights is not None:
            weights = self.meta_weights.tolist()
        elif weights is None:
            weights = [1.0 / len(predictions)] * len(predictions)

        weights = np.array(weights)
        weights = weights / weights.sum()

        stacked = np.stack(predictions, axis=0)
        combined = np.sum(stacked * weights[:, np.newaxis, ...], axis=0)

        return combined

    def combine_gradients(
        self,
        gradients: List[Dict[str, np.ndarray]],
        weights: Optional[List[float]] = None
    ) -> Dict[str, np.ndarray]:
        """Combine gradients using meta-weights."""
        if self._fitted and self.meta_weights is not None:
            weights = self.meta_weights.tolist()

        combiner = WeightedVotingCombiner()
        return combiner.combine_gradients(gradients, weights)


class MixtureOfExpertsCombiner(EnsembleCombiner):
    """Mixture of Experts with gating network."""

    def __init__(self, num_experts: int, input_dim: int):
        self.num_experts = num_experts
        self.input_dim = input_dim
        # Simple linear gating weights
        self.gate_weights = np.random.randn(input_dim, num_experts) * 0.01
        self.gate_bias = np.zeros(num_experts)

    def compute_gate(self, x: np.ndarray) -> np.ndarray:
        """Compute gating weights for input."""
        # x: (batch, input_dim)
        logits = x @ self.gate_weights + self.gate_bias
        # Softmax
        exp_logits = np.exp(logits - logits.max(axis=-1, keepdims=True))
        gates = exp_logits / exp_logits.sum(axis=-1, keepdims=True)
        return gates  # (batch, num_experts)

    def combine_predictions(
        self,
        predictions: List[np.ndarray],
        weights: Optional[List[float]] = None,
        inputs: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Combine predictions using input-dependent gating."""
        if inputs is not None and len(predictions) == self.num_experts:
            gates = self.compute_gate(inputs)
            stacked = np.stack(predictions, axis=-1)  # (batch, classes, experts)
            # Weighted combination
            combined = np.einsum('bce,be->bc', stacked, gates)
            return combined
        else:
            # Fallback to uniform
            return np.mean(predictions, axis=0)

    def combine_gradients(
        self,
        gradients: List[Dict[str, np.ndarray]],
        weights: Optional[List[float]] = None
    ) -> Dict[str, np.ndarray]:
        """Combine gradients (uniform for simplicity)."""
        combiner = WeightedVotingCombiner()
        return combiner.combine_gradients(gradients, weights)


class FederatedEnsemble:
    """
    Multi-Model Federation: Manages ensemble of federated models.

    Supports multiple training strategies and combination methods.
    """

    def __init__(self, config: EnsembleConfig):
        self.config = config
        self.registry = ModelRegistry(max_models=config.num_models * 2)
        self.combiner = self._create_combiner()
        self._model_ids: List[str] = []
        self._diversity_scores: Dict[str, float] = {}

    def _create_combiner(self) -> EnsembleCombiner:
        """Create the appropriate combiner based on strategy."""
        if self.config.strategy == EnsembleStrategy.WEIGHTED_VOTING:
            return WeightedVotingCombiner(self.config.temperature)
        elif self.config.strategy == EnsembleStrategy.STACKING:
            return StackingCombiner(self.config.meta_learner_type)
        elif self.config.strategy == EnsembleStrategy.MIXTURE_OF_EXPERTS:
            return MixtureOfExpertsCombiner(self.config.num_models, 100)
        else:
            return WeightedVotingCombiner()

    def add_model(
        self,
        weights: Dict[str, np.ndarray],
        algorithm: AggregationAlgorithm,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Add a model to the ensemble."""
        # Check diversity before adding
        if self._model_ids and self.config.diversity_weight > 0:
            diversity = self._compute_diversity(weights)
            if diversity < 0.1:  # Too similar to existing models
                logger.warning("Model too similar to existing ensemble members")
                return ""

        model_id = self.registry.register_model(weights, algorithm, metadata)
        self._model_ids.append(model_id)

        # Prune if exceeding capacity
        if len(self._model_ids) > self.config.num_models:
            self._prune_ensemble()

        return model_id

    def aggregate_round(
        self,
        client_updates: List[Dict[str, np.ndarray]],
        client_weights: Optional[List[float]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Aggregate client updates across all ensemble models.

        Returns the combined aggregated weights.
        """
        if not self._model_ids:
            # No ensemble yet, return simple average
            return self._simple_aggregate(client_updates, client_weights)

        # Get current ensemble weights
        ensemble_weights = []
        for model_id in self._model_ids:
            model = self.registry.get_model(model_id)
            if model:
                ensemble_weights.append(model.weights)

        if not ensemble_weights:
            return self._simple_aggregate(client_updates, client_weights)

        # Aggregate client updates
        aggregated_update = self._simple_aggregate(client_updates, client_weights)

        # Apply update to each ensemble member
        updated_ensemble = []
        for i, ew in enumerate(ensemble_weights):
            updated = {}
            for name in ew:
                if name in aggregated_update:
                    # Apply scaled update (diversity-aware)
                    scale = 1.0 + (i * 0.1 * self.config.diversity_weight)
                    updated[name] = ew[name] + scale * (aggregated_update[name] - ew[name])
                else:
                    updated[name] = ew[name]
            updated_ensemble.append(updated)

        # Combine ensemble
        performance_weights = self._get_performance_weights()
        combined = self.combiner.combine_gradients(updated_ensemble, performance_weights)

        # Update registry
        for model_id, updated in zip(self._model_ids, updated_ensemble):
            self.registry.update_model(model_id, updated)

        return combined

    def predict(
        self,
        inputs: np.ndarray,
        model_forward_fn: Callable[[Dict[str, np.ndarray], np.ndarray], np.ndarray]
    ) -> np.ndarray:
        """Generate ensemble predictions."""
        predictions = []

        for model_id in self._model_ids:
            model = self.registry.get_model(model_id)
            if model:
                pred = model_forward_fn(model.weights, inputs)
                predictions.append(pred)

        if not predictions:
            raise ValueError("No models in ensemble")

        weights = self._get_performance_weights()
        return self.combiner.combine_predictions(predictions, weights)

    def update_performance(self, model_id: str, performance: float):
        """Update performance metric for a model."""
        self.registry.update_model(model_id,
                                   self.registry.get_model(model_id).weights,
                                   performance)

        if self.config.enable_pruning:
            self._prune_ensemble()

    def _simple_aggregate(
        self,
        updates: List[Dict[str, np.ndarray]],
        weights: Optional[List[float]] = None
    ) -> Dict[str, np.ndarray]:
        """Simple weighted average aggregation."""
        if weights is None:
            weights = [1.0 / len(updates)] * len(updates)

        weights = np.array(weights)
        weights = weights / weights.sum()

        aggregated = {}
        for name in updates[0]:
            stacked = np.stack([u[name] for u in updates], axis=0)
            aggregated[name] = np.sum(
                stacked * weights[:, np.newaxis, ...].reshape(
                    [len(weights)] + [1] * (stacked.ndim - 1)
                ),
                axis=0
            )

        return aggregated

    def _compute_diversity(self, new_weights: Dict[str, np.ndarray]) -> float:
        """Compute diversity of new model w.r.t. ensemble."""
        if not self._model_ids:
            return 1.0

        diversities = []
        new_flat = np.concatenate([w.flatten() for w in new_weights.values()])

        for model_id in self._model_ids:
            model = self.registry.get_model(model_id)
            if model:
                existing_flat = np.concatenate([w.flatten() for w in model.weights.values()])
                # Cosine distance
                sim = np.dot(new_flat, existing_flat) / (
                    np.linalg.norm(new_flat) * np.linalg.norm(existing_flat) + 1e-8
                )
                diversities.append(1 - sim)

        return np.mean(diversities) if diversities else 1.0

    def _get_performance_weights(self) -> List[float]:
        """Get performance-based weights for ensemble combination."""
        if not self._model_ids:
            return []

        performances = []
        for model_id in self._model_ids:
            model = self.registry.get_model(model_id)
            if model and model.performance_history:
                performances.append(np.mean(model.performance_history[-5:]))
            else:
                performances.append(0.5)  # Default

        # Softmax normalization
        performances = np.array(performances)
        exp_perf = np.exp((performances - performances.max()) / self.config.temperature)
        weights = exp_perf / exp_perf.sum()

        return weights.tolist()

    def _prune_ensemble(self):
        """Prune underperforming models from ensemble."""
        if len(self._model_ids) <= 1:
            return

        performances = []
        for model_id in self._model_ids:
            model = self.registry.get_model(model_id)
            if model and model.performance_history:
                performances.append((model_id, np.mean(model.performance_history[-5:])))
            else:
                performances.append((model_id, 0.5))

        if not performances:
            return

        best_perf = max(p[1] for p in performances)
        threshold = best_perf * self.config.pruning_threshold

        to_remove = [mid for mid, perf in performances if perf < threshold]

        # Keep at least one model
        if len(to_remove) >= len(self._model_ids):
            to_remove = to_remove[:-1]

        for model_id in to_remove:
            self._model_ids.remove(model_id)
            self.registry.remove_model(model_id)
            logger.info(f"Pruned model {model_id} from ensemble")

    def get_ensemble_info(self) -> Dict[str, Any]:
        """Get information about current ensemble state."""
        models_info = []
        for model_id in self._model_ids:
            model = self.registry.get_model(model_id)
            if model:
                models_info.append({
                    "model_id": model_id,
                    "algorithm": model.algorithm.value,
                    "num_updates": len(model.performance_history),
                    "avg_performance": np.mean(model.performance_history[-10:]) if model.performance_history else None,
                    "created_at": model.created_at.isoformat()
                })

        return {
            "strategy": self.config.strategy.name,
            "num_models": len(self._model_ids),
            "max_models": self.config.num_models,
            "models": models_info
        }


# =============================================================================
# FEDERATED MODEL SELECTION
# =============================================================================

@dataclass
class AlgorithmPerformance:
    """Performance metrics for an FL algorithm."""
    algorithm: AggregationAlgorithm
    rounds_evaluated: int
    total_loss: float
    total_accuracy: float
    convergence_rates: List[float]
    communication_costs: List[float]
    client_variances: List[float]

    @property
    def avg_loss(self) -> float:
        return self.total_loss / max(self.rounds_evaluated, 1)

    @property
    def avg_accuracy(self) -> float:
        return self.total_accuracy / max(self.rounds_evaluated, 1)

    @property
    def avg_convergence(self) -> float:
        return np.mean(self.convergence_rates) if self.convergence_rates else 0.0


class TaskAnalyzer:
    """Analyzes task characteristics to recommend FL algorithms."""

    def __init__(self):
        self._algorithm_recommendations = {
            # (task_type, data_characteristic): recommended algorithms
            (TaskType.BINARY_CLASSIFICATION, DataCharacteristic.IID): [
                AggregationAlgorithm.FEDAVG,
                AggregationAlgorithm.FEDADAM,
            ],
            (TaskType.BINARY_CLASSIFICATION, DataCharacteristic.NON_IID_LABEL_SKEW): [
                AggregationAlgorithm.SCAFFOLD,
                AggregationAlgorithm.FEDPROX,
            ],
            (TaskType.BINARY_CLASSIFICATION, DataCharacteristic.NON_IID_EXTREME): [
                AggregationAlgorithm.SCAFFOLD,
                AggregationAlgorithm.FEDDYN,
            ],
            (TaskType.MULTICLASS_CLASSIFICATION, DataCharacteristic.NON_IID_LABEL_SKEW): [
                AggregationAlgorithm.SCAFFOLD,
                AggregationAlgorithm.FEDPROX,
                AggregationAlgorithm.FEDNOVA,
            ],
            (TaskType.REGRESSION, DataCharacteristic.IID): [
                AggregationAlgorithm.FEDAVG,
                AggregationAlgorithm.FEDADAGRAD,
            ],
            (TaskType.REGRESSION, DataCharacteristic.NON_IID_FEATURE_SKEW): [
                AggregationAlgorithm.FEDPROX,
                AggregationAlgorithm.FEDYOGI,
            ],
        }

    def analyze_data_distribution(
        self,
        client_label_distributions: List[Dict[int, int]]
    ) -> DataCharacteristic:
        """Analyze client data distributions to determine heterogeneity type."""
        if not client_label_distributions:
            return DataCharacteristic.IID

        # Compute label distribution statistics
        all_labels = set()
        for dist in client_label_distributions:
            all_labels.update(dist.keys())

        # Compute per-label variance across clients
        label_variances = []
        for label in all_labels:
            counts = [dist.get(label, 0) for dist in client_label_distributions]
            if sum(counts) > 0:
                normalized = np.array(counts) / max(sum(counts), 1)
                label_variances.append(np.var(normalized))

        avg_variance = np.mean(label_variances) if label_variances else 0

        # Compute quantity skew
        total_samples = [sum(dist.values()) for dist in client_label_distributions]
        quantity_cv = np.std(total_samples) / max(np.mean(total_samples), 1)

        # Classify
        if avg_variance < 0.01 and quantity_cv < 0.2:
            return DataCharacteristic.IID
        elif avg_variance > 0.3:
            return DataCharacteristic.NON_IID_EXTREME
        elif avg_variance > 0.1:
            return DataCharacteristic.NON_IID_LABEL_SKEW
        elif quantity_cv > 0.5:
            return DataCharacteristic.NON_IID_QUANTITY_SKEW
        else:
            return DataCharacteristic.NON_IID_FEATURE_SKEW

    def recommend_algorithms(
        self,
        task_type: TaskType,
        data_characteristic: DataCharacteristic,
        num_clients: int,
        has_byzantine_risk: bool = False
    ) -> List[AggregationAlgorithm]:
        """Recommend FL algorithms based on task analysis."""
        recommendations = self._algorithm_recommendations.get(
            (task_type, data_characteristic),
            [AggregationAlgorithm.FEDAVG, AggregationAlgorithm.FEDPROX]
        )

        # Add Byzantine-resilient if needed
        if has_byzantine_risk:
            recommendations = [AggregationAlgorithm.KRUM, AggregationAlgorithm.TRIMMED_MEAN] + recommendations

        # For many clients, prefer communication-efficient
        if num_clients > 50:
            if AggregationAlgorithm.FEDNOVA not in recommendations:
                recommendations.append(AggregationAlgorithm.FEDNOVA)

        return recommendations[:5]  # Top 5


class BanditSelector:
    """Multi-armed bandit for algorithm selection."""

    def __init__(
        self,
        algorithms: List[AggregationAlgorithm],
        method: str = "ucb"
    ):
        self.algorithms = algorithms
        self.method = method
        self.counts = {alg: 0 for alg in algorithms}
        self.rewards = {alg: 0.0 for alg in algorithms}
        self.total_pulls = 0

    def select(self, exploration_weight: float = 2.0) -> AggregationAlgorithm:
        """Select next algorithm to try."""
        self.total_pulls += 1

        # Ensure each algorithm is tried at least once
        for alg in self.algorithms:
            if self.counts[alg] == 0:
                return alg

        if self.method == "ucb":
            return self._ucb_select(exploration_weight)
        elif self.method == "thompson":
            return self._thompson_select()
        else:  # epsilon_greedy
            return self._epsilon_greedy_select()

    def update(self, algorithm: AggregationAlgorithm, reward: float):
        """Update algorithm statistics with observed reward."""
        self.counts[algorithm] += 1
        # Running average
        n = self.counts[algorithm]
        self.rewards[algorithm] = self.rewards[algorithm] * (n-1)/n + reward/n

    def _ucb_select(self, c: float) -> AggregationAlgorithm:
        """Upper Confidence Bound selection."""
        ucb_values = {}
        for alg in self.algorithms:
            if self.counts[alg] == 0:
                ucb_values[alg] = float('inf')
            else:
                exploitation = self.rewards[alg]
                exploration = c * np.sqrt(np.log(self.total_pulls) / self.counts[alg])
                ucb_values[alg] = exploitation + exploration

        return max(ucb_values, key=ucb_values.get)

    def _thompson_select(self) -> AggregationAlgorithm:
        """Thompson Sampling selection."""
        samples = {}
        for alg in self.algorithms:
            # Beta distribution for bounded rewards
            alpha = self.rewards[alg] * self.counts[alg] + 1
            beta = (1 - self.rewards[alg]) * self.counts[alg] + 1
            samples[alg] = np.random.beta(max(alpha, 1), max(beta, 1))

        return max(samples, key=samples.get)

    def _epsilon_greedy_select(self, epsilon: float = 0.1) -> AggregationAlgorithm:
        """Epsilon-greedy selection."""
        if np.random.random() < epsilon:
            return np.random.choice(self.algorithms)
        else:
            return max(self.rewards, key=self.rewards.get)

    def get_best_algorithm(self) -> AggregationAlgorithm:
        """Return the algorithm with highest average reward."""
        return max(self.rewards, key=self.rewards.get)


class FederatedModelSelector:
    """
    Automatic FL algorithm selection.

    Uses task analysis and multi-armed bandits to select
    the best aggregation algorithm.
    """

    def __init__(self, config: ModelSelectionConfig):
        self.config = config
        self.task_analyzer = TaskAnalyzer()
        self.bandit: Optional[BanditSelector] = None
        self.performance_history: Dict[AggregationAlgorithm, AlgorithmPerformance] = {}
        self._current_algorithm: Optional[AggregationAlgorithm] = None
        self._exploration_complete = False
        self._round_counter = 0

    def initialize(
        self,
        task_type: TaskType,
        client_distributions: List[Dict[int, int]],
        num_clients: int,
        has_byzantine_risk: bool = False
    ):
        """Initialize selector based on task analysis."""
        # Analyze data
        data_char = self.task_analyzer.analyze_data_distribution(client_distributions)

        # Get recommendations
        recommended = self.task_analyzer.recommend_algorithms(
            task_type, data_char, num_clients, has_byzantine_risk
        )

        # Filter by config
        algorithms = [
            alg for alg in recommended
            if alg in self.config.algorithms_to_try
        ]

        if not algorithms:
            algorithms = list(self.config.algorithms_to_try)

        # Initialize bandit
        self.bandit = BanditSelector(algorithms, self.config.bandit_algorithm)

        # Initialize performance tracking
        for alg in algorithms:
            self.performance_history[alg] = AlgorithmPerformance(
                algorithm=alg,
                rounds_evaluated=0,
                total_loss=0.0,
                total_accuracy=0.0,
                convergence_rates=[],
                communication_costs=[],
                client_variances=[]
            )

        logger.info(f"Initialized model selector with algorithms: {[a.value for a in algorithms]}")
        logger.info(f"Data characteristic: {data_char.name}")

    def select_algorithm(self) -> AggregationAlgorithm:
        """Select the algorithm for current round."""
        if self.bandit is None:
            return self.config.algorithms_to_try[0]

        self._round_counter += 1

        # Exploration phase
        total_exploration = len(self.bandit.algorithms) * self.config.exploration_rounds
        if self._round_counter <= total_exploration:
            self._current_algorithm = self.bandit.select()
            return self._current_algorithm

        # Exploitation phase
        if not self._exploration_complete:
            self._exploration_complete = True
            logger.info("Exploration complete, switching to exploitation")

        # Mostly exploit best, occasionally explore
        if np.random.random() < (1 - self.config.exploitation_ratio):
            self._current_algorithm = self.bandit.select()
        else:
            self._current_algorithm = self.bandit.get_best_algorithm()

        return self._current_algorithm

    def record_performance(
        self,
        algorithm: AggregationAlgorithm,
        loss: float,
        accuracy: float,
        convergence_rate: float = 0.0,
        communication_cost: float = 0.0,
        client_variance: float = 0.0
    ):
        """Record performance metrics for an algorithm."""
        if algorithm not in self.performance_history:
            return

        perf = self.performance_history[algorithm]
        perf.rounds_evaluated += 1
        perf.total_loss += loss
        perf.total_accuracy += accuracy
        perf.convergence_rates.append(convergence_rate)
        perf.communication_costs.append(communication_cost)
        perf.client_variances.append(client_variance)

        # Update bandit
        if self.bandit:
            # Compute reward based on criterion
            reward = self._compute_reward(loss, accuracy, convergence_rate, client_variance)
            self.bandit.update(algorithm, reward)

    def _compute_reward(
        self,
        loss: float,
        accuracy: float,
        convergence_rate: float,
        client_variance: float
    ) -> float:
        """Compute reward based on selection criterion."""
        if self.config.criterion == SelectionCriterion.ACCURACY:
            return accuracy
        elif self.config.criterion == SelectionCriterion.CONVERGENCE_SPEED:
            return convergence_rate
        elif self.config.criterion == SelectionCriterion.FAIRNESS:
            # Lower variance is better
            return 1.0 / (1.0 + client_variance)
        else:
            # Combined metric
            return 0.5 * accuracy + 0.3 * (1 - loss) + 0.2 * (1 / (1 + client_variance))

    def get_best_algorithm(self) -> AggregationAlgorithm:
        """Get the best performing algorithm."""
        if self.bandit:
            return self.bandit.get_best_algorithm()
        return self.config.algorithms_to_try[0]

    def get_selection_report(self) -> Dict[str, Any]:
        """Generate a report on algorithm selection."""
        report = {
            "total_rounds": self._round_counter,
            "exploration_complete": self._exploration_complete,
            "current_algorithm": self._current_algorithm.value if self._current_algorithm else None,
            "best_algorithm": self.get_best_algorithm().value,
            "algorithm_stats": {}
        }

        for alg, perf in self.performance_history.items():
            report["algorithm_stats"][alg.value] = {
                "rounds_evaluated": perf.rounds_evaluated,
                "avg_loss": perf.avg_loss,
                "avg_accuracy": perf.avg_accuracy,
                "avg_convergence": perf.avg_convergence
            }

        return report


# =============================================================================
# ADAPTIVE AGGREGATION
# =============================================================================

class AggregationStrategy(ABC):
    """Abstract base class for aggregation strategies."""

    @abstractmethod
    def aggregate(
        self,
        client_updates: List[Dict[str, np.ndarray]],
        client_weights: List[float],
        global_weights: Dict[str, np.ndarray],
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """Aggregate client updates."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name."""
        pass


class FedAvgStrategy(AggregationStrategy):
    """FedAvg aggregation strategy."""

    def aggregate(
        self,
        client_updates: List[Dict[str, np.ndarray]],
        client_weights: List[float],
        global_weights: Dict[str, np.ndarray],
        **kwargs
    ) -> Dict[str, np.ndarray]:
        weights = np.array(client_weights)
        weights = weights / weights.sum()

        aggregated = {}
        for name in client_updates[0]:
            stacked = np.stack([u[name] for u in client_updates], axis=0)
            aggregated[name] = np.sum(
                stacked * weights[:, np.newaxis, ...].reshape(
                    [len(weights)] + [1] * (stacked.ndim - 1)
                ),
                axis=0
            )

        return aggregated

    @property
    def name(self) -> str:
        return "FedAvg"


class FedProxStrategy(AggregationStrategy):
    """FedProx aggregation strategy."""

    def __init__(self, mu: float = 0.1):
        self.mu = mu

    def aggregate(
        self,
        client_updates: List[Dict[str, np.ndarray]],
        client_weights: List[float],
        global_weights: Dict[str, np.ndarray],
        **kwargs
    ) -> Dict[str, np.ndarray]:
        # FedProx: same aggregation as FedAvg, but clients use proximal term
        # Here we just do weighted averaging
        weights = np.array(client_weights)
        weights = weights / weights.sum()

        aggregated = {}
        for name in client_updates[0]:
            stacked = np.stack([u[name] for u in client_updates], axis=0)
            aggregated[name] = np.sum(
                stacked * weights[:, np.newaxis, ...].reshape(
                    [len(weights)] + [1] * (stacked.ndim - 1)
                ),
                axis=0
            )

        return aggregated

    @property
    def name(self) -> str:
        return f"FedProx(μ={self.mu})"


class SCAFFOLDStrategy(AggregationStrategy):
    """SCAFFOLD aggregation strategy with control variates."""

    def __init__(self, lr: float = 0.1):
        self.lr = lr
        self.server_control: Optional[Dict[str, np.ndarray]] = None
        self.client_controls: Dict[int, Dict[str, np.ndarray]] = {}

    def aggregate(
        self,
        client_updates: List[Dict[str, np.ndarray]],
        client_weights: List[float],
        global_weights: Dict[str, np.ndarray],
        client_ids: Optional[List[int]] = None,
        **kwargs
    ) -> Dict[str, np.ndarray]:
        # Initialize controls if needed
        if self.server_control is None:
            self.server_control = {name: np.zeros_like(w) for name, w in global_weights.items()}

        weights = np.array(client_weights)
        weights = weights / weights.sum()

        # Aggregate updates
        aggregated = {}
        delta_controls = {}

        for name in client_updates[0]:
            stacked = np.stack([u[name] for u in client_updates], axis=0)
            aggregated[name] = np.sum(
                stacked * weights[:, np.newaxis, ...].reshape(
                    [len(weights)] + [1] * (stacked.ndim - 1)
                ),
                axis=0
            )

            # Update control variate
            delta_controls[name] = aggregated[name] - global_weights[name]

        # Update server control
        for name in self.server_control:
            if name in delta_controls:
                self.server_control[name] += self.lr * delta_controls[name] / len(client_updates)

        return aggregated

    @property
    def name(self) -> str:
        return f"SCAFFOLD(lr={self.lr})"


class FedAdamStrategy(AggregationStrategy):
    """FedAdam adaptive aggregation strategy."""

    def __init__(
        self,
        lr: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.999,
        tau: float = 1e-3
    ):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.tau = tau
        self.m: Optional[Dict[str, np.ndarray]] = None  # First moment
        self.v: Optional[Dict[str, np.ndarray]] = None  # Second moment
        self.t = 0

    def aggregate(
        self,
        client_updates: List[Dict[str, np.ndarray]],
        client_weights: List[float],
        global_weights: Dict[str, np.ndarray],
        **kwargs
    ) -> Dict[str, np.ndarray]:
        self.t += 1

        # Initialize moments if needed
        if self.m is None:
            self.m = {name: np.zeros_like(w) for name, w in global_weights.items()}
            self.v = {name: np.zeros_like(w) for name, w in global_weights.items()}

        weights = np.array(client_weights)
        weights = weights / weights.sum()

        # Compute pseudo-gradient (difference from global)
        delta = {}
        for name in client_updates[0]:
            stacked = np.stack([u[name] for u in client_updates], axis=0)
            avg_update = np.sum(
                stacked * weights[:, np.newaxis, ...].reshape(
                    [len(weights)] + [1] * (stacked.ndim - 1)
                ),
                axis=0
            )
            delta[name] = avg_update - global_weights[name]

        # Adam update
        aggregated = {}
        for name in global_weights:
            if name in delta:
                # Update biased first moment estimate
                self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * delta[name]
                # Update biased second raw moment estimate
                self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * (delta[name] ** 2)

                # Bias correction
                m_hat = self.m[name] / (1 - self.beta1 ** self.t)
                v_hat = self.v[name] / (1 - self.beta2 ** self.t)

                # Update
                aggregated[name] = global_weights[name] + self.lr * m_hat / (np.sqrt(v_hat) + self.tau)
            else:
                aggregated[name] = global_weights[name]

        return aggregated

    @property
    def name(self) -> str:
        return f"FedAdam(lr={self.lr})"


class FedYogiStrategy(AggregationStrategy):
    """FedYogi adaptive aggregation strategy."""

    def __init__(
        self,
        lr: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.999,
        tau: float = 1e-3
    ):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.tau = tau
        self.m: Optional[Dict[str, np.ndarray]] = None
        self.v: Optional[Dict[str, np.ndarray]] = None
        self.t = 0

    def aggregate(
        self,
        client_updates: List[Dict[str, np.ndarray]],
        client_weights: List[float],
        global_weights: Dict[str, np.ndarray],
        **kwargs
    ) -> Dict[str, np.ndarray]:
        self.t += 1

        if self.m is None:
            self.m = {name: np.zeros_like(w) for name, w in global_weights.items()}
            self.v = {name: np.zeros_like(w) for name, w in global_weights.items()}

        weights = np.array(client_weights)
        weights = weights / weights.sum()

        delta = {}
        for name in client_updates[0]:
            stacked = np.stack([u[name] for u in client_updates], axis=0)
            avg_update = np.sum(
                stacked * weights[:, np.newaxis, ...].reshape(
                    [len(weights)] + [1] * (stacked.ndim - 1)
                ),
                axis=0
            )
            delta[name] = avg_update - global_weights[name]

        aggregated = {}
        for name in global_weights:
            if name in delta:
                self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * delta[name]
                # Yogi update for v
                self.v[name] = self.v[name] - (1 - self.beta2) * np.sign(
                    self.v[name] - delta[name] ** 2
                ) * (delta[name] ** 2)

                m_hat = self.m[name] / (1 - self.beta1 ** self.t)
                v_hat = self.v[name] / (1 - self.beta2 ** self.t)

                aggregated[name] = global_weights[name] + self.lr * m_hat / (np.sqrt(np.abs(v_hat)) + self.tau)
            else:
                aggregated[name] = global_weights[name]

        return aggregated

    @property
    def name(self) -> str:
        return f"FedYogi(lr={self.lr})"


class KrumStrategy(AggregationStrategy):
    """Krum Byzantine-resilient aggregation."""

    def __init__(self, num_byzantine: int = 0):
        self.num_byzantine = num_byzantine

    def aggregate(
        self,
        client_updates: List[Dict[str, np.ndarray]],
        client_weights: List[float],
        global_weights: Dict[str, np.ndarray],
        **kwargs
    ) -> Dict[str, np.ndarray]:
        n = len(client_updates)
        f = self.num_byzantine

        if n <= 2 * f + 2:
            # Not enough clients, fall back to FedAvg
            strategy = FedAvgStrategy()
            return strategy.aggregate(client_updates, client_weights, global_weights)

        # Flatten updates for distance computation
        flattened = []
        for update in client_updates:
            flat = np.concatenate([v.flatten() for v in update.values()])
            flattened.append(flat)
        flattened = np.array(flattened)

        # Compute pairwise distances
        scores = []
        for i in range(n):
            distances = np.linalg.norm(flattened - flattened[i], axis=1)
            # Sum of n-f-2 smallest distances (excluding self)
            sorted_distances = np.sort(distances)
            score = np.sum(sorted_distances[1:n-f-1])  # Exclude self (0) and f+1 largest
            scores.append(score)

        # Select client with minimum score
        selected_idx = np.argmin(scores)

        return client_updates[selected_idx]

    @property
    def name(self) -> str:
        return f"Krum(f={self.num_byzantine})"


class TrimmedMeanStrategy(AggregationStrategy):
    """Trimmed Mean Byzantine-resilient aggregation."""

    def __init__(self, beta: float = 0.1):
        self.beta = beta  # Fraction to trim from each end

    def aggregate(
        self,
        client_updates: List[Dict[str, np.ndarray]],
        client_weights: List[float],
        global_weights: Dict[str, np.ndarray],
        **kwargs
    ) -> Dict[str, np.ndarray]:
        n = len(client_updates)
        trim_count = int(n * self.beta)

        aggregated = {}
        for name in client_updates[0]:
            stacked = np.stack([u[name] for u in client_updates], axis=0)

            # Sort along client axis
            sorted_vals = np.sort(stacked, axis=0)

            # Trim extremes
            if trim_count > 0 and n > 2 * trim_count:
                trimmed = sorted_vals[trim_count:-trim_count]
            else:
                trimmed = sorted_vals

            # Mean of remaining
            aggregated[name] = np.mean(trimmed, axis=0)

        return aggregated

    @property
    def name(self) -> str:
        return f"TrimmedMean(β={self.beta})"


class MedianStrategy(AggregationStrategy):
    """Coordinate-wise median aggregation."""

    def aggregate(
        self,
        client_updates: List[Dict[str, np.ndarray]],
        client_weights: List[float],
        global_weights: Dict[str, np.ndarray],
        **kwargs
    ) -> Dict[str, np.ndarray]:
        aggregated = {}
        for name in client_updates[0]:
            stacked = np.stack([u[name] for u in client_updates], axis=0)
            aggregated[name] = np.median(stacked, axis=0)

        return aggregated

    @property
    def name(self) -> str:
        return "Median"


class FedNovaStrategy(AggregationStrategy):
    """FedNova normalized averaging strategy."""

    def aggregate(
        self,
        client_updates: List[Dict[str, np.ndarray]],
        client_weights: List[float],
        global_weights: Dict[str, np.ndarray],
        local_steps: Optional[List[int]] = None,
        **kwargs
    ) -> Dict[str, np.ndarray]:
        if local_steps is None:
            local_steps = [1] * len(client_updates)

        # Normalize by local steps
        total_steps = sum(local_steps)
        weights = np.array([s / total_steps for s in local_steps])

        # Also incorporate sample weights
        sample_weights = np.array(client_weights)
        combined_weights = weights * sample_weights
        combined_weights = combined_weights / combined_weights.sum()

        aggregated = {}
        for name in client_updates[0]:
            stacked = np.stack([u[name] for u in client_updates], axis=0)
            aggregated[name] = np.sum(
                stacked * combined_weights[:, np.newaxis, ...].reshape(
                    [len(combined_weights)] + [1] * (stacked.ndim - 1)
                ),
                axis=0
            )

        return aggregated

    @property
    def name(self) -> str:
        return "FedNova"


class AdaptiveAggregator:
    """
    Adaptive Aggregation: Dynamic switching between FL algorithms.

    Monitors runtime metrics and switches algorithms when beneficial.
    """

    def __init__(self, config: AdaptiveAggregationConfig):
        self.config = config
        self.strategies = self._create_strategies()
        self.current_algorithm = config.initial_algorithm
        self.current_strategy = self.strategies[self.current_algorithm]

        self.metrics_history: List[AggregationMetrics] = []
        self.global_weights: Optional[Dict[str, np.ndarray]] = None
        self._round_counter = 0
        self._last_switch_round = 0
        self._algorithm_performance: Dict[AggregationAlgorithm, List[float]] = {
            alg: [] for alg in self.strategies
        }

    def _create_strategies(self) -> Dict[AggregationAlgorithm, AggregationStrategy]:
        """Create all available aggregation strategies."""
        return {
            AggregationAlgorithm.FEDAVG: FedAvgStrategy(),
            AggregationAlgorithm.FEDPROX: FedProxStrategy(self.config.fedprox_mu),
            AggregationAlgorithm.SCAFFOLD: SCAFFOLDStrategy(self.config.scaffold_lr),
            AggregationAlgorithm.FEDADAM: FedAdamStrategy(
                self.config.fedadam_lr,
                self.config.fedadam_beta1,
                self.config.fedadam_beta2,
                self.config.fedadam_tau
            ),
            AggregationAlgorithm.FEDYOGI: FedYogiStrategy(self.config.fedyogi_lr),
            AggregationAlgorithm.KRUM: KrumStrategy(self.config.krum_num_byzantine),
            AggregationAlgorithm.TRIMMED_MEAN: TrimmedMeanStrategy(self.config.trimmed_mean_beta),
            AggregationAlgorithm.MEDIAN: MedianStrategy(),
            AggregationAlgorithm.FEDNOVA: FedNovaStrategy(),
        }

    def aggregate(
        self,
        client_updates: List[Dict[str, np.ndarray]],
        client_weights: List[float],
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """Aggregate client updates using current strategy."""
        self._round_counter += 1

        if self.global_weights is None:
            self.global_weights = client_updates[0].copy()

        # Perform aggregation
        import time
        start_time = time.time()

        aggregated = self.current_strategy.aggregate(
            client_updates,
            client_weights,
            self.global_weights,
            **kwargs
        )

        computation_time = time.time() - start_time

        # Update global weights
        self.global_weights = aggregated

        return aggregated

    def record_metrics(
        self,
        loss: float,
        accuracy: float,
        client_losses: Optional[List[float]] = None
    ):
        """Record round metrics for adaptive switching."""
        # Compute variance if client losses provided
        client_variance = np.var(client_losses) if client_losses else 0.0

        # Compute convergence rate (loss reduction)
        convergence_rate = 0.0
        if self.metrics_history:
            prev_loss = self.metrics_history[-1].loss
            if prev_loss > 0:
                convergence_rate = (prev_loss - loss) / prev_loss

        metrics = AggregationMetrics(
            round_number=self._round_counter,
            algorithm=self.current_algorithm,
            loss=loss,
            accuracy=accuracy,
            client_variance=client_variance,
            convergence_rate=convergence_rate,
            communication_cost=0.0,  # Would need to track actual bytes
            computation_time=0.0,
            num_participants=len(client_losses) if client_losses else 0
        )

        self.metrics_history.append(metrics)
        self._algorithm_performance[self.current_algorithm].append(accuracy)

        # Check if we should switch
        self._maybe_switch_algorithm()

    def _maybe_switch_algorithm(self):
        """Check if algorithm switch would be beneficial."""
        # Cooldown check
        if self._round_counter - self._last_switch_round < self.config.cooldown_rounds:
            return

        # Need enough history to evaluate
        if len(self.metrics_history) < self.config.evaluation_window:
            return

        # Get recent metrics
        recent = self.metrics_history[-self.config.evaluation_window:]
        current_score = self._compute_algorithm_score(recent)

        # Try other algorithms (based on their historical performance)
        best_alternative = None
        best_score = current_score

        for alg, performances in self._algorithm_performance.items():
            if alg == self.current_algorithm:
                continue

            if len(performances) >= 3:
                # Use recent performance as estimate
                alt_score = np.mean(performances[-5:])
                if alt_score > best_score + self.config.switch_threshold:
                    best_score = alt_score
                    best_alternative = alg

        # Heuristic switches based on metrics
        if best_alternative is None:
            best_alternative = self._heuristic_switch(recent)

        if best_alternative and best_alternative != self.current_algorithm:
            self._switch_to(best_alternative)

    def _compute_algorithm_score(self, metrics: List[AggregationMetrics]) -> float:
        """Compute weighted score for algorithm performance."""
        if not metrics:
            return 0.0

        weights = self.config.metrics_weights

        avg_loss = np.mean([m.loss for m in metrics])
        avg_accuracy = np.mean([m.accuracy for m in metrics])
        avg_variance = np.mean([m.client_variance for m in metrics])
        avg_convergence = np.mean([m.convergence_rate for m in metrics])

        score = (
            weights.get("accuracy", 0.3) * avg_accuracy +
            weights.get("loss", 0.4) * (1 - min(avg_loss, 1)) +
            weights.get("client_variance", 0.2) * (1 / (1 + avg_variance)) +
            weights.get("convergence_rate", 0.1) * max(avg_convergence, 0)
        )

        return score

    def _heuristic_switch(
        self,
        recent_metrics: List[AggregationMetrics]
    ) -> Optional[AggregationAlgorithm]:
        """Use heuristics to suggest algorithm switch."""
        if not recent_metrics:
            return None

        avg_variance = np.mean([m.client_variance for m in recent_metrics])
        avg_convergence = np.mean([m.convergence_rate for m in recent_metrics])

        current = self.current_algorithm

        # High variance -> try variance reduction methods
        if avg_variance > 0.1:
            if current == AggregationAlgorithm.FEDAVG:
                return AggregationAlgorithm.SCAFFOLD
            elif current == AggregationAlgorithm.FEDPROX:
                return AggregationAlgorithm.SCAFFOLD

        # Slow convergence -> try adaptive methods
        if avg_convergence < 0.01:
            if current in [AggregationAlgorithm.FEDAVG, AggregationAlgorithm.FEDPROX]:
                return AggregationAlgorithm.FEDADAM
            elif current == AggregationAlgorithm.SCAFFOLD:
                return AggregationAlgorithm.FEDYOGI

        # Good convergence with FedAdam -> try simpler FedAvg
        if avg_convergence > 0.05 and current == AggregationAlgorithm.FEDADAM:
            return AggregationAlgorithm.FEDAVG

        return None

    def _switch_to(self, algorithm: AggregationAlgorithm):
        """Switch to a different aggregation algorithm."""
        logger.info(
            f"Switching from {self.current_algorithm.value} to {algorithm.value} "
            f"at round {self._round_counter}"
        )

        self.current_algorithm = algorithm
        self.current_strategy = self.strategies[algorithm]
        self._last_switch_round = self._round_counter

    def force_switch(self, algorithm: AggregationAlgorithm):
        """Force switch to a specific algorithm."""
        self._switch_to(algorithm)

    def get_status(self) -> Dict[str, Any]:
        """Get current aggregator status."""
        recent_metrics = self.metrics_history[-10:] if self.metrics_history else []

        return {
            "current_algorithm": self.current_algorithm.value,
            "round": self._round_counter,
            "last_switch_round": self._last_switch_round,
            "recent_avg_accuracy": np.mean([m.accuracy for m in recent_metrics]) if recent_metrics else None,
            "recent_avg_loss": np.mean([m.loss for m in recent_metrics]) if recent_metrics else None,
            "algorithm_usage": {
                alg.value: len(perfs) for alg, perfs in self._algorithm_performance.items()
            }
        }


# =============================================================================
# CROSS-SILO MANAGER
# =============================================================================

class CrossSiloManager:
    """
    Unified manager for cross-silo FL enhancements.

    Combines Multi-Model Federation, Model Selection, and Adaptive Aggregation.
    """

    def __init__(
        self,
        ensemble_config: Optional[EnsembleConfig] = None,
        selection_config: Optional[ModelSelectionConfig] = None,
        aggregation_config: Optional[AdaptiveAggregationConfig] = None
    ):
        self.ensemble_config = ensemble_config or EnsembleConfig()
        self.selection_config = selection_config or ModelSelectionConfig()
        self.aggregation_config = aggregation_config or AdaptiveAggregationConfig()

        self.ensemble = FederatedEnsemble(self.ensemble_config)
        self.selector = FederatedModelSelector(self.selection_config)
        self.aggregator = AdaptiveAggregator(self.aggregation_config)

        self._initialized = False

    def initialize(
        self,
        task_type: TaskType,
        client_distributions: List[Dict[int, int]],
        num_clients: int,
        initial_weights: Dict[str, np.ndarray],
        has_byzantine_risk: bool = False
    ):
        """Initialize all components."""
        # Initialize selector
        self.selector.initialize(
            task_type,
            client_distributions,
            num_clients,
            has_byzantine_risk
        )

        # Initialize aggregator with selected algorithm
        initial_alg = self.selector.select_algorithm()
        self.aggregator.force_switch(initial_alg)
        self.aggregator.global_weights = initial_weights

        # Initialize ensemble
        self.ensemble.add_model(
            initial_weights,
            initial_alg,
            {"role": "primary", "round": 0}
        )

        self._initialized = True
        logger.info(f"CrossSiloManager initialized with algorithm {initial_alg.value}")

    def aggregate_round(
        self,
        client_updates: List[Dict[str, np.ndarray]],
        client_weights: List[float],
        client_losses: Optional[List[float]] = None,
        round_loss: Optional[float] = None,
        round_accuracy: Optional[float] = None,
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """
        Perform one round of aggregation with all enhancements.

        Returns:
            Aggregated model weights
        """
        if not self._initialized:
            raise RuntimeError("Manager not initialized. Call initialize() first.")

        # Select algorithm for this round
        algorithm = self.selector.select_algorithm()

        # Switch aggregator if needed
        if algorithm != self.aggregator.current_algorithm:
            self.aggregator.force_switch(algorithm)

        # Perform aggregation
        aggregated = self.aggregator.aggregate(
            client_updates,
            client_weights,
            **kwargs
        )

        # Record metrics
        if round_loss is not None and round_accuracy is not None:
            self.aggregator.record_metrics(
                round_loss,
                round_accuracy,
                client_losses
            )

            self.selector.record_performance(
                algorithm,
                round_loss,
                round_accuracy,
                client_variance=np.var(client_losses) if client_losses else 0.0
            )

        # Update ensemble
        self.ensemble.aggregate_round(client_updates, client_weights)

        return aggregated

    def get_model_weights(self) -> Dict[str, np.ndarray]:
        """Get current global model weights."""
        return self.aggregator.global_weights

    def get_comprehensive_report(self) -> Dict[str, Any]:
        """Get comprehensive report of all components."""
        return {
            "ensemble": self.ensemble.get_ensemble_info(),
            "selector": self.selector.get_selection_report(),
            "aggregator": self.aggregator.get_status()
        }


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_ensemble_config(
    strategy: str = "weighted_voting",
    num_models: int = 3,
    diversity_weight: float = 0.3,
    **kwargs
) -> EnsembleConfig:
    """Create ensemble configuration."""
    strategy_map = {
        "weighted_voting": EnsembleStrategy.WEIGHTED_VOTING,
        "majority_voting": EnsembleStrategy.MAJORITY_VOTING,
        "stacking": EnsembleStrategy.STACKING,
        "bagging": EnsembleStrategy.BAGGING,
        "mixture_of_experts": EnsembleStrategy.MIXTURE_OF_EXPERTS,
    }

    return EnsembleConfig(
        strategy=strategy_map.get(strategy, EnsembleStrategy.WEIGHTED_VOTING),
        num_models=num_models,
        diversity_weight=diversity_weight,
        **kwargs
    )


def create_selection_config(
    criterion: str = "accuracy",
    algorithms: Optional[List[str]] = None,
    exploration_rounds: int = 5,
    **kwargs
) -> ModelSelectionConfig:
    """Create model selection configuration."""
    criterion_map = {
        "accuracy": SelectionCriterion.ACCURACY,
        "convergence": SelectionCriterion.CONVERGENCE_SPEED,
        "communication": SelectionCriterion.COMMUNICATION_EFFICIENCY,
        "privacy": SelectionCriterion.PRIVACY,
        "robustness": SelectionCriterion.ROBUSTNESS,
        "fairness": SelectionCriterion.FAIRNESS,
    }

    alg_map = {
        "fedavg": AggregationAlgorithm.FEDAVG,
        "fedprox": AggregationAlgorithm.FEDPROX,
        "scaffold": AggregationAlgorithm.SCAFFOLD,
        "fedadam": AggregationAlgorithm.FEDADAM,
        "fedyogi": AggregationAlgorithm.FEDYOGI,
        "krum": AggregationAlgorithm.KRUM,
        "trimmed_mean": AggregationAlgorithm.TRIMMED_MEAN,
        "fednova": AggregationAlgorithm.FEDNOVA,
    }

    algs = [
        alg_map.get(a, AggregationAlgorithm.FEDAVG)
        for a in (algorithms or ["fedavg", "fedprox", "scaffold"])
    ]

    return ModelSelectionConfig(
        criterion=criterion_map.get(criterion, SelectionCriterion.ACCURACY),
        algorithms_to_try=algs,
        exploration_rounds=exploration_rounds,
        **kwargs
    )


def create_adaptive_config(
    initial_algorithm: str = "fedavg",
    evaluation_window: int = 5,
    switch_threshold: float = 0.05,
    **kwargs
) -> AdaptiveAggregationConfig:
    """Create adaptive aggregation configuration."""
    alg_map = {
        "fedavg": AggregationAlgorithm.FEDAVG,
        "fedprox": AggregationAlgorithm.FEDPROX,
        "scaffold": AggregationAlgorithm.SCAFFOLD,
        "fedadam": AggregationAlgorithm.FEDADAM,
        "fedyogi": AggregationAlgorithm.FEDYOGI,
        "krum": AggregationAlgorithm.KRUM,
        "trimmed_mean": AggregationAlgorithm.TRIMMED_MEAN,
    }

    return AdaptiveAggregationConfig(
        initial_algorithm=alg_map.get(initial_algorithm, AggregationAlgorithm.FEDAVG),
        evaluation_window=evaluation_window,
        switch_threshold=switch_threshold,
        **kwargs
    )


def create_cross_silo_manager(
    ensemble_strategy: str = "weighted_voting",
    selection_criterion: str = "accuracy",
    initial_algorithm: str = "fedavg",
    **kwargs
) -> CrossSiloManager:
    """Create a fully configured CrossSiloManager."""
    ensemble_config = create_ensemble_config(strategy=ensemble_strategy, **kwargs)
    selection_config = create_selection_config(criterion=selection_criterion, **kwargs)
    aggregation_config = create_adaptive_config(initial_algorithm=initial_algorithm, **kwargs)

    return CrossSiloManager(
        ensemble_config=ensemble_config,
        selection_config=selection_config,
        aggregation_config=aggregation_config
    )

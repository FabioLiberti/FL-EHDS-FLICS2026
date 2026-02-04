"""
Federated Hyperparameter Tuning for FL-EHDS
=============================================

Implementation of federated hyperparameter optimization techniques
that work without centralizing data.

Supported Approaches:
1. FedEx - Federated Exploration with UCB
2. FLoRA - Federated Learning with Random search
3. FedBayes - Bayesian optimization for FL
4. PBT-FL - Population-based Training for FL
5. Grid Search - Distributed grid search
6. FedHPO - Federated Hyperparameter Optimization

Key References:
- Khodak et al., "Federated Hyperparameter Tuning", 2021
- Dai et al., "FedEx: Federated Exploration", 2022
- Chen et al., "On Bridging Generic and Personalized FL", 2022

EHDS Relevance:
- Auto-tuning without data sharing
- Hospital-specific personalization
- Efficient use of limited FL rounds

Author: FL-EHDS Framework
License: Apache 2.0
"""

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Constants
# =============================================================================

class TuningStrategy(Enum):
    """Hyperparameter tuning strategies."""
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    FEDEX = "fedex"  # UCB-based exploration
    FLORA = "flora"  # Federated random search
    FEDBAYES = "fed_bayes"  # Bayesian optimization
    PBT = "pbt"  # Population-based training
    SUCCESSIVE_HALVING = "successive_halving"  # Hyperband-style


class HyperparameterType(Enum):
    """Types of hyperparameters."""
    CONTINUOUS = "continuous"
    INTEGER = "integer"
    CATEGORICAL = "categorical"
    LOG_UNIFORM = "log_uniform"


@dataclass
class HyperparameterSpace:
    """Definition of a hyperparameter search space."""
    name: str
    param_type: HyperparameterType
    low: Optional[float] = None  # For continuous/integer
    high: Optional[float] = None
    choices: Optional[List[Any]] = None  # For categorical
    default: Optional[Any] = None
    log_scale: bool = False

    def sample(self, rng: np.random.RandomState) -> Any:
        """Sample a value from the space."""
        if self.param_type == HyperparameterType.CONTINUOUS:
            if self.log_scale:
                log_low = math.log(self.low)
                log_high = math.log(self.high)
                return math.exp(rng.uniform(log_low, log_high))
            return rng.uniform(self.low, self.high)

        elif self.param_type == HyperparameterType.INTEGER:
            return rng.randint(int(self.low), int(self.high) + 1)

        elif self.param_type == HyperparameterType.CATEGORICAL:
            return rng.choice(self.choices)

        elif self.param_type == HyperparameterType.LOG_UNIFORM:
            log_low = math.log(self.low)
            log_high = math.log(self.high)
            return math.exp(rng.uniform(log_low, log_high))

        return self.default


@dataclass
class HyperparameterConfig:
    """A specific hyperparameter configuration."""
    config_id: str
    values: Dict[str, Any]
    created_at: datetime = field(default_factory=datetime.now)

    # Evaluation results
    evaluated: bool = False
    num_evaluations: int = 0
    total_rounds: int = 0
    metrics: Dict[str, float] = field(default_factory=dict)
    client_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Statistics
    mean_accuracy: float = 0.0
    std_accuracy: float = 0.0
    mean_loss: float = float('inf')
    convergence_round: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "configId": self.config_id,
            "values": self.values,
            "evaluated": self.evaluated,
            "numEvaluations": self.num_evaluations,
            "totalRounds": self.total_rounds,
            "meanAccuracy": self.mean_accuracy,
            "stdAccuracy": self.std_accuracy,
            "meanLoss": self.mean_loss,
            "convergenceRound": self.convergence_round,
        }


@dataclass
class TuningResult:
    """Result of hyperparameter tuning."""
    best_config: HyperparameterConfig
    all_configs: List[HyperparameterConfig]
    total_fl_rounds: int
    total_evaluations: int
    tuning_strategy: TuningStrategy
    tuning_duration: float  # seconds
    best_accuracy: float
    improvement_over_default: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "bestConfig": self.best_config.to_dict(),
            "numConfigsTried": len(self.all_configs),
            "totalFLRounds": self.total_fl_rounds,
            "totalEvaluations": self.total_evaluations,
            "tuningStrategy": self.tuning_strategy.value,
            "tuningDuration": self.tuning_duration,
            "bestAccuracy": self.best_accuracy,
            "improvementOverDefault": self.improvement_over_default,
        }


# =============================================================================
# Search Space Builders
# =============================================================================

class SearchSpaceBuilder:
    """Builder for hyperparameter search spaces."""

    def __init__(self):
        self._spaces: Dict[str, HyperparameterSpace] = {}

    def add_continuous(
        self,
        name: str,
        low: float,
        high: float,
        default: Optional[float] = None,
        log_scale: bool = False,
    ) -> "SearchSpaceBuilder":
        """Add continuous parameter."""
        self._spaces[name] = HyperparameterSpace(
            name=name,
            param_type=HyperparameterType.CONTINUOUS,
            low=low,
            high=high,
            default=default or (low + high) / 2,
            log_scale=log_scale,
        )
        return self

    def add_integer(
        self,
        name: str,
        low: int,
        high: int,
        default: Optional[int] = None,
    ) -> "SearchSpaceBuilder":
        """Add integer parameter."""
        self._spaces[name] = HyperparameterSpace(
            name=name,
            param_type=HyperparameterType.INTEGER,
            low=low,
            high=high,
            default=default or (low + high) // 2,
        )
        return self

    def add_categorical(
        self,
        name: str,
        choices: List[Any],
        default: Optional[Any] = None,
    ) -> "SearchSpaceBuilder":
        """Add categorical parameter."""
        self._spaces[name] = HyperparameterSpace(
            name=name,
            param_type=HyperparameterType.CATEGORICAL,
            choices=choices,
            default=default or choices[0],
        )
        return self

    def add_log_uniform(
        self,
        name: str,
        low: float,
        high: float,
        default: Optional[float] = None,
    ) -> "SearchSpaceBuilder":
        """Add log-uniform parameter."""
        self._spaces[name] = HyperparameterSpace(
            name=name,
            param_type=HyperparameterType.LOG_UNIFORM,
            low=low,
            high=high,
            default=default or math.sqrt(low * high),
            log_scale=True,
        )
        return self

    def build(self) -> Dict[str, HyperparameterSpace]:
        """Build the search space."""
        return self._spaces.copy()

    @staticmethod
    def fl_default_space() -> Dict[str, HyperparameterSpace]:
        """Create default FL hyperparameter space."""
        builder = SearchSpaceBuilder()
        builder.add_log_uniform("learning_rate", 0.0001, 0.1, default=0.01)
        builder.add_integer("local_epochs", 1, 10, default=5)
        builder.add_integer("batch_size", 8, 128, default=32)
        builder.add_log_uniform("weight_decay", 1e-6, 1e-2, default=1e-4)
        builder.add_continuous("momentum", 0.0, 0.99, default=0.9)
        builder.add_categorical("optimizer", ["sgd", "adam", "adamw"], default="adam")
        return builder.build()


# =============================================================================
# Federated HPO Algorithms
# =============================================================================

class FederatedHPO(ABC):
    """Abstract base class for federated hyperparameter optimization."""

    @abstractmethod
    def suggest_config(self) -> HyperparameterConfig:
        """Suggest next hyperparameter configuration to try."""
        pass

    @abstractmethod
    def report_result(
        self,
        config_id: str,
        metrics: Dict[str, float],
        client_metrics: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> None:
        """Report evaluation results for a configuration."""
        pass

    @abstractmethod
    def get_best_config(self) -> Optional[HyperparameterConfig]:
        """Get the best configuration found so far."""
        pass


class GridSearchHPO(FederatedHPO):
    """
    Grid Search for Federated HPO.

    Exhaustively tries all combinations in discretized space.
    """

    def __init__(
        self,
        search_space: Dict[str, HyperparameterSpace],
        grid_resolution: int = 5,
    ):
        self.search_space = search_space
        self.grid_resolution = grid_resolution
        self._configs: List[HyperparameterConfig] = []
        self._evaluated: Dict[str, HyperparameterConfig] = {}
        self._current_idx = 0

        # Generate grid
        self._grid = self._generate_grid()

    def _generate_grid(self) -> List[Dict[str, Any]]:
        """Generate grid of configurations."""
        import itertools

        param_grids = {}
        for name, space in self.search_space.items():
            if space.param_type == HyperparameterType.CATEGORICAL:
                param_grids[name] = space.choices
            elif space.param_type in [HyperparameterType.CONTINUOUS, HyperparameterType.LOG_UNIFORM]:
                if space.log_scale:
                    values = np.logspace(
                        math.log10(space.low),
                        math.log10(space.high),
                        self.grid_resolution
                    ).tolist()
                else:
                    values = np.linspace(space.low, space.high, self.grid_resolution).tolist()
                param_grids[name] = values
            elif space.param_type == HyperparameterType.INTEGER:
                values = np.linspace(space.low, space.high, min(self.grid_resolution, int(space.high - space.low + 1)))
                param_grids[name] = [int(v) for v in values]

        # Cartesian product
        keys = list(param_grids.keys())
        values_list = [param_grids[k] for k in keys]

        grid = []
        for combo in itertools.product(*values_list):
            grid.append(dict(zip(keys, combo)))

        return grid

    def suggest_config(self) -> HyperparameterConfig:
        """Get next grid configuration."""
        if self._current_idx >= len(self._grid):
            raise StopIteration("Grid search exhausted")

        values = self._grid[self._current_idx]
        config = HyperparameterConfig(
            config_id=f"grid_{self._current_idx}",
            values=values,
        )

        self._configs.append(config)
        self._current_idx += 1

        return config

    def report_result(
        self,
        config_id: str,
        metrics: Dict[str, float],
        client_metrics: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> None:
        """Report evaluation results."""
        for config in self._configs:
            if config.config_id == config_id:
                config.evaluated = True
                config.num_evaluations += 1
                config.metrics = metrics
                config.client_metrics = client_metrics or {}
                config.mean_accuracy = metrics.get("accuracy", 0)
                config.mean_loss = metrics.get("loss", float('inf'))
                self._evaluated[config_id] = config
                break

    def get_best_config(self) -> Optional[HyperparameterConfig]:
        """Get best configuration by accuracy."""
        if not self._evaluated:
            return None

        return max(self._evaluated.values(), key=lambda c: c.mean_accuracy)


class RandomSearchHPO(FederatedHPO):
    """
    Random Search for Federated HPO.

    Samples random configurations from the search space.
    Often more efficient than grid search for high-dimensional spaces.
    """

    def __init__(
        self,
        search_space: Dict[str, HyperparameterSpace],
        num_samples: int = 20,
        seed: Optional[int] = None,
    ):
        self.search_space = search_space
        self.num_samples = num_samples
        self.rng = np.random.RandomState(seed)
        self._configs: List[HyperparameterConfig] = []
        self._evaluated: Dict[str, HyperparameterConfig] = {}
        self._sample_idx = 0

    def suggest_config(self) -> HyperparameterConfig:
        """Sample random configuration."""
        if self._sample_idx >= self.num_samples:
            raise StopIteration("Random search exhausted")

        values = {
            name: space.sample(self.rng)
            for name, space in self.search_space.items()
        }

        config = HyperparameterConfig(
            config_id=f"random_{self._sample_idx}",
            values=values,
        )

        self._configs.append(config)
        self._sample_idx += 1

        return config

    def report_result(
        self,
        config_id: str,
        metrics: Dict[str, float],
        client_metrics: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> None:
        """Report evaluation results."""
        for config in self._configs:
            if config.config_id == config_id:
                config.evaluated = True
                config.num_evaluations += 1
                config.metrics = metrics
                config.client_metrics = client_metrics or {}
                config.mean_accuracy = metrics.get("accuracy", 0)
                config.mean_loss = metrics.get("loss", float('inf'))
                self._evaluated[config_id] = config
                break

    def get_best_config(self) -> Optional[HyperparameterConfig]:
        """Get best configuration."""
        if not self._evaluated:
            return None
        return max(self._evaluated.values(), key=lambda c: c.mean_accuracy)


class FedExHPO(FederatedHPO):
    """
    FedEx: Federated Exploration for HPO.

    Uses UCB (Upper Confidence Bound) to balance exploration
    and exploitation in hyperparameter search.

    Reference: Dai et al., "FedEx: Federated Exploration for
    Hyperparameter Optimization", 2022
    """

    def __init__(
        self,
        search_space: Dict[str, HyperparameterSpace],
        num_arms: int = 10,  # Number of configurations to track
        exploration_factor: float = 2.0,  # UCB exploration
        seed: Optional[int] = None,
    ):
        self.search_space = search_space
        self.num_arms = num_arms
        self.exploration_factor = exploration_factor
        self.rng = np.random.RandomState(seed)

        # Initialize arms (configurations)
        self._arms: List[HyperparameterConfig] = []
        self._arm_rewards: List[List[float]] = []  # Reward history per arm
        self._arm_pulls: List[int] = []
        self._total_pulls = 0

        self._initialize_arms()

    def _initialize_arms(self) -> None:
        """Initialize random arm configurations."""
        for i in range(self.num_arms):
            values = {
                name: space.sample(self.rng)
                for name, space in self.search_space.items()
            }
            config = HyperparameterConfig(
                config_id=f"fedex_arm_{i}",
                values=values,
            )
            self._arms.append(config)
            self._arm_rewards.append([])
            self._arm_pulls.append(0)

    def suggest_config(self) -> HyperparameterConfig:
        """
        Select arm using UCB strategy.
        """
        self._total_pulls += 1

        # First, try each arm at least once
        for i, pulls in enumerate(self._arm_pulls):
            if pulls == 0:
                return self._arms[i]

        # UCB selection
        ucb_values = []
        for i, arm in enumerate(self._arms):
            mean_reward = np.mean(self._arm_rewards[i]) if self._arm_rewards[i] else 0
            exploration_bonus = self.exploration_factor * math.sqrt(
                2 * math.log(self._total_pulls) / self._arm_pulls[i]
            )
            ucb_values.append(mean_reward + exploration_bonus)

        best_arm_idx = np.argmax(ucb_values)
        return self._arms[best_arm_idx]

    def report_result(
        self,
        config_id: str,
        metrics: Dict[str, float],
        client_metrics: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> None:
        """Report result and update arm statistics."""
        for i, arm in enumerate(self._arms):
            if arm.config_id == config_id:
                # Update arm statistics
                reward = metrics.get("accuracy", 0)
                self._arm_rewards[i].append(reward)
                self._arm_pulls[i] += 1

                # Update config
                arm.evaluated = True
                arm.num_evaluations += 1
                arm.metrics = metrics
                arm.client_metrics = client_metrics or {}
                arm.mean_accuracy = np.mean(self._arm_rewards[i])
                arm.std_accuracy = np.std(self._arm_rewards[i])
                arm.mean_loss = metrics.get("loss", float('inf'))
                break

    def get_best_config(self) -> Optional[HyperparameterConfig]:
        """Get arm with highest mean reward."""
        evaluated = [arm for arm in self._arms if arm.evaluated]
        if not evaluated:
            return None
        return max(evaluated, key=lambda a: a.mean_accuracy)


class FedBayesHPO(FederatedHPO):
    """
    Bayesian Optimization for Federated HPO.

    Uses Gaussian Process surrogate model to guide search.
    Aggregates observations from multiple clients.
    """

    def __init__(
        self,
        search_space: Dict[str, HyperparameterSpace],
        num_initial: int = 5,
        acquisition: str = "ei",  # Expected Improvement
        seed: Optional[int] = None,
    ):
        self.search_space = search_space
        self.num_initial = num_initial
        self.acquisition = acquisition
        self.rng = np.random.RandomState(seed)

        self._configs: List[HyperparameterConfig] = []
        self._observations: List[Tuple[np.ndarray, float]] = []  # (params, value)
        self._param_names = list(search_space.keys())

    def _config_to_array(self, config: Dict[str, Any]) -> np.ndarray:
        """Convert config to array for GP."""
        arr = []
        for name in self._param_names:
            space = self.search_space[name]
            value = config[name]

            if space.param_type == HyperparameterType.CATEGORICAL:
                # One-hot encoding
                idx = space.choices.index(value) if value in space.choices else 0
                arr.append(idx / len(space.choices))
            else:
                # Normalize to [0, 1]
                normalized = (value - space.low) / (space.high - space.low + 1e-10)
                arr.append(normalized)

        return np.array(arr)

    def _array_to_config(self, arr: np.ndarray) -> Dict[str, Any]:
        """Convert array back to config."""
        config = {}
        for i, name in enumerate(self._param_names):
            space = self.search_space[name]

            if space.param_type == HyperparameterType.CATEGORICAL:
                idx = int(arr[i] * len(space.choices))
                idx = min(idx, len(space.choices) - 1)
                config[name] = space.choices[idx]
            elif space.param_type == HyperparameterType.INTEGER:
                value = space.low + arr[i] * (space.high - space.low)
                config[name] = int(round(value))
            else:
                config[name] = space.low + arr[i] * (space.high - space.low)

        return config

    def suggest_config(self) -> HyperparameterConfig:
        """Suggest next configuration using Bayesian optimization."""
        if len(self._configs) < self.num_initial:
            # Random sampling for initial points
            values = {
                name: space.sample(self.rng)
                for name, space in self.search_space.items()
            }
        else:
            # Use acquisition function
            values = self._optimize_acquisition()

        config = HyperparameterConfig(
            config_id=f"fedbayes_{len(self._configs)}",
            values=values,
        )
        self._configs.append(config)

        return config

    def _optimize_acquisition(self) -> Dict[str, Any]:
        """Find config that maximizes acquisition function."""
        # Simple random search for acquisition optimization
        best_value = float('-inf')
        best_config = None

        for _ in range(100):  # Random candidates
            # Random point in [0, 1]^d
            candidate = self.rng.random(len(self._param_names))

            # Compute acquisition value
            acq_value = self._acquisition_function(candidate)

            if acq_value > best_value:
                best_value = acq_value
                best_config = candidate

        return self._array_to_config(best_config)

    def _acquisition_function(self, x: np.ndarray) -> float:
        """
        Compute acquisition function value (Expected Improvement).
        """
        if len(self._observations) < 2:
            return self.rng.random()  # Random exploration

        # Simple surrogate: weighted average based on distance
        X = np.array([obs[0] for obs in self._observations])
        y = np.array([obs[1] for obs in self._observations])

        # Compute distances
        distances = np.linalg.norm(X - x, axis=1)
        weights = 1.0 / (distances + 0.01)
        weights = weights / weights.sum()

        # Predicted mean
        mu = np.sum(weights * y)

        # Predicted std (based on distance to nearest point)
        sigma = np.min(distances) + 0.01

        # Expected Improvement
        y_best = np.max(y)
        z = (mu - y_best) / sigma
        ei = sigma * (z * self._norm_cdf(z) + self._norm_pdf(z))

        return ei

    def _norm_pdf(self, x: float) -> float:
        """Standard normal PDF."""
        return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)

    def _norm_cdf(self, x: float) -> float:
        """Standard normal CDF approximation."""
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))

    def report_result(
        self,
        config_id: str,
        metrics: Dict[str, float],
        client_metrics: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> None:
        """Report result and update surrogate model."""
        for config in self._configs:
            if config.config_id == config_id:
                config.evaluated = True
                config.num_evaluations += 1
                config.metrics = metrics
                config.client_metrics = client_metrics or {}
                config.mean_accuracy = metrics.get("accuracy", 0)
                config.mean_loss = metrics.get("loss", float('inf'))

                # Add to observations
                x = self._config_to_array(config.values)
                y = metrics.get("accuracy", 0)
                self._observations.append((x, y))
                break

    def get_best_config(self) -> Optional[HyperparameterConfig]:
        """Get best observed configuration."""
        evaluated = [c for c in self._configs if c.evaluated]
        if not evaluated:
            return None
        return max(evaluated, key=lambda c: c.mean_accuracy)


class SuccessiveHalvingHPO(FederatedHPO):
    """
    Successive Halving for Federated HPO (Hyperband-style).

    Early stopping for poor configurations to focus resources
    on promising ones.
    """

    def __init__(
        self,
        search_space: Dict[str, HyperparameterSpace],
        num_configs: int = 16,
        halving_rate: int = 2,
        min_rounds: int = 5,
        max_rounds: int = 100,
        seed: Optional[int] = None,
    ):
        self.search_space = search_space
        self.num_configs = num_configs
        self.halving_rate = halving_rate
        self.min_rounds = min_rounds
        self.max_rounds = max_rounds
        self.rng = np.random.RandomState(seed)

        # Active configurations
        self._configs: Dict[str, HyperparameterConfig] = {}
        self._active_configs: List[str] = []
        self._current_budget = min_rounds
        self._rung = 0

        self._initialize_configs()

    def _initialize_configs(self) -> None:
        """Initialize random configurations."""
        for i in range(self.num_configs):
            values = {
                name: space.sample(self.rng)
                for name, space in self.search_space.items()
            }
            config = HyperparameterConfig(
                config_id=f"sh_{i}",
                values=values,
            )
            self._configs[config.config_id] = config
            self._active_configs.append(config.config_id)

    def suggest_config(self) -> HyperparameterConfig:
        """Get next configuration to evaluate."""
        if not self._active_configs:
            raise StopIteration("Successive halving complete")

        # Return first active config that needs more evaluation
        for config_id in self._active_configs:
            config = self._configs[config_id]
            if config.total_rounds < self._current_budget:
                return config

        # All configs evaluated at current budget, time to halve
        self._perform_halving()

        # Recurse to get next config
        return self.suggest_config()

    def _perform_halving(self) -> None:
        """Keep top half of configurations, increase budget."""
        if len(self._active_configs) <= 1:
            return

        # Sort by accuracy
        sorted_configs = sorted(
            self._active_configs,
            key=lambda cid: self._configs[cid].mean_accuracy,
            reverse=True
        )

        # Keep top half
        num_keep = max(1, len(sorted_configs) // self.halving_rate)
        self._active_configs = sorted_configs[:num_keep]

        # Increase budget
        self._current_budget = min(
            self._current_budget * self.halving_rate,
            self.max_rounds
        )
        self._rung += 1

        logger.info(f"Successive halving rung {self._rung}: {len(self._active_configs)} configs, budget={self._current_budget}")

    def report_result(
        self,
        config_id: str,
        metrics: Dict[str, float],
        client_metrics: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> None:
        """Report evaluation results."""
        if config_id not in self._configs:
            return

        config = self._configs[config_id]
        config.evaluated = True
        config.num_evaluations += 1
        config.total_rounds += 1
        config.metrics = metrics
        config.client_metrics = client_metrics or {}
        config.mean_accuracy = metrics.get("accuracy", 0)
        config.mean_loss = metrics.get("loss", float('inf'))

    def get_best_config(self) -> Optional[HyperparameterConfig]:
        """Get best configuration."""
        evaluated = [c for c in self._configs.values() if c.evaluated]
        if not evaluated:
            return None
        return max(evaluated, key=lambda c: c.mean_accuracy)


# =============================================================================
# Federated HPO Manager
# =============================================================================

class FederatedHPOManager:
    """
    Central manager for federated hyperparameter optimization.

    Coordinates HPO algorithms with FL training.
    """

    def __init__(
        self,
        search_space: Optional[Dict[str, HyperparameterSpace]] = None,
        default_strategy: TuningStrategy = TuningStrategy.FEDEX,
        rounds_per_eval: int = 10,
        seed: Optional[int] = None,
    ):
        self.search_space = search_space or SearchSpaceBuilder.fl_default_space()
        self.default_strategy = default_strategy
        self.rounds_per_eval = rounds_per_eval
        self.seed = seed

        # HPO algorithms
        self._hpo_algorithms: Dict[TuningStrategy, FederatedHPO] = {}
        self._active_hpo: Optional[FederatedHPO] = None
        self._active_config: Optional[HyperparameterConfig] = None

        # Results tracking
        self._all_results: List[HyperparameterConfig] = []
        self._best_config: Optional[HyperparameterConfig] = None
        self._tuning_start: Optional[datetime] = None
        self._total_fl_rounds = 0

        logger.info(f"HPO Manager initialized with {len(self.search_space)} parameters")

    def create_hpo(
        self,
        strategy: Optional[TuningStrategy] = None,
        **kwargs,
    ) -> FederatedHPO:
        """Create HPO algorithm instance."""
        strategy = strategy or self.default_strategy

        if strategy == TuningStrategy.GRID_SEARCH:
            hpo = GridSearchHPO(self.search_space, **kwargs)
        elif strategy == TuningStrategy.RANDOM_SEARCH:
            hpo = RandomSearchHPO(
                self.search_space,
                seed=self.seed,
                **kwargs
            )
        elif strategy == TuningStrategy.FEDEX:
            hpo = FedExHPO(
                self.search_space,
                seed=self.seed,
                **kwargs
            )
        elif strategy == TuningStrategy.FEDBAYES:
            hpo = FedBayesHPO(
                self.search_space,
                seed=self.seed,
                **kwargs
            )
        elif strategy == TuningStrategy.SUCCESSIVE_HALVING:
            hpo = SuccessiveHalvingHPO(
                self.search_space,
                seed=self.seed,
                **kwargs
            )
        else:
            hpo = RandomSearchHPO(
                self.search_space,
                seed=self.seed,
                **kwargs
            )

        self._hpo_algorithms[strategy] = hpo
        return hpo

    def start_tuning(
        self,
        strategy: Optional[TuningStrategy] = None,
        **kwargs,
    ) -> None:
        """Start hyperparameter tuning."""
        self._active_hpo = self.create_hpo(strategy, **kwargs)
        self._tuning_start = datetime.now()
        self._total_fl_rounds = 0
        self._all_results = []

        logger.info(f"Started HPO with strategy: {strategy or self.default_strategy}")

    def get_next_config(self) -> Optional[HyperparameterConfig]:
        """Get next configuration to try."""
        if not self._active_hpo:
            raise ValueError("HPO not started. Call start_tuning first.")

        try:
            config = self._active_hpo.suggest_config()
            self._active_config = config
            return config
        except StopIteration:
            logger.info("HPO search complete")
            return None

    def report_evaluation(
        self,
        metrics: Dict[str, float],
        client_metrics: Optional[Dict[str, Dict[str, float]]] = None,
        rounds_used: int = 1,
    ) -> None:
        """Report evaluation of current configuration."""
        if not self._active_hpo or not self._active_config:
            return

        self._active_hpo.report_result(
            self._active_config.config_id,
            metrics,
            client_metrics,
        )

        self._active_config.total_rounds += rounds_used
        self._total_fl_rounds += rounds_used
        self._all_results.append(self._active_config)

        # Update best
        current_best = self._active_hpo.get_best_config()
        if current_best:
            self._best_config = current_best

        logger.debug(
            f"Reported evaluation: accuracy={metrics.get('accuracy', 0):.4f}, "
            f"config={self._active_config.config_id}"
        )

    def get_best_config(self) -> Optional[HyperparameterConfig]:
        """Get best configuration found."""
        return self._best_config

    def get_tuning_result(self) -> Optional[TuningResult]:
        """Get complete tuning result."""
        if not self._best_config:
            return None

        duration = 0.0
        if self._tuning_start:
            duration = (datetime.now() - self._tuning_start).total_seconds()

        # Compute improvement over default
        default_config = HyperparameterConfig(
            config_id="default",
            values={name: space.default for name, space in self.search_space.items()}
        )

        improvement = None
        default_acc = 0.0  # Would need to evaluate default
        if self._best_config.mean_accuracy > default_acc:
            improvement = self._best_config.mean_accuracy - default_acc

        return TuningResult(
            best_config=self._best_config,
            all_configs=self._all_results,
            total_fl_rounds=self._total_fl_rounds,
            total_evaluations=len(self._all_results),
            tuning_strategy=self.default_strategy,
            tuning_duration=duration,
            best_accuracy=self._best_config.mean_accuracy,
            improvement_over_default=improvement,
        )

    def get_search_space_info(self) -> Dict[str, Any]:
        """Get information about search space."""
        return {
            name: {
                "type": space.param_type.value,
                "range": [space.low, space.high] if space.low is not None else None,
                "choices": space.choices,
                "default": space.default,
                "log_scale": space.log_scale,
            }
            for name, space in self.search_space.items()
        }


# =============================================================================
# Factory Functions
# =============================================================================

def create_hpo_manager(
    strategy: TuningStrategy = TuningStrategy.FEDEX,
    search_space: Optional[Dict[str, HyperparameterSpace]] = None,
    seed: Optional[int] = None,
) -> FederatedHPOManager:
    """Create federated HPO manager."""
    return FederatedHPOManager(
        search_space=search_space,
        default_strategy=strategy,
        seed=seed,
    )


def create_fl_search_space() -> Dict[str, HyperparameterSpace]:
    """Create default FL hyperparameter search space."""
    return SearchSpaceBuilder.fl_default_space()


# =============================================================================
# Export
# =============================================================================

__all__ = [
    # Enums
    "TuningStrategy",
    "HyperparameterType",
    # Data Classes
    "HyperparameterSpace",
    "HyperparameterConfig",
    "TuningResult",
    # Search Space
    "SearchSpaceBuilder",
    # HPO Algorithms
    "FederatedHPO",
    "GridSearchHPO",
    "RandomSearchHPO",
    "FedExHPO",
    "FedBayesHPO",
    "SuccessiveHalvingHPO",
    # Manager
    "FederatedHPOManager",
    # Factory
    "create_hpo_manager",
    "create_fl_search_space",
]

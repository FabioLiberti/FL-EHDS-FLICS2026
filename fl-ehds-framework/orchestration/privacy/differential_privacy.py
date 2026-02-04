"""
Differential Privacy Module
===========================
Implements differential privacy mechanisms for gradient protection
in federated learning.

References:
    - Dwork & Roth, "The Algorithmic Foundations of Differential Privacy", 2014
    - Abadi et al., "Deep Learning with Differential Privacy", CCS 2016
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import structlog

from core.exceptions import PrivacyBudgetExceededError

logger = structlog.get_logger(__name__)


@dataclass
class PrivacySpent:
    """Record of privacy budget spent."""

    epsilon: float
    delta: float
    round_number: int
    mechanism: str
    num_queries: int = 1


class PrivacyAccountant:
    """
    Tracks privacy budget consumption across training rounds.

    Supports multiple accounting methods:
    - Simple composition
    - Advanced composition
    - Rényi Differential Privacy (RDP)
    """

    def __init__(
        self,
        total_epsilon: float,
        total_delta: float,
        accountant_type: str = "rdp",
    ):
        """
        Initialize privacy accountant.

        Args:
            total_epsilon: Total privacy budget (ε).
            total_delta: Failure probability (δ).
            accountant_type: Accounting method ('simple', 'advanced', 'rdp').
        """
        self.total_epsilon = total_epsilon
        self.total_delta = total_delta
        self.accountant_type = accountant_type

        self._spent_history: List[PrivacySpent] = []
        self._current_epsilon = 0.0
        self._current_delta = 0.0

    def spend(
        self,
        epsilon: float,
        delta: float,
        round_number: int,
        mechanism: str = "gaussian",
    ) -> None:
        """
        Record privacy expenditure.

        Args:
            epsilon: Epsilon spent.
            delta: Delta spent.
            round_number: Current round.
            mechanism: Mechanism used.

        Raises:
            PrivacyBudgetExceededError: If budget is exceeded.
        """
        # Check if spending would exceed budget
        new_epsilon, new_delta = self._compose(epsilon, delta)

        if new_epsilon > self.total_epsilon or new_delta > self.total_delta:
            raise PrivacyBudgetExceededError(
                current_epsilon=new_epsilon,
                max_epsilon=self.total_epsilon,
                round_number=round_number,
            )

        # Record expenditure
        self._spent_history.append(
            PrivacySpent(
                epsilon=epsilon,
                delta=delta,
                round_number=round_number,
                mechanism=mechanism,
            )
        )

        self._current_epsilon = new_epsilon
        self._current_delta = new_delta

        logger.info(
            "Privacy budget spent",
            round=round_number,
            spent_epsilon=epsilon,
            total_epsilon=self._current_epsilon,
            remaining=self.total_epsilon - self._current_epsilon,
        )

    def _compose(self, epsilon: float, delta: float) -> Tuple[float, float]:
        """
        Compose privacy guarantee with current state.

        Args:
            epsilon: New epsilon to add.
            delta: New delta to add.

        Returns:
            Composed (epsilon, delta).
        """
        if self.accountant_type == "simple":
            # Simple composition: ε_total = sum(ε_i)
            return self._current_epsilon + epsilon, self._current_delta + delta

        elif self.accountant_type == "advanced":
            # Advanced composition for k queries
            k = len(self._spent_history) + 1
            # ε_total = sqrt(2k * ln(1/δ')) * ε + k * ε * (e^ε - 1)
            delta_prime = self.total_delta / 2
            epsilon_composed = (
                np.sqrt(2 * k * np.log(1 / delta_prime)) * epsilon
                + k * epsilon * (np.exp(epsilon) - 1)
            )
            return epsilon_composed, self._current_delta + delta

        elif self.accountant_type == "rdp":
            # Simplified RDP composition
            # In practice, would use proper RDP accountant
            return self._current_epsilon + epsilon, self._current_delta + delta

        return self._current_epsilon + epsilon, self._current_delta + delta

    def get_remaining_budget(self) -> Tuple[float, float]:
        """Get remaining privacy budget."""
        return (
            self.total_epsilon - self._current_epsilon,
            self.total_delta - self._current_delta,
        )

    def get_spent_budget(self) -> Tuple[float, float]:
        """Get total spent privacy budget."""
        return self._current_epsilon, self._current_delta

    def can_spend(self, epsilon: float, delta: float) -> bool:
        """Check if expenditure is within budget."""
        new_epsilon, new_delta = self._compose(epsilon, delta)
        return new_epsilon <= self.total_epsilon and new_delta <= self.total_delta

    def get_history(self) -> List[PrivacySpent]:
        """Get history of privacy expenditures."""
        return self._spent_history.copy()

    def reset(self) -> None:
        """Reset accountant state."""
        self._spent_history.clear()
        self._current_epsilon = 0.0
        self._current_delta = 0.0


class DifferentialPrivacy:
    """
    Differential privacy mechanism for gradient protection.

    Implements Gaussian mechanism for (ε, δ)-DP guarantees.
    """

    def __init__(
        self,
        epsilon: float = 1.0,
        delta: float = 1e-5,
        max_grad_norm: float = 1.0,
        mechanism: str = "gaussian",
        accountant: Optional[PrivacyAccountant] = None,
    ):
        """
        Initialize differential privacy mechanism.

        Args:
            epsilon: Privacy budget per round.
            delta: Failure probability.
            max_grad_norm: Maximum gradient L2 norm (sensitivity).
            mechanism: Noise mechanism ('gaussian', 'laplace').
            accountant: Optional privacy accountant.
        """
        self.epsilon = epsilon
        self.delta = delta
        self.max_grad_norm = max_grad_norm
        self.mechanism = mechanism
        self.accountant = accountant

        # Compute noise scale
        self._noise_scale = self._compute_noise_scale()

    def _compute_noise_scale(self) -> float:
        """
        Compute noise scale for the specified privacy parameters.

        For Gaussian mechanism with (ε, δ)-DP:
        σ = sensitivity * sqrt(2 * ln(1.25/δ)) / ε
        """
        if self.mechanism == "gaussian":
            sensitivity = self.max_grad_norm
            return sensitivity * np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon

        elif self.mechanism == "laplace":
            sensitivity = self.max_grad_norm
            return sensitivity / self.epsilon

        raise ValueError(f"Unknown mechanism: {self.mechanism}")

    def add_noise(
        self,
        gradients: Dict[str, Any],
        round_number: int,
    ) -> Dict[str, Any]:
        """
        Add calibrated noise to gradients.

        Args:
            gradients: Dictionary of gradient tensors.
            round_number: Current training round.

        Returns:
            Noised gradients dictionary.

        Raises:
            PrivacyBudgetExceededError: If budget is exceeded.
        """
        # Check and record privacy expenditure
        if self.accountant:
            self.accountant.spend(
                epsilon=self.epsilon,
                delta=self.delta,
                round_number=round_number,
                mechanism=self.mechanism,
            )

        noised_gradients = {}

        for key, grad in gradients.items():
            grad_array = np.array(grad)

            if self.mechanism == "gaussian":
                noise = np.random.normal(0, self._noise_scale, grad_array.shape)
            elif self.mechanism == "laplace":
                noise = np.random.laplace(0, self._noise_scale, grad_array.shape)
            else:
                raise ValueError(f"Unknown mechanism: {self.mechanism}")

            noised_gradients[key] = grad_array + noise

        logger.debug(
            "DP noise added to gradients",
            round=round_number,
            mechanism=self.mechanism,
            noise_scale=self._noise_scale,
            num_params=len(gradients),
        )

        return noised_gradients

    def add_noise_to_aggregated(
        self,
        aggregated_gradients: Dict[str, Any],
        num_clients: int,
        round_number: int,
    ) -> Dict[str, Any]:
        """
        Add noise to aggregated gradients (central DP).

        Noise is scaled by 1/num_clients for amplification by sampling.

        Args:
            aggregated_gradients: Aggregated gradient dictionary.
            num_clients: Number of participating clients.
            round_number: Current training round.

        Returns:
            Noised aggregated gradients.
        """
        # Scale noise by number of clients (subsampling amplification)
        effective_noise_scale = self._noise_scale / num_clients

        noised = {}
        for key, grad in aggregated_gradients.items():
            grad_array = np.array(grad)
            noise = np.random.normal(0, effective_noise_scale, grad_array.shape)
            noised[key] = grad_array + noise

        logger.debug(
            "DP noise added to aggregated gradients",
            round=round_number,
            num_clients=num_clients,
            effective_noise_scale=effective_noise_scale,
        )

        return noised

    def get_noise_scale(self) -> float:
        """Get current noise scale."""
        return self._noise_scale

    def compute_privacy_spent(self, num_rounds: int) -> Tuple[float, float]:
        """
        Compute total privacy spent for given number of rounds.

        Args:
            num_rounds: Number of training rounds.

        Returns:
            (epsilon, delta) total spent.
        """
        # Simple composition
        total_epsilon = self.epsilon * num_rounds
        total_delta = self.delta * num_rounds
        return total_epsilon, total_delta

    def recommend_rounds(
        self,
        target_epsilon: float,
        target_delta: float,
    ) -> int:
        """
        Recommend maximum rounds for target privacy budget.

        Args:
            target_epsilon: Target total epsilon.
            target_delta: Target total delta.

        Returns:
            Maximum recommended training rounds.
        """
        # Simple composition
        max_by_epsilon = int(target_epsilon / self.epsilon)
        max_by_delta = int(target_delta / self.delta)
        return min(max_by_epsilon, max_by_delta)

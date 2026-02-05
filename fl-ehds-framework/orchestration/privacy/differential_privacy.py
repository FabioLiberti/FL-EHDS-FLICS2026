"""
Differential Privacy Module
===========================
Implements differential privacy mechanisms for gradient protection
in federated learning.

References:
    - Dwork & Roth, "The Algorithmic Foundations of Differential Privacy", 2014
    - Abadi et al., "Deep Learning with Differential Privacy", CCS 2016
    - Mironov, "Rényi Differential Privacy", CSF 2017
    - Balle et al., "Hypothesis Testing Interpretations and Rényi DP", AISTATS 2020
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from scipy import special
import structlog

from core.exceptions import PrivacyBudgetExceededError

logger = structlog.get_logger(__name__)

# Default RDP orders for accounting (following Google DP library)
DEFAULT_RDP_ORDERS = [1.5, 1.75, 2, 2.5, 3, 4, 5, 6, 8, 16, 32, 64, 256, 512, 1024]


def compute_rdp_gaussian(sigma: float, alpha: float) -> float:
    """
    Compute RDP guarantee for Gaussian mechanism.

    For the Gaussian mechanism with noise scale σ and sensitivity 1:
    ρ(α) = α / (2σ²)

    Reference: Mironov, "Rényi Differential Privacy", CSF 2017, Proposition 3

    Args:
        sigma: Noise multiplier (noise_scale / sensitivity).
        alpha: RDP order (α > 1).

    Returns:
        RDP guarantee at order α.
    """
    if alpha <= 1:
        raise ValueError("RDP order α must be > 1")
    if sigma <= 0:
        raise ValueError("Sigma must be positive")

    return alpha / (2 * sigma ** 2)


def compute_rdp_gaussian_subsampled(
    sigma: float,
    alpha: float,
    sampling_rate: float,
) -> float:
    """
    Compute RDP guarantee for subsampled Gaussian mechanism.

    Uses the analytical moments accountant from Abadi et al. (2016)
    with improvements from Wang et al. (2019).

    Args:
        sigma: Noise multiplier.
        alpha: RDP order.
        sampling_rate: Probability of including each record (q).

    Returns:
        RDP guarantee at order α under subsampling.
    """
    if alpha <= 1:
        raise ValueError("RDP order α must be > 1")
    if not 0 < sampling_rate <= 1:
        raise ValueError("Sampling rate must be in (0, 1]")

    if sampling_rate == 1.0:
        return compute_rdp_gaussian(sigma, alpha)

    q = sampling_rate

    # For integer orders, use closed-form expression
    if alpha == int(alpha):
        alpha = int(alpha)
        # Use log-sum-exp for numerical stability
        log_terms = []
        for k in range(alpha + 1):
            log_binom = special.gammaln(alpha + 1) - special.gammaln(k + 1) - special.gammaln(alpha - k + 1)
            log_term = (
                log_binom
                + k * np.log(q)
                + (alpha - k) * np.log(1 - q)
                + k * (k - 1) / (2 * sigma ** 2)
            )
            log_terms.append(log_term)

        log_rdp = special.logsumexp(log_terms) / (alpha - 1)
        return max(0, log_rdp)

    # For non-integer orders, use upper bound from Mironov
    # This is a slight approximation but tight for small q
    rdp_no_subsample = compute_rdp_gaussian(sigma, alpha)

    # Privacy amplification by subsampling (approximate)
    if q < 0.1:
        # For small q, RDP ≈ 2 * q² * α / σ²
        return 2 * (q ** 2) * alpha / (sigma ** 2)
    else:
        # Conservative: use full RDP scaled by q
        return min(rdp_no_subsample, q * rdp_no_subsample + (1 - q) * 0)


def rdp_to_eps_delta(rdp_values: List[float], orders: List[float], delta: float) -> Tuple[float, float]:
    """
    Convert RDP guarantees to (ε, δ)-DP.

    Uses the conversion theorem:
    ε = ρ(α) + log(1/δ) / (α - 1) - log(α) / (α - 1)

    Minimizes over all orders to get the tightest bound.

    Reference: Balle et al., "Hypothesis Testing Interpretations", AISTATS 2020

    Args:
        rdp_values: RDP values at each order.
        orders: Corresponding RDP orders.
        delta: Target δ.

    Returns:
        Tuple of (optimal_epsilon, optimal_order).
    """
    if len(rdp_values) != len(orders):
        raise ValueError("Length mismatch between RDP values and orders")

    if delta <= 0:
        raise ValueError("Delta must be positive")

    best_epsilon = float('inf')
    best_order = orders[0]

    for rdp, alpha in zip(rdp_values, orders):
        if alpha <= 1:
            continue

        # Conversion formula
        eps = rdp + np.log(1 / delta) / (alpha - 1)

        # Improved bound using log(alpha) term (Balle et al. 2020)
        eps_improved = rdp + (np.log(1 / delta) + np.log(alpha)) / (alpha - 1) - np.log(alpha / (alpha - 1))

        eps = min(eps, eps_improved)

        if eps < best_epsilon:
            best_epsilon = eps
            best_order = alpha

    return best_epsilon, best_order


@dataclass
class PrivacySpent:
    """Record of privacy budget spent."""

    epsilon: float
    delta: float
    round_number: int
    mechanism: str
    num_queries: int = 1
    noise_multiplier: Optional[float] = None
    sampling_rate: Optional[float] = None
    rdp_values: Optional[List[float]] = field(default=None, repr=False)


class PrivacyAccountant:
    """
    Tracks privacy budget consumption across training rounds.

    Supports multiple accounting methods:
    - Simple composition: ε_total = Σε_i (loose but simple)
    - Advanced composition: ε_total = √(2k·ln(1/δ'))·ε + k·ε·(e^ε-1) (tighter for many queries)
    - Rényi Differential Privacy (RDP): Composes in Rényi divergence, converts to (ε,δ)-DP

    RDP provides the tightest bounds for Gaussian mechanism composition and is
    the recommended method for deep learning with DP.

    Reference: Mironov, "Rényi Differential Privacy", CSF 2017
    """

    def __init__(
        self,
        total_epsilon: float,
        total_delta: float,
        accountant_type: str = "rdp",
        rdp_orders: Optional[List[float]] = None,
    ):
        """
        Initialize privacy accountant.

        Args:
            total_epsilon: Total privacy budget (ε).
            total_delta: Failure probability (δ).
            accountant_type: Accounting method ('simple', 'advanced', 'rdp').
            rdp_orders: Orders α for RDP accounting (default: [1.5, 2, ..., 1024]).
        """
        self.total_epsilon = total_epsilon
        self.total_delta = total_delta
        self.accountant_type = accountant_type
        self.rdp_orders = rdp_orders if rdp_orders is not None else DEFAULT_RDP_ORDERS

        self._spent_history: List[PrivacySpent] = []
        self._current_epsilon = 0.0
        self._current_delta = 0.0

        # RDP-specific: cumulative RDP values at each order
        self._cumulative_rdp: List[float] = [0.0] * len(self.rdp_orders)
        self._optimal_order: Optional[float] = None

    def spend(
        self,
        epsilon: float,
        delta: float,
        round_number: int,
        mechanism: str = "gaussian",
        noise_multiplier: Optional[float] = None,
        sampling_rate: float = 1.0,
    ) -> None:
        """
        Record privacy expenditure.

        Args:
            epsilon: Per-round epsilon (used for simple/advanced composition).
            delta: Per-round delta.
            round_number: Current round.
            mechanism: Mechanism used ('gaussian' or 'laplace').
            noise_multiplier: Noise σ/sensitivity (required for RDP with gaussian).
            sampling_rate: Fraction of data sampled per round (for amplification).

        Raises:
            PrivacyBudgetExceededError: If budget is exceeded.
        """
        # Compute RDP values for this round if using RDP accountant
        rdp_values = None
        if self.accountant_type == "rdp" and mechanism == "gaussian":
            if noise_multiplier is None:
                # Estimate noise multiplier from epsilon/delta
                noise_multiplier = np.sqrt(2 * np.log(1.25 / delta)) / epsilon
                logger.warning(
                    "No noise_multiplier provided, estimated from epsilon/delta",
                    estimated_sigma=noise_multiplier,
                )

            rdp_values = []
            for alpha in self.rdp_orders:
                if sampling_rate < 1.0:
                    rdp = compute_rdp_gaussian_subsampled(noise_multiplier, alpha, sampling_rate)
                else:
                    rdp = compute_rdp_gaussian(noise_multiplier, alpha)
                rdp_values.append(rdp)

        # Check if spending would exceed budget
        new_epsilon, new_delta = self._compose(epsilon, delta, rdp_values)

        if new_epsilon > self.total_epsilon:
            raise PrivacyBudgetExceededError(
                current_epsilon=new_epsilon,
                max_epsilon=self.total_epsilon,
                round_number=round_number,
            )

        # Update cumulative RDP if using RDP accountant
        if self.accountant_type == "rdp" and rdp_values is not None:
            self._cumulative_rdp = [
                cum + new for cum, new in zip(self._cumulative_rdp, rdp_values)
            ]

        # Record expenditure
        self._spent_history.append(
            PrivacySpent(
                epsilon=epsilon,
                delta=delta,
                round_number=round_number,
                mechanism=mechanism,
                noise_multiplier=noise_multiplier,
                sampling_rate=sampling_rate,
                rdp_values=rdp_values,
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
            accountant_type=self.accountant_type,
            optimal_order=self._optimal_order if self.accountant_type == "rdp" else None,
        )

    def _compose(
        self,
        epsilon: float,
        delta: float,
        rdp_values: Optional[List[float]] = None,
    ) -> Tuple[float, float]:
        """
        Compose privacy guarantee with current state.

        For RDP accounting, this uses proper Rényi composition and converts
        to (ε,δ)-DP at the end.

        Args:
            epsilon: New epsilon to add (for simple/advanced composition).
            delta: New delta to add.
            rdp_values: RDP values at each order (for RDP composition).

        Returns:
            Composed (epsilon, delta).
        """
        if self.accountant_type == "simple":
            # Simple composition: ε_total = sum(ε_i)
            return self._current_epsilon + epsilon, self._current_delta + delta

        elif self.accountant_type == "advanced":
            # Advanced composition theorem (Dwork et al.)
            # For k-fold composition of (ε, δ)-DP mechanisms:
            # Total is (ε', kδ + δ') where:
            # ε' = √(2k·ln(1/δ'))·ε + k·ε·(e^ε - 1)
            k = len(self._spent_history) + 1
            delta_prime = self.total_delta / (2 * k)  # Reserve budget for composition

            if epsilon > 0:
                epsilon_composed = (
                    np.sqrt(2 * k * np.log(1 / delta_prime)) * epsilon
                    + k * epsilon * (np.exp(epsilon) - 1)
                )
            else:
                epsilon_composed = 0.0

            delta_composed = k * delta + delta_prime
            return epsilon_composed, delta_composed

        elif self.accountant_type == "rdp":
            # RDP composition: sum RDP values at each order, then convert to (ε,δ)
            if rdp_values is None:
                # Fall back to simple composition if no RDP values provided
                logger.warning(
                    "No RDP values provided for RDP accountant, using simple composition"
                )
                return self._current_epsilon + epsilon, self._current_delta + delta

            # Add new RDP values to cumulative (preview, not committed yet)
            composed_rdp = [
                cum + new for cum, new in zip(self._cumulative_rdp, rdp_values)
            ]

            # Convert composed RDP to (ε, δ)-DP
            composed_epsilon, optimal_order = rdp_to_eps_delta(
                composed_rdp, self.rdp_orders, self.total_delta
            )
            self._optimal_order = optimal_order

            logger.debug(
                "RDP composition computed",
                composed_epsilon=composed_epsilon,
                optimal_order=optimal_order,
                rdp_at_optimal=composed_rdp[self.rdp_orders.index(optimal_order)]
                if optimal_order in self.rdp_orders
                else None,
            )

            return composed_epsilon, self.total_delta

        # Default: simple composition
        return self._current_epsilon + epsilon, self._current_delta + delta

    def get_remaining_budget(self) -> Tuple[float, float]:
        """Get remaining privacy budget."""
        return (
            max(0.0, self.total_epsilon - self._current_epsilon),
            max(0.0, self.total_delta - self._current_delta),
        )

    def get_spent_budget(self) -> Tuple[float, float]:
        """Get total spent privacy budget."""
        return self._current_epsilon, self._current_delta

    def get_rdp_budget(self) -> Optional[Dict[str, Any]]:
        """
        Get detailed RDP budget information.

        Returns:
            Dictionary with RDP-specific information, or None if not using RDP.
        """
        if self.accountant_type != "rdp":
            return None

        return {
            "cumulative_rdp": dict(zip(self.rdp_orders, self._cumulative_rdp)),
            "current_epsilon": self._current_epsilon,
            "optimal_order": self._optimal_order,
            "remaining_epsilon": self.total_epsilon - self._current_epsilon,
            "num_compositions": len(self._spent_history),
        }

    def can_spend(
        self,
        epsilon: float,
        delta: float,
        noise_multiplier: Optional[float] = None,
        sampling_rate: float = 1.0,
    ) -> bool:
        """
        Check if expenditure is within budget.

        Args:
            epsilon: Per-round epsilon.
            delta: Per-round delta.
            noise_multiplier: Noise σ/sensitivity (for RDP).
            sampling_rate: Sampling rate (for RDP with subsampling).

        Returns:
            True if the expenditure would not exceed budget.
        """
        rdp_values = None
        if self.accountant_type == "rdp" and noise_multiplier is not None:
            rdp_values = []
            for alpha in self.rdp_orders:
                if sampling_rate < 1.0:
                    rdp = compute_rdp_gaussian_subsampled(noise_multiplier, alpha, sampling_rate)
                else:
                    rdp = compute_rdp_gaussian(noise_multiplier, alpha)
                rdp_values.append(rdp)

        new_epsilon, new_delta = self._compose(epsilon, delta, rdp_values)
        return new_epsilon <= self.total_epsilon

    def get_history(self) -> List[PrivacySpent]:
        """Get history of privacy expenditures."""
        return self._spent_history.copy()

    def compute_epsilon_for_rounds(
        self,
        num_rounds: int,
        noise_multiplier: float,
        sampling_rate: float = 1.0,
    ) -> float:
        """
        Compute total epsilon for a given number of rounds (prospective).

        Useful for planning: determine the privacy cost before training.

        Args:
            num_rounds: Number of training rounds.
            noise_multiplier: Noise σ/sensitivity.
            sampling_rate: Fraction of data sampled per round.

        Returns:
            Total epsilon after num_rounds compositions.
        """
        if self.accountant_type == "rdp":
            # Compute RDP for one round
            rdp_per_round = []
            for alpha in self.rdp_orders:
                if sampling_rate < 1.0:
                    rdp = compute_rdp_gaussian_subsampled(noise_multiplier, alpha, sampling_rate)
                else:
                    rdp = compute_rdp_gaussian(noise_multiplier, alpha)
                rdp_per_round.append(rdp)

            # Compose for num_rounds (RDP adds linearly)
            total_rdp = [rdp * num_rounds for rdp in rdp_per_round]

            # Convert to (ε, δ)-DP
            total_epsilon, _ = rdp_to_eps_delta(total_rdp, self.rdp_orders, self.total_delta)
            return total_epsilon

        else:
            # Simple composition
            per_round_epsilon = np.sqrt(2 * np.log(1.25 / self.total_delta)) / noise_multiplier
            return per_round_epsilon * num_rounds

    def compute_noise_for_target_epsilon(
        self,
        target_epsilon: float,
        num_rounds: int,
        sampling_rate: float = 1.0,
    ) -> float:
        """
        Compute required noise multiplier for target epsilon (inverse problem).

        Uses binary search to find the minimum noise needed.

        Args:
            target_epsilon: Target total epsilon.
            num_rounds: Number of training rounds.
            sampling_rate: Fraction of data sampled per round.

        Returns:
            Minimum noise multiplier to achieve target epsilon.
        """
        # Binary search for noise multiplier
        low, high = 0.1, 100.0
        tolerance = 0.001

        while high - low > tolerance:
            mid = (low + high) / 2
            eps = self.compute_epsilon_for_rounds(num_rounds, mid, sampling_rate)

            if eps > target_epsilon:
                low = mid  # Need more noise
            else:
                high = mid  # Can use less noise

        return high

    def reset(self) -> None:
        """Reset accountant state."""
        self._spent_history.clear()
        self._current_epsilon = 0.0
        self._current_delta = 0.0
        self._cumulative_rdp = [0.0] * len(self.rdp_orders)
        self._optimal_order = None


class DifferentialPrivacy:
    """
    Differential privacy mechanism for gradient protection.

    Implements Gaussian mechanism for (ε, δ)-DP guarantees with proper
    RDP (Rényi Differential Privacy) accounting for composition.

    Reference: Abadi et al., "Deep Learning with Differential Privacy", CCS 2016
    """

    def __init__(
        self,
        epsilon: float = 1.0,
        delta: float = 1e-5,
        max_grad_norm: float = 1.0,
        mechanism: str = "gaussian",
        accountant: Optional[PrivacyAccountant] = None,
        sampling_rate: float = 1.0,
    ):
        """
        Initialize differential privacy mechanism.

        Args:
            epsilon: Privacy budget per round.
            delta: Failure probability.
            max_grad_norm: Maximum gradient L2 norm (sensitivity).
            mechanism: Noise mechanism ('gaussian', 'laplace').
            accountant: Optional privacy accountant.
            sampling_rate: Fraction of data sampled per round (for privacy amplification).
        """
        self.epsilon = epsilon
        self.delta = delta
        self.max_grad_norm = max_grad_norm
        self.mechanism = mechanism
        self.accountant = accountant
        self.sampling_rate = sampling_rate

        # Compute noise scale
        self._noise_scale = self._compute_noise_scale()

        # Noise multiplier = noise_scale / sensitivity (used for RDP accounting)
        self._noise_multiplier = self._noise_scale / self.max_grad_norm

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
        sampling_rate: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Add calibrated noise to gradients.

        Args:
            gradients: Dictionary of gradient tensors.
            round_number: Current training round.
            sampling_rate: Override sampling rate for this round (for amplification).

        Returns:
            Noised gradients dictionary.

        Raises:
            PrivacyBudgetExceededError: If budget is exceeded.
        """
        effective_sampling_rate = sampling_rate if sampling_rate is not None else self.sampling_rate

        # Check and record privacy expenditure with full RDP information
        if self.accountant:
            self.accountant.spend(
                epsilon=self.epsilon,
                delta=self.delta,
                round_number=round_number,
                mechanism=self.mechanism,
                noise_multiplier=self._noise_multiplier,
                sampling_rate=effective_sampling_rate,
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
            noise_multiplier=self._noise_multiplier,
            sampling_rate=effective_sampling_rate,
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

    def get_noise_multiplier(self) -> float:
        """Get noise multiplier (σ/sensitivity) for RDP accounting."""
        return self._noise_multiplier

    def compute_privacy_spent(
        self,
        num_rounds: int,
        use_rdp: bool = True,
    ) -> Tuple[float, float]:
        """
        Compute total privacy spent for given number of rounds.

        Uses RDP composition for tighter bounds when use_rdp=True.

        Args:
            num_rounds: Number of training rounds.
            use_rdp: Whether to use RDP composition (default True for tighter bounds).

        Returns:
            (epsilon, delta) total spent.
        """
        if use_rdp and self.mechanism == "gaussian":
            # Use RDP composition for tighter privacy accounting
            rdp_per_round = []
            for alpha in DEFAULT_RDP_ORDERS:
                if self.sampling_rate < 1.0:
                    rdp = compute_rdp_gaussian_subsampled(
                        self._noise_multiplier, alpha, self.sampling_rate
                    )
                else:
                    rdp = compute_rdp_gaussian(self._noise_multiplier, alpha)
                rdp_per_round.append(rdp)

            # RDP composition: sum over rounds
            total_rdp = [rdp * num_rounds for rdp in rdp_per_round]

            # Convert to (ε, δ)-DP
            total_epsilon, _ = rdp_to_eps_delta(total_rdp, DEFAULT_RDP_ORDERS, self.delta)
            return total_epsilon, self.delta

        # Fall back to simple composition
        total_epsilon = self.epsilon * num_rounds
        total_delta = min(1.0, self.delta * num_rounds)
        return total_epsilon, total_delta

    def recommend_rounds(
        self,
        target_epsilon: float,
        target_delta: Optional[float] = None,
        use_rdp: bool = True,
    ) -> int:
        """
        Recommend maximum rounds for target privacy budget.

        Uses binary search with RDP composition for accurate recommendations.

        Args:
            target_epsilon: Target total epsilon.
            target_delta: Target total delta (uses self.delta if None).
            use_rdp: Whether to use RDP composition.

        Returns:
            Maximum recommended training rounds.
        """
        if target_delta is None:
            target_delta = self.delta

        if use_rdp and self.mechanism == "gaussian":
            # Binary search for maximum rounds under RDP
            low, high = 1, 10000
            best_rounds = 1

            while low <= high:
                mid = (low + high) // 2
                eps, _ = self.compute_privacy_spent(mid, use_rdp=True)

                if eps <= target_epsilon:
                    best_rounds = mid
                    low = mid + 1
                else:
                    high = mid - 1

            return best_rounds

        # Simple composition fallback
        # Simple composition
        max_by_epsilon = int(target_epsilon / self.epsilon)
        max_by_delta = int(target_delta / self.delta)
        return min(max_by_epsilon, max_by_delta)

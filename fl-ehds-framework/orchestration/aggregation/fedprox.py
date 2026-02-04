"""
FedProx Aggregator
==================
Federated Proximal (FedProx) algorithm implementation for handling
heterogeneous (non-IID) data distributions.

Reference:
    Li et al., "Federated Optimization in Heterogeneous Networks",
    MLSys 2020
"""

from typing import Any, Dict, List, Optional
import numpy as np
import structlog

from .fedavg import FedAvgAggregator
from core.models import GradientUpdate
from core.exceptions import AggregationError

logger = structlog.get_logger(__name__)


class FedProxAggregator(FedAvgAggregator):
    """
    Federated Proximal (FedProx) aggregation algorithm.

    Extends FedAvg with a proximal term that limits how far local
    models can drift from the global model. This helps with:
    - Non-IID data distributions (67% barrier in EHDS contexts)
    - Systems heterogeneity (variable compute capabilities)
    - Partial work (clients may not complete all local epochs)
    """

    def __init__(
        self,
        num_rounds: int = 100,
        min_clients: int = 3,
        max_clients: Optional[int] = None,
        mu: float = 0.01,
        adaptive_mu: bool = False,
        mu_schedule: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize FedProx aggregator.

        Args:
            num_rounds: Maximum training rounds.
            min_clients: Minimum clients required per round.
            max_clients: Maximum clients per round.
            mu: Proximal term coefficient (larger = more regularization).
            adaptive_mu: Dynamically adjust mu based on drift.
            mu_schedule: Mu scheduling strategy ('linear', 'cosine', None).
            **kwargs: Additional arguments for base class.
        """
        super().__init__(
            num_rounds=num_rounds,
            min_clients=min_clients,
            max_clients=max_clients,
            **kwargs,
        )
        self.mu = mu
        self.initial_mu = mu
        self.adaptive_mu = adaptive_mu
        self.mu_schedule = mu_schedule

        # Track drift for adaptive mu
        self._drift_history: List[float] = []

    def get_algorithm_name(self) -> str:
        """Return algorithm name."""
        return "FedProx"

    def aggregate(
        self,
        updates: List[GradientUpdate],
        global_model_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Aggregate client updates using FedProx.

        FedProx aggregation is similar to FedAvg, but assumes clients
        have used the proximal term during local training. The server
        side aggregation is the same weighted average.

        Args:
            updates: List of gradient updates from clients.
            global_model_state: Current global model state.

        Returns:
            Updated global model state.
        """
        # Update mu if using schedule
        if self.mu_schedule:
            self._update_mu()

        # Compute average drift for monitoring
        if updates:
            avg_drift = self._compute_average_drift(updates, global_model_state)
            self._drift_history.append(avg_drift)

            logger.debug(
                "Client drift computed",
                round=self._current_round,
                avg_drift=avg_drift,
                mu=self.mu,
            )

            # Adaptive mu adjustment
            if self.adaptive_mu:
                self._adapt_mu(avg_drift)

        # Perform standard FedAvg aggregation
        # (FedProx's proximal term is applied client-side during training)
        aggregated = super().aggregate(updates, global_model_state)

        logger.info(
            "FedProx aggregation complete",
            round=self._current_round,
            mu=self.mu,
            avg_drift=self._drift_history[-1] if self._drift_history else None,
        )

        return aggregated

    def _compute_average_drift(
        self,
        updates: List[GradientUpdate],
        global_state: Dict[str, Any],
    ) -> float:
        """
        Compute average drift of client models from global model.

        Args:
            updates: Client gradient updates.
            global_state: Current global model state.

        Returns:
            Average L2 drift across clients.
        """
        drifts = []

        for update in updates:
            drift = 0.0
            for key in update.gradients:
                if key in global_state:
                    local_param = np.array(update.gradients[key])
                    global_param = np.array(global_state[key])
                    drift += np.sum((local_param - global_param) ** 2)
            drifts.append(np.sqrt(drift))

        return np.mean(drifts) if drifts else 0.0

    def _update_mu(self) -> None:
        """Update mu according to schedule."""
        progress = self._current_round / self.num_rounds

        if self.mu_schedule == "linear":
            # Linear decay
            self.mu = self.initial_mu * (1 - progress)

        elif self.mu_schedule == "cosine":
            # Cosine annealing
            self.mu = self.initial_mu * (1 + np.cos(np.pi * progress)) / 2

        elif self.mu_schedule == "warmup":
            # Warmup then decay
            warmup_rounds = min(10, self.num_rounds // 5)
            if self._current_round < warmup_rounds:
                self.mu = self.initial_mu * (self._current_round / warmup_rounds)
            else:
                self.mu = self.initial_mu

    def _adapt_mu(self, current_drift: float) -> None:
        """
        Adaptively adjust mu based on observed drift.

        Args:
            current_drift: Current round's average drift.
        """
        if len(self._drift_history) < 2:
            return

        # Compare to recent drift average
        recent_avg = np.mean(self._drift_history[-5:])
        drift_trend = current_drift / (recent_avg + 1e-10)

        # Adjust mu based on drift trend
        if drift_trend > 1.2:  # Drift increasing
            self.mu = min(self.mu * 1.5, 1.0)  # Increase mu
            logger.debug("Increasing mu due to drift", new_mu=self.mu)
        elif drift_trend < 0.8:  # Drift decreasing
            self.mu = max(self.mu * 0.8, 0.001)  # Decrease mu
            logger.debug("Decreasing mu due to convergence", new_mu=self.mu)

    def get_proximal_term(self, round_number: Optional[int] = None) -> float:
        """
        Get the current proximal term coefficient.

        Args:
            round_number: Optional specific round (uses current if None).

        Returns:
            Current mu value.
        """
        return self.mu

    def get_drift_history(self) -> List[float]:
        """Get history of drift values across rounds."""
        return self._drift_history.copy()

    def compute_client_proximal_loss(
        self,
        local_params: Dict[str, Any],
        global_params: Dict[str, Any],
    ) -> float:
        """
        Compute the proximal loss term for a client.

        This is the term that should be added to the client's local loss:
        (mu/2) * ||w - w_global||^2

        Args:
            local_params: Client's current parameters.
            global_params: Global model parameters.

        Returns:
            Proximal loss term value.
        """
        proximal_term = 0.0

        for key in local_params:
            if key in global_params:
                local = np.array(local_params[key])
                global_p = np.array(global_params[key])
                proximal_term += np.sum((local - global_p) ** 2)

        return (self.mu / 2) * proximal_term

    def reset(self) -> None:
        """Reset aggregator state."""
        super().reset()
        self.mu = self.initial_mu
        self._drift_history.clear()

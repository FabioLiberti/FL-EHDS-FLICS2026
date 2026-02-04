"""
FedAvg Aggregator
=================
Federated Averaging (FedAvg) algorithm implementation.

Reference:
    McMahan et al., "Communication-Efficient Learning of Deep Networks
    from Decentralized Data", AISTATS 2017
"""

from typing import Any, Dict, List, Optional
import numpy as np
import structlog

from .base import BaseAggregator
from core.models import GradientUpdate
from core.exceptions import AggregationError, InsufficientClientsError

logger = structlog.get_logger(__name__)


class FedAvgAggregator(BaseAggregator):
    """
    Federated Averaging (FedAvg) aggregation algorithm.

    Computes weighted average of model updates from participating clients,
    with weights proportional to local dataset sizes.
    """

    def __init__(
        self,
        num_rounds: int = 100,
        min_clients: int = 3,
        max_clients: Optional[int] = None,
        weighted: bool = True,
        momentum: float = 0.0,
        **kwargs,
    ):
        """
        Initialize FedAvg aggregator.

        Args:
            num_rounds: Maximum training rounds.
            min_clients: Minimum clients required per round.
            max_clients: Maximum clients per round.
            weighted: Use sample-weighted averaging (vs uniform).
            momentum: Server-side momentum (0 = no momentum).
            **kwargs: Additional arguments for base class.
        """
        super().__init__(
            num_rounds=num_rounds,
            min_clients=min_clients,
            max_clients=max_clients,
            **kwargs,
        )
        self.weighted = weighted
        self.momentum = momentum

        # Momentum buffer
        self._momentum_buffer: Optional[Dict[str, np.ndarray]] = None

    def get_algorithm_name(self) -> str:
        """Return algorithm name."""
        return "FedAvg"

    def aggregate(
        self,
        updates: List[GradientUpdate],
        global_model_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Aggregate client updates using FedAvg.

        Args:
            updates: List of gradient updates from clients.
            global_model_state: Current global model state.

        Returns:
            Updated global model state.

        Raises:
            InsufficientClientsError: If too few clients participated.
            AggregationError: If aggregation fails.
        """
        # Validate minimum clients
        if len(updates) < self.min_clients:
            raise InsufficientClientsError(
                required=self.min_clients,
                available=len(updates),
                round_number=self._current_round,
            )

        # Validate updates
        valid_updates, rejected = self.validate_updates(updates)
        if len(valid_updates) < self.min_clients:
            raise InsufficientClientsError(
                required=self.min_clients,
                available=len(valid_updates),
                round_number=self._current_round,
            )

        if rejected:
            logger.warning(
                "Some updates rejected",
                rejected_clients=rejected,
                valid_count=len(valid_updates),
            )

        try:
            # Compute weights
            weighting = "samples" if self.weighted else "uniform"
            weights = self.compute_weights(valid_updates, weighting)

            # Aggregate gradients
            aggregated_state = self._weighted_average(
                valid_updates, weights, global_model_state
            )

            # Apply momentum if configured
            if self.momentum > 0:
                aggregated_state = self._apply_momentum(
                    aggregated_state, global_model_state
                )

            # Update round counter
            self._current_round += 1

            # Log aggregation
            total_samples = sum(u.num_samples for u in valid_updates)
            avg_loss = sum(u.local_loss * w for u, w in zip(valid_updates, weights))

            logger.info(
                "FedAvg aggregation complete",
                round=self._current_round,
                clients=len(valid_updates),
                total_samples=total_samples,
                avg_loss=avg_loss,
            )

            return aggregated_state

        except Exception as e:
            raise AggregationError(
                f"FedAvg aggregation failed: {str(e)}",
                round_number=self._current_round,
                participating_clients=len(updates),
            )

    def _weighted_average(
        self,
        updates: List[GradientUpdate],
        weights: List[float],
        global_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Compute weighted average of model parameters.

        Args:
            updates: Client gradient updates.
            weights: Aggregation weights.
            global_state: Current global model state.

        Returns:
            Aggregated model state.
        """
        aggregated = {}

        # Get parameter keys from first update
        param_keys = updates[0].gradients.keys()

        for key in param_keys:
            # Stack parameters from all clients
            param_stack = []
            for update in updates:
                param = update.gradients.get(key)
                if param is not None:
                    # Convert to numpy if needed
                    if hasattr(param, "numpy"):
                        param = param.numpy()
                    elif not isinstance(param, np.ndarray):
                        param = np.array(param)
                    param_stack.append(param)

            if param_stack:
                # Weighted average
                weighted_params = [
                    w * p for w, p in zip(weights[: len(param_stack)], param_stack)
                ]
                aggregated[key] = sum(weighted_params)

        return aggregated

    def _apply_momentum(
        self,
        aggregated_state: Dict[str, Any],
        previous_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Apply server-side momentum.

        Args:
            aggregated_state: Newly aggregated state.
            previous_state: Previous global state.

        Returns:
            State with momentum applied.
        """
        if self._momentum_buffer is None:
            self._momentum_buffer = {}

        result = {}
        for key in aggregated_state:
            # Compute update delta
            if key in previous_state:
                delta = aggregated_state[key] - np.array(previous_state[key])
            else:
                delta = aggregated_state[key]

            # Update momentum buffer
            if key in self._momentum_buffer:
                self._momentum_buffer[key] = (
                    self.momentum * self._momentum_buffer[key] + delta
                )
            else:
                self._momentum_buffer[key] = delta

            # Apply momentum
            if key in previous_state:
                result[key] = np.array(previous_state[key]) + self._momentum_buffer[key]
            else:
                result[key] = self._momentum_buffer[key]

        return result

    def aggregate_with_compression(
        self,
        updates: List[GradientUpdate],
        global_model_state: Dict[str, Any],
        compression_rate: float = 0.1,
    ) -> Dict[str, Any]:
        """
        Aggregate with gradient compression for bandwidth efficiency.

        Args:
            updates: Client gradient updates.
            global_model_state: Current global model state.
            compression_rate: Fraction of gradients to keep (top-k).

        Returns:
            Aggregated model state.
        """
        # Decompress gradients before aggregation
        decompressed_updates = []
        for update in updates:
            decompressed_grads = self._decompress_gradients(
                update.gradients, compression_rate
            )
            decompressed_update = GradientUpdate(
                client_id=update.client_id,
                round_number=update.round_number,
                gradients=decompressed_grads,
                num_samples=update.num_samples,
                local_loss=update.local_loss,
            )
            decompressed_updates.append(decompressed_update)

        return self.aggregate(decompressed_updates, global_model_state)

    def _decompress_gradients(
        self,
        compressed: Dict[str, Any],
        compression_rate: float,
    ) -> Dict[str, Any]:
        """Decompress gradient updates."""
        # Implementation placeholder for gradient decompression
        # Would reconstruct full gradients from compressed representation
        return compressed

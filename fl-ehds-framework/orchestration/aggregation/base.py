"""
Base Aggregator
===============
Abstract base class for FL aggregation algorithms.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import structlog

from core.models import GradientUpdate, FLRound, ModelCheckpoint

logger = structlog.get_logger(__name__)


class BaseAggregator(ABC):
    """
    Abstract base class for federated learning aggregation algorithms.

    Provides common interface and utilities for gradient aggregation
    within the FL-EHDS framework.
    """

    def __init__(
        self,
        num_rounds: int = 100,
        min_clients: int = 3,
        max_clients: Optional[int] = None,
        early_stopping: bool = True,
        early_stopping_patience: int = 10,
        early_stopping_min_delta: float = 0.001,
    ):
        """
        Initialize aggregator.

        Args:
            num_rounds: Maximum training rounds.
            min_clients: Minimum clients required per round.
            max_clients: Maximum clients per round (None = all available).
            early_stopping: Enable early stopping.
            early_stopping_patience: Rounds without improvement before stopping.
            early_stopping_min_delta: Minimum improvement threshold.
        """
        self.num_rounds = num_rounds
        self.min_clients = min_clients
        self.max_clients = max_clients
        self.early_stopping = early_stopping
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta

        # State tracking
        self._current_round = 0
        self._best_loss = float("inf")
        self._rounds_without_improvement = 0
        self._global_model_state: Optional[Dict[str, Any]] = None
        self._round_history: List[FLRound] = []

    @abstractmethod
    def aggregate(
        self,
        updates: List[GradientUpdate],
        global_model_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Aggregate client gradient updates into updated global model.

        Args:
            updates: List of gradient updates from clients.
            global_model_state: Current global model state.

        Returns:
            Updated global model state.
        """
        pass

    @abstractmethod
    def get_algorithm_name(self) -> str:
        """Return the name of the aggregation algorithm."""
        pass

    def select_clients(
        self,
        available_clients: List[str],
        round_number: int,
    ) -> List[str]:
        """
        Select clients for training round.

        Args:
            available_clients: List of available client IDs.
            round_number: Current round number.

        Returns:
            List of selected client IDs.
        """
        import random

        if len(available_clients) < self.min_clients:
            logger.warning(
                "Insufficient clients available",
                available=len(available_clients),
                required=self.min_clients,
            )
            return []

        # Determine number of clients to select
        num_select = len(available_clients)
        if self.max_clients:
            num_select = min(num_select, self.max_clients)

        # Random selection (can be overridden for other strategies)
        selected = random.sample(available_clients, num_select)

        logger.info(
            "Clients selected for round",
            round_number=round_number,
            selected=len(selected),
            available=len(available_clients),
        )

        return selected

    def should_stop(self, current_loss: float) -> bool:
        """
        Check if training should stop early.

        Args:
            current_loss: Loss from current round.

        Returns:
            True if training should stop.
        """
        if not self.early_stopping:
            return False

        if current_loss < self._best_loss - self.early_stopping_min_delta:
            self._best_loss = current_loss
            self._rounds_without_improvement = 0
            return False

        self._rounds_without_improvement += 1

        if self._rounds_without_improvement >= self.early_stopping_patience:
            logger.info(
                "Early stopping triggered",
                rounds_without_improvement=self._rounds_without_improvement,
                best_loss=self._best_loss,
            )
            return True

        return False

    def validate_updates(
        self,
        updates: List[GradientUpdate],
    ) -> Tuple[List[GradientUpdate], List[str]]:
        """
        Validate gradient updates.

        Args:
            updates: List of gradient updates to validate.

        Returns:
            Tuple of (valid_updates, rejected_client_ids).
        """
        valid = []
        rejected = []

        for update in updates:
            if self._is_valid_update(update):
                valid.append(update)
            else:
                rejected.append(update.client_id)
                logger.warning(
                    "Invalid gradient update rejected",
                    client_id=update.client_id,
                    round=update.round_number,
                )

        return valid, rejected

    def _is_valid_update(self, update: GradientUpdate) -> bool:
        """Check if gradient update is valid."""
        # Basic validation
        if not update.gradients:
            return False
        if update.num_samples <= 0:
            return False
        if update.round_number != self._current_round:
            return False
        return True

    def compute_weights(
        self,
        updates: List[GradientUpdate],
        weighting: str = "samples",
    ) -> List[float]:
        """
        Compute aggregation weights for each update.

        Args:
            updates: List of gradient updates.
            weighting: Weighting strategy ('samples', 'uniform', 'loss').

        Returns:
            List of weights (sum to 1.0).
        """
        if weighting == "uniform":
            weight = 1.0 / len(updates)
            return [weight] * len(updates)

        elif weighting == "samples":
            total_samples = sum(u.num_samples for u in updates)
            return [u.num_samples / total_samples for u in updates]

        elif weighting == "loss":
            # Inverse loss weighting (lower loss = higher weight)
            inv_losses = [1.0 / (u.local_loss + 1e-10) for u in updates]
            total = sum(inv_losses)
            return [l / total for l in inv_losses]

        else:
            raise ValueError(f"Unknown weighting strategy: {weighting}")

    def create_round_record(
        self,
        round_number: int,
        updates: List[GradientUpdate],
        aggregated_loss: float,
        metrics: Dict[str, float],
    ) -> FLRound:
        """
        Create record of completed round.

        Args:
            round_number: Round number.
            updates: Updates received.
            aggregated_loss: Aggregated loss value.
            metrics: Round metrics.

        Returns:
            FLRound record.
        """
        from datetime import datetime

        round_record = FLRound(
            round_number=round_number,
            status="completed",
            participating_clients=[u.client_id for u in updates],
            total_samples=sum(u.num_samples for u in updates),
            aggregated_loss=aggregated_loss,
            metrics=metrics,
            completed_at=datetime.utcnow(),
        )

        self._round_history.append(round_record)
        return round_record

    def get_round_history(self) -> List[FLRound]:
        """Get history of completed rounds."""
        return self._round_history.copy()

    def reset(self) -> None:
        """Reset aggregator state."""
        self._current_round = 0
        self._best_loss = float("inf")
        self._rounds_without_improvement = 0
        self._global_model_state = None
        self._round_history.clear()
        logger.info("Aggregator state reset")

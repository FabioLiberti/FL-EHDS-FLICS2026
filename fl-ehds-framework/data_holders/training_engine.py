"""
Training Engine Module
======================
Adaptive local training engine for data holders with support
for hardware heterogeneity (78% barrier).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import structlog

from core.models import GradientUpdate, TrainingConfig
from core.exceptions import TrainingError, ResourceConstraintError

logger = structlog.get_logger(__name__)


@dataclass
class HardwareProfile:
    """Hardware capabilities of a data holder."""

    device_type: str  # 'cpu', 'gpu', 'tpu'
    memory_gb: float
    compute_units: int
    network_bandwidth_mbps: float
    storage_available_gb: float

    def can_handle(self, model_size_mb: float, batch_size: int) -> bool:
        """Check if hardware can handle the workload."""
        required_memory = model_size_mb / 1024 + (batch_size * 0.01)  # Rough estimate
        return self.memory_gb >= required_memory


@dataclass
class TrainingMetrics:
    """Metrics from local training."""

    loss: float
    accuracy: Optional[float]
    num_samples: int
    num_epochs: int
    training_time_seconds: float
    gradients_computed: bool


class TrainingEngine:
    """
    Local training engine for FL data holders.

    Implements adaptive training strategies to handle hardware
    heterogeneity across participating organizations.
    """

    def __init__(
        self,
        config: Optional[TrainingConfig] = None,
        hardware_profile: Optional[HardwareProfile] = None,
        model_factory: Optional[Callable] = None,
    ):
        """
        Initialize training engine.

        Args:
            config: Training configuration.
            hardware_profile: Local hardware capabilities.
            model_factory: Factory function to create model instances.
        """
        self.config = config or TrainingConfig()
        self.hardware_profile = hardware_profile
        self.model_factory = model_factory

        # State
        self._model = None
        self._optimizer = None
        self._current_round = 0
        self._training_history: List[TrainingMetrics] = []

    def initialize_model(
        self,
        model_state: Dict[str, Any],
        round_number: int,
    ) -> None:
        """
        Initialize local model from global state.

        Args:
            model_state: Global model state dictionary.
            round_number: Current training round.
        """
        self._current_round = round_number

        if self.model_factory:
            self._model = self.model_factory()
            self._load_model_state(model_state)
        else:
            # Store state directly for frameworks that manage their own models
            self._model = model_state

        logger.info(
            "Model initialized for training",
            round=round_number,
            model_params=len(model_state) if model_state else 0,
        )

    def _load_model_state(self, state: Dict[str, Any]) -> None:
        """Load state into model."""
        if hasattr(self._model, "load_state_dict"):
            self._model.load_state_dict(state)
        elif hasattr(self._model, "set_weights"):
            self._model.set_weights(list(state.values()))

    def train(
        self,
        data: Any,
        labels: Any,
        global_model_state: Optional[Dict[str, Any]] = None,
        proximal_mu: float = 0.0,
    ) -> Tuple[Dict[str, Any], TrainingMetrics]:
        """
        Perform local training.

        Args:
            data: Training data.
            labels: Training labels.
            global_model_state: Global model state (for FedProx).
            proximal_mu: Proximal term coefficient (for FedProx).

        Returns:
            Tuple of (gradients/updates, training_metrics).
        """
        start_time = datetime.utcnow()

        # Adjust batch size if needed
        effective_batch_size = self._adjust_batch_size(len(data))

        try:
            # Perform training epochs
            total_loss = 0.0
            num_batches = 0

            for epoch in range(self.config.local_epochs):
                epoch_loss = self._train_epoch(
                    data,
                    labels,
                    effective_batch_size,
                    global_model_state,
                    proximal_mu,
                )
                total_loss += epoch_loss
                num_batches += 1

            avg_loss = total_loss / num_batches

            # Compute gradients/updates
            gradients = self._compute_gradients(global_model_state)

            # Compute metrics
            training_time = (datetime.utcnow() - start_time).total_seconds()
            metrics = TrainingMetrics(
                loss=avg_loss,
                accuracy=None,  # Would compute if validation data available
                num_samples=len(data),
                num_epochs=self.config.local_epochs,
                training_time_seconds=training_time,
                gradients_computed=True,
            )

            self._training_history.append(metrics)

            logger.info(
                "Local training completed",
                round=self._current_round,
                loss=avg_loss,
                samples=len(data),
                time_seconds=training_time,
            )

            return gradients, metrics

        except Exception as e:
            raise TrainingError(
                f"Local training failed: {str(e)}",
                epoch=self.config.local_epochs,
            )

    def _train_epoch(
        self,
        data: Any,
        labels: Any,
        batch_size: int,
        global_state: Optional[Dict[str, Any]],
        proximal_mu: float,
    ) -> float:
        """
        Train for one epoch.

        Args:
            data: Training data.
            labels: Training labels.
            batch_size: Batch size.
            global_state: Global model state for proximal term.
            proximal_mu: Proximal coefficient.

        Returns:
            Average loss for the epoch.
        """
        # This is a simplified implementation
        # In practice, would use PyTorch/TensorFlow training loop

        num_samples = len(data)
        num_batches = max(1, num_samples // batch_size)
        total_loss = 0.0

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_samples)

            # Simulate batch training
            batch_loss = self._train_batch(
                data[start_idx:end_idx],
                labels[start_idx:end_idx],
                global_state,
                proximal_mu,
            )
            total_loss += batch_loss

        return total_loss / num_batches

    def _train_batch(
        self,
        batch_data: Any,
        batch_labels: Any,
        global_state: Optional[Dict[str, Any]],
        proximal_mu: float,
    ) -> float:
        """Train on a single batch."""
        # Placeholder for actual training logic
        # Returns simulated loss
        return np.random.uniform(0.1, 1.0)

    def _compute_gradients(
        self,
        global_state: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Compute model gradients or weight updates.

        Args:
            global_state: Global model state.

        Returns:
            Dictionary of gradients/updates per layer.
        """
        if self._model is None:
            raise TrainingError("Model not initialized")

        # Get current model state
        if hasattr(self._model, "state_dict"):
            current_state = self._model.state_dict()
        elif isinstance(self._model, dict):
            current_state = self._model
        else:
            raise TrainingError("Unable to extract model state")

        # If global state provided, compute delta (updates)
        if global_state:
            gradients = {}
            for key in current_state:
                if key in global_state:
                    gradients[key] = (
                        np.array(current_state[key]) - np.array(global_state[key])
                    )
                else:
                    gradients[key] = np.array(current_state[key])
            return gradients

        return {k: np.array(v) for k, v in current_state.items()}

    def _adjust_batch_size(self, dataset_size: int) -> int:
        """
        Adjust batch size based on hardware and dataset.

        Args:
            dataset_size: Size of local dataset.

        Returns:
            Adjusted batch size.
        """
        if not self.config.adaptive_batching:
            return min(self.config.batch_size, dataset_size)

        batch_size = self.config.batch_size

        # Adjust based on hardware profile
        if self.hardware_profile:
            if self.hardware_profile.memory_gb < 4:
                batch_size = min(batch_size, self.config.min_batch_size * 2)
            elif self.hardware_profile.memory_gb < 8:
                batch_size = min(batch_size, self.config.batch_size // 2)

        # Ensure batch size doesn't exceed dataset
        batch_size = min(batch_size, dataset_size)

        # Ensure within bounds
        batch_size = max(self.config.min_batch_size, batch_size)
        batch_size = min(self.config.max_batch_size, batch_size)

        logger.debug(
            "Batch size adjusted",
            original=self.config.batch_size,
            adjusted=batch_size,
            dataset_size=dataset_size,
        )

        return batch_size

    def create_gradient_update(
        self,
        client_id: str,
        gradients: Dict[str, Any],
        metrics: TrainingMetrics,
    ) -> GradientUpdate:
        """
        Create gradient update message.

        Args:
            client_id: Client identifier.
            gradients: Computed gradients.
            metrics: Training metrics.

        Returns:
            GradientUpdate object.
        """
        return GradientUpdate(
            client_id=client_id,
            round_number=self._current_round,
            gradients=gradients,
            num_samples=metrics.num_samples,
            local_loss=metrics.loss,
            local_metrics={
                "training_time": metrics.training_time_seconds,
                "epochs": metrics.num_epochs,
            },
        )

    def get_training_history(self) -> List[TrainingMetrics]:
        """Get history of training metrics."""
        return self._training_history.copy()


class AdaptiveTrainer(TrainingEngine):
    """
    Extended training engine with adaptive capabilities for
    handling hardware heterogeneity (78% barrier).
    """

    def __init__(
        self,
        config: Optional[TrainingConfig] = None,
        hardware_profile: Optional[HardwareProfile] = None,
        model_factory: Optional[Callable] = None,
        adaptive_lr: bool = True,
        gradient_accumulation: bool = True,
    ):
        """
        Initialize adaptive trainer.

        Args:
            config: Training configuration.
            hardware_profile: Hardware capabilities.
            model_factory: Model factory function.
            adaptive_lr: Enable adaptive learning rate.
            gradient_accumulation: Enable gradient accumulation.
        """
        super().__init__(config, hardware_profile, model_factory)
        self.adaptive_lr = adaptive_lr
        self.gradient_accumulation = gradient_accumulation

        self._effective_lr = config.learning_rate if config else 0.01
        self._accumulation_steps = 1

    def configure_for_hardware(self) -> Dict[str, Any]:
        """
        Configure training parameters based on hardware profile.

        Returns:
            Configuration adjustments made.
        """
        adjustments = {}

        if not self.hardware_profile:
            return adjustments

        # Adjust for limited memory
        if self.hardware_profile.memory_gb < 4:
            # Use gradient accumulation for effective larger batches
            self._accumulation_steps = 4
            adjustments["gradient_accumulation_steps"] = 4
            adjustments["effective_batch_size"] = (
                self.config.min_batch_size * self._accumulation_steps
            )

        # Adjust for limited compute
        if self.hardware_profile.compute_units < 4:
            # Reduce local epochs
            self.config.local_epochs = max(1, self.config.local_epochs // 2)
            adjustments["reduced_epochs"] = self.config.local_epochs

        # Adjust for network constraints
        if self.hardware_profile.network_bandwidth_mbps < 10:
            # Would enable gradient compression here
            adjustments["compression_enabled"] = True

        logger.info(
            "Training configured for hardware",
            memory_gb=self.hardware_profile.memory_gb,
            adjustments=adjustments,
        )

        return adjustments

    def adjust_learning_rate(
        self,
        round_number: int,
        global_loss: Optional[float] = None,
    ) -> float:
        """
        Adjust learning rate based on training progress.

        Args:
            round_number: Current round.
            global_loss: Optional global loss for adjustment.

        Returns:
            Adjusted learning rate.
        """
        if not self.adaptive_lr:
            return self._effective_lr

        base_lr = self.config.learning_rate

        # Learning rate warmup for first few rounds
        warmup_rounds = 5
        if round_number < warmup_rounds:
            self._effective_lr = base_lr * (round_number + 1) / warmup_rounds
        else:
            # Cosine decay
            progress = (round_number - warmup_rounds) / 100  # Assume 100 total rounds
            self._effective_lr = base_lr * (1 + np.cos(np.pi * progress)) / 2

        return self._effective_lr

    def estimate_training_time(
        self,
        dataset_size: int,
        model_size_params: int,
    ) -> float:
        """
        Estimate training time based on hardware and workload.

        Args:
            dataset_size: Number of training samples.
            model_size_params: Number of model parameters.

        Returns:
            Estimated training time in seconds.
        """
        if not self.hardware_profile:
            return 0.0

        # Base estimates (simplified)
        flops_per_sample = model_size_params * 6  # Forward + backward
        total_flops = flops_per_sample * dataset_size * self.config.local_epochs

        # Estimate based on compute units (very rough)
        gflops_per_unit = 100 if self.hardware_profile.device_type == "gpu" else 10
        total_gflops = total_flops / 1e9
        compute_seconds = total_gflops / (
            self.hardware_profile.compute_units * gflops_per_unit
        )

        return compute_seconds

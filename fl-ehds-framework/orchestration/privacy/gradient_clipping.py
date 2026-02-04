"""
Gradient Clipping Module
========================
Implements gradient clipping for bounding sensitivity in differential
privacy and mitigating gradient inversion attacks.

Reference:
    Zhu et al., "Deep Leakage from Gradients", NeurIPS 2019
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import structlog

from core.exceptions import GradientClippingError

logger = structlog.get_logger(__name__)


class GradientClipper:
    """
    Gradient clipping for privacy protection.

    Bounds the L2 norm of gradients to limit the sensitivity of
    individual contributions, enabling differential privacy guarantees.
    """

    def __init__(
        self,
        max_norm: float = 1.0,
        norm_type: str = "l2",
        per_layer: bool = False,
        adaptive: bool = False,
        adaptive_percentile: float = 0.5,
    ):
        """
        Initialize gradient clipper.

        Args:
            max_norm: Maximum gradient norm.
            norm_type: Norm type ('l2', 'linf').
            per_layer: Clip each layer separately vs global clip.
            adaptive: Adaptively adjust clipping threshold.
            adaptive_percentile: Percentile for adaptive clipping.
        """
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.per_layer = per_layer
        self.adaptive = adaptive
        self.adaptive_percentile = adaptive_percentile

        # Statistics tracking
        self._clip_counts: List[int] = []
        self._norm_history: List[float] = []

    def clip(
        self,
        gradients: Dict[str, Any],
        record_stats: bool = True,
    ) -> Tuple[Dict[str, Any], bool]:
        """
        Clip gradients to bounded norm.

        Args:
            gradients: Dictionary of gradient tensors.
            record_stats: Whether to record clipping statistics.

        Returns:
            Tuple of (clipped_gradients, was_clipped).
        """
        if self.per_layer:
            return self._clip_per_layer(gradients, record_stats)
        else:
            return self._clip_global(gradients, record_stats)

    def _clip_global(
        self,
        gradients: Dict[str, Any],
        record_stats: bool,
    ) -> Tuple[Dict[str, Any], bool]:
        """
        Apply global gradient clipping.

        Args:
            gradients: Gradient dictionary.
            record_stats: Record statistics.

        Returns:
            Tuple of (clipped_gradients, was_clipped).
        """
        # Compute global norm
        global_norm = self._compute_global_norm(gradients)

        if record_stats:
            self._norm_history.append(global_norm)

        # Check if clipping needed
        if global_norm <= self.max_norm:
            return gradients, False

        # Compute clipping factor
        clip_factor = self.max_norm / (global_norm + 1e-10)

        # Apply clipping
        clipped = {}
        for key, grad in gradients.items():
            clipped[key] = np.array(grad) * clip_factor

        if record_stats:
            self._clip_counts.append(1)

        logger.debug(
            "Gradients clipped",
            original_norm=global_norm,
            max_norm=self.max_norm,
            clip_factor=clip_factor,
        )

        return clipped, True

    def _clip_per_layer(
        self,
        gradients: Dict[str, Any],
        record_stats: bool,
    ) -> Tuple[Dict[str, Any], bool]:
        """
        Apply per-layer gradient clipping.

        Args:
            gradients: Gradient dictionary.
            record_stats: Record statistics.

        Returns:
            Tuple of (clipped_gradients, was_clipped).
        """
        clipped = {}
        any_clipped = False
        clip_count = 0

        for key, grad in gradients.items():
            grad_array = np.array(grad)
            layer_norm = self._compute_norm(grad_array)

            if layer_norm > self.max_norm:
                clip_factor = self.max_norm / (layer_norm + 1e-10)
                clipped[key] = grad_array * clip_factor
                any_clipped = True
                clip_count += 1
            else:
                clipped[key] = grad_array

        if record_stats and any_clipped:
            self._clip_counts.append(clip_count)

        return clipped, any_clipped

    def _compute_global_norm(self, gradients: Dict[str, Any]) -> float:
        """Compute global gradient norm."""
        if self.norm_type == "l2":
            squared_sum = sum(
                np.sum(np.array(grad) ** 2) for grad in gradients.values()
            )
            return np.sqrt(squared_sum)

        elif self.norm_type == "linf":
            return max(
                np.max(np.abs(np.array(grad))) for grad in gradients.values()
            )

        raise ValueError(f"Unknown norm type: {self.norm_type}")

    def _compute_norm(self, tensor: np.ndarray) -> float:
        """Compute norm of a single tensor."""
        if self.norm_type == "l2":
            return np.sqrt(np.sum(tensor ** 2))
        elif self.norm_type == "linf":
            return np.max(np.abs(tensor))
        raise ValueError(f"Unknown norm type: {self.norm_type}")

    def clip_batch(
        self,
        gradients_batch: List[Dict[str, Any]],
    ) -> List[Tuple[Dict[str, Any], bool]]:
        """
        Clip a batch of gradient updates.

        Args:
            gradients_batch: List of gradient dictionaries.

        Returns:
            List of (clipped_gradients, was_clipped) tuples.
        """
        results = []
        for gradients in gradients_batch:
            clipped, was_clipped = self.clip(gradients)
            results.append((clipped, was_clipped))
        return results

    def update_max_norm(self, new_max_norm: float) -> None:
        """
        Update the clipping threshold.

        Args:
            new_max_norm: New maximum gradient norm.
        """
        old_norm = self.max_norm
        self.max_norm = new_max_norm
        logger.info(
            "Clipping threshold updated",
            old_max_norm=old_norm,
            new_max_norm=new_max_norm,
        )

    def compute_adaptive_threshold(
        self,
        gradient_norms: List[float],
    ) -> float:
        """
        Compute adaptive clipping threshold from gradient norm distribution.

        Args:
            gradient_norms: List of observed gradient norms.

        Returns:
            Adaptive clipping threshold.
        """
        if not gradient_norms:
            return self.max_norm

        threshold = np.percentile(gradient_norms, self.adaptive_percentile * 100)

        logger.debug(
            "Adaptive threshold computed",
            percentile=self.adaptive_percentile,
            threshold=threshold,
            num_samples=len(gradient_norms),
        )

        return threshold

    def get_statistics(self) -> Dict[str, Any]:
        """Get clipping statistics."""
        total_clips = sum(self._clip_counts)
        total_updates = len(self._norm_history)

        return {
            "total_updates": total_updates,
            "total_clips": total_clips,
            "clip_rate": total_clips / total_updates if total_updates > 0 else 0,
            "max_norm_threshold": self.max_norm,
            "avg_gradient_norm": (
                np.mean(self._norm_history) if self._norm_history else 0
            ),
            "max_gradient_norm": (
                np.max(self._norm_history) if self._norm_history else 0
            ),
        }

    def reset_statistics(self) -> None:
        """Reset clipping statistics."""
        self._clip_counts.clear()
        self._norm_history.clear()

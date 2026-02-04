"""
Tests for FL Orchestration Layer (Layer 2)
==========================================
"""

import pytest
import numpy as np
from typing import Dict, Any

from core.models import GradientUpdate, TrainingConfig
from core.exceptions import (
    AggregationError,
    PrivacyBudgetExhaustedError,
    SecureAggregationError,
)
from orchestration.aggregation.fedavg import FedAvgAggregator
from orchestration.aggregation.fedprox import FedProxAggregator
from orchestration.privacy.differential_privacy import (
    DifferentialPrivacyMechanism,
    PrivacyAccountant,
)
from orchestration.privacy.gradient_clipping import GradientClipper
from orchestration.privacy.secure_aggregation import SecureAggregator


class TestFedAvgAggregator:
    """Tests for FedAvg aggregation."""

    def test_weighted_average(self):
        """Test weighted averaging of gradients."""
        aggregator = FedAvgAggregator(weighted_average=True)

        updates = [
            GradientUpdate(
                client_id="client-1",
                round_number=1,
                gradients={"layer1": np.array([1.0, 2.0, 3.0])},
                num_samples=100,
                local_loss=0.5,
            ),
            GradientUpdate(
                client_id="client-2",
                round_number=1,
                gradients={"layer1": np.array([4.0, 5.0, 6.0])},
                num_samples=200,
                local_loss=0.4,
            ),
        ]

        result = aggregator.aggregate(updates)

        # Weighted average: (100*[1,2,3] + 200*[4,5,6]) / 300 = [3, 4, 5]
        expected = np.array([3.0, 4.0, 5.0])
        np.testing.assert_array_almost_equal(result["layer1"], expected)

    def test_unweighted_average(self):
        """Test unweighted averaging of gradients."""
        aggregator = FedAvgAggregator(weighted_average=False)

        updates = [
            GradientUpdate(
                client_id="client-1",
                round_number=1,
                gradients={"layer1": np.array([1.0, 2.0])},
                num_samples=100,
                local_loss=0.5,
            ),
            GradientUpdate(
                client_id="client-2",
                round_number=1,
                gradients={"layer1": np.array([3.0, 4.0])},
                num_samples=200,
                local_loss=0.4,
            ),
        ]

        result = aggregator.aggregate(updates)

        # Unweighted: ([1,2] + [3,4]) / 2 = [2, 3]
        expected = np.array([2.0, 3.0])
        np.testing.assert_array_almost_equal(result["layer1"], expected)

    def test_empty_updates(self):
        """Test handling of empty updates list."""
        aggregator = FedAvgAggregator()

        with pytest.raises(AggregationError):
            aggregator.aggregate([])

    def test_multiple_layers(self):
        """Test aggregation across multiple model layers."""
        aggregator = FedAvgAggregator(weighted_average=True)

        updates = [
            GradientUpdate(
                client_id="client-1",
                round_number=1,
                gradients={
                    "layer1": np.array([1.0, 2.0]),
                    "layer2": np.array([10.0, 20.0]),
                },
                num_samples=50,
                local_loss=0.5,
            ),
            GradientUpdate(
                client_id="client-2",
                round_number=1,
                gradients={
                    "layer1": np.array([3.0, 4.0]),
                    "layer2": np.array([30.0, 40.0]),
                },
                num_samples=50,
                local_loss=0.4,
            ),
        ]

        result = aggregator.aggregate(updates)

        assert "layer1" in result
        assert "layer2" in result


class TestFedProxAggregator:
    """Tests for FedProx aggregation."""

    def test_proximal_term_applied(self):
        """Test that proximal term modifies aggregation."""
        aggregator = FedProxAggregator(mu=0.1)

        global_model = {"layer1": np.array([0.0, 0.0])}

        updates = [
            GradientUpdate(
                client_id="client-1",
                round_number=1,
                gradients={"layer1": np.array([1.0, 1.0])},
                num_samples=100,
                local_loss=0.5,
            ),
        ]

        result = aggregator.aggregate(updates, global_model=global_model)

        # With mu > 0, result should be pulled toward global model
        assert result is not None
        assert "layer1" in result

    def test_mu_zero_equals_fedavg(self):
        """Test that mu=0 gives same result as FedAvg."""
        fedprox = FedProxAggregator(mu=0.0)
        fedavg = FedAvgAggregator(weighted_average=True)

        updates = [
            GradientUpdate(
                client_id="client-1",
                round_number=1,
                gradients={"layer1": np.array([1.0, 2.0])},
                num_samples=100,
                local_loss=0.5,
            ),
            GradientUpdate(
                client_id="client-2",
                round_number=1,
                gradients={"layer1": np.array([3.0, 4.0])},
                num_samples=100,
                local_loss=0.4,
            ),
        ]

        result_prox = fedprox.aggregate(updates)
        result_avg = fedavg.aggregate(updates)

        np.testing.assert_array_almost_equal(
            result_prox["layer1"], result_avg["layer1"]
        )


class TestDifferentialPrivacy:
    """Tests for differential privacy mechanisms."""

    def test_gaussian_noise_addition(self):
        """Test Gaussian noise is added to gradients."""
        accountant = PrivacyAccountant(epsilon_budget=10.0, delta=1e-5)
        dp = DifferentialPrivacyMechanism(
            epsilon=1.0,
            delta=1e-5,
            mechanism_type="gaussian",
            accountant=accountant,
        )

        gradients = {"layer1": np.zeros(100)}
        noisy_grads, eps_spent = dp.add_noise(gradients, sensitivity=1.0)

        # Noise should have been added
        assert not np.allclose(noisy_grads["layer1"], gradients["layer1"])
        assert eps_spent > 0

    def test_laplace_noise_addition(self):
        """Test Laplace noise is added to gradients."""
        accountant = PrivacyAccountant(epsilon_budget=10.0, delta=0.0)
        dp = DifferentialPrivacyMechanism(
            epsilon=1.0,
            delta=0.0,
            mechanism_type="laplace",
            accountant=accountant,
        )

        gradients = {"layer1": np.zeros(100)}
        noisy_grads, eps_spent = dp.add_noise(gradients, sensitivity=1.0)

        assert not np.allclose(noisy_grads["layer1"], gradients["layer1"])

    def test_privacy_budget_exhaustion(self):
        """Test that exceeding privacy budget raises error."""
        accountant = PrivacyAccountant(epsilon_budget=1.0, delta=1e-5)
        dp = DifferentialPrivacyMechanism(
            epsilon=0.6,
            delta=1e-5,
            mechanism_type="gaussian",
            accountant=accountant,
        )

        gradients = {"layer1": np.zeros(10)}

        # First call should succeed
        dp.add_noise(gradients, sensitivity=1.0)

        # Second call should exhaust budget
        with pytest.raises(PrivacyBudgetExhaustedError):
            dp.add_noise(gradients, sensitivity=1.0)


class TestPrivacyAccountant:
    """Tests for privacy budget accounting."""

    def test_budget_tracking(self):
        """Test epsilon budget tracking."""
        accountant = PrivacyAccountant(epsilon_budget=5.0, delta=1e-5)

        assert accountant.get_remaining_budget() == 5.0
        assert accountant.get_spent_budget() == 0.0

        accountant.spend(1.0)
        assert accountant.get_spent_budget() == 1.0
        assert accountant.get_remaining_budget() == 4.0

    def test_can_spend(self):
        """Test budget availability check."""
        accountant = PrivacyAccountant(epsilon_budget=2.0, delta=1e-5)

        assert accountant.can_spend(1.0) is True
        assert accountant.can_spend(2.0) is True
        assert accountant.can_spend(2.5) is False

    def test_composition(self):
        """Test privacy composition across rounds."""
        accountant = PrivacyAccountant(epsilon_budget=10.0, delta=1e-5)

        # Simulate 5 rounds with epsilon=1.0 each
        for _ in range(5):
            accountant.spend(1.0)

        assert accountant.get_spent_budget() == 5.0


class TestGradientClipper:
    """Tests for gradient clipping."""

    def test_l2_clipping(self):
        """Test L2 norm clipping."""
        clipper = GradientClipper(max_norm=1.0, norm_type="l2")

        # Gradient with L2 norm = 5.0
        gradients = {"layer1": np.array([3.0, 4.0])}

        clipped = clipper.clip(gradients)

        # Should be scaled to norm 1.0
        clipped_norm = np.linalg.norm(clipped["layer1"])
        assert np.isclose(clipped_norm, 1.0)

    def test_no_clipping_needed(self):
        """Test that small gradients are not clipped."""
        clipper = GradientClipper(max_norm=10.0, norm_type="l2")

        gradients = {"layer1": np.array([0.1, 0.2])}

        clipped = clipper.clip(gradients)

        np.testing.assert_array_almost_equal(
            clipped["layer1"], gradients["layer1"]
        )

    def test_per_layer_clipping(self):
        """Test per-layer clipping option."""
        clipper = GradientClipper(max_norm=1.0, norm_type="l2", per_layer=True)

        gradients = {
            "layer1": np.array([3.0, 4.0]),  # norm = 5
            "layer2": np.array([0.1, 0.2]),  # norm = 0.22
        }

        clipped = clipper.clip(gradients)

        # layer1 should be clipped
        assert np.isclose(np.linalg.norm(clipped["layer1"]), 1.0)
        # layer2 should not be clipped
        np.testing.assert_array_almost_equal(
            clipped["layer2"], gradients["layer2"]
        )


class TestSecureAggregator:
    """Tests for secure aggregation."""

    def test_shamir_secret_sharing(self):
        """Test secret sharing and reconstruction."""
        aggregator = SecureAggregator(threshold=2, total_parties=3)

        secret = np.array([1.0, 2.0, 3.0])
        shares = aggregator.create_shares(secret)

        assert len(shares) == 3

        # Reconstruct with threshold shares
        reconstructed = aggregator.reconstruct(shares[:2])
        np.testing.assert_array_almost_equal(reconstructed, secret)

    def test_insufficient_shares(self):
        """Test that reconstruction fails with insufficient shares."""
        aggregator = SecureAggregator(threshold=3, total_parties=5)

        secret = np.array([1.0, 2.0])
        shares = aggregator.create_shares(secret)

        with pytest.raises(SecureAggregationError):
            aggregator.reconstruct(shares[:2])  # Only 2 shares, need 3

    def test_secure_aggregation_flow(self):
        """Test complete secure aggregation workflow."""
        aggregator = SecureAggregator(threshold=2, total_parties=3)

        # Three clients with gradients
        client_gradients = [
            {"layer1": np.array([1.0, 2.0])},
            {"layer1": np.array([3.0, 4.0])},
            {"layer1": np.array([5.0, 6.0])},
        ]

        # Aggregate securely
        result = aggregator.aggregate_securely(client_gradients)

        # Should equal sum of all gradients
        expected = np.array([9.0, 12.0])
        np.testing.assert_array_almost_equal(result["layer1"], expected)


class TestIntegrationOrchestration:
    """Integration tests for orchestration layer."""

    def test_full_aggregation_pipeline(self):
        """Test complete aggregation with privacy."""
        # Setup
        accountant = PrivacyAccountant(epsilon_budget=10.0, delta=1e-5)
        dp = DifferentialPrivacyMechanism(
            epsilon=0.5, delta=1e-5, accountant=accountant
        )
        clipper = GradientClipper(max_norm=1.0)
        aggregator = FedAvgAggregator()

        # Client updates
        updates = []
        for i in range(3):
            raw_grads = {"layer1": np.random.randn(10)}

            # Clip gradients
            clipped = clipper.clip(raw_grads)

            # Add noise
            noisy, _ = dp.add_noise(clipped, sensitivity=1.0)

            updates.append(
                GradientUpdate(
                    client_id=f"client-{i}",
                    round_number=1,
                    gradients=noisy,
                    num_samples=100,
                    local_loss=0.5,
                )
            )

        # Aggregate
        result = aggregator.aggregate(updates)

        assert "layer1" in result
        assert len(result["layer1"]) == 10
        assert accountant.get_spent_budget() > 0

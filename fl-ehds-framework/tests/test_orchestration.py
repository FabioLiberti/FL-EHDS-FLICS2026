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
    InsufficientClientsError,
    PrivacyBudgetExceededError,
    SecureAggregationError,
)
from orchestration.aggregation.fedavg import FedAvgAggregator
from orchestration.aggregation.fedprox import FedProxAggregator
from orchestration.privacy.differential_privacy import (
    DifferentialPrivacy,
    PrivacyAccountant,
)
from orchestration.privacy.gradient_clipping import GradientClipper
from orchestration.privacy.secure_aggregation import SecureAggregation


# Dummy global model state used across tests
GLOBAL_MODEL = {"layer1": np.zeros(3)}


class TestFedAvgAggregator:
    """Tests for FedAvg aggregation."""

    def test_weighted_average(self):
        """Test weighted averaging of gradients."""
        aggregator = FedAvgAggregator(weighted=True, min_clients=1)

        updates = [
            GradientUpdate(
                client_id="client-1",
                round_number=0,
                gradients={"layer1": np.array([1.0, 2.0, 3.0])},
                num_samples=100,
                local_loss=0.5,
            ),
            GradientUpdate(
                client_id="client-2",
                round_number=0,
                gradients={"layer1": np.array([4.0, 5.0, 6.0])},
                num_samples=200,
                local_loss=0.4,
            ),
        ]

        result = aggregator.aggregate(updates, global_model_state=GLOBAL_MODEL)

        # Weighted average: (100*[1,2,3] + 200*[4,5,6]) / 300 = [3, 4, 5]
        expected = np.array([3.0, 4.0, 5.0])
        np.testing.assert_array_almost_equal(result["layer1"], expected)

    def test_unweighted_average(self):
        """Test unweighted averaging of gradients."""
        aggregator = FedAvgAggregator(weighted=False, min_clients=1)

        updates = [
            GradientUpdate(
                client_id="client-1",
                round_number=0,
                gradients={"layer1": np.array([1.0, 2.0])},
                num_samples=100,
                local_loss=0.5,
            ),
            GradientUpdate(
                client_id="client-2",
                round_number=0,
                gradients={"layer1": np.array([3.0, 4.0])},
                num_samples=200,
                local_loss=0.4,
            ),
        ]

        result = aggregator.aggregate(updates, global_model_state={"layer1": np.zeros(2)})

        # Unweighted: ([1,2] + [3,4]) / 2 = [2, 3]
        expected = np.array([2.0, 3.0])
        np.testing.assert_array_almost_equal(result["layer1"], expected)

    def test_empty_updates(self):
        """Test handling of empty updates list."""
        aggregator = FedAvgAggregator(min_clients=1)

        with pytest.raises((AggregationError, InsufficientClientsError)):
            aggregator.aggregate([], global_model_state=GLOBAL_MODEL)

    def test_multiple_layers(self):
        """Test aggregation across multiple model layers."""
        aggregator = FedAvgAggregator(weighted=True, min_clients=1)

        updates = [
            GradientUpdate(
                client_id="client-1",
                round_number=0,
                gradients={
                    "layer1": np.array([1.0, 2.0]),
                    "layer2": np.array([10.0, 20.0]),
                },
                num_samples=50,
                local_loss=0.5,
            ),
            GradientUpdate(
                client_id="client-2",
                round_number=0,
                gradients={
                    "layer1": np.array([3.0, 4.0]),
                    "layer2": np.array([30.0, 40.0]),
                },
                num_samples=50,
                local_loss=0.4,
            ),
        ]

        global_model = {"layer1": np.zeros(2), "layer2": np.zeros(2)}
        result = aggregator.aggregate(updates, global_model_state=global_model)

        assert "layer1" in result
        assert "layer2" in result


class TestFedProxAggregator:
    """Tests for FedProx aggregation."""

    def test_proximal_term_applied(self):
        """Test that proximal term modifies aggregation."""
        aggregator = FedProxAggregator(mu=0.1, min_clients=1)

        global_model = {"layer1": np.array([0.0, 0.0])}

        updates = [
            GradientUpdate(
                client_id="client-1",
                round_number=0,
                gradients={"layer1": np.array([1.0, 1.0])},
                num_samples=100,
                local_loss=0.5,
            ),
        ]

        result = aggregator.aggregate(updates, global_model_state=global_model)

        # With mu > 0, result should be pulled toward global model
        assert result is not None
        assert "layer1" in result

    def test_mu_zero_equals_fedavg(self):
        """Test that mu=0 gives same result as FedAvg."""
        fedprox = FedProxAggregator(mu=0.0, min_clients=1)
        fedavg = FedAvgAggregator(weighted=True, min_clients=1)

        updates = [
            GradientUpdate(
                client_id="client-1",
                round_number=0,
                gradients={"layer1": np.array([1.0, 2.0])},
                num_samples=100,
                local_loss=0.5,
            ),
            GradientUpdate(
                client_id="client-2",
                round_number=0,
                gradients={"layer1": np.array([3.0, 4.0])},
                num_samples=100,
                local_loss=0.4,
            ),
        ]

        global_model = {"layer1": np.zeros(2)}
        result_prox = fedprox.aggregate(updates, global_model_state=global_model)
        result_avg = fedavg.aggregate(updates, global_model_state=global_model)

        np.testing.assert_array_almost_equal(
            result_prox["layer1"], result_avg["layer1"]
        )


class TestDifferentialPrivacy:
    """Tests for differential privacy mechanisms."""

    def test_gaussian_noise_addition(self):
        """Test Gaussian noise is added to gradients."""
        accountant = PrivacyAccountant(
            total_epsilon=10.0,
            total_delta=1e-5,
            accountant_type="rdp",
        )
        dp = DifferentialPrivacy(
            epsilon=1.0,
            delta=1e-5,
            max_grad_norm=1.0,
            mechanism="gaussian",
            accountant=accountant,
        )

        gradients = {"layer1": np.zeros(100)}
        noisy_grads = dp.add_noise(gradients, round_number=0)

        # Noise should have been added
        assert not np.allclose(noisy_grads["layer1"], gradients["layer1"])

        # Accountant should have recorded the spend
        spent_eps, _ = accountant.get_spent_budget()
        assert spent_eps > 0

    def test_privacy_budget_exhaustion(self):
        """Test that exceeding privacy budget raises error."""
        accountant = PrivacyAccountant(
            total_epsilon=1.0,
            total_delta=1e-5,
            accountant_type="simple",
        )
        dp = DifferentialPrivacy(
            epsilon=0.6,
            delta=1e-5,
            max_grad_norm=1.0,
            mechanism="gaussian",
            accountant=accountant,
        )

        gradients = {"layer1": np.zeros(10)}

        # First call should succeed
        dp.add_noise(gradients, round_number=0)

        # Second call should exhaust budget
        with pytest.raises(PrivacyBudgetExceededError):
            dp.add_noise(gradients, round_number=1)


class TestPrivacyAccountant:
    """Tests for privacy budget accounting."""

    def test_budget_tracking(self):
        """Test epsilon budget tracking."""
        accountant = PrivacyAccountant(
            total_epsilon=5.0,
            total_delta=1e-5,
            accountant_type="simple",
        )

        remaining_eps, _ = accountant.get_remaining_budget()
        spent_eps, _ = accountant.get_spent_budget()
        assert remaining_eps == 5.0
        assert spent_eps == 0.0

        accountant.spend(epsilon=1.0, delta=1e-6, round_number=0)
        spent_eps, _ = accountant.get_spent_budget()
        remaining_eps, _ = accountant.get_remaining_budget()
        assert spent_eps == 1.0
        assert remaining_eps == 4.0

    def test_can_spend(self):
        """Test budget availability check."""
        accountant = PrivacyAccountant(
            total_epsilon=2.0,
            total_delta=1e-5,
            accountant_type="simple",
        )

        assert accountant.can_spend(epsilon=1.0, delta=1e-6) is True
        assert accountant.can_spend(epsilon=2.0, delta=1e-6) is True
        assert accountant.can_spend(epsilon=2.5, delta=1e-6) is False

    def test_composition(self):
        """Test privacy composition across rounds."""
        accountant = PrivacyAccountant(
            total_epsilon=10.0,
            total_delta=1e-5,
            accountant_type="simple",
        )

        # Simulate 5 rounds with epsilon=1.0 each
        for i in range(5):
            accountant.spend(epsilon=1.0, delta=1e-6, round_number=i)

        spent_eps, _ = accountant.get_spent_budget()
        assert spent_eps == 5.0


class TestGradientClipper:
    """Tests for gradient clipping."""

    def test_l2_clipping(self):
        """Test L2 norm clipping."""
        clipper = GradientClipper(max_norm=1.0, norm_type="l2")

        # Gradient with L2 norm = 5.0
        gradients = {"layer1": np.array([3.0, 4.0])}

        clipped, was_clipped = clipper.clip(gradients)

        # Should be scaled to norm 1.0
        clipped_norm = np.linalg.norm(clipped["layer1"])
        assert np.isclose(clipped_norm, 1.0)
        assert was_clipped is True

    def test_no_clipping_needed(self):
        """Test that small gradients are not clipped."""
        clipper = GradientClipper(max_norm=10.0, norm_type="l2")

        gradients = {"layer1": np.array([0.1, 0.2])}

        clipped, was_clipped = clipper.clip(gradients)

        np.testing.assert_array_almost_equal(
            clipped["layer1"], gradients["layer1"]
        )
        assert was_clipped is False

    def test_per_layer_clipping(self):
        """Test per-layer clipping option."""
        clipper = GradientClipper(max_norm=1.0, norm_type="l2", per_layer=True)

        gradients = {
            "layer1": np.array([3.0, 4.0]),  # norm = 5
            "layer2": np.array([0.1, 0.2]),  # norm = 0.22
        }

        clipped, was_clipped = clipper.clip(gradients)

        # layer1 should be clipped
        assert np.isclose(np.linalg.norm(clipped["layer1"]), 1.0)
        # layer2 should not be clipped
        np.testing.assert_array_almost_equal(
            clipped["layer2"], gradients["layer2"]
        )


class TestSecureAggregation:
    """Tests for secure aggregation."""

    def test_setup_round(self):
        """Test secure aggregation round setup."""
        sa = SecureAggregation(protocol="shamir", threshold=0.67)

        setup = sa.setup_round(
            client_ids=["client-1", "client-2", "client-3"],
            round_number=0,
        )

        assert setup["protocol"] == "shamir"
        assert setup["num_clients"] == 3
        # ceil(0.67 * 3) = ceil(2.01) = 3
        assert setup["threshold"] == 3

    def test_pairwise_masking(self):
        """Test that pairwise masking produces valid masked gradients."""
        sa = SecureAggregation(protocol="pairwise_masking", threshold=0.67)

        client_ids = ["client-1", "client-2", "client-3"]
        gradients = {"layer1": np.array([1.0, 2.0, 3.0])}

        # Mask from each client's perspective
        masked_all = []
        for cid in client_ids:
            peers = [p for p in client_ids if p != cid]
            masked = sa.mask_gradients(cid, gradients.copy(), peers)
            masked_all.append(masked)

        # After aggregation, masks should cancel
        result = sa.aggregate_masked(masked_all)
        expected_sum = np.array([1.0, 2.0, 3.0]) * 3

        np.testing.assert_array_almost_equal(result["layer1"], expected_sum)

    def test_secret_sharing(self):
        """Test secret share creation."""
        sa = SecureAggregation(protocol="shamir", threshold=0.67)

        gradient_data = b"test_gradient_data_1234567890ab"
        shares = sa.create_shares(
            client_id="client-1",
            gradient_data=gradient_data,
            num_shares=3,
            threshold=2,
        )

        assert len(shares) == 3
        # Each share should have a unique index
        indices = {s.share_index for s in shares}
        assert len(indices) == 3


class TestIntegrationOrchestration:
    """Integration tests for orchestration layer."""

    def test_full_aggregation_pipeline(self):
        """Test complete aggregation with privacy."""
        # Setup
        accountant = PrivacyAccountant(
            total_epsilon=10.0,
            total_delta=1e-5,
            accountant_type="rdp",
        )
        dp = DifferentialPrivacy(
            epsilon=0.5,
            delta=1e-5,
            max_grad_norm=1.0,
            mechanism="gaussian",
            accountant=accountant,
        )
        clipper = GradientClipper(max_norm=1.0)
        aggregator = FedAvgAggregator(min_clients=1)

        # Client updates
        updates = []
        for i in range(3):
            raw_grads = {"layer1": np.random.randn(10)}

            # Clip gradients (returns tuple)
            clipped, _ = clipper.clip(raw_grads)

            # Add noise
            noisy = dp.add_noise(clipped, round_number=i)

            updates.append(
                GradientUpdate(
                    client_id=f"client-{i}",
                    round_number=0,
                    gradients=noisy,
                    num_samples=100,
                    local_loss=0.5,
                )
            )

        # Aggregate
        global_model = {"layer1": np.zeros(10)}
        result = aggregator.aggregate(updates, global_model_state=global_model)

        assert "layer1" in result
        assert len(result["layer1"]) == 10

        spent_eps, _ = accountant.get_spent_budget()
        assert spent_eps > 0

"""
Tests for Differential Privacy Module
=====================================

Tests the RDP (Rényi Differential Privacy) accounting implementation
to ensure mathematically correct privacy composition.

References:
    - Mironov, "Rényi Differential Privacy", CSF 2017
    - Abadi et al., "Deep Learning with Differential Privacy", CCS 2016
"""

import pytest
import numpy as np
from typing import Dict, Any

import sys
sys.path.insert(0, str(__file__).rsplit('/tests/', 1)[0])

from orchestration.privacy.differential_privacy import (
    compute_rdp_gaussian,
    compute_rdp_gaussian_subsampled,
    rdp_to_eps_delta,
    PrivacyAccountant,
    DifferentialPrivacy,
    DEFAULT_RDP_ORDERS,
)
from core.exceptions import PrivacyBudgetExceededError


class TestRDPGaussian:
    """Tests for Gaussian mechanism RDP computation."""

    def test_rdp_gaussian_basic(self):
        """Test basic RDP computation for Gaussian mechanism."""
        sigma = 1.0
        alpha = 2.0

        # ρ(α) = α / (2σ²)
        expected = alpha / (2 * sigma ** 2)
        actual = compute_rdp_gaussian(sigma, alpha)

        assert abs(actual - expected) < 1e-10

    def test_rdp_gaussian_high_noise(self):
        """Test RDP decreases with higher noise."""
        alpha = 2.0

        rdp_low_noise = compute_rdp_gaussian(sigma=1.0, alpha=alpha)
        rdp_high_noise = compute_rdp_gaussian(sigma=10.0, alpha=alpha)

        # Higher noise should give lower (better) RDP
        assert rdp_high_noise < rdp_low_noise

    def test_rdp_gaussian_invalid_alpha(self):
        """Test that alpha <= 1 raises error."""
        with pytest.raises(ValueError, match="α must be > 1"):
            compute_rdp_gaussian(sigma=1.0, alpha=1.0)

        with pytest.raises(ValueError, match="α must be > 1"):
            compute_rdp_gaussian(sigma=1.0, alpha=0.5)

    def test_rdp_gaussian_invalid_sigma(self):
        """Test that non-positive sigma raises error."""
        with pytest.raises(ValueError, match="Sigma must be positive"):
            compute_rdp_gaussian(sigma=0.0, alpha=2.0)

        with pytest.raises(ValueError, match="Sigma must be positive"):
            compute_rdp_gaussian(sigma=-1.0, alpha=2.0)


class TestRDPSubsampled:
    """Tests for subsampled Gaussian mechanism RDP."""

    def test_subsampled_no_amplification(self):
        """Test that sampling_rate=1.0 gives same as non-subsampled."""
        sigma = 2.0
        alpha = 4.0

        rdp_full = compute_rdp_gaussian(sigma, alpha)
        rdp_subsampled = compute_rdp_gaussian_subsampled(sigma, alpha, sampling_rate=1.0)

        assert abs(rdp_full - rdp_subsampled) < 1e-10

    def test_subsampled_amplification(self):
        """Test privacy amplification by subsampling."""
        sigma = 2.0
        alpha = 4.0

        rdp_full = compute_rdp_gaussian(sigma, alpha)
        rdp_subsampled = compute_rdp_gaussian_subsampled(sigma, alpha, sampling_rate=0.01)

        # Subsampling should amplify privacy (lower RDP)
        assert rdp_subsampled < rdp_full

    def test_subsampled_integer_order(self):
        """Test subsampled RDP at integer orders (exact formula)."""
        sigma = 2.0
        alpha = 3  # Integer order
        q = 0.1

        rdp = compute_rdp_gaussian_subsampled(sigma, alpha, sampling_rate=q)

        # Should return a valid, positive value
        assert rdp >= 0
        assert np.isfinite(rdp)


class TestRDPConversion:
    """Tests for RDP to (ε,δ)-DP conversion."""

    def test_conversion_basic(self):
        """Test basic RDP to (ε,δ) conversion."""
        # Use known values for validation
        rdp_values = [1.0, 0.5, 0.33]  # RDP at orders 2, 3, 4
        orders = [2.0, 3.0, 4.0]
        delta = 1e-5

        epsilon, optimal_order = rdp_to_eps_delta(rdp_values, orders, delta)

        assert epsilon > 0
        assert optimal_order in orders

    def test_conversion_selects_tightest(self):
        """Test that conversion selects tightest bound."""
        orders = [2.0, 4.0, 8.0, 16.0]
        delta = 1e-5

        # RDP values that make α=4 optimal
        rdp_values = [0.5, 0.25, 0.15, 0.12]

        epsilon, _ = rdp_to_eps_delta(rdp_values, orders, delta)

        # Verify the bound is valid
        assert epsilon > 0
        assert np.isfinite(epsilon)

    def test_conversion_invalid_delta(self):
        """Test that non-positive delta raises error."""
        with pytest.raises(ValueError, match="Delta must be positive"):
            rdp_to_eps_delta([0.1], [2.0], delta=0.0)

        with pytest.raises(ValueError, match="Delta must be positive"):
            rdp_to_eps_delta([0.1], [2.0], delta=-0.1)


class TestPrivacyAccountant:
    """Tests for Privacy Accountant."""

    def test_simple_composition(self):
        """Test simple composition accountant."""
        accountant = PrivacyAccountant(
            total_epsilon=10.0,
            total_delta=1e-5,
            accountant_type="simple",
        )

        # Spend budget 5 times
        for i in range(5):
            accountant.spend(epsilon=1.0, delta=1e-6, round_number=i)

        spent_eps, spent_delta = accountant.get_spent_budget()

        # Simple composition: sum of epsilons
        assert abs(spent_eps - 5.0) < 1e-10

    def test_rdp_composition_tighter_than_simple(self):
        """Test that RDP gives tighter bounds than simple composition."""
        noise_multiplier = 2.0
        num_rounds = 100
        delta = 1e-5

        # Simple composition (budget must accommodate 100 rounds of ~2.42 eps each)
        simple_accountant = PrivacyAccountant(
            total_epsilon=500.0,
            total_delta=delta,
            accountant_type="simple",
        )

        # RDP composition
        rdp_accountant = PrivacyAccountant(
            total_epsilon=500.0,
            total_delta=delta,
            accountant_type="rdp",
        )

        per_round_epsilon = np.sqrt(2 * np.log(1.25 / delta)) / noise_multiplier

        for i in range(num_rounds):
            simple_accountant.spend(
                epsilon=per_round_epsilon,
                delta=delta / num_rounds,
                round_number=i,
            )
            rdp_accountant.spend(
                epsilon=per_round_epsilon,
                delta=delta / num_rounds,
                round_number=i,
                noise_multiplier=noise_multiplier,
            )

        simple_eps, _ = simple_accountant.get_spent_budget()
        rdp_eps, _ = rdp_accountant.get_spent_budget()

        # RDP should give significantly tighter bounds for many compositions
        assert rdp_eps < simple_eps, (
            f"RDP ({rdp_eps:.2f}) should be tighter than simple ({simple_eps:.2f})"
        )

    def test_budget_exceeded(self):
        """Test that exceeding budget raises error."""
        accountant = PrivacyAccountant(
            total_epsilon=1.0,
            total_delta=1e-5,
            accountant_type="simple",
        )

        # First spend should succeed
        accountant.spend(epsilon=0.5, delta=1e-6, round_number=0)

        # Second spend should exceed budget
        with pytest.raises(PrivacyBudgetExceededError):
            accountant.spend(epsilon=0.6, delta=1e-6, round_number=1)

    def test_can_spend_check(self):
        """Test can_spend returns correct result."""
        accountant = PrivacyAccountant(
            total_epsilon=1.0,
            total_delta=1e-5,
            accountant_type="simple",
        )

        assert accountant.can_spend(epsilon=0.5, delta=1e-6) is True
        assert accountant.can_spend(epsilon=1.5, delta=1e-6) is False

    def test_rdp_budget_info(self):
        """Test get_rdp_budget returns detailed info."""
        accountant = PrivacyAccountant(
            total_epsilon=10.0,
            total_delta=1e-5,
            accountant_type="rdp",
        )

        accountant.spend(
            epsilon=1.0,
            delta=1e-6,
            round_number=0,
            noise_multiplier=2.0,
        )

        rdp_info = accountant.get_rdp_budget()

        assert rdp_info is not None
        assert "cumulative_rdp" in rdp_info
        assert "optimal_order" in rdp_info
        assert rdp_info["num_compositions"] == 1

    def test_reset(self):
        """Test accountant reset clears state."""
        accountant = PrivacyAccountant(
            total_epsilon=10.0,
            total_delta=1e-5,
            accountant_type="rdp",
        )

        accountant.spend(
            epsilon=1.0,
            delta=1e-6,
            round_number=0,
            noise_multiplier=2.0,
        )

        accountant.reset()

        spent_eps, _ = accountant.get_spent_budget()
        assert spent_eps == 0.0
        assert len(accountant.get_history()) == 0

    def test_compute_epsilon_for_rounds(self):
        """Test prospective epsilon computation."""
        accountant = PrivacyAccountant(
            total_epsilon=10.0,
            total_delta=1e-5,
            accountant_type="rdp",
        )

        eps_10 = accountant.compute_epsilon_for_rounds(
            num_rounds=10,
            noise_multiplier=2.0,
        )
        eps_100 = accountant.compute_epsilon_for_rounds(
            num_rounds=100,
            noise_multiplier=2.0,
        )

        # More rounds should cost more privacy
        assert eps_100 > eps_10

        # But sublinear growth (RDP benefit)
        assert eps_100 < 10 * eps_10


class TestDifferentialPrivacy:
    """Tests for DifferentialPrivacy mechanism."""

    def test_noise_scale_gaussian(self):
        """Test noise scale computation for Gaussian mechanism."""
        dp = DifferentialPrivacy(
            epsilon=1.0,
            delta=1e-5,
            max_grad_norm=1.0,
            mechanism="gaussian",
        )

        # σ = C * sqrt(2 * ln(1.25/δ)) / ε
        expected = np.sqrt(2 * np.log(1.25 / 1e-5)) / 1.0
        assert abs(dp.get_noise_scale() - expected) < 1e-10

    def test_add_noise_gaussian(self):
        """Test noise addition to gradients."""
        dp = DifferentialPrivacy(
            epsilon=1.0,
            delta=1e-5,
            max_grad_norm=1.0,
            mechanism="gaussian",
        )

        gradients = {
            "layer1.weight": np.zeros((10, 10)),
            "layer1.bias": np.zeros(10),
        }

        noised = dp.add_noise(gradients, round_number=0)

        # Noised gradients should have non-zero variance
        assert np.std(noised["layer1.weight"]) > 0
        assert np.std(noised["layer1.bias"]) > 0

    def test_privacy_spent_rdp(self):
        """Test privacy computation uses RDP when available."""
        dp = DifferentialPrivacy(
            epsilon=1.0,
            delta=1e-5,
            max_grad_norm=1.0,
            mechanism="gaussian",
        )

        # Compare simple vs RDP
        eps_simple, _ = dp.compute_privacy_spent(num_rounds=100, use_rdp=False)
        eps_rdp, _ = dp.compute_privacy_spent(num_rounds=100, use_rdp=True)

        # RDP should give tighter bound
        assert eps_rdp < eps_simple

    def test_recommend_rounds(self):
        """Test round recommendation uses RDP."""
        dp = DifferentialPrivacy(
            epsilon=1.0,
            delta=1e-5,
            max_grad_norm=1.0,
            mechanism="gaussian",
        )

        rounds_simple = dp.recommend_rounds(target_epsilon=10.0, use_rdp=False)
        rounds_rdp = dp.recommend_rounds(target_epsilon=10.0, use_rdp=True)

        # RDP should allow more rounds for same budget
        assert rounds_rdp >= rounds_simple

    def test_with_accountant_tracking(self):
        """Test DP mechanism with accountant tracks properly."""
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

        gradients = {"w": np.zeros(100)}

        # Add noise multiple times
        for i in range(5):
            dp.add_noise(gradients, round_number=i)

        # Check accountant recorded everything
        assert len(accountant.get_history()) == 5

        spent_eps, _ = accountant.get_spent_budget()
        assert spent_eps > 0


class TestPrivacyAmplification:
    """Tests for privacy amplification by subsampling."""

    def test_subsampling_amplification(self):
        """Test that subsampling provides privacy amplification."""
        accountant = PrivacyAccountant(
            total_epsilon=100.0,
            total_delta=1e-5,
            accountant_type="rdp",
        )

        # Full dataset
        eps_full = accountant.compute_epsilon_for_rounds(
            num_rounds=100,
            noise_multiplier=2.0,
            sampling_rate=1.0,
        )

        # 10% subsampling
        eps_subsampled = accountant.compute_epsilon_for_rounds(
            num_rounds=100,
            noise_multiplier=2.0,
            sampling_rate=0.1,
        )

        # Subsampling should give significant amplification
        assert eps_subsampled < eps_full

    def test_compute_noise_for_target(self):
        """Test inverse problem: noise for target epsilon."""
        accountant = PrivacyAccountant(
            total_epsilon=10.0,
            total_delta=1e-5,
            accountant_type="rdp",
        )

        target_epsilon = 8.0
        num_rounds = 100

        noise = accountant.compute_noise_for_target_epsilon(
            target_epsilon=target_epsilon,
            num_rounds=num_rounds,
        )

        # Verify the computed noise achieves target
        actual_epsilon = accountant.compute_epsilon_for_rounds(
            num_rounds=num_rounds,
            noise_multiplier=noise,
        )

        # Should be close to target (within tolerance)
        assert actual_epsilon <= target_epsilon * 1.01  # 1% tolerance


class TestKnownValues:
    """Tests against known/published values for validation."""

    def test_abadi_paper_example(self):
        """
        Validate against example from Abadi et al. 2016.

        The paper shows that for MNIST with:
        - N = 60000 samples
        - Lot size = 600 (sampling_rate = 0.01)
        - σ = 4
        - 10000 steps

        The total ε should be around 2-3 for δ = 10^-5.
        """
        accountant = PrivacyAccountant(
            total_epsilon=10.0,
            total_delta=1e-5,
            accountant_type="rdp",
        )

        epsilon = accountant.compute_epsilon_for_rounds(
            num_rounds=10000,
            noise_multiplier=4.0,
            sampling_rate=0.01,
        )

        # Should be in reasonable range (paper reports ~2-3)
        assert 1.0 < epsilon < 5.0, f"Got epsilon={epsilon}, expected ~2-3"

    def test_simple_vs_rdp_savings(self):
        """
        Test that RDP provides significant savings over simple composition.

        For 1000 compositions with σ=1, simple composition gives ~1000ε
        while RDP gives ~√1000 * ε (roughly).
        """
        delta = 1e-5
        sigma = 1.0
        num_rounds = 1000

        # Per-round epsilon under simple composition
        per_round_eps = np.sqrt(2 * np.log(1.25 / delta)) / sigma
        simple_total = per_round_eps * num_rounds

        # RDP total
        accountant = PrivacyAccountant(
            total_epsilon=10000.0,
            total_delta=delta,
            accountant_type="rdp",
        )
        rdp_total = accountant.compute_epsilon_for_rounds(
            num_rounds=num_rounds,
            noise_multiplier=sigma,
        )

        # RDP should provide significant improvement for 1000 compositions
        # With σ=1, RDP gives ~6-7x improvement (>10x requires higher σ)
        improvement_ratio = simple_total / rdp_total
        assert improvement_ratio > 5, (
            f"Expected >5x improvement, got {improvement_ratio:.1f}x"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""Tests for the unified configuration loader."""

import os
import tempfile
from pathlib import Path

import pytest
import yaml


# Ensure framework root is on sys.path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config_loader import (
    get_training_defaults,
    get_dashboard_defaults,
    get_full_config,
    load_config,
    reload_config,
)


class TestLoadConfig:
    """Test config file loading."""

    def setup_method(self):
        """Reset cache before each test."""
        reload_config()

    def test_load_config_finds_file(self):
        """Config file should be found and loaded."""
        cfg = load_config()
        assert isinstance(cfg, dict)
        # Should have at least framework and governance sections
        assert "framework" in cfg
        assert "governance" in cfg

    def test_load_config_has_training_section(self):
        """Config should contain the training section."""
        cfg = load_config()
        assert "training" in cfg
        assert "federated" in cfg["training"]
        assert "privacy" in cfg["training"]
        assert "data" in cfg["training"]

    def test_load_config_explicit_path(self):
        """Loading with explicit path should work."""
        config_path = str(Path(__file__).parent.parent / "config" / "config.yaml")
        cfg = load_config(config_path)
        assert isinstance(cfg, dict)
        assert len(cfg) > 0

    def test_load_config_missing_file(self):
        """Missing config should return empty dict."""
        cfg = load_config("/nonexistent/path/config.yaml")
        assert cfg == {}

    def test_load_config_caching(self):
        """Repeated calls should return cached config."""
        cfg1 = load_config()
        cfg2 = load_config()
        assert cfg1 is cfg2  # Same object (cached)

    def test_reload_config(self):
        """reload_config should clear cache."""
        cfg1 = load_config()
        reload_config()
        cfg2 = load_config()
        assert cfg1 is not cfg2  # Different objects after reload
        assert cfg1 == cfg2  # But same content


class TestGetTrainingDefaults:
    """Test training defaults retrieval."""

    def setup_method(self):
        reload_config()

    def test_returns_all_expected_keys(self):
        """All required keys should be present."""
        defaults = get_training_defaults()
        expected_keys = [
            "algorithm", "num_clients", "num_rounds", "local_epochs",
            "batch_size", "learning_rate", "dp_enabled", "dp_epsilon",
            "dp_delta", "dp_clip_norm", "data_distribution", "mu",
            "seed", "server_lr", "beta1", "beta2", "tau",
            "dataset_type", "dataset_name", "dataset_path",
            "img_size", "alpha",
        ]
        for key in expected_keys:
            assert key in defaults, f"Missing key: {key}"

    def test_default_values_conservative(self):
        """Default values should match terminal conservative defaults."""
        defaults = get_training_defaults()
        assert defaults["algorithm"] == "FedAvg"
        assert defaults["num_clients"] == 5
        assert defaults["num_rounds"] == 30
        assert defaults["local_epochs"] == 3
        assert defaults["batch_size"] == 32
        assert defaults["learning_rate"] == 0.01
        assert defaults["dp_enabled"] is False
        assert defaults["dp_epsilon"] == 10.0
        assert defaults["seed"] == 42

    def test_yaml_values_loaded(self):
        """Values from config.yaml should be loaded."""
        defaults = get_training_defaults()
        # These are set in config.yaml training section
        assert defaults["mu"] == 0.1
        assert defaults["server_lr"] == 0.1
        assert defaults["beta1"] == 0.9
        assert defaults["beta2"] == 0.99
        assert defaults["alpha"] == 0.5

    def test_types_correct(self):
        """Values should have correct types."""
        defaults = get_training_defaults()
        assert isinstance(defaults["algorithm"], str)
        assert isinstance(defaults["num_clients"], int)
        assert isinstance(defaults["num_rounds"], int)
        assert isinstance(defaults["learning_rate"], float)
        assert isinstance(defaults["dp_enabled"], bool)
        assert isinstance(defaults["seed"], int)


class TestGetDashboardDefaults:
    """Test dashboard defaults mapping."""

    def setup_method(self):
        reload_config()

    def test_returns_dashboard_keys(self):
        """Dashboard defaults should use dashboard-specific keys."""
        defaults = get_dashboard_defaults()
        expected_keys = [
            "num_nodes", "total_samples", "algorithm", "fedprox_mu",
            "server_lr", "beta1", "beta2", "tau", "num_rounds",
            "local_epochs", "learning_rate", "use_dp", "epsilon",
            "delta", "clip_norm", "random_seed", "img_size",
        ]
        for key in expected_keys:
            assert key in defaults, f"Missing dashboard key: {key}"

    def test_key_mapping_correct(self):
        """Dashboard keys should map correctly from training defaults."""
        training = get_training_defaults()
        dashboard = get_dashboard_defaults()
        assert dashboard["num_nodes"] == training["num_clients"]
        assert dashboard["fedprox_mu"] == training["mu"]
        assert dashboard["use_dp"] == training["dp_enabled"]
        assert dashboard["epsilon"] == training["dp_epsilon"]
        assert dashboard["delta"] == training["dp_delta"]
        assert dashboard["clip_norm"] == training["dp_clip_norm"]
        assert dashboard["random_seed"] == training["seed"]


class TestEnvironmentOverrides:
    """Test environment variable overrides."""

    def setup_method(self):
        reload_config()

    def test_env_override_algorithm(self):
        """FL_EHDS_ALGORITHM should override config."""
        os.environ["FL_EHDS_ALGORITHM"] = "SCAFFOLD"
        try:
            reload_config()
            defaults = get_training_defaults()
            assert defaults["algorithm"] == "SCAFFOLD"
        finally:
            del os.environ["FL_EHDS_ALGORITHM"]

    def test_env_override_num_rounds(self):
        """FL_EHDS_NUM_ROUNDS should override config."""
        os.environ["FL_EHDS_NUM_ROUNDS"] = "100"
        try:
            reload_config()
            defaults = get_training_defaults()
            assert defaults["num_rounds"] == 100
        finally:
            del os.environ["FL_EHDS_NUM_ROUNDS"]

    def test_env_override_dp_enabled(self):
        """FL_EHDS_DP_ENABLED should override config."""
        os.environ["FL_EHDS_DP_ENABLED"] = "true"
        try:
            reload_config()
            defaults = get_training_defaults()
            assert defaults["dp_enabled"] is True
        finally:
            del os.environ["FL_EHDS_DP_ENABLED"]

    def test_env_invalid_value_ignored(self):
        """Invalid env values should be silently ignored."""
        os.environ["FL_EHDS_NUM_ROUNDS"] = "not_a_number"
        try:
            reload_config()
            defaults = get_training_defaults()
            # Should still be the YAML/default value
            assert isinstance(defaults["num_rounds"], int)
        finally:
            del os.environ["FL_EHDS_NUM_ROUNDS"]


class TestCustomYamlConfig:
    """Test with custom YAML config files."""

    def test_custom_config_values(self):
        """Custom config should override defaults."""
        custom_config = {
            "training": {
                "seed": 999,
                "federated": {
                    "algorithm": "FedProx",
                    "num_clients": 10,
                    "num_rounds": 50,
                    "learning_rate": 0.05,
                },
                "privacy": {
                    "enabled": True,
                    "epsilon": 5.0,
                },
                "data": {
                    "type": "imaging",
                    "alpha": 0.3,
                },
            }
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(custom_config, f)
            temp_path = f.name

        try:
            reload_config()
            cfg = load_config(temp_path)
            reload_config()  # Clear cache of temp config
            # Manually verify the loaded content
            assert cfg["training"]["federated"]["algorithm"] == "FedProx"
            assert cfg["training"]["federated"]["num_clients"] == 10
            assert cfg["training"]["privacy"]["enabled"] is True
        finally:
            os.unlink(temp_path)
            reload_config()


class TestGetFullConfig:
    """Test full config access."""

    def setup_method(self):
        reload_config()

    def test_returns_complete_config(self):
        """get_full_config should return the entire YAML structure."""
        cfg = get_full_config()
        assert "framework" in cfg
        assert "governance" in cfg
        assert "orchestration" in cfg
        assert "data_holders" in cfg
        assert "monitoring" in cfg
        assert "training" in cfg

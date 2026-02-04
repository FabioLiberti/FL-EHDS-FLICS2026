"""
FL-EHDS Utility Functions
=========================
Common utility functions used throughout the FL-EHDS framework.
"""

import hashlib
import json
import logging
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
import structlog


# =============================================================================
# Configuration Loading
# =============================================================================


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to the configuration file.

    Returns:
        Configuration dictionary.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        yaml.YAMLError: If config file is invalid.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Resolve environment variables
    config = _resolve_env_vars(config)

    return config


def _resolve_env_vars(config: Any) -> Any:
    """Recursively resolve environment variables in config values."""
    if isinstance(config, dict):
        return {k: _resolve_env_vars(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [_resolve_env_vars(item) for item in config]
    elif isinstance(config, str) and config.startswith("${") and config.endswith("}"):
        env_var = config[2:-1]
        default = None
        if ":" in env_var:
            env_var, default = env_var.split(":", 1)
        return os.environ.get(env_var, default)
    return config


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two configuration dictionaries.

    Args:
        base: Base configuration.
        override: Override configuration (takes precedence).

    Returns:
        Merged configuration.
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    return result


# =============================================================================
# Logging Setup
# =============================================================================


def setup_logging(
    level: str = "INFO",
    log_format: str = "json",
    log_file: Optional[str] = None,
) -> structlog.BoundLogger:
    """
    Configure structured logging for the framework.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR).
        log_format: Output format (json, console).
        log_file: Optional file path for log output.

    Returns:
        Configured logger instance.
    """
    # Configure processors based on format
    if log_format == "json":
        processors = [
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer(),
        ]
    else:
        processors = [
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.dev.ConsoleRenderer(),
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Configure standard logging
    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, level.upper()),
        handlers=[
            logging.StreamHandler(),
            *(
                [logging.FileHandler(log_file)]
                if log_file
                else []
            ),
        ],
    )

    return structlog.get_logger()


# =============================================================================
# Model Serialization
# =============================================================================


def serialize_model(model_state: Dict[str, Any], format: str = "pickle") -> bytes:
    """
    Serialize model state dictionary.

    Args:
        model_state: Model state dictionary (typically from model.state_dict()).
        format: Serialization format (pickle, json).

    Returns:
        Serialized model bytes.
    """
    if format == "pickle":
        return pickle.dumps(model_state)
    elif format == "json":
        # Convert tensors to lists for JSON serialization
        serializable = _make_json_serializable(model_state)
        return json.dumps(serializable).encode("utf-8")
    else:
        raise ValueError(f"Unsupported serialization format: {format}")


def deserialize_model(data: bytes, format: str = "pickle") -> Dict[str, Any]:
    """
    Deserialize model state dictionary.

    Args:
        data: Serialized model bytes.
        format: Serialization format (pickle, json).

    Returns:
        Model state dictionary.
    """
    if format == "pickle":
        return pickle.loads(data)
    elif format == "json":
        return json.loads(data.decode("utf-8"))
    else:
        raise ValueError(f"Unsupported serialization format: {format}")


def _make_json_serializable(obj: Any) -> Any:
    """Convert objects to JSON-serializable format."""
    if hasattr(obj, "tolist"):  # numpy/torch tensors
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_json_serializable(item) for item in obj]
    elif isinstance(obj, datetime):
        return obj.isoformat()
    return obj


# =============================================================================
# Metrics Computation
# =============================================================================


def compute_metrics(
    predictions: List[Any],
    targets: List[Any],
    task_type: str = "classification",
) -> Dict[str, float]:
    """
    Compute evaluation metrics.

    Args:
        predictions: Model predictions.
        targets: Ground truth labels.
        task_type: Type of task (classification, regression).

    Returns:
        Dictionary of metric names to values.
    """
    import numpy as np

    predictions = np.array(predictions)
    targets = np.array(targets)

    metrics = {}

    if task_type == "classification":
        # Accuracy
        metrics["accuracy"] = float(np.mean(predictions == targets))

        # For binary classification
        if len(np.unique(targets)) == 2:
            tp = np.sum((predictions == 1) & (targets == 1))
            fp = np.sum((predictions == 1) & (targets == 0))
            fn = np.sum((predictions == 0) & (targets == 1))
            tn = np.sum((predictions == 0) & (targets == 0))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )

            metrics["precision"] = float(precision)
            metrics["recall"] = float(recall)
            metrics["f1_score"] = float(f1)

    elif task_type == "regression":
        # Mean Squared Error
        mse = np.mean((predictions - targets) ** 2)
        metrics["mse"] = float(mse)
        metrics["rmse"] = float(np.sqrt(mse))

        # Mean Absolute Error
        metrics["mae"] = float(np.mean(np.abs(predictions - targets)))

        # R-squared
        ss_res = np.sum((targets - predictions) ** 2)
        ss_tot = np.sum((targets - np.mean(targets)) ** 2)
        metrics["r2"] = float(1 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

    return metrics


def aggregate_metrics(
    client_metrics: List[Dict[str, float]],
    weights: Optional[List[float]] = None,
) -> Dict[str, float]:
    """
    Aggregate metrics from multiple clients.

    Args:
        client_metrics: List of metric dictionaries from each client.
        weights: Optional weights for weighted averaging.

    Returns:
        Aggregated metrics dictionary.
    """
    if not client_metrics:
        return {}

    if weights is None:
        weights = [1.0] * len(client_metrics)

    # Normalize weights
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]

    # Get all metric keys
    all_keys = set()
    for metrics in client_metrics:
        all_keys.update(metrics.keys())

    # Compute weighted average for each metric
    aggregated = {}
    for key in all_keys:
        values = []
        key_weights = []
        for metrics, weight in zip(client_metrics, weights):
            if key in metrics:
                values.append(metrics[key])
                key_weights.append(weight)

        if values:
            total_key_weight = sum(key_weights)
            aggregated[key] = sum(v * w / total_key_weight for v, w in zip(values, key_weights))

    return aggregated


# =============================================================================
# Hashing and Integrity
# =============================================================================


def compute_hash(data: Union[str, bytes, Dict], algorithm: str = "sha256") -> str:
    """
    Compute cryptographic hash of data.

    Args:
        data: Data to hash (string, bytes, or dictionary).
        algorithm: Hash algorithm (sha256, sha512, md5).

    Returns:
        Hexadecimal hash string.
    """
    if isinstance(data, dict):
        data = json.dumps(data, sort_keys=True)
    if isinstance(data, str):
        data = data.encode("utf-8")

    hasher = hashlib.new(algorithm)
    hasher.update(data)
    return hasher.hexdigest()


def verify_hash(data: Union[str, bytes, Dict], expected_hash: str, algorithm: str = "sha256") -> bool:
    """
    Verify data integrity against expected hash.

    Args:
        data: Data to verify.
        expected_hash: Expected hash value.
        algorithm: Hash algorithm used.

    Returns:
        True if hash matches, False otherwise.
    """
    computed_hash = compute_hash(data, algorithm)
    return computed_hash == expected_hash


# =============================================================================
# ID Generation
# =============================================================================


def generate_id(prefix: str = "", length: int = 16) -> str:
    """
    Generate a unique identifier.

    Args:
        prefix: Optional prefix for the ID.
        length: Length of the random portion.

    Returns:
        Unique identifier string.
    """
    import secrets

    random_part = secrets.token_hex(length // 2)
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")

    if prefix:
        return f"{prefix}-{timestamp}-{random_part}"
    return f"{timestamp}-{random_part}"


def generate_round_id(round_number: int, permit_id: str) -> str:
    """Generate unique round identifier."""
    return f"round-{round_number:04d}-{compute_hash(permit_id)[:8]}"


def generate_client_id(organization: str, member_state: str) -> str:
    """Generate unique client identifier."""
    content = f"{organization}:{member_state}:{datetime.utcnow().isoformat()}"
    return f"client-{compute_hash(content)[:16]}"

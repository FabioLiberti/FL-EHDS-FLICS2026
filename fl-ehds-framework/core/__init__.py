"""
FL-EHDS Core Module
===================
Core utilities, models, and base classes for the FL-EHDS framework.

Includes:
- FL Algorithms (FedAvg, FedProx, SCAFFOLD, FedAdam, etc.)
- Data models and configuration classes
- Utility functions
"""

# FL Algorithms
try:
    from .fl_algorithms import (
        ALGORITHM_INFO,
        FLAlgorithm,
        FedAvg,
        FedProx,
        SCAFFOLD,
        FedAdam,
        FedYogi,
        FedAdagrad,
        FedNova,
        FedDyn,
        Ditto,
        create_algorithm
    )
except ImportError:
    pass  # fl_algorithms may not be available

from .models import (
    DataPermit,
    FLClient,
    FLRound,
    GradientUpdate,
    ModelCheckpoint,
    TrainingConfig,
    PrivacyConfig,
    ComplianceRecord,
)
from .utils import (
    load_config,
    setup_logging,
    compute_metrics,
    serialize_model,
    deserialize_model,
)
from .exceptions import (
    FLEHDSError,
    PermitError,
    OptOutError,
    PrivacyBudgetExceededError,
    ComplianceViolationError,
    CommunicationError,
)

__all__ = [
    # Models
    "DataPermit",
    "FLClient",
    "FLRound",
    "GradientUpdate",
    "ModelCheckpoint",
    "TrainingConfig",
    "PrivacyConfig",
    "ComplianceRecord",
    # Utils
    "load_config",
    "setup_logging",
    "compute_metrics",
    "serialize_model",
    "deserialize_model",
    # Exceptions
    "FLEHDSError",
    "PermitError",
    "OptOutError",
    "PrivacyBudgetExceededError",
    "ComplianceViolationError",
    "CommunicationError",
]

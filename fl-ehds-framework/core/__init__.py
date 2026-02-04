"""
FL-EHDS Core Module
===================
Core utilities, models, and base classes for the FL-EHDS framework.
"""

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

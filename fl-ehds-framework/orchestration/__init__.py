"""
FL-EHDS Orchestration Layer (Layer 2)
=====================================
FL orchestration components including aggregation algorithms,
privacy protection mechanisms, and compliance enforcement.
"""

from .aggregation import FedAvgAggregator, FedProxAggregator, BaseAggregator
from .privacy import (
    DifferentialPrivacy,
    GradientClipper,
    SecureAggregation,
    PrivacyAccountant,
)
from .compliance import PurposeLimiter, OutputController

__all__ = [
    # Aggregation
    "BaseAggregator",
    "FedAvgAggregator",
    "FedProxAggregator",
    # Privacy
    "DifferentialPrivacy",
    "GradientClipper",
    "SecureAggregation",
    "PrivacyAccountant",
    # Compliance
    "PurposeLimiter",
    "OutputController",
]

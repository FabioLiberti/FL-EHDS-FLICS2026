"""
FL-EHDS Governance Layer (Layer 1)
==================================
EHDS compliance components including HDAB integration, data permits,
opt-out registry, and compliance logging.
"""

from .hdab_integration import HDABClient, HDABConfig
from .data_permits import DataPermitManager, PermitValidator
from .optout_registry import OptOutRegistry, OptOutChecker
from .compliance_logging import ComplianceLogger, AuditTrail

__all__ = [
    # HDAB Integration
    "HDABClient",
    "HDABConfig",
    # Data Permits
    "DataPermitManager",
    "PermitValidator",
    # Opt-out Registry
    "OptOutRegistry",
    "OptOutChecker",
    # Compliance Logging
    "ComplianceLogger",
    "AuditTrail",
]

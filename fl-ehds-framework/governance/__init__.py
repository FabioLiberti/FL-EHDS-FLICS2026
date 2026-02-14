"""
FL-EHDS Governance Layer (Layer 1)
==================================
EHDS compliance components including HDAB integration, data permits,
opt-out registry, and compliance logging.
"""

from .hdab_integration import (
    HDABClient,
    HDABConfig,
    AuthToken,
    PermitStore,
    MultiHDABCoordinator,
    get_shared_permit_store,
)
from .data_permits import DataPermitManager, PermitValidator
from .optout_registry import (
    OptOutRegistry,
    OptOutChecker,
    OptOutCacheEntry,
    RegistryStats,
)
from .compliance_logging import ComplianceLogger, AuditTrail
from .persistence import GovernanceDB

__all__ = [
    # HDAB Integration
    "HDABClient",
    "HDABConfig",
    "AuthToken",
    "PermitStore",
    "MultiHDABCoordinator",
    "get_shared_permit_store",
    # Data Permits
    "DataPermitManager",
    "PermitValidator",
    # Opt-out Registry
    "OptOutRegistry",
    "OptOutChecker",
    "OptOutCacheEntry",
    "RegistryStats",
    # Compliance Logging
    "ComplianceLogger",
    "AuditTrail",
    # Persistence
    "GovernanceDB",
]

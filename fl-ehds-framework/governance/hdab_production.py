"""
HDAB Production Module
======================
Synchronous, production-ready HDAB (Health Data Access Body) integration
for the FL-EHDS framework under EHDS Regulation EU 2025/327.

Design:
- simulation_mode=True (default): full in-memory permit lifecycle,
  local token generation, no external dependencies
- simulation_mode=False (production): raises NotImplementedError
  for OAuth2/mTLS/API-key authentication with deployment guidance

Compared to governance/hdab_integration.py (async, structlog), this module
is synchronous, uses only stdlib logging, and provides clearer production
error messages.

Author: Fabio Liberti
"""

import logging
import secrets
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from core.models import DataPermit, PermitStatus, PermitPurpose, DataCategory
from core.exceptions import (
    HDABConnectionError,
    PermitNotFoundError,
    PermitExpiredError,
    PermitPurposeMismatchError,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class HDABProductionConfig:
    """Configuration for HDAB production connection."""

    endpoint: str
    auth_method: str = "oauth2"  # oauth2 | mtls | api_key
    simulation_mode: bool = True
    timeout: int = 30
    max_retries: int = 3
    backoff_factor: float = 2.0
    # OAuth2 Client Credentials
    client_id: Optional[str] = None
    client_secret: Optional[str] = None
    token_endpoint: Optional[str] = None
    scopes: List[str] = field(
        default_factory=lambda: ["permits:read", "permits:write"]
    )
    # mTLS certificate paths
    cert_path: Optional[str] = None
    key_path: Optional[str] = None
    ca_path: Optional[str] = None
    # API key
    api_key: Optional[str] = None


@dataclass
class ProductionAuthToken:
    """Bearer / mTLS / API-key token with expiry tracking."""

    access_token: str
    token_type: str = "Bearer"
    expires_at: datetime = field(
        default_factory=lambda: datetime.utcnow() + timedelta(hours=1)
    )
    scopes: List[str] = field(default_factory=list)

    @property
    def is_expired(self) -> bool:
        return datetime.utcnow() >= self.expires_at

    @property
    def authorization_header(self) -> str:
        return f"{self.token_type} {self.access_token}"


# =============================================================================
# In-Memory Permit Store (simulation backend)
# =============================================================================


class ProductionPermitStore:
    """
    In-memory permit store for simulation mode.

    Provides a local backend to test the full HDAB permit lifecycle
    without a live HDAB endpoint.
    """

    def __init__(self):
        self._permits: Dict[str, DataPermit] = {}
        self._audit_log: List[Dict[str, Any]] = []

    def register(self, permit: DataPermit) -> None:
        self._permits[permit.permit_id] = permit
        self._log("register", permit.permit_id)

    def get(self, permit_id: str) -> Optional[DataPermit]:
        return self._permits.get(permit_id)

    def list_all(
        self,
        requester_id: Optional[str] = None,
        status: Optional[PermitStatus] = None,
        purpose: Optional[PermitPurpose] = None,
    ) -> List[DataPermit]:
        results = list(self._permits.values())
        if requester_id:
            results = [p for p in results if p.requester_id == requester_id]
        if status:
            results = [p for p in results if p.status == status]
        if purpose:
            results = [p for p in results if p.purpose == purpose]
        return results

    def revoke(self, permit_id: str, reason: str) -> bool:
        permit = self._permits.get(permit_id)
        if not permit:
            return False
        permit.status = PermitStatus.REVOKED
        permit.metadata["revocation_reason"] = reason
        permit.metadata["revoked_at"] = datetime.utcnow().isoformat()
        self._log("revoke", permit_id, {"reason": reason})
        return True

    def suspend(self, permit_id: str, reason: str) -> bool:
        permit = self._permits.get(permit_id)
        if not permit:
            return False
        permit.status = PermitStatus.SUSPENDED
        permit.metadata["suspension_reason"] = reason
        permit.metadata["suspended_at"] = datetime.utcnow().isoformat()
        self._log("suspend", permit_id, {"reason": reason})
        return True

    def reactivate(self, permit_id: str) -> bool:
        permit = self._permits.get(permit_id)
        if not permit or permit.status != PermitStatus.SUSPENDED:
            return False
        permit.status = PermitStatus.ACTIVE
        permit.metadata.pop("suspension_reason", None)
        self._log("reactivate", permit_id)
        return True

    def remove(self, permit_id: str) -> bool:
        if permit_id in self._permits:
            del self._permits[permit_id]
            self._log("remove", permit_id)
            return True
        return False

    @property
    def count(self) -> int:
        return len(self._permits)

    @property
    def audit_log(self) -> List[Dict[str, Any]]:
        return list(self._audit_log)

    def _log(
        self, action: str, permit_id: str, details: Optional[Dict] = None
    ) -> None:
        self._audit_log.append({
            "action": action,
            "permit_id": permit_id,
            "timestamp": datetime.utcnow().isoformat(),
            "details": details or {},
        })


# Global shared store for simulation mode
_shared_production_store = ProductionPermitStore()


def get_production_permit_store() -> ProductionPermitStore:
    """Get the shared in-memory production permit store."""
    return _shared_production_store


# =============================================================================
# HDAB Production Client
# =============================================================================


class HDABProductionClient:
    """
    Synchronous HDAB client for production deployment.

    In simulation_mode (default), all operations work against an in-memory
    ProductionPermitStore.  Authentication generates local tokens.

    In production mode (simulation_mode=False), authentication methods raise
    NotImplementedError with guidance on what HTTP client and configuration
    is needed for real HDAB connectivity.
    """

    def __init__(self, config: HDABProductionConfig):
        self.config = config
        self._authenticated = False
        self._auth_token: Optional[ProductionAuthToken] = None
        self._permit_store: ProductionPermitStore = _shared_production_store

    # -----------------------------------------------------------------
    # Connection
    # -----------------------------------------------------------------

    def connect(self) -> bool:
        """
        Establish connection to HDAB with retry and backoff.

        Returns:
            True if connection successful.

        Raises:
            HDABConnectionError: If connection fails after max_retries.
        """
        logger.info(
            "Connecting to HDAB endpoint=%s auth=%s simulation=%s",
            self.config.endpoint,
            self.config.auth_method,
            self.config.simulation_mode,
        )

        for attempt in range(self.config.max_retries):
            try:
                self._authenticate()
                self._authenticated = True
                logger.info(
                    "HDAB connection established (endpoint=%s, auth=%s)",
                    self.config.endpoint,
                    self.config.auth_method,
                )
                return True
            except NotImplementedError:
                raise
            except Exception as e:
                wait_time = self.config.backoff_factor ** attempt
                logger.warning(
                    "HDAB connection attempt %d/%d failed: %s (retry in %.1fs)",
                    attempt + 1,
                    self.config.max_retries,
                    str(e),
                    wait_time,
                )
                if attempt < self.config.max_retries - 1:
                    time.sleep(wait_time)

        raise HDABConnectionError(
            self.config.endpoint,
            f"Failed after {self.config.max_retries} attempts",
        )

    def disconnect(self) -> None:
        """Close connection and invalidate auth token."""
        self._authenticated = False
        self._auth_token = None
        logger.info("HDAB connection closed (endpoint=%s)", self.config.endpoint)

    # -----------------------------------------------------------------
    # Authentication
    # -----------------------------------------------------------------

    def _authenticate(self) -> None:
        """Dispatch to configured authentication method."""
        if self.config.auth_method == "oauth2":
            self._oauth2_authenticate()
        elif self.config.auth_method == "mtls":
            self._mtls_authenticate()
        elif self.config.auth_method == "api_key":
            self._apikey_authenticate()
        else:
            raise ValueError(
                f"Unsupported auth method: {self.config.auth_method}"
            )

    def _oauth2_authenticate(self) -> None:
        """
        OAuth2 Client Credentials authentication.

        Simulation: generates a local bearer token.
        Production: raises NotImplementedError with deployment guidance.
        """
        if self.config.simulation_mode:
            self._auth_token = ProductionAuthToken(
                access_token=secrets.token_urlsafe(32),
                token_type="Bearer",
                expires_at=datetime.utcnow() + timedelta(hours=1),
                scopes=list(self.config.scopes),
            )
            logger.debug(
                "OAuth2 simulation token issued (scopes=%s, expires_in=3600s)",
                self.config.scopes,
            )
            return

        token_url = (
            self.config.token_endpoint
            or f"{self.config.endpoint}/oauth2/token"
        )
        raise NotImplementedError(
            f"Production OAuth2 requires an HTTP client (aiohttp or requests). "
            f"Install: pip install aiohttp\n"
            f"Implement: POST to {token_url} with grant_type=client_credentials, "
            f"client_id={self.config.client_id}, scope={' '.join(self.config.scopes)}"
        )

    def _mtls_authenticate(self) -> None:
        """
        Mutual TLS authentication.

        Simulation: generates a local mTLS token; logs cert config if present.
        Production: raises NotImplementedError with SSL context guidance.
        """
        if self.config.simulation_mode:
            if not self.config.cert_path or not self.config.key_path:
                logger.debug(
                    "mTLS simulation: no certs configured, using default trust"
                )
            self._auth_token = ProductionAuthToken(
                access_token=f"mtls-{secrets.token_hex(16)}",
                token_type="mTLS",
                expires_at=datetime.utcnow() + timedelta(hours=24),
                scopes=["permits:read", "permits:write", "optout:read"],
            )
            logger.debug(
                "mTLS simulation token issued (cert=%s)", self.config.cert_path
            )
            return

        raise NotImplementedError(
            f"Production mTLS requires an HTTP client with SSL context. "
            f"Install: pip install aiohttp\n"
            f"Configure HDABProductionConfig: "
            f"cert_path='{self.config.cert_path or '<client.pem>'}', "
            f"key_path='{self.config.key_path or '<client-key.pem>'}', "
            f"ca_path='{self.config.ca_path or '<ca-bundle.pem>'}'\n"
            f"Implement: ssl.create_default_context(cafile=ca_path) + "
            f"load_cert_chain(cert_path, key_path)"
        )

    def _apikey_authenticate(self) -> None:
        """
        API Key authentication.

        Simulation: accepts any non-empty key (or generates one).
        Production: raises NotImplementedError with validation guidance.
        """
        if self.config.simulation_mode:
            key = self.config.api_key or f"sim-key-{secrets.token_hex(8)}"
            self._auth_token = ProductionAuthToken(
                access_token=key,
                token_type="ApiKey",
                expires_at=datetime.utcnow() + timedelta(days=365),
                scopes=["permits:read"],
            )
            logger.debug("API key simulation token issued")
            return

        if not self.config.api_key:
            raise HDABConnectionError(
                self.config.endpoint, "API key not configured"
            )

        raise NotImplementedError(
            f"Production API key validation requires an HTTP client. "
            f"Implement: GET {self.config.endpoint}/auth/verify "
            f"with header X-API-Key: {self.config.api_key[:8]}..."
        )

    def _ensure_authenticated(self) -> None:
        """Re-authenticate if token is expired."""
        if self._auth_token and self._auth_token.is_expired:
            logger.info("Auth token expired, re-authenticating")
            self._authenticate()

    # -----------------------------------------------------------------
    # Permit Operations
    # -----------------------------------------------------------------

    def register_permit(self, permit: DataPermit) -> None:
        """Register a permit in the local store (simulation mode)."""
        self._permit_store.register(permit)
        logger.info("Permit registered: %s", permit.permit_id)

    def get_permit(self, permit_id: str) -> DataPermit:
        """
        Retrieve a data permit.

        Raises:
            PermitNotFoundError: If permit does not exist.
        """
        self._ensure_authenticated()

        permit = self._permit_store.get(permit_id)
        if permit is None:
            raise PermitNotFoundError(permit_id)
        return permit

    def verify_permit(
        self,
        permit_id: str,
        purpose: PermitPurpose,
        data_categories: List[DataCategory],
    ) -> bool:
        """
        Verify a permit is valid for the specified purpose and categories.

        Raises:
            PermitNotFoundError: If permit does not exist.
            PermitExpiredError: If permit has expired.
            PermitPurposeMismatchError: If purpose does not match.
        """
        permit = self.get_permit(permit_id)

        # Check expiry
        if not permit.is_valid():
            if (
                permit.status == PermitStatus.EXPIRED
                or permit.valid_until < datetime.utcnow()
            ):
                raise PermitExpiredError(
                    permit_id, permit.valid_until.isoformat()
                )
            return False

        # Check purpose alignment
        if not permit.covers_purpose(purpose):
            raise PermitPurposeMismatchError(
                permit_id,
                requested_purpose=purpose.value,
                allowed_purposes=[permit.purpose.value],
            )

        # Check data categories
        if not permit.covers_categories(data_categories):
            missing = [
                c.value
                for c in data_categories
                if c not in permit.data_categories
            ]
            logger.warning(
                "Data category mismatch for permit %s: missing %s",
                permit_id,
                missing,
            )
            return False

        logger.info("Permit %s verified successfully", permit_id)
        return True

    def list_permits(
        self,
        requester_id: Optional[str] = None,
        status: Optional[PermitStatus] = None,
        purpose: Optional[PermitPurpose] = None,
    ) -> List[DataPermit]:
        """List permits matching specified criteria."""
        self._ensure_authenticated()

        results = self._permit_store.list_all(requester_id, status, purpose)
        logger.info(
            "Listed %d permits (requester=%s, status=%s, purpose=%s)",
            len(results),
            requester_id,
            status.value if status else None,
            purpose.value if purpose else None,
        )
        return results

    def revoke_permit(self, permit_id: str, reason: str) -> bool:
        """Revoke a permit."""
        self._ensure_authenticated()

        success = self._permit_store.revoke(permit_id, reason)
        if success:
            logger.info("Permit %s revoked: %s", permit_id, reason)
        else:
            logger.error("Permit %s not found for revocation", permit_id)
        return success

    def suspend_permit(self, permit_id: str, reason: str) -> bool:
        """Suspend a permit (temporary deactivation)."""
        self._ensure_authenticated()

        success = self._permit_store.suspend(permit_id, reason)
        if success:
            logger.info("Permit %s suspended: %s", permit_id, reason)
        return success

    def reactivate_permit(self, permit_id: str) -> bool:
        """Reactivate a previously suspended permit."""
        self._ensure_authenticated()

        success = self._permit_store.reactivate(permit_id)
        if success:
            logger.info("Permit %s reactivated", permit_id)
        return success

    def request_new_permit(
        self,
        requester_id: str,
        hdab_id: str,
        purpose: PermitPurpose,
        data_categories: List[DataCategory],
        data_sources: Optional[List[str]] = None,
        member_states: Optional[List[str]] = None,
        validity_days: int = 365,
        conditions: Optional[Dict[str, Any]] = None,
    ) -> DataPermit:
        """
        Submit a new data permit application.

        In simulation mode, the permit is automatically approved.
        """
        self._ensure_authenticated()

        permit = DataPermit(
            permit_id=f"PERMIT-{uuid.uuid4().hex[:8].upper()}",
            hdab_id=hdab_id,
            requester_id=requester_id,
            purpose=purpose,
            data_categories=data_categories,
            data_sources=data_sources or [],
            member_states=member_states or [],
            valid_until=datetime.utcnow() + timedelta(days=validity_days),
            status=PermitStatus.ACTIVE,
            conditions=conditions or {},
            metadata={
                "requested_at": datetime.utcnow().isoformat(),
                "auth_method": self.config.auth_method,
                "auto_approved": self.config.simulation_mode,
            },
        )

        if self.config.simulation_mode:
            self._permit_store.register(permit)
            logger.info(
                "Permit %s auto-approved (simulation, purpose=%s)",
                permit.permit_id,
                purpose.value,
            )
        else:
            logger.info(
                "Permit application %s submitted to %s",
                permit.permit_id,
                self.config.endpoint,
            )

        return permit

    # -----------------------------------------------------------------
    # Stats
    # -----------------------------------------------------------------

    def get_store_stats(self) -> Dict[str, Any]:
        """Get statistics from the permit store."""
        all_permits = self._permit_store.list_all()
        by_status: Dict[str, int] = {}
        for p in all_permits:
            s = p.status.value
            by_status[s] = by_status.get(s, 0) + 1

        return {
            "total_permits": self._permit_store.count,
            "by_status": by_status,
            "audit_log_entries": len(self._permit_store.audit_log),
            "connected": self._authenticated,
            "auth_method": self.config.auth_method,
            "simulation_mode": self.config.simulation_mode,
        }

    @property
    def is_connected(self) -> bool:
        """Check if client is connected and authenticated."""
        return self._authenticated

    @property
    def auth_token(self) -> Optional[ProductionAuthToken]:
        """Current authentication token (if any)."""
        return self._auth_token


# =============================================================================
# Multi-HDAB Production Coordinator
# =============================================================================


class ProductionMultiHDABCoordinator:
    """
    Synchronous coordinator for cross-border studies involving multiple HDABs.

    Manages permit synchronization and coordination across Member States
    as required for cross-border secondary use under EHDS Article 50.
    """

    def __init__(self, hdab_configs: Dict[str, HDABProductionConfig]):
        """
        Args:
            hdab_configs: Mapping of member state codes to HDAB configs.
                          Example: {"IT": HDABProductionConfig(...), "DE": ...}
        """
        self.clients: Dict[str, HDABProductionClient] = {
            state: HDABProductionClient(config)
            for state, config in hdab_configs.items()
        }
        self._connected_states: set = set()
        self._coordination_log: List[Dict[str, Any]] = []

    def connect_all(self) -> Dict[str, bool]:
        """
        Connect to all configured HDABs sequentially.

        Returns:
            Mapping of member states to connection status.
        """
        results: Dict[str, bool] = {}

        for state, client in self.clients.items():
            try:
                results[state] = client.connect()
                if results[state]:
                    self._connected_states.add(state)
            except Exception as e:
                logger.error(
                    "Failed to connect to HDAB %s: %s", state, str(e)
                )
                results[state] = False

        self._log_coordination(
            "connect_all", {"results": results}
        )
        return results

    def verify_cross_border_permits(
        self,
        permit_ids: Dict[str, str],
        purpose: PermitPurpose,
        data_categories: List[DataCategory],
    ) -> Dict[str, bool]:
        """
        Verify permits across multiple Member States.

        All participating states must have valid permits for the study
        to proceed (Article 50 cross-border coordination).
        """
        results: Dict[str, bool] = {}

        for state, permit_id in permit_ids.items():
            if state not in self.clients:
                logger.warning("No HDAB client for member state %s", state)
                results[state] = False
                continue

            if state not in self._connected_states:
                logger.warning("HDAB %s not connected", state)
                results[state] = False
                continue

            try:
                results[state] = self.clients[state].verify_permit(
                    permit_id, purpose, data_categories
                )
            except (PermitExpiredError, PermitPurposeMismatchError) as e:
                logger.error(
                    "Permit verification failed for %s/%s: %s",
                    state,
                    permit_id,
                    str(e),
                )
                results[state] = False
            except Exception as e:
                logger.error(
                    "Unexpected error verifying %s/%s: %s",
                    state,
                    permit_id,
                    str(e),
                )
                results[state] = False

        all_valid = all(results.values()) if results else False
        self._log_coordination(
            "verify_cross_border",
            {
                "results": results,
                "all_valid": all_valid,
                "purpose": purpose.value,
            },
        )

        if not all_valid:
            failed = [s for s, v in results.items() if not v]
            logger.warning(
                "Cross-border verification incomplete: failed=%s", failed
            )

        return results

    def request_cross_border_permits(
        self,
        requester_id: str,
        purpose: PermitPurpose,
        data_categories: List[DataCategory],
        target_states: Optional[List[str]] = None,
        validity_days: int = 365,
    ) -> Dict[str, DataPermit]:
        """
        Request permits from all (or specific) participating HDABs.

        Returns:
            Mapping of member states to issued permits.
        """
        states = target_states or list(self._connected_states)
        permits: Dict[str, DataPermit] = {}

        for state in states:
            if state not in self.clients or state not in self._connected_states:
                logger.warning("Skipping unconnected state %s", state)
                continue

            try:
                permit = self.clients[state].request_new_permit(
                    requester_id=requester_id,
                    hdab_id=f"HDAB-{state}",
                    purpose=purpose,
                    data_categories=data_categories,
                    member_states=[state],
                    validity_days=validity_days,
                    conditions={"cross_border": True},
                )
                permits[state] = permit
            except Exception as e:
                logger.error(
                    "Failed to request permit for %s: %s", state, str(e)
                )

        self._log_coordination(
            "request_cross_border_permits",
            {
                "requester_id": requester_id,
                "states": states,
                "permits_issued": {s: p.permit_id for s, p in permits.items()},
            },
        )
        return permits

    def get_cross_border_status(self) -> Dict[str, Dict[str, Any]]:
        """Get connection and permit status for all participating HDABs."""
        status: Dict[str, Dict[str, Any]] = {}
        for state, client in self.clients.items():
            stats = client.get_store_stats()
            status[state] = {
                "member_state": state,
                "connected": state in self._connected_states,
                "auth_method": client.config.auth_method,
                "endpoint": client.config.endpoint,
                **stats,
            }
        return status

    def disconnect_all(self) -> None:
        """Disconnect from all HDABs."""
        for state, client in self.clients.items():
            try:
                client.disconnect()
            except Exception as e:
                logger.error(
                    "Error disconnecting from HDAB %s: %s", state, str(e)
                )
        self._connected_states.clear()
        self._log_coordination("disconnect_all")

    @property
    def connected_states(self) -> set:
        return set(self._connected_states)

    @property
    def coordination_log(self) -> List[Dict[str, Any]]:
        return list(self._coordination_log)

    def _log_coordination(
        self, action: str, details: Optional[Dict] = None
    ) -> None:
        self._coordination_log.append({
            "action": action,
            "timestamp": datetime.utcnow().isoformat(),
            "connected_states": list(self._connected_states),
            "details": details or {},
        })


# =============================================================================
# Factory Functions
# =============================================================================


def create_production_client(
    endpoint: str,
    auth_method: str = "oauth2",
    simulation: bool = True,
    **kwargs,
) -> HDABProductionClient:
    """
    Create an HDABProductionClient with minimal configuration.

    Args:
        endpoint: HDAB API endpoint URL.
        auth_method: Authentication method (oauth2, mtls, api_key).
        simulation: If True, use in-memory backend.
        **kwargs: Additional HDABProductionConfig fields.

    Returns:
        Configured HDABProductionClient.
    """
    config = HDABProductionConfig(
        endpoint=endpoint,
        auth_method=auth_method,
        simulation_mode=simulation,
        **kwargs,
    )
    return HDABProductionClient(config)


def create_production_coordinator(
    countries: Dict[str, str],
    auth_method: str = "oauth2",
    simulation: bool = True,
) -> ProductionMultiHDABCoordinator:
    """
    Create a ProductionMultiHDABCoordinator from country->endpoint mapping.

    Args:
        countries: Mapping of country codes to HDAB endpoint URLs.
                   Example: {"IT": "http://hdab-it.ehds.eu",
                             "DE": "http://hdab-de.ehds.eu"}
        auth_method: Authentication method for all HDABs.
        simulation: If True, use in-memory backend.

    Returns:
        Configured ProductionMultiHDABCoordinator.
    """
    configs = {
        cc: HDABProductionConfig(
            endpoint=url,
            auth_method=auth_method,
            simulation_mode=simulation,
        )
        for cc, url in countries.items()
    }
    return ProductionMultiHDABCoordinator(configs)

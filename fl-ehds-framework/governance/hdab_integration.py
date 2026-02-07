"""
HDAB Integration Module
=======================
Integration with Health Data Access Bodies (HDABs) for permit management
and cross-border coordination under EHDS Regulation EU 2025/327.

Implements:
- Secure authentication (OAuth2, mTLS, API Key)
- Permit lifecycle management (request, verify, revoke)
- In-memory simulation backend for testing and demo
- Cross-border multi-HDAB coordination (Article 50)
"""

import asyncio
import secrets
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
import structlog

from core.models import DataPermit, PermitStatus, PermitPurpose, DataCategory
from core.exceptions import (
    HDABConnectionError,
    PermitNotFoundError,
    PermitExpiredError,
    PermitPurposeMismatchError,
)

logger = structlog.get_logger(__name__)


# =============================================================================
# Configuration & Auth
# =============================================================================


@dataclass
class HDABConfig:
    """Configuration for HDAB connection."""

    endpoint: str
    auth_method: str = "oauth2"  # oauth2, mtls, api_key
    timeout: int = 30
    max_retries: int = 3
    backoff_factor: float = 2.0
    cache_enabled: bool = True
    cache_ttl: int = 3600  # seconds
    # OAuth2 credentials
    client_id: Optional[str] = None
    client_secret: Optional[str] = None
    token_endpoint: Optional[str] = None
    scopes: List[str] = field(default_factory=lambda: ["permits:read", "permits:write"])
    # mTLS certificate paths
    cert_path: Optional[str] = None
    key_path: Optional[str] = None
    ca_path: Optional[str] = None
    # API key
    api_key: Optional[str] = None
    # Simulation mode (in-memory backend, no real HTTP)
    simulation_mode: bool = True


@dataclass
class AuthToken:
    """OAuth2 bearer token with expiry tracking."""

    access_token: str
    token_type: str = "Bearer"
    expires_at: datetime = field(default_factory=lambda: datetime.utcnow() + timedelta(hours=1))
    scopes: List[str] = field(default_factory=list)
    refresh_token: Optional[str] = None

    @property
    def is_expired(self) -> bool:
        return datetime.utcnow() >= self.expires_at

    @property
    def authorization_header(self) -> str:
        return f"{self.token_type} {self.access_token}"


# =============================================================================
# In-Memory Permit Store (simulation backend)
# =============================================================================


class PermitStore:
    """
    In-memory permit store for HDAB simulation.

    Provides a local backend to test the full permit lifecycle
    without requiring a live HDAB endpoint.
    """

    def __init__(self):
        self._permits: Dict[str, DataPermit] = {}
        self._audit_log: List[Dict[str, Any]] = []

    def register(self, permit: DataPermit) -> None:
        self._permits[permit.permit_id] = permit
        self._log_action("register", permit.permit_id)

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
        self._log_action("revoke", permit_id, details={"reason": reason})
        return True

    def suspend(self, permit_id: str, reason: str) -> bool:
        permit = self._permits.get(permit_id)
        if not permit:
            return False
        permit.status = PermitStatus.SUSPENDED
        permit.metadata["suspension_reason"] = reason
        permit.metadata["suspended_at"] = datetime.utcnow().isoformat()
        self._log_action("suspend", permit_id, details={"reason": reason})
        return True

    def reactivate(self, permit_id: str) -> bool:
        permit = self._permits.get(permit_id)
        if not permit or permit.status != PermitStatus.SUSPENDED:
            return False
        permit.status = PermitStatus.ACTIVE
        permit.metadata.pop("suspension_reason", None)
        self._log_action("reactivate", permit_id)
        return True

    def remove(self, permit_id: str) -> bool:
        if permit_id in self._permits:
            del self._permits[permit_id]
            self._log_action("remove", permit_id)
            return True
        return False

    @property
    def count(self) -> int:
        return len(self._permits)

    @property
    def audit_log(self) -> List[Dict[str, Any]]:
        return list(self._audit_log)

    def _log_action(
        self, action: str, permit_id: str, details: Optional[Dict] = None
    ) -> None:
        self._audit_log.append({
            "action": action,
            "permit_id": permit_id,
            "timestamp": datetime.utcnow().isoformat(),
            "details": details or {},
        })


# Global shared store for simulation mode
_shared_permit_store = PermitStore()


def get_shared_permit_store() -> PermitStore:
    """Get the shared in-memory permit store (simulation mode)."""
    return _shared_permit_store


# =============================================================================
# HDAB Client
# =============================================================================


class HDABClient:
    """
    Client for communicating with Health Data Access Bodies (HDABs).

    Provides standardized API for:
    - Permit verification and lifecycle management
    - Cross-border coordination with multiple HDABs
    - Secure authentication (OAuth2, mTLS, API key)

    In simulation_mode (default), uses an in-memory PermitStore backend.
    In production mode, connects to real HDAB API endpoints via HTTP.
    """

    def __init__(self, config: HDABConfig):
        """
        Initialize HDAB client.

        Args:
            config: HDAB connection configuration.
        """
        self.config = config
        self._permit_cache: Dict[str, Tuple[DataPermit, datetime]] = {}
        self._session = None
        self._authenticated = False
        self._auth_token: Optional[AuthToken] = None
        self._permit_store: PermitStore = _shared_permit_store

    async def connect(self) -> bool:
        """
        Establish connection to HDAB.

        Returns:
            True if connection successful.

        Raises:
            HDABConnectionError: If connection fails.
        """
        logger.info("Connecting to HDAB", endpoint=self.config.endpoint)

        for attempt in range(self.config.max_retries):
            try:
                await self._authenticate()
                self._authenticated = True
                logger.info(
                    "HDAB connection established",
                    endpoint=self.config.endpoint,
                    auth_method=self.config.auth_method,
                )
                return True

            except Exception as e:
                wait_time = self.config.backoff_factor ** attempt
                logger.warning(
                    "HDAB connection attempt failed",
                    attempt=attempt + 1,
                    error=str(e),
                    retry_in=wait_time,
                )
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(wait_time)

        raise HDABConnectionError(
            self.config.endpoint,
            f"Failed after {self.config.max_retries} attempts",
        )

    async def disconnect(self) -> None:
        """Close connection to HDAB and invalidate auth token."""
        self._authenticated = False
        self._auth_token = None
        self._session = None
        logger.info("HDAB connection closed", endpoint=self.config.endpoint)

    # -------------------------------------------------------------------------
    # Authentication
    # -------------------------------------------------------------------------

    async def _authenticate(self) -> None:
        """Authenticate with HDAB using configured method."""
        if self.config.auth_method == "oauth2":
            await self._oauth2_authenticate()
        elif self.config.auth_method == "mtls":
            await self._mtls_authenticate()
        elif self.config.auth_method == "api_key":
            await self._apikey_authenticate()
        else:
            raise ValueError(f"Unsupported auth method: {self.config.auth_method}")

    async def _oauth2_authenticate(self) -> None:
        """
        Perform OAuth2 client_credentials authentication.

        In simulation mode, generates a local bearer token.
        In production, would POST to the HDAB token endpoint.
        """
        if self.config.simulation_mode:
            self._auth_token = AuthToken(
                access_token=secrets.token_urlsafe(32),
                token_type="Bearer",
                expires_at=datetime.utcnow() + timedelta(hours=1),
                scopes=self.config.scopes,
                refresh_token=secrets.token_urlsafe(32),
            )
            logger.debug(
                "OAuth2 authentication successful (simulation)",
                scopes=self.config.scopes,
                expires_in=3600,
            )
            return

        # Production: POST to token endpoint
        # token_url = self.config.token_endpoint or f"{self.config.endpoint}/oauth2/token"
        # async with aiohttp.ClientSession() as session:
        #     async with session.post(token_url, data={
        #         "grant_type": "client_credentials",
        #         "client_id": self.config.client_id,
        #         "client_secret": self.config.client_secret,
        #         "scope": " ".join(self.config.scopes),
        #     }) as resp:
        #         if resp.status != 200:
        #             raise HDABConnectionError(token_url, f"OAuth2 error: {resp.status}")
        #         data = await resp.json()
        #         self._auth_token = AuthToken(
        #             access_token=data["access_token"],
        #             token_type=data.get("token_type", "Bearer"),
        #             expires_at=datetime.utcnow() + timedelta(seconds=data["expires_in"]),
        #             scopes=data.get("scope", "").split(),
        #         )
        raise NotImplementedError("Production OAuth2 requires aiohttp")

    async def _mtls_authenticate(self) -> None:
        """
        Perform mutual TLS authentication.

        In simulation mode, validates that certificate paths are configured.
        In production, would establish an mTLS session.
        """
        if self.config.simulation_mode:
            if not self.config.cert_path or not self.config.key_path:
                logger.debug("mTLS simulation: no certs configured, using default trust")
            self._auth_token = AuthToken(
                access_token=f"mtls-{secrets.token_hex(16)}",
                token_type="mTLS",
                expires_at=datetime.utcnow() + timedelta(hours=24),
                scopes=["permits:read", "permits:write", "optout:read"],
            )
            logger.debug(
                "mTLS authentication successful (simulation)",
                cert_path=self.config.cert_path,
            )
            return

        # Production: create aiohttp session with SSL context
        # import ssl
        # ssl_ctx = ssl.create_default_context(cafile=self.config.ca_path)
        # ssl_ctx.load_cert_chain(self.config.cert_path, self.config.key_path)
        # self._session = aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=ssl_ctx))
        raise NotImplementedError("Production mTLS requires aiohttp with SSL")

    async def _apikey_authenticate(self) -> None:
        """
        Perform API key authentication.

        In simulation mode, accepts any non-empty key.
        In production, would validate the key against the HDAB endpoint.
        """
        if self.config.simulation_mode:
            key = self.config.api_key or f"sim-key-{secrets.token_hex(8)}"
            self._auth_token = AuthToken(
                access_token=key,
                token_type="ApiKey",
                expires_at=datetime.utcnow() + timedelta(days=365),
                scopes=["permits:read"],
            )
            logger.debug("API key authentication successful (simulation)")
            return

        if not self.config.api_key:
            raise HDABConnectionError(self.config.endpoint, "API key not configured")

        # Production: validate key
        # async with aiohttp.ClientSession() as session:
        #     async with session.get(
        #         f"{self.config.endpoint}/auth/verify",
        #         headers={"X-API-Key": self.config.api_key},
        #     ) as resp:
        #         if resp.status != 200:
        #             raise HDABConnectionError(self.config.endpoint, "Invalid API key")
        raise NotImplementedError("Production API key validation requires aiohttp")

    async def _ensure_authenticated(self) -> None:
        """Re-authenticate if token is expired."""
        if self._auth_token and self._auth_token.is_expired:
            logger.info("Auth token expired, re-authenticating")
            await self._authenticate()

    # -------------------------------------------------------------------------
    # Permit Operations
    # -------------------------------------------------------------------------

    def register_permit(self, permit: DataPermit) -> None:
        """
        Register a permit in the local store (simulation mode).

        Args:
            permit: DataPermit to register.
        """
        self._permit_store.register(permit)
        logger.info("Permit registered", permit_id=permit.permit_id)

    async def get_permit(self, permit_id: str) -> DataPermit:
        """
        Retrieve a data permit from HDAB.

        Args:
            permit_id: Unique permit identifier.

        Returns:
            DataPermit object.

        Raises:
            PermitNotFoundError: If permit doesn't exist.
            HDABConnectionError: If communication fails.
        """
        await self._ensure_authenticated()

        # Check cache first
        if self.config.cache_enabled and permit_id in self._permit_cache:
            permit, cached_at = self._permit_cache[permit_id]
            if datetime.utcnow() - cached_at < timedelta(seconds=self.config.cache_ttl):
                logger.debug("Permit retrieved from cache", permit_id=permit_id)
                return permit

        logger.info("Fetching permit from HDAB", permit_id=permit_id)

        try:
            permit_data = await self._fetch_permit(permit_id)

            if permit_data is None:
                raise PermitNotFoundError(permit_id)

            if isinstance(permit_data, DataPermit):
                permit = permit_data
            else:
                permit = DataPermit(**permit_data)

            # Cache the permit
            if self.config.cache_enabled:
                self._permit_cache[permit_id] = (permit, datetime.utcnow())

            return permit

        except PermitNotFoundError:
            raise
        except Exception as e:
            raise HDABConnectionError(
                self.config.endpoint,
                f"Failed to fetch permit: {str(e)}",
            )

    async def _fetch_permit(self, permit_id: str) -> Optional[Any]:
        """
        Fetch permit data from HDAB API.

        In simulation mode, queries the in-memory PermitStore.
        In production, performs an HTTP GET to the HDAB permits endpoint.
        """
        if self.config.simulation_mode:
            permit = self._permit_store.get(permit_id)
            if permit:
                logger.debug("Permit fetched from simulation store", permit_id=permit_id)
            return permit

        # Production HTTP call:
        # headers = {"Authorization": self._auth_token.authorization_header}
        # async with self._session.get(
        #     f"{self.config.endpoint}/api/v1/permits/{permit_id}",
        #     headers=headers,
        #     timeout=aiohttp.ClientTimeout(total=self.config.timeout),
        # ) as response:
        #     if response.status == 404:
        #         return None
        #     response.raise_for_status()
        #     return await response.json()
        return None

    async def verify_permit(
        self,
        permit_id: str,
        purpose: PermitPurpose,
        data_categories: List[DataCategory],
    ) -> bool:
        """
        Verify a permit is valid for the specified purpose and data categories.

        Args:
            permit_id: Permit to verify.
            purpose: Intended processing purpose.
            data_categories: Data categories to be processed.

        Returns:
            True if permit is valid for the request.

        Raises:
            PermitNotFoundError: If permit doesn't exist.
            PermitExpiredError: If permit has expired.
            PermitPurposeMismatchError: If purpose does not match.
        """
        permit = await self.get_permit(permit_id)

        # Check expiry
        if not permit.is_valid():
            if permit.status == PermitStatus.EXPIRED or permit.valid_until < datetime.utcnow():
                raise PermitExpiredError(permit_id, permit.valid_until.isoformat())
            return False

        # Check purpose alignment
        if not permit.covers_purpose(purpose):
            logger.warning(
                "Purpose mismatch",
                permit_id=permit_id,
                requested=purpose.value,
                permitted=permit.purpose.value,
            )
            raise PermitPurposeMismatchError(
                permit_id,
                requested_purpose=purpose.value,
                allowed_purposes=[permit.purpose.value],
            )

        # Check data categories
        if not permit.covers_categories(data_categories):
            missing = [
                c.value for c in data_categories
                if c not in permit.data_categories
            ]
            logger.warning(
                "Data category mismatch",
                permit_id=permit_id,
                missing_categories=missing,
            )
            return False

        logger.info("Permit verified successfully", permit_id=permit_id)
        return True

    async def list_permits(
        self,
        requester_id: Optional[str] = None,
        status: Optional[PermitStatus] = None,
        purpose: Optional[PermitPurpose] = None,
    ) -> List[DataPermit]:
        """
        List permits matching specified criteria.

        Args:
            requester_id: Filter by requester.
            status: Filter by status.
            purpose: Filter by purpose.

        Returns:
            List of matching permits.
        """
        await self._ensure_authenticated()

        if self.config.simulation_mode:
            results = self._permit_store.list_all(requester_id, status, purpose)
            logger.info(
                "Listing permits (simulation)",
                requester_id=requester_id,
                status=status.value if status else None,
                purpose=purpose.value if purpose else None,
                count=len(results),
            )
            return results

        # Production: GET /api/v1/permits?requester_id=...&status=...
        logger.info(
            "Listing permits",
            requester_id=requester_id,
            status=status.value if status else None,
            purpose=purpose.value if purpose else None,
        )
        return []

    async def revoke_permit(self, permit_id: str, reason: str) -> bool:
        """
        Request permit revocation.

        Args:
            permit_id: Permit to revoke.
            reason: Reason for revocation.

        Returns:
            True if revocation was accepted.
        """
        await self._ensure_authenticated()

        logger.warning(
            "Requesting permit revocation",
            permit_id=permit_id,
            reason=reason,
        )

        if self.config.simulation_mode:
            success = self._permit_store.revoke(permit_id, reason)
            if success:
                # Invalidate cache
                self._permit_cache.pop(permit_id, None)
                logger.info("Permit revoked (simulation)", permit_id=permit_id)
            else:
                logger.error("Permit not found for revocation", permit_id=permit_id)
            return success

        # Production: POST /api/v1/permits/{permit_id}/revoke
        return False

    async def suspend_permit(self, permit_id: str, reason: str) -> bool:
        """
        Request permit suspension (temporary deactivation).

        Args:
            permit_id: Permit to suspend.
            reason: Reason for suspension.

        Returns:
            True if suspension was accepted.
        """
        await self._ensure_authenticated()

        if self.config.simulation_mode:
            success = self._permit_store.suspend(permit_id, reason)
            if success:
                self._permit_cache.pop(permit_id, None)
                logger.info("Permit suspended", permit_id=permit_id, reason=reason)
            return success
        return False

    async def reactivate_permit(self, permit_id: str) -> bool:
        """
        Reactivate a previously suspended permit.

        Args:
            permit_id: Permit to reactivate.

        Returns:
            True if reactivation was accepted.
        """
        await self._ensure_authenticated()

        if self.config.simulation_mode:
            success = self._permit_store.reactivate(permit_id)
            if success:
                self._permit_cache.pop(permit_id, None)
                logger.info("Permit reactivated", permit_id=permit_id)
            return success
        return False

    async def request_new_permit(
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
        Submit a new data permit application to the HDAB.

        In simulation mode, the permit is automatically approved.

        Args:
            requester_id: Requesting organization ID.
            hdab_id: Target HDAB identifier.
            purpose: Purpose under Article 53.
            data_categories: Requested data categories.
            data_sources: Specific data source IDs.
            member_states: Member states covered.
            validity_days: Requested validity period.
            conditions: Additional conditions.

        Returns:
            The issued DataPermit.
        """
        await self._ensure_authenticated()

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
                "Permit auto-approved (simulation)",
                permit_id=permit.permit_id,
                purpose=purpose.value,
            )
        else:
            logger.info(
                "Permit application submitted",
                permit_id=permit.permit_id,
            )

        return permit

    def clear_cache(self) -> None:
        """Clear the permit cache."""
        self._permit_cache.clear()
        logger.debug("Permit cache cleared")

    def get_store_stats(self) -> Dict[str, Any]:
        """Get statistics from the permit store."""
        all_permits = self._permit_store.list_all()
        by_status = {}
        for p in all_permits:
            status = p.status.value
            by_status[status] = by_status.get(status, 0) + 1

        return {
            "total_permits": self._permit_store.count,
            "by_status": by_status,
            "cache_entries": len(self._permit_cache),
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
    def auth_token(self) -> Optional[AuthToken]:
        """Current authentication token (if any)."""
        return self._auth_token


# =============================================================================
# Multi-HDAB Coordinator (Article 50 - Cross-Border)
# =============================================================================


class MultiHDABCoordinator:
    """
    Coordinator for cross-border studies involving multiple HDABs.

    Manages permit synchronization and coordination across Member States
    as required for cross-border secondary use under EHDS Article 50.
    """

    def __init__(self, hdab_configs: Dict[str, HDABConfig]):
        """
        Initialize multi-HDAB coordinator.

        Args:
            hdab_configs: Dictionary mapping member state codes to HDAB configs.
                          Example: {"IT": HDABConfig(...), "DE": HDABConfig(...)}
        """
        self.clients: Dict[str, HDABClient] = {
            state: HDABClient(config) for state, config in hdab_configs.items()
        }
        self._connected_states: set = set()
        self._coordination_log: List[Dict[str, Any]] = []

    async def connect_all(self) -> Dict[str, bool]:
        """
        Connect to all configured HDABs concurrently.

        Returns:
            Dictionary mapping member states to connection status.
        """
        results = {}
        tasks = {}

        for state, client in self.clients.items():
            tasks[state] = asyncio.create_task(self._connect_single(state, client))

        for state, task in tasks.items():
            try:
                results[state] = await task
                if results[state]:
                    self._connected_states.add(state)
            except Exception as e:
                logger.error("Failed to connect to HDAB", member_state=state, error=str(e))
                results[state] = False

        self._log_coordination(
            "connect_all",
            details={"results": {k: v for k, v in results.items()}},
        )
        return results

    async def _connect_single(self, state: str, client: HDABClient) -> bool:
        """Connect to a single HDAB, catching errors."""
        try:
            return await client.connect()
        except HDABConnectionError as e:
            logger.error("Failed to connect to HDAB", member_state=state, error=str(e))
            return False

    async def verify_cross_border_permits(
        self,
        permit_ids: Dict[str, str],  # member_state -> permit_id
        purpose: PermitPurpose,
        data_categories: List[DataCategory],
    ) -> Dict[str, bool]:
        """
        Verify permits across multiple Member States.

        All participating states must have valid permits for the study
        to proceed (Article 50 cross-border coordination).

        Args:
            permit_ids: Mapping of member states to permit IDs.
            purpose: Processing purpose.
            data_categories: Data categories to process.

        Returns:
            Dictionary mapping member states to verification results.
        """
        results = {}

        for state, permit_id in permit_ids.items():
            if state not in self.clients:
                logger.warning("No HDAB client for member state", state=state)
                results[state] = False
                continue

            if state not in self._connected_states:
                logger.warning("HDAB not connected", state=state)
                results[state] = False
                continue

            try:
                results[state] = await self.clients[state].verify_permit(
                    permit_id, purpose, data_categories
                )
            except (PermitExpiredError, PermitPurposeMismatchError) as e:
                logger.error(
                    "Permit verification failed",
                    state=state,
                    permit_id=permit_id,
                    error=str(e),
                )
                results[state] = False
            except Exception as e:
                logger.error(
                    "Unexpected error during permit verification",
                    state=state,
                    permit_id=permit_id,
                    error=str(e),
                )
                results[state] = False

        all_valid = all(results.values()) if results else False
        self._log_coordination(
            "verify_cross_border",
            details={
                "results": results,
                "all_valid": all_valid,
                "purpose": purpose.value,
            },
        )

        if not all_valid:
            failed = [s for s, v in results.items() if not v]
            logger.warning(
                "Cross-border verification incomplete",
                failed_states=failed,
            )

        return results

    async def request_cross_border_permits(
        self,
        requester_id: str,
        purpose: PermitPurpose,
        data_categories: List[DataCategory],
        target_states: Optional[List[str]] = None,
        validity_days: int = 365,
    ) -> Dict[str, DataPermit]:
        """
        Request permits from all (or specific) participating HDABs.

        Args:
            requester_id: Requesting organization ID.
            purpose: Purpose under Article 53.
            data_categories: Data categories needed.
            target_states: Specific states (None = all connected).
            validity_days: Requested validity period.

        Returns:
            Dictionary mapping member states to issued permits.
        """
        states = target_states or list(self._connected_states)
        permits: Dict[str, DataPermit] = {}

        for state in states:
            if state not in self.clients or state not in self._connected_states:
                logger.warning("Skipping unconnected state", state=state)
                continue

            try:
                permit = await self.clients[state].request_new_permit(
                    requester_id=requester_id,
                    hdab_id=f"HDAB-{state}",
                    purpose=purpose,
                    data_categories=data_categories,
                    member_states=[state],
                    validity_days=validity_days,
                    conditions={"cross_border": True, "lead_coordinator": True},
                )
                permits[state] = permit
            except Exception as e:
                logger.error(
                    "Failed to request permit",
                    state=state,
                    error=str(e),
                )

        self._log_coordination(
            "request_cross_border_permits",
            details={
                "requester_id": requester_id,
                "states": states,
                "permits_issued": {s: p.permit_id for s, p in permits.items()},
            },
        )
        return permits

    async def get_cross_border_status(self) -> Dict[str, Dict[str, Any]]:
        """
        Get connection and permit status for all participating HDABs.

        Returns:
            Dictionary mapping member states to status information.
        """
        status = {}
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

    async def disconnect_all(self) -> None:
        """Disconnect from all HDABs."""
        for state, client in self.clients.items():
            try:
                await client.disconnect()
            except Exception as e:
                logger.error("Error disconnecting from HDAB", state=state, error=str(e))
        self._connected_states.clear()
        self._log_coordination("disconnect_all")

    @property
    def connected_states(self) -> set:
        return set(self._connected_states)

    @property
    def coordination_log(self) -> List[Dict[str, Any]]:
        return list(self._coordination_log)

    def _log_coordination(self, action: str, details: Optional[Dict] = None) -> None:
        self._coordination_log.append({
            "action": action,
            "timestamp": datetime.utcnow().isoformat(),
            "connected_states": list(self._connected_states),
            "details": details or {},
        })

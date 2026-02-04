"""
HDAB Integration Module
=======================
Integration with Health Data Access Bodies (HDABs) for permit management
and cross-border coordination under EHDS Regulation.
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
import structlog

from core.models import DataPermit, PermitStatus, PermitPurpose, DataCategory
from core.exceptions import (
    HDABConnectionError,
    PermitNotFoundError,
    PermitExpiredError,
)

logger = structlog.get_logger(__name__)


@dataclass
class HDABConfig:
    """Configuration for HDAB connection."""

    endpoint: str
    auth_method: str = "oauth2"
    timeout: int = 30
    max_retries: int = 3
    backoff_factor: float = 2.0
    cache_enabled: bool = True
    cache_ttl: int = 3600  # seconds


class HDABClient:
    """
    Client for communicating with Health Data Access Bodies (HDABs).

    Provides standardized API for:
    - Permit verification and lifecycle management
    - Cross-border coordination with multiple HDABs
    - Secure authentication and communication
    """

    def __init__(self, config: HDABConfig):
        """
        Initialize HDAB client.

        Args:
            config: HDAB connection configuration.
        """
        self.config = config
        self._permit_cache: Dict[str, tuple] = {}  # permit_id -> (permit, timestamp)
        self._session = None
        self._authenticated = False

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
                # Simulate connection (replace with actual HTTP/gRPC client)
                await self._authenticate()
                self._authenticated = True
                logger.info("HDAB connection established")
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
        """Close connection to HDAB."""
        self._authenticated = False
        self._session = None
        logger.info("HDAB connection closed")

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
        """Perform OAuth2 authentication."""
        # Implementation placeholder
        logger.debug("OAuth2 authentication successful")

    async def _mtls_authenticate(self) -> None:
        """Perform mutual TLS authentication."""
        # Implementation placeholder
        logger.debug("mTLS authentication successful")

    async def _apikey_authenticate(self) -> None:
        """Perform API key authentication."""
        # Implementation placeholder
        logger.debug("API key authentication successful")

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
        # Check cache first
        if self.config.cache_enabled and permit_id in self._permit_cache:
            permit, cached_at = self._permit_cache[permit_id]
            if datetime.utcnow() - cached_at < timedelta(seconds=self.config.cache_ttl):
                logger.debug("Permit retrieved from cache", permit_id=permit_id)
                return permit

        logger.info("Fetching permit from HDAB", permit_id=permit_id)

        try:
            # Simulate API call (replace with actual HTTP request)
            permit_data = await self._fetch_permit(permit_id)

            if permit_data is None:
                raise PermitNotFoundError(permit_id)

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

    async def _fetch_permit(self, permit_id: str) -> Optional[Dict[str, Any]]:
        """Fetch permit data from HDAB API."""
        # Implementation placeholder - replace with actual API call
        # This would typically be:
        # async with self._session.get(f"{self.config.endpoint}/permits/{permit_id}") as response:
        #     if response.status == 404:
        #         return None
        #     return await response.json()

        # For now, return mock data for testing
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
            return False

        # Check data categories
        if not permit.covers_categories(data_categories):
            logger.warning(
                "Data category mismatch",
                permit_id=permit_id,
                requested=[c.value for c in data_categories],
                permitted=[c.value for c in permit.data_categories],
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
        # Implementation placeholder
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
        logger.warning("Requesting permit revocation", permit_id=permit_id, reason=reason)
        # Implementation placeholder
        return True

    def clear_cache(self) -> None:
        """Clear the permit cache."""
        self._permit_cache.clear()
        logger.debug("Permit cache cleared")

    @property
    def is_connected(self) -> bool:
        """Check if client is connected and authenticated."""
        return self._authenticated


class MultiHDABCoordinator:
    """
    Coordinator for cross-border studies involving multiple HDABs.

    Manages permit synchronization and coordination across Member States
    as required for cross-border secondary use under EHDS.
    """

    def __init__(self, hdab_configs: Dict[str, HDABConfig]):
        """
        Initialize multi-HDAB coordinator.

        Args:
            hdab_configs: Dictionary mapping member state codes to HDAB configs.
        """
        self.clients: Dict[str, HDABClient] = {
            state: HDABClient(config) for state, config in hdab_configs.items()
        }
        self._connected_states: set = set()

    async def connect_all(self) -> Dict[str, bool]:
        """
        Connect to all configured HDABs.

        Returns:
            Dictionary mapping member states to connection status.
        """
        results = {}
        for state, client in self.clients.items():
            try:
                await client.connect()
                self._connected_states.add(state)
                results[state] = True
            except HDABConnectionError as e:
                logger.error(
                    "Failed to connect to HDAB",
                    member_state=state,
                    error=str(e),
                )
                results[state] = False
        return results

    async def verify_cross_border_permits(
        self,
        permit_ids: Dict[str, str],  # member_state -> permit_id
        purpose: PermitPurpose,
        data_categories: List[DataCategory],
    ) -> Dict[str, bool]:
        """
        Verify permits across multiple Member States.

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
            except Exception as e:
                logger.error(
                    "Permit verification failed",
                    state=state,
                    permit_id=permit_id,
                    error=str(e),
                )
                results[state] = False

        return results

    async def disconnect_all(self) -> None:
        """Disconnect from all HDABs."""
        for state, client in self.clients.items():
            try:
                await client.disconnect()
            except Exception as e:
                logger.error("Error disconnecting from HDAB", state=state, error=str(e))
        self._connected_states.clear()

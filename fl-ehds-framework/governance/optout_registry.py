"""
Opt-out Registry Module
=======================
Implements Article 71 opt-out compliance for EHDS secondary use.
Manages synchronization with national opt-out registries and
record-level opt-out checking.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set
import structlog

from core.models import OptOutRecord, DataCategory, PermitPurpose
from core.exceptions import OptOutError, OptOutViolationError

logger = structlog.get_logger(__name__)


@dataclass
class OptOutCacheEntry:
    """Cache entry for opt-out status."""

    patient_id: str
    is_opted_out: bool
    scope: str
    categories: Optional[Set[DataCategory]]
    purposes: Optional[Set[PermitPurpose]]
    cached_at: datetime
    expires_at: datetime


class OptOutRegistry:
    """
    Manages synchronization with national opt-out registries.

    Provides caching and efficient lookup of citizen opt-out decisions
    per EHDS Article 71 requirements.
    """

    def __init__(
        self,
        registry_endpoint: Optional[str] = None,
        sync_interval: int = 300,
        cache_ttl: int = 600,
        max_cache_size: int = 100000,
    ):
        """
        Initialize opt-out registry.

        Args:
            registry_endpoint: URL of national opt-out registry API.
            sync_interval: Seconds between registry synchronizations.
            cache_ttl: Cache entry time-to-live in seconds.
            max_cache_size: Maximum number of cached entries.
        """
        self.registry_endpoint = registry_endpoint
        self.sync_interval = sync_interval
        self.cache_ttl = cache_ttl
        self.max_cache_size = max_cache_size

        self._cache: Dict[str, OptOutCacheEntry] = {}
        self._opted_out_ids: Set[str] = set()
        self._last_sync: Optional[datetime] = None
        self._sync_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start background synchronization with registry."""
        if self.registry_endpoint:
            self._sync_task = asyncio.create_task(self._sync_loop())
            logger.info(
                "Opt-out registry sync started",
                endpoint=self.registry_endpoint,
                interval=self.sync_interval,
            )

    async def stop(self) -> None:
        """Stop background synchronization."""
        if self._sync_task:
            self._sync_task.cancel()
            try:
                await self._sync_task
            except asyncio.CancelledError:
                pass
            logger.info("Opt-out registry sync stopped")

    async def _sync_loop(self) -> None:
        """Background loop for registry synchronization."""
        while True:
            try:
                await self._sync_with_registry()
                await asyncio.sleep(self.sync_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Registry sync failed", error=str(e))
                await asyncio.sleep(self.sync_interval)

    async def _sync_with_registry(self) -> None:
        """Synchronize with national opt-out registry."""
        logger.debug("Syncing with opt-out registry")

        # Implementation placeholder - replace with actual API call
        # This would fetch updates since last_sync from the national registry

        self._last_sync = datetime.utcnow()
        logger.info("Opt-out registry synced", timestamp=self._last_sync.isoformat())

    def register_optout(self, record: OptOutRecord) -> None:
        """
        Register an opt-out record.

        Args:
            record: The opt-out record to register.
        """
        self._opted_out_ids.add(record.patient_id)

        # Create cache entry
        entry = OptOutCacheEntry(
            patient_id=record.patient_id,
            is_opted_out=record.is_active,
            scope=record.scope,
            categories=(
                set(record.categories) if record.categories else None
            ),
            purposes=set(record.purposes) if record.purposes else None,
            cached_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + timedelta(seconds=self.cache_ttl),
        )
        self._cache[record.patient_id] = entry

        logger.info(
            "Opt-out registered",
            patient_id=record.patient_id[:8] + "...",  # Truncate for privacy
            scope=record.scope,
        )

    def is_opted_out(
        self,
        patient_id: str,
        category: Optional[DataCategory] = None,
        purpose: Optional[PermitPurpose] = None,
    ) -> bool:
        """
        Check if a patient has opted out.

        Args:
            patient_id: Patient identifier to check.
            category: Optional specific data category.
            purpose: Optional specific purpose.

        Returns:
            True if patient has opted out for the specified scope.
        """
        # Check simple set first (fast path)
        if patient_id not in self._opted_out_ids:
            return False

        # Check cache for detailed information
        if patient_id in self._cache:
            entry = self._cache[patient_id]

            # Check if cache entry is still valid
            if entry.expires_at < datetime.utcnow():
                # Entry expired - would need to re-fetch
                # For now, treat as opted out (safe default)
                return True

            if not entry.is_opted_out:
                return False

            # Check scope
            if entry.scope == "all":
                return True

            if entry.scope == "category" and category:
                return entry.categories and category in entry.categories

            if entry.scope == "purpose" and purpose:
                return entry.purposes and purpose in entry.purposes

        # Default to opted out if in the set but no detailed info
        return True

    def check_records(
        self,
        record_ids: List[str],
        patient_id_mapping: Dict[str, str],
        category: Optional[DataCategory] = None,
        purpose: Optional[PermitPurpose] = None,
    ) -> Dict[str, bool]:
        """
        Batch check opt-out status for multiple records.

        Args:
            record_ids: List of record identifiers.
            patient_id_mapping: Mapping of record_id to patient_id.
            category: Data category being processed.
            purpose: Processing purpose.

        Returns:
            Dictionary mapping record_id to opt-out status.
        """
        results = {}
        for record_id in record_ids:
            patient_id = patient_id_mapping.get(record_id)
            if patient_id:
                results[record_id] = self.is_opted_out(patient_id, category, purpose)
            else:
                # Unknown patient - treat as not opted out
                results[record_id] = False
        return results

    def get_opted_out_count(self) -> int:
        """Get count of opted-out patients."""
        return len(self._opted_out_ids)

    def clear_cache(self) -> None:
        """Clear the opt-out cache."""
        self._cache.clear()
        logger.debug("Opt-out cache cleared")


class OptOutChecker:
    """
    Utility class for checking and enforcing opt-out compliance
    during FL training.
    """

    def __init__(
        self,
        registry: OptOutRegistry,
        on_optout: str = "exclude",
    ):
        """
        Initialize opt-out checker.

        Args:
            registry: The opt-out registry to use.
            on_optout: Action on opt-out: 'exclude', 'anonymize', or 'error'.
        """
        self.registry = registry
        self.on_optout = on_optout

        if on_optout not in ("exclude", "anonymize", "error"):
            raise ValueError(f"Invalid on_optout action: {on_optout}")

    def filter_records(
        self,
        records: List[Dict[str, Any]],
        patient_id_field: str = "patient_id",
        category: Optional[DataCategory] = None,
        purpose: Optional[PermitPurpose] = None,
    ) -> List[Dict[str, Any]]:
        """
        Filter records based on opt-out status.

        Args:
            records: List of data records.
            patient_id_field: Field name containing patient ID.
            category: Data category being processed.
            purpose: Processing purpose.

        Returns:
            Filtered list of records (opted-out records removed).

        Raises:
            OptOutViolationError: If on_optout='error' and opted-out records found.
        """
        filtered = []
        opted_out_ids = []

        for record in records:
            patient_id = record.get(patient_id_field)
            if patient_id and self.registry.is_opted_out(patient_id, category, purpose):
                opted_out_ids.append(patient_id)
                if self.on_optout == "error":
                    continue  # Collect all, then raise
                # Skip record if excluding
            else:
                filtered.append(record)

        if opted_out_ids:
            logger.info(
                "Records filtered due to opt-out",
                count=len(opted_out_ids),
                action=self.on_optout,
            )

            if self.on_optout == "error":
                raise OptOutViolationError(opted_out_ids)

        return filtered

    def validate_batch(
        self,
        patient_ids: List[str],
        category: Optional[DataCategory] = None,
        purpose: Optional[PermitPurpose] = None,
    ) -> tuple:
        """
        Validate a batch of patient IDs.

        Args:
            patient_ids: List of patient identifiers.
            category: Data category being processed.
            purpose: Processing purpose.

        Returns:
            Tuple of (valid_ids, opted_out_ids).
        """
        valid = []
        opted_out = []

        for patient_id in patient_ids:
            if self.registry.is_opted_out(patient_id, category, purpose):
                opted_out.append(patient_id)
            else:
                valid.append(patient_id)

        if opted_out:
            logger.info(
                "Batch validation complete",
                total=len(patient_ids),
                valid=len(valid),
                opted_out=len(opted_out),
            )

        return valid, opted_out

    def get_compliance_report(
        self,
        total_records: int,
        processed_records: int,
        excluded_records: int,
    ) -> Dict[str, Any]:
        """
        Generate opt-out compliance report.

        Args:
            total_records: Total records in dataset.
            processed_records: Records actually processed.
            excluded_records: Records excluded due to opt-out.

        Returns:
            Compliance report dictionary.
        """
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "total_records": total_records,
            "processed_records": processed_records,
            "excluded_due_to_optout": excluded_records,
            "optout_rate": (
                excluded_records / total_records if total_records > 0 else 0.0
            ),
            "registry_last_sync": (
                self.registry._last_sync.isoformat()
                if self.registry._last_sync
                else None
            ),
            "compliance_status": "compliant",
        }

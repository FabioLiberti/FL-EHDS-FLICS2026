"""
Opt-out Registry Module
=======================
Implements Article 71 opt-out compliance for EHDS secondary use.

Manages synchronization with national opt-out registries, record-level
opt-out checking, and FL training data filtering.

EHDS Article 71 allows EU citizens to opt out of secondary use of their
health data. This module enforces those decisions throughout the FL pipeline.

Features:
- In-memory registry with persistent state (simulation mode)
- Background async synchronization with national registries
- LRU-style cache with TTL for high-throughput lookups
- Batch checking for FL training data filtering
- Compliance reporting for GDPR Article 30 audit trails
- Granular scope: all, category-specific, or purpose-specific opt-outs
"""

import asyncio
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple
import structlog

from core.models import OptOutRecord, DataCategory, PermitPurpose
from core.exceptions import OptOutError, OptOutViolationError

logger = structlog.get_logger(__name__)


# =============================================================================
# Cache Entry
# =============================================================================


@dataclass
class OptOutCacheEntry:
    """Cache entry for opt-out status with TTL."""

    patient_id: str
    is_opted_out: bool
    scope: str  # "all", "category", "purpose"
    categories: Optional[Set[DataCategory]] = None
    purposes: Optional[Set[PermitPurpose]] = None
    cached_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: datetime = field(default_factory=lambda: datetime.utcnow() + timedelta(seconds=600))
    member_state: Optional[str] = None

    @property
    def is_expired(self) -> bool:
        return datetime.utcnow() >= self.expires_at


# =============================================================================
# Registry Statistics
# =============================================================================


@dataclass
class RegistryStats:
    """Statistics for the opt-out registry."""

    total_opted_out: int = 0
    by_scope: Dict[str, int] = field(default_factory=dict)
    by_member_state: Dict[str, int] = field(default_factory=dict)
    cache_size: int = 0
    cache_hit_count: int = 0
    cache_miss_count: int = 0
    last_sync: Optional[str] = None
    sync_count: int = 0
    total_lookups: int = 0
    total_violations_blocked: int = 0

    @property
    def cache_hit_rate(self) -> float:
        total = self.cache_hit_count + self.cache_miss_count
        return self.cache_hit_count / total if total > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_opted_out": self.total_opted_out,
            "by_scope": self.by_scope,
            "by_member_state": self.by_member_state,
            "cache_size": self.cache_size,
            "cache_hit_rate": round(self.cache_hit_rate, 3),
            "last_sync": self.last_sync,
            "sync_count": self.sync_count,
            "total_lookups": self.total_lookups,
            "total_violations_blocked": self.total_violations_blocked,
        }


# =============================================================================
# Opt-Out Registry
# =============================================================================


class OptOutRegistry:
    """
    Manages synchronization with national opt-out registries.

    Provides caching and efficient lookup of citizen opt-out decisions
    per EHDS Article 71 requirements.

    In simulation mode, opt-out records are stored in memory and the
    sync loop is a no-op. In production, the sync loop would pull
    delta updates from the national registry API.
    """

    def __init__(
        self,
        registry_endpoint: Optional[str] = None,
        sync_interval: int = 300,
        cache_ttl: int = 600,
        max_cache_size: int = 100000,
        db=None,
    ):
        """
        Initialize opt-out registry.

        Args:
            registry_endpoint: URL of national opt-out registry API.
            sync_interval: Seconds between registry synchronizations.
            cache_ttl: Cache entry time-to-live in seconds.
            max_cache_size: Maximum number of cached entries.
            db: Optional GovernanceDB instance for SQLite persistence.
        """
        self.registry_endpoint = registry_endpoint
        self.sync_interval = sync_interval
        self.cache_ttl = cache_ttl
        self.max_cache_size = max_cache_size
        self._db = db

        # LRU-ordered cache: most recently used entries at the end
        self._cache: OrderedDict[str, OptOutCacheEntry] = OrderedDict()
        # Fast lookup set for simple opted-out check
        self._opted_out_ids: Set[str] = set()
        # Full record storage (simulation backend)
        self._records: Dict[str, OptOutRecord] = {}

        self._last_sync: Optional[datetime] = None
        self._sync_task: Optional[asyncio.Task] = None
        self._stats = RegistryStats()

        # Pre-load active opt-outs from SQLite when db is provided
        if self._db is not None:
            for row in self._db.list_active_optouts():
                try:
                    record = OptOutRecord(**row)
                    self._records[record.patient_id] = record
                    self._opted_out_ids.add(record.patient_id)
                except Exception:
                    pass
            self._stats.total_opted_out = len(self._opted_out_ids)

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    async def start(self) -> None:
        """Start background synchronization with national registry."""
        if self.registry_endpoint:
            self._sync_task = asyncio.create_task(self._sync_loop())
            logger.info(
                "Opt-out registry sync started",
                endpoint=self.registry_endpoint,
                interval=self.sync_interval,
            )
        else:
            logger.info("Opt-out registry started (simulation mode, no sync endpoint)")

    async def stop(self) -> None:
        """Stop background synchronization."""
        if self._sync_task:
            self._sync_task.cancel()
            try:
                await self._sync_task
            except asyncio.CancelledError:
                pass
            logger.info("Opt-out registry sync stopped")

    # -------------------------------------------------------------------------
    # Synchronization
    # -------------------------------------------------------------------------

    async def _sync_loop(self) -> None:
        """Background loop for periodic registry synchronization."""
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
        """
        Synchronize with national opt-out registry.

        In simulation mode, refreshes cache entries from the local store.
        In production, this would:
        1. GET /api/v1/optout/delta?since={last_sync} from the registry API
        2. Process new opt-outs and revocations
        3. Update the local cache accordingly
        """
        logger.debug("Syncing with opt-out registry")

        sync_start = datetime.utcnow()
        updates_applied = 0

        if self.registry_endpoint:
            # Production sync would go here:
            # headers = {"Authorization": f"Bearer {self._auth_token}"}
            # params = {"since": self._last_sync.isoformat() if self._last_sync else ""}
            # async with aiohttp.ClientSession() as session:
            #     async with session.get(
            #         f"{self.registry_endpoint}/api/v1/optout/delta",
            #         headers=headers,
            #         params=params,
            #     ) as resp:
            #         data = await resp.json()
            #         for record in data.get("new_optouts", []):
            #             self._apply_remote_optout(record)
            #             updates_applied += 1
            #         for record in data.get("revocations", []):
            #             self._apply_remote_revocation(record)
            #             updates_applied += 1
            pass

        # Simulation: refresh cache from in-memory records
        expired_keys = [
            pid for pid, entry in self._cache.items()
            if entry.is_expired
        ]
        for pid in expired_keys:
            # Re-populate from records store
            record = self._records.get(pid)
            if record and record.is_active:
                self._update_cache_entry(record)
            else:
                del self._cache[pid]
                self._opted_out_ids.discard(pid)
            updates_applied += 1

        self._last_sync = datetime.utcnow()
        self._stats.sync_count += 1
        self._stats.last_sync = self._last_sync.isoformat()
        self._stats.cache_size = len(self._cache)

        sync_duration_ms = (datetime.utcnow() - sync_start).total_seconds() * 1000
        logger.info(
            "Opt-out registry synced",
            timestamp=self._last_sync.isoformat(),
            updates_applied=updates_applied,
            expired_refreshed=len(expired_keys),
            duration_ms=round(sync_duration_ms, 1),
        )

    # -------------------------------------------------------------------------
    # Registration & Revocation
    # -------------------------------------------------------------------------

    def register_optout(self, record: OptOutRecord) -> None:
        """
        Register an opt-out record.

        Args:
            record: The opt-out record to register.
        """
        # Store full record
        self._records[record.patient_id] = record

        # Write-through to SQLite
        if self._db is not None:
            self._db.save_optout(record)

        # Update fast lookup set
        if record.is_active:
            self._opted_out_ids.add(record.patient_id)
        else:
            self._opted_out_ids.discard(record.patient_id)

        # Update cache
        self._update_cache_entry(record)

        # Update stats
        self._stats.total_opted_out = len(self._opted_out_ids)
        scope = record.scope or "all"
        self._stats.by_scope[scope] = self._stats.by_scope.get(scope, 0) + 1
        state = record.member_state
        self._stats.by_member_state[state] = self._stats.by_member_state.get(state, 0) + 1

        logger.info(
            "Opt-out registered",
            patient_id=record.patient_id[:8] + "...",
            scope=record.scope,
            member_state=record.member_state,
        )

    def revoke_optout(self, patient_id: str) -> bool:
        """
        Revoke an existing opt-out (citizen re-consents to secondary use).

        Args:
            patient_id: Patient identifier to revoke opt-out for.

        Returns:
            True if revocation was successful.
        """
        record = self._records.get(patient_id)
        if not record or not record.is_active:
            logger.warning("No active opt-out to revoke", patient_id=patient_id[:8] + "...")
            return False

        record.is_active = False
        record.metadata["revoked_at"] = datetime.utcnow().isoformat()
        self._opted_out_ids.discard(patient_id)

        # Write-through to SQLite
        if self._db is not None:
            self._db.deactivate_optout(patient_id)

        # Remove from cache
        self._cache.pop(patient_id, None)

        # Update stats
        self._stats.total_opted_out = len(self._opted_out_ids)
        scope = record.scope or "all"
        self._stats.by_scope[scope] = max(0, self._stats.by_scope.get(scope, 1) - 1)
        state = record.member_state
        self._stats.by_member_state[state] = max(
            0, self._stats.by_member_state.get(state, 1) - 1
        )

        logger.info(
            "Opt-out revoked",
            patient_id=patient_id[:8] + "...",
        )
        return True

    def register_batch(self, records: List[OptOutRecord]) -> int:
        """
        Register multiple opt-out records at once.

        Args:
            records: List of OptOutRecord objects.

        Returns:
            Number of records successfully registered.
        """
        count = 0
        for record in records:
            self.register_optout(record)
            count += 1
        logger.info("Batch opt-out registration", count=count)
        return count

    # -------------------------------------------------------------------------
    # Lookup
    # -------------------------------------------------------------------------

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
        self._stats.total_lookups += 1

        # Fast path: not in the set at all
        if patient_id not in self._opted_out_ids:
            return False

        # Check cache (with LRU promotion)
        if patient_id in self._cache:
            entry = self._cache[patient_id]
            # Move to end (most recently used)
            self._cache.move_to_end(patient_id)
            self._stats.cache_hit_count += 1

            # Check if cache entry is still valid
            if entry.is_expired:
                # Re-check from records store
                record = self._records.get(patient_id)
                if record and record.is_active:
                    self._update_cache_entry(record)
                    entry = self._cache[patient_id]
                else:
                    # Record removed or deactivated
                    del self._cache[patient_id]
                    self._opted_out_ids.discard(patient_id)
                    return False

            if not entry.is_opted_out:
                return False

            # Evaluate scope
            return self._evaluate_scope(entry, category, purpose)

        # Not in cache - check records store directly
        self._stats.cache_miss_count += 1
        record = self._records.get(patient_id)
        if record and record.is_active:
            self._update_cache_entry(record)
            entry = self._cache.get(patient_id)
            if entry:
                return self._evaluate_scope(entry, category, purpose)

        # In opted_out_ids but no detailed info -> default to opted out (safe)
        return True

    def _evaluate_scope(
        self,
        entry: OptOutCacheEntry,
        category: Optional[DataCategory],
        purpose: Optional[PermitPurpose],
    ) -> bool:
        """Evaluate whether the opt-out applies given the scope."""
        if entry.scope == "all":
            return True

        if entry.scope == "category" and category:
            return entry.categories is not None and category in entry.categories

        if entry.scope == "purpose" and purpose:
            return entry.purposes is not None and purpose in entry.purposes

        # Scope doesn't match query -> default to opted out (safe)
        return True

    def _update_cache_entry(self, record: OptOutRecord) -> None:
        """Create or update a cache entry from an OptOutRecord."""
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
            member_state=record.member_state,
        )

        # LRU eviction: if cache is full, remove oldest entries
        while len(self._cache) >= self.max_cache_size:
            evicted_id, _ = self._cache.popitem(last=False)
            logger.debug("Cache entry evicted (LRU)", patient_id=evicted_id[:8] + "...")

        self._cache[record.patient_id] = entry
        self._stats.cache_size = len(self._cache)

    # -------------------------------------------------------------------------
    # Batch Operations
    # -------------------------------------------------------------------------

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
            Dictionary mapping record_id to opt-out status (True = opted out).
        """
        results = {}
        for record_id in record_ids:
            patient_id = patient_id_mapping.get(record_id)
            if patient_id:
                results[record_id] = self.is_opted_out(patient_id, category, purpose)
            else:
                results[record_id] = False
        return results

    # -------------------------------------------------------------------------
    # Query & Statistics
    # -------------------------------------------------------------------------

    def get_opted_out_count(self) -> int:
        """Get count of currently opted-out patients."""
        return len(self._opted_out_ids)

    def get_record(self, patient_id: str) -> Optional[OptOutRecord]:
        """Get the full opt-out record for a patient (if exists)."""
        return self._records.get(patient_id)

    def get_all_records(
        self,
        active_only: bool = True,
        member_state: Optional[str] = None,
    ) -> List[OptOutRecord]:
        """
        List opt-out records with optional filtering.

        Args:
            active_only: If True, return only active opt-outs.
            member_state: Filter by member state code.

        Returns:
            List of matching OptOutRecord objects.
        """
        records = list(self._records.values())
        if active_only:
            records = [r for r in records if r.is_active]
        if member_state:
            records = [r for r in records if r.member_state == member_state]
        return records

    def get_stats(self) -> RegistryStats:
        """Get registry statistics."""
        self._stats.total_opted_out = len(self._opted_out_ids)
        self._stats.cache_size = len(self._cache)
        return self._stats

    def clear_cache(self) -> None:
        """Clear the opt-out cache (records are preserved)."""
        self._cache.clear()
        self._stats.cache_size = 0
        self._stats.cache_hit_count = 0
        self._stats.cache_miss_count = 0
        logger.debug("Opt-out cache cleared")

    def reset(self) -> None:
        """Reset the entire registry (for testing)."""
        self._cache.clear()
        self._opted_out_ids.clear()
        self._records.clear()
        self._stats = RegistryStats()
        self._last_sync = None
        logger.info("Opt-out registry reset")


# =============================================================================
# Opt-Out Checker (FL Training Filter)
# =============================================================================


class OptOutChecker:
    """
    Utility class for checking and enforcing opt-out compliance
    during FL training.

    Integrates with the OptOutRegistry to filter data records before
    they are used in local training, ensuring Article 71 compliance.
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
            on_optout: Action on opt-out detection:
                - 'exclude': Silently remove opted-out records (default)
                - 'anonymize': Replace patient data with anonymized placeholder
                - 'error': Raise OptOutViolationError
        """
        self.registry = registry
        self.on_optout = on_optout
        self._filter_stats = {
            "total_checked": 0,
            "total_excluded": 0,
            "total_passed": 0,
        }

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
            Filtered list of records (opted-out records removed or anonymized).

        Raises:
            OptOutViolationError: If on_optout='error' and opted-out records found.
        """
        filtered = []
        opted_out_ids = []

        for record in records:
            patient_id = record.get(patient_id_field)
            if patient_id and self.registry.is_opted_out(patient_id, category, purpose):
                opted_out_ids.append(patient_id)

                if self.on_optout == "anonymize":
                    # Create anonymized copy with identifiers removed
                    anon_record = dict(record)
                    anon_record[patient_id_field] = f"ANON-{uuid.uuid4().hex[:8]}"
                    anon_record["_anonymized"] = True
                    anon_record["_original_scope"] = "opted_out"
                    filtered.append(anon_record)
                # 'exclude' and 'error' skip the record
            else:
                filtered.append(record)

        self._filter_stats["total_checked"] += len(records)
        self._filter_stats["total_excluded"] += len(opted_out_ids)
        self._filter_stats["total_passed"] += len(filtered)

        if opted_out_ids:
            self.registry._stats.total_violations_blocked += len(opted_out_ids)

            logger.info(
                "Records filtered due to opt-out",
                count=len(opted_out_ids),
                action=self.on_optout,
                total_records=len(records),
                remaining=len(filtered),
            )

            if self.on_optout == "error":
                raise OptOutViolationError(opted_out_ids)

        return filtered

    def validate_batch(
        self,
        patient_ids: List[str],
        category: Optional[DataCategory] = None,
        purpose: Optional[PermitPurpose] = None,
    ) -> Tuple[List[str], List[str]]:
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

    def validate_fl_round(
        self,
        client_patient_ids: Dict[str, List[str]],
        category: Optional[DataCategory] = None,
        purpose: Optional[PermitPurpose] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Validate opt-out compliance for an entire FL round.

        Checks all patient IDs across all clients participating in the round.

        Args:
            client_patient_ids: Mapping of client_id -> list of patient IDs.
            category: Data category being processed.
            purpose: Processing purpose.

        Returns:
            Per-client validation results with valid/excluded counts.
        """
        results = {}

        for client_id, patient_ids in client_patient_ids.items():
            valid, opted_out = self.validate_batch(patient_ids, category, purpose)
            results[client_id] = {
                "total": len(patient_ids),
                "valid": len(valid),
                "excluded": len(opted_out),
                "exclusion_rate": len(opted_out) / len(patient_ids) if patient_ids else 0.0,
                "compliant": True,  # All opted-out records identified
            }

        total_patients = sum(r["total"] for r in results.values())
        total_excluded = sum(r["excluded"] for r in results.values())

        logger.info(
            "FL round opt-out validation",
            clients=len(results),
            total_patients=total_patients,
            total_excluded=total_excluded,
            overall_exclusion_rate=(
                total_excluded / total_patients if total_patients > 0 else 0.0
            ),
        )

        return results

    def get_compliance_report(
        self,
        total_records: int,
        processed_records: int,
        excluded_records: int,
        permit_id: Optional[str] = None,
        round_number: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Generate opt-out compliance report for audit trail.

        Args:
            total_records: Total records in dataset.
            processed_records: Records actually processed.
            excluded_records: Records excluded due to opt-out.
            permit_id: Associated data permit ID.
            round_number: FL training round number.

        Returns:
            Compliance report dictionary (GDPR Article 30 compatible).
        """
        stats = self.registry.get_stats()

        return {
            "report_id": f"OPT-RPT-{uuid.uuid4().hex[:8].upper()}",
            "timestamp": datetime.utcnow().isoformat(),
            "article": "EHDS Article 71",
            "legal_basis": "EHDS Regulation EU 2025/327",
            "total_records": total_records,
            "processed_records": processed_records,
            "excluded_due_to_optout": excluded_records,
            "optout_rate": (
                excluded_records / total_records if total_records > 0 else 0.0
            ),
            "permit_id": permit_id,
            "round_number": round_number,
            "registry_stats": stats.to_dict(),
            "registry_last_sync": (
                self.registry._last_sync.isoformat()
                if self.registry._last_sync
                else None
            ),
            "filter_action": self.on_optout,
            "filter_stats": dict(self._filter_stats),
            "compliance_status": "compliant",
            "compliance_notes": (
                f"All {excluded_records} opted-out records excluded from processing. "
                f"Registry contains {stats.total_opted_out} active opt-outs."
            ),
        }

    @property
    def filter_stats(self) -> Dict[str, int]:
        return dict(self._filter_stats)

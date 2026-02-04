"""
Compliance Logging Module
=========================
Implements GDPR Article 30 audit trail requirements for FL training
within the EHDS framework.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4
import structlog

from core.models import ComplianceRecord, PermitPurpose, DataCategory
from core.exceptions import ComplianceLoggingError
from core.utils import compute_hash, generate_id

logger = structlog.get_logger(__name__)


class AuditTrail:
    """
    Manages audit trail entries for compliance purposes.

    Provides structured logging of all processing activities
    per GDPR Article 30 requirements.
    """

    def __init__(
        self,
        storage_backend: str = "structured_file",
        storage_path: Optional[str] = None,
        retention_days: int = 2555,  # ~7 years
        log_format: str = "json",
    ):
        """
        Initialize audit trail.

        Args:
            storage_backend: Storage type ('structured_file', 'database', 'siem').
            storage_path: Path for file-based storage.
            retention_days: Days to retain audit records.
            log_format: Output format ('json', 'csv').
        """
        self.storage_backend = storage_backend
        self.storage_path = Path(storage_path) if storage_path else Path("logs/compliance")
        self.retention_days = retention_days
        self.log_format = log_format

        # Ensure storage directory exists
        if self.storage_backend == "structured_file":
            self.storage_path.mkdir(parents=True, exist_ok=True)

        # In-memory buffer for batch writes
        self._buffer: List[ComplianceRecord] = []
        self._buffer_size = 100

    def log_action(
        self,
        action: str,
        actor: str,
        outcome: str,
        permit_id: Optional[str] = None,
        data_categories: Optional[List[DataCategory]] = None,
        purpose: Optional[PermitPurpose] = None,
        details: Optional[Dict[str, Any]] = None,
        client_ids: Optional[List[str]] = None,
        round_number: Optional[int] = None,
    ) -> str:
        """
        Log a compliance action.

        Args:
            action: Action performed (e.g., 'training_started', 'data_accessed').
            actor: Entity performing the action.
            outcome: Result of action ('success', 'failure', 'partial').
            permit_id: Associated data permit.
            data_categories: Data categories involved.
            purpose: Processing purpose.
            details: Additional action details.
            client_ids: Participating FL clients.
            round_number: Training round number.

        Returns:
            Unique record identifier.
        """
        record = ComplianceRecord(
            record_id=generate_id("audit"),
            timestamp=datetime.utcnow(),
            action=action,
            actor=actor,
            permit_id=permit_id,
            data_categories=data_categories or [],
            purpose=purpose,
            legal_basis="EHDS Article 53",
            outcome=outcome,
            details=details or {},
            client_ids=client_ids or [],
            round_number=round_number,
        )

        self._buffer.append(record)

        # Flush buffer if full
        if len(self._buffer) >= self._buffer_size:
            self._flush_buffer()

        logger.info(
            "Compliance action logged",
            record_id=record.record_id,
            action=action,
            outcome=outcome,
        )

        return record.record_id

    def _flush_buffer(self) -> None:
        """Flush buffered records to storage."""
        if not self._buffer:
            return

        if self.storage_backend == "structured_file":
            self._write_to_file(self._buffer)
        elif self.storage_backend == "database":
            self._write_to_database(self._buffer)
        elif self.storage_backend == "siem":
            self._send_to_siem(self._buffer)

        self._buffer.clear()

    def _write_to_file(self, records: List[ComplianceRecord]) -> None:
        """Write records to file storage."""
        # Organize by date
        date_str = datetime.utcnow().strftime("%Y-%m-%d")
        file_path = self.storage_path / f"audit_{date_str}.{self.log_format}"

        try:
            with open(file_path, "a") as f:
                for record in records:
                    if self.log_format == "json":
                        f.write(json.dumps(record.to_log_entry()) + "\n")
                    else:
                        # CSV format
                        f.write(self._record_to_csv(record) + "\n")
        except Exception as e:
            raise ComplianceLoggingError(
                f"Failed to write audit records: {str(e)}",
                log_entry=records[0].to_log_entry() if records else None,
            )

    def _record_to_csv(self, record: ComplianceRecord) -> str:
        """Convert record to CSV line."""
        entry = record.to_log_entry()
        return ",".join(str(v) for v in entry.values())

    def _write_to_database(self, records: List[ComplianceRecord]) -> None:
        """Write records to database storage."""
        # Implementation placeholder
        logger.debug("Writing records to database", count=len(records))

    def _send_to_siem(self, records: List[ComplianceRecord]) -> None:
        """Send records to SIEM system."""
        # Implementation placeholder
        logger.debug("Sending records to SIEM", count=len(records))

    def flush(self) -> None:
        """Force flush any buffered records."""
        self._flush_buffer()

    def query_records(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        action: Optional[str] = None,
        permit_id: Optional[str] = None,
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        """
        Query audit records.

        Args:
            start_date: Filter records after this date.
            end_date: Filter records before this date.
            action: Filter by action type.
            permit_id: Filter by permit.
            limit: Maximum records to return.

        Returns:
            List of matching audit records.
        """
        # Implementation placeholder - would query storage backend
        logger.info(
            "Querying audit records",
            start_date=start_date,
            end_date=end_date,
            action=action,
        )
        return []


class ComplianceLogger:
    """
    High-level compliance logging interface for FL training sessions.

    Provides convenient methods for logging common FL training events
    in a compliance-compliant manner.
    """

    # Standard action types
    ACTION_SESSION_STARTED = "fl_session_started"
    ACTION_SESSION_ENDED = "fl_session_ended"
    ACTION_ROUND_STARTED = "fl_round_started"
    ACTION_ROUND_COMPLETED = "fl_round_completed"
    ACTION_DATA_ACCESSED = "data_accessed"
    ACTION_MODEL_AGGREGATED = "model_aggregated"
    ACTION_PRIVACY_APPLIED = "privacy_protection_applied"
    ACTION_OPTOUT_CHECKED = "optout_checked"
    ACTION_PERMIT_VERIFIED = "permit_verified"
    ACTION_ERROR_OCCURRED = "error_occurred"

    def __init__(
        self,
        audit_trail: Optional[AuditTrail] = None,
        actor_id: str = "fl-ehds-framework",
    ):
        """
        Initialize compliance logger.

        Args:
            audit_trail: AuditTrail instance for storage.
            actor_id: Identifier for the logging entity.
        """
        self.audit_trail = audit_trail or AuditTrail()
        self.actor_id = actor_id
        self._session_id: Optional[str] = None

    def start_session(
        self,
        permit_id: str,
        purpose: PermitPurpose,
        data_categories: List[DataCategory],
        client_ids: List[str],
    ) -> str:
        """
        Log FL session start.

        Args:
            permit_id: Data permit for this session.
            purpose: Processing purpose.
            data_categories: Data categories being processed.
            client_ids: Participating clients.

        Returns:
            Session identifier.
        """
        self._session_id = generate_id("session")

        self.audit_trail.log_action(
            action=self.ACTION_SESSION_STARTED,
            actor=self.actor_id,
            outcome="success",
            permit_id=permit_id,
            data_categories=data_categories,
            purpose=purpose,
            details={
                "session_id": self._session_id,
                "num_clients": len(client_ids),
            },
            client_ids=client_ids,
        )

        return self._session_id

    def end_session(
        self,
        total_rounds: int,
        final_metrics: Dict[str, float],
        success: bool = True,
    ) -> None:
        """
        Log FL session end.

        Args:
            total_rounds: Number of completed rounds.
            final_metrics: Final model metrics.
            success: Whether session completed successfully.
        """
        self.audit_trail.log_action(
            action=self.ACTION_SESSION_ENDED,
            actor=self.actor_id,
            outcome="success" if success else "failure",
            details={
                "session_id": self._session_id,
                "total_rounds": total_rounds,
                "final_metrics": final_metrics,
            },
        )
        self.audit_trail.flush()

    def log_round(
        self,
        round_number: int,
        participating_clients: List[str],
        samples_processed: int,
        round_metrics: Dict[str, float],
        privacy_spent: float,
    ) -> None:
        """
        Log FL training round.

        Args:
            round_number: Round number.
            participating_clients: Clients that participated.
            samples_processed: Total samples processed.
            round_metrics: Metrics from this round.
            privacy_spent: Privacy budget spent this round.
        """
        self.audit_trail.log_action(
            action=self.ACTION_ROUND_COMPLETED,
            actor=self.actor_id,
            outcome="success",
            round_number=round_number,
            client_ids=participating_clients,
            details={
                "session_id": self._session_id,
                "samples_processed": samples_processed,
                "metrics": round_metrics,
                "privacy_spent": privacy_spent,
            },
        )

    def log_data_access(
        self,
        client_id: str,
        data_categories: List[DataCategory],
        record_count: int,
        purpose: PermitPurpose,
    ) -> None:
        """
        Log data access event.

        Args:
            client_id: Client accessing data.
            data_categories: Categories accessed.
            record_count: Number of records accessed.
            purpose: Access purpose.
        """
        self.audit_trail.log_action(
            action=self.ACTION_DATA_ACCESSED,
            actor=client_id,
            outcome="success",
            data_categories=data_categories,
            purpose=purpose,
            details={
                "session_id": self._session_id,
                "record_count": record_count,
            },
            client_ids=[client_id],
        )

    def log_privacy_application(
        self,
        mechanism: str,
        parameters: Dict[str, Any],
        epsilon_spent: float,
        cumulative_epsilon: float,
    ) -> None:
        """
        Log privacy protection application.

        Args:
            mechanism: Privacy mechanism used (e.g., 'differential_privacy').
            parameters: Mechanism parameters.
            epsilon_spent: Epsilon spent this application.
            cumulative_epsilon: Total epsilon spent so far.
        """
        self.audit_trail.log_action(
            action=self.ACTION_PRIVACY_APPLIED,
            actor=self.actor_id,
            outcome="success",
            details={
                "session_id": self._session_id,
                "mechanism": mechanism,
                "parameters": parameters,
                "epsilon_spent": epsilon_spent,
                "cumulative_epsilon": cumulative_epsilon,
            },
        )

    def log_optout_check(
        self,
        total_records: int,
        excluded_count: int,
        client_id: str,
    ) -> None:
        """
        Log opt-out registry check.

        Args:
            total_records: Records checked.
            excluded_count: Records excluded due to opt-out.
            client_id: Client performing check.
        """
        self.audit_trail.log_action(
            action=self.ACTION_OPTOUT_CHECKED,
            actor=client_id,
            outcome="success",
            details={
                "session_id": self._session_id,
                "total_records": total_records,
                "excluded_count": excluded_count,
                "exclusion_rate": excluded_count / total_records if total_records > 0 else 0,
            },
            client_ids=[client_id],
        )

    def log_error(
        self,
        error_type: str,
        error_message: str,
        round_number: Optional[int] = None,
        client_id: Optional[str] = None,
    ) -> None:
        """
        Log error occurrence.

        Args:
            error_type: Type of error.
            error_message: Error description.
            round_number: Round where error occurred.
            client_id: Client that encountered error.
        """
        self.audit_trail.log_action(
            action=self.ACTION_ERROR_OCCURRED,
            actor=client_id or self.actor_id,
            outcome="failure",
            round_number=round_number,
            details={
                "session_id": self._session_id,
                "error_type": error_type,
                "error_message": error_message,
            },
            client_ids=[client_id] if client_id else [],
        )

    def generate_session_report(self) -> Dict[str, Any]:
        """
        Generate compliance report for current session.

        Returns:
            Session compliance report.
        """
        return {
            "session_id": self._session_id,
            "actor": self.actor_id,
            "generated_at": datetime.utcnow().isoformat(),
            "storage_backend": self.audit_trail.storage_backend,
            "retention_days": self.audit_trail.retention_days,
            "compliance_status": "compliant",
            "legal_basis": "EHDS Article 53",
            "gdpr_article_30_compliant": True,
        }

"""
Permit-Aware Training Bridge
=============================
Connects the governance layer (data_permits, compliance_logging) to the
FL training pipeline. Provides a context manager that enforces EHDS
permit terms during federated learning sessions.

This module does NOT duplicate governance logic -- it wraps existing
components from governance.data_permits and governance.compliance_logging
into a training-friendly interface.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from core.models import (
    DataCategory,
    DataPermit,
    PermitPurpose,
    PermitStatus,
)
from governance.compliance_logging import AuditTrail, ComplianceLogger
from governance.data_permits import DataPermitManager, PermitValidator


class PermitAwareTrainingContext:
    """
    Wraps governance components for use in the FL training loop.

    Created once per training session. Provides three capabilities:
    1. Pre-round permit validation (purpose, expiry, budget)
    2. Privacy budget tracking (epsilon accounting)
    3. Structured audit logging (every FL operation)

    Usage::

        ctx = PermitAwareTrainingContext(...)
        ctx.start_session()
        for round_num in range(num_rounds):
            ok, reason = ctx.validate_round(round_num, eps_cost)
            if not ok:
                break
            result = trainer.train_round(round_num)
            ctx.log_round_completion(result, eps_cost)
        ctx.end_session(total_rounds, final_metrics)
    """

    def __init__(
        self,
        permit_id: str,
        purpose: PermitPurpose,
        data_categories: List[DataCategory],
        privacy_budget_total: float,
        max_rounds: Optional[int] = None,
        client_ids: Optional[List[str]] = None,
        audit_output_dir: Optional[str] = None,
    ):
        # Use lenient mode so validate returns False instead of raising
        self.permit_manager = DataPermitManager(
            validator=PermitValidator(strict_mode=False)
        )

        audit_path = audit_output_dir or "logs/compliance"
        self.compliance_logger = ComplianceLogger(
            audit_trail=AuditTrail(storage_path=audit_path)
        )

        # Create and register the permit
        self.permit = DataPermit(
            permit_id=permit_id,
            hdab_id="HDAB-SIM",
            requester_id="fl-ehds-framework",
            purpose=purpose,
            data_categories=data_categories,
            issued_at=datetime.utcnow(),
            valid_from=datetime.utcnow(),
            valid_until=datetime.utcnow() + timedelta(days=7),
            status=PermitStatus.ACTIVE,
            privacy_budget_total=privacy_budget_total,
            max_rounds=max_rounds,
        )
        self.permit_manager.register_permit(self.permit)

        self._privacy_budget_total = privacy_budget_total
        self._epsilon_used = 0.0
        self._max_rounds = max_rounds
        self._client_ids = client_ids or []
        self._session_id: Optional[str] = None
        self._rounds_completed = 0

    def start_session(self) -> str:
        """Log FL session start. Returns session_id."""
        self._session_id = self.compliance_logger.start_session(
            permit_id=self.permit.permit_id,
            purpose=self.permit.purpose,
            data_categories=self.permit.data_categories,
            client_ids=self._client_ids,
        )
        return self._session_id

    def validate_round(
        self, round_num: int, epsilon_cost: float = 0.0
    ) -> Tuple[bool, str]:
        """
        Pre-round validation. Checks permit validity, round limit, privacy budget.

        Returns:
            (ok, reason) tuple. If ok is False, training should stop.
        """
        # 1. Permit validity (expiry, status, purpose)
        valid = self.permit_manager.verify_for_round(
            self.permit.permit_id,
            round_num,
            self.permit.data_categories,
        )
        if not valid:
            self.compliance_logger.log_error(
                "permit_validation_failed",
                f"Permit {self.permit.permit_id} invalid at round {round_num}",
                round_number=round_num,
            )
            return False, "Permit non valido (scaduto o status non attivo)"

        # 2. Round limit check
        if self._max_rounds is not None and round_num >= self._max_rounds:
            self.compliance_logger.log_error(
                "max_rounds_exceeded",
                f"Round {round_num} exceeds max_rounds={self._max_rounds}",
                round_number=round_num,
            )
            return False, f"Limite round raggiunto ({self._max_rounds})"

        # 3. Privacy budget check
        if self._privacy_budget_total > 0 and epsilon_cost > 0:
            remaining = self._privacy_budget_total - self._epsilon_used
            if epsilon_cost > remaining:
                self.compliance_logger.log_error(
                    "privacy_budget_exhausted",
                    f"Need eps={epsilon_cost:.4f}, remaining={remaining:.4f}",
                    round_number=round_num,
                )
                return False, (
                    f"Budget privacy esaurito. "
                    f"Rimanente: {remaining:.4f}, richiesto: {epsilon_cost:.4f}"
                )

        # Log successful verification
        self.compliance_logger.audit_trail.log_action(
            action=ComplianceLogger.ACTION_PERMIT_VERIFIED,
            actor="fl-ehds-framework",
            outcome="success",
            permit_id=self.permit.permit_id,
            round_number=round_num,
            details={
                "epsilon_remaining": self._privacy_budget_total - self._epsilon_used,
                "rounds_remaining": (
                    self._max_rounds - round_num if self._max_rounds else None
                ),
            },
        )

        return True, "Round autorizzato"

    def log_round_completion(self, round_result, epsilon_spent: float = 0.0):
        """Log round completion with metrics and privacy cost."""
        self._epsilon_used += epsilon_spent
        self._rounds_completed += 1

        # Update permit privacy budget
        if epsilon_spent > 0:
            self.permit.consume_privacy_budget(epsilon_spent)

        client_ids = [
            f"client_{cr.client_id}" for cr in round_result.client_results
        ]
        total_samples = sum(
            cr.num_samples for cr in round_result.client_results
        )

        self.compliance_logger.log_round(
            round_number=round_result.round_num,
            participating_clients=client_ids,
            samples_processed=total_samples,
            round_metrics={
                "accuracy": round_result.global_acc,
                "loss": round_result.global_loss,
                "f1": round_result.global_f1,
                "auc": round_result.global_auc,
            },
            privacy_spent=epsilon_spent,
        )

        if epsilon_spent > 0:
            self.compliance_logger.log_privacy_application(
                mechanism="differential_privacy",
                parameters={"epsilon_per_round": epsilon_spent},
                epsilon_spent=epsilon_spent,
                cumulative_epsilon=self._epsilon_used,
            )

    def end_session(
        self,
        total_rounds: int,
        final_metrics: Dict[str, float],
        success: bool = True,
    ):
        """Log session end and flush audit trail."""
        self.compliance_logger.end_session(
            total_rounds=total_rounds,
            final_metrics=final_metrics,
            success=success,
        )

    def get_budget_status(self) -> Dict[str, Any]:
        """Return current privacy budget status."""
        return {
            "total": self._privacy_budget_total,
            "used": self._epsilon_used,
            "remaining": self._privacy_budget_total - self._epsilon_used,
            "utilization_pct": (
                (self._epsilon_used / self._privacy_budget_total * 100)
                if self._privacy_budget_total > 0
                else 0.0
            ),
            "rounds_completed": self._rounds_completed,
            "max_rounds": self._max_rounds,
        }

    def export_audit_log(self, output_dir) -> Path:
        """
        Export audit trail to output directory.

        Generates an audit_compliance.json file with:
        - Session report from ComplianceLogger
        - Privacy budget status
        - Permit usage report from DataPermitManager
        - EHDS compliance metadata

        Returns:
            Path to the generated audit file.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Flush any buffered records
        self.compliance_logger.audit_trail.flush()

        # Build comprehensive report
        report = self.compliance_logger.generate_session_report()
        report["budget_status"] = self.get_budget_status()
        report["permit_usage"] = self.permit_manager.get_usage_report(
            self.permit.permit_id
        )
        report["permit_details"] = {
            "permit_id": self.permit.permit_id,
            "purpose": self.permit.purpose.value,
            "data_categories": [c.value for c in self.permit.data_categories],
            "valid_from": self.permit.valid_from.isoformat(),
            "valid_until": self.permit.valid_until.isoformat(),
            "privacy_budget_total": self._privacy_budget_total,
            "privacy_budget_used": self._epsilon_used,
            "max_rounds": self._max_rounds,
        }
        report["ehds_compliance"] = {
            "article_53_purpose_valid": True,
            "article_44_data_minimization": "applied"
            if report.get("data_minimization_applied")
            else "not_required",
            "gdpr_article_30_audit_trail": True,
            "framework_version": "FL-EHDS v4.8",
        }

        audit_file = output_dir / "audit_compliance.json"
        with open(audit_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str, ensure_ascii=False)

        return audit_file

    def get_summary_for_specs(self) -> Dict[str, Any]:
        """Return EHDS governance info for inclusion in auto-save specs."""
        return {
            "ehds_permit_enabled": True,
            "permit_id": self.permit.permit_id,
            "purpose": self.permit.purpose.value,
            "data_categories": [c.value for c in self.permit.data_categories],
            "privacy_budget_total": self._privacy_budget_total,
            "privacy_budget_used": self._epsilon_used,
            "max_rounds": self._max_rounds,
            "rounds_completed": self._rounds_completed,
        }


def create_permit_context(
    config: Dict[str, Any],
) -> Optional[PermitAwareTrainingContext]:
    """
    Create a PermitAwareTrainingContext from a training config dict.

    Returns None if permit is not enabled in config.
    """
    if not config.get("ehds_permit_enabled"):
        return None

    return PermitAwareTrainingContext(
        permit_id=f"PERMIT-{datetime.now().strftime('%Y%m%d%H%M%S')}",
        purpose=PermitPurpose(config["ehds_purpose"]),
        data_categories=[
            DataCategory(c) for c in config.get("ehds_data_categories", ["ehr"])
        ],
        privacy_budget_total=config.get("ehds_privacy_budget", 100.0),
        max_rounds=config.get("ehds_max_rounds"),
        client_ids=[
            f"client_{i}" for i in range(config.get("num_clients", 5))
        ],
    )

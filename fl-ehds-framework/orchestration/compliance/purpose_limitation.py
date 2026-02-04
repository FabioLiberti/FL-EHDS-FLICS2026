"""
Purpose Limitation Module
=========================
Technical enforcement of EHDS Article 53 permitted purposes
and output controls for model results.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Set
import structlog

from core.models import PermitPurpose, DataCategory
from core.exceptions import PurposeLimitationViolationError, ComplianceViolationError

logger = structlog.get_logger(__name__)


class PurposeLimiter:
    """
    Enforces purpose limitation per EHDS Article 53.

    Ensures FL training and model outputs are restricted to
    permitted purposes defined in data permits.
    """

    # Permitted purposes under EHDS Article 53
    PERMITTED_PURPOSES: Set[PermitPurpose] = {
        PermitPurpose.SCIENTIFIC_RESEARCH,
        PermitPurpose.PUBLIC_HEALTH_SURVEILLANCE,
        PermitPurpose.HEALTH_POLICY,
        PermitPurpose.EDUCATION_TRAINING,
        PermitPurpose.AI_SYSTEM_DEVELOPMENT,
        PermitPurpose.PERSONALIZED_MEDICINE,
        PermitPurpose.OFFICIAL_STATISTICS,
        PermitPurpose.PATIENT_SAFETY,
    }

    # Prohibited purposes (explicit exclusions)
    PROHIBITED_PURPOSES: Set[str] = {
        "insurance_underwriting",
        "employment_decisions",
        "direct_marketing",
        "credit_scoring",
        "law_enforcement",
        "immigration_control",
    }

    def __init__(
        self,
        permitted_purposes: Optional[List[PermitPurpose]] = None,
        verify_on_init: bool = True,
        verify_per_round: bool = False,
        on_violation: str = "abort",
    ):
        """
        Initialize purpose limiter.

        Args:
            permitted_purposes: Specific permitted purposes (default: all Article 53).
            verify_on_init: Verify purpose at session start.
            verify_per_round: Verify purpose each training round.
            on_violation: Action on violation ('abort', 'warn', 'log').
        """
        self.permitted_purposes = (
            set(permitted_purposes) if permitted_purposes else self.PERMITTED_PURPOSES
        )
        self.verify_on_init = verify_on_init
        self.verify_per_round = verify_per_round
        self.on_violation = on_violation

        # Session state
        self._session_purpose: Optional[PermitPurpose] = None
        self._verified_at: Optional[datetime] = None
        self._violation_count = 0

    def set_session_purpose(self, purpose: PermitPurpose) -> bool:
        """
        Set and verify the purpose for current FL session.

        Args:
            purpose: Purpose for this session.

        Returns:
            True if purpose is valid.

        Raises:
            PurposeLimitationViolationError: If purpose not permitted.
        """
        if purpose not in self.permitted_purposes:
            self._handle_violation(
                f"Purpose '{purpose.value}' not in permitted purposes",
                purpose,
            )
            return False

        self._session_purpose = purpose
        self._verified_at = datetime.utcnow()

        logger.info(
            "Session purpose set",
            purpose=purpose.value,
            verified_at=self._verified_at.isoformat(),
        )

        return True

    def verify_purpose(
        self,
        requested_purpose: PermitPurpose,
        round_number: Optional[int] = None,
    ) -> bool:
        """
        Verify a requested purpose is permitted.

        Args:
            requested_purpose: Purpose to verify.
            round_number: Current training round (for logging).

        Returns:
            True if purpose is valid.

        Raises:
            PurposeLimitationViolationError: If on_violation='abort'.
        """
        # Check against explicit prohibitions
        if requested_purpose.value in self.PROHIBITED_PURPOSES:
            self._handle_violation(
                f"Purpose '{requested_purpose.value}' is explicitly prohibited",
                requested_purpose,
            )
            return False

        # Check against permitted purposes
        if requested_purpose not in self.permitted_purposes:
            self._handle_violation(
                f"Purpose '{requested_purpose.value}' not permitted under Article 53",
                requested_purpose,
            )
            return False

        # Check against session purpose if set
        if self._session_purpose and requested_purpose != self._session_purpose:
            self._handle_violation(
                f"Purpose mismatch: session={self._session_purpose.value}, "
                f"requested={requested_purpose.value}",
                requested_purpose,
            )
            return False

        logger.debug(
            "Purpose verified",
            purpose=requested_purpose.value,
            round=round_number,
        )

        return True

    def verify_data_access(
        self,
        purpose: PermitPurpose,
        data_categories: List[DataCategory],
    ) -> bool:
        """
        Verify data access is aligned with purpose.

        Args:
            purpose: Access purpose.
            data_categories: Categories being accessed.

        Returns:
            True if access is permitted.
        """
        # Verify purpose first
        if not self.verify_purpose(purpose):
            return False

        # Check for sensitive data restrictions
        sensitive_categories = {DataCategory.GENOMIC}

        if sensitive_categories & set(data_categories):
            # Additional restrictions for sensitive data
            if purpose not in {
                PermitPurpose.SCIENTIFIC_RESEARCH,
                PermitPurpose.PERSONALIZED_MEDICINE,
            }:
                logger.warning(
                    "Sensitive data access restricted",
                    purpose=purpose.value,
                    sensitive_categories=[
                        c.value for c in sensitive_categories & set(data_categories)
                    ],
                )
                return False

        return True

    def _handle_violation(
        self,
        message: str,
        purpose: PermitPurpose,
    ) -> None:
        """Handle purpose limitation violation."""
        self._violation_count += 1

        if self.on_violation == "abort":
            raise PurposeLimitationViolationError(
                purpose.value,
                [p.value for p in self.permitted_purposes],
            )

        elif self.on_violation == "warn":
            logger.warning(
                "Purpose limitation violation",
                message=message,
                purpose=purpose.value,
                violation_count=self._violation_count,
            )

        else:  # log
            logger.info(
                "Purpose limitation check failed",
                message=message,
                purpose=purpose.value,
            )

    def get_violation_count(self) -> int:
        """Get total violation count."""
        return self._violation_count


class OutputController:
    """
    Controls FL model outputs for compliance.

    Ensures model outputs meet minimum aggregation requirements
    and are properly anonymized.
    """

    def __init__(
        self,
        min_aggregation_count: int = 5,
        anonymize_outputs: bool = True,
        allow_gradient_inspection: bool = False,
    ):
        """
        Initialize output controller.

        Args:
            min_aggregation_count: Minimum records for aggregated output.
            anonymize_outputs: Apply output anonymization.
            allow_gradient_inspection: Allow inspection of gradients.
        """
        self.min_aggregation_count = min_aggregation_count
        self.anonymize_outputs = anonymize_outputs
        self.allow_gradient_inspection = allow_gradient_inspection

    def validate_output(
        self,
        output_type: str,
        aggregation_count: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Validate model output meets compliance requirements.

        Args:
            output_type: Type of output ('model', 'gradients', 'metrics').
            aggregation_count: Number of records in aggregation.
            metadata: Additional output metadata.

        Returns:
            True if output is compliant.

        Raises:
            ComplianceViolationError: If output violates requirements.
        """
        # Check gradient inspection
        if output_type == "gradients" and not self.allow_gradient_inspection:
            raise ComplianceViolationError(
                "Gradient inspection is not permitted",
                violation_type="gradient_inspection",
                article="Privacy Protection",
            )

        # Check minimum aggregation
        if aggregation_count < self.min_aggregation_count:
            raise ComplianceViolationError(
                f"Output aggregation count ({aggregation_count}) below "
                f"minimum ({self.min_aggregation_count})",
                violation_type="insufficient_aggregation",
                article="Data Protection",
            )

        logger.debug(
            "Output validated",
            output_type=output_type,
            aggregation_count=aggregation_count,
        )

        return True

    def anonymize_model_output(
        self,
        model_output: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Apply anonymization to model output.

        Args:
            model_output: Raw model output.

        Returns:
            Anonymized output.
        """
        if not self.anonymize_outputs:
            return model_output

        anonymized = {}

        for key, value in model_output.items():
            # Remove potentially identifying metadata
            if key in ("client_ids", "patient_ids", "record_ids"):
                continue
            # Round numerical values to reduce precision
            if isinstance(value, (int, float)):
                anonymized[key] = round(value, 4)
            elif isinstance(value, list) and all(
                isinstance(v, (int, float)) for v in value
            ):
                anonymized[key] = [round(v, 4) for v in value]
            else:
                anonymized[key] = value

        return anonymized

    def filter_metrics(
        self,
        metrics: Dict[str, float],
        allowed_metrics: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Filter metrics to permitted set.

        Args:
            metrics: All computed metrics.
            allowed_metrics: Permitted metric names (None = all).

        Returns:
            Filtered metrics dictionary.
        """
        if allowed_metrics is None:
            return metrics

        return {k: v for k, v in metrics.items() if k in allowed_metrics}

    def generate_compliance_certificate(
        self,
        session_id: str,
        output_type: str,
        aggregation_count: int,
        purpose: PermitPurpose,
    ) -> Dict[str, Any]:
        """
        Generate compliance certificate for output.

        Args:
            session_id: FL session identifier.
            output_type: Type of output.
            aggregation_count: Records in aggregation.
            purpose: Processing purpose.

        Returns:
            Compliance certificate dictionary.
        """
        return {
            "certificate_type": "FL-EHDS Output Compliance",
            "session_id": session_id,
            "generated_at": datetime.utcnow().isoformat(),
            "output_type": output_type,
            "aggregation_count": aggregation_count,
            "min_required": self.min_aggregation_count,
            "purpose": purpose.value,
            "anonymization_applied": self.anonymize_outputs,
            "compliance_status": "compliant",
            "legal_basis": "EHDS Article 53",
        }

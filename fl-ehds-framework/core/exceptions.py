"""
FL-EHDS Custom Exceptions
=========================
Custom exception classes for the FL-EHDS framework.
"""

from typing import Optional, Any


class FLEHDSError(Exception):
    """Base exception for all FL-EHDS errors."""

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[dict] = None,
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}

    def __str__(self) -> str:
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message

    def to_dict(self) -> dict:
        """Convert exception to dictionary for logging/serialization."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "details": self.details,
        }


# =============================================================================
# Layer 1: Governance Exceptions
# =============================================================================


class GovernanceError(FLEHDSError):
    """Base exception for governance layer errors."""

    pass


class PermitError(GovernanceError):
    """Exception raised for data permit related errors."""

    def __init__(
        self,
        message: str,
        permit_id: Optional[str] = None,
        reason: Optional[str] = None,
    ):
        super().__init__(
            message,
            error_code="PERMIT_ERROR",
            details={"permit_id": permit_id, "reason": reason},
        )
        self.permit_id = permit_id
        self.reason = reason


class PermitNotFoundError(PermitError):
    """Exception raised when a data permit is not found."""

    def __init__(self, permit_id: str):
        super().__init__(
            f"Data permit not found: {permit_id}",
            permit_id=permit_id,
            reason="not_found",
        )


class PermitExpiredError(PermitError):
    """Exception raised when a data permit has expired."""

    def __init__(self, permit_id: str, expiry_date: str):
        super().__init__(
            f"Data permit expired: {permit_id} (expired on {expiry_date})",
            permit_id=permit_id,
            reason="expired",
        )
        self.expiry_date = expiry_date


class PermitPurposeMismatchError(PermitError):
    """Exception raised when requested purpose doesn't match permit."""

    def __init__(
        self, permit_id: str, requested_purpose: str, allowed_purposes: list
    ):
        super().__init__(
            f"Purpose mismatch for permit {permit_id}: "
            f"'{requested_purpose}' not in {allowed_purposes}",
            permit_id=permit_id,
            reason="purpose_mismatch",
        )
        self.requested_purpose = requested_purpose
        self.allowed_purposes = allowed_purposes


class OptOutError(GovernanceError):
    """Exception raised for opt-out registry related errors."""

    def __init__(
        self,
        message: str,
        record_ids: Optional[list] = None,
    ):
        super().__init__(
            message,
            error_code="OPTOUT_ERROR",
            details={"record_ids": record_ids},
        )
        self.record_ids = record_ids or []


class OptOutViolationError(OptOutError):
    """Exception raised when attempting to process opted-out records."""

    def __init__(self, record_ids: list):
        super().__init__(
            f"Attempted to process {len(record_ids)} opted-out record(s)",
            record_ids=record_ids,
        )


class HDABConnectionError(GovernanceError):
    """Exception raised when HDAB communication fails."""

    def __init__(self, hdab_endpoint: str, reason: str):
        super().__init__(
            f"Failed to connect to HDAB at {hdab_endpoint}: {reason}",
            error_code="HDAB_CONNECTION_ERROR",
            details={"endpoint": hdab_endpoint, "reason": reason},
        )


class ComplianceLoggingError(GovernanceError):
    """Exception raised when compliance logging fails."""

    def __init__(self, message: str, log_entry: Optional[dict] = None):
        super().__init__(
            message,
            error_code="COMPLIANCE_LOG_ERROR",
            details={"log_entry": log_entry},
        )


# =============================================================================
# Layer 2: Orchestration Exceptions
# =============================================================================


class OrchestrationError(FLEHDSError):
    """Base exception for FL orchestration layer errors."""

    pass


class AggregationError(OrchestrationError):
    """Exception raised during gradient aggregation."""

    def __init__(
        self,
        message: str,
        round_number: Optional[int] = None,
        participating_clients: Optional[int] = None,
    ):
        super().__init__(
            message,
            error_code="AGGREGATION_ERROR",
            details={
                "round_number": round_number,
                "participating_clients": participating_clients,
            },
        )


class InsufficientClientsError(AggregationError):
    """Exception raised when not enough clients participate."""

    def __init__(
        self, required: int, available: int, round_number: Optional[int] = None
    ):
        super().__init__(
            f"Insufficient clients: {available} available, {required} required",
            round_number=round_number,
            participating_clients=available,
        )
        self.required = required
        self.available = available


class PrivacyError(OrchestrationError):
    """Base exception for privacy-related errors."""

    pass


class PrivacyBudgetExceededError(PrivacyError):
    """Exception raised when privacy budget (epsilon) is exceeded."""

    def __init__(
        self,
        current_epsilon: float,
        max_epsilon: float,
        round_number: Optional[int] = None,
    ):
        super().__init__(
            f"Privacy budget exceeded: {current_epsilon:.4f} > {max_epsilon:.4f}",
            error_code="PRIVACY_BUDGET_EXCEEDED",
            details={
                "current_epsilon": current_epsilon,
                "max_epsilon": max_epsilon,
                "round_number": round_number,
            },
        )
        self.current_epsilon = current_epsilon
        self.max_epsilon = max_epsilon


class GradientClippingError(PrivacyError):
    """Exception raised during gradient clipping."""

    def __init__(self, message: str, gradient_norm: Optional[float] = None):
        super().__init__(
            message,
            error_code="GRADIENT_CLIPPING_ERROR",
            details={"gradient_norm": gradient_norm},
        )


class SecureAggregationError(PrivacyError):
    """Exception raised during secure aggregation protocol."""

    def __init__(self, message: str, phase: Optional[str] = None):
        super().__init__(
            message,
            error_code="SECURE_AGGREGATION_ERROR",
            details={"phase": phase},
        )


class ComplianceViolationError(OrchestrationError):
    """Exception raised when compliance rules are violated."""

    def __init__(
        self,
        message: str,
        violation_type: str,
        article: Optional[str] = None,
    ):
        super().__init__(
            message,
            error_code="COMPLIANCE_VIOLATION",
            details={"violation_type": violation_type, "article": article},
        )
        self.violation_type = violation_type
        self.article = article


class PurposeLimitationViolationError(ComplianceViolationError):
    """Exception raised when purpose limitation is violated (Article 53)."""

    def __init__(self, requested_purpose: str, allowed_purposes: list):
        super().__init__(
            f"Purpose limitation violation: '{requested_purpose}' not permitted",
            violation_type="purpose_limitation",
            article="Article 53",
        )
        self.requested_purpose = requested_purpose
        self.allowed_purposes = allowed_purposes


# =============================================================================
# Layer 3: Data Holder Exceptions
# =============================================================================


class DataHolderError(FLEHDSError):
    """Base exception for data holder layer errors."""

    pass


class TrainingError(DataHolderError):
    """Exception raised during local training."""

    def __init__(
        self,
        message: str,
        epoch: Optional[int] = None,
        batch: Optional[int] = None,
    ):
        super().__init__(
            message,
            error_code="TRAINING_ERROR",
            details={"epoch": epoch, "batch": batch},
        )


class ResourceConstraintError(TrainingError):
    """Exception raised when hardware resources are insufficient."""

    def __init__(
        self,
        resource_type: str,
        required: Any,
        available: Any,
    ):
        super().__init__(
            f"Insufficient {resource_type}: required {required}, available {available}",
        )
        self.resource_type = resource_type
        self.required = required
        self.available = available


class FHIRPreprocessingError(DataHolderError):
    """Exception raised during FHIR data preprocessing."""

    def __init__(
        self,
        message: str,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
    ):
        super().__init__(
            message,
            error_code="FHIR_PREPROCESSING_ERROR",
            details={"resource_type": resource_type, "resource_id": resource_id},
        )


class FHIRValidationError(FHIRPreprocessingError):
    """Exception raised when FHIR resource validation fails."""

    def __init__(
        self,
        resource_type: str,
        resource_id: str,
        validation_errors: list,
    ):
        super().__init__(
            f"FHIR validation failed for {resource_type}/{resource_id}",
            resource_type=resource_type,
            resource_id=resource_id,
        )
        self.validation_errors = validation_errors


class CommunicationError(DataHolderError):
    """Exception raised during secure communication."""

    def __init__(
        self,
        message: str,
        endpoint: Optional[str] = None,
        operation: Optional[str] = None,
    ):
        super().__init__(
            message,
            error_code="COMMUNICATION_ERROR",
            details={"endpoint": endpoint, "operation": operation},
        )


class EncryptionError(CommunicationError):
    """Exception raised during encryption/decryption."""

    def __init__(self, message: str, operation: str):
        super().__init__(
            message,
            operation=operation,
        )


class AuthenticationError(CommunicationError):
    """Exception raised during authentication."""

    def __init__(self, message: str, client_id: Optional[str] = None):
        super().__init__(
            message,
            operation="authentication",
        )
        self.client_id = client_id


# =============================================================================
# Configuration Exceptions
# =============================================================================


class ConfigurationError(FLEHDSError):
    """Exception raised for configuration errors."""

    def __init__(self, message: str, config_key: Optional[str] = None):
        super().__init__(
            message,
            error_code="CONFIGURATION_ERROR",
            details={"config_key": config_key},
        )

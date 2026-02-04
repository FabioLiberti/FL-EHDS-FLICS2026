"""
Data Permit Management Module
=============================
Manages data permits issued by HDABs for secondary use of health data
under EHDS Article 53.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Set
import structlog

from core.models import (
    DataPermit,
    PermitStatus,
    PermitPurpose,
    DataCategory,
)
from core.exceptions import (
    PermitError,
    PermitNotFoundError,
    PermitExpiredError,
    PermitPurposeMismatchError,
)

logger = structlog.get_logger(__name__)


class PermitValidator:
    """
    Validates data permits against EHDS requirements.

    Ensures permits are valid, not expired, and cover the intended
    processing purpose and data categories.
    """

    # Permitted purposes under EHDS Article 53
    ARTICLE_53_PURPOSES: Set[PermitPurpose] = {
        PermitPurpose.SCIENTIFIC_RESEARCH,
        PermitPurpose.PUBLIC_HEALTH_SURVEILLANCE,
        PermitPurpose.HEALTH_POLICY,
        PermitPurpose.EDUCATION_TRAINING,
        PermitPurpose.AI_SYSTEM_DEVELOPMENT,
        PermitPurpose.PERSONALIZED_MEDICINE,
        PermitPurpose.OFFICIAL_STATISTICS,
        PermitPurpose.PATIENT_SAFETY,
    }

    def __init__(
        self,
        strict_mode: bool = True,
        verify_expiry: bool = True,
        check_purpose_alignment: bool = True,
    ):
        """
        Initialize permit validator.

        Args:
            strict_mode: If True, raise exceptions on validation failure.
            verify_expiry: If True, check permit expiration dates.
            check_purpose_alignment: If True, verify purpose matches Article 53.
        """
        self.strict_mode = strict_mode
        self.verify_expiry = verify_expiry
        self.check_purpose_alignment = check_purpose_alignment

    def validate(
        self,
        permit: DataPermit,
        requested_purpose: Optional[PermitPurpose] = None,
        requested_categories: Optional[List[DataCategory]] = None,
    ) -> bool:
        """
        Validate a data permit.

        Args:
            permit: The permit to validate.
            requested_purpose: Optional specific purpose to validate against.
            requested_categories: Optional specific categories to validate against.

        Returns:
            True if permit is valid.

        Raises:
            PermitExpiredError: If permit has expired (strict mode).
            PermitPurposeMismatchError: If purpose doesn't match (strict mode).
            PermitError: For other validation failures (strict mode).
        """
        validation_results = {
            "status_valid": False,
            "not_expired": False,
            "purpose_permitted": False,
            "purpose_matches": False,
            "categories_covered": False,
        }

        # Check status
        validation_results["status_valid"] = permit.status == PermitStatus.ACTIVE

        # Check expiry
        if self.verify_expiry:
            now = datetime.utcnow()
            validation_results["not_expired"] = (
                permit.valid_from <= now <= permit.valid_until
            )
        else:
            validation_results["not_expired"] = True

        # Check purpose is permitted under Article 53
        if self.check_purpose_alignment:
            validation_results["purpose_permitted"] = (
                permit.purpose in self.ARTICLE_53_PURPOSES
            )
        else:
            validation_results["purpose_permitted"] = True

        # Check requested purpose matches permit
        if requested_purpose:
            validation_results["purpose_matches"] = permit.purpose == requested_purpose
        else:
            validation_results["purpose_matches"] = True

        # Check requested categories are covered
        if requested_categories:
            validation_results["categories_covered"] = permit.covers_categories(
                requested_categories
            )
        else:
            validation_results["categories_covered"] = True

        # Log validation results
        all_valid = all(validation_results.values())
        log_method = logger.info if all_valid else logger.warning
        log_method(
            "Permit validation complete",
            permit_id=permit.permit_id,
            valid=all_valid,
            results=validation_results,
        )

        # Handle failures in strict mode
        if self.strict_mode and not all_valid:
            if not validation_results["not_expired"]:
                raise PermitExpiredError(permit.permit_id, permit.valid_until.isoformat())

            if not validation_results["purpose_matches"] and requested_purpose:
                raise PermitPurposeMismatchError(
                    permit.permit_id,
                    requested_purpose.value,
                    [permit.purpose.value],
                )

            if not validation_results["status_valid"]:
                raise PermitError(
                    f"Permit {permit.permit_id} is not active (status: {permit.status.value})",
                    permit_id=permit.permit_id,
                    reason="invalid_status",
                )

            if not validation_results["categories_covered"]:
                raise PermitError(
                    f"Permit {permit.permit_id} does not cover requested data categories",
                    permit_id=permit.permit_id,
                    reason="category_mismatch",
                )

        return all_valid


class DataPermitManager:
    """
    Manages the lifecycle of data permits for FL training sessions.

    Handles permit registration, validation, usage tracking, and
    compliance reporting.
    """

    def __init__(
        self,
        validator: Optional[PermitValidator] = None,
        allowed_purposes: Optional[List[PermitPurpose]] = None,
        allowed_categories: Optional[List[DataCategory]] = None,
    ):
        """
        Initialize permit manager.

        Args:
            validator: Custom permit validator (uses default if None).
            allowed_purposes: Restrict to specific purposes (all if None).
            allowed_categories: Restrict to specific categories (all if None).
        """
        self.validator = validator or PermitValidator()
        self.allowed_purposes = (
            set(allowed_purposes) if allowed_purposes else None
        )
        self.allowed_categories = (
            set(allowed_categories) if allowed_categories else None
        )

        # Active permits for current session
        self._active_permits: Dict[str, DataPermit] = {}
        # Usage tracking
        self._permit_usage: Dict[str, Dict[str, Any]] = {}

    def register_permit(self, permit: DataPermit) -> bool:
        """
        Register a permit for use in FL training.

        Args:
            permit: The data permit to register.

        Returns:
            True if registration successful.

        Raises:
            PermitError: If permit validation fails.
        """
        # Validate permit
        if not self.validator.validate(permit):
            return False

        # Check against manager restrictions
        if self.allowed_purposes and permit.purpose not in self.allowed_purposes:
            raise PermitPurposeMismatchError(
                permit.permit_id,
                permit.purpose.value,
                [p.value for p in self.allowed_purposes],
            )

        if self.allowed_categories:
            disallowed = set(permit.data_categories) - self.allowed_categories
            if disallowed:
                raise PermitError(
                    f"Permit contains disallowed data categories: {disallowed}",
                    permit_id=permit.permit_id,
                    reason="category_restriction",
                )

        # Register permit
        self._active_permits[permit.permit_id] = permit
        self._permit_usage[permit.permit_id] = {
            "registered_at": datetime.utcnow(),
            "rounds_used": 0,
            "last_used": None,
            "data_accessed": set(),
        }

        logger.info(
            "Permit registered",
            permit_id=permit.permit_id,
            purpose=permit.purpose.value,
            valid_until=permit.valid_until.isoformat(),
        )
        return True

    def get_permit(self, permit_id: str) -> DataPermit:
        """
        Retrieve a registered permit.

        Args:
            permit_id: Permit identifier.

        Returns:
            The registered DataPermit.

        Raises:
            PermitNotFoundError: If permit is not registered.
        """
        if permit_id not in self._active_permits:
            raise PermitNotFoundError(permit_id)
        return self._active_permits[permit_id]

    def verify_for_round(
        self,
        permit_id: str,
        round_number: int,
        data_categories: List[DataCategory],
    ) -> bool:
        """
        Verify permit is valid for a training round.

        Args:
            permit_id: Permit to verify.
            round_number: Current training round.
            data_categories: Categories being processed this round.

        Returns:
            True if permit is valid for this round.
        """
        try:
            permit = self.get_permit(permit_id)
        except PermitNotFoundError:
            logger.error("Permit not registered", permit_id=permit_id)
            return False

        # Re-validate (check expiry hasn't occurred mid-training)
        if not self.validator.validate(
            permit,
            requested_purpose=permit.purpose,
            requested_categories=data_categories,
        ):
            return False

        # Update usage tracking
        self._permit_usage[permit_id]["rounds_used"] += 1
        self._permit_usage[permit_id]["last_used"] = datetime.utcnow()
        self._permit_usage[permit_id]["data_accessed"].update(
            c.value for c in data_categories
        )

        logger.debug(
            "Permit verified for round",
            permit_id=permit_id,
            round_number=round_number,
        )
        return True

    def deregister_permit(self, permit_id: str) -> bool:
        """
        Remove a permit from active use.

        Args:
            permit_id: Permit to deregister.

        Returns:
            True if permit was deregistered.
        """
        if permit_id in self._active_permits:
            del self._active_permits[permit_id]
            logger.info("Permit deregistered", permit_id=permit_id)
            return True
        return False

    def get_usage_report(self, permit_id: str) -> Dict[str, Any]:
        """
        Get usage report for a permit.

        Args:
            permit_id: Permit identifier.

        Returns:
            Dictionary containing usage statistics.
        """
        if permit_id not in self._permit_usage:
            return {}

        usage = self._permit_usage[permit_id]
        return {
            "permit_id": permit_id,
            "registered_at": usage["registered_at"].isoformat(),
            "rounds_used": usage["rounds_used"],
            "last_used": usage["last_used"].isoformat() if usage["last_used"] else None,
            "data_categories_accessed": list(usage["data_accessed"]),
        }

    def get_all_active_permits(self) -> List[DataPermit]:
        """Get all currently registered permits."""
        return list(self._active_permits.values())

    def check_any_permit_expiring(self, within_hours: int = 24) -> List[DataPermit]:
        """
        Check for permits expiring soon.

        Args:
            within_hours: Time window to check.

        Returns:
            List of permits expiring within the specified window.
        """
        from datetime import timedelta

        threshold = datetime.utcnow() + timedelta(hours=within_hours)
        expiring = []

        for permit in self._active_permits.values():
            if permit.valid_until < threshold:
                expiring.append(permit)

        if expiring:
            logger.warning(
                "Permits expiring soon",
                count=len(expiring),
                within_hours=within_hours,
            )

        return expiring

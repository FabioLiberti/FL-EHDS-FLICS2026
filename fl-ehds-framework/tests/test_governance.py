"""
Tests for Governance Layer (Layer 1)
====================================
"""

import pytest
from datetime import datetime, timedelta

from core.models import (
    DataPermit,
    PermitPurpose,
    PermitStatus,
    DataCategory,
    OptOutRecord,
)
from core.exceptions import (
    PermitExpiredError,
    PermitPurposeMismatchError,
    OptOutViolationError,
)
from governance.data_permits import DataPermitManager, PermitValidator
from governance.optout_registry import OptOutRegistry, OptOutChecker


class TestPermitValidator:
    """Tests for PermitValidator."""

    def test_validate_active_permit(self):
        """Test validation of active permit."""
        permit = DataPermit(
            permit_id="TEST-001",
            hdab_id="HDAB-IT",
            requester_id="REQ-001",
            purpose=PermitPurpose.SCIENTIFIC_RESEARCH,
            data_categories=[DataCategory.EHR, DataCategory.LAB_RESULTS],
            valid_until=datetime.utcnow() + timedelta(days=30),
        )

        validator = PermitValidator(strict_mode=False)
        assert validator.validate(permit) is True

    def test_validate_expired_permit(self):
        """Test validation catches expired permit."""
        # Use model_construct to bypass Pydantic's valid_until future-date validator,
        # since we intentionally need an expired permit for this test.
        permit = DataPermit.model_construct(
            permit_id="TEST-002",
            hdab_id="HDAB-IT",
            requester_id="REQ-001",
            purpose=PermitPurpose.SCIENTIFIC_RESEARCH,
            data_categories=[DataCategory.EHR],
            valid_from=datetime.utcnow() - timedelta(days=60),
            valid_until=datetime.utcnow() - timedelta(days=1),
            issued_at=datetime.utcnow() - timedelta(days=60),
            status=PermitStatus.EXPIRED,
            conditions={},
            max_rounds=None,
        )

        validator = PermitValidator(strict_mode=True, verify_expiry=True)

        with pytest.raises(PermitExpiredError):
            validator.validate(permit)

    def test_validate_purpose_mismatch(self):
        """Test validation catches purpose mismatch."""
        permit = DataPermit(
            permit_id="TEST-003",
            hdab_id="HDAB-IT",
            requester_id="REQ-001",
            purpose=PermitPurpose.SCIENTIFIC_RESEARCH,
            data_categories=[DataCategory.EHR],
            valid_until=datetime.utcnow() + timedelta(days=30),
        )

        validator = PermitValidator(strict_mode=True)

        with pytest.raises(PermitPurposeMismatchError):
            validator.validate(
                permit,
                requested_purpose=PermitPurpose.AI_SYSTEM_DEVELOPMENT,
            )


class TestDataPermitManager:
    """Tests for DataPermitManager."""

    def test_register_permit(self):
        """Test permit registration."""
        manager = DataPermitManager()

        permit = DataPermit(
            permit_id="TEST-004",
            hdab_id="HDAB-DE",
            requester_id="REQ-002",
            purpose=PermitPurpose.PUBLIC_HEALTH_SURVEILLANCE,
            data_categories=[DataCategory.REGISTRY],
            valid_until=datetime.utcnow() + timedelta(days=90),
        )

        result = manager.register_permit(permit)
        assert result is True
        assert manager.get_permit("TEST-004") == permit

    def test_verify_for_round(self):
        """Test permit verification for training round."""
        manager = DataPermitManager()

        permit = DataPermit(
            permit_id="TEST-005",
            hdab_id="HDAB-FR",
            requester_id="REQ-003",
            purpose=PermitPurpose.SCIENTIFIC_RESEARCH,
            data_categories=[DataCategory.EHR, DataCategory.LAB_RESULTS],
            valid_until=datetime.utcnow() + timedelta(days=30),
        )

        manager.register_permit(permit)

        assert manager.verify_for_round(
            "TEST-005",
            round_number=1,
            data_categories=[DataCategory.EHR],
        ) is True


class TestOptOutRegistry:
    """Tests for OptOutRegistry."""

    def test_register_optout(self):
        """Test opt-out registration."""
        registry = OptOutRegistry()

        record = OptOutRecord(
            record_id="OPT-001",
            patient_id="PAT-12345",
            scope="all",
            member_state="IT",
        )

        registry.register_optout(record)

        assert registry.is_opted_out("PAT-12345") is True
        assert registry.is_opted_out("PAT-99999") is False

    def test_optout_checker_filter(self):
        """Test opt-out checker filtering."""
        registry = OptOutRegistry()

        # Register opt-out
        record = OptOutRecord(
            record_id="OPT-002",
            patient_id="PAT-OPTED",
            scope="all",
            member_state="DE",
        )
        registry.register_optout(record)

        checker = OptOutChecker(registry, on_optout="exclude")

        records = [
            {"patient_id": "PAT-OPTED", "data": "test1"},
            {"patient_id": "PAT-ACTIVE", "data": "test2"},
            {"patient_id": "PAT-ACTIVE2", "data": "test3"},
        ]

        filtered = checker.filter_records(records)

        assert len(filtered) == 2
        assert all(r["patient_id"] != "PAT-OPTED" for r in filtered)


class TestIntegration:
    """Integration tests for governance layer."""

    def test_full_governance_flow(self):
        """Test complete governance flow."""
        # 1. Create and validate permit
        permit = DataPermit(
            permit_id="INT-001",
            hdab_id="HDAB-EU",
            requester_id="REQ-INT",
            purpose=PermitPurpose.SCIENTIFIC_RESEARCH,
            data_categories=[DataCategory.EHR, DataCategory.LAB_RESULTS],
            valid_until=datetime.utcnow() + timedelta(days=30),
        )

        validator = PermitValidator()
        assert validator.validate(permit) is True

        # 2. Register permit
        manager = DataPermitManager()
        manager.register_permit(permit)

        # 3. Setup opt-out registry
        registry = OptOutRegistry()
        registry.register_optout(
            OptOutRecord(
                record_id="OPT-INT",
                patient_id="PAT-EXCLUDED",
                scope="all",
                member_state="IT",
            )
        )

        # 4. Verify for training round
        assert manager.verify_for_round(
            "INT-001",
            round_number=1,
            data_categories=[DataCategory.EHR],
        ) is True

        # 5. Check opt-out compliance
        checker = OptOutChecker(registry)
        valid, opted_out = checker.validate_batch(
            ["PAT-001", "PAT-EXCLUDED", "PAT-002"]
        )

        assert "PAT-EXCLUDED" in opted_out
        assert "PAT-001" in valid
        assert "PAT-002" in valid

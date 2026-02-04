"""
HDAB API Compliance for FL-EHDS
================================

Health Data Access Body (HDAB) API implementation for EHDS compliance.
Implements interfaces as specified in EHDS Regulation EU 2025/327.

Key Functions:
- Data Permit Application (Article 45-46)
- Access Request Processing (Article 45)
- Secure Processing Environment Integration
- Cross-border Data Access Coordination
- Opt-out Registry Management (Article 33)
- Compliance Monitoring and Reporting

Reference:
- EHDS Regulation EU 2025/327
- EHDS Technical Infrastructure Specifications

Author: FL-EHDS Framework
License: Apache 2.0
"""

import hashlib
import json
import logging
import secrets
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger(__name__)


# =============================================================================
# HDAB Enums and Constants
# =============================================================================

class PermitStatus(Enum):
    """Data Permit Status (Article 46)."""
    DRAFT = "draft"
    SUBMITTED = "submitted"
    UNDER_REVIEW = "under_review"
    ADDITIONAL_INFO_REQUESTED = "additional_info_requested"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"
    REVOKED = "revoked"
    SUSPENDED = "suspended"


class PermitType(Enum):
    """Data Permit Types."""
    STANDARD = "standard"  # Standard research permit
    URGENT = "urgent"  # Expedited for public health emergencies
    CROSS_BORDER = "cross_border"  # Multi-country access
    FEDERATED = "federated"  # For federated learning use cases


class DataCategory(Enum):
    """EHDS Data Categories for Secondary Use."""
    EHR = "electronic_health_records"
    CLAIMS = "claims_reimbursement_data"
    REGISTRIES = "disease_registries"
    GENOMIC = "genomic_data"
    IMAGES = "medical_imaging"
    SOCIAL = "social_determinants"
    SURVEYS = "health_surveys"
    CLINICAL_TRIALS = "clinical_trial_data"
    BIOBANKS = "biobank_samples"
    PUBLIC_HEALTH = "public_health_surveillance"
    DEVICES = "medical_device_data"


class PurposeOfUse(Enum):
    """Authorized Purposes for Secondary Use (Article 34)."""
    PUBLIC_HEALTH = "public_health_protection"
    POLICY_PLANNING = "health_policy_planning"
    RESEARCH_PUBLIC = "public_interest_research"
    RESEARCH_PRIVATE = "private_sector_research"
    PRODUCT_SAFETY = "product_safety_monitoring"
    REGULATORY = "regulatory_activities"
    INNOVATION = "health_innovation"
    AI_TRAINING = "ai_system_training"
    QUALITY_ASSURANCE = "healthcare_quality"
    PERSONALIZED_MEDICINE = "personalized_medicine"
    STATISTICS = "official_statistics"
    EDUCATION = "health_education"


class ProcessingEnvironmentType(Enum):
    """Secure Processing Environment Types."""
    HDAB_HOSTED = "hdab_hosted"  # Environment hosted by HDAB
    ACCREDITED_EXTERNAL = "accredited_external"  # External accredited environment
    FEDERATED = "federated"  # Federated learning (data stays at source)
    HYBRID = "hybrid"  # Combination of approaches


class RequestorType(Enum):
    """Types of Data Requestors."""
    PUBLIC_INSTITUTION = "public_institution"
    RESEARCH_ORGANIZATION = "research_organization"
    UNIVERSITY = "university"
    HEALTH_AUTHORITY = "health_authority"
    PHARMACEUTICAL = "pharmaceutical_company"
    MEDICAL_DEVICE = "medical_device_company"
    STARTUP = "health_tech_startup"
    EU_INSTITUTION = "eu_institution"
    INTERNATIONAL_ORG = "international_organization"


class ComplianceStatus(Enum):
    """Compliance Check Status."""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PENDING_REVIEW = "pending_review"
    REMEDIATION_REQUIRED = "remediation_required"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class Requestor:
    """Data access requestor information."""
    requestor_id: str
    requestor_type: RequestorType
    organization_name: str
    legal_entity_id: Optional[str] = None  # e.g., EU VAT number
    country: str = ""
    contact_email: str = ""
    contact_name: str = ""
    accreditation_ids: List[str] = field(default_factory=list)
    ethical_approval_refs: List[str] = field(default_factory=list)
    verified: bool = False
    verification_date: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "requestorId": self.requestor_id,
            "requestorType": self.requestor_type.value,
            "organizationName": self.organization_name,
            "legalEntityId": self.legal_entity_id,
            "country": self.country,
            "contactEmail": self.contact_email,
            "contactName": self.contact_name,
            "accreditationIds": self.accreditation_ids,
            "ethicalApprovalRefs": self.ethical_approval_refs,
            "verified": self.verified,
            "verificationDate": self.verification_date.isoformat() if self.verification_date else None,
        }


@dataclass
class DatasetDescriptor:
    """Description of requested dataset."""
    dataset_id: str
    data_holder_id: str
    data_holder_name: str
    data_holder_country: str
    data_categories: List[DataCategory]
    population_description: str
    time_period_start: Optional[datetime] = None
    time_period_end: Optional[datetime] = None
    estimated_record_count: Optional[int] = None
    variables_requested: List[str] = field(default_factory=list)
    linkage_required: bool = False
    linkage_datasets: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "datasetId": self.dataset_id,
            "dataHolderId": self.data_holder_id,
            "dataHolderName": self.data_holder_name,
            "dataHolderCountry": self.data_holder_country,
            "dataCategories": [dc.value for dc in self.data_categories],
            "populationDescription": self.population_description,
            "timePeriodStart": self.time_period_start.isoformat() if self.time_period_start else None,
            "timePeriodEnd": self.time_period_end.isoformat() if self.time_period_end else None,
            "estimatedRecordCount": self.estimated_record_count,
            "variablesRequested": self.variables_requested,
            "linkageRequired": self.linkage_required,
            "linkageDatasets": self.linkage_datasets,
        }


@dataclass
class DataPermitApplication:
    """Data Permit Application (Article 45)."""
    application_id: str
    requestor: Requestor
    status: PermitStatus
    permit_type: PermitType

    # Purpose and justification
    purpose_of_use: PurposeOfUse
    research_question: str
    scientific_justification: str
    expected_benefits: str
    methodology_summary: str

    # Requested data
    datasets: List[DatasetDescriptor]

    # Processing details
    processing_environment: ProcessingEnvironmentType
    processing_location_country: str
    data_retention_period_months: int
    output_description: str
    anonymization_approach: str

    # Timelines
    application_date: datetime
    requested_access_start: datetime
    requested_access_end: datetime

    # Review tracking
    assigned_reviewer: Optional[str] = None
    review_start_date: Optional[datetime] = None
    decision_date: Optional[datetime] = None
    decision_reason: Optional[str] = None

    # Federated Learning specific
    fl_algorithm: Optional[str] = None
    fl_rounds_planned: Optional[int] = None
    fl_privacy_budget: Optional[float] = None  # Epsilon for DP
    fl_client_selection: Optional[str] = None

    # Compliance
    ethical_approval: bool = False
    ethical_approval_ref: Optional[str] = None
    data_protection_assessment: bool = False
    dpia_ref: Optional[str] = None

    # Fees
    fee_amount: Optional[float] = None
    fee_paid: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "applicationId": self.application_id,
            "requestor": self.requestor.to_dict(),
            "status": self.status.value,
            "permitType": self.permit_type.value,
            "purposeOfUse": self.purpose_of_use.value,
            "researchQuestion": self.research_question,
            "scientificJustification": self.scientific_justification,
            "expectedBenefits": self.expected_benefits,
            "methodologySummary": self.methodology_summary,
            "datasets": [ds.to_dict() for ds in self.datasets],
            "processingEnvironment": self.processing_environment.value,
            "processingLocationCountry": self.processing_location_country,
            "dataRetentionPeriodMonths": self.data_retention_period_months,
            "outputDescription": self.output_description,
            "anonymizationApproach": self.anonymization_approach,
            "applicationDate": self.application_date.isoformat(),
            "requestedAccessStart": self.requested_access_start.isoformat(),
            "requestedAccessEnd": self.requested_access_end.isoformat(),
            "assignedReviewer": self.assigned_reviewer,
            "reviewStartDate": self.review_start_date.isoformat() if self.review_start_date else None,
            "decisionDate": self.decision_date.isoformat() if self.decision_date else None,
            "decisionReason": self.decision_reason,
            "flAlgorithm": self.fl_algorithm,
            "flRoundsPlanned": self.fl_rounds_planned,
            "flPrivacyBudget": self.fl_privacy_budget,
            "flClientSelection": self.fl_client_selection,
            "ethicalApproval": self.ethical_approval,
            "ethicalApprovalRef": self.ethical_approval_ref,
            "dataProtectionAssessment": self.data_protection_assessment,
            "dpiaRef": self.dpia_ref,
            "feeAmount": self.fee_amount,
            "feePaid": self.fee_paid,
        }


@dataclass
class DataPermit:
    """Approved Data Permit (Article 46)."""
    permit_id: str
    application_id: str
    requestor_id: str
    status: PermitStatus

    # Authorized access
    authorized_purposes: List[PurposeOfUse]
    authorized_datasets: List[str]  # Dataset IDs
    authorized_variables: Dict[str, List[str]]  # dataset_id -> variables

    # Validity
    issue_date: datetime
    valid_from: datetime
    valid_until: datetime

    # Conditions
    processing_environment_id: str
    max_query_count: Optional[int] = None
    privacy_budget_total: Optional[float] = None
    privacy_budget_used: float = 0.0
    output_review_required: bool = True
    linkage_authorized: bool = False

    # Cross-border
    cross_border_countries: List[str] = field(default_factory=list)
    lead_hdab_country: Optional[str] = None

    # FL-specific conditions
    fl_max_rounds: Optional[int] = None
    fl_min_clients_per_round: Optional[int] = None
    fl_max_clients_per_round: Optional[int] = None
    fl_aggregation_threshold: Optional[int] = None

    # Tracking
    access_count: int = 0
    last_access: Optional[datetime] = None
    queries_executed: int = 0
    fl_rounds_completed: int = 0

    def is_valid(self) -> bool:
        """Check if permit is currently valid."""
        if self.status != PermitStatus.APPROVED:
            return False
        now = datetime.now()
        return self.valid_from <= now <= self.valid_until

    def check_privacy_budget(self, epsilon_cost: float) -> bool:
        """Check if privacy budget allows operation."""
        if self.privacy_budget_total is None:
            return True
        return (self.privacy_budget_used + epsilon_cost) <= self.privacy_budget_total

    def consume_privacy_budget(self, epsilon_cost: float) -> bool:
        """Consume privacy budget. Returns False if exceeded."""
        if not self.check_privacy_budget(epsilon_cost):
            return False
        self.privacy_budget_used += epsilon_cost
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "permitId": self.permit_id,
            "applicationId": self.application_id,
            "requestorId": self.requestor_id,
            "status": self.status.value,
            "authorizedPurposes": [p.value for p in self.authorized_purposes],
            "authorizedDatasets": self.authorized_datasets,
            "authorizedVariables": self.authorized_variables,
            "issueDate": self.issue_date.isoformat(),
            "validFrom": self.valid_from.isoformat(),
            "validUntil": self.valid_until.isoformat(),
            "processingEnvironmentId": self.processing_environment_id,
            "maxQueryCount": self.max_query_count,
            "privacyBudgetTotal": self.privacy_budget_total,
            "privacyBudgetUsed": self.privacy_budget_used,
            "outputReviewRequired": self.output_review_required,
            "linkageAuthorized": self.linkage_authorized,
            "crossBorderCountries": self.cross_border_countries,
            "leadHdabCountry": self.lead_hdab_country,
            "flMaxRounds": self.fl_max_rounds,
            "flMinClientsPerRound": self.fl_min_clients_per_round,
            "flMaxClientsPerRound": self.fl_max_clients_per_round,
            "flAggregationThreshold": self.fl_aggregation_threshold,
            "accessCount": self.access_count,
            "lastAccess": self.last_access.isoformat() if self.last_access else None,
            "queriesExecuted": self.queries_executed,
            "flRoundsCompleted": self.fl_rounds_completed,
        }


@dataclass
class OptOutRecord:
    """Patient Opt-out Record (Article 33)."""
    opt_out_id: str
    patient_pseudonym: str  # Pseudonymized identifier
    hdab_country: str
    opt_out_date: datetime
    opt_out_scope: str  # "all" or specific categories
    excluded_categories: List[DataCategory] = field(default_factory=list)
    reason: Optional[str] = None
    effective_from: Optional[datetime] = None
    revocation_date: Optional[datetime] = None
    is_active: bool = True


@dataclass
class ComplianceReport:
    """Compliance monitoring report."""
    report_id: str
    permit_id: str
    report_date: datetime
    compliance_status: ComplianceStatus
    checks_performed: List[str]
    issues_found: List[str]
    remediation_actions: List[str]
    next_review_date: Optional[datetime] = None


@dataclass
class AccessLog:
    """Data access audit log."""
    log_id: str
    permit_id: str
    timestamp: datetime
    action_type: str  # query, fl_round, download, etc.
    user_id: str
    datasets_accessed: List[str]
    record_count_affected: Optional[int] = None
    privacy_cost: Optional[float] = None
    query_hash: Optional[str] = None
    result_hash: Optional[str] = None
    success: bool = True
    error_message: Optional[str] = None


# =============================================================================
# HDAB API Interfaces
# =============================================================================

class HDABAPIClient(ABC):
    """Abstract base class for HDAB API client."""

    @abstractmethod
    def submit_application(self, application: DataPermitApplication) -> str:
        """Submit a new data permit application."""
        pass

    @abstractmethod
    def get_application_status(self, application_id: str) -> DataPermitApplication:
        """Get application status."""
        pass

    @abstractmethod
    def get_permit(self, permit_id: str) -> Optional[DataPermit]:
        """Get an approved permit."""
        pass

    @abstractmethod
    def validate_permit(self, permit_id: str, operation: str) -> Tuple[bool, str]:
        """Validate permit for a specific operation."""
        pass

    @abstractmethod
    def check_opt_out(self, patient_pseudonyms: List[str]) -> List[str]:
        """Check which patients have opted out."""
        pass

    @abstractmethod
    def log_access(self, access_log: AccessLog) -> str:
        """Log data access for audit."""
        pass


class HDABServiceSimulator(HDABAPIClient):
    """
    HDAB Service Simulator for testing and development.

    Simulates HDAB API responses for FL-EHDS framework testing.
    """

    def __init__(
        self,
        hdab_id: str,
        country: str,
        auto_approve: bool = False,
        review_delay_days: int = 30,
    ):
        self.hdab_id = hdab_id
        self.country = country
        self.auto_approve = auto_approve
        self.review_delay_days = review_delay_days

        self._applications: Dict[str, DataPermitApplication] = {}
        self._permits: Dict[str, DataPermit] = {}
        self._opt_outs: Dict[str, OptOutRecord] = {}
        self._access_logs: List[AccessLog] = []
        self._compliance_reports: Dict[str, ComplianceReport] = {}

        logger.info(f"HDAB Service Simulator initialized: {hdab_id} ({country})")

    def submit_application(self, application: DataPermitApplication) -> str:
        """Submit a new data permit application."""
        application.status = PermitStatus.SUBMITTED
        self._applications[application.application_id] = application

        logger.info(f"Application submitted: {application.application_id}")

        # Auto-approve if configured (for testing)
        if self.auto_approve:
            self._auto_approve_application(application)

        return application.application_id

    def _auto_approve_application(self, application: DataPermitApplication) -> None:
        """Automatically approve application (for testing)."""
        application.status = PermitStatus.APPROVED
        application.decision_date = datetime.now()
        application.decision_reason = "Auto-approved for testing"

        # Create permit
        permit = DataPermit(
            permit_id=f"PERMIT-{uuid.uuid4().hex[:8].upper()}",
            application_id=application.application_id,
            requestor_id=application.requestor.requestor_id,
            status=PermitStatus.APPROVED,
            authorized_purposes=[application.purpose_of_use],
            authorized_datasets=[ds.dataset_id for ds in application.datasets],
            authorized_variables={
                ds.dataset_id: ds.variables_requested
                for ds in application.datasets
            },
            issue_date=datetime.now(),
            valid_from=application.requested_access_start,
            valid_until=application.requested_access_end,
            processing_environment_id=f"SPE-{self.hdab_id}",
            privacy_budget_total=application.fl_privacy_budget,
            fl_max_rounds=application.fl_rounds_planned,
            fl_min_clients_per_round=3,
            fl_max_clients_per_round=100,
            fl_aggregation_threshold=5,
        )

        self._permits[permit.permit_id] = permit
        logger.info(f"Permit auto-approved: {permit.permit_id}")

    def get_application_status(self, application_id: str) -> DataPermitApplication:
        """Get application status."""
        if application_id not in self._applications:
            raise ValueError(f"Application not found: {application_id}")
        return self._applications[application_id]

    def get_permit(self, permit_id: str) -> Optional[DataPermit]:
        """Get an approved permit."""
        return self._permits.get(permit_id)

    def validate_permit(self, permit_id: str, operation: str) -> Tuple[bool, str]:
        """
        Validate permit for a specific operation.

        Args:
            permit_id: The permit ID to validate
            operation: Type of operation (query, fl_round, download, etc.)

        Returns:
            Tuple of (is_valid, reason)
        """
        permit = self._permits.get(permit_id)
        if not permit:
            return False, "Permit not found"

        if not permit.is_valid():
            if permit.status != PermitStatus.APPROVED:
                return False, f"Permit status is {permit.status.value}"
            return False, "Permit has expired"

        # Check operation-specific constraints
        if operation == "fl_round":
            if permit.fl_max_rounds and permit.fl_rounds_completed >= permit.fl_max_rounds:
                return False, "Maximum FL rounds exceeded"

        if operation == "query":
            if permit.max_query_count and permit.queries_executed >= permit.max_query_count:
                return False, "Maximum query count exceeded"

        return True, "Permit is valid"

    def check_opt_out(self, patient_pseudonyms: List[str]) -> List[str]:
        """
        Check which patients have opted out.

        Returns list of patient pseudonyms that have opted out.
        """
        opted_out = []
        for pseudonym in patient_pseudonyms:
            if pseudonym in self._opt_outs:
                record = self._opt_outs[pseudonym]
                if record.is_active:
                    opted_out.append(pseudonym)
        return opted_out

    def register_opt_out(
        self,
        patient_pseudonym: str,
        scope: str = "all",
        reason: Optional[str] = None,
    ) -> OptOutRecord:
        """Register a patient opt-out."""
        record = OptOutRecord(
            opt_out_id=str(uuid.uuid4()),
            patient_pseudonym=patient_pseudonym,
            hdab_country=self.country,
            opt_out_date=datetime.now(),
            opt_out_scope=scope,
            reason=reason,
            effective_from=datetime.now(),
            is_active=True,
        )
        self._opt_outs[patient_pseudonym] = record
        logger.info(f"Opt-out registered for {patient_pseudonym}")
        return record

    def revoke_opt_out(self, patient_pseudonym: str) -> bool:
        """Revoke a patient opt-out."""
        if patient_pseudonym not in self._opt_outs:
            return False

        record = self._opt_outs[patient_pseudonym]
        record.is_active = False
        record.revocation_date = datetime.now()
        return True

    def log_access(self, access_log: AccessLog) -> str:
        """Log data access for audit."""
        self._access_logs.append(access_log)

        # Update permit statistics
        if access_log.permit_id in self._permits:
            permit = self._permits[access_log.permit_id]
            permit.access_count += 1
            permit.last_access = access_log.timestamp

            if access_log.action_type == "query":
                permit.queries_executed += 1
            elif access_log.action_type == "fl_round":
                permit.fl_rounds_completed += 1

            if access_log.privacy_cost:
                permit.consume_privacy_budget(access_log.privacy_cost)

        logger.debug(f"Access logged: {access_log.log_id}")
        return access_log.log_id

    def consume_privacy_budget(
        self,
        permit_id: str,
        epsilon_cost: float,
    ) -> Tuple[bool, float]:
        """
        Consume privacy budget from permit.

        Returns:
            Tuple of (success, remaining_budget)
        """
        permit = self._permits.get(permit_id)
        if not permit:
            return False, 0.0

        if not permit.check_privacy_budget(epsilon_cost):
            remaining = (permit.privacy_budget_total or 0.0) - permit.privacy_budget_used
            return False, remaining

        permit.consume_privacy_budget(epsilon_cost)
        remaining = (permit.privacy_budget_total or 0.0) - permit.privacy_budget_used
        return True, remaining

    def get_permit_statistics(self, permit_id: str) -> Dict[str, Any]:
        """Get permit usage statistics."""
        permit = self._permits.get(permit_id)
        if not permit:
            return {}

        return {
            "permitId": permit_id,
            "accessCount": permit.access_count,
            "queriesExecuted": permit.queries_executed,
            "flRoundsCompleted": permit.fl_rounds_completed,
            "privacyBudgetUsed": permit.privacy_budget_used,
            "privacyBudgetRemaining": (
                (permit.privacy_budget_total or 0) - permit.privacy_budget_used
                if permit.privacy_budget_total else None
            ),
            "daysRemaining": (permit.valid_until - datetime.now()).days,
            "lastAccess": permit.last_access.isoformat() if permit.last_access else None,
        }

    def get_access_logs(
        self,
        permit_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[AccessLog]:
        """Get access logs with filters."""
        logs = self._access_logs.copy()

        if permit_id:
            logs = [l for l in logs if l.permit_id == permit_id]
        if start_time:
            logs = [l for l in logs if l.timestamp >= start_time]
        if end_time:
            logs = [l for l in logs if l.timestamp <= end_time]

        return logs

    def create_compliance_report(
        self,
        permit_id: str,
        checks: List[str],
    ) -> ComplianceReport:
        """Create compliance report for a permit."""
        permit = self._permits.get(permit_id)
        if not permit:
            raise ValueError(f"Permit not found: {permit_id}")

        issues = []
        # Check various compliance aspects
        if permit.privacy_budget_total:
            usage_pct = permit.privacy_budget_used / permit.privacy_budget_total * 100
            if usage_pct > 80:
                issues.append(f"Privacy budget {usage_pct:.1f}% consumed")

        if permit.fl_max_rounds and permit.fl_rounds_completed > permit.fl_max_rounds * 0.9:
            issues.append("Approaching maximum FL rounds limit")

        status = ComplianceStatus.COMPLIANT if not issues else ComplianceStatus.REMEDIATION_REQUIRED

        report = ComplianceReport(
            report_id=str(uuid.uuid4()),
            permit_id=permit_id,
            report_date=datetime.now(),
            compliance_status=status,
            checks_performed=checks,
            issues_found=issues,
            remediation_actions=[],
            next_review_date=datetime.now() + timedelta(days=30),
        )

        self._compliance_reports[report.report_id] = report
        return report


# =============================================================================
# FL-EHDS HDAB Integration
# =============================================================================

class FLEHDSPermitManager:
    """
    FL-EHDS Permit Manager.

    Manages data permits for federated learning within EHDS framework.
    Coordinates with HDAB APIs and enforces compliance.
    """

    def __init__(
        self,
        hdab_client: HDABAPIClient,
        organization_id: str,
        organization_name: str,
        country: str,
    ):
        self.hdab_client = hdab_client
        self.organization_id = organization_id
        self.organization_name = organization_name
        self.country = country

        self._active_permits: Dict[str, DataPermit] = {}
        self._permit_validation_cache: Dict[str, Tuple[bool, str, datetime]] = {}
        self._cache_ttl = timedelta(minutes=5)

    def request_fl_permit(
        self,
        research_question: str,
        justification: str,
        methodology: str,
        datasets: List[DatasetDescriptor],
        fl_algorithm: str,
        fl_rounds: int,
        privacy_budget: float,
        access_duration_months: int = 12,
        ethical_approval_ref: Optional[str] = None,
    ) -> str:
        """
        Request a new FL data permit.

        Returns application ID for tracking.
        """
        requestor = Requestor(
            requestor_id=self.organization_id,
            requestor_type=RequestorType.RESEARCH_ORGANIZATION,
            organization_name=self.organization_name,
            country=self.country,
            verified=True,
            verification_date=datetime.now(),
        )

        application = DataPermitApplication(
            application_id=f"APP-{uuid.uuid4().hex[:8].upper()}",
            requestor=requestor,
            status=PermitStatus.DRAFT,
            permit_type=PermitType.FEDERATED,
            purpose_of_use=PurposeOfUse.AI_TRAINING,
            research_question=research_question,
            scientific_justification=justification,
            expected_benefits="Improved healthcare AI models trained on distributed data",
            methodology_summary=methodology,
            datasets=datasets,
            processing_environment=ProcessingEnvironmentType.FEDERATED,
            processing_location_country=self.country,
            data_retention_period_months=0,  # No data leaves source in FL
            output_description="Trained ML model weights (aggregated, no individual data)",
            anonymization_approach="Federated Learning with Differential Privacy",
            application_date=datetime.now(),
            requested_access_start=datetime.now(),
            requested_access_end=datetime.now() + timedelta(days=30 * access_duration_months),
            fl_algorithm=fl_algorithm,
            fl_rounds_planned=fl_rounds,
            fl_privacy_budget=privacy_budget,
            fl_client_selection="Random selection with minimum data requirements",
            ethical_approval=bool(ethical_approval_ref),
            ethical_approval_ref=ethical_approval_ref,
            data_protection_assessment=True,
        )

        application_id = self.hdab_client.submit_application(application)
        logger.info(f"FL permit application submitted: {application_id}")
        return application_id

    def get_application_status(self, application_id: str) -> Dict[str, Any]:
        """Get permit application status."""
        application = self.hdab_client.get_application_status(application_id)
        return {
            "applicationId": application.application_id,
            "status": application.status.value,
            "submissionDate": application.application_date.isoformat(),
            "decisionDate": application.decision_date.isoformat() if application.decision_date else None,
            "reviewerAssigned": application.assigned_reviewer is not None,
        }

    def activate_permit(self, permit_id: str) -> DataPermit:
        """Activate a permit for use."""
        permit = self.hdab_client.get_permit(permit_id)
        if not permit:
            raise ValueError(f"Permit not found: {permit_id}")

        if not permit.is_valid():
            raise ValueError(f"Permit is not valid: {permit.status.value}")

        self._active_permits[permit_id] = permit
        logger.info(f"Permit activated: {permit_id}")
        return permit

    def validate_fl_round(
        self,
        permit_id: str,
        round_number: int,
        client_count: int,
        epsilon_cost: float,
    ) -> Tuple[bool, str]:
        """
        Validate that an FL round can proceed.

        Checks permit validity, FL constraints, and privacy budget.
        """
        # Check cache first
        cache_key = f"{permit_id}:{round_number}"
        if cache_key in self._permit_validation_cache:
            is_valid, reason, cached_time = self._permit_validation_cache[cache_key]
            if datetime.now() - cached_time < self._cache_ttl:
                return is_valid, reason

        # Validate with HDAB
        is_valid, reason = self.hdab_client.validate_permit(permit_id, "fl_round")
        if not is_valid:
            return False, reason

        # Get permit for additional checks
        permit = self._active_permits.get(permit_id) or self.hdab_client.get_permit(permit_id)
        if not permit:
            return False, "Permit not found"

        # Check FL-specific constraints
        if permit.fl_min_clients_per_round and client_count < permit.fl_min_clients_per_round:
            return False, f"Minimum {permit.fl_min_clients_per_round} clients required"

        if permit.fl_max_clients_per_round and client_count > permit.fl_max_clients_per_round:
            return False, f"Maximum {permit.fl_max_clients_per_round} clients allowed"

        if permit.fl_max_rounds and round_number > permit.fl_max_rounds:
            return False, f"Maximum {permit.fl_max_rounds} rounds allowed"

        # Check privacy budget
        if not permit.check_privacy_budget(epsilon_cost):
            remaining = (permit.privacy_budget_total or 0) - permit.privacy_budget_used
            return False, f"Insufficient privacy budget. Remaining: {remaining:.4f}"

        # Cache result
        self._permit_validation_cache[cache_key] = (True, "Valid", datetime.now())
        return True, "FL round authorized"

    def filter_opted_out_clients(
        self,
        patient_pseudonyms: List[str],
    ) -> Tuple[List[str], List[str]]:
        """
        Filter out patients who have opted out.

        Returns:
            Tuple of (permitted_patients, opted_out_patients)
        """
        opted_out = self.hdab_client.check_opt_out(patient_pseudonyms)
        opted_out_set = set(opted_out)
        permitted = [p for p in patient_pseudonyms if p not in opted_out_set]
        return permitted, opted_out

    def log_fl_round(
        self,
        permit_id: str,
        round_number: int,
        client_ids: List[str],
        epsilon_cost: float,
        success: bool = True,
        error_message: Optional[str] = None,
    ) -> str:
        """Log FL round execution for audit."""
        access_log = AccessLog(
            log_id=str(uuid.uuid4()),
            permit_id=permit_id,
            timestamp=datetime.now(),
            action_type="fl_round",
            user_id=self.organization_id,
            datasets_accessed=client_ids,
            record_count_affected=len(client_ids),
            privacy_cost=epsilon_cost,
            query_hash=hashlib.sha256(f"round_{round_number}".encode()).hexdigest(),
            success=success,
            error_message=error_message,
        )
        return self.hdab_client.log_access(access_log)

    def get_permit_status(self, permit_id: str) -> Dict[str, Any]:
        """Get current permit status and usage."""
        permit = self._active_permits.get(permit_id) or self.hdab_client.get_permit(permit_id)
        if not permit:
            return {"error": "Permit not found"}

        return {
            "permitId": permit_id,
            "status": permit.status.value,
            "isValid": permit.is_valid(),
            "validUntil": permit.valid_until.isoformat(),
            "flRoundsCompleted": permit.fl_rounds_completed,
            "flRoundsRemaining": (
                permit.fl_max_rounds - permit.fl_rounds_completed
                if permit.fl_max_rounds else None
            ),
            "privacyBudgetUsed": permit.privacy_budget_used,
            "privacyBudgetRemaining": (
                permit.privacy_budget_total - permit.privacy_budget_used
                if permit.privacy_budget_total else None
            ),
        }

    def close_permit(self, permit_id: str) -> bool:
        """Close a permit when FL training is complete."""
        if permit_id in self._active_permits:
            del self._active_permits[permit_id]
            logger.info(f"Permit closed: {permit_id}")
            return True
        return False


class CrossBorderHDABCoordinator:
    """
    Cross-Border HDAB Coordinator.

    Coordinates data permits across multiple EU member states
    for cross-border federated learning.
    """

    def __init__(
        self,
        lead_hdab: HDABServiceSimulator,
        lead_country: str,
    ):
        self.lead_hdab = lead_hdab
        self.lead_country = lead_country
        self._participating_hdabs: Dict[str, HDABServiceSimulator] = {
            lead_country: lead_hdab
        }
        self._cross_border_permits: Dict[str, List[str]] = {}  # master_permit -> [country_permits]

    def add_participating_hdab(
        self,
        country: str,
        hdab: HDABServiceSimulator,
    ) -> None:
        """Add a participating country's HDAB."""
        self._participating_hdabs[country] = hdab
        logger.info(f"Added participating HDAB: {country}")

    def request_cross_border_permit(
        self,
        application: DataPermitApplication,
        target_countries: List[str],
    ) -> Dict[str, str]:
        """
        Request coordinated cross-border permit.

        Returns mapping of country -> permit_id.
        """
        application.permit_type = PermitType.CROSS_BORDER

        # Submit to lead HDAB
        master_app_id = self.lead_hdab.submit_application(application)

        # Track permits by country
        permit_mapping: Dict[str, str] = {}

        # Wait for lead approval (simulated)
        master_permit = None
        for pid, permit in self.lead_hdab._permits.items():
            if permit.application_id == master_app_id:
                master_permit = permit
                permit_mapping[self.lead_country] = pid
                permit.cross_border_countries = target_countries
                permit.lead_hdab_country = self.lead_country
                break

        if not master_permit:
            logger.warning(f"Master permit not approved for {master_app_id}")
            return {}

        # Coordinate with other HDABs
        for country in target_countries:
            if country == self.lead_country:
                continue

            if country not in self._participating_hdabs:
                logger.warning(f"No HDAB available for {country}")
                continue

            hdab = self._participating_hdabs[country]

            # Create country-specific application
            country_app = DataPermitApplication(
                application_id=f"APP-{country[:2]}-{uuid.uuid4().hex[:6].upper()}",
                requestor=application.requestor,
                status=PermitStatus.DRAFT,
                permit_type=PermitType.CROSS_BORDER,
                purpose_of_use=application.purpose_of_use,
                research_question=application.research_question,
                scientific_justification=application.scientific_justification,
                expected_benefits=application.expected_benefits,
                methodology_summary=application.methodology_summary,
                datasets=[ds for ds in application.datasets if ds.data_holder_country == country],
                processing_environment=application.processing_environment,
                processing_location_country=country,
                data_retention_period_months=application.data_retention_period_months,
                output_description=application.output_description,
                anonymization_approach=application.anonymization_approach,
                application_date=datetime.now(),
                requested_access_start=application.requested_access_start,
                requested_access_end=application.requested_access_end,
                fl_algorithm=application.fl_algorithm,
                fl_rounds_planned=application.fl_rounds_planned,
                fl_privacy_budget=application.fl_privacy_budget,
            )

            hdab.submit_application(country_app)

            # Get country permit
            for pid, permit in hdab._permits.items():
                if permit.application_id == country_app.application_id:
                    permit_mapping[country] = pid
                    permit.cross_border_countries = target_countries
                    permit.lead_hdab_country = self.lead_country
                    break

        self._cross_border_permits[permit_mapping.get(self.lead_country, "")] = list(permit_mapping.values())
        return permit_mapping

    def validate_cross_border_fl(
        self,
        permit_mapping: Dict[str, str],
        round_number: int,
        clients_per_country: Dict[str, int],
        epsilon_cost: float,
    ) -> Dict[str, Tuple[bool, str]]:
        """
        Validate FL round across all participating countries.

        Returns mapping of country -> (is_valid, reason).
        """
        results = {}

        for country, permit_id in permit_mapping.items():
            if country not in self._participating_hdabs:
                results[country] = (False, "HDAB not available")
                continue

            hdab = self._participating_hdabs[country]
            is_valid, reason = hdab.validate_permit(permit_id, "fl_round")

            if is_valid:
                # Check privacy budget
                permit = hdab.get_permit(permit_id)
                if permit and not permit.check_privacy_budget(epsilon_cost):
                    is_valid = False
                    reason = "Insufficient privacy budget"

            results[country] = (is_valid, reason)

        return results

    def get_cross_border_status(
        self,
        permit_mapping: Dict[str, str],
    ) -> Dict[str, Dict[str, Any]]:
        """Get status of all permits in cross-border arrangement."""
        status = {}

        for country, permit_id in permit_mapping.items():
            if country in self._participating_hdabs:
                hdab = self._participating_hdabs[country]
                permit = hdab.get_permit(permit_id)
                if permit:
                    status[country] = permit.to_dict()

        return status


# =============================================================================
# Factory Functions
# =============================================================================

def create_hdab_simulator(
    hdab_id: str,
    country: str,
    auto_approve: bool = False,
) -> HDABServiceSimulator:
    """Create HDAB service simulator for testing."""
    return HDABServiceSimulator(
        hdab_id=hdab_id,
        country=country,
        auto_approve=auto_approve,
    )


def create_permit_manager(
    hdab_client: HDABAPIClient,
    organization_id: str,
    organization_name: str,
    country: str,
) -> FLEHDSPermitManager:
    """Create FL-EHDS permit manager."""
    return FLEHDSPermitManager(
        hdab_client=hdab_client,
        organization_id=organization_id,
        organization_name=organization_name,
        country=country,
    )


def create_cross_border_coordinator(
    lead_country: str,
    auto_approve: bool = True,
) -> CrossBorderHDABCoordinator:
    """Create cross-border HDAB coordinator."""
    lead_hdab = HDABServiceSimulator(
        hdab_id=f"HDAB-{lead_country}",
        country=lead_country,
        auto_approve=auto_approve,
    )
    return CrossBorderHDABCoordinator(
        lead_hdab=lead_hdab,
        lead_country=lead_country,
    )


# =============================================================================
# Export
# =============================================================================

__all__ = [
    # Enums
    "PermitStatus",
    "PermitType",
    "DataCategory",
    "PurposeOfUse",
    "ProcessingEnvironmentType",
    "RequestorType",
    "ComplianceStatus",
    # Data Classes
    "Requestor",
    "DatasetDescriptor",
    "DataPermitApplication",
    "DataPermit",
    "OptOutRecord",
    "ComplianceReport",
    "AccessLog",
    # API Interfaces
    "HDABAPIClient",
    "HDABServiceSimulator",
    # FL-EHDS Integration
    "FLEHDSPermitManager",
    "CrossBorderHDABCoordinator",
    # Factory Functions
    "create_hdab_simulator",
    "create_permit_manager",
    "create_cross_border_coordinator",
]

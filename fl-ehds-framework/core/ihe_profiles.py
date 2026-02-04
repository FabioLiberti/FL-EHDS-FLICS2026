"""
IHE Profiles Integration for FL-EHDS
=====================================

Integrating the Healthcare Enterprise (IHE) profiles for secure,
standards-based health data exchange in the FL-EHDS framework.

Supported IHE Profiles:
- XDS.b (Cross-Enterprise Document Sharing)
- XCA (Cross-Community Access)
- MHD (Mobile Health Documents) - FHIR-based
- PIXm/PDQm (Patient Identifier Cross-referencing / Patient Demographics Query)
- ATNA (Audit Trail and Node Authentication)
- BPPC (Basic Patient Privacy Consents)
- XUA (Cross-Enterprise User Assertion)
- CT (Consistent Time)

EHDS Compliance:
- Cross-border data exchange (Art. 12 EHDS Regulation)
- Audit trail requirements (Art. 50 EHDS Regulation)
- Consent management for secondary use (Art. 33 EHDS Regulation)

Author: FL-EHDS Framework
License: Apache 2.0
"""

import hashlib
import hmac
import json
import logging
import secrets
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import base64

logger = logging.getLogger(__name__)


# =============================================================================
# IHE Profile Enums and Constants
# =============================================================================

class IHEProfile(Enum):
    """Supported IHE Integration Profiles."""
    XDS_B = "XDS.b"  # Cross-Enterprise Document Sharing
    XCA = "XCA"  # Cross-Community Access
    MHD = "MHD"  # Mobile Health Documents
    PIX = "PIXm"  # Patient Identifier Cross-referencing for Mobile
    PDQ = "PDQm"  # Patient Demographics Query for Mobile
    ATNA = "ATNA"  # Audit Trail and Node Authentication
    BPPC = "BPPC"  # Basic Patient Privacy Consents
    XUA = "XUA"  # Cross-Enterprise User Assertion
    CT = "CT"  # Consistent Time


class DocumentStatus(Enum):
    """XDS Document Status."""
    APPROVED = "urn:oasis:names:tc:ebxml-regrep:StatusType:Approved"
    DEPRECATED = "urn:oasis:names:tc:ebxml-regrep:StatusType:Deprecated"
    SUBMITTED = "urn:oasis:names:tc:ebxml-regrep:StatusType:Submitted"


class AuditEventType(Enum):
    """ATNA Audit Event Types (DICOM/IHE)."""
    # Application Activity
    APPLICATION_START = "110100"
    APPLICATION_STOP = "110101"

    # Audit Log Used
    AUDIT_LOG_USED = "110102"

    # Begin/End of Transfer
    BEGIN_TRANSFERRING = "110103"
    END_TRANSFERRING = "110104"

    # Patient Record Events
    PATIENT_RECORD = "110110"

    # Query
    QUERY = "110112"

    # Security Alert
    SECURITY_ALERT = "110113"

    # User Authentication
    USER_AUTHENTICATION = "110114"

    # Data Import/Export
    IMPORT = "110107"
    EXPORT = "110106"

    # FL-EHDS Specific
    FL_TRAINING_START = "FL001"
    FL_TRAINING_END = "FL002"
    FL_MODEL_UPDATE = "FL003"
    FL_AGGREGATION = "FL004"
    FL_DATA_ACCESS = "FL005"
    FL_PERMIT_CHECK = "FL006"


class AuditEventOutcome(Enum):
    """ATNA Audit Event Outcome."""
    SUCCESS = 0
    MINOR_FAILURE = 4
    SERIOUS_FAILURE = 8
    MAJOR_FAILURE = 12


class ConsentStatus(Enum):
    """BPPC Consent Status."""
    ACTIVE = "active"
    REJECTED = "rejected"
    REVOKED = "revoked"
    EXPIRED = "expired"
    PENDING = "pending"


class ConsentScope(Enum):
    """EHDS Consent Scopes for Secondary Use."""
    RESEARCH_ALL = "research-all"  # All research purposes
    RESEARCH_PUBLIC_HEALTH = "research-public-health"
    RESEARCH_RARE_DISEASE = "research-rare-disease"
    RESEARCH_CLINICAL_TRIAL = "research-clinical-trial"
    AI_TRAINING = "ai-training"
    QUALITY_IMPROVEMENT = "quality-improvement"
    POLICY_PLANNING = "policy-planning"
    OPT_OUT = "opt-out"  # Complete opt-out (Art. 33 EHDS)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class XDSDocumentEntry:
    """XDS Document Entry metadata."""
    document_unique_id: str
    patient_id: str
    class_code: str  # e.g., "34133-9" (Summary of episode note)
    type_code: str  # Document type
    format_code: str  # e.g., "urn:ihe:pcc:xds-ms:2007"
    creation_time: datetime
    service_start_time: Optional[datetime] = None
    service_stop_time: Optional[datetime] = None
    healthcare_facility_type: Optional[str] = None
    practice_setting_code: Optional[str] = None
    confidentiality_code: str = "N"  # N=Normal, R=Restricted, V=Very Restricted
    language_code: str = "en-US"
    status: DocumentStatus = DocumentStatus.APPROVED
    author_institution: Optional[str] = None
    author_person: Optional[str] = None
    title: Optional[str] = None
    comments: Optional[str] = None
    size: Optional[int] = None
    hash: Optional[str] = None
    repository_unique_id: Optional[str] = None
    home_community_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "documentUniqueId": self.document_unique_id,
            "patientId": self.patient_id,
            "classCode": self.class_code,
            "typeCode": self.type_code,
            "formatCode": self.format_code,
            "creationTime": self.creation_time.isoformat(),
            "serviceStartTime": self.service_start_time.isoformat() if self.service_start_time else None,
            "serviceStopTime": self.service_stop_time.isoformat() if self.service_stop_time else None,
            "healthcareFacilityTypeCode": self.healthcare_facility_type,
            "practiceSettingCode": self.practice_setting_code,
            "confidentialityCode": self.confidentiality_code,
            "languageCode": self.language_code,
            "status": self.status.value,
            "authorInstitution": self.author_institution,
            "authorPerson": self.author_person,
            "title": self.title,
            "comments": self.comments,
            "size": self.size,
            "hash": self.hash,
            "repositoryUniqueId": self.repository_unique_id,
            "homeCommunityId": self.home_community_id,
        }


@dataclass
class PatientIdentifier:
    """PIX Patient Identifier."""
    identifier: str
    assigning_authority: str  # OID of the assigning authority
    identifier_type: str = "MR"  # MR=Medical Record, PI=Patient Internal

    @property
    def hl7_cx(self) -> str:
        """Return HL7 CX format: ID^^^&AssigningAuthority&ISO"""
        return f"{self.identifier}^^^&{self.assigning_authority}&ISO"


@dataclass
class PatientDemographics:
    """PDQ Patient Demographics."""
    identifiers: List[PatientIdentifier]
    family_name: str
    given_name: str
    birth_date: Optional[datetime] = None
    gender: Optional[str] = None  # M, F, O, U
    address_city: Optional[str] = None
    address_country: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "identifiers": [
                {"id": pid.identifier, "authority": pid.assigning_authority}
                for pid in self.identifiers
            ],
            "name": {"family": self.family_name, "given": self.given_name},
            "birthDate": self.birth_date.isoformat() if self.birth_date else None,
            "gender": self.gender,
            "address": {"city": self.address_city, "country": self.address_country},
            "telecom": {"phone": self.phone, "email": self.email},
        }


@dataclass
class AuditEvent:
    """ATNA Audit Event."""
    event_id: str
    event_type: AuditEventType
    event_datetime: datetime
    event_outcome: AuditEventOutcome

    # Active Participant (who did it)
    user_id: str
    user_name: Optional[str] = None
    user_role: Optional[str] = None
    user_is_requestor: bool = True
    network_access_point_id: Optional[str] = None  # IP address or hostname
    network_access_point_type: int = 2  # 1=machine name, 2=IP address

    # Audit Source
    audit_source_id: str = ""
    audit_source_type: int = 4  # 4=Application Server
    audit_source_enterprise_site_id: Optional[str] = None

    # Participant Object (what was accessed)
    participant_object_id: Optional[str] = None
    participant_object_type: int = 2  # 1=Person, 2=System Object, 3=Organization, 4=Other
    participant_object_role: int = 3  # 1=Patient, 3=Report, 4=Resource, 24=Query
    participant_object_name: Optional[str] = None
    participant_object_query: Optional[str] = None  # Base64 encoded query
    participant_object_detail: Optional[Dict[str, str]] = None

    def to_xml(self) -> str:
        """Convert to DICOM/IHE ATNA XML format."""
        xml = f'''<?xml version="1.0" encoding="UTF-8"?>
<AuditMessage>
    <EventIdentification EventActionCode="R" EventDateTime="{self.event_datetime.isoformat()}"
        EventOutcomeIndicator="{self.event_outcome.value}">
        <EventID code="{self.event_type.value}"
            codeSystemName="DCM" displayName="{self.event_type.name}"/>
    </EventIdentification>
    <ActiveParticipant UserID="{self.user_id}" UserIsRequestor="{str(self.user_is_requestor).lower()}"'''

        if self.network_access_point_id:
            xml += f'''
        NetworkAccessPointID="{self.network_access_point_id}"
        NetworkAccessPointTypeCode="{self.network_access_point_type}"'''

        xml += f'''/>
    <AuditSourceIdentification AuditSourceID="{self.audit_source_id}">
        <AuditSourceTypeCode code="{self.audit_source_type}"/>
    </AuditSourceIdentification>'''

        if self.participant_object_id:
            xml += f'''
    <ParticipantObjectIdentification ParticipantObjectID="{self.participant_object_id}"
        ParticipantObjectTypeCode="{self.participant_object_type}"
        ParticipantObjectTypeCodeRole="{self.participant_object_role}">'''
            if self.participant_object_name:
                xml += f'''
        <ParticipantObjectName>{self.participant_object_name}</ParticipantObjectName>'''
            if self.participant_object_query:
                xml += f'''
        <ParticipantObjectQuery>{self.participant_object_query}</ParticipantObjectQuery>'''
            xml += '''
    </ParticipantObjectIdentification>'''

        xml += '''
</AuditMessage>'''
        return xml

    def to_fhir(self) -> Dict[str, Any]:
        """Convert to FHIR AuditEvent resource."""
        return {
            "resourceType": "AuditEvent",
            "id": self.event_id,
            "type": {
                "system": "http://dicom.nema.org/resources/ontology/DCM",
                "code": self.event_type.value,
                "display": self.event_type.name,
            },
            "recorded": self.event_datetime.isoformat(),
            "outcome": str(self.event_outcome.value),
            "agent": [{
                "who": {"identifier": {"value": self.user_id}},
                "name": self.user_name,
                "requestor": self.user_is_requestor,
                "network": {
                    "address": self.network_access_point_id,
                    "type": str(self.network_access_point_type),
                } if self.network_access_point_id else None,
            }],
            "source": {
                "site": self.audit_source_enterprise_site_id,
                "observer": {"identifier": {"value": self.audit_source_id}},
                "type": [{"code": str(self.audit_source_type)}],
            },
            "entity": [{
                "what": {"identifier": {"value": self.participant_object_id}},
                "type": {"code": str(self.participant_object_type)},
                "role": {"code": str(self.participant_object_role)},
                "name": self.participant_object_name,
                "query": self.participant_object_query,
            }] if self.participant_object_id else None,
        }


@dataclass
class ConsentDocument:
    """BPPC Consent Document for EHDS secondary use."""
    consent_id: str
    patient_id: str
    status: ConsentStatus
    scope: ConsentScope
    date_time: datetime
    period_start: datetime
    period_end: Optional[datetime] = None
    policy_uri: str = "urn:ehds:policy:secondary-use"
    grantor_organization: Optional[str] = None  # Organization that obtained consent
    data_categories: List[str] = field(default_factory=list)  # Categories covered
    purposes: List[str] = field(default_factory=list)  # Authorized purposes
    opt_out_reason: Optional[str] = None

    def is_valid(self) -> bool:
        """Check if consent is currently valid."""
        if self.status not in [ConsentStatus.ACTIVE]:
            return False
        now = datetime.now()
        if now < self.period_start:
            return False
        if self.period_end and now > self.period_end:
            return False
        return True

    def to_fhir(self) -> Dict[str, Any]:
        """Convert to FHIR Consent resource."""
        return {
            "resourceType": "Consent",
            "id": self.consent_id,
            "status": self.status.value,
            "scope": {
                "coding": [{
                    "system": "http://terminology.hl7.org/CodeSystem/consentscope",
                    "code": "research",
                    "display": "Research",
                }]
            },
            "category": [{
                "coding": [{
                    "system": "urn:ehds:consent-category",
                    "code": self.scope.value,
                    "display": self.scope.name,
                }]
            }],
            "patient": {"reference": f"Patient/{self.patient_id}"},
            "dateTime": self.date_time.isoformat(),
            "policy": [{"uri": self.policy_uri}],
            "provision": {
                "type": "permit" if self.scope != ConsentScope.OPT_OUT else "deny",
                "period": {
                    "start": self.period_start.isoformat(),
                    "end": self.period_end.isoformat() if self.period_end else None,
                },
                "purpose": [
                    {"code": purpose, "system": "urn:ehds:purpose"}
                    for purpose in self.purposes
                ],
                "class": [
                    {"code": cat, "system": "urn:ehds:data-category"}
                    for cat in self.data_categories
                ],
            },
        }


@dataclass
class XUAAssertion:
    """XUA SAML Assertion for cross-enterprise authentication."""
    assertion_id: str
    issuer: str  # Identity Provider
    subject_id: str  # User identifier
    subject_name: str
    subject_role: str  # Healthcare role
    subject_organization: str
    subject_organization_id: str
    purpose_of_use: str  # POU code
    issue_instant: datetime
    not_before: datetime
    not_on_or_after: datetime
    audience: Optional[str] = None  # Target service
    authn_context: str = "urn:oasis:names:tc:SAML:2.0:ac:classes:PasswordProtectedTransport"
    home_community_id: Optional[str] = None

    def is_valid(self) -> bool:
        """Check if assertion is temporally valid."""
        now = datetime.now()
        return self.not_before <= now < self.not_on_or_after

    def to_saml_claims(self) -> Dict[str, str]:
        """Extract SAML claims as dictionary."""
        return {
            "sub": self.subject_id,
            "name": self.subject_name,
            "role": self.subject_role,
            "organization": self.subject_organization,
            "organization_id": self.subject_organization_id,
            "purpose_of_use": self.purpose_of_use,
            "iss": self.issuer,
            "aud": self.audience or "",
            "iat": str(int(self.issue_instant.timestamp())),
            "nbf": str(int(self.not_before.timestamp())),
            "exp": str(int(self.not_on_or_after.timestamp())),
            "home_community_id": self.home_community_id or "",
        }


# =============================================================================
# IHE Profile Implementations
# =============================================================================

class ATNAAuditLogger:
    """
    ATNA (Audit Trail and Node Authentication) Implementation.

    Provides secure audit logging compliant with:
    - IHE ATNA Profile
    - DICOM Audit Message Schema
    - EHDS Article 50 requirements
    """

    def __init__(
        self,
        audit_source_id: str,
        audit_repository_url: Optional[str] = None,
        node_id: Optional[str] = None,
        enterprise_site_id: Optional[str] = None,
    ):
        self.audit_source_id = audit_source_id
        self.audit_repository_url = audit_repository_url
        self.node_id = node_id or str(uuid.uuid4())
        self.enterprise_site_id = enterprise_site_id
        self._audit_log: List[AuditEvent] = []
        self._listeners: List[Callable[[AuditEvent], None]] = []

    def add_listener(self, listener: Callable[[AuditEvent], None]) -> None:
        """Add audit event listener."""
        self._listeners.append(listener)

    def log_event(
        self,
        event_type: AuditEventType,
        outcome: AuditEventOutcome,
        user_id: str,
        participant_object_id: Optional[str] = None,
        user_name: Optional[str] = None,
        user_role: Optional[str] = None,
        network_access_point: Optional[str] = None,
        object_name: Optional[str] = None,
        object_query: Optional[str] = None,
        object_detail: Optional[Dict[str, str]] = None,
    ) -> AuditEvent:
        """Log an audit event."""
        event = AuditEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            event_datetime=datetime.now(),
            event_outcome=outcome,
            user_id=user_id,
            user_name=user_name,
            user_role=user_role,
            user_is_requestor=True,
            network_access_point_id=network_access_point,
            audit_source_id=self.audit_source_id,
            audit_source_enterprise_site_id=self.enterprise_site_id,
            participant_object_id=participant_object_id,
            participant_object_name=object_name,
            participant_object_query=object_query,
            participant_object_detail=object_detail,
        )

        self._audit_log.append(event)

        # Notify listeners
        for listener in self._listeners:
            try:
                listener(event)
            except Exception as e:
                logger.error(f"Audit listener error: {e}")

        # Send to repository if configured
        if self.audit_repository_url:
            self._send_to_repository(event)

        logger.info(
            f"ATNA Audit: {event_type.name} by {user_id}, "
            f"outcome={outcome.name}, object={participant_object_id}"
        )

        return event

    def log_fl_training_start(
        self,
        user_id: str,
        fl_round_id: str,
        model_id: str,
        client_count: int,
    ) -> AuditEvent:
        """Log FL training round start."""
        return self.log_event(
            event_type=AuditEventType.FL_TRAINING_START,
            outcome=AuditEventOutcome.SUCCESS,
            user_id=user_id,
            participant_object_id=fl_round_id,
            object_name=f"FL Round: {model_id}",
            object_detail={
                "model_id": model_id,
                "client_count": str(client_count),
            },
        )

    def log_fl_training_end(
        self,
        user_id: str,
        fl_round_id: str,
        outcome: AuditEventOutcome,
        metrics: Optional[Dict[str, float]] = None,
    ) -> AuditEvent:
        """Log FL training round end."""
        return self.log_event(
            event_type=AuditEventType.FL_TRAINING_END,
            outcome=outcome,
            user_id=user_id,
            participant_object_id=fl_round_id,
            object_detail={k: str(v) for k, v in (metrics or {}).items()},
        )

    def log_fl_model_update(
        self,
        user_id: str,
        client_id: str,
        round_id: str,
        update_size: int,
    ) -> AuditEvent:
        """Log FL model update from client."""
        return self.log_event(
            event_type=AuditEventType.FL_MODEL_UPDATE,
            outcome=AuditEventOutcome.SUCCESS,
            user_id=user_id,
            participant_object_id=client_id,
            object_name=f"Update for round {round_id}",
            object_detail={
                "round_id": round_id,
                "update_size_bytes": str(update_size),
            },
        )

    def log_fl_data_access(
        self,
        user_id: str,
        data_permit_id: str,
        patient_count: int,
        purpose: str,
        outcome: AuditEventOutcome = AuditEventOutcome.SUCCESS,
    ) -> AuditEvent:
        """Log FL data access for training."""
        return self.log_event(
            event_type=AuditEventType.FL_DATA_ACCESS,
            outcome=outcome,
            user_id=user_id,
            participant_object_id=data_permit_id,
            object_name=f"Data access: {purpose}",
            object_detail={
                "patient_count": str(patient_count),
                "purpose": purpose,
            },
        )

    def log_fl_permit_check(
        self,
        user_id: str,
        permit_id: str,
        check_result: bool,
        reason: Optional[str] = None,
    ) -> AuditEvent:
        """Log data permit verification."""
        return self.log_event(
            event_type=AuditEventType.FL_PERMIT_CHECK,
            outcome=AuditEventOutcome.SUCCESS if check_result else AuditEventOutcome.MINOR_FAILURE,
            user_id=user_id,
            participant_object_id=permit_id,
            object_name="Permit verification",
            object_detail={
                "result": str(check_result),
                "reason": reason or "",
            },
        )

    def log_query(
        self,
        user_id: str,
        query_type: str,
        query_params: Dict[str, Any],
        result_count: int,
        outcome: AuditEventOutcome = AuditEventOutcome.SUCCESS,
    ) -> AuditEvent:
        """Log data query event."""
        query_b64 = base64.b64encode(
            json.dumps(query_params).encode()
        ).decode()

        return self.log_event(
            event_type=AuditEventType.QUERY,
            outcome=outcome,
            user_id=user_id,
            participant_object_id=query_type,
            object_name=f"Query: {query_type}",
            object_query=query_b64,
            object_detail={"result_count": str(result_count)},
        )

    def log_security_alert(
        self,
        user_id: str,
        alert_type: str,
        description: str,
        severity: AuditEventOutcome = AuditEventOutcome.SERIOUS_FAILURE,
    ) -> AuditEvent:
        """Log security alert."""
        return self.log_event(
            event_type=AuditEventType.SECURITY_ALERT,
            outcome=severity,
            user_id=user_id,
            participant_object_id=alert_type,
            object_name=description,
        )

    def _send_to_repository(self, event: AuditEvent) -> bool:
        """Send audit event to central repository (stub)."""
        # In production, this would use TLS syslog or REST API
        logger.debug(f"Would send audit event {event.event_id} to {self.audit_repository_url}")
        return True

    def get_audit_log(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        event_type: Optional[AuditEventType] = None,
        user_id: Optional[str] = None,
    ) -> List[AuditEvent]:
        """Query audit log with filters."""
        result = self._audit_log.copy()

        if start_time:
            result = [e for e in result if e.event_datetime >= start_time]
        if end_time:
            result = [e for e in result if e.event_datetime <= end_time]
        if event_type:
            result = [e for e in result if e.event_type == event_type]
        if user_id:
            result = [e for e in result if e.user_id == user_id]

        return result

    def export_audit_trail(
        self,
        format: str = "json",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> str:
        """Export audit trail for compliance reporting."""
        events = self.get_audit_log(start_time, end_time)

        if format == "json":
            return json.dumps([e.to_fhir() for e in events], indent=2)
        elif format == "xml":
            return "\n".join(e.to_xml() for e in events)
        else:
            raise ValueError(f"Unsupported format: {format}")


class BPPCConsentManager:
    """
    BPPC (Basic Patient Privacy Consents) Implementation.

    Manages patient consents for secondary data use under EHDS Regulation,
    particularly Article 33 opt-out mechanism.
    """

    def __init__(
        self,
        organization_id: str,
        default_policy_uri: str = "urn:ehds:policy:secondary-use:default",
        audit_logger: Optional[ATNAAuditLogger] = None,
    ):
        self.organization_id = organization_id
        self.default_policy_uri = default_policy_uri
        self.audit_logger = audit_logger
        self._consents: Dict[str, ConsentDocument] = {}
        self._opt_out_registry: Set[str] = set()  # Patient IDs who opted out

    def register_consent(
        self,
        patient_id: str,
        scope: ConsentScope,
        purposes: List[str],
        data_categories: Optional[List[str]] = None,
        period_years: int = 5,
    ) -> ConsentDocument:
        """Register a new consent for secondary use."""
        consent = ConsentDocument(
            consent_id=str(uuid.uuid4()),
            patient_id=patient_id,
            status=ConsentStatus.ACTIVE,
            scope=scope,
            date_time=datetime.now(),
            period_start=datetime.now(),
            period_end=datetime.now() + timedelta(days=365 * period_years),
            policy_uri=self.default_policy_uri,
            grantor_organization=self.organization_id,
            data_categories=data_categories or ["all"],
            purposes=purposes,
        )

        self._consents[consent.consent_id] = consent

        # Handle opt-out
        if scope == ConsentScope.OPT_OUT:
            self._opt_out_registry.add(patient_id)
            consent.status = ConsentStatus.ACTIVE  # Opt-out is active denial

        if self.audit_logger:
            self.audit_logger.log_event(
                event_type=AuditEventType.PATIENT_RECORD,
                outcome=AuditEventOutcome.SUCCESS,
                user_id=self.organization_id,
                participant_object_id=patient_id,
                object_name=f"Consent registered: {scope.value}",
            )

        logger.info(f"Consent registered: {consent.consent_id} for patient {patient_id}, scope={scope.value}")
        return consent

    def register_opt_out(
        self,
        patient_id: str,
        reason: Optional[str] = None,
    ) -> ConsentDocument:
        """Register patient opt-out under EHDS Article 33."""
        consent = self.register_consent(
            patient_id=patient_id,
            scope=ConsentScope.OPT_OUT,
            purposes=[],
            data_categories=[],
        )
        consent.opt_out_reason = reason
        return consent

    def revoke_consent(self, consent_id: str, reason: Optional[str] = None) -> bool:
        """Revoke an existing consent."""
        if consent_id not in self._consents:
            return False

        consent = self._consents[consent_id]
        consent.status = ConsentStatus.REVOKED

        # If this was an opt-out being revoked, remove from opt-out registry
        if consent.scope == ConsentScope.OPT_OUT:
            self._opt_out_registry.discard(consent.patient_id)

        if self.audit_logger:
            self.audit_logger.log_event(
                event_type=AuditEventType.PATIENT_RECORD,
                outcome=AuditEventOutcome.SUCCESS,
                user_id=self.organization_id,
                participant_object_id=consent.patient_id,
                object_name=f"Consent revoked: {consent_id}",
            )

        logger.info(f"Consent revoked: {consent_id}")
        return True

    def check_consent(
        self,
        patient_id: str,
        purpose: str,
        data_category: Optional[str] = None,
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if data use is permitted for a patient.

        Returns:
            Tuple of (is_permitted, reason)
        """
        # Check opt-out first (EHDS Article 33)
        if patient_id in self._opt_out_registry:
            return False, "Patient has opted out of secondary data use"

        # Find active consents for this patient
        patient_consents = [
            c for c in self._consents.values()
            if c.patient_id == patient_id and c.is_valid()
        ]

        if not patient_consents:
            # Default: EHDS allows secondary use unless opted out
            return True, "Default EHDS secondary use policy (no explicit opt-out)"

        # Check if any consent covers this purpose
        for consent in patient_consents:
            if consent.scope == ConsentScope.OPT_OUT:
                continue

            # Check purpose
            if purpose in consent.purposes or "all" in consent.purposes:
                # Check data category if specified
                if data_category:
                    if data_category in consent.data_categories or "all" in consent.data_categories:
                        return True, f"Covered by consent {consent.consent_id}"
                else:
                    return True, f"Covered by consent {consent.consent_id}"

        return False, "No matching consent found for this purpose"

    def get_opted_out_patients(self) -> Set[str]:
        """Get set of patient IDs who have opted out."""
        return self._opt_out_registry.copy()

    def filter_patients_by_consent(
        self,
        patient_ids: List[str],
        purpose: str,
        data_category: Optional[str] = None,
    ) -> Tuple[List[str], List[str]]:
        """
        Filter patient list by consent status.

        Returns:
            Tuple of (permitted_patients, excluded_patients)
        """
        permitted = []
        excluded = []

        for patient_id in patient_ids:
            is_permitted, _ = self.check_consent(patient_id, purpose, data_category)
            if is_permitted:
                permitted.append(patient_id)
            else:
                excluded.append(patient_id)

        return permitted, excluded

    def get_consent(self, consent_id: str) -> Optional[ConsentDocument]:
        """Get consent by ID."""
        return self._consents.get(consent_id)

    def get_patient_consents(self, patient_id: str) -> List[ConsentDocument]:
        """Get all consents for a patient."""
        return [c for c in self._consents.values() if c.patient_id == patient_id]

    def export_consents_fhir(self, patient_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Export consents as FHIR Consent resources."""
        consents = self._consents.values()
        if patient_id:
            consents = [c for c in consents if c.patient_id == patient_id]
        return [c.to_fhir() for c in consents]


class XUASecurityContext:
    """
    XUA (Cross-Enterprise User Assertion) Implementation.

    Manages SAML-based authentication for cross-enterprise/cross-border
    data access in EHDS context.
    """

    def __init__(
        self,
        issuer: str,
        valid_duration_minutes: int = 60,
        audit_logger: Optional[ATNAAuditLogger] = None,
    ):
        self.issuer = issuer
        self.valid_duration = timedelta(minutes=valid_duration_minutes)
        self.audit_logger = audit_logger
        self._active_assertions: Dict[str, XUAAssertion] = {}
        self._revoked_assertions: Set[str] = set()

    def create_assertion(
        self,
        subject_id: str,
        subject_name: str,
        subject_role: str,
        subject_organization: str,
        subject_organization_id: str,
        purpose_of_use: str,
        audience: Optional[str] = None,
        home_community_id: Optional[str] = None,
    ) -> XUAAssertion:
        """Create a new SAML assertion for user authentication."""
        now = datetime.now()

        assertion = XUAAssertion(
            assertion_id=str(uuid.uuid4()),
            issuer=self.issuer,
            subject_id=subject_id,
            subject_name=subject_name,
            subject_role=subject_role,
            subject_organization=subject_organization,
            subject_organization_id=subject_organization_id,
            purpose_of_use=purpose_of_use,
            issue_instant=now,
            not_before=now,
            not_on_or_after=now + self.valid_duration,
            audience=audience,
            home_community_id=home_community_id,
        )

        self._active_assertions[assertion.assertion_id] = assertion

        if self.audit_logger:
            self.audit_logger.log_event(
                event_type=AuditEventType.USER_AUTHENTICATION,
                outcome=AuditEventOutcome.SUCCESS,
                user_id=subject_id,
                user_name=subject_name,
                user_role=subject_role,
                participant_object_id=assertion.assertion_id,
                object_name="SAML Assertion created",
            )

        logger.info(f"XUA Assertion created: {assertion.assertion_id} for {subject_id}")
        return assertion

    def validate_assertion(self, assertion_id: str) -> Tuple[bool, Optional[str]]:
        """Validate an assertion."""
        if assertion_id in self._revoked_assertions:
            return False, "Assertion has been revoked"

        assertion = self._active_assertions.get(assertion_id)
        if not assertion:
            return False, "Assertion not found"

        if not assertion.is_valid():
            return False, "Assertion has expired"

        return True, None

    def get_assertion(self, assertion_id: str) -> Optional[XUAAssertion]:
        """Get assertion by ID."""
        is_valid, _ = self.validate_assertion(assertion_id)
        if not is_valid:
            return None
        return self._active_assertions.get(assertion_id)

    def revoke_assertion(self, assertion_id: str) -> bool:
        """Revoke an assertion."""
        if assertion_id not in self._active_assertions:
            return False

        self._revoked_assertions.add(assertion_id)
        assertion = self._active_assertions.pop(assertion_id)

        if self.audit_logger:
            self.audit_logger.log_event(
                event_type=AuditEventType.USER_AUTHENTICATION,
                outcome=AuditEventOutcome.SUCCESS,
                user_id=assertion.subject_id,
                participant_object_id=assertion_id,
                object_name="SAML Assertion revoked",
            )

        return True

    def cleanup_expired(self) -> int:
        """Remove expired assertions. Returns count of cleaned assertions."""
        expired = [
            aid for aid, a in self._active_assertions.items()
            if not a.is_valid()
        ]
        for aid in expired:
            del self._active_assertions[aid]
        return len(expired)


class PIXPDQManager:
    """
    PIXm/PDQm (Patient Identifier Cross-referencing / Patient Demographics Query)
    Implementation for Mobile/FHIR-based patient identity management.
    """

    def __init__(
        self,
        domain_oid: str,
        audit_logger: Optional[ATNAAuditLogger] = None,
    ):
        self.domain_oid = domain_oid
        self.audit_logger = audit_logger
        self._patient_index: Dict[str, PatientDemographics] = {}
        self._cross_references: Dict[str, Dict[str, str]] = {}  # local_id -> {domain: id}

    def register_patient(
        self,
        patient: PatientDemographics,
        cross_references: Optional[Dict[str, str]] = None,
    ) -> str:
        """Register patient in the index."""
        local_id = patient.identifiers[0].identifier if patient.identifiers else str(uuid.uuid4())

        self._patient_index[local_id] = patient

        if cross_references:
            self._cross_references[local_id] = cross_references

        if self.audit_logger:
            self.audit_logger.log_event(
                event_type=AuditEventType.PATIENT_RECORD,
                outcome=AuditEventOutcome.SUCCESS,
                user_id="system",
                participant_object_id=local_id,
                object_name="Patient registered in PIX",
            )

        return local_id

    def pix_query(
        self,
        source_identifier: str,
        source_domain: str,
        target_domain: Optional[str] = None,
    ) -> List[PatientIdentifier]:
        """
        PIX Query: Find patient identifiers across domains.

        Args:
            source_identifier: Known patient identifier
            source_domain: Domain/OID of the source identifier
            target_domain: Optional target domain to query

        Returns:
            List of matching patient identifiers
        """
        results = []

        # Find patient by source identifier
        for local_id, patient in self._patient_index.items():
            for pid in patient.identifiers:
                if pid.identifier == source_identifier and pid.assigning_authority == source_domain:
                    # Found the patient, return cross-references
                    if target_domain:
                        xrefs = self._cross_references.get(local_id, {})
                        if target_domain in xrefs:
                            results.append(PatientIdentifier(
                                identifier=xrefs[target_domain],
                                assigning_authority=target_domain,
                            ))
                    else:
                        # Return all identifiers
                        results.extend(patient.identifiers)
                        xrefs = self._cross_references.get(local_id, {})
                        for domain, xref_id in xrefs.items():
                            results.append(PatientIdentifier(
                                identifier=xref_id,
                                assigning_authority=domain,
                            ))
                    break

        if self.audit_logger:
            self.audit_logger.log_query(
                user_id="system",
                query_type="PIX",
                query_params={
                    "source_identifier": source_identifier,
                    "source_domain": source_domain,
                    "target_domain": target_domain,
                },
                result_count=len(results),
            )

        return results

    def pdq_query(
        self,
        family_name: Optional[str] = None,
        given_name: Optional[str] = None,
        birth_date: Optional[datetime] = None,
        gender: Optional[str] = None,
        address_city: Optional[str] = None,
        address_country: Optional[str] = None,
    ) -> List[PatientDemographics]:
        """
        PDQ Query: Search patients by demographics.

        Returns:
            List of matching patients
        """
        results = []

        for patient in self._patient_index.values():
            match = True

            if family_name and patient.family_name.lower() != family_name.lower():
                match = False
            if given_name and patient.given_name.lower() != given_name.lower():
                match = False
            if birth_date and patient.birth_date != birth_date:
                match = False
            if gender and patient.gender != gender:
                match = False
            if address_city and patient.address_city and patient.address_city.lower() != address_city.lower():
                match = False
            if address_country and patient.address_country and patient.address_country.lower() != address_country.lower():
                match = False

            if match:
                results.append(patient)

        if self.audit_logger:
            self.audit_logger.log_query(
                user_id="system",
                query_type="PDQ",
                query_params={
                    "family_name": family_name,
                    "given_name": given_name,
                    "birth_date": birth_date.isoformat() if birth_date else None,
                    "gender": gender,
                },
                result_count=len(results),
            )

        return results

    def add_cross_reference(
        self,
        local_id: str,
        target_domain: str,
        target_id: str,
    ) -> bool:
        """Add cross-reference for a patient."""
        if local_id not in self._patient_index:
            return False

        if local_id not in self._cross_references:
            self._cross_references[local_id] = {}

        self._cross_references[local_id][target_domain] = target_id
        return True


class XDSDocumentRegistry:
    """
    XDS.b Document Registry Implementation.

    Manages document metadata for cross-enterprise document sharing.
    """

    def __init__(
        self,
        registry_id: str,
        home_community_id: str,
        audit_logger: Optional[ATNAAuditLogger] = None,
    ):
        self.registry_id = registry_id
        self.home_community_id = home_community_id
        self.audit_logger = audit_logger
        self._documents: Dict[str, XDSDocumentEntry] = {}
        self._patient_documents: Dict[str, List[str]] = {}  # patient_id -> [doc_ids]

    def register_document(
        self,
        document: XDSDocumentEntry,
    ) -> str:
        """Register a document in the registry."""
        document.repository_unique_id = self.registry_id
        document.home_community_id = self.home_community_id

        self._documents[document.document_unique_id] = document

        if document.patient_id not in self._patient_documents:
            self._patient_documents[document.patient_id] = []
        self._patient_documents[document.patient_id].append(document.document_unique_id)

        if self.audit_logger:
            self.audit_logger.log_event(
                event_type=AuditEventType.IMPORT,
                outcome=AuditEventOutcome.SUCCESS,
                user_id="system",
                participant_object_id=document.document_unique_id,
                object_name=f"Document registered: {document.title or document.type_code}",
            )

        logger.info(f"Document registered: {document.document_unique_id}")
        return document.document_unique_id

    def query_documents(
        self,
        patient_id: Optional[str] = None,
        class_code: Optional[str] = None,
        type_code: Optional[str] = None,
        status: Optional[DocumentStatus] = None,
        creation_time_from: Optional[datetime] = None,
        creation_time_to: Optional[datetime] = None,
        healthcare_facility_type: Optional[str] = None,
        confidentiality_code: Optional[str] = None,
    ) -> List[XDSDocumentEntry]:
        """Query documents with filters."""
        results = list(self._documents.values())

        if patient_id:
            results = [d for d in results if d.patient_id == patient_id]
        if class_code:
            results = [d for d in results if d.class_code == class_code]
        if type_code:
            results = [d for d in results if d.type_code == type_code]
        if status:
            results = [d for d in results if d.status == status]
        if creation_time_from:
            results = [d for d in results if d.creation_time >= creation_time_from]
        if creation_time_to:
            results = [d for d in results if d.creation_time <= creation_time_to]
        if healthcare_facility_type:
            results = [d for d in results if d.healthcare_facility_type == healthcare_facility_type]
        if confidentiality_code:
            results = [d for d in results if d.confidentiality_code == confidentiality_code]

        if self.audit_logger:
            self.audit_logger.log_query(
                user_id="system",
                query_type="XDS_REGISTRY",
                query_params={
                    "patient_id": patient_id,
                    "class_code": class_code,
                    "type_code": type_code,
                },
                result_count=len(results),
            )

        return results

    def get_document(self, document_id: str) -> Optional[XDSDocumentEntry]:
        """Get document by ID."""
        return self._documents.get(document_id)

    def deprecate_document(self, document_id: str) -> bool:
        """Deprecate a document."""
        if document_id not in self._documents:
            return False

        self._documents[document_id].status = DocumentStatus.DEPRECATED
        return True

    def get_patient_documents(self, patient_id: str) -> List[XDSDocumentEntry]:
        """Get all documents for a patient."""
        doc_ids = self._patient_documents.get(patient_id, [])
        return [self._documents[did] for did in doc_ids if did in self._documents]


class XCAGateway:
    """
    XCA (Cross-Community Access) Gateway Implementation.

    Enables cross-community/cross-border document sharing for EHDS.
    """

    def __init__(
        self,
        gateway_id: str,
        home_community_id: str,
        audit_logger: Optional[ATNAAuditLogger] = None,
    ):
        self.gateway_id = gateway_id
        self.home_community_id = home_community_id
        self.audit_logger = audit_logger
        self._responding_gateways: Dict[str, str] = {}  # community_id -> endpoint
        self._local_registry: Optional[XDSDocumentRegistry] = None

    def register_responding_gateway(
        self,
        community_id: str,
        endpoint: str,
    ) -> None:
        """Register a responding gateway for cross-community queries."""
        self._responding_gateways[community_id] = endpoint
        logger.info(f"Registered XCA gateway: {community_id} -> {endpoint}")

    def set_local_registry(self, registry: XDSDocumentRegistry) -> None:
        """Set local document registry."""
        self._local_registry = registry

    def cross_gateway_query(
        self,
        patient_id: str,
        target_communities: Optional[List[str]] = None,
        query_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, List[XDSDocumentEntry]]:
        """
        Cross-Gateway Query: Query documents across communities.

        Args:
            patient_id: Patient identifier
            target_communities: Optional list of community IDs to query
            query_params: Additional query parameters

        Returns:
            Dict mapping community_id to list of documents
        """
        results: Dict[str, List[XDSDocumentEntry]] = {}

        # Query local registry
        if self._local_registry:
            local_docs = self._local_registry.query_documents(
                patient_id=patient_id,
                **(query_params or {}),
            )
            results[self.home_community_id] = local_docs

        # Query remote gateways
        communities = target_communities or list(self._responding_gateways.keys())
        for community_id in communities:
            if community_id == self.home_community_id:
                continue

            if community_id in self._responding_gateways:
                # In production, this would make actual SOAP/REST call
                remote_docs = self._query_remote_gateway(
                    community_id,
                    patient_id,
                    query_params,
                )
                results[community_id] = remote_docs

        if self.audit_logger:
            total_docs = sum(len(docs) for docs in results.values())
            self.audit_logger.log_query(
                user_id="system",
                query_type="XCA_CROSS_GATEWAY",
                query_params={
                    "patient_id": patient_id,
                    "target_communities": communities,
                },
                result_count=total_docs,
            )

        return results

    def _query_remote_gateway(
        self,
        community_id: str,
        patient_id: str,
        query_params: Optional[Dict[str, Any]],
    ) -> List[XDSDocumentEntry]:
        """Query a remote responding gateway (stub)."""
        # In production, this would make actual HTTP/SOAP call
        logger.debug(f"Would query remote gateway {community_id} for patient {patient_id}")
        return []

    def cross_gateway_retrieve(
        self,
        document_requests: List[Tuple[str, str]],  # (community_id, document_id)
    ) -> Dict[str, bytes]:
        """
        Cross-Gateway Retrieve: Retrieve documents from communities.

        Args:
            document_requests: List of (community_id, document_id) tuples

        Returns:
            Dict mapping document_id to document content
        """
        results: Dict[str, bytes] = {}

        for community_id, document_id in document_requests:
            if community_id == self.home_community_id and self._local_registry:
                # Local retrieval
                doc = self._local_registry.get_document(document_id)
                if doc:
                    results[document_id] = b""  # Placeholder for actual content
            else:
                # Remote retrieval
                content = self._retrieve_from_remote(community_id, document_id)
                if content:
                    results[document_id] = content

        if self.audit_logger:
            self.audit_logger.log_event(
                event_type=AuditEventType.EXPORT,
                outcome=AuditEventOutcome.SUCCESS,
                user_id="system",
                participant_object_id=",".join(d[1] for d in document_requests),
                object_name=f"Cross-gateway retrieve: {len(document_requests)} documents",
            )

        return results

    def _retrieve_from_remote(
        self,
        community_id: str,
        document_id: str,
    ) -> Optional[bytes]:
        """Retrieve document from remote gateway (stub)."""
        logger.debug(f"Would retrieve document {document_id} from {community_id}")
        return None


# =============================================================================
# FL-EHDS IHE Integration Manager
# =============================================================================

class IHEIntegrationManager:
    """
    Central manager for IHE profile integration in FL-EHDS.

    Coordinates ATNA audit logging, BPPC consent management,
    XUA authentication, and document sharing capabilities.
    """

    def __init__(
        self,
        organization_id: str,
        home_community_id: str,
        audit_repository_url: Optional[str] = None,
    ):
        self.organization_id = organization_id
        self.home_community_id = home_community_id

        # Initialize ATNA audit logger
        self.audit_logger = ATNAAuditLogger(
            audit_source_id=organization_id,
            audit_repository_url=audit_repository_url,
            enterprise_site_id=home_community_id,
        )

        # Initialize consent manager
        self.consent_manager = BPPCConsentManager(
            organization_id=organization_id,
            audit_logger=self.audit_logger,
        )

        # Initialize security context
        self.security_context = XUASecurityContext(
            issuer=f"urn:ehds:idp:{organization_id}",
            audit_logger=self.audit_logger,
        )

        # Initialize patient identity manager
        self.patient_manager = PIXPDQManager(
            domain_oid=f"2.16.840.1.113883.2.1.{organization_id}",
            audit_logger=self.audit_logger,
        )

        # Initialize document registry
        self.document_registry = XDSDocumentRegistry(
            registry_id=organization_id,
            home_community_id=home_community_id,
            audit_logger=self.audit_logger,
        )

        # Initialize XCA gateway
        self.xca_gateway = XCAGateway(
            gateway_id=organization_id,
            home_community_id=home_community_id,
            audit_logger=self.audit_logger,
        )
        self.xca_gateway.set_local_registry(self.document_registry)

        logger.info(f"IHE Integration Manager initialized for {organization_id}")

    def verify_fl_data_access(
        self,
        user_assertion_id: str,
        patient_ids: List[str],
        purpose: str,
        data_category: Optional[str] = None,
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        Verify data access for FL training.

        Returns:
            Tuple of (permitted_patients, consent_denied, authentication_failed)
        """
        # Validate user authentication
        is_valid, auth_error = self.security_context.validate_assertion(user_assertion_id)
        if not is_valid:
            self.audit_logger.log_security_alert(
                user_id="system",
                alert_type="AUTH_FAILURE",
                description=f"Authentication failed: {auth_error}",
            )
            return [], [], patient_ids

        assertion = self.security_context.get_assertion(user_assertion_id)

        # Check consent for each patient
        permitted, denied = self.consent_manager.filter_patients_by_consent(
            patient_ids,
            purpose,
            data_category,
        )

        # Log the access attempt
        self.audit_logger.log_fl_data_access(
            user_id=assertion.subject_id if assertion else "unknown",
            data_permit_id=f"FL-{purpose}",
            patient_count=len(permitted),
            purpose=purpose,
            outcome=AuditEventOutcome.SUCCESS if permitted else AuditEventOutcome.MINOR_FAILURE,
        )

        return permitted, denied, []

    def create_fl_session(
        self,
        researcher_id: str,
        researcher_name: str,
        organization: str,
        organization_id: str,
        purpose: str,
    ) -> XUAAssertion:
        """Create authenticated session for FL training."""
        return self.security_context.create_assertion(
            subject_id=researcher_id,
            subject_name=researcher_name,
            subject_role="Researcher",
            subject_organization=organization,
            subject_organization_id=organization_id,
            purpose_of_use=purpose,
            home_community_id=self.home_community_id,
        )

    def log_fl_round(
        self,
        user_id: str,
        round_number: int,
        model_id: str,
        participating_clients: List[str],
        metrics: Optional[Dict[str, float]] = None,
        success: bool = True,
    ) -> Tuple[AuditEvent, AuditEvent]:
        """Log FL training round start and end."""
        round_id = f"{model_id}-round-{round_number}"

        start_event = self.audit_logger.log_fl_training_start(
            user_id=user_id,
            fl_round_id=round_id,
            model_id=model_id,
            client_count=len(participating_clients),
        )

        end_event = self.audit_logger.log_fl_training_end(
            user_id=user_id,
            fl_round_id=round_id,
            outcome=AuditEventOutcome.SUCCESS if success else AuditEventOutcome.SERIOUS_FAILURE,
            metrics=metrics,
        )

        return start_event, end_event

    def get_cross_border_documents(
        self,
        patient_id: str,
        target_countries: Optional[List[str]] = None,
    ) -> Dict[str, List[XDSDocumentEntry]]:
        """Query documents across EHDS participating countries."""
        return self.xca_gateway.cross_gateway_query(
            patient_id=patient_id,
            target_communities=target_countries,
        )

    def export_compliance_report(
        self,
        start_time: datetime,
        end_time: datetime,
        format: str = "json",
    ) -> str:
        """Export compliance report for EHDS audit requirements."""
        audit_trail = self.audit_logger.export_audit_trail(
            format=format,
            start_time=start_time,
            end_time=end_time,
        )

        # Add consent summary
        if format == "json":
            report = {
                "report_generated": datetime.now().isoformat(),
                "period": {
                    "start": start_time.isoformat(),
                    "end": end_time.isoformat(),
                },
                "organization": self.organization_id,
                "home_community": self.home_community_id,
                "opt_out_count": len(self.consent_manager.get_opted_out_patients()),
                "audit_events": json.loads(audit_trail),
            }
            return json.dumps(report, indent=2)

        return audit_trail


# =============================================================================
# Convenience Functions
# =============================================================================

def create_ihe_manager(
    organization_id: str,
    home_community_id: str,
    audit_repository_url: Optional[str] = None,
) -> IHEIntegrationManager:
    """Create and configure IHE Integration Manager."""
    return IHEIntegrationManager(
        organization_id=organization_id,
        home_community_id=home_community_id,
        audit_repository_url=audit_repository_url,
    )


def create_audit_logger(
    audit_source_id: str,
    audit_repository_url: Optional[str] = None,
) -> ATNAAuditLogger:
    """Create standalone ATNA audit logger."""
    return ATNAAuditLogger(
        audit_source_id=audit_source_id,
        audit_repository_url=audit_repository_url,
    )


def create_consent_manager(
    organization_id: str,
    audit_logger: Optional[ATNAAuditLogger] = None,
) -> BPPCConsentManager:
    """Create standalone BPPC consent manager."""
    return BPPCConsentManager(
        organization_id=organization_id,
        audit_logger=audit_logger,
    )


# =============================================================================
# Export
# =============================================================================

__all__ = [
    # Enums
    "IHEProfile",
    "DocumentStatus",
    "AuditEventType",
    "AuditEventOutcome",
    "ConsentStatus",
    "ConsentScope",
    # Data Classes
    "XDSDocumentEntry",
    "PatientIdentifier",
    "PatientDemographics",
    "AuditEvent",
    "ConsentDocument",
    "XUAAssertion",
    # IHE Profile Implementations
    "ATNAAuditLogger",
    "BPPCConsentManager",
    "XUASecurityContext",
    "PIXPDQManager",
    "XDSDocumentRegistry",
    "XCAGateway",
    # Integration Manager
    "IHEIntegrationManager",
    # Factory Functions
    "create_ihe_manager",
    "create_audit_logger",
    "create_consent_manager",
]

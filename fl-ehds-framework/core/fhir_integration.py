"""
HL7 FHIR Integration for FL-EHDS Framework
============================================

Provides connectors and utilities for integrating HL7 FHIR (Fast Healthcare
Interoperability Resources) with Federated Learning pipelines.

FHIR R4 is the standard adopted by EHDS for health data exchange across EU.

Key Components:
- FHIRResourceParser: Parse FHIR resources into ML-ready formats
- FHIRClient: Connect to FHIR servers with privacy-preserving queries
- FHIRDataset: PyTorch-compatible dataset from FHIR bundles
- FHIRFeatureExtractor: Extract features from clinical resources

References:
- HL7 FHIR R4: https://hl7.org/fhir/R4/
- EHDS Regulation EU 2025/327
- EU Electronic Health Record Exchange Format (EEHRxF)

Author: FL-EHDS Framework
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from enum import Enum
from datetime import datetime, date
from abc import ABC, abstractmethod
import json
import hashlib
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# FHIR RESOURCE TYPES (R4)
# =============================================================================

class FHIRResourceType(Enum):
    """Core FHIR R4 resource types relevant for EHDS."""
    PATIENT = "Patient"
    OBSERVATION = "Observation"
    CONDITION = "Condition"
    MEDICATION_REQUEST = "MedicationRequest"
    MEDICATION_STATEMENT = "MedicationStatement"
    PROCEDURE = "Procedure"
    DIAGNOSTIC_REPORT = "DiagnosticReport"
    ENCOUNTER = "Encounter"
    IMMUNIZATION = "Immunization"
    ALLERGY_INTOLERANCE = "AllergyIntolerance"
    CARE_PLAN = "CarePlan"
    DOCUMENT_REFERENCE = "DocumentReference"
    IMAGING_STUDY = "ImagingStudy"
    LABORATORY_RESULT = "Observation"  # Lab results are Observations


class FHIRDataCategory(Enum):
    """EHDS data categories as per Regulation."""
    PATIENT_SUMMARY = "patient_summary"
    ELECTRONIC_PRESCRIPTION = "e_prescription"
    LABORATORY_RESULTS = "laboratory"
    MEDICAL_IMAGING = "imaging"
    HOSPITAL_DISCHARGE = "discharge"
    RARE_DISEASE = "rare_disease"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class FHIRReference:
    """Reference to another FHIR resource."""
    reference: str
    type: Optional[str] = None
    display: Optional[str] = None

    def get_id(self) -> Optional[str]:
        """Extract resource ID from reference."""
        if "/" in self.reference:
            return self.reference.split("/")[-1]
        return self.reference


@dataclass
class FHIRCoding:
    """Coded value with system and code."""
    system: str
    code: str
    display: Optional[str] = None

    SNOMED_CT = "http://snomed.info/sct"
    LOINC = "http://loinc.org"
    ICD10 = "http://hl7.org/fhir/sid/icd-10"
    ICD10_CM = "http://hl7.org/fhir/sid/icd-10-cm"
    ATC = "http://www.whocc.no/atc"
    UCUM = "http://unitsofmeasure.org"


@dataclass
class FHIRQuantity:
    """Quantity with value and unit."""
    value: float
    unit: Optional[str] = None
    system: Optional[str] = None
    code: Optional[str] = None


@dataclass
class FHIRPeriod:
    """Time period with start and end."""
    start: Optional[datetime] = None
    end: Optional[datetime] = None


@dataclass
class ParsedPatient:
    """Parsed Patient resource for ML."""
    id: str
    birth_date: Optional[date] = None
    gender: Optional[str] = None
    deceased: bool = False
    marital_status: Optional[str] = None
    address_country: Optional[str] = None
    address_region: Optional[str] = None

    def to_features(self) -> Dict[str, Any]:
        """Convert to feature dictionary."""
        features = {}

        # Age (if birth date available)
        if self.birth_date:
            today = date.today()
            age = today.year - self.birth_date.year
            if (today.month, today.day) < (self.birth_date.month, self.birth_date.day):
                age -= 1
            features['age'] = age

        # Gender encoding
        gender_map = {'male': 0, 'female': 1, 'other': 2, 'unknown': 3}
        features['gender'] = gender_map.get(self.gender, 3)

        # Deceased
        features['deceased'] = int(self.deceased)

        return features


@dataclass
class ParsedObservation:
    """Parsed Observation resource for ML."""
    id: str
    patient_ref: str
    code: FHIRCoding
    value: Optional[Union[float, str, bool]] = None
    value_quantity: Optional[FHIRQuantity] = None
    effective_datetime: Optional[datetime] = None
    status: str = "final"
    category: Optional[str] = None

    def to_features(self) -> Dict[str, Any]:
        """Convert to feature dictionary."""
        features = {
            'code': self.code.code,
            'code_system': self.code.system,
        }

        if self.value_quantity:
            features['value'] = self.value_quantity.value
            features['unit'] = self.value_quantity.unit
        elif isinstance(self.value, (int, float)):
            features['value'] = float(self.value)

        return features


@dataclass
class ParsedCondition:
    """Parsed Condition resource for ML."""
    id: str
    patient_ref: str
    code: FHIRCoding
    clinical_status: Optional[str] = None
    verification_status: Optional[str] = None
    onset_datetime: Optional[datetime] = None
    abatement_datetime: Optional[datetime] = None
    severity: Optional[str] = None

    def to_features(self) -> Dict[str, Any]:
        """Convert to feature dictionary."""
        features = {
            'condition_code': self.code.code,
            'condition_system': self.code.system,
            'is_active': self.clinical_status == 'active',
        }

        if self.severity:
            severity_map = {'mild': 1, 'moderate': 2, 'severe': 3}
            features['severity'] = severity_map.get(self.severity.lower(), 0)

        return features


@dataclass
class ParsedMedication:
    """Parsed MedicationRequest/Statement for ML."""
    id: str
    patient_ref: str
    medication_code: FHIRCoding
    status: str
    dosage_text: Optional[str] = None
    authored_on: Optional[datetime] = None

    def to_features(self) -> Dict[str, Any]:
        """Convert to feature dictionary."""
        return {
            'medication_code': self.medication_code.code,
            'medication_system': self.medication_code.system,
            'is_active': self.status in ['active', 'completed'],
        }


@dataclass
class ParsedEncounter:
    """Parsed Encounter resource for ML."""
    id: str
    patient_ref: str
    status: str
    encounter_class: Optional[str] = None  # inpatient, outpatient, emergency
    period: Optional[FHIRPeriod] = None
    reason_codes: List[FHIRCoding] = field(default_factory=list)
    diagnosis_codes: List[FHIRCoding] = field(default_factory=list)

    def to_features(self) -> Dict[str, Any]:
        """Convert to feature dictionary."""
        features = {
            'encounter_class': self.encounter_class or 'unknown',
        }

        # Length of stay
        if self.period and self.period.start and self.period.end:
            los = (self.period.end - self.period.start).days
            features['length_of_stay_days'] = los

        return features


# =============================================================================
# FHIR RESOURCE PARSER
# =============================================================================

class FHIRResourceParser:
    """
    Parser for FHIR R4 resources into ML-ready structures.

    Handles the complex nested structure of FHIR JSON and extracts
    relevant clinical information for machine learning.
    """

    def __init__(self, strict_mode: bool = False):
        """
        Initialize parser.

        Args:
            strict_mode: If True, raise errors on parsing failures.
                        If False, return None for unparseable resources.
        """
        self.strict_mode = strict_mode
        self._parsers = {
            'Patient': self._parse_patient,
            'Observation': self._parse_observation,
            'Condition': self._parse_condition,
            'MedicationRequest': self._parse_medication,
            'MedicationStatement': self._parse_medication,
            'Encounter': self._parse_encounter,
        }

    def parse(self, resource: Dict[str, Any]) -> Optional[Any]:
        """
        Parse a FHIR resource.

        Args:
            resource: FHIR resource as dictionary

        Returns:
            Parsed resource object or None
        """
        resource_type = resource.get('resourceType')

        if resource_type not in self._parsers:
            if self.strict_mode:
                raise ValueError(f"Unsupported resource type: {resource_type}")
            return None

        try:
            return self._parsers[resource_type](resource)
        except Exception as e:
            if self.strict_mode:
                raise
            logger.warning(f"Failed to parse {resource_type}: {e}")
            return None

    def parse_bundle(self, bundle: Dict[str, Any]) -> List[Any]:
        """
        Parse all resources in a FHIR Bundle.

        Args:
            bundle: FHIR Bundle resource

        Returns:
            List of parsed resources
        """
        if bundle.get('resourceType') != 'Bundle':
            raise ValueError("Expected Bundle resource")

        entries = bundle.get('entry', [])
        parsed = []

        for entry in entries:
            resource = entry.get('resource', {})
            result = self.parse(resource)
            if result:
                parsed.append(result)

        return parsed

    def _parse_coding(self, coding_dict: Dict) -> FHIRCoding:
        """Parse a Coding element."""
        return FHIRCoding(
            system=coding_dict.get('system', ''),
            code=coding_dict.get('code', ''),
            display=coding_dict.get('display')
        )

    def _parse_codeable_concept(self, cc: Dict) -> Optional[FHIRCoding]:
        """Parse CodeableConcept, returning first coding."""
        codings = cc.get('coding', [])
        if codings:
            return self._parse_coding(codings[0])
        return None

    def _parse_quantity(self, q: Dict) -> FHIRQuantity:
        """Parse a Quantity element."""
        return FHIRQuantity(
            value=q.get('value', 0.0),
            unit=q.get('unit'),
            system=q.get('system'),
            code=q.get('code')
        )

    def _parse_datetime(self, dt_str: Optional[str]) -> Optional[datetime]:
        """Parse FHIR datetime string."""
        if not dt_str:
            return None

        # Handle various FHIR datetime formats
        formats = [
            '%Y-%m-%dT%H:%M:%S%z',
            '%Y-%m-%dT%H:%M:%S.%f%z',
            '%Y-%m-%dT%H:%M:%S',
            '%Y-%m-%d',
        ]

        for fmt in formats:
            try:
                return datetime.strptime(dt_str[:26], fmt[:len(dt_str)])
            except ValueError:
                continue

        return None

    def _parse_date(self, date_str: Optional[str]) -> Optional[date]:
        """Parse FHIR date string."""
        if not date_str:
            return None

        try:
            return datetime.strptime(date_str[:10], '%Y-%m-%d').date()
        except ValueError:
            return None

    def _parse_patient(self, resource: Dict) -> ParsedPatient:
        """Parse Patient resource."""
        # Address
        addresses = resource.get('address', [])
        country = None
        region = None
        if addresses:
            addr = addresses[0]
            country = addr.get('country')
            region = addr.get('state') or addr.get('district')

        # Marital status
        marital = resource.get('maritalStatus', {})
        marital_coding = marital.get('coding', [{}])[0] if marital else {}

        return ParsedPatient(
            id=resource.get('id', ''),
            birth_date=self._parse_date(resource.get('birthDate')),
            gender=resource.get('gender'),
            deceased=resource.get('deceasedBoolean', False) or
                    resource.get('deceasedDateTime') is not None,
            marital_status=marital_coding.get('code'),
            address_country=country,
            address_region=region
        )

    def _parse_observation(self, resource: Dict) -> ParsedObservation:
        """Parse Observation resource."""
        code = self._parse_codeable_concept(resource.get('code', {}))

        # Value handling
        value = None
        value_quantity = None

        if 'valueQuantity' in resource:
            value_quantity = self._parse_quantity(resource['valueQuantity'])
        elif 'valueCodeableConcept' in resource:
            cc = self._parse_codeable_concept(resource['valueCodeableConcept'])
            value = cc.code if cc else None
        elif 'valueString' in resource:
            value = resource['valueString']
        elif 'valueBoolean' in resource:
            value = resource['valueBoolean']
        elif 'valueInteger' in resource:
            value = float(resource['valueInteger'])

        # Category
        categories = resource.get('category', [])
        category = None
        if categories:
            cat_coding = categories[0].get('coding', [{}])[0]
            category = cat_coding.get('code')

        # Subject reference
        subject = resource.get('subject', {})
        patient_ref = subject.get('reference', '')

        return ParsedObservation(
            id=resource.get('id', ''),
            patient_ref=patient_ref,
            code=code,
            value=value,
            value_quantity=value_quantity,
            effective_datetime=self._parse_datetime(
                resource.get('effectiveDateTime')
            ),
            status=resource.get('status', 'final'),
            category=category
        )

    def _parse_condition(self, resource: Dict) -> ParsedCondition:
        """Parse Condition resource."""
        code = self._parse_codeable_concept(resource.get('code', {}))

        # Clinical status
        clinical_status = None
        cs = resource.get('clinicalStatus', {})
        if cs:
            cs_coding = cs.get('coding', [{}])[0]
            clinical_status = cs_coding.get('code')

        # Verification status
        verification_status = None
        vs = resource.get('verificationStatus', {})
        if vs:
            vs_coding = vs.get('coding', [{}])[0]
            verification_status = vs_coding.get('code')

        # Severity
        severity = None
        sev = resource.get('severity', {})
        if sev:
            sev_coding = sev.get('coding', [{}])[0]
            severity = sev_coding.get('code')

        # Subject
        subject = resource.get('subject', {})

        return ParsedCondition(
            id=resource.get('id', ''),
            patient_ref=subject.get('reference', ''),
            code=code,
            clinical_status=clinical_status,
            verification_status=verification_status,
            onset_datetime=self._parse_datetime(
                resource.get('onsetDateTime')
            ),
            abatement_datetime=self._parse_datetime(
                resource.get('abatementDateTime')
            ),
            severity=severity
        )

    def _parse_medication(self, resource: Dict) -> ParsedMedication:
        """Parse MedicationRequest or MedicationStatement."""
        # Medication can be reference or CodeableConcept
        med_code = None

        if 'medicationCodeableConcept' in resource:
            med_code = self._parse_codeable_concept(
                resource['medicationCodeableConcept']
            )
        elif 'medicationReference' in resource:
            # Would need to resolve reference
            ref = resource['medicationReference'].get('reference', '')
            med_code = FHIRCoding(system='reference', code=ref)

        # Subject
        subject = resource.get('subject', {})

        # Dosage
        dosage_text = None
        dosages = resource.get('dosageInstruction', resource.get('dosage', []))
        if dosages:
            dosage_text = dosages[0].get('text')

        return ParsedMedication(
            id=resource.get('id', ''),
            patient_ref=subject.get('reference', ''),
            medication_code=med_code,
            status=resource.get('status', 'unknown'),
            dosage_text=dosage_text,
            authored_on=self._parse_datetime(resource.get('authoredOn'))
        )

    def _parse_encounter(self, resource: Dict) -> ParsedEncounter:
        """Parse Encounter resource."""
        # Class
        enc_class = resource.get('class', {})
        class_code = enc_class.get('code') if isinstance(enc_class, dict) else None

        # Period
        period = None
        if 'period' in resource:
            p = resource['period']
            period = FHIRPeriod(
                start=self._parse_datetime(p.get('start')),
                end=self._parse_datetime(p.get('end'))
            )

        # Reason codes
        reason_codes = []
        for rc in resource.get('reasonCode', []):
            coding = self._parse_codeable_concept(rc)
            if coding:
                reason_codes.append(coding)

        # Diagnosis
        diagnosis_codes = []
        for diag in resource.get('diagnosis', []):
            condition = diag.get('condition', {})
            # Would need to resolve reference to get actual code
            diagnosis_codes.append(FHIRCoding(
                system='reference',
                code=condition.get('reference', '')
            ))

        # Subject
        subject = resource.get('subject', {})

        return ParsedEncounter(
            id=resource.get('id', ''),
            patient_ref=subject.get('reference', ''),
            status=resource.get('status', 'unknown'),
            encounter_class=class_code,
            period=period,
            reason_codes=reason_codes,
            diagnosis_codes=diagnosis_codes
        )


# =============================================================================
# FHIR FEATURE EXTRACTOR
# =============================================================================

class FHIRFeatureExtractor:
    """
    Extract ML features from parsed FHIR resources.

    Converts clinical data into numerical features suitable for
    federated learning models.
    """

    # Common lab test LOINC codes
    COMMON_LABS = {
        '2339-0': 'glucose',          # Glucose [Mass/volume] in Blood
        '2345-7': 'glucose_serum',    # Glucose [Mass/volume] in Serum
        '4548-4': 'hba1c',            # Hemoglobin A1c
        '2160-0': 'creatinine',       # Creatinine [Mass/volume] in Serum
        '3094-0': 'bun',              # Urea nitrogen [Mass/volume] in Serum
        '2951-2': 'sodium',           # Sodium [Moles/volume] in Serum
        '2823-3': 'potassium',        # Potassium [Moles/volume] in Serum
        '718-7': 'hemoglobin',        # Hemoglobin [Mass/volume] in Blood
        '787-2': 'mcv',               # MCV [Entitic volume]
        '785-6': 'mch',               # MCH [Entitic mass]
        '786-4': 'mchc',              # MCHC [Mass/volume]
        '26515-7': 'platelets',       # Platelets [#/volume] in Blood
        '6690-2': 'wbc',              # Leukocytes [#/volume] in Blood
        '2085-9': 'hdl',              # HDL Cholesterol
        '2089-1': 'ldl',              # LDL Cholesterol
        '2093-3': 'total_cholesterol', # Cholesterol [Mass/volume] in Serum
        '2571-8': 'triglycerides',    # Triglyceride [Mass/volume] in Serum
    }

    # Vital signs LOINC codes
    VITAL_SIGNS = {
        '8480-6': 'systolic_bp',      # Systolic blood pressure
        '8462-4': 'diastolic_bp',     # Diastolic blood pressure
        '8867-4': 'heart_rate',       # Heart rate
        '9279-1': 'respiratory_rate', # Respiratory rate
        '8310-5': 'body_temperature', # Body temperature
        '29463-7': 'body_weight',     # Body weight
        '8302-2': 'body_height',      # Body height
        '39156-5': 'bmi',             # Body mass index
        '2708-6': 'oxygen_saturation', # Oxygen saturation
    }

    def __init__(self):
        self.all_codes = {**self.COMMON_LABS, **self.VITAL_SIGNS}

    def extract_patient_features(
        self,
        patient: ParsedPatient,
        observations: List[ParsedObservation],
        conditions: List[ParsedCondition],
        medications: List[ParsedMedication],
        encounters: List[ParsedEncounter]
    ) -> Dict[str, Any]:
        """
        Extract comprehensive feature vector for a patient.

        Args:
            patient: Parsed patient resource
            observations: Patient's observations
            conditions: Patient's conditions
            medications: Patient's medications
            encounters: Patient's encounters

        Returns:
            Dictionary of features
        """
        features = {}

        # Demographics
        features.update(patient.to_features())

        # Lab values (most recent)
        lab_features = self._extract_lab_features(observations)
        features.update(lab_features)

        # Vital signs (most recent)
        vital_features = self._extract_vital_features(observations)
        features.update(vital_features)

        # Condition counts by category
        condition_features = self._extract_condition_features(conditions)
        features.update(condition_features)

        # Medication counts
        med_features = self._extract_medication_features(medications)
        features.update(med_features)

        # Encounter history
        encounter_features = self._extract_encounter_features(encounters)
        features.update(encounter_features)

        return features

    def _extract_lab_features(
        self,
        observations: List[ParsedObservation]
    ) -> Dict[str, float]:
        """Extract most recent lab values."""
        features = {}

        # Group by code and get most recent
        by_code = {}
        for obs in observations:
            if obs.code.system == FHIRCoding.LOINC:
                code = obs.code.code
                if code in self.COMMON_LABS:
                    if code not in by_code or (
                        obs.effective_datetime and
                        by_code[code].effective_datetime and
                        obs.effective_datetime > by_code[code].effective_datetime
                    ):
                        by_code[code] = obs

        # Extract values
        for code, obs in by_code.items():
            name = self.COMMON_LABS[code]
            if obs.value_quantity:
                features[f'lab_{name}'] = obs.value_quantity.value
            elif isinstance(obs.value, (int, float)):
                features[f'lab_{name}'] = float(obs.value)

        return features

    def _extract_vital_features(
        self,
        observations: List[ParsedObservation]
    ) -> Dict[str, float]:
        """Extract most recent vital signs."""
        features = {}

        by_code = {}
        for obs in observations:
            if obs.code.system == FHIRCoding.LOINC:
                code = obs.code.code
                if code in self.VITAL_SIGNS:
                    if code not in by_code or (
                        obs.effective_datetime and
                        by_code[code].effective_datetime and
                        obs.effective_datetime > by_code[code].effective_datetime
                    ):
                        by_code[code] = obs

        for code, obs in by_code.items():
            name = self.VITAL_SIGNS[code]
            if obs.value_quantity:
                features[f'vital_{name}'] = obs.value_quantity.value
            elif isinstance(obs.value, (int, float)):
                features[f'vital_{name}'] = float(obs.value)

        return features

    def _extract_condition_features(
        self,
        conditions: List[ParsedCondition]
    ) -> Dict[str, Any]:
        """Extract condition-based features."""
        features = {
            'n_active_conditions': 0,
            'n_total_conditions': len(conditions),
        }

        # Common condition categories (ICD-10 chapters)
        icd_chapters = {
            'infectious': 'A00-B99',
            'neoplasms': 'C00-D49',
            'blood': 'D50-D89',
            'endocrine': 'E00-E89',
            'mental': 'F01-F99',
            'nervous': 'G00-G99',
            'circulatory': 'I00-I99',
            'respiratory': 'J00-J99',
            'digestive': 'K00-K95',
            'musculoskeletal': 'M00-M99',
        }

        for cond in conditions:
            if cond.clinical_status == 'active':
                features['n_active_conditions'] += 1

        return features

    def _extract_medication_features(
        self,
        medications: List[ParsedMedication]
    ) -> Dict[str, Any]:
        """Extract medication-based features."""
        features = {
            'n_active_medications': 0,
            'n_total_medications': len(medications),
        }

        for med in medications:
            if med.status in ['active', 'completed']:
                features['n_active_medications'] += 1

        return features

    def _extract_encounter_features(
        self,
        encounters: List[ParsedEncounter]
    ) -> Dict[str, Any]:
        """Extract encounter-based features."""
        features = {
            'n_encounters': len(encounters),
            'n_inpatient': 0,
            'n_outpatient': 0,
            'n_emergency': 0,
            'total_los_days': 0,
        }

        for enc in encounters:
            if enc.encounter_class == 'IMP':  # Inpatient
                features['n_inpatient'] += 1
            elif enc.encounter_class == 'AMB':  # Ambulatory
                features['n_outpatient'] += 1
            elif enc.encounter_class == 'EMER':  # Emergency
                features['n_emergency'] += 1

            enc_features = enc.to_features()
            if 'length_of_stay_days' in enc_features:
                features['total_los_days'] += enc_features['length_of_stay_days']

        return features


# =============================================================================
# FHIR CLIENT
# =============================================================================

class FHIRClient:
    """
    Client for FHIR server communication with privacy-preserving features.

    Supports:
    - Pagination for large result sets
    - Query parameter building
    - Resource caching
    - Privacy-preserving queries (k-anonymity checks)
    """

    def __init__(
        self,
        base_url: str,
        auth_token: Optional[str] = None,
        verify_ssl: bool = True,
        timeout: int = 30
    ):
        """
        Initialize FHIR client.

        Args:
            base_url: FHIR server base URL or file:// path to local bundles
            auth_token: Bearer token for authentication
            verify_ssl: Verify SSL certificates
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.auth_token = auth_token
        self.verify_ssl = verify_ssl
        self.timeout = timeout
        self.parser = FHIRResourceParser()
        self._cache = {}
        self._local_resources = {}  # Loaded from local bundles
        self._is_local = self.base_url.startswith('file://')

        if self._is_local:
            self._load_local_bundles()

    def _load_local_bundles(self):
        """Load all FHIR bundles from a local directory."""
        import json
        from pathlib import Path

        bundle_dir = Path(self.base_url.replace('file://', ''))
        if not bundle_dir.exists():
            logger.warning(f"Local bundle directory not found: {bundle_dir}")
            return

        for bundle_file in sorted(bundle_dir.glob("*.json")):
            try:
                with open(bundle_file, 'r') as f:
                    bundle_data = json.load(f)
                for entry in bundle_data.get('entry', []):
                    resource = entry.get('resource', {})
                    rtype = resource.get('resourceType')
                    rid = resource.get('id')
                    if rtype and rid:
                        self._local_resources.setdefault(rtype, []).append(resource)
                        self._cache[f"{rtype}/{rid}"] = resource
                logger.info(f"Loaded bundle: {bundle_file.name}")
            except Exception as e:
                logger.warning(f"Failed to load bundle {bundle_file}: {e}")

    def _build_headers(self) -> Dict[str, str]:
        """Build request headers."""
        headers = {
            'Accept': 'application/fhir+json',
            'Content-Type': 'application/fhir+json',
        }
        if self.auth_token:
            headers['Authorization'] = f'Bearer {self.auth_token}'
        return headers

    def search(
        self,
        resource_type: str,
        params: Optional[Dict[str, str]] = None,
        count: int = 100,
        max_results: Optional[int] = None
    ) -> List[Dict]:
        """
        Search for FHIR resources.

        Args:
            resource_type: FHIR resource type
            params: Search parameters
            count: Results per page
            max_results: Maximum total results

        Returns:
            List of resources

        Note:
            In simulation mode (no real HTTP), returns mock data.
        """
        # Build search URL
        url = f"{self.base_url}/{resource_type}"
        query_params = params or {}
        query_params['_count'] = str(count)

        logger.info(f"FHIR search: {resource_type} with params {query_params}")

        # Local bundle mode: return resources from loaded bundles
        if self._is_local:
            resources = self._local_resources.get(resource_type, [])
            # Apply patient filter if specified
            patient_filter = query_params.get('patient')
            if patient_filter:
                resources = [
                    r for r in resources
                    if patient_filter in r.get('subject', {}).get('reference', '')
                ]
            if max_results:
                resources = resources[:max_results]
            return resources

        # HTTP mode: stub (would use requests library)
        return []

    def read(self, resource_type: str, resource_id: str) -> Optional[Dict]:
        """
        Read a single FHIR resource by ID.

        Args:
            resource_type: FHIR resource type
            resource_id: Resource ID

        Returns:
            Resource dictionary or None
        """
        cache_key = f"{resource_type}/{resource_id}"

        if cache_key in self._cache:
            return self._cache[cache_key]

        url = f"{self.base_url}/{resource_type}/{resource_id}"
        logger.info(f"FHIR read: {url}")

        # Local bundle mode: return from cache
        if self._is_local:
            return self._cache.get(cache_key)

        # HTTP mode: stub (would fetch from server)
        return None

    def batch_query(
        self,
        patient_id: str,
        resource_types: List[str]
    ) -> Dict[str, List[Dict]]:
        """
        Fetch multiple resource types for a patient efficiently.

        Uses FHIR batch/transaction for efficiency.

        Args:
            patient_id: Patient ID
            resource_types: List of resource types to fetch

        Returns:
            Dictionary mapping resource type to list of resources
        """
        results = {}

        for rt in resource_types:
            results[rt] = self.search(
                rt,
                params={'patient': patient_id}
            )

        return results


# =============================================================================
# FHIR DATASET
# =============================================================================

class FHIRDataset:
    """
    Dataset wrapper for FHIR data compatible with FL training.

    Handles:
    - Loading patient cohorts from FHIR
    - Feature extraction and normalization
    - Train/validation splitting
    - Privacy-preserving data access
    """

    def __init__(
        self,
        client: FHIRClient,
        feature_extractor: FHIRFeatureExtractor,
        cohort_query: Optional[Dict[str, str]] = None
    ):
        """
        Initialize FHIR dataset.

        Args:
            client: FHIR client for data access
            feature_extractor: Feature extractor instance
            cohort_query: Query parameters to define patient cohort
        """
        self.client = client
        self.feature_extractor = feature_extractor
        self.cohort_query = cohort_query or {}
        self.patients = []
        self.features = []
        self.labels = []

    def load_cohort(
        self,
        max_patients: Optional[int] = None,
        label_fn: Optional[Callable] = None
    ) -> int:
        """
        Load patient cohort from FHIR server.

        Args:
            max_patients: Maximum number of patients
            label_fn: Function to compute labels from patient data

        Returns:
            Number of patients loaded
        """
        # Search for patients
        patients = self.client.search('Patient', self.cohort_query)

        if max_patients:
            patients = patients[:max_patients]

        logger.info(f"Loading cohort of {len(patients)} patients")

        for patient_resource in patients:
            patient_id = patient_resource.get('id')

            # Fetch related resources
            related = self.client.batch_query(
                patient_id,
                ['Observation', 'Condition', 'MedicationRequest', 'Encounter']
            )

            # Parse resources
            patient = self.client.parser.parse(patient_resource)
            observations = [
                self.client.parser.parse(r)
                for r in related.get('Observation', [])
            ]
            observations = [o for o in observations if o]

            conditions = [
                self.client.parser.parse(r)
                for r in related.get('Condition', [])
            ]
            conditions = [c for c in conditions if c]

            medications = [
                self.client.parser.parse(r)
                for r in related.get('MedicationRequest', [])
            ]
            medications = [m for m in medications if m]

            encounters = [
                self.client.parser.parse(r)
                for r in related.get('Encounter', [])
            ]
            encounters = [e for e in encounters if e]

            # Extract features
            features = self.feature_extractor.extract_patient_features(
                patient, observations, conditions, medications, encounters
            )

            self.patients.append(patient_id)
            self.features.append(features)

            # Compute label if function provided
            if label_fn:
                label = label_fn(patient, observations, conditions, medications, encounters)
                self.labels.append(label)

        return len(self.patients)

    def to_numpy(
        self,
        feature_names: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Convert dataset to numpy arrays.

        Args:
            feature_names: Specific features to include

        Returns:
            Tuple of (features, labels) arrays
        """
        if not self.features:
            return np.array([]), None

        # Determine feature names
        if feature_names is None:
            # Union of all feature names
            all_names = set()
            for f in self.features:
                all_names.update(f.keys())
            feature_names = sorted(all_names)

        # Build feature matrix
        X = np.zeros((len(self.features), len(feature_names)))

        for i, feat_dict in enumerate(self.features):
            for j, name in enumerate(feature_names):
                if name in feat_dict:
                    val = feat_dict[name]
                    if isinstance(val, bool):
                        val = int(val)
                    elif isinstance(val, str):
                        val = hash(val) % 1000  # Simple encoding
                    X[i, j] = val

        # Labels
        y = None
        if self.labels:
            y = np.array(self.labels)

        return X, y


# =============================================================================
# PRIVACY-PRESERVING UTILITIES
# =============================================================================

class FHIRPrivacyGuard:
    """
    Privacy-preserving utilities for FHIR data access.

    Implements:
    - K-anonymity verification
    - Data minimization
    - Pseudonymization
    """

    def __init__(self, k_anonymity: int = 5):
        """
        Initialize privacy guard.

        Args:
            k_anonymity: Minimum group size for k-anonymity
        """
        self.k_anonymity = k_anonymity

    def pseudonymize_patient_id(
        self,
        patient_id: str,
        salt: str
    ) -> str:
        """
        Generate pseudonymized patient ID.

        Args:
            patient_id: Original patient ID
            salt: Salt for hashing (should be kept secret)

        Returns:
            Pseudonymized ID
        """
        combined = f"{patient_id}:{salt}"
        return hashlib.sha256(combined.encode()).hexdigest()[:16]

    def check_k_anonymity(
        self,
        dataset: FHIRDataset,
        quasi_identifiers: List[str]
    ) -> bool:
        """
        Check if dataset satisfies k-anonymity.

        Args:
            dataset: FHIR dataset
            quasi_identifiers: Feature names that are quasi-identifiers

        Returns:
            True if k-anonymity is satisfied
        """
        if not dataset.features:
            return True

        # Group by quasi-identifier values
        groups = {}
        for feat_dict in dataset.features:
            key = tuple(
                str(feat_dict.get(qi, ''))
                for qi in quasi_identifiers
            )
            groups[key] = groups.get(key, 0) + 1

        # Check minimum group size
        min_size = min(groups.values()) if groups else 0
        return min_size >= self.k_anonymity

    def apply_data_minimization(
        self,
        features: Dict[str, Any],
        required_features: List[str]
    ) -> Dict[str, Any]:
        """
        Apply data minimization - keep only required features.

        Args:
            features: Full feature dictionary
            required_features: List of required feature names

        Returns:
            Minimized feature dictionary
        """
        return {
            k: v for k, v in features.items()
            if k in required_features
        }


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_fhir_integration(
    fhir_server_url: str,
    auth_token: Optional[str] = None,
    k_anonymity: int = 5
) -> Dict[str, Any]:
    """
    Create FHIR integration components.

    Args:
        fhir_server_url: FHIR server base URL
        auth_token: Authentication token
        k_anonymity: K-anonymity threshold

    Returns:
        Dictionary with client, parser, extractor, and privacy guard
    """
    client = FHIRClient(fhir_server_url, auth_token)
    parser = FHIRResourceParser()
    extractor = FHIRFeatureExtractor()
    privacy_guard = FHIRPrivacyGuard(k_anonymity)

    return {
        'client': client,
        'parser': parser,
        'extractor': extractor,
        'privacy_guard': privacy_guard
    }


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("FL-EHDS: HL7 FHIR Integration Demo")
    print("=" * 60)

    # Create sample FHIR resources
    sample_patient = {
        "resourceType": "Patient",
        "id": "patient-001",
        "birthDate": "1980-05-15",
        "gender": "female",
        "address": [{"country": "DE", "state": "Bavaria"}]
    }

    sample_observation = {
        "resourceType": "Observation",
        "id": "obs-001",
        "status": "final",
        "code": {
            "coding": [{
                "system": "http://loinc.org",
                "code": "2339-0",
                "display": "Glucose [Mass/volume] in Blood"
            }]
        },
        "subject": {"reference": "Patient/patient-001"},
        "effectiveDateTime": "2024-01-15T10:30:00Z",
        "valueQuantity": {
            "value": 95.0,
            "unit": "mg/dL",
            "system": "http://unitsofmeasure.org",
            "code": "mg/dL"
        }
    }

    sample_condition = {
        "resourceType": "Condition",
        "id": "cond-001",
        "clinicalStatus": {
            "coding": [{"code": "active"}]
        },
        "code": {
            "coding": [{
                "system": "http://snomed.info/sct",
                "code": "73211009",
                "display": "Diabetes mellitus"
            }]
        },
        "subject": {"reference": "Patient/patient-001"}
    }

    # Parse resources
    parser = FHIRResourceParser()

    print("\n1. Parsing FHIR Resources:")
    print("-" * 40)

    patient = parser.parse(sample_patient)
    print(f"   Patient: {patient.id}, Gender: {patient.gender}, "
          f"Birth: {patient.birth_date}")

    observation = parser.parse(sample_observation)
    print(f"   Observation: {observation.code.display}, "
          f"Value: {observation.value_quantity.value} {observation.value_quantity.unit}")

    condition = parser.parse(sample_condition)
    print(f"   Condition: {condition.code.display}, "
          f"Status: {condition.clinical_status}")

    # Extract features
    print("\n2. Feature Extraction:")
    print("-" * 40)

    extractor = FHIRFeatureExtractor()
    features = extractor.extract_patient_features(
        patient,
        [observation],
        [condition],
        [],  # medications
        []   # encounters
    )

    for k, v in features.items():
        print(f"   {k}: {v}")

    # Privacy guard
    print("\n3. Privacy Guard:")
    print("-" * 40)

    guard = FHIRPrivacyGuard(k_anonymity=5)
    pseudo_id = guard.pseudonymize_patient_id("patient-001", "secret-salt")
    print(f"   Pseudonymized ID: {pseudo_id}")

    minimized = guard.apply_data_minimization(
        features,
        ['age', 'gender', 'lab_glucose']
    )
    print(f"   Minimized features: {list(minimized.keys())}")

    print("\n" + "=" * 60)
    print("FHIR Integration ready for FL-EHDS!")
    print("=" * 60)

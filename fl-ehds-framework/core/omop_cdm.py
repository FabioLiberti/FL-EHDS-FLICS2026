"""
OMOP CDM Support for FL-EHDS Framework
========================================

Provides utilities for working with OMOP Common Data Model in
Federated Learning pipelines for EHDS.

OMOP CDM is the standard for observational health data harmonization,
enabling consistent representation across EU healthcare institutions.

Key Components:
- OMOPDomainMapper: Map clinical data to OMOP domains
- OMOPVocabularyService: Vocabulary lookups (SNOMED, ICD, LOINC)
- OMOPCohortBuilder: Build patient cohorts using OMOP criteria
- OMOPFeatureExtractor: Extract ML features from OMOP tables
- OMOPFederatedQuery: Privacy-preserving distributed queries

References:
- OMOP CDM v5.4: https://ohdsi.github.io/CommonDataModel/
- OHDSI: https://www.ohdsi.org/
- EHDS and OMOP: European Health Data Space interoperability

Author: FL-EHDS Framework
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from enum import Enum
from datetime import datetime, date, timedelta
from abc import ABC, abstractmethod
import json
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# OMOP CDM DOMAINS
# =============================================================================

class OMOPDomain(Enum):
    """OMOP CDM clinical domains."""
    CONDITION = "Condition"
    DRUG = "Drug"
    PROCEDURE = "Procedure"
    MEASUREMENT = "Measurement"
    OBSERVATION = "Observation"
    DEVICE = "Device"
    SPECIMEN = "Specimen"
    VISIT = "Visit"
    DEATH = "Death"
    NOTE = "Note"


class OMOPVocabulary(Enum):
    """Standard vocabularies in OMOP CDM."""
    SNOMED = "SNOMED"
    ICD10 = "ICD10"
    ICD10CM = "ICD10CM"
    ICD9CM = "ICD9CM"
    LOINC = "LOINC"
    RXNORM = "RxNorm"
    ATC = "ATC"
    CPT4 = "CPT4"
    HCPCS = "HCPCS"
    NDC = "NDC"
    UCUM = "UCUM"


# =============================================================================
# OMOP CDM DATA CLASSES
# =============================================================================

@dataclass
class OMOPConcept:
    """OMOP Concept table representation."""
    concept_id: int
    concept_name: str
    domain_id: str
    vocabulary_id: str
    concept_class_id: str
    standard_concept: Optional[str] = None  # 'S' = Standard
    concept_code: Optional[str] = None
    valid_start_date: Optional[date] = None
    valid_end_date: Optional[date] = None


@dataclass
class OMOPPerson:
    """OMOP Person table representation."""
    person_id: int
    gender_concept_id: int
    year_of_birth: int
    month_of_birth: Optional[int] = None
    day_of_birth: Optional[int] = None
    birth_datetime: Optional[datetime] = None
    race_concept_id: Optional[int] = None
    ethnicity_concept_id: Optional[int] = None
    location_id: Optional[int] = None
    provider_id: Optional[int] = None
    care_site_id: Optional[int] = None

    def get_age(self, reference_date: Optional[date] = None) -> int:
        """Calculate age as of reference date."""
        ref = reference_date or date.today()
        age = ref.year - self.year_of_birth
        if self.month_of_birth and self.day_of_birth:
            if (ref.month, ref.day) < (self.month_of_birth, self.day_of_birth):
                age -= 1
        return age


@dataclass
class OMOPVisitOccurrence:
    """OMOP Visit Occurrence table representation."""
    visit_occurrence_id: int
    person_id: int
    visit_concept_id: int  # Inpatient, Outpatient, ER, etc.
    visit_start_date: date
    visit_start_datetime: Optional[datetime] = None
    visit_end_date: Optional[date] = None
    visit_end_datetime: Optional[datetime] = None
    visit_type_concept_id: Optional[int] = None
    provider_id: Optional[int] = None
    care_site_id: Optional[int] = None
    visit_source_value: Optional[str] = None
    admitted_from_concept_id: Optional[int] = None
    discharged_to_concept_id: Optional[int] = None

    # Standard visit concept IDs
    INPATIENT_VISIT = 9201
    OUTPATIENT_VISIT = 9202
    EMERGENCY_VISIT = 9203

    def length_of_stay(self) -> Optional[int]:
        """Calculate length of stay in days."""
        if self.visit_end_date:
            return (self.visit_end_date - self.visit_start_date).days
        return None


@dataclass
class OMOPConditionOccurrence:
    """OMOP Condition Occurrence table representation."""
    condition_occurrence_id: int
    person_id: int
    condition_concept_id: int
    condition_start_date: date
    condition_start_datetime: Optional[datetime] = None
    condition_end_date: Optional[date] = None
    condition_end_datetime: Optional[datetime] = None
    condition_type_concept_id: Optional[int] = None
    condition_status_concept_id: Optional[int] = None
    stop_reason: Optional[str] = None
    provider_id: Optional[int] = None
    visit_occurrence_id: Optional[int] = None
    condition_source_value: Optional[str] = None
    condition_source_concept_id: Optional[int] = None


@dataclass
class OMOPDrugExposure:
    """OMOP Drug Exposure table representation."""
    drug_exposure_id: int
    person_id: int
    drug_concept_id: int
    drug_exposure_start_date: date
    drug_exposure_start_datetime: Optional[datetime] = None
    drug_exposure_end_date: Optional[date] = None
    drug_exposure_end_datetime: Optional[datetime] = None
    verbatim_end_date: Optional[date] = None
    drug_type_concept_id: Optional[int] = None
    stop_reason: Optional[str] = None
    refills: Optional[int] = None
    quantity: Optional[float] = None
    days_supply: Optional[int] = None
    sig: Optional[str] = None  # Dosage instructions
    route_concept_id: Optional[int] = None
    lot_number: Optional[str] = None
    provider_id: Optional[int] = None
    visit_occurrence_id: Optional[int] = None
    drug_source_value: Optional[str] = None
    drug_source_concept_id: Optional[int] = None


@dataclass
class OMOPMeasurement:
    """OMOP Measurement table representation (lab results, vitals)."""
    measurement_id: int
    person_id: int
    measurement_concept_id: int
    measurement_date: date
    measurement_datetime: Optional[datetime] = None
    measurement_time: Optional[str] = None
    measurement_type_concept_id: Optional[int] = None
    operator_concept_id: Optional[int] = None
    value_as_number: Optional[float] = None
    value_as_concept_id: Optional[int] = None
    unit_concept_id: Optional[int] = None
    range_low: Optional[float] = None
    range_high: Optional[float] = None
    provider_id: Optional[int] = None
    visit_occurrence_id: Optional[int] = None
    measurement_source_value: Optional[str] = None
    measurement_source_concept_id: Optional[int] = None
    unit_source_value: Optional[str] = None
    value_source_value: Optional[str] = None

    def is_abnormal(self) -> Optional[bool]:
        """Check if value is outside normal range."""
        if self.value_as_number is None:
            return None
        if self.range_low is not None and self.value_as_number < self.range_low:
            return True
        if self.range_high is not None and self.value_as_number > self.range_high:
            return True
        return False


@dataclass
class OMOPProcedureOccurrence:
    """OMOP Procedure Occurrence table representation."""
    procedure_occurrence_id: int
    person_id: int
    procedure_concept_id: int
    procedure_date: date
    procedure_datetime: Optional[datetime] = None
    procedure_end_date: Optional[date] = None
    procedure_end_datetime: Optional[datetime] = None
    procedure_type_concept_id: Optional[int] = None
    modifier_concept_id: Optional[int] = None
    quantity: Optional[int] = None
    provider_id: Optional[int] = None
    visit_occurrence_id: Optional[int] = None
    procedure_source_value: Optional[str] = None
    procedure_source_concept_id: Optional[int] = None


@dataclass
class OMOPDeath:
    """OMOP Death table representation."""
    person_id: int
    death_date: date
    death_datetime: Optional[datetime] = None
    death_type_concept_id: Optional[int] = None
    cause_concept_id: Optional[int] = None
    cause_source_value: Optional[str] = None
    cause_source_concept_id: Optional[int] = None


# =============================================================================
# VOCABULARY SERVICE
# =============================================================================

class OMOPVocabularyService:
    """
    Service for OMOP vocabulary lookups and mappings.

    Provides:
    - Concept lookup by ID or code
    - Vocabulary mapping (e.g., ICD10 to SNOMED)
    - Concept hierarchy traversal
    - Concept set operations
    """

    # Common concept IDs
    GENDER_MALE = 8507
    GENDER_FEMALE = 8532
    RACE_WHITE = 8527
    RACE_BLACK = 8516
    RACE_ASIAN = 8515

    # Common condition concept IDs
    DIABETES_TYPE2 = 201826
    HYPERTENSION = 320128
    HEART_FAILURE = 316139
    COPD = 255573
    ASTHMA = 317009
    CKD = 46271022

    # Common measurement concept IDs (LOINC-based)
    GLUCOSE_BLOOD = 3004501
    HBA1C = 3004410
    CREATININE_SERUM = 3016723
    HEMOGLOBIN = 3000963
    WBC_COUNT = 3010813
    PLATELET_COUNT = 3024929
    SYSTOLIC_BP = 3004249
    DIASTOLIC_BP = 3012888
    HEART_RATE = 3027018
    BMI = 3038553

    def __init__(self, concept_cache: Optional[Dict[int, OMOPConcept]] = None):
        """
        Initialize vocabulary service.

        Args:
            concept_cache: Pre-loaded concept cache
        """
        self.concept_cache = concept_cache or {}
        self._build_default_concepts()
        self._build_mapping_tables()

    def _build_default_concepts(self):
        """Build cache of commonly used concepts."""
        defaults = [
            OMOPConcept(self.GENDER_MALE, "Male", "Gender", "Gender", "Gender"),
            OMOPConcept(self.GENDER_FEMALE, "Female", "Gender", "Gender", "Gender"),
            OMOPConcept(self.DIABETES_TYPE2, "Type 2 diabetes mellitus",
                       "Condition", "SNOMED", "Clinical Finding", "S"),
            OMOPConcept(self.HYPERTENSION, "Essential hypertension",
                       "Condition", "SNOMED", "Clinical Finding", "S"),
            OMOPConcept(self.GLUCOSE_BLOOD, "Glucose [Mass/volume] in Blood",
                       "Measurement", "LOINC", "Lab Test", "S"),
            OMOPConcept(self.HBA1C, "Hemoglobin A1c/Hemoglobin.total in Blood",
                       "Measurement", "LOINC", "Lab Test", "S"),
        ]

        for concept in defaults:
            self.concept_cache[concept.concept_id] = concept

    def _build_mapping_tables(self):
        """Build local vocabulary -> OMOP standard concept ID mappings.

        Covers 6 European coding systems used in EHDS cross-border scenarios.
        Each maps common clinical conditions to standard OMOP (SNOMED) concept IDs.
        """
        # Additional common condition concept IDs
        MI = 4329847           # Myocardial infarction
        STROKE = 381591        # Cerebrovascular disease
        PNEUMONIA = 255848     # Pneumonia
        DEPRESSION = 440383    # Depressive disorder
        OBESITY = 433736       # Obesity
        ATRIAL_FIB = 313217    # Atrial fibrillation
        CANCER = 443392        # Malignant neoplasm
        ANEMIA = 439777        # Anemia
        UTI = 81902            # Urinary tract infection

        # WHO ICD-10 base mappings
        icd10_base = {
            "E11": self.DIABETES_TYPE2, "E11.9": self.DIABETES_TYPE2,
            "I10": self.HYPERTENSION,
            "I50": self.HEART_FAILURE, "I50.9": self.HEART_FAILURE,
            "J44": self.COPD, "J44.1": self.COPD,
            "J45": self.ASTHMA, "J45.9": self.ASTHMA,
            "N18": self.CKD, "N18.9": self.CKD,
            "I21": MI, "I21.9": MI,
            "I63": STROKE, "I63.9": STROKE,
            "J18": PNEUMONIA, "J18.9": PNEUMONIA,
            "F32": DEPRESSION, "F33": DEPRESSION,
            "E66": OBESITY, "E66.9": OBESITY,
            "I48": ATRIAL_FIB, "I48.9": ATRIAL_FIB,
            "C80": CANCER, "C80.1": CANCER,
            "D64": ANEMIA, "D64.9": ANEMIA,
            "N39.0": UTI,
        }

        # German ICD-10-GM (uses trailing zeros and extended codes)
        icd10gm = {
            "E11.90": self.DIABETES_TYPE2, "E11.91": self.DIABETES_TYPE2,
            "I10.00": self.HYPERTENSION, "I10.90": self.HYPERTENSION,
            "I50.90": self.HEART_FAILURE, "I50.19": self.HEART_FAILURE,
            "J44.10": self.COPD, "J44.19": self.COPD,
            "J45.90": self.ASTHMA, "J45.99": self.ASTHMA,
            "N18.90": self.CKD, "N18.5": self.CKD,
            "I21.90": MI, "I21.0": MI,
            "I63.90": STROKE, "I63.5": STROKE,
            "J18.90": PNEUMONIA, "J18.0": PNEUMONIA,
            "F32.9": DEPRESSION, "F33.1": DEPRESSION,
            "E66.00": OBESITY, "E66.09": OBESITY,
            "I48.90": ATRIAL_FIB, "I48.0": ATRIAL_FIB,
            "C80.0": CANCER, "C80.9": CANCER,
            "D64.9": ANEMIA, "D64.8": ANEMIA,
            "N39.0": UTI,
        }

        # French CIM-10 (no dots, French adaptation)
        cim10 = {
            "E119": self.DIABETES_TYPE2, "E110": self.DIABETES_TYPE2,
            "I10": self.HYPERTENSION,
            "I509": self.HEART_FAILURE, "I500": self.HEART_FAILURE,
            "J441": self.COPD, "J449": self.COPD,
            "J459": self.ASTHMA, "J450": self.ASTHMA,
            "N189": self.CKD, "N185": self.CKD,
            "I219": MI, "I210": MI,
            "I639": STROKE, "I630": STROKE,
            "J189": PNEUMONIA, "J180": PNEUMONIA,
            "F329": DEPRESSION, "F331": DEPRESSION,
            "E669": OBESITY, "E660": OBESITY,
            "I489": ATRIAL_FIB, "I480": ATRIAL_FIB,
            "C800": CANCER, "C809": CANCER,
            "D649": ANEMIA, "D648": ANEMIA,
            "N390": UTI,
        }

        # Italian ICD-9-CM (legacy system, still used for some records)
        icd9cm = {
            "250.00": self.DIABETES_TYPE2, "250.02": self.DIABETES_TYPE2,
            "401.9": self.HYPERTENSION, "401.1": self.HYPERTENSION,
            "428.0": self.HEART_FAILURE, "428.9": self.HEART_FAILURE,
            "496": self.COPD, "491.21": self.COPD,
            "493.90": self.ASTHMA, "493.00": self.ASTHMA,
            "585.9": self.CKD, "585.6": self.CKD,
            "410.9": MI, "410.71": MI,
            "436": STROKE, "434.91": STROKE,
            "486": PNEUMONIA, "485": PNEUMONIA,
            "311": DEPRESSION, "296.20": DEPRESSION,
            "278.00": OBESITY, "278.01": OBESITY,
            "427.31": ATRIAL_FIB,
            "199.1": CANCER, "199.0": CANCER,
            "285.9": ANEMIA, "280.9": ANEMIA,
            "599.0": UTI,
        }

        # Spanish ICD-10-ES (close to WHO ICD-10 with minor extensions)
        icd10es = {
            "E11.9": self.DIABETES_TYPE2, "E11.65": self.DIABETES_TYPE2,
            "I10": self.HYPERTENSION,
            "I50.9": self.HEART_FAILURE, "I50.20": self.HEART_FAILURE,
            "J44.1": self.COPD, "J44.0": self.COPD,
            "J45.90": self.ASTHMA, "J45.20": self.ASTHMA,
            "N18.9": self.CKD, "N18.6": self.CKD,
            "I21.9": MI, "I21.09": MI,
            "I63.9": STROKE, "I63.50": STROKE,
            "J18.9": PNEUMONIA, "J18.1": PNEUMONIA,
            "F32.9": DEPRESSION, "F33.0": DEPRESSION,
            "E66.9": OBESITY, "E66.01": OBESITY,
            "I48.91": ATRIAL_FIB, "I48.0": ATRIAL_FIB,
            "C80.1": CANCER, "C80.0": CANCER,
            "D64.9": ANEMIA, "D64.89": ANEMIA,
            "N39.0": UTI,
        }

        # Dutch ICD-10-NL (WHO standard with NL extensions)
        icd10nl = {
            "E11.9": self.DIABETES_TYPE2, "E11": self.DIABETES_TYPE2,
            "I10": self.HYPERTENSION,
            "I50.9": self.HEART_FAILURE, "I50": self.HEART_FAILURE,
            "J44.1": self.COPD, "J44": self.COPD,
            "J45.9": self.ASTHMA, "J45": self.ASTHMA,
            "N18.9": self.CKD, "N18": self.CKD,
            "I21.9": MI, "I21": MI,
            "I63.9": STROKE, "I63": STROKE,
            "J18.9": PNEUMONIA, "J18": PNEUMONIA,
            "F32.9": DEPRESSION, "F33": DEPRESSION,
            "E66.9": OBESITY, "E66": OBESITY,
            "I48.9": ATRIAL_FIB, "I48": ATRIAL_FIB,
            "C80.1": CANCER, "C80": CANCER,
            "D64.9": ANEMIA, "D64": ANEMIA,
            "N39.0": UTI,
        }

        self.mapping_tables = {
            "ICD10": icd10_base,
            "ICD10GM": icd10gm,
            "CIM10": cim10,
            "ICD9CM": icd9cm,
            "ICD10ES": icd10es,
            "ICD10NL": icd10nl,
        }

        # Descendant concept map for condition hierarchy queries
        self.descendant_map = {
            self.DIABETES_TYPE2: {self.DIABETES_TYPE2, 443238, 4193704},
            self.HYPERTENSION: {self.HYPERTENSION, 316866},
            self.HEART_FAILURE: {self.HEART_FAILURE, 319835, 443580},
            self.COPD: {self.COPD, 4063381},
            self.ASTHMA: {self.ASTHMA, 4051466},
            self.CKD: {self.CKD, 193782},
            MI: {MI, 312327, 434376},
            STROKE: {STROKE, 443454},
            PNEUMONIA: {PNEUMONIA, 4110056},
            DEPRESSION: {DEPRESSION, 4152280},
            OBESITY: {OBESITY, 4215968},
            ATRIAL_FIB: {ATRIAL_FIB, 4141360},
            CANCER: {CANCER, 4180790},
            ANEMIA: {ANEMIA, 4144746},
            UTI: {UTI, 4112752},
        }

    def get_concept(self, concept_id: int) -> Optional[OMOPConcept]:
        """
        Get concept by ID.

        Args:
            concept_id: OMOP concept ID

        Returns:
            Concept or None if not found
        """
        return self.concept_cache.get(concept_id)

    def get_concept_name(self, concept_id: int) -> str:
        """Get concept name by ID."""
        concept = self.get_concept(concept_id)
        return concept.concept_name if concept else f"Unknown ({concept_id})"

    def map_to_standard(
        self,
        source_code: str,
        source_vocabulary: str
    ) -> Optional[int]:
        """
        Map source code to standard concept ID.

        Args:
            source_code: Source vocabulary code (e.g., ICD10 code)
            source_vocabulary: Source vocabulary name

        Returns:
            Standard concept ID or None
        """
        table = self.mapping_tables.get(source_vocabulary, {})
        mapped = table.get(source_code)
        if mapped is not None:
            return mapped
        # Prefix matching: E11 matches E11.9 and vice versa
        for code, concept_id in table.items():
            if source_code.startswith(code) or code.startswith(source_code):
                return concept_id
        return None

    def get_descendants(
        self,
        ancestor_concept_id: int,
        include_self: bool = True
    ) -> Set[int]:
        """
        Get all descendant concepts of an ancestor.

        Args:
            ancestor_concept_id: Ancestor concept ID
            include_self: Whether to include the ancestor itself

        Returns:
            Set of descendant concept IDs
        """
        descendants = self.descendant_map.get(ancestor_concept_id, set())
        if not descendants and include_self:
            return {ancestor_concept_id}
        if include_self:
            descendants = descendants | {ancestor_concept_id}
        else:
            descendants = descendants - {ancestor_concept_id}
        return descendants

    def get_ancestors(
        self,
        descendant_concept_id: int,
        include_self: bool = True
    ) -> Set[int]:
        """
        Get all ancestor concepts of a descendant.

        Args:
            descendant_concept_id: Descendant concept ID
            include_self: Whether to include the descendant itself

        Returns:
            Set of ancestor concept IDs
        """
        ancestors = set()
        if include_self:
            ancestors.add(descendant_concept_id)
        return ancestors


# =============================================================================
# COHORT BUILDER
# =============================================================================

@dataclass
class CohortCriteria:
    """Criteria for cohort definition."""
    # Inclusion criteria
    required_conditions: List[int] = field(default_factory=list)
    required_drugs: List[int] = field(default_factory=list)
    required_measurements: List[int] = field(default_factory=list)

    # Exclusion criteria
    excluded_conditions: List[int] = field(default_factory=list)
    excluded_drugs: List[int] = field(default_factory=list)

    # Demographics
    min_age: Optional[int] = None
    max_age: Optional[int] = None
    gender_concept_ids: Optional[List[int]] = None

    # Temporal criteria
    observation_period_days: int = 365  # Minimum observation period
    index_date_start: Optional[date] = None
    index_date_end: Optional[date] = None


class OMOPCohortBuilder:
    """
    Build patient cohorts from OMOP CDM data.

    Supports:
    - Phenotype definitions
    - Temporal constraints
    - Federated cohort building (local computation)
    """

    def __init__(self, vocabulary_service: OMOPVocabularyService):
        """
        Initialize cohort builder.

        Args:
            vocabulary_service: Vocabulary service for concept resolution
        """
        self.vocab = vocabulary_service

    def build_cohort(
        self,
        persons: List[OMOPPerson],
        conditions: List[OMOPConditionOccurrence],
        drugs: List[OMOPDrugExposure],
        measurements: List[OMOPMeasurement],
        criteria: CohortCriteria,
        reference_date: Optional[date] = None
    ) -> List[int]:
        """
        Build cohort based on criteria.

        Args:
            persons: List of persons
            conditions: List of conditions
            drugs: List of drug exposures
            measurements: List of measurements
            criteria: Cohort criteria
            reference_date: Reference date for age calculation

        Returns:
            List of person IDs in cohort
        """
        ref_date = reference_date or date.today()
        cohort = []

        # Index data by person
        person_conditions = self._group_by_person(conditions, 'person_id')
        person_drugs = self._group_by_person(drugs, 'person_id')
        person_measurements = self._group_by_person(measurements, 'person_id')

        for person in persons:
            if self._meets_criteria(
                person,
                person_conditions.get(person.person_id, []),
                person_drugs.get(person.person_id, []),
                person_measurements.get(person.person_id, []),
                criteria,
                ref_date
            ):
                cohort.append(person.person_id)

        logger.info(f"Built cohort with {len(cohort)} patients from {len(persons)} candidates")
        return cohort

    def _group_by_person(
        self,
        records: List[Any],
        person_id_attr: str
    ) -> Dict[int, List[Any]]:
        """Group records by person ID."""
        grouped = {}
        for record in records:
            pid = getattr(record, person_id_attr)
            if pid not in grouped:
                grouped[pid] = []
            grouped[pid].append(record)
        return grouped

    def _meets_criteria(
        self,
        person: OMOPPerson,
        conditions: List[OMOPConditionOccurrence],
        drugs: List[OMOPDrugExposure],
        measurements: List[OMOPMeasurement],
        criteria: CohortCriteria,
        ref_date: date
    ) -> bool:
        """Check if person meets cohort criteria."""
        # Age criteria
        age = person.get_age(ref_date)
        if criteria.min_age and age < criteria.min_age:
            return False
        if criteria.max_age and age > criteria.max_age:
            return False

        # Gender criteria
        if criteria.gender_concept_ids:
            if person.gender_concept_id not in criteria.gender_concept_ids:
                return False

        # Required conditions
        person_condition_ids = {c.condition_concept_id for c in conditions}
        for req_cond in criteria.required_conditions:
            # Check if person has condition or any descendant
            descendants = self.vocab.get_descendants(req_cond)
            if not person_condition_ids.intersection(descendants):
                return False

        # Excluded conditions
        for excl_cond in criteria.excluded_conditions:
            descendants = self.vocab.get_descendants(excl_cond)
            if person_condition_ids.intersection(descendants):
                return False

        # Required drugs
        person_drug_ids = {d.drug_concept_id for d in drugs}
        for req_drug in criteria.required_drugs:
            descendants = self.vocab.get_descendants(req_drug)
            if not person_drug_ids.intersection(descendants):
                return False

        # Excluded drugs
        for excl_drug in criteria.excluded_drugs:
            descendants = self.vocab.get_descendants(excl_drug)
            if person_drug_ids.intersection(descendants):
                return False

        return True


# =============================================================================
# FEATURE EXTRACTOR
# =============================================================================

class OMOPFeatureExtractor:
    """
    Extract ML features from OMOP CDM data.

    Provides:
    - Temporal feature aggregation
    - Standardized feature encoding
    - Support for common ML pipelines
    """

    # Standard feature windows (days before index date)
    WINDOW_SHORT = 30
    WINDOW_MEDIUM = 90
    WINDOW_LONG = 365

    def __init__(self, vocabulary_service: OMOPVocabularyService):
        """
        Initialize feature extractor.

        Args:
            vocabulary_service: Vocabulary service
        """
        self.vocab = vocabulary_service

    def extract_features(
        self,
        person: OMOPPerson,
        conditions: List[OMOPConditionOccurrence],
        drugs: List[OMOPDrugExposure],
        measurements: List[OMOPMeasurement],
        visits: List[OMOPVisitOccurrence],
        procedures: List[OMOPProcedureOccurrence],
        index_date: Optional[date] = None
    ) -> Dict[str, Any]:
        """
        Extract comprehensive feature vector for a patient.

        Args:
            person: Patient demographics
            conditions: Condition occurrences
            drugs: Drug exposures
            measurements: Measurements/lab results
            visits: Visit occurrences
            procedures: Procedure occurrences
            index_date: Reference date for feature extraction

        Returns:
            Feature dictionary
        """
        idx_date = index_date or date.today()
        features = {}

        # Demographics
        features.update(self._extract_demographics(person, idx_date))

        # Condition features
        features.update(self._extract_condition_features(conditions, idx_date))

        # Drug features
        features.update(self._extract_drug_features(drugs, idx_date))

        # Measurement features
        features.update(self._extract_measurement_features(measurements, idx_date))

        # Visit features
        features.update(self._extract_visit_features(visits, idx_date))

        # Procedure features
        features.update(self._extract_procedure_features(procedures, idx_date))

        return features

    def _extract_demographics(
        self,
        person: OMOPPerson,
        index_date: date
    ) -> Dict[str, Any]:
        """Extract demographic features."""
        return {
            'age': person.get_age(index_date),
            'gender_male': int(person.gender_concept_id == self.vocab.GENDER_MALE),
            'gender_female': int(person.gender_concept_id == self.vocab.GENDER_FEMALE),
        }

    def _extract_condition_features(
        self,
        conditions: List[OMOPConditionOccurrence],
        index_date: date
    ) -> Dict[str, Any]:
        """Extract condition-based features."""
        features = {
            'n_conditions_30d': 0,
            'n_conditions_90d': 0,
            'n_conditions_365d': 0,
            'n_unique_conditions': 0,
        }

        unique_conditions = set()

        for cond in conditions:
            days_before = (index_date - cond.condition_start_date).days
            if days_before < 0:
                continue  # Future condition

            unique_conditions.add(cond.condition_concept_id)

            if days_before <= 30:
                features['n_conditions_30d'] += 1
            if days_before <= 90:
                features['n_conditions_90d'] += 1
            if days_before <= 365:
                features['n_conditions_365d'] += 1

        features['n_unique_conditions'] = len(unique_conditions)

        # Common condition flags
        common_conditions = {
            'has_diabetes': self.vocab.DIABETES_TYPE2,
            'has_hypertension': self.vocab.HYPERTENSION,
            'has_heart_failure': self.vocab.HEART_FAILURE,
            'has_copd': self.vocab.COPD,
            'has_ckd': self.vocab.CKD,
        }

        for feat_name, concept_id in common_conditions.items():
            descendants = self.vocab.get_descendants(concept_id)
            features[feat_name] = int(
                any(c.condition_concept_id in descendants for c in conditions)
            )

        return features

    def _extract_drug_features(
        self,
        drugs: List[OMOPDrugExposure],
        index_date: date
    ) -> Dict[str, Any]:
        """Extract drug-based features."""
        features = {
            'n_drugs_30d': 0,
            'n_drugs_90d': 0,
            'n_drugs_365d': 0,
            'n_unique_drugs': 0,
            'total_days_supply': 0,
        }

        unique_drugs = set()

        for drug in drugs:
            days_before = (index_date - drug.drug_exposure_start_date).days
            if days_before < 0:
                continue

            unique_drugs.add(drug.drug_concept_id)

            if days_before <= 30:
                features['n_drugs_30d'] += 1
            if days_before <= 90:
                features['n_drugs_90d'] += 1
            if days_before <= 365:
                features['n_drugs_365d'] += 1

            if drug.days_supply:
                features['total_days_supply'] += drug.days_supply

        features['n_unique_drugs'] = len(unique_drugs)

        return features

    def _extract_measurement_features(
        self,
        measurements: List[OMOPMeasurement],
        index_date: date
    ) -> Dict[str, Any]:
        """Extract measurement-based features."""
        features = {
            'n_measurements_30d': 0,
            'n_measurements_90d': 0,
            'n_abnormal_measurements': 0,
        }

        # Track most recent value for key measurements
        key_measurements = {
            self.vocab.GLUCOSE_BLOOD: 'glucose_blood',
            self.vocab.HBA1C: 'hba1c',
            self.vocab.CREATININE_SERUM: 'creatinine',
            self.vocab.HEMOGLOBIN: 'hemoglobin',
            self.vocab.SYSTOLIC_BP: 'systolic_bp',
            self.vocab.DIASTOLIC_BP: 'diastolic_bp',
            self.vocab.BMI: 'bmi',
        }

        most_recent = {}

        for meas in measurements:
            days_before = (index_date - meas.measurement_date).days
            if days_before < 0:
                continue

            if days_before <= 30:
                features['n_measurements_30d'] += 1
            if days_before <= 90:
                features['n_measurements_90d'] += 1

            if meas.is_abnormal():
                features['n_abnormal_measurements'] += 1

            # Track most recent for key measurements
            if meas.measurement_concept_id in key_measurements:
                feat_name = key_measurements[meas.measurement_concept_id]
                if (feat_name not in most_recent or
                    meas.measurement_date > most_recent[feat_name][0]):
                    most_recent[feat_name] = (meas.measurement_date, meas.value_as_number)

        # Add most recent values
        for feat_name, (_, value) in most_recent.items():
            if value is not None:
                features[f'last_{feat_name}'] = value

        return features

    def _extract_visit_features(
        self,
        visits: List[OMOPVisitOccurrence],
        index_date: date
    ) -> Dict[str, Any]:
        """Extract visit-based features."""
        features = {
            'n_visits_30d': 0,
            'n_visits_90d': 0,
            'n_visits_365d': 0,
            'n_inpatient_visits': 0,
            'n_er_visits': 0,
            'total_los_days': 0,
        }

        for visit in visits:
            days_before = (index_date - visit.visit_start_date).days
            if days_before < 0:
                continue

            if days_before <= 30:
                features['n_visits_30d'] += 1
            if days_before <= 90:
                features['n_visits_90d'] += 1
            if days_before <= 365:
                features['n_visits_365d'] += 1

            if visit.visit_concept_id == OMOPVisitOccurrence.INPATIENT_VISIT:
                features['n_inpatient_visits'] += 1
                los = visit.length_of_stay()
                if los:
                    features['total_los_days'] += los

            elif visit.visit_concept_id == OMOPVisitOccurrence.EMERGENCY_VISIT:
                features['n_er_visits'] += 1

        return features

    def _extract_procedure_features(
        self,
        procedures: List[OMOPProcedureOccurrence],
        index_date: date
    ) -> Dict[str, Any]:
        """Extract procedure-based features."""
        features = {
            'n_procedures_30d': 0,
            'n_procedures_90d': 0,
            'n_procedures_365d': 0,
        }

        for proc in procedures:
            days_before = (index_date - proc.procedure_date).days
            if days_before < 0:
                continue

            if days_before <= 30:
                features['n_procedures_30d'] += 1
            if days_before <= 90:
                features['n_procedures_90d'] += 1
            if days_before <= 365:
                features['n_procedures_365d'] += 1

        return features


# =============================================================================
# FEDERATED QUERY INTERFACE
# =============================================================================

@dataclass
class FederatedQueryResult:
    """Result from a federated query."""
    site_id: str
    query_id: str
    result_count: int
    aggregate_values: Dict[str, float]
    execution_time_ms: int
    privacy_budget_used: float = 0.0


class OMOPFederatedQuery:
    """
    Privacy-preserving distributed queries over OMOP CDM.

    Supports:
    - Count queries with differential privacy
    - Aggregate statistics
    - Cohort counts without sharing individual data
    """

    def __init__(
        self,
        site_id: str,
        vocabulary_service: OMOPVocabularyService,
        epsilon: float = 1.0
    ):
        """
        Initialize federated query interface.

        Args:
            site_id: Local site identifier
            vocabulary_service: Vocabulary service
            epsilon: Differential privacy budget
        """
        self.site_id = site_id
        self.vocab = vocabulary_service
        self.epsilon = epsilon
        self.privacy_budget_spent = 0.0

    def execute_count_query(
        self,
        cohort_criteria: CohortCriteria,
        persons: List[OMOPPerson],
        conditions: List[OMOPConditionOccurrence],
        drugs: List[OMOPDrugExposure],
        measurements: List[OMOPMeasurement],
        add_noise: bool = True
    ) -> FederatedQueryResult:
        """
        Execute a privacy-preserving count query.

        Args:
            cohort_criteria: Criteria for cohort
            persons: Local person data
            conditions: Local condition data
            drugs: Local drug data
            measurements: Local measurement data
            add_noise: Whether to add DP noise

        Returns:
            Query result with noisy count
        """
        import time
        start_time = time.time()

        # Build cohort locally
        builder = OMOPCohortBuilder(self.vocab)
        cohort = builder.build_cohort(
            persons, conditions, drugs, measurements,
            cohort_criteria
        )

        true_count = len(cohort)

        # Add Laplace noise for differential privacy
        if add_noise:
            sensitivity = 1  # Each person changes count by at most 1
            scale = sensitivity / self.epsilon
            noise = np.random.laplace(0, scale)
            noisy_count = max(0, int(true_count + noise))
            self.privacy_budget_spent += self.epsilon
        else:
            noisy_count = true_count

        execution_time = int((time.time() - start_time) * 1000)

        return FederatedQueryResult(
            site_id=self.site_id,
            query_id=f"count_{hash(str(cohort_criteria))}",
            result_count=noisy_count,
            aggregate_values={},
            execution_time_ms=execution_time,
            privacy_budget_used=self.epsilon if add_noise else 0
        )

    def execute_aggregate_query(
        self,
        persons: List[OMOPPerson],
        measurements: List[OMOPMeasurement],
        measurement_concept_id: int,
        aggregation: str = "mean"
    ) -> FederatedQueryResult:
        """
        Execute aggregate query over measurements.

        Args:
            persons: Local persons
            measurements: Local measurements
            measurement_concept_id: Target measurement concept
            aggregation: Aggregation type (mean, median, min, max)

        Returns:
            Query result with aggregate value
        """
        import time
        start_time = time.time()

        # Filter relevant measurements
        values = [
            m.value_as_number
            for m in measurements
            if m.measurement_concept_id == measurement_concept_id
            and m.value_as_number is not None
        ]

        if not values:
            agg_value = 0.0
        elif aggregation == "mean":
            agg_value = np.mean(values)
        elif aggregation == "median":
            agg_value = np.median(values)
        elif aggregation == "min":
            agg_value = np.min(values)
        elif aggregation == "max":
            agg_value = np.max(values)
        else:
            agg_value = np.mean(values)

        # Add noise (for mean, use Gaussian mechanism)
        sensitivity = 100  # Assume bounded range
        noise_scale = sensitivity / self.epsilon
        noisy_value = agg_value + np.random.normal(0, noise_scale)

        execution_time = int((time.time() - start_time) * 1000)

        return FederatedQueryResult(
            site_id=self.site_id,
            query_id=f"agg_{measurement_concept_id}_{aggregation}",
            result_count=len(values),
            aggregate_values={aggregation: noisy_value},
            execution_time_ms=execution_time,
            privacy_budget_used=self.epsilon
        )


# =============================================================================
# OMOP TO ML DATASET CONVERTER
# =============================================================================

class OMOPDataset:
    """
    Convert OMOP CDM data to ML-ready format.

    Handles:
    - Feature matrix construction
    - Label extraction
    - Train/test splitting
    - Normalization
    """

    def __init__(
        self,
        vocabulary_service: OMOPVocabularyService,
        feature_extractor: OMOPFeatureExtractor
    ):
        """
        Initialize dataset converter.

        Args:
            vocabulary_service: Vocabulary service
            feature_extractor: Feature extractor
        """
        self.vocab = vocabulary_service
        self.extractor = feature_extractor
        self.feature_names: List[str] = []
        self.feature_matrix: Optional[np.ndarray] = None
        self.labels: Optional[np.ndarray] = None
        self.person_ids: List[int] = []

    def build_from_omop(
        self,
        persons: List[OMOPPerson],
        conditions: Dict[int, List[OMOPConditionOccurrence]],
        drugs: Dict[int, List[OMOPDrugExposure]],
        measurements: Dict[int, List[OMOPMeasurement]],
        visits: Dict[int, List[OMOPVisitOccurrence]],
        procedures: Dict[int, List[OMOPProcedureOccurrence]],
        label_fn: Optional[callable] = None,
        index_date: Optional[date] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Build feature matrix from OMOP data.

        Args:
            persons: List of persons
            conditions: Conditions indexed by person_id
            drugs: Drugs indexed by person_id
            measurements: Measurements indexed by person_id
            visits: Visits indexed by person_id
            procedures: Procedures indexed by person_id
            label_fn: Function to compute labels (person_id -> label)
            index_date: Reference date

        Returns:
            Tuple of (feature_matrix, labels)
        """
        all_features = []
        all_labels = []

        for person in persons:
            pid = person.person_id

            features = self.extractor.extract_features(
                person,
                conditions.get(pid, []),
                drugs.get(pid, []),
                measurements.get(pid, []),
                visits.get(pid, []),
                procedures.get(pid, []),
                index_date
            )

            all_features.append(features)
            self.person_ids.append(pid)

            if label_fn:
                all_labels.append(label_fn(pid))

        # Determine feature names from first patient
        if all_features:
            self.feature_names = sorted(all_features[0].keys())

        # Build matrix
        n_patients = len(all_features)
        n_features = len(self.feature_names)
        self.feature_matrix = np.zeros((n_patients, n_features))

        for i, feat_dict in enumerate(all_features):
            for j, name in enumerate(self.feature_names):
                val = feat_dict.get(name, 0)
                if isinstance(val, bool):
                    val = int(val)
                self.feature_matrix[i, j] = val

        if all_labels:
            self.labels = np.array(all_labels)

        return self.feature_matrix, self.labels

    def get_feature_names(self) -> List[str]:
        """Get list of feature names."""
        return self.feature_names

    def normalize(self, method: str = 'zscore') -> np.ndarray:
        """
        Normalize features.

        Args:
            method: Normalization method ('zscore', 'minmax')

        Returns:
            Normalized feature matrix
        """
        if self.feature_matrix is None:
            raise ValueError("Feature matrix not built yet")

        if method == 'zscore':
            mean = self.feature_matrix.mean(axis=0)
            std = self.feature_matrix.std(axis=0)
            std[std == 0] = 1  # Avoid division by zero
            return (self.feature_matrix - mean) / std

        elif method == 'minmax':
            min_val = self.feature_matrix.min(axis=0)
            max_val = self.feature_matrix.max(axis=0)
            range_val = max_val - min_val
            range_val[range_val == 0] = 1
            return (self.feature_matrix - min_val) / range_val

        return self.feature_matrix


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_omop_integration() -> Dict[str, Any]:
    """
    Create OMOP CDM integration components.

    Returns:
        Dictionary with vocabulary service, cohort builder,
        feature extractor, and dataset converter
    """
    vocab = OMOPVocabularyService()
    cohort_builder = OMOPCohortBuilder(vocab)
    extractor = OMOPFeatureExtractor(vocab)
    dataset = OMOPDataset(vocab, extractor)

    return {
        'vocabulary_service': vocab,
        'cohort_builder': cohort_builder,
        'feature_extractor': extractor,
        'dataset': dataset
    }


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("FL-EHDS: OMOP CDM Support Demo")
    print("=" * 60)

    # Create sample OMOP data
    sample_person = OMOPPerson(
        person_id=1001,
        gender_concept_id=8532,  # Female
        year_of_birth=1965,
        month_of_birth=6,
        day_of_birth=15
    )

    sample_condition = OMOPConditionOccurrence(
        condition_occurrence_id=5001,
        person_id=1001,
        condition_concept_id=201826,  # Type 2 diabetes
        condition_start_date=date(2020, 3, 15)
    )

    sample_measurement = OMOPMeasurement(
        measurement_id=8001,
        person_id=1001,
        measurement_concept_id=3004501,  # Glucose
        measurement_date=date(2024, 1, 10),
        value_as_number=126.5,
        unit_concept_id=8840,
        range_low=70,
        range_high=100
    )

    sample_visit = OMOPVisitOccurrence(
        visit_occurrence_id=2001,
        person_id=1001,
        visit_concept_id=9201,  # Inpatient
        visit_start_date=date(2024, 1, 5),
        visit_end_date=date(2024, 1, 8)
    )

    # Initialize services
    vocab = OMOPVocabularyService()
    extractor = OMOPFeatureExtractor(vocab)

    print("\n1. Person Demographics:")
    print("-" * 40)
    print(f"   Person ID: {sample_person.person_id}")
    print(f"   Gender: {vocab.get_concept_name(sample_person.gender_concept_id)}")
    print(f"   Age: {sample_person.get_age()}")

    print("\n2. Condition:")
    print("-" * 40)
    print(f"   Condition: {vocab.get_concept_name(sample_condition.condition_concept_id)}")
    print(f"   Start Date: {sample_condition.condition_start_date}")

    print("\n3. Measurement:")
    print("-" * 40)
    print(f"   Measurement: {vocab.get_concept_name(sample_measurement.measurement_concept_id)}")
    print(f"   Value: {sample_measurement.value_as_number}")
    print(f"   Abnormal: {sample_measurement.is_abnormal()}")

    print("\n4. Visit:")
    print("-" * 40)
    print(f"   Visit Type: {vocab.get_concept_name(sample_visit.visit_concept_id)}")
    print(f"   LOS: {sample_visit.length_of_stay()} days")

    print("\n5. Feature Extraction:")
    print("-" * 40)
    features = extractor.extract_features(
        sample_person,
        [sample_condition],
        [],  # drugs
        [sample_measurement],
        [sample_visit],
        []   # procedures
    )
    for k, v in sorted(features.items()):
        print(f"   {k}: {v}")

    print("\n6. Cohort Building:")
    print("-" * 40)
    criteria = CohortCriteria(
        required_conditions=[201826],  # Diabetes
        min_age=50,
        max_age=80
    )
    cohort_builder = OMOPCohortBuilder(vocab)
    cohort = cohort_builder.build_cohort(
        [sample_person],
        [sample_condition],
        [],
        [sample_measurement],
        criteria
    )
    print(f"   Cohort size: {len(cohort)}")
    print(f"   Patient IDs: {cohort}")

    print("\n" + "=" * 60)
    print("OMOP CDM Integration ready for FL-EHDS!")
    print("=" * 60)

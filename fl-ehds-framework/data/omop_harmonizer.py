#!/usr/bin/env python3
"""
FL-EHDS OMOP-CDM Harmonization Layer

Provides vocabulary harmonization for cross-border federated learning:
1. Generates synthetic patients with LOCAL coding systems (ICD-10-GM, CIM-10, ICD-9-CM, etc.)
2. Converts local codes to standard OMOP concept IDs via OMOPVocabularyService
3. Builds OMOP CDM tables (Person, ConditionOccurrence, Measurement, DrugExposure, etc.)
4. Extracts ~36 standardized features via OMOPFeatureExtractor
5. Quantifies vocabulary heterogeneity before/after harmonization (Jaccard, JSD)

This module bridges the gap between heterogeneous European coding systems
and the FL training pipeline, demonstrating that OMOP-CDM can serve as
a harmonization layer for EHDS cross-border scenarios.

Author: Fabio Liberti
"""

import logging
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from core.omop_cdm import (
    OMOPConditionOccurrence,
    OMOPDataset,
    OMOPDeath,
    OMOPDrugExposure,
    OMOPFeatureExtractor,
    OMOPMeasurement,
    OMOPPerson,
    OMOPProcedureOccurrence,
    OMOPVisitOccurrence,
    OMOPVocabularyService,
)
from data.fhir_loader import SyntheticFHIRLoader

logger = logging.getLogger(__name__)


# =============================================================================
# COUNTRY VOCABULARY PROFILES
# =============================================================================

COUNTRY_VOCABULARY_PROFILES = {
    "DE": {
        "name": "Germany",
        "coding_system": "ICD10GM",
        "medication_system": "ATC",
        "description": "ICD-10-GM (German Modification)",
    },
    "FR": {
        "name": "France",
        "coding_system": "CIM10",
        "medication_system": "ATC",
        "description": "CIM-10 (Classification Internationale des Maladies)",
    },
    "IT": {
        "name": "Italy",
        "coding_system": "ICD9CM",
        "medication_system": "ATC",
        "description": "ICD-9-CM (legacy, transitioning to ICD-10)",
    },
    "ES": {
        "name": "Spain",
        "coding_system": "ICD10ES",
        "medication_system": "ATC",
        "description": "ICD-10-ES (Spanish edition)",
    },
    "NL": {
        "name": "Netherlands",
        "coding_system": "ICD10NL",
        "medication_system": "ATC",
        "description": "ICD-10-NL (Dutch edition)",
    },
}

# Local condition codes per vocabulary (condition_key -> local_code)
LOCAL_CONDITION_CODES = {
    "ICD10GM": {
        "diabetes": "E11.90", "hypertension": "I10.00",
        "heart_failure": "I50.90", "copd": "J44.10",
        "asthma": "J45.90", "ckd": "N18.90",
        "mi": "I21.90", "stroke": "I63.90",
        "pneumonia": "J18.90", "depression": "F32.9",
        "obesity": "E66.00", "atrial_fib": "I48.90",
        "cancer": "C80.0", "anemia": "D64.9", "uti": "N39.0",
    },
    "CIM10": {
        "diabetes": "E119", "hypertension": "I10",
        "heart_failure": "I509", "copd": "J441",
        "asthma": "J459", "ckd": "N189",
        "mi": "I219", "stroke": "I639",
        "pneumonia": "J189", "depression": "F329",
        "obesity": "E669", "atrial_fib": "I489",
        "cancer": "C800", "anemia": "D649", "uti": "N390",
    },
    "ICD9CM": {
        "diabetes": "250.00", "hypertension": "401.9",
        "heart_failure": "428.0", "copd": "496",
        "asthma": "493.90", "ckd": "585.9",
        "mi": "410.9", "stroke": "436",
        "pneumonia": "486", "depression": "311",
        "obesity": "278.00", "atrial_fib": "427.31",
        "cancer": "199.1", "anemia": "285.9", "uti": "599.0",
    },
    "ICD10ES": {
        "diabetes": "E11.9", "hypertension": "I10",
        "heart_failure": "I50.9", "copd": "J44.1",
        "asthma": "J45.90", "ckd": "N18.9",
        "mi": "I21.9", "stroke": "I63.9",
        "pneumonia": "J18.9", "depression": "F32.9",
        "obesity": "E66.9", "atrial_fib": "I48.91",
        "cancer": "C80.1", "anemia": "D64.9", "uti": "N39.0",
    },
    "ICD10NL": {
        "diabetes": "E11.9", "hypertension": "I10",
        "heart_failure": "I50.9", "copd": "J44.1",
        "asthma": "J45.9", "ckd": "N18.9",
        "mi": "I21.9", "stroke": "I63.9",
        "pneumonia": "J18.9", "depression": "F32.9",
        "obesity": "E66.9", "atrial_fib": "I48.9",
        "cancer": "C80.1", "anemia": "D64.9", "uti": "N39.0",
    },
}

# Drug concept IDs (simplified ATC mapping)
DRUG_CONCEPTS = {
    "metformin": 1503297,
    "lisinopril": 1308216,
    "furosemide": 956874,
    "salbutamol": 1154343,
    "omeprazole": 948078,
    "aspirin": 1112807,
    "atorvastatin": 1545958,
    "amlodipine": 1332418,
    "paracetamol": 1125315,
    "amoxicillin": 1713332,
}

# Condition -> likely drug associations
CONDITION_DRUG_MAP = {
    "diabetes": ["metformin"],
    "hypertension": ["lisinopril", "amlodipine"],
    "heart_failure": ["furosemide", "lisinopril"],
    "copd": ["salbutamol"],
    "asthma": ["salbutamol"],
    "mi": ["aspirin", "atorvastatin"],
    "depression": ["omeprazole"],
    "cancer": ["paracetamol"],
    "pneumonia": ["amoxicillin"],
}


# =============================================================================
# SYNTHETIC OMOP LOADER
# =============================================================================

class SyntheticOMOPLoader:
    """
    Generate synthetic patient data with local coding systems,
    then convert to OMOP CDM tables with harmonized concept IDs.

    Reuses SyntheticFHIRLoader for clinical value generation (age, vitals, etc.)
    and adds local diagnosis codes based on country vocabulary.
    """

    # Condition probabilities by hospital profile
    CONDITION_PROBS = {
        "general": {
            "diabetes": 0.15, "hypertension": 0.25, "heart_failure": 0.08,
            "copd": 0.10, "asthma": 0.07, "ckd": 0.05, "mi": 0.04,
            "stroke": 0.03, "pneumonia": 0.06, "depression": 0.12,
            "obesity": 0.18, "atrial_fib": 0.05, "cancer": 0.04,
            "anemia": 0.08, "uti": 0.06,
        },
        "cardiac": {
            "diabetes": 0.25, "hypertension": 0.45, "heart_failure": 0.30,
            "copd": 0.15, "asthma": 0.05, "ckd": 0.12, "mi": 0.20,
            "stroke": 0.08, "pneumonia": 0.04, "depression": 0.10,
            "obesity": 0.22, "atrial_fib": 0.25, "cancer": 0.03,
            "anemia": 0.10, "uti": 0.03,
        },
        "pediatric": {
            "diabetes": 0.03, "hypertension": 0.02, "heart_failure": 0.01,
            "copd": 0.01, "asthma": 0.15, "ckd": 0.01, "mi": 0.001,
            "stroke": 0.001, "pneumonia": 0.10, "depression": 0.05,
            "obesity": 0.08, "atrial_fib": 0.001, "cancer": 0.02,
            "anemia": 0.06, "uti": 0.08,
        },
        "geriatric": {
            "diabetes": 0.30, "hypertension": 0.50, "heart_failure": 0.20,
            "copd": 0.20, "asthma": 0.08, "ckd": 0.18, "mi": 0.10,
            "stroke": 0.12, "pneumonia": 0.12, "depression": 0.18,
            "obesity": 0.15, "atrial_fib": 0.18, "cancer": 0.08,
            "anemia": 0.15, "uti": 0.12,
        },
        "oncology": {
            "diabetes": 0.18, "hypertension": 0.28, "heart_failure": 0.10,
            "copd": 0.12, "asthma": 0.05, "ckd": 0.08, "mi": 0.05,
            "stroke": 0.04, "pneumonia": 0.08, "depression": 0.22,
            "obesity": 0.14, "atrial_fib": 0.06, "cancer": 0.60,
            "anemia": 0.25, "uti": 0.07,
        },
    }

    def __init__(
        self,
        num_patients: int = 500,
        hospital_profile: str = "general",
        country_code: str = "DE",
        seed: int = 42,
    ):
        self.num_patients = num_patients
        self.hospital_profile = hospital_profile
        self.country_code = country_code
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        self.vocab = OMOPVocabularyService()
        self.vocab_profile = COUNTRY_VOCABULARY_PROFILES[country_code]
        self.coding_system = self.vocab_profile["coding_system"]
        self.local_codes = LOCAL_CONDITION_CODES[self.coding_system]
        self.condition_probs = self.CONDITION_PROBS.get(
            hospital_profile, self.CONDITION_PROBS["general"]
        )

        # Use SyntheticFHIRLoader for clinical value generation
        self.fhir_loader = SyntheticFHIRLoader(
            num_patients=num_patients,
            random_seed=seed,
            hospital_profile=hospital_profile,
        )

        # Storage for raw and harmonized code sets
        self.raw_codes: Set[str] = set()
        self.omop_concept_ids: Set[int] = set()

    def generate(self) -> Dict[str, Any]:
        """
        Generate synthetic patient data with local codes and OMOP conversion.

        Returns:
            Dict with OMOP CDM tables:
                persons: List[OMOPPerson]
                conditions: Dict[person_id, List[OMOPConditionOccurrence]]
                drugs: Dict[person_id, List[OMOPDrugExposure]]
                measurements: Dict[person_id, List[OMOPMeasurement]]
                visits: Dict[person_id, List[OMOPVisitOccurrence]]
                procedures: Dict[person_id, List[OMOPProcedureOccurrence]]
                deaths: Dict[person_id, OMOPDeath]
                labels: Dict[person_id, int]  (mortality label)
        """
        # Generate base clinical records via FHIR loader
        fhir_records = self.fhir_loader.load_patients()

        persons = []
        conditions = {}
        drugs = {}
        measurements = {}
        visits = {}
        procedures = {}
        deaths = {}
        labels = {}

        ref_date = date.today()
        cond_id_counter = 1
        drug_id_counter = 1
        meas_id_counter = 1
        visit_id_counter = 1
        proc_id_counter = 1

        for i, record in enumerate(fhir_records):
            pid = i + 1

            # --- Person ---
            age = max(0, int(record.age or 50))
            birth_year = ref_date.year - age
            gender_id = (
                self.vocab.GENDER_MALE
                if record.gender == "male"
                else self.vocab.GENDER_FEMALE
            )
            person = OMOPPerson(
                person_id=pid,
                gender_concept_id=gender_id,
                year_of_birth=birth_year,
                month_of_birth=self.rng.randint(1, 13),
                day_of_birth=self.rng.randint(1, 29),
            )
            persons.append(person)

            # --- Conditions (local codes -> OMOP) ---
            patient_conditions = []
            patient_condition_keys = []
            for cond_key, prob in self.condition_probs.items():
                if self.rng.random() < prob:
                    local_code = self.local_codes[cond_key]
                    self.raw_codes.add(local_code)

                    # Map to OMOP standard concept ID
                    omop_id = self.vocab.map_to_standard(
                        local_code, self.coding_system
                    )
                    if omop_id is not None:
                        self.omop_concept_ids.add(omop_id)
                        days_ago = self.rng.randint(1, 730)
                        cond = OMOPConditionOccurrence(
                            condition_occurrence_id=cond_id_counter,
                            person_id=pid,
                            condition_concept_id=omop_id,
                            condition_start_date=ref_date - timedelta(days=days_ago),
                            condition_source_value=local_code,
                            condition_source_concept_id=0,
                        )
                        patient_conditions.append(cond)
                        patient_condition_keys.append(cond_key)
                        cond_id_counter += 1

            conditions[pid] = patient_conditions

            # --- Drugs (based on conditions) ---
            patient_drugs = []
            prescribed_drugs = set()
            for cond_key in patient_condition_keys:
                for drug_name in CONDITION_DRUG_MAP.get(cond_key, []):
                    if drug_name not in prescribed_drugs:
                        prescribed_drugs.add(drug_name)
                        drug_concept = DRUG_CONCEPTS.get(drug_name, 0)
                        if drug_concept:
                            days_ago = self.rng.randint(1, 365)
                            supply = self.rng.randint(7, 90)
                            drug = OMOPDrugExposure(
                                drug_exposure_id=drug_id_counter,
                                person_id=pid,
                                drug_concept_id=drug_concept,
                                drug_exposure_start_date=(
                                    ref_date - timedelta(days=days_ago)
                                ),
                                drug_exposure_end_date=(
                                    ref_date - timedelta(days=max(0, days_ago - supply))
                                ),
                                days_supply=supply,
                                drug_source_value=drug_name,
                            )
                            patient_drugs.append(drug)
                            drug_id_counter += 1
            drugs[pid] = patient_drugs

            # --- Measurements (from clinical values) ---
            patient_measurements = []
            meas_map = [
                (self.vocab.GLUCOSE_BLOOD, record.glucose, 70, 100),
                (self.vocab.SYSTOLIC_BP, record.systolic_bp, 90, 140),
                (self.vocab.DIASTOLIC_BP, record.diastolic_bp, 60, 90),
                (self.vocab.HEART_RATE, record.heart_rate, 60, 100),
                (self.vocab.BMI, record.bmi, 18.5, 30),
                (self.vocab.HEMOGLOBIN, record.hemoglobin or self.rng.normal(14, 2), 12, 17),
                (self.vocab.CREATININE_SERUM, record.creatinine or self.rng.normal(1.0, 0.3), 0.6, 1.2),
                (self.vocab.HBA1C, self.rng.normal(5.7 + (1.5 if "diabetes" in patient_condition_keys else 0), 0.5), 4.0, 5.7),
            ]
            for concept_id, value, low, high in meas_map:
                if value is not None:
                    # Add 2-3 measurements at different times
                    for t in range(self.rng.randint(1, 4)):
                        days_ago = self.rng.randint(1, 365)
                        noise = self.rng.normal(0, abs(value) * 0.05)
                        m = OMOPMeasurement(
                            measurement_id=meas_id_counter,
                            person_id=pid,
                            measurement_concept_id=concept_id,
                            measurement_date=ref_date - timedelta(days=days_ago),
                            value_as_number=float(value + noise),
                            range_low=low,
                            range_high=high,
                        )
                        patient_measurements.append(m)
                        meas_id_counter += 1
            measurements[pid] = patient_measurements

            # --- Visits ---
            patient_visits = []
            n_visits = self.rng.poisson(3)
            for v in range(n_visits):
                days_ago = self.rng.randint(1, 365)
                visit_type = self.rng.choice(
                    [OMOPVisitOccurrence.INPATIENT_VISIT,
                     OMOPVisitOccurrence.OUTPATIENT_VISIT,
                     OMOPVisitOccurrence.EMERGENCY_VISIT],
                    p=[0.3, 0.5, 0.2],
                )
                start = ref_date - timedelta(days=days_ago)
                los = self.rng.randint(1, 8) if visit_type == OMOPVisitOccurrence.INPATIENT_VISIT else 0
                visit = OMOPVisitOccurrence(
                    visit_occurrence_id=visit_id_counter,
                    person_id=pid,
                    visit_concept_id=int(visit_type),
                    visit_start_date=start,
                    visit_end_date=start + timedelta(days=los),
                )
                patient_visits.append(visit)
                visit_id_counter += 1
            visits[pid] = patient_visits

            # --- Procedures (correlated with visits) ---
            patient_procedures = []
            for visit in patient_visits:
                if self.rng.random() < 0.4:
                    proc = OMOPProcedureOccurrence(
                        procedure_occurrence_id=proc_id_counter,
                        person_id=pid,
                        procedure_concept_id=self.rng.choice([4107731, 4230359, 4181530]),
                        procedure_date=visit.visit_start_date,
                    )
                    patient_procedures.append(proc)
                    proc_id_counter += 1
            procedures[pid] = patient_procedures

            # --- Mortality label ---
            mortality = 1 if record.outcome_30day_mortality else 0
            labels[pid] = mortality

            if mortality and self.rng.random() < 0.7:
                deaths[pid] = OMOPDeath(
                    person_id=pid,
                    death_date=ref_date - timedelta(days=self.rng.randint(1, 30)),
                )

        return {
            "persons": persons,
            "conditions": conditions,
            "drugs": drugs,
            "measurements": measurements,
            "visits": visits,
            "procedures": procedures,
            "deaths": deaths,
            "labels": labels,
        }

    def get_raw_code_set(self) -> Set[str]:
        """Return the set of raw local codes used (before harmonization)."""
        return self.raw_codes.copy()

    def get_omop_concept_set(self) -> Set[int]:
        """Return the set of OMOP concept IDs (after harmonization)."""
        return self.omop_concept_ids.copy()


# =============================================================================
# VOCABULARY HETEROGENEITY METRICS
# =============================================================================

class VocabularyHeterogeneityMetrics:
    """
    Quantify vocabulary heterogeneity across hospitals/countries.

    Measures how different the coding systems are before and after
    OMOP standardization, justifying scientifically why cross-border
    data is non-IID even when describing the same pathologies.
    """

    @staticmethod
    def pairwise_jaccard(sets: Dict[int, Set]) -> np.ndarray:
        """
        Compute NxN Jaccard distance matrix between code sets.

        Jaccard distance = 1 - |A intersection B| / |A union B|
        Lower distance = more similar vocabularies.
        """
        ids = sorted(sets.keys())
        n = len(ids)
        matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                a = sets[ids[i]]
                b = sets[ids[j]]
                if len(a) == 0 and len(b) == 0:
                    matrix[i, j] = 0.0
                else:
                    intersection = len(a & b)
                    union = len(a | b)
                    matrix[i, j] = 1.0 - (intersection / union) if union > 0 else 0.0
        return matrix

    @staticmethod
    def jensen_shannon_divergence(distributions: Dict[int, Dict[str, float]]) -> float:
        """
        Compute average pairwise Jensen-Shannon Divergence of code frequency
        distributions across hospitals.

        JSD is bounded [0, 1] (with log base 2) and symmetric.
        """
        ids = sorted(distributions.keys())
        if len(ids) < 2:
            return 0.0

        # Build common vocabulary across all hospitals
        all_codes = set()
        for dist in distributions.values():
            all_codes.update(dist.keys())
        all_codes = sorted(all_codes)

        # Convert to probability vectors
        vectors = {}
        for hid, dist in distributions.items():
            total = sum(dist.values()) or 1.0
            vec = np.array([dist.get(c, 0) / total for c in all_codes])
            # Add small epsilon to avoid log(0)
            vec = vec + 1e-10
            vec = vec / vec.sum()
            vectors[hid] = vec

        # Average pairwise JSD
        jsd_sum = 0.0
        count = 0
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                p = vectors[ids[i]]
                q = vectors[ids[j]]
                m = 0.5 * (p + q)
                jsd = 0.5 * (
                    np.sum(p * np.log2(p / m)) + np.sum(q * np.log2(q / m))
                )
                jsd_sum += jsd
                count += 1

        return jsd_sum / count if count > 0 else 0.0

    @staticmethod
    def compute_report(
        raw_sets: Dict[int, Set[str]],
        omop_sets: Dict[int, Set[int]],
        raw_distributions: Optional[Dict[int, Dict[str, float]]] = None,
        omop_distributions: Optional[Dict[int, Dict[int, float]]] = None,
        country_assignments: Optional[Dict[int, str]] = None,
    ) -> Dict[str, Any]:
        """
        Full heterogeneity report: before/after Jaccard, JSD, overlap.

        Returns:
            Dict with metrics and per-hospital details.
        """
        metrics = VocabularyHeterogeneityMetrics

        # Jaccard distances
        raw_jaccard = metrics.pairwise_jaccard(raw_sets)
        omop_jaccard = metrics.pairwise_jaccard(omop_sets)

        # Extract upper triangle (pairwise comparisons)
        n = raw_jaccard.shape[0]
        raw_distances = []
        omop_distances = []
        for i in range(n):
            for j in range(i + 1, n):
                raw_distances.append(raw_jaccard[i, j])
                omop_distances.append(omop_jaccard[i, j])

        raw_jaccard_mean = float(np.mean(raw_distances)) if raw_distances else 0.0
        omop_jaccard_mean = float(np.mean(omop_distances)) if omop_distances else 0.0

        # JSD
        raw_jsd = 0.0
        omop_jsd = 0.0
        if raw_distributions:
            # Convert keys to strings for raw distributions
            str_dists = {
                k: {str(code): count for code, count in v.items()}
                for k, v in raw_distributions.items()
            }
            raw_jsd = metrics.jensen_shannon_divergence(str_dists)
        if omop_distributions:
            str_dists = {
                k: {str(code): count for code, count in v.items()}
                for k, v in omop_distributions.items()
            }
            omop_jsd = metrics.jensen_shannon_divergence(str_dists)

        # Reduction
        jaccard_reduction = (
            (1.0 - omop_jaccard_mean / raw_jaccard_mean) * 100
            if raw_jaccard_mean > 0
            else 0.0
        )

        report = {
            "raw_jaccard_mean": raw_jaccard_mean,
            "omop_jaccard_mean": omop_jaccard_mean,
            "jaccard_reduction_pct": jaccard_reduction,
            "raw_jsd": raw_jsd,
            "omop_jsd": omop_jsd,
            "raw_jaccard_matrix": raw_jaccard.tolist(),
            "omop_jaccard_matrix": omop_jaccard.tolist(),
            "num_hospitals": n,
        }

        if country_assignments:
            report["per_hospital_vocabularies"] = {
                hid: {
                    "country": cc,
                    "coding_system": COUNTRY_VOCABULARY_PROFILES[cc]["coding_system"],
                    "description": COUNTRY_VOCABULARY_PROFILES[cc]["description"],
                    "n_raw_codes": len(raw_sets.get(hid, set())),
                    "n_omop_concepts": len(omop_sets.get(hid, set())),
                }
                for hid, cc in country_assignments.items()
            }

        return report


# =============================================================================
# BRIDGE FUNCTION: load_omop_data()
# =============================================================================

DEFAULT_HOSPITAL_PROFILES = ["general", "cardiac", "pediatric", "geriatric", "oncology"]
DEFAULT_COUNTRY_CODES = ["DE", "FR", "IT", "ES", "NL"]


def load_omop_data(
    num_clients: int = 5,
    samples_per_client: int = 500,
    hospital_profiles: Optional[List[str]] = None,
    country_codes: Optional[List[str]] = None,
    label_name: str = "mortality_30day",
    test_split: float = 0.2,
    seed: int = 42,
) -> Tuple[
    Dict[int, Tuple[np.ndarray, np.ndarray]],
    Dict[int, Tuple[np.ndarray, np.ndarray]],
    Dict[str, Any],
]:
    """
    Load OMOP-harmonized data for federated learning.

    Each client represents a hospital in a different EU country using
    its own local coding system. Data is harmonized through OMOP-CDM
    vocabulary mapping, producing ~36 standardized features.

    Args:
        num_clients: Number of FL clients (hospitals)
        samples_per_client: Patients per hospital
        hospital_profiles: Hospital profile per client (cycles if fewer)
        country_codes: Country code per client (cycles if fewer)
        label_name: Target label (mortality_30day)
        test_split: Fraction of data for testing (per client)
        seed: Random seed

    Returns:
        (client_train_data, client_test_data, metadata)
        where each data dict maps client_id -> (X: np.ndarray, y: np.ndarray)
    """
    profiles = hospital_profiles or DEFAULT_HOSPITAL_PROFILES
    countries = country_codes or DEFAULT_COUNTRY_CODES

    assigned_profiles = [profiles[i % len(profiles)] for i in range(num_clients)]
    assigned_countries = [countries[i % len(countries)] for i in range(num_clients)]

    # Per-hospital data accumulators
    raw_code_sets: Dict[int, Set[str]] = {}
    omop_code_sets: Dict[int, Set[int]] = {}
    raw_distributions: Dict[int, Dict[str, float]] = {}
    omop_distributions: Dict[int, Dict[int, float]] = {}
    country_assignments: Dict[int, str] = {}

    client_train_data: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    client_test_data: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

    all_feature_names: Optional[List[str]] = None

    rng = np.random.RandomState(seed)

    for client_id in range(num_clients):
        profile = assigned_profiles[client_id]
        country = assigned_countries[client_id]
        client_seed = seed + client_id * 100

        logger.info(
            f"Client {client_id}: {country} ({profile}) - "
            f"{COUNTRY_VOCABULARY_PROFILES[country]['coding_system']}"
        )

        # Generate patient data with local codes
        loader = SyntheticOMOPLoader(
            num_patients=samples_per_client,
            hospital_profile=profile,
            country_code=country,
            seed=client_seed,
        )
        omop_data = loader.generate()

        # Track code sets for heterogeneity analysis
        raw_code_sets[client_id] = loader.get_raw_code_set()
        omop_code_sets[client_id] = loader.get_omop_concept_set()
        country_assignments[client_id] = country

        # Build frequency distributions for JSD
        raw_freq: Dict[str, float] = {}
        for conds in [omop_data["conditions"].get(pid, []) for pid in range(1, samples_per_client + 1)]:
            for c in conds:
                code = c.condition_source_value or ""
                raw_freq[code] = raw_freq.get(code, 0) + 1.0
        raw_distributions[client_id] = raw_freq

        omop_freq: Dict[int, float] = {}
        for conds in [omop_data["conditions"].get(pid, []) for pid in range(1, samples_per_client + 1)]:
            for c in conds:
                omop_freq[c.condition_concept_id] = (
                    omop_freq.get(c.condition_concept_id, 0) + 1.0
                )
        omop_distributions[client_id] = omop_freq

        # Extract features via OMOP infrastructure
        vocab = OMOPVocabularyService()
        extractor = OMOPFeatureExtractor(vocab)
        dataset = OMOPDataset(vocab, extractor)

        X, y_raw = dataset.build_from_omop(
            persons=omop_data["persons"],
            conditions=omop_data["conditions"],
            drugs=omop_data["drugs"],
            measurements=omop_data["measurements"],
            visits=omop_data["visits"],
            procedures=omop_data["procedures"],
            label_fn=lambda pid: omop_data["labels"].get(pid, 0),
        )

        # Capture feature names from first client
        if all_feature_names is None:
            all_feature_names = dataset.get_feature_names()

        # Normalize (zscore per client)
        X_norm = dataset.normalize(method="zscore")

        y = y_raw.astype(np.int64) if y_raw is not None else np.zeros(len(X), dtype=np.int64)

        # Train/test split
        n = len(X_norm)
        n_test = max(1, int(n * test_split))
        perm = rng.permutation(n)
        train_idx = perm[:-n_test]
        test_idx = perm[-n_test:]

        client_train_data[client_id] = (
            X_norm[train_idx].astype(np.float32),
            y[train_idx],
        )
        client_test_data[client_id] = (
            X_norm[test_idx].astype(np.float32),
            y[test_idx],
        )

    # Compute heterogeneity report
    het_report = VocabularyHeterogeneityMetrics.compute_report(
        raw_sets=raw_code_sets,
        omop_sets=omop_code_sets,
        raw_distributions=raw_distributions,
        omop_distributions=omop_distributions,
        country_assignments=country_assignments,
    )

    metadata = {
        "feature_names": all_feature_names or [],
        "num_features": len(all_feature_names) if all_feature_names else 0,
        "label_name": label_name,
        "profiles_assigned": assigned_profiles,
        "countries_assigned": assigned_countries,
        "samples_per_client": samples_per_client,
        "source": "omop_harmonizer",
        "test_split": test_split,
        "heterogeneity_report": het_report,
        "coding_systems": {
            i: COUNTRY_VOCABULARY_PROFILES[c]["coding_system"]
            for i, c in enumerate(assigned_countries)
        },
    }

    # Log summary
    logger.info(
        f"OMOP harmonization complete: {num_clients} clients, "
        f"{len(all_feature_names) if all_feature_names else 0} features"
    )
    logger.info(
        f"Jaccard distance: raw={het_report['raw_jaccard_mean']:.3f} -> "
        f"OMOP={het_report['omop_jaccard_mean']:.3f} "
        f"(reduction: {het_report['jaccard_reduction_pct']:.1f}%)"
    )

    return client_train_data, client_test_data, metadata

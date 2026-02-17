"""
Synthea FHIR R4 Dataset Loader for Federated Learning.

Loads and preprocesses Synthea synthetic FHIR R4 patient bundles,
extracting tabular features from native FHIR JSON for federated
learning experiments. This is a FHIR-native dataset — no CSV
conversion needed.

Each patient bundle contains real FHIR resources:
    Patient, Condition, Observation, Encounter, MedicationRequest,
    Procedure, DiagnosticReport, Immunization, CarePlan, etc.

The loader extracts per-patient tabular features:
    - Demographics: age, gender
    - Vitals: BMI (from Observation), number of conditions
    - Conditions: presence of top chronic conditions (diabetes, hypertension, etc.)
    - Utilization: encounter count, medication count

Target: presence of a cardiovascular condition (hypertension, CHD, stroke, etc.)

FHIR R4 mapping:
    - ALL features extracted directly from FHIR resources (native)

EHDS compatibility:
    - FHIR R4 native format (Art. 46 interoperability)
    - Synthetic data eliminates privacy concerns (Art. 33 pseudonymization)
    - Realistic US clinical profiles for cross-Atlantic benchmarking
    - SNOMED-CT and LOINC coded observations

Reference: Walonoski et al., "Synthea: An approach, method, and
    software for generating synthetic patients and the synthetic
    electronic health care record", JAMIA, 2018.

Author: Fabio Liberti
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Chronic conditions to detect (SNOMED-CT display names, case-insensitive)
CARDIOVASCULAR_CONDITIONS = [
    "hypertension", "coronary heart disease", "myocardial infarction",
    "atrial fibrillation", "stroke", "heart failure",
    "ischemic heart disease", "cardiac arrest",
]

CHRONIC_CONDITIONS = [
    "diabetes", "asthma", "chronic obstructive pulmonary",
    "chronic kidney", "alzheimer", "cancer", "obesity",
]

FEATURE_NAMES = [
    "age", "gender", "num_conditions", "num_encounters",
    "num_medications", "num_procedures",
    "has_diabetes", "has_hypertension", "has_copd",
    "has_asthma", "has_ckd", "has_obesity",
    "has_cancer", "has_alzheimer",
]

# Normalization ranges
FEATURE_RANGES = {
    "age": (0, 100),
    "num_conditions": (0, 30),
    "num_encounters": (0, 100),
    "num_medications": (0, 50),
    "num_procedures": (0, 50),
}


def load_synthea_fhir_data(
    num_clients: int = 5,
    is_iid: bool = False,
    alpha: float = 0.5,
    test_split: float = 0.2,
    seed: int = 42,
    max_patients: Optional[int] = None,
    data_path: Optional[str] = None,
) -> Tuple[Dict[int, Tuple[np.ndarray, np.ndarray]],
           Dict[int, Tuple[np.ndarray, np.ndarray]],
           Dict[str, Any]]:
    """Load Synthea FHIR R4 patient bundles for federated learning.

    Args:
        num_clients: Number of FL clients.
        is_iid: IID partitioning if True, Dirichlet non-IID if False.
        alpha: Dirichlet alpha for non-IID partitioning.
        test_split: Fraction for test set per client.
        seed: Random seed.
        max_patients: If set, limit to this many patient bundles.
        data_path: Path to FHIR bundles directory. Auto-discovers if None.

    Returns:
        (client_train_data, client_test_data, metadata)
    """
    rng = np.random.RandomState(seed)

    if data_path is None:
        base = Path(__file__).parent
        candidates = [
            base / "synthea_fhir" / "fhir",
            base / "synthea_fhir" / "output" / "fhir",
            base / "synthea" / "fhir",
        ]
        for p in candidates:
            if p.exists() and p.is_dir():
                data_path = str(p)
                break
        if data_path is None:
            raise FileNotFoundError(
                f"Synthea FHIR bundles not found. Searched: {[str(c) for c in candidates]}"
            )

    data_dir = Path(data_path)
    bundle_files = sorted(data_dir.glob("*.json"))

    if max_patients is not None:
        bundle_files = bundle_files[:max_patients]

    logger.info("Loading Synthea FHIR bundles from %s (%d files)", data_path, len(bundle_files))

    X_all = []
    y_all = []
    skipped = 0

    for bundle_path in bundle_files:
        try:
            features, label = _extract_patient_features(bundle_path)
            if features is not None:
                X_all.append(features)
                y_all.append(label)
            else:
                skipped += 1
        except (json.JSONDecodeError, KeyError, TypeError):
            skipped += 1
            continue

    if skipped > 0:
        logger.info("Skipped %d bundles (parse errors or incomplete)", skipped)

    X_all = np.array(X_all, dtype=np.float32)
    y_all = np.array(y_all, dtype=np.int64)

    logger.info("Preprocessed: %d patients, %d features, cvd_rate=%.1f%%",
                len(y_all), X_all.shape[1], y_all.mean() * 100)

    # Partition
    if is_iid:
        client_train, client_test = _partition_iid(X_all, y_all, num_clients, test_split, rng)
    else:
        client_train, client_test = _partition_dirichlet(X_all, y_all, num_clients, alpha, test_split, rng)

    # Metadata
    num_classes = 2
    class_counts = np.bincount(y_all, minlength=num_classes)

    metadata = {
        "dataset_name": "Synthea FHIR R4 (Synthetic Patients)",
        "source": "synthea_fhir_loader",
        "total_samples": len(y_all),
        "num_features": X_all.shape[1],
        "feature_names": FEATURE_NAMES,
        "num_classes": num_classes,
        "class_names": {0: "No CVD", 1: "Cardiovascular condition"},
        "class_distribution": {int(k): int(v) for k, v in enumerate(class_counts)},
        "label_name": "cardiovascular_condition",
        "partition_method": "iid" if is_iid else f"dirichlet(alpha={alpha})",
        "test_split": test_split,
        "fhir_native": True,
        "fhir_mapping": {
            "Patient": ["age", "gender"],
            "Condition": ["has_diabetes", "has_hypertension", "has_copd",
                          "has_asthma", "has_ckd", "has_obesity",
                          "has_cancer", "has_alzheimer", "num_conditions"],
            "Encounter": ["num_encounters"],
            "MedicationRequest": ["num_medications"],
            "Procedure": ["num_procedures"],
        },
        "ehds_compliance": {
            "coding_system": "SNOMED-CT, LOINC, RxNorm (FHIR R4 native)",
            "pseudonymized": True,
            "synthetic": True,
            "fhir_native": True,
            "multi_center": False,
            "data_categories": ["ehr", "conditions", "medications", "encounters"],
        },
        "samples_per_client": {
            cid: len(client_train[cid][1]) + len(client_test[cid][1])
            for cid in client_train
        },
    }

    return client_train, client_test, metadata


def _extract_patient_features(bundle_path: Path) -> Tuple[Optional[List[float]], Optional[int]]:
    """Extract tabular features from a single Synthea FHIR bundle."""
    with open(bundle_path, "r", encoding="utf-8") as f:
        bundle = json.load(f)

    entries = bundle.get("entry", [])
    if not entries:
        return None, None

    # Categorize resources
    patient_resource = None
    conditions = []
    encounters = []
    medications = []
    procedures = []

    for entry in entries:
        resource = entry.get("resource", {})
        rt = resource.get("resourceType", "")
        if rt == "Patient":
            patient_resource = resource
        elif rt == "Condition":
            conditions.append(resource)
        elif rt == "Encounter":
            encounters.append(resource)
        elif rt == "MedicationRequest":
            medications.append(resource)
        elif rt == "Procedure":
            procedures.append(resource)

    if patient_resource is None:
        return None, None

    # 1. Age
    birth_date = patient_resource.get("birthDate", "")
    try:
        birth_year = int(birth_date[:4])
        age = 2019 - birth_year  # Synthea sep2019 dataset
    except (ValueError, IndexError):
        age = 50
    lo, hi = FEATURE_RANGES["age"]
    age_norm = max(0.0, min(1.0, (age - lo) / (hi - lo + 1e-8)))

    # 2. Gender
    gender = patient_resource.get("gender", "unknown")
    gender_val = 1.0 if gender == "male" else 0.0

    # 3. Condition analysis
    condition_texts = []
    for cond in conditions:
        code_obj = cond.get("code", {})
        for coding in code_obj.get("coding", []):
            display = coding.get("display", "").lower()
            condition_texts.append(display)
        text = code_obj.get("text", "").lower()
        if text:
            condition_texts.append(text)

    num_conditions = len(conditions)
    lo, hi = FEATURE_RANGES["num_conditions"]
    num_cond_norm = max(0.0, min(1.0, (num_conditions - lo) / (hi - lo + 1e-8)))

    # 4. Encounter count
    num_encounters = len(encounters)
    lo, hi = FEATURE_RANGES["num_encounters"]
    num_enc_norm = max(0.0, min(1.0, (num_encounters - lo) / (hi - lo + 1e-8)))

    # 5. Medication count
    num_meds = len(medications)
    lo, hi = FEATURE_RANGES["num_medications"]
    num_med_norm = max(0.0, min(1.0, (num_meds - lo) / (hi - lo + 1e-8)))

    # 6. Procedure count
    num_procs = len(procedures)
    lo, hi = FEATURE_RANGES["num_procedures"]
    num_proc_norm = max(0.0, min(1.0, (num_procs - lo) / (hi - lo + 1e-8)))

    # 7. Chronic condition flags
    all_text = " ".join(condition_texts)
    has_diabetes = 1.0 if "diabetes" in all_text else 0.0
    has_hypertension = 1.0 if "hypertension" in all_text else 0.0
    has_copd = 1.0 if "chronic obstructive" in all_text or "copd" in all_text else 0.0
    has_asthma = 1.0 if "asthma" in all_text else 0.0
    has_ckd = 1.0 if "chronic kidney" in all_text else 0.0
    has_obesity = 1.0 if "obesity" in all_text or "body mass index 30" in all_text else 0.0
    has_cancer = 1.0 if "cancer" in all_text or "neoplasm" in all_text or "carcinoma" in all_text else 0.0
    has_alzheimer = 1.0 if "alzheimer" in all_text else 0.0

    # Target: cardiovascular condition present
    has_cvd = 0
    for kw in CARDIOVASCULAR_CONDITIONS:
        if kw in all_text:
            has_cvd = 1
            break

    features = [
        age_norm, gender_val, num_cond_norm, num_enc_norm,
        num_med_norm, num_proc_norm,
        has_diabetes, has_hypertension, has_copd,
        has_asthma, has_ckd, has_obesity,
        has_cancer, has_alzheimer,
    ]

    return features, has_cvd


def _partition_iid(X, y, num_clients, test_split, rng):
    indices = np.arange(len(X))
    rng.shuffle(indices)
    splits = np.array_split(indices, num_clients)
    client_train, client_test = {}, {}
    for cid, split_idx in enumerate(splits):
        rng.shuffle(split_idx)
        n_test = max(1, int(len(split_idx) * test_split))
        client_train[cid] = (X[split_idx[n_test:]], y[split_idx[n_test:]])
        client_test[cid] = (X[split_idx[:n_test]], y[split_idx[:n_test]])
    return client_train, client_test


def _partition_dirichlet(X, y, num_clients, alpha, test_split, rng):
    num_classes = len(np.unique(y))
    client_indices = {i: [] for i in range(num_clients)}
    for c in range(num_classes):
        class_idx = np.where(y == c)[0]
        rng.shuffle(class_idx)
        proportions = rng.dirichlet([alpha] * num_clients)
        proportions = (proportions * len(class_idx)).astype(int)
        proportions[0] += len(class_idx) - proportions.sum()
        start = 0
        for cid in range(num_clients):
            end = start + proportions[cid]
            client_indices[cid].extend(class_idx[start:end].tolist())
            start = end
    client_train, client_test = {}, {}
    for cid in range(num_clients):
        indices = np.array(client_indices[cid])
        rng.shuffle(indices)
        n_test = max(1, int(len(indices) * test_split))
        client_train[cid] = (X[indices[n_test:]], y[indices[n_test:]])
        client_test[cid] = (X[indices[:n_test]], y[indices[:n_test]])
    return client_train, client_test

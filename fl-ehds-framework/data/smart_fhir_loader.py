"""
SMART Bulk FHIR Dataset Loader for Federated Learning.

Loads and preprocesses the SMART on FHIR Bulk Data sample dataset
(NDJSON format) for federated learning experiments. Extracts tabular
features from Patient, Condition, Observation, and Encounter FHIR
resources.

The dataset contains 120 patients with full clinical records in
FHIR R4 NDJSON Bulk Data format — the standard format for EHDS
secondary use data access (Art. 46).

FHIR R4 mapping:
    - ALL features extracted directly from FHIR NDJSON resources
    - Patient -> demographics (age, gender)
    - Condition -> chronic disease flags
    - Observation -> vital signs, lab values
    - Encounter -> healthcare utilization

EHDS compatibility:
    - FHIR R4 NDJSON Bulk Data format (Art. 46 standard)
    - Demonstrates cross-border data exchange capability
    - SMART on FHIR authorization model (Art. 50 access control)
    - SNOMED-CT and LOINC coded resources

Reference: SMART Health IT, "Sample Bulk FHIR Datasets",
    github.com/smart-on-fhir/sample-bulk-fhir-datasets.

Author: Fabio Liberti
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

FEATURE_NAMES = [
    "age", "gender", "num_conditions", "num_encounters",
    "num_medications", "num_observations",
    "has_diabetes", "has_hypertension", "has_copd",
    "has_heart_disease", "has_ckd", "has_cancer",
]

FEATURE_RANGES = {
    "age": (0, 100),
    "num_conditions": (0, 30),
    "num_encounters": (0, 100),
    "num_medications": (0, 50),
    "num_observations": (0, 200),
}

CARDIOVASCULAR_KEYWORDS = [
    "hypertension", "coronary", "myocardial", "atrial fibrillation",
    "stroke", "heart failure", "ischemic heart", "cardiac",
]


def load_smart_fhir_data(
    num_clients: int = 4,
    is_iid: bool = False,
    alpha: float = 0.5,
    test_split: float = 0.2,
    seed: int = 42,
    data_path: Optional[str] = None,
) -> Tuple[Dict[int, Tuple[np.ndarray, np.ndarray]],
           Dict[int, Tuple[np.ndarray, np.ndarray]],
           Dict[str, Any]]:
    """Load SMART Bulk FHIR dataset for federated learning.

    Args:
        num_clients: Number of FL clients.
        is_iid: IID partitioning if True, Dirichlet non-IID if False.
        alpha: Dirichlet alpha for non-IID partitioning.
        test_split: Fraction for test set per client.
        seed: Random seed.
        data_path: Path to NDJSON directory. Auto-discovers if None.

    Returns:
        (client_train_data, client_test_data, metadata)
    """
    rng = np.random.RandomState(seed)

    if data_path is None:
        base = Path(__file__).parent
        candidates = [
            base / "smart_bulk_fhir",
            base / "smart_fhir",
        ]
        for p in candidates:
            if p.exists() and (p / "Patient.000.ndjson").exists():
                data_path = str(p)
                break
        if data_path is None:
            raise FileNotFoundError(
                f"SMART Bulk FHIR dataset not found. Searched: {[str(c) for c in candidates]}"
            )

    data_dir = Path(data_path)
    logger.info("Loading SMART Bulk FHIR dataset from %s", data_path)

    # Load all resources indexed by patient ID
    patients = _load_ndjson(data_dir, "Patient")
    conditions = _load_ndjson_by_subject(data_dir, "Condition")
    encounters = _load_ndjson_by_subject(data_dir, "Encounter")
    medications = _load_ndjson_by_subject(data_dir, "MedicationRequest")
    observations = _load_ndjson_by_subject(data_dir, "Observation")

    logger.info("Loaded %d patients, %d conditions, %d encounters",
                len(patients), sum(len(v) for v in conditions.values()),
                sum(len(v) for v in encounters.values()))

    X_all = []
    y_all = []

    for patient in patients:
        pid = patient.get("id", "")
        if not pid:
            continue

        features, label = _extract_features(
            patient, conditions.get(pid, []),
            encounters.get(pid, []), medications.get(pid, []),
            observations.get(pid, [])
        )
        X_all.append(features)
        y_all.append(label)

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
        "dataset_name": "SMART Bulk FHIR (NDJSON)",
        "source": "smart_fhir_loader",
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
        "bulk_data_format": "NDJSON",
        "fhir_mapping": {
            "Patient": ["age", "gender"],
            "Condition": ["has_diabetes", "has_hypertension", "has_copd",
                          "has_heart_disease", "has_ckd", "has_cancer",
                          "num_conditions"],
            "Encounter": ["num_encounters"],
            "MedicationRequest": ["num_medications"],
            "Observation": ["num_observations"],
        },
        "ehds_compliance": {
            "coding_system": "SNOMED-CT, LOINC (FHIR R4 NDJSON)",
            "pseudonymized": True,
            "fhir_native": True,
            "bulk_data": True,
            "multi_center": False,
            "data_categories": ["ehr", "conditions", "medications", "encounters"],
        },
        "samples_per_client": {
            cid: len(client_train[cid][1]) + len(client_test[cid][1])
            for cid in client_train
        },
    }

    return client_train, client_test, metadata


def _load_ndjson(data_dir: Path, resource_type: str) -> List[dict]:
    """Load all NDJSON files for a resource type."""
    resources = []
    for ndjson_file in sorted(data_dir.glob(f"{resource_type}.*.ndjson")):
        with open(ndjson_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        resources.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    return resources


def _load_ndjson_by_subject(data_dir: Path, resource_type: str) -> Dict[str, List[dict]]:
    """Load NDJSON resources indexed by subject (patient) reference."""
    by_patient = {}
    resources = _load_ndjson(data_dir, resource_type)
    for res in resources:
        # Extract patient reference
        subject = res.get("subject", res.get("patient", {}))
        if isinstance(subject, dict):
            ref = subject.get("reference", "")
        else:
            ref = ""
        # Reference format: "Patient/uuid"
        pid = ref.split("/")[-1] if "/" in ref else ref
        if pid:
            by_patient.setdefault(pid, []).append(res)
    return by_patient


def _extract_features(patient, conditions, encounters, medications, observations):
    """Extract tabular features from FHIR resources for one patient."""
    # 1. Age
    birth_date = patient.get("birthDate", "")
    try:
        birth_year = int(birth_date[:4])
        age = 2020 - birth_year
    except (ValueError, IndexError):
        age = 50
    lo, hi = FEATURE_RANGES["age"]
    age_norm = max(0.0, min(1.0, (age - lo) / (hi - lo + 1e-8)))

    # 2. Gender
    gender = patient.get("gender", "unknown")
    gender_val = 1.0 if gender == "male" else 0.0

    # 3. Counts
    num_cond = len(conditions)
    lo, hi = FEATURE_RANGES["num_conditions"]
    num_cond_norm = max(0.0, min(1.0, (num_cond - lo) / (hi - lo + 1e-8)))

    num_enc = len(encounters)
    lo, hi = FEATURE_RANGES["num_encounters"]
    num_enc_norm = max(0.0, min(1.0, (num_enc - lo) / (hi - lo + 1e-8)))

    num_med = len(medications)
    lo, hi = FEATURE_RANGES["num_medications"]
    num_med_norm = max(0.0, min(1.0, (num_med - lo) / (hi - lo + 1e-8)))

    num_obs = len(observations)
    lo, hi = FEATURE_RANGES["num_observations"]
    num_obs_norm = max(0.0, min(1.0, (num_obs - lo) / (hi - lo + 1e-8)))

    # 4. Condition flags
    condition_texts = []
    for cond in conditions:
        code_obj = cond.get("code", {})
        for coding in code_obj.get("coding", []):
            display = coding.get("display", "").lower()
            condition_texts.append(display)
        text = code_obj.get("text", "").lower()
        if text:
            condition_texts.append(text)

    all_text = " ".join(condition_texts)
    has_diabetes = 1.0 if "diabetes" in all_text else 0.0
    has_hypertension = 1.0 if "hypertension" in all_text else 0.0
    has_copd = 1.0 if "chronic obstructive" in all_text or "copd" in all_text else 0.0
    has_heart = 1.0 if any(kw in all_text for kw in ["coronary", "heart failure", "cardiac"]) else 0.0
    has_ckd = 1.0 if "chronic kidney" in all_text else 0.0
    has_cancer = 1.0 if "cancer" in all_text or "neoplasm" in all_text or "carcinoma" in all_text else 0.0

    # Target: cardiovascular condition
    has_cvd = 0
    for kw in CARDIOVASCULAR_KEYWORDS:
        if kw in all_text:
            has_cvd = 1
            break

    features = [
        age_norm, gender_val, num_cond_norm, num_enc_norm,
        num_med_norm, num_obs_norm,
        has_diabetes, has_hypertension, has_copd,
        has_heart, has_ckd, has_cancer,
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

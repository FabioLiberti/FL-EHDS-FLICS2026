"""
Chronic Kidney Disease (UCI) Dataset Loader for Federated Learning.

Loads and preprocesses the UCI Chronic Kidney Disease dataset
for federated learning experiments. 400 patients with 24 clinical
features and binary CKD classification.

FHIR R4 mapping:
    - age -> Patient.birthDate (derived)
    - bp -> Observation (vitals: blood pressure)
    - sg, al, su -> Observation (urinalysis)
    - rbc, pc, pcc, ba -> Observation (urine microscopy)
    - bgr, bu, sc, sod, pot -> Observation (blood chemistry)
    - hemo, pcv, wc, rc -> Observation (hematology / CBC)
    - htn, dm, cad -> Condition (comorbidities)
    - appet, pe, ane -> Observation (clinical signs)
    - class -> Condition (chronic kidney disease)

EHDS compatibility:
    - Standard clinical laboratory features (LOINC-mappable)
    - Multi-parameter renal panel (creatinine, urea, electrolytes)
    - Comorbidity coding (hypertension, diabetes, CAD)
    - Small dataset (400) tests FL on limited-data scenarios

Reference: L. Chronic_Kidney_Disease, UCI ML Repository ID 336.
    Soundarapandian et al., Apollo Hospitals, India.

Author: Fabio Liberti
"""

import csv
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Column definitions
COLUMNS = [
    "id", "age", "bp", "sg", "al", "su", "rbc", "pc", "pcc", "ba",
    "bgr", "bu", "sc", "sod", "pot", "hemo", "pcv", "wc", "rc",
    "htn", "dm", "cad", "appet", "pe", "ane", "class",
]

# Feature names after preprocessing (24 features)
FEATURE_NAMES = [
    "age", "bp", "sg", "al", "su",
    "rbc", "pc", "pcc", "ba",
    "bgr", "bu", "sc", "sod", "pot",
    "hemo", "pcv", "wc", "rc",
    "htn", "dm", "cad", "appet", "pe", "ane",
]

# Numeric features with clinical normalization ranges
NUMERIC_RANGES = {
    "age": (2, 90),
    "bp": (50, 180),
    "sg": (1.005, 1.025),
    "al": (0, 5),
    "su": (0, 5),
    "bgr": (22, 490),
    "bu": (1.5, 391),
    "sc": (0.4, 76),
    "sod": (4.5, 163),
    "pot": (2.5, 47),
    "hemo": (3.1, 17.8),
    "pcv": (9, 54),
    "wc": (2200, 26400),
    "rc": (2.1, 8),
}

# Binary/categorical features
BINARY_MAP = {
    "rbc": {"normal": 1.0, "abnormal": 0.0},
    "pc": {"normal": 1.0, "abnormal": 0.0},
    "pcc": {"present": 1.0, "notpresent": 0.0},
    "ba": {"present": 1.0, "notpresent": 0.0},
    "htn": {"yes": 1.0, "no": 0.0},
    "dm": {"yes": 1.0, "no": 0.0},
    "cad": {"yes": 1.0, "no": 0.0},
    "appet": {"good": 1.0, "poor": 0.0},
    "pe": {"yes": 1.0, "no": 0.0},
    "ane": {"yes": 1.0, "no": 0.0},
}

# Median values for imputation (from literature)
MEDIANS = {
    "age": 51.0, "bp": 70.0, "sg": 1.020, "al": 0.0, "su": 0.0,
    "bgr": 121.0, "bu": 36.0, "sc": 1.2, "sod": 138.0, "pot": 4.6,
    "hemo": 12.5, "pcv": 38.0, "wc": 8000.0, "rc": 4.7,
}


def load_ckd_data(
    num_clients: int = 4,
    is_iid: bool = False,
    alpha: float = 0.5,
    test_split: float = 0.2,
    seed: int = 42,
    data_path: Optional[str] = None,
) -> Tuple[Dict[int, Tuple[np.ndarray, np.ndarray]],
           Dict[int, Tuple[np.ndarray, np.ndarray]],
           Dict[str, Any]]:
    """Load Chronic Kidney Disease dataset for federated learning.

    Args:
        num_clients: Number of FL clients.
        is_iid: IID partitioning if True, Dirichlet non-IID if False.
        alpha: Dirichlet alpha for non-IID partitioning.
        test_split: Fraction for test set per client.
        seed: Random seed.
        data_path: Path to CSV. Auto-discovers if None.

    Returns:
        (client_train_data, client_test_data, metadata)
    """
    rng = np.random.RandomState(seed)

    if data_path is None:
        base = Path(__file__).parent
        candidates = [
            base / "chronic_kidney_disease" / "kidney_disease.csv",
            base / "ckd" / "kidney_disease.csv",
        ]
        for p in candidates:
            if p.exists():
                data_path = str(p)
                break
        if data_path is None:
            raise FileNotFoundError(
                f"CKD dataset not found. Searched: {[str(c) for c in candidates]}"
            )

    logger.info("Loading Chronic Kidney Disease dataset from %s", data_path)

    X_all = []
    y_all = []

    with open(data_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Target: class (ckd / notckd) - handle whitespace
            target_str = row.get("class", "").strip().replace("\t", "")
            if target_str not in ("ckd", "notckd"):
                continue
            label = 1 if target_str == "ckd" else 0

            features = []
            valid = True

            for feat_name in FEATURE_NAMES:
                raw = row.get(feat_name, "?").strip().replace("\t", "")

                if feat_name in NUMERIC_RANGES:
                    # Numeric feature
                    if raw in ("?", "", "nan"):
                        val = MEDIANS.get(feat_name, 0.0)
                    else:
                        try:
                            val = float(raw)
                        except ValueError:
                            val = MEDIANS.get(feat_name, 0.0)
                    lo, hi = NUMERIC_RANGES[feat_name]
                    normalized = max(0, min(1, (val - lo) / (hi - lo + 1e-8)))
                    features.append(normalized)
                elif feat_name in BINARY_MAP:
                    # Binary/categorical feature
                    mapping = BINARY_MAP[feat_name]
                    if raw in ("?", "", "nan"):
                        features.append(0.5)  # unknown -> midpoint
                    else:
                        features.append(mapping.get(raw.lower(), 0.5))
                else:
                    valid = False
                    break

            if not valid:
                continue

            X_all.append(features)
            y_all.append(label)

    X_all = np.array(X_all, dtype=np.float32)
    y_all = np.array(y_all, dtype=np.int64)

    logger.info("Preprocessed: %d samples, %d features, pos_rate=%.1f%% (CKD)",
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
        "dataset_name": "Chronic Kidney Disease (UCI)",
        "source": "ckd_loader",
        "total_samples": len(y_all),
        "num_features": X_all.shape[1],
        "feature_names": FEATURE_NAMES,
        "num_classes": num_classes,
        "class_names": {0: "Not CKD", 1: "Chronic Kidney Disease"},
        "class_distribution": {int(k): int(v) for k, v in enumerate(class_counts)},
        "label_name": "ckd",
        "partition_method": "iid" if is_iid else f"dirichlet(alpha={alpha})",
        "test_split": test_split,
        "fhir_mapping": {
            "Patient": ["age"],
            "Observation_vitals": ["bp"],
            "Observation_urinalysis": ["sg", "al", "su", "rbc", "pc", "pcc", "ba"],
            "Observation_chemistry": ["bgr", "bu", "sc", "sod", "pot"],
            "Observation_hematology": ["hemo", "pcv", "wc", "rc"],
            "Condition": ["htn", "dm", "cad", "ckd"],
            "Observation_clinical": ["appet", "pe", "ane"],
        },
        "ehds_compliance": {
            "coding_system": "Clinical laboratory values (LOINC-mappable)",
            "pseudonymized": True,
            "multi_center": False,
            "data_categories": ["lab_results", "conditions", "vitals"],
        },
        "samples_per_client": {
            cid: len(client_train[cid][1]) + len(client_test[cid][1])
            for cid in client_train
        },
    }

    return client_train, client_test, metadata


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

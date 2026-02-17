"""
Stroke Prediction Dataset Loader for Federated Learning.

Loads and preprocesses the Kaggle Stroke Prediction dataset
for federated learning experiments. The dataset contains 5,110
patients with 11 clinical features and binary stroke outcome.

FHIR R4 mapping:
    - id -> Patient.identifier
    - gender, age -> Patient demographics
    - hypertension, heart_disease -> Condition.code
    - avg_glucose_level, bmi -> Observation (lab/vitals)
    - smoking_status -> Observation (social history)
    - ever_married, work_type, Residence_type -> Patient extensions
    - stroke -> Condition (cerebrovascular event)

EHDS compatibility:
    - Standard clinical features map to FHIR Observation/Condition
    - Demographics + risk factors enable cross-border analytics
    - Severe class imbalance (~5% positive) reflects real stroke
      epidemiology in European populations

Reference: fedesoriano, "Stroke Prediction Dataset", Kaggle, 2021.
    (Confidential source - used for research purposes)

Author: Fabio Liberti
"""

import csv
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Feature names after preprocessing (10 numeric features)
FEATURE_NAMES = [
    "age",
    "gender",
    "hypertension",
    "heart_disease",
    "ever_married",
    "work_type",
    "residence_type",
    "avg_glucose_level",
    "bmi",
    "smoking_status",
]

# Encoding maps
GENDER_MAP = {"Male": 1, "Female": 0, "Other": 0.5}
WORK_TYPE_MAP = {
    "Private": 0.0, "Self-employed": 0.25, "Govt_job": 0.5,
    "children": 0.75, "Never_worked": 1.0,
}
SMOKING_MAP = {
    "never smoked": 0.0, "formerly smoked": 0.33,
    "smokes": 0.67, "Unknown": 0.5,
}

# Normalization ranges (clinical)
FEATURE_RANGES = {
    "age": (0, 100),
    "avg_glucose_level": (50, 300),
    "bmi": (10, 60),
}


def load_stroke_data(
    num_clients: int = 5,
    is_iid: bool = False,
    alpha: float = 0.5,
    test_split: float = 0.2,
    seed: int = 42,
    data_path: Optional[str] = None,
) -> Tuple[Dict[int, Tuple[np.ndarray, np.ndarray]],
           Dict[int, Tuple[np.ndarray, np.ndarray]],
           Dict[str, Any]]:
    """Load Stroke Prediction dataset for federated learning.

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
            base / "stroke_prediction" / "healthcare-dataset-stroke-data.csv",
            base / "stroke" / "healthcare-dataset-stroke-data.csv",
        ]
        for p in candidates:
            if p.exists():
                data_path = str(p)
                break
        if data_path is None:
            raise FileNotFoundError(
                f"Stroke dataset not found. Searched: {[str(c) for c in candidates]}"
            )

    logger.info("Loading Stroke Prediction dataset from %s", data_path)

    # Read and preprocess
    X_all = []
    y_all = []

    with open(data_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            features = []

            # 1. Age (0-100)
            try:
                age = float(row.get("age", 50))
            except ValueError:
                age = 50.0
            lo, hi = FEATURE_RANGES["age"]
            features.append(max(0, min(1, (age - lo) / (hi - lo))))

            # 2. Gender
            features.append(GENDER_MAP.get(row.get("gender", "Female"), 0))

            # 3. Hypertension (binary)
            features.append(float(row.get("hypertension", 0)))

            # 4. Heart disease (binary)
            features.append(float(row.get("heart_disease", 0)))

            # 5. Ever married (binary)
            features.append(1.0 if row.get("ever_married", "No") == "Yes" else 0.0)

            # 6. Work type (ordinal encoded)
            features.append(WORK_TYPE_MAP.get(row.get("work_type", "Private"), 0.0))

            # 7. Residence type (binary)
            features.append(1.0 if row.get("Residence_type", "Urban") == "Urban" else 0.0)

            # 8. Average glucose level
            try:
                glucose = float(row.get("avg_glucose_level", 100))
            except ValueError:
                glucose = 100.0
            lo, hi = FEATURE_RANGES["avg_glucose_level"]
            features.append(max(0, min(1, (glucose - lo) / (hi - lo))))

            # 9. BMI (handle N/A)
            bmi_str = row.get("bmi", "N/A")
            if bmi_str in ("N/A", "", "?"):
                bmi = 28.0  # median imputation
            else:
                try:
                    bmi = float(bmi_str)
                except ValueError:
                    bmi = 28.0
            lo, hi = FEATURE_RANGES["bmi"]
            features.append(max(0, min(1, (bmi - lo) / (hi - lo))))

            # 10. Smoking status
            features.append(SMOKING_MAP.get(row.get("smoking_status", "Unknown"), 0.5))

            # Target
            try:
                label = int(row.get("stroke", 0))
            except ValueError:
                label = 0

            X_all.append(features)
            y_all.append(label)

    X_all = np.array(X_all, dtype=np.float32)
    y_all = np.array(y_all, dtype=np.int64)

    logger.info("Preprocessed: %d samples, %d features, pos_rate=%.1f%%",
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
        "dataset_name": "Stroke Prediction (Kaggle)",
        "source": "stroke_loader",
        "total_samples": len(y_all),
        "num_features": X_all.shape[1],
        "feature_names": FEATURE_NAMES,
        "num_classes": num_classes,
        "class_names": {0: "No stroke", 1: "Stroke"},
        "class_distribution": {int(k): int(v) for k, v in enumerate(class_counts)},
        "label_name": "stroke",
        "partition_method": "iid" if is_iid else f"dirichlet(alpha={alpha})",
        "test_split": test_split,
        "fhir_mapping": {
            "Patient": ["age", "gender", "ever_married", "work_type", "residence_type"],
            "Condition": ["hypertension", "heart_disease", "stroke"],
            "Observation": ["avg_glucose_level", "bmi", "smoking_status"],
        },
        "ehds_compliance": {
            "coding_system": "Clinical observations (numeric)",
            "pseudonymized": True,
            "multi_center": False,
            "data_categories": ["ehr", "risk_factors"],
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

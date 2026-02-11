"""
UCI Heart Disease Dataset Loader for Federated Learning.

Loads the multi-center UCI Heart Disease dataset (4 hospitals:
Cleveland, Hungarian, Switzerland, VA Long Beach) for federated
learning experiments with natural non-IID partitioning.

920 patients, 13 features, binary target (heart disease presence).

FHIR R4 mapping:
    - age, sex -> Patient demographics
    - cp (chest pain) -> Condition.code
    - trestbps, chol, fbs, thalach -> Observation (vitals/lab)
    - restecg -> DiagnosticReport (ECG)
    - exang, oldpeak, slope, ca, thal -> Observation (stress test)
    - num (target) -> Condition (heart disease)

EHDS compatibility:
    - 4 real hospitals enable natural federated partitioning
    - Standard clinical features map to FHIR Observation/Condition
    - Multi-center international data (US + Hungarian + Swiss)

Reference: Detrano et al., "International application of a new
probability algorithm for the diagnosis of coronary artery disease",
American Journal of Cardiology, 1989.

Author: Fabio Liberti
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# Column names for the processed UCI Heart Disease files
COLUMN_NAMES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope",
    "ca", "thal", "num",
]

# Feature descriptions for metadata
FEATURE_NAMES = [
    "age", "sex", "chest_pain_type", "resting_bp",
    "cholesterol", "fasting_blood_sugar", "resting_ecg",
    "max_heart_rate", "exercise_angina", "st_depression",
    "st_slope", "num_major_vessels", "thalassemia",
]

# Hospital files with origin info
HOSPITAL_FILES = {
    "Cleveland": "processed.cleveland.data",
    "Hungarian": "processed.hungarian.data",
    "Switzerland": "processed.switzerland.data",
    "VA_Long_Beach": "processed.va.data",
}

# Normalization ranges (from medical literature)
FEATURE_RANGES = {
    "age": (20, 80),
    "sex": (0, 1),
    "cp": (1, 4),
    "trestbps": (80, 200),
    "chol": (100, 600),
    "fbs": (0, 1),
    "restecg": (0, 2),
    "thalach": (60, 210),
    "exang": (0, 1),
    "oldpeak": (-3, 7),
    "slope": (1, 3),
    "ca": (0, 3),
    "thal": (3, 7),
}


def load_heart_disease_data(
    num_clients: int = 4,
    partition_by_hospital: bool = True,
    is_iid: bool = False,
    alpha: float = 0.5,
    test_split: float = 0.2,
    seed: int = 42,
    data_path: Optional[str] = None,
) -> Tuple[Dict[int, Tuple[np.ndarray, np.ndarray]],
           Dict[int, Tuple[np.ndarray, np.ndarray]],
           Dict[str, Any]]:
    """Load UCI Heart Disease dataset for federated learning.

    Args:
        num_clients: Number of FL clients. If partition_by_hospital=True
            and num_clients <= 4, maps directly to real hospitals.
        partition_by_hospital: If True, each client gets data from a
            different hospital (natural non-IID).
        is_iid: IID partitioning (only when partition_by_hospital=False).
        alpha: Dirichlet alpha for non-IID.
        test_split: Fraction for test set per client.
        seed: Random seed.
        data_path: Path to heart_disease directory. Auto-discovers if None.

    Returns:
        (client_train_data, client_test_data, metadata)
    """
    rng = np.random.RandomState(seed)

    # Locate data directory
    if data_path is None:
        base = Path(__file__).parent
        candidates = [
            base / "heart_disease",
            base / "Heart_Disease",
        ]
        for p in candidates:
            if p.exists():
                data_path = str(p)
                break
        if data_path is None:
            raise FileNotFoundError(
                f"heart_disease directory not found. Searched: {[str(c) for c in candidates]}"
            )

    data_dir = Path(data_path)
    logger.info("Loading Heart Disease dataset from %s", data_dir)

    # Load all 4 hospital files
    hospital_data = {}
    for hospital_name, filename in HOSPITAL_FILES.items():
        filepath = data_dir / filename
        if not filepath.exists():
            logger.warning("File not found: %s (skipping hospital %s)", filepath, hospital_name)
            continue

        X, y = _load_hospital_file(filepath, rng)
        if len(X) > 0:
            hospital_data[hospital_name] = (X, y)
            logger.info("  %s: %d samples, pos_rate=%.1f%%",
                        hospital_name, len(y), y.mean() * 100)

    if not hospital_data:
        raise FileNotFoundError(f"No valid hospital data found in {data_dir}")

    # Combine all data
    all_X = np.concatenate([X for X, y in hospital_data.values()])
    all_y = np.concatenate([y for X, y in hospital_data.values()])

    logger.info("Total: %d samples, %d features, pos_rate=%.1f%%",
                len(all_y), all_X.shape[1], all_y.mean() * 100)

    # Partition into clients
    if partition_by_hospital and num_clients <= len(hospital_data):
        client_train, client_test, hospital_assignment = _partition_by_hospital(
            hospital_data, num_clients, test_split, rng
        )
    elif is_iid:
        client_train, client_test = _partition_iid(
            all_X, all_y, num_clients, test_split, rng
        )
        hospital_assignment = None
    else:
        client_train, client_test = _partition_dirichlet(
            all_X, all_y, num_clients, alpha, test_split, rng
        )
        hospital_assignment = None

    # Metadata
    num_classes = 2
    class_counts = np.bincount(all_y.astype(int), minlength=num_classes)

    metadata = {
        "dataset_name": "UCI Heart Disease (4 Centers)",
        "source": "heart_disease_loader",
        "total_samples": len(all_y),
        "num_features": all_X.shape[1],
        "feature_names": FEATURE_NAMES,
        "num_classes": num_classes,
        "class_names": {0: "No heart disease", 1: "Heart disease present"},
        "class_distribution": {int(k): int(v) for k, v in enumerate(class_counts)},
        "label_name": "heart_disease",
        "hospitals": list(hospital_data.keys()),
        "hospital_assignment": hospital_assignment,
        "partition_method": "hospital" if (partition_by_hospital and num_clients <= len(hospital_data)) else ("iid" if is_iid else f"dirichlet(alpha={alpha})"),
        "test_split": test_split,
        "fhir_mapping": {
            "Patient": ["age", "sex"],
            "Observation": ["resting_bp", "cholesterol", "fasting_blood_sugar",
                            "max_heart_rate", "st_depression", "num_major_vessels"],
            "Condition": ["chest_pain_type", "heart_disease"],
            "DiagnosticReport": ["resting_ecg", "exercise_angina", "st_slope", "thalassemia"],
        },
        "ehds_compliance": {
            "coding_system": "Clinical observations (numeric)",
            "pseudonymized": True,
            "multi_center": True,
            "international": True,
            "data_categories": ["ehr", "lab_results"],
        },
        "samples_per_client": {
            cid: len(client_train[cid][1]) + len(client_test[cid][1])
            for cid in client_train
        },
    }

    return client_train, client_test, metadata


def _load_hospital_file(filepath: Path, rng: np.random.RandomState) -> Tuple[np.ndarray, np.ndarray]:
    """Load and preprocess a single hospital file."""
    rows = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            values = line.split(",")
            if len(values) != 14:
                continue
            rows.append(values)

    if not rows:
        return np.array([]), np.array([])

    X_list = []
    y_list = []

    for row in rows:
        features = []
        valid = True

        for i, col_name in enumerate(COLUMN_NAMES[:-1]):  # Skip target
            val = row[i].strip()
            if val == "?":
                # Impute with median of the range
                lo, hi = FEATURE_RANGES[col_name]
                features.append((lo + hi) / 2)
            else:
                try:
                    features.append(float(val))
                except ValueError:
                    valid = False
                    break

        if not valid:
            continue

        # Target: num > 0 = heart disease
        target_val = row[13].strip()
        if target_val == "?":
            continue
        try:
            target = 1 if float(target_val) > 0 else 0
        except ValueError:
            continue

        # Normalize features to [0, 1]
        normalized = []
        for i, col_name in enumerate(COLUMN_NAMES[:-1]):
            lo, hi = FEATURE_RANGES[col_name]
            norm_val = (features[i] - lo) / (hi - lo + 1e-8)
            normalized.append(max(0.0, min(1.0, norm_val)))

        X_list.append(normalized)
        y_list.append(target)

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int64)


def _partition_by_hospital(
    hospital_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    num_clients: int,
    test_split: float,
    rng: np.random.RandomState,
) -> Tuple[Dict[int, Tuple[np.ndarray, np.ndarray]],
           Dict[int, Tuple[np.ndarray, np.ndarray]],
           Dict[int, str]]:
    """Partition by real hospital (natural non-IID)."""
    hospitals = list(hospital_data.keys())
    client_train = {}
    client_test = {}
    hospital_assignment = {}

    for cid in range(min(num_clients, len(hospitals))):
        hospital_name = hospitals[cid]
        X, y = hospital_data[hospital_name]
        hospital_assignment[cid] = hospital_name

        indices = np.arange(len(y))
        rng.shuffle(indices)

        n_test = max(1, int(len(indices) * test_split))
        test_idx = indices[:n_test]
        train_idx = indices[n_test:]

        client_train[cid] = (X[train_idx], y[train_idx])
        client_test[cid] = (X[test_idx], y[test_idx])

    return client_train, client_test, hospital_assignment


def _partition_iid(
    X: np.ndarray, y: np.ndarray, num_clients: int,
    test_split: float, rng: np.random.RandomState,
) -> Tuple[Dict[int, Tuple[np.ndarray, np.ndarray]],
           Dict[int, Tuple[np.ndarray, np.ndarray]]]:
    """IID random partitioning."""
    indices = np.arange(len(X))
    rng.shuffle(indices)

    splits = np.array_split(indices, num_clients)
    client_train = {}
    client_test = {}

    for cid, split_idx in enumerate(splits):
        rng.shuffle(split_idx)
        n_test = max(1, int(len(split_idx) * test_split))
        client_train[cid] = (X[split_idx[n_test:]], y[split_idx[n_test:]])
        client_test[cid] = (X[split_idx[:n_test]], y[split_idx[:n_test]])

    return client_train, client_test


def _partition_dirichlet(
    X: np.ndarray, y: np.ndarray, num_clients: int,
    alpha: float, test_split: float, rng: np.random.RandomState,
) -> Tuple[Dict[int, Tuple[np.ndarray, np.ndarray]],
           Dict[int, Tuple[np.ndarray, np.ndarray]]]:
    """Non-IID Dirichlet partitioning."""
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

    client_train = {}
    client_test = {}
    for cid in range(num_clients):
        indices = np.array(client_indices[cid])
        rng.shuffle(indices)
        n_test = max(1, int(len(indices) * test_split))
        client_train[cid] = (X[indices[n_test:]], y[indices[n_test:]])
        client_test[cid] = (X[indices[:n_test]], y[indices[:n_test]])

    return client_train, client_test

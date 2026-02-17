"""
Cardiovascular Disease Dataset Loader for Federated Learning.

Loads and preprocesses the Kaggle Cardiovascular Disease dataset
for federated learning experiments. 70,000 patients with 11
clinical features and binary cardiovascular disease outcome.

FHIR R4 mapping:
    - age, gender -> Patient demographics
    - height, weight -> Observation (vitals / anthropometrics)
    - ap_hi, ap_lo -> Observation (blood pressure systolic/diastolic)
    - cholesterol, gluc -> Observation (lab values, ordinal 1-3)
    - smoke, alco -> Observation (social history)
    - active -> Observation (physical activity)
    - cardio -> Condition (cardiovascular disease)

EHDS compatibility:
    - Large-scale clinical dataset (70K patients)
    - Standard vital signs and lab values (LOINC-mappable)
    - Cardiovascular risk factors relevant to European CVD burden
    - Age in days enables precise epidemiological analysis

Reference: Ulianova, "Cardiovascular Disease dataset", Kaggle, 2019.

Author: Fabio Liberti
"""

import csv
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

FEATURE_NAMES = [
    "age", "gender", "height", "weight",
    "ap_hi", "ap_lo", "cholesterol", "gluc",
    "smoke", "alco", "active",
]

# Normalization ranges (clinical)
FEATURE_RANGES = {
    "age": (10798, 23713),       # age in days (~29-65 years)
    "height": (130, 200),        # cm
    "weight": (40, 160),         # kg
    "ap_hi": (80, 200),          # systolic mmHg
    "ap_lo": (50, 130),          # diastolic mmHg
    "cholesterol": (1, 3),       # ordinal: 1=normal, 2=above, 3=well above
    "gluc": (1, 3),              # ordinal: 1=normal, 2=above, 3=well above
}


def load_cardiovascular_data(
    num_clients: int = 5,
    is_iid: bool = False,
    alpha: float = 0.5,
    test_split: float = 0.2,
    seed: int = 42,
    max_samples: Optional[int] = None,
    data_path: Optional[str] = None,
) -> Tuple[Dict[int, Tuple[np.ndarray, np.ndarray]],
           Dict[int, Tuple[np.ndarray, np.ndarray]],
           Dict[str, Any]]:
    """Load Cardiovascular Disease dataset for federated learning.

    Args:
        num_clients: Number of FL clients.
        is_iid: IID partitioning if True, Dirichlet non-IID if False.
        alpha: Dirichlet alpha for non-IID partitioning.
        test_split: Fraction for test set per client.
        seed: Random seed.
        max_samples: If set, subsample to this many records.
        data_path: Path to CSV. Auto-discovers if None.

    Returns:
        (client_train_data, client_test_data, metadata)
    """
    rng = np.random.RandomState(seed)

    if data_path is None:
        base = Path(__file__).parent
        candidates = [
            base / "cardiovascular_disease" / "cardio_train.csv",
            base / "cardiovascular" / "cardio_train.csv",
        ]
        for p in candidates:
            if p.exists():
                data_path = str(p)
                break
        if data_path is None:
            raise FileNotFoundError(
                f"Cardiovascular dataset not found. Searched: {[str(c) for c in candidates]}"
            )

    logger.info("Loading Cardiovascular Disease dataset from %s", data_path)

    X_all = []
    y_all = []

    with open(data_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            try:
                label = int(row.get("cardio", 0))
            except ValueError:
                continue

            features = []
            valid = True

            for feat_name in FEATURE_NAMES:
                raw = row.get(feat_name, "").strip()
                try:
                    val = float(raw)
                except ValueError:
                    valid = False
                    break

                if feat_name in FEATURE_RANGES:
                    lo, hi = FEATURE_RANGES[feat_name]
                    val = max(0.0, min(1.0, (val - lo) / (hi - lo + 1e-8)))
                # Binary features (smoke, alco, active, gender) already 0/1

                features.append(val)

            if not valid:
                continue

            X_all.append(features)
            y_all.append(label)

    X_all = np.array(X_all, dtype=np.float32)
    y_all = np.array(y_all, dtype=np.int64)

    # Filter outliers (extreme blood pressure values)
    valid_mask = (X_all[:, FEATURE_NAMES.index("ap_hi")] >= 0) & \
                 (X_all[:, FEATURE_NAMES.index("ap_hi")] <= 1) & \
                 (X_all[:, FEATURE_NAMES.index("ap_lo")] >= 0) & \
                 (X_all[:, FEATURE_NAMES.index("ap_lo")] <= 1)
    X_all = X_all[valid_mask]
    y_all = y_all[valid_mask]

    if max_samples is not None and len(X_all) > max_samples:
        idx = rng.choice(len(X_all), max_samples, replace=False)
        X_all = X_all[idx]
        y_all = y_all[idx]

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
        "dataset_name": "Cardiovascular Disease (Kaggle)",
        "source": "cardiovascular_loader",
        "total_samples": len(y_all),
        "num_features": X_all.shape[1],
        "feature_names": FEATURE_NAMES,
        "num_classes": num_classes,
        "class_names": {0: "No CVD", 1: "Cardiovascular Disease"},
        "class_distribution": {int(k): int(v) for k, v in enumerate(class_counts)},
        "label_name": "cardiovascular_disease",
        "partition_method": "iid" if is_iid else f"dirichlet(alpha={alpha})",
        "test_split": test_split,
        "fhir_mapping": {
            "Patient": ["age", "gender"],
            "Observation_vitals": ["height", "weight", "ap_hi", "ap_lo"],
            "Observation_lab": ["cholesterol", "gluc"],
            "Observation_social": ["smoke", "alco", "active"],
            "Condition": ["cardiovascular_disease"],
        },
        "ehds_compliance": {
            "coding_system": "Clinical observations (numeric/ordinal)",
            "pseudonymized": True,
            "multi_center": False,
            "data_categories": ["ehr", "vitals", "lab_results", "risk_factors"],
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

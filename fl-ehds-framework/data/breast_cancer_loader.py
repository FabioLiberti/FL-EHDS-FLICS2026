"""
Breast Cancer Wisconsin (Diagnostic) Loader for Federated Learning.

Loads and preprocesses the UCI Breast Cancer Wisconsin (Diagnostic)
dataset for federated learning experiments. 569 patients with 30
numeric features from digitized FNA images and binary diagnosis.

FHIR R4 mapping:
    - 30 cell nucleus features -> Observation (pathology measurements)
    - diagnosis (M/B) -> Condition (malignant/benign neoplasm)
    - All features from DiagnosticReport (cytology FNA)

EHDS compatibility:
    - Standard pathology measurements (LOINC-mappable)
    - Binary classification (malignant vs benign)
    - Small dataset tests FL on limited-data scenarios
    - Cancer screening relevant to EU Cancer Plan integration

Reference: Wolberg et al., "Breast Cancer Wisconsin (Diagnostic)",
    UCI ML Repository ID 17, 1995.

Author: Fabio Liberti
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# 30 feature names: 10 base features x 3 (mean, se, worst)
BASE_FEATURES = [
    "radius", "texture", "perimeter", "area", "smoothness",
    "compactness", "concavity", "concave_points", "symmetry",
    "fractal_dimension",
]

FEATURE_NAMES = []
for suffix in ["mean", "se", "worst"]:
    for feat in BASE_FEATURES:
        FEATURE_NAMES.append(f"{feat}_{suffix}")

# Normalization ranges (from dataset statistics)
FEATURE_RANGES = {
    "radius_mean": (6.0, 29.0), "texture_mean": (9.0, 40.0),
    "perimeter_mean": (40.0, 190.0), "area_mean": (140.0, 2510.0),
    "smoothness_mean": (0.05, 0.17), "compactness_mean": (0.02, 0.35),
    "concavity_mean": (0.0, 0.43), "concave_points_mean": (0.0, 0.2),
    "symmetry_mean": (0.1, 0.31), "fractal_dimension_mean": (0.05, 0.1),
    "radius_se": (0.1, 2.9), "texture_se": (0.3, 4.9),
    "perimeter_se": (0.7, 22.0), "area_se": (6.0, 542.0),
    "smoothness_se": (0.001, 0.032), "compactness_se": (0.002, 0.14),
    "concavity_se": (0.0, 0.4), "concave_points_se": (0.0, 0.053),
    "symmetry_se": (0.008, 0.08), "fractal_dimension_se": (0.001, 0.03),
    "radius_worst": (7.0, 37.0), "texture_worst": (12.0, 50.0),
    "perimeter_worst": (50.0, 252.0), "area_worst": (185.0, 4255.0),
    "smoothness_worst": (0.07, 0.23), "compactness_worst": (0.03, 1.06),
    "concavity_worst": (0.0, 1.25), "concave_points_worst": (0.0, 0.29),
    "symmetry_worst": (0.15, 0.66), "fractal_dimension_worst": (0.05, 0.21),
}


def load_breast_cancer_data(
    num_clients: int = 4,
    is_iid: bool = False,
    alpha: float = 0.5,
    test_split: float = 0.2,
    seed: int = 42,
    data_path: Optional[str] = None,
) -> Tuple[Dict[int, Tuple[np.ndarray, np.ndarray]],
           Dict[int, Tuple[np.ndarray, np.ndarray]],
           Dict[str, Any]]:
    """Load Breast Cancer Wisconsin (Diagnostic) dataset for federated learning.

    Args:
        num_clients: Number of FL clients.
        is_iid: IID partitioning if True, Dirichlet non-IID if False.
        alpha: Dirichlet alpha for non-IID partitioning.
        test_split: Fraction for test set per client.
        seed: Random seed.
        data_path: Path to wdbc.data. Auto-discovers if None.

    Returns:
        (client_train_data, client_test_data, metadata)
    """
    rng = np.random.RandomState(seed)

    if data_path is None:
        base = Path(__file__).parent
        candidates = [
            base / "breast_cancer_wisconsin" / "wdbc.data",
            base / "breast_cancer" / "wdbc.data",
        ]
        for p in candidates:
            if p.exists():
                data_path = str(p)
                break
        if data_path is None:
            raise FileNotFoundError(
                f"Breast Cancer Wisconsin dataset not found. Searched: {[str(c) for c in candidates]}"
            )

    logger.info("Loading Breast Cancer Wisconsin dataset from %s", data_path)

    X_all = []
    y_all = []

    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) != 32:  # ID + diagnosis + 30 features
                continue

            # Column 1: diagnosis (M=malignant, B=benign)
            diagnosis = parts[1].strip()
            if diagnosis == "M":
                label = 1
            elif diagnosis == "B":
                label = 0
            else:
                continue

            # Columns 2-31: 30 numeric features
            features = []
            valid = True
            for i, feat_name in enumerate(FEATURE_NAMES):
                try:
                    val = float(parts[i + 2])
                except (ValueError, IndexError):
                    valid = False
                    break

                lo, hi = FEATURE_RANGES[feat_name]
                val = max(0.0, min(1.0, (val - lo) / (hi - lo + 1e-8)))
                features.append(val)

            if not valid:
                continue

            X_all.append(features)
            y_all.append(label)

    X_all = np.array(X_all, dtype=np.float32)
    y_all = np.array(y_all, dtype=np.int64)

    logger.info("Preprocessed: %d samples, %d features, malignant_rate=%.1f%%",
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
        "dataset_name": "Breast Cancer Wisconsin (Diagnostic, UCI)",
        "source": "breast_cancer_loader",
        "total_samples": len(y_all),
        "num_features": X_all.shape[1],
        "feature_names": FEATURE_NAMES,
        "num_classes": num_classes,
        "class_names": {0: "Benign", 1: "Malignant"},
        "class_distribution": {int(k): int(v) for k, v in enumerate(class_counts)},
        "label_name": "malignant",
        "partition_method": "iid" if is_iid else f"dirichlet(alpha={alpha})",
        "test_split": test_split,
        "fhir_mapping": {
            "DiagnosticReport": ["cytology_fna"],
            "Observation_pathology": FEATURE_NAMES,
            "Condition": ["malignant_neoplasm"],
        },
        "ehds_compliance": {
            "coding_system": "Pathology measurements (LOINC-mappable)",
            "pseudonymized": True,
            "multi_center": False,
            "data_categories": ["pathology", "diagnostics"],
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
        # Ensure each client gets at least 1 sample per class (prevents empty partitions)
        for i in range(num_clients):
            if proportions[i] == 0 and len(class_idx) > num_clients:
                proportions[i] = 1
        # Adjust first client for any remainder
        proportions[0] += len(class_idx) - proportions.sum()
        start = 0
        for cid in range(num_clients):
            end = start + proportions[cid]
            client_indices[cid].extend(class_idx[start:end].tolist())
            start = end
    client_train, client_test = {}, {}
    for cid in range(num_clients):
        indices = np.array(client_indices[cid], dtype=np.int64)
        if len(indices) == 0:
            # Safety fallback: empty client gets a random sample
            indices = np.array([rng.randint(0, len(X))], dtype=np.int64)
        rng.shuffle(indices)
        n_test = max(1, int(len(indices) * test_split))
        if n_test >= len(indices):
            n_test = max(1, len(indices) // 2)
        client_train[cid] = (X[indices[n_test:]], y[indices[n_test:]])
        client_test[cid] = (X[indices[:n_test]], y[indices[:n_test]])
    return client_train, client_test

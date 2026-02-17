"""
CDC Diabetes Health Indicators (BRFSS 2015) Loader for Federated Learning.

Loads and preprocesses the CDC Diabetes Health Indicators dataset
from the Behavioral Risk Factor Surveillance System (BRFSS) 2015.
253,680 survey responses with 21 health indicator features.

FHIR R4 mapping:
    - Age, Sex, Education, Income -> Patient demographics
    - BMI -> Observation (vitals)
    - HighBP, HighChol, Stroke, HeartDiseaseorAttack -> Condition.code
    - Smoker, HvyAlcoholConsump, PhysActivity -> Observation (social)
    - GenHlth, MentHlth, PhysHlth, DiffWalk -> Observation (PRO)
    - Diabetes_binary -> Condition (diabetes mellitus)

EHDS compatibility:
    - Large-scale population survey (253K respondents)
    - 21 features covering demographics, vitals, conditions, behaviors
    - Binary diabetes classification (prediabetes/diabetes vs none)
    - All features already numeric and clean (no missing values)

Reference: CDC, "Behavioral Risk Factor Surveillance System (BRFSS)",
    2015 Survey Data. UCI ML Repository ID 891.

Author: Fabio Liberti
"""

import csv
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

FEATURE_NAMES = [
    "HighBP", "HighChol", "CholCheck", "BMI", "Smoker",
    "Stroke", "HeartDiseaseorAttack", "PhysActivity", "Fruits",
    "Veggies", "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost",
    "GenHlth", "MentHlth", "PhysHlth", "DiffWalk", "Sex",
    "Age", "Education", "Income",
]

# Normalization ranges
FEATURE_RANGES = {
    "BMI": (10, 60),
    "GenHlth": (1, 5),
    "MentHlth": (0, 30),
    "PhysHlth": (0, 30),
    "Age": (1, 13),
    "Education": (1, 6),
    "Income": (1, 8),
}


def load_cdc_diabetes_data(
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
    """Load CDC Diabetes Health Indicators dataset for federated learning.

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
            base / "cdc_diabetes" / "diabetes_binary_health_indicators_BRFSS2015.csv",
            base / "cdc_diabetes" / "diabetes_binary_5050split_health_indicators_BRFSS2015.csv",
        ]
        for p in candidates:
            if p.exists():
                data_path = str(p)
                break
        if data_path is None:
            raise FileNotFoundError(
                f"CDC Diabetes dataset not found. Searched: {[str(c) for c in candidates]}"
            )

    logger.info("Loading CDC Diabetes Health Indicators from %s", data_path)

    # Read CSV (all features are already numeric)
    X_all = []
    y_all = []

    with open(data_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Target: Diabetes_binary (0.0 or 1.0)
            try:
                label = int(float(row.get("Diabetes_binary", 0)))
            except ValueError:
                continue

            features = []
            for feat_name in FEATURE_NAMES:
                try:
                    val = float(row.get(feat_name, 0))
                except ValueError:
                    val = 0.0

                # Normalize continuous features to [0,1]
                if feat_name in FEATURE_RANGES:
                    lo, hi = FEATURE_RANGES[feat_name]
                    val = max(0, min(1, (val - lo) / (hi - lo)))
                # Binary features are already 0/1

                features.append(val)

            X_all.append(features)
            y_all.append(label)

    X_all = np.array(X_all, dtype=np.float32)
    y_all = np.array(y_all, dtype=np.int64)

    # Subsample if requested
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
        "dataset_name": "CDC Diabetes Health Indicators (BRFSS 2015)",
        "source": "cdc_diabetes_loader",
        "total_samples": len(y_all),
        "num_features": X_all.shape[1],
        "feature_names": FEATURE_NAMES,
        "num_classes": num_classes,
        "class_names": {0: "No diabetes", 1: "Prediabetes/Diabetes"},
        "class_distribution": {int(k): int(v) for k, v in enumerate(class_counts)},
        "label_name": "diabetes_binary",
        "partition_method": "iid" if is_iid else f"dirichlet(alpha={alpha})",
        "test_split": test_split,
        "fhir_mapping": {
            "Patient": ["Sex", "Age", "Education", "Income"],
            "Condition": ["HighBP", "HighChol", "Stroke", "HeartDiseaseorAttack",
                          "diabetes_binary"],
            "Observation": ["BMI", "GenHlth", "MentHlth", "PhysHlth",
                            "CholCheck", "PhysActivity", "Fruits", "Veggies"],
            "RiskAssessment": ["Smoker", "HvyAlcoholConsump", "DiffWalk"],
        },
        "ehds_compliance": {
            "coding_system": "BRFSS survey codes",
            "pseudonymized": True,
            "multi_center": False,
            "population_survey": True,
            "data_categories": ["survey", "risk_factors", "conditions"],
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

"""
Cirrhosis Patient Survival Prediction Loader for Federated Learning.

Loads and preprocesses the UCI Cirrhosis Patient Survival dataset
(Mayo Clinic PBC trial, 1974-1984) for federated learning experiments.
418 patients with 17 clinical features.

FHIR R4 mapping:
    - Age, Sex -> Patient demographics
    - Drug -> MedicationStatement (D-penicillamine vs placebo)
    - Ascites, Hepatomegaly, Spiders, Edema -> Condition (clinical signs)
    - Bilirubin, Cholesterol, Albumin, Copper -> Observation (liver panel)
    - Alk_Phos, SGOT -> Observation (liver enzymes)
    - Tryglicerides -> Observation (lipid panel)
    - Platelets -> Observation (CBC)
    - Prothrombin -> Observation (coagulation)
    - Stage -> Condition.stage (histologic stage 1-4)
    - Status (D/C/CL) -> Encounter.hospitalization.dischargeDisposition

EHDS compatibility:
    - Standard hepatology laboratory values (LOINC-mappable)
    - Clinical staging (histologic) common in European hepatology
    - Survival/mortality endpoint relevant for EHDS secondary use
    - Drug trial data enables treatment-outcome FL studies

Reference: Dickson et al., "Prognosis in primary biliary cirrhosis:
    Model for decision making", Hepatology, 1989.
    UCI ML Repository ID 878.

Author: Fabio Liberti
"""

import csv
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

FEATURE_NAMES = [
    "n_days",
    "drug",
    "age",
    "sex",
    "ascites",
    "hepatomegaly",
    "spiders",
    "edema",
    "bilirubin",
    "cholesterol",
    "albumin",
    "copper",
    "alk_phos",
    "sgot",
    "tryglicerides",
    "platelets",
    "prothrombin",
    "stage",
]

# Normalization ranges (clinical)
NUMERIC_RANGES = {
    "n_days": (41, 4795),
    "age": (9598, 28650),  # age in days in the dataset
    "bilirubin": (0.3, 28.0),
    "cholesterol": (120, 1775),
    "albumin": (1.96, 4.64),
    "copper": (4, 588),
    "alk_phos": (289, 13862),
    "sgot": (26, 457),
    "tryglicerides": (33, 598),
    "platelets": (62, 721),
    "prothrombin": (9.0, 18.0),
    "stage": (1, 4),
}

# Categorical encodings
SEX_MAP = {"M": 1.0, "F": 0.0}
DRUG_MAP = {"D-penicillamine": 1.0, "Placebo": 0.0}
BINARY_MAP = {"Y": 1.0, "N": 0.0}
EDEMA_MAP = {"N": 0.0, "S": 0.5, "Y": 1.0}

# Median values for imputation
MEDIANS = {
    "n_days": 1730, "age": 18628, "bilirubin": 1.4, "cholesterol": 309.5,
    "albumin": 3.53, "copper": 73.0, "alk_phos": 1259.0, "sgot": 114.7,
    "tryglicerides": 108.0, "platelets": 257.0, "prothrombin": 10.6,
    "stage": 3.0,
}


def load_cirrhosis_data(
    num_clients: int = 4,
    is_iid: bool = False,
    alpha: float = 0.5,
    label_type: str = "binary",
    test_split: float = 0.2,
    seed: int = 42,
    data_path: Optional[str] = None,
) -> Tuple[Dict[int, Tuple[np.ndarray, np.ndarray]],
           Dict[int, Tuple[np.ndarray, np.ndarray]],
           Dict[str, Any]]:
    """Load Cirrhosis Patient Survival dataset for federated learning.

    Args:
        num_clients: Number of FL clients.
        is_iid: IID partitioning if True, Dirichlet non-IID if False.
        alpha: Dirichlet alpha for non-IID partitioning.
        label_type: "binary" (D vs C+CL) or "multiclass" (D/C/CL).
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
            base / "cirrhosis" / "cirrhosis.csv",
            base / "Cirrhosis" / "cirrhosis.csv",
        ]
        for p in candidates:
            if p.exists():
                data_path = str(p)
                break
        if data_path is None:
            raise FileNotFoundError(
                f"Cirrhosis dataset not found. Searched: {[str(c) for c in candidates]}"
            )

    logger.info("Loading Cirrhosis Patient Survival dataset from %s", data_path)

    X_all = []
    y_all = []

    with open(data_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Target: Status (D=Death, C=Censored, CL=Censored/Liver transplant)
            status = row.get("Status", "").strip()
            if status not in ("D", "C", "CL"):
                continue

            if label_type == "binary":
                label = 1 if status == "D" else 0
            else:
                label = {"C": 0, "CL": 1, "D": 2}.get(status, 0)

            features = []

            # 1. N_Days (time to event)
            features.append(_normalize_numeric(row, "N_Days", "n_days"))

            # 2. Drug
            drug = row.get("Drug", "").strip()
            if drug in DRUG_MAP:
                features.append(DRUG_MAP[drug])
            else:
                features.append(0.5)  # unknown

            # 3. Age (in days in this dataset)
            features.append(_normalize_numeric(row, "Age", "age"))

            # 4. Sex
            sex = row.get("Sex", "").strip()
            features.append(SEX_MAP.get(sex, 0.5))

            # 5. Ascites
            features.append(BINARY_MAP.get(row.get("Ascites", "").strip(), 0.5))

            # 6. Hepatomegaly
            features.append(BINARY_MAP.get(row.get("Hepatomegaly", "").strip(), 0.5))

            # 7. Spiders
            features.append(BINARY_MAP.get(row.get("Spiders", "").strip(), 0.5))

            # 8. Edema
            features.append(EDEMA_MAP.get(row.get("Edema", "").strip(), 0.5))

            # 9-18. Numeric lab values
            for csv_col, feat_key in [
                ("Bilirubin", "bilirubin"),
                ("Cholesterol", "cholesterol"),
                ("Albumin", "albumin"),
                ("Copper", "copper"),
                ("Alk_Phos", "alk_phos"),
                ("SGOT", "sgot"),
                ("Tryglicerides", "tryglicerides"),
                ("Platelets", "platelets"),
                ("Prothrombin", "prothrombin"),
                ("Stage", "stage"),
            ]:
                features.append(_normalize_numeric(row, csv_col, feat_key))

            X_all.append(features)
            y_all.append(label)

    X_all = np.array(X_all, dtype=np.float32)
    y_all = np.array(y_all, dtype=np.int64)

    logger.info("Preprocessed: %d samples, %d features, mortality_rate=%.1f%%",
                len(y_all), X_all.shape[1],
                (y_all == (1 if label_type == "binary" else 2)).mean() * 100)

    # Partition
    if is_iid:
        client_train, client_test = _partition_iid(X_all, y_all, num_clients, test_split, rng)
    else:
        client_train, client_test = _partition_dirichlet(X_all, y_all, num_clients, alpha, test_split, rng)

    # Metadata
    num_classes = len(np.unique(y_all))
    class_counts = np.bincount(y_all, minlength=num_classes)

    if label_type == "binary":
        class_names = {0: "Alive/Censored", 1: "Death"}
    else:
        class_names = {0: "Censored", 1: "Liver Transplant", 2: "Death"}

    metadata = {
        "dataset_name": "Cirrhosis Patient Survival (Mayo/UCI)",
        "source": "cirrhosis_loader",
        "total_samples": len(y_all),
        "num_features": X_all.shape[1],
        "feature_names": FEATURE_NAMES,
        "num_classes": num_classes,
        "class_names": class_names,
        "class_distribution": {int(k): int(v) for k, v in enumerate(class_counts)},
        "label_type": label_type,
        "label_name": "mortality" if label_type == "binary" else "status_3class",
        "partition_method": "iid" if is_iid else f"dirichlet(alpha={alpha})",
        "test_split": test_split,
        "fhir_mapping": {
            "Patient": ["age", "sex"],
            "MedicationStatement": ["drug"],
            "Condition": ["ascites", "hepatomegaly", "spiders", "edema", "stage"],
            "Observation_liver": ["bilirubin", "cholesterol", "albumin", "copper",
                                  "alk_phos", "sgot"],
            "Observation_lipid": ["tryglicerides"],
            "Observation_hematology": ["platelets", "prothrombin"],
            "Encounter": ["n_days", "status"],
        },
        "ehds_compliance": {
            "coding_system": "Clinical laboratory values (LOINC-mappable)",
            "pseudonymized": True,
            "multi_center": False,
            "clinical_trial": True,
            "data_categories": ["lab_results", "conditions", "medications"],
        },
        "samples_per_client": {
            cid: len(client_train[cid][1]) + len(client_test[cid][1])
            for cid in client_train
        },
    }

    return client_train, client_test, metadata


def _normalize_numeric(row: dict, csv_col: str, feat_key: str) -> float:
    """Extract, impute, and normalize a numeric value."""
    raw = row.get(csv_col, "").strip()
    if raw in ("", "NA", "?", "nan"):
        val = MEDIANS.get(feat_key, 0.0)
    else:
        try:
            val = float(raw)
        except ValueError:
            val = MEDIANS.get(feat_key, 0.0)
    lo, hi = NUMERIC_RANGES[feat_key]
    return max(0.0, min(1.0, (val - lo) / (hi - lo + 1e-8)))


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

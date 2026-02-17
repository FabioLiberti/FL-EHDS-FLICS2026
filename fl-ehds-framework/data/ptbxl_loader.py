"""
PTB-XL ECG Tabular Metadata Loader for Federated Learning.

Loads and preprocesses the PTB-XL ECG dataset metadata for federated
learning experiments using ONLY tabular features (no waveforms needed).
21,799 12-lead ECG records from the Physikalisch-Technische
Bundesanstalt (PTB), Germany — a native European dataset.

The loader extracts:
    - Demographics: age, sex
    - Anthropometrics: height, weight (when available)
    - SCP diagnostic codes -> 5 superclass binary features (NORM, MI, STTC, CD, HYP)
    - Site ID for natural hospital-based FL partitioning

FHIR R4 mapping:
    - age, sex -> Patient demographics
    - height, weight -> Observation (vitals)
    - scp_codes -> DiagnosticReport / Observation (ECG interpretation)
    - site -> Organization (recording institution)
    - diagnostic_superclass -> Condition.code (cardiac diagnosis)

EHDS compatibility:
    - European origin (PTB, Berlin, Germany)
    - SCP-ECG coding system (European standard EN 1064)
    - Multi-site data enables natural federated partitioning
    - 52 recording sites simulate real hospital federation
    - Diagnostic codes map to ICD-10 cardiac categories

Reference: Wagner et al., "PTB-XL, a large publicly available
    electrocardiography dataset", Scientific Data, 2020.
    PhysioNet DOI: 10.13026/6sec-a640.

Author: Fabio Liberti
"""

import ast
import csv
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# 5 SCP-ECG diagnostic superclasses
DIAGNOSTIC_SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]

# Feature names: demographics + 5 diagnostic superclass indicators
FEATURE_NAMES = [
    "age", "sex", "height", "weight",
    "diag_NORM", "diag_MI", "diag_STTC", "diag_CD", "diag_HYP",
]

# Normalization ranges
FEATURE_RANGES = {
    "age": (2, 95),
    "height": (130, 200),  # cm
    "weight": (40, 150),   # kg
}

# Median values for imputation
MEDIANS = {
    "age": 63.0, "height": 168.0, "weight": 75.0,
}


def load_ptbxl_data(
    num_clients: int = 5,
    partition_by_site: bool = True,
    is_iid: bool = False,
    alpha: float = 0.5,
    test_split: float = 0.2,
    seed: int = 42,
    min_site_samples: int = 50,
    data_path: Optional[str] = None,
    scp_path: Optional[str] = None,
) -> Tuple[Dict[int, Tuple[np.ndarray, np.ndarray]],
           Dict[int, Tuple[np.ndarray, np.ndarray]],
           Dict[str, Any]]:
    """Load PTB-XL ECG tabular metadata for federated learning.

    Args:
        num_clients: Number of FL clients.
        partition_by_site: If True, partition by recording site (natural FL).
        is_iid: IID partitioning (only when partition_by_site=False).
        alpha: Dirichlet alpha for non-IID partitioning.
        test_split: Fraction for test set per client.
        seed: Random seed.
        min_site_samples: Minimum samples for a site to be a client
            (only when partition_by_site=True).
        data_path: Path to ptbxl_database.csv. Auto-discovers if None.
        scp_path: Path to scp_statements.csv. Auto-discovers if None.

    Returns:
        (client_train_data, client_test_data, metadata)
    """
    rng = np.random.RandomState(seed)

    if data_path is None:
        base = Path(__file__).parent
        candidates = [
            base / "ptb_xl" / "ptbxl_database.csv",
            base / "ptbxl" / "ptbxl_database.csv",
        ]
        for p in candidates:
            if p.exists():
                data_path = str(p)
                break
        if data_path is None:
            raise FileNotFoundError(
                f"PTB-XL dataset not found. Searched: {[str(c) for c in candidates]}"
            )

    if scp_path is None:
        base = Path(data_path).parent
        scp_candidates = [
            base / "scp_statements.csv",
        ]
        for p in scp_candidates:
            if p.exists():
                scp_path = str(p)
                break

    logger.info("Loading PTB-XL tabular metadata from %s", data_path)

    # Load SCP statement -> superclass mapping
    scp_to_superclass = _load_scp_mapping(scp_path) if scp_path else {}

    X_all = []
    y_all = []
    site_ids = []

    with open(data_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Parse SCP codes
            scp_raw = row.get("scp_codes", "{}")
            try:
                scp_dict = ast.literal_eval(scp_raw)
            except (ValueError, SyntaxError):
                continue

            # Determine diagnostic superclass (primary label)
            superclass_scores = {sc: 0.0 for sc in DIAGNOSTIC_SUPERCLASSES}
            for code, likelihood in scp_dict.items():
                sc = scp_to_superclass.get(code)
                if sc and sc in superclass_scores:
                    superclass_scores[sc] = max(superclass_scores[sc], float(likelihood))

            # Primary label = superclass with highest score
            primary_label = max(superclass_scores, key=superclass_scores.get)
            if superclass_scores[primary_label] == 0.0:
                # No diagnostic code found, default to NORM if SR present
                if "SR" in scp_dict or "NORM" in scp_dict:
                    primary_label = "NORM"
                else:
                    continue

            label = DIAGNOSTIC_SUPERCLASSES.index(primary_label)

            # Extract features
            features = []

            # Age
            age_val = _parse_numeric(row.get("age", ""), MEDIANS["age"])
            if age_val > 100:
                age_val = MEDIANS["age"]  # filter outliers (max=300 in data)
            lo, hi = FEATURE_RANGES["age"]
            features.append(max(0.0, min(1.0, (age_val - lo) / (hi - lo + 1e-8))))

            # Sex (0=female, 1=male in dataset)
            sex_val = _parse_numeric(row.get("sex", ""), 0.5)
            features.append(sex_val)

            # Height
            height_val = _parse_numeric(row.get("height", ""), MEDIANS["height"])
            lo, hi = FEATURE_RANGES["height"]
            features.append(max(0.0, min(1.0, (height_val - lo) / (hi - lo + 1e-8))))

            # Weight
            weight_val = _parse_numeric(row.get("weight", ""), MEDIANS["weight"])
            lo, hi = FEATURE_RANGES["weight"]
            features.append(max(0.0, min(1.0, (weight_val - lo) / (hi - lo + 1e-8))))

            # 5 diagnostic superclass indicators (binary/continuous)
            for sc in DIAGNOSTIC_SUPERCLASSES:
                score = superclass_scores[sc]
                features.append(1.0 if score > 0 else 0.0)

            X_all.append(features)
            y_all.append(label)

            # Site for partition
            site_raw = row.get("site", "").strip()
            try:
                site_ids.append(int(float(site_raw)))
            except ValueError:
                site_ids.append(-1)

    X_all = np.array(X_all, dtype=np.float32)
    y_all = np.array(y_all, dtype=np.int64)
    site_ids = np.array(site_ids, dtype=np.int32)

    logger.info("Preprocessed: %d samples, %d features, %d classes",
                len(y_all), X_all.shape[1], len(DIAGNOSTIC_SUPERCLASSES))

    # Partition
    if partition_by_site:
        client_train, client_test, site_assignment = _partition_by_site(
            X_all, y_all, site_ids, num_clients, min_site_samples, test_split, rng
        )
    elif is_iid:
        client_train, client_test = _partition_iid(X_all, y_all, num_clients, test_split, rng)
        site_assignment = None
    else:
        client_train, client_test = _partition_dirichlet(X_all, y_all, num_clients, alpha, test_split, rng)
        site_assignment = None

    # Metadata
    num_classes = len(DIAGNOSTIC_SUPERCLASSES)
    class_counts = np.bincount(y_all, minlength=num_classes)

    metadata = {
        "dataset_name": "PTB-XL ECG Tabular (PhysioNet)",
        "source": "ptbxl_loader",
        "total_samples": len(y_all),
        "num_features": X_all.shape[1],
        "feature_names": FEATURE_NAMES,
        "num_classes": num_classes,
        "class_names": {i: sc for i, sc in enumerate(DIAGNOSTIC_SUPERCLASSES)},
        "class_distribution": {int(k): int(v) for k, v in enumerate(class_counts)},
        "label_name": "ecg_diagnostic_superclass",
        "partition_method": "site" if partition_by_site else ("iid" if is_iid else f"dirichlet(alpha={alpha})"),
        "test_split": test_split,
        "site_assignment": site_assignment,
        "european_origin": True,
        "fhir_mapping": {
            "Patient": ["age", "sex"],
            "Observation_vitals": ["height", "weight"],
            "DiagnosticReport": ["ecg_interpretation"],
            "Condition": ["NORM", "MI", "STTC", "CD", "HYP"],
            "Organization": ["site"],
        },
        "ehds_compliance": {
            "coding_system": "SCP-ECG (EN 1064), ICD-10 cardiac",
            "pseudonymized": True,
            "multi_center": True,
            "european_origin": True,
            "num_sites": len(np.unique(site_ids[site_ids >= 0])),
            "data_categories": ["ecg", "diagnostics", "vitals"],
        },
        "samples_per_client": {
            cid: len(client_train[cid][1]) + len(client_test[cid][1])
            for cid in client_train
        },
    }

    return client_train, client_test, metadata


def _load_scp_mapping(scp_path: str) -> Dict[str, str]:
    """Load SCP code -> diagnostic superclass mapping."""
    mapping = {}
    with open(scp_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            code = row.get("", "").strip()  # first column (unnamed index)
            if not code:
                # Try alternative column names
                for key in row:
                    if row[key].strip() and len(row[key].strip()) <= 10:
                        code = row[key].strip()
                        break
            diag = row.get("diagnostic", "").strip()
            superclass = row.get("diagnostic_class", "").strip()
            if code and diag == "1.0" and superclass:
                mapping[code] = superclass
    return mapping


def _parse_numeric(val_str: str, default: float) -> float:
    """Parse numeric value with default for missing."""
    val_str = val_str.strip()
    if not val_str or val_str in ("", "nan", "?"):
        return default
    try:
        return float(val_str)
    except ValueError:
        return default


def _partition_by_site(X, y, site_ids, num_clients, min_samples, test_split, rng):
    """Partition by recording site (natural hospital-based FL)."""
    unique_sites = np.unique(site_ids[site_ids >= 0])
    # Filter sites with enough samples
    valid_sites = [s for s in unique_sites if np.sum(site_ids == s) >= min_samples]
    valid_sites.sort(key=lambda s: -np.sum(site_ids == s))  # largest first

    # Take top num_clients sites
    selected_sites = valid_sites[:num_clients]

    client_train, client_test = {}, {}
    site_assignment = {}

    for cid, site in enumerate(selected_sites):
        mask = site_ids == site
        indices = np.where(mask)[0]
        rng.shuffle(indices)
        n_test = max(1, int(len(indices) * test_split))
        client_train[cid] = (X[indices[n_test:]], y[indices[n_test:]])
        client_test[cid] = (X[indices[:n_test]], y[indices[:n_test]])
        site_assignment[cid] = int(site)

    # If remaining data, distribute to last client
    used_mask = np.isin(site_ids, selected_sites)
    remaining = np.where(~used_mask & (site_ids >= 0))[0]
    if len(remaining) > 0 and len(selected_sites) > 0:
        last_cid = len(selected_sites) - 1
        rng.shuffle(remaining)
        n_test = max(1, int(len(remaining) * test_split))
        # Append to last client
        existing_train_X, existing_train_y = client_train[last_cid]
        existing_test_X, existing_test_y = client_test[last_cid]
        client_train[last_cid] = (
            np.concatenate([existing_train_X, X[remaining[n_test:]]]),
            np.concatenate([existing_train_y, y[remaining[n_test:]]]),
        )
        client_test[last_cid] = (
            np.concatenate([existing_test_X, X[remaining[:n_test]]]),
            np.concatenate([existing_test_y, y[remaining[:n_test]]]),
        )

    return client_train, client_test, site_assignment


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

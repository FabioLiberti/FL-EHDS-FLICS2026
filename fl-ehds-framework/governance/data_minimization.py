"""
Data Minimization Module (EHDS Article 44)
==========================================
Enforces the principle of data minimization for FL training:
only features strictly necessary for the declared purpose
are included in the training dataset.

Provides:
- Feature importance calculation (mutual information)
- Purpose-based feature group filtering
- Minimization report generation for audit trail
"""

import logging
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class DataMinimizer:
    """
    EHDS Article 44 data minimization enforcement.

    Two-level filtering:
    1. Purpose-based: only feature groups relevant to the declared purpose
    2. Importance-based: within allowed groups, exclude features with
       mutual information below threshold
    """

    # Feature groups allowed per purpose (None = all features allowed)
    PURPOSE_FEATURE_PROFILES: Dict[str, Optional[Set[str]]] = {
        "scientific_research": None,
        "ai_system_development": None,
        "public_health_surveillance": {
            "demographics", "conditions", "vitals", "medications",
        },
        "health_policy": {"demographics", "conditions", "visits"},
        "official_statistics": {"demographics"},
        "education_training": None,
        "personalized_medicine": None,
        "patient_safety": {
            "conditions", "medications", "vitals", "measurements",
        },
    }

    # Map feature names -> semantic groups
    # Covers: FHIR 10 features, OMOP ~36 features, synthetic 10 features
    FEATURE_GROUPS: Dict[str, str] = {
        # Demographics
        "age": "demographics",
        "gender": "demographics",
        "gender_male": "demographics",
        "gender_female": "demographics",
        "bmi": "demographics",
        # Vitals
        "systolic_bp": "demographics",
        "diastolic_bp": "demographics",
        "heart_rate": "vitals",
        "glucose": "vitals",
        "cholesterol": "vitals",
        # Conditions (FHIR)
        "num_conditions": "conditions",
        # Medications (FHIR)
        "num_medications": "medications",
        # OMOP condition features
        "n_conditions_30d": "conditions",
        "n_conditions_90d": "conditions",
        "n_conditions_365d": "conditions",
        "n_unique_conditions": "conditions",
        "has_diabetes": "conditions",
        "has_hypertension": "conditions",
        "has_heart_failure": "conditions",
        "has_copd": "conditions",
        "has_ckd": "conditions",
        # OMOP drug features
        "n_drugs_30d": "medications",
        "n_drugs_90d": "medications",
        "n_drugs_365d": "medications",
        "n_unique_drugs": "medications",
        "total_days_supply": "medications",
        # OMOP measurement features
        "n_measurements_30d": "measurements",
        "n_measurements_90d": "measurements",
        "n_abnormal_measurements": "measurements",
        "last_glucose_blood": "measurements",
        "last_hba1c": "measurements",
        "last_creatinine": "measurements",
        "last_hemoglobin": "measurements",
        "last_systolic_bp": "measurements",
        "last_diastolic_bp": "measurements",
        "last_bmi": "measurements",
        # OMOP visit features
        "n_visits_30d": "visits",
        "n_visits_90d": "visits",
        "n_visits_365d": "visits",
        "n_inpatient_visits": "visits",
        "n_er_visits": "visits",
        "total_los_days": "visits",
        # OMOP procedure features
        "n_procedures_30d": "procedures",
        "n_procedures_90d": "procedures",
        "n_procedures_365d": "procedures",
    }

    @staticmethod
    def get_purpose_allowed_groups(purpose: str) -> Optional[Set[str]]:
        """
        Return set of feature groups allowed for the given purpose.
        None means all features are allowed.
        """
        return DataMinimizer.PURPOSE_FEATURE_PROFILES.get(purpose)

    @staticmethod
    def compute_feature_importance(
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Compute feature importance using mutual information.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Label array (n_samples,)
            feature_names: Optional names for features

        Returns:
            Dict mapping feature name/index to importance score
        """
        try:
            from sklearn.feature_selection import mutual_info_classif
        except ImportError:
            logger.warning("sklearn not available, using variance as importance proxy")
            variances = np.var(X, axis=0)
            total = variances.sum()
            if total == 0:
                total = 1.0
            scores = variances / total
            names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]
            return {n: float(s) for n, s in zip(names, scores)}

        mi_scores = mutual_info_classif(X, y, random_state=42)

        names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]
        return {n: float(s) for n, s in zip(names, mi_scores)}

    @staticmethod
    def apply_minimization(
        train_data: Dict[int, tuple],
        test_data: Optional[Dict[int, tuple]],
        purpose: str,
        feature_names: Optional[List[str]] = None,
        importance_threshold: float = 0.01,
    ) -> Tuple[Dict[int, tuple], Optional[Dict[int, tuple]], Dict[str, Any]]:
        """
        Apply data minimization to federated training data.

        Two-phase filtering:
        1. Purpose-based: remove feature groups not relevant to purpose
        2. Importance-based: remove features with MI < threshold

        Args:
            train_data: {client_id: (X, y)} training data
            test_data: {client_id: (X, y)} test data (optional)
            purpose: EHDS Article 53 purpose string
            feature_names: Feature names matching X columns
            importance_threshold: MI threshold for feature selection

        Returns:
            (filtered_train, filtered_test, minimization_report)
        """
        if not train_data:
            return train_data, test_data, {"status": "no_data"}

        # Get feature count from first client
        first_client = next(iter(train_data.values()))
        n_features = first_client[0].shape[1]
        names = feature_names or [f"feature_{i}" for i in range(n_features)]

        # Phase 1: Purpose-based group filtering
        allowed_groups = DataMinimizer.get_purpose_allowed_groups(purpose)

        if allowed_groups is not None:
            # Filter by group membership
            keep_indices = []
            for i, name in enumerate(names):
                group = DataMinimizer.FEATURE_GROUPS.get(name, "other")
                if group in allowed_groups or group == "other":
                    keep_indices.append(i)
            purpose_removed = [
                names[i] for i in range(n_features) if i not in keep_indices
            ]
        else:
            keep_indices = list(range(n_features))
            purpose_removed = []

        if not keep_indices:
            logger.warning("Purpose filtering removed all features, keeping all")
            keep_indices = list(range(n_features))
            purpose_removed = []

        # Phase 2: Importance-based filtering (on purpose-filtered features)
        # Combine all client data for importance calculation
        all_X = np.vstack([train_data[cid][0][:, keep_indices] for cid in train_data])
        all_y = np.concatenate([train_data[cid][1] for cid in train_data])

        filtered_names = [names[i] for i in keep_indices]
        importance_scores = DataMinimizer.compute_feature_importance(
            all_X, all_y, filtered_names
        )

        # Keep features above threshold
        final_indices = []
        importance_removed = []
        for j, idx in enumerate(keep_indices):
            fname = names[idx]
            if importance_scores.get(fname, 0.0) >= importance_threshold:
                final_indices.append(idx)
            else:
                importance_removed.append(
                    (fname, importance_scores.get(fname, 0.0))
                )

        if not final_indices:
            logger.warning("Importance filtering removed all features, keeping top 3")
            sorted_by_importance = sorted(
                keep_indices,
                key=lambda i: importance_scores.get(names[i], 0.0),
                reverse=True,
            )
            final_indices = sorted_by_importance[:3]
            importance_removed = []

        # Apply filtering to all clients
        filtered_train = {}
        for cid, (X, y) in train_data.items():
            filtered_train[cid] = (X[:, final_indices], y)

        filtered_test = None
        if test_data:
            filtered_test = {}
            for cid, (X, y) in test_data.items():
                filtered_test[cid] = (X[:, final_indices], y)

        # Build report
        final_names = [names[i] for i in final_indices]
        report = {
            "original_features": n_features,
            "kept_features": len(final_indices),
            "kept_feature_names": final_names,
            "purpose": purpose,
            "allowed_groups": list(allowed_groups) if allowed_groups else "all",
            "purpose_removed": purpose_removed,
            "importance_removed": [
                {"name": n, "score": f"{s:.4f}"} for n, s in importance_removed
            ],
            "importance_threshold": importance_threshold,
            "importance_scores": {
                k: round(v, 4) for k, v in importance_scores.items()
            },
            "reduction_pct": round(
                (1 - len(final_indices) / n_features) * 100, 1
            ),
        }

        logger.info(
            f"Data minimization: {n_features} -> {len(final_indices)} features "
            f"({report['reduction_pct']}% reduction)"
        )

        return filtered_train, filtered_test, report

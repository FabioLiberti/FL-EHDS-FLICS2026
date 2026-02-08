"""
Data Quality Framework for Federated Learning (EHDS Art. 69)
============================================================

Provides quality-weighted aggregation, pre-training anomaly detection,
and EHDS-compliant quality label assignment for federated learning clients.

Reuses DataQualityAssessor from core/incentive_mechanisms.py (zero duplication).

Quality-Weighted Aggregation Formula:
    w_h = (n_h / N) * quality_h^alpha, then normalized so sum(w) = 1.0

    alpha = 0: quality ignored (backward compatible, pure FedAvg weighting)
    alpha = 1: linear quality influence (default)
    alpha = 2: strong quality penalty

EHDS Art. 69 Quality Labels:
    GOLD:         overall_score >= 0.85
    SILVER:       overall_score >= 0.70
    BRONZE:       overall_score >= 0.55
    INSUFFICIENT: overall_score <  0.55
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from scipy import stats as sp_stats
except ImportError:
    sp_stats = None

from core.incentive_mechanisms import DataQualityAssessor, DataQualityDimension

logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class EHDSQualityLabel(Enum):
    """EHDS Art. 69 quality label tiers."""
    GOLD = "gold"
    SILVER = "silver"
    BRONZE = "bronze"
    INSUFFICIENT = "insufficient"


@dataclass
class ClientQualityReport:
    """Per-client quality assessment report."""
    client_id: int
    hospital_name: str
    # Dimension scores (4 from DataQualityAssessor + 1 new)
    completeness: float = 0.0
    accuracy: float = 0.0
    uniqueness: float = 0.0
    diversity: float = 0.0
    consistency: float = 0.0
    # Overall
    overall_score: float = 0.0
    quality_label: EHDSQualityLabel = EHDSQualityLabel.INSUFFICIENT
    # Aggregation weight modifier
    quality_weight: float = 1.0
    # Anomaly detection results
    anomalies: List[str] = field(default_factory=list)
    is_anomalous: bool = False
    # Metadata
    num_samples: int = 0
    num_classes: int = 0
    assessed_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "client_id": self.client_id,
            "hospital_name": self.hospital_name,
            "completeness": round(self.completeness, 4),
            "accuracy": round(self.accuracy, 4),
            "uniqueness": round(self.uniqueness, 4),
            "diversity": round(self.diversity, 4),
            "consistency": round(self.consistency, 4),
            "overall_score": round(self.overall_score, 4),
            "quality_label": self.quality_label.value,
            "quality_weight": round(self.quality_weight, 4),
            "anomalies": self.anomalies,
            "is_anomalous": self.is_anomalous,
            "num_samples": self.num_samples,
            "num_classes": self.num_classes,
            "assessed_at": self.assessed_at,
        }


# =============================================================================
# DATA QUALITY MANAGER
# =============================================================================

class DataQualityManager:
    """
    EHDS Art. 69 Data Quality Framework for Federated Learning.

    Provides:
    - Per-client quality assessment across 5 dimensions
    - EHDS quality label assignment (Art. 69)
    - Quality-weighted aggregation weight computation
    - Pre-training anomaly detection (statistical tests)
    - Comprehensive quality reports for compliance audit

    Reuses DataQualityAssessor from core/incentive_mechanisms.py.
    """

    def __init__(
        self,
        hospitals: list,
        config: Dict[str, Any],
    ):
        self.hospitals = hospitals
        self.config = config
        self.alpha = config.get("alpha", 1.0)

        # Dimension weights (5 dimensions, EHDS-aligned)
        self.dimension_weights = {
            "completeness": config.get("weight_completeness", 0.20),
            "accuracy": config.get("weight_accuracy", 0.25),
            "uniqueness": config.get("weight_uniqueness", 0.15),
            "diversity": config.get("weight_diversity", 0.20),
            "consistency": config.get("weight_consistency", 0.20),
        }

        # Reuse existing DataQualityAssessor (4 dimensions)
        self._assessor = DataQualityAssessor(dimension_weights={
            DataQualityDimension.COMPLETENESS: 0.25,
            DataQualityDimension.ACCURACY: 0.30,
            DataQualityDimension.UNIQUENESS: 0.20,
            DataQualityDimension.DIVERSITY: 0.25,
        })

        # Anomaly detection thresholds
        self._ks_threshold = config.get("ks_threshold", 0.05)
        self._missing_threshold = config.get("missing_threshold", 0.30)
        self._entropy_threshold = config.get("entropy_threshold", 0.30)
        self._iqr_multiplier = config.get("iqr_multiplier", 3.0)

        # Quality label thresholds (EHDS Art. 69)
        self._label_thresholds = {
            EHDSQualityLabel.GOLD: config.get("gold_threshold", 0.85),
            EHDSQualityLabel.SILVER: config.get("silver_threshold", 0.70),
            EHDSQualityLabel.BRONZE: config.get("bronze_threshold", 0.55),
        }

        # State
        self.client_reports: Dict[int, ClientQualityReport] = {}
        self._global_feature_stats: Optional[Dict[str, Any]] = None

        # Hospital name lookup
        self._hospital_names = {}
        for h in hospitals:
            self._hospital_names[h.hospital_id] = h.name

    # -------------------------------------------------------------------------
    # PUBLIC API
    # -------------------------------------------------------------------------

    def assess_all_clients(
        self,
        client_data: Dict[int, Tuple[np.ndarray, np.ndarray]],
    ) -> Dict[int, ClientQualityReport]:
        """
        Pre-training: assess quality for all clients.

        Called ONCE before the training loop begins.

        Args:
            client_data: {client_id: (X, y)} -- training data arrays

        Returns:
            Dict of client_id -> ClientQualityReport
        """
        # Step 1: Compute global feature statistics (for anomaly detection)
        self._compute_global_stats(client_data)

        # Step 2: Assess each client
        for client_id, (X, y) in client_data.items():
            report = self._assess_client(client_id, X, y)
            self.client_reports[client_id] = report

        # Log summary
        scores = [r.overall_score for r in self.client_reports.values()]
        anomalous = sum(1 for r in self.client_reports.values() if r.is_anomalous)
        logger.info(
            f"Data quality assessment: {len(self.client_reports)} clients, "
            f"mean={np.mean(scores):.3f}, std={np.std(scores):.3f}, "
            f"anomalous={anomalous}"
        )

        return self.client_reports

    def get_quality_weights(self) -> Dict[int, float]:
        """
        Return quality-based aggregation weights for all assessed clients.

        Returns:
            {client_id: quality_weight} where quality_weight = overall_score^alpha
        """
        return {
            cid: report.quality_weight
            for cid, report in self.client_reports.items()
        }

    def export_report(self) -> Dict[str, Any]:
        """Export comprehensive quality report for auto-save (JSON)."""
        label_counts: Dict[str, int] = {}
        for report in self.client_reports.values():
            lbl = report.quality_label.value
            label_counts[lbl] = label_counts.get(lbl, 0) + 1

        scores = [r.overall_score for r in self.client_reports.values()]

        return {
            "framework": "EHDS Art. 69 Data Quality",
            "assessed_at": datetime.now().isoformat(),
            "config": {
                "alpha": self.alpha,
                "dimension_weights": self.dimension_weights,
                "label_thresholds": {
                    k.value: v for k, v in self._label_thresholds.items()
                },
                "anomaly_detection": {
                    "ks_threshold": self._ks_threshold,
                    "missing_threshold": self._missing_threshold,
                    "entropy_threshold": self._entropy_threshold,
                    "iqr_multiplier": self._iqr_multiplier,
                },
            },
            "per_client": {
                str(cid): report.to_dict()
                for cid, report in self.client_reports.items()
            },
            "summary": {
                "total_clients": len(self.client_reports),
                "mean_quality": float(np.mean(scores)) if scores else 0.0,
                "std_quality": float(np.std(scores)) if scores else 0.0,
                "min_quality": float(np.min(scores)) if scores else 0.0,
                "max_quality": float(np.max(scores)) if scores else 0.0,
                "anomalous_clients": sum(
                    1 for r in self.client_reports.values() if r.is_anomalous
                ),
                "label_distribution": label_counts,
            },
        }

    def get_display_summary(self) -> Dict[str, Any]:
        """Summary dict for terminal display."""
        scores = [r.overall_score for r in self.client_reports.values()]
        label_counts: Dict[str, int] = {}
        for report in self.client_reports.values():
            lbl = report.quality_label.value
            label_counts[lbl] = label_counts.get(lbl, 0) + 1

        return {
            "total_clients": len(self.client_reports),
            "mean_quality": float(np.mean(scores)) if scores else 0.0,
            "std_quality": float(np.std(scores)) if scores else 0.0,
            "anomalous_clients": sum(
                1 for r in self.client_reports.values() if r.is_anomalous
            ),
            "label_distribution": label_counts,
            "alpha": self.alpha,
        }

    # -------------------------------------------------------------------------
    # INTERNAL: ASSESSMENT
    # -------------------------------------------------------------------------

    def _compute_global_stats(
        self,
        client_data: Dict[int, Tuple[np.ndarray, np.ndarray]],
    ) -> None:
        """Compute pooled feature statistics across all clients."""
        all_X_list = []
        all_y_list = []
        for X, y in client_data.values():
            all_X_list.append(X)
            all_y_list.append(y)

        all_X = np.concatenate(all_X_list, axis=0)
        all_y = np.concatenate(all_y_list)

        unique_vals, counts = np.unique(all_y, return_counts=True)

        self._global_feature_stats = {
            "means": np.nanmean(all_X, axis=0),
            "stds": np.nanstd(all_X, axis=0),
            "q1": np.nanpercentile(all_X, 25, axis=0),
            "q3": np.nanpercentile(all_X, 75, axis=0),
            "all_X": all_X,
            "class_distribution": dict(zip(unique_vals.tolist(), counts.tolist())),
        }

    def _assess_client(
        self,
        client_id: int,
        X: np.ndarray,
        y: np.ndarray,
    ) -> ClientQualityReport:
        """Assess a single client's data quality across 5 dimensions."""
        # Use existing DataQualityAssessor for 4 dimensions
        _, dim_scores = self._assessor.assess(X, y)

        completeness = dim_scores.get("completeness", 1.0)
        accuracy = dim_scores.get("accuracy", 0.85)
        uniqueness = dim_scores.get("uniqueness", 1.0)
        diversity = dim_scores.get("diversity", 0.5)

        # NEW: Consistency metric (5th dimension)
        consistency = self._compute_consistency(X)

        # Compute overall score with 5 weighted dimensions
        overall = (
            self.dimension_weights["completeness"] * completeness
            + self.dimension_weights["accuracy"] * accuracy
            + self.dimension_weights["uniqueness"] * uniqueness
            + self.dimension_weights["diversity"] * diversity
            + self.dimension_weights["consistency"] * consistency
        )

        # Assign EHDS quality label
        label = self._assign_quality_label(overall)

        # Compute aggregation weight modifier: score^alpha
        if overall > 0:
            quality_weight = overall ** self.alpha
        else:
            quality_weight = 0.0

        # Run anomaly detection
        anomalies = self._detect_anomalies(client_id, X, y)

        return ClientQualityReport(
            client_id=client_id,
            hospital_name=self._hospital_names.get(
                client_id, f"Client_{client_id}"
            ),
            completeness=completeness,
            accuracy=accuracy,
            uniqueness=uniqueness,
            diversity=diversity,
            consistency=consistency,
            overall_score=overall,
            quality_label=label,
            quality_weight=quality_weight,
            anomalies=anomalies,
            is_anomalous=len(anomalies) > 0,
            num_samples=len(X),
            num_classes=len(np.unique(y)),
            assessed_at=datetime.now().isoformat(),
        )

    def _compute_consistency(self, X: np.ndarray) -> float:
        """
        Consistency metric: fraction of feature values within expected ranges.

        Compares each client's feature values against the global pool
        (3-sigma rule). Catches data entry errors, unit mismatches,
        and sensor calibration differences between hospitals.

        Returns:
            Score in [0, 1] where 1 = fully consistent.
        """
        if X.size == 0:
            return 0.0

        if self._global_feature_stats is not None:
            means = self._global_feature_stats["means"]
            stds = self._global_feature_stats["stds"]

            within_range = 0.0
            total_checks = 0

            for j in range(min(X.shape[1], len(means))):
                if stds[j] > 1e-10:
                    lower = means[j] - 3 * stds[j]
                    upper = means[j] + 3 * stds[j]
                    col = X[:, j]
                    valid = col[~np.isnan(col)]
                    if len(valid) > 0:
                        fraction_in_range = float(
                            np.mean((valid >= lower) & (valid <= upper))
                        )
                        within_range += fraction_in_range
                        total_checks += 1

            if total_checks > 0:
                return within_range / total_checks

        # Fallback: coefficient of variation as proxy
        stds_local = np.nanstd(X, axis=0)
        means_local = np.nanmean(X, axis=0)
        cv = np.divide(
            stds_local, np.abs(means_local) + 1e-10
        )
        return float(np.clip(1.0 - np.mean(cv) / 2.0, 0, 1))

    def _assign_quality_label(self, overall_score: float) -> EHDSQualityLabel:
        """Assign EHDS Art. 69 quality label based on overall score."""
        if overall_score >= self._label_thresholds[EHDSQualityLabel.GOLD]:
            return EHDSQualityLabel.GOLD
        elif overall_score >= self._label_thresholds[EHDSQualityLabel.SILVER]:
            return EHDSQualityLabel.SILVER
        elif overall_score >= self._label_thresholds[EHDSQualityLabel.BRONZE]:
            return EHDSQualityLabel.BRONZE
        else:
            return EHDSQualityLabel.INSUFFICIENT

    # -------------------------------------------------------------------------
    # INTERNAL: ANOMALY DETECTION
    # -------------------------------------------------------------------------

    def _detect_anomalies(
        self,
        client_id: int,
        X: np.ndarray,
        y: np.ndarray,
    ) -> List[str]:
        """
        Pre-training anomaly detection for a client.

        Runs 4 statistical tests:
        1. Missing value rate check
        2. Class imbalance (entropy) check
        3. KS test for feature distribution shift vs global
        4. IQR-based outlier detection on feature means

        Returns list of anomaly descriptions (empty = no anomalies).
        """
        anomalies = []

        # 1. Missing value check
        missing_rate = float(np.mean(np.isnan(X)))
        if missing_rate > self._missing_threshold:
            anomalies.append(
                f"High missing rate: {missing_rate:.1%} > "
                f"{self._missing_threshold:.1%}"
            )

        # 2. Class imbalance check (normalized entropy)
        unique_classes, counts = np.unique(y, return_counts=True)
        if len(counts) > 1:
            probs = counts / counts.sum()
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            max_entropy = np.log(len(unique_classes))
            norm_entropy = entropy / max_entropy if max_entropy > 0 else 0
            if norm_entropy < self._entropy_threshold:
                anomalies.append(
                    f"Low class entropy: {norm_entropy:.3f} < "
                    f"{self._entropy_threshold}"
                )
        elif len(counts) == 1:
            anomalies.append(
                f"Single class only: all samples belong to class "
                f"{unique_classes[0]}"
            )

        # 3. Feature distribution test (KS test vs global)
        if (
            self._global_feature_stats is not None
            and sp_stats is not None
        ):
            all_X = self._global_feature_stats["all_X"]
            ks_failures = 0
            ks_tested = 0
            for j in range(min(X.shape[1], all_X.shape[1])):
                client_col = X[:, j][~np.isnan(X[:, j])]
                global_col = all_X[:, j][~np.isnan(all_X[:, j])]
                if len(client_col) > 5 and len(global_col) > 5:
                    _, p_val = sp_stats.ks_2samp(client_col, global_col)
                    ks_tested += 1
                    if p_val < self._ks_threshold:
                        ks_failures += 1

            if ks_tested > 0 and ks_failures / ks_tested > 0.5:
                anomalies.append(
                    f"Distribution shift: {ks_failures}/{ks_tested} features "
                    f"differ from global (KS test p<{self._ks_threshold})"
                )

        # 4. Outlier detection (IQR on feature means)
        if self._global_feature_stats is not None:
            q1 = self._global_feature_stats["q1"]
            q3 = self._global_feature_stats["q3"]
            iqr = q3 - q1
            client_means = np.nanmean(X, axis=0)

            lower_bound = q1 - self._iqr_multiplier * iqr
            upper_bound = q3 + self._iqr_multiplier * iqr

            outlier_features = 0
            for j in range(min(len(client_means), len(iqr))):
                if iqr[j] > 1e-10:
                    if (
                        client_means[j] < lower_bound[j]
                        or client_means[j] > upper_bound[j]
                    ):
                        outlier_features += 1

            if outlier_features > 0:
                anomalies.append(
                    f"Feature mean outliers: {outlier_features} features "
                    f"outside {self._iqr_multiplier}x IQR"
                )

        return anomalies

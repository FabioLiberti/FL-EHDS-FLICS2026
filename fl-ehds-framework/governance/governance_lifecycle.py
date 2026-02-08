"""
EHDS Chapter IV Governance Lifecycle Bridge.

Orchestrates the full HDAB data permit lifecycle for cross-border
federated learning, integrating existing governance modules:

- hdab_integration.py: HDAB connect, authenticate, request permits
- data_permits.py: Permit validation (purpose, expiry, categories)
- permit_training.py: Per-round budget tracking + audit logging
- data_minimization.py: Purpose-based feature selection (Art. 44)
- compliance_logging.py: GDPR Art. 30 audit trail

Two-phase lifecycle:
    Phase 1 (pre_training):  HDAB connect -> permits -> minimization
    Phase 2 (training loop): validate_round -> log_round -> end_session

Author: Fabio Liberti
"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Purpose string -> PermitPurpose enum mapping
PURPOSE_MAP = {
    "scientific_research": "scientific_research",
    "public_health_surveillance": "public_health_surveillance",
    "health_policy": "health_policy",
    "education_training": "education_training",
    "ai_system_development": "ai_system_development",
    "personalized_medicine": "personalized_medicine",
    "official_statistics": "official_statistics",
    "patient_safety": "patient_safety",
}

# Dataset type -> DataCategory value mapping
DATASET_CATEGORY_MAP = {
    "synthetic": "ehr",
    "imaging": "imaging",
    "fhir": "ehr",
    "tabular": "ehr",
    "genomic": "genomic",
}


class GovernanceLifecycleBridge:
    """
    Full EHDS Chapter IV governance lifecycle bridge.

    Coordinates HDAB simulation, permit issuance, purpose validation,
    data minimization, and compliance audit for cross-border FL training.

    Two-phase lifecycle:
    - Phase 1 (pre_training): HDAB connect, permits, minimization
    - Phase 2 (training loop): validate_round, log_round, end_session
    """

    def __init__(
        self,
        hospitals: List,
        countries: List[str],
        purpose: str,
        global_epsilon: float,
        num_rounds: int,
        config: Dict[str, Any],
        seed: int = 42,
    ):
        self._hospitals = hospitals
        self._countries = countries
        self._purpose = purpose
        self._global_epsilon = global_epsilon
        self._num_rounds = num_rounds
        self._config = config
        self._seed = seed

        # Sub-components (created lazily in pre_training)
        self._hdab_coordinator = None
        self._permit_context = None
        self._permits: Dict[str, Any] = {}
        self._minimization_report: Optional[Dict] = None
        self._purpose_violations: List[str] = []
        self._hdab_status: Dict[str, bool] = {}
        self._session_started = False

    # -----------------------------------------------------------------
    # ASYNC WRAPPER
    # -----------------------------------------------------------------

    @staticmethod
    def _run_async(coro):
        """Run async coroutine synchronously."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                raise RuntimeError("closed loop")
            if loop.is_running():
                # Running inside existing loop (e.g. Jupyter)
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    return pool.submit(asyncio.run, coro).result()
            return loop.run_until_complete(coro)
        except RuntimeError:
            return asyncio.run(coro)

    # -----------------------------------------------------------------
    # PHASE 1: PRE-TRAINING
    # -----------------------------------------------------------------

    def pre_training(
        self,
        train_data: Optional[Dict[int, tuple]] = None,
        test_data: Optional[Dict[int, tuple]] = None,
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Execute all pre-training governance steps.

        1. Connect to simulated HDABs
        2. Request cross-border permits
        3. Verify permits
        4. Apply data minimization (if enabled + tabular data)
        5. Start compliance audit session

        Returns:
            Dict with: minimized_train, minimized_test, input_dim,
            feature_names, minimization_report, permits, hdab_status,
            purpose_violations
        """
        from core.models import PermitPurpose, DataCategory
        from governance.hdab_integration import HDABConfig, MultiHDABCoordinator

        result = {
            "minimized_train": None,
            "minimized_test": None,
            "input_dim": None,
            "feature_names": feature_names,
            "minimization_report": None,
            "permits": {},
            "hdab_status": {},
            "purpose_violations": [],
        }

        # --- Step 1: Map purpose and categories ---
        purpose_str = PURPOSE_MAP.get(self._purpose, self._purpose)
        try:
            purpose_enum = PermitPurpose(purpose_str)
        except ValueError:
            self._purpose_violations.append(
                f"Purpose '{self._purpose}' is not a valid EHDS Art. 53 purpose"
            )
            result["purpose_violations"] = self._purpose_violations
            return result

        ds_type = self._config.get("dataset_type", "synthetic")
        category_str = DATASET_CATEGORY_MAP.get(ds_type, "ehr")
        try:
            data_category = DataCategory(category_str)
        except ValueError:
            data_category = DataCategory.EHR

        # --- Step 2: Create HDAB coordinator and connect ---
        auth_method = self._config.get("hdab_auth_method", "oauth2")
        validity_days = self._config.get("permit_validity_days", 365)

        hdab_configs = {}
        for cc in set(self._countries):
            hdab_configs[cc] = HDABConfig(
                endpoint=f"https://hdab.{cc.lower()}.ehds.europa.eu/api/v1",
                auth_method=auth_method,
                simulation_mode=True,
            )

        self._hdab_coordinator = MultiHDABCoordinator(hdab_configs)
        self._hdab_status = self._run_async(
            self._hdab_coordinator.connect_all()
        )
        result["hdab_status"] = self._hdab_status

        connected = [cc for cc, ok in self._hdab_status.items() if ok]
        if not connected:
            self._purpose_violations.append(
                "Failed to connect to any HDAB"
            )
            result["purpose_violations"] = self._purpose_violations
            return result

        logger.info(
            f"HDAB connected: {len(connected)}/{len(self._countries)} countries"
        )

        # --- Step 3: Request cross-border permits ---
        permits = self._run_async(
            self._hdab_coordinator.request_cross_border_permits(
                requester_id="fl-ehds-framework",
                purpose=purpose_enum,
                data_categories=[data_category],
                validity_days=validity_days,
            )
        )
        self._permits = permits
        result["permits"] = {
            cc: {
                "permit_id": p.permit_id,
                "status": p.status.value if hasattr(p.status, 'value') else str(p.status),
                "purpose": p.purpose.value if hasattr(p.purpose, 'value') else str(p.purpose),
                "valid_until": str(p.valid_until),
            }
            for cc, p in permits.items()
        }

        if not permits:
            self._purpose_violations.append(
                "No permits could be issued by any HDAB"
            )
            result["purpose_violations"] = self._purpose_violations
            return result

        logger.info(
            f"Permits issued: {len(permits)} for purpose={self._purpose}"
        )

        # --- Step 4: Verify permits ---
        permit_ids = {cc: p.permit_id for cc, p in permits.items()}
        verification = self._run_async(
            self._hdab_coordinator.verify_cross_border_permits(
                permit_ids=permit_ids,
                purpose=purpose_enum,
                data_categories=[data_category],
            )
        )
        failed_states = [cc for cc, ok in verification.items() if not ok]
        if failed_states:
            for cc in failed_states:
                self._purpose_violations.append(
                    f"HDAB {cc}: permit verification failed for purpose={self._purpose}"
                )

        # --- Step 5: Data Minimization (Art. 44) ---
        minimization_enabled = self._config.get(
            "data_minimization_enabled", False
        )
        if minimization_enabled and train_data is not None:
            from governance.data_minimization import DataMinimizer

            threshold = self._config.get("importance_threshold", 0.01)
            filtered_train, filtered_test, min_report = (
                DataMinimizer.apply_minimization(
                    train_data=train_data,
                    test_data=test_data,
                    purpose=self._purpose,
                    feature_names=feature_names,
                    importance_threshold=threshold,
                )
            )
            self._minimization_report = min_report
            result["minimized_train"] = filtered_train
            result["minimized_test"] = filtered_test
            result["input_dim"] = min_report["kept_features"]
            result["feature_names"] = min_report["kept_feature_names"]
            result["minimization_report"] = min_report

            logger.info(
                f"Data minimization: {min_report['original_features']} -> "
                f"{min_report['kept_features']} features "
                f"(-{min_report['reduction_pct']}%)"
            )

        # --- Step 6: Start Permit-Aware Training Context ---
        # Use the first available permit for the training context
        first_permit = next(iter(permits.values()))
        from governance.permit_training import PermitAwareTrainingContext

        client_ids = [
            f"client_{h.hospital_id}" for h in self._hospitals
        ]
        self._permit_context = PermitAwareTrainingContext(
            permit_id=first_permit.permit_id,
            purpose=purpose_enum,
            data_categories=[data_category],
            privacy_budget_total=self._global_epsilon,
            max_rounds=self._num_rounds,
            client_ids=client_ids,
        )
        self._permit_context.start_session()
        self._session_started = True

        result["purpose_violations"] = self._purpose_violations
        return result

    # -----------------------------------------------------------------
    # PHASE 2: TRAINING LOOP
    # -----------------------------------------------------------------

    def validate_round(
        self, round_num: int, epsilon_cost: float
    ) -> Tuple[bool, str]:
        """Pre-round governance check (permit + budget)."""
        if self._permit_context is None:
            return True, "No permit context (governance not initialized)"
        return self._permit_context.validate_round(round_num, epsilon_cost)

    def log_round_completion(
        self, round_num: int, round_result, epsilon_spent: float
    ):
        """Post-round logging to compliance audit trail."""
        if self._permit_context is None:
            return
        self._permit_context.log_round_completion(round_result, epsilon_spent)

    # -----------------------------------------------------------------
    # SESSION END
    # -----------------------------------------------------------------

    def end_session(
        self,
        total_rounds: int,
        final_metrics: Dict[str, float],
        success: bool = True,
    ):
        """
        End governance session.

        1. End permit-aware training context (flush audit)
        2. Disconnect from all HDABs
        """
        if self._permit_context and self._session_started:
            self._permit_context.end_session(
                total_rounds=total_rounds,
                final_metrics=final_metrics,
                success=success,
            )

        if self._hdab_coordinator:
            try:
                self._run_async(self._hdab_coordinator.disconnect_all())
            except Exception as e:
                logger.warning(f"Error disconnecting HDABs: {e}")

    # -----------------------------------------------------------------
    # REPORTS
    # -----------------------------------------------------------------

    def export_report(self) -> Dict[str, Any]:
        """Export complete governance lifecycle report."""
        report = {
            "report_type": "EHDS Chapter IV Governance Lifecycle",
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "purpose": self._purpose,
            "countries": self._countries,
            "hdab_status": self._hdab_status,
            "permits": self.get_permits_summary(),
            "purpose_violations": self._purpose_violations,
        }

        if self._permit_context:
            report["budget_status"] = self.get_budget_status()

        if self._minimization_report:
            report["minimization"] = self._minimization_report

        if self._hdab_coordinator:
            report["coordination_log"] = [
                entry for entry in self._hdab_coordinator.coordination_log
            ]

        return report

    def get_budget_status(self) -> Dict[str, Any]:
        """Return privacy budget status from permit context."""
        if self._permit_context is None:
            return {
                "total": self._global_epsilon,
                "used": 0.0,
                "remaining": self._global_epsilon,
                "utilization_pct": 0.0,
                "rounds_completed": 0,
                "max_rounds": self._num_rounds,
            }
        return self._permit_context.get_budget_status()

    def get_permits_summary(self) -> Dict[str, Any]:
        """Summary of all permits: per-country status."""
        per_country = {}
        for cc, permit in self._permits.items():
            per_country[cc] = {
                "permit_id": permit.permit_id,
                "status": (
                    permit.status.value
                    if hasattr(permit.status, "value")
                    else str(permit.status)
                ),
                "purpose": (
                    permit.purpose.value
                    if hasattr(permit.purpose, "value")
                    else str(permit.purpose)
                ),
                "valid_until": str(permit.valid_until),
            }
        return {
            "total_permits": len(self._permits),
            "per_country": per_country,
        }

    def get_minimization_report(self) -> Optional[Dict[str, Any]]:
        """Return minimization report if applied."""
        return self._minimization_report

    def export_audit_log(self, output_dir: str):
        """Export compliance audit log to output directory."""
        if self._permit_context:
            self._permit_context.export_audit_log(output_dir)

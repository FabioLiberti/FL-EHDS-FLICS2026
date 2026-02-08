"""
Cross-Border Federated Learning Simulation for EHDS.

Simulates federated learning across European hospitals in different
EU member states, each with distinct regulatory constraints:
- Per-country differential privacy requirements
- Network latency proportional to geographic distance
- HDAB national policy enforcement
- Opt-out rates and data retention rules
- Jurisdiction-aware audit trail

This module is the key differentiator of FL-EHDS: it models the
real-world regulatory heterogeneity of cross-border health data use
under EHDS Regulation (EU) 2025/327.
"""

import csv
import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# =============================================================================
# EU COUNTRY PROFILES
# =============================================================================

@dataclass
class EUCountryProfile:
    """Regulatory and network profile for an EU member state."""
    code: str                    # ISO 3166-1 alpha-2
    name: str                    # Full country name
    # Privacy constraints
    dp_epsilon_max: float        # Maximum allowed epsilon (lower = stricter)
    dp_delta_max: float          # Maximum allowed delta
    min_aggregation_count: int   # Minimum clients before aggregation
    # HDAB policy
    hdab_strictness: int         # 1-5 scale (5 = strictest)
    allowed_purposes: List[str]  # EHDS Art. 53 purposes allowed
    data_retention_days: int     # Maximum model retention
    # Population characteristics
    opt_out_rate: float          # Expected opt-out percentage (0.0-1.0)
    # Network
    latency_ms: Tuple[int, int]  # (min_ms, max_ms) simulated round-trip
    # Geographic center (for distance calc)
    lat: float
    lon: float


# Realistic EU country profiles based on GDPR enforcement history,
# healthcare data culture, and EHDS preparedness reports
EU_COUNTRY_PROFILES: Dict[str, EUCountryProfile] = {
    "DE": EUCountryProfile(
        code="DE", name="Germany",
        dp_epsilon_max=1.0, dp_delta_max=1e-6,
        min_aggregation_count=5, hdab_strictness=5,
        allowed_purposes=["scientific_research", "public_health_surveillance"],
        data_retention_days=365,
        opt_out_rate=0.12,
        latency_ms=(15, 35), lat=51.2, lon=10.4,
    ),
    "FR": EUCountryProfile(
        code="FR", name="France",
        dp_epsilon_max=3.0, dp_delta_max=1e-5,
        min_aggregation_count=3, hdab_strictness=4,
        allowed_purposes=["scientific_research", "public_health_surveillance", "health_policy"],
        data_retention_days=730,
        opt_out_rate=0.08,
        latency_ms=(12, 30), lat=46.6, lon=2.2,
    ),
    "IT": EUCountryProfile(
        code="IT", name="Italy",
        dp_epsilon_max=5.0, dp_delta_max=1e-5,
        min_aggregation_count=3, hdab_strictness=3,
        allowed_purposes=["scientific_research", "public_health_surveillance",
                          "health_policy", "ai_system_development"],
        data_retention_days=1095,
        opt_out_rate=0.05,
        latency_ms=(18, 40), lat=42.5, lon=12.5,
    ),
    "ES": EUCountryProfile(
        code="ES", name="Spain",
        dp_epsilon_max=8.0, dp_delta_max=1e-5,
        min_aggregation_count=2, hdab_strictness=2,
        allowed_purposes=["scientific_research", "public_health_surveillance",
                          "health_policy", "education_training", "ai_system_development"],
        data_retention_days=1825,
        opt_out_rate=0.04,
        latency_ms=(20, 45), lat=40.4, lon=-3.7,
    ),
    "NL": EUCountryProfile(
        code="NL", name="Netherlands",
        dp_epsilon_max=2.0, dp_delta_max=1e-6,
        min_aggregation_count=4, hdab_strictness=4,
        allowed_purposes=["scientific_research", "public_health_surveillance", "health_policy"],
        data_retention_days=730,
        opt_out_rate=0.10,
        latency_ms=(10, 25), lat=52.4, lon=4.9,
    ),
    "SE": EUCountryProfile(
        code="SE", name="Sweden",
        dp_epsilon_max=2.0, dp_delta_max=1e-6,
        min_aggregation_count=3, hdab_strictness=4,
        allowed_purposes=["scientific_research", "public_health_surveillance"],
        data_retention_days=365,
        opt_out_rate=0.06,
        latency_ms=(25, 50), lat=59.3, lon=18.1,
    ),
    "PL": EUCountryProfile(
        code="PL", name="Poland",
        dp_epsilon_max=10.0, dp_delta_max=1e-4,
        min_aggregation_count=2, hdab_strictness=2,
        allowed_purposes=["scientific_research", "public_health_surveillance",
                          "health_policy", "education_training",
                          "ai_system_development", "personalized_medicine"],
        data_retention_days=1825,
        opt_out_rate=0.03,
        latency_ms=(30, 60), lat=52.2, lon=21.0,
    ),
    "AT": EUCountryProfile(
        code="AT", name="Austria",
        dp_epsilon_max=1.5, dp_delta_max=1e-6,
        min_aggregation_count=4, hdab_strictness=5,
        allowed_purposes=["scientific_research", "public_health_surveillance"],
        data_retention_days=365,
        opt_out_rate=0.11,
        latency_ms=(12, 30), lat=48.2, lon=16.4,
    ),
    "BE": EUCountryProfile(
        code="BE", name="Belgium",
        dp_epsilon_max=3.0, dp_delta_max=1e-5,
        min_aggregation_count=3, hdab_strictness=3,
        allowed_purposes=["scientific_research", "public_health_surveillance",
                          "health_policy", "ai_system_development"],
        data_retention_days=1095,
        opt_out_rate=0.07,
        latency_ms=(10, 25), lat=50.8, lon=4.4,
    ),
    "PT": EUCountryProfile(
        code="PT", name="Portugal",
        dp_epsilon_max=6.0, dp_delta_max=1e-5,
        min_aggregation_count=2, hdab_strictness=2,
        allowed_purposes=["scientific_research", "public_health_surveillance",
                          "health_policy", "education_training"],
        data_retention_days=1460,
        opt_out_rate=0.04,
        latency_ms=(25, 55), lat=38.7, lon=-9.1,
    ),
}


# =============================================================================
# HOSPITAL NODE WITH JURISDICTION
# =============================================================================

@dataclass
class HospitalNode:
    """A hospital node with jurisdiction metadata."""
    hospital_id: int
    name: str
    country_code: str
    country_profile: EUCountryProfile
    effective_epsilon: float       # min(global_epsilon, country_max)
    effective_delta: float
    num_samples: int = 0
    num_samples_after_optout: int = 0
    opted_out_samples: int = 0
    cumulative_epsilon_spent: float = 0.0
    rounds_participated: int = 0


# =============================================================================
# AUDIT LOG
# =============================================================================

@dataclass
class AuditEntry:
    """Single entry in the cross-border compliance audit trail."""
    timestamp: str
    round_num: int
    hospital_id: int
    hospital_name: str
    country_code: str
    jurisdiction: str
    action: str               # "train", "aggregate", "optout_check", "latency"
    epsilon_this_round: float
    epsilon_cumulative: float
    epsilon_budget_remaining: float
    samples_used: int
    samples_opted_out: int
    latency_ms: float
    compliance_status: str    # "compliant", "warning", "violation"
    details: str


class CrossBorderAuditLog:
    """Compliance audit trail for cross-border federated training."""

    def __init__(self):
        self.entries: List[AuditEntry] = []

    def log(self, **kwargs) -> AuditEntry:
        entry = AuditEntry(
            timestamp=datetime.now().isoformat(),
            **kwargs,
        )
        self.entries.append(entry)
        return entry

    def get_entries_by_country(self, country_code: str) -> List[AuditEntry]:
        return [e for e in self.entries if e.country_code == country_code]

    def get_entries_by_round(self, round_num: int) -> List[AuditEntry]:
        return [e for e in self.entries if e.round_num == round_num]

    def get_violations(self) -> List[AuditEntry]:
        return [e for e in self.entries if e.compliance_status == "violation"]

    def to_csv(self, path: str):
        if not self.entries:
            return
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(self.entries[0]).keys()))
            writer.writeheader()
            for entry in self.entries:
                writer.writerow(asdict(entry))

    def summary_by_country(self) -> Dict[str, Dict]:
        """Aggregate audit metrics per country."""
        countries = {}
        for entry in self.entries:
            if entry.action != "train":
                continue
            cc = entry.country_code
            if cc not in countries:
                countries[cc] = {
                    "country_code": cc,
                    "jurisdiction": entry.jurisdiction,
                    "total_rounds": 0,
                    "total_samples_used": 0,
                    "total_opted_out": 0,
                    "total_latency_ms": 0.0,
                    "epsilon_spent": 0.0,
                    "violations": 0,
                }
            c = countries[cc]
            c["total_rounds"] += 1
            c["total_samples_used"] += entry.samples_used
            c["total_opted_out"] += entry.samples_opted_out
            c["total_latency_ms"] += entry.latency_ms
            c["epsilon_spent"] = max(c["epsilon_spent"], entry.epsilon_cumulative)
            if entry.compliance_status == "violation":
                c["violations"] += 1
        return countries


# =============================================================================
# CROSS-BORDER ROUND RESULT
# =============================================================================

@dataclass
class CrossBorderRoundResult:
    """Result of a single cross-border federated round."""
    round_num: int
    global_loss: float
    global_acc: float
    global_f1: float
    global_precision: float
    global_recall: float
    global_auc: float
    time_seconds: float
    latency_overhead_ms: float
    per_hospital: List[Dict[str, Any]]
    compliance_status: str


# =============================================================================
# CROSS-BORDER FEDERATED TRAINER
# =============================================================================

HOSPITAL_NAMES = {
    "DE": ["Charite Berlin", "LMU Klinikum Munchen", "Uniklinik Heidelberg"],
    "FR": ["AP-HP Paris", "HCL Lyon", "CHU Bordeaux"],
    "IT": ["Bambino Gesu Roma", "Policlinico Milano", "AOU Padova"],
    "ES": ["La Paz Madrid", "Clinic Barcelona", "Virgen del Rocio Sevilla"],
    "NL": ["UMC Utrecht", "Erasmus MC Rotterdam", "AMC Amsterdam"],
    "SE": ["Karolinska Stockholm", "Sahlgrenska Goteborg"],
    "PL": ["WUM Warszawa", "UJ Krakow"],
    "AT": ["AKH Wien", "Med Uni Graz"],
    "BE": ["UZ Leuven", "Erasme Bruxelles"],
    "PT": ["Santa Maria Lisboa", "S. Joao Porto"],
}


class CrossBorderFederatedTrainer:
    """
    Federated learning trainer with cross-border EHDS simulation.

    Wraps the existing FederatedTrainer/ImageFederatedTrainer and adds:
    - Per-country epsilon enforcement
    - Network latency simulation
    - Opt-out rate simulation
    - HDAB purpose validation
    - Compliance audit trail with jurisdiction tracking
    """

    def __init__(
        self,
        countries: List[str],
        hospitals_per_country: int = 1,
        algorithm: str = "FedAvg",
        num_rounds: int = 15,
        local_epochs: int = 3,
        batch_size: int = 32,
        learning_rate: float = 0.01,
        global_epsilon: float = 10.0,
        purpose: str = "scientific_research",
        dataset_type: str = "synthetic",
        dataset_path: Optional[str] = None,
        img_size: int = 128,
        seed: int = 42,
        simulate_latency: bool = True,
        mu: float = 0.1,
        server_lr: float = 0.1,
        beta1: float = 0.9,
        beta2: float = 0.99,
        tau: float = 1e-3,
        progress_callback=None,
        # Jurisdiction privacy parameters
        jurisdiction_privacy_enabled: bool = False,
        country_overrides: Optional[Dict[str, Dict]] = None,
        hospital_allocation_fraction: float = 1.0,
        noise_strategy: str = "global",
        min_active_clients: int = 2,
        # IHE Integration Profiles
        ihe_enabled: bool = False,
        ihe_config: Optional[Dict[str, Any]] = None,
        # Data Quality Framework (EHDS Art. 69)
        data_quality_enabled: bool = False,
        data_quality_config: Optional[Dict[str, Any]] = None,
        # MyHealth@EU / NCPeH Integration (EHDS Art. 5-12)
        myhealth_eu_enabled: bool = False,
        myhealth_eu_config: Optional[Dict[str, Any]] = None,
        # Governance Lifecycle (EHDS Chapter IV, Art. 33-44)
        governance_lifecycle_enabled: bool = False,
        governance_config: Optional[Dict[str, Any]] = None,
        # Secure Processing Environment (EHDS Art. 50)
        secure_processing_enabled: bool = False,
        secure_processing_config: Optional[Dict[str, Any]] = None,
    ):
        self.countries = countries
        self.hospitals_per_country = hospitals_per_country
        self.algorithm = algorithm
        self.num_rounds = num_rounds
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.global_epsilon = global_epsilon
        self.purpose = purpose
        self.dataset_type = dataset_type
        self.dataset_path = dataset_path
        self.img_size = img_size
        self.seed = seed
        self.simulate_latency = simulate_latency
        self.mu = mu
        self.server_lr = server_lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.tau = tau
        self.progress_callback = progress_callback

        # Jurisdiction privacy
        self.jurisdiction_privacy_enabled = jurisdiction_privacy_enabled
        self.country_overrides = country_overrides or {}
        self.hospital_allocation_fraction = hospital_allocation_fraction
        self.noise_strategy = noise_strategy
        self.min_active_clients = min_active_clients
        self.jurisdiction_manager = None
        # Opt-out schedule: {country_code: round_number}
        self.optout_schedule: Dict[str, int] = {}

        # IHE Integration Profiles
        self.ihe_enabled = ihe_enabled
        self.ihe_config = ihe_config or {}
        self.ihe_bridge = None

        # Data Quality Framework (EHDS Art. 69)
        self.data_quality_enabled = data_quality_enabled
        self.data_quality_config = data_quality_config or {}
        self.quality_manager = None

        # MyHealth@EU / NCPeH Integration (EHDS Art. 5-12)
        self.myhealth_eu_enabled = myhealth_eu_enabled
        self.myhealth_eu_config = myhealth_eu_config or {}
        self.myhealth_bridge = None

        # Governance Lifecycle (EHDS Chapter IV, Art. 33-44)
        self.governance_lifecycle_enabled = governance_lifecycle_enabled
        self.governance_config = governance_config or {}
        self.governance_bridge = None
        self._minimization_report = None

        # Secure Processing Environment (EHDS Art. 50)
        self.secure_processing_enabled = secure_processing_enabled
        self.secure_processing_config = secure_processing_config or {}
        self.secure_processing_bridge = None

        self.rng = np.random.RandomState(seed)
        self.audit_log = CrossBorderAuditLog()
        self.history: List[CrossBorderRoundResult] = []

        # Build hospital nodes
        self.hospitals: List[HospitalNode] = []
        self._build_hospitals()

        # Validate purpose against all country HDABs
        self.purpose_violations: List[str] = []
        self._validate_purpose()

        # Underlying trainer (created in train())
        self._trainer = None

    def _build_hospitals(self):
        """Create hospital nodes with jurisdiction metadata."""
        hospital_id = 0
        for cc in self.countries:
            profile = EU_COUNTRY_PROFILES[cc]
            names = HOSPITAL_NAMES.get(cc, [f"Hospital {cc}"])
            for i in range(self.hospitals_per_country):
                name = names[i % len(names)]
                effective_eps = min(self.global_epsilon, profile.dp_epsilon_max)
                effective_delta = min(1e-5, profile.dp_delta_max)
                self.hospitals.append(HospitalNode(
                    hospital_id=hospital_id,
                    name=name,
                    country_code=cc,
                    country_profile=profile,
                    effective_epsilon=effective_eps,
                    effective_delta=effective_delta,
                ))
                hospital_id += 1

    def _validate_purpose(self):
        """Check if the declared purpose is allowed by all national HDABs."""
        for h in self.hospitals:
            if self.purpose not in h.country_profile.allowed_purposes:
                self.purpose_violations.append(
                    f"{h.name} ({h.country_code}): purpose '{self.purpose}' "
                    f"not allowed by HDAB (allowed: {h.country_profile.allowed_purposes})"
                )

    def _simulate_optout(self, hospital: HospitalNode, num_samples: int) -> int:
        """Simulate opt-out for a hospital based on national rate."""
        rate = hospital.country_profile.opt_out_rate
        opted_out = int(num_samples * rate)
        # Add some randomness
        opted_out = max(0, opted_out + self.rng.randint(-2, 3))
        opted_out = min(opted_out, num_samples - 1)  # Keep at least 1 sample
        return opted_out

    def _simulate_latency(self, hospital: HospitalNode) -> float:
        """Simulate network latency for a hospital."""
        if not self.simulate_latency:
            return 0.0
        lo, hi = hospital.country_profile.latency_ms
        latency = self.rng.uniform(lo, hi)
        return latency

    def get_num_clients(self) -> int:
        return len(self.hospitals)

    def get_strictest_epsilon(self) -> float:
        """Return the most restrictive epsilon across all jurisdictions."""
        return min(h.effective_epsilon for h in self.hospitals)

    def train(self) -> Dict[str, Any]:
        """
        Run cross-border federated training.

        Creates the underlying trainer, runs training round by round,
        and applies per-country privacy and latency constraints.
        """
        from terminal.fl_trainer import FederatedTrainer, ImageFederatedTrainer

        num_clients = self.get_num_clients()
        # Use the strictest epsilon across all jurisdictions
        effective_epsilon = self.get_strictest_epsilon()

        # Per-round epsilon budget
        per_round_epsilon = effective_epsilon / max(self.num_rounds, 1)

        # Map IID (cross-border is always non-IID by nature)
        is_iid = False
        alpha = 0.5  # Non-IID Dirichlet

        # Create the underlying trainer
        if self.dataset_type == "imaging" and self.dataset_path:
            self._trainer = ImageFederatedTrainer(
                data_dir=self.dataset_path,
                num_clients=num_clients,
                algorithm=self.algorithm,
                local_epochs=self.local_epochs,
                batch_size=self.batch_size,
                learning_rate=self.learning_rate,
                is_iid=is_iid, alpha=alpha,
                mu=self.mu,
                dp_enabled=True,
                dp_epsilon=effective_epsilon,
                dp_clip_norm=1.0,
                seed=self.seed,
                img_size=self.img_size,
                server_lr=self.server_lr,
                beta1=self.beta1, beta2=self.beta2, tau=self.tau,
            )
        else:
            self._trainer = FederatedTrainer(
                num_clients=num_clients,
                samples_per_client=200,
                algorithm=self.algorithm,
                local_epochs=self.local_epochs,
                batch_size=self.batch_size,
                learning_rate=self.learning_rate,
                is_iid=is_iid, alpha=alpha,
                mu=self.mu,
                dp_enabled=True,
                dp_epsilon=effective_epsilon,
                dp_clip_norm=1.0,
                seed=self.seed,
                server_lr=self.server_lr,
                beta1=self.beta1, beta2=self.beta2, tau=self.tau,
            )

        # Get data stats and assign to hospitals
        stats = self._trainer.get_client_data_stats()
        for i, hospital in enumerate(self.hospitals):
            if i in stats:
                hospital.num_samples = stats[i]["num_samples"]
                opted_out = self._simulate_optout(hospital, hospital.num_samples)
                hospital.opted_out_samples = opted_out
                hospital.num_samples_after_optout = hospital.num_samples - opted_out

        # Log initial purpose validation
        for v in self.purpose_violations:
            self.audit_log.log(
                round_num=-1, hospital_id=-1, hospital_name="HDAB",
                country_code="EU", jurisdiction="EU-wide",
                action="purpose_validation",
                epsilon_this_round=0, epsilon_cumulative=0,
                epsilon_budget_remaining=effective_epsilon,
                samples_used=0, samples_opted_out=0,
                latency_ms=0, compliance_status="violation",
                details=v,
            )

        # Initialize jurisdiction privacy manager (if enabled)
        if self.jurisdiction_privacy_enabled:
            from governance.jurisdiction_privacy import JurisdictionPrivacyManager
            client_jurisdictions = {
                h.hospital_id: h.country_code for h in self.hospitals
            }
            jurisdiction_budgets = {
                cc: {
                    "epsilon_max": EU_COUNTRY_PROFILES[cc].dp_epsilon_max,
                    "delta": EU_COUNTRY_PROFILES[cc].dp_delta_max,
                }
                for cc in set(h.country_code for h in self.hospitals)
            }
            # Apply per-country overrides from config
            for cc, override in self.country_overrides.items():
                if cc in jurisdiction_budgets:
                    jurisdiction_budgets[cc].update(override)
                else:
                    jurisdiction_budgets[cc] = override
            hospital_names = {h.hospital_id: h.name for h in self.hospitals}
            self.jurisdiction_manager = JurisdictionPrivacyManager(
                client_jurisdictions=client_jurisdictions,
                jurisdiction_budgets=jurisdiction_budgets,
                global_epsilon=self.global_epsilon,
                num_rounds=self.num_rounds,
                hospital_names=hospital_names,
                hospital_allocation_fraction=self.hospital_allocation_fraction,
                noise_strategy=self.noise_strategy,
                min_active_clients=self.min_active_clients,
            )

        # Initialize IHE bridge (if enabled)
        if self.ihe_enabled:
            from governance.ihe_fl_bridge import IHEFLBridge
            self.ihe_bridge = IHEFLBridge(
                hospitals=self.hospitals,
                config=self.ihe_config,
            )
            self.ihe_bridge.start_session(purpose=self.purpose)

        # Initialize Data Quality Manager (if enabled) - EHDS Art. 69
        if self.data_quality_enabled:
            from governance.data_quality_framework import DataQualityManager
            self.quality_manager = DataQualityManager(
                hospitals=self.hospitals,
                config=self.data_quality_config,
            )
            # Assess all client data quality BEFORE training starts
            client_data_dict = {}
            num_clients = len(self.hospitals)
            for cid in range(num_clients):
                if hasattr(self._trainer, 'client_data') and cid in self._trainer.client_data:
                    client_data_dict[cid] = self._trainer.client_data[cid]
            if client_data_dict:
                quality_reports = self.quality_manager.assess_all_clients(client_data_dict)
                # Log anomaly warnings to audit trail
                for cid, report in quality_reports.items():
                    if report.is_anomalous:
                        hospital = self.hospitals[cid] if cid < len(self.hospitals) else None
                        h_name = hospital.name if hospital else f"Client_{cid}"
                        for anomaly in report.anomalies:
                            self.audit_log.log(
                                round_num=-1,
                                hospital_id=cid,
                                hospital_name=h_name,
                                country_code=hospital.country_code if hospital else "??",
                                jurisdiction=f"{hospital.country_profile.name} HDAB" if hospital else "Unknown",
                                action="quality_anomaly",
                                epsilon_this_round=0,
                                epsilon_cumulative=0,
                                epsilon_budget_remaining=0,
                                samples_used=report.num_samples,
                                samples_opted_out=0,
                                latency_ms=0,
                                compliance_status="warning",
                                details=f"Data quality anomaly: {anomaly}",
                            )

        # Initialize MyHealth@EU bridge (if enabled) - EHDS Art. 5-12
        if self.myhealth_eu_enabled:
            from governance.myhealth_eu_bridge import MyHealthEUBridge
            self.myhealth_bridge = MyHealthEUBridge(
                hospitals=self.hospitals,
                config=self.myhealth_eu_config,
                seed=self.seed,
            )
            self.myhealth_bridge.start_session()

        # Initialize Governance Lifecycle (if enabled) - EHDS Ch. IV, Art. 33-44
        if self.governance_lifecycle_enabled:
            from governance.governance_lifecycle import GovernanceLifecycleBridge
            self.governance_bridge = GovernanceLifecycleBridge(
                hospitals=self.hospitals,
                countries=self.countries,
                purpose=self.purpose,
                global_epsilon=self.global_epsilon,
                num_rounds=self.num_rounds,
                config=self.governance_config,
                seed=self.seed,
            )

            # Feature names for minimization (synthetic tabular data)
            feature_names = self.governance_config.get("feature_names")
            if feature_names is None and self.dataset_type == "synthetic":
                feature_names = [
                    'age', 'bmi', 'systolic_bp', 'glucose', 'cholesterol',
                    'heart_rate', 'resp_rate', 'temperature', 'oxygen_sat',
                    'prev_conditions',
                ]

            # Phase 1: pre-training (HDAB connect, permits, minimization)
            train_data = getattr(self._trainer, 'client_data', None)
            test_data = getattr(self._trainer, 'client_test_data', None)
            gov_result = self.governance_bridge.pre_training(
                train_data=train_data if self.dataset_type != "imaging" else None,
                test_data=test_data if self.dataset_type != "imaging" else None,
                feature_names=feature_names,
            )

            # Apply minimized data if features were reduced
            if gov_result.get("minimized_train") is not None:
                self._trainer.client_data = gov_result["minimized_train"]
                if gov_result.get("minimized_test"):
                    self._trainer.client_test_data = gov_result["minimized_test"]
                # Rebuild model with new input_dim
                new_dim = gov_result["input_dim"]
                self._trainer._rebuild_model(new_dim)

            self._minimization_report = gov_result.get("minimization_report")

            # Log purpose violations from HDAB permits
            for v in gov_result.get("purpose_violations", []):
                self.purpose_violations.append(v)
                self.audit_log.log(
                    round_num=-1, hospital_id=-1, hospital_name="GOVERNANCE",
                    country_code="EU", jurisdiction="EU HDAB",
                    action="permit_purpose_rejection",
                    epsilon_this_round=0, epsilon_cumulative=0,
                    epsilon_budget_remaining=self.global_epsilon,
                    samples_used=0, samples_opted_out=0,
                    latency_ms=0, compliance_status="violation",
                    details=v,
                )

        # Initialize Secure Processing Environment (if enabled) - EHDS Art. 50
        if self.secure_processing_enabled:
            from governance.secure_processing import SecureProcessingBridge
            self.secure_processing_bridge = SecureProcessingBridge(
                num_clients=len(self.hospitals),
                config=self.secure_processing_config,
            )
            self.secure_processing_bridge.start_session()

        # Training loop
        total_start = time.time()
        for round_num in range(self.num_rounds):
            round_start = time.time()
            total_latency = 0.0

            # Apply scheduled opt-outs for this round
            if self.jurisdiction_manager and self.optout_schedule:
                for cc, opt_round in self.optout_schedule.items():
                    if round_num == opt_round:
                        removed = self.jurisdiction_manager.simulate_optout(
                            cc, round_num
                        )
                        if removed:
                            self.audit_log.log(
                                round_num=round_num, hospital_id=-1,
                                hospital_name="SYSTEM",
                                country_code=cc, jurisdiction=f"{cc} HDAB",
                                action="art48_optout",
                                epsilon_this_round=0, epsilon_cumulative=0,
                                epsilon_budget_remaining=0,
                                samples_used=0, samples_opted_out=0,
                                latency_ms=0, compliance_status="compliant",
                                details=f"Art. 48 opt-out: {cc} withdrew, "
                                        f"{len(removed)} clients removed",
                            )

            # Jurisdiction privacy: determine active clients for this round
            active_client_ids = None
            eff_noise = None
            if self.jurisdiction_manager:
                active_client_ids, eff_noise = (
                    self.jurisdiction_manager.pre_round_check(round_num)
                )
                if len(active_client_ids) < self.min_active_clients:
                    # Not enough clients to continue
                    self.audit_log.log(
                        round_num=round_num, hospital_id=-1,
                        hospital_name="SYSTEM",
                        country_code="EU", jurisdiction="EU-wide",
                        action="training_terminated",
                        epsilon_this_round=0, epsilon_cumulative=0,
                        epsilon_budget_remaining=0,
                        samples_used=0, samples_opted_out=0,
                        latency_ms=0, compliance_status="compliant",
                        details=f"Only {len(active_client_ids)} active clients, "
                                f"min={self.min_active_clients}",
                    )
                    break
                # Set noise override on underlying trainer
                self._trainer._noise_scale_override = eff_noise

            # Determine which hospitals are active this round
            active_hospitals = self.hospitals
            if active_client_ids is not None:
                active_set = set(active_client_ids)
                active_hospitals = [
                    h for h in self.hospitals if h.hospital_id in active_set
                ]

            # IHE: pre-round operations (CT sync, XDS retrieve, mTLS verify)
            if self.ihe_bridge:
                self.ihe_bridge.pre_round(round_num, active_hospitals)

            # Simulate per-hospital latency and log audit entries
            for hospital in active_hospitals:
                latency = self._simulate_latency(hospital)
                total_latency += latency

                # Update cumulative epsilon
                hospital.cumulative_epsilon_spent += per_round_epsilon
                hospital.rounds_participated += 1

                # Check compliance
                budget_remaining = hospital.effective_epsilon - hospital.cumulative_epsilon_spent
                if budget_remaining < 0:
                    status = "violation"
                elif budget_remaining < per_round_epsilon * 2:
                    status = "warning"
                else:
                    status = "compliant"

                self.audit_log.log(
                    round_num=round_num,
                    hospital_id=hospital.hospital_id,
                    hospital_name=hospital.name,
                    country_code=hospital.country_code,
                    jurisdiction=f"{hospital.country_profile.name} HDAB",
                    action="train",
                    epsilon_this_round=per_round_epsilon,
                    epsilon_cumulative=hospital.cumulative_epsilon_spent,
                    epsilon_budget_remaining=budget_remaining,
                    samples_used=hospital.num_samples_after_optout,
                    samples_opted_out=hospital.opted_out_samples,
                    latency_ms=latency,
                    compliance_status=status,
                    details=f"Algorithm={self.algorithm}, "
                            f"local_epochs={self.local_epochs}, "
                            f"country_max_eps={hospital.country_profile.dp_epsilon_max}",
                )

            # Log skipped hospitals (budget exhausted)
            if active_client_ids is not None:
                for hospital in self.hospitals:
                    if hospital.hospital_id not in active_set:
                        self.audit_log.log(
                            round_num=round_num,
                            hospital_id=hospital.hospital_id,
                            hospital_name=hospital.name,
                            country_code=hospital.country_code,
                            jurisdiction=f"{hospital.country_profile.name} HDAB",
                            action="skipped",
                            epsilon_this_round=0,
                            epsilon_cumulative=hospital.cumulative_epsilon_spent,
                            epsilon_budget_remaining=0,
                            samples_used=0,
                            samples_opted_out=hospital.opted_out_samples,
                            latency_ms=0,
                            compliance_status="compliant",
                            details="Budget exhausted or opted out",
                        )

            # Simulate latency delay (scaled down: real ms -> sleep in ms/100)
            if self.simulate_latency and total_latency > 0:
                time.sleep(total_latency / 5000.0)  # Subtle delay

            # Get quality weights for this round (if Data Quality Framework enabled)
            quality_weights_for_round = None
            if self.quality_manager:
                quality_weights_for_round = self.quality_manager.get_quality_weights()
                # Filter to active clients only
                if active_client_ids is not None:
                    active_set = set(active_client_ids)
                    quality_weights_for_round = {
                        cid: w for cid, w in quality_weights_for_round.items()
                        if cid in active_set
                    }

            # Run actual FL round via underlying trainer
            if (self.myhealth_bridge and
                    self.myhealth_eu_config.get("hierarchical_aggregation", True)):
                # MyHealth@EU hierarchical: train clients, then 2-level aggregate
                self.myhealth_bridge.pre_round(round_num, active_hospitals)

                # Train clients WITHOUT aggregation
                client_results = self._trainer.train_clients(
                    round_num, active_clients=active_client_ids)

                # 2-level hierarchical aggregation via NCPeH
                self.myhealth_bridge.hierarchical_aggregate(
                    self._trainer, client_results,
                    quality_weights=quality_weights_for_round)

                # Apply DP noise if needed (post-aggregation)
                if self._trainer.dp_enabled:
                    import torch
                    noise_scale = getattr(
                        self._trainer, '_noise_scale_override', None)
                    if noise_scale is None:
                        noise_scale = (self._trainer.dp_clip_norm /
                                       self._trainer.dp_epsilon)
                    for param in self._trainer.global_model.parameters():
                        param.data += torch.randn_like(param) * noise_scale

                # Evaluate after hierarchical aggregation
                metrics = self._trainer._evaluate()
                from terminal.fl_trainer import RoundResult
                round_result = RoundResult(
                    round_num=round_num,
                    global_loss=metrics["loss"],
                    global_acc=metrics["accuracy"],
                    global_f1=metrics["f1"],
                    global_precision=metrics["precision"],
                    global_recall=metrics["recall"],
                    global_auc=metrics["auc"],
                    client_results=client_results,
                    time_seconds=0.0,
                )

                self.myhealth_bridge.post_round(round_num, metrics)
            else:
                # Standard flat aggregation
                round_result = self._trainer.train_round(
                    round_num, active_clients=active_client_ids,
                    quality_weights=quality_weights_for_round,
                )

            # Post-round: record jurisdiction privacy spending
            if self.jurisdiction_manager and active_client_ids and eff_noise:
                self.jurisdiction_manager.record_round(
                    round_num, active_client_ids, eff_noise
                )

            # IHE: post-round operations (ATNA audit)
            if self.ihe_bridge:
                self.ihe_bridge.post_round(
                    round_num, active_hospitals,
                    metrics={
                        "accuracy": round_result.global_acc,
                        "loss": round_result.global_loss,
                        "f1": round_result.global_f1,
                    },
                    success=True,
                )

            # Governance: log round completion (EHDS Ch. IV)
            if self.governance_bridge:
                self.governance_bridge.log_round_completion(
                    round_num, round_result, per_round_epsilon
                )

            # Secure Processing: post-round (enclave log + watermark embed)
            if self.secure_processing_bridge:
                per_client_samples = [
                    h.num_samples_after_optout for h in self.hospitals
                ]
                updated_state = self.secure_processing_bridge.post_round(
                    round_num=round_num,
                    model_state_dict=self._trainer.global_model.state_dict(),
                    num_clients_trained=len(self.hospitals),
                    per_client_samples=per_client_samples,
                )
                self._trainer.global_model.load_state_dict(updated_state)

            # Build per-hospital info for this round
            per_hospital_info = []
            for hospital in self.hospitals:
                per_hospital_info.append({
                    "hospital_id": hospital.hospital_id,
                    "name": hospital.name,
                    "country": hospital.country_code,
                    "epsilon_spent": hospital.cumulative_epsilon_spent,
                    "epsilon_max": hospital.effective_epsilon,
                    "samples": hospital.num_samples_after_optout,
                    "opted_out": hospital.opted_out_samples,
                    "latency_ms": self._simulate_latency(hospital),
                })

            round_time = time.time() - round_start
            avg_latency = total_latency / max(len(self.hospitals), 1)

            # Check round-level compliance
            violations = [h for h in self.hospitals
                          if h.cumulative_epsilon_spent > h.effective_epsilon]
            round_compliance = "violation" if violations else "compliant"

            cb_result = CrossBorderRoundResult(
                round_num=round_num,
                global_loss=round_result.global_loss,
                global_acc=round_result.global_acc,
                global_f1=round_result.global_f1,
                global_precision=round_result.global_precision,
                global_recall=round_result.global_recall,
                global_auc=round_result.global_auc,
                time_seconds=round_time,
                latency_overhead_ms=avg_latency,
                per_hospital=per_hospital_info,
                compliance_status=round_compliance,
            )
            self.history.append(cb_result)

            # Progress callback
            if self.progress_callback:
                self.progress_callback(
                    round_num=round_num,
                    total_rounds=self.num_rounds,
                    result=cb_result,
                )

        total_time = time.time() - total_start

        # End IHE session
        if self.ihe_bridge:
            self.ihe_bridge.end_session()

        # End MyHealth@EU session
        if self.myhealth_bridge:
            self.myhealth_bridge.end_session()

        # End governance session (EHDS Ch. IV)
        governance_report = None
        if self.governance_bridge:
            final_metrics = {}
            if self.history:
                last = self.history[-1]
                final_metrics = {
                    "accuracy": last.global_acc,
                    "f1": last.global_f1,
                    "auc": last.global_auc,
                    "loss": last.global_loss,
                }
            self.governance_bridge.end_session(
                total_rounds=len(self.history),
                final_metrics=final_metrics,
            )
            governance_report = self.governance_bridge.export_report()

        # End Secure Processing session (EHDS Art. 50)
        secure_processing_report = None
        if self.secure_processing_bridge:
            self.secure_processing_bridge.verify_final_model(
                self._trainer.global_model.state_dict(),
                round_num=len(self.history) - 1,
            )
            secure_processing_report = self.secure_processing_bridge.end_session()

        # Quality report in return dict
        quality_report = None
        if self.quality_manager:
            quality_report = self.quality_manager.export_report()

        # MyHealth@EU report in return dict
        myhealth_eu_report = None
        if self.myhealth_bridge:
            myhealth_eu_report = self.myhealth_bridge.export_report()

        return {
            "history": self.history,
            "hospitals": self.hospitals,
            "audit_log": self.audit_log,
            "total_time": total_time,
            "effective_epsilon": effective_epsilon,
            "purpose_violations": self.purpose_violations,
            "quality_report": quality_report,
            "myhealth_eu_report": myhealth_eu_report,
            "governance_report": governance_report,
            "minimization_report": self._minimization_report,
            "secure_processing_report": secure_processing_report,
        }

    def save_results(self, output_dir: str):
        """Save all cross-border experiment outputs."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # 1. Audit trail CSV
        self.audit_log.to_csv(str(out / "audit_trail.csv"))

        # 2. Per-country summary CSV
        country_summary = self.audit_log.summary_by_country()
        if country_summary:
            with open(out / "summary_by_country.csv", "w", newline="") as f:
                first = list(country_summary.values())[0]
                writer = csv.DictWriter(f, fieldnames=list(first.keys()))
                writer.writeheader()
                for row in country_summary.values():
                    writer.writerow(row)

        # 3. Convergence history CSV
        with open(out / "history_cross_border.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["round", "accuracy", "loss", "f1", "precision",
                             "recall", "auc", "time_s", "avg_latency_ms",
                             "compliance"])
            for r in self.history:
                writer.writerow([
                    r.round_num, f"{r.global_acc:.4f}", f"{r.global_loss:.4f}",
                    f"{r.global_f1:.4f}", f"{r.global_precision:.4f}",
                    f"{r.global_recall:.4f}", f"{r.global_auc:.4f}",
                    f"{r.time_seconds:.2f}", f"{r.latency_overhead_ms:.1f}",
                    r.compliance_status,
                ])

        # 4. Hospital info CSV
        with open(out / "hospitals.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "name", "country", "effective_epsilon",
                             "epsilon_spent", "samples", "opted_out",
                             "opt_out_rate", "hdab_strictness"])
            for h in self.hospitals:
                writer.writerow([
                    h.hospital_id, h.name, h.country_code,
                    h.effective_epsilon, f"{h.cumulative_epsilon_spent:.4f}",
                    h.num_samples, h.opted_out_samples,
                    f"{h.country_profile.opt_out_rate:.0%}",
                    h.country_profile.hdab_strictness,
                ])

        # 5. LaTeX table
        self._save_latex_table(out)

        # 6. Results JSON
        results_json = {
            "config": {
                "countries": self.countries,
                "hospitals_per_country": self.hospitals_per_country,
                "algorithm": self.algorithm,
                "num_rounds": self.num_rounds,
                "global_epsilon": self.global_epsilon,
                "effective_epsilon": self.get_strictest_epsilon(),
                "purpose": self.purpose,
                "dataset_type": self.dataset_type,
            },
            "final_metrics": {
                "accuracy": self.history[-1].global_acc if self.history else 0,
                "f1": self.history[-1].global_f1 if self.history else 0,
                "auc": self.history[-1].global_auc if self.history else 0,
            },
            "per_country_epsilon": {
                h.country_code: {
                    "max_allowed": h.effective_epsilon,
                    "spent": h.cumulative_epsilon_spent,
                    "compliant": h.cumulative_epsilon_spent <= h.effective_epsilon,
                }
                for h in self.hospitals
            },
            "violations": len(self.audit_log.get_violations()),
            "purpose_violations": self.purpose_violations,
        }
        with open(out / "results.json", "w") as f:
            json.dump(results_json, f, indent=2, default=str)

        # 7. Summary text
        self._save_summary(out)

        # 8. Plots
        self._save_plots(out)

        # 9. Jurisdiction privacy report (if enabled)
        if self.jurisdiction_manager:
            report = self.jurisdiction_manager.export_report()
            with open(out / "jurisdiction_privacy.json", "w") as f:
                json.dump(report, f, indent=2, default=str)

            # 10. Dropout timeline CSV
            timeline = self.jurisdiction_manager.get_dropout_timeline()
            if timeline:
                with open(out / "dropout_timeline.csv", "w", newline="") as f:
                    writer = csv.DictWriter(
                        f, fieldnames=list(timeline[0].keys())
                    )
                    writer.writeheader()
                    writer.writerows(timeline)

            # 11. Jurisdiction budget consumption plot
            self._save_jurisdiction_budget_plot(out)

        # 12. IHE compliance report (if enabled)
        if self.ihe_bridge:
            ihe_report = self.ihe_bridge.export_ihe_report()
            with open(out / "ihe_compliance_report.json", "w") as f:
                json.dump(ihe_report, f, indent=2, default=str)

            # 13. ATNA audit trail (FHIR JSON)
            audit_json = self.ihe_bridge.ihe_manager.audit_logger.export_audit_trail(
                format="json"
            )
            with open(out / "atna_audit_trail.json", "w") as f:
                f.write(audit_json)

            # 14. ATNA audit trail (DICOM XML)
            audit_xml = self.ihe_bridge.ihe_manager.audit_logger.export_audit_trail(
                format="xml"
            )
            with open(out / "atna_audit_trail.xml", "w") as f:
                f.write(audit_xml)

        # 15. Data Quality report (if enabled) - EHDS Art. 69
        if self.quality_manager:
            quality_report = self.quality_manager.export_report()
            with open(out / "quality_report.json", "w") as f:
                json.dump(quality_report, f, indent=2, default=str)

            # 16. Quality labels CSV
            with open(out / "quality_labels.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "client_id", "hospital", "overall_score", "quality_label",
                    "quality_weight", "completeness", "accuracy", "uniqueness",
                    "diversity", "consistency", "anomalous", "anomalies",
                ])
                for cid, report in sorted(
                    self.quality_manager.client_reports.items()
                ):
                    writer.writerow([
                        cid, report.hospital_name,
                        f"{report.overall_score:.4f}",
                        report.quality_label.value,
                        f"{report.quality_weight:.4f}",
                        f"{report.completeness:.4f}",
                        f"{report.accuracy:.4f}",
                        f"{report.uniqueness:.4f}",
                        f"{report.diversity:.4f}",
                        f"{report.consistency:.4f}",
                        report.is_anomalous,
                        "; ".join(report.anomalies),
                    ])

        # 17. MyHealth@EU report (if enabled)
        if self.myhealth_bridge:
            mheu_report = self.myhealth_bridge.export_report()
            with open(out / "myhealth_eu_report.json", "w") as f:
                json.dump(mheu_report, f, indent=2, default=str)

            # NCPeH topology CSV
            with open(out / "ncpeh_topology.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "ncp_id", "country", "bandwidth_mbps", "tier",
                    "services", "hospitals", "samples",
                    "agg_latency_ms", "comm_bytes",
                ])
                for cc in sorted(self.myhealth_bridge.ncp_nodes.keys()):
                    ncp = self.myhealth_bridge.ncp_nodes[cc]
                    writer.writerow([
                        ncp.ncp_id, ncp.country_name,
                        ncp.bandwidth_mbps, ncp.infrastructure_tier,
                        "|".join(ncp.services),
                        len(ncp.hospital_ids), ncp.national_samples,
                        f"{ncp.aggregation_latency_ms:.1f}",
                        ncp.communication_bytes,
                    ])

            # Inter-NCP latency matrix CSV
            codes, matrix = self.myhealth_bridge.get_latency_matrix_display()
            with open(out / "inter_ncp_latency.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["from/to"] + codes)
                for c1 in codes:
                    row = [c1] + [f"{matrix[c1][c2]:.1f}" for c2 in codes]
                    writer.writerow(row)

        # 18. Governance lifecycle report (if enabled) - EHDS Ch. IV
        if self.governance_bridge:
            gov_report = self.governance_bridge.export_report()
            with open(out / "governance_lifecycle.json", "w") as f:
                json.dump(gov_report, f, indent=2, default=str)

            # 19. Permits summary
            permits = self.governance_bridge.get_permits_summary()
            with open(out / "permits_summary.json", "w") as f:
                json.dump(permits, f, indent=2, default=str)

        # 20. Data minimization report (if applied)
        if self._minimization_report:
            with open(out / "minimization_report.json", "w") as f:
                json.dump(self._minimization_report, f, indent=2, default=str)

        # 21. Secure Processing report (if enabled) - EHDS Art. 50
        if self.secure_processing_bridge:
            sp_report = self.secure_processing_bridge.export_report()
            with open(out / "secure_processing.json", "w") as f:
                json.dump(sp_report, f, indent=2, default=str)

    def _save_latex_table(self, out: Path):
        """Generate LaTeX table for cross-border results."""
        lines = []
        lines.append(r"\begin{table}[htbp]")
        lines.append(r"\centering")
        lines.append(r"\caption{Cross-Border Federated Learning: Per-Jurisdiction Privacy Analysis}")
        lines.append(r"\label{tab:cross_border}")
        lines.append(r"\small")
        lines.append(r"\begin{tabular}{llccccc}")
        lines.append(r"\toprule")
        lines.append(r"\textbf{Country} & \textbf{Hospital} & \textbf{$\varepsilon_{\max}$} & "
                     r"\textbf{$\varepsilon_{\text{spent}}$} & \textbf{Samples} & "
                     r"\textbf{Opt-out} & \textbf{HDAB} \\")
        lines.append(r"\midrule")

        for h in self.hospitals:
            eps_max = f"{h.effective_epsilon:.1f}"
            eps_spent = f"{h.cumulative_epsilon_spent:.2f}"
            optout = f"{h.opted_out_samples}"
            hdab = r"\star" * h.country_profile.hdab_strictness
            name_escaped = h.name.replace("&", r"\&")
            lines.append(
                f"{h.country_code} & {name_escaped} & ${eps_max}$ & "
                f"${eps_spent}$ & {h.num_samples_after_optout} & "
                f"{optout} & {hdab} \\\\"
            )

        lines.append(r"\midrule")
        # Final row: global results
        if self.history:
            last = self.history[-1]
            lines.append(
                f"\\textit{{Global}} & \\textit{{{self.algorithm}}} & "
                f"${self.get_strictest_epsilon():.1f}$ & "
                f"${self.hospitals[0].cumulative_epsilon_spent:.2f}$ & "
                f"& & \\textit{{Acc={last.global_acc:.1%}}} \\\\"
            )

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append("")

        # Footnote
        country_list = ", ".join(self.countries)
        lines.append(f"\\footnotesize{{Cross-border FL: {country_list}. "
                     f"Algorithm: {self.algorithm}. "
                     f"{self.num_rounds} rounds, {self.local_epochs} local epochs. "
                     f"$\\varepsilon_{{\\max}}$ = national HDAB limit. "
                     f"HDAB strictness: $\\star$=low to $\\star\\star\\star\\star\\star$=high.}}")
        lines.append(r"\end{table}")

        with open(out / "table_cross_border.tex", "w") as f:
            f.write("\n".join(lines))

    def _save_summary(self, out: Path):
        """Save human-readable summary."""
        lines = []
        lines.append("CROSS-BORDER FEDERATED LEARNING - EHDS SIMULATION")
        lines.append("=" * 60)
        lines.append("")
        lines.append(f"Countries: {', '.join(self.countries)}")
        lines.append(f"Hospitals: {len(self.hospitals)}")
        lines.append(f"Algorithm: {self.algorithm}")
        lines.append(f"Rounds: {self.num_rounds}")
        lines.append(f"Purpose: {self.purpose}")
        lines.append(f"Global epsilon: {self.global_epsilon}")
        lines.append(f"Effective epsilon (strictest): {self.get_strictest_epsilon()}")
        lines.append("")
        lines.append("PER-COUNTRY SUMMARY")
        lines.append("-" * 60)

        country_summary = self.audit_log.summary_by_country()
        for cc, data in country_summary.items():
            profile = EU_COUNTRY_PROFILES[cc]
            lines.append(f"\n  {profile.name} ({cc}):")
            lines.append(f"    HDAB Strictness: {profile.hdab_strictness}/5")
            lines.append(f"    Max Epsilon: {profile.dp_epsilon_max}")
            lines.append(f"    Epsilon Spent: {data['epsilon_spent']:.4f}")
            lines.append(f"    Samples Used: {data['total_samples_used']}")
            lines.append(f"    Opted Out: {data['total_opted_out']}")
            lines.append(f"    Avg Latency: {data['total_latency_ms']/max(data['total_rounds'],1):.1f} ms")
            lines.append(f"    Violations: {data['violations']}")

        lines.append("")
        lines.append("FINAL METRICS")
        lines.append("-" * 60)
        if self.history:
            last = self.history[-1]
            lines.append(f"  Accuracy:  {last.global_acc:.4f}")
            lines.append(f"  F1:        {last.global_f1:.4f}")
            lines.append(f"  AUC:       {last.global_auc:.4f}")
            lines.append(f"  Loss:      {last.global_loss:.4f}")

        lines.append("")
        lines.append(f"Purpose Violations: {len(self.purpose_violations)}")
        for v in self.purpose_violations:
            lines.append(f"  - {v}")
        lines.append("")
        lines.append(f"Audit Trail: {len(self.audit_log.entries)} entries")
        lines.append(f"Compliance Violations: {len(self.audit_log.get_violations())}")

        with open(out / "summary.txt", "w") as f:
            f.write("\n".join(lines))

    def _save_plots(self, out: Path):
        """Generate cross-border specific plots."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            rounds = [r.round_num for r in self.history]
            acc = [r.global_acc for r in self.history]
            loss = [r.global_loss for r in self.history]
            f1 = [r.global_f1 for r in self.history]
            auc = [r.global_auc for r in self.history]

            # --- Plot 1: Convergence ---
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            axes[0].plot(rounds, acc, "b-o", markersize=3, label="Accuracy")
            axes[0].set_xlabel("Round")
            axes[0].set_ylabel("Accuracy")
            axes[0].set_title("Convergence: Accuracy")
            axes[0].grid(True, alpha=0.3)
            axes[1].plot(rounds, loss, "r-o", markersize=3, label="Loss")
            axes[1].set_xlabel("Round")
            axes[1].set_ylabel("Loss")
            axes[1].set_title("Convergence: Loss")
            axes[1].grid(True, alpha=0.3)
            fig.suptitle(f"Cross-Border FL ({', '.join(self.countries)}) - {self.algorithm}")
            plt.tight_layout()
            plt.savefig(out / "plot_convergence.png", dpi=150, bbox_inches="tight")
            plt.close()

            # --- Plot 2: Per-country epsilon consumption ---
            country_codes = list(dict.fromkeys(h.country_code for h in self.hospitals))
            eps_max = [EU_COUNTRY_PROFILES[cc].dp_epsilon_max for cc in country_codes]
            eps_spent = []
            for cc in country_codes:
                cc_hospitals = [h for h in self.hospitals if h.country_code == cc]
                eps_spent.append(max(h.cumulative_epsilon_spent for h in cc_hospitals))

            fig, ax = plt.subplots(figsize=(10, 5))
            x = np.arange(len(country_codes))
            width = 0.35
            bars1 = ax.bar(x - width/2, eps_max, width, label="Max Allowed (HDAB)", color="#2196F3", alpha=0.7)
            bars2 = ax.bar(x + width/2, eps_spent, width, label="Spent", color="#FF5722", alpha=0.7)
            ax.set_xlabel("Country")
            ax.set_ylabel("Privacy Budget (epsilon)")
            ax.set_title("Per-Country Privacy Budget: Allowed vs. Spent")
            ax.set_xticks(x)
            ax.set_xticklabels([f"{cc}\n{EU_COUNTRY_PROFILES[cc].name}" for cc in country_codes])
            ax.legend()
            ax.grid(True, alpha=0.3, axis="y")
            plt.tight_layout()
            plt.savefig(out / "plot_epsilon_by_country.png", dpi=150, bbox_inches="tight")
            plt.close()

            # --- Plot 3: HDAB strictness vs opt-out ---
            fig, ax = plt.subplots(figsize=(8, 5))
            hdab_vals = [EU_COUNTRY_PROFILES[cc].hdab_strictness for cc in country_codes]
            optout_vals = [EU_COUNTRY_PROFILES[cc].opt_out_rate * 100 for cc in country_codes]
            colors_map = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(country_codes)))
            scatter = ax.scatter(hdab_vals, optout_vals, c=eps_max, cmap="coolwarm",
                                s=200, edgecolors="black", linewidth=1, zorder=5)
            for i, cc in enumerate(country_codes):
                ax.annotate(cc, (hdab_vals[i], optout_vals[i]),
                           textcoords="offset points", xytext=(8, 5), fontsize=10, fontweight="bold")
            ax.set_xlabel("HDAB Strictness (1-5)")
            ax.set_ylabel("Opt-out Rate (%)")
            ax.set_title("National HDAB Strictness vs. Opt-out Rate")
            ax.grid(True, alpha=0.3)
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label("Max Epsilon Allowed")
            plt.tight_layout()
            plt.savefig(out / "plot_hdab_vs_optout.png", dpi=150, bbox_inches="tight")
            plt.close()

            # --- Plot 4: Latency heatmap ---
            fig, ax = plt.subplots(figsize=(8, 5))
            latencies_avg = []
            labels = []
            for h in self.hospitals:
                lo, hi = h.country_profile.latency_ms
                latencies_avg.append((lo + hi) / 2)
                labels.append(f"{h.name}\n({h.country_code})")
            bars = ax.barh(range(len(self.hospitals)), latencies_avg, color="#4CAF50", alpha=0.8)
            ax.set_yticks(range(len(self.hospitals)))
            ax.set_yticklabels(labels, fontsize=8)
            ax.set_xlabel("Average Latency (ms)")
            ax.set_title("Network Latency per Hospital")
            ax.grid(True, alpha=0.3, axis="x")
            plt.tight_layout()
            plt.savefig(out / "plot_latency_per_hospital.png", dpi=150, bbox_inches="tight")
            plt.close()

        except ImportError:
            pass  # matplotlib not available

    def _save_jurisdiction_budget_plot(self, out: Path):
        """Plot per-client epsilon consumption over rounds with HDAB ceiling lines."""
        if not self.jurisdiction_manager:
            return
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(12, 6))

            # Country colors
            country_colors = {
                "DE": "#000000", "FR": "#0055A4", "IT": "#008C45",
                "ES": "#AA151B", "NL": "#FF6600", "SE": "#006AA7",
                "PL": "#DC143C", "AT": "#ED2939", "BE": "#FDDA24",
                "PT": "#006600",
            }

            # Plot per-client epsilon history
            for cid, state in self.jurisdiction_manager.client_states.items():
                if not state.epsilon_history:
                    continue
                cc = state.jurisdiction
                color = country_colors.get(cc, "#888888")
                rounds = list(range(len(state.epsilon_history)))
                ax.plot(
                    rounds, state.epsilon_history,
                    color=color, linewidth=1.5, alpha=0.8,
                    label=f"{state.hospital_name} ({cc})",
                )
                # Mark dropout point
                if state.deactivation_round is not None:
                    dr = min(state.deactivation_round, len(state.epsilon_history) - 1)
                    marker = "X" if state.opted_out else "s"
                    ax.scatter(
                        [dr], [state.epsilon_history[dr]],
                        color=color, marker=marker, s=100, zorder=5,
                        edgecolors="black", linewidth=1,
                    )

            # Draw HDAB ceiling lines
            drawn_ceilings = set()
            for cid, state in self.jurisdiction_manager.client_states.items():
                cc = state.jurisdiction
                if cc not in drawn_ceilings:
                    color = country_colors.get(cc, "#888888")
                    ax.axhline(
                        y=state.epsilon_ceiling, color=color,
                        linestyle="--", alpha=0.5, linewidth=1,
                    )
                    ax.text(
                        len(self.history) * 0.98, state.epsilon_ceiling,
                        f"{cc} ceiling={state.epsilon_ceiling:.1f}",
                        ha="right", va="bottom", fontsize=8, color=color,
                    )
                    drawn_ceilings.add(cc)

            ax.set_xlabel("Round")
            ax.set_ylabel("Cumulative Epsilon (RDP)")
            ax.set_title("Per-Client Privacy Budget Consumption by Jurisdiction")
            ax.legend(loc="upper left", fontsize=7, ncol=2)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(
                out / "plot_jurisdiction_budget.png",
                dpi=150, bbox_inches="tight",
            )
            plt.close()

        except ImportError:
            pass

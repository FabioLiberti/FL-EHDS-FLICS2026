"""
MyHealth@EU / NCPeH Integration Bridge for Federated Learning.

Implements the MyHealth@EU cross-border health data exchange infrastructure
as a bridge for FL training, following EHDS Chapter II (Art. 5-12).

Architecture:
- Each EU Member State has a National Contact Point for eHealth (NCPeH)
- NCPeH acts as a national aggregation node in 2-level hierarchical FL:
  Level 1: Hospitals -> NCPeH (national aggregation)
  Level 2: NCPeH -> EU aggregator (continental aggregation)
- MyHealth@EU services (Patient Summary, ePrescription) are simulated
  for cross-border data exchange compliance

References:
- EHDS Regulation (EU) 2025/327, Chapter II, Art. 5-12
- MyHealth@EU (ex eHDSI) technical specifications
- NCPeH architecture: Commission Implementing Decision (EU) 2019/1765

Author: Fabio Liberti
"""

import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class NCPeHNode:
    """National Contact Point for eHealth (NCPeH).

    Each EU Member State operates an NCPeH that connects to the
    MyHealth@EU central services in Brussels (BE).
    """
    country_code: str
    country_name: str
    ncp_id: str                         # e.g. "NCPeH-DE"
    lat: float
    lon: float
    bandwidth_mbps: float               # Inter-NCP bandwidth capacity
    infrastructure_tier: int            # 1=basic, 2=established, 3=advanced
    services: List[str] = field(default_factory=list)
    operational_since: int = 2021       # Year NCP became operational
    hospital_ids: List[int] = field(default_factory=list)
    # Per-round state
    national_samples: int = 0
    aggregation_latency_ms: float = 0.0
    communication_bytes: int = 0


@dataclass
class MyHealthEURoundMetrics:
    """Per-round metrics from MyHealth@EU bridge."""
    round_num: int
    ncp_aggregation_latencies: Dict[str, float] = field(default_factory=dict)
    inter_ncp_latencies: Dict[str, float] = field(default_factory=dict)
    total_hierarchical_latency_ms: float = 0.0
    communication_cost_bytes: int = 0
    flat_communication_cost_bytes: int = 0
    patient_summaries_exchanged: int = 0
    eprescriptions_processed: int = 0


# =============================================================================
# NCPeH TOPOLOGY (10 EU MEMBER STATES)
# =============================================================================
# Realistic NCPeH node data based on MyHealth@EU deployment status.
# Coordinates reuse EU_COUNTRY_PROFILES from terminal/cross_border.py.
# Bandwidth and tier based on EU Digital Economy and Society Index (DESI).

NCPEH_TOPOLOGY: Dict[str, NCPeHNode] = {
    "DE": NCPeHNode(
        country_code="DE", country_name="Germany",
        ncp_id="NCPeH-DE", lat=51.2, lon=10.4,
        bandwidth_mbps=100.0, infrastructure_tier=3,
        services=["patient_summary", "eprescription"],
        operational_since=2021,
    ),
    "FR": NCPeHNode(
        country_code="FR", country_name="France",
        ncp_id="NCPeH-FR", lat=46.6, lon=2.2,
        bandwidth_mbps=80.0, infrastructure_tier=3,
        services=["patient_summary", "eprescription"],
        operational_since=2021,
    ),
    "IT": NCPeHNode(
        country_code="IT", country_name="Italy",
        ncp_id="NCPeH-IT", lat=42.5, lon=12.5,
        bandwidth_mbps=60.0, infrastructure_tier=2,
        services=["patient_summary", "eprescription"],
        operational_since=2022,
    ),
    "ES": NCPeHNode(
        country_code="ES", country_name="Spain",
        ncp_id="NCPeH-ES", lat=40.0, lon=-3.7,
        bandwidth_mbps=60.0, infrastructure_tier=2,
        services=["patient_summary", "eprescription"],
        operational_since=2022,
    ),
    "NL": NCPeHNode(
        country_code="NL", country_name="Netherlands",
        ncp_id="NCPeH-NL", lat=52.1, lon=5.3,
        bandwidth_mbps=100.0, infrastructure_tier=3,
        services=["patient_summary", "eprescription"],
        operational_since=2020,
    ),
    "SE": NCPeHNode(
        country_code="SE", country_name="Sweden",
        ncp_id="NCPeH-SE", lat=62.0, lon=15.0,
        bandwidth_mbps=80.0, infrastructure_tier=3,
        services=["patient_summary", "eprescription"],
        operational_since=2020,
    ),
    "PL": NCPeHNode(
        country_code="PL", country_name="Poland",
        ncp_id="NCPeH-PL", lat=52.0, lon=20.0,
        bandwidth_mbps=40.0, infrastructure_tier=2,
        services=["patient_summary"],
        operational_since=2023,
    ),
    "AT": NCPeHNode(
        country_code="AT", country_name="Austria",
        ncp_id="NCPeH-AT", lat=47.5, lon=14.6,
        bandwidth_mbps=60.0, infrastructure_tier=2,
        services=["patient_summary", "eprescription"],
        operational_since=2021,
    ),
    "BE": NCPeHNode(
        country_code="BE", country_name="Belgium",
        ncp_id="NCPeH-BE", lat=50.85, lon=4.35,
        bandwidth_mbps=100.0, infrastructure_tier=3,
        services=["patient_summary", "eprescription", "central_services"],
        operational_since=2019,
    ),
    "PT": NCPeHNode(
        country_code="PT", country_name="Portugal",
        ncp_id="NCPeH-PT", lat=39.4, lon=-8.2,
        bandwidth_mbps=40.0, infrastructure_tier=2,
        services=["patient_summary", "eprescription"],
        operational_since=2022,
    ),
}


# =============================================================================
# PATIENT SUMMARY & ePRESCRIPTION SIMULATORS
# =============================================================================

class PatientSummarySimulator:
    """Simulates MyHealth@EU Patient Summary (PS) exchange per Art. 5-7.

    The Patient Summary contains essential clinical data for unscheduled
    cross-border care: demographics, allergies, conditions, medications.
    """

    # EHDS Art. 5 Patient Summary mandatory sections
    PS_SECTIONS = [
        "patient_demographics",
        "allergies_intolerances",
        "active_conditions",
        "current_medications",
        "immunizations",
        "medical_devices",
        "surgical_procedures",
        "diagnostic_results",
    ]

    def __init__(self, rng: np.random.RandomState):
        self._rng = rng
        self.total_exchanged = 0

    def simulate_exchange(
        self, ncp_from: str, ncp_to: str, num_records: int
    ) -> Dict[str, Any]:
        """Simulate PS exchange between two NCPs."""
        # Average PS record size: 2-5 KB (FHIR Bundle)
        avg_record_bytes = self._rng.randint(2048, 5120)
        total_bytes = num_records * avg_record_bytes
        # Sections included (all 8 for complete PS)
        sections = list(self.PS_SECTIONS)
        self.total_exchanged += num_records
        return {
            "from_ncp": ncp_from,
            "to_ncp": ncp_to,
            "num_records": num_records,
            "sections": sections,
            "total_bytes": total_bytes,
            "format": "FHIR R4 Bundle",
            "coding_system": "ICD-10",
        }


class ePrescriptionSimulator:
    """Simulates MyHealth@EU ePrescription/eDispensation per Art. 8-9.

    Cross-border ePrescription allows a prescription issued in country A
    to be dispensed in country B via the NCPeH infrastructure.
    """

    def __init__(self, rng: np.random.RandomState):
        self._rng = rng
        self.total_processed = 0

    def simulate_dispensation(
        self, ncp_prescribing: str, ncp_dispensing: str, num_prescriptions: int
    ) -> Dict[str, Any]:
        """Simulate cross-border ePrescription/eDispensation."""
        # Average ePrescription: 1-2 KB (FHIR MedicationRequest)
        avg_bytes = self._rng.randint(1024, 2048)
        total_bytes = num_prescriptions * avg_bytes
        self.total_processed += num_prescriptions
        return {
            "prescribing_ncp": ncp_prescribing,
            "dispensing_ncp": ncp_dispensing,
            "num_prescriptions": num_prescriptions,
            "total_bytes": total_bytes,
            "format": "FHIR R4 MedicationRequest",
            "coding_system": "ATC",
        }


# =============================================================================
# MYHEALTH@EU BRIDGE
# =============================================================================

def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine distance in km between two geographic coordinates."""
    R = 6371.0  # Earth radius in km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


class MyHealthEUBridge:
    """Bridge connecting MyHealth@EU NCPeH infrastructure to FL training.

    Provides:
    1. NCPeH topology with realistic inter-NCP latency model
    2. Hierarchical 2-level aggregation (hospital -> NCP -> EU)
    3. Patient Summary and ePrescription service simulation
    4. Communication cost tracking and comparison vs flat FL

    Integration pattern follows governance bridge convention
    (same as IHEFLBridge, JurisdictionPrivacyManager).
    """

    def __init__(
        self,
        hospitals: list,
        config: Dict[str, Any],
        seed: int = 42,
    ):
        self.config = config
        self.rng = np.random.RandomState(seed)
        self.ncp_nodes: Dict[str, NCPeHNode] = {}
        self.round_metrics: List[MyHealthEURoundMetrics] = []
        self._client_to_country: Dict[int, str] = {}
        self._session_start: Optional[datetime] = None
        self._latency_cache: Dict[str, float] = {}
        self.central_node = config.get("central_node", "BE")

        # Service simulators
        self._ps_sim = PatientSummarySimulator(self.rng)
        self._ep_sim = ePrescriptionSimulator(self.rng)

        # Build NCP topology from hospital list
        self._build_ncp_topology(hospitals)

        # Pre-compute inter-NCP latency matrix
        self._latency_matrix: Dict[Tuple[str, str], float] = {}
        self._build_latency_matrix()

    def _build_ncp_topology(self, hospitals: list) -> None:
        """Map hospitals to their country's NCPeH node."""
        for h in hospitals:
            cc = h.country_code
            self._client_to_country[h.hospital_id] = cc
            if cc not in self.ncp_nodes:
                if cc in NCPEH_TOPOLOGY:
                    template = NCPEH_TOPOLOGY[cc]
                    self.ncp_nodes[cc] = NCPeHNode(
                        country_code=template.country_code,
                        country_name=template.country_name,
                        ncp_id=template.ncp_id,
                        lat=template.lat,
                        lon=template.lon,
                        bandwidth_mbps=template.bandwidth_mbps,
                        infrastructure_tier=template.infrastructure_tier,
                        services=list(template.services),
                        operational_since=template.operational_since,
                        hospital_ids=[h.hospital_id],
                    )
                else:
                    # Fallback for unknown countries
                    self.ncp_nodes[cc] = NCPeHNode(
                        country_code=cc,
                        country_name=cc,
                        ncp_id=f"NCPeH-{cc}",
                        lat=50.0, lon=10.0,
                        bandwidth_mbps=40.0,
                        infrastructure_tier=1,
                        services=["patient_summary"],
                        operational_since=2024,
                        hospital_ids=[h.hospital_id],
                    )
            else:
                self.ncp_nodes[cc].hospital_ids.append(h.hospital_id)

    def _build_latency_matrix(self) -> None:
        """Pre-compute inter-NCP latency estimates."""
        codes = list(self.ncp_nodes.keys())
        for i, c1 in enumerate(codes):
            for c2 in codes[i + 1:]:
                lat = self._compute_inter_ncp_latency(
                    self.ncp_nodes[c1], self.ncp_nodes[c2])
                self._latency_matrix[(c1, c2)] = lat
                self._latency_matrix[(c2, c1)] = lat
            self._latency_matrix[(c1, c1)] = 0.0

    def _compute_inter_ncp_latency(
        self, ncp1: NCPeHNode, ncp2: NCPeHNode
    ) -> float:
        """Compute latency between two NCPeH nodes.

        Model: base_latency + propagation_delay + bandwidth_overhead + jitter
        - Propagation: ~5 us/km in fiber optic
        - Bandwidth: processing overhead inversely proportional to bandwidth
        - Jitter: Gaussian noise simulating network variability
        """
        distance_km = _haversine(ncp1.lat, ncp1.lon, ncp2.lat, ncp2.lon)
        propagation_ms = distance_km * 0.005  # ~5 us/km
        min_bw = min(ncp1.bandwidth_mbps, ncp2.bandwidth_mbps)
        bandwidth_overhead_ms = 1000.0 / min_bw if min_bw > 0 else 50.0
        base_ms = 5.0
        jitter = abs(self.rng.normal(0, 2.0))
        return base_ms + propagation_ms + bandwidth_overhead_ms + jitter

    def get_latency(self, cc1: str, cc2: str) -> float:
        """Get pre-computed latency between two countries (ms)."""
        return self._latency_matrix.get((cc1, cc2), 30.0)

    # -----------------------------------------------------------------
    # LIFECYCLE HOOKS (called by CrossBorderFederatedTrainer)
    # -----------------------------------------------------------------

    def start_session(self) -> None:
        """Initialize session, log NCP operational status."""
        self._session_start = datetime.now()

    def pre_round(self, round_num: int, active_hospitals: list) -> None:
        """Pre-round: simulate hospital-to-NCP upload and service exchanges.

        For each active NCP:
        1. Simulate hospital model upload to NCPeH (intra-country latency)
        2. Simulate Patient Summary exchange between NCPs (if enabled)
        3. Simulate ePrescription processing (if enabled)
        """
        ps_config = self.config.get("patient_summary_enabled", True)
        ep_config = self.config.get("eprescription_enabled", True)

        # Group active hospitals by country
        active_by_country: Dict[str, List] = defaultdict(list)
        for h in active_hospitals:
            active_by_country[h.country_code].append(h)

        # Simulate PS exchanges between active NCPs (pairwise)
        active_ncps = list(active_by_country.keys())
        if ps_config and len(active_ncps) >= 2:
            for i, c1 in enumerate(active_ncps):
                for c2 in active_ncps[i + 1:]:
                    n1 = self.ncp_nodes.get(c1)
                    n2 = self.ncp_nodes.get(c2)
                    if n1 and n2:
                        if ("patient_summary" in n1.services and
                                "patient_summary" in n2.services):
                            num_records = self.rng.randint(5, 25)
                            self._ps_sim.simulate_exchange(
                                c1, c2, num_records)

        # Simulate ePrescription (only between NCPs that support it)
        if ep_config and len(active_ncps) >= 2:
            for i, c1 in enumerate(active_ncps):
                for c2 in active_ncps[i + 1:]:
                    n1 = self.ncp_nodes.get(c1)
                    n2 = self.ncp_nodes.get(c2)
                    if n1 and n2:
                        if ("eprescription" in n1.services and
                                "eprescription" in n2.services):
                            num_presc = self.rng.randint(2, 10)
                            self._ep_sim.simulate_dispensation(
                                c1, c2, num_presc)

    def hierarchical_aggregate(
        self,
        trainer,
        client_results: list,
        quality_weights: Optional[Dict[int, float]] = None,
    ) -> None:
        """2-level hierarchical aggregation via NCPeH.

        Level 1 (National): For each country's NCP, aggregate its hospitals'
        model updates using sample-proportional (+ quality) weighting.

        Level 2 (EU): Aggregate national models across NCPs using the
        configured weight strategy (sample_proportional or equal).

        Modifies trainer.global_model in-place.
        """
        import torch

        # Group client_results by country (NCP)
        ncp_results: Dict[str, list] = defaultdict(list)
        for cr in client_results:
            country = self._client_to_country.get(cr.client_id)
            if country:
                ncp_results[country].append(cr)

        if not ncp_results:
            return

        # Level 1: National aggregation per NCP
        national_updates: Dict[str, Tuple[Dict, int]] = {}
        ncp_latencies = {}

        for country_code, cr_list in ncp_results.items():
            total_s = sum(cr.num_samples for cr in cr_list)
            if total_s == 0:
                continue

            # Compute per-hospital weights (with quality if available)
            raw_w = {}
            for cr in cr_list:
                w = cr.num_samples / total_s
                if quality_weights and cr.client_id in quality_weights:
                    w *= quality_weights[cr.client_id]
                raw_w[cr.client_id] = w
            # Normalize within NCP
            w_sum = sum(raw_w.values())
            if w_sum > 0:
                norm_w = {cid: w / w_sum for cid, w in raw_w.items()}
            else:
                norm_w = {cr.client_id: 1.0 / len(cr_list) for cr in cr_list}

            # Weighted average of hospital model updates
            param_names = list(cr_list[0].model_update.keys())
            nat_update = {}
            for name in param_names:
                weighted = torch.zeros_like(cr_list[0].model_update[name])
                for cr in cr_list:
                    weighted += cr.model_update[name] * norm_w[cr.client_id]
                nat_update[name] = weighted

            national_updates[country_code] = (nat_update, total_s)

            # NCP aggregation latency (proportional to hospital count)
            ncp = self.ncp_nodes.get(country_code)
            if ncp:
                lat_ms = len(cr_list) * 2.0 + abs(self.rng.normal(5.0, 1.0))
                ncp.national_samples = total_s
                ncp.aggregation_latency_ms = lat_ms
                ncp_latencies[country_code] = lat_ms

        if not national_updates:
            return

        # Level 2: EU aggregation across NCPs
        weight_strategy = self.config.get(
            "ncp_weight_strategy", "sample_proportional")
        total_eu = sum(s for _, s in national_updates.values())

        for name, param in trainer.global_model.named_parameters():
            eu_update = torch.zeros_like(param)
            for cc, (nat_upd, n_samples) in national_updates.items():
                if name not in nat_upd:
                    continue
                if weight_strategy == "equal":
                    w = 1.0 / len(national_updates)
                else:
                    w = n_samples / total_eu if total_eu > 0 else 1.0 / len(national_updates)
                eu_update += nat_upd[name] * w
            param.data += eu_update

        # Compute inter-NCP latencies (each NCP -> central node)
        inter_ncp_lats = {}
        central = self.central_node
        for cc in national_updates:
            if cc != central:
                key = f"{cc}->{central}"
                inter_ncp_lats[key] = self.get_latency(cc, central)

        # Communication cost estimation
        # Model size approximation (4 bytes per float32 parameter)
        model_params = sum(
            p.numel() for p in trainer.global_model.parameters())
        model_bytes = model_params * 4
        num_hospitals = sum(
            len(cr_list) for cr_list in ncp_results.values())
        num_ncps = len(national_updates)

        # Hierarchical: hospitals send to NCP (local), NCPs send to EU
        # Only NCP-level models traverse the WAN
        hier_bytes = num_hospitals * model_bytes + num_ncps * model_bytes
        # Flat: every hospital sends directly to EU central
        flat_bytes = num_hospitals * model_bytes * 2  # Upload + download

        total_hier_latency = (
            max(ncp_latencies.values()) if ncp_latencies else 0.0) + (
            max(inter_ncp_lats.values()) if inter_ncp_lats else 0.0)

        # Record round metrics
        self.round_metrics.append(MyHealthEURoundMetrics(
            round_num=len(self.round_metrics),
            ncp_aggregation_latencies=dict(ncp_latencies),
            inter_ncp_latencies=dict(inter_ncp_lats),
            total_hierarchical_latency_ms=total_hier_latency,
            communication_cost_bytes=hier_bytes,
            flat_communication_cost_bytes=flat_bytes,
            patient_summaries_exchanged=self._ps_sim.total_exchanged,
            eprescriptions_processed=self._ep_sim.total_processed,
        ))

        # Update NCP communication bytes
        for cc, ncp in self.ncp_nodes.items():
            if cc in national_updates:
                ncp.communication_bytes += model_bytes

    def post_round(self, round_num: int, metrics: Dict) -> None:
        """Post-round: record metrics (latencies are computed in aggregate)."""
        pass

    def end_session(self) -> None:
        """End session."""
        pass

    # -----------------------------------------------------------------
    # REPORTING
    # -----------------------------------------------------------------

    def export_report(self) -> Dict[str, Any]:
        """Export full MyHealth@EU report for auto-save JSON."""
        # NCP topology
        ncp_info = {}
        for cc, ncp in self.ncp_nodes.items():
            ncp_info[cc] = {
                "ncp_id": ncp.ncp_id,
                "country_name": ncp.country_name,
                "bandwidth_mbps": ncp.bandwidth_mbps,
                "infrastructure_tier": ncp.infrastructure_tier,
                "services": ncp.services,
                "operational_since": ncp.operational_since,
                "num_hospitals": len(ncp.hospital_ids),
                "hospital_ids": ncp.hospital_ids,
                "national_samples": ncp.national_samples,
                "aggregation_latency_ms": round(ncp.aggregation_latency_ms, 2),
                "communication_bytes": ncp.communication_bytes,
            }

        # Inter-NCP latency matrix
        latency_matrix = {}
        codes = sorted(self.ncp_nodes.keys())
        for c1 in codes:
            row = {}
            for c2 in codes:
                if c1 == c2:
                    row[c2] = 0.0
                else:
                    row[c2] = round(self.get_latency(c1, c2), 1)
            latency_matrix[c1] = row

        # Per-round metrics
        round_data = []
        for rm in self.round_metrics:
            round_data.append({
                "round": rm.round_num,
                "ncp_latencies": {k: round(v, 1)
                                  for k, v in rm.ncp_aggregation_latencies.items()},
                "inter_ncp_latencies": {k: round(v, 1)
                                        for k, v in rm.inter_ncp_latencies.items()},
                "total_latency_ms": round(rm.total_hierarchical_latency_ms, 1),
                "comm_bytes": rm.communication_cost_bytes,
                "flat_comm_bytes": rm.flat_communication_cost_bytes,
                "ps_exchanged": rm.patient_summaries_exchanged,
                "ep_processed": rm.eprescriptions_processed,
            })

        # Summary
        total_hier = sum(rm.communication_cost_bytes for rm in self.round_metrics)
        total_flat = sum(rm.flat_communication_cost_bytes for rm in self.round_metrics)
        saving_pct = ((total_flat - total_hier) / total_flat * 100
                      if total_flat > 0 else 0.0)

        return {
            "framework": "MyHealth@EU / NCPeH",
            "ehds_articles": "Chapter II, Art. 5-12",
            "session_start": str(self._session_start) if self._session_start else None,
            "central_node": self.central_node,
            "aggregation": "hierarchical_2_level",
            "ncp_weight_strategy": self.config.get("ncp_weight_strategy", "sample_proportional"),
            "ncp_topology": ncp_info,
            "inter_ncp_latency_matrix": latency_matrix,
            "round_metrics": round_data,
            "summary": {
                "num_ncps": len(self.ncp_nodes),
                "total_rounds": len(self.round_metrics),
                "total_hierarchical_bytes": total_hier,
                "total_flat_bytes": total_flat,
                "communication_saving_pct": round(saving_pct, 1),
                "patient_summaries_total": self._ps_sim.total_exchanged,
                "eprescriptions_total": self._ep_sim.total_processed,
            },
        }

    def get_display_summary(self) -> Dict[str, Any]:
        """Per-NCP summary for terminal display."""
        ncp_rows = []
        for cc in sorted(self.ncp_nodes.keys()):
            ncp = self.ncp_nodes[cc]
            # Average inter-NCP latency to central
            inter_lat = self.get_latency(cc, self.central_node) if cc != self.central_node else 0.0
            ncp_rows.append({
                "ncp_id": ncp.ncp_id,
                "country": ncp.country_name,
                "hospitals": len(ncp.hospital_ids),
                "samples": ncp.national_samples,
                "ncp_latency_ms": round(ncp.aggregation_latency_ms, 1),
                "inter_ncp_ms": round(inter_lat, 1),
                "bandwidth_mbps": ncp.bandwidth_mbps,
                "tier": ncp.infrastructure_tier,
                "services": ", ".join(
                    s.replace("patient_summary", "PS").replace(
                        "eprescription", "eP").replace(
                        "central_services", "Central")
                    for s in ncp.services
                ),
            })

        # Communication cost comparison
        total_hier = sum(rm.communication_cost_bytes for rm in self.round_metrics)
        total_flat = sum(rm.flat_communication_cost_bytes for rm in self.round_metrics)
        saving_pct = ((total_flat - total_hier) / total_flat * 100
                      if total_flat > 0 else 0.0)

        return {
            "ncp_rows": ncp_rows,
            "central_node": f"NCPeH-{self.central_node}",
            "aggregation_type": "2-level hierarchical (Hospital -> NCP -> EU)",
            "total_hier_kb": round(total_hier / 1024, 1),
            "total_flat_kb": round(total_flat / 1024, 1),
            "saving_pct": round(saving_pct, 1),
            "ps_total": self._ps_sim.total_exchanged,
            "ep_total": self._ep_sim.total_processed,
        }

    def get_latency_matrix_display(self) -> Tuple[List[str], Dict[str, Dict[str, float]]]:
        """Get latency matrix for display.

        Returns:
            (country_codes, {from_cc: {to_cc: latency_ms}})
        """
        codes = sorted(self.ncp_nodes.keys())
        matrix = {}
        for c1 in codes:
            row = {}
            for c2 in codes:
                if c1 == c2:
                    row[c2] = 0.0
                else:
                    row[c2] = round(self.get_latency(c1, c2), 1)
            matrix[c1] = row
        return codes, matrix

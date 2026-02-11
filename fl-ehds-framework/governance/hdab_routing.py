"""
EHDS Art. 57-58: Cross-Border Data Access Request Routing.

Implements the "HealthData@EU" Single Application Point (SAP) concept:
- Single Application Point: unified entry for multi-country requests
- HDAB Routing: intelligent routing to relevant national HDABs
- Lead HDAB Selection: coordinator election with configurable strategy
- Joint Approval: coordinated multi-HDAB decision with veto power

Art. 57 establishes the HealthData@EU platform for secondary use of
health data across Member States.  Art. 58 mandates a single application
point where researchers submit one request that is routed to the relevant
HDABs, with one HDAB acting as lead coordinator.

Author: Fabio Liberti
"""

import logging
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ApplicationState(Enum):
    """EHDS Art. 58 application lifecycle states."""
    SUBMITTED = "submitted"
    ROUTING = "routing"
    UNDER_REVIEW = "under_review"
    JOINT_DECISION = "joint_decision"
    APPROVED = "approved"
    REJECTED = "rejected"
    TRAINING_AUTHORIZED = "training_authorized"
    COMPLETED = "completed"


class HDABDecision(Enum):
    """Individual HDAB decision on a data access application."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    DEFERRED = "deferred"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class DataAccessApplication:
    """Cross-border data access application (EHDS Art. 57)."""
    application_id: str
    requester_id: str
    purpose: str
    data_categories: List[str]
    requested_countries: List[str]
    submitted_at: datetime = field(default_factory=datetime.utcnow)
    state: ApplicationState = ApplicationState.SUBMITTED
    routed_to_hdabs: List[str] = field(default_factory=list)
    lead_hdab: Optional[str] = None
    hdab_decisions: Dict[str, HDABDecision] = field(default_factory=dict)
    decision_rationales: Dict[str, str] = field(default_factory=dict)
    joint_decision: Optional[HDABDecision] = None
    state_history: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class RoutingDecision:
    """Result of HDAB routing analysis."""
    application_id: str
    target_hdabs: List[str]
    lead_hdab: str
    routing_strategy: str
    routing_rationale: str
    hdab_scores: Dict[str, float] = field(default_factory=dict)
    data_available: Dict[str, int] = field(default_factory=dict)
    purpose_compatible: Dict[str, bool] = field(default_factory=dict)
    routed_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class JointApprovalResult:
    """Result of coordinated multi-HDAB joint approval."""
    session_id: str
    application_id: str
    lead_hdab: str
    participating_hdabs: List[str]
    decisions: Dict[str, HDABDecision] = field(default_factory=dict)
    rationales: Dict[str, str] = field(default_factory=dict)
    joint_decision: HDABDecision = HDABDecision.PENDING
    consensus_level: float = 0.0
    veto_exercised_by: Optional[str] = None
    decided_at: datetime = field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# HDABRoutingBridge
# ---------------------------------------------------------------------------

class HDABRoutingBridge:
    """
    EHDS Art. 57-58 Single Application Point and HDAB Routing.

    Simulates the HealthData@EU platform for cross-border data access:
    1. Researcher submits a single application to the SAP
    2. Application is routed to relevant national HDABs
    3. A Lead HDAB is elected to coordinate the process
    4. Each HDAB reviews and decides independently
    5. Joint decision is made (unanimous or consensus-based)
    6. Training is authorized if approved
    """

    def __init__(
        self,
        hospitals: List,
        countries: List[str],
        purpose: str,
        config: Dict[str, Any],
    ):
        self._hospitals = hospitals
        self._countries = countries
        self._purpose = purpose
        self._config = config

        # Import country profiles lazily
        from terminal.cross_border import EU_COUNTRY_PROFILES
        self._country_profiles = {
            cc: EU_COUNTRY_PROFILES[cc]
            for cc in countries
            if cc in EU_COUNTRY_PROFILES
        }

        # RNG for decision simulation
        self._rng = np.random.RandomState(config.get("seed", 42))

        # Results (populated by process_application)
        self._application: Optional[DataAccessApplication] = None
        self._routing_decision: Optional[RoutingDecision] = None
        self._approval_result: Optional[JointApprovalResult] = None

    # -----------------------------------------------------------------
    # PUBLIC API
    # -----------------------------------------------------------------

    def process_application(
        self, requester_id: str = "fl-ehds-framework"
    ) -> Dict[str, Any]:
        """
        Process a cross-border data access application through the SAP.

        Full workflow:
        1. Create application
        2. Route to relevant HDABs
        3. Select Lead HDAB
        4. Simulate joint approval
        5. Return authorization result

        Returns:
            Dict with keys: approved, lead_hdab, routed_hdabs,
            rejection_reason, routing_decision, approval_result
        """
        # Step 1: Create application
        app = self._create_application(requester_id)
        self._application = app
        self._transition_state(app, ApplicationState.SUBMITTED)

        logger.info(
            f"SAP: Application {app.application_id} submitted "
            f"for purpose={app.purpose}, countries={app.requested_countries}"
        )

        # Step 2: Route to relevant HDABs
        self._transition_state(app, ApplicationState.ROUTING)
        routing = self._route_application(app)
        self._routing_decision = routing

        if not routing.target_hdabs:
            self._transition_state(app, ApplicationState.REJECTED)
            app.joint_decision = HDABDecision.REJECTED
            return {
                "approved": False,
                "lead_hdab": None,
                "routed_hdabs": [],
                "rejection_reason": "No eligible HDABs found for this purpose",
                "routing_decision": routing,
                "approval_result": None,
            }

        logger.info(
            f"SAP: Routed to {len(routing.target_hdabs)} HDABs: "
            f"{routing.target_hdabs}, Lead: {routing.lead_hdab}"
        )

        # Step 3: Distribute to HDABs for review
        self._transition_state(app, ApplicationState.UNDER_REVIEW)
        app.routed_to_hdabs = routing.target_hdabs
        app.lead_hdab = routing.lead_hdab

        # Step 4: Joint approval
        self._transition_state(app, ApplicationState.JOINT_DECISION)
        approval = self._simulate_joint_approval(
            app, routing.lead_hdab, routing.target_hdabs
        )
        self._approval_result = approval

        # Step 5: Final decision
        app.hdab_decisions = dict(approval.decisions)
        app.decision_rationales = dict(approval.rationales)
        app.joint_decision = approval.joint_decision

        if approval.joint_decision == HDABDecision.APPROVED:
            self._transition_state(app, ApplicationState.APPROVED)
            self._transition_state(app, ApplicationState.TRAINING_AUTHORIZED)
            logger.info(
                f"SAP: Application APPROVED. "
                f"Consensus: {approval.consensus_level:.0%}"
            )
            return {
                "approved": True,
                "lead_hdab": routing.lead_hdab,
                "routed_hdabs": routing.target_hdabs,
                "rejection_reason": None,
                "routing_decision": routing,
                "approval_result": approval,
            }
        else:
            self._transition_state(app, ApplicationState.REJECTED)
            veto_info = ""
            if approval.veto_exercised_by:
                veto_info = f" (veto by {approval.veto_exercised_by})"
            reason = (
                f"Joint decision: REJECTED{veto_info}. "
                f"Consensus: {approval.consensus_level:.0%}"
            )
            logger.warning(f"SAP: Application REJECTED. {reason}")
            return {
                "approved": False,
                "lead_hdab": routing.lead_hdab,
                "routed_hdabs": routing.target_hdabs,
                "rejection_reason": reason,
                "routing_decision": routing,
                "approval_result": approval,
            }

    def end_session(self):
        """Mark application as completed."""
        if self._application and self._application.state in (
            ApplicationState.TRAINING_AUTHORIZED,
            ApplicationState.APPROVED,
        ):
            self._transition_state(
                self._application, ApplicationState.COMPLETED
            )

    def export_report(self) -> Dict[str, Any]:
        """Export full routing and approval report."""
        report: Dict[str, Any] = {
            "module": "hdab_routing",
            "regulation_ref": "EHDS Art. 57-58",
        }

        if self._application:
            app = self._application
            report["application"] = {
                "application_id": app.application_id,
                "requester_id": app.requester_id,
                "purpose": app.purpose,
                "data_categories": app.data_categories,
                "requested_countries": app.requested_countries,
                "submitted_at": str(app.submitted_at),
                "final_state": app.state.value,
                "joint_decision": (
                    app.joint_decision.value if app.joint_decision else None
                ),
                "state_history": app.state_history,
            }

        if self._routing_decision:
            rd = self._routing_decision
            report["routing"] = {
                "target_hdabs": rd.target_hdabs,
                "lead_hdab": rd.lead_hdab,
                "routing_strategy": rd.routing_strategy,
                "routing_rationale": rd.routing_rationale,
                "hdab_scores": rd.hdab_scores,
                "data_available": rd.data_available,
                "purpose_compatible": rd.purpose_compatible,
                "routed_at": str(rd.routed_at),
            }

        if self._approval_result:
            ar = self._approval_result
            report["approval"] = {
                "session_id": ar.session_id,
                "lead_hdab": ar.lead_hdab,
                "participating_hdabs": ar.participating_hdabs,
                "individual_decisions": {
                    cc: d.value for cc, d in ar.decisions.items()
                },
                "rationales": ar.rationales,
                "joint_decision": ar.joint_decision.value,
                "consensus_level": ar.consensus_level,
                "veto_exercised_by": ar.veto_exercised_by,
                "decided_at": str(ar.decided_at),
            }

        return report

    # -----------------------------------------------------------------
    # INTERNAL: Application creation
    # -----------------------------------------------------------------

    def _create_application(
        self, requester_id: str
    ) -> DataAccessApplication:
        """Create a DataAccessApplication from trainer configuration."""
        app_id = f"APP-{uuid.uuid4().hex[:8].upper()}"
        return DataAccessApplication(
            application_id=app_id,
            requester_id=requester_id,
            purpose=self._purpose,
            data_categories=["ehr"],  # default; could be extended
            requested_countries=list(self._countries),
        )

    # -----------------------------------------------------------------
    # INTERNAL: Routing
    # -----------------------------------------------------------------

    def _route_application(
        self, app: DataAccessApplication
    ) -> RoutingDecision:
        """
        Route application to relevant HDABs based on strategy.

        Strategies:
        - data_driven: prioritize countries with most available data
        - comprehensive: include all purpose-compatible countries
        - minimal: fewest countries meeting requirements (cost-optimized)
        """
        strategy = self._config.get("routing_strategy", "data_driven")

        # Phase 1: Check purpose compatibility per country
        purpose_compatible: Dict[str, bool] = {}
        for cc, profile in self._country_profiles.items():
            purpose_compatible[cc] = (
                self._purpose in profile.allowed_purposes
            )

        eligible = [
            cc for cc, compat in purpose_compatible.items() if compat
        ]

        # Phase 2: Assess data availability per country
        data_available: Dict[str, int] = defaultdict(int)
        for h in self._hospitals:
            if h.country_code in eligible:
                samples = getattr(h, "num_samples_after_optout", None)
                if samples is None or samples == 0:
                    samples = getattr(h, "num_samples", 0)
                data_available[h.country_code] += samples

        # Phase 3: Score each country
        max_data = max(data_available.values()) if data_available else 1
        hdab_scores: Dict[str, float] = {}

        for cc in eligible:
            data_norm = data_available[cc] / max_data if max_data > 0 else 0
            strictness = self._country_profiles[cc].hdab_strictness
            strictness_norm = 1 - (strictness / 5.0)

            if strategy == "data_driven":
                hdab_scores[cc] = data_norm
            elif strategy == "comprehensive":
                hdab_scores[cc] = 1.0 if data_available[cc] > 0 else 0.0
            elif strategy == "minimal":
                hdab_scores[cc] = data_norm * strictness_norm
            else:
                hdab_scores[cc] = 0.7 * data_norm + 0.3 * strictness_norm

        # Phase 4: Select target HDABs
        min_countries = self._config.get("min_countries", 2)

        if strategy == "comprehensive":
            target_hdabs = [
                cc for cc in eligible if data_available[cc] > 0
            ]
        else:
            # Sort by score descending
            sorted_countries = sorted(
                eligible,
                key=lambda cc: hdab_scores.get(cc, 0),
                reverse=True,
            )
            # Select countries with data
            target_hdabs = [
                cc for cc in sorted_countries if data_available[cc] > 0
            ]

        # Ensure minimum countries
        if len(target_hdabs) < min_countries and len(eligible) >= min_countries:
            sorted_all = sorted(
                eligible,
                key=lambda cc: hdab_scores.get(cc, 0),
                reverse=True,
            )
            target_hdabs = sorted_all[:min_countries]

        # Phase 5: Select Lead HDAB
        lead_hdab = ""
        if target_hdabs:
            lead_hdab = self._select_lead_hdab(target_hdabs)

        # Build rationale
        n_compat = len(eligible)
        n_total = len(self._country_profiles)
        total_samples = sum(data_available[cc] for cc in target_hdabs)
        rationale = (
            f"Strategy: {strategy}. "
            f"Purpose-compatible: {n_compat}/{n_total} countries. "
            f"Selected: {len(target_hdabs)} HDABs. "
            f"Total data: {total_samples} samples. "
            f"Lead HDAB: {lead_hdab}."
        )

        return RoutingDecision(
            application_id=app.application_id,
            target_hdabs=target_hdabs,
            lead_hdab=lead_hdab,
            routing_strategy=strategy,
            routing_rationale=rationale,
            hdab_scores=dict(hdab_scores),
            data_available=dict(data_available),
            purpose_compatible=purpose_compatible,
        )

    # -----------------------------------------------------------------
    # INTERNAL: Lead HDAB Selection
    # -----------------------------------------------------------------

    def _select_lead_hdab(self, candidates: List[str]) -> str:
        """
        Select lead HDAB from candidates using configured strategy.

        Strategies:
        - most_data: country with largest available dataset
        - lowest_strictness: most permissive HDAB
        - balanced: weighted combination of data volume and permissiveness
        """
        strategy = self._config.get("lead_selection_strategy", "most_data")

        # Compute per-country samples
        country_samples: Dict[str, int] = defaultdict(int)
        for h in self._hospitals:
            if h.country_code in candidates:
                samples = getattr(h, "num_samples_after_optout", None)
                if samples is None or samples == 0:
                    samples = getattr(h, "num_samples", 0)
                country_samples[h.country_code] += samples

        if strategy == "most_data":
            if country_samples:
                return max(
                    candidates,
                    key=lambda cc: country_samples.get(cc, 0),
                )
            return candidates[0]

        elif strategy == "lowest_strictness":
            return min(
                candidates,
                key=lambda cc: self._country_profiles[cc].hdab_strictness,
            )

        elif strategy == "balanced":
            max_samples = max(country_samples.values()) if country_samples else 1
            scores = {}
            for cc in candidates:
                data_score = country_samples.get(cc, 0) / max_samples
                strict_score = 1 - (
                    self._country_profiles[cc].hdab_strictness / 5.0
                )
                scores[cc] = 0.6 * data_score + 0.4 * strict_score
            return max(candidates, key=lambda cc: scores.get(cc, 0))

        # Fallback
        return candidates[0]

    # -----------------------------------------------------------------
    # INTERNAL: Joint Approval
    # -----------------------------------------------------------------

    def _simulate_joint_approval(
        self,
        app: DataAccessApplication,
        lead_hdab: str,
        participating_hdabs: List[str],
    ) -> JointApprovalResult:
        """
        Simulate coordinated multi-HDAB joint approval.

        Each HDAB decides independently based on purpose compatibility
        and HDAB strictness.  The joint decision uses either:
        - Veto model: any rejection blocks approval
        - Consensus model: requires threshold fraction of approvals
        """
        session_id = f"JAS-{uuid.uuid4().hex[:8].upper()}"
        veto_power = self._config.get("veto_power", True)
        consensus_threshold = self._config.get("consensus_threshold", 1.0)

        decisions: Dict[str, HDABDecision] = {}
        rationales: Dict[str, str] = {}

        # Each HDAB reviews independently
        for cc in participating_hdabs:
            decision, rationale = self._simulate_hdab_decision(cc, app)
            decisions[cc] = decision
            rationales[cc] = rationale

        # Calculate consensus
        n_approved = sum(
            1 for d in decisions.values() if d == HDABDecision.APPROVED
        )
        n_total = len(decisions)
        consensus_level = n_approved / n_total if n_total > 0 else 0.0

        # Joint decision
        veto_exercised_by = None

        if veto_power:
            # Any rejection blocks
            rejected_by = [
                cc for cc, d in decisions.items()
                if d == HDABDecision.REJECTED
            ]
            if rejected_by:
                joint_decision = HDABDecision.REJECTED
                veto_exercised_by = rejected_by[0]
            else:
                joint_decision = HDABDecision.APPROVED
        else:
            # Consensus-based
            if consensus_level >= consensus_threshold:
                joint_decision = HDABDecision.APPROVED
            else:
                joint_decision = HDABDecision.REJECTED

        return JointApprovalResult(
            session_id=session_id,
            application_id=app.application_id,
            lead_hdab=lead_hdab,
            participating_hdabs=participating_hdabs,
            decisions=decisions,
            rationales=rationales,
            joint_decision=joint_decision,
            consensus_level=consensus_level,
            veto_exercised_by=veto_exercised_by,
        )

    def _simulate_hdab_decision(
        self,
        country_code: str,
        app: DataAccessApplication,
    ) -> Tuple[HDABDecision, str]:
        """
        Simulate a single HDAB's decision on the application.

        Decision logic:
        1. Check purpose compatibility (allowed_purposes) -> reject if not
        2. Probabilistic approval based on strictness:
           - strictness 1 -> 95% approval probability
           - strictness 5 -> 75% approval probability
        3. Return (decision, rationale)
        """
        profile = self._country_profiles.get(country_code)
        if profile is None:
            return (
                HDABDecision.REJECTED,
                f"HDAB {country_code}: Unknown country profile",
            )

        # Check purpose compatibility
        if app.purpose not in profile.allowed_purposes:
            return (
                HDABDecision.REJECTED,
                f"HDAB {country_code}: Purpose '{app.purpose}' not in "
                f"allowed purposes ({', '.join(profile.allowed_purposes)})",
            )

        # Probabilistic approval based on strictness
        # strictness 1 -> 0.95, strictness 5 -> 0.75
        approval_prob = 1.0 - (profile.hdab_strictness * 0.05)
        roll = self._rng.random()

        if roll < approval_prob:
            return (
                HDABDecision.APPROVED,
                f"HDAB {country_code} ({profile.name}): "
                f"Approved for '{app.purpose}' "
                f"(strictness={profile.hdab_strictness}, "
                f"p={approval_prob:.0%})",
            )
        else:
            return (
                HDABDecision.REJECTED,
                f"HDAB {country_code} ({profile.name}): "
                f"Rejected - additional safeguards required "
                f"(strictness={profile.hdab_strictness}, "
                f"p={approval_prob:.0%})",
            )

    # -----------------------------------------------------------------
    # INTERNAL: State management
    # -----------------------------------------------------------------

    def _transition_state(
        self,
        app: DataAccessApplication,
        new_state: ApplicationState,
    ) -> None:
        """Record state transition in application history."""
        app.state = new_state
        app.state_history.append({
            "state": new_state.value,
            "timestamp": datetime.utcnow().isoformat(),
        })

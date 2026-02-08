"""
Jurisdiction-Level Privacy Budget Management
=============================================
Implements hierarchical differential privacy for cross-border federated
learning under EHDS: each country's HDAB sets a national epsilon ceiling,
and each hospital (FL node) has its own RDP accountant bounded by that ceiling.

Two-level hierarchy:
1. HDAB (national): epsilon ceiling from data protection authority
2. Hospital (node): individual RDP budget <= national ceiling

When a hospital exhausts its privacy budget, it stops participating but
training continues with the remaining clients. Supports Art. 48 EHDS
opt-out simulation (country-level withdrawal at any round).

Reuses:
- PrivacyAccountant from orchestration/privacy/differential_privacy.py
- EU_COUNTRY_PROFILES from terminal/cross_border.py (for defaults)
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from orchestration.privacy.differential_privacy import PrivacyAccountant

logger = logging.getLogger(__name__)


@dataclass
class ClientPrivacyState:
    """Per-client privacy tracking state."""
    client_id: int
    jurisdiction: str             # ISO 3166-1 alpha-2 country code
    hospital_name: str
    epsilon_ceiling: float        # min(global, national) * allocation_fraction
    delta: float
    accountant: PrivacyAccountant
    calibrated_noise_multiplier: float = 1.0  # σ/Δ calibrated for budget over num_rounds
    active: bool = True
    rounds_participated: int = 0
    deactivation_round: Optional[int] = None
    deactivation_reason: Optional[str] = None
    opted_out: bool = False
    optout_round: Optional[int] = None
    # History of per-round epsilon spending
    epsilon_history: List[float] = field(default_factory=list)


class JurisdictionPrivacyManager:
    """
    Per-jurisdiction, per-client privacy budget accounting.

    Creates one PrivacyAccountant per client, each bounded by
    min(global_epsilon, national_ceiling) * hospital_allocation_fraction.

    Provides:
    - Pre-round client filtering (skip exhausted/opted-out clients)
    - Post-round RDP accounting per client
    - Noise scale computation (global strategy: strictest active client)
    - Art. 48 opt-out simulation
    - Comprehensive reporting for audit trail
    """

    def __init__(
        self,
        client_jurisdictions: Dict[int, str],
        jurisdiction_budgets: Dict[str, Dict],
        global_epsilon: float,
        num_rounds: int = 30,
        hospital_names: Optional[Dict[int, str]] = None,
        hospital_allocation_fraction: float = 1.0,
        noise_strategy: str = "global",
        accountant_type: str = "rdp",
        min_active_clients: int = 2,
    ):
        """
        Args:
            client_jurisdictions: {client_id: country_code}
            jurisdiction_budgets: {country_code: {"epsilon_max": float, "delta": float}}
            global_epsilon: Global epsilon ceiling across all jurisdictions
            num_rounds: Expected number of training rounds (for noise calibration)
            hospital_names: {client_id: hospital_name} (optional, for display)
            hospital_allocation_fraction: Fraction of national ceiling per hospital
            noise_strategy: "global" = noise calibrated to strictest active client
            accountant_type: "rdp", "simple", or "advanced"
            min_active_clients: Minimum clients needed to continue training
        """
        self.client_jurisdictions = client_jurisdictions
        self.jurisdiction_budgets = jurisdiction_budgets
        self.global_epsilon = global_epsilon
        self.num_rounds = num_rounds
        self.hospital_names = hospital_names or {}
        self.hospital_allocation_fraction = hospital_allocation_fraction
        self.noise_strategy = noise_strategy
        self.min_active_clients = min_active_clients
        self._dropout_events: List[Dict[str, Any]] = []

        # Create per-client privacy state with dedicated RDP accountant
        # Each client gets a calibrated noise_multiplier via compute_noise_for_target_epsilon
        self.client_states: Dict[int, ClientPrivacyState] = {}
        for client_id, country_code in client_jurisdictions.items():
            budget = jurisdiction_budgets.get(country_code, {})
            national_ceiling = budget.get("epsilon_max", global_epsilon)
            delta = budget.get("delta", 1e-5)

            # Client budget = min(global, national) * allocation fraction
            client_epsilon = min(global_epsilon, national_ceiling) * hospital_allocation_fraction

            accountant = PrivacyAccountant(
                total_epsilon=client_epsilon,
                total_delta=delta,
                accountant_type=accountant_type,
            )

            # Calibrate noise: find σ/Δ that spreads this client's budget over num_rounds
            calibrated_sigma = accountant.compute_noise_for_target_epsilon(
                target_epsilon=client_epsilon,
                num_rounds=num_rounds,
            )

            self.client_states[client_id] = ClientPrivacyState(
                client_id=client_id,
                jurisdiction=country_code,
                hospital_name=self.hospital_names.get(client_id, f"Client_{client_id}"),
                epsilon_ceiling=client_epsilon,
                delta=delta,
                accountant=accountant,
                calibrated_noise_multiplier=calibrated_sigma,
            )
            logger.info(
                f"Client {client_id} ({country_code}): "
                f"eps_ceiling={client_epsilon:.2f}, "
                f"calibrated_sigma={calibrated_sigma:.4f}"
            )

        logger.info(
            f"JurisdictionPrivacyManager: {len(self.client_states)} clients, "
            f"{len(set(client_jurisdictions.values()))} jurisdictions, "
            f"strategy={noise_strategy}, num_rounds={num_rounds}"
        )

    def get_active_clients(self) -> List[int]:
        """Return client_ids that can still participate."""
        return [
            cid for cid, state in self.client_states.items()
            if state.active and not state.opted_out
        ]

    def pre_round_check(
        self, round_num: int, noise_multiplier: float = 0.0
    ) -> Tuple[List[int], float]:
        """
        Pre-round: determine active clients and compute calibrated noise scale.

        For each active client, checks if their privacy budget has remaining
        epsilon. Deactivates clients whose budget is exhausted.

        The effective noise scale is the MAX calibrated noise multiplier among
        active clients (strictest client needs most noise to stay within budget).
        This is the σ/Δ that should be applied to the aggregated model update.

        Args:
            round_num: Current round number
            noise_multiplier: Base noise multiplier (ignored when using calibrated noise)

        Returns:
            (active_client_ids, effective_noise_scale)
            noise_scale is the calibrated σ/Δ satisfying the strictest active client
        """
        active_clients = []

        for cid in self.get_active_clients():
            state = self.client_states[cid]
            remaining_eps = state.accountant.get_remaining_budget()[0]

            if remaining_eps > 0:
                active_clients.append(cid)
            else:
                # Deactivate this client
                state.active = False
                state.deactivation_round = round_num
                state.deactivation_reason = "budget_exhausted"
                self._dropout_events.append({
                    "round": round_num,
                    "client_id": cid,
                    "hospital": state.hospital_name,
                    "country": state.jurisdiction,
                    "reason": "budget_exhausted",
                    "epsilon_spent": state.accountant.get_spent_budget()[0],
                    "epsilon_ceiling": state.epsilon_ceiling,
                })
                logger.info(
                    f"Client {cid} ({state.hospital_name}, {state.jurisdiction}) "
                    f"deactivated at round {round_num}: budget exhausted "
                    f"(spent={state.accountant.get_spent_budget()[0]:.4f}, "
                    f"ceiling={state.epsilon_ceiling:.4f})"
                )

        # Compute effective noise scale from calibrated per-client noise multipliers
        if self.noise_strategy == "global" and active_clients:
            # Strictest client (lowest ε) has highest calibrated σ/Δ
            # Use MAX to satisfy all active clients' budgets
            effective_noise_scale = max(
                self.client_states[cid].calibrated_noise_multiplier
                for cid in active_clients
            )
        else:
            effective_noise_scale = noise_multiplier

        return active_clients, effective_noise_scale

    def record_round(
        self,
        round_num: int,
        participating_clients: List[int],
        effective_noise_multiplier: float,
        sampling_rate: float = 1.0,
    ):
        """
        Post-round: record epsilon spending for each participating client.

        Uses the PrivacyAccountant.spend() method which internally tracks
        RDP composition. The effective_noise_multiplier should be the calibrated
        σ/Δ that was actually applied to the model (from pre_round_check).

        If spending exceeds a client's budget, deactivates the client.

        Args:
            round_num: Current round number
            participating_clients: Client IDs that participated this round
            effective_noise_multiplier: Calibrated noise σ/Δ used this round
            sampling_rate: Fraction of data sampled per round
        """
        for cid in participating_clients:
            state = self.client_states[cid]
            if not state.active or state.opted_out:
                continue

            try:
                # Record spending via RDP accountant using the calibrated noise
                # that was actually applied to the model this round
                per_round_eps = state.epsilon_ceiling / max(1, self.num_rounds)
                state.accountant.spend(
                    epsilon=per_round_eps,
                    delta=state.delta,
                    round_number=round_num,
                    mechanism="gaussian",
                    noise_multiplier=effective_noise_multiplier,
                    sampling_rate=sampling_rate,
                )
                state.rounds_participated += 1
                spent = state.accountant.get_spent_budget()[0]
                state.epsilon_history.append(spent)

            except Exception:
                # PrivacyBudgetExceededError -> deactivate client
                state.active = False
                state.deactivation_round = round_num
                state.deactivation_reason = "budget_exceeded_on_spend"
                spent = state.accountant.get_spent_budget()[0]
                state.epsilon_history.append(spent)
                self._dropout_events.append({
                    "round": round_num,
                    "client_id": cid,
                    "hospital": state.hospital_name,
                    "country": state.jurisdiction,
                    "reason": "budget_exceeded_on_spend",
                    "epsilon_spent": spent,
                    "epsilon_ceiling": state.epsilon_ceiling,
                })
                logger.info(
                    f"Client {cid} ({state.hospital_name}) budget exceeded "
                    f"at round {round_num}, deactivated"
                )

    def simulate_optout(self, country_code: str, from_round: int) -> List[int]:
        """
        Simulate Art. 48 EHDS opt-out: deactivate all clients from a country.

        Does not destroy state (for before/after comparison).

        Args:
            country_code: ISO country code to opt out
            from_round: Round number from which opt-out takes effect

        Returns:
            List of deactivated client_ids
        """
        removed = []
        for cid, state in self.client_states.items():
            if state.jurisdiction == country_code and state.active:
                state.active = False
                state.opted_out = True
                state.optout_round = from_round
                state.deactivation_round = from_round
                state.deactivation_reason = "art48_optout"
                removed.append(cid)
                self._dropout_events.append({
                    "round": from_round,
                    "client_id": cid,
                    "hospital": state.hospital_name,
                    "country": state.jurisdiction,
                    "reason": "art48_optout",
                    "epsilon_spent": state.accountant.get_spent_budget()[0],
                    "epsilon_ceiling": state.epsilon_ceiling,
                })

        logger.info(
            f"Art. 48 opt-out: {country_code} withdrew at round {from_round}, "
            f"{len(removed)} clients deactivated"
        )
        return removed

    def get_jurisdiction_status(self) -> Dict[str, Dict[str, Any]]:
        """Return per-jurisdiction privacy budget summary."""
        jurisdictions: Dict[str, Dict[str, Any]] = {}

        for cid, state in self.client_states.items():
            cc = state.jurisdiction
            if cc not in jurisdictions:
                jurisdictions[cc] = {
                    "country_code": cc,
                    "epsilon_ceiling": state.epsilon_ceiling,
                    "total": 0,
                    "active": 0,
                    "exhausted": 0,
                    "opted_out": 0,
                    "epsilon_spent_max": 0.0,
                    "remaining_min": float("inf"),
                    "clients": [],
                }

            info = jurisdictions[cc]
            info["total"] += 1
            info["clients"].append(cid)

            spent = state.accountant.get_spent_budget()[0]
            remaining = state.accountant.get_remaining_budget()[0]

            info["epsilon_spent_max"] = max(info["epsilon_spent_max"], spent)

            if state.active and not state.opted_out:
                info["active"] += 1
                info["remaining_min"] = min(info["remaining_min"], remaining)
            elif state.opted_out:
                info["opted_out"] += 1
            else:
                info["exhausted"] += 1

        # Fix inf remaining for jurisdictions with no active clients
        for info in jurisdictions.values():
            if info["remaining_min"] == float("inf"):
                info["remaining_min"] = 0.0

        return jurisdictions

    def get_client_status(self, client_id: int) -> Dict[str, Any]:
        """Return status for a single client."""
        state = self.client_states[client_id]
        spent_eps, spent_delta = state.accountant.get_spent_budget()
        remaining_eps, remaining_delta = state.accountant.get_remaining_budget()

        return {
            "client_id": client_id,
            "jurisdiction": state.jurisdiction,
            "hospital_name": state.hospital_name,
            "epsilon_ceiling": state.epsilon_ceiling,
            "epsilon_spent": spent_eps,
            "epsilon_remaining": remaining_eps,
            "delta": state.delta,
            "active": state.active,
            "opted_out": state.opted_out,
            "rounds_participated": state.rounds_participated,
            "deactivation_round": state.deactivation_round,
            "deactivation_reason": state.deactivation_reason,
        }

    def get_dropout_timeline(self) -> List[Dict[str, Any]]:
        """Return dropout/optout events sorted by round."""
        return sorted(self._dropout_events, key=lambda e: e["round"])

    def export_report(self) -> Dict[str, Any]:
        """Export comprehensive jurisdiction privacy report for auto-save."""
        per_client = {}
        for cid, state in self.client_states.items():
            spent = state.accountant.get_spent_budget()[0]
            remaining = state.accountant.get_remaining_budget()[0]
            per_client[str(cid)] = {
                "hospital_name": state.hospital_name,
                "jurisdiction": state.jurisdiction,
                "epsilon_ceiling": state.epsilon_ceiling,
                "calibrated_noise_multiplier": round(state.calibrated_noise_multiplier, 4),
                "epsilon_spent": round(spent, 6),
                "epsilon_remaining": round(remaining, 6),
                "rounds_participated": state.rounds_participated,
                "active": state.active,
                "opted_out": state.opted_out,
                "deactivation_round": state.deactivation_round,
                "deactivation_reason": state.deactivation_reason,
                "epsilon_history": [round(e, 6) for e in state.epsilon_history],
            }

        # RDP budget details per client (if available)
        rdp_details = {}
        for cid, state in self.client_states.items():
            rdp_info = state.accountant.get_rdp_budget()
            if rdp_info:
                rdp_details[str(cid)] = {
                    "current_epsilon": round(rdp_info["current_epsilon"], 6),
                    "optimal_order": rdp_info["optimal_order"],
                    "remaining_epsilon": round(rdp_info["remaining_epsilon"], 6),
                    "num_compositions": rdp_info["num_compositions"],
                }

        return {
            "config": {
                "global_epsilon": self.global_epsilon,
                "num_rounds": self.num_rounds,
                "hospital_allocation_fraction": self.hospital_allocation_fraction,
                "noise_strategy": self.noise_strategy,
                "min_active_clients": self.min_active_clients,
            },
            "jurisdiction_budgets": {
                cc: {
                    "epsilon_max": b.get("epsilon_max"),
                    "delta": b.get("delta"),
                }
                for cc, b in self.jurisdiction_budgets.items()
            },
            "per_jurisdiction": self.get_jurisdiction_status(),
            "per_client": per_client,
            "rdp_details": rdp_details,
            "dropout_timeline": self.get_dropout_timeline(),
            "summary": {
                "total_clients": len(self.client_states),
                "active_clients": len(self.get_active_clients()),
                "exhausted_clients": sum(
                    1 for s in self.client_states.values()
                    if not s.active and not s.opted_out
                ),
                "opted_out_clients": sum(
                    1 for s in self.client_states.values() if s.opted_out
                ),
                "total_dropout_events": len(self._dropout_events),
            },
        }

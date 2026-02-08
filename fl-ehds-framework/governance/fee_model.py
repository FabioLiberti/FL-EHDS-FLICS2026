"""
EHDS Art. 42 Fee Model and Sustainability.

Simulates HDAB cost-recovery fee structures for cross-border
federated learning and provides budget constraint optimization.

Fee components per hospital per round:
    - Base access fee (fixed per session, charged once)
    - Data volume fee (per patient record per round)
    - Computation fee (per FL round)
    - Transfer fee (per MB of model update)

Budget optimization: given a max budget, find the optimal
configuration (fewer hospitals, rounds, or samples) that
stays within the budget while maximizing utility.

Author: Fabio Liberti
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# =====================================================================
# DATA STRUCTURES
# =====================================================================

@dataclass
class RoundFeeRecord:
    """Fee record for a single training round."""
    round_num: int
    per_hospital: Dict[int, Dict[str, float]] = field(default_factory=dict)
    round_total_eur: float = 0.0


@dataclass
class BudgetOptimizationResult:
    """Result of budget constraint optimization."""
    feasible: bool = False
    original_cost_eur: float = 0.0
    optimized_cost_eur: float = 0.0
    strategy: str = "none"
    hospitals_removed: List[str] = field(default_factory=list)
    rounds_reduced_to: Optional[int] = None
    sampling_factor: float = 1.0
    explanation: str = ""


# =====================================================================
# FEE MODEL BRIDGE
# =====================================================================

class FeeModelBridge:
    """EHDS Art. 42 Fee Model bridge for cross-border FL.

    Tracks per-hospital, per-round fees based on country fee profiles
    and provides budget constraint optimization.
    """

    def __init__(
        self,
        hospitals: List,
        config: Dict[str, Any],
        num_rounds: int,
        model_size_mb: float = 2.0,
    ):
        self._hospitals = hospitals
        self._config = config
        self._num_rounds = num_rounds
        self._model_size_mb = model_size_mb

        # Per-hospital cumulative fees
        self._hospital_fees: Dict[int, Dict[str, float]] = {}
        for h in hospitals:
            self._hospital_fees[h.hospital_id] = {
                "base_access": 0.0,
                "data_volume": 0.0,
                "computation": 0.0,
                "transfer": 0.0,
                "total": 0.0,
            }

        # Round history
        self._round_history: List[RoundFeeRecord] = []

        # Base access fee charged once per hospital at session start
        self._base_fees_charged = False

        # Optimization result (if run)
        self.optimization_result: Optional[BudgetOptimizationResult] = None

    # -----------------------------------------------------------------
    # COST ESTIMATION
    # -----------------------------------------------------------------

    def estimate_total_cost(
        self,
        hospitals: Optional[List] = None,
        num_rounds: Optional[int] = None,
        sampling_factor: float = 1.0,
    ) -> float:
        """Estimate total cost for a given configuration.

        Args:
            hospitals: List of HospitalNode (default: self._hospitals)
            num_rounds: Number of rounds (default: self._num_rounds)
            sampling_factor: Fraction of samples to use (0.0-1.0)

        Returns:
            Estimated total cost in EUR
        """
        if hospitals is None:
            hospitals = self._hospitals
        if num_rounds is None:
            num_rounds = self._num_rounds

        total = 0.0
        for h in hospitals:
            p = h.country_profile
            # Base access (once per session)
            total += p.fee_base_eur
            # Data volume (records * rounds * sampling)
            records = h.num_samples_after_optout or h.num_samples or 200
            total += p.fee_per_record_eur * records * num_rounds * sampling_factor
            # Computation (per round)
            total += p.fee_per_round_eur * num_rounds
            # Transfer (per MB per round, upload + download)
            total += p.fee_per_mb_eur * self._model_size_mb * num_rounds * 2

        return total

    def estimate_cost_breakdown(
        self,
        hospitals: Optional[List] = None,
        num_rounds: Optional[int] = None,
        sampling_factor: float = 1.0,
    ) -> Dict[str, float]:
        """Estimate cost breakdown by component."""
        if hospitals is None:
            hospitals = self._hospitals
        if num_rounds is None:
            num_rounds = self._num_rounds

        base = 0.0
        data = 0.0
        compute = 0.0
        transfer = 0.0

        for h in hospitals:
            p = h.country_profile
            records = h.num_samples_after_optout or h.num_samples or 200
            base += p.fee_base_eur
            data += p.fee_per_record_eur * records * num_rounds * sampling_factor
            compute += p.fee_per_round_eur * num_rounds
            transfer += p.fee_per_mb_eur * self._model_size_mb * num_rounds * 2

        return {
            "base_access": round(base, 2),
            "data_volume": round(data, 2),
            "computation": round(compute, 2),
            "transfer": round(transfer, 2),
            "total": round(base + data + compute + transfer, 2),
        }

    # -----------------------------------------------------------------
    # BUDGET OPTIMIZATION (Academic contribution)
    # -----------------------------------------------------------------

    def optimize_for_budget(
        self, max_budget_eur: float
    ) -> BudgetOptimizationResult:
        """Optimize FL configuration to fit within budget.

        Greedy multi-strategy optimization:
        1. Remove most expensive hospitals (by cost per sample)
        2. Reduce number of rounds
        3. Reduce sampling factor

        Args:
            max_budget_eur: Maximum budget in EUR

        Returns:
            BudgetOptimizationResult with optimized configuration
        """
        min_hospitals = self._config.get("min_hospitals", 2)
        min_rounds = self._config.get("min_rounds", 3)
        min_sampling = self._config.get("min_sampling_factor", 0.25)

        original_cost = self.estimate_total_cost()

        if original_cost <= max_budget_eur:
            return BudgetOptimizationResult(
                feasible=True,
                original_cost_eur=round(original_cost, 2),
                optimized_cost_eur=round(original_cost, 2),
                strategy="none",
                explanation="Current configuration is within budget",
            )

        # Sort hospitals by cost efficiency (most expensive per sample first)
        ranked = self._rank_hospitals_by_cost()
        current_hospitals = list(self._hospitals)
        removed = []
        current_rounds = self._num_rounds
        current_sampling = 1.0

        # Strategy 1: Remove most expensive hospitals
        while len(current_hospitals) > min_hospitals:
            # Find most expensive hospital still in list
            worst = None
            worst_cost = -1
            for h in current_hospitals:
                if h.hospital_id in ranked:
                    cost = ranked[h.hospital_id]
                    if cost > worst_cost:
                        worst_cost = cost
                        worst = h
            if worst is None:
                break

            current_hospitals.remove(worst)
            removed.append(f"{worst.name} ({worst.country_code})")

            cost = self.estimate_total_cost(
                hospitals=current_hospitals,
                num_rounds=current_rounds,
                sampling_factor=current_sampling,
            )
            if cost <= max_budget_eur:
                return BudgetOptimizationResult(
                    feasible=True,
                    original_cost_eur=round(original_cost, 2),
                    optimized_cost_eur=round(cost, 2),
                    strategy="hospital_reduction",
                    hospitals_removed=removed,
                    explanation=(
                        f"Removed {len(removed)} expensive hospital(s) "
                        f"to fit budget ({original_cost:.0f} -> {cost:.0f} EUR)"
                    ),
                )

        # Strategy 2: Reduce rounds
        while current_rounds > min_rounds:
            current_rounds -= 1
            cost = self.estimate_total_cost(
                hospitals=current_hospitals,
                num_rounds=current_rounds,
                sampling_factor=current_sampling,
            )
            if cost <= max_budget_eur:
                return BudgetOptimizationResult(
                    feasible=True,
                    original_cost_eur=round(original_cost, 2),
                    optimized_cost_eur=round(cost, 2),
                    strategy="combined" if removed else "round_reduction",
                    hospitals_removed=removed,
                    rounds_reduced_to=current_rounds,
                    explanation=(
                        f"Reduced to {current_rounds} rounds"
                        + (f" + removed {len(removed)} hospital(s)" if removed else "")
                        + f" ({original_cost:.0f} -> {cost:.0f} EUR)"
                    ),
                )

        # Strategy 3: Reduce sampling
        for factor in [0.75, 0.5, 0.25]:
            if factor < min_sampling:
                continue
            current_sampling = factor
            cost = self.estimate_total_cost(
                hospitals=current_hospitals,
                num_rounds=current_rounds,
                sampling_factor=current_sampling,
            )
            if cost <= max_budget_eur:
                return BudgetOptimizationResult(
                    feasible=True,
                    original_cost_eur=round(original_cost, 2),
                    optimized_cost_eur=round(cost, 2),
                    strategy="combined",
                    hospitals_removed=removed,
                    rounds_reduced_to=current_rounds if current_rounds < self._num_rounds else None,
                    sampling_factor=current_sampling,
                    explanation=(
                        f"Sampling at {current_sampling:.0%}"
                        + (f", {current_rounds} rounds" if current_rounds < self._num_rounds else "")
                        + (f", removed {len(removed)} hospital(s)" if removed else "")
                        + f" ({original_cost:.0f} -> {cost:.0f} EUR)"
                    ),
                )

        # Infeasible
        final_cost = self.estimate_total_cost(
            hospitals=current_hospitals,
            num_rounds=current_rounds,
            sampling_factor=min_sampling,
        )
        return BudgetOptimizationResult(
            feasible=False,
            original_cost_eur=round(original_cost, 2),
            optimized_cost_eur=round(final_cost, 2),
            strategy="infeasible",
            hospitals_removed=removed,
            rounds_reduced_to=current_rounds,
            sampling_factor=min_sampling,
            explanation=(
                f"Cannot fit within {max_budget_eur:.0f} EUR budget. "
                f"Minimum achievable: {final_cost:.0f} EUR "
                f"({len(current_hospitals)} hospitals, {current_rounds} rounds, "
                f"{min_sampling:.0%} sampling)"
            ),
        )

    def _rank_hospitals_by_cost(self) -> Dict[int, float]:
        """Rank hospitals by cost per sample (higher = more expensive)."""
        ranked = {}
        for h in self._hospitals:
            p = h.country_profile
            records = h.num_samples_after_optout or h.num_samples or 200
            if records == 0:
                records = 1
            # Total cost for this hospital across all rounds
            total = (
                p.fee_base_eur
                + p.fee_per_record_eur * records * self._num_rounds
                + p.fee_per_round_eur * self._num_rounds
                + p.fee_per_mb_eur * self._model_size_mb * self._num_rounds * 2
            )
            ranked[h.hospital_id] = total / records
        return ranked

    # -----------------------------------------------------------------
    # PER-ROUND RECORDING
    # -----------------------------------------------------------------

    def record_round(self, round_num: int, active_hospitals: List):
        """Record fees for a completed training round.

        Args:
            round_num: Current round number (0-indexed)
            active_hospitals: List of hospitals that participated
        """
        record = RoundFeeRecord(round_num=round_num)

        for h in active_hospitals:
            p = h.country_profile
            records = h.num_samples_after_optout or h.num_samples or 200

            fees = {}
            # Base access: charged on first round only
            if round_num == 0 and not self._base_fees_charged:
                fees["base_access"] = p.fee_base_eur
            else:
                fees["base_access"] = 0.0

            fees["data_volume"] = p.fee_per_record_eur * records
            fees["computation"] = p.fee_per_round_eur
            fees["transfer"] = p.fee_per_mb_eur * self._model_size_mb * 2
            fees["total"] = sum(fees.values())

            record.per_hospital[h.hospital_id] = fees
            record.round_total_eur += fees["total"]

            # Update cumulative
            if h.hospital_id in self._hospital_fees:
                for key in ["base_access", "data_volume", "computation", "transfer", "total"]:
                    self._hospital_fees[h.hospital_id][key] += fees[key]

        if round_num == 0:
            self._base_fees_charged = True

        self._round_history.append(record)

    # -----------------------------------------------------------------
    # REPORTS
    # -----------------------------------------------------------------

    def export_report(self) -> Dict[str, Any]:
        """Export comprehensive fee model report."""
        # Aggregate totals
        total_cost = sum(
            hf["total"] for hf in self._hospital_fees.values()
        )
        total_base = sum(hf["base_access"] for hf in self._hospital_fees.values())
        total_data = sum(hf["data_volume"] for hf in self._hospital_fees.values())
        total_compute = sum(hf["computation"] for hf in self._hospital_fees.values())
        total_transfer = sum(hf["transfer"] for hf in self._hospital_fees.values())

        # Per-country aggregation
        country_fees: Dict[str, Dict[str, float]] = {}
        for h in self._hospitals:
            cc = h.country_code
            hf = self._hospital_fees.get(h.hospital_id, {})
            if cc not in country_fees:
                country_fees[cc] = {
                    "base_access": 0.0, "data_volume": 0.0,
                    "computation": 0.0, "transfer": 0.0, "total": 0.0,
                    "hospitals": 0,
                }
            for key in ["base_access", "data_volume", "computation", "transfer", "total"]:
                country_fees[cc][key] += hf.get(key, 0.0)
            country_fees[cc]["hospitals"] += 1

        # Round for display
        for cc in country_fees:
            for key in ["base_access", "data_volume", "computation", "transfer", "total"]:
                country_fees[cc][key] = round(country_fees[cc][key], 2)

        # Per-hospital detail
        per_hospital = {}
        for h in self._hospitals:
            hf = self._hospital_fees.get(h.hospital_id, {})
            per_hospital[h.hospital_id] = {
                "name": h.name,
                "country": h.country_code,
                "base_access": round(hf.get("base_access", 0.0), 2),
                "data_volume": round(hf.get("data_volume", 0.0), 2),
                "computation": round(hf.get("computation", 0.0), 2),
                "transfer": round(hf.get("transfer", 0.0), 2),
                "total": round(hf.get("total", 0.0), 2),
            }

        # Round history (simplified)
        round_totals = [
            {"round": r.round_num, "total_eur": round(r.round_total_eur, 2)}
            for r in self._round_history
        ]

        report = {
            "report_type": "EHDS Art. 42 Fee Model",
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "total_cost_eur": round(total_cost, 2),
            "cost_breakdown": {
                "base_access": round(total_base, 2),
                "data_volume": round(total_data, 2),
                "computation": round(total_compute, 2),
                "transfer": round(total_transfer, 2),
            },
            "num_hospitals": len(self._hospitals),
            "num_rounds_recorded": len(self._round_history),
            "model_size_mb": self._model_size_mb,
            "fees_by_country": country_fees,
            "fees_by_hospital": per_hospital,
            "round_history": round_totals,
        }

        if self.optimization_result:
            opt = self.optimization_result
            report["budget_optimization"] = {
                "feasible": opt.feasible,
                "original_cost_eur": opt.original_cost_eur,
                "optimized_cost_eur": opt.optimized_cost_eur,
                "strategy": opt.strategy,
                "hospitals_removed": opt.hospitals_removed,
                "rounds_reduced_to": opt.rounds_reduced_to,
                "sampling_factor": opt.sampling_factor,
                "explanation": opt.explanation,
            }

        return report

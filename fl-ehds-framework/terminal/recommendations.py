"""
Algorithm recommendations based on healthcare use cases.
Provides guided selection and automatic comparison setup.
"""

from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class UseCase:
    """Healthcare use case definition."""
    id: str
    name: str
    description: str
    data_characteristics: str
    recommended_algorithms: List[str]
    recommended_params: Dict[str, any]
    rationale: str


# Healthcare use cases with algorithm recommendations
HEALTHCARE_USE_CASES = [
    UseCase(
        id="multi_hospital_iid",
        name="Multi-Hospital Collaboration (Similar Populations)",
        description="Multiple hospitals with similar patient demographics collaborating on a common model.",
        data_characteristics="IID or near-IID data distribution",
        recommended_algorithms=["FedAvg", "FedAdam", "FedYogi"],
        recommended_params={
            "is_iid": True,
            "alpha": 10.0,
            "num_rounds": 30,
            "local_epochs": 3,
        },
        rationale="With similar data distributions, simpler algorithms like FedAvg work well. "
                  "FedAdam/FedYogi can accelerate convergence."
    ),
    UseCase(
        id="rare_disease",
        name="Rare Disease Research (Heterogeneous Data)",
        description="Studying rare diseases where each center has different patient subpopulations.",
        data_characteristics="Highly non-IID, label skew",
        recommended_algorithms=["SCAFFOLD", "FedProx", "FedNova"],
        recommended_params={
            "is_iid": False,
            "alpha": 0.3,
            "num_rounds": 50,
            "local_epochs": 5,
            "mu": 0.1,
        },
        rationale="Non-IID data requires algorithms that handle client drift. "
                  "SCAFFOLD corrects drift with control variates, FedProx adds regularization."
    ),
    UseCase(
        id="personalized_medicine",
        name="Personalized Medicine (Patient-Specific Models)",
        description="Creating personalized risk models while benefiting from collaborative learning.",
        data_characteristics="Each hospital needs a tailored model",
        recommended_algorithms=["Ditto", "Per-FedAvg", "FedProx"],
        recommended_params={
            "is_iid": False,
            "alpha": 0.5,
            "num_rounds": 40,
            "local_epochs": 3,
            "mu": 0.5,
        },
        rationale="Personalization algorithms (Ditto, Per-FedAvg) maintain client-specific models "
                  "while learning from global knowledge."
    ),
    UseCase(
        id="resource_constrained",
        name="Resource-Constrained Devices (IoT/Edge)",
        description="Training on resource-limited devices with variable computation capabilities.",
        data_characteristics="Heterogeneous computation, variable local epochs",
        recommended_algorithms=["FedNova", "FedAvg", "FedProx"],
        recommended_params={
            "is_iid": False,
            "alpha": 1.0,
            "num_rounds": 30,
            "local_epochs": 2,
        },
        rationale="FedNova handles heterogeneous local computation by normalizing updates. "
                  "Simple algorithms reduce device overhead."
    ),
    UseCase(
        id="privacy_critical",
        name="Privacy-Critical (Strong DP Guarantees)",
        description="Scenarios requiring strong differential privacy (e.g., sensitive mental health data).",
        data_characteristics="Strong privacy requirements, epsilon < 5",
        recommended_algorithms=["FedAvg", "FedProx"],
        recommended_params={
            "is_iid": False,
            "alpha": 0.5,
            "num_rounds": 30,
            "local_epochs": 1,
            "dp_enabled": True,
            "dp_epsilon": 1.0,
            "dp_clip_norm": 1.0,
        },
        rationale="Simple aggregation works best with DP. Fewer local epochs reduce privacy budget consumption."
    ),
    UseCase(
        id="cross_border_ehds",
        name="Cross-Border EHDS Compliance",
        description="Pan-European collaboration under EHDS regulations with diverse healthcare systems.",
        data_characteristics="High heterogeneity, regulatory compliance needed",
        recommended_algorithms=["SCAFFOLD", "FedProx", "Ditto"],
        recommended_params={
            "is_iid": False,
            "alpha": 0.5,
            "num_rounds": 50,
            "local_epochs": 3,
            "mu": 0.1,
            "dp_enabled": True,
            "dp_epsilon": 10.0,
        },
        rationale="Cross-border data is naturally heterogeneous. SCAFFOLD handles this well. "
                  "DP ensures GDPR compliance."
    ),
    UseCase(
        id="fast_convergence",
        name="Fast Convergence Needed (Emergency Response)",
        description="Time-critical scenarios like pandemic response requiring quick model deployment.",
        data_characteristics="Need rapid convergence, moderate heterogeneity",
        recommended_algorithms=["FedAdam", "FedYogi", "SCAFFOLD"],
        recommended_params={
            "is_iid": False,
            "alpha": 1.0,
            "num_rounds": 20,
            "local_epochs": 5,
            "server_lr": 0.1,
            "beta1": 0.9,
            "beta2": 0.99,
        },
        rationale="Adaptive server optimizers (FedAdam, FedYogi) accelerate convergence. "
                  "More local epochs per round reduce communication rounds."
    ),
]


def get_use_cases() -> List[UseCase]:
    """Return all healthcare use cases."""
    return HEALTHCARE_USE_CASES


def get_use_case_by_id(use_case_id: str) -> UseCase:
    """Get a specific use case by ID."""
    for uc in HEALTHCARE_USE_CASES:
        if uc.id == use_case_id:
            return uc
    return None


def recommend_algorithms(
    is_iid: bool = False,
    privacy_critical: bool = False,
    personalization_needed: bool = False,
    resource_constrained: bool = False,
    fast_convergence: bool = False,
) -> Tuple[List[str], str]:
    """
    Recommend algorithms based on scenario characteristics.

    Returns: (recommended_algorithms, rationale)
    """
    recommendations = []
    rationale_parts = []

    if is_iid:
        recommendations.extend(["FedAvg", "FedAdam"])
        rationale_parts.append("IID data: simple algorithms work well")
    else:
        recommendations.extend(["FedProx", "SCAFFOLD"])
        rationale_parts.append("Non-IID data: need drift correction")

    if privacy_critical:
        # Filter to DP-compatible algorithms
        recommendations = [a for a in recommendations if a in ["FedAvg", "FedProx"]]
        if not recommendations:
            recommendations = ["FedAvg", "FedProx"]
        rationale_parts.append("Privacy: simpler aggregation for DP")

    if personalization_needed:
        recommendations.extend(["Ditto", "Per-FedAvg"])
        rationale_parts.append("Personalization: client-specific models")

    if resource_constrained:
        recommendations.insert(0, "FedNova")
        rationale_parts.append("Resource constraints: FedNova handles heterogeneous computation")

    if fast_convergence:
        recommendations.extend(["FedAdam", "FedYogi"])
        rationale_parts.append("Fast convergence: adaptive server optimizers")

    # Remove duplicates while preserving order
    seen = set()
    unique_recommendations = []
    for algo in recommendations:
        if algo not in seen:
            seen.add(algo)
            unique_recommendations.append(algo)

    return unique_recommendations[:4], "; ".join(rationale_parts)


def get_comparison_config(use_case: UseCase) -> Dict:
    """
    Generate a comparison configuration for a use case.

    Returns config dict for running algorithm comparison.
    """
    config = {
        "algorithms": use_case.recommended_algorithms,
        "num_clients": 5,
        "num_rounds": use_case.recommended_params.get("num_rounds", 30),
        "local_epochs": use_case.recommended_params.get("local_epochs", 3),
        "batch_size": 32,
        "learning_rate": 0.01,
        "num_seeds": 3,
        "is_iid": use_case.recommended_params.get("is_iid", False),
        "alpha": use_case.recommended_params.get("alpha", 0.5),
        "dp_enabled": use_case.recommended_params.get("dp_enabled", False),
        "dp_epsilon": use_case.recommended_params.get("dp_epsilon", 10.0),
        "mu": use_case.recommended_params.get("mu", 0.1),
        "server_lr": use_case.recommended_params.get("server_lr", 0.1),
        "beta1": use_case.recommended_params.get("beta1", 0.9),
        "beta2": use_case.recommended_params.get("beta2", 0.99),
        "tau": use_case.recommended_params.get("tau", 1e-3),
    }
    return config

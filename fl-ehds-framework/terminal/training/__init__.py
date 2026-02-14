"""
FL-EHDS Training Package
=========================
Re-exports all public symbols from split modules for backward compatibility.
"""

from terminal.training.data_generation import (
    _detect_device,
    generate_healthcare_data,
)
from terminal.training.models import (
    HealthcareMLP,
    HealthcareCNN,
    HealthcareResNet,
    load_image_dataset,
)
from terminal.training.federated import (
    ClientResult,
    RoundResult,
    FederatedTrainer,
)
from terminal.training.federated_image import (
    ImageFederatedTrainer,
)
from terminal.training.centralized import (
    CentralizedResult,
    CentralizedTrainer,
    CentralizedImageTrainer,
    run_fl_vs_centralized_comparison,
    generate_comparison_latex_table,
)
from terminal.training.byzantine_bridge import (
    client_results_to_gradients,
    aggregation_result_to_tensors,
)

__all__ = [
    # Data generation
    "_detect_device",
    "generate_healthcare_data",
    # Models
    "HealthcareMLP",
    "HealthcareCNN",
    "HealthcareResNet",
    "load_image_dataset",
    # Federated
    "ClientResult",
    "RoundResult",
    "FederatedTrainer",
    # Image Federated
    "ImageFederatedTrainer",
    # Centralized
    "CentralizedResult",
    "CentralizedTrainer",
    "CentralizedImageTrainer",
    "run_fl_vs_centralized_comparison",
    "generate_comparison_latex_table",
    # Byzantine bridge
    "client_results_to_gradients",
    "aggregation_result_to_tensors",
]

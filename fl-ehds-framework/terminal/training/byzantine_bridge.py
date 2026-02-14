"""
Bridge utilities between PyTorch trainer tensors and Byzantine defense numpy arrays.

Converts ClientResult model updates (Dict[str, Tensor]) to ClientGradient
(Dict[str, ndarray]) expected by core.byzantine_resilience, and back.
"""

from typing import Dict, List, Optional

import numpy as np
import torch

from core.byzantine_resilience import (
    ByzantineConfig,
    ByzantineDefenseManager,
    ClientGradient,
    AggregationResult,
)
from terminal.training.federated import ClientResult


def client_results_to_gradients(
    client_results: List[ClientResult],
) -> List[ClientGradient]:
    """Convert PyTorch ClientResult list to numpy ClientGradient list.

    Each ClientResult.model_update contains {param_name: Tensor} representing
    the pseudo-gradient (delta) from local training. This converts each tensor
    to a numpy array for the Byzantine aggregator.
    """
    gradients = []
    for cr in client_results:
        gradient_dict: Dict[str, np.ndarray] = {}
        for name, tensor in cr.model_update.items():
            gradient_dict[name] = tensor.detach().cpu().numpy()

        gradients.append(ClientGradient(
            client_id=cr.client_id,
            gradient=gradient_dict,
            samples_used=cr.num_samples,
        ))
    return gradients


def aggregation_result_to_tensors(
    result: AggregationResult,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Convert aggregated numpy gradient back to PyTorch tensors.

    Returns dict {param_name: Tensor} on the specified device,
    ready to be applied to the global model.
    """
    tensors: Dict[str, torch.Tensor] = {}
    for name, arr in result.aggregated_gradient.items():
        tensors[name] = torch.from_numpy(arr).to(device)
    return tensors

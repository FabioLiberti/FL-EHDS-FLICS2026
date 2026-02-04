"""
Aggregation Module
==================
Federated Learning aggregation algorithms.
"""

from .base import BaseAggregator
from .fedavg import FedAvgAggregator
from .fedprox import FedProxAggregator

__all__ = [
    "BaseAggregator",
    "FedAvgAggregator",
    "FedProxAggregator",
]

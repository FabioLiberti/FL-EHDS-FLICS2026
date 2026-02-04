"""
FL-EHDS Data Holders Layer (Layer 3)
====================================
Data holder components including adaptive training engine,
FHIR preprocessing, and secure communication.
"""

from .training_engine import TrainingEngine, AdaptiveTrainer
from .fhir_preprocessing import FHIRPreprocessor, FHIRValidator
from .secure_communication import SecureCommunicator, GradientTransport

__all__ = [
    # Training Engine
    "TrainingEngine",
    "AdaptiveTrainer",
    # FHIR Preprocessing
    "FHIRPreprocessor",
    "FHIRValidator",
    # Secure Communication
    "SecureCommunicator",
    "GradientTransport",
]

"""
Screen modules for FL-EHDS terminal interface.
Each screen provides a specific functionality accessible from the main menu.
"""

from terminal.screens.training import TrainingScreen
from terminal.screens.algorithms import AlgorithmsScreen
from terminal.screens.privacy import PrivacyScreen
from terminal.screens.byzantine import ByzantineScreen
from terminal.screens.benchmark import BenchmarkScreen
from terminal.screens.output import OutputScreen

__all__ = [
    "TrainingScreen",
    "AlgorithmsScreen",
    "PrivacyScreen",
    "ByzantineScreen",
    "BenchmarkScreen",
    "OutputScreen",
]

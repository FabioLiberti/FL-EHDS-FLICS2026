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
from terminal.screens.paper_experiments import PaperExperimentsScreen
from terminal.screens.vertical_fl import VerticalFLScreen
from terminal.screens.continual_learning import ContinualLearningScreen
from terminal.screens.multi_task import MultiTaskScreen
from terminal.screens.hierarchical import HierarchicalScreen

__all__ = [
    "TrainingScreen",
    "AlgorithmsScreen",
    "PrivacyScreen",
    "ByzantineScreen",
    "BenchmarkScreen",
    "OutputScreen",
    "PaperExperimentsScreen",
    "VerticalFLScreen",
    "ContinualLearningScreen",
    "MultiTaskScreen",
    "HierarchicalScreen",
]

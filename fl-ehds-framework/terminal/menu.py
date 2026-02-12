"""
Main menu navigation for FL-EHDS terminal interface.
Provides arrow-key navigation with Enter to select.
"""

import sys
from typing import List, Tuple, Callable, Optional

try:
    import questionary
    from questionary import Style as QStyle
    HAS_QUESTIONARY = True
except ImportError:
    HAS_QUESTIONARY = False

from terminal.colors import (
    Colors, Style, print_header, print_section,
    print_error, print_info, clear_screen
)


# Custom questionary style (no emojis, minimal colors)
MENU_STYLE = QStyle([
    ('qmark', 'fg:cyan bold'),
    ('question', 'bold'),
    ('answer', 'fg:cyan bold'),
    ('pointer', 'fg:cyan bold'),
    ('highlighted', 'fg:cyan bold'),
    ('selected', 'fg:green'),
    ('separator', 'fg:white'),
    ('instruction', 'fg:white'),
    ('text', ''),
]) if HAS_QUESTIONARY else None


class MenuItem:
    """Represents a menu item."""

    def __init__(
        self,
        key: str,
        label: str,
        handler: Optional[Callable] = None,
        enabled: bool = True,
        description: str = ""
    ):
        self.key = key
        self.label = label
        self.handler = handler
        self.enabled = enabled
        self.description = description

    def __str__(self) -> str:
        return f"{self.key}. {self.label}"


class Menu:
    """Base menu class with arrow navigation support."""

    def __init__(self, title: str, items: List[MenuItem]):
        self.title = title
        self.items = items
        self.selected_index = 0

    def display(self) -> Optional[MenuItem]:
        """Display menu and return selected item."""
        if HAS_QUESTIONARY:
            return self._display_questionary()
        else:
            return self._display_fallback()

    def _display_questionary(self) -> Optional[MenuItem]:
        """Display menu using questionary (arrow navigation)."""
        choices = []
        for item in self.items:
            if item.enabled:
                choices.append(questionary.Choice(
                    title=f"{item.key}. {item.label}",
                    value=item
                ))
            else:
                choices.append(questionary.Choice(
                    title=f"{item.key}. {item.label} (non disponibile)",
                    value=item,
                    disabled="Non disponibile"
                ))

        result = questionary.select(
            f"\n{self.title}",
            choices=choices,
            style=MENU_STYLE,
            instruction="(Frecce: naviga | Enter: seleziona | Ctrl+C: esci)"
        ).ask()

        return result

    def _display_fallback(self) -> Optional[MenuItem]:
        """Display menu using simple number input (fallback)."""
        print(f"\n{Style.TITLE}{self.title}{Colors.RESET}")
        print("-" * 60)

        for item in self.items:
            if item.enabled:
                print(f"  {item.key}. {item.label}")
            else:
                print(f"  {Style.MUTED}{item.key}. {item.label} (non disponibile){Colors.RESET}")

        print("-" * 60)
        print(f"{Style.MUTED}(Digita il numero e premi Enter, 'q' per uscire){Colors.RESET}")

        while True:
            try:
                choice = input(f"\n{Style.INFO}Scelta: {Colors.RESET}").strip().lower()

                if choice == 'q':
                    return None

                # Find item by key
                for item in self.items:
                    if item.key == choice:
                        if item.enabled:
                            return item
                        else:
                            print_error("Opzione non disponibile")
                            break
                else:
                    print_error("Opzione non valida")

            except (EOFError, KeyboardInterrupt):
                return None

    def run(self) -> bool:
        """Run menu and execute selected handler. Returns False to exit."""
        item = self.display()

        if item is None:
            return False

        if item.handler:
            try:
                result = item.handler()
                if result == "exit":
                    return False
            except Exception as e:
                print_error(f"Errore durante l'esecuzione: {str(e)}")
                input(f"\n{Style.MUTED}Premi Enter per continuare...{Colors.RESET}")

        return True


class MainMenu:
    """Main menu for FL-EHDS terminal interface."""

    def __init__(self):
        self.running = True

    def run(self):
        """Run the main menu loop."""
        while self.running:
            clear_screen()
            print_header()

            menu = Menu("MENU PRINCIPALE", [
                MenuItem("1", "Training Federato", self._training_menu),
                MenuItem("2", "Confronto Algoritmi FL", self._algorithms_menu),
                MenuItem("3", "Confronto Guidato per Caso d'Uso", self._guided_comparison_menu),
                MenuItem("4", "Gestione Dataset", self._datasets_menu),
                MenuItem("5", "Analisi Privacy (RDP)", self._privacy_menu),
                MenuItem("6", "Vertical Federated Learning", self._vertical_menu),
                MenuItem("7", "Byzantine Resilience", self._byzantine_menu),
                MenuItem("8", "Continual Learning", self._continual_menu),
                MenuItem("9", "Multi-Task FL", self._multitask_menu),
                MenuItem("10", "Hierarchical FL", self._hierarchical_menu),
                MenuItem("11", "Cross-Border Federation (EHDS)", self._compliance_menu),
                MenuItem("12", "Benchmark Suite", self._benchmark_menu),
                MenuItem("13", "Configurazione Globale", self._config_menu),
                MenuItem("14", "Esporta Risultati", self._export_menu),
                MenuItem("15", "Esperimenti Paper FLICS 2026", self._paper_experiments_menu),
                MenuItem("0", "Esci", self._exit),
            ])

            if not menu.run():
                break

    def _training_menu(self):
        """Open training submenu."""
        from terminal.screens.training import TrainingScreen
        screen = TrainingScreen()
        screen.run()

    def _algorithms_menu(self):
        """Open algorithm comparison submenu."""
        from terminal.screens.algorithms import AlgorithmsScreen
        screen = AlgorithmsScreen()
        screen.run()

    def _guided_comparison_menu(self):
        """Open guided comparison submenu."""
        from terminal.screens.guided_comparison import GuidedComparisonScreen
        screen = GuidedComparisonScreen()
        screen.run()

    def _datasets_menu(self):
        """Open dataset management submenu."""
        from terminal.screens.datasets import DatasetScreen
        screen = DatasetScreen()
        screen.run()

    def _privacy_menu(self):
        """Open privacy analysis submenu."""
        from terminal.screens.privacy import PrivacyScreen
        screen = PrivacyScreen()
        screen.run()

    def _vertical_menu(self):
        """Open vertical FL submenu."""
        from terminal.screens.vertical_fl import VerticalFLScreen
        VerticalFLScreen().run()

    def _byzantine_menu(self):
        """Open byzantine resilience submenu."""
        from terminal.screens.byzantine import ByzantineScreen
        screen = ByzantineScreen()
        screen.run()

    def _continual_menu(self):
        """Open continual learning submenu."""
        from terminal.screens.continual_learning import ContinualLearningScreen
        ContinualLearningScreen().run()

    def _multitask_menu(self):
        """Open multi-task FL submenu."""
        from terminal.screens.multi_task import MultiTaskScreen
        MultiTaskScreen().run()

    def _hierarchical_menu(self):
        """Open hierarchical FL submenu."""
        from terminal.screens.hierarchical import HierarchicalScreen
        HierarchicalScreen().run()

    def _compliance_menu(self):
        """Open EHDS cross-border compliance submenu."""
        from terminal.screens.cross_border import CrossBorderScreen
        screen = CrossBorderScreen()
        screen.run()

    def _benchmark_menu(self):
        """Open benchmark suite submenu."""
        from terminal.screens.benchmark import BenchmarkScreen
        screen = BenchmarkScreen()
        screen.run()

    def _config_menu(self):
        """Open configuration submenu."""
        print_info("Configurazione - In sviluppo")
        input(f"\n{Style.MUTED}Premi Enter per continuare...{Colors.RESET}")

    def _export_menu(self):
        """Open export submenu."""
        from terminal.screens.output import OutputScreen
        screen = OutputScreen()
        screen.run()

    def _paper_experiments_menu(self):
        """Open paper experiments workflow submenu."""
        from terminal.screens.paper_experiments import PaperExperimentsScreen
        screen = PaperExperimentsScreen()
        screen.run()

    def _exit(self):
        """Exit the application."""
        self.running = False
        print(f"\n{Style.SUCCESS}Arrivederci!{Colors.RESET}\n")
        return "exit"

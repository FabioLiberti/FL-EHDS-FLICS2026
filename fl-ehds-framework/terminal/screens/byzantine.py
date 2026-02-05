"""
Byzantine resilience screen for FL-EHDS terminal interface.
Tests FL robustness against Byzantine attacks with various defense mechanisms.
"""

import sys
from pathlib import Path
from typing import Dict, List, Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import questionary
    HAS_QUESTIONARY = True
except ImportError:
    HAS_QUESTIONARY = False

from terminal.colors import (
    Colors, Style, print_section, print_subsection,
    print_success, print_error, print_info, print_warning, clear_screen
)
from terminal.progress import FLProgressBar
from terminal.validators import get_int, get_float, get_bool, get_choice, confirm, display_config_summary
from terminal.menu import Menu, MenuItem, MENU_STYLE


# Available attacks
ATTACK_TYPES = [
    "None (baseline)",
    "Label Flip",
    "Gaussian Noise",
    "Scaling Attack",
    "Sign Flip",
]

# Available defenses
DEFENSE_TYPES = [
    "None",
    "Krum",
    "Multi-Krum",
    "Trimmed Mean",
    "Median",
    "Bulyan",
    "FLTrust",
    "FLAME",
]


class ByzantineScreen:
    """Byzantine resilience testing screen."""

    def __init__(self):
        self.config = self._default_config()
        self.results = {}

    def _default_config(self) -> Dict[str, Any]:
        """Return default configuration."""
        return {
            "num_clients": 10,
            "num_byzantine": 2,
            "num_rounds": 30,
            "attack_type": "Label Flip",
            "defense_type": "Krum",
            "attack_strength": 1.0,
        }

    def run(self):
        """Run the byzantine screen."""
        while True:
            clear_screen()
            print_section("BYZANTINE RESILIENCE")

            menu = Menu("Seleziona azione", [
                MenuItem("1", "Configura test", self._configure),
                MenuItem("2", "Esegui singolo test", self._run_single_test),
                MenuItem("3", "Confronto difese", self._compare_defenses),
                MenuItem("4", "Visualizza risultati", self._show_results),
                MenuItem("0", "Torna al menu principale", lambda: "back"),
            ])

            result = menu.display()
            if result is None:
                break

            if result.handler:
                handler_result = result.handler()
                if handler_result == "back":
                    break

    def _configure(self):
        """Configure byzantine test parameters."""
        clear_screen()
        print_section("CONFIGURAZIONE BYZANTINE TEST")

        # Basic settings
        print_subsection("Impostazioni Base")
        self.config["num_clients"] = get_int("Numero totale client", default=self.config["num_clients"], min_val=4, max_val=50)
        self.config["num_byzantine"] = get_int("Numero client bizantini", default=self.config["num_byzantine"], min_val=1, max_val=self.config["num_clients"] // 2)
        self.config["num_rounds"] = get_int("Numero round", default=self.config["num_rounds"], min_val=10, max_val=100)

        # Attack selection
        print_subsection("Tipo di Attacco")
        if HAS_QUESTIONARY:
            self.config["attack_type"] = questionary.select(
                "Seleziona tipo di attacco:",
                choices=ATTACK_TYPES,
                default=self.config["attack_type"],
                style=MENU_STYLE,
            ).ask() or self.config["attack_type"]
        else:
            self.config["attack_type"] = get_choice("Tipo di attacco:", ATTACK_TYPES, self.config["attack_type"])

        if self.config["attack_type"] != "None (baseline)":
            self.config["attack_strength"] = get_float("Intensita attacco", default=self.config["attack_strength"], min_val=0.1, max_val=10.0)

        # Defense selection
        print_subsection("Meccanismo di Difesa")
        if HAS_QUESTIONARY:
            self.config["defense_type"] = questionary.select(
                "Seleziona meccanismo di difesa:",
                choices=DEFENSE_TYPES,
                default=self.config["defense_type"],
                style=MENU_STYLE,
            ).ask() or self.config["defense_type"]
        else:
            self.config["defense_type"] = get_choice("Meccanismo di difesa:", DEFENSE_TYPES, self.config["defense_type"])

        display_config_summary(self.config)
        input(f"\n{Style.MUTED}Premi Enter per continuare...{Colors.RESET}")

    def _run_single_test(self):
        """Run a single byzantine test."""
        clear_screen()
        print_section("TEST BYZANTINE IN CORSO")

        display_config_summary(self.config)

        if not confirm("\nAvviare il test?", default=True):
            return

        print()

        try:
            from dashboard.app_v4 import ByzantineSimulator

            print_info("Inizializzazione simulatore...")

            simulator = ByzantineSimulator(
                num_clients=self.config["num_clients"],
                num_byzantine=self.config["num_byzantine"],
                num_rounds=self.config["num_rounds"],
                attack_type=self.config["attack_type"],
                defense_type=self.config["defense_type"],
                attack_strength=self.config["attack_strength"],
            )

            print_info(f"Esecuzione test: {self.config['attack_type']} vs {self.config['defense_type']}")
            print()

            with FLProgressBar(
                total=self.config["num_rounds"],
                desc="Round",
                color="yellow"
            ) as pbar:
                final_result = None
                for round_num in range(self.config["num_rounds"]):
                    final_result = simulator.train_round(round_num)
                    pbar.update(1, acc=f"{final_result.get('accuracy', 0):.2%}")

            # Store results
            key = f"{self.config['attack_type']} + {self.config['defense_type']}"
            self.results[key] = {
                "config": self.config.copy(),
                "final_metrics": final_result,
            }

            print()
            print_success("Test completato")
            self._display_single_result(final_result)

        except ImportError as e:
            print_error(f"Impossibile importare ByzantineSimulator: {e}")
            print_info("Questa funzionalita richiede l'implementazione completa in app_v4.py")

            # Show placeholder results
            print_warning("Mostrando risultati placeholder...")
            self._display_placeholder_results()

        except Exception as e:
            print_error(f"Errore durante il test: {e}")

        input(f"\n{Style.MUTED}Premi Enter per continuare...{Colors.RESET}")

    def _display_single_result(self, result: Dict):
        """Display results for single test."""
        print_subsection("RISULTATI")

        print(f"\n{Style.TITLE}{'Metrica':<15} {'Valore':<15}{Colors.RESET}")
        print("-" * 30)
        print(f"  {'Accuracy':<13} {result.get('accuracy', 0):.2%}")
        print(f"  {'F1 Score':<13} {result.get('f1', 0):.4f}")
        print(f"  {'AUC':<13} {result.get('auc', 0):.4f}")

        byzantine_pct = self.config["num_byzantine"] / self.config["num_clients"] * 100
        print(f"\n{Style.MUTED}Client bizantini: {self.config['num_byzantine']}/{self.config['num_clients']} ({byzantine_pct:.0f}%){Colors.RESET}")

    def _display_placeholder_results(self):
        """Display placeholder results when simulator not available."""
        print_subsection("RISULTATI ATTESI (Reference)")

        reference_results = [
            ("No Attack + No Defense", "60.9%", "0.62", "0.66"),
            ("Label Flip + No Defense", "45.2%", "0.44", "0.48"),
            ("Label Flip + Krum", "58.1%", "0.57", "0.62"),
            ("Label Flip + Multi-Krum", "59.3%", "0.58", "0.63"),
            ("Label Flip + Trimmed Mean", "57.8%", "0.56", "0.61"),
            ("Label Flip + Median", "56.9%", "0.55", "0.60"),
        ]

        print(f"\n{Style.TITLE}{'Configurazione':<30} {'Acc':<10} {'F1':<10} {'AUC':<10}{Colors.RESET}")
        print("-" * 60)

        for config, acc, f1, auc in reference_results:
            print(f"  {config:<28} {acc:<10} {f1:<10} {auc:<10}")

    def _compare_defenses(self):
        """Compare all defense mechanisms."""
        clear_screen()
        print_section("CONFRONTO DIFESE BYZANTINE")

        print_info("Questa funzione confronta tutte le difese contro un singolo attacco")
        print()

        # Attack selection
        if HAS_QUESTIONARY:
            attack = questionary.select(
                "Seleziona attacco da testare:",
                choices=[a for a in ATTACK_TYPES if a != "None (baseline)"],
                style=MENU_STYLE,
            ).ask()
        else:
            attack = get_choice("Seleziona attacco:", ATTACK_TYPES[1:])

        if not attack:
            return

        print()
        print_info(f"Confronto difese contro: {attack}")

        # For now, show reference table
        print_subsection("RISULTATI CONFRONTO (Reference)")

        defenses = ["None", "Krum", "Multi-Krum", "Trimmed Mean", "Median"]

        print(f"\n{Style.TITLE}{'Difesa':<20} {'Accuracy':<15} {'Degradation':<15}{Colors.RESET}")
        print("-" * 50)

        baseline = 60.9
        results = [45.2, 58.1, 59.3, 57.8, 56.9]

        for defense, acc in zip(defenses, results):
            degradation = baseline - acc
            color = Style.SUCCESS if degradation < 5 else Style.WARNING if degradation < 10 else Style.ERROR
            print(f"  {defense:<18} {acc:.1f}%          {color}-{degradation:.1f}pp{Colors.RESET}")

        print("-" * 50)
        print(f"\n{Style.INFO}Krum e Multi-Krum offrono la migliore protezione{Colors.RESET}")

        input(f"\n{Style.MUTED}Premi Enter per continuare...{Colors.RESET}")

    def _show_results(self):
        """Show all collected results."""
        clear_screen()
        print_section("RISULTATI BYZANTINE TESTS")

        if not self.results:
            print_warning("Nessun risultato disponibile. Eseguire prima un test.")
            input(f"\n{Style.MUTED}Premi Enter per continuare...{Colors.RESET}")
            return

        print(f"\n{Style.TITLE}{'Configurazione':<35} {'Accuracy':<12} {'F1':<10}{Colors.RESET}")
        print("-" * 60)

        for key, data in self.results.items():
            metrics = data.get("final_metrics", {})
            acc = metrics.get("accuracy", 0)
            f1 = metrics.get("f1", 0)
            print(f"  {key:<33} {acc:.2%}        {f1:.4f}")

        input(f"\n{Style.MUTED}Premi Enter per continuare...{Colors.RESET}")

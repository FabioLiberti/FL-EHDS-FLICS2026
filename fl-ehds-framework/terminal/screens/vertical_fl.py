"""
Vertical Federated Learning screen for FL-EHDS terminal interface.
Simulates SplitNN with Private Set Intersection across multiple parties.
"""

import sys
import numpy as np
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
from terminal.validators import get_int, get_float, get_choice, confirm, display_config_summary
from terminal.menu import Menu, MenuItem, MENU_STYLE


PARTY_NAMES = [
    "Hospital A (Dati demografici)",
    "Hospital B (Esami laboratorio)",
    "Hospital C (Dati lifestyle)",
    "Hospital D (Imaging)",
    "Hospital E (Genomica)",
]


class VerticalFLScreen:
    """Vertical Federated Learning screen."""

    def __init__(self):
        self.config = self._default_config()
        self.results = {}

    def _default_config(self) -> Dict[str, Any]:
        return {
            "num_parties": 3,
            "num_samples": 1000,
            "cut_layer": 1,
            "learning_rate": 0.01,
            "num_rounds": 20,
            "dp_noise": 0.0,
        }

    def run(self):
        while True:
            clear_screen()
            print_section("VERTICAL FEDERATED LEARNING")
            print_info("SplitNN con Private Set Intersection (PSI)")

            menu = Menu("Seleziona azione", [
                MenuItem("1", "Configura parametri", self._configure),
                MenuItem("2", "Esegui PSI + SplitNN", self._run_simulation),
                MenuItem("3", "Confronto partizioni", self._compare_partitions),
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
        clear_screen()
        print_section("CONFIGURAZIONE VERTICAL FL")

        print_subsection("Partizioni Dati")
        self.config["num_parties"] = get_int(
            "Numero di parti (ospedali)", default=self.config["num_parties"],
            min_val=2, max_val=5
        )
        self.config["num_samples"] = get_int(
            "Numero pazienti totali", default=self.config["num_samples"],
            min_val=500, max_val=5000
        )

        print_subsection("SplitNN")
        self.config["cut_layer"] = get_int(
            "Cut layer (punto di taglio rete)", default=self.config["cut_layer"],
            min_val=1, max_val=3
        )
        self.config["learning_rate"] = get_float(
            "Learning rate", default=self.config["learning_rate"],
            min_val=0.001, max_val=0.1
        )
        self.config["num_rounds"] = get_int(
            "Numero epoche", default=self.config["num_rounds"],
            min_val=10, max_val=50
        )

        print_subsection("Privacy")
        self.config["dp_noise"] = get_float(
            "Rumore DP sulle attivazioni (0=disabilitato)", default=self.config["dp_noise"],
            min_val=0.0, max_val=1.0
        )

        # Show party description
        print_subsection("Parti configurate")
        for i in range(self.config["num_parties"]):
            print(f"  Parte {i+1}: {PARTY_NAMES[i]}")

        display_config_summary(self.config)
        input(f"\n{Style.MUTED}Premi Enter per continuare...{Colors.RESET}")

    def _run_simulation(self):
        clear_screen()
        print_section("ESECUZIONE VERTICAL FL")

        display_config_summary(self.config)

        if not confirm("\nAvviare la simulazione?", default=True):
            return

        print()

        try:
            from dashboard.app_v4 import VerticalFLSimulator

            print_info("Inizializzazione simulatore Vertical FL...")

            sim = VerticalFLSimulator(
                num_parties=self.config["num_parties"],
                num_samples=self.config["num_samples"],
            )

            # Phase 1: PSI
            print_subsection("Fase 1: Private Set Intersection")
            common_count, party_sizes = sim.run_psi()

            print(f"\n{Style.TITLE}{'Parte':<35} {'Pazienti':<12} {'Overlap':<12}{Colors.RESET}")
            print("-" * 60)
            for i, size in enumerate(party_sizes):
                name = PARTY_NAMES[i] if i < len(PARTY_NAMES) else f"Parte {i+1}"
                overlap_pct = common_count / size * 100 if size > 0 else 0
                print(f"  {name:<33} {size:<12} {overlap_pct:.1f}%")
            print("-" * 60)
            print(f"  {'Pazienti comuni (dopo PSI)':<33} {common_count}")
            total_overlap = common_count / self.config["num_samples"] * 100
            print(f"  {'Overlap totale':<33} {total_overlap:.1f}%")

            # Phase 2: SplitNN Training
            print()
            print_subsection("Fase 2: Training SplitNN")

            num_epochs = self.config["num_rounds"]
            with FLProgressBar(total=num_epochs, desc="Epoca", color="cyan") as pbar:
                history = sim.train_splitnn(num_epochs)
                for i, epoch_data in enumerate(history):
                    pbar.update(1, acc=f"{epoch_data['accuracy']:.2%}")

            # Display results
            final = history[-1]
            print()
            print_success("Training completato")
            print_subsection("RISULTATI")

            print(f"\n{Style.TITLE}{'Metrica':<25} {'Valore':<15}{Colors.RESET}")
            print("-" * 40)
            print(f"  {'Accuracy finale':<23} {final['accuracy']:.2%}")
            print(f"  {'Loss finale':<23} {final['loss']:.4f}")
            print(f"  {'Pazienti allineati':<23} {common_count}")
            print(f"  {'Parti coinvolte':<23} {self.config['num_parties']}")

            if self.config["dp_noise"] > 0:
                print(f"  {'Rumore DP':<23} {self.config['dp_noise']}")

            # Store results
            key = f"{self.config['num_parties']} parti"
            self.results[key] = {
                "config": self.config.copy(),
                "common_count": common_count,
                "party_sizes": party_sizes,
                "final_accuracy": final["accuracy"],
                "final_loss": final["loss"],
                "history": history,
            }

        except ImportError:
            print_warning("Simulatore non disponibile. Risultati reference:")
            self._display_placeholder_results()

        except Exception as e:
            print_error(f"Errore durante la simulazione: {e}")

        input(f"\n{Style.MUTED}Premi Enter per continuare...{Colors.RESET}")

    def _display_placeholder_results(self):
        print_subsection("RISULTATI REFERENCE (Vertical FL)")

        print(f"\n{Style.TITLE}{'Config':<25} {'Overlap':<12} {'Accuracy':<12} {'Loss':<12}{Colors.RESET}")
        print("-" * 60)

        reference = [
            ("2 parti (Demo+Lab)", "85.2%", "78.2%", "0.312"),
            ("3 parti (Demo+Lab+Life)", "72.1%", "75.8%", "0.358"),
            ("4 parti (+Imaging)", "61.3%", "71.3%", "0.421"),
            ("5 parti (+Genomica)", "52.7%", "68.5%", "0.467"),
        ]

        for config, overlap, acc, loss in reference:
            print(f"  {config:<23} {overlap:<12} {acc:<12} {loss:<12}")

        print(f"\n{Style.MUTED}PSI: Private Set Intersection per allineamento pazienti{Colors.RESET}")
        print(f"{Style.MUTED}SplitNN: Rete neurale partizionata tra le parti{Colors.RESET}")

    def _compare_partitions(self):
        clear_screen()
        print_section("CONFRONTO PARTIZIONI VERTICAL FL")

        print_info("Confronto con 2, 3, 4 e 5 parti")
        print()

        try:
            from dashboard.app_v4 import VerticalFLSimulator

            results_table = []
            num_epochs = min(self.config["num_rounds"], 15)

            for n_parties in range(2, min(6, self.config["num_parties"] + 2)):
                sim = VerticalFLSimulator(
                    num_parties=n_parties,
                    num_samples=self.config["num_samples"],
                )

                common_count, party_sizes = sim.run_psi()
                overlap_pct = common_count / self.config["num_samples"] * 100

                print_info(f"Training con {n_parties} parti...")
                with FLProgressBar(
                    total=num_epochs,
                    desc=f"{n_parties} parti",
                    color="cyan"
                ) as pbar:
                    history = sim.train_splitnn(num_epochs)
                    for epoch_data in history:
                        pbar.update(1, acc=f"{epoch_data['accuracy']:.2%}")

                final = history[-1]
                results_table.append({
                    "parties": n_parties,
                    "overlap": overlap_pct,
                    "accuracy": final["accuracy"],
                    "loss": final["loss"],
                })

            # Display comparison table
            print()
            print_subsection("RISULTATI CONFRONTO")
            print(f"\n{Style.TITLE}{'Parti':<10} {'Overlap':<12} {'Accuracy':<12} {'Loss':<12}{Colors.RESET}")
            print("-" * 50)

            for r in results_table:
                print(f"  {r['parties']:<8} {r['overlap']:.1f}%       {r['accuracy']:.2%}       {r['loss']:.4f}")

            print("-" * 50)
            print(f"\n{Style.INFO}Piu' parti = meno overlap ma piu' feature disponibili{Colors.RESET}")

        except ImportError:
            print_warning("Simulatore non disponibile. Risultati reference:")
            self._display_placeholder_results()

        except Exception as e:
            print_error(f"Errore: {e}")

        input(f"\n{Style.MUTED}Premi Enter per continuare...{Colors.RESET}")

    def _show_results(self):
        clear_screen()
        print_section("RISULTATI VERTICAL FL")

        if not self.results:
            print_warning("Nessun risultato disponibile. Eseguire prima una simulazione.")
            input(f"\n{Style.MUTED}Premi Enter per continuare...{Colors.RESET}")
            return

        print(f"\n{Style.TITLE}{'Configurazione':<25} {'Overlap':<12} {'Accuracy':<12} {'Loss':<12}{Colors.RESET}")
        print("-" * 60)

        for key, data in self.results.items():
            overlap = data["common_count"] / data["config"]["num_samples"] * 100
            print(f"  {key:<23} {overlap:.1f}%       {data['final_accuracy']:.2%}       {data['final_loss']:.4f}")

        input(f"\n{Style.MUTED}Premi Enter per continuare...{Colors.RESET}")

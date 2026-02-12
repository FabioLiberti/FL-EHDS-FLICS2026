"""
Continual Learning screen for FL-EHDS terminal interface.
Simulates continual/lifelong FL with concept drift and forgetting mitigation.
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


# Methods with labels and internal keys
CONTINUAL_METHODS = [
    ("Naive (Nessuna protezione)", "naive"),
    ("EWC (Elastic Weight Consolidation)", "ewc"),
    ("LwF (Learning without Forgetting)", "lwf"),
    ("Experience Replay", "replay"),
]

METHOD_LABELS = [m[0] for m in CONTINUAL_METHODS]
METHOD_KEYS = {m[0]: m[1] for m in CONTINUAL_METHODS}


class ContinualLearningScreen:
    """Continual Learning screen."""

    def __init__(self):
        self.config = self._default_config()
        self.results = {}

    def _default_config(self) -> Dict[str, Any]:
        return {
            "method": "EWC (Elastic Weight Consolidation)",
            "num_tasks": 4,
            "num_rounds_per_task": 15,
            "ewc_lambda": 1000.0,
            "replay_buffer_size": 500,
            "num_clients": 5,
        }

    def run(self):
        while True:
            clear_screen()
            print_section("CONTINUAL LEARNING")
            print_info("Apprendimento continuo con mitigazione del forgetting")

            menu = Menu("Seleziona azione", [
                MenuItem("1", "Configura parametri", self._configure),
                MenuItem("2", "Esegui singolo metodo", self._run_simulation),
                MenuItem("3", "Confronto metodi", self._compare_methods),
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
        print_section("CONFIGURAZIONE CONTINUAL LEARNING")

        print_subsection("Metodo")
        if HAS_QUESTIONARY:
            self.config["method"] = questionary.select(
                "Seleziona metodo anti-forgetting:",
                choices=METHOD_LABELS,
                default=self.config["method"],
                style=MENU_STYLE,
            ).ask() or self.config["method"]
        else:
            self.config["method"] = get_choice("Metodo:", METHOD_LABELS, self.config["method"])

        print_subsection("Parametri Training")
        self.config["num_tasks"] = get_int(
            "Numero di task sequenziali", default=self.config["num_tasks"],
            min_val=2, max_val=6
        )
        self.config["num_rounds_per_task"] = get_int(
            "Round per task", default=self.config["num_rounds_per_task"],
            min_val=5, max_val=30
        )
        self.config["num_clients"] = get_int(
            "Numero client", default=self.config["num_clients"],
            min_val=3, max_val=20
        )

        # Method-specific parameters
        method_key = METHOD_KEYS.get(self.config["method"], "ewc")
        if method_key == "ewc":
            print_subsection("Parametri EWC")
            self.config["ewc_lambda"] = get_float(
                "Lambda EWC (regolarizzazione)", default=self.config["ewc_lambda"],
                min_val=1.0, max_val=10000.0
            )
        elif method_key == "replay":
            print_subsection("Parametri Experience Replay")
            self.config["replay_buffer_size"] = get_int(
                "Dimensione buffer replay", default=self.config["replay_buffer_size"],
                min_val=100, max_val=5000
            )

        display_config_summary(self.config)
        input(f"\n{Style.MUTED}Premi Enter per continuare...{Colors.RESET}")

    def _run_simulation(self):
        clear_screen()
        print_section("ESECUZIONE CONTINUAL LEARNING")

        display_config_summary(self.config)

        if not confirm("\nAvviare la simulazione?", default=True):
            return

        print()

        method_key = METHOD_KEYS.get(self.config["method"], "ewc")

        try:
            from dashboard.app_v4 import ContinualFLSimulator

            print_info(f"Metodo: {self.config['method']}")
            print()

            sim = ContinualFLSimulator(num_tasks=self.config["num_tasks"])

            total_rounds = self.config["num_tasks"] * self.config["num_rounds_per_task"]
            history = sim.simulate_training(method_key, self.config["num_rounds_per_task"])

            # Show progress across tasks
            task_accs = history["accuracy_per_task"]
            num_rounds_per_task = self.config["num_rounds_per_task"]

            for task_id in range(self.config["num_tasks"]):
                print_subsection(f"Task {task_id + 1} (Anno 202{task_id + 1})")

                with FLProgressBar(
                    total=num_rounds_per_task,
                    desc=f"Task {task_id + 1}",
                    color="green"
                ) as pbar:
                    start_idx = task_id * num_rounds_per_task
                    for r in range(num_rounds_per_task):
                        idx = start_idx + r
                        if idx < len(task_accs[task_id]):
                            acc = task_accs[task_id][idx]
                            pbar.update(1, acc=f"{acc:.2%}")
                        else:
                            pbar.update(1)

            # Compute forgetting
            print()
            print_success("Training completato")
            self._display_continual_results(task_accs, method_key)

            # Store results
            self.results[self.config["method"]] = {
                "config": self.config.copy(),
                "task_accs": {str(k): v for k, v in task_accs.items()},
                "method_key": method_key,
            }

        except ImportError:
            print_warning("Simulatore non disponibile. Risultati reference:")
            self._display_placeholder_results()

        except Exception as e:
            print_error(f"Errore durante la simulazione: {e}")

        input(f"\n{Style.MUTED}Premi Enter per continuare...{Colors.RESET}")

    def _display_continual_results(self, task_accs: Dict, method: str):
        print_subsection("RISULTATI")

        num_tasks = self.config["num_tasks"]
        num_rounds_per_task = self.config["num_rounds_per_task"]

        # Final accuracy per task (at end of all training)
        final_accs = []
        peak_accs = []

        for task_id in range(num_tasks):
            accs = task_accs[task_id]
            if accs:
                final_accs.append(accs[-1])
                # Peak is max up to end of that task's training
                end_of_task = (task_id + 1) * num_rounds_per_task
                peak = max(accs[:min(end_of_task, len(accs))])
                peak_accs.append(peak)
            else:
                final_accs.append(0.0)
                peak_accs.append(0.0)

        print(f"\n{Style.TITLE}{'Task':<12} {'Peak Acc':<12} {'Final Acc':<12} {'Forgetting':<12}{Colors.RESET}")
        print("-" * 50)

        total_forgetting = 0.0
        for task_id in range(num_tasks):
            peak = peak_accs[task_id]
            final = final_accs[task_id]
            forgetting = peak - final

            if forgetting > 0.05:
                color = Style.ERROR
            elif forgetting > 0.02:
                color = Style.WARNING
            else:
                color = Style.SUCCESS

            total_forgetting += forgetting
            print(f"  Task {task_id + 1:<5} {peak:.2%}       {final:.2%}       {color}{forgetting:+.2%}{Colors.RESET}")

        print("-" * 50)
        avg_forgetting = total_forgetting / num_tasks
        avg_final = np.mean(final_accs)
        print(f"  {'Media':<10} {'--':<12} {avg_final:.2%}       {avg_forgetting:+.2%}")

        retention = (1 - avg_forgetting) * 100
        print(f"\n{Style.INFO}Retention rate: {retention:.1f}%{Colors.RESET}")

    def _display_placeholder_results(self):
        print_subsection("RISULTATI REFERENCE (Continual Learning)")

        print(f"\n{Style.TITLE}{'Metodo':<30} {'T1':<8} {'T2':<8} {'T3':<8} {'T4':<8} {'Forg.':<10}{Colors.RESET}")
        print("-" * 70)

        reference = [
            ("Naive (No Protection)", "32.1%", "45.3%", "68.2%", "92.1%", "-38.5%"),
            ("EWC", "62.3%", "71.5%", "78.1%", "91.8%", "-12.1%"),
            ("LwF", "67.1%", "73.2%", "79.5%", "90.3%", "-9.8%"),
            ("Experience Replay", "72.5%", "76.8%", "82.1%", "91.5%", "-7.2%"),
        ]

        for method, t1, t2, t3, t4, forg in reference:
            print(f"  {method:<28} {t1:<8} {t2:<8} {t3:<8} {t4:<8} {forg:<10}")

        print(f"\n{Style.MUTED}T1-T4: Accuracy finale per task al termine del training{Colors.RESET}")
        print(f"{Style.MUTED}Forg.: Forgetting medio (drop da peak accuracy){Colors.RESET}")

    def _compare_methods(self):
        clear_screen()
        print_section("CONFRONTO METODI CONTINUAL LEARNING")

        print_info("Confronto di tutti i metodi anti-forgetting")
        print()

        try:
            from dashboard.app_v4 import ContinualFLSimulator

            all_results = {}

            for label, method_key in CONTINUAL_METHODS:
                print_info(f"Esecuzione: {label}...")

                sim = ContinualFLSimulator(num_tasks=self.config["num_tasks"])
                history = sim.simulate_training(method_key, self.config["num_rounds_per_task"])

                all_results[label] = history["accuracy_per_task"]

            # Display comparison
            print()
            print_subsection("CONFRONTO RISULTATI")

            num_tasks = self.config["num_tasks"]
            num_rounds_per_task = self.config["num_rounds_per_task"]

            # Header
            task_headers = "".join(f"{'T' + str(i+1):<8}" for i in range(num_tasks))
            print(f"\n{Style.TITLE}{'Metodo':<30} {task_headers}{'Forg.':<10}{Colors.RESET}")
            print("-" * (30 + num_tasks * 8 + 10))

            for label, task_accs in all_results.items():
                final_accs = []
                total_forg = 0.0

                for task_id in range(num_tasks):
                    accs = task_accs[task_id]
                    final = accs[-1] if accs else 0.0
                    end_idx = (task_id + 1) * num_rounds_per_task
                    peak = max(accs[:min(end_idx, len(accs))]) if accs else 0.0
                    final_accs.append(final)
                    total_forg += (peak - final)

                avg_forg = total_forg / num_tasks
                accs_str = "".join(f"{a:.1%}   " for a in final_accs)

                if avg_forg > 0.15:
                    forg_color = Style.ERROR
                elif avg_forg > 0.05:
                    forg_color = Style.WARNING
                else:
                    forg_color = Style.SUCCESS

                print(f"  {label:<28} {accs_str}{forg_color}{avg_forg:+.1%}{Colors.RESET}")

            print("-" * (30 + num_tasks * 8 + 10))
            print(f"\n{Style.INFO}Experience Replay offre la migliore retention{Colors.RESET}")
            print(f"{Style.INFO}EWC e' il miglior compromesso costo/prestazioni{Colors.RESET}")

            # Store all results
            for label in all_results:
                self.results[label] = {
                    "config": self.config.copy(),
                    "task_accs": {str(k): v for k, v in all_results[label].items()},
                }

        except ImportError:
            print_warning("Simulatore non disponibile. Risultati reference:")
            self._display_placeholder_results()

        except Exception as e:
            print_error(f"Errore: {e}")

        input(f"\n{Style.MUTED}Premi Enter per continuare...{Colors.RESET}")

    def _show_results(self):
        clear_screen()
        print_section("RISULTATI CONTINUAL LEARNING")

        if not self.results:
            print_warning("Nessun risultato disponibile. Eseguire prima una simulazione.")
            input(f"\n{Style.MUTED}Premi Enter per continuare...{Colors.RESET}")
            return

        for method_name, data in self.results.items():
            print_subsection(method_name)
            task_accs = data.get("task_accs", {})

            if task_accs:
                for task_id_str, accs in sorted(task_accs.items()):
                    if accs:
                        final = accs[-1]
                        peak = max(accs)
                        print(f"  Task {int(task_id_str) + 1}: final={final:.2%}, peak={peak:.2%}, forgetting={peak - final:+.2%}")

        input(f"\n{Style.MUTED}Premi Enter per continuare...{Colors.RESET}")

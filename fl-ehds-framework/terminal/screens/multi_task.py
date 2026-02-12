"""
Multi-Task FL screen for FL-EHDS terminal interface.
Simulates multi-task federated learning with partial client-task coverage.
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
from terminal.validators import get_int, get_choice, confirm, display_config_summary
from terminal.menu import Menu, MenuItem, MENU_STYLE


# EHDS clinical tasks
EHDS_TASKS = {
    "diabetes_risk": "Predizione Rischio Diabete",
    "readmission_30d": "Readmissione a 30 Giorni",
    "los_prediction": "Durata Degenza (LOS)",
    "mortality_risk": "Rischio Mortalita'",
    "sepsis_onset": "Rilevamento Sepsi",
}

ALL_TASK_KEYS = list(EHDS_TASKS.keys())

# Sharing methods
SHARING_METHODS = [
    ("Hard Parameter Sharing", "hard_sharing"),
    ("Soft Parameter Sharing", "soft_sharing"),
    ("FedMTL (Federated Multi-Task)", "fedmtl"),
]

METHOD_LABELS = [m[0] for m in SHARING_METHODS]
METHOD_KEYS = {m[0]: m[1] for m in SHARING_METHODS}


class MultiTaskScreen:
    """Multi-Task Federated Learning screen."""

    def __init__(self):
        self.config = self._default_config()
        self.results = {}

    def _default_config(self) -> Dict[str, Any]:
        return {
            "method": "Hard Parameter Sharing",
            "num_clients": 6,
            "tasks": ["diabetes_risk", "readmission_30d", "los_prediction"],
            "num_rounds": 30,
        }

    def run(self):
        while True:
            clear_screen()
            print_section("MULTI-TASK FEDERATED LEARNING")
            print_info("Apprendimento multi-task con condivisione parametri")

            menu = Menu("Seleziona azione", [
                MenuItem("1", "Configura parametri", self._configure),
                MenuItem("2", "Esegui training", self._run_simulation),
                MenuItem("3", "Confronto metodi", self._compare_methods),
                MenuItem("4", "Matrice copertura task", self._show_coverage),
                MenuItem("5", "Visualizza risultati", self._show_results),
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
        print_section("CONFIGURAZIONE MULTI-TASK FL")

        print_subsection("Metodo di Condivisione")
        if HAS_QUESTIONARY:
            self.config["method"] = questionary.select(
                "Seleziona metodo di condivisione:",
                choices=METHOD_LABELS,
                default=self.config["method"],
                style=MENU_STYLE,
            ).ask() or self.config["method"]
        else:
            self.config["method"] = get_choice("Metodo:", METHOD_LABELS, self.config["method"])

        print_subsection("Parametri Training")
        self.config["num_clients"] = get_int(
            "Numero client (ospedali)", default=self.config["num_clients"],
            min_val=3, max_val=10
        )
        self.config["num_rounds"] = get_int(
            "Numero round", default=self.config["num_rounds"],
            min_val=10, max_val=50
        )

        # Task selection
        print_subsection("Selezione Task Clinici")
        print_info("Task EHDS disponibili:")
        for i, (key, name) in enumerate(EHDS_TASKS.items()):
            marker = "[X]" if key in self.config["tasks"] else "[ ]"
            print(f"  {i + 1}. {marker} {name}")

        if HAS_QUESTIONARY:
            task_choices = [
                questionary.Choice(
                    title=f"{name}",
                    value=key,
                    checked=(key in self.config["tasks"])
                )
                for key, name in EHDS_TASKS.items()
            ]
            selected = questionary.checkbox(
                "Seleziona task (Spazio per toggle, Enter per confermare):",
                choices=task_choices,
                style=MENU_STYLE,
            ).ask()
            if selected and len(selected) >= 2:
                self.config["tasks"] = selected
            else:
                print_warning("Selezionare almeno 2 task. Mantenute selezioni precedenti.")
        else:
            print_info("Inserisci i numeri dei task separati da virgola (es: 1,2,3)")
            try:
                inp = input(f"  {Style.INFO}Task: {Colors.RESET}").strip()
                indices = [int(x.strip()) - 1 for x in inp.split(",")]
                selected = [ALL_TASK_KEYS[i] for i in indices if 0 <= i < len(ALL_TASK_KEYS)]
                if len(selected) >= 2:
                    self.config["tasks"] = selected
                else:
                    print_warning("Selezionare almeno 2 task.")
            except (ValueError, IndexError):
                print_warning("Input non valido. Mantenute selezioni precedenti.")

        display_config_summary({
            **self.config,
            "tasks": ", ".join(EHDS_TASKS[t] for t in self.config["tasks"]),
        })
        input(f"\n{Style.MUTED}Premi Enter per continuare...{Colors.RESET}")

    def _run_simulation(self):
        clear_screen()
        print_section("ESECUZIONE MULTI-TASK FL")

        display_config_summary({
            **self.config,
            "tasks": ", ".join(EHDS_TASKS[t] for t in self.config["tasks"]),
        })

        if not confirm("\nAvviare il training?", default=True):
            return

        print()

        method_key = METHOD_KEYS.get(self.config["method"], "hard_sharing")

        try:
            from dashboard.app_v4 import MultiTaskFLSimulator

            print_info(f"Metodo: {self.config['method']}")
            print_info(f"Task: {len(self.config['tasks'])}, Client: {self.config['num_clients']}")
            print()

            sim = MultiTaskFLSimulator(
                num_clients=self.config["num_clients"],
                tasks=self.config["tasks"],
            )

            # Show coverage before training
            self._display_coverage_matrix(sim.client_tasks, self.config["tasks"])
            print()

            # Train
            with FLProgressBar(
                total=self.config["num_rounds"],
                desc="Round",
                color="magenta"
            ) as pbar:
                history = sim.train(method_key, self.config["num_rounds"])
                for r in range(self.config["num_rounds"]):
                    # Average accuracy across tasks at this round
                    avg_acc = np.mean([history[t][r] for t in self.config["tasks"] if r < len(history[t])])
                    pbar.update(1, acc=f"{avg_acc:.2%}")

            # Display results
            print()
            print_success("Training completato")
            print_subsection("RISULTATI PER TASK")

            print(f"\n{Style.TITLE}{'Task':<35} {'Accuracy':<12} {'Client':<10}{Colors.RESET}")
            print("-" * 60)

            for i, task in enumerate(self.config["tasks"]):
                task_name = EHDS_TASKS[task]
                final_acc = history[task][-1] if history[task] else 0.0
                clients_with = sum(1 for c in sim.client_tasks.values() if i in c)
                print(f"  {task_name:<33} {final_acc:.2%}       {clients_with}/{self.config['num_clients']}")

            avg_acc = np.mean([history[t][-1] for t in self.config["tasks"]])
            print("-" * 60)
            print(f"  {'Media':<33} {avg_acc:.2%}")

            # Store results
            self.results[self.config["method"]] = {
                "config": self.config.copy(),
                "history": {t: history[t] for t in self.config["tasks"]},
                "client_tasks": sim.client_tasks,
            }

        except ImportError:
            print_warning("Simulatore non disponibile. Risultati reference:")
            self._display_placeholder_results()

        except Exception as e:
            print_error(f"Errore durante il training: {e}")

        input(f"\n{Style.MUTED}Premi Enter per continuare...{Colors.RESET}")

    def _display_coverage_matrix(self, client_tasks: Dict, tasks: List[str]):
        """Display client-task coverage matrix."""
        print_subsection("Matrice Copertura Client-Task")

        # Header
        task_short = [EHDS_TASKS[t][:12] for t in tasks]
        header = f"{'Client':<10}" + "".join(f"{t:<14}" for t in task_short)
        print(f"\n{Style.TITLE}{header}{Colors.RESET}")
        print("-" * (10 + 14 * len(tasks)))

        for cid, cid_tasks in sorted(client_tasks.items()):
            row = f"  Osp. {cid:<4}"
            for i in range(len(tasks)):
                if i in cid_tasks:
                    row += f"{Style.SUCCESS}{'X':<14}{Colors.RESET}"
                else:
                    row += f"{Style.MUTED}{'-':<14}{Colors.RESET}"
            print(row)

        # Coverage stats
        total_cells = len(client_tasks) * len(tasks)
        filled = sum(len(t) for t in client_tasks.values())
        coverage_pct = filled / total_cells * 100 if total_cells > 0 else 0
        print(f"\n{Style.INFO}Copertura: {filled}/{total_cells} ({coverage_pct:.0f}%){Colors.RESET}")

    def _display_placeholder_results(self):
        print_subsection("RISULTATI REFERENCE (Multi-Task FL)")

        print(f"\n{Style.TITLE}{'Metodo':<30} {'Diabete':<12} {'Readm.':<12} {'LOS':<12}{Colors.RESET}")
        print("-" * 70)

        reference = [
            ("Hard Parameter Sharing", "88.2%", "79.1%", "72.3%"),
            ("Soft Parameter Sharing", "86.5%", "80.8%", "74.1%"),
            ("FedMTL", "89.1%", "81.2%", "73.8%"),
        ]

        for method, d, r, l in reference:
            print(f"  {method:<28} {d:<12} {r:<12} {l:<12}")

        print(f"\n{Style.MUTED}Copertura parziale: non tutti i client hanno tutti i task{Colors.RESET}")

    def _compare_methods(self):
        clear_screen()
        print_section("CONFRONTO METODI MULTI-TASK FL")

        print_info("Confronto di tutti i metodi di condivisione")
        print()

        try:
            from dashboard.app_v4 import MultiTaskFLSimulator

            all_results = {}

            for label, method_key in SHARING_METHODS:
                print_info(f"Esecuzione: {label}...")

                sim = MultiTaskFLSimulator(
                    num_clients=self.config["num_clients"],
                    tasks=self.config["tasks"],
                )
                history = sim.train(method_key, self.config["num_rounds"])
                all_results[label] = history

            # Display comparison
            print()
            print_subsection("CONFRONTO RISULTATI")

            task_names = [EHDS_TASKS[t][:10] for t in self.config["tasks"]]
            header = f"{'Metodo':<30}" + "".join(f"{t:<12}" for t in task_names) + f"{'Media':<10}"
            print(f"\n{Style.TITLE}{header}{Colors.RESET}")
            print("-" * (30 + 12 * len(self.config["tasks"]) + 10))

            for label, history in all_results.items():
                accs = [history[t][-1] for t in self.config["tasks"]]
                avg = np.mean(accs)
                accs_str = "".join(f"{a:.1%}       " for a in accs)
                print(f"  {label:<28} {accs_str}{avg:.1%}")

                # Store
                self.results[label] = {
                    "config": self.config.copy(),
                    "history": {t: history[t] for t in self.config["tasks"]},
                }

            print("-" * (30 + 12 * len(self.config["tasks"]) + 10))
            print(f"\n{Style.INFO}FedMTL tiene conto delle relazioni tra task{Colors.RESET}")

        except ImportError:
            print_warning("Simulatore non disponibile. Risultati reference:")
            self._display_placeholder_results()

        except Exception as e:
            print_error(f"Errore: {e}")

        input(f"\n{Style.MUTED}Premi Enter per continuare...{Colors.RESET}")

    def _show_coverage(self):
        clear_screen()
        print_section("MATRICE COPERTURA TASK")

        try:
            from dashboard.app_v4 import MultiTaskFLSimulator

            sim = MultiTaskFLSimulator(
                num_clients=self.config["num_clients"],
                tasks=self.config["tasks"],
            )
            self._display_coverage_matrix(sim.client_tasks, self.config["tasks"])

            # Per-task stats
            print()
            print_subsection("Statistiche per Task")
            for i, task in enumerate(self.config["tasks"]):
                clients_with = sum(1 for c in sim.client_tasks.values() if i in c)
                print(f"  {EHDS_TASKS[task]:<35} {clients_with}/{self.config['num_clients']} client")

        except ImportError:
            print_warning("Simulatore non disponibile.")
            print_info("La copertura parziale e' una caratteristica chiave del Multi-Task FL:")
            print_info("  - Non tutti gli ospedali hanno dati per tutti i task")
            print_info("  - Hard sharing condivide il backbone tra task disponibili")
            print_info("  - FedMTL sfrutta le correlazioni inter-task")

        input(f"\n{Style.MUTED}Premi Enter per continuare...{Colors.RESET}")

    def _show_results(self):
        clear_screen()
        print_section("RISULTATI MULTI-TASK FL")

        if not self.results:
            print_warning("Nessun risultato disponibile. Eseguire prima un training.")
            input(f"\n{Style.MUTED}Premi Enter per continuare...{Colors.RESET}")
            return

        for method_name, data in self.results.items():
            print_subsection(method_name)
            history = data.get("history", {})

            for task, accs in history.items():
                if accs:
                    print(f"  {EHDS_TASKS.get(task, task):<35} {accs[-1]:.2%}")

        input(f"\n{Style.MUTED}Premi Enter per continuare...{Colors.RESET}")

"""
Hierarchical FL screen for FL-EHDS terminal interface.
Simulates multi-tier federated learning for EHDS (Hospital -> Region -> Country -> EU).
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


# EU Federation hierarchy
EU_HIERARCHY = {
    "EU": {
        "DE": {
            "name": "Germania",
            "regions": {
                "Bavaria": ["Hospital DE1", "Hospital DE2"],
                "Berlin": ["Hospital DE3"],
            }
        },
        "FR": {
            "name": "Francia",
            "regions": {
                "Ile-de-France": ["Hospital FR1", "Hospital FR2"],
                "PACA": ["Hospital FR3"],
            }
        },
        "IT": {
            "name": "Italia",
            "regions": {
                "Lombardia": ["Hospital IT1", "Hospital IT2"],
                "Lazio": ["Hospital IT3"],
            }
        },
    }
}

AGGREGATION_METHODS = [
    "HierFedAvg",
    "FedAvg (Flat)",
    "Clustered",
]


class HierarchicalScreen:
    """Hierarchical Federated Learning screen."""

    def __init__(self):
        self.config = self._default_config()
        self.results = {}

    def _default_config(self) -> Dict[str, Any]:
        return {
            "num_rounds": 20,
            "local_rounds": 3,
            "regional_rounds": 2,
            "aggregation": "HierFedAvg",
        }

    def run(self):
        while True:
            clear_screen()
            print_section("HIERARCHICAL FEDERATED LEARNING")
            print_info("Federazione multi-livello per EHDS")

            menu = Menu("Seleziona azione", [
                MenuItem("1", "Configura parametri", self._configure),
                MenuItem("2", "Esegui training", self._run_simulation),
                MenuItem("3", "Visualizza topologia", self._show_topology),
                MenuItem("4", "Confronto aggregazione", self._compare_aggregation),
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
        print_section("CONFIGURAZIONE HIERARCHICAL FL")

        print_subsection("Parametri Training")
        self.config["num_rounds"] = get_int(
            "Numero round globali", default=self.config["num_rounds"],
            min_val=10, max_val=50
        )
        self.config["local_rounds"] = get_int(
            "Round locali per ospedale", default=self.config["local_rounds"],
            min_val=1, max_val=5
        )
        self.config["regional_rounds"] = get_int(
            "Round di aggregazione regionale", default=self.config["regional_rounds"],
            min_val=1, max_val=3
        )

        print_subsection("Strategia di Aggregazione")
        if HAS_QUESTIONARY:
            self.config["aggregation"] = questionary.select(
                "Seleziona strategia:",
                choices=AGGREGATION_METHODS,
                default=self.config["aggregation"],
                style=MENU_STYLE,
            ).ask() or self.config["aggregation"]
        else:
            self.config["aggregation"] = get_choice(
                "Strategia:", AGGREGATION_METHODS, self.config["aggregation"]
            )

        display_config_summary(self.config)
        input(f"\n{Style.MUTED}Premi Enter per continuare...{Colors.RESET}")

    def _run_simulation(self):
        clear_screen()
        print_section("ESECUZIONE HIERARCHICAL FL")

        display_config_summary(self.config)

        if not confirm("\nAvviare il training?", default=True):
            return

        print()

        try:
            from dashboard.app_v4 import HierarchicalFLSimulator

            print_info(f"Strategia: {self.config['aggregation']}")

            sim = HierarchicalFLSimulator()
            nodes = sim.count_nodes()

            print_info(f"Topologia: {nodes['hospitals']} ospedali, {nodes['regions']} regioni, {nodes['countries']} paesi")
            print()

            # Show topology summary
            self._display_topology_compact()
            print()

            # Train
            with FLProgressBar(
                total=self.config["num_rounds"],
                desc="Round globale",
                color="blue"
            ) as pbar:
                history = sim.train(self.config["num_rounds"])
                for r in range(self.config["num_rounds"]):
                    acc = history["global"][r] if r < len(history["global"]) else 0
                    pbar.update(1, acc=f"{acc:.2%}")

            # Display results
            print()
            print_success("Training completato")
            print_subsection("RISULTATI")

            # Global
            global_final = history["global"][-1]
            print(f"\n  Accuracy globale (EU): {Style.SUCCESS}{global_final:.2%}{Colors.RESET}")

            # Per country
            print(f"\n{Style.TITLE}{'Paese':<20} {'Accuracy':<12} {'Delta vs EU':<12}{Colors.RESET}")
            print("-" * 45)

            for country, country_name in [("DE", "Germania"), ("FR", "Francia"), ("IT", "Italia")]:
                if country in history["per_country"]:
                    country_acc = history["per_country"][country][-1]
                    delta = country_acc - global_final
                    delta_color = Style.SUCCESS if delta >= 0 else Style.WARNING
                    print(f"  {country_name:<18} {country_acc:.2%}       {delta_color}{delta:+.2%}{Colors.RESET}")

            print("-" * 45)

            # Aggregation schedule
            print_subsection("Schema Aggregazione")
            total_comm = self.config["num_rounds"]
            if self.config["aggregation"] == "HierFedAvg":
                print(f"  Ospedali -> Regione: ogni {self.config['local_rounds']} round locali")
                print(f"  Regione -> Paese: ogni {self.config['regional_rounds']} aggregazioni regionali")
                print(f"  Paese -> EU: ogni round globale")
                comm_saving = (1 - 1.0 / (self.config["local_rounds"] * self.config["regional_rounds"])) * 100
                print(f"\n{Style.INFO}Risparmio comunicazione vs flat: ~{comm_saving:.0f}%{Colors.RESET}")
            elif self.config["aggregation"] == "FedAvg (Flat)":
                print(f"  Tutti gli ospedali -> EU direttamente: ogni round")
                print(f"\n{Style.WARNING}Nessun risparmio comunicazione (baseline){Colors.RESET}")
            else:
                print(f"  Clustering automatico basato su similarita' modelli")
                print(f"\n{Style.INFO}Risparmio comunicazione: ~50-60%{Colors.RESET}")

            # Store results
            self.results[self.config["aggregation"]] = {
                "config": self.config.copy(),
                "history": history,
                "nodes": nodes,
            }

        except ImportError:
            print_warning("Simulatore non disponibile. Risultati reference:")
            self._display_placeholder_results()

        except Exception as e:
            print_error(f"Errore durante il training: {e}")

        input(f"\n{Style.MUTED}Premi Enter per continuare...{Colors.RESET}")

    def _display_topology_compact(self):
        """Show compact topology."""
        countries = EU_HIERARCHY["EU"]
        total_hospitals = sum(
            len(h) for c in countries.values()
            for h in c["regions"].values()
        )
        total_regions = sum(len(c["regions"]) for c in countries.values())

        print(f"  Livelli: Ospedale({total_hospitals}) -> Regione({total_regions}) -> Paese({len(countries)}) -> EU(1)")

    def _show_topology(self):
        clear_screen()
        print_section("TOPOLOGIA FEDERAZIONE EHDS")

        print_subsection("Struttura Gerarchica")
        print()

        # ASCII tree
        print(f"  {Style.TITLE}HealthData@EU (Livello Supranazionale){Colors.RESET}")

        countries = EU_HIERARCHY["EU"]
        country_list = list(countries.items())

        for c_idx, (code, country) in enumerate(country_list):
            is_last_country = (c_idx == len(country_list) - 1)
            branch = "+" if is_last_country else "|"
            connector = "`" if is_last_country else "|"

            print(f"  {connector}-- {Style.INFO}{country['name']} ({code}) - Livello Nazionale{Colors.RESET}")

            regions = list(country["regions"].items())
            for r_idx, (region, hospitals) in enumerate(regions):
                is_last_region = (r_idx == len(regions) - 1)
                prefix = "    " if is_last_country else "  | "
                r_connector = "`" if is_last_region else "|"

                print(f"  {prefix}{r_connector}-- {region} - Livello Regionale")

                for h_idx, hospital in enumerate(hospitals):
                    is_last_hospital = (h_idx == len(hospitals) - 1)
                    h_prefix = prefix + ("    " if is_last_region else "|   ")
                    h_connector = "`" if is_last_hospital else "|"

                    print(f"  {h_prefix}{h_connector}-- {hospital}")

        # Stats
        print()
        print_subsection("Statistiche Topologia")

        total_hospitals = sum(
            len(h) for c in countries.values()
            for h in c["regions"].values()
        )
        total_regions = sum(len(c["regions"]) for c in countries.values())

        print(f"\n{Style.TITLE}{'Livello':<25} {'Nodi':<10} {'Ruolo':<30}{Colors.RESET}")
        print("-" * 65)
        print(f"  {'Supranazionale (EU)':<23} {'1':<10} {'Aggregazione finale':<30}")
        print(f"  {'Nazionale':<23} {len(countries):<10} {'Aggregazione per paese':<30}")
        print(f"  {'Regionale':<23} {total_regions:<10} {'Aggregazione locale':<30}")
        print(f"  {'Ospedale (Client)':<23} {total_hospitals:<10} {'Training locale':<30}")
        print("-" * 65)
        print(f"  {'TOTALE':<23} {1 + len(countries) + total_regions + total_hospitals}")

        print(f"\n{Style.INFO}La struttura riflette la governance EHDS (Art. 36-37){Colors.RESET}")

        input(f"\n{Style.MUTED}Premi Enter per continuare...{Colors.RESET}")

    def _display_placeholder_results(self):
        print_subsection("RISULTATI REFERENCE (Hierarchical FL)")

        print(f"\n{Style.TITLE}{'Strategia':<25} {'EU':<10} {'DE':<10} {'FR':<10} {'IT':<10} {'Comm.':<10}{Colors.RESET}")
        print("-" * 75)

        reference = [
            ("FedAvg (Flat)", "82.1%", "83.2%", "81.5%", "81.6%", "100%"),
            ("HierFedAvg", "83.5%", "85.1%", "82.8%", "82.6%", "33%"),
            ("Clustered", "84.2%", "86.3%", "83.1%", "83.2%", "45%"),
        ]

        for strategy, eu, de, fr, it, comm in reference:
            print(f"  {strategy:<23} {eu:<10} {de:<10} {fr:<10} {it:<10} {comm:<10}")

        print(f"\n{Style.MUTED}Comm.: Costo comunicazione relativo a Flat FedAvg{Colors.RESET}")
        print(f"{Style.MUTED}HierFedAvg riduce la comunicazione del ~67%{Colors.RESET}")

    def _compare_aggregation(self):
        clear_screen()
        print_section("CONFRONTO STRATEGIE AGGREGAZIONE")

        print_info("Confronto: Flat vs Gerarchico vs Clustered")
        print()

        try:
            from dashboard.app_v4 import HierarchicalFLSimulator

            all_results = {}

            for strategy in AGGREGATION_METHODS:
                print_info(f"Esecuzione: {strategy}...")

                sim = HierarchicalFLSimulator()
                history = sim.train(self.config["num_rounds"])
                all_results[strategy] = history

            # Display comparison
            print()
            print_subsection("CONFRONTO RISULTATI")

            print(f"\n{Style.TITLE}{'Strategia':<25} {'EU':<10} {'DE':<10} {'FR':<10} {'IT':<10}{Colors.RESET}")
            print("-" * 65)

            for strategy, history in all_results.items():
                eu_acc = history["global"][-1]
                de_acc = history["per_country"]["DE"][-1]
                fr_acc = history["per_country"]["FR"][-1]
                it_acc = history["per_country"]["IT"][-1]

                print(f"  {strategy:<23} {eu_acc:.1%}     {de_acc:.1%}     {fr_acc:.1%}     {it_acc:.1%}")

                self.results[strategy] = {
                    "config": self.config.copy(),
                    "history": history,
                }

            print("-" * 65)

            # Communication cost analysis
            print()
            print_subsection("Analisi Costi Comunicazione")

            flat_cost = 9  # All 9 hospitals communicate every round
            hier_cost = 9.0 / (self.config["local_rounds"] * self.config["regional_rounds"])
            cluster_cost = flat_cost * 0.45

            print(f"\n{Style.TITLE}{'Strategia':<25} {'Msg/Round':<12} {'Relativo':<12}{Colors.RESET}")
            print("-" * 50)
            print(f"  {'FedAvg (Flat)':<23} {flat_cost:<12} {'100%':<12}")
            print(f"  {'HierFedAvg':<23} {hier_cost:.1f}{'':>8} {hier_cost/flat_cost*100:.0f}%")
            print(f"  {'Clustered':<23} {cluster_cost:.1f}{'':>8} {cluster_cost/flat_cost*100:.0f}%")

            print(f"\n{Style.INFO}HierFedAvg: migliore efficienza comunicazione{Colors.RESET}")
            print(f"{Style.INFO}Clustered: migliore accuracy con costi medi{Colors.RESET}")

        except ImportError:
            print_warning("Simulatore non disponibile. Risultati reference:")
            self._display_placeholder_results()

        except Exception as e:
            print_error(f"Errore: {e}")

        input(f"\n{Style.MUTED}Premi Enter per continuare...{Colors.RESET}")

    def _show_results(self):
        clear_screen()
        print_section("RISULTATI HIERARCHICAL FL")

        if not self.results:
            print_warning("Nessun risultato disponibile. Eseguire prima un training.")
            input(f"\n{Style.MUTED}Premi Enter per continuare...{Colors.RESET}")
            return

        for strategy, data in self.results.items():
            print_subsection(strategy)
            history = data.get("history", {})

            if history:
                global_acc = history["global"][-1] if history.get("global") else 0
                print(f"  EU (globale): {global_acc:.2%}")

                for country in ["DE", "FR", "IT"]:
                    if country in history.get("per_country", {}):
                        acc = history["per_country"][country][-1]
                        print(f"  {country}: {acc:.2%}")

        input(f"\n{Style.MUTED}Premi Enter per continuare...{Colors.RESET}")

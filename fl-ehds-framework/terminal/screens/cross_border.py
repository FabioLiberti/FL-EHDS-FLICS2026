"""
Cross-Border Federation Screen.

Terminal interface for running cross-border FL experiments across
EU member states with jurisdiction-aware privacy, latency, and
HDAB policy enforcement.
"""

import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from terminal.colors import Colors, Style, print_section, print_subsection
from terminal.colors import print_info, print_success, print_warning, print_error
from terminal.colors import clear_screen
from terminal.validators import get_int, get_float, get_choice, confirm
from terminal.cross_border import (
    EU_COUNTRY_PROFILES,
    CrossBorderFederatedTrainer,
    HOSPITAL_NAMES,
)


FL_ALGORITHMS = [
    "FedAvg", "FedProx", "SCAFFOLD", "FedNova", "FedDyn",
    "FedAdam", "FedYogi", "FedAdagrad", "Per-FedAvg", "Ditto",
]

EHDS_PURPOSES = [
    "scientific_research",
    "public_health_surveillance",
    "health_policy",
    "education_training",
    "ai_system_development",
    "personalized_medicine",
]

# Pre-configured cross-border scenarios for quick experiments
SCENARIOS = {
    "Western Europe (strict)": {
        "countries": ["DE", "FR", "NL", "AT"],
        "description": "Strict HDAB countries with low epsilon limits",
    },
    "Southern Europe (moderate)": {
        "countries": ["IT", "ES", "PT"],
        "description": "Moderate privacy requirements, higher epsilon",
    },
    "Full EU (heterogeneous)": {
        "countries": ["DE", "FR", "IT", "ES", "NL"],
        "description": "Mixed strictness across 5 major EU economies",
    },
    "Nordic + DACH (privacy-first)": {
        "countries": ["SE", "DE", "AT", "NL"],
        "description": "Strictest HDAB policies in the EU",
    },
    "All 10 countries": {
        "countries": ["DE", "FR", "IT", "ES", "NL", "SE", "PL", "AT", "BE", "PT"],
        "description": "Complete heterogeneous pan-European federation",
    },
}


class CrossBorderScreen:
    """Cross-border federation simulation screen."""

    def __init__(self):
        self.config = self._default_config()
        self.results = None

    def _default_config(self) -> Dict[str, Any]:
        """Return default configuration from YAML with hardcoded fallbacks."""
        yaml_defaults = {}
        cb_defaults = {}
        try:
            from config.config_loader import get_training_defaults, load_config
            yaml_defaults = get_training_defaults()
            cfg = load_config()
            cb_defaults = cfg.get("cross_border", {})
        except (ImportError, Exception):
            pass

        return {
            "countries": cb_defaults.get("countries", ["DE", "FR", "IT", "ES", "NL"]),
            "hospitals_per_country": cb_defaults.get("hospitals_per_country", 1),
            "algorithm": yaml_defaults.get("algorithm", "FedAvg"),
            "num_rounds": yaml_defaults.get("num_rounds", 15),
            "local_epochs": yaml_defaults.get("local_epochs", 3),
            "batch_size": yaml_defaults.get("batch_size", 32),
            "learning_rate": yaml_defaults.get("learning_rate", 0.01),
            "global_epsilon": cb_defaults.get("global_epsilon", yaml_defaults.get("dp_epsilon", 10.0)),
            "purpose": cb_defaults.get("purpose", "scientific_research"),
            "simulate_latency": cb_defaults.get("simulate_latency", True),
            "dataset_type": yaml_defaults.get("dataset_type", "synthetic"),
            "dataset_path": None,
            "img_size": yaml_defaults.get("img_size", 128),
            "seed": yaml_defaults.get("seed", 42),
            "mu": yaml_defaults.get("mu", 0.1),
            "server_lr": yaml_defaults.get("server_lr", 0.1),
            "beta1": yaml_defaults.get("beta1", 0.9),
            "beta2": yaml_defaults.get("beta2", 0.99),
            "tau": yaml_defaults.get("tau", 1e-3),
        }

    def run(self):
        """Main screen loop."""
        while True:
            clear_screen()
            print_section("CROSS-BORDER FEDERATION (EHDS)")
            print(f"\n{Style.INFO}Simulazione FL cross-border con giurisdizioni EU diverse,{Colors.RESET}")
            print(f"{Style.INFO}policy HDAB nazionali, latenza di rete e opt-out.{Colors.RESET}\n")

            print(f"  {Style.TITLE}1{Colors.RESET} - Configura esperimento")
            print(f"  {Style.TITLE}2{Colors.RESET} - Scenari pre-configurati")
            print(f"  {Style.TITLE}3{Colors.RESET} - Mostra profili paesi EU")
            print(f"  {Style.TITLE}4{Colors.RESET} - Esegui simulazione")
            print(f"  {Style.TITLE}0{Colors.RESET} - Torna al menu principale")

            choice = get_choice("\nScegli opzione", ["1", "2", "3", "4", "0"], default="1")

            if choice == "1":
                self._configure()
            elif choice == "2":
                self._select_scenario()
            elif choice == "3":
                self._show_country_profiles()
            elif choice == "4":
                self._run_simulation()
            elif choice == "0":
                return

    def _show_country_profiles(self):
        """Display all EU country profiles."""
        clear_screen()
        print_section("PROFILI PAESI EU - HDAB POLICIES")

        print(f"\n{'Country':<5} {'Name':<14} {'eps_max':>8} {'HDAB':>5} "
              f"{'Opt-out':>8} {'Latency':>12} {'Purposes':>4}")
        print("-" * 70)

        for cc, p in sorted(EU_COUNTRY_PROFILES.items(), key=lambda x: x[1].hdab_strictness, reverse=True):
            stars = "*" * p.hdab_strictness
            latency_str = f"{p.latency_ms[0]}-{p.latency_ms[1]}ms"
            print(f"  {cc:<4} {p.name:<14} {p.dp_epsilon_max:>7.1f} {stars:>5} "
                  f"{p.opt_out_rate:>7.0%} {latency_str:>12} {len(p.allowed_purposes):>4}")

        print()
        print(f"{Style.MUTED}HDAB strictness: * = low ... ***** = highest{Colors.RESET}")
        print(f"{Style.MUTED}eps_max: National max allowed epsilon (lower = stricter privacy){Colors.RESET}")
        print(f"{Style.MUTED}Purposes: Number of EHDS Art.53 purposes allowed{Colors.RESET}")

        input(f"\n{Style.MUTED}Premi Enter per continuare...{Colors.RESET}")

    def _select_scenario(self):
        """Select a pre-configured scenario."""
        clear_screen()
        print_section("SCENARI PRE-CONFIGURATI")

        for i, (name, scenario) in enumerate(SCENARIOS.items(), 1):
            countries_str = ", ".join(scenario["countries"])
            print(f"\n  {Style.TITLE}{i}{Colors.RESET} - {name}")
            print(f"      Paesi: {countries_str}")
            print(f"      {Style.MUTED}{scenario['description']}{Colors.RESET}")

        choice = get_choice("\nScegli scenario", [str(i+1) for i in range(len(SCENARIOS))] + ["0"], default="3")

        if choice == "0":
            return

        scenario_name = list(SCENARIOS.keys())[int(choice) - 1]
        scenario = SCENARIOS[scenario_name]
        self.config["countries"] = scenario["countries"]

        print_success(f"\nScenario selezionato: {scenario_name}")
        print(f"  Paesi: {', '.join(scenario['countries'])}")
        print(f"  Ospedali totali: {len(scenario['countries']) * self.config['hospitals_per_country']}")

        input(f"\n{Style.MUTED}Premi Enter per continuare...{Colors.RESET}")

    def _configure(self):
        """Interactive configuration."""
        clear_screen()
        print_section("CONFIGURAZIONE CROSS-BORDER")

        # Country selection
        print_subsection("Selezione Paesi")
        available = list(EU_COUNTRY_PROFILES.keys())
        print(f"  Disponibili: {', '.join(available)}")
        print(f"  Attuali: {', '.join(self.config['countries'])}")

        if confirm("Modificare la selezione paesi?", default=False):
            countries_input = input(f"  Inserisci codici paesi separati da virgola (es. DE,FR,IT): ").strip()
            if countries_input:
                selected = [c.strip().upper() for c in countries_input.split(",")]
                valid = [c for c in selected if c in EU_COUNTRY_PROFILES]
                if valid:
                    self.config["countries"] = valid
                    print_success(f"  Selezionati: {', '.join(valid)}")
                else:
                    print_warning("  Nessun paese valido, mantengo selezione attuale")

        self.config["hospitals_per_country"] = get_int(
            "Ospedali per paese",
            default=self.config["hospitals_per_country"],
            min_val=1, max_val=3,
        )

        # EHDS Purpose
        print_subsection("Scopo EHDS (Art. 53)")
        for i, p in enumerate(EHDS_PURPOSES, 1):
            print(f"  {i}. {p}")
        purpose_idx = get_int("Seleziona scopo", default=1, min_val=1, max_val=len(EHDS_PURPOSES))
        self.config["purpose"] = EHDS_PURPOSES[purpose_idx - 1]

        # Algorithm
        print_subsection("Algoritmo FL")
        for i, a in enumerate(FL_ALGORITHMS, 1):
            print(f"  {i:2d}. {a}")
        algo_idx = get_int("Seleziona algoritmo", default=1, min_val=1, max_val=len(FL_ALGORITHMS))
        self.config["algorithm"] = FL_ALGORITHMS[algo_idx - 1]

        # Training params
        print_subsection("Parametri Training")
        self.config["num_rounds"] = get_int("Numero round", default=self.config["num_rounds"], min_val=5, max_val=100)
        self.config["local_epochs"] = get_int("Epoche locali per round", default=self.config["local_epochs"], min_val=1, max_val=10)
        self.config["learning_rate"] = get_float("Learning rate", default=self.config["learning_rate"], min_val=0.0001, max_val=1.0)

        # Privacy
        print_subsection("Privacy Budget Globale")
        print_info("Il budget effettivo sara min(globale, max nazionale)")
        self.config["global_epsilon"] = get_float(
            "Epsilon globale",
            default=self.config["global_epsilon"],
            min_val=0.1, max_val=100.0
        )

        # Latency
        self.config["simulate_latency"] = confirm("Simulare latenza di rete?", default=True)

        # Dataset
        print_subsection("Dataset")
        self.config["dataset_type"] = "synthetic"
        try:
            from terminal.screens.datasets import DatasetManager
            manager = DatasetManager()
            datasets = {}
            for name, ds in manager.datasets.items():
                if ds.type == "imaging":
                    datasets[name] = str(ds.path)
            if datasets and confirm("Usare dataset imaging reale?", default=False):
                names = list(datasets.keys())
                for i, n in enumerate(names, 1):
                    print(f"  {i}. {n}")
                ds_idx = get_int("Dataset", default=1, min_val=1, max_val=len(names))
                self.config["dataset_type"] = "imaging"
                self.config["dataset_path"] = datasets[names[ds_idx - 1]]
                self.config["img_size"] = get_int("Image size", default=64, min_val=32, max_val=224)
        except (ImportError, Exception):
            pass

        self.config["seed"] = get_int("Random seed", default=self.config["seed"], min_val=0, max_val=99999)

        # Show summary
        self._show_config_summary()
        input(f"\n{Style.MUTED}Premi Enter per continuare...{Colors.RESET}")

    def _show_config_summary(self):
        """Display configuration summary."""
        print_subsection("RIEPILOGO CONFIGURAZIONE CROSS-BORDER")
        c = self.config
        total_hospitals = len(c["countries"]) * c["hospitals_per_country"]

        print(f"\n  {'Paesi':<22} {', '.join(c['countries'])}")
        print(f"  {'Ospedali/paese':<22} {c['hospitals_per_country']}")
        print(f"  {'Ospedali totali':<22} {total_hospitals}")
        print(f"  {'Scopo EHDS':<22} {c['purpose']}")
        print(f"  {'Algoritmo':<22} {c['algorithm']}")
        print(f"  {'Round':<22} {c['num_rounds']}")
        print(f"  {'Epoche locali':<22} {c['local_epochs']}")
        print(f"  {'Learning rate':<22} {c['learning_rate']}")
        print(f"  {'Epsilon globale':<22} {c['global_epsilon']}")
        print(f"  {'Latenza simulata':<22} {'Si' if c['simulate_latency'] else 'No'}")
        print(f"  {'Dataset':<22} {c['dataset_type']}")

        # Show per-country effective epsilon
        print_subsection("EPSILON EFFETTIVO PER GIURISDIZIONE")
        for cc in c["countries"]:
            p = EU_COUNTRY_PROFILES[cc]
            eff_eps = min(c["global_epsilon"], p.dp_epsilon_max)
            status = "OK" if eff_eps == c["global_epsilon"] else f"limitato a {eff_eps:.1f}"
            stars = "*" * p.hdab_strictness
            print(f"  {cc} ({p.name:<14}) eps_max={p.dp_epsilon_max:<5.1f} effettivo={eff_eps:<5.1f} HDAB={stars:<5} [{status}]")

    def _run_simulation(self):
        """Execute cross-border simulation."""
        clear_screen()
        print_section("ESECUZIONE CROSS-BORDER FL")

        self._show_config_summary()
        print()

        if not confirm("Avviare la simulazione cross-border?", default=True):
            return

        print()

        c = self.config

        # Check purpose violations before starting
        for cc in c["countries"]:
            p = EU_COUNTRY_PROFILES[cc]
            if c["purpose"] not in p.allowed_purposes:
                print_warning(f"  HDAB {p.name} ({cc}): scopo '{c['purpose']}' NON ammesso!")

        if not confirm("\nProcedere comunque?", default=True):
            return

        def progress_callback(round_num, total_rounds, result):
            eps_str = f"eps={result.per_hospital[0]['epsilon_spent']:.3f}" if result.per_hospital else ""
            status = result.compliance_status.upper()
            status_color = Style.SUCCESS if status == "COMPLIANT" else Style.WARNING
            print(f"  Round {round_num+1:3d}/{total_rounds} | "
                  f"Acc={result.global_acc:.2%} | F1={result.global_f1:.3f} | "
                  f"Loss={result.global_loss:.4f} | {eps_str} | "
                  f"Latency={result.latency_overhead_ms:.0f}ms | "
                  f"{status_color}[{status}]{Colors.RESET}")

        try:
            print_info("Inizializzazione trainer cross-border...")
            trainer = CrossBorderFederatedTrainer(
                countries=c["countries"],
                hospitals_per_country=c["hospitals_per_country"],
                algorithm=c["algorithm"],
                num_rounds=c["num_rounds"],
                local_epochs=c["local_epochs"],
                batch_size=c["batch_size"],
                learning_rate=c["learning_rate"],
                global_epsilon=c["global_epsilon"],
                purpose=c["purpose"],
                dataset_type=c["dataset_type"],
                dataset_path=c.get("dataset_path"),
                img_size=c.get("img_size", 128),
                seed=c["seed"],
                simulate_latency=c["simulate_latency"],
                mu=c.get("mu", 0.1),
                server_lr=c.get("server_lr", 0.1),
                beta1=c.get("beta1", 0.9),
                beta2=c.get("beta2", 0.99),
                tau=c.get("tau", 1e-3),
                progress_callback=progress_callback,
            )

            # Show hospital mapping
            print_subsection("MAPPING OSPEDALI")
            for h in trainer.hospitals:
                p = h.country_profile
                print(f"  [{h.hospital_id}] {h.name:<30} {h.country_code} | "
                      f"eps_eff={h.effective_epsilon:.1f} | "
                      f"HDAB={'*'*p.hdab_strictness}")

            print()
            print_info(f"Avvio training {c['algorithm']} su {len(trainer.hospitals)} ospedali in {len(c['countries'])} paesi...")
            print()

            # Run
            results = trainer.train()

            # Display results
            self._display_results(trainer, results)

            # Auto-save
            self._auto_save(trainer, results)

        except ImportError as e:
            print_error(f"Errore import: {e}")
            print_info("Assicurarsi che PyTorch sia installato")
        except Exception as e:
            print_error(f"Errore: {e}")
            import traceback
            traceback.print_exc()

        input(f"\n{Style.MUTED}Premi Enter per continuare...{Colors.RESET}")

    def _display_results(self, trainer, results):
        """Display cross-border results."""
        print()
        print_section("RISULTATI CROSS-BORDER")

        # Final metrics
        if trainer.history:
            last = trainer.history[-1]
            print_subsection("METRICHE FINALI")
            print(f"  {'Accuracy':<18} {Style.HIGHLIGHT}{last.global_acc:.2%}{Colors.RESET}")
            print(f"  {'F1 Score':<18} {last.global_f1:.4f}")
            print(f"  {'AUC-ROC':<18} {last.global_auc:.4f}")
            print(f"  {'Loss':<18} {last.global_loss:.4f}")

        # Per-country analysis
        print_subsection("ANALISI PER GIURISDIZIONE")
        country_summary = trainer.audit_log.summary_by_country()
        print(f"\n  {'Country':<6} {'HDAB':>5} {'eps_max':>8} {'eps_used':>9} "
              f"{'Samples':>8} {'Opt-out':>8} {'Latency':>9} {'Status':>10}")
        print("  " + "-" * 70)

        for cc in trainer.countries:
            if cc in country_summary:
                data = country_summary[cc]
                p = EU_COUNTRY_PROFILES[cc]
                eps_max = p.dp_epsilon_max
                eps_used = data["epsilon_spent"]
                compliant = eps_used <= eps_max
                status_str = f"{Style.SUCCESS}OK{Colors.RESET}" if compliant else f"{Style.ERROR}OVER{Colors.RESET}"
                avg_lat = data["total_latency_ms"] / max(data["total_rounds"], 1)
                print(f"  {cc:<6} {'*'*p.hdab_strictness:>5} {eps_max:>8.1f} "
                      f"{eps_used:>9.3f} {data['total_samples_used']:>8} "
                      f"{data['total_opted_out']:>8} {avg_lat:>8.1f}ms {status_str:>10}")

        # Purpose violations
        if results["purpose_violations"]:
            print_subsection("VIOLAZIONI SCOPO HDAB")
            for v in results["purpose_violations"]:
                print(f"  {Style.WARNING}{v}{Colors.RESET}")

        # Compliance summary
        violations = trainer.audit_log.get_violations()
        print_subsection("COMPLIANCE SUMMARY")
        print(f"  Audit entries: {len(trainer.audit_log.entries)}")
        print(f"  Violations: {len(violations)}")
        print(f"  Effective epsilon: {results['effective_epsilon']:.1f}")
        print(f"  Total time: {results['total_time']:.1f}s")

    def _auto_save(self, trainer, results):
        """Auto-save all outputs."""
        base_dir = Path(__file__).parent.parent.parent / "results" / "cross_border"
        base_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        countries_str = "_".join(self.config["countries"][:5])
        algo = self.config["algorithm"]
        folder_name = f"cb_{countries_str}_{algo}_{self.config['num_rounds']}r_{timestamp}"
        output_dir = base_dir / folder_name

        try:
            trainer.save_results(str(output_dir))
            print_success(f"\nRisultati salvati in: results/cross_border/{folder_name}/")
            print(f"  - audit_trail.csv ({len(trainer.audit_log.entries)} entries)")
            print(f"  - summary_by_country.csv")
            print(f"  - history_cross_border.csv")
            print(f"  - hospitals.csv")
            print(f"  - table_cross_border.tex")
            print(f"  - results.json")
            print(f"  - summary.txt")
            print(f"  - plot_convergence.png")
            print(f"  - plot_epsilon_by_country.png")
            print(f"  - plot_hdab_vs_optout.png")
            print(f"  - plot_latency_per_hospital.png")
        except Exception as e:
            print_error(f"Errore salvataggio: {e}")

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
            # Jurisdiction privacy
            "jurisdiction_privacy_enabled": False,
            "noise_strategy": "global",
            "hospital_allocation_fraction": 1.0,
            "min_active_clients": 2,
            "country_overrides": {},
            # IHE Integration
            "ihe_enabled": False,
            "ihe_config": {},
            # Data Quality Framework (EHDS Art. 69)
            "data_quality_enabled": False,
            "data_quality_config": {},
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
            print(f"  {Style.TITLE}5{Colors.RESET} - Simula opt-out paese (Art. 48)")
            print(f"  {Style.TITLE}6{Colors.RESET} - Report IHE Compliance")
            print(f"  {Style.TITLE}0{Colors.RESET} - Torna al menu principale")

            choice = get_choice("\nScegli opzione", ["1", "2", "3", "4", "5", "6", "0"], default="1")

            if choice == "1":
                self._configure()
            elif choice == "2":
                self._select_scenario()
            elif choice == "3":
                self._show_country_profiles()
            elif choice == "4":
                self._run_simulation()
            elif choice == "5":
                self._simulate_optout()
            elif choice == "6":
                self._show_ihe_report()
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

        # Jurisdiction privacy
        print_subsection("Privacy per Giurisdizione (HDAB)")
        self.config["jurisdiction_privacy_enabled"] = confirm(
            "Abilitare privacy budget per giurisdizione?", default=False
        )
        if self.config["jurisdiction_privacy_enabled"]:
            print_info("Ogni paese ha un epsilon massimo nazionale (HDAB ceiling).")
            print_info("I client di quel paese non possono superare il ceiling.")
            print_info("Quando esauriscono il budget, smettono di partecipare.\n")

            # Show national ceilings
            for cc in self.config["countries"]:
                p = EU_COUNTRY_PROFILES[cc]
                print(f"  {cc} ({p.name}): eps_max = {p.dp_epsilon_max}")

            if confirm("\nModificare i ceiling nazionali?", default=False):
                overrides = {}
                for cc in self.config["countries"]:
                    p = EU_COUNTRY_PROFILES[cc]
                    new_eps = get_float(
                        f"  {cc} ({p.name}) epsilon max",
                        default=p.dp_epsilon_max, min_val=0.1, max_val=100.0
                    )
                    if new_eps != p.dp_epsilon_max:
                        overrides[cc] = {
                            "epsilon_max": new_eps,
                            "delta": p.dp_delta_max,
                        }
                self.config["country_overrides"] = overrides

            self.config["min_active_clients"] = get_int(
                "Min client attivi per continuare",
                default=2, min_val=1, max_val=10
            )

        # Latency
        self.config["simulate_latency"] = confirm("Simulare latenza di rete?", default=True)

        # IHE Integration Profiles
        print_subsection("IHE Integration Profiles")
        self.config["ihe_enabled"] = confirm(
            "Abilitare profili IHE per FL?", default=False
        )
        if self.config["ihe_enabled"]:
            print_info("Profili IHE: ATNA (audit), XDS-I.b (imaging),")
            print_info("CT (time sync), mTLS (security), XUA (auth), BPPC (consent)\n")

            ihe_cfg = {}
            ihe_cfg["atna_audit"] = confirm("  ATNA audit trail?", default=True)
            ihe_cfg["xds_imaging_simulation"] = confirm("  XDS-I.b imaging retrieve?", default=True)
            ihe_cfg["consistent_time"] = confirm("  CT time synchronization?", default=True)
            ihe_cfg["mtls_simulation"] = confirm("  mTLS security?", default=True)

            if ihe_cfg.get("consistent_time", True):
                ihe_cfg["max_clock_drift_ms"] = get_float(
                    "  Max clock drift (ms)", default=50.0, min_val=1.0, max_val=500.0
                )

            self.config["ihe_config"] = ihe_cfg

        # Data Quality Framework (EHDS Art. 69)
        print_subsection("Data Quality Framework (Art. 69 EHDS)")
        self.config["data_quality_enabled"] = confirm(
            "Abilitare quality-weighted aggregation?", default=False
        )
        if self.config["data_quality_enabled"]:
            print_info("I pesi di aggregazione saranno moltiplicati per il quality score.")
            print_info("Formula: w_h = (n_h/N) * quality_h^alpha, poi normalizzato.\n")

            dq_cfg = {}
            dq_cfg["alpha"] = get_float(
                "  Alpha (0=ignora qualita, 1=lineare, 2=forte penalita)",
                default=1.0, min_val=0.0, max_val=3.0
            )

            print_info("\nSoglie quality label EHDS Art. 69:")
            print_info("  GOLD >= 0.85, SILVER >= 0.70, BRONZE >= 0.55, INSUFFICIENT < 0.55")

            if confirm("\nAbilitare anomaly detection pre-training?", default=True):
                dq_cfg["ks_threshold"] = 0.05
                dq_cfg["missing_threshold"] = 0.3
                dq_cfg["entropy_threshold"] = 0.3
                dq_cfg["iqr_multiplier"] = 3.0

            self.config["data_quality_config"] = dq_cfg

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
        print(f"  {'IHE Profiles':<22} {'Si' if c.get('ihe_enabled') else 'No'}")
        dq_enabled = c.get('data_quality_enabled', False)
        dq_alpha = c.get('data_quality_config', {}).get('alpha', 1.0) if dq_enabled else '-'
        print(f"  {'Data Quality':<22} {'Si (alpha=' + str(dq_alpha) + ')' if dq_enabled else 'No'}")

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
                jurisdiction_privacy_enabled=c.get("jurisdiction_privacy_enabled", False),
                country_overrides=c.get("country_overrides", {}),
                hospital_allocation_fraction=c.get("hospital_allocation_fraction", 1.0),
                noise_strategy=c.get("noise_strategy", "global"),
                min_active_clients=c.get("min_active_clients", 2),
                ihe_enabled=c.get("ihe_enabled", False),
                ihe_config=c.get("ihe_config", {}),
                data_quality_enabled=c.get("data_quality_enabled", False),
                data_quality_config=c.get("data_quality_config", {}),
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

        # Jurisdiction privacy status
        if hasattr(trainer, 'jurisdiction_manager') and trainer.jurisdiction_manager:
            print_subsection("BUDGET PRIVACY PER GIURISDIZIONE (RDP)")
            jstatus = trainer.jurisdiction_manager.get_jurisdiction_status()
            print(f"\n  {'Country':<6} {'Ceiling':>8} {'Spent':>10} "
                  f"{'Active':>8} {'Exhausted':>10} {'Opt-out':>8}")
            print("  " + "-" * 55)
            for cc, info in sorted(jstatus.items()):
                print(f"  {cc:<6} {info['epsilon_ceiling']:>8.1f} "
                      f"{info['epsilon_spent_max']:>10.4f} "
                      f"{info['active']}/{info['total']:>5} "
                      f"{info['exhausted']:>10} {info['opted_out']:>8}")

            timeline = trainer.jurisdiction_manager.get_dropout_timeline()
            if timeline:
                print_subsection("TIMELINE DROPOUT CLIENT")
                for event in timeline:
                    reason = event['reason'].replace('_', ' ')
                    print(f"  Round {event['round']:3d}: {event['hospital']:<30} "
                          f"({event['country']}) - {reason}")

        # IHE Compliance section
        if hasattr(trainer, 'ihe_bridge') and trainer.ihe_bridge:
            print_subsection("IHE COMPLIANCE REPORT")
            summary = trainer.ihe_bridge.get_audit_summary()
            print(f"  ATNA Audit Events:     {summary['total_events']}")
            print(f"  CT Sync Rounds:        {summary['ct_syncs']}")
            print(f"  XDS-I.b Retrieves:     {summary['xds_retrieves']}")
            print(f"  mTLS Certificates:     {summary['certificates_valid']}/{summary['certificates_total']}")
            print(f"  Max Clock Drift:       {summary['max_drift_ms']:.1f} ms")

            if summary.get('events_by_type'):
                print()
                for evt_type, count in sorted(summary['events_by_type'].items()):
                    print(f"    {evt_type:<30} {count:>4}")

        # Data Quality section (EHDS Art. 69)
        if hasattr(trainer, 'quality_manager') and trainer.quality_manager:
            print_subsection("DATA QUALITY (EHDS Art. 69)")
            print(f"\n  {'Hospital':<30} {'Score':>6} {'Label':>12} {'Weight':>7} "
                  f"{'Comp':>5} {'Acc':>5} {'Uniq':>5} {'Div':>5} {'Cons':>5} {'Anomaly':>8}")
            print("  " + "-" * 100)

            for cid, report in sorted(trainer.quality_manager.client_reports.items()):
                label_color = {
                    "gold": Style.SUCCESS,
                    "silver": Style.HIGHLIGHT,
                    "bronze": Style.WARNING,
                    "insufficient": Style.ERROR,
                }.get(report.quality_label.value, "")

                anomaly_str = "YES" if report.is_anomalous else "no"
                anomaly_color = Style.WARNING if report.is_anomalous else ""

                print(f"  {report.hospital_name:<30} "
                      f"{report.overall_score:>5.3f} "
                      f"{label_color}{report.quality_label.value:>12}{Colors.RESET} "
                      f"{report.quality_weight:>6.3f} "
                      f"{report.completeness:>5.2f} {report.accuracy:>5.2f} "
                      f"{report.uniqueness:>5.2f} {report.diversity:>5.2f} "
                      f"{report.consistency:>5.2f} "
                      f"{anomaly_color}{anomaly_str:>8}{Colors.RESET}")

            summary = trainer.quality_manager.get_display_summary()
            print(f"\n  Mean quality: {summary['mean_quality']:.3f} +/- {summary['std_quality']:.3f}")
            print(f"  Anomalous clients: {summary['anomalous_clients']}/{summary['total_clients']}")
            labels = summary.get("label_distribution", {})
            if labels:
                label_str = ", ".join(f"{k}: {v}" for k, v in sorted(labels.items()))
                print(f"  Label distribution: {label_str}")
            print(f"  Alpha (quality exponent): {summary['alpha']}")

            # Show anomaly details
            anomalous = [r for r in trainer.quality_manager.client_reports.values() if r.is_anomalous]
            if anomalous:
                print_subsection("ANOMALIE RILEVATE (Pre-Training)")
                for report in anomalous:
                    for a in report.anomalies:
                        print(f"  {Style.WARNING}{report.hospital_name}: {a}{Colors.RESET}")

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
            if hasattr(trainer, 'ihe_bridge') and trainer.ihe_bridge:
                print(f"  - ihe_compliance_report.json")
                print(f"  - atna_audit_trail.json")
                print(f"  - atna_audit_trail.xml")
            if hasattr(trainer, 'quality_manager') and trainer.quality_manager:
                print(f"  - quality_report.json")
                print(f"  - quality_labels.csv")
        except Exception as e:
            print_error(f"Errore salvataggio: {e}")

    def _simulate_optout(self):
        """Simulate Art. 48 EHDS country opt-out during training."""
        clear_screen()
        print_section("SIMULAZIONE OPT-OUT PAESE (Art. 48 EHDS)")

        print_info("Simula cosa succede quando un intero paese si ritira")
        print_info("dal training federato ad un determinato round.\n")

        # Select country to opt out
        print_subsection("Seleziona paese da escludere")
        for i, cc in enumerate(self.config["countries"], 1):
            p = EU_COUNTRY_PROFILES[cc]
            print(f"  {i}. {cc} ({p.name}) - eps_max={p.dp_epsilon_max}, "
                  f"HDAB={'*'*p.hdab_strictness}")

        country_idx = get_int("Paese", default=1,
                              min_val=1, max_val=len(self.config["countries"]))
        optout_country = self.config["countries"][country_idx - 1]

        # Select opt-out round
        optout_round = get_int(
            f"Round di opt-out per {optout_country}",
            default=self.config["num_rounds"] // 2,
            min_val=1, max_val=self.config["num_rounds"] - 1,
        )

        print_info(f"\nSimulazione: {optout_country} esce al round {optout_round}")
        print_info("Jurisdiction privacy verra abilitato automaticamente.\n")

        if not confirm("Procedere con la simulazione?", default=True):
            return

        c = self.config.copy()
        c["jurisdiction_privacy_enabled"] = True

        def progress_callback(round_num, total_rounds, result):
            active_str = ""
            if hasattr(trainer, 'jurisdiction_manager') and trainer.jurisdiction_manager:
                active = trainer.jurisdiction_manager.get_active_clients()
                active_str = f" | Attivi={len(active)}/{len(trainer.hospitals)}"
            print(f"  Round {round_num+1:3d}/{total_rounds} | "
                  f"Acc={result.global_acc:.2%} | F1={result.global_f1:.3f} | "
                  f"Loss={result.global_loss:.4f}{active_str}")

        try:
            trainer = CrossBorderFederatedTrainer(
                countries=c["countries"],
                hospitals_per_country=c["hospitals_per_country"],
                algorithm=c["algorithm"],
                num_rounds=c["num_rounds"],
                local_epochs=c["local_epochs"],
                batch_size=c.get("batch_size", 32),
                learning_rate=c["learning_rate"],
                global_epsilon=c["global_epsilon"],
                purpose=c["purpose"],
                dataset_type=c.get("dataset_type", "synthetic"),
                dataset_path=c.get("dataset_path"),
                img_size=c.get("img_size", 128),
                seed=c["seed"],
                simulate_latency=c.get("simulate_latency", True),
                mu=c.get("mu", 0.1),
                server_lr=c.get("server_lr", 0.1),
                beta1=c.get("beta1", 0.9),
                beta2=c.get("beta2", 0.99),
                tau=c.get("tau", 1e-3),
                progress_callback=progress_callback,
                jurisdiction_privacy_enabled=True,
                country_overrides=c.get("country_overrides", {}),
                min_active_clients=c.get("min_active_clients", 2),
                ihe_enabled=c.get("ihe_enabled", False),
                ihe_config=c.get("ihe_config", {}),
                data_quality_enabled=c.get("data_quality_enabled", False),
                data_quality_config=c.get("data_quality_config", {}),
            )

            # Schedule opt-out: will be triggered during training
            # We need to hook into the training loop
            # Override the progress callback to trigger opt-out at the right round
            original_train = trainer.train

            def train_with_optout():
                result = original_train()
                return result

            # Instead, we trigger opt-out before training starts by modifying
            # the jurisdiction manager after it's created during train()
            # Schedule the opt-out in the trainer
            trainer.optout_schedule = {optout_country: optout_round}

            print_info(f"Avvio training con opt-out di {optout_country} al round {optout_round}...")
            print()

            # Run training (opt-out will trigger at scheduled round)
            results = trainer.train()

            # Display results
            print()
            self._display_results(trainer, results)

            # Show opt-out impact
            print_subsection(f"IMPATTO OPT-OUT {optout_country} (ROUND {optout_round})")
            if trainer.history:
                # Find metrics before and after opt-out
                pre_optout = [r for r in trainer.history if r.round_num < optout_round]
                post_optout = [r for r in trainer.history if r.round_num >= optout_round]

                if pre_optout and post_optout:
                    pre_acc = pre_optout[-1].global_acc
                    post_acc = post_optout[-1].global_acc
                    delta_acc = post_acc - pre_acc
                    sign = "+" if delta_acc >= 0 else ""
                    print(f"  Accuracy pre-optout (round {optout_round-1}):  {pre_acc:.2%}")
                    print(f"  Accuracy post-optout (round {len(trainer.history)-1}): {post_acc:.2%}")
                    print(f"  Delta: {sign}{delta_acc:.2%}")

            # Auto-save
            self._auto_save(trainer, results)

        except Exception as e:
            print_error(f"Errore: {e}")
            import traceback
            traceback.print_exc()

        input(f"\n{Style.MUTED}Premi Enter per continuare...{Colors.RESET}")

    def _show_ihe_report(self):
        """Display IHE Integration Profiles mapping and compliance info."""
        clear_screen()
        print_section("IHE INTEGRATION PROFILES PER FL")

        print_info("Mapping tra profili IHE standard e operazioni Federated Learning\n")

        # Academic mapping table
        print_subsection("MAPPING IHE PROFILE -> FL OPERATION")
        print(f"\n  {'Profile':<10} {'FL Operation':<45} {'EHDS Art.':<10}")
        print("  " + "-" * 68)
        mapping = [
            ("ATNA", "Node authentication + audit per FL round", "Art. 50"),
            ("XDS-I.b", "DICOM study retrieve before local training", "Art. 12"),
            ("CT", "NTP sync for round ordering across countries", "Art. 50"),
            ("XUA", "Researcher SAML assertion per FL session", "Art. 46"),
            ("BPPC", "Patient consent for secondary use training", "Art. 33"),
            ("mTLS", "Mutual TLS between FL nodes and aggregator", "Art. 50"),
            ("PIXm/PDQm", "Patient ID cross-referencing (optional)", "Art. 12"),
            ("XCA", "Cross-community document query", "Art. 12"),
        ]
        for profile, operation, article in mapping:
            print(f"  {profile:<10} {operation:<45} {article:<10}")

        # Implementation status
        print_subsection("STATO IMPLEMENTAZIONE")
        print(f"  {'Profile':<10} {'Classe':<35} {'File':<35}")
        print("  " + "-" * 78)
        impl = [
            ("ATNA", "ATNAAuditLogger", "core/ihe_profiles.py"),
            ("XDS-I.b", "XDSImagingSimulator", "governance/ihe_fl_bridge.py"),
            ("CT", "ConsistentTimeSynchronizer", "governance/ihe_fl_bridge.py"),
            ("XUA", "XUASecurityContext", "core/ihe_profiles.py"),
            ("BPPC", "BPPCConsentManager", "core/ihe_profiles.py"),
            ("mTLS", "NodeCertificate", "governance/ihe_fl_bridge.py"),
            ("PIXm/PDQm", "PIXPDQManager", "core/ihe_profiles.py"),
            ("XCA", "XCAGateway", "core/ihe_profiles.py"),
        ]
        for profile, cls, filepath in impl:
            print(f"  {profile:<10} {cls:<35} {filepath:<35}")

        print_subsection("COME USARE")
        print_info("1. Menu 1 (Configura) -> Abilita 'IHE Integration Profiles'")
        print_info("2. Menu 4 (Esegui simulazione) -> I profili IHE vengono attivati")
        print_info("3. I risultati includono: ihe_compliance_report.json,")
        print_info("   atna_audit_trail.json, atna_audit_trail.xml")
        print_info("")
        print_info("Ogni round FL genera automaticamente:")
        print_info("  - ATNA: audit events per round start/end + model updates")
        print_info("  - CT: time sync tra tutti i nodi prima di ogni round")
        print_info("  - XDS-I.b: simulazione retrieve DICOM per ospedali imaging")
        print_info("  - mTLS: verifica validita certificati nodo")

        input(f"\n{Style.MUTED}Premi Enter per continuare...{Colors.RESET}")

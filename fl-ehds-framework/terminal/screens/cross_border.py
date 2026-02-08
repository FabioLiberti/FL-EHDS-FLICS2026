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
        self._last_trainer = None
        self._last_results = None
        self._last_compliance_report = None

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
            # MyHealth@EU / NCPeH Integration (EHDS Art. 5-12)
            "myhealth_eu_enabled": False,
            "myhealth_eu_config": {},
            # Governance Lifecycle (EHDS Chapter IV, Art. 33-44)
            "governance_lifecycle_enabled": False,
            "governance_config": {},
            # Secure Processing Environment (EHDS Art. 50)
            "secure_processing_enabled": False,
            "secure_processing_config": {},
            # Fee Model (EHDS Art. 42)
            "fee_model_enabled": False,
            "fee_model_config": {},
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
            print(f"  {Style.TITLE}7{Colors.RESET} - MyHealth@EU NCPeH Topology")
            print(f"  {Style.TITLE}8{Colors.RESET} - EHDS Compliance Report")
            print(f"  {Style.TITLE}9{Colors.RESET} - Governance Lifecycle (Art. 33-44)")
            print(f"  {Style.TITLE}A{Colors.RESET} - EHDS Benchmark (Privacy-Utility-Cost)")
            print(f"  {Style.TITLE}0{Colors.RESET} - Torna al menu principale")

            choice = get_choice("\nScegli opzione", ["1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "a", "0"], default="1")

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
            elif choice == "7":
                self._show_ncpeh_topology()
            elif choice == "8":
                self._show_ehds_compliance()
            elif choice == "9":
                self._show_governance_info()
            elif choice.upper() == "A":
                self._run_ehds_benchmark()
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

        # MyHealth@EU / NCPeH Integration (EHDS Art. 5-12)
        print_subsection("MyHealth@EU / NCPeH Integration (Art. 5-12 EHDS)")
        self.config["myhealth_eu_enabled"] = confirm(
            "Abilitare aggregazione gerarchica via NCPeH?", default=False
        )
        if self.config["myhealth_eu_enabled"]:
            print_info("Aggregazione gerarchica a 2 livelli:")
            print_info("  Livello 1: Ospedali -> NCP nazionale")
            print_info("  Livello 2: NCP -> Aggregatore EU\n")

            mheu_cfg = {}
            mheu_cfg["hierarchical_aggregation"] = True

            weight_choice = get_int(
                "  Strategia pesi NCP (1=sample_proportional, 2=equal)",
                default=1, min_val=1, max_val=2
            )
            mheu_cfg["ncp_weight_strategy"] = (
                "sample_proportional" if weight_choice == 1 else "equal"
            )

            mheu_cfg["simulate_ncp_latency"] = confirm(
                "  Simulare latenza inter-NCP?", default=True
            )
            mheu_cfg["patient_summary_enabled"] = confirm(
                "  Simulare Patient Summary exchange?", default=True
            )
            mheu_cfg["eprescription_enabled"] = confirm(
                "  Simulare ePrescription cross-border?", default=False
            )
            mheu_cfg["track_communication_cost"] = True
            mheu_cfg["central_node"] = "BE"

            self.config["myhealth_eu_config"] = mheu_cfg

        # Governance Lifecycle (EHDS Chapter IV, Art. 33-44)
        print_subsection("Governance Layer (EHDS Chapter IV, Art. 33-44)")
        self.config["governance_lifecycle_enabled"] = confirm(
            "Abilitare HDAB + Permit Lifecycle + Data Minimization?",
            default=False,
        )
        if self.config["governance_lifecycle_enabled"]:
            print_info("Moduli attivati: HDAB simulation, Permit validation,")
            print_info("                 Purpose limitation, Audit trail GDPR Art. 30")

            gov_cfg = self.config.get("governance_config", {})

            # Data Minimization (Art. 44)
            gov_cfg["data_minimization_enabled"] = confirm(
                "  Abilitare Data Minimization (Art. 44)?", default=True
            )
            if gov_cfg["data_minimization_enabled"]:
                print_info("  Rimuove features non necessarie per lo scopo dichiarato.")
                gov_cfg["importance_threshold"] = get_float(
                    "  Soglia importanza MI", default=0.01, min_val=0.0, max_val=0.5,
                )

            self.config["governance_config"] = gov_cfg

        # Secure Processing Environment (EHDS Art. 50)
        print_subsection("Secure Processing Environment (EHDS Art. 50)")
        self.config["secure_processing_enabled"] = confirm(
            "Abilitare Secure Processing (Enclave + Watermark + Time-limit)?",
            default=False,
        )
        if self.config["secure_processing_enabled"]:
            sp_cfg = self.config.get("secure_processing_config", {})
            sp_cfg["enclave_enabled"] = confirm(
                "  Enclave simulation (TEE)?", default=True
            )
            sp_cfg["watermarking_enabled"] = confirm(
                "  Model watermarking (Art. 37)?", default=True
            )
            sp_cfg["time_limited_enabled"] = confirm(
                "  Time-limited access?", default=True
            )
            if sp_cfg["time_limited_enabled"]:
                sp_cfg["permit_duration_hours"] = get_float(
                    "  Durata permit (ore)",
                    default=24.0, min_val=0.1, max_val=8760.0,
                )
            self.config["secure_processing_config"] = sp_cfg

        # Fee Model and Sustainability (EHDS Art. 42)
        print_subsection("Fee Model and Sustainability (EHDS Art. 42)")
        self.config["fee_model_enabled"] = confirm(
            "Abilitare Fee Model (costi HDAB + budget optimization)?",
            default=False,
        )
        if self.config["fee_model_enabled"]:
            fm_cfg = self.config.get("fee_model_config", {})
            fm_cfg["model_size_mb"] = get_float(
                "  Model update size (MB)", default=2.0, min_val=0.1, max_val=100.0
            )
            if confirm("  Impostare budget massimo?", default=False):
                fm_cfg["max_budget_eur"] = get_float(
                    "  Budget massimo (EUR)",
                    default=5000.0, min_val=100.0, max_val=1000000.0,
                )
                fm_cfg["enable_optimization"] = True
            else:
                fm_cfg["max_budget_eur"] = None
                fm_cfg["enable_optimization"] = False
            self.config["fee_model_config"] = fm_cfg

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
        mheu_enabled = c.get('myhealth_eu_enabled', False)
        if mheu_enabled:
            mheu_cfg = c.get('myhealth_eu_config', {})
            strategy = mheu_cfg.get('ncp_weight_strategy', 'sample_proportional')
            print(f"  {'MyHealth@EU':<22} Si (NCPeH, {strategy})")
        else:
            print(f"  {'MyHealth@EU':<22} No")
        if c.get("governance_lifecycle_enabled"):
            min_str = "SI" if c.get("governance_config", {}).get("data_minimization_enabled") else "NO"
            print(f"  {'Governance':<22} HDAB + Permits + Minimization={min_str}")
        else:
            print(f"  {'Governance':<22} No")
        if c.get("secure_processing_enabled"):
            sp = c.get("secure_processing_config", {})
            parts = []
            if sp.get("enclave_enabled", True):
                parts.append("Enclave")
            if sp.get("watermarking_enabled", True):
                parts.append("Watermark")
            if sp.get("time_limited_enabled", True):
                parts.append(f"TimeLimit={sp.get('permit_duration_hours', 24)}h")
            print(f"  {'Secure Processing':<22} {', '.join(parts)}")
        else:
            print(f"  {'Secure Processing':<22} No")

        if c.get("fee_model_enabled"):
            fm = c.get("fee_model_config", {})
            budget_str = f"{fm.get('max_budget_eur'):.0f} EUR" if fm.get("max_budget_eur") else "No limit"
            print(f"  {'Fee Model':<22} Budget={budget_str}, Model={fm.get('model_size_mb', 2.0)}MB")
        else:
            print(f"  {'Fee Model':<22} No")

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
                myhealth_eu_enabled=c.get("myhealth_eu_enabled", False),
                myhealth_eu_config=c.get("myhealth_eu_config", {}),
                governance_lifecycle_enabled=c.get("governance_lifecycle_enabled", False),
                governance_config=c.get("governance_config", {}),
                secure_processing_enabled=c.get("secure_processing_enabled", False),
                secure_processing_config=c.get("secure_processing_config", {}),
                fee_model_enabled=c.get("fee_model_enabled", False),
                fee_model_config=c.get("fee_model_config", {}),
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

            # Store for compliance report (menu 8)
            self._last_trainer = trainer
            self._last_results = results

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

        # MyHealth@EU / NCPeH results
        if hasattr(trainer, 'myhealth_bridge') and trainer.myhealth_bridge:
            print_subsection("MYHEALTH@EU - NCPeH HIERARCHICAL AGGREGATION")
            summary = trainer.myhealth_bridge.get_display_summary()

            # NCP table
            print(f"\n  {'NCP Node':<14} {'Country':<14} {'Hosp':>5} {'Samples':>8} "
                  f"{'NCP Lat':>8} {'Inter':>8} {'BW':>8} {'Tier':>5} {'Services'}")
            print(f"  {'-'*90}")
            for ncp in summary["ncp_rows"]:
                print(f"  {ncp['ncp_id']:<14} {ncp['country']:<14} "
                      f"{ncp['hospitals']:>5} {ncp['samples']:>8} "
                      f"{ncp['ncp_latency_ms']:>6.1f}ms "
                      f"{ncp['inter_ncp_ms']:>6.1f}ms "
                      f"{ncp['bandwidth_mbps']:>5.0f}Mb "
                      f"{ncp['tier']:>5} {ncp['services']}")
            print(f"  {'-'*90}")

            print(f"\n  Aggregation:  {summary['aggregation_type']}")
            print(f"  Central Node: {summary['central_node']} (Brussels)")
            if summary['total_flat_kb'] > 0:
                print(f"  Comm cost:    {summary['total_hier_kb']:.1f} KB "
                      f"(vs {summary['total_flat_kb']:.1f} KB flat = "
                      f"-{summary['saving_pct']:.1f}%)")
            if summary['ps_total'] > 0:
                print(f"  Patient Summaries exchanged: {summary['ps_total']}")
            if summary['ep_total'] > 0:
                print(f"  ePrescriptions processed: {summary['ep_total']}")

        # Governance Lifecycle (EHDS Ch. IV)
        if hasattr(trainer, 'governance_bridge') and trainer.governance_bridge:
            print_subsection("GOVERNANCE LIFECYCLE (EHDS Ch. IV)")

            budget = trainer.governance_bridge.get_budget_status()
            print(f"  Privacy Budget: {budget['used']:.4f} / {budget['total']:.1f} "
                  f"({budget['utilization_pct']:.1f}% used)")

            permits = trainer.governance_bridge.get_permits_summary()
            print(f"  Permits issued: {permits.get('total_permits', 0)}")
            for cc, pinfo in permits.get('per_country', {}).items():
                status_str = pinfo.get('status', '?')
                pid = pinfo.get('permit_id', '')[:12]
                print(f"    {cc}: [{status_str}] {pid}...")

            if trainer._minimization_report:
                print_subsection("DATA MINIMIZATION (Art. 44)")
                mr = trainer._minimization_report
                print(f"  Features: {mr['original_features']} -> {mr['kept_features']} "
                      f"(-{mr['reduction_pct']}%)")
                print(f"  Purpose: {mr['purpose']}")
                print(f"  Kept: {', '.join(mr['kept_feature_names'])}")
                if mr.get('purpose_removed'):
                    print(f"  Removed (purpose filter): {', '.join(mr['purpose_removed'])}")

        # Secure Processing Environment (Art. 50)
        if hasattr(trainer, 'secure_processing_bridge') and trainer.secure_processing_bridge:
            print_subsection("SECURE PROCESSING ENVIRONMENT (Art. 50)")
            sp = trainer.secure_processing_bridge
            sp_report = sp.export_report()
            # Enclave
            if sp_report.get("enclave"):
                enc = sp_report["enclave"]
                print(f"  Enclaves: {enc.get('active_enclaves', 0)} active, "
                      f"{enc.get('total_violations', 0)} violations")
            # Watermark
            if sp_report.get("watermark"):
                wm = sp_report["watermark"]
                sig_short = wm.get('signature_id', 'N/A')
                if len(sig_short) > 12:
                    sig_short = sig_short[:12] + "..."
                print(f"  Watermark: {wm.get('final_status', 'N/A')}, "
                      f"sig={sig_short}")
                if wm.get("verifications"):
                    last_v = wm["verifications"][-1]
                    print(f"    Last verification: {last_v['result']} "
                          f"(confidence={last_v['confidence']:.2%})")
            # Time Guard
            if sp_report.get("time_guard"):
                tg = sp_report["time_guard"]
                print(f"  Time Guard: {tg.get('remaining_hours', 0):.1f}h remaining, "
                      f"expired={tg.get('expired', False)}")

        # Fee Model (Art. 42)
        if hasattr(trainer, 'fee_model_bridge') and trainer.fee_model_bridge:
            print_subsection("FEE MODEL (EHDS Art. 42)")
            fee_report = trainer.fee_model_bridge.export_report()
            bd = fee_report["cost_breakdown"]
            print(f"  Total Cost:      {Style.HIGHLIGHT}{fee_report['total_cost_eur']:.2f} EUR{Colors.RESET}")
            print(f"  Base Access:     {bd['base_access']:.2f} EUR")
            print(f"  Data Volume:     {bd['data_volume']:.2f} EUR")
            print(f"  Computation:     {bd['computation']:.2f} EUR")
            print(f"  Transfer:        {bd['transfer']:.2f} EUR")

            # Per-country breakdown
            print(f"\n  {'Country':<6} {'Total EUR':>10} {'Hospitals':>10}")
            print("  " + "-" * 30)
            for cc, cf in sorted(fee_report.get("fees_by_country", {}).items()):
                print(f"  {cc:<6} {cf['total']:>10.2f} {cf['hospitals']:>10}")

            # Budget optimization result
            if fee_report.get("budget_optimization"):
                opt = fee_report["budget_optimization"]
                print_subsection("BUDGET OPTIMIZATION")
                status = "FEASIBLE" if opt["feasible"] else "INFEASIBLE"
                status_color = Style.SUCCESS if opt["feasible"] else Style.ERROR
                print(f"  Status:          {status_color}{status}{Colors.RESET}")
                print(f"  Original Cost:   {opt['original_cost_eur']:.2f} EUR")
                print(f"  Optimized Cost:  {opt['optimized_cost_eur']:.2f} EUR")
                print(f"  Strategy:        {opt['strategy']}")
                print(f"  {opt['explanation']}")

        # Compliance summary
        violations = trainer.audit_log.get_violations()
        print_subsection("COMPLIANCE SUMMARY")
        print(f"  Audit entries: {len(trainer.audit_log.entries)}")
        print(f"  Violations: {len(violations)}")
        print(f"  Effective epsilon: {results['effective_epsilon']:.1f}")
        print(f"  Total time: {results['total_time']:.1f}s")

        # EHDS Compliance Report
        try:
            from governance.ehds_compliance_report import EHDSComplianceReport
            report = EHDSComplianceReport()
            report.generate_from_trainer(trainer, self.config)
            self._last_compliance_report = report
            print(report.to_terminal_display())
        except Exception:
            pass

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
            if hasattr(trainer, 'myhealth_bridge') and trainer.myhealth_bridge:
                print(f"  - myhealth_eu_report.json")
                print(f"  - ncpeh_topology.csv")
                print(f"  - inter_ncp_latency.csv")
            if hasattr(trainer, 'governance_bridge') and trainer.governance_bridge:
                print(f"  - governance_lifecycle.json")
                print(f"  - permits_summary.json")
            if hasattr(trainer, '_minimization_report') and trainer._minimization_report:
                print(f"  - minimization_report.json")
            if hasattr(trainer, 'secure_processing_bridge') and trainer.secure_processing_bridge:
                print(f"  - secure_processing.json")
            if hasattr(trainer, 'fee_model_bridge') and trainer.fee_model_bridge:
                print(f"  - fee_model.json")
                print(f"  - fee_breakdown.csv")
            # EHDS Compliance Report
            if self._last_compliance_report:
                import json as _json
                report_path = output_dir / "ehds_compliance_report.json"
                with open(report_path, "w", encoding="utf-8") as f:
                    _json.dump(self._last_compliance_report.to_json(),
                               f, indent=2, default=str)
                latex_path = output_dir / "table_ehds_compliance.tex"
                with open(latex_path, "w", encoding="utf-8") as f:
                    f.write(self._last_compliance_report.to_latex_table())
                print(f"  - ehds_compliance_report.json")
                print(f"  - table_ehds_compliance.tex")
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
                myhealth_eu_enabled=c.get("myhealth_eu_enabled", False),
                myhealth_eu_config=c.get("myhealth_eu_config", {}),
                governance_lifecycle_enabled=c.get("governance_lifecycle_enabled", False),
                governance_config=c.get("governance_config", {}),
                secure_processing_enabled=c.get("secure_processing_enabled", False),
                secure_processing_config=c.get("secure_processing_config", {}),
                fee_model_enabled=c.get("fee_model_enabled", False),
                fee_model_config=c.get("fee_model_config", {}),
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

            # Store for compliance report (menu 8)
            self._last_trainer = trainer
            self._last_results = results

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

    def _show_ncpeh_topology(self):
        """Display NCPeH topology and inter-NCP latency matrix."""
        clear_screen()
        print_section("MyHealth@EU - NCPeH TOPOLOGY")

        print_info("National Contact Points for eHealth (NCPeH)")
        print_info("Infrastructure operativa per lo scambio dati sanitari cross-border\n")

        try:
            from governance.myhealth_eu_bridge import NCPEH_TOPOLOGY, MyHealthEUBridge

            # NCP node table
            print_subsection("NCPeH NODES")
            print(f"\n  {'Node':<14} {'Country':<14} {'Tier':>5} {'Bandwidth':>10} "
                  f"{'Services':<22} {'Since':>6}")
            print(f"  {'-'*75}")
            for cc in sorted(NCPEH_TOPOLOGY.keys()):
                ncp = NCPEH_TOPOLOGY[cc]
                services = ", ".join(
                    s.replace("patient_summary", "PS").replace(
                        "eprescription", "eP").replace(
                        "central_services", "Central")
                    for s in ncp.services
                )
                print(f"  {ncp.ncp_id:<14} {ncp.country_name:<14} "
                      f"{ncp.infrastructure_tier:>5} "
                      f"{ncp.bandwidth_mbps:>7.0f}Mb "
                      f"{services:<22} {ncp.operational_since:>6}")

            print(f"\n  Tier: 1=basic, 2=established, 3=advanced")
            print(f"  PS=Patient Summary, eP=ePrescription\n")

            # Inter-NCP latency matrix
            # Build a temporary bridge to compute latencies
            from terminal.cross_border import EU_COUNTRY_PROFILES, HospitalNode
            temp_hospitals = []
            for i, cc in enumerate(sorted(NCPEH_TOPOLOGY.keys())):
                profile = EU_COUNTRY_PROFILES.get(cc)
                if profile:
                    temp_hospitals.append(HospitalNode(
                        hospital_id=i, name=f"Temp_{cc}",
                        country_code=cc, country_profile=profile,
                        effective_epsilon=1.0, effective_delta=1e-5,
                    ))
            if temp_hospitals:
                bridge = MyHealthEUBridge(
                    hospitals=temp_hospitals, config={}, seed=42)
                codes, matrix = bridge.get_latency_matrix_display()

                print_subsection("INTER-NCP LATENCY MATRIX (ms)")
                # Header
                header = f"  {'':>5}" + "".join(f"{cc:>7}" for cc in codes)
                print(header)
                print(f"  {'-'*(5 + 7 * len(codes))}")
                for c1 in codes:
                    row = f"  {c1:>5}"
                    for c2 in codes:
                        if c1 == c2:
                            row += f"{'--':>7}"
                        else:
                            row += f"{matrix[c1][c2]:>7.1f}"
                    print(row)

            # Architecture description
            print_subsection("ARCHITETTURA FL GERARCHICA VIA NCPeH")
            print_info("Livello 1: Ospedali inviano model update al NCPeH nazionale")
            print_info("Livello 2: NCPeH inviano modello aggregato nazionale a EU (Brussels)")
            print_info("")
            print_info("Vantaggi:")
            print_info("  - Riduzione costo comunicazione cross-border (~30-50%)")
            print_info("  - Aggregazione nazionale preserva sovranita dati")
            print_info("  - Allineamento con infrastruttura MyHealth@EU operativa")
            print_info("")
            print_info("EHDS Reference: Chapter II, Art. 5-12")

        except ImportError as e:
            print_error(f"Errore import: {e}")
        except Exception as e:
            print_error(f"Errore: {e}")
            import traceback
            traceback.print_exc()

        input(f"\n{Style.MUTED}Premi Enter per continuare...{Colors.RESET}")

    def _show_ehds_compliance(self):
        """Display EHDS Compliance Report (menu item 8)."""
        clear_screen()
        print_section("EHDS COMPLIANCE REPORT")

        if self._last_trainer is None:
            print_error("Nessuna simulazione eseguita.")
            print_info("Eseguire prima la simulazione (menu 4) o opt-out (menu 5).")
            input(f"\n{Style.MUTED}Premi Enter per continuare...{Colors.RESET}")
            return

        try:
            from governance.ehds_compliance_report import EHDSComplianceReport

            report = EHDSComplianceReport()
            report.generate_from_trainer(self._last_trainer, self.config)
            self._last_compliance_report = report

            # Display report
            print(report.to_terminal_display())

            # Offer to save
            save = confirm("Salvare report JSON + LaTeX?", default=True)
            if save:
                import json as _json
                base_dir = Path(__file__).parent.parent.parent / "results" / "cross_border"
                base_dir.mkdir(parents=True, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                report_dir = base_dir / f"ehds_compliance_{timestamp}"
                report_dir.mkdir(parents=True, exist_ok=True)

                with open(report_dir / "ehds_compliance_report.json", "w",
                          encoding="utf-8") as f:
                    _json.dump(report.to_json(), f, indent=2, default=str)
                with open(report_dir / "table_ehds_compliance.tex", "w",
                          encoding="utf-8") as f:
                    f.write(report.to_latex_table())
                print_success(f"Salvato in: results/cross_border/ehds_compliance_{timestamp}/")

            # Offer scenario comparison
            compare = confirm("Confrontare con uno scenario diverso?", default=False)
            if compare:
                self._compare_ehds_scenarios(report)

        except ImportError as e:
            print_error(f"Errore import: {e}")
        except Exception as e:
            print_error(f"Errore: {e}")
            import traceback
            traceback.print_exc()

        input(f"\n{Style.MUTED}Premi Enter per continuare...{Colors.RESET}")

    def _show_governance_info(self):
        """Display governance lifecycle info (menu item 9)."""
        clear_screen()
        print_section("HDAB GOVERNANCE LIFECYCLE (EHDS Chapter IV)")

        print_info("Full EHDS Chapter IV lifecycle for secondary use of health data\n")

        print_subsection("LIFECYCLE STEPS")
        steps = [
            ("1", "HDAB Connect & Auth (Art. 50)", "MultiHDABCoordinator", "governance/hdab_integration.py"),
            ("2", "Permit Request (Art. 53)", "HDABClient.request_new_permit()", "governance/hdab_integration.py"),
            ("3", "Purpose Validation (Art. 33-34)", "PermitValidator", "governance/data_permits.py"),
            ("4", "Data Minimization (Art. 44)", "DataMinimizer", "governance/data_minimization.py"),
            ("5", "Training with Budget (Art. 42)", "PermitAwareTrainingContext", "governance/permit_training.py"),
            ("6", "Per-Round Audit (GDPR Art. 30)", "ComplianceLogger", "governance/compliance_logging.py"),
            ("7", "Session Closure", "GovernanceLifecycleBridge", "governance/governance_lifecycle.py"),
        ]

        print(f"\n  {'#':<4} {'Step':<38} {'Module':<32} {'File'}")
        print("  " + "-" * 105)
        for num, step, module, filepath in steps:
            print(f"  {num:<4} {step:<38} {module:<32} {filepath}")

        print_subsection("DATA MINIMIZATION (Art. 44)")
        print_info("Purpose-based feature selection in two phases:")
        print_info("  Phase 1: Remove features not relevant to declared purpose")
        print_info("  Phase 2: MI-based importance filtering (threshold-based)")
        print_info("")
        print_info("Supported purposes (EHDS Art. 53):")
        purposes = [
            "scientific_research", "public_health_surveillance",
            "health_policy", "education_training",
            "ai_system_development", "personalized_medicine",
            "official_statistics", "patient_safety",
        ]
        for p in purposes:
            print(f"    - {p}")

        print_subsection("PERMIT-AWARE TRAINING")
        print_info("Privacy budget tracked per-round under HDAB permit:")
        print_info("  - Each round consumes epsilon_spent from total budget")
        print_info("  - Training halts if budget exhausted or permit expired")
        print_info("  - Audit trail written in GDPR Art. 30 format")

        print_subsection("COME USARE")
        print_info("1. Menu 1 (Configura) -> Abilita 'Governance Layer'")
        print_info("2. Menu 4 (Esegui simulazione) -> Governance attivato automaticamente")
        print_info("3. Output: governance_lifecycle.json, permits_summary.json,")
        print_info("   minimization_report.json (if minimization enabled)")

        input(f"\n{Style.MUTED}Premi Enter per continuare...{Colors.RESET}")

    def _compare_ehds_scenarios(self, report_a):
        """Run a second scenario and compare compliance reports."""
        from governance.ehds_compliance_report import EHDSComplianceReport
        from terminal.cross_border import CrossBorderFederatedTrainer

        print_subsection("CONFIGURAZIONE SCENARIO B")
        print_info("Scegli il parametro da modificare:")
        print(f"  1 - Epsilon globale (attuale: {self.config['global_epsilon']})")
        print(f"  2 - Algoritmo (attuale: {self.config['algorithm']})")
        print(f"  3 - Numero round (attuale: {self.config['num_rounds']})")

        param = get_choice("Parametro", ["1", "2", "3"], default="1")

        import copy
        config_b = copy.deepcopy(self.config)

        if param == "1":
            new_eps = get_number("Nuovo epsilon globale", default=1.0, min_val=0.1, max_val=100.0)
            config_b["global_epsilon"] = new_eps
        elif param == "2":
            algos = ["FedAvg", "FedProx", "SCAFFOLD", "FedNova", "FedAdam"]
            algo = get_choice("Algoritmo", algos, default="FedProx")
            config_b["algorithm"] = algo
        else:
            new_rounds = int(get_number("Numero round", default=30, min_val=1, max_val=200))
            config_b["num_rounds"] = new_rounds

        c = config_b
        print_info(f"Esecuzione scenario B...")

        try:
            trainer_b = CrossBorderFederatedTrainer(
                countries=c["countries"],
                hospitals_per_country=c["hospitals_per_country"],
                algorithm=c["algorithm"],
                num_rounds=c["num_rounds"],
                local_epochs=c["local_epochs"],
                batch_size=c["batch_size"],
                learning_rate=c["learning_rate"],
                global_epsilon=c["global_epsilon"],
                purpose=c["purpose"],
                dataset_type=c.get("dataset_type", "synthetic"),
                dataset_path=c.get("dataset_path"),
                img_size=c.get("img_size", 128),
                seed=c.get("seed", 42),
                simulate_latency=c.get("simulate_latency", True),
                mu=c.get("mu", 0.1),
                server_lr=c.get("server_lr", 0.1),
                beta1=c.get("beta1", 0.9),
                beta2=c.get("beta2", 0.99),
                tau=c.get("tau", 1e-3),
                jurisdiction_privacy_enabled=c.get("jurisdiction_privacy_enabled", False),
                country_overrides=c.get("country_overrides", {}),
                hospital_allocation_fraction=c.get("hospital_allocation_fraction", 1.0),
                noise_strategy=c.get("noise_strategy", "global"),
                min_active_clients=c.get("min_active_clients", 2),
                ihe_enabled=c.get("ihe_enabled", False),
                ihe_config=c.get("ihe_config", {}),
                data_quality_enabled=c.get("data_quality_enabled", False),
                data_quality_config=c.get("data_quality_config", {}),
                myhealth_eu_enabled=c.get("myhealth_eu_enabled", False),
                myhealth_eu_config=c.get("myhealth_eu_config", {}),
            )

            trainer_b.train()

            report_b = EHDSComplianceReport()
            report_b.generate_from_trainer(trainer_b, config_b)

            comparison = EHDSComplianceReport.compare_scenarios(report_a, report_b)
            print(EHDSComplianceReport.format_comparison_terminal(comparison))

        except Exception as e:
            print_error(f"Errore scenario B: {e}")
            import traceback
            traceback.print_exc()

    # -----------------------------------------------------------------
    # EHDS BENCHMARK (Menu A)
    # -----------------------------------------------------------------

    def _run_ehds_benchmark(self):
        """Run EHDS benchmark: Privacy-Utility-Cost tradeoff analysis."""
        clear_screen()
        print_section("EHDS BENCHMARK - PRIVACY-UTILITY-COST TRADEOFF")

        print(f"\n{Style.INFO}Esegue un benchmark EHDS-nativo su una griglia di configurazioni:{Colors.RESET}")
        print(f"  - Epsilon (privacy budget) x Algoritmi FL x Livelli Governance")
        print(f"  - Genera: 5 plot, 2 tabelle LaTeX, CSV, JSON\n")

        from benchmarks.ehds_benchmark import (
            EHDSBenchmarkConfig, run_ehds_benchmark, save_results,
        )

        # --- Interactive config ---
        print_subsection("CONFIGURAZIONE BENCHMARK")

        eps_str = input(f"  Epsilon values (comma-separated) [{Style.MUTED}1,5,10,50,100{Colors.RESET}]: ").strip()
        if eps_str:
            epsilons = [float(x.strip()) for x in eps_str.split(",")]
        else:
            epsilons = [1.0, 5.0, 10.0, 50.0, 100.0]

        print(f"\n  Algoritmi disponibili: {', '.join(FL_ALGORITHMS)}")
        algo_str = input(f"  Algoritmi (comma-separated) [{Style.MUTED}FedAvg,FedProx,SCAFFOLD{Colors.RESET}]: ").strip()
        if algo_str:
            algorithms = [a.strip() for a in algo_str.split(",")]
        else:
            algorithms = ["FedAvg", "FedProx", "SCAFFOLD"]

        gov_str = input(f"  Governance levels (comma-separated) [{Style.MUTED}minimal,full{Colors.RESET}]: ").strip()
        if gov_str:
            governance_configs = [g.strip() for g in gov_str.split(",")]
        else:
            governance_configs = ["minimal", "full"]

        num_rounds = get_int("  Rounds per config", default=10, min_val=3, max_val=100)

        total_runs = len(epsilons) * len(algorithms) * len(governance_configs)
        est_time = total_runs * 8  # ~8 sec per run

        print(f"\n{Style.HIGHLIGHT}Riepilogo benchmark:{Colors.RESET}")
        print(f"  Epsilons:     {epsilons}")
        print(f"  Algoritmi:    {algorithms}")
        print(f"  Governance:   {governance_configs}")
        print(f"  Rounds:       {num_rounds}")
        print(f"  Paesi:        {self.config['countries']}")
        print(f"  Totale runs:  {total_runs}")
        print(f"  Tempo stimato: ~{est_time // 60}m {est_time % 60}s")

        if not confirm("\nAvviare il benchmark?", default=True):
            return

        config = EHDSBenchmarkConfig(
            epsilons=epsilons,
            algorithms=algorithms,
            countries=self.config["countries"],
            hospitals_per_country=self.config.get("hospitals_per_country", 1),
            num_rounds=num_rounds,
            local_epochs=self.config.get("local_epochs", 3),
            batch_size=self.config.get("batch_size", 32),
            learning_rate=self.config.get("learning_rate", 0.01),
            purpose=self.config.get("purpose", "scientific_research"),
            dataset_type=self.config.get("dataset_type", "synthetic"),
            governance_configs=governance_configs,
            min_acceptable_accuracy=0.55,
            seeds=[self.config.get("seed", 42)],
        )

        print()
        print_info("Avvio benchmark EHDS...")
        print()

        def progress_cb(current, total, label, accuracy, compliance):
            pct = current / total * 100
            bar_len = 30
            filled = int(bar_len * current / total)
            bar = "#" * filled + "-" * (bar_len - filled)
            print(f"  [{bar}] {pct:5.1f}% | {current}/{total} | "
                  f"{label:<35} | Acc={accuracy:.2%} | Compl={compliance:.1f}%")

        try:
            results = run_ehds_benchmark(config, progress_callback=progress_cb)

            # Display summary
            self._display_benchmark_summary(results)

            # Auto-save
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_dir = Path(__file__).parent.parent.parent / "results" / "ehds_benchmark"
            folder = f"benchmark_{len(algorithms)}algo_{len(epsilons)}eps_{timestamp}"
            output_dir = base_dir / folder

            saved = save_results(results, str(output_dir))

            print_success(f"\nRisultati salvati in: results/ehds_benchmark/{folder}/")
            for f_name in saved:
                print(f"  - {f_name}")

        except Exception as e:
            print_error(f"Errore benchmark: {e}")
            import traceback
            traceback.print_exc()

        input(f"\n{Style.MUTED}Premi Enter per continuare...{Colors.RESET}")

    def _display_benchmark_summary(self, results):
        """Display benchmark summary with best configs and Pareto analysis."""
        from benchmarks.ehds_benchmark import EHDSBenchmarkResults

        runs = results.runs
        if not runs:
            print_warning("Nessun run completato.")
            return

        print()
        print_section("RISULTATI BENCHMARK EHDS")

        # Best by accuracy
        best_acc = max(runs, key=lambda r: r.final_accuracy)
        print_subsection("MIGLIORE PER ACCURACY")
        print(f"  {best_acc.config_label}: {Style.HIGHLIGHT}{best_acc.final_accuracy:.2%}{Colors.RESET} "
              f"(F1={best_acc.final_f1:.3f}, Compliance={best_acc.compliance_score:.1f}%)")

        # Best by compliance
        best_comp = max(runs, key=lambda r: r.compliance_score)
        print_subsection("MIGLIORE PER COMPLIANCE")
        print(f"  {best_comp.config_label}: {Style.HIGHLIGHT}{best_comp.compliance_score:.1f}%{Colors.RESET} "
              f"(Acc={best_comp.final_accuracy:.2%})")

        # Best by combined (normalized sum)
        best_combined = max(runs, key=lambda r: r.final_accuracy + r.compliance_score / 100)
        print_subsection("MIGLIORE COMBINATO (Acc + Compliance)")
        print(f"  {best_combined.config_label}: Acc={best_combined.final_accuracy:.2%}, "
              f"Compliance={best_combined.compliance_score:.1f}%")

        # Privacy-Utility analysis
        print_subsection("ANALISI PRIVACY-UTILITY")
        eps_groups = {}
        for r in runs:
            eps_groups.setdefault(r.effective_epsilon, []).append(r.final_accuracy)

        print(f"\n  {'Epsilon':>10} {'Avg Accuracy':>14} {'Std':>8}")
        print("  " + "-" * 35)
        for eps in sorted(eps_groups.keys()):
            accs = eps_groups[eps]
            avg = sum(accs) / len(accs)
            std = (sum((a - avg) ** 2 for a in accs) / len(accs)) ** 0.5
            print(f"  {eps:>10.1f} {avg:>13.2%} {std:>8.4f}")

        # Time-to-compliance
        reached = [r for r in runs if r.rounds_to_compliance is not None]
        print_subsection("TIME-TO-COMPLIANCE")
        print(f"  Configurazioni che raggiungono threshold: {len(reached)}/{len(runs)}")
        if reached:
            fastest = min(reached, key=lambda r: r.rounds_to_compliance)
            print(f"  Piu veloce: {fastest.config_label} (round {fastest.rounds_to_compliance})")

        # Overall stats
        print_subsection("STATISTICHE GENERALI")
        print(f"  Runs completati:  {len(runs)}")
        print(f"  Tempo totale:     {results.total_time_seconds:.1f}s")
        avg_acc = sum(r.final_accuracy for r in runs) / len(runs)
        avg_comp = sum(r.compliance_score for r in runs) / len(runs)
        print(f"  Media accuracy:   {avg_acc:.2%}")
        print(f"  Media compliance: {avg_comp:.1f}%")

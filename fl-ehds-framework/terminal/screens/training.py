"""
Training screen for FL-EHDS terminal interface.
Provides federated learning training with algorithm selection and DP options.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional
import time
import json
from datetime import datetime

# Add parent directories to path for imports
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
from terminal.progress import FLProgressBar, TrainingProgress
from terminal.validators import (
    get_int, get_float, get_bool, get_choice, confirm, display_config_summary
)
from terminal.menu import Menu, MenuItem, MENU_STYLE


# Available FL algorithms
FL_ALGORITHMS = [
    "FedAvg",
    "FedProx",
    "SCAFFOLD",
    "FedNova",
    "FedDyn",
    "FedAdam",
    "FedYogi",
    "FedAdagrad",
    "Per-FedAvg",
    "Ditto",
]

# Data distribution types
DATA_DISTRIBUTIONS = [
    "IID (uniforme)",
    "Non-IID (label skew)",
    "Non-IID (quantity skew)",
]


class TrainingScreen:
    """Training screen with parameter configuration and execution."""

    def __init__(self):
        self.config = self._default_config()
        self.results = None

    def _default_config(self) -> Dict[str, Any]:
        """Return default configuration from config.yaml with hardcoded fallbacks."""
        fallback = {
            "algorithm": "FedAvg",
            "num_clients": 5,
            "num_rounds": 30,
            "local_epochs": 3,
            "batch_size": 32,
            "learning_rate": 0.01,
            "dp_enabled": False,
            "dp_epsilon": 10.0,
            "dp_delta": 1e-5,
            "dp_clip_norm": 1.0,
            "data_distribution": "Non-IID (label skew)",
            "mu": 0.1,
            "seed": 42,
            "server_lr": 0.1,
            "beta1": 0.9,
            "beta2": 0.99,
            "tau": 1e-3,
            "dataset_type": "synthetic",
            "dataset_name": None,
            "dataset_path": None,
            "model_type": "resnet18",
            "freeze_backbone": False,
            "use_class_weights": True,
            "ehds_permit_enabled": False,
            "ehds_purpose": "ai_system_development",
            "ehds_data_categories": ["ehr"],
            "ehds_privacy_budget": 100.0,
            "ehds_max_rounds": None,
            "ehds_data_minimization": False,
        }
        try:
            from config.config_loader import get_training_defaults
            yaml_defaults = get_training_defaults()
            fallback.update(yaml_defaults)
        except (ImportError, Exception):
            pass
        return fallback

    def _get_available_datasets(self) -> Dict[str, Dict]:
        """Get available imaging datasets."""
        try:
            from terminal.screens.datasets import DatasetManager
            manager = DatasetManager()
            return {
                name: {
                    "type": ds.type,
                    "samples": ds.total_samples,
                    "classes": ds.num_classes,
                    "class_names": ds.class_names,
                    "path": str(ds.path) if ds.path else None,
                }
                for name, ds in manager.datasets.items()
                if ds.type == "imaging"
            }
        except Exception:
            return {}

    def run(self):
        """Run the training screen."""
        while True:
            clear_screen()
            print_section("TRAINING FEDERATO")

            menu = Menu("Seleziona azione", [
                MenuItem("1", "Configura parametri", self._configure),
                MenuItem("2", "Avvia training", self._run_training),
                MenuItem("3", "Visualizza risultati", self._show_results),
                MenuItem("4", "Genera grafici convergenza", self._generate_plots),
                MenuItem("5", "Esporta risultati", self._export_results),
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
        """Configure training parameters."""
        clear_screen()
        print_section("CONFIGURAZIONE TRAINING")

        # Algorithm selection
        print_subsection("Algoritmo FL")
        if HAS_QUESTIONARY:
            self.config["algorithm"] = questionary.select(
                "Seleziona algoritmo:",
                choices=FL_ALGORITHMS,
                default=self.config["algorithm"],
                style=MENU_STYLE,
            ).ask() or self.config["algorithm"]
        else:
            self.config["algorithm"] = get_choice(
                "Seleziona algoritmo:",
                FL_ALGORITHMS,
                default=self.config["algorithm"]
            )

        # Algorithm profile suggestion
        try:
            from config.config_loader import get_algorithm_profile
            profile = get_algorithm_profile(self.config["algorithm"])
            if profile:
                print()
                print_info(f"Profilo suggerito per {self.config['algorithm']}:")
                print(f"  {profile.get('description', '')}")
                is_img = self.config.get("dataset_type") == "imaging"
                lr_key = "learning_rate_imaging" if is_img else "learning_rate"
                lr_val = profile.get(lr_key, profile.get("learning_rate"))
                print(f"  LR={lr_val}  Epochs={profile.get('local_epochs')}  "
                      f"Rounds={profile.get('num_rounds')}  Batch={profile.get('batch_size')}")
                if "mu" in profile:
                    print(f"  mu={profile['mu']}")
                if "server_lr" in profile:
                    print(f"  server_lr={profile['server_lr']}  "
                          f"beta1={profile.get('beta1')}  beta2={profile.get('beta2')}")
                print(f"  Consigliato per: {', '.join(profile.get('recommended_for', []))}")
                if confirm("\nApplicare profilo suggerito?", default=True):
                    self.config["learning_rate"] = lr_val
                    for k in ["local_epochs", "num_rounds", "batch_size", "mu",
                              "server_lr", "beta1", "beta2", "tau"]:
                        if k in profile:
                            self.config[k] = profile[k]
                    print_success("Profilo applicato. Puoi modificare i parametri di seguito.")
        except (ImportError, Exception):
            pass

        # Dataset selection
        print_subsection("Selezione Dataset")
        available_datasets = self._get_available_datasets()

        dataset_choices = [
            "Sintetico (Healthcare Tabular)",
            "FHIR R4 (Ospedali Sintetici)",
            "OMOP-CDM (Armonizzazione Cross-Border)",
        ]
        dataset_map = {
            "Sintetico (Healthcare Tabular)": "synthetic",
            "FHIR R4 (Ospedali Sintetici)": "fhir",
            "OMOP-CDM (Armonizzazione Cross-Border)": "omop",
        }

        # Add real tabular datasets if available
        try:
            diabetes_path = Path(__file__).parent.parent.parent / "data" / "diabetes" / "diabetic_data.csv"
            if diabetes_path.exists():
                label = "Diabetes 130-US (101K encounters, 130 ospedali, readmission)"
                dataset_choices.insert(3, label)
                dataset_map[label] = "diabetes"
        except Exception:
            pass
        try:
            heart_path = Path(__file__).parent.parent.parent / "data" / "heart_disease"
            if heart_path.exists():
                label = "Heart Disease UCI (920 pazienti, 4 ospedali, diagnosi)"
                dataset_choices.insert(len([c for c in dataset_choices if "img" not in c.lower()]), label)
                dataset_map[label] = "heart_disease"
        except Exception:
            pass

        if available_datasets:
            for name, info in available_datasets.items():
                label = f"{name} ({info['samples']:,} img, {info['classes']} classi)"
                dataset_choices.append(label)
                dataset_map[label] = name

        if HAS_QUESTIONARY:
            selected = questionary.select(
                "Seleziona dataset:",
                choices=dataset_choices,
                style=MENU_STYLE,
            ).ask() or dataset_choices[0]
        else:
            print("Dataset disponibili:")
            for i, choice in enumerate(dataset_choices):
                print(f"  {i + 1}. {choice}")
            idx = get_int("Selezione", default=1, min_val=1, max_val=len(dataset_choices)) - 1
            selected = dataset_choices[idx]

        selected_key = dataset_map[selected]
        if selected_key == "synthetic":
            self.config["dataset_type"] = "synthetic"
            self.config["dataset_name"] = None
            self.config["dataset_path"] = None
            self.config["learning_rate"] = 0.01  # Default for tabular
        elif selected_key == "fhir":
            self.config["dataset_type"] = "fhir"
            self.config["dataset_name"] = "fhir_synthetic"
            self.config["dataset_path"] = None
            self.config["learning_rate"] = 0.01  # Tabular data
            # Load FHIR config and show info
            try:
                from config.config_loader import get_fhir_config
                fhir_cfg = get_fhir_config()
                profiles = fhir_cfg.get("profiles", [])
                num_c = self.config.get("num_clients", 5)
                assigned = [profiles[i % len(profiles)] for i in range(num_c)]
                print_info("FHIR R4: dati generati da profili ospedalieri (non-IID naturale)")
                print(f"  Profili: {', '.join(assigned)}")
                print(f"  Pazienti/ospedale: {fhir_cfg.get('samples_per_client', 500)}")
                print(f"  Features: {len(fhir_cfg.get('feature_spec', []))} ({fhir_cfg.get('label', 'mortality_30day')})")
            except (ImportError, Exception):
                print_info("FHIR R4: ospedali sintetici con profili diversi")
        elif selected_key == "diabetes":
            self.config["dataset_type"] = "diabetes"
            self.config["dataset_name"] = "diabetes_130us"
            self.config["dataset_path"] = str(Path(__file__).parent.parent.parent / "data" / "diabetes" / "diabetic_data.csv")
            self.config["learning_rate"] = 0.01  # Tabular data
            print_info("Diabetes 130-US: 101,766 encounter da 130 ospedali USA")
            print_info("  Target: readmission <30 giorni (binario)")
            print_info("  22 features: demographics + diagnosi ICD-9 + farmaci + lab")
            print_info("  Partizione per ospedale (non-IID naturale)")
            print_info("  FHIR R4: Patient, Encounter, Condition, Observation, MedicationStatement")
        elif selected_key == "heart_disease":
            self.config["dataset_type"] = "heart_disease"
            self.config["dataset_name"] = "heart_disease_uci"
            self.config["dataset_path"] = str(Path(__file__).parent.parent.parent / "data" / "heart_disease")
            self.config["learning_rate"] = 0.01
            self.config["num_clients"] = 4  # 4 real hospitals
            print_info("Heart Disease UCI: 920 pazienti da 4 ospedali (Cleveland, Hungarian, Swiss, VA)")
            print_info("  Target: presenza malattia cardiaca (binario)")
            print_info("  13 features: demographics + vitali + ECG + stress test")
            print_info("  Partizione per ospedale (non-IID naturale, 4 client)")
            print_info("  FHIR R4: Patient, Observation, Condition, DiagnosticReport")
        elif selected_key == "omop":
            self.config["dataset_type"] = "omop"
            self.config["dataset_name"] = "omop_harmonized"
            self.config["dataset_path"] = None
            self.config["learning_rate"] = 0.01  # Tabular data
            # Load OMOP config and show info
            try:
                from config.config_loader import get_omop_config
                omop_cfg = get_omop_config()
                profiles = omop_cfg.get("profiles", [])
                countries = omop_cfg.get("country_codes", [])
                num_c = self.config.get("num_clients", 5)
                assigned_p = [profiles[i % len(profiles)] for i in range(num_c)]
                assigned_c = [countries[i % len(countries)] for i in range(num_c)]
                from data.omop_harmonizer import COUNTRY_VOCABULARY_PROFILES
                print_info("OMOP-CDM: armonizzazione vocabolari cross-border (non-IID da eterogeneita)")
                print(f"  Pazienti/ospedale: {omop_cfg.get('samples_per_client', 500)}")
                print(f"  ~36 features standardizzate OMOP (temporal windows: 30d/90d/365d)")
                for i in range(num_c):
                    vp = COUNTRY_VOCABULARY_PROFILES.get(assigned_c[i], {})
                    print(f"  Client {i}: {assigned_c[i]} ({assigned_p[i]}) - {vp.get('coding_system', '?')}")
            except (ImportError, Exception):
                print_info("OMOP-CDM: armonizzazione vocabolari europei cross-border")
        else:
            self.config["dataset_type"] = "imaging"
            self.config["dataset_name"] = selected_key
            self.config["dataset_path"] = available_datasets[selected_key]["path"]
            self.config["learning_rate"] = 0.001  # Better for CNN
            print_info(f"Dataset imaging selezionato: {selected_key}")
            print(f"  Path: {self.config['dataset_path']}")
            print(f"  Classi: {available_datasets[selected_key]['class_names']}")

        # Dataset parameter suggestion
        if self.config["dataset_type"] in ("imaging", "fhir", "omop", "diabetes") and self.config.get("dataset_name"):
            try:
                from config.config_loader import get_dataset_parameters
                ds_params = get_dataset_parameters(self.config["dataset_name"])
                if ds_params:
                    print()
                    print_info(f"Parametri suggeriti per {self.config['dataset_name']}:")
                    print(f"  LR={ds_params.get('learning_rate')}  "
                          f"Batch={ds_params.get('batch_size')}  "
                          f"Rounds={ds_params.get('num_rounds')}  "
                          f"ImgSize={ds_params.get('img_size')}")
                    print(f"  Algoritmi consigliati: "
                          f"{', '.join(ds_params.get('recommended_algorithms', []))}")
                    if ds_params.get("notes"):
                        print(f"  Note: {ds_params['notes']}")
                    if ds_params.get("class_weight"):
                        print(f"  Class weighting: consigliato (dataset sbilanciato)")
                    if confirm("\nApplicare parametri dataset?", default=True):
                        for k in ["learning_rate", "batch_size", "num_rounds", "local_epochs", "img_size"]:
                            if k in ds_params and ds_params[k] is not None:
                                self.config[k] = ds_params[k]
                        print_success("Parametri dataset applicati.")
            except (ImportError, Exception):
                pass

        # Model selection (imaging datasets only)
        if self.config["dataset_type"] == "imaging":
            model_choice = questionary.select(
                "Modello:",
                choices=[
                    "ResNet18 (pretrained ImageNet, consigliato)",
                    "CNN custom (leggera, ~500K params)",
                ],
                default="ResNet18 (pretrained ImageNet, consigliato)",
            ).ask()
            if model_choice and "CNN" in model_choice:
                self.config["model_type"] = "cnn"
                print_info("Modello: CNN custom (128x128)")
            else:
                self.config["model_type"] = "resnet18"
                self.config["batch_size"] = 16  # ResNet uses more memory
                print_info("Modello: ResNet18 pretrained (224x224, GroupNorm)")

        # Basic parameters
        print_subsection("Parametri Base")
        self.config["num_clients"] = get_int(
            "Numero client",
            default=self.config["num_clients"],
            min_val=2, max_val=100
        )
        self.config["num_rounds"] = get_int(
            "Numero round",
            default=self.config["num_rounds"],
            min_val=1, max_val=1000
        )
        self.config["local_epochs"] = get_int(
            "Epoche locali",
            default=self.config["local_epochs"],
            min_val=1, max_val=50
        )
        self.config["batch_size"] = get_int(
            "Batch size",
            default=self.config["batch_size"],
            min_val=1, max_val=512
        )
        self.config["learning_rate"] = get_float(
            "Learning rate",
            default=self.config["learning_rate"],
            min_val=0.0001, max_val=1.0
        )

        # Algorithm-specific parameters
        if self.config["algorithm"] in ["FedProx", "Ditto"]:
            label = "Parametro mu (FedProx)" if self.config["algorithm"] == "FedProx" else "Lambda regolarizzazione (Ditto)"
            self.config["mu"] = get_float(
                label,
                default=self.config["mu"],
                min_val=0.0, max_val=1.0
            )

        # Server optimizer parameters for adaptive algorithms
        if self.config["algorithm"] in ["FedAdam", "FedYogi", "FedAdagrad"]:
            print_info(f"Parametri server optimizer per {self.config['algorithm']}:")
            self.config["server_lr"] = get_float(
                "  Server learning rate",
                default=self.config["server_lr"],
                min_val=0.001, max_val=1.0
            )
            if self.config["algorithm"] in ["FedAdam", "FedYogi"]:
                self.config["beta1"] = get_float(
                    "  Beta1 (momentum)",
                    default=self.config["beta1"],
                    min_val=0.0, max_val=0.999
                )
                self.config["beta2"] = get_float(
                    "  Beta2 (velocity)",
                    default=self.config["beta2"],
                    min_val=0.0, max_val=0.999
                )
            self.config["tau"] = get_float(
                "  Tau (numerical stability)",
                default=self.config["tau"],
                min_val=1e-8, max_val=1e-1
            )

        # Data distribution
        print_subsection("Distribuzione Dati")
        if HAS_QUESTIONARY:
            self.config["data_distribution"] = questionary.select(
                "Seleziona distribuzione:",
                choices=DATA_DISTRIBUTIONS,
                default=self.config["data_distribution"],
                style=MENU_STYLE,
            ).ask() or self.config["data_distribution"]
        else:
            self.config["data_distribution"] = get_choice(
                "Seleziona distribuzione:",
                DATA_DISTRIBUTIONS,
                default=self.config["data_distribution"]
            )

        # Differential Privacy
        print_subsection("Differential Privacy")
        self.config["dp_enabled"] = get_bool(
            "Abilitare Differential Privacy?",
            default=self.config["dp_enabled"]
        )

        if self.config["dp_enabled"]:
            self.config["dp_epsilon"] = get_float(
                "  Epsilon target",
                default=self.config["dp_epsilon"],
                min_val=0.1, max_val=100.0
            )
            self.config["dp_delta"] = get_float(
                "  Delta",
                default=self.config["dp_delta"],
                min_val=1e-10, max_val=1e-3
            )
            self.config["dp_clip_norm"] = get_float(
                "  Gradient clipping norm",
                default=self.config["dp_clip_norm"],
                min_val=0.1, max_val=10.0
            )

        # EHDS Data Permit (optional)
        print_subsection("EHDS Data Permit (Opzionale)")
        self.config["ehds_permit_enabled"] = get_bool(
            "Abilitare EHDS Data Permit governance?",
            default=self.config["ehds_permit_enabled"]
        )

        if self.config["ehds_permit_enabled"]:
            purpose_choices = [
                "ai_system_development",
                "scientific_research",
                "public_health_surveillance",
                "health_policy",
                "education_training",
                "personalized_medicine",
                "official_statistics",
                "patient_safety",
            ]
            if HAS_QUESTIONARY:
                self.config["ehds_purpose"] = questionary.select(
                    "EHDS purpose (Article 53):",
                    choices=purpose_choices,
                    default=self.config["ehds_purpose"],
                    style=MENU_STYLE,
                ).ask() or self.config["ehds_purpose"]
            else:
                self.config["ehds_purpose"] = get_choice(
                    "EHDS purpose (Article 53):",
                    purpose_choices,
                    default=self.config["ehds_purpose"],
                )

            # Auto-detect data categories from dataset type
            category_map = {
                "synthetic": ["ehr"],
                "fhir": ["ehr", "lab_results"],
                "omop": ["ehr", "lab_results"],
                "diabetes": ["ehr", "lab_results"],
                "heart_disease": ["ehr", "lab_results"],
                "imaging": ["imaging"],
            }
            self.config["ehds_data_categories"] = category_map.get(
                self.config["dataset_type"], ["ehr"]
            )
            print_info(f"  Data categories: {', '.join(self.config['ehds_data_categories'])}")

            # Privacy budget
            if self.config["dp_enabled"]:
                self.config["ehds_privacy_budget"] = self.config["dp_epsilon"]
                print_info(f"  Privacy budget da DP epsilon: {self.config['ehds_privacy_budget']}")
            else:
                self.config["ehds_privacy_budget"] = get_float(
                    "  Privacy budget (epsilon totale)",
                    default=self.config["ehds_privacy_budget"],
                    min_val=0.1, max_val=1000.0
                )

            # Max rounds
            self.config["ehds_max_rounds"] = get_int(
                "  Massimo round autorizzati",
                default=self.config["num_rounds"],
                min_val=1, max_val=10000
            )

            # Data minimization (only for tabular)
            if self.config["dataset_type"] != "imaging":
                self.config["ehds_data_minimization"] = get_bool(
                    "  Applicare data minimization (Art. 44)?",
                    default=self.config["ehds_data_minimization"]
                )
            else:
                self.config["ehds_data_minimization"] = False

        # Seed
        print_subsection("Riproducibilita")
        self.config["seed"] = get_int(
            "Random seed",
            default=self.config["seed"],
            min_val=0, max_val=99999
        )

        # Show summary
        display_config_summary(self.config)

        input(f"\n{Style.MUTED}Premi Enter per continuare...{Colors.RESET}")

    def _run_training(self):
        """Execute real federated learning training with PyTorch."""
        clear_screen()
        print_section("TRAINING FEDERATO REALE (PyTorch)")

        # Show configuration
        display_config_summary(self.config)

        if not confirm("\nAvviare il training con questa configurazione?", default=True):
            return

        print()

        # All algorithms are now supported by the real trainer
        supported_algorithms = FL_ALGORITHMS  # All 9 algorithms implemented

        try:
            from terminal.fl_trainer import FederatedTrainer, ImageFederatedTrainer

            # Map data distribution
            is_iid = "IID" in self.config["data_distribution"]

            # Progress state
            current_state = {"round": 0, "client": 0, "epoch": 0}

            def progress_callback(event_type, **kwargs):
                """Handle progress updates."""
                if event_type == "round_start":
                    r = kwargs.get("round_num", 0)
                    current_state["round"] = r
                    print(f"\n{Style.TITLE}--- Round {r}/{self.config['num_rounds']} ---{Colors.RESET}")

                elif event_type == "client_start":
                    client_id = kwargs.get("client_id", 0)
                    total = kwargs.get("total_clients", 0)
                    current_state["client"] = client_id
                    print(f"  Client {client_id + 1}/{total}: ", end="", flush=True)

                elif event_type == "epoch":
                    epoch = kwargs.get("epoch", 0)
                    total_epochs = kwargs.get("total_epochs", 0)
                    loss = kwargs.get("loss", 0)
                    acc = kwargs.get("acc", 0)
                    print(f"E{epoch} ", end="", flush=True)

                elif event_type == "client_end":
                    loss = kwargs.get("loss", 0)
                    acc = kwargs.get("acc", 0)
                    print(f"-> loss={loss:.4f}, acc={acc:.2%}")

                elif event_type == "round_end":
                    loss = kwargs.get("loss", 0)
                    acc = kwargs.get("acc", 0)
                    t = kwargs.get("time", 0)
                    print(f"  {Style.SUCCESS}Round completato: loss={loss:.4f}, acc={acc:.2%}, tempo={t:.1f}s{Colors.RESET}")

            # Create trainer based on dataset type
            print_info("Inizializzazione trainer PyTorch...")

            if self.config["dataset_type"] == "imaging" and self.config["dataset_path"]:
                # Imaging dataset - use ImageFederatedTrainer
                print_info(f"Caricamento dataset imaging: {self.config['dataset_name']}")
                print_info(f"Path: {self.config['dataset_path']}")
                print_info(f"Distribuzione su {self.config['num_clients']} client...")

                trainer = ImageFederatedTrainer(
                    data_dir=self.config["dataset_path"],
                    num_clients=self.config["num_clients"],
                    algorithm=self.config["algorithm"],
                    local_epochs=self.config["local_epochs"],
                    batch_size=self.config["batch_size"],
                    learning_rate=self.config["learning_rate"],
                    is_iid=is_iid,
                    alpha=0.5,
                    mu=self.config["mu"],
                    dp_enabled=self.config["dp_enabled"],
                    dp_epsilon=self.config["dp_epsilon"],
                    dp_clip_norm=self.config["dp_clip_norm"],
                    seed=self.config["seed"],
                    progress_callback=progress_callback,
                    # Server optimizer params for FedAdam, FedYogi, FedAdagrad
                    server_lr=self.config.get("server_lr", 0.1),
                    beta1=self.config.get("beta1", 0.9),
                    beta2=self.config.get("beta2", 0.99),
                    tau=self.config.get("tau", 1e-3),
                    # Model selection
                    model_type=self.config.get("model_type", "resnet18"),
                    freeze_backbone=self.config.get("freeze_backbone", False),
                    use_class_weights=self.config.get("use_class_weights", True),
                )
            elif self.config["dataset_type"] == "heart_disease":
                # Heart Disease UCI (4 hospitals)
                from data.heart_disease_loader import load_heart_disease_data
                print_info("Caricamento Heart Disease UCI (920 pazienti, 4 ospedali)...")

                hd_train, hd_test, hd_meta = load_heart_disease_data(
                    num_clients=self.config["num_clients"],
                    partition_by_hospital=not is_iid,
                    is_iid=is_iid,
                    test_split=0.2,
                    seed=self.config["seed"],
                    data_path=self.config.get("dataset_path"),
                )

                print_info(f"Heart Disease: {hd_meta['total_samples']} campioni, "
                           f"{hd_meta['num_features']} features, "
                           f"label={hd_meta['label_name']}")
                hosp_assign = hd_meta.get("hospital_assignment", {})
                for cid in range(self.config["num_clients"]):
                    train_n = len(hd_train[cid][1])
                    test_n = len(hd_test[cid][1])
                    pos_rate = hd_train[cid][1].mean()
                    hosp = hosp_assign.get(cid, "mixed")
                    print(f"  Client {cid} ({hosp}): {train_n} train, {test_n} test, "
                          f"disease_rate={pos_rate:.1%}")

                trainer = FederatedTrainer(
                    num_clients=self.config["num_clients"],
                    algorithm=self.config["algorithm"],
                    local_epochs=self.config["local_epochs"],
                    batch_size=self.config["batch_size"],
                    learning_rate=self.config["learning_rate"],
                    mu=self.config["mu"],
                    dp_enabled=self.config["dp_enabled"],
                    dp_epsilon=self.config["dp_epsilon"],
                    dp_clip_norm=self.config["dp_clip_norm"],
                    seed=self.config["seed"],
                    progress_callback=progress_callback,
                    server_lr=self.config.get("server_lr", 0.1),
                    beta1=self.config.get("beta1", 0.9),
                    beta2=self.config.get("beta2", 0.99),
                    tau=self.config.get("tau", 1e-3),
                    external_data=hd_train,
                    external_test_data=hd_test,
                    input_dim=hd_meta["num_features"],
                )
            elif self.config["dataset_type"] == "diabetes":
                # Diabetes 130-US real dataset
                from data.diabetes_loader import load_diabetes_data
                print_info("Caricamento Diabetes 130-US (101,766 encounters)...")

                diab_train, diab_test, diab_meta = load_diabetes_data(
                    num_clients=self.config["num_clients"],
                    partition_by_hospital=not is_iid,
                    is_iid=is_iid,
                    alpha=0.5,
                    label_type="binary",
                    test_split=0.2,
                    seed=self.config["seed"],
                    data_path=self.config.get("dataset_path"),
                )

                # Show dataset info
                print_info(f"Diabetes: {diab_meta['total_samples']} campioni, "
                           f"{diab_meta['num_features']} features, "
                           f"label={diab_meta['label_name']}")
                for cid in range(self.config["num_clients"]):
                    train_n = len(diab_train[cid][1])
                    test_n = len(diab_test[cid][1])
                    pos_rate = diab_train[cid][1].mean()
                    print(f"  Client {cid}: {train_n} train, {test_n} test, "
                          f"readmission_rate={pos_rate:.1%}")

                print_info(f"Partizione: {diab_meta['partition_method']}")
                print_info(f"FHIR mapping: {list(diab_meta['fhir_mapping'].keys())}")

                trainer = FederatedTrainer(
                    num_clients=self.config["num_clients"],
                    algorithm=self.config["algorithm"],
                    local_epochs=self.config["local_epochs"],
                    batch_size=self.config["batch_size"],
                    learning_rate=self.config["learning_rate"],
                    mu=self.config["mu"],
                    dp_enabled=self.config["dp_enabled"],
                    dp_epsilon=self.config["dp_epsilon"],
                    dp_clip_norm=self.config["dp_clip_norm"],
                    seed=self.config["seed"],
                    progress_callback=progress_callback,
                    server_lr=self.config.get("server_lr", 0.1),
                    beta1=self.config.get("beta1", 0.9),
                    beta2=self.config.get("beta2", 0.99),
                    tau=self.config.get("tau", 1e-3),
                    external_data=diab_train,
                    external_test_data=diab_test,
                    input_dim=diab_meta["num_features"],
                )
            elif self.config["dataset_type"] == "fhir":
                # FHIR R4 dataset - use FederatedTrainer with external data
                from data.fhir_loader import load_fhir_data
                print_info("Caricamento dati FHIR R4 (ospedali sintetici)...")

                fhir_cfg = {}
                try:
                    from config.config_loader import get_fhir_config
                    fhir_cfg = get_fhir_config()
                except (ImportError, Exception):
                    pass

                fhir_train, fhir_test, fhir_meta = load_fhir_data(
                    num_clients=self.config["num_clients"],
                    samples_per_client=fhir_cfg.get("samples_per_client", 500),
                    hospital_profiles=fhir_cfg.get("profiles"),
                    feature_spec=fhir_cfg.get("feature_spec"),
                    label_name=fhir_cfg.get("label", "mortality_30day"),
                    opt_out_registry_path=fhir_cfg.get("opt_out_registry_path"),
                    purpose=fhir_cfg.get("purpose", "ai_training"),
                    seed=self.config["seed"],
                )

                profiles_assigned = fhir_meta.get("profiles_assigned", [])
                print_info(f"FHIR: {len(fhir_train)} ospedali, "
                           f"{fhir_meta.get('num_features', 10)} features, "
                           f"label={fhir_meta.get('label_name', '?')}")
                for nid in range(len(profiles_assigned)):
                    profile = profiles_assigned[nid]
                    train_n = len(fhir_train[nid][1])
                    test_n = len(fhir_test[nid][1])
                    pos_rate = fhir_train[nid][1].mean()
                    print(f"  Client {nid}: {profile} ({train_n} train, {test_n} test, pos_rate={pos_rate:.1%})")

                if fhir_meta.get("total_opted_out", 0) > 0:
                    print_info(f"  Opt-out EHDS Art.71: {fhir_meta['total_opted_out']} record esclusi")

                trainer = FederatedTrainer(
                    num_clients=self.config["num_clients"],
                    algorithm=self.config["algorithm"],
                    local_epochs=self.config["local_epochs"],
                    batch_size=self.config["batch_size"],
                    learning_rate=self.config["learning_rate"],
                    mu=self.config["mu"],
                    dp_enabled=self.config["dp_enabled"],
                    dp_epsilon=self.config["dp_epsilon"],
                    dp_clip_norm=self.config["dp_clip_norm"],
                    seed=self.config["seed"],
                    progress_callback=progress_callback,
                    server_lr=self.config.get("server_lr", 0.1),
                    beta1=self.config.get("beta1", 0.9),
                    beta2=self.config.get("beta2", 0.99),
                    tau=self.config.get("tau", 1e-3),
                    external_data=fhir_train,
                    external_test_data=fhir_test,
                )
            elif self.config["dataset_type"] == "omop":
                # OMOP-CDM harmonized dataset
                from data.omop_harmonizer import load_omop_data
                print_info("Caricamento dati OMOP-CDM (armonizzazione cross-border)...")

                omop_cfg = {}
                try:
                    from config.config_loader import get_omop_config
                    omop_cfg = get_omop_config()
                except (ImportError, Exception):
                    pass

                omop_train, omop_test, omop_meta = load_omop_data(
                    num_clients=self.config["num_clients"],
                    samples_per_client=omop_cfg.get("samples_per_client", 500),
                    hospital_profiles=omop_cfg.get("profiles"),
                    country_codes=omop_cfg.get("country_codes"),
                    label_name=omop_cfg.get("label", "mortality_30day"),
                    seed=self.config["seed"],
                )

                # Show OMOP info
                countries_assigned = omop_meta.get("countries_assigned", [])
                profiles_assigned = omop_meta.get("profiles_assigned", [])
                coding_systems = omop_meta.get("coding_systems", {})
                print_info(f"OMOP: {len(omop_train)} ospedali, "
                           f"{omop_meta.get('num_features', 0)} features standardizzate")
                for nid in range(len(countries_assigned)):
                    country = countries_assigned[nid]
                    profile = profiles_assigned[nid]
                    cs = coding_systems.get(nid, "?")
                    train_n = len(omop_train[nid][1])
                    test_n = len(omop_test[nid][1])
                    pos_rate = omop_train[nid][1].mean()
                    print(f"  Client {nid}: {country} ({profile}) - {cs} "
                          f"({train_n} train, {test_n} test, pos_rate={pos_rate:.1%})")

                # Show heterogeneity report
                het = omop_meta.get("heterogeneity_report", {})
                if het:
                    print_info(f"Eterogeneita vocabolario:")
                    print(f"  Jaccard distance: raw={het.get('raw_jaccard_mean', 0):.3f} -> "
                          f"OMOP={het.get('omop_jaccard_mean', 0):.3f} "
                          f"(riduzione: {het.get('jaccard_reduction_pct', 0):.1f}%)")
                    if het.get("raw_jsd", 0) > 0:
                        print(f"  Jensen-Shannon Divergence: raw={het['raw_jsd']:.4f} -> OMOP={het.get('omop_jsd', 0):.4f}")

                trainer = FederatedTrainer(
                    num_clients=self.config["num_clients"],
                    algorithm=self.config["algorithm"],
                    local_epochs=self.config["local_epochs"],
                    batch_size=self.config["batch_size"],
                    learning_rate=self.config["learning_rate"],
                    mu=self.config["mu"],
                    dp_enabled=self.config["dp_enabled"],
                    dp_epsilon=self.config["dp_epsilon"],
                    dp_clip_norm=self.config["dp_clip_norm"],
                    seed=self.config["seed"],
                    progress_callback=progress_callback,
                    server_lr=self.config.get("server_lr", 0.1),
                    beta1=self.config.get("beta1", 0.9),
                    beta2=self.config.get("beta2", 0.99),
                    tau=self.config.get("tau", 1e-3),
                    external_data=omop_train,
                    external_test_data=omop_test,
                    input_dim=omop_meta.get("num_features"),
                )
            else:
                # Synthetic tabular dataset - use FederatedTrainer
                print_info(f"Generazione dataset healthcare sintetico ({self.config['num_clients']} client)...")

                trainer = FederatedTrainer(
                    num_clients=self.config["num_clients"],
                    samples_per_client=200,
                    algorithm=self.config["algorithm"],
                    local_epochs=self.config["local_epochs"],
                    batch_size=self.config["batch_size"],
                    learning_rate=self.config["learning_rate"],
                    is_iid=is_iid,
                    alpha=0.5,
                    mu=self.config["mu"],
                    dp_enabled=self.config["dp_enabled"],
                    dp_epsilon=self.config["dp_epsilon"],
                    dp_clip_norm=self.config["dp_clip_norm"],
                    seed=self.config["seed"],
                    progress_callback=progress_callback,
                    # Server optimizer params for FedAdam, FedYogi, FedAdagrad
                    server_lr=self.config.get("server_lr", 0.1),
                    beta1=self.config.get("beta1", 0.9),
                    beta2=self.config.get("beta2", 0.99),
                    tau=self.config.get("tau", 1e-3),
                )

            # Show data distribution
            print_subsection("Distribuzione Dati per Client")
            stats = trainer.get_client_data_stats()
            for cid, stat in stats.items():
                dist = stat["label_distribution"]
                balance = stat["class_balance"]
                print(f"  Client {cid}: {stat['num_samples']} samples, "
                      f"labels={dist}, balance={balance:.2f}")

            # EHDS Permit context (if enabled)
            self._permit_context = None
            if self.config.get("ehds_permit_enabled"):
                from governance.permit_training import create_permit_context
                self._permit_context = create_permit_context(self.config)
                if self._permit_context:
                    self._permit_context.start_session()
                    print()
                    print_success(f"EHDS Permit attivato: {self._permit_context.permit.permit_id}")
                    print_info(f"  Purpose: {self.config['ehds_purpose']}")
                    print_info(f"  Budget: epsilon={self.config['ehds_privacy_budget']}")
                    print_info(f"  Max rounds: {self.config.get('ehds_max_rounds', 'illimitati')}")

            # Data minimization (Art. 44, tabular only)
            minimization_report = None
            if (self.config.get("ehds_data_minimization")
                    and self.config["dataset_type"] != "imaging"
                    and hasattr(trainer, 'client_data')):
                from governance.data_minimization import DataMinimizer
                # Get feature names from metadata if available
                feat_names = None
                if self.config["dataset_type"] == "fhir" and 'fhir_meta' in dir():
                    feat_names = fhir_meta.get("feature_names")
                elif self.config["dataset_type"] == "omop" and 'omop_meta' in dir():
                    feat_names = omop_meta.get("feature_names")

                # Build train_data dict from trainer
                train_dict = {}
                for cid in range(trainer.num_clients):
                    X_c, y_c = trainer.client_data[cid]
                    train_dict[cid] = (X_c.numpy(), y_c.numpy())
                test_dict = None
                if hasattr(trainer, 'client_test_data') and trainer.client_test_data:
                    test_dict = {}
                    for cid in range(trainer.num_clients):
                        if cid in trainer.client_test_data:
                            X_t, y_t = trainer.client_test_data[cid]
                            test_dict[cid] = (X_t.numpy(), y_t.numpy())

                min_train, min_test, minimization_report = DataMinimizer.apply_minimization(
                    train_dict, test_dict,
                    purpose=self.config["ehds_purpose"],
                    feature_names=feat_names,
                )
                print()
                print_info(f"Data minimization: {minimization_report['original_features']} -> "
                           f"{minimization_report['kept_features']} features "
                           f"(-{minimization_report['reduction_pct']}%)")

                # Rebuild trainer data with minimized features
                import torch
                for cid in range(trainer.num_clients):
                    X_min, y_min = min_train[cid]
                    trainer.client_data[cid] = (torch.tensor(X_min, dtype=torch.float32),
                                                 torch.tensor(y_min, dtype=torch.long))
                if min_test:
                    for cid in min_test:
                        X_mt, y_mt = min_test[cid]
                        trainer.client_test_data[cid] = (torch.tensor(X_mt, dtype=torch.float32),
                                                          torch.tensor(y_mt, dtype=torch.long))
                # Update model input dimension
                new_dim = minimization_report['kept_features']
                trainer.model = trainer._create_model(input_dim=new_dim)
                trainer.global_model_state = trainer.model.state_dict()

            print()
            print_info(f"Avvio training {self.config['algorithm']} con PyTorch...")

            start_time = time.time()

            # Run training
            for round_num in range(self.config["num_rounds"]):
                # Pre-round permit validation
                if self._permit_context:
                    eps_per_round = (
                        self.config["dp_epsilon"] / self.config["num_rounds"]
                    ) if self.config["dp_enabled"] else 0.0
                    ok, reason = self._permit_context.validate_round(round_num, eps_per_round)
                    if not ok:
                        print_warning(f"\nTraining interrotto dal permit: {reason}")
                        break

                round_result = trainer.train_round(round_num)

                # Post-round audit logging
                if self._permit_context:
                    eps_spent = (
                        self.config["dp_epsilon"] / self.config["num_rounds"]
                    ) if self.config["dp_enabled"] else 0.0
                    self._permit_context.log_round_completion(round_result, eps_spent)

            elapsed_time = time.time() - start_time

            # Store results with all metrics
            self.results = {
                "config": self.config.copy(),
                "history": [
                    {
                        "round": r.round_num,
                        "global_loss": r.global_loss,
                        "global_accuracy": r.global_acc,
                        "global_f1": r.global_f1,
                        "global_precision": r.global_precision,
                        "global_recall": r.global_recall,
                        "global_auc": r.global_auc,
                        "time_seconds": r.time_seconds,
                        "client_results": [
                            {
                                "client_id": cr.client_id,
                                "loss": cr.train_loss,
                                "accuracy": cr.train_acc,
                                "num_samples": cr.num_samples
                            }
                            for cr in r.client_results
                        ]
                    }
                    for r in trainer.history
                ],
                "final_metrics": {
                    "global_accuracy": trainer.history[-1].global_acc if trainer.history else 0,
                    "global_loss": trainer.history[-1].global_loss if trainer.history else 0,
                    "global_f1": trainer.history[-1].global_f1 if trainer.history else 0,
                    "global_precision": trainer.history[-1].global_precision if trainer.history else 0,
                    "global_recall": trainer.history[-1].global_recall if trainer.history else 0,
                    "global_auc": trainer.history[-1].global_auc if trainer.history else 0,
                },
                "elapsed_time": elapsed_time,
                "data_stats": stats,
            }

            # End EHDS permit session
            if self._permit_context:
                self._permit_context.end_session(
                    total_rounds=len(trainer.history),
                    final_metrics=self.results.get("final_metrics", {}),
                    success=True,
                )
                budget = self._permit_context.get_budget_status()
                print()
                print_info(f"EHDS Budget: {budget['used']:.4f}/{budget['total']:.4f} epsilon "
                           f"({budget['utilization_pct']:.1f}%)")

            # Show final results
            print()
            print_success(f"Training completato in {elapsed_time:.1f} secondi")
            self._display_final_results()

            # Auto-save all outputs
            self._auto_save_training_outputs(elapsed_time)

        except ImportError as e:
            print_error(f"Errore import: {e}")
            print_info("Assicurarsi che PyTorch sia installato: pip install torch")

        except Exception as e:
            print_error(f"Errore durante il training: {e}")
            import traceback
            traceback.print_exc()

        input(f"\n{Style.MUTED}Premi Enter per continuare...{Colors.RESET}")

    def _run_training_simulated(self):
        """Execute simulated training (legacy - uses FLSimulatorV4)."""
        clear_screen()
        print_section("TRAINING SIMULATO (Legacy)")

        display_config_summary(self.config)

        if not confirm("\nAvviare il training simulato?", default=True):
            return

        print()

        try:
            # Import simulator from existing backend
            from dashboard.app_v4 import FLSimulatorV4

            # Map data distribution to heterogeneity type
            is_iid = "IID" in self.config["data_distribution"]
            heterogeneity_type = "iid" if is_iid else "label_skew"

            # Create config dictionary for FLSimulatorV4
            simulator_config = {
                "num_nodes": self.config["num_clients"],
                "num_rounds": self.config["num_rounds"],
                "local_epochs": self.config["local_epochs"],
                "learning_rate": self.config["learning_rate"],
                "algorithm": self.config["algorithm"],
                "random_seed": self.config["seed"],
                "heterogeneity_type": heterogeneity_type,
                "label_skew_alpha": 0.5 if not is_iid else 10.0,  # Higher alpha = more uniform
                "total_samples": 2000,
                "participation_rate": 1.0,
                "clip_norm": self.config["dp_clip_norm"],
                "fedprox_mu": self.config["mu"],  # For FedProx
                # DP settings (simulator uses 'use_dp' and 'epsilon')
                "use_dp": self.config["dp_enabled"],
                "epsilon": self.config["dp_epsilon"] if self.config["dp_enabled"] else 10.0,
            }

            # Create simulator
            print_info("Inizializzazione simulatore...")

            simulator = FLSimulatorV4(simulator_config)

            # Run training with progress bar
            print_info(f"Avvio training {self.config['algorithm']}...")
            print()

            start_time = time.time()

            with FLProgressBar(
                total=self.config["num_rounds"],
                desc="Round di Training",
                unit="round",
                color="green"
            ) as pbar:
                results_list = []
                for round_num in range(self.config["num_rounds"]):
                    # Run single round
                    round_result = simulator.train_round(round_num)
                    results_list.append(round_result)

                    # Update progress (simulator returns 'global_accuracy')
                    pbar.update(1, **{
                        "acc": f"{round_result.get('global_accuracy', 0):.2%}",
                        "priv": f"{round_result.get('privacy_spent', 0):.2f}",
                    })

            elapsed_time = time.time() - start_time

            # Store results
            self.results = {
                "config": self.config.copy(),
                "history": results_list,
                "final_metrics": results_list[-1] if results_list else {},
                "elapsed_time": elapsed_time,
            }

            # Show final results
            print()
            print_success(f"Training completato in {elapsed_time:.1f} secondi")
            self._display_final_results()

        except ImportError as e:
            print_error(f"Impossibile importare il simulatore: {e}")
            print_info("Assicurarsi che dashboard/app_v4.py sia presente")

        except Exception as e:
            print_error(f"Errore durante il training: {e}")

        input(f"\n{Style.MUTED}Premi Enter per continuare...{Colors.RESET}")

    def _display_final_results(self):
        """Display final training results with all metrics."""
        if not self.results:
            print_warning("Nessun risultato disponibile")
            return

        print_subsection("RISULTATI FINALI")

        final = self.results.get("final_metrics", {})

        print(f"\n{Style.TITLE}{'Metrica':<20} {'Valore':<15}{Colors.RESET}")
        print("-" * 40)

        # Global accuracy
        acc = final.get("global_accuracy", 0)
        print(f"  {'Accuracy':<18} {Style.HIGHLIGHT}{acc:.2%}{Colors.RESET}")

        # Global loss
        loss = final.get("global_loss", 0)
        print(f"  {'Loss':<18} {loss:.4f}")

        # F1 Score
        f1 = final.get("global_f1", 0)
        print(f"  {'F1 Score':<18} {f1:.4f}")

        # Precision
        precision = final.get("global_precision", 0)
        print(f"  {'Precision':<18} {precision:.4f}")

        # Recall
        recall = final.get("global_recall", 0)
        print(f"  {'Recall':<18} {recall:.4f}")

        # AUC
        auc = final.get("global_auc", 0)
        print(f"  {'AUC-ROC':<18} {auc:.4f}")

        print("-" * 40)

        # Training time
        elapsed = self.results.get("elapsed_time", 0)
        print(f"  {'Tempo Totale':<18} {elapsed:.1f} s")

        # Number of clients
        print(f"  {'Client':<18} {self.config['num_clients']}")

        # Privacy info if DP enabled
        if self.config["dp_enabled"]:
            print()
            print(f"{Style.WARNING}Differential Privacy:{Colors.RESET}")
            print(f"  Epsilon: {self.config['dp_epsilon']:.4f}")
            print(f"  Delta: {self.config['dp_delta']:.2e}")
            print(f"  Clip Norm: {self.config['dp_clip_norm']:.2f}")

    def _auto_save_training_outputs(self, elapsed_time: float):
        """Automatically save all training outputs."""
        base_dir = Path(__file__).parent.parent.parent / "results"
        base_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create experiment-specific folder with descriptive name
        algo = self.config["algorithm"]
        dp_str = f"_DP{self.config['dp_epsilon']}" if self.config["dp_enabled"] else ""
        dist_short = ("OMOP" if self.config["dataset_type"] == "omop" else
            "FHIR" if self.config["dataset_type"] == "fhir" else
            ("IID" if "IID" in self.config["data_distribution"] else "NonIID"))
        folder_name = f"training_{algo}{dp_str}_{dist_short}_{self.config['num_clients']}clients_{self.config['num_rounds']}rounds_{timestamp}"
        output_dir = base_dir / folder_name
        output_dir.mkdir(exist_ok=True)

        saved_files = []

        print()
        print_subsection("SALVATAGGIO AUTOMATICO RISULTATI")

        # Build specs
        full_specs = {
            "experiment_type": "single_training",
            "timestamp": timestamp,
            "elapsed_time_seconds": elapsed_time,
            "training_config": {
                "algorithm": self.config["algorithm"],
                "num_clients": self.config["num_clients"],
                "num_rounds": self.config["num_rounds"],
                "local_epochs": self.config["local_epochs"],
                "batch_size": self.config["batch_size"],
                "learning_rate": self.config["learning_rate"],
                "data_distribution": self.config["data_distribution"],
                "dp_enabled": self.config["dp_enabled"],
                "seed": self.config["seed"],
            },
            "model_config": {
                "architecture": "HealthcareCNN (GroupNorm, GAP)" if self.config["dataset_type"] == "imaging" else "HealthcareMLP",
                "layers": "4-block CNN (32->64->128->256) + GAP" if self.config["dataset_type"] == "imaging" else "10 -> 64 -> 32 -> 2",
                "total_params": "~500K" if self.config["dataset_type"] == "imaging" else 2946,
                "optimizer": "Adam (wd=1e-5)" if self.config["dataset_type"] == "imaging" else "SGD",
                "loss_function": "CrossEntropyLoss",
                "normalization": "GroupNorm (FL-stable)" if self.config["dataset_type"] == "imaging" else "N/A",
                "data_augmentation": "HFlip + Brightness (vectorized)" if self.config["dataset_type"] == "imaging" else "N/A",
                "evaluation": "Held-out test set (20%)" if self.config["dataset_type"] in ("imaging", "fhir") else "Training data",
                "image_size": "128x128" if self.config["dataset_type"] == "imaging" else "N/A",
                "gradient_clipping": "max_norm=1.0" if self.config["dataset_type"] == "imaging" else "N/A",
            },
        }

        if self.config["dp_enabled"]:
            full_specs["training_config"]["dp_epsilon"] = self.config["dp_epsilon"]
            full_specs["training_config"]["dp_delta"] = self.config["dp_delta"]

        # Add EHDS governance info to specs
        if self.config.get("ehds_permit_enabled") and hasattr(self, '_permit_context') and self._permit_context:
            full_specs["ehds_governance"] = self._permit_context.get_summary_for_specs()

        # 1. Save JSON results
        json_file = output_dir / "results.json"
        export_data = {
            "specs": full_specs,
            "final_metrics": self.results.get("final_metrics", {}),
            "history": self.results.get("history", []),
            "data_stats": self.results.get("data_stats", {}),
        }
        with open(json_file, "w") as f:
            json.dump(export_data, f, indent=2, default=str)
        saved_files.append(("JSON (Risultati)", json_file))

        # 2. Generate convergence plots with all metrics
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import numpy as np

            history = self.results.get("history", [])
            if history:
                rounds = [h["round"] + 1 for h in history]

                # Plot 1: Accuracy and Loss
                fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                accs = [h.get("global_accuracy", 0) for h in history]
                losses = [h.get("global_loss", 0) for h in history]

                axes[0].plot(rounds, accs, 'g-', linewidth=2, marker='o', markersize=3)
                axes[0].set_xlabel("Round")
                axes[0].set_ylabel("Accuracy")
                axes[0].set_title(f"Training Accuracy - {self.config['algorithm']}")
                axes[0].grid(True, alpha=0.3)

                axes[1].plot(rounds, losses, 'b-', linewidth=2, marker='o', markersize=3)
                axes[1].set_xlabel("Round")
                axes[1].set_ylabel("Loss")
                axes[1].set_title(f"Training Loss - {self.config['algorithm']}")
                axes[1].grid(True, alpha=0.3)

                spec_text = f"{self.config['num_clients']} clients | {self.config['num_rounds']} rounds | lr={self.config['learning_rate']}"
                fig.text(0.5, 0.02, spec_text, ha='center', fontsize=9, style='italic')
                plt.tight_layout(rect=[0, 0.05, 1, 1])
                plot_file = output_dir / "plot_accuracy_loss.png"
                plt.savefig(plot_file, dpi=150, bbox_inches="tight")
                plt.close()
                saved_files.append(("PNG (Accuracy/Loss)", plot_file))

                # Plot 2: F1, Precision, Recall
                fig, axes = plt.subplots(1, 3, figsize=(18, 5))
                f1s = [h.get("global_f1", 0) for h in history]
                precs = [h.get("global_precision", 0) for h in history]
                recs = [h.get("global_recall", 0) for h in history]

                axes[0].plot(rounds, f1s, 'r-', linewidth=2, marker='o', markersize=3)
                axes[0].set_xlabel("Round")
                axes[0].set_ylabel("F1")
                axes[0].set_title(f"F1 Score - {self.config['algorithm']}")
                axes[0].grid(True, alpha=0.3)

                axes[1].plot(rounds, precs, 'm-', linewidth=2, marker='o', markersize=3)
                axes[1].set_xlabel("Round")
                axes[1].set_ylabel("Precision")
                axes[1].set_title(f"Precision - {self.config['algorithm']}")
                axes[1].grid(True, alpha=0.3)

                axes[2].plot(rounds, recs, 'c-', linewidth=2, marker='o', markersize=3)
                axes[2].set_xlabel("Round")
                axes[2].set_ylabel("Recall")
                axes[2].set_title(f"Recall - {self.config['algorithm']}")
                axes[2].grid(True, alpha=0.3)

                plt.tight_layout()
                plot_file = output_dir / "plot_f1_precision_recall.png"
                plt.savefig(plot_file, dpi=150, bbox_inches="tight")
                plt.close()
                saved_files.append(("PNG (F1/Prec/Rec)", plot_file))

                # Plot 3: AUC
                fig, ax = plt.subplots(figsize=(10, 5))
                aucs = [h.get("global_auc", 0) for h in history]
                ax.plot(rounds, aucs, 'orange', linewidth=2, marker='o', markersize=3)
                ax.set_xlabel("Round")
                ax.set_ylabel("AUC")
                ax.set_title(f"AUC-ROC - {self.config['algorithm']}")
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                plot_file = output_dir / "plot_auc.png"
                plt.savefig(plot_file, dpi=150, bbox_inches="tight")
                plt.close()
                saved_files.append(("PNG (AUC)", plot_file))

                # Plot 4: Combined metrics
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(rounds, accs, label='Accuracy', linewidth=2)
                ax.plot(rounds, f1s, label='F1', linewidth=2)
                ax.plot(rounds, precs, label='Precision', linewidth=2)
                ax.plot(rounds, recs, label='Recall', linewidth=2)
                ax.plot(rounds, aucs, label='AUC', linewidth=2)
                ax.set_xlabel("Round")
                ax.set_ylabel("Score")
                ax.set_title(f"All Metrics Convergence - {self.config['algorithm']}")
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                plot_file = output_dir / "plot_all_metrics.png"
                plt.savefig(plot_file, dpi=150, bbox_inches="tight")
                plt.close()
                saved_files.append(("PNG (All Metrics)", plot_file))

        except ImportError:
            print_warning("matplotlib non disponibile - grafici non generati")
        except Exception as e:
            print_warning(f"Errore generazione grafici: {e}")

        # 3. Generate CSV with history
        csv_file = output_dir / "history_all_metrics.csv"
        with open(csv_file, "w") as f:
            f.write("round,accuracy,loss,f1,precision,recall,auc,time_seconds\n")
            for h in self.results.get("history", []):
                f.write(f"{h['round']+1},{h.get('global_accuracy',0):.4f},{h.get('global_loss',0):.4f},"
                        f"{h.get('global_f1',0):.4f},{h.get('global_precision',0):.4f},"
                        f"{h.get('global_recall',0):.4f},{h.get('global_auc',0):.4f},"
                        f"{h.get('time_seconds',0):.2f}\n")
        saved_files.append(("CSV (History)", csv_file))

        # 4. Generate summary
        summary_file = output_dir / "summary.txt"
        final = self.results.get("final_metrics", {})
        with open(summary_file, "w") as f:
            f.write(f"FL-EHDS Training Summary\n")
            f.write(f"{'='*50}\n\n")
            f.write(f"Algorithm: {self.config['algorithm']}\n")
            f.write(f"Clients: {self.config['num_clients']}\n")
            f.write(f"Rounds: {self.config['num_rounds']}\n")
            f.write(f"Local Epochs: {self.config['local_epochs']}\n")
            f.write(f"Learning Rate: {self.config['learning_rate']}\n")
            f.write(f"Distribution: {self.config['data_distribution']}\n")
            f.write(f"DP Enabled: {self.config['dp_enabled']}\n")
            if self.config['dp_enabled']:
                f.write(f"DP Epsilon: {self.config['dp_epsilon']}\n")
            f.write(f"\nFinal Metrics:\n")
            f.write(f"  Accuracy: {final.get('global_accuracy', 0):.4f}\n")
            f.write(f"  Loss: {final.get('global_loss', 0):.4f}\n")
            f.write(f"  F1: {final.get('global_f1', 0):.4f}\n")
            f.write(f"  Precision: {final.get('global_precision', 0):.4f}\n")
            f.write(f"  Recall: {final.get('global_recall', 0):.4f}\n")
            f.write(f"  AUC: {final.get('global_auc', 0):.4f}\n")
            f.write(f"\nTraining Time: {elapsed_time:.1f} seconds\n")
        saved_files.append(("TXT (Summary)", summary_file))

        # 5. EHDS Audit Log (if permit enabled)
        if self.config.get("ehds_permit_enabled") and hasattr(self, '_permit_context') and self._permit_context:
            try:
                audit_file = self._permit_context.export_audit_log(output_dir)
                saved_files.append(("JSON (EHDS Audit)", audit_file))
            except Exception as e:
                print_warning(f"Errore salvataggio audit log: {e}")

        # === Show summary of saved files ===
        print()
        print(f"{Style.SUCCESS}=== RISULTATI SALVATI ==={Colors.RESET}")
        print(f"\nDirectory: {output_dir}\n")
        print(f"{Style.TITLE}{'Tipo':<25} {'File':<50}{Colors.RESET}")
        print("-" * 75)
        for file_type, file_path in saved_files:
            print(f"  {file_type:<23} {file_path.name}")
        print("-" * 75)
        print(f"\n{Style.SUCCESS}Totale: {len(saved_files)} file salvati{Colors.RESET}")

    def _show_results(self):
        """Show detailed results."""
        clear_screen()
        print_section("RISULTATI TRAINING")

        if not self.results:
            print_warning("Nessun risultato disponibile. Eseguire prima il training.")
            input(f"\n{Style.MUTED}Premi Enter per continuare...{Colors.RESET}")
            return

        # Show configuration used
        print_subsection("Configurazione utilizzata")
        for key, value in self.results["config"].items():
            print(f"  {key}: {value}")

        # Show data distribution if available
        data_stats = self.results.get("data_stats", {})
        if data_stats:
            print_subsection("Distribuzione Dati per Client")
            for cid, stat in data_stats.items():
                dist = stat.get("label_distribution", {})
                balance = stat.get("class_balance", 1.0)
                print(f"  Client {cid}: {stat.get('num_samples', 0)} samples, "
                      f"labels={dist}, balance={balance:.2f}")

        # Show metrics history (last 10 rounds)
        print_subsection("Storico metriche (ultimi 10 round)")

        history = self.results.get("history", [])[-10:]

        print(f"\n{'Round':<8} {'Loss':<12} {'Accuracy':<12} {'Tempo (s)':<12}")
        print("-" * 48)

        for h in history:
            round_num = h.get('round', 0) + 1
            loss = h.get('global_loss', 0)
            acc = h.get('global_accuracy', 0)
            time_s = h.get('time_seconds', 0)
            print(f"  {round_num:<6} {loss:<12.4f} {acc:<12.2%} {time_s:<12.2f}")

        # Final results
        self._display_final_results()

        input(f"\n{Style.MUTED}Premi Enter per continuare...{Colors.RESET}")

    def _generate_plots(self):
        """Generate convergence plots."""
        clear_screen()
        print_section("GENERA GRAFICI CONVERGENZA")

        if not self.results:
            print_warning("Nessun risultato disponibile. Eseguire prima il training.")
            input(f"\n{Style.MUTED}Premi Enter per continuare...{Colors.RESET}")
            return

        history = self.results.get("history", [])
        if not history:
            print_warning("Nessuno storico disponibile per generare grafici")
            input(f"\n{Style.MUTED}Premi Enter per continuare...{Colors.RESET}")
            return

        try:
            import matplotlib.pyplot as plt
            import numpy as np
            from datetime import datetime
            from pathlib import Path

            # Extract data
            rounds = [h.get("round", i) + 1 for i, h in enumerate(history)]
            losses = [h.get("global_loss", 0) for h in history]
            accuracies = [h.get("global_accuracy", 0) for h in history]

            # Create figure with 2 subplots
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            # Loss plot
            axes[0].plot(rounds, losses, 'b-', linewidth=2, marker='o', markersize=4)
            axes[0].set_xlabel("Round", fontsize=12)
            axes[0].set_ylabel("Loss", fontsize=12)
            axes[0].set_title(f"Training Loss - {self.config['algorithm']}", fontsize=14)
            axes[0].grid(True, alpha=0.3)
            axes[0].set_xlim(1, max(rounds))

            # Accuracy plot
            axes[1].plot(rounds, accuracies, 'g-', linewidth=2, marker='o', markersize=4)
            axes[1].set_xlabel("Round", fontsize=12)
            axes[1].set_ylabel("Accuracy", fontsize=12)
            axes[1].set_title(f"Test Accuracy - {self.config['algorithm']}", fontsize=14)
            axes[1].grid(True, alpha=0.3)
            axes[1].set_xlim(1, max(rounds))
            axes[1].set_ylim(0, 1)

            # Add configuration info
            config_text = (
                f"Algorithm: {self.config['algorithm']} | "
                f"Clients: {self.config['num_clients']} | "
                f"Local Epochs: {self.config['local_epochs']} | "
                f"DP: {'Yes (e=' + str(self.config['dp_epsilon']) + ')' if self.config['dp_enabled'] else 'No'}"
            )
            fig.suptitle(config_text, fontsize=10, y=0.02)

            plt.tight_layout()
            plt.subplots_adjust(bottom=0.15)

            # Save to file
            output_dir = Path(__file__).parent.parent.parent / "results"
            output_dir.mkdir(exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = output_dir / f"training_convergence_{self.config['algorithm']}_{timestamp}.png"
            plt.savefig(filename, dpi=150, bbox_inches="tight")

            print_success(f"Grafico salvato: {filename}")

            # Ask if user wants to open the plot
            if confirm("\nAprire il grafico?", default=True):
                import subprocess
                import sys

                if sys.platform == "darwin":
                    subprocess.run(["open", str(filename)], check=False)
                elif sys.platform == "linux":
                    subprocess.run(["xdg-open", str(filename)], check=False)
                elif sys.platform == "win32":
                    subprocess.run(["start", str(filename)], shell=True, check=False)

            # Also offer ASCII preview in terminal
            if confirm("\nMostrare anteprima ASCII nel terminale?", default=False):
                self._show_ascii_plot(rounds, accuracies, "Accuracy")

            plt.close()

        except ImportError:
            print_error("matplotlib non disponibile")
            print_info("Installare con: pip install matplotlib")
            print()
            print_info("Mostrando anteprima ASCII...")
            self._show_ascii_plot(
                list(range(1, len(history) + 1)),
                [h.get("global_accuracy", 0) for h in history],
                "Accuracy"
            )

        except Exception as e:
            print_error(f"Errore nella generazione del grafico: {e}")
            import traceback
            traceback.print_exc()

        input(f"\n{Style.MUTED}Premi Enter per continuare...{Colors.RESET}")

    def _show_ascii_plot(self, x_values, y_values, title):
        """Show a simple ASCII plot in the terminal."""
        print_subsection(f"GRAFICO ASCII: {title}")

        height = 15
        width = 60

        if not y_values:
            return

        min_y = min(y_values)
        max_y = max(y_values)
        range_y = max_y - min_y if max_y > min_y else 1

        # Create grid
        grid = [[' ' for _ in range(width)] for _ in range(height)]

        # Plot points
        for i, y in enumerate(y_values):
            x_pos = int((i / (len(y_values) - 1)) * (width - 1)) if len(y_values) > 1 else 0
            y_pos = int(((y - min_y) / range_y) * (height - 1)) if range_y > 0 else height // 2
            y_pos = height - 1 - y_pos  # Flip y-axis
            if 0 <= x_pos < width and 0 <= y_pos < height:
                grid[y_pos][x_pos] = '*'

        # Print grid with y-axis labels
        print()
        for i, row in enumerate(grid):
            y_label = max_y - (i / (height - 1)) * range_y if height > 1 else max_y
            print(f"  {y_label:6.2f} |{''.join(row)}|")

        # X-axis
        print(f"         +{'-' * width}+")
        print(f"         1{' ' * (width - 2)}{len(y_values)}")
        print(f"         {'Round':^{width}}")

    def _export_results(self):
        """Export results to file."""
        if not self.results:
            print_warning("Nessun risultato da esportare")
            input(f"\n{Style.MUTED}Premi Enter per continuare...{Colors.RESET}")
            return

        from terminal.screens.output import OutputScreen
        output = OutputScreen()
        output.export_training_results(self.results)

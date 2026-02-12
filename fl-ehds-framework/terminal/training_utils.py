"""
Shared training utilities for FL-EHDS terminal screens.
Extracts common code from training.py, algorithms.py, and guided_comparison.py.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Callable

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import questionary
    HAS_QUESTIONARY = True
except ImportError:
    HAS_QUESTIONARY = False

from terminal.colors import (
    Colors, Style, print_info, print_success, print_warning
)
from terminal.validators import get_int, get_float, get_bool, get_choice, confirm
from terminal.menu import MENU_STYLE


# ─── Shared Constants ─────────────────────────────────────────

FL_ALGORITHMS = [
    "FedAvg", "FedProx", "SCAFFOLD", "FedNova", "FedDyn",
    "FedAdam", "FedYogi", "FedAdagrad", "Per-FedAvg", "Ditto",
]

DATA_DISTRIBUTIONS = [
    "IID (uniforme)",
    "Non-IID (label skew)",
    "Non-IID (quantity skew)",
]

EHDS_PURPOSE_CHOICES = [
    "ai_system_development", "scientific_research",
    "public_health_surveillance", "health_policy",
    "education_training", "personalized_medicine",
    "official_statistics", "patient_safety",
]

EHDS_CATEGORY_MAP = {
    "synthetic": ["ehr"],
    "fhir": ["ehr", "lab_results"],
    "omop": ["ehr", "lab_results"],
    "diabetes": ["ehr", "lab_results"],
    "heart_disease": ["ehr", "lab_results"],
    "imaging": ["imaging"],
}


# ─── 1. get_available_datasets ────────────────────────────────

def get_available_datasets() -> Dict[str, Dict]:
    """Get available imaging datasets from DatasetManager."""
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


# ─── 2. build_dataset_choices ─────────────────────────────────

def build_dataset_choices(available_datasets: Dict[str, Dict]) -> Tuple[List[str], Dict[str, str]]:
    """Build dataset selection UI choices and label-to-key map."""
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

    try:
        diabetes_path = PROJECT_ROOT / "data" / "diabetes" / "diabetic_data.csv"
        if diabetes_path.exists():
            label = "Diabetes 130-US (101K encounters, 130 ospedali, readmission)"
            dataset_choices.insert(3, label)
            dataset_map[label] = "diabetes"
    except Exception:
        pass

    try:
        heart_path = PROJECT_ROOT / "data" / "heart_disease"
        if heart_path.exists():
            label = "Heart Disease UCI (920 pazienti, 4 ospedali, diagnosi)"
            dataset_choices.insert(
                len([c for c in dataset_choices if "img" not in c.lower()]), label
            )
            dataset_map[label] = "heart_disease"
    except Exception:
        pass

    if available_datasets:
        for name, info in available_datasets.items():
            label = f"{name} ({info['samples']:,} img, {info['classes']} classi)"
            dataset_choices.append(label)
            dataset_map[label] = name

    return dataset_choices, dataset_map


# ─── 3. apply_dataset_selection ───────────────────────────────

def apply_dataset_selection(config: Dict[str, Any], selected_key: str,
                            available_datasets: Dict[str, Dict]) -> None:
    """Set config fields based on selected dataset key. Modifies config in-place."""
    if selected_key == "synthetic":
        config["dataset_type"] = "synthetic"
        config["dataset_name"] = None
        config["dataset_path"] = None
        config["learning_rate"] = 0.01
    elif selected_key == "fhir":
        config["dataset_type"] = "fhir"
        config["dataset_name"] = "fhir_synthetic"
        config["dataset_path"] = None
        config["learning_rate"] = 0.01
        try:
            from config.config_loader import get_fhir_config
            fhir_cfg = get_fhir_config()
            profiles = fhir_cfg.get("profiles", [])
            num_c = config.get("num_clients", 5)
            assigned = [profiles[i % len(profiles)] for i in range(num_c)]
            print_info("FHIR R4: dati generati da profili ospedalieri (non-IID naturale)")
            print(f"  Profili: {', '.join(assigned)}")
            print(f"  Pazienti/ospedale: {fhir_cfg.get('samples_per_client', 500)}")
            print(f"  Features: {len(fhir_cfg.get('feature_spec', []))} "
                  f"({fhir_cfg.get('label', 'mortality_30day')})")
        except (ImportError, Exception):
            print_info("FHIR R4: ospedali sintetici con profili diversi")
    elif selected_key == "diabetes":
        config["dataset_type"] = "diabetes"
        config["dataset_name"] = "diabetes_130us"
        config["dataset_path"] = str(PROJECT_ROOT / "data" / "diabetes" / "diabetic_data.csv")
        config["learning_rate"] = 0.01
        print_info("Diabetes 130-US: 101,766 encounter da 130 ospedali USA")
        print_info("  Target: readmission <30 giorni (binario)")
        print_info("  22 features: demographics + diagnosi ICD-9 + farmaci + lab")
        print_info("  Partizione per ospedale (non-IID naturale)")
        print_info("  FHIR R4: Patient, Encounter, Condition, Observation, MedicationStatement")
    elif selected_key == "heart_disease":
        config["dataset_type"] = "heart_disease"
        config["dataset_name"] = "heart_disease_uci"
        config["dataset_path"] = str(PROJECT_ROOT / "data" / "heart_disease")
        config["learning_rate"] = 0.01
        config["num_clients"] = 4
        print_info("Heart Disease UCI: 920 pazienti da 4 ospedali (Cleveland, Hungarian, Swiss, VA)")
        print_info("  Target: presenza malattia cardiaca (binario)")
        print_info("  13 features: demographics + vitali + ECG + stress test")
        print_info("  Partizione per ospedale (non-IID naturale, 4 client)")
        print_info("  FHIR R4: Patient, Observation, Condition, DiagnosticReport")
    elif selected_key == "omop":
        config["dataset_type"] = "omop"
        config["dataset_name"] = "omop_harmonized"
        config["dataset_path"] = None
        config["learning_rate"] = 0.01
        try:
            from config.config_loader import get_omop_config
            omop_cfg = get_omop_config()
            profiles = omop_cfg.get("profiles", [])
            countries = omop_cfg.get("country_codes", [])
            num_c = config.get("num_clients", 5)
            assigned_p = [profiles[i % len(profiles)] for i in range(num_c)]
            assigned_c = [countries[i % len(countries)] for i in range(num_c)]
            from data.omop_harmonizer import COUNTRY_VOCABULARY_PROFILES
            print_info("OMOP-CDM: armonizzazione vocabolari cross-border (non-IID da eterogeneita)")
            print(f"  Pazienti/ospedale: {omop_cfg.get('samples_per_client', 500)}")
            print(f"  ~36 features standardizzate OMOP (temporal windows: 30d/90d/365d)")
            for i in range(num_c):
                vp = COUNTRY_VOCABULARY_PROFILES.get(assigned_c[i], {})
                print(f"  Client {i}: {assigned_c[i]} ({assigned_p[i]}) - "
                      f"{vp.get('coding_system', '?')}")
        except (ImportError, Exception):
            print_info("OMOP-CDM: armonizzazione vocabolari europei cross-border")
    else:
        # Imaging dataset
        config["dataset_type"] = "imaging"
        config["dataset_name"] = selected_key
        config["dataset_path"] = available_datasets[selected_key]["path"]
        config["learning_rate"] = 0.001
        print_info(f"Dataset imaging selezionato: {selected_key}")
        print(f"  Path: {config['dataset_path']}")
        print(f"  Classi: {available_datasets[selected_key]['class_names']}")


# ─── 4. apply_dataset_parameters ─────────────────────────────

def apply_dataset_parameters(config: Dict[str, Any]) -> None:
    """Suggest and apply dataset-specific parameters from config.yaml."""
    if config["dataset_type"] not in ("imaging", "fhir", "omop", "diabetes", "heart_disease"):
        return
    if not config.get("dataset_name"):
        return
    try:
        from config.config_loader import get_dataset_parameters
        ds_params = get_dataset_parameters(config["dataset_name"])
        if not ds_params:
            return
        print()
        print_info(f"Parametri suggeriti per {config['dataset_name']}:")
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
                    config[k] = ds_params[k]
            print_success("Parametri dataset applicati.")
    except (ImportError, Exception):
        pass


# ─── 5. select_imaging_model ─────────────────────────────────

def select_imaging_model(config: Dict[str, Any]) -> None:
    """Interactive model selection for imaging datasets. Modifies config in-place."""
    if config["dataset_type"] != "imaging":
        return
    model_choice = questionary.select(
        "Modello:",
        choices=[
            "ResNet18 (pretrained ImageNet, consigliato)",
            "CNN custom (leggera, ~500K params)",
        ],
        default="ResNet18 (pretrained ImageNet, consigliato)",
    ).ask()
    if model_choice and "CNN" in model_choice:
        config["model_type"] = "cnn"
        print_info("Modello: CNN custom (128x128)")
    else:
        config["model_type"] = "resnet18"
        config["batch_size"] = 16
        print_info("Modello: ResNet18 pretrained (224x224, GroupNorm)")


# ─── 6. configure_ehds_permit ────────────────────────────────

def configure_ehds_permit(config: Dict[str, Any]) -> None:
    """Interactive EHDS Data Permit configuration. Modifies config in-place."""
    from terminal.colors import print_subsection
    print_subsection("EHDS Data Permit (Opzionale)")
    config["ehds_permit_enabled"] = get_bool(
        "Abilitare EHDS Data Permit governance?",
        default=config.get("ehds_permit_enabled", False)
    )

    if not config["ehds_permit_enabled"]:
        return

    if HAS_QUESTIONARY:
        config["ehds_purpose"] = questionary.select(
            "EHDS purpose (Article 53):",
            choices=EHDS_PURPOSE_CHOICES,
            default=config.get("ehds_purpose", "ai_system_development"),
            style=MENU_STYLE,
        ).ask() or config.get("ehds_purpose", "ai_system_development")
    else:
        config["ehds_purpose"] = get_choice(
            "EHDS purpose (Article 53):",
            EHDS_PURPOSE_CHOICES,
            default=config.get("ehds_purpose", "ai_system_development"),
        )

    config["ehds_data_categories"] = EHDS_CATEGORY_MAP.get(
        config["dataset_type"], ["ehr"]
    )
    print_info(f"  Data categories: {', '.join(config['ehds_data_categories'])}")

    # Privacy budget - auto-fill from DP if enabled
    if config.get("dp_enabled"):
        config["ehds_privacy_budget"] = config["dp_epsilon"]
        print_info(f"  Privacy budget da DP epsilon: {config['ehds_privacy_budget']}")
    else:
        config["ehds_privacy_budget"] = get_float(
            "  Privacy budget (epsilon totale)",
            default=config.get("ehds_privacy_budget", 100.0),
            min_val=0.1, max_val=1000.0
        )

    config["ehds_max_rounds"] = get_int(
        "  Massimo round autorizzati",
        default=config.get("num_rounds", 30),
        min_val=1, max_val=10000
    )

    if config["dataset_type"] != "imaging":
        config["ehds_data_minimization"] = get_bool(
            "  Applicare data minimization (Art. 44)?",
            default=config.get("ehds_data_minimization", False)
        )
    else:
        config["ehds_data_minimization"] = False


# ─── 7. create_trainer ───────────────────────────────────────

def create_trainer(
    config: Dict[str, Any],
    algorithm: str,
    seed: int,
    progress_callback: Optional[Callable] = None,
    is_iid: Optional[bool] = None,
    dp_enabled: bool = False,
    dp_epsilon: float = 10.0,
    verbose: bool = True,
) -> Tuple[Any, Dict[str, Any]]:
    """Create FL trainer based on dataset type.

    Returns (trainer, meta_dict) where meta_dict contains dataset-specific
    metadata for display purposes.
    """
    from terminal.fl_trainer import FederatedTrainer, ImageFederatedTrainer

    # Auto-detect is_iid
    if is_iid is None:
        if "is_iid" in config:
            is_iid = config["is_iid"]
        elif "data_distribution" in config:
            is_iid = "IID" in config["data_distribution"]
        else:
            is_iid = False

    # Common params from config
    num_clients = config["num_clients"]
    local_epochs = config["local_epochs"]
    batch_size = config["batch_size"]
    learning_rate = config["learning_rate"]
    mu = config.get("mu", 0.1)
    dp_clip_norm = config.get("dp_clip_norm", 1.0)
    server_lr = config.get("server_lr", 0.1)
    beta1 = config.get("beta1", 0.9)
    beta2 = config.get("beta2", 0.99)
    tau = config.get("tau", 1e-3)

    dataset_type = config["dataset_type"]
    meta = {"dataset_type": dataset_type}

    if dataset_type == "imaging" and config.get("dataset_path"):
        if verbose:
            print_info(f"Caricamento dataset imaging: {config['dataset_name']}")
            print_info(f"Path: {config['dataset_path']}")
            print_info(f"Distribuzione su {num_clients} client...")

        trainer = ImageFederatedTrainer(
            data_dir=config["dataset_path"],
            num_clients=num_clients,
            algorithm=algorithm,
            local_epochs=local_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            is_iid=is_iid,
            alpha=config.get("alpha", 0.5),
            mu=mu,
            dp_enabled=dp_enabled,
            dp_epsilon=dp_epsilon,
            dp_clip_norm=dp_clip_norm,
            seed=seed,
            progress_callback=progress_callback,
            server_lr=server_lr,
            beta1=beta1,
            beta2=beta2,
            tau=tau,
            model_type=config.get("model_type", "resnet18"),
            freeze_backbone=config.get("freeze_backbone", False),
            use_class_weights=config.get("use_class_weights", True),
        )

    elif dataset_type == "heart_disease":
        from data.heart_disease_loader import load_heart_disease_data
        if verbose:
            print_info("Caricamento Heart Disease UCI (920 pazienti, 4 ospedali)...")

        hd_train, hd_test, hd_meta = load_heart_disease_data(
            num_clients=num_clients,
            partition_by_hospital=not is_iid,
            is_iid=is_iid,
            test_split=0.2,
            seed=seed,
            data_path=config.get("dataset_path"),
        )
        meta.update(hd_meta)

        if verbose:
            print_info(f"Heart Disease: {hd_meta['total_samples']} campioni, "
                       f"{hd_meta['num_features']} features, "
                       f"label={hd_meta['label_name']}")
            hosp_assign = hd_meta.get("hospital_assignment", {})
            for cid in range(num_clients):
                train_n = len(hd_train[cid][1])
                test_n = len(hd_test[cid][1])
                pos_rate = hd_train[cid][1].mean()
                hosp = hosp_assign.get(cid, "mixed")
                print(f"  Client {cid} ({hosp}): {train_n} train, {test_n} test, "
                      f"disease_rate={pos_rate:.1%}")

        trainer = FederatedTrainer(
            num_clients=num_clients, algorithm=algorithm,
            local_epochs=local_epochs, batch_size=batch_size,
            learning_rate=learning_rate, mu=mu,
            dp_enabled=dp_enabled, dp_epsilon=dp_epsilon,
            dp_clip_norm=dp_clip_norm, seed=seed,
            progress_callback=progress_callback,
            server_lr=server_lr, beta1=beta1, beta2=beta2, tau=tau,
            external_data=hd_train, external_test_data=hd_test,
            input_dim=hd_meta["num_features"],
        )

    elif dataset_type == "diabetes":
        from data.diabetes_loader import load_diabetes_data
        if verbose:
            print_info("Caricamento Diabetes 130-US (101,766 encounters)...")

        diab_train, diab_test, diab_meta = load_diabetes_data(
            num_clients=num_clients,
            partition_by_hospital=not is_iid,
            is_iid=is_iid,
            alpha=0.5,
            label_type="binary",
            test_split=0.2,
            seed=seed,
            data_path=config.get("dataset_path"),
        )
        meta.update(diab_meta)

        if verbose:
            print_info(f"Diabetes: {diab_meta['total_samples']} campioni, "
                       f"{diab_meta['num_features']} features, "
                       f"label={diab_meta['label_name']}")
            for cid in range(num_clients):
                train_n = len(diab_train[cid][1])
                test_n = len(diab_test[cid][1])
                pos_rate = diab_train[cid][1].mean()
                print(f"  Client {cid}: {train_n} train, {test_n} test, "
                      f"readmission_rate={pos_rate:.1%}")
            print_info(f"Partizione: {diab_meta['partition_method']}")
            print_info(f"FHIR mapping: {list(diab_meta['fhir_mapping'].keys())}")

        trainer = FederatedTrainer(
            num_clients=num_clients, algorithm=algorithm,
            local_epochs=local_epochs, batch_size=batch_size,
            learning_rate=learning_rate, mu=mu,
            dp_enabled=dp_enabled, dp_epsilon=dp_epsilon,
            dp_clip_norm=dp_clip_norm, seed=seed,
            progress_callback=progress_callback,
            server_lr=server_lr, beta1=beta1, beta2=beta2, tau=tau,
            external_data=diab_train, external_test_data=diab_test,
            input_dim=diab_meta["num_features"],
        )

    elif dataset_type == "fhir":
        from data.fhir_loader import load_fhir_data
        if verbose:
            print_info("Caricamento dati FHIR R4 (ospedali sintetici)...")

        fhir_cfg = {}
        try:
            from config.config_loader import get_fhir_config
            fhir_cfg = get_fhir_config()
        except (ImportError, Exception):
            pass

        fhir_train, fhir_test, fhir_meta = load_fhir_data(
            num_clients=num_clients,
            samples_per_client=fhir_cfg.get("samples_per_client", 500),
            hospital_profiles=fhir_cfg.get("profiles"),
            feature_spec=fhir_cfg.get("feature_spec"),
            label_name=fhir_cfg.get("label", "mortality_30day"),
            opt_out_registry_path=fhir_cfg.get("opt_out_registry_path"),
            purpose=fhir_cfg.get("purpose", "ai_training"),
            seed=seed,
        )
        meta.update(fhir_meta)

        if verbose:
            profiles_assigned = fhir_meta.get("profiles_assigned", [])
            print_info(f"FHIR: {len(fhir_train)} ospedali, "
                       f"{fhir_meta.get('num_features', 10)} features, "
                       f"label={fhir_meta.get('label_name', '?')}")
            for nid in range(len(profiles_assigned)):
                profile = profiles_assigned[nid]
                train_n = len(fhir_train[nid][1])
                test_n = len(fhir_test[nid][1])
                pos_rate = fhir_train[nid][1].mean()
                print(f"  Client {nid}: {profile} ({train_n} train, {test_n} test, "
                      f"pos_rate={pos_rate:.1%})")
            if fhir_meta.get("total_opted_out", 0) > 0:
                print_info(f"  Opt-out EHDS Art.71: {fhir_meta['total_opted_out']} record esclusi")

        trainer = FederatedTrainer(
            num_clients=num_clients, algorithm=algorithm,
            local_epochs=local_epochs, batch_size=batch_size,
            learning_rate=learning_rate, mu=mu,
            dp_enabled=dp_enabled, dp_epsilon=dp_epsilon,
            dp_clip_norm=dp_clip_norm, seed=seed,
            progress_callback=progress_callback,
            server_lr=server_lr, beta1=beta1, beta2=beta2, tau=tau,
            external_data=fhir_train, external_test_data=fhir_test,
        )

    elif dataset_type == "omop":
        from data.omop_harmonizer import load_omop_data
        if verbose:
            print_info("Caricamento dati OMOP-CDM (armonizzazione cross-border)...")

        omop_cfg = {}
        try:
            from config.config_loader import get_omop_config
            omop_cfg = get_omop_config()
        except (ImportError, Exception):
            pass

        omop_train, omop_test, omop_meta = load_omop_data(
            num_clients=num_clients,
            samples_per_client=omop_cfg.get("samples_per_client", 500),
            hospital_profiles=omop_cfg.get("profiles"),
            country_codes=omop_cfg.get("country_codes"),
            label_name=omop_cfg.get("label", "mortality_30day"),
            seed=seed,
        )
        meta.update(omop_meta)

        if verbose:
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
            het = omop_meta.get("heterogeneity_report", {})
            if het:
                print_info("Eterogeneita vocabolario:")
                print(f"  Jaccard distance: raw={het.get('raw_jaccard_mean', 0):.3f} -> "
                      f"OMOP={het.get('omop_jaccard_mean', 0):.3f} "
                      f"(riduzione: {het.get('jaccard_reduction_pct', 0):.1f}%)")
                if het.get("raw_jsd", 0) > 0:
                    print(f"  Jensen-Shannon Divergence: raw={het['raw_jsd']:.4f} -> "
                          f"OMOP={het.get('omop_jsd', 0):.4f}")

        trainer = FederatedTrainer(
            num_clients=num_clients, algorithm=algorithm,
            local_epochs=local_epochs, batch_size=batch_size,
            learning_rate=learning_rate, mu=mu,
            dp_enabled=dp_enabled, dp_epsilon=dp_epsilon,
            dp_clip_norm=dp_clip_norm, seed=seed,
            progress_callback=progress_callback,
            server_lr=server_lr, beta1=beta1, beta2=beta2, tau=tau,
            external_data=omop_train, external_test_data=omop_test,
            input_dim=omop_meta.get("num_features"),
        )

    else:
        # Synthetic tabular dataset
        if verbose:
            print_info(f"Generazione dataset healthcare sintetico ({num_clients} client)...")

        trainer = FederatedTrainer(
            num_clients=num_clients,
            samples_per_client=200,
            algorithm=algorithm,
            local_epochs=local_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            is_iid=is_iid,
            alpha=config.get("alpha", 0.5),
            mu=mu,
            dp_enabled=dp_enabled,
            dp_epsilon=dp_epsilon,
            dp_clip_norm=dp_clip_norm,
            seed=seed,
            progress_callback=progress_callback,
            server_lr=server_lr,
            beta1=beta1,
            beta2=beta2,
            tau=tau,
        )

    return trainer, meta


# ─── 8. setup_ehds_permit_context ────────────────────────────

def setup_ehds_permit_context(config: Dict[str, Any]) -> Optional[Any]:
    """Create and start EHDS permit context if enabled. Returns context or None."""
    if not config.get("ehds_permit_enabled"):
        return None
    from governance.permit_training import create_permit_context
    ctx = create_permit_context(config)
    if ctx:
        ctx.start_session()
        print_success(f"EHDS Permit attivato: {ctx.permit.permit_id}")
        print_info(f"  Purpose: {config.get('ehds_purpose', 'N/A')}")
        print_info(f"  Budget: epsilon={config.get('ehds_privacy_budget', 'N/A')}")
        print_info(f"  Max rounds: {config.get('ehds_max_rounds', 'illimitati')}")
    return ctx


# ─── 9. apply_data_minimization ──────────────────────────────

def apply_data_minimization(config: Dict[str, Any], trainer: Any,
                            meta: Dict[str, Any]) -> Optional[Dict]:
    """Apply EHDS Art. 44 data minimization if configured.

    Returns minimization_report or None.
    """
    if not (config.get("ehds_data_minimization")
            and config["dataset_type"] != "imaging"
            and hasattr(trainer, 'client_data')):
        return None

    from governance.data_minimization import DataMinimizer
    import torch

    feat_names = meta.get("feature_names")

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
        purpose=config["ehds_purpose"],
        feature_names=feat_names,
    )
    print()
    print_info(f"Data minimization: {minimization_report['original_features']} -> "
               f"{minimization_report['kept_features']} features "
               f"(-{minimization_report['reduction_pct']}%)")

    # Rebuild trainer data with minimized features
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

    return minimization_report


# ─── 10. calculate_stats ─────────────────────────────────────

def calculate_stats(results: List[Dict]) -> Dict[str, Any]:
    """Calculate mean and std for metrics from a list of result dicts."""
    import numpy as np

    metrics = {}
    for key in ["accuracy", "f1", "precision", "recall", "auc", "loss"]:
        values = [r.get(key, 0) for r in results if r]
        if values:
            metrics[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
            }
    return metrics


# ─── 11. average_history ─────────────────────────────────────

def average_history(histories: List[List[Dict]]) -> List[Dict]:
    """Average multiple history runs with all metrics."""
    import numpy as np

    if not histories or not histories[0]:
        return []

    num_rounds = len(histories[0])
    avg_history = []

    for r in range(num_rounds):
        entry = {"round": r}
        for key in ["accuracy", "loss", "f1", "precision", "recall", "auc"]:
            values = [h[r].get(key, 0) for h in histories if len(h) > r]
            if values:
                entry[key] = float(np.mean(values))
        avg_history.append(entry)

    return avg_history


# ─── 12. generate_comparison_latex ────────────────────────────

def generate_comparison_latex(
    results: Dict[str, Dict],
    specs: Dict[str, Any],
    caption: str = "Federated Learning Algorithm Comparison",
    label: str = "tab:fl_comparison",
) -> str:
    """Generate LaTeX table with full training specifications."""
    lines = []
    lines.append("% FL-EHDS Algorithm Comparison Results")
    if "use_case" in specs:
        lines.append(f"% Use Case: {specs['use_case']['name']}")
    lines.append(f"% Generated: {specs['timestamp']}")
    lines.append(f"% Training time: {specs['elapsed_time_seconds']:.1f} seconds")
    lines.append("")
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{lccccc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Algorithm} & \textbf{Accuracy} & \textbf{F1} & "
                 r"\textbf{Precision} & \textbf{Recall} & \textbf{AUC} \\")
    lines.append(r"\midrule")

    # Find best accuracy for highlighting
    best_acc = max(
        results[algo].get("accuracy", {}).get("mean", 0)
        for algo in results
    )

    for algo, metrics in results.items():
        acc = metrics.get("accuracy", {})
        f1 = metrics.get("f1", {})
        prec = metrics.get("precision", {})
        rec = metrics.get("recall", {})
        auc = metrics.get("auc", {})

        acc_mean = acc.get("mean", 0) * 100
        acc_std = acc.get("std", 0) * 100
        f1_mean = f1.get("mean", 0)
        f1_std = f1.get("std", 0)
        prec_mean = prec.get("mean", 0)
        prec_std = prec.get("std", 0)
        rec_mean = rec.get("mean", 0)
        rec_std = rec.get("std", 0)
        auc_mean = auc.get("mean", 0)
        auc_std = auc.get("std", 0)

        safe_algo = algo.replace("_", r"\_").replace("&", r"\&")

        acc_str = f"{acc_mean:.1f}\\%$\\pm${acc_std:.1f}"
        f1_str = f"{f1_mean:.3f}$\\pm${f1_std:.3f}"
        prec_str = f"{prec_mean:.3f}$\\pm${prec_std:.3f}"
        rec_str = f"{rec_mean:.3f}$\\pm${rec_std:.3f}"
        auc_str = f"{auc_mean:.3f}$\\pm${auc_std:.3f}"

        if abs(acc.get("mean", 0) - best_acc) < 0.001:
            lines.append(f"\\textbf{{{safe_algo}}} & \\textbf{{{acc_str}}} & "
                         f"\\textbf{{{f1_str}}} & \\textbf{{{prec_str}}} & "
                         f"\\textbf{{{rec_str}}} & \\textbf{{{auc_str}}} \\\\")
        else:
            lines.append(f"{safe_algo} & {acc_str} & {f1_str} & "
                         f"{prec_str} & {rec_str} & {auc_str} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append("")
    lines.append(r"\vspace{2mm}")

    # Add full training specs
    tc = specs["training_config"]
    mc = specs["model_config"]
    lines.append(r"\begin{minipage}{\textwidth}")
    lines.append(r"\footnotesize")
    if "use_case" in specs:
        lines.append(f"\\textit{{Use Case: {specs['use_case']['name']}}} \\\\")
    lines.append(r"\textit{Training Configuration:} \\")
    lines.append(f"Clients: {tc['num_clients']} | "
                 f"Rounds: {tc['num_rounds']} | "
                 f"Local Epochs: {tc['local_epochs']} | "
                 f"Batch Size: {tc['batch_size']} | "
                 f"Learning Rate: {tc['learning_rate']} \\\\")
    # Distribution line
    if "is_iid" in tc:
        dist = "IID" if tc['is_iid'] else f"Non-IID (alpha={tc.get('alpha', 0.5)})"
    else:
        dist = tc.get("data_distribution", "Non-IID")
    lines.append(f"Data Distribution: {dist} | "
                 f"Samples/Client: {tc.get('samples_per_client', 200)} | "
                 f"Seeds: {tc['num_seeds']} \\\\")
    lines.append(f"Model: {mc['architecture']} ({mc['layers']}) | "
                 f"Optimizer: {mc['optimizer']} | "
                 f"Loss: {mc['loss_function']}")
    lines.append(r"\end{minipage}")
    lines.append(r"\end{table}")

    return "\n".join(lines)

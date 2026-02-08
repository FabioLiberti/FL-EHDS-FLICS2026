"""
Unified configuration loader for FL-EHDS framework.

Loads config.yaml and provides suggested defaults to both Terminal CLI
and Streamlit Dashboard. Each interface can override values interactively.

Precedence (lowest to highest):
    1. Hardcoded Python fallbacks (always present)
    2. config.yaml training: section (project-level suggestions)
    3. Environment variables FL_EHDS_* (container-level overrides)
    4. User interactive input (terminal prompts / dashboard sidebar)
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


# Search order for config file
_CONFIG_SEARCH_PATHS = [
    os.environ.get("FL_EHDS_CONFIG", ""),
    "config/config.yaml",
    str(Path(__file__).parent / "config.yaml"),
]

_cached_config: Optional[Dict] = None


def _find_config_file() -> Optional[Path]:
    """Find config.yaml from search paths."""
    for path_str in _CONFIG_SEARCH_PATHS:
        if not path_str:
            continue
        p = Path(path_str)
        if p.is_file():
            return p
    return None


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load and cache the full config.yaml.

    Args:
        config_path: Optional explicit path. If None, uses search order.

    Returns:
        Full parsed YAML dict. Returns empty dict if no config found.
    """
    global _cached_config

    if _cached_config is not None and config_path is None:
        return _cached_config

    if config_path:
        p = Path(config_path)
    else:
        p = _find_config_file()

    if p is None or not p.is_file():
        _cached_config = {}
        return _cached_config

    with open(p, "r", encoding="utf-8") as f:
        _cached_config = yaml.safe_load(f) or {}

    return _cached_config


def get_training_defaults() -> Dict[str, Any]:
    """
    Return a flat dict of training defaults.

    Keys match exactly those used by TrainingScreen._default_config()
    and the dashboard sidebar configuration. Values come from config.yaml
    with hardcoded fallbacks if YAML is unavailable.

    Returns:
        Dict with all training configuration keys.
    """
    cfg = load_config()

    # Navigate into training section
    training = cfg.get("training", {})
    federated = training.get("federated", {})
    privacy = training.get("privacy", {})
    data = training.get("data", {})
    server_opt = federated.get("server_optimizer", {})

    # Hardcoded fallback defaults (conservative, from terminal)
    defaults = {
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
        "img_size": 128,
        "alpha": 0.5,
    }

    # Override from YAML (only keys that exist)
    yaml_mapping = {
        "algorithm": federated.get("algorithm"),
        "num_clients": federated.get("num_clients"),
        "num_rounds": federated.get("num_rounds"),
        "local_epochs": federated.get("local_epochs"),
        "batch_size": federated.get("batch_size"),
        "learning_rate": federated.get("learning_rate"),
        "dp_enabled": privacy.get("enabled"),
        "dp_epsilon": privacy.get("epsilon"),
        "dp_delta": privacy.get("delta"),
        "dp_clip_norm": privacy.get("clip_norm"),
        "mu": federated.get("mu"),
        "seed": training.get("seed"),
        "server_lr": server_opt.get("lr"),
        "beta1": server_opt.get("beta1"),
        "beta2": server_opt.get("beta2"),
        "tau": server_opt.get("tau"),
        "dataset_type": data.get("type"),
        "dataset_path": data.get("path"),
        "img_size": data.get("img_size"),
        "alpha": data.get("alpha"),
    }

    for key, value in yaml_mapping.items():
        if value is not None:
            defaults[key] = value

    # Map data distribution from YAML format
    dist = data.get("distribution")
    if dist == "iid":
        defaults["data_distribution"] = "IID"
    elif dist == "non_iid":
        defaults["data_distribution"] = "Non-IID (label skew)"

    # Override from environment variables
    env_mapping = {
        "FL_EHDS_ALGORITHM": ("algorithm", str),
        "FL_EHDS_NUM_CLIENTS": ("num_clients", int),
        "FL_EHDS_NUM_ROUNDS": ("num_rounds", int),
        "FL_EHDS_LOCAL_EPOCHS": ("local_epochs", int),
        "FL_EHDS_BATCH_SIZE": ("batch_size", int),
        "FL_EHDS_LEARNING_RATE": ("learning_rate", float),
        "FL_EHDS_DP_ENABLED": ("dp_enabled", lambda x: x.lower() in ("1", "true", "yes")),
        "FL_EHDS_DP_EPSILON": ("dp_epsilon", float),
        "FL_EHDS_SEED": ("seed", int),
        "FL_EHDS_IMG_SIZE": ("img_size", int),
        "FL_EHDS_ALPHA": ("alpha", float),
    }

    for env_var, (key, converter) in env_mapping.items():
        val = os.environ.get(env_var)
        if val is not None:
            try:
                defaults[key] = converter(val)
            except (ValueError, TypeError):
                pass

    return defaults


def get_dashboard_defaults() -> Dict[str, Any]:
    """
    Return defaults formatted for dashboard sidebar widgets.

    Maps flat keys to the dashboard's expected config dict keys.

    Returns:
        Dict with dashboard-compatible keys.
    """
    d = get_training_defaults()
    return {
        "num_nodes": d["num_clients"],
        "total_samples": 2000,
        "algorithm": d["algorithm"],
        "fedprox_mu": d["mu"],
        "server_lr": d["server_lr"],
        "beta1": d["beta1"],
        "beta2": d["beta2"],
        "tau": d["tau"],
        "num_rounds": d["num_rounds"],
        "local_epochs": d["local_epochs"],
        "learning_rate": d["learning_rate"],
        "use_dp": d["dp_enabled"],
        "epsilon": d["dp_epsilon"],
        "delta": d["dp_delta"],
        "clip_norm": d["dp_clip_norm"],
        "random_seed": d["seed"],
        "img_size": d["img_size"],
    }


def get_full_config() -> Dict[str, Any]:
    """
    Return the complete parsed config.yaml as a nested dict.

    Returns:
        Full YAML configuration dict.
    """
    return load_config()


def reload_config():
    """Force reload of config (clears cache)."""
    global _cached_config
    _cached_config = None


def get_algorithm_profile(algorithm: str) -> Dict[str, Any]:
    """Get recommended hyperparameters for a specific FL algorithm."""
    cfg = load_config()
    profiles = cfg.get("algorithm_profiles", {})
    return profiles.get(algorithm, {})


def get_clinical_use_cases() -> Dict[str, Dict]:
    """Get all clinical use case profiles from config."""
    cfg = load_config()
    return cfg.get("clinical_use_cases", {})


def get_dataset_parameters(dataset_name: Optional[str] = None) -> Dict:
    """Get dataset-specific optimal parameters. None = all datasets."""
    cfg = load_config()
    params = cfg.get("dataset_parameters", {})
    if dataset_name is None:
        return params
    return params.get(dataset_name, {})


def get_fhir_config() -> Dict[str, Any]:
    """Get FHIR pipeline configuration."""
    cfg = load_config()
    fhir = cfg.get("fhir", {})
    return {
        "profiles": fhir.get("default_profiles",
            ["general", "cardiac", "pediatric", "geriatric", "oncology"]),
        "samples_per_client": fhir.get("samples_per_client", 500),
        "feature_spec": fhir.get("feature_spec", [
            "age", "gender", "bmi", "systolic_bp", "diastolic_bp",
            "heart_rate", "glucose", "cholesterol",
            "num_conditions", "num_medications"]),
        "label": fhir.get("label", "mortality_30day"),
        "opt_out_registry_path": fhir.get("opt_out", {}).get("registry_path"),
        "purpose": fhir.get("opt_out", {}).get("purpose", "ai_training"),
    }


def get_omop_config() -> Dict[str, Any]:
    """Get OMOP CDM harmonization configuration."""
    cfg = load_config()
    omop = cfg.get("omop", {})
    return {
        "country_codes": omop.get("country_codes", ["DE", "FR", "IT", "ES", "NL"]),
        "profiles": omop.get("profiles",
            ["general", "cardiac", "pediatric", "geriatric", "oncology"]),
        "samples_per_client": omop.get("samples_per_client", 500),
        "normalization": omop.get("features", {}).get("normalization", "zscore"),
        "label": omop.get("label", "mortality_30day"),
        "compute_jaccard": omop.get("heterogeneity", {}).get("compute_jaccard", True),
        "compute_jsd": omop.get("heterogeneity", {}).get("compute_jsd", True),
    }


def get_jurisdiction_privacy_config() -> Dict[str, Any]:
    """Get jurisdiction-level privacy budget configuration."""
    cfg = load_config()
    cb = cfg.get("cross_border", {})
    jp = cb.get("jurisdiction_privacy", {})
    return {
        "enabled": jp.get("enabled", False),
        "noise_strategy": jp.get("noise_strategy", "global"),
        "accountant_type": jp.get("accountant_type", "rdp"),
        "hospital_allocation_fraction": jp.get("hospital_allocation_fraction", 1.0),
        "min_active_clients": jp.get("min_active_clients", 2),
        "country_overrides": jp.get("country_overrides", {}),
    }


def get_ihe_config() -> Dict[str, Any]:
    """Get IHE Integration Profiles configuration for cross-border FL."""
    cfg = load_config()
    cb = cfg.get("cross_border", {})
    ihe = cb.get("ihe", {})
    return {
        "enabled": ihe.get("enabled", False),
        "atna_audit": ihe.get("atna_audit", True),
        "xds_imaging_simulation": ihe.get("xds_imaging_simulation", True),
        "consistent_time": ihe.get("consistent_time", True),
        "mtls_simulation": ihe.get("mtls_simulation", True),
        "xua_authentication": ihe.get("xua_authentication", True),
        "bppc_consent_check": ihe.get("bppc_consent_check", True),
        "ntp_server": ihe.get("ntp_server", "ntp.ehds.europa.eu"),
        "max_clock_drift_ms": ihe.get("max_clock_drift_ms", 50),
        "certificate_authority": ihe.get("certificate_authority", "EHDS-CA"),
        "certificate_validity_days": ihe.get("certificate_validity_days", 365),
    }


def get_data_quality_config() -> Dict[str, Any]:
    """Get Data Quality Framework configuration (EHDS Art. 69)."""
    cfg = load_config()
    cb = cfg.get("cross_border", {})
    dq = cb.get("data_quality", {})
    return {
        "enabled": dq.get("enabled", False),
        "alpha": dq.get("alpha", 1.0),
        "weight_completeness": dq.get("weight_completeness", 0.20),
        "weight_accuracy": dq.get("weight_accuracy", 0.25),
        "weight_uniqueness": dq.get("weight_uniqueness", 0.15),
        "weight_diversity": dq.get("weight_diversity", 0.20),
        "weight_consistency": dq.get("weight_consistency", 0.20),
        "gold_threshold": dq.get("gold_threshold", 0.85),
        "silver_threshold": dq.get("silver_threshold", 0.70),
        "bronze_threshold": dq.get("bronze_threshold", 0.55),
        "ks_threshold": dq.get("ks_threshold", 0.05),
        "missing_threshold": dq.get("missing_threshold", 0.30),
        "entropy_threshold": dq.get("entropy_threshold", 0.30),
        "iqr_multiplier": dq.get("iqr_multiplier", 3.0),
    }


def get_myhealth_eu_config() -> Dict[str, Any]:
    """Get MyHealth@EU / NCPeH integration configuration (EHDS Art. 5-12)."""
    cfg = load_config()
    cb = cfg.get("cross_border", {})
    mh = cb.get("myhealth_eu", {})
    return {
        "enabled": mh.get("enabled", False),
        "hierarchical_aggregation": mh.get("hierarchical_aggregation", True),
        "simulate_ncp_latency": mh.get("simulate_ncp_latency", True),
        "bandwidth_model": mh.get("bandwidth_model", "realistic"),
        "central_node": mh.get("central_node", "BE"),
        "patient_summary_enabled": mh.get("patient_summary_enabled", True),
        "eprescription_enabled": mh.get("eprescription_enabled", True),
        "ncp_weight_strategy": mh.get("ncp_weight_strategy", "sample_proportional"),
        "track_communication_cost": mh.get("track_communication_cost", True),
    }


def get_governance_training_config() -> Dict[str, Any]:
    """Get governance training integration configuration."""
    cfg = load_config()
    gov = cfg.get("governance", {})
    integration = gov.get("training_integration", {})
    minimization = integration.get("data_minimization", {})
    return {
        "enabled": integration.get("enabled", False),
        "default_purpose": integration.get("default_purpose", "ai_system_development"),
        "default_privacy_budget": integration.get("default_privacy_budget", 100.0),
        "minimization_enabled": minimization.get("enabled", False),
        "importance_threshold": minimization.get("importance_threshold", 0.01),
    }

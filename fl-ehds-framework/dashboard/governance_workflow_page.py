"""
EHDS Governance Workflow Page for FL-EHDS Dashboard.

Interactive EHDS governance lifecycle with four sub-tabs:
1. Lifecycle Config - Country/hospital selection, Art. 53 purpose, privacy budget
2. Lifecycle Execution - Run GovernanceLifecycleBridge.pre_training() with progress
3. Compliance Dashboard - EHDSComplianceReport, per-article status, audit trail
4. Fee Analysis - FeeModelBridge cost visualization and budget optimization

Uses real governance module instances for all operations.

Author: Fabio Liberti
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

import sys
from pathlib import Path

_framework_root = str(Path(__file__).parent.parent)
if _framework_root not in sys.path:
    sys.path.insert(0, _framework_root)

from core.models import (
    DataCategory,
    DataPermit,
    PermitPurpose,
    PermitStatus,
)

# Purpose and category mappings
PURPOSE_OPTIONS = {
    "scientific_research": "Ricerca Scientifica (Art. 53.1a)",
    "public_health_surveillance": "Sorveglianza Sanitaria (Art. 53.1b)",
    "health_policy": "Politica Sanitaria (Art. 53.1c)",
    "education_training": "Formazione/Istruzione (Art. 53.1d)",
    "ai_system_development": "Sviluppo Sistemi AI (Art. 53.1e)",
    "personalized_medicine": "Medicina Personalizzata (Art. 53.1f)",
    "official_statistics": "Statistiche Ufficiali (Art. 53.1g)",
    "patient_safety": "Sicurezza del Paziente (Art. 53.1h)",
}

COUNTRY_HOSPITALS = {
    "IT": [
        {"id": 1, "name": "Ospedale Bambino Gesu - Roma"},
        {"id": 2, "name": "Policlinico Gemelli - Roma"},
        {"id": 3, "name": "San Raffaele - Milano"},
    ],
    "DE": [
        {"id": 4, "name": "Charite - Berlin"},
        {"id": 5, "name": "LMU Klinikum - Munchen"},
    ],
    "FR": [
        {"id": 6, "name": "AP-HP Necker - Paris"},
        {"id": 7, "name": "CHU Lyon"},
    ],
    "ES": [
        {"id": 8, "name": "Hospital La Paz - Madrid"},
        {"id": 9, "name": "Hospital Clinic - Barcelona"},
    ],
    "NL": [
        {"id": 10, "name": "UMC Utrecht"},
    ],
    "BE": [
        {"id": 11, "name": "UZ Leuven"},
    ],
    "AT": [
        {"id": 12, "name": "AKH Wien"},
    ],
}


class _SimpleCountryProfile:
    """Minimal country fee profile for FeeModelBridge compatibility (Art. 42)."""

    _PROFILES = {
        "IT": (300, 0.03, 30, 0.06),
        "DE": (500, 0.05, 50, 0.10),
        "FR": (400, 0.04, 40, 0.08),
        "ES": (250, 0.03, 25, 0.05),
        "NL": (350, 0.04, 35, 0.07),
        "BE": (300, 0.03, 30, 0.06),
        "AT": (350, 0.04, 35, 0.07),
    }

    def __init__(self, country_code: str):
        base, per_rec, per_round, per_mb = self._PROFILES.get(
            country_code, (300, 0.03, 30, 0.06)
        )
        self.fee_base_eur = base
        self.fee_per_record_eur = per_rec
        self.fee_per_round_eur = per_round
        self.fee_per_mb_eur = per_mb


class SimpleHospital:
    """Lightweight hospital representation for governance bridge."""

    def __init__(self, hospital_id: int, name: str, country_code: str):
        self.hospital_id = hospital_id
        self.name = name
        self.country_code = country_code
        self.country_profile = _SimpleCountryProfile(country_code)
        self.num_samples = 500
        self.num_samples_after_optout = None


def _init_governance_state():
    """Initialize session state for governance workflow."""
    if "gov_config" not in st.session_state:
        st.session_state.gov_config = {
            "countries": ["IT"],
            "hospitals": [],
            "purpose": "scientific_research",
            "global_epsilon": 10.0,
            "num_rounds": 30,
            "dataset_type": "synthetic",
            "hdab_auth_method": "oauth2",
            "permit_validity_days": 365,
            "data_minimization_enabled": False,
            "importance_threshold": 0.01,
            "fee_model_enabled": False,
            "secure_processing_enabled": False,
            "data_quality_enabled": False,
            "hdab_routing_enabled": False,
        }
    if "gov_bridge" not in st.session_state:
        st.session_state.gov_bridge = None
    if "gov_pre_training_result" not in st.session_state:
        st.session_state.gov_pre_training_result = None
    if "gov_compliance_report" not in st.session_state:
        st.session_state.gov_compliance_report = None
    if "gov_fee_bridge" not in st.session_state:
        st.session_state.gov_fee_bridge = None


def _render_lifecycle_config():
    """Sub-tab: Configure governance lifecycle parameters."""
    st.markdown("##### Configurazione Governance Lifecycle")
    st.markdown(
        "Configura i parametri del ciclo di vita governance EHDS prima del training. "
        "Ogni parametro corrisponde a un articolo specifico della EHDS Regulation."
    )

    config = st.session_state.gov_config

    # Country and Hospital Selection
    st.markdown("**Paesi e Ospedali Partecipanti**")
    col1, col2 = st.columns([1, 2])

    with col1:
        selected_countries = st.multiselect(
            "Paesi EU",
            options=list(COUNTRY_HOSPITALS.keys()),
            default=config["countries"],
            key="gov_countries",
        )
        config["countries"] = selected_countries

    with col2:
        available_hospitals = []
        for cc in selected_countries:
            for h in COUNTRY_HOSPITALS.get(cc, []):
                available_hospitals.append(
                    f"[{cc}] {h['name']} (ID: {h['id']})"
                )

        if available_hospitals:
            selected_hospitals = st.multiselect(
                "Ospedali",
                options=available_hospitals,
                default=available_hospitals[:3] if len(available_hospitals) >= 3 else available_hospitals,
                key="gov_hospitals",
            )
            # Parse hospital objects
            hospitals = []
            for h_str in selected_hospitals:
                cc = h_str[1:3]
                for h in COUNTRY_HOSPITALS.get(cc, []):
                    if h["name"] in h_str:
                        hospitals.append(SimpleHospital(h["id"], h["name"], cc))
            config["hospitals"] = hospitals
        else:
            st.info("Seleziona almeno un paese per vedere gli ospedali disponibili.")
            config["hospitals"] = []

    st.markdown("---")

    # Purpose and Privacy
    st.markdown("**Finalita e Privacy (Art. 53 + DP)**")
    purp_col1, purp_col2, purp_col3 = st.columns(3)

    with purp_col1:
        purpose_keys = list(PURPOSE_OPTIONS.keys())
        purpose_labels = list(PURPOSE_OPTIONS.values())
        purpose_idx = st.selectbox(
            "Finalita EHDS",
            options=range(len(purpose_keys)),
            format_func=lambda i: purpose_labels[i],
            index=purpose_keys.index(config["purpose"]),
            key="gov_purpose",
        )
        config["purpose"] = purpose_keys[purpose_idx]

    with purp_col2:
        config["global_epsilon"] = st.number_input(
            "Privacy Budget (epsilon)",
            min_value=0.1, max_value=100.0,
            value=config["global_epsilon"],
            step=0.5,
            key="gov_epsilon",
        )

    with purp_col3:
        config["num_rounds"] = st.number_input(
            "Numero Round FL",
            min_value=5, max_value=200,
            value=config["num_rounds"],
            key="gov_rounds",
        )

    st.markdown("---")

    # Sub-bridges toggles
    st.markdown("**Moduli Governance (opzionali)**")
    mod_cols = st.columns(3)

    with mod_cols[0]:
        config["data_minimization_enabled"] = st.checkbox(
            "Data Minimization (Art. 44)",
            value=config["data_minimization_enabled"],
            help="Filtra features non necessarie per la finalita dichiarata.",
            key="gov_minimization",
        )
        config["secure_processing_enabled"] = st.checkbox(
            "Secure Processing (Art. 50)",
            value=config["secure_processing_enabled"],
            help="Simula TEE, watermarking e time guard.",
            key="gov_secure",
        )

    with mod_cols[1]:
        config["fee_model_enabled"] = st.checkbox(
            "Fee Model (Art. 42)",
            value=config["fee_model_enabled"],
            help="Calcola costi di accesso ai dati per ogni ospedale.",
            key="gov_fee",
        )
        config["data_quality_enabled"] = st.checkbox(
            "Data Quality (Art. 69)",
            value=config["data_quality_enabled"],
            help="Valuta qualita dati e assegna label GOLD/SILVER/BRONZE.",
            key="gov_quality",
        )

    with mod_cols[2]:
        config["hdab_routing_enabled"] = st.checkbox(
            "HDAB Routing (Art. 57-58)",
            value=config["hdab_routing_enabled"],
            help="Single Application Point e joint approval per cross-border.",
            key="gov_routing",
        )

        config["hdab_auth_method"] = st.selectbox(
            "Metodo Autenticazione HDAB",
            options=["oauth2", "mtls", "api_key"],
            index=0,
            key="gov_auth",
        )

    st.markdown("---")

    # Dataset type
    config["dataset_type"] = st.selectbox(
        "Tipo Dataset",
        options=["synthetic", "tabular", "imaging", "genomic"],
        index=0,
        key="gov_dataset_type",
    )

    # Summary
    with st.expander("Riepilogo Configurazione", expanded=False):
        st.json(
            {
                "countries": config["countries"],
                "hospitals": len(config.get("hospitals", [])),
                "purpose": config["purpose"],
                "epsilon": config["global_epsilon"],
                "rounds": config["num_rounds"],
                "minimization": config["data_minimization_enabled"],
                "secure_processing": config["secure_processing_enabled"],
                "fee_model": config["fee_model_enabled"],
                "data_quality": config["data_quality_enabled"],
                "hdab_routing": config["hdab_routing_enabled"],
                "auth_method": config["hdab_auth_method"],
            }
        )


def _render_lifecycle_execution():
    """Sub-tab: Execute pre-training governance lifecycle."""
    st.markdown("##### Esecuzione Lifecycle Governance")
    st.markdown(
        "Esegui la fase pre-training del ciclo di vita governance EHDS: "
        "connessione HDAB, emissione permit, minimizzazione dati."
    )

    config = st.session_state.gov_config
    hospitals = config.get("hospitals", [])

    if not hospitals:
        st.warning(
            "Nessun ospedale selezionato. Configura i parametri nella tab "
            "'Configurazione Lifecycle'."
        )
        return

    countries = list(set(h.country_code for h in hospitals))

    # Pre-training execution
    if st.button("Esegui Pre-Training Governance", type="primary", key="gov_execute"):
        from governance.governance_lifecycle import GovernanceLifecycleBridge

        progress = st.progress(0)
        status = st.empty()
        detail_container = st.empty()

        try:
            # Step 1: Create bridge
            status.markdown("**Step 1/4:** Inizializzazione GovernanceLifecycleBridge...")
            progress.progress(0.1)

            bridge = GovernanceLifecycleBridge(
                hospitals=hospitals,
                countries=countries,
                purpose=config["purpose"],
                global_epsilon=config["global_epsilon"],
                num_rounds=config["num_rounds"],
                config=config,
                seed=42,
            )

            # Step 2: Pre-training
            status.markdown("**Step 2/4:** Connessione HDAB e richiesta permit...")
            progress.progress(0.3)

            result = bridge.pre_training()

            # Step 3: Show results
            status.markdown("**Step 3/4:** Verifica risultati...")
            progress.progress(0.7)

            # Store bridge and result in session state
            st.session_state.gov_bridge = bridge
            st.session_state.gov_pre_training_result = result

            # Step 4: Display results
            progress.progress(1.0)
            status.success("Pre-training governance completato!")

            # Show detailed results
            _display_pre_training_results(result, detail_container)

        except Exception as e:
            status.error(f"Errore durante l'esecuzione: {e}")
            progress.progress(0)

    # Show previous results if available
    if st.session_state.gov_pre_training_result is not None:
        st.markdown("---")
        st.markdown("##### Risultati Precedenti")
        _display_pre_training_results(
            st.session_state.gov_pre_training_result, st.container()
        )


def _display_pre_training_results(result: Dict, container):
    """Display pre-training governance results."""
    with container:
        # HDAB connections
        hdab_status = result.get("hdab_status", {})
        if hdab_status:
            st.markdown("**Connessioni HDAB:**")
            hdab_cols = st.columns(len(hdab_status))
            for i, (cc, connected) in enumerate(hdab_status.items()):
                with hdab_cols[i]:
                    if connected:
                        st.success(f"HDAB-{cc}: Connesso")
                    else:
                        st.error(f"HDAB-{cc}: Fallito")

        # Permits
        permits = result.get("permits", {})
        if permits:
            st.markdown("**Permit Emessi:**")
            permit_rows = []
            for cc, p_info in permits.items():
                permit_rows.append({
                    "Paese": cc,
                    "Permit ID": p_info.get("permit_id", "N/A")[:16],
                    "Stato": p_info.get("status", "N/A"),
                    "Finalita": p_info.get("purpose", "N/A"),
                    "Scadenza": p_info.get("valid_until", "N/A")[:10],
                })
            st.dataframe(pd.DataFrame(permit_rows), use_container_width=True)

        # Minimization report
        min_report = result.get("minimization_report")
        if min_report:
            st.markdown("**Data Minimization (Art. 44):**")
            min_cols = st.columns(3)
            with min_cols[0]:
                st.metric("Features Originali", min_report["original_features"])
            with min_cols[1]:
                st.metric("Features Mantenute", min_report["kept_features"])
            with min_cols[2]:
                st.metric("Riduzione", f"{min_report['reduction_pct']}%")

        # Purpose violations
        violations = result.get("purpose_violations", [])
        if violations:
            st.warning(f"**Violazioni Riscontrate ({len(violations)}):**")
            for v in violations:
                st.markdown(f"- {v}")
        elif not violations and permits:
            st.success("Nessuna violazione riscontrata. Training autorizzato.")


def _render_compliance_dashboard():
    """Sub-tab: EHDS Compliance Report visualization."""
    st.markdown("##### Dashboard Compliance EHDS")
    st.markdown(
        "Genera e visualizza il report di conformita EHDS per-articolo "
        "basato sulla configurazione e sulle operazioni governance effettuate."
    )

    config = st.session_state.gov_config
    bridge = st.session_state.gov_bridge

    if bridge is None:
        st.info(
            "Esegui prima il pre-training governance nella tab 'Esecuzione Lifecycle' "
            "per generare il report di compliance."
        )

        # Show standalone compliance config
        if st.button("Genera Report Compliance (standalone)", key="gov_compliance_standalone"):
            _generate_standalone_compliance(config)
        return

    # Generate compliance from bridge
    if st.button("Genera Report Compliance", type="primary", key="gov_compliance_gen"):
        _generate_compliance_from_bridge(bridge, config)

    # Display existing report
    report = st.session_state.gov_compliance_report
    if report is not None:
        _display_compliance_report(report)


def _generate_standalone_compliance(config: Dict):
    """Generate compliance report without actual training."""
    from governance.ehds_compliance_report import (
        ArticleAssessment,
        ComplianceStatus,
        EHDSComplianceReport,
        ARTICLE_REGISTRY,
    )

    report = EHDSComplianceReport()
    report.generated_at = datetime.now().isoformat(timespec="seconds")
    report.scenario_label = (
        f"{len(config['countries'])} countries, "
        f"{len(config.get('hospitals', []))} hospitals, "
        f"purpose={config['purpose']}"
    )
    report.training_config = config

    # Assess each article based on config only
    for chapter, art_id, title, _ in ARTICLE_REGISTRY:
        status = ComplianceStatus.NOT_ASSESSED
        evidence = "Non valutato (training non eseguito)"

        # Purposes (Art. 33)
        if art_id == "Art. 33":
            from governance.ehds_compliance_report import ARTICLE_53_PURPOSES
            if config["purpose"] in ARTICLE_53_PURPOSES:
                status = ComplianceStatus.COMPLIANT
                evidence = f"Purpose: {config['purpose']} (valido Art. 53)"
            else:
                status = ComplianceStatus.NON_COMPLIANT
                evidence = f"Purpose: {config['purpose']} NON in Art. 53"

        # Data categories (Art. 34)
        elif art_id == "Art. 34":
            status = ComplianceStatus.COMPLIANT
            evidence = f"Dataset type: {config['dataset_type']}"

        # Fee model (Art. 42)
        elif art_id == "Art. 42":
            if config.get("fee_model_enabled"):
                status = ComplianceStatus.COMPLIANT
                evidence = "Fee model abilitato"
            else:
                status = ComplianceStatus.NOT_ASSESSED
                evidence = "Fee model non abilitato"

        # Data minimization (Art. 44)
        elif art_id == "Art. 44":
            if config.get("data_minimization_enabled"):
                status = ComplianceStatus.COMPLIANT
                evidence = "Minimizzazione dati abilitata"
            else:
                status = ComplianceStatus.NOT_ASSESSED
                evidence = "Minimizzazione non abilitata"

        # Secure processing (Art. 50)
        elif art_id == "Art. 50":
            if config.get("secure_processing_enabled"):
                status = ComplianceStatus.COMPLIANT
                evidence = "TEE/Watermark/TimeGuard abilitato"
            else:
                status = ComplianceStatus.NOT_ASSESSED
                evidence = "Secure processing non abilitato"

        # Data permits (Art. 53)
        elif art_id == "Art. 53":
            n_hospitals = len(config.get("hospitals", []))
            if n_hospitals > 0:
                status = ComplianceStatus.PARTIAL
                evidence = f"{n_hospitals} ospedali configurati (permit pre-training)"
            else:
                status = ComplianceStatus.NOT_ASSESSED
                evidence = "Nessun ospedale configurato"

        # Cross-border (Art. 57-58)
        elif "Art. 57" in art_id:
            n_countries = len(config.get("countries", []))
            if n_countries > 1:
                if config.get("hdab_routing_enabled"):
                    status = ComplianceStatus.COMPLIANT
                    evidence = f"SAP routing: {n_countries} paesi"
                else:
                    status = ComplianceStatus.PARTIAL
                    evidence = f"{n_countries} paesi, routing non abilitato"
            else:
                status = ComplianceStatus.NOT_ASSESSED
                evidence = "Singolo paese, cross-border non applicabile"

        # Data quality (Art. 69)
        elif art_id == "Art. 69":
            if config.get("data_quality_enabled"):
                status = ComplianceStatus.COMPLIANT
                evidence = "Quality framework GOLD/SILVER/BRONZE abilitato"
            else:
                status = ComplianceStatus.NOT_ASSESSED
                evidence = "Quality framework non abilitato"

        # Opt-out (Art. 71)
        elif art_id == "Art. 71":
            status = ComplianceStatus.COMPLIANT
            evidence = "Opt-out registry disponibile (Art. 71)"

        # Audit trail (GDPR Art. 30)
        elif art_id == "Art. 30":
            status = ComplianceStatus.PARTIAL
            evidence = "Audit trail configurato (pre-training)"

        assessment = ArticleAssessment(
            article_id=art_id,
            title=title,
            chapter=chapter,
            status=status,
            evidence=evidence,
        )
        report.assessments.append(assessment)

    st.session_state.gov_compliance_report = report


def _generate_compliance_from_bridge(bridge, config: Dict):
    """Generate compliance report from governance bridge results."""
    from governance.ehds_compliance_report import (
        ArticleAssessment,
        ComplianceStatus,
        EHDSComplianceReport,
        ARTICLE_REGISTRY,
    )

    report = EHDSComplianceReport()
    report.generated_at = datetime.now().isoformat(timespec="seconds")

    hospitals = config.get("hospitals", [])
    report.scenario_label = (
        f"{len(config['countries'])} countries, "
        f"{len(hospitals)} hospitals, "
        f"purpose={config['purpose']}, "
        f"eps={config['global_epsilon']}"
    )
    report.training_config = config

    # Assess from bridge state
    permits_summary = bridge.get_permits_summary()
    budget_status = bridge.get_budget_status()
    minimization = bridge.get_minimization_report()

    for chapter, art_id, title, _ in ARTICLE_REGISTRY:
        status = ComplianceStatus.NOT_ASSESSED
        evidence = "Non valutato"

        if art_id == "Art. 33":
            from governance.ehds_compliance_report import ARTICLE_53_PURPOSES
            n_permits = permits_summary.get("total_permits", 0)
            if config["purpose"] in ARTICLE_53_PURPOSES and n_permits > 0:
                status = ComplianceStatus.COMPLIANT
                evidence = f"HDAB validato: {n_permits} permit, purpose={config['purpose']}"
            elif config["purpose"] in ARTICLE_53_PURPOSES:
                status = ComplianceStatus.PARTIAL
                evidence = f"Purpose valido ma nessun permit emesso"
            else:
                status = ComplianceStatus.NON_COMPLIANT
                evidence = f"Purpose '{config['purpose']}' non in Art. 53"

        elif art_id == "Art. 34":
            status = ComplianceStatus.COMPLIANT
            evidence = f"Dataset: {config['dataset_type']}"

        elif art_id == "Art. 42":
            if config.get("fee_model_enabled"):
                status = ComplianceStatus.COMPLIANT
                evidence = "Fee model attivo con tracking costi"
            else:
                status = ComplianceStatus.NOT_ASSESSED
                evidence = "Fee model non abilitato"

        elif art_id == "Art. 44":
            if minimization:
                status = ComplianceStatus.COMPLIANT
                evidence = (
                    f"{minimization['original_features']} -> "
                    f"{minimization['kept_features']} features "
                    f"(-{minimization['reduction_pct']}%)"
                )
            elif config.get("data_minimization_enabled"):
                status = ComplianceStatus.PARTIAL
                evidence = "Abilitato ma non applicato (nessun dato tabular)"
            else:
                status = ComplianceStatus.NOT_ASSESSED
                evidence = "Non abilitato"

        elif "Art. 46" in art_id:
            auth = config.get("hdab_auth_method", "oauth2")
            status = ComplianceStatus.COMPLIANT
            evidence = f"Autenticazione HDAB: {auth}"

        elif "Art. 48" in art_id:
            n_countries = len(config.get("countries", []))
            if n_countries > 1:
                status = ComplianceStatus.COMPLIANT
                evidence = f"Privacy per-giurisdizione: {n_countries} paesi"
            else:
                status = ComplianceStatus.PARTIAL
                evidence = "Singolo paese"

        elif art_id == "Art. 50":
            if config.get("secure_processing_enabled"):
                status = ComplianceStatus.COMPLIANT
                evidence = "TEE + Watermark + TimeGuard attivo"
            else:
                status = ComplianceStatus.NOT_ASSESSED
                evidence = "Secure processing non abilitato"

        elif art_id == "Art. 53":
            n_permits = permits_summary.get("total_permits", 0)
            used = budget_status.get("used", 0)
            total = budget_status.get("total", 0)
            if n_permits > 0:
                status = ComplianceStatus.COMPLIANT
                evidence = f"{n_permits} HDAB permit, budget {used:.2f}/{total:.1f}"
            else:
                status = ComplianceStatus.PARTIAL
                evidence = "Nessun permit emesso"

        elif "Art. 57" in art_id:
            n_countries = len(config.get("countries", []))
            if n_countries > 1 and config.get("hdab_routing_enabled"):
                status = ComplianceStatus.COMPLIANT
                evidence = f"SAP routing: {n_countries} paesi con joint approval"
            elif n_countries > 1:
                status = ComplianceStatus.PARTIAL
                evidence = f"{n_countries} paesi, SAP non abilitato"
            else:
                status = ComplianceStatus.NOT_ASSESSED
                evidence = "Cross-border non applicabile"

        elif art_id == "Art. 69":
            if config.get("data_quality_enabled"):
                status = ComplianceStatus.COMPLIANT
                evidence = "Quality labels GOLD/SILVER/BRONZE assegnate"
            else:
                status = ComplianceStatus.NOT_ASSESSED
                evidence = "Quality framework non abilitato"

        elif art_id == "Art. 71":
            status = ComplianceStatus.COMPLIANT
            evidence = "Opt-out registry attivo (Art. 71)"

        elif art_id == "Art. 30":
            status = ComplianceStatus.COMPLIANT
            evidence = "Audit trail GDPR Art. 30 attivo"

        assessment = ArticleAssessment(
            article_id=art_id,
            title=title,
            chapter=chapter,
            status=status,
            evidence=evidence,
        )
        report.assessments.append(assessment)

    st.session_state.gov_compliance_report = report


def _display_compliance_report(report):
    """Display the compliance report in a visual format."""
    from governance.ehds_compliance_report import ComplianceStatus

    summary = report.get_summary()

    # Score and summary metrics
    st.markdown(f"**Scenario:** {report.scenario_label}")
    st.markdown(f"**Generato:** {report.generated_at}")

    score = summary["compliance_score_pct"]
    score_cols = st.columns(5)
    with score_cols[0]:
        st.metric("Compliance Score", f"{score}%")
    with score_cols[1]:
        st.metric("Compliant", summary["compliant"])
    with score_cols[2]:
        st.metric("Partial", summary["partial"])
    with score_cols[3]:
        st.metric("Not Assessed", summary["not_assessed"])
    with score_cols[4]:
        st.metric("Non Compliant", summary["non_compliant"])

    # Per-article table
    st.markdown("---")
    st.markdown("##### Valutazione per Articolo")

    rows = []
    current_chapter = ""
    for a in report.assessments:
        status_emoji = {
            ComplianceStatus.COMPLIANT: "COMPLIANT",
            ComplianceStatus.PARTIAL: "PARTIAL",
            ComplianceStatus.NOT_ASSESSED: "NOT ASSESSED",
            ComplianceStatus.NON_COMPLIANT: "NON COMPLIANT",
        }

        rows.append({
            "Capitolo": a.chapter.split(" - ")[0] if " - " in a.chapter else a.chapter,
            "Articolo": a.article_id,
            "Titolo": a.title,
            "Stato": status_emoji.get(a.status, str(a.status)),
            "Evidenza": a.evidence,
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, height=500)

    # Budget status
    bridge = st.session_state.gov_bridge
    if bridge is not None:
        budget = bridge.get_budget_status()
        st.markdown("##### Privacy Budget")
        bud_cols = st.columns(4)
        with bud_cols[0]:
            st.metric("Budget Totale (epsilon)", f"{budget['total']:.2f}")
        with bud_cols[1]:
            st.metric("Utilizzato", f"{budget['used']:.4f}")
        with bud_cols[2]:
            st.metric("Rimanente", f"{budget['remaining']:.4f}")
        with bud_cols[3]:
            st.metric("Round Completati", f"{budget['rounds_completed']}/{budget['max_rounds']}")

    # Export options
    st.markdown("---")
    exp_cols = st.columns(2)
    with exp_cols[0]:
        if st.button("Esporta JSON", key="gov_export_json"):
            report_json = report.to_json()
            st.json(report_json)

    with exp_cols[1]:
        if st.button("Genera LaTeX Table", key="gov_export_latex"):
            latex = report.to_latex_table()
            st.code(latex, language="latex")


def _render_fee_analysis():
    """Sub-tab: Fee model analysis and budget optimization."""
    st.markdown("##### Analisi Costi (EHDS Art. 42)")
    st.markdown(
        "Analizza i costi di accesso ai dati sanitari per l'FL training "
        "e ottimizza la configurazione entro un budget massimo."
    )

    config = st.session_state.gov_config
    hospitals = config.get("hospitals", [])

    if not hospitals:
        st.warning("Configura gli ospedali nella tab 'Configurazione Lifecycle'.")
        return

    # Fee estimation parameters
    st.markdown("**Parametri Stima Costi**")
    fee_cols = st.columns(4)
    with fee_cols[0]:
        model_size_mb = st.number_input(
            "Dimensione Modello (MB)",
            min_value=0.1, max_value=100.0, value=2.0, step=0.5,
            key="gov_model_size",
        )
    with fee_cols[1]:
        max_budget = st.number_input(
            "Budget Massimo (EUR)",
            min_value=100, max_value=100000, value=5000, step=500,
            key="gov_max_budget",
        )
    with fee_cols[2]:
        samples_per_hospital = st.number_input(
            "Campioni per Ospedale",
            min_value=50, max_value=10000, value=500,
            key="gov_samples",
        )
    with fee_cols[3]:
        num_rounds = config.get("num_rounds", 30)
        st.metric("Round FL", num_rounds)

    if st.button("Calcola Stima Costi", type="primary", key="gov_fee_calc"):
        from governance.fee_model import FeeModelBridge

        fee_bridge = FeeModelBridge(
            hospitals=hospitals,
            config=config,
            num_rounds=num_rounds,
            model_size_mb=model_size_mb,
        )

        # Estimate total cost
        estimated_cost = fee_bridge.estimate_total_cost()

        st.session_state.gov_fee_bridge = fee_bridge

        # Display cost breakdown
        st.markdown("---")
        st.markdown("##### Stima Costi")

        cost_cols = st.columns(2)
        with cost_cols[0]:
            st.metric("Costo Stimato Totale", f"{estimated_cost:,.2f} EUR")
            if estimated_cost <= max_budget:
                st.success(
                    f"Entro il budget ({max_budget:,.0f} EUR). "
                    f"Risparmio: {max_budget - estimated_cost:,.2f} EUR"
                )
            else:
                st.warning(
                    f"Supera il budget di {estimated_cost - max_budget:,.2f} EUR. "
                    "Prova l'ottimizzazione budget."
                )

        with cost_cols[1]:
            # Per-hospital cost breakdown
            per_hospital_costs = []
            for h in hospitals:
                h_fee = fee_bridge._hospital_fees.get(h.hospital_id, {})
                per_hospital_costs.append({
                    "Ospedale": h.name,
                    "Paese": h.country_code,
                    "Costo Base": f"{h_fee.get('base_access', 0):.0f} EUR",
                    "Costo Dati": f"{h_fee.get('data_volume', 0):.0f} EUR",
                    "Costo Compute": f"{h_fee.get('computation', 0):.0f} EUR",
                    "Costo Transfer": f"{h_fee.get('transfer', 0):.0f} EUR",
                })
            if per_hospital_costs:
                st.dataframe(
                    pd.DataFrame(per_hospital_costs), use_container_width=True
                )

        # Budget optimization
        if estimated_cost > max_budget:
            st.markdown("##### Ottimizzazione Budget")
            if st.button("Esegui Ottimizzazione", key="gov_fee_optimize"):
                opt_result = fee_bridge.optimize_for_budget(max_budget_eur=max_budget)
                if opt_result.feasible:
                    st.success(
                        f"Ottimizzazione riuscita: strategia '{opt_result.strategy}'\n\n"
                        f"- Costo originale: {opt_result.original_cost_eur:,.2f} EUR\n"
                        f"- Costo ottimizzato: {opt_result.optimized_cost_eur:,.2f} EUR\n"
                        f"- {opt_result.explanation}"
                    )
                else:
                    st.error(
                        "Ottimizzazione non possibile con il budget dato. "
                        "Considera di ridurre ospedali o round."
                    )


def render_governance_workflow_tab():
    """Main entry point for the Governance Workflow tab."""
    st.markdown("### EHDS Governance Workflow")
    st.markdown(
        "Ciclo di vita completo della governance EHDS Chapter IV per "
        "il federated learning cross-border. Configura, esegui e verifica "
        "la compliance con la EHDS Regulation (EU) 2025/327."
    )

    _init_governance_state()

    tabs = st.tabs([
        "Configurazione Lifecycle",
        "Esecuzione Lifecycle",
        "Dashboard Compliance",
        "Analisi Costi (Art. 42)",
    ])

    with tabs[0]:
        _render_lifecycle_config()

    with tabs[1]:
        _render_lifecycle_execution()

    with tabs[2]:
        _render_compliance_dashboard()

    with tabs[3]:
        _render_fee_analysis()

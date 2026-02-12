"""
Permit Manager Page for FL-EHDS Dashboard.

Dedicated permit lifecycle management with three sub-tabs:
1. Permit Creation - Form with Art. 53 purposes, data categories, cross-border config
2. Permit Dashboard - Active permits with status, budget, actions
3. Opt-out Management - Art. 71 citizen opt-out registry

Uses real governance modules: DataPermitManager, OptOutRegistry, OptOutChecker.

Author: Fabio Liberti
"""

import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

# Governance imports
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
    OptOutRecord,
)
from governance.data_permits import DataPermitManager, PermitValidator
from governance.hdab_integration import PermitStore, get_shared_permit_store
from governance.optout_registry import OptOutRegistry, OptOutChecker


# Purpose display labels (Italian)
PURPOSE_LABELS = {
    PermitPurpose.SCIENTIFIC_RESEARCH: "Ricerca Scientifica (Art. 53.1a)",
    PermitPurpose.PUBLIC_HEALTH_SURVEILLANCE: "Sorveglianza Sanitaria (Art. 53.1b)",
    PermitPurpose.HEALTH_POLICY: "Politica Sanitaria (Art. 53.1c)",
    PermitPurpose.EDUCATION_TRAINING: "Formazione/Istruzione (Art. 53.1d)",
    PermitPurpose.AI_SYSTEM_DEVELOPMENT: "Sviluppo Sistemi AI (Art. 53.1e)",
    PermitPurpose.PERSONALIZED_MEDICINE: "Medicina Personalizzata (Art. 53.1f)",
    PermitPurpose.OFFICIAL_STATISTICS: "Statistiche Ufficiali (Art. 53.1g)",
    PermitPurpose.PATIENT_SAFETY: "Sicurezza del Paziente (Art. 53.1h)",
}

CATEGORY_LABELS = {
    DataCategory.EHR: "Cartelle Cliniche Elettroniche (EHR)",
    DataCategory.LAB_RESULTS: "Risultati di Laboratorio",
    DataCategory.IMAGING: "Imaging Diagnostico",
    DataCategory.GENOMIC: "Dati Genomici",
    DataCategory.REGISTRY: "Registri di Malattia",
    DataCategory.CLAIMS: "Dati Assicurativi",
    DataCategory.QUESTIONNAIRE: "Questionari Paziente",
}


def _init_session_state():
    """Initialize session state for permit manager."""
    if "pm_permit_store" not in st.session_state:
        st.session_state.pm_permit_store = get_shared_permit_store()
    if "pm_permit_manager" not in st.session_state:
        st.session_state.pm_permit_manager = DataPermitManager(
            validator=PermitValidator(strict_mode=False)
        )
    if "pm_optout_registry" not in st.session_state:
        st.session_state.pm_optout_registry = OptOutRegistry()
    if "pm_optout_checker" not in st.session_state:
        st.session_state.pm_optout_checker = OptOutChecker(
            st.session_state.pm_optout_registry
        )


def _render_permit_creation():
    """Sub-tab: Create a new EHDS data permit."""
    st.markdown("##### Crea Nuovo Data Permit (EHDS Art. 53)")
    st.markdown(
        "Compila il modulo per richiedere un nuovo data permit. "
        "Tutti i campi vengono validati contro i requisiti EHDS Art. 53."
    )

    with st.form("pm_create_permit_form"):
        # Basic info
        col1, col2 = st.columns(2)
        with col1:
            requester_org = st.text_input(
                "Organizzazione Richiedente",
                value="Universita degli Studi",
            )
            requester_id = st.text_input(
                "ID Richiedente",
                value="researcher-001",
            )
        with col2:
            hdab_id = st.selectbox(
                "HDAB Emittente",
                options=["HDAB-IT", "HDAB-DE", "HDAB-FR", "HDAB-ES", "HDAB-NL"],
            )
            research_question = st.text_input(
                "Research Question",
                value="Federated model for diabetic retinopathy progression",
            )

        # Purpose (Art. 53)
        purpose_options = list(PURPOSE_LABELS.keys())
        purpose_labels = list(PURPOSE_LABELS.values())
        purpose_idx = st.selectbox(
            "Finalita (EHDS Art. 53)",
            options=range(len(purpose_options)),
            format_func=lambda i: purpose_labels[i],
        )
        selected_purpose = purpose_options[purpose_idx]

        # Data categories
        st.markdown("**Categorie Dati (EHDS Art. 34)**")
        cat_cols = st.columns(4)
        selected_categories = []
        for i, (cat, label) in enumerate(CATEGORY_LABELS.items()):
            with cat_cols[i % 4]:
                if st.checkbox(label, key=f"pm_cat_{cat.value}",
                               value=(cat == DataCategory.EHR)):
                    selected_categories.append(cat)

        # Cross-border config
        st.markdown("**Configurazione Cross-Border**")
        cb_cols = st.columns(3)
        with cb_cols[0]:
            member_states = st.multiselect(
                "Stati Membri",
                options=["IT", "DE", "FR", "ES", "NL", "BE", "AT", "PT", "GR", "PL"],
                default=["IT"],
            )
        with cb_cols[1]:
            validity_days = st.number_input(
                "Validita (giorni)", min_value=30, max_value=730, value=365
            )
        with cb_cols[2]:
            max_rounds = st.number_input(
                "Max Round FL", min_value=5, max_value=500, value=50
            )

        # Privacy budget
        budget_cols = st.columns(2)
        with budget_cols[0]:
            privacy_budget = st.number_input(
                "Privacy Budget (epsilon)",
                min_value=0.1, max_value=100.0, value=10.0, step=0.5,
            )
        with budget_cols[1]:
            min_clients = st.number_input(
                "Minimo Client FL",
                min_value=2, max_value=20, value=3,
            )

        submitted = st.form_submit_button("Crea Permit")

        if submitted:
            if not selected_categories:
                st.error("Seleziona almeno una categoria di dati.")
                return

            permit_id = f"PERMIT-{uuid.uuid4().hex[:12].upper()}"
            now = datetime.utcnow()

            permit = DataPermit(
                permit_id=permit_id,
                hdab_id=hdab_id,
                requester_id=requester_id,
                purpose=selected_purpose,
                data_categories=selected_categories,
                member_states=member_states,
                issued_at=now,
                valid_from=now,
                valid_until=now + timedelta(days=validity_days),
                status=PermitStatus.ACTIVE,
                privacy_budget_total=privacy_budget,
                conditions={
                    "max_rounds": max_rounds,
                    "privacy_budget": privacy_budget,
                    "min_clients": min_clients,
                    "research_question": research_question,
                },
                metadata={
                    "organization": requester_org,
                    "created_via": "dashboard_permit_manager",
                },
            )

            store = st.session_state.pm_permit_store
            pm = st.session_state.pm_permit_manager
            store.register(permit)
            pm.register_permit(permit)
            st.success(
                f"Permit **{permit_id}** creato con successo!\n\n"
                f"- Finalita: {PURPOSE_LABELS[selected_purpose]}\n"
                f"- Categorie: {', '.join(c.value for c in selected_categories)}\n"
                f"- Validita: {validity_days} giorni\n"
                f"- Budget privacy: epsilon = {privacy_budget}"
            )


def _render_permit_dashboard():
    """Sub-tab: View and manage active permits."""
    st.markdown("##### Dashboard Permit Attivi")

    store = st.session_state.pm_permit_store

    # Get all permits via PermitStore
    all_permits = store.list_all()

    if not all_permits:
        st.info(
            "Nessun permit presente. Crea un nuovo permit nella tab 'Creazione Permit'."
        )
        return

    # Summary metrics
    active = sum(1 for p in all_permits if p.status == PermitStatus.ACTIVE)
    suspended = sum(1 for p in all_permits if p.status == PermitStatus.SUSPENDED)
    revoked = sum(1 for p in all_permits if p.status == PermitStatus.REVOKED)
    expired = sum(1 for p in all_permits if p.status == PermitStatus.EXPIRED)

    metric_cols = st.columns(4)
    with metric_cols[0]:
        st.metric("Attivi", active)
    with metric_cols[1]:
        st.metric("Sospesi", suspended)
    with metric_cols[2]:
        st.metric("Revocati", revoked)
    with metric_cols[3]:
        st.metric("Scaduti", expired)

    # Permit table
    rows = []
    for p in all_permits:
        budget_total = p.privacy_budget_total or 0
        budget_used = p.privacy_budget_used or 0
        budget_pct = (budget_used / budget_total * 100) if budget_total > 0 else 0

        days_remaining = (p.valid_until - datetime.utcnow()).days if p.valid_until else 0

        rows.append({
            "ID": p.permit_id[:16],
            "HDAB": p.hdab_id,
            "Stato": p.status.value,
            "Finalita": p.purpose.value,
            "Categorie": ", ".join(c.value for c in p.data_categories),
            "Budget epsilon": f"{budget_used:.2f}/{budget_total:.1f} ({budget_pct:.0f}%)",
            "Scadenza (gg)": max(0, days_remaining),
            "Max Rounds": p.conditions.get("max_rounds", "N/A"),
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)

    # Permit actions
    st.markdown("##### Azioni su Permit")
    action_cols = st.columns([2, 1, 1])

    with action_cols[0]:
        permit_ids = [p.permit_id[:16] for p in all_permits]
        selected_permit_display = st.selectbox(
            "Seleziona Permit",
            options=permit_ids,
            key="pm_action_select",
        )
        # Get full ID
        selected_permit_full = next(
            (p.permit_id for p in all_permits
             if p.permit_id[:16] == selected_permit_display),
            None,
        )

    with action_cols[1]:
        action = st.selectbox(
            "Azione",
            options=["Verifica", "Sospendi", "Revoca"],
            key="pm_action_type",
        )

    with action_cols[2]:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Esegui Azione", key="pm_action_btn"):
            if selected_permit_full:
                if action == "Verifica":
                    p = store.get(selected_permit_full)
                    if p and p.status == PermitStatus.ACTIVE:
                        st.success(f"Permit {selected_permit_display} VALIDO e ATTIVO")
                    else:
                        status_str = p.status.value if p else "non trovato"
                        st.warning(f"Permit {selected_permit_display}: {status_str}")
                elif action == "Sospendi":
                    ok = store.suspend(
                        selected_permit_full, "Sospeso da dashboard"
                    )
                    if ok:
                        st.warning(f"Permit {selected_permit_display} SOSPESO")
                    else:
                        st.error("Impossibile sospendere il permit")
                elif action == "Revoca":
                    ok = store.revoke(
                        selected_permit_full, "Revocato da dashboard"
                    )
                    if ok:
                        st.error(f"Permit {selected_permit_display} REVOCATO")
                    else:
                        st.error("Impossibile revocare il permit")

    # Audit log
    audit_log = store.audit_log
    if audit_log:
        with st.expander(f"Audit Log ({len(audit_log)} eventi)", expanded=False):
            audit_df = pd.DataFrame(audit_log[-30:])
            st.dataframe(audit_df, use_container_width=True)


def _render_optout_management():
    """Sub-tab: Art. 71 citizen opt-out management."""
    st.markdown("##### Gestione Opt-Out Cittadini (EHDS Art. 71)")
    st.markdown(
        "L'Art. 71 EHDS garantisce ai cittadini EU il diritto di rifiutare "
        "l'uso secondario dei propri dati sanitari. Registra e gestisci le "
        "richieste di opt-out."
    )

    registry = st.session_state.pm_optout_registry
    checker = st.session_state.pm_optout_checker

    # Register opt-out form
    with st.expander("Registra Nuovo Opt-Out", expanded=True):
        with st.form("pm_optout_form"):
            opt_cols = st.columns(2)
            with opt_cols[0]:
                patient_id = st.text_input(
                    "ID Paziente", value="PAT-001"
                )
                member_state = st.selectbox(
                    "Stato Membro",
                    options=["IT", "DE", "FR", "ES", "NL", "BE", "AT", "PT"],
                )
            with opt_cols[1]:
                scope = st.selectbox(
                    "Ambito Opt-Out",
                    options=["all", "category", "purpose"],
                    format_func=lambda x: {
                        "all": "Tutti i dati",
                        "category": "Per categoria",
                        "purpose": "Per finalita",
                    }[x],
                )
                reason = st.text_input(
                    "Motivazione (opzionale)", value=""
                )

            # Category/purpose selection based on scope
            opt_categories = None
            opt_purposes = None
            if scope == "category":
                cat_labels = list(CATEGORY_LABELS.values())
                cat_keys = list(CATEGORY_LABELS.keys())
                selected_cats = st.multiselect(
                    "Categorie da escludere",
                    options=range(len(cat_keys)),
                    format_func=lambda i: cat_labels[i],
                    default=[0],
                )
                opt_categories = [cat_keys[i] for i in selected_cats]
            elif scope == "purpose":
                purp_labels = list(PURPOSE_LABELS.values())
                purp_keys = list(PURPOSE_LABELS.keys())
                selected_purps = st.multiselect(
                    "Finalita da escludere",
                    options=range(len(purp_keys)),
                    format_func=lambda i: purp_labels[i],
                    default=[0],
                )
                opt_purposes = [purp_keys[i] for i in selected_purps]

            opt_submitted = st.form_submit_button("Registra Opt-Out")

            if opt_submitted:
                record = OptOutRecord(
                    record_id=f"OPT-{uuid.uuid4().hex[:8].upper()}",
                    patient_id=patient_id,
                    member_state=member_state,
                    scope=scope,
                    categories=(
                        list(opt_categories) if opt_categories else None
                    ),
                    purposes=(
                        list(opt_purposes) if opt_purposes else None
                    ),
                    metadata={"reason": reason} if reason else {},
                )
                registry.register_optout(record)
                st.success(
                    f"Opt-out registrato per paziente **{patient_id}** "
                    f"({member_state}) - Ambito: {scope}"
                )

    # Batch check
    st.markdown("##### Verifica Batch per Training FL")
    check_cols = st.columns([2, 1])
    with check_cols[0]:
        check_ids = st.text_area(
            "ID Pazienti (uno per riga)",
            value="PAT-001\nPAT-002\nPAT-003\nPAT-004\nPAT-005",
            height=100,
        )
    with check_cols[1]:
        check_purpose_idx = st.selectbox(
            "Finalita Training",
            options=range(len(list(PURPOSE_LABELS.keys()))),
            format_func=lambda i: list(PURPOSE_LABELS.values())[i],
            key="pm_check_purpose",
        )
        check_category_idx = st.selectbox(
            "Categoria Dati",
            options=range(len(list(CATEGORY_LABELS.keys()))),
            format_func=lambda i: list(CATEGORY_LABELS.values())[i],
            key="pm_check_category",
        )

    if st.button("Verifica Opt-Out", key="pm_batch_check"):
        ids = [line.strip() for line in check_ids.strip().split("\n") if line.strip()]
        purpose = list(PURPOSE_LABELS.keys())[check_purpose_idx]
        category = list(CATEGORY_LABELS.keys())[check_category_idx]

        results = []
        allowed_count = 0
        blocked_count = 0
        for pid in ids:
            is_opted = registry.is_opted_out(pid)
            status = "BLOCCATO" if is_opted else "CONSENTITO"
            if is_opted:
                blocked_count += 1
            else:
                allowed_count += 1
            results.append({
                "Paziente": pid,
                "Stato": status,
                "Finalita": purpose.value,
                "Categoria": category.value,
            })

        result_df = pd.DataFrame(results)
        st.dataframe(result_df, use_container_width=True)

        result_cols = st.columns(3)
        with result_cols[0]:
            st.metric("Totale", len(ids))
        with result_cols[1]:
            st.metric("Consentiti", allowed_count)
        with result_cols[2]:
            st.metric("Bloccati", blocked_count)

    # Registry statistics
    stats = registry.get_stats()
    if stats.total_opted_out > 0:
        with st.expander(
            f"Statistiche Registry ({stats.total_opted_out} opt-out)", expanded=False
        ):
            stat_cols = st.columns(3)
            with stat_cols[0]:
                st.metric("Totale Opt-Out", stats.total_opted_out)
            with stat_cols[1]:
                st.metric("Lookup Totali", stats.total_lookups)
            with stat_cols[2]:
                st.metric(
                    "Cache Hit Rate",
                    f"{stats.cache_hit_rate:.0%}",
                )

            if stats.by_member_state:
                st.markdown("**Per Stato Membro:**")
                ms_df = pd.DataFrame(
                    [
                        {"Stato": k, "Opt-Out": v}
                        for k, v in stats.by_member_state.items()
                    ]
                )
                st.dataframe(ms_df, use_container_width=True)

            if stats.by_scope:
                st.markdown("**Per Ambito:**")
                scope_df = pd.DataFrame(
                    [
                        {"Ambito": k, "Opt-Out": v}
                        for k, v in stats.by_scope.items()
                    ]
                )
                st.dataframe(scope_df, use_container_width=True)


def render_permit_manager_tab():
    """Main entry point for the Permit Manager tab."""
    st.markdown("### Gestione Data Permit EHDS")
    st.markdown(
        "Gestisci l'intero ciclo di vita dei data permit secondo "
        "l'EHDS Regulation (EU) 2025/327, incluso il diritto di opt-out (Art. 71)."
    )

    _init_session_state()

    tabs = st.tabs([
        "Creazione Permit",
        "Dashboard Permit",
        "Gestione Opt-Out",
    ])

    with tabs[0]:
        _render_permit_creation()

    with tabs[1]:
        _render_permit_dashboard()

    with tabs[2]:
        _render_optout_management()

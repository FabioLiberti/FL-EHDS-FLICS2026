"""EHDS compliance, infrastructure, HDAB, opt-out, and guide tabs for FL-EHDS dashboard."""

import streamlit as st
import numpy as np
import pandas as pd
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any


# ---------------------------------------------------------------------------
# Optional imports: governance modules (HDAB + Opt-out Registry)
# ---------------------------------------------------------------------------
import sys as _sys

_framework_root = str(Path(__file__).parent.parent)
if _framework_root not in _sys.path:
    _sys.path.insert(0, _framework_root)

try:
    import asyncio
    from governance.hdab_integration import (
        HDABClient,
        HDABConfig,
        PermitStore,
        get_shared_permit_store,
        MultiHDABCoordinator,
    )
    from governance.optout_registry import OptOutRegistry, OptOutChecker
    from governance.persistence import GovernanceDB
    from core.models import (
        DataPermit,
        PermitStatus,
        PermitPurpose,
        DataCategory,
        OptOutRecord,
    )
    HAS_GOVERNANCE = True
except ImportError:
    HAS_GOVERNANCE = False

# ---------------------------------------------------------------------------
# Optional imports: cross-border federation module
# ---------------------------------------------------------------------------
try:
    from terminal.cross_border import EU_COUNTRY_PROFILES, CrossBorderFederatedTrainer
    _has_cross_border = True
except ImportError:
    _has_cross_border = False

# ---------------------------------------------------------------------------
# Optional imports: fl_ehds.core helpers used in code examples (not called
# at runtime, but referenced in st.code blocks -- no actual import needed)
# ---------------------------------------------------------------------------

__all__ = [
    "render_ehds_tab",
    "render_infrastructure_tab",
    "render_guide_tab",
]


# ===================================================================== #
#                       Cross-Border Federation                          #
# ===================================================================== #

def _render_cross_border_tab():
    """Render Cross-Border Federation simulation sub-tab."""
    st.markdown("#### Cross-Border Federation Simulation")

    st.markdown("""
    Simulazione di Federated Learning tra ospedali in diverse giurisdizioni EU,
    ciascuna con regole DP, policy HDAB e latenza di rete differenti.
    """)

    try:
        from terminal.cross_border import EU_COUNTRY_PROFILES, CrossBorderFederatedTrainer
        _has_cb = True
    except ImportError:
        _has_cb = False

    if not _has_cb:
        st.warning("Modulo cross-border non disponibile.")
        return

    # Country profiles table
    st.markdown("##### Profili Regolamentari EU")
    profiles_data = {
        "Paese": [], "Nome": [], "Epsilon Max": [], "HDAB Strictness": [],
        "Opt-out Rate": [], "Latenza (ms)": [], "Scopi Ammessi": [],
    }
    for cc, p in sorted(EU_COUNTRY_PROFILES.items(), key=lambda x: x[1].hdab_strictness, reverse=True):
        profiles_data["Paese"].append(cc)
        profiles_data["Nome"].append(p.name)
        profiles_data["Epsilon Max"].append(p.dp_epsilon_max)
        profiles_data["HDAB Strictness"].append("*" * p.hdab_strictness)
        profiles_data["Opt-out Rate"].append(f"{p.opt_out_rate:.0%}")
        profiles_data["Latenza (ms)"].append(f"{p.latency_ms[0]}-{p.latency_ms[1]}")
        profiles_data["Scopi Ammessi"].append(len(p.allowed_purposes))
    st.dataframe(pd.DataFrame(profiles_data), use_container_width=True, hide_index=True)

    st.markdown("---")

    # Simulation configuration
    st.markdown("##### Configurazione Simulazione")
    col1, col2, col3 = st.columns(3)

    with col1:
        available_countries = list(EU_COUNTRY_PROFILES.keys())
        selected_countries = st.multiselect(
            "Paesi", available_countries, default=["DE", "FR", "IT", "ES", "NL"],
        )
        cb_purpose = st.selectbox("Scopo EHDS (Art. 53)", [
            "scientific_research", "public_health_surveillance",
            "health_policy", "ai_system_development",
        ], index=0)

    with col2:
        cb_algorithm = st.selectbox("Algoritmo FL", [
            "FedAvg", "FedProx", "SCAFFOLD", "FedNova", "FedDyn",
            "FedAdam", "FedYogi", "FedAdagrad", "Per-FedAvg", "Ditto",
        ], index=0)
        cb_rounds = st.slider("Round FL", 5, 50, 15)
        cb_epsilon = st.slider("Epsilon Globale", 1.0, 50.0, 10.0, 0.5)

    with col3:
        cb_hospitals_per = st.slider("Ospedali per Paese", 1, 3, 1)
        cb_lr = st.select_slider("Learning Rate", [0.0001, 0.0005, 0.001, 0.005, 0.01], value=0.01)
        cb_latency = st.checkbox("Simula Latenza Rete", value=True)

    # Show effective epsilon per country
    if selected_countries:
        st.markdown("##### Epsilon Effettivo per Giurisdizione")
        eps_data = {"Paese": [], "Nome": [], "Eps Max Nazionale": [], "Eps Effettivo": [], "Stato": []}
        for cc in selected_countries:
            p = EU_COUNTRY_PROFILES[cc]
            eff = min(cb_epsilon, p.dp_epsilon_max)
            status = "OK" if eff == cb_epsilon else f"Limitato a {eff:.1f}"
            eps_data["Paese"].append(cc)
            eps_data["Nome"].append(p.name)
            eps_data["Eps Max Nazionale"].append(p.dp_epsilon_max)
            eps_data["Eps Effettivo"].append(eff)
            eps_data["Stato"].append(status)
        st.dataframe(pd.DataFrame(eps_data), use_container_width=True, hide_index=True)

    # Purpose violations preview
    violations = []
    for cc in selected_countries:
        p = EU_COUNTRY_PROFILES[cc]
        if cb_purpose not in p.allowed_purposes:
            violations.append(f"**{p.name} ({cc})**: scopo '{cb_purpose}' non ammesso da HDAB")
    if violations:
        st.warning("**Violazioni Scopo HDAB:**\n" + "\n".join(f"- {v}" for v in violations))

    st.markdown("---")

    # Run simulation
    if st.button("Avvia Simulazione Cross-Border", type="primary", disabled=len(selected_countries) < 2):
        if len(selected_countries) < 2:
            st.error("Selezionare almeno 2 paesi.")
            return

        progress_bar = st.progress(0)
        status_text = st.empty()
        metrics_container = st.empty()

        try:
            trainer = CrossBorderFederatedTrainer(
                countries=selected_countries,
                hospitals_per_country=cb_hospitals_per,
                algorithm=cb_algorithm,
                num_rounds=cb_rounds,
                local_epochs=3,
                batch_size=32,
                learning_rate=cb_lr,
                global_epsilon=cb_epsilon,
                purpose=cb_purpose,
                dataset_type="synthetic",
                seed=42,
                simulate_latency=cb_latency,
            )

            def cb_progress(round_num, total_rounds, result):
                progress_bar.progress((round_num + 1) / total_rounds)
                status_text.text(
                    f"Round {round_num+1}/{total_rounds} | "
                    f"Acc: {result.global_acc:.3f} | F1: {result.global_f1:.3f} | "
                    f"[{result.compliance_status.upper()}]"
                )

            trainer.progress_callback = cb_progress
            results = trainer.train()
            progress_bar.progress(1.0)
            status_text.text("Simulazione completata!")

            # Display results
            st.markdown("##### Risultati Cross-Border")
            last = trainer.history[-1]
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Accuracy", f"{last.global_acc:.2%}")
            r2.metric("F1 Score", f"{last.global_f1:.4f}")
            r3.metric("AUC-ROC", f"{last.global_auc:.4f}")
            r4.metric("Loss", f"{last.global_loss:.4f}")

            # Per-country analysis
            st.markdown("##### Analisi per Giurisdizione")
            country_summary = trainer.audit_log.summary_by_country()
            cs_data = {
                "Paese": [], "HDAB": [], "Eps Max": [], "Eps Usato": [],
                "Campioni": [], "Opt-out": [], "Latenza Media": [], "Stato": [],
            }
            for cc in selected_countries:
                if cc in country_summary:
                    d = country_summary[cc]
                    p = EU_COUNTRY_PROFILES[cc]
                    avg_lat = d["total_latency_ms"] / max(d["total_rounds"], 1)
                    cs_data["Paese"].append(cc)
                    cs_data["HDAB"].append("*" * p.hdab_strictness)
                    cs_data["Eps Max"].append(p.dp_epsilon_max)
                    cs_data["Eps Usato"].append(f"{d['epsilon_spent']:.3f}")
                    cs_data["Campioni"].append(d["total_samples_used"])
                    cs_data["Opt-out"].append(d["total_opted_out"])
                    cs_data["Latenza Media"].append(f"{avg_lat:.1f}ms")
                    cs_data["Stato"].append("OK" if d["epsilon_spent"] <= p.dp_epsilon_max else "OVER")
            st.dataframe(pd.DataFrame(cs_data), use_container_width=True, hide_index=True)

            # Convergence chart
            if trainer.history:
                st.markdown("##### Convergenza")
                chart_data = pd.DataFrame({
                    "Round": [h.round_num + 1 for h in trainer.history],
                    "Accuracy": [h.global_acc for h in trainer.history],
                    "Loss": [h.global_loss for h in trainer.history],
                    "F1": [h.global_f1 for h in trainer.history],
                })
                c1, c2 = st.columns(2)
                with c1:
                    st.line_chart(chart_data.set_index("Round")[["Accuracy", "F1"]])
                with c2:
                    st.line_chart(chart_data.set_index("Round")[["Loss"]])

            # Audit trail summary
            st.markdown("##### Compliance Summary")
            audit_violations = trainer.audit_log.get_violations()
            m1, m2, m3 = st.columns(3)
            m1.metric("Audit Entries", len(trainer.audit_log.entries))
            m2.metric("Violations", len(audit_violations))
            m3.metric("Tempo Totale", f"{results['total_time']:.1f}s")

        except Exception as e:
            st.error(f"Errore simulazione: {e}")
            import traceback
            st.code(traceback.format_exc())


# ===================================================================== #
#                         EHDS Interoperability Tab                      #
# ===================================================================== #

def render_ehds_tab():
    """Render EHDS Interoperability tab."""
    st.markdown("### 🇪🇺 EHDS Interoperability")

    st.markdown("""
    <div class="info-box">
    <strong>European Health Data Space (EHDS)</strong><br>
    Regolamento EU 2025/327 per lo spazio europeo dei dati sanitari.
    Questa sezione mostra le componenti di interoperabilità per FL conforme all'EHDS.
    </div>
    """, unsafe_allow_html=True)

    # Five sub-tabs for interoperability + cross-border simulation
    ehds_tabs = st.tabs([
        "🌍 Cross-Border FL",
        "🔗 HL7 FHIR",
        "📊 OMOP CDM",
        "📋 IHE Profiles",
        "🏛️ HDAB API"
    ])

    # Cross-Border Federation Simulation Tab
    with ehds_tabs[0]:
        _render_cross_border_tab()

    # HL7 FHIR Tab
    with ehds_tabs[1]:
        st.markdown("#### HL7 FHIR R4 Integration")

        st.markdown("""
        **Fast Healthcare Interoperability Resources (FHIR)** è lo standard principale
        per lo scambio di dati sanitari nell'EHDS.

        ##### Risorse FHIR Supportate
        """)

        fhir_resources = {
            "Patient": {"icon": "👤", "desc": "Dati demografici del paziente", "use": "Identificazione pseudonimizzata"},
            "Observation": {"icon": "🔬", "desc": "Valori laboratorio, segni vitali", "use": "Feature cliniche per ML"},
            "Condition": {"icon": "🏥", "desc": "Diagnosi, problemi di salute", "use": "Label per classificazione"},
            "MedicationRequest": {"icon": "💊", "desc": "Prescrizioni farmaci", "use": "Feature farmacologiche"},
            "Encounter": {"icon": "📅", "desc": "Visite, ricoveri", "use": "Contesto temporale"}
        }

        cols = st.columns(3)
        for i, (resource, info) in enumerate(fhir_resources.items()):
            with cols[i % 3]:
                st.markdown(f"""
                **{info['icon']} {resource}**
                - {info['desc']}
                - *FL Use:* {info['use']}
                """)

        st.markdown("##### Pipeline FHIR → FL")
        st.code("""
# Esempio: Pipeline FHIR per Federated Learning
from fl_ehds.core import FHIRClient, FHIRDataset, FHIRPrivacyGuard

# 1. Connessione FHIR Server
client = FHIRClient("https://hospital.eu/fhir")

# 2. Query risorse con filtri
patients = client.search("Patient", {"_count": 1000})
observations = client.search("Observation", {"code": "8480-6"})  # Sistolica

# 3. Privacy check (k-anonymity)
privacy = FHIRPrivacyGuard(k_threshold=5)
safe_data = privacy.verify_k_anonymity(patients)

# 4. Conversione in dataset ML
dataset = FHIRDataset(safe_data)
X, y = dataset.to_tensors()
        """, language="python")

        # FHIR Vocabulary mapping demo
        st.markdown("##### Mappatura Codici LOINC/SNOMED")
        vocab_data = {
            "Codice": ["8480-6", "8462-4", "2339-0", "29463-7", "718-7"],
            "Sistema": ["LOINC", "LOINC", "LOINC", "LOINC", "LOINC"],
            "Descrizione": ["Pressione sistolica", "Pressione diastolica", "Glucosio", "Peso corporeo", "Emoglobina"],
            "Unità": ["mmHg", "mmHg", "mg/dL", "kg", "g/dL"]
        }
        st.dataframe(pd.DataFrame(vocab_data), use_container_width=True)

    # OMOP CDM Tab
    with ehds_tabs[2]:
        st.markdown("#### OMOP Common Data Model")

        st.markdown("""
        **OMOP CDM** (Observational Medical Outcomes Partnership Common Data Model)
        standardizza i dati clinici per analisi federate cross-istituzionali.
        """)

        st.markdown("##### Domini OMOP")
        omop_domains = {
            "Dominio": ["Condition", "Drug", "Measurement", "Procedure", "Visit", "Person"],
            "Descrizione": ["Diagnosi ICD-10", "Farmaci ATC/RxNorm", "Valori lab/vitali", "Procedure mediche", "Visite/ricoveri", "Demografia"],
            "Vocabolario": ["SNOMED-CT, ICD-10", "RxNorm, ATC", "LOINC", "CPT, ICD-10-PCS", "CMS", "Standard"],
            "FL Feature": ["Label/covariate", "Drug exposure", "Valori numerici", "Procedure flags", "Tempo/costi", "Stratificazione"]
        }
        st.dataframe(pd.DataFrame(omop_domains), use_container_width=True)

        st.markdown("##### Esempio: Cohort Builder Federato")
        st.code("""
# Definizione coorte OMOP per FL
from fl_ehds.core import OMOPCohortBuilder, OMOPFederatedQuery

# Coorte: Diabetici con HbA1c > 7%
builder = OMOPCohortBuilder()
builder.add_condition(concept_id=201826)  # Type 2 DM
builder.add_measurement(
    concept_id=3004410,  # HbA1c
    value_operator=">",
    value=7.0
)

# Query federata (privacy-preserving)
query = OMOPFederatedQuery(builder, epsilon=1.0)
cohort_size = query.count()  # DP count
features = query.get_features(["age", "gender", "hba1c"])
        """, language="python")

        # Vocabulary demo
        st.markdown("##### Concept ID Standard")
        concept_data = {
            "Concept ID": [201826, 4024552, 320128, 3004410, 4182210],
            "Nome": ["Type 2 diabetes", "Hypertension", "Heart failure", "HbA1c", "BMI"],
            "Dominio": ["Condition", "Condition", "Condition", "Measurement", "Measurement"],
            "Vocabolario": ["SNOMED", "SNOMED", "SNOMED", "LOINC", "LOINC"]
        }
        st.dataframe(pd.DataFrame(concept_data), use_container_width=True)

    # IHE Profiles Tab
    with ehds_tabs[3]:
        st.markdown("#### IHE Integration Profiles")

        st.markdown("""
        **Integrating the Healthcare Enterprise (IHE)** definisce profili di integrazione
        per lo scambio sicuro di dati sanitari. Fondamentale per EHDS cross-border.
        """)

        ihe_profiles = {
            "Profilo": ["ATNA", "BPPC", "XDS.b", "XCA", "PIXm/PDQm", "XUA"],
            "Nome Completo": [
                "Audit Trail & Node Authentication",
                "Basic Patient Privacy Consents",
                "Cross-Enterprise Document Sharing",
                "Cross-Community Access",
                "Patient ID Cross-referencing (Mobile)",
                "Cross-Enterprise User Assertion"
            ],
            "Funzione EHDS": [
                "Audit trail Art. 50",
                "Gestione opt-out Art. 33",
                "Condivisione documenti",
                "Accesso cross-border",
                "Identificazione pazienti",
                "Autenticazione utenti"
            ],
            "FL Integration": [
                "Log accessi FL training",
                "Verifica consenso prima FL",
                "Metadati dataset",
                "FL cross-nazionale",
                "Linking pseudonimi",
                "SAML token per FL session"
            ]
        }
        st.dataframe(pd.DataFrame(ihe_profiles), use_container_width=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("##### ATNA Audit Logging")
            st.code("""
# Audit logging per FL conforme IHE ATNA
from fl_ehds.core import ATNAAuditLogger

logger = ATNAAuditLogger(
    audit_source_id="hospital-01",
    audit_repository_url="https://hdab.eu/audit"
)

# Log FL round
logger.log_fl_training_start(
    user_id="researcher@univ.eu",
    fl_round_id="round-42",
    model_id="diabetes-predictor",
    client_count=5
)
            """, language="python")

        with col2:
            st.markdown("##### BPPC Consent Check")
            st.code("""
# Verifica consenso prima del training
from fl_ehds.core import BPPCConsentManager

consent_mgr = BPPCConsentManager(
    organization_id="hospital-01"
)

# Filter pazienti con opt-out
permitted, excluded = consent_mgr.filter_patients_by_consent(
    patient_ids=training_cohort,
    purpose="ai-training",
    data_category="ehr"
)
# Use only 'permitted' for FL
            """, language="python")

        st.markdown("##### XCA Cross-Border Flow")
        st.markdown("""
        ```
        ┌─────────────┐        ┌─────────────┐        ┌─────────────┐
        │  Hospital   │        │   National  │        │     EU      │
        │   Italy     │◄──────►│   HDAB IT   │◄──────►│  HealthData │
        └─────────────┘        └─────────────┘        │   @EU       │
              │                      │                └──────┬──────┘
              │                      │                       │
        ┌─────┴─────┐          ┌─────┴─────┐          ┌─────┴─────┐
        │ FL Client │          │ FL Client │          │ FL Server │
        │  (Local)  │────XCA──►│ (Aggreg.) │────XCA──►│ (Central) │
        └───────────┘          └───────────┘          └───────────┘
        ```
        """)

    # HDAB API Tab
    with ehds_tabs[4]:
        st.markdown("#### Health Data Access Body (HDAB) API")

        st.markdown("""
        Gli **Health Data Access Bodies** (HDAB) sono gli organismi nazionali responsabili
        per autorizzare l'accesso secondario ai dati sanitari nell'EHDS (Art. 36-37).
        """)

        st.markdown("##### Workflow Data Permit per FL")

        permit_steps = {
            "Fase": ["1. Richiesta", "2. Valutazione", "3. Approvazione", "4. Esecuzione", "5. Audit"],
            "Attore": ["Ricercatore", "HDAB", "HDAB", "Piattaforma FL", "HDAB"],
            "Azione": [
                "Submit DataPermitApplication",
                "Review scientific merit, ethics",
                "Issue DataPermit with conditions",
                "Execute FL training",
                "Compliance monitoring"
            ],
            "API": [
                "POST /applications",
                "GET /applications/{id}",
                "GET /permits/{id}",
                "POST /permits/{id}/access",
                "GET /permits/{id}/logs"
            ]
        }
        st.dataframe(pd.DataFrame(permit_steps), use_container_width=True)

        if HAS_GOVERNANCE:
            _render_hdab_interactive()
        else:
            st.warning("Moduli governance non disponibili. Mostro solo documentazione.")
            _render_hdab_static()

        st.markdown("---")

        st.markdown("##### Opt-Out Registry (Art. 33/71)")
        st.markdown("""
        <div class="warning-box">
        <strong>Importante:</strong> I pazienti possono esercitare il diritto di opt-out
        per l'uso secondario dei loro dati (Art. 71). Il sistema FL deve verificare l'opt-out
        PRIMA di includere i dati nel training.
        </div>
        """, unsafe_allow_html=True)

        if HAS_GOVERNANCE:
            _render_optout_interactive()
        else:
            st.info("Moduli governance non disponibili per demo interattiva.")


# ===================================================================== #
#                      HDAB Interactive / Static                         #
# ===================================================================== #

def _render_hdab_interactive():
    """Interactive HDAB permit management using real governance modules."""
    import uuid as _uuid
    from datetime import datetime as _dt, timedelta as _td

    st.markdown("##### Gestione Permit Interattiva (Live)")

    # Initialize session state for HDAB with SQLite persistence
    if "governance_db" not in st.session_state:
        st.session_state.governance_db = GovernanceDB()
    if "hdab_client" not in st.session_state:
        config = HDABConfig(
            endpoint="https://hdab-it.ehds.eu/api/v1",
            simulation_mode=True,
            auth_method="oauth2",
            client_id="fl-ehds-demo",
            client_secret="demo-secret",
        )
        st.session_state.hdab_client = HDABClient(config=config)
        st.session_state.permit_store = get_shared_permit_store(db=st.session_state.governance_db)
        st.session_state.hdab_connected = False

    client = st.session_state.hdab_client
    store = st.session_state.permit_store

    # Connection
    col_conn1, col_conn2 = st.columns([1, 3])
    with col_conn1:
        if st.button("🔗 Connetti HDAB-IT", key="hdab_connect"):
            connected = asyncio.run(client.connect())
            st.session_state.hdab_connected = connected
    with col_conn2:
        if st.session_state.hdab_connected:
            st.success("Connesso a HDAB-IT (simulation mode) | OAuth2 authenticated")
        else:
            st.info("HDAB non ancora connesso. Clicca per connettere.")

    st.markdown("---")

    # Permit request form
    with st.expander("📝 Richiedi Nuovo Data Permit", expanded=False):
        with st.form("permit_request_form"):
            req_org = st.text_input("Organizzazione", value="Universita degli Studi Roma")
            req_question = st.text_input(
                "Research Question",
                value="Predictive model for diabetic retinopathy progression"
            )
            req_purpose = st.selectbox(
                "Purpose",
                options=["RESEARCH", "PUBLIC_HEALTH", "POLICY", "INNOVATION"],
            )
            req_algo = st.selectbox(
                "Algoritmo FL",
                options=["FedAvg", "FedProx", "SCAFFOLD", "FedAdam"],
            )
            col_p1, col_p2 = st.columns(2)
            with col_p1:
                req_rounds = st.number_input("FL Rounds", 10, 200, 50)
                req_epsilon = st.number_input("Privacy Budget (epsilon)", 0.1, 10.0, 1.0)
            with col_p2:
                req_clients = st.number_input("Min Clients", 2, 10, 3)
                req_duration = st.number_input("Durata (giorni)", 30, 365, 180)

            submitted = st.form_submit_button("Invia Richiesta Permit")
            if submitted:
                permit_id = str(_uuid.uuid4())[:12]
                purpose_map = {
                    "RESEARCH": PermitPurpose.SCIENTIFIC_RESEARCH,
                    "PUBLIC_HEALTH": PermitPurpose.PUBLIC_HEALTH_SURVEILLANCE,
                    "POLICY": PermitPurpose.HEALTH_POLICY,
                    "INNOVATION": PermitPurpose.AI_SYSTEM_DEVELOPMENT,
                }
                permit = DataPermit(
                    permit_id=permit_id,
                    hdab_id="HDAB-IT",
                    requester_id=req_org.lower().replace(" ", "-"),
                    purpose=purpose_map.get(req_purpose, PermitPurpose.SCIENTIFIC_RESEARCH),
                    status=PermitStatus.ACTIVE,
                    issued_at=_dt.utcnow(),
                    valid_from=_dt.utcnow(),
                    valid_until=_dt.utcnow() + _td(days=req_duration),
                    conditions={
                        "max_rounds": req_rounds,
                        "privacy_budget": req_epsilon,
                        "min_clients": req_clients,
                        "algorithm": req_algo,
                        "research_question": req_question,
                    },
                    data_categories=[DataCategory.EHR],
                    metadata={"organization": req_org},
                )
                store.register(permit)
                st.success(f"Permit **{permit_id}** creato e attivo!")

    # Show existing permits
    all_permits = store.list_all()
    if all_permits:
        st.markdown(f"##### Permit Attivi ({len(all_permits)})")
        permit_data = []
        for p in all_permits:
            permit_data.append({
                "ID": p.permit_id[:12],
                "Richiedente": p.requester_id,
                "Stato": p.status.value if hasattr(p.status, "value") else str(p.status),
                "Purpose": p.purpose.value if hasattr(p.purpose, "value") else str(p.purpose),
                "Scadenza": str(p.valid_until)[:10] if p.valid_until else "N/A",
                "Max Rounds": p.conditions.get("max_rounds", "N/A"),
                "Privacy ε": p.conditions.get("privacy_budget", "N/A"),
            })
        st.dataframe(pd.DataFrame(permit_data), use_container_width=True)

        # Permit actions
        col_act1, col_act2 = st.columns(2)
        with col_act1:
            sel_permit = st.selectbox(
                "Seleziona Permit",
                options=[p.permit_id[:12] for p in all_permits],
                key="permit_action_select",
            )
        with col_act2:
            action = st.selectbox(
                "Azione",
                options=["Verifica", "Sospendi", "Revoca"],
                key="permit_action_type",
            )
            if st.button("Esegui", key="permit_action_btn"):
                full_id = next(
                    (p.permit_id for p in all_permits if p.permit_id[:12] == sel_permit),
                    None,
                )
                if full_id:
                    if action == "Verifica":
                        p = store.get(full_id)
                        if p:
                            is_valid = p.status == PermitStatus.ACTIVE
                            if is_valid:
                                st.success(f"Permit {sel_permit} VALIDO e ATTIVO")
                            else:
                                st.warning(f"Permit {sel_permit} stato: {p.status}")
                    elif action == "Sospendi":
                        ok = store.suspend(full_id, "Sospeso da dashboard")
                        if ok:
                            st.warning(f"Permit {sel_permit} SOSPESO")
                    elif action == "Revoca":
                        ok = store.revoke(full_id, "Revocato da dashboard")
                        if ok:
                            st.error(f"Permit {sel_permit} REVOCATO")

        # Audit log
        audit = store.audit_log
        if audit:
            with st.expander(f"📋 Audit Log ({len(audit)} eventi)"):
                audit_df = pd.DataFrame(audit[-20:])
                st.dataframe(audit_df, use_container_width=True)
    else:
        st.info("Nessun permit presente. Crea un nuovo permit con il form sopra.")

    # Cross-border
    st.markdown("##### Cross-Border FL Coordination")
    st.markdown("""
    Per FL che coinvolge piu' paesi EU, il `MultiHDABCoordinator` gestisce
    i permit coordinati con gli HDAB nazionali di ogni stato membro coinvolto.
    """)

    col_cb1, col_cb2, col_cb3 = st.columns(3)
    with col_cb1:
        st.markdown("**HDAB-IT** (Lead)")
        st.markdown("Italia - Roma")
    with col_cb2:
        st.markdown("**HDAB-DE**")
        st.markdown("Germania - Berlin")
    with col_cb3:
        st.markdown("**HDAB-FR**")
        st.markdown("Francia - Paris")


def _render_hdab_static():
    """Static HDAB documentation (fallback when governance modules unavailable)."""
    st.code("""
# Setup HDAB client (simulation mode)
from governance.hdab_integration import HDABClient, HDABConfig

config = HDABConfig(endpoint="https://hdab-it.ehds.eu/api/v1", simulation_mode=True)
client = HDABClient(config=config, member_state="Italy")
await client.connect()

# Request permit
permit = await client.request_new_permit(
    requester_id="univ-roma",
    purpose=PermitPurpose.RESEARCH,
    data_categories=[DataCategory.EHR],
    conditions={"max_rounds": 100, "privacy_budget": 1.0}
)
    """, language="python")


# ===================================================================== #
#                        Opt-Out Interactive                              #
# ===================================================================== #

def _render_optout_interactive():
    """Interactive opt-out registry using real governance modules."""
    import uuid as _uuid
    from datetime import datetime as _dt

    # Initialize session state with SQLite persistence
    if "governance_db" not in st.session_state:
        st.session_state.governance_db = GovernanceDB()
    if "optout_registry" not in st.session_state:
        db = st.session_state.governance_db
        st.session_state.optout_registry = OptOutRegistry(
            cache_ttl=600,
            max_cache_size=1000,
            db=db,
        )
        # Pre-populate with sample opt-outs only if DB is empty
        if st.session_state.optout_registry.get_opted_out_count() == 0:
            sample_patients = [
                ("IT-PAT-001", "Italy", "all"),
                ("IT-PAT-042", "Italy", "category"),
                ("DE-PAT-017", "Germany", "all"),
                ("FR-PAT-099", "France", "purpose"),
            ]
            for pid, state, scope in sample_patients:
                record = OptOutRecord(
                    record_id=f"OPT-{pid}",
                    patient_id=pid,
                    member_state=state,
                    scope=scope,
                    is_active=True,
                    opt_out_date=_dt.utcnow(),
                    metadata={},
                )
                st.session_state.optout_registry.register_optout(record)

    registry = st.session_state.optout_registry

    col_opt1, col_opt2 = st.columns(2)

    with col_opt1:
        st.markdown("##### Registra Opt-Out")
        with st.form("optout_form"):
            patient_id = st.text_input("Patient ID (pseudonimo)", value="IT-PAT-")
            member_state = st.selectbox(
                "Stato Membro",
                options=["Italy", "Germany", "France", "Spain", "Netherlands"],
            )
            scope = st.selectbox(
                "Scope Opt-Out",
                options=["all", "category", "purpose"],
                help="all=nessun uso secondario; category=specifiche categorie; purpose=specifici scopi",
            )
            if st.form_submit_button("Registra Opt-Out"):
                record = OptOutRecord(
                    record_id=f"OPT-{patient_id}",
                    patient_id=patient_id,
                    member_state=member_state,
                    scope=scope,
                    is_active=True,
                    opt_out_date=_dt.utcnow(),
                    metadata={},
                )
                registry.register_optout(record)
                st.success(f"Opt-out registrato per {patient_id}")

    with col_opt2:
        st.markdown("##### Verifica Opt-Out")
        check_id = st.text_input("Patient ID da verificare", value="IT-PAT-001", key="optout_check")
        if st.button("Verifica", key="optout_check_btn"):
            is_out = registry.is_opted_out(check_id)
            if is_out:
                st.error(f"OPTED OUT: {check_id} ha esercitato il diritto di opt-out")
            else:
                st.success(f"OK: {check_id} NON ha opt-out attivo - dati utilizzabili")

    # Registry stats
    stats = registry.get_stats()
    st.markdown("##### Statistiche Registry")
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    col_s1.metric("Opt-Out Totali", stats.total_opted_out)
    col_s2.metric("Cache Size", stats.cache_size)
    col_s3.metric("Cache Hit Rate", f"{stats.cache_hit_rate:.0%}")
    col_s4.metric("Lookup Totali", stats.total_lookups)

    if stats.by_member_state:
        st.markdown("##### Opt-Out per Stato Membro")
        state_df = pd.DataFrame(
            [{"Stato": k, "Opt-Out": v} for k, v in stats.by_member_state.items()]
        )
        st.dataframe(state_df, use_container_width=True)

    # FL data filtering demo
    st.markdown("##### Filtraggio Dati per FL Training")
    st.markdown("""
    Il `OptOutChecker` filtra automaticamente i record paziente prima di ogni round FL,
    garantendo che nessun dato di pazienti con opt-out venga incluso nel training.
    """)
    demo_patients = ["IT-PAT-001", "IT-PAT-002", "IT-PAT-042", "IT-PAT-100", "DE-PAT-017", "DE-PAT-050"]
    if st.button("Simula Filtraggio Pre-Round", key="optout_filter_btn"):
        permitted = []
        excluded = []
        for pid in demo_patients:
            if registry.is_opted_out(pid):
                excluded.append(pid)
            else:
                permitted.append(pid)
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            st.success(f"Pazienti ammessi: {len(permitted)}")
            for p in permitted:
                st.markdown(f"  - {p}")
        with col_f2:
            st.error(f"Pazienti esclusi (opt-out): {len(excluded)}")
            for p in excluded:
                st.markdown(f"  - {p}")


# ===================================================================== #
#                       Infrastructure Components Tab                    #
# ===================================================================== #

def render_infrastructure_tab():
    """Render infrastructure modules tab."""
    st.markdown("### ⚙️ Infrastructure Components")

    st.markdown("""
    <div class='info-box'>
    <strong>🏗️ Enterprise Infrastructure</strong><br>
    Componenti per deployment production-grade del FL-EHDS framework.
    Ottimizzazioni per latenza, bandwidth, scalabilità e observability.
    </div>
    """, unsafe_allow_html=True)

    # Seven sub-tabs for infrastructure components
    infra_tabs = st.tabs([
        "🔐 Watermarking",
        "📡 Communication",
        "📦 Serialization",
        "💾 Caching",
        "☸️ Orchestration",
        "📊 Monitoring",
        "🔗 Cross-Silo"
    ])

    # Watermarking Tab
    with infra_tabs[0]:
        st.markdown("#### 🔐 Model Watermarking")
        st.markdown("""
        **IP Protection & Provenance Tracking**

        Protezione della proprietà intellettuale dei modelli FL per trasferimenti cross-border.
        """)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Metodi di Embedding:**")
            watermark_methods = {
                "Spread Spectrum": "Embedding nel dominio della frequenza, robusto a modifiche",
                "LSB (Least Significant Bit)": "Embedding nel dominio spaziale, alta capacità",
                "Backdoor": "Trigger-based watermarking, verificabile",
                "Passport": "Layer-specific embedding, per modelli profondi"
            }
            for method, desc in watermark_methods.items():
                st.markdown(f"- **{method}**: {desc}")

        with col2:
            st.markdown("**Caratteristiche:**")
            st.markdown("""
            - ✅ Federated watermark coordination
            - ✅ Multi-client contribution tracking
            - ✅ Verification con confidence scoring
            - ✅ EHDS provenance compliance (Art. 50)
            """)

        st.code("""
# Model Watermarking Example
from fl_ehds.core import create_watermark_manager, WatermarkType

manager = create_watermark_manager(
    watermark_type=WatermarkType.SPREAD_SPECTRUM,
    watermark_strength=0.01,
    verify_threshold=0.8
)

# Embed watermark
result = manager.embed_watermark(
    model_weights=global_model,
    watermark_data="EHDS-PERMIT-2025-001",
    owner_id="hospital_consortium_eu"
)

# Verify watermark
verification = manager.verify_watermark(
    model_weights=received_model,
    expected_watermark="EHDS-PERMIT-2025-001"
)
print(f"Confidence: {verification.confidence:.2%}")
        """, language="python")

    # Communication Tab
    with infra_tabs[1]:
        st.markdown("#### 📡 gRPC/WebSocket Communication")
        st.markdown("""
        **High-Performance Communication Layer**

        Riduzione latenza ~50% rispetto a REST per operazioni FL.
        """)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**gRPC Features:**")
            st.markdown("""
            - Bidirectional streaming per model updates
            - Connection pooling e multiplexing
            - Automatic retry con exponential backoff
            - Compression (gzip, lz4, zstd)
            - TLS/mTLS per trasferimenti sicuri
            """)

        with col2:
            st.markdown("**WebSocket Features:**")
            st.markdown("""
            - Real-time event streaming
            - Pub/Sub per metriche e notifiche
            - Low-latency monitoring
            - Connection state management
            """)

        # Performance comparison
        st.markdown("**Performance Comparison:**")
        perf_data = {
            "Metrica": ["Latenza Media", "Throughput", "Overhead Protocollo", "Streaming Support"],
            "REST": ["~100ms", "~10 MB/s", "~20%", "❌ No"],
            "gRPC": ["~50ms", "~25 MB/s", "~5%", "✅ Sì"],
            "Miglioramento": ["-50%", "+150%", "-75%", "—"]
        }
        st.table(pd.DataFrame(perf_data))

    # Serialization Tab
    with infra_tabs[2]:
        st.markdown("#### 📦 Protocol Buffers Serialization")
        st.markdown("""
        **Bandwidth Optimization ~30%**

        Serializzazione binaria efficiente per modelli e gradienti.
        """)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Caratteristiche:**")
            st.markdown("""
            - Variable-length encoding (protobuf-style)
            - Schema versioning per backward compatibility
            - Delta serialization (solo diff)
            - Quantization-aware serialization
            - Streaming per modelli grandi
            """)

        with col2:
            st.markdown("**EHDS Compliance:**")
            st.markdown("""
            - Metadata embedding per audit
            - Permit ID nei payload
            - Checksum per integrità
            - Encryption support
            """)

        st.code("""
# Serialization Example
from fl_ehds.core import create_serialization_manager

serializer = create_serialization_manager(
    format="protobuf",
    compression="balanced",
    enable_quantization=True,
    quantization_bits=8
)

# Serialize model (30% bandwidth reduction)
serialized = serializer.serialize(model_weights)
print(f"Original: {serialized.original_size} bytes")
print(f"Compressed: {len(serialized.data)} bytes")

# Delta serialization (only changes)
delta_bytes = serializer.serialize_delta(old_weights, new_weights)
print(f"Delta size: {len(delta_bytes)} bytes")
        """, language="python")

    # Caching Tab
    with infra_tabs[3]:
        st.markdown("#### 💾 Redis Distributed Caching")
        st.markdown("""
        **Fast Recovery & Checkpoint Management**

        Caching distribuito per checkpoint, stato client e metriche.
        """)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Cache Regions:**")
            cache_regions = {
                "fl:models": "Global model versions",
                "fl:checkpoints": "Training checkpoints",
                "fl:gradients": "Gradient buffers",
                "fl:clients": "Client state",
                "fl:metrics": "Training metrics",
                "fl:locks": "Distributed locks"
            }
            for region, desc in cache_regions.items():
                st.markdown(f"- `{region}`: {desc}")

        with col2:
            st.markdown("**Features:**")
            st.markdown("""
            - ✅ LRU/LFU eviction policies
            - ✅ TTL-based expiration
            - ✅ Distributed locking per aggregation
            - ✅ Compression automatica
            - ✅ Redis Cluster support
            """)

        st.code("""
# Caching Example
from fl_ehds.core import create_cache_manager

cache = create_cache_manager(backend="redis", redis_host="localhost")
await cache.connect()

# Save checkpoint
checkpoint_id = await cache.checkpoints.save_checkpoint(
    round_number=10,
    weights=global_model,
    metadata={"loss": 0.5, "accuracy": 0.85},
    permit_id="EHDS-PERMIT-001"
)

# Distributed lock for aggregation
async with cache.acquire_lock("aggregation_round_10"):
    aggregated = aggregate_updates(client_updates)
        """, language="python")

    # Orchestration Tab
    with infra_tabs[4]:
        st.markdown("#### ☸️ Kubernetes/Ray Orchestration")
        st.markdown("""
        **Enterprise Scalability**

        Orchestrazione per deployment FL su larga scala.
        """)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Kubernetes:**")
            st.markdown("""
            - Deployment management
            - Auto-scaling (CPU/Memory/Queue)
            - Rolling updates
            - Health checks & recovery
            - Multi-region federation
            """)

        with col2:
            st.markdown("**Ray:**")
            st.markdown("""
            - Distributed computing
            - Actor-based FL clients
            - Dynamic task scheduling
            - Resource management
            - Fault tolerance
            """)

        # Architecture diagram (text-based)
        st.markdown("**Deployment Architecture:**")
        st.code("""
┌─────────────────────────────────────────────────────────────┐
│                    Kubernetes Cluster                        │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ Aggregator  │  │ Aggregator  │  │  Gateway    │         │
│  │   (Pod)     │  │   (Pod)     │  │ Cross-Border│         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
│                                                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │  FL Client  │  │  FL Client  │  │  FL Client  │  ...    │
│  │  Hospital 1 │  │  Hospital 2 │  │  Hospital N │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
│                                                              │
│  ┌──────────────────────────────────────────────┐          │
│  │              Ray Cluster (Workers)            │          │
│  │   Training Tasks • Aggregation • HPO          │          │
│  └──────────────────────────────────────────────┘          │
└─────────────────────────────────────────────────────────────┘
        """)

    # Monitoring Tab
    with infra_tabs[5]:
        st.markdown("#### 📊 Prometheus/Grafana Monitoring")
        st.markdown("""
        **Production Observability**

        Monitoring completo per pipeline FL in produzione.
        """)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Metriche FL:**")
            fl_metrics = [
                "fl_ehds_rounds_total",
                "fl_ehds_round_duration_seconds",
                "fl_ehds_clients_per_round",
                "fl_ehds_global_model_loss",
                "fl_ehds_global_model_accuracy",
                "fl_ehds_bytes_transmitted_total",
                "fl_ehds_communication_latency_seconds"
            ]
            for metric in fl_metrics:
                st.markdown(f"- `{metric}`")

        with col2:
            st.markdown("**Metriche EHDS Compliance:**")
            ehds_metrics = [
                "fl_ehds_permit_validations_total",
                "fl_ehds_consent_checks_total",
                "fl_ehds_cross_border_transfers_total",
                "fl_ehds_audit_events_total",
                "fl_ehds_privacy_budget_used"
            ]
            for metric in ehds_metrics:
                st.markdown(f"- `{metric}`")

        st.markdown("**Alert Rules (predefinite):**")
        alerts_data = {
            "Alert": ["HighClientDropout", "AggregationErrors", "HighLatency", "PrivacyBudgetLow"],
            "Condizione": ["active_clients < 5", "aggregation_errors > 0", "latency > 5s", "budget_used > 90%"],
            "Severity": ["Warning", "Critical", "Warning", "Warning"]
        }
        st.table(pd.DataFrame(alerts_data))

        st.markdown("**Grafana Dashboard:**")
        st.markdown("""
        Il framework genera automaticamente dashboard Grafana con:
        - Overview panel (rounds, clients, accuracy)
        - Training progress graphs
        - Client metrics heatmaps
        - Communication metrics
        - EHDS compliance panels
        """)

    # Cross-Silo Enhancements Tab
    with infra_tabs[6]:
        st.markdown("#### 🔗 Cross-Silo Enhancements")
        st.markdown("""
        **Enterprise FL Enhancements**

        Funzionalità avanzate per deployment cross-silo in ambiente EHDS multi-istituzionale.
        """)

        # Three columns for the three main features
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**🎭 Multi-Model Federation**")
            st.markdown("""
            Ensemble di modelli federati per:
            - Diversità e robustezza
            - Miglior generalizzazione
            - Ridondanza per fault tolerance

            **Strategie:**
            - Weighted Voting
            - Stacking (meta-learner)
            - Mixture of Experts
            - Bagging/Boosting
            """)

        with col2:
            st.markdown("**🎯 Model Selection**")
            st.markdown("""
            Selezione automatica algoritmo FL:
            - Task Analysis automatica
            - Multi-Armed Bandit (UCB, Thompson)
            - Exploration/Exploitation

            **Criteri:**
            - Accuracy
            - Convergence Speed
            - Fairness
            - Privacy
            """)

        with col3:
            st.markdown("**⚡ Adaptive Aggregation**")
            st.markdown("""
            Switching dinamico tra algoritmi:
            - FedAvg → FedProx → SCAFFOLD
            - Basato su metriche runtime
            - Cooldown tra switch

            **Algoritmi supportati:**
            - FedAvg, FedProx, SCAFFOLD
            - FedAdam, FedYogi, FedNova
            - Krum, TrimmedMean, Median
            """)

        st.markdown("---")
        st.markdown("**Decision Flow:**")
        st.code("""
┌─────────────────────────────────────────────────────────────────────┐
│                    CrossSiloManager                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   ┌──────────────────┐    ┌──────────────────┐                      │
│   │  Task Analyzer   │───>│ Model Selector   │                      │
│   │ (IID/Non-IID?)   │    │ (UCB Bandit)     │                      │
│   └──────────────────┘    └────────┬─────────┘                      │
│                                    │                                 │
│                           Select Algorithm                           │
│                                    ↓                                 │
│   ┌──────────────────────────────────────────────────────────────┐  │
│   │                   Adaptive Aggregator                         │  │
│   │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ │  │
│   │  │ FedAvg  │ │ FedProx │ │SCAFFOLD │ │ FedAdam │ │  Krum   │ │  │
│   │  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘ │  │
│   └──────────────────────────────────────────────────────────────┘  │
│                                    │                                 │
│                           Switch if needed                           │
│                                    ↓                                 │
│   ┌──────────────────────────────────────────────────────────────┐  │
│   │                   Federated Ensemble                          │  │
│   │              (Multiple Global Models)                          │  │
│   │         Model 1 + Model 2 + Model 3 → Combined                │  │
│   └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
        """)

        st.markdown("**Esempio Completo:**")
        st.code("""
from fl_ehds.core import (
    create_cross_silo_manager,
    TaskType,
    DataCharacteristic
)

# Create manager with all enhancements
manager = create_cross_silo_manager(
    ensemble_strategy="weighted_voting",
    selection_criterion="accuracy",
    initial_algorithm="fedavg"
)

# Initialize with task info
manager.initialize(
    task_type=TaskType.BINARY_CLASSIFICATION,
    client_distributions=client_label_dists,
    num_clients=10,
    initial_weights=model.state_dict(),
    has_byzantine_risk=False
)

# Training loop
for round in range(100):
    # Collect client updates
    updates = [client.train() for client in clients]

    # Aggregate with adaptive selection
    global_weights = manager.aggregate_round(
        client_updates=updates,
        client_weights=[c.num_samples for c in clients],
        client_losses=[c.loss for c in clients],
        round_loss=avg_loss,
        round_accuracy=avg_accuracy
    )

    # Get status (which algorithm is being used?)
    status = manager.get_comprehensive_report()
    print(f"Round {round}: Algorithm = {status['aggregator']['current_algorithm']}")
        """, language="python")

        st.markdown("**Performance Comparison:**")
        perf_data = {
            "Scenario": ["IID Data", "Mild Non-IID", "Extreme Non-IID", "Byzantine Clients"],
            "FedAvg": ["58.2%", "55.1%", "48.3%", "35.2%"],
            "Adaptive": ["58.2%", "57.8%", "54.6%", "52.1%"],
            "Improvement": ["0%", "+2.7%", "+6.3%", "+16.9%"]
        }
        st.table(pd.DataFrame(perf_data))


# ===================================================================== #
#                            User Guide Tab                              #
# ===================================================================== #

def render_guide_tab():
    """Render user guide tab."""
    st.markdown("### 📚 Guida Utente FL-EHDS")

    st.markdown("""
    ## Come Usare la Dashboard

    ### 1. Configurazione Nodi
    - **Numero di Nodi**: Quanti ospedali/istituzioni partecipano
    - **Campioni Totali**: Dataset totale distribuito tra i nodi

    ### 2. Scelta Algoritmo
    | Scenario | Algoritmo Consigliato |
    |----------|----------------------|
    | Dati IID, baseline | FedAvg |
    | Dati Non-IID moderati | FedProx (μ=0.1) |
    | Non-IID estremo | SCAFFOLD |
    | Modelli grandi | FedAdam o FedYogi |
    | Fairness richiesta | Ditto |

    ### 3. Configurazione Eterogeneità
    | α Dirichlet | Livello Non-IID |
    |-------------|-----------------|
    | 0.1 | Estremo (ogni nodo ha 1 classe) |
    | 0.5 | Alto |
    | 1.0 | Moderato |
    | 10.0 | Quasi IID |

    ### 4. Privacy Settings
    | ε Value | Privacy Level | Uso |
    |---------|---------------|-----|
    | 0.1-1 | Molto forte | Dati ultra-sensibili |
    | 1-10 | Forte | Ricerca clinica |
    | 10-50 | Moderata | Healthcare generale |

    ### 5. Moduli Avanzati FL (Nuovi in v4)

    | Modulo | Descrizione | Quando Usare |
    |--------|-------------|--------------|
    | **Vertical FL** | Feature diverse per stessi pazienti | Multi-istituzione con dati complementari |
    | **Byzantine** | Protezione da partecipanti malevoli | Network cross-border non fidato |
    | **Continual** | Adattamento a drift temporale | Dati che evolvono nel tempo |
    | **Multi-Task** | Apprendimento simultaneo di più task | Più modelli predittivi necessari |
    | **Hierarchical** | Aggregazione multi-livello | Federazioni EU large-scale |
    | **EHDS** | Interoperabilità EU (FHIR, OMOP, IHE, HDAB) | Compliance EHDS Regulation |

    ### 6. Interpretazione Risultati
    - **Accuracy**: % predizioni corrette sul test set globale
    - **Convergenza**: Curve che salgono = modello sta imparando
    - **Privacy Spent**: Budget ε consumato (deve restare < totale)
    """)

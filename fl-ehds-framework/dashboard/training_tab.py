"""Training tab and real-time training execution for FL-EHDS dashboard."""

import time
from datetime import datetime
from typing import Dict

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Optional imports -- real PyTorch training bridge
# ---------------------------------------------------------------------------
try:
    from dashboard.real_trainer_bridge import (
        RealFLTrainer,
        RealImageFLTrainer,
    )
    HAS_REAL_TRAINING = True
except ImportError:
    try:
        from real_trainer_bridge import (
            RealFLTrainer,
            RealImageFLTrainer,
        )
        HAS_REAL_TRAINING = True
    except ImportError:
        HAS_REAL_TRAINING = False

# ---------------------------------------------------------------------------
# Optional imports -- training monitor
# ---------------------------------------------------------------------------
try:
    from dashboard.training_monitor import TrainingMonitor, run_monitored_training
    HAS_TRAINING_MONITOR = True
except ImportError:
    try:
        from training_monitor import TrainingMonitor, run_monitored_training
        HAS_TRAINING_MONITOR = True
    except ImportError:
        HAS_TRAINING_MONITOR = False

# ---------------------------------------------------------------------------
# Imports from split dashboard modules
# ---------------------------------------------------------------------------
from dashboard.simulators import FLSimulatorV4
from dashboard.constants import show_algorithm_help as _show_algorithm_help
from dashboard.constants import show_model_help as _show_model_help


def _get_simulator(config: Dict):
    """Instantiate FLSimulatorV4."""
    return FLSimulatorV4(config)


# ============================================================================
# Public API
# ============================================================================

__all__ = [
    "render_training_tab",
    "run_training_v4",
    "run_training_real_tabular",
    "run_training_real_imaging",
]


# ============================================================================
# Governance helpers
# ============================================================================


def _render_governance_training_banner(config: Dict) -> bool:
    """Render governance status banner in Training tab.

    Shows governance state when configured via EHDS Governance tab,
    and provides toggle to enable governance during training.

    Returns:
        True if governance is enabled for training.
    """
    gov_bridge = st.session_state.get("gov_bridge")
    gov_result = st.session_state.get("gov_pre_training_result")

    if gov_bridge is None:
        # No governance configured - show hint
        with st.expander("EHDS Governance (opzionale)", expanded=False):
            st.info(
                "Per abilitare la governance EHDS durante il training, "
                "configura ed esegui il pre-training nel tab "
                "**EHDS Governance** (Configurazione + Esecuzione Lifecycle)."
            )
        return False

    # Governance is configured - show status and toggle
    gov_config = st.session_state.get("gov_config", {})

    with st.container():
        st.markdown("##### EHDS Governance Attiva")

        g_cols = st.columns(4)
        with g_cols[0]:
            hdab_status = gov_result.get("hdab_status", {}) if gov_result else {}
            connected = sum(1 for v in hdab_status.values() if v)
            st.metric("HDAB Connessi", f"{connected}/{len(hdab_status)}")
        with g_cols[1]:
            permits = gov_result.get("permits", {}) if gov_result else {}
            st.metric("Permit Attivi", len(permits))
        with g_cols[2]:
            budget = gov_bridge.get_budget_status()
            st.metric(
                "Budget Privacy",
                f"eps={budget.get('total', 0):.1f}",
                delta=f"-{budget.get('used', 0):.4f} usato",
            )
        with g_cols[3]:
            violations = gov_result.get("purpose_violations", []) if gov_result else []
            if violations:
                st.metric("Violazioni", len(violations))
            else:
                st.metric("Stato", "OK")

        # Governance toggle
        use_governance = st.checkbox(
            "Abilita Governance durante Training",
            value=True,
            help=(
                "Valida ogni round FL con il permit attivo, "
                "traccia il budget privacy epsilon, e genera "
                "audit trail GDPR Art. 30 automaticamente."
            ),
            key="training_use_governance",
        )

        if use_governance:
            modules = []
            if gov_config.get("data_minimization_enabled"):
                modules.append("Minimizzazione")
            if gov_config.get("secure_processing_enabled"):
                modules.append("Secure Processing")
            if gov_config.get("fee_model_enabled"):
                modules.append("Fee Model")
            if gov_config.get("data_quality_enabled"):
                modules.append("Data Quality")
            if gov_config.get("hdab_routing_enabled"):
                modules.append("HDAB Routing")
            mod_str = ", ".join(modules) if modules else "Nessuno"
            st.caption(
                f"Finalita: {gov_config.get('purpose', 'N/A')} | "
                f"Moduli attivi: {mod_str}"
            )

        st.markdown("---")
        return use_governance


def _generate_post_training_compliance(
    config: Dict, training_results: Dict, gov_bridge, gov_config: Dict,
):
    """Generate EHDSComplianceReport after governance-aware training.

    Creates a compliance report using training results and governance state,
    then stores it in session state for viewing in the Compliance Dashboard.
    """
    from governance.ehds_compliance_report import (
        ArticleAssessment,
        ComplianceStatus,
        EHDSComplianceReport,
        ARTICLE_REGISTRY,
    )

    report = EHDSComplianceReport()
    report.generated_at = datetime.now().isoformat(timespec="seconds")
    report.scenario_label = (
        f"{config.get('algorithm', 'FedAvg')} | "
        f"{len(gov_config.get('countries', []))} paesi | "
        f"{len(gov_config.get('hospitals', []))} ospedali | "
        f"purpose={gov_config.get('purpose', 'N/A')}"
    )
    report.training_config = {**config, **gov_config}
    report.final_metrics = {
        "accuracy": training_results.get("final_accuracy", 0),
        "loss": training_results.get("final_loss", 0),
        "f1": training_results.get("final_f1", 0),
        "auc": training_results.get("final_auc", 0),
    }

    budget = gov_bridge.get_budget_status()
    permits_summary = gov_bridge.get_permits_summary()
    min_report = gov_bridge.get_minimization_report()

    for chapter, art_id, title, _ in ARTICLE_REGISTRY:
        status = ComplianceStatus.NOT_ASSESSED
        evidence = "Non valutato"

        # Art. 33 - Permitted Purposes
        if art_id == "Art. 33":
            from governance.ehds_compliance_report import ARTICLE_53_PURPOSES
            if gov_config.get("purpose") in ARTICLE_53_PURPOSES:
                status = ComplianceStatus.COMPLIANT
                evidence = (
                    f"Purpose '{gov_config['purpose']}' validato Art. 53. "
                    f"Training completato: {training_results.get('total_rounds', 0)} round."
                )
            else:
                status = ComplianceStatus.NON_COMPLIANT
                evidence = f"Purpose '{gov_config.get('purpose')}' non valido Art. 53"

        # Art. 34 - Data Categories
        elif art_id == "Art. 34":
            status = ComplianceStatus.COMPLIANT
            evidence = f"Dataset: {gov_config.get('dataset_type', 'tabular')}"

        # Art. 42 - Fee Model
        elif art_id == "Art. 42":
            if gov_config.get("fee_model_enabled"):
                status = ComplianceStatus.COMPLIANT
                evidence = "Fee model calcolato pre-training"
            else:
                status = ComplianceStatus.NOT_ASSESSED
                evidence = "Fee model non abilitato"

        # Art. 44 - Data Minimization
        elif art_id == "Art. 44":
            if min_report:
                status = ComplianceStatus.COMPLIANT
                evidence = (
                    f"Minimizzazione applicata: "
                    f"{min_report['original_features']} -> "
                    f"{min_report['kept_features']} features "
                    f"(-{min_report['reduction_pct']}%)"
                )
            elif gov_config.get("data_minimization_enabled"):
                status = ComplianceStatus.PARTIAL
                evidence = "Minimizzazione abilitata ma non applicata (nessun dato tabulare)"
            else:
                status = ComplianceStatus.NOT_ASSESSED
                evidence = "Minimizzazione non abilitata"

        # Art. 46 - Privacy (DP)
        elif art_id == "Art. 46":
            if config.get("use_dp"):
                pct = budget.get("utilization_pct", 0)
                status = ComplianceStatus.COMPLIANT
                evidence = (
                    f"DP attiva: eps={budget.get('total', 0):.1f}, "
                    f"consumato={budget.get('used', 0):.4f} ({pct:.1f}%), "
                    f"rimanente={budget.get('remaining', 0):.4f}"
                )
            else:
                status = ComplianceStatus.PARTIAL
                evidence = "DP non abilitata - privacy via FL senza rumore"

        # Art. 48 - Audit Trail
        elif art_id == "Art. 48":
            rounds_completed = budget.get("rounds_completed", 0)
            if rounds_completed > 0:
                status = ComplianceStatus.COMPLIANT
                evidence = (
                    f"Audit trail: {rounds_completed} round loggati "
                    f"con permit validation + budget tracking"
                )
            else:
                status = ComplianceStatus.PARTIAL
                evidence = "Audit trail inizializzato, nessun round loggato"

        # Art. 50 - Secure Processing
        elif art_id == "Art. 50":
            if gov_config.get("secure_processing_enabled"):
                status = ComplianceStatus.COMPLIANT
                evidence = "TEE/Watermark/TimeGuard abilitato"
            else:
                status = ComplianceStatus.PARTIAL
                evidence = "FL inherently secure (dati non condivisi)"

        # Art. 53 - Data Permits
        elif art_id == "Art. 53":
            n_permits = permits_summary.get("total_permits", 0)
            if n_permits > 0:
                status = ComplianceStatus.COMPLIANT
                evidence = (
                    f"{n_permits} permit emessi e validati. "
                    f"Training eseguito con validazione per-round."
                )
            else:
                status = ComplianceStatus.NON_COMPLIANT
                evidence = "Nessun permit emesso"

        # Art. 57-58 - Cross-border
        elif "Art. 57" in art_id:
            n_countries = len(gov_config.get("countries", []))
            if n_countries > 1:
                if gov_config.get("hdab_routing_enabled"):
                    status = ComplianceStatus.COMPLIANT
                    evidence = f"SAP cross-border: {n_countries} paesi, HDAB routing attivo"
                else:
                    status = ComplianceStatus.PARTIAL
                    evidence = f"{n_countries} paesi, routing HDAB non abilitato"
            else:
                status = ComplianceStatus.NOT_ASSESSED
                evidence = "Singolo paese, cross-border non applicabile"

        # Art. 69 - Data Quality
        elif art_id == "Art. 69":
            if gov_config.get("data_quality_enabled"):
                status = ComplianceStatus.COMPLIANT
                evidence = "Quality framework GOLD/SILVER/BRONZE attivo"
            else:
                status = ComplianceStatus.NOT_ASSESSED
                evidence = "Quality framework non abilitato"

        # Art. 71 - Opt-out
        elif art_id == "Art. 71":
            status = ComplianceStatus.COMPLIANT
            evidence = "OptOutRegistry disponibile (Art. 71 EHDS)"

        # GDPR Art. 30 - Records of Processing
        elif "GDPR" in art_id or "Art. 30" in art_id:
            status = ComplianceStatus.COMPLIANT
            evidence = (
                f"Audit trail completo: {training_results.get('total_rounds', 0)} round, "
                f"permit validation, budget tracking, governance events"
            )

        assessment = ArticleAssessment(
            article_id=art_id,
            title=title,
            chapter=chapter,
            status=status,
            evidence=evidence,
        )
        report.assessments.append(assessment)

    # Store in session state for Compliance Dashboard
    st.session_state.gov_compliance_report = report
    st.session_state.last_compliance_report = report
    return report


def _show_compliance_summary(report):
    """Show a compact compliance summary after training."""
    if not report or not report.assessments:
        return

    compliant = sum(1 for a in report.assessments if a.status.value == "COMPLIANT")
    partial = sum(1 for a in report.assessments if a.status.value == "PARTIAL")
    non_compliant = sum(1 for a in report.assessments if a.status.value == "NON_COMPLIANT")
    total = len(report.assessments)

    with st.expander("Report Compliance EHDS (post-training)", expanded=True):
        comp_cols = st.columns(4)
        comp_cols[0].metric("Compliant", compliant)
        comp_cols[1].metric("Parziale", partial)
        comp_cols[2].metric("Non Compliant", non_compliant)
        comp_cols[3].metric("Totale Articoli", total)

        # Key articles summary table
        key_articles = [
            a for a in report.assessments
            if a.article_id in ("Art. 33", "Art. 44", "Art. 46", "Art. 48",
                                "Art. 50", "Art. 53", "Art. 57-58", "Art. 69", "Art. 71")
        ]
        if key_articles:
            rows = []
            status_icons = {
                "COMPLIANT": "COMPLIANT",
                "PARTIAL": "PARTIAL",
                "NOT_ASSESSED": "N/A",
                "NON_COMPLIANT": "NON COMPLIANT",
            }
            for a in key_articles:
                rows.append({
                    "Articolo": a.article_id,
                    "Titolo": a.title[:40],
                    "Stato": status_icons.get(a.status.value, a.status.value),
                    "Evidenza": a.evidence[:60],
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.caption(
            "Report completo disponibile nel tab EHDS Governance > Dashboard Compliance"
        )


# =============================================================================
# TAB RENDERERS
# =============================================================================

def render_training_tab(config: Dict):
    """Render training tab."""
    st.markdown("### Federated Learning Training")

    mode = config.get("training_mode", "simulation")
    mode_labels = {
        "simulation": "Simulazione NumPy",
        "real_tabular": "PyTorch Reale (Tabulare)",
        "real_imaging": "PyTorch Reale (Imaging CNN)",
    }

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Nodi", config['num_nodes'])
    with col2:
        st.metric("Algoritmo", config['algorithm'])
    with col3:
        st.metric("Modalita'", mode_labels.get(mode, mode))
    with col4:
        st.metric("Rounds", config['num_rounds'])
    with col5:
        dp_str = f"eps={config['epsilon']}" if config['use_dp'] else "Off"
        st.metric("DP", dp_str)

    if mode == "real_imaging" and config.get("selected_dataset"):
        ds = config["selected_dataset"]
        st.info(
            f"Dataset: **{ds['name']}** | "
            f"Classi: {ds['num_classes']} | "
            f"Immagini: {ds['total_images']:,} | "
            f"Img Size: {config.get('img_size', 64)}px"
        )
    elif mode == "real_tabular" and config.get("selected_tabular_dataset"):
        _tab_ds_labels = {
            "diabetes": "Diabetes 130-US (101K encounters, 22 features)",
            "heart_disease": "Heart Disease UCI (920 pazienti, 13 features)",
        }
        _tab_label = _tab_ds_labels.get(config["selected_tabular_dataset"], "Sintetico Healthcare")
        st.info(f"Dataset Tabular: **{_tab_label}**")

    # Governance status banner (only for real training modes)
    use_governance = False
    if mode.startswith("real_"):
        use_governance = _render_governance_training_banner(config)

    st.markdown("---")

    with st.expander("Informazioni sull'Algoritmo Selezionato", expanded=False):
        _show_algorithm_help(config['algorithm'])

    with st.expander("Informazioni sul Modello Selezionato", expanded=False):
        _show_model_help(config['model'])

    if st.button("Avvia Training", type="primary", use_container_width=True):
        if mode == "simulation":
            run_training_v4(config)
        elif mode == "real_tabular":
            run_training_real_tabular(config, use_governance=use_governance)
        elif mode == "real_imaging":
            run_training_real_imaging(config, use_governance=use_governance)

    # Show persisted results from previous run (survives Streamlit reruns)
    if "last_training_results" in st.session_state and st.session_state.last_training_results:
        prev = st.session_state.last_training_results
        with st.expander("Risultati ultimo training", expanded=True):
            st.markdown(f"**Algoritmo:** {prev.get('algorithm', '?')} | "
                        f"**Rounds:** {prev.get('num_rounds', '?')} | "
                        f"**Modalita':** {prev.get('mode', '?')}")
            final = prev.get("final_metrics", {})
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Accuracy", f"{final.get('accuracy', 0):.3f}")
            col_b.metric("F1 Score", f"{final.get('f1', 0):.3f}")
            col_c.metric("AUC", f"{final.get('auc', 0):.3f}")
            col_d, col_e, col_f = st.columns(3)
            col_d.metric("Loss", f"{final.get('loss', 0):.4f}")
            col_e.metric("Precision", f"{final.get('precision', 0):.3f}")
            col_f.metric("Recall", f"{final.get('recall', 0):.3f}")
            if prev.get("history"):
                st.caption(f"Training completato con {len(prev['history'])} round")

    # Show compliance report if generated during last training
    if "last_compliance_report" in st.session_state and st.session_state.last_compliance_report:
        _show_compliance_summary(st.session_state.last_compliance_report)


def run_training_v4(config: Dict):
    """Run training with NumPy simulation."""
    simulator = _get_simulator(config)

    progress = st.progress(0)
    status = st.empty()

    col1, col2 = st.columns(2)

    with col1:
        acc_chart = st.empty()
    with col2:
        metrics_display = st.empty()

    for r in range(1, config['num_rounds'] + 1):
        result = simulator.train_round(r)

        progress.progress(r / config['num_rounds'])
        status.markdown(
            f"**Round {r}/{config['num_rounds']}** | "
            f"Accuracy: {result['global_accuracy']:.2%} | "
            f"Participants: {len(result['participating'])}/{config['num_nodes']}"
        )

        if r % 5 == 0 or r == config['num_rounds']:
            accs = [h['global_accuracy'] for h in simulator.history]
            rounds_x = list(range(1, len(accs) + 1))
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=rounds_x, y=accs,
                mode='lines', line=dict(color='blue', width=2),
                name='Accuracy',
            ))
            fig.add_trace(go.Scatter(
                x=rounds_x, y=accs,
                mode='none', fill='tozeroy',
                fillcolor='rgba(0,0,255,0.15)', showlegend=False,
            ))
            fig.update_layout(
                title=f"{config['algorithm']} - Training Convergence (Simulation)",
                xaxis_title='Round',
                yaxis_title='Accuracy',
                yaxis=dict(range=[0.4, 0.85]),
                template='plotly_white',
            )
            acc_chart.plotly_chart(fig, use_container_width=True)

        time.sleep(0.02)

    final_acc = simulator.history[-1]['global_accuracy']
    status.success(f"Training Completato! Accuracy Finale: {final_acc:.2%}")

    with metrics_display:
        df = pd.DataFrame([
            {
                "Nodo": f"Node {i+1}",
                "Accuracy": f"{result['node_metrics'][i]['accuracy']:.2%}",
                "Campioni": result['node_metrics'][i]['samples'],
                "Partecipa": "Y" if result['node_metrics'][i]['participating'] else "-"
            }
            for i in range(config['num_nodes'])
        ])
        st.dataframe(df, use_container_width=True)

    # Persist results in session state
    st.session_state.last_training_results = {
        "algorithm": config['algorithm'],
        "num_rounds": config['num_rounds'],
        "mode": "simulation",
        "final_metrics": {
            "accuracy": final_acc, "loss": 0, "f1": 0,
            "precision": 0, "recall": 0, "auc": 0,
        },
        "history": [{"accuracy": h["global_accuracy"]} for h in simulator.history],
    }


def run_training_real_tabular(config: Dict, use_governance: bool = False):
    """Run real PyTorch training on tabular data (synthetic, diabetes, or heart disease)."""
    if not HAS_REAL_TRAINING:
        st.error("Modulo real_trainer_bridge non disponibile.")
        return

    tabular_ds = config.get("selected_tabular_dataset")
    bridge_config = {
        "num_clients": config["num_nodes"],
        "samples_per_client": config["total_samples"] // config["num_nodes"],
        "algorithm": config["algorithm"],
        "local_epochs": config["local_epochs"],
        "batch_size": 32,
        "learning_rate": config["learning_rate"],
        "is_iid": config.get("label_skew_alpha", 0.5) > 5.0,
        "alpha": config.get("label_skew_alpha", 0.5),
        "use_dp": config["use_dp"],
        "epsilon": config["epsilon"],
        "clip_norm": config["clip_norm"],
        "seed": config["random_seed"],
        "fedprox_mu": config.get("fedprox_mu", 0.1),
        "server_lr": config.get("server_lr", 0.1),
        "beta1": config.get("beta1", 0.9),
        "beta2": config.get("beta2", 0.99),
        "tau": config.get("tau", 1e-3),
        "tabular_dataset": tabular_ds,
    }

    # Resolve governance bridge if enabled
    gov_bridge = None
    gov_config = None
    if use_governance:
        gov_bridge = st.session_state.get("gov_bridge")
        gov_config = st.session_state.get("gov_config", {})
        if gov_bridge is None:
            st.warning("Governance abilitata ma bridge non disponibile. Training senza governance.")
            use_governance = False

    if HAS_TRAINING_MONITOR:
        monitor = TrainingMonitor()
        monitor.setup(show_governance=use_governance)

        try:
            trainer = RealFLTrainer(bridge_config)

            if use_governance and gov_bridge is not None:
                # Governance-aware training
                progress_cb = monitor.create_progress_callback()
                gov_cb = monitor.create_governance_callback(gov_bridge)

                results = trainer.train_with_governance(
                    num_rounds=config["num_rounds"],
                    governance_bridge=gov_bridge,
                    progress_callback=progress_cb,
                    governance_callback=gov_cb,
                )
            else:
                # Standard training
                callback = monitor.create_progress_callback()
                results = trainer.train(config["num_rounds"], callback)

            monitor.show_final_summary(results)

            mode_label = "real_pytorch_governance" if use_governance else "real_pytorch"
            st.session_state.last_training_results = {
                "algorithm": config["algorithm"],
                "num_rounds": config["num_rounds"],
                "mode": mode_label,
                "final_metrics": {
                    "accuracy": results["final_accuracy"],
                    "loss": results["final_loss"],
                    "f1": results["final_f1"],
                    "auc": results["final_auc"],
                },
                "history": results["history"],
            }

            # Post-training compliance report generation
            if use_governance and gov_bridge is not None:
                report = _generate_post_training_compliance(
                    config, results, gov_bridge, gov_config,
                )
                _show_compliance_summary(report)

        except Exception as e:
            st.error(f"Errore training: {e}")
    else:
        # Fallback: old pattern without TrainingMonitor
        progress = st.progress(0)
        status = st.empty()
        col1, col2 = st.columns(2)
        with col1:
            chart_area = st.empty()
        with col2:
            metrics_area = st.empty()

        history = []

        def progress_callback(round_num, total, metrics):
            history.append(metrics)
            progress.progress(round_num / total)
            status.markdown(
                f"**Round {round_num}/{total}** | "
                f"Acc: {metrics['accuracy']:.3f} | "
                f"Loss: {metrics['loss']:.4f} | "
                f"F1: {metrics['f1']:.3f} | "
                f"AUC: {metrics['auc']:.3f}"
            )
            if round_num % 3 == 0 or round_num == total:
                _update_real_training_chart(chart_area, history, config["algorithm"])

        try:
            trainer = RealFLTrainer(bridge_config)
            results = trainer.train(config["num_rounds"], progress_callback)
            _show_real_training_results(status, metrics_area, results)
        except Exception as e:
            st.error(f"Errore training: {e}")


def run_training_real_imaging(config: Dict, use_governance: bool = False):
    """Run real PyTorch CNN training on clinical imaging dataset."""
    if not HAS_REAL_TRAINING:
        st.error("Modulo real_trainer_bridge non disponibile.")
        return

    ds = config.get("selected_dataset")
    if not ds:
        st.error("Nessun dataset selezionato. Scegli un dataset nella sidebar.")
        return

    bridge_config = {
        "data_dir": ds["path"],
        "num_clients": config["num_nodes"],
        "algorithm": config["algorithm"],
        "local_epochs": config["local_epochs"],
        "batch_size": 32,
        "learning_rate": 0.001,
        "is_iid": config.get("label_skew_alpha", 0.5) > 5.0,
        "alpha": config.get("label_skew_alpha", 0.5),
        "use_dp": config["use_dp"],
        "epsilon": config["epsilon"],
        "clip_norm": config["clip_norm"],
        "seed": config["random_seed"],
        "fedprox_mu": config.get("fedprox_mu", 0.1),
        "img_size": config.get("img_size", 64),
        "server_lr": config.get("server_lr", 0.1),
        "beta1": config.get("beta1", 0.9),
        "beta2": config.get("beta2", 0.99),
        "tau": config.get("tau", 1e-3),
    }

    st.warning(
        f"Training CNN su **{ds['name']}** ({ds['total_images']:,} immagini). "
        f"Questo puo' richiedere diversi minuti su CPU."
    )

    # Resolve governance bridge if enabled
    gov_bridge = None
    gov_config = None
    if use_governance:
        gov_bridge = st.session_state.get("gov_bridge")
        gov_config = st.session_state.get("gov_config", {})
        if gov_bridge is None:
            st.warning("Governance abilitata ma bridge non disponibile. Training senza governance.")
            use_governance = False

    if HAS_TRAINING_MONITOR:
        monitor = TrainingMonitor()
        monitor.setup(show_governance=use_governance)

        try:
            trainer = RealImageFLTrainer(bridge_config)

            if use_governance and gov_bridge is not None:
                # Governance-aware training via manual loop
                # (RealImageFLTrainer doesn't have train_with_governance,
                # so we use the same pattern as run_monitored_training)
                progress_cb = monitor.create_progress_callback()
                gov_cb = monitor.create_governance_callback(gov_bridge)

                epsilon_per_round = (
                    config["epsilon"] / config["num_rounds"]
                    if config.get("use_dp")
                    else 0.0
                )

                history = []
                for r in range(config["num_rounds"]):
                    # Pre-round governance check
                    ok, reason = gov_cb(r, epsilon_per_round)
                    if not ok:
                        monitor.update_governance_event({
                            "type": "training_stopped",
                            "round": r,
                            "message": f"Training interrotto: {reason}",
                        })
                        st.warning(f"Training interrotto al round {r}: {reason}")
                        break

                    result = trainer.trainer.train_round(r)
                    round_data = {
                        "round": r + 1,
                        "accuracy": result.global_acc,
                        "loss": result.global_loss,
                        "f1": result.global_f1,
                        "precision": result.global_precision,
                        "recall": result.global_recall,
                        "auc": result.global_auc,
                        "time": result.time_seconds,
                        "node_metrics": {
                            cr.client_id: {
                                "accuracy": cr.train_acc,
                                "loss": cr.train_loss,
                                "samples": cr.num_samples,
                            }
                            for cr in result.client_results
                        },
                    }
                    history.append(round_data)
                    progress_cb(r + 1, config["num_rounds"], round_data)
                    gov_bridge.log_round_completion(r, result, epsilon_per_round)

                # End governance session
                if history:
                    final = history[-1]
                    gov_bridge.end_session(
                        total_rounds=len(history),
                        final_metrics={
                            "accuracy": final["accuracy"],
                            "loss": final["loss"],
                            "f1": final["f1"],
                            "auc": final["auc"],
                        },
                        success=True,
                    )

                final = history[-1] if history else {}
                results = {
                    "history": history,
                    "final_accuracy": final.get("accuracy", 0),
                    "final_loss": final.get("loss", 0),
                    "final_f1": final.get("f1", 0),
                    "final_auc": final.get("auc", 0),
                    "algorithm": config["algorithm"],
                    "num_clients": config["num_nodes"],
                    "total_rounds": len(history),
                    "training_mode": "real_pytorch_imaging_governance",
                }
            else:
                # Standard training
                callback = monitor.create_progress_callback()
                results = trainer.train(config["num_rounds"], callback)

            monitor.show_final_summary(results)

            mode_label = (
                "real_pytorch_imaging_governance" if use_governance
                else "real_pytorch_imaging"
            )
            st.session_state.last_training_results = {
                "algorithm": config["algorithm"],
                "num_rounds": config["num_rounds"],
                "mode": mode_label,
                "final_metrics": {
                    "accuracy": results["final_accuracy"],
                    "loss": results["final_loss"],
                    "f1": results["final_f1"],
                    "auc": results["final_auc"],
                },
                "history": results["history"],
            }

            # Post-training compliance report generation
            if use_governance and gov_bridge is not None:
                report = _generate_post_training_compliance(
                    config, results, gov_bridge, gov_config,
                )
                _show_compliance_summary(report)

        except Exception as e:
            st.error(f"Errore training imaging: {e}")
    else:
        # Fallback: old pattern without TrainingMonitor
        progress = st.progress(0)
        status = st.empty()
        col1, col2 = st.columns(2)
        with col1:
            chart_area = st.empty()
        with col2:
            metrics_area = st.empty()

        history = []

        def progress_callback(round_num, total, metrics):
            history.append(metrics)
            progress.progress(round_num / total)
            status.markdown(
                f"**Round {round_num}/{total}** | "
                f"Acc: {metrics['accuracy']:.3f} | "
                f"Loss: {metrics['loss']:.4f} | "
                f"F1: {metrics['f1']:.3f} | "
                f"AUC: {metrics['auc']:.3f}"
            )
            _update_real_training_chart(chart_area, history, config["algorithm"])

        try:
            trainer = RealImageFLTrainer(bridge_config)
            results = trainer.train(config["num_rounds"], progress_callback)
            _show_real_training_results(status, metrics_area, results)
        except Exception as e:
            st.error(f"Errore training imaging: {e}")


def _update_real_training_chart(chart_area, history, algorithm):
    """Update live training chart with 6 metrics."""
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=["Accuracy", "Loss", "F1 Score", "Precision", "Recall", "AUC"],
    )

    rounds = list(range(1, len(history) + 1))
    metrics_map = [
        ("accuracy", "Accuracy", 1, 1, "blue"),
        ("loss", "Loss", 1, 2, "red"),
        ("f1", "F1 Score", 1, 3, "green"),
        ("precision", "Precision", 2, 1, "orange"),
        ("recall", "Recall", 2, 2, "purple"),
        ("auc", "AUC", 2, 3, "brown"),
    ]

    for key, name, row, col, color in metrics_map:
        vals = [h.get(key, 0) for h in history]
        fig.add_trace(
            go.Scatter(
                x=rounds, y=vals,
                mode='lines', line=dict(color=color, width=2),
                name=name, showlegend=False,
            ),
            row=row, col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=rounds, y=vals,
                mode='none', fill='tozeroy',
                fillcolor=f'rgba({",".join(_color_to_rgb(color))},0.15)',
                showlegend=False,
            ),
            row=row, col=col,
        )
        fig.update_xaxes(title_text="Round", row=row, col=col)

    fig.update_layout(
        title=dict(text=f"{algorithm} - Real PyTorch Training", font=dict(size=14)),
        template='plotly_white',
        height=500,
    )
    chart_area.plotly_chart(fig, use_container_width=True)


def _color_to_rgb(color_name: str) -> list:
    """Convert a named color to an RGB triplet (as strings) for rgba()."""
    _map = {
        "blue": ["0", "0", "255"],
        "red": ["255", "0", "0"],
        "green": ["0", "128", "0"],
        "orange": ["255", "165", "0"],
        "purple": ["128", "0", "128"],
        "brown": ["139", "69", "19"],
    }
    return _map.get(color_name, ["100", "100", "100"])


def _show_real_training_results(status_area, metrics_area, results):
    """Display final results after real training."""
    status_area.success(
        f"Training Completato! "
        f"Acc: {results['final_accuracy']:.3f} | "
        f"F1: {results['final_f1']:.3f} | "
        f"AUC: {results['final_auc']:.3f}"
    )

    with metrics_area:
        st.markdown("#### Metriche Finali")
        final = results["history"][-1] if results.get("history") else {}

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Accuracy", f"{final.get('accuracy', 0):.3f}")
        col_b.metric("F1 Score", f"{final.get('f1', 0):.3f}")
        col_c.metric("AUC", f"{final.get('auc', 0):.3f}")

        col_d, col_e, col_f = st.columns(3)
        col_d.metric("Loss", f"{final.get('loss', 0):.4f}")
        col_e.metric("Precision", f"{final.get('precision', 0):.3f}")
        col_f.metric("Recall", f"{final.get('recall', 0):.3f}")

        if final.get("node_metrics"):
            st.markdown("#### Metriche per Nodo")
            node_data = []
            for node_id, nm in sorted(final["node_metrics"].items()):
                node_data.append({
                    "Nodo": f"Hospital {node_id}",
                    "Accuracy": f"{nm.get('accuracy', 0):.3f}",
                    "Loss": f"{nm.get('loss', 0):.4f}",
                    "Campioni": nm.get("samples", 0),
                })
            st.dataframe(pd.DataFrame(node_data), use_container_width=True)

    # Persist results in session state so they survive Streamlit reruns
    st.session_state.last_training_results = {
        "algorithm": results.get("algorithm", ""),
        "num_rounds": results.get("num_rounds", len(results.get("history", []))),
        "mode": results.get("mode", "real"),
        "final_metrics": final,
        "history": results.get("history", []),
    }

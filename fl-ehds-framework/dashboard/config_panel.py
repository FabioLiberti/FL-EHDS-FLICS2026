"""Configuration panel and visualization utilities for FL-EHDS dashboard."""

import sys
from pathlib import Path
from typing import Dict

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import networkx as nx

from dashboard.constants import (
    ALGORITHMS,
    MODELS,
    HETEROGENEITY_TYPES,
    PARTICIPATION_MODES,
    BYZANTINE_METHODS,
    CONTINUAL_METHODS,
    MULTITASK_METHODS,
    EHDS_TASKS,
    show_algorithm_help,
    show_model_help,
)

# Import real PyTorch training bridge (optional)
try:
    from dashboard.real_trainer_bridge import (
        RealFLTrainer, RealImageFLTrainer,
        discover_datasets, check_pytorch_available,
        create_streamlit_progress_callback,
    )
    HAS_REAL_TRAINING = True
except ImportError:
    try:
        from real_trainer_bridge import (
            RealFLTrainer, RealImageFLTrainer,
            discover_datasets, check_pytorch_available,
            create_streamlit_progress_callback,
        )
        HAS_REAL_TRAINING = True
    except ImportError:
        HAS_REAL_TRAINING = False

__all__ = [
    "plot_vertical_fl_architecture",
    "plot_hierarchy_tree",
    "_load_yaml_defaults",
    "_cfg_default",
    "create_config_panel",
    "HAS_REAL_TRAINING",
]


# =============================================================================
# VISUALISATION HELPERS
# =============================================================================

def plot_vertical_fl_architecture(num_parties: int = 3):
    """Plot Vertical FL / SplitNN architecture."""
    party_y = 0.7
    party_colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']

    shapes = []
    annotations = []

    # Party boxes and arrows
    for i in range(num_parties):
        x = 0.15 + i * (0.7 / num_parties)
        color = party_colors[i % len(party_colors)]

        # Rounded rectangle for party box
        shapes.append(dict(
            type="rect",
            x0=x - 0.08, y0=party_y - 0.15,
            x1=x + 0.08, y1=party_y + 0.15,
            xref="paper", yref="paper",
            fillcolor=color, opacity=0.7,
            line=dict(color=color, width=1),
        ))

        # Party label
        annotations.append(dict(
            x=x, y=party_y,
            xref="paper", yref="paper",
            text=f"<b>Party {i+1}</b><br>(Features {i*5+1}-{(i+1)*5})",
            showarrow=False,
            font=dict(size=11, color="white"),
            align="center",
        ))

        # Line from party to coordinator
        shapes.append(dict(
            type="line",
            x0=x, y0=party_y - 0.15,
            x1=0.5, y1=0.35,
            xref="paper", yref="paper",
            line=dict(color="gray", width=2),
        ))

    # Coordinator box
    shapes.append(dict(
        type="rect",
        x0=0.35, y0=0.1,
        x1=0.65, y1=0.3,
        xref="paper", yref="paper",
        fillcolor="#9b59b6", opacity=0.8,
        line=dict(color="#9b59b6", width=1),
    ))

    # Coordinator label
    annotations.append(dict(
        x=0.5, y=0.2,
        xref="paper", yref="paper",
        text="<b>Split Learning<br>Coordinator</b>",
        showarrow=False,
        font=dict(size=13, color="white"),
        align="center",
    ))

    # PSI annotation box
    annotations.append(dict(
        x=0.5, y=0.5,
        xref="paper", yref="paper",
        text="<i>Private Set Intersection (PSI)<br>Allinea Patient IDs</i>",
        showarrow=False,
        font=dict(size=11, color="#333"),
        align="center",
        bgcolor="wheat", opacity=0.7,
        bordercolor="#ddd", borderwidth=1, borderpad=6,
    ))

    fig = go.Figure()
    fig.update_layout(
        title=dict(
            text="<b>Architettura Vertical FL / Split Learning</b>",
            font=dict(size=16),
            x=0.5,
        ),
        shapes=shapes,
        annotations=annotations,
        xaxis=dict(visible=False, range=[0, 1]),
        yaxis=dict(visible=False, range=[0, 1]),
        width=900, height=500,
        margin=dict(l=20, r=20, t=60, b=20),
        plot_bgcolor="white",
    )

    return fig


def plot_hierarchy_tree():
    """Plot EHDS hierarchy as a tree."""
    G = nx.DiGraph()

    countries = ["Germany", "France", "Italy"]
    regions = {
        "Germany": ["Bavaria", "Berlin"],
        "France": ["Île-de-France", "PACA"],
        "Italy": ["Lombardia", "Lazio"]
    }
    all_regions = [r for rs in regions.values() for r in rs]

    # Add nodes with subset attribute for multipartite layout
    G.add_node("EU", subset=0)

    for country in countries:
        G.add_node(country, subset=1)
        G.add_edge("EU", country)
        for region in regions[country]:
            G.add_node(region, subset=2)
            G.add_edge(country, region)
            for i in range(2):
                hospital = f"H-{region[:3]}{i+1}"
                G.add_node(hospital, subset=3)
                G.add_edge(region, hospital)

    pos = nx.multipartite_layout(G, subset_key='subset', align='horizontal')

    # Swap axes: (x, y) -> (y, -x) for top-down orientation
    pos = {k: (v[1], -v[0]) for k, v in pos.items()}

    # Build colour map per node
    color_map = {}
    for node in G.nodes():
        if node == "EU":
            color_map[node] = '#003399'
        elif node in countries:
            color_map[node] = '#2ecc71'
        elif node in all_regions:
            color_map[node] = '#f39c12'
        else:
            color_map[node] = '#3498db'

    # --- Edge traces (lines with arrows via annotations) ---
    edge_x, edge_y = [], []
    edge_annotations = []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        # Arrow annotation pointing from parent toward child
        edge_annotations.append(dict(
            ax=x0, ay=y0,
            x=x1, y=y1,
            xref="x", yref="y",
            axref="x", ayref="y",
            showarrow=True,
            arrowhead=2, arrowsize=1.2, arrowwidth=1.5,
            arrowcolor="gray",
            text="",
        ))

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(width=1.5, color="gray"),
        hoverinfo="none",
        showlegend=False,
    )

    # --- Node trace ---
    node_x = [pos[n][0] for n in G.nodes()]
    node_y = [pos[n][1] for n in G.nodes()]
    node_colors = [color_map[n] for n in G.nodes()]
    node_labels = list(G.nodes())

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        marker=dict(size=30, color=node_colors, line=dict(width=1, color="white")),
        text=node_labels,
        textposition="middle center",
        textfont=dict(size=9, color="white", family="Arial Black"),
        hoverinfo="text",
        hovertext=node_labels,
        showlegend=False,
    )

    # --- Legend traces (invisible scatter points for colour legend) ---
    legend_items = [
        ("Livello EU", '#003399'),
        ("Livello Nazionale", '#2ecc71'),
        ("Livello Regionale", '#f39c12'),
        ("Livello Ospedale", '#3498db'),
    ]
    legend_traces = []
    for label, color in legend_items:
        legend_traces.append(go.Scatter(
            x=[None], y=[None],
            mode="markers",
            marker=dict(size=12, color=color),
            name=label,
            showlegend=True,
        ))

    fig = go.Figure(data=[edge_trace, node_trace] + legend_traces)
    fig.update_layout(
        title=dict(
            text="<b>Topologia EHDS Hierarchical FL</b>",
            font=dict(size=16),
            x=0.5,
        ),
        annotations=edge_annotations,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        width=1000, height=650,
        margin=dict(l=20, r=20, t=60, b=20),
        plot_bgcolor="white",
        legend=dict(
            x=0.95, y=0.98,
            xanchor="right", yanchor="top",
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="#ccc", borderwidth=1,
            font=dict(size=11),
        ),
    )

    return fig


# =============================================================================
# CONFIGURATION PANEL (Complete from v3)
# =============================================================================

def _load_yaml_defaults() -> Dict:
    """Load suggested defaults from config.yaml."""
    try:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from config.config_loader import get_dashboard_defaults
        return get_dashboard_defaults()
    except (ImportError, Exception):
        return {}

def _cfg_default(defaults: Dict, key: str, fallback, min_val=None, max_val=None, options=None):
    """Get default value from YAML config with clamping for widget constraints."""
    val = defaults.get(key, fallback)
    try:
        if options is not None:
            val = min(options, key=lambda x: abs(x - val))
        if min_val is not None:
            val = max(min_val, val)
        if max_val is not None:
            val = min(max_val, val)
    except (TypeError, ValueError):
        val = fallback
    return val


def create_config_panel() -> Dict:
    """Create comprehensive configuration panel with explanations."""
    _yaml_defaults = _load_yaml_defaults()

    def d(key, fallback, min_val=None, max_val=None, options=None):
        return _cfg_default(_yaml_defaults, key, fallback, min_val, max_val, options)

    st.sidebar.markdown("# ⚙️ Configurazione FL")

    # === NODES ===
    with st.sidebar.expander("🖥️ NODI", expanded=True):
        st.markdown("""
        <div class="info-box">
        <strong>Cosa sono i nodi?</strong><br>
        Ogni nodo rappresenta un ospedale o istituzione sanitaria nel network federato.
        I dati rimangono locali su ogni nodo - solo i gradienti vengono condivisi.
        </div>
        """, unsafe_allow_html=True)

        num_nodes = st.slider(
            "Numero di Nodi",
            min_value=2, max_value=15, value=d("num_nodes", 5, 2, 15),
            help="Numero di ospedali/istituzioni nel network FL"
        )

        total_samples = st.number_input(
            "Campioni Totali",
            min_value=500, max_value=10000, value=d("total_samples", 2000, 500, 10000), step=100,
            help="Numero totale di record paziente distribuiti tra i nodi"
        )

    # === ALGORITHM ===
    with st.sidebar.expander("🧮 ALGORITMO FL", expanded=True):
        st.markdown("""
        <div class="info-box">
        <strong>Algoritmo FL</strong><br>
        Determina come i modelli locali vengono aggregati.
        Algoritmi diversi hanno performance diverse con dati non-IID.
        </div>
        """, unsafe_allow_html=True)

        algorithm = st.selectbox(
            "Algoritmo",
            options=list(ALGORITHMS.keys()),
            format_func=lambda x: f"{x} - {ALGORITHMS[x]['name']}",
            help="Seleziona l'algoritmo di aggregazione federata"
        )

        st.markdown(f"**📖 {ALGORITHMS[algorithm]['name']}**")
        st.markdown(f"*{ALGORITHMS[algorithm]['description'][:100]}...*")
        st.markdown(f"Complessità: {ALGORITHMS[algorithm]['complexity']}")

        if algorithm == 'FedProx':
            fedprox_mu = st.slider(
                "μ (Proximal)",
                min_value=0.001, max_value=1.0, value=d("fedprox_mu", 0.1, 0.001, 1.0), step=0.01,
                help="Coefficiente prossimale. Più alto = più regolarizzazione verso il modello globale"
            )
        else:
            fedprox_mu = d("fedprox_mu", 0.1)

        if algorithm in ['FedAdam', 'FedYogi', 'FedAdagrad']:
            col1, col2 = st.columns(2)
            with col1:
                server_lr = st.number_input(
                    "Server LR", 0.01, 1.0, d("server_lr", 0.1, 0.01, 1.0),
                    help="Learning rate lato server"
                )
                beta1 = st.number_input(
                    "β1", 0.0, 1.0, d("beta1", 0.9, 0.0, 1.0),
                    help="Decadimento primo momento"
                )
            with col2:
                beta2 = st.number_input(
                    "β2", 0.0, 1.0, d("beta2", 0.99, 0.0, 1.0),
                    help="Decadimento secondo momento"
                )
                tau = st.number_input(
                    "τ", 1e-8, 1e-1, d("tau", 1e-3, 1e-8, 1e-1), format="%.0e",
                    help="Parametro di adattività"
                )
        else:
            server_lr = d("server_lr", 0.1)
            beta1 = d("beta1", 0.9)
            beta2 = d("beta2", 0.99)
            tau = d("tau", 1e-3)

    # === MODEL ===
    with st.sidebar.expander("🧠 MODELLO", expanded=True):
        st.markdown("""
        <div class="info-box">
        <strong>Architettura del Modello</strong><br>
        Il tipo di rete neurale da addestrare.
        La scelta dipende dal tipo di dati (tabulari, immagini, sequenze).
        </div>
        """, unsafe_allow_html=True)

        model = st.selectbox(
            "Architettura",
            options=list(MODELS.keys()),
            format_func=lambda x: f"{x} ({MODELS[x]['type']})",
            help="Seleziona l'architettura del modello"
        )

        st.markdown(f"**Input:** {MODELS[model]['input_type']}")
        st.markdown(f"**Parametri:** {MODELS[model]['params_count']}")
        st.markdown(f"**Use case:** {MODELS[model]['use_case']}")

    # === TRAINING ===
    with st.sidebar.expander("🔄 TRAINING", expanded=True):
        st.markdown("""
        <div class="info-box">
        <strong>Parametri di Training</strong><br>
        • <strong>Rounds</strong>: Iterazioni globali server-client<br>
        • <strong>Local Epochs</strong>: Epoche locali per round<br>
        • <strong>Learning Rate</strong>: Velocità di apprendimento
        </div>
        """, unsafe_allow_html=True)

        num_rounds = st.slider(
            "Training Rounds",
            min_value=10, max_value=200, value=d("num_rounds", 50, 10, 200),
            help="Numero di round di comunicazione server-client"
        )

        local_epochs = st.slider(
            "Local Epochs",
            min_value=1, max_value=10, value=d("local_epochs", 3, 1, 10),
            help="Epoche di training locale per round (E). Più epoche = meno comunicazione ma più drift"
        )

        _lr_options = [0.001, 0.01, 0.05, 0.1, 0.2, 0.5]
        learning_rate = st.select_slider(
            "Learning Rate (η)",
            options=_lr_options,
            value=d("learning_rate", 0.1, options=_lr_options),
            help="Tasso di apprendimento per SGD locale"
        )

    # === HETEROGENEITY ===
    with st.sidebar.expander("📊 ETEROGENEITÀ DATI", expanded=False):
        st.markdown("""
        <div class="warning-box">
        <strong>Non-IID Data</strong><br>
        In EHDS, gli ospedali hanno pazienti con caratteristiche diverse.
        L'eterogeneità dei dati è la sfida principale del FL.
        </div>
        """, unsafe_allow_html=True)

        heterogeneity_type = st.selectbox(
            "Tipo di Eterogeneità",
            options=list(HETEROGENEITY_TYPES.keys()),
            format_func=lambda x: HETEROGENEITY_TYPES[x]['name'],
            help="Come i dati sono distribuiti tra i nodi"
        )

        st.markdown(f"*{HETEROGENEITY_TYPES[heterogeneity_type]['description'][:150]}...*")

        label_skew_alpha = st.slider(
            "α Dirichlet (Label Skew)",
            min_value=0.1, max_value=10.0, value=d("label_skew_alpha", 0.5, 0.1, 10.0), step=0.1,
            help="α piccolo = non-IID estremo (ogni nodo ha principalmente una classe). α grande = IID"
        )

        feature_skew_strength = st.slider(
            "Feature Skew Strength",
            min_value=0.0, max_value=2.0, value=0.5, step=0.1,
            help="Quanto le distribuzioni delle feature differiscono tra nodi"
        )

    # === PARTICIPATION ===
    with st.sidebar.expander("👥 PARTECIPAZIONE", expanded=False):
        st.markdown("""
        <div class="info-box">
        <strong>Partecipazione Dinamica</strong><br>
        Non tutti i nodi partecipano ad ogni round.
        Cause: manutenzione, problemi di rete, fusi orari.
        </div>
        """, unsafe_allow_html=True)

        participation_mode = st.selectbox(
            "Modalità Partecipazione",
            options=list(PARTICIPATION_MODES.keys()),
            format_func=lambda x: PARTICIPATION_MODES[x]['name'],
            help="Come viene determinata la partecipazione dei nodi"
        )

        st.markdown(f"*{PARTICIPATION_MODES[participation_mode]['description']}*")

        participation_rate = st.slider(
            "Tasso Base di Partecipazione",
            min_value=0.5, max_value=1.0, value=0.85, step=0.05,
            help="Probabilità base che un nodo partecipi ad un round"
        )

    # === PRIVACY ===
    with st.sidebar.expander("🔒 DIFFERENTIAL PRIVACY", expanded=False):
        st.markdown("""
        <div class="success-box">
        <strong>Privacy Differenziale</strong><br>
        Aggiunge rumore calibrato per proteggere i dati individuali.
        Richiesto per compliance GDPR/EHDS.
        </div>
        """, unsafe_allow_html=True)

        use_dp = st.checkbox(
            "Abilita DP",
            value=d("use_dp", True),
            help="Abilita protezione Differential Privacy"
        )

        if use_dp:
            col1, col2 = st.columns(2)

            with col1:
                epsilon = st.number_input(
                    "Budget ε",
                    min_value=0.1, max_value=100.0, value=d("epsilon", 10.0, 0.1, 100.0), step=0.5,
                    help="Privacy budget. Più basso = privacy più forte ma meno accuratezza"
                )

                st.markdown("""
                **Guida ε:**
                - 0.1-1: Privacy molto forte
                - 1-10: Privacy forte (ricerca clinica)
                - 10-50: Privacy moderata
                """)

            with col2:
                _delta_options = [1e-3, 1e-4, 1e-5, 1e-6]
                delta = st.select_slider(
                    "δ (Failure Prob)",
                    options=_delta_options,
                    value=d("delta", 1e-5, options=_delta_options),
                    help="Probabilità di fallimento privacy"
                )

                clip_norm = st.slider(
                    "Clip Norm C",
                    min_value=0.1, max_value=5.0, value=d("clip_norm", 1.0, 0.1, 5.0), step=0.1,
                    help="Norma massima del gradiente (sensibilità)"
                )
        else:
            epsilon = d("epsilon", 10.0)
            delta = d("delta", 1e-5)
            clip_norm = d("clip_norm", 1.0)

    # === SEED ===
    with st.sidebar.expander("🎲 RIPRODUCIBILITÀ", expanded=False):
        random_seed = st.number_input(
            "Random Seed",
            min_value=0, max_value=9999, value=d("random_seed", 42, 0, 9999),
            help="Seed per riproducibilità degli esperimenti"
        )

    # === TRAINING MODE ===
    training_mode = "simulation"
    selected_dataset = None
    selected_tabular_dataset = None
    img_size = 64

    if HAS_REAL_TRAINING:
        with st.sidebar.expander("🔬 MODALITA' TRAINING", expanded=False):
            st.markdown("""
            <div class="success-box">
            <strong>Training Reale</strong><br>
            Passa dal simulatore NumPy al training PyTorch reale
            con reti neurali CNN su dataset clinici.
            </div>
            """, unsafe_allow_html=True)

            training_mode = st.radio(
                "Modalita'",
                options=["simulation", "real_tabular", "real_imaging"],
                format_func=lambda x: {
                    "simulation": "Simulazione (NumPy)",
                    "real_tabular": "Reale - Dati Tabulari (PyTorch)",
                    "real_imaging": "Reale - Imaging Clinico (PyTorch CNN)",
                }[x],
                help="Simulazione usa NumPy; Reale usa PyTorch con reti neurali",
                key="training_mode_radio"
            )

            selected_tabular_dataset = None
            if training_mode == "real_tabular":
                # Tabular dataset selection
                tabular_choices = ["Sintetico (Healthcare)"]
                tabular_map = {"Sintetico (Healthcare)": "synthetic"}
                try:
                    _diab_path = Path(__file__).parent.parent / "data" / "diabetes" / "diabetic_data.csv"
                    if _diab_path.exists():
                        tabular_choices.append("Diabetes 130-US (101K encounters)")
                        tabular_map["Diabetes 130-US (101K encounters)"] = "diabetes"
                except Exception:
                    pass
                try:
                    _heart_path = Path(__file__).parent.parent / "data" / "heart_disease"
                    if _heart_path.exists():
                        tabular_choices.append("Heart Disease UCI (920 pazienti)")
                        tabular_map["Heart Disease UCI (920 pazienti)"] = "heart_disease"
                except Exception:
                    pass
                if len(tabular_choices) > 1:
                    _sel_tab = st.selectbox(
                        "Dataset Tabular",
                        options=tabular_choices,
                        help="Seleziona il dataset tabulare clinico"
                    )
                    selected_tabular_dataset = tabular_map.get(_sel_tab, "synthetic")

            if training_mode == "real_imaging":
                datasets = discover_datasets()
                if datasets:
                    ds_options = {d["name"]: d for d in datasets}
                    selected_ds_name = st.selectbox(
                        "Dataset Clinico",
                        options=list(ds_options.keys()),
                        help="Seleziona il dataset di imaging clinico"
                    )
                    selected_dataset = ds_options[selected_ds_name]
                    st.caption(
                        f"Classi: {selected_dataset['num_classes']} | "
                        f"Immagini: {selected_dataset['total_images']:,}"
                    )
                else:
                    st.warning("Nessun dataset trovato in data/")

                _img_options = [32, 64, 128, 224]
                img_size = st.select_slider(
                    "Dimensione Immagine",
                    options=_img_options,
                    value=d("img_size", 64, options=_img_options),
                    help="Immagini ridimensionate a NxN pixel"
                )

            if training_mode.startswith("real"):
                available, msg = check_pytorch_available()
                if available:
                    st.success(f"PyTorch: {msg}")
                else:
                    st.error(f"PyTorch non disponibile: {msg}")

    # Dataset parameters reference table
    try:
        from config.config_loader import get_dataset_parameters as _get_ds_params
        _all_ds_params = _get_ds_params()
        if _all_ds_params:
            with st.sidebar.expander("Parametri Suggeriti per Dataset", expanded=False):
                import pandas as pd
                _rows = []
                for _name, _p in _all_ds_params.items():
                    _rows.append({
                        "Dataset": _name,
                        "LR": _p.get("learning_rate", "-"),
                        "Batch": _p.get("batch_size", "-"),
                        "Rounds": _p.get("num_rounds", "-"),
                        "Alpha": _p.get("alpha", "-"),
                        "ImgSize": _p.get("img_size") or "-",
                        "Algoritmi": ", ".join(_p.get("recommended_algorithms", [])[:2]),
                    })
                st.dataframe(pd.DataFrame(_rows), use_container_width=True, hide_index=True)
                st.caption("Parametri suggeriti automaticamente alla selezione del dataset")
    except Exception:
        pass

    # Governance status indicator in sidebar
    _gov_bridge = st.session_state.get("gov_bridge")
    if _gov_bridge is not None:
        _gov_cfg = st.session_state.get("gov_config", {})
        _gov_countries = _gov_cfg.get("countries", [])
        _gov_purpose = _gov_cfg.get("purpose", "N/A")
        st.sidebar.markdown("---")
        st.sidebar.markdown("**EHDS Governance**")
        st.sidebar.caption(
            f"Attiva | {len(_gov_countries)} paesi | "
            f"eps={_gov_cfg.get('global_epsilon', 0):.1f} | "
            f"{_gov_purpose}"
        )

    return {
        "num_nodes": num_nodes,
        "total_samples": total_samples,
        "algorithm": algorithm,
        "fedprox_mu": fedprox_mu,
        "server_lr": server_lr,
        "beta1": beta1,
        "beta2": beta2,
        "tau": tau,
        "model": model,
        "num_rounds": num_rounds,
        "local_epochs": local_epochs,
        "learning_rate": learning_rate,
        "heterogeneity_type": heterogeneity_type,
        "label_skew_alpha": label_skew_alpha,
        "feature_skew_strength": feature_skew_strength,
        "participation_mode": participation_mode,
        "participation_rate": participation_rate,
        "use_dp": use_dp,
        "epsilon": epsilon,
        "delta": delta,
        "clip_norm": clip_norm,
        "random_seed": random_seed,
        "training_mode": training_mode,
        "selected_dataset": selected_dataset,
        "selected_tabular_dataset": selected_tabular_dataset,
        "img_size": img_size,
    }

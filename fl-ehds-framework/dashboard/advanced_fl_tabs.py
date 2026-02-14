"""Advanced FL tab renderers: algorithms, models, vertical, byzantine, continual, multitask, hierarchical."""

import time

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from dashboard.constants import (
    ALGORITHMS,
    MODELS,
    BYZANTINE_METHODS,
    CONTINUAL_METHODS,
    MULTITASK_METHODS,
)
from dashboard.simulators import (
    VerticalFLSimulator,
    ByzantineSimulator,
    ContinualFLSimulator,
    MultiTaskFLSimulator,
    HierarchicalFLSimulator,
)
from dashboard.config_panel import plot_vertical_fl_architecture, plot_hierarchy_tree


# Local constants used by multi-task tab
EHDS_TASKS = {
    "diabetes_risk": {"name": "Predizione Rischio Diabete", "type": "binary"},
    "readmission_30d": {"name": "Readmissione a 30 Giorni", "type": "binary"},
    "los_prediction": {"name": "Durata Degenza (LOS)", "type": "regression"},
    "mortality_risk": {"name": "Rischio Mortalità", "type": "binary"},
    "sepsis_onset": {"name": "Rilevamento Sepsi", "type": "binary"},
}


def show_algorithm_help(algorithm: str):
    """Display detailed help for an algorithm."""
    if algorithm not in ALGORITHMS:
        return

    info = ALGORITHMS[algorithm]

    st.markdown(f"""
    <div class="algorithm-card">
        <h4>📊 {info['name']}</h4>
        <p><em>{info['paper']}</em></p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(info['description'])

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**✅ Vantaggi:**")
        for pro in info['pros']:
            st.markdown(f"- {pro}")

    with col2:
        st.markdown("**⚠️ Svantaggi:**")
        for con in info['cons']:
            st.markdown(f"- {con}")

    st.markdown(f"**🎯 Best for:** {info['best_for']}")
    st.markdown(f"**⚙️ Complessità:** {info['complexity']}")


def render_algorithms_tab():
    """Render algorithms comparison tab."""
    st.markdown("### 🧮 Confronto Algoritmi FL")

    st.markdown("""
    Questa sezione fornisce un confronto dettagliato di tutti gli algoritmi FL disponibili.
    Ogni algoritmo ha caratteristiche diverse per gestire l'eterogeneità dei dati.
    """)

    df_data = []
    for name, info in ALGORITHMS.items():
        df_data.append({
            "Algoritmo": name,
            "Nome Completo": info['name'],
            "Paper": info['paper'],
            "Best For": info['best_for'],
            "Complessità": info['complexity']
        })

    st.dataframe(pd.DataFrame(df_data), use_container_width=True)

    st.markdown("---")

    selected = st.selectbox(
        "Seleziona un algoritmo per dettagli",
        options=list(ALGORITHMS.keys())
    )

    show_algorithm_help(selected)


def render_models_tab():
    """Render models comparison tab."""
    st.markdown("### 🧠 Catalogo Modelli")

    st.markdown("""
    Tutti i modelli disponibili per il training federato.
    La scelta dipende dal tipo di dati e dalla complessità del task.
    """)

    model_types = list(set(MODELS[m]['type'] for m in MODELS))
    selected_type = st.multiselect(
        "Filtra per tipo",
        options=model_types,
        default=model_types
    )

    for model_name, info in MODELS.items():
        if info['type'] in selected_type:
            with st.expander(f"**{model_name}** ({info['type']}) - {info['params_count']}"):
                st.markdown(info['description'])
                st.markdown(f"**Input:** {info['input_type']}")
                st.markdown(f"**Use Case:** {info['use_case']}")
                st.markdown(f"**Complessità:** {info['complexity']}")


def render_vertical_fl_tab():
    """Render Vertical FL / Split Learning tab."""
    st.markdown("### 📊 Vertical Federated Learning / Split Learning")

    st.markdown("""
    <div class="info-box">
    <strong>Cos'è il Vertical FL?</strong><br>
    Nel Vertical FL, diverse parti detengono <em>feature diverse</em> per gli <em>stessi pazienti</em>.
    Questo è comune in EHDS dove l'Ospedale A ha dati demografici, l'Ospedale B ha risultati di laboratorio,
    e l'Ospedale C ha dati sullo stile di vita.
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("#### Configurazione")
        num_parties = st.slider("Numero di Parti", 2, 5, 3, key="vfl_parties")
        num_samples = st.slider("Pazienti per Parte", 500, 5000, 1000, key="vfl_samples")
        use_dp = st.checkbox("Aggiungi Rumore DP agli Embeddings", value=True, key="vfl_dp")

        if st.button("🔒 Esegui PSI + Train", key="vfl_run"):
            simulator = VerticalFLSimulator(num_parties, num_samples)

            st.markdown("##### Private Set Intersection")
            common, sizes = simulator.run_psi()

            psi_df = pd.DataFrame({
                "Parte": [f"Parte {i+1}" for i in range(num_parties)],
                "Pazienti": sizes
            })
            st.dataframe(psi_df, use_container_width=True)
            st.success(f"✅ Pazienti comuni dopo PSI: **{common}**")

            st.markdown("##### Training SplitNN")
            progress = st.progress(0)
            history = simulator.train_splitnn(10)

            for i, h in enumerate(history):
                progress.progress((i + 1) / 10)
                time.sleep(0.1)

            final = history[-1]
            col_a, col_b = st.columns(2)
            col_a.metric("Accuracy Finale", f"{final['accuracy']:.2%}")
            col_b.metric("Loss Finale", f"{final['loss']:.4f}")

            # Plot training curve
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=[h['epoch'] for h in history],
                y=[h['accuracy'] for h in history],
                mode='lines+markers',
                name='Accuracy',
                line=dict(color='blue', width=2)
            ))
            fig.update_layout(
                title='SplitNN Training Convergence',
                xaxis_title='Epoch',
                yaxis_title='Accuracy',
                template='plotly_white'
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Architettura")
        fig = plot_vertical_fl_architecture(num_parties)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        #### Caso d'Uso EHDS
        | Parte | Tipo Dati | Features |
        |-------|-----------|----------|
        | Ospedale A | Demografici | Età, Genere, Località |
        | Ospedale B | Lab Results | Esami sangue, Vitali |
        | Ospedale C | Stile di Vita | Dieta, Esercizio, Sonno |

        **Componenti Chiave:**
        1. **PSI (Private Set Intersection)**: Trova in modo sicuro gli ID pazienti comuni
        2. **SplitNN**: Ogni parte processa le sue feature, solo gli embeddings sono condivisi
        3. **Rumore DP**: Aggiunto agli embeddings per privacy aggiuntiva
        """)


def render_byzantine_tab():
    """Render Byzantine Resilience tab."""
    st.markdown("### 🛡️ Byzantine Resilience")

    st.markdown("""
    <div class="danger-box">
    <strong>Modello di Minaccia Bizantino</strong><br>
    In EHDS cross-border, alcuni partecipanti potrebbero essere compromessi o malevoli.
    L'aggregazione Byzantine-resilient protegge il modello globale da attacchi di poisoning.
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("#### Configurazione Attacco")

        num_clients = st.slider("Client Totali", 5, 20, 10, key="byz_clients")
        num_byzantine = st.slider("Client Bizantini", 0, num_clients // 2, 2, key="byz_num")

        attack_type = st.selectbox(
            "Tipo di Attacco",
            options=list(ByzantineSimulator.ATTACK_TYPES.keys()),
            format_func=lambda x: f"{x}: {ByzantineSimulator.ATTACK_TYPES[x][:40]}...",
            key="byz_attack"
        )

        st.markdown(f"*{ByzantineSimulator.ATTACK_TYPES[attack_type]}*")

        st.markdown("#### Metodo di Difesa")
        defense = st.selectbox(
            "Metodo di Aggregazione",
            options=list(BYZANTINE_METHODS.keys()),
            format_func=lambda x: BYZANTINE_METHODS[x]['name'],
            key="byz_defense"
        )

        method_info = BYZANTINE_METHODS[defense]
        st.markdown(f"**{method_info['name']}**")
        st.markdown(f"*{method_info['description']}*")
        st.markdown(f"Tolleranza: `{method_info['tolerance']}`")

        if st.button("⚔️ Simula Attacco", key="byz_run"):
            simulator = ByzantineSimulator(num_clients, num_byzantine)
            history = simulator.simulate_attack(attack_type, defense, 20)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(range(20)), y=history["no_defense"],
                mode='lines', name='Senza Difesa (FedAvg)',
                line=dict(color='red', width=2, dash='dash')
            ))
            fig.add_trace(go.Scatter(
                x=list(range(20)), y=history["with_defense"],
                mode='lines', name=f'Con {method_info["name"]}',
                line=dict(color='green', width=2),
                fill='tozeroy', fillcolor='rgba(0,128,0,0.15)'
            ))
            fig.update_layout(
                title=f'Attacco Bizantino ({attack_type}) vs Difesa ({defense})',
                xaxis_title='Round',
                yaxis_title='Accuracy',
                yaxis_range=[0, 1],
                template='plotly_white'
            )
            st.plotly_chart(fig, use_container_width=True)

            final_no_defense = history["no_defense"][-1]
            final_with_defense = history["with_defense"][-1]
            improvement = final_with_defense - final_no_defense

            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Senza Difesa", f"{final_no_defense:.2%}")
            col_b.metric("Con Difesa", f"{final_with_defense:.2%}")
            col_c.metric("Miglioramento", f"+{improvement:.2%}")

    with col2:
        st.markdown("#### Confronto Metodi di Difesa")

        defense_df = pd.DataFrame([
            {
                "Metodo": info["name"],
                "Tolleranza": info["tolerance"],
                "Paper": info["paper"],
                "Descrizione": info["description"][:60] + "..."
            }
            for info in BYZANTINE_METHODS.values()
        ])
        st.dataframe(defense_df, use_container_width=True)

        st.markdown("""
        #### Tipi di Attacco
        | Attacco | Descrizione | Impatto |
        |---------|-------------|---------|
        | Label Flip | Inverte le label di training | Alto - degrada l'accuracy |
        | Scale | Moltiplica i gradienti | Medio - overshoots gli update |
        | Noise | Aggiunge rumore random | Medio - aggiunge varianza |
        | Sign Flip | Nega i gradienti | Alto - inverte l'apprendimento |
        | Lie Attack | Crafted per evadere detection | Molto Alto - targeted |
        """)


def render_continual_tab():
    """Render Continual Learning tab."""
    st.markdown("### 🔄 Continual Federated Learning")

    st.markdown("""
    <div class="warning-box">
    <strong>Concept Drift in Healthcare</strong><br>
    La conoscenza medica evolve: nuovi trattamenti, linee guida cambiate, shift di popolazione.
    Il Continual FL deve imparare nuovi pattern mantenendo la conoscenza importante.
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("#### Configurazione")

        num_tasks = st.slider("Numero di Task/Periodi", 2, 6, 4, key="cont_tasks")
        method = st.selectbox(
            "Metodo Continual Learning",
            options=list(CONTINUAL_METHODS.keys()),
            format_func=lambda x: CONTINUAL_METHODS[x]['name'],
            key="cont_method"
        )

        method_info = CONTINUAL_METHODS[method]
        st.markdown(f"**{method_info['name']}**")
        st.markdown(f"*{method_info['description']}*")

        st.markdown("**Pro:**")
        for pro in method_info['pros']:
            st.markdown(f"- ✅ {pro}")

        st.markdown("**Contro:**")
        for con in method_info['cons']:
            st.markdown(f"- ⚠️ {con}")

        if st.button("📚 Train Across Tasks", key="cont_run"):
            simulator = ContinualFLSimulator(num_tasks)

            history_method = simulator.simulate_training(method, 15)
            history_baseline = simulator.simulate_training("none", 15)

            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=(f"Con {method_info['name']}", "Senza Continual Learning (Catastrophic Forgetting)")
            )
            colors = [f'hsl({int(h)},70%,50%)' for h in np.linspace(0, 300, num_tasks)]

            for i in range(num_tasks):
                fig.add_trace(go.Scatter(
                    x=list(range(len(history_method["accuracy_per_task"][i]))),
                    y=history_method["accuracy_per_task"][i],
                    mode='lines', name=f'Task {i+1}',
                    line=dict(color=colors[i], width=2),
                    legendgroup=f'task{i+1}', showlegend=True
                ), row=1, col=1)
                fig.add_vline(x=i*15, line_dash='dash', line_color=colors[i], opacity=0.5, row=1, col=1)

            for i in range(num_tasks):
                fig.add_trace(go.Scatter(
                    x=list(range(len(history_baseline["accuracy_per_task"][i]))),
                    y=history_baseline["accuracy_per_task"][i],
                    mode='lines', name=f'Task {i+1}',
                    line=dict(color=colors[i], width=2),
                    legendgroup=f'task{i+1}', showlegend=False
                ), row=1, col=2)
                fig.add_vline(x=i*15, line_dash='dash', line_color=colors[i], opacity=0.5, row=1, col=2)

            fig.update_xaxes(title_text='Training Steps', row=1, col=1)
            fig.update_xaxes(title_text='Training Steps', row=1, col=2)
            fig.update_yaxes(title_text='Accuracy', range=[0, 1], row=1, col=1)
            fig.update_yaxes(title_text='Accuracy', range=[0, 1], row=1, col=2)
            fig.update_layout(template='plotly_white')
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("#### Analisi Forgetting")
            final_accs_method = [history_method["accuracy_per_task"][i][-1] for i in range(num_tasks)]
            final_accs_baseline = [history_baseline["accuracy_per_task"][i][-1] for i in range(num_tasks)]

            forget_df = pd.DataFrame({
                "Task": [f"Task {i+1}" for i in range(num_tasks)],
                f"Con {method.upper()}": [f"{a:.2%}" for a in final_accs_method],
                "Baseline": [f"{a:.2%}" for a in final_accs_baseline]
            })
            st.dataframe(forget_df, use_container_width=True)

    with col2:
        st.markdown("#### Rilevamento Concept Drift")

        simulator = ContinualFLSimulator()
        drift_points, performance = simulator.detect_drift()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(100)), y=performance,
            mode='lines', name='Performance',
            line=dict(color='blue', width=2),
            fill='tozeroy', fillcolor='rgba(0,0,255,0.15)'
        ))
        for dp in drift_points:
            fig.add_vline(x=dp, line_dash='dash', line_color='red', opacity=0.7)
            fig.add_annotation(x=dp, y=0.75, text=f"Drift @{dp}",
                               font=dict(size=9, color='red'), showarrow=False)
        fig.update_layout(
            title='Drift Detection: Monitoraggio Performance',
            xaxis_title='Time Step',
            yaxis_title='Performance Modello',
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        #### Scenario EHDS: Evoluzione Conoscenza Medica

        | Periodo | Evento | Impatto |
        |---------|--------|---------|
        | 2021-2022 | Protocolli COVID-19 | Nuovi pattern diagnostici |
        | 2023 | Linee guida diabete aggiornate | Soglie rischio cambiate |
        | 2024 | Nuovi farmaci approvati | Shift outcome trattamento |
        | 2025 | Invecchiamento popolazione | Cambio distribuzione demografica |

        **Metodi Drift Detection:**
        - **DDM (Drift Detection Method)**: Test statistico su tasso errore
        - **Performance-based**: Monitora accuracy su validation set
        - **Distribution-based**: Confronta distribuzioni feature
        """)


def render_multitask_tab():
    """Render Multi-Task FL tab."""
    st.markdown("### 🎯 Multi-Task Federated Learning")

    st.markdown("""
    <div class="info-box">
    <strong>Multi-Task Learning in EHDS</strong><br>
    Gli ospedali spesso necessitano di multipli modelli predittivi: rischio readmissione, durata degenza, mortalità.
    Il Multi-task FL li apprende insieme, condividendo conoscenza rispettando la località dei dati.
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("#### Selezione Task")

        selected_tasks = st.multiselect(
            "Task EHDS",
            options=list(EHDS_TASKS.keys()),
            default=["diabetes_risk", "readmission_30d", "los_prediction"],
            format_func=lambda x: EHDS_TASKS[x]["name"],
            key="mtl_tasks"
        )

        if len(selected_tasks) < 2:
            st.warning("Seleziona almeno 2 task")
            return

        num_clients = st.slider("Numero di Ospedali", 3, 12, 6, key="mtl_clients")

        method = st.selectbox(
            "Architettura MTL",
            options=list(MULTITASK_METHODS.keys()),
            format_func=lambda x: MULTITASK_METHODS[x]['name'],
            key="mtl_method"
        )

        method_info = MULTITASK_METHODS[method]
        st.markdown(f"**{method_info['name']}**")
        st.markdown(f"*{method_info['description']}*")
        st.markdown(f"Architettura: `{method_info['architecture']}`")

        if st.button("🏥 Train Multi-Task", key="mtl_run"):
            simulator = MultiTaskFLSimulator(num_clients, selected_tasks)

            st.markdown("##### Copertura Client-Task")
            coverage_data = []
            for c in range(num_clients):
                row = {"Ospedale": f"Ospedale {c+1}"}
                for i, task in enumerate(selected_tasks):
                    row[EHDS_TASKS[task]["name"][:15]] = "✓" if i in simulator.client_tasks[c] else "○"
                coverage_data.append(row)
            st.dataframe(pd.DataFrame(coverage_data), use_container_width=True)

            history = simulator.train(method, 30)

            fig = go.Figure()
            colors = px.colors.qualitative.Set2[:len(selected_tasks)]

            for i, task in enumerate(selected_tasks):
                fig.add_trace(go.Scatter(
                    x=list(range(len(history[task]))),
                    y=history[task],
                    mode='lines', name=EHDS_TASKS[task]["name"],
                    line=dict(color=colors[i], width=2)
                ))

            fig.update_layout(
                title=f'Multi-Task FL con {method_info["name"]}',
                xaxis_title='Round',
                yaxis_title='Accuracy',
                yaxis_range=[0.4, 1.0],
                template='plotly_white'
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("##### Performance Finale per Task")
            final_df = pd.DataFrame({
                "Task": [EHDS_TASKS[t]["name"] for t in selected_tasks],
                "Tipo": [EHDS_TASKS[t]["type"] for t in selected_tasks],
                "Accuracy": [f"{history[t][-1]:.2%}" for t in selected_tasks]
            })
            st.dataframe(final_df, use_container_width=True)

    with col2:
        st.markdown("#### Confronto Architetture")

        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=('<b>Hard Sharing</b>', '<b>Soft Sharing</b>', '<b>FedMTL</b>'),
            horizontal_spacing=0.08
        )

        # --- Hard Sharing (col 1) ---
        # Backbone box
        fig.add_shape(type='rect', x0=0.2, y0=0.1, x1=0.8, y1=0.5,
                      fillcolor='rgba(52,152,219,0.7)', line=dict(color='#3498db'),
                      xref='x', yref='y', row=1, col=1)
        fig.add_annotation(x=0.5, y=0.3, text='Backbone<br>Condiviso',
                           font=dict(size=10, color='white'), showarrow=False,
                           xref='x', yref='y', row=1, col=1)
        for i, c in enumerate(['#e74c3c', '#2ecc71', '#f39c12']):
            fig.add_shape(type='circle',
                          x0=0.25+i*0.25-0.08, y0=0.7-0.08,
                          x1=0.25+i*0.25+0.08, y1=0.7+0.08,
                          fillcolor=c, line=dict(color=c),
                          xref='x', yref='y', row=1, col=1)
            fig.add_annotation(x=0.25+i*0.25, y=0.88, text=f'T{i+1}',
                               font=dict(size=9), showarrow=False,
                               xref='x', yref='y', row=1, col=1)

        # --- Soft Sharing (col 2) ---
        for i, c in enumerate(['#3498db', '#e74c3c', '#2ecc71']):
            fig.add_shape(type='rect',
                          x0=0.1+i*0.3, y0=0.1, x1=0.3+i*0.3, y1=0.8,
                          fillcolor=f'rgba({int(c[1:3],16)},{int(c[3:5],16)},{int(c[5:7],16)},0.7)',
                          line=dict(color=c),
                          xref='x2', yref='y2', row=1, col=2)
            fig.add_annotation(x=0.2+i*0.3, y=0.45, text=f'Net {i+1}',
                               font=dict(size=9, color='white'), showarrow=False,
                               textangle=-90,
                               xref='x2', yref='y2', row=1, col=2)
        for i in range(2):
            fig.add_annotation(x=0.3+i*0.3, y=0.5, ax=0.4+i*0.3, ay=0.5,
                               xref='x2', yref='y2', axref='x2', ayref='y2',
                               showarrow=True, arrowhead=3, arrowcolor='gray',
                               arrowwidth=1.5)

        # --- FedMTL (col 3) ---
        fig.add_shape(type='rect', x0=0.3, y0=0.5, x1=0.7, y1=0.8,
                      fillcolor='rgba(155,89,182,0.7)', line=dict(color='#9b59b6'),
                      xref='x3', yref='y3', row=1, col=3)
        fig.add_annotation(x=0.5, y=0.65, text='Task-Aware<br>Aggregation',
                           font=dict(size=9, color='white'), showarrow=False,
                           xref='x3', yref='y3', row=1, col=3)
        for i, c in enumerate(['#e74c3c', '#2ecc71', '#f39c12']):
            fig.add_shape(type='circle',
                          x0=0.25+i*0.25-0.08, y0=0.25-0.08,
                          x1=0.25+i*0.25+0.08, y1=0.25+0.08,
                          fillcolor=c, line=dict(color=c),
                          xref='x3', yref='y3', row=1, col=3)
            fig.add_annotation(x=0.25+i*0.25, y=0.33, ax=0.4+i*0.05, ay=0.5,
                               xref='x3', yref='y3', axref='x3', ayref='y3',
                               showarrow=True, arrowhead=2, arrowcolor='gray')

        # Common layout for all subplots
        for axis_suffix in ['', '2', '3']:
            fig.update_layout(**{
                f'xaxis{axis_suffix}': dict(range=[0, 1], showgrid=False, zeroline=False, visible=False),
                f'yaxis{axis_suffix}': dict(range=[0, 1], showgrid=False, zeroline=False, visible=False,
                                            scaleanchor=f'x{axis_suffix}' if axis_suffix else 'x'),
            })
        fig.update_layout(height=350, template='plotly_white', showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        #### Quando Usare Ogni Approccio

        | Metodo | Ideale Per | Trade-off |
        |--------|------------|-----------|
        | Hard Sharing | Task correlati, compute limitato | Meno flessibilità task-specifica |
        | Soft Sharing | Complessità task diversa | Più parametri |
        | FedMTL | Copertura task parziale tra client | Overhead comunicazione |
        """)


def render_hierarchical_tab():
    """Render Hierarchical FL tab."""
    st.markdown("### 🏛️ Hierarchical Federated Learning")

    st.markdown("""
    <div class="success-box">
    <strong>Aggregazione Multi-Livello EHDS</strong><br>
    EHDS copre 27+ stati membri EU con autorità sanitarie regionali e locali.
    L'Hierarchical FL aggrega a più livelli: Ospedale → Regione → Paese → EU.
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("#### Configurazione Gerarchia")

        simulator = HierarchicalFLSimulator()
        counts = simulator.count_nodes()

        st.markdown("##### Topologia Corrente")
        count_df = pd.DataFrame({
            "Livello": ["🇪🇺 EU", "🏳️ Paesi", "📍 Regioni", "🏥 Ospedali"],
            "Conteggio": [counts["eu"], counts["countries"], counts["regions"], counts["hospitals"]]
        })
        st.dataframe(count_df, use_container_width=True)

        agg_strategy = st.selectbox(
            "Strategia Aggregazione",
            options=["Pesata per campioni", "Uniforme", "Pesata per performance"],
            key="hier_strategy"
        )

        sync_mode = st.radio(
            "Sincronizzazione",
            options=["Completamente Sincrono", "Semi-Async (regionale)", "Async"],
            key="hier_sync"
        )

        if st.button("🌍 Train Gerarchicamente", key="hier_run"):
            history = simulator.train(20)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(range(len(history["global"]))),
                y=history["global"],
                mode='lines', name='EU Globale',
                line=dict(color='black', width=3)
            ))
            colors_country = {'DE': '#000000', 'FR': '#0055A4', 'IT': '#009246'}
            for country, accs in history["per_country"].items():
                fig.add_trace(go.Scatter(
                    x=list(range(len(accs))),
                    y=accs,
                    mode='lines', name=country,
                    line=dict(color=colors_country[country], width=2, dash='dash'),
                    opacity=0.7
                ))
            fig.update_layout(
                title='Hierarchical FL: Convergenza per Livello',
                xaxis_title='Round',
                yaxis_title='Accuracy',
                yaxis_range=[0.4, 1.0],
                template='plotly_white'
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("##### Performance Finale per Paese")
            final_df = pd.DataFrame({
                "Paese": ["🇩🇪 Germania", "🇫🇷 Francia", "🇮🇹 Italia"],
                "Accuracy": [f"{history['per_country'][c][-1]:.2%}" for c in ['DE', 'FR', 'IT']]
            })
            st.dataframe(final_df, use_container_width=True)
            st.metric("Accuracy Globale EU", f"{history['global'][-1]:.2%}")

    with col2:
        st.markdown("#### Visualizzazione Topologia EHDS")
        fig = plot_hierarchy_tree()
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        #### Benefici dell'Hierarchical FL per EHDS

        1. **Comunicazione Ridotta**: Aggregazione regionale prima del sync cross-border
        2. **Compliance Legale**: Gli HDAB nazionali possono applicare regole specifiche del paese
        3. **Fault Tolerance**: Failure regionali non impattano altre regioni
        4. **Layer di Privacy**: Aggregazione aggiuntiva = più privacy

        #### Flusso di Aggregazione
        ```
        Update Ospedale → Server Regionale → HDAB Nazionale → Coordinatore EU
             ↓                   ↓                ↓              ↓
          DP Locale         Avg Regionale    Avg Nazionale   Modello Globale EU
        ```
        """)


__all__ = [
    "render_algorithms_tab",
    "render_models_tab",
    "render_vertical_fl_tab",
    "render_byzantine_tab",
    "render_continual_tab",
    "render_multitask_tab",
    "render_hierarchical_tab",
]

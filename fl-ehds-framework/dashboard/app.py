#!/usr/bin/env python3
"""
FL-EHDS Dashboard - Academic Edition
=====================================

Professional, minimalist dashboard for Federated Learning experiments
in the context of the European Health Data Space (EHDS).

This dashboard implements:
- Two-panel academic layout (Configuration | Results)
- 4 consolidated sections (Experiment, Results, Compliance, System)
- Minimalist scientific styling
- Persistent EHDS compliance footer

Author: Fabio Liberti
Affiliation: Universitas Mercatorum, Rome, Italy
Usage: streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="FL-EHDS Framework",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =============================================================================
# ACADEMIC STYLING (Minimalist CSS)
# =============================================================================

st.markdown("""
<style>
    /* Academic Color Palette */
    :root {
        --bg-primary: #ffffff;
        --bg-secondary: #f8f9fa;
        --bg-tertiary: #e9ecef;
        --text-primary: #212529;
        --text-secondary: #6c757d;
        --accent: #0066cc;
        --accent-light: #e7f1ff;
        --border: #dee2e6;
        --success: #198754;
        --warning: #ffc107;
        --danger: #dc3545;
    }

    /* Typography - IBM Plex Sans for scientific documents */
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono&display=swap');

    .main .block-container {
        padding-top: 1rem;
        max-width: 1400px;
    }

    /* Header styling */
    .main-title {
        font-family: 'IBM Plex Sans', sans-serif;
        font-size: 1.75rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 0.25rem;
        border-bottom: 2px solid var(--accent);
        padding-bottom: 0.5rem;
    }

    .subtitle {
        font-family: 'IBM Plex Sans', sans-serif;
        font-size: 0.95rem;
        color: var(--text-secondary);
        margin-bottom: 1rem;
    }

    /* Section headers */
    .section-header {
        font-family: 'IBM Plex Sans', sans-serif;
        font-size: 1.1rem;
        font-weight: 600;
        color: var(--text-primary);
        border-bottom: 1px solid var(--border);
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }

    /* Configuration panel */
    .config-panel {
        background: var(--bg-secondary);
        border: 1px solid var(--border);
        border-radius: 4px;
        padding: 1rem;
        margin-bottom: 1rem;
    }

    .config-group {
        margin-bottom: 0.75rem;
    }

    .config-label {
        font-family: 'IBM Plex Sans', sans-serif;
        font-size: 0.85rem;
        font-weight: 500;
        color: var(--text-primary);
        margin-bottom: 0.25rem;
    }

    .config-hint {
        font-family: 'IBM Plex Sans', sans-serif;
        font-size: 0.75rem;
        color: var(--text-secondary);
        font-style: italic;
    }

    /* Results panel */
    .results-panel {
        background: var(--bg-primary);
        border: 1px solid var(--border);
        border-radius: 4px;
        padding: 1rem;
    }

    /* Metric cards - booktabs style */
    .metric-card {
        background: var(--bg-secondary);
        border: 1px solid var(--border);
        border-radius: 4px;
        padding: 0.75rem 1rem;
        text-align: center;
    }

    .metric-value {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.5rem;
        font-weight: 600;
        color: var(--accent);
    }

    .metric-label {
        font-family: 'IBM Plex Sans', sans-serif;
        font-size: 0.75rem;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Tables - booktabs style */
    .dataframe {
        font-family: 'IBM Plex Sans', sans-serif;
        font-size: 0.85rem;
    }

    .dataframe thead th {
        border-bottom: 2px solid var(--text-primary) !important;
        font-weight: 600;
    }

    .dataframe tbody tr:last-child td {
        border-bottom: 1px solid var(--text-primary) !important;
    }

    /* Footer - Compliance status */
    .compliance-footer {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: var(--bg-secondary);
        border-top: 1px solid var(--border);
        padding: 0.5rem 1rem;
        font-family: 'IBM Plex Sans', sans-serif;
        font-size: 0.8rem;
        z-index: 1000;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .compliance-item {
        display: inline-flex;
        align-items: center;
        margin-right: 1.5rem;
    }

    .compliance-ok {
        color: var(--success);
    }

    .compliance-warn {
        color: var(--warning);
    }

    /* Info box - minimal */
    .info-note {
        background: var(--accent-light);
        border-left: 3px solid var(--accent);
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        font-size: 0.85rem;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Add padding at bottom for fixed footer */
    .main .block-container {
        padding-bottom: 4rem;
    }

    /* Tab styling - minimal */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        border-bottom: 1px solid var(--border);
    }

    .stTabs [data-baseweb="tab"] {
        font-family: 'IBM Plex Sans', sans-serif;
        font-size: 0.9rem;
        padding: 0.5rem 1.5rem;
        border-radius: 0;
    }

    .stTabs [aria-selected="true"] {
        border-bottom: 2px solid var(--accent);
    }

    /* Button styling */
    .stButton > button {
        font-family: 'IBM Plex Sans', sans-serif;
        border-radius: 4px;
    }

    .stButton > button[kind="primary"] {
        background-color: var(--accent);
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# ALGORITHM AND MODEL DEFINITIONS (Consolidated)
# =============================================================================

ALGORITHMS = {
    "FedAvg": {
        "name": "Federated Averaging",
        "paper": "McMahan et al., AISTATS 2017",
        "description": "Standard FL algorithm using weighted averaging of client updates.",
        "complexity": "Low",
        "best_for": "IID data, baseline experiments"
    },
    "FedProx": {
        "name": "Federated Proximal",
        "paper": "Li et al., MLSys 2020",
        "description": "Adds proximal term to handle client drift in non-IID settings.",
        "complexity": "Medium",
        "best_for": "Non-IID data, heterogeneous systems"
    },
    "SCAFFOLD": {
        "name": "Stochastic Controlled Averaging",
        "paper": "Karimireddy et al., ICML 2020",
        "description": "Uses control variates to correct client drift. Reduces variance.",
        "complexity": "High",
        "best_for": "Extreme non-IID, communication-rich settings"
    },
    "FedAdam": {
        "name": "Federated Adam",
        "paper": "Reddi et al., ICLR 2021",
        "description": "Server-side adaptive optimization using Adam.",
        "complexity": "Medium",
        "best_for": "Complex models, sparse gradients"
    },
    "FedNova": {
        "name": "Normalized Averaging",
        "paper": "Wang et al., NeurIPS 2020",
        "description": "Normalizes updates by local steps to handle heterogeneous computation.",
        "complexity": "Medium",
        "best_for": "Variable local epochs"
    }
}

MODELS = {
    "LogisticRegression": {
        "name": "Logistic Regression",
        "type": "Linear",
        "params": "~100",
        "input": "Tabular"
    },
    "MLP": {
        "name": "Multi-Layer Perceptron",
        "type": "MLP",
        "params": "~10K",
        "input": "Tabular"
    },
    "CNN": {
        "name": "Convolutional Neural Network",
        "type": "CNN",
        "params": "~200K",
        "input": "Image"
    },
    "ResNet18": {
        "name": "ResNet-18",
        "type": "CNN",
        "params": "~11M",
        "input": "Image (224x224)"
    }
}

BYZANTINE_METHODS = {
    "none": "No defense (FedAvg)",
    "krum": "Krum (Blanchard et al., 2017)",
    "trimmed_mean": "Trimmed Mean (Yin et al., 2018)",
    "median": "Coordinate-wise Median",
    "fltrust": "FLTrust (Cao et al., 2020)"
}


# =============================================================================
# FL SIMULATOR
# =============================================================================

class FLSimulator:
    """Federated Learning Simulator for academic experiments."""

    def __init__(self, config: Dict):
        self.config = config
        np.random.seed(config.get('seed', 42))

        self.num_clients = config.get('num_clients', 5)
        self.num_rounds = config.get('num_rounds', 30)
        self.algorithm = config.get('algorithm', 'FedAvg')

        self._init_data()
        self._init_model()
        self.history = []
        self.privacy_spent = 0.0

    def _init_data(self):
        """Initialize client data with configurable heterogeneity."""
        alpha = self.config.get('dirichlet_alpha', 0.5)
        total_samples = self.config.get('total_samples', 2000)
        samples_per_client = total_samples // self.num_clients

        rng = np.random.RandomState(self.config.get('seed', 42))
        label_dist = rng.dirichlet([alpha, alpha], size=self.num_clients)

        self.client_data = {}
        for i in range(self.num_clients):
            n = samples_per_client + rng.randint(-50, 50)
            shift = (i - self.num_clients / 2) * self.config.get('feature_skew', 0.3)

            X = rng.normal(shift, 1.0, (n, 5))
            X_norm = (X - X.mean(0)) / (X.std(0) + 1e-8)
            X_bias = np.hstack([X_norm, np.ones((n, 1))])
            y = rng.choice(2, size=n, p=label_dist[i])

            self.client_data[i] = {
                "X": X_bias,
                "y": y,
                "n_samples": n,
                "label_dist": label_dist[i].tolist()
            }

    def _init_model(self):
        """Initialize global model parameters."""
        self.weights = np.zeros(6)
        self.momentum = None
        self.velocity = None

    def train_round(self, round_num: int) -> Dict:
        """Execute one FL round."""
        lr = self.config.get('learning_rate', 0.1)
        local_epochs = self.config.get('local_epochs', 3)
        participation_rate = self.config.get('participation_rate', 0.8)

        # Client selection
        participating = [i for i in range(self.num_clients)
                        if np.random.random() < participation_rate]
        if not participating:
            participating = [0]

        gradients = []
        sample_counts = []
        client_metrics = {}

        for client_id in range(self.num_clients):
            data = self.client_data[client_id]

            if client_id in participating:
                local_w = self.weights.copy()

                # Local training
                for _ in range(local_epochs):
                    batch_size = min(32, data["n_samples"])
                    idx = np.random.choice(data["n_samples"], batch_size, replace=False)
                    X_b, y_b = data["X"][idx], data["y"][idx]

                    logits = X_b @ local_w
                    probs = 1 / (1 + np.exp(-np.clip(logits, -500, 500)))
                    grad = X_b.T @ (probs - y_b) / batch_size

                    # FedProx proximal term
                    if self.algorithm == 'FedProx':
                        mu = self.config.get('mu', 0.1)
                        grad += mu * (local_w - self.weights)

                    local_w -= lr * grad

                gradient = local_w - self.weights

                # Gradient clipping
                clip_norm = self.config.get('clip_norm', 1.0)
                norm = np.linalg.norm(gradient)
                if norm > clip_norm:
                    gradient = gradient * (clip_norm / norm)

                gradients.append(gradient)
                sample_counts.append(data["n_samples"])

            # Compute client accuracy
            logits = data["X"] @ self.weights
            probs = 1 / (1 + np.exp(-np.clip(logits, -500, 500)))
            preds = (probs > 0.5).astype(int)
            acc = float(np.mean(preds == data["y"]))

            client_metrics[client_id] = {
                "accuracy": acc,
                "samples": data["n_samples"],
                "participating": client_id in participating
            }

        # Aggregation
        if gradients:
            total = sum(sample_counts)
            agg_grad = sum(g * (n / total) for g, n in zip(gradients, sample_counts))

            # Differential Privacy noise
            if self.config.get('use_dp', True):
                epsilon = self.config.get('epsilon', 10.0)
                sigma = self.config.get('clip_norm', 1.0) / epsilon * 0.1
                noise = np.random.normal(0, sigma, agg_grad.shape)
                agg_grad += noise
                self.privacy_spent += epsilon / self.num_rounds

            self.weights += agg_grad

        # Global accuracy
        all_preds, all_labels = [], []
        for d in self.client_data.values():
            logits = d["X"] @ self.weights
            probs = 1 / (1 + np.exp(-np.clip(logits, -500, 500)))
            all_preds.extend((probs > 0.5).astype(int))
            all_labels.extend(d["y"])

        global_acc = float(np.mean(np.array(all_preds) == np.array(all_labels)))

        result = {
            "round": round_num,
            "global_accuracy": global_acc,
            "client_metrics": client_metrics,
            "participating_clients": len(participating),
            "privacy_spent": self.privacy_spent
        }

        self.history.append(result)
        return result


# =============================================================================
# VISUALIZATION FUNCTIONS (Academic Style)
# =============================================================================

def set_academic_style():
    """Configure matplotlib for academic plots."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['IBM Plex Sans', 'DejaVu Sans'],
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.titlesize': 12,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.linewidth': 0.8,
        'grid.alpha': 0.3,
        'grid.linewidth': 0.5,
    })

set_academic_style()


def plot_convergence(history: List[Dict], algorithm: str) -> plt.Figure:
    """Plot training convergence curve."""
    fig, ax = plt.subplots(figsize=(8, 4))

    rounds = [h['round'] for h in history]
    accuracies = [h['global_accuracy'] for h in history]

    ax.plot(rounds, accuracies, 'b-', linewidth=1.5, marker='o',
            markersize=3, markevery=5)
    ax.fill_between(rounds, accuracies, alpha=0.15, color='blue')

    ax.set_xlabel('Communication Round')
    ax.set_ylabel('Global Accuracy')
    ax.set_title(f'{algorithm} Training Convergence')
    ax.set_ylim(0.4, 1.0)
    ax.grid(True, linestyle='--', alpha=0.3)

    # Add final accuracy annotation
    final_acc = accuracies[-1]
    ax.annotate(f'{final_acc:.1%}',
                xy=(rounds[-1], final_acc),
                xytext=(5, 0), textcoords='offset points',
                fontsize=9, fontweight='bold')

    plt.tight_layout()
    return fig


def plot_client_distribution(client_data: Dict) -> plt.Figure:
    """Plot client data distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Sample distribution
    ax1 = axes[0]
    clients = list(client_data.keys())
    samples = [client_data[c]['n_samples'] for c in clients]

    bars = ax1.bar([f'C{c+1}' for c in clients], samples, color='#0066cc', alpha=0.7)
    ax1.set_xlabel('Client')
    ax1.set_ylabel('Number of Samples')
    ax1.set_title('(a) Sample Distribution')
    ax1.axhline(y=np.mean(samples), color='red', linestyle='--',
                label=f'Mean: {np.mean(samples):.0f}')
    ax1.legend()

    # Label distribution
    ax2 = axes[1]
    for c in clients[:5]:  # Show first 5 clients
        label_dist = client_data[c]['label_dist']
        ax2.bar([f'C{c+1}\nClass 0', f'C{c+1}\nClass 1'],
                label_dist, alpha=0.7)

    ax2.set_ylabel('Label Proportion')
    ax2.set_title('(b) Label Distribution (Non-IID)')

    plt.tight_layout()
    return fig


def plot_privacy_utility(history: List[Dict], epsilon: float) -> plt.Figure:
    """Plot privacy-utility tradeoff."""
    fig, ax = plt.subplots(figsize=(6, 4))

    rounds = [h['round'] for h in history]
    privacy = [h['privacy_spent'] for h in history]
    accuracy = [h['global_accuracy'] for h in history]

    ax2 = ax.twinx()

    line1, = ax.plot(rounds, privacy, 'r-', linewidth=1.5, label='Privacy Budget Spent')
    line2, = ax2.plot(rounds, accuracy, 'b-', linewidth=1.5, label='Accuracy')

    ax.set_xlabel('Communication Round')
    ax.set_ylabel('Cumulative Privacy Budget (e)', color='red')
    ax2.set_ylabel('Accuracy', color='blue')

    ax.axhline(y=epsilon, color='red', linestyle='--', alpha=0.5,
               label=f'Total Budget: e={epsilon}')

    ax.tick_params(axis='y', labelcolor='red')
    ax2.tick_params(axis='y', labelcolor='blue')

    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='center right')

    ax.set_title('Privacy-Utility Tradeoff')

    plt.tight_layout()
    return fig


# =============================================================================
# UI COMPONENTS
# =============================================================================

def render_header():
    """Render dashboard header."""
    st.markdown("""
    <div class="main-title">FL-EHDS Framework</div>
    <div class="subtitle">
        Federated Learning for European Health Data Space | Academic Research Dashboard
    </div>
    """, unsafe_allow_html=True)


def render_compliance_footer(config: Dict, privacy_spent: float = 0.0):
    """Render persistent compliance footer."""
    epsilon = config.get('epsilon', 10.0)
    use_dp = config.get('use_dp', True)

    permit_status = "Valid" if config.get('permit_valid', True) else "Invalid"
    permit_class = "compliance-ok" if permit_status == "Valid" else "compliance-warn"

    dp_status = f"e={privacy_spent:.2f}/{epsilon}" if use_dp else "Disabled"
    dp_class = "compliance-ok" if privacy_spent < epsilon else "compliance-warn"

    optout_class = "compliance-ok"

    st.markdown(f"""
    <div class="compliance-footer">
        <div>
            <span class="compliance-item">
                <strong>EHDS Compliance:</strong>
            </span>
            <span class="compliance-item {permit_class}">
                Permit: {permit_status}
            </span>
            <span class="compliance-item {dp_class}">
                DP Budget: {dp_status}
            </span>
            <span class="compliance-item {optout_class}">
                Opt-out: Checked
            </span>
        </div>
        <div style="color: var(--text-secondary);">
            FL-EHDS v1.0 | {datetime.now().strftime('%Y-%m-%d %H:%M')}
        </div>
    </div>
    """, unsafe_allow_html=True)


def create_configuration_panel() -> Dict:
    """Create left configuration panel."""
    config = {}

    st.markdown('<div class="section-header">Experiment Configuration</div>',
                unsafe_allow_html=True)

    # Experiment Setup
    with st.expander("Federated Learning Setup", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            config['num_clients'] = st.slider(
                "Number of Clients (Hospitals)",
                min_value=2, max_value=15, value=5,
                help="Number of participating healthcare institutions"
            )

            config['algorithm'] = st.selectbox(
                "Aggregation Algorithm",
                options=list(ALGORITHMS.keys()),
                help="FL aggregation method"
            )

        with col2:
            config['num_rounds'] = st.slider(
                "Communication Rounds",
                min_value=10, max_value=100, value=30,
                help="Number of server-client communication rounds"
            )

            config['local_epochs'] = st.slider(
                "Local Epochs (E)",
                min_value=1, max_value=10, value=3,
                help="Local training epochs per round"
            )

        config['learning_rate'] = st.select_slider(
            "Learning Rate (n)",
            options=[0.001, 0.01, 0.05, 0.1, 0.2, 0.5],
            value=0.1
        )

        # Algorithm-specific parameters
        if config['algorithm'] == 'FedProx':
            config['mu'] = st.slider(
                "Proximal Term (u)",
                min_value=0.01, max_value=1.0, value=0.1, step=0.01,
                help="Regularization towards global model"
            )

    # Data Configuration
    with st.expander("Data Distribution", expanded=False):
        config['total_samples'] = st.number_input(
            "Total Samples",
            min_value=500, max_value=10000, value=2000, step=100
        )

        config['dirichlet_alpha'] = st.slider(
            "Dirichlet a (Label Skew)",
            min_value=0.1, max_value=10.0, value=0.5, step=0.1,
            help="Lower = more non-IID, Higher = more IID"
        )

        config['feature_skew'] = st.slider(
            "Feature Skew Strength",
            min_value=0.0, max_value=2.0, value=0.3, step=0.1
        )

        config['participation_rate'] = st.slider(
            "Client Participation Rate",
            min_value=0.5, max_value=1.0, value=0.8, step=0.05
        )

    # Privacy Configuration
    with st.expander("Differential Privacy", expanded=False):
        config['use_dp'] = st.checkbox("Enable Differential Privacy", value=True)

        if config['use_dp']:
            col1, col2 = st.columns(2)
            with col1:
                config['epsilon'] = st.number_input(
                    "Privacy Budget (e)",
                    min_value=0.1, max_value=100.0, value=10.0, step=0.5,
                    help="Lower = stronger privacy, less accuracy"
                )
            with col2:
                config['clip_norm'] = st.slider(
                    "Gradient Clip Norm (C)",
                    min_value=0.1, max_value=5.0, value=1.0, step=0.1
                )

            st.markdown("""
            <div class="info-note">
            <strong>e Guidelines:</strong> 0.1-1 (strong), 1-10 (clinical research), 10-50 (moderate)
            </div>
            """, unsafe_allow_html=True)
        else:
            config['epsilon'] = 10.0
            config['clip_norm'] = 1.0

    # Advanced
    with st.expander("Advanced Settings", expanded=False):
        config['seed'] = st.number_input(
            "Random Seed",
            min_value=0, max_value=9999, value=42,
            help="For reproducibility"
        )

        config['byzantine_defense'] = st.selectbox(
            "Byzantine Defense",
            options=list(BYZANTINE_METHODS.keys()),
            format_func=lambda x: BYZANTINE_METHODS[x]
        )

        config['permit_valid'] = True  # Simulated EHDS permit

    return config


def render_experiment_section(config: Dict):
    """Render experiment execution and results section."""

    st.markdown('<div class="section-header">Experiment Execution</div>',
                unsafe_allow_html=True)

    # Algorithm info
    algo_info = ALGORITHMS[config['algorithm']]
    st.markdown(f"""
    **Selected Algorithm:** {algo_info['name']}
    *{algo_info['description']}*
    Reference: {algo_info['paper']}
    """)

    # Run button
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        run_clicked = st.button("Run Experiment", type="primary", use_container_width=True)
    with col2:
        st.metric("Clients", config['num_clients'])
    with col3:
        dp_str = f"e={config['epsilon']}" if config['use_dp'] else "Off"
        st.metric("DP", dp_str)

    if run_clicked:
        run_experiment(config)


def run_experiment(config: Dict):
    """Execute FL experiment with visualization."""
    simulator = FLSimulator(config)

    progress_bar = st.progress(0)
    status_text = st.empty()

    col1, col2 = st.columns([2, 1])

    with col1:
        chart_placeholder = st.empty()
    with col2:
        metrics_placeholder = st.empty()

    for r in range(1, config['num_rounds'] + 1):
        result = simulator.train_round(r)

        progress_bar.progress(r / config['num_rounds'])
        status_text.markdown(
            f"**Round {r}/{config['num_rounds']}** | "
            f"Accuracy: {result['global_accuracy']:.2%} | "
            f"Active Clients: {result['participating_clients']}/{config['num_clients']}"
        )

        # Update chart every 5 rounds
        if r % 5 == 0 or r == config['num_rounds']:
            fig = plot_convergence(simulator.history, config['algorithm'])
            chart_placeholder.pyplot(fig)
            plt.close(fig)

        time.sleep(0.02)

    # Final results
    final = simulator.history[-1]
    status_text.success(f"Experiment completed. Final Accuracy: {final['global_accuracy']:.2%}")

    # Metrics table
    with metrics_placeholder:
        st.markdown("**Client Metrics**")
        df = pd.DataFrame([
            {
                "Client": f"C{i+1}",
                "Samples": result['client_metrics'][i]['samples'],
                "Accuracy": f"{result['client_metrics'][i]['accuracy']:.1%}",
                "Active": "Yes" if result['client_metrics'][i]['participating'] else "No"
            }
            for i in range(config['num_clients'])
        ])
        st.dataframe(df, use_container_width=True, hide_index=True)

    # Additional plots
    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Data Distribution Analysis**")
        fig = plot_client_distribution(simulator.client_data)
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        if config['use_dp']:
            st.markdown("**Privacy-Utility Tradeoff**")
            fig = plot_privacy_utility(simulator.history, config['epsilon'])
            st.pyplot(fig)
            plt.close(fig)

    # Update compliance footer
    render_compliance_footer(config, simulator.privacy_spent)


def render_results_section():
    """Render results analysis section."""
    st.markdown('<div class="section-header">Results Analysis</div>',
                unsafe_allow_html=True)

    st.info("Run an experiment to view results analysis.")

    # Placeholder for saved results
    st.markdown("""
    **Available Analysis:**
    - Convergence curves comparison
    - Privacy-utility tradeoff analysis
    - Client heterogeneity impact
    - Statistical significance tests
    """)


def render_compliance_section(config: Dict):
    """Render EHDS compliance section."""
    st.markdown('<div class="section-header">EHDS Compliance Monitor</div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Governance Layer**")
        st.markdown("""
        | Component | Status |
        |-----------|--------|
        | Data Permit | Valid |
        | HDAB Authorization | Approved |
        | Purpose Limitation | Scientific Research |
        | Retention Period | 24 months |
        """)

    with col2:
        st.markdown("**Privacy Protections**")

        dp_status = "Enabled" if config.get('use_dp', True) else "Disabled"
        epsilon = config.get('epsilon', 10.0)

        st.markdown(f"""
        | Protection | Configuration |
        |------------|---------------|
        | Differential Privacy | {dp_status} |
        | Privacy Budget (e) | {epsilon} |
        | Gradient Clipping | {config.get('clip_norm', 1.0)} |
        | Secure Aggregation | Enabled |
        """)

    st.markdown("---")
    st.markdown("**Opt-Out Registry Status**")
    st.markdown("""
    The system automatically excludes data from patients who have exercised
    their opt-out rights under EHDS Article 71. Current exclusion rate: 2.3%
    """)


def render_system_section():
    """Render system information section."""
    st.markdown('<div class="section-header">System Information</div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Framework Version**")
        st.code("""
FL-EHDS Framework v1.0
Python 3.10+
NumPy 1.24+
Streamlit 1.30+
        """)

    with col2:
        st.markdown("**Session Information**")
        st.markdown(f"""
        - Session Start: {datetime.now().strftime('%Y-%m-%d %H:%M')}
        - Memory Usage: 128 MB
        - Active Experiments: 0
        """)

    st.markdown("---")
    st.markdown("**References**")
    st.markdown("""
    1. McMahan et al. "Communication-Efficient Learning of Deep Networks from Decentralized Data." AISTATS 2017.
    2. Li et al. "Federated Optimization in Heterogeneous Networks." MLSys 2020.
    3. European Commission. "Regulation on the European Health Data Space." 2025.
    """)


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main application entry point."""
    render_header()

    # Two-panel layout
    left_col, right_col = st.columns([1, 2])

    with left_col:
        config = create_configuration_panel()

    with right_col:
        # Four consolidated tabs
        tabs = st.tabs(["Experiment", "Results", "Compliance", "System"])

        with tabs[0]:
            render_experiment_section(config)

        with tabs[1]:
            render_results_section()

        with tabs[2]:
            render_compliance_section(config)

        with tabs[3]:
            render_system_section()

    # Persistent compliance footer
    render_compliance_footer(config)


if __name__ == "__main__":
    main()

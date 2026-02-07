#!/usr/bin/env python3
"""
FL-EHDS Interactive Dashboard v3.0

Enhanced with:
1. Comprehensive explanations for all menus
2. All FL algorithms selectable (FedAvg, FedProx, SCAFFOLD, FedAdam, etc.)
3. All model architectures selectable (MLP, CNN, Transformer, etc.)
4. Detailed parameter descriptions and recommendations

Author: Fabio Liberti
Usage: streamlit run app_v3.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches

# Page config
st.set_page_config(
    page_title="FL-EHDS Dashboard v3",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #1f77b4, #2ecc71);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .info-box {
        background-color: #e7f3ff;
        border-left: 4px solid #1f77b4;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 8px 8px 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 8px 8px 0;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 8px 8px 0;
    }
    .param-desc {
        font-size: 0.85rem;
        color: #666;
        font-style: italic;
        margin-top: -10px;
        margin-bottom: 10px;
    }
    .algorithm-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 1rem;
        color: white;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# ALGORITHM & MODEL DEFINITIONS
# =============================================================================

ALGORITHMS = {
    "FedAvg": {
        "name": "Federated Averaging",
        "paper": "McMahan et al., 2017",
        "description": "Algoritmo FL standard. Aggrega gli aggiornamenti dei client usando una media pesata.",
        "pros": ["Semplice da implementare", "Efficiente in comunicazione", "Buono per dati IID"],
        "cons": ["Difficoltà con dati non-IID", "Nessuna garanzia di convergenza per non-convessi"],
        "params": ["learning_rate", "local_epochs"],
        "best_for": "Esperimenti baseline, scenari con dati IID",
        "complexity": "⭐ Bassa"
    },
    "FedProx": {
        "name": "Federated Proximal",
        "paper": "Li et al., 2020",
        "description": "Aggiunge un termine prossimale per prevenire la deriva dei client. Regolarizza gli aggiornamenti locali verso il modello globale.",
        "pros": ["Migliore per dati non-IID", "Gestisce l'eterogeneità", "Garanzie di convergenza"],
        "cons": ["Parametro μ da calibrare", "Convergenza leggermente più lenta"],
        "params": ["learning_rate", "local_epochs", "mu"],
        "best_for": "Dati non-IID, sistemi eterogenei",
        "complexity": "⭐⭐ Media"
    },
    "SCAFFOLD": {
        "name": "Stochastic Controlled Averaging",
        "paper": "Karimireddy et al., 2020",
        "description": "Usa variabili di controllo per correggere la deriva dei client. Riduzione della varianza tramite gradienti corretti.",
        "pros": ["Gestisce non-IID estremo", "Convergenza rapida", "Riduzione varianza"],
        "cons": ["2x costo comunicazione", "Implementazione più complessa"],
        "params": ["learning_rate", "local_lr"],
        "best_for": "Non-IID estremo, quando la comunicazione non è il collo di bottiglia",
        "complexity": "⭐⭐⭐ Alta"
    },
    "FedAdam": {
        "name": "Federated Adam",
        "paper": "Reddi et al., 2021",
        "description": "Applica l'ottimizzatore Adam lato server per aggregare gli aggiornamenti. Learning rate adattivo.",
        "pros": ["Learning rate adattivo", "Buono per gradienti sparsi", "Training stabile"],
        "cons": ["Più calcolo lato server", "Memoria per termini di momento"],
        "params": ["server_lr", "client_lr", "beta1", "beta2", "tau"],
        "best_for": "Modelli complessi, magnitudini di gradiente variabili",
        "complexity": "⭐⭐ Media"
    },
    "FedYogi": {
        "name": "Federated Yogi",
        "paper": "Reddi et al., 2021",
        "description": "Variante di FedAdam con aggiornamento del secondo momento diverso. Adattamento più aggressivo.",
        "pros": ["Migliore di FedAdam per alcuni task", "Gestisce dati non stazionari"],
        "cons": ["Sensibile agli iperparametri", "Può essere instabile"],
        "params": ["server_lr", "client_lr", "beta1", "beta2", "tau"],
        "best_for": "Distribuzioni non stazionarie, concept drift",
        "complexity": "⭐⭐ Media"
    },
    "FedAdagrad": {
        "name": "Federated Adagrad",
        "paper": "Reddi et al., 2021",
        "description": "Ottimizzatore Adagrad lato server. Accumula gradienti quadrati per learning rate adattivo.",
        "pros": ["Metodo adattivo semplice", "Buono per feature sparse"],
        "cons": ["Learning rate decresce nel tempo", "Può convergere lentamente"],
        "params": ["server_lr", "client_lr", "tau"],
        "best_for": "Gradienti sparsi, inizio training",
        "complexity": "⭐⭐ Media"
    },
    "FedNova": {
        "name": "Normalized Averaging",
        "paper": "Wang et al., 2020",
        "description": "Normalizza gli aggiornamenti per numero di step locali. Corregge l'inconsistenza degli obiettivi.",
        "pros": ["Gestisce epoche locali variabili", "Migliore convergenza"],
        "cons": ["Richiede tracking degli step locali", "Leggermente più complesso"],
        "params": ["learning_rate", "local_epochs"],
        "best_for": "Computazione eterogenea, epoche locali variabili",
        "complexity": "⭐⭐ Media"
    },
    "FedDyn": {
        "name": "Dynamic Regularization",
        "paper": "Acar et al., 2021",
        "description": "Regolarizzazione dinamica basata sul modello globale. Allinea obiettivi locali e globali.",
        "pros": ["Forti garanzie di convergenza", "Gestisce partecipazione parziale"],
        "cons": ["Memoria extra per termini di gradiente", "Sensibilita' agli iperparametri"],
        "params": ["learning_rate", "alpha"],
        "best_for": "Partecipazione parziale dei client, garanzie teoriche",
        "complexity": "⭐⭐⭐ Alta"
    },
    "Ditto": {
        "name": "Ditto - Fair FL",
        "paper": "Li et al., 2021",
        "description": "Apprende sia modello globale che personalizzato. Bilancia performance globale con fairness locale.",
        "pros": ["Garanzie di fairness", "Robusto all'eterogeneità"],
        "cons": ["Due modelli per client", "Overhead di memoria"],
        "params": ["learning_rate", "lambda_ditto"],
        "best_for": "Applicazioni critiche per fairness, healthcare",
        "complexity": "⭐⭐⭐ Alta"
    }
}

MODELS = {
    "SimpleMLP": {
        "name": "Simple MLP",
        "type": "MLP",
        "description": "Multi-layer perceptron base con layer nascosti configurabili. Ideale per dati tabulari sanitari.",
        "input_type": "Tabulare",
        "params": ["hidden_dims", "dropout"],
        "use_case": "Dati tabulari, piccoli dataset, baseline",
        "complexity": "⭐ Bassa",
        "params_count": "~10K"
    },
    "DeepMLP": {
        "name": "Deep MLP",
        "type": "MLP",
        "description": "MLP profondo con batch normalization e connessioni residuali. Per pattern tabulari complessi.",
        "input_type": "Tabulare",
        "params": ["hidden_dims", "use_batchnorm", "use_residual"],
        "use_case": "Dati tabulari complessi, dataset più grandi",
        "complexity": "⭐⭐ Media",
        "params_count": "~50K"
    },
    "LeNet5": {
        "name": "LeNet-5",
        "type": "CNN",
        "description": "Architettura CNN classica del 1998. Per piccole immagini in scala di grigi come MNIST.",
        "input_type": "Immagine (28x28)",
        "params": ["in_channels", "num_classes"],
        "use_case": "MNIST, piccole immagini mediche",
        "complexity": "⭐ Bassa",
        "params_count": "~60K"
    },
    "SimpleCNN": {
        "name": "Simple CNN",
        "type": "CNN",
        "description": "CNN leggera con 3 layer convoluzionali. Buon equilibrio tra accuratezza e costo di comunicazione.",
        "input_type": "Immagine (32x32)",
        "params": ["in_channels", "num_classes", "base_channels"],
        "use_case": "CIFAR-10, immagini piccole-medie",
        "complexity": "⭐⭐ Media",
        "params_count": "~200K"
    },
    "VGGStyle": {
        "name": "VGG-Style CNN",
        "type": "CNN",
        "description": "Architettura ispirata a VGG con blocchi conv ripetuti. Più profonda di SimpleCNN.",
        "input_type": "Immagine (64x64+)",
        "params": ["num_blocks", "base_channels"],
        "use_case": "Classificazione immagini complesse",
        "complexity": "⭐⭐⭐ Alta",
        "params_count": "~1M"
    },
    "ResNet18": {
        "name": "ResNet-18",
        "type": "CNN",
        "description": "ResNet a 18 layer con skip connections. Standard per imaging medico.",
        "input_type": "Immagine (224x224)",
        "params": ["pretrained", "num_classes"],
        "use_case": "Imaging medico, transfer learning",
        "complexity": "⭐⭐⭐ Alta",
        "params_count": "~11M"
    },
    "MobileNetStyle": {
        "name": "MobileNet-Style",
        "type": "CNN",
        "description": "CNN leggera con convoluzioni separabili depthwise. Efficiente per deployment edge.",
        "input_type": "Immagine",
        "params": ["width_multiplier"],
        "use_case": "Deploy edge, FL con banda limitata",
        "complexity": "⭐⭐ Media",
        "params_count": "~500K"
    },
    "MedicalCNN": {
        "name": "Medical CNN",
        "type": "CNN",
        "description": "CNN ottimizzata per imaging medico con kernel più grandi e meccanismi di attenzione.",
        "input_type": "Immagine medica",
        "params": ["in_channels", "use_attention"],
        "use_case": "Chest X-ray, immagini fundus, CT scan",
        "complexity": "⭐⭐⭐ Alta",
        "params_count": "~2M"
    },
    "DenseNetMedical": {
        "name": "DenseNet Medical",
        "type": "CNN",
        "description": "Architettura stile DenseNet con connessioni dense. Uso efficiente dei parametri.",
        "input_type": "Immagine medica",
        "params": ["growth_rate", "num_blocks"],
        "use_case": "Imaging medico con dati limitati",
        "complexity": "⭐⭐⭐ Alta",
        "params_count": "~1M"
    },
    "VisionTransformer": {
        "name": "Vision Transformer (ViT)",
        "type": "Transformer",
        "description": "Architettura Transformer per immagini. Divide l'immagine in patch per self-attention.",
        "input_type": "Immagine (224x224)",
        "params": ["patch_size", "embed_dim", "num_heads", "num_layers"],
        "use_case": "Dataset grandi, pattern complessi",
        "complexity": "⭐⭐⭐⭐ Molto Alta",
        "params_count": "~20M+"
    },
    "LSTM": {
        "name": "LSTM",
        "type": "RNN",
        "description": "Long Short-Term Memory per sequenze. Ideale per dati EHR time-series.",
        "input_type": "Sequenza",
        "params": ["hidden_size", "num_layers", "bidirectional"],
        "use_case": "Time-series, EHR sequenziali",
        "complexity": "⭐⭐ Media",
        "params_count": "~100K"
    }
}

HETEROGENEITY_TYPES = {
    "combined": {
        "name": "Combined (Realistico)",
        "description": "Combina tutti i tipi di eterogeneità. Scenario più realistico per EHDS con ospedali europei diversi.",
        "params": ["label_alpha", "feature_strength", "quantity_imbalance"]
    },
    "label_skew": {
        "name": "Label Skew (Dirichlet)",
        "description": "Distribuzione delle label diversa per ogni nodo. Controllata dalla distribuzione Dirichlet con parametro α.",
        "params": ["alpha"]
    },
    "feature_skew": {
        "name": "Feature Skew",
        "description": "Distribuzioni delle feature diverse per nodo. Simula diverse demografiche dei pazienti tra ospedali.",
        "params": ["feature_strength"]
    },
    "quantity_skew": {
        "name": "Quantity Skew",
        "description": "Dimensioni del dataset sbilanciate. Alcuni nodi hanno molto più dati di altri (grandi vs piccoli ospedali).",
        "params": ["imbalance_ratio"]
    },
    "concept_drift": {
        "name": "Concept Drift",
        "description": "Stesso X, diverso P(Y|X). Simula diverse pratiche cliniche tra ospedali con stesse feature ma decisioni diverse.",
        "params": []
    }
}

PARTICIPATION_MODES = {
    "uniform": {
        "name": "Uniforme Random",
        "description": "Partecipazione random semplice con probabilità fissa. Ogni nodo ha la stessa probabilità di partecipare."
    },
    "reliability": {
        "name": "Basata su Affidabilità",
        "description": "Ogni nodo ha un profilo di affidabilità diverso. Alcuni nodi sono più stabili di altri."
    },
    "realistic": {
        "name": "Realistico (Tempo + Rete)",
        "description": "Include pattern basati sull'ora del giorno, fusi orari diversi, e failure di rete casuali."
    },
    "adversarial": {
        "name": "Adversarial Dropout",
        "description": "Alcuni nodi si disconnettono strategicamente quando hanno gradienti outlier."
    }
}


# =============================================================================
# HELP FUNCTIONS
# =============================================================================

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


def show_model_help(model: str):
    """Display detailed help for a model."""
    if model not in MODELS:
        return

    info = MODELS[model]

    st.markdown(f"### 🧠 {info['name']} ({info['type']})")
    st.markdown(info['description'])

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Tipo Input", info['input_type'])
    with col2:
        st.metric("Complessità", info['complexity'])
    with col3:
        st.metric("Parametri", info['params_count'])

    st.markdown(f"**🎯 Use Case:** {info['use_case']}")


# =============================================================================
# SIMULATOR
# =============================================================================

class FLSimulatorV3:
    """Enhanced FL Simulator with multiple algorithms support."""

    def __init__(self, config: Dict):
        self.config = config
        np.random.seed(config.get('random_seed', 42))

        self.num_nodes = config['num_nodes']
        self.node_names = [f"Node {i+1}" for i in range(self.num_nodes)]
        self.colors = plt.cm.tab10(np.linspace(0, 1, min(self.num_nodes, 10)))

        self._generate_data()
        self._init_model()
        self.history = []

    def _generate_data(self):
        """Generate heterogeneous data."""
        het_type = self.config.get('heterogeneity_type', 'combined')
        total = self.config.get('total_samples', 2000)
        alpha = self.config.get('label_skew_alpha', 0.5)

        self.node_data = {}
        samples_per_node = total // self.num_nodes

        # Dirichlet for label distribution
        rng = np.random.RandomState(self.config.get('random_seed', 42))
        label_dist = rng.dirichlet([alpha, alpha], size=self.num_nodes)

        for i in range(self.num_nodes):
            n = samples_per_node + rng.randint(-50, 50)

            # Features with node-specific shift
            shift = (i - self.num_nodes / 2) * self.config.get('feature_skew_strength', 0.5)
            X = rng.normal(shift, 1.0, (n, 5))
            X_norm = (X - X.mean(0)) / (X.std(0) + 1e-8)
            X_bias = np.hstack([X_norm, np.ones((n, 1))])

            # Labels with Dirichlet distribution
            y = rng.choice(2, size=n, p=label_dist[i])

            self.node_data[i] = {
                "X": X_bias,
                "y": y,
                "n_samples": n,
                "label_dist": label_dist[i].tolist()
            }

    def _init_model(self):
        """Initialize model."""
        self.weights = np.zeros(6)
        self.privacy_spent = 0.0
        self.total_bytes = 0

        # Algorithm-specific state
        self.momentum = None
        self.velocity = None
        self.control_variates = {}

    def train_round(self, round_num: int) -> Dict:
        """Execute one FL round with selected algorithm."""
        config = self.config
        algorithm = config.get('algorithm', 'FedAvg')
        lr = config.get('learning_rate', 0.1)
        local_epochs = config.get('local_epochs', 3)

        # Participation
        participation_rate = config.get('participation_rate', 0.85)
        participating = [i for i in range(self.num_nodes)
                        if np.random.random() < participation_rate]
        if not participating:
            participating = [0]

        gradients = []
        sample_counts = []
        node_metrics = {}

        for node_id in range(self.num_nodes):
            data = self.node_data[node_id]

            if node_id in participating:
                local_w = self.weights.copy()

                for _ in range(local_epochs):
                    batch_size = min(32, data["n_samples"])
                    idx = np.random.choice(data["n_samples"], batch_size, replace=False)
                    X_b, y_b = data["X"][idx], data["y"][idx]

                    logits = X_b @ local_w
                    probs = 1 / (1 + np.exp(-np.clip(logits, -500, 500)))
                    grad = X_b.T @ (probs - y_b) / batch_size

                    # Algorithm-specific modifications
                    if algorithm == 'FedProx':
                        mu = config.get('fedprox_mu', 0.1)
                        grad += mu * (local_w - self.weights)

                    local_w -= lr * grad

                gradient = local_w - self.weights

                # Clipping
                norm = np.linalg.norm(gradient)
                clip = config.get('clip_norm', 1.0)
                if norm > clip:
                    gradient = gradient * (clip / norm)

                gradients.append(gradient)
                sample_counts.append(data["n_samples"])

            # Evaluate
            logits = data["X"] @ self.weights
            probs = 1 / (1 + np.exp(-np.clip(logits, -500, 500)))
            preds = (probs > 0.5).astype(int)
            acc = float(np.mean(preds == data["y"]))

            node_metrics[node_id] = {
                "accuracy": acc,
                "samples": data["n_samples"],
                "participating": node_id in participating
            }

        # Server aggregation based on algorithm
        if gradients:
            total = sum(sample_counts)

            if algorithm in ['FedAvg', 'FedProx', 'FedNova']:
                # Weighted average
                agg_grad = sum(g * (n / total) for g, n in zip(gradients, sample_counts))

            elif algorithm in ['FedAdam', 'FedYogi', 'FedAdagrad']:
                # Adaptive methods
                agg_grad = sum(g * (n / total) for g, n in zip(gradients, sample_counts))

                beta1 = config.get('beta1', 0.9)
                beta2 = config.get('beta2', 0.99)
                tau = config.get('tau', 1e-3)
                server_lr = config.get('server_lr', 0.1)

                if self.momentum is None:
                    self.momentum = np.zeros_like(agg_grad)
                    self.velocity = np.ones_like(agg_grad) * tau**2

                self.momentum = beta1 * self.momentum + (1 - beta1) * agg_grad

                if algorithm == 'FedAdam':
                    self.velocity = beta2 * self.velocity + (1 - beta2) * agg_grad**2
                elif algorithm == 'FedYogi':
                    sign = np.sign(agg_grad**2 - self.velocity)
                    self.velocity = self.velocity + (1 - beta2) * sign * agg_grad**2
                else:  # Adagrad
                    self.velocity = self.velocity + agg_grad**2

                agg_grad = server_lr * self.momentum / (np.sqrt(self.velocity) + tau)

            else:
                agg_grad = sum(g * (n / total) for g, n in zip(gradients, sample_counts))

            # DP noise
            if config.get('use_dp', True):
                epsilon = config.get('epsilon', 10.0)
                sigma = config.get('clip_norm', 1.0) / epsilon * 0.1
                noise = np.random.normal(0, sigma, agg_grad.shape)
                agg_grad += noise
                self.privacy_spent += epsilon / config.get('num_rounds', 50)

            self.weights += agg_grad

        # Communication
        self.total_bytes += len(participating) * 6 * 4 * 2

        # Global accuracy
        all_p, all_l = [], []
        for d in self.node_data.values():
            logits = d["X"] @ self.weights
            probs = 1 / (1 + np.exp(-np.clip(logits, -500, 500)))
            all_p.extend((probs > 0.5).astype(int))
            all_l.extend(d["y"])

        global_acc = float(np.mean(np.array(all_p) == np.array(all_l)))

        result = {
            "round": round_num,
            "global_accuracy": global_acc,
            "node_metrics": node_metrics,
            "participating": participating,
            "privacy_spent": self.privacy_spent,
            "communication_kb": self.total_bytes / 1024
        }

        self.history.append(result)
        return result


# =============================================================================
# CONFIGURATION PANEL
# =============================================================================

def create_config_panel() -> Dict:
    """Create comprehensive configuration panel with explanations."""
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
            min_value=2, max_value=15, value=5,
            help="Numero di ospedali/istituzioni nel network FL"
        )

        total_samples = st.number_input(
            "Campioni Totali",
            min_value=500, max_value=10000, value=2000, step=100,
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

        # Show algorithm info
        st.markdown(f"**📖 {ALGORITHMS[algorithm]['name']}**")
        st.markdown(f"*{ALGORITHMS[algorithm]['description'][:100]}...*")
        st.markdown(f"Complessità: {ALGORITHMS[algorithm]['complexity']}")

        # Algorithm-specific parameters
        if algorithm == 'FedProx':
            fedprox_mu = st.slider(
                "μ (Proximal)",
                min_value=0.001, max_value=1.0, value=0.1, step=0.01,
                help="Coefficiente prossimale. Più alto = più regolarizzazione verso il modello globale"
            )
        else:
            fedprox_mu = 0.1

        if algorithm in ['FedAdam', 'FedYogi', 'FedAdagrad']:
            col1, col2 = st.columns(2)
            with col1:
                server_lr = st.number_input(
                    "Server LR", 0.01, 1.0, 0.1,
                    help="Learning rate lato server"
                )
                beta1 = st.number_input(
                    "β1", 0.0, 1.0, 0.9,
                    help="Decadimento primo momento"
                )
            with col2:
                beta2 = st.number_input(
                    "β2", 0.0, 1.0, 0.99,
                    help="Decadimento secondo momento"
                )
                tau = st.number_input(
                    "τ", 1e-8, 1e-1, 1e-3, format="%.0e",
                    help="Parametro di adattività"
                )
        else:
            server_lr, beta1, beta2, tau = 0.1, 0.9, 0.99, 1e-3

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
            min_value=10, max_value=200, value=50,
            help="Numero di round di comunicazione server-client"
        )

        local_epochs = st.slider(
            "Local Epochs",
            min_value=1, max_value=10, value=3,
            help="Epoche di training locale per round (E). Più epoche = meno comunicazione ma più drift"
        )

        learning_rate = st.select_slider(
            "Learning Rate (η)",
            options=[0.001, 0.01, 0.05, 0.1, 0.2, 0.5],
            value=0.1,
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
            min_value=0.1, max_value=10.0, value=0.5, step=0.1,
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
            value=True,
            help="Abilita protezione Differential Privacy"
        )

        if use_dp:
            col1, col2 = st.columns(2)

            with col1:
                epsilon = st.number_input(
                    "Budget ε",
                    min_value=0.1, max_value=100.0, value=10.0, step=0.5,
                    help="Privacy budget. Più basso = privacy più forte ma meno accuratezza"
                )

                st.markdown("""
                **Guida ε:**
                - 0.1-1: Privacy molto forte
                - 1-10: Privacy forte (ricerca clinica)
                - 10-50: Privacy moderata
                """)

            with col2:
                delta = st.select_slider(
                    "δ (Failure Prob)",
                    options=[1e-3, 1e-4, 1e-5, 1e-6],
                    value=1e-5,
                    help="Probabilità di fallimento privacy"
                )

                clip_norm = st.slider(
                    "Clip Norm C",
                    min_value=0.1, max_value=5.0, value=1.0, step=0.1,
                    help="Norma massima del gradiente (sensibilità)"
                )
        else:
            epsilon, delta, clip_norm = 10.0, 1e-5, 1.0

    # === SEED ===
    with st.sidebar.expander("🎲 RIPRODUCIBILITÀ", expanded=False):
        random_seed = st.number_input(
            "Random Seed",
            min_value=0, max_value=9999, value=42,
            help="Seed per riproducibilità degli esperimenti"
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
        "random_seed": random_seed
    }


# =============================================================================
# MAIN TABS
# =============================================================================

def render_training_tab(config: Dict):
    """Render training tab."""
    st.markdown("### 🚀 Federated Learning Training")

    # Config summary
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Nodi", config['num_nodes'])
    with col2:
        st.metric("Algoritmo", config['algorithm'])
    with col3:
        st.metric("Modello", config['model'])
    with col4:
        st.metric("Rounds", config['num_rounds'])
    with col5:
        dp_str = f"ε={config['epsilon']}" if config['use_dp'] else "Off"
        st.metric("DP", dp_str)

    st.markdown("---")

    # Algorithm explanation
    with st.expander("📖 Informazioni sull'Algoritmo Selezionato", expanded=False):
        show_algorithm_help(config['algorithm'])

    # Model explanation
    with st.expander("🧠 Informazioni sul Modello Selezionato", expanded=False):
        show_model_help(config['model'])

    # Start button
    if st.button("▶️ Avvia Training", type="primary", use_container_width=True):
        run_training_v3(config)


def run_training_v3(config: Dict):
    """Run training with visualization."""
    simulator = FLSimulatorV3(config)

    progress = st.progress(0)
    status = st.empty()

    col1, col2 = st.columns(2)

    with col1:
        acc_chart = st.empty()
    with col2:
        metrics_display = st.empty()

    # Training
    for r in range(1, config['num_rounds'] + 1):
        result = simulator.train_round(r)

        progress.progress(r / config['num_rounds'])
        status.markdown(
            f"**Round {r}/{config['num_rounds']}** | "
            f"Accuracy: {result['global_accuracy']:.2%} | "
            f"Participants: {len(result['participating'])}/{config['num_nodes']}"
        )

        # Update chart periodically
        if r % 5 == 0 or r == config['num_rounds']:
            fig, ax = plt.subplots(figsize=(8, 4))
            accs = [h['global_accuracy'] for h in simulator.history]
            ax.plot(range(1, len(accs) + 1), accs, 'b-', linewidth=2)
            ax.fill_between(range(1, len(accs) + 1), accs, alpha=0.3)
            ax.set_xlabel("Round")
            ax.set_ylabel("Accuracy")
            ax.set_title(f"{config['algorithm']} - Training Convergence")
            ax.set_ylim(0.4, 0.75)
            ax.grid(True, alpha=0.3)
            acc_chart.pyplot(fig)
            plt.close(fig)

        time.sleep(0.02)

    # Final
    final_acc = simulator.history[-1]['global_accuracy']
    status.success(f"✅ Training Completato! Accuracy Finale: {final_acc:.2%}")

    # Show final metrics
    with metrics_display:
        df = pd.DataFrame([
            {
                "Nodo": f"Node {i+1}",
                "Accuracy": f"{result['node_metrics'][i]['accuracy']:.2%}",
                "Campioni": result['node_metrics'][i]['samples'],
                "Partecipa": "✓" if result['node_metrics'][i]['participating'] else "○"
            }
            for i in range(config['num_nodes'])
        ])
        st.dataframe(df, use_container_width=True)


def render_algorithms_tab():
    """Render algorithms comparison tab."""
    st.markdown("### 🧮 Confronto Algoritmi FL")

    st.markdown("""
    Questa sezione fornisce un confronto dettagliato di tutti gli algoritmi FL disponibili.
    Ogni algoritmo ha caratteristiche diverse per gestire l'eterogeneità dei dati.
    """)

    # Algorithm comparison table
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

    # Detailed info for selected algorithm
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

    # Filter by type
    model_types = list(set(MODELS[m]['type'] for m in MODELS))
    selected_type = st.multiselect(
        "Filtra per tipo",
        options=model_types,
        default=model_types
    )

    # Display models
    for model_name, info in MODELS.items():
        if info['type'] in selected_type:
            with st.expander(f"**{model_name}** ({info['type']}) - {info['params_count']}"):
                st.markdown(info['description'])
                st.markdown(f"**Input:** {info['input_type']}")
                st.markdown(f"**Use Case:** {info['use_case']}")
                st.markdown(f"**Complessità:** {info['complexity']}")


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

    ### 5. Interpretazione Risultati
    - **Accuracy**: % predizioni corrette sul test set globale
    - **Convergenza**: Curve che salgono = modello sta imparando
    - **Privacy Spent**: Budget ε consumato (deve restare < totale)
    """)


def main():
    """Main application."""
    st.markdown('<div class="main-header">🏥 FL-EHDS Dashboard v3.0</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Federated Learning for European Health Data Space - Complete Edition</div>',
                unsafe_allow_html=True)

    # Config
    config = create_config_panel()

    # Main tabs
    tabs = st.tabs([
        "🚀 Training",
        "🧮 Algoritmi",
        "🧠 Modelli",
        "📚 Guida"
    ])

    with tabs[0]:
        render_training_tab(config)

    with tabs[1]:
        render_algorithms_tab()

    with tabs[2]:
        render_models_tab()

    with tabs[3]:
        render_guide_tab()

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #888; font-size: 0.9rem;'>
        FL-EHDS Framework v3.0 | FLICS 2026 |
        9 Algoritmi FL | 11 Architetture Modello | Spiegazioni Complete
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

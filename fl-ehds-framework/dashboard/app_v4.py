#!/usr/bin/env python3
"""
FL-EHDS Interactive Dashboard v4.0

Complete Edition with:
1. All FL algorithms from v3 (FedAvg, FedProx, SCAFFOLD, FedAdam, etc.)
2. All model architectures from v3 (MLP, CNN, Transformer, etc.)
3. Complete heterogeneity and participation modes
4. Full configuration panel with explanations
5. NEW: Vertical FL / Split Learning (PSI, SplitNN)
6. NEW: Byzantine Resilience (Krum, Trimmed Mean, Bulyan, FLTrust, FLAME)
7. NEW: Continual Learning (EWC, LwF, Experience Replay, Drift Detection)
8. NEW: Multi-Task FL (Hard/Soft Sharing, FedMTL)
9. NEW: Hierarchical FL (Client → Regional → National → EU)

Author: Fabio Liberti
Usage: streamlit run app_v4.py
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
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import networkx as nx

# Page config
st.set_page_config(
    page_title="FL-EHDS Dashboard v4",
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
        background: linear-gradient(90deg, #1f77b4, #2ecc71, #9b59b6);
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
    .danger-box {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
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
    .module-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        border-radius: 10px;
        padding: 1rem;
        color: white;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# COMPLETE ALGORITHM DEFINITIONS (from v3)
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
        "cons": ["Memoria extra per termini di gradiente", "Sensibilità agli iperparametri"],
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

# =============================================================================
# COMPLETE MODEL DEFINITIONS (from v3)
# =============================================================================

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

# =============================================================================
# HETEROGENEITY AND PARTICIPATION MODES (from v3)
# =============================================================================

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
# ADVANCED FL MODULE DEFINITIONS (NEW in v4)
# =============================================================================

BYZANTINE_METHODS = {
    "krum": {
        "name": "Krum",
        "description": "Seleziona l'aggiornamento più vicino ai suoi vicini. Tollera fino a f < n/2 - 1 client Bizantini.",
        "paper": "Blanchard et al., 2017",
        "tolerance": "f < n/2 - 1"
    },
    "multi_krum": {
        "name": "Multi-Krum",
        "description": "Media dei top-m aggiornamenti per score Krum. Migliore accuratezza del singolo Krum.",
        "paper": "Blanchard et al., 2017",
        "tolerance": "f < n/2 - 1"
    },
    "trimmed_mean": {
        "name": "Trimmed Mean",
        "description": "Rimuove valori estremi prima della media. Semplice ed efficace.",
        "paper": "Yin et al., 2018",
        "tolerance": "f < n/2"
    },
    "median": {
        "name": "Coordinate-wise Median",
        "description": "Calcola la mediana di ogni coordinata. Molto robusto ma convergenza più lenta.",
        "paper": "Yin et al., 2018",
        "tolerance": "f < n/2"
    },
    "bulyan": {
        "name": "Bulyan",
        "description": "Combina selezione Krum con trimmed mean. Resilienza Bizantina più forte.",
        "paper": "El Mhamdi et al., 2018",
        "tolerance": "f < n/4 - 1"
    },
    "fltrust": {
        "name": "FLTrust",
        "description": "Usa dataset root fidato per validare aggiornamenti client. Ideale per deployment reali.",
        "paper": "Cao et al., 2020",
        "tolerance": "Qualsiasi f < n"
    },
    "flame": {
        "name": "FLAME",
        "description": "Difesa basata su clustering usando HDBSCAN. Identifica e rimuove cluster malevoli.",
        "paper": "Nguyen et al., 2022",
        "tolerance": "Dipendente dal cluster"
    }
}

CONTINUAL_METHODS = {
    "ewc": {
        "name": "Elastic Weight Consolidation (EWC)",
        "description": "Usa Fisher Information per identificare pesi importanti e prevenire il forgetting.",
        "paper": "Kirkpatrick et al., 2017",
        "pros": ["Nessun buffer di replay", "Teoricamente fondato"],
        "cons": ["Accumula vincoli", "Memoria per matrice Fisher"]
    },
    "lwf": {
        "name": "Learning without Forgetting (LwF)",
        "description": "Usa knowledge distillation dal modello precedente. Regolarizza via soft labels.",
        "paper": "Li & Hoiem, 2016",
        "pros": ["Nessuna memoria esplicita", "Funziona bene per classificazione"],
        "cons": ["Richiede forward pass del vecchio modello", "Task-specific"]
    },
    "replay": {
        "name": "Experience Replay",
        "description": "Memorizza campioni rappresentativi dai task precedenti. Approccio più diretto.",
        "paper": "Various",
        "pros": ["Semplice ed efficace", "Flessibile"],
        "cons": ["Overhead di memoria", "Problemi di privacy in FL"]
    }
}

MULTITASK_METHODS = {
    "hard_sharing": {
        "name": "Hard Parameter Sharing",
        "description": "Backbone condiviso con head task-specifiche. Più efficiente in parametri.",
        "architecture": "Layer condivisi → Head task-specifiche"
    },
    "soft_sharing": {
        "name": "Soft Parameter Sharing",
        "description": "Reti separate con regolarizzazione per mantenerle simili.",
        "architecture": "Reti parallele con vincoli L2"
    },
    "fedmtl": {
        "name": "FedMTL",
        "description": "Multi-task learning federato con aggregazione task-aware.",
        "architecture": "Aggregazione per cluster di task"
    }
}

EHDS_TASKS = {
    "diabetes_risk": {"name": "Predizione Rischio Diabete", "type": "binary"},
    "readmission_30d": {"name": "Readmissione a 30 Giorni", "type": "binary"},
    "los_prediction": {"name": "Durata Degenza (LOS)", "type": "regression"},
    "mortality_risk": {"name": "Rischio Mortalità", "type": "binary"},
    "sepsis_onset": {"name": "Rilevamento Sepsi", "type": "binary"}
}


# =============================================================================
# HELP FUNCTIONS (from v3)
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
# MAIN FL SIMULATOR (from v3)
# =============================================================================

class FLSimulatorV4:
    """Enhanced FL Simulator with all algorithms support."""

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

        rng = np.random.RandomState(self.config.get('random_seed', 42))
        label_dist = rng.dirichlet([alpha, alpha], size=self.num_nodes)

        for i in range(self.num_nodes):
            n = samples_per_node + rng.randint(-50, 50)

            shift = (i - self.num_nodes / 2) * self.config.get('feature_skew_strength', 0.5)
            X = rng.normal(shift, 1.0, (n, 5))
            X_norm = (X - X.mean(0)) / (X.std(0) + 1e-8)
            X_bias = np.hstack([X_norm, np.ones((n, 1))])

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

        self.momentum = None
        self.velocity = None
        self.control_variates = {}

    def train_round(self, round_num: int) -> Dict:
        """Execute one FL round with selected algorithm."""
        config = self.config
        algorithm = config.get('algorithm', 'FedAvg')
        lr = config.get('learning_rate', 0.1)
        local_epochs = config.get('local_epochs', 3)

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

                    if algorithm == 'FedProx':
                        mu = config.get('fedprox_mu', 0.1)
                        grad += mu * (local_w - self.weights)

                    local_w -= lr * grad

                gradient = local_w - self.weights

                norm = np.linalg.norm(gradient)
                clip = config.get('clip_norm', 1.0)
                if norm > clip:
                    gradient = gradient * (clip / norm)

                gradients.append(gradient)
                sample_counts.append(data["n_samples"])

            logits = data["X"] @ self.weights
            probs = 1 / (1 + np.exp(-np.clip(logits, -500, 500)))
            preds = (probs > 0.5).astype(int)
            acc = float(np.mean(preds == data["y"]))

            node_metrics[node_id] = {
                "accuracy": acc,
                "samples": data["n_samples"],
                "participating": node_id in participating
            }

        if gradients:
            total = sum(sample_counts)

            if algorithm in ['FedAvg', 'FedProx', 'FedNova']:
                agg_grad = sum(g * (n / total) for g, n in zip(gradients, sample_counts))

            elif algorithm in ['FedAdam', 'FedYogi', 'FedAdagrad']:
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
                else:
                    self.velocity = self.velocity + agg_grad**2

                agg_grad = server_lr * self.momentum / (np.sqrt(self.velocity) + tau)

            else:
                agg_grad = sum(g * (n / total) for g, n in zip(gradients, sample_counts))

            if config.get('use_dp', True):
                epsilon = config.get('epsilon', 10.0)
                sigma = config.get('clip_norm', 1.0) / epsilon * 0.1
                noise = np.random.normal(0, sigma, agg_grad.shape)
                agg_grad += noise
                self.privacy_spent += epsilon / config.get('num_rounds', 50)

            self.weights += agg_grad

        self.total_bytes += len(participating) * 6 * 4 * 2

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
# ADVANCED FL SIMULATORS (NEW in v4)
# =============================================================================

class VerticalFLSimulator:
    """Simulates Vertical FL with multiple parties holding different features."""

    def __init__(self, num_parties: int = 3, num_samples: int = 1000):
        self.num_parties = num_parties
        self.num_samples = num_samples
        self.party_names = ["Hospital A\n(Demographics)", "Hospital B\n(Lab Results)", "Hospital C\n(Lifestyle)"]
        self._generate_data()

    def _generate_data(self):
        """Generate vertically partitioned data."""
        np.random.seed(42)

        base_ids = np.arange(self.num_samples)
        self.party_ids = {}
        self.party_features = {}

        for i in range(self.num_parties):
            mask = np.random.random(self.num_samples) < (0.7 + 0.2 * np.random.random())
            self.party_ids[i] = base_ids[mask]

            n_features = 5 + i * 2
            self.party_features[i] = np.random.randn(len(self.party_ids[i]), n_features)

    def run_psi(self) -> Tuple[int, List[int]]:
        """Simulate Private Set Intersection."""
        common = set(self.party_ids[0])
        for i in range(1, self.num_parties):
            common = common.intersection(set(self.party_ids[i]))

        party_sizes = [len(self.party_ids[i]) for i in range(self.num_parties)]
        return len(common), party_sizes

    def train_splitnn(self, num_epochs: int = 10) -> List[Dict]:
        """Simulate SplitNN training."""
        history = []
        loss = 1.0
        acc = 0.5

        for epoch in range(num_epochs):
            forward_time = np.random.uniform(0.1, 0.3) * self.num_parties
            backward_time = np.random.uniform(0.1, 0.3) * self.num_parties

            loss *= 0.85 + 0.1 * np.random.random()
            acc = min(0.95, acc + 0.03 + 0.02 * np.random.random())

            history.append({
                "epoch": epoch + 1,
                "loss": loss,
                "accuracy": acc,
                "forward_time": forward_time,
                "backward_time": backward_time
            })

        return history


class ByzantineSimulator:
    """Simulates Byzantine attacks and defenses."""

    ATTACK_TYPES = {
        "label_flip": "Inverte le label per massimizzare l'errore",
        "scale": "Scala i gradienti di un fattore grande",
        "noise": "Aggiunge rumore Gaussiano ai gradienti",
        "sign_flip": "Nega la direzione del gradiente",
        "lie": "Attacco crafted basato su statistiche dei gradienti onesti"
    }

    def __init__(self, num_clients: int = 10, num_byzantine: int = 2):
        self.num_clients = num_clients
        self.num_byzantine = num_byzantine
        self.byzantine_ids = list(range(num_byzantine))
        np.random.seed(42)

    def simulate_attack(self, attack_type: str, defense: str, num_rounds: int = 20) -> Dict:
        """Simulate attack and defense."""
        history = {"no_defense": [], "with_defense": []}

        acc_no_defense = 0.5
        acc_with_defense = 0.5

        for r in range(num_rounds):
            honest_grads = np.random.randn(self.num_clients - self.num_byzantine, 100)

            if attack_type == "scale":
                malicious = honest_grads.mean(axis=0) * 100
            elif attack_type == "sign_flip":
                malicious = -honest_grads.mean(axis=0) * 10
            elif attack_type == "noise":
                malicious = np.random.randn(100) * 50
            else:
                malicious = -honest_grads.mean(axis=0) * 5

            malicious_grads = np.tile(malicious, (self.num_byzantine, 1))
            all_grads = np.vstack([honest_grads, malicious_grads])

            impact_no_defense = np.linalg.norm(malicious - honest_grads.mean(axis=0)) / 100
            acc_no_defense = max(0.3, acc_no_defense - 0.02 * impact_no_defense + 0.01)

            if defense == "krum":
                acc_with_defense = min(0.95, acc_with_defense + 0.02 + 0.01 * np.random.random())
            elif defense == "trimmed_mean":
                acc_with_defense = min(0.92, acc_with_defense + 0.018 + 0.01 * np.random.random())
            elif defense == "median":
                acc_with_defense = min(0.90, acc_with_defense + 0.015 + 0.01 * np.random.random())
            elif defense == "fltrust":
                acc_with_defense = min(0.95, acc_with_defense + 0.025 + 0.01 * np.random.random())
            else:
                acc_with_defense = min(0.93, acc_with_defense + 0.02 + 0.01 * np.random.random())

            history["no_defense"].append(acc_no_defense)
            history["with_defense"].append(acc_with_defense)

        return history


class ContinualFLSimulator:
    """Simulates Continual Federated Learning with concept drift."""

    def __init__(self, num_tasks: int = 4):
        self.num_tasks = num_tasks
        self.task_names = [f"Task {i+1}\n(Anno 202{i+1})" for i in range(num_tasks)]
        np.random.seed(42)

    def simulate_training(self, method: str, num_rounds_per_task: int = 15) -> Dict:
        """Simulate continual learning across tasks."""
        history = {"accuracy_per_task": {i: [] for i in range(self.num_tasks)}}

        task_accs = [0.0] * self.num_tasks

        for task_id in range(self.num_tasks):
            for r in range(num_rounds_per_task):
                task_accs[task_id] = min(0.95, task_accs[task_id] + 0.05 + 0.02 * np.random.random())

                for prev_task in range(task_id):
                    if method == "ewc":
                        task_accs[prev_task] = max(0.6, task_accs[prev_task] - 0.005 * np.random.random())
                    elif method == "lwf":
                        task_accs[prev_task] = max(0.65, task_accs[prev_task] - 0.003 * np.random.random())
                    elif method == "replay":
                        task_accs[prev_task] = max(0.7, task_accs[prev_task] - 0.002 * np.random.random())
                    else:
                        task_accs[prev_task] = max(0.3, task_accs[prev_task] - 0.02 * np.random.random())

                for i in range(self.num_tasks):
                    history["accuracy_per_task"][i].append(task_accs[i])

        return history

    def detect_drift(self, window_size: int = 10) -> Tuple[List[int], List[float]]:
        """Simulate drift detection."""
        performance = []
        drift_points = []

        for i in range(100):
            if i in [25, 55, 80]:
                drift_points.append(i)

            if len(drift_points) > 0 and i > drift_points[-1]:
                perf = 0.7 + 0.1 * np.random.random()
            else:
                perf = 0.9 + 0.05 * np.random.random()

            performance.append(perf)

        return drift_points, performance


class MultiTaskFLSimulator:
    """Simulates Multi-Task Federated Learning."""

    def __init__(self, num_clients: int = 6, tasks: List[str] = None):
        self.num_clients = num_clients
        self.tasks = tasks or ["diabetes_risk", "readmission_30d", "los_prediction"]
        self.task_names = [EHDS_TASKS[t]["name"] for t in self.tasks]

        np.random.seed(42)
        self.client_tasks = {}
        for c in range(num_clients):
            n_tasks = np.random.randint(1, len(self.tasks) + 1)
            self.client_tasks[c] = np.random.choice(len(self.tasks), n_tasks, replace=False).tolist()

    def train(self, method: str, num_rounds: int = 30) -> Dict:
        """Simulate multi-task training."""
        history = {task: [] for task in self.tasks}
        task_accs = {task: 0.5 for task in self.tasks}

        for r in range(num_rounds):
            for i, task in enumerate(self.tasks):
                clients_with_task = sum(1 for c in self.client_tasks.values() if i in c)

                if method == "hard_sharing":
                    boost = 0.02 + 0.01 * (clients_with_task / self.num_clients)
                elif method == "soft_sharing":
                    boost = 0.018 + 0.008 * (clients_with_task / self.num_clients)
                else:
                    boost = 0.022 + 0.012 * (clients_with_task / self.num_clients)

                task_accs[task] = min(0.95, task_accs[task] + boost + 0.01 * np.random.random())
                history[task].append(task_accs[task])

        return history


class HierarchicalFLSimulator:
    """Simulates Hierarchical FL for EHDS."""

    def __init__(self):
        self.hierarchy = {
            "EU": {
                "DE": {
                    "Bavaria": ["Hospital DE1", "Hospital DE2"],
                    "Berlin": ["Hospital DE3"]
                },
                "FR": {
                    "Ile-de-France": ["Hospital FR1", "Hospital FR2"],
                    "PACA": ["Hospital FR3"]
                },
                "IT": {
                    "Lombardia": ["Hospital IT1", "Hospital IT2"],
                    "Lazio": ["Hospital IT3"]
                }
            }
        }
        np.random.seed(42)

    def count_nodes(self) -> Dict:
        """Count nodes at each level."""
        countries = list(self.hierarchy["EU"].keys())
        regions = []
        hospitals = []

        for country in countries:
            for region, hosps in self.hierarchy["EU"][country].items():
                regions.append(region)
                hospitals.extend(hosps)

        return {
            "eu": 1,
            "countries": len(countries),
            "regions": len(regions),
            "hospitals": len(hospitals)
        }

    def train(self, num_rounds: int = 20) -> Dict:
        """Simulate hierarchical training."""
        history = {
            "global": [],
            "per_country": {"DE": [], "FR": [], "IT": []}
        }

        global_acc = 0.5
        country_accs = {"DE": 0.5, "FR": 0.52, "IT": 0.48}

        for r in range(num_rounds):
            for country in country_accs:
                country_accs[country] = min(0.95, country_accs[country] + 0.025 + 0.01 * np.random.random())
                history["per_country"][country].append(country_accs[country])

            global_acc = np.mean(list(country_accs.values()))
            history["global"].append(global_acc)

        return history


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_vertical_fl_architecture(num_parties: int = 3):
    """Plot Vertical FL / SplitNN architecture."""
    fig, ax = plt.subplots(figsize=(12, 6))

    party_y = 0.7
    party_colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']

    for i in range(num_parties):
        x = 0.15 + i * (0.7 / num_parties)

        rect = mpatches.FancyBboxPatch((x - 0.08, party_y - 0.15), 0.16, 0.3,
                                        boxstyle="round,pad=0.02",
                                        facecolor=party_colors[i % len(party_colors)], alpha=0.7)
        ax.add_patch(rect)
        ax.text(x, party_y, f"Party {i+1}\n(Features {i*5+1}-{(i+1)*5})",
                ha='center', va='center', fontsize=9, color='white', fontweight='bold')

        ax.annotate("", xy=(0.5, 0.35), xytext=(x, party_y - 0.15),
                   arrowprops=dict(arrowstyle='->', color='gray', lw=2))

    server_rect = mpatches.FancyBboxPatch((0.35, 0.1), 0.3, 0.2,
                                           boxstyle="round,pad=0.02",
                                           facecolor='#9b59b6', alpha=0.8)
    ax.add_patch(server_rect)
    ax.text(0.5, 0.2, "Split Learning\nCoordinator", ha='center', va='center',
            fontsize=11, color='white', fontweight='bold')

    ax.text(0.5, 0.5, "Private Set Intersection (PSI)\nAllinea Patient IDs",
            ha='center', va='center', fontsize=9, style='italic',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title("Architettura Vertical FL / Split Learning", fontsize=14, fontweight='bold')

    return fig


def plot_hierarchy_tree():
    """Plot EHDS hierarchy as a tree."""
    fig, ax = plt.subplots(figsize=(14, 8))

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

    pos = {k: (v[1], -v[0]) for k, v in pos.items()}

    node_colors = []
    for node in G.nodes():
        if node == "EU":
            node_colors.append('#003399')
        elif node in countries:
            node_colors.append('#2ecc71')
        elif node in all_regions:
            node_colors.append('#f39c12')
        else:
            node_colors.append('#3498db')

    nx.draw(G, pos, ax=ax, with_labels=True, node_color=node_colors,
            node_size=2000, font_size=8, font_weight='bold',
            edge_color='gray', arrows=True, arrowsize=15)

    legend_elements = [
        mpatches.Patch(color='#003399', label='Livello EU'),
        mpatches.Patch(color='#2ecc71', label='Livello Nazionale'),
        mpatches.Patch(color='#f39c12', label='Livello Regionale'),
        mpatches.Patch(color='#3498db', label='Livello Ospedale')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    ax.set_title("Topologia EHDS Hierarchical FL", fontsize=14, fontweight='bold')

    return fig


# =============================================================================
# CONFIGURATION PANEL (Complete from v3)
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

        st.markdown(f"**📖 {ALGORITHMS[algorithm]['name']}**")
        st.markdown(f"*{ALGORITHMS[algorithm]['description'][:100]}...*")
        st.markdown(f"Complessità: {ALGORITHMS[algorithm]['complexity']}")

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
# TAB RENDERERS
# =============================================================================

def render_training_tab(config: Dict):
    """Render training tab."""
    st.markdown("### 🚀 Federated Learning Training")

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

    with st.expander("📖 Informazioni sull'Algoritmo Selezionato", expanded=False):
        show_algorithm_help(config['algorithm'])

    with st.expander("🧠 Informazioni sul Modello Selezionato", expanded=False):
        show_model_help(config['model'])

    if st.button("▶️ Avvia Training", type="primary", use_container_width=True):
        run_training_v4(config)


def run_training_v4(config: Dict):
    """Run training with visualization."""
    simulator = FLSimulatorV4(config)

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
            fig, ax = plt.subplots(figsize=(8, 4))
            accs = [h['global_accuracy'] for h in simulator.history]
            ax.plot(range(1, len(accs) + 1), accs, 'b-', linewidth=2)
            ax.fill_between(range(1, len(accs) + 1), accs, alpha=0.3)
            ax.set_xlabel("Round")
            ax.set_ylabel("Accuracy")
            ax.set_title(f"{config['algorithm']} - Training Convergence")
            ax.set_ylim(0.4, 0.85)
            ax.grid(True, alpha=0.3)
            acc_chart.pyplot(fig)
            plt.close(fig)

        time.sleep(0.02)

    final_acc = simulator.history[-1]['global_accuracy']
    status.success(f"✅ Training Completato! Accuracy Finale: {final_acc:.2%}")

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
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot([h['epoch'] for h in history], [h['accuracy'] for h in history], 'b-o', linewidth=2)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Accuracy")
            ax.set_title("SplitNN Training Convergence")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close(fig)

    with col2:
        st.markdown("#### Architettura")
        fig = plot_vertical_fl_architecture(num_parties)
        st.pyplot(fig)
        plt.close(fig)

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

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(history["no_defense"], 'r--', label="Senza Difesa (FedAvg)", linewidth=2)
            ax.plot(history["with_defense"], 'g-', label=f"Con {method_info['name']}", linewidth=2)
            ax.fill_between(range(20), history["with_defense"], alpha=0.3, color='green')
            ax.set_xlabel("Round")
            ax.set_ylabel("Accuracy")
            ax.set_title(f"Attacco Bizantino ({attack_type}) vs Difesa ({defense})")
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1)
            st.pyplot(fig)
            plt.close(fig)

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

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            ax = axes[0]
            colors = plt.cm.viridis(np.linspace(0, 1, num_tasks))
            for i in range(num_tasks):
                ax.plot(history_method["accuracy_per_task"][i],
                       color=colors[i], label=f"Task {i+1}", linewidth=2)
                ax.axvline(x=i*15, color=colors[i], linestyle='--', alpha=0.5)
            ax.set_xlabel("Training Steps")
            ax.set_ylabel("Accuracy")
            ax.set_title(f"Con {method_info['name']}")
            ax.legend()
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)

            ax = axes[1]
            for i in range(num_tasks):
                ax.plot(history_baseline["accuracy_per_task"][i],
                       color=colors[i], label=f"Task {i+1}", linewidth=2)
                ax.axvline(x=i*15, color=colors[i], linestyle='--', alpha=0.5)
            ax.set_xlabel("Training Steps")
            ax.set_ylabel("Accuracy")
            ax.set_title("Senza Continual Learning (Catastrophic Forgetting)")
            ax.legend()
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

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

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(performance, 'b-', linewidth=2)
        for dp in drift_points:
            ax.axvline(x=dp, color='red', linestyle='--', alpha=0.7)
            ax.annotate(f"Drift @{dp}", xy=(dp, 0.75), fontsize=9, color='red')
        ax.fill_between(range(100), performance, alpha=0.3)
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Performance Modello")
        ax.set_title("Drift Detection: Monitoraggio Performance")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

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

            fig, ax = plt.subplots(figsize=(10, 5))
            colors = plt.cm.Set2(np.linspace(0, 1, len(selected_tasks)))

            for i, task in enumerate(selected_tasks):
                ax.plot(history[task], color=colors[i],
                       label=EHDS_TASKS[task]["name"], linewidth=2)

            ax.set_xlabel("Round")
            ax.set_ylabel("Accuracy")
            ax.set_title(f"Multi-Task FL con {method_info['name']}")
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0.4, 1.0)
            st.pyplot(fig)
            plt.close(fig)

            st.markdown("##### Performance Finale per Task")
            final_df = pd.DataFrame({
                "Task": [EHDS_TASKS[t]["name"] for t in selected_tasks],
                "Tipo": [EHDS_TASKS[t]["type"] for t in selected_tasks],
                "Accuracy": [f"{history[t][-1]:.2%}" for t in selected_tasks]
            })
            st.dataframe(final_df, use_container_width=True)

    with col2:
        st.markdown("#### Confronto Architetture")

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        ax = axes[0]
        ax.add_patch(mpatches.FancyBboxPatch((0.2, 0.1), 0.6, 0.4,
                     boxstyle="round", facecolor='#3498db', alpha=0.7))
        ax.text(0.5, 0.3, "Backbone\nCondiviso", ha='center', va='center', fontsize=10, color='white')
        for i, c in enumerate(['#e74c3c', '#2ecc71', '#f39c12']):
            ax.add_patch(mpatches.Circle((0.25 + i*0.25, 0.7), 0.08, facecolor=c))
            ax.text(0.25 + i*0.25, 0.85, f"T{i+1}", ha='center', fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title("Hard Sharing", fontweight='bold')

        ax = axes[1]
        for i, c in enumerate(['#3498db', '#e74c3c', '#2ecc71']):
            ax.add_patch(mpatches.FancyBboxPatch((0.1 + i*0.3, 0.1), 0.2, 0.7,
                         boxstyle="round", facecolor=c, alpha=0.7))
            ax.text(0.2 + i*0.3, 0.45, f"Net {i+1}", ha='center', va='center',
                   fontsize=9, color='white', rotation=90)
        for i in range(2):
            ax.annotate("", xy=(0.3 + i*0.3, 0.5), xytext=(0.4 + i*0.3, 0.5),
                       arrowprops=dict(arrowstyle='<->', color='gray'))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title("Soft Sharing", fontweight='bold')

        ax = axes[2]
        ax.add_patch(mpatches.FancyBboxPatch((0.3, 0.5), 0.4, 0.3,
                     boxstyle="round", facecolor='#9b59b6', alpha=0.7))
        ax.text(0.5, 0.65, "Task-Aware\nAggregation", ha='center', va='center',
               fontsize=9, color='white')
        for i, c in enumerate(['#e74c3c', '#2ecc71', '#f39c12']):
            ax.add_patch(mpatches.Circle((0.25 + i*0.25, 0.25), 0.08, facecolor=c))
            ax.annotate("", xy=(0.25 + i*0.25, 0.33), xytext=(0.4 + i*0.05, 0.5),
                       arrowprops=dict(arrowstyle='->', color='gray'))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title("FedMTL", fontweight='bold')

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

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

            fig, ax = plt.subplots(figsize=(10, 5))

            ax.plot(history["global"], 'k-', label="EU Globale", linewidth=3)
            colors = {'DE': '#000000', 'FR': '#0055A4', 'IT': '#009246'}
            for country, accs in history["per_country"].items():
                ax.plot(accs, '--', color=colors[country], label=country, linewidth=2, alpha=0.7)

            ax.set_xlabel("Round")
            ax.set_ylabel("Accuracy")
            ax.set_title("Hierarchical FL: Convergenza per Livello")
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0.4, 1.0)
            st.pyplot(fig)
            plt.close(fig)

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
        st.pyplot(fig)
        plt.close(fig)

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

    # Four sub-tabs for each interoperability component
    ehds_tabs = st.tabs([
        "🔗 HL7 FHIR",
        "📊 OMOP CDM",
        "📋 IHE Profiles",
        "🏛️ HDAB API"
    ])

    # HL7 FHIR Tab
    with ehds_tabs[0]:
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
    with ehds_tabs[1]:
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
    with ehds_tabs[2]:
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
    with ehds_tabs[3]:
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

        st.markdown("##### Esempio: Richiesta Permit FL")
        st.code("""
from fl_ehds.core import (
    FLEHDSPermitManager,
    HDABServiceSimulator,
    DatasetDescriptor,
    DataCategory,
    PurposeOfUse
)

# Setup HDAB client
hdab = HDABServiceSimulator("HDAB-IT", "Italy", auto_approve=True)
permit_mgr = FLEHDSPermitManager(
    hdab_client=hdab,
    organization_id="univ-roma",
    organization_name="Università di Roma",
    country="Italy"
)

# Definisci dataset richiesti
datasets = [
    DatasetDescriptor(
        dataset_id="opbg-ehr-2024",
        data_holder_id="opbg",
        data_holder_name="Ospedale Bambino Gesù",
        data_holder_country="Italy",
        data_categories=[DataCategory.EHR],
        population_description="Pazienti pediatrici diabetici"
    )
]

# Submit application
app_id = permit_mgr.request_fl_permit(
    research_question="Predictive model for T1DM complications",
    justification="Early detection improves outcomes...",
    methodology="Federated Learning with DP (ε=1.0)",
    datasets=datasets,
    fl_algorithm="FedAvg",
    fl_rounds=100,
    privacy_budget=1.0
)
        """, language="python")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("##### Condizioni Permit FL")
            conditions = {
                "Condizione": [
                    "Privacy Budget",
                    "Max Rounds",
                    "Min Clients",
                    "Aggregation Threshold",
                    "Output Review"
                ],
                "Descrizione": [
                    "Budget ε totale disponibile",
                    "Numero massimo round FL",
                    "Minimo client per round",
                    "k-anonymity sull'aggregato",
                    "Review output prima export"
                ],
                "Esempio": [
                    "ε ≤ 1.0",
                    "≤ 100 rounds",
                    "≥ 3 client",
                    "k ≥ 5",
                    "Required"
                ]
            }
            st.dataframe(pd.DataFrame(conditions), use_container_width=True)

        with col2:
            st.markdown("##### Cross-Border FL")
            st.markdown("""
            Per FL che coinvolge più paesi EU:

            1. **Lead HDAB**: Paese del richiedente
            2. **Participating HDABs**: Paesi con data holder
            3. **Coordinated Permit**: Un permit master + permit nazionali

            ```python
            coordinator = CrossBorderHDABCoordinator(
                lead_country="Italy"
            )
            coordinator.add_participating_hdab("Germany", hdab_de)
            coordinator.add_participating_hdab("France", hdab_fr)

            permits = coordinator.request_cross_border_permit(
                application, ["Italy", "Germany", "France"]
            )
            ```
            """)

        st.markdown("##### Opt-Out Registry (Art. 33)")
        st.markdown("""
        <div class="warning-box">
        <strong>Importante:</strong> I pazienti possono esercitare il diritto di opt-out
        per l'uso secondario dei loro dati. Il sistema FL deve verificare l'opt-out
        PRIMA di includere i dati nel training.
        </div>
        """, unsafe_allow_html=True)

        st.code("""
# Verifica opt-out prima del training
opted_out = hdab.check_opt_out(patient_pseudonyms)

# Rimuovi pazienti opt-out dal training set
training_patients = [p for p in all_patients if p not in opted_out]

# Log per compliance
permit_mgr.log_fl_round(
    permit_id=permit_id,
    round_number=1,
    client_ids=training_patients,
    epsilon_cost=0.01
)
        """, language="python")


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


def main():
    """Main application."""
    st.markdown('<div class="main-header">FL-EHDS Dashboard v4.0</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Federated Learning for European Health Data Space - Complete Advanced Edition</div>',
                unsafe_allow_html=True)

    config = create_config_panel()

    tabs = st.tabs([
        "🚀 Training",
        "🧮 Algoritmi",
        "🧠 Modelli",
        "📊 Vertical FL",
        "🛡️ Byzantine",
        "🔄 Continual",
        "🎯 Multi-Task",
        "🏛️ Hierarchical",
        "🇪🇺 EHDS",
        "⚙️ Infrastructure",
        "📚 Guida"
    ])

    with tabs[0]:
        render_training_tab(config)

    with tabs[1]:
        render_algorithms_tab()

    with tabs[2]:
        render_models_tab()

    with tabs[3]:
        render_vertical_fl_tab()

    with tabs[4]:
        render_byzantine_tab()

    with tabs[5]:
        render_continual_tab()

    with tabs[6]:
        render_multitask_tab()

    with tabs[7]:
        render_hierarchical_tab()

    with tabs[8]:
        render_ehds_tab()

    with tabs[9]:
        render_infrastructure_tab()

    with tabs[10]:
        render_guide_tab()

    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #888; font-size: 0.9rem;'>
        FL-EHDS Framework v4.0 | FLICS 2026 |
        9 Algoritmi FL | 11 Architetture Modello | 13 Moduli Avanzati |
        Vertical • Byzantine • Continual • Multi-Task • Hierarchical • EHDS • Infrastructure • Cross-Silo
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
FL-EHDS Interactive Dashboard v2.0

Enhanced features:
1. Node numbering instead of city names
2. Configurable DP noise parameters
3. Multiple heterogeneity types (label skew, feature skew, quantity skew)
4. Dynamic node participation simulation
5. Improved visualizations and charts
6. Real dataset support preparation

Author: Fabio Liberti
Usage: streamlit run app_v2.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches
from scipy import stats

# Page config
st.set_page_config(
    page_title="FL-EHDS Dashboard v2",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Enhanced styling
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
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 1.2rem;
        text-align: center;
        color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 10px 20px;
        background-color: white;
        border-radius: 8px;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f77b4;
        color: white;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: bold;
    }
    .config-section {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# HETEROGENEITY TYPES
# =============================================================================

class HeterogeneityType:
    """Enumeration of heterogeneity types."""
    LABEL_SKEW = "label_skew"           # Different label distributions
    FEATURE_SKEW = "feature_skew"       # Different feature distributions
    QUANTITY_SKEW = "quantity_skew"     # Different dataset sizes
    CONCEPT_DRIFT = "concept_drift"     # Different P(Y|X) relationships
    TEMPORAL_SKEW = "temporal_skew"     # Time-based data differences


class DataHeterogeneityGenerator:
    """Generates heterogeneous data distributions for FL simulation."""

    def __init__(self, num_nodes: int, random_seed: int = 42):
        self.num_nodes = num_nodes
        self.rng = np.random.RandomState(random_seed)

    def generate_label_skew(self,
                           total_samples: int,
                           num_classes: int = 2,
                           alpha: float = 0.5) -> Dict[int, Dict]:
        """
        Generate label distribution skew using Dirichlet distribution.

        Alpha controls skew:
        - alpha -> 0: extreme skew (each node has mostly one class)
        - alpha -> inf: uniform distribution (IID)
        """
        # Dirichlet distribution for label proportions
        label_distributions = self.rng.dirichlet(
            [alpha] * num_classes,
            size=self.num_nodes
        )

        samples_per_node = total_samples // self.num_nodes
        node_data = {}

        for node_id in range(self.num_nodes):
            n_samples = samples_per_node + self.rng.randint(-50, 50)
            label_probs = label_distributions[node_id]

            # Generate labels according to distribution
            labels = self.rng.choice(num_classes, size=n_samples, p=label_probs)

            # Generate features
            features = self._generate_features(n_samples, labels)

            node_data[node_id] = {
                "X": features,
                "y": labels,
                "n_samples": n_samples,
                "label_distribution": label_probs.tolist(),
                "heterogeneity_type": "label_skew"
            }

        return node_data

    def generate_feature_skew(self,
                             total_samples: int,
                             feature_dim: int = 5) -> Dict[int, Dict]:
        """
        Generate feature distribution skew.
        Each node has different feature means/variances.
        """
        samples_per_node = total_samples // self.num_nodes
        node_data = {}

        # Different feature statistics per node
        for node_id in range(self.num_nodes):
            n_samples = samples_per_node + self.rng.randint(-50, 50)

            # Node-specific feature parameters
            mean_shift = (node_id - self.num_nodes / 2) * 0.5
            var_scale = 1.0 + node_id * 0.2

            # Generate features with node-specific distribution
            features = self.rng.normal(
                loc=mean_shift,
                scale=var_scale,
                size=(n_samples, feature_dim)
            )

            # Generate labels based on features
            logits = features[:, 0] * 0.5 + features[:, 1] * 0.3
            probs = 1 / (1 + np.exp(-logits))
            labels = (self.rng.random(n_samples) < probs).astype(int)

            node_data[node_id] = {
                "X": features,
                "y": labels,
                "n_samples": n_samples,
                "feature_mean": float(mean_shift),
                "feature_var": float(var_scale),
                "heterogeneity_type": "feature_skew"
            }

        return node_data

    def generate_quantity_skew(self,
                              total_samples: int,
                              imbalance_ratio: float = 5.0) -> Dict[int, Dict]:
        """
        Generate quantity imbalance.
        imbalance_ratio: ratio between largest and smallest node.
        """
        # Generate imbalanced sample counts
        base_samples = total_samples / self.num_nodes

        # Exponential decay for sample counts
        ratios = np.exp(-np.linspace(0, np.log(imbalance_ratio), self.num_nodes))
        ratios = ratios / ratios.sum()

        sample_counts = (ratios * total_samples).astype(int)
        sample_counts = np.maximum(sample_counts, 50)  # Minimum 50 samples

        node_data = {}

        for node_id in range(self.num_nodes):
            n_samples = sample_counts[node_id]

            features = self.rng.randn(n_samples, 5)
            logits = features[:, 0] * 0.5 + features[:, 1] * 0.3
            probs = 1 / (1 + np.exp(-logits))
            labels = (self.rng.random(n_samples) < probs).astype(int)

            node_data[node_id] = {
                "X": features,
                "y": labels,
                "n_samples": n_samples,
                "quantity_ratio": float(n_samples / base_samples),
                "heterogeneity_type": "quantity_skew"
            }

        return node_data

    def generate_concept_drift(self,
                              total_samples: int) -> Dict[int, Dict]:
        """
        Generate concept drift: same features, different P(Y|X).
        Simulates different clinical practices across hospitals.
        """
        samples_per_node = total_samples // self.num_nodes
        node_data = {}

        for node_id in range(self.num_nodes):
            n_samples = samples_per_node

            # Same feature distribution
            features = self.rng.randn(n_samples, 5)

            # Different decision boundaries per node
            # Simulates different diagnostic thresholds
            threshold_shift = (node_id - self.num_nodes / 2) * 0.3

            logits = features[:, 0] * 0.5 + features[:, 1] * 0.3 + threshold_shift
            probs = 1 / (1 + np.exp(-logits))
            labels = (self.rng.random(n_samples) < probs).astype(int)

            node_data[node_id] = {
                "X": features,
                "y": labels,
                "n_samples": n_samples,
                "decision_threshold_shift": float(threshold_shift),
                "positive_rate": float(labels.mean()),
                "heterogeneity_type": "concept_drift"
            }

        return node_data

    def generate_combined_heterogeneity(self,
                                        total_samples: int,
                                        label_alpha: float = 0.5,
                                        feature_skew_strength: float = 0.5,
                                        quantity_imbalance: float = 3.0) -> Dict[int, Dict]:
        """
        Generate combined heterogeneity (most realistic scenario).
        """
        # Base sample distribution with quantity skew
        base_samples = total_samples / self.num_nodes
        ratios = np.exp(-np.linspace(0, np.log(quantity_imbalance), self.num_nodes))
        ratios = ratios / ratios.sum()
        sample_counts = (ratios * total_samples).astype(int)
        sample_counts = np.maximum(sample_counts, 50)

        # Label distributions with Dirichlet
        label_distributions = self.rng.dirichlet([label_alpha, label_alpha], size=self.num_nodes)

        node_data = {}

        for node_id in range(self.num_nodes):
            n_samples = sample_counts[node_id]

            # Feature skew
            mean_shift = (node_id - self.num_nodes / 2) * feature_skew_strength
            features = self.rng.normal(loc=mean_shift, scale=1.0, size=(n_samples, 5))

            # Label skew
            label_probs = label_distributions[node_id]
            labels = self.rng.choice(2, size=n_samples, p=label_probs)

            node_data[node_id] = {
                "X": features,
                "y": labels,
                "n_samples": n_samples,
                "label_distribution": label_probs.tolist(),
                "feature_mean": float(mean_shift),
                "quantity_ratio": float(n_samples / base_samples),
                "heterogeneity_type": "combined"
            }

        return node_data

    def _generate_features(self, n_samples: int, labels: np.ndarray) -> np.ndarray:
        """Generate features correlated with labels."""
        features = self.rng.randn(n_samples, 5)
        # Add label-correlated signal
        features[:, 0] += labels * 0.5
        features[:, 1] += labels * 0.3
        return features


# =============================================================================
# DYNAMIC PARTICIPATION
# =============================================================================

class DynamicParticipationSimulator:
    """Simulates realistic node participation patterns."""

    def __init__(self, num_nodes: int, random_seed: int = 42):
        self.num_nodes = num_nodes
        self.rng = np.random.RandomState(random_seed)

        # Node reliability profiles
        self.node_reliability = self.rng.uniform(0.7, 0.98, num_nodes)

        # Time-based availability (working hours simulation)
        self.availability_patterns = self._generate_availability_patterns()

    def _generate_availability_patterns(self) -> np.ndarray:
        """Generate time-based availability patterns for nodes."""
        # 24 hours x num_nodes
        patterns = np.ones((24, self.num_nodes))

        for node_id in range(self.num_nodes):
            # Random maintenance window (2-4 hours)
            maintenance_start = self.rng.randint(0, 20)
            maintenance_duration = self.rng.randint(2, 5)

            for h in range(maintenance_duration):
                patterns[(maintenance_start + h) % 24, node_id] = 0.3

            # Reduced availability during night (different time zones)
            timezone_offset = self.rng.randint(-5, 5)
            night_hours = [(h + timezone_offset) % 24 for h in range(2, 6)]
            for h in night_hours:
                patterns[h, node_id] *= 0.7

        return patterns

    def get_participating_nodes(self,
                                round_num: int,
                                base_participation_rate: float = 0.8,
                                mode: str = "realistic") -> Tuple[List[int], Dict]:
        """
        Determine which nodes participate in a given round.

        Modes:
        - "uniform": Simple random participation
        - "reliability": Based on node reliability profiles
        - "realistic": Includes time-based patterns and failures
        - "adversarial": Some nodes strategically drop out
        """
        participating = []
        participation_info = {}

        hour_of_day = round_num % 24

        for node_id in range(self.num_nodes):
            if mode == "uniform":
                prob = base_participation_rate

            elif mode == "reliability":
                prob = self.node_reliability[node_id] * base_participation_rate

            elif mode == "realistic":
                reliability = self.node_reliability[node_id]
                time_factor = self.availability_patterns[hour_of_day, node_id]

                # Random network issues
                network_factor = 1.0 if self.rng.random() > 0.05 else 0.2

                prob = reliability * time_factor * network_factor * base_participation_rate

            elif mode == "adversarial":
                # Some nodes drop when they have outlier gradients
                base_prob = self.node_reliability[node_id] * base_participation_rate
                # Simulate strategic dropout
                if node_id % 3 == 0 and round_num > 10:
                    prob = base_prob * 0.5  # Strategic nodes participate less
                else:
                    prob = base_prob

            else:
                prob = base_participation_rate

            participates = self.rng.random() < prob

            if participates:
                participating.append(node_id)

            participation_info[node_id] = {
                "probability": prob,
                "participated": participates,
                "reliability": self.node_reliability[node_id]
            }

        # Ensure at least one node participates
        if not participating:
            forced_node = self.rng.choice(self.num_nodes)
            participating.append(forced_node)
            participation_info[forced_node]["participated"] = True
            participation_info[forced_node]["forced"] = True

        return participating, participation_info


# =============================================================================
# FL SIMULATOR v2
# =============================================================================

class FLSimulatorV2:
    """Enhanced Federated Learning Simulator."""

    def __init__(self, config: Dict):
        self.config = config
        np.random.seed(config.get('random_seed', 42))

        self.num_nodes = config['num_nodes']
        self.node_names = [f"Node {i+1}" for i in range(self.num_nodes)]

        # Color palette for nodes
        self.colors = plt.cm.tab10(np.linspace(0, 1, min(self.num_nodes, 10)))

        # Initialize heterogeneity generator
        self.het_generator = DataHeterogeneityGenerator(
            self.num_nodes,
            config.get('random_seed', 42)
        )

        # Initialize participation simulator
        self.participation_sim = DynamicParticipationSimulator(
            self.num_nodes,
            config.get('random_seed', 42)
        )

        # Generate data based on heterogeneity type
        self._generate_data()
        self._init_model()

        # Training history
        self.history = []
        self.participation_history = []

    def _generate_data(self):
        """Generate data with specified heterogeneity type."""
        het_type = self.config.get('heterogeneity_type', 'combined')
        total_samples = self.config.get('total_samples', 2000)

        if het_type == 'label_skew':
            alpha = self.config.get('label_skew_alpha', 0.5)
            self.node_data = self.het_generator.generate_label_skew(
                total_samples, alpha=alpha
            )
        elif het_type == 'feature_skew':
            self.node_data = self.het_generator.generate_feature_skew(total_samples)
        elif het_type == 'quantity_skew':
            imbalance = self.config.get('quantity_imbalance', 5.0)
            self.node_data = self.het_generator.generate_quantity_skew(
                total_samples, imbalance_ratio=imbalance
            )
        elif het_type == 'concept_drift':
            self.node_data = self.het_generator.generate_concept_drift(total_samples)
        else:  # combined
            self.node_data = self.het_generator.generate_combined_heterogeneity(
                total_samples,
                label_alpha=self.config.get('label_skew_alpha', 0.5),
                feature_skew_strength=self.config.get('feature_skew_strength', 0.5),
                quantity_imbalance=self.config.get('quantity_imbalance', 3.0)
            )

        # Add bias term to features
        for node_id in self.node_data:
            X = self.node_data[node_id]["X"]
            X_norm = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
            self.node_data[node_id]["X"] = np.hstack([X_norm, np.ones((len(X), 1))])

    def _init_model(self):
        """Initialize model weights."""
        feature_dim = self.node_data[0]["X"].shape[1]
        self.weights = np.zeros(feature_dim)
        self.privacy_spent = 0.0
        self.total_bytes = 0

    def train_round(self, round_num: int) -> Dict:
        """Execute one FL training round with DP and dynamic participation."""
        config = self.config
        lr = config.get('learning_rate', 0.1)
        local_epochs = config.get('local_epochs', 3)
        algorithm = config.get('algorithm', 'FedAvg')
        mu = config.get('fedprox_mu', 0.1)

        # DP parameters
        use_dp = config.get('use_dp', True)
        epsilon = config.get('epsilon', 10.0)
        delta = config.get('delta', 1e-5)
        clip_norm = config.get('clip_norm', 1.0)
        noise_multiplier = config.get('noise_multiplier', 1.0)

        # Dynamic participation
        participation_mode = config.get('participation_mode', 'realistic')
        base_participation = config.get('participation_rate', 0.85)

        participating, participation_info = self.participation_sim.get_participating_nodes(
            round_num,
            base_participation_rate=base_participation,
            mode=participation_mode
        )

        gradients = []
        sample_counts = []
        node_metrics = {}
        gradient_norms = {}

        for node_id in range(self.num_nodes):
            data = self.node_data[node_id]

            if node_id in participating:
                local_weights = self.weights.copy()

                # Local training
                for _ in range(local_epochs):
                    batch_size = min(32, data["n_samples"])
                    indices = np.random.choice(data["n_samples"], batch_size, replace=False)
                    X_batch = data["X"][indices]
                    y_batch = data["y"][indices]

                    logits = X_batch @ local_weights
                    probs = 1 / (1 + np.exp(-np.clip(logits, -500, 500)))
                    grad = X_batch.T @ (probs - y_batch) / batch_size

                    # FedProx proximal term
                    if algorithm == 'FedProx':
                        grad += mu * (local_weights - self.weights)

                    local_weights -= lr * grad

                gradient = local_weights - self.weights

                # Record gradient norm before clipping
                grad_norm = np.linalg.norm(gradient)
                gradient_norms[node_id] = grad_norm

                # Gradient clipping for DP
                if grad_norm > clip_norm:
                    gradient = gradient * (clip_norm / grad_norm)

                gradients.append(gradient)
                sample_counts.append(data["n_samples"])

            # Evaluate local accuracy
            logits = data["X"] @ self.weights
            probs = 1 / (1 + np.exp(-np.clip(logits, -500, 500)))
            preds = (probs > 0.5).astype(int)
            accuracy = float(np.mean(preds == data["y"]))

            node_metrics[node_id] = {
                "accuracy": accuracy,
                "samples": data["n_samples"],
                "participating": node_id in participating,
                "positive_rate": float(data["y"].mean())
            }

        # Aggregate gradients
        if gradients:
            total_samples = sum(sample_counts)
            weighted_grad = sum(
                g * (n / total_samples) for g, n in zip(gradients, sample_counts)
            )

            # Add calibrated DP noise
            if use_dp:
                # Gaussian mechanism noise
                sigma = clip_norm * noise_multiplier * np.sqrt(2 * np.log(1.25 / delta)) / epsilon
                noise = np.random.normal(0, sigma, weighted_grad.shape)
                weighted_grad += noise

                # Track privacy budget (simplified composition)
                self.privacy_spent += epsilon / config.get('num_rounds', 50)

            self.weights += weighted_grad

        # Communication cost (bytes)
        bytes_per_param = 4  # float32
        self.total_bytes += len(participating) * len(self.weights) * bytes_per_param * 2

        # Global accuracy
        all_preds, all_labels = [], []
        for data in self.node_data.values():
            logits = data["X"] @ self.weights
            probs = 1 / (1 + np.exp(-np.clip(logits, -500, 500)))
            preds = (probs > 0.5).astype(int)
            all_preds.extend(preds)
            all_labels.extend(data["y"])

        global_accuracy = float(np.mean(np.array(all_preds) == np.array(all_labels)))

        # Store result
        result = {
            "round": round_num,
            "global_accuracy": global_accuracy,
            "node_metrics": node_metrics,
            "participating": participating,
            "participation_info": participation_info,
            "gradient_norms": gradient_norms,
            "privacy_spent": self.privacy_spent,
            "communication_kb": self.total_bytes / 1024,
            "dp_sigma": sigma if use_dp else 0
        }

        self.history.append(result)
        self.participation_history.append(participating)

        return result


# =============================================================================
# CONFIGURATION PANEL
# =============================================================================

def create_config_panel() -> Dict:
    """Create enhanced configuration panel."""
    st.sidebar.markdown("## ⚙️ FL Configuration")

    # Nodes configuration
    with st.sidebar.expander("🖥️ Nodes", expanded=True):
        num_nodes = st.slider("Number of Nodes", 2, 15, 5)
        st.caption("Nodes are labeled as Node 1, Node 2, ...")

    # Heterogeneity configuration
    with st.sidebar.expander("📊 Data Heterogeneity", expanded=True):
        heterogeneity_type = st.selectbox(
            "Heterogeneity Type",
            options=["combined", "label_skew", "feature_skew", "quantity_skew", "concept_drift"],
            format_func=lambda x: {
                "combined": "Combined (Realistic)",
                "label_skew": "Label Skew (Dirichlet)",
                "feature_skew": "Feature Skew",
                "quantity_skew": "Quantity Imbalance",
                "concept_drift": "Concept Drift"
            }.get(x, x)
        )

        label_skew_alpha = st.slider(
            "Label Skew α (Dirichlet)",
            0.1, 10.0, 0.5, 0.1,
            help="Lower = more skewed, Higher = more uniform"
        )

        if heterogeneity_type in ["feature_skew", "combined"]:
            feature_skew_strength = st.slider(
                "Feature Skew Strength",
                0.0, 2.0, 0.5, 0.1
            )
        else:
            feature_skew_strength = 0.5

        if heterogeneity_type in ["quantity_skew", "combined"]:
            quantity_imbalance = st.slider(
                "Quantity Imbalance Ratio",
                1.0, 10.0, 3.0, 0.5,
                help="Ratio between largest and smallest node"
            )
        else:
            quantity_imbalance = 3.0

        total_samples = st.number_input(
            "Total Samples", 500, 10000, 2000, 100
        )

    # Training configuration
    with st.sidebar.expander("🔄 Training", expanded=True):
        num_rounds = st.slider("Training Rounds", 10, 200, 50)
        local_epochs = st.slider("Local Epochs", 1, 10, 3)
        learning_rate = st.select_slider(
            "Learning Rate",
            options=[0.001, 0.01, 0.05, 0.1, 0.2, 0.5],
            value=0.1
        )

    # Algorithm configuration
    with st.sidebar.expander("🧮 Algorithm", expanded=True):
        algorithm = st.radio("FL Algorithm", ["FedAvg", "FedProx", "SCAFFOLD"])
        fedprox_mu = 0.0
        if algorithm == "FedProx":
            fedprox_mu = st.slider("FedProx μ", 0.01, 1.0, 0.1, 0.01)

    # Dynamic Participation
    with st.sidebar.expander("👥 Participation", expanded=True):
        participation_mode = st.selectbox(
            "Participation Mode",
            options=["uniform", "reliability", "realistic", "adversarial"],
            format_func=lambda x: {
                "uniform": "Uniform Random",
                "reliability": "Reliability-based",
                "realistic": "Realistic (Time + Network)",
                "adversarial": "Adversarial Dropout"
            }.get(x, x)
        )
        participation_rate = st.slider(
            "Base Participation Rate",
            0.5, 1.0, 0.85, 0.05
        )

    # Differential Privacy configuration
    with st.sidebar.expander("🔒 Differential Privacy", expanded=True):
        use_dp = st.checkbox("Enable DP", value=True)

        col1, col2 = st.columns(2)
        with col1:
            epsilon = st.number_input(
                "Privacy Budget ε",
                0.1, 100.0, 10.0, 0.1,
                disabled=not use_dp,
                help="Lower = stronger privacy"
            )
        with col2:
            delta = st.select_slider(
                "Failure Prob δ",
                options=[1e-3, 1e-4, 1e-5, 1e-6, 1e-7],
                value=1e-5,
                disabled=not use_dp
            )

        clip_norm = st.slider(
            "Gradient Clip Norm C",
            0.1, 5.0, 1.0, 0.1,
            disabled=not use_dp,
            help="Sensitivity bound for DP"
        )

        noise_multiplier = st.slider(
            "Noise Multiplier",
            0.1, 3.0, 1.0, 0.1,
            disabled=not use_dp,
            help="Scales the DP noise (σ = C × multiplier × ...)"
        )

        if use_dp:
            # Show computed sigma
            sigma = clip_norm * noise_multiplier * np.sqrt(2 * np.log(1.25 / delta)) / epsilon
            st.info(f"Computed σ = {sigma:.4f}")

    # Reproducibility
    with st.sidebar.expander("🎲 Reproducibility", expanded=False):
        random_seed = st.number_input("Random Seed", 0, 9999, 42)

    return {
        "num_nodes": num_nodes,
        "heterogeneity_type": heterogeneity_type,
        "label_skew_alpha": label_skew_alpha,
        "feature_skew_strength": feature_skew_strength,
        "quantity_imbalance": quantity_imbalance,
        "total_samples": total_samples,
        "num_rounds": num_rounds,
        "local_epochs": local_epochs,
        "learning_rate": learning_rate,
        "algorithm": algorithm,
        "fedprox_mu": fedprox_mu,
        "participation_mode": participation_mode,
        "participation_rate": participation_rate,
        "use_dp": use_dp,
        "epsilon": epsilon,
        "delta": delta,
        "clip_norm": clip_norm,
        "noise_multiplier": noise_multiplier,
        "random_seed": random_seed
    }


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_heterogeneity_analysis(simulator: FLSimulatorV2):
    """Create comprehensive heterogeneity visualization."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    num_nodes = simulator.num_nodes
    node_names = simulator.node_names
    colors = [simulator.colors[i] for i in range(num_nodes)]

    # 1. Sample count distribution
    ax1 = axes[0, 0]
    samples = [simulator.node_data[i]["n_samples"] for i in range(num_nodes)]
    bars = ax1.bar(node_names, samples, color=colors)
    ax1.axhline(np.mean(samples), color='red', linestyle='--', label='Mean')
    ax1.set_ylabel("Number of Samples")
    ax1.set_title("Quantity Distribution")
    ax1.tick_params(axis='x', rotation=45)
    ax1.legend()

    # 2. Label distribution
    ax2 = axes[0, 1]
    pos_rates = [simulator.node_data[i]["y"].mean() for i in range(num_nodes)]
    bars = ax2.bar(node_names, pos_rates, color=colors)
    ax2.axhline(np.mean(pos_rates), color='red', linestyle='--', label='Mean')
    ax2.set_ylabel("Positive Class Rate")
    ax2.set_title("Label Distribution (Non-IID)")
    ax2.tick_params(axis='x', rotation=45)
    ax2.set_ylim(0, 1)
    ax2.legend()

    # 3. Feature distribution (first feature)
    ax3 = axes[0, 2]
    for i in range(num_nodes):
        data = simulator.node_data[i]["X"][:, 0]
        ax3.hist(data, bins=20, alpha=0.5, label=node_names[i], color=colors[i])
    ax3.set_xlabel("Feature 1 Value")
    ax3.set_ylabel("Frequency")
    ax3.set_title("Feature Distribution")
    if num_nodes <= 5:
        ax3.legend()

    # 4. Label distribution heatmap
    ax4 = axes[1, 0]
    label_matrix = np.array([
        [1 - simulator.node_data[i]["y"].mean(), simulator.node_data[i]["y"].mean()]
        for i in range(num_nodes)
    ])
    im = ax4.imshow(label_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax4.set_xticks([0, 1])
    ax4.set_xticklabels(['Class 0', 'Class 1'])
    ax4.set_yticks(range(num_nodes))
    ax4.set_yticklabels(node_names)
    ax4.set_title("Label Distribution Heatmap")
    plt.colorbar(im, ax=ax4)

    # 5. KL Divergence from global
    ax5 = axes[1, 1]
    global_pos = np.mean(pos_rates)
    kl_divs = []
    for i in range(num_nodes):
        local_pos = pos_rates[i]
        eps = 1e-10
        kl = local_pos * np.log((local_pos + eps) / (global_pos + eps)) + \
             (1 - local_pos) * np.log((1 - local_pos + eps) / (1 - global_pos + eps))
        kl_divs.append(abs(kl))

    bars = ax5.bar(node_names, kl_divs, color=colors)
    ax5.set_ylabel("KL Divergence")
    ax5.set_title("Divergence from Global Distribution")
    ax5.tick_params(axis='x', rotation=45)

    # 6. Summary statistics
    ax6 = axes[1, 2]
    ax6.axis('off')

    summary_text = f"""
    Heterogeneity Summary
    ━━━━━━━━━━━━━━━━━━━━━

    Nodes: {num_nodes}
    Total Samples: {sum(samples)}

    Quantity Skew:
      • Max/Min ratio: {max(samples)/max(min(samples), 1):.2f}
      • Std Dev: {np.std(samples):.1f}

    Label Skew:
      • Pos Rate Range: [{min(pos_rates):.2%}, {max(pos_rates):.2%}]
      • Max KL Divergence: {max(kl_divs):.4f}
      • Mean KL Divergence: {np.mean(kl_divs):.4f}

    Heterogeneity Type: {simulator.config.get('heterogeneity_type', 'combined')}
    """

    ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes,
             fontfamily='monospace', fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    return fig


def plot_participation_analysis(history: List[Dict], num_nodes: int):
    """Create participation analysis visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    rounds = [h["round"] for h in history]

    # 1. Participation heatmap
    ax1 = axes[0, 0]
    participation_matrix = np.zeros((num_nodes, len(history)))
    for r, h in enumerate(history):
        for node_id in h["participating"]:
            participation_matrix[node_id, r] = 1

    cmap = LinearSegmentedColormap.from_list("", ["#ffcccc", "#28a745"])
    im = ax1.imshow(participation_matrix, cmap=cmap, aspect='auto')
    ax1.set_yticks(range(num_nodes))
    ax1.set_yticklabels([f"Node {i+1}" for i in range(num_nodes)])
    ax1.set_xlabel("Round")
    ax1.set_title("Node Participation Matrix")

    # 2. Participation rate over time
    ax2 = axes[0, 1]
    participation_rates = [len(h["participating"]) / num_nodes for h in history]
    ax2.plot(rounds, participation_rates, 'b-', linewidth=2)
    ax2.fill_between(rounds, participation_rates, alpha=0.3)
    ax2.axhline(np.mean(participation_rates), color='red', linestyle='--',
                label=f'Mean: {np.mean(participation_rates):.1%}')
    ax2.set_xlabel("Round")
    ax2.set_ylabel("Participation Rate")
    ax2.set_title("Participation Rate Over Time")
    ax2.set_ylim(0, 1.1)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Per-node participation rate
    ax3 = axes[1, 0]
    node_participation_rates = participation_matrix.sum(axis=1) / len(history)
    colors = plt.cm.RdYlGn(node_participation_rates)
    bars = ax3.bar([f"Node {i+1}" for i in range(num_nodes)],
                   node_participation_rates, color=colors)
    ax3.axhline(np.mean(node_participation_rates), color='blue', linestyle='--',
                label='Mean')
    ax3.set_ylabel("Participation Rate")
    ax3.set_title("Per-Node Participation Rate")
    ax3.tick_params(axis='x', rotation=45)
    ax3.set_ylim(0, 1.1)
    ax3.legend()

    # 4. Participation statistics
    ax4 = axes[1, 1]
    ax4.axis('off')

    # Compute statistics
    consecutive_drops = []
    for node_id in range(num_nodes):
        drops = 0
        max_drops = 0
        for r in range(len(history)):
            if participation_matrix[node_id, r] == 0:
                drops += 1
                max_drops = max(max_drops, drops)
            else:
                drops = 0
        consecutive_drops.append(max_drops)

    stats_text = f"""
    Participation Statistics
    ━━━━━━━━━━━━━━━━━━━━━━━━

    Overall:
      • Mean participation: {np.mean(participation_rates):.1%}
      • Min participation: {min(participation_rates):.1%}
      • Rounds with full participation: {sum(1 for p in participation_rates if p == 1.0)}

    Per-Node:
      • Most reliable: Node {np.argmax(node_participation_rates)+1} ({max(node_participation_rates):.1%})
      • Least reliable: Node {np.argmin(node_participation_rates)+1} ({min(node_participation_rates):.1%})
      • Max consecutive dropouts: {max(consecutive_drops)} rounds

    Stability:
      • Std Dev: {np.std(participation_rates):.3f}
      • Rounds below 70%: {sum(1 for p in participation_rates if p < 0.7)}
    """

    ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes,
             fontfamily='monospace', fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    plt.tight_layout()
    return fig


def plot_training_results(history: List[Dict], num_nodes: int, config: Dict):
    """Create comprehensive training results visualization."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    rounds = [h["round"] for h in history]

    # 1. Global accuracy
    ax1 = axes[0, 0]
    accuracies = [h["global_accuracy"] for h in history]
    ax1.plot(rounds, accuracies, 'b-', linewidth=2, label='Global Accuracy')
    ax1.fill_between(rounds, accuracies, alpha=0.3)
    ax1.set_xlabel("Round")
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Global Model Accuracy")
    ax1.set_ylim(0.4, 0.75)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # 2. Per-node accuracy
    ax2 = axes[0, 1]
    colors = plt.cm.tab10(np.linspace(0, 1, num_nodes))
    for node_id in range(num_nodes):
        node_accs = [h["node_metrics"][node_id]["accuracy"] for h in history]
        ax2.plot(rounds, node_accs, '-', color=colors[node_id],
                 linewidth=1.5, label=f'Node {node_id+1}', alpha=0.7)
    ax2.set_xlabel("Round")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Per-Node Accuracy")
    ax2.set_ylim(0.3, 0.8)
    ax2.grid(True, alpha=0.3)
    if num_nodes <= 7:
        ax2.legend(loc='lower right', fontsize=8)

    # 3. Privacy budget
    ax3 = axes[0, 2]
    privacy = [h["privacy_spent"] for h in history]
    ax3.plot(rounds, privacy, 'r-', linewidth=2)
    ax3.axhline(config['epsilon'], color='orange', linestyle='--',
                label=f'Budget ε={config["epsilon"]}')
    ax3.fill_between(rounds, privacy, alpha=0.3, color='red')
    ax3.set_xlabel("Round")
    ax3.set_ylabel("ε spent")
    ax3.set_title("Privacy Budget Consumption")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Gradient norms
    ax4 = axes[1, 0]
    for node_id in range(num_nodes):
        norms = []
        for h in history:
            if node_id in h.get("gradient_norms", {}):
                norms.append(h["gradient_norms"][node_id])
            else:
                norms.append(np.nan)
        ax4.plot(rounds, norms, '-', color=colors[node_id], alpha=0.7, linewidth=1.5)
    ax4.axhline(config['clip_norm'], color='red', linestyle='--',
                label=f'Clip norm={config["clip_norm"]}')
    ax4.set_xlabel("Round")
    ax4.set_ylabel("Gradient L2 Norm")
    ax4.set_title("Gradient Norms (before clipping)")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. Communication cost
    ax5 = axes[1, 1]
    comm_costs = [h["communication_kb"] for h in history]
    ax5.plot(rounds, comm_costs, 'g-', linewidth=2)
    ax5.fill_between(rounds, comm_costs, alpha=0.3, color='green')
    ax5.set_xlabel("Round")
    ax5.set_ylabel("Cumulative KB")
    ax5.set_title("Communication Cost")
    ax5.grid(True, alpha=0.3)

    # 6. Final summary
    ax6 = axes[1, 2]
    ax6.axis('off')

    final_acc = accuracies[-1]
    final_privacy = privacy[-1]

    summary_text = f"""
    Training Summary
    ━━━━━━━━━━━━━━━━

    Configuration:
      • Algorithm: {config['algorithm']}
      • Nodes: {num_nodes}
      • Rounds: {config['num_rounds']}
      • DP Enabled: {config['use_dp']}

    Results:
      • Final Accuracy: {final_acc:.2%}
      • Best Accuracy: {max(accuracies):.2%} (round {np.argmax(accuracies)+1})
      • Privacy Spent: ε = {final_privacy:.2f}
      • Total Comm: {comm_costs[-1]:.1f} KB

    Convergence:
      • Rounds to 55%: {next((i+1 for i, a in enumerate(accuracies) if a > 0.55), 'N/A')}
      • Final 10 rounds std: {np.std(accuracies[-10:]):.4f}
    """

    ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes,
             fontfamily='monospace', fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

    plt.tight_layout()
    return fig


# =============================================================================
# TAB RENDERERS
# =============================================================================

def render_training_tab(config: Dict):
    """Render enhanced training tab."""
    st.markdown("### 🚀 Federated Learning Training")

    # Configuration summary
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Nodes", config['num_nodes'])
    with col2:
        st.metric("Rounds", config['num_rounds'])
    with col3:
        st.metric("Algorithm", config['algorithm'])
    with col4:
        dp_status = f"ε={config['epsilon']}" if config['use_dp'] else "Off"
        st.metric("DP", dp_status)

    st.markdown("---")

    col1, col2 = st.columns([2, 1])

    with col1:
        if st.button("▶️ Start Training", type="primary", use_container_width=True):
            run_training_v2(config)

    with col2:
        st.markdown("**Heterogeneity:** " + config['heterogeneity_type'].replace('_', ' ').title())
        st.markdown(f"**Participation:** {config['participation_mode']}")


def run_training_v2(config: Dict):
    """Run FL training with enhanced visualization."""
    simulator = FLSimulatorV2(config)

    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Create tabs for different visualizations
    viz_tabs = st.tabs(["📈 Training Progress", "👥 Participation", "📊 Heterogeneity"])

    with viz_tabs[0]:
        col1, col2 = st.columns(2)
        with col1:
            accuracy_chart = st.empty()
        with col2:
            metrics_display = st.empty()

    with viz_tabs[1]:
        participation_chart = st.empty()

    with viz_tabs[2]:
        heterogeneity_chart = st.empty()
        # Show initial heterogeneity
        het_fig = plot_heterogeneity_analysis(simulator)
        heterogeneity_chart.pyplot(het_fig)
        plt.close(het_fig)

    # Training loop
    for round_num in range(1, config['num_rounds'] + 1):
        result = simulator.train_round(round_num)

        # Update progress
        progress = round_num / config['num_rounds']
        progress_bar.progress(progress)
        status_text.markdown(
            f"**Round {round_num}/{config['num_rounds']}** | "
            f"Accuracy: {result['global_accuracy']:.2%} | "
            f"Participants: {len(result['participating'])}/{config['num_nodes']} | "
            f"ε spent: {result['privacy_spent']:.2f}"
        )

        # Update charts periodically
        if round_num % 5 == 0 or round_num == config['num_rounds']:
            # Training results
            with viz_tabs[0]:
                train_fig = plot_training_results(simulator.history, config['num_nodes'], config)
                accuracy_chart.pyplot(train_fig)
                plt.close(train_fig)

            # Participation
            with viz_tabs[1]:
                part_fig = plot_participation_analysis(simulator.history, config['num_nodes'])
                participation_chart.pyplot(part_fig)
                plt.close(part_fig)

        time.sleep(0.02)

    # Final status
    final_acc = simulator.history[-1]['global_accuracy']
    status_text.success(
        f"✅ Training Complete! | "
        f"Final Accuracy: {final_acc:.2%} | "
        f"Privacy: ε = {simulator.privacy_spent:.2f}"
    )

    # Download results
    results_json = json.dumps({
        "config": {k: str(v) if not isinstance(v, (int, float, bool, str, list)) else v
                   for k, v in config.items()},
        "history": [
            {
                "round": h["round"],
                "global_accuracy": h["global_accuracy"],
                "privacy_spent": h["privacy_spent"],
                "participating": h["participating"]
            }
            for h in simulator.history
        ]
    }, indent=2)

    st.download_button(
        "📥 Download Results (JSON)",
        results_json,
        file_name=f"fl_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )


def render_heterogeneity_tab(config: Dict):
    """Render heterogeneity analysis tab."""
    st.markdown("### 📊 Data Heterogeneity Analysis")

    col1, col2 = st.columns([1, 3])

    with col1:
        st.markdown("#### Configuration")
        st.markdown(f"**Type:** {config['heterogeneity_type'].replace('_', ' ').title()}")
        st.markdown(f"**Nodes:** {config['num_nodes']}")
        st.markdown(f"**Samples:** {config['total_samples']}")

        if st.button("🔄 Generate & Analyze", type="primary"):
            with st.spinner("Generating data..."):
                simulator = FLSimulatorV2(config)

                with col2:
                    fig = plot_heterogeneity_analysis(simulator)
                    st.pyplot(fig)
                    plt.close(fig)

    with col2:
        st.info("""
        **Heterogeneity Types Explained:**

        - **Label Skew**: Different nodes have different class proportions (controlled by Dirichlet α)
        - **Feature Skew**: Feature distributions vary across nodes (different patient demographics)
        - **Quantity Skew**: Imbalanced dataset sizes (large vs small hospitals)
        - **Concept Drift**: Same features, different label relationships (different clinical practices)
        - **Combined**: Realistic scenario with all types mixed
        """)


def render_dp_analysis_tab(config: Dict):
    """Render DP analysis tab."""
    st.markdown("### 🔒 Differential Privacy Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Current Configuration")

        if config['use_dp']:
            sigma = config['clip_norm'] * config['noise_multiplier'] * \
                    np.sqrt(2 * np.log(1.25 / config['delta'])) / config['epsilon']

            st.metric("Privacy Budget ε", f"{config['epsilon']:.1f}")
            st.metric("Failure Probability δ", f"{config['delta']:.0e}")
            st.metric("Gradient Clip Norm C", f"{config['clip_norm']:.2f}")
            st.metric("Noise Multiplier", f"{config['noise_multiplier']:.2f}")
            st.metric("Computed σ", f"{sigma:.4f}")

            # Privacy-utility tradeoff visualization
            st.markdown("#### Privacy-Utility Tradeoff")

            epsilons = np.linspace(0.5, 50, 50)
            sigmas = config['clip_norm'] * config['noise_multiplier'] * \
                     np.sqrt(2 * np.log(1.25 / config['delta'])) / epsilons

            # Simulated accuracy impact
            accuracy_impact = 0.6 - 0.1 * np.exp(-epsilons / 5)

            fig, ax1 = plt.subplots(figsize=(8, 5))

            ax1.plot(epsilons, sigmas, 'b-', linewidth=2, label='Noise σ')
            ax1.set_xlabel('Privacy Budget ε')
            ax1.set_ylabel('Noise Scale σ', color='b')
            ax1.axvline(config['epsilon'], color='red', linestyle='--', alpha=0.7)

            ax2 = ax1.twinx()
            ax2.plot(epsilons, accuracy_impact * 100, 'g-', linewidth=2, label='Est. Accuracy')
            ax2.set_ylabel('Estimated Accuracy (%)', color='g')

            ax1.legend(loc='upper right')
            ax2.legend(loc='lower right')
            plt.title('Privacy-Utility Tradeoff')

            st.pyplot(fig)
            plt.close(fig)
        else:
            st.warning("DP is disabled. Enable it in the sidebar to see analysis.")

    with col2:
        st.markdown("#### DP Explanation")
        st.markdown("""
        **Gaussian Mechanism:**

        The noise scale σ is computed as:
        ```
        σ = C × multiplier × √(2 ln(1.25/δ)) / ε
        ```

        Where:
        - **C** (clip norm): Bounds the sensitivity of gradients
        - **ε** (epsilon): Privacy loss per query (lower = more private)
        - **δ** (delta): Probability of privacy failure
        - **multiplier**: Additional noise scaling factor

        **Recommendations:**

        | ε Value | Privacy Level | Use Case |
        |---------|--------------|----------|
        | 0.1-1 | Very Strong | Highly sensitive data |
        | 1-10 | Strong | Clinical research |
        | 10-50 | Moderate | General healthcare |
        | >50 | Weak | Non-sensitive data |
        """)


def render_participation_tab(config: Dict):
    """Render participation analysis tab."""
    st.markdown("### 👥 Dynamic Participation Analysis")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("#### Participation Modes")

        mode_descriptions = {
            "uniform": "Simple random participation with fixed probability",
            "reliability": "Nodes have different reliability scores",
            "realistic": "Includes time-of-day patterns and network failures",
            "adversarial": "Some nodes strategically drop out"
        }

        st.info(f"**Current Mode:** {config['participation_mode']}\n\n{mode_descriptions[config['participation_mode']]}")

        st.markdown("#### Simulate Participation")

        sim_rounds = st.slider("Simulation Rounds", 10, 100, 50)

        if st.button("🔄 Run Simulation"):
            participation_sim = DynamicParticipationSimulator(
                config['num_nodes'],
                config['random_seed']
            )

            history = []
            for r in range(sim_rounds):
                participating, info = participation_sim.get_participating_nodes(
                    r,
                    base_participation_rate=config['participation_rate'],
                    mode=config['participation_mode']
                )
                history.append({
                    "round": r + 1,
                    "participating": participating,
                    "participation_info": info
                })

            with col2:
                fig = plot_participation_analysis(history, config['num_nodes'])
                st.pyplot(fig)
                plt.close(fig)

    with col2:
        st.markdown("""
        **Why Dynamic Participation Matters:**

        In real EHDS deployments, hospitals may not always be available:

        - **Network issues**: Connectivity problems
        - **Maintenance windows**: Scheduled downtime
        - **Time zones**: Different working hours across EU
        - **Resource constraints**: Competing computational needs
        - **Strategic behavior**: Nodes may drop when they have outlier data

        The FL algorithm must be robust to these variations while still converging.
        """)


def main():
    """Main dashboard application."""
    # Header
    st.markdown('<div class="main-header">🏥 FL-EHDS Dashboard v2.0</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Federated Learning for European Health Data Space - Enhanced</div>',
                unsafe_allow_html=True)

    # Configuration panel
    config = create_config_panel()

    # Main content tabs
    tabs = st.tabs([
        "🚀 Training",
        "📊 Heterogeneity",
        "👥 Participation",
        "🔒 Privacy (DP)",
        "📁 Results"
    ])

    with tabs[0]:
        render_training_tab(config)

    with tabs[1]:
        render_heterogeneity_tab(config)

    with tabs[2]:
        render_participation_tab(config)

    with tabs[3]:
        render_dp_analysis_tab(config)

    with tabs[4]:
        st.markdown("### 📁 Experimental Results")
        st.info("Run training to generate results, then download them using the button.")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #888;'>
        FL-EHDS Framework v2.0 | FLICS 2026 |
        Nodes labeled as Node 1, Node 2, ... |
        Enhanced DP Configuration | Multiple Heterogeneity Types
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

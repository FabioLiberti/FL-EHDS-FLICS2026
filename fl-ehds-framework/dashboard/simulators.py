"""FL simulator classes for the Streamlit dashboard."""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any

# Plotly qualitative Tab10 palette (replaces plt.cm.tab10)
_TAB10_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]

from dashboard.constants import ALGORITHMS, MODELS, EHDS_TASKS

__all__ = [
    "FLSimulatorV4",
    "VerticalFLSimulator",
    "ByzantineSimulator",
    "ContinualFLSimulator",
    "MultiTaskFLSimulator",
    "HierarchicalFLSimulator",
]


class FLSimulatorV4:
    """Enhanced FL Simulator with all algorithms support."""

    def __init__(self, config: Dict):
        self.config = config
        np.random.seed(config.get('random_seed', 42))

        self.num_nodes = config['num_nodes']
        self.node_names = [f"Node {i+1}" for i in range(self.num_nodes)]
        n = min(self.num_nodes, 10)
        self.colors = [_TAB10_COLORS[i % len(_TAB10_COLORS)] for i in range(n)]

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

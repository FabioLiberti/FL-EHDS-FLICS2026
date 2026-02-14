#!/usr/bin/env python3
"""
FL-EHDS Hierarchical Federated Learning

Implements multi-level hierarchical FL for EHDS where:
- Level 0: Individual hospitals (clients)
- Level 1: Regional aggregators (HDAB regions)
- Level 2: National aggregators (Member States)
- Level 3: EU-level aggregator (HealthData@EU)

Benefits:
1. Reduced communication to central server
2. Regional model specialization
3. Compliance with data sovereignty
4. Fault tolerance through redundancy

Approaches:
1. HierFedAvg - Standard hierarchical averaging
2. Clustered FL - Automatic client clustering
3. Multi-tier Aggregation - Different algorithms per tier
4. FedTree - Tree-structured federation

Author: Fabio Liberti
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from collections import defaultdict
from copy import deepcopy
from enum import Enum


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class TierLevel(Enum):
    """Federation hierarchy levels."""
    CLIENT = 0       # Individual hospitals
    REGIONAL = 1     # Regional HDAB
    NATIONAL = 2     # Member State level
    SUPRANATIONAL = 3  # EU level (HealthData@EU)


@dataclass
class FederationNode:
    """Node in the federation hierarchy."""
    node_id: str
    tier: TierLevel
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    # Geographic/organizational info
    region: Optional[str] = None
    country: Optional[str] = None
    # Model state
    local_model: Optional[Dict[str, np.ndarray]] = None
    # Metadata
    sample_count: int = 0
    last_update_round: int = -1


@dataclass
class HierarchicalConfig:
    """Configuration for hierarchical FL."""
    # Structure
    num_tiers: int = 3  # Client -> Regional -> National
    regional_rounds: int = 5  # Local rounds before regional sync
    national_rounds: int = 3  # Regional rounds before national sync
    # Aggregation
    aggregation_algorithm: str = "fedavg"  # Per-tier algorithm
    use_weighted_aggregation: bool = True
    # Communication
    compress_between_tiers: bool = False
    compression_ratio: float = 0.1
    # Privacy
    add_tier_noise: bool = False
    noise_multiplier: float = 0.01


@dataclass
class HierarchicalUpdate:
    """Update packet in hierarchical FL."""
    source_id: str
    source_tier: TierLevel
    target_id: str
    target_tier: TierLevel
    gradient: Dict[str, np.ndarray]
    sample_count: int
    round_number: int


# =============================================================================
# FEDERATION TOPOLOGY
# =============================================================================

class FederationTopology:
    """
    Manages the hierarchical topology of the federation.

    Example EHDS structure:
    - EU (1 node)
      - Germany (1 node)
        - Bavaria Region (1 node)
          - Munich Hospital (client)
          - Augsburg Hospital (client)
        - Berlin Region (1 node)
          - Charité (client)
      - France (1 node)
        - Île-de-France (1 node)
          - AP-HP (client)
      ...
    """

    def __init__(self):
        self.nodes: Dict[str, FederationNode] = {}
        self.root_id: Optional[str] = None

    def add_node(self, node: FederationNode) -> None:
        """Add a node to the topology."""
        self.nodes[node.node_id] = node

        if node.tier == TierLevel.SUPRANATIONAL:
            self.root_id = node.node_id

    def build_ehds_topology(self,
                           countries: Dict[str, List[str]],
                           regions_per_country: Dict[str, Dict[str, List[str]]]) -> None:
        """
        Build EHDS-like hierarchy.

        Args:
            countries: {country_code: [region_ids]}
            regions_per_country: {country_code: {region_id: [client_ids]}}
        """
        # Create EU-level root
        eu_node = FederationNode(
            node_id="EU",
            tier=TierLevel.SUPRANATIONAL,
            children_ids=list(countries.keys())
        )
        self.add_node(eu_node)

        # Create country-level nodes
        for country_code, region_ids in countries.items():
            country_node = FederationNode(
                node_id=country_code,
                tier=TierLevel.NATIONAL,
                parent_id="EU",
                children_ids=region_ids,
                country=country_code
            )
            self.add_node(country_node)

            # Create regional nodes
            for region_id in region_ids:
                client_ids = regions_per_country.get(country_code, {}).get(region_id, [])

                region_node = FederationNode(
                    node_id=region_id,
                    tier=TierLevel.REGIONAL,
                    parent_id=country_code,
                    children_ids=client_ids,
                    region=region_id,
                    country=country_code
                )
                self.add_node(region_node)

                # Create client nodes
                for client_id in client_ids:
                    client_node = FederationNode(
                        node_id=client_id,
                        tier=TierLevel.CLIENT,
                        parent_id=region_id,
                        region=region_id,
                        country=country_code
                    )
                    self.add_node(client_node)

    def get_children(self, node_id: str) -> List[FederationNode]:
        """Get child nodes."""
        node = self.nodes.get(node_id)
        if node is None:
            return []
        return [self.nodes[cid] for cid in node.children_ids if cid in self.nodes]

    def get_parent(self, node_id: str) -> Optional[FederationNode]:
        """Get parent node."""
        node = self.nodes.get(node_id)
        if node is None or node.parent_id is None:
            return None
        return self.nodes.get(node.parent_id)

    def get_nodes_at_tier(self, tier: TierLevel) -> List[FederationNode]:
        """Get all nodes at a specific tier."""
        return [n for n in self.nodes.values() if n.tier == tier]

    def get_clients(self) -> List[FederationNode]:
        """Get all client nodes."""
        return self.get_nodes_at_tier(TierLevel.CLIENT)

    def print_topology(self) -> None:
        """Print the topology tree."""
        def print_subtree(node_id: str, indent: int = 0):
            node = self.nodes.get(node_id)
            if node:
                prefix = "  " * indent
                tier_name = node.tier.name
                print(f"{prefix}[{tier_name}] {node.node_id}")
                for child_id in node.children_ids:
                    print_subtree(child_id, indent + 1)

        if self.root_id:
            print_subtree(self.root_id)


# =============================================================================
# HIERARCHICAL FEDAVG
# =============================================================================

class HierFedAvg:
    """
    Hierarchical Federated Averaging.

    Aggregation happens at each tier before propagating up.

    Round structure:
    1. Clients train locally for K rounds
    2. Regional aggregators aggregate client updates
    3. National aggregators aggregate regional updates
    4. EU aggregator produces global model
    5. Model propagates back down the hierarchy
    """

    def __init__(self,
                 topology: FederationTopology,
                 config: HierarchicalConfig):
        self.topology = topology
        self.config = config

        # Global model
        self.global_model: Dict[str, np.ndarray] = {}

        # Track rounds per tier
        self.tier_rounds: Dict[TierLevel, int] = {
            TierLevel.CLIENT: 0,
            TierLevel.REGIONAL: 0,
            TierLevel.NATIONAL: 0,
            TierLevel.SUPRANATIONAL: 0
        }

    def initialize(self, model_template: Dict[str, np.ndarray]) -> None:
        """Initialize model at all nodes."""
        self.global_model = {k: v.copy() for k, v in model_template.items()}

        # Distribute to all nodes
        for node in self.topology.nodes.values():
            node.local_model = {k: v.copy() for k, v in self.global_model.items()}

    def client_train(self,
                    client_id: str,
                    data: Tuple[np.ndarray, np.ndarray],
                    epochs: int = 1,
                    lr: float = 0.01) -> Dict[str, np.ndarray]:
        """
        Client-level local training.

        Returns:
            Gradient update to send to regional aggregator.
        """
        node = self.topology.nodes.get(client_id)
        if node is None or node.tier != TierLevel.CLIENT:
            raise ValueError(f"Invalid client: {client_id}")

        X, y = data
        model = deepcopy(node.local_model)
        initial_model = deepcopy(model)

        for _ in range(epochs):
            if 'weights' in model:
                batch_size = min(32, len(X))
                indices = np.random.choice(len(X), batch_size, replace=False)
                X_batch, y_batch = X[indices], y[indices]

                logits = X_batch @ model['weights']
                if 'bias' in model:
                    logits += model['bias']

                probs = 1 / (1 + np.exp(-np.clip(logits, -500, 500)))
                error = probs.flatten() - y_batch.flatten()
                grad = X_batch.T @ error / batch_size

                model['weights'] -= lr * grad
                if 'bias' in model:
                    model['bias'] -= lr * np.mean(error)

        # Update node's local model
        node.local_model = model
        node.sample_count = len(X)
        node.last_update_round = self.tier_rounds[TierLevel.CLIENT]

        # Compute gradient delta
        gradient = {k: model[k] - initial_model[k] for k in model}

        return gradient

    def regional_aggregate(self, region_id: str) -> Dict[str, np.ndarray]:
        """
        Aggregate client updates at regional level.

        Returns:
            Regional gradient to send to national aggregator.
        """
        region_node = self.topology.nodes.get(region_id)
        if region_node is None or region_node.tier != TierLevel.REGIONAL:
            raise ValueError(f"Invalid region: {region_id}")

        # Get client updates
        clients = self.topology.get_children(region_id)

        if not clients:
            return {}

        total_samples = sum(c.sample_count for c in clients)
        if total_samples == 0:
            return {}

        # Weighted average of client models
        aggregated = {k: np.zeros_like(v) for k, v in self.global_model.items()}

        for client in clients:
            if client.local_model is None:
                continue

            weight = client.sample_count / total_samples
            for key in aggregated:
                if key in client.local_model:
                    aggregated[key] += weight * client.local_model[key]

        # Update regional model
        initial_model = region_node.local_model
        region_node.local_model = aggregated
        region_node.sample_count = total_samples
        region_node.last_update_round = self.tier_rounds[TierLevel.REGIONAL]

        # Propagate aggregated model back to clients
        for client in clients:
            client.local_model = deepcopy(aggregated)

        # Compute gradient (regional vs global)
        gradient = {k: aggregated[k] - initial_model[k] for k in aggregated}

        return gradient

    def national_aggregate(self, country_id: str) -> Dict[str, np.ndarray]:
        """
        Aggregate regional updates at national level.

        Returns:
            National gradient to send to EU aggregator.
        """
        country_node = self.topology.nodes.get(country_id)
        if country_node is None or country_node.tier != TierLevel.NATIONAL:
            raise ValueError(f"Invalid country: {country_id}")

        regions = self.topology.get_children(country_id)

        if not regions:
            return {}

        total_samples = sum(r.sample_count for r in regions)
        if total_samples == 0:
            return {}

        # Weighted average
        aggregated = {k: np.zeros_like(v) for k, v in self.global_model.items()}

        for region in regions:
            if region.local_model is None:
                continue

            weight = region.sample_count / total_samples
            for key in aggregated:
                if key in region.local_model:
                    aggregated[key] += weight * region.local_model[key]

        initial_model = country_node.local_model
        country_node.local_model = aggregated
        country_node.sample_count = total_samples
        country_node.last_update_round = self.tier_rounds[TierLevel.NATIONAL]

        # Propagate down
        for region in regions:
            region.local_model = deepcopy(aggregated)
            for client in self.topology.get_children(region.node_id):
                client.local_model = deepcopy(aggregated)

        gradient = {k: aggregated[k] - initial_model[k] for k in aggregated}

        return gradient

    def eu_aggregate(self) -> None:
        """
        EU-level global aggregation.

        Updates the global model and propagates to all nodes.
        """
        eu_node = self.topology.nodes.get(self.topology.root_id)
        if eu_node is None:
            return

        countries = self.topology.get_children(self.topology.root_id)

        if not countries:
            return

        total_samples = sum(c.sample_count for c in countries)
        if total_samples == 0:
            return

        # Weighted average
        aggregated = {k: np.zeros_like(v) for k, v in self.global_model.items()}

        for country in countries:
            if country.local_model is None:
                continue

            weight = country.sample_count / total_samples
            for key in aggregated:
                if key in country.local_model:
                    aggregated[key] += weight * country.local_model[key]

        # Add tier noise if configured
        if self.config.add_tier_noise:
            for key in aggregated:
                noise = np.random.normal(0, self.config.noise_multiplier, aggregated[key].shape)
                aggregated[key] += noise

        # Update global model
        self.global_model = aggregated
        eu_node.local_model = deepcopy(aggregated)
        eu_node.sample_count = total_samples
        eu_node.last_update_round = self.tier_rounds[TierLevel.SUPRANATIONAL]

        # Propagate to all nodes
        for node in self.topology.nodes.values():
            node.local_model = deepcopy(aggregated)


# =============================================================================
# HIERARCHICAL FL COORDINATOR
# =============================================================================

class HierarchicalFLCoordinator:
    """
    Coordinates hierarchical FL training.

    Manages the multi-level aggregation schedule and
    synchronization across tiers.
    """

    def __init__(self,
                 topology: FederationTopology,
                 config: HierarchicalConfig):
        self.topology = topology
        self.config = config
        self.hier_fedavg = HierFedAvg(topology, config)

        # Training state
        self.global_round = 0
        self.client_round = 0
        self.regional_round = 0
        self.national_round = 0

        # History
        self.history = {
            'global_rounds': [],
            'tier_accuracies': {tier.name: [] for tier in TierLevel},
            'communication_cost': []
        }

    def initialize(self, model_template: Dict[str, np.ndarray]) -> None:
        """Initialize the hierarchical federation."""
        self.hier_fedavg.initialize(model_template)

    def run_round(self,
                 client_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
                 client_epochs: int = 1,
                 lr: float = 0.01) -> Dict:
        """
        Run one complete hierarchical round.

        Schedule:
        1. All clients train locally
        2. Every K client rounds: regional aggregation
        3. Every M regional rounds: national aggregation
        4. Every N national rounds: EU aggregation
        """
        round_info = {
            'global_round': self.global_round,
            'aggregations': [],
            'client_metrics': {}
        }

        # Client training
        for client_id, data in client_data.items():
            self.hier_fedavg.client_train(client_id, data, client_epochs, lr)

            # Compute metrics
            X, y = data
            model = self.topology.nodes[client_id].local_model
            if model and 'weights' in model:
                logits = X @ model['weights']
                if 'bias' in model:
                    logits += model['bias']
                preds = (logits > 0).astype(float).flatten()
                acc = np.mean(preds == y)
                round_info['client_metrics'][client_id] = {'accuracy': acc}

        self.client_round += 1

        # Regional aggregation
        if self.client_round % self.config.regional_rounds == 0:
            for region in self.topology.get_nodes_at_tier(TierLevel.REGIONAL):
                self.hier_fedavg.regional_aggregate(region.node_id)
            round_info['aggregations'].append('regional')
            self.regional_round += 1

            # National aggregation
            if self.regional_round % self.config.national_rounds == 0:
                for country in self.topology.get_nodes_at_tier(TierLevel.NATIONAL):
                    self.hier_fedavg.national_aggregate(country.node_id)
                round_info['aggregations'].append('national')
                self.national_round += 1

                # EU aggregation (every national round for simplicity)
                self.hier_fedavg.eu_aggregate()
                round_info['aggregations'].append('supranational')
                self.global_round += 1

        return round_info

    def evaluate_by_tier(self,
                        test_data: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
        """Evaluate model performance at each tier."""
        X, y = test_data
        results = {}

        for tier in TierLevel:
            nodes = self.topology.get_nodes_at_tier(tier)
            if not nodes:
                continue

            # Use first node's model as representative
            model = nodes[0].local_model
            if model is None or 'weights' not in model:
                continue

            logits = X @ model['weights']
            if 'bias' in model:
                logits += model['bias']
            preds = (logits > 0).astype(float).flatten()
            acc = np.mean(preds == y)

            results[tier.name] = acc

        return results


# =============================================================================
# CLUSTERED HIERARCHICAL FL
# =============================================================================

class ClusteredHierFL:
    """
    Hierarchical FL with automatic client clustering.

    Instead of fixed geographic hierarchy, clusters clients
    based on data similarity or model similarity.

    Useful when geographic proximity doesn't imply data similarity.
    """

    def __init__(self, num_clusters: int = 3):
        self.num_clusters = num_clusters
        self.cluster_assignments: Dict[str, int] = {}
        self.cluster_models: Dict[int, Dict[str, np.ndarray]] = {}
        self.global_model: Dict[str, np.ndarray] = {}

    def initialize(self, model_template: Dict[str, np.ndarray]) -> None:
        """Initialize global and cluster models."""
        self.global_model = {k: v.copy() for k, v in model_template.items()}

        for i in range(self.num_clusters):
            self.cluster_models[i] = {k: v.copy() for k, v in model_template.items()}

    def assign_clusters(self,
                       client_gradients: Dict[str, Dict[str, np.ndarray]]) -> None:
        """
        Assign clients to clusters based on gradient similarity.

        Uses k-means on flattened gradients.
        """
        if not client_gradients:
            return

        client_ids = list(client_gradients.keys())

        # Flatten gradients
        flat_grads = []
        for cid in client_ids:
            grad = client_gradients[cid]
            flat = np.concatenate([g.flatten() for g in grad.values()])
            flat_grads.append(flat)

        flat_grads = np.array(flat_grads)

        # Simple k-means
        centers = flat_grads[:self.num_clusters]  # Initialize with first k

        for _ in range(10):  # K-means iterations
            # Assign to nearest center
            distances = np.array([
                [np.linalg.norm(g - c) for c in centers]
                for g in flat_grads
            ])
            assignments = np.argmin(distances, axis=1)

            # Update centers
            new_centers = []
            for k in range(self.num_clusters):
                cluster_grads = flat_grads[assignments == k]
                if len(cluster_grads) > 0:
                    new_centers.append(np.mean(cluster_grads, axis=0))
                else:
                    new_centers.append(centers[k])
            centers = np.array(new_centers)

        # Store assignments
        for i, cid in enumerate(client_ids):
            self.cluster_assignments[cid] = assignments[i]

    def cluster_aggregate(self,
                         client_updates: Dict[str, Dict[str, np.ndarray]],
                         sample_counts: Dict[str, int]) -> None:
        """Aggregate within clusters, then globally."""
        # Aggregate per cluster
        for cluster_id in range(self.num_clusters):
            cluster_clients = [
                cid for cid, c in self.cluster_assignments.items()
                if c == cluster_id and cid in client_updates
            ]

            if not cluster_clients:
                continue

            total_samples = sum(sample_counts.get(cid, 1) for cid in cluster_clients)

            aggregated = {k: np.zeros_like(v) for k, v in self.global_model.items()}

            for cid in cluster_clients:
                weight = sample_counts.get(cid, 1) / total_samples
                for key in aggregated:
                    if key in client_updates[cid]:
                        aggregated[key] += weight * client_updates[cid][key]

            # Update cluster model
            for key in self.cluster_models[cluster_id]:
                self.cluster_models[cluster_id][key] += aggregated[key]

        # Global aggregation of cluster models
        cluster_sizes = [
            sum(1 for c in self.cluster_assignments.values() if c == i)
            for i in range(self.num_clusters)
        ]
        total_clients = sum(cluster_sizes)

        if total_clients > 0:
            for key in self.global_model:
                self.global_model[key] = sum(
                    (cluster_sizes[i] / total_clients) * self.cluster_models[i][key]
                    for i in range(self.num_clusters)
                )


# =============================================================================
# FACTORY & DEMO
# =============================================================================

def create_hierarchical_fl(countries: Dict[str, List[str]],
                          regions: Dict[str, Dict[str, List[str]]],
                          config: Optional[HierarchicalConfig] = None) -> HierarchicalFLCoordinator:
    """
    Factory function for hierarchical FL.

    Args:
        countries: {country_code: [region_ids]}
        regions: {country_code: {region_id: [client_ids]}}
        config: HierarchicalConfig
    """
    topology = FederationTopology()
    topology.build_ehds_topology(countries, regions)

    if config is None:
        config = HierarchicalConfig()

    return HierarchicalFLCoordinator(topology, config)


if __name__ == "__main__":
    print("FL-EHDS Hierarchical Federated Learning Demo")
    print("=" * 70)

    np.random.seed(42)

    # Build EHDS-like topology
    # EU -> Countries -> Regions -> Hospitals

    countries = {
        "DE": ["DE-Bavaria", "DE-Berlin"],
        "FR": ["FR-IDF", "FR-PACA"],
        "IT": ["IT-Lombardy", "IT-Lazio"]
    }

    regions = {
        "DE": {
            "DE-Bavaria": ["DE-MUC-01", "DE-MUC-02", "DE-AUG-01"],
            "DE-Berlin": ["DE-BER-01", "DE-BER-02"]
        },
        "FR": {
            "FR-IDF": ["FR-PAR-01", "FR-PAR-02"],
            "FR-PACA": ["FR-MRS-01"]
        },
        "IT": {
            "IT-Lombardy": ["IT-MIL-01", "IT-MIL-02"],
            "IT-Lazio": ["IT-ROM-01"]
        }
    }

    config = HierarchicalConfig(
        regional_rounds=3,
        national_rounds=2,
        add_tier_noise=True,
        noise_multiplier=0.001
    )

    coordinator = create_hierarchical_fl(countries, regions, config)

    print("\nFederation Topology:")
    print("-" * 40)
    coordinator.topology.print_topology()

    # Generate synthetic data for each client
    clients = coordinator.topology.get_clients()
    print(f"\nTotal clients: {len(clients)}")

    input_dim = 6
    model_template = {
        'weights': np.zeros(input_dim),
        'bias': np.zeros(1)
    }

    coordinator.initialize(model_template)

    # Generate data with country-specific distributions
    client_data = {}
    country_biases = {"DE": 0.3, "FR": -0.2, "IT": 0.1}

    for client in clients:
        n_samples = np.random.randint(100, 300)
        country = client.country

        # Country-specific distribution
        bias = country_biases.get(country, 0)
        X = np.random.randn(n_samples, input_dim - 1)
        X = np.hstack([X, np.ones((n_samples, 1))])

        noise = np.random.randn(n_samples) * 0.3
        y = (X[:, 0] + X[:, 1] + bias + noise > 0).astype(float)

        client_data[client.node_id] = (X, y)

    # Training
    print("\n" + "-" * 70)
    print("Hierarchical FL Training")
    print("-" * 70)

    n_total_rounds = 30

    for round_num in range(n_total_rounds):
        round_info = coordinator.run_round(
            client_data,
            client_epochs=2,
            lr=0.1
        )

        if round_info['aggregations']:
            agg_str = " + ".join(round_info['aggregations'])
            avg_acc = np.mean([
                m['accuracy'] for m in round_info['client_metrics'].values()
            ]) if round_info['client_metrics'] else 0

            print(f"Round {round_num+1:2d}: Client Avg Acc={avg_acc:.2%}, "
                  f"Aggregations: {agg_str}")

    # Final evaluation by tier
    print("\n" + "-" * 70)
    print("Final Evaluation by Tier")
    print("-" * 70)

    # Combine all data for test
    all_X = np.vstack([d[0] for d in client_data.values()])
    all_y = np.concatenate([d[1] for d in client_data.values()])

    tier_results = coordinator.evaluate_by_tier((all_X, all_y))

    for tier_name, acc in tier_results.items():
        print(f"  {tier_name}: {acc:.2%}")

    # Per-country evaluation
    print("\n" + "-" * 70)
    print("Per-Country Accuracy")
    print("-" * 70)

    for country_code in countries.keys():
        country_clients = [
            cid for cid, (X, y) in client_data.items()
            if coordinator.topology.nodes[cid].country == country_code
        ]

        if not country_clients:
            continue

        country_X = np.vstack([client_data[cid][0] for cid in country_clients])
        country_y = np.concatenate([client_data[cid][1] for cid in country_clients])

        model = coordinator.hier_fedavg.global_model
        logits = country_X @ model['weights'] + model['bias']
        preds = (logits > 0).astype(float).flatten()
        acc = np.mean(preds == country_y)

        print(f"  {country_code}: {acc:.2%}")

    print("\n" + "=" * 70)
    print("Demo completed!")

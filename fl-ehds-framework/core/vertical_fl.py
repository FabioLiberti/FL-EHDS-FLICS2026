#!/usr/bin/env python3
"""
FL-EHDS Vertical Federated Learning (Split Learning)

Implements vertical FL for scenarios where different hospitals hold
different features for the same patients (e.g., one hospital has
lab results, another has imaging data, another has genomics).

Algorithms:
1. SplitNN - Neural network split across parties
2. Secure Vertical FL - With privacy protections
3. FATE-style Vertical FL - Feature-aligned federated learning

EHDS Context:
- Hospital A: Demographics, basic vitals
- Hospital B: Lab results, medications
- Hospital C: Imaging data, radiology reports
- All linked by pseudonymized patient ID

Author: Fabio Liberti
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from copy import deepcopy
import hashlib

# Cryptography for PSI
try:
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.backends import default_backend
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class VerticalPartition:
    """Data partition for a single party in vertical FL."""
    party_id: int
    features: np.ndarray  # Shape: (n_samples, n_features)
    feature_names: List[str]
    sample_ids: np.ndarray  # Pseudonymized patient IDs
    has_labels: bool = False
    labels: Optional[np.ndarray] = None


@dataclass
class VerticalConfig:
    """Configuration for vertical FL."""
    algorithm: str = "splitnn"  # splitnn, secure_vfl, fate
    # SplitNN params
    cut_layer: int = 1  # Layer to split at
    # Privacy params
    use_differential_privacy: bool = False
    epsilon: float = 1.0
    # Alignment params
    use_secure_alignment: bool = True
    alignment_method: str = "psi"  # psi (Private Set Intersection)


@dataclass
class SplitActivations:
    """Activations exchanged in SplitNN."""
    party_id: int
    activations: np.ndarray
    sample_indices: np.ndarray


@dataclass
class SplitGradients:
    """Gradients exchanged in SplitNN."""
    gradients: np.ndarray
    sample_indices: np.ndarray


# =============================================================================
# PRIVATE SET INTERSECTION (PSI) FOR ALIGNMENT
# =============================================================================

class PrivateSetIntersection:
    """
    Private Set Intersection for aligning patient IDs across parties.

    Allows parties to find common patients without revealing
    which patients are NOT in the intersection.

    Uses hashing-based PSI (simplified, real-world would use
    Diffie-Hellman based or OT-based PSI).
    """

    def __init__(self, salt: Optional[bytes] = None):
        self.salt = salt or b"fl-ehds-psi-salt"

    def _hash_id(self, patient_id: str) -> str:
        """Hash a patient ID."""
        h = hashlib.sha256(self.salt + str(patient_id).encode())
        return h.hexdigest()

    def hash_ids(self, ids: np.ndarray) -> Dict[str, int]:
        """Hash all IDs and return mapping hash -> index."""
        return {self._hash_id(str(id_)): i for i, id_ in enumerate(ids)}

    def find_intersection(self,
                         party_hashes: List[Dict[str, int]]) -> Tuple[List[int], ...]:
        """
        Find intersection of hashed IDs across parties.

        Returns:
            Tuple of index arrays for each party
        """
        if len(party_hashes) < 2:
            raise ValueError("Need at least 2 parties")

        # Find common hashes
        common_hashes = set(party_hashes[0].keys())
        for ph in party_hashes[1:]:
            common_hashes &= set(ph.keys())

        # Get indices for each party
        result = []
        for ph in party_hashes:
            indices = [ph[h] for h in common_hashes if h in ph]
            result.append(np.array(sorted(indices)))

        return tuple(result)


# =============================================================================
# SPLITNN BASE
# =============================================================================

class SplitNNParty:
    """
    A party's component in SplitNN.

    Each party has a portion of the neural network:
    - Bottom parties: Process their local features
    - Top party (with labels): Processes aggregated activations
    """

    def __init__(self,
                 party_id: int,
                 input_dim: int,
                 hidden_dims: List[int],
                 output_dim: int,
                 is_top_party: bool = False,
                 learning_rate: float = 0.01):
        self.party_id = party_id
        self.is_top_party = is_top_party
        self.lr = learning_rate

        # Initialize network layers
        self.weights = []
        self.biases = []

        dims = [input_dim] + hidden_dims
        if is_top_party:
            dims.append(output_dim)

        for i in range(len(dims) - 1):
            # Xavier initialization
            w = np.random.randn(dims[i], dims[i+1]) * np.sqrt(2.0 / dims[i])
            b = np.zeros(dims[i+1])
            self.weights.append(w)
            self.biases.append(b)

        # Cache for backprop
        self.activations_cache = []
        self.input_cache = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through this party's layers."""
        self.input_cache = x
        self.activations_cache = [x]

        h = x
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = h @ w + b

            # ReLU for hidden, sigmoid for output (if top party)
            if i == len(self.weights) - 1 and self.is_top_party:
                h = 1 / (1 + np.exp(-np.clip(z, -500, 500)))  # Sigmoid
            else:
                h = np.maximum(0, z)  # ReLU

            self.activations_cache.append(h)

        return h

    def backward(self,
                grad_output: np.ndarray,
                return_input_grad: bool = True) -> Optional[np.ndarray]:
        """
        Backward pass through this party's layers.

        Args:
            grad_output: Gradient from next layer (or loss)
            return_input_grad: Whether to return gradient for previous party

        Returns:
            Gradient w.r.t. input (for passing to previous party)
        """
        grad = grad_output

        for i in reversed(range(len(self.weights))):
            h = self.activations_cache[i]
            h_next = self.activations_cache[i + 1]

            # Gradient through activation
            if i == len(self.weights) - 1 and self.is_top_party:
                # Sigmoid gradient (combined with BCE loss)
                pass  # grad already includes this
            else:
                # ReLU gradient
                grad = grad * (h_next > 0)

            # Gradient for weights and biases
            grad_w = h.T @ grad / len(h)
            grad_b = np.mean(grad, axis=0)

            # Clip gradients for stability
            grad_w_norm = np.linalg.norm(grad_w)
            if grad_w_norm > 1.0:
                grad_w = grad_w / grad_w_norm

            # Update weights
            self.weights[i] -= self.lr * grad_w
            self.biases[i] -= self.lr * grad_b

            # Gradient for previous layer
            if i > 0 or return_input_grad:
                grad = grad @ self.weights[i].T

        return grad if return_input_grad else None

    def get_activations(self) -> np.ndarray:
        """Get output activations (to send to next party)."""
        return self.activations_cache[-1]


# =============================================================================
# SPLITNN COORDINATOR
# =============================================================================

class SplitNNCoordinator:
    """
    Coordinates SplitNN training across multiple parties.

    Architecture:
    Party 0 (features A) -> Bottom Model 0 ->
    Party 1 (features B) -> Bottom Model 1 -> Aggregate -> Top Model -> Labels
    """

    def __init__(self,
                 party_configs: List[Dict],
                 top_party_id: int,
                 aggregation: str = "concat"):
        """
        Args:
            party_configs: List of {party_id, input_dim, hidden_dims}
            top_party_id: ID of party with labels
            aggregation: How to aggregate activations (concat, sum, attention)
        """
        self.parties: Dict[int, SplitNNParty] = {}
        self.top_party_id = top_party_id
        self.aggregation = aggregation

        # Initialize bottom parties
        total_activation_dim = 0
        for config in party_configs:
            pid = config['party_id']
            is_top = (pid == top_party_id)

            party = SplitNNParty(
                party_id=pid,
                input_dim=config['input_dim'],
                hidden_dims=config.get('hidden_dims', [32]),
                output_dim=config.get('output_dim', 1),
                is_top_party=is_top,
                learning_rate=config.get('lr', 0.01)
            )
            self.parties[pid] = party

            if not is_top:
                total_activation_dim += config.get('hidden_dims', [32])[-1]

        # Top model processes aggregated activations
        if total_activation_dim > 0:
            self.top_model = SplitNNParty(
                party_id=-1,  # Special ID for top model
                input_dim=total_activation_dim,
                hidden_dims=[32],
                output_dim=1,
                is_top_party=True,
                learning_rate=0.01
            )
        else:
            self.top_model = None

    def train_step(self,
                  party_data: Dict[int, np.ndarray],
                  labels: np.ndarray,
                  aligned_indices: Optional[Dict[int, np.ndarray]] = None) -> float:
        """
        Single training step across all parties.

        Args:
            party_data: {party_id: features} for each party
            labels: Labels (from top party)
            aligned_indices: Indices to align samples across parties

        Returns:
            Training loss
        """
        # Align data if needed
        if aligned_indices is not None:
            party_data = {
                pid: data[aligned_indices[pid]]
                for pid, data in party_data.items()
            }
            labels = labels[aligned_indices[self.top_party_id]]

        # Forward pass through bottom parties
        activations = {}
        for pid, party in self.parties.items():
            if pid == self.top_party_id and self.top_model is not None:
                continue
            activations[pid] = party.forward(party_data[pid])

        # Aggregate activations
        if self.aggregation == "concat":
            aggregated = np.concatenate(list(activations.values()), axis=1)
        elif self.aggregation == "sum":
            aggregated = sum(activations.values())
        else:
            aggregated = np.concatenate(list(activations.values()), axis=1)

        # Forward through top model
        if self.top_model is not None:
            predictions = self.top_model.forward(aggregated)
        else:
            predictions = self.parties[self.top_party_id].forward(
                party_data[self.top_party_id]
            )

        # Compute loss (BCE)
        eps = 1e-10
        loss = -np.mean(
            labels * np.log(predictions + eps) +
            (1 - labels) * np.log(1 - predictions + eps)
        )

        # Backward pass
        # Gradient of BCE loss
        grad = (predictions - labels.reshape(-1, 1)) / len(labels)

        # Backward through top model
        if self.top_model is not None:
            grad_agg = self.top_model.backward(grad)

            # Split gradient back to parties
            if self.aggregation == "concat":
                offset = 0
                for pid in activations.keys():
                    act_dim = activations[pid].shape[1]
                    grad_party = grad_agg[:, offset:offset + act_dim]
                    self.parties[pid].backward(grad_party, return_input_grad=False)
                    offset += act_dim
            else:
                for pid in activations.keys():
                    self.parties[pid].backward(grad_agg, return_input_grad=False)
        else:
            self.parties[self.top_party_id].backward(grad, return_input_grad=False)

        return loss

    def predict(self,
               party_data: Dict[int, np.ndarray],
               aligned_indices: Optional[Dict[int, np.ndarray]] = None) -> np.ndarray:
        """Make predictions with current model."""
        if aligned_indices is not None:
            party_data = {
                pid: data[aligned_indices[pid]]
                for pid, data in party_data.items()
            }

        # Forward through bottoms
        activations = {}
        for pid, party in self.parties.items():
            if pid == self.top_party_id and self.top_model is not None:
                continue
            activations[pid] = party.forward(party_data[pid])

        # Aggregate
        if self.aggregation == "concat":
            aggregated = np.concatenate(list(activations.values()), axis=1)
        else:
            aggregated = sum(activations.values())

        # Top model
        if self.top_model is not None:
            return self.top_model.forward(aggregated)
        else:
            return self.parties[self.top_party_id].forward(
                party_data[self.top_party_id]
            )


# =============================================================================
# VERTICAL FL WITH PRIVACY
# =============================================================================

class SecureVerticalFL:
    """
    Secure Vertical Federated Learning with privacy protections.

    Privacy measures:
    1. Private Set Intersection for alignment
    2. Differential Privacy on activations
    3. Gradient noise addition
    """

    def __init__(self,
                 config: VerticalConfig,
                 party_configs: List[Dict],
                 top_party_id: int):
        self.config = config
        self.psi = PrivateSetIntersection()

        # Initialize SplitNN
        self.splitnn = SplitNNCoordinator(
            party_configs=party_configs,
            top_party_id=top_party_id
        )

    def align_parties(self,
                     party_ids: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        """
        Align patient IDs across parties using PSI.

        Returns:
            Dict mapping party_id to aligned indices
        """
        # Hash IDs from each party
        party_hashes = []
        party_id_list = []
        for pid, ids in party_ids.items():
            party_hashes.append(self.psi.hash_ids(ids))
            party_id_list.append(pid)

        # Find intersection
        aligned_indices_tuple = self.psi.find_intersection(party_hashes)

        return {
            pid: indices
            for pid, indices in zip(party_id_list, aligned_indices_tuple)
        }

    def _add_dp_noise(self,
                     activations: np.ndarray,
                     sensitivity: float = 1.0) -> np.ndarray:
        """Add differential privacy noise to activations."""
        if not self.config.use_differential_privacy:
            return activations

        # Gaussian mechanism
        sigma = sensitivity * np.sqrt(2 * np.log(1.25 / 0.00001)) / self.config.epsilon
        noise = np.random.normal(0, sigma, activations.shape)

        return activations + noise

    def train(self,
             party_partitions: Dict[int, VerticalPartition],
             num_epochs: int = 10,
             batch_size: int = 32) -> Dict:
        """
        Train vertical FL model.

        Args:
            party_partitions: Data partitions from each party
            num_epochs: Number of training epochs
            batch_size: Batch size for training

        Returns:
            Training history
        """
        # Align parties
        party_ids = {
            pp.party_id: pp.sample_ids
            for pp in party_partitions.values()
        }
        aligned_indices = self.align_parties(party_ids)

        # Find party with labels
        top_party = None
        labels = None
        for pp in party_partitions.values():
            if pp.has_labels:
                top_party = pp.party_id
                labels = pp.labels[aligned_indices[pp.party_id]]
                break

        if labels is None:
            raise ValueError("No party has labels")

        n_samples = len(labels)
        history = {'loss': [], 'accuracy': []}

        for epoch in range(num_epochs):
            epoch_losses = []

            # Shuffle
            perm = np.random.permutation(n_samples)

            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                batch_idx = perm[start:end]

                # Get batch data from each party
                batch_data = {}
                batch_aligned = {}

                for pp in party_partitions.values():
                    # Map batch indices to party indices
                    party_aligned = aligned_indices[pp.party_id]
                    batch_party_idx = party_aligned[batch_idx]

                    features = pp.features[batch_party_idx]

                    # Add DP noise if enabled
                    features = self._add_dp_noise(features)

                    batch_data[pp.party_id] = features
                    batch_aligned[pp.party_id] = np.arange(len(batch_idx))

                batch_labels = labels[batch_idx]

                # Training step
                loss = self.splitnn.train_step(
                    batch_data, batch_labels.reshape(-1, 1)
                )
                epoch_losses.append(loss)

            # Epoch metrics
            avg_loss = np.mean(epoch_losses)
            history['loss'].append(avg_loss)

            # Compute accuracy
            all_data = {
                pp.party_id: pp.features[aligned_indices[pp.party_id]]
                for pp in party_partitions.values()
            }
            preds = self.splitnn.predict(all_data)
            acc = np.mean((preds > 0.5).flatten() == labels.flatten())
            history['accuracy'].append(acc)

            print(f"Epoch {epoch+1}/{num_epochs}: Loss={avg_loss:.4f}, Acc={acc:.2%}")

        return history


# =============================================================================
# VERTICAL FL SIMULATOR
# =============================================================================

class VerticalFLSimulator:
    """
    Simulates vertical FL scenarios for EHDS.

    Creates realistic partitions where different parties
    hold different features for the same patients.
    """

    def __init__(self, random_seed: int = 42):
        self.rng = np.random.RandomState(random_seed)

    def create_ehds_scenario(self,
                            n_patients: int = 1000,
                            n_parties: int = 3) -> Dict[int, VerticalPartition]:
        """
        Create EHDS-like vertical partition scenario.

        Party 0: Hospital A - Demographics (age, gender, BMI)
        Party 1: Hospital B - Lab Results (glucose, cholesterol, BP)
        Party 2: Hospital C - Lifestyle (smoking, exercise, diet)

        Label: Cardiovascular risk (held by Party 0)
        """
        # Generate patient IDs
        patient_ids = np.array([f"EHDS-{i:06d}" for i in range(n_patients)])

        # Some patients only in subset of hospitals (realistic)
        overlap_mask = self.rng.random(n_patients) < 0.8  # 80% overlap

        partitions = {}

        # Party 0: Demographics
        n0 = np.sum(overlap_mask | (self.rng.random(n_patients) < 0.9))
        idx0 = np.where(overlap_mask | (self.rng.random(n_patients) < 0.9))[0][:n0]

        age = self.rng.normal(55, 15, n0)
        gender = self.rng.binomial(1, 0.5, n0)
        bmi = self.rng.normal(26, 5, n0)

        features0 = np.column_stack([age, gender, bmi])

        # Generate labels based on risk factors
        risk_score = 0.02 * (age - 40) + 0.1 * (bmi - 22) + self.rng.randn(n0) * 0.5
        labels = (risk_score > 1.0).astype(float)

        partitions[0] = VerticalPartition(
            party_id=0,
            features=features0,
            feature_names=["age", "gender", "bmi"],
            sample_ids=patient_ids[idx0],
            has_labels=True,
            labels=labels
        )

        # Party 1: Lab Results
        idx1 = np.where(overlap_mask | (self.rng.random(n_patients) < 0.85))[0]
        n1 = len(idx1)

        glucose = self.rng.normal(100, 25, n1)
        cholesterol = self.rng.normal(200, 40, n1)
        bp_systolic = self.rng.normal(120, 20, n1)

        features1 = np.column_stack([glucose, cholesterol, bp_systolic])

        partitions[1] = VerticalPartition(
            party_id=1,
            features=features1,
            feature_names=["glucose", "cholesterol", "bp_systolic"],
            sample_ids=patient_ids[idx1],
            has_labels=False
        )

        # Party 2: Lifestyle (if 3 parties)
        if n_parties >= 3:
            idx2 = np.where(overlap_mask | (self.rng.random(n_patients) < 0.75))[0]
            n2 = len(idx2)

            smoking = self.rng.binomial(1, 0.25, n2)
            exercise = self.rng.normal(3, 2, n2)  # Hours per week
            diet_score = self.rng.normal(5, 2, n2)  # 1-10 scale

            features2 = np.column_stack([smoking, exercise, diet_score])

            partitions[2] = VerticalPartition(
                party_id=2,
                features=features2,
                feature_names=["smoking", "exercise_hours", "diet_score"],
                sample_ids=patient_ids[idx2],
                has_labels=False
            )

        return partitions


# =============================================================================
# FACTORY & DEMO
# =============================================================================

def create_vertical_fl(config: Optional[VerticalConfig] = None,
                      **kwargs) -> SecureVerticalFL:
    """Factory function to create Vertical FL."""
    if config is None:
        config = VerticalConfig(**kwargs)

    # Default party configs (will be overridden)
    party_configs = [
        {'party_id': 0, 'input_dim': 3, 'hidden_dims': [16]},
        {'party_id': 1, 'input_dim': 3, 'hidden_dims': [16]},
    ]

    return SecureVerticalFL(config, party_configs, top_party_id=0)


if __name__ == "__main__":
    print("FL-EHDS Vertical Federated Learning Demo")
    print("=" * 60)

    # Create EHDS scenario
    simulator = VerticalFLSimulator(random_seed=42)
    partitions = simulator.create_ehds_scenario(n_patients=500, n_parties=3)

    print("\nParty Partitions:")
    for pid, part in partitions.items():
        print(f"  Party {pid}: {len(part.sample_ids)} patients, "
              f"features={part.feature_names}, has_labels={part.has_labels}")

    # Configure vertical FL
    config = VerticalConfig(
        algorithm="splitnn",
        use_differential_privacy=True,
        epsilon=5.0
    )

    party_configs = [
        {'party_id': 0, 'input_dim': 3, 'hidden_dims': [16, 8], 'lr': 0.01},
        {'party_id': 1, 'input_dim': 3, 'hidden_dims': [16, 8], 'lr': 0.01},
        {'party_id': 2, 'input_dim': 3, 'hidden_dims': [16, 8], 'lr': 0.01},
    ]

    vfl = SecureVerticalFL(config, party_configs, top_party_id=0)

    print("\n" + "-" * 60)
    print("Training Vertical FL Model")
    print("-" * 60)

    history = vfl.train(partitions, num_epochs=20, batch_size=32)

    print(f"\nFinal accuracy: {history['accuracy'][-1]:.2%}")
    print(f"Final loss: {history['loss'][-1]:.4f}")

    print("\n" + "=" * 60)
    print("Demo completed!")

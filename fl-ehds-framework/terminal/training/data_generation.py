"""
Data generation and device utilities for FL training.
"""

import numpy as np
from typing import Dict, Tuple

import torch


def _detect_device(device=None) -> torch.device:
    """Auto-detect best available device: CUDA > MPS > CPU."""
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def generate_healthcare_data(
    num_clients: int,
    samples_per_client: int = 200,
    num_features: int = 10,
    is_iid: bool = False,
    alpha: float = 0.5,
    seed: int = 42,
    test_split: float = 0.2
) -> Tuple[Dict[int, Tuple[np.ndarray, np.ndarray]], Dict[int, Tuple[np.ndarray, np.ndarray]]]:
    """
    Generate synthetic healthcare data for FL experiments.

    Features simulate clinical measurements:
    - Age (normalized)
    - BMI
    - Blood pressure (systolic)
    - Blood glucose
    - Cholesterol
    - Heart rate
    - Respiratory rate
    - Temperature
    - Oxygen saturation
    - Previous conditions count

    Target: Binary classification (disease risk: 0 = low, 1 = high)

    Returns:
        (client_train_data, client_test_data) - each is Dict[int, (X, y)]
    """
    np.random.seed(seed)

    client_data = {}

    # Generate base data
    total_samples = num_clients * samples_per_client

    # Clinical features with realistic correlations
    age = np.random.normal(55, 15, total_samples).clip(18, 90)
    bmi = np.random.normal(26, 5, total_samples).clip(15, 45)
    bp_systolic = 100 + 0.5 * age + 0.3 * bmi + np.random.normal(0, 10, total_samples)
    glucose = 80 + 0.2 * age + 0.5 * bmi + np.random.normal(0, 15, total_samples)
    cholesterol = 150 + 0.3 * age + 0.4 * bmi + np.random.normal(0, 30, total_samples)
    heart_rate = 70 + 0.1 * age + np.random.normal(0, 10, total_samples)
    resp_rate = 14 + np.random.normal(0, 2, total_samples)
    temperature = 36.6 + np.random.normal(0, 0.3, total_samples)
    oxygen_sat = 98 - 0.05 * age + np.random.normal(0, 1, total_samples)
    prev_conditions = np.random.poisson(1.5, total_samples)

    # Stack and normalize
    X = np.column_stack([
        age, bmi, bp_systolic, glucose, cholesterol,
        heart_rate, resp_rate, temperature, oxygen_sat, prev_conditions
    ])

    # Normalize to [0, 1]
    X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-8)

    # Generate labels based on risk factors
    risk_score = (
        0.3 * X[:, 0] +  # age
        0.2 * X[:, 1] +  # bmi
        0.15 * X[:, 2] + # bp
        0.15 * X[:, 3] + # glucose
        0.1 * X[:, 4] +  # cholesterol
        0.1 * X[:, 9]    # prev_conditions
    )
    risk_score += np.random.normal(0, 0.1, total_samples)
    y = (risk_score > np.median(risk_score)).astype(np.int64)

    if is_iid:
        # IID: Random distribution
        indices = np.random.permutation(total_samples)
        for i in range(num_clients):
            start = i * samples_per_client
            end = start + samples_per_client
            client_indices = indices[start:end]
            client_data[i] = (X[client_indices], y[client_indices])
    else:
        # Non-IID: Dirichlet distribution for label skew
        label_indices = {0: np.where(y == 0)[0], 1: np.where(y == 1)[0]}

        # Dirichlet allocation
        proportions = np.random.dirichlet([alpha] * num_clients, 2)

        for i in range(num_clients):
            client_indices = []
            for label in [0, 1]:
                n_samples = int(proportions[label, i] * len(label_indices[label]))
                n_samples = max(10, min(n_samples, len(label_indices[label])))
                chosen = np.random.choice(
                    label_indices[label],
                    size=min(n_samples, samples_per_client // 2),
                    replace=False
                )
                client_indices.extend(chosen)

            client_indices = np.array(client_indices)
            np.random.shuffle(client_indices)
            client_data[i] = (X[client_indices], y[client_indices])

    # Split each client's data into train/test
    client_train_data = {}
    client_test_data = {}

    for client_id, (X_c, y_c) in client_data.items():
        n = len(y_c)
        n_test = max(1, int(n * test_split))
        n_train = n - n_test

        perm = np.random.permutation(n)
        train_idx = perm[:n_train]
        test_idx = perm[n_train:]

        client_train_data[client_id] = (X_c[train_idx], y_c[train_idx])
        client_test_data[client_id] = (X_c[test_idx], y_c[test_idx])

    return client_train_data, client_test_data

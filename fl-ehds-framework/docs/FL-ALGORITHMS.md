# Federated Learning Algorithms - FL-EHDS Framework

## Overview

This document describes all 9 FL algorithms implemented in the FL-EHDS framework for privacy-preserving healthcare analytics in compliance with the European Health Data Space (EHDS) regulation.

## Implemented Algorithms

| Algorithm | Category | Description | Key Parameters | Best For |
|-----------|----------|-------------|----------------|----------|
| **FedAvg** | Baseline | Weighted averaging of client updates by sample count | - | Baseline comparison, IID data |
| **FedProx** | Regularization | Adds proximal term to handle heterogeneous data | `mu` (0.01-1.0) | Non-IID data, system heterogeneity |
| **SCAFFOLD** | Variance Reduction | Control variates to correct client drift | Server/client controls | Highly non-IID data, many rounds |
| **FedNova** | Normalization | Normalizes updates by local computation steps | - | Heterogeneous local epochs |
| **FedAdam** | Adaptive Server | Server-side Adam optimizer for faster convergence | `server_lr`, `beta1`, `beta2`, `tau` | Fast convergence needed |
| **FedYogi** | Adaptive Server | Like FedAdam but prevents fast velocity growth | `server_lr`, `beta1`, `beta2`, `tau` | Stable adaptive optimization |
| **FedAdagrad** | Adaptive Server | Server-side Adagrad (no momentum) | `server_lr`, `tau` | Sparse gradients |
| **Per-FedAvg** | Personalization | Local fine-tuning after global aggregation | - | Client-specific models needed |
| **Ditto** | Personalization | L2 regularization towards global model | `mu` (lambda) | Balance global/local performance |

---

## Algorithm Details

### 1. FedAvg (Federated Averaging)
**Reference**: McMahan et al., 2017

The baseline FL algorithm that performs weighted averaging of client model updates.

```
Global update: w_{t+1} = sum_k (n_k / n) * w_k^{t+1}
```

**Pros**: Simple, efficient communication
**Cons**: Struggles with non-IID data

---

### 2. FedProx (Federated Proximal)
**Reference**: Li et al., 2020

Adds a proximal term to the local objective to keep client updates close to the global model.

```
Local objective: min_w F_k(w) + (mu/2) * ||w - w^t||^2
```

**Parameters**:
- `mu`: Proximal regularization strength (default: 0.1)
  - Higher mu = more conservative updates
  - Typical range: 0.01 - 1.0

**Pros**: Handles statistical heterogeneity, robust to partial work
**Cons**: Additional hyperparameter tuning

---

### 3. SCAFFOLD
**Reference**: Karimireddy et al., 2020

Uses control variates to correct for client drift in non-IID settings.

```
Local update: w = w - lr * (grad + c - c_i)
Control update: c_i = c_i - c + (w_old - w_new) / (K * lr)
```

**Pros**: Provably faster convergence on non-IID data
**Cons**: 2x communication cost (sends control variates)

---

### 4. FedNova (Federated Normalized Averaging)
**Reference**: Wang et al., 2020

Normalizes client updates by the number of local steps to handle heterogeneous computation.

```
Global update: w_{t+1} = w_t + (sum tau_i / K) * sum_k (tau_k / sum tau) * delta_k
```

**Pros**: Handles heterogeneous local epochs/steps
**Cons**: Requires tracking local step counts

---

### 5. FedAdam
**Reference**: Reddi et al., 2021

Applies Adam optimizer on the server side for adaptive learning rates.

```
m_t = beta1 * m_{t-1} + (1 - beta1) * delta
v_t = beta2 * v_{t-1} + (1 - beta2) * delta^2
w_{t+1} = w_t + server_lr * m_t / (sqrt(v_t) + tau)
```

**Parameters**:
- `server_lr`: Server learning rate (default: 0.1)
- `beta1`: Momentum coefficient (default: 0.9)
- `beta2`: Velocity coefficient (default: 0.99)
- `tau`: Numerical stability (default: 1e-3)

**Pros**: Fast convergence, adaptive learning
**Cons**: More hyperparameters

---

### 6. FedYogi
**Reference**: Reddi et al., 2021

Similar to FedAdam but with modified velocity update to prevent fast growth.

```
v_t = v_{t-1} - (1 - beta2) * sign(v_{t-1} - delta^2) * delta^2
```

**Pros**: More stable than FedAdam
**Cons**: Slightly slower convergence

---

### 7. FedAdagrad
**Reference**: Reddi et al., 2021

Server-side Adagrad optimizer (accumulates squared gradients, no momentum).

```
v_t = v_{t-1} + delta^2
w_{t+1} = w_t + server_lr * delta / (sqrt(v_t) + tau)
```

**Pros**: Good for sparse gradients
**Cons**: Learning rate decay over time

---

### 8. Per-FedAvg (Personalized FedAvg)
**Reference**: Fallah et al., 2020

Performs local fine-tuning after receiving the global model.

```
1. Receive global model w_t
2. Local training: w_k = LocalTrain(w_t)
3. Fine-tune: w_k^pers = FineTune(w_t, local_data)
```

**Pros**: Personalized models for each client
**Cons**: Requires separate evaluation for personalized models

---

### 9. Ditto
**Reference**: Li et al., 2021

Trains personalized models with L2 regularization towards the global model.

```
Local personalized objective: min_v F_k(v) + (lambda/2) * ||v - w||^2
```

**Parameters**:
- `mu` (lambda): Regularization strength towards global model

**Pros**: Balances global generalization and local personalization
**Cons**: Maintains two models per client

---

## Recommendations by Use Case

### Healthcare Scenarios

| Scenario | Recommended Algorithms | Rationale |
|----------|----------------------|-----------|
| **Multi-hospital collaboration (similar populations)** | FedAvg, FedAdam | IID-like data, fast convergence |
| **Rare disease studies (heterogeneous data)** | FedProx, SCAFFOLD | Handles non-IID distributions |
| **Personalized medicine** | Ditto, Per-FedAvg | Patient-specific predictions |
| **Resource-constrained devices** | FedAvg, FedNova | Simple, handles variable computation |
| **Privacy-critical (with DP)** | FedAvg + DP, FedProx + DP | Compatible with gradient clipping |
| **Cross-border EHDS compliance** | SCAFFOLD, FedProx | Robust to data heterogeneity |

### Data Distribution Guidelines

| Data Type | Alpha (Dirichlet) | Recommended |
|-----------|-------------------|-------------|
| IID (uniform) | alpha > 10 | FedAvg, FedAdam |
| Mild non-IID | alpha = 1.0 - 10.0 | FedProx (mu=0.01) |
| Moderate non-IID | alpha = 0.5 - 1.0 | FedProx (mu=0.1), SCAFFOLD |
| Severe non-IID | alpha < 0.5 | SCAFFOLD, Ditto |

---

## Performance Comparison

Results from experiments with synthetic healthcare data (5 clients, 30 rounds, 3 seeds):

| Algorithm | Accuracy (Non-IID) | Convergence Speed | Communication |
|-----------|-------------------|-------------------|---------------|
| FedAvg | 60.9% +/- 0.02 | Baseline | 1x |
| FedProx | 60.9% +/- 0.02 | Similar | 1x |
| SCAFFOLD | 60.5% +/- 0.01 | Faster | 2x |
| FedNova | 60.7% +/- 0.02 | Similar | 1x |
| FedAdam | ~61% | Faster | 1x |
| FedYogi | ~61% | Faster | 1x |
| FedAdagrad | ~60% | Slower | 1x |
| Per-FedAvg | ~60% (global) | Similar | 1x |
| Ditto | ~60% (global) | Similar | 1x |

---

## Usage in FL-EHDS Framework

### Terminal Interface
```bash
cd fl-ehds-framework
conda activate flics2026
python -m terminal
# Select "Training Federato" or "Confronto Algoritmi FL"
```

### Programmatic Usage
```python
from terminal.fl_trainer import FederatedTrainer

trainer = FederatedTrainer(
    num_clients=5,
    algorithm="FedProx",  # or any of the 9 algorithms
    mu=0.1,               # FedProx/Ditto specific
    server_lr=0.1,        # FedAdam/Yogi/Adagrad specific
    dp_enabled=True,      # Optional: enable DP
    dp_epsilon=10.0,
)

for round_num in range(30):
    result = trainer.train_round(round_num)
    print(f"Round {round_num}: Acc={result.global_acc:.2%}")
```

---

## References

1. McMahan et al. (2017). "Communication-Efficient Learning of Deep Networks from Decentralized Data"
2. Li et al. (2020). "Federated Optimization in Heterogeneous Networks" (FedProx)
3. Karimireddy et al. (2020). "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning"
4. Wang et al. (2020). "Tackling the Objective Inconsistency Problem in Heterogeneous Federated Optimization" (FedNova)
5. Reddi et al. (2021). "Adaptive Federated Optimization" (FedAdam, FedYogi, FedAdagrad)
6. Fallah et al. (2020). "Personalized Federated Learning with Theoretical Guarantees" (Per-FedAvg)
7. Li et al. (2021). "Ditto: Fair and Robust Federated Learning Through Personalization"

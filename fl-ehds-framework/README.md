# FL-EHDS Framework

**Privacy-Preserving Federated Learning for the European Health Data Space**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)

This directory contains the full implementation of the FL-EHDS framework. For the project overview, paper details, and benchmark results, see the [root README](../README.md).

---

## Quick Start

```bash
# Install
conda create -n flics2026 python=3.11 -y && conda activate flics2026
pip install -e .

# Terminal CLI
python -m terminal

# Web Dashboard
streamlit run dashboard/app_v4.py

# Run experiments
python -m experiments.centralized_vs_federated --dataset chest_xray --quick
```

---

## Directory Structure

```
fl-ehds-framework/
|
|-- core/                           # Core FL engine
|   |-- fl_algorithms.py            # 10 aggregation algorithms
|   |-- orchestration.py            # FL orchestration logic
|   |-- models.py                   # Data models (Pydantic)
|   |-- secure_aggregation.py       # TenSEAL CKKS secure aggregation
|   |-- byzantine_resilience.py     # Byzantine fault tolerance
|   |-- async_fl.py                 # Asynchronous FL support
|   |-- utils.py                    # Utility functions
|   +-- exceptions.py               # Custom exceptions
|
|-- orchestration/                  # Layer 2: FL Orchestration
|   |-- aggregation/
|   |   |-- fedavg.py               # FedAvg strategy
|   |   +-- fedprox.py              # FedProx strategy
|   |-- privacy/
|   |   |-- differential_privacy.py # DP with RDP accounting
|   |   |-- gradient_clipping.py    # Gradient norm clipping
|   |   +-- secure_aggregation.py   # Cryptographic SecAgg
|   +-- compliance/
|       +-- purpose_limitation.py   # EHDS Art. 53 enforcement
|
|-- governance/                     # Layer 1: Governance
|   |-- hdab_integration.py         # Health Data Access Body API
|   |-- data_permits.py             # Data permit lifecycle
|   |-- optout_registry.py          # Art. 71 opt-out registry
|   +-- compliance_logging.py       # GDPR Art. 30 audit trails
|
|-- data_holders/                   # Layer 3: Data Holders
|   |-- training_engine.py          # Adaptive local training
|   |-- fhir_preprocessing.py       # HL7 FHIR R4 transformation
|   +-- secure_communication.py     # E2E encrypted gradients
|
|-- terminal/                       # Terminal CLI Interface
|   |-- __main__.py                 # Entry point
|   |-- fl_trainer.py               # FL trainer (10 algos + imaging)
|   +-- screens/
|       |-- training.py             # Training screen
|       |-- comparison.py           # Algorithm comparison
|       |-- guided_comparison.py    # Pre-configured scenarios
|       |-- algorithms.py           # Algorithm explorer
|       |-- datasets.py             # Dataset management
|       |-- privacy.py              # Privacy dashboard
|       |-- benchmark.py            # Benchmark suite
|       +-- byzantine.py            # Byzantine resilience
|
|-- dashboard/                      # Streamlit Web Dashboard
|   |-- app_v4.py                   # Main dashboard
|   |-- dataset_page.py             # Dataset browser
|   +-- real_trainer_bridge.py      # Terminal-Web bridge
|
|-- models/                         # Neural Network Models
|   |-- cnn_fl_trainer.py           # HealthcareCNN (5-block, GroupNorm)
|   +-- model_zoo.py                # Model registry
|
|-- benchmarks/                     # Experiment Scripts
|   |-- run_experiments.py          # Main benchmark (9 algorithms)
|   |-- run_imaging_experiments.py  # Clinical imaging benchmarks
|   |-- run_extended_experiments.py # Scalability analysis
|   +-- run_heterogeneity_experiments.py # Non-IID studies
|
|-- experiments/                    # Specialized Experiments
|   +-- centralized_vs_federated/   # CvF comparison suite
|       |-- run_comparison.py       # Main script
|       +-- visualizations.py       # 10-chart visualization suite
|
|-- tests/                          # Unit Tests (pytest)
|   |-- test_governance.py
|   |-- test_differential_privacy.py
|   |-- test_orchestration.py
|   +-- test_data_holders.py
|
|-- data/                           # Clinical Datasets (not in repo)
|   |-- chest_xray/                 # 5,856 images, 2 classes
|   |-- Brain_Tumor/                # 7,023 images, 4 classes
|   |-- Retinopatia/                # 35,126 images, 5 classes
|   |-- Skin Cancer/                # 3,297 images, 2 classes
|   |-- Brain Tumor MRI/            # 3,264 images, 4 classes
|   +-- ISIC/                       # 2,357 images, 9 classes
|
|-- results/                        # Auto-generated outputs
|-- config/                         # YAML configuration
|-- docs/                           # Documentation
|-- notebooks/                      # Jupyter notebooks
+-- setup.py                        # Package configuration
```

---

## FL Algorithms

All 10 algorithms are implemented in `terminal/fl_trainer.py` with real PyTorch training:

| # | Algorithm | Line | Key Parameter |
|---|-----------|------|---------------|
| 1 | FedAvg | L700 | -- |
| 2 | FedProx | L720 | mu=0.1 |
| 3 | SCAFFOLD | L730 | control variates |
| 4 | FedNova | L750 | tau_eff normalization |
| 5 | FedDyn | L836 | alpha regularization |
| 6 | FedAdam | L765 | server_lr=0.1, beta1=0.9 |
| 7 | FedYogi | L791 | beta2=0.99, tau=1e-3 |
| 8 | FedAdagrad | L819 | server_lr=0.1 |
| 9 | Per-FedAvg | L850 | local fine-tuning step |
| 10 | Ditto | L860 | lambda=0.1 |

---

## Configuration

Edit `config/config.yaml`:

```yaml
framework:
  name: "FL-EHDS"
  version: "1.0.0"

orchestration:
  aggregation:
    algorithm: "fedavg"
    num_rounds: 30
    min_clients: 3
  privacy:
    differential_privacy:
      enabled: true
      epsilon: 1.0
      delta: 1e-5
    gradient_clipping:
      max_norm: 1.0
    secure_aggregation:
      enabled: true

data_holders:
  training:
    batch_size: 32
    local_epochs: 5
    learning_rate: 0.001
```

---

## Testing

```bash
pytest tests/ -v
pytest tests/ --cov=fl_ehds --cov-report=html
```

---

## Author

**Fabio Liberti, PhD** -- Universitas Mercatorum, Rome, Italy
[ORCID: 0000-0003-3019-5411](https://orcid.org/0000-0003-3019-5411)

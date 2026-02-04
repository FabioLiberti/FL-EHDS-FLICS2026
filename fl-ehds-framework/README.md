# FL-EHDS Framework

**Privacy-Preserving Federated Learning Framework for the European Health Data Space**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)

## Overview

FL-EHDS is a three-layer compliance framework designed for cross-border health analytics under the European Health Data Space (EHDS) Regulation (EU) 2025/327. The framework bridges the technology-governance divide by integrating technical FL capabilities with regulatory compliance requirements.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    LAYER 1: GOVERNANCE                          │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌───────────┐ │
│  │    HDAB     │ │    Data     │ │   Opt-out   │ │ Compliance│ │
│  │ Integration │ │   Permits   │ │  Registry   │ │  Logging  │ │
│  └─────────────┘ └─────────────┘ └─────────────┘ └───────────┘ │
├─────────────────────────────────────────────────────────────────┤
│              LAYER 2: FL ORCHESTRATION (within SPE)             │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐   │
│  │   Aggregation   │ │     Privacy     │ │   Compliance    │   │
│  │  FedAvg/FedProx │ │ DP, SecAgg, Clip│ │ Purpose Limit.  │   │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘   │
├─────────────────────────────────────────────────────────────────┤
│                   LAYER 3: DATA HOLDERS                         │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐   │
│  │ Training Engine │ │ FHIR Preprocess │ │ Secure Comms    │   │
│  │   (Adaptive)    │ │  (Healthcare)   │ │ (E2E Encrypted) │   │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
fl-ehds-framework/
├── config/                 # Configuration files
│   └── config.yaml        # Main configuration
├── core/                   # Core utilities and base classes
│   ├── models.py          # Data models and schemas
│   ├── utils.py           # Utility functions
│   └── exceptions.py      # Custom exceptions
├── governance/             # Layer 1: Governance
│   ├── hdab_integration.py    # HDAB API integration
│   ├── data_permits.py        # Data permit management
│   ├── optout_registry.py     # Opt-out registry (Art. 71)
│   └── compliance_logging.py  # Audit trail logging
├── orchestration/          # Layer 2: FL Orchestration
│   ├── aggregation/           # Aggregation algorithms
│   │   ├── fedavg.py         # FedAvg implementation
│   │   └── fedprox.py        # FedProx for non-IID data
│   ├── privacy/               # Privacy protection
│   │   ├── differential_privacy.py
│   │   ├── gradient_clipping.py
│   │   └── secure_aggregation.py
│   └── compliance/            # Compliance enforcement
│       └── purpose_limitation.py
├── data_holders/           # Layer 3: Data Holders
│   ├── training_engine.py     # Adaptive local training
│   ├── fhir_preprocessing.py  # FHIR data transformation
│   └── secure_communication.py # Encrypted gradient exchange
├── tests/                  # Test suite
└── examples/               # Usage examples
```

## Installation

```bash
# Clone the repository
git clone https://github.com/your-org/fl-ehds-framework.git
cd fl-ehds-framework

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

## Quick Start

```python
from fl_ehds import FLEHDSFramework
from fl_ehds.governance import DataPermit, HDABClient
from fl_ehds.orchestration import FedAvgAggregator
from fl_ehds.data_holders import TrainingEngine

# Initialize framework
framework = FLEHDSFramework(config_path="config/config.yaml")

# Layer 1: Verify data permit
permit = DataPermit(
    permit_id="EHDS-2026-001",
    purpose="scientific_research",
    data_categories=["ehr", "lab_results"]
)
framework.governance.verify_permit(permit)

# Layer 2: Configure FL orchestration
aggregator = FedAvgAggregator(
    num_rounds=100,
    min_clients=3,
    privacy_budget=1.0  # epsilon for DP
)

# Layer 3: Initialize data holders
training_engine = TrainingEngine(
    model_type="neural_network",
    adaptive_batching=True
)

# Run federated training
results = framework.run(
    aggregator=aggregator,
    training_engine=training_engine,
    permit=permit
)
```

## Key Features

### Layer 1: Governance
- **HDAB Integration**: Standardized API for Health Data Access Body communication
- **Data Permits**: Automated permit verification and lifecycle management
- **Opt-out Registry**: Article 71 compliance with granular opt-out checking
- **Compliance Logging**: GDPR Article 30 audit trail generation

### Layer 2: FL Orchestration
- **FedAvg/FedProx**: Standard aggregation with non-IID data handling
- **Differential Privacy**: Configurable ε-budget with automatic noise calibration
- **Gradient Clipping**: Bounded sensitivity for privacy guarantees
- **Secure Aggregation**: Cryptographic protection of individual gradients
- **Purpose Limitation**: Technical enforcement of permitted uses (Article 53)

### Layer 3: Data Holders
- **Adaptive Training**: Resource-aware model partitioning for heterogeneous hardware
- **FHIR Preprocessing**: Healthcare data normalization and transformation
- **Secure Communication**: End-to-end encrypted gradient transmission

## EHDS Compliance

| EHDS Requirement | Framework Component | Implementation |
|------------------|---------------------|----------------|
| Article 53 (Purposes) | `purpose_limitation.py` | Purpose validation before training |
| Article 71 (Opt-out) | `optout_registry.py` | Per-record opt-out checking |
| GDPR Article 30 | `compliance_logging.py` | Complete audit trail |
| SPE requirement | `orchestration/` | All processing within SPE boundaries |
| Data minimization | `privacy/` | DP + gradient clipping |

## Configuration

Edit `config/config.yaml`:

```yaml
framework:
  name: "FL-EHDS"
  version: "1.0.0"

governance:
  hdab_endpoint: "https://hdab.example.eu/api/v1"
  permit_cache_ttl: 3600
  optout_sync_interval: 300

orchestration:
  aggregation:
    algorithm: "fedavg"  # or "fedprox"
    num_rounds: 100
    min_clients: 3
  privacy:
    differential_privacy:
      enabled: true
      epsilon: 1.0
      delta: 1e-5
    gradient_clipping:
      enabled: true
      max_norm: 1.0
    secure_aggregation:
      enabled: true

data_holders:
  training:
    batch_size: 32
    local_epochs: 5
    adaptive_batching: true
  preprocessing:
    fhir_version: "R4"
    normalize: true
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific layer tests
pytest tests/test_governance.py -v
pytest tests/test_orchestration.py -v
pytest tests/test_data_holders.py -v

# Run with coverage
pytest tests/ --cov=fl_ehds --cov-report=html
```

## References

- EHDS Regulation: [EU 2025/327](https://eur-lex.europa.eu/)
- FedAvg: McMahan et al., "Communication-Efficient Learning of Deep Networks from Decentralized Data" (2017)
- FedProx: Li et al., "Federated Optimization in Heterogeneous Networks" (2020)
- Differential Privacy: Dwork & Roth, "The Algorithmic Foundations of Differential Privacy" (2014)

## License

Apache License 2.0

## Author

Fabio Liberti, PhD
Department of Computer Science
Universitas Mercatorum, Rome, Italy
ORCID: [0000-0003-3019-5411](https://orcid.org/0000-0003-3019-5411)

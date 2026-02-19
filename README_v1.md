# FL-EHDS: A Privacy-Preserving Federated Learning Framework for the European Health Data Space

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![Conference: FLICS 2026](https://img.shields.io/badge/Conference-FLICS%202026-orange.svg)](https://www.flics-conference.org/)

> **Accompanying paper**: *FL-EHDS: A Privacy-Preserving Federated Learning Framework for the European Health Data Space*
> Submitted to [FLICS 2026](https://www.flics-conference.org/) -- 2nd International Conference on Federated Learning and Intelligent Computing Systems, Valencia, Spain, June 9--12, 2026.

---

## Abstract

FL-EHDS is a **three-layer compliance framework** that enables privacy-preserving federated learning across European healthcare institutions under the [EHDS Regulation (EU) 2025/327](https://eur-lex.europa.eu/). The framework implements **10 federated aggregation algorithms**, **Renyi Differential Privacy (RDP)** accounting, and **secure aggregation**, validated on **6 real-world clinical imaging datasets** (56,923 images) spanning radiology, dermatology, and ophthalmology.

### Key Results

| Metric | Centralized | Federated (FedAvg) | Privacy Cost |
|--------|:-----------:|:-------------------:|:------------:|
| Accuracy | 95.4% | 91.2% | -4.2 pp |
| F1 Score | 0.939 | 0.890 | -0.049 |
| AUC | 0.985 | 0.960 | -0.025 |

*Chest X-ray pneumonia detection -- Non-IID (Dirichlet alpha=0.5), 5 hospitals, HealthcareCNN (~422K params)*

---

## Architecture

FL-EHDS bridges the technology-governance divide through a three-layer design aligned with EHDS articles:

```
+-------------------------------------------------------------------+
|                   LAYER 1: GOVERNANCE                              |
|  +-------------+ +-------------+ +-------------+ +--------------+ |
|  |    HDAB     | |    Data     | |   Opt-out   | |  Compliance  | |
|  | Integration | |   Permits   | |  Registry   | |   Logging    | |
|  | (Art. 46)   | | (Art. 46)   | | (Art. 71)   | | (GDPR Art.30)| |
|  +-------------+ +-------------+ +-------------+ +--------------+ |
+-------------------------------------------------------------------+
|          LAYER 2: FL ORCHESTRATION (within SPE)                    |
|  +------------------+ +------------------+ +------------------+   |
|  |   Aggregation    | |     Privacy      | |   Compliance     |   |
|  | 10 FL Algorithms | | DP + RDP + SecAgg| | Purpose Limit.   |   |
|  | (Art. 50)        | | (Art. 53)        | | (Art. 53)        |   |
|  +------------------+ +------------------+ +------------------+   |
+-------------------------------------------------------------------+
|                   LAYER 3: DATA HOLDERS                            |
|  +------------------+ +------------------+ +------------------+   |
|  | Training Engine  | | FHIR Preprocess  | | Secure Comms     |   |
|  | CNN + Tabular    | | (HL7 FHIR R4)   | | E2E Encrypted    |   |
|  +------------------+ +------------------+ +------------------+   |
+-------------------------------------------------------------------+
```

---

## Federated Learning Algorithms

The framework implements **10 state-of-the-art FL aggregation strategies** with full PyTorch training:

| Algorithm | Category | Key Mechanism | Reference |
|-----------|----------|---------------|-----------|
| **FedAvg** | Baseline | Weighted model averaging | McMahan et al. (2017) |
| **FedProx** | Robustness | Proximal regularization term (mu) | Li et al. (2020) |
| **SCAFFOLD** | Variance Reduction | Control variates for drift correction | Karimireddy et al. (2020) |
| **FedNova** | Heterogeneity | Normalized averaging for unequal local steps | Wang et al. (2020) |
| **FedDyn** | Dynamic Reg. | Proximal + linear gradient correction | Acar et al. (2021) |
| **FedAdam** | Server Optimizer | Server-side Adam (momentum + adaptive LR) | Reddi et al. (2021) |
| **FedYogi** | Server Optimizer | Controlled adaptive learning rate | Reddi et al. (2021) |
| **FedAdagrad** | Server Optimizer | Server-side sum of squared gradients | Reddi et al. (2021) |
| **Per-FedAvg** | Personalization | MAML-inspired local fine-tuning | Fallah et al. (2020) |
| **Ditto** | Personalization | L2-regularized personal models | Li et al. (2021) |

All algorithms support both **IID** and **Non-IID** (Dirichlet-based) data partitioning across clients.

---

## Clinical Datasets

Validated on **6 real-world medical imaging datasets** covering major diagnostic domains:

| Dataset | Domain | Images | Classes | Task |
|---------|--------|-------:|:-------:|------|
| **Diabetic Retinopathy** | Ophthalmology | 35,126 | 5 | DR severity grading (0-4) |
| **Brain Tumor** | Neuroradiology | 7,023 | 4 | Tumor type classification |
| **Chest X-ray** | Pulmonology | 5,856 | 2 | Pneumonia detection |
| **Skin Cancer** | Dermatology | 3,297 | 2 | Benign vs. malignant |
| **Brain Tumor MRI** | Neuroradiology | 3,264 | 4 | MRI-based tumor classification |
| **ISIC** | Dermatology | 2,357 | 9 | Skin lesion classification |
| | | **56,923** | | |

The framework also includes a **synthetic tabular** healthcare dataset generator for controlled experimentation.

---

## Privacy and Compliance

### Differential Privacy

- **Renyi Differential Privacy (RDP)** accounting with 6-10x tighter bounds than naive composition
- Configurable privacy budget (epsilon) with automatic noise calibration
- Per-round tracking with cumulative budget monitoring

### EHDS Regulatory Mapping

| EHDS Article | Requirement | Framework Component |
|:------------:|-------------|---------------------|
| Art. 46 | Health Data Access Body authorization | `governance/hdab_integration.py` |
| Art. 50 | Secure Processing Environment | `orchestration/` (all training within SPE) |
| Art. 53 | Purpose limitation & minimization | `orchestration/compliance/purpose_limitation.py` |
| Art. 71 | Natural persons' right to opt out | `governance/optout_registry.py` |
| GDPR Art. 30 | Records of processing activities | `governance/compliance_logging.py` |

### Additional Privacy Features

- **Secure Aggregation**: Cryptographic protection of individual model updates (TenSEAL CKKS)
- **Gradient Clipping**: Bounded sensitivity for formal privacy guarantees
- **Byzantine Resilience**: Robust aggregation against malicious participants

---

## Installation

### Prerequisites

- Python >= 3.9
- PyTorch >= 2.0
- CUDA (optional, for GPU acceleration)

### Setup with Conda (Recommended)

```bash
git clone https://github.com/fabioliberti/FL-EHDS-FLICS2026.git
cd FL-EHDS-FLICS2026/fl-ehds-framework

conda create -n flics2026 python=3.11 -y
conda activate flics2026

pip install -e .
```

### Setup with pip

```bash
git clone https://github.com/fabioliberti/FL-EHDS-FLICS2026.git
cd FL-EHDS-FLICS2026/fl-ehds-framework

python -m venv venv
source venv/bin/activate

pip install -e .
```

### Dependencies

Core: `torch`, `numpy`, `scipy`, `scikit-learn`, `pydantic`, `cryptography`, `structlog`
CLI: `questionary`, `tqdm`
Dashboard: `streamlit`, `plotly`
Healthcare: `fhir.resources`, `hl7apy` *(optional)*

---

## Usage

### Terminal Interface (CLI)

The terminal interface provides full access to all framework features with interactive menus:

```bash
cd fl-ehds-framework
python -m terminal
```

**Available screens:**

| Menu | Function |
|------|----------|
| 1. Training | Single algorithm training with dataset selection |
| 2. Comparison | Multi-algorithm benchmark (up to 10 algorithms) |
| 3. Guided Comparison | Pre-configured clinical scenarios |
| 4. Algorithm Explorer | Detailed algorithm documentation |
| 5. Dataset Management | Browse, preview, and analyze datasets |
| 6. Privacy Dashboard | DP budget analysis and RDP accounting |
| 7. Benchmark Suite | Reproducible experiment configurations |
| 8. Byzantine Resilience | Adversarial robustness testing |

### Web Dashboard (Streamlit)

```bash
cd fl-ehds-framework
streamlit run dashboard/app.py
```

### Programmatic API

```python
from terminal.fl_trainer import ImageFederatedTrainer

trainer = ImageFederatedTrainer(
    data_dir="data/chest_xray",
    num_clients=5,
    algorithm="FedAvg",
    num_rounds=15,
    local_epochs=3,
    is_iid=False,
    alpha=0.5,
    img_size=64,
    device="cuda"  # or "cpu"
)

results = trainer.train()
# results contain: accuracy, f1, precision, recall, auc, loss per round
```

---

## Experiments

### Centralized vs. Federated Comparison

```bash
cd fl-ehds-framework
python -m experiments.centralized_vs_federated --dataset chest_xray --quick
```

Options: `--algorithms FedAvg FedProx SCAFFOLD`, `--num-rounds 15`, `--seeds 42 123 456`

### Imaging Benchmarks

```bash
python -m benchmarks.run_imaging_experiments --dataset chest_xray --algorithms FedAvg FedProx
```

### Full Benchmark Suite

```bash
python -m benchmarks.run_experiments          # Tabular data, 9 algorithms
python -m benchmarks.run_extended_experiments  # Scalability and convergence
python -m benchmarks.run_heterogeneity_experiments  # Non-IID analysis
```

All experiments auto-save results (JSON, CSV, LaTeX tables, PNG plots) to `results/`.

---

## Benchmark Results

### Tabular Data (Synthetic, 3 seeds, 30 rounds, 5 hospitals)

| Algorithm | Accuracy | F1 | AUC |
|-----------|:--------:|:--:|:---:|
| FedAvg (IID) | 60.5 +/- 0.02 | 0.62 +/- 0.02 | 0.66 +/- 0.01 |
| FedAvg (Non-IID) | 60.9 +/- 0.02 | 0.61 +/- 0.01 | 0.66 +/- 0.01 |
| FedProx (mu=0.1) | 60.9 +/- 0.02 | 0.62 +/- 0.01 | 0.66 +/- 0.01 |
| SCAFFOLD | 60.5 +/- 0.01 | 0.61 +/- 0.02 | 0.66 +/- 0.01 |
| FedNova | 60.7 +/- 0.02 | 0.62 +/- 0.01 | 0.66 +/- 0.01 |
| DP (epsilon=10) | 55.7 +/- 0.01 | 0.61 +/- 0.04 | 0.55 +/- 0.03 |
| DP (epsilon=1) | 55.1 +/- 0.01 | 0.59 +/- 0.04 | 0.55 +/- 0.01 |

### Clinical Imaging (Chest X-ray, Non-IID, 5 hospitals)

| Approach | Accuracy | F1 | AUC | Time |
|----------|:--------:|:--:|:---:|:----:|
| Centralized (pooled) | 95.4% | 0.939 | 0.985 | 154s |
| FedAvg (Non-IID) | 91.2% | 0.890 | 0.960 | 145s |
| **Gap** | **4.2 pp** | **0.049** | **0.025** | |

---

## Project Structure

```
FL-EHDS-FLICS2026/
|-- main.tex                        # Conference paper (LaTeX, IEEE format)
|-- figures/                        # Paper figures
|-- fl-ehds-framework/              # Framework source code
    |-- core/                       # Core FL engine (32 modules)
    |   |-- fl_algorithms.py        # Algorithm implementations
    |   |-- secure_aggregation.py   # CKKS-based secure aggregation
    |   |-- byzantine_resilience.py # Byzantine fault tolerance
    |   +-- ...
    |-- orchestration/              # Layer 2: FL orchestration
    |   |-- aggregation/            # FedAvg, FedProx strategies
    |   |-- privacy/                # DP, gradient clipping, SecAgg
    |   +-- compliance/             # Purpose limitation enforcement
    |-- governance/                 # Layer 1: EHDS governance
    |   |-- hdab_integration.py     # Health Data Access Body API
    |   |-- optout_registry.py      # Art. 71 opt-out management
    |   +-- compliance_logging.py   # GDPR Art. 30 audit trails
    |-- data_holders/               # Layer 3: Data holder components
    |-- terminal/                   # CLI interface (questionary + tqdm)
    |   |-- fl_trainer.py           # 10 FL algorithms + image training
    |   +-- screens/                # Interactive terminal screens
    |-- dashboard/                  # Streamlit web interface
    |-- models/                     # HealthcareCNN + model zoo
    |-- benchmarks/                 # Reproducible experiment scripts
    |-- experiments/                # Centralized vs. federated comparison
    |-- tests/                      # Unit test suite (pytest)
    |-- data/                       # Clinical datasets (not in repo)
    +-- results/                    # Auto-generated experiment outputs
```

**Codebase**: ~75,000 lines of Python across 106 modules.

---

## Reproducibility

All experiments are fully reproducible with fixed random seeds. Each run generates:

| Output | Format | Description |
|--------|--------|-------------|
| `results.json` | JSON | Complete results, histories, configuration |
| `summary_results.csv` | CSV | Final metrics with standard deviations |
| `history_all_metrics.csv` | CSV | Per-round metrics (Acc, Loss, F1, Prec, Rec, AUC) |
| `table_results.tex` | LaTeX | Publication-ready table |
| `plot_convergence_*.png` | PNG | Convergence curves for all metrics |
| `plot_metrics_comparison.png` | PNG | Bar chart of final metric comparison |

---

## Citation

If you use this framework in your research, please cite:

```bibtex
@inproceedings{liberti2026flehds,
  title     = {{FL-EHDS}: A Privacy-Preserving Federated Learning Framework
               for the European Health Data Space},
  author    = {Liberti, Fabio},
  booktitle = {Proceedings of the 2nd International Conference on Federated
               Learning and Intelligent Computing Systems (FLICS)},
  year      = {2026},
  address   = {Valencia, Spain}
}
```

---

## References

1. McMahan, B. et al. "Communication-Efficient Learning of Deep Networks from Decentralized Data." AISTATS (2017).
2. Li, T. et al. "Federated Optimization in Heterogeneous Networks." MLSys (2020).
3. Karimireddy, S.P. et al. "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning." ICML (2020).
4. Wang, J. et al. "Tackling the Objective Inconsistency Problem in Heterogeneous Federated Optimization." NeurIPS (2020).
5. Acar, D.A.E. et al. "Federated Learning Based on Dynamic Regularization." ICLR (2021).
6. Reddi, S. et al. "Adaptive Federated Optimization." ICLR (2021).
7. Fallah, A. et al. "Personalized Federated Learning with Moreau Envelopes." NeurIPS (2020).
8. Li, T. et al. "Ditto: Fair and Robust Federated Learning Through Personalization." ICML (2021).
9. European Union. "Regulation (EU) 2025/327 on the European Health Data Space." (2025).
10. Dwork, C. & Roth, A. "The Algorithmic Foundations of Differential Privacy." Foundations and Trends in Theoretical Computer Science (2014).

---

## License

This project is licensed under the [Apache License 2.0](https://opensource.org/licenses/Apache-2.0).

---

## Author

**Fabio Liberti, PhD**
Department of Computer Science
Universitas Mercatorum, Rome, Italy
[fabio.liberti@unimercatorum.it](mailto:fabio.liberti@unimercatorum.it) | [ORCID: 0000-0003-3019-5411](https://orcid.org/0000-0003-3019-5411)

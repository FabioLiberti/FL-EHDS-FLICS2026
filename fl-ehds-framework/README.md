# FL-EHDS Framework

**Privacy-Preserving Federated Learning for the European Health Data Space**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](../LICENSE)
![Code](https://img.shields.io/badge/Code-~40K%20lines-2ea44f)
![Modules](https://img.shields.io/badge/Modules-159-2ea44f)

This directory contains the full implementation of the FL-EHDS framework, the open-source reference implementation accompanying the paper:

> **FL-EHDS: A Privacy-Preserving Federated Learning Framework for the European Health Data Space**
> Fabio Liberti — *IEEE FLICS 2026* (Valencia, Spain, June 9--12, 2026)

For the project overview, architecture diagram, and benchmark results summary, see the [root README](../README.md).

---

## Paper and Supplementary Material

The framework supports all experiments reported in the paper (9 pages, IEEE IEEEtran format) and the accompanying supplementary material (29 pages):

| Document | Content |
|:---------|:--------|
| **Main paper** | Three-layer architecture, 7-algorithm comparison on 3 tabular + 3 imaging datasets, DP ablation, opt-out simulation, key findings |
| **Supplementary** | 8 algorithm pseudocodes, 19-dataset landscape (Table S-I), extended tabular results (1,740+ experiments), heterogeneity/scalability sweeps, 10-seed Wilcoxon validation, confusion matrix analysis, communication cost analysis, extended threat model, RDP composition comparison |

**Key experimental findings** (from 1,760+ total experiments):

| Finding | Evidence |
|:--------|:---------|
| Personalised FL narrows gap to **6.6 pp** | Ditto 75.1% vs. centralised 81.7% (Heart Disease UCI) |
| Algorithm choice yields up to **12.6 pp** | Ditto 75.1% vs. FedAvg 62.5% (Heart Disease) |
| **HPFL** outperforms FedAvg on **all 3 tabular datasets** | p = 0.004, 0.002, 0.031 (Wilcoxon, 10-seed); pooled p < 0.001 |
| DP at epsilon = 10 imposes **< 2 pp cost** | Across PTB-XL and Cardiovascular datasets |
| Art. 71 opt-out up to 30% is **negligible** | < 1 pp drop on adequately-sized datasets |
| PTB-XL validates **European FL** | 92.5% accuracy (HPFL), 5-class ECG, Jain 0.999 |

---

## Quick Start

```bash
# Install
conda create -n flehds python=3.11 -y && conda activate flehds
pip install -e .

# Terminal CLI (11 screens)
python -m terminal

# Web Dashboard
streamlit run dashboard/app_v4.py

# Run tabular benchmark (baseline: 105 experiments, ~45 min)
python -m benchmarks.run_tabular_optimized

# Run imaging experiments (4 algos x 3 seeds, ~4h)
python -m benchmarks.run_imaging_extended
```

---

## FL Algorithms (17)

All 17 algorithms are implemented with real PyTorch training, spanning six categories from foundational methods to ICML 2024 and ICLR 2025 advances:

| # | Algorithm | Venue | Category | Key Mechanism |
|:-:|:----------|:------|:---------|:--------------|
| 1 | FedAvg | AISTATS 2017 | Baseline | Weighted model averaging |
| 2 | FedProx | MLSys 2020 | Non-IID | Proximal regularisation (mu) |
| 3 | SCAFFOLD | ICML 2020 | Non-IID | Control variates for drift correction |
| 4 | FedNova | NeurIPS 2020 | Non-IID | Normalised averaging |
| 5 | FedDyn | ICLR 2021 | Non-IID | Dynamic regularisation |
| 6 | FedAdam | ICLR 2021 | Adaptive | Server-side Adam momentum |
| 7 | FedYogi | ICLR 2021 | Adaptive | Controlled adaptive learning rate |
| 8 | FedAdagrad | ICLR 2021 | Adaptive | Server-side gradient accumulation |
| 9 | Per-FedAvg | NeurIPS 2020 | Personalisation | MAML-based meta-learning |
| 10 | Ditto | ICML 2021 | Personalisation | L2-regularised personal models |
| 11 | FedLC | ICML 2022 | Label skew | Logit calibration |
| 12 | FedSAM | ICML 2022 | Generalisation | Sharpness-aware flat minima |
| 13 | FedDecorr | ICLR 2023 | Representation | Decorrelation against dimensional collapse |
| 14 | FedSpeed | ICLR 2023 | Efficiency | Fewer communication rounds |
| 15 | FedExP | ICLR 2023 | Server-side | POCS-based step size |
| 16 | **FedLESAM** | **ICML 2024** | **Generalisation** | **Globally-guided sharpness-aware optimisation (Spotlight)** |
| 17 | **HPFL** | **ICLR 2025** | **Personalisation** | **Shared backbone + personalised classifiers** |

**Byzantine resilience** (6 methods): Krum, Multi-Krum, Trimmed Mean, Coordinate-wise Median, Bulyan, FLTrust -- defending against up to f < n/3 adversarial clients.

**Composable strategies**: FedLC and FedDecorr can augment any base aggregation algorithm.

---

## Dataset Coverage

The framework supports **19 healthcare datasets** across four modalities. Eight are experimentally evaluated in the paper:

### Tabular Clinical (HealthcareMLP, ~2.9K--10K params)

| Dataset | Samples | Features | Classes | FL Partition |
|:--------|--------:|---------:|:-------:|:-------------|
| PTB-XL ECG | 21,799 | 9 | 5 | Natural (52 German recording sites) |
| Cardiovascular Disease | 70,000 | 11 | 2 | Dirichlet (alpha = 0.5) |
| Diabetes 130-US | 101,766 | 22 | 2 | Dirichlet (alpha = 0.5) |
| Heart Disease UCI | 920 | 13 | 2 | Natural (4 international hospitals) |
| Breast Cancer Wisconsin | 569 | 30 | 2 | Dirichlet (alpha = 0.5) |

### Medical Imaging (ResNet-18, ~11.2M params)

| Dataset | Samples | Classes | FL Partition |
|:--------|--------:|:-------:|:-------------|
| Chest X-ray | 5,856 | 2 | Dirichlet (alpha = 0.5) |
| Brain Tumor MRI | 7,023 | 4 | Dirichlet (alpha = 0.5) |
| Skin Cancer | 3,297 | 2 | Dirichlet (alpha = 0.5) |

Additional supported datasets (11): Stroke Prediction, CDC Diabetes BRFSS, CKD UCI, Cirrhosis Mayo, Synthea FHIR R4, SMART Bulk FHIR, FHIR R4 Synthetic, OMOP-CDM Harmonized, Diabetic Retinopathy, Brain Tumor MRI alt., ISIC Skin Lesions. Full details in Supplementary Material, Table S-I.

---

## Directory Structure

```
fl-ehds-framework/
|
|-- governance/                     # Layer 1: EHDS Governance
|   |-- hdab_integration.py         #   Health Data Access Body API (OAuth2/mTLS)
|   |-- data_permits.py             #   Data permit lifecycle (Art. 53)
|   |-- optout_registry.py          #   Art. 71 opt-out registry (LRU-cached)
|   |-- cross_border.py             #   Multi-HDAB coordination (10 EU profiles)
|   +-- compliance_logging.py       #   GDPR Art. 30 audit trails (7-year)
|
|-- orchestration/                  # Layer 2: FL Orchestration (SPE)
|   |-- aggregation/                #   17 FL algorithms (FedAvg -> HPFL)
|   |-- privacy/
|   |   |-- differential_privacy.py #   DP-SGD with RDP accounting
|   |   |-- gradient_clipping.py    #   L2 gradient norm clipping
|   |   +-- secure_aggregation.py   #   Pairwise masking + ECDH + Shamir
|   |-- byzantine/                  #   6 resilience methods + attack simulation
|   +-- compliance/
|       +-- purpose_limitation.py   #   EHDS Art. 53 enforcement
|
|-- data_holders/                   # Layer 3: Data Holders
|   |-- training_engine.py          #   Adaptive local training (CUDA/MPS/CPU)
|   |-- fhir_preprocessing.py       #   HL7 FHIR R4 transformation pipeline
|   |-- omop_harmonization.py       #   OMOP-CDM vocabulary harmonisation
|   +-- secure_communication.py     #   E2E encrypted gradients (AES-256-GCM)
|
|-- models/                         # Neural Network Architectures
|   |-- healthcare_mlp.py           #   MLP (~2.9K-10K) + DeepMLP (~110K params)
|   |-- healthcare_cnn.py           #   5-block CNN, GroupNorm (~12M params)
|   +-- healthcare_resnet.py        #   ResNet-18, GroupNorm + FedBN (~11.2M)
|
|-- terminal/                       # Terminal CLI Interface (11 screens)
|   |-- __main__.py                 #   Entry point
|   |-- fl_trainer.py               #   FL trainer (17 algos + imaging)
|   +-- screens/                    #   Training, Comparison, Privacy, ...
|
|-- dashboard/                      # Streamlit Web Dashboard
|   |-- app_v4.py                   #   Main dashboard
|   +-- real_trainer_bridge.py      #   Terminal-Web bridge
|
|-- benchmarks/                     # Reproducible Experiment Suite
|   |-- run_tabular_optimized.py    #   Baseline (105 exps, 7 algos x 3 DS x 5 seeds)
|   |-- run_tabular_sweep.py        #   Heterogeneity + scaling + lr (1,125 exps)
|   |-- run_tabular_dp.py           #   DP ablation (180 exps, 4 epsilon levels)
|   |-- run_tabular_seeds10.py      #   10-seed validation (105 exps)
|   |-- run_tabular_optout.py       #   Art. 71 opt-out impact (225 exps)
|   |-- run_tabular_deep_mlp.py     #   Deep MLP differentiation (70 exps)
|   |-- run_imaging_extended.py     #   Chest X-ray (4 algos x 3 seeds)
|   |-- run_imaging_multi.py        #   Brain Tumor + Skin Cancer
|   |-- run_confusion_matrix_bc.py  #   Breast Cancer confusion matrix (10 seeds)
|   |-- analyze_tabular_extended.py #   Tables, figures, statistical tests
|   +-- paper_results*/             #   Auto-generated results and figures
|
|-- experiments/                    # Specialized Experiments
|   +-- centralized_vs_federated/   #   Centralised vs. FL comparison suite
|
|-- data/                           # Clinical Datasets (auto-downloaded, not in repo)
|-- tests/                          # Unit Tests (pytest)
|-- config/                         # YAML configuration
+-- setup.py                        # Package configuration
```

---

## Reproducing Paper Experiments

All experiments reported in the paper are fully reproducible. Results, checkpoints, and analysis outputs are auto-saved to `benchmarks/paper_results_tabular/` and `benchmarks/paper_results_delta/`.

### Tabular (1,740+ experiments)

```bash
# Phase 1 -- Baseline comparison (105 exps, ~45 min)
python -m benchmarks.run_tabular_optimized

# Phase 2 -- Multi-phase sweep: heterogeneity, client scaling, lr (1,125 exps, ~4.5h)
python -m benchmarks.run_tabular_sweep --phase all

# Phase 3 -- Differential privacy ablation (180 exps, ~1.5h)
python -m benchmarks.run_tabular_dp

# Phase 4 -- 10-seed statistical validation (105 exps, ~40 min)
python -m benchmarks.run_tabular_seeds10

# Phase 5 -- Article 71 opt-out impact (225 exps, ~1.5h)
python -m benchmarks.run_tabular_optout

# Phase 6 -- Deep MLP differentiation (70 exps, ~1.5h)
python -m benchmarks.run_tabular_deep_mlp

# Analysis -- Generates all tables, figures, and statistical tests
python -m benchmarks.analyze_tabular_extended
```

### Imaging (~20 experiments)

```bash
# Chest X-ray (4 algos x 3 seeds, ~4h)
python -m benchmarks.run_imaging_extended

# Brain Tumor + Skin Cancer (2 algos x 1 seed, ~2.5h)
python -m benchmarks.run_imaging_multi --light

# Confusion matrices
python -m benchmarks.run_confusion_matrix_chest
python -m benchmarks.run_confusion_matrix_bc
```

### Per-Dataset Configuration

| Dataset | lr | Batch | Rounds | Clients | Local Epochs | Model |
|:--------|:--:|:-----:|:------:|:-------:|:------------:|:------|
| PTB-XL ECG | 0.005 | 64 | 30 | 5 | 3 | HealthcareMLP (~2.9K params) |
| Cardiovascular | 0.01 | 64 | 25 | 5 | 3 | HealthcareMLP (~10K params) |
| Breast Cancer | 0.001 | 16 | 40 | 3 | 1 | HealthcareMLP (~10K params) |
| Heart Disease | 0.01 | 32 | 20 | 4 | 3 | HealthcareMLP (~10K params) |
| Diabetes | 0.01 | 64 | 25 | 5 | 3 | HealthcareMLP (~10K params) |
| Chest X-ray | 0.001 | 32 | 20 | 5 | 3 | ResNet-18 (~11.2M params) |
| Brain Tumor | 0.0005 | 32 | 10 | 5 | 3 | ResNet-18 (~11.2M params) |
| Skin Cancer | 0.001 | 32 | 20 | 5 | 3 | ResNet-18 (~11.2M params) |

All tabular: Adam optimiser, early stopping (patience = 6). Imaging: GroupNorm (replacing BatchNorm for FL stability), FedBN, class-weighted loss, mixed precision.

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
      epsilon: 10.0
      delta: 1e-5
    gradient_clipping:
      max_norm: 1.0
    secure_aggregation:
      enabled: true

data_holders:
  training:
    batch_size: 64
    local_epochs: 3
    learning_rate: 0.005
```

---

## Testing

```bash
pytest tests/ -v
pytest tests/ --cov=fl_ehds --cov-report=html
```

---

## Citation

```bibtex
@inproceedings{liberti2026flehds,
  title     = {{FL-EHDS}: A Privacy-Preserving Federated Learning Framework
               for the {European Health Data Space}},
  author    = {Liberti, Fabio},
  booktitle = {Proceedings of the IEEE International Conference on
               Federated Learning in Integrated Computing and Services (FLICS)},
  year      = {2026},
  address   = {Valencia, Spain}
}
```

---

## Author

**Fabio Liberti** -- Department of Computer Science, Universitas Mercatorum, Rome, Italy
[ORCID: 0000-0003-3019-5411](https://orcid.org/0000-0003-3019-5411)

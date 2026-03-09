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

The framework supports all experiments reported in the paper (9 pages, IEEE IEEEtran format) and the accompanying supplementary material (98 pages, 7 appendices, 109 tables, 64 figures):

| Document | Content |
|:---------|:--------|
| **Main paper** | Three-layer architecture, 7-algorithm primary benchmark, 17-algorithm non-IID sweep, 3 tabular + 3 imaging datasets, DP ablation, 720 governance hypothesis tests, compound EHDS stress, opt-out simulation |
| **Supplementary** | 15 algorithm pseudocodes (S1–S15), 19-dataset landscape (Table S-I), extended tabular results, 10 cascading analysis phases, heterogeneity/scalability sweeps, 10-seed Wilcoxon validation, confusion matrix analysis, communication cost analysis, Byzantine resilience on imaging, extended threat model, RDP composition comparison, EHDS governance validation (Appendix G) |

**Key experimental findings** (from 6,004+ total experiments):

| Finding | Evidence |
|:--------|:---------|
| Personalisation gains up to **26.8 pp** | BC: Ditto 79.1% vs. FedAvg 52.3%; Brain Tumor: +23.5 pp |
| Best-FL gap to centralised **<= 2.4 pp** | PTB-XL: HPFL 92.5% vs. centralised 92.6% |
| **HPFL** outperforms FedAvg on **all 3 tabular** datasets | p = 0.004, 0.002, 0.031 (Wilcoxon, 10-seed); pooled p < 0.001 |
| DP at epsilon = 10 imposes **< 2 pp cost** | Across PTB-XL and Cardiovascular datasets |
| Full EHDS compliance costs **-0.7 pp** | Ditto under data minimisation + opt-out + DP (p < 0.001) |
| Compound stress: personalisation wins **81%** | +9.6 pp mean over FedAvg (216 experiments) |
| Governance overhead **< 1.1%** per round | 18 experiments; within measurement noise |
| PTB-XL validates **European FL** | 92.5% accuracy (HPFL), 5-class ECG, 52 sites, Jain 0.999 |

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

Additional supported datasets (11): Stroke Prediction (5,110), CDC Diabetes BRFSS (253,680), CKD UCI (400), Cirrhosis Mayo (418), Synthea FHIR R4 (1,180), SMART Bulk FHIR (120), FHIR R4 Synthetic (configurable), OMOP-CDM Harmonized (configurable), Diabetic Retinopathy (35,126 images, 5-class), Brain Tumor MRI alt. (3,264 images, 4-class), ISIC Skin Lesions (2,357 images, 9-class). Full details in Supplementary Material, Table S-I.

---

## Directory Structure

```
fl-ehds-framework/
|
|-- core/                           # Core FL Engine (31 modules)
|   |-- fl_algorithms.py            #   17 FL algorithms (FedAvg -> HPFL)
|   |-- personalized_fl.py          #   Ditto, Per-FedAvg, HPFL
|   |-- byzantine_resilience.py     #   6 defence methods + 5 attack types
|   |-- secure_aggregation.py       #   Pairwise masking + ECDH + Shamir
|   |-- gradient_compression.py     #   Top-k sparsification
|   |-- vertical_fl.py              #   Vertical (split) federated learning
|   |-- continual_fl.py             #   Continual learning + EWC
|   |-- fairness_fl.py              #   q-FedAvg, FedMinMax
|   |-- async_fl.py                 #   Asynchronous FL with staleness weighting
|   |-- hierarchical_fl.py          #   Hierarchical aggregation
|   |-- fhir_integration.py         #   FHIR R4 integration
|   |-- omop_cdm.py                 #   OMOP-CDM harmonisation
|   +-- ...                         #   (+19 modules: monitoring, caching, etc.)
|
|-- governance/                     # Layer 1: EHDS Governance (18 modules)
|   |-- hdab_integration.py         #   Health Data Access Body API (OAuth2/mTLS)
|   |-- data_permits.py             #   Art. 53 lifecycle (PENDING->ACTIVE->EXPIRED)
|   |-- optout_registry.py          #   Art. 71 opt-out (record/patient/dataset)
|   |-- data_minimization.py        #   GDPR data minimisation enforcement
|   |-- jurisdiction_privacy.py     #   Cross-border DP budget coordination
|   |-- compliance_logging.py       #   GDPR Art. 30 audit trails (7-year)
|   |-- secure_processing.py        #   Art. 50 SPE boundary
|   +-- ...                         #   (+11 modules: fees, routing, IHE bridge)
|
|-- orchestration/                  # Layer 2: FL Orchestration (SPE)
|   |-- aggregation/                #   FedAvg, FedProx base implementations
|   |-- privacy/
|   |   |-- differential_privacy.py #   DP-SGD with RDP accounting
|   |   |-- gradient_clipping.py    #   L2 gradient norm clipping (C = 1.0)
|   |   +-- secure_aggregation.py   #   Pairwise masking + ECDH + Shamir
|   +-- compliance/
|       +-- purpose_limitation.py   #   EHDS Art. 53 enforcement
|
|-- data_holders/                   # Layer 3: Data Holders
|   |-- training_engine.py          #   Adaptive local training (CUDA/MPS/CPU)
|   |-- fhir_preprocessing.py       #   HL7 FHIR R4 transformation pipeline
|   +-- secure_communication.py     #   E2E encrypted gradients (AES-256-GCM)
|
|-- models/                         # Neural Network Architectures
|   |-- model_zoo.py                #   HealthcareMLP, DeepMLP, TabNet, CNN, ResNet-18
|   +-- cnn_fl_trainer.py           #   Imaging FL trainer (GroupNorm + FedBN)
|
|-- data/                           # Dataset Loaders (13 loaders, 19 datasets)
|   |-- real_datasets.py            #   Unified loader interface
|   |-- ptbxl_loader.py             #   PTB-XL ECG (21,799, 5-class, 52 EU sites)
|   |-- cardiovascular_loader.py    #   Cardiovascular Disease (70,000)
|   |-- diabetes_loader.py          #   Diabetes 130-US (101,766)
|   |-- heart_disease_loader.py     #   Heart Disease UCI (920, 4 hospitals)
|   |-- breast_cancer_loader.py     #   Breast Cancer Wisconsin (569)
|   |-- cdc_diabetes_loader.py      #   CDC Diabetes BRFSS (253,680)
|   |-- stroke_loader.py            #   Stroke Prediction (5,110)
|   |-- ckd_loader.py               #   CKD UCI (400)
|   |-- cirrhosis_loader.py         #   Cirrhosis Mayo (418)
|   |-- synthea_fhir_loader.py      #   Synthea FHIR R4 (1,180)
|   +-- smart_fhir_loader.py        #   SMART Bulk FHIR (120)
|
|-- terminal/                       # Terminal CLI (15 screens)
|   |-- __main__.py                 #   Entry point
|   |-- fl_trainer.py               #   FL trainer (17 algos + imaging)
|   |-- screens/                    #   Training, Byzantine, Privacy, Governance, ...
|   +-- training/                   #   Federated + centralised training backends
|
|-- dashboard/                      # Streamlit Web Dashboard (14 modules)
|   |-- app_v4.py                   #   Main dashboard
|   +-- ...                         #   Training, governance, paper experiments pages
|
|-- benchmarks/                     # Reproducible Experiment Suite (83 scripts)
|   |-- run_tabular_optimized.py    #   Baseline (105 exps, 7 algos x 3 DS x 5 seeds)
|   |-- run_tabular_sweep.py        #   Heterogeneity + scaling + lr (1,125 exps)
|   |-- run_tabular_dp.py           #   DP ablation (180 exps, 4 epsilon levels)
|   |-- run_tabular_seeds10.py      #   10-seed validation (105 exps)
|   |-- run_tabular_optout.py       #   Art. 71 opt-out impact (225 exps)
|   |-- run_tabular_deep_mlp.py     #   Deep MLP differentiation (70 exps)
|   |-- run_imaging_*.py            #   14 imaging experiment scripts
|   |-- run_governance_*.py         #   4 EHDS governance validation scripts (720 exps)
|   |-- run_analysis_cascade*.py    #   10 cascading analysis phases
|   |-- run_thesis_robustness.py    #   Thesis robustness (516 exps)
|   |-- analyze_tabular_extended.py #   Tables, figures, statistical tests
|   |-- paper_results_tabular/      #   Tabular checkpoints (247 files)
|   |-- paper_results_delta/        #   DP/delta checkpoints (26 files)
|   |-- paper_results/              #   Imaging checkpoints (16 files)
|   +-- results_optout/             #   Opt-out experiment results
|
|-- notebooks/                      # Jupyter/Colab Notebooks (9)
|   |-- fl_ehds_demo.ipynb          #   Framework demo
|   +-- colab_imaging_*.ipynb       #   8 imaging experiment notebooks
|
|-- experiments/                    # Structured Experiments
|   +-- centralized_vs_federated/   #   Centralised vs. FL comparison suite
|
|-- docs/                           # Documentation
|   |-- FL-ALGORITHMS.md            #   Algorithm catalogue
|   |-- FL-EHDS-Framework.md        #   Architecture details
|   +-- PRISMA/                     #   Systematic review documents
|
|-- tests/                          # Unit Tests (pytest, 5 test modules)
|-- config/                         # YAML configuration
|-- deployment/                     # Docker + K8s + Ray configs
+-- setup.py                        # Package configuration
```

---

## Reproducing Paper Experiments

All experiments reported in the paper are fully reproducible. Results, checkpoints, and analysis outputs are auto-saved to `benchmarks/paper_results_tabular/` and `benchmarks/paper_results_delta/`. All scripts support SIGINT-safe interruption with atomic checkpointing (mkstemp + fsync + replace + .bak).

### Tabular (1,810+ experiments)

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

### Imaging (~20+ experiments)

```bash
# Chest X-ray (4 algos x 3 seeds, ~4h)
python -m benchmarks.run_imaging_extended

# Brain Tumor + Skin Cancer (2 algos x 1 seed, ~2.5h)
python -m benchmarks.run_imaging_multi --light

# Confusion matrices
python -m benchmarks.run_confusion_matrix_chest
python -m benchmarks.run_confusion_matrix_bc
```

### EHDS Governance Validation (720+ experiments)

```bash
# Governance hypothesis testing -- H1, H2, H3
python -m benchmarks.run_governance_hypotheses
python -m benchmarks.run_governance_hypotheses_cv

# Governance overhead benchmarking (18 exps)
python -m benchmarks.run_governance_validation

# Extended governance validation
python -m benchmarks.run_governance_extended
```

### Thesis Robustness and Cascading Analysis

```bash
# Thesis robustness (516 exps: lambda, data fraction, compound stress)
python -m benchmarks.run_thesis_robustness

# Cascading analysis phases (Cascades 2-10)
python -m benchmarks.run_analysis_cascade2   # through cascade10
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

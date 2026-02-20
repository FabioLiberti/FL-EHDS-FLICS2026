<p align="center">
  <h1 align="center">FL-EHDS</h1>
  <p align="center">
    <strong>A Privacy-Preserving Federated Learning Framework<br/>for the European Health Data Space</strong>
  </p>
</p>

<p align="center">
  <a href="https://www.flics-conference.org/"><img src="https://img.shields.io/badge/FLICS%202026-Candidate-success?style=for-the-badge" alt="FLICS 2026"/></a>
</p>

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python 3.10+"/></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch 2.0+"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue?style=flat-square" alt="MIT License"/></a>
  <img src="https://img.shields.io/badge/Code-~40K%20lines-2ea44f?style=flat-square" alt="~40K lines"/>
  <img src="https://img.shields.io/badge/Modules-159-2ea44f?style=flat-square" alt="159 modules"/>
  <img src="https://img.shields.io/badge/Experiments-1%2C760%2B-orange?style=flat-square" alt="1,760+ experiments"/>
</p>

<p align="center">
  <strong>IEEE 2nd International Conference on Federated Learning and Intelligent Computing Systems - FLICS2026</strong><br/>
  <em>(Valencia, Spain — June 9–12, 2026)</em>
</p>

---


<p align="center">
  <em>FL-EHDS represents the meeting point between federated artificial intelligence and European health data governance:<br/>a framework that translates the regulatory complexity of the EHDS into an operational, federated, and secure computational architecture.<br/>By bridging the gap between regulation and technology, FL-EHDS demonstrates that data sovereignty is not an obstacle to collaborative research, but its strongest foundation.</em>
</p>


---

<p align="center">
  <img src="paper/figures/fig1_composite_final.png" alt="FL-EHDS Architecture" width="100%"/>
</p>

<p align="center"><sub>
<strong>Figure 1.</strong> FL-EHDS composite architecture. <strong>(a)</strong> Three-layer compliance framework — Layer 1 (Governance) manages HDAB integration, data permit authorisation, and Article 71 opt-out registries; Layer 2 (FL Orchestration) operates within the Secure Processing Environment with gradient aggregation, differential privacy, and GDPR-compliant audit logging; Layer 3 (Data Holders) implements local model computation with raw health data never leaving institutional boundaries. <strong>(b)</strong> EHDS interoperability pipeline — heterogeneous sources across 27 Member States flow through terminology harmonisation (SNOMED CT, ICD-10, LOINC, ATC, UCUM), interoperability standards (FHIR R4, OMOP CDM, IHE profiles), and security/compliance layers before reaching the FL training engine.
</sub></p>

---

## Table of Contents

<table>
<tr>
<td width="50%" valign="top">

- [Abstract](#abstract)
- [Motivation](#motivation)
- [Key Contributions](#key-contributions)
- [Architecture](#architecture)
- [Algorithm Catalogue](#algorithm-catalogue)
- [Dataset Coverage](#dataset-coverage)
- [Experimental Highlights](#experimental-highlights)

</td>
<td width="50%" valign="top">

- [Privacy and Compliance](#privacy-and-compliance)
- [Installation](#installation)
- [Usage](#usage)
- [Reproducing Experiments](#reproducing-experiments)
- [Repository Structure](#repository-structure)
- [Citation](#citation)
- [License](#license)

</td>
</tr>
</table>

---

## Abstract

**FL-EHDS** is a three-layer compliance framework that bridges the technology–governance divide for cross-border health analytics under the [European Health Data Space (EHDS)](https://health.ec.europa.eu/ehealth-digital-health-and-care/european-health-data-space_en), Regulation (EU) 2025/327. The framework integrates **17 federated learning algorithms** (2017–2025, including ICML 2024 Spotlight and ICLR 2025 advances) with EHDS governance mechanisms — Health Data Access Bodies (HDABs), data permits, citizen opt-out registries — and data holder components for adaptive training with FHIR R4 preprocessing and OMOP-CDM harmonisation.

Experimental validation across **1,760+ experiments** on 8 tabular clinical and medical imaging datasets demonstrates that personalised FL narrows the centralised–federated accuracy gap to **6.6 percentage points** while preserving full data sovereignty, and that algorithm selection produces up to **12.6 pp** accuracy differences on heterogeneous clinical data. Our evidence synthesis reveals that **unresolved regulatory questions** — gradient data classification under GDPR, cross-border privacy budget harmonisation — constitute the critical adoption blocker, not technical limitations.

---

## Motivation

The EHDS mandates cross-border secondary use of health data across 27 EU Member States by 2029, yet fewer than one in four FL implementations achieve sustained production deployment in healthcare (Fröhlich et al., JMIR 2025). The dominant barriers are not purely technical: unresolved legal questions — gradient data classification under GDPR, cross-border privacy budget harmonisation, controller/processor allocation — create compliance uncertainties that no engineering solution alone can resolve.

Existing FL frameworks provide robust distributed training but lack EHDS-specific governance. Legal analyses examine GDPR constraints but abstract from implementation feasibility. Policy documents assess Member State readiness but do not integrate FL technical considerations.

**No existing work provides an integrated framework addressing all three dimensions**: systematic barrier evidence, technical implementation with state-of-the-art algorithms, and EHDS governance operationalisation — a gap confirmed by recent systematic reviews of FL frameworks for biomedical research (Chavero-Diez et al., NAR Genomics 2026).

<table>
<tr>
<th align="center">Dimension</th>
<th align="center">FL-EHDS</th>
<th align="center">Flower</th>
<th align="center">NVIDIA FLARE</th>
<th align="center">TFF</th>
</tr>
<tr><td>FL Algorithms</td><td align="center"><strong>17 built-in</strong></td><td align="center">12+ strategies</td><td align="center">5 built-in</td><td align="center">3 built-in</td></tr>
<tr><td>Byzantine Resilience</td><td align="center"><strong>6 methods</strong></td><td align="center">4 methods</td><td align="center">—</td><td align="center">—</td></tr>
<tr><td>Differential Privacy</td><td align="center">Central + Local</td><td align="center">Central + Local</td><td align="center">Built-in</td><td align="center">Adaptive clip.</td></tr>
<tr><td>Secure Aggregation</td><td align="center">Pairwise + HE</td><td align="center">SecAgg+</td><td align="center">Built-in + HE</td><td align="center">Mask-based</td></tr>
<tr><td>EHDS Governance</td><td align="center"><strong>Full</strong></td><td align="center">—</td><td align="center">—</td><td align="center">—</td></tr>
<tr><td>HDAB Integration</td><td align="center"><strong>Yes</strong></td><td align="center">—</td><td align="center">—</td><td align="center">—</td></tr>
<tr><td>Data Permits (Art. 53)</td><td align="center"><strong>Yes</strong></td><td align="center">—</td><td align="center">—</td><td align="center">—</td></tr>
<tr><td>Opt-out (Art. 71)</td><td align="center"><strong>Yes</strong></td><td align="center">—</td><td align="center">—</td><td align="center">—</td></tr>
<tr><td>Healthcare Standards</td><td align="center"><strong>FHIR R4 + OMOP</strong></td><td align="center">MONAI</td><td align="center">MONAI</td><td align="center">—</td></tr>
</table>

---

## Key Contributions

<table>
<tr>
<td width="60"><strong>C1</strong></td>
<td><strong>Barrier Taxonomy.</strong> Systematic evidence synthesis of 47 documents (PRISMA methodology, GRADE-CERQual confidence assessment) identifying legal uncertainties as the critical adoption blocker.</td>
</tr>
<tr>
<td><strong>C2</strong></td>
<td><strong>FL-EHDS Framework.</strong> Three-layer reference architecture mapping identified barriers to governance-aware mitigation strategies, designed for incremental deployment during the 2025–2031 EHDS transition.</td>
</tr>
<tr>
<td><strong>C3</strong></td>
<td><strong>Reference Implementation.</strong> Open-source Python codebase (~40K lines, 159 modules) with 17 FL algorithms, EHDS governance modules, and an interactive deployment dashboard.</td>
</tr>
<tr>
<td><strong>C4</strong></td>
<td><strong>Experimental Validation.</strong> 1,760+ experiments across tabular clinical and medical imaging datasets with differential privacy ablation (ε ∈ {1, 5, 10, 50}), Article 71 opt-out simulation, and 10-seed statistical validation.</td>
</tr>
</table>

---

## Architecture

FL-EHDS is organised into three layers following the EHDS data flow. Raw health data never leaves institutional boundaries — only encrypted model gradients are exchanged within the Secure Processing Environment.

<table>
<tr>
<th>Layer</th>
<th>Scope</th>
<th>Key Components</th>
</tr>
<tr>
<td><strong>L1 — Governance</strong></td>
<td>HDAB integration, regulatory compliance</td>
<td>Data Permit Manager (Art. 53), Opt-Out Registry (Art. 71), Cross-Border Coordinator (Arts. 46, 50), GDPR Art. 30 audit trail</td>
</tr>
<tr>
<td><strong>L2 — FL Orchestration</strong></td>
<td>Secure Processing Environment (SPE)</td>
<td>17 aggregation algorithms, DP-SGD with RDP accounting, secure aggregation (pairwise masking, ECDH), 6 Byzantine resilience methods, compliance module</td>
</tr>
<tr>
<td><strong>L3 — Data Holders</strong></td>
<td>Institutional computation</td>
<td>Adaptive local training engine, FHIR R4 preprocessing (6 resource types), OMOP-CDM harmonisation (SNOMED CT, ICD-10, LOINC, ATC, UCUM), secure gradient communication (AES-256-GCM, mTLS)</td>
</tr>
</table>

The governance layer includes a fully functional simulation backend (OAuth2/mTLS authentication, permit CRUD, LRU-cached registry lookups) that requires only endpoint configuration — not architectural changes — for production binding to HDAB services (expected 2027–2029).

### EHDS Compliance Mapping

<table>
<tr>
<th align="center">EHDS Article</th>
<th>Requirement</th>
<th>Framework Component</th>
</tr>
<tr><td align="center">Art. 33</td><td>Secondary use authorisation</td><td>HDAB API + Permit validation</td></tr>
<tr><td align="center">Art. 46</td><td>Cross-border processing</td><td>Multi-HDAB coordinator (10 EU country profiles)</td></tr>
<tr><td align="center">Art. 50</td><td>Secure Processing Environment</td><td>All aggregation executed within SPE boundary</td></tr>
<tr><td align="center">Art. 53</td><td>Permitted purposes</td><td>Purpose limitation module with permit lifecycle</td></tr>
<tr><td align="center">Art. 71</td><td>Citizen opt-out mechanism</td><td>Registry filtering (record / patient / dataset level)</td></tr>
<tr><td align="center">GDPR Art. 30</td><td>Records of processing activities</td><td>Immutable audit trail (7-year retention)</td></tr>
</table>

---

## Algorithm Catalogue

17 FL algorithms spanning six categories, from foundational methods to ICML 2024 and ICLR 2025 advances:

| Algorithm | Venue | Category | Key Mechanism |
|:----------|:------|:---------|:--------------|
| FedAvg | AISTATS 2017 | Baseline | Weighted model averaging |
| FedProx | MLSys 2020 | Non-IID | Proximal regularisation (μ) |
| SCAFFOLD | ICML 2020 | Non-IID | Control variates for drift correction |
| FedNova | NeurIPS 2020 | Non-IID | Normalised averaging for unequal local steps |
| FedDyn | ICLR 2021 | Non-IID | Dynamic regularisation |
| FedAdam | ICLR 2021 | Adaptive | Server-side Adam momentum |
| FedYogi | ICLR 2021 | Adaptive | Controlled adaptive learning rate |
| FedAdagrad | ICLR 2021 | Adaptive | Server-side gradient accumulation |
| Per-FedAvg | NeurIPS 2020 | Personalisation | MAML-based meta-learning |
| Ditto | ICML 2021 | Personalisation | L2-regularised personal models |
| FedLC | ICML 2022 | Label skew | Logit calibration |
| FedSAM | ICML 2022 | Generalisation | Sharpness-aware flat minima |
| FedDecorr | ICLR 2023 | Representation | Decorrelation against dimensional collapse |
| FedSpeed | ICLR 2023 | Efficiency | Fewer communication rounds |
| FedExP | ICLR 2023 | Server-side | POCS-based step size |
| **FedLESAM** | **ICML 2024 Spotlight** | **Generalisation** | **Globally-guided sharpness-aware optimisation** |
| **HPFL** | **ICLR 2025** | **Personalisation** | **Shared backbone + personalised classifiers** |

Byzantine resilience (6 methods): Krum, Multi-Krum, Trimmed Mean, Coordinate-wise Median, Bulyan, FLTrust — defending against up to *f < n/3* adversarial clients.

<details>
<summary><strong>Algorithm Selection Guide for EHDS Deployments</strong></summary>
<br/>

| EHDS Scenario | Recommended | Rationale |
|:--------------|:-----------:|:----------|
| Homogeneous Member States | FedAvg | Simplicity, well-studied convergence bounds |
| Heterogeneous Member States | SCAFFOLD | Variance reduction under client drift |
| Resource-limited institutions | FedAdam | Fast convergence, fewer rounds needed |
| Privacy-critical studies | FedAvg + DP | Well-studied DP composition bounds |
| Sparse participation / dropout | FedProx | Proximal term provides dropout resilience |
| Label-imbalanced populations | FedLC | Class-frequency logit calibration |
| Communication-constrained | FedSpeed | Optimised for fewer communication rounds |
| Per-hospital personalisation | HPFL | Shared backbone + local decision boundaries |

</details>

---

## Dataset Coverage

The framework supports **19 healthcare datasets** across four modalities. Eight are experimentally evaluated in the paper:

### Evaluated Datasets

| Dataset | Samples | Type | Classes | FL Partition | EHDS Category |
|:--------|--------:|:-----|:-------:|:-------------|:--------------|
| PTB-XL ECG | 21,799 | Tabular | 5 | Natural (52 EU sites) | SCP-ECG diagnostics |
| Cardiovascular Disease | 70,000 | Tabular | 2 | Dirichlet (α = 0.5) | Vitals, lab, risk factors |
| Diabetes 130-US | 101,766 | Tabular | 2 | Dirichlet (α = 0.5) | EHR, ICD-9, medications |
| Heart Disease UCI | 920 | Tabular | 2 | Natural (4 hospitals) | Vitals, ECG, lab results |
| Breast Cancer Wisconsin | 569 | Tabular | 2 | Dirichlet (α = 0.5) | Pathology (FNA cytology) |
| Chest X-ray | 5,856 | Imaging | 2 | Dirichlet (α = 0.5) | Radiology (DICOM) |
| Brain Tumor MRI | 3,064 | Imaging | 4 | Dirichlet (α = 0.5) | Neuro-imaging (DICOM) |
| Skin Cancer | 3,297 | Imaging | 2 | Dirichlet (α = 0.5) | Dermatology (DICOM) |

<details>
<summary><strong>Additional Supported Datasets (11)</strong></summary>
<br/>

Stroke Prediction, CDC Diabetes BRFSS, CKD UCI, Cirrhosis Mayo, Synthea FHIR R4, SMART Bulk FHIR, Diabetic Retinopathy (35,126 images, 5-class), Brain Tumor MRI alt. (3,264 images, 4-class), ISIC Skin Lesions (2,357 images, 9-class). These are integrated in the framework but not evaluated in the current paper. Full details in Supplementary Material, Table S1.

</details>

---

## Experimental Highlights

Results from the primary evaluation: 7 algorithms × 3 datasets × 5 seeds, plus sweep phases, DP ablation, opt-out simulation, and extended 10-seed validation.

| Finding | Evidence |
|:--------|:---------|
| Personalised FL narrows the gap to **6.6 pp** | Ditto 75.1% vs. centralised 81.7% on Heart Disease UCI |
| Algorithm selection yields up to **12.6 pp** | Ditto 75.1% vs. FedAvg 62.5% (Heart Disease); 11.4 pp on Cardiovascular |
| **HPFL** outperforms FedAvg on **all** datasets | p = 0.004, 0.002, 0.031 (Wilcoxon, 10-seed); pooled p < 0.001 |
| DP at ε = 10 imposes **negligible cost** | < 2 pp accuracy cost across PTB-XL and Cardiovascular |
| DP noise as **regularisation** | FedAvg ε = 5 → 78.7% vs. 52.3% without DP on Breast Cancer (+26.4 pp) |
| Art. 71 opt-out at 30% is **negligible** | < 1 pp drop on adequately sized datasets |
| Personalisation **scales** | Ditto: −0.8 pp from K = 5→100 (vs. −4.7 pp FedAvg) |
| PTB-XL validates **European FL** | 92.5% accuracy (HPFL), 5-class ECG, Jain fairness 0.999 |

### Primary Benchmark — 7 Algorithms × 3 Datasets

<sub>Best accuracy per dataset in <strong>bold</strong>. Mean ± std over 5 seeds. PX = PTB-XL ECG (5 clients, site-based), CV = Cardiovascular (5 clients, α = 0.5), BC = Breast Cancer (3 clients, α = 0.5).</sub>

| Algorithm | PX Acc (%) | PX Jain | CV Acc (%) | CV Jain | BC Acc (%) | BC Jain |
|:----------|:---------:|:-------:|:---------:|:-------:|:---------:|:-------:|
| FedAvg | 91.9 ± 0.5 | 0.999 | 71.1 ± 1.8 | 0.981 | 52.3 ± 17.9 | 0.608 |
| FedProx | 91.6 ± 0.7 | 0.999 | 71.5 ± 1.2 | 0.986 | 52.3 ± 17.9 | 0.608 |
| Ditto | 91.8 ± 0.3 | 0.999 | **82.5 ± 4.7** | 0.980 | **79.1 ± 12.5** | 0.606 |
| FedLC | 91.9 ± 0.5 | 0.999 | 71.1 ± 1.6 | 0.982 | 52.1 ± 18.1 | 0.606 |
| FedExP | 92.0 ± 0.2 | 0.999 | 71.1 ± 1.8 | 0.981 | 52.3 ± 17.9 | 0.608 |
| FedLESAM | 91.9 ± 0.5 | 0.999 | 71.1 ± 1.8 | 0.981 | 52.3 ± 17.9 | 0.608 |
| **HPFL** | **92.5 ± 0.3** | 0.999 | 82.3 ± 4.5 | 0.984 | 74.1 ± 20.9 | **0.867** |

### Centralised vs. Federated — Heart Disease UCI

<sub>4 hospitals, natural non-IID. Centralised: 60 epochs, Adam (lr = 0.01). FL: 20 rounds × 3 local epochs. Mean ± std, 5 seeds.</sub>

| Approach | Accuracy | F1 | AUC | Gap |
|:---------|:--------:|:--:|:---:|:---:|
| Centralised (upper bound) | 81.7 ± 2.9% | .815 | .882 | — |
| FL — Ditto | 75.1 ± 2.0% | .761 | .826 | −6.6 pp |
| FL — FedAvg | 62.5 ± 8.0% | .736 | .834 | −19.2 pp |

### Privacy–Utility Tradeoff — PTB-XL ECG

<sub>Accuracy (%) under central DP. Gaussian mechanism, C = 1.0, δ = 10⁻⁵. Mean over 5 seeds.</sub>

| Algorithm | ε = 1 | ε = 5 | ε = 10 | No DP |
|:----------|:-----:|:-----:|:------:|:-----:|
| FedAvg | 52.3 | — | 92.4 | 91.9 |
| Ditto | 89.2 | — | 91.6 | 91.8 |
| HPFL | 87.1 | — | 92.4 | 92.5 |

> **Key insight.** Personalised methods are remarkably DP-robust. At ε = 1, FedAvg collapses (−39.6 pp) while Ditto and HPFL retain > 87% accuracy. At ε = 10, privacy imposes negligible utility cost for all algorithms.

### Statistical Significance — 10-Seed Validation

<sub>Wilcoxon signed-rank test. HPFL is the only algorithm significantly outperforming FedAvg on all three datasets individually.</sub>

| Algorithm | vs. FedAvg (PX) | vs. FedAvg (CV) | vs. FedAvg (BC) | Pooled |
|:----------|:---------------:|:---------------:|:---------------:|:------:|
| Ditto | p = 0.492 | p = 0.002 | p = 0.016 | p < 0.001 |
| **HPFL** | **p = 0.004** | **p = 0.002** | **p = 0.031** | **p < 0.001** |

---

## Privacy and Compliance

### Differential Privacy

The framework implements **Rényi Differential Privacy (RDP)** accounting with 5–6× tighter bounds than naive composition:

<table>
<tr><td><strong>Mechanism</strong></td><td>Gaussian noise with L2 gradient clipping (max norm = 1.0)</td></tr>
<tr><td><strong>Accounting</strong></td><td>RDP → (ε, δ)-DP conversion with optimal order selection</td></tr>
<tr><td><strong>Budget</strong></td><td>Configurable per data permit (default ε = 1.0, δ = 10⁻⁵)</td></tr>
<tr><td><strong>Enforcement</strong></td><td><code>BudgetExhaustedError</code> terminates training at HDAB-approved threshold</td></tr>
<tr><td><strong>Tracking</strong></td><td>Per-round cumulative expenditure with audit logging</td></tr>
</table>

### Secure Aggregation

Pairwise masking protocol with ECDH key exchange (SECP384R1 curve), Shamir secret sharing with threshold reconstruction, homomorphic encryption support (CKKS scheme via TenSEAL). Dropout threshold: 50% client participation required.

### Threat Model

| Adversary | Capability | Defence |
|:----------|:-----------|:--------|
| **A1** — Honest-but-curious server | Follows protocol, infers from gradients | Central DP + Secure aggregation |
| **A2** — Malicious clients (< n/3) | Arbitrary protocol deviation | 6 Byzantine-resilient aggregation rules |
| **A3** — External attacker | Black-box model access | Art. 71 output filtering + HDAB permit control |

### Byzantine Resilience

Six defence methods: Krum, Multi-Krum, Trimmed Mean, Coordinate-wise Median, Bulyan, FLTrust. Attack simulation: label flipping, gradient scaling, additive noise, sign flipping, model replacement.

---

## Installation

### Prerequisites

- Python ≥ 3.10 &ensp;·&ensp; PyTorch ≥ 2.0 &ensp;·&ensp; CUDA-capable GPU (optional; CPU and Apple Silicon MPS supported)

### pip

```bash
git clone https://github.com/FabioLiberti/FL-EHDS-FLICS2026.git
cd FL-EHDS-FLICS2026/fl-ehds-framework
pip install -e .
```

### Conda

```bash
git clone https://github.com/FabioLiberti/FL-EHDS-FLICS2026.git
cd FL-EHDS-FLICS2026/fl-ehds-framework

conda create -n flehds python=3.11 -y
conda activate flehds
pip install -e .
```

<details>
<summary><strong>Dependencies</strong></summary>
<br/>

**Core:** `torch`, `numpy`, `scipy`, `scikit-learn`, `pydantic`, `cryptography`, `structlog`, `fhir.resources`

**CLI:** `questionary`, `tqdm` &ensp;·&ensp; **Dashboard:** `streamlit`, `plotly` &ensp;·&ensp; **Healthcare:** `hl7apy`

</details>

---

## Usage

### Terminal Interface

```bash
cd fl-ehds-framework
python -m terminal
```

The terminal interface provides full access through 11 specialised screens:

| Screen | Function |
|:-------|:---------|
| Training | Single algorithm training with dataset selection |
| Comparison | Multi-algorithm benchmark (up to 17 algorithms) |
| Guided Comparison | Pre-configured clinical scenarios |
| Algorithm Explorer | Detailed documentation and selection guidance |
| Dataset Management | Browse, preview, and analyse 19 datasets |
| Privacy Dashboard | DP budget analysis, RDP accounting, ε-allocation |
| Benchmark Suite | Reproducible experiment configurations |
| Byzantine Resilience | Adversarial robustness testing (6 attack types) |
| EHDS Governance | Permit lifecycle, opt-out registry, compliance status |
| Monitoring | Real-time convergence, communication cost |
| Cross-Border | Multi-HDAB coordination across EU country profiles |

### Web Dashboard

```bash
streamlit run dashboard/app.py
```

### Programmatic API

```python
from fl_ehds.orchestration import FederatedTrainer

trainer = FederatedTrainer(
    dataset="ptb_xl",
    num_clients=5,
    algorithm="HPFL",
    num_rounds=30,
    local_epochs=3,
    privacy={"epsilon": 10.0, "delta": 1e-5, "clip_norm": 1.0},
    partition="site_based",
    permit_id="HDAB-DE-2026-042",
    device="cuda"
)

results = trainer.train()
# → accuracy, f1, auc, loss, jain_fairness, privacy_spent per round
```

---

## Reproducing Experiments

All experiments reported in the paper are fully reproducible. Results, checkpoints, and analysis outputs are auto-saved to `benchmarks/paper_results/` and `benchmarks/paper_results_tabular/`.

### Tabular Experiments

```bash
# Phase 1 — Baseline comparison (105 experiments, ~45 min)
python -m benchmarks.run_tabular_optimized

# Phase 2 — Multi-phase sweep: heterogeneity, client scaling, lr (1,125 experiments, ~4.5h)
python -m benchmarks.run_tabular_sweep --phase all

# Phase 3 — Differential privacy ablation (180 experiments, ~1.5h)
python -m benchmarks.run_tabular_dp

# Phase 4 — 10-seed statistical validation (105 experiments, ~40 min)
python -m benchmarks.run_tabular_seeds10

# Phase 5 — Article 71 opt-out impact (225 experiments, ~1.5h)
python -m benchmarks.run_tabular_optout

# Phase 6 — Deep MLP differentiation (70 experiments, ~1.5h)
python -m benchmarks.run_tabular_deep_mlp

# Analysis — Generates all tables, figures, and statistical tests
python -m benchmarks.analyze_tabular_extended
```

### Imaging Experiments

```bash
# Full run (7 algorithms × 5 datasets × 3 seeds)
python -m benchmarks.run_full_experiments

# Quick validation (~1–2h)
python -m benchmarks.run_full_experiments --quick

# Resume after interruption
python -m benchmarks.run_full_experiments --resume
```

### Per-Dataset Configuration

| Dataset | lr | Batch | Rounds | K | Partition | Model |
|:--------|:--:|:-----:|:------:|:-:|:----------|:------|
| PTB-XL ECG | 0.005 | 64 | 30 | 5 | Site-based | HealthcareMLP (~10K params) |
| Cardiovascular | 0.01 | 64 | 25 | 5 | Dirichlet α = 0.5 | HealthcareMLP (~10K params) |
| Breast Cancer | 0.001 | 16 | 40 | 3 | Dirichlet α = 0.5 | HealthcareMLP (~10K params) |
| Chest X-ray | 0.001 | 32 | 25 | 5 | Dirichlet α = 0.5 | ResNet-18 (~11.2M params) |
| Brain Tumor MRI | 0.001 | 32 | 25 | 5 | Dirichlet α = 0.5 | ResNet-18 (~11.2M params) |
| Skin Cancer | 0.001 | 32 | 25 | 5 | Dirichlet α = 0.5 | ResNet-18 (~11.2M params) |

<sub>All tabular experiments: Adam optimiser, early stopping (patience = 6). Imaging: GroupNorm (replacing BatchNorm for FL stability), FedBN.</sub>

<details>
<summary><strong>Reproducibility Outputs</strong></summary>
<br/>

Each experiment run auto-generates:

| Output | Format | Description |
|:-------|:-------|:------------|
| `results.json` | JSON | Complete results, training histories, full configuration |
| `summary_results.csv` | CSV | Final metrics with standard deviations |
| `history_all_metrics.csv` | CSV | Per-round Acc, Loss, F1, Precision, Recall, AUC |
| `table_results.tex` | LaTeX | Publication-ready table |
| `plot_convergence_*.png` | PNG | Convergence curves for all metrics |
| `plot_metrics_comparison.png` | PNG | Bar chart of final metric comparison |

</details>

---

## Repository Structure

```
FL-EHDS-FLICS2026/
├── paper/                              # Conference paper and figures
│   ├── main.tex                        #   LaTeX source (IEEE format)
│   ├── supplementary.tex               #   Supplementary material
│   └── figures/                        #   Paper figures incl. architecture diagram
│
├── fl-ehds-framework/                  # Framework source (~40K lines, 159 modules)
│   │
│   ├── governance/                     # ── Layer 1: EHDS Governance ──
│   │   ├── hdab_integration/           #   OAuth2/mTLS, permit store, strictness 1–5
│   │   ├── permit_manager/             #   Art. 53 lifecycle (PENDING→ACTIVE→EXPIRED)
│   │   ├── optout_registry/            #   Art. 71 filtering (record/patient/dataset)
│   │   ├── cross_border/              #   Multi-HDAB coord., 10 EU country profiles
│   │   └── compliance_logging/         #   GDPR Art. 30 audit trail, 7-year retention
│   │
│   ├── orchestration/                  # ── Layer 2: FL Orchestration (SPE) ──
│   │   ├── algorithms/                 #   17 FL algorithms (FedAvg → HPFL)
│   │   ├── privacy/                    #   DP-SGD, RDP accounting, secure aggregation
│   │   ├── byzantine/                  #   6 resilience methods + attack simulation
│   │   ├── compliance/                 #   Purpose limitation, anonymity assessment
│   │   └── communication/              #   gRPC/WebSocket, compression, serialisation
│   │
│   ├── data_holders/                   # ── Layer 3: Data Holders ──
│   │   ├── training/                   #   Adaptive local training (CUDA/MPS/CPU)
│   │   ├── preprocessing/              #   FHIR R4 pipeline, OMOP-CDM harmonisation
│   │   └── security/                   #   AES-256-GCM, ECDHE, mTLS, nonce replay
│   │
│   ├── models/                         # Model architectures
│   │   ├── healthcare_mlp.py           #   MLP (~10K) and DeepMLP (~110K params)
│   │   ├── healthcare_cnn.py           #   5-block CNN with GroupNorm (~12M)
│   │   └── healthcare_resnet.py        #   ResNet-18, GroupNorm + FedBN (~11.2M)
│   │
│   ├── monitoring/                     # Prometheus metrics, Grafana dashboards
│   ├── terminal/                       # CLI interface (11 screens)
│   ├── dashboard/                      # Streamlit web interface
│   │
│   ├── benchmarks/                     # Reproducible experiment suite
│   │   ├── run_tabular_optimized.py
│   │   ├── run_tabular_sweep.py
│   │   ├── run_tabular_dp.py
│   │   ├── run_tabular_seeds10.py
│   │   ├── run_tabular_optout.py
│   │   ├── run_tabular_deep_mlp.py
│   │   ├── run_full_experiments.py
│   │   ├── analyze_tabular_extended.py
│   │   └── paper_results/              #   Auto-generated results and figures
│   │
│   ├── datasets/                       # Dataset loaders and preprocessing
│   ├── tests/                          # Unit test suite (pytest)
│   └── data/                           # Clinical datasets (auto-downloaded)
│
├── requirements.txt
├── LICENSE
└── README.md
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

## References

1. European Commission. "Regulation (EU) 2025/327 on the European Health Data Space." *Official Journal of the EU*, 2025.
2. McMahan, B. et al. "Communication-Efficient Learning of Deep Networks from Decentralized Data." *AISTATS*, 2017.
3. Li, T. et al. "Federated Optimization in Heterogeneous Networks." *MLSys*, 2020.
4. Karimireddy, S.P. et al. "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning." *ICML*, 2020.
5. Li, T. et al. "Ditto: Fair and Robust Federated Learning Through Personalization." *ICML*, 2021.
6. Reddi, S. et al. "Adaptive Federated Optimization." *ICLR*, 2021.
7. Qu, Z. et al. "FedLESAM: Federated Learning with Locally Estimated Sharpness-Aware Minimization." *ICML*, 2024. (Spotlight)
8. Chen, Y. et al. "HPFL: Hot-Pluggable Federated Learning." *ICLR*, 2025.
9. Fröhlich, H. et al. "Reality Check: The Aspirations of the EHDS." *JMIR*, 2025.
10. Dwork, C. and Roth, A. "The Algorithmic Foundations of Differential Privacy." *Found. Trends Theor. Comput. Sci.*, 2014.

---

## License

This project is released under the [MIT License](LICENSE).

---

## Acknowledgements

The author thanks Prof. Sadi Alawadi for supervision and guidance.

---

<p align="center">
  <strong>Fabio Liberti</strong><br/>
  Department of Computer Science, Universitas Mercatorum, Rome, Italy<br/>
  <a href="mailto:fabio.liberti@studenti.unimercatorum.it">fabio.liberti@studenti.unimercatorum.it</a>&ensp;·&ensp;ORCID <a href="https://orcid.org/0000-0003-3019-5411">0000-0003-3019-5411</a>
</p>

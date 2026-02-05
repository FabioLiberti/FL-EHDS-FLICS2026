# FL-EHDS Framework: Technical Contribution

## Framework Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        FL-EHDS COMPLIANCE FRAMEWORK                         │
│                    Privacy-Preserving Cross-Border Analytics                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    LAYER 1: GOVERNANCE LAYER                        │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │   │
│  │  │    HDAB A    │  │    HDAB B    │  │    HDAB C    │   ...        │   │
│  │  │  (Member     │  │  (Member     │  │  (Member     │              │   │
│  │  │   State A)   │  │   State B)   │  │   State C)   │              │   │
│  │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘              │   │
│  │         │                 │                 │                       │   │
│  │         └────────────┬────┴────┬────────────┘                       │   │
│  │                      ▼         ▼                                    │   │
│  │              ┌───────────────────────┐                              │   │
│  │              │   DATA PERMIT LAYER   │                              │   │
│  │              │  • Authorization      │                              │   │
│  │              │  • Purpose limitation │                              │   │
│  │              │  • Opt-out registry   │                              │   │
│  │              └───────────┬───────────┘                              │   │
│  └──────────────────────────┼──────────────────────────────────────────┘   │
│                             │                                               │
│  ┌──────────────────────────┼──────────────────────────────────────────┐   │
│  │                    LAYER 2: FL ORCHESTRATION                        │   │
│  │                          ▼                                          │   │
│  │  ┌───────────────────────────────────────────────────────────────┐ │   │
│  │  │              HealthData@EU Aggregation Server                 │ │   │
│  │  │  ┌─────────────────────────────────────────────────────────┐  │ │   │
│  │  │  │              Secure Processing Environment (SPE)        │  │ │   │
│  │  │  │                                                         │  │ │   │
│  │  │  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐       │  │ │   │
│  │  │  │  │  FedAvg/    │ │ Gradient    │ │ Model       │       │  │ │   │
│  │  │  │  │  FedProx    │ │ Compression │ │ Validation  │       │  │ │   │
│  │  │  │  │  Aggregator │ │ Module      │ │ Module      │       │  │ │   │
│  │  │  │  └─────────────┘ └─────────────┘ └─────────────┘       │  │ │   │
│  │  │  │                                                         │  │ │   │
│  │  │  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐       │  │ │   │
│  │  │  │  │ Differential│ │ Membership  │ │ Audit       │       │  │ │   │
│  │  │  │  │ Privacy     │ │ Inference   │ │ Logging     │       │  │ │   │
│  │  │  │  │ Module      │ │ Defense     │ │ Module      │       │  │ │   │
│  │  │  │  └─────────────┘ └─────────────┘ └─────────────┘       │  │ │   │
│  │  │  └─────────────────────────────────────────────────────────┘  │ │   │
│  │  └───────────────────────────────────────────────────────────────┘ │   │
│  └────────────────────────────────────────────────────────────────────┘   │
│                             │                                               │
│  ┌──────────────────────────┼──────────────────────────────────────────┐   │
│  │                    LAYER 3: DATA HOLDER LAYER                       │   │
│  │         ┌────────────────┼────────────────┐                         │   │
│  │         ▼                ▼                ▼                         │   │
│  │  ┌────────────┐   ┌────────────┐   ┌────────────┐                  │   │
│  │  │ Hospital A │   │ Registry B │   │ Research C │   ...            │   │
│  │  │            │   │            │   │ Center     │                  │   │
│  │  │ ┌────────┐ │   │ ┌────────┐ │   │ ┌────────┐ │                  │   │
│  │  │ │ Local  │ │   │ │ Local  │ │   │ │ Local  │ │                  │   │
│  │  │ │ EHR    │ │   │ │ Disease│ │   │ │ Genomic│ │                  │   │
│  │  │ │ Data   │ │   │ │ Data   │ │   │ │ Data   │ │                  │   │
│  │  │ └────────┘ │   │ └────────┘ │   │ └────────┘ │                  │   │
│  │  │     │      │   │     │      │   │     │      │                  │   │
│  │  │ ┌───┴────┐ │   │ ┌───┴────┐ │   │ ┌───┴────┐ │                  │   │
│  │  │ │ Local  │ │   │ │ Local  │ │   │ │ Local  │ │                  │   │
│  │  │ │Training│ │   │ │Training│ │   │ │Training│ │                  │   │
│  │  │ │ Engine │ │   │ │ Engine │ │   │ │ Engine │ │                  │   │
│  │  │ └───┬────┘ │   │ └───┬────┘ │   │ └───┬────┘ │                  │   │
│  │  └─────┼──────┘   └─────┼──────┘   └─────┼──────┘                  │   │
│  │        │                │                │                          │   │
│  │        └────── ∇ ───────┴────── ∇ ───────┘                          │   │
│  │                    (Gradients only)                                 │   │
│  └────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Framework Components Description

### Layer 1: Governance Layer
- **HDABs**: Health Data Access Bodies per Member State
- **Data Permit Management**: Authorization workflow
- **Opt-out Registry**: Citizen preference enforcement
- **Cross-border Coordination**: Multi-HDAB synchronization

### Layer 2: FL Orchestration Layer (within SPE)
- **Aggregation Algorithms**:
  - FedAvg (baseline) - weighted averaging
  - FedProx - proximal regularization for non-IID robustness
  - SCAFFOLD - control variates for variance reduction
  - FedNova - normalized averaging for heterogeneous local steps
- **Privacy Modules**:
  - Rényi Differential Privacy (RDP) accounting with tight composition
  - Gaussian mechanism: ρ(α) = α/(2σ²)
  - Privacy amplification by subsampling
  - Optimal (ε,δ)-DP conversion: 6-10x tighter than simple composition
  - Gradient clipping (L2 norm bounding)
- **Security Modules**: Membership inference defense, gradient protection
- **Compliance Modules**: Audit logging, GDPR compliance verification

### Layer 3: Data Holder Layer
- **Local Training Engines**: On-premise model training
- **Data Preprocessing**: FHIR normalization, quality checks
- **Gradient Computation**: Local gradient calculation
- **Secure Communication**: Encrypted gradient transmission

## Barrier-Mitigation Mapping

| Barrier (from SLR) | Framework Component | Mitigation Strategy |
|-------------------|---------------------|---------------------|
| Hardware heterogeneity (78%) | Adaptive Training Engine | Resource-aware model partitioning, FedNova normalized aggregation |
| Non-IID data (67%) | FedProx/SCAFFOLD Aggregator | Proximal term regularization, control variates for variance reduction |
| Gradient privacy risk | RDP Module + MIA Defense | Rényi DP with tight composition (6-10x improvement), gradient clipping |
| Legal uncertainty | Governance Layer | Explicit data permit workflow |
| HDAB capacity gaps | Standardized APIs | Reference implementation |
| Interoperability (34% FHIR) | Data Preprocessing | FHIR-native data ingestion |

## Experimental Results (February 2026)

Benchmark results with 5 hospitals, 30 rounds, 3 local epochs (mean ± std over 3 runs):

| Algorithm | Accuracy | F1 | AUC |
|-----------|----------|-----|-----|
| FedAvg (IID) | 60.5% ± 0.02 | 0.62 ± 0.02 | 0.66 ± 0.01 |
| FedAvg (Non-IID) | 60.9% ± 0.02 | 0.61 ± 0.01 | 0.66 ± 0.01 |
| FedProx (μ=0.1) | 60.9% ± 0.02 | 0.62 ± 0.01 | 0.66 ± 0.01 |
| SCAFFOLD | 60.5% ± 0.01 | 0.61 ± 0.02 | 0.66 ± 0.01 |
| FedNova | 60.7% ± 0.02 | 0.62 ± 0.01 | 0.66 ± 0.01 |
| FedAvg + DP (ε=10) | 55.7% ± 0.01 | 0.61 ± 0.04 | 0.55 ± 0.03 |
| FedAvg + DP (ε=1) | 55.1% ± 0.01 | 0.59 ± 0.04 | 0.55 ± 0.01 |

**Key Finding**: All advanced FL algorithms (FedProx, SCAFFOLD, FedNova) perform within 0.4pp of baseline FedAvg on moderately non-IID healthcare data.

## Compliance Checkpoints

```
┌─────────────────────────────────────────────────────────────────┐
│                  FL-EHDS COMPLIANCE CHECKLIST                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  PRE-TRAINING                                                   │
│  □ Data permit obtained from relevant HDAB(s)                   │
│  □ Opt-out registry consulted                                   │
│  □ Purpose limitation verified (Art. 53 compliance)             │
│  □ Data holder agreements in place                              │
│                                                                 │
│  DURING TRAINING                                                │
│  □ Gradients processed within SPE                               │
│  □ Differential privacy applied (ε ≤ threshold)                 │
│  □ Audit logs maintained                                        │
│  □ No raw data transmission verified                            │
│                                                                 │
│  POST-TRAINING                                                  │
│  □ Model anonymity assessment (MIA test)                        │
│  □ Gradient data deletion confirmed                             │
│  □ Results reported to HDAB                                     │
│  □ Model use limited to permitted purposes                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Key Metrics for Framework Evaluation

| Metric | Description | Target |
|--------|-------------|--------|
| **Production Readiness** | % of components at TRL ≥ 7 | > 50% (vs current 23%) |
| **Compliance Score** | Checklist items satisfied | 100% |
| **Privacy Budget** | Differential privacy ε | ≤ 1.0 |
| **Communication Efficiency** | Gradient compression ratio | ≥ 10x |
| **Convergence Time** | Rounds to target accuracy | Comparable to centralized |
| **Cross-border Latency** | End-to-end round time | < 1 hour |


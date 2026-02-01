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
- **Aggregation Algorithms**: FedAvg, FedProx, with non-IID adaptations
- **Privacy Modules**: Differential privacy, gradient clipping
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
| Hardware heterogeneity (78%) | Adaptive Training Engine | Resource-aware model partitioning |
| Non-IID data (67%) | FedProx Aggregator | Proximal term regularization |
| Gradient privacy risk | DP Module + MIA Defense | ε-differential privacy, gradient clipping |
| Legal uncertainty | Governance Layer | Explicit data permit workflow |
| HDAB capacity gaps | Standardized APIs | Reference implementation |
| Interoperability (34% FHIR) | Data Preprocessing | FHIR-native data ingestion |

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


# FL-EHDS: A Privacy-Preserving Federated Learning Framework for the European Health Data Space

## Paper Outline for FLICS 2026 (8 pages IEEE double-column)

---

## METADATA

| Field | Value |
|-------|-------|
| **Target Conference** | FLICS 2026 - Valencia, Spain |
| **Track** | Main Track 1 – Federated Learning Systems & Applications |
| **Alternative** | FLHA 2026 Workshop (Federated Learning in Healthcare) |
| **Submission Deadline** | February 20, 2026 |
| **Format** | IEEE Conference (8 pages max) |
| **Template** | IEEE Conference Template (double-column) |

---

## ABSTRACT (150 words max)

The European Health Data Space (EHDS), effective March 2025, mandates cross-border health data analytics while preserving privacy. Federated Learning (FL) emerges as the key enabling technology, yet a systematic evidence synthesis reveals critical gaps: only 23% of FL implementations achieve production deployment, with hardware heterogeneity (78%) and non-IID data (67%) as dominant barriers. Legal uncertainties regarding gradient data status under GDPR remain unresolved. We present FL-EHDS, a three-layer compliance framework integrating governance mechanisms (HDABs, data permits), FL orchestration (aggregation within Secure Processing Environments), and data holder components. The framework maps evidence-based barriers to specific mitigation strategies and provides compliance checkpoints aligned with EHDS requirements. We contribute: (1) first systematic barrier taxonomy for FL in EHDS contexts; (2) a reference architecture addressing identified gaps; (3) an implementation roadmap for the 2025-2031 transition period. FL-EHDS bridges the technology-governance divide critical for successful EHDS operationalization.

**Keywords:** Federated Learning, European Health Data Space, Privacy-Preserving Technologies, GDPR Compliance, Health Data Governance, Cross-Border Analytics

---

## 1. INTRODUCTION (1 page)

### 1.1 Problem Statement (0.3 pages)
- EHDS Regulation (EU) 2025/327: cross-border secondary use ambition
- Fundamental tension: data utility vs. privacy/sovereignty
- FL as theoretically ideal solution
- Gap: policy assumes technological maturity that may not exist

### 1.2 Motivation and Gap (0.4 pages)
- Disciplinary fragmentation: technical literature ignores legal constraints; legal scholarship abstracts from feasibility
- No integrated framework addressing EHDS-specific requirements
- Implementation timeline pressure (2027 delegated acts, 2029 application)
- Cite: Fröhlich et al. (2025) "reality check"; Quinn et al. (2024) legal analysis

### 1.3 Contributions (0.3 pages)
1. **Systematic Evidence Synthesis**: First FL barrier taxonomy specific to EHDS (n=47 documents, PRISMA methodology)
2. **FL-EHDS Framework**: Three-layer reference architecture mapping barriers to mitigation strategies
3. **Compliance Toolkit**: Checkpoints and metrics for GDPR/EHDS alignment
4. **Implementation Roadmap**: Prioritized actions for 2025-2031 transition

---

## 2. BACKGROUND (0.7 pages)

### 2.1 European Health Data Space Overview (0.3 pages)
- Primary vs. secondary use distinction
- Key mechanisms: HDABs, data permits, SPEs, opt-out (Art. 71)
- Timeline: Table 1 (2025 → 2027 → 2029 → 2031)
- Article 53 permitted purposes

### 2.2 Federated Learning Fundamentals (0.4 pages)
- Core principle: model to data, not data to model
- Standard FL workflow: local training → gradient exchange → aggregation → redistribution
- Relevance for EHDS: GDPR data minimization alignment
- Key algorithms: FedAvg, FedProx (brief)
- Known challenges: non-IID, communication, privacy attacks

**Figure 1**: FL workflow diagram (simplified from existing Figure 1)

---

## 3. FL-EHDS FRAMEWORK (2 pages) — **MAIN CONTRIBUTION**

### 3.1 Framework Overview (0.4 pages)
- Three-layer architecture rationale
- Design principles: compliance-by-design, privacy-by-default, interoperability-first

**Figure 2**: FL-EHDS Architecture Diagram (the main technical contribution)

### 3.2 Layer 1: Governance Layer (0.5 pages)
- HDAB integration points
- Data permit workflow automation
- Opt-out registry consultation protocol
- Cross-border coordination mechanisms
- Multi-HDAB synchronization for pan-European studies

### 3.3 Layer 2: FL Orchestration Layer (0.6 pages)
- Secure Processing Environment (SPE) requirements
- Aggregation module: FedAvg with non-IID adaptations (FedProx)
- Privacy protection modules:
  - Differential privacy integration (ε-budget management)
  - Gradient clipping
  - Membership inference defense
- Compliance modules:
  - Audit logging (GDPR Art. 30 alignment)
  - Purpose limitation enforcement

### 3.4 Layer 3: Data Holder Layer (0.5 pages)
- Local training engine specifications
- Hardware heterogeneity accommodation (resource-aware partitioning)
- FHIR-native data preprocessing
- Gradient computation and secure transmission
- Interoperability requirements (EIF compliance)

**Table 2**: Framework component specifications and requirements

---

## 4. EVIDENCE SYNTHESIS: BARRIERS AND MITIGATION (1.5 pages)

### 4.1 Methodology Summary (0.3 pages)
- PRISMA 2020 approach (brief)
- 8 databases, May 2022 – January 2026
- 47 documents included (44 peer-reviewed)
- MMAT quality assessment; GRADE-CERQual confidence

### 4.2 Technical Barriers (0.5 pages)
**Table 3**: Barrier taxonomy with prevalence and evidence

| Barrier | Prevalence | Key Evidence | Framework Mitigation |
|---------|------------|--------------|---------------------|
| Hardware heterogeneity | 78% | Fröhlich et al. 2025 | Adaptive Training Engine |
| Non-IID data | 67% | Multiple studies | FedProx + stratified sampling |
| Communication costs | High | HealthData@EU Pilot | Gradient compression |
| Production deployment | 23% | Fröhlich et al. 2025 | Reference implementation |

### 4.3 Legal Uncertainties (0.4 pages)
- Gradient data status under GDPR: personal, anonymous, or context-dependent?
- Aggregated model anonymity threshold
- Controller/processor responsibilities in FL
- Framework response: explicit compliance checkpoints, audit trails

### 4.4 Organizational Barriers (0.3 pages)
- HDAB capacity asymmetries (Nordic 2-3 years ahead)
- Interoperability gaps (34% FHIR compliance)
- Access timeline variation (3 weeks to 12+ months)
- Framework response: standardized APIs, reference workflows

---

## 5. IMPLEMENTATION ROADMAP (1.5 pages)

### 5.1 Phased Implementation Strategy (0.5 pages)

**Table 4**: Roadmap aligned with EHDS milestones

| Phase | Timeline | Actions | Dependencies |
|-------|----------|---------|--------------|
| **Foundation** | 2025-2026 | Reference implementation; pilot deployment | — |
| **Clarification** | 2027 | Delegated acts integration; legal guidance | EU regulatory decisions |
| **Scaling** | 2028-2029 | Multi-MS deployment; HDAB onboarding | Infrastructure readiness |
| **Full Operation** | 2029-2031 | Production across all categories | Interoperability maturity |

### 5.2 Stakeholder-Specific Recommendations (0.5 pages)

**For EU Policymakers:**
- Clarify gradient data status in delegated acts (2027)
- Establish FL-specific HDAB guidance
- Fund cross-border pilot programs

**For National Authorities:**
- Early HDAB capacity investment
- Staff training on FL evaluation
- Standardized authorization workflows

**For Healthcare Organizations:**
- FHIR compliance acceleration
- Local training infrastructure assessment
- Pilot participation

**For Researchers:**
- Shift to implementation studies
- Interdisciplinary collaboration
- Negative result reporting

### 5.3 Evaluation Metrics (0.5 pages)

**Table 5**: FL-EHDS success metrics

| Metric | Baseline (2025) | Target (2029) | Measurement |
|--------|-----------------|---------------|-------------|
| Production deployment | 23% | >50% | Operational FL systems |
| FHIR compliance | 34% | >70% | EIF assessment |
| Legal clarity | Unresolved | Resolved | DPA guidance |
| Cross-border pilots | 5 use cases | 20+ use cases | HealthData@EU registry |
| HDAB FL capacity | Variable | Standardized | Capability assessment |

---

## 6. DISCUSSION AND CONCLUSIONS (0.7 pages)

### 6.1 Key Findings (0.3 pages)
- FL is necessary but not sufficient: governance co-investment required
- Legal uncertainties are the critical blocker, not technical barriers
- Technology-governance divide poses effectiveness and legitimacy risks
- 2027 delegated acts represent critical window

### 6.2 Limitations (0.2 pages)
- Framework not yet empirically validated
- Evidence base predominantly anticipatory (pre-operational EHDS)
- Rapidly evolving regulatory and technical landscape

### 6.3 Future Work (0.2 pages)
- Empirical validation through HealthData@EU pilot integration
- Citizen attitude studies
- Economic sustainability modeling
- Longitudinal implementation tracking

---

## REFERENCES (0.3 pages, ~25-30 references)

### Priority references to include:
1. European Commission (2025) - EHDS Regulation
2. Fröhlich et al. (2025) - Reality check, 23% finding
3. Quinn et al. (2024) - GDPR-EHDS legal tensions
4. van Drumpt et al. (2025) - PETs research agenda
5. Forster et al. (2025) - User journeys, access timelines
6. TEHDAS Joint Action (2024) - Member State readiness
7. Hussein et al. (2025) - Interoperability framework
8. Rieke et al. (2020) - FL in digital health (foundational)
9. McMahan et al. (2017) - FedAvg (foundational)
10. Kairouz et al. (2021) - Advances in FL
11. Christiansen et al. (2025) - HealthData@EU Pilot
12. Staunton et al. (2024) - Ethical reflections
13. Li et al. (2020) - Federated optimization, FedProx
14. Zhu et al. (2019) - Gradient inversion attacks
15. Shokri et al. (2017) - Membership inference

---

## FIGURES AND TABLES

| Element | Description | Placement |
|---------|-------------|-----------|
| **Figure 1** | FL workflow (simplified) | Section 2.2 |
| **Figure 2** | FL-EHDS Architecture (main contribution) | Section 3.1 |
| **Table 1** | EHDS Timeline | Section 2.1 |
| **Table 2** | Framework component specs | Section 3.4 |
| **Table 3** | Barrier taxonomy | Section 4.2 |
| **Table 4** | Implementation roadmap | Section 5.1 |
| **Table 5** | Success metrics | Section 5.3 |

---

## DIFFERENCES FROM ORIGINAL PAPER

| Aspect | Original (SLR) | FLICS Version |
|--------|----------------|---------------|
| **Length** | ~17 pages | 8 pages |
| **Focus** | Literature synthesis | Framework + evidence |
| **Contribution** | Review findings | Technical framework |
| **Methodology detail** | Extensive (2+ pages) | Brief (0.3 pages) |
| **Audience** | Journal readers | Conference attendees |
| **Actionability** | Research gaps | Implementation roadmap |
| **Technical depth** | Policy-oriented | Architecture-oriented |

---

## WRITING TIPS FOR IEEE FORMAT

1. **Active voice**: "We propose" not "A framework is proposed"
2. **Concise**: Max 1 idea per sentence
3. **Figures first**: Design figures, then write around them
4. **Quantitative**: Use numbers (23%, 78%, 67%) prominently
5. **Forward reference**: "Section 3 presents..." in Introduction
6. **No appendices**: Everything in main text
7. **Self-contained captions**: Figures/tables understandable standalone


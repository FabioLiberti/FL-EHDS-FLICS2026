# FL-EHDS Cascading Experiment Registry

**Project:** FL-EHDS — A Privacy-Preserving Federated Learning Framework for the European Health Data Space
**Conference:** FLICS 2026
**Author:** Fabio Liberti

This document provides a structured registry of all cascading experiments executed
for the FL-EHDS framework evaluation. Each cascade is a self-contained phase of
experimentation, with blocks addressing specific research questions. All experiments
use real clinical datasets with atomic checkpointing for reproducibility.

---

## Summary

| Cascade | Blocks | Experiments | Time (full) | Focus |
|---------|--------|-------------|-------------|-------|
| 2 | D | 27 | 6:51 | Privacy-efficiency foundations |
| 3 | C-D | 75 | 18:37 | Data efficiency and compound stress |
| 4 | A-D | 135 | 11:52 | Fairness, continual learning, personalization, unlearning |
| 5 | E-F | 87 | 3:56 | Client selection and gradient inversion |
| 6 | G-L | 234 | 30:27 | Dataset coverage, hierarchical/async FL, Byzantine, Shapley |
| 7 | M-R | 243 | 25:13 | Calibration, conformal, attribution, demographic fairness, drift, DP composition |
| 8 | S-W | 119 | 10:57 | Secure aggregation, compression, local/central DP, vertical FL, CDC Diabetes |
| 9 | X-AB | 144 | 18:24 | Scalability K=50, DP+compression, convergence dynamics, clinical imbalance |
| 10 | AC-TH | 384 | 22:16 | Clinical imbalance deep-dive: DP regularization, loss mitigation, threshold tuning |
| **Standalone** | | **~644** | | Scalability, Non-IID+DP, Byzantine+DP, MIA, partial participation, etc. |
| **Total** | | **~2,092+** | | |

### Datasets used across all cascades

| Dataset | Samples | Features | Classes | Clinical Domain | Cascades |
|---------|---------|----------|---------|-----------------|----------|
| Cardiovascular | 70,000 | 11 | 2 | Cardiology | 2-9, standalone |
| PTB-XL ECG | 21,799 | 9 | 5 | Cardiac electrophysiology | 2-9, standalone |
| Breast Cancer | 569 | 30 | 2 | Oncology (radiology) | 2-8, standalone |
| CDC Diabetes | 253,680 | 21 | 2 | Endocrinology / Public health | 8-10 |
| Stroke | 5,110 | 10 | 2 | Neurology | 6, 9, 10 |
| CKD | 399 | 24 | 2 | Nephrology | 6, 9 |
| Cirrhosis | 418 | 18 | 2 | Hepatology | 6, 9, 10 |
| Chest X-ray | 5,856 | images | 2 | Radiology | imaging track |
| Brain Tumor MRI | 7,023 | images | 4 | Neuro-radiology | imaging track |
| Skin Cancer | 3,297 | images | 2 | Dermatology | imaging track |

### Algorithms tested

**Baseline:** FedAvg, FedProx, SCAFFOLD
**Personalized:** Ditto (ICML 2021), HPFL (ICLR 2025)
**Fairness-aware:** QFedAvg, AFL, FedMGDA, PropFair, FedMinMax
**Continual:** Replay, EWC, FedProx-continual
**Personalization extended:** Per-FedAvg, FedPer, APFL, pFedMe
**Unlearning:** Exact retrain, Gradient ascent, FedEraser
**Client selection:** Random, Loss-based, Importance, Resource-aware, Fairness-aware (Oort)
**Byzantine-robust:** Median, Krum, Multi-Krum, Trimmed Mean, Bulyan
**Compression:** SignSGD, QSGD, TernGrad, TopK, RandomK, Threshold, PowerSGD

---

## Cascade 2: Privacy-Efficiency Foundations

**Checkpoint:** `checkpoint_cascade2.json`
**Experiments:** 27 | **Runtime:** 6:51

### Block D: Learning Rate Sensitivity under Privacy (27 experiments)

**Research question:** How does differential privacy interact with learning rate
selection across different algorithms and datasets?

| Factor | Values |
|--------|--------|
| Datasets | Cardiovascular, PTB-XL, Breast Cancer |
| Algorithms | FedAvg, Ditto, HPFL |
| DP levels | No-DP, eps=10, eps=1 |
| Seeds | 42 |

**Key finding:** Learning rate sensitivity increases under DP. Ditto and HPFL
maintain stability across LR ranges where FedAvg collapses.

---

## Cascade 3: Data Efficiency and Compound Stress

**Checkpoint:** `checkpoint_cascade3.json`
**Experiments:** 75 | **Runtime:** 18:37

### Block C: Data Efficiency (27 experiments)

**Research question:** How much training data is needed for adequate FL performance?
What is the minimum viable dataset size for EHDS deployment?

| Factor | Values |
|--------|--------|
| Datasets | Cardiovascular, PTB-XL, Breast Cancer |
| Algorithms | FedAvg, Ditto, HPFL |
| Data fractions | 25%, 50%, 75%, 100% |
| Seeds | 42 |

**Key finding:** 50% of training data achieves within 1% of full-data performance.
PTB-XL reaches 90% accuracy with only 4,274 samples (25% fraction).

### Block D: Compound Stress Testing (48 experiments)

**Research question:** How do algorithms perform under simultaneous adversarial
conditions (non-IID + DP + partial participation)? Does the combination cause
super-additive degradation?

| Factor | Values |
|--------|--------|
| Datasets | Cardiovascular, PTB-XL, Breast Cancer |
| Algorithms | FedAvg, Ditto, HPFL |
| Non-IID alpha | 0.1 (extreme), 0.5 (moderate), 1.0 (mild) |
| DP | No-DP, eps=10 |
| Participation | 60%, 100% |
| Seeds | 42, 123 |

**Key finding:** HPFL is uniquely robust to compound stress. Under simultaneous
non-IID (alpha=0.1) + DP (eps=10) + 60% participation: HPFL 91.1% vs FedAvg
52.9% (38.2pp gap). HPFL worst-case (86.2%) exceeds FedAvg best-case (72.6%).

---

## Cascade 4: Advanced FL Paradigms

**Checkpoint:** `checkpoint_cascade4.json`
**Experiments:** 135 | **Runtime:** 11:52

### Block A: Fairness-Aware Aggregation (36 experiments)

**Research question:** Can fairness-aware aggregation methods solve the
majority-class collapse problem observed with standard FedAvg?

| Factor | Values |
|--------|--------|
| Datasets | Cardiovascular, PTB-XL, Breast Cancer |
| Algorithms | QFedAvg, AFL, FedMGDA, PropFair, FedMinMax, FedAvg (baseline) |
| Seeds | 42, 123 |

**Key finding:** FedMinMax solves majority-class collapse on Breast Cancer
(54.5% -> 85.7%). Different fairness algorithms have complementary strengths.

### Block B: Continual FL with Concept Drift (36 experiments)

**Research question:** How resilient are FL algorithms to temporal data
distribution shifts? Which continual learning strategies are effective?

| Factor | Values |
|--------|--------|
| Datasets | Cardiovascular, PTB-XL, Breast Cancer |
| Drift types | None, Mild (10% label flip), Severe (20-40% flip) |
| Strategies | None, Replay, EWC, Retrain |
| Seeds | 42, 123 |

**Key finding:** Experience Replay maintains accuracy under drift. EWC is
counterproductive (-22.4pp). FedProx-continual offers no benefit over replay.

### Block C: Extended Personalization (45 experiments)

**Research question:** How do recent personalization methods compare on EHDS
tasks? Is the advantage modality-dependent?

| Factor | Values |
|--------|--------|
| Datasets | Cardiovascular, PTB-XL, Breast Cancer |
| Algorithms | Ditto, HPFL, Per-FedAvg, FedPer, APFL, pFedMe, FedAvg (baseline) |
| IID modes | IID, NonIID (alpha=0.5) |
| Seeds | 42 |

**Key finding:** Ditto/FedPer/APFL cluster at ~91.6% on Cardiovascular NonIID.
Per-FedAvg (MAML-based) fails catastrophically (17.4%). Personalization effect
is method-dependent, not modality-dependent.

### Block D: Federated Unlearning (18 experiments)

**Research question:** Can FL models comply with GDPR Article 17 (right to
erasure) and EHDS Article 71 (opt-out) without full retraining?

| Factor | Values |
|--------|--------|
| Datasets | Cardiovascular, PTB-XL, Breast Cancer |
| Methods | Exact retrain, Gradient ascent, FedEraser |
| Seeds | 42, 123 |

**Key finding:** Gradient ascent achieves Article 17 compliance at 5%
computational cost with <=0.2pp accuracy drop. FedEraser viable for
larger-scale hospital withdrawal.

---

## Cascade 5: Client-Centric Mechanisms

**Checkpoint:** `checkpoint_cascade5.json`
**Experiments:** 87 | **Runtime:** 3:56

### Block E: Client Selection Strategies (60 experiments)

**Research question:** Which client selection strategy optimizes the
accuracy-fairness trade-off in heterogeneous EHDS deployments?

| Factor | Values |
|--------|--------|
| Datasets | Cardiovascular, PTB-XL, Breast Cancer |
| Strategies | Random, Loss-based, Importance sampling, Resource-aware, Fairness-aware (Oort) |
| Participation rate | 60% |
| Seeds | 42, 123, 456 |

**Key finding:** Oort selection achieves best accuracy (78.5%) AND fairness
(Jain 0.938). Paradoxically, fairness-aware selection performs worst (71.5%,
Jain 0.868).

### Block F: Gradient Inversion Attack (27 experiments)

**Research question:** How vulnerable are FL gradients to reconstruction attacks?
Does DP provide effective protection?

| Factor | Values |
|--------|--------|
| Datasets | Cardiovascular, PTB-XL, Breast Cancer |
| DP levels | No-DP, eps=10, eps=1 |
| Seeds | 42, 123, 456 |

**Key finding:** Without DP, 2/3 attacks succeed (MSE 0.10, cosine 0.816).
DP eps=10 achieves 100% attack failure (MSE 1,927x higher, cosine -> 0.021).

---

## Cascade 6: Extended Evaluation and Advanced Mechanisms

**Checkpoint:** `checkpoint_cascade6.json`
**Experiments:** 234 | **Runtime:** 30:27

### Block G: Extended Dataset Coverage (90 experiments)

**Research question:** Do FL algorithm advantages generalize across diverse
clinical domains? How do new datasets (Stroke, CKD, Cirrhosis) compare?

| Factor | Values |
|--------|--------|
| Datasets | Cardiovascular, PTB-XL, Breast Cancer, Stroke, CKD, Cirrhosis |
| Algorithms | FedAvg, FedProx, SCAFFOLD, Ditto, HPFL |
| IID modes | IID, NonIID |
| Seeds | 42, 123 |

**Key finding:** Personalization advantage (Ditto/HPFL) generalizes across all
6 clinical domains. HPFL dominates on small-sample datasets (CKD 90.7%,
Cirrhosis 83.6%).

### Block H: Hierarchical FL (18 experiments)

**Research question:** Does multi-level governance (hospital -> regional -> national)
introduce accuracy penalties? Is hierarchical FL viable for EHDS?

| Factor | Values |
|--------|--------|
| Datasets | Cardiovascular, PTB-XL, Breast Cancer |
| Algorithms | FedAvg, Ditto, HPFL |
| Hierarchy levels | 2 (flat), 3 (hierarchical) |
| Seeds | 42, 123 |

**Key finding:** Hierarchical FL introduces no accuracy penalty (<0.5pp range).
Multi-level governance is viable for cross-border EHDS deployments.

### Block I: Asynchronous FL (36 experiments)

**Research question:** How robust is FL to client communication delays? What
staleness level is tolerable?

| Factor | Values |
|--------|--------|
| Datasets | Cardiovascular, PTB-XL, Breast Cancer |
| Algorithms | FedAvg, Ditto |
| Staleness levels | 0 (sync), 1, 3, 5 |
| Seeds | 42, 123 |

**Key finding:** Staleness <= 3 rounds is tolerable (<2pp degradation).
Staleness = 5 causes significant degradation for FedAvg but Ditto remains
robust.

### Block J: Multi-Task Learning (24 experiments)

**Research question:** Can FL models learn multiple clinical tasks simultaneously
(diagnosis + severity + readmission)?

| Factor | Values |
|--------|--------|
| Datasets | Cardiovascular, PTB-XL |
| Tasks | Single-task, Dual-task (diagnosis + severity), Triple-task |
| Algorithms | FedAvg, Ditto, HPFL |
| Seeds | 42 |

**Key finding:** Multi-task learning is feasible with <2pp primary task
degradation. Dual-task offers best tradeoff.

### Block K: Byzantine-Robust Aggregation (36 experiments)

**Research question:** Which Byzantine defense is most effective against
model poisoning attacks in FL?

| Factor | Values |
|--------|--------|
| Datasets | Cardiovascular, PTB-XL, Breast Cancer |
| Defenses | None, Median, Krum, Multi-Krum, Trimmed Mean, Bulyan |
| Attack fraction | 20% Byzantine clients |
| Seeds | 42, 123 |

**Key finding:** Multi-Krum and Trimmed Mean fully neutralize model poisoning.
HPFL provides an additional natural defense layer due to local personalization.

### Block L: Shapley Value Attribution (30 experiments)

**Research question:** Can we fairly quantify each hospital's contribution to the
federated model? Are Shapley values computationally feasible?

| Factor | Values |
|--------|--------|
| Datasets | Cardiovascular, PTB-XL, Breast Cancer |
| Algorithms | FedAvg, Ditto, HPFL |
| Seeds | 42 |
| Clients | 5, with leave-one-out + full combination |

**Key finding:** Shapley values correctly identify high-contribution clients.
Computational cost scales as O(2^K) but is feasible for K <= 10 with
approximation.

---

## Cascade 7: Trustworthiness and Statistical Rigor

**Checkpoint:** `checkpoint_cascade7.json`
**Experiments:** 243 | **Runtime:** 25:13

### Block M: Model Calibration (54 experiments)

**Research question:** Are FL model probability outputs well-calibrated? Does
DP affect calibration? Is post-hoc calibration effective?

| Factor | Values |
|--------|--------|
| Datasets | Cardiovascular, PTB-XL, Breast Cancer |
| Algorithms | FedAvg, Ditto, HPFL |
| DP levels | No-DP, eps=10, eps=1 |
| Seeds | 42, 123 |

**Key finding:** FL models are systematically overconfident. Temperature scaling
reduces Expected Calibration Error (ECE). DP slightly improves calibration by
preventing overconfident predictions.

### Block N: Conformal Prediction (36 experiments)

**Research question:** Can conformal prediction provide valid uncertainty
quantification for FL models? Do prediction sets maintain coverage under DP?

| Factor | Values |
|--------|--------|
| Datasets | Cardiovascular, PTB-XL, Breast Cancer |
| IID modes | IID, NonIID |
| DP levels | No-DP, eps=10 |
| Seeds | 42, 123 |

**Key finding:** Conformal prediction maintains nominal coverage (95%) under
both IID and non-IID conditions. DP increases prediction set sizes but
maintains validity.

### Block O: Feature Attribution (36 experiments)

**Research question:** Are feature importance rankings consistent across FL
clients? Does DP distort clinical interpretability?

| Factor | Values |
|--------|--------|
| Datasets | Cardiovascular, PTB-XL, Breast Cancer |
| Algorithms | FedAvg |
| IID modes | IID, NonIID |
| DP levels | No-DP, eps=10 |
| Seeds | 42, 123 |

**Key finding:** Top-3 feature rankings are consistent across clients under IID.
Non-IID introduces ranking divergence. DP (eps=10) does not distort clinical
feature importance.

### Block P: Demographic Fairness (36 experiments)

**Research question:** Do FL algorithms exhibit demographic biases? Are
accuracy disparities across subgroups clinically significant?

| Factor | Values |
|--------|--------|
| Algorithms | FedAvg, Ditto, HPFL |
| IID modes | IID, NonIID |
| DP levels | No-DP, eps=10 |
| Seeds | 42, 123 |
| Subgroups | Age, Sex |

**Key finding:** Ditto reduces demographic accuracy gaps by up to 60% vs FedAvg.
HPFL achieves most equitable outcomes (smallest max-min gap across subgroups).

### Block Q: Concept Drift Detection (36 experiments)

**Research question:** Can FL detect and adapt to temporal distribution shifts
in clinical data? Which adaptation strategy is optimal?

| Factor | Values |
|--------|--------|
| Algorithms | FedAvg, Ditto, HPFL |
| Drift severity | Mild, Moderate, Severe |
| Adaptation | None, Retrain, Replay, Fine-tune |
| Seeds | 42, 123 |

**Key finding:** Drift detection via accuracy monitoring is feasible. Replay-based
adaptation is most effective. Ditto/HPFL provide inherent drift resistance via
local models.

### Block R: DP Composition Analysis (45 experiments)

**Research question:** How does privacy budget accumulate across multiple FL
studies on the same data? When is the budget exhausted?

| Factor | Values |
|--------|--------|
| Dataset | Cardiovascular |
| DP epsilon | 1.0, 5.0, 10.0 |
| Number of sequential studies | 1, 2, 5, 10, 20 |
| Composition methods | Simple (linear), Advanced (sqrt), RDP |

**Key finding:** RDP composition enables 5x more studies than simple composition
for the same total budget. At eps=10 with RDP, 20 sequential studies remain
within clinically acceptable accuracy bounds.

---

## Cascade 8: Communication, Privacy and Scalability

**Checkpoint:** `checkpoint_cascade8.json`
**Experiments:** 119 | **Runtime:** 10:57

### Block S: Secure Aggregation Overhead (13 experiments)

**Research question:** What is the computational overhead of cryptographic
secure aggregation? Does it affect model accuracy?

| Factor | Values |
|--------|--------|
| Methods | Pairwise Masking (ECDH), Shamir Secret Sharing |
| Gradient dims | 100, 1000, 5000 |
| Seeds | 42, 123 |
| FL integration | 10-round FedAvg on Cardiovascular with SecAgg |

**Key finding:** Pairwise masking adds sub-millisecond overhead with zero
accuracy degradation (masks cancel to machine precision, error < 1e-16).
Shamir SS reconstruction error ~3-4e-6 (quantization noise only). FL
integration achieves 0.7266 accuracy, matching non-SecAgg baseline.

### Block T: Gradient Compression Benchmark (24 experiments)

**Research question:** Which gradient compression method offers the best
compression-accuracy trade-off for EHDS communication efficiency?

| Factor | Values |
|--------|--------|
| Dataset | Cardiovascular |
| Methods | SignSGD, QSGD, TernGrad, TopK, RandomK, Threshold, PowerSGD, None (baseline) |
| Rounds | 15 |
| Seeds | 42, 123, 456 |

**Key finding:** SignSGD achieves 27.84x compression (96.4% bandwidth saved)
with only -0.17pp accuracy loss. QSGD/TernGrad offer 3.93x with no loss.
RandomK fails completely (near-random 0.4982 accuracy).

### Block U: Local DP vs Central DP (42 experiments)

**Research question:** Is per-client noise (local DP) or post-aggregation noise
(central DP) more effective? How does the choice interact with dataset size?

| Factor | Values |
|--------|--------|
| Datasets | Cardiovascular, PTB-XL, Breast Cancer |
| DP modes | Local, Central, None (baseline) |
| DP epsilon | 1.0, 5.0, 10.0 |
| Seeds | 42, 123 |

**Key finding:** DP acts as a regularizer on small datasets: Breast Cancer
improves 15.5pp (0.6429 -> 0.7857) with central DP eps=1. Central DP is not
consistently superior to local DP (needs more seeds for stable comparison).

### Block V: Vertical FL Simulation (24 experiments)

**Research question:** How does vertical FL (feature-partitioned) perform when
hospitals hold different features for the same patients? What is the impact
of incomplete patient overlap?

| Factor | Values |
|--------|--------|
| Dataset | Cardiovascular (vertical split) |
| Parties | 2 (demographics + clinical), 3 (+behavioral) |
| DP | No-DP, eps=5 |
| Patient overlap | 100%, 80% |
| Seeds | 42, 123, 456 |

**Key finding:** 2-party noDP achieves 69.6% test accuracy. DP reduces 5-10pp.
3-party split is more robust to reduced overlap (66.7% vs 52.5% for 2-party
at 80% overlap). PSI alignment correctly identifies shared patients.

### Block W: CDC Diabetes Scalability (16 experiments)

**Research question:** Can the FL framework handle large-scale population
health datasets (253K samples)? How does class imbalance (86/14) affect
personalized algorithms?

| Factor | Values |
|--------|--------|
| Dataset | CDC Diabetes (253,680 samples, 21 features) |
| Algorithms | FedAvg, Ditto, HPFL |
| IID modes | IID, NonIID (alpha=0.5) |
| DP | None, eps=10, eps=1 |
| Seeds | 42, 123 |

**Key finding:** Ditto rescues the minority class on NonIID data (F1: 0.0 -> 0.80,
recall: 0.0 -> 0.82). DP destroys F1 on imbalanced data (eps=1: recall 0.24%).
Framework scales to 253K samples without issues.

---

## Cascade 9: Deployment Readiness and Cross-Cutting Validation

**Checkpoint:** `checkpoint_cascade9.json`
**Experiments:** ~140 (planned) | **Runtime:** ~15-20 min (estimated)

### Block X: Extended Scalability (24 experiments)

**Research question:** How does FL performance scale to K=50 clients,
representative of a real multi-hospital EHDS deployment? Can the largest
dataset (253K) support this scale?

| Factor | Values |
|--------|--------|
| Datasets | CDC Diabetes (K={10, 20, 50}), Cardiovascular (K=50) |
| Algorithms | FedAvg, Ditto, HPFL |
| IID mode | NonIID (alpha=0.5) |
| Seeds | 42, 123 |

**Motivation:** Existing scalability tests (checkpoint_scalability_topk.json)
cover K={3,5,10,15,20} on Cardiovascular and PTB-XL only. CDC Diabetes
(253K samples) is untested with K scaling, and K=50 has never been tested
on any dataset.

### Block Y: Combined DP + Compression (24 experiments)

**Research question:** Can we simultaneously maintain differential privacy
AND reduce communication bandwidth without excessive accuracy loss?
Is the degradation additive or synergistic?

| Factor | Values |
|--------|--------|
| Dataset | Cardiovascular |
| Compression | SignSGD (27.84x), QSGD (3.93x) |
| DP epsilon | 10, 1 |
| Algorithms | FedAvg, Ditto, HPFL |
| Seeds | 42, 123 |

**Motivation:** Block T tested compression alone, Block U tested DP alone.
A real EHDS deployment would use BOTH simultaneously. No existing experiment
tests this combination with non-TopK compressors. SignSGD (best from Block T)
is particularly interesting because 1-bit quantization may interact
non-trivially with DP noise.

### Block Z: Convergence Dynamics (18 experiments)

**Research question:** How many communication rounds are needed for convergence?
Which algorithms converge fastest? What is the cost-benefit of additional
rounds for EHDS deployment planning?

| Factor | Values |
|--------|--------|
| Datasets | Cardiovascular, PTB-XL, CDC Diabetes |
| Algorithms | FedAvg, Ditto, HPFL |
| IID mode | NonIID (alpha=0.5) |
| Tracking | Per-round accuracy (all 30 rounds) |
| Seed | 42 |

**Motivation:** No existing cascade tracks per-round accuracy. The paper
references convergence dynamics but lacks detailed round-by-round data
for all datasets. This is essential for deployment cost estimation
(each round = one communication cycle across hospitals).

### Block AA: Clinical Imbalance Robustness (54 experiments)

**Research question:** Can personalized FL handle extreme class imbalance
found in real clinical datasets? Does Ditto's minority-class rescue
(observed on CDC Diabetes) generalize to other imbalanced domains?

| Factor | Values |
|--------|--------|
| Datasets | Stroke (4.9% positive), CKD (62.7% positive), Cirrhosis (38.5% mortality) |
| Algorithms | FedAvg, Ditto, HPFL |
| IID modes | IID, NonIID (alpha=0.5) |
| DP | None, eps=10 |
| Seeds | 42, 123 |
| Metrics | Accuracy, F1, Precision, Recall, DEI |

**Motivation:** Cascade 6 Block G tested these datasets for basic accuracy only.
F1/precision/recall/DEI metrics are missing. Stroke (4.9% positive) is the
most extreme imbalance in the dataset portfolio — if Ditto rescues the
minority class here as it did on CDC Diabetes, this is a strong result for
EHDS clinical deployment.

### Block AB: CDC Diabetes NonIID Depth (20 experiments)

**Research question:** How robust is the Ditto advantage on CDC Diabetes
across different levels of data heterogeneity? Is there an optimal
alpha for personalization benefit?

| Factor | Values |
|--------|--------|
| Dataset | CDC Diabetes |
| Algorithms | FedAvg, Ditto, HPFL |
| NonIID alpha | 0.1 (extreme), 0.25, 0.75, 1.0 (mild) |
| Seeds | 42, 123 |
| Metrics | Accuracy, F1, Precision, Recall |

**Motivation:** Block W used only alpha=0.5. The dramatic Ditto result
(F1: 0.0 -> 0.80) needs validation at other heterogeneity levels to
determine if this is a robust finding or specific to alpha=0.5.

---

## Cascade 10: Clinical Imbalance Deep-Dive

**Checkpoint:** `checkpoint_cascade10.json`
**Experiments:** 384 | **Runtime:** 22:16

Cascade 10 is a systematic investigation of the majority-class collapse phenomenon
discovered in Cascade 9 Block AA, where 37% of experiments on imbalanced datasets
produced F1=0.0. This cascade tests DP noise as a regularizer, class-weighted loss
functions as mitigation, local epoch tuning for Ditto, and post-hoc threshold
optimization as a rescue mechanism.

### Block AC: Complete Condition Matrix (90 experiments)

**Research question:** Does DP noise act as an implicit regularizer against
majority-class collapse? Is the effect consistent across privacy budgets (eps=1, eps=10)?

| Factor | Values |
|--------|--------|
| Datasets | Stroke (4.9% positive), Cirrhosis (37% positive) |
| Algorithms | FedAvg, Ditto, HPFL |
| Conditions | IID+eps10, IID+eps1, NonIID+eps1 (filling gaps from Block AA) |
| Seeds | 42, 123, 456, 789, 999 |

**Key finding:** DP noise eliminates majority-class collapse. Without DP, collapse
rate is 75-83% (Block AA). With any DP level (eps=1 or eps=10), collapse drops to
0-3.3%. The DP gradient noise prevents the optimizer from converging to the trivial
"always predict majority" solution. NonIID+eps1 is the most protective condition
(Stroke F1=0.22, Cirrhosis F1=0.57), confirming the "double regularization"
hypothesis: data heterogeneity + DP noise force diverse gradient signals that
benefit minority-class learning.

### Block AD: Mitigation Strategies (234 experiments)

**Research question:** Can class-weighted cross-entropy or focal loss eliminate
the F1=0.0 collapse? Which strategy is superior for clinical deployment?

| Factor | Values |
|--------|--------|
| Datasets | Stroke (5 seeds), Cirrhosis (5 seeds), CDC Diabetes (3 seeds) |
| Algorithms | FedAvg, Ditto, HPFL |
| Loss types | Weighted CE (w = N_total / (N_classes × N_k)), Focal (gamma=2.0) |
| Conditions | IID+noDP, NonIID+noDP, NonIID+eps10 |
| Class weights | Stroke: 0.53/10.36 (19.7×), CDC: 0.58/3.58 (6.2×), Cirrhosis: 0.82/1.29 (1.6×) |

**Key findings:**
- **Collapse nearly eliminated**: from 55.6% (Block AA baseline) to 0.85% (2/234).
  Focal loss achieves zero collapses (0/117). Weighted CE has 2 residual collapses
  (both seed=42, Cirrhosis, NonIID+noDP).
- **Weighted CE wins 18/27 matchups** against focal on mean F1, but focal is more
  robust (zero collapses, lower variance).
- **Best configurations**: weighted_ce + Ditto + NonIID+noDP achieves F1=0.39 on
  Stroke (from 0.00 baseline), F1=0.72 on Cirrhosis, F1=0.57 on CDC Diabetes.
- **Threshold tuning amplifies focal loss**: focal + threshold tuning achieves up
  to +177% F1 improvement (Stroke Ditto NonIID+noDP: 0.18 → 0.50).
- **Ditto is the best algorithm** across all 3 datasets, ranking #1 consistently.

### Block AE: Ditto Local Epochs Sweep (40 experiments)

**Research question:** Does increasing local training epochs improve Ditto's
minority-class rescue? What is the optimal epoch count?

| Factor | Values |
|--------|--------|
| Datasets | Stroke, Cirrhosis |
| Algorithm | Ditto |
| Local epochs | 5, 10 (baseline: 3 from Block AA) |
| Conditions | NonIID+noDP, NonIID+eps10 |
| Seeds | 42, 123, 456, 789, 999 |

**Key findings:**
- **Cirrhosis + noDP: monotonic improvement** — F1 rises from 0.36 (ep=3) to 0.57
  (ep=5) to 0.78 (ep=10), with variance shrinking 5× (std: 0.36 → 0.07).
- **Stroke + noDP: collapse broken at ep=10** — F1 from 0.00 (ep=3) to 0.36 (ep=10),
  but with high variance (std=0.25). At ep=5, 4/5 seeds still collapse.
- **DP interaction**: Under eps=10, improvements plateau. DP already regularizes
  against collapse, leaving less room for epoch-based improvement.
- **ep=5 is a "danger zone"** for Stroke: enough epochs to overfit to majority class,
  not enough to learn minority class patterns.

### Block TH: Threshold Rescue (20 experiments)

**Research question:** Can post-hoc threshold optimization rescue models that
collapsed to F1=0.0? Is the rescue clinically meaningful?

| Factor | Values |
|--------|--------|
| Source | Cascade 9 Block AA experiments with F1=0.0 |
| Models re-trained | 20 (all F1=0.0 from Block AA) |
| Threshold sweep | 0.05 to 0.95, step 0.05 |

**Key findings:**
- **All 20 collapsed models produce TT-F1 > 0** after threshold tuning, but quality
  varies dramatically by dataset.
- **Cirrhosis: effective rescue** — 7/7 models achieve TT-F1 ≥ 0.56, best: 0.87
  (Ditto NonIID+noDP). Optimal thresholds cluster at 0.30-0.45, indicating the
  models learned minority class features but were miscalibrated.
- **Stroke: marginal rescue** — 9/13 models achieve TT-F1 < 0.20. Optimal threshold
  is 0.05 for 12/20 models (the minimum tested), indicating minimal learned signal.
- **Conclusion**: Threshold tuning is effective for moderately imbalanced datasets
  (Cirrhosis) where the model learned discriminative features, but insufficient for
  severely imbalanced datasets (Stroke 4.9%) where the fundamental learning failed.

---

## Standalone Experiment Files

These experiments address specific research questions outside the cascade
structure.

### Scalability + DP x TopK (189 experiments)

**File:** `checkpoint_scalability_topk.json`

| Block | Experiments | Question |
|-------|-------------|----------|
| A_scalability | 135 | Performance at K={3, 5, 10, 15, 20} clients with DP |
| B_dp_topk | 54 | Combined DP + TopK sparsification |

- Datasets: Cardiovascular, PTB-XL
- Algorithms: FedAvg, Ditto, HPFL
- DP: No-DP, eps=1, eps=10
- Seeds: 42, 123, 456

### Non-IID x DP Interaction (126 experiments)

**File:** `checkpoint_noniid_dp.json`

- Datasets: Cardiovascular, PTB-XL
- Algorithms: FedAvg, Ditto, HPFL
- Non-IID alpha: multiple levels
- DP: multiple epsilon values

### Byzantine + DP Multi-Defense (198 experiments)

**File:** `byzantine_dp_v2_results.json`

- Algorithms: FedAvg, HPFL
- Attacks: None, Label flipping, Gradient scaling
- Defenses: None, Median, Krum, Multi-Krum, Trimmed Mean
- DP: No-DP, eps=1, eps=10

**Key finding:** Byzantine + DP are synergistic. Under simultaneous defense,
accuracy is completely DP-invariant.

### DP Per-Class Impact (80 experiments)

**File:** `dp_per_class_results.json`

- Datasets: Breast Cancer, PTB-XL
- Algorithms: FedAvg, HPFL
- Per-class accuracy tracking under varying DP

### Membership Inference Attack (18 experiments)

**File:** `checkpoint_mia.json`

- Datasets: Cardiovascular, PTB-XL
- Algorithms: FedAvg, Ditto, HPFL
- DP: No-DP, eps=1, eps=10

**Key finding:** FedAvg/Ditto AUC ~ 0.50 (inherent FL privacy). HPFL AUC = 0.666
(vulnerable due to personalized heads). DP eps=1 mitigates HPFL to AUC 0.337.

### Partial Client Participation (18 experiments)

**File:** `checkpoint_partial_participation.json`

- Datasets: Cardiovascular, PTB-XL
- 3 of 5 clients per round
- Algorithms: FedAvg, Ditto, HPFL

### Privacy Budget Exhaustion (9 experiments)

**File:** `checkpoint_budget_exhaustion.json`

- Algorithms: FedAvg, Ditto, HPFL
- Budget profiles: Tight, Medium, Generous

### Analysis Cascade Supplement (24 experiments)

**File:** `checkpoint_analysis_cascade.json`

- Ablation studies on Cardiovascular, PTB-XL, Breast Cancer

---

## Imaging Track (Executed on M1 Mac)

Parallel evaluation on medical imaging datasets using ResNet-18 (~11.2M params):

| Experiment Set | Experiments | Datasets |
|---|---|---|
| Imaging Seeds x5 | 24 | Chest X-ray, Brain Tumor, Skin Cancer |
| Non-IID Imaging | 54 | Same, alpha={0.1, 0.5, 1.0} |
| TopK Imaging | 27 | Same, TopK={1.0, 0.05, 0.01} |
| DP Imaging | ~40 | Same, eps={1, 10} |
| Extended | ~20 | HPFL, FedLESAM on chest X-ray |

**Key finding:** Ditto +28.0pp on Brain Tumor (p=0.015). HPFL counterproductive
on all imaging tasks. Personalization effect is method-dependent, NOT
modality-dependent.

---

## Experiment Infrastructure

### Atomic Checkpoint Pattern

All cascades use the same atomic save pattern to ensure experiment
reproducibility and crash resilience:

```python
fd, tmp = tempfile.mkstemp(dir=output_dir, prefix=".ckpt_", suffix=".tmp")
with os.fdopen(fd, "w") as f:
    json.dump(data, f, indent=2, default=str)
    f.flush()
    os.fsync(f.fileno())
if path.exists():
    shutil.copy2(path, path + ".bak")
os.replace(tmp, path)
```

### Reproducibility

- All experiments use explicit random seeds (typically 42, 123, 456)
- PyTorch manual seeds set before model creation
- NumPy RandomState for data partitioning
- Data caching prevents re-partitioning across experiments within a cascade

### Signal Handling

All cascades support graceful shutdown (SIGINT/Ctrl+C):
- Current experiment completes
- Checkpoint saved
- Resume from last checkpoint on next run

---

## Research Questions Index

For quick reference, all research questions addressed across cascades:

| # | Question | Cascade.Block |
|---|----------|---------------|
| 1 | How does DP interact with learning rate selection? | 2.D |
| 2 | How much training data is needed for adequate FL? | 3.C |
| 3 | How do algorithms perform under compound stress? | 3.D |
| 4 | Can fairness-aware aggregation solve class collapse? | 4.A |
| 5 | How resilient is FL to concept drift? | 4.B |
| 6 | How do personalization methods compare? | 4.C |
| 7 | Can FL comply with GDPR right to erasure? | 4.D |
| 8 | Which client selection strategy is optimal? | 5.E |
| 9 | How vulnerable are gradients to reconstruction? | 5.F |
| 10 | Do advantages generalize across clinical domains? | 6.G |
| 11 | Is hierarchical FL viable for EHDS? | 6.H |
| 12 | How robust is FL to communication delays? | 6.I |
| 13 | Can FL learn multiple clinical tasks? | 6.J |
| 14 | Which Byzantine defense is most effective? | 6.K |
| 15 | Can we fairly quantify hospital contributions? | 6.L |
| 16 | Are FL probability outputs well-calibrated? | 7.M |
| 17 | Does conformal prediction maintain coverage? | 7.N |
| 18 | Are feature rankings consistent across clients? | 7.O |
| 19 | Do FL algorithms exhibit demographic biases? | 7.P |
| 20 | Can FL detect and adapt to temporal drift? | 7.Q |
| 21 | How does privacy budget accumulate across studies? | 7.R |
| 22 | What is the overhead of secure aggregation? | 8.S |
| 23 | Which compression method is optimal? | 8.T |
| 24 | Is local or central DP more effective? | 8.U |
| 25 | How does vertical FL perform with partial overlap? | 8.V |
| 26 | Can FL handle 253K-sample population health data? | 8.W |
| 27 | How does FL scale to K=50 hospitals? | 9.X |
| 28 | Are DP and compression compatible? | 9.Y |
| 29 | How many rounds are needed for convergence? | 9.Z |
| 30 | Can personalized FL handle extreme class imbalance? | 9.AA |
| 31 | Is the Ditto advantage robust to heterogeneity level? | 9.AB |
| 32 | Does DP noise act as implicit regularizer against majority-class collapse? | 10.AC |
| 33 | Can weighted CE or focal loss eliminate class-imbalance collapse in FL? | 10.AD |
| 34 | Does increasing local epochs improve Ditto's minority-class rescue? | 10.AE |
| 35 | Can post-hoc threshold tuning rescue collapsed FL models? | 10.TH |

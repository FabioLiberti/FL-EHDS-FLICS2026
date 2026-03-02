# FL-EHDS FLICS 2026 — SAL (Stato Avanzamento Lavori)

**Ultimo aggiornamento**: 2026-03-01
**Autore**: Fabio Liberti
**Conferenza**: FLICS 2026, Valencia, 9-12 giugno 2026
**Paper**: "FL-EHDS: A Privacy-Preserving Federated Learning Framework for the European Health Data Space"

---

## ESPERIMENTI COMPLETATI: 2,023 + ~208 in corso

### Riepilogo per Blocco

| # | Blocco | Esperimenti | Checkpoint | Macchina | Status |
|---|--------|-------------|------------|----------|--------|
| 1 | Tabular Baseline (optimized) | 105 | checkpoint_tabular.json | Air M3 | DONE |
| 2 | Seeds x10 | 105 | checkpoint_seeds10.json | Air M3 | DONE |
| 3 | Differential Privacy | 180 | checkpoint_dp.json | Air M3 | DONE |
| 4 | Art.71 Opt-out (statico) | 225 | checkpoint_optout.json | Air M3 | DONE |
| 5 | Deep MLP | 70 | checkpoint_deep_mlp.json | Air M3 | DONE |
| 6 | Scalability K=50,100 | 84 | checkpoint_scalability.json | Air M3 | DONE |
| 7 | Scalability + DP | 54 | checkpoint_scalability_dp.json | Air M3 | DONE |
| 8 | Scalability DP (CV) | 18 | checkpoint_scalability_dp_cv.json | Air M3 | DONE |
| 9 | Epochs Sweep | 140 | checkpoint_epochs_sweep.json | Air M3 | DONE |
| 10 | Top-K PTB-XL | 9 | checkpoint_topk_ptbxl.json | Air M3 | DONE |
| 11 | Confusion Matrix BC | 40 | checkpoint_confusion_bc.json | Air M3 | DONE |
| 12 | RDP vs Naive Composition | 1 | rdp_comparison_results.json | Air M3 | DONE |
| 13 | Byzantine + DP (BC) | 198 | byzantine_dp_v2_results.json | Air M3 | DONE |
| 14 | Byzantine + DP (CV) | 198 | checkpoint_byzantine_dp_cv.json | Air M3 | DONE |
| 15 | DP Per-class DEI | 80 | dp_per_class_results.json | Air M3 | DONE |
| 16 | Imaging CNN Baseline | 27 | (in p12_multidataset) | Pro M1 | DONE |
| 17 | Imaging Multidataset 5 seeds | 129 | checkpoint_p12_multidataset.json | Pro M1 | DONE |
| 18 | Imaging Seeds5 expansion | 24 | checkpoint_imaging_seeds5.json | Pro M1 | DONE |
| 19 | Imaging DP | 108 | checkpoint_imaging_dp.json | Pro M1 | DONE |
| 20 | Completion (delta) | 8 | checkpoint_completion.json | Pro M1 | DONE |
| 21 | Confusion Matrix BT | 6 | checkpoint_confusion_bt.json | Pro M1 | DONE |
| 22 | Chest X-ray Extended | 6 | checkpoint_chest_extended.json | Pro M1 | DONE |
| 23 | Confusion Matrix Chest | 6 | checkpoint_confusion_chest.json | Pro M1 | DONE |
| 24 | Significance Tests | 5 | checkpoint_p21_significance.json | Pro M1 | DONE |
| 25 | Attack Robustness | 4 | checkpoint_p22_attack.json | Pro M1 | DONE |
| — | **EXTENDED CASCADE (234 exp, 43h 37m)** | | | **Air M3** | |
| 26 | DP Gradient Clipping | 72 | checkpoint_dp_clipping.json | Air M3 | DONE |
| 27 | Centralized vs Federated | 45 | checkpoint_centralized_vs_fed.json | Air M3 | DONE |
| 28 | DP on PTB-XL 5-class | 36 | checkpoint_dp_ptbxl.json | Air M3 | DONE |
| 29 | Top-K Imaging | 27 | checkpoint_topk_imaging.json | Air M3 | DONE |
| 30 | Non-IID Imaging | 54 | checkpoint_noniid_imaging.json | Air M3 | DONE |
| — | **NUOVI ESPERIMENTI (sessione corrente)** | | | **Air M3** | |
| 31 | **Cross-Border Heterogeneous DP** | 45 | checkpoint_crossborder_dp.json | Air M3 | DONE |
| 32 | **Dynamic Art.71 Opt-Out** | 90 | checkpoint_dynamic_optout.json | Air M3 | DONE |
| — | **FINAL CASCADE (45 exp, 12m 11s)** | | | **Air M3** | |
| 33 | Membership Inference Attack (MIA) | 18 | checkpoint_mia.json | Air M3 | DONE |
| 34 | Partial Client Participation (3/5) | 18 | checkpoint_partial_participation.json | Air M3 | DONE |
| 35 | Privacy Budget Exhaustion | 9 | checkpoint_budget_exhaustion.json | Air M3 | DONE |
| — | **IN CORSO** | | | | |
| 36 | Imaging Opt-out (Art.71) | ~154 | (Colab) | Colab | IN PROGRESS |
| 37 | Imaging Scalability K=10,20 | 54 | (RunPod) | RunPod | IN PROGRESS |

**TOTALE COMPLETATI: 2,068 esperimenti**
**IN CORSO: ~208 esperimenti**

---

## DATASET (8 totali, tutti in locale)

### Tabular (MLP ~10K params)
| Dataset | Records | Classes | Clients | Partitioning |
|---------|---------|---------|---------|-------------|
| PTB-XL ECG | 21,799 | 5 (NORM,MI,STTC,CD,HYP) | 5 | partition_by_site (52 siti EU) |
| Cardiovascular | 70,000 | 2 | 5 | Dirichlet α=0.5 |
| Breast Cancer | 569 | 2 | 3 | Dirichlet α=0.5 |
| Heart Disease UCI | 920 | 2 | 4 | Natural (4 ospedali) |
| Diabetes 130-US | 101,766 | 2 | 5 | Dirichlet α=0.5 |

### Imaging (ResNet-18 ~11.2M params)
| Dataset | Images | Classes | Path locale |
|---------|--------|---------|-------------|
| Brain Tumor MRI | 3,064 | 4 | data/Brain_Tumor/ |
| Skin Cancer | 3,297 | 2 | data/Skin Cancer/ (SPAZIO nel nome!) |
| Chest X-ray | 5,856 | 2 | data/chest_xray/ |

---

## FINDING PRINCIPALI (per il paper)

### 1. Personalizzazione domina (Sec. 4)
- Ditto/HPFL 12.6-26.8pp sopra FedAvg
- HPFL p<0.001 su tutti i dataset core (pooled Wilcoxon, 10 seed)
- Solo metodi con modelli locali separati (Ditto, HPFL) differenziano su tabular compatto

### 2. DP essenzialmente gratis a ε=10 (Sec. 4)
- <2pp di costo su tutti gli algoritmi, da K=5 a K=50
- Cliff tra ε=1 e ε=10 su PTB-XL (diverso da CV che ha cliff a ε=10)
- HPFL più robusto: -3.85pp a ε=1 su PTB-XL vs FedAvg -26.48pp

### 3. DP Clipping Sensitivity (Blocco 26)
- C=0.5 ottimale per 4/6 combo, C=1.0 per 2/6
- FedAvg più sensibile (16.35pp range su CV), Ditto più robusto (0.51pp su PX)
- Raccomandazione: C=1.0 come default robusto

### 4. Centralized vs Federated (Blocco 27)
- PTB-XL unico FL success con FedAvg: gap solo 1.02pp, FL gain +7.80pp
- BC, CV, HD, DM: FL peggiore del training locale di 16-23pp con FedAvg
- Dimostra che FedAvg da solo non basta su dati eterogenei

### 5. Top-K Imaging (Blocco 29)
- Skin Cancer MIGLIORA con Top-5% (+2.4pp, tutti i seed — regularizzazione implicita)
- Brain Tumor non tollera compressione (-30% a Top-5%)
- chest_xray: trade-off moderato (-8pp a 95% BW savings)

### 6. Non-IID Imaging (Blocco 30)
- HPFL trasforma pipeline rotte a α=0.1:
  - BT: +38.9pp (40.9% → 79.7%)
  - SC: +29.7pp (65.9% → 95.7%)
  - CX: +22.8pp (72.8% → 95.6%)
- HPFL mantiene fairness anche a α=0.1 (Jain>0.90)
- Counter-intuivo: HPFL migliora al diminuire di alpha

### 7. ★ Cross-Border Heterogeneous DP (Blocco 31) — UNICO
- **Ditto cross-border invariante**: solo -0.9pp da No-DP a Mixed (2×DE:ε=1 + 3×IT:ε=10)
- **HPFL collassa sotto local DP**: 54.4% a ε=1 uniforme (vs 88.77% con central DP)
- **Mixed meglio di strictest-wins per FedAvg**: 89.3% vs 85.6% (+3.8pp)
- DEI crolla per tutti sotto Mixed (Ditto: 0.767→0.296) — accuracy nasconde danno diagnostico
- **Nessun altro paper FL testa budget DP eterogenei per client**

### 8. ★ Dynamic Art.71 Opt-Out (Blocco 32) — UNICO
- **Ritiro dinamico essenzialmente gratis su PTB-XL**: <1pp anche con 40% capacity loss
- **Ditto/HPFL shock-resistant**: 0.0pp immediate drop su CV vs FedAvg -6.0pp
- **HPFL protegge client ritirato su CV**: 76.1% vs 55% di FedAvg/Ditto (classifier head congelato)
- **Late withdrawal (R20) = zero cost** per tutti gli algoritmi
- **Recovery in 1 round** per Ditto/HPFL, FedAvg lento o mai
- **Nessun altro paper FL studia il ritiro mid-training sistematicamente**

### 9. Byzantine + DP sinergistici (Blocco 13-14)
- Difese Byzantine rendono accuracy completamente DP-invariante
- HPFL naturalmente resiliente: DP migliora robustezza (59.7%→86.4% a ε=10)

### 10. DEI (Diagnostic Equity Index) — Metrica originale
- DEI = min_c(R_c) · (1 - CV(R)), penalizza worst-case class + varianza
- FedAvg DEI 0.092 vs HPFL DEI 0.740 su BC (8×)
- DEI stabile sotto DP per HPFL (0.724-0.756), FedAvg sempre near-zero

### 11. MIA — Membership Inference Attack (Blocco 33)
- **FedAvg/Ditto naturalmente immuni** (AUC≈0.50, random guessing)
- **HPFL vulnerabile** (AUC=0.666 su PTB-XL) — personalized heads memorizzano
- **DP a eps=10 NON chiude la falla HPFL** (AUC=0.653)
- **DP a eps=1 riduce leakage HPFL** (AUC=0.337) con 75.5% accuracy preservata
- **Privacy-personalization tension**: la stessa architettura che migliora DEI aumenta MIA vulnerability

### 12. Partial Client Participation (Blocco 34)
- **Regularizzazione implicita**: 3/5 client per round non degrada accuracy
- **HPFL più stabile** cross-seed (std 0.22% vs 0.62% FedAvg)
- Su CV, HPFL **migliora** sotto partial participation (87.6% vs 87.0%)

### 13. Privacy Budget Exhaustion (Blocco 35)
- **0 deattivazioni**: noise multiplier calibrato distribuisce budget su tutti i 30 round
- **Budget tight (ε=2)**: FedAvg crolla a 41%, HPFL mantiene 81% (2×)
- **Personalizzazione necessaria** (non opzionale) sotto vincoli di privacy stretti

---

## FINDING HPFL: LOCAL DP vs CENTRAL DP (ATTENZIONE!)

Risultato critico emerso dal Cross-Border DP:
- **Central DP (ε=1)**: HPFL mantiene 88.77% su PTB-XL (solo -3.85pp)
- **Local DP (ε=1)**: HPFL collassa a 54.4% (-38.2pp)

Lo shared backbone di HPFL viene corrotto dal rumore locale sui gradienti.
Ditto è immune perché mantiene modelli locali completi.

**Implicazione**: per EHDS cross-border con local DP, usare Ditto (non HPFL).

---

## ARCHITETTURA MACCHINE

| Macchina | Ruolo | Git Status |
|----------|-------|------------|
| MacBook Air M3 | Tabular + light imaging + esperimenti EHDS-specifici | v12.8 + nuovi esperimenti |
| MacBook Pro M1 | Imaging pesante | v12.6, da sincronizzare dopo RunPod |
| RunPod GPU | Imaging scalability | IN CORSO (54 exp) |
| Google Colab | Imaging opt-out | IN CORSO (~154 exp) |

**Protocollo sync**: commit → pull --rebase origin main → push

---

## FILE CHIAVE DEL PROGETTO

### Paper
- `paper/paper2rel/flics_fl_ehds_v2A.tex` — Paper principale (753 righe, IEEE 8pp)
- `paper/paper2rel/flics_fl_ehds_supplementary_v2A.tex` — Supplementary
- `paper/paper2rel/figures/` — Figure

### Script Esperimenti
- `benchmarks/run_dp_clipping.py` — DP clipping sensitivity (72 exp)
- `benchmarks/run_centralized_vs_federated.py` — Local/Central/FL (45 exp)
- `benchmarks/run_dp_ptbxl.py` — DP su PTB-XL 5-class (36 exp)
- `benchmarks/run_topk_imaging.py` — Top-K sparsification (27 exp)
- `benchmarks/run_noniid_imaging.py` — Non-IID imaging (54 exp)
- `benchmarks/run_crossborder_dp.py` — ★ Cross-Border DP eterogeneo (45 exp)
- `benchmarks/run_dynamic_optout.py` — ★ Dynamic Art.71 opt-out (90 exp)
- `benchmarks/run_extended.sh` — Cascade runner per blocchi 26-30

### Checkpoint Results
- `benchmarks/paper_results_tabular/` — 20 file JSON (tabular)
- `benchmarks/paper_results_delta/` — 7 file JSON (imaging delta)
- `benchmarks/paper_results/` — 5 file JSON (imaging core)

### Framework
- `terminal/training/federated.py` — FederatedTrainer (train_round, _aggregate, DP)
- `governance/jurisdiction_privacy.py` — JurisdictionPrivacyManager (per-client DP)
- `data/ptbxl_loader.py` — PTB-XL loader con partition_by_site
- `EXPERIMENT_STATUS.md` — Status tracking (aggiornato da git hooks)

---

## TODO PENDENTI

### Alta Priorità
1. ~~**Integrare nel paper LaTeX** i risultati nuovi~~ — **FATTO** (v2A aggiornato)
   - ✅ Cross-Border Heterogeneous DP (Blocco 31)
   - ✅ Dynamic Art.71 Opt-Out (Blocco 32)
   - ✅ MIA, Partial Participation, Budget Exhaustion (Blocchi 33-35)
   - ✅ Abstract, Intro, Section 4, Discussion 5.1, Conclusions aggiornati
   - ✅ Supplementary: 5 nuove tabelle (S-CrossBorder, S-DynOptout, S-MIA, S-Partial, S-Budget)
   - ⚠️ **Paper ora a 12 pagine** — serve comprimere a 8 (IEEE limit)
2. **Commit e push** di tutti i nuovi file
3. **Sync Pro M1** dopo completamento RunPod

### Media Priorità
4. **Comprimere paper a 8 pagine** — spostare contenuto in supplementary
5. **Convergence Analysis** — estrarre curve round-by-round dai checkpoint esistenti (ZERO computazione)
6. **Aggiornare EXPERIMENT_STATUS.md** con i nuovi esperimenti

### Bassa Priorità
7. DP + Non-IID combinati su imaging (richiede GPU)
8. DEI su imaging sotto DP (richiede GPU)
9. Full Stack Stress Test (DP + Non-IID + Top-K + Byzantine) (richiede GPU)

---

## NOTE TECNICHE

### Safety Features (tutti gli script)
- Atomic checkpoint: `tempfile.mkstemp` + `os.fdopen` + `os.fsync` + `os.replace`
- Backup: `.bak` prima di ogni sovrascrittura
- SIGINT/SIGTERM handler con salvataggio checkpoint
- Auto-resume: rilancia senza `--fresh` per riprendere
- `--quick` per validazione rapida
- `--fresh` per ripartire da zero
- GPU cleanup dopo ogni esperimento

### Pattern Cross-Border DP (per-client local DP)
```python
# Monkey-patch _train_client per settare dp_epsilon diverso per client
def make_patched(orig, eps_map, trn):
    def patched_train_client(client_id, round_num):
        old_eps = trn.dp_epsilon
        trn.dp_epsilon = eps_map[client_id]
        try:
            result = orig(client_id, round_num)
        finally:
            trn.dp_epsilon = old_eps
        return result
    return patched_train_client
```

### Pattern Dynamic Opt-Out
```python
# Usa active_clients parameter di train_round
for r in range(num_rounds):
    if withdraw_round is not None and r >= withdraw_round:
        active = [c for c in all_clients if c not in withdraw_clients]
    else:
        active = all_clients
    result = trainer.train_round(r, active_clients=active)
```

### Path dataset imaging con spazio
```python
# ATTENZIONE: "Skin Cancer" con spazio, NON "Skin_Cancer"
FRAMEWORK_DIR / "data" / "Skin Cancer"
```

# Guida Completa: Esperimenti FL-EHDS su Google Colab

## Indice

1. [Contesto e Obiettivo](#1-contesto-e-obiettivo)
2. [Modifiche Algoritmiche Implementate](#2-modifiche-algoritmiche-implementate)
3. [Confronto Configurazioni: Locale vs Colab](#3-confronto-configurazioni-locale-vs-colab)
4. [Preparazione Colab](#4-preparazione-colab)
5. [Piano di Esecuzione a Micro-Batch](#5-piano-di-esecuzione-a-micro-batch)
6. [Istruzioni Passo-Passo](#6-istruzioni-passo-passo)
7. [Recupero Risultati](#7-recupero-risultati)
8. [Fallback Locale](#8-fallback-locale)

---

## 1. Contesto e Obiettivo

### Stato esperimenti

| Tipo | Completati | Target | Stato |
|------|-----------|--------|-------|
| Tabulari (Diabetes, Heart_Disease) | 30/30 | 30 | COMPLETO |
| Imaging (Brain_Tumor, chest_xray, Skin_Cancer) | 0/45 | 45 | DA FARE |
| **Totale** | **30/75** | **75** | **40%** |

I 30 esperimenti tabulari sono completi (5 algoritmi x 3 seed x 2 dataset) e non vanno
ripetuti. Restano 45 esperimenti imaging: 3 dataset x 5 algoritmi x 3 seed.

### Perche' Colab

| | Locale (MPS Apple) | Colab T4 | Colab A100 |
|---|---|---|---|
| Tempo per esperimento | ~25-35 min | ~4-6 min | ~2-3 min |
| 45 esperimenti | ~20 ore | ~3-4 ore | ~1.5-2 ore |
| Qualita' risultati | Ridotta (config alleggerita) | Ottimale | Ottimale |

Su locale, per contenere i tempi, avevamo ridotto a 1 epoca e 12 round con accuracy
risultante di 0.46-0.54. Su Colab possiamo usare la configurazione completa.

---

## 2. Modifiche Algoritmiche Implementate

### 2.1 FedBN (Federated Batch/Group Normalization)

**Problema**: In FL non-IID, le statistiche dei layer di normalizzazione (media, varianza)
differiscono tra client. Aggregarle corrompe le rappresentazioni locali.

**Soluzione**: Durante l'aggregazione, i parametri dei GroupNorm NON vengono mediati.
Ogni client mantiene le proprie statistiche di normalizzazione.

**Riferimento**: Li et al., "FedBN: Federated Learning on Non-IID Features via Local
Batch Normalization", ICLR 2021.

**Impatto atteso**: +5-10pp accuracy su dati non-IID.

**Flag**: `use_fedbn=True` (default: False, backward compatible)

### 2.2 Freeze Parziale del Backbone

**Prima**: `freeze_backbone=True` congelava conv1 + bn1 + layer1 + layer2 (~6M params frozen,
~5M trainabili). Troppo restrittivo per adattare il modello a imaging medico.

**Ora**: `freeze_level` controlla la granularita':

| freeze_level | Layer congelati | Params trainabili | Uso |
|---|---|---|---|
| 0 | Nessuno | ~11.2M | Fine-tuning completo |
| 1 | conv1 + bn1 | ~9.5M | **Colab (raccomandato)** |
| 2 | + layer1 | ~7.8M | Compromesso |
| 3 | + layer2 | ~5.0M | Locale (veloce) |

**Razionale Colab**: `freeze_level=1` congela solo il primo strato convoluzionale
(feature generiche come edge, texture) e lascia liberi i layer che apprendono
feature specifiche per imaging medico.

**Backward compatible**: `freeze_backbone=True` equivale a `freeze_level=3`.

### 2.3 Cosine Learning Rate Scheduling

**Prima**: LR fisso per tutti i round.

**Ora**: LR segue una curva coseno:
- Round 1: LR pieno (esplorazione)
- Round N/2: LR al 55% (transizione)
- Round N: LR al 10% (raffinamento)

Formula: `lr(t) = lr_base * (0.1 + 0.9 * (1 + cos(pi * t/T)) / 2)`

Questo era gia' presente nel codice (`_get_round_lr`) ma non veniva attivato
perche' `num_rounds` non era impostato. Ora viene impostato automaticamente.

### 2.4 Algoritmi Moderni (2022-2023)

Sostituiti gli algoritmi datati con SOTA recente:

| Algoritmo | Anno | Venue | Caratteristica chiave |
|---|---|---|---|
| **FedAvg** | 2017 | AISTATS | Baseline di riferimento |
| **FedLC** | 2022 | ICML | Calibrazione logit per label skew |
| **FedSAM** | 2022 | ICML | Sharpness-aware per generalizzazione |
| **FedDecorr** | 2023 | ICLR | Anti-collapse dimensionale |
| **FedExP** | 2023 | ICLR | Convergenza accelerata server-side |

Tutti e 5 erano gia' implementati nel trainer. La modifica e' solo nella lista
degli algoritmi testati per il paper.

---

## 3. Confronto Configurazioni: Locale vs Colab

### Profilo LOCALE (flag: nessuno, default)

```
num_rounds=12, local_epochs=1, freeze_backbone=True (level 3)
use_fedbn=False, early_stopping: patience=4, min_rounds=6
```

Accuracy attesa: 0.50-0.60 | Tempo: ~25 min/exp

### Profilo COLAB (flag: --colab)

```
num_rounds=20, local_epochs=3, freeze_level=1
use_fedbn=True, early_stopping: patience=5, min_rounds=10
```

Accuracy attesa: 0.75-0.90 | Tempo su T4: ~5 min/exp

### Cosa cambia il flag --colab

Il flag `--colab` nella riga di comando sostituisce automaticamente:
- `IMAGING_CONFIG` -> `COLAB_IMAGING_CONFIG`
- `DATASET_OVERRIDES` -> `COLAB_DATASET_OVERRIDES`
- `EARLY_STOPPING_CONFIG` -> `COLAB_EARLY_STOPPING_CONFIG`

Non modifica nient'altro. I risultati tabulari gia' completati NON vengono toccati.

---

## 4. Preparazione Colab

### 4.1 File da caricare su Google Drive

Creare la cartella `MyDrive/FL-EHDS-FLICS2026/` e caricare:

```
FL-EHDS-FLICS2026/
  fl-ehds-framework/           <-- intero framework
    benchmarks/
      paper_results/
        checkpoint_p12_multidataset.json   <-- contiene i 30 risultati tabulari
      run_paper_experiments.py
    terminal/
      training/
        federated_image.py
        models.py
        ...
    data/
      Brain_Tumor/              <-- ~500 MB
        glioma_tumor/
        healthy/
        meningioma_tumor/
        pituitary_tumor/
      chest_xray/               <-- ~1.2 GB
        train/
        test/
      Skin Cancer/              <-- ~300 MB
        benign/
        malignant/
    notebooks/
      colab_experiments.ipynb
```

**Dimensione totale**: ~2 GB (dataset) + ~50 MB (codice)

**Tempo upload**: ~10-20 min con connessione media

### 4.2 Aprire il notebook

1. Vai su https://colab.research.google.com
2. File > Apri notebook > Google Drive > FL-EHDS-FLICS2026/fl-ehds-framework/notebooks/colab_experiments.ipynb
3. Runtime > Cambia tipo di runtime > **GPU** (T4 gratuito, A100 con Pro)
4. Esegui le celle di setup (1-4)

### 4.3 Verifiche preliminari

La cella "Verify GPU" deve mostrare:
```
CUDA available: True
GPU: Tesla T4 (o A100-SXM4-40GB)
```

La cella "Verify datasets" deve mostrare:
```
Brain_Tumor: ~7023 files
chest_xray: ~5856 files
Skin Cancer: ~3297 files
```

---

## 5. Piano di Esecuzione a Micro-Batch

### Strategia

Come in locale, gli esperimenti sono scomposti in micro-batch indipendenti.
Ogni micro-batch salva nel checkpoint e puo' essere interrotto/ripreso.

### Ordine di esecuzione raccomandato

Eseguire **per dataset**, in modo da avere risultati completi e autoconsistenti
per almeno un dataset il prima possibile.

#### FASE 1: Brain_Tumor (15 esperimenti, ~1h su T4)

| Micro-batch | Comando | Esperimenti | Tempo T4 |
|---|---|---|---|
| BT-1 | `--colab --resume --only p12 --dataset Brain_Tumor --algo FedAvg` | 3 | ~15 min |
| BT-2 | `--colab --resume --only p12 --dataset Brain_Tumor --algo FedLC` | 3 | ~15 min |
| BT-3 | `--colab --resume --only p12 --dataset Brain_Tumor --algo FedSAM` | 3 | ~15 min |
| BT-4 | `--colab --resume --only p12 --dataset Brain_Tumor --algo FedDecorr` | 3 | ~15 min |
| BT-5 | `--colab --resume --only p12 --dataset Brain_Tumor --algo FedExP` | 3 | ~15 min |

**Dopo BT-5**: hai 15/45 imaging completi. Puoi gia' generare la tabella
Brain_Tumor per il paper.

**Comando unico alternativo** (se non temi timeout):
```
python -m benchmarks.run_paper_experiments --colab --resume --only p12 --dataset Brain_Tumor
```

#### FASE 2: chest_xray (15 esperimenti, ~1.5h su T4)

| Micro-batch | Comando | Esperimenti | Tempo T4 |
|---|---|---|---|
| CX-1 | `--colab --resume --only p12 --dataset chest_xray --algo FedAvg` | 3 | ~20 min |
| CX-2 | `--colab --resume --only p12 --dataset chest_xray --algo FedLC` | 3 | ~20 min |
| CX-3 | `--colab --resume --only p12 --dataset chest_xray --algo FedSAM` | 3 | ~20 min |
| CX-4 | `--colab --resume --only p12 --dataset chest_xray --algo FedDecorr` | 3 | ~20 min |
| CX-5 | `--colab --resume --only p12 --dataset chest_xray --algo FedExP` | 3 | ~20 min |

chest_xray e' il dataset piu' grande (~5856 immagini), quindi impiega un po' di piu'.

**Dopo CX-5**: hai 30/45 imaging completi (60/75 totali).

#### FASE 3: Skin_Cancer (15 esperimenti, ~45 min su T4)

| Micro-batch | Comando | Esperimenti | Tempo T4 |
|---|---|---|---|
| SC-1 | `--colab --resume --only p12 --dataset Skin_Cancer --algo FedAvg` | 3 | ~10 min |
| SC-2 | `--colab --resume --only p12 --dataset Skin_Cancer --algo FedLC` | 3 | ~10 min |
| SC-3 | `--colab --resume --only p12 --dataset Skin_Cancer --algo FedSAM` | 3 | ~10 min |
| SC-4 | `--colab --resume --only p12 --dataset Skin_Cancer --algo FedDecorr` | 3 | ~10 min |
| SC-5 | `--colab --resume --only p12 --dataset Skin_Cancer --algo FedExP` | 3 | ~10 min |

**Dopo SC-5**: tutti i 45 imaging completi. **75/75 esperimenti totali.**

### Riepilogo tempi

| Fase | Dataset | Esperimenti | Tempo T4 | Tempo A100 |
|---|---|---|---|---|
| 1 | Brain_Tumor | 15 | ~1h 15min | ~40 min |
| 2 | chest_xray | 15 | ~1h 40min | ~55 min |
| 3 | Skin_Cancer | 15 | ~50 min | ~30 min |
| **Totale** | **Tutti** | **45** | **~3h 45min** | **~2h 05min** |

### Punti di autoconsistenza

Dopo ogni FASE hai risultati completi e pubblicabili per quel dataset:
- **Dopo Fase 1**: tabella Brain_Tumor (4 classi, imaging cerebrale)
- **Dopo Fase 2**: tabella chest_xray (2 classi, radiografie toraciche)
- **Dopo Fase 3**: tabella Skin_Cancer (2 classi, dermatoscopia)

Puoi fermarti dopo qualsiasi fase e avere comunque risultati utilizzabili nel paper.

---

## 6. Istruzioni Passo-Passo

### Passo 1: Upload su Drive (~15 min)

1. Comprimi `fl-ehds-framework/` in un file zip
2. Carica su Google Drive nella cartella desiderata
3. Oppure usa `git clone` se il repo e' accessibile

### Passo 2: Apri Colab e Setup (~5 min)

1. Apri `colab_experiments.ipynb` da Drive
2. Imposta GPU: Runtime > Change runtime type > GPU
3. Esegui la cella "Mount Google Drive"
4. Esegui la cella "Setup" (scegli l'opzione di upload corretta)
5. Esegui "Install dependencies"
6. Esegui "Verify GPU" — deve mostrare CUDA available: True
7. Esegui "Verify datasets" — tutti e 3 devono essere presenti

### Passo 3: Pulisci checkpoint (~1 min)

Esegui la cella "Clean imaging checkpoint". Questo rimuove eventuali risultati
imaging vecchi (con config diversa) ma MANTIENE i 30 risultati tabulari.

### Passo 4: Esegui Fase 1 — Brain_Tumor (~1h)

In una nuova cella:
```python
%cd /content/fl-ehds-framework
!python -m benchmarks.run_paper_experiments \
    --colab --resume --only p12 --dataset Brain_Tumor
```

**Cosa vedrai**: per ogni esperimento, una riga tipo:
```
[1/15] Brain_Tumor / FedAvg / seed=42 -> acc=0.823 jain=0.945 (312s)
```

Al termine, verifica con la cella "Verify Results".

### Passo 5: Esegui Fase 2 — chest_xray (~1.5h)

```python
!python -m benchmarks.run_paper_experiments \
    --colab --resume --only p12 --dataset chest_xray
```

### Passo 6: Esegui Fase 3 — Skin_Cancer (~45 min)

```python
!python -m benchmarks.run_paper_experiments \
    --colab --resume --only p12 --dataset Skin_Cancer
```

### Passo 7: Verifica e Download

1. Esegui la cella "Verify Results" — deve mostrare 75/75
2. Esegui la cella "Download Results" — copia i JSON su Drive
3. Scarica la cartella `FL-EHDS-results/` da Drive al computer locale

### Passo 8: Genera output in locale

Sul computer locale, copia i file JSON in:
```
fl-ehds-framework/benchmarks/paper_results/
```

Poi:
```bash
cd fl-ehds-framework
python -m benchmarks.run_paper_experiments --only output
```

Questo genera tutte le tabelle e figure per il paper.

---

## 7. Recupero Risultati

### Se Colab si disconnette

Nessun problema. Il sistema di checkpoint salva dopo OGNI esperimento.

1. Riapri il notebook
2. Ri-monta Drive
3. Ri-lancia lo stesso comando con `--resume`
4. Riparte automaticamente dall'ultimo esperimento completato

### Se vuoi risultati parziali

Dopo ogni fase puoi scaricare `checkpoint_p12_multidataset.json` e generare
le tabelle in locale. I risultati sono incrementali.

### Formato del checkpoint

```json
{
  "completed": {
    "Brain_Tumor_FedAvg_42": {
      "final_metrics": {"accuracy": 0.823, "f1": 0.81, ...},
      "best_metrics": {"accuracy": 0.835, ...},
      "best_round": 14,
      "stopped_early": true,
      "actual_rounds": 18,
      "history": [...],
      "config": {...},
      "fairness": {"jain_index": 0.945, "gini": 0.032, ...},
      "runtime_seconds": 312
    },
    ...
  }
}
```

---

## 8. Fallback Locale

Se Colab non funziona o non e' disponibile, il sistema locale resta invariato.

### Esecuzione locale (config alleggerita)

```bash
cd fl-ehds-framework

# Micro-batch singolo (1 algo x 1 dataset x 3 seed, ~1h)
python -m benchmarks.run_paper_experiments --slice 1 --resume

# Slices disponibili: 1-15 (vedi SLICE_DEFINITIONS nel codice)
# 1-5: Brain_Tumor (FedAvg, FedLC, FedSAM, FedDecorr, FedExP)
# 6-10: chest_xray
# 11-15: Skin_Cancer
```

### Differenze config locale

| Parametro | Locale | Colab |
|---|---|---|
| num_rounds | 12 | 20 |
| local_epochs | 1 | 3 |
| freeze_level | 3 (completo) | 1 (parziale) |
| use_fedbn | No | Si |
| early_stop patience | 4 | 5 |
| early_stop min_rounds | 6 | 10 |
| Accuracy attesa | 0.50-0.60 | 0.75-0.90 |

### IMPORTANTE: Non mischiare risultati

I risultati locali e Colab usano configurazioni diverse e NON sono confrontabili.
Scegli UNA delle due strade e completa tutti gli imaging con quella.

Se inizi su Colab, completa su Colab. Se devi tornare in locale, pulisci prima
i risultati imaging dal checkpoint.

---

## Appendice: File Modificati

| File | Modifica | Backward compatible |
|---|---|---|
| `terminal/training/models.py` | Aggiunto `freeze_level` (0-3) | Si, `freeze_backbone=True` = level 3 |
| `terminal/training/federated_image.py` | Aggiunto `use_fedbn`, `freeze_level` | Si, default False/None |
| `benchmarks/run_paper_experiments.py` | Aggiunto `--colab`, config Colab | Si, senza flag = config locale |
| `notebooks/colab_experiments.ipynb` | NUOVO: notebook Colab | N/A |

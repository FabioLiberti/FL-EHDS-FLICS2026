# FL-EHDS Terminal Interface

## Documentazione Completa

**Data**: 5 Febbraio 2026
**Autore**: Fabio Liberti
**Versione**: 1.0.0

---

## Panoramica

Interfaccia a riga di comando (CLI) per il framework FL-EHDS che permette l'accesso a tutte le funzionalita di Federated Learning senza utilizzare l'interfaccia web Streamlit.

### Motivazione

L'interfaccia web (`dashboard/app_v4.py`) non e' sempre apprezzata in contesti accademici. La CLI offre:
- Navigazione rapida da terminale
- Output testuale per documentazione
- Esecuzione batch senza GUI
- Compatibilita con server headless

---

## Struttura Directory

```
fl-ehds-framework/
│
├── dashboard/                         # Frontend WEB (esistente)
│   └── app_v4.py                      # Streamlit dashboard
│
├── orchestration/                     # Backend FL (esistente)
│   ├── aggregation/                   # FedAvg, FedProx, SCAFFOLD, FedNova
│   └── privacy/                       # Differential Privacy, RDP
│
└── terminal/                          # Frontend CLI (NUOVO)
    ├── __init__.py                    # Package init + versione
    ├── __main__.py                    # Entry point per python -m terminal
    ├── main.py                        # Main entry point
    ├── menu.py                        # Menu principale + navigazione frecce
    ├── colors.py                      # Schema colori ANSI (no emoji)
    ├── progress.py                    # Wrapper tqdm con colori
    ├── validators.py                  # Validazione input numerici
    ├── requirements.txt               # Dipendenze CLI specifiche
    └── screens/
        ├── __init__.py
        ├── training.py                # Training federato
        ├── algorithms.py              # Confronto algoritmi
        ├── privacy.py                 # Analisi RDP
        ├── byzantine.py               # Byzantine resilience
        ├── benchmark.py               # Benchmark suite
        └── output.py                  # Export LaTeX/CSV/grafici
```

---

## Installazione

### 1. Dipendenze CLI

```bash
cd fl-ehds-framework
pip install -r terminal/requirements.txt
```

### 2. Dipendenze CLI (requirements.txt)

```
questionary>=2.0.0      # Menu con navigazione frecce
tqdm>=4.66.0            # Progress bar colorate
rich>=13.0.0            # Tabelle formattate (opzionale)
colorama>=0.4.6         # Colori cross-platform
numpy>=1.24.0           # Calcoli numerici
scipy>=1.11.0           # Funzioni scientifiche
torch>=2.0.0            # Deep learning
matplotlib>=3.8.0       # Generazione grafici
```

---

## Utilizzo

### Avvio Interfaccia

```bash
# Metodo 1: Come modulo Python
cd fl-ehds-framework
python -m terminal

# Metodo 2: Script diretto
python terminal/main.py
```

### Menu Principale

```
================================================================================
                         FL-EHDS FRAMEWORK v4.0
              Privacy-Preserving Federated Learning for EHDS
================================================================================

MENU PRINCIPALE
---------------
> 1. Training Federato
  2. Confronto Algoritmi FL
  3. Analisi Privacy (RDP)
  4. Vertical Federated Learning
  5. Byzantine Resilience
  6. Continual Learning
  7. Multi-Task FL
  8. Hierarchical FL
  9. EHDS Compliance
  10. Benchmark Suite
  11. Configurazione Globale
  12. Esporta Risultati
  0. Esci

[Frecce: naviga | Enter: seleziona | q: esci]
```

---

## Funzionalita Implementate

### 1. Training Federato (`screens/training.py`)

**Algoritmi supportati:**
- FedAvg (baseline)
- FedProx (non-IID robust)
- SCAFFOLD (variance reduction)
- FedNova (heterogeneous steps)
- FedAdam, FedYogi, FedAdagrad (adaptive)
- Per-FedAvg, Ditto (personalization)

**Parametri configurabili:**
| Parametro | Default | Range |
|-----------|---------|-------|
| Numero client | 5 | 2-100 |
| Numero round | 30 | 1-1000 |
| Epoche locali | 3 | 1-50 |
| Batch size | 32 | 1-512 |
| Learning rate | 0.01 | 0.0001-1.0 |
| DP epsilon | 10.0 | 0.1-100 |
| DP delta | 1e-5 | 1e-10-1e-3 |

**Esempio interazione:**

```
================================================================================
                           TRAINING FEDERATO
================================================================================

Seleziona algoritmo FL:
  > FedAvg
    FedProx
    SCAFFOLD
    FedNova

[Frecce + Enter per confermare]

--------------------------------------------------------------------------------
CONFIGURAZIONE PARAMETRI (digita i valori)
--------------------------------------------------------------------------------

Numero client [5]: 10
Numero round [30]: 50
Epoche locali [3]: 5
Batch size [32]: 64
Learning rate [0.01]: 0.001

Abilitare Differential Privacy? (s/n) [n]: s
  -> Epsilon target [10.0]: 1.0
```

---

### 2. Confronto Algoritmi (`screens/algorithms.py`)

Esegue benchmark comparativi tra algoritmi con:
- Multiple run per significativita statistica (std dev)
- Configurazioni IID e Non-IID
- Varianti con Differential Privacy

**Output esempio:**

```
TABELLA COMPARATIVA
-------------------

Algoritmo                 Accuracy           F1                 AUC
-------------------------------------------------------------------------------
FedAvg (IID)              60.5% +/- 0.02     0.62 +/- 0.02      0.66 +/- 0.01
FedAvg (Non-IID)          60.9% +/- 0.02     0.61 +/- 0.01      0.66 +/- 0.01
FedProx (mu=0.1)          60.9% +/- 0.02     0.62 +/- 0.01      0.66 +/- 0.01
SCAFFOLD                  60.5% +/- 0.01     0.61 +/- 0.02      0.66 +/- 0.01
FedNova                   60.7% +/- 0.02     0.62 +/- 0.01      0.66 +/- 0.01
FedAvg + DP (e=10)        55.7% +/- 0.01     0.61 +/- 0.04      0.55 +/- 0.03
FedAvg + DP (e=1)         55.1% +/- 0.01     0.59 +/- 0.04      0.55 +/- 0.01
-------------------------------------------------------------------------------
```

---

### 3. Analisi Privacy RDP (`screens/privacy.py`)

**Funzioni disponibili:**
1. **Calcola epsilon per N round** - Stima epsilon totale
2. **Calcola round massimi** - Per target epsilon dato
3. **Calcola noise richiesto** - Per target epsilon e round
4. **Confronto RDP vs Semplice** - Mostra miglioramento
5. **Privacy-Utility Tradeoff** - Analisi impatto su accuracy

**Esempio output:**

```
CONFRONTO METODI DI COMPOSIZIONE
--------------------------------

Round      RDP Epsilon     Simple Epsilon      Improvement
------------------------------------------------------------
10         0.4521          2.1340              4.7x
30         0.9834          6.4020              6.5x
50         1.4012          10.6700             7.6x
100        2.3456          21.3400             9.1x
200        3.8912          42.6800             11.0x
------------------------------------------------------------

Nota: RDP diventa piu vantaggioso con piu round
```

---

### 4. Byzantine Resilience (`screens/byzantine.py`)

**Attacchi disponibili:**
- Label Flip
- Gaussian Noise
- Scaling Attack
- Sign Flip

**Difese disponibili:**
- Krum
- Multi-Krum
- Trimmed Mean
- Median
- Bulyan
- FLTrust
- FLAME

---

### 5. Benchmark Suite (`screens/benchmark.py`)

**Tipi di benchmark:**
1. **Completo** - Tutti algoritmi, IID/Non-IID, DP
2. **Scalabilita** - Variazione numero client (5, 10, 20, 50)
3. **Privacy** - Variazione epsilon (0.1, 0.5, 1.0, 5.0, 10.0)
4. **Rapido** - Singolo seed per test veloci

---

### 6. Export Risultati (`screens/output.py`)

**Formati supportati:**
- **LaTeX** - Tabelle per paper accademici
- **CSV** - Per analisi esterne
- **JSON** - Dati strutturati completi
- **PNG/PDF** - Grafici convergenza

**Esempio tabella LaTeX generata:**

```latex
\begin{table}[htbp]
\centering
\caption{Experimental Results}
\label{tab:results}
\small
\begin{tabular}{lccc}
\toprule
\textbf{Configuration} & \textbf{Accuracy} & \textbf{F1} & \textbf{AUC} \\
\midrule
FedAvg (IID) & 60.5\%$\pm$0.02 & 0.62$\pm$0.02 & 0.66$\pm$0.01 \\
FedAvg (Non-IID) & 60.9\%$\pm$0.02 & 0.61$\pm$0.01 & 0.66$\pm$0.01 \\
FedProx ($\mu$=0.1) & 60.9\%$\pm$0.02 & 0.62$\pm$0.01 & 0.66$\pm$0.01 \\
\bottomrule
\end{tabular}
\end{table}
```

---

## Schema Colori

La CLI utilizza colori ANSI senza emoji per compatibilita:

| Colore | Utilizzo |
|--------|----------|
| VERDE | Successo, completato, metriche positive |
| GIALLO | Warning, in progress, attenzione |
| ROSSO | Errore, fallimento, metriche negative |
| CIANO | Informazioni, titoli, sezioni |
| BIANCO | Testo normale |

---

## Progress Bar

Training con visualizzazione progresso tqdm:

```
Round di Training:
[################------------------------------------]  32% | 16/50 | 02:34<05:12

  Round 16/50:
  |-- Client Training: [##########] 100% | 10/10 client
  |-- Aggregazione:    completata
  |-- Privacy budget:  epsilon=0.32/1.0 (32% consumato)
  |-- Metriche round:
      Loss: 0.5623  Acc: 58.2%  F1: 0.57  AUC: 0.62
```

---

## Architettura Tecnica

### Principio: Import Condiviso

La CLI **non duplica** il codice backend. Importa direttamente:

```python
# terminal/screens/training.py

# Import dal backend esistente (NESSUNA COPIA)
from dashboard.app_v4 import FLSimulatorV4
from orchestration.privacy.differential_privacy import PrivacyAccountant

# Solo la presentazione e' diversa
simulator = FLSimulatorV4(
    num_clients=config['num_clients'],
    num_rounds=config['num_rounds'],
    algorithm=config['algorithm'],
    # ...
)
results = simulator.run()
```

### Fallback Automatici

```python
# Se questionary non installato -> menu numerici
try:
    import questionary
    HAS_QUESTIONARY = True
except ImportError:
    HAS_QUESTIONARY = False

# Se tqdm non installato -> progress semplice
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
```

---

## Statistiche Implementazione

| Aspetto | Valore |
|---------|--------|
| Righe di codice CLI | ~800 |
| Righe backend riusate | ~3500+ |
| Percentuale riuso | ~85% |
| File nuovi | 12 |
| Dipendenze aggiuntive | 4 (questionary, tqdm, rich, colorama) |

---

## Confronto Web vs Terminal

| Aspetto | Web (Streamlit) | Terminal (CLI) |
|---------|-----------------|----------------|
| Interattivita | Click/Slider | Frecce/Input |
| Grafici | Plotly interattivi | Matplotlib statici |
| Output | Browser | Stdout + file |
| Dipendenze | streamlit, plotly | questionary, tqdm |
| Server richiesto | Si | No |
| Batch execution | Limitata | Nativa |
| Documentazione output | Screenshot | Copy-paste testo |

---

## Comandi Rapidi

```bash
# Avvio standard
python -m terminal

# Training rapido FedAvg
python -m terminal  # poi 1 -> 2

# Benchmark completo
python -m terminal  # poi 10 -> 1

# Analisi privacy
python -m terminal  # poi 3 -> 1
```

---

## Troubleshooting

### Import Error: No module named 'streamlit'

Il backend `app_v4.py` richiede streamlit. Installare con:
```bash
pip install streamlit
```

### Import Error: No module named 'structlog'

Il modulo privacy richiede structlog. Installare con:
```bash
pip install structlog
```

### Menu non mostra frecce

Questionary non installato. Installare con:
```bash
pip install questionary
```
Oppure usare il fallback numerico (funziona automaticamente).

### Progress bar non colorata

tqdm non installato. Installare con:
```bash
pip install tqdm
```

---

## Sviluppi Futuri

- [ ] Vertical FL screen completo
- [ ] Continual Learning screen
- [ ] Multi-Task FL screen
- [ ] Hierarchical FL screen
- [ ] EHDS Compliance checker
- [ ] Configuration persistence (YAML)
- [ ] Batch mode da command line args

---

*Ultimo aggiornamento: 5 Febbraio 2026*

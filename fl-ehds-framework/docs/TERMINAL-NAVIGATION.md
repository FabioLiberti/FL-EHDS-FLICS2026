# FL-EHDS Terminal - Navigation Tree

## Quick Reference

```
python -m terminal
```

---

## Complete Navigation Tree

```
================================================================================
                         FL-EHDS TERMINAL APPLICATION
                              NAVIGATION MAP
================================================================================

MAIN MENU
|
+-- [1] Training Federato ---------------------------------> TrainingScreen
|   |
|   +-- [1] Configura parametri
|   |       |-- Algoritmo (9 opzioni)
|   |       |-- Numero client, round, epoche locali
|   |       |-- Batch size, learning rate
|   |       |-- Distribuzione dati (IID/Non-IID)
|   |       |-- Differential Privacy (on/off)
|   |       +-- Random seed
|   |
|   +-- [2] Avvia training
|   |       |-- Generazione dataset
|   |       |-- Training PyTorch reale
|   |       +-- Progress per client/round
|   |
|   +-- [3] Visualizza risultati
|   |       |-- Configurazione utilizzata
|   |       |-- Distribuzione dati per client
|   |       +-- Storico metriche
|   |
|   +-- [4] Genera grafici convergenza
|   |       |-- Loss vs Round (PNG)
|   |       |-- Accuracy vs Round (PNG)
|   |       +-- ASCII preview (opzionale)
|   |
|   +-- [5] Esporta risultati
|   |       +-- [1] JSON
|   |       +-- [2] CSV
|   |       +-- [3] Entrambi
|   |       +-- [0] Annulla
|   |
|   +-- [0] Torna al menu principale
|
+-- [2] Confronto Algoritmi FL ----------------------------> AlgorithmsScreen
|   |
|   +-- [1] Configura confronto
|   |       |-- Selezione algoritmi (checkbox)
|   |       |-- Parametri training
|   |       |-- Numero seed (per std dev)
|   |       +-- Opzioni DP
|   |
|   +-- [2] Esegui confronto
|   |       +-- N algoritmi x M seed runs
|   |
|   +-- [3] Visualizza risultati
|   |       +-- Tabella comparativa
|   |
|   +-- [4] Genera tabella comparativa
|   |       +-- LaTeX output
|   |
|   +-- [5] Esporta risultati
|   |       +-- JSON + LaTeX
|   |
|   +-- [0] Torna al menu principale
|
+-- [3] Confronto Guidato per Caso d'Uso -----------------> GuidedComparisonScreen
|   |
|   +-- [1] Seleziona caso d'uso sanitario
|   |       |-- Multi-Hospital (IID)
|   |       |-- Rare Disease (Non-IID)
|   |       |-- Personalized Medicine
|   |       |-- Resource-Constrained
|   |       |-- Privacy-Critical (DP)
|   |       |-- Cross-Border EHDS
|   |       +-- Fast Convergence
|   |
|   +-- [2] Visualizza configurazione consigliata
|   |       |-- Algoritmi raccomandati
|   |       |-- Parametri suggeriti
|   |       +-- Motivazione
|   |
|   +-- [3] Esegui confronto algoritmi
|   |       +-- Confronto automatico
|   |
|   +-- [4] Visualizza risultati
|   |       |-- Tabella comparativa
|   |       +-- Convergenza per algoritmo
|   |
|   +-- [5] Genera report comparativo
|   |       |-- LaTeX table
|   |       +-- Convergence plot (PNG)
|   |
|   +-- [0] Torna al menu principale
|
+-- [4] Analisi Privacy (RDP) -----------------------------> PrivacyScreen
|   |
|   +-- [1] Calcola epsilon per N round
|   |       |-- Input: rounds, sigma, q, delta
|   |       +-- Output: epsilon (RDP)
|   |
|   +-- [2] Calcola round massimi per target epsilon
|   |       |-- Input: target epsilon, sigma, q, delta
|   |       +-- Output: max rounds
|   |
|   +-- [3] Calcola noise per target epsilon
|   |       |-- Input: target epsilon, rounds, q, delta
|   |       +-- Output: noise multiplier
|   |
|   +-- [4] Confronto RDP vs Composizione Semplice
|   |       +-- Tabella comparativa bounds
|   |
|   +-- [5] Analisi Privacy-Utility Tradeoff
|   |       +-- Grafico epsilon vs accuracy
|   |
|   +-- [0] Torna al menu principale
|
+-- [5] Vertical Federated Learning -----------------------> [In Sviluppo]
|
+-- [6] Byzantine Resilience ------------------------------> ByzantineScreen
|   |
|   +-- [1] Configura test
|   |       |-- Numero client/bizantini
|   |       |-- Tipo attacco (5 opzioni)
|   |       |-- Tipo difesa (8 opzioni)
|   |       +-- Forza attacco
|   |
|   +-- [2] Esegui singolo test
|   |       +-- Training con attacco/difesa
|   |
|   +-- [3] Confronto difese
|   |       +-- Test tutte le difese
|   |
|   +-- [4] Visualizza risultati
|   |       +-- Tabella con/senza attacco
|   |
|   +-- [0] Torna al menu principale
|
+-- [7] Continual Learning --------------------------------> [In Sviluppo]
|
+-- [8] Multi-Task FL -------------------------------------> [In Sviluppo]
|
+-- [9] Hierarchical FL -----------------------------------> [In Sviluppo]
|
+-- [10] EHDS Compliance ----------------------------------> [In Sviluppo]
|
+-- [11] Benchmark Suite ----------------------------------> BenchmarkScreen
|   |
|   +-- [1] Benchmark Completo (tutti algoritmi)
|   |       |-- 9 algoritmi x IID/Non-IID
|   |       |-- + DP variants
|   |       +-- 3 seed per config
|   |
|   +-- [2] Benchmark Scalabilita (client)
|   |       +-- Test con 5, 10, 20, 50 client
|   |
|   +-- [3] Benchmark Privacy (epsilon)
|   |       +-- Test con e=0.1, 0.5, 1, 5, 10
|   |
|   +-- [4] Benchmark Rapido (singolo seed)
|   |       +-- Come completo ma 1 seed
|   |
|   +-- [5] Visualizza risultati salvati
|   |       +-- Tabella in memoria
|   |
|   +-- [6] Genera tabelle per paper
|   |       +-- LaTeX output
|   |
|   +-- [0] Torna al menu principale
|
+-- [12] Configurazione Globale ---------------------------> [In Sviluppo]
|
+-- [13] Esporta Risultati --------------------------------> OutputScreen
|   |
|   +-- [1] Genera tabella LaTeX
|   |       +-- (richiede risultati)
|   |
|   +-- [2] Esporta CSV
|   |       +-- (richiede risultati)
|   |
|   +-- [3] Genera grafici
|   |       +-- (richiede risultati)
|   |
|   +-- [4] Esporta JSON
|   |       +-- (richiede risultati)
|   |
|   +-- [5] Visualizza file generati
|   |       +-- Lista file in results/
|   |
|   +-- [0] Torna al menu principale
|
+-- [0] Esci

================================================================================
```

---

## Percorsi Comuni

### Workflow 1: Training Singolo Algoritmo
```
Main [1] -> Training [1] Configure -> [2] Run -> [4] Plot -> [5] Export
```

### Workflow 2: Confronto Algoritmi
```
Main [2] -> Algorithms [1] Configure -> [2] Run -> [4] Generate Table
```

### Workflow 3: Confronto Guidato per Caso Sanitario
```
Main [3] -> Guided [1] Select Use Case -> [3] Run -> [5] Generate Report
```

### Workflow 4: Analisi Privacy
```
Main [4] -> Privacy [1] Compute Epsilon -> [4] Compare Composition
```

### Workflow 5: Test Byzantine
```
Main [6] -> Byzantine [1] Configure -> [2] Run Test -> [3] Compare Defenses
```

### Workflow 6: Benchmark Completo per Paper
```
Main [11] -> Benchmark [1] Full Benchmark -> [6] Generate Paper Tables
```

---

## Algoritmi Disponibili (9)

| # | Algoritmo | Categoria | Parametri Specifici |
|---|-----------|-----------|---------------------|
| 1 | FedAvg | Baseline | - |
| 2 | FedProx | Regularization | mu |
| 3 | SCAFFOLD | Variance Reduction | control variates |
| 4 | FedNova | Normalization | - |
| 5 | FedAdam | Adaptive Server | server_lr, beta1, beta2, tau |
| 6 | FedYogi | Adaptive Server | server_lr, beta1, beta2, tau |
| 7 | FedAdagrad | Adaptive Server | server_lr, tau |
| 8 | Per-FedAvg | Personalization | - |
| 9 | Ditto | Personalization | mu (lambda) |

---

## Casi d'Uso Sanitari (7)

| # | Caso d'Uso | Algoritmi Consigliati |
|---|------------|----------------------|
| 1 | Multi-Hospital (IID) | FedAvg, FedAdam, FedYogi |
| 2 | Rare Disease (Non-IID) | SCAFFOLD, FedProx, FedNova |
| 3 | Personalized Medicine | Ditto, Per-FedAvg, FedProx |
| 4 | Resource-Constrained | FedNova, FedAvg, FedProx |
| 5 | Privacy-Critical (DP) | FedAvg, FedProx |
| 6 | Cross-Border EHDS | SCAFFOLD, FedProx, Ditto |
| 7 | Fast Convergence | FedAdam, FedYogi, SCAFFOLD |

---

## Attacchi Byzantine (5)

| # | Attacco | Descrizione |
|---|---------|-------------|
| 1 | None | Baseline (no attack) |
| 2 | Label Flip | Inverte le label |
| 3 | Gaussian Noise | Aggiunge rumore ai gradienti |
| 4 | Scaling Attack | Scala i gradienti |
| 5 | Sign Flip | Inverte il segno dei gradienti |

## Difese Byzantine (8)

| # | Difesa | Descrizione |
|---|--------|-------------|
| 1 | None | Nessuna difesa |
| 2 | Krum | Seleziona update piu vicino |
| 3 | Multi-Krum | Seleziona k update piu vicini |
| 4 | Trimmed Mean | Media dopo trimming |
| 5 | Median | Mediana coordinate-wise |
| 6 | Bulyan | Krum + trimmed mean |
| 7 | FLTrust | Trust-based aggregation |
| 8 | FLAME | Clustering-based defense |

---

## Output Directory

```
fl-ehds-framework/results/
├── training_convergence_*.png      # Grafici training
├── comparison_convergence_*.png    # Grafici confronto
├── table_results_*.tex             # Tabelle LaTeX
├── training_results_*.json         # Risultati JSON
├── training_history_*.csv          # Storico CSV
└── comparison_results_*.json       # Confronti JSON
```

---

## Shortcuts

| Tasto | Azione |
|-------|--------|
| Frecce | Naviga menu |
| Enter | Seleziona |
| Ctrl+C | Esci |
| 0 | Torna indietro |
| q | Esci (fallback mode) |

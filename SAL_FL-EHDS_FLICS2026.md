# SAL - Stato Avanzamento Lavori
## Progetto: FL-EHDS Paper per FLICS 2026

**Data**: 1 Febbraio 2026  
**Autore**: Fabio Liberti  
**Affiliazione**: Universitas Mercatorum, Rome, Italy

---

## 📋 OVERVIEW PROGETTO

### Obiettivo
Sviluppo di un paper scientifico su **Federated Learning per European Health Data Space (EHDS)** da sottomettere alla conferenza **FLICS 2026** (Federated Learning in Complex Systems).

### Documenti di Riferimento
1. `FL_EHDS_Paper_Structure.md` - Struttura dettagliata del paper (~6,500 parole target)
2. `SLR_EHDS_Extended_Abstract.pdf` - Abstract precedente della SLR
3. `SLR_EHDS_Complete_v3_0.pdf` - Paper SLR completo precedente (52 documenti, 5 assi tematici)

### Titolo Paper
> **FL-EHDS: A Privacy-Preserving Federated Learning Framework for the European Health Data Space**

---

## ✅ LAVORO COMPLETATO

### 1. Figura TikZ (Architettura FL-EHDS)
**File**: `figures/fig2-fl-ehds-architecture.tex`

- Diagramma a 3 livelli in scala di grigi (B/W)
- Layer 1: Governance (HDABs, Data Permits, Opt-out Registry)
- Layer 2: FL Orchestration (Aggregation, Privacy, Compliance)
- Layer 3: Data Holders (Training Engines, FHIR Preprocessing)
- Frecce per flusso gradienti (solide ↑) e distribuzione modello (tratteggiate ↓)
- Legenda inclusa
- Formato: TikZ nativo LaTeX (no SVG)

### 2. Main.tex Versione Iniziale
**File**: `main.tex` (in Overleaf)

- Formato IEEE Conference
- 6 sezioni base
- 12 riferimenti bibliografici
- Figura TikZ integrata via `\input{figures/fig2-fl-ehds-architecture}`

### 3. Main.tex Versione Migliorata
**File**: `main_improved.tex`

#### Miglioramenti applicati:
| Aspetto | Prima | Dopo |
|---------|-------|------|
| References | 12 | **22** |
| Sezioni | 6 | **7** (+ Discussion) |
| Abstract | ~150 parole | ~250 parole |
| Related Work | ❌ Assente | ✅ Aggiunta |
| Conclusions | ~4 righe | ~15 righe |
| ORCID | ❌ | ✅ Placeholder |
| Acknowledgments | Incompleto | ✅ Con tutor |
| Methodology | Implicita | ✅ Esplicita (§4.1) |

#### Bibliografia Aggiornata (22 ref, 2024-2026):

**EHDS & Policy (11):**
1. EU Commission 2025 - Regolamento EHDS
2. Staunton et al. 2024 - EJHG - Aspetti etici
3. Quinn et al. 2024 - CLSR - GDPR vs EHDS
4. TEHDAS 2024 - EJPH - Member State readiness
5. Fröhlich et al. 2025 - JMIR - Reality check (23% deployment)
6. van Drumpt et al. 2025 - Front Digit Health - PETs
7. Hussein et al. 2025 - JMIR - Interoperability framework
8. Forster et al. 2025 - EJPH - User journeys
9. Svingel et al. 2025 - EJPH - HDAB recommendations
10. Christiansen et al. 2025 - EJPH - HealthData@EU Pilot
11. Ganna et al. 2024 - Nat Med - Research boost

**Federated Learning (11):**
12. McMahan et al. 2017 - AISTATS - FedAvg
13. Li et al. 2020 - MLSys - FedProx
14. Kairouz et al. 2021 - FTML - Open problems
15. Rieke et al. 2020 - npj Digital Medicine - FL digital health
16. Bonawitz et al. 2019 - MLSys - FL at scale
17. Teo et al. 2024 - Cell Rep Med - Systematic review (612 articles)
18. Peng et al. 2024 - CMPB - Systematic review
19. Zhu et al. 2019 - NeurIPS - Gradient inversion attacks
20. Shokri et al. 2017 - IEEE S&P - Membership inference
21. Dwork & Roth 2014 - FTCS - Differential privacy
22. Abadi et al. 2016 - CCS - DP deep learning

### 4. Setup Repository/Workflow

**Struttura Overleaf:**
```
FL-EHDS-FLICS2026/
├── main.tex                              ← Paper principale
└── figures/
    └── fig2-fl-ehds-architecture.tex     ← Figura TikZ
```

**Workflow:**
- VS Code ↔ GitHub ↔ Overleaf (sincronizzazione bidirezionale)
- Git pull per aggiornare VS Code dopo push da Overleaf
- GitVersion per versioning semantico

---

## 📊 DATI CHIAVE DEL PAPER

### Statistiche Evidence Synthesis
| Metrica | Valore | Fonte |
|---------|--------|-------|
| FL production deployment | **23%** | Fröhlich et al. 2025 |
| Hardware heterogeneity barrier | **78%** | Fröhlich et al. 2025 |
| Non-IID data challenges | **67%** | Fröhlich et al. 2025 |
| FHIR compliance EU | **34%** | Hussein et al. 2025 |
| Nordic advantage | **2-3 anni** | TEHDAS 2024 |
| Access timeline range | 3 settimane - 12+ mesi | Forster et al. 2025 |

### Key Finding
> **Legal uncertainties—not technical barriers—constitute the critical blocker** for FL adoption in EHDS contexts.

### Timeline EHDS
| Data | Milestone | Rilevanza FL |
|------|-----------|--------------|
| Mar 2025 | Entry into force | Framework attivo |
| Mar 2027 | Delegated acts | **Clarificazione gradient status** |
| Mar 2029 | Secondary use | FL deve essere operativo |
| Mar 2031 | Genetic, imaging | Requisiti estesi |

---

## 🎯 PROSSIMI PASSI

### Immediati (pre-submission)
- [ ] Inserire ORCID reale nel main.tex (riga 30)
- [ ] Sostituire main.tex in Overleaf con versione migliorata
- [ ] Compilare PDF e verificare:
  - [ ] Figura TikZ renderizza correttamente
  - [ ] Lunghezza ≤ 8 pagine
  - [ ] Tutte le citazioni presenti
- [ ] Rileggere per typos e grammatica
- [ ] Push finale su GitHub

### Submission FLICS 2026
- [ ] Verificare deadline esatta conferenza
- [ ] Preparare cover letter (se richiesta)
- [ ] Registrarsi su sistema submission
- [ ] Upload PDF finale

### Post-submission (future work)
- [ ] Validazione empirica tramite HealthData@EU pilots
- [ ] Citizen attitude studies
- [ ] Economic sustainability modeling per HDABs
- [ ] Longitudinal implementation tracking

---

## 📁 FILE CREATI IN QUESTA SESSIONE

| File | Percorso | Descrizione |
|------|----------|-------------|
| `fig2-fl-ehds-architecture.tex` | `figures/` | Figura TikZ architettura |
| `main.tex` | root | Paper versione base |
| `main_improved.tex` | outputs | Paper versione migliorata (22 ref) |
| `ISTRUZIONI-OVERLEAF.md` | outputs | Guida setup Overleaf |
| `SAL_FL-EHDS_FLICS2026.md` | outputs | Questo file |

---

## 🔧 CONFIGURAZIONE TECNICA

### Pacchetti LaTeX Richiesti
```latex
\usepackage{cite}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{graphicx}
\usepackage{textcomp}
\usepackage{xcolor}
\usepackage{booktabs}
\usepackage{hyperref}
\usepackage{tikz}
\usetikzlibrary{shapes.geometric, arrows.meta, positioning, fit, backgrounds, calc}
```

### Compilatore
- **pdfLaTeX** (default Overleaf)
- Alternativa: XeLaTeX se necessario per font

---

## 📝 NOTE PER RIPRESA CONVERSAZIONE

### Contesto
- Paper per conferenza FLICS 2026 (Federated Learning in Complex Systems)
- Basato su SLR precedente (52 documenti, 5 assi tematici)
- Focus specifico su FL come enabling technology per EHDS
- Target: ~6,500 parole, 8 pagine max formato IEEE

### Decisioni Chiave Prese
1. **TikZ invece di SVG** per figura (nativo LaTeX, più controllabile)
2. **Grayscale** per compatibilità stampa IEEE
3. **22 riferimenti** (mix EHDS policy + FL technical, 2024-2026)
4. **Key message**: Legal uncertainties > Technical barriers
5. **Framework a 3 livelli**: Governance → Orchestration → Data Holders

### Punti di Attenzione
- ORCID da inserire (attualmente placeholder)
- Tutor da verificare negli Acknowledgments (Prof. Sadi Alawadi)
- Figure reference: `\ref{fig:architecture}` deve matchare label nel TikZ

### Comandi Git Utili
```bash
# Dopo modifiche in Overleaf
git pull

# Se conflitti
git stash
git pull
git stash pop

# Push da VS Code
git add .
git commit -m "descrizione"
git push
```

---

## 📞 CONTATTI

**Autore**: Fabio Liberti  
**Email**: fabio.liberti@studenti.unimercatorum.it  
**ORCID**: 0000-0003-3019-5411  
**Tutor**: Prof. Sadi Alawadi

---

*Ultimo aggiornamento: 1 Febbraio 2026*

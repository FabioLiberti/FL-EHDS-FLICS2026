# Revisione Finale Approfondita — FL-EHDS
## Paper Principale + Supplementary Material
### Per: FLICS 2026 Conference Submission

---

# SEZIONE A: ERRORI CRITICI (da correggere obbligatoriamente)

---

## A1. Errori Grammaticali / Ortografici / Typo

### Paper Principale

| # | Pag. | Posizione | Errore | Correzione | Note |
|---|------|-----------|--------|------------|------|
| 1 | 1 | Abstract, riga 5 | "17 aggregation algorithms including 2024–2025 advances" | "17 aggregation algorithms including 2024–2025 advances**,** differential privacy" | Manca virgola prima di "differential privacy" per chiarezza nella lista |
| 2 | 1 | Sect. I, par. 2 | "Fröhlich et al. [5] report that only 23%" | Verificare: il cognome è "Fröhlich" con umlaut. Nel testo a pag. 2 e 5 appare "Frohlich" senza umlaut ma con "¨" separato come artefatto PDF | Uniformare: usare "Fr\"ohlich" in LaTeX per rendering corretto |
| 3 | 2 | Sect. II-A, riga 3 | "from 3 weeks (Finland) to over 12 months (France)" | OK grammaticalmente, ma la dash è un en-dash: verificare che in LaTeX sia `--` | Controllo tipografico |
| 4 | 3 | Sect. III-B, Privacy Protection | "Rényi DP (RDP) [26]" — il nome "Renyi" appare senza accento a pag. 3 riga 2 del paragrafo ("Renyi DP") | Uniformare a "R\'enyi" ovunque | Appare corretto in alcuni punti, inconsistente in altri |
| 5 | 4 | Algorithm 1 | Formula aggregazione: "θ^(t) ← θ^(t−1) + 1/(Σnh) Σ nh · Δ_h^(t)" | La notazione `P1` e `P` nel PDF sembra un artefatto di rendering. Verificare il LaTeX: dovrebbe essere `\frac{1}{\sum_h n_h} \sum_{h \in \mathcal{H}_t} n_h \cdot \Delta_h^{(t)}` | Possibile problema di rendering del PDF |
| 6 | 5 | Sect. IV-A | "PTB-XL ECG [50], 21,799 European-origin records from 52 German recording sites" | Coerenza: nell'abstract e Table VI si dice "52 EU sites" ma qui "52 German recording sites". PTB-XL è tedesco (Physikalisch-Technische Bundesanstalt, Berlin) | Suggerimento: uniformare a "52 European recording sites" o chiarire che è tedesco. La designazione "EU sites" in Table VI potrebbe fuorviare |
| 7 | 5 | Sect. IV-A | "Skin Cancer (3,297, binary)" | Table V dice "3,297" ma Supplementary pag. 11 dice "3,297 dermoscopy images" — OK coerente |  |
| 8 | 5 | Table V, nota | "default K=5" per PTB-XL — ma il testo (Sect. IV-A) menziona "site-based" partitioning con 52 siti raggruppati in K cluster | Chiarire meglio la relazione tra 52 siti e K=5 nella nota della tabella |
| 9 | 6 | Table VI, nota F1 | "NORM-class one-vs-rest for PTB-XL (5-class)—the near-100% values indicate all algorithms classify the majority class correctly" | Ambiguo: se F1 è calcolato per NORM-class (una sola classe), non è F1 macro. Specificare più chiaramente | Suggerire: "F1: NORM-class one-vs-rest recall for PTB-XL" o usare F1-macro |
| 10 | 7 | Sect. V-A, ultimo paragrafo | "Article 50(4) mandates that SPEs provide 'a high level of security'" | Verificare la citazione esatta dell'articolo 50(4) del Regolamento 2025/327 | Controllo legale |
| 11 | 8 | Sect. VI | "Coordinated action across EU policymakers, national authorities, and healthcare organizations is essential" | Frase finale forte ma potrebbe beneficiare di un verbo più specifico: "...is essential **to ensure**..." | Stilistico, opzionale |

### Supplementary Material

| # | Pag. | Posizione | Errore | Correzione | Note |
|---|------|-----------|--------|------------|------|
| 12 | 2 | Fig. S-1, PRISMA box | "Not peer-reviewd" | **"Not peer-reviewed"** | Typo critico in un diagramma PRISMA — visibile ai revisori |
| 13 | 3 | Alg. S1, commento | "// Opt-out filtering (Article 71)" | OK | |
| 14 | 4 | Sect. II-F, ultimo par. | "Algorithm S6 (Opt-Out Filtering)" con punto finale dopo la parentesi | Verificare punteggiatura consistente | |
| 15 | 7 | Table S-I, FHIR-Native section | "Synthea FHIR R4" ha Samples = "1,180" e Feat = "14" | Verificare che 1,180 sia corretto per Synthea (il dataset standard Synthea è configurabile) | |
| 16 | 7 | Table S-I, nota | "config. = configurable sample count" | Potrebbe essere più chiaro: "config. = sample count depends on generation parameters" | |
| 17 | 8 | Table S-II | "MS = Member States" nella nota | OK ma ripetuto anche nel testo. Definire MS una sola volta | |
| 18 | 9 | Sect. VI-F, Alg. S11 | "pFedMe Local Update" — ma nel testo prima si dice "Algorithm S11 shows pFedMe [34]" e la referenza [34] è "Fallah, Mokhtari, and Ozdaglar" che è Per-FedAvg, NON pFedMe | **ERRORE CRITICO**: Ref [34] in supplementary è Per-FedAvg (MAML-based). pFedMe è Dinh et al. (NeurIPS 2020). La ref [34] nel supplementary è diversa dal main paper dove [34] = Per-FedAvg | Verificare: nel supplementary references, [34] = "T. Dinh, N. Tran, and J. Nguyen, 'Personalized federated learning with Moreau envelopes,' NeurIPS 2020" — questo È pFedMe. Ma nel MAIN PAPER, ref [34] = "A. Fallah, A. Mokhtari, and A. Ozdaglar, 'Personalized federated learning with theoretical guarantees: A model-agnostic meta-learning approach,' NeurIPS 2020" = Per-FedAvg. **Le reference list sono DIVERSE tra paper e supplementary!** |
| 19 | 12 | Sect. IX-A | "Chest X-ray [49]: 5,860 pediatric radiographs" | Main paper Table V dice "5,856". Supplementary dice "5,860" | **Inconsistenza numeri: 5,856 vs 5,860** — uniformare |
| 20 | 15 | Table S-IV | Titolo: "FL-EHDS Component Readiness for EHDS Production Deployment" | OK | |
| 21 | 18 | Table S-VI | "SCAFFOLD 66.3±5.1" su Heart Disease e "11.2±0.0" su Diabetes | La std di 0.0 per SCAFFOLD su Diabetes è sospetta — potrebbe indicare che l'algoritmo converge sempre allo stesso punto di fallimento | Non è un errore, ma vale la pena commentare |

---

## A2. Inconsistenze Numeriche e Dati

| # | Tipo | Dettaglio | Dove |
|---|------|-----------|------|
| 1 | **Chest X-ray samples** | Main paper Tab. V: 5,856 / Suppl. Sect. IX-A: 5,860 | Correggere a un valore unico (il dataset Kermany ha 5,856 nel paper originale) |
| 2 | **PTB-XL "EU sites" vs "German sites"** | Main Tab. VI header: "52 EU sites" / Main Sect. IV-A: "52 German recording sites" / Suppl. Tab. S-I: "PTB-XL: European-origin dataset (PTB, Berlin, Germany)" | PTB-XL è tedesco. "52 EU sites" è tecnicamente corretto (Germania ∈ EU) ma potenzialmente fuorviante. Suggerimento: "52 European (German) recording sites" |
| 3 | **Breast Cancer confusion matrix** | Suppl. Tab. S-XX: FedAvg Acc = 57.3±9.5% / Main Tab. VI: FedAvg Acc = 52.3±17.9% | La discrepanza è probabilmente dovuta al diverso numero di seeds (10 vs 5) e diverse configurazioni. Ma va spiegata |
| 4 | **HPFL PTB-XL accuracy** | Main Tab. VI: 92.5±0.3 / Suppl. Tab. S-XIV (opt-out 0%): 92.8±0.4 | Lieve discrepanza (92.5 vs 92.8) — probabilmente configurazioni diverse. Specificare |
| 5 | **Experiment count** | Abstract: "1,740+" / Suppl. XV intro: somma = 105+560+385+180+180+105+225 = 1,740 | OK, ma le imaging experiments (12+2+6=20 circa) porterebbero oltre 1,760. Chiarire se "1,740+" include solo tabular |
| 6 | **Reference [34]** | Main paper [34] = Fallah et al. (Per-FedAvg) / Supplementary [34] = Dinh et al. (pFedMe) | **CRITICO**: Le reference list DEVONO essere identiche o va chiarito che il supplementary ha una propria numerazione |
| 7 | **Heart Disease Table VII vs S-VI** | Tab. VII: Centralized = 81.7±2.9% / Tab. S-VI: FedAvg = 62.5±8.0%, Ditto = 75.1±2.0% | Coerente |
| 8 | **DP Table VIII vs S-X** | Tab. VIII (main): FedAvg ε=1 → 52.3 / Tab. S-X: FedAvg ε=1 → 52.3±13.3 | Coerente |

---

## A3. Problemi di Referenze e Citazioni

| # | Problema | Dettaglio |
|---|----------|-----------|
| 1 | **Ref [34] discrepanza** (vedi sopra) | La reference [34] cambia significato tra main e supplementary. Questo è un errore grave se la conferenza richiede reference list condivise |
| 2 | **Self-citation del repository** | Nota 1 (pag. 1): URL GitHub. Verificare che il repo sia effettivamente pubblico e accessibile al momento della submission |
| 3 | **Ref [18] — "2026"** | Chavero-Diez et al. è datato 2026 — confermare che sia effettivamente pubblicato o in press |
| 4 | **Ref [44] — Flower** | Citato come "arXiv:2007.14390, 2023" — l'anno potrebbe essere stato aggiornato. Verificare la versione più recente |
| 5 | **Cross-references figure** | Main paper Fig. 1 menziona "(a)" e "(b)" — verificare che entrambi i pannelli siano chiaramente etichettati nell'immagine |
| 6 | **"see Supplementary Material"** | Verificare che ogni rimando al supplementary sia specifico (Table S-X, Fig. S-19, etc.) e non generico |

---

# SEZIONE B: PROBLEMI DI PRESENTAZIONE E STILE

---

## B1. Consistenza Terminologica

| Termine | Varianti trovate | Raccomandazione |
|---------|-----------------|-----------------|
| Percentage points | "pp", "12.6pp", "percentage points" | Uniformare a "pp" con definizione alla prima occorrenza: "percentage points (pp)" |
| Non-IID | "non-IID", "Non-IID" | Uniformare capitalizzazione: "non-IID" nel testo, "Non-IID" a inizio frase |
| Member State(s) | "Member States", "Member State", "MS" | Coerente nel main paper; nel suppl. abbreviato a "MS" in Table S-II |
| Health Data Access Bodies | "HDABs", "HDAB" | OK — plurale/singolare appropriato |
| Federated Learning | "FL" definito nella prima occorrenza — OK | |
| Differential Privacy | "DP" definito — OK ma a volte scritto per esteso quando non necessario | |
| "privacy budget" vs "privacy parameter" | Usati in modo intercambiabile (Sect. V-A) | Preferire "privacy budget" consistentemente |
| ε-budget vs ε budget | Con e senza trattino | Uniformare a "ε-budget" |
| "data holder" | Talvolta capitalizzato ("Data Holders"), talvolta no | Uniformare: capitalizzato solo nei titoli di sezione/layer |

## B2. Stile e Chiarezza

| # | Sezione | Osservazione | Suggerimento |
|---|---------|--------------|--------------|
| 1 | Abstract | "yields two key findings" — ma poi ne elenca effettivamente di più nella sezione risultati | Considerare: "yields several key findings, primarily:" |
| 2 | Sect. I, contributo 3 | "non-obvious result" — linguaggio un po' informale per un paper accademico | Considerare: "counterintuitive result" o "unexpected result" |
| 3 | Sect. IV-E, punto 6 | "Privacy is essentially free at ε=10" — linguaggio informale | Considerare: "Privacy imposes negligible utility cost at ε=10" |
| 4 | Sect. V-A | "a German hospital (εmax=1.0, strict BDSG interpretation) federating with an Italian hospital (εmax=5.0, Garante guidance)" | Eccellente esempio concreto — mantenere |
| 5 | Sect. V-D | "Practical deployment scenario" — molto utile per i revisori | Mantenere |
| 6 | Table VI nota | "BC std reflects single-class collapse (see text)" — il "see text" è vago | Specificare: "(see Section IV-E, point 6 and Supplementary Table S-XX)" |
| 7 | Sect. IV-E, punto 8 | "Our limited imaging evaluation suggests personalization offers no advantage" | Buona cautela nel linguaggio |

---

# SEZIONE C: PROBLEMI STRUTTURALI

---

## C1. Organizzazione del Paper

| # | Osservazione | Raccomandazione |
|---|-------------|-----------------|
| 1 | La sezione IV-E (Key Findings) ha 8 punti — potrebbe essere pesante per un reviewer | Considerare di raggruppare: (1-2) Algorithm choice, (3) European validation, (4) Heterogeneity, (5) Architecture choice, (6-7) Privacy & opt-out, (8) Modality |
| 2 | Table I (Framework Comparison) è fondamentale ma compatta | Aggiungere "Lines of Code" o "Maturity Level" per rafforzare il confronto |
| 3 | La sezione V (Discussion) copre sia risultati sperimentali che implicazioni legali | OK per la struttura corrente — la separazione è chiara |
| 4 | Limitazioni (V-E) sono ben bilanciate tra onestà e forza del contributo | Mantenere |

## C2. Figure e Tabelle

| # | Elemento | Problema | Suggerimento |
|---|----------|----------|--------------|
| 1 | Fig. 1 | La risoluzione sembra adeguata ma i testi piccoli nel pannello (b) potrebbero essere illeggibili in stampa | Verificare con stampa a dimensione colonna IEEE |
| 2 | Fig. 2 | Solo 4 algoritmi (FedAvg, FedProx, SCAFFOLD, Ditto) — perché non HPFL? | Aggiungere HPFL o spiegare l'assenza |
| 3 | Fig. 3 | Asse x non intuitivo ("higher = more IID") | Il label annotato aiuta, ma considerare invertire l'asse |
| 4 | Table VI | Le colonne "Jain" sono poco differenziate per PTB-XL (tutte 0.999) | OK — è un risultato, non un problema di presentazione |
| 5 | Table VIII | Solo 3 algoritmi × 3 ε + No-DP = compatta ed efficace | OK |

---

# SEZIONE D: SUPPLEMENTARY MATERIAL — PROBLEMI SPECIFICI

---

## D1. Errori nel Supplementary

| # | Pag. | Problema | Correzione |
|---|------|----------|------------|
| 1 | 2 | PRISMA: "Not peer-reviewd" | → "Not peer-reviewed" |
| 2 | 5 | Sect. III note: "Figures S-2–S-9 were generated from an extended 50-round, 5-client training run" — ma le sezioni C–H (pag. 5) hanno solo titoli senza contenuto | Aggiungere "See repository for complete figures" o includere i contenuti |
| 3 | 7 | Table S-I: "OMOP-CDM Harmonized" ha Feat. = "~36" | Specificare il numero esatto o rimuovere la tilde |
| 4 | 9 | Eq. EWC: F_i manca definizione formale | Aggiungere: "where F_i = E[∇²log p(D|θ*)]_i is the diagonal Fisher Information Matrix" |
| 5 | 11 | Fig. S-13 | Il diagramma è dettagliato ma potrebbe beneficiare di frecce direzionali più chiare |
| 6 | 12 | Sect. IX-C | "Chest X-ray experiments use 4 algorithms × 3 seeds = 12 experiments per dataset" — il "per dataset" è fuorviante perché Chest X-ray È il dataset | → "= 12 experiments" (rimuovere "per dataset") |
| 7 | 13 | Table S-III, nota | "†FedLESAM produces results identical to FedAvg" | OK — ben documentato |
| 8 | 18 | Table S-VI | Buona presentazione dei risultati aggiuntivi | |
| 9 | 19-22 | Sezione XV (Extended Tabular) | Eccellente copertura — le 1,740 experiments sono ben documentate | |
| 10 | 22 | Table S-XV | lr per Breast Cancer = 0.001, bs=16 — coerente con main paper Sect. IV-A | OK |
| 11 | 24 | Table S-XX | "Collapse: seeds exhibiting single-class prediction out of 10" — chiaro e ben documentato | |
| 12 | 25 | Sect. XV-M, Fig. S-18 | RDP composition comparison — eccellente visualizzazione | |
| 13 | 25 | Table S-XXII: "2,885" parameters | Main paper Sect. IV-A dice "~10K parameters" per HealthcareMLP. Table S-XXII dice 2,885 per PTB-XL | La discrepanza (2,885 vs ~10K) probabilmente riflette il numero diverso di features input (9 per PTB-XL vs più per altri dataset). Specificare che "~10K" è approssimativo e varia per dataset |

## D2. Sezioni Vuote/Incomplete nel Supplementary

| Sezione | Pag. | Stato |
|---------|------|-------|
| III-C: Client Participation Matrix | 5 | Solo titolo — il contenuto è Fig. S-4 a pag. 6 |
| III-D: Gradient Norm Evolution | 5 | Solo titolo — il contenuto è Fig. S-5 a pag. 6 |
| III-E: Communication Cost Analysis | 5 | Solo titolo — il contenuto è Fig. S-6 a pag. 6 |
| III-F: Learning Rate Sensitivity | 5 | Solo titolo — il contenuto è Fig. S-7 a pag. 6 |
| III-G: Batch Size Impact | 5 | Solo titolo — il contenuto è Fig. S-8 a pag. 6 |
| III-H: Per-Client Accuracy Trajectories | 5 | Solo titolo — il contenuto è Fig. S-9 a pag. 6 |
| V-E: Convergence Speed | 7 | Solo titolo — il contenuto è Fig. S-12 a pag. 8 |

**Nota**: Queste sezioni hanno il titolo senza testo perché il contenuto è fornito dalle figure immediatamente successive. Tuttavia, l'assenza di testo introduttivo per ogni sottosezione è inusuale. Aggiungere almeno una riga di testo per ciascuna o rimuovere le intestazioni e lasciare solo le caption delle figure.

---

# SEZIONE E: CHECKLIST PRE-SUBMISSION

---

## E1. Requisiti Formali

| Requisito | Stato | Azione |
|-----------|-------|--------|
| Page limit (se IEEE conference, tipicamente 8-10 pagine) | Main paper: 9 pagine (OK) | Verificare limiti FLICS 2026 |
| References format IEEE | ✓ Formato IEEE | OK |
| Abstract word count (tipicamente ≤250) | Abstract attuale: ~220 parole | OK |
| Keywords/Index Terms | ✓ Presenti | OK |
| Author information | Singolo autore, affiliazione chiara | OK |
| ORCID | ✓ Presente | OK |
| Repository link | ✓ GitHub link con footnote | Verificare accessibilità |
| Acknowledgments | ✓ Prof. Alawadi menzionato | OK |

## E2. Checklist Tecnica

| Elemento | Stato | Note |
|----------|-------|------|
| Tutti i numeri in tabelle verificati cross-referenza | ⚠️ Discrepanze trovate (vedi A2) | Correggere |
| Figure ad alta risoluzione | ⚠️ Verificare Fig. 1 per stampa | |
| Equazioni numerate | Non numerate (non necessario per IEEE conf.) | OK |
| Acronimi definiti alla prima occorrenza | ✓ | OK |
| Supplementary self-contained | ✓ Ha proprie references | ⚠️ Ma ref [34] differisce |

---

# SEZIONE F: RIEPILOGO PRIORITÀ CORREZIONI

---

## 🔴 CRITICHE (correggere prima della submission)

1. **PRISMA "peer-reviewd"** → "peer-reviewed" (Suppl. pag. 2, Fig. S-1)
2. **Reference [34] discrepanza** tra main paper e supplementary — allineare
3. **Chest X-ray 5,856 vs 5,860** — uniformare a 5,856
4. **PTB-XL "52 EU sites" vs "52 German recording sites"** — uniformare
5. **Algorithm 1 rendering** — verificare che la formula di aggregazione sia leggibile nel PDF compilato
6. **"~10K params" vs 2,885 params** — chiarire che il conteggio parametri varia per dataset

## 🟡 IMPORTANTI (fortemente raccomandate)

7. Uniformare "Fröhlich" con umlaut corretto in LaTeX
8. Uniformare "Rényi" con accento corretto ovunque
9. Aggiungere HPFL a Fig. 2 o giustificarne l'assenza
10. Completare le sottosezioni vuote nel supplementary (III-C–H, V-E)
11. Breast Cancer accuracy discrepanza (57.3% in S-XX vs 52.3% in Tab. VI) — spiegare
12. Specificare che "1,740+" conta solo esperimenti tabular
13. Table VI nota F1: chiarire la metrica per PTB-XL
14. Suppl. Sect. IX-C: rimuovere "per dataset" ridondante

## 🟢 OPZIONALI (miglioramenti stilistici)

15. Sostituire "non-obvious" con "counterintuitive"
16. Riformulare "privacy is essentially free" in modo più formale
17. Definire "pp" alla prima occorrenza
18. Uniformare capitalizzazione "data holder" / "Data Holder"
19. Aggiungere una riga introduttiva alle sezioni mancanti nel supplementary
20. Considerare di aggiungere "Lines of Code" alla Table I

---

# SEZIONE G: VALUTAZIONE COMPLESSIVA

## Punti di Forza
- **Contributo chiaramente definito**: integrazione (non novità algoritmica) — correttamente posizionato
- **Validazione sperimentale estensiva**: 1,740+ esperimenti con ablation approfondite
- **Onestà sulle limitazioni**: il paper è molto trasparente sui limiti (governance simulata, imaging limitato)
- **Risultato principale forte**: la scelta architetturale (personalized vs global) che domina su quella algoritmica è un insight pratico di valore
- **Rilevanza policy**: le raccomandazioni per i delegated acts del 2027 sono concrete e azionabili
- **Supplementary eccezionale**: 29 pagine con analisi dettagliatissima, confusion matrices, scalabilità

## Aree di Miglioramento Principali
- **Inconsistenze numeriche** tra main e supplementary (vedi A2)
- **Reference [34]** è un errore grave che mina la credibilità
- **Typo PRISMA** è visivamente prominente
- **Imaging evaluation limitata** (3 seeds max, 1 seed per Brain Tumor) — ben riconosciuta ma potrebbe essere un punto debole per i reviewer

## Raccomandazione Finale
Il paper è sostanzialmente solido e ben scritto. Le correzioni critiche (6 items 🔴) sono tutte risolvibili in poche ore. Il contributo è chiaro, la validazione è estensiva, e il posizionamento rispetto alla letteratura è appropriato. Dopo le correzioni indicate, il paper è pronto per la submission.

"""
Module Integration Page for FL-EHDS Dashboard.

Three sub-tabs:
1. Compatibility Matrix - Interactive NxN heatmap showing module compatibility
2. Recommended Stacks - Pre-built configurations for common scenarios
3. Integration Explorer - Interactive module selector with warnings

Author: Fabio Liberti
"""

from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
import streamlit as st


# =============================================================================
# MODULE REGISTRY
# =============================================================================

# Module definitions grouped by category
MODULE_CATEGORIES = {
    "Algoritmi FL": {
        "FedAvg": {
            "description": "Media pesata standard dei modelli client",
            "module": "core.fl_algorithms",
            "ehds_articles": [],
        },
        "FedProx": {
            "description": "FedAvg con termine di prossimita per dati non-IID",
            "module": "core.fl_algorithms",
            "ehds_articles": [],
        },
        "SCAFFOLD": {
            "description": "Correzione varianza con control variates",
            "module": "core.fl_algorithms",
            "ehds_articles": [],
        },
        "FedNova": {
            "description": "Normalizzazione per eterogeneita computazionale",
            "module": "core.fl_algorithms",
            "ehds_articles": [],
        },
        "FedAdam": {
            "description": "Ottimizzazione server-side con Adam",
            "module": "core.fl_algorithms",
            "ehds_articles": [],
        },
    },
    "Privacy": {
        "Differential Privacy": {
            "description": "Noise clipping e Renyi DP accounting",
            "module": "orchestration.privacy.differential_privacy",
            "ehds_articles": ["Art. 44", "Art. 50"],
        },
        "Secure Aggregation": {
            "description": "Aggregazione crittografica dei modelli",
            "module": "orchestration.privacy.secure_aggregation",
            "ehds_articles": ["Art. 50"],
        },
        "Jurisdiction Privacy": {
            "description": "Budget epsilon per-paese con accounting RDP",
            "module": "governance.jurisdiction_privacy",
            "ehds_articles": ["Art. 48"],
        },
    },
    "Resilienza": {
        "Krum": {
            "description": "Selezione aggiornamento piu vicino ai vicini",
            "module": "core.byzantine_resilience",
            "ehds_articles": [],
        },
        "Multi-Krum": {
            "description": "Selezione top-m aggiornamenti robusti",
            "module": "core.byzantine_resilience",
            "ehds_articles": [],
        },
        "Trimmed Mean": {
            "description": "Media troncata (rimuove estremi)",
            "module": "core.byzantine_resilience",
            "ehds_articles": [],
        },
        "FLTrust": {
            "description": "Trust scoring basato su dataset root",
            "module": "core.byzantine_resilience",
            "ehds_articles": [],
        },
        "FLAME": {
            "description": "Clustering + clipping per difesa backdoor",
            "module": "core.byzantine_resilience",
            "ehds_articles": [],
        },
    },
    "Architettura Avanzata": {
        "Continual (EWC)": {
            "description": "Elastic Weight Consolidation per drift temporale",
            "module": "core.continual_fl",
            "ehds_articles": [],
        },
        "Transfer Learning": {
            "description": "Pre-train su dati pubblici, fine-tune federato",
            "module": "core.federated_transfer",
            "ehds_articles": [],
        },
        "Vertical FL": {
            "description": "Split Learning per features distribuite",
            "module": "core.vertical_fl",
            "ehds_articles": [],
        },
        "Multi-Task": {
            "description": "Apprendimento simultaneo di task multipli",
            "module": "core.multitask_fl",
            "ehds_articles": [],
        },
        "Personalized FL": {
            "description": "Ditto, Per-FedAvg, FedPer, APFL",
            "module": "core.personalized_fl",
            "ehds_articles": [],
        },
    },
    "Governance EHDS": {
        "Data Permits": {
            "description": "Validazione permit HDAB (Art. 53)",
            "module": "governance.data_permits",
            "ehds_articles": ["Art. 53"],
        },
        "Opt-Out Registry": {
            "description": "Diritto di opt-out cittadini (Art. 71)",
            "module": "governance.optout_registry",
            "ehds_articles": ["Art. 71"],
        },
        "Compliance Audit": {
            "description": "GDPR Art. 30 audit trail completo",
            "module": "governance.compliance_logging",
            "ehds_articles": ["Art. 30 GDPR"],
        },
        "Data Minimization": {
            "description": "Selezione features per finalita (Art. 44)",
            "module": "governance.data_minimization",
            "ehds_articles": ["Art. 44"],
        },
        "Secure Processing": {
            "description": "TEE, watermark, time guard (Art. 50)",
            "module": "governance.secure_processing",
            "ehds_articles": ["Art. 50"],
        },
        "Fee Model": {
            "description": "Tracking costi e ottimizzazione budget (Art. 42)",
            "module": "governance.fee_model",
            "ehds_articles": ["Art. 42"],
        },
        "Data Quality": {
            "description": "Quality labels GOLD/SILVER/BRONZE (Art. 69)",
            "module": "governance.data_quality_framework",
            "ehds_articles": ["Art. 69"],
        },
        "HDAB Routing": {
            "description": "Single Application Point cross-border (Art. 57-58)",
            "module": "governance.hdab_routing",
            "ehds_articles": ["Art. 57-58"],
        },
    },
}


# =============================================================================
# COMPATIBILITY MATRIX
# =============================================================================

# Compatibility levels: "full", "partial", "incompatible", "independent"
# "full" = work perfectly together
# "partial" = work with caveats
# "incompatible" = cannot be used simultaneously
# "independent" = no interaction, can be combined freely

COMPATIBILITY_RULES: Dict[Tuple[str, str], Tuple[str, str]] = {
    # Algorithm x Algorithm = incompatible (pick one)
    ("FedAvg", "FedProx"): ("incompatible", "Scegli un solo algoritmo di aggregazione"),
    ("FedAvg", "SCAFFOLD"): ("incompatible", "Scegli un solo algoritmo di aggregazione"),
    ("FedAvg", "FedNova"): ("incompatible", "Scegli un solo algoritmo di aggregazione"),
    ("FedAvg", "FedAdam"): ("incompatible", "Scegli un solo algoritmo di aggregazione"),
    ("FedProx", "SCAFFOLD"): ("incompatible", "Scegli un solo algoritmo di aggregazione"),
    ("FedProx", "FedNova"): ("incompatible", "Scegli un solo algoritmo di aggregazione"),
    ("FedProx", "FedAdam"): ("incompatible", "Scegli un solo algoritmo di aggregazione"),
    ("SCAFFOLD", "FedNova"): ("incompatible", "Scegli un solo algoritmo di aggregazione"),
    ("SCAFFOLD", "FedAdam"): ("incompatible", "Scegli un solo algoritmo di aggregazione"),
    ("FedNova", "FedAdam"): ("incompatible", "Scegli un solo algoritmo di aggregazione"),
    # Algorithm x Privacy = full
    ("FedAvg", "Differential Privacy"): ("full", "DP si applica dopo ogni round di aggregazione"),
    ("FedProx", "Differential Privacy"): ("full", "DP con prossimita preserva convergenza"),
    ("SCAFFOLD", "Differential Privacy"): ("partial", "Control variates possono amplificare il noise DP"),
    ("FedAdam", "Differential Privacy"): ("full", "Adam server-side + DP client-side"),
    # Algorithm x Byzantine = full (mostly)
    ("FedAvg", "Krum"): ("full", "Krum sostituisce la media pesata di FedAvg"),
    ("FedAvg", "Trimmed Mean"): ("full", "Trimmed Mean come alternativa robusta"),
    ("FedAvg", "FLTrust"): ("full", "Trust scoring + aggregazione pesata"),
    ("FedAvg", "FLAME"): ("full", "Clustering difensivo + aggregazione"),
    ("SCAFFOLD", "Krum"): ("partial", "Control variates + Krum: overhead aggiuntivo"),
    # Algorithm x Governance = full
    ("FedAvg", "Data Permits"): ("full", "Permit validato prima di ogni round"),
    ("FedProx", "Data Permits"): ("full", "Permit validato prima di ogni round"),
    ("FedAvg", "Opt-Out Registry"): ("full", "Dati opt-out esclusi dal training"),
    ("FedAvg", "Compliance Audit"): ("full", "Ogni round loggato nel audit trail"),
    ("FedAvg", "Data Minimization"): ("full", "Features filtrate pre-training"),
    ("FedAvg", "Secure Processing"): ("full", "TEE enclave per ogni client"),
    ("FedAvg", "Fee Model"): ("full", "Costi calcolati per round"),
    ("FedAvg", "Data Quality"): ("full", "Pesi aggregazione basati su qualita dati"),
    ("FedAvg", "HDAB Routing"): ("full", "SAP routing pre-training"),
    # Privacy x Governance = full
    ("Differential Privacy", "Data Permits"): ("full", "Budget DP tracciato nel permit"),
    ("Differential Privacy", "Compliance Audit"): ("full", "Epsilon spending nel audit trail"),
    ("Differential Privacy", "Jurisdiction Privacy"): ("full", "Epsilon per-paese + DP globale"),
    ("Secure Aggregation", "Secure Processing"): ("full", "Crypto + TEE per massima protezione"),
    ("Secure Aggregation", "Differential Privacy"): ("full", "SecAgg + DP: gold standard privacy"),
    # Byzantine x Privacy = partial
    ("Krum", "Differential Privacy"): ("partial", "Krum puo selezionare noise outlier come 'buono'"),
    ("Trimmed Mean", "Differential Privacy"): ("full", "Trimming robusto anche con noise DP"),
    ("FLAME", "Differential Privacy"): ("full", "FLAME clipping compatibile con DP clipping"),
    # Byzantine x Byzantine = incompatible (pick one)
    ("Krum", "Multi-Krum"): ("incompatible", "Scegli una sola strategia Byzantine"),
    ("Krum", "Trimmed Mean"): ("incompatible", "Scegli una sola strategia Byzantine"),
    ("Krum", "FLTrust"): ("incompatible", "Scegli una sola strategia Byzantine"),
    ("Krum", "FLAME"): ("incompatible", "Scegli una sola strategia Byzantine"),
    ("Multi-Krum", "Trimmed Mean"): ("incompatible", "Scegli una sola strategia Byzantine"),
    ("Multi-Krum", "FLTrust"): ("incompatible", "Scegli una sola strategia Byzantine"),
    ("Multi-Krum", "FLAME"): ("incompatible", "Scegli una sola strategia Byzantine"),
    ("Trimmed Mean", "FLTrust"): ("incompatible", "Scegli una sola strategia Byzantine"),
    ("Trimmed Mean", "FLAME"): ("incompatible", "Scegli una sola strategia Byzantine"),
    ("FLTrust", "FLAME"): ("incompatible", "Scegli una sola strategia Byzantine"),
    # Architecture = mostly independent
    ("Vertical FL", "Differential Privacy"): ("partial", "DP per split learning richiede noise su embeddings"),
    ("Vertical FL", "Krum"): ("incompatible", "Byzantine non applicabile a split learning"),
    ("Multi-Task", "FedAvg"): ("partial", "Multi-task usa aggregazione custom per-task"),
    ("Continual (EWC)", "FedAvg"): ("full", "EWC come regolarizzazione locale + FedAvg"),
    ("Transfer Learning", "FedAvg"): ("full", "Pre-train centralizzato + fine-tune federato"),
    ("Personalized FL", "FedAvg"): ("full", "Ditto/APFL come estensione di FedAvg"),
    # Governance x Governance = full (all compose)
    ("Data Permits", "Opt-Out Registry"): ("full", "Permit + opt-out si completano"),
    ("Data Permits", "Compliance Audit"): ("full", "Ogni azione permit loggata"),
    ("Data Permits", "Data Minimization"): ("full", "Minimizzazione applicata dopo permit"),
    ("Data Permits", "Secure Processing"): ("full", "TEE vincolato al permit"),
    ("Data Permits", "Fee Model"): ("full", "Costi associati al permit"),
    ("Data Permits", "Data Quality"): ("full", "Quality labels influenzano pesi"),
    ("Data Permits", "HDAB Routing"): ("full", "SAP routing distribuisce permit"),
    ("Opt-Out Registry", "Compliance Audit"): ("full", "Opt-out loggato nel audit"),
    ("Data Minimization", "Compliance Audit"): ("full", "Riduzione features loggata"),
    ("Secure Processing", "Compliance Audit"): ("full", "TEE events nel audit trail"),
    ("Fee Model", "HDAB Routing"): ("full", "Costi distribuiti per HDAB"),
    ("Data Quality", "Fee Model"): ("full", "Qualita influenza pricing"),
}


# =============================================================================
# RECOMMENDED STACKS
# =============================================================================

RECOMMENDED_STACKS = [
    {
        "name": "EHDS Clinical Research (Base)",
        "description": (
            "Stack base per ricerca clinica con conformita EHDS. "
            "Ideale per progetti single-country con dati tabular."
        ),
        "modules": [
            "FedAvg", "Differential Privacy", "Data Permits",
            "Compliance Audit", "Opt-Out Registry",
        ],
        "ehds_coverage": ["Art. 33", "Art. 44", "Art. 53", "Art. 71", "Art. 30 GDPR"],
        "use_case": "Ricerca clinica, analisi epidemiologica",
        "complexity": "Bassa",
    },
    {
        "name": "Cross-Border Regulated (Avanzato)",
        "description": (
            "Stack completo per FL cross-border con governance multi-HDAB. "
            "Massima copertura normativa EHDS."
        ),
        "modules": [
            "FedProx", "Differential Privacy", "Jurisdiction Privacy",
            "Data Permits", "Compliance Audit", "Opt-Out Registry",
            "Data Minimization", "Secure Processing", "Fee Model",
            "Data Quality", "HDAB Routing",
        ],
        "ehds_coverage": [
            "Art. 33", "Art. 34", "Art. 42", "Art. 44", "Art. 48",
            "Art. 50", "Art. 53", "Art. 57-58", "Art. 69", "Art. 71",
            "Art. 30 GDPR",
        ],
        "use_case": "FL cross-border multi-paese, studi paneuropei",
        "complexity": "Alta",
    },
    {
        "name": "Byzantine-Resilient Clinical",
        "description": (
            "Stack per ambienti con client potenzialmente non affidabili. "
            "SCAFFOLD per convergenza + Krum per resilienza."
        ),
        "modules": [
            "SCAFFOLD", "Differential Privacy", "Krum",
            "Continual (EWC)", "Data Permits", "Compliance Audit",
        ],
        "ehds_coverage": ["Art. 33", "Art. 44", "Art. 53", "Art. 30 GDPR"],
        "use_case": "FL in reti non fidate, ambienti con data drift",
        "complexity": "Media",
    },
    {
        "name": "Multi-Hospital Imaging",
        "description": (
            "Stack ottimizzato per imaging diagnostico multi-ospedale. "
            "FedAvg + DP + quality-weighted aggregation."
        ),
        "modules": [
            "FedAvg", "Differential Privacy", "Data Permits",
            "Compliance Audit", "Data Quality", "Opt-Out Registry",
        ],
        "ehds_coverage": ["Art. 33", "Art. 44", "Art. 53", "Art. 69", "Art. 71", "Art. 30 GDPR"],
        "use_case": "Classificazione imaging (X-ray, MRI, CT)",
        "complexity": "Media",
    },
    {
        "name": "Privacy-Maximum",
        "description": (
            "Stack con massima protezione privacy. "
            "DP + Secure Aggregation + TEE + Minimizzazione."
        ),
        "modules": [
            "FedAvg", "Differential Privacy", "Secure Aggregation",
            "Secure Processing", "Data Minimization", "Data Permits",
            "Compliance Audit", "Opt-Out Registry",
        ],
        "ehds_coverage": [
            "Art. 33", "Art. 44", "Art. 50", "Art. 53", "Art. 71",
            "Art. 30 GDPR",
        ],
        "use_case": "Dati altamente sensibili (genomici, psichiatrici)",
        "complexity": "Alta",
    },
]


# =============================================================================
# RENDERING FUNCTIONS
# =============================================================================


def _get_all_module_names() -> List[str]:
    """Get flat list of all module names."""
    names = []
    for cat_modules in MODULE_CATEGORIES.values():
        names.extend(cat_modules.keys())
    return names


def _get_compatibility(mod_a: str, mod_b: str) -> Tuple[str, str]:
    """Get compatibility between two modules."""
    if mod_a == mod_b:
        return ("self", "Stesso modulo")

    # Check both orderings
    key1 = (mod_a, mod_b)
    key2 = (mod_b, mod_a)

    if key1 in COMPATIBILITY_RULES:
        return COMPATIBILITY_RULES[key1]
    if key2 in COMPATIBILITY_RULES:
        return COMPATIBILITY_RULES[key2]

    # Default: independent (can be combined freely)
    return ("independent", "Indipendenti, combinabili liberamente")


def _render_compatibility_matrix():
    """Sub-tab: Interactive compatibility matrix."""
    st.markdown("##### Matrice di Compatibilita Moduli")
    st.markdown(
        "Visualizza la compatibilita tra i moduli del framework. "
        "Seleziona una categoria per esplorare le interazioni."
    )

    # Category filter
    categories = list(MODULE_CATEGORIES.keys())
    selected_cats = st.multiselect(
        "Categorie da visualizzare",
        options=categories,
        default=categories[:3],
        key="mi_matrix_cats",
    )

    if not selected_cats:
        st.info("Seleziona almeno una categoria.")
        return

    # Gather modules from selected categories
    selected_modules = []
    for cat in selected_cats:
        selected_modules.extend(MODULE_CATEGORIES[cat].keys())

    n = len(selected_modules)
    if n == 0:
        return

    # Build matrix
    matrix_data = []
    for mod_a in selected_modules:
        row = {}
        for mod_b in selected_modules:
            level, _ = _get_compatibility(mod_a, mod_b)
            row[mod_b] = level
        matrix_data.append(row)

    df = pd.DataFrame(matrix_data, index=selected_modules)

    # Color mapping for display
    level_display = {
        "full": "OK",
        "partial": "~",
        "incompatible": "X",
        "independent": "-",
        "self": "=",
    }

    display_df = df.map(lambda x: level_display.get(x, x))
    st.dataframe(display_df, use_container_width=True, height=min(400, n * 35 + 50))

    # Legend
    st.markdown(
        "**Legenda:** "
        "`OK` = Pienamente compatibile | "
        "`~` = Compatibile con limitazioni | "
        "`X` = Incompatibile | "
        "`-` = Indipendente | "
        "`=` = Stesso modulo"
    )

    # Detail on click
    st.markdown("---")
    st.markdown("**Dettaglio Compatibilita**")
    detail_cols = st.columns(2)
    with detail_cols[0]:
        mod_a = st.selectbox(
            "Modulo A",
            options=selected_modules,
            key="mi_detail_a",
        )
    with detail_cols[1]:
        mod_b = st.selectbox(
            "Modulo B",
            options=selected_modules,
            key="mi_detail_b",
        )

    if mod_a and mod_b:
        level, note = _get_compatibility(mod_a, mod_b)
        level_labels = {
            "full": "Pienamente Compatibile",
            "partial": "Compatibile con Limitazioni",
            "incompatible": "Incompatibile",
            "independent": "Indipendente",
            "self": "Stesso Modulo",
        }
        st.info(f"**{mod_a}** + **{mod_b}**: {level_labels.get(level, level)}\n\n{note}")


def _render_recommended_stacks():
    """Sub-tab: Pre-built recommended configurations."""
    st.markdown("##### Stack Consigliati")
    st.markdown(
        "Configurazioni pre-costruite e testate per scenari comuni di FL in ambito sanitario. "
        "Ogni stack e progettato per coprire specifici articoli EHDS."
    )

    for i, stack in enumerate(RECOMMENDED_STACKS):
        with st.expander(
            f"**{stack['name']}** - Complessita: {stack['complexity']}",
            expanded=(i == 0),
        ):
            st.markdown(stack["description"])

            info_cols = st.columns(2)
            with info_cols[0]:
                st.markdown("**Moduli inclusi:**")
                for mod in stack["modules"]:
                    # Find category
                    cat_label = ""
                    for cat_name, cat_mods in MODULE_CATEGORIES.items():
                        if mod in cat_mods:
                            cat_label = cat_name
                            break
                    st.markdown(f"- {mod} *({cat_label})*")

            with info_cols[1]:
                st.markdown("**Copertura EHDS:**")
                for art in stack["ehds_coverage"]:
                    st.markdown(f"- {art}")

                st.markdown(f"\n**Use Case:** {stack['use_case']}")

            # Compatibility check
            warnings = []
            modules = stack["modules"]
            for j, mod_a in enumerate(modules):
                for mod_b in modules[j + 1:]:
                    level, note = _get_compatibility(mod_a, mod_b)
                    if level == "partial":
                        warnings.append(f"**{mod_a}** + **{mod_b}**: {note}")
                    elif level == "incompatible":
                        warnings.append(
                            f"**{mod_a}** + **{mod_b}**: INCOMPATIBILE - {note}"
                        )

            if warnings:
                st.markdown("**Avvertenze:**")
                for w in warnings:
                    st.warning(w)
            else:
                st.success("Tutti i moduli sono pienamente compatibili.")


def _render_integration_explorer():
    """Sub-tab: Interactive module selector with compatibility warnings."""
    st.markdown("##### Integration Explorer")
    st.markdown(
        "Seleziona i moduli che vuoi combinare e verifica "
        "la compatibilita e la copertura EHDS."
    )

    # Module selection by category
    selected_modules = []

    for cat_name, cat_modules in MODULE_CATEGORIES.items():
        st.markdown(f"**{cat_name}**")
        mod_cols = st.columns(min(4, len(cat_modules)))
        for i, (mod_name, mod_info) in enumerate(cat_modules.items()):
            with mod_cols[i % len(mod_cols)]:
                if st.checkbox(
                    mod_name,
                    key=f"mi_explorer_{mod_name}",
                    help=mod_info["description"],
                ):
                    selected_modules.append(mod_name)

    if not selected_modules:
        st.info("Seleziona almeno un modulo per vedere l'analisi di compatibilita.")
        return

    st.markdown("---")

    # Compatibility analysis
    st.markdown(f"##### Analisi ({len(selected_modules)} moduli selezionati)")

    warnings = []
    errors = []
    for i, mod_a in enumerate(selected_modules):
        for mod_b in selected_modules[i + 1:]:
            level, note = _get_compatibility(mod_a, mod_b)
            if level == "partial":
                warnings.append(f"**{mod_a}** + **{mod_b}**: {note}")
            elif level == "incompatible":
                errors.append(f"**{mod_a}** + **{mod_b}**: {note}")

    if errors:
        st.error(f"**{len(errors)} incompatibilita rilevate:**")
        for e in errors:
            st.markdown(f"- {e}")

    if warnings:
        st.warning(f"**{len(warnings)} avvertenze:**")
        for w in warnings:
            st.markdown(f"- {w}")

    if not errors and not warnings:
        st.success("Tutti i moduli selezionati sono pienamente compatibili!")

    # EHDS article coverage
    st.markdown("---")
    st.markdown("##### Copertura Articoli EHDS")

    covered_articles: Set[str] = set()
    for mod_name in selected_modules:
        for cat_modules in MODULE_CATEGORIES.values():
            if mod_name in cat_modules:
                covered_articles.update(cat_modules[mod_name].get("ehds_articles", []))

    all_articles = [
        "Art. 33", "Art. 34", "Art. 42", "Art. 44", "Art. 46",
        "Art. 48", "Art. 50", "Art. 53", "Art. 57-58", "Art. 69",
        "Art. 71", "Art. 30 GDPR",
    ]

    coverage_rows = []
    for art in all_articles:
        is_covered = art in covered_articles
        coverage_rows.append({
            "Articolo": art,
            "Stato": "Coperto" if is_covered else "Non coperto",
            "Modulo": next(
                (m for m in selected_modules
                 for cat in MODULE_CATEGORIES.values()
                 if m in cat and art in cat[m].get("ehds_articles", [])),
                "-",
            ),
        })

    coverage_df = pd.DataFrame(coverage_rows)
    st.dataframe(coverage_df, use_container_width=True)

    covered = sum(1 for r in coverage_rows if r["Stato"] == "Coperto")
    total = len(all_articles)
    st.metric(
        "Copertura EHDS",
        f"{covered}/{total} articoli ({covered / total * 100:.0f}%)",
    )


def render_module_integration_tab():
    """Main entry point for the Module Integration tab."""
    st.markdown("### Integrazione Moduli")
    st.markdown(
        "Esplora la compatibilita tra i moduli del framework FL-EHDS, "
        "scopri gli stack consigliati e costruisci configurazioni personalizzate."
    )

    tabs = st.tabs([
        "Matrice Compatibilita",
        "Stack Consigliati",
        "Integration Explorer",
    ])

    with tabs[0]:
        _render_compatibility_matrix()

    with tabs[1]:
        _render_recommended_stacks()

    with tabs[2]:
        _render_integration_explorer()

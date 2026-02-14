#!/usr/bin/env python3
"""
FL-EHDS Interactive Dashboard v4.0

Complete Edition with:
1. All FL algorithms from v3 (FedAvg, FedProx, SCAFFOLD, FedAdam, etc.)
2. All model architectures from v3 (MLP, CNN, Transformer, etc.)
3. Complete heterogeneity and participation modes
4. Full configuration panel with explanations
5. NEW: Vertical FL / Split Learning (PSI, SplitNN)
6. NEW: Byzantine Resilience (Krum, Trimmed Mean, Bulyan, FLTrust, FLAME)
7. NEW: Continual Learning (EWC, LwF, Experience Replay, Drift Detection)
8. NEW: Multi-Task FL (Hard/Soft Sharing, FedMTL)
9. NEW: Hierarchical FL (Client → Regional → National → EU)

This module has been split into dashboard/ sub-modules for maintainability.
Simulators and constants are re-exported here for backward compatibility.

Author: Fabio Liberti
Usage: streamlit run app_v4.py
"""

import streamlit as st
from pathlib import Path

# ============================================================================
# Re-exports for backward compatibility
# Terminal screens import simulators from dashboard.app_v4
# ============================================================================
from dashboard.constants import (  # noqa: F401
    ALGORITHMS,
    MODELS,
    HETEROGENEITY_TYPES,
    PARTICIPATION_MODES,
    BYZANTINE_METHODS,
    CONTINUAL_METHODS,
    MULTITASK_METHODS,
    show_algorithm_help,
    show_model_help,
)
from dashboard.simulators import (  # noqa: F401
    FLSimulatorV4,
    VerticalFLSimulator,
    ByzantineSimulator,
    ContinualFLSimulator,
    MultiTaskFLSimulator,
    HierarchicalFLSimulator,
)
from dashboard.config_panel import (
    create_config_panel,
    plot_vertical_fl_architecture,
    plot_hierarchy_tree,
)
from dashboard.training_tab import render_training_tab
from dashboard.advanced_fl_tabs import (
    render_algorithms_tab,
    render_models_tab,
    render_vertical_fl_tab,
    render_byzantine_tab,
    render_continual_tab,
    render_multitask_tab,
    render_hierarchical_tab,
)
from dashboard.ehds_tab import (
    render_ehds_tab,
    render_infrastructure_tab,
    render_guide_tab,
)

# ============================================================================
# Optional page imports
# ============================================================================

# Dataset management page
try:
    from dashboard.dataset_page import render_dataset_tab
    HAS_DATASET_PAGE = True
except ImportError:
    try:
        from dataset_page import render_dataset_tab
        HAS_DATASET_PAGE = True
    except ImportError:
        HAS_DATASET_PAGE = False

# Paper experiments page
try:
    from dashboard.paper_experiments_page import render_paper_experiments_tab
    HAS_PAPER_EXPERIMENTS = True
except ImportError:
    try:
        from paper_experiments_page import render_paper_experiments_tab
        HAS_PAPER_EXPERIMENTS = True
    except ImportError:
        HAS_PAPER_EXPERIMENTS = False

# Governance workflow page
try:
    from dashboard.governance_workflow_page import render_governance_workflow_tab
    HAS_GOVERNANCE_WORKFLOW = True
except ImportError:
    try:
        from governance_workflow_page import render_governance_workflow_tab
        HAS_GOVERNANCE_WORKFLOW = True
    except ImportError:
        HAS_GOVERNANCE_WORKFLOW = False

# Permit manager page
try:
    from dashboard.permit_manager_page import render_permit_manager_tab
    HAS_PERMIT_MANAGER = True
except ImportError:
    try:
        from permit_manager_page import render_permit_manager_tab
        HAS_PERMIT_MANAGER = True
    except ImportError:
        HAS_PERMIT_MANAGER = False

# Module integration page
try:
    from dashboard.module_integration_page import render_module_integration_tab
    HAS_MODULE_INTEGRATION = True
except ImportError:
    try:
        from module_integration_page import render_module_integration_tab
        HAS_MODULE_INTEGRATION = True
    except ImportError:
        HAS_MODULE_INTEGRATION = False

# ============================================================================
# Streamlit page config and CSS
# ============================================================================

st.set_page_config(
    page_title="FL-EHDS Dashboard v4",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #1f77b4, #2ecc71, #9b59b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .info-box {
        background-color: #e7f3ff;
        border-left: 4px solid #1f77b4;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 8px 8px 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 8px 8px 0;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 8px 8px 0;
    }
    .danger-box {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 8px 8px 0;
    }
    .param-desc {
        font-size: 0.85rem;
        color: #666;
        font-style: italic;
        margin-top: -10px;
        margin-bottom: 10px;
    }
    .algorithm-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 1rem;
        color: white;
        margin: 0.5rem 0;
    }
    .module-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        border-radius: 10px;
        padding: 1rem;
        color: white;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# Main application
# ============================================================================


def main():
    """Main application."""
    st.markdown('<div class="main-header">FL-EHDS Dashboard v4.0</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Federated Learning for European Health Data Space - Complete Advanced Edition</div>',
                unsafe_allow_html=True)

    config = create_config_panel()

    tabs = st.tabs([
        "🚀 Training",
        "🧮 Algoritmi",
        "🧠 Modelli",
        "📁 Dataset",
        "📊 Vertical FL",
        "🛡️ Byzantine",
        "🔄 Continual",
        "🎯 Multi-Task",
        "🏛️ Hierarchical",
        "🇪🇺 EHDS Governance",
        "📋 Permit Manager",
        "🔗 Integrazione Moduli",
        "⚙️ Infrastructure",
        "📊 Paper Experiments",
        "🇪🇺 EHDS Reference",
        "📚 Guida"
    ])

    with tabs[0]:
        render_training_tab(config)

    with tabs[1]:
        render_algorithms_tab()

    with tabs[2]:
        render_models_tab()

    with tabs[3]:
        if HAS_DATASET_PAGE:
            render_dataset_tab()
        else:
            st.warning("Modulo dataset non disponibile")
            st.info("Assicurati che dataset_page.py sia presente nella cartella dashboard/")

    with tabs[4]:
        render_vertical_fl_tab()

    with tabs[5]:
        render_byzantine_tab()

    with tabs[6]:
        render_continual_tab()

    with tabs[7]:
        render_multitask_tab()

    with tabs[8]:
        render_hierarchical_tab()

    with tabs[9]:
        if HAS_GOVERNANCE_WORKFLOW:
            render_governance_workflow_tab()
        else:
            st.warning("Modulo Governance Workflow non disponibile")
            st.info("Assicurati che governance_workflow_page.py sia presente nella cartella dashboard/")

    with tabs[10]:
        if HAS_PERMIT_MANAGER:
            render_permit_manager_tab()
        else:
            st.warning("Modulo Permit Manager non disponibile")
            st.info("Assicurati che permit_manager_page.py sia presente nella cartella dashboard/")

    with tabs[11]:
        if HAS_MODULE_INTEGRATION:
            render_module_integration_tab()
        else:
            st.warning("Modulo Integrazione non disponibile")
            st.info("Assicurati che module_integration_page.py sia presente nella cartella dashboard/")

    with tabs[12]:
        render_infrastructure_tab()

    with tabs[13]:
        if HAS_PAPER_EXPERIMENTS:
            render_paper_experiments_tab()
        else:
            st.warning("Modulo Paper Experiments non disponibile")
            st.info("Assicurati che paper_experiments_page.py sia presente nella cartella dashboard/")

    with tabs[14]:
        render_ehds_tab()

    with tabs[15]:
        render_guide_tab()

    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #888; font-size: 0.9rem;'>
        FL-EHDS Framework v4.0 | FLICS 2026 |
        9 Algoritmi FL | 11 Architetture Modello | 16 Moduli Governance |
        Governance • Permits • Integration • Dataset • Vertical • Byzantine • Continual • Multi-Task • Hierarchical • EHDS
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

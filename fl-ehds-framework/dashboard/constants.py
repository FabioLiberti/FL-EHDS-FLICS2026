"""FL-EHDS Dashboard constants and configuration dictionaries."""

import streamlit as st

# =============================================================================
# COMPLETE ALGORITHM DEFINITIONS
# =============================================================================

ALGORITHMS = {
    "FedAvg": {
        "name": "Federated Averaging",
        "paper": "McMahan et al., 2017",
        "description": "Algoritmo FL standard. Aggrega gli aggiornamenti dei client usando una media pesata.",
        "pros": ["Semplice da implementare", "Efficiente in comunicazione", "Buono per dati IID"],
        "cons": ["Difficoltà con dati non-IID", "Nessuna garanzia di convergenza per non-convessi"],
        "params": ["learning_rate", "local_epochs"],
        "best_for": "Esperimenti baseline, scenari con dati IID",
        "complexity": "⭐ Bassa"
    },
    "FedProx": {
        "name": "Federated Proximal",
        "paper": "Li et al., 2020",
        "description": "Aggiunge un termine prossimale per prevenire la deriva dei client. Regolarizza gli aggiornamenti locali verso il modello globale.",
        "pros": ["Migliore per dati non-IID", "Gestisce l'eterogeneità", "Garanzie di convergenza"],
        "cons": ["Parametro μ da calibrare", "Convergenza leggermente più lenta"],
        "params": ["learning_rate", "local_epochs", "mu"],
        "best_for": "Dati non-IID, sistemi eterogenei",
        "complexity": "⭐⭐ Media"
    },
    "SCAFFOLD": {
        "name": "Stochastic Controlled Averaging",
        "paper": "Karimireddy et al., 2020",
        "description": "Usa variabili di controllo per correggere la deriva dei client. Riduzione della varianza tramite gradienti corretti.",
        "pros": ["Gestisce non-IID estremo", "Convergenza rapida", "Riduzione varianza"],
        "cons": ["2x costo comunicazione", "Implementazione più complessa"],
        "params": ["learning_rate", "local_lr"],
        "best_for": "Non-IID estremo, quando la comunicazione non è il collo di bottiglia",
        "complexity": "⭐⭐⭐ Alta"
    },
    "FedAdam": {
        "name": "Federated Adam",
        "paper": "Reddi et al., 2021",
        "description": "Applica l'ottimizzatore Adam lato server per aggregare gli aggiornamenti. Learning rate adattivo.",
        "pros": ["Learning rate adattivo", "Buono per gradienti sparsi", "Training stabile"],
        "cons": ["Più calcolo lato server", "Memoria per termini di momento"],
        "params": ["server_lr", "client_lr", "beta1", "beta2", "tau"],
        "best_for": "Modelli complessi, magnitudini di gradiente variabili",
        "complexity": "⭐⭐ Media"
    },
    "FedYogi": {
        "name": "Federated Yogi",
        "paper": "Reddi et al., 2021",
        "description": "Variante di FedAdam con aggiornamento del secondo momento diverso. Adattamento più aggressivo.",
        "pros": ["Migliore di FedAdam per alcuni task", "Gestisce dati non stazionari"],
        "cons": ["Sensibile agli iperparametri", "Può essere instabile"],
        "params": ["server_lr", "client_lr", "beta1", "beta2", "tau"],
        "best_for": "Distribuzioni non stazionarie, concept drift",
        "complexity": "⭐⭐ Media"
    },
    "FedAdagrad": {
        "name": "Federated Adagrad",
        "paper": "Reddi et al., 2021",
        "description": "Ottimizzatore Adagrad lato server. Accumula gradienti quadrati per learning rate adattivo.",
        "pros": ["Metodo adattivo semplice", "Buono per feature sparse"],
        "cons": ["Learning rate decresce nel tempo", "Può convergere lentamente"],
        "params": ["server_lr", "client_lr", "tau"],
        "best_for": "Gradienti sparsi, inizio training",
        "complexity": "⭐⭐ Media"
    },
    "FedNova": {
        "name": "Normalized Averaging",
        "paper": "Wang et al., 2020",
        "description": "Normalizza gli aggiornamenti per numero di step locali. Corregge l'inconsistenza degli obiettivi.",
        "pros": ["Gestisce epoche locali variabili", "Migliore convergenza"],
        "cons": ["Richiede tracking degli step locali", "Leggermente più complesso"],
        "params": ["learning_rate", "local_epochs"],
        "best_for": "Computazione eterogenea, epoche locali variabili",
        "complexity": "⭐⭐ Media"
    },
    "FedDyn": {
        "name": "Dynamic Regularization",
        "paper": "Acar et al., 2021",
        "description": "Regolarizzazione dinamica basata sul modello globale. Allinea obiettivi locali e globali.",
        "pros": ["Forti garanzie di convergenza", "Gestisce partecipazione parziale"],
        "cons": ["Memoria extra per termini di gradiente", "Sensibilita' agli iperparametri"],
        "params": ["learning_rate", "alpha"],
        "best_for": "Partecipazione parziale dei client, garanzie teoriche",
        "complexity": "⭐⭐⭐ Alta"
    },
    "Ditto": {
        "name": "Ditto - Fair FL",
        "paper": "Li et al., 2021",
        "description": "Apprende sia modello globale che personalizzato. Bilancia performance globale con fairness locale.",
        "pros": ["Garanzie di fairness", "Robusto all'eterogeneità"],
        "cons": ["Due modelli per client", "Overhead di memoria"],
        "params": ["learning_rate", "lambda_ditto"],
        "best_for": "Applicazioni critiche per fairness, healthcare",
        "complexity": "⭐⭐⭐ Alta"
    }
}

# =============================================================================
# COMPLETE MODEL DEFINITIONS
# =============================================================================

MODELS = {
    "SimpleMLP": {
        "name": "Simple MLP",
        "type": "MLP",
        "description": "Multi-layer perceptron base con layer nascosti configurabili. Ideale per dati tabulari sanitari.",
        "input_type": "Tabulare",
        "params": ["hidden_dims", "dropout"],
        "use_case": "Dati tabulari, piccoli dataset, baseline",
        "complexity": "⭐ Bassa",
        "params_count": "~10K"
    },
    "DeepMLP": {
        "name": "Deep MLP",
        "type": "MLP",
        "description": "MLP profondo con batch normalization e connessioni residuali. Per pattern tabulari complessi.",
        "input_type": "Tabulare",
        "params": ["hidden_dims", "use_batchnorm", "use_residual"],
        "use_case": "Dati tabulari complessi, dataset più grandi",
        "complexity": "⭐⭐ Media",
        "params_count": "~50K"
    },
    "LeNet5": {
        "name": "LeNet-5",
        "type": "CNN",
        "description": "Architettura CNN classica del 1998. Per piccole immagini in scala di grigi come MNIST.",
        "input_type": "Immagine (28x28)",
        "params": ["in_channels", "num_classes"],
        "use_case": "MNIST, piccole immagini mediche",
        "complexity": "⭐ Bassa",
        "params_count": "~60K"
    },
    "SimpleCNN": {
        "name": "Simple CNN",
        "type": "CNN",
        "description": "CNN leggera con 3 layer convoluzionali. Buon equilibrio tra accuratezza e costo di comunicazione.",
        "input_type": "Immagine (32x32)",
        "params": ["in_channels", "num_classes", "base_channels"],
        "use_case": "CIFAR-10, immagini piccole-medie",
        "complexity": "⭐⭐ Media",
        "params_count": "~200K"
    },
    "VGGStyle": {
        "name": "VGG-Style CNN",
        "type": "CNN",
        "description": "Architettura ispirata a VGG con blocchi conv ripetuti. Più profonda di SimpleCNN.",
        "input_type": "Immagine (64x64+)",
        "params": ["num_blocks", "base_channels"],
        "use_case": "Classificazione immagini complesse",
        "complexity": "⭐⭐⭐ Alta",
        "params_count": "~1M"
    },
    "ResNet18": {
        "name": "ResNet-18",
        "type": "CNN",
        "description": "ResNet a 18 layer con skip connections. Standard per imaging medico.",
        "input_type": "Immagine (224x224)",
        "params": ["pretrained", "num_classes"],
        "use_case": "Imaging medico, transfer learning",
        "complexity": "⭐⭐⭐ Alta",
        "params_count": "~11M"
    },
    "MobileNetStyle": {
        "name": "MobileNet-Style",
        "type": "CNN",
        "description": "CNN leggera con convoluzioni separabili depthwise. Efficiente per deployment edge.",
        "input_type": "Immagine",
        "params": ["width_multiplier"],
        "use_case": "Deploy edge, FL con banda limitata",
        "complexity": "⭐⭐ Media",
        "params_count": "~500K"
    },
    "MedicalCNN": {
        "name": "Medical CNN",
        "type": "CNN",
        "description": "CNN ottimizzata per imaging medico con kernel più grandi e meccanismi di attenzione.",
        "input_type": "Immagine medica",
        "params": ["in_channels", "use_attention"],
        "use_case": "Chest X-ray, immagini fundus, CT scan",
        "complexity": "⭐⭐⭐ Alta",
        "params_count": "~2M"
    },
    "DenseNetMedical": {
        "name": "DenseNet Medical",
        "type": "CNN",
        "description": "Architettura stile DenseNet con connessioni dense. Uso efficiente dei parametri.",
        "input_type": "Immagine medica",
        "params": ["growth_rate", "num_blocks"],
        "use_case": "Imaging medico con dati limitati",
        "complexity": "⭐⭐⭐ Alta",
        "params_count": "~1M"
    },
    "VisionTransformer": {
        "name": "Vision Transformer (ViT)",
        "type": "Transformer",
        "description": "Architettura Transformer per immagini. Divide l'immagine in patch per self-attention.",
        "input_type": "Immagine (224x224)",
        "params": ["patch_size", "embed_dim", "num_heads", "num_layers"],
        "use_case": "Dataset grandi, pattern complessi",
        "complexity": "⭐⭐⭐⭐ Molto Alta",
        "params_count": "~20M+"
    },
    "LSTM": {
        "name": "LSTM",
        "type": "RNN",
        "description": "Long Short-Term Memory per sequenze. Ideale per dati EHR time-series.",
        "input_type": "Sequenza",
        "params": ["hidden_size", "num_layers", "bidirectional"],
        "use_case": "Time-series, EHR sequenziali",
        "complexity": "⭐⭐ Media",
        "params_count": "~100K"
    }
}

# =============================================================================
# HETEROGENEITY AND PARTICIPATION MODES
# =============================================================================

HETEROGENEITY_TYPES = {
    "combined": {
        "name": "Combined (Realistico)",
        "description": "Combina tutti i tipi di eterogeneità. Scenario più realistico per EHDS con ospedali europei diversi.",
        "params": ["label_alpha", "feature_strength", "quantity_imbalance"]
    },
    "label_skew": {
        "name": "Label Skew (Dirichlet)",
        "description": "Distribuzione delle label diversa per ogni nodo. Controllata dalla distribuzione Dirichlet con parametro α.",
        "params": ["alpha"]
    },
    "feature_skew": {
        "name": "Feature Skew",
        "description": "Distribuzioni delle feature diverse per nodo. Simula diverse demografiche dei pazienti tra ospedali.",
        "params": ["feature_strength"]
    },
    "quantity_skew": {
        "name": "Quantity Skew",
        "description": "Dimensioni del dataset sbilanciate. Alcuni nodi hanno molto più dati di altri (grandi vs piccoli ospedali).",
        "params": ["imbalance_ratio"]
    },
    "concept_drift": {
        "name": "Concept Drift",
        "description": "Stesso X, diverso P(Y|X). Simula diverse pratiche cliniche tra ospedali con stesse feature ma decisioni diverse.",
        "params": []
    }
}

PARTICIPATION_MODES = {
    "uniform": {
        "name": "Uniforme Random",
        "description": "Partecipazione random semplice con probabilità fissa. Ogni nodo ha la stessa probabilità di partecipare."
    },
    "reliability": {
        "name": "Basata su Affidabilità",
        "description": "Ogni nodo ha un profilo di affidabilità diverso. Alcuni nodi sono più stabili di altri."
    },
    "realistic": {
        "name": "Realistico (Tempo + Rete)",
        "description": "Include pattern basati sull'ora del giorno, fusi orari diversi, e failure di rete casuali."
    },
    "adversarial": {
        "name": "Adversarial Dropout",
        "description": "Alcuni nodi si disconnettono strategicamente quando hanno gradienti outlier."
    }
}

# =============================================================================
# ADVANCED FL MODULE DEFINITIONS
# =============================================================================

BYZANTINE_METHODS = {
    "krum": {
        "name": "Krum",
        "description": "Seleziona l'aggiornamento più vicino ai suoi vicini. Tollera fino a f < n/2 - 1 client Bizantini.",
        "paper": "Blanchard et al., 2017",
        "tolerance": "f < n/2 - 1"
    },
    "multi_krum": {
        "name": "Multi-Krum",
        "description": "Media dei top-m aggiornamenti per score Krum. Migliore accuratezza del singolo Krum.",
        "paper": "Blanchard et al., 2017",
        "tolerance": "f < n/2 - 1"
    },
    "trimmed_mean": {
        "name": "Trimmed Mean",
        "description": "Rimuove valori estremi prima della media. Semplice ed efficace.",
        "paper": "Yin et al., 2018",
        "tolerance": "f < n/2"
    },
    "median": {
        "name": "Coordinate-wise Median",
        "description": "Calcola la mediana di ogni coordinata. Molto robusto ma convergenza più lenta.",
        "paper": "Yin et al., 2018",
        "tolerance": "f < n/2"
    },
    "bulyan": {
        "name": "Bulyan",
        "description": "Combina selezione Krum con trimmed mean. Resilienza Bizantina più forte.",
        "paper": "El Mhamdi et al., 2018",
        "tolerance": "f < n/4 - 1"
    },
    "fltrust": {
        "name": "FLTrust",
        "description": "Usa dataset root fidato per validare aggiornamenti client. Ideale per deployment reali.",
        "paper": "Cao et al., 2020",
        "tolerance": "Qualsiasi f < n"
    },
    "flame": {
        "name": "FLAME",
        "description": "Difesa basata su clustering usando HDBSCAN. Identifica e rimuove cluster malevoli.",
        "paper": "Nguyen et al., 2022",
        "tolerance": "Dipendente dal cluster"
    }
}

CONTINUAL_METHODS = {
    "ewc": {
        "name": "Elastic Weight Consolidation (EWC)",
        "description": "Usa Fisher Information per identificare pesi importanti e prevenire il forgetting.",
        "paper": "Kirkpatrick et al., 2017",
        "pros": ["Nessun buffer di replay", "Teoricamente fondato"],
        "cons": ["Accumula vincoli", "Memoria per matrice Fisher"]
    },
    "lwf": {
        "name": "Learning without Forgetting (LwF)",
        "description": "Usa knowledge distillation dal modello precedente. Regolarizza via soft labels.",
        "paper": "Li & Hoiem, 2016",
        "pros": ["Nessuna memoria esplicita", "Funziona bene per classificazione"],
        "cons": ["Richiede forward pass del vecchio modello", "Task-specific"]
    },
    "replay": {
        "name": "Experience Replay",
        "description": "Memorizza campioni rappresentativi dai task precedenti. Approccio più diretto.",
        "paper": "Various",
        "pros": ["Semplice ed efficace", "Flessibile"],
        "cons": ["Overhead di memoria", "Problemi di privacy in FL"]
    }
}

MULTITASK_METHODS = {
    "hard_sharing": {
        "name": "Hard Parameter Sharing",
        "description": "Backbone condiviso con head task-specifiche. Più efficiente in parametri.",
        "architecture": "Layer condivisi → Head task-specifiche"
    },
    "soft_sharing": {
        "name": "Soft Parameter Sharing",
        "description": "Reti separate con regolarizzazione per mantenerle simili.",
        "architecture": "Reti parallele con vincoli L2"
    },
    "fedmtl": {
        "name": "FedMTL",
        "description": "Multi-task learning federato con aggregazione task-aware.",
        "architecture": "Aggregazione per cluster di task"
    }
}

EHDS_TASKS = {
    "diabetes_risk": {"name": "Predizione Rischio Diabete", "type": "binary"},
    "readmission_30d": {"name": "Readmissione a 30 Giorni", "type": "binary"},
    "los_prediction": {"name": "Durata Degenza (LOS)", "type": "regression"},
    "mortality_risk": {"name": "Rischio Mortalità", "type": "binary"},
    "sepsis_onset": {"name": "Rilevamento Sepsi", "type": "binary"}
}

# =============================================================================
# HELP DISPLAY FUNCTIONS
# =============================================================================


def show_algorithm_help(algorithm: str):
    """Display detailed help for an algorithm."""
    if algorithm not in ALGORITHMS:
        return

    info = ALGORITHMS[algorithm]

    st.markdown(f"""
    <div class="algorithm-card">
        <h4>📊 {info['name']}</h4>
        <p><em>{info['paper']}</em></p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(info['description'])

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**✅ Vantaggi:**")
        for pro in info['pros']:
            st.markdown(f"- {pro}")

    with col2:
        st.markdown("**⚠️ Svantaggi:**")
        for con in info['cons']:
            st.markdown(f"- {con}")

    st.markdown(f"**🎯 Best for:** {info['best_for']}")
    st.markdown(f"**⚙️ Complessità:** {info['complexity']}")


def show_model_help(model: str):
    """Display detailed help for a model."""
    if model not in MODELS:
        return

    info = MODELS[model]

    st.markdown(f"### 🧠 {info['name']} ({info['type']})")
    st.markdown(info['description'])

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Tipo Input", info['input_type'])
    with col2:
        st.metric("Complessità", info['complexity'])
    with col3:
        st.metric("Parametri", info['params_count'])

    st.markdown(f"**🎯 Use Case:** {info['use_case']}")


__all__ = [
    "ALGORITHMS",
    "MODELS",
    "HETEROGENEITY_TYPES",
    "PARTICIPATION_MODES",
    "BYZANTINE_METHODS",
    "CONTINUAL_METHODS",
    "MULTITASK_METHODS",
    "EHDS_TASKS",
    "show_algorithm_help",
    "show_model_help",
]

"""
Dataset Management Page for FL-EHDS Streamlit Dashboard.
Provides dataset browsing, statistics, previews, and FL distribution simulation.

Author: Fabio Liberti
"""

import streamlit as st
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import json

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


@dataclass
class DatasetInfo:
    """Information about a dataset."""
    name: str
    type: str  # "tabular" or "imaging"
    path: Optional[Path]
    description: str
    num_classes: int
    class_names: List[str]
    total_samples: int
    image_size: Optional[Tuple[int, int]] = None
    features: Optional[List[str]] = None
    clinical_relevance: str = ""
    ehds_category: str = ""


class DatasetManager:
    """Manages dataset discovery and statistics."""

    def __init__(self):
        self.data_dir = Path(__file__).parent.parent / "data"
        self.datasets: Dict[str, DatasetInfo] = {}
        self._discover_datasets()

    def _discover_datasets(self):
        """Discover available datasets."""
        # Add synthetic tabular dataset (always available)
        self.datasets["synthetic_healthcare"] = DatasetInfo(
            name="Synthetic Healthcare (Tabular)",
            type="tabular",
            path=None,
            description="Dati clinici sintetici generati per simulazioni FL",
            num_classes=2,
            class_names=["Low Risk", "High Risk"],
            total_samples=0,
            features=[
                "age", "bmi", "bp_systolic", "glucose", "cholesterol",
                "heart_rate", "resp_rate", "temperature", "oxygen_sat", "prev_conditions"
            ],
            clinical_relevance="Predizione rischio malattia cardiovascolare",
            ehds_category="Health Risk Assessment"
        )

        # Discover imaging datasets
        if self.data_dir.exists():
            self._discover_imaging_datasets()

    def _discover_imaging_datasets(self):
        """Discover imaging datasets in data directory.

        Handles two folder structures:
        1. Direct: dataset/class1/, dataset/class2/ (e.g., Retinopatia)
        2. Split: dataset/train/class1/, dataset/test/class1/ (e.g., chest_xray, ISIC)
        """
        imaging_configs = {
            "Retinopatia": {
                "description": "Diabetic Retinopathy - Stadi di retinopatia diabetica",
                "clinical_relevance": "Screening retinopatia diabetica per prevenzione cecita",
                "ehds_category": "Ophthalmology / Diabetes Management"
            },
            "Brain_Tumor": {
                "description": "Brain Tumor MRI - Classificazione tumori cerebrali",
                "clinical_relevance": "Diagnosi e classificazione tumori cerebrali da MRI",
                "ehds_category": "Oncology / Neurology"
            },
            "chest_xray": {
                "description": "Chest X-ray - Rilevamento polmonite",
                "clinical_relevance": "Screening polmonite da radiografie toraciche",
                "ehds_category": "Pulmonology / Radiology"
            },
            "Skin Cancer": {
                "description": "Skin Lesion - Classificazione lesioni cutanee",
                "clinical_relevance": "Screening melanoma e lesioni cutanee sospette",
                "ehds_category": "Dermatology / Oncology"
            },
            "ISIC": {
                "description": "ISIC Skin Lesions - Lesioni cutanee multiclasse",
                "clinical_relevance": "Classificazione dettagliata lesioni dermatologiche",
                "ehds_category": "Dermatology"
            },
            "Brain Tumor MRI": {
                "description": "Brain Tumor MRI Dataset - Tumori cerebrali",
                "clinical_relevance": "Classificazione tumori cerebrali da risonanza magnetica",
                "ehds_category": "Oncology / Neurology"
            }
        }

        # Common split folder names (case insensitive check)
        split_folders = {'train', 'test', 'val', 'valid', 'validation',
                         'training', 'testing', 'data'}

        for folder in self.data_dir.iterdir():
            if folder.is_dir() and not folder.name.startswith('.'):
                subdirs = [d for d in folder.iterdir() if d.is_dir() and not d.name.startswith('.')]
                if not subdirs:
                    continue

                # Check if subdirs are split folders or class folders
                subdir_names_lower = {d.name.lower() for d in subdirs}
                is_split_structure = bool(subdir_names_lower & split_folders)

                class_counts = {}  # class_name -> count
                sample_image_size = None
                base_path = folder  # For direct class structure

                if is_split_structure:
                    # Structure: dataset/train/class1, dataset/test/class2, etc.
                    for split_dir in subdirs:
                        if split_dir.name.lower() in split_folders:
                            class_dirs = [d for d in split_dir.iterdir()
                                         if d.is_dir() and not d.name.startswith('.')]
                            for class_dir in class_dirs:
                                # Skip nested split folders (e.g., data/train/, data/test/)
                                if class_dir.name.lower() in split_folders:
                                    continue

                                images = list(class_dir.glob("*.jpg")) + \
                                         list(class_dir.glob("*.jpeg")) + \
                                         list(class_dir.glob("*.png")) + \
                                         list(class_dir.glob("*.JPG")) + \
                                         list(class_dir.glob("*.JPEG")) + \
                                         list(class_dir.glob("*.PNG"))

                                # Only add if directory contains images
                                if images:
                                    class_counts[class_dir.name] = class_counts.get(class_dir.name, 0) + len(images)

                                    if sample_image_size is None and HAS_PIL:
                                        try:
                                            with Image.open(images[0]) as img:
                                                sample_image_size = img.size
                                        except Exception:
                                            pass

                    # Set base_path to train folder for get_sample_images
                    for split_dir in subdirs:
                        if split_dir.name.lower() in {'train', 'training'}:
                            base_path = split_dir
                            break
                    else:
                        for split_dir in subdirs:
                            if split_dir.name.lower() in split_folders:
                                base_path = split_dir
                                break
                else:
                    # Structure: dataset/class1, dataset/class2, etc.
                    for subdir in subdirs:
                        images = list(subdir.glob("*.jpg")) + \
                                 list(subdir.glob("*.jpeg")) + \
                                 list(subdir.glob("*.png")) + \
                                 list(subdir.glob("*.JPG")) + \
                                 list(subdir.glob("*.JPEG")) + \
                                 list(subdir.glob("*.PNG"))
                        if images:
                            class_counts[subdir.name] = len(images)

                            if sample_image_size is None and HAS_PIL:
                                try:
                                    with Image.open(images[0]) as img:
                                        sample_image_size = img.size
                                except Exception:
                                    pass

                total_images = sum(class_counts.values())
                if total_images > 0:
                    class_names = sorted(class_counts.keys())
                    config = imaging_configs.get(folder.name, {})
                    self.datasets[folder.name] = DatasetInfo(
                        name=folder.name,
                        type="imaging",
                        path=base_path,  # Points to where classes are
                        description=config.get("description", f"Dataset immagini: {folder.name}"),
                        num_classes=len(class_names),
                        class_names=class_names,
                        total_samples=total_images,
                        image_size=sample_image_size,
                        clinical_relevance=config.get("clinical_relevance", ""),
                        ehds_category=config.get("ehds_category", "Clinical Imaging")
                    )

    def get_class_distribution(self, dataset_id: str) -> Dict[str, int]:
        """Get number of samples per class.

        The ds.path points to where class folders are (could be train/ subfolder).
        """
        if dataset_id not in self.datasets:
            return {}

        ds = self.datasets[dataset_id]
        if ds.type != "imaging" or not ds.path:
            return {}

        distribution = {}
        for subdir in sorted(ds.path.iterdir()):
            if subdir.is_dir() and not subdir.name.startswith('.'):
                images = list(subdir.glob("*.jpg")) + \
                         list(subdir.glob("*.jpeg")) + \
                         list(subdir.glob("*.png")) + \
                         list(subdir.glob("*.JPG")) + \
                         list(subdir.glob("*.JPEG")) + \
                         list(subdir.glob("*.PNG"))
                if images:  # Only include if has images
                    distribution[subdir.name] = len(images)
        return distribution

    def get_sample_images(self, dataset_id: str, per_class: int = 2) -> Dict[str, List[Path]]:
        """Get sample image paths organized by class.

        The ds.path points to where class folders are (train/ subfolder for split structure).
        """
        if dataset_id not in self.datasets:
            return {}

        ds = self.datasets[dataset_id]
        if ds.type != "imaging" or not ds.path:
            return {}

        samples = {}
        for class_dir in sorted(ds.path.iterdir()):
            if class_dir.is_dir() and not class_dir.name.startswith('.'):
                # Support all image extensions (case insensitive)
                images = list(class_dir.glob("*.jpg")) + \
                         list(class_dir.glob("*.jpeg")) + \
                         list(class_dir.glob("*.png")) + \
                         list(class_dir.glob("*.JPG")) + \
                         list(class_dir.glob("*.JPEG")) + \
                         list(class_dir.glob("*.PNG"))
                if images:
                    samples[class_dir.name] = images[:per_class]

        return samples

    def get_folder_size_mb(self, path: Path) -> float:
        """Get folder size in MB."""
        total_size = 0
        for file in path.rglob("*"):
            if file.is_file():
                total_size += file.stat().st_size
        return total_size / (1024 * 1024)


# Initialize manager (cached)
@st.cache_resource
def get_dataset_manager():
    """Get cached dataset manager."""
    return DatasetManager()


def render_dataset_tab():
    """Render the dataset management tab."""
    st.header("Gestione Dataset")

    manager = get_dataset_manager()

    # Sub-tabs for different functionalities
    sub_tabs = st.tabs([
        "Lista Dataset",
        "Statistiche",
        "Preview Immagini",
        "Distribuzione Classi",
        "Simulazione FL"
    ])

    with sub_tabs[0]:
        render_dataset_list(manager)

    with sub_tabs[1]:
        render_dataset_statistics(manager)

    with sub_tabs[2]:
        render_image_preview(manager)

    with sub_tabs[3]:
        render_class_distribution(manager)

    with sub_tabs[4]:
        render_fl_simulation(manager)


def render_dataset_list(manager: DatasetManager):
    """Render dataset list."""
    st.subheader("Dataset Disponibili")

    # Tabular datasets
    st.markdown("### Dataset Tabulari")
    tabular = [d for d in manager.datasets.values() if d.type == "tabular"]

    if tabular:
        for ds in tabular:
            with st.expander(f"{ds.name}", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Tipo:** Tabular")
                    st.markdown(f"**Classi:** {ds.num_classes} ({', '.join(ds.class_names)})")
                    st.markdown(f"**Features:** {len(ds.features) if ds.features else 0}")
                with col2:
                    st.markdown(f"**Descrizione:** {ds.description}")
                    st.markdown(f"**Categoria EHDS:** {ds.ehds_category}")

                if ds.features:
                    st.markdown("**Lista Features:**")
                    st.code(", ".join(ds.features))
    else:
        st.info("Nessun dataset tabolare disponibile")

    st.markdown("---")

    # Imaging datasets
    st.markdown("### Dataset Imaging Clinico")
    imaging = [d for d in manager.datasets.values() if d.type == "imaging"]
    imaging.sort(key=lambda x: x.total_samples, reverse=True)

    if imaging:
        # Summary metrics
        total_images = sum(d.total_samples for d in imaging)
        total_classes = sum(d.num_classes for d in imaging)

        col1, col2, col3 = st.columns(3)
        col1.metric("Dataset Totali", len(imaging))
        col2.metric("Immagini Totali", f"{total_images:,}")
        col3.metric("Classi Totali", total_classes)

        st.markdown("")

        # Dataset cards
        for ds in imaging:
            with st.expander(f"{ds.name} - {ds.total_samples:,} immagini", expanded=False):
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown(f"**Tipo:** Imaging")
                    st.markdown(f"**Classi:** {ds.num_classes}")
                    if ds.image_size:
                        st.markdown(f"**Dimensione img:** {ds.image_size[0]}x{ds.image_size[1]}")

                with col2:
                    st.markdown(f"**Campioni:** {ds.total_samples:,}")
                    if ds.path:
                        size_mb = manager.get_folder_size_mb(ds.path)
                        st.markdown(f"**Spazio su disco:** {size_mb:.1f} MB")

                with col3:
                    st.markdown(f"**Categoria EHDS:** {ds.ehds_category}")
                    st.markdown(f"**Rilevanza:** {ds.clinical_relevance}")

                # Class names
                st.markdown("**Classi:**")
                st.code(", ".join(ds.class_names))
    else:
        st.warning(f"Nessun dataset imaging trovato in `{manager.data_dir}`")
        st.info("Aggiungi dataset nella cartella 'data/' con sottocartelle per classe")


def render_dataset_statistics(manager: DatasetManager):
    """Render detailed dataset statistics."""
    st.subheader("Statistiche Dettagliate")

    # Dataset selection
    dataset_options = list(manager.datasets.keys())
    if not dataset_options:
        st.warning("Nessun dataset disponibile")
        return

    selected = st.selectbox(
        "Seleziona dataset",
        dataset_options,
        format_func=lambda x: manager.datasets[x].name
    )

    if selected:
        ds = manager.datasets[selected]

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Informazioni Generali")
            st.markdown(f"**Nome:** {ds.name}")
            st.markdown(f"**Tipo:** {ds.type.capitalize()}")
            st.markdown(f"**Descrizione:** {ds.description}")
            st.markdown(f"**Numero classi:** {ds.num_classes}")
            st.markdown(f"**Campioni totali:** {ds.total_samples:,}")

            if ds.image_size:
                st.markdown(f"**Dimensione immagini:** {ds.image_size[0]}x{ds.image_size[1]} pixel")

        with col2:
            st.markdown("### Rilevanza Clinica")
            st.markdown(f"**Categoria EHDS:** {ds.ehds_category}")
            st.markdown(f"**Rilevanza:** {ds.clinical_relevance}")

            if ds.features:
                st.markdown("### Features")
                for i, feat in enumerate(ds.features, 1):
                    st.markdown(f"{i}. {feat}")

        # Class distribution for imaging datasets
        if ds.type == "imaging" and ds.path:
            st.markdown("---")
            st.markdown("### Distribuzione Classi")

            distribution = manager.get_class_distribution(selected)
            if distribution:
                total = sum(distribution.values())
                counts = list(distribution.values())

                # Create dataframe
                import pandas as pd
                df = pd.DataFrame({
                    "Classe": list(distribution.keys()),
                    "Campioni": counts,
                    "Percentuale": [c / total * 100 for c in counts]
                })
                st.dataframe(df, use_container_width=True)

                # Imbalance analysis
                max_count = max(counts)
                min_count = min(counts)
                imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')

                if imbalance_ratio > 5:
                    st.error(f"Sbilanciamento elevato (ratio: {imbalance_ratio:.1f}x) - Consigliato: weighted loss, oversampling")
                elif imbalance_ratio > 2:
                    st.warning(f"Sbilanciamento moderato (ratio: {imbalance_ratio:.1f}x)")
                else:
                    st.success(f"Dataset bilanciato (ratio: {imbalance_ratio:.1f}x)")


def render_image_preview(manager: DatasetManager):
    """Render image preview section."""
    st.subheader("Preview Immagini")

    if not HAS_PIL:
        st.error("PIL non disponibile - impossibile visualizzare immagini")
        return

    # Filter to imaging datasets only
    imaging_datasets = {k: v for k, v in manager.datasets.items() if v.type == "imaging"}

    if not imaging_datasets:
        st.warning("Nessun dataset imaging disponibile")
        return

    selected = st.selectbox(
        "Seleziona dataset",
        list(imaging_datasets.keys()),
        format_func=lambda x: imaging_datasets[x].name,
        key="preview_select"
    )

    if selected:
        ds = imaging_datasets[selected]
        samples_per_class = st.slider("Immagini per classe", 1, 5, 2)

        samples = manager.get_sample_images(selected, per_class=samples_per_class)

        if samples:
            for class_name, image_paths in samples.items():
                st.markdown(f"### Classe: {class_name}")

                cols = st.columns(len(image_paths))
                for idx, (col, img_path) in enumerate(zip(cols, image_paths)):
                    with col:
                        try:
                            img = Image.open(img_path)
                            st.image(img, caption=img_path.name[:20], use_container_width=True)
                        except Exception as e:
                            st.error(f"Errore: {e}")

                st.markdown("---")
        else:
            st.info("Nessuna immagine trovata")


def render_class_distribution(manager: DatasetManager):
    """Render class distribution charts."""
    st.subheader("Grafici Distribuzione Classi")

    if not HAS_MATPLOTLIB:
        st.error("Matplotlib non disponibile")
        return

    # Filter to imaging datasets
    imaging_datasets = {k: v for k, v in manager.datasets.items() if v.type == "imaging"}

    if not imaging_datasets:
        st.warning("Nessun dataset imaging disponibile")
        return

    selected = st.selectbox(
        "Seleziona dataset",
        list(imaging_datasets.keys()),
        format_func=lambda x: imaging_datasets[x].name,
        key="dist_select"
    )

    if selected:
        ds = imaging_datasets[selected]
        distribution = manager.get_class_distribution(selected)

        if distribution:
            classes = list(distribution.keys())
            counts = list(distribution.values())
            total = sum(counts)

            col1, col2 = st.columns(2)

            with col1:
                # Bar chart
                fig, ax = plt.subplots(figsize=(10, 6))
                colors = plt.cm.viridis(np.linspace(0, 0.8, len(classes)))
                bars = ax.bar(range(len(classes)), counts, color=colors)
                ax.set_xticks(range(len(classes)))
                ax.set_xticklabels(classes, rotation=45, ha='right', fontsize=9)
                ax.set_ylabel('Numero campioni')
                ax.set_title(f'{ds.name} - Distribuzione per classe')
                ax.grid(True, alpha=0.3, axis='y')

                # Add labels
                for bar, count in zip(bars, counts):
                    pct = count / total * 100
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                            f'{count:,}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=8)

                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

            with col2:
                # Pie chart
                fig, ax = plt.subplots(figsize=(8, 8))
                colors = plt.cm.viridis(np.linspace(0, 0.8, len(classes)))
                ax.pie(counts, labels=classes, autopct='%1.1f%%', colors=colors, startangle=90)
                ax.set_title('Proporzione classi')
                st.pyplot(fig)
                plt.close()

            # Summary statistics
            st.markdown("### Statistiche")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Classi", len(classes))
            col2.metric("Totale", f"{total:,}")
            col3.metric("Media/classe", f"{total // len(classes):,}")
            col4.metric("Range", f"{min(counts):,} - {max(counts):,}")


def render_fl_simulation(manager: DatasetManager):
    """Render FL distribution simulation."""
    st.subheader("Simulazione Distribuzione FL")

    # Dataset selection
    dataset_options = list(manager.datasets.keys())
    selected = st.selectbox(
        "Seleziona dataset",
        dataset_options,
        format_func=lambda x: manager.datasets[x].name,
        key="fl_sim_select"
    )

    if not selected:
        return

    ds = manager.datasets[selected]

    # Configuration
    col1, col2, col3 = st.columns(3)

    with col1:
        num_clients = st.slider("Numero client/ospedali", 2, 10, 5)

    with col2:
        is_iid = st.checkbox("Distribuzione IID", value=False)

    with col3:
        if not is_iid:
            alpha = st.slider("Parametro Dirichlet (alpha)", 0.1, 2.0, 0.5, 0.1)
        else:
            alpha = 1.0

    # Run simulation button
    if st.button("Esegui Simulazione", type="primary"):
        # Get class distribution
        if ds.type == "imaging" and ds.path:
            distribution = manager.get_class_distribution(selected)
            class_names = list(distribution.keys())
            class_counts = list(distribution.values())
        else:
            class_names = ds.class_names
            class_counts = [500] * len(class_names)  # Simulated

        num_classes = len(class_names)
        total_samples = sum(class_counts)

        np.random.seed(42)

        # Simulate distribution
        if is_iid:
            samples_per_client = total_samples // num_clients
            client_distributions = {
                i: {cn: cc // num_clients for cn, cc in zip(class_names, class_counts)}
                for i in range(num_clients)
            }
        else:
            proportions = np.random.dirichlet([alpha] * num_clients, num_classes)
            client_distributions = {i: {} for i in range(num_clients)}

            for class_idx, (class_name, class_count) in enumerate(zip(class_names, class_counts)):
                for client_id in range(num_clients):
                    samples = int(proportions[class_idx, client_id] * class_count)
                    client_distributions[client_id][class_name] = samples

        # Calculate client totals
        client_samples = {
            i: sum(client_distributions[i].values())
            for i in range(num_clients)
        }

        # Display results
        st.markdown("---")
        st.markdown("### Risultati Simulazione")

        col1, col2, col3 = st.columns(3)
        col1.metric("Dataset", ds.name)
        col2.metric("Distribuzione", "IID" if is_iid else f"Non-IID (alpha={alpha})")
        col3.metric("Client", num_clients)

        # Create dataframe
        import pandas as pd
        data = []
        for client_id in range(num_clients):
            row = {"Client": f"Client {client_id}"}
            for cn in class_names:
                row[cn] = client_distributions[client_id].get(cn, 0)
            row["Totale"] = client_samples[client_id]
            row["%"] = f"{client_samples[client_id] / total_samples * 100:.1f}%"
            data.append(row)

        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True)

        # Visualization
        if HAS_MATPLOTLIB:
            col1, col2 = st.columns(2)

            with col1:
                # Stacked bar chart
                fig, ax = plt.subplots(figsize=(10, 6))
                x = np.arange(num_clients)
                bottom = np.zeros(num_clients)

                colors = plt.cm.tab10(np.linspace(0, 1, len(class_names)))

                for class_idx, class_name in enumerate(class_names):
                    heights = [client_distributions[i].get(class_name, 0) for i in range(num_clients)]
                    ax.bar(x, heights, bottom=bottom, label=class_name[:15], color=colors[class_idx])
                    bottom += heights

                ax.set_xlabel('Client ID')
                ax.set_ylabel('Numero campioni')
                ax.set_title(f'Distribuzione dati per client\n{"IID" if is_iid else f"Non-IID (alpha={alpha})"}')
                ax.set_xticks(x)
                ax.set_xticklabels([f'C{i}' for i in range(num_clients)])
                ax.legend(loc='upper right', fontsize=7, ncol=2)
                ax.grid(True, alpha=0.3, axis='y')

                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

            with col2:
                # Heatmap
                matrix = np.zeros((num_clients, len(class_names)))
                for i in range(num_clients):
                    total = client_samples[i]
                    if total > 0:
                        for j, cn in enumerate(class_names):
                            matrix[i, j] = client_distributions[i].get(cn, 0) / total

                fig, ax = plt.subplots(figsize=(10, 6))
                im = ax.imshow(matrix, aspect='auto', cmap='YlOrRd')
                ax.set_xlabel('Classe')
                ax.set_ylabel('Client')
                ax.set_title('Proporzione classi per client')
                ax.set_xticks(range(len(class_names)))
                ax.set_xticklabels([cn[:8] for cn in class_names], rotation=45, ha='right', fontsize=8)
                ax.set_yticks(range(num_clients))
                ax.set_yticklabels([f'C{i}' for i in range(num_clients)])
                plt.colorbar(im, ax=ax, label='Proporzione')

                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

        # Heterogeneity analysis
        st.markdown("### Analisi Eterogeneita")

        client_totals = list(client_samples.values())
        sample_imbalance = max(client_totals) / min(client_totals) if min(client_totals) > 0 else float('inf')

        # Label skew per client
        label_skews = []
        for client_id in range(num_clients):
            client_dist = list(client_distributions[client_id].values())
            total = sum(client_dist)
            if total > 0:
                proportions_local = [c / total for c in client_dist]
                entropy = -sum(p * np.log(p + 1e-10) for p in proportions_local if p > 0)
                max_entropy = np.log(num_classes)
                label_skews.append(entropy / max_entropy if max_entropy > 0 else 0)

        avg_label_skew = np.mean(label_skews) if label_skews else 0

        col1, col2 = st.columns(2)
        col1.metric("Sbilanciamento campioni", f"{sample_imbalance:.2f}x")
        col2.metric("Uniformita label media", f"{avg_label_skew:.2%}")

        # Recommendations
        if is_iid:
            st.success("Distribuzione IID: ogni client ha campioni bilanciati")
        else:
            if alpha < 0.3:
                st.error("Alta eterogeneita: alpha basso causa forte skew - Consigliati: FedProx, SCAFFOLD, Per-FedAvg")
            elif alpha < 1.0:
                st.warning("Eterogeneita moderata - FedAvg dovrebbe funzionare, FedProx puo migliorare")
            else:
                st.success("Bassa eterogeneita: distribuzione quasi uniforme")


def render_dataset_summary_card():
    """Render a summary card for the dashboard home."""
    manager = get_dataset_manager()

    imaging = [d for d in manager.datasets.values() if d.type == "imaging"]
    total_images = sum(d.total_samples for d in imaging)

    st.markdown("""
    <div class="module-card">
        <h3>Dataset Clinici</h3>
        <p>{} dataset imaging | {:,} immagini totali</p>
    </div>
    """.format(len(imaging), total_images), unsafe_allow_html=True)


if __name__ == "__main__":
    # For testing standalone
    st.set_page_config(page_title="Dataset Manager", layout="wide")
    render_dataset_tab()

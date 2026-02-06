"""
Dataset management screen for FL-EHDS terminal interface.
Provides dataset browsing, statistics, previews, and FL distribution simulation.
"""

import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import json

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import questionary
    HAS_QUESTIONARY = True
except ImportError:
    HAS_QUESTIONARY = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from terminal.colors import (
    Colors, Style, print_header, print_section, print_subsection,
    print_error, print_info, print_warning, print_success, clear_screen
)
from terminal.menu import Menu, MenuItem


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
        self.data_dir = Path(__file__).parent.parent.parent / "data"
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
            total_samples=0,  # Generated on demand
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
        """Discover imaging datasets in data directory."""
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
                    # Merge classes from all split folders, prefer 'train' for path
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

                                    # Get sample image size from first image found
                                    if sample_image_size is None and HAS_PIL:
                                        try:
                                            with Image.open(images[0]) as img:
                                                sample_image_size = img.size
                                        except Exception:
                                            pass

                    # Set base_path to train folder if exists (for get_sample_images)
                    for split_dir in subdirs:
                        if split_dir.name.lower() in {'train', 'training'}:
                            base_path = split_dir
                            break
                    else:
                        # Use first split folder
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
                        if images:  # Only add if has images
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
                    # Store original folder for reference
                    self.datasets[folder.name].root_path = folder

    def get_dataset_statistics(self, dataset_id: str) -> Dict[str, Any]:
        """Get detailed statistics for a dataset."""
        if dataset_id not in self.datasets:
            return {}

        ds = self.datasets[dataset_id]
        stats = {
            "name": ds.name,
            "type": ds.type,
            "description": ds.description,
            "num_classes": ds.num_classes,
            "class_names": ds.class_names,
            "total_samples": ds.total_samples,
            "clinical_relevance": ds.clinical_relevance,
            "ehds_category": ds.ehds_category,
        }

        if ds.type == "imaging" and ds.path:
            stats["image_size"] = ds.image_size
            stats["class_distribution"] = self._get_class_distribution(ds.path)
            stats["disk_size_mb"] = self._get_folder_size_mb(ds.path)
        elif ds.type == "tabular":
            stats["features"] = ds.features
            stats["feature_count"] = len(ds.features) if ds.features else 0

        return stats

    def _get_class_distribution(self, path: Path) -> Dict[str, int]:
        """Get number of samples per class.

        The path points to where class folders are (could be train/ subfolder).
        """
        distribution = {}
        for subdir in sorted(path.iterdir()):
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

    def _get_folder_size_mb(self, path: Path) -> float:
        """Get folder size in MB."""
        total_size = 0
        for file in path.rglob("*"):
            if file.is_file():
                total_size += file.stat().st_size
        return total_size / (1024 * 1024)

    def get_sample_images(self, dataset_id: str, num_samples: int = 5) -> List[Tuple[Path, str]]:
        """Get sample image paths with their class names.

        The ds.path points to where class folders are (train/ subfolder for split structure).
        """
        if dataset_id not in self.datasets:
            return []

        ds = self.datasets[dataset_id]
        if ds.type != "imaging" or not ds.path:
            return []

        samples = []
        samples_per_class = max(1, num_samples // ds.num_classes + 1)

        for class_dir in sorted(ds.path.iterdir()):
            if class_dir.is_dir() and not class_dir.name.startswith('.'):
                # Support all image extensions (case insensitive)
                images = list(class_dir.glob("*.jpg")) + \
                         list(class_dir.glob("*.jpeg")) + \
                         list(class_dir.glob("*.png")) + \
                         list(class_dir.glob("*.JPG")) + \
                         list(class_dir.glob("*.JPEG")) + \
                         list(class_dir.glob("*.PNG"))
                for img_path in images[:samples_per_class]:
                    samples.append((img_path, class_dir.name))
                    if len(samples) >= num_samples:
                        break
            if len(samples) >= num_samples:
                break

        return samples[:num_samples]


class DatasetScreen:
    """Dataset management screen."""

    def __init__(self):
        self.manager = DatasetManager()
        self.selected_dataset: Optional[str] = None

    def run(self):
        """Run the dataset management screen."""
        while True:
            clear_screen()
            print_section("GESTIONE DATASET")

            menu = Menu("Seleziona azione", [
                MenuItem("1", "Lista Dataset Disponibili", self._list_datasets),
                MenuItem("2", "Statistiche Dettagliate", self._show_statistics),
                MenuItem("3", "Preview Immagini", self._preview_images, enabled=HAS_PIL and HAS_MATPLOTLIB),
                MenuItem("4", "Grafico Distribuzione Classi", self._plot_distribution, enabled=HAS_MATPLOTLIB),
                MenuItem("5", "Simula Distribuzione FL", self._simulate_fl_distribution, enabled=HAS_NUMPY),
                MenuItem("6", "Seleziona Dataset per Training", self._select_dataset),
                MenuItem("7", "Esporta Info Dataset", self._export_info),
                MenuItem("0", "Torna al Menu Principale", lambda: "back"),
            ])

            result = menu.display()
            if result is None or (result and result.handler and result.handler() == "back"):
                break
            elif result and result.handler:
                try:
                    result.handler()
                except Exception as e:
                    print_error(f"Errore: {e}")
                input(f"\n{Style.MUTED}Premi Enter per continuare...{Colors.RESET}")

    def _list_datasets(self):
        """List all available datasets."""
        clear_screen()
        print_section("DATASET DISPONIBILI")

        # Tabular datasets
        print_subsection("DATASET TABULARI")
        tabular = [d for d in self.manager.datasets.values() if d.type == "tabular"]
        if tabular:
            for ds in tabular:
                self._print_dataset_summary(ds)
        else:
            print(f"  {Style.MUTED}Nessun dataset tabolare disponibile{Colors.RESET}")

        print()

        # Imaging datasets
        print_subsection("DATASET IMAGING CLINICO")
        imaging = [d for d in self.manager.datasets.values() if d.type == "imaging"]
        if imaging:
            # Sort by total samples descending
            imaging.sort(key=lambda x: x.total_samples, reverse=True)
            for ds in imaging:
                self._print_dataset_summary(ds)

            # Summary
            print()
            total_images = sum(d.total_samples for d in imaging)
            print(f"{Style.SUCCESS}Totale: {len(imaging)} dataset, {total_images:,} immagini{Colors.RESET}")
        else:
            print(f"  {Style.MUTED}Nessun dataset imaging trovato in {self.manager.data_dir}{Colors.RESET}")
            print(f"  {Style.MUTED}Aggiungi dataset nella cartella 'data/' con sottocartelle per classe{Colors.RESET}")

    def _print_dataset_summary(self, ds: DatasetInfo):
        """Print a dataset summary line."""
        if ds.type == "imaging":
            size_str = f"{ds.total_samples:,} immagini"
            if ds.image_size:
                size_str += f" ({ds.image_size[0]}x{ds.image_size[1]})"
        else:
            size_str = f"{len(ds.features) if ds.features else 0} features"

        print(f"\n  {Style.TITLE}{ds.name}{Colors.RESET}")
        print(f"    Tipo: {ds.type.capitalize()}")
        print(f"    Classi: {ds.num_classes} ({', '.join(ds.class_names[:4])}{'...' if len(ds.class_names) > 4 else ''})")
        print(f"    Dimensione: {size_str}")
        print(f"    {Style.MUTED}{ds.description}{Colors.RESET}")
        if ds.ehds_category:
            print(f"    EHDS: {ds.ehds_category}")

    def _show_statistics(self):
        """Show detailed statistics for a dataset."""
        clear_screen()
        print_section("STATISTICHE DATASET")

        # Select dataset
        dataset_id = self._select_dataset_dialog()
        if not dataset_id:
            return

        stats = self.manager.get_dataset_statistics(dataset_id)
        if not stats:
            print_error("Impossibile caricare statistiche")
            return

        print()
        print(f"{Style.TITLE}=== {stats['name']} ==={Colors.RESET}")
        print()

        # Basic info
        print_subsection("INFORMAZIONI GENERALI")
        print(f"  Tipo:              {stats['type'].capitalize()}")
        print(f"  Descrizione:       {stats['description']}")
        print(f"  Numero classi:     {stats['num_classes']}")
        print(f"  Campioni totali:   {stats['total_samples']:,}")

        if stats.get('image_size'):
            print(f"  Dimensione img:    {stats['image_size'][0]}x{stats['image_size'][1]} pixel")
        if stats.get('disk_size_mb'):
            print(f"  Spazio su disco:   {stats['disk_size_mb']:.1f} MB")

        # Clinical relevance
        if stats.get('clinical_relevance'):
            print()
            print_subsection("RILEVANZA CLINICA")
            print(f"  {stats['clinical_relevance']}")
            if stats.get('ehds_category'):
                print(f"  Categoria EHDS: {stats['ehds_category']}")

        # Class distribution
        if stats.get('class_distribution'):
            print()
            print_subsection("DISTRIBUZIONE CLASSI")
            total = sum(stats['class_distribution'].values())
            for class_name, count in stats['class_distribution'].items():
                pct = count / total * 100
                bar_len = int(pct / 2)
                bar = "#" * bar_len
                print(f"  {class_name:<20} {count:>6} ({pct:>5.1f}%) {Style.SUCCESS}{bar}{Colors.RESET}")

            # Class imbalance analysis
            counts = list(stats['class_distribution'].values())
            if counts:
                max_count = max(counts)
                min_count = min(counts)
                imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
                print()
                if imbalance_ratio > 5:
                    print(f"  {Style.WARNING}Sbilanciamento elevato (ratio: {imbalance_ratio:.1f}x){Colors.RESET}")
                    print(f"  {Style.MUTED}Consigliato: weighted loss, oversampling, o algoritmi robusti{Colors.RESET}")
                elif imbalance_ratio > 2:
                    print(f"  {Style.INFO}Sbilanciamento moderato (ratio: {imbalance_ratio:.1f}x){Colors.RESET}")
                else:
                    print(f"  {Style.SUCCESS}Dataset bilanciato (ratio: {imbalance_ratio:.1f}x){Colors.RESET}")

        # Features (for tabular)
        if stats.get('features'):
            print()
            print_subsection("FEATURES")
            for i, feat in enumerate(stats['features'], 1):
                print(f"  {i:2}. {feat}")

    def _preview_images(self):
        """Preview sample images from a dataset."""
        if not HAS_PIL or not HAS_MATPLOTLIB:
            print_error("Richiede PIL e matplotlib")
            return

        clear_screen()
        print_section("PREVIEW IMMAGINI")

        # Select imaging dataset
        imaging_ids = [k for k, v in self.manager.datasets.items() if v.type == "imaging"]
        if not imaging_ids:
            print_error("Nessun dataset imaging disponibile")
            return

        dataset_id = self._select_dataset_dialog(filter_type="imaging")
        if not dataset_id:
            return

        ds = self.manager.datasets[dataset_id]
        samples = self.manager.get_sample_images(dataset_id, num_samples=12)

        if not samples:
            print_error("Nessuna immagine trovata")
            return

        print()
        print_info(f"Generazione preview per {ds.name}...")

        # Create figure
        n_cols = min(4, len(samples))
        n_rows = (len(samples) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = [[axes]]
        elif n_rows == 1:
            axes = [axes]
        elif n_cols == 1:
            axes = [[ax] for ax in axes]

        for idx, (img_path, class_name) in enumerate(samples):
            row = idx // n_cols
            col = idx % n_cols
            try:
                img = Image.open(img_path)
                axes[row][col].imshow(img)
                axes[row][col].set_title(f"{class_name}\n{img_path.name[:20]}...", fontsize=8)
                axes[row][col].axis('off')
            except Exception as e:
                axes[row][col].text(0.5, 0.5, f"Errore:\n{e}", ha='center', va='center')
                axes[row][col].axis('off')

        # Hide empty subplots
        for idx in range(len(samples), n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row][col].axis('off')

        plt.suptitle(f"{ds.name} - Sample Images", fontsize=12, fontweight='bold')
        plt.tight_layout()

        # Save to results folder
        output_dir = Path(__file__).parent.parent.parent / "results" / "dataset_previews"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"preview_{dataset_id.replace(' ', '_')}.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()

        print_success(f"Preview salvata: {output_file}")

    def _plot_distribution(self):
        """Plot class distribution for a dataset."""
        if not HAS_MATPLOTLIB:
            print_error("Richiede matplotlib")
            return

        clear_screen()
        print_section("GRAFICO DISTRIBUZIONE CLASSI")

        dataset_id = self._select_dataset_dialog(filter_type="imaging")
        if not dataset_id:
            return

        stats = self.manager.get_dataset_statistics(dataset_id)
        if not stats.get('class_distribution'):
            print_error("Distribuzione classi non disponibile")
            return

        print()
        print_info(f"Generazione grafico per {stats['name']}...")

        dist = stats['class_distribution']
        classes = list(dist.keys())
        counts = list(dist.values())
        total = sum(counts)
        percentages = [c / total * 100 for c in counts]

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Bar chart
        colors = plt.cm.viridis(np.linspace(0, 0.8, len(classes)))
        bars = ax1.bar(range(len(classes)), counts, color=colors)
        ax1.set_xticks(range(len(classes)))
        ax1.set_xticklabels(classes, rotation=45, ha='right', fontsize=9)
        ax1.set_ylabel('Numero campioni')
        ax1.set_title(f'{stats["name"]}\nDistribuzione per classe')
        ax1.grid(True, alpha=0.3, axis='y')

        # Add count labels on bars
        for bar, count, pct in zip(bars, counts, percentages):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                    f'{count:,}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=8)

        # Pie chart
        ax2.pie(counts, labels=classes, autopct='%1.1f%%', colors=colors, startangle=90)
        ax2.set_title('Proporzione classi')

        plt.tight_layout()

        # Save
        output_dir = Path(__file__).parent.parent.parent / "results" / "dataset_previews"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"distribution_{dataset_id.replace(' ', '_')}.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()

        print_success(f"Grafico salvato: {output_file}")

        # Print summary
        print()
        print_subsection("RIEPILOGO")
        print(f"  Classi: {len(classes)}")
        print(f"  Totale campioni: {total:,}")
        print(f"  Media per classe: {total // len(classes):,}")
        print(f"  Range: {min(counts):,} - {max(counts):,}")

    def _simulate_fl_distribution(self):
        """Simulate FL data distribution across clients."""
        if not HAS_NUMPY:
            print_error("Richiede numpy")
            return

        clear_screen()
        print_section("SIMULAZIONE DISTRIBUZIONE FL")

        dataset_id = self._select_dataset_dialog()
        if not dataset_id:
            return

        stats = self.manager.get_dataset_statistics(dataset_id)

        # Get simulation parameters
        print()
        print_info("Configurazione simulazione:")
        print()

        try:
            num_clients = int(input(f"  Numero client/ospedali [5]: ").strip() or "5")
            is_iid = input(f"  Distribuzione IID? (s/n) [n]: ").strip().lower() in ('s', 'y', 'si', 'yes')
            if not is_iid:
                alpha = float(input(f"  Parametro Dirichlet alpha [0.5]: ").strip() or "0.5")
            else:
                alpha = 1.0
        except ValueError:
            print_error("Input non valido")
            return

        print()
        print_info("Simulazione in corso...")

        # Get class distribution
        if stats.get('class_distribution'):
            class_counts = list(stats['class_distribution'].values())
            class_names = list(stats['class_distribution'].keys())
        else:
            # For synthetic dataset
            class_counts = [500, 500]  # Simulated balanced binary
            class_names = stats.get('class_names', ['Class 0', 'Class 1'])

        num_classes = len(class_counts)
        total_samples = sum(class_counts)

        # Simulate distribution
        np.random.seed(42)

        if is_iid:
            # IID: equal random distribution
            samples_per_client = total_samples // num_clients
            client_samples = {i: samples_per_client for i in range(num_clients)}
            client_distributions = {
                i: {cn: cc // num_clients for cn, cc in zip(class_names, class_counts)}
                for i in range(num_clients)
            }
        else:
            # Non-IID: Dirichlet distribution
            proportions = np.random.dirichlet([alpha] * num_clients, num_classes)

            client_distributions = {i: {} for i in range(num_clients)}
            client_samples = {i: 0 for i in range(num_clients)}

            for class_idx, (class_name, class_count) in enumerate(zip(class_names, class_counts)):
                for client_id in range(num_clients):
                    samples = int(proportions[class_idx, client_id] * class_count)
                    client_distributions[client_id][class_name] = samples
                    client_samples[client_id] += samples

        # Display results
        print()
        print_subsection("DISTRIBUZIONE SIMULATA")

        dist_type = "IID" if is_iid else f"Non-IID (alpha={alpha})"
        print(f"\n  {Style.TITLE}Dataset:{Colors.RESET} {stats['name']}")
        print(f"  {Style.TITLE}Tipo distribuzione:{Colors.RESET} {dist_type}")
        print(f"  {Style.TITLE}Client:{Colors.RESET} {num_clients}")
        print(f"  {Style.TITLE}Campioni totali:{Colors.RESET} {total_samples:,}")
        print()

        # Table header
        header = f"{'Client':<10}"
        for cn in class_names[:6]:  # Limit to 6 classes for display
            header += f" {cn[:8]:<10}"
        header += f" {'Totale':<10} {'%':<6}"
        print(f"  {Style.TITLE}{header}{Colors.RESET}")
        print("  " + "-" * len(header))

        # Client rows
        for client_id in range(num_clients):
            row = f"  Client {client_id:<3}"
            for cn in class_names[:6]:
                count = client_distributions[client_id].get(cn, 0)
                row += f" {count:<10}"
            total = client_samples[client_id]
            pct = total / total_samples * 100
            row += f" {total:<10} {pct:.1f}%"
            print(row)

        # Analysis
        print()
        print_subsection("ANALISI ETEROGENEITA")

        # Calculate heterogeneity metrics
        client_totals = list(client_samples.values())
        sample_imbalance = max(client_totals) / min(client_totals) if min(client_totals) > 0 else float('inf')

        # Label skew per client
        label_skews = []
        for client_id in range(num_clients):
            client_dist = list(client_distributions[client_id].values())
            if sum(client_dist) > 0:
                proportions = [c / sum(client_dist) for c in client_dist]
                # Entropy as skew measure
                entropy = -sum(p * np.log(p + 1e-10) for p in proportions if p > 0)
                max_entropy = np.log(num_classes)
                label_skews.append(entropy / max_entropy if max_entropy > 0 else 0)

        avg_label_skew = np.mean(label_skews) if label_skews else 0

        print(f"  Sbilanciamento campioni: {sample_imbalance:.2f}x")
        print(f"  Uniformita label media:  {avg_label_skew:.2%}")

        if is_iid:
            print(f"\n  {Style.SUCCESS}Distribuzione IID: ogni client ha campioni bilanciati{Colors.RESET}")
        else:
            if alpha < 0.3:
                print(f"\n  {Style.WARNING}Alta eterogeneita: alpha basso causa forte skew{Colors.RESET}")
                print(f"  {Style.MUTED}Consigliati: FedProx, SCAFFOLD, Per-FedAvg{Colors.RESET}")
            elif alpha < 1.0:
                print(f"\n  {Style.INFO}Eterogeneita moderata{Colors.RESET}")
                print(f"  {Style.MUTED}FedAvg dovrebbe funzionare, FedProx puo migliorare{Colors.RESET}")
            else:
                print(f"\n  {Style.SUCCESS}Bassa eterogeneita: distribuzione quasi uniforme{Colors.RESET}")

        # Generate plot if matplotlib available
        if HAS_MATPLOTLIB:
            self._plot_fl_distribution(
                client_distributions, client_samples, class_names,
                stats['name'], is_iid, alpha, num_clients
            )

    def _plot_fl_distribution(self, client_dist, client_samples, class_names,
                               dataset_name, is_iid, alpha, num_clients):
        """Generate FL distribution visualization."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Stacked bar chart
        ax1 = axes[0]
        x = np.arange(num_clients)
        bottom = np.zeros(num_clients)

        colors = plt.cm.tab10(np.linspace(0, 1, len(class_names)))

        for class_idx, class_name in enumerate(class_names):
            heights = [client_dist[i].get(class_name, 0) for i in range(num_clients)]
            ax1.bar(x, heights, bottom=bottom, label=class_name[:15], color=colors[class_idx])
            bottom += heights

        ax1.set_xlabel('Client ID')
        ax1.set_ylabel('Numero campioni')
        ax1.set_title(f'Distribuzione dati per client\n{"IID" if is_iid else f"Non-IID (alpha={alpha})"}')
        ax1.set_xticks(x)
        ax1.set_xticklabels([f'C{i}' for i in range(num_clients)])
        ax1.legend(loc='upper right', fontsize=7, ncol=2)
        ax1.grid(True, alpha=0.3, axis='y')

        # Heatmap of class proportions
        ax2 = axes[1]
        matrix = np.zeros((num_clients, len(class_names)))
        for i in range(num_clients):
            total = client_samples[i]
            if total > 0:
                for j, cn in enumerate(class_names):
                    matrix[i, j] = client_dist[i].get(cn, 0) / total

        im = ax2.imshow(matrix, aspect='auto', cmap='YlOrRd')
        ax2.set_xlabel('Classe')
        ax2.set_ylabel('Client')
        ax2.set_title('Proporzione classi per client\n(piu scuro = piu campioni)')
        ax2.set_xticks(range(len(class_names)))
        ax2.set_xticklabels([cn[:8] for cn in class_names], rotation=45, ha='right', fontsize=8)
        ax2.set_yticks(range(num_clients))
        ax2.set_yticklabels([f'C{i}' for i in range(num_clients)])
        plt.colorbar(im, ax=ax2, label='Proporzione')

        plt.suptitle(f'{dataset_name} - Simulazione FL', fontsize=12, fontweight='bold')
        plt.tight_layout()

        # Save
        output_dir = Path(__file__).parent.parent.parent / "results" / "dataset_previews"
        output_dir.mkdir(parents=True, exist_ok=True)
        dist_str = "IID" if is_iid else f"NonIID_a{alpha}"
        output_file = output_dir / f"fl_distribution_{dataset_name.replace(' ', '_')}_{dist_str}.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()

        print()
        print_success(f"Grafico salvato: {output_file}")

    def _select_dataset(self):
        """Select a dataset for training."""
        clear_screen()
        print_section("SELEZIONA DATASET PER TRAINING")

        dataset_id = self._select_dataset_dialog()
        if not dataset_id:
            return

        self.selected_dataset = dataset_id
        ds = self.manager.datasets[dataset_id]

        print()
        print_success(f"Dataset selezionato: {ds.name}")
        print()
        print(f"  Tipo:    {ds.type}")
        print(f"  Classi:  {ds.num_classes}")
        print(f"  Samples: {ds.total_samples:,}")

        if ds.type == "imaging":
            print()
            print_info("Per usare questo dataset nel training:")
            print(f"  from terminal.fl_trainer import ImageFederatedTrainer")
            print()
            print(f"  trainer = ImageFederatedTrainer(")
            print(f"      data_dir='{ds.path}',")
            print(f"      num_clients=5,")
            print(f"      algorithm='FedAvg',")
            print(f"      is_iid=False,")
            print(f"  )")
        else:
            print()
            print_info("Per usare questo dataset nel training:")
            print(f"  from terminal.fl_trainer import FederatedTrainer")
            print()
            print(f"  trainer = FederatedTrainer(")
            print(f"      num_clients=5,")
            print(f"      samples_per_client=200,")
            print(f"      algorithm='FedAvg',")
            print(f"      is_iid=False,")
            print(f"  )")

    def _export_info(self):
        """Export dataset information to JSON."""
        clear_screen()
        print_section("ESPORTA INFORMAZIONI DATASET")

        export_data = {
            "framework": "FL-EHDS",
            "version": "1.0.0",
            "datasets": {}
        }

        for dataset_id, ds in self.manager.datasets.items():
            stats = self.manager.get_dataset_statistics(dataset_id)
            export_data["datasets"][dataset_id] = {
                "name": ds.name,
                "type": ds.type,
                "path": str(ds.path) if ds.path else None,
                "description": ds.description,
                "num_classes": ds.num_classes,
                "class_names": ds.class_names,
                "total_samples": ds.total_samples,
                "image_size": list(ds.image_size) if ds.image_size else None,
                "features": ds.features,
                "clinical_relevance": ds.clinical_relevance,
                "ehds_category": ds.ehds_category,
                "class_distribution": stats.get("class_distribution"),
                "disk_size_mb": stats.get("disk_size_mb"),
            }

        output_dir = Path(__file__).parent.parent.parent / "results"
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / "datasets_info.json"

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        print()
        print_success(f"Informazioni esportate: {output_file}")
        print()
        print(f"  Dataset totali: {len(export_data['datasets'])}")
        print(f"  Tabulari: {sum(1 for d in export_data['datasets'].values() if d['type'] == 'tabular')}")
        print(f"  Imaging: {sum(1 for d in export_data['datasets'].values() if d['type'] == 'imaging')}")

    def _select_dataset_dialog(self, filter_type: Optional[str] = None) -> Optional[str]:
        """Show dataset selection dialog."""
        datasets = self.manager.datasets

        if filter_type:
            datasets = {k: v for k, v in datasets.items() if v.type == filter_type}

        if not datasets:
            print_error(f"Nessun dataset {'di tipo ' + filter_type if filter_type else ''} disponibile")
            return None

        choices = []
        for ds_id, ds in datasets.items():
            if ds.type == "imaging":
                label = f"{ds.name} ({ds.total_samples:,} img, {ds.num_classes} classi)"
            else:
                label = f"{ds.name} ({len(ds.features) if ds.features else 0} features)"
            choices.append((ds_id, label))

        print()
        print("Dataset disponibili:")
        for i, (ds_id, label) in enumerate(choices, 1):
            print(f"  {i}. {label}")

        print()
        try:
            choice = input("Seleziona dataset (numero): ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(choices):
                return choices[idx][0]
        except (ValueError, IndexError):
            pass

        print_error("Selezione non valida")
        return None


def main():
    """Main entry point for testing."""
    screen = DatasetScreen()
    screen.run()


if __name__ == "__main__":
    main()

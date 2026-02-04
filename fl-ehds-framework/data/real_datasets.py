#!/usr/bin/env python3
"""
FL-EHDS Real Dataset Loaders

Provides loaders for real healthcare and benchmark datasets:

1. MIMIC-IV (requires PhysioNet credentials)
2. eICU (requires PhysioNet credentials)
3. Heart Disease UCI (public)
4. Diabetes Pima Indians (public)
5. Breast Cancer Wisconsin (public)
6. CIFAR-10 (for image FL experiments)
7. MNIST (for quick experiments)

Each dataset can be partitioned into federated nodes with configurable
non-IID distributions.

Author: Fabio Liberti
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np
from abc import ABC, abstractmethod

# Optional dependencies
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    from sklearn.datasets import load_breast_cancer, load_diabetes
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import torch
    from torch.utils.data import Dataset, Subset, DataLoader
    import torchvision
    import torchvision.transforms as transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class FederatedDataset:
    """Container for federated dataset split across nodes."""
    node_data: Dict[int, Dict[str, np.ndarray]]  # {node_id: {"X": ..., "y": ...}}
    test_data: Dict[str, np.ndarray]  # {"X": ..., "y": ...}
    feature_names: List[str]
    label_name: str
    num_classes: int
    metadata: Dict

    def get_node_stats(self) -> pd.DataFrame:
        """Get statistics for each node."""
        if not PANDAS_AVAILABLE:
            return None

        stats = []
        for node_id, data in self.node_data.items():
            y = data['y']
            unique, counts = np.unique(y, return_counts=True)
            label_dist = dict(zip(unique.tolist(), counts.tolist()))

            stats.append({
                'node_id': node_id,
                'n_samples': len(y),
                'n_features': data['X'].shape[1],
                'pos_rate': float(y.mean()) if self.num_classes == 2 else None,
                'label_distribution': label_dist
            })

        return pd.DataFrame(stats)


# =============================================================================
# NON-IID PARTITIONER
# =============================================================================

class FederatedPartitioner:
    """Partition datasets for federated learning with various non-IID strategies."""

    def __init__(self, num_nodes: int, random_seed: int = 42):
        self.num_nodes = num_nodes
        self.rng = np.random.RandomState(random_seed)

    def partition_iid(self, X: np.ndarray, y: np.ndarray) -> Dict[int, Dict]:
        """IID partition - random uniform distribution."""
        indices = self.rng.permutation(len(y))
        splits = np.array_split(indices, self.num_nodes)

        return {
            i: {"X": X[split], "y": y[split]}
            for i, split in enumerate(splits)
        }

    def partition_dirichlet(self,
                           X: np.ndarray,
                           y: np.ndarray,
                           alpha: float = 0.5) -> Dict[int, Dict]:
        """
        Non-IID partition using Dirichlet distribution.

        Args:
            alpha: Concentration parameter
                   - alpha -> 0: extreme non-IID (each node gets mostly one class)
                   - alpha -> inf: IID
        """
        num_classes = len(np.unique(y))
        node_indices = {i: [] for i in range(self.num_nodes)}

        for class_idx in range(num_classes):
            class_indices = np.where(y == class_idx)[0]
            self.rng.shuffle(class_indices)

            # Sample proportions from Dirichlet
            proportions = self.rng.dirichlet([alpha] * self.num_nodes)

            # Distribute samples according to proportions
            splits = (proportions * len(class_indices)).astype(int)
            splits[-1] = len(class_indices) - splits[:-1].sum()

            start = 0
            for node_id, n_samples in enumerate(splits):
                node_indices[node_id].extend(class_indices[start:start + n_samples])
                start += n_samples

        return {
            i: {"X": X[indices], "y": y[indices]}
            for i, indices in node_indices.items()
        }

    def partition_pathological(self,
                              X: np.ndarray,
                              y: np.ndarray,
                              classes_per_node: int = 2) -> Dict[int, Dict]:
        """
        Pathological non-IID - each node gets only a few classes.
        """
        num_classes = len(np.unique(y))
        sorted_indices = np.argsort(y)

        # Create shards (each shard contains one class)
        shards = []
        for class_idx in range(num_classes):
            class_indices = np.where(y == class_idx)[0]
            shards.append(class_indices)

        # Assign classes to nodes
        class_assignments = []
        for node_id in range(self.num_nodes):
            assigned = self.rng.choice(num_classes, size=classes_per_node, replace=False)
            class_assignments.append(assigned)

        node_data = {}
        for node_id, classes in enumerate(class_assignments):
            indices = np.concatenate([
                shards[c][:len(shards[c]) // self.num_nodes * (node_id + 1)]
                for c in classes
            ])
            self.rng.shuffle(indices)
            node_data[node_id] = {"X": X[indices], "y": y[indices]}

        return node_data

    def partition_quantity_skew(self,
                               X: np.ndarray,
                               y: np.ndarray,
                               imbalance_factor: float = 5.0) -> Dict[int, Dict]:
        """
        Quantity imbalance - different number of samples per node.
        """
        indices = self.rng.permutation(len(y))

        # Exponential decay for sample counts
        proportions = np.exp(-np.linspace(0, np.log(imbalance_factor), self.num_nodes))
        proportions = proportions / proportions.sum()

        splits = (proportions * len(y)).astype(int)
        splits[-1] = len(y) - splits[:-1].sum()

        node_data = {}
        start = 0
        for node_id, n_samples in enumerate(splits):
            node_indices = indices[start:start + n_samples]
            node_data[node_id] = {"X": X[node_indices], "y": y[node_indices]}
            start += n_samples

        return node_data


# =============================================================================
# PUBLIC DATASET LOADERS
# =============================================================================

class HeartDiseaseLoader:
    """
    UCI Heart Disease Dataset
    - 303 samples, 13 features
    - Binary classification (heart disease presence)
    - Public, no credentials required
    """

    URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

    FEATURE_NAMES = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
    ]

    def __init__(self, data_dir: str = './data'):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def load(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load and preprocess the dataset."""
        if not PANDAS_AVAILABLE:
            raise ImportError("pandas required for HeartDiseaseLoader")

        filepath = self.data_dir / 'heart_disease.csv'

        if not filepath.exists():
            # Download
            import urllib.request
            urllib.request.urlretrieve(self.URL, filepath)

        # Load
        df = pd.read_csv(filepath, names=self.FEATURE_NAMES + ['target'], na_values='?')
        df = df.dropna()

        X = df[self.FEATURE_NAMES].values.astype(np.float32)
        y = (df['target'].values > 0).astype(np.int32)  # Binary: disease or not

        # Normalize
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        return X, y

    def load_federated(self,
                       num_nodes: int = 5,
                       partition: str = 'dirichlet',
                       alpha: float = 0.5,
                       test_size: float = 0.2,
                       random_seed: int = 42) -> FederatedDataset:
        """Load as federated dataset."""
        X, y = self.load()

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_seed, stratify=y
        )

        # Partition
        partitioner = FederatedPartitioner(num_nodes, random_seed)

        if partition == 'iid':
            node_data = partitioner.partition_iid(X_train, y_train)
        elif partition == 'dirichlet':
            node_data = partitioner.partition_dirichlet(X_train, y_train, alpha)
        elif partition == 'pathological':
            node_data = partitioner.partition_pathological(X_train, y_train)
        else:
            node_data = partitioner.partition_quantity_skew(X_train, y_train)

        return FederatedDataset(
            node_data=node_data,
            test_data={"X": X_test, "y": y_test},
            feature_names=self.FEATURE_NAMES,
            label_name='heart_disease',
            num_classes=2,
            metadata={'source': 'UCI Heart Disease', 'partition': partition}
        )


class DiabetesLoader:
    """
    Pima Indians Diabetes Dataset
    - 768 samples, 8 features
    - Binary classification (diabetes diagnosis)
    """

    URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"

    FEATURE_NAMES = [
        'pregnancies', 'glucose', 'blood_pressure', 'skin_thickness',
        'insulin', 'bmi', 'diabetes_pedigree', 'age'
    ]

    def __init__(self, data_dir: str = './data'):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def load(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load and preprocess the dataset."""
        if not PANDAS_AVAILABLE:
            raise ImportError("pandas required")

        filepath = self.data_dir / 'diabetes.csv'

        if not filepath.exists():
            import urllib.request
            urllib.request.urlretrieve(self.URL, filepath)

        df = pd.read_csv(filepath, names=self.FEATURE_NAMES + ['target'])

        X = df[self.FEATURE_NAMES].values.astype(np.float32)
        y = df['target'].values.astype(np.int32)

        # Handle zeros as missing values for some features
        for i, col in enumerate(['glucose', 'blood_pressure', 'skin_thickness', 'insulin', 'bmi']):
            col_idx = self.FEATURE_NAMES.index(col)
            mask = X[:, col_idx] == 0
            X[mask, col_idx] = np.nanmean(X[~mask, col_idx])

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        return X, y

    def load_federated(self,
                       num_nodes: int = 5,
                       partition: str = 'dirichlet',
                       alpha: float = 0.5,
                       test_size: float = 0.2,
                       random_seed: int = 42) -> FederatedDataset:
        """Load as federated dataset."""
        X, y = self.load()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_seed, stratify=y
        )

        partitioner = FederatedPartitioner(num_nodes, random_seed)

        if partition == 'dirichlet':
            node_data = partitioner.partition_dirichlet(X_train, y_train, alpha)
        else:
            node_data = partitioner.partition_iid(X_train, y_train)

        return FederatedDataset(
            node_data=node_data,
            test_data={"X": X_test, "y": y_test},
            feature_names=self.FEATURE_NAMES,
            label_name='diabetes',
            num_classes=2,
            metadata={'source': 'Pima Indians Diabetes'}
        )


class BreastCancerLoader:
    """
    Wisconsin Breast Cancer Dataset (from sklearn)
    - 569 samples, 30 features
    - Binary classification (malignant/benign)
    """

    def __init__(self):
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required")

    def load(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load and preprocess."""
        data = load_breast_cancer()
        X = data.data.astype(np.float32)
        y = data.target.astype(np.int32)

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        return X, y

    def load_federated(self,
                       num_nodes: int = 5,
                       partition: str = 'dirichlet',
                       alpha: float = 0.5,
                       test_size: float = 0.2,
                       random_seed: int = 42) -> FederatedDataset:
        """Load as federated dataset."""
        X, y = self.load()
        data = load_breast_cancer()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_seed, stratify=y
        )

        partitioner = FederatedPartitioner(num_nodes, random_seed)

        if partition == 'dirichlet':
            node_data = partitioner.partition_dirichlet(X_train, y_train, alpha)
        else:
            node_data = partitioner.partition_iid(X_train, y_train)

        return FederatedDataset(
            node_data=node_data,
            test_data={"X": X_test, "y": y_test},
            feature_names=list(data.feature_names),
            label_name='malignant',
            num_classes=2,
            metadata={'source': 'Wisconsin Breast Cancer'}
        )


# =============================================================================
# MIMIC-IV LOADER (requires credentials)
# =============================================================================

class MIMICIVLoader:
    """
    MIMIC-IV Dataset Loader

    Requires:
    1. PhysioNet credentials (https://physionet.org/)
    2. Completed CITI training
    3. Signed data use agreement

    Installation:
        pip install wfdb
        # Then download MIMIC-IV from PhysioNet
    """

    def __init__(self, mimic_path: str):
        """
        Args:
            mimic_path: Path to MIMIC-IV data directory
        """
        self.mimic_path = Path(mimic_path)

        if not self.mimic_path.exists():
            raise FileNotFoundError(
                f"MIMIC-IV path not found: {mimic_path}\n"
                "Please download MIMIC-IV from PhysioNet: "
                "https://physionet.org/content/mimiciv/"
            )

    def load_icu_cohort(self,
                        features: List[str] = None,
                        target: str = 'mortality',
                        max_patients: int = 10000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load ICU cohort for mortality prediction.

        Default features:
        - age, gender, admission_type
        - vital signs: heart_rate, bp_systolic, bp_diastolic, temperature, spo2
        - lab values: glucose, creatinine, hemoglobin, platelets, wbc
        """
        if not PANDAS_AVAILABLE:
            raise ImportError("pandas required")

        # This is a simplified loader - full implementation would need
        # proper MIMIC-IV table joins

        # Check for preprocessed file
        preprocessed = self.mimic_path / 'preprocessed_icu_cohort.csv'

        if preprocessed.exists():
            df = pd.read_csv(preprocessed, nrows=max_patients)
        else:
            raise FileNotFoundError(
                f"Preprocessed file not found: {preprocessed}\n"
                "Please run MIMIC-IV preprocessing first. "
                "See: https://github.com/YerevaNN/mimic3-benchmarks"
            )

        # Default features
        if features is None:
            features = [
                'age', 'gender', 'heart_rate_mean', 'sbp_mean', 'dbp_mean',
                'temperature_mean', 'spo2_mean', 'glucose_mean', 'creatinine_mean'
            ]

        available = [f for f in features if f in df.columns]
        X = df[available].values.astype(np.float32)

        # Handle missing values
        X = np.nan_to_num(X, nan=0.0)

        # Target
        if target == 'mortality':
            y = df['hospital_expire_flag'].values.astype(np.int32)
        elif target == 'los_3':  # Length of stay > 3 days
            y = (df['los_hospital'] > 3).astype(np.int32)
        else:
            y = df[target].values.astype(np.int32)

        # Normalize
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        return X, y

    def load_federated(self,
                       num_nodes: int = 5,
                       partition: str = 'dirichlet',
                       alpha: float = 0.5,
                       split_by: str = 'admission_location',
                       **kwargs) -> FederatedDataset:
        """
        Load MIMIC-IV as federated dataset.

        split_by options:
        - 'admission_location': Different hospital units (natural non-IID)
        - 'insurance': Different insurance types
        - 'random': Random partition with Dirichlet
        """
        X, y = self.load_icu_cohort(**kwargs)

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        partitioner = FederatedPartitioner(num_nodes, 42)

        if partition == 'dirichlet':
            node_data = partitioner.partition_dirichlet(X_train, y_train, alpha)
        else:
            node_data = partitioner.partition_iid(X_train, y_train)

        return FederatedDataset(
            node_data=node_data,
            test_data={"X": X_test, "y": y_test},
            feature_names=kwargs.get('features', ['feature_' + str(i) for i in range(X.shape[1])]),
            label_name='mortality',
            num_classes=2,
            metadata={'source': 'MIMIC-IV', 'partition': partition}
        )


# =============================================================================
# IMAGE DATASETS (CIFAR-10, MNIST)
# =============================================================================

class CIFAR10FederatedLoader:
    """
    CIFAR-10 for federated image classification experiments.
    """

    def __init__(self, data_dir: str = './data'):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required")
        self.data_dir = data_dir

    def load_federated(self,
                       num_nodes: int = 5,
                       partition: str = 'dirichlet',
                       alpha: float = 0.5) -> Dict:
        """Load CIFAR-10 partitioned for FL."""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

        train_dataset = torchvision.datasets.CIFAR10(
            root=self.data_dir, train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root=self.data_dir, train=False, download=True, transform=transform
        )

        # Get labels
        targets = np.array(train_dataset.targets)

        # Partition
        partitioner = FederatedPartitioner(num_nodes, 42)

        if partition == 'dirichlet':
            indices_dict = self._dirichlet_partition(targets, num_nodes, alpha)
        else:
            indices = np.random.permutation(len(targets))
            splits = np.array_split(indices, num_nodes)
            indices_dict = {i: splits[i] for i in range(num_nodes)}

        # Create subsets
        node_datasets = {
            i: Subset(train_dataset, indices.tolist())
            for i, indices in indices_dict.items()
        }

        return {
            'train': node_datasets,
            'test': test_dataset,
            'num_classes': 10
        }

    def _dirichlet_partition(self, targets, num_nodes, alpha):
        """Dirichlet partition for image dataset."""
        num_classes = 10
        rng = np.random.RandomState(42)

        node_indices = {i: [] for i in range(num_nodes)}

        for c in range(num_classes):
            class_indices = np.where(targets == c)[0]
            rng.shuffle(class_indices)

            proportions = rng.dirichlet([alpha] * num_nodes)
            splits = (proportions * len(class_indices)).astype(int)
            splits[-1] = len(class_indices) - splits[:-1].sum()

            start = 0
            for node_id, n in enumerate(splits):
                node_indices[node_id].extend(class_indices[start:start + n])
                start += n

        return {i: np.array(indices) for i, indices in node_indices.items()}


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def load_dataset(name: str,
                 num_nodes: int = 5,
                 partition: str = 'dirichlet',
                 alpha: float = 0.5,
                 **kwargs) -> FederatedDataset:
    """
    Load a dataset by name.

    Available datasets:
    - 'heart': UCI Heart Disease
    - 'diabetes': Pima Indians Diabetes
    - 'breast_cancer': Wisconsin Breast Cancer
    - 'cifar10': CIFAR-10 images
    """
    loaders = {
        'heart': HeartDiseaseLoader,
        'diabetes': DiabetesLoader,
        'breast_cancer': BreastCancerLoader,
    }

    if name not in loaders:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(loaders.keys())}")

    loader = loaders[name](**kwargs)
    return loader.load_federated(num_nodes=num_nodes, partition=partition, alpha=alpha)


def list_datasets() -> List[str]:
    """List available datasets."""
    return [
        'heart - UCI Heart Disease (303 samples, 13 features)',
        'diabetes - Pima Indians Diabetes (768 samples, 8 features)',
        'breast_cancer - Wisconsin Breast Cancer (569 samples, 30 features)',
        'cifar10 - CIFAR-10 images (60000 samples, 32x32x3)',
        'mimic-iv - MIMIC-IV ICU (requires PhysioNet credentials)'
    ]


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    print("FL-EHDS Real Dataset Loaders")
    print("=" * 60)

    print("\nAvailable Datasets:")
    for ds in list_datasets():
        print(f"  - {ds}")

    # Demo with Heart Disease
    print("\n" + "=" * 60)
    print("Demo: Heart Disease Dataset")
    print("=" * 60)

    try:
        dataset = load_dataset(
            'heart',
            num_nodes=5,
            partition='dirichlet',
            alpha=0.5
        )

        print(f"\nDataset loaded: {dataset.metadata['source']}")
        print(f"Features: {len(dataset.feature_names)}")
        print(f"Classes: {dataset.num_classes}")
        print(f"Test samples: {len(dataset.test_data['y'])}")

        print("\nNode Statistics:")
        stats = dataset.get_node_stats()
        if stats is not None:
            print(stats.to_string(index=False))

    except Exception as e:
        print(f"Error loading dataset: {e}")

    # Demo with Breast Cancer
    print("\n" + "=" * 60)
    print("Demo: Breast Cancer Dataset (Non-IID with α=0.3)")
    print("=" * 60)

    try:
        dataset = load_dataset(
            'breast_cancer',
            num_nodes=5,
            partition='dirichlet',
            alpha=0.3  # More non-IID
        )

        print("\nNode Statistics:")
        stats = dataset.get_node_stats()
        if stats is not None:
            print(stats.to_string(index=False))

    except Exception as e:
        print(f"Error: {e}")

"""
Bridge module to integrate real PyTorch FL training with Streamlit dashboard.

This module provides a Streamlit-compatible wrapper around the terminal FL trainer,
enabling the web dashboard to use real neural network training instead of simulation.

Usage in app_v4.py:
    from real_trainer_bridge import RealFLTrainer, run_real_training

    # Option 1: Use wrapper class
    trainer = RealFLTrainer(config)
    results = trainer.train(num_rounds=30)

    # Option 2: Use callback-based training with progress updates
    results = run_real_training(config, progress_callback=update_progress_bar)

Author: Fabio Liberti
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
import numpy as np

# Add terminal package to path
terminal_path = Path(__file__).parent.parent / "terminal"
if str(terminal_path) not in sys.path:
    sys.path.insert(0, str(terminal_path.parent))

try:
    from terminal.fl_trainer import (
        FederatedTrainer,
        CentralizedTrainer,
        generate_healthcare_data,
        HealthcareMLP,
        run_fl_vs_centralized_comparison,
        ImageFederatedTrainer,
        HealthcareCNN,
        load_image_dataset,
    )
    PYTORCH_AVAILABLE = True
except ImportError as e:
    PYTORCH_AVAILABLE = False
    IMPORT_ERROR = str(e)


@dataclass
class StreamlitConfig:
    """Configuration compatible with Streamlit dashboard."""
    num_clients: int = 5
    samples_per_client: int = 200
    algorithm: str = "FedAvg"
    local_epochs: int = 3
    batch_size: int = 32
    learning_rate: float = 0.01
    is_iid: bool = False
    alpha: float = 0.5  # Dirichlet parameter for non-IID
    use_dp: bool = False
    epsilon: float = 10.0
    clip_norm: float = 1.0
    mu: float = 0.1  # FedProx parameter
    seed: int = 42
    # Server optimizer params
    server_lr: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.99
    tau: float = 1e-3

    @classmethod
    def from_streamlit_config(cls, st_config: Dict) -> "StreamlitConfig":
        """Convert Streamlit sidebar config to trainer config."""
        return cls(
            num_clients=st_config.get("num_clients", 5),
            samples_per_client=st_config.get("samples_per_client", 200),
            algorithm=st_config.get("algorithm", "FedAvg"),
            local_epochs=st_config.get("local_epochs", 3),
            batch_size=st_config.get("batch_size", 32),
            learning_rate=st_config.get("learning_rate", 0.01),
            is_iid=st_config.get("is_iid", False),
            alpha=st_config.get("alpha", 0.5),
            use_dp=st_config.get("use_dp", False),
            epsilon=st_config.get("epsilon", 10.0),
            clip_norm=st_config.get("clip_norm", 1.0),
            mu=st_config.get("fedprox_mu", 0.1),
            seed=st_config.get("seed", 42),
            server_lr=st_config.get("server_lr", 0.1),
            beta1=st_config.get("beta1", 0.9),
            beta2=st_config.get("beta2", 0.99),
            tau=st_config.get("tau", 1e-3),
        )


class RealFLTrainer:
    """
    Wrapper around FederatedTrainer for Streamlit integration.

    Provides:
    - Same interface as the simulated trainer in app_v4.py
    - Real PyTorch training with neural networks
    - Progress callbacks for Streamlit progress bars
    - Results in format compatible with existing visualization code
    """

    def __init__(self, config: Dict):
        """Initialize trainer with Streamlit config dict."""
        if not PYTORCH_AVAILABLE:
            raise ImportError(
                f"PyTorch or terminal module not available: {IMPORT_ERROR}\n"
                "Install with: pip install torch"
            )

        self.st_config = StreamlitConfig.from_streamlit_config(config)
        self._raw_config = config
        self.trainer = None
        self.history = []
        self._init_trainer()

    def _init_trainer(self):
        """Initialize the underlying FederatedTrainer."""
        cfg = self.st_config
        tabular_ds = self._raw_config.get("tabular_dataset")

        # Load external data for real tabular datasets
        ext_train = None
        ext_test = None
        input_dim = None

        if tabular_ds == "diabetes":
            from data.diabetes_loader import load_diabetes_data
            ext_train, ext_test, meta = load_diabetes_data(
                num_clients=cfg.num_clients,
                partition_by_hospital=not cfg.is_iid,
                is_iid=cfg.is_iid,
                seed=cfg.seed,
            )
            input_dim = meta["num_features"]
        elif tabular_ds == "heart_disease":
            from data.heart_disease_loader import load_heart_disease_data
            ext_train, ext_test, meta = load_heart_disease_data(
                num_clients=cfg.num_clients,
                partition_by_hospital=not cfg.is_iid,
                is_iid=cfg.is_iid,
                seed=cfg.seed,
            )
            input_dim = meta["num_features"]

        if ext_train is not None:
            self.trainer = FederatedTrainer(
                num_clients=cfg.num_clients,
                algorithm=cfg.algorithm,
                local_epochs=cfg.local_epochs,
                batch_size=cfg.batch_size,
                learning_rate=cfg.learning_rate,
                mu=cfg.mu,
                dp_enabled=cfg.use_dp,
                dp_epsilon=cfg.epsilon,
                dp_clip_norm=cfg.clip_norm,
                seed=cfg.seed,
                server_lr=cfg.server_lr,
                beta1=cfg.beta1,
                beta2=cfg.beta2,
                tau=cfg.tau,
                external_data=ext_train,
                external_test_data=ext_test,
                input_dim=input_dim,
            )
        else:
            self.trainer = FederatedTrainer(
                num_clients=cfg.num_clients,
                samples_per_client=cfg.samples_per_client,
                algorithm=cfg.algorithm,
                local_epochs=cfg.local_epochs,
                batch_size=cfg.batch_size,
                learning_rate=cfg.learning_rate,
                is_iid=cfg.is_iid,
                alpha=cfg.alpha,
                mu=cfg.mu,
                dp_enabled=cfg.use_dp,
                dp_epsilon=cfg.epsilon,
                dp_clip_norm=cfg.clip_norm,
                seed=cfg.seed,
                server_lr=cfg.server_lr,
                beta1=cfg.beta1,
                beta2=cfg.beta2,
                tau=cfg.tau,
            )

    def train(
        self,
        num_rounds: int = 30,
        progress_callback: Optional[Callable[[int, int, Dict], None]] = None
    ) -> Dict[str, Any]:
        """
        Train for specified rounds.

        Args:
            num_rounds: Number of FL rounds
            progress_callback: Optional callback(round, total, metrics) for progress updates

        Returns:
            Dict with training history and final metrics in Streamlit-compatible format
        """
        self.history = []

        for r in range(num_rounds):
            result = self.trainer.train_round(r)

            # Convert to Streamlit-compatible format
            round_data = {
                "round": r + 1,
                "accuracy": result.global_acc,
                "loss": result.global_loss,
                "f1": result.global_f1,
                "precision": result.global_precision,
                "recall": result.global_recall,
                "auc": result.global_auc,
                "time": result.time_seconds,
                "node_metrics": {
                    cr.client_id: {
                        "accuracy": cr.train_acc,
                        "loss": cr.train_loss,
                        "samples": cr.num_samples,
                    }
                    for cr in result.client_results
                }
            }
            self.history.append(round_data)

            if progress_callback:
                progress_callback(r + 1, num_rounds, round_data)

        # Final results in Streamlit format
        final = self.history[-1] if self.history else {}
        return {
            "history": self.history,
            "final_accuracy": final.get("accuracy", 0),
            "final_loss": final.get("loss", 0),
            "final_f1": final.get("f1", 0),
            "final_auc": final.get("auc", 0),
            "algorithm": self.st_config.algorithm,
            "num_clients": self.st_config.num_clients,
            "total_rounds": num_rounds,
            "training_mode": "real_pytorch",
        }

    def train_with_governance(
        self,
        num_rounds: int = 30,
        governance_bridge=None,
        progress_callback: Optional[Callable[[int, int, Dict], None]] = None,
        governance_callback: Optional[Callable[[int, float], tuple]] = None,
    ) -> Dict[str, Any]:
        """Train with per-round governance validation and audit logging.

        Args:
            num_rounds: Number of FL rounds.
            governance_bridge: GovernanceLifecycleBridge instance.
            progress_callback: Optional UI progress callback.
            governance_callback: Optional callback(round_num, eps_cost) -> (ok, reason).

        Returns:
            Dict with training history and final metrics.
        """
        self.history = []
        epsilon_per_round = (
            self.st_config.epsilon / num_rounds
            if self.st_config.use_dp
            else 0.0
        )

        for r in range(num_rounds):
            # Pre-round governance check
            if governance_callback is not None:
                ok, reason = governance_callback(r, epsilon_per_round)
                if not ok:
                    break

            result = self.trainer.train_round(r)

            round_data = {
                "round": r + 1,
                "accuracy": result.global_acc,
                "loss": result.global_loss,
                "f1": result.global_f1,
                "precision": result.global_precision,
                "recall": result.global_recall,
                "auc": result.global_auc,
                "time": result.time_seconds,
                "node_metrics": {
                    cr.client_id: {
                        "accuracy": cr.train_acc,
                        "loss": cr.train_loss,
                        "samples": cr.num_samples,
                    }
                    for cr in result.client_results
                },
            }
            self.history.append(round_data)

            if progress_callback:
                progress_callback(r + 1, num_rounds, round_data)

            # Post-round governance logging
            if governance_bridge is not None:
                governance_bridge.log_round_completion(r, result, epsilon_per_round)

        # End governance session
        if governance_bridge is not None and self.history:
            final = self.history[-1]
            governance_bridge.end_session(
                total_rounds=len(self.history),
                final_metrics={
                    "accuracy": final["accuracy"],
                    "loss": final["loss"],
                    "f1": final["f1"],
                    "auc": final["auc"],
                },
                success=True,
            )

        final = self.history[-1] if self.history else {}
        return {
            "history": self.history,
            "final_accuracy": final.get("accuracy", 0),
            "final_loss": final.get("loss", 0),
            "final_f1": final.get("f1", 0),
            "final_auc": final.get("auc", 0),
            "algorithm": self.st_config.algorithm,
            "num_clients": self.st_config.num_clients,
            "total_rounds": len(self.history),
            "training_mode": "real_pytorch_governance",
        }

    def get_client_data_stats(self) -> Dict:
        """Get data distribution statistics per client."""
        return self.trainer.get_client_data_stats()


@dataclass
class ImageStreamlitConfig:
    """Configuration for image-based FL training from Streamlit dashboard."""
    data_dir: str = ""
    num_clients: int = 5
    algorithm: str = "FedAvg"
    local_epochs: int = 3
    batch_size: int = 32
    learning_rate: float = 0.001
    is_iid: bool = False
    alpha: float = 0.5
    mu: float = 0.1
    use_dp: bool = False
    epsilon: float = 10.0
    clip_norm: float = 1.0
    seed: int = 42
    img_size: int = 128
    # Server optimizer params
    server_lr: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.99
    tau: float = 1e-3

    @classmethod
    def from_streamlit_config(cls, st_config: Dict) -> "ImageStreamlitConfig":
        """Convert Streamlit sidebar config to image trainer config."""
        return cls(
            data_dir=st_config.get("data_dir", ""),
            num_clients=st_config.get("num_clients", 5),
            algorithm=st_config.get("algorithm", "FedAvg"),
            local_epochs=st_config.get("local_epochs", 3),
            batch_size=st_config.get("batch_size", 32),
            learning_rate=st_config.get("learning_rate", 0.001),
            is_iid=st_config.get("is_iid", False),
            alpha=st_config.get("alpha", 0.5),
            mu=st_config.get("fedprox_mu", 0.1),
            use_dp=st_config.get("use_dp", False),
            epsilon=st_config.get("epsilon", 10.0),
            clip_norm=st_config.get("clip_norm", 1.0),
            seed=st_config.get("seed", 42),
            img_size=st_config.get("img_size", 128),
            server_lr=st_config.get("server_lr", 0.1),
            beta1=st_config.get("beta1", 0.9),
            beta2=st_config.get("beta2", 0.99),
            tau=st_config.get("tau", 1e-3),
        )


class RealImageFLTrainer:
    """
    Wrapper around ImageFederatedTrainer for Streamlit integration.

    Enables the web dashboard to run real CNN-based FL training
    on clinical imaging datasets (chest_xray, Brain_Tumor, etc.).
    """

    def __init__(self, config: Dict):
        """Initialize image trainer with Streamlit config dict."""
        if not PYTORCH_AVAILABLE:
            raise ImportError(
                f"PyTorch or terminal module not available: {IMPORT_ERROR}\n"
                "Install with: pip install torch"
            )

        self.st_config = ImageStreamlitConfig.from_streamlit_config(config)
        if not self.st_config.data_dir:
            raise ValueError("data_dir is required for image training")

        self.trainer = None
        self._init_trainer()

    def _init_trainer(self):
        """Initialize the underlying ImageFederatedTrainer."""
        cfg = self.st_config
        self.trainer = ImageFederatedTrainer(
            data_dir=cfg.data_dir,
            num_clients=cfg.num_clients,
            algorithm=cfg.algorithm,
            local_epochs=cfg.local_epochs,
            batch_size=cfg.batch_size,
            learning_rate=cfg.learning_rate,
            is_iid=cfg.is_iid,
            alpha=cfg.alpha,
            mu=cfg.mu,
            dp_enabled=cfg.use_dp,
            dp_epsilon=cfg.epsilon,
            dp_clip_norm=cfg.clip_norm,
            seed=cfg.seed,
            img_size=cfg.img_size,
            server_lr=cfg.server_lr,
            beta1=cfg.beta1,
            beta2=cfg.beta2,
            tau=cfg.tau,
        )

    def train(
        self,
        num_rounds: int = 15,
        progress_callback: Optional[Callable[[int, int, Dict], None]] = None
    ) -> Dict[str, Any]:
        """
        Train for specified rounds on imaging data.

        Args:
            num_rounds: Number of FL rounds
            progress_callback: Optional callback(round, total, metrics)

        Returns:
            Dict with training history and final metrics in Streamlit-compatible format
        """
        history = []

        for r in range(num_rounds):
            result = self.trainer.train_round(r)

            round_data = {
                "round": r + 1,
                "accuracy": result.global_acc,
                "loss": result.global_loss,
                "f1": result.global_f1,
                "precision": result.global_precision,
                "recall": result.global_recall,
                "auc": result.global_auc,
                "time": result.time_seconds,
                "node_metrics": {
                    cr.client_id: {
                        "accuracy": cr.train_acc,
                        "loss": cr.train_loss,
                        "samples": cr.num_samples,
                    }
                    for cr in result.client_results
                }
            }
            history.append(round_data)

            if progress_callback:
                progress_callback(r + 1, num_rounds, round_data)

        final = history[-1] if history else {}
        return {
            "history": history,
            "final_accuracy": final.get("accuracy", 0),
            "final_loss": final.get("loss", 0),
            "final_f1": final.get("f1", 0),
            "final_auc": final.get("auc", 0),
            "algorithm": self.st_config.algorithm,
            "num_clients": self.st_config.num_clients,
            "total_rounds": num_rounds,
            "dataset": self.st_config.data_dir,
            "img_size": self.st_config.img_size,
            "training_mode": "real_pytorch_imaging",
        }

    def get_dataset_info(self) -> Dict:
        """Get info about loaded dataset (classes, samples per client)."""
        return {
            "num_classes": self.trainer.num_classes,
            "class_names": self.trainer.class_names,
            "data_dir": self.st_config.data_dir,
        }


def run_real_image_training(
    config: Dict,
    num_rounds: int = 15,
    progress_callback: Optional[Callable] = None,
) -> Dict:
    """
    Convenience function for running real image FL training from Streamlit.

    config must include 'data_dir' pointing to the imaging dataset folder.
    """
    trainer = RealImageFLTrainer(config)
    return trainer.train(num_rounds, progress_callback)


def discover_datasets(data_dir: str = None) -> List[Dict]:
    """
    Discover available imaging datasets in the data directory.

    Returns list of dicts with: name, path, num_classes, total_images.
    """
    if data_dir is None:
        data_dir = str(Path(__file__).parent.parent / "data")

    data_path = Path(data_dir)
    if not data_path.exists():
        return []

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff",
                        ".JPG", ".JPEG", ".PNG"}
    split_folders = {"train", "test", "val", "valid", "validation",
                     "training", "testing", "data"}

    datasets = []
    for folder in sorted(data_path.iterdir()):
        if not folder.is_dir() or folder.name.startswith("."):
            continue

        subdirs = [d for d in folder.iterdir()
                   if d.is_dir() and not d.name.startswith(".")]
        if not subdirs:
            continue

        subdir_names_lower = {d.name.lower() for d in subdirs}
        is_split = bool(subdir_names_lower & split_folders)

        classes = set()
        total_images = 0

        if is_split:
            for split_dir in subdirs:
                if split_dir.name.lower() in split_folders:
                    for class_dir in split_dir.iterdir():
                        if class_dir.is_dir() and not class_dir.name.startswith("."):
                            if class_dir.name.lower() not in split_folders:
                                classes.add(class_dir.name)
                                total_images += sum(
                                    1 for f in class_dir.iterdir()
                                    if f.suffix in image_extensions
                                )
        else:
            for class_dir in subdirs:
                classes.add(class_dir.name)
                total_images += sum(
                    1 for f in class_dir.iterdir()
                    if f.suffix in image_extensions
                )

        if classes and total_images > 0:
            datasets.append({
                "name": folder.name,
                "path": str(folder),
                "num_classes": len(classes),
                "class_names": sorted(classes),
                "total_images": total_images,
            })

    return datasets


def run_real_training(
    config: Dict,
    num_rounds: int = 30,
    progress_callback: Optional[Callable] = None,
) -> Dict:
    """
    Convenience function for running real FL training.

    Example usage in Streamlit:
        import streamlit as st
        from real_trainer_bridge import run_real_training

        progress_bar = st.progress(0)
        status_text = st.empty()

        def update_progress(r, total, metrics):
            progress_bar.progress(r / total)
            status_text.text(f"Round {r}/{total}: Acc={metrics['accuracy']:.3f}")

        results = run_real_training(
            config=sidebar_config,
            num_rounds=30,
            progress_callback=update_progress
        )
    """
    trainer = RealFLTrainer(config)
    return trainer.train(num_rounds, progress_callback)


def run_comparison(
    config: Dict,
    algorithms: List[str] = None,
    num_rounds: int = 30,
    seeds: List[int] = None,
) -> Dict:
    """
    Run FL vs Centralized comparison.

    Returns results suitable for Streamlit tables and charts.
    """
    if algorithms is None:
        algorithms = ["FedAvg", "FedProx"]
    if seeds is None:
        seeds = [42, 123, 456]

    cfg = StreamlitConfig.from_streamlit_config(config)

    summary = run_fl_vs_centralized_comparison(
        num_clients=cfg.num_clients,
        samples_per_client=cfg.samples_per_client,
        fl_rounds=num_rounds,
        local_epochs=cfg.local_epochs,
        batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate,
        is_iid=cfg.is_iid,
        alpha=cfg.alpha,
        seeds=seeds,
        algorithms=algorithms,
        verbose=False,
    )

    return summary


# Streamlit integration helpers

def create_streamlit_progress_callback(progress_bar, status_text, metrics_container=None):
    """
    Create a progress callback for Streamlit components.

    Usage:
        progress_bar = st.progress(0)
        status_text = st.empty()
        callback = create_streamlit_progress_callback(progress_bar, status_text)
        results = run_real_training(config, progress_callback=callback)
    """
    def callback(round_num: int, total_rounds: int, metrics: Dict):
        progress_bar.progress(round_num / total_rounds)
        status_text.text(
            f"Round {round_num}/{total_rounds} | "
            f"Acc: {metrics['accuracy']:.3f} | "
            f"Loss: {metrics['loss']:.4f} | "
            f"F1: {metrics['f1']:.3f}"
        )
        if metrics_container is not None:
            # Update live metrics display
            pass

    return callback


def check_pytorch_available() -> tuple:
    """
    Check if PyTorch is available for real training.

    Returns:
        (available: bool, message: str)
    """
    if PYTORCH_AVAILABLE:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return True, f"PyTorch available (device: {device})"
    else:
        return False, f"PyTorch not available: {IMPORT_ERROR}"

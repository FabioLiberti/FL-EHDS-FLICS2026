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
        self.trainer = None
        self.history = []
        self._init_trainer()

    def _init_trainer(self):
        """Initialize the underlying FederatedTrainer."""
        cfg = self.st_config
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

    def get_client_data_stats(self) -> Dict:
        """Get data distribution statistics per client."""
        return self.trainer.get_client_data_stats()


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

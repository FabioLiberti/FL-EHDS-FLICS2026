#!/usr/bin/env python3
"""
FL-EHDS Interactive Training Demo

Demonstrates federated learning with real-time visualization.
Choose your visualization mode:
- terminal: Rich terminal UI (default, works everywhere)
- tensorboard: TensorBoard logging (launch with: tensorboard --logdir=runs)
- gui: Matplotlib real-time plots
- all: All visualizations combined

Usage:
    python demo_fl_training.py                    # Terminal mode
    python demo_fl_training.py --mode tensorboard # TensorBoard mode
    python demo_fl_training.py --mode gui         # GUI mode
    python demo_fl_training.py --rounds 100       # Custom rounds

Author: Fabio Liberti
"""

import argparse
import time
import sys
import numpy as np
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from visualization.fl_visualizer import (
    TrainingMetrics,
    create_visualizer,
    RICH_AVAILABLE,
    TENSORBOARD_AVAILABLE,
    MATPLOTLIB_AVAILABLE
)


class FLEHDSDemo:
    """Federated Learning demo with EHDS compliance features."""

    def __init__(
        self,
        num_hospitals: int = 5,
        samples_per_hospital: list = None,
        num_features: int = 5,
        random_seed: int = 42
    ):
        self.num_hospitals = num_hospitals
        self.samples_per_hospital = samples_per_hospital or [400, 500, 350, 450, 380]
        self.num_features = num_features
        self.random_seed = random_seed

        np.random.seed(random_seed)

        self.hospital_names = [
            "IT-Roma", "DE-Berlin", "FR-Paris", "ES-Madrid", "NL-Amsterdam"
        ][:num_hospitals]

        # Generate data
        self._generate_data()

        # Model weights
        self.weights = np.zeros(num_features + 1)

        # Privacy tracking
        self.privacy_budget_total = 10.0
        self.privacy_budget_spent = 0.0

        # Communication tracking
        self.total_bytes = 0

    def _generate_data(self):
        """Generate synthetic healthcare data."""
        self.hospital_data = {}

        hospital_params = [
            {"mean_age": 50, "pos_rate": 0.40},
            {"mean_age": 55, "pos_rate": 0.42},
            {"mean_age": 52, "pos_rate": 0.45},
            {"mean_age": 58, "pos_rate": 0.55},
            {"mean_age": 60, "pos_rate": 0.60},
        ][:self.num_hospitals]

        for name, n_samples, params in zip(
            self.hospital_names, self.samples_per_hospital, hospital_params
        ):
            age = np.random.normal(params["mean_age"], 15, n_samples)
            bmi = np.random.normal(26, 5, n_samples)
            systolic = np.random.normal(125, 20, n_samples)
            glucose = np.random.normal(100, 25, n_samples)
            chol = np.random.normal(200, 40, n_samples)

            X = np.column_stack([age, bmi, systolic, glucose, chol])

            # Normalize
            X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

            # Generate labels
            risk = 0.5 * X[:, 0] + 0.3 * X[:, 1] + 0.2 * X[:, 2]
            prob = 1 / (1 + np.exp(-risk))
            y = (np.random.random(n_samples) < prob).astype(int)

            # Add bias term
            X_bias = np.hstack([X, np.ones((n_samples, 1))])

            self.hospital_data[name] = {
                "X": X_bias,
                "y": y,
                "n_samples": n_samples
            }

    def train_round(
        self,
        round_num: int,
        learning_rate: float = 0.1,
        local_epochs: int = 3,
        participation_rate: float = 0.85,
        add_dp_noise: bool = True,
        epsilon_per_round: float = 0.2
    ) -> TrainingMetrics:
        """Execute one FL training round."""
        start_time = time.time()

        # Select participating clients
        participating = []
        for name in self.hospital_names:
            if np.random.random() < participation_rate:
                participating.append(name)

        if not participating:
            participating = [self.hospital_names[0]]

        # Local training
        gradients = []
        sample_counts = []
        client_accuracies = {}
        client_losses = {}
        gradient_norms = {}

        for name in self.hospital_names:
            data = self.hospital_data[name]

            if name in participating:
                # Train locally
                local_weights = self.weights.copy()

                for _ in range(local_epochs):
                    batch_size = min(32, data["n_samples"])
                    indices = np.random.choice(data["n_samples"], batch_size, replace=False)
                    X_batch = data["X"][indices]
                    y_batch = data["y"][indices]

                    logits = X_batch @ local_weights
                    probs = 1 / (1 + np.exp(-np.clip(logits, -500, 500)))
                    grad = X_batch.T @ (probs - y_batch) / batch_size
                    local_weights -= learning_rate * grad

                gradient = local_weights - self.weights

                # Gradient clipping
                norm = np.linalg.norm(gradient)
                if norm > 1.0:
                    gradient = gradient / norm

                gradient_norms[name] = float(norm)
                gradients.append(gradient)
                sample_counts.append(data["n_samples"])

            # Evaluate on all clients
            logits = data["X"] @ self.weights
            probs = 1 / (1 + np.exp(-np.clip(logits, -500, 500)))
            preds = (probs > 0.5).astype(int)
            client_accuracies[name] = float(np.mean(preds == data["y"]))
            client_losses[name] = float(-np.mean(
                data["y"] * np.log(probs + 1e-8) + (1 - data["y"]) * np.log(1 - probs + 1e-8)
            ))

            if name not in gradient_norms:
                gradient_norms[name] = 0.0

        # Aggregate
        if gradients:
            total_samples = sum(sample_counts)
            weighted_grad = sum(
                g * (n / total_samples) for g, n in zip(gradients, sample_counts)
            )

            # Add DP noise
            if add_dp_noise and self.privacy_budget_spent < self.privacy_budget_total:
                noise_scale = 1.0 / epsilon_per_round
                noise = np.random.normal(0, noise_scale, weighted_grad.shape)
                weighted_grad += noise * 0.01  # Scaled noise
                self.privacy_budget_spent += epsilon_per_round

            self.weights += weighted_grad

        # Communication cost (6 params * 4 bytes * 2 directions * num_participants)
        round_bytes = len(participating) * (self.num_features + 1) * 4 * 2
        self.total_bytes += round_bytes

        # Global metrics
        all_preds = []
        all_labels = []
        for data in self.hospital_data.values():
            logits = data["X"] @ self.weights
            probs = 1 / (1 + np.exp(-np.clip(logits, -500, 500)))
            preds = (probs > 0.5).astype(int)
            all_preds.extend(preds)
            all_labels.extend(data["y"])

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        accuracy = float(np.mean(all_preds == all_labels))

        # F1
        tp = np.sum((all_preds == 1) & (all_labels == 1))
        fp = np.sum((all_preds == 1) & (all_labels == 0))
        fn = np.sum((all_preds == 0) & (all_labels == 1))
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        # Loss
        all_logits = []
        for data in self.hospital_data.values():
            all_logits.extend(data["X"] @ self.weights)
        all_logits = np.array(all_logits)
        all_probs = 1 / (1 + np.exp(-np.clip(all_logits, -500, 500)))
        loss = -np.mean(
            all_labels * np.log(all_probs + 1e-8) + (1 - all_labels) * np.log(1 - all_probs + 1e-8)
        )

        round_time = time.time() - start_time

        return TrainingMetrics(
            round=round_num,
            global_accuracy=accuracy,
            global_loss=float(loss),
            global_f1=float(f1),
            client_accuracies=client_accuracies,
            client_losses=client_losses,
            client_samples={name: self.hospital_data[name]["n_samples"] for name in self.hospital_names},
            participating_clients=participating,
            privacy_budget_spent=self.privacy_budget_spent,
            privacy_budget_remaining=self.privacy_budget_total - self.privacy_budget_spent,
            communication_bytes=self.total_bytes,
            round_time_seconds=round_time,
            gradient_norms=gradient_norms
        )


def run_demo(
    mode: str = "terminal",
    num_rounds: int = 50,
    delay: float = 0.1
):
    """Run the FL training demo."""
    print("=" * 60)
    print("FL-EHDS Training Demo")
    print("=" * 60)
    print(f"Mode: {mode}")
    print(f"Rounds: {num_rounds}")
    print(f"Rich available: {RICH_AVAILABLE}")
    print(f"TensorBoard available: {TENSORBOARD_AVAILABLE}")
    print(f"Matplotlib available: {MATPLOTLIB_AVAILABLE}")
    print("=" * 60)
    print()

    # Create demo
    demo = FLEHDSDemo(num_hospitals=5)

    # Determine available mode
    if mode == "terminal" and not RICH_AVAILABLE:
        print("Rich not available, falling back to simple output")
        mode = "simple"
    elif mode == "tensorboard" and not TENSORBOARD_AVAILABLE:
        print("TensorBoard not available, falling back to terminal")
        mode = "terminal" if RICH_AVAILABLE else "simple"
    elif mode == "gui" and not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available, falling back to terminal")
        mode = "terminal" if RICH_AVAILABLE else "simple"

    # Simple mode (no dependencies)
    if mode == "simple":
        print("Training with simple output...")
        for round_num in range(1, num_rounds + 1):
            metrics = demo.train_round(round_num)
            if round_num % 10 == 0 or round_num == 1:
                print(f"Round {round_num:3d}: Acc={metrics.global_accuracy:.2%}, "
                      f"F1={metrics.global_f1:.3f}, ε={metrics.privacy_budget_spent:.2f}")
        print("\nTraining complete!")
        return

    # Create visualizer
    try:
        visualizer = create_visualizer(
            mode=mode,
            total_rounds=num_rounds,
            num_clients=5,
            tensorboard_dir="runs/fl_ehds_demo"
        )
    except ImportError as e:
        print(f"Error creating visualizer: {e}")
        print("Falling back to simple output")
        run_demo("simple", num_rounds, delay)
        return

    # Run training with visualization
    if mode == "terminal":
        with visualizer:
            for round_num in range(1, num_rounds + 1):
                metrics = demo.train_round(round_num)
                visualizer.update(metrics)
                time.sleep(delay)

        print("\n" + "=" * 60)
        print("Training Complete!")
        print(f"Final Accuracy: {metrics.global_accuracy:.2%}")
        print(f"Final F1: {metrics.global_f1:.3f}")
        print(f"Privacy Budget Used: {metrics.privacy_budget_spent:.2f}/{demo.privacy_budget_total}")
        print("=" * 60)

    elif mode == "tensorboard":
        print("Logging to TensorBoard...")
        print("Launch TensorBoard with: tensorboard --logdir=runs")
        print()

        with visualizer:
            for round_num in range(1, num_rounds + 1):
                metrics = demo.train_round(round_num)
                visualizer.update(metrics)

                if round_num % 10 == 0:
                    print(f"Round {round_num}: Acc={metrics.global_accuracy:.2%}")

                time.sleep(delay)

        print("\nTraining complete! View results in TensorBoard.")

    elif mode == "gui":
        print("Running with GUI visualization...")
        print("Close the window to exit.")

        for round_num in range(1, num_rounds + 1):
            metrics = demo.train_round(round_num)
            visualizer.update(metrics)
            time.sleep(delay)

        print("\nTraining complete!")
        visualizer.save("fl_training_results.png")
        visualizer.show()

    elif mode == "all":
        with visualizer:
            for round_num in range(1, num_rounds + 1):
                metrics = demo.train_round(round_num)
                visualizer.update(metrics)
                time.sleep(delay)

        print("\nTraining complete!")


def main():
    parser = argparse.ArgumentParser(description="FL-EHDS Training Demo")
    parser.add_argument(
        "--mode", "-m",
        choices=["terminal", "tensorboard", "gui", "all", "simple"],
        default="terminal",
        help="Visualization mode (default: terminal)"
    )
    parser.add_argument(
        "--rounds", "-r",
        type=int,
        default=50,
        help="Number of training rounds (default: 50)"
    )
    parser.add_argument(
        "--delay", "-d",
        type=float,
        default=0.1,
        help="Delay between rounds in seconds (default: 0.1)"
    )

    args = parser.parse_args()

    run_demo(
        mode=args.mode,
        num_rounds=args.rounds,
        delay=args.delay
    )


if __name__ == "__main__":
    main()

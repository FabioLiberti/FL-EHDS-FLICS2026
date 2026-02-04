#!/usr/bin/env python3
"""
EHDS Article 71 Opt-Out Impact Simulator

Simulates the impact of citizen opt-out rates on FL model performance.
Key question: "How much does accuracy degrade with N% opt-out?"

This tool helps HDABs and researchers understand:
1. Minimum viable sample sizes per hospital
2. Critical opt-out thresholds beyond which models fail
3. Demographic bias risks when opt-out is non-random
4. Recommendations for citizen engagement strategies

Author: Fabio Liberti
EHDS Article 71 Reference: Regulation (EU) 2025/327
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import warnings

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


@dataclass
class OptOutScenario:
    """Represents an opt-out scenario configuration."""
    name: str
    overall_rate: float  # 0.0 to 1.0
    demographic_bias: Optional[Dict[str, float]] = None  # e.g., {"age>65": 0.3, "age<30": 0.1}
    purpose_specific: bool = False  # If True, opt-out applies only to specific purposes
    description: str = ""


@dataclass
class SimulationResult:
    """Results from a single simulation run."""
    scenario_name: str
    opt_out_rate: float
    original_samples: int
    remaining_samples: int
    accuracy: float
    accuracy_drop: float  # Compared to baseline (0% opt-out)
    f1_score: float
    auc_roc: float
    per_hospital_samples: Dict[str, int]
    per_hospital_accuracy: Dict[str, float]
    demographic_distribution_shift: Dict[str, float]
    convergence_rounds: int
    model_viable: bool  # True if accuracy > threshold


@dataclass
class SimulationSummary:
    """Summary across multiple opt-out rates."""
    timestamp: str
    num_hospitals: int
    baseline_accuracy: float
    critical_optout_threshold: float  # Rate at which accuracy drops below viability
    results: List[SimulationResult]
    recommendations: List[str]


class OptOutImpactSimulator:
    """
    Simulates FL training under various opt-out scenarios.

    Implements Article 71 of EHDS Regulation (EU) 2025/327 which allows
    citizens to opt-out of secondary use of their electronic health data.
    """

    def __init__(
        self,
        num_hospitals: int = 5,
        samples_per_hospital: List[int] = None,
        num_features: int = 5,
        viability_threshold: float = 0.55,
        random_seed: int = 42
    ):
        """
        Initialize the simulator.

        Args:
            num_hospitals: Number of participating hospitals
            samples_per_hospital: List of sample counts per hospital
            num_features: Number of features in the dataset
            viability_threshold: Minimum accuracy for model to be considered viable
            random_seed: Random seed for reproducibility
        """
        self.num_hospitals = num_hospitals
        self.samples_per_hospital = samples_per_hospital or [400, 500, 350, 450, 380]
        self.num_features = num_features
        self.viability_threshold = viability_threshold
        self.random_seed = random_seed

        np.random.seed(random_seed)

        # Generate base datasets for each hospital
        self._generate_hospital_data()

    def _generate_hospital_data(self):
        """Generate synthetic healthcare data for each hospital."""
        self.hospital_data = {}
        self.hospital_names = [
            "Hospital-IT-Roma",
            "Hospital-DE-Berlin",
            "Hospital-FR-Paris",
            "Hospital-ES-Madrid",
            "Hospital-NL-Amsterdam"
        ][:self.num_hospitals]

        # Demographic parameters per hospital (simulating geographic variation)
        hospital_params = [
            {"mean_age": 50, "age_std": 15, "pos_rate": 0.40},
            {"mean_age": 55, "age_std": 12, "pos_rate": 0.42},
            {"mean_age": 52, "age_std": 18, "pos_rate": 0.45},
            {"mean_age": 58, "age_std": 14, "pos_rate": 0.55},
            {"mean_age": 60, "age_std": 10, "pos_rate": 0.60},
        ][:self.num_hospitals]

        for i, (name, n_samples, params) in enumerate(
            zip(self.hospital_names, self.samples_per_hospital, hospital_params)
        ):
            # Generate features
            age = np.random.normal(params["mean_age"], params["age_std"], n_samples)
            age = np.clip(age, 18, 95)

            bmi = np.random.normal(26, 5, n_samples)
            bmi = np.clip(bmi, 15, 50)

            systolic_bp = np.random.normal(125, 20, n_samples)
            systolic_bp = np.clip(systolic_bp, 80, 200)

            glucose = np.random.normal(100, 25, n_samples)
            glucose = np.clip(glucose, 60, 300)

            cholesterol = np.random.normal(200, 40, n_samples)
            cholesterol = np.clip(cholesterol, 100, 400)

            X = np.column_stack([age, bmi, systolic_bp, glucose, cholesterol])

            # Generate labels based on risk factors
            risk_score = (
                0.02 * (age - 50) +
                0.03 * (bmi - 25) +
                0.01 * (systolic_bp - 120) +
                0.005 * (glucose - 100) +
                0.002 * (cholesterol - 200)
            )
            prob = 1 / (1 + np.exp(-risk_score + np.log(params["pos_rate"] / (1 - params["pos_rate"]))))
            y = (np.random.random(n_samples) < prob).astype(int)

            self.hospital_data[name] = {
                "X": X,
                "y": y,
                "age": age,
                "demographics": {
                    "mean_age": float(np.mean(age)),
                    "age_over_65": float(np.mean(age > 65)),
                    "age_under_30": float(np.mean(age < 30)),
                    "pos_rate": float(np.mean(y))
                }
            }

    def apply_optout(
        self,
        opt_out_rate: float,
        demographic_bias: Optional[Dict[str, float]] = None
    ) -> Dict[str, Dict]:
        """
        Apply opt-out filtering to hospital data.

        Args:
            opt_out_rate: Overall opt-out rate (0.0 to 1.0)
            demographic_bias: Dictionary mapping demographic groups to their opt-out rates
                              e.g., {"age>65": 0.3} means 30% of patients over 65 opt out

        Returns:
            Filtered hospital data
        """
        filtered_data = {}

        for name, data in self.hospital_data.items():
            n_samples = len(data["y"])
            age = data["age"]

            # Calculate per-sample opt-out probability
            opt_out_prob = np.full(n_samples, opt_out_rate)

            if demographic_bias:
                # Adjust opt-out probability based on demographics
                for demo_key, demo_rate in demographic_bias.items():
                    if demo_key == "age>65":
                        mask = age > 65
                    elif demo_key == "age<30":
                        mask = age < 30
                    elif demo_key == "age>50":
                        mask = age > 50
                    else:
                        continue

                    # Blend the demographic-specific rate
                    opt_out_prob[mask] = demo_rate

            # Apply opt-out
            keep_mask = np.random.random(n_samples) > opt_out_prob
            n_remaining = np.sum(keep_mask)

            if n_remaining < 10:
                warnings.warn(f"{name}: Only {n_remaining} samples remaining after opt-out")

            filtered_data[name] = {
                "X": data["X"][keep_mask],
                "y": data["y"][keep_mask],
                "age": age[keep_mask],
                "original_samples": n_samples,
                "remaining_samples": n_remaining,
                "opt_out_rate": 1 - (n_remaining / n_samples),
                "demographics": {
                    "mean_age": float(np.mean(age[keep_mask])) if n_remaining > 0 else 0,
                    "age_over_65": float(np.mean(age[keep_mask] > 65)) if n_remaining > 0 else 0,
                    "age_under_30": float(np.mean(age[keep_mask] < 30)) if n_remaining > 0 else 0,
                    "pos_rate": float(np.mean(data["y"][keep_mask])) if n_remaining > 0 else 0
                }
            }

        return filtered_data

    def train_federated_model(
        self,
        hospital_data: Dict[str, Dict],
        rounds: int = 50,
        local_epochs: int = 3,
        learning_rate: float = 0.1
    ) -> Tuple[float, float, float, Dict[str, float], int]:
        """
        Train a federated model on the (possibly filtered) hospital data.

        Returns:
            Tuple of (accuracy, f1, auc, per_hospital_accuracy, convergence_round)
        """
        # Simple logistic regression weights
        n_features = self.num_features
        weights = np.zeros(n_features + 1)  # +1 for bias

        # Normalize features globally
        all_X = np.vstack([d["X"] for d in hospital_data.values() if len(d["X"]) > 0])
        if len(all_X) == 0:
            return 0.0, 0.0, 0.5, {}, rounds

        mean = np.mean(all_X, axis=0)
        std = np.std(all_X, axis=0) + 1e-8

        for name in hospital_data:
            if len(hospital_data[name]["X"]) > 0:
                hospital_data[name]["X_norm"] = (hospital_data[name]["X"] - mean) / std

        best_acc = 0
        convergence_round = rounds

        for round_num in range(rounds):
            # Local training at each hospital
            gradients = []
            sample_counts = []

            for name, data in hospital_data.items():
                if len(data.get("X_norm", [])) < 10:
                    continue

                X = data["X_norm"]
                y = data["y"]
                n = len(y)

                # Add bias term
                X_bias = np.hstack([X, np.ones((n, 1))])

                # Local SGD
                local_weights = weights.copy()
                for _ in range(local_epochs):
                    # Mini-batch gradient
                    batch_size = min(32, n)
                    indices = np.random.choice(n, batch_size, replace=False)
                    X_batch = X_bias[indices]
                    y_batch = y[indices]

                    # Logistic regression gradient
                    logits = X_batch @ local_weights
                    probs = 1 / (1 + np.exp(-np.clip(logits, -500, 500)))
                    grad = X_batch.T @ (probs - y_batch) / batch_size

                    local_weights -= learning_rate * grad

                gradients.append(local_weights - weights)
                sample_counts.append(n)

            if not gradients:
                break

            # FedAvg aggregation
            total_samples = sum(sample_counts)
            weighted_grad = sum(
                g * (n / total_samples) for g, n in zip(gradients, sample_counts)
            )
            weights += weighted_grad

            # Evaluate accuracy
            all_preds = []
            all_labels = []
            for data in hospital_data.values():
                if len(data.get("X_norm", [])) > 0:
                    X_bias = np.hstack([data["X_norm"], np.ones((len(data["X_norm"]), 1))])
                    probs = 1 / (1 + np.exp(-np.clip(X_bias @ weights, -500, 500)))
                    preds = (probs > 0.5).astype(int)
                    all_preds.extend(preds)
                    all_labels.extend(data["y"])

            if all_labels:
                acc = np.mean(np.array(all_preds) == np.array(all_labels))
                if acc > best_acc + 0.01:
                    best_acc = acc
                    convergence_round = round_num

        # Final evaluation
        per_hospital_acc = {}
        all_preds = []
        all_labels = []
        all_probs = []

        for name, data in hospital_data.items():
            if len(data.get("X_norm", [])) > 0:
                X_bias = np.hstack([data["X_norm"], np.ones((len(data["X_norm"]), 1))])
                probs = 1 / (1 + np.exp(-np.clip(X_bias @ weights, -500, 500)))
                preds = (probs > 0.5).astype(int)
                per_hospital_acc[name] = float(np.mean(preds == data["y"]))
                all_preds.extend(preds)
                all_labels.extend(data["y"])
                all_probs.extend(probs)

        if not all_labels:
            return 0.0, 0.0, 0.5, {}, rounds

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        accuracy = float(np.mean(all_preds == all_labels))

        # F1 score
        tp = np.sum((all_preds == 1) & (all_labels == 1))
        fp = np.sum((all_preds == 1) & (all_labels == 0))
        fn = np.sum((all_preds == 0) & (all_labels == 1))
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        # AUC-ROC (simple approximation)
        sorted_indices = np.argsort(all_probs)[::-1]
        sorted_labels = all_labels[sorted_indices]
        tpr = np.cumsum(sorted_labels) / (np.sum(all_labels) + 1e-8)
        fpr = np.cumsum(1 - sorted_labels) / (np.sum(1 - all_labels) + 1e-8)
        auc = np.trapz(tpr, fpr)

        return accuracy, float(f1), float(auc), per_hospital_acc, convergence_round

    def run_simulation(
        self,
        opt_out_rates: List[float] = None,
        scenarios: List[OptOutScenario] = None,
        num_trials: int = 3
    ) -> SimulationSummary:
        """
        Run comprehensive opt-out impact simulation.

        Args:
            opt_out_rates: List of opt-out rates to test (0.0 to 1.0)
            scenarios: Custom scenarios with demographic biases
            num_trials: Number of trials per configuration for averaging

        Returns:
            SimulationSummary with all results
        """
        if opt_out_rates is None:
            opt_out_rates = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70]

        results = []

        # Baseline (0% opt-out)
        baseline_acc, baseline_f1, baseline_auc, _, _ = self.train_federated_model(
            self.hospital_data
        )

        print(f"Baseline (0% opt-out): Accuracy={baseline_acc:.2%}, F1={baseline_f1:.3f}")

        # Test each opt-out rate
        for rate in opt_out_rates:
            accs, f1s, aucs = [], [], []
            per_hospital_accs = {name: [] for name in self.hospital_names}
            convergence_rounds = []

            for trial in range(num_trials):
                np.random.seed(self.random_seed + trial)
                filtered = self.apply_optout(rate)
                acc, f1, auc, ph_acc, conv = self.train_federated_model(filtered)

                accs.append(acc)
                f1s.append(f1)
                aucs.append(auc)
                convergence_rounds.append(conv)
                for name, a in ph_acc.items():
                    per_hospital_accs[name].append(a)

            # Average results
            avg_acc = np.mean(accs)
            avg_f1 = np.mean(f1s)
            avg_auc = np.mean(aucs)
            avg_conv = int(np.mean(convergence_rounds))

            # Sample counts from last trial
            total_original = sum(self.samples_per_hospital)
            filtered = self.apply_optout(rate)
            total_remaining = sum(d["remaining_samples"] for d in filtered.values())

            # Demographic shift
            original_demos = {
                "mean_age": np.mean([d["demographics"]["mean_age"] for d in self.hospital_data.values()]),
                "pos_rate": np.mean([d["demographics"]["pos_rate"] for d in self.hospital_data.values()])
            }
            filtered_demos = {
                "mean_age": np.mean([d["demographics"]["mean_age"] for d in filtered.values()]),
                "pos_rate": np.mean([d["demographics"]["pos_rate"] for d in filtered.values()])
            }
            demo_shift = {
                "mean_age_shift": filtered_demos["mean_age"] - original_demos["mean_age"],
                "pos_rate_shift": filtered_demos["pos_rate"] - original_demos["pos_rate"]
            }

            result = SimulationResult(
                scenario_name=f"uniform_{int(rate*100)}pct",
                opt_out_rate=rate,
                original_samples=total_original,
                remaining_samples=total_remaining,
                accuracy=avg_acc,
                accuracy_drop=baseline_acc - avg_acc,
                f1_score=avg_f1,
                auc_roc=avg_auc,
                per_hospital_samples={n: d["remaining_samples"] for n, d in filtered.items()},
                per_hospital_accuracy={n: np.mean(per_hospital_accs[n]) for n in self.hospital_names},
                demographic_distribution_shift=demo_shift,
                convergence_rounds=avg_conv,
                model_viable=avg_acc >= self.viability_threshold
            )
            results.append(result)

            status = "✓" if result.model_viable else "✗"
            print(f"Opt-out {rate:.0%}: Acc={avg_acc:.2%} (drop={result.accuracy_drop:.1%}) {status}")

        # Find critical threshold
        critical_threshold = 1.0
        for r in results:
            if not r.model_viable:
                critical_threshold = r.opt_out_rate
                break

        # Generate recommendations
        recommendations = self._generate_recommendations(results, critical_threshold, baseline_acc)

        return SimulationSummary(
            timestamp=datetime.now().isoformat(),
            num_hospitals=self.num_hospitals,
            baseline_accuracy=baseline_acc,
            critical_optout_threshold=critical_threshold,
            results=results,
            recommendations=recommendations
        )

    def _generate_recommendations(
        self,
        results: List[SimulationResult],
        critical_threshold: float,
        baseline_acc: float
    ) -> List[str]:
        """Generate HDAB recommendations based on simulation results."""
        recommendations = []

        # Critical threshold warning
        if critical_threshold < 0.3:
            recommendations.append(
                f"⚠️ CRITICAL: Model becomes non-viable at {critical_threshold:.0%} opt-out. "
                "Aggressive citizen engagement required."
            )
        elif critical_threshold < 0.5:
            recommendations.append(
                f"⚡ WARNING: Critical opt-out threshold is {critical_threshold:.0%}. "
                "Monitor opt-out rates closely."
            )
        else:
            recommendations.append(
                f"✓ Model is robust up to {critical_threshold:.0%} opt-out rate."
            )

        # Per-hospital analysis
        if results:
            last_result = results[-1]
            min_samples = min(last_result.per_hospital_samples.values())
            if min_samples < 50:
                recommendations.append(
                    f"⚠️ Some hospitals may fall below minimum viable sample size (current min: {min_samples}). "
                    "Consider regional data pooling."
                )

        # Accuracy degradation gradient
        if len(results) >= 2:
            acc_drop_10pct = next((r.accuracy_drop for r in results if r.opt_out_rate >= 0.10), 0)
            if acc_drop_10pct > 0.03:
                recommendations.append(
                    f"📉 Significant accuracy drop ({acc_drop_10pct:.1%}) at 10% opt-out. "
                    "Consider differential privacy to build trust."
                )

        # Positive recommendations
        recommendations.append(
            "📊 Publish opt-out statistics transparently to build citizen trust "
            "(EHDS Article 71 compliance)."
        )
        recommendations.append(
            "🔒 Emphasize FL's privacy-preserving nature: data never leaves hospitals."
        )

        return recommendations

    def plot_results(self, summary: SimulationSummary, save_path: Optional[str] = None):
        """Generate visualization of simulation results."""
        if not MATPLOTLIB_AVAILABLE:
            print("Matplotlib not available. Install with: pip install matplotlib")
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle("EHDS Article 71 Opt-Out Impact Analysis", fontsize=14, fontweight='bold')

        rates = [r.opt_out_rate for r in summary.results]
        accs = [r.accuracy for r in summary.results]
        f1s = [r.f1_score for r in summary.results]
        samples = [r.remaining_samples for r in summary.results]

        # 1. Accuracy vs Opt-out Rate
        ax1 = axes[0, 0]
        ax1.plot(rates, accs, 'b-o', linewidth=2, markersize=8)
        ax1.axhline(y=summary.baseline_accuracy, color='g', linestyle='--', label='Baseline')
        ax1.axhline(y=self.viability_threshold, color='r', linestyle='--', label='Viability Threshold')
        ax1.axvline(x=summary.critical_optout_threshold, color='orange', linestyle=':', label='Critical Threshold')
        ax1.fill_between(rates, self.viability_threshold, max(accs), alpha=0.2, color='green')
        ax1.fill_between(rates, 0, self.viability_threshold, alpha=0.2, color='red')
        ax1.set_xlabel("Opt-out Rate")
        ax1.set_ylabel("Accuracy")
        ax1.set_title("Model Accuracy vs Opt-out Rate")
        ax1.legend(loc='lower left')
        ax1.set_xlim(0, max(rates))
        ax1.set_ylim(0.4, max(accs) + 0.05)
        ax1.grid(True, alpha=0.3)

        # 2. Accuracy Drop
        ax2 = axes[0, 1]
        drops = [r.accuracy_drop * 100 for r in summary.results]
        colors = ['green' if d < 3 else 'orange' if d < 5 else 'red' for d in drops]
        ax2.bar(range(len(rates)), drops, color=colors)
        ax2.set_xticks(range(len(rates)))
        ax2.set_xticklabels([f"{r:.0%}" for r in rates], rotation=45)
        ax2.set_xlabel("Opt-out Rate")
        ax2.set_ylabel("Accuracy Drop (pp)")
        ax2.set_title("Accuracy Degradation")
        ax2.grid(True, alpha=0.3, axis='y')

        # 3. Remaining Samples
        ax3 = axes[1, 0]
        original = summary.results[0].original_samples
        ax3.fill_between(rates, samples, alpha=0.6, color='blue')
        ax3.axhline(y=original, color='g', linestyle='--', label=f'Original ({original})')
        ax3.axhline(y=50 * self.num_hospitals, color='r', linestyle='--', label='Minimum viable')
        ax3.set_xlabel("Opt-out Rate")
        ax3.set_ylabel("Total Samples")
        ax3.set_title("Remaining Training Data")
        ax3.legend()
        ax3.set_xlim(0, max(rates))
        ax3.grid(True, alpha=0.3)

        # 4. Per-Hospital Accuracy at Different Opt-out Rates
        ax4 = axes[1, 1]
        selected_rates = [0.0, 0.1, 0.3, 0.5]
        selected_results = [r for r in summary.results if r.opt_out_rate in selected_rates]
        x = np.arange(self.num_hospitals)
        width = 0.2
        for i, result in enumerate(selected_results):
            accs = [result.per_hospital_accuracy.get(name, 0) for name in self.hospital_names]
            ax4.bar(x + i * width, accs, width, label=f"{result.opt_out_rate:.0%} opt-out")
        ax4.set_xticks(x + width * 1.5)
        ax4.set_xticklabels([n.split('-')[1] for n in self.hospital_names], rotation=45)
        ax4.set_ylabel("Accuracy")
        ax4.set_title("Per-Hospital Accuracy")
        ax4.legend(loc='lower right')
        ax4.set_ylim(0.4, 0.75)
        ax4.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
            plt.close(fig)
        else:
            plt.show()

    def save_results(self, summary: SimulationSummary, filepath: str):
        """Save simulation results to JSON."""

        def convert_numpy(obj):
            """Convert numpy types to Python native types."""
            if isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(i) for i in obj]
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        data = {
            "timestamp": summary.timestamp,
            "num_hospitals": summary.num_hospitals,
            "baseline_accuracy": summary.baseline_accuracy,
            "critical_optout_threshold": summary.critical_optout_threshold,
            "viability_threshold": self.viability_threshold,
            "recommendations": summary.recommendations,
            "results": [convert_numpy(asdict(r)) for r in summary.results]
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Results saved to: {filepath}")


def main():
    """Run the opt-out impact simulation."""
    print("=" * 60)
    print("EHDS Article 71 Opt-Out Impact Simulator")
    print("=" * 60)
    print()

    # Create simulator
    simulator = OptOutImpactSimulator(
        num_hospitals=5,
        samples_per_hospital=[400, 500, 350, 450, 380],
        viability_threshold=0.55
    )

    # Run simulation
    print("Running simulation across opt-out rates...")
    print("-" * 60)

    summary = simulator.run_simulation(
        opt_out_rates=[0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70],
        num_trials=3
    )

    print("-" * 60)
    print()

    # Print recommendations
    print("HDAB RECOMMENDATIONS:")
    print("-" * 60)
    for rec in summary.recommendations:
        print(f"  {rec}")
    print()

    # Save results
    output_dir = Path(__file__).parent.parent / "benchmarks" / "results_optout"
    output_dir.mkdir(parents=True, exist_ok=True)

    simulator.save_results(summary, str(output_dir / "optout_impact_results.json"))

    # Plot if available
    if MATPLOTLIB_AVAILABLE:
        simulator.plot_results(summary, str(output_dir / "optout_impact_analysis.pdf"))

    print("=" * 60)
    print(f"Critical opt-out threshold: {summary.critical_optout_threshold:.0%}")
    print(f"Baseline accuracy: {summary.baseline_accuracy:.2%}")
    print("=" * 60)


if __name__ == "__main__":
    main()

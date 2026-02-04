#!/usr/bin/env python3
"""
FL-EHDS Heterogeneity and Participation Analysis

Generates comprehensive visualizations for:
1. Node participation patterns per round
2. Training time distribution per node
3. Non-IID data distribution visualization
4. Statistical heterogeneity metrics (KL divergence, Earth Mover's Distance)
5. Label distribution skew analysis
6. Feature distribution comparison

Author: Fabio Liberti
"""

import numpy as np
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import warnings

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

# Set style
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3


class HeterogeneityAnalyzer:
    """Analyzes and visualizes statistical heterogeneity in FL settings."""

    def __init__(self, num_hospitals: int = 5, random_seed: int = 42):
        self.num_hospitals = num_hospitals
        self.random_seed = random_seed
        np.random.seed(random_seed)

        self.hospital_names = [
            "IT-Roma", "DE-Berlin", "FR-Paris", "ES-Madrid", "NL-Amsterdam"
        ][:num_hospitals]

        self.hospital_colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']

        # Generate heterogeneous data
        self._generate_hospital_data()

    def _generate_hospital_data(self):
        """Generate synthetic healthcare data with controlled heterogeneity."""
        self.hospital_data = {}

        # Different demographic parameters per hospital (geographic variation)
        hospital_params = [
            {"n_samples": 400, "mean_age": 48, "age_std": 12, "pos_rate": 0.38, "bmi_mean": 24.5},
            {"n_samples": 520, "mean_age": 54, "age_std": 14, "pos_rate": 0.44, "bmi_mean": 26.0},
            {"n_samples": 340, "mean_age": 51, "age_std": 16, "pos_rate": 0.41, "bmi_mean": 25.2},
            {"n_samples": 480, "mean_age": 59, "age_std": 11, "pos_rate": 0.56, "bmi_mean": 27.5},
            {"n_samples": 390, "mean_age": 62, "age_std": 9, "pos_rate": 0.63, "bmi_mean": 26.8},
        ][:self.num_hospitals]

        for name, params in zip(self.hospital_names, hospital_params):
            n = params["n_samples"]

            # Generate features with hospital-specific distributions
            age = np.random.normal(params["mean_age"], params["age_std"], n)
            age = np.clip(age, 18, 95)

            bmi = np.random.normal(params["bmi_mean"], 4.5, n)
            bmi = np.clip(bmi, 15, 50)

            systolic = np.random.normal(120 + (params["mean_age"] - 50) * 0.5, 18, n)
            systolic = np.clip(systolic, 80, 200)

            glucose = np.random.normal(95 + (params["bmi_mean"] - 25) * 3, 22, n)
            glucose = np.clip(glucose, 60, 300)

            cholesterol = np.random.normal(190 + (params["mean_age"] - 50) * 1.2, 35, n)
            cholesterol = np.clip(cholesterol, 100, 400)

            # Generate labels with hospital-specific positive rates
            risk_score = (
                0.025 * (age - 50) +
                0.035 * (bmi - 25) +
                0.012 * (systolic - 120) +
                0.006 * (glucose - 100) +
                0.003 * (cholesterol - 200)
            )
            base_prob = params["pos_rate"]
            prob = 1 / (1 + np.exp(-risk_score + np.log((1 - base_prob) / base_prob)))
            y = (np.random.random(n) < prob).astype(int)

            self.hospital_data[name] = {
                "age": age,
                "bmi": bmi,
                "systolic": systolic,
                "glucose": glucose,
                "cholesterol": cholesterol,
                "labels": y,
                "n_samples": n,
                "params": params
            }

    def simulate_training(self, num_rounds: int = 50) -> Dict:
        """Simulate FL training and collect metrics."""
        results = {
            "participation_matrix": [],
            "training_times": {name: [] for name in self.hospital_names},
            "gradient_norms": {name: [] for name in self.hospital_names},
            "round_accuracies": [],
        }

        for round_num in range(num_rounds):
            # Simulate participation (some clients may drop out)
            participation = []
            for i, name in enumerate(self.hospital_names):
                # Participation probability varies by hospital
                base_prob = 0.85 + i * 0.02
                participates = np.random.random() < base_prob
                participation.append(participates)

                if participates:
                    # Simulate training time (proportional to dataset size + noise)
                    base_time = self.hospital_data[name]["n_samples"] * 0.001
                    time_noise = np.random.exponential(0.02)
                    training_time = base_time + time_noise
                    results["training_times"][name].append(training_time)

                    # Simulate gradient norm (decreasing over rounds with variation)
                    base_norm = 0.5 * np.exp(-0.03 * round_num) + 0.1
                    norm_noise = np.random.normal(0, 0.05)
                    gradient_norm = max(0.01, base_norm + norm_noise)
                    results["gradient_norms"][name].append(gradient_norm)
                else:
                    results["training_times"][name].append(0)
                    results["gradient_norms"][name].append(0)

            results["participation_matrix"].append(participation)

            # Simulate accuracy (improving over rounds)
            acc = 0.52 + 0.08 * (1 - np.exp(-0.05 * round_num)) + np.random.normal(0, 0.01)
            results["round_accuracies"].append(acc)

        return results

    def compute_heterogeneity_metrics(self) -> Dict:
        """Compute statistical heterogeneity metrics."""
        metrics = {}

        # 1. Label distribution (positive rate) per hospital
        label_distributions = {}
        for name, data in self.hospital_data.items():
            pos_rate = np.mean(data["labels"])
            label_distributions[name] = {"positive": pos_rate, "negative": 1 - pos_rate}
        metrics["label_distributions"] = label_distributions

        # 2. Feature statistics per hospital
        feature_stats = {}
        for name, data in self.hospital_data.items():
            feature_stats[name] = {
                "age": {"mean": np.mean(data["age"]), "std": np.std(data["age"])},
                "bmi": {"mean": np.mean(data["bmi"]), "std": np.std(data["bmi"])},
                "systolic": {"mean": np.mean(data["systolic"]), "std": np.std(data["systolic"])},
                "glucose": {"mean": np.mean(data["glucose"]), "std": np.std(data["glucose"])},
                "cholesterol": {"mean": np.mean(data["cholesterol"]), "std": np.std(data["cholesterol"])},
            }
        metrics["feature_stats"] = feature_stats

        # 3. KL Divergence between label distributions
        global_pos_rate = np.mean([np.mean(d["labels"]) for d in self.hospital_data.values()])
        kl_divergences = {}
        for name, data in self.hospital_data.items():
            local_pos = np.mean(data["labels"])
            # KL(local || global)
            eps = 1e-10
            kl = local_pos * np.log((local_pos + eps) / (global_pos_rate + eps)) + \
                 (1 - local_pos) * np.log((1 - local_pos + eps) / (1 - global_pos_rate + eps))
            kl_divergences[name] = kl
        metrics["kl_divergences"] = kl_divergences

        # 4. Earth Mover's Distance for age distributions
        global_age = np.concatenate([d["age"] for d in self.hospital_data.values()])
        emd_distances = {}
        for name, data in self.hospital_data.items():
            # Simplified EMD using mean difference
            emd = abs(np.mean(data["age"]) - np.mean(global_age))
            emd_distances[name] = emd
        metrics["emd_age"] = emd_distances

        # 5. Non-IID score (combined metric)
        noniid_scores = {}
        for name in self.hospital_names:
            score = (
                abs(kl_divergences[name]) * 10 +
                emd_distances[name] / 5
            )
            noniid_scores[name] = score
        metrics["noniid_scores"] = noniid_scores

        return metrics

    def plot_participation_heatmap(self, results: Dict, save_path: str):
        """Plot participation matrix as heatmap."""
        fig, ax = plt.subplots(figsize=(14, 6))

        matrix = np.array(results["participation_matrix"]).T
        rounds = matrix.shape[1]

        # Custom colormap
        cmap = LinearSegmentedColormap.from_list("participation", ["#ffcccc", "#28a745"])

        im = ax.imshow(matrix, cmap=cmap, aspect='auto', interpolation='nearest')

        ax.set_yticks(range(len(self.hospital_names)))
        ax.set_yticklabels(self.hospital_names)
        ax.set_xlabel("Training Round", fontsize=12)
        ax.set_ylabel("Hospital", fontsize=12)
        ax.set_title("Client Participation Matrix (50 Rounds)", fontsize=14, fontweight='bold')

        # Add participation rates
        for i, name in enumerate(self.hospital_names):
            rate = np.mean(matrix[i]) * 100
            ax.text(rounds + 1, i, f"{rate:.0f}%", va='center', fontsize=11, fontweight='bold')

        # Add legend
        legend_elements = [
            mpatches.Patch(facecolor='#28a745', label='Participated'),
            mpatches.Patch(facecolor='#ffcccc', label='Dropped')
        ]
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {save_path}")

    def plot_training_times(self, results: Dict, save_path: str):
        """Plot training time distribution per node."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left: Box plot
        ax1 = axes[0]
        times_data = []
        labels = []
        for name in self.hospital_names:
            times = [t for t in results["training_times"][name] if t > 0]
            if times:
                times_data.append(times)
                labels.append(name)

        bp = ax1.boxplot(times_data, labels=labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], self.hospital_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax1.set_ylabel("Training Time (seconds)", fontsize=12)
        ax1.set_xlabel("Hospital", fontsize=12)
        ax1.set_title("Training Time Distribution per Node", fontsize=14, fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)

        # Right: Time series
        ax2 = axes[1]
        for i, name in enumerate(self.hospital_names):
            times = results["training_times"][name]
            rounds = range(len(times))
            # Only plot non-zero (participating rounds)
            participating = [(r, t) for r, t in zip(rounds, times) if t > 0]
            if participating:
                r, t = zip(*participating)
                ax2.scatter(r, t, label=name, alpha=0.6, s=20, color=self.hospital_colors[i])

        ax2.set_xlabel("Round", fontsize=12)
        ax2.set_ylabel("Training Time (s)", fontsize=12)
        ax2.set_title("Training Time per Round", fontsize=14, fontweight='bold')
        ax2.legend(loc='upper right', fontsize=9)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {save_path}")

    def plot_data_distribution(self, save_path: str):
        """Plot non-IID data distribution across nodes."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        features = ['age', 'bmi', 'systolic', 'glucose', 'cholesterol']
        titles = ['Age Distribution', 'BMI Distribution', 'Systolic BP Distribution',
                  'Glucose Distribution', 'Cholesterol Distribution']

        for idx, (feature, title) in enumerate(zip(features, titles)):
            ax = axes[idx // 3, idx % 3]

            for i, name in enumerate(self.hospital_names):
                data = self.hospital_data[name][feature]
                ax.hist(data, bins=25, alpha=0.5, label=name, color=self.hospital_colors[i], density=True)

            ax.set_xlabel(feature.capitalize(), fontsize=11)
            ax.set_ylabel("Density", fontsize=11)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.legend(fontsize=8)

        # Last subplot: Label distribution
        ax = axes[1, 2]
        x = np.arange(len(self.hospital_names))
        width = 0.35

        pos_rates = [np.mean(self.hospital_data[name]["labels"]) for name in self.hospital_names]
        neg_rates = [1 - p for p in pos_rates]

        bars1 = ax.bar(x - width/2, pos_rates, width, label='Positive', color='#e74c3c', alpha=0.8)
        bars2 = ax.bar(x + width/2, neg_rates, width, label='Negative', color='#3498db', alpha=0.8)

        ax.set_ylabel('Proportion', fontsize=11)
        ax.set_xlabel('Hospital', fontsize=11)
        ax.set_title('Label Distribution (Non-IID)', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([n.split('-')[0] for n in self.hospital_names], rotation=45)
        ax.legend()

        # Add percentage labels
        for bar, rate in zip(bars1, pos_rates):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{rate:.0%}', ha='center', fontsize=9)

        plt.suptitle("Non-IID Data Distribution Across Hospitals", fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {save_path}")

    def plot_statistical_heterogeneity(self, metrics: Dict, save_path: str):
        """Plot statistical heterogeneity metrics."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # 1. KL Divergence
        ax1 = axes[0, 0]
        kl_values = list(metrics["kl_divergences"].values())
        bars = ax1.bar(self.hospital_names, kl_values, color=self.hospital_colors, alpha=0.8)
        ax1.set_ylabel("KL Divergence", fontsize=12)
        ax1.set_xlabel("Hospital", fontsize=12)
        ax1.set_title("KL Divergence from Global Label Distribution", fontsize=13, fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        for bar, val in zip(bars, kl_values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                    f'{val:.4f}', ha='center', fontsize=10)

        # 2. Earth Mover's Distance (Age)
        ax2 = axes[0, 1]
        emd_values = list(metrics["emd_age"].values())
        bars = ax2.bar(self.hospital_names, emd_values, color=self.hospital_colors, alpha=0.8)
        ax2.set_ylabel("EMD (years)", fontsize=12)
        ax2.set_xlabel("Hospital", fontsize=12)
        ax2.set_title("Earth Mover's Distance (Age vs Global)", fontsize=13, fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        for bar, val in zip(bars, emd_values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{val:.1f}', ha='center', fontsize=10)

        # 3. Feature Mean Comparison (Radar-like bar chart)
        ax3 = axes[1, 0]
        features = ['age', 'bmi', 'systolic', 'glucose', 'cholesterol']
        x = np.arange(len(features))
        width = 0.15

        for i, name in enumerate(self.hospital_names):
            stats = metrics["feature_stats"][name]
            # Normalize to percentage of global mean
            global_means = {f: np.mean([metrics["feature_stats"][n][f]["mean"]
                                        for n in self.hospital_names]) for f in features}
            normalized = [(stats[f]["mean"] / global_means[f] - 1) * 100 for f in features]
            ax3.bar(x + i * width, normalized, width, label=name, color=self.hospital_colors[i], alpha=0.8)

        ax3.set_ylabel("Deviation from Global Mean (%)", fontsize=12)
        ax3.set_xlabel("Feature", fontsize=12)
        ax3.set_title("Feature Distribution Heterogeneity", fontsize=13, fontweight='bold')
        ax3.set_xticks(x + width * 2)
        ax3.set_xticklabels([f.capitalize() for f in features])
        ax3.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax3.legend(loc='upper right', fontsize=9)

        # 4. Non-IID Score Summary
        ax4 = axes[1, 1]
        noniid_scores = list(metrics["noniid_scores"].values())
        bars = ax4.barh(self.hospital_names, noniid_scores, color=self.hospital_colors, alpha=0.8)
        ax4.set_xlabel("Non-IID Score (higher = more heterogeneous)", fontsize=12)
        ax4.set_title("Combined Non-IID Heterogeneity Score", fontsize=13, fontweight='bold')

        # Add score labels
        for bar, score in zip(bars, noniid_scores):
            ax4.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                    f'{score:.2f}', va='center', fontsize=11, fontweight='bold')

        # Color code by severity
        avg_score = np.mean(noniid_scores)
        ax4.axvline(x=avg_score, color='red', linestyle='--', linewidth=2, label=f'Mean: {avg_score:.2f}')
        ax4.legend()

        plt.suptitle("Statistical Heterogeneity Analysis", fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {save_path}")

    def plot_label_skew_analysis(self, save_path: str):
        """Detailed label skew visualization."""
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        # 1. Stacked bar chart of label distribution
        ax1 = axes[0]
        pos_rates = [np.mean(self.hospital_data[name]["labels"]) for name in self.hospital_names]
        neg_rates = [1 - p for p in pos_rates]

        ax1.bar(self.hospital_names, pos_rates, label='Positive (High Risk)', color='#e74c3c', alpha=0.8)
        ax1.bar(self.hospital_names, neg_rates, bottom=pos_rates, label='Negative (Low Risk)', color='#3498db', alpha=0.8)

        ax1.set_ylabel("Proportion", fontsize=12)
        ax1.set_xlabel("Hospital", fontsize=12)
        ax1.set_title("Label Distribution per Hospital", fontsize=13, fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        ax1.legend(loc='upper right')

        # Add global average line
        global_pos = np.mean(pos_rates)
        ax1.axhline(y=global_pos, color='green', linestyle='--', linewidth=2, label=f'Global Avg: {global_pos:.0%}')

        # 2. Sample count vs positive rate scatter
        ax2 = axes[1]
        sample_counts = [self.hospital_data[name]["n_samples"] for name in self.hospital_names]

        scatter = ax2.scatter(sample_counts, pos_rates, s=200, c=self.hospital_colors, alpha=0.8, edgecolors='black')

        for i, name in enumerate(self.hospital_names):
            ax2.annotate(name, (sample_counts[i], pos_rates[i]),
                        xytext=(10, 5), textcoords='offset points', fontsize=10)

        ax2.set_xlabel("Sample Count", fontsize=12)
        ax2.set_ylabel("Positive Rate", fontsize=12)
        ax2.set_title("Sample Size vs Label Imbalance", fontsize=13, fontweight='bold')

        # 3. Cumulative sample contribution by label
        ax3 = axes[2]
        sorted_hospitals = sorted(self.hospital_names,
                                  key=lambda x: self.hospital_data[x]["n_samples"], reverse=True)

        cumulative_pos = []
        cumulative_neg = []
        cum_pos = 0
        cum_neg = 0

        for name in sorted_hospitals:
            data = self.hospital_data[name]
            pos_count = np.sum(data["labels"])
            neg_count = len(data["labels"]) - pos_count
            cum_pos += pos_count
            cum_neg += neg_count
            cumulative_pos.append(cum_pos)
            cumulative_neg.append(cum_neg)

        x = range(1, len(sorted_hospitals) + 1)
        ax3.plot(x, cumulative_pos, 'o-', color='#e74c3c', linewidth=2, markersize=8, label='Positive Samples')
        ax3.plot(x, cumulative_neg, 's-', color='#3498db', linewidth=2, markersize=8, label='Negative Samples')

        ax3.set_xlabel("Number of Hospitals", fontsize=12)
        ax3.set_ylabel("Cumulative Samples", fontsize=12)
        ax3.set_title("Cumulative Sample Distribution", fontsize=13, fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(sorted_hospitals, rotation=45)
        ax3.legend()

        plt.suptitle("Label Skew Analysis (Non-IID Labels)", fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {save_path}")

    def generate_all_plots(self, output_dir: Path):
        """Generate all heterogeneity analysis plots."""
        output_dir.mkdir(parents=True, exist_ok=True)

        print("Simulating FL training...")
        results = self.simulate_training(num_rounds=50)

        print("Computing heterogeneity metrics...")
        metrics = self.compute_heterogeneity_metrics()

        print("\nGenerating plots...")

        # 1. Participation heatmap
        self.plot_participation_heatmap(results, str(output_dir / "fig_participation_heatmap.pdf"))

        # 2. Training times
        self.plot_training_times(results, str(output_dir / "fig_training_times.pdf"))

        # 3. Data distribution (non-IID)
        self.plot_data_distribution(str(output_dir / "fig_data_distribution_noniid.pdf"))

        # 4. Statistical heterogeneity
        self.plot_statistical_heterogeneity(metrics, str(output_dir / "fig_statistical_heterogeneity.pdf"))

        # 5. Label skew analysis
        self.plot_label_skew_analysis(str(output_dir / "fig_label_skew_analysis.pdf"))

        # Save metrics to JSON
        metrics_serializable = {
            "label_distributions": metrics["label_distributions"],
            "kl_divergences": {k: float(v) for k, v in metrics["kl_divergences"].items()},
            "emd_age": {k: float(v) for k, v in metrics["emd_age"].items()},
            "noniid_scores": {k: float(v) for k, v in metrics["noniid_scores"].items()},
            "feature_stats": {
                name: {f: {"mean": float(s["mean"]), "std": float(s["std"])}
                       for f, s in stats.items()}
                for name, stats in metrics["feature_stats"].items()
            }
        }

        with open(output_dir / "heterogeneity_metrics.json", 'w') as f:
            json.dump(metrics_serializable, f, indent=2)
        print(f"Saved: {output_dir / 'heterogeneity_metrics.json'}")

        return metrics


def main():
    print("=" * 60)
    print("FL-EHDS Heterogeneity & Participation Analysis")
    print("=" * 60)
    print()

    output_dir = Path(__file__).parent / "results_heterogeneity"

    analyzer = HeterogeneityAnalyzer(num_hospitals=5)
    metrics = analyzer.generate_all_plots(output_dir)

    print("\n" + "=" * 60)
    print("HETEROGENEITY SUMMARY")
    print("=" * 60)

    print("\nLabel Distribution (Positive Rate):")
    for name in analyzer.hospital_names:
        rate = metrics["label_distributions"][name]["positive"]
        print(f"  {name}: {rate:.1%}")

    print("\nNon-IID Scores (higher = more heterogeneous):")
    for name, score in metrics["noniid_scores"].items():
        print(f"  {name}: {score:.3f}")

    print("\n" + "=" * 60)
    print(f"All plots saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()

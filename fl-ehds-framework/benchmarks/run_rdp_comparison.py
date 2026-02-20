#!/usr/bin/env python3
"""
FL-EHDS Test C — RDP vs Naive Composition Comparison.

Generates a figure comparing cumulative epsilon under:
  1. Naive (simple) composition: ε_total = T × ε_per_round
  2. Advanced composition: ε_total = sqrt(2T ln(1/δ')) × ε + T × ε × (e^ε - 1)
  3. RDP (Rényi) composition: tightest bounds via moments accountant

Uses the same parameters as the paper: PTB-XL with 30 rounds, ε=10, δ=1e-5.

This is a PURELY ANALYTICAL computation — no model training required.
Runtime: ~10 seconds.

Usage:
    cd fl-ehds-framework
    python -m benchmarks.run_rdp_comparison

Output:
    benchmarks/paper_results_tabular/rdp_vs_naive_composition.pdf
    benchmarks/paper_results_tabular/rdp_vs_naive_composition.png

Author: Fabio Liberti
"""

import sys
import os
from pathlib import Path
import numpy as np

FRAMEWORK_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(FRAMEWORK_DIR))

from orchestration.privacy.differential_privacy import (
    compute_rdp_gaussian,
    compute_rdp_gaussian_subsampled,
    rdp_to_eps_delta,
    DEFAULT_RDP_ORDERS,
)

OUTPUT_DIR = FRAMEWORK_DIR / "benchmarks" / "paper_results_tabular"

# Paper parameters
TARGET_EPSILON = 10.0
DELTA = 1e-5
NUM_ROUNDS = 30
SIGMA = 1.1  # Noise multiplier (calibrated for ε≈10 over 30 rounds with RDP)
SAMPLING_RATE = 1.0  # Full participation (K/K = 1)
MAX_ROUNDS_PLOT = 60  # Extended to show where budgets are exhausted


def naive_composition(eps_per_round, T):
    """Simple composition: ε_total = T × ε"""
    return T * eps_per_round


def advanced_composition(eps_per_round, delta, T, delta_prime_frac=0.5):
    """
    Advanced composition theorem (Dwork, Rothblum, Vadhan 2010):
    ε_total = sqrt(2T ln(1/δ')) × ε + T × ε × (e^ε - 1)
    where δ_total = T × δ + δ'
    """
    delta_prime = delta * delta_prime_frac
    term1 = np.sqrt(2 * T * np.log(1 / delta_prime)) * eps_per_round
    term2 = T * eps_per_round * (np.exp(eps_per_round) - 1)
    return term1 + term2


def rdp_composition(sigma, T, delta, sampling_rate=1.0, orders=None):
    """
    RDP composition: compose T rounds in Rényi divergence, convert to (ε,δ)-DP.
    """
    if orders is None:
        orders = DEFAULT_RDP_ORDERS

    rdp_per_round = []
    for alpha in orders:
        if sampling_rate < 1.0:
            rdp = compute_rdp_gaussian_subsampled(sigma, alpha, sampling_rate)
        else:
            rdp = compute_rdp_gaussian(sigma, alpha)
        rdp_per_round.append(rdp)

    # Compose T rounds (RDP composes additively)
    total_rdp = [r * T for r in rdp_per_round]

    # Convert to (ε, δ)-DP
    epsilon, best_order = rdp_to_eps_delta(total_rdp, orders, delta)
    return epsilon, best_order


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  FL-EHDS RDP vs Naive Composition Analysis")
    print("  σ={}, δ={}, T={}".format(SIGMA, DELTA, NUM_ROUNDS))
    print("=" * 60)

    # First, compute per-round epsilon for naive/advanced
    # For Gaussian mechanism: ε_per_round = sqrt(2 ln(1.25/δ)) / σ
    eps_per_round = np.sqrt(2 * np.log(1.25 / DELTA)) / SIGMA
    print("\n  Per-round ε (Gaussian): {:.4f}".format(eps_per_round))

    rounds = np.arange(1, MAX_ROUNDS_PLOT + 1)

    # Compute cumulative epsilon for each method
    naive_eps = np.array([naive_composition(eps_per_round, T) for T in rounds])
    advanced_eps = np.array([advanced_composition(eps_per_round, DELTA, T) for T in rounds])
    rdp_eps = np.array([rdp_composition(SIGMA, T, DELTA, SAMPLING_RATE)[0] for T in rounds])

    # Print comparison at key points
    print("\n  {:<6s} | {:>12s} | {:>12s} | {:>12s} | {:>10s}".format(
        "Rounds", "Naive", "Advanced", "RDP", "RDP tighter"))
    print("  " + "-" * 60)
    for T in [1, 5, 10, 15, 20, 25, 30, 40, 50, 60]:
        if T > MAX_ROUNDS_PLOT:
            break
        idx = T - 1
        ratio = naive_eps[idx] / rdp_eps[idx] if rdp_eps[idx] > 0 else float('inf')
        print("  {:>6d} | {:>10.2f}  | {:>10.2f}  | {:>10.2f}  | {:>8.1f}x".format(
            T, naive_eps[idx], advanced_eps[idx], rdp_eps[idx], ratio))

    # At T=30 (paper setting)
    idx_30 = 29
    print("\n  At T=30 (paper setting):")
    print("    Naive:    ε = {:.2f}".format(naive_eps[idx_30]))
    print("    Advanced: ε = {:.2f}".format(advanced_eps[idx_30]))
    print("    RDP:      ε = {:.2f}".format(rdp_eps[idx_30]))
    print("    RDP tighter: {:.1f}x vs naive, {:.1f}x vs advanced".format(
        naive_eps[idx_30] / rdp_eps[idx_30],
        advanced_eps[idx_30] / rdp_eps[idx_30],
    ))

    # Find how many rounds each method allows before exceeding ε=10
    for name, eps_curve in [("Naive", naive_eps), ("Advanced", advanced_eps), ("RDP", rdp_eps)]:
        exceeded = np.where(eps_curve > TARGET_EPSILON)[0]
        if len(exceeded) > 0:
            max_rounds = exceeded[0]  # 0-indexed, so this is the first round exceeding budget
            print("    {} exceeds ε={} at round {}".format(name, TARGET_EPSILON, max_rounds + 1))
        else:
            print("    {} stays under ε={} for all {} rounds".format(name, TARGET_EPSILON, MAX_ROUNDS_PLOT))

    # With subsampling (q=0.2, simulating partial client participation)
    print("\n  With subsampling q=0.2 (partial participation):")
    rdp_sub_eps = np.array([rdp_composition(SIGMA, T, DELTA, 0.2)[0] for T in rounds])
    print("    RDP (q=1.0) at T=30: ε = {:.2f}".format(rdp_eps[idx_30]))
    print("    RDP (q=0.2) at T=30: ε = {:.2f}".format(rdp_sub_eps[idx_30]))
    print("    Amplification factor: {:.1f}x".format(rdp_eps[idx_30] / rdp_sub_eps[idx_30]))

    # Generate figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Left panel: Full comparison
        ax1.plot(rounds, naive_eps, "r--", linewidth=2, label="Naive Composition")
        ax1.plot(rounds, advanced_eps, "b-.", linewidth=2, label="Advanced Composition")
        ax1.plot(rounds, rdp_eps, "g-", linewidth=2.5, label="RDP (Moments Accountant)")
        ax1.axhline(y=TARGET_EPSILON, color="k", linestyle=":", alpha=0.5, label="Budget ε={}".format(int(TARGET_EPSILON)))
        ax1.axvline(x=NUM_ROUNDS, color="gray", linestyle=":", alpha=0.3)
        ax1.fill_between(rounds, 0, TARGET_EPSILON, alpha=0.05, color="green")

        ax1.set_xlabel("Training Rounds (T)", fontsize=12)
        ax1.set_ylabel("Cumulative ε", fontsize=12)
        ax1.set_title("Privacy Budget Consumption\n(σ={}, δ={})".format(SIGMA, DELTA), fontsize=13)
        ax1.legend(fontsize=10, loc="upper left")
        ax1.set_xlim(1, MAX_ROUNDS_PLOT)
        ax1.set_ylim(0, max(naive_eps[-1], 50))
        ax1.grid(True, alpha=0.3)

        # Annotate T=30
        ax1.annotate("T=30\nε_RDP={:.1f}\nε_naive={:.1f}".format(rdp_eps[idx_30], naive_eps[idx_30]),
                     xy=(30, rdp_eps[idx_30]), xytext=(38, rdp_eps[idx_30] + 10),
                     arrowprops=dict(arrowstyle="->", color="green"),
                     fontsize=9, color="green", fontweight="bold")

        # Right panel: Tightness ratio
        ratio_naive = naive_eps / rdp_eps
        ratio_advanced = advanced_eps / rdp_eps

        ax2.plot(rounds, ratio_naive, "r-", linewidth=2, label="Naive / RDP")
        ax2.plot(rounds, ratio_advanced, "b-", linewidth=2, label="Advanced / RDP")
        ax2.axhline(y=1, color="k", linestyle=":", alpha=0.3)
        ax2.axvline(x=NUM_ROUNDS, color="gray", linestyle=":", alpha=0.3)

        ax2.set_xlabel("Training Rounds (T)", fontsize=12)
        ax2.set_ylabel("Tightness Ratio (higher = RDP advantage)", fontsize=12)
        ax2.set_title("RDP Tightness Advantage", fontsize=13)
        ax2.legend(fontsize=10)
        ax2.set_xlim(1, MAX_ROUNDS_PLOT)
        ax2.grid(True, alpha=0.3)

        # Annotate T=30
        ax2.annotate("{:.1f}x tighter\nthan naive".format(ratio_naive[idx_30]),
                     xy=(30, ratio_naive[idx_30]), xytext=(38, ratio_naive[idx_30]),
                     arrowprops=dict(arrowstyle="->", color="red"),
                     fontsize=9, color="red", fontweight="bold")

        plt.tight_layout()

        fig_path = OUTPUT_DIR / "rdp_vs_naive_composition"
        plt.savefig(str(fig_path) + ".png", dpi=200, bbox_inches="tight")
        plt.savefig(str(fig_path) + ".pdf", bbox_inches="tight")
        print("\n  Figure saved: {}".format(fig_path))
        plt.close()

    except ImportError:
        print("\n  matplotlib not available — skipping figure")

    # Save numerical results
    results = {
        "parameters": {
            "sigma": SIGMA, "delta": DELTA, "target_epsilon": TARGET_EPSILON,
            "sampling_rate": SAMPLING_RATE,
        },
        "per_round_epsilon": eps_per_round,
        "at_T30": {
            "naive": float(naive_eps[idx_30]),
            "advanced": float(advanced_eps[idx_30]),
            "rdp": float(rdp_eps[idx_30]),
            "ratio_naive_rdp": float(naive_eps[idx_30] / rdp_eps[idx_30]),
            "ratio_advanced_rdp": float(advanced_eps[idx_30] / rdp_eps[idx_30]),
        },
        "timestamp": datetime.now().isoformat() if 'datetime' in dir() else None,
    }

    import json
    from datetime import datetime
    results["timestamp"] = datetime.now().isoformat()
    with open(OUTPUT_DIR / "rdp_comparison_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("  DONE.")
    print("=" * 60)


if __name__ == "__main__":
    main()

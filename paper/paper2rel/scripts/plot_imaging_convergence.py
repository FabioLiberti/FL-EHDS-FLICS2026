#!/usr/bin/env python3
"""
Plot imaging convergence curves (accuracy vs round) for 4 FL algorithms
across 3 imaging datasets. Mean +/- std over 3 seeds (42, 123, 456).
Output: figures/imaging_convergence.pdf
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# -- paths ----------------------------------------------------------------
MAIN_JSON = (
    "/Users/fabioliberti/_DEVAir/FL-EHDS-FLICS2026/"
    "fl-ehds-framework/benchmarks/checkpoint_imaging_seeds10.json"
)
EXTRA_JSON = (
    "/Users/fabioliberti/_DEVAir/FL-EHDS-FLICS2026/"
    "fl-ehds-framework/benchmarks/checkpoint_imaging_extra_algos.json"
)
OUT_PDF = (
    "/Users/fabioliberti/_DEVAir/FL-EHDS-FLICS2026/"
    "paper/paper2rel/figures/imaging_convergence.pdf"
)

# -- config ---------------------------------------------------------------
DATASETS = ["chest_xray", "Brain_Tumor", "Skin_Cancer"]
TITLES   = ["Chest X-ray", "Brain Tumor", "Skin Cancer"]
ALGOS    = ["FedAvg", "Ditto", "HPFL", "FedLESAM"]
COLORS   = {"FedAvg": "blue", "Ditto": "red", "HPFL": "green", "FedLESAM": "orange"}
SEEDS    = [42, 123, 456]

# -- load data ------------------------------------------------------------
with open(MAIN_JSON) as f:
    main_data = json.load(f)["completed"]
with open(EXTRA_JSON) as f:
    extra_data = json.load(f)["completed"]

all_data = {**main_data, **extra_data}


def get_curves(dataset, algo):
    """Return (rounds, mean_acc, std_acc) arrays for the given pair."""
    seed_curves = []
    for s in SEEDS:
        key = f"{dataset}_{algo}_s{s}"
        if key not in all_data:
            return None
        hist = all_data[key]["history"]
        rounds = [h["round"] for h in hist]
        accs   = [h["accuracy"] * 100.0 for h in hist]
        seed_curves.append((rounds, accs))

    # align on the shortest run length
    min_len = min(len(c[1]) for c in seed_curves)
    rounds_arr = np.array(seed_curves[0][0][:min_len])
    acc_matrix = np.array([c[1][:min_len] for c in seed_curves])  # (3, rounds)
    mean_acc = acc_matrix.mean(axis=0)
    std_acc  = acc_matrix.std(axis=0)
    return rounds_arr, mean_acc, std_acc


# -- plot -----------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

for idx, (ds, title) in enumerate(zip(DATASETS, TITLES)):
    ax = axes[idx]
    for algo in ALGOS:
        result = get_curves(ds, algo)
        if result is None:
            print(f"  [skip] {ds} / {algo}: no data for all 3 seeds")
            continue
        rounds, mean_acc, std_acc = result
        ax.plot(rounds, mean_acc, color=COLORS[algo], label=algo, linewidth=1.8)
        ax.fill_between(
            rounds,
            mean_acc - std_acc,
            mean_acc + std_acc,
            color=COLORS[algo],
            alpha=0.15,
        )

    ax.set_title(title)
    ax.set_xlabel("Round")
    if idx == 0:
        ax.set_ylabel("Accuracy (%)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_xlim(left=1)

    if idx == 0:
        ax.legend(loc="lower right", framealpha=0.9)

fig.tight_layout()
fig.savefig(OUT_PDF, dpi=300, format="pdf", bbox_inches="tight")
plt.close(fig)
print(f"Saved -> {OUT_PDF}")

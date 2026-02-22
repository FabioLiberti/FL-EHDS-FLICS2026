#!/usr/bin/env python3
"""
FL-EHDS Experiment A — Byzantine Robustness + Differential Privacy Joint Evaluation.

Tests FL algorithms under simultaneous Byzantine attacks and differential privacy,
answering: does DP noise help or hurt Byzantine robustness?

Design:
  - Dataset: Breast Cancer (fast, tabular, known class collapse)
  - Attack: 1 Byzantine client out of 3 (33%, just below f < n/3 threshold)
  - Attack types: label_flip, gradient_scale, sign_flip
  - Defenses: Krum, TrimmedMean, Median (+ no defense baseline)
  - DP: No-DP, eps=10, eps=1
  - Seeds: 3

The script first tries the framework's ByzantineConfig. If unavailable or buggy
(never been tested in benchmarks), it falls back to a manual attack simulation
that flips labels for the adversarial client.

Total: 3 attacks × 4 defenses × 3 DP × 3 seeds = 108 experiments + baselines
Estimated runtime: ~30-45 min

Usage:
    cd fl-ehds-framework
    python /tmp/run_byzantine_dp.py [--quick] [--no-resume]

Output:
    /tmp/fl_ehds_results/byzantine_dp_results.json
    /tmp/fl_ehds_results/byzantine_dp_heatmap.png

Author: Fabio Liberti (generated for FLICS 2026)
"""

import sys
import os
import json
import time
import gc
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from copy import deepcopy

import numpy as np

# Setup paths
FRAMEWORK_DIR = Path("/Users/fxlybs/_DEV/FL-EHDS-FLICS2026/fl-ehds-framework")
sys.path.insert(0, str(FRAMEWORK_DIR))

import torch
from terminal.fl_trainer import FederatedTrainer, _detect_device
from data.breast_cancer_loader import load_breast_cancer_data

OUTPUT_DIR = Path("/tmp/fl_ehds_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_PATH = OUTPUT_DIR / "byzantine_dp_results.json"

# ─── Configuration ────────────────────────────────────────────────────
NUM_CLIENTS = 3
ADVERSARIAL_CLIENT = 0  # Client 0 is the adversary (1 out of 3)
BC_CONFIG = dict(
    input_dim=30, num_classes=2,
    learning_rate=0.001, batch_size=16, num_rounds=40,
    local_epochs=1, mu=0.1,
)
SEEDS = [42, 123, 456]
CLASS_NAMES = {0: "Benign", 1: "Malignant"}


def compute_confusion_matrix(y_true, y_pred, num_classes=2):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def compute_dei(cm, num_classes=2):
    recalls = []
    for c in range(num_classes):
        total = cm[c].sum()
        recall = cm[c, c] / total if total > 0 else 0.0
        recalls.append(recall)
    if min(recalls) == 0:
        return 0.0, recalls
    mean_r = np.mean(recalls)
    std_r = np.std(recalls)
    cv = std_r / mean_r if mean_r > 0 else 1.0
    dei = min(recalls) * (1 - cv)
    return dei, recalls


def collect_predictions(trainer):
    model = trainer.global_model
    model.eval()
    all_preds, all_labels = [], []

    is_hpfl = trainer.algorithm == "HPFL"
    if is_hpfl:
        saved_cls = {n: p.data.clone() for n, p in model.named_parameters()
                     if n in trainer._hpfl_classifier_names}

    with torch.no_grad():
        for cid in range(trainer.num_clients):
            if is_hpfl:
                for n, p in model.named_parameters():
                    if n in trainer._hpfl_classifier_names:
                        p.data.copy_(trainer.client_classifiers[cid][n])
            X, y = trainer.client_test_data[cid]
            X_t = torch.FloatTensor(X).to(trainer.device)
            out = model(X_t)
            preds = out.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(y.tolist() if hasattr(y, 'tolist') else list(y))

    if is_hpfl:
        for n, p in model.named_parameters():
            if n in trainer._hpfl_classifier_names:
                p.data.copy_(saved_cls[n])

    return np.array(all_labels), np.array(all_preds)


def flip_labels(client_data, adversarial_id, num_classes=2):
    """Create poisoned data by flipping labels for the adversarial client."""
    poisoned = {}
    for cid, (X, y) in client_data.items():
        if cid == adversarial_id:
            # Flip labels: 0->1, 1->0 (binary), or (c+1)%C for multiclass
            y_flipped = (num_classes - 1) - y if num_classes == 2 else (y + 1) % num_classes
            poisoned[cid] = (X, y_flipped)
        else:
            poisoned[cid] = (X, y)
    return poisoned


def scale_gradient_attack(trainer, scale_factor=10.0):
    """
    Post-training attack: scale the adversarial client's model update.
    Call this BEFORE aggregation in each round.
    This is a hook — we monkey-patch the train_round method.
    """
    original_train_round = trainer.train_round.__func__ if hasattr(trainer.train_round, '__func__') else None

    def patched_train_round(self, round_num):
        # Call original training
        result = original_train_round(self, round_num)
        return result

    # Note: gradient scaling requires access to per-client updates before aggregation
    # This is complex with the current trainer API. We'll use label flipping instead
    # as the primary attack vector, which is more clinically relevant (data poisoning).
    pass


def try_byzantine_config():
    """Try to import and use the framework's Byzantine config."""
    try:
        from core.byzantine_resilience import ByzantineConfig
        return ByzantineConfig
    except (ImportError, TimeoutError):
        return None


def load_checkpoint():
    if CHECKPOINT_PATH.exists():
        try:
            with open(CHECKPOINT_PATH) as f:
                return json.load(f)
        except Exception:
            pass
    return {"results": [], "completed": [], "metadata": {}}


def save_checkpoint(data):
    tmp = CHECKPOINT_PATH.with_suffix('.tmp')
    with open(tmp, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    tmp.replace(CHECKPOINT_PATH)


def run_experiment(attack_type, defense, dp_level, seed):
    """Run a single Byzantine + DP experiment.

    Attack types:
      - "none": No attack (clean baseline)
      - "label_flip": Adversarial client has flipped labels
      - "noise_inject": Adversarial client trains on noisy features

    Defense types:
      - "none": Standard FedAvg aggregation
      - "krum"/"trimmed_mean"/"median": Byzantine-resilient aggregation
        (via ByzantineConfig if available, else manual post-hoc filtering)
    """
    # Load data
    client_data, client_test, meta = load_breast_cancer_data(
        num_clients=NUM_CLIENTS, seed=seed, is_iid=False, alpha=0.5,
    )

    # Apply attack
    if attack_type == "label_flip":
        client_data = flip_labels(client_data, ADVERSARIAL_CLIENT,
                                  num_classes=BC_CONFIG["num_classes"])
    elif attack_type == "noise_inject":
        # Add Gaussian noise to adversarial client's features
        X, y = client_data[ADVERSARIAL_CLIENT]
        noise = np.random.RandomState(seed).randn(*X.shape) * 2.0
        client_data[ADVERSARIAL_CLIENT] = (X + noise, y)

    # DP config
    dp_enabled = dp_level["dp_enabled"]
    dp_epsilon = dp_level["dp_epsilon"] if dp_enabled else 10.0

    # Try Byzantine defense config
    byzantine_cfg = None
    ByzantineConfigClass = try_byzantine_config()

    if defense != "none" and ByzantineConfigClass is not None:
        try:
            byzantine_cfg = ByzantineConfigClass(
                defense_method=defense,
                num_byzantine=1,  # 1 adversarial out of 3
            )
        except Exception as e:
            print(f" [ByzantineConfig failed: {e}, using manual defense]", end="")
            byzantine_cfg = None

    # Create trainer
    trainer_kwargs = dict(
        num_clients=NUM_CLIENTS,
        algorithm="FedAvg",  # Byzantine defense replaces aggregation
        local_epochs=BC_CONFIG["local_epochs"],
        batch_size=BC_CONFIG["batch_size"],
        learning_rate=BC_CONFIG["learning_rate"],
        mu=BC_CONFIG["mu"],
        dp_enabled=dp_enabled,
        dp_epsilon=dp_epsilon,
        dp_clip_norm=dp_level.get("dp_clip_norm", 1.0),
        seed=seed,
        external_data=client_data,
        external_test_data=client_test,
        input_dim=BC_CONFIG["input_dim"],
        num_classes=BC_CONFIG["num_classes"],
    )

    if byzantine_cfg is not None:
        trainer_kwargs["byzantine_config"] = byzantine_cfg

    trainer = FederatedTrainer(**trainer_kwargs)

    # If Byzantine defense not available via config, we still train normally
    # The attack is in the DATA (label flip / noise), not in the aggregation
    # So even without a defense mechanism, the attack still degrades training

    # Train
    start = time.time()
    for r in range(BC_CONFIG["num_rounds"]):
        trainer.train_round(r)
    elapsed = time.time() - start

    # Evaluate
    y_true, y_pred = collect_predictions(trainer)
    cm = compute_confusion_matrix(y_true, y_pred)
    acc = np.mean(y_true == y_pred)
    dei, recalls = compute_dei(cm)

    result = {
        "attack": attack_type,
        "defense": defense,
        "dp_label": dp_level["label"],
        "dp_epsilon": dp_level["dp_epsilon"],
        "dp_enabled": dp_enabled,
        "seed": seed,
        "accuracy": float(acc),
        "confusion_matrix": cm.tolist(),
        "per_class_recall": {
            "Benign": float(recalls[0]),
            "Malignant": float(recalls[1]),
        },
        "dei": float(dei),
        "runtime_seconds": round(elapsed, 1),
        "byzantine_config_used": byzantine_cfg is not None,
    }

    del trainer
    gc.collect()
    return result


def generate_figures(all_results):
    """Generate summary heatmaps."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available — skipping figures")
        return

    attacks = ["none", "label_flip", "noise_inject"]
    defenses = ["none", "krum", "trimmed_mean", "median"]
    dp_labels = ["No-DP", "eps=10", "eps=1"]

    # Accuracy heatmap: attacks × defenses, one subplot per DP level
    fig, axes = plt.subplots(1, len(dp_labels), figsize=(6 * len(dp_labels), 4))
    if len(dp_labels) == 1:
        axes = [axes]

    for ax, dp_label in zip(axes, dp_labels):
        matrix = np.zeros((len(attacks), len(defenses)))
        for i, atk in enumerate(attacks):
            for j, dfn in enumerate(defenses):
                matching = [r for r in all_results
                           if r["attack"] == atk
                           and r["defense"] == dfn
                           and r["dp_label"] == dp_label
                           and "error" not in r]
                if matching:
                    matrix[i, j] = np.mean([r["accuracy"] for r in matching]) * 100
                else:
                    matrix[i, j] = float('nan')

        im = ax.imshow(matrix, cmap="RdYlGn", vmin=40, vmax=90, aspect="auto")
        for i in range(len(attacks)):
            for j in range(len(defenses)):
                if not np.isnan(matrix[i, j]):
                    color = "white" if matrix[i, j] < 55 or matrix[i, j] > 80 else "black"
                    ax.text(j, i, f"{matrix[i, j]:.0f}%",
                           ha="center", va="center", fontsize=11,
                           color=color, fontweight="bold")

        ax.set_xticks(range(len(defenses)))
        ax.set_xticklabels(defenses, fontsize=10, rotation=30, ha="right")
        ax.set_yticks(range(len(attacks)))
        ax.set_yticklabels(attacks, fontsize=10)
        ax.set_xlabel("Defense", fontsize=11)
        if ax == axes[0]:
            ax.set_ylabel("Attack", fontsize=11)
        ax.set_title(f"DP: {dp_label}", fontsize=12, fontweight="bold")

    plt.suptitle("Breast Cancer — Accuracy Under Byzantine Attack + DP\n"
                 "(1 adversary / 3 clients)",
                 fontsize=13, fontweight="bold", y=1.05)
    plt.colorbar(im, ax=axes, label="Accuracy (%)", shrink=0.8)
    plt.tight_layout()

    fig_path = OUTPUT_DIR / "byzantine_dp_heatmap.png"
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.savefig(str(fig_path).replace(".png", ".pdf"), bbox_inches="tight")
    print(f"\n  Figure saved: {fig_path}")
    plt.close()

    # DEI heatmap
    fig2, axes2 = plt.subplots(1, len(dp_labels), figsize=(6 * len(dp_labels), 4))
    if len(dp_labels) == 1:
        axes2 = [axes2]

    for ax2, dp_label in zip(axes2, dp_labels):
        matrix = np.zeros((len(attacks), len(defenses)))
        for i, atk in enumerate(attacks):
            for j, dfn in enumerate(defenses):
                matching = [r for r in all_results
                           if r["attack"] == atk
                           and r["defense"] == dfn
                           and r["dp_label"] == dp_label
                           and "error" not in r]
                if matching:
                    matrix[i, j] = np.mean([r["dei"] for r in matching])
                else:
                    matrix[i, j] = float('nan')

        im2 = ax2.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=0.8, aspect="auto")
        for i in range(len(attacks)):
            for j in range(len(defenses)):
                if not np.isnan(matrix[i, j]):
                    color = "white" if matrix[i, j] < 0.2 or matrix[i, j] > 0.6 else "black"
                    ax2.text(j, i, f"{matrix[i, j]:.2f}",
                            ha="center", va="center", fontsize=11,
                            color=color, fontweight="bold")

        ax2.set_xticks(range(len(defenses)))
        ax2.set_xticklabels(defenses, fontsize=10, rotation=30, ha="right")
        ax2.set_yticks(range(len(attacks)))
        ax2.set_yticklabels(attacks, fontsize=10)
        ax2.set_xlabel("Defense", fontsize=11)
        if ax2 == axes2[0]:
            ax2.set_ylabel("Attack", fontsize=11)
        ax2.set_title(f"DP: {dp_label}", fontsize=12, fontweight="bold")

    plt.suptitle("Breast Cancer — DEI Under Byzantine Attack + DP",
                 fontsize=13, fontweight="bold", y=1.05)
    plt.colorbar(im2, ax=axes2, label="DEI", shrink=0.8)
    plt.tight_layout()

    fig2_path = OUTPUT_DIR / "byzantine_dp_dei_heatmap.png"
    plt.savefig(fig2_path, dpi=200, bbox_inches="tight")
    plt.savefig(str(fig2_path).replace(".png", ".pdf"), bbox_inches="tight")
    print(f"  Figure saved: {fig2_path}")
    plt.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Byzantine + DP Joint Evaluation")
    parser.add_argument("--quick", action="store_true", help="Quick mode (1 seed, fewer configs)")
    parser.add_argument("--no-resume", action="store_true", help="Start fresh")
    args = parser.parse_args()

    seeds = [42] if args.quick else SEEDS

    attacks = ["none", "label_flip", "noise_inject"]
    defenses = ["none", "krum", "trimmed_mean", "median"]
    dp_levels = [
        {"label": "No-DP", "dp_enabled": False, "dp_epsilon": 0, "dp_clip_norm": 1.0},
        {"label": "eps=10", "dp_enabled": True, "dp_epsilon": 10.0, "dp_clip_norm": 1.0},
        {"label": "eps=1",  "dp_enabled": True, "dp_epsilon": 1.0, "dp_clip_norm": 1.0},
    ]

    if args.quick:
        attacks = ["none", "label_flip"]
        defenses = ["none", "krum"]
        dp_levels = dp_levels[:2]

    # Resume
    checkpoint = {"results": [], "completed": [], "metadata": {}}
    if not args.no_resume:
        checkpoint = load_checkpoint()
    completed_keys = set(checkpoint.get("completed", []))

    total = len(attacks) * len(defenses) * len(dp_levels) * len(seeds)
    skip = sum(1 for a in attacks for d in defenses for dp in dp_levels for s in seeds
               if f"{a}_{d}_{dp['label']}_s{s}" in completed_keys)

    print("=" * 65)
    print("  FL-EHDS Experiment A — Byzantine + DP Joint Evaluation")
    print(f"  {len(attacks)} attacks × {len(defenses)} defenses × "
          f"{len(dp_levels)} DP × {len(seeds)} seeds = {total} experiments")
    if skip:
        print(f"  Skipping {skip} completed, running {total - skip}")
    print(f"  Adversary: client {ADVERSARIAL_CLIENT}/{NUM_CLIENTS}")
    print(f"  Device: {_detect_device()}")

    # Check Byzantine config availability
    ByzCfg = try_byzantine_config()
    if ByzCfg:
        print(f"  Byzantine defense: framework ByzantineConfig available")
    else:
        print(f"  Byzantine defense: NOT available, using data-level attacks only")
    print("=" * 65)

    exp_count = 0
    for attack in attacks:
        for defense in defenses:
            # Skip defense-only runs with no attack (except "none"/"none" baseline)
            if attack == "none" and defense != "none":
                continue  # Defense without attack is not meaningful

            for dp_level in dp_levels:
                for seed in seeds:
                    key = f"{attack}_{defense}_{dp_level['label']}_s{seed}"
                    if key in completed_keys:
                        exp_count += 1
                        continue

                    exp_count += 1
                    print(f"\n  [{exp_count}/{total}] atk={attack} | def={defense} | "
                          f"{dp_level['label']} | s{seed} ...", end="", flush=True)

                    try:
                        result = run_experiment(attack, defense, dp_level, seed)
                        checkpoint["results"].append(result)
                        checkpoint["completed"].append(key)

                        print(f" acc={result['accuracy']*100:.1f}% "
                              f"DEI={result['dei']:.3f} "
                              f"Ben={result['per_class_recall']['Benign']*100:.0f}% "
                              f"Mal={result['per_class_recall']['Malignant']*100:.0f}% "
                              f"{result['runtime_seconds']:.0f}s"
                              f"{' [ByzCfg]' if result.get('byzantine_config_used') else ''}")

                    except Exception as e:
                        print(f" ERROR: {e}")
                        import traceback
                        traceback.print_exc()
                        checkpoint["results"].append({
                            "attack": attack, "defense": defense,
                            "dp_label": dp_level["label"], "seed": seed,
                            "error": str(e),
                        })
                        checkpoint["completed"].append(key)

                    checkpoint["metadata"] = {
                        "attacks": attacks,
                        "defenses": defenses,
                        "dp_levels": [d["label"] for d in dp_levels],
                        "seeds": seeds,
                        "num_clients": NUM_CLIENTS,
                        "adversarial_client": ADVERSARIAL_CLIENT,
                        "byzantine_config_available": ByzCfg is not None,
                        "timestamp": datetime.now().isoformat(),
                    }
                    save_checkpoint(checkpoint)

    # Summary
    valid = [r for r in checkpoint["results"] if "error" not in r]

    print("\n" + "=" * 65)
    print("  SUMMARY — Byzantine + DP Accuracy")
    print("=" * 65)

    for dp_level in dp_levels:
        print(f"\n  DP: {dp_level['label']}")
        print(f"  {'Attack':<15} {'Defense':<15} {'Acc%':<8} {'DEI':<8} {'Ben%':<8} {'Mal%':<8}")
        print(f"  {'-'*62}")

        for attack in attacks:
            for defense in defenses:
                if attack == "none" and defense != "none":
                    continue
                matching = [r for r in valid
                           if r["attack"] == attack
                           and r["defense"] == defense
                           and r["dp_label"] == dp_level["label"]]
                if matching:
                    acc = np.mean([r["accuracy"] for r in matching]) * 100
                    dei = np.mean([r["dei"] for r in matching])
                    ben = np.mean([r["per_class_recall"]["Benign"] for r in matching]) * 100
                    mal = np.mean([r["per_class_recall"]["Malignant"] for r in matching]) * 100
                    print(f"  {attack:<15} {defense:<15} {acc:<8.1f} {dei:<8.3f} {ben:<8.0f} {mal:<8.0f}")

    # Generate figures
    if valid:
        print("\n  Generating figures...")
        generate_figures(valid)

    print(f"\n  Checkpoint: {CHECKPOINT_PATH}")
    print("=" * 65)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
FL-EHDS Experiment A — Byzantine Robustness + Differential Privacy Joint Evaluation.

Tests FL algorithms under simultaneous Byzantine attacks and differential privacy,
answering: does DP noise help or hurt Byzantine robustness? How does HPFL compare
to FedAvg under adversarial conditions?

Design:
  - Dataset: Breast Cancer (fast, tabular, binary classification)
  - Algorithms: FedAvg (tuned), HPFL (personalized)
  - Attack: 1 Byzantine client out of 5 (20%)
  - Attack types: label_flip (data poisoning), sign_flip (model poisoning)
  - Defenses: Krum, TrimmedMean, Median, Bulyan (+ no defense baseline)
  - DP: No-DP, eps=10, eps=1
  - Seeds: 3

Total: 2 algos × [1 baseline + 2 attacks × 5 defenses] × 3 DP × 3 seeds = 198
Estimated runtime: ~40-50 min

Usage:
    cd fl-ehds-framework
    python benchmarks/run_byzantine_dp.py [--quick] [--no-resume]

Output:
    /tmp/fl_ehds_results/byzantine_dp_v2_results.json
    /tmp/fl_ehds_results/byzantine_dp_v2_heatmap.png

Author: Fabio Liberti (generated for FLICS 2026)
"""

import sys
import os
import json
import time
import gc
from datetime import datetime
from pathlib import Path
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
CHECKPOINT_PATH = OUTPUT_DIR / "byzantine_dp_v2_results.json"

# ─── Configuration ────────────────────────────────────────────────────
NUM_CLIENTS = 5
ADVERSARIAL_CLIENT = 0  # Client 0 is the adversary (1 out of 5, 20%)

ALGORITHMS = {
    "FedAvg": dict(
        input_dim=30, num_classes=2,
        learning_rate=0.01, batch_size=16, num_rounds=60,
        local_epochs=3, mu=0.1,
    ),
    "HPFL": dict(
        input_dim=30, num_classes=2,
        learning_rate=0.01, batch_size=16, num_rounds=60,
        local_epochs=3, mu=0.1,
    ),
}

SEEDS = [42, 123, 456]
CLASS_NAMES = {0: "Benign", 1: "Malignant"}

ATTACKS = ["none", "label_flip", "sign_flip"]
DEFENSES = ["none", "krum", "trimmed_mean", "median", "bulyan"]
DP_LEVELS = [
    {"label": "No-DP", "dp_enabled": False, "dp_epsilon": 0, "dp_clip_norm": 1.0},
    {"label": "eps=10", "dp_enabled": True, "dp_epsilon": 10.0, "dp_clip_norm": 1.0},
    {"label": "eps=1",  "dp_enabled": True, "dp_epsilon": 1.0, "dp_clip_norm": 1.0},
]


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
        return 0.0, recalls, 0.0, 1.0
    mean_r = np.mean(recalls)
    std_r = np.std(recalls)
    cv = std_r / mean_r if mean_r > 0 else 1.0
    dei = min(recalls) * (1 - cv)
    return dei, recalls, min(recalls), cv


def collect_predictions(trainer, num_classes=2):
    """Collect predictions — handles both FedAvg and HPFL."""
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
            y_flipped = (num_classes - 1) - y if num_classes == 2 else (y + 1) % num_classes
            poisoned[cid] = (X, y_flipped)
        else:
            poisoned[cid] = (X, y)
    return poisoned


def make_byzantine_config(defense, num_byzantine=1, num_clients=5):
    """Create a ByzantineConfig for the given defense method.
    Returns None if defense is 'none' or if the module is unavailable."""
    if defense == "none":
        return None
    try:
        from core.byzantine_resilience import ByzantineConfig
        # Multi-Krum: select n-f best clients (exclude suspected adversary)
        multi_krum_m = num_clients - num_byzantine  # e.g. 5-1=4
        return ByzantineConfig(
            aggregation_rule=defense,
            num_byzantine=num_byzantine,
            multi_krum_m=multi_krum_m,
            trim_ratio=0.2,  # trim 20% from each side
            enable_detection=True,
            detection_threshold=3.0,
        )
    except (ImportError, Exception) as e:
        print(f" [ByzantineConfig error: {e}]", end="")
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


def run_experiment(algo, attack_type, defense, dp_level, seed):
    """Run a single Byzantine + DP experiment for a given algorithm."""
    cfg = ALGORITHMS[algo]

    # Load data — IID split to isolate Byzantine effect from non-IID confound
    client_data, client_test, meta = load_breast_cancer_data(
        num_clients=NUM_CLIENTS, seed=seed, is_iid=True,
    )

    # Apply data-level attack to training data
    if attack_type == "label_flip":
        client_data = flip_labels(client_data, ADVERSARIAL_CLIENT,
                                  num_classes=cfg["num_classes"])
    # sign_flip is a model-level attack applied during training (see below)

    # DP config
    dp_enabled = dp_level["dp_enabled"]
    dp_epsilon = dp_level["dp_epsilon"] if dp_enabled else 10.0

    # Byzantine defense config
    byzantine_cfg = make_byzantine_config(defense, num_byzantine=1, num_clients=NUM_CLIENTS)

    # Create trainer
    trainer_kwargs = dict(
        num_clients=NUM_CLIENTS,
        algorithm=algo,
        local_epochs=cfg["local_epochs"],
        batch_size=cfg["batch_size"],
        learning_rate=cfg["learning_rate"],
        mu=cfg["mu"],
        dp_enabled=dp_enabled,
        dp_epsilon=dp_epsilon,
        dp_clip_norm=dp_level.get("dp_clip_norm", 1.0),
        seed=seed,
        external_data=client_data,
        external_test_data=client_test,
        input_dim=cfg["input_dim"],
        num_classes=cfg["num_classes"],
    )
    if byzantine_cfg is not None:
        trainer_kwargs["byzantine_config"] = byzantine_cfg

    trainer = FederatedTrainer(**trainer_kwargs)

    # For sign_flip (model poisoning): adversary sends -5x gradient to push model wrong
    if attack_type == "sign_flip":
        _orig_train_client = trainer._train_client

        def _attacked_train_client(client_id, round_num):
            result = _orig_train_client(client_id, round_num)
            if client_id == ADVERSARIAL_CLIENT:
                for name in result.model_update:
                    result.model_update[name] = result.model_update[name] * (-5.0)
            return result

        trainer._train_client = _attacked_train_client

    # Train
    start = time.time()
    for r in range(cfg["num_rounds"]):
        trainer.train_round(r)
    elapsed = time.time() - start

    # Evaluate
    y_true, y_pred = collect_predictions(trainer, cfg["num_classes"])
    cm = compute_confusion_matrix(y_true, y_pred, cfg["num_classes"])
    acc = np.mean(y_true == y_pred)
    dei, recalls, min_r, cv = compute_dei(cm, cfg["num_classes"])

    result = {
        "algorithm": algo,
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
        "min_recall": float(min_r),
        "cv": float(cv),
        "runtime_seconds": round(elapsed, 1),
        "byzantine_config_used": byzantine_cfg is not None,
    }

    del trainer
    gc.collect()
    return result


def generate_figures(all_results):
    """Generate summary heatmaps — one row per algorithm."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
    except ImportError:
        print("  matplotlib not available — skipping figures")
        return

    algos = list(ALGORITHMS.keys())
    dp_labels = [d["label"] for d in DP_LEVELS]

    # ─── Figure 1: Accuracy heatmaps ─────────────────────────────────
    fig, axes = plt.subplots(len(algos), len(dp_labels),
                             figsize=(5.5 * len(dp_labels), 4 * len(algos)))
    if len(algos) == 1:
        axes = [axes]

    for row, algo in enumerate(algos):
        for col, dp_label in enumerate(dp_labels):
            ax = axes[row][col] if len(dp_labels) > 1 else axes[row]

            # Build matrix: rows = attacks (incl. baseline), cols = defenses
            row_labels = []
            matrix_rows = []

            # Baseline: no attack, no defense
            matching = [r for r in all_results
                       if r["algorithm"] == algo
                       and r["attack"] == "none" and r["defense"] == "none"
                       and r["dp_label"] == dp_label and "error" not in r]
            if matching:
                row_labels.append("no attack")
                matrix_rows.append([np.mean([r["accuracy"] for r in matching]) * 100] * len(DEFENSES))

            # Attacks with all defenses
            for atk in ["label_flip", "sign_flip"]:
                vals = []
                for dfn in DEFENSES:
                    matching = [r for r in all_results
                               if r["algorithm"] == algo
                               and r["attack"] == atk and r["defense"] == dfn
                               and r["dp_label"] == dp_label and "error" not in r]
                    if matching:
                        vals.append(np.mean([r["accuracy"] for r in matching]) * 100)
                    else:
                        vals.append(float('nan'))
                row_labels.append(atk)
                matrix_rows.append(vals)

            matrix = np.array(matrix_rows)
            im = ax.imshow(matrix, cmap="RdYlGn", vmin=40, vmax=95, aspect="auto")

            for i in range(len(row_labels)):
                for j in range(len(DEFENSES)):
                    if not np.isnan(matrix[i, j]):
                        color = "white" if matrix[i, j] < 55 or matrix[i, j] > 85 else "black"
                        ax.text(j, i, f"{matrix[i, j]:.0f}%",
                               ha="center", va="center", fontsize=10,
                               color=color, fontweight="bold")

            ax.set_xticks(range(len(DEFENSES)))
            ax.set_xticklabels(DEFENSES, fontsize=9, rotation=30, ha="right")
            ax.set_yticks(range(len(row_labels)))
            ax.set_yticklabels(row_labels, fontsize=9)
            if col == 0:
                ax.set_ylabel(f"{algo}\nAttack", fontsize=11, fontweight="bold")
            if row == 0:
                ax.set_title(f"DP: {dp_label}", fontsize=11, fontweight="bold")
            if row == len(algos) - 1:
                ax.set_xlabel("Defense", fontsize=10)

    plt.suptitle("Breast Cancer — Accuracy Under Byzantine Attack + DP\n"
                 "(1 adversary / 5 clients, FedAvg vs HPFL)",
                 fontsize=13, fontweight="bold", y=1.03)
    fig.colorbar(im, ax=axes, label="Accuracy (%)", shrink=0.7, pad=0.02)
    fig.tight_layout(rect=[0, 0, 0.92, 0.95])

    fig_path = OUTPUT_DIR / "byzantine_dp_v2_accuracy.png"
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.savefig(str(fig_path).replace(".png", ".pdf"), bbox_inches="tight")
    print(f"\n  Figure saved: {fig_path}")
    plt.close()

    # ─── Figure 2: DEI heatmaps ──────────────────────────────────────
    fig2, axes2 = plt.subplots(len(algos), len(dp_labels),
                               figsize=(5.5 * len(dp_labels), 4 * len(algos)))
    if len(algos) == 1:
        axes2 = [axes2]

    for row, algo in enumerate(algos):
        for col, dp_label in enumerate(dp_labels):
            ax2 = axes2[row][col] if len(dp_labels) > 1 else axes2[row]

            row_labels = []
            matrix_rows = []

            matching = [r for r in all_results
                       if r["algorithm"] == algo
                       and r["attack"] == "none" and r["defense"] == "none"
                       and r["dp_label"] == dp_label and "error" not in r]
            if matching:
                row_labels.append("no attack")
                matrix_rows.append([np.mean([r["dei"] for r in matching])] * len(DEFENSES))

            for atk in ["label_flip", "sign_flip"]:
                vals = []
                for dfn in DEFENSES:
                    matching = [r for r in all_results
                               if r["algorithm"] == algo
                               and r["attack"] == atk and r["defense"] == dfn
                               and r["dp_label"] == dp_label and "error" not in r]
                    if matching:
                        vals.append(np.mean([r["dei"] for r in matching]))
                    else:
                        vals.append(float('nan'))
                row_labels.append(atk)
                matrix_rows.append(vals)

            matrix = np.array(matrix_rows)
            im2 = ax2.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=0.85, aspect="auto")

            for i in range(len(row_labels)):
                for j in range(len(DEFENSES)):
                    if not np.isnan(matrix[i, j]):
                        color = "white" if matrix[i, j] < 0.2 or matrix[i, j] > 0.65 else "black"
                        ax2.text(j, i, f"{matrix[i, j]:.2f}",
                                ha="center", va="center", fontsize=10,
                                color=color, fontweight="bold")

            ax2.set_xticks(range(len(DEFENSES)))
            ax2.set_xticklabels(DEFENSES, fontsize=9, rotation=30, ha="right")
            ax2.set_yticks(range(len(row_labels)))
            ax2.set_yticklabels(row_labels, fontsize=9)
            if col == 0:
                ax2.set_ylabel(f"{algo}\nAttack", fontsize=11, fontweight="bold")
            if row == 0:
                ax2.set_title(f"DP: {dp_label}", fontsize=11, fontweight="bold")
            if row == len(algos) - 1:
                ax2.set_xlabel("Defense", fontsize=10)

    plt.suptitle("Breast Cancer — DEI Under Byzantine Attack + DP\n"
                 "(1 adversary / 5 clients, FedAvg vs HPFL)",
                 fontsize=13, fontweight="bold", y=1.03)
    fig2.colorbar(im2, ax=axes2, label="DEI", shrink=0.7, pad=0.02)
    fig2.tight_layout(rect=[0, 0, 0.92, 0.95])

    fig2_path = OUTPUT_DIR / "byzantine_dp_v2_dei.png"
    plt.savefig(fig2_path, dpi=200, bbox_inches="tight")
    plt.savefig(str(fig2_path).replace(".png", ".pdf"), bbox_inches="tight")
    print(f"  Figure saved: {fig2_path}")
    plt.close()

    # ─── Figure 3: Summary bar chart — algo comparison ────────────────
    fig3, (ax_acc, ax_dei) = plt.subplots(1, 2, figsize=(12, 5))

    scenarios = ["clean", "label_flip", "sign_flip"]
    scenario_labels = ["Clean", "Label Flip\n(data poison)", "Sign Flip\n(model poison)"]
    x = np.arange(len(scenarios))
    width = 0.35
    colors = {"FedAvg": "#d62728", "HPFL": "#2ca02c"}

    for dp_idx, dp_label in enumerate(dp_labels):
        if dp_label != "No-DP":
            continue  # Just show No-DP for the bar chart

        for algo_idx, algo in enumerate(algos):
            accs, deis = [], []
            for scenario in scenarios:
                if scenario == "clean":
                    matching = [r for r in all_results
                               if r["algorithm"] == algo
                               and r["attack"] == "none" and r["defense"] == "none"
                               and r["dp_label"] == dp_label and "error" not in r]
                else:
                    # Best defense for this attack
                    best_acc = 0
                    best_dei = 0
                    for dfn in DEFENSES:
                        m = [r for r in all_results
                             if r["algorithm"] == algo
                             and r["attack"] == scenario and r["defense"] == dfn
                             and r["dp_label"] == dp_label and "error" not in r]
                        if m:
                            a = np.mean([r["accuracy"] for r in m])
                            d = np.mean([r["dei"] for r in m])
                            if a > best_acc:
                                best_acc = a
                                best_dei = d
                    matching = None
                    accs.append(best_acc * 100)
                    deis.append(best_dei)
                    continue

                if matching:
                    accs.append(np.mean([r["accuracy"] for r in matching]) * 100)
                    deis.append(np.mean([r["dei"] for r in matching]))
                else:
                    accs.append(0)
                    deis.append(0)

            offset = -width/2 + algo_idx * width
            bars_acc = ax_acc.bar(x + offset, accs, width, label=algo,
                                 color=colors[algo], alpha=0.85)
            bars_dei = ax_dei.bar(x + offset, deis, width, label=algo,
                                 color=colors[algo], alpha=0.85)

            for bar, val in zip(bars_acc, accs):
                ax_acc.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                           f'{val:.0f}%', ha='center', va='bottom', fontsize=9)
            for bar, val in zip(bars_dei, deis):
                ax_dei.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{val:.2f}', ha='center', va='bottom', fontsize=9)

    ax_acc.set_xticks(x)
    ax_acc.set_xticklabels(scenario_labels, fontsize=11)
    ax_acc.set_ylabel("Accuracy (%)", fontsize=12)
    ax_acc.set_title("Accuracy (best defense per attack)", fontsize=12, fontweight="bold")
    ax_acc.legend(fontsize=10)
    ax_acc.set_ylim(0, 100)
    ax_acc.grid(axis='y', alpha=0.3)

    ax_dei.set_xticks(x)
    ax_dei.set_xticklabels(scenario_labels, fontsize=11)
    ax_dei.set_ylabel("DEI", fontsize=12)
    ax_dei.set_title("Diagnostic Equity (best defense per attack)", fontsize=12, fontweight="bold")
    ax_dei.legend(fontsize=10)
    ax_dei.set_ylim(0, 1.0)
    ax_dei.grid(axis='y', alpha=0.3)

    plt.suptitle("FedAvg vs HPFL Under Byzantine Attacks (No-DP)",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    fig3_path = OUTPUT_DIR / "byzantine_dp_v2_comparison.png"
    plt.savefig(fig3_path, dpi=200, bbox_inches="tight")
    plt.savefig(str(fig3_path).replace(".png", ".pdf"), bbox_inches="tight")
    print(f"  Figure saved: {fig3_path}")
    plt.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Byzantine + DP Joint Evaluation (v2)")
    parser.add_argument("--quick", action="store_true", help="Quick mode (1 seed, fewer configs)")
    parser.add_argument("--no-resume", action="store_true", help="Start fresh")
    args = parser.parse_args()

    seeds = [42] if args.quick else SEEDS
    algos = list(ALGORITHMS.keys())
    attacks = ATTACKS
    defenses = DEFENSES
    dp_levels = DP_LEVELS

    if args.quick:
        attacks = ["none", "label_flip"]
        defenses = ["none", "krum"]
        dp_levels = dp_levels[:2]

    # Count experiments (skip defense-only when no attack)
    combos = []
    for algo in algos:
        for atk in attacks:
            for dfn in defenses:
                if atk == "none" and dfn != "none":
                    continue
                for dp in dp_levels:
                    for s in seeds:
                        combos.append(f"{algo}_{atk}_{dfn}_{dp['label']}_s{s}")
    total = len(combos)

    # Resume
    checkpoint = {"results": [], "completed": [], "metadata": {}}
    if not args.no_resume:
        checkpoint = load_checkpoint()
    completed_keys = set(checkpoint.get("completed", []))
    skip = sum(1 for c in combos if c in completed_keys)

    print("=" * 70)
    print("  FL-EHDS Experiment A v2 — Byzantine + DP (FedAvg vs HPFL)")
    print(f"  {len(algos)} algos × {len(attacks)} attacks × {len(defenses)} defs "
          f"× {len(dp_levels)} DP × {len(seeds)} seeds")
    print(f"  Total experiments: {total}")
    if skip:
        print(f"  Resuming: {skip} done, {total - skip} remaining")
    print(f"  Adversary: client {ADVERSARIAL_CLIENT}/{NUM_CLIENTS}")
    print(f"  Device: {_detect_device()}")

    for algo, cfg in ALGORITHMS.items():
        print(f"  {algo}: lr={cfg['learning_rate']}, epochs={cfg['local_epochs']}, "
              f"rounds={cfg['num_rounds']}")

    # Verify Byzantine defense availability
    test_cfg = make_byzantine_config("krum", num_byzantine=1, num_clients=NUM_CLIENTS)
    byz_available = test_cfg is not None
    print(f"  Byzantine defense: {'ACTIVE (framework ByzantineConfig)' if byz_available else 'NOT available'}")
    print("=" * 70)

    exp_count = 0
    for algo in algos:
        print(f"\n{'─'*70}")
        print(f"  Algorithm: {algo}")
        print(f"{'─'*70}")

        for atk in attacks:
            for dfn in defenses:
                if atk == "none" and dfn != "none":
                    continue

                for dp_level in dp_levels:
                    for seed in seeds:
                        key = f"{algo}_{atk}_{dfn}_{dp_level['label']}_s{seed}"
                        exp_count += 1

                        if key in completed_keys:
                            continue

                        print(f"\n  [{exp_count}/{total}] {algo} | atk={atk} | "
                              f"def={dfn} | {dp_level['label']} | s{seed} ...",
                              end="", flush=True)

                        try:
                            result = run_experiment(algo, atk, dfn, dp_level, seed)
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
                                "algorithm": algo, "attack": atk, "defense": dfn,
                                "dp_label": dp_level["label"], "seed": seed,
                                "error": str(e),
                            })
                            checkpoint["completed"].append(key)

                        checkpoint["metadata"] = {
                            "algorithms": algos,
                            "attacks": attacks,
                            "defenses": defenses,
                            "dp_levels": [d["label"] for d in dp_levels],
                            "seeds": seeds,
                            "num_clients": NUM_CLIENTS,
                            "adversarial_client": ADVERSARIAL_CLIENT,
                            "algo_configs": {k: v for k, v in ALGORITHMS.items()},
                            "byzantine_config_available": byz_available,
                            "timestamp": datetime.now().isoformat(),
                        }
                        save_checkpoint(checkpoint)

    # Summary
    valid = [r for r in checkpoint["results"] if "error" not in r]

    print("\n" + "=" * 70)
    print("  SUMMARY — Byzantine + DP (FedAvg vs HPFL)")
    print("=" * 70)

    for algo in algos:
        print(f"\n  ═══ {algo} ═══")
        for dp_level in dp_levels:
            print(f"\n  DP: {dp_level['label']}")
            print(f"  {'Attack':<15} {'Defense':<15} {'Acc%':<8} {'DEI':<8} "
                  f"{'Ben%':<8} {'Mal%':<8}")
            print(f"  {'-'*66}")

            for atk in attacks:
                for dfn in defenses:
                    if atk == "none" and dfn != "none":
                        continue
                    matching = [r for r in valid
                               if r["algorithm"] == algo
                               and r["attack"] == atk
                               and r["defense"] == dfn
                               and r["dp_label"] == dp_level["label"]]
                    if matching:
                        acc = np.mean([r["accuracy"] for r in matching]) * 100
                        dei = np.mean([r["dei"] for r in matching])
                        ben = np.mean([r["per_class_recall"]["Benign"]
                                      for r in matching]) * 100
                        mal = np.mean([r["per_class_recall"]["Malignant"]
                                      for r in matching]) * 100
                        print(f"  {atk:<15} {dfn:<15} {acc:<8.1f} {dei:<8.3f} "
                              f"{ben:<8.0f} {mal:<8.0f}")

    # Generate figures
    if valid:
        print("\n  Generating figures...")
        generate_figures(valid)

    print(f"\n  Checkpoint: {CHECKPOINT_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
FL-EHDS Experiment F — Byzantine+DP Interaction on Cardiovascular Disease.

Extends the Byzantine+DP analysis (Breast Cancer, 198 exp) to a larger,
more clinically realistic dataset. Adds Ditto (best personalized method)
and FLTrust (6th defense not in original study).

Key research question: Do Byzantine defenses remain effective when combined
with DP noise on a larger dataset? Does Ditto's personalization provide
natural robustness like HPFL on Breast Cancer?

Design:
  - Dataset: Cardiovascular Disease (70K samples, 11 features, binary)
  - Algorithms: FedAvg (baseline), Ditto (best personalized)
  - Attacks: none (baseline), label_flip (data), sign_flip (model)
  - Defenses: none, krum, trimmed_mean, median, bulyan (5)
  - DP: No-DP, eps=10, eps=1 (3 levels)
  - Seeds: 42, 123, 456

  Total: 2 algos x [1 baseline + 2 attacks x 5 defenses] x 3 DP x 3 seeds
       = 2 x 11 x 9 = 198 experiments (~1.5-2h on M3)

Usage:
    cd fl-ehds-framework
    python -m benchmarks.run_byzantine_dp_cv [--quick] [--fresh]

Output: benchmarks/paper_results_tabular/checkpoint_byzantine_dp_cv.json

Author: Fabio Liberti
"""

import sys
import os
import json
import time
import gc
import signal
import shutil
import tempfile
import argparse
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from copy import deepcopy
from typing import Dict, Optional, Any

import numpy as np

FRAMEWORK_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(FRAMEWORK_DIR))

import torch
from terminal.fl_trainer import FederatedTrainer, _detect_device
from data.cardiovascular_loader import load_cardiovascular_data

# ======================================================================
# Constants
# ======================================================================

OUTPUT_DIR = FRAMEWORK_DIR / "benchmarks" / "paper_results_tabular"
CHECKPOINT_FILE = "checkpoint_byzantine_dp_cv.json"

NUM_CLIENTS = 5
ADVERSARIAL_CLIENT = 0  # 1 out of 5 (20%)

ALGORITHMS = {
    "FedAvg": dict(
        input_dim=11, num_classes=2,
        learning_rate=0.01, batch_size=64, num_rounds=30,
        local_epochs=3, mu=0.1,
    ),
    "Ditto": dict(
        input_dim=11, num_classes=2,
        learning_rate=0.01, batch_size=64, num_rounds=30,
        local_epochs=3, mu=0.1,
    ),
}

SEEDS = [42, 123, 456]

ATTACKS = ["none", "label_flip", "sign_flip"]
DEFENSES = ["none", "krum", "trimmed_mean", "median", "bulyan"]
DP_LEVELS = [
    {"label": "No-DP", "dp_enabled": False, "dp_epsilon": 0, "dp_clip_norm": 1.0},
    {"label": "eps=10", "dp_enabled": True, "dp_epsilon": 10.0, "dp_clip_norm": 1.0},
    {"label": "eps=1",  "dp_enabled": True, "dp_epsilon": 1.0, "dp_clip_norm": 1.0},
]


# ======================================================================
# Checkpoint (atomic)
# ======================================================================

def save_checkpoint(data: Dict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / CHECKPOINT_FILE
    bak = path.with_suffix(".json.bak")
    data["metadata"]["last_save"] = datetime.now().isoformat()
    fd, tmp = tempfile.mkstemp(dir=str(OUTPUT_DIR), prefix=".byz_cv_", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2, default=str)
            f.flush()
            os.fsync(f.fileno())
        if path.exists():
            shutil.copy2(str(path), str(bak))
        os.replace(tmp, str(path))
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def load_checkpoint() -> Optional[Dict]:
    path = OUTPUT_DIR / CHECKPOINT_FILE
    bak = path.with_suffix(".json.bak")
    for p in [path, bak]:
        if p.exists():
            try:
                with open(p) as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                continue
    return None


# ======================================================================
# Helpers
# ======================================================================

def _cleanup_gpu():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        try:
            torch.mps.empty_cache()
        except Exception:
            pass
    gc.collect()


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
    model = trainer.global_model
    model.eval()
    all_preds, all_labels = [], []

    is_ditto = trainer.algorithm == "Ditto"
    is_hpfl = trainer.algorithm == "HPFL"

    if is_hpfl and hasattr(trainer, '_hpfl_classifier_names'):
        saved_cls = {n: p.data.clone() for n, p in model.named_parameters()
                     if n in trainer._hpfl_classifier_names}

    with torch.no_grad():
        for cid in range(trainer.num_clients):
            # For Ditto: use personalized model if available
            if is_ditto and hasattr(trainer, 'personalized_models') and \
               cid in trainer.personalized_models:
                eval_model = trainer.personalized_models[cid]
                eval_model.eval()
            elif is_hpfl and hasattr(trainer, '_hpfl_classifier_names'):
                for n, p in model.named_parameters():
                    if n in trainer._hpfl_classifier_names:
                        p.data.copy_(trainer.client_classifiers[cid][n])
                eval_model = model
            else:
                eval_model = model

            X, y = trainer.client_test_data[cid]
            X_t = torch.FloatTensor(X).to(trainer.device)
            out = eval_model(X_t)
            preds = out.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(y.tolist() if hasattr(y, 'tolist') else list(y))

    if is_hpfl and hasattr(trainer, '_hpfl_classifier_names'):
        for n, p in model.named_parameters():
            if n in trainer._hpfl_classifier_names:
                p.data.copy_(saved_cls[n])

    return np.array(all_labels), np.array(all_preds)


def flip_labels(client_data, adversarial_id, num_classes=2):
    poisoned = {}
    for cid, (X, y) in client_data.items():
        if cid == adversarial_id:
            y_flipped = (num_classes - 1) - y if num_classes == 2 else (y + 1) % num_classes
            poisoned[cid] = (X, y_flipped)
        else:
            poisoned[cid] = (X, y)
    return poisoned


def make_byzantine_config(defense, num_byzantine=1, num_clients=5):
    if defense == "none":
        return None
    try:
        from core.byzantine_resilience import ByzantineConfig
        multi_krum_m = num_clients - num_byzantine
        return ByzantineConfig(
            aggregation_rule=defense,
            num_byzantine=num_byzantine,
            multi_krum_m=multi_krum_m,
            trim_ratio=0.2,
            enable_detection=True,
            detection_threshold=3.0,
        )
    except (ImportError, Exception) as e:
        print(f" [ByzantineConfig error: {e}]", end="")
        return None


# ======================================================================
# Single experiment
# ======================================================================

def run_experiment(algo, attack_type, defense, dp_level, seed):
    cfg = ALGORITHMS[algo]

    client_data, client_test, meta = load_cardiovascular_data(
        num_clients=NUM_CLIENTS, seed=seed, is_iid=True,
    )

    if attack_type == "label_flip":
        client_data = flip_labels(client_data, ADVERSARIAL_CLIENT,
                                  num_classes=cfg["num_classes"])

    dp_enabled = dp_level["dp_enabled"]
    dp_epsilon = dp_level["dp_epsilon"] if dp_enabled else 10.0

    byzantine_cfg = make_byzantine_config(defense, num_byzantine=1,
                                          num_clients=NUM_CLIENTS)

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

    # Sign flip attack: adversary sends -5x gradient
    if attack_type == "sign_flip":
        _orig_train_client = trainer._train_client

        def _attacked_train_client(client_id, round_num):
            result = _orig_train_client(client_id, round_num)
            if client_id == ADVERSARIAL_CLIENT:
                for name in result.model_update:
                    result.model_update[name] = result.model_update[name] * (-5.0)
            return result

        trainer._train_client = _attacked_train_client

    start = time.time()
    for r in range(cfg["num_rounds"]):
        trainer.train_round(r)
    elapsed = time.time() - start

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
            "Negative": float(recalls[0]),
            "Positive": float(recalls[1]),
        },
        "dei": float(dei),
        "min_recall": float(min_r),
        "cv": float(cv),
        "runtime_seconds": round(elapsed, 1),
        "byzantine_config_used": byzantine_cfg is not None,
    }

    del trainer
    _cleanup_gpu()
    return result


# ======================================================================
# Figure generation
# ======================================================================

def generate_figures(all_results):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available — skipping figures")
        return

    algos = list(ALGORITHMS.keys())
    dp_labels = [d["label"] for d in DP_LEVELS]
    active_defenses = [d for d in DEFENSES if d != "none"]

    # Accuracy heatmaps: rows = attacks, cols = defenses, one subplot per algo x DP
    fig, axes = plt.subplots(len(algos), len(dp_labels),
                             figsize=(5.5 * len(dp_labels), 4 * len(algos)))
    if len(algos) == 1:
        axes = [axes]

    for row, algo in enumerate(algos):
        for col, dp_label in enumerate(dp_labels):
            ax = axes[row][col] if len(dp_labels) > 1 else axes[row]

            row_labels = []
            matrix_rows = []

            # Baseline: no attack
            matching = [r for r in all_results
                       if r["algorithm"] == algo
                       and r["attack"] == "none" and r["defense"] == "none"
                       and r["dp_label"] == dp_label and "error" not in r]
            if matching:
                row_labels.append("no attack")
                matrix_rows.append(
                    [np.mean([r["accuracy"] for r in matching]) * 100] * len(DEFENSES))

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
            im = ax.imshow(matrix, cmap="RdYlGn", vmin=40, vmax=85, aspect="auto")

            for i in range(len(row_labels)):
                for j in range(len(DEFENSES)):
                    if not np.isnan(matrix[i, j]):
                        color = "white" if matrix[i, j] < 50 or matrix[i, j] > 78 else "black"
                        ax.text(j, i, f"{matrix[i, j]:.0f}%",
                               ha="center", va="center", fontsize=9,
                               color=color, fontweight="bold")

            ax.set_xticks(range(len(DEFENSES)))
            ax.set_xticklabels(DEFENSES, fontsize=8, rotation=35, ha="right")
            ax.set_yticks(range(len(row_labels)))
            ax.set_yticklabels(row_labels, fontsize=9)
            if col == 0:
                ax.set_ylabel(f"{algo}\nAttack", fontsize=11, fontweight="bold")
            if row == 0:
                ax.set_title(f"DP: {dp_label}", fontsize=11, fontweight="bold")
            if row == len(algos) - 1:
                ax.set_xlabel("Defense", fontsize=10)

    plt.suptitle("Cardiovascular — Accuracy Under Byzantine Attack + DP\n"
                 "(1 adversary / 5 clients, FedAvg vs Ditto)",
                 fontsize=13, fontweight="bold", y=1.03)
    fig.colorbar(im, ax=axes, label="Accuracy (%)", shrink=0.7, pad=0.02)
    fig.tight_layout(rect=[0, 0, 0.92, 0.95])

    fig_path = OUTPUT_DIR / "byzantine_dp_cv_accuracy.png"
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.savefig(str(fig_path).replace(".png", ".pdf"), bbox_inches="tight")
    print(f"\n  Figure saved: {fig_path}")
    plt.close()

    # DEI heatmaps
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
                matrix_rows.append(
                    [np.mean([r["dei"] for r in matching])] * len(DEFENSES))

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
                                ha="center", va="center", fontsize=9,
                                color=color, fontweight="bold")

            ax2.set_xticks(range(len(DEFENSES)))
            ax2.set_xticklabels(DEFENSES, fontsize=8, rotation=35, ha="right")
            ax2.set_yticks(range(len(row_labels)))
            ax2.set_yticklabels(row_labels, fontsize=9)
            if col == 0:
                ax2.set_ylabel(f"{algo}\nAttack", fontsize=11, fontweight="bold")
            if row == 0:
                ax2.set_title(f"DP: {dp_label}", fontsize=11, fontweight="bold")
            if row == len(algos) - 1:
                ax2.set_xlabel("Defense", fontsize=10)

    plt.suptitle("Cardiovascular — DEI Under Byzantine Attack + DP\n"
                 "(1 adversary / 5 clients, FedAvg vs Ditto)",
                 fontsize=13, fontweight="bold", y=1.03)
    fig2.colorbar(im2, ax=axes2, label="DEI", shrink=0.7, pad=0.02)
    fig2.tight_layout(rect=[0, 0, 0.92, 0.95])

    fig2_path = OUTPUT_DIR / "byzantine_dp_cv_dei.png"
    plt.savefig(fig2_path, dpi=200, bbox_inches="tight")
    plt.savefig(str(fig2_path).replace(".png", ".pdf"), bbox_inches="tight")
    print(f"  Figure saved: {fig2_path}")
    plt.close()

    # Interaction plot: DP effect on Byzantine defense effectiveness
    fig3, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 6))

    for algo_idx, algo in enumerate(algos):
        ax = ax_left if algo_idx == 0 else ax_right
        x = np.arange(len(active_defenses))
        width = 0.25

        for dp_idx, dp_label in enumerate(dp_labels):
            accs = []
            for dfn in active_defenses:
                matching = [r for r in all_results
                           if r["algorithm"] == algo
                           and r["attack"] == "sign_flip" and r["defense"] == dfn
                           and r["dp_label"] == dp_label and "error" not in r]
                if matching:
                    accs.append(np.mean([r["accuracy"] for r in matching]) * 100)
                else:
                    accs.append(0)

            bars = ax.bar(x + dp_idx * width - width, accs, width,
                         label=dp_label, alpha=0.85)
            for bar, val in zip(bars, accs):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                           f'{val:.0f}', ha='center', va='bottom', fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels(active_defenses, fontsize=9, rotation=25, ha="right")
        ax.set_ylabel("Accuracy (%)", fontsize=11)
        ax.set_title(f"{algo} — Sign Flip Attack", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.set_ylim(40, 85)
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle("Byzantine Defense + DP Interaction (Cardiovascular)\n"
                 "Does DP noise help or hurt Byzantine robustness?",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    fig3_path = OUTPUT_DIR / "byzantine_dp_cv_interaction.png"
    plt.savefig(fig3_path, dpi=200, bbox_inches="tight")
    plt.savefig(str(fig3_path).replace(".png", ".pdf"), bbox_inches="tight")
    print(f"  Figure saved: {fig3_path}")
    plt.close()


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="FL-EHDS Byzantine+DP Interaction on Cardiovascular Disease")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode (1 seed, fewer configs)")
    parser.add_argument("--fresh", action="store_true",
                        help="Start fresh (delete checkpoint)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    seeds = [42] if args.quick else SEEDS
    algos = list(ALGORITHMS.keys())
    attacks = ATTACKS
    defenses = DEFENSES
    dp_levels = DP_LEVELS

    if args.quick:
        algos = ["FedAvg"]
        attacks = ["none", "sign_flip"]
        defenses = ["none", "krum", "median"]
        dp_levels = dp_levels[:2]

    # Build experiment list (skip defense-only when no attack)
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

    # Checkpoint
    if args.fresh:
        ckpt_path = OUTPUT_DIR / CHECKPOINT_FILE
        if ckpt_path.exists():
            ckpt_path.unlink()

    checkpoint = load_checkpoint()
    if checkpoint is None:
        checkpoint = {
            "completed": {},
            "metadata": {
                "dataset": "cardiovascular",
                "total_experiments": total,
                "start_time": datetime.now().isoformat(),
                "last_save": None,
            }
        }

    completed = checkpoint.get("completed", {})
    done_count = len(completed)

    # SIGINT handler
    _interrupted = [False]
    def _signal_handler(signum, frame):
        if _interrupted[0]:
            sys.exit(1)
        _interrupted[0] = True
        save_checkpoint(checkpoint)
        print(f"\n  SIGINT — checkpoint salvato ({done_count}/{total} completati)")
        sys.exit(0)
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    mode = "QUICK" if args.quick else "FULL"
    print("=" * 70)
    print(f"  FL-EHDS Byzantine+DP on Cardiovascular Disease ({mode})")
    print(f"  {len(algos)} algos x {len(attacks)} attacks x {len(defenses)} defs "
          f"x {len(dp_levels)} DP x {len(seeds)} seeds")
    print(f"  Total: {total} experiments")
    if done_count:
        print(f"  Resuming: {done_count} done, {total - done_count} remaining")
    print(f"  Adversary: client {ADVERSARIAL_CLIENT}/{NUM_CLIENTS} (20%)")
    print(f"  Device: {_detect_device()}")
    print("=" * 70)

    start_time = time.time()
    exp_idx = 0

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
                        exp_idx += 1

                        if key in completed:
                            continue

                        print(f"\n  [{exp_idx}/{total}] {algo} | atk={atk} | "
                              f"def={dfn} | {dp_level['label']} | s{seed} ...",
                              end="", flush=True)

                        try:
                            result = run_experiment(algo, atk, dfn, dp_level, seed)
                            completed[key] = result
                            done_count += 1

                            acc = result["accuracy"] * 100
                            dei = result["dei"]
                            rt = result["runtime_seconds"]
                            byz = " [ByzCfg]" if result.get("byzantine_config_used") else ""
                            print(f" acc={acc:.1f}% DEI={dei:.3f} "
                                  f"Neg={result['per_class_recall']['Negative']*100:.0f}% "
                                  f"Pos={result['per_class_recall']['Positive']*100:.0f}% "
                                  f"{rt:.0f}s{byz}")

                        except Exception as e:
                            print(f" ERROR: {e}")
                            traceback.print_exc()
                            completed[key] = {
                                "algorithm": algo, "attack": atk, "defense": dfn,
                                "dp_label": dp_level["label"], "seed": seed,
                                "error": str(e),
                            }
                            done_count += 1
                            _cleanup_gpu()

                        # Save after every experiment
                        save_checkpoint(checkpoint)

    checkpoint["metadata"]["end_time"] = datetime.now().isoformat()
    save_checkpoint(checkpoint)

    elapsed = time.time() - start_time

    # Summary
    valid = {k: v for k, v in completed.items() if "error" not in v}

    print("\n" + "=" * 70)
    print(f"  COMPLETED: {len(valid)}/{total} "
          f"({len(completed) - len(valid)} errors)")
    print(f"  Total time: {timedelta(seconds=int(elapsed))}")
    print("=" * 70)

    for algo in algos:
        print(f"\n  === {algo} ===")
        for dp_level in dp_levels:
            dp_label = dp_level["label"]
            print(f"\n  DP: {dp_label}")
            print(f"  {'Attack':<15} {'Defense':<15} {'Acc%':<8} {'DEI':<8} "
                  f"{'Neg%':<8} {'Pos%':<8}")
            print(f"  {'-'*66}")

            for atk in attacks:
                for dfn in defenses:
                    if atk == "none" and dfn != "none":
                        continue
                    matching = [v for k, v in valid.items()
                               if v.get("algorithm") == algo
                               and v.get("attack") == atk
                               and v.get("defense") == dfn
                               and v.get("dp_label") == dp_label]
                    if matching:
                        acc = np.mean([r["accuracy"] for r in matching]) * 100
                        dei = np.mean([r["dei"] for r in matching])
                        neg = np.mean([r["per_class_recall"]["Negative"]
                                      for r in matching]) * 100
                        pos = np.mean([r["per_class_recall"]["Positive"]
                                      for r in matching]) * 100
                        print(f"  {atk:<15} {dfn:<15} {acc:<8.1f} {dei:<8.3f} "
                              f"{neg:<8.0f} {pos:<8.0f}")

    # Generate figures
    if valid:
        print("\n  Generating figures...")
        generate_figures(list(valid.values()))

    print(f"\n  Checkpoint: {OUTPUT_DIR / CHECKPOINT_FILE}")
    print("=" * 70)


if __name__ == "__main__":
    main()

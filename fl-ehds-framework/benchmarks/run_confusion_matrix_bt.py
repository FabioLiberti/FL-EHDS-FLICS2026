#!/usr/bin/env python3
"""
FL-EHDS — Confusion Matrix on Brain Tumor MRI (FedAvg vs Ditto).

Retrains FedAvg and Ditto on Brain Tumor MRI (4-class) and generates
confusion matrices to diagnose HOW Ditto achieves +28pp over FedAvg:
  - Which classes benefit most from personalization?
  - Does Ditto reduce clinically dangerous misclassifications?
  - Per-client class distribution analysis

Total: 2 algos x 3 seeds = 6 experiments (~3-4h)

Features:
  - Incremental checkpoint after each experiment (crash-safe)
  - Round-level checkpoint saved to /tmp (resume mid-training)
  - Resume support: skips completed experiments automatically

Usage:
    cd fl-ehds-framework
    python -m benchmarks.run_confusion_matrix_bt [--quick] [--no-resume]

Output:
    benchmarks/paper_results_delta/confusion_matrix_bt.pdf
    benchmarks/paper_results_delta/checkpoint_confusion_bt.json

Author: Fabio Liberti
"""

import sys
import os
import json
import time
import gc
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

print("[boot] imports ok, loading framework...", flush=True)

FRAMEWORK_DIR = Path(__file__).parent.parent

# Use local copy of framework code if available (avoids OneDrive hangs)
_LOCAL_FRAMEWORK = Path("/tmp/fl_ehds_terminal")
if _LOCAL_FRAMEWORK.exists():
    # Point to /tmp so "from terminal.xxx import" works
    sys.path.insert(0, "/tmp/fl_ehds_terminal_root")
    # Create symlink structure: /tmp/fl_ehds_terminal_root/terminal -> /tmp/fl_ehds_terminal
    _root = Path("/tmp/fl_ehds_terminal_root")
    _root.mkdir(exist_ok=True)
    _link = _root / "terminal"
    if not _link.exists():
        _link.symlink_to(_LOCAL_FRAMEWORK)
    print("[boot] Using LOCAL framework code (/tmp)", flush=True)
else:
    sys.path.insert(0, str(FRAMEWORK_DIR))
    print("[boot] Using OneDrive framework code (may be slow)", flush=True)

print("[boot] importing torch...", flush=True)
import torch
print(f"[boot] torch {torch.__version__} ok", flush=True)

print("[boot] importing fl_trainer...", flush=True)
from terminal.fl_trainer import (
    ImageFederatedTrainer,
    _detect_device,
)
print("[boot] fl_trainer ok", flush=True)

ONEDRIVE_OUTPUT_DIR = FRAMEWORK_DIR / "benchmarks" / "paper_results_delta"

# ALL working directories on /tmp (local disk, no OneDrive interference)
OUTPUT_DIR = Path("/tmp/fl_ehds_results")
ROUND_CKPT_DIR = Path("/tmp/fl_ehds_round_ckpts")

# Data: use local /tmp copy (Brain_Tumor_2 or Brain_Tumor_local)
print("[init] Searching for local data...", flush=True)
DATA_DIR = None
_DATA_IS_LOCAL = False
for _d in ["/tmp/Brain_Tumor_2", "/tmp/Brain_Tumor_local"]:
    _p = Path(_d)
    if _p.exists() and _p.is_dir():
        _count = sum(1 for _ in _p.rglob("*.jpg"))
        print(f"[init]   {_d}: {_count} jpg files", flush=True)
        if _count > 1000:
            DATA_DIR = str(_p)
            _DATA_IS_LOCAL = True
            break
if DATA_DIR is None:
    # Fallback to OneDrive (slow but functional)
    for _d in [str(FRAMEWORK_DIR / "data" / "Brain_Tumor_2"),
               str(FRAMEWORK_DIR / "data" / "Brain_Tumor")]:
        if Path(_d).exists():
            DATA_DIR = _d
            break
if DATA_DIR is None:
    print("[ERROR] No Brain Tumor dataset found!", flush=True)
    sys.exit(1)
print(f"[init] Using: {DATA_DIR} (local={_DATA_IS_LOCAL})", flush=True)

# Same config as run_imaging_delta/completion for consistency
IMAGING_CONFIG = dict(
    num_clients=5,
    num_rounds=20,
    local_epochs=2,
    batch_size=32,
    learning_rate=0.0005,   # Brain Tumor specific
    model_type="resnet18",
    is_iid=False,
    alpha=0.5,
    freeze_backbone=False,
    freeze_level=2,
    use_fedbn=True,
    use_class_weights=True,
    use_amp=True,
    mu=0.1,
)

EARLY_STOPPING = dict(
    enabled=True,
    patience=4,
    min_delta=0.003,
    min_rounds=8,
    metric="accuracy",
)

ALGORITHMS = ["FedAvg", "Ditto"]
SEEDS = [42, 123, 456]
NUM_CLASSES = 4
CLASS_NAMES = {0: "Glioma", 1: "Healthy", 2: "Meningioma", 3: "Pituitary"}


class EarlyStoppingMonitor:
    def __init__(self, patience=4, min_delta=0.003, min_rounds=8, metric="accuracy"):
        self.patience = patience
        self.min_delta = min_delta
        self.min_rounds = min_rounds
        self.metric = metric
        self.best_value = -float('inf')
        self.counter = 0

    def check(self, round_num, metrics):
        value = metrics.get(self.metric, 0)
        if value > self.best_value + self.min_delta:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
        if round_num < self.min_rounds:
            return False
        return self.counter >= self.patience


def collect_predictions(trainer) -> Tuple[np.ndarray, np.ndarray, Dict[int, Tuple[np.ndarray, np.ndarray]]]:
    """Collect all predictions + per-client predictions after training."""
    model = trainer.global_model
    model.eval()

    all_preds = []
    all_labels = []
    per_client = {}

    is_hpfl = trainer.algorithm == "HPFL"
    is_ditto = trainer.algorithm == "Ditto"

    if is_hpfl:
        saved_cls = {n: p.data.clone() for n, p in model.named_parameters()
                     if n in trainer._hpfl_classifier_names}

    # For Ditto, use local models if available
    if is_ditto and hasattr(trainer, 'personalized_models') and trainer.personalized_models:
        with torch.no_grad():
            for cid in range(trainer.num_clients):
                local_model = trainer.personalized_models[cid]
                local_model.eval()

                X, y = trainer.client_test_data[cid]
                client_preds = []
                client_labels = []

                for i in range(0, len(y), trainer.batch_size):
                    X_batch = torch.FloatTensor(X[i:i+trainer.batch_size]).to(trainer.device)
                    y_batch = y[i:i+trainer.batch_size]

                    if trainer.use_amp and hasattr(torch, 'autocast'):
                        with torch.autocast(device_type=str(trainer.device).split(':')[0], dtype=torch.float16):
                            out = local_model(X_batch)
                    else:
                        out = local_model(X_batch)

                    preds = out.argmax(dim=1).cpu().numpy()
                    labels = y_batch if isinstance(y_batch, np.ndarray) else y_batch.numpy()
                    client_preds.extend(preds.tolist())
                    client_labels.extend(labels.tolist())

                per_client[cid] = (np.array(client_labels), np.array(client_preds))
                all_preds.extend(client_preds)
                all_labels.extend(client_labels)

        return np.array(all_labels), np.array(all_preds), per_client

    # Standard path for FedAvg (and fallback for Ditto without personalized_models)
    with torch.no_grad():
        for cid in range(trainer.num_clients):
            if is_hpfl:
                for n, p in model.named_parameters():
                    if n in trainer._hpfl_classifier_names:
                        p.data.copy_(trainer.client_classifiers[cid][n])

            X, y = trainer.client_test_data[cid]
            client_preds = []
            client_labels = []

            for i in range(0, len(y), trainer.batch_size):
                X_batch = torch.FloatTensor(X[i:i+trainer.batch_size]).to(trainer.device)
                y_batch = y[i:i+trainer.batch_size]

                if trainer.use_amp and hasattr(torch, 'autocast'):
                    with torch.autocast(device_type=str(trainer.device).split(':')[0], dtype=torch.float16):
                        out = model(X_batch)
                else:
                    out = model(X_batch)

                preds = out.argmax(dim=1).cpu().numpy()
                labels = y_batch if isinstance(y_batch, np.ndarray) else y_batch.numpy()
                client_preds.extend(preds.tolist())
                client_labels.extend(labels.tolist())

            per_client[cid] = (np.array(client_labels), np.array(client_preds))
            all_preds.extend(client_preds)
            all_labels.extend(client_labels)

    if is_hpfl:
        for n, p in model.named_parameters():
            if n in trainer._hpfl_classifier_names:
                p.data.copy_(saved_cls[n])

    return np.array(all_labels), np.array(all_preds), per_client


def compute_confusion_matrix(y_true, y_pred, num_classes=NUM_CLASSES):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


# ================================================================
# Round-level checkpointing (saved to /tmp, NOT OneDrive)
# ================================================================

def _round_ckpt_path(algo, seed):
    """Single file per experiment, overwritten each round."""
    return ROUND_CKPT_DIR / f"{algo}_s{seed}_latest.pt"


def _save_round_ckpt(algo, seed, round_num, trainer, best_acc, best_round,
                     es, early_stopped=False):
    """Save model state after each round for mid-training resume."""
    ROUND_CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = {
        'round': round_num,
        'best_acc': best_acc,
        'best_round': best_round,
        'early_stopped': early_stopped,
        'es_best': es.best_value if es else 0,
        'es_counter': es.counter if es else 0,
        'global_model': trainer.global_model.state_dict(),
    }
    # For Ditto: save personalized models
    if hasattr(trainer, 'personalized_models') and trainer.personalized_models:
        ckpt['personalized_models'] = {
            str(cid): m.state_dict()
            for cid, m in trainer.personalized_models.items()
        }
    # Save history length for LR scheduling consistency
    ckpt['history_len'] = len(trainer.history)

    torch.save(ckpt, _round_ckpt_path(algo, seed))


def _load_round_ckpt(algo, seed):
    """Load round-level checkpoint if it exists."""
    path = _round_ckpt_path(algo, seed)
    if path.exists():
        try:
            return torch.load(path, map_location='cpu', weights_only=False)
        except Exception as e:
            print(f"    Warning: Could not load round checkpoint: {e}")
    return None


def _clean_round_ckpt(algo, seed):
    """Remove round checkpoint after experiment completes successfully."""
    path = _round_ckpt_path(algo, seed)
    if path.exists():
        path.unlink()


def run_experiment(algorithm, seed, config, quick=False):
    """Train and collect predictions with round-level checkpointing."""
    num_rounds = 3 if quick else config["num_rounds"]
    local_epochs = 1 if quick else config["local_epochs"]

    trainer = ImageFederatedTrainer(
        data_dir=DATA_DIR,
        num_clients=config["num_clients"],
        algorithm=algorithm,
        local_epochs=local_epochs,
        batch_size=config["batch_size"],
        learning_rate=config["learning_rate"],
        is_iid=config["is_iid"],
        alpha=config["alpha"],
        mu=config.get("mu", 0.1),
        seed=seed,
        model_type=config["model_type"],
        freeze_backbone=config.get("freeze_backbone", False),
        freeze_level=config.get("freeze_level"),
        use_fedbn=config.get("use_fedbn", False),
        use_class_weights=config.get("use_class_weights", True),
        use_amp=config.get("use_amp", True),
    )
    trainer.num_rounds = num_rounds

    es = None
    if not quick and EARLY_STOPPING.get("enabled"):
        es = EarlyStoppingMonitor(
            **{k: v for k, v in EARLY_STOPPING.items() if k != "enabled"}
        )

    best_acc = 0.0
    best_round = 0
    start_round = 0

    # --- Check for round-level checkpoint (mid-training resume) ---
    round_ckpt = _load_round_ckpt(algorithm, seed)
    if round_ckpt is not None:
        start_round = round_ckpt['round']
        best_acc = round_ckpt['best_acc']
        best_round = round_ckpt['best_round']

        # Restore model state
        trainer.global_model.load_state_dict(round_ckpt['global_model'])

        # Restore Ditto personalized models
        if 'personalized_models' in round_ckpt and hasattr(trainer, 'personalized_models'):
            for cid_str, state in round_ckpt['personalized_models'].items():
                trainer.personalized_models[int(cid_str)].load_state_dict(state)

        # Restore early stopping state
        if es:
            es.best_value = round_ckpt.get('es_best', -float('inf'))
            es.counter = round_ckpt.get('es_counter', 0)

        # Restore history length for LR consistency
        history_len = round_ckpt.get('history_len', start_round)
        # Pad history with dummy entries so train_round gets correct LR
        while len(trainer.history) < history_len:
            trainer.history.append(None)

        if round_ckpt.get('early_stopped', False):
            print(f"    -> Restored: already early-stopped at R{start_round}")
            y_true, y_pred, per_client = collect_predictions(trainer)
            _clean_round_ckpt(algorithm, seed)
            del trainer
            gc.collect()
            return y_true, y_pred, per_client, best_acc, best_round

        print(f"    -> Resuming from R{start_round}/{num_rounds} "
              f"(best:{best_acc:.1%} at r{best_round})", flush=True)

    # --- Training loop ---
    for r in range(start_round, num_rounds):
        rr = trainer.train_round(r)
        if rr.global_acc > best_acc:
            best_acc = rr.global_acc
            best_round = r + 1

        should_stop = False
        if es and es.check(r + 1, {"accuracy": rr.global_acc}):
            should_stop = True

        # Save round checkpoint AFTER each round
        _save_round_ckpt(algorithm, seed, r + 1, trainer, best_acc, best_round,
                         es, early_stopped=should_stop)

        print("    R{}/{} | Acc:{:.1%} | Best:{:.1%}(r{}) | ckpt".format(
            r + 1, num_rounds, rr.global_acc, best_acc, best_round),
            end="", flush=True)

        if should_stop:
            print(f"\n    -> Early stop at R{r + 1}")
            break
        else:
            print("", flush=True)  # newline

    y_true, y_pred, per_client = collect_predictions(trainer)

    # Clean up round checkpoint (experiment completed successfully)
    _clean_round_ckpt(algorithm, seed)

    del trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        try:
            torch.mps.empty_cache()
        except Exception:
            pass
    gc.collect()

    return y_true, y_pred, per_client, best_acc, best_round


# ================================================================
# Experiment-level checkpointing (saved to OneDrive output dir)
# ================================================================

def _checkpoint_path():
    return OUTPUT_DIR / "checkpoint_confusion_bt.json"


def _save_checkpoint(results, all_cms, per_client_cms, seeds, completed_keys):
    """Save checkpoint after each experiment for crash-safety."""
    checkpoint = {
        "results": results,
        "aggregated_cms": {algo: cm.tolist() if isinstance(cm, np.ndarray) else cm
                          for algo, cm in all_cms.items()},
        "per_client_cms": per_client_cms,
        "completed": list(completed_keys),
        "metadata": {
            "algorithms": ALGORITHMS,
            "seeds": seeds,
            "class_names": CLASS_NAMES,
            "num_classes": NUM_CLASSES,
            "dataset": "Brain_Tumor",
            "config": IMAGING_CONFIG,
            "timestamp": datetime.now().isoformat(),
        }
    }
    with open(_checkpoint_path(), "w") as f:
        json.dump(checkpoint, f, indent=2, default=str)


def _load_checkpoint():
    """Load existing checkpoint for resume support."""
    cp = _checkpoint_path()
    if not cp.exists():
        return None
    try:
        with open(cp) as f:
            data = json.load(f)
        # Must have 'completed' key (new format with incremental saves)
        if "completed" not in data:
            return None
        return data
    except Exception:
        return None


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Confusion Matrix for Brain Tumor MRI (FedAvg vs Ditto)"
    )
    parser.add_argument("--quick", action="store_true",
                        help="Quick validation (1 seed, 3 rounds)")
    parser.add_argument("--no-resume", action="store_true",
                        help="Start from scratch, ignore existing checkpoint")
    args = parser.parse_args()

    seeds = [42] if args.quick else SEEDS
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Resume logic ---
    completed_keys = set()
    results = {}
    all_cms = {}
    per_client_cms = {}

    if not args.no_resume:
        prev = _load_checkpoint()
        if prev and set(prev["metadata"].get("seeds", [])) == set(seeds):
            completed_keys = set(prev.get("completed", []))
            results = prev.get("results", {})
            # Restore numpy cms
            for algo, cm_list in prev.get("aggregated_cms", {}).items():
                all_cms[algo] = np.array(cm_list)
            per_client_cms = prev.get("per_client_cms", {})
            if completed_keys:
                print(f"  RESUMING: {len(completed_keys)} experiments already done:")
                for k in sorted(completed_keys):
                    print(f"    - {k}")
                print()

    print("=" * 60)
    print("  FL-EHDS Confusion Matrix — Brain Tumor MRI (4-class)")
    print("  {} algos x {} seeds = {} experiments".format(
        len(ALGORITHMS), len(seeds), len(ALGORITHMS) * len(seeds)))
    print("  Classes: {}".format(", ".join(CLASS_NAMES.values())))
    print("  Device: {}".format(_detect_device()))
    print("  Mode: {}".format("QUICK" if args.quick else "FULL"))
    if completed_keys:
        remaining = len(ALGORITHMS) * len(seeds) - len(completed_keys)
        print(f"  Resume: {len(completed_keys)} done, {remaining} remaining")
    print(f"  Data: {DATA_DIR}")
    print(f"  Data source: {'LOCAL' if _DATA_IS_LOCAL else 'OneDrive (may be slow)'}")
    print(f"  Round checkpoints: {ROUND_CKPT_DIR}")
    if not _DATA_IS_LOCAL:
        print(f"  TIP: Copy data locally for speed: cp -r data/Brain_Tumor_2 /tmp/Brain_Tumor_2")
    print("=" * 60)

    for algo in ALGORITHMS:
        if algo not in all_cms:
            all_cms[algo] = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
        if algo not in per_client_cms:
            per_client_cms[algo] = {
                str(cid): np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int).tolist()
                for cid in range(IMAGING_CONFIG["num_clients"])
            }
        if algo not in results:
            results[algo] = []

        for seed in seeds:
            key = f"{algo}_s{seed}"
            if key in completed_keys:
                print(f"\n  {algo} seed={seed} ... SKIPPED (already done)", flush=True)
                continue

            print("\n  {} seed={} ...".format(algo, seed), flush=True)
            start = time.time()

            y_true, y_pred, per_client, best_acc, best_round = run_experiment(
                algo, seed, IMAGING_CONFIG, quick=args.quick
            )

            cm = compute_confusion_matrix(y_true, y_pred)
            all_cms[algo] += cm
            elapsed = time.time() - start

            client_cms = {}
            for cid, (cl, cp) in per_client.items():
                ccm = compute_confusion_matrix(cl, cp)
                # Accumulate into per_client_cms
                existing = np.array(per_client_cms[algo][str(cid)])
                per_client_cms[algo][str(cid)] = (existing + ccm).tolist()
                client_cms[cid] = ccm.tolist()

            per_class_recall = {}
            for c in range(NUM_CLASSES):
                total = cm[c].sum()
                per_class_recall[CLASS_NAMES[c]] = float(cm[c, c] / total) if total > 0 else 0.0

            acc = np.trace(cm) / cm.sum()

            results[algo].append({
                "seed": seed,
                "accuracy": float(acc),
                "best_accuracy": float(best_acc),
                "best_round": best_round,
                "confusion_matrix": cm.tolist(),
                "per_class_recall": per_class_recall,
                "per_client_cms": client_cms,
                "runtime_seconds": round(elapsed, 1),
            })

            recall_str = " | ".join(
                f"{CLASS_NAMES[c][:4]}:{per_class_recall[CLASS_NAMES[c]]:.0%}"
                for c in range(NUM_CLASSES)
            )
            print(f"  -> Acc:{acc:.1%} | {recall_str} | {elapsed:.0f}s", flush=True)

            # --- INCREMENTAL SAVE after each experiment ---
            completed_keys.add(key)
            _save_checkpoint(results, all_cms, per_client_cms, seeds, completed_keys)
            print(f"  -> Experiment checkpoint saved ({len(completed_keys)}/{len(ALGORITHMS)*len(seeds)})",
                  flush=True)

        # Print aggregated
        algo_cm = all_cms[algo]
        print(f"\n  {algo} aggregated CM ({len(seeds)} seeds):")
        print(f"  {'':>12} " + "  ".join(f"{CLASS_NAMES[c]:>10}" for c in range(NUM_CLASSES)))
        for i in range(NUM_CLASSES):
            row = "  ".join(f"{algo_cm[i, j]:>10d}" for j in range(NUM_CLASSES))
            print(f"  {CLASS_NAMES[i]:>12} {row}")
        agg_acc = np.trace(algo_cm) / algo_cm.sum() if algo_cm.sum() > 0 else 0
        print(f"  Aggregated accuracy: {agg_acc:.1%}")

    # Final save
    _save_checkpoint(results, all_cms, per_client_cms, seeds, completed_keys)

    # ================================================================
    # Generate figure: 2 aggregate CMs + per-class recall comparison
    # ================================================================
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec

        fig = plt.figure(figsize=(14, 10))
        gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 0.8], hspace=0.35, wspace=0.3)

        # --- Top row: aggregate confusion matrices ---
        for col, algo in enumerate(ALGORITHMS):
            ax = fig.add_subplot(gs[0, col])
            cm = all_cms[algo]
            total_per_row = cm.sum(axis=1, keepdims=True)
            total_per_row[total_per_row == 0] = 1  # avoid /0
            cm_pct = cm / total_per_row * 100

            im = ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=100)

            for i in range(NUM_CLASSES):
                for j in range(NUM_CLASSES):
                    color = "white" if cm_pct[i, j] > 50 else "black"
                    ax.text(j, i, f"{cm[i, j]}\n({cm_pct[i, j]:.0f}%)",
                            ha="center", va="center", fontsize=9, color=color,
                            fontweight="bold")

            class_labels = [CLASS_NAMES[c][:6] for c in range(NUM_CLASSES)]
            ax.set_xticks(range(NUM_CLASSES))
            ax.set_yticks(range(NUM_CLASSES))
            ax.set_xticklabels(class_labels, fontsize=8, rotation=30)
            ax.set_yticklabels(class_labels, fontsize=8)
            ax.set_xlabel("Predicted", fontsize=10)
            if col == 0:
                ax.set_ylabel("True", fontsize=10)

            agg_acc = np.trace(cm) / cm.sum() * 100
            ax.set_title(f"{algo}\nAcc: {agg_acc:.1f}%",
                         fontsize=11, fontweight="bold")

        # --- Bottom row: per-class recall comparison ---
        ax_recall = fig.add_subplot(gs[1, :])

        x = np.arange(NUM_CLASSES)
        width = 0.35

        fedavg_recalls = []
        ditto_recalls = []
        for c in range(NUM_CLASSES):
            fa_cm = all_cms["FedAvg"]
            fa_recall = fa_cm[c, c] / fa_cm[c].sum() * 100 if fa_cm[c].sum() > 0 else 0
            fedavg_recalls.append(fa_recall)

            dt_cm = all_cms["Ditto"]
            dt_recall = dt_cm[c, c] / dt_cm[c].sum() * 100 if dt_cm[c].sum() > 0 else 0
            ditto_recalls.append(dt_recall)

        bars1 = ax_recall.bar(x - width / 2, fedavg_recalls, width,
                              label="FedAvg", color="#4C72B0", alpha=0.85)
        bars2 = ax_recall.bar(x + width / 2, ditto_recalls, width,
                              label="Ditto", color="#C44E52", alpha=0.85)

        for bars in [bars1, bars2]:
            for bar in bars:
                h = bar.get_height()
                ax_recall.text(bar.get_x() + bar.get_width() / 2, h + 1,
                               f"{h:.0f}%", ha="center", va="bottom",
                               fontsize=9, fontweight="bold")

        # Delta labels
        for i, (fa, dt) in enumerate(zip(fedavg_recalls, ditto_recalls)):
            delta = dt - fa
            if abs(delta) > 0.5:
                sign = "+" if delta > 0 else ""
                ax_recall.annotate(
                    f"{sign}{delta:.0f}pp",
                    xy=(i + width / 2, max(fa, dt) + 5),
                    fontsize=8, color="green" if delta > 0 else "red",
                    ha="center", fontweight="bold"
                )

        ax_recall.set_xlabel("Tumor Class", fontsize=11)
        ax_recall.set_ylabel("Per-Class Recall (%)", fontsize=11)
        ax_recall.set_title(
            f"Per-Class Recall: FedAvg vs Ditto (aggregated over {len(seeds)} seeds)",
            fontsize=11, fontweight="bold"
        )
        ax_recall.set_xticks(x)
        ax_recall.set_xticklabels([CLASS_NAMES[c] for c in range(NUM_CLASSES)], fontsize=10)
        ax_recall.set_ylim(0, 110)
        ax_recall.axhline(y=25, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax_recall.text(3.5, 26, "random baseline (4-class)", fontsize=7, color="gray")
        ax_recall.legend(fontsize=10)
        ax_recall.grid(axis="y", alpha=0.3)

        plt.suptitle(
            f"Brain Tumor MRI: FedAvg vs Ditto Confusion Matrices\n"
            f"(4-class, aggregated over {len(seeds)} seeds)",
            fontsize=13, fontweight="bold", y=0.98
        )

        fig_path = OUTPUT_DIR / "confusion_matrix_bt"
        plt.savefig(str(fig_path) + ".png", dpi=200, bbox_inches="tight")
        plt.savefig(str(fig_path) + ".pdf", bbox_inches="tight")
        print(f"\n  Figure saved: {fig_path}")
        plt.close()

    except ImportError:
        print("\n  matplotlib not available — skipping figure generation")

    # Copy final results to OneDrive for persistence
    import shutil
    ONEDRIVE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for fname in ["checkpoint_confusion_bt.json", "confusion_matrix_bt.png",
                   "confusion_matrix_bt.pdf"]:
        src = OUTPUT_DIR / fname
        if src.exists():
            shutil.copy2(str(src), str(ONEDRIVE_OUTPUT_DIR / fname))
    print(f"\n  Results copied to OneDrive: {ONEDRIVE_OUTPUT_DIR}")

    print("\n" + "=" * 60)
    print("  DONE.")
    print(f"  Checkpoint: {OUTPUT_DIR / 'checkpoint_confusion_bt.json'}")
    print(f"  Figure:     {OUTPUT_DIR / 'confusion_matrix_bt.pdf'}")
    print("=" * 60)


if __name__ == "__main__":
    main()

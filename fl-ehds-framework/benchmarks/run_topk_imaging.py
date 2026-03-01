#!/usr/bin/env python3
"""
FL-EHDS Experiment — Top-K Sparsification on Imaging.

Tests communication-efficient FL on imaging datasets using Top-K
gradient sparsification. Extends PTB-XL Top-K results to imaging.

Design:
  - Datasets: Brain Tumor (4-class), Skin Cancer (binary), Chest X-ray (binary)
  - K ratios: baseline (100%), 5%, 1%
  - Algorithm: FedAvg
  - Seeds: 42, 123, 456
  - Total: 3 datasets x 3 K x 3 seeds = 27 experiments (~3h on M3)

Usage:
    cd fl-ehds-framework
    python -m benchmarks.run_topk_imaging [--quick] [--fresh]

Output: benchmarks/paper_results_delta/checkpoint_topk_imaging.json

Author: Fabio Liberti
"""

import sys
import os
import json
import time
import shutil
import signal
import tempfile
import argparse
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Any

import numpy as np

FRAMEWORK_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(FRAMEWORK_DIR))

import torch

from terminal.fl_trainer import (
    ImageFederatedTrainer,
    _detect_device,
)

# ======================================================================
# Constants
# ======================================================================

OUTPUT_DIR = FRAMEWORK_DIR / "benchmarks" / "paper_results_delta"
CHECKPOINT_FILE = "checkpoint_topk_imaging.json"
LOG_FILE = "experiment_topk_imaging.log"

SEEDS = [42, 123, 456]
K_RATIOS = [1.0, 0.05, 0.01]  # 100% (baseline), 5%, 1%
K_LABELS = {1.0: "baseline", 0.05: "top5pct", 0.01: "top1pct"}

IMAGING_DATASETS = {
    "Brain_Tumor": {
        "data_dir": str(FRAMEWORK_DIR / "data" / "Brain_Tumor"),
        "num_classes": 4,
        "short": "BT",
    },
    "Skin_Cancer": {
        "data_dir": str(FRAMEWORK_DIR / "data" / "Skin Cancer"),  # space in dir name
        "num_classes": 2,
        "short": "SC",
    },
    "chest_xray": {
        "data_dir": str(FRAMEWORK_DIR / "data" / "chest_xray"),
        "num_classes": 2,
        "short": "CX",
    },
}

IMAGING_CONFIG = dict(
    num_clients=5,
    num_rounds=15,
    local_epochs=2,
    batch_size=32,
    learning_rate=0.001,
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


# ======================================================================
# Logging
# ======================================================================

_log_file = None


def log(msg, also_print=True):
    ts = datetime.now().strftime("%H:%M:%S")
    line = "[{}] {}".format(ts, msg)
    if also_print:
        print(line, flush=True)
    if _log_file:
        try:
            _log_file.write(line + "\n")
            _log_file.flush()
        except Exception:
            pass


# ======================================================================
# Checkpoint (atomic write with backup)
# ======================================================================

def save_checkpoint(data):
    # type: (Dict) -> None
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / CHECKPOINT_FILE
    bak = OUTPUT_DIR / (CHECKPOINT_FILE + ".bak")
    data["metadata"]["last_save"] = datetime.now().isoformat()

    fd, tmp = tempfile.mkstemp(dir=str(OUTPUT_DIR), prefix=".topk_img_", suffix=".tmp")
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


def load_checkpoint():
    # type: () -> Optional[Dict]
    path = OUTPUT_DIR / CHECKPOINT_FILE
    bak = OUTPUT_DIR / (CHECKPOINT_FILE + ".bak")
    for p in [path, bak]:
        if p.exists():
            try:
                with open(p) as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                continue
    return None


# ======================================================================
# GPU cleanup
# ======================================================================

def _cleanup_gpu():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        try:
            torch.mps.empty_cache()
        except Exception:
            pass
    import gc
    gc.collect()


# ======================================================================
# Top-K Sparsification
# ======================================================================

def apply_topk_to_model_update(model, old_state, k_ratio):
    """Apply Top-k sparsification to the model update."""
    if k_ratio >= 1.0:
        return {"compression_ratio": 1.0, "kept_params": 0, "total_params": 0}

    total_params = 0
    kept_params = 0

    with torch.no_grad():
        for name, param in model.named_parameters():
            if name not in old_state:
                continue
            delta = param.data - old_state[name]
            flat = delta.flatten()
            n = flat.numel()
            k = max(1, int(n * k_ratio))
            total_params += n
            kept_params += k

            # Top-k selection by magnitude
            _, indices = torch.topk(flat.abs(), k)
            mask = torch.zeros_like(flat)
            mask[indices] = 1.0
            sparse_delta = flat * mask

            # Apply sparsified update
            param.data.copy_(old_state[name] + sparse_delta.view_as(param.data))

    return {
        "compression_ratio": round(kept_params / max(total_params, 1), 4),
        "kept_params": kept_params,
        "total_params": total_params,
    }


# ======================================================================
# Re-evaluation after sparsification
# ======================================================================

def _evaluate_after_sparsification(trainer):
    """Re-evaluate global model accuracy across all clients after sparsification."""
    model = trainer.global_model
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for cid in range(trainer.num_clients):
            X, y = trainer.client_test_data[cid]
            X_t = torch.FloatTensor(X).to(trainer.device) if isinstance(X, np.ndarray) else X.to(trainer.device)
            y_t = torch.LongTensor(y).to(trainer.device) if isinstance(y, np.ndarray) else y.to(trainer.device)
            for i in range(0, len(y_t), 64):
                out = model(X_t[i:i + 64])
                correct += (out.argmax(1) == y_t[i:i + 64]).sum().item()
                total += len(y_t[i:i + 64])
    return correct / total if total > 0 else 0.0


# ======================================================================
# Training
# ======================================================================

def run_single_experiment(dataset_name, ds_config, k_ratio, seed, cfg, quick=False):
    """Run FedAvg on an imaging dataset with optional Top-K sparsification."""
    start = time.time()
    label = K_LABELS[k_ratio]
    num_rounds = 3 if quick else cfg["num_rounds"]

    trainer = ImageFederatedTrainer(
        data_dir=ds_config["data_dir"],
        num_clients=cfg["num_clients"],
        algorithm="FedAvg",
        local_epochs=cfg["local_epochs"],
        batch_size=cfg["batch_size"],
        learning_rate=cfg["learning_rate"],
        is_iid=cfg["is_iid"],
        alpha=cfg["alpha"],
        mu=cfg["mu"],
        seed=seed,
        model_type=cfg["model_type"],
        freeze_backbone=cfg.get("freeze_backbone", False),
        freeze_level=cfg.get("freeze_level"),
        use_fedbn=cfg.get("use_fedbn", False),
        use_class_weights=cfg.get("use_class_weights", True),
        use_amp=cfg.get("use_amp", True),
    )

    history = []
    best_acc = 0.0
    compression_stats = []

    for r in range(num_rounds):
        # Save state before round
        old_state = {n: p.data.clone() for n, p in trainer.global_model.named_parameters()}

        # Run FL round (aggregation happens internally)
        rr = trainer.train_round(r)

        # Apply Top-k sparsification to the update
        stats = apply_topk_to_model_update(trainer.global_model, old_state, k_ratio)
        compression_stats.append(stats)

        # Re-evaluate after sparsification (if applied)
        if k_ratio < 1.0:
            acc_after = _evaluate_after_sparsification(trainer)
        else:
            acc_after = rr.global_acc

        metrics = {
            "round": r + 1,
            "accuracy": acc_after,
            "accuracy_before_sparsification": rr.global_acc,
            "loss": rr.global_loss,
            "f1": rr.global_f1,
        }
        history.append(metrics)

        if acc_after > best_acc:
            best_acc = acc_after

        log("  R{}/{} | {} | {} | s{} | Acc:{:.1%} (pre-sparse:{:.1%}) | Best:{:.1%}".format(
            r + 1, num_rounds, dataset_name, label, seed,
            acc_after, rr.global_acc, best_acc))

    elapsed = time.time() - start
    avg_compression = np.mean([s["compression_ratio"] for s in compression_stats])

    result = {
        "dataset": dataset_name,
        "algorithm": "FedAvg",
        "k_ratio": k_ratio,
        "label": label,
        "seed": seed,
        "history": history,
        "final_metrics": history[-1] if history else {},
        "best_accuracy": best_acc,
        "actual_rounds": len(history),
        "runtime_seconds": round(elapsed, 1),
        "avg_compression_ratio": avg_compression,
        "bandwidth_savings_pct": (1 - avg_compression) * 100,
        "total_params": compression_stats[0]["total_params"] if compression_stats else 0,
    }

    _cleanup_gpu()
    return result


# ======================================================================
# Figure generation
# ======================================================================

def generate_figure(completed, ds_order, k_ratios, seeds):
    """Line chart: one subplot per dataset, x=K ratio, y=accuracy, with error bars."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log("matplotlib not available -- skipping figure")
        return

    fig, axes = plt.subplots(1, len(ds_order), figsize=(6 * len(ds_order), 5), squeeze=False)

    for col, ds_name in enumerate(ds_order):
        ax = axes[0][col]
        ds_short = IMAGING_DATASETS[ds_name]["short"]

        x_vals = []
        y_means = []
        y_stds = []

        for k_ratio in k_ratios:
            k_label = K_LABELS[k_ratio]
            accs = []
            for seed in seeds:
                key = "{}_{}_{}_s{}".format(ds_short, "FedAvg", k_label, seed)
                # Also try the canonical key format
                key2 = "{}_{}_s{}".format(ds_short, k_label, seed)
                entry = completed.get(key) or completed.get(key2)
                if entry and "best_accuracy" in entry:
                    accs.append(entry["best_accuracy"] * 100)
            if accs:
                x_vals.append(k_ratio)
                y_means.append(np.mean(accs))
                y_stds.append(np.std(accs))

        if x_vals:
            ax.errorbar(
                x_vals, y_means, yerr=y_stds,
                marker="o", capsize=4, linewidth=2, markersize=7,
                color="#1f77b4", label="FedAvg",
            )

        ax.set_xlabel("K ratio (fraction kept)", fontsize=12)
        ax.set_ylabel("Accuracy (%)", fontsize=12)
        ax.set_title("{} ({}-class)".format(
            ds_name.replace("_", " "),
            IMAGING_DATASETS[ds_name]["num_classes"]),
            fontsize=13, fontweight="bold")
        ax.set_xscale("log")
        ax.set_xticks(k_ratios)
        ax.get_xaxis().set_major_formatter(
            plt.FuncFormatter(lambda v, _: "{:.0%}".format(v)))
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)

    fig.suptitle("Top-K Sparsification on Imaging Datasets", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    fig_path = OUTPUT_DIR / "topk_imaging_accuracy.png"
    plt.savefig(str(fig_path), dpi=200, bbox_inches="tight")
    plt.savefig(str(fig_path).replace(".png", ".pdf"), bbox_inches="tight")
    log("Figure saved: {}".format(fig_path))
    plt.close()


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="FL-EHDS Top-K Sparsification on Imaging Datasets")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 1 seed, 1 dataset (Brain_Tumor), 2 K ratios")
    parser.add_argument("--fresh", action="store_true",
                        help="Delete existing checkpoint and start fresh")
    args = parser.parse_args()

    global _log_file
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _log_file = open(OUTPUT_DIR / LOG_FILE, "a")

    # Determine scope based on --quick flag
    seeds = [42] if args.quick else SEEDS
    k_ratios = [1.0, 0.05] if args.quick else K_RATIOS
    ds_names = ["Brain_Tumor"] if args.quick else list(IMAGING_DATASETS.keys())

    cfg = IMAGING_CONFIG.copy()

    # Build experiment list: datasets x K ratios x seeds
    experiments = []
    for ds_name in ds_names:
        for k_ratio in k_ratios:
            for seed in seeds:
                experiments.append((ds_name, k_ratio, seed))

    total_exps = len(experiments)

    # --fresh: remove checkpoint
    if args.fresh:
        ckpt_path = OUTPUT_DIR / CHECKPOINT_FILE
        if ckpt_path.exists():
            ckpt_path.unlink()
            log("Removed existing checkpoint (--fresh)")

    # Load or create checkpoint
    checkpoint_data = load_checkpoint()
    if checkpoint_data is None:
        checkpoint_data = {
            "completed": {},
            "metadata": {
                "total_experiments": total_exps,
                "k_ratios": k_ratios,
                "seeds": seeds,
                "datasets": ds_names,
                "imaging_config": cfg,
                "start_time": datetime.now().isoformat(),
                "last_save": None,
                "version": "topk_imaging_v1",
            },
        }

    # Signal handlers
    _interrupted = [False]

    def _signal_handler(signum, frame):
        if _interrupted[0]:
            sys.exit(1)
        _interrupted[0] = True
        save_checkpoint(checkpoint_data)
        print("\n  Checkpoint saved. Resume by re-running without --fresh.")
        sys.exit(0)

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # Banner
    mode = "QUICK" if args.quick else "FULL"
    log("\n" + "=" * 66)
    log("  FL-EHDS Top-K Sparsification on Imaging ({})".format(mode))
    log("  Datasets:  {}".format(ds_names))
    log("  K ratios:  {}".format(k_ratios))
    log("  Seeds:     {}".format(seeds))
    log("  Total:     {} experiments".format(total_exps))
    log("  Device:    {}".format(_detect_device()))
    log("  Rounds:    {}".format(cfg["num_rounds"]))
    log("  Output:    {}".format(OUTPUT_DIR))
    log("=" * 66)

    global_start = time.time()
    completed = checkpoint_data.get("completed", {})
    done_count = len(completed)

    for exp_idx, (ds_name, k_ratio, seed) in enumerate(experiments, 1):
        ds_short = IMAGING_DATASETS[ds_name]["short"]
        k_label = K_LABELS[k_ratio]
        key = "{}_{}_s{}".format(ds_short, k_label, seed)

        if key in completed:
            continue

        if _interrupted[0]:
            break

        log("\n--- [{}/{}] {} | {} | seed={} ---".format(
            exp_idx, total_exps, ds_name, k_label, seed))

        try:
            result = run_single_experiment(
                dataset_name=ds_name,
                ds_config=IMAGING_DATASETS[ds_name],
                k_ratio=k_ratio,
                seed=seed,
                cfg=cfg,
                quick=args.quick,
            )

            completed[key] = result
            done_count += 1
            checkpoint_data["metadata"]["done"] = done_count
            checkpoint_data["metadata"]["elapsed_seconds"] = time.time() - global_start
            save_checkpoint(checkpoint_data)

            acc = result["best_accuracy"] * 100
            bw = result["bandwidth_savings_pct"]
            rt = result["runtime_seconds"]
            log("--- Done: {} | {} | s{} | Best={:.1f}% | BW savings: {:.0f}% | "
                "{:.0f}s | [{}/{}] ---".format(
                    ds_name, k_label, seed, acc, bw, rt, done_count, total_exps))

            # ETA
            elapsed = time.time() - global_start
            avg = elapsed / done_count
            remaining = (total_exps - done_count) * avg
            eta = str(timedelta(seconds=int(remaining)))
            log("  ETA: ~{}".format(eta))

        except Exception as e:
            log("ERROR in {}: {}".format(key, e))
            log(traceback.format_exc(), also_print=False)
            traceback.print_exc()
            completed[key] = {
                "dataset": ds_name,
                "k_ratio": k_ratio,
                "seed": seed,
                "error": str(e),
                "traceback": traceback.format_exc(),
            }
            save_checkpoint(checkpoint_data)
            _cleanup_gpu()
            continue

    # Finalize
    checkpoint_data["metadata"]["end_time"] = datetime.now().isoformat()
    checkpoint_data["metadata"]["total_elapsed"] = time.time() - global_start
    save_checkpoint(checkpoint_data)

    elapsed_total = time.time() - global_start
    log("\n" + "=" * 66)
    log("  COMPLETED: {}/{}".format(done_count, total_exps))
    log("  Total time: {}".format(timedelta(seconds=int(elapsed_total))))
    log("  Checkpoint: {}".format(OUTPUT_DIR / CHECKPOINT_FILE))
    log("=" * 66)

    # ---- Summary table ----
    log("\n  Top-K Results (Imaging, FedAvg):")
    log("  {:>12s} | {:>12s} | {:>12s} | {:>12s}".format(
        "Dataset", "K ratio", "Acc (%)", "BW Savings"))
    log("  " + "-" * 60)

    for ds_name in ds_names:
        ds_short = IMAGING_DATASETS[ds_name]["short"]
        for k_ratio in k_ratios:
            k_label = K_LABELS[k_ratio]
            accs = []
            bws = []
            for seed in seeds:
                key = "{}_{}_s{}".format(ds_short, k_label, seed)
                entry = completed.get(key)
                if entry and "best_accuracy" in entry:
                    accs.append(entry["best_accuracy"] * 100)
                    bws.append(entry.get("bandwidth_savings_pct", 0))
            if accs:
                log("  {:>12s} | {:>12s} | {:5.1f}+-{:.1f}  | {:>8.0f}%".format(
                    ds_name, k_label, np.mean(accs), np.std(accs), np.mean(bws)))

    # ---- Generate figure ----
    try:
        generate_figure(completed, ds_names, k_ratios, seeds)
    except Exception as e:
        log("Figure generation failed: {}".format(e))
        traceback.print_exc()

    if _log_file:
        _log_file.close()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
FL-EHDS Experiment — Client Scalability + DP x Top-K Combined.

Block A: Client Scalability (Cardiovascular, IID)
  Tests how algorithms scale with increasing number of hospitals.
  Directly relevant to EHDS deployment across 27 EU member states.
  - K clients: 3, 5, 10, 15, 20
  - Algorithms: FedAvg, Ditto, HPFL
  - DP: No-DP, eps=1, eps=10
  - Seeds: 42, 123, 456
  Total: 5 x 3 x 3 x 3 = 135 experiments

Block B: DP + Top-K Combined
  Tests interaction between privacy and communication efficiency.
  Practical EHDS needs both simultaneously.
  - Datasets: CV, PTB-XL
  - Algorithms: FedAvg, Ditto, HPFL
  - Top-K: 100% (baseline), 5%, 1%
  - DP: No-DP, eps=1, eps=10
  - Seed: 42
  Total: 2 x 3 x 3 x 3 = 54 experiments

Grand total: 189 experiments (~1-1.5h on Air M3)

Usage:
    cd fl-ehds-framework
    python -m benchmarks.run_scalability_topk [--quick] [--fresh]

Output:
    benchmarks/paper_results_tabular/checkpoint_scalability_topk.json

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
import gc
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any

import numpy as np

FRAMEWORK_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(FRAMEWORK_DIR))

import torch

from terminal.fl_trainer import FederatedTrainer, _detect_device
from data.ptbxl_loader import load_ptbxl_data
from data.cardiovascular_loader import load_cardiovascular_data

# ======================================================================
# Constants
# ======================================================================

OUTPUT_DIR = FRAMEWORK_DIR / "benchmarks" / "paper_results_tabular"
CHECKPOINT_FILE = "checkpoint_scalability_topk.json"
LOG_FILE = "experiment_scalability_topk.log"

ALGORITHMS = ["FedAvg", "Ditto", "HPFL"]
SEEDS = [42, 123, 456]

PTB_XL_CLASS_NAMES = {0: "NORM", 1: "MI", 2: "STTC", 3: "CD", 4: "HYP"}
CV_CLASS_NAMES = {0: "Healthy", 1: "Disease"}

# Block A: Client scalability
K_CLIENTS = [3, 5, 10, 15, 20]

# Block B: Top-K ratios
TOPK_RATIOS = [1.0, 0.05, 0.01]
TOPK_LABELS = {1.0: "NoTopK", 0.05: "Top5pct", 0.01: "Top1pct"}

# DP levels
DP_LEVELS = [
    {"label": "No-DP", "dp_enabled": False, "dp_epsilon": 0},
    {"label": "DP-eps1", "dp_enabled": True, "dp_epsilon": 1.0},
    {"label": "DP-eps10", "dp_enabled": True, "dp_epsilon": 10.0},
]

# Training configs
CV_TRAINING = dict(
    learning_rate=0.01, batch_size=64, num_rounds=25, local_epochs=3, mu=0.1,
)

PX_TRAINING = dict(
    learning_rate=0.005, batch_size=64, num_rounds=30, local_epochs=3, mu=0.1,
)

# ======================================================================
# Logging
# ======================================================================

_log_file = None
_shutdown = False


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


def _handle_signal(signum, frame):
    global _shutdown
    _shutdown = True
    log("Signal {} received — finishing current experiment then saving...".format(signum))


# ======================================================================
# Checkpoint (atomic write with backup)
# ======================================================================

def save_checkpoint(data):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / CHECKPOINT_FILE
    bak = OUTPUT_DIR / (CHECKPOINT_FILE + ".bak")
    data["metadata"]["last_save"] = datetime.now().isoformat()
    fd, tmp = tempfile.mkstemp(dir=str(OUTPUT_DIR), prefix=".sctk_", suffix=".tmp")
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
    gc.collect()


# ======================================================================
# Shared evaluation helpers
# ======================================================================

def _evaluate_per_client(trainer):
    model = trainer.global_model
    model.eval()
    per_client = {}

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
            X_t = torch.FloatTensor(X).to(trainer.device) if isinstance(X, np.ndarray) else X.to(trainer.device)
            y_t = torch.LongTensor(y).to(trainer.device) if isinstance(y, np.ndarray) else y.to(trainer.device)
            correct = total = 0
            for i in range(0, len(y_t), 64):
                out = model(X_t[i:i + 64])
                correct += (out.argmax(1) == y_t[i:i + 64]).sum().item()
                total += len(y_t[i:i + 64])
            per_client[str(cid)] = correct / total if total > 0 else 0.0

    if is_hpfl:
        for n, p in model.named_parameters():
            if n in trainer._hpfl_classifier_names:
                p.data.copy_(saved_cls[n])
    return per_client


def _collect_predictions(trainer):
    model = trainer.global_model
    model.eval()
    all_preds = []
    all_labels = []

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
            X_t = torch.FloatTensor(X).to(trainer.device) if isinstance(X, np.ndarray) else X.to(trainer.device)
            out = model(X_t)
            preds = out.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(y.tolist() if hasattr(y, "tolist") else list(y))

    if is_hpfl:
        for n, p in model.named_parameters():
            if n in trainer._hpfl_classifier_names:
                p.data.copy_(saved_cls[n])

    return np.array(all_labels), np.array(all_preds)


def _compute_per_class_metrics(y_true, y_pred, num_classes, class_names):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1

    per_class = {}
    for c in range(num_classes):
        cls_name = class_names.get(c, "Class_{}".format(c))
        tp = cm[c, c]
        fn = cm[c, :].sum() - tp
        fp = cm[:, c].sum() - tp
        support = cm[c, :].sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        per_class[cls_name] = {
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1": round(float(f1), 4),
            "support": int(support),
        }

    per_class["macro_avg"] = {
        "precision": round(float(np.mean([v["precision"] for k, v in per_class.items() if k != "macro_avg"])), 4),
        "recall": round(float(np.mean([v["recall"] for k, v in per_class.items() if k != "macro_avg"])), 4),
        "f1": round(float(np.mean([v["f1"] for k, v in per_class.items() if k != "macro_avg"])), 4),
    }

    return per_class, cm.tolist()


def _compute_dei(per_class_metrics, num_classes, class_names):
    recalls = []
    for c in range(num_classes):
        cls_name = class_names.get(c, "Class_{}".format(c))
        if cls_name in per_class_metrics:
            recalls.append(per_class_metrics[cls_name]["recall"])
    if not recalls:
        return 0.0
    recalls = np.array(recalls)
    min_recall = float(np.min(recalls))
    mean_recall = float(np.mean(recalls))
    if mean_recall == 0:
        return 0.0
    cv = float(np.std(recalls) / mean_recall)
    return round(min_recall * (1.0 - cv), 4)


def _compute_jain_index(per_client_acc):
    accs = list(per_client_acc.values())
    if not accs:
        return 0.0
    accs = np.array(accs)
    n = len(accs)
    if n == 0 or np.sum(accs ** 2) == 0:
        return 0.0
    return float((np.sum(accs) ** 2) / (n * np.sum(accs ** 2)))


def format_time(seconds):
    return str(timedelta(seconds=int(seconds)))


# ======================================================================
# Top-K Sparsification
# ======================================================================

def apply_topk_to_model_update(model, old_state, k_ratio, exclude_params=None):
    """
    Apply Top-k sparsification to the model update.
    After train_round(), compute delta = new - old, keep top k_ratio% by magnitude.

    Args:
        exclude_params: set of param names to exclude (e.g. HPFL classifier heads)
    """
    if k_ratio >= 1.0:
        return {"compression_ratio": 1.0, "kept_params": 0, "total_params": 0}

    exclude_params = exclude_params or set()
    total_params = 0
    kept_params = 0

    with torch.no_grad():
        for name, param in model.named_parameters():
            if name not in old_state or name in exclude_params:
                continue

            delta = param.data - old_state[name]
            flat = delta.flatten()
            n = flat.numel()
            k = max(1, int(n * k_ratio))
            total_params += n
            kept_params += k

            _, topk_idx = torch.topk(flat.abs(), k)
            mask = torch.zeros_like(flat)
            mask[topk_idx] = 1.0
            sparsified = flat * mask

            param.data = old_state[name] + sparsified.reshape(param.data.shape)

    compression_ratio = kept_params / total_params if total_params > 0 else 1.0
    return {
        "compression_ratio": compression_ratio,
        "kept_params": kept_params,
        "total_params": total_params,
        "bandwidth_savings_pct": round((1 - compression_ratio) * 100, 1),
    }


# ======================================================================
# Block A: Client Scalability
# ======================================================================

def run_scalability_experiment(algo, num_clients, dp_cfg, seed, quick=False):
    """Run one client-scalability experiment on CV (IID partitioning)."""
    start = time.time()
    tcfg = CV_TRAINING
    num_classes = 2
    input_dim = 11
    num_rounds = 5 if quick else tcfg["num_rounds"]

    client_data, client_test, metadata = load_cardiovascular_data(
        num_clients=num_clients, seed=seed, is_iid=True,
    )

    # Pre-check for empty clients
    for cid in range(num_clients):
        X_c, y_c = client_data[cid]
        if len(y_c) == 0:
            raise ValueError("Client {} has 0 training samples — skipping".format(cid))

    trainer_kwargs = dict(
        num_clients=num_clients, algorithm=algo,
        local_epochs=tcfg["local_epochs"], batch_size=tcfg["batch_size"],
        learning_rate=tcfg["learning_rate"], mu=tcfg["mu"], seed=seed,
        external_data=client_data, external_test_data=client_test,
        input_dim=input_dim, num_classes=num_classes,
    )

    if dp_cfg["dp_enabled"]:
        trainer_kwargs["dp_enabled"] = True
        trainer_kwargs["dp_epsilon"] = dp_cfg["dp_epsilon"]
        trainer_kwargs["dp_clip_norm"] = 1.0

    trainer = FederatedTrainer(**trainer_kwargs)

    history = []
    best_acc = 0.0

    for r in range(num_rounds):
        result = trainer.train_round(r)
        history.append({
            "round": r + 1,
            "accuracy": result.global_acc,
            "loss": result.global_loss,
            "f1": result.global_f1,
        })
        if result.global_acc > best_acc:
            best_acc = result.global_acc

    per_client_acc = _evaluate_per_client(trainer)
    jain = _compute_jain_index(per_client_acc)
    final_acc = float(np.mean(list(per_client_acc.values())))

    y_true, y_pred = _collect_predictions(trainer)
    per_class, cm = _compute_per_class_metrics(y_true, y_pred, num_classes, CV_CLASS_NAMES)
    dei = _compute_dei(per_class, num_classes, CV_CLASS_NAMES)

    elapsed = time.time() - start

    return {
        "block": "A_scalability",
        "dataset": "Cardiovascular",
        "algorithm": algo,
        "num_clients": num_clients,
        "dp_level": dp_cfg["label"],
        "dp_epsilon": dp_cfg["dp_epsilon"],
        "seed": seed,
        "final_accuracy": round(final_acc, 4),
        "best_accuracy": round(best_acc, 4),
        "per_client_accuracy": per_client_acc,
        "jain_index": round(jain, 4),
        "dei": dei,
        "per_class_metrics": per_class,
        "confusion_matrix": cm,
        "time_seconds": round(elapsed, 1),
    }


# ======================================================================
# Block B: DP + Top-K Combined
# ======================================================================

def run_dp_topk_experiment(dataset, algo, topk_ratio, dp_cfg, seed, quick=False):
    """Run one DP + Top-K combined experiment."""
    start = time.time()

    if dataset == "Cardiovascular":
        num_clients = 5
        tcfg = CV_TRAINING
        num_classes = 2
        input_dim = 11
        class_names = CV_CLASS_NAMES
        num_rounds = 5 if quick else tcfg["num_rounds"]

        client_data, client_test, metadata = load_cardiovascular_data(
            num_clients=num_clients, seed=seed, is_iid=True,
        )
    else:  # PTB_XL
        num_clients = 5
        tcfg = PX_TRAINING
        num_classes = 5
        input_dim = 9
        class_names = PTB_XL_CLASS_NAMES
        num_rounds = 5 if quick else tcfg["num_rounds"]

        client_data, client_test, metadata = load_ptbxl_data(
            num_clients=num_clients, seed=seed,
            partition_by_site=True, min_site_samples=50,
        )

    trainer_kwargs = dict(
        num_clients=num_clients, algorithm=algo,
        local_epochs=tcfg["local_epochs"], batch_size=tcfg["batch_size"],
        learning_rate=tcfg["learning_rate"], mu=tcfg["mu"], seed=seed,
        external_data=client_data, external_test_data=client_test,
        input_dim=input_dim, num_classes=num_classes,
    )

    if dp_cfg["dp_enabled"]:
        trainer_kwargs["dp_enabled"] = True
        trainer_kwargs["dp_epsilon"] = dp_cfg["dp_epsilon"]
        trainer_kwargs["dp_clip_norm"] = 1.0

    trainer = FederatedTrainer(**trainer_kwargs)

    # Identify HPFL classifier params (exclude from Top-K)
    exclude_params = set()
    if algo == "HPFL" and hasattr(trainer, "_hpfl_classifier_names"):
        exclude_params = set(trainer._hpfl_classifier_names)

    history = []
    best_acc = 0.0
    compression_stats = []

    for r in range(num_rounds):
        # Save state before round
        old_state = {n: p.data.clone() for n, p in trainer.global_model.named_parameters()}

        result = trainer.train_round(r)

        # Apply Top-K sparsification to backbone update
        stats = apply_topk_to_model_update(
            trainer.global_model, old_state, topk_ratio, exclude_params)
        compression_stats.append(stats)

        # Re-evaluate after sparsification if applied
        if topk_ratio < 1.0:
            per_client_acc = _evaluate_per_client(trainer)
            acc_after = float(np.mean(list(per_client_acc.values())))
        else:
            acc_after = result.global_acc

        history.append({
            "round": r + 1,
            "accuracy": acc_after,
            "accuracy_before_topk": result.global_acc,
            "loss": result.global_loss,
        })
        if acc_after > best_acc:
            best_acc = acc_after

    # Final evaluation
    per_client_acc = _evaluate_per_client(trainer)
    jain = _compute_jain_index(per_client_acc)
    final_acc = float(np.mean(list(per_client_acc.values())))

    y_true, y_pred = _collect_predictions(trainer)
    per_class, cm = _compute_per_class_metrics(y_true, y_pred, num_classes, class_names)
    dei = _compute_dei(per_class, num_classes, class_names)

    avg_compression = np.mean([s["compression_ratio"] for s in compression_stats])
    elapsed = time.time() - start

    return {
        "block": "B_dp_topk",
        "dataset": dataset,
        "algorithm": algo,
        "topk_ratio": topk_ratio,
        "topk_label": TOPK_LABELS[topk_ratio],
        "dp_level": dp_cfg["label"],
        "dp_epsilon": dp_cfg["dp_epsilon"],
        "seed": seed,
        "final_accuracy": round(final_acc, 4),
        "best_accuracy": round(best_acc, 4),
        "per_client_accuracy": per_client_acc,
        "jain_index": round(jain, 4),
        "dei": dei,
        "per_class_metrics": per_class,
        "confusion_matrix": cm,
        "avg_compression_ratio": round(avg_compression, 4),
        "bandwidth_savings_pct": round((1 - avg_compression) * 100, 1),
        "time_seconds": round(elapsed, 1),
    }


# ======================================================================
# Main
# ======================================================================

def main():
    global _log_file

    parser = argparse.ArgumentParser(
        description="FL-EHDS: Client Scalability + DP x Top-K Combined")
    parser.add_argument("--quick", action="store_true",
                        help="Quick validation (5 rounds)")
    parser.add_argument("--fresh", action="store_true",
                        help="Delete checkpoint and start fresh")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _log_file = open(str(OUTPUT_DIR / LOG_FILE), "a")

    log("=" * 70)
    log("FL-EHDS: Client Scalability + DP x Top-K Combined")
    log("Mode: {}".format("QUICK" if args.quick else "FULL"))
    log("Device: {}".format(_detect_device("cpu")))
    log("=" * 70)

    if args.fresh:
        for suffix in ["", ".bak"]:
            p = OUTPUT_DIR / (CHECKPOINT_FILE + suffix)
            if p.exists():
                p.unlink()
                log("Deleted {}".format(CHECKPOINT_FILE + suffix))

    ckpt = load_checkpoint()
    if ckpt is None:
        ckpt = {
            "metadata": {
                "experiment": "Client Scalability + DP x Top-K",
                "started": datetime.now().isoformat(),
            },
            "results": {},
            "completed": [],
        }

    # ---- Build experiment list ----
    experiments = []

    # Block A: Client Scalability (CV, IID)
    for algo in ALGORITHMS:
        for k in K_CLIENTS:
            for dp_cfg in DP_LEVELS:
                for seed in SEEDS:
                    exp_key = "A_{}_K{}_{}_s{}".format(algo, k, dp_cfg["label"], seed)
                    experiments.append(("A", exp_key, algo, k, dp_cfg, seed))

    # Block B: DP + Top-K (CV + PTB-XL, seed=42 only)
    for ds in ["Cardiovascular", "PTB_XL"]:
        ds_short = "CV" if ds == "Cardiovascular" else "PX"
        for algo in ALGORITHMS:
            for topk in TOPK_RATIOS:
                for dp_cfg in DP_LEVELS:
                    exp_key = "B_{}_{}_{}_{}_s42".format(
                        ds_short, algo, TOPK_LABELS[topk], dp_cfg["label"])
                    experiments.append(("B", exp_key, ds, algo, topk, dp_cfg))

    total = len(experiments)
    done = len(ckpt["completed"])
    log("Total: {} experiments ({} Block A + {} Block B), {} already done".format(
        total,
        len([e for e in experiments if e[0] == "A"]),
        len([e for e in experiments if e[0] == "B"]),
        done))

    cascade_start = time.time()

    for idx, exp in enumerate(experiments, 1):
        if _shutdown:
            log("Shutdown requested — saving checkpoint")
            break

        block = exp[0]
        exp_key = exp[1]

        if exp_key in ckpt["completed"]:
            continue

        try:
            if block == "A":
                _, _, algo, k, dp_cfg, seed = exp
                log("[{}/{}] SCALE | {} | K={} | {} | seed={} ...".format(
                    idx, total, algo, k, dp_cfg["label"], seed))
                result = run_scalability_experiment(
                    algo, k, dp_cfg, seed, quick=args.quick)
            else:
                _, _, ds, algo, topk, dp_cfg = exp
                log("[{}/{}] DP+TopK | {} | {} | {} | {} ...".format(
                    idx, total, ds, algo, TOPK_LABELS[topk], dp_cfg["label"]))
                result = run_dp_topk_experiment(
                    ds, algo, topk, dp_cfg, seed=42, quick=args.quick)

            ckpt["results"][exp_key] = result
            ckpt["completed"].append(exp_key)
            save_checkpoint(ckpt)

            log("  -> acc={:.1f}% | Jain={:.3f} | DEI={:.3f} | {:.0f}s".format(
                result["final_accuracy"] * 100,
                result["jain_index"],
                result["dei"],
                result["time_seconds"]))

            _cleanup_gpu()

        except Exception as e:
            log("  ERROR: {}".format(e))
            log("  " + traceback.format_exc().split("\n")[-2])
            _cleanup_gpu()

    total_time = time.time() - cascade_start

    ckpt["metadata"]["total_time"] = format_time(total_time)
    ckpt["metadata"]["completed_count"] = len(ckpt["completed"])
    ckpt["metadata"]["total_count"] = total
    save_checkpoint(ckpt)

    log("=" * 70)
    log("COMPLETE: {}/{} experiments in {}".format(
        len(ckpt["completed"]), total, format_time(total_time)))
    log("Checkpoint: {}".format(CHECKPOINT_FILE))
    log("=" * 70)

    # ---- Summary tables ----

    # Block A summary
    log("")
    log("BLOCK A — Client Scalability (CV, IID):")
    log("{:<8} {:<6} {:<10} {:<8} {:<8} {:<8}".format(
        "Algo", "K", "DP", "Acc%", "Jain", "DEI"))
    log("-" * 55)
    for algo in ALGORITHMS:
        for k in K_CLIENTS:
            for dp_cfg in DP_LEVELS:
                keys = ["A_{}_K{}_{}_s{}".format(algo, k, dp_cfg["label"], s) for s in SEEDS]
                accs = [ckpt["results"][k_]["final_accuracy"]
                        for k_ in keys if k_ in ckpt["results"]]
                jains = [ckpt["results"][k_]["jain_index"]
                         for k_ in keys if k_ in ckpt["results"]]
                deis = [ckpt["results"][k_]["dei"]
                        for k_ in keys if k_ in ckpt["results"]]
                if accs:
                    log("{:<8} {:<6} {:<10} {:<8.1f} {:<8.3f} {:<8.3f}".format(
                        algo, k, dp_cfg["label"],
                        np.mean(accs) * 100, np.mean(jains), np.mean(deis)))

    # Block B summary
    log("")
    log("BLOCK B — DP + Top-K Combined:")
    log("{:<6} {:<8} {:<10} {:<10} {:<8} {:<8}".format(
        "DS", "Algo", "TopK", "DP", "Acc%", "DEI"))
    log("-" * 60)
    for ds_short, ds_full in [("CV", "Cardiovascular"), ("PX", "PTB_XL")]:
        for algo in ALGORITHMS:
            for topk in TOPK_RATIOS:
                for dp_cfg in DP_LEVELS:
                    k_ = "B_{}_{}_{}_{}_s42".format(
                        ds_short, algo, TOPK_LABELS[topk], dp_cfg["label"])
                    if k_ in ckpt["results"]:
                        r = ckpt["results"][k_]
                        log("{:<6} {:<8} {:<10} {:<10} {:<8.1f} {:<8.3f}".format(
                            ds_short, algo, TOPK_LABELS[topk], dp_cfg["label"],
                            r["final_accuracy"] * 100, r["dei"]))

    if _log_file:
        _log_file.close()


if __name__ == "__main__":
    main()

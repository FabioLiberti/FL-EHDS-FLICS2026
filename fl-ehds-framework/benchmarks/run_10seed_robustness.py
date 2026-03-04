#!/usr/bin/env python3
"""
10-Seed Robustness Benchmark for FL-EHDS Paper.

Runs core FL algorithms across all three tabular datasets with 10 seeds
for statistical significance (confidence intervals, p-values).

Grid (full):
  4 algorithms x 3 datasets x 2 IID modes x 10 seeds = 240 experiments

Grid (quick):
  1 algorithm  x 1 dataset  x 1 IID mode  x  2 seeds =   2 experiments

Core algorithms: FedAvg, FedProx, Ditto, HPFL
Datasets:
  - Cardiovascular  (input_dim=11,  num_classes=2)
  - PTB-XL          (input_dim=9,   num_classes=5)
  - CDC_Diabetes    (input_dim=21,  num_classes=2)

Seeds: [42, 123, 456, 789, 999, 1111, 2222, 3333, 4444, 5555]

Training: num_clients=5, num_rounds=25, local_epochs=3, batch_size=32,
          learning_rate=0.01, early_stopping(patience=6, min_rounds=12)

No DP in this benchmark.

Usage:
    cd fl-ehds-framework
    python -m benchmarks.run_10seed_robustness [--quick] [--dry-run] [--fresh]

Estimated time: ~3 hours on MacBook Air M3.

Author: Fabio Liberti
"""

import argparse
import gc
import json
import os
import signal
import shutil
import sys
import tempfile
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch

FRAMEWORK_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(FRAMEWORK_DIR))

OUTPUT_DIR = Path(__file__).parent / "paper_results_tabular"
CHECKPOINT_FILE = "checkpoint_10seed_robustness.json"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

_shutdown = False


def _signal_handler(signum, frame):
    global _shutdown
    _shutdown = True
    print("\nGraceful shutdown requested...")


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


# ======================================================================
# Logging
# ======================================================================

def log(msg):
    ts = time.strftime("%H:%M:%S")
    print("[{}] {}".format(ts, msg), flush=True)


# ======================================================================
# Checkpoint (atomic save with .bak)
# ======================================================================

def save_checkpoint(data):
    path = OUTPUT_DIR / CHECKPOINT_FILE
    bak = OUTPUT_DIR / (CHECKPOINT_FILE + ".bak")
    fd, tmp = tempfile.mkstemp(dir=str(OUTPUT_DIR), prefix=".ckpt_rob_", suffix=".tmp")
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
    for p in [OUTPUT_DIR / CHECKPOINT_FILE, OUTPUT_DIR / (CHECKPOINT_FILE + ".bak")]:
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
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


# ======================================================================
# Configuration
# ======================================================================

CORE_ALGORITHMS = ["FedAvg", "FedProx", "Ditto", "HPFL"]

SEEDS = [42, 123, 456, 789, 999, 1111, 2222, 3333, 4444, 5555]

IID_MODES = [True, False]

TRAINING_CONFIG = dict(
    num_clients=5,
    num_rounds=25,
    local_epochs=3,
    batch_size=32,
    learning_rate=0.01,
)

EARLY_STOPPING = dict(
    patience=6,
    min_delta=0.003,
    min_rounds=12,
)

DATASET_CONFIGS = {
    "Cardiovascular": {
        "input_dim": 11,
        "num_classes": 2,
    },
    "PTB_XL": {
        "input_dim": 9,
        "num_classes": 5,
    },
    "CDC_Diabetes": {
        "input_dim": 21,
        "num_classes": 2,
    },
}


# ======================================================================
# Data loading
# ======================================================================

def load_data(dataset, num_clients, is_iid, seed, alpha=0.5):
    """Load dataset by name, returning (client_data, client_test, metadata)."""
    if dataset == "Cardiovascular":
        from data.cardiovascular_loader import load_cardiovascular_data
        return load_cardiovascular_data(
            num_clients=num_clients, is_iid=is_iid, seed=seed, alpha=alpha)
    elif dataset == "PTB_XL":
        from data.ptbxl_loader import load_ptbxl_data
        return load_ptbxl_data(num_clients=num_clients, seed=seed)
    elif dataset == "CDC_Diabetes":
        from data.cdc_diabetes_loader import load_cdc_diabetes_data
        return load_cdc_diabetes_data(
            num_clients=num_clients, is_iid=is_iid, seed=seed, alpha=alpha)
    else:
        raise ValueError("Unknown dataset: {}".format(dataset))


# ======================================================================
# Single experiment
# ======================================================================

def run_single_experiment(algorithm, dataset, is_iid, seed):
    """Run one FL experiment and return result dict."""
    from terminal.training.federated import FederatedTrainer

    ds_cfg = DATASET_CONFIGS[dataset]
    num_clients = TRAINING_CONFIG["num_clients"]
    num_rounds = TRAINING_CONFIG["num_rounds"]

    client_data, client_test, meta = load_data(
        dataset=dataset,
        num_clients=num_clients,
        is_iid=is_iid,
        seed=seed,
        alpha=0.5,
    )

    trainer = FederatedTrainer(
        num_clients=num_clients,
        algorithm=algorithm,
        local_epochs=TRAINING_CONFIG["local_epochs"],
        batch_size=TRAINING_CONFIG["batch_size"],
        learning_rate=TRAINING_CONFIG["learning_rate"],
        input_dim=ds_cfg["input_dim"],
        num_classes=ds_cfg["num_classes"],
        external_data=client_data,
        external_test_data=client_test,
        dp_enabled=False,
    )

    # --- Training with early stopping ---
    best_acc = 0.0
    best_round = 0
    patience = EARLY_STOPPING["patience"]
    min_delta = EARLY_STOPPING["min_delta"]
    min_rounds = EARLY_STOPPING["min_rounds"]
    no_improve = 0
    history = []

    for r in range(num_rounds):
        if _shutdown:
            break

        rr = trainer.train_round(r)
        metrics = {
            "round": r + 1,
            "accuracy": rr.global_acc,
            "loss": rr.global_loss,
            "f1": rr.global_f1,
            "precision": rr.global_precision,
            "recall": rr.global_recall,
            "auc": rr.global_auc,
        }
        history.append(metrics)

        if rr.global_acc > best_acc + min_delta:
            best_acc = rr.global_acc
            best_round = r + 1
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience and r >= min_rounds:
            break

    # --- Per-client evaluation ---
    per_client_acc = {}
    all_preds, all_labels = [], []

    for cid in client_test:
        Xc, yc = client_test[cid]

        if hasattr(trainer, "personal_models") and trainer.personal_models:
            model = trainer.personal_models.get(cid, trainer.global_model)
        else:
            model = trainer.global_model

        model_device = next(model.parameters()).device
        X_t = torch.FloatTensor(Xc).to(model_device)

        model.eval()
        with torch.no_grad():
            preds_c = model(X_t).argmax(dim=1).cpu().numpy()
        acc_c = float((preds_c == yc).mean())
        per_client_acc[str(cid)] = round(acc_c, 4)
        all_preds.append(preds_c)
        all_labels.append(yc)

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    accuracy = float((all_preds == all_labels).mean())

    # --- F1 (macro for multi-class, binary otherwise) ---
    num_classes = ds_cfg["num_classes"]
    if num_classes == 2:
        tp = int(((all_preds == 1) & (all_labels == 1)).sum())
        fp = int(((all_preds == 1) & (all_labels == 0)).sum())
        fn = int(((all_preds == 0) & (all_labels == 1)).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-10)
    else:
        # Macro F1 for multi-class
        f1s, precs, recs = [], [], []
        for c in range(num_classes):
            tp_c = int(((all_preds == c) & (all_labels == c)).sum())
            fp_c = int(((all_preds == c) & (all_labels != c)).sum())
            fn_c = int(((all_preds != c) & (all_labels == c)).sum())
            p_c = tp_c / max(tp_c + fp_c, 1)
            r_c = tp_c / max(tp_c + fn_c, 1)
            f_c = 2 * p_c * r_c / max(p_c + r_c, 1e-10)
            precs.append(p_c)
            recs.append(r_c)
            f1s.append(f_c)
        precision = float(np.mean(precs))
        recall = float(np.mean(recs))
        f1 = float(np.mean(f1s))

    # --- Fairness (Jain index) ---
    accs = list(per_client_acc.values())
    jain = float(sum(accs) ** 2 / (len(accs) * sum(a ** 2 for a in accs))) if accs else 0

    samples_per_client = {str(cid): len(client_data[cid][1]) for cid in client_data}

    return {
        "dataset": dataset,
        "algorithm": algorithm,
        "num_clients": num_clients,
        "seed": seed,
        "is_iid": is_iid,
        "num_rounds": len(history),
        "total_train_samples": sum(len(client_data[c][1]) for c in client_data),
        "accuracy": round(accuracy, 4),
        "f1": round(f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "per_client_accuracy": per_client_acc,
        "fairness": {
            "mean": round(np.mean(accs), 4),
            "std": round(np.std(accs), 4),
            "min": round(min(accs), 4),
            "max": round(max(accs), 4),
            "jain_index": round(jain, 4),
        },
        "best_metrics": {"accuracy": round(best_acc, 4), "round": best_round},
        "samples_per_client": samples_per_client,
        "history": history,
        "final_metrics": history[-1] if history else {},
    }


# ======================================================================
# Statistical summary
# ======================================================================

def compute_statistics(checkpoint):
    """Compute per-(algo, dataset, iid) statistics across seeds."""
    from collections import defaultdict

    groups = defaultdict(list)
    for key, result in checkpoint.get("completed", {}).items():
        if "error" in result:
            continue
        group_key = (result["algorithm"], result["dataset"], result["is_iid"])
        groups[group_key].append(result)

    stats = {}
    for (algo, dataset, is_iid), results in sorted(groups.items()):
        accuracies = [r["accuracy"] for r in results]
        f1_scores = [r["f1"] for r in results]
        jain_values = [r["fairness"]["jain_index"] for r in results]

        n = len(accuracies)
        iid_tag = "IID" if is_iid else "NonIID"
        stat_key = "{}_{}_{}".format(algo, dataset, iid_tag)

        # 95% CI using t-distribution approximation (t ~ 2.262 for df=9)
        t_val = 2.262 if n == 10 else 1.96  # exact t for n=10 (df=9)
        acc_mean = float(np.mean(accuracies))
        acc_std = float(np.std(accuracies, ddof=1)) if n > 1 else 0.0
        acc_se = acc_std / np.sqrt(n) if n > 1 else 0.0
        acc_ci = (
            round(acc_mean - t_val * acc_se, 4),
            round(acc_mean + t_val * acc_se, 4),
        )

        f1_mean = float(np.mean(f1_scores))
        f1_std = float(np.std(f1_scores, ddof=1)) if n > 1 else 0.0
        f1_se = f1_std / np.sqrt(n) if n > 1 else 0.0
        f1_ci = (
            round(f1_mean - t_val * f1_se, 4),
            round(f1_mean + t_val * f1_se, 4),
        )

        stats[stat_key] = {
            "algorithm": algo,
            "dataset": dataset,
            "is_iid": is_iid,
            "n_seeds": n,
            "accuracy": {
                "mean": round(acc_mean, 4),
                "std": round(acc_std, 4),
                "ci_95": acc_ci,
                "min": round(float(min(accuracies)), 4),
                "max": round(float(max(accuracies)), 4),
            },
            "f1": {
                "mean": round(f1_mean, 4),
                "std": round(f1_std, 4),
                "ci_95": f1_ci,
            },
            "jain_index": {
                "mean": round(float(np.mean(jain_values)), 4),
                "std": round(float(np.std(jain_values, ddof=1)) if n > 1 else 0.0, 4),
            },
        }

    return stats


def print_summary(checkpoint):
    """Print a formatted summary table with CIs."""
    stats = compute_statistics(checkpoint)
    if not stats:
        log("No completed experiments to summarize.")
        return

    log("")
    log("=" * 90)
    log("  STATISTICAL SUMMARY (10-Seed Robustness)")
    log("=" * 90)
    log("  {:<8} {:<16} {:<7} {:>3} | {:>6} +/- {:>5}  [{:>6}, {:>6}] | {:>6} +/- {:>5}".format(
        "Algo", "Dataset", "IID", "N",
        "Acc", "Std", "CI_lo", "CI_hi",
        "F1", "Std"))
    log("  " + "-" * 86)

    for key in sorted(stats.keys()):
        s = stats[key]
        iid_tag = "IID" if s["is_iid"] else "NonIID"
        log("  {:<8} {:<16} {:<7} {:>3} | {:>5.1f}% +/- {:>4.1f}%  [{:>5.1f}%, {:>5.1f}%] | {:>.4f} +/- {:>.4f}".format(
            s["algorithm"], s["dataset"], iid_tag, s["n_seeds"],
            s["accuracy"]["mean"] * 100,
            s["accuracy"]["std"] * 100,
            s["accuracy"]["ci_95"][0] * 100,
            s["accuracy"]["ci_95"][1] * 100,
            s["f1"]["mean"],
            s["f1"]["std"],
        ))

    log("=" * 90)


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="10-Seed Robustness Benchmark for FL-EHDS Paper")
    parser.add_argument("--quick", action="store_true",
                        help="Quick validation (1 algo x 1 dataset x 1 IID x 2 seeds = 2 experiments)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show experiment plan without executing")
    parser.add_argument("--fresh", action="store_true",
                        help="Discard existing checkpoint and start fresh")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Build experiment list ---
    if args.quick:
        algos = ["FedAvg"]
        datasets = ["Cardiovascular"]
        iid_modes = [True]
        seeds = [42, 123]
    else:
        algos = CORE_ALGORITHMS
        datasets = list(DATASET_CONFIGS.keys())
        iid_modes = IID_MODES
        seeds = SEEDS

    experiments = []
    for algo in algos:
        for dataset in datasets:
            for is_iid in iid_modes:
                for seed in seeds:
                    iid_tag = "IID" if is_iid else "NonIID"
                    key = "{}_{}_{}_{}_s{}".format(algo, dataset, iid_tag,
                                                   TRAINING_CONFIG["num_clients"], seed)
                    experiments.append((key, algo, dataset, is_iid, seed))

    total = len(experiments)

    # --- Header ---
    log("=" * 70)
    log("  10-Seed Robustness Benchmark")
    log("=" * 70)
    log("  Device:      {}".format(DEVICE))
    log("  Algorithms:  {} ({})".format(len(algos), ", ".join(algos)))
    log("  Datasets:    {} ({})".format(len(datasets), ", ".join(datasets)))
    log("  IID modes:   {}".format(iid_modes))
    log("  Seeds:       {} seeds {}".format(len(seeds), seeds))
    log("  Grid:        {} algos x {} datasets x {} IID x {} seeds = {} experiments".format(
        len(algos), len(datasets), len(iid_modes), len(seeds), total))
    log("  Training:    rounds={}, epochs={}, bs={}, lr={}".format(
        TRAINING_CONFIG["num_rounds"], TRAINING_CONFIG["local_epochs"],
        TRAINING_CONFIG["batch_size"], TRAINING_CONFIG["learning_rate"]))
    log("  Early stop:  patience={}, min_rounds={}".format(
        EARLY_STOPPING["patience"], EARLY_STOPPING["min_rounds"]))
    log("  Checkpoint:  {}".format(OUTPUT_DIR / CHECKPOINT_FILE))
    log("=" * 70)

    # --- Dry run ---
    if args.dry_run:
        for key, algo, dataset, is_iid, seed in experiments:
            iid_tag = "IID" if is_iid else "NonIID"
            log("  [DRY-RUN] {} | {} | {} | seed={}".format(algo, dataset, iid_tag, seed))
        log("")
        log("DRY-RUN complete: {} experiments planned.".format(total))
        return

    # --- Load or create checkpoint ---
    checkpoint = None if args.fresh else load_checkpoint()
    if checkpoint is None:
        checkpoint = {
            "completed": {},
            "metadata": {
                "experiment": "10seed_robustness",
                "total_experiments": total,
                "algorithms": algos,
                "datasets": datasets,
                "seeds": seeds,
                "iid_modes": iid_modes,
                "training_config": TRAINING_CONFIG,
                "early_stopping": EARLY_STOPPING,
                "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
        }
    else:
        done = len(checkpoint.get("completed", {}))
        log("AUTO-RESUMED from checkpoint: {}/{} completed".format(done, total))

    # --- Run experiments ---
    t0 = time.time()
    completed = 0
    skipped = 0

    for idx, (key, algo, dataset, is_iid, seed) in enumerate(experiments, 1):
        if _shutdown:
            log("Shutdown requested. Saving checkpoint...")
            save_checkpoint(checkpoint)
            break

        if key in checkpoint["completed"]:
            skipped += 1
            continue

        completed += 1
        iid_tag = "IID" if is_iid else "NonIID"
        log("  [{}/{}] {} | {} | {} | seed={}".format(
            skipped + completed, total, algo, dataset, iid_tag, seed))

        try:
            t_exp = time.time()
            result = run_single_experiment(
                algorithm=algo,
                dataset=dataset,
                is_iid=is_iid,
                seed=seed,
            )
            exp_time = time.time() - t_exp
            result["runtime_seconds"] = round(exp_time, 1)
            checkpoint["completed"][key] = result
            save_checkpoint(checkpoint)
            _cleanup_gpu()

            log("    Acc={:.1f}% | F1={:.4f} | Jain={:.3f} | {:.1f}s".format(
                result["accuracy"] * 100, result["f1"],
                result["fairness"]["jain_index"], exp_time))

        except Exception as e:
            log("    ERROR: {}".format(e))
            traceback.print_exc()
            checkpoint["completed"][key] = {
                "error": str(e),
                "algorithm": algo,
                "dataset": dataset,
                "is_iid": is_iid,
                "seed": seed,
            }
            save_checkpoint(checkpoint)
            _cleanup_gpu()

        # Progress estimate every 20 experiments
        if completed % 20 == 0 and completed > 0:
            elapsed = time.time() - t0
            avg = elapsed / completed
            remaining = (total - skipped - completed) * avg
            log("  PROGRESS: {}/{} done | {:.0f} min elapsed | ~{:.0f} min remaining".format(
                skipped + completed, total,
                elapsed / 60, remaining / 60))

    # --- Finalize ---
    elapsed = time.time() - t0
    checkpoint["metadata"]["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    checkpoint["metadata"]["total_time_seconds"] = round(elapsed, 1)

    # Compute and store statistics
    stats = compute_statistics(checkpoint)
    checkpoint["statistics"] = stats
    save_checkpoint(checkpoint)

    log("")
    log("=" * 70)
    log("COMPLETED: {} new / {} skipped / {} total in {:.0f} min".format(
        completed, skipped, total, elapsed / 60))
    log("Checkpoint: {}".format(OUTPUT_DIR / CHECKPOINT_FILE))
    log("=" * 70)

    print_summary(checkpoint)


if __name__ == "__main__":
    main()

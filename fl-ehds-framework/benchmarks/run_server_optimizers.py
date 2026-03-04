#!/usr/bin/env python3
"""
Server-Side Optimizer Benchmark for FL-EHDS Paper.

Tests advanced server-side optimizers across 3 tabular datasets:
  6 algorithms × 3 datasets × 2 IID modes × 3 seeds = 108 experiments

Algorithms: FedAdam, FedYogi, FedAdagrad, SCAFFOLD, FedNova, FedDyn
Datasets:
  - Cardiovascular  (70,000 samples, 11 features, binary classification)
  - PTB-XL          (21,837 samples, 9 features, 5-class classification)
  - CDC_Diabetes    (253,680 samples, 21 features, binary classification)

Server optimizer hyperparameters: server_lr=0.1, beta1=0.9, beta2=0.99, tau=1e-3

Key findings expected:
  - Adaptive server LR vs fixed-LR algorithms (FedAdam/FedYogi vs SCAFFOLD)
  - Convergence speed comparison across dataset scales
  - Non-IID robustness of server-side optimizers
  - Fairness (Jain index) under heterogeneous data distributions

Usage:
    cd fl-ehds-framework
    python -m benchmarks.run_server_optimizers [--quick] [--dry-run] [--fresh]

Estimated time: ~2.5 hours on MacBook Air M3.

Author: Fabio Liberti
"""

import argparse
import json
import os
import signal
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
import torch

FRAMEWORK_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(FRAMEWORK_DIR))

OUTPUT_DIR = Path(__file__).parent / "paper_results_tabular"
CHECKPOINT_FILE = "checkpoint_server_optimizers.json"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

_shutdown = False


def _signal_handler(signum, frame):
    global _shutdown
    _shutdown = True
    print("\nGraceful shutdown requested...")


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


def log(msg):
    ts = time.strftime("%H:%M:%S")
    print("[{}] {}".format(ts, msg))


def save_checkpoint(data):
    path = OUTPUT_DIR / CHECKPOINT_FILE
    bak = OUTPUT_DIR / (CHECKPOINT_FILE + ".bak")
    fd, tmp = tempfile.mkstemp(dir=str(OUTPUT_DIR), prefix=".ckpt_", suffix=".tmp")
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


def _cleanup_gpu():
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


# ======================================================================
# Algorithms & Datasets
# ======================================================================

ALL_ALGORITHMS = ["FedAdam", "FedYogi", "FedAdagrad", "SCAFFOLD", "FedNova", "FedDyn"]

DATASETS = {
    "Cardiovascular": {"input_dim": 11, "num_classes": 2},
    "PTB-XL":         {"input_dim": 9,  "num_classes": 5},
    "CDC_Diabetes":   {"input_dim": 21, "num_classes": 2},
}

TRAINING_CONFIG = dict(
    learning_rate=0.01,
    batch_size=32,
    num_rounds=25,
    local_epochs=3,
    num_clients=5,
    server_lr=0.1,
    beta1=0.9,
    beta2=0.99,
    tau=1e-3,
)


def load_data(dataset, num_clients=5, seed=42, is_iid=False, alpha=0.5):
    """Load dataset by name."""
    if dataset == "Cardiovascular":
        from data.cardiovascular_loader import load_cardiovascular_data
        return load_cardiovascular_data(
            num_clients=num_clients, is_iid=is_iid, seed=seed, alpha=alpha)
    elif dataset == "PTB-XL":
        from data.ptbxl_loader import load_ptbxl_data
        return load_ptbxl_data(num_clients=num_clients, seed=seed)
    elif dataset == "CDC_Diabetes":
        from data.cdc_diabetes_loader import load_cdc_diabetes_data
        return load_cdc_diabetes_data(
            num_clients=num_clients, is_iid=is_iid, seed=seed, alpha=alpha)
    else:
        raise ValueError("Unknown dataset: {}".format(dataset))


def run_single_experiment(algorithm, dataset, seed, is_iid, num_rounds=25):
    """Run one FL experiment with a server-side optimizer."""
    from terminal.training.federated import FederatedTrainer

    num_clients = TRAINING_CONFIG["num_clients"]
    ds_cfg = DATASETS[dataset]

    client_data, client_test, meta = load_data(
        dataset, num_clients=num_clients, seed=seed, is_iid=is_iid, alpha=0.5)

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
        server_lr=TRAINING_CONFIG["server_lr"],
        beta1=TRAINING_CONFIG["beta1"],
        beta2=TRAINING_CONFIG["beta2"],
        tau=TRAINING_CONFIG["tau"],
    )

    # Early stopping
    best_acc = 0.0
    best_round = 0
    patience = 6
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

        if rr.global_acc > best_acc + 0.003:
            best_acc = rr.global_acc
            best_round = r + 1
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience and r >= 12:
            break

    # Per-client accuracy
    per_client_acc = {}
    all_preds, all_labels = [], []

    for cid in client_test:
        Xc, yc = client_test[cid]

        if hasattr(trainer, 'personal_models') and trainer.personal_models:
            model = trainer.personal_models.get(cid, trainer.global_model)
        else:
            model = trainer.global_model

        # Match tensor device to model device
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

    # F1
    num_classes = ds_cfg["num_classes"]
    if num_classes == 2:
        # Binary F1
        tp = int(((all_preds == 1) & (all_labels == 1)).sum())
        fp = int(((all_preds == 1) & (all_labels == 0)).sum())
        fn = int(((all_preds == 0) & (all_labels == 1)).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-10)
    else:
        # Macro F1 for multi-class
        f1s = []
        precs = []
        recs = []
        for c in range(num_classes):
            tp = int(((all_preds == c) & (all_labels == c)).sum())
            fp = int(((all_preds == c) & (all_labels != c)).sum())
            fn = int(((all_preds != c) & (all_labels == c)).sum())
            p = tp / max(tp + fp, 1)
            r = tp / max(tp + fn, 1)
            f = 2 * p * r / max(p + r, 1e-10)
            precs.append(p)
            recs.append(r)
            f1s.append(f)
        precision = float(np.mean(precs))
        recall = float(np.mean(recs))
        f1 = float(np.mean(f1s))

    # Fairness
    accs = list(per_client_acc.values())
    jain = float(sum(accs) ** 2 / (len(accs) * sum(a ** 2 for a in accs))) if accs else 0

    samples_per_client = {str(cid): len(client_data[cid][1]) for cid in client_data}

    return {
        "dataset": dataset,
        "algorithm": algorithm,
        "num_clients": num_clients,
        "seed": seed,
        "is_iid": is_iid,
        "alpha": 0.5,
        "dp_epsilon": None,
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
        "server_optimizer_params": {
            "server_lr": TRAINING_CONFIG["server_lr"],
            "beta1": TRAINING_CONFIG["beta1"],
            "beta2": TRAINING_CONFIG["beta2"],
            "tau": TRAINING_CONFIG["tau"],
        },
        "history": history,
        "final_metrics": history[-1] if history else {},
    }


def main():
    parser = argparse.ArgumentParser(description="Server-Side Optimizer FL Benchmark")
    parser.add_argument("--quick", action="store_true",
                        help="Quick validation (1 experiment)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show plan without executing")
    parser.add_argument("--fresh", action="store_true",
                        help="Start fresh (ignore checkpoint)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Build experiment list ──
    experiments = []

    if args.quick:
        algos = ["FedAdam"]
        datasets = ["Cardiovascular"]
        iid_modes = [True]
        seeds = [42]
    else:
        algos = ALL_ALGORITHMS
        datasets = list(DATASETS.keys())
        iid_modes = [True, False]
        seeds = [42, 123, 456]

    for algo in algos:
        for dataset in datasets:
            for is_iid in iid_modes:
                for seed in seeds:
                    iid_tag = "IID" if is_iid else "NonIID"
                    key = "{}_{}_{}_{}_s{}".format(algo, dataset, "K5", iid_tag, seed)
                    experiments.append((key, algo, dataset, seed, is_iid))

    total = len(experiments)

    log("=" * 70)
    log("  Server-Side Optimizer Benchmark")
    log("=" * 70)
    log("  Device: {}".format(DEVICE))
    log("  Algorithms: {}".format(", ".join(algos)))
    log("  Datasets: {}".format(", ".join(datasets)))
    log("  IID modes: {}".format(iid_modes))
    log("  Seeds: {}".format(seeds))
    log("  Grid: {} algos x {} datasets x {} IID x {} seeds = {} experiments".format(
        len(algos), len(datasets), len(iid_modes), len(seeds), total))
    log("  Server LR={}, beta1={}, beta2={}, tau={}".format(
        TRAINING_CONFIG["server_lr"], TRAINING_CONFIG["beta1"],
        TRAINING_CONFIG["beta2"], TRAINING_CONFIG["tau"]))
    log("=" * 70)

    if args.dry_run:
        for key, algo, dataset, seed, is_iid in experiments:
            log("  [DRY-RUN] {}".format(key))
        log("\nDRY-RUN: {} experiments.".format(total))
        return

    # Load or create checkpoint
    checkpoint = None if args.fresh else load_checkpoint()
    if checkpoint is None:
        checkpoint = {
            "completed": {},
            "metadata": {
                "experiment": "server_optimizers",
                "total_experiments": total,
                "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
        }

    t0 = time.time()
    completed = 0
    skipped = 0

    for idx, (key, algo, dataset, seed, is_iid) in enumerate(experiments, 1):
        if _shutdown:
            log("Shutdown. Saving checkpoint...")
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
                algorithm=algo, dataset=dataset, seed=seed, is_iid=is_iid,
                num_rounds=TRAINING_CONFIG["num_rounds"])
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
            checkpoint["completed"][key] = {"error": str(e)}
            save_checkpoint(checkpoint)
            _cleanup_gpu()

    elapsed = time.time() - t0
    checkpoint["metadata"]["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    checkpoint["metadata"]["total_time_seconds"] = round(elapsed, 1)
    save_checkpoint(checkpoint)

    log("")
    log("=" * 70)
    log("COMPLETED: {}/{} ({} skipped) in {:.0f} min".format(
        completed, total, skipped, elapsed / 60))
    log("Checkpoint: {}".format(OUTPUT_DIR / CHECKPOINT_FILE))
    log("=" * 70)


if __name__ == "__main__":
    main()

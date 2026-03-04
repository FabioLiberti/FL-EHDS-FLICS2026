#!/usr/bin/env python3
"""
Full CDC Diabetes Benchmark for FL-EHDS Paper.

Extends Cascade 8 Block W (16 experiments → ~210 experiments):
  Block A: 7 algorithms × 2 IID modes × 5 seeds = 70 experiments (~20 min)
  Block B: DP ablation: 3 algo × 4 eps × 2 IID × 3 seeds = 72 experiments (~25 min)
  Block C: Scalability: 3 algo × 4 K-values × 2 IID × 3 seeds = 72 experiments (~30 min)

Dataset: CDC BRFSS 2015 (253,680 samples, 21 features, binary diabetes prediction).
Largest tabular dataset — validates framework scalability at epidemiological scale.

Key findings expected:
  - Personalization effect on large-sample dataset (HPFL/Ditto vs FedAvg)
  - DP impact on 253K samples (noise-to-signal ratio much better than small datasets)
  - Class imbalance handling (~13% positive rate → F1 as primary metric)
  - Scalability to K=20 clients with 253K samples

Usage:
    cd fl-ehds-framework
    python -m benchmarks.run_cdc_diabetes_full [--quick] [--dry-run] [--block A|B|C]

Estimated time: ~75 min total on MacBook Air M3.

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
CHECKPOINT_FILE = "checkpoint_cdc_diabetes.json"
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
# Algorithms
# ======================================================================

ALL_ALGORITHMS = ["FedAvg", "FedProx", "Ditto", "FedLC", "FedExP", "FedLESAM", "HPFL"]
CORE_ALGORITHMS = ["FedAvg", "Ditto", "HPFL"]

TRAINING_CONFIG = dict(
    learning_rate=0.01,
    batch_size=128,
    num_rounds=25,
    local_epochs=3,
    mu=0.1,
    input_dim=21,
    num_classes=2,
)


def load_cdc_data(num_clients=5, seed=42, is_iid=False, alpha=0.5):
    """Load CDC Diabetes dataset."""
    from data.cdc_diabetes_loader import load_cdc_diabetes_data
    return load_cdc_diabetes_data(
        num_clients=num_clients, seed=seed, is_iid=is_iid, alpha=alpha)


def run_single_experiment(algorithm, num_clients, seed, is_iid, alpha,
                          dp_epsilon=None, num_rounds=25):
    """Run one FL experiment on CDC Diabetes."""
    from terminal.training.federated import FederatedTrainer

    client_data, client_test, meta = load_cdc_data(
        num_clients=num_clients, seed=seed, is_iid=is_iid, alpha=alpha)

    trainer = FederatedTrainer(
        num_clients=num_clients,
        algorithm=algorithm,
        local_epochs=TRAINING_CONFIG["local_epochs"],
        batch_size=TRAINING_CONFIG["batch_size"],
        learning_rate=TRAINING_CONFIG["learning_rate"],
        input_dim=TRAINING_CONFIG["input_dim"],
        num_classes=TRAINING_CONFIG["num_classes"],
        external_data=client_data,
        external_test_data=client_test,
        dp_enabled=dp_epsilon is not None,
        dp_epsilon=dp_epsilon if dp_epsilon else 10.0,
        dp_clip_norm=1.0,
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

    # F1 (binary)
    tp = int(((all_preds == 1) & (all_labels == 1)).sum())
    fp = int(((all_preds == 1) & (all_labels == 0)).sum())
    fn = int(((all_preds == 0) & (all_labels == 1)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-10)

    # Fairness
    accs = list(per_client_acc.values())
    jain = float(sum(accs) ** 2 / (len(accs) * sum(a ** 2 for a in accs))) if accs else 0

    samples_per_client = {str(cid): len(client_data[cid][1]) for cid in client_data}

    return {
        "dataset": "CDC_Diabetes",
        "algorithm": algorithm,
        "num_clients": num_clients,
        "seed": seed,
        "is_iid": is_iid,
        "alpha": alpha,
        "dp_epsilon": dp_epsilon,
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


def main():
    parser = argparse.ArgumentParser(description="Full CDC Diabetes FL Benchmark")
    parser.add_argument("--quick", action="store_true", help="Quick validation (3 experiments)")
    parser.add_argument("--dry-run", action="store_true", help="Show plan without executing")
    parser.add_argument("--fresh", action="store_true", help="Start fresh")
    parser.add_argument("--block", choices=["A", "B", "C"], help="Run only one block")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Check dataset exists
    data_dir = FRAMEWORK_DIR / "data" / "cdc_diabetes"
    csv_path = data_dir / "diabetes_binary_health_indicators_BRFSS2015.csv"
    if not csv_path.exists():
        log("ERROR: CDC Diabetes dataset not found at {}".format(csv_path))
        log("Download from: https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset")
        log("Place CSV in: {}".format(data_dir))
        sys.exit(1)

    # ── Build experiment list ──
    experiments = []

    # Block A: Full algorithm comparison (7 algo × 2 IID × 5 seeds = 70)
    if args.block is None or args.block == "A":
        if args.quick:
            algos_a = ["FedAvg", "HPFL"]
            seeds_a = [42]
            iid_modes_a = [True]
        else:
            algos_a = ALL_ALGORITHMS
            seeds_a = [42, 123, 456, 789, 999]
            iid_modes_a = [True, False]

        for algo in algos_a:
            for is_iid in iid_modes_a:
                for seed in seeds_a:
                    iid_tag = "IID" if is_iid else "NonIID"
                    key = "A_{}_K5_{}_s{}".format(algo, iid_tag, seed)
                    experiments.append((key, "A", algo, 5, seed, is_iid, 0.5, None))

    # Block B: DP ablation (3 algo × 4 eps × 2 IID × 3 seeds = 72)
    if args.block is None or args.block == "B":
        if args.quick:
            algos_b = ["FedAvg"]
            eps_list = [10.0]
            seeds_b = [42]
            iid_modes_b = [True]
        else:
            algos_b = CORE_ALGORITHMS
            eps_list = [1.0, 5.0, 10.0, 50.0]
            seeds_b = [42, 123, 456]
            iid_modes_b = [True, False]

        for algo in algos_b:
            for eps in eps_list:
                for is_iid in iid_modes_b:
                    for seed in seeds_b:
                        iid_tag = "IID" if is_iid else "NonIID"
                        key = "B_{}_eps{}_{}_s{}".format(algo, eps, iid_tag, seed)
                        experiments.append((key, "B", algo, 5, seed, is_iid, 0.5, eps))

    # Block C: Scalability (3 algo × 4 K × 2 IID × 3 seeds = 72)
    if args.block is None or args.block == "C":
        if args.quick:
            algos_c = ["FedAvg"]
            k_values = [3]
            seeds_c = [42]
            iid_modes_c = [True]
        else:
            algos_c = CORE_ALGORITHMS
            k_values = [3, 5, 10, 20]
            seeds_c = [42, 123, 456]
            iid_modes_c = [True, False]

        for algo in algos_c:
            for k in k_values:
                for is_iid in iid_modes_c:
                    for seed in seeds_c:
                        iid_tag = "IID" if is_iid else "NonIID"
                        key = "C_{}_K{}_{}_s{}".format(algo, k, iid_tag, seed)
                        experiments.append((key, "C", algo, k, seed, is_iid, 0.5, None))

    total = len(experiments)

    log("=" * 70)
    log("  CDC Diabetes Full Benchmark (253,680 samples)")
    log("=" * 70)
    log("  Device: {}".format(DEVICE))
    blocks_desc = {
        "A": "Algorithm comparison (7 algo × 2 IID × 5 seeds)",
        "B": "DP ablation (3 algo × 4 eps × 2 IID × 3 seeds)",
        "C": "Scalability (3 algo × 4 K × 2 IID × 3 seeds)",
    }
    block_counts = {}
    for _, block, *_ in experiments:
        block_counts[block] = block_counts.get(block, 0) + 1
    for b, desc in blocks_desc.items():
        if b in block_counts:
            log("  Block {}: {} — {} experiments".format(b, desc, block_counts[b]))
    log("  Total: {} experiments".format(total))
    log("=" * 70)

    if args.dry_run:
        for key, block, algo, k, seed, is_iid, alpha, dp_eps in experiments:
            log("  [DRY-RUN] {}".format(key))
        log("\nDRY-RUN: {} experiments.".format(total))
        return

    # Load or create checkpoint
    checkpoint = None if args.fresh else load_checkpoint()
    if checkpoint is None:
        checkpoint = {
            "completed": {},
            "metadata": {
                "experiment": "cdc_diabetes_full",
                "total_experiments": total,
                "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
        }

    t0 = time.time()
    completed = 0
    skipped = 0

    for idx, (key, block, algo, k, seed, is_iid, alpha, dp_eps) in enumerate(experiments, 1):
        if _shutdown:
            log("Shutdown. Saving checkpoint...")
            save_checkpoint(checkpoint)
            break

        if key in checkpoint["completed"]:
            skipped += 1
            continue

        completed += 1
        iid_tag = "IID" if is_iid else "NonIID"
        dp_tag = "eps={}".format(dp_eps) if dp_eps else "noDP"
        log("  [{}/{}] Block {} | {} | K={} | {} | {} | seed={}".format(
            skipped + completed, total, block, algo, k, iid_tag, dp_tag, seed))

        try:
            t_exp = time.time()
            result = run_single_experiment(
                algorithm=algo, num_clients=k, seed=seed,
                is_iid=is_iid, alpha=alpha, dp_epsilon=dp_eps)
            exp_time = time.time() - t_exp
            result["runtime_seconds"] = round(exp_time, 1)
            result["block"] = block
            checkpoint["completed"][key] = result
            save_checkpoint(checkpoint)
            _cleanup_gpu()

            log("    Acc={:.1f}% | F1={:.4f} | Jain={:.3f} | {:.1f}s".format(
                result["accuracy"] * 100, result["f1"],
                result["fairness"]["jain_index"], exp_time))

        except Exception as e:
            log("    ERROR: {}".format(e))
            traceback.print_exc()
            checkpoint["completed"][key] = {"error": str(e), "block": block}
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

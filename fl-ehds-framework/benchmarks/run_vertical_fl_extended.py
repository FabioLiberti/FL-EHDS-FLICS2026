#!/usr/bin/env python3
"""
Extended Vertical FL Experiments for FL-EHDS Paper.

Extends Cascade 8 Block V (24 experiments → ~120 experiments):
  - 3 datasets: Cardiovascular, PTB-XL, CDC Diabetes
  - 2-4 party configurations (EHDS scenario: hospital + lab + pharmacy + registry)
  - Overlap rates: 40%, 60%, 80%, 100% (partial patient population overlap)
  - DP levels: noDP, eps=5, eps=1
  - 3 seeds per configuration
  - Comparison metric: vertical vs horizontal FL accuracy gap

Key EHDS contribution: First systematic evaluation of vertical FL across
multiple clinical datasets with realistic party configurations and partial overlap.

Usage:
    cd fl-ehds-framework
    python -m benchmarks.run_vertical_fl_extended [--quick] [--dry-run]

Estimated time: ~45 min on MacBook Air M3.

Author: Fabio Liberti
"""

import argparse
import io
import json
import os
import signal
import shutil
import sys
import tempfile
import time
import traceback
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

FRAMEWORK_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(FRAMEWORK_DIR))

OUTPUT_DIR = Path(__file__).parent / "paper_results_tabular"
CHECKPOINT_FILE = "checkpoint_vertical_fl.json"

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


# ======================================================================
# Dataset vertical split definitions
# ======================================================================

DATASET_SPLITS = {
    "Cardiovascular": {
        "loader": "cardiovascular",
        "total_features": 11,
        "parties": {
            2: {
                "Party A (Demographics)": [0, 1, 2, 3],     # age, gender, height, weight
                "Party B (Clinical)": [4, 5, 6, 7, 8, 9, 10],  # BP, chol, gluc, lifestyle
            },
            3: {
                "Party A (Demographics)": [0, 1, 2, 3],     # age, gender, height, weight
                "Party B (Clinical Labs)": [4, 5, 6, 7],    # BP, cholesterol, glucose
                "Party C (Lifestyle)": [8, 9, 10],           # smoke, alco, active
            },
            4: {
                "Party A (Demographics)": [0, 1],            # age, gender
                "Party B (Anthropometrics)": [2, 3],         # height, weight
                "Party C (Vitals+Labs)": [4, 5, 6, 7],      # BP, chol, gluc
                "Party D (Lifestyle)": [8, 9, 10],           # smoke, alco, active
            },
        },
    },
    "PTB_XL": {
        "loader": "ptb_xl",
        "total_features": 9,
        "parties": {
            2: {
                "Party A (Signal Stats)": [0, 1, 2, 3, 4],  # first 5 PCA features
                "Party B (Clinical)": [5, 6, 7, 8],          # remaining features
            },
            3: {
                "Party A (Signal Primary)": [0, 1, 2],
                "Party B (Signal Secondary)": [3, 4, 5],
                "Party C (Clinical)": [6, 7, 8],
            },
        },
    },
    "CDC_Diabetes": {
        "loader": "cdc_diabetes",
        "total_features": 21,
        "parties": {
            2: {
                "Party A (Demographics+Access)": [17, 18, 19, 20, 11, 12],  # Sex,Age,Edu,Income,Healthcare,Cost
                "Party B (Clinical+Lifestyle)": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14, 15, 16],
            },
            3: {
                "Party A (Demographics)": [17, 18, 19, 20],  # Sex, Age, Education, Income
                "Party B (Conditions+Labs)": [0, 1, 2, 3, 5, 6],  # HighBP,HighChol,CholCheck,BMI,Stroke,Heart
                "Party C (Lifestyle+Wellness)": [4, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
            },
            4: {
                "Party A (Registry)": [17, 18, 19, 20],      # Demographics
                "Party B (Hospital)": [0, 1, 5, 6, 13, 14, 15, 16],  # Conditions, GenHlth, MentHlth, PhysHlth, DiffWalk
                "Party C (Lab)": [2, 3],                      # CholCheck, BMI
                "Party D (Pharmacy)": [4, 7, 8, 9, 10, 11, 12],  # Smoker, PhysActivity, Fruits, Veggies, Alco, Healthcare, Cost
            },
        },
    },
}


def load_dataset_centralized(ds_name, seed=42):
    """Load dataset as centralized data for vertical splitting."""
    if ds_name == "Cardiovascular":
        from data.cardiovascular_loader import load_cardiovascular_data
        client_data, client_test, meta = load_cardiovascular_data(
            num_clients=1, is_iid=True, seed=seed)
    elif ds_name == "PTB_XL":
        from data.ptbxl_loader import load_ptbxl_data
        client_data, client_test, meta = load_ptbxl_data(
            num_clients=1, seed=seed)
    elif ds_name == "CDC_Diabetes":
        from data.cdc_diabetes_loader import load_cdc_diabetes_data
        client_data, client_test, meta = load_cdc_diabetes_data(
            num_clients=1, is_iid=True, seed=seed)
    else:
        raise ValueError("Unknown dataset: {}".format(ds_name))

    X_train = np.concatenate([client_data[c][0] for c in client_data])
    y_train = np.concatenate([client_data[c][1] for c in client_data])
    X_test = np.concatenate([client_test[c][0] for c in client_test])
    y_test = np.concatenate([client_test[c][1] for c in client_test])

    # PTB-XL: binarize 5-class → NORM(0) vs ABNORMAL(1)
    # SplitNN uses sigmoid (binary only); clinically meaningful screening task
    if ds_name == "PTB_XL":
        y_train = (y_train > 0).astype(int)
        y_test = (y_test > 0).astype(int)

    return X_train, y_train, X_test, y_test, meta


def run_vertical_experiment(ds_name, n_parties, overlap, use_dp, dp_eps, seed, num_epochs=30):
    """Run a single vertical FL experiment."""
    from core.vertical_fl import (
        VerticalPartition, VerticalConfig, SecureVerticalFL,
        PrivateSetIntersection,
    )

    np.random.seed(seed)
    X_train, y_train, X_test, y_test, meta = load_dataset_centralized(ds_name, seed)

    n_samples = len(y_train)
    patient_ids = np.array(["P-{:06d}".format(i) for i in range(n_samples)])
    test_ids = np.array(["T-{:06d}".format(i) for i in range(len(y_test))])

    ds_splits = DATASET_SPLITS[ds_name]
    party_split = ds_splits["parties"][n_parties]
    party_names = list(party_split.keys())
    party_features = list(party_split.values())

    # Simulate partial overlap (non-label parties have subset of patients)
    if overlap < 1.0:
        n_overlap = int(n_samples * overlap)
        idx_overlap = np.sort(np.random.choice(n_samples, n_overlap, replace=False))
    else:
        idx_overlap = np.arange(n_samples)

    # Create vertical partitions
    partitions = {}
    for pid in range(n_parties):
        feats = party_features[pid]
        if pid == 0:
            # Label party: has all patients
            partitions[pid] = VerticalPartition(
                party_id=pid,
                features=X_train[:, feats],
                feature_names=[str(f) for f in feats],
                sample_ids=patient_ids,
                has_labels=True,
                labels=y_train.astype(float),
            )
        else:
            # Non-label parties: partial overlap
            partitions[pid] = VerticalPartition(
                party_id=pid,
                features=X_train[idx_overlap][:, feats],
                feature_names=[str(f) for f in feats],
                sample_ids=patient_ids[idx_overlap],
                has_labels=False,
            )

    # PSI alignment
    psi = PrivateSetIntersection()
    party_hashes = [psi.hash_ids(partitions[pid].sample_ids) for pid in sorted(partitions.keys())]
    aligned_tuple = psi.find_intersection(party_hashes)
    n_aligned = len(aligned_tuple[0])

    # Configure SplitNN
    p_configs = []
    for pid in range(n_parties):
        feats = party_features[pid]
        hidden_dim = max(8, len(feats) * 2)
        p_configs.append({
            "party_id": pid,
            "input_dim": len(feats),
            "hidden_dims": [hidden_dim, 8],
            "lr": 0.01,
        })

    config = VerticalConfig(
        algorithm="splitnn",
        use_differential_privacy=use_dp,
        epsilon=dp_eps if use_dp else 1.0,
    )

    vfl = SecureVerticalFL(config, p_configs, top_party_id=0)

    # Train (suppress output)
    f_buf = io.StringIO()
    with redirect_stdout(f_buf):
        history = vfl.train(partitions, num_epochs=num_epochs, batch_size=64)

    train_acc = history["accuracy"][-1] if history["accuracy"] else 0.0
    train_loss = history["loss"][-1] if history["loss"] else float("inf")

    # Per-epoch training curves
    training_history = {
        "accuracy": [round(float(a), 4) for a in history.get("accuracy", [])],
        "loss": [round(float(l), 4) for l in history.get("loss", [])],
    }

    # Test accuracy
    test_partitions = {}
    for pid in range(n_parties):
        feats = party_features[pid]
        test_partitions[pid] = VerticalPartition(
            party_id=pid,
            features=X_test[:, feats],
            feature_names=[str(f) for f in feats],
            sample_ids=test_ids,
            has_labels=(pid == 0),
            labels=y_test.astype(float) if pid == 0 else None,
        )

    test_data = {pid: tp.features for pid, tp in test_partitions.items()}
    preds = vfl.splitnn.predict(test_data)
    pred_labels = (preds.flatten() > 0.5).astype(int)
    test_acc = float(np.mean(pred_labels == y_test.flatten()))

    # F1 score
    tp = int(((pred_labels == 1) & (y_test.flatten() == 1)).sum())
    fp = int(((pred_labels == 1) & (y_test.flatten() == 0)).sum())
    fn = int(((pred_labels == 0) & (y_test.flatten() == 1)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-10)

    return {
        "dataset": ds_name,
        "n_parties": n_parties,
        "party_names": party_names,
        "use_dp": use_dp,
        "dp_epsilon": dp_eps if use_dp else None,
        "overlap_rate": overlap,
        "seed": seed,
        "num_epochs": num_epochs,
        "n_total_samples": n_samples,
        "n_aligned_samples": n_aligned,
        "alignment_rate": round(n_aligned / n_samples, 4),
        "train_accuracy": round(float(train_acc), 4),
        "train_loss": round(float(train_loss), 4),
        "test_accuracy": round(float(test_acc), 4),
        "test_f1": round(float(f1), 4),
        "test_precision": round(float(precision), 4),
        "test_recall": round(float(recall), 4),
        "convergence_epochs": len(history["accuracy"]),
        "training_history": training_history,
    }


def main():
    parser = argparse.ArgumentParser(description="Extended Vertical FL Experiments")
    parser.add_argument("--quick", action="store_true", help="Quick validation run")
    parser.add_argument("--dry-run", action="store_true", help="Show plan without executing")
    parser.add_argument("--fresh", action="store_true", help="Start fresh (ignore checkpoint)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Experiment grid
    if args.quick:
        datasets = ["Cardiovascular"]
        n_parties_map = {"Cardiovascular": [2]}
        overlaps = [1.0]
        dp_configs = [(False, None)]
        seeds = [42]
    else:
        datasets = ["Cardiovascular", "PTB_XL", "CDC_Diabetes"]
        n_parties_map = {
            "Cardiovascular": [2, 3, 4],
            "PTB_XL": [2, 3],
            "CDC_Diabetes": [2, 3, 4],
        }
        overlaps = [1.0, 0.8, 0.6]
        dp_configs = [(False, None), (True, 5.0), (True, 1.0)]
        seeds = [42, 123, 456]

    # Build experiment list
    experiments = []
    for ds in datasets:
        for np_ in n_parties_map[ds]:
            for ovlp in overlaps:
                for use_dp, dp_eps in dp_configs:
                    for seed in seeds:
                        dp_tag = "eps{}".format(dp_eps) if use_dp else "noDP"
                        key = "VFL_{}_{}p_{}_ovlp{}_s{}".format(ds, np_, dp_tag, ovlp, seed)
                        experiments.append((key, ds, np_, ovlp, use_dp, dp_eps, seed))

    total = len(experiments)

    log("=" * 70)
    log("Extended Vertical FL Experiments")
    log("=" * 70)
    log("  Datasets: {}".format(datasets))
    log("  Total experiments: {}".format(total))
    log("  Overlap rates: {}".format(overlaps))
    log("  DP configs: {}".format(["noDP", "eps=5", "eps=1"] if not args.quick else ["noDP"]))
    log("  Seeds: {}".format(seeds))
    log("=" * 70)

    if args.dry_run:
        for key, ds, np_, ovlp, use_dp, dp_eps, seed in experiments:
            log("  [DRY-RUN] {}".format(key))
        log("\nDRY-RUN: {} experiments would be run.".format(total))
        return

    # Load or create checkpoint
    checkpoint = None if args.fresh else load_checkpoint()
    if checkpoint is None:
        checkpoint = {
            "results": {},
            "metadata": {
                "experiment": "vertical_fl_extended",
                "total_experiments": total,
                "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
        }

    t0 = time.time()
    completed = 0
    skipped = 0

    for idx, (key, ds, np_, ovlp, use_dp, dp_eps, seed) in enumerate(experiments, 1):
        if _shutdown:
            log("Shutdown requested. Saving checkpoint...")
            save_checkpoint(checkpoint)
            break

        if key in checkpoint["results"]:
            skipped += 1
            continue

        completed += 1
        dp_tag = "eps={}".format(dp_eps) if use_dp else "noDP"
        log("  [{}/{}] {} | {}p | {} | ovlp={} | seed={}".format(
            skipped + completed, total, ds, np_, dp_tag, ovlp, seed))

        try:
            t_exp = time.time()
            result = run_vertical_experiment(ds, np_, ovlp, use_dp, dp_eps, seed)
            exp_time = time.time() - t_exp
            result["runtime_seconds"] = round(exp_time, 1)
            checkpoint["results"][key] = result
            save_checkpoint(checkpoint)

            log("    acc={:.1f}% | F1={:.4f} | aligned={:.0f}% | {:.1f}s".format(
                result["test_accuracy"] * 100, result["test_f1"],
                result["alignment_rate"] * 100, exp_time))
        except Exception as e:
            log("    ERROR: {}".format(e))
            traceback.print_exc()
            checkpoint["results"][key] = {"error": str(e)}
            save_checkpoint(checkpoint)

    elapsed = time.time() - t0
    checkpoint["metadata"]["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    checkpoint["metadata"]["total_time_seconds"] = round(elapsed, 1)
    save_checkpoint(checkpoint)

    log("")
    log("=" * 70)
    log("COMPLETED: {}/{} experiments ({} skipped) in {:.0f} min".format(
        completed, total, skipped, elapsed / 60))
    log("Checkpoint: {}".format(OUTPUT_DIR / CHECKPOINT_FILE))
    log("=" * 70)


if __name__ == "__main__":
    main()

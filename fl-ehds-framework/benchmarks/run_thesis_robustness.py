#!/usr/bin/env python3
"""
Thesis Robustness Cascade — FL-EHDS (FLICS 2026)
=================================================
Central thesis: "Personalization architecture — not aggregation algorithm —
is the critical deployment decision for FL in EHDS"

Phase A: Ditto Lambda Sensitivity          (75 experiments)
Phase B: Personalization vs Dataset Size   (225 experiments)
Phase C: Compound Stress Frontier          (216 experiments)
                                      Total: 516 experiments

Estimated runtime: ~4-5 hours on CPU
Checkpoint: paper_results_tabular/checkpoint_thesis_robustness.json
Resume:     python -m benchmarks.run_thesis_robustness

Author: Fabio Liberti
"""

import gc
import json
import os
import shutil
import signal
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Framework path setup
# ---------------------------------------------------------------------------
FRAMEWORK_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(FRAMEWORK_DIR))

import torch
from terminal.fl_trainer import FederatedTrainer

# ---------------------------------------------------------------------------
# Output configuration
# ---------------------------------------------------------------------------
OUTPUT_DIR = FRAMEWORK_DIR / "benchmarks" / "paper_results_tabular"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_FILE = "checkpoint_thesis_robustness.json"
LOG_FILE = "experiment_thesis_robustness.log"

# ---------------------------------------------------------------------------
# Seeds
# ---------------------------------------------------------------------------
SEEDS_5 = [42, 123, 456, 789, 999]
SEEDS_3 = [42, 123, 456]

# ---------------------------------------------------------------------------
# Dataset configurations
# ---------------------------------------------------------------------------
DATASET_CONFIGS = {
    "CV": {
        "name": "Cardiovascular",
        "short": "CV",
        "input_dim": 11,
        "num_classes": 2,
        "num_clients": 5,
        "num_rounds": 25,
        "local_epochs": 3,
        "learning_rate": 0.01,
        "batch_size": 64,
        "mu": 0.1,
    },
    "PX": {
        "name": "PTB-XL",
        "short": "PX",
        "input_dim": 9,
        "num_classes": 5,
        "num_clients": 5,
        "num_rounds": 30,
        "local_epochs": 3,
        "learning_rate": 0.005,
        "batch_size": 64,
        "mu": 0.1,
    },
    "HD": {
        "name": "Heart Disease",
        "short": "HD",
        "input_dim": 13,
        "num_classes": 2,
        "num_clients": 4,
        "num_rounds": 20,
        "local_epochs": 3,
        "learning_rate": 0.01,
        "batch_size": 64,
        "mu": 0.1,
    },
}

ALGORITHMS = ["FedAvg", "Ditto", "HPFL"]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
_log_fh = None


def log(msg: str) -> None:
    global _log_fh
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    if _log_fh is None:
        _log_fh = open(OUTPUT_DIR / LOG_FILE, "a")
    try:
        _log_fh.write(line + "\n")
        _log_fh.flush()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Checkpoint (atomic write with backup)
# ---------------------------------------------------------------------------
_checkpoint_data: Dict[str, Any] = {}


def save_checkpoint() -> None:
    global _checkpoint_data
    _checkpoint_data["metadata"]["last_save"] = datetime.now().isoformat()
    path = OUTPUT_DIR / CHECKPOINT_FILE
    bak = OUTPUT_DIR / (CHECKPOINT_FILE + ".bak")
    fd, tmp = tempfile.mkstemp(
        dir=str(OUTPUT_DIR), prefix=".thesis_", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(_checkpoint_data, f, indent=2, default=str)
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
    bak = OUTPUT_DIR / (CHECKPOINT_FILE + ".bak")
    for p in [path, bak]:
        if p.exists():
            try:
                with open(p) as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                continue
    return None


# ---------------------------------------------------------------------------
# Signal handling (graceful CTRL-C)
# ---------------------------------------------------------------------------
_shutdown = False


def _signal_handler(signum, frame):
    global _shutdown
    if _shutdown:
        log("Second interrupt — force exit")
        sys.exit(1)
    _shutdown = True
    done = len(_checkpoint_data.get("completed", {}))
    total = _checkpoint_data.get("metadata", {}).get("total_experiments", "?")
    log(f"\nINTERRUPT — saving checkpoint ({done}/{total})...")
    save_checkpoint()
    log("Checkpoint saved. Resume: python -m benchmarks.run_thesis_robustness")
    sys.exit(0)


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


# ---------------------------------------------------------------------------
# GPU/memory cleanup
# ---------------------------------------------------------------------------
def _cleanup():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        try:
            torch.mps.empty_cache()
        except Exception:
            pass
    gc.collect()


# ---------------------------------------------------------------------------
# Dataset loaders (cached per seed)
# ---------------------------------------------------------------------------
_data_cache: Dict[str, Any] = {}


def load_dataset(ds_key: str, seed: int, is_iid: bool = False,
                 alpha: float = 0.5) -> Tuple[Dict, Dict, Dict]:
    """Load dataset with caching to avoid re-loading across experiments."""
    cache_key = f"{ds_key}_s{seed}_iid{is_iid}_a{alpha}"
    if cache_key in _data_cache:
        # Return deep copy of training data (test data can be shared)
        cached = _data_cache[cache_key]
        train_copy = {
            cid: (X.copy(), y.copy()) for cid, (X, y) in cached[0].items()
        }
        return train_copy, cached[1], cached[2]

    if ds_key == "CV":
        from data.cardiovascular_loader import load_cardiovascular_data
        train, test, meta = load_cardiovascular_data(
            num_clients=DATASET_CONFIGS["CV"]["num_clients"],
            is_iid=is_iid,
            alpha=alpha,
            seed=seed,
        )
    elif ds_key == "PX":
        from data.ptbxl_loader import load_ptbxl_data
        train, test, meta = load_ptbxl_data(
            num_clients=DATASET_CONFIGS["PX"]["num_clients"],
            partition_by_site=True,
            seed=seed,
            min_site_samples=50,
        )
    elif ds_key == "HD":
        from data.heart_disease_loader import load_heart_disease_data
        train, test, meta = load_heart_disease_data(
            num_clients=DATASET_CONFIGS["HD"]["num_clients"],
            partition_by_hospital=True,
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown dataset: {ds_key}")

    _data_cache[cache_key] = (train, test, meta)
    train_copy = {cid: (X.copy(), y.copy()) for cid, (X, y) in train.items()}
    return train_copy, test, meta


# ---------------------------------------------------------------------------
# Data manipulation helpers
# ---------------------------------------------------------------------------
def subsample_data(
    client_data: Dict[int, Tuple[np.ndarray, np.ndarray]],
    fraction: float,
    seed: int,
) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """Subsample each client's training data to a given fraction."""
    rng = np.random.RandomState(seed + 7777)
    result = {}
    for cid, (X, y) in client_data.items():
        n = len(y)
        k = max(2, int(n * fraction))
        indices = rng.choice(n, size=k, replace=False)
        indices.sort()
        result[cid] = (X[indices], y[indices])
    return result


def apply_optout(
    client_data: Dict[int, Tuple[np.ndarray, np.ndarray]],
    optout_rate: float,
    seed: int,
) -> Tuple[Dict[int, Tuple[np.ndarray, np.ndarray]], int, int]:
    """Remove optout_rate fraction of samples uniformly."""
    if optout_rate <= 0:
        return dict(client_data), 0, sum(len(y) for _, (_, y) in client_data.items())

    rng = np.random.RandomState(seed + 8888)
    # Build flat index
    all_indices = []
    for cid in sorted(client_data.keys()):
        n = len(client_data[cid][1])
        for i in range(n):
            all_indices.append((cid, i))

    total = len(all_indices)
    n_remove = int(total * optout_rate)
    remove_flat = set(
        rng.choice(total, size=n_remove, replace=False).tolist()
    )

    # Build removal mask per client
    per_client_remove: Dict[int, set] = {cid: set() for cid in client_data}
    for flat_idx in remove_flat:
        cid, local_idx = all_indices[flat_idx]
        per_client_remove[cid].add(local_idx)

    filtered = {}
    excluded = 0
    for cid in sorted(client_data.keys()):
        X, y = client_data[cid]
        mask = np.array([i not in per_client_remove[cid] for i in range(len(y))])
        excluded += int((~mask).sum())
        if mask.sum() < 2:
            # Keep at least 2 samples
            filtered[cid] = (X[:2], y[:2])
        else:
            filtered[cid] = (X[mask], y[mask])

    return filtered, excluded, total


# ---------------------------------------------------------------------------
# Core FL training function
# ---------------------------------------------------------------------------
def run_fl_experiment(
    algorithm: str,
    ds_key: str,
    seed: int,
    mu: float = 0.1,
    is_iid: bool = False,
    alpha: float = 0.5,
    dp_enabled: bool = False,
    dp_epsilon: float = 10.0,
    participation_rate: float = 1.0,
    optout_rate: float = 0.0,
    data_fraction: float = 1.0,
) -> Dict[str, Any]:
    """Run a single FL experiment and return results dict."""
    cfg = DATASET_CONFIGS[ds_key]
    t0 = time.time()

    # Load data
    train_data, test_data, meta = load_dataset(ds_key, seed, is_iid, alpha)

    # Apply data fraction (subsampling)
    if data_fraction < 1.0:
        train_data = subsample_data(train_data, data_fraction, seed)

    # Apply opt-out
    n_excluded, n_total = 0, 0
    if optout_rate > 0:
        train_data, n_excluded, n_total = apply_optout(
            train_data, optout_rate, seed
        )

    # Create trainer
    trainer = FederatedTrainer(
        num_clients=cfg["num_clients"],
        algorithm=algorithm,
        local_epochs=cfg["local_epochs"],
        batch_size=cfg["batch_size"],
        learning_rate=cfg["learning_rate"],
        mu=mu,
        seed=seed,
        external_data=train_data,
        external_test_data=test_data,
        input_dim=cfg["input_dim"],
        num_classes=cfg["num_classes"],
        dp_enabled=dp_enabled,
        dp_epsilon=dp_epsilon if dp_enabled else 0,
        dp_clip_norm=1.0,
    )

    # Partial participation setup
    all_clients = list(range(cfg["num_clients"]))
    use_partial = participation_rate < 1.0
    n_active = max(2, int(cfg["num_clients"] * participation_rate))
    part_rng = np.random.RandomState(seed + 9999)

    # Train
    history = []
    for r in range(cfg["num_rounds"]):
        if _shutdown:
            break
        if use_partial:
            active = sorted(
                part_rng.choice(
                    all_clients, size=n_active, replace=False
                ).tolist()
            )
            result = trainer.train_round(r, active_clients=active)
        else:
            result = trainer.train_round(r)

        entry = {
            "round": r + 1,
            "accuracy": getattr(result, "global_acc", 0),
            "loss": getattr(result, "global_loss", 0),
        }
        # Safely try to get F1 and AUC
        for attr in ["global_f1", "global_auc"]:
            val = getattr(result, attr, None)
            if val is not None:
                entry[attr.replace("global_", "")] = val
        history.append(entry)

    elapsed = time.time() - t0
    final = history[-1] if history else {}

    # Per-client accuracy if available
    per_client_accs = None
    for attr_name in ["per_client_accs", "client_accuracies", "per_client_results"]:
        val = getattr(result, attr_name, None) if history else None
        if val is not None:
            per_client_accs = val
            break

    # Compute fairness metrics
    fairness = {}
    if per_client_accs and isinstance(per_client_accs, (list, dict)):
        accs = list(per_client_accs.values()) if isinstance(per_client_accs, dict) else per_client_accs
        accs = [float(a) for a in accs if a is not None]
        if len(accs) >= 2 and max(accs) > 0:
            n = len(accs)
            sum_x = sum(accs)
            sum_x2 = sum(a**2 for a in accs)
            fairness = {
                "jain": round((sum_x**2) / (n * sum_x2), 4) if sum_x2 > 0 else 0,
                "dei_ratio": round(min(accs) / max(accs), 4),
                "range": round(max(accs) - min(accs), 4),
                "std": round(float(np.std(accs)), 4),
            }

    return {
        "algorithm": algorithm,
        "dataset": ds_key,
        "seed": seed,
        "mu": mu,
        "is_iid": is_iid,
        "alpha": alpha,
        "dp_enabled": dp_enabled,
        "dp_epsilon": dp_epsilon if dp_enabled else None,
        "participation_rate": participation_rate,
        "optout_rate": optout_rate,
        "data_fraction": data_fraction,
        "final_accuracy": final.get("accuracy", 0),
        "final_f1": final.get("f1", None),
        "final_auc": final.get("auc", None),
        "final_loss": final.get("loss", None),
        "rounds_completed": len(history),
        "fairness": fairness,
        "optout_excluded": n_excluded,
        "optout_total": n_total,
        "runtime_seconds": round(elapsed, 1),
    }


# ===================================================================
# PHASE A: Ditto Lambda Sensitivity (75 experiments)
# ===================================================================
def build_phase_a() -> List[Dict]:
    """Ditto lambda (mu) sensitivity: 5 lambda x 3 datasets x 5 seeds."""
    experiments = []
    LAMBDAS = [0.01, 0.05, 0.1, 0.5, 1.0]
    for lam in LAMBDAS:
        for ds_key in ["CV", "PX", "HD"]:
            for seed in SEEDS_5:
                experiments.append({
                    "key": f"A_Ditto_lam{lam}_{ds_key}_s{seed}",
                    "phase": "A",
                    "algorithm": "Ditto",
                    "dataset": ds_key,
                    "seed": seed,
                    "mu": lam,
                    "description": f"Ditto lambda={lam} on {ds_key}",
                })
    return experiments


# ===================================================================
# PHASE B: Personalization Gap vs Dataset Size (225 experiments)
# ===================================================================
def build_phase_b() -> List[Dict]:
    """Data fraction sweep: 5 fractions x 3 algos x 3 datasets x 5 seeds."""
    experiments = []
    FRACTIONS = [0.10, 0.25, 0.50, 0.75, 1.0]
    for frac in FRACTIONS:
        for algo in ALGORITHMS:
            for ds_key in ["CV", "PX", "HD"]:
                for seed in SEEDS_5:
                    experiments.append({
                        "key": f"B_{algo}_f{frac}_{ds_key}_s{seed}",
                        "phase": "B",
                        "algorithm": algo,
                        "dataset": ds_key,
                        "seed": seed,
                        "data_fraction": frac,
                        "description": f"{algo} frac={frac} on {ds_key}",
                    })
    return experiments


# ===================================================================
# PHASE C: Compound Stress Frontier (216 experiments)
# ===================================================================
def build_phase_c() -> List[Dict]:
    """Compound stress on CV: IID x DP x Participation x Opt-out x Algo x Seed."""
    experiments = []
    IID_LEVELS = [True, False]
    DP_LEVELS = [
        {"label": "noDP", "dp_enabled": False, "dp_epsilon": 0},
        {"label": "eps10", "dp_enabled": True, "dp_epsilon": 10.0},
        {"label": "eps1", "dp_enabled": True, "dp_epsilon": 1.0},
    ]
    PARTICIPATION = [
        {"label": "full", "rate": 1.0},
        {"label": "part60", "rate": 0.6},
    ]
    OPTOUT = [
        {"label": "noOO", "rate": 0.0},
        {"label": "oo20", "rate": 0.2},
    ]

    ds_key = "CV"  # Cardiovascular — supports IID/non-IID cleanly
    for iid in IID_LEVELS:
        iid_lbl = "IID" if iid else "NonIID"
        for dp in DP_LEVELS:
            for part in PARTICIPATION:
                for oo in OPTOUT:
                    cond_label = f"{iid_lbl}_{dp['label']}_{part['label']}_{oo['label']}"
                    for algo in ALGORITHMS:
                        for seed in SEEDS_3:
                            experiments.append({
                                "key": f"C_{algo}_{cond_label}_s{seed}",
                                "phase": "C",
                                "algorithm": algo,
                                "dataset": ds_key,
                                "seed": seed,
                                "is_iid": iid,
                                "dp_enabled": dp["dp_enabled"],
                                "dp_epsilon": dp["dp_epsilon"],
                                "participation_rate": part["rate"],
                                "optout_rate": oo["rate"],
                                "description": f"{algo} {cond_label}",
                            })
    return experiments


# ===================================================================
# MAIN
# ===================================================================
def main():
    global _checkpoint_data, _shutdown

    # Build all experiments
    all_experiments = []
    phase_a = build_phase_a()
    phase_b = build_phase_b()
    phase_c = build_phase_c()
    all_experiments.extend(phase_a)
    all_experiments.extend(phase_b)
    all_experiments.extend(phase_c)
    total = len(all_experiments)

    log("=" * 70)
    log("THESIS ROBUSTNESS CASCADE — FL-EHDS")
    log("=" * 70)
    log(f"Phase A: Ditto Lambda Sensitivity       {len(phase_a):>4} experiments")
    log(f"Phase B: Personalization vs Dataset Size {len(phase_b):>4} experiments")
    log(f"Phase C: Compound Stress Frontier       {len(phase_c):>4} experiments")
    log(f"{'TOTAL':>40s} {total:>4} experiments")
    log(f"Checkpoint: {CHECKPOINT_FILE}")
    log("=" * 70)

    # Load or init checkpoint
    _checkpoint_data = load_checkpoint() or {
        "completed": {},
        "metadata": {
            "cascade": "thesis_robustness",
            "total_experiments": total,
            "phases": {
                "A": f"Ditto lambda sensitivity ({len(phase_a)} exp)",
                "B": f"Personalization vs dataset size ({len(phase_b)} exp)",
                "C": f"Compound stress frontier ({len(phase_c)} exp)",
            },
            "start_time": datetime.now().isoformat(),
        },
    }
    _checkpoint_data["metadata"]["total_experiments"] = total

    completed = _checkpoint_data.setdefault("completed", {})
    done_count = len(completed)

    if done_count > 0:
        log(f"AUTO-RESUMED: {done_count}/{total} already completed")
        # Count per phase
        for phase_id, phase_list in [("A", phase_a), ("B", phase_b), ("C", phase_c)]:
            phase_done = sum(1 for e in phase_list if e["key"] in completed)
            log(f"  Phase {phase_id}: {phase_done}/{len(phase_list)}")

    log("")
    t_start = time.time()

    # Run experiments
    for idx, exp in enumerate(all_experiments, 1):
        if _shutdown:
            break

        key = exp["key"]
        if key in completed:
            continue

        desc = exp.get("description", key)
        log(f"[{done_count + 1}/{total}] Phase {exp['phase']}: {desc} ...")

        try:
            result = run_fl_experiment(
                algorithm=exp["algorithm"],
                ds_key=exp["dataset"],
                seed=exp["seed"],
                mu=exp.get("mu", DATASET_CONFIGS[exp["dataset"]]["mu"]),
                is_iid=exp.get("is_iid", False),
                alpha=exp.get("alpha", 0.5),
                dp_enabled=exp.get("dp_enabled", False),
                dp_epsilon=exp.get("dp_epsilon", 10.0),
                participation_rate=exp.get("participation_rate", 1.0),
                optout_rate=exp.get("optout_rate", 0.0),
                data_fraction=exp.get("data_fraction", 1.0),
            )

            result["key"] = key
            result["phase"] = exp["phase"]
            completed[key] = result
            done_count += 1
            save_checkpoint()

            acc = result.get("final_accuracy", 0)
            rt = result.get("runtime_seconds", 0)
            log(f"  -> acc={acc * 100:.1f}%  ({rt:.0f}s)")

        except Exception as e:
            log(f"  ERROR: {e}")
            completed[key] = {
                "key": key,
                "phase": exp["phase"],
                "error": str(e),
                "algorithm": exp["algorithm"],
                "dataset": exp["dataset"],
                "seed": exp["seed"],
            }
            done_count += 1
            save_checkpoint()

        _cleanup()

    # Final summary
    elapsed_total = time.time() - t_start
    _checkpoint_data["metadata"]["end_time"] = datetime.now().isoformat()
    _checkpoint_data["metadata"]["total_elapsed_seconds"] = round(elapsed_total, 1)
    save_checkpoint()

    log("")
    log("=" * 70)
    log("CASCADE COMPLETE")
    log(f"Total: {done_count}/{total} experiments")
    log(f"Elapsed: {elapsed_total / 3600:.1f} hours")
    log(f"Checkpoint: {OUTPUT_DIR / CHECKPOINT_FILE}")
    log("=" * 70)

    # Print phase summaries
    for phase_id, phase_list in [("A", phase_a), ("B", phase_b), ("C", phase_c)]:
        phase_results = [
            completed[e["key"]]
            for e in phase_list
            if e["key"] in completed and "error" not in completed[e["key"]]
        ]
        if not phase_results:
            continue

        log(f"\n--- Phase {phase_id} Summary ---")
        errors = sum(
            1 for e in phase_list
            if e["key"] in completed and "error" in completed[e["key"]]
        )
        if errors:
            log(f"  Errors: {errors}")

        if phase_id == "A":
            # Lambda sensitivity: show accuracy per lambda
            log("  Lambda | CV acc | PX acc | HD acc")
            for lam in [0.01, 0.05, 0.1, 0.5, 1.0]:
                accs = {}
                for ds in ["CV", "PX", "HD"]:
                    ds_results = [
                        r["final_accuracy"]
                        for r in phase_results
                        if r.get("mu") == lam and r.get("dataset") == ds
                    ]
                    accs[ds] = np.mean(ds_results) * 100 if ds_results else 0
                log(f"  {lam:>5.2f}  | {accs['CV']:>5.1f}% | {accs['PX']:>5.1f}% | {accs['HD']:>5.1f}%")

        elif phase_id == "B":
            # Dataset size: show personalization gap per fraction
            log("  Frac  | FedAvg | Ditto  | HPFL   | Gap (D-FA)")
            for frac in [0.10, 0.25, 0.50, 0.75, 1.0]:
                algo_accs = {}
                for algo in ALGORITHMS:
                    vals = [
                        r["final_accuracy"]
                        for r in phase_results
                        if r.get("data_fraction") == frac and r.get("algorithm") == algo
                    ]
                    algo_accs[algo] = np.mean(vals) * 100 if vals else 0
                gap = algo_accs["Ditto"] - algo_accs["FedAvg"]
                log(
                    f"  {frac:>4.0%}  | {algo_accs['FedAvg']:>5.1f}% | "
                    f"{algo_accs['Ditto']:>5.1f}% | {algo_accs['HPFL']:>5.1f}% | "
                    f"{gap:>+5.1f}pp"
                )

        elif phase_id == "C":
            # Compound stress: show best/worst conditions
            algo_results: Dict[str, List[float]] = {a: [] for a in ALGORITHMS}
            for r in phase_results:
                algo_results[r["algorithm"]].append(r["final_accuracy"])
            log("  Algorithm | Mean acc | Std   | Min   | Max")
            for algo in ALGORITHMS:
                vals = algo_results[algo]
                if vals:
                    log(
                        f"  {algo:<9s} | {np.mean(vals) * 100:>6.1f}% | "
                        f"{np.std(vals) * 100:>4.1f}% | {np.min(vals) * 100:>5.1f}% | "
                        f"{np.max(vals) * 100:>5.1f}%"
                    )

    log("\nDone.")
    if _log_fh:
        _log_fh.close()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Thesis Robustness — QUICK VALIDATION (smoke test)
==================================================
Same logic as run_thesis_robustness.py but:
  - 1 seed only (42)
  - Phase A: 3 lambdas x 1 dataset (HD) = 3 experiments
  - Phase B: 3 fractions x 3 algos x 1 dataset (HD) = 9 experiments
  - Phase C: 4 key conditions x 3 algos x 1 seed on CV = 12 experiments
  Total: 24 experiments (~5-10 minutes)

Usage: cd fl-ehds-framework && python -m benchmarks.run_thesis_robustness_quick
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

FRAMEWORK_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(FRAMEWORK_DIR))

import torch
from terminal.fl_trainer import FederatedTrainer

OUTPUT_DIR = FRAMEWORK_DIR / "benchmarks" / "paper_results_tabular"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_FILE = "checkpoint_thesis_quick.json"

DATASET_CONFIGS = {
    "CV": {
        "input_dim": 11, "num_classes": 2, "num_clients": 5,
        "num_rounds": 10, "local_epochs": 2, "learning_rate": 0.01,
        "batch_size": 64, "mu": 0.1,
    },
    "HD": {
        "input_dim": 13, "num_classes": 2, "num_clients": 4,
        "num_rounds": 10, "local_epochs": 2, "learning_rate": 0.01,
        "batch_size": 64, "mu": 0.1,
    },
}

ALGORITHMS = ["FedAvg", "Ditto", "HPFL"]

# --- Logging ---
_log_fh = None
def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)

# --- Checkpoint ---
_checkpoint_data: Dict[str, Any] = {}

def save_checkpoint():
    _checkpoint_data["metadata"]["last_save"] = datetime.now().isoformat()
    path = OUTPUT_DIR / CHECKPOINT_FILE
    bak = OUTPUT_DIR / (CHECKPOINT_FILE + ".bak")
    fd, tmp = tempfile.mkstemp(dir=str(OUTPUT_DIR), prefix=".quick_", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(_checkpoint_data, f, indent=2, default=str)
            f.flush(); os.fsync(f.fileno())
        if path.exists():
            shutil.copy2(str(path), str(bak))
        os.replace(tmp, str(path))
    except Exception:
        try: os.unlink(tmp)
        except OSError: pass
        raise

def load_checkpoint():
    for p in [OUTPUT_DIR / CHECKPOINT_FILE, OUTPUT_DIR / (CHECKPOINT_FILE + ".bak")]:
        if p.exists():
            try:
                with open(p) as f: return json.load(f)
            except (json.JSONDecodeError, IOError): continue
    return None

# --- Signal ---
_shutdown = False
def _signal_handler(signum, frame):
    global _shutdown
    if _shutdown: sys.exit(1)
    _shutdown = True
    log(f"INTERRUPT — saving checkpoint..."); save_checkpoint(); sys.exit(0)
signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)

# --- Data ---
_data_cache = {}

def load_dataset(ds_key, seed, is_iid=False, alpha=0.5):
    cache_key = f"{ds_key}_s{seed}_iid{is_iid}_a{alpha}"
    if cache_key in _data_cache:
        cached = _data_cache[cache_key]
        return {c: (X.copy(), y.copy()) for c, (X, y) in cached[0].items()}, cached[1], cached[2]
    if ds_key == "CV":
        from data.cardiovascular_loader import load_cardiovascular_data
        train, test, meta = load_cardiovascular_data(num_clients=5, is_iid=is_iid, alpha=alpha, seed=seed)
    elif ds_key == "HD":
        from data.heart_disease_loader import load_heart_disease_data
        train, test, meta = load_heart_disease_data(num_clients=4, partition_by_hospital=True, seed=seed)
    else:
        raise ValueError(ds_key)
    _data_cache[cache_key] = (train, test, meta)
    return {c: (X.copy(), y.copy()) for c, (X, y) in train.items()}, test, meta

def subsample_data(client_data, fraction, seed):
    rng = np.random.RandomState(seed + 7777)
    result = {}
    for cid, (X, y) in client_data.items():
        k = max(2, int(len(y) * fraction))
        idx = sorted(rng.choice(len(y), size=k, replace=False))
        result[cid] = (X[idx], y[idx])
    return result

def apply_optout(client_data, rate, seed):
    if rate <= 0:
        return dict(client_data), 0, sum(len(y) for _, (_, y) in client_data.items())
    rng = np.random.RandomState(seed + 8888)
    all_idx = [(c, i) for c in sorted(client_data) for i in range(len(client_data[c][1]))]
    total = len(all_idx)
    n_rm = int(total * rate)
    rm_flat = set(rng.choice(total, size=n_rm, replace=False).tolist())
    per_c = {c: set() for c in client_data}
    for fi in rm_flat:
        c, li = all_idx[fi]; per_c[c].add(li)
    filt = {}; excl = 0
    for c in sorted(client_data):
        X, y = client_data[c]
        mask = np.array([i not in per_c[c] for i in range(len(y))])
        excl += int((~mask).sum())
        filt[c] = (X[mask], y[mask]) if mask.sum() >= 2 else (X[:2], y[:2])
    return filt, excl, total

# --- FL runner ---
def run_fl(algo, ds_key, seed, mu=0.1, is_iid=False, dp_enabled=False,
           dp_epsilon=10.0, participation_rate=1.0, optout_rate=0.0, data_fraction=1.0):
    cfg = DATASET_CONFIGS[ds_key]
    t0 = time.time()
    train, test, meta = load_dataset(ds_key, seed, is_iid)
    if data_fraction < 1.0:
        train = subsample_data(train, data_fraction, seed)
    if optout_rate > 0:
        train, _, _ = apply_optout(train, optout_rate, seed)

    trainer = FederatedTrainer(
        num_clients=cfg["num_clients"], algorithm=algo,
        local_epochs=cfg["local_epochs"], batch_size=cfg["batch_size"],
        learning_rate=cfg["learning_rate"], mu=mu, seed=seed,
        external_data=train, external_test_data=test,
        input_dim=cfg["input_dim"], num_classes=cfg["num_classes"],
        dp_enabled=dp_enabled, dp_epsilon=dp_epsilon if dp_enabled else 0,
        dp_clip_norm=1.0,
    )
    use_partial = participation_rate < 1.0
    n_active = max(2, int(cfg["num_clients"] * participation_rate))
    part_rng = np.random.RandomState(seed + 9999)
    all_c = list(range(cfg["num_clients"]))

    history = []
    for r in range(cfg["num_rounds"]):
        if _shutdown: break
        if use_partial:
            active = sorted(part_rng.choice(all_c, size=n_active, replace=False).tolist())
            res = trainer.train_round(r, active_clients=active)
        else:
            res = trainer.train_round(r)
        history.append({"round": r+1, "accuracy": getattr(res, "global_acc", 0)})

    final = history[-1] if history else {}
    return {
        "algorithm": algo, "dataset": ds_key, "seed": seed, "mu": mu,
        "is_iid": is_iid, "dp_enabled": dp_enabled,
        "dp_epsilon": dp_epsilon if dp_enabled else None,
        "participation_rate": participation_rate, "optout_rate": optout_rate,
        "data_fraction": data_fraction,
        "final_accuracy": final.get("accuracy", 0),
        "runtime_seconds": round(time.time() - t0, 1),
    }

# --- Experiments ---
def build_experiments():
    exps = []
    # Phase A: Ditto lambda on HD (3 lambdas x 1 seed)
    for lam in [0.01, 0.1, 1.0]:
        exps.append({"key": f"A_Ditto_lam{lam}_HD_s42", "phase": "A",
                      "algo": "Ditto", "ds": "HD", "seed": 42, "mu": lam})
    # Phase B: data fraction on HD (3 fractions x 3 algos x 1 seed)
    for frac in [0.10, 0.50, 1.0]:
        for algo in ALGORITHMS:
            exps.append({"key": f"B_{algo}_f{frac}_HD_s42", "phase": "B",
                          "algo": algo, "ds": "HD", "seed": 42, "frac": frac})
    # Phase C: compound stress on CV (4 key conditions x 3 algos x 1 seed)
    conditions = [
        {"label": "baseline",        "iid": False, "dp": False, "eps": 0,  "part": 1.0, "oo": 0.0},
        {"label": "DP_eps1",         "iid": False, "dp": True,  "eps": 1,  "part": 1.0, "oo": 0.0},
        {"label": "compound_light",  "iid": False, "dp": True,  "eps": 10, "part": 0.6, "oo": 0.2},
        {"label": "compound_heavy",  "iid": False, "dp": True,  "eps": 1,  "part": 0.6, "oo": 0.2},
    ]
    for cond in conditions:
        for algo in ALGORITHMS:
            exps.append({"key": f"C_{algo}_{cond['label']}_s42", "phase": "C",
                          "algo": algo, "ds": "CV", "seed": 42,
                          "iid": cond["iid"], "dp": cond["dp"], "eps": cond["eps"],
                          "part": cond["part"], "oo": cond["oo"]})
    return exps

# --- Main ---
def main():
    global _checkpoint_data, _shutdown
    experiments = build_experiments()
    total = len(experiments)

    log("=" * 60)
    log("THESIS ROBUSTNESS — QUICK VALIDATION")
    log(f"Total: {total} experiments (expect ~5-10 min)")
    log("=" * 60)

    _checkpoint_data = load_checkpoint() or {
        "completed": {},
        "metadata": {"total_experiments": total, "start_time": datetime.now().isoformat()},
    }
    completed = _checkpoint_data.setdefault("completed", {})
    done = len(completed)
    if done > 0:
        log(f"Resumed: {done}/{total} done")

    t_start = time.time()
    for exp in experiments:
        if _shutdown: break
        key = exp["key"]
        if key in completed: continue

        log(f"[{done+1}/{total}] {key} ...")
        try:
            result = run_fl(
                algo=exp["algo"], ds_key=exp["ds"], seed=exp["seed"],
                mu=exp.get("mu", DATASET_CONFIGS[exp["ds"]]["mu"]),
                is_iid=exp.get("iid", False),
                dp_enabled=exp.get("dp", False), dp_epsilon=exp.get("eps", 10),
                participation_rate=exp.get("part", 1.0),
                optout_rate=exp.get("oo", 0.0),
                data_fraction=exp.get("frac", 1.0),
            )
            result["key"] = key; result["phase"] = exp["phase"]
            completed[key] = result; done += 1; save_checkpoint()
            log(f"  -> acc={result['final_accuracy']*100:.1f}%  ({result['runtime_seconds']:.0f}s)")
        except Exception as e:
            log(f"  ERROR: {e}")
            import traceback; traceback.print_exc()
            completed[key] = {"key": key, "phase": exp["phase"], "error": str(e)}
            done += 1; save_checkpoint()
        gc.collect()

    elapsed = time.time() - t_start
    _checkpoint_data["metadata"]["end_time"] = datetime.now().isoformat()
    save_checkpoint()

    # Summary
    log("\n" + "=" * 60)
    log(f"QUICK VALIDATION COMPLETE: {done}/{total} in {elapsed:.0f}s")
    log("=" * 60)

    # Phase A
    log("\n--- Phase A: Ditto Lambda ---")
    for lam in [0.01, 0.1, 1.0]:
        k = f"A_Ditto_lam{lam}_HD_s42"
        r = completed.get(k, {})
        acc = r.get("final_accuracy", 0)
        log(f"  lambda={lam:>4.2f}  acc={acc*100:.1f}%")

    # Phase B
    log("\n--- Phase B: Data Fraction (HD) ---")
    log("  Frac  | FedAvg | Ditto  | HPFL   | Gap(D-FA)")
    for frac in [0.10, 0.50, 1.0]:
        accs = {}
        for algo in ALGORITHMS:
            k = f"B_{algo}_f{frac}_HD_s42"
            accs[algo] = completed.get(k, {}).get("final_accuracy", 0) * 100
        gap = accs["Ditto"] - accs["FedAvg"]
        log(f"  {frac:>4.0%}  | {accs['FedAvg']:>5.1f}% | {accs['Ditto']:>5.1f}% | {accs['HPFL']:>5.1f}% | {gap:>+5.1f}pp")

    # Phase C
    log("\n--- Phase C: Compound Stress (CV) ---")
    log("  Condition        | FedAvg | Ditto  | HPFL")
    for cond in ["baseline", "DP_eps1", "compound_light", "compound_heavy"]:
        accs = {}
        for algo in ALGORITHMS:
            k = f"C_{algo}_{cond}_s42"
            accs[algo] = completed.get(k, {}).get("final_accuracy", 0) * 100
        log(f"  {cond:<18s}| {accs['FedAvg']:>5.1f}% | {accs['Ditto']:>5.1f}% | {accs['HPFL']:>5.1f}%")

    errors = [k for k, v in completed.items() if "error" in v]
    if errors:
        log(f"\nERRORS ({len(errors)}):")
        for k in errors:
            log(f"  {k}: {completed[k]['error']}")

    log(f"\nCheckpoint: {OUTPUT_DIR / CHECKPOINT_FILE}")

if __name__ == "__main__":
    main()

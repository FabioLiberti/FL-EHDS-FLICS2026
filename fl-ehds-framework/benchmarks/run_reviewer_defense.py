#!/usr/bin/env python3
"""
Reviewer Defense Experiments — Cascade Script
==============================================
4 phases, 236 total experiments targeting the top reviewer objections.

Phase 1A: Model Complexity Sweep (120 exp.)
  5 MLP sizes × 4 algos × 2 datasets × 3 seeds
  Answers: "Is algorithm convergence an artifact of model simplicity?"

Phase 3A: CDC Diabetes Full Comparison (35 exp.)
  7 algos × 5 seeds on 253K-sample population-scale dataset
  Answers: "Are results representative at population scale?"

Phase 3B: PTB-XL Site Granularity (36 exp.)
  4 K-values × 3 algos × 3 seeds
  Answers: "Does the result hold at different HDAB participation levels?"

Phase 5A: Extended Personalization Baselines (45 exp.)
  3 additional algos × 3 datasets × 5 seeds
  Answers: "Do other personalization methods also converge to FedAvg?"

Usage:
  Quick validation (~24 exp., ~5-10 min):
    cd fl-ehds-framework && python -m benchmarks.run_reviewer_defense --quick

  Full run (~236 exp., ~6-9 hours):
    cd fl-ehds-framework && python -m benchmarks.run_reviewer_defense --full
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
from terminal.training.models import HealthcareMLP
from terminal.fl_trainer import FederatedTrainer
import terminal.training.federated as _fed_module

OUTPUT_DIR = FRAMEWORK_DIR / "benchmarks" / "paper_results_tabular"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_FILE = "checkpoint_reviewer_defense.json"

# ============================================================================
# CONFIGURATION
# ============================================================================

# Full-mode dataset configs (quick mode overrides num_rounds)
DATASET_CONFIGS = {
    "CV": {
        "input_dim": 11, "num_classes": 2, "num_clients": 5,
        "num_rounds": 25, "local_epochs": 3, "learning_rate": 0.01,
        "batch_size": 64, "mu": 0.1,
    },
    "HD": {
        "input_dim": 13, "num_classes": 2, "num_clients": 4,
        "num_rounds": 20, "local_epochs": 3, "learning_rate": 0.01,
        "batch_size": 64, "mu": 0.1,
    },
    "PX": {
        "input_dim": 9, "num_classes": 5, "num_clients": 5,
        "num_rounds": 30, "local_epochs": 3, "learning_rate": 0.005,
        "batch_size": 64, "mu": 0.1,
    },
    "BC": {
        "input_dim": 30, "num_classes": 2, "num_clients": 3,
        "num_rounds": 20, "local_epochs": 3, "learning_rate": 0.01,
        "batch_size": 32, "mu": 0.1,
    },
    "CDC": {
        "input_dim": 21, "num_classes": 2, "num_clients": 5,
        "num_rounds": 20, "local_epochs": 2, "learning_rate": 0.01,
        "batch_size": 128, "mu": 0.1,
    },
}

# Model sizes for Phase 1A (param counts approximate for CV input_dim=11)
MODEL_SIZES = {
    "S":   [64, 32],                # ~3K params (baseline)
    "M":   [128, 64, 32],           # ~12K params
    "L":   [256, 128, 64],          # ~44K params
    "XL":  [512, 256, 128],         # ~170K params
    "XXL": [1024, 512, 256, 128],   # ~660K params
}

SEEDS_3 = [42, 123, 456]
SEEDS_5 = [42, 123, 456, 789, 999]
ALGORITHMS_MAIN = ["FedAvg", "FedProx", "Ditto", "FedLC", "FedExP", "FedLESAM", "HPFL"]

# ============================================================================
# LOGGING
# ============================================================================

def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

# ============================================================================
# CHECKPOINT (atomic save with backup)
# ============================================================================

_checkpoint_data: Dict[str, Any] = {}

def save_checkpoint():
    _checkpoint_data["metadata"]["last_save"] = datetime.now().isoformat()
    path = OUTPUT_DIR / CHECKPOINT_FILE
    bak = OUTPUT_DIR / (CHECKPOINT_FILE + ".bak")
    fd, tmp = tempfile.mkstemp(dir=str(OUTPUT_DIR), prefix=".rdef_", suffix=".tmp")
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

def load_checkpoint():
    for p in [OUTPUT_DIR / CHECKPOINT_FILE, OUTPUT_DIR / (CHECKPOINT_FILE + ".bak")]:
        if p.exists():
            try:
                with open(p) as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                continue
    return None

# ============================================================================
# GRACEFUL SHUTDOWN
# ============================================================================

_shutdown = False

def _signal_handler(signum, frame):
    global _shutdown
    if _shutdown:
        sys.exit(1)
    _shutdown = True
    log(f"INTERRUPT — saving checkpoint...")
    save_checkpoint()
    sys.exit(0)

signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)

# ============================================================================
# MODEL SIZE MONKEY-PATCH
# ============================================================================

_original_MLP = HealthcareMLP
_current_hidden_dims: Optional[List[int]] = None

def _patch_model_dims(hidden_dims: List[int]):
    """Monkey-patch HealthcareMLP to use custom hidden_dims."""
    global _current_hidden_dims
    _current_hidden_dims = list(hidden_dims)

    _dims = list(hidden_dims)

    class _CustomMLP(_original_MLP):
        def __init__(self, input_dim=10, hidden_dims=None, num_classes=2):
            super().__init__(input_dim=input_dim, hidden_dims=_dims, num_classes=num_classes)

    _CustomMLP.__name__ = f"HealthcareMLP_{'x'.join(map(str, _dims))}"
    _CustomMLP.__qualname__ = _CustomMLP.__name__

    # Patch in both modules where HealthcareMLP is used
    import terminal.training.models as _m
    _m.HealthcareMLP = _CustomMLP
    _fed_module.HealthcareMLP = _CustomMLP

def _unpatch_model_dims():
    """Restore original HealthcareMLP."""
    global _current_hidden_dims
    _current_hidden_dims = None
    import terminal.training.models as _m
    _m.HealthcareMLP = _original_MLP
    _fed_module.HealthcareMLP = _original_MLP

def _count_params(input_dim, hidden_dims, num_classes):
    """Count parameters for a given MLP architecture."""
    total = 0
    prev = input_dim
    for h in hidden_dims:
        total += prev * h + h  # Linear weight + bias
        prev = h
    total += prev * num_classes + num_classes  # Output layer
    return total

# ============================================================================
# DATA LOADING (cached)
# ============================================================================

_data_cache = {}

def load_dataset(ds_key, seed, num_clients=None):
    """Load dataset with caching. num_clients overrides default (for Phase 3B)."""
    cfg = DATASET_CONFIGS[ds_key]
    nc = num_clients or cfg["num_clients"]
    cache_key = f"{ds_key}_s{seed}_nc{nc}"
    if cache_key in _data_cache:
        cached = _data_cache[cache_key]
        return {c: (X.copy(), y.copy()) for c, (X, y) in cached[0].items()}, cached[1], cached[2]

    if ds_key == "CV":
        from data.cardiovascular_loader import load_cardiovascular_data
        train, test, meta = load_cardiovascular_data(num_clients=nc, is_iid=False, alpha=0.5, seed=seed)
    elif ds_key == "HD":
        from data.heart_disease_loader import load_heart_disease_data
        train, test, meta = load_heart_disease_data(num_clients=nc, partition_by_hospital=True, seed=seed)
    elif ds_key == "PX":
        from data.ptbxl_loader import load_ptbxl_data
        train, test, meta = load_ptbxl_data(num_clients=nc, partition_by_site=True, seed=seed)
    elif ds_key == "BC":
        from data.breast_cancer_loader import load_breast_cancer_data
        train, test, meta = load_breast_cancer_data(num_clients=nc, is_iid=False, alpha=0.5, seed=seed)
    elif ds_key == "CDC":
        from data.cdc_diabetes_loader import load_cdc_diabetes_data
        train, test, meta = load_cdc_diabetes_data(num_clients=nc, is_iid=False, alpha=0.5, seed=seed)
    else:
        raise ValueError(f"Unknown dataset: {ds_key}")

    _data_cache[cache_key] = (train, test, meta)
    return {c: (X.copy(), y.copy()) for c, (X, y) in train.items()}, test, meta

# ============================================================================
# FL RUNNER
# ============================================================================

def run_fl(algo, ds_key, seed, hidden_dims=None, num_clients=None, num_rounds_override=None):
    """Run a single FL experiment. Returns result dict."""
    cfg = DATASET_CONFIGS[ds_key]
    nc = num_clients or cfg["num_clients"]
    nr = num_rounds_override or cfg["num_rounds"]
    t0 = time.time()

    # Apply model patch if needed
    if hidden_dims is not None and hidden_dims != [64, 32]:
        _patch_model_dims(hidden_dims)
    else:
        _unpatch_model_dims()

    try:
        train, test, meta = load_dataset(ds_key, seed, num_clients=nc)

        trainer = FederatedTrainer(
            num_clients=nc, algorithm=algo,
            local_epochs=cfg["local_epochs"], batch_size=cfg["batch_size"],
            learning_rate=cfg["learning_rate"], mu=cfg["mu"], seed=seed,
            external_data=train, external_test_data=test,
            input_dim=cfg["input_dim"], num_classes=cfg["num_classes"],
        )

        # Log param count on first run
        n_params = sum(p.numel() for p in trainer.global_model.parameters())

        history = []
        for r in range(nr):
            if _shutdown:
                break
            res = trainer.train_round(r)
            history.append({"round": r + 1, "accuracy": getattr(res, "global_acc", 0)})

        final = history[-1] if history else {}
        return {
            "algorithm": algo, "dataset": ds_key, "seed": seed,
            "hidden_dims": hidden_dims or [64, 32],
            "num_clients": nc, "num_rounds": nr,
            "num_params": n_params,
            "final_accuracy": final.get("accuracy", 0),
            "runtime_seconds": round(time.time() - t0, 1),
        }
    finally:
        _unpatch_model_dims()

# ============================================================================
# EXPERIMENT BUILDERS
# ============================================================================

def build_phase_1a(quick=False):
    """Phase 1A: Model Complexity Sweep."""
    exps = []
    if quick:
        sizes = ["M", "L", "XL"]
        algos = ["FedAvg", "Ditto", "HPFL"]
        datasets = ["CV"]
        seeds = [42]
    else:
        sizes = ["S", "M", "L", "XL", "XXL"]
        algos = ["FedAvg", "Ditto", "HPFL", "FedProx"]
        datasets = ["CV", "HD"]
        seeds = SEEDS_3

    for sz_name in sizes:
        dims = MODEL_SIZES[sz_name]
        for algo in algos:
            for ds in datasets:
                for seed in seeds:
                    exps.append({
                        "key": f"1A_{algo}_{sz_name}_{ds}_s{seed}",
                        "phase": "1A",
                        "algo": algo, "ds": ds, "seed": seed,
                        "hidden_dims": dims, "sz_name": sz_name,
                    })
    return exps

def build_phase_3a(quick=False):
    """Phase 3A: CDC Diabetes Full Comparison."""
    exps = []
    if quick:
        algos = ["FedAvg", "Ditto", "HPFL"]
        seeds = [42]
    else:
        algos = ALGORITHMS_MAIN
        seeds = SEEDS_5

    for algo in algos:
        for seed in seeds:
            exps.append({
                "key": f"3A_{algo}_CDC_s{seed}",
                "phase": "3A",
                "algo": algo, "ds": "CDC", "seed": seed,
            })
    return exps

def build_phase_3b(quick=False):
    """Phase 3B: PTB-XL Site Granularity."""
    exps = []
    if quick:
        k_values = [3, 10, 20]
        algos = ["FedAvg", "Ditto", "HPFL"]
        seeds = [42]
    else:
        k_values = [3, 5, 10, 20]
        algos = ["FedAvg", "Ditto", "HPFL"]
        seeds = SEEDS_3

    for k in k_values:
        for algo in algos:
            for seed in seeds:
                exps.append({
                    "key": f"3B_{algo}_PX_K{k}_s{seed}",
                    "phase": "3B",
                    "algo": algo, "ds": "PX", "seed": seed,
                    "num_clients": k,
                })
    return exps

def build_phase_5a(quick=False):
    """Phase 5A: Extended Personalization Baselines."""
    exps = []
    # Algos missing from main table but available in FederatedTrainer
    extended_algos = ["Per-FedAvg", "FedDyn", "FedSAM"]

    if quick:
        algos = extended_algos
        datasets = ["CV"]
        seeds = [42]
    else:
        algos = extended_algos
        datasets = ["PX", "CV", "BC"]
        seeds = SEEDS_5

    for algo in algos:
        for ds in datasets:
            for seed in seeds:
                exps.append({
                    "key": f"5A_{algo}_{ds}_s{seed}",
                    "phase": "5A",
                    "algo": algo, "ds": ds, "seed": seed,
                })
    return exps

def build_all_experiments(quick=False):
    """Build complete experiment list."""
    exps = []
    exps.extend(build_phase_1a(quick))
    exps.extend(build_phase_3a(quick))
    exps.extend(build_phase_3b(quick))
    exps.extend(build_phase_5a(quick))
    return exps

# ============================================================================
# SUMMARY PRINTERS
# ============================================================================

def print_phase_1a_summary(completed):
    """Phase 1A: Does convergence persist at larger models?"""
    log("\n--- Phase 1A: Model Complexity Sweep ---")
    # Collect all available datasets and model sizes from results
    datasets_found = sorted(set(v.get("dataset", "") for k, v in completed.items()
                                if k.startswith("1A_") and "error" not in v))
    sizes_found = sorted(set(tuple(v.get("hidden_dims", [])) for k, v in completed.items()
                             if k.startswith("1A_") and "error" not in v),
                         key=lambda x: _count_params(11, list(x), 2))
    sizes_found = [list(t) for t in sizes_found]

    for ds in datasets_found:
        idim = DATASET_CONFIGS.get(ds, {}).get("input_dim", 11)
        ncls = DATASET_CONFIGS.get(ds, {}).get("num_classes", 2)
        log(f"\n  Dataset: {ds}")
        header = f"  {'Size':<6} {'Params':>8}"
        algos_in = sorted(set(v.get("algorithm", "") for k, v in completed.items()
                              if k.startswith("1A_") and v.get("dataset") == ds and "error" not in v))
        for a in algos_in:
            header += f" | {a:>8}"
        log(header)
        log("  " + "-" * len(header))

        for dims in sizes_found:
            if not isinstance(dims, list):
                continue
            sz_label = next((n for n, d in MODEL_SIZES.items() if d == dims), "?")
            nparams = _count_params(idim, dims, ncls)
            row = f"  {sz_label:<6} {nparams:>8,}"
            for a in algos_in:
                accs = [v["final_accuracy"] * 100 for k, v in completed.items()
                        if k.startswith("1A_") and v.get("dataset") == ds
                        and v.get("algorithm") == a
                        and v.get("hidden_dims") == dims
                        and "error" not in v]
                if accs:
                    row += f" | {np.mean(accs):>7.1f}%"
                else:
                    row += f" |     ---"
            log(row)

        # Convergence check: do non-Ditto/HPFL algorithms still cluster?
        log(f"\n  Convergence analysis ({ds}):")
        for dims in sizes_found:
            if not isinstance(dims, list):
                continue
            sz_label = next((n for n, d in MODEL_SIZES.items() if d == dims), "?")
            global_accs = []
            personal_accs = []
            for k, v in completed.items():
                if (k.startswith("1A_") and v.get("dataset") == ds
                        and v.get("hidden_dims") == dims and "error" not in v):
                    a = v["algorithm"]
                    acc = v["final_accuracy"] * 100
                    if a in ["Ditto", "HPFL"]:
                        personal_accs.append(acc)
                    else:
                        global_accs.append(acc)
            if global_accs and personal_accs:
                g_mean = np.mean(global_accs)
                p_mean = np.mean(personal_accs)
                gap = p_mean - g_mean
                g_range = max(global_accs) - min(global_accs)
                log(f"    {sz_label}: Global={g_mean:.1f}% (range {g_range:.1f}pp)  "
                    f"Personal={p_mean:.1f}%  Gap={gap:+.1f}pp")

def print_phase_3a_summary(completed):
    """Phase 3A: CDC Diabetes results."""
    log("\n--- Phase 3A: CDC Diabetes (253K samples) ---")
    log(f"  {'Algorithm':<12} | {'Accuracy':>10} | {'Std':>6}")
    log("  " + "-" * 38)
    for algo in ALGORITHMS_MAIN:
        accs = [v["final_accuracy"] * 100 for k, v in completed.items()
                if k.startswith("3A_") and v.get("algorithm") == algo and "error" not in v]
        if accs:
            log(f"  {algo:<12} | {np.mean(accs):>9.1f}% | {np.std(accs):>5.1f}")

    # Check convergence
    global_algos = ["FedAvg", "FedProx", "FedLC", "FedExP", "FedLESAM"]
    personal_algos = ["Ditto", "HPFL"]
    g_accs = [v["final_accuracy"] * 100 for k, v in completed.items()
              if k.startswith("3A_") and v.get("algorithm") in global_algos and "error" not in v]
    p_accs = [v["final_accuracy"] * 100 for k, v in completed.items()
              if k.startswith("3A_") and v.get("algorithm") in personal_algos and "error" not in v]
    if g_accs and p_accs:
        log(f"\n  Global mean: {np.mean(g_accs):.1f}% (range {max(g_accs)-min(g_accs):.1f}pp)")
        log(f"  Personal mean: {np.mean(p_accs):.1f}%")
        log(f"  Gap: {np.mean(p_accs)-np.mean(g_accs):+.1f}pp")

def print_phase_3b_summary(completed):
    """Phase 3B: PTB-XL site granularity."""
    log("\n--- Phase 3B: PTB-XL Site Granularity ---")
    k_values = sorted(set(v.get("num_clients", 0) for k, v in completed.items()
                          if k.startswith("3B_") and "error" not in v))
    log(f"  {'K':>4} | {'FedAvg':>8} | {'Ditto':>8} | {'HPFL':>8} | {'Gap(D-FA)':>10}")
    log("  " + "-" * 52)
    for k in k_values:
        accs = {}
        for algo in ["FedAvg", "Ditto", "HPFL"]:
            vals = [v["final_accuracy"] * 100 for key, v in completed.items()
                    if key.startswith("3B_") and v.get("algorithm") == algo
                    and v.get("num_clients") == k and "error" not in v]
            accs[algo] = np.mean(vals) if vals else 0
        gap = accs["Ditto"] - accs["FedAvg"]
        log(f"  {k:>4} | {accs['FedAvg']:>7.1f}% | {accs['Ditto']:>7.1f}% | "
            f"{accs['HPFL']:>7.1f}% | {gap:>+9.1f}pp")

def print_phase_5a_summary(completed):
    """Phase 5A: Extended baselines — do they converge to FedAvg?"""
    log("\n--- Phase 5A: Extended Personalization Baselines ---")
    datasets_found = sorted(set(v.get("dataset", "") for k, v in completed.items()
                                if k.startswith("5A_") and "error" not in v))
    extended_algos = ["Per-FedAvg", "FedDyn", "FedSAM"]

    for ds in datasets_found:
        log(f"\n  Dataset: {ds}")
        log(f"  {'Algorithm':<12} | {'Accuracy':>10} | {'Std':>6} | {'vs FedAvg main table'}")
        log("  " + "-" * 55)
        for algo in extended_algos:
            accs = [v["final_accuracy"] * 100 for k, v in completed.items()
                    if k.startswith("5A_") and v.get("algorithm") == algo
                    and v.get("dataset") == ds and "error" not in v]
            if accs:
                log(f"  {algo:<12} | {np.mean(accs):>9.1f}% | {np.std(accs):>5.1f} | (compare to Table III)")

# ============================================================================
# MAIN
# ============================================================================

def main():
    global _checkpoint_data, _shutdown

    # Parse mode
    if "--quick" in sys.argv:
        quick = True
    elif "--full" in sys.argv:
        quick = False
    else:
        print("Usage: python -m benchmarks.run_reviewer_defense --quick|--full")
        sys.exit(1)

    mode_label = "QUICK" if quick else "FULL"
    experiments = build_all_experiments(quick=quick)
    total = len(experiments)

    # Verify no duplicate keys
    keys = [e["key"] for e in experiments]
    dupes = [k for k in set(keys) if keys.count(k) > 1]
    if dupes:
        log(f"FATAL: {len(dupes)} duplicate experiment keys: {dupes[:5]}")
        sys.exit(1)

    # Count per phase
    phase_counts = {}
    for e in experiments:
        phase_counts[e["phase"]] = phase_counts.get(e["phase"], 0) + 1

    log("=" * 65)
    log(f"REVIEWER DEFENSE EXPERIMENTS — {mode_label} MODE")
    log(f"Total: {total} experiments")
    for ph, cnt in sorted(phase_counts.items()):
        log(f"  Phase {ph}: {cnt} experiments")
    if quick:
        log("  (num_rounds reduced to 10 for quick validation)")
    log("=" * 65)

    # Log model sizes
    log("\nModel sizes (Phase 1A):")
    for name, dims in MODEL_SIZES.items():
        for ds in ["CV", "HD"]:
            cfg = DATASET_CONFIGS[ds]
            npar = _count_params(cfg["input_dim"], dims, cfg["num_classes"])
            log(f"  {name}: {dims} → {npar:,} params ({ds})")
        break  # Just show CV

    # Load or init checkpoint
    _checkpoint_data = load_checkpoint() or {
        "completed": {},
        "metadata": {
            "total_experiments": total,
            "mode": mode_label,
            "start_time": datetime.now().isoformat(),
        },
    }
    completed = _checkpoint_data.setdefault("completed", {})
    done = len(completed)
    if done > 0:
        log(f"\nResumed: {done} previously completed")

    t_start = time.time()

    for exp in experiments:
        if _shutdown:
            break
        key = exp["key"]
        if key in completed:
            continue

        log(f"\n[{done+1}/{total}] {key} ...")
        try:
            result = run_fl(
                algo=exp["algo"],
                ds_key=exp["ds"],
                seed=exp["seed"],
                hidden_dims=exp.get("hidden_dims"),
                num_clients=exp.get("num_clients"),
                num_rounds_override=10 if quick else None,
            )
            result["key"] = key
            result["phase"] = exp["phase"]
            if "sz_name" in exp:
                result["sz_name"] = exp["sz_name"]
            completed[key] = result
            done += 1
            save_checkpoint()
            log(f"  -> acc={result['final_accuracy']*100:.1f}%  "
                f"params={result.get('num_params', '?'):,}  ({result['runtime_seconds']:.0f}s)")
        except Exception as e:
            log(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            completed[key] = {"key": key, "phase": exp["phase"], "error": str(e)}
            done += 1
            save_checkpoint()
        gc.collect()

    elapsed = time.time() - t_start
    _checkpoint_data["metadata"]["end_time"] = datetime.now().isoformat()
    _checkpoint_data["metadata"]["elapsed_seconds"] = round(elapsed, 1)
    save_checkpoint()

    # ========================================================================
    # SUMMARY
    # ========================================================================
    log("\n" + "=" * 65)
    log(f"REVIEWER DEFENSE {mode_label} COMPLETE: {done}/{total} in {elapsed:.0f}s")
    log("=" * 65)

    errors = [k for k, v in completed.items() if "error" in v]
    if errors:
        log(f"\nERRORS ({len(errors)}):")
        for k in errors[:10]:
            log(f"  {k}: {completed[k]['error']}")
        if len(errors) > 10:
            log(f"  ... and {len(errors) - 10} more")

    print_phase_1a_summary(completed)
    print_phase_3a_summary(completed)
    print_phase_3b_summary(completed)
    print_phase_5a_summary(completed)

    log(f"\nCheckpoint: {OUTPUT_DIR / CHECKPOINT_FILE}")

if __name__ == "__main__":
    main()

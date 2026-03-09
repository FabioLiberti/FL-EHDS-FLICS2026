#!/usr/bin/env python3
"""
Cascade 2 — Ditto + HPFL TabNet completion
============================================
Adds the missing Ditto experiments to all tabular phases,
plus HPFL on TabNet (Phase 3A).

8 experiments, ~5 min on M3 Air.

Usage:
  cd fl-ehds-framework && python -m benchmarks.run_cascade2_ditto_completion
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

FRAMEWORK_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(FRAMEWORK_DIR))

from benchmarks.run_reviewer_cascade_2 import (
    OUTPUT_DIR,
    CHECKPOINT_FILE,
    _run_phase_2a,
    _run_phase_1b,
    _run_phase_3a,
    _cleanup_gpu,
    load_checkpoint,
    save_checkpoint,
    log,
)
import benchmarks.run_reviewer_cascade_2 as _rc2

# ============================================================================
# Experiments to add
# ============================================================================

EXPS = [
    # Phase 2A: Ditto governance overhead
    {
        "key": "2A_Ditto_no_governance_s42",
        "phase": "2A",
        "algo": "Ditto",
        "seed": 42,
        "governance": False,
        "dataset": "Cardiovascular",
    },
    {
        "key": "2A_Ditto_with_governance_s42",
        "phase": "2A",
        "algo": "Ditto",
        "seed": 42,
        "governance": True,
        "dataset": "Cardiovascular",
    },
    # Phase 1B: Ditto noise stress
    {
        "key": "1B_Cardiovascular_Ditto_label_0p1_s42",
        "phase": "1B",
        "algo": "Ditto",
        "seed": 42,
        "dataset": "Cardiovascular",
        "noise_type": "label",
        "noise_level": 0.1,
    },
    {
        "key": "1B_Cardiovascular_Ditto_label_0p3_s42",
        "phase": "1B",
        "algo": "Ditto",
        "seed": 42,
        "dataset": "Cardiovascular",
        "noise_type": "label",
        "noise_level": 0.3,
    },
    {
        "key": "1B_Cardiovascular_Ditto_feature_0p2_s42",
        "phase": "1B",
        "algo": "Ditto",
        "seed": 42,
        "dataset": "Cardiovascular",
        "noise_type": "feature",
        "noise_level": 0.2,
    },
    {
        "key": "1B_Cardiovascular_Ditto_missing_0p2_s42",
        "phase": "1B",
        "algo": "Ditto",
        "seed": 42,
        "dataset": "Cardiovascular",
        "noise_type": "missing",
        "noise_level": 0.2,
    },
    # Phase 3A: Ditto + HPFL on TabNet
    {
        "key": "3A_Cardiovascular_Ditto_TabNet_s42",
        "phase": "3A",
        "algo": "Ditto",
        "seed": 42,
        "dataset": "Cardiovascular",
    },
    {
        "key": "3A_Cardiovascular_HPFL_TabNet_s42",
        "phase": "3A",
        "algo": "HPFL",
        "seed": 42,
        "dataset": "Cardiovascular",
    },
]

RUNNERS = {
    "2A": _run_phase_2a,
    "1B": _run_phase_1b,
    "3A": _run_phase_3a,
}


def main():
    log("=" * 60)
    log("CASCADE 2 — DITTO + HPFL TABNET COMPLETION (8 experiments)")
    log("=" * 60)

    data = load_checkpoint() or {
        "completed": {},
        "metadata": {
            "total_experiments": 17,
            "mode": "QUICK",
            "start_time": datetime.now().isoformat(),
        },
    }
    _rc2._checkpoint_data = data
    completed = data.setdefault("completed", {})

    t_start = time.time()
    done = 0
    errors = 0
    total = len(EXPS)

    for exp in EXPS:
        key = exp["key"]
        if key in completed and "error" not in completed[key]:
            log(f"  [{key}] already completed, skipping")
            done += 1
            continue

        # Remove previous error if any
        if key in completed and "error" in completed[key]:
            del completed[key]

        done += 1
        log(f"\n[{done}/{total}] {key} ...")
        try:
            runner = RUNNERS[exp["phase"]]
            result = runner(exp, quick=True)
            result["key"] = key
            result["phase"] = exp["phase"]
            completed[key] = result
            save_checkpoint()

            acc_key = "best_accuracy" if "best_accuracy" in result else "final_accuracy"
            acc = result.get(acc_key, 0)
            rt = result.get("runtime_seconds", result.get("total_runtime_s", 0))
            log(f"  -> {acc*100:.1f}%  ({rt:.0f}s)")

        except Exception as e:
            log(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            completed[key] = {"key": key, "phase": exp["phase"], "error": str(e)}
            save_checkpoint()
            errors += 1

        _cleanup_gpu()

    elapsed = time.time() - t_start
    data["metadata"]["last_save"] = datetime.now().isoformat()
    data["metadata"]["total_experiments"] = len(completed)
    save_checkpoint()

    # ================================================================
    # SUMMARY
    # ================================================================
    log(f"\n{'=' * 60}")
    log(f"DONE: {done}/{total} in {elapsed:.0f}s ({errors} errors)")
    log(f"{'=' * 60}")

    if errors:
        for k, v in completed.items():
            if "error" in v:
                log(f"  ERROR: {k}: {v['error']}")

    # Full 3-algorithm comparison tables
    log("\n--- Phase 2A: Governance Overhead (3 algorithms) ---")
    log(f"  {'Algo':<8} | {'No-Gov (s/r)':>13} | {'With-Gov (s/r)':>15} | {'Overhead':>9} | {'Acc':>6}")
    log("  " + "-" * 58)
    for algo in ["FedAvg", "Ditto", "HPFL"]:
        ng = completed.get(f"2A_{algo}_no_governance_s42", {})
        wg = completed.get(f"2A_{algo}_with_governance_s42", {})
        if "error" not in ng and "error" not in wg and ng and wg:
            ng_t = ng["mean_round_time_s"]
            wg_t = wg["mean_round_time_s"]
            overhead = ((wg_t - ng_t) / ng_t * 100) if ng_t > 0 else 0
            log(f"  {algo:<8} | {ng_t:>12.4f}s | {wg_t:>14.4f}s | {overhead:>+8.1f}% | {wg['final_accuracy']*100:>5.1f}%")

    log("\n--- Phase 1B: Data Quality Stress (3 algorithms) ---")
    log(f"  {'Noise':<8} | {'Level':>5} | {'FedAvg':>8} | {'Ditto':>8} | {'HPFL':>8}")
    log("  " + "-" * 48)
    noise_configs = [
        ("label", 0.1), ("label", 0.3), ("feature", 0.2), ("missing", 0.2),
    ]
    for nt, nl in noise_configs:
        nl_str = str(nl).replace(".", "p")
        row = {}
        for algo in ["FedAvg", "Ditto", "HPFL"]:
            k = f"1B_Cardiovascular_{algo}_{nt}_{nl_str}_s42"
            v = completed.get(k, {})
            if "error" not in v and v:
                row[algo] = f"{v['best_accuracy']*100:>6.1f}%"
            else:
                row[algo] = "   N/A "
        log(f"  {nt:<8} | {nl:>5.2f} | {row['FedAvg']:>8} | {row['Ditto']:>8} | {row['HPFL']:>8}")

    log("\n--- Phase 3A: TabNet (3 algorithms) ---")
    for algo in ["FedAvg", "Ditto", "HPFL"]:
        k = f"3A_Cardiovascular_{algo}_TabNet_s42"
        v = completed.get(k, {})
        if "error" not in v and v:
            log(f"  {algo:<8}: best={v['best_accuracy']*100:.1f}%  "
                f"params={v.get('num_params', '?'):,}  {v.get('runtime_seconds', 0):.0f}s")

    total_tabular = sum(1 for k, v in completed.items() if "error" not in v
                        and not k.startswith("5B_") and not k.startswith("4A_"))
    log(f"\nTotal tabular experiments completed: {total_tabular}")
    log(f"Checkpoint: {OUTPUT_DIR / CHECKPOINT_FILE}")


if __name__ == "__main__":
    main()

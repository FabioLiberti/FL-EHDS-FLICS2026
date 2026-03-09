#!/usr/bin/env python3
"""
Cascade 2 — Tabular-only completion
====================================
Runs ONLY the non-imaging experiments that remain from quick mode:
  1. 2A_FedAvg_with_governance_s42   (governance overhead)
  2. 2A_HPFL_with_governance_s42     (governance overhead)
  3. 3A_Cardiovascular_FedAvg_TabNet  (TabNet architecture)

Writes results back into the same checkpoint_reviewer_cascade_2.json.

Usage:
  cd fl-ehds-framework && python -m benchmarks.run_cascade2_tabular_only
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
    _checkpoint_data,
    _run_phase_2a,
    _run_phase_3a,
    _cleanup_gpu,
    load_checkpoint,
    save_checkpoint,
    log,
)
import benchmarks.run_reviewer_cascade_2 as _rc2

# The 3 non-imaging experiments to complete
TABULAR_EXPS = [
    {
        "key": "2A_FedAvg_with_governance_s42",
        "phase": "2A",
        "algo": "FedAvg",
        "seed": 42,
        "governance": True,
        "dataset": "Cardiovascular",
    },
    {
        "key": "2A_HPFL_with_governance_s42",
        "phase": "2A",
        "algo": "HPFL",
        "seed": 42,
        "governance": True,
        "dataset": "Cardiovascular",
    },
    {
        "key": "3A_Cardiovascular_FedAvg_TabNet_s42",
        "phase": "3A",
        "algo": "FedAvg",
        "seed": 42,
        "dataset": "Cardiovascular",
    },
]


def main():
    log("=" * 60)
    log("CASCADE 2 — TABULAR-ONLY COMPLETION (3 experiments)")
    log("=" * 60)

    # Load existing checkpoint
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

    # Remove previous errors for these keys
    for exp in TABULAR_EXPS:
        if exp["key"] in completed and "error" in completed[exp["key"]]:
            log(f"  Removing previous error: {exp['key']}")
            del completed[exp["key"]]

    t_start = time.time()
    done = 0
    errors = 0

    for exp in TABULAR_EXPS:
        key = exp["key"]
        if key in completed and "error" not in completed[key]:
            log(f"  [{key}] already completed, skipping")
            done += 1
            continue

        log(f"\n[{done+1}/3] {key} ...")
        try:
            if exp["phase"] == "2A":
                result = _run_phase_2a(exp, quick=True)
            elif exp["phase"] == "3A":
                result = _run_phase_3a(exp, quick=True)
            else:
                raise ValueError(f"Unknown phase: {exp['phase']}")

            result["key"] = key
            result["phase"] = exp["phase"]
            completed[key] = result
            save_checkpoint()
            done += 1

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
    save_checkpoint()

    log(f"\n{'=' * 60}")
    log(f"DONE: {done}/3 OK, {errors} errors, {elapsed:.0f}s total")
    log(f"{'=' * 60}")

    # Quick recap of all non-imaging results
    log("\n--- Phase 2A: Governance Overhead (all results) ---")
    for algo in ["FedAvg", "HPFL"]:
        ng_key = f"2A_{algo}_no_governance_s42"
        wg_key = f"2A_{algo}_with_governance_s42"
        ng = completed.get(ng_key, {})
        wg = completed.get(wg_key, {})
        if "error" not in ng and "error" not in wg:
            ng_t = ng.get("mean_round_time_s", 0)
            wg_t = wg.get("mean_round_time_s", 0)
            overhead = ((wg_t - ng_t) / ng_t * 100) if ng_t > 0 else 0
            log(f"  {algo:<8}: no-gov {ng_t:.4f}s/round, with-gov {wg_t:.4f}s/round → overhead {overhead:+.1f}%")
        elif "error" in wg:
            log(f"  {algo:<8}: with_governance ERRORED: {wg.get('error', '?')}")

    log("\n--- Phase 3A: TabNet (result) ---")
    tab_key = "3A_Cardiovascular_FedAvg_TabNet_s42"
    tab = completed.get(tab_key, {})
    if "error" not in tab and tab:
        log(f"  FedAvg TabNet: best={tab.get('best_accuracy', 0)*100:.1f}%, "
            f"params={tab.get('num_params', '?'):,}, {tab.get('runtime_seconds', 0):.0f}s")
    elif "error" in tab:
        log(f"  TabNet ERRORED: {tab.get('error', '?')}")

    log(f"\nCheckpoint: {OUTPUT_DIR / CHECKPOINT_FILE}")


if __name__ == "__main__":
    main()

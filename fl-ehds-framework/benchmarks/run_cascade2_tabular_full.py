#!/usr/bin/env python3
"""
Cascade 2 — Full Tabular Only (phases 2A + 1B + 3A)
=====================================================
Runs all tabular experiments in full mode (3 seeds, 30 rounds)
while skipping imaging phases (5B, 4A).

Experiments:
  Phase 2A: 3 algos × 2 conditions × 3 seeds     =  18 exp (~6 min)
  Phase 1B: 3 algos × 10 noise × 2 datasets × 3s = 180 exp (~1.5 h)
  Phase 3A: 3 algos × 3 datasets × 5 seeds        =  45 exp (~30 min)
  Total: 243 experiments, ~2-3 hours on M3 Air

Results go into the same checkpoint_reviewer_cascade_2.json.

Usage:
  cd fl-ehds-framework && python -m benchmarks.run_cascade2_tabular_full
"""

import sys
import time
from datetime import datetime
from pathlib import Path

FRAMEWORK_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(FRAMEWORK_DIR))

from benchmarks.run_reviewer_cascade_2 import (
    OUTPUT_DIR,
    CHECKPOINT_FILE,
    PHASE_DESCRIPTIONS,
    _build_phase_2a,
    _build_phase_1b,
    _build_phase_3a,
    _run_phase_2a,
    _run_phase_1b,
    _run_phase_3a,
    _cleanup_gpu,
    load_checkpoint,
    save_checkpoint,
    log,
)
import benchmarks.run_reviewer_cascade_2 as _rc2

TABULAR_PHASES = ["2A", "1B", "3A"]

BUILDERS = {
    "2A": _build_phase_2a,
    "1B": _build_phase_1b,
    "3A": _build_phase_3a,
}

RUNNERS = {
    "2A": _run_phase_2a,
    "1B": _run_phase_1b,
    "3A": _run_phase_3a,
}


def main():
    quick = False  # FULL mode

    # Build all tabular experiments
    experiments = []
    for phase in TABULAR_PHASES:
        experiments.extend(BUILDERS[phase](quick))

    total = len(experiments)

    # Phase counts
    phase_counts = {}
    for e in experiments:
        phase_counts[e["phase"]] = phase_counts.get(e["phase"], 0) + 1

    log("=" * 65)
    log("CASCADE 2 — FULL TABULAR MODE (skip imaging)")
    log(f"Total experiments: {total}")
    for ph in TABULAR_PHASES:
        log(f"  Phase {ph}: {phase_counts.get(ph, 0):>3} exp — {PHASE_DESCRIPTIONS[ph]}")
    log("=" * 65)

    # Verify no duplicate keys
    keys = [e["key"] for e in experiments]
    dupes = [k for k in set(keys) if keys.count(k) > 1]
    if dupes:
        log(f"FATAL: {len(dupes)} duplicate keys: {dupes[:5]}")
        sys.exit(1)

    # Load or init checkpoint
    data = load_checkpoint() or {
        "completed": {},
        "metadata": {
            "mode": "FULL-TABULAR",
            "start_time": datetime.now().isoformat(),
        },
    }
    _rc2._checkpoint_data = data
    completed = data.setdefault("completed", {})
    data["metadata"]["total_experiments_tabular"] = total

    already = sum(1 for e in experiments if e["key"] in completed and "error" not in completed[e["key"]])
    if already > 0:
        log(f"Resumed: {already}/{total} already completed")

    t_start = time.time()
    done = already
    current_phase = ""

    for exp in experiments:
        if _rc2._shutdown:
            break
        key = exp["key"]
        if key in completed and "error" not in completed[key]:
            continue

        # Remove previous error
        if key in completed and "error" in completed[key]:
            del completed[key]

        # Phase transition
        if exp["phase"] != current_phase:
            current_phase = exp["phase"]
            ph_total = phase_counts[current_phase]
            ph_done = sum(1 for e in experiments
                          if e["phase"] == current_phase and e["key"] in completed
                          and "error" not in completed[e["key"]])
            log(f"\n{'=' * 65}")
            log(f"PHASE {current_phase}: {PHASE_DESCRIPTIONS[current_phase]}")
            log(f"  ({ph_done}/{ph_total} done)")
            log(f"{'=' * 65}")

        done += 1
        log(f"\n[{done}/{total}] {key} ...")

        try:
            runner = RUNNERS[exp["phase"]]
            result = runner(exp, quick=False)
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

        _cleanup_gpu()

    elapsed = time.time() - t_start
    data["metadata"]["end_time_tabular"] = datetime.now().isoformat()
    data["metadata"]["elapsed_tabular_s"] = round(elapsed, 1)
    save_checkpoint()

    # ============================================================
    # FINAL SUMMARY
    # ============================================================
    total_ok = sum(1 for e in experiments if e["key"] in completed and "error" not in completed[e["key"]])
    total_err = sum(1 for e in experiments if e["key"] in completed and "error" in completed[e["key"]])
    log(f"\n{'=' * 65}")
    log(f"FULL TABULAR COMPLETE: {total_ok}/{total} OK, {total_err} errors, {elapsed:.0f}s")
    log(f"{'=' * 65}")

    if total_err:
        for e in experiments:
            v = completed.get(e["key"], {})
            if "error" in v:
                log(f"  ERROR: {e['key']}: {v['error']}")

    # Import summary printers
    from benchmarks.run_reviewer_cascade_2 import (
        print_2a_summary, print_1b_summary, print_3a_summary,
    )
    print_2a_summary(completed)
    print_1b_summary(completed)
    print_3a_summary(completed)

    log(f"\nCheckpoint: {OUTPUT_DIR / CHECKPOINT_FILE}")


if __name__ == "__main__":
    main()

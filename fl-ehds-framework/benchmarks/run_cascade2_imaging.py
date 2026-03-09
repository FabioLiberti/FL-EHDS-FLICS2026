#!/usr/bin/env python3
"""
Cascade 2 — Imaging Experiments (phases 5B + 4A)
==================================================
Runs the two imaging phases that were skipped by the tabular scripts.

Phase 5B: HPFL split-point ablation on imaging (ResNet-18)
  Quick:  1 dataset × 3 depths × 1 seed           =   3 exp  (~5 min)
  Full:   3 datasets × 3 depths × 3 seeds          =  27 exp  (~2 h)

Phase 4A: EfficientNet-B0 imaging (architecture generality)
  Quick:  1 dataset × 1 algo × 1 seed              =   1 exp  (~3 min)
  Full:   3 datasets × 3 algos × 3 seeds            =  27 exp  (~3 h)

  Total quick:   4 experiments  (~8 min on M3 Air)
  Total full:   54 experiments  (~5 h on M3 Air)

Results go into the same checkpoint_reviewer_cascade_2.json.

Usage:
  cd fl-ehds-framework

  # Quick mode (~8 min)
  python -m benchmarks.run_cascade2_imaging --quick

  # Full mode (~5 h)
  python -m benchmarks.run_cascade2_imaging --full

  # Single phase only
  python -m benchmarks.run_cascade2_imaging --quick --phase 5B
  python -m benchmarks.run_cascade2_imaging --full --phase 4A
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
    _build_phase_5b,
    _build_phase_4a,
    _run_phase_5b,
    _run_phase_4a,
    _cleanup_gpu,
    load_checkpoint,
    save_checkpoint,
    log,
    print_5b_summary,
    print_4a_summary,
)
import benchmarks.run_reviewer_cascade_2 as _rc2

IMAGING_PHASES = ["5B", "4A"]

BUILDERS = {
    "5B": _build_phase_5b,
    "4A": _build_phase_4a,
}

RUNNERS = {
    "5B": _run_phase_5b,
    "4A": _run_phase_4a,
}

SUMMARIES = {
    "5B": print_5b_summary,
    "4A": print_4a_summary,
}


def main():
    # Parse arguments
    if "--quick" in sys.argv:
        quick = True
    elif "--full" in sys.argv:
        quick = False
    else:
        print("Usage: python -m benchmarks.run_cascade2_imaging --quick|--full [--phase 5B|4A]")
        sys.exit(1)

    # Optional single-phase filter
    phase_filter = None
    if "--phase" in sys.argv:
        idx = sys.argv.index("--phase")
        if idx + 1 < len(sys.argv):
            phase_filter = sys.argv[idx + 1]
            if phase_filter not in IMAGING_PHASES:
                print(f"Unknown phase: {phase_filter}. Valid: {IMAGING_PHASES}")
                sys.exit(1)

    phases = [phase_filter] if phase_filter else IMAGING_PHASES
    mode_label = "QUICK" if quick else "FULL"

    # Build experiments
    experiments = []
    for phase in phases:
        experiments.extend(BUILDERS[phase](quick))

    total = len(experiments)

    # Phase counts
    phase_counts = {}
    for e in experiments:
        phase_counts[e["phase"]] = phase_counts.get(e["phase"], 0) + 1

    log("=" * 65)
    log(f"CASCADE 2 — IMAGING {mode_label} MODE")
    log(f"Total experiments: {total}")
    for ph in phases:
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
            "mode": f"{mode_label}-IMAGING",
            "start_time": datetime.now().isoformat(),
        },
    }
    _rc2._checkpoint_data = data
    completed = data.setdefault("completed", {})
    data["metadata"]["total_experiments_imaging"] = total

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
            result = runner(exp, quick=quick)
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
    data["metadata"]["end_time_imaging"] = datetime.now().isoformat()
    data["metadata"]["elapsed_imaging_s"] = round(elapsed, 1)
    save_checkpoint()

    # ============================================================
    # FINAL SUMMARY
    # ============================================================
    total_ok = sum(1 for e in experiments if e["key"] in completed and "error" not in completed[e["key"]])
    total_err = sum(1 for e in experiments if e["key"] in completed and "error" in completed[e["key"]])
    log(f"\n{'=' * 65}")
    log(f"IMAGING {mode_label} COMPLETE: {total_ok}/{total} OK, {total_err} errors, {elapsed:.0f}s")
    log(f"{'=' * 65}")

    if total_err:
        for e in experiments:
            v = completed.get(e["key"], {})
            if "error" in v:
                log(f"  ERROR: {e['key']}: {v['error']}")

    for ph in phases:
        SUMMARIES[ph](completed)

    log(f"\nCheckpoint: {OUTPUT_DIR / CHECKPOINT_FILE}")


if __name__ == "__main__":
    main()

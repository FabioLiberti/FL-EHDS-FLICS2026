#!/usr/bin/env python3
"""
Isolated runner for new tabular datasets (PTB-XL, Cardiovascular, Breast Cancer).

Uses a SEPARATE output directory to avoid checkpoint conflicts with the
main run_paper_experiments.py process currently running (~35h imaging).

Results are saved to benchmarks/paper_results_tabular/ and can be merged
into the main checkpoint after the imaging run completes.

Usage:
    cd fl-ehds-framework
    python -m benchmarks.run_tabular_new [--quick] [--dataset PTB_XL|Cardiovascular|Breast_Cancer]
"""

import sys
import time
import argparse
from pathlib import Path
from datetime import datetime

# Setup paths
FRAMEWORK_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(FRAMEWORK_DIR))

# Import the experiment module
import benchmarks.run_paper_experiments as rpe

# === CRITICAL: Override OUTPUT_DIR to isolated directory ===
ISOLATED_OUTPUT = FRAMEWORK_DIR / "benchmarks" / "paper_results_tabular"
rpe.OUTPUT_DIR = ISOLATED_OUTPUT

# Only the 3 new tabular datasets
NEW_TABULAR = ["PTB_XL", "Cardiovascular", "Breast_Cancer"]


def main():
    parser = argparse.ArgumentParser(description="Run new tabular experiments (isolated)")
    parser.add_argument("--quick", action="store_true", help="Quick mode: 1 seed, 5 rounds")
    parser.add_argument("--dataset", type=str, default=None,
                        choices=NEW_TABULAR, help="Run only one dataset")
    args = parser.parse_args()

    if args.quick:
        rpe.SEEDS = [42]
        rpe.TABULAR_CONFIG["num_rounds"] = 5
        rpe.TABULAR_CONFIG["local_epochs"] = 1

    ISOLATED_OUTPUT.mkdir(parents=True, exist_ok=True)

    datasets_to_run = [args.dataset] if args.dataset else NEW_TABULAR

    print("=" * 70)
    print("  FL-EHDS: New Tabular Datasets (Isolated Runner)")
    print("=" * 70)
    print(f"  Started:  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Device:   {rpe._detect_device()}")
    print(f"  Seeds:    {rpe.SEEDS}")
    print(f"  Output:   {ISOLATED_OUTPUT}")
    print(f"  Datasets: {datasets_to_run}")
    print(f"  Config:   {rpe.TABULAR_CONFIG}")
    if args.quick:
        print(f"  Mode:     QUICK (1 seed, 5 rounds)")
    print("=" * 70)

    t0 = time.time()

    for ds_name in datasets_to_run:
        print(f"\n>>> P1.2 Tabular: {ds_name}")
        rpe.run_p12_multi_dataset(
            resume=True,
            filter_dataset=ds_name,
            use_amp=False,
            use_early_stopping=False,
        )

    # Generate output tables
    print("\n>>> Generating tables...")
    p12 = rpe.load_checkpoint("p12_multidataset")
    if p12:
        sig = rpe.run_p21_significance(p12)
        tex = rpe.generate_multi_dataset_table(p12, sig or {})
        (ISOLATED_OUTPUT / "table_new_tabular.tex").write_text(tex)
        print(f"  Saved table_new_tabular.tex")

        comm = rpe.compute_communication_costs(p12)
        print(f"  Computed communication costs")

    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print(f"  Completed in {elapsed/60:.1f} min ({elapsed:.0f}s)")
    print(f"  Output: {ISOLATED_OUTPUT}")
    print("=" * 70)


if __name__ == "__main__":
    main()

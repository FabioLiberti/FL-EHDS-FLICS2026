"""
Paper Experiments screen for FL-EHDS terminal interface.
Provides incremental workflow for FLICS 2026 paper experiments.
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional, List

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from terminal.colors import (
    Colors, Style, print_section, print_subsection,
    print_success, print_error, print_info, print_warning, clear_screen
)
from terminal.validators import get_int, get_bool, confirm, display_config_summary
from terminal.menu import Menu, MenuItem


# Constants matching run_paper_experiments.py
ALGORITHMS = ["FedAvg", "FedProx", "SCAFFOLD", "FedNova", "Ditto"]
IMAGING_DATASETS = ["Brain_Tumor", "chest_xray", "Skin_Cancer"]
TABULAR_DATASETS = ["Diabetes", "Heart_Disease"]
ALL_DATASETS = IMAGING_DATASETS + TABULAR_DATASETS
SEEDS = [42, 123, 456]
NONIID_ALPHAS = [0.1, 0.5, 1.0, 5.0]
NONIID_ALGOS = ["FedAvg", "FedProx", "SCAFFOLD"]


class PaperExperimentsScreen:
    """Incremental workflow for FLICS 2026 paper experiments."""

    def __init__(self):
        pass

    def run(self):
        """Run the paper experiments menu loop."""
        while True:
            clear_screen()
            print_section("ESPERIMENTI PAPER FLICS 2026")
            print_info("Workflow incrementale con checkpoint/resume")
            print()

            menu = Menu("Seleziona operazione", [
                MenuItem("1", "Status Dashboard", self._status_dashboard),
                MenuItem("2", "P1.2: Multi-Dataset FL", self._run_p12),
                MenuItem("3", "P1.3: Studio Ablativo", self._run_p13),
                MenuItem("4", "P1.4: Non-IID Severity (alpha sweep)", self._run_p14),
                MenuItem("5", "P2.2: Privacy Attack (DLG)", self._run_p22),
                MenuItem("6", "Genera Output (Tabelle + Figure)", self._generate_outputs),
                MenuItem("7", "Apri cartella output", self._open_output),
                MenuItem("0", "Torna al menu principale", lambda: "back"),
            ])

            result = menu.display()
            if result is None:
                break

            if result.handler:
                handler_result = result.handler()
                if handler_result == "back":
                    break

    # ------------------------------------------------------------------
    # 1. Status Dashboard
    # ------------------------------------------------------------------
    def _status_dashboard(self):
        """Show experiment completion status."""
        clear_screen()
        print_section("STATUS DASHBOARD")

        try:
            from benchmarks.generate_paper_outputs import (
                load_all_checkpoints, print_status
            )
            checkpoints = load_all_checkpoints()
            print()
            print_status(checkpoints)

            # Summary counts
            p12 = checkpoints.get("p12")
            p13 = checkpoints.get("p13")
            p14 = checkpoints.get("p14")
            p22 = checkpoints.get("p22")

            total_p12 = len(ALGORITHMS) * len(ALL_DATASETS) * len(SEEDS)
            total_p13 = 39  # clip(9) + epsilon(18) + model(6) + classweights(6)
            total_p14 = len(NONIID_ALPHAS) * len(NONIID_ALGOS) * len(SEEDS)
            total_p22 = 12  # 4 eps x 3 seeds

            done_p12 = 0
            done_p13 = 0
            done_p14 = 0
            done_p22 = 0

            if p12 and "completed" in p12:
                done_p12 = sum(1 for v in p12["completed"].values() if "error" not in v)
            if p13 and "completed" in p13:
                done_p13 = sum(1 for v in p13["completed"].values() if "error" not in v)
            if p14 and "completed" in p14:
                done_p14 = sum(1 for v in p14["completed"].values() if "error" not in v)
            if p22 and "completed" in p22:
                done_p22 = sum(1 for v in p22["completed"].values() if "error" not in v)

            print()
            print_subsection("Riepilogo Completamento")
            self._print_progress_bar("P1.2 Multi-Dataset", done_p12, total_p12)
            self._print_progress_bar("P1.3 Ablation    ", done_p13, total_p13)
            self._print_progress_bar("P1.4 Non-IID     ", done_p14, total_p14)
            self._print_progress_bar("P2.2 Attack      ", done_p22, total_p22)

        except ImportError as e:
            print_error(f"Impossibile importare generate_paper_outputs: {e}")
        except Exception as e:
            print_error(f"Errore: {e}")

        input(f"\n{Style.MUTED}Premi Enter per continuare...{Colors.RESET}")

    def _print_progress_bar(self, label: str, done: int, total: int):
        """Print a colored progress bar."""
        pct = (done / total * 100) if total > 0 else 0
        bar_len = 30
        filled = int(bar_len * done / total) if total > 0 else 0
        bar = "=" * filled + "-" * (bar_len - filled)

        if pct == 100:
            color = Style.SUCCESS
        elif pct > 0:
            color = Colors.YELLOW
        else:
            color = Style.MUTED

        print(f"  {label}  [{color}{bar}{Colors.RESET}] {done}/{total} ({pct:.0f}%)")

    # ------------------------------------------------------------------
    # 2. P1.2 Multi-Dataset
    # ------------------------------------------------------------------
    def _run_p12(self):
        """Run P1.2 multi-dataset experiments with optional filtering."""
        clear_screen()
        print_section("P1.2: MULTI-DATASET FL COMPARISON")
        print_info("5 algoritmi x 5 dataset x 3 seed = 75 esperimenti")
        print_info("Resume automatico: gli esperimenti completati vengono saltati")
        print()

        # Dataset selection
        print_subsection("Seleziona Dataset")
        ds_options = ["Tutti"] + ALL_DATASETS
        for i, ds in enumerate(ds_options):
            if i == 0:
                print(f"  {Style.HIGHLIGHT}{i}. {ds}{Colors.RESET}")
            else:
                ds_type = "imaging" if ds in IMAGING_DATASETS else "tabular"
                print(f"  {Style.HIGHLIGHT}{i}. {ds}{Colors.RESET} ({ds_type})")

        ds_idx = get_int("\nDataset", 0, 0, len(ds_options) - 1)
        filter_dataset = None if ds_idx == 0 else ds_options[ds_idx]

        # Algorithm selection
        print()
        print_subsection("Seleziona Algoritmo")
        algo_options = ["Tutti"] + ALGORITHMS
        for i, algo in enumerate(algo_options):
            print(f"  {Style.HIGHLIGHT}{i}. {algo}{Colors.RESET}")

        algo_idx = get_int("\nAlgoritmo", 0, 0, len(algo_options) - 1)
        filter_algo = None if algo_idx == 0 else algo_options[algo_idx]

        # Summary
        ds_label = filter_dataset or "Tutti (5)"
        algo_label = filter_algo or "Tutti (5)"
        print()
        print(f"  Dataset:   {Style.HIGHLIGHT}{ds_label}{Colors.RESET}")
        print(f"  Algoritmo: {Style.HIGHLIGHT}{algo_label}{Colors.RESET}")

        n_ds = 1 if filter_dataset else len(ALL_DATASETS)
        n_algo = 1 if filter_algo else len(ALGORITHMS)
        n_total = n_ds * n_algo * len(SEEDS)
        print(f"  Esperimenti max: {n_total} (meno i completati)")

        if not confirm("\nAvviare?"):
            return

        print()
        try:
            from benchmarks.run_paper_experiments import run_p12_multi_dataset
            results = run_p12_multi_dataset(
                resume=True,
                filter_dataset=filter_dataset,
                filter_algo=filter_algo,
            )
            completed = results.get("completed", {})
            ok_count = sum(1 for v in completed.values() if "error" not in v)
            print_success(f"\nCompletati: {ok_count} esperimenti nel checkpoint")
        except KeyboardInterrupt:
            print_warning("\nInterrotto dall'utente. I risultati parziali sono salvati nel checkpoint.")
        except Exception as e:
            print_error(f"Errore: {e}")

        input(f"\n{Style.MUTED}Premi Enter per continuare...{Colors.RESET}")

    # ------------------------------------------------------------------
    # 3. P1.3 Ablation Study
    # ------------------------------------------------------------------
    def _run_p13(self):
        """Run P1.3 ablation study on chest_xray."""
        clear_screen()
        print_section("P1.3: STUDIO ABLATIVO (chest_xray)")
        print_info("Resume automatico: gli esperimenti completati vengono saltati")
        print()

        print_subsection("Fattori Ablazione")
        print(f"  {Style.HIGHLIGHT}A. Gradient Clipping{Colors.RESET}: C = 0.5, 1.0, 2.0  (eps=5, 3 seed = 9 run)")
        print(f"  {Style.HIGHLIGHT}B. Privacy Epsilon{Colors.RESET}:   eps = 0.5, 1, 2, 5, 10, inf  (C=1.0, 3 seed = 18 run)")
        print(f"  {Style.HIGHLIGHT}C. Modello{Colors.RESET}:           CNN vs ResNet18  (no DP, 3 seed = 6 run)")
        print(f"  {Style.HIGHLIGHT}D. Class Weights{Colors.RESET}:     on vs off  (no DP, 3 seed = 6 run)")
        print(f"\n  Totale massimo: 39 esperimenti")

        if not confirm("\nAvviare ablation study?"):
            return

        print()
        try:
            from benchmarks.run_paper_experiments import run_p13_ablation
            results = run_p13_ablation(resume=True)
            completed = results.get("completed", {})
            ok_count = sum(1 for v in completed.values() if "error" not in v)
            print_success(f"\nCompletati: {ok_count} esperimenti nel checkpoint")
        except KeyboardInterrupt:
            print_warning("\nInterrotto dall'utente. I risultati parziali sono salvati nel checkpoint.")
        except Exception as e:
            print_error(f"Errore: {e}")

        input(f"\n{Style.MUTED}Premi Enter per continuare...{Colors.RESET}")

    # ------------------------------------------------------------------
    # 4. P1.4 Non-IID Severity
    # ------------------------------------------------------------------
    def _run_p14(self):
        """Run P1.4 non-IID severity study (alpha sweep)."""
        clear_screen()
        print_section("P1.4: NON-IID SEVERITY STUDY (alpha sweep)")
        print_info("Resume automatico: gli esperimenti completati vengono saltati")
        print()

        print_subsection("Configurazione")
        print(f"  Dataset:    chest_xray (2 classi)")
        print(f"  Algoritmi:  FedAvg, FedProx, SCAFFOLD")
        print(f"  Alpha:      0.1, 0.5, 1.0, 5.0 (Dirichlet)")
        print(f"  Seed:       42, 123, 456")
        total = len(NONIID_ALPHAS) * len(NONIID_ALGOS) * len(SEEDS)
        print(f"\n  Totale massimo: {total} esperimenti")
        print_info("  Alpha basso = piu' eterogeneo (non-IID severo)")

        if not confirm("\nAvviare non-IID severity study?"):
            return

        print()
        try:
            from benchmarks.run_paper_experiments import run_p14_noniid_severity
            results = run_p14_noniid_severity(resume=True)
            completed = results.get("completed", {})
            ok_count = sum(1 for v in completed.values() if "error" not in v)
            print_success(f"\nCompletati: {ok_count} esperimenti nel checkpoint")
        except KeyboardInterrupt:
            print_warning("\nInterrotto dall'utente. I risultati parziali sono salvati nel checkpoint.")
        except Exception as e:
            print_error(f"Errore: {e}")

        input(f"\n{Style.MUTED}Premi Enter per continuare...{Colors.RESET}")

    # ------------------------------------------------------------------
    # 5. P2.2 Privacy Attack (was 4)
    # ------------------------------------------------------------------
    def _run_p22(self):
        """Run P2.2 DLG privacy attack evaluation."""
        clear_screen()
        print_section("P2.2: PRIVACY ATTACK (Gradient Inversion)")
        print_info("DLG attack su dati sintetici tabular con e senza DP")
        print()

        print_subsection("Configurazione")
        print(f"  Epsilon: inf (no DP), 10, 5, 1")
        print(f"  Seed: 42, 123, 456")
        print(f"  Totale: 12 esperimenti (~60 secondi)")

        if not confirm("\nAvviare privacy attack evaluation?"):
            return

        print()
        try:
            from benchmarks.run_paper_experiments import run_p22_privacy_attack
            results = run_p22_privacy_attack()
            completed = results.get("completed", {})
            ok_count = sum(1 for v in completed.values() if "error" not in v)
            print_success(f"\nCompletati: {ok_count} esperimenti")

            # Show quick results
            ok = {k: v for k, v in completed.items() if "error" not in v}
            if ok:
                import numpy as np
                print()
                print_subsection("Risultati MSE Ricostruzione")
                for eps_label in ["inf", "10.0", "5.0", "1.0"]:
                    vals = [v["reconstruction_mse"] for k, v in ok.items()
                            if k.startswith(f"eps_{eps_label}") and "reconstruction_mse" in v]
                    if vals:
                        eps_disp = "inf (no DP)" if eps_label == "inf" else f"eps={eps_label}"
                        print(f"  {eps_disp:<16} MSE = {np.mean(vals):.3f} +/- {np.std(vals):.3f}")
        except KeyboardInterrupt:
            print_warning("\nInterrotto dall'utente.")
        except Exception as e:
            print_error(f"Errore: {e}")

        input(f"\n{Style.MUTED}Premi Enter per continuare...{Colors.RESET}")

    # ------------------------------------------------------------------
    # 6. Generate Output
    # ------------------------------------------------------------------
    def _generate_outputs(self):
        """Generate LaTeX tables and figures from checkpoints."""
        clear_screen()
        print_section("GENERA OUTPUT (Tabelle LaTeX + Figure)")
        print_info("Genera output incrementali dai checkpoint disponibili")
        print()

        try:
            from benchmarks.generate_paper_outputs import (
                load_all_checkpoints, compute_significance,
                generate_multi_dataset_table, generate_ablation_table,
                generate_attack_table, generate_figures,
                generate_fairness_table, generate_noniid_table,
                generate_communication_table, OUTPUT_DIR
            )

            checkpoints = load_all_checkpoints()
            p12 = checkpoints.get("p12")
            p13 = checkpoints.get("p13")
            p14 = checkpoints.get("p14")
            p22 = checkpoints.get("p22")
            comm = checkpoints.get("comm")

            has_data = p12 or p13 or p14 or p22
            if not has_data:
                print_warning("Nessun checkpoint trovato. Esegui prima degli esperimenti.")
                input(f"\n{Style.MUTED}Premi Enter per continuare...{Colors.RESET}")
                return

            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            generated = []

            # Compute significance
            sig = {}
            if p12:
                try:
                    sig = compute_significance(p12)
                except Exception:
                    sig = checkpoints.get("p21") or {}

            # Tables
            print_subsection("Generazione Tabelle LaTeX")
            if p12:
                tex = generate_multi_dataset_table(p12, sig)
                path = OUTPUT_DIR / "table_multi_dataset.tex"
                path.write_text(tex)
                generated.append(path.name)
                print_success(f"  {path.name}")

                tex = generate_fairness_table(p12)
                path = OUTPUT_DIR / "table_fairness.tex"
                path.write_text(tex)
                generated.append(path.name)
                print_success(f"  {path.name}")

            if p13:
                tex = generate_ablation_table(p13)
                path = OUTPUT_DIR / "table_ablation.tex"
                path.write_text(tex)
                generated.append(path.name)
                print_success(f"  {path.name}")

            if p14:
                tex = generate_noniid_table(p14)
                path = OUTPUT_DIR / "table_noniid.tex"
                path.write_text(tex)
                generated.append(path.name)
                print_success(f"  {path.name}")

            if p22:
                tex = generate_attack_table(p22)
                path = OUTPUT_DIR / "table_attack.tex"
                path.write_text(tex)
                generated.append(path.name)
                print_success(f"  {path.name}")

            if comm:
                tex = generate_communication_table(comm)
                if tex:
                    path = OUTPUT_DIR / "table_communication.tex"
                    path.write_text(tex)
                    generated.append(path.name)
                    print_success(f"  {path.name}")

            if not generated:
                print_info("  Nessuna tabella generata (dati insufficienti)")

            # Figures
            print()
            print_subsection("Generazione Figure")
            try:
                n = generate_figures(p12 or {}, p13 or {}, p22 or {}, OUTPUT_DIR)
                if n and n > 0:
                    print_success(f"  {n} figure base generate")
                else:
                    print_info("  Nessuna figura base generata (dati insufficienti)")
            except Exception as e:
                print_error(f"  Errore figure base: {e}")

            # New figures (fairness, noniid, communication)
            try:
                from benchmarks.generate_paper_outputs import (
                    _fig_fairness, _fig_noniid_alpha, _fig_communication
                )
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                plt.rcParams.update({
                    "font.size": 11, "font.family": "serif",
                    "axes.labelsize": 12, "axes.titlesize": 13,
                    "xtick.labelsize": 10, "ytick.labelsize": 10,
                    "legend.fontsize": 9, "figure.dpi": 150,
                })

                extra_figs = 0
                if p12 and p12.get("completed"):
                    ok12 = {k: v for k, v in p12["completed"].items() if "error" not in v}
                    if ok12:
                        _fig_fairness(ok12, OUTPUT_DIR, plt)
                        extra_figs += 1
                if p14 and p14.get("completed"):
                    _fig_noniid_alpha(p14["completed"], OUTPUT_DIR, plt)
                    extra_figs += 1
                if comm:
                    _fig_communication(comm.get("per_dataset", {}), OUTPUT_DIR, plt)
                    extra_figs += 1
                if extra_figs > 0:
                    print_success(f"  {extra_figs} figure aggiuntive generate")
            except Exception as e:
                print_error(f"  Errore figure aggiuntive: {e}")

            # List all output files
            print()
            print_subsection("File nella cartella output")
            if OUTPUT_DIR.exists():
                for f in sorted(OUTPUT_DIR.iterdir()):
                    if f.suffix in (".tex", ".pdf", ".png") and not f.name.startswith("checkpoint"):
                        size = f.stat().st_size
                        size_str = f"{size / 1024:.0f} KB" if size > 1024 else f"{size} B"
                        print(f"  {f.name:<40} {size_str}")

            print(f"\n  Output dir: {OUTPUT_DIR}")

        except ImportError as e:
            print_error(f"Impossibile importare: {e}")
        except Exception as e:
            print_error(f"Errore: {e}")

        input(f"\n{Style.MUTED}Premi Enter per continuare...{Colors.RESET}")

    # ------------------------------------------------------------------
    # 7. Open Output Folder
    # ------------------------------------------------------------------
    def _open_output(self):
        """Open the output directory in the system file manager."""
        from benchmarks.generate_paper_outputs import OUTPUT_DIR

        if OUTPUT_DIR.exists():
            if sys.platform == "darwin":
                os.system(f'open "{OUTPUT_DIR}"')
            elif sys.platform == "linux":
                os.system(f'xdg-open "{OUTPUT_DIR}"')
            else:
                os.system(f'explorer "{OUTPUT_DIR}"')
            print_success(f"Aperta cartella: {OUTPUT_DIR}")
        else:
            print_warning(f"La cartella non esiste ancora: {OUTPUT_DIR}")
            print_info("Esegui prima degli esperimenti o genera gli output.")

        input(f"\n{Style.MUTED}Premi Enter per continuare...{Colors.RESET}")

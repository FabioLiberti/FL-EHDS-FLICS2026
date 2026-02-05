"""
Output and export screen for FL-EHDS terminal interface.
Handles LaTeX table generation, CSV export, and figure generation.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import json

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from terminal.colors import (
    Colors, Style, print_section, print_subsection,
    print_success, print_error, print_info, print_warning, clear_screen
)
from terminal.validators import get_input, get_choice, confirm
from terminal.menu import Menu, MenuItem


class OutputScreen:
    """Output and export functionality."""

    def __init__(self):
        self.output_dir = Path(__file__).parent.parent.parent / "results"
        self.output_dir.mkdir(exist_ok=True)

    def run(self):
        """Run the output screen."""
        while True:
            clear_screen()
            print_section("ESPORTA RISULTATI")

            menu = Menu("Seleziona formato", [
                MenuItem("1", "Genera tabella LaTeX", self._generate_latex_prompt),
                MenuItem("2", "Esporta CSV", self._export_csv_prompt),
                MenuItem("3", "Genera grafici", self._generate_plots_prompt),
                MenuItem("4", "Esporta JSON", self._export_json_prompt),
                MenuItem("5", "Visualizza file generati", self._show_generated_files),
                MenuItem("0", "Torna al menu principale", lambda: "back"),
            ])

            result = menu.display()
            if result is None:
                break

            if result.handler:
                handler_result = result.handler()
                if handler_result == "back":
                    break

    def _generate_latex_prompt(self):
        """Prompt for LaTeX generation."""
        print_info("Per generare tabelle LaTeX, eseguire prima un benchmark o training")
        input(f"\n{Style.MUTED}Premi Enter per continuare...{Colors.RESET}")

    def _export_csv_prompt(self):
        """Prompt for CSV export."""
        print_info("Per esportare CSV, eseguire prima un benchmark o training")
        input(f"\n{Style.MUTED}Premi Enter per continuare...{Colors.RESET}")

    def _generate_plots_prompt(self):
        """Prompt for plot generation."""
        print_info("Per generare grafici, eseguire prima un benchmark o training")
        input(f"\n{Style.MUTED}Premi Enter per continuare...{Colors.RESET}")

    def _export_json_prompt(self):
        """Prompt for JSON export."""
        print_info("Per esportare JSON, eseguire prima un benchmark o training")
        input(f"\n{Style.MUTED}Premi Enter per continuare...{Colors.RESET}")

    def _show_generated_files(self):
        """Show list of generated files."""
        clear_screen()
        print_section("FILE GENERATI")

        if not self.output_dir.exists():
            print_warning("Nessun file generato")
            input(f"\n{Style.MUTED}Premi Enter per continuare...{Colors.RESET}")
            return

        files = list(self.output_dir.glob("*"))

        if not files:
            print_warning("Nessun file nella directory di output")
        else:
            print(f"\nDirectory: {self.output_dir}\n")
            print(f"{Style.TITLE}{'Nome':<40} {'Dimensione':<15} {'Data':<20}{Colors.RESET}")
            print("-" * 75)

            for f in sorted(files):
                if f.is_file():
                    size = f.stat().st_size
                    mtime = datetime.fromtimestamp(f.stat().st_mtime)
                    size_str = f"{size / 1024:.1f} KB" if size > 1024 else f"{size} B"
                    print(f"  {f.name:<38} {size_str:<13} {mtime.strftime('%Y-%m-%d %H:%M')}")

        input(f"\n{Style.MUTED}Premi Enter per continuare...{Colors.RESET}")

    def generate_latex_table(self, results: Dict, config: Dict):
        """Generate LaTeX table from results."""
        clear_screen()
        print_section("GENERAZIONE TABELLA LATEX")

        if not results:
            print_warning("Nessun risultato da esportare")
            input(f"\n{Style.MUTED}Premi Enter per continuare...{Colors.RESET}")
            return

        # Generate LaTeX
        latex = self._create_latex_table(results, config)

        # Show preview
        print_subsection("ANTEPRIMA")
        print(latex)

        # Save to file
        if confirm("\nSalvare tabella su file?", default=True):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = self.output_dir / f"table_results_{timestamp}.tex"

            with open(filename, "w") as f:
                f.write(latex)

            print_success(f"Tabella salvata in: {filename}")

        input(f"\n{Style.MUTED}Premi Enter per continuare...{Colors.RESET}")

    def _create_latex_table(self, results: Dict, config: Dict) -> str:
        """Create LaTeX table string."""
        # Get number of seeds for caption
        num_seeds = config.get("num_seeds", 3)
        num_clients = config.get("num_clients", 5)
        num_rounds = config.get("num_rounds", 30)

        latex = []
        latex.append(r"\begin{table}[htbp]")
        latex.append(r"\centering")
        latex.append(r"\caption{Experimental Results}")
        latex.append(r"\label{tab:results}")
        latex.append(r"\small")
        latex.append(r"\begin{tabular}{lccc}")
        latex.append(r"\toprule")
        latex.append(r"\textbf{Configuration} & \textbf{Accuracy} & \textbf{F1} & \textbf{AUC} \\")
        latex.append(r"\midrule")

        for config_name, metrics in results.items():
            acc = metrics.get("accuracy", {})
            f1 = metrics.get("f1", {})
            auc = metrics.get("auc", {})

            # Format with +/- notation
            acc_mean = acc.get("mean", 0) * 100  # Convert to percentage
            acc_std = acc.get("std", 0) * 100
            f1_mean = f1.get("mean", 0)
            f1_std = f1.get("std", 0)
            auc_mean = auc.get("mean", 0)
            auc_std = auc.get("std", 0)

            # Escape special characters in config name
            safe_name = config_name.replace("_", r"\_").replace("&", r"\&")

            latex.append(
                f"{safe_name} & {acc_mean:.1f}\\%$\\pm${acc_std:.2f} & "
                f"{f1_mean:.2f}$\\pm${f1_std:.2f} & {auc_mean:.2f}$\\pm${auc_std:.2f} \\\\"
            )

        latex.append(r"\bottomrule")
        latex.append(r"\end{tabular}")
        latex.append("")
        latex.append(r"\vspace{1mm}")
        latex.append(
            f"\\footnotesize{{{num_clients} clients, {num_rounds} rounds. "
            f"Results are mean $\\pm$ std over {num_seeds} runs.}}"
        )
        latex.append(r"\end{table}")

        return "\n".join(latex)

    def export_training_results(self, results: Dict):
        """Export training results to multiple formats."""
        clear_screen()
        print_section("ESPORTA RISULTATI TRAINING")

        if not results:
            print_warning("Nessun risultato da esportare")
            input(f"\n{Style.MUTED}Premi Enter per continuare...{Colors.RESET}")
            return

        menu = Menu("Seleziona formato", [
            MenuItem("1", "JSON", lambda: "json"),
            MenuItem("2", "CSV", lambda: "csv"),
            MenuItem("3", "Entrambi", lambda: "both"),
            MenuItem("0", "Annulla", lambda: "cancel"),
        ])

        result = menu.display()
        if not result or result.handler() == "cancel":
            return

        format_choice = result.handler()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if format_choice in ["json", "both"]:
            json_file = self.output_dir / f"training_results_{timestamp}.json"
            with open(json_file, "w") as f:
                json.dump(results, f, indent=2, default=str)
            print_success(f"JSON salvato: {json_file}")

        if format_choice in ["csv", "both"]:
            csv_file = self.output_dir / f"training_history_{timestamp}.csv"
            self._export_history_csv(results.get("history", []), csv_file)
            print_success(f"CSV salvato: {csv_file}")

        input(f"\n{Style.MUTED}Premi Enter per continuare...{Colors.RESET}")

    def _export_history_csv(self, history: list, filepath: Path):
        """Export training history to CSV."""
        if not history:
            return

        # Get all keys from first entry
        keys = list(history[0].keys()) if history else []

        with open(filepath, "w") as f:
            # Header
            f.write(",".join(keys) + "\n")

            # Data rows
            for entry in history:
                row = [str(entry.get(k, "")) for k in keys]
                f.write(",".join(row) + "\n")

    def export_comparison_results(self, results: Dict, config: Dict):
        """Export algorithm comparison results."""
        clear_screen()
        print_section("ESPORTA RISULTATI CONFRONTO")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # JSON export
        json_file = self.output_dir / f"comparison_results_{timestamp}.json"
        export_data = {
            "config": config,
            "results": results,
            "timestamp": timestamp,
        }

        with open(json_file, "w") as f:
            json.dump(export_data, f, indent=2, default=str)

        print_success(f"Risultati salvati in: {json_file}")

        # Also generate LaTeX table
        if confirm("Generare anche tabella LaTeX?", default=True):
            self.generate_latex_table(results, config)
        else:
            input(f"\n{Style.MUTED}Premi Enter per continuare...{Colors.RESET}")

    def generate_convergence_plot(self, history: list, config: Dict):
        """Generate convergence plot."""
        try:
            import matplotlib.pyplot as plt
            import numpy as np

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            rounds = range(1, len(history) + 1)

            # Loss plot
            losses = [h.get("loss", 0) for h in history]
            axes[0].plot(rounds, losses, "b-", linewidth=2)
            axes[0].set_xlabel("Round")
            axes[0].set_ylabel("Loss")
            axes[0].set_title("Training Loss")
            axes[0].grid(True, alpha=0.3)

            # Accuracy plot
            accs = [h.get("accuracy", 0) for h in history]
            axes[1].plot(rounds, accs, "g-", linewidth=2)
            axes[1].set_xlabel("Round")
            axes[1].set_ylabel("Accuracy")
            axes[1].set_title("Test Accuracy")
            axes[1].grid(True, alpha=0.3)

            plt.tight_layout()

            # Save
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = self.output_dir / f"convergence_{timestamp}.png"
            plt.savefig(filename, dpi=150, bbox_inches="tight")
            plt.close()

            print_success(f"Grafico salvato: {filename}")

        except ImportError:
            print_error("matplotlib non disponibile per la generazione dei grafici")
        except Exception as e:
            print_error(f"Errore nella generazione del grafico: {e}")

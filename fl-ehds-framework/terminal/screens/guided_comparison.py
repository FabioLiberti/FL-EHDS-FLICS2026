"""
Guided algorithm comparison screen for FL-EHDS terminal interface.
Recommends algorithms based on healthcare use cases and runs automated comparisons.
"""

import sys
from pathlib import Path
from typing import Dict, List, Any
import time
import json
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import questionary
    HAS_QUESTIONARY = True
except ImportError:
    HAS_QUESTIONARY = False

from terminal.colors import (
    Colors, Style, print_section, print_subsection,
    print_success, print_error, print_info, print_warning, clear_screen
)
from terminal.validators import confirm, display_config_summary
from terminal.menu import Menu, MenuItem, MENU_STYLE
from terminal.recommendations import (
    get_use_cases, get_use_case_by_id, get_comparison_config, UseCase
)


class GuidedComparisonScreen:
    """Guided algorithm comparison based on healthcare use cases."""

    def __init__(self):
        self.selected_use_case = None
        self.config = None
        self.results = {}
        self.histories = {}  # Store convergence history for all algorithms

    def run(self):
        """Run the guided comparison screen."""
        while True:
            clear_screen()
            print_section("CONFRONTO GUIDATO PER CASO D'USO")

            menu = Menu("Seleziona azione", [
                MenuItem("1", "Seleziona caso d'uso sanitario", self._select_use_case),
                MenuItem("2", "Visualizza configurazione consigliata", self._show_config),
                MenuItem("3", "Esegui confronto algoritmi", self._run_comparison),
                MenuItem("4", "Visualizza risultati", self._show_results),
                MenuItem("5", "Genera report comparativo", self._generate_report),
                MenuItem("0", "Torna al menu principale", lambda: "back"),
            ])

            result = menu.display()
            if result is None:
                break

            if result.handler:
                handler_result = result.handler()
                if handler_result == "back":
                    break

    def _select_use_case(self):
        """Select healthcare use case."""
        clear_screen()
        print_section("SELEZIONA CASO D'USO SANITARIO")

        use_cases = get_use_cases()

        print_info("Seleziona lo scenario che meglio descrive il tuo caso d'uso:")
        print()

        if HAS_QUESTIONARY:
            choices = [
                questionary.Choice(
                    title=f"{uc.name}",
                    value=uc.id
                )
                for uc in use_cases
            ]

            selected_id = questionary.select(
                "Caso d'uso:",
                choices=choices,
                style=MENU_STYLE,
                instruction="(Frecce: naviga | Enter: seleziona)"
            ).ask()

            if selected_id:
                self.selected_use_case = get_use_case_by_id(selected_id)
                self.config = get_comparison_config(self.selected_use_case)
        else:
            for i, uc in enumerate(use_cases, 1):
                print(f"  {i}. {uc.name}")
                print(f"     {Style.MUTED}{uc.description}{Colors.RESET}")
                print()

            try:
                choice = int(input(f"\n{Style.INFO}Scelta (1-{len(use_cases)}): {Colors.RESET}"))
                if 1 <= choice <= len(use_cases):
                    self.selected_use_case = use_cases[choice - 1]
                    self.config = get_comparison_config(self.selected_use_case)
            except ValueError:
                print_error("Scelta non valida")

        if self.selected_use_case:
            self._display_use_case_details()

        input(f"\n{Style.MUTED}Premi Enter per continuare...{Colors.RESET}")

    def _display_use_case_details(self):
        """Display details of selected use case."""
        uc = self.selected_use_case

        print()
        print_subsection(f"CASO SELEZIONATO: {uc.name.upper()}")

        print(f"\n{Style.TITLE}Descrizione:{Colors.RESET}")
        print(f"  {uc.description}")

        print(f"\n{Style.TITLE}Caratteristiche dati:{Colors.RESET}")
        print(f"  {uc.data_characteristics}")

        print(f"\n{Style.TITLE}Algoritmi consigliati:{Colors.RESET}")
        for algo in uc.recommended_algorithms:
            print(f"  - {Style.HIGHLIGHT}{algo}{Colors.RESET}")

        print(f"\n{Style.TITLE}Motivazione:{Colors.RESET}")
        print(f"  {uc.rationale}")

    def _show_config(self):
        """Show recommended configuration."""
        clear_screen()
        print_section("CONFIGURAZIONE CONSIGLIATA")

        if not self.selected_use_case:
            print_warning("Nessun caso d'uso selezionato. Selezionare prima un caso d'uso.")
            input(f"\n{Style.MUTED}Premi Enter per continuare...{Colors.RESET}")
            return

        self._display_use_case_details()

        print_subsection("PARAMETRI CONFIGURAZIONE")
        display_config_summary(self.config)

        input(f"\n{Style.MUTED}Premi Enter per continuare...{Colors.RESET}")

    def _run_comparison(self):
        """Execute algorithm comparison for selected use case."""
        clear_screen()
        print_section("ESECUZIONE CONFRONTO")

        if not self.selected_use_case:
            print_warning("Nessun caso d'uso selezionato.")
            input(f"\n{Style.MUTED}Premi Enter per continuare...{Colors.RESET}")
            return

        print(f"{Style.TITLE}Caso d'uso:{Colors.RESET} {self.selected_use_case.name}")
        print(f"{Style.TITLE}Algoritmi:{Colors.RESET} {', '.join(self.config['algorithms'])}")
        print()

        display_config_summary(self.config)

        # Ask for verbose mode
        verbose = confirm("\nMostrare progresso dettagliato (round per round)?", default=True)

        if not confirm("\nAvviare il confronto?", default=True):
            return

        print()

        try:
            from terminal.fl_trainer import FederatedTrainer, HealthcareMLP
            import torch
            import numpy as np

            self.results = {}
            self.histories = {}  # Reset histories
            algorithms = self.config["algorithms"]
            num_seeds = self.config["num_seeds"]
            num_rounds = self.config["num_rounds"]
            num_clients = self.config["num_clients"]
            local_epochs = self.config["local_epochs"]

            # === VERIFICATION: Show that training is real ===
            print_subsection("VERIFICA TRAINING REALE")
            print(f"{Style.TITLE}Neural Network:{Colors.RESET}")
            model = HealthcareMLP()
            total_params = sum(p.numel() for p in model.parameters())
            print(f"  Modello: HealthcareMLP (PyTorch nn.Module)")
            print(f"  Architettura: 10 -> 64 -> 32 -> 2 (MLP)")
            print(f"  Parametri totali: {total_params:,}")
            print(f"  Device: {torch.device('cpu')}")
            print(f"  Optimizer: SGD con lr={self.config['learning_rate']}")
            print(f"  Loss: CrossEntropyLoss")
            first_layer = list(model.parameters())[0]
            print(f"  Pesi iniziali (layer 0, primi 5): {first_layer.data.flatten()[:5].tolist()}")

            print(f"\n{Style.TITLE}Dataset sintetico sanitario:{Colors.RESET}")
            print(f"  Features: age, bmi, bp_systolic, glucose, cholesterol, ...")
            print(f"  Target: Rischio malattia (binario)")
            print(f"  Distribuzione: {'IID' if self.config['is_iid'] else 'Non-IID (Dirichlet)'}")
            print()

            total_runs = len(algorithms) * num_seeds
            print_info(f"Totale run: {total_runs} ({len(algorithms)} algoritmi x {num_seeds} seed)")
            print_info(f"Per ogni run: {num_rounds} round x {num_clients} client x {local_epochs} epoche")
            print()

            start_time = time.time()

            # Progress callback for verbose mode
            def make_progress_callback(algorithm_name, seed_num):
                def progress_callback(event_type, **kwargs):
                    if not verbose:
                        return
                    if event_type == "round_start":
                        r = kwargs.get("round_num", 0)
                        print(f"      Round {r}/{num_rounds}: ", end="", flush=True)
                    elif event_type == "client_end":
                        client_id = kwargs.get("client_id", 0)
                        print(f"C{client_id} ", end="", flush=True)
                    elif event_type == "round_end":
                        acc = kwargs.get("acc", 0)
                        print(f"-> acc={acc:.2%}")
                return progress_callback

            for algorithm in algorithms:
                print(f"\n{Style.TITLE}===== Testing {algorithm} ====={Colors.RESET}")

                algo_results = []
                algo_histories = []

                for seed in range(num_seeds):
                    print(f"  Run {seed + 1}/{num_seeds} (seed={seed})")

                    progress_cb = make_progress_callback(algorithm, seed) if verbose else None

                    trainer = FederatedTrainer(
                        num_clients=num_clients,
                        samples_per_client=200,
                        algorithm=algorithm,
                        local_epochs=local_epochs,
                        batch_size=self.config["batch_size"],
                        learning_rate=self.config["learning_rate"],
                        is_iid=self.config["is_iid"],
                        alpha=self.config["alpha"],
                        mu=self.config.get("mu", 0.1),
                        dp_enabled=self.config.get("dp_enabled", False),
                        dp_epsilon=self.config.get("dp_epsilon", 10.0),
                        seed=seed,
                        server_lr=self.config.get("server_lr", 0.1),
                        beta1=self.config.get("beta1", 0.9),
                        beta2=self.config.get("beta2", 0.99),
                        tau=self.config.get("tau", 1e-3),
                        progress_callback=progress_cb,
                    )

                    # Show data distribution for first run
                    if seed == 0:
                        stats = trainer.get_client_data_stats()
                        print(f"    Distribuzione dati per client:")
                        for cid, cstats in stats.items():
                            print(f"      Client {cid}: {cstats['num_samples']} samples, labels={cstats['label_distribution']}")

                    # Run all rounds with all metrics
                    history = []
                    final_result = None
                    for round_num in range(num_rounds):
                        result = trainer.train_round(round_num)
                        history.append({
                            "round": round_num,
                            "accuracy": result.global_acc,
                            "loss": result.global_loss,
                            "f1": result.global_f1,
                            "precision": result.global_precision,
                            "recall": result.global_recall,
                            "auc": result.global_auc,
                        })
                        final_result = {
                            "accuracy": result.global_acc,
                            "loss": result.global_loss,
                            "f1": result.global_f1,
                            "precision": result.global_precision,
                            "recall": result.global_recall,
                            "auc": result.global_auc,
                        }

                    algo_results.append({
                        "final": final_result,
                        "history": history,
                    })
                    algo_histories.append(history)

                    # Verify weights changed (proof of real training)
                    trained_weights = list(trainer.global_model.parameters())[0].data.flatten()[:5].tolist()
                    print(f"    {Style.SUCCESS}Completato{Colors.RESET}: Acc={final_result['accuracy']:.2%}, F1={final_result['f1']:.3f}, "
                          f"Prec={final_result['precision']:.3f}, Rec={final_result['recall']:.3f}, AUC={final_result['auc']:.3f}")
                    print(f"    Pesi finali (layer 0, primi 5): {[f'{w:.4f}' for w in trained_weights]}")

                # Calculate statistics and store history
                self.results[algorithm] = self._calculate_stats(algo_results)
                self.histories[algorithm] = self._average_history(algo_histories)

            elapsed = time.time() - start_time
            print()
            print_success(f"Confronto completato in {elapsed:.1f} secondi")

            self._show_comparison_table()

            # === AUTO-SAVE ALL OUTPUTS ===
            self._auto_save_all_outputs(elapsed)

        except ImportError as e:
            print_error(f"Impossibile importare il trainer: {e}")
        except Exception as e:
            print_error(f"Errore durante il confronto: {e}")
            import traceback
            traceback.print_exc()

        input(f"\n{Style.MUTED}Premi Enter per continuare...{Colors.RESET}")

    def _calculate_stats(self, results: List[Dict]) -> Dict[str, Any]:
        """Calculate mean and std for all metrics."""
        import numpy as np

        final_results = [r["final"] for r in results]
        histories = [r["history"] for r in results]

        metrics = {}
        for key in ["accuracy", "f1", "precision", "recall", "auc", "loss"]:
            values = [r.get(key, 0) for r in final_results if r]
            if values:
                metrics[key] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                }

        # Average convergence history
        if histories:
            avg_history = []
            num_rounds = len(histories[0])
            for r in range(num_rounds):
                entry = {"round": r}
                for key in ["accuracy", "loss", "f1", "precision", "recall", "auc"]:
                    values = [h[r].get(key, 0) for h in histories if len(h) > r]
                    if values:
                        entry[key] = float(np.mean(values))
                avg_history.append(entry)
            metrics["history"] = avg_history

        return metrics

    def _average_history(self, histories: List[List[Dict]]) -> List[Dict]:
        """Average multiple history runs with all metrics."""
        import numpy as np

        if not histories or not histories[0]:
            return []

        num_rounds = len(histories[0])
        avg_history = []

        for r in range(num_rounds):
            entry = {"round": r}
            for key in ["accuracy", "loss", "f1", "precision", "recall", "auc"]:
                values = [h[r].get(key, 0) for h in histories if len(h) > r]
                if values:
                    entry[key] = float(np.mean(values))
            avg_history.append(entry)

        return avg_history

    def _auto_save_all_outputs(self, elapsed_time: float):
        """Automatically save all outputs after comparison completes."""
        base_dir = Path(__file__).parent.parent.parent / "results"
        base_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create experiment-specific folder with descriptive name
        use_case_short = self.selected_use_case.id.replace("_", "")[:15] if self.selected_use_case else "unknown"
        dist_short = "IID" if self.config["is_iid"] else "NonIID"
        n_algos = len(self.config["algorithms"])
        folder_name = f"guided_{use_case_short}_{dist_short}_{n_algos}algos_{self.config['num_clients']}clients_{timestamp}"
        output_dir = base_dir / folder_name
        output_dir.mkdir(exist_ok=True)

        saved_files = []

        print()
        print_subsection("SALVATAGGIO AUTOMATICO RISULTATI")

        # Build full config with all specs
        full_specs = {
            "experiment_type": "guided_comparison",
            "use_case": {
                "id": self.selected_use_case.id,
                "name": self.selected_use_case.name,
                "description": self.selected_use_case.description,
            },
            "timestamp": timestamp,
            "elapsed_time_seconds": elapsed_time,
            "training_config": {
                "algorithms": self.config["algorithms"],
                "num_clients": self.config["num_clients"],
                "num_rounds": self.config["num_rounds"],
                "local_epochs": self.config["local_epochs"],
                "batch_size": self.config["batch_size"],
                "learning_rate": self.config["learning_rate"],
                "num_seeds": self.config["num_seeds"],
                "is_iid": self.config["is_iid"],
                "alpha": self.config["alpha"],
                "samples_per_client": 200,
                "dp_enabled": self.config.get("dp_enabled", False),
            },
            "model_config": {
                "architecture": "HealthcareMLP",
                "layers": "10 -> 64 -> 32 -> 2",
                "total_params": 2946,
                "optimizer": "SGD",
                "loss_function": "CrossEntropyLoss",
            },
            "dataset_config": {
                "type": "synthetic_healthcare",
                "features": ["age", "bmi", "bp_systolic", "glucose", "cholesterol",
                            "heart_rate", "resp_rate", "temperature", "oxygen_sat", "prev_conditions"],
                "target": "disease_risk_binary",
            },
        }

        if self.config.get("dp_enabled"):
            full_specs["training_config"]["dp_epsilon"] = self.config.get("dp_epsilon", 10.0)

        # 1. Save JSON results
        json_file = output_dir / "results.json"
        export_data = {
            "specs": full_specs,
            "results": self.results,
            "histories": self.histories if hasattr(self, 'histories') else {},
        }
        with open(json_file, "w") as f:
            json.dump(export_data, f, indent=2, default=str)
        saved_files.append(("JSON (Risultati)", json_file))

        # 2. Generate LaTeX table with specs
        latex_file = output_dir / "table_results.tex"
        latex_content = self._generate_latex_with_specs(full_specs)
        with open(latex_file, "w") as f:
            f.write(latex_content)
        saved_files.append(("LaTeX (Tabella)", latex_file))

        # 3. Generate convergence plots (all metrics)
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt
            import numpy as np

            colors = plt.cm.tab10(np.linspace(0, 1, len(self.histories)))
            use_case_name = self.selected_use_case.name if self.selected_use_case else "Unknown"

            # Plot 1: Accuracy and Loss
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            for (algo, history), color in zip(self.histories.items(), colors):
                if history:
                    rounds = [h["round"] + 1 for h in history]
                    accs = [h["accuracy"] for h in history]
                    losses = [h["loss"] for h in history]
                    axes[0].plot(rounds, accs, label=algo, color=color, linewidth=2)
                    axes[1].plot(rounds, losses, label=algo, color=color, linewidth=2)

            axes[0].set_xlabel("Round")
            axes[0].set_ylabel("Accuracy")
            axes[0].set_title(f"Convergence - {use_case_name}\n(Accuracy)")
            axes[0].legend(loc='lower right', fontsize=8)
            axes[0].grid(True, alpha=0.3)

            axes[1].set_xlabel("Round")
            axes[1].set_ylabel("Loss")
            axes[1].set_title(f"Convergence - {use_case_name}\n(Loss)")
            axes[1].legend(loc='upper right', fontsize=8)
            axes[1].grid(True, alpha=0.3)

            dist = "IID" if self.config["is_iid"] else f"Non-IID (alpha={self.config['alpha']})"
            spec_text = f"Distribution: {dist} | Seeds: {self.config['num_seeds']} | {self.config['num_clients']} clients"
            fig.text(0.5, 0.02, spec_text, ha='center', fontsize=9, style='italic')
            plt.tight_layout(rect=[0, 0.05, 1, 1])
            plot_file = output_dir / "plot_accuracy_loss.png"
            plt.savefig(plot_file, dpi=150, bbox_inches="tight")
            plt.close()
            saved_files.append(("PNG (Accuracy/Loss)", plot_file))

            # Plot 2: F1, Precision, Recall
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            for (algo, history), color in zip(self.histories.items(), colors):
                if history:
                    rounds = [h["round"] + 1 for h in history]
                    f1s = [h.get("f1", 0) for h in history]
                    precs = [h.get("precision", 0) for h in history]
                    recs = [h.get("recall", 0) for h in history]
                    axes[0].plot(rounds, f1s, label=algo, color=color, linewidth=2)
                    axes[1].plot(rounds, precs, label=algo, color=color, linewidth=2)
                    axes[2].plot(rounds, recs, label=algo, color=color, linewidth=2)

            for ax, title, ylabel in zip(axes, ["F1 Score", "Precision", "Recall"], ["F1", "Precision", "Recall"]):
                ax.set_xlabel("Round")
                ax.set_ylabel(ylabel)
                ax.set_title(f"{use_case_name} - {title}")
                ax.legend(loc='lower right', fontsize=7)
                ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plot_file = output_dir / "plot_f1_precision_recall.png"
            plt.savefig(plot_file, dpi=150, bbox_inches="tight")
            plt.close()
            saved_files.append(("PNG (F1/Prec/Rec)", plot_file))

            # Plot 3: AUC
            fig, ax = plt.subplots(figsize=(10, 5))
            for (algo, history), color in zip(self.histories.items(), colors):
                if history:
                    rounds = [h["round"] + 1 for h in history]
                    aucs = [h.get("auc", 0) for h in history]
                    ax.plot(rounds, aucs, label=algo, color=color, linewidth=2)

            ax.set_xlabel("Round")
            ax.set_ylabel("AUC")
            ax.set_title(f"{use_case_name} - AUC-ROC Convergence")
            ax.legend(loc='lower right', fontsize=8)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plot_file = output_dir / "plot_auc.png"
            plt.savefig(plot_file, dpi=150, bbox_inches="tight")
            plt.close()
            saved_files.append(("PNG (AUC)", plot_file))

            # Plot 4: Bar chart comparison
            fig, ax = plt.subplots(figsize=(12, 6))
            algos = list(self.results.keys())
            x = np.arange(len(algos))
            width = 0.15

            metrics_to_plot = ["accuracy", "f1", "precision", "recall", "auc"]
            for i, metric in enumerate(metrics_to_plot):
                values = [self.results[a].get(metric, {}).get("mean", 0) for a in algos]
                stds = [self.results[a].get(metric, {}).get("std", 0) for a in algos]
                ax.bar(x + i * width, values, width, label=metric.capitalize(), yerr=stds)

            ax.set_xlabel("Algorithm")
            ax.set_ylabel("Score")
            ax.set_title(f"{use_case_name} - Final Metrics Comparison")
            ax.set_xticks(x + width * 2)
            ax.set_xticklabels(algos, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            plot_file = output_dir / "plot_metrics_comparison.png"
            plt.savefig(plot_file, dpi=150, bbox_inches="tight")
            plt.close()
            saved_files.append(("PNG (Metrics Bar)", plot_file))

        except ImportError:
            print_warning("matplotlib non disponibile - grafici non generati")
        except Exception as e:
            print_warning(f"Errore generazione grafici: {e}")

        # 4. Generate CSV with history (all metrics)
        csv_file = output_dir / "history_all_metrics.csv"
        with open(csv_file, "w") as f:
            # Header with all metrics
            algos = list(self.histories.keys())
            metrics = ["acc", "loss", "f1", "prec", "rec", "auc"]
            header = "round," + ",".join([f"{a}_{m}" for a in algos for m in metrics])
            f.write(header + "\n")

            # Data rows
            if self.histories:
                first_history = list(self.histories.values())[0]
                for r in range(len(first_history)):
                    row = [str(r + 1)]
                    for algo in algos:
                        hist = self.histories.get(algo, [])
                        if r < len(hist):
                            h = hist[r]
                            row.extend([
                                f"{h.get('accuracy', 0):.4f}",
                                f"{h.get('loss', 0):.4f}",
                                f"{h.get('f1', 0):.4f}",
                                f"{h.get('precision', 0):.4f}",
                                f"{h.get('recall', 0):.4f}",
                                f"{h.get('auc', 0):.4f}",
                            ])
                        else:
                            row.extend([""] * 6)
                    f.write(",".join(row) + "\n")
        saved_files.append(("CSV (History)", csv_file))

        # 5. Generate summary CSV
        summary_file = output_dir / "summary_results.csv"
        with open(summary_file, "w") as f:
            f.write("algorithm,accuracy_mean,accuracy_std,f1_mean,f1_std,precision_mean,precision_std,recall_mean,recall_std,auc_mean,auc_std,loss_mean,loss_std\n")
            for algo, metrics in self.results.items():
                row = [algo]
                for m in ["accuracy", "f1", "precision", "recall", "auc", "loss"]:
                    row.append(f"{metrics.get(m, {}).get('mean', 0):.4f}")
                    row.append(f"{metrics.get(m, {}).get('std', 0):.4f}")
                f.write(",".join(row) + "\n")
        saved_files.append(("CSV (Summary)", summary_file))

        # === Show summary of saved files ===
        print()
        print(f"{Style.SUCCESS}=== RISULTATI SALVATI ==={Colors.RESET}")
        print(f"\nDirectory: {output_dir}\n")
        print(f"{Style.TITLE}{'Tipo':<25} {'File':<60}{Colors.RESET}")
        print("-" * 85)
        for file_type, file_path in saved_files:
            print(f"  {file_type:<23} {file_path.name}")
        print("-" * 85)
        print(f"\n{Style.SUCCESS}Totale: {len(saved_files)} file salvati{Colors.RESET}")

    def _generate_latex_with_specs(self, specs: Dict) -> str:
        """Generate LaTeX table with full training specifications."""
        lines = []
        lines.append("% FL-EHDS Guided Comparison Results")
        lines.append(f"% Use Case: {specs['use_case']['name']}")
        lines.append(f"% Generated: {specs['timestamp']}")
        lines.append(f"% Training time: {specs['elapsed_time_seconds']:.1f} seconds")
        lines.append("")
        lines.append(r"\begin{table}[htbp]")
        lines.append(r"\centering")
        lines.append(f"\\caption{{Algorithm Comparison for {specs['use_case']['name']}}}")
        lines.append(r"\label{tab:guided_comparison}")
        lines.append(r"\small")
        lines.append(r"\begin{tabular}{lccccc}")
        lines.append(r"\toprule")
        lines.append(r"\textbf{Algorithm} & \textbf{Accuracy} & \textbf{F1} & \textbf{Precision} & \textbf{Recall} & \textbf{AUC} \\")
        lines.append(r"\midrule")

        # Find best accuracy for highlighting
        best_acc = max(
            self.results[algo].get("accuracy", {}).get("mean", 0)
            for algo in self.results
        )

        for algo, metrics in self.results.items():
            acc = metrics.get("accuracy", {})
            f1 = metrics.get("f1", {})
            prec = metrics.get("precision", {})
            rec = metrics.get("recall", {})
            auc = metrics.get("auc", {})

            acc_mean = acc.get("mean", 0) * 100
            acc_std = acc.get("std", 0) * 100
            f1_mean = f1.get("mean", 0)
            f1_std = f1.get("std", 0)
            prec_mean = prec.get("mean", 0)
            prec_std = prec.get("std", 0)
            rec_mean = rec.get("mean", 0)
            rec_std = rec.get("std", 0)
            auc_mean = auc.get("mean", 0)
            auc_std = auc.get("std", 0)

            # Escape special chars
            safe_algo = algo.replace("_", r"\_").replace("&", r"\&")

            # Format metrics
            acc_str = f"{acc_mean:.1f}\\%$\\pm${acc_std:.1f}"
            f1_str = f"{f1_mean:.3f}$\\pm${f1_std:.3f}"
            prec_str = f"{prec_mean:.3f}$\\pm${prec_std:.3f}"
            rec_str = f"{rec_mean:.3f}$\\pm${rec_std:.3f}"
            auc_str = f"{auc_mean:.3f}$\\pm${auc_std:.3f}"

            # Bold if best
            if abs(acc.get("mean", 0) - best_acc) < 0.001:
                lines.append(f"\\textbf{{{safe_algo}}} & \\textbf{{{acc_str}}} & \\textbf{{{f1_str}}} & "
                            f"\\textbf{{{prec_str}}} & \\textbf{{{rec_str}}} & \\textbf{{{auc_str}}} \\\\")
            else:
                lines.append(f"{safe_algo} & {acc_str} & {f1_str} & {prec_str} & {rec_str} & {auc_str} \\\\")

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append("")
        lines.append(r"\vspace{2mm}")

        # Add full training specs
        tc = specs["training_config"]
        mc = specs["model_config"]
        lines.append(r"\begin{minipage}{\textwidth}")
        lines.append(r"\footnotesize")
        lines.append(f"\\textit{{Use Case: {specs['use_case']['name']}}} \\\\")
        lines.append(r"\textit{Training Configuration:} \\")
        lines.append(f"Clients: {tc['num_clients']} | "
                    f"Rounds: {tc['num_rounds']} | "
                    f"Local Epochs: {tc['local_epochs']} | "
                    f"Batch Size: {tc['batch_size']} | "
                    f"Learning Rate: {tc['learning_rate']} \\\\")
        dist = "IID" if tc['is_iid'] else f"Non-IID (alpha={tc['alpha']})"
        lines.append(f"Data Distribution: {dist} | "
                    f"Samples/Client: {tc['samples_per_client']} | "
                    f"Seeds: {tc['num_seeds']} \\\\")
        lines.append(f"Model: {mc['architecture']} ({mc['layers']}) | "
                    f"Optimizer: {mc['optimizer']} | "
                    f"Loss: {mc['loss_function']}")
        lines.append(r"\end{minipage}")
        lines.append(r"\end{table}")

        return "\n".join(lines)

    def _show_comparison_table(self):
        """Display comparison table with all metrics."""
        if not self.results:
            print_warning("Nessun risultato disponibile")
            return

        print_subsection("TABELLA COMPARATIVA")

        # Header
        header = f"{'Algoritmo':<15} {'Accuracy':<12} {'F1':<10} {'Precision':<10} {'Recall':<10} {'AUC':<10}"
        print(f"\n{Style.TITLE}{header}{Colors.RESET}")
        print("-" * 80)

        # Find best accuracy
        best_acc = max(
            self.results[algo].get("accuracy", {}).get("mean", 0)
            for algo in self.results
        )

        # Rows
        for algo, metrics in self.results.items():
            acc = metrics.get("accuracy", {})
            f1 = metrics.get("f1", {})
            prec = metrics.get("precision", {})
            rec = metrics.get("recall", {})
            auc = metrics.get("auc", {})

            acc_mean = acc.get("mean", 0)
            acc_str = f"{acc_mean:.1%}"
            f1_str = f"{f1.get('mean', 0):.3f}"
            prec_str = f"{prec.get('mean', 0):.3f}"
            rec_str = f"{rec.get('mean', 0):.3f}"
            auc_str = f"{auc.get('mean', 0):.3f}"

            # Highlight best
            if abs(acc_mean - best_acc) < 0.001:
                print(f"  {Style.SUCCESS}{algo:<13}{Colors.RESET} {acc_str:<10} {f1_str:<8} {prec_str:<8} {rec_str:<8} {auc_str:<8}")
            else:
                print(f"  {algo:<13} {acc_str:<10} {f1_str:<8} {prec_str:<8} {rec_str:<8} {auc_str:<8}")

        print("-" * 80)

        # Show standard deviations
        print(f"\n{Style.TITLE}Deviazioni Standard:{Colors.RESET}")
        print(f"{'Algoritmo':<15} {'Acc std':<10} {'F1 std':<10} {'Prec std':<10} {'Rec std':<10} {'AUC std':<10}")
        print("-" * 65)
        for algo, metrics in self.results.items():
            acc_std = metrics.get("accuracy", {}).get("std", 0)
            f1_std = metrics.get("f1", {}).get("std", 0)
            prec_std = metrics.get("precision", {}).get("std", 0)
            rec_std = metrics.get("recall", {}).get("std", 0)
            auc_std = metrics.get("auc", {}).get("std", 0)
            print(f"  {algo:<13} {acc_std:.4f}   {f1_std:.4f}   {prec_std:.4f}   {rec_std:.4f}   {auc_std:.4f}")

        # Recommendation
        best_algo = max(self.results.keys(),
                       key=lambda a: self.results[a].get("accuracy", {}).get("mean", 0))
        print(f"\n{Style.SUCCESS}Algoritmo consigliato per questo caso d'uso: {best_algo}{Colors.RESET}")

    def _show_results(self):
        """Show detailed results."""
        clear_screen()
        print_section("RISULTATI CONFRONTO")

        if not self.results:
            print_warning("Nessun risultato disponibile. Eseguire prima il confronto.")
            input(f"\n{Style.MUTED}Premi Enter per continuare...{Colors.RESET}")
            return

        if self.selected_use_case:
            print(f"{Style.TITLE}Caso d'uso:{Colors.RESET} {self.selected_use_case.name}")
            print()

        self._show_comparison_table()

        # Show convergence summary
        print_subsection("CONVERGENZA")
        for algo, metrics in self.results.items():
            history = metrics.get("history", [])
            if history:
                # Show accuracy at round 10, 20, final
                checkpoints = [10, 20, len(history) - 1]
                acc_values = []
                for cp in checkpoints:
                    if cp < len(history):
                        acc_values.append(f"R{cp + 1}:{history[cp]['accuracy']:.1%}")
                print(f"  {algo:<15} {' -> '.join(acc_values)}")

        input(f"\n{Style.MUTED}Premi Enter per continuare...{Colors.RESET}")

    def _generate_report(self):
        """Generate comparison report."""
        clear_screen()
        print_section("GENERA REPORT COMPARATIVO")

        if not self.results:
            print_warning("Nessun risultato da esportare")
            input(f"\n{Style.MUTED}Premi Enter per continuare...{Colors.RESET}")
            return

        from terminal.screens.output import OutputScreen
        from datetime import datetime

        output = OutputScreen()

        # Generate LaTeX table
        print_info("Generazione tabella LaTeX...")
        output.generate_latex_table(self.results, self.config)

        # Generate convergence plot
        if confirm("\nGenerare grafico convergenza?", default=True):
            self._generate_convergence_plot()

    def _generate_convergence_plot(self):
        """Generate convergence plot for all algorithms."""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            from datetime import datetime
            from pathlib import Path

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            colors = plt.cm.tab10(np.linspace(0, 1, len(self.results)))

            for (algo, metrics), color in zip(self.results.items(), colors):
                history = metrics.get("history", [])
                if history:
                    rounds = [h["round"] + 1 for h in history]
                    accs = [h["accuracy"] for h in history]
                    losses = [h["loss"] for h in history]

                    axes[0].plot(rounds, accs, label=algo, color=color, linewidth=2)
                    axes[1].plot(rounds, losses, label=algo, color=color, linewidth=2)

            axes[0].set_xlabel("Round")
            axes[0].set_ylabel("Accuracy")
            axes[0].set_title("Convergenza Accuracy")
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            axes[1].set_xlabel("Round")
            axes[1].set_ylabel("Loss")
            axes[1].set_title("Convergenza Loss")
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

            plt.tight_layout()

            # Save
            output_dir = Path(__file__).parent.parent.parent / "results"
            output_dir.mkdir(exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = output_dir / f"comparison_convergence_{timestamp}.png"
            plt.savefig(filename, dpi=150, bbox_inches="tight")
            plt.close()

            print_success(f"Grafico salvato: {filename}")

            # Also show in terminal if possible
            if confirm("\nAprire il grafico?", default=True):
                import subprocess
                subprocess.run(["open", str(filename)], check=False)

        except ImportError:
            print_error("matplotlib non disponibile per la generazione dei grafici")
            print_info("Installare con: pip install matplotlib")
        except Exception as e:
            print_error(f"Errore nella generazione del grafico: {e}")

"""
Algorithm comparison screen for FL-EHDS terminal interface.
Compares multiple FL algorithms with statistical analysis.
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
from terminal.progress import FLProgressBar, progress_bar
from terminal.validators import (
    get_int, get_float, get_bool, confirm, display_config_summary
)
from terminal.menu import Menu, MenuItem, MENU_STYLE
from terminal.training_utils import (
    FL_ALGORITHMS, get_available_datasets, build_dataset_choices,
    apply_dataset_selection, apply_dataset_parameters, select_imaging_model,
    configure_ehds_permit, create_trainer, setup_ehds_permit_context,
    calculate_stats, average_history, generate_comparison_latex,
)


class AlgorithmsScreen:
    """Algorithm comparison screen."""

    def __init__(self):
        self.config = self._default_config()
        self.results = {}
        self.histories = {}  # Store convergence history for all algorithms

    def _default_config(self) -> Dict[str, Any]:
        """Return default configuration from config.yaml with hardcoded fallbacks."""
        fallback = {
            "algorithms": FL_ALGORITHMS.copy(),
            "num_clients": 5,
            "num_rounds": 30,
            "local_epochs": 3,
            "batch_size": 32,
            "learning_rate": 0.01,
            "num_seeds": 3,
            "data_distribution": "Non-IID",
            "include_dp": False,
            "dp_epsilons": [1.0, 10.0],
            "dataset_type": "synthetic",
            "dataset_name": None,
            "dataset_path": None,
            "model_type": "resnet18",
            "freeze_backbone": False,
            "use_class_weights": True,
        }
        try:
            from config.config_loader import get_training_defaults
            yaml_defaults = get_training_defaults()
            # Map relevant keys (preserve algorithms list and comparison-specific keys)
            for key in ["num_clients", "num_rounds", "local_epochs", "batch_size",
                        "learning_rate", "dataset_type", "dataset_path", "seed"]:
                if key in yaml_defaults:
                    fallback[key] = yaml_defaults[key]
            if yaml_defaults.get("dp_enabled"):
                fallback["include_dp"] = True
        except (ImportError, Exception):
            pass
        return fallback

    def run(self):
        """Run the algorithms screen."""
        while True:
            clear_screen()
            print_section("CONFRONTO ALGORITMI FL")

            menu = Menu("Seleziona azione", [
                MenuItem("1", "Configura confronto", self._configure),
                MenuItem("2", "Esegui confronto", self._run_comparison),
                MenuItem("3", "Visualizza risultati", self._show_results),
                MenuItem("4", "Genera tabella comparativa", self._generate_table),
                MenuItem("5", "Esporta risultati", self._export_results),
                MenuItem("0", "Torna al menu principale", lambda: "back"),
            ])

            result = menu.display()
            if result is None:
                break

            if result.handler:
                handler_result = result.handler()
                if handler_result == "back":
                    break

    def _configure(self):
        """Configure comparison parameters."""
        clear_screen()
        print_section("CONFIGURAZIONE CONFRONTO")

        # Algorithm selection
        print_subsection("Algoritmi da confrontare")
        if HAS_QUESTIONARY:
            selected = questionary.checkbox(
                "Seleziona algoritmi (Spazio per selezionare, Enter per confermare):",
                choices=[
                    questionary.Choice(alg, checked=(alg in self.config["algorithms"]))
                    for alg in FL_ALGORITHMS
                ],
                style=MENU_STYLE,
            ).ask()
            if selected:
                self.config["algorithms"] = selected
        else:
            print("Algoritmi disponibili:")
            for i, alg in enumerate(FL_ALGORITHMS, 1):
                selected = "*" if alg in self.config["algorithms"] else " "
                print(f"  {selected} {i}. {alg}")
            print_info("Modifica manualmente config['algorithms'] per cambiare selezione")

        # Dataset selection
        print_subsection("Selezione Dataset")
        available_datasets = get_available_datasets()
        dataset_choices, dataset_map = build_dataset_choices(available_datasets)

        if HAS_QUESTIONARY:
            selected = questionary.select(
                "Seleziona dataset:",
                choices=dataset_choices,
                style=MENU_STYLE,
            ).ask() or dataset_choices[0]
        else:
            print("Dataset disponibili:")
            for i, choice in enumerate(dataset_choices):
                print(f"  {i + 1}. {choice}")
            idx = get_int("Selezione", default=1, min_val=1, max_val=len(dataset_choices)) - 1
            selected = dataset_choices[idx]

        apply_dataset_selection(self.config, dataset_map[selected], available_datasets)
        apply_dataset_parameters(self.config)
        select_imaging_model(self.config)

        # Basic parameters
        print_subsection("Parametri Training")
        self.config["num_clients"] = get_int("Numero client", default=self.config["num_clients"], min_val=2, max_val=50)
        self.config["num_rounds"] = get_int("Numero round", default=self.config["num_rounds"], min_val=10, max_val=200)
        self.config["local_epochs"] = get_int("Epoche locali", default=self.config["local_epochs"], min_val=1, max_val=10)
        self.config["batch_size"] = get_int("Batch size", default=self.config["batch_size"], min_val=8, max_val=256)
        self.config["learning_rate"] = get_float("Learning rate", default=self.config["learning_rate"], min_val=0.001, max_val=0.1)

        # Statistical settings
        print_subsection("Impostazioni Statistiche")
        self.config["num_seeds"] = get_int(
            "Numero di run (per std dev)",
            default=self.config["num_seeds"],
            min_val=1, max_val=10
        )

        # DP options
        print_subsection("Differential Privacy")
        self.config["include_dp"] = get_bool(
            "Includere varianti con DP?",
            default=self.config["include_dp"]
        )

        # EHDS Data Permit (optional)
        configure_ehds_permit(self.config)

        display_config_summary(self.config)
        input(f"\n{Style.MUTED}Premi Enter per continuare...{Colors.RESET}")

    def _run_comparison(self):
        """Execute algorithm comparison."""
        clear_screen()
        print_section("CONFRONTO IN CORSO")

        display_config_summary(self.config)

        # Ask for verbose mode
        verbose = confirm("\nMostrare progresso dettagliato (round per round)?", default=True)

        if not confirm("\nAvviare il confronto?", default=True):
            return

        print()

        try:
            self.results = {}
            self.histories = {}  # Store convergence history for plots
            total_runs = len(self.config["algorithms"]) * self.config["num_seeds"]

            if self.config["include_dp"]:
                total_runs += len(self.config["dp_epsilons"]) * self.config["num_seeds"]

            print_info(f"Totale run da eseguire: {total_runs}")
            print_info(f"Per ogni run: {self.config['num_rounds']} round x {self.config['num_clients']} client x {self.config['local_epochs']} epoche")
            print()

            run_count = 0
            start_time = time.time()

            # EHDS Permit context (if enabled)
            self._permit_context = setup_ehds_permit_context(self.config)
            if self._permit_context:
                print()

            # Progress callback for verbose mode
            def make_progress_callback(algorithm_name, seed_num):
                def progress_callback(event_type, **kwargs):
                    if not verbose:
                        return
                    if event_type == "round_start":
                        r = kwargs.get("round_num", 0)
                        print(f"      Round {r}/{self.config['num_rounds']}: ", end="", flush=True)
                    elif event_type == "client_end":
                        client_id = kwargs.get("client_id", 0)
                        print(f"C{client_id} ", end="", flush=True)
                    elif event_type == "round_end":
                        acc = kwargs.get("acc", 0)
                        print(f"-> acc={acc:.2%}")
                return progress_callback

            # Run each algorithm
            for algorithm in self.config["algorithms"]:
                print(f"\n{Style.TITLE}===== Testing {algorithm} ====={Colors.RESET}")

                algo_results = []
                algo_histories = []

                for seed in range(self.config["num_seeds"]):
                    run_count += 1
                    print(f"  Run {seed + 1}/{self.config['num_seeds']} (seed={seed})")

                    progress_cb = make_progress_callback(algorithm, seed) if verbose else None

                    trainer, _ = create_trainer(
                        config=self.config, algorithm=algorithm, seed=seed,
                        progress_callback=progress_cb, verbose=False,
                    )

                    # Show data distribution for first run
                    if seed == 0:
                        stats = trainer.get_client_data_stats()
                        print(f"    Distribuzione dati per client:")
                        for cid, cstats in stats.items():
                            print(f"      Client {cid}: {cstats['num_samples']} samples, labels={cstats['label_distribution']}")

                    # Run all rounds and collect history with all metrics
                    history = []
                    final_result = None
                    for round_num in range(self.config["num_rounds"]):
                        # Pre-round permit check
                        if self._permit_context:
                            ok, reason = self._permit_context.validate_round(round_num)
                            if not ok:
                                print_warning(f"    Training interrotto: {reason}")
                                break

                        result = trainer.train_round(round_num)

                        # Post-round audit
                        if self._permit_context:
                            self._permit_context.log_round_completion(result)

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

                    if final_result is None:
                        print_warning(f"    Run {seed} skipped (permit expired or no rounds completed)")
                        continue

                    algo_results.append(final_result)
                    algo_histories.append(history)

                    # Verify weights changed (proof of real training)
                    trained_weights = list(trainer.global_model.parameters())[0].data.flatten()[:5].tolist()
                    print(f"    {Style.SUCCESS}Completato{Colors.RESET}: Acc={final_result['accuracy']:.2%}, F1={final_result['f1']:.3f}, "
                          f"Prec={final_result['precision']:.3f}, Rec={final_result['recall']:.3f}, AUC={final_result['auc']:.3f}")
                    print(f"    Pesi finali (layer 0, primi 5): {[f'{w:.4f}' for w in trained_weights]}")

                # Calculate statistics and store history
                if algo_results:
                    self.results[algorithm] = calculate_stats(algo_results)
                    self.histories[algorithm] = average_history(algo_histories)
                else:
                    print_warning(f"  Nessun risultato valido per {algorithm} (tutti i seed falliti)")
                    self.results[algorithm] = {}
                    self.histories[algorithm] = []

            # DP variants if enabled
            if self.config["include_dp"]:
                for epsilon in self.config["dp_epsilons"]:
                    algo_name = f"FedAvg + DP (e={epsilon})"
                    print(f"\n{Style.TITLE}===== Testing {algo_name} ====={Colors.RESET}")

                    dp_results = []
                    dp_histories = []

                    for seed in range(self.config["num_seeds"]):
                        run_count += 1
                        print(f"  Run {seed + 1}/{self.config['num_seeds']} (seed={seed})")

                        progress_cb = make_progress_callback(algo_name, seed) if verbose else None

                        trainer, _ = create_trainer(
                            config=self.config, algorithm="FedAvg", seed=seed,
                            progress_callback=progress_cb,
                            dp_enabled=True, dp_epsilon=epsilon, verbose=False,
                        )

                        history = []
                        final_result = None
                        for round_num in range(self.config["num_rounds"]):
                            # Pre-round permit check
                            if self._permit_context:
                                eps_cost = epsilon / self.config["num_rounds"]
                                ok, reason = self._permit_context.validate_round(round_num, eps_cost)
                                if not ok:
                                    print_warning(f"    Training interrotto: {reason}")
                                    break

                            result = trainer.train_round(round_num)

                            # Post-round audit
                            if self._permit_context:
                                eps_cost = epsilon / self.config["num_rounds"]
                                self._permit_context.log_round_completion(result, eps_cost)

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

                        if final_result is None:
                            print_warning(f"    Run {seed} skipped (permit expired or no rounds completed)")
                            continue

                        dp_results.append(final_result)
                        dp_histories.append(history)
                        print(f"    {Style.SUCCESS}Completato{Colors.RESET}: Acc={final_result['accuracy']:.2%}, F1={final_result['f1']:.3f}, "
                              f"Prec={final_result['precision']:.3f}, Rec={final_result['recall']:.3f}, AUC={final_result['auc']:.3f}")

                    if dp_results:
                        self.results[algo_name] = calculate_stats(dp_results)
                        self.histories[algo_name] = average_history(dp_histories)
                    else:
                        print_warning(f"  Nessun risultato valido per {algo_name} (tutti i seed falliti)")
                        self.results[algo_name] = {}
                        self.histories[algo_name] = []

            elapsed = time.time() - start_time

            # End EHDS permit session
            if self._permit_context:
                final_metrics = {}
                if self.results:
                    first_algo = next(iter(self.results.values()), {})
                    final_metrics = {k: v.get("mean", 0) for k, v in first_algo.items() if isinstance(v, dict)}
                self._permit_context.end_session(
                    total_rounds=self.config["num_rounds"] * total_runs,
                    final_metrics=final_metrics,
                    success=True,
                )
                budget = self._permit_context.get_budget_status()
                print()
                print_info(f"EHDS Budget: {budget['used']:.4f}/{budget['total']:.4f} epsilon "
                           f"({budget['utilization_pct']:.1f}%)")

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

    def _auto_save_all_outputs(self, elapsed_time: float):
        """Automatically save all outputs after comparison completes."""
        base_dir = Path(__file__).parent.parent.parent / "results"
        base_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create experiment-specific folder with descriptive name
        dist_short = ("OMOP" if self.config.get("dataset_type") == "omop" else
            "FHIR" if self.config.get("dataset_type") == "fhir" else
            "Diabetes" if self.config.get("dataset_type") == "diabetes" else
            "Heart" if self.config.get("dataset_type") == "heart_disease" else
            ("IID" if self.config["data_distribution"] == "IID" else "NonIID"))
        n_algos = len(self.config["algorithms"])
        folder_name = f"comparison_{dist_short}_{n_algos}algos_{self.config['num_clients']}clients_{self.config['num_rounds']}rounds_{timestamp}"
        output_dir = base_dir / folder_name
        output_dir.mkdir(exist_ok=True)

        saved_files = []

        print()
        print_subsection("SALVATAGGIO AUTOMATICO RISULTATI")

        # Build full config with all specs
        full_specs = {
            "experiment_type": "algorithm_comparison",
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
                "data_distribution": self.config["data_distribution"],
                "samples_per_client": 200,
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
            "ehds_governance": (
                self._permit_context.get_summary_for_specs()
                if hasattr(self, '_permit_context') and self._permit_context else None
            ),
            "include_dp": self.config["include_dp"],
        }

        if self.config["include_dp"]:
            full_specs["dp_epsilons"] = self.config["dp_epsilons"]

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
        latex_content = generate_comparison_latex(self.results, full_specs)
        with open(latex_file, "w") as f:
            f.write(latex_content)
        saved_files.append(("LaTeX (Tabella)", latex_file))

        # 3. Generate convergence plots (multiple: Accuracy, Loss, F1, AUC)
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt
            import numpy as np

            colors = plt.cm.tab10(np.linspace(0, 1, len(self.histories)))

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
            axes[0].set_title(f"Convergence - Accuracy\n({self.config['num_clients']} clients, {self.config['num_rounds']} rounds)")
            axes[0].legend(loc='lower right', fontsize=8)
            axes[0].grid(True, alpha=0.3)

            axes[1].set_xlabel("Round")
            axes[1].set_ylabel("Loss")
            axes[1].set_title(f"Convergence - Loss\n({self.config['local_epochs']} local epochs, lr={self.config['learning_rate']})")
            axes[1].legend(loc='upper right', fontsize=8)
            axes[1].grid(True, alpha=0.3)

            spec_text = f"Distribution: {self.config['data_distribution']} | Seeds: {self.config['num_seeds']} | Model: HealthcareMLP"
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
                ax.set_title(f"Convergence - {title}")
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
            ax.set_title(f"Convergence - AUC-ROC\n({self.config['num_clients']} clients, {self.config['num_rounds']} rounds)")
            ax.legend(loc='lower right', fontsize=8)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plot_file = output_dir / "plot_auc.png"
            plt.savefig(plot_file, dpi=150, bbox_inches="tight")
            plt.close()
            saved_files.append(("PNG (AUC)", plot_file))

            # Plot 4: Bar chart comparison of final metrics
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
            ax.set_title("Final Metrics Comparison")
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

        # 6. EHDS Audit Log (if permit enabled)
        if self.config.get("ehds_permit_enabled") and hasattr(self, '_permit_context') and self._permit_context:
            try:
                audit_file = self._permit_context.export_audit_log(output_dir)
                saved_files.append(("JSON (EHDS Audit)", audit_file))
            except Exception as e:
                print_warning(f"Errore salvataggio audit log: {e}")

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

    def _show_comparison_table(self):
        """Display comparison table with all metrics."""
        if not self.results:
            print_warning("Nessun risultato disponibile")
            return

        print_subsection("TABELLA COMPARATIVA")

        # Header
        header = f"{'Algoritmo':<18} {'Accuracy':<14} {'F1':<10} {'Precision':<10} {'Recall':<10} {'AUC':<10}"
        print(f"\n{Style.TITLE}{header}{Colors.RESET}")
        print("-" * 85)

        # Find best accuracy for highlighting
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

            acc_str = f"{acc.get('mean', 0):.1%}"
            f1_str = f"{f1.get('mean', 0):.3f}"
            prec_str = f"{prec.get('mean', 0):.3f}"
            rec_str = f"{rec.get('mean', 0):.3f}"
            auc_str = f"{auc.get('mean', 0):.3f}"

            # Highlight best
            if abs(acc.get("mean", 0) - best_acc) < 0.001:
                print(f"  {Style.SUCCESS}{algo:<16}{Colors.RESET} {acc_str:<12} {f1_str:<8} {prec_str:<8} {rec_str:<8} {auc_str:<8}")
            else:
                print(f"  {algo:<16} {acc_str:<12} {f1_str:<8} {prec_str:<8} {rec_str:<8} {auc_str:<8}")

        print("-" * 85)

        # Second table with standard deviations
        print(f"\n{Style.TITLE}Deviazioni Standard:{Colors.RESET}")
        print(f"{'Algoritmo':<18} {'Acc std':<10} {'F1 std':<10} {'Prec std':<10} {'Rec std':<10} {'AUC std':<10}")
        print("-" * 68)
        for algo, metrics in self.results.items():
            acc_std = metrics.get("accuracy", {}).get("std", 0)
            f1_std = metrics.get("f1", {}).get("std", 0)
            prec_std = metrics.get("precision", {}).get("std", 0)
            rec_std = metrics.get("recall", {}).get("std", 0)
            auc_std = metrics.get("auc", {}).get("std", 0)
            print(f"  {algo:<16} {acc_std:.4f}   {f1_std:.4f}   {prec_std:.4f}   {rec_std:.4f}   {auc_std:.4f}")

        print(f"\n{Style.MUTED}Risultati su {self.config['num_seeds']} run, "
              f"{self.config['num_clients']} client, {self.config['num_rounds']} round{Colors.RESET}")

    def _show_results(self):
        """Show detailed results."""
        clear_screen()
        print_section("RISULTATI CONFRONTO")

        if not self.results:
            print_warning("Nessun risultato disponibile. Eseguire prima il confronto.")
            input(f"\n{Style.MUTED}Premi Enter per continuare...{Colors.RESET}")
            return

        self._show_comparison_table()
        input(f"\n{Style.MUTED}Premi Enter per continuare...{Colors.RESET}")

    def _generate_table(self):
        """Generate LaTeX table."""
        if not self.results:
            print_warning("Nessun risultato disponibile")
            input(f"\n{Style.MUTED}Premi Enter per continuare...{Colors.RESET}")
            return

        from terminal.screens.output import OutputScreen
        output = OutputScreen()
        output.generate_latex_table(self.results, self.config)

    def _export_results(self):
        """Export results to file."""
        if not self.results:
            print_warning("Nessun risultato da esportare")
            input(f"\n{Style.MUTED}Premi Enter per continuare...{Colors.RESET}")
            return

        from terminal.screens.output import OutputScreen
        output = OutputScreen()
        output.export_comparison_results(self.results, self.config)

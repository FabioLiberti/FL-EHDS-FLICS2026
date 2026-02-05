"""
Privacy analysis screen for FL-EHDS terminal interface.
Provides RDP accounting, epsilon estimation, and privacy-utility tradeoff analysis.
"""

import sys
from pathlib import Path
from typing import Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from terminal.colors import (
    Colors, Style, print_section, print_subsection,
    print_success, print_error, print_info, print_warning, clear_screen
)
from terminal.validators import get_int, get_float, get_bool, display_config_summary
from terminal.menu import Menu, MenuItem


class PrivacyScreen:
    """Privacy analysis screen with RDP accounting."""

    def __init__(self):
        self.config = self._default_config()

    def _default_config(self) -> Dict[str, Any]:
        """Return default configuration."""
        return {
            "num_rounds": 100,
            "noise_multiplier": 1.0,
            "sampling_rate": 0.01,
            "delta": 1e-5,
            "target_epsilon": 1.0,
        }

    def run(self):
        """Run the privacy screen."""
        while True:
            clear_screen()
            print_section("ANALISI PRIVACY (RDP)")

            menu = Menu("Seleziona analisi", [
                MenuItem("1", "Calcola epsilon per N round", self._compute_epsilon),
                MenuItem("2", "Calcola round massimi per target epsilon", self._compute_max_rounds),
                MenuItem("3", "Calcola noise per target epsilon", self._compute_noise),
                MenuItem("4", "Confronto RDP vs Composizione Semplice", self._compare_composition),
                MenuItem("5", "Analisi Privacy-Utility Tradeoff", self._privacy_utility_tradeoff),
                MenuItem("0", "Torna al menu principale", lambda: "back"),
            ])

            result = menu.display()
            if result is None:
                break

            if result.handler:
                handler_result = result.handler()
                if handler_result == "back":
                    break

    def _compute_epsilon(self):
        """Compute epsilon for given number of rounds."""
        clear_screen()
        print_section("CALCOLO EPSILON")

        print_subsection("Parametri")
        num_rounds = get_int("Numero di round", default=self.config["num_rounds"], min_val=1, max_val=10000)
        noise_multiplier = get_float("Noise multiplier (sigma/sensitivity)", default=self.config["noise_multiplier"], min_val=0.1, max_val=100)
        sampling_rate = get_float("Sampling rate (q)", default=self.config["sampling_rate"], min_val=0.001, max_val=1.0)
        delta = get_float("Delta", default=self.config["delta"], min_val=1e-10, max_val=1e-3)

        try:
            from orchestration.privacy.differential_privacy import (
                PrivacyAccountant, compute_rdp_gaussian, compute_rdp_gaussian_subsampled,
                rdp_to_eps_delta, DEFAULT_RDP_ORDERS
            )

            # Create accountant for prospective calculation
            accountant = PrivacyAccountant(
                total_epsilon=float('inf'),
                total_delta=delta,
                accountant_type="rdp"
            )

            # Compute epsilon using RDP
            epsilon_rdp = accountant.compute_epsilon_for_rounds(
                num_rounds=num_rounds,
                noise_multiplier=noise_multiplier,
                sampling_rate=sampling_rate
            )

            # Simple composition for comparison
            import numpy as np
            per_round_epsilon = np.sqrt(2 * np.log(1.25 / delta)) / noise_multiplier
            epsilon_simple = per_round_epsilon * num_rounds

            # Display results
            print_subsection("RISULTATI")

            print(f"\n{Style.TITLE}Metodo di Composizione          Epsilon Totale{Colors.RESET}")
            print("-" * 50)
            print(f"  RDP (Renyi DP):                {Style.SUCCESS}{epsilon_rdp:.4f}{Colors.RESET}")
            print(f"  Composizione Semplice:         {Style.WARNING}{epsilon_simple:.4f}{Colors.RESET}")

            improvement = epsilon_simple / epsilon_rdp if epsilon_rdp > 0 else 0
            print(f"\n{Style.INFO}Miglioramento RDP: {improvement:.1f}x piu tight{Colors.RESET}")

            print(f"\n{Style.MUTED}Parametri utilizzati:{Colors.RESET}")
            print(f"  Round: {num_rounds}")
            print(f"  Noise multiplier: {noise_multiplier}")
            print(f"  Sampling rate: {sampling_rate}")
            print(f"  Delta: {delta:.2e}")

        except ImportError as e:
            print_error(f"Impossibile importare il modulo privacy: {e}")
        except Exception as e:
            print_error(f"Errore durante il calcolo: {e}")

        input(f"\n{Style.MUTED}Premi Enter per continuare...{Colors.RESET}")

    def _compute_max_rounds(self):
        """Compute maximum rounds for target epsilon."""
        clear_screen()
        print_section("CALCOLO ROUND MASSIMI")

        print_subsection("Parametri")
        target_epsilon = get_float("Epsilon target", default=self.config["target_epsilon"], min_val=0.1, max_val=100)
        noise_multiplier = get_float("Noise multiplier", default=self.config["noise_multiplier"], min_val=0.1, max_val=100)
        sampling_rate = get_float("Sampling rate", default=self.config["sampling_rate"], min_val=0.001, max_val=1.0)
        delta = get_float("Delta", default=self.config["delta"], min_val=1e-10, max_val=1e-3)

        try:
            from orchestration.privacy.differential_privacy import DifferentialPrivacy

            dp = DifferentialPrivacy(
                epsilon=1.0,  # Per-round (will be computed)
                delta=delta,
                max_grad_norm=1.0,
                sampling_rate=sampling_rate,
            )

            # Override noise scale for our noise_multiplier
            dp._noise_multiplier = noise_multiplier

            max_rounds_rdp = dp.recommend_rounds(target_epsilon, use_rdp=True)
            max_rounds_simple = dp.recommend_rounds(target_epsilon, use_rdp=False)

            print_subsection("RISULTATI")

            print(f"\n{Style.TITLE}Per epsilon target = {target_epsilon}:{Colors.RESET}")
            print("-" * 50)
            print(f"  Round massimi (RDP):           {Style.SUCCESS}{max_rounds_rdp}{Colors.RESET}")
            print(f"  Round massimi (Semplice):      {Style.WARNING}{max_rounds_simple}{Colors.RESET}")

            improvement = max_rounds_rdp / max_rounds_simple if max_rounds_simple > 0 else 0
            print(f"\n{Style.INFO}Con RDP puoi fare {improvement:.1f}x piu round{Colors.RESET}")

        except ImportError as e:
            print_error(f"Impossibile importare il modulo privacy: {e}")
        except Exception as e:
            print_error(f"Errore durante il calcolo: {e}")

        input(f"\n{Style.MUTED}Premi Enter per continuare...{Colors.RESET}")

    def _compute_noise(self):
        """Compute required noise for target epsilon."""
        clear_screen()
        print_section("CALCOLO NOISE RICHIESTO")

        print_subsection("Parametri")
        target_epsilon = get_float("Epsilon target", default=self.config["target_epsilon"], min_val=0.1, max_val=100)
        num_rounds = get_int("Numero di round", default=self.config["num_rounds"], min_val=1, max_val=10000)
        sampling_rate = get_float("Sampling rate", default=self.config["sampling_rate"], min_val=0.001, max_val=1.0)
        delta = get_float("Delta", default=self.config["delta"], min_val=1e-10, max_val=1e-3)

        try:
            from orchestration.privacy.differential_privacy import PrivacyAccountant

            accountant = PrivacyAccountant(
                total_epsilon=target_epsilon,
                total_delta=delta,
                accountant_type="rdp"
            )

            required_noise = accountant.compute_noise_for_target_epsilon(
                target_epsilon=target_epsilon,
                num_rounds=num_rounds,
                sampling_rate=sampling_rate
            )

            print_subsection("RISULTATI")

            print(f"\n{Style.TITLE}Per {num_rounds} round con epsilon = {target_epsilon}:{Colors.RESET}")
            print("-" * 50)
            print(f"  Noise multiplier richiesto:    {Style.SUCCESS}{required_noise:.4f}{Colors.RESET}")

            print(f"\n{Style.MUTED}Questo significa:{Colors.RESET}")
            print(f"  sigma = {required_noise:.4f} * sensitivity")
            print(f"  dove sensitivity = max gradient norm (tipicamente 1.0)")

        except ImportError as e:
            print_error(f"Impossibile importare il modulo privacy: {e}")
        except Exception as e:
            print_error(f"Errore durante il calcolo: {e}")

        input(f"\n{Style.MUTED}Premi Enter per continuare...{Colors.RESET}")

    def _compare_composition(self):
        """Compare RDP vs simple composition."""
        clear_screen()
        print_section("CONFRONTO METODI DI COMPOSIZIONE")

        print_subsection("Parametri")
        noise_multiplier = get_float("Noise multiplier", default=self.config["noise_multiplier"], min_val=0.1, max_val=100)
        sampling_rate = get_float("Sampling rate", default=self.config["sampling_rate"], min_val=0.001, max_val=1.0)
        delta = get_float("Delta", default=self.config["delta"], min_val=1e-10, max_val=1e-3)

        try:
            from orchestration.privacy.differential_privacy import PrivacyAccountant
            import numpy as np

            round_counts = [10, 30, 50, 100, 200, 500, 1000]

            print_subsection("RISULTATI")

            print(f"\n{Style.TITLE}{'Round':<10} {'RDP Epsilon':<15} {'Simple Epsilon':<18} {'Improvement':<12}{Colors.RESET}")
            print("-" * 60)

            for rounds in round_counts:
                accountant = PrivacyAccountant(
                    total_epsilon=float('inf'),
                    total_delta=delta,
                    accountant_type="rdp"
                )

                eps_rdp = accountant.compute_epsilon_for_rounds(rounds, noise_multiplier, sampling_rate)

                per_round = np.sqrt(2 * np.log(1.25 / delta)) / noise_multiplier
                eps_simple = per_round * rounds

                improvement = eps_simple / eps_rdp if eps_rdp > 0 else 0

                print(f"  {rounds:<8} {eps_rdp:<15.4f} {eps_simple:<18.4f} {improvement:<12.1f}x")

            print("-" * 60)
            print(f"\n{Style.INFO}Nota: RDP diventa piu vantaggioso con piu round{Colors.RESET}")

        except ImportError as e:
            print_error(f"Impossibile importare il modulo privacy: {e}")
        except Exception as e:
            print_error(f"Errore durante il calcolo: {e}")

        input(f"\n{Style.MUTED}Premi Enter per continuare...{Colors.RESET}")

    def _privacy_utility_tradeoff(self):
        """Analyze privacy-utility tradeoff."""
        clear_screen()
        print_section("ANALISI PRIVACY-UTILITY TRADEOFF")

        print_info("Questa analisi mostra come l'accuracy varia al variare di epsilon")
        print()

        print_subsection("Risultati Benchmark (Riferimento)")

        # Reference results from benchmark
        results = [
            ("FedAvg (no DP)", "inf", "60.9%", "0.61", "0.66"),
            ("FedAvg + DP", "10.0", "55.7%", "0.61", "0.55"),
            ("FedAvg + DP", "1.0", "55.1%", "0.59", "0.55"),
        ]

        print(f"\n{Style.TITLE}{'Configurazione':<20} {'Epsilon':<10} {'Accuracy':<12} {'F1':<10} {'AUC':<10}{Colors.RESET}")
        print("-" * 62)

        for config, eps, acc, f1, auc in results:
            print(f"  {config:<18} {eps:<10} {acc:<12} {f1:<10} {auc:<10}")

        print("-" * 62)

        print(f"\n{Style.WARNING}Osservazioni:{Colors.RESET}")
        print("  - Passando da epsilon=inf a epsilon=10: -5.2pp accuracy")
        print("  - Passando da epsilon=10 a epsilon=1: -0.6pp accuracy")
        print("  - AUC degrada significativamente con DP (0.66 -> 0.55)")

        print(f"\n{Style.INFO}Per EHDS si raccomanda epsilon <= 1.0 per dati sanitari sensibili{Colors.RESET}")

        input(f"\n{Style.MUTED}Premi Enter per continuare...{Colors.RESET}")

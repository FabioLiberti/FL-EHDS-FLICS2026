"""
Entry point for centralized vs federated comparison experiment.

Usage:
    cd fl-ehds-framework
    python -m experiments.centralized_vs_federated
    python -m experiments.centralized_vs_federated --dataset chest_xray --quick
"""

from experiments.centralized_vs_federated.run_comparison import main

if __name__ == "__main__":
    main()

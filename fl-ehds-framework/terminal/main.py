#!/usr/bin/env python3
"""
FL-EHDS Terminal Interface - Main Entry Point
==============================================
Command-line interface for federated learning experiments.

Usage:
    python -m terminal.main

Or with the fl-ehds command (if installed):
    fl-ehds
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from terminal.menu import MainMenu
from terminal.colors import Colors, print_header, print_error


def main():
    """Main entry point for FL-EHDS terminal interface."""
    try:
        # Clear screen and show header
        print("\033[2J\033[H", end="")
        print_header()

        # Start main menu loop
        menu = MainMenu()
        menu.run()

    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}Interruzione rilevata. Uscita...{Colors.RESET}")
        sys.exit(0)
    except Exception as e:
        print_error(f"Errore fatale: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()

"""
Color scheme and styling utilities for FL-EHDS terminal interface.
Uses ANSI escape codes for cross-platform compatibility.
"""

import os
import sys


class Colors:
    """ANSI color codes for terminal output."""

    # Basic colors
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    UNDERLINE = "\033[4m"

    # Foreground colors
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    # Bright foreground colors
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"

    # Background colors
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"

    @classmethod
    def disable(cls):
        """Disable colors (for non-terminal output)."""
        for attr in dir(cls):
            if not attr.startswith('_') and attr.isupper():
                setattr(cls, attr, "")

    @classmethod
    def enable_if_terminal(cls):
        """Enable colors only if running in a terminal."""
        if not sys.stdout.isatty():
            cls.disable()


# Check for color support
if os.environ.get("NO_COLOR") or not sys.stdout.isatty():
    Colors.disable()


# Semantic color aliases
class Style:
    """Semantic styles for consistent UI."""

    # Status indicators
    SUCCESS = Colors.GREEN
    WARNING = Colors.YELLOW
    ERROR = Colors.RED
    INFO = Colors.CYAN

    # UI elements
    HEADER = Colors.BOLD + Colors.CYAN
    TITLE = Colors.BOLD + Colors.WHITE
    SUBTITLE = Colors.DIM + Colors.WHITE
    HIGHLIGHT = Colors.BOLD + Colors.YELLOW
    MUTED = Colors.DIM

    # Progress colors
    PROGRESS_BAR = Colors.GREEN
    PROGRESS_TEXT = Colors.CYAN

    # Table colors
    TABLE_HEADER = Colors.BOLD + Colors.CYAN
    TABLE_ROW = Colors.WHITE
    TABLE_HIGHLIGHT = Colors.BRIGHT_GREEN

    # Menu colors
    MENU_SELECTED = Colors.BOLD + Colors.CYAN
    MENU_NORMAL = Colors.WHITE
    MENU_DISABLED = Colors.DIM


def print_header():
    """Print the application header."""
    header = f"""
{Style.HEADER}================================================================================
                         FL-EHDS FRAMEWORK v4.0
              Privacy-Preserving Federated Learning for EHDS
================================================================================{Colors.RESET}
"""
    print(header)


def print_section(title: str):
    """Print a section title."""
    width = 80
    line = "=" * width
    print(f"\n{Style.TITLE}{line}")
    print(f"{title.center(width)}")
    print(f"{line}{Colors.RESET}\n")


def print_subsection(title: str):
    """Print a subsection title."""
    print(f"\n{Style.SUBTITLE}--- {title} ---{Colors.RESET}\n")


def print_success(message: str):
    """Print a success message."""
    print(f"{Style.SUCCESS}[OK] {message}{Colors.RESET}")


def print_warning(message: str):
    """Print a warning message."""
    print(f"{Style.WARNING}[!] {message}{Colors.RESET}")


def print_error(message: str):
    """Print an error message."""
    print(f"{Style.ERROR}[ERRORE] {message}{Colors.RESET}")


def print_info(message: str):
    """Print an info message."""
    print(f"{Style.INFO}[i] {message}{Colors.RESET}")


def format_metric(name: str, value: float, change: float = None, unit: str = "") -> str:
    """Format a metric with optional change indicator."""
    result = f"{name}: {Style.HIGHLIGHT}{value:.4f}{unit}{Colors.RESET}"

    if change is not None:
        if change > 0:
            result += f" {Style.SUCCESS}[+{change:.4f}]{Colors.RESET}"
        elif change < 0:
            result += f" {Style.ERROR}[{change:.4f}]{Colors.RESET}"
        else:
            result += f" {Style.MUTED}[0.0000]{Colors.RESET}"

    return result


def clear_screen():
    """Clear the terminal screen."""
    print("\033[2J\033[H", end="")


def move_cursor_up(lines: int = 1):
    """Move cursor up N lines."""
    print(f"\033[{lines}A", end="")


def clear_line():
    """Clear the current line."""
    print("\033[2K", end="")

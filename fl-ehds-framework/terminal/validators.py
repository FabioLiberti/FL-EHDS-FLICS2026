"""
Input validation utilities for FL-EHDS terminal interface.
Provides type-safe parameter input with defaults and validation.
"""

from typing import Any, Optional, Callable, List, TypeVar
from terminal.colors import Colors, Style, print_error

T = TypeVar('T')


def get_input(
    prompt: str,
    default: T = None,
    validator: Callable[[str], T] = None,
    choices: List[T] = None,
    required: bool = False,
) -> T:
    """
    Get validated input from user.

    Args:
        prompt: The prompt to display
        default: Default value if user presses Enter
        validator: Function to validate/convert input
        choices: List of valid choices
        required: Whether input is required (no default allowed)

    Returns:
        Validated input value
    """
    default_str = f" [{default}]" if default is not None else ""

    while True:
        try:
            raw = input(f"{prompt}{default_str}: ").strip()

            # Use default if empty
            if not raw:
                if default is not None:
                    return default
                elif required:
                    print_error("Questo campo e' obbligatorio")
                    continue
                else:
                    return None

            # Validate against choices
            if choices is not None:
                # Try to convert to same type as choices
                try:
                    if isinstance(choices[0], int):
                        value = int(raw)
                    elif isinstance(choices[0], float):
                        value = float(raw)
                    else:
                        value = raw
                except ValueError:
                    value = raw

                if value not in choices:
                    print_error(f"Valore non valido. Scegli tra: {choices}")
                    continue
                return value

            # Apply validator
            if validator:
                try:
                    return validator(raw)
                except ValueError as e:
                    print_error(str(e))
                    continue

            return raw

        except (EOFError, KeyboardInterrupt):
            print()
            return default


def get_int(
    prompt: str,
    default: int = None,
    min_val: int = None,
    max_val: int = None,
) -> Optional[int]:
    """Get integer input with optional range validation."""

    def validate(raw: str) -> int:
        try:
            value = int(raw)
        except ValueError:
            raise ValueError("Inserisci un numero intero valido")

        if min_val is not None and value < min_val:
            raise ValueError(f"Il valore deve essere >= {min_val}")
        if max_val is not None and value > max_val:
            raise ValueError(f"Il valore deve essere <= {max_val}")

        return value

    return get_input(prompt, default=default, validator=validate)


def get_float(
    prompt: str,
    default: float = None,
    min_val: float = None,
    max_val: float = None,
) -> Optional[float]:
    """Get float input with optional range validation."""

    def validate(raw: str) -> float:
        try:
            value = float(raw)
        except ValueError:
            raise ValueError("Inserisci un numero decimale valido")

        if min_val is not None and value < min_val:
            raise ValueError(f"Il valore deve essere >= {min_val}")
        if max_val is not None and value > max_val:
            raise ValueError(f"Il valore deve essere <= {max_val}")

        return value

    return get_input(prompt, default=default, validator=validate)


def get_bool(prompt: str, default: bool = False) -> bool:
    """Get boolean input (yes/no)."""
    default_str = "S/n" if default else "s/N"

    while True:
        raw = input(f"{prompt} ({default_str}): ").strip().lower()

        if not raw:
            return default

        if raw in ('s', 'si', 'y', 'yes', '1', 'true'):
            return True
        elif raw in ('n', 'no', '0', 'false'):
            return False
        else:
            print_error("Inserisci 's' per si o 'n' per no")


def get_choice(
    prompt: str,
    choices: List[str],
    default: str = None,
) -> str:
    """Get choice from a list of options."""

    # Display choices
    print(f"\n{prompt}")
    for i, choice in enumerate(choices, 1):
        marker = "*" if choice == default else " "
        print(f"  {marker} {i}. {choice}")

    default_idx = choices.index(default) + 1 if default in choices else None

    while True:
        raw = input(f"\nScelta [1-{len(choices)}]: ").strip()

        if not raw and default_idx:
            return default

        try:
            idx = int(raw)
            if 1 <= idx <= len(choices):
                return choices[idx - 1]
            else:
                print_error(f"Inserisci un numero tra 1 e {len(choices)}")
        except ValueError:
            # Maybe they typed the choice directly
            if raw in choices:
                return raw
            print_error("Inserisci il numero dell'opzione")


def confirm(prompt: str, default: bool = False) -> bool:
    """Ask for confirmation."""
    return get_bool(f"{Style.WARNING}{prompt}{Colors.RESET}", default)


def display_config_summary(config: dict, title: str = "RIEPILOGO CONFIGURAZIONE"):
    """Display a configuration summary."""
    print(f"\n{Style.TITLE}{'-' * 60}")
    print(f"{title.center(60)}")
    print(f"{'-' * 60}{Colors.RESET}")

    max_key_len = max(len(str(k)) for k in config.keys())

    for key, value in config.items():
        key_str = str(key).ljust(max_key_len)

        if isinstance(value, bool):
            value_str = f"{Style.SUCCESS}Abilitato{Colors.RESET}" if value else f"{Style.MUTED}Disabilitato{Colors.RESET}"
        elif isinstance(value, float):
            value_str = f"{Style.HIGHLIGHT}{value:.4f}{Colors.RESET}"
        else:
            value_str = f"{Style.HIGHLIGHT}{value}{Colors.RESET}"

        print(f"  {key_str}: {value_str}")

    print(f"{Style.TITLE}{'-' * 60}{Colors.RESET}")

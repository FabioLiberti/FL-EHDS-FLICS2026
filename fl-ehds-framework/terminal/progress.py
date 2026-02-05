"""
Progress bar utilities for FL-EHDS terminal interface.
Wraps tqdm with custom styling for FL training visualization.
"""

import sys
from typing import Optional, Iterable, Any
from contextlib import contextmanager

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

from terminal.colors import Colors, Style


class FLProgressBar:
    """Custom progress bar for federated learning training."""

    def __init__(
        self,
        total: int,
        desc: str = "Training",
        unit: str = "round",
        color: str = "green",
        leave: bool = True,
        position: int = 0,
    ):
        self.total = total
        self.desc = desc
        self.unit = unit
        self.color = color
        self.leave = leave
        self.position = position
        self.current = 0
        self._pbar = None

    def __enter__(self):
        if HAS_TQDM:
            # Map color names to tqdm color codes
            color_map = {
                "green": "green",
                "yellow": "yellow",
                "red": "red",
                "cyan": "cyan",
                "blue": "blue",
                "white": "white",
            }

            self._pbar = tqdm(
                total=self.total,
                desc=self.desc,
                unit=self.unit,
                colour=color_map.get(self.color, "green"),
                leave=self.leave,
                position=self.position,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
                file=sys.stdout,
            )
        return self

    def __exit__(self, *args):
        if self._pbar:
            self._pbar.close()

    def update(self, n: int = 1, **kwargs):
        """Update progress bar."""
        self.current += n
        if self._pbar:
            self._pbar.update(n)
            if kwargs:
                self._pbar.set_postfix(**kwargs)
        else:
            # Fallback without tqdm
            pct = (self.current / self.total) * 100
            print(f"\r{self.desc}: {self.current}/{self.total} ({pct:.1f}%)", end="", flush=True)
            if self.current >= self.total:
                print()

    def set_description(self, desc: str):
        """Update description."""
        self.desc = desc
        if self._pbar:
            self._pbar.set_description(desc)

    def set_postfix(self, **kwargs):
        """Set postfix text."""
        if self._pbar:
            self._pbar.set_postfix(**kwargs)


class TrainingProgress:
    """Multi-level progress display for FL training."""

    def __init__(
        self,
        num_rounds: int,
        num_clients: int,
        show_client_progress: bool = True,
    ):
        self.num_rounds = num_rounds
        self.num_clients = num_clients
        self.show_client_progress = show_client_progress
        self.current_round = 0
        self.metrics_history = []

    @contextmanager
    def round_progress(self, round_num: int):
        """Context manager for a single training round."""
        self.current_round = round_num

        if HAS_TQDM:
            pbar = tqdm(
                total=self.num_clients,
                desc=f"Round {round_num + 1}/{self.num_rounds}",
                unit="client",
                colour="cyan",
                leave=False,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}",
            )
            try:
                yield pbar
            finally:
                pbar.close()
        else:
            # Simple fallback
            class SimplePbar:
                def __init__(self, total):
                    self.total = total
                    self.n = 0

                def update(self, n=1):
                    self.n += n
                    print(f"\r  Client {self.n}/{self.total}", end="", flush=True)
                    if self.n >= self.total:
                        print()

                def set_postfix(self, **kwargs):
                    pass

            yield SimplePbar(self.num_clients)

    def display_round_summary(
        self,
        round_num: int,
        loss: float,
        accuracy: float,
        f1: float = None,
        auc: float = None,
        epsilon: float = None,
    ):
        """Display summary for completed round."""
        print(f"\n{Style.INFO}Round {round_num + 1} completato:{Colors.RESET}")
        print(f"  Loss: {Style.HIGHLIGHT}{loss:.4f}{Colors.RESET}")
        print(f"  Accuracy: {Style.HIGHLIGHT}{accuracy:.2%}{Colors.RESET}", end="")

        if f1 is not None:
            print(f"  F1: {Style.HIGHLIGHT}{f1:.4f}{Colors.RESET}", end="")
        if auc is not None:
            print(f"  AUC: {Style.HIGHLIGHT}{auc:.4f}{Colors.RESET}", end="")
        if epsilon is not None:
            print(f"  Epsilon: {Style.WARNING}{epsilon:.4f}{Colors.RESET}", end="")

        print()

        # Store in history
        self.metrics_history.append({
            "round": round_num,
            "loss": loss,
            "accuracy": accuracy,
            "f1": f1,
            "auc": auc,
            "epsilon": epsilon,
        })


def progress_bar(
    iterable: Iterable = None,
    total: int = None,
    desc: str = "",
    unit: str = "it",
    color: str = "green",
    leave: bool = True,
) -> Iterable:
    """Create a progress bar for an iterable."""
    if HAS_TQDM:
        return tqdm(
            iterable,
            total=total,
            desc=desc,
            unit=unit,
            colour=color,
            leave=leave,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        )
    else:
        # Return iterable with simple progress
        if iterable is not None:
            return iterable
        return range(total) if total else []


def spinner(desc: str = "Loading"):
    """Create a simple spinner for indeterminate progress."""
    import itertools
    import time
    import threading

    spinner_chars = itertools.cycle(['|', '/', '-', '\\'])
    stop_event = threading.Event()

    def spin():
        while not stop_event.is_set():
            sys.stdout.write(f"\r{desc} {next(spinner_chars)}")
            sys.stdout.flush()
            time.sleep(0.1)
        sys.stdout.write(f"\r{desc} OK\n")

    thread = threading.Thread(target=spin)
    thread.start()

    class SpinnerContext:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            stop_event.set()
            thread.join()

    return SpinnerContext()

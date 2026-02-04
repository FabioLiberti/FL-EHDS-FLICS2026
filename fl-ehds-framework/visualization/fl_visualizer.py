#!/usr/bin/env python3
"""
FL-EHDS Real-time Training Visualization

Provides multiple visualization backends:
1. Rich Terminal - Live updating tables and progress bars
2. TensorBoard - Scalars, histograms, and custom plots
3. Matplotlib Animation - Real-time GUI plots

Author: Fabio Liberti
"""

import json
import time
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np

# Terminal visualization
try:
    from rich.console import Console
    from rich.table import Table
    from rich.live import Live
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
    from rich.layout import Layout
    from rich.text import Text
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    # Stub classes for type hints when Rich is not available
    Console = Table = Live = Panel = Layout = Text = None

# TensorBoard
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

# Matplotlib for GUI
try:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


@dataclass
class TrainingMetrics:
    """Container for FL training metrics."""
    round: int = 0
    global_accuracy: float = 0.0
    global_loss: float = 0.0
    global_f1: float = 0.0
    client_accuracies: Dict[str, float] = field(default_factory=dict)
    client_losses: Dict[str, float] = field(default_factory=dict)
    client_samples: Dict[str, int] = field(default_factory=dict)
    participating_clients: List[str] = field(default_factory=list)
    privacy_budget_spent: float = 0.0
    privacy_budget_remaining: float = 0.0
    communication_bytes: int = 0
    round_time_seconds: float = 0.0
    gradient_norms: Dict[str, float] = field(default_factory=dict)


class RichTerminalVisualizer:
    """Real-time terminal visualization using Rich library."""

    def __init__(self, total_rounds: int, num_clients: int):
        if not RICH_AVAILABLE:
            raise ImportError("Rich library not available. Install with: pip install rich")

        self.console = Console()
        self.total_rounds = total_rounds
        self.num_clients = num_clients
        self.history: List[TrainingMetrics] = []
        self.start_time = time.time()
        self._live = None

    def _create_layout(self, metrics: TrainingMetrics) -> Any:
        """Create the terminal layout."""
        layout = Layout()

        # Header
        header = Panel(
            Text("FL-EHDS Training Monitor", style="bold cyan", justify="center"),
            style="cyan"
        )

        # Progress section
        progress_text = f"Round {metrics.round}/{self.total_rounds}"
        progress_pct = (metrics.round / self.total_rounds) * 100
        progress_bar = f"[{'█' * int(progress_pct/2)}{'░' * (50 - int(progress_pct/2))}] {progress_pct:.1f}%"

        progress_panel = Panel(
            f"{progress_text}\n{progress_bar}\n\nElapsed: {time.time() - self.start_time:.1f}s",
            title="Progress",
            style="green"
        )

        # Global metrics table
        global_table = Table(title="Global Metrics", show_header=True)
        global_table.add_column("Metric", style="cyan")
        global_table.add_column("Value", style="green")
        global_table.add_row("Accuracy", f"{metrics.global_accuracy:.2%}")
        global_table.add_row("Loss", f"{metrics.global_loss:.4f}")
        global_table.add_row("F1 Score", f"{metrics.global_f1:.3f}")
        global_table.add_row("Privacy ε spent", f"{metrics.privacy_budget_spent:.2f}")
        global_table.add_row("Comm. (KB)", f"{metrics.communication_bytes / 1024:.1f}")

        # Client metrics table
        client_table = Table(title="Client Status", show_header=True)
        client_table.add_column("Client", style="cyan")
        client_table.add_column("Status", style="yellow")
        client_table.add_column("Accuracy", style="green")
        client_table.add_column("Samples", style="blue")
        client_table.add_column("Grad Norm", style="magenta")

        for client_id in sorted(metrics.client_accuracies.keys()):
            status = "●" if client_id in metrics.participating_clients else "○"
            status_style = "green" if client_id in metrics.participating_clients else "red"
            acc = metrics.client_accuracies.get(client_id, 0)
            samples = metrics.client_samples.get(client_id, 0)
            grad_norm = metrics.gradient_norms.get(client_id, 0)
            client_table.add_row(
                client_id,
                Text(status, style=status_style),
                f"{acc:.2%}",
                str(samples),
                f"{grad_norm:.4f}"
            )

        # Convergence sparkline (last 20 rounds)
        if len(self.history) > 1:
            recent_acc = [m.global_accuracy for m in self.history[-20:]]
            sparkline = self._create_sparkline(recent_acc)
            convergence_panel = Panel(sparkline, title="Accuracy Trend (last 20 rounds)")
        else:
            convergence_panel = Panel("Collecting data...", title="Accuracy Trend")

        # Compose layout
        layout.split_column(
            Layout(header, size=3),
            Layout(progress_panel, size=5),
            Layout(name="body")
        )
        layout["body"].split_row(
            Layout(Panel(global_table, title="Global")),
            Layout(Panel(client_table, title="Clients"))
        )

        return layout

    def _create_sparkline(self, values: List[float]) -> str:
        """Create ASCII sparkline."""
        if not values:
            return ""

        blocks = " ▁▂▃▄▅▆▇█"
        min_val, max_val = min(values), max(values)

        if max_val == min_val:
            return blocks[4] * len(values)

        sparkline = ""
        for v in values:
            idx = int((v - min_val) / (max_val - min_val) * 8)
            sparkline += blocks[idx]

        return f"{sparkline}\nMin: {min_val:.2%}  Max: {max_val:.2%}"

    def update(self, metrics: TrainingMetrics):
        """Update the display with new metrics."""
        self.history.append(metrics)

        if self._live is None:
            return

        layout = self._create_layout(metrics)
        self._live.update(layout)

    def start(self):
        """Start live display."""
        self._live = Live(
            self._create_layout(TrainingMetrics()),
            console=self.console,
            refresh_per_second=4,
            screen=True
        )
        self._live.start()

    def stop(self):
        """Stop live display."""
        if self._live:
            self._live.stop()
            self._live = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()


class TensorBoardVisualizer:
    """TensorBoard logging for FL training."""

    def __init__(self, log_dir: str = "runs/fl_experiment"):
        if not TENSORBOARD_AVAILABLE:
            raise ImportError("TensorBoard not available. Install with: pip install tensorboard")

        self.log_dir = Path(log_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.writer = SummaryWriter(str(self.log_dir / timestamp))

    def update(self, metrics: TrainingMetrics):
        """Log metrics to TensorBoard."""
        step = metrics.round

        # Global metrics
        self.writer.add_scalar("Global/Accuracy", metrics.global_accuracy, step)
        self.writer.add_scalar("Global/Loss", metrics.global_loss, step)
        self.writer.add_scalar("Global/F1", metrics.global_f1, step)

        # Privacy
        self.writer.add_scalar("Privacy/Budget_Spent", metrics.privacy_budget_spent, step)
        self.writer.add_scalar("Privacy/Budget_Remaining", metrics.privacy_budget_remaining, step)

        # Communication
        self.writer.add_scalar("Communication/Bytes_Total", metrics.communication_bytes, step)
        self.writer.add_scalar("Communication/Round_Time", metrics.round_time_seconds, step)

        # Per-client metrics
        client_accs = {f"Client/{k}": v for k, v in metrics.client_accuracies.items()}
        if client_accs:
            self.writer.add_scalars("Accuracy/PerClient", client_accs, step)

        client_norms = {f"Client/{k}": v for k, v in metrics.gradient_norms.items()}
        if client_norms:
            self.writer.add_scalars("GradientNorm/PerClient", client_norms, step)

        # Participation rate
        if metrics.client_accuracies:
            participation_rate = len(metrics.participating_clients) / len(metrics.client_accuracies)
            self.writer.add_scalar("Participation/Rate", participation_rate, step)

        self.writer.flush()

    def close(self):
        """Close the writer."""
        self.writer.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class MatplotlibVisualizer:
    """Real-time matplotlib visualization."""

    def __init__(self, num_clients: int = 5):
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("Matplotlib not available. Install with: pip install matplotlib")

        self.num_clients = num_clients
        self.history: List[TrainingMetrics] = []

        # Setup figure
        plt.ion()  # Interactive mode
        self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 8))
        self.fig.suptitle("FL-EHDS Training Monitor", fontsize=14, fontweight='bold')

        # Initialize plots
        self._setup_plots()

    def _setup_plots(self):
        """Initialize plot elements."""
        # Global accuracy
        self.ax_acc = self.axes[0, 0]
        self.ax_acc.set_title("Global Accuracy")
        self.ax_acc.set_xlabel("Round")
        self.ax_acc.set_ylabel("Accuracy")
        self.ax_acc.set_ylim(0, 1)
        self.line_acc, = self.ax_acc.plot([], [], 'b-', linewidth=2)

        # Client accuracies
        self.ax_clients = self.axes[0, 1]
        self.ax_clients.set_title("Per-Client Accuracy")
        self.ax_clients.set_xlabel("Round")
        self.ax_clients.set_ylabel("Accuracy")
        self.ax_clients.set_ylim(0, 1)
        self.client_lines = {}

        # Privacy budget
        self.ax_privacy = self.axes[1, 0]
        self.ax_privacy.set_title("Privacy Budget")
        self.ax_privacy.set_xlabel("Round")
        self.ax_privacy.set_ylabel("ε spent")
        self.line_privacy, = self.ax_privacy.plot([], [], 'r-', linewidth=2)

        # Gradient norms
        self.ax_grads = self.axes[1, 1]
        self.ax_grads.set_title("Gradient Norms")
        self.ax_grads.set_xlabel("Round")
        self.ax_grads.set_ylabel("L2 Norm")
        self.grad_lines = {}

        plt.tight_layout()

    def update(self, metrics: TrainingMetrics):
        """Update plots with new metrics."""
        self.history.append(metrics)
        rounds = [m.round for m in self.history]

        # Update global accuracy
        accs = [m.global_accuracy for m in self.history]
        self.line_acc.set_data(rounds, accs)
        self.ax_acc.set_xlim(0, max(rounds) + 1)

        # Update client accuracies
        colors = plt.cm.tab10(np.linspace(0, 1, self.num_clients))
        for i, client_id in enumerate(sorted(metrics.client_accuracies.keys())):
            client_accs = [m.client_accuracies.get(client_id, 0) for m in self.history]
            if client_id not in self.client_lines:
                self.client_lines[client_id], = self.ax_clients.plot(
                    [], [], '-', color=colors[i], label=client_id, linewidth=1.5
                )
            self.client_lines[client_id].set_data(rounds, client_accs)
        self.ax_clients.set_xlim(0, max(rounds) + 1)
        if len(self.client_lines) <= 7:
            self.ax_clients.legend(loc='lower right', fontsize=8)

        # Update privacy budget
        privacy = [m.privacy_budget_spent for m in self.history]
        self.line_privacy.set_data(rounds, privacy)
        self.ax_privacy.set_xlim(0, max(rounds) + 1)
        self.ax_privacy.set_ylim(0, max(privacy) * 1.1 + 0.1)

        # Update gradient norms
        for i, client_id in enumerate(sorted(metrics.gradient_norms.keys())):
            norms = [m.gradient_norms.get(client_id, 0) for m in self.history]
            if client_id not in self.grad_lines:
                self.grad_lines[client_id], = self.ax_grads.plot(
                    [], [], '-', color=colors[i], linewidth=1.5
                )
            self.grad_lines[client_id].set_data(rounds, norms)
        self.ax_grads.set_xlim(0, max(rounds) + 1)
        if self.history:
            max_norm = max(max(m.gradient_norms.values()) if m.gradient_norms else 1 for m in self.history)
            self.ax_grads.set_ylim(0, max_norm * 1.1)

        # Redraw
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.01)

    def show(self):
        """Show the plot (blocking)."""
        plt.ioff()
        plt.show()

    def save(self, filepath: str):
        """Save the current figure."""
        self.fig.savefig(filepath, dpi=150, bbox_inches='tight')

    def close(self):
        """Close the figure."""
        plt.close(self.fig)


class CompositeVisualizer:
    """Combines multiple visualization backends."""

    def __init__(
        self,
        total_rounds: int,
        num_clients: int,
        use_rich: bool = True,
        use_tensorboard: bool = True,
        use_matplotlib: bool = False,
        tensorboard_dir: str = "runs/fl_experiment"
    ):
        self.visualizers = []

        if use_rich and RICH_AVAILABLE:
            self.rich_viz = RichTerminalVisualizer(total_rounds, num_clients)
            self.visualizers.append(self.rich_viz)
        else:
            self.rich_viz = None

        if use_tensorboard and TENSORBOARD_AVAILABLE:
            self.tb_viz = TensorBoardVisualizer(tensorboard_dir)
            self.visualizers.append(self.tb_viz)
        else:
            self.tb_viz = None

        if use_matplotlib and MATPLOTLIB_AVAILABLE:
            self.mpl_viz = MatplotlibVisualizer(num_clients)
            self.visualizers.append(self.mpl_viz)
        else:
            self.mpl_viz = None

    def update(self, metrics: TrainingMetrics):
        """Update all visualizers."""
        for viz in self.visualizers:
            viz.update(metrics)

    def start(self):
        """Start visualizers that need it."""
        if self.rich_viz:
            self.rich_viz.start()

    def stop(self):
        """Stop all visualizers."""
        if self.rich_viz:
            self.rich_viz.stop()
        if self.tb_viz:
            self.tb_viz.close()
        if self.mpl_viz:
            self.mpl_viz.close()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()


# Convenience function
def create_visualizer(
    mode: str = "terminal",
    total_rounds: int = 50,
    num_clients: int = 5,
    tensorboard_dir: str = "runs/fl_experiment"
):
    """
    Create a visualizer based on mode.

    Args:
        mode: One of "terminal", "tensorboard", "gui", "all"
        total_rounds: Total number of FL rounds
        num_clients: Number of clients
        tensorboard_dir: TensorBoard log directory

    Returns:
        Visualizer instance
    """
    if mode == "terminal":
        return RichTerminalVisualizer(total_rounds, num_clients)
    elif mode == "tensorboard":
        return TensorBoardVisualizer(tensorboard_dir)
    elif mode == "gui":
        return MatplotlibVisualizer(num_clients)
    elif mode == "all":
        return CompositeVisualizer(
            total_rounds, num_clients,
            use_rich=True, use_tensorboard=True, use_matplotlib=True,
            tensorboard_dir=tensorboard_dir
        )
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'terminal', 'tensorboard', 'gui', or 'all'")


if __name__ == "__main__":
    # Demo
    print("FL-EHDS Visualizer Demo")
    print("=" * 50)
    print(f"Rich available: {RICH_AVAILABLE}")
    print(f"TensorBoard available: {TENSORBOARD_AVAILABLE}")
    print(f"Matplotlib available: {MATPLOTLIB_AVAILABLE}")

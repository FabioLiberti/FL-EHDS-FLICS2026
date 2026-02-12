"""
Real-Time Training Monitor for FL-EHDS Dashboard.

Provides streaming metric visualization during federated learning training,
with governance integration for permit validation and privacy budget tracking.

Uses Streamlit's st.empty() containers and st.line_chart().add_rows()
for incremental updates without full page reruns.

Author: Fabio Liberti
"""

import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
import streamlit as st


class TrainingMonitor:
    """Real-time training monitor with streaming metrics and governance events.

    Creates Streamlit containers for:
    - Progress bar and status text
    - 2x3 metric grid with line charts (accuracy, loss, F1, precision, recall, AUC)
    - Per-client metrics table
    - Privacy budget gauge
    - Governance event log

    Usage::

        monitor = TrainingMonitor()
        monitor.setup()
        callback = monitor.create_progress_callback()
        results = trainer.train(num_rounds, progress_callback=callback)
        monitor.show_final_summary(results)
    """

    def __init__(self):
        self._progress_bar = None
        self._status_text = None
        self._metric_charts: Dict[str, Any] = {}
        self._metric_values: Dict[str, Any] = {}
        self._client_table = None
        self._budget_container = None
        self._event_log_container = None
        self._history: List[Dict] = []
        self._governance_events: List[Dict] = []
        self._start_time: Optional[float] = None

    def setup(self, show_governance: bool = False):
        """Create all UI containers for streaming updates.

        Args:
            show_governance: If True, show privacy budget gauge and event log.
        """
        # Progress section
        self._status_text = st.empty()
        self._progress_bar = st.progress(0)

        # Metric charts - 2 rows x 3 columns
        st.markdown("##### Metriche in Tempo Reale")
        row1_cols = st.columns(3)
        row2_cols = st.columns(3)

        metric_names = ["accuracy", "loss", "f1", "precision", "recall", "auc"]
        metric_labels = ["Accuracy", "Loss", "F1 Score", "Precision", "Recall", "AUC"]
        all_cols = list(row1_cols) + list(row2_cols)

        for col, name, label in zip(all_cols, metric_names, metric_labels):
            with col:
                st.markdown(f"**{label}**")
                self._metric_charts[name] = st.empty()
                self._metric_values[name] = st.empty()

        # Per-client metrics table
        st.markdown("##### Metriche per Client")
        self._client_table = st.empty()

        # Governance section
        if show_governance:
            st.markdown("##### Governance & Privacy Budget")
            gov_cols = st.columns([2, 3])
            with gov_cols[0]:
                self._budget_container = st.empty()
            with gov_cols[1]:
                self._event_log_container = st.empty()

        self._start_time = time.time()

    def update_round(self, round_num: int, total_rounds: int, metrics: Dict):
        """Update all containers with new round data.

        Args:
            round_num: Current round number (1-based).
            total_rounds: Total number of rounds.
            metrics: Dict with accuracy, loss, f1, precision, recall, auc,
                     and optionally node_metrics.
        """
        self._history.append(metrics)

        # Progress bar
        progress_pct = round_num / total_rounds
        if self._progress_bar is not None:
            self._progress_bar.progress(progress_pct)

        # Status text
        elapsed = time.time() - self._start_time if self._start_time else 0
        eta = (elapsed / round_num) * (total_rounds - round_num) if round_num > 0 else 0
        if self._status_text is not None:
            self._status_text.markdown(
                f"**Round {round_num}/{total_rounds}** | "
                f"Acc: {metrics.get('accuracy', 0):.3f} | "
                f"Loss: {metrics.get('loss', 0):.4f} | "
                f"F1: {metrics.get('f1', 0):.3f} | "
                f"AUC: {metrics.get('auc', 0):.3f} | "
                f"Tempo: {elapsed:.0f}s (ETA: {eta:.0f}s)"
            )

        # Update metric charts (every round for first 10, then every 3rd)
        if round_num <= 10 or round_num % 3 == 0 or round_num == total_rounds:
            self._update_metric_charts()

        # Update client table (every 5 rounds or last round)
        if round_num % 5 == 0 or round_num == total_rounds:
            node_metrics = metrics.get("node_metrics", {})
            if node_metrics:
                self._update_client_table(node_metrics, round_num)

    def update_governance_event(self, event: Dict):
        """Add a governance event to the log.

        Args:
            event: Dict with type, message, timestamp, and optional details.
        """
        event.setdefault("timestamp", datetime.now().strftime("%H:%M:%S"))
        self._governance_events.append(event)

        if self._event_log_container is not None:
            df = pd.DataFrame(self._governance_events[-10:])
            self._event_log_container.dataframe(df, use_container_width=True)

    def update_budget(self, budget_status: Dict):
        """Update the privacy budget gauge.

        Args:
            budget_status: Dict with total, used, remaining, utilization_pct.
        """
        if self._budget_container is None:
            return

        total = budget_status.get("total", 1.0)
        used = budget_status.get("used", 0.0)
        remaining = budget_status.get("remaining", total)
        pct = budget_status.get("utilization_pct", 0.0)

        # Inverted progress: show remaining budget
        remaining_pct = max(0.0, min(1.0, remaining / total)) if total > 0 else 0.0

        self._budget_container.markdown(
            f"**Privacy Budget (epsilon)**\n\n"
            f"- Totale: {total:.2f}\n"
            f"- Utilizzato: {used:.4f} ({pct:.1f}%)\n"
            f"- Rimanente: {remaining:.4f}"
        )

    def _update_metric_charts(self):
        """Redraw all metric charts with current history."""
        if not self._history:
            return

        metric_names = ["accuracy", "loss", "f1", "precision", "recall", "auc"]
        for name in metric_names:
            values = [h.get(name, 0) for h in self._history]
            rounds = list(range(1, len(values) + 1))
            df = pd.DataFrame({"Round": rounds, name: values}).set_index("Round")

            if name in self._metric_charts and self._metric_charts[name] is not None:
                self._metric_charts[name].line_chart(df, height=150)

            if name in self._metric_values and self._metric_values[name] is not None:
                current = values[-1]
                if len(values) > 1:
                    delta = current - values[-2]
                    arrow = "+" if delta >= 0 else ""
                    self._metric_values[name].markdown(
                        f"`{current:.4f}` ({arrow}{delta:.4f})"
                    )
                else:
                    self._metric_values[name].markdown(f"`{current:.4f}`")

    def _update_client_table(self, node_metrics: Dict, round_num: int):
        """Update per-client metrics table."""
        if self._client_table is None:
            return

        rows = []
        for client_id, metrics in sorted(node_metrics.items()):
            rows.append({
                "Client": str(client_id),
                "Accuracy": f"{metrics.get('accuracy', 0):.3f}",
                "Loss": f"{metrics.get('loss', 0):.4f}",
                "Samples": metrics.get("samples", "N/A"),
            })

        if rows:
            df = pd.DataFrame(rows)
            self._client_table.dataframe(df, use_container_width=True)

    def show_final_summary(self, results: Dict):
        """Show final training summary after training completes.

        Args:
            results: Final results dict from trainer.
        """
        if self._progress_bar is not None:
            self._progress_bar.progress(1.0)

        elapsed = time.time() - self._start_time if self._start_time else 0

        if self._status_text is not None:
            self._status_text.success(
                f"Training completato in {elapsed:.1f}s | "
                f"Acc: {results.get('final_accuracy', 0):.3f} | "
                f"F1: {results.get('final_f1', 0):.3f} | "
                f"AUC: {results.get('final_auc', 0):.3f}"
            )

    def create_progress_callback(self) -> Callable:
        """Create a callback function compatible with RealFLTrainer.train().

        Returns:
            Callback function with signature (round_num, total_rounds, metrics).
        """
        def callback(round_num: int, total_rounds: int, metrics: Dict):
            self.update_round(round_num, total_rounds, metrics)

        return callback

    def create_governance_callback(
        self,
        governance_bridge,
    ) -> Callable:
        """Create a callback for governance validation per round.

        The returned callback validates the round with the governance bridge,
        updates the budget gauge, and logs governance events.

        Args:
            governance_bridge: GovernanceLifecycleBridge instance.

        Returns:
            Callback with signature (round_num, epsilon_cost) -> (ok, reason).
        """
        def callback(round_num: int, epsilon_cost: float = 0.0):
            ok, reason = governance_bridge.validate_round(round_num, epsilon_cost)

            event = {
                "type": "permit_validation",
                "round": round_num,
                "result": "OK" if ok else "BLOCKED",
                "message": reason,
            }
            self.update_governance_event(event)

            budget = governance_bridge.get_budget_status()
            self.update_budget(budget)

            return ok, reason

        return callback

    def get_history(self) -> List[Dict]:
        """Return the full training history."""
        return self._history

    def get_governance_events(self) -> List[Dict]:
        """Return all governance events."""
        return self._governance_events


def run_monitored_training(
    config: Dict,
    monitor: TrainingMonitor,
    num_rounds: int = 30,
    governance_bridge=None,
) -> Optional[Dict]:
    """High-level function to run training with real-time monitoring.

    Wires together RealFLTrainer, TrainingMonitor, and optional governance
    bridge for a complete monitored training session.

    Args:
        config: Streamlit sidebar config dict.
        monitor: TrainingMonitor instance (already setup()).
        num_rounds: Number of FL rounds.
        governance_bridge: Optional GovernanceLifecycleBridge for governance.

    Returns:
        Training results dict, or None on error.
    """
    try:
        from dashboard.real_trainer_bridge import RealFLTrainer
    except ImportError:
        from real_trainer_bridge import RealFLTrainer

    progress_callback = monitor.create_progress_callback()
    gov_callback = None
    if governance_bridge is not None:
        gov_callback = monitor.create_governance_callback(governance_bridge)

    try:
        # Build bridge config
        bridge_config = {
            "num_clients": config.get("num_nodes", 5),
            "samples_per_client": config.get("samples_per_client", 200),
            "algorithm": config.get("algorithm", "FedAvg"),
            "local_epochs": config.get("local_epochs", 3),
            "batch_size": config.get("batch_size", 32),
            "learning_rate": config.get("learning_rate", 0.01),
            "is_iid": config.get("is_iid", False),
            "alpha": config.get("label_skew_alpha", 0.5),
            "use_dp": config.get("use_dp", False),
            "epsilon": config.get("epsilon", 10.0),
            "clip_norm": config.get("clip_norm", 1.0),
            "seed": config.get("random_seed", 42),
            "fedprox_mu": config.get("fedprox_mu", 0.1),
            "tabular_dataset": config.get("tabular_dataset"),
            "server_lr": config.get("server_lr", 0.1),
            "beta1": config.get("beta1", 0.9),
            "beta2": config.get("beta2", 0.99),
            "tau": config.get("tau", 1e-3),
        }

        trainer = RealFLTrainer(bridge_config)

        # Custom training loop with governance
        history = []
        epsilon_per_round = (
            config.get("epsilon", 10.0) / num_rounds
            if config.get("use_dp", False)
            else 0.0
        )

        for r in range(num_rounds):
            # Governance pre-round check
            if gov_callback is not None:
                ok, reason = gov_callback(r, epsilon_per_round)
                if not ok:
                    monitor.update_governance_event({
                        "type": "training_stopped",
                        "round": r,
                        "message": f"Training interrotto: {reason}",
                    })
                    st.warning(f"Training interrotto al round {r}: {reason}")
                    break

            # Train one round
            result = trainer.trainer.train_round(r)
            round_data = {
                "round": r + 1,
                "accuracy": result.global_acc,
                "loss": result.global_loss,
                "f1": result.global_f1,
                "precision": result.global_precision,
                "recall": result.global_recall,
                "auc": result.global_auc,
                "time": result.time_seconds,
                "node_metrics": {
                    cr.client_id: {
                        "accuracy": cr.train_acc,
                        "loss": cr.train_loss,
                        "samples": cr.num_samples,
                    }
                    for cr in result.client_results
                },
            }
            history.append(round_data)

            # Progress callback for UI update
            progress_callback(r + 1, num_rounds, round_data)

            # Governance post-round logging
            if governance_bridge is not None:
                governance_bridge.log_round_completion(r, result, epsilon_per_round)

        # End governance session
        if governance_bridge is not None:
            final_metrics = {
                "accuracy": history[-1]["accuracy"] if history else 0,
                "loss": history[-1]["loss"] if history else 0,
                "f1": history[-1]["f1"] if history else 0,
                "auc": history[-1]["auc"] if history else 0,
            }
            governance_bridge.end_session(
                total_rounds=len(history),
                final_metrics=final_metrics,
                success=True,
            )

        # Build results
        final = history[-1] if history else {}
        results = {
            "history": history,
            "final_accuracy": final.get("accuracy", 0),
            "final_loss": final.get("loss", 0),
            "final_f1": final.get("f1", 0),
            "final_auc": final.get("auc", 0),
            "algorithm": config.get("algorithm", "FedAvg"),
            "num_clients": config.get("num_nodes", 5),
            "total_rounds": len(history),
            "training_mode": "real_pytorch_monitored",
        }

        monitor.show_final_summary(results)
        return results

    except Exception as e:
        st.error(f"Errore durante il training: {e}")
        return None

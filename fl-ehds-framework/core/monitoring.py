"""
FL-EHDS Monitoring Infrastructure
=================================
Production-grade monitoring using Prometheus and Grafana patterns.
Provides comprehensive observability for FL training pipelines.

Features:
- Prometheus metrics exposition
- Custom FL-specific metrics
- Distributed tracing
- Alert management
- Grafana dashboard generation
- Real-time metric streaming
- EHDS compliance monitoring
- Cross-border audit trails

References:
- Prometheus: https://prometheus.io/
- Grafana: https://grafana.com/
- OpenTelemetry for tracing
- EHDS Art. 50 monitoring requirements
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Dict, List, Optional, Any, Callable, Union,
    Tuple, Set, TypeVar, Generic
)
import asyncio
import json
import logging
import threading
import time
import uuid
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timedelta
from functools import wraps

import numpy as np

logger = logging.getLogger(__name__)

# =============================================================================
# Enums and Constants
# =============================================================================

class MetricType(Enum):
    """Prometheus metric types."""
    COUNTER = auto()
    GAUGE = auto()
    HISTOGRAM = auto()
    SUMMARY = auto()


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertState(Enum):
    """Alert states."""
    INACTIVE = auto()
    PENDING = auto()
    FIRING = auto()
    RESOLVED = auto()


class TraceStatus(Enum):
    """Distributed trace status."""
    OK = auto()
    ERROR = auto()
    TIMEOUT = auto()


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class MonitoringConfig:
    """Configuration for monitoring infrastructure."""
    # Prometheus settings
    prometheus_port: int = 8080
    prometheus_path: str = "/metrics"
    push_gateway_url: Optional[str] = None

    # Metric settings
    default_buckets: List[float] = field(
        default_factory=lambda: [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
    )
    quantiles: List[float] = field(
        default_factory=lambda: [0.5, 0.9, 0.95, 0.99]
    )

    # Alerting
    enable_alerting: bool = True
    alert_evaluation_interval: int = 15  # seconds
    alert_retention_hours: int = 24

    # Tracing
    enable_tracing: bool = True
    trace_sample_rate: float = 1.0
    trace_export_endpoint: Optional[str] = None

    # EHDS compliance
    enable_compliance_metrics: bool = True
    audit_all_operations: bool = True

    # Export
    export_interval: int = 15  # seconds
    batch_size: int = 1000


# =============================================================================
# Metrics
# =============================================================================

@dataclass
class MetricValue:
    """Container for metric values with labels."""
    value: float
    labels: Dict[str, str]
    timestamp: datetime = field(default_factory=datetime.now)


class Metric(ABC):
    """Base class for Prometheus-style metrics."""

    def __init__(
        self,
        name: str,
        description: str,
        label_names: Optional[List[str]] = None,
    ):
        self.name = name
        self.description = description
        self.label_names = label_names or []
        self._values: Dict[Tuple, MetricValue] = {}
        self._lock = threading.Lock()

    @abstractmethod
    def _format_value(self, labels: Dict[str, str]) -> str:
        """Format value for Prometheus exposition."""
        pass

    def _labels_key(self, labels: Dict[str, str]) -> Tuple:
        """Create hashable key from labels."""
        return tuple(sorted(labels.items()))

    def labels(self, **kwargs) -> "Metric":
        """Return metric with specific label values."""
        # Create a labeled metric instance
        return LabeledMetric(self, kwargs)

    def exposition(self) -> str:
        """Generate Prometheus exposition format."""
        lines = [
            f"# HELP {self.name} {self.description}",
            f"# TYPE {self.name} {self._type_name()}",
        ]

        with self._lock:
            for labels_tuple, metric_value in self._values.items():
                labels_dict = dict(labels_tuple)
                lines.append(self._format_value(labels_dict))

        return "\n".join(lines)

    @abstractmethod
    def _type_name(self) -> str:
        """Return Prometheus type name."""
        pass


class LabeledMetric:
    """Metric with specific label values."""

    def __init__(self, metric: Metric, labels: Dict[str, str]):
        self._metric = metric
        self._labels = labels
        self._key = metric._labels_key(labels)

    def inc(self, value: float = 1.0) -> None:
        """Increment (for Counter/Gauge)."""
        if hasattr(self._metric, 'inc'):
            self._metric._inc(self._labels, value)

    def dec(self, value: float = 1.0) -> None:
        """Decrement (for Gauge)."""
        if hasattr(self._metric, 'dec'):
            self._metric._dec(self._labels, value)

    def set(self, value: float) -> None:
        """Set value (for Gauge)."""
        if hasattr(self._metric, 'set'):
            self._metric._set(self._labels, value)

    def observe(self, value: float) -> None:
        """Observe value (for Histogram/Summary)."""
        if hasattr(self._metric, 'observe'):
            self._metric._observe(self._labels, value)


class Counter(Metric):
    """Prometheus Counter metric."""

    def _type_name(self) -> str:
        return "counter"

    def _inc(self, labels: Dict[str, str], value: float = 1.0) -> None:
        """Increment counter."""
        if value < 0:
            raise ValueError("Counter can only be incremented")

        key = self._labels_key(labels)
        with self._lock:
            if key not in self._values:
                self._values[key] = MetricValue(0.0, labels)
            self._values[key].value += value
            self._values[key].timestamp = datetime.now()

    def inc(self, value: float = 1.0) -> None:
        """Increment without labels."""
        self._inc({}, value)

    def _format_value(self, labels: Dict[str, str]) -> str:
        key = self._labels_key(labels)
        value = self._values.get(key, MetricValue(0.0, {})).value
        labels_str = self._format_labels(labels)
        return f"{self.name}{labels_str} {value}"

    def _format_labels(self, labels: Dict[str, str]) -> str:
        if not labels:
            return ""
        pairs = [f'{k}="{v}"' for k, v in sorted(labels.items())]
        return "{" + ",".join(pairs) + "}"


class Gauge(Metric):
    """Prometheus Gauge metric."""

    def _type_name(self) -> str:
        return "gauge"

    def _set(self, labels: Dict[str, str], value: float) -> None:
        """Set gauge value."""
        key = self._labels_key(labels)
        with self._lock:
            self._values[key] = MetricValue(value, labels)

    def _inc(self, labels: Dict[str, str], value: float = 1.0) -> None:
        """Increment gauge."""
        key = self._labels_key(labels)
        with self._lock:
            if key not in self._values:
                self._values[key] = MetricValue(0.0, labels)
            self._values[key].value += value
            self._values[key].timestamp = datetime.now()

    def _dec(self, labels: Dict[str, str], value: float = 1.0) -> None:
        """Decrement gauge."""
        self._inc(labels, -value)

    def set(self, value: float) -> None:
        """Set without labels."""
        self._set({}, value)

    def inc(self, value: float = 1.0) -> None:
        """Increment without labels."""
        self._inc({}, value)

    def dec(self, value: float = 1.0) -> None:
        """Decrement without labels."""
        self._dec({}, value)

    def _format_value(self, labels: Dict[str, str]) -> str:
        key = self._labels_key(labels)
        value = self._values.get(key, MetricValue(0.0, {})).value
        labels_str = self._format_labels(labels)
        return f"{self.name}{labels_str} {value}"

    def _format_labels(self, labels: Dict[str, str]) -> str:
        if not labels:
            return ""
        pairs = [f'{k}="{v}"' for k, v in sorted(labels.items())]
        return "{" + ",".join(pairs) + "}"


class Histogram(Metric):
    """Prometheus Histogram metric."""

    def __init__(
        self,
        name: str,
        description: str,
        label_names: Optional[List[str]] = None,
        buckets: Optional[List[float]] = None,
    ):
        super().__init__(name, description, label_names)
        self.buckets = sorted(buckets or [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0])
        self._counts: Dict[Tuple, Dict[str, int]] = {}
        self._sums: Dict[Tuple, float] = {}

    def _type_name(self) -> str:
        return "histogram"

    def _observe(self, labels: Dict[str, str], value: float) -> None:
        """Observe a value."""
        key = self._labels_key(labels)
        with self._lock:
            if key not in self._counts:
                self._counts[key] = {str(b): 0 for b in self.buckets}
                self._counts[key]["+Inf"] = 0
                self._sums[key] = 0.0

            # Update buckets
            for bucket in self.buckets:
                if value <= bucket:
                    self._counts[key][str(bucket)] += 1
            self._counts[key]["+Inf"] += 1
            self._sums[key] += value

    def observe(self, value: float) -> None:
        """Observe without labels."""
        self._observe({}, value)

    @contextmanager
    def time(self):
        """Context manager to time a block of code."""
        start = time.time()
        try:
            yield
        finally:
            self.observe(time.time() - start)

    def _format_value(self, labels: Dict[str, str]) -> str:
        key = self._labels_key(labels)
        lines = []
        labels_base = self._format_labels(labels)

        if key in self._counts:
            cumulative = 0
            for bucket in self.buckets:
                cumulative += self._counts[key].get(str(bucket), 0)
                bucket_labels = labels.copy()
                bucket_labels["le"] = str(bucket)
                labels_str = self._format_labels(bucket_labels)
                lines.append(f"{self.name}_bucket{labels_str} {cumulative}")

            # +Inf bucket
            cumulative = self._counts[key].get("+Inf", 0)
            bucket_labels = labels.copy()
            bucket_labels["le"] = "+Inf"
            labels_str = self._format_labels(bucket_labels)
            lines.append(f"{self.name}_bucket{labels_str} {cumulative}")

            # Sum and count
            lines.append(f"{self.name}_sum{labels_base} {self._sums.get(key, 0.0)}")
            lines.append(f"{self.name}_count{labels_base} {cumulative}")

        return "\n".join(lines)

    def _format_labels(self, labels: Dict[str, str]) -> str:
        if not labels:
            return ""
        pairs = [f'{k}="{v}"' for k, v in sorted(labels.items())]
        return "{" + ",".join(pairs) + "}"


class Summary(Metric):
    """Prometheus Summary metric."""

    def __init__(
        self,
        name: str,
        description: str,
        label_names: Optional[List[str]] = None,
        quantiles: Optional[List[float]] = None,
        max_age: int = 600,
    ):
        super().__init__(name, description, label_names)
        self.quantiles = quantiles or [0.5, 0.9, 0.95, 0.99]
        self.max_age = max_age
        self._observations: Dict[Tuple, List[Tuple[float, float]]] = defaultdict(list)

    def _type_name(self) -> str:
        return "summary"

    def _observe(self, labels: Dict[str, str], value: float) -> None:
        """Observe a value."""
        key = self._labels_key(labels)
        now = time.time()

        with self._lock:
            # Add observation
            self._observations[key].append((now, value))

            # Remove old observations
            cutoff = now - self.max_age
            self._observations[key] = [
                (t, v) for t, v in self._observations[key]
                if t > cutoff
            ]

    def observe(self, value: float) -> None:
        """Observe without labels."""
        self._observe({}, value)

    def _format_value(self, labels: Dict[str, str]) -> str:
        key = self._labels_key(labels)
        lines = []
        labels_base = self._format_labels(labels)

        with self._lock:
            observations = self._observations.get(key, [])
            if observations:
                values = sorted([v for _, v in observations])
                count = len(values)
                total = sum(values)

                # Calculate quantiles
                for q in self.quantiles:
                    idx = int(q * count)
                    idx = min(idx, count - 1)
                    q_labels = labels.copy()
                    q_labels["quantile"] = str(q)
                    labels_str = self._format_labels(q_labels)
                    lines.append(f"{self.name}{labels_str} {values[idx]}")

                lines.append(f"{self.name}_sum{labels_base} {total}")
                lines.append(f"{self.name}_count{labels_base} {count}")

        return "\n".join(lines)

    def _format_labels(self, labels: Dict[str, str]) -> str:
        if not labels:
            return ""
        pairs = [f'{k}="{v}"' for k, v in sorted(labels.items())]
        return "{" + ",".join(pairs) + "}"


# =============================================================================
# FL-Specific Metrics
# =============================================================================

class FLMetrics:
    """
    Predefined metrics for Federated Learning monitoring.
    """

    def __init__(self, prefix: str = "fl_ehds"):
        self.prefix = prefix

        # Training metrics
        self.rounds_total = Counter(
            f"{prefix}_rounds_total",
            "Total number of FL rounds completed",
            ["status"],
        )

        self.round_duration = Histogram(
            f"{prefix}_round_duration_seconds",
            "Duration of FL rounds",
            ["round_type"],
            buckets=[1, 5, 10, 30, 60, 120, 300, 600],
        )

        self.clients_per_round = Gauge(
            f"{prefix}_clients_per_round",
            "Number of clients participating in current round",
        )

        self.global_model_loss = Gauge(
            f"{prefix}_global_model_loss",
            "Loss of the global model",
            ["metric_type"],
        )

        self.global_model_accuracy = Gauge(
            f"{prefix}_global_model_accuracy",
            "Accuracy of the global model",
            ["metric_type"],
        )

        # Client metrics
        self.client_training_duration = Histogram(
            f"{prefix}_client_training_seconds",
            "Duration of client local training",
            ["client_id", "region"],
            buckets=[1, 5, 10, 30, 60, 120, 300],
        )

        self.client_samples = Gauge(
            f"{prefix}_client_samples",
            "Number of training samples per client",
            ["client_id"],
        )

        self.client_updates_total = Counter(
            f"{prefix}_client_updates_total",
            "Total model updates from clients",
            ["client_id", "status"],
        )

        # Communication metrics
        self.bytes_transmitted = Counter(
            f"{prefix}_bytes_transmitted_total",
            "Total bytes transmitted",
            ["direction", "type"],
        )

        self.communication_latency = Histogram(
            f"{prefix}_communication_latency_seconds",
            "Communication latency",
            ["operation"],
            buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
        )

        # Aggregation metrics
        self.aggregation_duration = Histogram(
            f"{prefix}_aggregation_duration_seconds",
            "Duration of model aggregation",
            ["algorithm"],
            buckets=[0.1, 0.5, 1, 5, 10, 30, 60],
        )

        self.aggregation_errors = Counter(
            f"{prefix}_aggregation_errors_total",
            "Total aggregation errors",
            ["error_type"],
        )

        # Privacy metrics
        self.privacy_budget_used = Gauge(
            f"{prefix}_privacy_budget_used",
            "Differential privacy budget consumed",
            ["client_id"],
        )

        self.noise_scale = Gauge(
            f"{prefix}_noise_scale",
            "Current DP noise scale",
        )

        # EHDS compliance metrics
        self.permit_validations = Counter(
            f"{prefix}_permit_validations_total",
            "Total EHDS permit validations",
            ["status"],
        )

        self.consent_checks = Counter(
            f"{prefix}_consent_checks_total",
            "Total consent checks performed",
            ["result"],
        )

        self.cross_border_transfers = Counter(
            f"{prefix}_cross_border_transfers_total",
            "Total cross-border data transfers",
            ["source_region", "target_region"],
        )

        self.audit_events = Counter(
            f"{prefix}_audit_events_total",
            "Total audit events recorded",
            ["event_type"],
        )

        # Resource metrics
        self.active_clients = Gauge(
            f"{prefix}_active_clients",
            "Number of active FL clients",
            ["region"],
        )

        self.model_size_bytes = Gauge(
            f"{prefix}_model_size_bytes",
            "Size of the global model in bytes",
        )

        self.checkpoint_count = Gauge(
            f"{prefix}_checkpoint_count",
            "Number of stored checkpoints",
        )

    def get_all_metrics(self) -> List[Metric]:
        """Get all defined metrics."""
        return [
            self.rounds_total,
            self.round_duration,
            self.clients_per_round,
            self.global_model_loss,
            self.global_model_accuracy,
            self.client_training_duration,
            self.client_samples,
            self.client_updates_total,
            self.bytes_transmitted,
            self.communication_latency,
            self.aggregation_duration,
            self.aggregation_errors,
            self.privacy_budget_used,
            self.noise_scale,
            self.permit_validations,
            self.consent_checks,
            self.cross_border_transfers,
            self.audit_events,
            self.active_clients,
            self.model_size_bytes,
            self.checkpoint_count,
        ]


# =============================================================================
# Alerting
# =============================================================================

@dataclass
class AlertRule:
    """Definition of an alert rule."""
    name: str
    description: str
    expression: str  # PromQL-like expression
    severity: AlertSeverity
    duration: int = 0  # seconds before firing
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)


@dataclass
class Alert:
    """An active alert instance."""
    rule: AlertRule
    state: AlertState
    started_at: datetime
    fired_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    value: float = 0.0
    labels: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.rule.name,
            "severity": self.rule.severity.value,
            "state": self.state.name,
            "started_at": self.started_at.isoformat(),
            "fired_at": self.fired_at.isoformat() if self.fired_at else None,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "value": self.value,
            "labels": self.labels,
            "description": self.rule.description,
        }


class AlertManager:
    """
    Manages alert rules and active alerts.
    """

    def __init__(self, config: MonitoringConfig):
        self.config = config
        self._rules: Dict[str, AlertRule] = {}
        self._alerts: Dict[str, Alert] = {}
        self._handlers: List[Callable[[Alert], None]] = []
        self._lock = threading.Lock()

    def add_rule(self, rule: AlertRule) -> None:
        """Add an alert rule."""
        with self._lock:
            self._rules[rule.name] = rule
        logger.info(f"Added alert rule: {rule.name}")

    def remove_rule(self, name: str) -> None:
        """Remove an alert rule."""
        with self._lock:
            self._rules.pop(name, None)
            self._alerts.pop(name, None)

    def add_handler(self, handler: Callable[[Alert], None]) -> None:
        """Add alert handler."""
        self._handlers.append(handler)

    def evaluate(self, metrics: Dict[str, float]) -> List[Alert]:
        """
        Evaluate all rules against current metrics.

        Args:
            metrics: Current metric values

        Returns:
            List of firing alerts
        """
        firing = []

        with self._lock:
            for name, rule in self._rules.items():
                # Simple expression evaluation
                value = self._evaluate_expression(rule.expression, metrics)

                if value is not None and value > 0:
                    # Condition is true
                    if name not in self._alerts:
                        # New alert
                        alert = Alert(
                            rule=rule,
                            state=AlertState.PENDING,
                            started_at=datetime.now(),
                            value=value,
                            labels=rule.labels.copy(),
                        )
                        self._alerts[name] = alert
                    else:
                        alert = self._alerts[name]
                        alert.value = value

                    # Check if should fire
                    if alert.state == AlertState.PENDING:
                        elapsed = (datetime.now() - alert.started_at).total_seconds()
                        if elapsed >= rule.duration:
                            alert.state = AlertState.FIRING
                            alert.fired_at = datetime.now()
                            firing.append(alert)
                            self._notify_handlers(alert)

                    elif alert.state == AlertState.FIRING:
                        firing.append(alert)

                else:
                    # Condition is false
                    if name in self._alerts:
                        alert = self._alerts[name]
                        if alert.state == AlertState.FIRING:
                            alert.state = AlertState.RESOLVED
                            alert.resolved_at = datetime.now()
                            self._notify_handlers(alert)
                        del self._alerts[name]

        return firing

    def _evaluate_expression(
        self,
        expression: str,
        metrics: Dict[str, float],
    ) -> Optional[float]:
        """
        Evaluate a simple expression against metrics.

        Supports: metric > value, metric < value, metric == value
        """
        for op in [">", "<", ">=", "<=", "=="]:
            if op in expression:
                parts = expression.split(op)
                if len(parts) == 2:
                    metric_name = parts[0].strip()
                    threshold = float(parts[1].strip())

                    if metric_name in metrics:
                        value = metrics[metric_name]
                        if op == ">" and value > threshold:
                            return value
                        elif op == "<" and value < threshold:
                            return value
                        elif op == ">=" and value >= threshold:
                            return value
                        elif op == "<=" and value <= threshold:
                            return value
                        elif op == "==" and value == threshold:
                            return value

        return None

    def _notify_handlers(self, alert: Alert) -> None:
        """Notify all handlers of alert."""
        for handler in self._handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Alert handler error: {e}")

    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts."""
        with self._lock:
            return [
                a for a in self._alerts.values()
                if a.state in (AlertState.PENDING, AlertState.FIRING)
            ]

    def get_alert_history(
        self,
        hours: int = 24,
    ) -> List[Dict[str, Any]]:
        """Get alert history."""
        # In production, this would query a database
        with self._lock:
            return [a.to_dict() for a in self._alerts.values()]


# =============================================================================
# Distributed Tracing
# =============================================================================

@dataclass
class Span:
    """A single span in a distributed trace."""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    operation_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    status: TraceStatus = TraceStatus.OK
    tags: Dict[str, str] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def duration_ms(self) -> float:
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds() * 1000
        return 0.0

    def set_tag(self, key: str, value: str) -> None:
        """Set a tag on the span."""
        self.tags[key] = value

    def log(self, message: str, **kwargs) -> None:
        """Add a log entry to the span."""
        self.logs.append({
            "timestamp": datetime.now().isoformat(),
            "message": message,
            **kwargs,
        })

    def finish(self, status: TraceStatus = TraceStatus.OK) -> None:
        """Finish the span."""
        self.end_time = datetime.now()
        self.status = status

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "operation_name": self.operation_name,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "status": self.status.name,
            "tags": self.tags,
            "logs": self.logs,
        }


class Tracer:
    """
    Distributed tracing for FL operations.
    """

    def __init__(self, config: MonitoringConfig):
        self.config = config
        self._traces: Dict[str, List[Span]] = {}
        self._current_span: Dict[int, Span] = {}  # thread_id -> span
        self._lock = threading.Lock()

    def start_span(
        self,
        operation_name: str,
        parent: Optional[Span] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> Span:
        """
        Start a new span.

        Args:
            operation_name: Name of the operation
            parent: Parent span (for child spans)
            tags: Initial tags

        Returns:
            New Span instance
        """
        # Get or create trace ID
        if parent:
            trace_id = parent.trace_id
            parent_span_id = parent.span_id
        else:
            trace_id = str(uuid.uuid4())
            parent_span_id = None

        span = Span(
            trace_id=trace_id,
            span_id=str(uuid.uuid4())[:16],
            parent_span_id=parent_span_id,
            operation_name=operation_name,
            start_time=datetime.now(),
            tags=tags or {},
        )

        with self._lock:
            if trace_id not in self._traces:
                self._traces[trace_id] = []
            self._traces[trace_id].append(span)

            # Set as current span for this thread
            self._current_span[threading.get_ident()] = span

        return span

    def finish_span(
        self,
        span: Span,
        status: TraceStatus = TraceStatus.OK,
    ) -> None:
        """Finish a span."""
        span.finish(status)

        # Export if configured
        if self.config.trace_export_endpoint:
            self._export_span(span)

    def get_current_span(self) -> Optional[Span]:
        """Get the current span for this thread."""
        return self._current_span.get(threading.get_ident())

    @contextmanager
    def trace(
        self,
        operation_name: str,
        tags: Optional[Dict[str, str]] = None,
    ):
        """Context manager for tracing an operation."""
        parent = self.get_current_span()
        span = self.start_span(operation_name, parent, tags)
        try:
            yield span
            self.finish_span(span, TraceStatus.OK)
        except Exception as e:
            span.log("error", error=str(e))
            self.finish_span(span, TraceStatus.ERROR)
            raise

    def get_trace(self, trace_id: str) -> List[Span]:
        """Get all spans for a trace."""
        with self._lock:
            return self._traces.get(trace_id, [])

    def _export_span(self, span: Span) -> None:
        """Export span to collection endpoint."""
        # In production, send to Jaeger/Zipkin
        logger.debug(f"Exported span: {span.operation_name}")


# =============================================================================
# Grafana Dashboard Generation
# =============================================================================

class GrafanaDashboardGenerator:
    """
    Generates Grafana dashboard JSON for FL monitoring.
    """

    def __init__(self, prefix: str = "fl_ehds"):
        self.prefix = prefix

    def generate_fl_dashboard(self) -> Dict[str, Any]:
        """Generate complete FL monitoring dashboard."""
        return {
            "title": "FL-EHDS Monitoring Dashboard",
            "uid": "fl-ehds-main",
            "tags": ["federated-learning", "ehds"],
            "timezone": "browser",
            "refresh": "30s",
            "time": {
                "from": "now-1h",
                "to": "now"
            },
            "panels": self._generate_panels(),
            "templating": self._generate_variables(),
        }

    def _generate_panels(self) -> List[Dict[str, Any]]:
        """Generate dashboard panels."""
        panels = []
        y_pos = 0

        # Row 1: Overview
        panels.append(self._create_row("Overview", y_pos))
        y_pos += 1

        panels.append(self._create_stat_panel(
            "Total Rounds",
            f"sum({self.prefix}_rounds_total)",
            0, y_pos, 6, 4
        ))

        panels.append(self._create_stat_panel(
            "Active Clients",
            f"sum({self.prefix}_active_clients)",
            6, y_pos, 6, 4
        ))

        panels.append(self._create_stat_panel(
            "Global Accuracy",
            f'{self.prefix}_global_model_accuracy{{metric_type="validation"}}',
            12, y_pos, 6, 4,
            unit="percentunit"
        ))

        panels.append(self._create_stat_panel(
            "Global Loss",
            f'{self.prefix}_global_model_loss{{metric_type="validation"}}',
            18, y_pos, 6, 4
        ))

        y_pos += 4

        # Row 2: Training Progress
        panels.append(self._create_row("Training Progress", y_pos))
        y_pos += 1

        panels.append(self._create_graph_panel(
            "Model Loss Over Time",
            [
                (f'{self.prefix}_global_model_loss{{metric_type="training"}}', "Training Loss"),
                (f'{self.prefix}_global_model_loss{{metric_type="validation"}}', "Validation Loss"),
            ],
            0, y_pos, 12, 8
        ))

        panels.append(self._create_graph_panel(
            "Model Accuracy Over Time",
            [
                (f'{self.prefix}_global_model_accuracy{{metric_type="training"}}', "Training Accuracy"),
                (f'{self.prefix}_global_model_accuracy{{metric_type="validation"}}', "Validation Accuracy"),
            ],
            12, y_pos, 12, 8
        ))

        y_pos += 8

        # Row 3: Client Metrics
        panels.append(self._create_row("Client Metrics", y_pos))
        y_pos += 1

        panels.append(self._create_graph_panel(
            "Clients Per Round",
            [(f"{self.prefix}_clients_per_round", "Clients")],
            0, y_pos, 12, 6
        ))

        panels.append(self._create_heatmap_panel(
            "Client Training Duration",
            f"{self.prefix}_client_training_seconds_bucket",
            12, y_pos, 12, 6
        ))

        y_pos += 6

        # Row 4: Communication
        panels.append(self._create_row("Communication", y_pos))
        y_pos += 1

        panels.append(self._create_graph_panel(
            "Bytes Transmitted",
            [
                (f'rate({self.prefix}_bytes_transmitted_total{{direction="upload"}}[5m])', "Upload"),
                (f'rate({self.prefix}_bytes_transmitted_total{{direction="download"}}[5m])', "Download"),
            ],
            0, y_pos, 12, 6
        ))

        panels.append(self._create_graph_panel(
            "Communication Latency",
            [(f"histogram_quantile(0.95, rate({self.prefix}_communication_latency_seconds_bucket[5m]))", "p95 Latency")],
            12, y_pos, 12, 6
        ))

        y_pos += 6

        # Row 5: EHDS Compliance
        panels.append(self._create_row("EHDS Compliance", y_pos))
        y_pos += 1

        panels.append(self._create_stat_panel(
            "Permit Validations",
            f'sum({self.prefix}_permit_validations_total{{status="success"}})',
            0, y_pos, 6, 4
        ))

        panels.append(self._create_stat_panel(
            "Cross-Border Transfers",
            f"sum({self.prefix}_cross_border_transfers_total)",
            6, y_pos, 6, 4
        ))

        panels.append(self._create_graph_panel(
            "Audit Events",
            [(f"rate({self.prefix}_audit_events_total[5m])", "Events/sec")],
            12, y_pos, 12, 4
        ))

        return panels

    def _create_row(
        self,
        title: str,
        y: int,
    ) -> Dict[str, Any]:
        """Create a row panel."""
        return {
            "type": "row",
            "title": title,
            "gridPos": {"x": 0, "y": y, "w": 24, "h": 1},
            "collapsed": False,
        }

    def _create_stat_panel(
        self,
        title: str,
        query: str,
        x: int,
        y: int,
        w: int,
        h: int,
        unit: str = "short",
    ) -> Dict[str, Any]:
        """Create a stat panel."""
        return {
            "type": "stat",
            "title": title,
            "gridPos": {"x": x, "y": y, "w": w, "h": h},
            "targets": [{
                "expr": query,
                "refId": "A",
            }],
            "options": {
                "colorMode": "value",
                "graphMode": "area",
            },
            "fieldConfig": {
                "defaults": {
                    "unit": unit,
                }
            },
        }

    def _create_graph_panel(
        self,
        title: str,
        queries: List[Tuple[str, str]],
        x: int,
        y: int,
        w: int,
        h: int,
    ) -> Dict[str, Any]:
        """Create a time series graph panel."""
        targets = []
        for i, (query, legend) in enumerate(queries):
            targets.append({
                "expr": query,
                "legendFormat": legend,
                "refId": chr(65 + i),  # A, B, C...
            })

        return {
            "type": "timeseries",
            "title": title,
            "gridPos": {"x": x, "y": y, "w": w, "h": h},
            "targets": targets,
            "options": {
                "legend": {"displayMode": "list"},
                "tooltip": {"mode": "multi"},
            },
        }

    def _create_heatmap_panel(
        self,
        title: str,
        query: str,
        x: int,
        y: int,
        w: int,
        h: int,
    ) -> Dict[str, Any]:
        """Create a heatmap panel."""
        return {
            "type": "heatmap",
            "title": title,
            "gridPos": {"x": x, "y": y, "w": w, "h": h},
            "targets": [{
                "expr": query,
                "format": "heatmap",
                "refId": "A",
            }],
        }

    def _generate_variables(self) -> Dict[str, Any]:
        """Generate dashboard variables."""
        return {
            "list": [
                {
                    "name": "region",
                    "type": "query",
                    "query": f'label_values({self.prefix}_active_clients, region)',
                    "multi": True,
                    "includeAll": True,
                },
                {
                    "name": "client",
                    "type": "query",
                    "query": f'label_values({self.prefix}_client_samples, client_id)',
                    "multi": True,
                    "includeAll": True,
                },
            ]
        }

    def export_json(self, path: str) -> None:
        """Export dashboard to JSON file."""
        dashboard = self.generate_fl_dashboard()
        with open(path, 'w') as f:
            json.dump(dashboard, f, indent=2)
        logger.info(f"Exported dashboard to {path}")


# =============================================================================
# Monitoring Manager
# =============================================================================

class MonitoringManager:
    """
    Main monitoring manager for FL-EHDS.
    Coordinates metrics, alerting, and tracing.
    """

    def __init__(self, config: Optional[MonitoringConfig] = None):
        self.config = config or MonitoringConfig()

        # Initialize components
        self.metrics = FLMetrics()
        self._custom_metrics: Dict[str, Metric] = {}
        self.alerts = AlertManager(self.config)
        self.tracer = Tracer(self.config)
        self.dashboard_generator = GrafanaDashboardGenerator()

        # Background tasks
        self._running = False
        self._export_task: Optional[asyncio.Task] = None
        self._alert_task: Optional[asyncio.Task] = None

        # Default alert rules
        self._setup_default_alerts()

    def _setup_default_alerts(self) -> None:
        """Setup default FL alert rules."""
        self.alerts.add_rule(AlertRule(
            name="HighClientDropout",
            description="High client dropout rate detected",
            expression="fl_ehds_active_clients < 5",
            severity=AlertSeverity.WARNING,
            duration=60,
        ))

        self.alerts.add_rule(AlertRule(
            name="AggregationErrors",
            description="Aggregation errors occurring",
            expression="fl_ehds_aggregation_errors_total > 0",
            severity=AlertSeverity.CRITICAL,
            duration=0,
        ))

        self.alerts.add_rule(AlertRule(
            name="HighLatency",
            description="Communication latency above threshold",
            expression="fl_ehds_communication_latency > 5",
            severity=AlertSeverity.WARNING,
            duration=120,
        ))

        self.alerts.add_rule(AlertRule(
            name="PrivacyBudgetLow",
            description="Privacy budget nearly exhausted",
            expression="fl_ehds_privacy_budget_used > 0.9",
            severity=AlertSeverity.WARNING,
            duration=0,
        ))

    async def start(self) -> None:
        """Start monitoring services."""
        self._running = True

        # Start background tasks
        if self.config.enable_alerting:
            self._alert_task = asyncio.create_task(self._alert_loop())

        self._export_task = asyncio.create_task(self._export_loop())

        logger.info("Monitoring manager started")

    async def stop(self) -> None:
        """Stop monitoring services."""
        self._running = False

        if self._alert_task:
            self._alert_task.cancel()

        if self._export_task:
            self._export_task.cancel()

        logger.info("Monitoring manager stopped")

    def register_metric(self, metric: Metric) -> None:
        """Register a custom metric."""
        self._custom_metrics[metric.name] = metric

    def get_prometheus_metrics(self) -> str:
        """Get all metrics in Prometheus exposition format."""
        lines = []

        # FL metrics
        for metric in self.metrics.get_all_metrics():
            lines.append(metric.exposition())

        # Custom metrics
        for metric in self._custom_metrics.values():
            lines.append(metric.exposition())

        return "\n\n".join(lines)

    def record_round_metrics(
        self,
        round_number: int,
        duration: float,
        num_clients: int,
        loss: float,
        accuracy: float,
    ) -> None:
        """Record metrics for an FL round."""
        self.metrics.rounds_total.labels(status="completed").inc()
        self.metrics.round_duration.labels(round_type="standard").observe(duration)
        self.metrics.clients_per_round.set(num_clients)
        self.metrics.global_model_loss.labels(metric_type="training").set(loss)
        self.metrics.global_model_accuracy.labels(metric_type="training").set(accuracy)

    def record_client_metrics(
        self,
        client_id: str,
        region: str,
        training_duration: float,
        num_samples: int,
    ) -> None:
        """Record client-specific metrics."""
        self.metrics.client_training_duration.labels(
            client_id=client_id, region=region
        ).observe(training_duration)
        self.metrics.client_samples.labels(client_id=client_id).set(num_samples)
        self.metrics.client_updates_total.labels(
            client_id=client_id, status="success"
        ).inc()

    def record_communication_metrics(
        self,
        direction: str,
        data_type: str,
        bytes_count: int,
        latency: float,
    ) -> None:
        """Record communication metrics."""
        self.metrics.bytes_transmitted.labels(
            direction=direction, type=data_type
        ).inc(bytes_count)
        self.metrics.communication_latency.labels(operation=data_type).observe(latency)

    def record_ehds_compliance(
        self,
        event_type: str,
        success: bool,
        details: Optional[Dict[str, str]] = None,
    ) -> None:
        """Record EHDS compliance metrics."""
        if event_type == "permit_validation":
            self.metrics.permit_validations.labels(
                status="success" if success else "failed"
            ).inc()
        elif event_type == "consent_check":
            self.metrics.consent_checks.labels(
                result="granted" if success else "denied"
            ).inc()
        elif event_type == "cross_border":
            if details:
                self.metrics.cross_border_transfers.labels(
                    source_region=details.get("source", "unknown"),
                    target_region=details.get("target", "unknown"),
                ).inc()

        self.metrics.audit_events.labels(event_type=event_type).inc()

    async def _alert_loop(self) -> None:
        """Background loop for alert evaluation."""
        while self._running:
            try:
                # Collect current metric values for evaluation
                metrics = self._collect_metric_values()

                # Evaluate alerts
                firing = self.alerts.evaluate(metrics)

                if firing:
                    logger.warning(f"Firing alerts: {[a.rule.name for a in firing]}")

                await asyncio.sleep(self.config.alert_evaluation_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Alert evaluation error: {e}")

    async def _export_loop(self) -> None:
        """Background loop for metric export."""
        while self._running:
            try:
                # Push metrics if gateway configured
                if self.config.push_gateway_url:
                    await self._push_metrics()

                await asyncio.sleep(self.config.export_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Metric export error: {e}")

    def _collect_metric_values(self) -> Dict[str, float]:
        """Collect current metric values for alerting."""
        # This would extract values from metrics for alert evaluation
        return {}

    async def _push_metrics(self) -> None:
        """Push metrics to Prometheus Push Gateway."""
        # In production, use prometheus_client push_to_gateway
        pass

    def export_dashboard(self, path: str) -> None:
        """Export Grafana dashboard to file."""
        self.dashboard_generator.export_json(path)


# =============================================================================
# Decorators
# =============================================================================

def timed(
    metric: Optional[Histogram] = None,
    operation: str = "unknown",
):
    """Decorator to time function execution."""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start = time.time()
            try:
                return await func(*args, **kwargs)
            finally:
                duration = time.time() - start
                if metric:
                    metric.labels(operation=operation).observe(duration)

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start = time.time()
            try:
                return func(*args, **kwargs)
            finally:
                duration = time.time() - start
                if metric:
                    metric.labels(operation=operation).observe(duration)

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


def counted(
    metric: Optional[Counter] = None,
    labels: Optional[Dict[str, str]] = None,
):
    """Decorator to count function calls."""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            if metric:
                if labels:
                    metric.labels(**labels).inc()
                else:
                    metric.inc()
            return await func(*args, **kwargs)

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            if metric:
                if labels:
                    metric.labels(**labels).inc()
                else:
                    metric.inc()
            return func(*args, **kwargs)

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


# =============================================================================
# Factory Functions
# =============================================================================

def create_monitoring_config(**kwargs) -> MonitoringConfig:
    """Create monitoring configuration."""
    return MonitoringConfig(**kwargs)


def create_monitoring_manager(
    config: Optional[MonitoringConfig] = None,
    **kwargs
) -> MonitoringManager:
    """Create monitoring manager."""
    if config is None:
        config = create_monitoring_config(**kwargs)
    return MonitoringManager(config)


# =============================================================================
# Example Usage
# =============================================================================

async def example_usage():
    """Example of monitoring infrastructure usage."""

    # Create monitoring manager
    monitoring = create_monitoring_manager(
        prometheus_port=8080,
        enable_alerting=True,
        enable_tracing=True,
    )

    await monitoring.start()

    # Record FL training metrics
    for round_num in range(5):
        # Simulate round
        with monitoring.tracer.trace(f"fl_round_{round_num}") as span:
            span.set_tag("round_number", str(round_num))

            # Record metrics
            monitoring.record_round_metrics(
                round_number=round_num,
                duration=np.random.uniform(10, 60),
                num_clients=np.random.randint(5, 20),
                loss=1.0 / (round_num + 1),
                accuracy=0.5 + 0.1 * round_num,
            )

            # Simulate client training
            for client_id in range(3):
                monitoring.record_client_metrics(
                    client_id=f"client_{client_id}",
                    region="EU",
                    training_duration=np.random.uniform(5, 30),
                    num_samples=np.random.randint(100, 1000),
                )

            # Record EHDS compliance
            monitoring.record_ehds_compliance(
                event_type="permit_validation",
                success=True,
            )

    # Get Prometheus metrics
    metrics_output = monitoring.get_prometheus_metrics()
    print("Prometheus Metrics:")
    print(metrics_output[:500] + "...")

    # Check alerts
    active_alerts = monitoring.alerts.get_active_alerts()
    print(f"\nActive alerts: {len(active_alerts)}")

    # Export dashboard
    dashboard = monitoring.dashboard_generator.generate_fl_dashboard()
    print(f"\nGenerated dashboard with {len(dashboard['panels'])} panels")

    await monitoring.stop()
    print("\nMonitoring demo complete")


if __name__ == "__main__":
    asyncio.run(example_usage())

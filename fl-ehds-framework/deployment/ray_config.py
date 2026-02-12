"""
Ray Configuration Classes for FL-EHDS
======================================
Pure configuration dataclasses for Ray cluster deployment of FL workloads.
No ray client dependency - only configuration and YAML generation.

Configures:
- Head node (GCS, dashboard, object store)
- Worker groups (FL clients, autoscaling)
- Actor specifications (aggregator, client resources)
- Runtime environment (pip, conda, env vars)

Author: Fabio Liberti
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


# =============================================================================
# Node Configurations
# =============================================================================

@dataclass
class RayHeadNodeConfig:
    """Ray head node configuration."""
    cpu: int = 4
    memory_gb: int = 8
    object_store_memory_gb: int = 2
    dashboard_port: int = 8265
    gcs_port: int = 6379
    num_gpus: int = 0
    # Head-specific
    include_dashboard: bool = True
    dashboard_host: str = "0.0.0.0"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "resources": {
                "CPU": self.cpu,
                "memory": self.memory_gb * 1024 * 1024 * 1024,
                "object_store_memory": self.object_store_memory_gb * 1024 * 1024 * 1024,
                "GPU": self.num_gpus,
            },
            "dashboard_port": self.dashboard_port,
            "gcs_server_port": self.gcs_port,
            "include_dashboard": self.include_dashboard,
            "dashboard_host": self.dashboard_host,
        }


@dataclass
class RayWorkerGroupConfig:
    """Ray worker group configuration."""
    group_name: str = "fl-workers"
    num_workers: int = 5
    cpu_per_worker: int = 2
    memory_gb_per_worker: int = 4
    gpu_per_worker: int = 0
    min_workers: int = 1
    max_workers: int = 20

    def to_dict(self) -> Dict[str, Any]:
        return {
            "group_name": self.group_name,
            "num_workers": self.num_workers,
            "min_workers": self.min_workers,
            "max_workers": self.max_workers,
            "resources_per_worker": {
                "CPU": self.cpu_per_worker,
                "memory": self.memory_gb_per_worker * 1024 * 1024 * 1024,
                "GPU": self.gpu_per_worker,
            },
        }


# =============================================================================
# Autoscaler and Actor Specifications
# =============================================================================

@dataclass
class RayAutoscalerConfig:
    """Ray autoscaler configuration."""
    enabled: bool = True
    idle_timeout_s: int = 60
    upscaling_speed: float = 1.0
    target_utilization: float = 0.8

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "idle_timeout_minutes": self.idle_timeout_s / 60,
            "upscaling_speed": self.upscaling_speed,
            "target_utilization_fraction": self.target_utilization,
        }


@dataclass
class RayActorSpec:
    """Ray actor resource specification for FL nodes."""
    name: str = "fl-client"
    num_cpus: float = 1.0
    num_gpus: float = 0.0
    memory_bytes: Optional[int] = None
    max_restarts: int = 3
    max_task_retries: int = 1
    # FL specific
    role: str = "client"

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "name": self.name,
            "num_cpus": self.num_cpus,
            "num_gpus": self.num_gpus,
            "max_restarts": self.max_restarts,
            "max_task_retries": self.max_task_retries,
            "role": self.role,
        }
        if self.memory_bytes:
            d["memory"] = self.memory_bytes
        return d


# =============================================================================
# Top-Level FL Ray Cluster Configuration
# =============================================================================

@dataclass
class FLRayClusterConfig:
    """Complete Ray cluster configuration for FL-EHDS deployment."""
    cluster_name: str = "fl-ehds-ray"
    head: RayHeadNodeConfig = field(default_factory=RayHeadNodeConfig)
    worker_groups: List[RayWorkerGroupConfig] = field(default_factory=list)
    autoscaler: RayAutoscalerConfig = field(default_factory=RayAutoscalerConfig)
    runtime_env: Dict[str, Any] = field(default_factory=dict)
    # FL EHDS specific
    aggregator_actor: RayActorSpec = field(default_factory=lambda: RayActorSpec(name="fl-aggregator", role="aggregator", num_cpus=2.0))
    client_actors: List[RayActorSpec] = field(default_factory=list)
    num_fl_rounds: int = 30
    fl_algorithm: str = "FedAvg"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cluster_name": self.cluster_name,
            "head_node": self.head.to_dict(),
            "worker_groups": [wg.to_dict() for wg in self.worker_groups],
            "autoscaler": self.autoscaler.to_dict(),
            "runtime_env": self.runtime_env,
            "fl_config": {
                "algorithm": self.fl_algorithm,
                "num_rounds": self.num_fl_rounds,
                "aggregator": self.aggregator_actor.to_dict(),
                "clients": [c.to_dict() for c in self.client_actors],
            },
        }

    def to_yaml(self) -> str:
        if not HAS_YAML:
            raise ImportError("PyYAML required for YAML output: pip install pyyaml")
        return yaml.dump(self.to_dict(), default_flow_style=False)

    @classmethod
    def create_fl_cluster(
        cls,
        num_clients: int = 5,
        cluster_name: str = "fl-ehds-ray",
        head_cpu: int = 4,
        head_memory_gb: int = 8,
        worker_cpu: int = 2,
        worker_memory_gb: int = 4,
        worker_gpu: int = 0,
        fl_algorithm: str = "FedAvg",
        num_rounds: int = 30,
        enable_autoscaler: bool = True,
        pip_packages: Optional[List[str]] = None,
    ) -> "FLRayClusterConfig":
        """Factory method to create a complete FL-EHDS Ray cluster config."""
        head = RayHeadNodeConfig(
            cpu=head_cpu,
            memory_gb=head_memory_gb,
        )

        worker_groups = [
            RayWorkerGroupConfig(
                group_name="fl-workers",
                num_workers=num_clients,
                cpu_per_worker=worker_cpu,
                memory_gb_per_worker=worker_memory_gb,
                gpu_per_worker=worker_gpu,
                min_workers=1,
                max_workers=num_clients * 2,
            ),
        ]

        autoscaler = RayAutoscalerConfig(enabled=enable_autoscaler)

        runtime_env = {
            "pip": pip_packages or [
                "torch>=2.0.0",
                "torchvision>=0.15.0",
                "numpy>=1.24.0",
                "scikit-learn>=1.3.0",
            ],
            "env_vars": {
                "FL_ALGORITHM": fl_algorithm,
                "FL_NUM_ROUNDS": str(num_rounds),
                "EHDS_AUDIT_LOG": "true",
            },
        }

        aggregator_actor = RayActorSpec(
            name="fl-aggregator",
            num_cpus=2.0,
            role="aggregator",
            max_restarts=5,
        )

        client_actors = [
            RayActorSpec(
                name=f"fl-client-{i}",
                num_cpus=float(worker_cpu),
                num_gpus=float(worker_gpu),
                role="client",
            )
            for i in range(num_clients)
        ]

        return cls(
            cluster_name=cluster_name,
            head=head,
            worker_groups=worker_groups,
            autoscaler=autoscaler,
            runtime_env=runtime_env,
            aggregator_actor=aggregator_actor,
            client_actors=client_actors,
            num_fl_rounds=num_rounds,
            fl_algorithm=fl_algorithm,
        )

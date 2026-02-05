"""
FL-EHDS Orchestration Infrastructure
====================================
Kubernetes and Ray-based orchestration for enterprise-scale FL.
Enables dynamic scaling, fault tolerance, and resource management.

Features:
- Kubernetes deployment for FL nodes
- Ray distributed computing integration
- Auto-scaling based on workload
- Resource quota management
- Multi-region federation support
- Health monitoring and recovery
- EHDS-compliant deployment policies
- Cross-border orchestration (Art. 50)

References:
- Kubernetes: https://kubernetes.io/
- Ray: https://www.ray.io/
- EHDS Regulation EU 2025/327
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Dict, List, Optional, Any, Callable, Union,
    Tuple, Set, TypeVar, AsyncIterator
)
import asyncio
import json
import logging
import os
import threading
import time
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import datetime, timedelta

import numpy as np

logger = logging.getLogger(__name__)

# =============================================================================
# Enums and Constants
# =============================================================================

class OrchestratorType(Enum):
    """Supported orchestration platforms."""
    KUBERNETES = auto()
    RAY = auto()
    HYBRID = auto()  # Kubernetes + Ray
    LOCAL = auto()   # For development


class NodeType(Enum):
    """Types of FL nodes."""
    AGGREGATOR = "aggregator"
    CLIENT = "client"
    COORDINATOR = "coordinator"
    GATEWAY = "gateway"  # Cross-border


class NodeStatus(Enum):
    """Node lifecycle status."""
    PENDING = auto()
    CREATING = auto()
    RUNNING = auto()
    READY = auto()
    TRAINING = auto()
    AGGREGATING = auto()
    DRAINING = auto()
    TERMINATED = auto()
    FAILED = auto()


class ScalingPolicy(Enum):
    """Auto-scaling policies."""
    MANUAL = auto()
    CPU_BASED = auto()
    MEMORY_BASED = auto()
    QUEUE_BASED = auto()
    CUSTOM = auto()


class DeploymentStrategy(Enum):
    """Deployment strategies."""
    ROLLING_UPDATE = auto()
    RECREATE = auto()
    BLUE_GREEN = auto()
    CANARY = auto()


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ResourceRequirements:
    """Resource requirements for a node."""
    cpu_cores: float = 1.0
    memory_gb: float = 2.0
    gpu_count: int = 0
    gpu_memory_gb: float = 0.0
    storage_gb: float = 10.0
    network_bandwidth_mbps: float = 100.0


@dataclass
class NodeConfig:
    """Configuration for an FL node."""
    node_type: NodeType
    name: str
    resources: ResourceRequirements = field(default_factory=ResourceRequirements)
    image: str = "fl-ehds:latest"
    replicas: int = 1
    environment: Dict[str, str] = field(default_factory=dict)
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)

    # Scheduling
    node_selector: Dict[str, str] = field(default_factory=dict)
    affinity_rules: List[Dict[str, Any]] = field(default_factory=list)
    tolerations: List[Dict[str, Any]] = field(default_factory=list)

    # EHDS compliance
    region: Optional[str] = None
    permit_id: Optional[str] = None
    data_residency: Optional[str] = None  # EU, national


@dataclass
class ScalingConfig:
    """Auto-scaling configuration."""
    policy: ScalingPolicy = ScalingPolicy.MANUAL
    min_replicas: int = 1
    max_replicas: int = 10
    target_cpu_utilization: float = 0.7
    target_memory_utilization: float = 0.8
    scale_up_cooldown: int = 60  # seconds
    scale_down_cooldown: int = 300
    custom_metrics: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class OrchestrationConfig:
    """Main orchestration configuration."""
    orchestrator: OrchestratorType = OrchestratorType.LOCAL
    namespace: str = "fl-ehds"

    # Kubernetes settings
    kubeconfig_path: Optional[str] = None
    context: Optional[str] = None

    # Ray settings
    ray_address: str = "auto"
    ray_namespace: str = "fl-ehds"

    # Deployment
    deployment_strategy: DeploymentStrategy = DeploymentStrategy.ROLLING_UPDATE

    # Scaling
    scaling: ScalingConfig = field(default_factory=ScalingConfig)

    # Health checks
    health_check_interval: int = 30
    health_check_timeout: int = 10
    failure_threshold: int = 3

    # EHDS compliance
    enforce_data_residency: bool = True
    audit_logging: bool = True
    cross_border_enabled: bool = True


@dataclass
class NodeState:
    """Current state of an FL node."""
    node_id: str
    node_type: NodeType
    status: NodeStatus
    created_at: datetime
    last_heartbeat: Optional[datetime] = None
    current_round: int = 0
    metrics: Dict[str, float] = field(default_factory=dict)
    error: Optional[str] = None

    # Resource usage
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    gpu_usage: float = 0.0

    # EHDS info
    permit_id: Optional[str] = None
    region: Optional[str] = None


# =============================================================================
# Kubernetes Integration
# =============================================================================

class KubernetesClient:
    """
    Kubernetes client for FL orchestration.
    Manages deployments, services, and resources.
    """

    def __init__(self, config: OrchestrationConfig):
        self.config = config
        self._client = None
        self._apps_v1 = None
        self._core_v1 = None

    async def connect(self) -> bool:
        """Initialize Kubernetes connection."""
        try:
            # In production:
            # from kubernetes import client, config as k8s_config
            #
            # if self.config.kubeconfig_path:
            #     k8s_config.load_kube_config(self.config.kubeconfig_path)
            # else:
            #     k8s_config.load_incluster_config()
            #
            # self._client = client.ApiClient()
            # self._apps_v1 = client.AppsV1Api()
            # self._core_v1 = client.CoreV1Api()

            logger.info("Kubernetes client initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Kubernetes: {e}")
            return False

    async def create_namespace(self) -> bool:
        """Create namespace if not exists."""
        # In production:
        # namespace = client.V1Namespace(
        #     metadata=client.V1ObjectMeta(name=self.config.namespace)
        # )
        # try:
        #     self._core_v1.create_namespace(namespace)
        # except ApiException as e:
        #     if e.status != 409:  # Already exists
        #         raise
        logger.info(f"Namespace {self.config.namespace} ready")
        return True

    async def create_deployment(
        self,
        node_config: NodeConfig,
    ) -> str:
        """
        Create Kubernetes deployment for FL node.

        Args:
            node_config: Node configuration

        Returns:
            Deployment name
        """
        deployment_name = f"{node_config.node_type.value}-{node_config.name}"

        # Build deployment spec
        deployment_spec = self._build_deployment_spec(node_config)

        # In production:
        # self._apps_v1.create_namespaced_deployment(
        #     namespace=self.config.namespace,
        #     body=deployment_spec
        # )

        logger.info(f"Created deployment: {deployment_name}")
        return deployment_name

    async def update_deployment(
        self,
        deployment_name: str,
        node_config: NodeConfig,
    ) -> bool:
        """Update existing deployment."""
        deployment_spec = self._build_deployment_spec(node_config)

        # In production:
        # self._apps_v1.patch_namespaced_deployment(
        #     name=deployment_name,
        #     namespace=self.config.namespace,
        #     body=deployment_spec
        # )

        logger.info(f"Updated deployment: {deployment_name}")
        return True

    async def delete_deployment(self, deployment_name: str) -> bool:
        """Delete deployment."""
        # In production:
        # self._apps_v1.delete_namespaced_deployment(
        #     name=deployment_name,
        #     namespace=self.config.namespace
        # )

        logger.info(f"Deleted deployment: {deployment_name}")
        return True

    async def scale_deployment(
        self,
        deployment_name: str,
        replicas: int,
    ) -> bool:
        """Scale deployment replicas."""
        # In production:
        # scale = client.V1Scale(
        #     spec=client.V1ScaleSpec(replicas=replicas)
        # )
        # self._apps_v1.patch_namespaced_deployment_scale(
        #     name=deployment_name,
        #     namespace=self.config.namespace,
        #     body=scale
        # )

        logger.info(f"Scaled {deployment_name} to {replicas} replicas")
        return True

    async def get_deployment_status(
        self,
        deployment_name: str,
    ) -> Dict[str, Any]:
        """Get deployment status."""
        # In production:
        # deployment = self._apps_v1.read_namespaced_deployment(
        #     name=deployment_name,
        #     namespace=self.config.namespace
        # )
        # return {
        #     "replicas": deployment.status.replicas,
        #     "ready_replicas": deployment.status.ready_replicas,
        #     "available_replicas": deployment.status.available_replicas,
        # }

        return {
            "replicas": 1,
            "ready_replicas": 1,
            "available_replicas": 1,
        }

    async def list_pods(
        self,
        label_selector: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List pods in namespace."""
        # In production:
        # pods = self._core_v1.list_namespaced_pod(
        #     namespace=self.config.namespace,
        #     label_selector=label_selector
        # )
        # return [self._pod_to_dict(pod) for pod in pods.items]

        return []

    async def get_pod_logs(
        self,
        pod_name: str,
        tail_lines: int = 100,
    ) -> str:
        """Get pod logs."""
        # In production:
        # return self._core_v1.read_namespaced_pod_log(
        #     name=pod_name,
        #     namespace=self.config.namespace,
        #     tail_lines=tail_lines
        # )

        return ""

    async def create_service(
        self,
        service_name: str,
        selector: Dict[str, str],
        ports: List[Dict[str, int]],
    ) -> bool:
        """Create Kubernetes service."""
        # In production:
        # service = client.V1Service(
        #     metadata=client.V1ObjectMeta(name=service_name),
        #     spec=client.V1ServiceSpec(
        #         selector=selector,
        #         ports=[
        #             client.V1ServicePort(**p) for p in ports
        #         ]
        #     )
        # )
        # self._core_v1.create_namespaced_service(
        #     namespace=self.config.namespace,
        #     body=service
        # )

        logger.info(f"Created service: {service_name}")
        return True

    async def create_configmap(
        self,
        name: str,
        data: Dict[str, str],
    ) -> bool:
        """Create ConfigMap for FL configuration."""
        # In production:
        # configmap = client.V1ConfigMap(
        #     metadata=client.V1ObjectMeta(name=name),
        #     data=data
        # )
        # self._core_v1.create_namespaced_config_map(
        #     namespace=self.config.namespace,
        #     body=configmap
        # )

        logger.info(f"Created configmap: {name}")
        return True

    async def create_secret(
        self,
        name: str,
        data: Dict[str, str],
    ) -> bool:
        """Create Secret for sensitive data."""
        # In production:
        # import base64
        # encoded_data = {
        #     k: base64.b64encode(v.encode()).decode()
        #     for k, v in data.items()
        # }
        # secret = client.V1Secret(
        #     metadata=client.V1ObjectMeta(name=name),
        #     data=encoded_data
        # )
        # self._core_v1.create_namespaced_secret(
        #     namespace=self.config.namespace,
        #     body=secret
        # )

        logger.info(f"Created secret: {name}")
        return True

    def _build_deployment_spec(
        self,
        node_config: NodeConfig,
    ) -> Dict[str, Any]:
        """Build Kubernetes deployment specification."""
        labels = {
            "app": "fl-ehds",
            "component": node_config.node_type.value,
            "name": node_config.name,
            **node_config.labels,
        }

        # Resource limits
        resources = {
            "requests": {
                "cpu": str(node_config.resources.cpu_cores),
                "memory": f"{node_config.resources.memory_gb}Gi",
            },
            "limits": {
                "cpu": str(node_config.resources.cpu_cores * 2),
                "memory": f"{node_config.resources.memory_gb * 1.5}Gi",
            },
        }

        if node_config.resources.gpu_count > 0:
            resources["limits"]["nvidia.com/gpu"] = str(node_config.resources.gpu_count)

        # Environment variables
        env_vars = [
            {"name": k, "value": v}
            for k, v in node_config.environment.items()
        ]

        # Add EHDS metadata
        if node_config.permit_id:
            env_vars.append({
                "name": "EHDS_PERMIT_ID",
                "value": node_config.permit_id
            })

        if node_config.region:
            env_vars.append({
                "name": "EHDS_REGION",
                "value": node_config.region
            })

        return {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": f"{node_config.node_type.value}-{node_config.name}",
                "labels": labels,
                "annotations": node_config.annotations,
            },
            "spec": {
                "replicas": node_config.replicas,
                "selector": {"matchLabels": labels},
                "template": {
                    "metadata": {"labels": labels},
                    "spec": {
                        "containers": [{
                            "name": "fl-node",
                            "image": node_config.image,
                            "resources": resources,
                            "env": env_vars,
                            "ports": [
                                {"containerPort": 50051, "name": "grpc"},
                                {"containerPort": 8765, "name": "websocket"},
                                {"containerPort": 8080, "name": "metrics"},
                            ],
                            "livenessProbe": {
                                "httpGet": {"path": "/health", "port": 8080},
                                "initialDelaySeconds": 30,
                                "periodSeconds": 10,
                            },
                            "readinessProbe": {
                                "httpGet": {"path": "/ready", "port": 8080},
                                "initialDelaySeconds": 5,
                                "periodSeconds": 5,
                            },
                        }],
                        "nodeSelector": node_config.node_selector,
                        "tolerations": node_config.tolerations,
                    },
                },
            },
        }


# =============================================================================
# Ray Integration
# =============================================================================

class RayOrchestrator:
    """
    Ray-based orchestration for distributed FL training.
    Provides dynamic task scheduling and resource management.
    """

    def __init__(self, config: OrchestrationConfig):
        self.config = config
        self._initialized = False
        self._actors: Dict[str, Any] = {}

    async def initialize(self) -> bool:
        """Initialize Ray cluster connection."""
        try:
            # In production:
            # import ray
            # ray.init(
            #     address=self.config.ray_address,
            #     namespace=self.config.ray_namespace,
            # )

            self._initialized = True
            logger.info(f"Ray initialized: {self.config.ray_address}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Ray: {e}")
            return False

    async def shutdown(self) -> None:
        """Shutdown Ray connection."""
        if self._initialized:
            # In production: ray.shutdown()
            self._initialized = False

    async def create_actor(
        self,
        actor_class: Any,
        name: str,
        resources: ResourceRequirements,
        **kwargs,
    ) -> str:
        """
        Create Ray actor for FL node.

        Args:
            actor_class: Actor class
            name: Actor name
            resources: Resource requirements
            **kwargs: Actor arguments

        Returns:
            Actor ID
        """
        actor_id = f"{name}_{uuid.uuid4().hex[:8]}"

        # In production:
        # @ray.remote(
        #     num_cpus=resources.cpu_cores,
        #     num_gpus=resources.gpu_count,
        #     memory=resources.memory_gb * 1024 * 1024 * 1024,
        # )
        # class RemoteActor(actor_class):
        #     pass
        #
        # actor = RemoteActor.options(name=actor_id).remote(**kwargs)
        # self._actors[actor_id] = actor

        logger.info(f"Created Ray actor: {actor_id}")
        return actor_id

    async def destroy_actor(self, actor_id: str) -> bool:
        """Destroy Ray actor."""
        if actor_id in self._actors:
            # In production: ray.kill(self._actors[actor_id])
            del self._actors[actor_id]
            logger.info(f"Destroyed actor: {actor_id}")
            return True
        return False

    async def call_actor(
        self,
        actor_id: str,
        method: str,
        *args,
        **kwargs,
    ) -> Any:
        """Call method on Ray actor."""
        if actor_id not in self._actors:
            raise ValueError(f"Actor not found: {actor_id}")

        # In production:
        # actor = self._actors[actor_id]
        # result = await getattr(actor, method).remote(*args, **kwargs)
        # return ray.get(result)

        return None

    async def submit_task(
        self,
        func: Callable,
        *args,
        resources: Optional[ResourceRequirements] = None,
        **kwargs,
    ) -> Any:
        """
        Submit task to Ray cluster.

        Args:
            func: Function to execute
            *args: Function arguments
            resources: Resource requirements
            **kwargs: Function keyword arguments

        Returns:
            Task result
        """
        # In production:
        # @ray.remote
        # def remote_func(*args, **kwargs):
        #     return func(*args, **kwargs)
        #
        # ref = remote_func.remote(*args, **kwargs)
        # return ray.get(ref)

        return func(*args, **kwargs)

    async def map_tasks(
        self,
        func: Callable,
        inputs: List[Any],
        resources: Optional[ResourceRequirements] = None,
    ) -> List[Any]:
        """
        Map function over inputs in parallel.

        Args:
            func: Function to execute
            inputs: List of inputs
            resources: Resource requirements

        Returns:
            List of results
        """
        # In production:
        # @ray.remote
        # def remote_func(x):
        #     return func(x)
        #
        # refs = [remote_func.remote(x) for x in inputs]
        # return ray.get(refs)

        return [func(x) for x in inputs]

    async def get_cluster_resources(self) -> Dict[str, Any]:
        """Get available cluster resources."""
        # In production:
        # resources = ray.cluster_resources()
        # return {
        #     "cpu": resources.get("CPU", 0),
        #     "gpu": resources.get("GPU", 0),
        #     "memory": resources.get("memory", 0),
        #     "nodes": len(ray.nodes()),
        # }

        return {
            "cpu": 8,
            "gpu": 0,
            "memory": 16 * 1024 * 1024 * 1024,
            "nodes": 1,
        }


# =============================================================================
# FL Training Actors
# =============================================================================

class FLClientActor:
    """
    Ray actor for FL client training.
    Handles local model training and gradient computation.
    """

    def __init__(
        self,
        client_id: str,
        model_config: Dict[str, Any],
        permit_id: str,
    ):
        self.client_id = client_id
        self.model_config = model_config
        self.permit_id = permit_id
        self._model = None
        self._data = None

    def initialize(self, initial_weights: Dict[str, np.ndarray]) -> bool:
        """Initialize client with model weights."""
        self._model = initial_weights.copy()
        logger.info(f"Client {self.client_id} initialized")
        return True

    def load_data(self, data: Any) -> bool:
        """Load local training data."""
        self._data = data
        return True

    def train(
        self,
        global_weights: Dict[str, np.ndarray],
        num_epochs: int = 1,
        learning_rate: float = 0.01,
    ) -> Dict[str, Any]:
        """
        Perform local training.

        Args:
            global_weights: Current global model weights
            num_epochs: Number of local epochs
            learning_rate: Learning rate

        Returns:
            Training results with updated weights
        """
        # Simulate local training
        local_weights = {}
        for name, weights in global_weights.items():
            # Simulate gradient update
            gradient = np.random.randn(*weights.shape).astype(np.float32) * 0.01
            local_weights[name] = weights - learning_rate * gradient

        return {
            "client_id": self.client_id,
            "weights": local_weights,
            "num_samples": 1000,  # Simulated
            "metrics": {
                "loss": np.random.uniform(0.1, 1.0),
                "accuracy": np.random.uniform(0.7, 0.95),
            },
        }

    def get_status(self) -> Dict[str, Any]:
        """Get client status."""
        return {
            "client_id": self.client_id,
            "status": "ready",
            "permit_id": self.permit_id,
        }


class FLAggregatorActor:
    """
    Ray actor for FL aggregation.
    Handles federated averaging and model updates.
    """

    def __init__(
        self,
        aggregator_id: str,
        aggregation_strategy: str = "fedavg",
    ):
        self.aggregator_id = aggregator_id
        self.strategy = aggregation_strategy
        self._global_model = None

    def initialize(self, initial_weights: Dict[str, np.ndarray]) -> bool:
        """Initialize with global model."""
        self._global_model = initial_weights.copy()
        return True

    def aggregate(
        self,
        client_updates: List[Dict[str, Any]],
    ) -> Dict[str, np.ndarray]:
        """
        Aggregate client updates.

        Args:
            client_updates: List of client update dictionaries

        Returns:
            Aggregated global model weights
        """
        if not client_updates:
            return self._global_model

        # FedAvg aggregation
        total_samples = sum(u.get("num_samples", 1) for u in client_updates)
        aggregated = {}

        # Get layer names from first update
        layer_names = client_updates[0]["weights"].keys()

        for name in layer_names:
            weighted_sum = None
            for update in client_updates:
                weight = update.get("num_samples", 1) / total_samples
                layer_weights = update["weights"][name] * weight

                if weighted_sum is None:
                    weighted_sum = layer_weights
                else:
                    weighted_sum += layer_weights

            aggregated[name] = weighted_sum

        self._global_model = aggregated
        return aggregated

    def get_global_model(self) -> Dict[str, np.ndarray]:
        """Get current global model."""
        return self._global_model


# =============================================================================
# Auto-Scaler
# =============================================================================

class AutoScaler:
    """
    Auto-scaling controller for FL cluster.
    Monitors metrics and adjusts capacity.
    """

    def __init__(
        self,
        config: ScalingConfig,
        kubernetes_client: Optional[KubernetesClient] = None,
        ray_orchestrator: Optional[RayOrchestrator] = None,
    ):
        self.config = config
        self._k8s = kubernetes_client
        self._ray = ray_orchestrator
        self._last_scale_up = None
        self._last_scale_down = None
        self._current_metrics: Dict[str, float] = {}

    async def evaluate_scaling(
        self,
        deployment_name: str,
        current_replicas: int,
        metrics: Dict[str, float],
    ) -> Optional[int]:
        """
        Evaluate if scaling is needed.

        Args:
            deployment_name: Deployment to evaluate
            current_replicas: Current replica count
            metrics: Current metrics

        Returns:
            New replica count if scaling needed, None otherwise
        """
        self._current_metrics = metrics
        now = datetime.now()

        # Check cooldowns
        if self._last_scale_up:
            if (now - self._last_scale_up).seconds < self.config.scale_up_cooldown:
                return None

        if self._last_scale_down:
            if (now - self._last_scale_down).seconds < self.config.scale_down_cooldown:
                return None

        # Evaluate based on policy
        if self.config.policy == ScalingPolicy.CPU_BASED:
            return self._evaluate_cpu_scaling(current_replicas, metrics)
        elif self.config.policy == ScalingPolicy.MEMORY_BASED:
            return self._evaluate_memory_scaling(current_replicas, metrics)
        elif self.config.policy == ScalingPolicy.QUEUE_BASED:
            return self._evaluate_queue_scaling(current_replicas, metrics)

        return None

    def _evaluate_cpu_scaling(
        self,
        current_replicas: int,
        metrics: Dict[str, float],
    ) -> Optional[int]:
        """Evaluate CPU-based scaling."""
        cpu_usage = metrics.get("cpu_utilization", 0)

        if cpu_usage > self.config.target_cpu_utilization:
            # Scale up
            new_replicas = min(
                current_replicas + 1,
                self.config.max_replicas
            )
            if new_replicas > current_replicas:
                self._last_scale_up = datetime.now()
                return new_replicas

        elif cpu_usage < self.config.target_cpu_utilization * 0.5:
            # Scale down
            new_replicas = max(
                current_replicas - 1,
                self.config.min_replicas
            )
            if new_replicas < current_replicas:
                self._last_scale_down = datetime.now()
                return new_replicas

        return None

    def _evaluate_memory_scaling(
        self,
        current_replicas: int,
        metrics: Dict[str, float],
    ) -> Optional[int]:
        """Evaluate memory-based scaling."""
        memory_usage = metrics.get("memory_utilization", 0)

        if memory_usage > self.config.target_memory_utilization:
            new_replicas = min(
                current_replicas + 1,
                self.config.max_replicas
            )
            if new_replicas > current_replicas:
                self._last_scale_up = datetime.now()
                return new_replicas

        return None

    def _evaluate_queue_scaling(
        self,
        current_replicas: int,
        metrics: Dict[str, float],
    ) -> Optional[int]:
        """Evaluate queue-based scaling."""
        queue_length = metrics.get("pending_tasks", 0)
        processing_rate = metrics.get("tasks_per_second", 1)

        # Estimate needed replicas
        if processing_rate > 0:
            estimated_time = queue_length / processing_rate
            if estimated_time > 300:  # More than 5 minutes backlog
                new_replicas = min(
                    current_replicas + 2,
                    self.config.max_replicas
                )
                if new_replicas > current_replicas:
                    self._last_scale_up = datetime.now()
                    return new_replicas

        return None

    async def apply_scaling(
        self,
        deployment_name: str,
        new_replicas: int,
    ) -> bool:
        """Apply scaling decision."""
        if self._k8s:
            return await self._k8s.scale_deployment(deployment_name, new_replicas)
        return False


# =============================================================================
# Orchestration Manager
# =============================================================================

class OrchestrationManager:
    """
    Main orchestration manager for FL-EHDS.
    Coordinates Kubernetes and Ray resources.
    """

    def __init__(self, config: Optional[OrchestrationConfig] = None):
        self.config = config or OrchestrationConfig()

        # Initialize components based on orchestrator type
        self._k8s: Optional[KubernetesClient] = None
        self._ray: Optional[RayOrchestrator] = None

        if self.config.orchestrator in (
            OrchestratorType.KUBERNETES,
            OrchestratorType.HYBRID
        ):
            self._k8s = KubernetesClient(self.config)

        if self.config.orchestrator in (
            OrchestratorType.RAY,
            OrchestratorType.HYBRID
        ):
            self._ray = RayOrchestrator(self.config)

        self._auto_scaler = AutoScaler(
            self.config.scaling,
            self._k8s,
            self._ray
        )

        # Node registry
        self._nodes: Dict[str, NodeState] = {}
        self._deployments: Dict[str, str] = {}

        # Background tasks
        self._health_check_task: Optional[asyncio.Task] = None
        self._scaling_task: Optional[asyncio.Task] = None
        self._running = False

    async def start(self) -> bool:
        """Start orchestration manager."""
        self._running = True

        # Initialize backends
        if self._k8s:
            if not await self._k8s.connect():
                return False
            await self._k8s.create_namespace()

        if self._ray:
            if not await self._ray.initialize():
                return False

        # Start background tasks
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        self._scaling_task = asyncio.create_task(self._auto_scaling_loop())

        logger.info("Orchestration manager started")
        return True

    async def stop(self) -> None:
        """Stop orchestration manager."""
        self._running = False

        if self._health_check_task:
            self._health_check_task.cancel()

        if self._scaling_task:
            self._scaling_task.cancel()

        if self._ray:
            await self._ray.shutdown()

        logger.info("Orchestration manager stopped")

    async def deploy_fl_cluster(
        self,
        num_clients: int,
        num_aggregators: int = 1,
        client_resources: Optional[ResourceRequirements] = None,
        aggregator_resources: Optional[ResourceRequirements] = None,
        permit_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Deploy complete FL cluster.

        Args:
            num_clients: Number of client nodes
            num_aggregators: Number of aggregator nodes
            client_resources: Client resource requirements
            aggregator_resources: Aggregator resource requirements
            permit_id: EHDS permit ID

        Returns:
            Deployment information
        """
        deployment_id = f"fl-cluster-{uuid.uuid4().hex[:8]}"
        client_resources = client_resources or ResourceRequirements()
        aggregator_resources = aggregator_resources or ResourceRequirements(
            cpu_cores=2.0,
            memory_gb=4.0,
        )

        deployed_nodes = []

        # Deploy aggregators
        for i in range(num_aggregators):
            aggregator_config = NodeConfig(
                node_type=NodeType.AGGREGATOR,
                name=f"agg-{i}",
                resources=aggregator_resources,
                permit_id=permit_id,
            )
            node_id = await self.create_node(aggregator_config)
            deployed_nodes.append(node_id)

        # Deploy clients
        for i in range(num_clients):
            client_config = NodeConfig(
                node_type=NodeType.CLIENT,
                name=f"client-{i}",
                resources=client_resources,
                permit_id=permit_id,
            )
            node_id = await self.create_node(client_config)
            deployed_nodes.append(node_id)

        logger.info(
            f"Deployed FL cluster {deployment_id}: "
            f"{num_aggregators} aggregators, {num_clients} clients"
        )

        return {
            "deployment_id": deployment_id,
            "nodes": deployed_nodes,
            "aggregators": num_aggregators,
            "clients": num_clients,
        }

    async def create_node(
        self,
        node_config: NodeConfig,
    ) -> str:
        """
        Create FL node.

        Args:
            node_config: Node configuration

        Returns:
            Node ID
        """
        node_id = f"{node_config.node_type.value}-{node_config.name}-{uuid.uuid4().hex[:8]}"

        # Create node state
        state = NodeState(
            node_id=node_id,
            node_type=node_config.node_type,
            status=NodeStatus.CREATING,
            created_at=datetime.now(),
            permit_id=node_config.permit_id,
            region=node_config.region,
        )
        self._nodes[node_id] = state

        # Deploy based on orchestrator
        if self._k8s and self.config.orchestrator != OrchestratorType.RAY:
            deployment_name = await self._k8s.create_deployment(node_config)
            self._deployments[node_id] = deployment_name

        if self._ray and self.config.orchestrator != OrchestratorType.KUBERNETES:
            # Create Ray actor
            actor_class = (
                FLClientActor if node_config.node_type == NodeType.CLIENT
                else FLAggregatorActor
            )
            await self._ray.create_actor(
                actor_class,
                node_id,
                node_config.resources,
            )

        state.status = NodeStatus.RUNNING
        logger.info(f"Created node: {node_id}")

        return node_id

    async def destroy_node(self, node_id: str) -> bool:
        """Destroy FL node."""
        if node_id not in self._nodes:
            return False

        self._nodes[node_id].status = NodeStatus.DRAINING

        # Remove from orchestrators
        if node_id in self._deployments:
            await self._k8s.delete_deployment(self._deployments[node_id])
            del self._deployments[node_id]

        if self._ray:
            await self._ray.destroy_actor(node_id)

        self._nodes[node_id].status = NodeStatus.TERMINATED
        del self._nodes[node_id]

        logger.info(f"Destroyed node: {node_id}")
        return True

    async def get_node_status(self, node_id: str) -> Optional[NodeState]:
        """Get node status."""
        return self._nodes.get(node_id)

    async def list_nodes(
        self,
        node_type: Optional[NodeType] = None,
        status: Optional[NodeStatus] = None,
    ) -> List[NodeState]:
        """List nodes with optional filters."""
        nodes = list(self._nodes.values())

        if node_type:
            nodes = [n for n in nodes if n.node_type == node_type]

        if status:
            nodes = [n for n in nodes if n.status == status]

        return nodes

    async def scale_nodes(
        self,
        node_type: NodeType,
        target_count: int,
    ) -> Dict[str, Any]:
        """Scale nodes of a specific type."""
        current_nodes = await self.list_nodes(node_type=node_type)
        current_count = len(current_nodes)

        created = []
        destroyed = []

        if target_count > current_count:
            # Scale up
            for i in range(target_count - current_count):
                config = NodeConfig(
                    node_type=node_type,
                    name=f"scaled-{i}",
                )
                node_id = await self.create_node(config)
                created.append(node_id)

        elif target_count < current_count:
            # Scale down
            to_remove = current_count - target_count
            for node in current_nodes[:to_remove]:
                await self.destroy_node(node.node_id)
                destroyed.append(node.node_id)

        return {
            "node_type": node_type.value,
            "previous_count": current_count,
            "target_count": target_count,
            "created": created,
            "destroyed": destroyed,
        }

    async def run_distributed_training(
        self,
        initial_weights: Dict[str, np.ndarray],
        num_rounds: int,
        clients_per_round: int,
    ) -> Dict[str, Any]:
        """
        Run distributed FL training using Ray.

        Args:
            initial_weights: Initial model weights
            num_rounds: Number of FL rounds
            clients_per_round: Clients participating per round

        Returns:
            Training results
        """
        if not self._ray:
            raise RuntimeError("Ray not initialized")

        # Get available client nodes
        client_nodes = await self.list_nodes(
            node_type=NodeType.CLIENT,
            status=NodeStatus.RUNNING
        )

        if len(client_nodes) < clients_per_round:
            raise ValueError(
                f"Not enough clients: {len(client_nodes)} < {clients_per_round}"
            )

        global_weights = initial_weights.copy()
        history = []

        for round_num in range(num_rounds):
            # Select clients for this round
            selected = np.random.choice(
                client_nodes,
                size=clients_per_round,
                replace=False
            )

            # Parallel client training
            async def train_client(node: NodeState):
                return await self._ray.call_actor(
                    node.node_id,
                    "train",
                    global_weights,
                )

            client_results = await asyncio.gather(*[
                train_client(node) for node in selected
            ])

            # Aggregate results
            aggregator_nodes = await self.list_nodes(
                node_type=NodeType.AGGREGATOR,
                status=NodeStatus.RUNNING
            )

            if aggregator_nodes:
                global_weights = await self._ray.call_actor(
                    aggregator_nodes[0].node_id,
                    "aggregate",
                    client_results,
                )

            # Record metrics
            avg_loss = np.mean([r["metrics"]["loss"] for r in client_results])
            avg_acc = np.mean([r["metrics"]["accuracy"] for r in client_results])

            history.append({
                "round": round_num,
                "loss": avg_loss,
                "accuracy": avg_acc,
                "clients": len(client_results),
            })

            logger.info(
                f"Round {round_num}: loss={avg_loss:.4f}, acc={avg_acc:.4f}"
            )

        return {
            "final_weights": global_weights,
            "history": history,
            "rounds_completed": num_rounds,
        }

    async def _health_check_loop(self) -> None:
        """Background health check loop."""
        while self._running:
            try:
                for node_id, state in list(self._nodes.items()):
                    if state.status in (NodeStatus.RUNNING, NodeStatus.READY):
                        # Check health
                        if self._k8s and node_id in self._deployments:
                            status = await self._k8s.get_deployment_status(
                                self._deployments[node_id]
                            )
                            if status.get("ready_replicas", 0) == 0:
                                state.status = NodeStatus.FAILED

                        state.last_heartbeat = datetime.now()

                await asyncio.sleep(self.config.health_check_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error: {e}")

    async def _auto_scaling_loop(self) -> None:
        """Background auto-scaling loop."""
        while self._running:
            try:
                if self.config.scaling.policy != ScalingPolicy.MANUAL:
                    # Collect metrics
                    metrics = await self._collect_cluster_metrics()

                    # Evaluate scaling for each deployment
                    for node_id, deployment_name in self._deployments.items():
                        status = await self._k8s.get_deployment_status(deployment_name)
                        current_replicas = status.get("replicas", 1)

                        new_replicas = await self._auto_scaler.evaluate_scaling(
                            deployment_name,
                            current_replicas,
                            metrics,
                        )

                        if new_replicas is not None:
                            await self._auto_scaler.apply_scaling(
                                deployment_name,
                                new_replicas
                            )

                await asyncio.sleep(60)  # Check every minute

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Auto-scaling error: {e}")

    async def _collect_cluster_metrics(self) -> Dict[str, float]:
        """Collect cluster-wide metrics."""
        metrics = {
            "cpu_utilization": 0.0,
            "memory_utilization": 0.0,
            "pending_tasks": 0,
            "active_nodes": len(self._nodes),
        }

        # Aggregate node metrics
        for state in self._nodes.values():
            metrics["cpu_utilization"] += state.cpu_usage
            metrics["memory_utilization"] += state.memory_usage

        if self._nodes:
            metrics["cpu_utilization"] /= len(self._nodes)
            metrics["memory_utilization"] /= len(self._nodes)

        return metrics


# =============================================================================
# INFRASTRUCTURE AS CODE (IaC) TEMPLATES
# =============================================================================

class IaCProvider(Enum):
    """Supported IaC providers."""
    TERRAFORM = auto()
    PULUMI = auto()
    CDK = auto()  # AWS CDK


@dataclass
class IaCConfig:
    """Configuration for IaC generation."""
    provider: IaCProvider = IaCProvider.TERRAFORM
    cloud_provider: str = "aws"  # aws, azure, gcp
    region: str = "eu-west-1"
    environment: str = "production"

    # Kubernetes settings
    k8s_version: str = "1.28"
    node_pool_size: int = 3
    node_pool_min: int = 1
    node_pool_max: int = 10
    node_machine_type: str = "m5.xlarge"
    gpu_enabled: bool = False
    gpu_machine_type: str = "p3.2xlarge"

    # Networking
    vpc_cidr: str = "10.0.0.0/16"
    enable_private_endpoints: bool = True

    # Security
    enable_encryption: bool = True
    kms_key_id: Optional[str] = None

    # EHDS compliance
    data_residency: str = "eu"
    enable_audit_logs: bool = True

    # Tags
    tags: Dict[str, str] = field(default_factory=lambda: {
        "project": "fl-ehds",
        "managed-by": "terraform",
    })


class TerraformGenerator:
    """
    Generates Terraform configurations for FL-EHDS infrastructure.

    Supports AWS, Azure, and GCP deployments with EHDS compliance.
    """

    def __init__(self, config: IaCConfig):
        self.config = config

    def generate_all(self) -> Dict[str, str]:
        """Generate all Terraform files."""
        files = {}

        files["main.tf"] = self._generate_main()
        files["variables.tf"] = self._generate_variables()
        files["outputs.tf"] = self._generate_outputs()
        files["providers.tf"] = self._generate_providers()
        files["network.tf"] = self._generate_network()
        files["kubernetes.tf"] = self._generate_kubernetes()
        files["fl-ehds.tf"] = self._generate_fl_ehds()
        files["monitoring.tf"] = self._generate_monitoring()
        files["security.tf"] = self._generate_security()

        if self.config.enable_audit_logs:
            files["audit.tf"] = self._generate_audit()

        return files

    def _generate_main(self) -> str:
        """Generate main.tf."""
        return f'''# FL-EHDS Infrastructure
# Generated by FL-EHDS Framework
# Provider: {self.config.cloud_provider.upper()}
# Region: {self.config.region}

terraform {{
  required_version = ">= 1.5.0"

  required_providers {{
    {"aws" if self.config.cloud_provider == "aws" else self.config.cloud_provider} = {{
      source  = "{self._get_provider_source()}"
      version = "~> {self._get_provider_version()}"
    }}
    kubernetes = {{
      source  = "hashicorp/kubernetes"
      version = "~> 2.23"
    }}
    helm = {{
      source  = "hashicorp/helm"
      version = "~> 2.11"
    }}
  }}

  backend "s3" {{
    bucket         = "fl-ehds-terraform-state"
    key            = "${{var.environment}}/terraform.tfstate"
    region         = "{self.config.region}"
    encrypt        = true
    dynamodb_table = "fl-ehds-terraform-locks"
  }}
}}

locals {{
  name_prefix = "fl-ehds-${{var.environment}}"
  common_tags = merge(var.tags, {{
    Environment = var.environment
    Project     = "fl-ehds"
    ManagedBy   = "terraform"
    EHDSCompliant = "true"
  }})
}}
'''

    def _generate_variables(self) -> str:
        """Generate variables.tf."""
        return f'''# FL-EHDS Terraform Variables

variable "environment" {{
  description = "Deployment environment"
  type        = string
  default     = "{self.config.environment}"

  validation {{
    condition     = contains(["development", "staging", "production"], var.environment)
    error_message = "Environment must be development, staging, or production."
  }}
}}

variable "region" {{
  description = "Cloud region for deployment"
  type        = string
  default     = "{self.config.region}"
}}

variable "vpc_cidr" {{
  description = "CIDR block for VPC"
  type        = string
  default     = "{self.config.vpc_cidr}"
}}

variable "kubernetes_version" {{
  description = "Kubernetes version"
  type        = string
  default     = "{self.config.k8s_version}"
}}

variable "node_pool_size" {{
  description = "Initial node pool size"
  type        = number
  default     = {self.config.node_pool_size}
}}

variable "node_pool_min" {{
  description = "Minimum node pool size"
  type        = number
  default     = {self.config.node_pool_min}
}}

variable "node_pool_max" {{
  description = "Maximum node pool size"
  type        = number
  default     = {self.config.node_pool_max}
}}

variable "node_machine_type" {{
  description = "Machine type for nodes"
  type        = string
  default     = "{self.config.node_machine_type}"
}}

variable "gpu_enabled" {{
  description = "Enable GPU nodes"
  type        = bool
  default     = {str(self.config.gpu_enabled).lower()}
}}

variable "enable_encryption" {{
  description = "Enable encryption at rest"
  type        = bool
  default     = {str(self.config.enable_encryption).lower()}
}}

variable "data_residency" {{
  description = "Data residency requirement (eu, national)"
  type        = string
  default     = "{self.config.data_residency}"
}}

variable "tags" {{
  description = "Tags to apply to resources"
  type        = map(string)
  default     = {json.dumps(self.config.tags)}
}}

variable "fl_aggregator_replicas" {{
  description = "Number of FL aggregator replicas"
  type        = number
  default     = 2
}}

variable "fl_client_max_replicas" {{
  description = "Maximum number of FL client replicas"
  type        = number
  default     = 100
}}
'''

    def _generate_outputs(self) -> str:
        """Generate outputs.tf."""
        return '''# FL-EHDS Terraform Outputs

output "cluster_endpoint" {
  description = "Kubernetes cluster endpoint"
  value       = module.kubernetes.cluster_endpoint
  sensitive   = true
}

output "cluster_name" {
  description = "Kubernetes cluster name"
  value       = module.kubernetes.cluster_name
}

output "cluster_ca_certificate" {
  description = "Cluster CA certificate"
  value       = module.kubernetes.cluster_ca_certificate
  sensitive   = true
}

output "fl_aggregator_endpoint" {
  description = "FL aggregator service endpoint"
  value       = kubernetes_service.fl_aggregator.status[0].load_balancer[0].ingress[0].hostname
}

output "monitoring_dashboard_url" {
  description = "Grafana dashboard URL"
  value       = "https://${kubernetes_ingress_v1.grafana.status[0].load_balancer[0].ingress[0].hostname}"
}

output "prometheus_endpoint" {
  description = "Prometheus endpoint"
  value       = "http://${kubernetes_service.prometheus.metadata[0].name}:9090"
}

output "vpc_id" {
  description = "VPC ID"
  value       = module.network.vpc_id
}

output "private_subnet_ids" {
  description = "Private subnet IDs"
  value       = module.network.private_subnet_ids
}
'''

    def _generate_providers(self) -> str:
        """Generate providers.tf."""
        if self.config.cloud_provider == "aws":
            return f'''# AWS Provider Configuration

provider "aws" {{
  region = var.region

  default_tags {{
    tags = local.common_tags
  }}
}}

provider "kubernetes" {{
  host                   = module.kubernetes.cluster_endpoint
  cluster_ca_certificate = base64decode(module.kubernetes.cluster_ca_certificate)
  exec {{
    api_version = "client.authentication.k8s.io/v1beta1"
    command     = "aws"
    args        = ["eks", "get-token", "--cluster-name", module.kubernetes.cluster_name]
  }}
}}

provider "helm" {{
  kubernetes {{
    host                   = module.kubernetes.cluster_endpoint
    cluster_ca_certificate = base64decode(module.kubernetes.cluster_ca_certificate)
    exec {{
      api_version = "client.authentication.k8s.io/v1beta1"
      command     = "aws"
      args        = ["eks", "get-token", "--cluster-name", module.kubernetes.cluster_name]
    }}
  }}
}}
'''
        elif self.config.cloud_provider == "azure":
            return '''# Azure Provider Configuration

provider "azurerm" {
  features {}
}

provider "kubernetes" {
  host                   = module.kubernetes.cluster_endpoint
  client_certificate     = base64decode(module.kubernetes.client_certificate)
  client_key             = base64decode(module.kubernetes.client_key)
  cluster_ca_certificate = base64decode(module.kubernetes.cluster_ca_certificate)
}

provider "helm" {
  kubernetes {
    host                   = module.kubernetes.cluster_endpoint
    client_certificate     = base64decode(module.kubernetes.client_certificate)
    client_key             = base64decode(module.kubernetes.client_key)
    cluster_ca_certificate = base64decode(module.kubernetes.cluster_ca_certificate)
  }
}
'''
        else:  # GCP
            return '''# GCP Provider Configuration

provider "google" {
  project = var.project_id
  region  = var.region
}

provider "kubernetes" {
  host                   = "https://${module.kubernetes.cluster_endpoint}"
  cluster_ca_certificate = base64decode(module.kubernetes.cluster_ca_certificate)
  token                  = data.google_client_config.default.access_token
}

provider "helm" {
  kubernetes {
    host                   = "https://${module.kubernetes.cluster_endpoint}"
    cluster_ca_certificate = base64decode(module.kubernetes.cluster_ca_certificate)
    token                  = data.google_client_config.default.access_token
  }
}

data "google_client_config" "default" {}
'''

    def _generate_network(self) -> str:
        """Generate network.tf."""
        return '''# FL-EHDS Network Configuration

module "network" {
  source = "./modules/network"

  name_prefix         = local.name_prefix
  vpc_cidr            = var.vpc_cidr
  availability_zones  = data.aws_availability_zones.available.names
  enable_nat_gateway  = true
  single_nat_gateway  = var.environment != "production"

  # EHDS compliance: private endpoints
  enable_vpc_endpoints = true
  vpc_endpoint_services = [
    "ecr.api",
    "ecr.dkr",
    "s3",
    "logs",
    "monitoring",
  ]

  tags = local.common_tags
}

data "aws_availability_zones" "available" {
  state = "available"
}
'''

    def _generate_kubernetes(self) -> str:
        """Generate kubernetes.tf."""
        return f'''# FL-EHDS Kubernetes Cluster

module "kubernetes" {{
  source = "./modules/kubernetes"

  cluster_name    = "${{local.name_prefix}}-cluster"
  cluster_version = var.kubernetes_version
  vpc_id          = module.network.vpc_id
  subnet_ids      = module.network.private_subnet_ids

  # Node groups
  node_groups = {{
    fl_workers = {{
      instance_types = [var.node_machine_type]
      desired_size   = var.node_pool_size
      min_size       = var.node_pool_min
      max_size       = var.node_pool_max
      disk_size      = 100

      labels = {{
        "fl-ehds/role" = "worker"
      }}

      taints = []
    }}

    fl_aggregators = {{
      instance_types = ["m5.2xlarge"]
      desired_size   = 2
      min_size       = 2
      max_size       = 5
      disk_size      = 100

      labels = {{
        "fl-ehds/role" = "aggregator"
      }}

      taints = [{{
        key    = "fl-ehds/role"
        value  = "aggregator"
        effect = "NO_SCHEDULE"
      }}]
    }}
  }}

  {"" if not self.config.gpu_enabled else '''
  # GPU node group
  gpu_node_groups = {
    fl_gpu_workers = {
      instance_types = [var.gpu_machine_type]
      desired_size   = 0
      min_size       = 0
      max_size       = var.node_pool_max
      disk_size      = 200

      labels = {
        "fl-ehds/role"     = "gpu-worker"
        "nvidia.com/gpu"   = "true"
      }

      taints = [{
        key    = "nvidia.com/gpu"
        value  = "true"
        effect = "NO_SCHEDULE"
      }]
    }
  }
'''}

  # Encryption
  enable_encryption = var.enable_encryption
  kms_key_arn       = var.enable_encryption ? aws_kms_key.eks[0].arn : null

  # EHDS compliance
  enable_audit_logging = true
  audit_log_retention  = 365

  tags = local.common_tags
}}

# Kubernetes namespace for FL-EHDS
resource "kubernetes_namespace" "fl_ehds" {{
  metadata {{
    name = "fl-ehds"

    labels = {{
      "app.kubernetes.io/name"       = "fl-ehds"
      "app.kubernetes.io/managed-by" = "terraform"
      "ehds.eu/compliant"            = "true"
    }}
  }}
}}
'''

    def _generate_fl_ehds(self) -> str:
        """Generate fl-ehds.tf for FL-EHDS specific resources."""
        return '''# FL-EHDS Application Resources

# FL Aggregator Deployment
resource "kubernetes_deployment" "fl_aggregator" {
  metadata {
    name      = "fl-aggregator"
    namespace = kubernetes_namespace.fl_ehds.metadata[0].name

    labels = {
      app = "fl-aggregator"
    }
  }

  spec {
    replicas = var.fl_aggregator_replicas

    selector {
      match_labels = {
        app = "fl-aggregator"
      }
    }

    template {
      metadata {
        labels = {
          app = "fl-aggregator"
        }

        annotations = {
          "prometheus.io/scrape" = "true"
          "prometheus.io/port"   = "8080"
        }
      }

      spec {
        node_selector = {
          "fl-ehds/role" = "aggregator"
        }

        toleration {
          key      = "fl-ehds/role"
          operator = "Equal"
          value    = "aggregator"
          effect   = "NoSchedule"
        }

        container {
          name  = "aggregator"
          image = "fl-ehds/aggregator:latest"

          port {
            container_port = 50051
            name           = "grpc"
          }

          port {
            container_port = 8080
            name           = "metrics"
          }

          env {
            name  = "EHDS_COMPLIANT"
            value = "true"
          }

          env {
            name = "EHDS_PERMIT_ID"
            value_from {
              secret_key_ref {
                name = kubernetes_secret.fl_ehds_config.metadata[0].name
                key  = "permit-id"
              }
            }
          }

          resources {
            requests = {
              cpu    = "2"
              memory = "4Gi"
            }
            limits = {
              cpu    = "4"
              memory = "8Gi"
            }
          }

          liveness_probe {
            http_get {
              path = "/health"
              port = 8080
            }
            initial_delay_seconds = 30
            period_seconds        = 10
          }

          readiness_probe {
            http_get {
              path = "/ready"
              port = 8080
            }
            initial_delay_seconds = 5
            period_seconds        = 5
          }
        }
      }
    }
  }
}

# FL Aggregator Service
resource "kubernetes_service" "fl_aggregator" {
  metadata {
    name      = "fl-aggregator"
    namespace = kubernetes_namespace.fl_ehds.metadata[0].name
  }

  spec {
    selector = {
      app = "fl-aggregator"
    }

    port {
      name        = "grpc"
      port        = 50051
      target_port = 50051
    }

    port {
      name        = "metrics"
      port        = 8080
      target_port = 8080
    }

    type = "LoadBalancer"

    # Annotations for AWS NLB
    annotations = {
      "service.beta.kubernetes.io/aws-load-balancer-type"            = "nlb"
      "service.beta.kubernetes.io/aws-load-balancer-internal"        = "true"
      "service.beta.kubernetes.io/aws-load-balancer-ssl-cert"        = aws_acm_certificate.fl_ehds.arn
      "service.beta.kubernetes.io/aws-load-balancer-ssl-ports"       = "443"
    }
  }
}

# FL-EHDS ConfigMap
resource "kubernetes_config_map" "fl_ehds" {
  metadata {
    name      = "fl-ehds-config"
    namespace = kubernetes_namespace.fl_ehds.metadata[0].name
  }

  data = {
    "config.yaml" = yamlencode({
      aggregation = {
        algorithm = "fedavg"
        rounds    = 100
        clients_per_round = 10
      }
      privacy = {
        differential_privacy = true
        epsilon = 1.0
        delta = 1e-5
      }
      ehds = {
        compliant = true
        data_residency = var.data_residency
        audit_logging = true
      }
    })
  }
}

# FL-EHDS Secrets
resource "kubernetes_secret" "fl_ehds_config" {
  metadata {
    name      = "fl-ehds-secrets"
    namespace = kubernetes_namespace.fl_ehds.metadata[0].name
  }

  data = {
    "permit-id"   = base64encode("EHDS-PERMIT-${random_id.permit.hex}")
    "api-key"     = base64encode(random_password.api_key.result)
  }

  type = "Opaque"
}

resource "random_id" "permit" {
  byte_length = 8
}

resource "random_password" "api_key" {
  length  = 32
  special = false
}

# Horizontal Pod Autoscaler for FL Clients
resource "kubernetes_horizontal_pod_autoscaler_v2" "fl_clients" {
  metadata {
    name      = "fl-clients-hpa"
    namespace = kubernetes_namespace.fl_ehds.metadata[0].name
  }

  spec {
    scale_target_ref {
      api_version = "apps/v1"
      kind        = "Deployment"
      name        = "fl-clients"
    }

    min_replicas = 1
    max_replicas = var.fl_client_max_replicas

    metric {
      type = "Resource"
      resource {
        name = "cpu"
        target {
          type                = "Utilization"
          average_utilization = 70
        }
      }
    }

    metric {
      type = "Resource"
      resource {
        name = "memory"
        target {
          type                = "Utilization"
          average_utilization = 80
        }
      }
    }
  }
}
'''

    def _generate_monitoring(self) -> str:
        """Generate monitoring.tf."""
        return '''# FL-EHDS Monitoring Stack

# Prometheus
resource "helm_release" "prometheus" {
  name       = "prometheus"
  namespace  = kubernetes_namespace.fl_ehds.metadata[0].name
  repository = "https://prometheus-community.github.io/helm-charts"
  chart      = "kube-prometheus-stack"
  version    = "51.0.0"

  values = [yamlencode({
    prometheus = {
      prometheusSpec = {
        retention = "30d"
        storageSpec = {
          volumeClaimTemplate = {
            spec = {
              accessModes = ["ReadWriteOnce"]
              resources = {
                requests = {
                  storage = "100Gi"
                }
              }
            }
          }
        }
        additionalScrapeConfigs = [
          {
            job_name = "fl-ehds"
            kubernetes_sd_configs = [{
              role = "pod"
              namespaces = {
                names = ["fl-ehds"]
              }
            }]
            relabel_configs = [
              {
                source_labels = ["__meta_kubernetes_pod_annotation_prometheus_io_scrape"]
                action        = "keep"
                regex         = "true"
              }
            ]
          }
        ]
      }
    }
    grafana = {
      adminPassword = random_password.grafana_admin.result
      persistence = {
        enabled = true
        size    = "10Gi"
      }
      dashboardProviders = {
        "dashboardproviders.yaml" = {
          apiVersion = 1
          providers = [{
            name            = "fl-ehds"
            folder          = "FL-EHDS"
            type            = "file"
            disableDeletion = false
            options = {
              path = "/var/lib/grafana/dashboards/fl-ehds"
            }
          }]
        }
      }
    }
    alertmanager = {
      alertmanagerSpec = {
        storage = {
          volumeClaimTemplate = {
            spec = {
              accessModes = ["ReadWriteOnce"]
              resources = {
                requests = {
                  storage = "10Gi"
                }
              }
            }
          }
        }
      }
    }
  })]
}

resource "random_password" "grafana_admin" {
  length  = 16
  special = false
}

# Grafana Ingress
resource "kubernetes_ingress_v1" "grafana" {
  metadata {
    name      = "grafana"
    namespace = kubernetes_namespace.fl_ehds.metadata[0].name

    annotations = {
      "kubernetes.io/ingress.class"                = "nginx"
      "cert-manager.io/cluster-issuer"             = "letsencrypt-prod"
      "nginx.ingress.kubernetes.io/ssl-redirect"   = "true"
    }
  }

  spec {
    tls {
      hosts       = ["grafana.fl-ehds.example.com"]
      secret_name = "grafana-tls"
    }

    rule {
      host = "grafana.fl-ehds.example.com"

      http {
        path {
          path      = "/"
          path_type = "Prefix"

          backend {
            service {
              name = "prometheus-grafana"
              port {
                number = 80
              }
            }
          }
        }
      }
    }
  }
}

# Prometheus Service for internal access
resource "kubernetes_service" "prometheus" {
  metadata {
    name      = "prometheus-server"
    namespace = kubernetes_namespace.fl_ehds.metadata[0].name
  }

  spec {
    selector = {
      "app.kubernetes.io/name" = "prometheus"
    }

    port {
      port        = 9090
      target_port = 9090
    }

    type = "ClusterIP"
  }
}
'''

    def _generate_security(self) -> str:
        """Generate security.tf."""
        return '''# FL-EHDS Security Configuration

# KMS Key for EKS encryption
resource "aws_kms_key" "eks" {
  count = var.enable_encryption ? 1 : 0

  description             = "KMS key for FL-EHDS EKS encryption"
  deletion_window_in_days = 30
  enable_key_rotation     = true

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "Enable IAM User Permissions"
        Effect = "Allow"
        Principal = {
          AWS = "arn:aws:iam::${data.aws_caller_identity.current.account_id}:root"
        }
        Action   = "kms:*"
        Resource = "*"
      },
      {
        Sid    = "Allow EKS to use the key"
        Effect = "Allow"
        Principal = {
          Service = "eks.amazonaws.com"
        }
        Action = [
          "kms:Encrypt",
          "kms:Decrypt",
          "kms:ReEncrypt*",
          "kms:GenerateDataKey*",
          "kms:DescribeKey"
        ]
        Resource = "*"
      }
    ]
  })

  tags = local.common_tags
}

resource "aws_kms_alias" "eks" {
  count = var.enable_encryption ? 1 : 0

  name          = "alias/fl-ehds-eks"
  target_key_id = aws_kms_key.eks[0].key_id
}

data "aws_caller_identity" "current" {}

# Network Policies
resource "kubernetes_network_policy" "fl_ehds_default" {
  metadata {
    name      = "default-deny-ingress"
    namespace = kubernetes_namespace.fl_ehds.metadata[0].name
  }

  spec {
    pod_selector {}

    policy_types = ["Ingress"]

    ingress {
      from {
        namespace_selector {
          match_labels = {
            name = "fl-ehds"
          }
        }
      }
    }
  }
}

resource "kubernetes_network_policy" "fl_aggregator" {
  metadata {
    name      = "fl-aggregator-policy"
    namespace = kubernetes_namespace.fl_ehds.metadata[0].name
  }

  spec {
    pod_selector {
      match_labels = {
        app = "fl-aggregator"
      }
    }

    policy_types = ["Ingress", "Egress"]

    ingress {
      ports {
        port     = 50051
        protocol = "TCP"
      }
      ports {
        port     = 8080
        protocol = "TCP"
      }
    }

    egress {}
  }
}

# Pod Security Policy (PSP) replacement with Pod Security Standards
resource "kubernetes_manifest" "pod_security_standard" {
  manifest = {
    apiVersion = "v1"
    kind       = "Namespace"
    metadata = {
      name = "fl-ehds"
      labels = {
        "pod-security.kubernetes.io/enforce"         = "restricted"
        "pod-security.kubernetes.io/enforce-version" = "latest"
        "pod-security.kubernetes.io/audit"           = "restricted"
        "pod-security.kubernetes.io/warn"            = "restricted"
      }
    }
  }
}

# ACM Certificate
resource "aws_acm_certificate" "fl_ehds" {
  domain_name       = "fl-ehds.example.com"
  validation_method = "DNS"

  subject_alternative_names = [
    "*.fl-ehds.example.com"
  ]

  lifecycle {
    create_before_destroy = true
  }

  tags = local.common_tags
}
'''

    def _generate_audit(self) -> str:
        """Generate audit.tf for EHDS compliance."""
        return '''# FL-EHDS Audit Logging (EHDS Compliance)

# CloudWatch Log Group for Audit Logs
resource "aws_cloudwatch_log_group" "fl_ehds_audit" {
  name              = "/fl-ehds/audit"
  retention_in_days = 365  # EHDS requires long retention

  kms_key_id = var.enable_encryption ? aws_kms_key.eks[0].arn : null

  tags = merge(local.common_tags, {
    "ehds.eu/audit" = "true"
  })
}

# S3 Bucket for Long-term Audit Storage
resource "aws_s3_bucket" "audit_logs" {
  bucket = "${local.name_prefix}-audit-logs"

  tags = merge(local.common_tags, {
    "ehds.eu/audit"          = "true"
    "ehds.eu/data-residency" = var.data_residency
  })
}

resource "aws_s3_bucket_versioning" "audit_logs" {
  bucket = aws_s3_bucket.audit_logs.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "audit_logs" {
  bucket = aws_s3_bucket.audit_logs.id

  rule {
    apply_server_side_encryption_by_default {
      kms_master_key_id = var.enable_encryption ? aws_kms_key.eks[0].arn : null
      sse_algorithm     = var.enable_encryption ? "aws:kms" : "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "audit_logs" {
  bucket = aws_s3_bucket.audit_logs.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Object Lock for Immutable Audit Logs
resource "aws_s3_bucket_object_lock_configuration" "audit_logs" {
  bucket = aws_s3_bucket.audit_logs.id

  rule {
    default_retention {
      mode = "GOVERNANCE"
      days = 365
    }
  }
}

# CloudWatch Log Subscription to S3
resource "aws_cloudwatch_log_subscription_filter" "audit_to_s3" {
  name            = "fl-ehds-audit-to-s3"
  log_group_name  = aws_cloudwatch_log_group.fl_ehds_audit.name
  filter_pattern  = ""
  destination_arn = aws_kinesis_firehose_delivery_stream.audit_logs.arn
  role_arn        = aws_iam_role.cloudwatch_to_firehose.arn
}

# Kinesis Firehose for log delivery
resource "aws_kinesis_firehose_delivery_stream" "audit_logs" {
  name        = "${local.name_prefix}-audit-firehose"
  destination = "extended_s3"

  extended_s3_configuration {
    role_arn   = aws_iam_role.firehose.arn
    bucket_arn = aws_s3_bucket.audit_logs.arn
    prefix     = "audit-logs/year=!{timestamp:yyyy}/month=!{timestamp:MM}/day=!{timestamp:dd}/"

    buffering_size     = 64
    buffering_interval = 60

    compression_format = "GZIP"
  }
}

# IAM Roles for Audit Pipeline
resource "aws_iam_role" "cloudwatch_to_firehose" {
  name = "${local.name_prefix}-cw-to-firehose"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "logs.amazonaws.com"
      }
    }]
  })
}

resource "aws_iam_role" "firehose" {
  name = "${local.name_prefix}-firehose"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "firehose.amazonaws.com"
      }
    }]
  })
}
'''

    def _get_provider_source(self) -> str:
        """Get Terraform provider source."""
        sources = {
            "aws": "hashicorp/aws",
            "azure": "hashicorp/azurerm",
            "gcp": "hashicorp/google",
        }
        return sources.get(self.config.cloud_provider, "hashicorp/aws")

    def _get_provider_version(self) -> str:
        """Get Terraform provider version."""
        versions = {
            "aws": "5.0",
            "azure": "3.0",
            "gcp": "5.0",
        }
        return versions.get(self.config.cloud_provider, "5.0")

    def export_files(self, output_dir: str) -> Dict[str, str]:
        """Export all Terraform files to directory."""
        import os

        files = self.generate_all()

        for filename, content in files.items():
            filepath = os.path.join(output_dir, filename)
            os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else output_dir, exist_ok=True)
            with open(filepath, 'w') as f:
                f.write(content)

        logger.info(f"Exported {len(files)} Terraform files to {output_dir}")
        return files


class PulumiGenerator:
    """
    Generates Pulumi configurations for FL-EHDS infrastructure.

    Supports Python, TypeScript, and Go.
    """

    def __init__(self, config: IaCConfig, language: str = "python"):
        self.config = config
        self.language = language

    def generate_all(self) -> Dict[str, str]:
        """Generate all Pulumi files."""
        files = {}

        if self.language == "python":
            files["__main__.py"] = self._generate_python_main()
            files["Pulumi.yaml"] = self._generate_pulumi_yaml()
            files["Pulumi.prod.yaml"] = self._generate_pulumi_stack_config()
            files["requirements.txt"] = self._generate_requirements()
        elif self.language == "typescript":
            files["index.ts"] = self._generate_typescript_main()
            files["Pulumi.yaml"] = self._generate_pulumi_yaml()
            files["package.json"] = self._generate_package_json()

        return files

    def _generate_pulumi_yaml(self) -> str:
        """Generate Pulumi.yaml project file."""
        return f'''name: fl-ehds-infrastructure
runtime:
  name: {self.language}
  options:
    virtualenv: venv
description: FL-EHDS Infrastructure as Code
config:
  pulumi:tags:
    value:
      pulumi:template: fl-ehds
'''

    def _generate_pulumi_stack_config(self) -> str:
        """Generate Pulumi stack configuration."""
        return f'''config:
  aws:region: {self.config.region}
  fl-ehds:environment: {self.config.environment}
  fl-ehds:vpcCidr: "{self.config.vpc_cidr}"
  fl-ehds:kubernetesVersion: "{self.config.k8s_version}"
  fl-ehds:nodePoolSize: {self.config.node_pool_size}
  fl-ehds:nodePoolMin: {self.config.node_pool_min}
  fl-ehds:nodePoolMax: {self.config.node_pool_max}
  fl-ehds:enableEncryption: {str(self.config.enable_encryption).lower()}
  fl-ehds:dataResidency: "{self.config.data_residency}"
'''

    def _generate_requirements(self) -> str:
        """Generate requirements.txt."""
        return '''pulumi>=3.0.0
pulumi-aws>=6.0.0
pulumi-kubernetes>=4.0.0
pulumi-random>=4.0.0
'''

    def _generate_python_main(self) -> str:
        """Generate Python Pulumi program."""
        return f'''"""
FL-EHDS Infrastructure with Pulumi

This program deploys the complete FL-EHDS infrastructure:
- VPC with private subnets
- EKS Kubernetes cluster
- FL aggregator and client deployments
- Monitoring stack (Prometheus/Grafana)
- EHDS-compliant audit logging

Author: FL-EHDS Framework
"""

import pulumi
import pulumi_aws as aws
import pulumi_kubernetes as k8s
import pulumi_random as random
from pulumi import Config, Output, export

# Configuration
config = Config()
environment = config.get("environment") or "{self.config.environment}"
vpc_cidr = config.get("vpcCidr") or "{self.config.vpc_cidr}"
k8s_version = config.get("kubernetesVersion") or "{self.config.k8s_version}"
node_pool_size = config.get_int("nodePoolSize") or {self.config.node_pool_size}
enable_encryption = config.get_bool("enableEncryption") or {str(self.config.enable_encryption).lower() == "true"}
data_residency = config.get("dataResidency") or "{self.config.data_residency}"

# Common tags
common_tags = {{
    "Project": "fl-ehds",
    "Environment": environment,
    "ManagedBy": "pulumi",
    "EHDSCompliant": "true",
}}

# ============================================================================
# VPC
# ============================================================================

vpc = aws.ec2.Vpc(
    "fl-ehds-vpc",
    cidr_block=vpc_cidr,
    enable_dns_hostnames=True,
    enable_dns_support=True,
    tags={{**common_tags, "Name": f"fl-ehds-{{environment}}-vpc"}},
)

# Get availability zones
azs = aws.get_availability_zones(state="available")

# Create subnets
private_subnets = []
public_subnets = []

for i, az in enumerate(azs.names[:3]):
    # Private subnet
    private_subnet = aws.ec2.Subnet(
        f"private-subnet-{{i}}",
        vpc_id=vpc.id,
        cidr_block=f"10.0.{{i * 32}}.0/19",
        availability_zone=az,
        tags={{
            **common_tags,
            "Name": f"fl-ehds-{{environment}}-private-{{az}}",
            "kubernetes.io/role/internal-elb": "1",
        }},
    )
    private_subnets.append(private_subnet)

    # Public subnet
    public_subnet = aws.ec2.Subnet(
        f"public-subnet-{{i}}",
        vpc_id=vpc.id,
        cidr_block=f"10.0.{{i * 32 + 16}}.0/20",
        availability_zone=az,
        map_public_ip_on_launch=True,
        tags={{
            **common_tags,
            "Name": f"fl-ehds-{{environment}}-public-{{az}}",
            "kubernetes.io/role/elb": "1",
        }},
    )
    public_subnets.append(public_subnet)

# Internet Gateway
igw = aws.ec2.InternetGateway(
    "fl-ehds-igw",
    vpc_id=vpc.id,
    tags={{**common_tags, "Name": f"fl-ehds-{{environment}}-igw"}},
)

# NAT Gateway
eip = aws.ec2.Eip(
    "nat-eip",
    domain="vpc",
    tags={{**common_tags, "Name": f"fl-ehds-{{environment}}-nat-eip"}},
)

nat = aws.ec2.NatGateway(
    "fl-ehds-nat",
    subnet_id=public_subnets[0].id,
    allocation_id=eip.id,
    tags={{**common_tags, "Name": f"fl-ehds-{{environment}}-nat"}},
)

# Route tables
public_rt = aws.ec2.RouteTable(
    "public-rt",
    vpc_id=vpc.id,
    routes=[
        aws.ec2.RouteTableRouteArgs(
            cidr_block="0.0.0.0/0",
            gateway_id=igw.id,
        )
    ],
    tags={{**common_tags, "Name": f"fl-ehds-{{environment}}-public-rt"}},
)

private_rt = aws.ec2.RouteTable(
    "private-rt",
    vpc_id=vpc.id,
    routes=[
        aws.ec2.RouteTableRouteArgs(
            cidr_block="0.0.0.0/0",
            nat_gateway_id=nat.id,
        )
    ],
    tags={{**common_tags, "Name": f"fl-ehds-{{environment}}-private-rt"}},
)

# Associate route tables
for i, subnet in enumerate(public_subnets):
    aws.ec2.RouteTableAssociation(
        f"public-rta-{{i}}",
        subnet_id=subnet.id,
        route_table_id=public_rt.id,
    )

for i, subnet in enumerate(private_subnets):
    aws.ec2.RouteTableAssociation(
        f"private-rta-{{i}}",
        subnet_id=subnet.id,
        route_table_id=private_rt.id,
    )

# ============================================================================
# EKS Cluster
# ============================================================================

# IAM Role for EKS
eks_role = aws.iam.Role(
    "eks-cluster-role",
    assume_role_policy=pulumi.Output.json_dumps({{
        "Version": "2012-10-17",
        "Statement": [{{
            "Action": "sts:AssumeRole",
            "Effect": "Allow",
            "Principal": {{
                "Service": "eks.amazonaws.com"
            }}
        }}]
    }}),
    tags=common_tags,
)

aws.iam.RolePolicyAttachment(
    "eks-cluster-policy",
    role=eks_role.name,
    policy_arn="arn:aws:iam::aws:policy/AmazonEKSClusterPolicy",
)

# KMS Key for encryption
kms_key = None
if enable_encryption:
    kms_key = aws.kms.Key(
        "eks-kms-key",
        description="KMS key for FL-EHDS EKS encryption",
        deletion_window_in_days=30,
        enable_key_rotation=True,
        tags=common_tags,
    )

# EKS Cluster
cluster = aws.eks.Cluster(
    "fl-ehds-cluster",
    role_arn=eks_role.arn,
    version=k8s_version,
    vpc_config=aws.eks.ClusterVpcConfigArgs(
        subnet_ids=[s.id for s in private_subnets + public_subnets],
        endpoint_private_access=True,
        endpoint_public_access=True,
    ),
    encryption_config=aws.eks.ClusterEncryptionConfigArgs(
        provider=aws.eks.ClusterEncryptionConfigProviderArgs(
            key_arn=kms_key.arn if kms_key else None,
        ),
        resources=["secrets"],
    ) if enable_encryption else None,
    enabled_cluster_log_types=[
        "api",
        "audit",
        "authenticator",
        "controllerManager",
        "scheduler",
    ],
    tags={{**common_tags, "Name": f"fl-ehds-{{environment}}-cluster"}},
)

# Node Group IAM Role
node_role = aws.iam.Role(
    "eks-node-role",
    assume_role_policy=pulumi.Output.json_dumps({{
        "Version": "2012-10-17",
        "Statement": [{{
            "Action": "sts:AssumeRole",
            "Effect": "Allow",
            "Principal": {{
                "Service": "ec2.amazonaws.com"
            }}
        }}]
    }}),
    tags=common_tags,
)

for policy in [
    "arn:aws:iam::aws:policy/AmazonEKSWorkerNodePolicy",
    "arn:aws:iam::aws:policy/AmazonEKS_CNI_Policy",
    "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly",
]:
    aws.iam.RolePolicyAttachment(
        f"node-{{policy.split('/')[-1]}}",
        role=node_role.name,
        policy_arn=policy,
    )

# EKS Node Group
node_group = aws.eks.NodeGroup(
    "fl-ehds-workers",
    cluster_name=cluster.name,
    node_role_arn=node_role.arn,
    subnet_ids=[s.id for s in private_subnets],
    scaling_config=aws.eks.NodeGroupScalingConfigArgs(
        desired_size=node_pool_size,
        min_size={self.config.node_pool_min},
        max_size={self.config.node_pool_max},
    ),
    instance_types=["{self.config.node_machine_type}"],
    labels={{
        "fl-ehds/role": "worker",
    }},
    tags=common_tags,
)

# ============================================================================
# Kubernetes Resources
# ============================================================================

# Create Kubernetes provider
k8s_provider = k8s.Provider(
    "k8s-provider",
    kubeconfig=cluster.kubeconfig,
)

# FL-EHDS Namespace
namespace = k8s.core.v1.Namespace(
    "fl-ehds-namespace",
    metadata=k8s.meta.v1.ObjectMetaArgs(
        name="fl-ehds",
        labels={{
            "app.kubernetes.io/name": "fl-ehds",
            "ehds.eu/compliant": "true",
        }},
    ),
    opts=pulumi.ResourceOptions(provider=k8s_provider),
)

# FL Aggregator Deployment
fl_aggregator = k8s.apps.v1.Deployment(
    "fl-aggregator",
    metadata=k8s.meta.v1.ObjectMetaArgs(
        name="fl-aggregator",
        namespace=namespace.metadata.name,
    ),
    spec=k8s.apps.v1.DeploymentSpecArgs(
        replicas=2,
        selector=k8s.meta.v1.LabelSelectorArgs(
            match_labels={{"app": "fl-aggregator"}},
        ),
        template=k8s.core.v1.PodTemplateSpecArgs(
            metadata=k8s.meta.v1.ObjectMetaArgs(
                labels={{"app": "fl-aggregator"}},
                annotations={{
                    "prometheus.io/scrape": "true",
                    "prometheus.io/port": "8080",
                }},
            ),
            spec=k8s.core.v1.PodSpecArgs(
                containers=[
                    k8s.core.v1.ContainerArgs(
                        name="aggregator",
                        image="fl-ehds/aggregator:latest",
                        ports=[
                            k8s.core.v1.ContainerPortArgs(
                                container_port=50051,
                                name="grpc",
                            ),
                            k8s.core.v1.ContainerPortArgs(
                                container_port=8080,
                                name="metrics",
                            ),
                        ],
                        resources=k8s.core.v1.ResourceRequirementsArgs(
                            requests={{
                                "cpu": "2",
                                "memory": "4Gi",
                            }},
                            limits={{
                                "cpu": "4",
                                "memory": "8Gi",
                            }},
                        ),
                        liveness_probe=k8s.core.v1.ProbeArgs(
                            http_get=k8s.core.v1.HTTPGetActionArgs(
                                path="/health",
                                port=8080,
                            ),
                            initial_delay_seconds=30,
                            period_seconds=10,
                        ),
                    ),
                ],
            ),
        ),
    ),
    opts=pulumi.ResourceOptions(provider=k8s_provider),
)

# FL Aggregator Service
fl_service = k8s.core.v1.Service(
    "fl-aggregator-service",
    metadata=k8s.meta.v1.ObjectMetaArgs(
        name="fl-aggregator",
        namespace=namespace.metadata.name,
    ),
    spec=k8s.core.v1.ServiceSpecArgs(
        selector={{"app": "fl-aggregator"}},
        ports=[
            k8s.core.v1.ServicePortArgs(
                name="grpc",
                port=50051,
                target_port=50051,
            ),
            k8s.core.v1.ServicePortArgs(
                name="metrics",
                port=8080,
                target_port=8080,
            ),
        ],
        type="LoadBalancer",
    ),
    opts=pulumi.ResourceOptions(provider=k8s_provider),
)

# ============================================================================
# Exports
# ============================================================================

export("cluster_name", cluster.name)
export("cluster_endpoint", cluster.endpoint)
export("kubeconfig", cluster.kubeconfig)
export("vpc_id", vpc.id)
export("private_subnet_ids", [s.id for s in private_subnets])
export("fl_aggregator_endpoint", fl_service.status.load_balancer.ingress[0].hostname)
'''

    def _generate_typescript_main(self) -> str:
        """Generate TypeScript Pulumi program."""
        return f'''/**
 * FL-EHDS Infrastructure with Pulumi (TypeScript)
 *
 * Deploys FL-EHDS infrastructure on AWS EKS.
 */

import * as pulumi from "@pulumi/pulumi";
import * as aws from "@pulumi/aws";
import * as k8s from "@pulumi/kubernetes";

// Configuration
const config = new pulumi.Config();
const environment = config.get("environment") || "{self.config.environment}";
const vpcCidr = config.get("vpcCidr") || "{self.config.vpc_cidr}";
const k8sVersion = config.get("kubernetesVersion") || "{self.config.k8s_version}";
const nodePoolSize = config.getNumber("nodePoolSize") || {self.config.node_pool_size};
const enableEncryption = config.getBoolean("enableEncryption") || {str(self.config.enable_encryption).lower()};

// Common tags
const commonTags = {{
    Project: "fl-ehds",
    Environment: environment,
    ManagedBy: "pulumi",
    EHDSCompliant: "true",
}};

// VPC
const vpc = new aws.ec2.Vpc("fl-ehds-vpc", {{
    cidrBlock: vpcCidr,
    enableDnsHostnames: true,
    enableDnsSupport: true,
    tags: {{ ...commonTags, Name: `fl-ehds-${{environment}}-vpc` }},
}});

// Get AZs
const azs = aws.getAvailabilityZones({{ state: "available" }});

// Create subnets
const privateSubnets: aws.ec2.Subnet[] = [];
const publicSubnets: aws.ec2.Subnet[] = [];

azs.then(zones => {{
    zones.names.slice(0, 3).forEach((az, i) => {{
        const privateSubnet = new aws.ec2.Subnet(`private-subnet-${{i}}`, {{
            vpcId: vpc.id,
            cidrBlock: `10.0.${{i * 32}}.0/19`,
            availabilityZone: az,
            tags: {{
                ...commonTags,
                Name: `fl-ehds-${{environment}}-private-${{az}}`,
                "kubernetes.io/role/internal-elb": "1",
            }},
        }});
        privateSubnets.push(privateSubnet);

        const publicSubnet = new aws.ec2.Subnet(`public-subnet-${{i}}`, {{
            vpcId: vpc.id,
            cidrBlock: `10.0.${{i * 32 + 16}}.0/20`,
            availabilityZone: az,
            mapPublicIpOnLaunch: true,
            tags: {{
                ...commonTags,
                Name: `fl-ehds-${{environment}}-public-${{az}}`,
                "kubernetes.io/role/elb": "1",
            }},
        }});
        publicSubnets.push(publicSubnet);
    }});
}});

// EKS Cluster Role
const eksRole = new aws.iam.Role("eks-cluster-role", {{
    assumeRolePolicy: JSON.stringify({{
        Version: "2012-10-17",
        Statement: [{{
            Action: "sts:AssumeRole",
            Effect: "Allow",
            Principal: {{
                Service: "eks.amazonaws.com"
            }}
        }}]
    }}),
    tags: commonTags,
}});

new aws.iam.RolePolicyAttachment("eks-cluster-policy", {{
    role: eksRole.name,
    policyArn: "arn:aws:iam::aws:policy/AmazonEKSClusterPolicy",
}});

// EKS Cluster
const cluster = new aws.eks.Cluster("fl-ehds-cluster", {{
    roleArn: eksRole.arn,
    version: k8sVersion,
    vpcConfig: {{
        subnetIds: pulumi.output(privateSubnets.map(s => s.id).concat(publicSubnets.map(s => s.id))),
        endpointPrivateAccess: true,
        endpointPublicAccess: true,
    }},
    enabledClusterLogTypes: [
        "api",
        "audit",
        "authenticator",
        "controllerManager",
        "scheduler",
    ],
    tags: {{ ...commonTags, Name: `fl-ehds-${{environment}}-cluster` }},
}});

// Exports
export const clusterName = cluster.name;
export const clusterEndpoint = cluster.endpoint;
export const vpcId = vpc.id;
'''

    def _generate_package_json(self) -> str:
        """Generate package.json for TypeScript."""
        return '''{
    "name": "fl-ehds-infrastructure",
    "version": "1.0.0",
    "main": "index.ts",
    "devDependencies": {
        "@types/node": "^18"
    },
    "dependencies": {
        "@pulumi/aws": "^6.0.0",
        "@pulumi/kubernetes": "^4.0.0",
        "@pulumi/pulumi": "^3.0.0"
    }
}
'''

    def export_files(self, output_dir: str) -> Dict[str, str]:
        """Export all Pulumi files to directory."""
        import os

        files = self.generate_all()

        for filename, content in files.items():
            filepath = os.path.join(output_dir, filename)
            os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else output_dir, exist_ok=True)
            with open(filepath, 'w') as f:
                f.write(content)

        logger.info(f"Exported {len(files)} Pulumi files to {output_dir}")
        return files


class IaCManager:
    """
    Manages Infrastructure as Code generation and deployment.

    Supports Terraform and Pulumi for multi-cloud FL-EHDS deployments.
    """

    def __init__(self, config: Optional[IaCConfig] = None):
        self.config = config or IaCConfig()

    def generate_terraform(self) -> Dict[str, str]:
        """Generate Terraform configurations."""
        generator = TerraformGenerator(self.config)
        return generator.generate_all()

    def generate_pulumi(self, language: str = "python") -> Dict[str, str]:
        """Generate Pulumi configurations."""
        generator = PulumiGenerator(self.config, language)
        return generator.generate_all()

    def export_terraform(self, output_dir: str) -> Dict[str, str]:
        """Export Terraform files to directory."""
        generator = TerraformGenerator(self.config)
        return generator.export_files(output_dir)

    def export_pulumi(self, output_dir: str, language: str = "python") -> Dict[str, str]:
        """Export Pulumi files to directory."""
        generator = PulumiGenerator(self.config, language)
        return generator.export_files(output_dir)


def create_iac_config(**kwargs) -> IaCConfig:
    """Factory function to create IaC configuration."""
    provider_map = {
        "terraform": IaCProvider.TERRAFORM,
        "pulumi": IaCProvider.PULUMI,
        "cdk": IaCProvider.CDK,
    }

    if "provider" in kwargs and isinstance(kwargs["provider"], str):
        kwargs["provider"] = provider_map.get(kwargs["provider"].lower(), IaCProvider.TERRAFORM)

    return IaCConfig(**kwargs)


def create_iac_manager(config: Optional[IaCConfig] = None, **kwargs) -> IaCManager:
    """Factory function to create IaC manager."""
    if config is None:
        config = create_iac_config(**kwargs)
    return IaCManager(config)


# =============================================================================
# Factory Functions
# =============================================================================

def create_orchestration_config(
    orchestrator: str = "local",
    namespace: str = "fl-ehds",
    **kwargs
) -> OrchestrationConfig:
    """
    Create orchestration configuration.

    Args:
        orchestrator: Orchestrator type ("local", "kubernetes", "ray", "hybrid")
        namespace: Kubernetes namespace
        **kwargs: Additional configuration

    Returns:
        OrchestrationConfig instance
    """
    orchestrator_map = {
        "local": OrchestratorType.LOCAL,
        "kubernetes": OrchestratorType.KUBERNETES,
        "k8s": OrchestratorType.KUBERNETES,
        "ray": OrchestratorType.RAY,
        "hybrid": OrchestratorType.HYBRID,
    }

    return OrchestrationConfig(
        orchestrator=orchestrator_map.get(orchestrator.lower(), OrchestratorType.LOCAL),
        namespace=namespace,
        **kwargs
    )


def create_orchestration_manager(
    config: Optional[OrchestrationConfig] = None,
    **kwargs
) -> OrchestrationManager:
    """
    Create orchestration manager.

    Args:
        config: Orchestration configuration
        **kwargs: Config overrides

    Returns:
        OrchestrationManager instance
    """
    if config is None:
        config = create_orchestration_config(**kwargs)
    return OrchestrationManager(config)


# =============================================================================
# Example Usage
# =============================================================================

async def example_usage():
    """Example of orchestration infrastructure usage."""

    # Create orchestration manager (local mode for demo)
    config = create_orchestration_config(
        orchestrator="local",
        namespace="fl-ehds-demo",
    )
    manager = create_orchestration_manager(config)

    await manager.start()

    # Deploy FL cluster
    deployment = await manager.deploy_fl_cluster(
        num_clients=5,
        num_aggregators=1,
        permit_id="EHDS-PERMIT-001",
    )
    print(f"Deployed cluster: {deployment}")

    # List nodes
    nodes = await manager.list_nodes()
    print(f"Active nodes: {len(nodes)}")

    for node in nodes:
        print(f"  - {node.node_id}: {node.node_type.value} ({node.status.name})")

    # Scale clients
    scale_result = await manager.scale_nodes(
        NodeType.CLIENT,
        target_count=8,
    )
    print(f"Scaled clients: {scale_result}")

    # Get node status
    if nodes:
        status = await manager.get_node_status(nodes[0].node_id)
        print(f"Node status: {status}")

    # Cleanup
    for node in await manager.list_nodes():
        await manager.destroy_node(node.node_id)

    await manager.stop()
    print("Orchestration demo complete")


if __name__ == "__main__":
    asyncio.run(example_usage())

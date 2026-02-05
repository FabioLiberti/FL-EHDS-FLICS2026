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

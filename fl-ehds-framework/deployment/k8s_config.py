"""
Kubernetes Configuration Classes for FL-EHDS
=============================================
Pure configuration dataclasses for K8s deployment of FL workloads.
No kubernetes client dependency - only configuration and YAML generation.

Produces K8s-API-compatible manifests for:
- Deployments (aggregator + client pods)
- Services (ClusterIP/NodePort/LoadBalancer)
- HPA (Horizontal Pod Autoscaler)
- NetworkPolicy (EHDS data residency enforcement)
- PVC (model checkpoint storage)
- ConfigMap (FL training parameters)

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
# Resource Specifications
# =============================================================================

@dataclass
class K8sResourceSpec:
    """K8s resource requests/limits."""
    cpu: str = "1"
    memory: str = "2Gi"
    gpu: int = 0
    ephemeral_storage: str = "10Gi"

    def to_dict(self) -> Dict[str, str]:
        d = {"cpu": self.cpu, "memory": self.memory}
        if self.gpu > 0:
            d["nvidia.com/gpu"] = str(self.gpu)
        if self.ephemeral_storage:
            d["ephemeral-storage"] = self.ephemeral_storage
        return d


# =============================================================================
# Container Specification
# =============================================================================

@dataclass
class K8sContainerSpec:
    """K8s container specification."""
    name: str = "fl-ehds"
    image: str = "fl-ehds:latest"
    image_pull_policy: str = "IfNotPresent"
    requests: K8sResourceSpec = field(default_factory=K8sResourceSpec)
    limits: K8sResourceSpec = field(default_factory=lambda: K8sResourceSpec(cpu="2", memory="4Gi"))
    env: Dict[str, str] = field(default_factory=dict)
    ports: List[int] = field(default_factory=lambda: [8080])
    command: Optional[List[str]] = None
    args: Optional[List[str]] = None
    volume_mounts: List[Dict[str, str]] = field(default_factory=list)
    liveness_probe: Optional[Dict[str, Any]] = None
    readiness_probe: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        container = {
            "name": self.name,
            "image": self.image,
            "imagePullPolicy": self.image_pull_policy,
            "resources": {
                "requests": self.requests.to_dict(),
                "limits": self.limits.to_dict(),
            },
            "ports": [{"containerPort": p} for p in self.ports],
        }
        if self.env:
            container["env"] = [{"name": k, "value": v} for k, v in self.env.items()]
        if self.command:
            container["command"] = self.command
        if self.args:
            container["args"] = self.args
        if self.volume_mounts:
            container["volumeMounts"] = self.volume_mounts
        if self.liveness_probe:
            container["livenessProbe"] = self.liveness_probe
        if self.readiness_probe:
            container["readinessProbe"] = self.readiness_probe
        return container


# =============================================================================
# K8s Object Specifications
# =============================================================================

@dataclass
class K8sDeploymentSpec:
    """K8s Deployment manifest configuration."""
    name: str = "fl-aggregator"
    namespace: str = "fl-ehds"
    replicas: int = 1
    container: K8sContainerSpec = field(default_factory=K8sContainerSpec)
    labels: Dict[str, str] = field(default_factory=lambda: {"app": "fl-ehds"})
    annotations: Dict[str, str] = field(default_factory=dict)
    node_selector: Dict[str, str] = field(default_factory=dict)
    tolerations: List[Dict[str, Any]] = field(default_factory=list)
    service_account: str = "fl-ehds-sa"
    restart_policy: str = "Always"
    volumes: List[Dict[str, Any]] = field(default_factory=list)
    # EHDS compliance
    data_residency: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        all_labels = {**self.labels, "component": self.name}
        if self.data_residency:
            all_labels["ehds/data-residency"] = self.data_residency

        spec: Dict[str, Any] = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": self.name,
                "namespace": self.namespace,
                "labels": all_labels,
            },
            "spec": {
                "replicas": self.replicas,
                "selector": {"matchLabels": {"component": self.name}},
                "template": {
                    "metadata": {
                        "labels": {**all_labels, "component": self.name},
                    },
                    "spec": {
                        "serviceAccountName": self.service_account,
                        "restartPolicy": self.restart_policy,
                        "containers": [self.container.to_dict()],
                    },
                },
            },
        }
        if self.annotations:
            spec["metadata"]["annotations"] = self.annotations
        if self.node_selector:
            spec["spec"]["template"]["spec"]["nodeSelector"] = self.node_selector
        if self.tolerations:
            spec["spec"]["template"]["spec"]["tolerations"] = self.tolerations
        if self.volumes:
            spec["spec"]["template"]["spec"]["volumes"] = self.volumes
        return spec


@dataclass
class K8sServiceSpec:
    """K8s Service manifest configuration."""
    name: str = "fl-aggregator-svc"
    namespace: str = "fl-ehds"
    service_type: str = "ClusterIP"
    ports: List[Dict[str, int]] = field(default_factory=lambda: [{"port": 8080, "targetPort": 8080}])
    selector: Dict[str, str] = field(default_factory=lambda: {"component": "fl-aggregator"})

    def to_dict(self) -> Dict[str, Any]:
        return {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": self.name,
                "namespace": self.namespace,
            },
            "spec": {
                "type": self.service_type,
                "ports": [
                    {
                        "port": p["port"],
                        "targetPort": p.get("targetPort", p["port"]),
                        "protocol": p.get("protocol", "TCP"),
                    }
                    for p in self.ports
                ],
                "selector": self.selector,
            },
        }


@dataclass
class K8sHPASpec:
    """K8s Horizontal Pod Autoscaler configuration."""
    name: str = "fl-clients-hpa"
    namespace: str = "fl-ehds"
    target_deployment: str = "fl-client"
    min_replicas: int = 1
    max_replicas: int = 10
    target_cpu_percent: int = 70
    target_memory_percent: Optional[int] = None
    scale_down_stabilization: int = 300

    def to_dict(self) -> Dict[str, Any]:
        metrics = [
            {
                "type": "Resource",
                "resource": {
                    "name": "cpu",
                    "target": {"type": "Utilization", "averageUtilization": self.target_cpu_percent},
                },
            }
        ]
        if self.target_memory_percent:
            metrics.append({
                "type": "Resource",
                "resource": {
                    "name": "memory",
                    "target": {"type": "Utilization", "averageUtilization": self.target_memory_percent},
                },
            })

        return {
            "apiVersion": "autoscaling/v2",
            "kind": "HorizontalPodAutoscaler",
            "metadata": {
                "name": self.name,
                "namespace": self.namespace,
            },
            "spec": {
                "scaleTargetRef": {
                    "apiVersion": "apps/v1",
                    "kind": "Deployment",
                    "name": self.target_deployment,
                },
                "minReplicas": self.min_replicas,
                "maxReplicas": self.max_replicas,
                "metrics": metrics,
                "behavior": {
                    "scaleDown": {
                        "stabilizationWindowSeconds": self.scale_down_stabilization,
                    },
                },
            },
        }


@dataclass
class K8sNetworkPolicySpec:
    """K8s NetworkPolicy for EHDS data residency enforcement."""
    name: str = "fl-ehds-network-policy"
    namespace: str = "fl-ehds"
    pod_selector: Dict[str, str] = field(default_factory=lambda: {"app": "fl-ehds"})
    ingress_from: List[Dict[str, Any]] = field(default_factory=list)
    egress_to: List[Dict[str, Any]] = field(default_factory=list)
    allow_cross_namespace: bool = False

    def to_dict(self) -> Dict[str, Any]:
        policy: Dict[str, Any] = {
            "apiVersion": "networking.k8s.io/v1",
            "kind": "NetworkPolicy",
            "metadata": {
                "name": self.name,
                "namespace": self.namespace,
            },
            "spec": {
                "podSelector": {"matchLabels": self.pod_selector},
                "policyTypes": ["Ingress", "Egress"],
            },
        }
        ingress = self.ingress_from or [
            {"from": [{"namespaceSelector": {"matchLabels": {"name": self.namespace}}}]}
        ]
        egress = self.egress_to or [
            {"to": [{"namespaceSelector": {"matchLabels": {"name": self.namespace}}}]}
        ]
        if self.allow_cross_namespace:
            egress.append({"to": [{"namespaceSelector": {}}]})

        policy["spec"]["ingress"] = ingress
        policy["spec"]["egress"] = egress
        return policy


@dataclass
class K8sPVCSpec:
    """K8s PersistentVolumeClaim for model checkpoints."""
    name: str = "fl-model-storage"
    namespace: str = "fl-ehds"
    storage: str = "50Gi"
    access_modes: List[str] = field(default_factory=lambda: ["ReadWriteOnce"])
    storage_class: str = "standard"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "apiVersion": "v1",
            "kind": "PersistentVolumeClaim",
            "metadata": {
                "name": self.name,
                "namespace": self.namespace,
            },
            "spec": {
                "accessModes": self.access_modes,
                "storageClassName": self.storage_class,
                "resources": {
                    "requests": {"storage": self.storage},
                },
            },
        }


@dataclass
class K8sConfigMapSpec:
    """K8s ConfigMap for FL training parameters."""
    name: str = "fl-training-config"
    namespace: str = "fl-ehds"
    data: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {
                "name": self.name,
                "namespace": self.namespace,
            },
            "data": self.data,
        }


# =============================================================================
# Top-Level FL Cluster Configuration
# =============================================================================

@dataclass
class FLK8sClusterConfig:
    """Complete K8s cluster configuration for FL-EHDS deployment."""
    namespace: str = "fl-ehds"
    aggregator: K8sDeploymentSpec = field(default_factory=lambda: K8sDeploymentSpec(name="fl-aggregator"))
    clients: List[K8sDeploymentSpec] = field(default_factory=list)
    services: List[K8sServiceSpec] = field(default_factory=list)
    hpa: Optional[K8sHPASpec] = None
    network_policy: Optional[K8sNetworkPolicySpec] = None
    model_storage: Optional[K8sPVCSpec] = None
    config_map: Optional[K8sConfigMapSpec] = None
    # EHDS compliance
    enforce_data_residency: bool = True
    audit_logging: bool = True

    def to_dict(self) -> Dict[str, Any]:
        manifests = {
            "aggregator": self.aggregator.to_dict(),
            "clients": [c.to_dict() for c in self.clients],
            "services": [s.to_dict() for s in self.services],
        }
        if self.hpa:
            manifests["hpa"] = self.hpa.to_dict()
        if self.network_policy:
            manifests["network_policy"] = self.network_policy.to_dict()
        if self.model_storage:
            manifests["pvc"] = self.model_storage.to_dict()
        if self.config_map:
            manifests["configmap"] = self.config_map.to_dict()
        return manifests

    def to_yaml(self) -> str:
        if not HAS_YAML:
            raise ImportError("PyYAML required for YAML output: pip install pyyaml")
        d = self.to_dict()
        docs = [d["aggregator"]]
        docs.extend(d["clients"])
        docs.extend(d["services"])
        for key in ("hpa", "network_policy", "pvc", "configmap"):
            if key in d:
                docs.append(d[key])
        return "---\n".join(yaml.dump(doc, default_flow_style=False) for doc in docs)

    @classmethod
    def create_fl_cluster(
        cls,
        num_clients: int = 5,
        namespace: str = "fl-ehds",
        image: str = "fl-ehds:latest",
        aggregator_cpu: str = "2",
        aggregator_memory: str = "4Gi",
        client_cpu: str = "1",
        client_memory: str = "2Gi",
        client_gpu: int = 0,
        enable_hpa: bool = True,
        enable_network_policy: bool = True,
        model_storage_gb: str = "50Gi",
        fl_algorithm: str = "FedAvg",
        num_rounds: int = 30,
        data_residency: Optional[str] = "EU",
    ) -> "FLK8sClusterConfig":
        """Factory method to create a complete FL-EHDS K8s cluster config."""
        # Aggregator
        agg_container = K8sContainerSpec(
            name="aggregator",
            image=image,
            requests=K8sResourceSpec(cpu=aggregator_cpu, memory=aggregator_memory),
            limits=K8sResourceSpec(cpu=str(int(aggregator_cpu) * 2) if aggregator_cpu.isdigit() else aggregator_cpu, memory=aggregator_memory),
            env={
                "FL_ROLE": "aggregator",
                "FL_ALGORITHM": fl_algorithm,
                "FL_NUM_ROUNDS": str(num_rounds),
                "FL_NUM_CLIENTS": str(num_clients),
                "EHDS_AUDIT_LOG": "true",
            },
            ports=[8080, 8443],
            liveness_probe={
                "httpGet": {"path": "/health", "port": 8080},
                "initialDelaySeconds": 30,
                "periodSeconds": 10,
            },
            readiness_probe={
                "httpGet": {"path": "/ready", "port": 8080},
                "initialDelaySeconds": 10,
                "periodSeconds": 5,
            },
        )

        aggregator = K8sDeploymentSpec(
            name="fl-aggregator",
            namespace=namespace,
            replicas=1,
            container=agg_container,
            labels={"app": "fl-ehds", "role": "aggregator"},
            data_residency=data_residency,
        )

        # Client deployments
        clients = []
        for i in range(num_clients):
            client_container = K8sContainerSpec(
                name=f"client-{i}",
                image=image,
                requests=K8sResourceSpec(cpu=client_cpu, memory=client_memory, gpu=client_gpu),
                limits=K8sResourceSpec(cpu=client_cpu, memory=client_memory, gpu=client_gpu),
                env={
                    "FL_ROLE": "client",
                    "FL_CLIENT_ID": str(i),
                    "FL_AGGREGATOR_URL": f"fl-aggregator-svc.{namespace}.svc.cluster.local:8080",
                    "EHDS_AUDIT_LOG": "true",
                },
                ports=[8081],
            )
            client_dep = K8sDeploymentSpec(
                name=f"fl-client-{i}",
                namespace=namespace,
                replicas=1,
                container=client_container,
                labels={"app": "fl-ehds", "role": "client", "client-id": str(i)},
                data_residency=data_residency,
            )
            clients.append(client_dep)

        # Services
        services = [
            K8sServiceSpec(
                name="fl-aggregator-svc",
                namespace=namespace,
                service_type="ClusterIP",
                ports=[
                    {"port": 8080, "targetPort": 8080},
                    {"port": 8443, "targetPort": 8443},
                ],
                selector={"component": "fl-aggregator"},
            ),
        ]

        # Optional components
        hpa = None
        if enable_hpa:
            hpa = K8sHPASpec(
                name="fl-clients-hpa",
                namespace=namespace,
                target_deployment="fl-client-0",
                min_replicas=1,
                max_replicas=num_clients * 2,
            )

        network_policy = None
        if enable_network_policy:
            network_policy = K8sNetworkPolicySpec(
                name="fl-ehds-network-policy",
                namespace=namespace,
            )

        model_storage = K8sPVCSpec(
            name="fl-model-storage",
            namespace=namespace,
            storage=model_storage_gb,
        )

        config_map = K8sConfigMapSpec(
            name="fl-training-config",
            namespace=namespace,
            data={
                "algorithm": fl_algorithm,
                "num_rounds": str(num_rounds),
                "num_clients": str(num_clients),
                "data_residency": data_residency or "EU",
            },
        )

        return cls(
            namespace=namespace,
            aggregator=aggregator,
            clients=clients,
            services=services,
            hpa=hpa,
            network_policy=network_policy,
            model_storage=model_storage,
            config_map=config_map,
        )

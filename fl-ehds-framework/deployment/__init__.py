"""
FL-EHDS Deployment Configuration
=================================
Pure configuration classes for K8s and Ray deployment.
No client dependencies - only dataclasses and YAML generation.
"""

from deployment.k8s_config import (
    K8sResourceSpec,
    K8sContainerSpec,
    K8sDeploymentSpec,
    K8sServiceSpec,
    K8sHPASpec,
    K8sNetworkPolicySpec,
    K8sPVCSpec,
    K8sConfigMapSpec,
    FLK8sClusterConfig,
)
from deployment.ray_config import (
    RayHeadNodeConfig,
    RayWorkerGroupConfig,
    RayAutoscalerConfig,
    RayActorSpec,
    FLRayClusterConfig,
)

__all__ = [
    # K8s
    "K8sResourceSpec",
    "K8sContainerSpec",
    "K8sDeploymentSpec",
    "K8sServiceSpec",
    "K8sHPASpec",
    "K8sNetworkPolicySpec",
    "K8sPVCSpec",
    "K8sConfigMapSpec",
    "FLK8sClusterConfig",
    # Ray
    "RayHeadNodeConfig",
    "RayWorkerGroupConfig",
    "RayAutoscalerConfig",
    "RayActorSpec",
    "FLRayClusterConfig",
]

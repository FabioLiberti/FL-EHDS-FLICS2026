"""
FL-EHDS Core Module
===================
Core utilities, models, and base classes for the FL-EHDS framework.

Includes:
- FL Algorithms (FedAvg, FedProx, SCAFFOLD, FedAdam, etc.)
- Personalized FL (Ditto, Per-FedAvg, FedPer, APFL)
- Asynchronous FL (FedAsync, FedBuff, AsyncSGD, FedAT)
- Model Compression (SignSGD, QSGD, TopK, PowerSGD)
- Fairness-aware FL (q-FedAvg, AFL, FedMGDA, PropFair)
- Secure Aggregation (Pairwise masking, Secret sharing, HE)
- Vertical FL / Split Learning (SplitNN, PSI)
- Byzantine Resilience (Krum, Trimmed Mean, Bulyan, FLTrust, FLAME)
- Continual Learning (EWC, LwF, Experience Replay, Drift Detection)
- Multi-Task FL (Hard/Soft Sharing, FedMTL)
- Hierarchical FL (Client→Regional→National→EU aggregation)
- EHDS Interoperability:
  - HL7 FHIR R4 Integration
  - OMOP CDM Support
  - IHE Profiles (ATNA, BPPC, XDS, XCA, PIX/PDQ, XUA)
  - HDAB API Compliance
- Advanced FL Components:
  - Federated Unlearning (FedEraser, SISA, Gradient Ascent, Influence Functions)
  - Federated Transfer Learning (FedMD, Progressive Unfreeze, Domain Adaptation)
  - Incentive Mechanisms (Shapley Value, Marginal Contribution, Reputation)
  - Client Selection (Oort, Active Learning, Importance Sampling, Fairness-Aware)
  - Federated HPO (FedEx, FedBayes, Successive Halving)
- Infrastructure Components:
  - Model Watermarking (IP protection, provenance tracking)
  - gRPC/WebSocket Communication (low-latency streaming)
  - Protocol Buffers Serialization (bandwidth optimization)
  - Redis Caching (distributed checkpoints, fast recovery)
  - Kubernetes/Ray Orchestration (enterprise scalability)
  - Prometheus/Grafana Monitoring (production observability)
- Data models and configuration classes
- Utility functions
"""

# FL Algorithms
try:
    from .fl_algorithms import (
        ALGORITHM_INFO,
        FLAlgorithm,
        FedAvg,
        FedProx,
        SCAFFOLD,
        FedAdam,
        FedYogi,
        FedAdagrad,
        FedNova,
        FedDyn,
        Ditto,
        create_algorithm
    )
except ImportError:
    pass  # fl_algorithms may not be available

# Personalized FL
try:
    from .personalized_fl import (
        PersonalizedFLAlgorithm,
        PersonalizationConfig,
        Ditto as DittoPFL,
        PerFedAvg,
        FedPer,
        APFL,
        create_personalized_fl,
    )
except ImportError:
    pass

# Asynchronous FL
try:
    from .async_fl import (
        AsyncFLAlgorithm,
        AsyncConfig,
        FedAsync,
        FedBuff,
        AsyncSGD,
        FedAT,
        AsyncFLServer,
        AsyncFLSimulator,
        create_async_fl,
    )
except ImportError:
    pass

# Model Compression
try:
    from .model_compression import (
        GradientCompressor,
        CompressionConfig,
        CompressedGradient,
        SignSGDCompressor,
        QSGDCompressor,
        TernGradCompressor,
        TopKCompressor,
        RandomKCompressor,
        ThresholdCompressor,
        PowerSGDCompressor,
        CompressionManager,
        create_compressor,
    )
except ImportError:
    pass

# Fairness-aware FL
try:
    from .fairness_fl import (
        FairFLAlgorithm,
        FairnessConfig,
        FairnessMetrics,
        QFedAvg,
        AFL,
        FedMGDA,
        PropFair,
        FedMinMax,
        FairFLTrainer,
        create_fair_fl,
        compute_fairness_metrics,
    )
except ImportError:
    pass

# Vertical FL / Split Learning
try:
    from .vertical_fl import (
        VerticalPartition,
        PrivateSetIntersection,
        SplitNNParty,
        SplitNNCoordinator,
        SecureVerticalFL,
        VerticalFLSimulator,
        create_vertical_fl,
    )
except ImportError:
    pass

# Byzantine Resilience
try:
    from .byzantine_resilience import (
        ByzantineAggregator,
        KrumAggregator,
        TrimmedMeanAggregator,
        MedianAggregator,
        BulyanAggregator,
        FLTrustAggregator,
        FLAMEAggregator,
        ByzantineAttacker,
        ByzantineDefenseManager,
        BYZANTINE_AGGREGATORS,
        create_byzantine_aggregator,
    )
except ImportError:
    pass

# Continual Learning FL
try:
    from .continual_fl import (
        ConceptDriftDetector,
        TaskInfo,
        EWCContinualFL,
        ReplayBuffer,
        ReplayContinualFL,
        LwFContinualFL,
        ContinualFLCoordinator,
        create_continual_fl,
    )
except ImportError:
    pass

# Multi-Task FL
try:
    from .multitask_fl import (
        TaskDefinition,
        MultiTaskData,
        HardSharingMTL,
        SoftSharingMTL,
        FedMTL,
        MultiTaskFLCoordinator,
        create_multitask_fl,
    )
except ImportError:
    pass

# Hierarchical FL
try:
    from .hierarchical_fl import (
        FederationNode,
        FederationTopology,
        HierFedAvg,
        ClusteredHierFL,
        HierarchicalFLCoordinator,
        create_hierarchical_fl,
    )
except ImportError:
    pass

# Secure Aggregation
try:
    from .secure_aggregation import (
        PairwiseMaskingProtocol,
        ShamirSecretSharing,
        SecureAggregationManager,
    )
except ImportError:
    pass

# =============================================================================
# EHDS Interoperability
# =============================================================================

# HL7 FHIR Integration
try:
    from .fhir_integration import (
        FHIRResourceParser,
        FHIRFeatureExtractor,
        FHIRClient,
        FHIRDataset,
        FHIRPrivacyGuard,
        create_fhir_client,
        create_fhir_dataset,
    )
except ImportError:
    pass

# OMOP CDM Support
try:
    from .omop_cdm import (
        OMOPDomain,
        OMOPVocabulary,
        OMOPPerson,
        OMOPVisitOccurrence,
        OMOPConditionOccurrence,
        OMOPDrugExposure,
        OMOPMeasurement,
        OMOPProcedureOccurrence,
        OMOPDeath,
        OMOPVocabularyService,
        OMOPCohortBuilder,
        OMOPFeatureExtractor,
        OMOPFederatedQuery,
        OMOPDataset,
        create_omop_vocabulary_service,
        create_cohort_builder,
        create_omop_dataset,
    )
except ImportError:
    pass

# IHE Profiles
try:
    from .ihe_profiles import (
        IHEProfile,
        DocumentStatus,
        AuditEventType,
        AuditEventOutcome,
        ConsentStatus,
        ConsentScope,
        XDSDocumentEntry,
        PatientIdentifier,
        PatientDemographics,
        AuditEvent,
        ConsentDocument,
        XUAAssertion,
        ATNAAuditLogger,
        BPPCConsentManager,
        XUASecurityContext,
        PIXPDQManager,
        XDSDocumentRegistry,
        XCAGateway,
        IHEIntegrationManager,
        create_ihe_manager,
        create_audit_logger,
        create_consent_manager,
    )
except ImportError:
    pass

# HDAB API Compliance
try:
    from .hdab_api import (
        PermitStatus,
        PermitType,
        DataCategory,
        PurposeOfUse,
        ProcessingEnvironmentType,
        RequestorType,
        ComplianceStatus,
        Requestor,
        DatasetDescriptor,
        DataPermitApplication,
        DataPermit as HDABDataPermit,
        OptOutRecord,
        ComplianceReport,
        AccessLog,
        HDABAPIClient,
        HDABServiceSimulator,
        FLEHDSPermitManager,
        CrossBorderHDABCoordinator,
        create_hdab_simulator,
        create_permit_manager,
        create_cross_border_coordinator,
    )
except ImportError:
    pass

# =============================================================================
# Advanced FL Components
# =============================================================================

# Federated Unlearning
try:
    from .federated_unlearning import (
        UnlearningMethod,
        UnlearningScope,
        VerificationMethod,
        UnlearningRequest,
        UnlearningResult,
        ClientCheckpoint,
        FederatedUnlearner,
        ExactRetrainUnlearner,
        FedEraserUnlearner,
        SISAUnlearner,
        GradientAscentUnlearner,
        InfluenceFunctionUnlearner,
        UnlearningVerifier,
        FederatedUnlearningManager,
        create_unlearning_manager,
        create_sisa_unlearner,
    )
except ImportError:
    pass

# Federated Transfer Learning
try:
    from .federated_transfer import (
        TransferStrategy,
        DistillationMode,
        DomainType,
        PretrainedModel,
        TransferConfig,
        PublicDataPretrainer,
        FederatedTransferLearner,
        PretrainFinetuneLearner,
        ProgressiveUnfreezeLearner,
        FedMD,
        DomainAdaptationLearner,
        FederatedTransferManager,
        create_transfer_manager,
        create_pretrainer,
    )
except ImportError:
    pass

# Incentive Mechanisms
try:
    from .incentive_mechanisms import (
        ContributionMetric,
        RewardType,
        DataQualityDimension,
        ClientContribution,
        RewardAllocation,
        ClientReputation,
        ShapleyValueCalculator,
        MarginalContributionCalculator,
        GradientBasedCalculator,
        DataQualityAssessor,
        RewardDistributor,
        ReputationSystem,
        IncentiveManager,
        create_incentive_manager,
        create_shapley_calculator,
    )
except ImportError:
    pass

# Client Selection Strategies
try:
    from .client_selection import (
        SelectionStrategy,
        ClientStatus,
        ClientProfile,
        SelectionResult,
        ClientSelector,
        RandomSelector,
        ActiveLearningSelector,
        ImportanceSamplingSelector,
        ResourceAwareSelector,
        FairnessAwareSelector,
        OortSelector,
        ClusteredSelector,
        ClientSelectionManager,
        create_selection_manager,
        create_selector,
    )
except ImportError:
    pass

# Federated Hyperparameter Tuning
try:
    from .hyperparameter_tuning import (
        TuningStrategy,
        HyperparameterType,
        HyperparameterSpace,
        HyperparameterConfig,
        TuningResult,
        SearchSpaceBuilder,
        FederatedHPO,
        GridSearchHPO,
        RandomSearchHPO,
        FedExHPO,
        FedBayesHPO,
        SuccessiveHalvingHPO,
        FederatedHPOManager,
        create_hpo_manager,
        create_fl_search_space,
    )
except ImportError:
    pass

# =============================================================================
# Infrastructure Components
# =============================================================================

# Model Watermarking
try:
    from .model_watermarking import (
        WatermarkType,
        WatermarkConfig,
        WatermarkResult,
        VerificationResult,
        WatermarkGenerator,
        WatermarkEmbedder,
        SpreadSpectrumEmbedder,
        LSBEmbedder,
        BackdoorEmbedder,
        PassportEmbedder,
        WatermarkExtractor,
        FederatedWatermarkCoordinator,
        ModelWatermarkManager,
        create_watermark_config,
        create_watermark_manager,
        create_federated_watermark_coordinator,
    )
except ImportError:
    pass

# gRPC/WebSocket Communication
try:
    from .communication import (
        TransportType,
        CompressionType,
        MessageType,
        ConnectionState,
        RetryPolicy,
        CommunicationConfig,
        Message,
        ModelChunk,
        StreamMetrics,
        CompressionManager,
        ConnectionPool,
        RetryHandler,
        GRPCServer,
        GRPCClient,
        WebSocketServer,
        WebSocketClient,
        CommunicationManager,
        create_communication_config,
        create_communication_manager,
        create_grpc_server,
        create_grpc_client,
    )
except ImportError:
    pass

# Protocol Buffers Serialization
try:
    from .serialization import (
        SerializationFormat,
        DType,
        CompressionLevel,
        SerializationConfig,
        TensorDescriptor,
        SerializedModel,
        DeltaUpdate,
        ModelSerializer,
        ProtobufStyleSerializer,
        DeltaSerializer,
        SerializationManager,
        EHDSCompliantSerializer,
        create_serialization_config,
        create_serialization_manager,
        create_ehds_serializer,
        benchmark_serialization,
    )
except ImportError:
    pass

# Redis Caching
try:
    from .caching import (
        CacheBackend,
        EvictionPolicy,
        CacheRegion,
        CacheConfig,
        CacheEntry,
        CacheStats,
        CacheBackendInterface,
        MemoryCacheBackend,
        RedisCacheBackend,
        DistributedLock,
        ModelCheckpointCache,
        ClientStateCache,
        MetricsCache,
        CacheManager,
        cached,
        create_cache_config,
        create_cache_manager,
    )
except ImportError:
    pass

# Kubernetes/Ray Orchestration
try:
    from .orchestration import (
        OrchestratorType,
        NodeType,
        NodeStatus,
        ScalingPolicy,
        DeploymentStrategy,
        ResourceRequirements,
        NodeConfig,
        ScalingConfig,
        OrchestrationConfig,
        NodeState,
        KubernetesClient,
        RayOrchestrator,
        FLClientActor,
        FLAggregatorActor,
        AutoScaler,
        OrchestrationManager,
        create_orchestration_config,
        create_orchestration_manager,
    )
except ImportError:
    pass

# Prometheus/Grafana Monitoring
try:
    from .monitoring import (
        MetricType,
        AlertSeverity,
        AlertState,
        TraceStatus,
        MonitoringConfig,
        MetricValue,
        Metric,
        Counter,
        Gauge,
        Histogram,
        Summary,
        FLMetrics,
        AlertRule,
        Alert,
        AlertManager,
        Span,
        Tracer,
        GrafanaDashboardGenerator,
        MonitoringManager,
        timed,
        counted,
        create_monitoring_config,
        create_monitoring_manager,
    )
except ImportError:
    pass

from .models import (
    DataPermit,
    FLClient,
    FLRound,
    GradientUpdate,
    ModelCheckpoint,
    TrainingConfig,
    PrivacyConfig,
    ComplianceRecord,
)
from .utils import (
    load_config,
    setup_logging,
    compute_metrics,
    serialize_model,
    deserialize_model,
)
from .exceptions import (
    FLEHDSError,
    PermitError,
    OptOutError,
    PrivacyBudgetExceededError,
    ComplianceViolationError,
    CommunicationError,
)

__all__ = [
    # Models
    "DataPermit",
    "FLClient",
    "FLRound",
    "GradientUpdate",
    "ModelCheckpoint",
    "TrainingConfig",
    "PrivacyConfig",
    "ComplianceRecord",
    # Utils
    "load_config",
    "setup_logging",
    "compute_metrics",
    "serialize_model",
    "deserialize_model",
    # Exceptions
    "FLEHDSError",
    "PermitError",
    "OptOutError",
    "PrivacyBudgetExceededError",
    "ComplianceViolationError",
    "CommunicationError",
    # ==========================================================================
    # EHDS Interoperability
    # ==========================================================================
    # FHIR Integration
    "FHIRResourceParser",
    "FHIRFeatureExtractor",
    "FHIRClient",
    "FHIRDataset",
    "FHIRPrivacyGuard",
    "create_fhir_client",
    "create_fhir_dataset",
    # OMOP CDM
    "OMOPDomain",
    "OMOPVocabulary",
    "OMOPPerson",
    "OMOPVocabularyService",
    "OMOPCohortBuilder",
    "OMOPFeatureExtractor",
    "OMOPFederatedQuery",
    "OMOPDataset",
    "create_omop_vocabulary_service",
    "create_cohort_builder",
    "create_omop_dataset",
    # IHE Profiles
    "IHEProfile",
    "AuditEventType",
    "AuditEventOutcome",
    "ConsentStatus",
    "ConsentScope",
    "AuditEvent",
    "ConsentDocument",
    "ATNAAuditLogger",
    "BPPCConsentManager",
    "XUASecurityContext",
    "PIXPDQManager",
    "XDSDocumentRegistry",
    "XCAGateway",
    "IHEIntegrationManager",
    "create_ihe_manager",
    "create_audit_logger",
    "create_consent_manager",
    # HDAB API
    "PermitStatus",
    "PermitType",
    "DataCategory",
    "PurposeOfUse",
    "ProcessingEnvironmentType",
    "RequestorType",
    "ComplianceStatus",
    "Requestor",
    "DatasetDescriptor",
    "DataPermitApplication",
    "HDABDataPermit",
    "OptOutRecord",
    "ComplianceReport",
    "AccessLog",
    "HDABAPIClient",
    "HDABServiceSimulator",
    "FLEHDSPermitManager",
    "CrossBorderHDABCoordinator",
    "create_hdab_simulator",
    "create_permit_manager",
    "create_cross_border_coordinator",
    # ==========================================================================
    # Advanced FL Components
    # ==========================================================================
    # Federated Unlearning
    "UnlearningMethod",
    "UnlearningScope",
    "UnlearningRequest",
    "UnlearningResult",
    "FederatedUnlearner",
    "FedEraserUnlearner",
    "SISAUnlearner",
    "GradientAscentUnlearner",
    "UnlearningVerifier",
    "FederatedUnlearningManager",
    "create_unlearning_manager",
    # Federated Transfer Learning
    "TransferStrategy",
    "DomainType",
    "PretrainedModel",
    "TransferConfig",
    "PublicDataPretrainer",
    "FederatedTransferLearner",
    "PretrainFinetuneLearner",
    "FedMD",
    "FederatedTransferManager",
    "create_transfer_manager",
    "create_pretrainer",
    # Incentive Mechanisms
    "ContributionMetric",
    "RewardType",
    "ClientContribution",
    "RewardAllocation",
    "ClientReputation",
    "ShapleyValueCalculator",
    "DataQualityAssessor",
    "RewardDistributor",
    "ReputationSystem",
    "IncentiveManager",
    "create_incentive_manager",
    "create_shapley_calculator",
    # Client Selection
    "SelectionStrategy",
    "ClientStatus",
    "ClientProfile",
    "SelectionResult",
    "ClientSelector",
    "RandomSelector",
    "ActiveLearningSelector",
    "ImportanceSamplingSelector",
    "ResourceAwareSelector",
    "FairnessAwareSelector",
    "OortSelector",
    "ClusteredSelector",
    "ClientSelectionManager",
    "create_selection_manager",
    "create_selector",
    # Federated HPO
    "TuningStrategy",
    "HyperparameterType",
    "HyperparameterSpace",
    "HyperparameterConfig",
    "TuningResult",
    "SearchSpaceBuilder",
    "FederatedHPO",
    "GridSearchHPO",
    "RandomSearchHPO",
    "FedExHPO",
    "FedBayesHPO",
    "SuccessiveHalvingHPO",
    "FederatedHPOManager",
    "create_hpo_manager",
    "create_fl_search_space",
    # ==========================================================================
    # Infrastructure Components
    # ==========================================================================
    # Model Watermarking
    "WatermarkType",
    "WatermarkConfig",
    "WatermarkResult",
    "VerificationResult",
    "WatermarkGenerator",
    "WatermarkEmbedder",
    "SpreadSpectrumEmbedder",
    "LSBEmbedder",
    "BackdoorEmbedder",
    "PassportEmbedder",
    "WatermarkExtractor",
    "FederatedWatermarkCoordinator",
    "ModelWatermarkManager",
    "create_watermark_config",
    "create_watermark_manager",
    "create_federated_watermark_coordinator",
    # Communication (gRPC/WebSocket)
    "TransportType",
    "CompressionType",
    "MessageType",
    "ConnectionState",
    "RetryPolicy",
    "CommunicationConfig",
    "Message",
    "ModelChunk",
    "StreamMetrics",
    "CompressionManager",
    "ConnectionPool",
    "RetryHandler",
    "GRPCServer",
    "GRPCClient",
    "WebSocketServer",
    "WebSocketClient",
    "CommunicationManager",
    "create_communication_config",
    "create_communication_manager",
    "create_grpc_server",
    "create_grpc_client",
    # Serialization (Protocol Buffers)
    "SerializationFormat",
    "DType",
    "CompressionLevel",
    "SerializationConfig",
    "TensorDescriptor",
    "SerializedModel",
    "DeltaUpdate",
    "ModelSerializer",
    "ProtobufStyleSerializer",
    "DeltaSerializer",
    "SerializationManager",
    "EHDSCompliantSerializer",
    "create_serialization_config",
    "create_serialization_manager",
    "create_ehds_serializer",
    "benchmark_serialization",
    # Caching (Redis)
    "CacheBackend",
    "EvictionPolicy",
    "CacheRegion",
    "CacheConfig",
    "CacheEntry",
    "CacheStats",
    "CacheBackendInterface",
    "MemoryCacheBackend",
    "RedisCacheBackend",
    "DistributedLock",
    "ModelCheckpointCache",
    "ClientStateCache",
    "MetricsCache",
    "CacheManager",
    "cached",
    "create_cache_config",
    "create_cache_manager",
    # Orchestration (Kubernetes/Ray)
    "OrchestratorType",
    "NodeType",
    "NodeStatus",
    "ScalingPolicy",
    "DeploymentStrategy",
    "ResourceRequirements",
    "NodeConfig",
    "ScalingConfig",
    "OrchestrationConfig",
    "NodeState",
    "KubernetesClient",
    "RayOrchestrator",
    "FLClientActor",
    "FLAggregatorActor",
    "AutoScaler",
    "OrchestrationManager",
    "create_orchestration_config",
    "create_orchestration_manager",
    # Monitoring (Prometheus/Grafana)
    "MetricType",
    "AlertSeverity",
    "AlertState",
    "TraceStatus",
    "MonitoringConfig",
    "MetricValue",
    "Metric",
    "Counter",
    "Gauge",
    "Histogram",
    "Summary",
    "FLMetrics",
    "AlertRule",
    "Alert",
    "AlertManager",
    "Span",
    "Tracer",
    "GrafanaDashboardGenerator",
    "MonitoringManager",
    "timed",
    "counted",
    "create_monitoring_config",
    "create_monitoring_manager",
]

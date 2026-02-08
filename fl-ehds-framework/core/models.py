"""
FL-EHDS Data Models
===================
Pydantic models for data structures used throughout the FL-EHDS framework.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, field_validator
import hashlib


# =============================================================================
# Enums
# =============================================================================


class PermitPurpose(str, Enum):
    """Permitted purposes under EHDS Article 53."""

    SCIENTIFIC_RESEARCH = "scientific_research"
    PUBLIC_HEALTH_SURVEILLANCE = "public_health_surveillance"
    HEALTH_POLICY = "health_policy"
    EDUCATION_TRAINING = "education_training"
    AI_SYSTEM_DEVELOPMENT = "ai_system_development"
    PERSONALIZED_MEDICINE = "personalized_medicine"
    OFFICIAL_STATISTICS = "official_statistics"
    PATIENT_SAFETY = "patient_safety"


class DataCategory(str, Enum):
    """Categories of health data under EHDS."""

    EHR = "ehr"  # Electronic Health Records
    LAB_RESULTS = "lab_results"
    IMAGING = "imaging"
    GENOMIC = "genomic"
    REGISTRY = "registry"  # Disease registries
    CLAIMS = "claims"  # Insurance claims
    QUESTIONNAIRE = "questionnaire"  # Patient-reported outcomes


class PermitStatus(str, Enum):
    """Status of a data permit."""

    PENDING = "pending"
    ACTIVE = "active"
    EXPIRED = "expired"
    REVOKED = "revoked"
    SUSPENDED = "suspended"


class ClientStatus(str, Enum):
    """Status of an FL client."""

    AVAILABLE = "available"
    TRAINING = "training"
    AGGREGATING = "aggregating"
    OFFLINE = "offline"
    ERROR = "error"


class RoundStatus(str, Enum):
    """Status of an FL training round."""

    INITIALIZING = "initializing"
    CLIENT_SELECTION = "client_selection"
    TRAINING = "training"
    AGGREGATION = "aggregation"
    COMPLETED = "completed"
    FAILED = "failed"


class AggregationAlgorithm(str, Enum):
    """Supported aggregation algorithms."""

    FEDAVG = "fedavg"
    FEDPROX = "fedprox"
    SCAFFOLD = "scaffold"
    FEDADAM = "fedadam"


# =============================================================================
# Layer 1: Governance Models
# =============================================================================


class DataPermit(BaseModel):
    """
    Data permit issued by an HDAB for secondary use of health data.
    Implements EHDS requirements for data access authorization.
    """

    permit_id: str = Field(..., description="Unique permit identifier")
    hdab_id: str = Field(..., description="Issuing HDAB identifier")
    requester_id: str = Field(..., description="Data requester identifier")
    purpose: PermitPurpose = Field(..., description="Permitted purpose (Article 53)")
    data_categories: List[DataCategory] = Field(
        ..., description="Authorized data categories"
    )
    data_sources: List[str] = Field(
        default_factory=list, description="Authorized data source identifiers"
    )
    member_states: List[str] = Field(
        default_factory=list, description="Member states covered by permit"
    )
    issued_at: datetime = Field(default_factory=datetime.utcnow)
    valid_from: datetime = Field(default_factory=datetime.utcnow)
    valid_until: datetime = Field(..., description="Permit expiration date")
    status: PermitStatus = Field(default=PermitStatus.ACTIVE)
    conditions: Dict[str, Any] = Field(
        default_factory=dict, description="Additional permit conditions"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)
    privacy_budget_total: Optional[float] = Field(
        default=None, description="Total epsilon budget for training"
    )
    privacy_budget_used: float = Field(
        default=0.0, description="Epsilon consumed so far"
    )
    max_rounds: Optional[int] = Field(
        default=None, description="Maximum training rounds authorized"
    )

    @field_validator("valid_until")
    @classmethod
    def validate_expiry(cls, v: datetime, info) -> datetime:
        """Ensure valid_until is in the future."""
        if v <= datetime.utcnow():
            raise ValueError("Permit expiration date must be in the future")
        return v

    def is_valid(self) -> bool:
        """Check if permit is currently valid."""
        now = datetime.utcnow()
        return (
            self.status == PermitStatus.ACTIVE
            and self.valid_from <= now <= self.valid_until
        )

    def covers_purpose(self, purpose: PermitPurpose) -> bool:
        """Check if permit covers the specified purpose."""
        return self.purpose == purpose

    def covers_categories(self, categories: List[DataCategory]) -> bool:
        """Check if permit covers all specified data categories."""
        return all(cat in self.data_categories for cat in categories)

    def check_privacy_budget(self, epsilon_cost: float) -> bool:
        """Check if privacy budget can cover the epsilon cost."""
        if self.privacy_budget_total is None:
            return True
        return (self.privacy_budget_used + epsilon_cost) <= self.privacy_budget_total

    def consume_privacy_budget(self, epsilon_cost: float) -> bool:
        """Consume epsilon from privacy budget. Returns False if insufficient."""
        if not self.check_privacy_budget(epsilon_cost):
            return False
        self.privacy_budget_used += epsilon_cost
        return True


class OptOutRecord(BaseModel):
    """Record of a citizen's opt-out decision under Article 71."""

    record_id: str = Field(..., description="Unique record identifier")
    patient_id: str = Field(..., description="Patient identifier (pseudonymized)")
    opt_out_date: datetime = Field(default_factory=datetime.utcnow)
    scope: str = Field(
        default="all", description="Opt-out scope: all, category, purpose"
    )
    categories: Optional[List[DataCategory]] = Field(
        default=None, description="Specific categories opted out (if scope=category)"
    )
    purposes: Optional[List[PermitPurpose]] = Field(
        default=None, description="Specific purposes opted out (if scope=purpose)"
    )
    member_state: str = Field(..., description="Member state of registration")
    is_active: bool = Field(default=True)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ComplianceRecord(BaseModel):
    """Audit trail record for GDPR Article 30 compliance."""

    record_id: str = Field(..., description="Unique record identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    action: str = Field(..., description="Action performed")
    actor: str = Field(..., description="Entity performing the action")
    permit_id: Optional[str] = Field(default=None)
    data_categories: List[DataCategory] = Field(default_factory=list)
    purpose: Optional[PermitPurpose] = Field(default=None)
    legal_basis: str = Field(default="EHDS Article 53")
    outcome: str = Field(..., description="Action outcome: success, failure, partial")
    details: Dict[str, Any] = Field(default_factory=dict)
    client_ids: List[str] = Field(
        default_factory=list, description="Participating clients"
    )
    round_number: Optional[int] = Field(default=None)

    def to_log_entry(self) -> dict:
        """Convert to structured log entry."""
        return {
            "record_id": self.record_id,
            "timestamp": self.timestamp.isoformat(),
            "action": self.action,
            "actor": self.actor,
            "permit_id": self.permit_id,
            "data_categories": [cat.value for cat in self.data_categories],
            "purpose": self.purpose.value if self.purpose else None,
            "legal_basis": self.legal_basis,
            "outcome": self.outcome,
            "details": self.details,
        }


# =============================================================================
# Layer 2: Orchestration Models
# =============================================================================


class GradientUpdate(BaseModel):
    """Gradient update from a single client."""

    client_id: str = Field(..., description="Client identifier")
    round_number: int = Field(..., ge=0)
    gradients: Dict[str, Any] = Field(
        ..., description="Layer-wise gradient tensors (serialized)"
    )
    num_samples: int = Field(..., ge=1, description="Number of training samples")
    local_loss: float = Field(..., description="Local training loss")
    local_metrics: Dict[str, float] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    is_clipped: bool = Field(default=False, description="Whether gradients were clipped")
    noise_added: bool = Field(
        default=False, description="Whether DP noise was added"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def compute_hash(self) -> str:
        """Compute hash of gradient update for integrity verification."""
        content = f"{self.client_id}:{self.round_number}:{self.num_samples}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]


class FLClient(BaseModel):
    """Federated Learning client (data holder) information."""

    client_id: str = Field(..., description="Unique client identifier")
    organization: str = Field(..., description="Organization name")
    member_state: str = Field(..., description="EU Member State")
    data_categories: List[DataCategory] = Field(
        ..., description="Available data categories"
    )
    num_samples: int = Field(..., ge=0, description="Number of available samples")
    status: ClientStatus = Field(default=ClientStatus.AVAILABLE)
    hardware_profile: Dict[str, Any] = Field(
        default_factory=dict, description="Hardware capabilities"
    )
    last_heartbeat: datetime = Field(default_factory=datetime.utcnow)
    rounds_participated: int = Field(default=0)
    total_samples_contributed: int = Field(default=0)
    average_training_time: Optional[float] = Field(
        default=None, description="Average training time per round (seconds)"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def is_available(self) -> bool:
        """Check if client is available for training."""
        return self.status == ClientStatus.AVAILABLE

    def update_participation(self, samples: int, training_time: float) -> None:
        """Update client statistics after round participation."""
        self.rounds_participated += 1
        self.total_samples_contributed += samples
        if self.average_training_time is None:
            self.average_training_time = training_time
        else:
            # Exponential moving average
            self.average_training_time = (
                0.9 * self.average_training_time + 0.1 * training_time
            )


class FLRound(BaseModel):
    """Information about a single FL training round."""

    round_number: int = Field(..., ge=0)
    status: RoundStatus = Field(default=RoundStatus.INITIALIZING)
    started_at: Optional[datetime] = Field(default=None)
    completed_at: Optional[datetime] = Field(default=None)
    selected_clients: List[str] = Field(default_factory=list)
    participating_clients: List[str] = Field(default_factory=list)
    total_samples: int = Field(default=0)
    aggregated_loss: Optional[float] = Field(default=None)
    metrics: Dict[str, float] = Field(default_factory=dict)
    privacy_spent: float = Field(
        default=0.0, description="Privacy budget spent this round"
    )
    cumulative_privacy: float = Field(
        default=0.0, description="Cumulative privacy budget spent"
    )
    errors: List[str] = Field(default_factory=list)

    def duration_seconds(self) -> Optional[float]:
        """Calculate round duration in seconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None


class ModelCheckpoint(BaseModel):
    """Model checkpoint saved during training."""

    checkpoint_id: str = Field(..., description="Unique checkpoint identifier")
    round_number: int = Field(..., ge=0)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    model_state: Dict[str, Any] = Field(
        ..., description="Serialized model state dict"
    )
    optimizer_state: Optional[Dict[str, Any]] = Field(default=None)
    metrics: Dict[str, float] = Field(default_factory=dict)
    privacy_budget_used: float = Field(default=0.0)
    is_best: bool = Field(default=False)
    metadata: Dict[str, Any] = Field(default_factory=dict)


# =============================================================================
# Configuration Models
# =============================================================================


class PrivacyConfig(BaseModel):
    """Configuration for privacy protection mechanisms."""

    # Differential Privacy
    dp_enabled: bool = Field(default=True)
    epsilon: float = Field(default=1.0, gt=0, description="Privacy budget")
    delta: float = Field(default=1e-5, gt=0, lt=1)
    noise_mechanism: str = Field(default="gaussian")
    accountant: str = Field(default="rdp")

    # Gradient Clipping
    clipping_enabled: bool = Field(default=True)
    max_grad_norm: float = Field(default=1.0, gt=0)
    norm_type: str = Field(default="l2")

    # Secure Aggregation
    secure_agg_enabled: bool = Field(default=True)
    secure_agg_protocol: str = Field(default="shamir")
    secure_agg_threshold: float = Field(default=0.67)


class TrainingConfig(BaseModel):
    """Configuration for local training at data holders."""

    batch_size: int = Field(default=32, ge=1)
    local_epochs: int = Field(default=5, ge=1)
    learning_rate: float = Field(default=0.01, gt=0)
    optimizer: str = Field(default="sgd")
    momentum: float = Field(default=0.9, ge=0)
    weight_decay: float = Field(default=1e-4, ge=0)

    # Adaptive training
    adaptive_batching: bool = Field(default=True)
    min_batch_size: int = Field(default=8, ge=1)
    max_batch_size: int = Field(default=128, ge=1)
    gradient_accumulation_steps: int = Field(default=1, ge=1)


class AggregationConfig(BaseModel):
    """Configuration for gradient aggregation."""

    algorithm: AggregationAlgorithm = Field(default=AggregationAlgorithm.FEDAVG)
    num_rounds: int = Field(default=100, ge=1)
    min_clients: int = Field(default=3, ge=1)
    max_clients: Optional[int] = Field(default=None)
    client_selection: str = Field(default="random")

    # FedProx specific
    fedprox_mu: float = Field(default=0.01, ge=0)

    # Early stopping
    early_stopping_enabled: bool = Field(default=True)
    early_stopping_patience: int = Field(default=10, ge=1)
    early_stopping_min_delta: float = Field(default=0.001, ge=0)

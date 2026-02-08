"""
Secure Processing Environment (EHDS Article 50)
================================================
Simulates a secure processing environment for cross-border FL training.

Three components:
1. EnclaveSimulator: TEE simulation with I/O validation per client
2. WatermarkingBridge: Model fingerprinting via existing ModelWatermarkManager
3. TimeGuard: Time-limited data access enforcement

SecureProcessingBridge orchestrates all three with the standard bridge lifecycle:
    start_session -> [pre_round + post_round] x N -> end_session

Author: Fabio Liberti
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# =========================================================================
# 1. ENCLAVE SIMULATOR
# =========================================================================


class EnclaveSimulator:
    """Simulates a Trusted Execution Environment (TEE) per FL client.

    Enforces EHDS Art. 50 secure processing:
    - Input validation: only training data and config enter the enclave
    - Output validation: only model updates (deltas/gradients/metrics) exit
    - No raw data export: blocks patient_records, features, raw_data
    - Memory isolation: simulated per-client boundaries
    """

    # Output types that are NEVER allowed to leave the enclave
    BLOCKED_OUTPUTS = frozenset([
        "raw_data", "patient_records", "features", "labels",
        "training_samples", "personal_data",
    ])

    def __init__(
        self,
        num_clients: int,
        allowed_output_types: Optional[List[str]] = None,
    ):
        self._num_clients = num_clients
        self._allowed_outputs = set(
            allowed_output_types or ["model_delta", "gradient", "metrics"]
        )
        self._enclave_log: List[Dict[str, Any]] = []
        self._active_enclaves: Dict[int, Dict[str, Any]] = {}
        self._violations: List[Dict[str, Any]] = []
        self._total_inputs = 0
        self._total_outputs = 0

    def initialize_enclaves(self) -> Dict[str, Any]:
        """Create simulated enclaves for all clients."""
        for cid in range(self._num_clients):
            self._active_enclaves[cid] = {
                "client_id": cid,
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "status": "active",
                "memory_isolated": True,
                "input_count": 0,
                "output_count": 0,
            }
        logger.info(
            f"Initialized {self._num_clients} secure enclaves "
            f"(allowed outputs: {sorted(self._allowed_outputs)})"
        )
        return {
            "active_enclaves": self._num_clients,
            "allowed_outputs": sorted(self._allowed_outputs),
            "blocked_outputs": sorted(self.BLOCKED_OUTPUTS),
        }

    def validate_input(
        self, client_id: int, data_type: str, data_size: int
    ) -> bool:
        """Validate input entering the enclave.

        Allowed: training_data, config, model_weights
        """
        allowed_inputs = {"training_data", "config", "model_weights", "labels"}
        allowed = data_type in allowed_inputs
        self._total_inputs += 1

        if client_id in self._active_enclaves:
            self._active_enclaves[client_id]["input_count"] += 1

        if not allowed:
            self._violations.append({
                "type": "invalid_input",
                "client_id": client_id,
                "data_type": data_type,
                "data_size": data_size,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            })

        return allowed

    def validate_output(
        self, client_id: int, output_type: str, output_size_bytes: int
    ) -> Tuple[bool, str]:
        """Validate output leaving the enclave.

        Returns:
            (allowed, reason)
        """
        # Always block known sensitive types
        if output_type in self.BLOCKED_OUTPUTS:
            reason = (
                f"BLOCKED: '{output_type}' cannot leave secure enclave "
                f"(Art. 50 data isolation)"
            )
            self._violations.append({
                "type": "blocked_output",
                "client_id": client_id,
                "output_type": output_type,
                "output_size_bytes": output_size_bytes,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "reason": reason,
            })
            logger.warning(
                f"Enclave violation: client {client_id} attempted to export "
                f"'{output_type}' ({output_size_bytes} bytes)"
            )
            return False, reason

        # Check if in allowed list
        if output_type not in self._allowed_outputs:
            reason = (
                f"Output type '{output_type}' not in allowed list: "
                f"{sorted(self._allowed_outputs)}"
            )
            self._violations.append({
                "type": "unauthorized_output",
                "client_id": client_id,
                "output_type": output_type,
                "output_size_bytes": output_size_bytes,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "reason": reason,
            })
            return False, reason

        self._total_outputs += 1
        if client_id in self._active_enclaves:
            self._active_enclaves[client_id]["output_count"] += 1

        return True, "OK"

    def log_round(
        self,
        round_num: int,
        client_id: int,
        input_samples: int,
        output_type: str,
        output_size_bytes: int,
    ):
        """Log enclave I/O for audit trail."""
        self._enclave_log.append({
            "round": round_num,
            "client_id": client_id,
            "input_samples": input_samples,
            "output_type": output_type,
            "output_size_bytes": output_size_bytes,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        })

    def get_violations(self) -> List[Dict[str, Any]]:
        """Return all enclave security violations."""
        return list(self._violations)

    def export_report(self) -> Dict[str, Any]:
        """Export enclave simulation report."""
        return {
            "active_enclaves": len(self._active_enclaves),
            "total_inputs": self._total_inputs,
            "total_outputs": self._total_outputs,
            "total_violations": len(self._violations),
            "violations": self._violations,
            "allowed_outputs": sorted(self._allowed_outputs),
            "per_client": {
                str(cid): info
                for cid, info in self._active_enclaves.items()
            },
            "log_entries": len(self._enclave_log),
        }


# =========================================================================
# 2. WATERMARKING BRIDGE
# =========================================================================


class WatermarkingBridge:
    """Wraps existing ModelWatermarkManager for FL governance (Art. 37/50).

    Embeds a spread-spectrum watermark into the global model after each
    aggregation round. Verifies watermark at session end for traceability.

    Uses core/model_watermarking.py infrastructure:
    - ModelWatermarkManager.initialize()
    - ModelWatermarkManager.create_watermark()
    - ModelWatermarkManager.embed_watermark()
    - ModelWatermarkManager.verify_watermark()
    """

    def __init__(
        self,
        organization_id: str = "fl-ehds-framework",
        organization_name: str = "FL-EHDS Research Consortium",
        watermark_strength: float = 0.01,
        watermark_length: int = 256,
    ):
        self._org_id = organization_id
        self._org_name = organization_name
        self._strength = watermark_strength
        self._length = watermark_length

        # Lazy-loaded from core.model_watermarking
        self._manager = None
        self._config = None
        self._watermark = None
        self._signature = None
        self._signature_id: Optional[str] = None
        self._round_verifications: List[Dict[str, Any]] = []
        self._embed_count = 0

    def initialize(self) -> str:
        """Initialize watermark manager, create watermark.

        Returns:
            signature_id for the created watermark
        """
        from core.model_watermarking import (
            EmbeddingMethod,
            ModelWatermarkManager,
            WatermarkConfig,
            WatermarkType,
        )

        self._manager = ModelWatermarkManager(self._org_id, self._org_name)
        self._config = WatermarkConfig(
            watermark_type=WatermarkType.WEIGHT_BASED,
            embedding_method=EmbeddingMethod.SPREAD_SPECTRUM,
            watermark_strength=self._strength,
            watermark_length=self._length,
        )

        self._manager.initialize()
        self._watermark, self._signature = self._manager.create_watermark(
            owner_id="fl-ehds-global-model",
            config=self._config,
        )
        self._signature_id = self._signature.signature_id

        logger.info(
            f"Watermark initialized: sig={self._signature_id[:12]}..., "
            f"strength={self._strength}, length={self._length}"
        )
        return self._signature_id

    def embed_watermark(self, model_state_dict: Dict) -> Dict:
        """Embed watermark into PyTorch model state dict.

        Converts torch tensors -> numpy, embeds watermark, converts back.

        Args:
            model_state_dict: PyTorch model state_dict()

        Returns:
            Watermarked state_dict (same format as input)
        """
        import numpy as np
        import torch

        # Convert torch tensors to numpy
        numpy_weights = {}
        for k, v in model_state_dict.items():
            if isinstance(v, torch.Tensor):
                numpy_weights[k] = v.cpu().numpy()
            else:
                numpy_weights[k] = np.array(v)

        # Embed watermark
        watermarked = self._manager.embed_watermark(
            numpy_weights,
            self._watermark,
            self._config,
            self._signature_id,
        )

        # Convert back to torch tensors
        result = {}
        for k, v in watermarked.items():
            if k in model_state_dict and isinstance(
                model_state_dict[k], torch.Tensor
            ):
                result[k] = torch.from_numpy(v).to(model_state_dict[k].device)
            else:
                result[k] = torch.from_numpy(v)

        self._embed_count += 1
        return result

    def verify_watermark(
        self, model_state_dict: Dict, round_num: int
    ) -> Dict[str, Any]:
        """Verify watermark presence in model.

        Args:
            model_state_dict: PyTorch model state_dict()
            round_num: Current round number

        Returns:
            Verification result dict
        """
        import numpy as np
        import torch

        numpy_weights = {}
        for k, v in model_state_dict.items():
            if isinstance(v, torch.Tensor):
                numpy_weights[k] = v.cpu().numpy()
            else:
                numpy_weights[k] = np.array(v)

        report = self._manager.verify_watermark(
            numpy_weights,
            self._signature_id,
            "fl-ehds-global-model",
            self._config,
        )

        result = {
            "round": round_num,
            "result": report.verification_result.value,
            "confidence": report.confidence_score,
            "matched_bits": report.matched_bits,
            "total_bits": report.extracted_bits,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        self._round_verifications.append(result)

        logger.info(
            f"Watermark verification round {round_num}: "
            f"{report.verification_result.value} "
            f"(confidence={report.confidence_score:.2%})"
        )
        return result

    def export_report(self) -> Dict[str, Any]:
        """Export watermarking report."""
        return {
            "signature_id": self._signature_id,
            "signature": (
                self._signature.to_dict() if self._signature else None
            ),
            "config": {
                "strength": self._strength,
                "length": self._length,
                "method": (
                    self._config.embedding_method.value
                    if self._config
                    else "spread_spectrum"
                ),
            },
            "embed_count": self._embed_count,
            "verifications": self._round_verifications,
            "final_status": (
                self._round_verifications[-1]["result"]
                if self._round_verifications
                else "not_verified"
            ),
        }


# =========================================================================
# 3. TIME GUARD
# =========================================================================


class TimeGuard:
    """Enforces time-limited data access for FL training (Art. 50).

    The data permit has a finite duration. Training must complete before
    the deadline, after which no further rounds are allowed.
    """

    def __init__(self, permit_duration_hours: float = 24.0):
        self._duration_hours = permit_duration_hours
        self._start_time: Optional[datetime] = None
        self._deadline: Optional[datetime] = None
        self._expired = False
        self._round_timestamps: List[Dict[str, Any]] = []

    def start(self):
        """Start the permit timer."""
        self._start_time = datetime.now()
        self._deadline = self._start_time + timedelta(
            hours=self._duration_hours
        )
        logger.info(
            f"TimeGuard started: deadline={self._deadline.isoformat(timespec='seconds')}, "
            f"duration={self._duration_hours}h"
        )

    def check_deadline(self, round_num: int) -> Tuple[bool, float]:
        """Check if training can continue.

        Returns:
            (allowed, remaining_hours)
            If deadline passed, returns (False, 0.0)
        """
        if self._deadline is None:
            return True, self._duration_hours

        now = datetime.now()
        if now >= self._deadline:
            self._expired = True
            logger.warning(
                f"TimeGuard: permit EXPIRED at round {round_num} "
                f"(deadline was {self._deadline.isoformat(timespec='seconds')})"
            )
            return False, 0.0

        remaining = (self._deadline - now).total_seconds() / 3600.0
        return True, remaining

    def log_round(self, round_num: int):
        """Log round timestamp for audit."""
        now = datetime.now()
        remaining = 0.0
        if self._deadline:
            remaining = max(
                0.0,
                (self._deadline - now).total_seconds() / 3600.0,
            )
        self._round_timestamps.append({
            "round": round_num,
            "timestamp": now.isoformat(timespec="seconds"),
            "remaining_hours": round(remaining, 4),
        })

    def export_report(self) -> Dict[str, Any]:
        """Export time guard report."""
        now = datetime.now()
        remaining = 0.0
        if self._deadline:
            remaining = max(
                0.0,
                (self._deadline - now).total_seconds() / 3600.0,
            )
        return {
            "permit_duration_hours": self._duration_hours,
            "start_time": (
                self._start_time.isoformat(timespec="seconds")
                if self._start_time
                else None
            ),
            "deadline": (
                self._deadline.isoformat(timespec="seconds")
                if self._deadline
                else None
            ),
            "expired": self._expired,
            "remaining_hours": round(remaining, 4),
            "rounds_logged": len(self._round_timestamps),
            "round_timestamps": self._round_timestamps,
        }


# =========================================================================
# 4. SECURE PROCESSING BRIDGE (Main Orchestrator)
# =========================================================================


class SecureProcessingBridge:
    """EHDS Art. 50 Secure Processing Environment bridge.

    Orchestrates three components:
    - EnclaveSimulator: TEE simulation with I/O validation
    - WatermarkingBridge: Model fingerprinting for traceability
    - TimeGuard: Time-limited data access enforcement

    Lifecycle:
        start_session -> [pre_round + post_round] x N -> end_session
    """

    def __init__(
        self,
        num_clients: int,
        config: Dict[str, Any],
    ):
        self._num_clients = num_clients
        self._config = config

        # Create sub-components based on config flags
        self._enclave: Optional[EnclaveSimulator] = None
        if config.get("enclave_enabled", True):
            self._enclave = EnclaveSimulator(
                num_clients=num_clients,
                allowed_output_types=config.get(
                    "allowed_outputs",
                    ["model_delta", "gradient", "metrics"],
                ),
            )

        self._watermark: Optional[WatermarkingBridge] = None
        if config.get("watermarking_enabled", True):
            self._watermark = WatermarkingBridge(
                watermark_strength=config.get("watermark_strength", 0.01),
                watermark_length=config.get("watermark_length", 256),
            )

        self._time_guard: Optional[TimeGuard] = None
        if config.get("time_limited_enabled", True):
            self._time_guard = TimeGuard(
                permit_duration_hours=config.get(
                    "permit_duration_hours", 24.0
                ),
            )

        self._session_active = False
        self._rounds_processed = 0

    def start_session(self):
        """Initialize all sub-components."""
        if self._enclave:
            self._enclave.initialize_enclaves()
        if self._watermark:
            self._watermark.initialize()
        if self._time_guard:
            self._time_guard.start()
        self._session_active = True
        logger.info(
            f"Secure Processing session started: "
            f"enclave={'ON' if self._enclave else 'OFF'}, "
            f"watermark={'ON' if self._watermark else 'OFF'}, "
            f"time_guard={'ON' if self._time_guard else 'OFF'}"
        )

    def pre_round(self, round_num: int) -> Tuple[bool, str]:
        """Pre-round checks: verify deadline, enclave active.

        Returns:
            (allowed, reason)
        """
        if self._time_guard:
            allowed, remaining = self._time_guard.check_deadline(round_num)
            if not allowed:
                return False, "Permit expired (deadline passed)"
            if remaining < 0.01:  # Less than 36 seconds
                logger.warning(
                    f"TimeGuard: less than 1 minute remaining at round {round_num}"
                )
        return True, "OK"

    def post_round(
        self,
        round_num: int,
        model_state_dict: Dict,
        num_clients_trained: int,
        per_client_samples: List[int],
    ) -> Dict:
        """Post-round processing: enclave I/O log + watermark embedding.

        Args:
            round_num: Current round number
            model_state_dict: Global model state_dict after aggregation
            num_clients_trained: Number of clients that trained this round
            per_client_samples: Samples per client

        Returns:
            Updated model_state_dict (with watermark if enabled)
        """
        # Log enclave I/O for each client
        if self._enclave:
            for cid in range(num_clients_trained):
                samples = (
                    per_client_samples[cid]
                    if cid < len(per_client_samples)
                    else 0
                )
                # Validate output type (model_delta is always allowed)
                self._enclave.validate_output(cid, "model_delta", 0)
                self._enclave.log_round(
                    round_num, cid, samples, "model_delta", 0
                )

        # Embed watermark into global model
        if self._watermark:
            model_state_dict = self._watermark.embed_watermark(
                model_state_dict
            )

        # Log time
        if self._time_guard:
            self._time_guard.log_round(round_num)

        self._rounds_processed += 1
        return model_state_dict

    def verify_final_model(
        self, model_state_dict: Dict, round_num: int
    ) -> Dict[str, Any]:
        """Verify watermark on the final global model.

        Returns:
            Verification result dict
        """
        if self._watermark:
            return self._watermark.verify_watermark(
                model_state_dict, round_num
            )
        return {"result": "watermarking_disabled"}

    def end_session(self) -> Dict[str, Any]:
        """End secure processing session, export combined report."""
        report = self.export_report()
        self._session_active = False
        logger.info(
            f"Secure Processing session ended: "
            f"{self._rounds_processed} rounds processed"
        )
        return report

    def export_report(self) -> Dict[str, Any]:
        """Export complete secure processing report."""
        report: Dict[str, Any] = {
            "secure_processing": True,
            "session_active": self._session_active,
            "rounds_processed": self._rounds_processed,
        }
        if self._enclave:
            report["enclave"] = self._enclave.export_report()
        if self._watermark:
            report["watermark"] = self._watermark.export_report()
        if self._time_guard:
            report["time_guard"] = self._time_guard.export_report()
        return report

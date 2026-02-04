"""
Basic FL Training Example
=========================
Demonstrates a complete FL training session using the FL-EHDS framework.

This example shows how to:
1. Set up governance (permits, opt-out)
2. Configure FL orchestration (aggregation, privacy)
3. Run local training at data holders
4. Aggregate results securely
"""

import asyncio
import numpy as np
from datetime import datetime, timedelta

# Core imports
from core.models import (
    DataPermit,
    PermitPurpose,
    DataCategory,
    GradientUpdate,
    TrainingConfig,
    PrivacyConfig,
)
from core.utils import load_config, setup_logging

# Layer 1: Governance
from governance.data_permits import DataPermitManager, PermitValidator
from governance.optout_registry import OptOutRegistry, OptOutChecker
from governance.compliance_logging import ComplianceLogger, AuditTrail

# Layer 2: Orchestration
from orchestration.aggregation import FedAvgAggregator
from orchestration.privacy import DifferentialPrivacy, GradientClipper, PrivacyAccountant
from orchestration.compliance import PurposeLimiter, OutputController

# Layer 3: Data Holders
from data_holders.training_engine import TrainingEngine, HardwareProfile
from data_holders.fhir_preprocessing import FHIRPreprocessor


def create_sample_data(num_samples: int = 100):
    """Create synthetic health data for demonstration."""
    np.random.seed(42)

    # Simulate patient features (age, lab values, etc.)
    X = np.random.randn(num_samples, 10).astype(np.float32)

    # Binary outcome (e.g., disease diagnosis)
    y = (X[:, 0] + X[:, 1] > 0).astype(np.float32)

    return X, y


def create_sample_model():
    """Create a simple model state (simulating neural network weights)."""
    return {
        "layer1.weight": np.random.randn(10, 32).astype(np.float32),
        "layer1.bias": np.zeros(32).astype(np.float32),
        "layer2.weight": np.random.randn(32, 16).astype(np.float32),
        "layer2.bias": np.zeros(16).astype(np.float32),
        "output.weight": np.random.randn(16, 1).astype(np.float32),
        "output.bias": np.zeros(1).astype(np.float32),
    }


async def run_fl_training():
    """Run a complete FL training session."""

    # Setup logging
    logger = setup_logging(level="INFO")
    print("\n" + "=" * 60)
    print("FL-EHDS Framework - Basic Training Example")
    print("=" * 60 + "\n")

    # =========================================================================
    # STEP 1: GOVERNANCE SETUP
    # =========================================================================
    print("[1/5] Setting up governance layer...")

    # Create data permit
    permit = DataPermit(
        permit_id="FLICS-2026-DEMO",
        hdab_id="HDAB-IT",
        requester_id="university-research-001",
        purpose=PermitPurpose.SCIENTIFIC_RESEARCH,
        data_categories=[DataCategory.EHR, DataCategory.LAB_RESULTS],
        member_states=["IT", "DE", "FR"],
        valid_until=datetime.utcnow() + timedelta(days=365),
    )

    # Validate and register permit
    validator = PermitValidator(strict_mode=True)
    if not validator.validate(permit):
        print("ERROR: Permit validation failed")
        return

    permit_manager = DataPermitManager()
    permit_manager.register_permit(permit)
    print(f"  - Permit registered: {permit.permit_id}")

    # Setup opt-out registry
    optout_registry = OptOutRegistry()
    optout_checker = OptOutChecker(optout_registry, on_optout="exclude")
    print("  - Opt-out registry initialized")

    # Setup compliance logging
    audit_trail = AuditTrail(storage_path="logs/demo")
    compliance_logger = ComplianceLogger(audit_trail, actor_id="fl-ehds-demo")
    print("  - Compliance logging enabled")

    # =========================================================================
    # STEP 2: ORCHESTRATION SETUP
    # =========================================================================
    print("\n[2/5] Configuring FL orchestration...")

    # Privacy configuration
    privacy_accountant = PrivacyAccountant(
        total_epsilon=10.0,
        total_delta=1e-5,
        accountant_type="rdp",
    )

    dp = DifferentialPrivacy(
        epsilon=1.0,
        delta=1e-5,
        max_grad_norm=1.0,
        accountant=privacy_accountant,
    )

    clipper = GradientClipper(max_norm=1.0, norm_type="l2")
    print(f"  - Differential privacy: ε={dp.epsilon}, δ={dp.delta}")
    print(f"  - Gradient clipping: max_norm={clipper.max_norm}")

    # Aggregation configuration
    aggregator = FedAvgAggregator(
        num_rounds=10,
        min_clients=2,
        weighted=True,
        early_stopping=True,
        early_stopping_patience=3,
    )
    print(f"  - Aggregation: FedAvg, {aggregator.num_rounds} rounds")

    # Purpose limitation
    purpose_limiter = PurposeLimiter()
    purpose_limiter.set_session_purpose(PermitPurpose.SCIENTIFIC_RESEARCH)
    print(f"  - Purpose: {permit.purpose.value}")

    # =========================================================================
    # STEP 3: SIMULATE DATA HOLDERS (CLIENTS)
    # =========================================================================
    print("\n[3/5] Initializing data holders...")

    clients = []
    num_clients = 3

    for i in range(num_clients):
        # Each client has different hardware
        hardware = HardwareProfile(
            device_type="gpu" if i == 0 else "cpu",
            memory_gb=16.0 if i == 0 else 8.0,
            compute_units=8 if i == 0 else 4,
            network_bandwidth_mbps=100.0,
            storage_available_gb=100.0,
        )

        config = TrainingConfig(
            batch_size=32,
            local_epochs=3,
            learning_rate=0.01,
            adaptive_batching=True,
        )

        engine = TrainingEngine(config=config, hardware_profile=hardware)
        clients.append({
            "id": f"hospital-{i + 1}",
            "engine": engine,
            "data": create_sample_data(num_samples=100 + i * 50),
        })
        print(f"  - Client {i + 1}: hospital-{i + 1} ({hardware.device_type})")

    # =========================================================================
    # STEP 4: RUN FL TRAINING
    # =========================================================================
    print("\n[4/5] Starting federated training...")
    print("-" * 40)

    # Start compliance logging session
    session_id = compliance_logger.start_session(
        permit_id=permit.permit_id,
        purpose=permit.purpose,
        data_categories=permit.data_categories,
        client_ids=[c["id"] for c in clients],
    )

    # Initialize global model
    global_model = create_sample_model()
    best_loss = float("inf")

    for round_num in range(aggregator.num_rounds):
        print(f"\n  Round {round_num + 1}/{aggregator.num_rounds}")

        # Verify permit is still valid
        if not permit_manager.verify_for_round(
            permit.permit_id,
            round_number=round_num,
            data_categories=permit.data_categories,
        ):
            print("  ERROR: Permit verification failed")
            break

        # Collect updates from each client
        updates = []

        for client in clients:
            # Initialize client model
            client["engine"].initialize_model(global_model, round_num)

            # Perform local training
            X, y = client["data"]
            gradients, metrics = client["engine"].train(
                data=X,
                labels=y,
                global_model_state=global_model,
            )

            # Apply gradient clipping
            clipped_gradients, was_clipped = clipper.clip(gradients)

            # Add differential privacy noise
            noised_gradients = dp.add_noise(clipped_gradients, round_num)

            # Create update
            update = GradientUpdate(
                client_id=client["id"],
                round_number=round_num,
                gradients=noised_gradients,
                num_samples=len(X),
                local_loss=metrics.loss,
                is_clipped=was_clipped,
                noise_added=True,
            )
            updates.append(update)

        # Aggregate updates
        global_model = aggregator.aggregate(updates, global_model)

        # Compute round metrics
        avg_loss = np.mean([u.local_loss for u in updates])
        total_samples = sum(u.num_samples for u in updates)

        print(f"    Clients: {len(updates)}, Samples: {total_samples}, "
              f"Avg Loss: {avg_loss:.4f}")

        # Log round
        compliance_logger.log_round(
            round_number=round_num,
            participating_clients=[u.client_id for u in updates],
            samples_processed=total_samples,
            round_metrics={"loss": avg_loss},
            privacy_spent=dp.epsilon,
        )

        # Check early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
        if aggregator.should_stop(avg_loss):
            print(f"\n  Early stopping at round {round_num + 1}")
            break

    # =========================================================================
    # STEP 5: FINALIZE
    # =========================================================================
    print("\n" + "-" * 40)
    print("\n[5/5] Training complete!")

    # Get privacy budget spent
    epsilon_spent, delta_spent = privacy_accountant.get_spent_budget()
    epsilon_remaining, _ = privacy_accountant.get_remaining_budget()

    print(f"\n  Final Results:")
    print(f"  - Best loss: {best_loss:.4f}")
    print(f"  - Privacy spent: ε={epsilon_spent:.2f} (remaining: {epsilon_remaining:.2f})")
    print(f"  - Gradient clips: {sum(clipper._clip_counts)}")

    # End compliance session
    compliance_logger.end_session(
        total_rounds=round_num + 1,
        final_metrics={"best_loss": best_loss},
        success=True,
    )

    # Generate compliance report
    output_controller = OutputController(min_aggregation_count=3)
    certificate = output_controller.generate_compliance_certificate(
        session_id=session_id,
        output_type="model",
        aggregation_count=sum(len(c["data"][0]) for c in clients),
        purpose=permit.purpose,
    )

    print(f"\n  Compliance Certificate Generated:")
    print(f"  - Session: {certificate['session_id']}")
    print(f"  - Status: {certificate['compliance_status']}")
    print(f"  - Legal basis: {certificate['legal_basis']}")

    print("\n" + "=" * 60)
    print("FL-EHDS Training Session Complete")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(run_fl_training())

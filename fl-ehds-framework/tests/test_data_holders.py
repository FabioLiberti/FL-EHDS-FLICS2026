"""
Tests for Data Holders Layer (Layer 3)
======================================
"""

import pytest
import numpy as np
from datetime import datetime
from typing import Dict, Any

from core.models import TrainingConfig, GradientUpdate
from core.exceptions import TrainingError, FHIRValidationError
from data_holders.training_engine import (
    TrainingEngine,
    AdaptiveTrainer,
    HardwareProfile,
    TrainingMetrics,
)
from data_holders.fhir_preprocessing import (
    FHIRPreprocessor,
    FeatureExtractor,
    DataNormalizer,
)
from data_holders.secure_communication import (
    SecureCommunicator,
    MessageEncryptor,
    ChannelManager,
)


class TestHardwareProfile:
    """Tests for hardware profile."""

    def test_can_handle_small_workload(self):
        """Test hardware can handle small workload."""
        profile = HardwareProfile(
            device_type="cpu",
            memory_gb=8.0,
            compute_units=4,
            network_bandwidth_mbps=100.0,
            storage_available_gb=50.0,
        )

        assert profile.can_handle(model_size_mb=100, batch_size=32)

    def test_cannot_handle_large_workload(self):
        """Test hardware cannot handle large workload."""
        profile = HardwareProfile(
            device_type="cpu",
            memory_gb=2.0,
            compute_units=2,
            network_bandwidth_mbps=10.0,
            storage_available_gb=10.0,
        )

        assert profile.can_handle(model_size_mb=4096, batch_size=128) is False


class TestTrainingEngine:
    """Tests for training engine."""

    def test_model_initialization(self):
        """Test model initialization from global state."""
        engine = TrainingEngine()

        model_state = {
            "layer1.weight": np.random.randn(10, 5),
            "layer1.bias": np.random.randn(10),
        }

        engine.initialize_model(model_state, round_number=1)

        assert engine._current_round == 1
        assert engine._model is not None

    def test_training_with_config(self):
        """Test training with custom configuration."""
        config = TrainingConfig(
            local_epochs=2,
            batch_size=16,
            learning_rate=0.01,
        )

        engine = TrainingEngine(config=config)

        # Initialize model
        model_state = {"weights": np.zeros(5), "bias": 0.0}
        engine.initialize_model(model_state, round_number=1)

        # Generate dummy data
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)

        # Train
        gradients, metrics = engine.train(X, y)

        assert isinstance(metrics, TrainingMetrics)
        assert metrics.num_samples == 100
        assert metrics.num_epochs == 2
        assert metrics.gradients_computed is True

    def test_batch_size_adjustment(self):
        """Test adaptive batch size adjustment."""
        config = TrainingConfig(
            batch_size=64,
            min_batch_size=8,
            max_batch_size=128,
            adaptive_batching=True,
        )

        profile = HardwareProfile(
            device_type="cpu",
            memory_gb=2.0,  # Limited memory
            compute_units=2,
            network_bandwidth_mbps=50.0,
            storage_available_gb=20.0,
        )

        engine = TrainingEngine(config=config, hardware_profile=profile)

        # Should reduce batch size for limited memory
        adjusted = engine._adjust_batch_size(dataset_size=1000)
        assert adjusted < config.batch_size

    def test_gradient_update_creation(self):
        """Test creation of gradient update message."""
        engine = TrainingEngine()
        engine._current_round = 5

        gradients = {"layer1": np.array([1.0, 2.0, 3.0])}
        metrics = TrainingMetrics(
            loss=0.5,
            accuracy=0.85,
            num_samples=100,
            num_epochs=3,
            training_time_seconds=10.5,
            gradients_computed=True,
        )

        update = engine.create_gradient_update("client-001", gradients, metrics)

        assert isinstance(update, GradientUpdate)
        assert update.client_id == "client-001"
        assert update.round_number == 5
        assert update.num_samples == 100
        assert update.local_loss == 0.5


class TestAdaptiveTrainer:
    """Tests for adaptive trainer."""

    def test_hardware_configuration(self):
        """Test automatic hardware configuration."""
        config = TrainingConfig(local_epochs=5)
        profile = HardwareProfile(
            device_type="cpu",
            memory_gb=2.0,  # Limited
            compute_units=2,  # Limited
            network_bandwidth_mbps=5.0,  # Limited
            storage_available_gb=10.0,
        )

        trainer = AdaptiveTrainer(
            config=config,
            hardware_profile=profile,
            adaptive_lr=True,
            gradient_accumulation=True,
        )

        adjustments = trainer.configure_for_hardware()

        # Should enable gradient accumulation for low memory
        assert "gradient_accumulation_steps" in adjustments
        # Should reduce epochs for limited compute
        assert "reduced_epochs" in adjustments
        # Should enable compression for low bandwidth
        assert "compression_enabled" in adjustments

    def test_learning_rate_warmup(self):
        """Test learning rate warmup."""
        config = TrainingConfig(learning_rate=0.1)
        trainer = AdaptiveTrainer(config=config, adaptive_lr=True)

        # During warmup (round < 5)
        lr_round_1 = trainer.adjust_learning_rate(round_number=0)
        lr_round_3 = trainer.adjust_learning_rate(round_number=2)
        lr_round_5 = trainer.adjust_learning_rate(round_number=4)

        # LR should increase during warmup
        assert lr_round_1 < lr_round_3 < lr_round_5

    def test_learning_rate_decay(self):
        """Test learning rate decay after warmup."""
        config = TrainingConfig(learning_rate=0.1)
        trainer = AdaptiveTrainer(config=config, adaptive_lr=True)

        lr_round_10 = trainer.adjust_learning_rate(round_number=10)
        lr_round_50 = trainer.adjust_learning_rate(round_number=50)

        # LR should decrease after warmup (cosine decay)
        assert lr_round_50 < lr_round_10

    def test_training_time_estimation(self):
        """Test training time estimation."""
        profile = HardwareProfile(
            device_type="gpu",
            memory_gb=16.0,
            compute_units=8,
            network_bandwidth_mbps=1000.0,
            storage_available_gb=100.0,
        )

        config = TrainingConfig(local_epochs=5)
        trainer = AdaptiveTrainer(config=config, hardware_profile=profile)

        estimate = trainer.estimate_training_time(
            dataset_size=10000,
            model_size_params=1000000,
        )

        assert estimate > 0


class TestFHIRPreprocessor:
    """Tests for FHIR R4 preprocessing."""

    def test_patient_resource_processing(self):
        """Test processing of Patient resource."""
        preprocessor = FHIRPreprocessor()

        patient = {
            "resourceType": "Patient",
            "id": "patient-001",
            "birthDate": "1980-05-15",
            "gender": "male",
            "address": [{"country": "IT"}],
        }

        features = preprocessor.process_resource(patient)

        assert "age" in features or "birth_year" in features
        assert "gender" in features

    def test_observation_resource_processing(self):
        """Test processing of Observation resource."""
        preprocessor = FHIRPreprocessor()

        observation = {
            "resourceType": "Observation",
            "id": "obs-001",
            "code": {
                "coding": [{"system": "http://loinc.org", "code": "8867-4"}]
            },
            "valueQuantity": {"value": 72, "unit": "bpm"},
            "effectiveDateTime": "2025-01-15T10:30:00Z",
        }

        features = preprocessor.process_resource(observation)

        assert features is not None

    def test_invalid_resource_handling(self):
        """Test handling of invalid FHIR resource."""
        preprocessor = FHIRPreprocessor(strict_validation=True)

        invalid_resource = {
            "resourceType": "Unknown",
            "id": "invalid-001",
        }

        with pytest.raises(FHIRValidationError):
            preprocessor.process_resource(invalid_resource)

    def test_batch_processing(self):
        """Test batch processing of FHIR resources."""
        preprocessor = FHIRPreprocessor()

        resources = [
            {"resourceType": "Patient", "id": f"pat-{i}", "gender": "female"}
            for i in range(5)
        ]

        results = preprocessor.process_batch(resources)

        assert len(results) == 5


class TestFeatureExtractor:
    """Tests for feature extraction."""

    def test_numeric_feature_extraction(self):
        """Test extraction of numeric features."""
        extractor = FeatureExtractor(
            feature_columns=["age", "weight", "height"]
        )

        record = {"age": 45, "weight": 70.5, "height": 175, "name": "John"}

        features = extractor.extract(record)

        assert len(features) == 3
        assert features[0] == 45

    def test_missing_feature_handling(self):
        """Test handling of missing features."""
        extractor = FeatureExtractor(
            feature_columns=["age", "weight"],
            missing_value_strategy="mean",
            default_values={"age": 50, "weight": 70},
        )

        record = {"age": 30}  # Missing weight

        features = extractor.extract(record)

        assert len(features) == 2
        assert features[1] == 70  # Default value

    def test_categorical_encoding(self):
        """Test encoding of categorical features."""
        extractor = FeatureExtractor(
            feature_columns=["gender"],
            categorical_columns=["gender"],
            encoding_strategy="onehot",
        )

        records = [
            {"gender": "male"},
            {"gender": "female"},
            {"gender": "male"},
        ]

        extractor.fit(records)
        features = extractor.extract(records[0])

        # One-hot encoding should produce multiple features
        assert len(features) >= 2


class TestDataNormalizer:
    """Tests for data normalization."""

    def test_standard_normalization(self):
        """Test standard (z-score) normalization."""
        normalizer = DataNormalizer(method="standard")

        data = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])

        normalizer.fit(data)
        normalized = normalizer.transform(data)

        # Mean should be ~0, std should be ~1
        assert np.abs(normalized.mean(axis=0)).max() < 0.1
        assert np.abs(normalized.std(axis=0) - 1).max() < 0.1

    def test_minmax_normalization(self):
        """Test min-max normalization."""
        normalizer = DataNormalizer(method="minmax")

        data = np.array([[0, 100], [50, 200], [100, 300]])

        normalizer.fit(data)
        normalized = normalizer.transform(data)

        # Values should be in [0, 1]
        assert normalized.min() >= 0
        assert normalized.max() <= 1

    def test_inverse_transform(self):
        """Test inverse transformation."""
        normalizer = DataNormalizer(method="standard")

        data = np.array([[1, 2], [3, 4], [5, 6]])

        normalizer.fit(data)
        normalized = normalizer.transform(data)
        restored = normalizer.inverse_transform(normalized)

        np.testing.assert_array_almost_equal(data, restored)


class TestSecureCommunicator:
    """Tests for secure communication."""

    def test_message_encryption(self):
        """Test message encryption and decryption."""
        encryptor = MessageEncryptor()

        message = {"type": "gradient_update", "data": [1.0, 2.0, 3.0]}

        encrypted = encryptor.encrypt(message)
        decrypted = encryptor.decrypt(encrypted)

        assert decrypted == message
        assert encrypted != str(message)

    def test_channel_establishment(self):
        """Test secure channel establishment."""
        manager = ChannelManager()

        # Create channel between client and server
        channel = manager.create_channel(
            source="client-001",
            destination="fl-server",
            encryption_type="aes-256",
        )

        assert channel is not None
        assert channel.is_secure is True

    def test_channel_message_integrity(self):
        """Test message integrity verification."""
        manager = ChannelManager()

        channel = manager.create_channel(
            source="client-001",
            destination="fl-server",
        )

        message = {"gradient": np.array([1.0, 2.0]).tolist()}

        # Send and verify
        sent = channel.send(message)
        received, verified = channel.receive_and_verify(sent)

        assert verified is True
        assert received == message

    def test_tampered_message_detection(self):
        """Test detection of tampered messages."""
        encryptor = MessageEncryptor(include_hmac=True)

        message = {"data": "sensitive"}
        encrypted = encryptor.encrypt(message)

        # Tamper with the message
        tampered = encrypted[:-5] + "XXXXX"

        with pytest.raises(Exception):  # Should raise integrity error
            encryptor.decrypt(tampered)


class TestIntegrationDataHolders:
    """Integration tests for data holders layer."""

    def test_complete_training_pipeline(self):
        """Test complete local training pipeline."""
        # Setup
        config = TrainingConfig(
            local_epochs=2,
            batch_size=32,
            learning_rate=0.01,
        )

        profile = HardwareProfile(
            device_type="cpu",
            memory_gb=8.0,
            compute_units=4,
            network_bandwidth_mbps=100.0,
            storage_available_gb=50.0,
        )

        trainer = AdaptiveTrainer(config=config, hardware_profile=profile)

        # Configure for hardware
        trainer.configure_for_hardware()

        # Initialize model
        model_state = {"weights": np.zeros(5), "bias": 0.0}
        trainer.initialize_model(model_state, round_number=1)

        # Generate data
        X = np.random.randn(200, 5)
        y = np.random.randint(0, 2, 200)

        # Train
        gradients, metrics = trainer.train(X, y)

        # Create update
        update = trainer.create_gradient_update("hospital-001", gradients, metrics)

        # Verify
        assert update.client_id == "hospital-001"
        assert update.round_number == 1
        assert metrics.loss > 0
        assert len(gradients) > 0

    def test_fhir_to_training_pipeline(self):
        """Test FHIR preprocessing to training flow."""
        # FHIR resources
        patients = [
            {
                "resourceType": "Patient",
                "id": f"pat-{i}",
                "birthDate": f"198{i}-01-01",
                "gender": "male" if i % 2 == 0 else "female",
            }
            for i in range(10)
        ]

        # Preprocess
        preprocessor = FHIRPreprocessor()
        processed = preprocessor.process_batch(patients)

        # Extract features
        extractor = FeatureExtractor(
            feature_columns=["age", "gender"],
            categorical_columns=["gender"],
        )
        extractor.fit(processed)

        features = np.array([extractor.extract(p) for p in processed])

        # Normalize
        normalizer = DataNormalizer(method="standard")
        normalizer.fit(features)
        normalized = normalizer.transform(features)

        # Should have processed data ready for training
        assert normalized.shape[0] == 10
        assert not np.isnan(normalized).any()

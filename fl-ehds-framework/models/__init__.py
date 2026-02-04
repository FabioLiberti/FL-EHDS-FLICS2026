"""
FL-EHDS Models Module

Provides neural network architectures and FL training utilities.

Includes:
- Model Zoo (MLP, CNN, Transformer, RNN)
- FL Training utilities
- Non-IID data partitioning
"""

# CNN FL Trainer
try:
    from .cnn_fl_trainer import (
        SimpleCNN,
        MedicalCNN,
        ResNetMedical,
        NonIIDPartitioner,
        DatasetManager,
        FLTrainingConfig,
        FederatedTrainer
    )
except ImportError:
    pass

# Model Zoo
try:
    from .model_zoo import (
        MODEL_INFO,
        # MLP Models
        SimpleMLP,
        DeepMLP,
        ResidualMLP,
        # CNN Models
        LeNet5,
        VGGStyle,
        ResNet18,
        MobileNetStyle,
        # Medical Models
        DenseNetMedical,
        AttentionModule,
        # Transformer Models
        VisionTransformer,
        PatchEmbedding,
        TransformerBlock,
        # RNN Models
        LSTMModel,
        GRUModel,
        # Factory functions
        create_model,
        get_model_info,
        list_models
    )
except ImportError:
    pass

__all__ = [
    # CNN FL Trainer
    'SimpleCNN',
    'MedicalCNN',
    'ResNetMedical',
    'NonIIDPartitioner',
    'DatasetManager',
    'FLTrainingConfig',
    'FederatedTrainer',
    # Model Zoo
    'MODEL_INFO',
    'SimpleMLP',
    'DeepMLP',
    'ResidualMLP',
    'LeNet5',
    'VGGStyle',
    'ResNet18',
    'MobileNetStyle',
    'DenseNetMedical',
    'VisionTransformer',
    'LSTMModel',
    'GRUModel',
    'create_model',
    'get_model_info',
    'list_models'
]

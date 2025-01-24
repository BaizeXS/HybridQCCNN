"""Configuration management for model training.

This package provides configuration tools including:
1. ConfigManager - Configuration loading and saving
2. Config schemas - Data structures for various configuration types:
   - Config: Main configuration class
   - DataConfig: Dataset configuration
   - ModelConfig: Model architecture configuration
   - TrainingConfig: Training parameters
   - QuantumConfig: Quantum component settings

Example:
    >>> from config import ConfigManager, ModelConfig
    >>> config_manager = ConfigManager()
    >>> model_config = ModelConfig(model_name='HybridResNet')
"""

from .manager import ConfigManager
from .schema import Config, DataConfig, ModelConfig, QuantumConfig, TrainingConfig

__all__ = [
    "ConfigManager",
    "Config",
    "DataConfig",
    "ModelConfig",
    "TrainingConfig",
    "QuantumConfig",
]

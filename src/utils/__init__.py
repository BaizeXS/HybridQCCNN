"""Utility modules for model management, training, and visualization.

This package provides various utility modules for:
1. Model Management - Model lifecycle management including building, training, and evaluation
2. Data Management - Dataset loading, preprocessing, and custom dataset implementation
3. Training - Training loops, metrics calculation, and evaluation tools
4. Visualization - Plotting tools for metrics, quantum states, and model analysis

Modules:
    model_management: Model lifecycle management tools
    data_management: Dataset management and preprocessing tools
    training: Training and evaluation utilities
    visualization: Visualization and plotting tools

Example:
    >>> from utils import ModelManager, DatasetManager, Trainer, MetricsPlotter
    >>> data_manager = DatasetManager(config)
    >>> model_manager = ModelManager(config)
    >>> trainer = Trainer(model, criterion, optimizer)
    >>> plotter = MetricsPlotter()
"""

from typing import List

from .data_management import CustomDataset, DatasetManager  # noqa: F401
from .model_management import ModelManager  # noqa: F401
from .training import MetricsCalculator, Trainer  # noqa: F401
from .visualization import MetricsPlotter, ModelPlotter, QuantumPlotter  # noqa: F401

# Model Management
__model_management__: List[str] = ["ModelManager"]

# Data Management
__data_management__: List[str] = ["DatasetManager", "CustomDataset"]

# Training
__training__: List[str] = ["Trainer", "MetricsCalculator"]

# Visualization
__visualization__: List[str] = ["MetricsPlotter", "QuantumPlotter", "ModelPlotter"]

__all__ = __model_management__ + __data_management__ + __training__ + __visualization__  # type: ignore

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

from .model_management import ModelManager
from .data_management import DatasetManager, CustomDataset
from .training import Trainer, MetricsCalculator
from .visualization import MetricsPlotter, QuantumPlotter, ModelPlotter

# Model Management
__model_management__ = ['ModelManager']

# Data Management
__data_management__ = ['DatasetManager', 'CustomDataset']

# Training
__training__ = ['Trainer', 'MetricsCalculator']

# Visualization
__visualization__ = ['MetricsPlotter', 'QuantumPlotter', 'ModelPlotter']

__all__ = (
    __model_management__ +
    __data_management__ +
    __training__ +
    __visualization__
)

# Version of the utils package
__version__ = '0.1.0'

# Package metadata
__author__ = 'BaizeXS'
__email__ = 'baizexs@gmail.com'
__description__ = 'Utilities for quantum-classical hybrid model management, training, and visualization'

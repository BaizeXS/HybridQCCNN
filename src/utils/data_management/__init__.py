"""Data management tools for deep learning datasets.

This module provides tools for:
1. Dataset loading and preprocessing
2. Custom dataset implementation
3. Data transformation pipeline
4. Training/validation/test data splitting

Main components:
- CustomDataset: Base class for custom dataset implementations
- DatasetManager: Central manager for dataset operations
"""

from .dataset_manager import CustomDataset, DatasetManager

__all__ = ["DatasetManager", "CustomDataset"]

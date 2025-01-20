"""Data management utilities for dataset handling and preprocessing.

This module provides tools for:
1. Dataset loading - Built-in and custom dataset support
2. Data preprocessing - Transformation pipelines and data augmentation
3. Data splitting - Training/validation/test set management
4. DataLoader configuration - Batch processing and memory optimization
"""

from .dataset_manager import DatasetManager, CustomDataset

__all__ = [
    'DatasetManager',
    'CustomDataset'
]

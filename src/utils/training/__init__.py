"""Training utilities for model training and evaluation.

This module provides:
1. Trainer - Main training loop and evaluation functionality
2. MetricsCalculator - Metrics computation and tracking
"""

from .metrics import MetricsCalculator
from .trainer import Trainer

__all__ = [
    'Trainer',
    'MetricsCalculator'
]

"""Quantum-classical hybrid neural network components.

This module provides:
1. QKernel - Quantum convolution kernel
2. Quanv2d - 2D quantum convolution layer
3. Enums - Output modes and aggregation methods for quantum layers
"""

from .qkernel import QKernel
from .quanv import Quanv2d, OutputMode, AggregationMethod

__all__ = [
    'QKernel',
    'Quanv2d',
    'OutputMode',
    'AggregationMethod'
]

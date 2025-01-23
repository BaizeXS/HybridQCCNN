"""Quantum-classical hybrid neural network components.

This package provides quantum components for neural networks including:
1. QKernel - Quantum convolution kernel
2. Quanv2d - 2D quantum convolution layer
3. Enums - Output modes and aggregation methods for quantum layers

Example:
    >>> from components import Quanv2d, OutputMode
    >>> quantum_conv = Quanv2d(in_channels=3, out_channels=16)
"""

from .qkernel import QKernel
from .quanv import AggregationMethod, OutputMode, Quanv2d

__all__ = ["QKernel", "Quanv2d", "OutputMode", "AggregationMethod"]

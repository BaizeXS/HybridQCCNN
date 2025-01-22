"""Benchmark models for testing and comparison.

This module provides:
1. ClassicNet - Classical CNN baseline model
2. HybridNet - Hybrid quantum-classical CNN model
"""

from .classic_net import ClassicNet
from .hybrid_net import HybridNet

__all__ = ["ClassicNet", "HybridNet"]

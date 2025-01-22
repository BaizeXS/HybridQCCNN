"""VGG-style model implementations.

This module provides:
1. SimpleVGG - Classical VGG implementation
2. HybridVGG - Hybrid quantum-classical VGG implementation
"""

from .hybrid_vgg import HybridVGG
from .simple_vgg import SimpleVGG

__all__ = ["SimpleVGG", "HybridVGG"]

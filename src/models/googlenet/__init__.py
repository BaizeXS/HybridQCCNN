"""GoogLeNet/Inception model implementations.

This module provides:
1. Classical components:
   - SimpleGoogLeNet - Classical GoogLeNet implementation
   - SimpleInception - Classical Inception module
   - SimpleInceptionAux - Auxiliary classifier
   - BasicConv2d - Basic convolution block
2. Hybrid components:
   - HybridGoogLeNet - Hybrid quantum-classical GoogLeNet
   - HybridInception - Hybrid Inception module
   - HybridConv2d - Hybrid convolution block
"""

from .hybrid_googlenet import HybridConv2d, HybridGoogLeNet, HybridInception
from .simple_googlenet import (
    BasicConv2d,
    SimpleGoogLeNet,
    SimpleInception,
    SimpleInceptionAux,
)

__all__ = [
    "SimpleGoogLeNet",
    "SimpleInception",
    "SimpleInceptionAux",
    "BasicConv2d",
    "HybridGoogLeNet",
    "HybridInception",
    "HybridConv2d",
]

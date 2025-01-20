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

from .hybrid_googlenet import HybridGoogLeNet, HybridInception, HybridConv2d
from .simple_googlenet import SimpleGoogLeNet, SimpleInception, SimpleInceptionAux, BasicConv2d

__all__ = [
    'SimpleGoogLeNet',
    'SimpleInception',
    'SimpleInceptionAux',
    'BasicConv2d',
    'HybridGoogLeNet',
    'HybridInception',
    'HybridConv2d'
]

"""ResNet model implementations.

This module provides:
1. Base architectures:
   - SimpleResNet - Classical ResNet implementation
   - HybridResNet - Hybrid quantum-classical ResNet
2. Building blocks:
   - BasicBlock - Classical residual block
   - HybridBlock - Hybrid residual block
3. Pre-configured models:
   - simple_resnet18/34 - Classical ResNet-18/34
   - hybrid_resnet18/34 - Hybrid ResNet-18/34
"""

from .hybrid_resnet import HybridBlock, HybridResNet, hybrid_resnet18, hybrid_resnet34
from .simple_resnet import BasicBlock, SimpleResNet, simple_resnet18, simple_resnet34

__all__ = [
    "SimpleResNet",
    "BasicBlock",
    "simple_resnet18",
    "simple_resnet34",
    "HybridResNet",
    "HybridBlock",
    "hybrid_resnet18",
    "hybrid_resnet34",
]

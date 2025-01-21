"""Neural network model architectures for quantum-classical hybrid systems.

This package provides various neural network architectures including:
1. Benchmark Models - Simple models for testing and comparison
2. VGG Models - VGG-style architectures
3. GoogLeNet Models - GoogLeNet/Inception-style architectures
4. ResNet Models - ResNet-style architectures

Example:
    >>> from models import SimpleVGG, HybridResNet
    >>> model = SimpleVGG(num_classes=10)
    >>> hybrid_model = HybridResNet(num_classes=10)
"""

from typing import Dict, Type, Union

# Benchmark Models
from .benchmark import ClassicNet, HybridNet
# GoogLeNet Models
from .googlenet import (
    SimpleGoogLeNet, HybridGoogLeNet,
    BasicConv2d, SimpleInception, SimpleInceptionAux,
    HybridInception, HybridConv2d
)
# ResNet Models
from .resnet import (
    SimpleResNet, HybridResNet,
    BasicBlock, HybridBlock,
    simple_resnet18, simple_resnet34,
    hybrid_resnet18, hybrid_resnet34
)
# VGG Models
from .vgg import SimpleVGG, HybridVGG

# Type alias for model factory functions
ModelFactory = Type[Union[
    ClassicNet, HybridNet,
    SimpleVGG, HybridVGG,
    SimpleGoogLeNet, HybridGoogLeNet,
    SimpleResNet, HybridResNet
]]

# Dictionary containing all available models
ALL_MODELS: Dict[str, ModelFactory] = {
    # Benchmark Models
    "ClassicNet": ClassicNet,
    "HybridNet": HybridNet,
    # VGG Models
    "SimpleVGG": SimpleVGG,
    "HybridVGG": HybridVGG,
    # GoogLeNet Models
    "SimpleGoogLeNet": SimpleGoogLeNet,
    "HybridGoogLeNet": HybridGoogLeNet,
    # ResNet Models
    "SimpleResNet": SimpleResNet,
    "HybridResNet": HybridResNet,
    "simple_resnet18": simple_resnet18,
    "simple_resnet34": simple_resnet34,
    "hybrid_resnet18": hybrid_resnet18,
    "hybrid_resnet34": hybrid_resnet34,
}

__all__ = [
    # Benchmark Models
    'ClassicNet', 'HybridNet',
    # VGG Models
    'SimpleVGG', 'HybridVGG',
    # GoogLeNet Models and Components
    'SimpleGoogLeNet', 'HybridGoogLeNet',
    'BasicConv2d', 'SimpleInception', 'SimpleInceptionAux',
    'HybridInception', 'HybridConv2d',
    # ResNet Models and Components
    'SimpleResNet', 'HybridResNet',
    'BasicBlock', 'HybridBlock',
    'simple_resnet18', 'simple_resnet34',
    'hybrid_resnet18', 'hybrid_resnet34',
    # Model Registry
    'ALL_MODELS'
]

# Version of the models package
__version__ = '0.1.0'

# Package metadata
__author__ = 'BaizeXS'
__email__ = 'baizexs@gmail.com'
__description__ = 'Neural network architectures for quantum-classical hybrid systems'

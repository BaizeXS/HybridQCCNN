from .hybrid_resnet import (
    HybridResNet,
    HybridBlock,
    hybrid_resnet18,
    hybrid_resnet34
)
from .simple_resnet import (
    SimpleResNet,
    BasicBlock,
    simple_resnet18,
    simple_resnet34
)

__all__ = [
    'SimpleResNet',
    'BasicBlock',
    'simple_resnet18',
    'simple_resnet34',
    'HybridResNet',
    'HybridBlock',
    'hybrid_resnet18',
    'hybrid_resnet34'
]

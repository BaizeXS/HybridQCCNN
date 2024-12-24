from .benchmark_classic import ClassicNet
from .benchmark_hybrid import HybridNet
from .simple_vgg import SimpleVGG
from .hybrid_vgg import HybridVGG
from .simple_googlenet import SimpleGoogLeNet
from .hybrid_googlenet import HybridGoogLeNet
from .simple_resnet import SimpleResNet, simple_resnet18, simple_resnet34
from .hybrid_resnet import HybridResNet, hybrid_resnet18, hybrid_resnet34

__all__ = [
    'ALL_MODELS',
    'ClassicNet',
    'HybridNet',
    'SimpleVGG',
    'HybridVGG',
    'SimpleGoogLeNet',
    'HybridGoogLeNet',
    'SimpleResNet',
    'HybridResNet',
    'simple_resnet18',
    'simple_resnet34',
    'hybrid_resnet18',
    'hybrid_resnet34',
]

ALL_MODELS = {
    'ClassicNet': ClassicNet,
    'HybridNet': HybridNet,
    'SimpleVGG': SimpleVGG,
    'HybridVGG': HybridVGG,
    'SimpleGoogLeNet': SimpleGoogLeNet,
    'HybridGoogLeNet': HybridGoogLeNet,
    'SimpleResNet': SimpleResNet,
    'HybridResNet': HybridResNet,
}

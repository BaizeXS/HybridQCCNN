import torch
import torch.nn as nn

from components.quanv import Quanv2d


class HybridVGG(nn.Module):

    def __init__(self, num_classes=10, **kwargs):
        super(HybridVGG, self).__init__()
        self.features = nn.Sequential(
            # Layer 1: Classical Convolution
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Layer 2: Quantum Convolution
            # Reduce the dimension first and then increase the dimension
            nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0),
            Quanv2d(3, 64, kernel_size=2, stride=1, padding=1, **kwargs),
            # No need to introduce ReLU here,
            # because quantum computing has already introduced nonlinearity.
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Layer 3: Classical Convolution
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

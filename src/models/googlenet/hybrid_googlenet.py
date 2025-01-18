from typing import Any, Optional

import torch
import torch.nn as nn
from torch import Tensor

from components.quanv import Quanv2d
from .simple_googlenet import BasicConv2d, SimpleInception, SimpleInceptionAux


class HybridGoogLeNet(nn.Module):

    def __init__(
            self,
            num_classes: int = 10,
            aux_logits: bool = True,
            dropout: float = 0.4,
            dropout_aux: float = 0.5,
            **kwargs: Any,
    ):
        super(HybridGoogLeNet, self).__init__()

        self.aux_logits = aux_logits

        self.conv1 = BasicConv2d(3, 8, kernel_size=3, stride=1, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.conv2 = BasicConv2d(8, 16, kernel_size=1)
        self.conv3 = BasicConv2d(16, 24, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.inception3a = HybridInception(24, 16, 8, 16, 4, 8, 8, **kwargs)
        self.inception3b = SimpleInception(48, 32, 16, 24, 8, 12, 16)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.inception4a = HybridInception(84, 48, 8, 24, 4, 8, 24, **kwargs)
        self.inception4b = SimpleInception(104, 32, 16, 32, 8, 16, 24)

        if self.aux_logits:
            self.aux = SimpleInceptionAux(104, num_classes, dropout=dropout_aux)
        else:
            self.aux = None

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(104, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool2(x)

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)

        x = self.inception4a(x)
        x = self.inception4b(x)

        aux: Optional[Tensor] = None
        if self.training and self.aux:
            aux = self.aux(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        if self.training and self.aux_logits:
            return x, aux

        return x


class HybridInception(nn.Module):

    def __init__(
            self,
            in_channels: int,
            ch1x1: int,
            ch3x3red: int,
            ch3x3: int,
            ch5x5red: int,
            ch5x5: int,
            pool_proj: int,
            **kwargs: Any,
    ):
        super(HybridInception, self).__init__()

        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3red, kernel_size=1),
            BasicConv2d(ch3x3red, ch3x3, kernel_size=3, padding=1)
        )

        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, ch5x5red, kernel_size=1),
            # We need to add an explicit padding layer here as the maximum quantum kernel size is 2 in this case.
            nn.ZeroPad2d((1, 0, 1, 0)),
            HybridConv2d(ch5x5red, ch5x5, kernel_size=2, stride=1, padding=0, **kwargs)
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, pool_proj, kernel_size=1)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)


class HybridConv2d(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, **kwargs: Any) -> None:
        super(HybridConv2d, self).__init__()
        self.quanv = Quanv2d(in_channels, out_channels, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.quanv(x)
        x = self.bn(x)
        return x

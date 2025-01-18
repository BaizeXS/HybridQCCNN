from typing import Optional, Type, Union, List, Any

import torch
import torch.nn as nn

from components.quanv import Quanv2d
from .simple_resnet import BasicBlock


class HybridBlock(nn.Module):
    expansion = 1

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            **kwargs: Any,
    ):
        super(HybridBlock, self).__init__()
        # Layer 1
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        # Layer 2
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=4,  # Reduce the channel number
                               kernel_size=1, stride=1, padding=0, bias=False)
        # We need to add an explicit padding layer here as the maximum quantum kernel size is 2 in this case.
        self.pad = nn.ZeroPad2d((1, 0, 1, 0))
        self.quanv = Quanv2d(in_channels=4, out_channels=out_channels,  # Increase the channel number
                             kernel_size=2, stride=1, padding=0, **kwargs)
        # Layer 3
        self.conv3 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.pad(out)
        out = self.quanv(out)

        out = self.conv3(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class HybridResNet(nn.Module):
    def __init__(
            self,
            block: Type[Union[BasicBlock]],
            num_blocks: List[int],
            num_classes: int = 10,
            **kwargs: Any,
    ):
        super(HybridResNet, self).__init__()

        self.in_channels = 16

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.in_channels,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1, use_quanv2d=False, **kwargs)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2, use_quanv2d=False, **kwargs)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2, use_quanv2d=True, **kwargs)
        self.layer4 = self._make_layer(block, 128, num_blocks[3], stride=2, use_quanv2d=True, **kwargs)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def _make_layer(
            self,
            block: Type[Union[BasicBlock]],
            channels: int,
            block_num: int,
            stride: int = 1,
            use_quanv2d: bool = False,
            **kwargs: Any,
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.in_channels != channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channels * block.expansion),
            )

        layers = []
        if use_quanv2d:
            layers.append(HybridBlock(self.in_channels, channels, stride=stride, downsample=downsample, **kwargs))
        else:
            layers.append(block(self.in_channels, channels, stride=stride, downsample=downsample))
        self.in_channels = channels * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channels, channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def hybrid_resnet18(num_classes=10, **kwargs):
    return HybridResNet(BasicBlock, [2, 2, 2, 2], num_classes, **kwargs)


def hybrid_resnet34(num_classes=10, **kwargs):
    return HybridResNet(BasicBlock, [3, 4, 6, 3], num_classes, **kwargs)

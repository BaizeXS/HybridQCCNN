import torch.nn as nn
import torch.nn.functional as F

from components.quanv import Quanv2d


class HybridNet(nn.Module):

    def __init__(self, num_classes=10, stride=2, **kwargs):
        super(HybridNet, self).__init__()
        self.quanv = Quanv2d(in_channels=1, out_channels=4, kernel_size=2, stride=stride, padding=0, **kwargs)
        self.conv = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(8 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        bs = x.shape[0]
        # Quantum Convolution
        x = self.quanv(x)
        x = self.conv(x)
        x = F.relu(x)
        # Linear
        x = x.view(bs, -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        return x

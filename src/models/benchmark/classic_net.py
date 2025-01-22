import torch.nn as nn
import torch.nn.functional as F


class ClassicNet(nn.Module):

    def __init__(self, num_classes=10):
        super(ClassicNet, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(
            in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(8 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        bs = x.shape[0]
        # Convolution
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        # Linear
        x = x.view(bs, -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

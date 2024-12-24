import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassicNet(nn.Module):

    def __init__(self, num_classes=10):
        super(ClassicNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(8 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        bs = x.shape[0]
        # Convolution
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        # Linear
        x = x.view(bs, -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


if __name__ == '__main__':
    net = ClassicNet()
    inputs = torch.randn((1, 1, 14, 14))
    outputs = net(inputs)
    print(outputs.shape)

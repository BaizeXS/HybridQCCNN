import torch
import torch.nn as nn

from components.quanv import Quanv2d, OutputMode, AggregationMethod


def basic_quanv2d_example():
    """
    Basic Quanv2d usage example
    """
    # Create a simple quantum convolution layer
    quanv = Quanv2d(
        in_channels=3,  # Number of input channels (e.g., RGB image)
        out_channels=8,  # Number of output channels
        kernel_size=2,  # Convolution kernel size 2x2
        stride=1,  # Stride
        padding=1,  # Padding
        output_mode=OutputMode.CLASSICAL,  # Use classical output mode
        aggregation_method=AggregationMethod.MEAN,  # Use mean aggregation
    )

    # Create an example input tensor (batch_size, channels, height, width)
    x = torch.randn(2, 3, 14, 14)

    # Forward propagation
    output = quanv(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")


def quantum_mode_example():
    """
    Example of using quantum output mode
    """
    quanv = Quanv2d(
        in_channels=3,  # Number of input channels
        out_channels=8,  # Number of output channels
        kernel_size=2,  # Convolution kernel size 2x2
        stride=1,  # Stride
        padding=1,  # Padding
        output_mode=OutputMode.QUANTUM,  # Use quantum output mode
        preserve_quantum_info=True,  # Preserve quantum information
    )

    # Create an example input tensor
    x = torch.randn(2, 3, 14, 14)

    # Forward propagation
    output = quanv(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")


def weighted_aggregation_example():
    """
    Example of using weighted aggregation method
    """
    quanv = Quanv2d(
        in_channels=3,  # Number of input channels
        out_channels=8,  # Number of output channels
        kernel_size=2,  # Convolution kernel size 2x2
        stride=1,  # Stride
        padding=1,  # Padding
        output_mode=OutputMode.CLASSICAL,  # Use classical output mode
        aggregation_method=AggregationMethod.WEIGHTED,  # Use weighted aggregation
    )

    # Create an example input tensor
    x = torch.randn(2, 3, 14, 14)

    # Forward propagation
    output = quanv(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")


def hybrid_cnn_example():
    """
    Example of creating a hybrid CNN model, combining quantum convolution and classical convolution
    """

    class HybridCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.quanv = Quanv2d(
                in_channels=3,
                out_channels=8,
                kernel_size=2,
                stride=1,
                padding=1
            )
            self.conv = nn.Conv2d(8, 16, kernel_size=3, padding=1)
            self.pool = nn.MaxPool2d(2)
            self.fc = nn.Linear(16 * 7 * 7, 10)  # Assume input is 14x14 image

        def forward(self, x):
            x = self.quanv(x)
            x = self.conv(x)
            x = torch.relu(x)
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x

    # Create model
    model = HybridCNN()

    # Create example input
    x = torch.randn(2, 3, 14, 14)

    # Forward propagation
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")


if __name__ == "__main__":
    print("Basic quantum convolution example:")
    basic_quanv2d_example()
    print("\nQuantum mode example:")
    quantum_mode_example()
    print("\nWeighted aggregation example:")
    weighted_aggregation_example()
    print("\nHybrid CNN model example:")
    hybrid_cnn_example()

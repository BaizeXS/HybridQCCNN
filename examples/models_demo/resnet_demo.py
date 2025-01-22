import torch

from components.quanv import AggregationMethod, OutputMode
from models.resnet import (
    hybrid_resnet18,
    hybrid_resnet34,
    simple_resnet18,
    simple_resnet34,
)


def simple_resnet_example():
    """
    Basic SimpleResNet example
    """
    # Create a classical ResNet model
    model = simple_resnet18(num_classes=10)

    # Create an example input tensor (batch_size, channels, height, width)
    x = torch.randn(2, 3, 32, 32)  # CIFAR format input

    # Forward propagation
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model structure:\n{model}")


def hybrid_resnet_basic_example():
    """
    Basic HybridResNet example
    """
    # Create a hybrid quantum-classical ResNet model
    model = hybrid_resnet18(
        num_classes=10,
        output_mode=OutputMode.QUANTUM,
        aggregation_method=AggregationMethod.MEAN,
    )

    # Create an example input tensor
    x = torch.randn(2, 3, 32, 32)  # CIFAR format input

    # Forward propagation
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model structure:\n{model}")


def hybrid_resnet_classical_mode_example():
    """
    HybridResNet example with classical output mode
    """
    model = hybrid_resnet34(
        num_classes=10,
        output_mode=OutputMode.CLASSICAL,
        aggregation_method=AggregationMethod.WEIGHTED,
    )

    x = torch.randn(2, 3, 32, 32)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")


def compare_resnet_models_example():
    """
    Compare SimpleResNet and HybridResNet example
    """
    # Create models with different depths
    models = {
        "SimpleResNet18": simple_resnet18(num_classes=10),
        "SimpleResNet34": simple_resnet34(num_classes=10),
        "HybridResNet18": hybrid_resnet18(num_classes=10),
        "HybridResNet34": hybrid_resnet34(num_classes=10),
    }

    # Create the same input
    x = torch.randn(2, 3, 32, 32)

    print("Model comparison:")
    # Compare outputs and parameters for each model
    for name, model in models.items():
        # Forward pass
        output = model(x)
        print(f"\n{name}:")
        print(f"Output shape: {output.shape}")

        # Count parameters
        params = sum(p.numel() for p in model.parameters())
        print(f"Number of parameters: {params:,}")


if __name__ == "__main__":
    print("SimpleResNet example:")
    simple_resnet_example()
    print("\nBasic HybridResNet example:")
    hybrid_resnet_basic_example()
    print("\nHybridResNet example with classical output mode:")
    hybrid_resnet_classical_mode_example()
    print("\nModel comparison example:")
    compare_resnet_models_example()

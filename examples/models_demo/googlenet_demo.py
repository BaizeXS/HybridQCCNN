import torch

from components.quanv import OutputMode, AggregationMethod
from models.googlenet import SimpleGoogLeNet, HybridGoogLeNet


def simple_googlenet_example():
    """
    Basic SimpleGoogLeNet example
    """
    # Create a classical GoogLeNet model
    model = SimpleGoogLeNet(num_classes=10, aux_logits=True)

    # Create an example input tensor (batch_size, channels, height, width)
    x = torch.randn(2, 3, 32, 32)  # CIFAR format input

    # Forward propagation
    output = model(x)
    if isinstance(output, tuple):
        output, aux = output
        print(f"Auxiliary output shape: {aux.shape}")

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model structure:\n{model}")


def hybrid_googlenet_basic_example():
    """
    Basic HybridGoogLeNet example
    """
    # Create a hybrid quantum-classical GoogLeNet model
    model = HybridGoogLeNet(
        num_classes=10,
        aux_logits=True,
        output_mode=OutputMode.QUANTUM,
        aggregation_method=AggregationMethod.MEAN
    )

    # Create an example input tensor
    x = torch.randn(2, 3, 32, 32)  # CIFAR format input

    # Forward propagation
    output = model(x)
    if isinstance(output, tuple):
        output, aux = output
        print(f"Auxiliary output shape: {aux.shape}")

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model structure:\n{model}")


def hybrid_googlenet_classical_mode_example():
    """
    HybridGoogLeNet example with classical output mode
    """
    model = HybridGoogLeNet(
        num_classes=10,
        aux_logits=False,
        output_mode=OutputMode.CLASSICAL,
        aggregation_method=AggregationMethod.WEIGHTED
    )

    x = torch.randn(2, 3, 32, 32)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")


def compare_googlenet_models_example():
    """
    Compare SimpleGoogLeNet and HybridGoogLeNet example
    """
    # Create two models
    classic_model = SimpleGoogLeNet(num_classes=10, aux_logits=False)
    hybrid_model = HybridGoogLeNet(num_classes=10, aux_logits=False)

    # Create the same input
    x = torch.randn(2, 3, 32, 32)

    # Get the outputs from the two models
    classic_output = classic_model(x)
    hybrid_output = hybrid_model(x)

    print("Model comparison:")
    print(f"SimpleGoogLeNet output shape: {classic_output.shape}")
    print(f"HybridGoogLeNet output shape: {hybrid_output.shape}")

    # Compare number of parameters
    classic_params = sum(p.numel() for p in classic_model.parameters())
    hybrid_params = sum(p.numel() for p in hybrid_model.parameters())
    print(f"\nNumber of parameters:")
    print(f"SimpleGoogLeNet: {classic_params:,}")
    print(f"HybridGoogLeNet: {hybrid_params:,}")


if __name__ == "__main__":
    print("SimpleGoogLeNet example:")
    simple_googlenet_example()
    print("\nBasic HybridGoogLeNet example:")
    hybrid_googlenet_basic_example()
    print("\nHybridGoogLeNet example with classical output mode:")
    hybrid_googlenet_classical_mode_example()
    print("\nModel comparison example:")
    compare_googlenet_models_example()

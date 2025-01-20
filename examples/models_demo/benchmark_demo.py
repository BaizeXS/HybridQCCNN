import torch
from models.benchmark import ClassicNet, HybridNet
from components.quanv import OutputMode, AggregationMethod

def classic_net_example():
    """
    Basic ClassicNet example
    """
    # Create a classic CNN model
    model = ClassicNet(num_classes=10)
    
    # Create an example input tensor (batch_size, channels, height, width)
    x = torch.randn(2, 1, 28, 28)  # MNIST format input
    
    # Forward propagation
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model structure:\n{model}")

def hybrid_net_basic_example():
    """
    Basic HybridNet example
    """
    # Create a hybrid quantum-classic CNN model
    model = HybridNet(
        num_classes=10,
        output_mode=OutputMode.QUANTUM,
        aggregation_method=AggregationMethod.MEAN
    )
    
    # Create an example input tensor
    x = torch.randn(2, 1, 28, 28)  # MNIST format input
    
    # Forward propagation
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model structure:\n{model}")

def hybrid_net_classical_mode_example():
    """
    HybridNet example with classical output mode
    """
    model = HybridNet(
        num_classes=10,
        output_mode=OutputMode.CLASSICAL,
        aggregation_method=AggregationMethod.WEIGHTED
    )
    
    x = torch.randn(2, 1, 28, 28)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

def compare_models_example():
    """
    Compare ClassicNet and HybridNet example
    """
    # Create two models
    classic_model = ClassicNet(num_classes=10)
    hybrid_model = HybridNet(num_classes=10)
    
    # Create the same input
    x = torch.randn(2, 1, 28, 28)
    
    # Get the outputs from the two models
    classic_output = classic_model(x)
    hybrid_output = hybrid_model(x)
    
    print("Model comparison:")
    print(f"ClassicNet output shape: {classic_output.shape}")
    print(f"HybridNet output shape: {hybrid_output.shape}")

if __name__ == "__main__":
    print("ClassicNet example:")
    classic_net_example()
    print("\nBasic HybridNet example:")
    hybrid_net_basic_example()
    print("\nHybridNet example with classical output mode:")
    hybrid_net_classical_mode_example()
    print("\nModel comparison example:")
    compare_models_example() 
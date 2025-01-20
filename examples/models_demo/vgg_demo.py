import torch
from models.vgg import SimpleVGG, HybridVGG
from components.quanv import OutputMode, AggregationMethod

def simple_vgg_example():
    """
    Basic SimpleVGG example
    """
    # Create a classical VGG model
    model = SimpleVGG(num_classes=10)
    
    # Create an example input tensor (batch_size, channels, height, width)
    x = torch.randn(2, 3, 32, 32)  # CIFAR format input
    
    # Forward propagation
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model structure:\n{model}")

def hybrid_vgg_basic_example():
    """
    Basic HybridVGG example
    """
    # Create a hybrid quantum-classical VGG model
    model = HybridVGG(
        num_classes=10,
        output_mode=OutputMode.QUANTUM,
        aggregation_method=AggregationMethod.MEAN
    )
    
    # Create an example input tensor
    x = torch.randn(2, 3, 32, 32)  # CIFAR format input
    
    # Forward propagation
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model structure:\n{model}")

def hybrid_vgg_classical_mode_example():
    """
    HybridVGG example with classical output mode
    """
    model = HybridVGG(
        num_classes=10,
        output_mode=OutputMode.CLASSICAL,
        aggregation_method=AggregationMethod.WEIGHTED
    )
    
    x = torch.randn(2, 3, 32, 32)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

def compare_vgg_models_example():
    """
    Compare SimpleVGG and HybridVGG example
    """
    # Create two models
    classic_model = SimpleVGG(num_classes=10)
    hybrid_model = HybridVGG(num_classes=10)
    
    # Create the same input
    x = torch.randn(2, 3, 32, 32)
    
    # Get the outputs from the two models
    classic_output = classic_model(x)
    hybrid_output = hybrid_model(x)
    
    print("Model comparison:")
    print(f"SimpleVGG output shape: {classic_output.shape}")
    print(f"HybridVGG output shape: {hybrid_output.shape}")
    
    # Compare number of parameters
    classic_params = sum(p.numel() for p in classic_model.parameters())
    hybrid_params = sum(p.numel() for p in hybrid_model.parameters())
    print(f"\nNumber of parameters:")
    print(f"SimpleVGG: {classic_params:,}")
    print(f"HybridVGG: {hybrid_params:,}")

if __name__ == "__main__":
    print("SimpleVGG example:")
    simple_vgg_example()
    print("\nBasic HybridVGG example:")
    hybrid_vgg_basic_example()
    print("\nHybridVGG example with classical output mode:")
    hybrid_vgg_classical_mode_example()
    print("\nModel comparison example:")
    compare_vgg_models_example() 
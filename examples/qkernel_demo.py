import torch
import pennylane as qml
import numpy as np
from src.components.qkernel import QKernel

def basic_qkernel_example():
    """
    Basic QKernel example
    """
    # Create a basic QKernel
    qkernel = QKernel(
        quantum_channels=1,     # Single channel input
        kernel_size=2,          # 2x2 convolution kernel
        num_param_blocks=2      # 2 parameter blocks
    )

    # Create example input data (4 inputs corresponding to a 2x2 convolution kernel)
    inputs = torch.tensor([0.5, -0.3, 0.2, 0.1])
    
    # Create random weights
    weights = torch.randn(2, 8)  # (num_param_blocks, 2 * num_qubits)
    
    # Calculate using the default circuit
    result = qkernel.circuit(inputs, weights)
    print(f"Input shape: {inputs.shape}")
    print(f"Weight shape: {weights.shape}")
    print(f"Output result: {result}")

def custom_circuit_example():
    """
    Example of using a custom quantum circuit
    """
    def custom_quantum_circuit(inputs, weights):
        """
        Custom quantum circuit implementation
        Using different quantum gates
        """
        num_qubits = len(inputs)
        
        # Encoding layer - use RX gate instead of RY gate
        for i in range(num_qubits):
            qml.RX(inputs[i], wires=i)
        
        # Parameterized layer
        for layer in range(len(weights)):
            # Entanglement using CNOT
            for i in range(num_qubits-1):
                qml.CNOT(wires=[i, i+1])
            
            # Rotation using RZ
            for i in range(num_qubits):
                qml.RZ(weights[layer, i], wires=i)
        
        # Return measurement results
        return [qml.expval(qml.PauliX(wires=i)) for i in range(num_qubits)]

    # Define weight shapes
    weight_shapes = {
        "weights": (2, 4)  # (num_param_blocks, num_qubits)
    }

    # Create QKernel using custom circuit
    custom_qkernel = QKernel(
        quantum_channels=1,
        kernel_size=2,
        num_param_blocks=2,
        kernel_circuit=custom_quantum_circuit,
        weight_shapes=weight_shapes
    )

    # Test custom circuit
    inputs = torch.tensor([0.1, 0.2, 0.3, 0.4])
    weights = torch.randn(2, 4)
    result = custom_qkernel.circuit(inputs, weights)
    print(f"Custom circuit output: {result}")

def multi_channel_example():
    """
    Example of a multi-channel QKernel
    """
    # Create a 3-channel QKernel (similar to RGB images)
    qkernel = QKernel(
        quantum_channels=3,     # 3 channels
        kernel_size=2,          # 2x2 convolution kernel
        num_param_blocks=3      # 3 parameter blocks
    )

    # Calculate the required number of qubits
    num_qubits = 3 * 2 * 2  # channels * kernel_height * kernel_width
    
    # Create example input
    inputs = torch.randn(num_qubits)
    weights = torch.randn(3, 2 * num_qubits)
    
    # Run circuit
    result = qkernel.circuit(inputs, weights)
    print(f"Number of qubits: {num_qubits}")
    print(f"Input shape: {inputs.shape}")
    print(f"Weight shape: {weights.shape}")
    print(f"Output dimension: {len(result)}")

def parameter_visualization_example():
    """
    Example of parameter visualization for a QKernel
    """
    qkernel = QKernel(
        quantum_channels=1,
        kernel_size=2,
        num_param_blocks=4
    )

    # Create random weights and analyze their distribution
    weights = torch.randn(*qkernel.weight_shapes["weights"])
    
    print("Weight statistics:")
    print(f"Shape: {weights.shape}")
    print(f"Mean: {weights.mean().item():.3f}")
    print(f"Standard deviation: {weights.std().item():.3f}")
    print(f"Minimum: {weights.min().item():.3f}")
    print(f"Maximum: {weights.max().item():.3f}")

if __name__ == "__main__":
    print("Basic QKernel example:")
    basic_qkernel_example()
    
    print("\nCustom quantum circuit example:")
    custom_circuit_example()
    
    print("\nMulti-channel QKernel example:")
    multi_channel_example()
    
    print("\nParameter visualization example:")
    parameter_visualization_example() 
"""Quantum convolution kernel implementation.

This module provides the quantum kernel class that implements:
- Quantum circuit for convolution operations
- Parameter management for quantum operations
- Input encoding and measurement
"""

from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pennylane as qml
import torch
from pennylane.measurements import ExpectationMP

ArrayLike = Union[list, np.ndarray, torch.Tensor]
QuantumCircuit = Callable[[ArrayLike, ArrayLike], List[float]]


class QKernel:
    """Quantum convolution kernel for quantum-classical hybrid neural networks.

    This class implements a quantum circuit that acts as a convolution kernel:
    - Encodes classical data into quantum states
    - Applies parameterized quantum operations
    - Measures quantum states to produce classical output

    The default kernel structure consists of:
    1. Encoding Layer: H + RY(input) gates
    2. Parametric Layer: CRZ + RY(param) gates
    3. Measurement Layer: PauliZ measurements

    Attributes:
        kernel_size (Tuple[int, int]): Size of the convolution kernel.
        num_qubits (int): Number of qubits in the quantum circuit.
        num_param_blocks (int): Number of parameterized quantum blocks.
        circuit (QuantumCircuit): Quantum circuit function.
        weight_shapes (Dict): Shapes of trainable weights.
    """

    def __init__(
        self,
        quantum_channels: int,
        kernel_size: Union[int, Tuple[int, int]] = 2,
        num_param_blocks: int = 2,
        kernel_circuit: Optional[QuantumCircuit] = None,
        weight_shapes: Optional[Dict[str, Tuple[int, ...]]] = None,
    ):
        """Initialize quantum kernel.

        Args:
            quantum_channels (int): Number of input quantum channels.
            kernel_size (Union[int, Tuple[int, int]]): Size of convolution kernel.
            num_param_blocks (int): Number of parameterized quantum blocks.
            kernel_circuit (Optional[QuantumCircuit]): Custom quantum circuit function.
            weight_shapes (Optional[Dict]): Custom weight shapes for quantum circuit.

        Raises:
            ValueError: If parameters are invalid.
        """
        # Validate parameters
        self._validate_params(
            quantum_channels,
            kernel_size,
            num_param_blocks,
            kernel_circuit,
            weight_shapes,
        )

        # Set kernel parameters
        self.kernel_size = (
            kernel_size
            if isinstance(kernel_size, tuple)
            else (kernel_size, kernel_size)
        )
        self.num_qubits = quantum_channels * self.kernel_size[0] * self.kernel_size[1]
        self.num_param_blocks = num_param_blocks

        # Set quantum circuit and weights
        self.circuit = kernel_circuit if kernel_circuit else self._default_circuit
        self.weight_shapes = (
            weight_shapes
            if weight_shapes
            else {"weights": (num_param_blocks, 2 * self.num_qubits)}
        )

    @staticmethod
    def _validate_params(
        quantum_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        num_param_blocks: int,
        kernel_circuit: Optional[QuantumCircuit],
        weight_shapes: Optional[Dict],
    ) -> None:
        """Validate initialization parameters.

        Args:
            quantum_channels: Number of quantum channels
            kernel_size: Kernel dimensions
            num_param_blocks: Number of parameter blocks
            kernel_circuit: Custom quantum circuit
            weight_shapes: Custom weight shapes

        Raises:
            ValueError: If any parameter is invalid
        """
        if not isinstance(quantum_channels, int) or quantum_channels <= 0:
            raise ValueError("quantum_channels must be a positive integer")

        if (
            not isinstance(kernel_size, (int, tuple))
            or (isinstance(kernel_size, int) and kernel_size <= 0)
            or (
                isinstance(kernel_size, tuple)
                and any(size <= 0 for size in kernel_size)
            )
        ):
            raise ValueError(
                "kernel_size must be a positive integer or a tuple of positive integers"
            )

        if not isinstance(num_param_blocks, int) or num_param_blocks <= 0:
            raise ValueError("num_param_blocks must be a positive integer")

        if kernel_circuit and not weight_shapes:
            raise ValueError(
                "Must provide weight_shapes when using custom kernel circuit"
            )

    def _default_circuit(
        self, inputs: ArrayLike, weights: ArrayLike
    ) -> list[ExpectationMP]:
        """Default quantum circuit implementation

        The circuit consists of three layers:
        1. Encoding Layer: Encode classical input data into quantum states
        2. Parametric Layer: Apply parameterized quantum operations
        3. Measurement Layer: Measure quantum states to get classical output

        Circuit structure per qubit:
        - Encoding: H + RY(input)
        - Parametric (repeated num_param_blocks times): CRZ + RY(param)

        Args:
            inputs: Input data to be processed
            weights: Trainable weights for the quantum circuit

        Returns:
            List of expectation values from quantum measurements
        """
        # Validate inputs
        if len(inputs) != self.num_qubits:
            raise ValueError(f"Expected {self.num_qubits} inputs, got {len(inputs)}")

        if weights.shape != (self.num_param_blocks, 2 * self.num_qubits):  # type: ignore
            expected_shape = (self.num_param_blocks, 2 * self.num_qubits)
            raise ValueError(
                f"Incorrect weights shape. Expected {expected_shape}, "
                f"got {weights.shape}"  # type: ignore
            )

        # 1. Encoding Layer
        for qubit in range(self.num_qubits):
            qml.Hadamard(wires=qubit)
            qml.RY(inputs[qubit], wires=qubit)  # type: ignore

        # 2. Parametric Layer
        for layer in range(self.num_param_blocks):
            # Entanglement
            for qubit in range(self.num_qubits):
                qml.CRZ(
                    weights[layer, qubit],  # type: ignore
                    wires=[qubit, (qubit + 1) % self.num_qubits],  # type: ignore
                )
            # Rotation
            for qubit in range(self.num_qubits):
                qml.RY(weights[layer, self.num_qubits + qubit], wires=qubit)  # type: ignore

        # 3. Measurement Layer
        return [qml.expval(qml.PauliZ(wires=qubit)) for qubit in range(self.num_qubits)]

    def __repr__(self) -> str:
        """String representation of the quantum kernel"""
        return (
            f"QKernel(num_qubits={self.num_qubits}, "
            f"kernel_size={self.kernel_size}, "
            f"num_param_blocks={self.num_param_blocks})"
        )

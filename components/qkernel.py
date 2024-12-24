from typing import Callable, Union, Dict, Tuple

import numpy as np
import pennylane as qml
import torch

ArrayLike = Union[list, np.ndarray, torch.Tensor]


class QKernel:

    def __init__(
            self,
            quantum_channels: int,
            kernel_size: Union[int, Tuple[int, int]] = 2,
            num_param_blocks: int = 2,
            kernel_circuit: Callable[[ArrayLike], None] = None,
            weight_shapes: Dict[str, Tuple[int, ...]] = None,
    ):
        """Quantum Kernel"""
        self.validate_params(quantum_channels, kernel_size, num_param_blocks, kernel_circuit, weight_shapes)

        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.num_qubits = quantum_channels * self.kernel_size[0] * self.kernel_size[1]
        self.num_param_blocks = num_param_blocks

        self.circuit = kernel_circuit if kernel_circuit else self._default_circuit
        self.weight_shapes = weight_shapes if weight_shapes else {"weights": (num_param_blocks, 2 * self.num_qubits)}

    @staticmethod
    def validate_params(
            quantum_channels: int,
            kernel_size: Union[int, Tuple[int, int]],
            num_param_blocks: int,
            kernel_circuit: Callable[[ArrayLike], None],
            weight_shapes: dict[str, tuple]
    ):
        if not isinstance(quantum_channels, int) or quantum_channels <= 0:
            raise ValueError("quantum_channels must be a positive integer")
        if (not isinstance(kernel_size, (int, tuple)) or
                (isinstance(kernel_size, int) and kernel_size <= 0) or
                (isinstance(kernel_size, tuple) and any(size <= 0 for size in kernel_size))):
            raise ValueError("kernel_size must be a positive integer or a tuple of positive integers")
        if not isinstance(num_param_blocks, int) or num_param_blocks <= 0:
            raise ValueError("num_param_blocks must be a positive integer")
        if kernel_circuit and not weight_shapes:
            raise ValueError("Must provide weight_shapes for custom kernel circuit")

    def _default_circuit(self, inputs: ArrayLike, weights: ArrayLike):
        # Encoding Layer
        for qubit in range(self.num_qubits):
            qml.Hadamard(wires=qubit)
            qml.RY(inputs[qubit], wires=qubit)

        # Parametric Layer
        for layer in range(self.num_param_blocks):
            # Entanglement
            for qubit in range(self.num_qubits):
                qml.CRZ(weights[layer, qubit], wires=[qubit, (qubit + 1) % self.num_qubits])
            # Rotation
            for qubit in range(self.num_qubits):
                qml.RY(weights[layer, self.num_qubits + qubit], wires=qubit)

        # Observation Layer
        _expectations = [qml.expval(qml.PauliZ(wires=qubit)) for qubit in range(self.num_qubits)]
        return _expectations

    def __repr__(self):
        return (f"QKernel(num_qubits={self.num_qubits}, kernel_size={self.kernel_size}, "
                f"num_param_blocks={self.num_param_blocks})")

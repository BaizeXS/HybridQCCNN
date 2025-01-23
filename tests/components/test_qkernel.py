import time

import numpy as np
import pennylane as qml
import pytest
import torch

from components import QKernel

# Test parameters
TEST_PARAMS = {
    "quantum_channels": 2,
    "kernel_size": 2,
    "num_param_blocks": 2,
    "num_qubits": 8,  # 2 channels * 2*2 kernel
}


@pytest.fixture
def default_kernel():
    """Return a QKernel with default configuration"""
    return QKernel(
        quantum_channels=TEST_PARAMS["quantum_channels"],
        kernel_size=TEST_PARAMS["kernel_size"],
        num_param_blocks=TEST_PARAMS["num_param_blocks"],
    )


@pytest.fixture
def sample_inputs(default_kernel):
    """Return standard test inputs"""
    return torch.randn(default_kernel.num_qubits)


@pytest.fixture
def sample_weights(default_kernel):
    """Return standard test weights"""
    return torch.randn(*default_kernel.weight_shapes["weights"])


def test_initialization(default_kernel):
    """Test kernel initialization parameters"""
    assert default_kernel.num_qubits == TEST_PARAMS["num_qubits"]
    assert default_kernel.kernel_size == (
        TEST_PARAMS["kernel_size"],
        TEST_PARAMS["kernel_size"],
    )
    assert default_kernel.num_param_blocks == TEST_PARAMS["num_param_blocks"]


def test_custom_circuit():
    """Test custom circuit functionality"""

    def custom_circuit(inputs, weights):
        return [0.0] * TEST_PARAMS["num_qubits"]

    custom_kernel = QKernel(
        quantum_channels=TEST_PARAMS["quantum_channels"],
        kernel_size=TEST_PARAMS["kernel_size"],
        kernel_circuit=custom_circuit,
        weight_shapes={"weights": (TEST_PARAMS["num_param_blocks"], 16)},
    )
    assert custom_kernel.circuit == custom_circuit


@pytest.mark.parametrize(
    "invalid_param",
    [
        {
            "quantum_channels": 0,
            "error_match": "quantum_channels must be a positive integer",
        },
        {
            "kernel_size": 0,
            "error_match": "kernel_size must be a positive integer",
        },
        {
            "num_param_blocks": 0,
            "error_match": "num_param_blocks must be a positive integer",
        },
    ],
)
def test_invalid_parameters(invalid_param):
    """Test invalid parameters"""
    params = {
        "quantum_channels": TEST_PARAMS["quantum_channels"],
        "kernel_size": TEST_PARAMS["kernel_size"],
        "num_param_blocks": TEST_PARAMS["num_param_blocks"],
        "kernel_circuit": None,
        "weight_shapes": None,
    }

    # Update invalid parameters
    for key, value in invalid_param.items():
        if key != "error_match":
            params[key] = value

    with pytest.raises(ValueError, match=invalid_param["error_match"]):
        QKernel(**params)


def test_quantum_circuit_execution(default_kernel, sample_inputs, sample_weights):
    """Test quantum circuit execution"""
    qdevice = qml.device("default.qubit", wires=default_kernel.num_qubits)
    qnode = qml.QNode(default_kernel.circuit, device=qdevice, interface="torch")

    # Execute circuit
    result = qnode(sample_inputs, sample_weights)

    # Verify output
    assert len(result) == default_kernel.num_qubits
    assert all(-1 <= x <= 1 for x in result)
    assert all(isinstance(x, torch.Tensor) for x in result)
    result_tensor = torch.stack(result)
    assert isinstance(result_tensor, torch.Tensor)


def test_default_circuit_structure(default_kernel, sample_inputs, sample_weights):
    """Test default quantum circuit structure"""
    with qml.tape.QuantumTape() as tape:
        default_kernel._default_circuit(sample_inputs, sample_weights)

    ops = tape.operations

    # Verify operation counts
    hadamard_count = sum(1 for op in ops if isinstance(op, qml.Hadamard))
    ry_count = sum(1 for op in ops if isinstance(op, qml.RY))
    crz_count = sum(1 for op in ops if isinstance(op, qml.CRZ))

    assert hadamard_count == default_kernel.num_qubits
    assert ry_count == default_kernel.num_qubits * (1 + default_kernel.num_param_blocks)
    assert crz_count == default_kernel.num_qubits * default_kernel.num_param_blocks


@pytest.mark.parametrize("input_type", ["list", "numpy", "torch"])
def test_input_type_validation(default_kernel, input_type):
    """Test input data type validation"""
    qdevice = qml.device("default.qubit", wires=default_kernel.num_qubits)
    qnode = qml.QNode(default_kernel.circuit, device=qdevice, interface="torch")

    if input_type == "list":
        inputs = [1.0] * default_kernel.num_qubits
    elif input_type == "numpy":
        inputs = np.array([1.0] * default_kernel.num_qubits)
    else:
        inputs = torch.ones(default_kernel.num_qubits)

    weights = torch.randn(*default_kernel.weight_shapes["weights"])
    result = qnode(inputs, weights)
    assert len(result) == default_kernel.num_qubits


def test_input_dimension_validation(default_kernel, sample_weights):
    """Test input dimension validation"""
    wrong_inputs = torch.randn(default_kernel.num_qubits + 1)
    with pytest.raises(
        ValueError, match=f"Expected {default_kernel.num_qubits} inputs"
    ):
        default_kernel._default_circuit(wrong_inputs, sample_weights)


@pytest.mark.parametrize(
    "kernel_config",
    [
        {"quantum_channels": 1, "kernel_size": 3, "expected": (3, 3)},
        {"quantum_channels": 1, "kernel_size": (2, 3), "expected": (2, 3)},
    ],
)
def test_kernel_size_variations(kernel_config):
    """Test different kernel size configurations"""
    qkernel = QKernel(
        quantum_channels=kernel_config["quantum_channels"],
        kernel_size=kernel_config["kernel_size"],
    )
    assert qkernel.kernel_size == kernel_config["expected"]


@pytest.mark.parametrize(
    "config",
    [
        {"quantum_channels": 1, "kernel_size": 2, "expected_qubits": 4},
        {"quantum_channels": 2, "kernel_size": 2, "expected_qubits": 8},
        {"quantum_channels": 1, "kernel_size": (2, 3), "expected_qubits": 6},
        {"quantum_channels": 2, "kernel_size": (2, 3), "expected_qubits": 12},
    ],
)
def test_qubit_count_calculation(config):
    """Test qubit count calculation for different configurations"""
    qkernel = QKernel(
        quantum_channels=config["quantum_channels"], kernel_size=config["kernel_size"]
    )
    assert qkernel.num_qubits == config["expected_qubits"]


@pytest.mark.benchmark
def test_circuit_performance(default_kernel, sample_inputs, sample_weights):
    """Test circuit execution performance"""
    qdevice = qml.device("default.qubit", wires=default_kernel.num_qubits)
    qnode = qml.QNode(default_kernel.circuit, device=qdevice, interface="torch")

    start_time = time.time()
    for _ in range(100):
        _ = qnode(sample_inputs, sample_weights)
    execution_time = time.time() - start_time

    assert execution_time < 5.0  # 100 executions should take less than 5 seconds

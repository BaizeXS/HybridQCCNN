import pennylane as qml
import torch

from components import QKernel
from utils.visualization import plot_qcircuit

if __name__ == "__main__":
    qkernel = QKernel(quantum_channels=1, kernel_size=2, num_param_blocks=2)
    qdevice = qml.device("default.qubit", wires=qkernel.num_qubits)
    qnode = qml.QNode(qkernel.circuit, device=qdevice, interface="torch", diff_method="best")

    test_inputs = torch.randn(qkernel.num_qubits)
    test_weights = torch.randn(*qkernel.weight_shapes["weights"])

    plot_qcircuit(qnode, test_inputs, test_weights)

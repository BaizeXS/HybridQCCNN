import torch
import pennylane as qml

from components import QKernel

if __name__ == "__main__":
    qkernel = QKernel(quantum_channels=2, kernel_size=2, num_param_blocks=2)
    qdevice = qml.device("lightning.qubit", wires=qkernel.num_qubits)
    qnode = qml.QNode(qkernel.circuit, device=qdevice, interface="torch", diff_method="best")
    qlayer = qml.qnn.TorchLayer(qnode, qkernel.weight_shapes)

    test_inputs = torch.randn(qkernel.num_qubits)
    test_weights = torch.randn(*qkernel.weight_shapes["weights"])

    qnode_result = qnode(test_inputs, test_weights)
    qlayer_result = qlayer(test_inputs)

    print("Basic functionality test qnode result: ", qnode_result)
    print("Basic functionality test qlayer result: ", qlayer_result)

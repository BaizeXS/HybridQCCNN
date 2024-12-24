import torch

from components.quafu_extensions.quafu_qkernel import QuafuQKernel
from components.quafu_extensions.quafu_torch_layer import QuafuTorchLayer

if __name__ == "__main__":
    USE_QCloud = False
    api_token = ("ccE4eawT5dsWtM359uqnDQX6vBVvVDMgIrPlKwAmT2x.Qf0cjMzgTO4EzNxojIwhXZiwCMzgTM6ICZpJye"
                 ".9JiN1IzUIJiOicGbhJCLiQ1VKJiOiAXe0Jye")
    qdevice = "Dongling" if USE_QCloud else "simulator"
    qkernel = QuafuQKernel(in_channels=1, kernel_size=2, num_qlayers=2, qdevice=qdevice, api_token=api_token)

    qlayer = QuafuTorchLayer(qkernel, qkernel.weight_shapes)

    test_inputs = torch.randn(qkernel.num_qubits)
    test_weights = torch.randn(*qkernel.weight_shapes["weights"])

    circuit_result = qkernel.circuit(test_inputs, test_weights)
    qlayer_result = qlayer.circuit(test_inputs, test_weights)

    print("Basic functionality test circuit result: ", circuit_result)
    print("Basic functionality test qlayer result: ", qlayer_result)

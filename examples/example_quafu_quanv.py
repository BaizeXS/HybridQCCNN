import time

import torch

from components import QuafuQuanv2d

if __name__ == "__main__":
    USE_QCloud = False
    api_token = ("ccE4eawT5dsWtM359uqnDQX6vBVvVDMgIrPlKwAmT2x.Qf0cjMzgTO4EzNxojIwhXZiwCMzgTM6ICZpJye"
                 ".9JiN1IzUIJiOicGbhJCLiQ1VKJiOiAXe0Jye")
    qdevice = "Dongling" if USE_QCloud else "simulator"

    test_quanv = QuafuQuanv2d(in_channels=1, out_channels=4, kernel_size=2, stride=2, padding=0,
                              qkernel=None, num_qlayers=2, qdevice=qdevice, api_token=api_token)

    test_start_time = time.time()
    test_inputs = torch.randn(1, 1, 14, 14)
    test_result = test_quanv(test_inputs)
    test_end_time = time.time()

    test_execute_time = test_end_time - test_start_time

    # print("Result: ", test_result)
    print("Result Shape: ", test_result.shape)
    print(f"Execute time: {test_execute_time: .2f}s.")

import time

import torch

from components import Quanv2d

if __name__ == "__main__":
    test_quanv = Quanv2d(in_channels=1, out_channels=4, kernel_size=2, stride=2, padding=0,
                         qkernel=None, num_qlayers=2, qdevice="lightning.qubit",
                         qdevice_kwargs={"batch_obs": True, "shots": 1000, "mcmc": True},
                         diff_method="best")

    test_start_time = time.time()
    test_inputs = torch.randn(1, 1, 14, 14)
    test_result = test_quanv(test_inputs)
    test_end_time = time.time()

    test_execute_time = test_end_time - test_start_time

    # print("Result: ", test_result)
    print("Result Shape: ", test_result.shape)
    print(f"Execute time: {test_execute_time: .2f}s.")

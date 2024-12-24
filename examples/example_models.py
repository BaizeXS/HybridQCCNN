import time

import torch

from models import *

if __name__ == '__main__':
    test_model = hybrid_resnet18(num_classes=10, qdevice="lightning.qubit", diff_method="best")
    test_model.eval()

    total_params = sum(p.numel() for p in test_model.parameters() if p.requires_grad)
    print("Number of parameters: ", total_params)

    test_input = torch.randn(1, 3, 64, 64)

    test_start_time = time.time()
    test_result = test_model(test_input)
    test_end_time = time.time()

    test_execution_time = test_end_time - test_start_time

    # print("Result: ", test_result)
    print(f"Result Shape: {test_result.shape}.")
    print(f"Execute time: {test_execution_time: .2f}s.")

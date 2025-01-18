import pytest
import torch

# 设置随机种子确保测试可重复
@pytest.fixture(autouse=True)
def setup_random_seed():
    torch.manual_seed(42)
    yield

# 可以添加更多全局fixture 
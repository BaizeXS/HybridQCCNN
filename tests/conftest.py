import random

import numpy as np
import pytest
import torch


# Set random seed to ensure test reproducibility
@pytest.fixture(autouse=True)
def setup_random_seed():
    """Set random seed to ensure test reproducibility"""
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    yield


@pytest.fixture
def device():
    """Return available computing device"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def tmp_path_with_cleanup(tmp_path):
    """Provide temporary directory and clean up after tests"""
    yield tmp_path
    import shutil

    shutil.rmtree(tmp_path)

from pathlib import Path

import pytest
from torch.utils.data import DataLoader

from config import DataConfig
from utils.data_management import CustomDataset, DatasetManager

# Test parameters
TEST_CONFIG = {
    "batch_size": 32,
    "train_split": 0.8,
    "num_workers": 2,
    "pin_memory": True,
    "train_transforms": [{"name": "ToTensor"}],
    "val_transforms": [{"name": "ToTensor"}],
    "test_transforms": [{"name": "ToTensor"}],
}


class MockDataset(CustomDataset):
    """Mock dataset for testing"""

    def _load_data(self):
        self.data = [i for i in range(100)]
        self.targets = [i % 10 for i in range(100)]


@pytest.fixture
def default_manager():
    """Return a DatasetManager with default configuration"""
    config = DataConfig(
        name="MNIST",
        input_shape=(1, 28, 28),
        num_classes=10,
        dataset_type="MNIST",
        **TEST_CONFIG,
    )
    return DatasetManager(config, data_dir="./datasets")  # type: ignore


@pytest.fixture
def mock_dataset_file(tmp_path):
    """Create a mock dataset file for testing"""
    dataset_file = tmp_path / "mock_dataset.py"
    dataset_code = """from utils.data_management import CustomDataset

class MockDataset(CustomDataset):
    def _load_data(self):
        self.data = [i for i in range(100)]
        self.targets = [i % 10 for i in range(100)]
"""
    dataset_file.write_text(dataset_code)
    return dataset_file


@pytest.fixture
def custom_manager(mock_dataset_file):
    """Return a DatasetManager with custom dataset configuration"""
    config = DataConfig(
        name="MockDataset",
        input_shape=(1, 28, 28),
        num_classes=10,
        dataset_type="CUSTOM",
        custom_dataset_path=mock_dataset_file,
        **TEST_CONFIG,
    )
    return DatasetManager(config, data_dir="./datasets")  # type: ignore


def test_initialization(default_manager):
    """Test manager initialization"""
    assert default_manager.config.dataset_type == "MNIST"
    assert default_manager.config.batch_size == TEST_CONFIG["batch_size"]


def test_transform_creation(default_manager):
    """Test transform creation from config"""
    transforms = [
        {"name": "ToTensor"},
        {"name": "Normalize", "args": {"mean": [0.5], "std": [0.5]}},
    ]
    transform = default_manager._get_transforms(transforms)
    assert transform is not None
    assert len(transform.transforms) == 2


def test_dataset_splitting(custom_manager):
    """Test dataset splitting functionality"""
    dataset = MockDataset(Path("./datasets"))
    train_dataset, val_dataset = custom_manager._split_dataset(
        dataset, TEST_CONFIG["train_split"]
    )

    total_size = len(dataset)
    expected_train_size = int(total_size * TEST_CONFIG["train_split"])

    assert len(train_dataset) == expected_train_size
    assert len(val_dataset) == total_size - expected_train_size


def test_dataloader_creation(custom_manager):
    """Test DataLoader creation"""
    dataset = MockDataset(Path("./datasets"))
    loader = custom_manager._create_loader(dataset, shuffle=True)

    assert isinstance(loader, DataLoader)
    assert loader.batch_size == TEST_CONFIG["batch_size"]
    assert loader.num_workers == TEST_CONFIG["num_workers"]
    assert loader.pin_memory == TEST_CONFIG["pin_memory"]


def test_get_data_loaders(default_manager):
    """Test complete data loader creation pipeline"""
    train_loader, val_loader, test_loader = default_manager.get_data_loaders()

    assert all(
        isinstance(loader, DataLoader)
        for loader in [train_loader, val_loader, test_loader]
    )


@pytest.mark.parametrize(
    "invalid_config",
    [
        # Invalid dataset_type
        {
            "name": "MNIST",
            "input_shape": (1, 28, 28),
            "num_classes": 10,
            "dataset_type": "UNKNOWN",
        },
        # CUSTOM type but no custom_dataset_path
        {
            "name": "MNIST",
            "input_shape": (1, 28, 28),
            "num_classes": 10,
            "dataset_type": "CUSTOM",
        },
        # Invalid batch_size
        {
            "name": "MNIST",
            "input_shape": (1, 28, 28),
            "num_classes": 10,
            "dataset_type": "MNIST",
            "batch_size": 0,
        },
        # Invalid train_split
        {
            "name": "MNIST",
            "input_shape": (1, 28, 28),
            "num_classes": 10,
            "dataset_type": "MNIST",
            "train_split": 1.5,
        },
    ],
)
def test_invalid_configurations(invalid_config):
    """Test handling of invalid configurations"""
    with pytest.raises(
        ValueError
    ):  # Only catch ValueError, as our validation raises ValueError
        DataConfig(**invalid_config)


def test_custom_dataset_implementation():
    """Test custom dataset implementation"""
    dataset = MockDataset(Path("./datasets"))

    assert len(dataset) == 100
    assert len(dataset.targets) == 100

    # Test getitem
    data, target = dataset[0]
    assert isinstance(data, int)
    assert isinstance(target, int)


@pytest.mark.parametrize(
    "transform_config",
    [
        [{"name": "ToTensor"}],
        [{"name": "Resize", "args": {"size": 224}}],
        [
            {"name": "ToTensor"},
            {"name": "Normalize", "args": {"mean": [0.5], "std": [0.5]}},
        ],
    ],
)
def test_transform_configurations(default_manager, transform_config):
    """Test various transform configurations"""
    transform = default_manager._get_transforms(transform_config)
    assert transform is not None
    assert len(transform.transforms) == len(transform_config)


def test_custom_dataset_loading(mock_dataset_file):
    """Test loading custom dataset from file"""
    config = DataConfig(
        name="MockDataset",
        input_shape=(1, 28, 28),
        num_classes=10,
        dataset_type="CUSTOM",
        custom_dataset_path=mock_dataset_file,
        batch_size=32,
        train_split=0.8,
        num_workers=2,
        pin_memory=True,
        train_transforms=[{"name": "ToTensor"}],
        val_transforms=[{"name": "ToTensor"}],
        test_transforms=[{"name": "ToTensor"}],
    )
    manager = DatasetManager(config, data_dir="./datasets")  # type: ignore
    train_loader, val_loader, test_loader = manager.get_data_loaders()

    assert train_loader is not None
    assert val_loader is not None
    assert test_loader is not None

import tempfile
from pathlib import Path

import pytest

from config import (
    Config, ConfigManager, DataConfig, ModelConfig,
    TrainingConfig, QuantumConfig
)

# Test parameters
TEST_PARAMS = {
    'name': 'test_model',
    'version': 'v1',
    'description': 'Test configuration'
}


@pytest.fixture
def sample_data_config():
    """Return a standard test data configuration"""
    return DataConfig(
        name="test_dataset",
        input_shape=(3, 32, 32),
        num_classes=10,
        dataset_type="CIFAR10",
        batch_size=64
    )


@pytest.fixture
def sample_model_config():
    """Return a standard test model configuration"""
    return ModelConfig(
        name="SimpleVGG",
        model_type="classic",
        model_kwargs={"num_classes": 10}
    )


@pytest.fixture
def sample_config(sample_data_config, sample_model_config):
    """Return a standard test configuration"""
    return Config(
        name=TEST_PARAMS['name'],
        version=TEST_PARAMS['version'],
        description=TEST_PARAMS['description'],
        data=sample_data_config,
        model=sample_model_config,
        training=TrainingConfig()
    )


@pytest.fixture
def config_manager():
    """Return a ConfigManager instance"""
    return ConfigManager()


def test_config_initialization(sample_config):
    """Test configuration initialization"""
    assert sample_config.name == TEST_PARAMS['name']
    assert sample_config.version == TEST_PARAMS['version']
    assert isinstance(sample_config.data, DataConfig)
    assert isinstance(sample_config.model, ModelConfig)
    assert isinstance(sample_config.training, TrainingConfig)


def test_config_paths(sample_config):
    """Test configuration path properties"""
    # Test base directory
    assert sample_config.base_dir == Path("./outputs")

    # Test tensorboard directory
    assert sample_config.tensorboard_dir == sample_config.base_dir / "tensorboard"


def test_config_custom_paths(sample_config):
    """Test configuration with custom output directory"""
    custom_output_dir = Path("/custom/output/path")
    sample_config.output_dir = custom_output_dir

    assert sample_config.base_dir == custom_output_dir
    assert sample_config.tensorboard_dir == custom_output_dir / "tensorboard"


def test_quantum_config():
    """Test quantum configuration"""
    quantum_config = QuantumConfig(
        q_layers=3,
        diff_method="parameter-shift",
        q_device="default.qubit",
        q_device_kwargs={"shots": 1000}
    )

    assert quantum_config.q_layers == 3
    assert quantum_config.diff_method == "parameter-shift"
    assert quantum_config.q_device_kwargs["shots"] == 1000


def test_config_serialization(sample_config):
    """Test configuration serialization"""
    config_dict = sample_config.to_dict()
    assert isinstance(config_dict, dict)
    assert config_dict['name'] == TEST_PARAMS['name']

    # Test deserialization
    restored_config = Config.from_dict(config_dict)
    assert restored_config.name == sample_config.name
    assert restored_config.model.name == sample_config.model.name


def test_config_manager_save_load(config_manager, sample_config):
    """Test ConfigManager save and load functionality"""
    with tempfile.NamedTemporaryFile(suffix='.yaml', delete=False) as tmp:
        # Save configuration
        config_manager.save_config(sample_config, tmp.name)

        # Load configuration
        loaded_config = config_manager.load_config(tmp.name)

        # Verify loaded configuration
        assert loaded_config.name == sample_config.name
        assert loaded_config.model.name == sample_config.model.name

        # Clean up
        Path(tmp.name).unlink()


def test_invalid_config_file(config_manager):
    """Test loading invalid configuration file"""
    with tempfile.NamedTemporaryFile(suffix='.yaml', delete=False) as tmp:
        # Write invalid YAML
        tmp.write(b"invalid: yaml: content: [")
        tmp.flush()

        with pytest.raises(ValueError, match="Invalid YAML format"):
            config_manager.load_config(tmp.name)

        # Clean up
        Path(tmp.name).unlink()


def test_file_not_found(config_manager):
    """Test loading non-existent file"""
    with pytest.raises(FileNotFoundError):
        config_manager.load_config("nonexistent_config.yaml")


def test_hybrid_model_config():
    """Test hybrid model configuration"""
    quantum_config = QuantumConfig()
    model_config = ModelConfig(
        name="HybridVGG",
        model_type="hybrid",
        model_kwargs={"num_classes": 10},
        quantum_config=quantum_config
    )

    assert model_config.model_type == "hybrid"
    assert model_config.quantum_config is not None
    assert model_config.quantum_config.q_layers == 2  # default value


@pytest.mark.parametrize("transform_list", [
    [{"name": "Normalize", "mean": 0.5, "std": 0.5}],
    [{"name": "RandomCrop", "size": 32}, {"name": "RandomHorizontalFlip"}],
    []
])
def test_data_config_transforms(transform_list):
    """Test data configuration with different transforms"""
    data_config = DataConfig(
        name="test_dataset",
        input_shape=(3, 32, 32),
        num_classes=10,
        dataset_type="CIFAR10",
        train_transforms=transform_list,
        val_transforms=transform_list,
        test_transforms=transform_list
    )

    assert data_config.train_transforms == transform_list
    assert data_config.val_transforms == transform_list
    assert data_config.test_transforms == transform_list


def test_config_serialization_types(sample_config):
    """Test serialization of different Python types"""
    # Some complex types to test
    sample_config.data.dataset_kwargs = {
        'tuple_value': (1, 2, 3),
        'nested_tuple': [(1, 2), (3, 4)],
        'path_value': Path('/some/path'),
        'mixed_list': [1, (2, 3), Path('/test')]
    }

    # Test serialization
    config_dict = sample_config.to_dict()

    # Verify types are converted correctly
    assert isinstance(config_dict['data']['input_shape'], list)
    assert isinstance(config_dict['data']['dataset_kwargs']['tuple_value'], list)
    assert isinstance(config_dict['data']['dataset_kwargs']['nested_tuple'][0], list)
    assert isinstance(config_dict['data']['dataset_kwargs']['path_value'], str)

    # Test full serialization cycle
    manager = ConfigManager()
    with tempfile.NamedTemporaryFile(suffix='.yaml', delete=False) as tmp:
        manager.save_config(sample_config, tmp.name)
        loaded_config = manager.load_config(tmp.name)

        # Verify paths are correct
        assert loaded_config.base_dir == sample_config.base_dir
        assert loaded_config.tensorboard_dir == sample_config.tensorboard_dir

        Path(tmp.name).unlink()


def test_directory_structure(sample_config):
    """Test the directory structure properties"""
    # Test with default output directory
    assert sample_config.base_dir == Path("./outputs")
    assert sample_config.tensorboard_dir == Path("./outputs/tensorboard")

    # Test with custom output directory
    custom_dir = Path("/custom/path")
    sample_config.output_dir = custom_dir
    assert sample_config.base_dir == custom_dir
    assert sample_config.tensorboard_dir == custom_dir / "tensorboard"


def test_path_resolution():
    """Test path resolution for different scenarios"""
    config = Config(
        name="test",
        version="1.0",
        description="Test config",
        data=DataConfig(
            name="test_dataset",
            input_shape=(3, 32, 32),
            num_classes=10,
            dataset_type="CIFAR10",
            dataset_path="datasets/custom"
        ),
        model=ModelConfig(
            name="test",
            model_type="classic",
            model_kwargs={}
        ),
        training=TrainingConfig(),
        output_dir=Path("./test_output")
    )

    assert config.base_dir == Path("./test_output")
    assert config.tensorboard_dir == Path("./test_output/tensorboard")


def test_config_validation(config_manager):
    """Test configuration validation"""
    # Test missing required fields
    invalid_config = {
        'name': 'test',
        'version': 'v1',
        'description': 'Test config',
        'data': {
            'name': 'test_dataset',
            'input_shape': (3, 32, 32),
            # Missing num_classes, dataset_type, batch_size
        },
        'model': {
            'name': 'test_model',
            'model_type': 'classic'
        },
        'training': {
            'learning_rate': 0.001,
            'weight_decay': 1e-4,
            'num_epochs': 10,
            'checkpoint_interval': 1
        }
    }

    with pytest.raises(ValueError, match="missing required fields"):
        config_manager._validate_config_dict(invalid_config)


def test_training_config():
    """Test training configuration"""
    training_config = TrainingConfig(
        learning_rate=0.001,
        weight_decay=1e-4,
        num_epochs=10,
        checkpoint_interval=1
    )

    assert training_config.learning_rate == 0.001
    assert training_config.num_epochs == 10
    assert training_config.checkpoint_interval == 1

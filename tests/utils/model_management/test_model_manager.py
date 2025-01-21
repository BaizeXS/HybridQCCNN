import json
import logging
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from config import Config, ModelConfig, DataConfig, TrainingConfig
from utils.model_management import ModelManager

# Test parameters
TEST_PARAMS = {
    'batch_size': 4,
    'input_shape': (1, 28, 28),
    'num_classes': 3,
    'dataset_size': 20
}

# Test model code
TEST_MODEL_CODE = """
import torch.nn as nn

class SimpleTestNet(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.conv = nn.Conv2d(1, 10, kernel_size=3)
        self.fc = nn.Linear(10 * 26 * 26, num_classes)
        
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
"""


@pytest.fixture
def test_model_path(tmp_path):
    """Create and return test model file path"""
    model_file = tmp_path / "test_model.py"
    model_file.write_text(TEST_MODEL_CODE)
    return model_file


@pytest.fixture
def test_config(test_model_path):
    """Return a test configuration"""
    return Config(
        name="test_model",
        version="v1",
        description="Test model",
        data=DataConfig(
            name="test_dataset",
            input_shape=(1, 28, 28),
            num_classes=TEST_PARAMS['num_classes'],
            dataset_type="MNIST",
            train_transforms=[],
            val_transforms=[],
            test_transforms=[]
        ),
        model=ModelConfig(
            name="SimpleTestNet",
            model_type="custom",
            model_kwargs={'num_classes': TEST_PARAMS['num_classes']},
            custom_model_path=test_model_path
        ),
        training=TrainingConfig(
            num_epochs=2,
            checkpoint_interval=1
        ),
        device="cuda" if torch.cuda.is_available() else "cpu",
        output_dir=Path("./test_outputs"),
        seed=42
    )


@pytest.fixture
def test_data():
    """Return test data"""
    x = torch.randn(TEST_PARAMS['dataset_size'], *TEST_PARAMS['input_shape'])
    y = torch.randint(0, TEST_PARAMS['num_classes'], (TEST_PARAMS['dataset_size'],))
    return torch.utils.data.TensorDataset(x, y)


@pytest.fixture
def test_loader(test_data):
    """Return test dataloader"""
    return torch.utils.data.DataLoader(test_data, batch_size=TEST_PARAMS['batch_size'])


@pytest.fixture
def model_manager(test_config):
    """Return a model manager instance"""
    return ModelManager(test_config, "test_model")


def test_model_initialization(model_manager):
    """Test model manager initialization"""
    assert isinstance(model_manager.model, nn.Module)
    assert model_manager.model_dir.exists()
    assert model_manager.checkpoint_dir.exists()
    assert model_manager.tensorboard_dir.exists()


def test_training(model_manager, test_loader):
    """Test model training"""
    model_manager.train(test_loader, val_loader=test_loader)

    # Check metrics
    assert len(model_manager.metrics['train']) > 0
    assert len(model_manager.metrics['val']) > 0
    assert 'loss' in model_manager.metrics['train']
    assert 'accuracy' in model_manager.metrics['train']


def test_checkpoint_saving_loading(model_manager, test_config):
    """Test checkpoint functionality"""
    # Save checkpoint
    model_manager.save_checkpoint(epoch=0)
    checkpoint_path = model_manager.checkpoint_dir / "checkpoint_epoch_0.pt"
    assert checkpoint_path.exists()

    # Create new manager and load checkpoint
    new_manager = ModelManager(test_config, "test_model_2")
    loaded_epoch = new_manager.load_checkpoint(checkpoint_path)
    assert loaded_epoch == 0


def test_prediction(model_manager):
    """Test model prediction"""
    inputs = torch.randn(5, *TEST_PARAMS['input_shape'])
    predictions = model_manager.predict(inputs)
    assert predictions.shape == (5,)
    assert torch.all(predictions >= 0) and torch.all(predictions < TEST_PARAMS['num_classes'])


def test_metrics_tracking(model_manager, test_loader):
    """Test metrics tracking"""
    model_manager.train(test_loader, val_loader=test_loader)

    # Check metrics structure
    assert all(phase in model_manager.metrics for phase in ['train', 'val'])
    assert all(metric in model_manager.metrics['train'] for metric in ['loss', 'accuracy'])

    # Check metrics values
    for phase in ['train', 'val']:
        for metric_values in model_manager.metrics[phase].values():
            assert len(metric_values) == model_manager.config.training.num_epochs


def test_model_evaluation(model_manager, test_loader):
    """Test model evaluation"""
    test_metrics = model_manager.test(test_loader)

    assert isinstance(test_metrics, dict)
    assert 'loss' in test_metrics
    assert 'accuracy' in test_metrics
    assert all(0 <= test_metrics[metric] <= 1 for metric in ['accuracy', 'precision', 'recall', 'f1'])


def test_metrics_saving_loading(model_manager, test_loader, test_config):
    """Test metrics saving and loading"""
    # Train model to generate some metrics
    model_manager.train(test_loader, val_loader=test_loader)

    # Save checkpoint with metrics
    epoch = 0
    model_manager.save_checkpoint(epoch=epoch)

    # Create new manager and load metrics
    new_manager = ModelManager(test_config, "test_model_2")

    # Load specific epoch metrics
    metrics_path = model_manager.metrics_dir / f"metrics_epoch_{epoch}.json"
    loaded_epoch = new_manager.load_metrics(epoch=0, metrics_path=metrics_path)
    assert loaded_epoch == 0
    assert len(new_manager.metrics['train']['loss']) == 1  # Only one epoch

    # Save complete history
    history_path = model_manager.metrics_dir / "metrics_history.json"
    history_metrics = {
        'metrics': model_manager.metrics,
        'conf_matrices': model_manager.conf_matrices,
        'epoch': model_manager.config.training.num_epochs - 1
    }
    with open(history_path, 'w') as f:
        json.dump(history_metrics, f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)

    # Load complete history
    loaded_epoch = new_manager.load_metrics(metrics_path=history_path)
    assert loaded_epoch == model_manager.config.training.num_epochs - 1
    assert len(new_manager.metrics['train']['loss']) == model_manager.config.training.num_epochs

    # Verify metrics were loaded correctly
    assert new_manager.metrics['train'].keys() == model_manager.metrics['train'].keys()
    for metric in model_manager.metrics['train'].keys():
        assert np.allclose(
            np.array(new_manager.metrics['train'][metric]),
            np.array(model_manager.metrics['train'][metric])
        )


def test_metrics_file_format(model_manager, test_loader):
    """Test metrics file format and content"""
    # Train model to generate some metrics
    model_manager.train(test_loader, val_loader=test_loader)

    # Save metrics
    epoch = 0
    model_manager.save_checkpoint(epoch=epoch)
    metrics_path = model_manager.metrics_dir / f"metrics_epoch_{epoch}.json"

    # Verify JSON format and content
    with open(metrics_path, 'r') as f:
        metrics_data = json.load(f)

    assert isinstance(metrics_data, dict)
    assert 'metrics' in metrics_data
    assert 'conf_matrices' in metrics_data
    assert 'epoch' in metrics_data
    assert metrics_data['epoch'] == epoch

    # Verify single epoch metrics format
    for phase in ['train', 'val']:
        assert phase in metrics_data['metrics']
        assert isinstance(metrics_data['metrics'][phase], dict)
        for metric_name, value in metrics_data['metrics'][phase].items():
            assert isinstance(value, (int, float)) or value is None

    # Verify confusion matrix format
    for phase in ['train', 'val']:
        assert phase in metrics_data['conf_matrices']
        matrix = metrics_data['conf_matrices'][phase]
        assert isinstance(matrix, list) or matrix is None
        if matrix is not None:
            assert all(isinstance(row, list) for row in matrix)


def test_num_classes_mismatch(test_config, caplog):
    """Test handling of mismatched num_classes between model and dataset."""
    # Modify model_kwargs to cause a mismatch
    test_config.model.model_kwargs['num_classes'] = TEST_PARAMS['num_classes'] + 1

    with caplog.at_level(logging.WARNING):
        manager = ModelManager(test_config, "test_model")

    # Verify warning message
    assert any(
        "Model num_classes" in record.message
        and "does not match dataset" in record.message
        for record in caplog.records
    )

    # Verify model uses dataset's num_classes
    assert manager.model.fc.out_features == TEST_PARAMS['num_classes']


@pytest.fixture(autouse=True)
def cleanup_test_dirs():
    """Clean up test directories"""
    yield
    test_dirs = ['./test_outputs', './runs']  # TensorBoard logs default to ./runs
    for d in test_dirs:
        if Path(d).exists():
            shutil.rmtree(d)

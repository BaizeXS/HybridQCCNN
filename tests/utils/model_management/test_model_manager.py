import pytest
import torch
import torch.nn as nn
from pathlib import Path
from utils.model_management import ModelManager
from config import Config, ModelConfig, DataConfig, TrainingConfig

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
        description="Test configuration",
        data=DataConfig(
            name="test_data",
            input_shape=TEST_PARAMS['input_shape'],
            num_classes=TEST_PARAMS['num_classes'],
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
        output_dir=Path("./test_outputs")
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

@pytest.fixture(autouse=True)
def cleanup_test_dirs():
    """Clean up test directories"""
    yield
    import shutil
    test_dirs = ['./test_outputs', './runs']  # TensorBoard logs default to ./runs
    for d in test_dirs:
        if Path(d).exists():
            shutil.rmtree(d) 
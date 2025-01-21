import pytest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from utils.training import Trainer, MetricsCalculator

# Test parameters
TEST_PARAMS = {
    'batch_size': 4,
    'input_dim': 5,
    'num_classes': 3,
    'dataset_size': 20
}


@pytest.fixture
def sample_model():
    """Return a simple test model"""
    return nn.Sequential(
        nn.Linear(TEST_PARAMS['input_dim'], TEST_PARAMS['num_classes'])
    )


@pytest.fixture
def sample_data():
    """Return test data"""
    x = torch.randn(TEST_PARAMS['dataset_size'], TEST_PARAMS['input_dim'])
    y = torch.randint(0, TEST_PARAMS['num_classes'], (TEST_PARAMS['dataset_size'],))
    return TensorDataset(x, y)


@pytest.fixture
def sample_dataloader(sample_data):
    """Return test dataloader"""
    return DataLoader(sample_data, batch_size=TEST_PARAMS['batch_size'])


@pytest.fixture
def trainer(sample_model):
    """Return a trainer instance"""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(sample_model.parameters(), lr=0.01)
    return Trainer(
        model=sample_model,
        criterion=criterion,
        optimizer=optimizer
    )


def test_trainer_initialization(trainer):
    """Test trainer initialization"""
    assert isinstance(trainer.model, nn.Module)
    assert isinstance(trainer.criterion, nn.Module)
    assert isinstance(trainer.optimizer, torch.optim.Optimizer)
    assert trainer.scheduler is None
    assert trainer.device == "cpu"


def test_train_epoch(trainer, sample_dataloader):
    """Test training for one epoch"""
    metrics, conf_matrix = trainer.train_epoch(sample_dataloader, epoch=1)

    # Check metrics
    assert isinstance(metrics, dict)
    assert 'loss' in metrics
    assert 'accuracy' in metrics
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'f1' in metrics

    # Check time statistics
    assert 'epoch_time' in metrics
    assert metrics['epoch_time'] > 0

    # Check confusion matrix
    assert conf_matrix.shape == (TEST_PARAMS['num_classes'], TEST_PARAMS['num_classes'])


def test_validation(trainer, sample_dataloader):
    """Test validation"""
    metrics, conf_matrix = trainer.validate(sample_dataloader)

    # Check basic metrics
    assert isinstance(metrics, dict)
    assert all(metric in metrics for metric in ['loss', 'accuracy', 'precision', 'recall', 'f1'])

    # Check time statistics
    assert 'phase_time' in metrics
    assert metrics['phase_time'] > 0

    # Check confusion matrix
    assert conf_matrix.shape == (TEST_PARAMS['num_classes'], TEST_PARAMS['num_classes'])


def test_metrics_calculator():
    """Test metrics calculation"""
    calculator = MetricsCalculator()

    # Test with 3-class classification problem
    outputs = torch.tensor([
        [0.1, 0.8, 0.1],
        [0.7, 0.2, 0.1],
        [0.1, 0.1, 0.8]
    ])
    targets = torch.tensor([1, 0, 2])
    loss = torch.tensor(0.5)

    metrics, conf_matrix = calculator.calculate(outputs, targets, loss)

    assert metrics['loss'] == 0.5
    assert 0 <= metrics['accuracy'] <= 1
    assert 0 <= metrics['precision'] <= 1
    assert 0 <= metrics['recall'] <= 1
    assert 0 <= metrics['f1'] <= 1
    assert conf_matrix.shape == (3, 3)  # Ensure it's a 3x3 confusion matrix


def test_training_with_scheduler(sample_model, sample_dataloader):
    """Test training with learning rate scheduler"""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(sample_model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    trainer = Trainer(
        model=sample_model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler
    )

    initial_lr = optimizer.param_groups[0]['lr']
    trainer.train_epoch(sample_dataloader, epoch=1)
    final_lr = optimizer.param_groups[0]['lr']

    assert final_lr == initial_lr * 0.1


@pytest.mark.parametrize("device", ["cpu"])
def test_device_handling(device, sample_model, sample_dataloader):
    """Test handling of different devices"""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    trainer = Trainer(
        model=sample_model,
        criterion=nn.CrossEntropyLoss(),
        optimizer=torch.optim.SGD(sample_model.parameters(), lr=0.01),
        device=device
    )

    # Check model and loss function devices
    assert next(trainer.model.parameters()).device.type == device
    # For loss function, we can check its running device by creating an example input
    dummy_input = torch.randn(2, TEST_PARAMS['num_classes']).to(device)
    dummy_target = torch.tensor([0, 1]).to(device)
    dummy_loss = trainer.criterion(dummy_input, dummy_target)
    assert dummy_loss.device.type == device

    metrics, _ = trainer.train_epoch(sample_dataloader, epoch=1)
    assert isinstance(metrics['loss'], float)


def test_googlenet_training(sample_dataloader):
    """Test training with GoogLeNet-style model"""

    class SimpleGoogLeNet(nn.Module):
        """Simple GoogLeNet mock for testing"""

        def __init__(self):
            super().__init__()
            self.main = nn.Linear(TEST_PARAMS['input_dim'], TEST_PARAMS['num_classes'])
            self.aux = nn.Linear(TEST_PARAMS['input_dim'], TEST_PARAMS['num_classes'])

        def forward(self, x):
            if self.training:
                return self.main(x), self.aux(x)
            return self.main(x)

    model = SimpleGoogLeNet()
    trainer = Trainer(
        model=model,
        criterion=nn.CrossEntropyLoss(),
        optimizer=torch.optim.SGD(model.parameters(), lr=0.01)
    )

    assert trainer.is_googlenet  # Check model type detection
    metrics, conf_matrix = trainer.train_epoch(sample_dataloader, epoch=1)
    assert isinstance(metrics, dict)
    assert 'loss' in metrics

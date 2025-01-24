import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from utils.training import MetricsCalculator, Trainer


def basic_training_example():
    """
    Basic training example
    """
    # Create a simple model
    model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2))

    # Create example data
    x = torch.randn(100, 10)
    y = torch.randint(0, 2, (100,))
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=16)

    # Initialize training components
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    # Add device conversion
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = criterion.to(device)

    # Initialize trainer
    trainer = Trainer(
        model=model, criterion=criterion, optimizer=optimizer, device=device  # type: ignore
    )

    # Train for one epoch
    metrics, conf_matrix = trainer.train_epoch(dataloader, epoch=1)
    print(f"Training metrics: {metrics}")
    print(f"Training time: {metrics['epoch_time']:.3f}s")
    print(f"Confusion matrix:\n{conf_matrix}")


def training_with_scheduler_example():
    """
    Example of training with learning rate scheduler
    """
    model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 3))

    # Create example data
    x = torch.randn(100, 10)
    y = torch.randint(0, 3, (100,))
    dataset = TensorDataset(x, y)
    train_loader = DataLoader(dataset, batch_size=16)
    val_loader = DataLoader(dataset, batch_size=32)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    trainer = Trainer(
        model=model, criterion=criterion, optimizer=optimizer, scheduler=scheduler  # type: ignore
    )

    # Train and validate
    print("Training metrics:")
    train_metrics, _ = trainer.train_epoch(train_loader, epoch=1)
    print(
        "Training - "
        f"Loss: {train_metrics['loss']:.4f}, Accuracy: {train_metrics['accuracy']:.4f}"
    )
    print(f"Training time: {train_metrics['epoch_time']:.3f}s")

    print("\nValidation metrics:")
    val_metrics, _ = trainer.validate(val_loader)
    print(
        f"Validation - Loss: {val_metrics['loss']:.4f}, Accuracy: {val_metrics['accuracy']:.4f}"
    )
    print(f"Validation time: {val_metrics['phase_time']:.3f}s")


def metrics_calculation_example():
    """
    Example of using MetricsCalculator
    """
    calculator = MetricsCalculator()

    # Create example predictions and targets
    outputs = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7], [0.9, 0.1]])
    targets = torch.tensor([1, 0, 1, 0])
    loss = torch.tensor(0.5)

    # Calculate metrics
    metrics, conf_matrix = calculator.calculate(outputs, targets, loss)

    print("Metrics:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")
    print(f"\nConfusion Matrix:\n{conf_matrix}")


if __name__ == "__main__":
    print("Basic training example:")
    basic_training_example()

    print("\nTraining with scheduler example:")
    training_with_scheduler_example()

    print("\nMetrics calculation example:")
    metrics_calculation_example()

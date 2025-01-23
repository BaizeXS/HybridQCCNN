from pathlib import Path

import numpy as np
import torch

from config import Config, DataConfig, ModelConfig, TrainingConfig
from utils.model_management import ModelManager


def basic_model_example():
    """
    Basic model management example
    """
    # Create a basic configuration
    data_config = DataConfig(
        name="MNIST",
        input_shape=(1, 28, 28),
        num_classes=10,
        dataset_type="MNIST",
        batch_size=32,
        train_split=0.8,
        num_workers=2,
        pin_memory=True,
        train_transforms=[
            {"name": "ToTensor"},
            {"name": "Normalize", "args": {"mean": [0.1307], "std": [0.3081]}},
        ],
        val_transforms=[
            {"name": "ToTensor"},
            {"name": "Normalize", "args": {"mean": [0.1307], "std": [0.3081]}},
        ],
        test_transforms=[
            {"name": "ToTensor"},
            {"name": "Normalize", "args": {"mean": [0.1307], "std": [0.3081]}},
        ],
    )

    model_config = ModelConfig(
        name="ClassicNet", model_type="classic", model_kwargs={"num_classes": 10}
    )

    training_config = TrainingConfig(
        learning_rate=0.001,
        num_epochs=10,
        scheduler_type="StepLR",
        scheduler_kwargs={"step_size": 30, "gamma": 0.1},
    )

    config = Config(
        name="mnist_classification",
        version="v1",
        description="Basic MNIST classification model",
        data=data_config,
        model=model_config,
        training=training_config,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # Create model manager
    manager = ModelManager(config, model_name="basic_demo")

    # Create example data
    train_data = torch.randn(100, 1, 28, 28)
    train_labels = torch.randint(0, 10, (100,))
    train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.data.batch_size
    )

    # Train model
    manager.train(train_loader, val_loader=None)

    print("\nTraining metrics (last 5 epochs):")
    for metric, values in manager.metrics["train"].items():
        # Format last 5 values, handle numpy arrays
        last_values = values[-5:]
        formatted_values = [
            float(v) if isinstance(v, (np.floating, np.integer)) else v
            for v in last_values
        ]
        print(f"{metric}: {formatted_values}...")

    print(f"\nModel directory: {manager.model_dir}")

    # Clean up
    manager.cleanup()


def custom_model_example():
    """
    Example with custom model
    """
    # Create custom model file
    custom_model_dir = Path("./templates")
    custom_model_dir.mkdir(exist_ok=True)

    model_file = custom_model_dir / "custom_net.py"
    model_code = """import torch.nn as nn

class CustomNet(nn.Module):
    def __init__(self, num_classes=10, hidden_dim=128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 5 * 5, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
"""
    model_file.write_text(model_code)

    # Create configuration for custom model
    config = Config(
        name="custom_model_demo",
        version="v1",
        description="Custom model demonstration",
        data=DataConfig(
            name="MNIST",
            input_shape=(1, 28, 28),
            num_classes=10,
            dataset_type="MNIST",
            batch_size=32,
            train_split=0.8,
            num_workers=2,
            pin_memory=True,
            train_transforms=[{"name": "ToTensor"}],
            val_transforms=[{"name": "ToTensor"}],
            test_transforms=[{"name": "ToTensor"}],
        ),
        model=ModelConfig(
            name="CustomNet",
            model_type="custom",
            model_kwargs={"num_classes": 10, "hidden_dim": 256},
            custom_model_path=model_file,
        ),
        training=TrainingConfig(learning_rate=0.001, num_epochs=5),
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # Create model manager with custom model
    manager = ModelManager(config, "custom_model")

    # Create example data
    train_data = torch.randn(100, 1, 28, 28)
    train_labels = torch.randint(0, 10, (100,))
    train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.data.batch_size
    )

    # Train and validate
    val_data = torch.randn(50, 1, 28, 28)
    val_labels = torch.randint(0, 10, (50,))
    val_dataset = torch.utils.data.TensorDataset(val_data, val_labels)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config.data.batch_size
    )

    # Train model
    manager.train(train_loader, val_loader=val_loader)

    print("\nTraining metrics (last epoch):")
    for phase in ["train", "val"]:
        print(f"\n{phase.capitalize()} phase:")
        for metric, values in manager.metrics[phase].items():
            # Format last value, handle numpy arrays
            value = (
                float(values[-1])
                if isinstance(values[-1], (np.floating, np.integer))
                else values[-1]
            )
            print(f"{metric}: {value:.4f}")

    # Clean up
    manager.cleanup()


def checkpoint_management_example():
    """
    Example of model checkpoint management
    """
    # Create basic configuration
    config = Config(
        name="checkpoint_demo",
        version="v1",
        description="Checkpoint management demonstration",
        data=DataConfig(
            name="MNIST",
            input_shape=(1, 28, 28),
            num_classes=10,
            dataset_type="MNIST",
            batch_size=32,
            train_split=0.8,
            num_workers=2,
            pin_memory=True,
            train_transforms=[{"name": "ToTensor"}],
            val_transforms=[{"name": "ToTensor"}],
            test_transforms=[{"name": "ToTensor"}],
        ),
        model=ModelConfig(
            name="ClassicNet", model_type="classic", model_kwargs={"num_classes": 10}
        ),
        training=TrainingConfig(checkpoint_interval=1),
    )

    # Create two model managers
    manager1 = ModelManager(config, "model1")
    manager2 = ModelManager(config, "model2")

    # Train manager1 for a bit
    train_data = torch.randn(100, 1, 28, 28)
    train_labels = torch.randint(0, 10, (100,))
    train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.data.batch_size
    )

    # Train both models
    manager1.train(train_loader, val_loader=None)
    print("\nModel 1 training completed")

    # Save and load checkpoint
    manager1.save_checkpoint(epoch=0)
    checkpoint_path = manager1.checkpoint_dir / "checkpoint_epoch_0.pt"
    print(f"\nSaved checkpoint to: {checkpoint_path}")

    # Load checkpoint to second model
    manager2.load_checkpoint(checkpoint_path)
    print("Successfully loaded checkpoint to second model")

    # Verify models have same weights
    print("\nVerifying model weights...")
    for (n1, p1), (_, p2) in zip(
        manager1.model.named_parameters(), manager2.model.named_parameters()
    ):
        if not torch.allclose(p1, p2):
            print(f"Warning: Parameters {n1} don't match!")
            break
    else:
        print("Verified: Both models have identical weights")

    # Clean up
    manager1.cleanup()
    manager2.cleanup()


if __name__ == "__main__":
    print("Basic model management example:")
    basic_model_example()

    print("\nCustom model example:")
    custom_model_example()

    print("\nCheckpoint management example:")
    checkpoint_management_example()

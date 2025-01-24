from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from config import Config, DataConfig, ModelConfig, TrainingConfig
from utils.model_management import ModelManager


# Helper function: create example data
def create_example_data(
    num_samples, batch_size, input_shape=(1, 28, 28), num_classes=10
):
    """Create example data"""
    data = torch.randn(num_samples, *input_shape)
    labels = torch.randint(0, num_classes, (num_samples,))
    dataset = TensorDataset(data, labels)
    return DataLoader(dataset, batch_size=batch_size)


# Helper function: create base config
def get_base_config(name="demo", model_type="classic", **kwargs):
    """Create base config"""
    model_name = (
        kwargs.get("model_name", "ClassicNet")
        if model_type == "custom"
        else "ClassicNet"
    )
    model_config_args = {
        "name": model_name,
        "model_type": model_type,
        "model_kwargs": {"num_classes": 10, **kwargs.get("model_kwargs", {})},
    }

    if model_type == "custom":
        model_config_args["custom_model_path"] = kwargs.get("custom_model_path")

    base_config = {
        "name": name,
        "version": "v1",
        "description": f"{name} demonstration",
        "data": DataConfig(
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
        "model": ModelConfig(**model_config_args),
        "training": TrainingConfig(
            learning_rate=0.001,
            num_epochs=kwargs.get("num_epochs", 5),
            **kwargs.get("training_kwargs", {}),
        ),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "output_dir": "./examples/custom_demo/outputs",
    }
    return Config(**base_config)


# Helper function: print training metrics
def print_metrics(manager, phases=None, last_n=5):
    """Print training metrics"""
    if phases is None:
        phases = ["train"]

    for phase in phases:
        if phase in manager.metrics:
            print(f"\n{phase.capitalize()} metrics (last {last_n} values):")
            for metric, values in manager.metrics[phase].items():
                last_values = [
                    float(v) if isinstance(v, (np.floating, np.integer)) else v
                    for v in values[-last_n:]
                ]
                print(f"{metric}: {last_values}")


def basic_model_example():
    """Basic model management example"""
    config = get_base_config("basic_model", num_epochs=10)
    manager = ModelManager(config, model_name="basic_demo")

    train_loader = create_example_data(100, config.data.batch_size)
    manager.train(train_loader, val_loader=None)

    print_metrics(manager)
    print(f"\nModel directory: {manager.model_dir}")
    manager.cleanup()


def custom_model_example():
    """Custom model example"""
    # Create custom model file
    custom_model_dir = Path("./examples/custom_demo")
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

    # Create config and train
    config = get_base_config(
        "custom_model",
        model_type="custom",
        model_name="CustomNet",
        model_kwargs={"hidden_dim": 256},
        custom_model_path=model_file,
    )

    manager = ModelManager(config, "custom_model")

    train_loader = create_example_data(100, config.data.batch_size)
    val_loader = create_example_data(50, config.data.batch_size)

    manager.train(train_loader, val_loader=val_loader)
    print_metrics(manager, phases=["train", "val"], last_n=1)
    manager.cleanup()


def checkpoint_management_example():
    """Checkpoint management example"""
    config = get_base_config(
        "checkpoint_demo", training_kwargs={"checkpoint_interval": 1}
    )

    # Create two model managers
    manager1 = ModelManager(config, "model1")
    manager2 = ModelManager(config, "model2")

    # Train first model
    train_loader = create_example_data(100, config.data.batch_size)
    manager1.train(train_loader, val_loader=None)
    print("\nModel 1 training completed")

    # Save and load checkpoint
    manager1.save_checkpoint(epoch=0)
    checkpoint_path = manager1.checkpoint_dir / "checkpoint_epoch_0.pt"
    print(f"\nSaved checkpoint to: {checkpoint_path}")

    manager2.load_checkpoint(checkpoint_path)
    print("Successfully loaded checkpoint to second model")

    # Verify model weights
    print("\nVerifying model weights...")
    weights_match = all(
        torch.allclose(p1, p2)
        for (_, p1), (_, p2) in zip(
            manager1.model.named_parameters(), manager2.model.named_parameters()
        )
    )
    print(
        "Verified: Both models have identical weights"
        if weights_match
        else "Warning: Models weights don't match!"
    )

    manager1.cleanup()
    manager2.cleanup()


if __name__ == "__main__":
    print("Basic model management example:")
    basic_model_example()

    print("\nCustom model example:")
    custom_model_example()

    print("\nCheckpoint management example:")
    checkpoint_management_example()

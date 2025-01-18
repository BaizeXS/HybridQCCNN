import torch
import torch.nn as nn
from pathlib import Path
from utils.model_management import ModelManager
from config import Config, ModelConfig, DataConfig, TrainingConfig

def basic_model_example():
    """
    Basic model management example showing standard model training workflow
    """
    # Create a basic configuration
    data_config = DataConfig(
        name="MNIST",
        input_shape=(1, 28, 28),
        num_classes=10,
        train_transforms=[{"name": "Normalize", "mean": 0.5, "std": 0.5}],
        val_transforms=[{"name": "Normalize", "mean": 0.5, "std": 0.5}],
        test_transforms=[{"name": "Normalize", "mean": 0.5, "std": 0.5}],
        dataset_kwargs={"root": "./datasets"}
    )
    
    model_config = ModelConfig(
        name="ClassicNet",
        model_type="classic",
        model_kwargs={"num_classes": 10}
    )
    
    training_config = TrainingConfig(
        batch_size=32,
        learning_rate=0.001,
        num_epochs=10,
        scheduler_type="StepLR",
        scheduler_kwargs={"step_size": 30, "gamma": 0.1}
    )
    
    config = Config(
        name="mnist_classification",
        version="v1",
        description="Basic MNIST classification model",
        data=data_config,
        model=model_config,
        training=training_config,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Create model manager
    manager = ModelManager(config, model_name="basic_demo")
    
    # Create example data
    train_data = torch.randn(100, 1, 28, 28)
    train_labels = torch.randint(0, 10, (100,))
    train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=config.training.batch_size
    )
    
    # Train model
    manager.train(train_loader)
    print(f"Training metrics: {manager.metrics['train']}")
    print(f"Model directory: {manager.model_dir}")

def custom_model_example():
    """
    Example of using a custom model with model manager
    """
    # Create custom model file
    custom_model_dir = Path("./custom_models")
    custom_model_dir.mkdir(exist_ok=True)
    
    model_file = custom_model_dir / "custom_net.py"
    model_code = """
import torch.nn as nn

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
            train_transforms=[{"name": "Normalize", "mean": 0.5, "std": 0.5}],
            val_transforms=[{"name": "Normalize", "mean": 0.5, "std": 0.5}],
            test_transforms=[{"name": "Normalize", "mean": 0.5, "std": 0.5}]
        ),
        model=ModelConfig(
            name="CustomNet",
            model_type="custom",
            model_kwargs={
                "num_classes": 10,
                "hidden_dim": 256
            },
            custom_model_path=model_file
        ),
        training=TrainingConfig(
            batch_size=32,
            learning_rate=0.001,
            num_epochs=5
        ),
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Create model manager with custom model
    manager = ModelManager(config, "custom_model")
    
    # Create example data
    train_data = torch.randn(100, 1, 28, 28)
    train_labels = torch.randint(0, 10, (100,))
    train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=config.training.batch_size
    )
    
    # Train and validate
    val_data = torch.randn(50, 1, 28, 28)
    val_labels = torch.randint(0, 10, (50,))
    val_dataset = torch.utils.data.TensorDataset(val_data, val_labels)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=config.training.batch_size
    )
    
    manager.train(train_loader, val_loader)
    
    # Print training results
    print("\nTraining metrics:")
    for metric, values in manager.metrics['train'].items():
        print(f"{metric}: {values}")
    
    print("\nValidation metrics:")
    for metric, values in manager.metrics['val'].items():
        print(f"{metric}: {values}")

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
            train_transforms=[{"name": "Normalize", "mean": 0.5, "std": 0.5}],
            val_transforms=[{"name": "Normalize", "mean": 0.5, "std": 0.5}],
            test_transforms=[{"name": "Normalize", "mean": 0.5, "std": 0.5}]
        ),
        model=ModelConfig(
            name="ClassicNet",
            model_type="classic",
            model_kwargs={"num_classes": 10}
        ),
        training=TrainingConfig(
            checkpoint_interval=1
        )
    )
    
    # Create two model managers
    manager1 = ModelManager(config, "model1")
    manager2 = ModelManager(config, "model2")
    
    # Train manager1 for a bit
    train_data = torch.randn(100, 1, 28, 28)
    train_labels = torch.randint(0, 10, (100,))
    train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=config.training.batch_size
    )
    
    manager1.train(train_loader)
    
    # Save checkpoint from manager1
    manager1.save_checkpoint(epoch=0)
    checkpoint_path = manager1.checkpoint_dir / "checkpoint_epoch_0.pt"
    print(f"Saved checkpoint to: {checkpoint_path}")
    
    # Load checkpoint to manager2
    manager2.load_checkpoint(checkpoint_path)
    print("Successfully loaded checkpoint to second model")
    
    # Verify both models have same weights
    for p1, p2 in zip(manager1.model.parameters(), manager2.model.parameters()):
        if not torch.allclose(p1, p2):
            print("Warning: Model parameters don't match!")
            break
    else:
        print("Verified: Both models have identical parameters")

if __name__ == "__main__":
    print("Basic model management example:")
    basic_model_example()
    
    print("\nCustom model example:")
    custom_model_example()
    
    print("\nCheckpoint management example:")
    checkpoint_management_example() 
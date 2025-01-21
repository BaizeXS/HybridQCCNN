from pathlib import Path

from config import ConfigManager, Config, ModelConfig, DataConfig, TrainingConfig, QuantumConfig


def basic_config_example():
    """
    Basic configuration example
    """
    # Create a basic configuration
    data_config = DataConfig(
        name="MNIST",
        input_shape=(1, 28, 28),
        num_classes=10,
        dataset_type="MNIST",
        batch_size=64,
        train_transforms=[{"name": "Normalize", "mean": 0.5, "std": 0.5}],
        val_transforms=[{"name": "Normalize", "mean": 0.5, "std": 0.5}],
        test_transforms=[{"name": "Normalize", "mean": 0.5, "std": 0.5}]
    )

    model_config = ModelConfig(
        name="SimpleVGG",
        model_type="classic",
        model_kwargs={"num_classes": 10}
    )

    training_config = TrainingConfig(
        learning_rate=0.001,
        num_epochs=10,
        checkpoint_interval=1
    )

    config = Config(
        name="mnist_classification",
        version="v1",
        description="Basic MNIST classification model",
        data=data_config,
        model=model_config,
        training=training_config
    )

    print(f"Configuration name: {config.name}")
    print(f"Base directory: {config.base_dir}")
    print(f"Tensorboard directory: {config.tensorboard_dir}")
    print(f"Configuration structure:\n{config.to_dict()}")


def hybrid_model_config_example():
    """
    Example of configuration for a hybrid quantum-classical model
    """
    # Create quantum configuration
    quantum_config = QuantumConfig(
        q_layers=2,
        diff_method="parameter-shift",
        q_device="default.qubit",
        q_device_kwargs={"shots": 1000}
    )

    # Create model configuration with quantum settings
    model_config = ModelConfig(
        name="HybridVGG",
        model_type="hybrid",
        model_kwargs={
            "num_classes": 10,
            "quantum_channels": 3
        },
        quantum_config=quantum_config
    )

    # Create complete configuration
    config = Config(
        name="hybrid_mnist",
        version="v1",
        description="Hybrid quantum-classical MNIST classification",
        data=DataConfig(
            name="MNIST",
            input_shape=(1, 28, 28),
            num_classes=10,
            dataset_type="MNIST",
            batch_size=32,
            train_transforms=[{"name": "Normalize", "mean": 0.5, "std": 0.5}],
            val_transforms=[{"name": "Normalize", "mean": 0.5, "std": 0.5}],
            test_transforms=[{"name": "Normalize", "mean": 0.5, "std": 0.5}]
        ),
        model=model_config,
        training=TrainingConfig(
            learning_rate=0.0005,
            num_epochs=5,
            checkpoint_interval=1
        ),
        device="cpu"  # Quantum simulations typically run on CPU
    )

    print(f"Quantum configuration:\n{config.model.quantum_config}")
    print(f"Model type: {config.model.model_type}")
    print(f"Device: {config.device}")
    print(f"Output directory structure:")
    print(f"  Base dir: {config.base_dir}")
    print(f"  Tensorboard dir: {config.tensorboard_dir}")


def config_manager_example():
    """
    Example of using ConfigManager for loading and saving
    """
    manager = ConfigManager()

    # Create a simple configuration
    config = Config(
        name="test_model",
        version="v1",
        description="Test configuration",
        data=DataConfig(
            name="CIFAR10",
            input_shape=(3, 32, 32),
            num_classes=10,
            dataset_type="CIFAR10",
            train_transforms=[],
            val_transforms=[],
            test_transforms=[]
        ),
        model=ModelConfig(
            name="SimpleResNet",
            model_type="classic",
            model_kwargs={"num_classes": 10}
        ),
        training=TrainingConfig()
    )

    # Save configuration
    save_path = Path("test_config.yaml")
    manager.save_config(config, save_path)
    print(f"Saved configuration to: {save_path}")

    # Load configuration
    loaded_config = manager.load_config(save_path)
    print(f"Loaded configuration name: {loaded_config.name}")
    print(f"Base directory: {loaded_config.base_dir}")
    print(f"Configurations match: {config.to_dict() == loaded_config.to_dict()}")

    # Clean up
    save_path.unlink()


if __name__ == "__main__":
    print("Basic configuration example:")
    basic_config_example()

    print("\nHybrid model configuration example:")
    hybrid_model_config_example()

    print("\nConfiguration manager example:")
    config_manager_example()

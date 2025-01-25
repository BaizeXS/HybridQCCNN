"""Configuration schema definitions.

This module defines the data structures for:
1. Data configuration - Dataset and data processing settings
2. Model configuration - Model architecture and parameters
3. Training configuration - Training process parameters
4. Quantum configuration - Quantum component settings

The configuration classes use dataclasses for clean and type-safe configuration management.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

VALID_DATASET_TYPES = {"MNIST", "FASHIONMNIST", "CIFAR10", "CIFAR100", "CUSTOM"}
VALID_MODEL_TYPES = {"classic", "hybrid", "custom"}
DEFAULT_DEVICE = "cpu"
DEFAULT_SEED = 42


@dataclass
class DataConfig:
    """Data configuration for dataset management.

    This class defines settings for:
    - Dataset selection and parameters
    - Data transformations for different phases
    - Input shape and class information

    Attributes:
        name (str): Name of the dataset, must be the same as the dataset class name
        input_shape (tuple): Input shape of the data
        num_classes (int): Number of classes in the dataset
        dataset_type (str): Type of the dataset ('CIFAR10', 'MNIST', 'custom' etc.)
        dataset_path (Optional[Path]): Path to dataset files
        custom_dataset_path (Optional[Path]): Path to custom dataset class implementation
        train_transforms (List[Dict]): Training data transformations
        val_transforms (List[Dict]): Validation data transformations
        test_transforms (List[Dict]): Test data transformations
        train_split (float): Train/val split ratio
        batch_size (int): Size of training batches
        num_workers (int): Number of workers for data loading
        pin_memory (bool): Whether to pin memory for data loading
        dataset_kwargs (Dict): Additional dataset parameters
    """

    name: str
    input_shape: tuple
    num_classes: int
    dataset_type: str
    dataset_path: Optional[Path] = None
    custom_dataset_path: Optional[Path] = None
    train_transforms: List[Dict[str, Any]] = field(
        default_factory=lambda: [
            {"name": "ToTensor"},
            {"name": "Normalize", "args": {"mean": [0.5], "std": [0.5]}},
        ]
    )
    val_transforms: List[Dict[str, Any]] = field(
        default_factory=lambda: [
            {"name": "ToTensor"},
            {"name": "Normalize", "args": {"mean": [0.5], "std": [0.5]}},
        ]
    )
    test_transforms: List[Dict[str, Any]] = field(
        default_factory=lambda: [
            {"name": "ToTensor"},
            {"name": "Normalize", "args": {"mean": [0.5], "std": [0.5]}},
        ]
    )
    train_split: float = 0.8
    batch_size: int = 64
    num_workers: int = 4
    pin_memory: bool = True
    dataset_kwargs: Optional[Dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Validate dataset_type
        if self.dataset_type not in VALID_DATASET_TYPES:
            raise ValueError(
                f"Invalid dataset_type: {self.dataset_type}. "
                f"Must be one of {VALID_DATASET_TYPES}"
            )

        # Validate CUSTOM dataset_type
        if self.dataset_type == "CUSTOM" and not self.custom_dataset_path:
            raise ValueError(
                "custom_dataset_path must be provided for CUSTOM dataset_type"
            )

        # Validate numeric parameters
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")

        if not 0 < self.train_split < 1:
            raise ValueError(
                f"train_split must be between 0 and 1, got {self.train_split}"
            )

        # Validate input_shape
        if (
            not isinstance(self.input_shape, (tuple, list))
            or len(self.input_shape) != 3
        ):
            raise ValueError(
                f"input_shape must be a 3-tuple (channels, height, width), "
                f"got {self.input_shape}"
            )

        # Validate num_classes
        if self.num_classes <= 1:
            raise ValueError(
                f"num_classes must be greater than 1, got {self.num_classes}"
            )


@dataclass
class QuantumConfig:
    """Quantum component configuration.

    This class defines settings for:
    - Quantum circuit parameters
    - Quantum device configuration
    - Differentiation methods

    Attributes:
        q_layers (int): Number of quantum parameter layers
        diff_method (str): Method for calculating quantum gradients
        q_device (str): Quantum device simulator name
        q_device_kwargs (Dict): Additional quantum device parameters
    """

    q_layers: int = 2
    diff_method: str = "best"
    q_device: str = "default.qubit"
    q_device_kwargs: Optional[Dict[str, Any]] = field(default_factory=dict)


@dataclass
class ModelConfig:
    """Model architecture configuration.

    This class defines settings for:
    - Model type and architecture
    - Model-specific parameters
    - Quantum components (if applicable)

    Attributes:
        name (str): Model name
        model_type (str): Model type ("classic", "hybrid", "custom")
        model_kwargs (Dict): Model parameters
        quantum_config (QuantumConfig): Quantum configuration
        custom_model_path (Path): Path to custom model code
    """

    name: str
    model_type: str = "classic"
    model_kwargs: Dict = field(default_factory=dict)
    quantum_config: Optional[QuantumConfig] = None
    custom_model_path: Optional[Union[str, Path]] = None

    def __post_init__(self):
        """Validate and process initialization parameters."""
        if self.model_type not in VALID_MODEL_TYPES:
            raise ValueError(f"Invalid model_type: {self.model_type}")
        if self.model_type == "custom" and not self.custom_model_path:
            raise ValueError("custom_model_path must be provided for custom models")
        if self.custom_model_path:
            self.custom_model_path = Path(self.custom_model_path)


@dataclass
class TrainingConfig:
    """Training process configuration.

    This class defines settings for:
    - Training hyperparameters
    - Optimization settings
    - Checkpointing and logging

    Attributes:
        learning_rate (float): Initial learning rate
        weight_decay (float): L2 regularization factor
        num_epochs (int): Number of training epochs
        checkpoint_interval (int): Epochs between checkpoints
        scheduler_type (str): Type of learning rate scheduler
        scheduler_kwargs (Dict): Scheduler parameters
    """

    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_epochs: int = 10
    checkpoint_interval: int = 5
    scheduler_type: str = "StepLR"
    scheduler_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {"step_size": 30, "gamma": 0.1}
    )

    def __post_init__(self):
        """Post-initialization processing"""
        self.learning_rate = float(self.learning_rate)
        self.weight_decay = float(self.weight_decay)
        self.num_epochs = int(self.num_epochs)
        self.checkpoint_interval = int(self.checkpoint_interval)


@dataclass
class Config:
    """Main configuration class.

    This class combines all configuration components:
    - Basic information
    - Data configuration
    - Model configuration
    - Training configuration
    - System settings

    Attributes:
        name (str): Configuration name
        version (str): Configuration version
        description (str): Configuration description
        data (DataConfig): Data configuration
        model (ModelConfig): Model configuration
        training (TrainingConfig): Training configuration
        device (str): Computing device
        seed (int): Random seed
        output_dir (Path): Output directory
    """

    name: str
    version: str
    description: str
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    device: str = DEFAULT_DEVICE
    seed: int = DEFAULT_SEED
    output_dir: Path = Path("./outputs")

    def __post_init__(self):
        """Post-initialization processing"""
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)

    @property
    def base_dir(self) -> Path:
        """Get base directory for outputs."""
        return self.output_dir

    @property
    def tensorboard_dir(self) -> Path:
        """Get tensorboard directory."""
        return self.base_dir / "tensorboard"

    def to_dict(self) -> dict:
        """Convert configuration to dictionary format."""

        def clean_data(obj):
            """Clean data for serialization."""
            if isinstance(obj, (tuple, set)):
                return list(obj)
            elif isinstance(obj, Path):
                return str(obj)
            elif isinstance(obj, dict):
                return {k: clean_data(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [clean_data(item) for item in obj]
            return obj

        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "data": clean_data(self.data.__dict__),
            "model": clean_data(self.model.__dict__),
            "training": clean_data(self.training.__dict__),
            "device": self.device,
            "seed": self.seed,
            "output_dir": str(self.output_dir),
        }

    @classmethod
    def from_dict(cls, config_dict: dict) -> "Config":
        """Create configuration from dictionary."""
        data_dict = config_dict["data"].copy()
        data_dict["input_shape"] = tuple(data_dict["input_shape"])
        data_config = DataConfig(**data_dict)

        quantum_config = None
        if config_dict["model"]["quantum_config"]:
            quantum_config = QuantumConfig(**config_dict["model"]["quantum_config"])

        model_config = ModelConfig(
            name=config_dict["model"]["name"],
            model_type=config_dict["model"]["model_type"],
            model_kwargs=config_dict["model"]["model_kwargs"],
            quantum_config=quantum_config,
            custom_model_path=config_dict["model"].get("custom_model_path"),
        )

        training_config = TrainingConfig(**config_dict["training"])

        return cls(
            name=config_dict["name"],
            version=config_dict["version"],
            description=config_dict["description"],
            data=data_config,
            model=model_config,
            training=training_config,
            device=config_dict.get("device", "cpu"),
            seed=config_dict.get("seed", 42),
            output_dir=Path(config_dict.get("output_dir", "./outputs")),
        )

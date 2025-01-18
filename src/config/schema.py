"""Definition of the configuration data structure."""
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from pathlib import Path

@dataclass
class DataConfig:
    """Data configuration.
    
    Attributes:
        name: Name of the dataset
        input_shape: Input shape of the data
        num_classes: Number of classes in the dataset
        train_transforms: List of transforms for training dataset
        val_transforms: List of transforms for validation dataset
        test_transforms: List of transforms for test dataset
        dataset_kwargs: Additional arguments for dataset initialization
    """
    name: str
    input_shape: tuple
    num_classes: int
    train_transforms: List[Dict[str, Any]]
    val_transforms: List[Dict[str, Any]]
    test_transforms: List[Dict[str, Any]]
    dataset_kwargs: Optional[Dict[str, Any]] = field(default_factory=dict)

@dataclass
class QuantumConfig:
    """Quantum configuration.
    
    Attributes:
        q_layers: Number of quantum parameter layers
        diff_method: Method for calculating quantum gradients
        q_device: Quantum device simulator name
        q_device_kwargs: Additional arguments for quantum device initialization
    """
    q_layers: int = 2
    diff_method: str = "best"
    q_device: str = "default.qubit"
    q_device_kwargs: Optional[Dict[str, Any]] = field(default_factory=dict)

@dataclass
class ModelConfig:
    """Model configuration.
    
    Attributes:
        name: Model name
        model_type: Type of model ("classic", "hybrid", or "custom")
        model_kwargs: Model initialization parameters
        quantum_config: Quantum configuration for hybrid models
        custom_model_path: Path to custom model implementation
    """
    name: str
    model_type: str
    model_kwargs: Dict[str, Any]
    quantum_config: Optional[QuantumConfig] = None
    custom_model_path: Optional[str] = None

@dataclass
class TrainingConfig:
    """Training configuration.
    
    Attributes:
        batch_size: Size of training batches
        learning_rate: Initial learning rate
        weight_decay: L2 regularization factor
        num_epochs: Number of training epochs
        checkpoint_interval: Epochs between checkpoints
        scheduler_type: Type of learning rate scheduler
        scheduler_kwargs: Parameters for the learning rate scheduler
    """
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_epochs: int = 10
    checkpoint_interval: int = 5
    scheduler_type: str = "StepLR"
    scheduler_kwargs: Optional[Dict[str, Any]] = field(
        default_factory=lambda: {"step_size": 30, "gamma": 0.1}
    )

@dataclass
class Config:
    """Main configuration class for model training.
    
    Attributes:
        name: Model configuration name
        version: Configuration version
        description: Detailed description of this configuration
        data: Data-related configuration
        model: Model-related configuration
        training: Training-related configuration
        device: Computing device ("cpu" or "cuda")
        seed: Random seed for reproducibility
        output_dir: Base directory for outputs
    """
    name: str
    version: str
    description: str
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    device: str = "cpu"
    seed: int = 42
    output_dir: Path = Path("./outputs")
    
    @property
    def base_dir(self) -> Path:
        """Return the base directory for outputs"""
        return self.output_dir
    
    @property
    def tensorboard_dir(self) -> Path:
        """Return the base directory for tensorboard"""
        return self.base_dir / "tensorboard"

    def to_dict(self) -> dict:
        """Convert configuration to dictionary format."""
        # Helper function to clean data types
        def clean_data(obj):
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
            'name': self.name,
            'version': self.version,
            'description': self.description,
            'data': {
                'name': self.data.name,
                'input_shape': list(self.data.input_shape),
                'num_classes': self.data.num_classes,
                'train_transforms': self.data.train_transforms,
                'val_transforms': self.data.val_transforms,
                'test_transforms': self.data.test_transforms,
                'dataset_kwargs': clean_data(self.data.dataset_kwargs)  # Clean dataset_kwargs
            },
            'model': {
                'name': self.model.name,
                'model_type': self.model.model_type,
                'model_kwargs': clean_data(self.model.model_kwargs),  # Clean model_kwargs too
                'quantum_config': self.model.quantum_config.__dict__ if self.model.quantum_config else None,
                'custom_model_path': self.model.custom_model_path
            },
            'training': clean_data(self.training.__dict__),  # Clean training config
            'device': self.device,
            'seed': self.seed,
            'output_dir': str(self.output_dir)
        }
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'Config':
        """Create a configuration instance from a dictionary."""
        data_dict = config_dict['data'].copy()
        data_dict['input_shape'] = tuple(data_dict['input_shape'])  # Convert list back to tuple
        data_config = DataConfig(**data_dict)
        
        quantum_config = None
        if config_dict['model']['quantum_config']:
            quantum_config = QuantumConfig(**config_dict['model']['quantum_config'])
        
        model_config = ModelConfig(
            name=config_dict['model']['name'],
            model_type=config_dict['model']['model_type'],
            model_kwargs=config_dict['model']['model_kwargs'],
            quantum_config=quantum_config,
            custom_model_path=config_dict['model'].get('custom_model_path')
        )
        
        training_config = TrainingConfig(**config_dict['training'])
        
        return cls(
            name=config_dict['name'],
            version=config_dict['version'],
            description=config_dict['description'],
            data=data_config,
            model=model_config,
            training=training_config,
            device=config_dict.get('device', 'cpu'),
            seed=config_dict.get('seed', 42),
            output_dir=Path(config_dict.get('output_dir', './outputs'))
        ) 
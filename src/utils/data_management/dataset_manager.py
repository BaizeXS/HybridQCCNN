"""Data management tools for deep learning datasets.

This module provides a comprehensive framework for dataset management, including:
1. Dataset loading and preprocessing
2. Custom dataset implementation support
3. Data transformation pipeline
4. Training/validation/test data splitting
5. DataLoader configuration and optimization

The module consists of two main classes:
- CustomDataset: Base class for implementing custom datasets
- DatasetManager: Central manager for all dataset operations

Typical usage:
    config = DataConfig(...)
    manager = DatasetManager(config, data_dir='path/to/data')
    train_loader, val_loader, test_loader = manager.get_data_loaders()
"""

from pathlib import Path
from typing import Tuple, Optional, List, Dict
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms
import logging
import importlib
import sys

from config import DataConfig

class CustomDataset(Dataset):
    """Base class for custom dataset implementations.
    
    This class provides a template for creating custom datasets compatible with
    PyTorch's Dataset interface. Users should inherit from this class and
    implement the _load_data method according to their specific data format.
    
    The class handles:
    - Basic dataset initialization
    - Data loading interface
    - Data transformation pipeline
    - Standard dataset operations (__len__, __getitem__)
    
    Attributes:
        data_dir (Path): Root directory containing the dataset files
        transform: Transformation pipeline to be applied to data samples
        train (bool): Whether this dataset is for training or testing
        data (list): List containing loaded data samples
        targets (list): List containing corresponding labels/targets
    
    Example:
        class ImageDataset(CustomDataset):
            def _load_data(self):
                # Implementation for loading image data
                image_files = list(self.data_dir.glob('*.jpg'))
                self.data = image_files
                self.targets = [get_label(f) for f in image_files]
    """
    
    def __init__(self, data_dir: Path, transform=None, train: bool = True):
        """Initialize the dataset.
        
        Args:
            data_dir (Path): Root directory of the dataset
            transform: Optional transform to be applied to data samples
            train (bool): Whether to load training or test data
        """
        self.data_dir = data_dir
        self.transform = transform
        self.train = train
        self.data = []
        self.targets = []
        
        self._load_data()
        
    def _load_data(self):
        """Load data samples and targets.
        
        This method should be implemented by subclasses to load data according
        to their specific format and structure.
        
        Raises:
            NotImplementedError: If the subclass doesn't implement this method
        """
        raise NotImplementedError("Subclass must implement _load_data method")
        
    def __len__(self):
        """Get the total number of samples in the dataset."""
        return len(self.data)
        
    def __getitem__(self, idx):
        """Get a single data sample and its corresponding target.
        
        Args:
            idx (int): Index of the sample to retrieve
            
        Returns:
            tuple: (transformed_data, target) pair
        """
        data = self.data[idx]
        target = self.targets[idx]
        
        if self.transform:
            data = self.transform(data)
            
        return data, target

class DatasetManager:
    """Dataset manager: responsible for dataset loading, preprocessing, and management.
    
    This class provides a unified interface for handling all dataset-related
    operations, including:
    - Dataset loading and validation
    - Data transformation pipeline setup
    - Training/validation set splitting
    - DataLoader configuration and creation
    
    The manager supports both built-in datasets (CIFAR10, MNIST) and custom
    datasets through the CustomDataset interface.
    
    Attributes:
        config (DataConfig): Configuration object containing dataset parameters
        data_dir (Path): Root directory for dataset storage
        logger (logging.Logger): Logger for tracking dataset operations
    
    Example:
        config = DataConfig(
            dataset_type='CIFAR10',
            batch_size=32,
            train_split=0.8
        )
        manager = DatasetManager(config, data_dir='./data')
        train_loader, val_loader, test_loader = manager.get_data_loaders()
    """
    
    def __init__(self, config: DataConfig, data_dir: Path):
        self.config = config
        self.data_dir = Path(data_dir)
        self.logger = logging.getLogger("dataset_manager")
        
        # Validate configuration
        self._validate_config()
        
    def _validate_config(self):
        """Validate dataset configuration.
        
        Raises:
            ValueError: If dataset configuration is invalid.
        """
        if self.config.dataset_type == "custom" and not self.config.dataset_path:
            raise ValueError("Custom dataset must provide dataset_path")
            
    def get_data_loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Get training, validation, and test data loaders.
        
        Returns:
            Tuple[DataLoader, DataLoader, DataLoader]: Training, validation, and test data loaders
        """
        # Get data transformations
        train_transform = self._get_transforms(self.config.train_transforms)
        val_transform = self._get_transforms(self.config.val_transforms)
        test_transform = self._get_transforms(self.config.test_transforms)
        
        # Load dataset
        train_dataset = self._get_dataset(train_transform, train=True)
        test_dataset = self._get_dataset(test_transform, train=False)
        
        # Split training and validation datasets
        train_dataset, val_dataset = self._split_dataset(
            train_dataset, 
            self.config.train_split
        )
        
        # Set validation dataset transformation
        val_dataset.dataset.transform = val_transform
        
        # Create data loaders
        train_loader = self._create_loader(
            train_dataset, 
            shuffle=True
        )
        val_loader = self._create_loader(
            val_dataset,
            shuffle=False
        )
        test_loader = self._create_loader(
            test_dataset,
            shuffle=False
        )
        
        return train_loader, val_loader, test_loader

    def _get_transforms(self, transform_configs: List[Dict]) -> transforms.Compose:
        """Create data transformations.
        
        Args:
            transform_configs (List[Dict]): List of transformation configurations
            
        Returns:
            transforms.Compose: Composed transformation
        """
        transform_list = []
        for t_config in transform_configs:
            transform_name = t_config['name']
            transform_args = t_config.get('args', {})
            
            if hasattr(transforms, transform_name):
                transform = getattr(transforms, transform_name)(**transform_args)
                transform_list.append(transform)
            else:
                raise ValueError(f"Unknown transformation method: {transform_name}")
                
        return transforms.Compose(transform_list)
      
    def _get_dataset(self, transform, train: bool = True) -> Dataset:
        """Load dataset based on configuration.
        
        Args:
            transform: Data transformation
            train (bool): Whether to load training dataset
            
        Returns:
            Dataset: Loaded dataset
            
        Raises:
            ValueError: If dataset type is unknown or custom dataset configuration is invalid
        """
        if self.config.dataset_type.upper() == 'CIFAR10':
            return datasets.CIFAR10(
                root=self.data_dir, 
                train=train,
                download=True,
                transform=transform
            )
        elif self.config.dataset_type.upper() == 'MNIST':
            return datasets.MNIST(
                root=self.data_dir,
                train=train,
                download=True,
                transform=transform
            )
        elif self.config.dataset_type.upper() == 'CUSTOM':
            # Load custom dataset class
            if not self.config.custom_dataset_path:
                raise ValueError("custom_dataset_path must be provided for custom dataset")
            
            # Import custom dataset module
            custom_dataset_path = Path(self.config.custom_dataset_path)
            spec = importlib.util.spec_from_file_location(
                custom_dataset_path.stem, custom_dataset_path
            )
            module = importlib.util.module_from_spec(spec)
            sys.modules[custom_dataset_path.stem] = module
            spec.loader.exec_module(module)
            
            # Get dataset class
            dataset_class = getattr(module, self.config.name)
            
            # Create dataset instance with additional kwargs
            return dataset_class(
                data_dir=self.data_dir,
                transform=transform,
                train=train,
                **self.config.dataset_kwargs
            )
        else:
            raise ValueError(f"Unknown dataset type: {self.config.dataset_type}")

    def _split_dataset(
        self, 
        dataset: Dataset,
        split_ratio: float
    ) -> Tuple[Dataset, Dataset]:
        """Split dataset into training and validation sets.
        
        Args:
            dataset (Dataset): Dataset to split
            split_ratio (float): Train/val split ratio
            
        Returns:
            Tuple[Dataset, Dataset]: Training and validation datasets
        """
        train_size = int(split_ratio * len(dataset))
        val_size = len(dataset) - train_size
        
        return random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
    def _create_loader(
        self,
        dataset: Dataset,
        shuffle: bool = False
    ) -> DataLoader:
        """Create data loader.
        
        Args:
            dataset (Dataset): Dataset to load
            shuffle (bool): Whether to shuffle data
            
        Returns:
            DataLoader: Data loader
        """
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory
        )
"""Configuration manager for model training.

This module provides configuration management functionality including:
- Loading configuration from YAML files
- Saving configuration to YAML files
- Converting between dictionary and Config objects
"""

from pathlib import Path
from typing import Union, Dict

import yaml

from .schema import Config


class ConfigManager:
    """Configuration manager: responsible for loading and saving configuration.
    
    This class handles configuration file operations including:
    - Loading configuration from YAML files
    - Saving configuration to YAML files
    - Validating configuration format
    
    Attributes:
        config_dir (Path): Base directory for configuration files.
    """

    # Define required fields for each configuration section
    REQUIRED_FIELDS = {
        'data': {
            'name', 'input_shape', 'num_classes',
            'dataset_type', 'batch_size'
        },
        'model': {
            'name', 'model_type'
        },
        'training': {
            'learning_rate', 'weight_decay', 'num_epochs'
        },
        'root': {
            'name', 'version', 'description',
            'data', 'model', 'training'
        }
    }

    def __init__(self, config_dir: str = "configs"):
        """Initialize configuration manager.
        
        Args:
            config_dir (str): Base directory for configuration files.
        """
        self.config_dir = Path(config_dir)

    def _validate_config_dict(self, config_dict: Dict, section: str = 'root') -> None:
        """Validate required fields in the configuration dictionary.
        
        Args:
            config_dict (Dict): Configuration dictionary to validate.
            section (str): Configuration section name ('root', 'data', 'model', 'training').
            
        Raises:
            ValueError: If required fields are missing.
        """
        required_fields = self.REQUIRED_FIELDS[section]
        missing_fields = required_fields - set(config_dict.keys())

        if missing_fields:
            raise ValueError(
                f"Configuration section '{section}' missing required fields: {missing_fields}"
            )

        # Recursively validate sub-configurations
        if section == 'root':
            for subsection in ['data', 'model', 'training']:
                if subsection in config_dict:
                    self._validate_config_dict(config_dict[subsection], subsection)

    def load_config(self, config_path: Union[str, Path]) -> Config:
        """Load configuration from a file.
        
        Args:
            config_path (Union[str, Path]): Path to the configuration file.
            
        Returns:
            Config: Loaded configuration object.
            
        Raises:
            FileNotFoundError: If config file doesn't exist.
            ValueError: If config file has invalid YAML format.
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path) as f:
            try:
                config_dict = yaml.safe_load(f)

                # Handle path type fields
                if config_dict['data'].get('dataset_path'):
                    config_dict['data']['dataset_path'] = Path(config_dict['data']['dataset_path'])

                # Validate configuration dictionary
                self._validate_config_dict(config_dict)

                return Config.from_dict(config_dict)
            except yaml.YAMLError as e:
                raise ValueError(f"Invalid YAML format: {e}")

    def save_config(self, config: Config, save_path: Union[str, Path]):
        """Save configuration to a file.
        
        Args:
            config (Config): Configuration object to save.
            save_path (Union[str, Path]): Path where to save the configuration.
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Validate configuration before saving
        config_dict = config.to_dict()
        self._validate_config_dict(config_dict)

        with open(save_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)

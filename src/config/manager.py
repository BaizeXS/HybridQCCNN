"""Configuration manager"""
from pathlib import Path
from typing import Union
import yaml

from .schema import Config

class ConfigManager:
    """Configuration manager: responsible for loading and saving configuration for a single model"""
    
    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
    
    def load_config(self, config_path: Union[str, Path]) -> Config:
        """Load configuration from a file"""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path) as f:
            try:
                config_dict = yaml.safe_load(f)
                return Config.from_dict(config_dict)
            except yaml.YAMLError as e:
                raise ValueError(f"Invalid YAML format: {e}")
    
    def save_config(self, config: Config, save_path: Union[str, Path]):
        """Save configuration to a file"""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as f:
            yaml.dump(config.to_dict(), f, default_flow_style=False)
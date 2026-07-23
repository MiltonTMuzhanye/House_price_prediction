import yaml
from pathlib import Path
from typing import Dict, Any
from .logger import logger

class ConfigManager:
    """Manages configuration loading and access"""
    
    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
        self.configs = {}
        self._load_all_configs()
    
    def _load_all_configs(self):
        """Load all YAML config files from config directory"""
        for config_file in self.config_dir.glob("*.yaml"):
            with open(config_file, 'r') as f:
                self.configs[config_file.stem] = yaml.safe_load(f)
    
    def get(self, key: str, default=None) -> Any:
        """Get configuration value using dot notation"""
        keys = key.split('.')
        value = self.configs
        
        try:
            for k in keys:
                if isinstance(value, dict):
                    value = value.get(k)
                else:
                    return default
            return value
        except (KeyError, AttributeError):
            return default
    
    def get_config(self, name: str) -> Dict:
        """Get entire config section"""
        return self.configs.get(name, {})

config_manager = ConfigManager()
config = config_manager.get_config
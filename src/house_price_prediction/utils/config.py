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


class ConfigProxy:
    """Proxy that makes config_manager accessible like a dictionary"""
    
    def __init__(self, manager: ConfigManager):
        self._manager = manager
    
    def get(self, key: str, default=None) -> Any:
        """Get config value - dict-like interface"""
        return self._manager.get(key, default)
    
    def __getitem__(self, key: str) -> Any:
        """Allow config['key'] access"""
        return self._manager.get(key)
    
    def __contains__(self, key: str) -> bool:
        """Allow 'key' in config checks"""
        return self._manager.get(key) is not None
    
    def __call__(self, key: str, default=None) -> Any:
        """Allow config('key') function call (backward compatibility)"""
        return self._manager.get(key, default)


config = ConfigProxy(config_manager)

__all__ = ['ConfigManager', 'config_manager', 'config']
import yaml
from pathlib import Path
from typing import Dict, Any
from .logger import logger


class ConfigManager:
    """Manages configuration loading and access."""

    def __init__(self, config_dir: str = "configs"):
        # Resolve project root reliably
        project_root = Path(__file__).resolve().parents[3]

        self.config_dir = project_root / config_dir
        self.configs: Dict[str, Any] = {}

        self._load_all_configs()

    def _load_all_configs(self):
        """Load all YAML configuration files."""

        if not self.config_dir.exists():
            raise FileNotFoundError(
                f"Configuration directory not found: {self.config_dir}"
            )

        for config_file in self.config_dir.glob("*.yaml"):

            with open(config_file, "r") as f:
                config_data = yaml.safe_load(f) or {}

            self.configs[config_file.stem] = config_data

            logger.debug(
                f"Loaded configuration: {config_file.name}"
            )

    def get(self, key: str, default=None) -> Any:
        """
        Get configuration value using dot notation.

        Examples:
            config.get("data.raw_path")
            config.get("model.random_forest_params")
            config.get("training.cv_folds")
        """

        keys = key.split(".")

        # First key identifies the configuration file
        config_name = keys[0]

        if config_name not in self.configs:
            return default

        value = self.configs[config_name]

        # Traverse remaining keys
        for k in keys[1:]:

            if not isinstance(value, dict):
                return default

            value = value.get(k)

            if value is None:
                return default

        return value

    def get_config(self, name: str) -> Dict:
        """Get an entire configuration file."""

        return self.configs.get(name, {})


class ConfigProxy:
    """Proxy that makes config_manager accessible like a dictionary."""

    def __init__(self, manager: ConfigManager):
        self._manager = manager

    def get(self, key: str, default=None) -> Any:
        return self._manager.get(key, default)

    def __getitem__(self, key: str) -> Any:
        return self._manager.get(key)

    def __contains__(self, key: str) -> bool:
        return self._manager.get(key) is not None

    def __call__(self, key: str, default=None) -> Any:
        return self._manager.get(key, default)


config_manager = ConfigManager()
config = ConfigProxy(config_manager)

__all__ = ["ConfigManager", "config_manager", "config"]
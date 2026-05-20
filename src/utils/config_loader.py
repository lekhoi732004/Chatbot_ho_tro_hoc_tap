"""
Config loader utility for edu-agent.
Loads and validates config.json, supports dot-notation access.
"""

import json
import os
from pathlib import Path
from typing import Any, Optional


class ConfigLoader:
    """Singleton config loader with dot-notation access."""

    _instance: Optional["ConfigLoader"] = None
    _config: dict = {}

    def __new__(cls, config_path: Optional[str] = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._loaded = False
        return cls._instance

    def __init__(self, config_path: Optional[str] = None):
        if self._loaded:
            return
        if config_path is None:
            # Auto-resolve from project root
            base = Path(__file__).resolve().parents[2]
            config_path = base / "config" / "config.json"
        self._load(str(config_path))
        self._loaded = True

    def _load(self, path: str) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            self._config = json.load(f)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Dot-notation access.
        Ví dụ: config.get("llm.executor.temperature")
        """
        keys = key.split(".")
        val = self._config
        for k in keys:
            if isinstance(val, dict) and k in val:
                val = val[k]
            else:
                return default
        return val

    def get_section(self, section: str) -> dict:
        return self._config.get(section, {})

    @property
    def raw(self) -> dict:
        return self._config

    def reload(self, config_path: Optional[str] = None) -> None:
        self._loaded = False
        self.__init__(config_path)


# Module-level singleton accessor
def get_config(config_path: Optional[str] = None) -> ConfigLoader:
    return ConfigLoader(config_path)

import os
import yaml
from pathlib import Path
from typing import Any, Dict

def _deep_merge(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge two dictionaries."""
    result = dict1.copy()
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result

def load_config(config_path: str, base_config_path: str = "configs/base.yaml") -> Dict[str, Any]:
    """Load and merge base configuration with dataset/experiment specific configs."""
    base_cfg: Dict[str, Any] = {}
    if base_config_path and Path(base_config_path).exists():
        with open(base_config_path, "r", encoding="utf-8") as f:
            base_cfg = yaml.safe_load(f) or {}
            
    cfg: Dict[str, Any] = {}
    if config_path and Path(config_path).exists():
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
            
    merged = _deep_merge(base_cfg, cfg)
    return merged

"""Configuration loading and management."""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file (defaults to configs/default.yaml)
        
    Returns:
        Configuration dictionary
    """
    if config_path is None:
        # Try to find default config
        possible_paths = [
            Path("configs/default.yaml"),
            Path(__file__).parent.parent.parent / "configs" / "default.yaml"
        ]
        
        for path in possible_paths:
            if path.exists():
                config_path = str(path)
                break
        else:
            return get_default_config()
    
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override with environment variables
    config = override_with_env(config)
    
    return config


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        config_path: Output path
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def override_with_env(config: Dict[str, Any]) -> Dict[str, Any]:
    """Override config with environment variables."""
    env_mappings = {
        "OPENAI_API_KEY": ["api_keys", "openai"],
        "ANTHROPIC_API_KEY": ["api_keys", "anthropic"],
        "DEEPSEEK_API_KEY": ["api_keys", "deepseek"]
    }
    
    for env_var, path in env_mappings.items():
        value = os.getenv(env_var)
        if value:
            current = config
            for key in path[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]
            current[path[-1]] = value
    
    return config


def get_default_config() -> Dict[str, Any]:
    """Get default configuration."""
    return {
        "translation": {
            "default_backend": "openai",
            "default_model": "gpt-4o",
            "temperature": 0.3,
            "num_candidates": 1,
            "enable_caching": True
        },
        "masking": {
            "enabled": True,
            "patterns": ["latex", "code", "url", "email", "doi"]
        },
        "context": {
            "enabled": True,
            "window_size": 3
        },
        "reranking": {
            "enabled": False,
            "strategy": "weighted"
        },
        "layout": {
            "preserve": True,
            "use_yolo": False
        },
        "api_keys": {
            "openai": os.getenv("OPENAI_API_KEY", ""),
            "anthropic": os.getenv("ANTHROPIC_API_KEY", ""),
            "deepseek": os.getenv("DEEPSEEK_API_KEY", "")
        }
    }

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Configuration loader for the DECOHERE pipeline.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional


def load_config(config_path: str = 'config.yaml') -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dict[str, Any]: Configuration dictionary
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def deep_merge(base_dict: Dict[str, Any], override_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dictionaries, with override_dict taking precedence.
    
    Args:
        base_dict: Base configuration dictionary
        override_dict: Override configuration dictionary
        
    Returns:
        Dictionary containing the merged configuration
    """
    merged = base_dict.copy()
    
    for key, value in override_dict.items():
        if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
            # Recursively merge nested dictionaries
            merged[key] = deep_merge(merged[key], value)
        else:
            # Override or add the value
            merged[key] = value
    
    return merged


def load_mode_config(mode: str, base_config_path: str = 'config/config.yaml') -> Dict[str, Any]:
    """
    Load mode-specific configuration and merge it with the base configuration.
    
    Args:
        mode: Mode to load configuration for ('day', 'week', or 'year')
        base_config_path: Path to the base configuration file
        
    Returns:
        Dictionary containing the merged configuration
    """
    # Load base configuration
    base_config = load_config(base_config_path)
    
    # Get mode-specific configuration path
    mode_config_path = base_config['modes'][mode]['config_file']
    
    # Load mode-specific configuration
    mode_config = load_config(mode_config_path)
    
    # Deep merge configurations (mode config takes precedence)
    merged_config = deep_merge(base_config, mode_config)
    
    return merged_config


def get_config_value(config: Dict[str, Any], key_path: str, default: Optional[Any] = None) -> Any:
    """
    Get a value from a nested configuration dictionary using a dot-separated key path.
    
    Args:
        config: Configuration dictionary
        key_path: Dot-separated path to the key (e.g., 'data.raw_data')
        default: Default value to return if the key is not found
        
    Returns:
        Value at the specified key path, or the default value if not found
    """
    keys = key_path.split('.')
    value = config
    
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    
    return value


def update_config_value(config: Dict[str, Any], key_path: str, value: Any) -> Dict[str, Any]:
    """
    Update a value in a nested configuration dictionary using a dot-separated key path.
    
    Args:
        config: Configuration dictionary
        key_path: Dot-separated path to the key (e.g., 'data.raw_data')
        value: New value to set
        
    Returns:
        Updated configuration dictionary
    """
    keys = key_path.split('.')
    current = config
    
    # Navigate to the parent of the target key
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    
    # Set the value
    current[keys[-1]] = value
    
    return config


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """
    Save configuration to a YAML file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save the configuration file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    # Save configuration
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False) 
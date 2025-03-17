#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Configuration utilities for loading and merging configuration files.
"""

import os
import yaml
from typing import Dict, Any, Optional


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary containing the configuration
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configuration dictionaries, with override_config taking precedence.
    
    Args:
        base_config: Base configuration dictionary
        override_config: Override configuration dictionary
        
    Returns:
        Merged configuration dictionary
    """
    merged_config = base_config.copy()
    
    for key, value in override_config.items():
        if isinstance(value, dict) and key in merged_config and isinstance(merged_config[key], dict):
            # Recursively merge nested dictionaries
            merged_config[key] = merge_configs(merged_config[key], value)
        else:
            # Override or add the value
            merged_config[key] = value
    
    return merged_config


def get_mode_config(base_config_path: str = 'config/config.yaml', 
                   mode: str = 'day') -> Dict[str, Any]:
    """
    Get the configuration for a specific mode by merging the base config with the mode-specific config.
    
    Args:
        base_config_path: Path to the base configuration file
        mode: Mode to get configuration for ('day', 'week', or 'year')
        
    Returns:
        Merged configuration dictionary for the specified mode
    """
    # Load base configuration
    base_config = load_config(base_config_path)
    
    # Get mode-specific configuration path
    if mode not in base_config.get('modes', {}):
        raise ValueError(f"Invalid mode: {mode}. Available modes: {list(base_config.get('modes', {}).keys())}")
    
    mode_config_path = base_config['modes'][mode]['config_file']
    
    # Load mode-specific configuration
    mode_config = load_config(mode_config_path)
    
    # Merge configurations
    merged_config = merge_configs(base_config, mode_config)
    
    return merged_config 
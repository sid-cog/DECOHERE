#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Logging utilities for the DECOHERE pipeline.
"""

import os
import logging
import sys
from typing import Dict, Any, Optional


def setup_logger(config: Dict[str, Any], name: Optional[str] = None) -> logging.Logger:
    """
    Set up a logger based on the configuration.
    
    Args:
        config: Configuration dictionary containing logging settings
        name: Name of the logger (if None, use the root logger)
        
    Returns:
        Configured logger
    """
    # Get logger
    logger = logging.getLogger(name)
    
    # Clear existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Set log level
    log_level = getattr(logging, config.get('logging', {}).get('level', 'INFO'))
    logger.setLevel(log_level)
    
    # Create formatters
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Add console handler if enabled
    if config.get('logging', {}).get('console', True):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # Add file handler if specified
    log_file = config.get('logging', {}).get('file')
    if log_file:
        # Create directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def log_config(logger: logging.Logger, config: Dict[str, Any]) -> None:
    """
    Log the configuration settings.
    
    Args:
        logger: Logger to use
        config: Configuration dictionary to log
    """
    logger.info("Configuration settings:")
    
    # Log main sections
    for section, settings in config.items():
        if isinstance(settings, dict):
            logger.info(f"  {section}:")
            for key, value in settings.items():
                if not isinstance(value, dict) and not isinstance(value, list):
                    logger.info(f"    {key}: {value}")
        else:
            logger.info(f"  {section}: {settings}")
    
    logger.info("Configuration loaded successfully") 
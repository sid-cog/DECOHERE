#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Logging configuration for the DECOHERE pipeline.
"""

import logging
import os
from pathlib import Path
from typing import Dict, Any

def setup_logging(config: Dict[str, Any]) -> logging.Logger:
    """
    Set up logging configuration for the pipeline.
    
    Args:
        config: Configuration dictionary containing logging settings
        
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logs directory if it doesn't exist
    log_dir = Path(config.get('logging', {}).get('log_dir', 'logs'))
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure logger
    logger = logging.getLogger('DECOHERE')
    logger.setLevel(logging.INFO)
    
    # Create handlers
    log_file = log_dir / f"pipeline_{os.getpid()}.log"
    file_handler = logging.FileHandler(log_file)
    console_handler = logging.StreamHandler()
    
    # Create formatters and add it to handlers
    log_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(log_format)
    console_handler.setFormatter(log_format)
    
    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger 
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to update the configuration file to include the ALL YEARS mode.
"""

import os
import sys
import logging
import yaml
import shutil
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger('update_config')

def main():
    """
    Main function to update the configuration file.
    """
    logger.info("Starting to update the configuration file...")
    
    # Define paths
    config_path = "config/config.yaml"
    backup_path = f"{config_path}.bak_before_efficient_storage"
    
    # Create backup
    logger.info(f"Creating backup at {backup_path}")
    shutil.copy2(config_path, backup_path)
    
    # Read current configuration
    logger.info("Reading current configuration")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Add ALL YEARS mode
    logger.info("Adding ALL YEARS mode to configuration")
    if 'modes' not in config:
        config['modes'] = {}
    
    # Add the ALL YEARS mode
    config['modes']['all_years'] = {
        'description': 'Process all available data across all years',
        'config_file': 'config/modes/all_years_mode.yaml'
    }
    
    # Create the ALL YEARS mode configuration file
    all_years_config_path = "config/modes/all_years_mode.yaml"
    os.makedirs(os.path.dirname(all_years_config_path), exist_ok=True)
    
    # Define the ALL YEARS mode configuration
    all_years_config = {
        'mode': 'all_years',
        'description': 'Process all available data across all years',
        'processing': {
            'reprocess_data': False,
            'enable_filling': True,
            'winsorize_threshold': 3.0
        },
        'storage': {
            'use_efficient_storage': True,
            'partition_by': ['year', 'month'],
            'compression': 'snappy'
        }
    }
    
    # Write the ALL YEARS mode configuration
    logger.info(f"Writing ALL YEARS mode configuration to {all_years_config_path}")
    with open(all_years_config_path, 'w') as f:
        yaml.dump(all_years_config, f, default_flow_style=False)
    
    # Add storage configuration to main config if not present
    if 'storage' not in config:
        config['storage'] = {
            'use_efficient_storage': True,
            'partition_by': ['year', 'month'],
            'compression': 'snappy'
        }
    
    # Write the updated configuration
    logger.info("Writing updated configuration")
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    logger.info("Configuration file has been updated to include the ALL YEARS mode")

if __name__ == "__main__":
    main() 
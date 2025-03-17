#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to set up the efficient storage system for the DECOHERE project.
This script will:
1. Install required dependencies
2. Update the DataProcessor class
3. Update the pipeline notebook
4. Update the configuration file
5. Run a demonstration of the efficient storage system
"""

import os
import sys
import logging
import subprocess
import importlib.util
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger('setup_efficient_storage')

def check_and_install_dependencies():
    """
    Check and install required dependencies.
    """
    logger.info("Checking and installing required dependencies...")
    
    # List of required packages
    required_packages = [
        'pyarrow>=14.0.1',
        'nbformat>=5.9.2',
        'pandas>=2.0.0',
        'ipywidgets>=8.0.0'
    ]
    
    # Install required packages
    for package in required_packages:
        logger.info(f"Installing {package}")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install {package}: {e}")
            return False
    
    logger.info("All dependencies installed successfully")
    return True

def run_update_scripts():
    """
    Run the update scripts.
    """
    logger.info("Running update scripts...")
    
    # Update the DataProcessor class
    logger.info("Updating DataProcessor class...")
    try:
        subprocess.check_call([sys.executable, 'src/data/update_data_processor.py'])
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to update DataProcessor class: {e}")
        return False
    
    # Update the pipeline notebook
    logger.info("Updating pipeline notebook...")
    try:
        subprocess.check_call([sys.executable, 'src/data/update_pipeline.py'])
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to update pipeline notebook: {e}")
        return False
    
    # Update the configuration file
    logger.info("Updating configuration file...")
    try:
        subprocess.check_call([sys.executable, 'src/data/update_config.py'])
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to update configuration file: {e}")
        return False
    
    logger.info("All update scripts completed successfully")
    return True

def run_demonstration():
    """
    Run a demonstration of the efficient storage system.
    """
    logger.info("Running demonstration of the efficient storage system...")
    
    try:
        subprocess.check_call([sys.executable, 'src/data/demo_efficient_storage.py'])
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to run demonstration: {e}")
        return False
    
    logger.info("Demonstration completed successfully")
    return True

def main():
    """
    Main function to set up the efficient storage system.
    """
    logger.info("Starting setup of the efficient storage system...")
    
    # Check and install dependencies
    if not check_and_install_dependencies():
        logger.error("Failed to install dependencies")
        return False
    
    # Run update scripts
    if not run_update_scripts():
        logger.error("Failed to run update scripts")
        return False
    
    # Run demonstration
    if not run_demonstration():
        logger.error("Failed to run demonstration")
        return False
    
    logger.info("Efficient storage system setup completed successfully")
    logger.info("\nYou can now use the efficient storage system in your code:")
    logger.info("1. The DataProcessor class has been updated to use the efficient storage system")
    logger.info("2. The pipeline notebook has been updated to support the 'all_years' mode")
    logger.info("3. The configuration file has been updated with the new storage options")
    logger.info("\nTo use the efficient storage system in your code, you can:")
    logger.info("- Use the DataProcessor.load_processed_data_by_mode() method")
    logger.info("- Run the pipeline notebook with the 'all_years' mode")
    logger.info("- Use the EfficientDataStorage class directly for advanced use cases")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 
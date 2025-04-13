#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to process data for a single date.
"""

import os
import sys
import yaml
import logging
from datetime import datetime
from pathlib import Path

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent.parent)
sys.path.append(project_root)

from src.data.data_processor import DataProcessor

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def load_config():
    """Load configuration from config.yaml."""
    config_path = os.path.join(project_root, 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    """Main function to process data for a single date."""
    # Set up logging
    logger = setup_logging()
    logger.info("Starting data processing for single date")
    
    # Load configuration
    config = load_config()
    
    # Initialize data processor
    processor = DataProcessor(config, logger)
    
    # Process data for 2024-09-02
    date = "2024-09-02"
    logger.info(f"Processing data for date: {date}")
    
    # Process the data
    processed_files = processor.process_data(start_date=date, end_date=date)
    
    if date in processed_files:
        logger.info(f"Successfully processed data for {date}")
        logger.info(f"Output file: {processed_files[date]}")
    else:
        logger.error(f"Failed to process data for {date}")

if __name__ == "__main__":
    main() 
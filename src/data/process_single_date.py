#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to process data for a single date.
"""

import os
import sys
import logging
import yaml
from pathlib import Path

# Add the project root to the Python path
project_root = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.insert(0, project_root)

from src.data.data_processor import DataProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger('process_single_date')

def load_config():
    """Load the configuration file."""
    config_path = os.path.join(project_root, 'config', 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    """Main function to process data for a single date."""
    # Load configuration
    config = load_config()
    
    # Initialize data processor
    data_processor = DataProcessor(config, logger)
    
    # Process data for specific date
    date = "2024-09-02"
    logger.info(f"Processing data for date: {date}")
    
    # Load raw data
    raw_data = data_processor.load_raw_data(date)
    logger.info(f"Loaded raw data with shape: {raw_data.shape}")
    
    # Transform raw data
    intermediate_data = data_processor.transform_raw_data(raw_data)
    logger.info(f"Transformed data with shape: {intermediate_data.shape}")
    
    # Process data
    processed_data = data_processor.process_data(intermediate_data, end_date=date)
    logger.info(f"Processed data with shape: {processed_data.shape}")
    
    # Generate features
    features = data_processor.processed_data_feat_gen(processed_data)
    logger.info(f"Generated features with shape: {features.shape}")
    
    # Store data in partitioned format
    data_processor.store_data(processed_data, date)
    logger.info("Stored processed data")
    
    # Store features
    data_processor.store_features(features, date)
    logger.info("Stored features")

if __name__ == "__main__":
    main() 
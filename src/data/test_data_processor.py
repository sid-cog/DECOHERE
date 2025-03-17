#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script to verify that the DataProcessor can correctly load data from the new storage structure.
"""

import os
import sys
import logging
import pandas as pd
from typing import Dict, List, Optional, Any, Union

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import the DataProcessor class
from src.data.data_processor import DataProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger('test_data_processor')

def load_config() -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Returns:
        Configuration dictionary
    """
    import yaml
    
    config_path = "config/config.yaml"
    
    # Check if the file exists
    if not os.path.exists(config_path):
        logger.error(f"Configuration file not found: {config_path}")
        return {}
    
    # Load the configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Loaded configuration from {config_path}")
    
    return config

def test_load_raw_data():
    """
    Test loading raw data from the new storage structure.
    """
    logger.info("Testing load_raw_data method...")
    
    # Load configuration
    config = load_config()
    if not config:
        logger.error("Failed to load configuration")
        return
    
    # Initialize the data processor
    data_processor = DataProcessor(config, logger)
    
    # Test loading raw data without a specific date to get all data
    logger.info("Loading raw data without date filter")
    raw_data = data_processor.load_raw_data()
    
    # Check if the data was loaded successfully
    if raw_data.empty:
        logger.error("Failed to load raw data")
        return
    
    logger.info(f"Successfully loaded raw data with shape: {raw_data.shape}")
    logger.info(f"Raw data columns: {raw_data.columns.tolist()}")
    
    # Check if sector columns are present
    sector_cols = [col for col in raw_data.columns if 'SECTOR_' in col]
    if sector_cols:
        logger.info(f"Found sector columns: {sector_cols}")
        
        # Sample of sector data
        sample_size = min(5, len(raw_data))
        sector_sample = raw_data.sample(sample_size)[['ID'] + sector_cols]
        logger.info(f"Sample of sector data:\n{sector_sample}")
        
        # Count of unique values in each sector column
        for col in sector_cols:
            unique_values = raw_data[col].nunique()
            logger.info(f"Column {col} has {unique_values} unique values")
    else:
        logger.warning("No sector columns found in raw data")
        
        # Let's try to load data from the fundamentals directory directly
        logger.info("Attempting to load data from fundamentals directory directly")
        try:
            fundamentals_path = os.path.join(config['data']['processed_data'], "fundamentals", "fundamentals_2024_09.pq")
            if os.path.exists(fundamentals_path):
                fundamentals_df = pd.read_parquet(fundamentals_path)
                logger.info(f"Successfully loaded fundamentals data with shape: {fundamentals_df.shape}")
                logger.info(f"Fundamentals data columns: {fundamentals_df.columns.tolist()}")
                
                # Check if sector columns are present
                sector_cols = [col for col in fundamentals_df.columns if 'SECTOR_' in col]
                if sector_cols:
                    logger.info(f"Found sector columns in fundamentals data: {sector_cols}")
                    
                    # Sample of sector data
                    sample_size = min(5, len(fundamentals_df))
                    sector_sample = fundamentals_df.sample(sample_size)[['ID'] + sector_cols]
                    logger.info(f"Sample of sector data from fundamentals:\n{sector_sample}")
                    
                    # Count of unique values in each sector column
                    for col in sector_cols:
                        unique_values = fundamentals_df[col].nunique()
                        logger.info(f"Column {col} has {unique_values} unique values")
                else:
                    logger.warning("No sector columns found in fundamentals data")
            else:
                logger.warning(f"Fundamentals file not found: {fundamentals_path}")
        except Exception as e:
            logger.error(f"Error loading fundamentals data: {e}")

def test_load_all_data_sources():
    """
    Test loading all data sources using the load_all_data_sources method.
    """
    logger.info("Testing load_all_data_sources method...")
    
    # Load configuration
    config = load_config()
    if not config:
        logger.error("Failed to load configuration")
        return
    
    # Initialize the data processor
    data_processor = DataProcessor(config, logger)
    
    # Test loading all data sources
    logger.info("Loading all data sources")
    try:
        all_data = data_processor.load_all_data_sources()
        
        # Check if the data was loaded successfully
        if not all_data:
            logger.error("Failed to load all data sources")
            return
        
        # Log information about each data source
        for source_name, source_data in all_data.items():
            if isinstance(source_data, pd.DataFrame) and not source_data.empty:
                logger.info(f"Successfully loaded {source_name} with shape: {source_data.shape}")
                
                # Sample of data
                sample_size = min(3, len(source_data))
                if source_name == 'sector_mapping' and not source_data.empty:
                    # For sector mapping, show the sector columns
                    sector_cols = [col for col in source_data.columns if 'SECTOR_' in col]
                    if sector_cols:
                        sample_cols = ['ID'] + sector_cols[:2]  # Show ID and first two sector columns
                        logger.info(f"Sample of {source_name} (columns: {sample_cols}):\n{source_data.sample(sample_size)[sample_cols]}")
                    else:
                        logger.info(f"Sample of {source_name}:\n{source_data.sample(sample_size)}")
                else:
                    # For other data sources, show a sample
                    logger.info(f"Sample of {source_name}:\n{source_data.sample(sample_size)}")
            elif isinstance(source_data, pd.DataFrame) and source_data.empty:
                logger.warning(f"Data source '{source_name}' is empty")
            else:
                logger.info(f"Loaded {source_name} (not a DataFrame)")
    except Exception as e:
        logger.error(f"Error loading all data sources: {e}")

def main():
    """
    Main function to run all tests.
    """
    logger.info("Starting tests for DataProcessor with new storage structure...")
    
    # Test loading raw data
    test_load_raw_data()
    
    # Test loading all data sources
    test_load_all_data_sources()
    
    logger.info("All tests completed")

if __name__ == "__main__":
    main() 
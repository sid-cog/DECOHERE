#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for the DECOHERE data processing pipeline.
This script tests the pipeline with a specific ID and PIT_DATE.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from datetime import datetime

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import the necessary modules
from data.data_processor import DataProcessor
from features.feature_generator import FeatureGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('test_pipeline.log')
    ]
)
logger = logging.getLogger('test_pipeline')

def load_config():
    """Load the configuration from the config file."""
    config_path = os.path.join(os.path.dirname(__file__), 'config', 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def test_pipeline(ticker_id='360ONE IB Equity', pit_date='2024-09-20'):
    """
    Test the data processing pipeline with a specific ID and PIT_DATE.
    
    Args:
        ticker_id: The ID of the ticker to test
        pit_date: The PIT_DATE to test
    """
    logger.info(f"Testing pipeline with ticker_id={ticker_id}, pit_date={pit_date}")
    
    # Load the configuration
    config = load_config()
    logger.info("Loaded configuration")
    
    # Initialize the data processor
    data_processor = DataProcessor(config)
    logger.info("Initialized data processor")
    
    # Load the raw data
    raw_data = data_processor.load_raw_data()
    logger.info(f"Loaded raw data with shape: {raw_data.shape}")
    
    # Filter for the specific ticker and date
    filtered_data = raw_data[(raw_data['ID'] == ticker_id) & (raw_data['PIT_DATE'] == pit_date)]
    logger.info(f"Filtered data for ticker_id={ticker_id}, pit_date={pit_date} with shape: {filtered_data.shape}")
    
    if filtered_data.empty:
        logger.error(f"No data found for ticker_id={ticker_id}, pit_date={pit_date}")
        # Try to find available tickers and dates
        available_tickers = raw_data['ID'].unique()[:5]
        available_dates = pd.to_datetime(raw_data['PIT_DATE']).dt.strftime('%Y-%m-%d').unique()[:5]
        logger.info(f"Available tickers (first 5): {available_tickers}")
        logger.info(f"Available dates (first 5): {available_dates}")
        return
    
    # Process the data
    processed_data = data_processor.transform_raw_data(filtered_data)
    logger.info(f"Processed data with shape: {processed_data.shape}")
    
    # Initialize the feature generator
    feature_generator = FeatureGenerator(config)
    logger.info("Initialized feature generator")
    
    # Generate features
    features = feature_generator.generate_features(processed_data)
    logger.info(f"Generated features with shape: {features.shape}")
    
    # Print sample features
    logger.info("Sample features:")
    logger.info(features.head().to_string())
    
    # Print success message
    logger.info("FUCK YEAH! Pipeline test completed successfully.")
    print("\nFUCK YEAH! Pipeline test completed successfully.\n")
    
    return features

if __name__ == "__main__":
    test_pipeline() 
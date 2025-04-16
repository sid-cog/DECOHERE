"""
Simple script to compare original and optimized implementations.
"""

import os
import sys
import yaml
import logging
import pandas as pd
import numpy as np
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.feature_generator import FeatureGenerator
from src.data.feature_generator_optimized import OptimizedFeatureGenerator
from src.data.data_processor import DataProcessor

def setup_logging():
    """Configure logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger('CompareImplementations')

def load_config():
    """Load configuration"""
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        'config',
        'config.yaml'
    )
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def compare_dataframes(df1: pd.DataFrame, df2: pd.DataFrame, logger: logging.Logger):
    """Compare two DataFrames and show differences"""
    # Check columns
    if not df1.columns.equals(df2.columns):
        logger.error("Column mismatch:")
        logger.error(f"Original columns: {df1.columns.tolist()}")
        logger.error(f"Optimized columns: {df2.columns.tolist()}")
        return
        
    # Check indices
    if not df1.index.equals(df2.index):
        logger.error("Index mismatch")
        return
        
    # Compare numerical values
    numerical_cols = df1.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        diff = pd.Series(np.abs(df1[col] - df2[col]), index=df1.index)
        max_diff = diff.max()
        mean_diff = diff.mean()
        
        if max_diff > 1e-10:
            max_idx = diff.idxmax()
            logger.info(f"\nColumn: {col}")
            logger.info(f"Max difference: {max_diff:.10f}")
            logger.info(f"Mean difference: {mean_diff:.10f}")
            logger.info(f"Location of max difference: {max_idx}")
            logger.info(f"Original value: {df1[col].loc[max_idx]:.10f}")
            logger.info(f"Optimized value: {df2[col].loc[max_idx]:.10f}")
            
    # Compare non-numerical columns
    non_numerical_cols = df1.select_dtypes(exclude=[np.number]).columns
    for col in non_numerical_cols:
        if not df1[col].equals(df2[col]):
            logger.error(f"Values differ in non-numerical column {col}")

def main():
    """Main function"""
    logger = setup_logging()
    config = load_config()
    
    # Test date
    test_date = "2024-09-02"
    logger.info(f"Testing implementations for date {test_date}")
    
    # Initialize components
    processor = DataProcessor(config, winsorize=True)
    original_generator = FeatureGenerator(config, logger)
    optimized_generator = OptimizedFeatureGenerator(config, logger)
    
    # Load and process data
    logger.info("Loading raw data...")
    raw_data = processor.load_raw_data(test_date)
    if raw_data.empty:
        logger.error("No data found")
        return
        
    logger.info(f"Raw data shape: {raw_data.shape}")
    logger.info("Transforming data...")
    transformed_data = processor.transform_raw_data(raw_data)
    logger.info(f"Transformed data shape: {transformed_data.shape}")
    
    # Generate features
    logger.info("\nGenerating features with original implementation...")
    start_time = pd.Timestamp.now()
    original_features = original_generator.generate_enhanced_features(transformed_data)
    original_duration = (pd.Timestamp.now() - start_time).total_seconds()
    logger.info(f"Original implementation completed in {original_duration:.2f} seconds")
    logger.info(f"Original features shape: {original_features.shape}")
    
    logger.info("\nGenerating features with optimized implementation...")
    start_time = pd.Timestamp.now()
    optimized_features = optimized_generator.generate_enhanced_features(transformed_data)
    optimized_duration = (pd.Timestamp.now() - start_time).total_seconds()
    logger.info(f"Optimized implementation completed in {optimized_duration:.2f} seconds")
    logger.info(f"Optimized features shape: {optimized_features.shape}")
    
    # Compare results
    logger.info("\nComparing results...")
    compare_dataframes(original_features, optimized_features, logger)
    
    # Show performance comparison
    speedup = original_duration / optimized_duration
    logger.info(f"\nPerformance Comparison:")
    logger.info(f"Original duration: {original_duration:.2f} seconds")
    logger.info(f"Optimized duration: {optimized_duration:.2f} seconds")
    logger.info(f"Speedup: {speedup:.2f}x")

if __name__ == "__main__":
    main() 
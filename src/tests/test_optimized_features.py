"""
Test script to compare outputs between original and optimized feature generators.
"""

import os
import sys
import yaml
import logging
import pandas as pd
import numpy as np
from typing import Dict, Tuple

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.feature_generator import FeatureGenerator
from src.data.feature_generator_optimized import OptimizedFeatureGenerator
from src.data.data_processor import DataProcessor

def setup_logging() -> logging.Logger:
    """Configure logging for tests"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger('TestOptimizedFeatures')

def load_config() -> Dict:
    """Load configuration from config file"""
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        'config',
        'config.yaml'
    )
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def compare_dataframes(df1: pd.DataFrame, df2: pd.DataFrame, logger: logging.Logger) -> bool:
    """
    Compare two DataFrames and log differences.
    Returns True if DataFrames are equal within numerical tolerance.
    """
    # Check if columns match
    if not df1.columns.equals(df2.columns):
        logger.error("Column mismatch:")
        logger.error(f"Original columns: {df1.columns.tolist()}")
        logger.error(f"Optimized columns: {df2.columns.tolist()}")
        return False
        
    # Check if indices match
    if not df1.index.equals(df2.index):
        logger.error("Index mismatch")
        return False
    
    # Compare numerical values with tolerance
    numerical_cols = df1.select_dtypes(include=[np.number]).columns
    differences = []
    
    for col in numerical_cols:
        # Convert to pandas Series for idxmax
        diff = pd.Series(np.abs(df1[col] - df2[col]), index=df1.index)
        max_diff = diff.max()
        mean_diff = diff.mean()
        if max_diff > 1e-10:  # Numerical tolerance
            max_idx = diff.idxmax()
            differences.append({
                'column': col,
                'max_diff': max_diff,
                'mean_diff': mean_diff,
                'max_diff_idx': max_idx,
                'max_diff_values': (df1[col].loc[max_idx], df2[col].loc[max_idx])
            })
    
    if differences:
        logger.error("\nFound differences in numerical columns:")
        for diff in differences:
            logger.error(f"\nColumn: {diff['column']}")
            logger.error(f"Max difference: {diff['max_diff']:.10f}")
            logger.error(f"Mean difference: {diff['mean_diff']:.10f}")
            logger.error(f"Location of max difference: {diff['max_diff_idx']}")
            logger.error(f"Original value: {diff['max_diff_values'][0]:.10f}")
            logger.error(f"Optimized value: {diff['max_diff_values'][1]:.10f}")
        return False
            
    # Compare non-numerical columns exactly
    non_numerical_cols = df1.select_dtypes(exclude=[np.number]).columns
    for col in non_numerical_cols:
        if not df1[col].equals(df2[col]):
            logger.error(f"Values differ in non-numerical column {col}")
            return False
    
    return True

def test_single_date(date_str: str, config: Dict, logger: logging.Logger) -> Tuple[bool, float, float]:
    """
    Test optimized implementation against original for a single date.
    Returns (success, original_duration, optimized_duration).
    """
    # Initialize components
    processor = DataProcessor(config, winsorize=True)
    original_generator = FeatureGenerator(config, logger)
    optimized_generator = OptimizedFeatureGenerator(config, logger)
    
    # Load and process raw data
    raw_data = processor.load_raw_data(date_str)
    if raw_data.empty:
        logger.warning(f"No data found for date {date_str}")
        return False, 0, 0
        
    transformed_data = processor.transform_raw_data(raw_data)
    
    # Generate features using original implementation
    logger.info("Running original implementation...")
    start_time = pd.Timestamp.now()
    original_features = original_generator.generate_enhanced_features(transformed_data)
    original_duration = (pd.Timestamp.now() - start_time).total_seconds()
    logger.info(f"Original implementation completed in {original_duration:.2f} seconds")
    
    # Generate features using optimized implementation
    logger.info("Running optimized implementation...")
    start_time = pd.Timestamp.now()
    optimized_features = optimized_generator.generate_enhanced_features(transformed_data)
    optimized_duration = (pd.Timestamp.now() - start_time).total_seconds()
    logger.info(f"Optimized implementation completed in {optimized_duration:.2f} seconds")
    
    # Compare results
    logger.info("Comparing results...")
    success = compare_dataframes(original_features, optimized_features, logger)
    
    if success:
        speedup = original_duration / optimized_duration
        logger.info(f"Results match! Speedup: {speedup:.2f}x")
    else:
        logger.error("Results do not match!")
        
    return success, original_duration, optimized_duration

def compare_with_source_of_truth(
    generated_df: pd.DataFrame,
    date_str: str,
    logger: logging.Logger
) -> bool:
    """Compare generated features with source of truth parquet file."""
    # Construct path to source of truth file
    year = date_str[:4]
    month = date_str[5:7]
    truth_path = f"/home/siddharth.johri/DECOHERE/data/features/fundamentals/enhanced_features/year={year}/month={month}/data_{date_str} copy.pq"
    
    # Load source of truth
    try:
        truth_df = pd.read_parquet(truth_path)
        logger.info(f"Loaded source of truth from {truth_path}")
    except Exception as e:
        logger.error(f"Failed to load source of truth: {str(e)}")
        return False
    
    # Compare DataFrames
    return compare_dataframes(truth_df, generated_df, logger)

def main():
    """Main test function"""
    logger = setup_logging()
    config = load_config()
    
    test_date = "2024-09-02"
    logger.info(f"Testing optimized implementation for date {test_date}")
    
    # Run test
    success, orig_time, opt_time = test_single_date(test_date, config, logger)
    
    if success:
        logger.info("\nTest Results:")
        logger.info(f"Original implementation time: {orig_time:.2f} seconds")
        logger.info(f"Optimized implementation time: {opt_time:.2f} seconds")
        logger.info(f"Speedup: {orig_time/opt_time:.2f}x")
    else:
        logger.error("Test failed - outputs do not match")
        sys.exit(1)

if __name__ == "__main__":
    main() 
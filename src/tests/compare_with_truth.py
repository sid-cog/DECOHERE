"""
Simple script to compare optimized implementation with source of truth.
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

from src.data.feature_generator_optimized import OptimizedFeatureGenerator
from src.data.data_processor import DataProcessor

def setup_logging():
    """Configure logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger('CompareWithTruth')

def load_config():
    """Load configuration"""
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        'config',
        'config.yaml'
    )
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def compare_with_truth(generated_df: pd.DataFrame, truth_path: str, logger: logging.Logger):
    """Compare generated DataFrame with source of truth"""
    try:
        # Load source of truth
        logger.info(f"Loading source of truth from {truth_path}")
        truth_df = pd.read_parquet(truth_path)
        logger.info(f"Source of truth shape: {truth_df.shape}")
        
        # Check columns
        if not generated_df.columns.equals(truth_df.columns):
            logger.error("Column mismatch:")
            logger.error(f"Generated columns: {generated_df.columns.tolist()}")
            logger.error(f"Truth columns: {truth_df.columns.tolist()}")
            return
            
        # Check indices
        if not generated_df.index.equals(truth_df.index):
            logger.error("Index mismatch")
            return
            
        # Compare numerical values
        numerical_cols = generated_df.select_dtypes(include=[np.number]).columns
        differences = []
        
        for col in numerical_cols:
            diff = pd.Series(np.abs(generated_df[col] - truth_df[col]), index=generated_df.index)
            max_diff = diff.max()
            mean_diff = diff.mean()
            
            if max_diff > 1e-10:
                max_idx = diff.idxmax()
                differences.append({
                    'column': col,
                    'max_diff': max_diff,
                    'mean_diff': mean_diff,
                    'max_diff_idx': max_idx,
                    'generated_value': generated_df[col].loc[max_idx],
                    'truth_value': truth_df[col].loc[max_idx]
                })
        
        if differences:
            logger.error("\nFound differences in numerical columns:")
            for diff in differences:
                logger.error(f"\nColumn: {diff['column']}")
                logger.error(f"Max difference: {diff['max_diff']:.10f}")
                logger.error(f"Mean difference: {diff['mean_diff']:.10f}")
                logger.error(f"Location of max difference: {diff['max_diff_idx']}")
                logger.error(f"Generated value: {diff['generated_value']:.10f}")
                logger.error(f"Truth value: {diff['truth_value']:.10f}")
        else:
            logger.info("No differences found in numerical columns!")
            
        # Compare non-numerical columns
        non_numerical_cols = generated_df.select_dtypes(exclude=[np.number]).columns
        for col in non_numerical_cols:
            if not generated_df[col].equals(truth_df[col]):
                logger.error(f"Values differ in non-numerical column {col}")
            else:
                logger.info(f"Non-numerical column {col} matches!")
                
    except Exception as e:
        logger.error(f"Error comparing with truth: {str(e)}")

def main():
    """Main function"""
    logger = setup_logging()
    config = load_config()
    
    # Test date
    test_date = "2024-09-02"
    logger.info(f"Testing optimized implementation for date {test_date}")
    
    # Initialize components
    processor = DataProcessor(config, winsorize=True)
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
    logger.info("\nGenerating features with optimized implementation...")
    start_time = pd.Timestamp.now()
    optimized_features = optimized_generator.generate_enhanced_features(transformed_data)
    duration = (pd.Timestamp.now() - start_time).total_seconds()
    logger.info(f"Optimized implementation completed in {duration:.2f} seconds")
    logger.info(f"Optimized features shape: {optimized_features.shape}")
    
    # Compare with source of truth
    year = test_date[:4]
    month = test_date[5:7]
    truth_path = f"/home/siddharth.johri/DECOHERE/data/features/fundamentals/enhanced_features/year={year}/month={month}/data_{test_date} copy.pq"
    
    logger.info("\nComparing with source of truth...")
    compare_with_truth(optimized_features, truth_path, logger)

if __name__ == "__main__":
    main() 
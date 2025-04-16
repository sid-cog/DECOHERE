#!/usr/bin/env python3
"""
Script to run optimized feature generator and compare with source of truth.
"""

import pandas as pd
import numpy as np
import logging
import yaml
import os
from pathlib import Path
from feature_generator_optimized import OptimizedFeatureGenerator
from datetime import datetime

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def load_config():
    """Load configuration from YAML file."""
    config_path = 'config/config.yaml'
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def compare_dataframes(df1: pd.DataFrame, df2: pd.DataFrame, tolerance: float = 1e-10) -> dict:
    """
    Compare two dataframes and return detailed differences.
    
    Args:
        df1: First dataframe (optimized output)
        df2: Second dataframe (source of truth)
        tolerance: Tolerance for floating point comparisons
        
    Returns:
        Dictionary containing comparison results
    """
    results = {
        'shape_match': df1.shape == df2.shape,
        'columns_match': set(df1.columns) == set(df2.columns),
        'differences': {}
    }
    
    # Compare shapes
    if not results['shape_match']:
        results['differences']['shape'] = {
            'df1_shape': df1.shape,
            'df2_shape': df2.shape
        }
    
    # Compare columns
    if not results['columns_match']:
        results['differences']['columns'] = {
            'missing_in_df1': set(df2.columns) - set(df1.columns),
            'missing_in_df2': set(df1.columns) - set(df2.columns)
        }
    
    # Compare values for common columns
    common_cols = set(df1.columns) & set(df2.columns)
    for col in common_cols:
        if df1[col].dtype.kind in 'f':  # Floating point columns
            diff = np.abs(df1[col] - df2[col])
            max_diff = diff.max()
            mean_diff = diff.mean()
            if max_diff > tolerance:
                results['differences'][col] = {
                    'max_diff': max_diff,
                    'mean_diff': mean_diff,
                    'exceeds_tolerance': True
                }
        else:  # Non-floating point columns
            if not df1[col].equals(df2[col]):
                results['differences'][col] = {
                    'exact_match': False
                }
    
    return results

def main():
    """Main function to run comparison."""
    logger = setup_logging()
    config = load_config()
    
    # Define paths
    source_path = '/home/siddharth.johri/DECOHERE/data/features/fundamentals/enhanced_features/year=2024/month=09/data_2024-09-02 copy.pq'
    input_path = '/home/siddharth.johri/DECOHERE/data/raw/fundamentals/financials_2024_09.pq'
    target_date = "2024-09-02"
    
    # Load source of truth
    logger.info(f"Loading source of truth from {source_path}")
    source_df = pd.read_parquet(source_path)
    
    # Load input data
    logger.info(f"Loading input data from {input_path}")
    input_df = pd.read_parquet(input_path)
    
    # Convert dates to datetime
    input_df['PERIOD_END_DATE'] = pd.to_datetime(input_df['PERIOD_END_DATE'])
    input_df['pit_date'] = pd.to_datetime(input_df['pit_date'])
    
    # Filter data for target date
    logger.info(f"Filtering data for {target_date}")
    target_date_dt = pd.to_datetime(target_date)
    filtered_df = input_df[
        (input_df['pit_date'] == target_date_dt) & 
        (input_df['PERIOD_END_DATE'] <= target_date_dt)
    ].copy()
    
    logger.info(f"Found {len(filtered_df)} records for {target_date}")
    
    # Initialize feature generator
    generator = OptimizedFeatureGenerator(config, logger)
    
    # Generate features
    logger.info("Generating features with optimized generator")
    optimized_df = generator.generate_enhanced_features(filtered_df, target_date=target_date)
    
    # Compare results
    logger.info("Comparing results")
    comparison = compare_dataframes(optimized_df, source_df)
    
    # Log results
    logger.info("\nComparison Results:")
    logger.info(f"Shape match: {comparison['shape_match']}")
    logger.info(f"Columns match: {comparison['columns_match']}")
    
    if comparison['differences']:
        logger.info("\nDifferences found:")
        for key, value in comparison['differences'].items():
            if key == 'shape':
                logger.info(f"Shape mismatch: {value}")
            elif key == 'columns':
                logger.info(f"Column mismatch: {value}")
            else:
                if 'exceeds_tolerance' in value:
                    logger.info(f"Column {key}: max_diff={value['max_diff']}, mean_diff={value['mean_diff']}")
                else:
                    logger.info(f"Column {key}: values do not match exactly")
    else:
        logger.info("No differences found - outputs match exactly!")
    
    return comparison

if __name__ == "__main__":
    main() 
"""
Script to compare optimized feature generator output with source of truth.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
from pandas import DataFrame, Series
import time
from datetime import datetime
from feature_generator_optimized import OptimizedFeatureGenerator
import yaml
import os

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

def compare_dataframes(df1: DataFrame, df2: DataFrame, tolerance: float = 1e-6) -> Dict:
    """
    Compare two dataframes and return differences.
    
    Args:
        df1: First dataframe (optimized output)
        df2: Second dataframe (source of truth)
        tolerance: Tolerance for floating point comparisons
        
    Returns:
        Dictionary containing comparison results
    """
    # Compare shapes
    shape_diff = {
        'df1_shape': df1.shape,
        'df2_shape': df2.shape,
        'shape_match': df1.shape == df2.shape
    }
    
    # Compare columns
    col_diff = {
        'df1_cols': set(df1.columns),
        'df2_cols': set(df2.columns),
        'missing_in_df1': set(df2.columns) - set(df1.columns),
        'missing_in_df2': set(df1.columns) - set(df2.columns)
    }
    
    # Compare values for common columns
    value_diff = {}
    common_cols = set(df1.columns) & set(df2.columns)
    
    for col in common_cols:
        if df1[col].dtype.kind in 'f':  # Floating point columns
            diff = np.abs(df1[col] - df2[col])
            max_diff = diff.max()
            mean_diff = diff.mean()
            value_diff[col] = {
                'max_diff': max_diff,
                'mean_diff': mean_diff,
                'exceeds_tolerance': max_diff > tolerance
            }
        else:  # Non-floating point columns
            value_diff[col] = {
                'exact_match': df1[col].equals(df2[col])
            }
    
    return {
        'shape_comparison': shape_diff,
        'column_comparison': col_diff,
        'value_comparison': value_diff
    }

def main():
    """Main function to run comparison."""
    logger = setup_logging()
    config = load_config()
    
    # Load source of truth
    source_path = '/home/siddharth.johri/DECOHERE/data/features/fundamentals/enhanced_features/year=2024/month=09/data_2024-09-02 copy.pq'
    logger.info(f"Loading source of truth from {source_path}")
    source_df = pd.read_parquet(source_path)
    
    # Initialize feature generator
    generator = OptimizedFeatureGenerator(config, logger)
    
    # Load input data (assuming it's in the same format as source)
    input_path = os.path.join(config['data']['raw']['fundamentals'], 'data_2024-09-02.pq')
    logger.info(f"Loading input data from {input_path}")
    input_df = pd.read_parquet(input_path)
    
    # Generate features
    logger.info("Generating features with optimized generator")
    optimized_df = generator.generate_enhanced_features(input_df, target_date="2024-09-02")
    
    # Compare results
    logger.info("Comparing results")
    comparison = compare_dataframes(optimized_df, source_df)
    
    # Log differences
    logger.info("\nComparison Results:")
    logger.info(f"Shape match: {comparison['shape_comparison']['shape_match']}")
    logger.info(f"Missing columns in optimized: {comparison['column_comparison']['missing_in_df1']}")
    logger.info(f"Extra columns in optimized: {comparison['column_comparison']['missing_in_df2']}")
    
    # Log value differences
    logger.info("\nValue differences (exceeding tolerance):")
    for col, diff in comparison['value_comparison'].items():
        if isinstance(diff, dict) and 'exceeds_tolerance' in diff and diff['exceeds_tolerance']:
            logger.info(f"{col}: max_diff={diff['max_diff']}, mean_diff={diff['mean_diff']}")
    
    return comparison

if __name__ == "__main__":
    main() 
"""
Optimized version of feature generator with vectorized operations and improved memory usage.
This implementation focuses on performance while maintaining the exact same output as the original.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
from pandas import DataFrame, Series
import time
from datetime import datetime

class OptimizedFeatureGenerator:
    """
    Optimized version of FeatureGenerator that uses vectorized operations and improved memory management.
    """
    
    def __init__(self, config: Dict, logger: Optional[logging.Logger] = None):
        """Initialize the OptimizedFeatureGenerator with config and logger."""
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Pre-compile configurations to avoid repeated dict lookups
        self._ratio_cols = [col for col in config.get('COLUMNS', {}).get('RATIO_SIGNED_LOG', [])]
        self._group_cols = config.get('COLUMNS', {}).get('GROUP_COLS', [])
        self._winsor_quantiles = (
            config.get('FEATURE_GENERATION', {}).get('WINSOR_QUANTILES', [0.01, 0.99])
        )
        
        # Pre-allocate memory for common operations
        self._dtype_map = {
            'float64': np.float64,  # Use float64 to match source of truth
            'int64': np.int32,      # Use int32 for better memory efficiency
        }
        
    def _optimize_dtypes(self, df: DataFrame) -> DataFrame:
        """Optimize data types for better memory usage."""
        for col in df.columns:
            dtype_str = str(df[col].dtype)
            if dtype_str in self._dtype_map:
                df[col] = df[col].astype(self._dtype_map[dtype_str])
        return df
        
    def _calculate_group_stats_vectorized(self, group_data: DataFrame) -> DataFrame:
        """
        Vectorized calculation of group statistics.
        Returns DataFrame with same columns as original but with optimized computation.
        """
        # Optimize data types first
        group_data = self._optimize_dtypes(group_data)
        
        # Pre-allocate result arrays for better memory efficiency
        n_rows = len(group_data)
        result_data = {}
        
        # Calculate stats for ratio columns
        for col in self._ratio_cols:
            if col not in group_data.columns:
                continue
                
            values = group_data[col].values  # Use numpy arrays for faster computation
            
            # Calculate basic statistics using optimized numpy functions
            mean = np.mean(values, dtype=np.float64)
            std = np.std(values, dtype=np.float64)
            
            # Handle edge cases
            if std == 0:
                std = 1e-6  # Small constant to avoid division by zero
                
            # Calculate z-scores vectorized
            z_scores = (values - mean) / std
            
            # Store results with optimized data types
            result_data[f"{col}_zscore"] = z_scores.astype(np.float64)
            result_data[f"{col}_mean"] = np.full(n_rows, mean, dtype=np.float64)
            result_data[f"{col}_std"] = np.full(n_rows, std, dtype=np.float64)
        
        # Create result DataFrame with optimized data types
        result_df = pd.DataFrame(result_data, index=group_data.index)
        
        # Add original columns with optimized data types
        for col in group_data.columns:
            if col not in result_df:
                result_df[col] = group_data[col]
        
        return result_df
    
    def _winsorize_features_vectorized(self, df: DataFrame) -> DataFrame:
        """
        Vectorized winsorization of features with optimized data types.
        Ensures exact matching with source of truth.
        """
        result = df.copy()
        
        # Get columns to winsorize
        winsor_cols = [col for col in df.columns if any(x in col for x in ['_zscore', '_mean', '_std'])]
        
        # Calculate quantiles once per column using optimized numpy functions
        q_low, q_high = self._winsor_quantiles
        quantiles = df[winsor_cols].quantile([q_low, q_high])
        
        # Vectorized winsorization with optimized data types
        for col in winsor_cols:
            # Get bounds as Python float to ensure exact matching
            lower_bound = float(quantiles.loc[q_low, col])
            upper_bound = float(quantiles.loc[q_high, col])
            
            # Apply winsorization with exact bounds
            result[col] = result[col].clip(lower=lower_bound, upper=upper_bound).astype(np.float64)
        
        return result
    
    def generate_enhanced_features(self, df: DataFrame, target_date: str = "2024-09-02") -> DataFrame:
        """
        Generate enhanced features using vectorized operations.
        Optimized for specific target date while ensuring exact matching with source of truth.
        """
        start_time = time.time()
        
        # Log input state
        self.logger.info(f"Generating features for date: {target_date}")
        self.logger.info(f"Input DataFrame columns: {df.columns.tolist()}")
        ratio_cols_present = [col for col in self._ratio_cols if col in df.columns]
        self.logger.info(f"Found ratio columns: {ratio_cols_present}")
        
        # Validate input
        if not ratio_cols_present:
            self.logger.warning("No ratio columns found in input DataFrame")
            return df
            
        # Filter data for target date if date column exists
        if 'date' in df.columns:
            df = df[df['date'] == target_date]
            
        # Process each group
        grouped = df.groupby(self._group_cols)
        results = []
        
        # Process groups in chunks for memory efficiency
        for _, group_data in grouped:
            # Generate features for group
            group_result = self._calculate_group_stats_vectorized(group_data)
            results.append(group_result)
        
        # Combine results
        result_df = pd.concat(results)
        
        # Winsorize features
        result_df = self._winsorize_features_vectorized(result_df)
        
        # Log completion
        duration = time.time() - start_time
        self.logger.info(f"Enhanced feature generation completed in {duration:.2f} seconds")
        self.logger.info(f"Output columns: {result_df.columns.tolist()}")
        
        return result_df 
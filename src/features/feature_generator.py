#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Feature Generator for the DECOHERE project.
This module contains the FeatureGenerator class, which is responsible for generating features from processed data.
"""

import os
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import time

class FeatureGenerator:
    """
    Feature Generator class for the DECOHERE project.
    This class is responsible for generating features from processed data.
    """
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Initialize the FeatureGenerator.
        
        Args:
            config: Configuration dictionary
            logger: Logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Get metrics from config
        self.metrics = config['features'].get('metrics', [])
        self.logger.info(f"Initialized FeatureGenerator with metrics: {self.metrics}")
        
        # Get financial metrics and their standard deviation fields
        self.financial_metrics = {
            'NET_INCOME': 'NET_INCOME_CSTAT_STD',
            'EBIT': 'EBIT_CSTAT_STD',
            'EBITDA': 'EBITDA_CSTAT_STD',
            'SALES': 'SALES_CSTAT_STD',
            'RETURN_COM_EQY': 'RETURN_COM_EQY_CSTAT_STD',
            'INVENTORY_TURNOVER': 'INVENTORY_TURNOVER_CSTAT_STD'
        }
        
        # Set paths
        self.processed_data_dir = config['data']['processed_data']
        self.features_dir = config['data']['features']['base_dir']
        
        # Create directories if they don't exist
        os.makedirs(self.processed_data_dir, exist_ok=True)
        os.makedirs(self.features_dir, exist_ok=True)
    
    def calculate_coefficient_of_variation(self, row: pd.Series, metric: str, std_field: str) -> float:
        """
        Calculate coefficient of variation (CV) for a metric using its value and standard deviation.
        CV = standard deviation / |mean|
        
        Args:
            row: Row containing the metric value and its standard deviation
            metric: Name of the metric field
            std_field: Name of the standard deviation field
            
        Returns:
            Coefficient of variation
        """
        value = row[metric]
        std = row[std_field]
        
        # If either value is NaN, return NaN
        if pd.isna(value) or pd.isna(std):
            return np.nan
            
        # Avoid division by zero
        if value == 0:
            return np.nan
            
        return std / abs(value)
    
    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate features from processed data.
        
        Args:
            df: Processed data
            
        Returns:
            DataFrame containing generated features
        """
        self.logger.info("Generating features...")
        
        if df.empty:
            self.logger.warning("Empty DataFrame provided. Cannot generate features.")
            return pd.DataFrame()
        
        # Make a copy of input DataFrame
        df_work = df.copy()
        
        # Calculate coefficient of variation for each financial metric
        for metric, std_field in self.financial_metrics.items():
            if metric in df_work.columns and std_field in df_work.columns:
                self.logger.info(f"Calculating coefficient of variation for {metric} using {std_field}")
                
                # Calculate CV for each row
                cv_column = f"{metric}_COEFF_OF_VAR"
                df_work[cv_column] = df_work.apply(
                    lambda row: self.calculate_coefficient_of_variation(row, metric, std_field), 
                    axis=1
                )
                
                self.logger.info(f"Generated {cv_column} with {df_work[cv_column].notna().sum()} non-null values")
        
        # Keep only ID, period-related columns, and CV columns
        cv_columns = [col for col in df_work.columns if col.endswith('_COEFF_OF_VAR')]
        keep_columns = ['ID', 'PIT_DATE', 'PERIOD_END_DATE', 'PERIOD'] + cv_columns
        df_features = df_work[keep_columns].copy()
        
        self.logger.info(f"Generated {len(cv_columns)} coefficient of variation features")
        
        return df_features
    
    def save_features(self, features_df: pd.DataFrame, date: str) -> str:
        """
        Save generated features to a CSV file.
        
        Args:
            features_df: DataFrame containing generated features
            date: Date to save features for
            
        Returns:
            Path to the saved features file
        """
        # Create directory if it doesn't exist
        features_dir = os.path.join(self.features_dir, date)
        os.makedirs(features_dir, exist_ok=True)
        
        # Save features
        features_path = os.path.join(features_dir, 'features.csv')
        features_df.to_csv(features_path, index=False)
        
        self.logger.info(f"Saved features to {features_path}")
        
        return features_path
    
    def fuck_yeah(self) -> str:
        """
        Returns a success message.
        
        Returns:
            Success message
        """
        return "FUCK YEAH!"

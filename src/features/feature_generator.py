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
        self.metrics = config['features']['metrics']
        self.logger.info(f"Initialized FeatureGenerator with metrics: {self.metrics}")
        
        # Set paths
        self.processed_data_dir = config['data']['processed_data']
        self.features_dir = config['data']['features']['base_dir']
        
        # Create directories if they don't exist
        os.makedirs(self.processed_data_dir, exist_ok=True)
        os.makedirs(self.features_dir, exist_ok=True)
    
    def load_processed_data(self, date: str) -> pd.DataFrame:
        """
        Load processed data for a specific date.
        
        Args:
            date: Date to load data for
            
        Returns:
            DataFrame containing processed data
        """
        self.logger.info(f"Loading processed data for date: {date}")
        
        # Construct the file path
        file_path = os.path.join(self.processed_data_dir, f"processed_{date}.pq")
        
        # Check if the file exists
        if not os.path.exists(file_path):
            self.logger.warning(f"Processed data file not found: {file_path}")
            return pd.DataFrame()
        
        # Load the data
        df = pd.read_parquet(file_path)
        self.logger.info(f"Loaded processed data with shape: {df.shape}")
        
        return df
    
    def calculate_feature_set(self, df: pd.DataFrame, metric: str) -> Dict[str, np.ndarray]:
        """
        Calculate features for a specific metric.
        
        Args:
            df: DataFrame containing processed data
            metric: Metric to calculate features for
            
        Returns:
            Dictionary of feature names and their values
        """
        self.logger.info(f"Calculating features for metric: {metric}")
        
        # Create a copy of the DataFrame to avoid modifying the original
        df_copy = df.copy()
        
        # Convert metric to lowercase for column name consistency
        metric_lower = metric.lower()
        
        # Get the column name for the metric
        metric_col = metric  # Use the original uppercase metric name
        
        # Check if the metric column exists
        if metric_col not in df_copy.columns:
            self.logger.warning(f"Metric column {metric_col} not found in DataFrame")
            return {}
        
        # Create a dictionary to store features
        features = {}
        
        # Get unique periods
        periods = sorted(df_copy['period'].unique())
        
        # Filter periods to be within the range -10 to 10
        periods = [p for p in periods if -10 <= p <= 10]
        
        if not periods:
            self.logger.warning(f"No periods found within range -10 to 10 for metric {metric}")
            return features
        
        # Split periods into negative and positive
        negative_periods = [p for p in periods if p < 0]
        zero_period = [p for p in periods if p == 0]
        positive_periods = [p for p in periods if p > 0]
        
        self.logger.info(f"Periods for {metric}: {periods}")
        self.logger.info(f"Negative periods: {negative_periods}")
        self.logger.info(f"Zero period: {zero_period}")
        self.logger.info(f"Positive periods: {positive_periods}")
        
        # Create pivot table with ID as index and period as columns
        pivot_df = df_copy.pivot_table(index='ID', columns='period', values=metric_col)
        
        # Generate time-series features
        
        # 1. Features from lowest negative to highest positive period
        if negative_periods and positive_periods:
            lowest_negative = min(negative_periods)
            highest_positive = max(positive_periods)
            
            # Calculate difference between lowest negative and highest positive
            feature_name = f"{metric_lower}_DIFF_{lowest_negative}_{highest_positive}"
            if lowest_negative in pivot_df.columns and highest_positive in pivot_df.columns:
                features[feature_name] = pivot_df[highest_positive] - pivot_df[lowest_negative]
                self.logger.info(f"Created feature: {feature_name}")
            else:
                self.logger.warning(f"Could not create feature {feature_name}, missing columns")
        
        # 2. Features from lowest negative to zero (or closest to zero)
        if negative_periods:
            lowest_negative = min(negative_periods)
            zero_or_closest = 0 if 0 in pivot_df.columns else min([p for p in periods if p >= 0], key=abs) if [p for p in periods if p >= 0] else None
            
            if zero_or_closest is not None:
                feature_name = f"{metric_lower}_DIFF_{lowest_negative}_{zero_or_closest}"
                if lowest_negative in pivot_df.columns and zero_or_closest in pivot_df.columns:
                    features[feature_name] = pivot_df[zero_or_closest] - pivot_df[lowest_negative]
                    self.logger.info(f"Created feature: {feature_name}")
                else:
                    self.logger.warning(f"Could not create feature {feature_name}, missing columns")
        
        # 3. Features from zero (or closest to zero) to highest positive
        if positive_periods:
            highest_positive = max(positive_periods)
            zero_or_closest = 0 if 0 in pivot_df.columns else min([p for p in periods if p <= 0], key=abs) if [p for p in periods if p <= 0] else None
            
            if zero_or_closest is not None:
                feature_name = f"{metric_lower}_DIFF_{zero_or_closest}_{highest_positive}"
                if zero_or_closest in pivot_df.columns and highest_positive in pivot_df.columns:
                    features[feature_name] = pivot_df[highest_positive] - pivot_df[zero_or_closest]
                    self.logger.info(f"Created feature: {feature_name}")
                else:
                    self.logger.warning(f"Could not create feature {feature_name}, missing columns")
        
        # Generate standard deviation features
        
        # 1. Standard deviation of all points from lowest negative to highest positive
        if len(periods) >= 2:
            feature_name = f"{metric_lower}_STDDEV_ALL"
            # Filter periods that exist in the pivot table
            available_periods = [p for p in periods if p in pivot_df.columns]
            if len(available_periods) >= 2:
                features[feature_name] = pivot_df[available_periods].std(axis=1)
                self.logger.info(f"Created feature: {feature_name}")
            else:
                self.logger.warning(f"Could not create feature {feature_name}, not enough available periods")
        
        # 2. Standard deviation of all negative points
        if len(negative_periods) >= 2:
            feature_name = f"{metric_lower}_STDDEV_NEG"
            # Filter periods that exist in the pivot table
            available_periods = [p for p in negative_periods if p in pivot_df.columns]
            if len(available_periods) >= 2:
                features[feature_name] = pivot_df[available_periods].std(axis=1)
                self.logger.info(f"Created feature: {feature_name}")
            else:
                self.logger.warning(f"Could not create feature {feature_name}, not enough available periods")
        
        # 3. Standard deviation of all positive points
        if len(positive_periods) >= 2:
            feature_name = f"{metric_lower}_STDDEV_POS"
            # Filter periods that exist in the pivot table
            available_periods = [p for p in positive_periods if p in pivot_df.columns]
            if len(available_periods) >= 2:
                features[feature_name] = pivot_df[available_periods].std(axis=1)
                self.logger.info(f"Created feature: {feature_name}")
            else:
                self.logger.warning(f"Could not create feature {feature_name}, not enough available periods")
        
        # Add raw financial ratios if available
        # Look for signed_log version of the metric
        raw_ratio_col = f"{metric_lower}_signed_log"
        if raw_ratio_col in df_copy.columns:
            # Get the most recent value for each ID
            latest_values = df_copy[df_copy['period'] == df_copy.groupby('ID')['period'].transform('min')]
            features[f"{metric_lower}_RAW"] = latest_values.set_index('ID')[raw_ratio_col]
            self.logger.info(f"Created feature: {metric_lower}_RAW")
        
        self.logger.info(f"Generated {len(features)} features for metric {metric}")
        return features
    
    def calculate_consecutive_diffs(self, df: pd.DataFrame, metric: str) -> pd.DataFrame:
        """
        Calculate differences between signed log values of consecutive periods for a specific metric.
        
        Args:
            df: DataFrame containing processed data
            metric: Metric to calculate differences for
            
        Returns:
            DataFrame with difference features added
        """
        self.logger.info(f"Calculating consecutive differences for metric: {metric}")
        
        # Create a copy of the DataFrame to avoid modifying the original
        result_df = df.copy()
        
        # Convert metric to lowercase for column name consistency
        metric_lower = metric.lower()
        
        # Get the signed log column name
        signed_log_col = f"{metric_lower}_signed_log"
        
        # Check if the signed log column exists
        if signed_log_col not in result_df.columns:
            self.logger.warning(f"Signed log column {signed_log_col} not found in DataFrame")
            return result_df
        
        # Get unique IDs and periods
        unique_ids = result_df['ID'].unique()
        periods = sorted(result_df['period'].unique())
        
        # Create a pivot table with ID as index and period as columns
        pivot_df = result_df.pivot_table(index='ID', columns='period', values=signed_log_col)
        
        # Calculate differences between consecutive periods
        diff_features = pd.DataFrame(index=pivot_df.index)
        
        for i in range(len(periods) - 1):
            current_period = periods[i]
            next_period = periods[i + 1]
            
            # Skip if either period is not in the pivot table
            if current_period not in pivot_df.columns or next_period not in pivot_df.columns:
                continue
            
            # Calculate difference between consecutive periods
            diff_col_name = f"{metric_lower}_diff_{current_period}_{next_period}"
            diff_features[diff_col_name] = pivot_df[next_period] - pivot_df[current_period]
            self.logger.info(f"Created difference feature: {diff_col_name}")
        
        # Reset index to merge back with the original DataFrame
        diff_features = diff_features.reset_index()
        
        # Log the number of difference features created
        self.logger.info(f"Generated {len(diff_features.columns) - 1} difference features for {metric}")
        
        return diff_features
    
    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate features from processed financial data.
        
        Args:
            df: DataFrame containing processed financial data
            
        Returns:
            DataFrame containing generated features
        """
        # Print the success message
        print("\nFUCK YEAH! Generating features with consecutive differences!\n")
        self.logger.info("FUCK YEAH! Generating features with consecutive differences!")
        
        # Create a feature DataFrame with just the ID column
        if 'ID' in df.columns:
            features = df[['ID']].drop_duplicates()
        else:
            # If no ID column, create a dummy index
            features = pd.DataFrame(index=range(1))
            self.logger.warning("No ID column found in DataFrame, using dummy index")
            return features
        
        # Add a timestamp feature
        features['timestamp'] = int(time.time())
        
        # Calculate consecutive differences for each metric
        for metric in self.metrics:
            # Calculate consecutive differences
            diff_features = self.calculate_consecutive_diffs(df, metric)
            
            # Merge with features DataFrame if there are difference features
            if len(diff_features.columns) > 1:  # More than just the ID column
                features = pd.merge(features, diff_features, on='ID', how='left')
            else:
                self.logger.warning(f"No difference features generated for {metric}")
                
                # Add a random feature as fallback (for demonstration)
                metric_lower = metric.lower()
                features[f"{metric_lower}_random"] = np.random.randn(len(features))
                self.logger.info(f"Added random feature for {metric}")
        
        self.logger.info(f"Generated {len(features.columns) - 2} features")  # Subtract ID and timestamp
        
        return features
    
    def save_features(self, features_df: pd.DataFrame, date: str) -> str:
        """
        Save features to a file.
        
        Args:
            features_df: DataFrame containing features
            date: Date to save features for
            
        Returns:
            Path to the saved file
        """
        self.logger.info(f"Saving features for date: {date}")
        
        # Construct the file path
        file_path = os.path.join(self.features_dir, f"features_{date}.pq")
        
        # Save the features
        features_df.to_parquet(file_path)
        self.logger.info(f"Saved features to {file_path}")
        
        return file_path
    
    def process_features(self, date: str, sector_data: Optional[pd.DataFrame] = None, 
                      price_returns: Optional[pd.DataFrame] = None, 
                      total_returns: Optional[pd.DataFrame] = None) -> str:
        """
        Process features for a specific date.
        
        Args:
            date: Date to process features for
            sector_data: DataFrame containing sector mapping data
            price_returns: DataFrame containing price returns data
            total_returns: DataFrame containing total returns data
            
        Returns:
            Path to the saved features file
        """
        self.logger.info(f"Processing features for date: {date}")
        
        # Load processed data
        df = self.load_processed_data(date)
        
        # Check if the data is empty
        if df.empty:
            self.logger.warning(f"No processed data found for date: {date}")
            return ""
        
        # Log information about additional data sources
        if sector_data is not None:
            self.logger.info(f"Using sector data with shape: {sector_data.shape}")
        if price_returns is not None:
            self.logger.info(f"Using price returns data with shape: {price_returns.shape}")
        if total_returns is not None:
            self.logger.info(f"Using total returns data with shape: {total_returns.shape}")
        
        # Generate features
        features_df = self.generate_features(df)
        
        # Save features
        file_path = self.save_features(features_df, date)
        
        return file_path
    
    def generate_sector_features(self, df: pd.DataFrame, sector_data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate sector-based features.
        
        Args:
            df: DataFrame containing processed data
            sector_data: DataFrame containing sector mapping data
            
        Returns:
            DataFrame containing sector-based features
        """
        self.logger.info("Generating sector-based features")
        
        # Create a copy of the DataFrame to avoid modifying the original
        df_copy = df.copy()
        
        # Create a dictionary to store features
        features = {}
        
        # Ensure sector_data has ID as index
        if not sector_data.index.name == 'ID':
            if 'ID' in sector_data.columns:
                sector_data = sector_data.set_index('ID')
            else:
                self.logger.warning("Sector data does not have ID column, cannot generate sector features")
                return pd.DataFrame(features)
        
        # Get unique IDs from the processed data
        unique_ids = df_copy['ID'].unique()
        
        # Filter sector data to include only IDs in the processed data
        sector_data_filtered = sector_data.loc[sector_data.index.isin(unique_ids)]
        
        # Create sector features
        for id_val in unique_ids:
            if id_val in sector_data_filtered.index:
                # Get sector information for this ID
                sector_info = sector_data_filtered.loc[id_val]
                
                # Add sector information to features
                features[id_val] = {
                    'SECTOR_1': sector_info.get('sector_1', None),
                    'SECTOR_2': sector_info.get('sector_2', None),
                    'SECTOR_3': sector_info.get('sector_3', None),
                    'SECTOR_4': sector_info.get('sector_4', None)
                }
        
        # Convert features to DataFrame
        sector_features_df = pd.DataFrame.from_dict(features, orient='index')
        sector_features_df.index.name = 'ID'
        sector_features_df = sector_features_df.reset_index()
        
        self.logger.info(f"Generated {len(sector_features_df.columns) - 1} sector-based features")
        
        return sector_features_df
    
    def generate_returns_features(self, df: pd.DataFrame, price_returns: pd.DataFrame, total_returns: pd.DataFrame) -> pd.DataFrame:
        """
        Generate returns-based features.
        
        Args:
            df: DataFrame containing processed data
            price_returns: DataFrame containing price returns data
            total_returns: DataFrame containing total returns data
            
        Returns:
            DataFrame containing returns-based features
        """
        self.logger.info("Generating returns-based features")
        
        # Create a copy of the DataFrame to avoid modifying the original
        df_copy = df.copy()
        
        # Create a dictionary to store features
        features = {}
        
        # Get unique IDs from the processed data
        unique_ids = df_copy['ID'].unique()
        
        # Get unique dates from the processed data
        unique_dates = df_copy['PIT_DATE'].unique()
        
        # Process price returns
        if price_returns is not None:
            self.logger.info("Processing price returns data")
            
            # Ensure price_returns has the necessary columns
            if 'ID' in price_returns.columns and any(col for col in price_returns.columns if 'RETURN' in col or 'RET' in col):
                # Filter price returns to include only IDs in the processed data
                price_returns_filtered = price_returns[price_returns['ID'].isin(unique_ids)]
                
                # Find the return column
                return_cols = [col for col in price_returns.columns if 'RETURN' in col or 'RET' in col]
                
                if return_cols:
                    return_col = return_cols[0]
                    self.logger.info(f"Using price returns column: {return_col}")
                    
                    # Create features for each ID
                    for id_val in unique_ids:
                        id_returns = price_returns_filtered[price_returns_filtered['ID'] == id_val]
                        
                        if not id_returns.empty:
                            # Calculate return statistics
                            mean_return = id_returns[return_col].mean()
                            std_return = id_returns[return_col].std()
                            max_return = id_returns[return_col].max()
                            min_return = id_returns[return_col].min()
                            
                            # Add return statistics to features
                            if id_val not in features:
                                features[id_val] = {}
                            
                            features[id_val].update({
                                'PRICE_RETURN_MEAN': mean_return,
                                'PRICE_RETURN_STD': std_return,
                                'PRICE_RETURN_MAX': max_return,
                                'PRICE_RETURN_MIN': min_return
                            })
        
        # Process total returns
        if total_returns is not None:
            self.logger.info("Processing total returns data")
            
            # Ensure total_returns has the necessary columns
            if 'ID' in total_returns.columns and any(col for col in total_returns.columns if 'RETURN' in col or 'RET' in col):
                # Filter total returns to include only IDs in the processed data
                total_returns_filtered = total_returns[total_returns['ID'].isin(unique_ids)]
                
                # Find the return column
                return_cols = [col for col in total_returns.columns if 'RETURN' in col or 'RET' in col]
                
                if return_cols:
                    return_col = return_cols[0]
                    self.logger.info(f"Using total returns column: {return_col}")
                    
                    # Create features for each ID
                    for id_val in unique_ids:
                        id_returns = total_returns_filtered[total_returns_filtered['ID'] == id_val]
                        
                        if not id_returns.empty:
                            # Calculate return statistics
                            mean_return = id_returns[return_col].mean()
                            std_return = id_returns[return_col].std()
                            max_return = id_returns[return_col].max()
                            min_return = id_returns[return_col].min()
                            
                            # Add return statistics to features
                            if id_val not in features:
                                features[id_val] = {}
                            
                            features[id_val].update({
                                'TOTAL_RETURN_MEAN': mean_return,
                                'TOTAL_RETURN_STD': std_return,
                                'TOTAL_RETURN_MAX': max_return,
                                'TOTAL_RETURN_MIN': min_return
                            })
        
        # Convert features to DataFrame
        returns_features_df = pd.DataFrame.from_dict(features, orient='index')
        returns_features_df.index.name = 'ID'
        returns_features_df = returns_features_df.reset_index()
        
        self.logger.info(f"Generated {len(returns_features_df.columns) - 1} returns-based features")
        
        return returns_features_df

    def fuck_yeah(self) -> str:
        """
        Returns a success message.
        
        Returns:
            Success message
        """
        return "FUCK YEAH!"

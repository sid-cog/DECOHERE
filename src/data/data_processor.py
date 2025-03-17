#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data processor for loading and transforming raw financial data.
"""

import os
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from src.data.efficient_data_storage import EfficientDataStorage
from src.data.efficient_data_storage import EfficientDataStorage
from pathlib import Path


class DataProcessor:
    """
    Data processor for loading and transforming raw financial data.
    """
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Initialize the data processor.
        
        Args:
            config: Configuration dictionary
            logger: Logger to use (if None, create a new logger)
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Extract configuration parameters
        self.raw_data_path = config['data']['raw_data']
        self.processed_data_dir = config['data']['processed_data']
        self.enable_filling = config['processing'].get('enable_filling', True)
        self.winsorize_threshold = config['processing'].get('winsorize_threshold', 3.0)
        
        # Data source paths
        self.financials_dir = config['data'].get('financials_dir', '/home/siddharth_johri/projects/data/financials')
        self.sector_mapping_path = config['data'].get('sector_mapping', '/home/siddharth_johri/projects/data/sector/sector.pq')
        self.price_returns_path = config['data'].get('price_returns', '/home/siddharth_johri/projects/data/returns/px_df.pq')
        self.total_returns_path = config['data'].get('total_returns', '/home/siddharth_johri/projects/data/returns/tr_df.pq')
        
        # New storage structure paths
        self.fundamentals_dir = os.path.join(self.processed_data_dir, "fundamentals")
        self.returns_dir = os.path.join(self.processed_data_dir, "returns")
        
        # Create directories if they don't exist
        os.makedirs(self.processed_data_dir, exist_ok=True)
        os.makedirs(self.fundamentals_dir, exist_ok=True)
        os.makedirs(self.returns_dir, exist_ok=True)
        
        # Initialize efficient data storage
        self.data_storage = EfficientDataStorage(config, logger)
        
        # Initialize efficient data storage
        self.data_storage = EfficientDataStorage(config, logger)
    
    def load_raw_data(self, date: Optional[str] = None) -> pd.DataFrame:
        """
        Load raw financial data.
        
        Args:
            date: Date to filter data for (if None, load all data)
            
        Returns:
            DataFrame containing raw financial data
        """
        self.logger.info(f"Loading raw data from {self.raw_data_path}")
        
        # Load raw data
        df = pd.read_parquet(self.raw_data_path)
        
        # Normalize column names to uppercase for consistency
        df.columns = [col.upper() for col in df.columns]
        
        # Filter by date if specified
        if date:
            self.logger.info(f"Filtering data for date: {date}")
            df = df[df['PIT_DATE'] == date]
        
        # Apply ticker limit if specified
        limit_tickers = self.config.get('filter', {}).get('limit_tickers')
        if limit_tickers:
            self.logger.info(f"Limiting to {limit_tickers} tickers")
            unique_tickers = df['ID'].unique()
            limited_tickers = unique_tickers[:limit_tickers]
            df = df[df['ID'].isin(limited_tickers)]
        
        self.logger.info(f"Loaded raw data with shape: {df.shape}")
        
        return df
    
    def signed_log_transform(self, x: np.ndarray) -> np.ndarray:
        """
        Apply signed log transform to handle skewed data.
        
        Args:
            x: Array of values to transform
            
        Returns:
            Transformed array
        """
        if isinstance(x, pd.Series):
            x = x.values
        
        # Handle NaN values
        mask = np.isnan(x)
        result = np.empty_like(x)
        result[mask] = np.nan
        
        # Apply transform to non-NaN values
        valid_mask = ~mask
        valid_x = x[valid_mask]
        result[valid_mask] = np.sign(valid_x) * np.log1p(np.abs(valid_x))
        
        return result
    
    def transform_raw_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform raw financial data.
        
        Args:
            df: DataFrame containing raw financial data
            
        Returns:
            DataFrame containing transformed data
        """
        self.logger.info("Transforming raw data")
        
        # Create a copy of the DataFrame to avoid modifying the original
        transformed_df = df.copy()
        
        # Get metrics to transform from config
        metrics = self.config['features']['metrics']
        
        # Apply signed log transform to each metric
        for metric in metrics:
            if metric in transformed_df.columns:
                self.logger.debug(f"Applying signed log transform to {metric}")
                transformed_df[f"{metric.lower()}_signed_log"] = self.signed_log_transform(transformed_df[metric])
        
        # Calculate periods for each ticker
        transformed_df = self.calculate_periods(transformed_df)
        
        # Fill missing values if enabled
        if self.enable_filling:
            self.logger.info("Filling missing values")
            transformed_df = self.fill_missing_values(transformed_df)
        
        # Winsorize features to handle outliers
        transformed_df = self.winsorize_features(transformed_df)
        
        self.logger.info(f"Transformed data with shape: {transformed_df.shape}")
        
        return transformed_df
    
    def calculate_periods(self, df: pd.DataFrame, id_col: str = 'ID', 
                         date_col: str = 'PIT_DATE', period_end_col: str = 'PERIOD_END_DATE') -> pd.DataFrame:
        """
        Calculate periods for each ticker based on the COHERE implementation.
        
        Args:
            df: DataFrame containing raw financial data
            id_col: Column name for ticker ID
            date_col: Column name for date
            period_end_col: Column name for period end date
            
        Returns:
            DataFrame with period column added
        """
        self.logger.info("Calculating periods for each ticker using COHERE logic")
        
        # Create a copy of the DataFrame to avoid modifying the original
        result_df = df.copy()
        
        # Ensure date columns are datetime
        result_df[date_col] = pd.to_datetime(result_df[date_col])
        result_df[period_end_col] = pd.to_datetime(result_df[period_end_col])
        
        # Extract fiscal month from period end date
        result_df['fiscal_month'] = result_df[period_end_col].dt.month
        
        # Define the period calculation function
        def calculate_periods_for_group(group):
            # Sort by period end date
            group = group.sort_values(period_end_col)
            
            # Get the fiscal year end month (most common month in the group)
            fiscal_month = group['fiscal_month'].mode()[0]
            
            # Drop rows where the PERIOD_END_DATE month is not the fiscal month
            group = group[group['fiscal_month'] == fiscal_month]
            
            # If group is empty after filtering, return empty DataFrame
            if group.empty:
                return group
            
            # For each period end date, calculate how many periods away it is from pit_date
            pit_date = group[date_col].iloc[0]
            
            # Calculate years difference
            years_diff = (group[period_end_col].dt.year - pit_date.year)
            
            # Adjust period based on fiscal month
            # If fiscal month is after pit_date month, subtract 1 from future periods
            # If fiscal month is before pit_date month, add 1 to past periods
            month_adjustment = ((group[period_end_col].dt.month == fiscal_month) & 
                              ((fiscal_month > pit_date.month) & (years_diff >= 0) |
                               (fiscal_month < pit_date.month) & (years_diff <= 0)))
            
            group['period'] = years_diff - month_adjustment.astype(int)
            return group
        
        # Apply period calculation by ID
        self.logger.info(f"Calculating periods by ID using id column: '{id_col}'")
        result_df = result_df.groupby(id_col).apply(calculate_periods_for_group).reset_index(drop=True)
        
        # Log period counts
        period_counts = result_df['period'].value_counts().sort_index()
        self.logger.info(f"Period counts: {dict(period_counts)}")
        self.logger.info(f"Available periods: {sorted(result_df['period'].unique())}")
        
        return result_df
    def fill_missing_values(self, df: pd.DataFrame, group_col: str = 'ID') -> pd.DataFrame:
        """
        Fill missing values in the DataFrame.
        
        Args:
            df: DataFrame containing data with missing values
            group_col: Column name to group by
            
        Returns:
            DataFrame with missing values filled
        """
        self.logger.info("Filling missing values")
        
        # Create a copy of the DataFrame
        result_df = df.copy()
        
        # Get columns to fill (exclude ID, date, and period columns)
        exclude_cols = [group_col, 'PERIOD_END_DATE', 'period']  # Keep PIT_DATE in the processed data
        fill_cols = [col for col in result_df.columns if col not in exclude_cols]
        
        # Fill missing values for each group
        for col in fill_cols:
            # Skip columns with no missing values
            if not result_df[col].isna().any():
                continue
            
            # Fill missing values with interpolation within each group
            result_df[col] = result_df.groupby(group_col)[col].transform(
                lambda x: self._fill_between_valid_values(x)
            )
        
        # Log missing value counts after filling
        missing_counts = result_df[fill_cols].isna().sum()
        self.logger.debug(f"Missing value counts after filling: {missing_counts[missing_counts > 0].to_dict()}")
        
        return result_df
    
    def _fill_between_valid_values(self, series: pd.Series) -> pd.Series:
        """
        Fill NaN values between valid values with equally spaced values.
        
        Args:
            series: Series to fill
            
        Returns:
            Series with NaN values filled
        """
        # Create a copy of the series to avoid modifying the original
        result = series.copy()
        
        # Find indices of NaN values
        nan_indices = series.index[series.isna()].tolist()
        
        # If no NaN values, return the original series
        if not nan_indices:
            return result
        
        # Group consecutive NaN indices
        nan_groups = []
        current_group = [nan_indices[0]]
        
        for i in range(1, len(nan_indices)):
            # Check if the current index is consecutive with the previous one
            # For integer indices, we need to check if the difference is 1
            # For other index types, we need to check if there are no values in between
            if isinstance(nan_indices[i], (int, np.integer)) and isinstance(nan_indices[i-1], (int, np.integer)):
                if nan_indices[i] == nan_indices[i-1] + 1:
                    current_group.append(nan_indices[i])
                else:
                    nan_groups.append(current_group)
                    current_group = [nan_indices[i]]
            else:
                # For non-integer indices, we need to check if there are no values in between
                # This is a simplification and may not work for all index types
                nan_groups.append(current_group)
                current_group = [nan_indices[i]]
        
        # Add the last group
        if current_group:
            nan_groups.append(current_group)
        
        # Fill each group of NaN values
        for group in nan_groups:
            start_idx = group[0]
            end_idx = group[-1]
            
            # Get the indices of the series
            series_indices = series.index.tolist()
            
            # Find the index of start_idx and end_idx in the series indices
            try:
                start_pos = series_indices.index(start_idx)
                end_pos = series_indices.index(end_idx)
                
                # Only process if we have valid values on both sides
                if start_pos > 0 and end_pos < len(series_indices) - 1:
                    # Get the values before and after the NaN sequence
                    start_val = series[series_indices[start_pos - 1]]
                    end_val = series[series_indices[end_pos + 1]]
                    
                    if pd.notna(start_val) and pd.notna(end_val):
                        # Calculate equally spaced values
                        num_points = len(group) + 2  # Include start and end points
                        step = (end_val - start_val) / (num_points - 1)
                        
                        # Fill the NaN values with equally spaced values
                        for i, idx in enumerate(group, 1):
                            result[idx] = start_val + step * i
            except (ValueError, IndexError):
                # Skip if we can't find the indices or if there's an error
                continue
        
        return result
    def winsorize_features(self, df: pd.DataFrame, threshold: Optional[float] = None) -> pd.DataFrame:
        """
        Winsorize features to handle outliers.
        
        Args:
            df: DataFrame containing features
            threshold: Z-score threshold for winsorization (if None, use config value)
            
        Returns:
            DataFrame with winsorized features
        """
        threshold = threshold or self.winsorize_threshold
        self.logger.info(f"Winsorizing features with threshold: {threshold}")
        
        # Create a copy of the DataFrame
        result_df = df.copy()
        
        # Get columns to winsorize (exclude ID, date, and period columns)
        exclude_cols = ['ID', 'PIT_DATE', 'PERIOD_END_DATE', 'period']
        winsorize_cols = [col for col in result_df.columns if col not in exclude_cols]
        
        # Winsorize each column
        for col in winsorize_cols:
            # Skip columns with no values
            if result_df[col].isna().all():
                continue
            
            # Calculate mean and standard deviation
            mean = result_df[col].mean()
            std = result_df[col].std()
            
            if std == 0:
                continue
            
            # Calculate z-scores
            z_scores = (result_df[col] - mean) / std
            
            # Identify outliers
            outliers = (z_scores.abs() > threshold)
            outlier_count = outliers.sum()
            
            if outlier_count > 0:
                self.logger.debug(f"Winsorizing {outlier_count} outliers in {col}")
                
                # Winsorize outliers
                result_df.loc[z_scores > threshold, col] = mean + threshold * std
                result_df.loc[z_scores < -threshold, col] = mean - threshold * std
        
        return result_df
    
    def save_processed_data(self, df: pd.DataFrame, date: str) -> str:
        """
        Save processed data to a file.
        
        Args:
            df: DataFrame containing processed data
            date: Date of the data
            
        Returns:
            Path to the saved file
        """
        # Use efficient data storage to store the data
        output_path = self.data_storage.store_processed_data(df, date)
        
        # For backward compatibility, also return the legacy file path
        legacy_file_path = os.path.join(self.processed_data_dir, f"processed_{date}.pq")
        
        self.logger.info(f"Saved processed data using efficient storage")
        
        return legacy_file_path
    
    def process_data(self, start_date: str, end_date: str) -> Dict[str, str]:
        """
        Process data for a date range.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            Dictionary mapping dates to processed data file paths
        """
        self.logger.info(f"Processing data for date range: {start_date} to {end_date}")
        
        # Generate list of dates
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        date_strs = [date.strftime('%Y-%m-%d') for date in dates]
        
        # Process each date
        processed_files = {}
        
        for date_str in date_strs:
            self.logger.info(f"Processing data for date: {date_str}")
            
            # Load raw data for the date
            raw_df = self.load_raw_data(date_str)
            
            # Skip if no data for this date
            if len(raw_df) == 0:
                self.logger.warning(f"No data found for date: {date_str}")
                continue
            
            # Transform raw data
            transformed_df = self.transform_raw_data(raw_df)
            
            # Save processed data
            output_file = self.save_processed_data(transformed_df, date_str)
            
            # Add to processed files dictionary
            processed_files[date_str] = output_file
        
        self.logger.info(f"Processed {len(processed_files)} dates")
        
        return processed_files
        
    def load_processed_data_by_mode(self, mode: str = 'day', date: str = None, 
                                   start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Load processed data based on the specified mode.
        
        Args:
            mode: Mode to load data for ('day', 'week', 'year', or 'all_years')
            date: Specific date for 'day' mode
            start_date: Start date for range modes
            end_date: End date for range modes
            
        Returns:
            DataFrame containing processed data
        """
        self.logger.info(f"Loading processed data in {mode} mode")
        
        # Use efficient data storage to load the data
        return self.data_storage.load_processed_data(
            date=date,
            mode=mode,
            start_date=start_date,
            end_date=end_date
        )
    
    def migrate_to_efficient_storage(self) -> bool:
        """
        Migrate all legacy processed data files to the efficient storage format.
        
        Returns:
            True if migration was successful, False otherwise
        """
        self.logger.info("Migrating to efficient storage format")
        return self.data_storage.migrate_legacy_to_partitioned()
    
    def get_available_dates(self) -> List[str]:
        """
        Get a list of all available dates in the processed data.
        
        Returns:
            List of dates (YYYY-MM-DD)
        """
        return self.data_storage.get_available_dates()
        
    def load_price_returns(self) -> pd.DataFrame:
        """
        Load price returns data.
        
        Returns:
            DataFrame containing price returns data
        """
        # Try to load from the new returns directory first
        price_returns_file = os.path.join(self.returns_dir, "price_returns.pq")
        
        if os.path.exists(price_returns_file):
            self.logger.info(f"Loading price returns data from {price_returns_file}")
            df = pd.read_parquet(price_returns_file)
            self.logger.info(f"Loaded price returns data with shape: {df.shape}")
            return df
        
        # Try to load from combined returns file
        combined_returns_file = os.path.join(self.returns_dir, "combined_returns.pq")
        
        if os.path.exists(combined_returns_file):
            self.logger.info(f"Loading price returns data from combined returns file: {combined_returns_file}")
            df = pd.read_parquet(combined_returns_file)
            
            # Check if the file contains price returns
            price_ret_cols = [col for col in df.columns if 'PRICE_RET' in col]
            
            if price_ret_cols:
                self.logger.info(f"Found price returns columns: {price_ret_cols}")
                return df
        
        # Fall back to the original price returns file if defined
        if hasattr(self, 'price_returns_path'):
            self.logger.info(f"Loading price returns data from original path: {self.price_returns_path}")
            
            try:
                # Load price returns data
                df = pd.read_parquet(self.price_returns_path)
                self.logger.info(f"Loaded price returns data with shape: {df.shape}")
                return df
            except Exception as e:
                self.logger.error(f"Error loading price returns data: {e}")
        
        # Return empty DataFrame if no data found
        self.logger.warning("No price returns data found")
        return pd.DataFrame()
        
    def load_total_returns(self) -> pd.DataFrame:
        """
        Load total returns data.
        
        Returns:
            DataFrame containing total returns data
        """
        # Try to load from the new returns directory first
        total_returns_file = os.path.join(self.returns_dir, "total_returns.pq")
        
        if os.path.exists(total_returns_file):
            self.logger.info(f"Loading total returns data from {total_returns_file}")
            df = pd.read_parquet(total_returns_file)
            self.logger.info(f"Loaded total returns data with shape: {df.shape}")
            return df
        
        # Try to load from combined returns file
        combined_returns_file = os.path.join(self.returns_dir, "combined_returns.pq")
        
        if os.path.exists(combined_returns_file):
            self.logger.info(f"Loading total returns data from combined returns file: {combined_returns_file}")
            df = pd.read_parquet(combined_returns_file)
            
            # Check if the file contains total returns
            total_ret_cols = [col for col in df.columns if 'TOTAL_RET' in col]
            
            if total_ret_cols:
                self.logger.info(f"Found total returns columns: {total_ret_cols}")
                return df
        
        # Fall back to the original total returns file if defined
        if hasattr(self, 'total_returns_path'):
            self.logger.info(f"Loading total returns data from original path: {self.total_returns_path}")
            
            try:
                # Load total returns data
                df = pd.read_parquet(self.total_returns_path)
                self.logger.info(f"Loaded total returns data with shape: {df.shape}")
                return df
            except Exception as e:
                self.logger.error(f"Error loading total returns data: {e}")
        
        # Return empty DataFrame if no data found
        self.logger.warning("No total returns data found")
        return pd.DataFrame()
        
    def load_all_data_sources(self, date: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Load all data sources.
        
        Args:
            date: Date to filter financial data for (if None, load all data)
            
        Returns:
            Dictionary containing all data sources
        """
        self.logger.info("Loading all data sources...")
        
        data = {}
        
        # Load financial data
        data['financials'] = self.load_raw_data(date)
        
        # Try to load sector mapping data
        try:
            # Try to extract sector data from fundamentals first
            fundamentals_files = [f for f in os.listdir(self.fundamentals_dir) 
                                if f.startswith('fundamentals_') and f.endswith('.pq')]
            
            if fundamentals_files:
                # Use the most recent fundamentals file
                latest_file = sorted(fundamentals_files)[-1]
                fundamentals_file = os.path.join(self.fundamentals_dir, latest_file)
                self.logger.info(f"Extracting sector data from fundamentals file: {fundamentals_file}")
                
                df = pd.read_parquet(fundamentals_file)
                
                # Extract sector columns
                sector_cols = [col for col in df.columns if 'SECTOR_' in col]
                
                if sector_cols:
                    self.logger.info(f"Found sector columns: {sector_cols}")
                    sector_df = df[['ID'] + sector_cols].drop_duplicates()
                    self.logger.info(f"Extracted sector data with shape: {sector_df.shape}")
                    data['sector_mapping'] = sector_df
                else:
                    self.logger.warning("No sector columns found in fundamentals data")
                    data['sector_mapping'] = pd.DataFrame()
            else:
                self.logger.warning("No fundamentals files found")
                data['sector_mapping'] = pd.DataFrame()
        except Exception as e:
            self.logger.error(f"Error loading sector mapping data: {e}")
            data['sector_mapping'] = pd.DataFrame()
        
        # Try to load returns data
        try:
            # Try to load from the new returns directory first
            price_returns_file = os.path.join(self.returns_dir, "price_returns.pq")
            
            if os.path.exists(price_returns_file):
                self.logger.info(f"Loading price returns data from {price_returns_file}")
                df = pd.read_parquet(price_returns_file)
                self.logger.info(f"Loaded price returns data with shape: {df.shape}")
                data['price_returns'] = df
            else:
                self.logger.warning(f"Price returns file not found: {price_returns_file}")
                data['price_returns'] = pd.DataFrame()
        except Exception as e:
            self.logger.error(f"Error loading price returns data: {e}")
            data['price_returns'] = pd.DataFrame()
            
        try:
            # Try to load from the new returns directory first
            total_returns_file = os.path.join(self.returns_dir, "total_returns.pq")
            
            if os.path.exists(total_returns_file):
                self.logger.info(f"Loading total returns data from {total_returns_file}")
                df = pd.read_parquet(total_returns_file)
                self.logger.info(f"Loaded total returns data with shape: {df.shape}")
                data['total_returns'] = df
            else:
                self.logger.warning(f"Total returns file not found: {total_returns_file}")
                data['total_returns'] = pd.DataFrame()
        except Exception as e:
            self.logger.error(f"Error loading total returns data: {e}")
            data['total_returns'] = pd.DataFrame()
        
        return data 
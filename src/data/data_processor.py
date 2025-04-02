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
from datetime import datetime
from src.data.data_types import DataType, DataStage


class DataProcessor:
    """
    Data processor for loading and transforming raw financial data.
    """
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Initialize the data processor.
        
        Args:
            config: Configuration dictionary
            logger: Optional logger instance. If not provided, a new logger will be created.
        """
        self.config = config
        
        # Initialize logger
        self.logger = logger if logger is not None else logging.getLogger(__name__)
        
        # Initialize data storage
        self.data_storage = EfficientDataStorage(config, logger)
        
        # Get data paths directly from config
        self.raw_data_path = config['data']['raw']['fundamentals']
        self.processed_data_dir = config['data']['processed']['fundamentals']
        self.feature_set_dir = config['data']['features']['fundamentals']
        
        # Get processing config
        processing_config = config.get('processing', {})
        self.scaling_variable = processing_config.get('scaling_variable', 'SALES')
        self.winsorization_threshold = processing_config.get('winsorization_threshold', 0.01)
        self.fill_method = processing_config.get('fill_method', 'linear')
        self.min_data_points = processing_config.get('min_data_points', 100)
        self.max_missing_ratio = processing_config.get('max_missing_ratio', 0.2)
        
        # Get feature config
        feature_config = config.get('features', {})
        self.identifier_fields = set(feature_config.get('identifier_fields', []))
        self.absolute_value_fields = set(feature_config.get('absolute_value_fields', []))
        self.standard_deviation_fields = set(feature_config.get('standard_deviation_fields', []))
        self.ratio_fields = set(feature_config.get('ratio_fields', []))
        
        # Get suffix config
        suffix_config = feature_config.get('suffixes', {})
        self.raw_suffix = suffix_config.get('raw', '_RAW')
        self.signed_log_suffix = suffix_config.get('signed_log', '_SIGNED_LOG')
        self.ratio_suffix = suffix_config.get('ratio', '_RATIO')
        self.scaled_suffix = suffix_config.get('scaled', '_SCALED_{SALES}')
        self.raw_signed_log_suffix = suffix_config.get('raw_signed_log', '_RAW_SIGNED_LOG')
        self.raw_scaled_suffix = suffix_config.get('raw_scaled', '_RAW_SCALED_{SALES}')
        self.raw_scaled_signed_log_suffix = suffix_config.get('raw_scaled_signed_log', '_RAW_SCALED_{SALES}_SIGNED_LOG')
        self.ratio_signed_log_suffix = suffix_config.get('ratio_signed_log', '_RATIO_SIGNED_LOG')
        
        # Create directories if they don't exist
        os.makedirs(self.raw_data_path, exist_ok=True)
        os.makedirs(self.processed_data_dir, exist_ok=True)
        os.makedirs(self.feature_set_dir, exist_ok=True)
        
        # Define valuation ratio fields that should not be scaled by SALES
        self.valuation_ratio_fields = [
            'PE_RATIO',
            'PREV_PE_RATIO',
            'PX_TO_BOOK_RATIO',
            'PREV_PX_TO_BOOK_RATIO'
        ]
        
        # Define operating ratio fields that should not be scaled by SALES
        self.operating_ratio_fields = [
            'INTEREST_EXPENSE_TO_TOTAL_DEBT',
            'RETURN_ON_ASSETS',
            'RETURN_COM_EQY',
            'DEBT_TO_EQUITY_RATIO',
            'NET_DEBT_TO_EQUITY_RATIO',
            'CURRENT_RATIO',
            'OPERATING_MARGIN',
            'ASSET_TURNOVER',
            'INVENTORY_TURNOVER',
            'INTEREST_COVERAGE',
            'QUICK_RATIO',
            'NET_INCOME_COEFF_OF_VAR',
            'EBIT_COEFF_OF_VAR',
            'EBITDA_COEFF_OF_VAR',
            'SALES_COEFF_OF_VAR'
        ]
    
    def load_raw_data(self, date: Optional[str] = None) -> pd.DataFrame:
        """
        Load raw financial data.
        
        Args:
            date: Date to filter data for (if None, load all data)
            
        Returns:
            DataFrame containing raw financial data
        """
        # Get base directory from config
        base_dir = self.config['data']['base_dir']
        financials_dir = self.config['data']['financials_dir']
        
        # Use the configured raw data path
        if date:
            # Extract year and month from date (YYYY-MM-DD -> YYYY-MM)
            year_month = date[:7]  # Get YYYY-MM from YYYY-MM-DD
            raw_data_path = os.path.join(base_dir, financials_dir, f"financials_{year_month.replace('-', '_')}.pq")
        else:
            raw_data_path = os.path.join(base_dir, financials_dir, "financials.pq")
            
        self.logger.info(f"Loading raw data from {raw_data_path}")
        
        # Load raw data
        df = pd.read_parquet(raw_data_path)
        
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
        Transform raw data by calculating periods and applying transformations.
        """
        # Calculate periods
        df = self.calculate_periods(df)
        
        # Ensure PERIOD column is integer type
        df['PERIOD'] = df['PERIOD'].astype(int)
        
        # Store PERIOD column separately
        period_col = df['PERIOD'].copy()
        
        # Calculate coefficient of variation fields (base calculation only)
        financial_metrics = {
            'NET_INCOME': 'NET_INCOME_CSTAT_STD',
            'EBIT': 'EBIT_CSTAT_STD',
            'EBITDA': 'EBITDA_CSTAT_STD',
            'SALES': 'SALES_CSTAT_STD'
        }
        
        for metric, std_field in financial_metrics.items():
            if metric in df.columns and std_field in df.columns:
                coeff_var_field = f'{metric}_COEFF_OF_VAR'
                # Calculate coefficient of variation (std / |value|)
                df[coeff_var_field] = df[std_field] / df[metric].abs()
                # Replace inf values with NaN
                df[coeff_var_field] = df[coeff_var_field].replace([np.inf, -np.inf], np.nan)
                # Add to operating ratio fields for further processing
                if coeff_var_field not in self.operating_ratio_fields:
                    self.operating_ratio_fields.append(coeff_var_field)

        # Define field categories
        absolute_value_fields = [
            'NET_INCOME', 'EBIT', 'EBITDA', 'SALES', 'NET_OPERATING_ASSETS',
            'INVENTORIES', 'FREE_CASH_FLOW', 'DIVIDEND', 'CAPEX', 'DEPRECIATION'
        ]
        
        standard_deviation_fields = [
            'NET_INCOME_CSTAT_STD', 'EBIT_CSTAT_STD', 'EBITDA_CSTAT_STD',
            'SALES_CSTAT_STD', 'RETURN_COM_EQY_CSTAT_STD', 'INVENTORY_TURNOVER_CSTAT_STD'
        ]
        
        # Process absolute value fields
        for field in absolute_value_fields:
            if field in df.columns:
                # Store raw value
                df[f'{field}{self.raw_suffix}'] = df[field]
                
                # Apply signed log to raw value
                df[f'{field}{self.raw_signed_log_suffix}'] = self.signed_log_transform(df[field])
                
                # Scale by SALES if not SALES itself
                if field != self.scaling_variable and self.scaling_variable in df.columns:
                    scaled_field = f'{field}{self.raw_scaled_suffix.format(SALES=self.scaling_variable)}'
                    df[scaled_field] = df[field] / df[self.scaling_variable].abs()
                    df[scaled_field] = df[scaled_field].replace([np.inf, -np.inf], np.nan)
                    
                    # Apply signed log to scaled value
                    df[f'{field}{self.raw_scaled_signed_log_suffix.format(SALES=self.scaling_variable)}'] = \
                        self.signed_log_transform(df[scaled_field])
        
        # Process standard deviation fields
        for field in standard_deviation_fields:
            if field in df.columns:
                # Store raw value
                df[f'{field}{self.raw_suffix}'] = df[field]
                
                # Apply signed log to raw value
                df[f'{field}{self.raw_signed_log_suffix}'] = self.signed_log_transform(df[field])
                
                # Get base metric name
                base_metric = field.replace('_CSTAT_STD', '')
                
                # Scale by SALES if base metric is not SALES and not in ratio fields
                if (base_metric != self.scaling_variable and 
                    base_metric not in self.operating_ratio_fields and 
                    base_metric not in self.valuation_ratio_fields):
                    scaled_field = f'{field}{self.raw_scaled_suffix.format(SALES=self.scaling_variable)}'
                    df[scaled_field] = df[field] / df[self.scaling_variable].abs()
                    df[scaled_field] = df[scaled_field].replace([np.inf, -np.inf], np.nan)
                    
                    # Apply signed log to scaled value
                    df[f'{field}{self.raw_scaled_signed_log_suffix.format(SALES=self.scaling_variable)}'] = \
                        self.signed_log_transform(df[scaled_field])
        
        # Process operating ratio fields (including coefficient of variation fields)
        for field in self.operating_ratio_fields:
            if field in df.columns:
                # Store ratio value
                df[f'{field}{self.ratio_suffix}'] = df[field]
                
                # Apply signed log to ratio
                df[f'{field}{self.ratio_signed_log_suffix}'] = self.signed_log_transform(df[field])
        
        # Process valuation ratio fields
        for field in self.valuation_ratio_fields:
            if field in df.columns:
                # Store ratio value
                df[f'{field}{self.ratio_suffix}'] = df[field]
                
                # Apply signed log to ratio
                df[f'{field}{self.ratio_signed_log_suffix}'] = self.signed_log_transform(df[field])
        
        # Handle missing values and outliers
        for field in df.columns:
            if field not in self.identifier_fields and field != 'PERIOD':  # Skip PERIOD column
                # Replace infinite values with NaN
                df[field] = df[field].replace([np.inf, -np.inf], np.nan)
                # Fill missing values with median
                df[field] = df[field].fillna(df[field].median())
        
        # Apply winsorization
        df = self.winsorize_features(df)
        
        # Restore PERIOD column
        df['PERIOD'] = period_col
        
        return df
    
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
        
        # Log fiscal month distribution for debugging before processing
        fiscal_month_counts = result_df.groupby(id_col)['fiscal_month'].nunique()
        self.logger.info(f"Number of unique fiscal months per ID: {fiscal_month_counts.value_counts()}")
        
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
            
            # Calculate if the period end date is before or after pit_date
            is_before_pit = group[period_end_col] <= pit_date
            
            # For historical periods (before PIT_DATE), count from most recent (0) to oldest (-1, -2, etc.)
            historical_periods = group[is_before_pit].copy()
            if not historical_periods.empty:
                historical_periods = historical_periods.sort_values(period_end_col, ascending=False)
                historical_periods['PERIOD'] = list(range(-len(historical_periods) + 1, 1))
            
            # For future periods (after PIT_DATE), count from nearest (1) to furthest (2, 3, etc.)
            future_periods = group[~is_before_pit].copy()
            if not future_periods.empty:
                future_periods = future_periods.sort_values(period_end_col)
                future_periods['PERIOD'] = list(range(1, len(future_periods) + 1))
            
            # Combine historical and future periods
            group = pd.concat([historical_periods, future_periods])
            
            # Drop the fiscal_month column as it's no longer needed
            group = group.drop('fiscal_month', axis=1)
            
            # Ensure PERIOD column is integer type
            group['PERIOD'] = group['PERIOD'].astype(int)
            
            return group
        
        # Apply period calculation by ID
        self.logger.info(f"Calculating periods by ID using id column: '{id_col}'")
        result_df = result_df.groupby(id_col).apply(calculate_periods_for_group).reset_index(drop=True)
        
        # Log period counts and verify PERIOD column exists
        if 'PERIOD' in result_df.columns:
            period_counts = result_df['PERIOD'].value_counts().sort_index()
            self.logger.info(f"Period counts: {dict(period_counts)}")
            periods = sorted(result_df['PERIOD'].unique())
            self.logger.info(f"Available periods: {[int(p) for p in periods]}")
        else:
            self.logger.error("PERIOD column was not created!")
            self.logger.info(f"Available columns: {result_df.columns.tolist()}")
        
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
        threshold = threshold or self.winsorization_threshold
        self.logger.info(f"Winsorizing features with threshold: {threshold}")
        
        # Create a copy of the DataFrame
        result_df = df.copy()
        
        # Get columns to winsorize (exclude ID, date, and period columns)
        exclude_cols = ['ID', 'PIT_DATE', 'PERIOD_END_DATE', 'PERIOD', 'period']
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
        Save processed data using the data storage system.
        
        Args:
            df: DataFrame containing processed data
            date: Date of the data
            
        Returns:
            Path to the saved data
        """
        self.logger.info(f"Saving processed data for date: {date}")
        
        # Create a copy of the DataFrame to avoid modifying the original
        df_copy = df.copy()
        
        # Ensure date column is datetime
        df_copy['PIT_DATE'] = pd.to_datetime(df_copy['PIT_DATE'])
        
        # Save using the data storage system
        output_path = self.data_storage.store_data(
            df=df_copy,
            data_type=DataType.FUNDAMENTALS,
            stage=DataStage.PROCESSED,
            date=date
        )
        
        self.logger.info(f"Saved processed data to: {output_path}")
        
        return output_path
    
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
            
            # Convert all column names to uppercase
            raw_df.columns = raw_df.columns.str.upper()
            
            # Transform raw data
            transformed_df = self.transform_raw_data(raw_df)
            
            # Save processed data
            output_file = self.save_processed_data(transformed_df, date_str)
            processed_files[date_str] = output_file
            
            # Save signed log data
            signed_log_file = self.save_signed_log_data(transformed_df)
            processed_files[f"{date_str}_signed_log"] = signed_log_file
            
            # Generate and save pre-feature data
            pre_feature_df = self.processed_data_feat_gen(transformed_df)
            pre_feature_file = self.save_pre_feature_set(pre_feature_df)
            processed_files[f"{date_str}_pre_feature"] = pre_feature_file
        
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
    
    def save_signed_log_data(self, df: pd.DataFrame) -> str:
        """
        Create a new DataFrame with only signed log columns and save it to parquet.
        Ensures all column names are uppercase.
        
        Args:
            df: DataFrame containing processed data
            
        Returns:
            Path to the saved file
        """
        self.logger.info("Creating signed log data DataFrame")
        
        # Create a copy to avoid modifying the original DataFrame
        df_copy = df.copy()
        
        # Keep identifier fields and all signed log columns
        cols_to_keep = []
        
        # Add identifier fields
        cols_to_keep.extend(self.identifier_fields)
        
        # Add signed log columns based on suffixes
        signed_log_suffixes = [
            self.raw_signed_log_suffix,
            self.raw_scaled_signed_log_suffix.format(SALES=self.scaling_variable),
            self.ratio_signed_log_suffix
        ]
        
        for suffix in signed_log_suffixes:
            signed_log_cols = [col for col in df_copy.columns if suffix in col]
            cols_to_keep.extend(signed_log_cols)
        
        # Create a new DataFrame with essential columns and signed log columns
        processed_data_signed_log = df_copy[cols_to_keep].copy()
        
        # Convert all column names to uppercase
        processed_data_signed_log.columns = [col.upper() for col in processed_data_signed_log.columns]
        
        # Get the date from PIT_DATE
        if 'PIT_DATE' in df_copy.columns and not df_copy['PIT_DATE'].empty:
            date_str = pd.to_datetime(df_copy['PIT_DATE'].iloc[0]).strftime('%Y-%m-%d')
        else:
            date_str = datetime.now().strftime('%Y-%m-%d')
        
        # Save using the data storage system with partitioned format
        output_path = self.data_storage.store_data(
            df=processed_data_signed_log,
            data_type=DataType.FUNDAMENTALS,
            stage=DataStage.PROCESSED,
            date=date_str,
            sub_type='signed_log'
        )
        
        self.logger.info(f"Saved signed log data with shape {processed_data_signed_log.shape} to {output_path}")
        
        return output_path

    def apply_signed_log(self, series: pd.Series) -> pd.Series:
        """
        Apply signed log transformation to a series.
        
        Args:
            series: Input series
            
        Returns:
            Series with signed log transformation applied
        """
        return np.sign(series) * np.log1p(np.abs(series))

    def apply_winsorization(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply winsorization to numeric columns.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with winsorization applied
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in self.identifier_fields:
                lower, upper = np.nanpercentile(df[col], 
                                              [self.winsorize_limits[0] * 100, 
                                               self.winsorize_limits[1] * 100])
                df[col] = df[col].clip(lower=lower, upper=upper)
        return df 

    def processed_data_feat_gen(self, processed_data: pd.DataFrame, scaling_field: str = 'SALES') -> pd.DataFrame:
        """
        Filter processed data to keep:
        1. Basic identifier and time fields
        2. Feature categories:
           - RAW_SIGNED_LOG
           - RAW_SCALED_{scaling_field}_SIGNED_LOG
           - RATIO_SIGNED_LOG
           - COEFF_OF_VAR fields
        
        Args:
            processed_data: Input DataFrame with processed data
            scaling_field: Field used for scaling (default: 'SALES')
            
        Returns:
            DataFrame containing identifier/time fields and specified feature categories
        """
        self.logger.info("Generating feature set from processed data")
        
        # Convert scaling_field to uppercase
        scaling_field = scaling_field.upper()
        
        # Define basic identifier and time fields to keep (in uppercase)
        base_fields = ['ID', 'PERIOD_END_DATE', 'PIT_DATE', 'PERIOD']
        
        # Define feature categories to keep
        columns_to_keep = base_fields.copy()  # Start with base fields
        
        # Add RAW_SIGNED_LOG columns
        raw_cols = [col for col in processed_data.columns if col.upper().endswith('_RAW_SIGNED_LOG')]
        columns_to_keep.extend(raw_cols)
        
        # Add RAW_SCALED_{scaling_field}_SIGNED_LOG columns
        scaled_cols = [col for col in processed_data.columns if f'_RAW_SCALED_{scaling_field}_SIGNED_LOG'.upper() in col.upper()]
        columns_to_keep.extend(scaled_cols)
        
        # Add RATIO_SIGNED_LOG columns
        ratio_cols = [col for col in processed_data.columns if col.upper().endswith('_RATIO_SIGNED_LOG')]
        columns_to_keep.extend(ratio_cols)
        
        # Add COEFF_OF_VAR columns
        cv_cols = [col for col in processed_data.columns if col.upper().endswith('_COEFF_OF_VAR')]
        columns_to_keep.extend(cv_cols)
        
        # Filter the DataFrame to keep only the specified columns
        filtered_df = processed_data[columns_to_keep].copy()
        
        # Convert all column names to uppercase
        filtered_df.columns = filtered_df.columns.str.upper()
        
        self.logger.info(f"Generated feature set with shape: {filtered_df.shape}")
        self.logger.info(f"Number of features: {len(filtered_df.columns)}")
        
        # Save the feature set using PIT_DATE
        self.save_pre_feature_set(filtered_df)
        
        return filtered_df
    
    def save_pre_feature_set(self, feature_set: pd.DataFrame) -> str:
        """
        Save pre-feature creation data to parquet file.
        Uses PIT_DATE to determine the save date - if multiple dates exist,
        uses the month from PIT_DATE for the filename.
        
        Args:
            feature_set: DataFrame containing the pre-feature data
            
        Returns:
            Path to the saved pre-feature file
        """
        # Get unique PIT_DATEs
        pit_dates = feature_set['PIT_DATE'].unique()
        
        if len(pit_dates) == 0:
            self.logger.warning("No PIT_DATE found in the data")
            return ""
            
        # If multiple dates, use the month from the first date
        if len(pit_dates) > 1:
            self.logger.info(f"Multiple PIT_DATEs found: {pit_dates}")
            # Use the first date's month for the filename
            date_str = pd.to_datetime(pit_dates[0]).strftime('%Y-%m-%d')
        else:
            date_str = pd.to_datetime(pit_dates[0]).strftime('%Y-%m-%d')
        
        # Save using the data storage system with partitioned format
        output_path = self.data_storage.store_data(
            df=feature_set,
            data_type=DataType.FUNDAMENTALS,
            stage=DataStage.FEATURES,
            date=date_str,
            sub_type='pre_feature_set'
        )
        
        self.logger.info(f"Saved pre-feature set to {output_path}")
        return output_path
    
    def load_pre_feature_set(self, date: str) -> pd.DataFrame:
        """
        Load pre-feature creation data from parquet file.
        
        Args:
            date: Date of the data (can be YYYY-MM or YYYY-MM-DD)
            
        Returns:
            DataFrame containing the pre-feature data
        """
        filename = f"pre_feature_set_{date}.pq"
        filepath = os.path.join(self.feature_set_dir, filename)
        
        if os.path.exists(filepath):
            self.logger.info(f"Loading pre-feature set from {filepath}")
            feature_set = pd.read_parquet(filepath)
            self.logger.info(f"Loaded pre-feature set with shape: {feature_set.shape}")
            return feature_set
        else:
            self.logger.warning(f"Pre-feature set file not found: {filepath}")
            return pd.DataFrame() 
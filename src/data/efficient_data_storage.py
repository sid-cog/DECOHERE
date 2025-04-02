#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Efficient storage and retrieval system for processed data files.
Supports multiple data types (fundamentals, returns, alternate) and stages (raw, intermediate, processed, features).
Uses partitioned Parquet datasets for optimal performance.
"""

import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import shutil
from pathlib import Path
from enum import Enum
import re  # Add import for regex

# Define data types and stages as enums
class DataType(Enum):
    FUNDAMENTALS = "fundamentals"
    RETURNS = "returns"
    ALTERNATE = "alternate"

class DataStage(Enum):
    RAW = "raw"
    INTERMEDIATE = "intermediate"
    PROCESSED = "processed"
    FEATURES = "features"

# Export the enums
__all__ = ['DataType', 'DataStage', 'EfficientDataStorage']

class EfficientDataStorage:
    """
    Efficient storage and retrieval system for processed data files.
    Uses partitioned Parquet datasets for optimal performance.
    """
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Initialize the data storage system.
        
        Args:
            config: Configuration dictionary
            logger: Logger to use (if None, create a new logger)
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Extract configuration parameters
        self.data_paths = config['data']
        self.storage_config = config['processing']['storage']
        
        # Validate and create directory structure
        if not self.validate_structure():
            raise ValueError("Invalid data structure configuration")
        
        # Create all necessary directories
        self._create_directories()
    
    def _create_directories(self):
        """Create all necessary directories in the data structure."""
        for stage in DataStage:
            stage_path = self.data_paths[stage.value]
            if isinstance(stage_path, dict):
                for data_type in stage_path:
                    path = stage_path[data_type]
                    if isinstance(path, dict):
                        for sub_type in path:
                            os.makedirs(path[sub_type], exist_ok=True)
                    else:
                        os.makedirs(path, exist_ok=True)
            else:
                os.makedirs(stage_path, exist_ok=True)
    
    def _get_partition_path(self, base_path: str, date: str) -> str:
        """
        Generate the partition path for a given date.
        
        Args:
            base_path: Base directory path
            date: Date string (YYYY-MM-DD)
            
        Returns:
            Partitioned path string
        """
        date_obj = datetime.strptime(date, '%Y-%m-%d')
        return os.path.join(base_path, f"year={date_obj.year}", f"month={date_obj.month:02d}")
    
    def store_data(self, df: pd.DataFrame, data_type: DataType, stage: DataStage, 
                  date: str, sub_type: Optional[str] = None) -> str:
        """
        Store data in partitioned parquet format.
        
        Args:
            df: DataFrame containing data
            data_type: Type of data (fundamentals, returns, alternate)
            stage: Stage of processing (raw, intermediate, processed, features)
            date: Date of the data (YYYY-MM-DD)
            sub_type: Sub-type for alternate data (e.g., type_A, type_B) or special types (e.g., signed_log, pre_feature_set)
            
        Returns:
            Path to the saved file
        """
        self.logger.info(f"Storing {data_type.value} data for date: {date} in {stage.value} stage")
        
        # Get the appropriate path
        base_path = os.path.join(self.data_paths['base_dir'], self.data_paths[stage.value][data_type.value])
        if sub_type:
            # For special types like signed_log and pre_feature_set, create a subdirectory
            base_path = os.path.join(base_path, sub_type)
        
        # Create partition path
        partition_path = self._get_partition_path(base_path, date)
        os.makedirs(partition_path, exist_ok=True)
        
        # Create filename
        filename = f"data_{date}.pq"
        filepath = os.path.join(partition_path, filename)
        
        # Save to parquet with specified compression
        df.to_parquet(
            filepath,
            index=False,
            compression=self.storage_config['compression']
        )
        
        # Verify the file was written
        if os.path.exists(filepath):
            self.logger.info(f"Saved data to: {filepath}")
            self.logger.info(f"File size: {os.path.getsize(filepath)} bytes")
        else:
            self.logger.error(f"Failed to save data to: {filepath}")
        
        return filepath
    
    def load_data(self, data_type: DataType, stage: DataStage, 
                 date: Optional[str] = None, mode: str = 'day',
                 start_date: Optional[str] = None, end_date: Optional[str] = None,
                 sub_type: Optional[str] = None) -> pd.DataFrame:
        """
        Load data efficiently based on mode and data type.
        
        Args:
            data_type: Type of data to load
            stage: Stage of processing to load from
            date: Specific date to load (YYYY-MM-DD)
            mode: Mode to load data for ('day', 'week', 'year', or 'all_years')
            start_date: Start date for range queries (YYYY-MM-DD)
            end_date: End date for range queries (YYYY-MM-DD)
            sub_type: Sub-type for alternate data
            
        Returns:
            DataFrame containing loaded data
        """
        # Get the appropriate path
        base_path = os.path.join(self.data_paths['base_dir'], self.data_paths[stage.value][data_type.value])
        if sub_type and data_type == DataType.ALTERNATE:
            base_path = os.path.join(base_path, sub_type)
        
        # Handle different modes
        if mode == 'day' and date:
            return self._load_day_data(base_path, date)
        elif mode in ['week', 'year'] and start_date and end_date:
            return self._load_range_data(base_path, start_date, end_date)
        elif mode == 'all_years':
            return self._load_all_years_data(base_path)
        else:
            self.logger.error(f"Invalid mode or missing date parameters: mode={mode}, date={date}")
            return pd.DataFrame()
    
    def _load_day_data(self, base_path: str, date: str) -> pd.DataFrame:
        """Load data for a specific day."""
        self.logger.info(f"Loading data for date: {date}")
        
        try:
            partition_path = self._get_partition_path(base_path, date)
            filename = f"data_{date}.pq"
            filepath = os.path.join(partition_path, filename)
            
            if not os.path.exists(filepath):
                self.logger.error(f"Data file does not exist: {filepath}")
                return pd.DataFrame()
            
            df = pd.read_parquet(filepath)
            self.logger.info(f"Loaded data with shape: {df.shape}")
            return df
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            return pd.DataFrame()
    
    def _load_range_data(self, base_path: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Load data for a date range."""
        self.logger.info(f"Loading data for date range: {start_date} to {end_date}")
        
        try:
            start_date_obj = datetime.strptime(start_date, '%Y-%m-%d')
            end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')
            
            dfs = []
            for year in range(start_date_obj.year, end_date_obj.year + 1):
                year_path = os.path.join(base_path, f"year={year}")
                if not os.path.exists(year_path):
                    continue
                    
                for month in range(1, 13):
                    month_path = os.path.join(year_path, f"month={month:02d}")
                    if not os.path.exists(month_path):
                        continue
                    
                    for filename in os.listdir(month_path):
                        if not filename.endswith('.pq'):
                            continue
                        
                        file_date = filename.replace('data_', '').replace('.pq', '')
                        try:
                            file_date_obj = datetime.strptime(file_date, '%Y-%m-%d')
                            if start_date_obj <= file_date_obj <= end_date_obj:
                                filepath = os.path.join(month_path, filename)
                                df = pd.read_parquet(filepath)
                                dfs.append(df)
                        except ValueError:
                            continue
            
            if dfs:
                combined_df = pd.concat(dfs, ignore_index=True)
                self.logger.info(f"Loaded data from {len(dfs)} files with shape: {combined_df.shape}")
                return combined_df
            else:
                self.logger.warning(f"No data found for date range: {start_date} to {end_date}")
                return pd.DataFrame()
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            return pd.DataFrame()
    
    def _load_all_years_data(self, base_path: str) -> pd.DataFrame:
        """Load all available data."""
        self.logger.info("Loading all available data")
        
        try:
            dfs = []
            for year_dir in os.listdir(base_path):
                year_path = os.path.join(base_path, year_dir)
                if not os.path.isdir(year_path):
                    continue
                
                for month_dir in os.listdir(year_path):
                    month_path = os.path.join(year_path, month_dir)
                    if not os.path.isdir(month_path):
                        continue
                    
                    for filename in os.listdir(month_path):
                        if not filename.endswith('.pq'):
                            continue
                        
                        filepath = os.path.join(month_path, filename)
                        df = pd.read_parquet(filepath)
                        dfs.append(df)
            
            if dfs:
                combined_df = pd.concat(dfs, ignore_index=True)
                self.logger.info(f"Loaded all data from {len(dfs)} files with shape: {combined_df.shape}")
                return combined_df
            else:
                self.logger.warning("No data files found")
                return pd.DataFrame()
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            return pd.DataFrame()
    
    def get_available_dates(self, data_type: DataType, stage: DataStage, 
                          sub_type: Optional[str] = None) -> List[str]:
        """
        Get a list of all available dates for a specific data type and stage.
        
        Args:
            data_type: Type of data
            stage: Stage of processing
            sub_type: Sub-type for alternate data
            
        Returns:
            List of dates (YYYY-MM-DD)
        """
        try:
            dates = []
            base_path = self.data_paths[stage.value][data_type.value]
            if sub_type and data_type == DataType.ALTERNATE:
                base_path = base_path[sub_type]
            
            for year_dir in os.listdir(base_path):
                year_path = os.path.join(base_path, year_dir)
                if not os.path.isdir(year_path):
                    continue
                
                for month_dir in os.listdir(year_path):
                    month_path = os.path.join(year_path, month_dir)
                    if not os.path.isdir(month_path):
                        continue
                    
                    for filename in os.listdir(month_path):
                        if not filename.endswith('.pq'):
                            continue
                        
                        date = filename.replace('data_', '').replace('.pq', '')
                        try:
                            datetime.strptime(date, '%Y-%m-%d')
                            dates.append(date)
                        except ValueError:
                            continue
            
            return sorted(dates)
        except Exception as e:
            self.logger.error(f"Error getting available dates: {e}")
            return []
    
    def migrate_data(self, old_path: str, data_type: DataType, stage: DataStage, 
                    sub_type: Optional[str] = None) -> bool:
        """
        Migrate data from old structure to new partitioned structure.
        
        Args:
            old_path: Path to the old data directory
            data_type: Type of data being migrated
            stage: Stage of processing
            sub_type: Sub-type for alternate data
            
        Returns:
            True if successful, False otherwise
        """
        self.logger.info(f"Migrating {data_type.value} data from {old_path}")
        
        try:
            # Get target path
            base_path = self.data_paths[stage.value][data_type.value]
            if sub_type and data_type == DataType.ALTERNATE:
                base_path = base_path[sub_type]
            
            # Process each file in old directory
            for filename in os.listdir(old_path):
                if not filename.endswith('.pq'):
                    continue
                
                # Extract date from filename
                try:
                    if '_' in filename:
                        date = filename.split('_')[-1].replace('.pq', '')
                    else:
                        date = filename.replace('.pq', '')
                    
                    # Validate date format
                    datetime.strptime(date, '%Y-%m-%d')
                    
                    # Read old file
                    old_filepath = os.path.join(old_path, filename)
                    df = pd.read_parquet(old_filepath)
                    
                    # Store in new structure
                    self.store_data(df, data_type, stage, date, sub_type)
                    
                    self.logger.info(f"Migrated {filename}")
                except Exception as e:
                    self.logger.error(f"Error migrating {filename}: {e}")
                    continue
            
            self.logger.info("Migration completed successfully")
            return True
        except Exception as e:
            self.logger.error(f"Error during migration: {e}")
            return False

    def processed_data_feat_gen(self, df: pd.DataFrame, scaling_field: str = 'SALES') -> pd.DataFrame:
        """
        Filter processed data to keep only specific feature categories:
        - ID column
        - RAW_SIGNED_LOG
        - RAW_SCALED_{scaling_field}_SIGNED_LOG
        - RATIO_SIGNED_LOG
        - COEFF_OF_VAR fields
        
        Args:
            df: DataFrame containing processed data
            scaling_field: Field used for scaling (default: 'SALES')
            
        Returns:
            DataFrame containing only the selected features
        """
        self.logger.info("Generating features from processed data")
        self.logger.info(f"Initial shape: {df.shape}")
        
        try:
            # Create patterns for column filtering (using regex end-of-string anchor $)
            patterns = [
                '_RAW_SIGNED_LOG$',
                f'_RAW_SCALED_{scaling_field}_SIGNED_LOG$',
                '_RATIO_SIGNED_LOG$',
                '_COEFF_OF_VAR$'  # Restored $ for regex matching
            ]
            
            # Filter columns based on patterns using regex search
            selected_cols = []
            # Ensure ID column is always included first if it exists
            if 'ID' in df.columns:
                selected_cols.append('ID')
            # --- Add PIT_DATE and PERIOD if they exist --- 
            if 'PIT_DATE' in df.columns:
                selected_cols.append('PIT_DATE')
            if 'PERIOD' in df.columns:
                selected_cols.append('PERIOD')
                
            for col in df.columns:
                # Skip columns already added or not relevant feature types
                if col in ['ID', 'PIT_DATE', 'PERIOD']:
                    continue
                col_upper = col.upper()
                if any(re.search(pattern, col_upper) for pattern in patterns):
                    selected_cols.append(col)
            
            # Create filtered DataFrame
            # Use .loc to avoid potential SettingWithCopyWarning and ensure order
            # Also ensure no duplicates in selected_cols before creating the frame
            unique_selected_cols = list(dict.fromkeys(selected_cols)) # Preserve order while removing duplicates
            filtered_df = df.loc[:, unique_selected_cols].copy()
            
            self.logger.info(f"Selected {len(selected_cols)} features")
            self.logger.info(f"Final shape: {filtered_df.shape}")
            self.logger.info(f"Selected columns: {selected_cols}")
            
            return filtered_df
        except Exception as e:
            self.logger.error(f"Error generating features: {e}")
            self.logger.error(f"Error type: {type(e)}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return pd.DataFrame()

    def validate_structure(self) -> bool:
        """
        Validate the data structure and ensure all required directories exist.
        
        Returns:
            True if structure is valid, False otherwise
        """
        self.logger.info("Validating data structure")
        
        try:
            # Check each stage
            for stage in DataStage:
                stage_path = self.data_paths[stage.value]
                if not isinstance(stage_path, dict):
                    self.logger.error(f"Invalid structure for stage {stage.value}: expected dict")
                    return False
                
                # Check each data type
                for data_type in DataType:
                    if data_type.value not in stage_path:
                        self.logger.error(f"Missing {data_type.value} in stage {stage.value}")
                        return False
                    
                    path = stage_path[data_type.value]
                    
                    # Handle alternate data types
                    if data_type == DataType.ALTERNATE:
                        if not isinstance(path, dict):
                            self.logger.error(f"Invalid structure for alternate data in stage {stage.value}")
                            return False
                        
                        # Check alternate data sub-types
                        for sub_type in ['type_A', 'type_B']:
                            if sub_type not in path:
                                self.logger.error(f"Missing {sub_type} in alternate data for stage {stage.value}")
                                return False
                            
                            if not os.path.exists(path[sub_type]):
                                self.logger.warning(f"Creating missing directory: {path[sub_type]}")
                                os.makedirs(path[sub_type], exist_ok=True)
                    else:
                        if not os.path.exists(path):
                            self.logger.warning(f"Creating missing directory: {path}")
                            os.makedirs(path, exist_ok=True)
            
            # Check additional directories
            for dir_name in ['models', 'reporting']:
                path = self.data_paths[dir_name]
                if not os.path.exists(path):
                    self.logger.warning(f"Creating missing directory: {path}")
                    os.makedirs(path, exist_ok=True)
            
            self.logger.info("Data structure validation completed successfully")
            return True
        except Exception as e:
            self.logger.error(f"Error validating data structure: {e}")
            return False

    def store_processed_data(self, df: pd.DataFrame, date: str) -> str:
        """
        Store processed data using partitioned storage.
        
        Args:
            df: DataFrame containing processed data
            date: Date of the data
            
        Returns:
            Path to the stored data
        """
        self.logger.info(f"Storing processed data for date: {date}")
        
        # Create a copy of the DataFrame to avoid modifying the original
        df_copy = df.copy()
        
        # Ensure date column is datetime
        df_copy['PIT_DATE'] = pd.to_datetime(df_copy['PIT_DATE'])
        
        # Use the store_data method with the correct data type and stage
        return self.store_data(
            df=df_copy,
            data_type=DataType.FUNDAMENTALS,
            stage=DataStage.PROCESSED,
            date=date
        ) 
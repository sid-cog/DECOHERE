#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Efficient storage and retrieval system for processed data files.
Supports day, week, year, and ALL YEARS modes.
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
        self.processed_data_dir = config['data']['processed_data']
        self.partitioned_data_dir = os.path.join(self.processed_data_dir, "partitioned")
        
        # Create directories if they don't exist
        os.makedirs(self.processed_data_dir, exist_ok=True)
        os.makedirs(self.partitioned_data_dir, exist_ok=True)
    
    def store_processed_data(self, df: pd.DataFrame, date: str) -> str:
        """
        Store processed data efficiently using partitioning.
        
        Args:
            df: DataFrame containing processed data
            date: Date of the data (YYYY-MM-DD)
            
        Returns:
            Path to the saved partitioned dataset
        """
        self.logger.info(f"Storing processed data for date: {date}")
        
        # Parse the date
        date_obj = datetime.strptime(date, '%Y-%m-%d')
        
        # Add date components for partitioning
        df['year'] = date_obj.year
        df['month'] = date_obj.month
        df['day'] = date_obj.day
        df['date'] = date
        
        # Convert to PyArrow Table
        table = pa.Table.from_pandas(df)
        
        # Define partitioning
        partition_cols = ['year', 'month']
        
        # Write to partitioned dataset
        pq.write_to_dataset(
            table,
            root_path=self.partitioned_data_dir,
            partition_cols=partition_cols,
            existing_data_behavior='overwrite_or_ignore'
        )
        
        # Also save as a single file for backward compatibility
        legacy_file_path = os.path.join(self.processed_data_dir, f"processed_{date}.pq")
        df.to_parquet(legacy_file_path)
        
        self.logger.info(f"Stored processed data in partitioned format and legacy format")
        
        return self.partitioned_data_dir
    
    def load_processed_data(self, date: str = None, mode: str = 'day', 
                           start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Load processed data efficiently based on mode.
        
        Args:
            date: Specific date to load (YYYY-MM-DD)
            mode: Mode to load data for ('day', 'week', 'year', or 'all_years')
            start_date: Start date for range queries (YYYY-MM-DD)
            end_date: End date for range queries (YYYY-MM-DD)
            
        Returns:
            DataFrame containing processed data
        """
        # Handle different modes
        if mode == 'day' and date:
            return self._load_day_data(date)
        elif mode == 'week' and start_date and end_date:
            return self._load_range_data(start_date, end_date)
        elif mode == 'year' and start_date and end_date:
            return self._load_range_data(start_date, end_date)
        elif mode == 'all_years':
            return self._load_all_years_data()
        else:
            self.logger.error(f"Invalid mode or missing date parameters: mode={mode}, date={date}, start_date={start_date}, end_date={end_date}")
            return pd.DataFrame()
    
    def _load_day_data(self, date: str) -> pd.DataFrame:
        """
        Load processed data for a specific day.
        
        Args:
            date: Date to load data for (YYYY-MM-DD)
            
        Returns:
            DataFrame containing processed data
        """
        self.logger.info(f"Loading processed data for date: {date}")
        
        # Parse the date
        date_obj = datetime.strptime(date, '%Y-%m-%d')
        year, month, day = date_obj.year, date_obj.month, date_obj.day
        
        # Try loading from partitioned dataset first
        try:
            # Create filter for the specific date
            filters = [
                ('year', '=', year),
                ('month', '=', month),
                ('day', '=', day)
            ]
            
            # Load from partitioned dataset
            dataset = ds.dataset(self.partitioned_data_dir, format='parquet')
            df = dataset.to_table(filter=ds.field('year') == year and 
                                 ds.field('month') == month and 
                                 ds.field('day') == day).to_pandas()
            
            if not df.empty:
                self.logger.info(f"Loaded processed data from partitioned dataset with shape: {df.shape}")
                return df
        except Exception as e:
            self.logger.warning(f"Error loading from partitioned dataset: {e}")
        
        # Fall back to legacy file
        legacy_file_path = os.path.join(self.processed_data_dir, f"processed_{date}.pq")
        if os.path.exists(legacy_file_path):
            df = pd.read_parquet(legacy_file_path)
            self.logger.info(f"Loaded processed data from legacy file with shape: {df.shape}")
            return df
        
        self.logger.warning(f"No processed data found for date: {date}")
        return pd.DataFrame()
    
    def _load_range_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Load processed data for a date range.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame containing processed data
        """
        self.logger.info(f"Loading processed data for date range: {start_date} to {end_date}")
        
        # Parse dates
        start_date_obj = datetime.strptime(start_date, '%Y-%m-%d')
        end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Generate date range
        date_range = pd.date_range(start=start_date, end=end_date)
        
        # Try loading from partitioned dataset
        try:
            # Create filter for the date range
            dataset = ds.dataset(self.partitioned_data_dir, format='parquet')
            
            # Convert dates to timestamps for filtering
            start_ts = pd.Timestamp(start_date)
            end_ts = pd.Timestamp(end_date)
            
            # Load data within the date range
            df = dataset.to_table(
                filter=((ds.field('year') > start_date_obj.year) | 
                       ((ds.field('year') == start_date_obj.year) & 
                        (ds.field('month') >= start_date_obj.month))) & 
                      ((ds.field('year') < end_date_obj.year) | 
                       ((ds.field('year') == end_date_obj.year) & 
                        (ds.field('month') <= end_date_obj.month)))
            ).to_pandas()
            
            # Further filter by exact dates
            df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
            
            if not df.empty:
                self.logger.info(f"Loaded processed data from partitioned dataset with shape: {df.shape}")
                return df
        except Exception as e:
            self.logger.warning(f"Error loading from partitioned dataset: {e}")
        
        # Fall back to loading individual files
        dfs = []
        for date in date_range:
            date_str = date.strftime('%Y-%m-%d')
            legacy_file_path = os.path.join(self.processed_data_dir, f"processed_{date_str}.pq")
            if os.path.exists(legacy_file_path):
                try:
                    df = pd.read_parquet(legacy_file_path)
                    dfs.append(df)
                except Exception as e:
                    self.logger.warning(f"Error loading file {legacy_file_path}: {e}")
        
        if dfs:
            combined_df = pd.concat(dfs, ignore_index=True)
            self.logger.info(f"Loaded processed data from legacy files with shape: {combined_df.shape}")
            return combined_df
        
        self.logger.warning(f"No processed data found for date range: {start_date} to {end_date}")
        return pd.DataFrame()
    
    def _load_all_years_data(self) -> pd.DataFrame:
        """
        Load all processed data across all years.
        
        Returns:
            DataFrame containing all processed data
        """
        self.logger.info("Loading all processed data across all years")
        
        # Try loading from partitioned dataset
        try:
            dataset = ds.dataset(self.partitioned_data_dir, format='parquet')
            df = dataset.to_table().to_pandas()
            
            if not df.empty:
                self.logger.info(f"Loaded all processed data from partitioned dataset with shape: {df.shape}")
                return df
        except Exception as e:
            self.logger.warning(f"Error loading from partitioned dataset: {e}")
        
        # Fall back to loading all legacy files
        legacy_files = [f for f in os.listdir(self.processed_data_dir) 
                       if f.startswith('processed_') and f.endswith('.pq')]
        
        if not legacy_files:
            self.logger.warning("No processed data files found")
            return pd.DataFrame()
        
        dfs = []
        for file in legacy_files:
            file_path = os.path.join(self.processed_data_dir, file)
            try:
                df = pd.read_parquet(file_path)
                dfs.append(df)
            except Exception as e:
                self.logger.warning(f"Error loading file {file_path}: {e}")
        
        if dfs:
            combined_df = pd.concat(dfs, ignore_index=True)
            self.logger.info(f"Loaded all processed data from legacy files with shape: {combined_df.shape}")
            return combined_df
        
        return pd.DataFrame()
    
    def migrate_legacy_to_partitioned(self) -> bool:
        """
        Migrate all legacy processed data files to the partitioned format.
        
        Returns:
            True if migration was successful, False otherwise
        """
        self.logger.info("Migrating legacy processed data files to partitioned format")
        
        # Find all legacy processed data files
        legacy_files = [f for f in os.listdir(self.processed_data_dir) 
                       if f.startswith('processed_') and f.endswith('.pq')]
        
        if not legacy_files:
            self.logger.warning("No legacy processed data files found")
            return False
        
        success = True
        for file in legacy_files:
            try:
                # Extract date from filename
                date = file.replace('processed_', '').replace('.pq', '')
                
                # Load legacy file
                file_path = os.path.join(self.processed_data_dir, file)
                df = pd.read_parquet(file_path)
                
                # Store in partitioned format
                self.store_processed_data(df, date)
                
                self.logger.info(f"Migrated {file} to partitioned format")
            except Exception as e:
                self.logger.error(f"Error migrating {file}: {e}")
                success = False
        
        return success
    
    def get_available_dates(self) -> List[str]:
        """
        Get a list of all available dates in the processed data.
        
        Returns:
            List of dates (YYYY-MM-DD)
        """
        # Try to get dates from partitioned dataset
        try:
            dataset = ds.dataset(self.partitioned_data_dir, format='parquet')
            df = dataset.to_table(columns=['date']).to_pandas()
            if not df.empty:
                return sorted(df['date'].unique().tolist())
        except Exception as e:
            self.logger.warning(f"Error getting dates from partitioned dataset: {e}")
        
        # Fall back to legacy files
        legacy_files = [f for f in os.listdir(self.processed_data_dir) 
                       if f.startswith('processed_') and f.endswith('.pq')]
        
        dates = [file.replace('processed_', '').replace('.pq', '') for file in legacy_files]
        return sorted(dates) 
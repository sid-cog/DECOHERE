#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to organize the parquet file structure for storage:
1. Financials and sector data stored together in one folder
2. Price and total returns data stored together in a different folder
3. Process only one month of financial data and ensure sector fields are correctly mapped
"""

import os
import sys
import logging
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger('organize_data_storage')

class DataStorageOrganizer:
    """
    Class to organize the data storage structure.
    """
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Initialize the data storage organizer.
        
        Args:
            config: Configuration dictionary
            logger: Logger to use (if None, create a new logger)
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Extract configuration parameters
        self.base_data_dir = config['data'].get('processed_data', 'data/processed')
        
        # Define new storage structure
        self.fundamentals_dir = os.path.join(self.base_data_dir, "fundamentals")
        self.returns_dir = os.path.join(self.base_data_dir, "returns")
        
        # Source data paths
        self.financials_dir = config['data'].get('financials_dir', '/home/siddharth_johri/projects/data/financials')
        self.sector_mapping_path = config['data'].get('sector_mapping', '/home/siddharth_johri/projects/data/sector/sector.pq')
        self.sector_mapping_path2 = '/home/siddharth_johri/projects/data/sector/sector_2.pq'  # Additional sector file
        self.price_returns_path = config['data'].get('price_returns', '/home/siddharth_johri/projects/data/returns/px_df.pq')
        self.total_returns_path = config['data'].get('total_returns', '/home/siddharth_johri/projects/data/returns/tr_df.pq')
        
        # Create directories if they don't exist
        os.makedirs(self.fundamentals_dir, exist_ok=True)
        os.makedirs(self.returns_dir, exist_ok=True)
    
    def load_one_month_financials(self, year: int, month: int) -> pd.DataFrame:
        """
        Load one month of financial data.
        
        Args:
            year: Year to load data for
            month: Month to load data for
            
        Returns:
            DataFrame containing financial data
        """
        self.logger.info(f"Loading financial data for {year}-{month:02d}")
        
        # Construct the file path
        file_path = os.path.join(self.financials_dir, f"financials_{year}_{month:02d}.pq")
        
        # Check if the file exists
        if not os.path.exists(file_path):
            self.logger.warning(f"Financial data file not found: {file_path}")
            # Try the consolidated file
            consolidated_file = os.path.join(self.financials_dir, "processed_financials.pq")
            if os.path.exists(consolidated_file):
                self.logger.info(f"Using consolidated file: {consolidated_file}")
                df = pd.read_parquet(consolidated_file)
                # Filter for the specified year and month
                if 'PIT_DATE' in df.columns:
                    df['year'] = pd.to_datetime(df['PIT_DATE']).dt.year
                    df['month'] = pd.to_datetime(df['PIT_DATE']).dt.month
                    df = df[(df['year'] == year) & (df['month'] == month)]
                    self.logger.info(f"Filtered data for {year}-{month:02d} with shape: {df.shape}")
                    return df
            
            # If we still don't have data, use the raw data path from config
            raw_data_path = self.config['data'].get('raw_data')
            if raw_data_path and os.path.exists(raw_data_path):
                self.logger.info(f"Using raw data path: {raw_data_path}")
                df = pd.read_parquet(raw_data_path)
                # Filter for the specified year and month if possible
                if 'PIT_DATE' in df.columns:
                    df['year'] = pd.to_datetime(df['PIT_DATE']).dt.year
                    df['month'] = pd.to_datetime(df['PIT_DATE']).dt.month
                    df = df[(df['year'] == year) & (df['month'] == month)]
                    self.logger.info(f"Filtered data for {year}-{month:02d} with shape: {df.shape}")
                    return df
                return df
            
            return pd.DataFrame()
        
        # Load the data
        df = pd.read_parquet(file_path)
        self.logger.info(f"Loaded financial data with shape: {df.shape}")
        
        return df
    
    def load_sector_mapping(self) -> pd.DataFrame:
        """
        Load sector mapping data from both sector files and combine them.
        
        Returns:
            DataFrame containing sector mapping data
        """
        sector_dfs = []
        
        # Load primary sector mapping file
        self.logger.info(f"Loading sector mapping data from {self.sector_mapping_path}")
        if os.path.exists(self.sector_mapping_path):
            try:
                df1 = pd.read_parquet(self.sector_mapping_path)
                self.logger.info(f"Loaded primary sector mapping data with shape: {df1.shape}")
                self.logger.info(f"Primary sector mapping columns: {df1.columns.tolist()}")
                
                # Standardize column names
                df1.columns = [col.lower() for col in df1.columns]
                
                # Ensure ID column is properly named
                if 'id' not in df1.columns and df1.index.name and df1.index.name.lower() == 'id':
                    df1 = df1.reset_index()
                    df1.columns = [col.lower() for col in df1.columns]
                
                # Fix ID format: replace "IN" with "IB" in the middle of the ID
                if 'id' in df1.columns:
                    # Check if IDs contain "IN"
                    sample_id = df1['id'].iloc[0] if not df1.empty else ""
                    self.logger.info(f"Sample ID before conversion: {sample_id}")
                    
                    # Replace "IN" with "IB" in the ID
                    df1['id'] = df1['id'].str.replace(' IN ', ' IB ', regex=False)
                    
                    # Log a sample of the converted IDs
                    sample_id_after = df1['id'].iloc[0] if not df1.empty else ""
                    self.logger.info(f"Sample ID after conversion: {sample_id_after}")
                
                # Rename sector columns to a standard format if needed
                sector_cols = [col for col in df1.columns if 'sector' in col.lower()]
                if sector_cols:
                    rename_dict = {}
                    for i, col in enumerate(sector_cols):
                        rename_dict[col] = f"sector_{i+1}"
                    df1 = df1.rename(columns=rename_dict)
                
                sector_dfs.append(df1)
            except Exception as e:
                self.logger.warning(f"Error loading primary sector mapping data: {e}")
        else:
            self.logger.warning(f"Primary sector mapping file not found: {self.sector_mapping_path}")
        
        # Load secondary sector mapping file if it exists
        self.logger.info(f"Loading secondary sector mapping data from {self.sector_mapping_path2}")
        if os.path.exists(self.sector_mapping_path2):
            try:
                df2 = pd.read_parquet(self.sector_mapping_path2)
                self.logger.info(f"Loaded secondary sector mapping data with shape: {df2.shape}")
                self.logger.info(f"Secondary sector mapping columns: {df2.columns.tolist()}")
                
                # Standardize column names
                df2.columns = [col.lower() for col in df2.columns]
                
                # Ensure ID column is properly named
                if 'id' not in df2.columns and df2.index.name and df2.index.name.lower() == 'id':
                    df2 = df2.reset_index()
                    df2.columns = [col.lower() for col in df2.columns]
                
                # Fix ID format: replace "IN" with "IB" in the middle of the ID
                if 'id' in df2.columns:
                    # Check if IDs contain "IN"
                    sample_id = df2['id'].iloc[0] if not df2.empty else ""
                    self.logger.info(f"Sample ID before conversion: {sample_id}")
                    
                    # Replace "IN" with "IB" in the ID
                    df2['id'] = df2['id'].str.replace(' IN ', ' IB ', regex=False)
                    
                    # Log a sample of the converted IDs
                    sample_id_after = df2['id'].iloc[0] if not df2.empty else ""
                    self.logger.info(f"Sample ID after conversion: {sample_id_after}")
                
                # Rename sector columns to a standard format if needed
                sector_cols = [col for col in df2.columns if 'sector' in col.lower()]
                if sector_cols:
                    rename_dict = {}
                    for i, col in enumerate(sector_cols):
                        rename_dict[col] = f"sector_{i+1}"
                    df2 = df2.rename(columns=rename_dict)
                
                sector_dfs.append(df2)
            except Exception as e:
                self.logger.warning(f"Error loading secondary sector mapping data: {e}")
        else:
            self.logger.warning(f"Secondary sector mapping file not found: {self.sector_mapping_path2}")
        
        # Combine sector DataFrames if we have multiple
        if len(sector_dfs) > 1:
            # Check if they have the same structure
            if set(sector_dfs[0].columns) == set(sector_dfs[1].columns):
                # Simple concatenation if they have the same columns
                combined_df = pd.concat(sector_dfs, ignore_index=True)
                # Remove duplicates if any
                if 'id' in combined_df.columns:
                    combined_df = combined_df.drop_duplicates(subset=['id'])
                
                self.logger.info(f"Combined sector mapping data with shape: {combined_df.shape}")
                return combined_df
            else:
                # If they have different structures, try to merge them
                self.logger.info("Sector files have different structures, attempting to merge")
                
                # Ensure both have ID as a column
                for i, df in enumerate(sector_dfs):
                    if df.index.name == 'id':
                        sector_dfs[i] = df.reset_index()
                
                # Merge on ID
                if 'id' in sector_dfs[0].columns and 'id' in sector_dfs[1].columns:
                    # Get all sector columns from both dataframes
                    sector_cols_1 = [col for col in sector_dfs[0].columns if 'sector' in col.lower()]
                    sector_cols_2 = [col for col in sector_dfs[1].columns if 'sector' in col.lower()]
                    
                    # Create a mapping for renaming to avoid conflicts
                    rename_dict = {}
                    for i, col in enumerate(sector_cols_2):
                        if col in sector_cols_1:
                            rename_dict[col] = f"{col}_2"
                    
                    # Rename columns in the second dataframe to avoid conflicts
                    if rename_dict:
                        sector_dfs[1] = sector_dfs[1].rename(columns=rename_dict)
                    
                    # Merge the dataframes
                    merged_df = pd.merge(sector_dfs[0], sector_dfs[1], on='id', how='outer')
                    self.logger.info(f"Merged sector mapping data with shape: {merged_df.shape}")
                    return merged_df
                else:
                    self.logger.warning("Cannot merge sector files: ID column not found in one or both files")
                    # Return the first one as fallback
                    return sector_dfs[0]
        elif len(sector_dfs) == 1:
            # Return the only DataFrame we have
            return sector_dfs[0]
        else:
            # Return an empty DataFrame if no sector data was loaded
            self.logger.warning("No sector mapping data loaded")
            return pd.DataFrame()
    
    def load_returns_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load price and total returns data.
        
        Returns:
            Dictionary containing price and total returns DataFrames
        """
        returns_data = {}
        
        # Load price returns
        self.logger.info(f"Loading price returns data from {self.price_returns_path}")
        if os.path.exists(self.price_returns_path):
            price_returns = pd.read_parquet(self.price_returns_path)
            self.logger.info(f"Loaded price returns data with shape: {price_returns.shape}")
            returns_data['price_returns'] = price_returns
        else:
            self.logger.warning(f"Price returns file not found: {self.price_returns_path}")
        
        # Load total returns
        self.logger.info(f"Loading total returns data from {self.total_returns_path}")
        if os.path.exists(self.total_returns_path):
            total_returns = pd.read_parquet(self.total_returns_path)
            self.logger.info(f"Loaded total returns data with shape: {total_returns.shape}")
            returns_data['total_returns'] = total_returns
        else:
            self.logger.warning(f"Total returns file not found: {self.total_returns_path}")
        
        return returns_data
    
    def merge_financials_and_sector(self, financials_df: pd.DataFrame, sector_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge financial data with sector mapping.
        
        Args:
            financials_df: DataFrame containing financial data
            sector_df: DataFrame containing sector mapping data
            
        Returns:
            DataFrame containing merged data
        """
        self.logger.info("Merging financial data with sector mapping")
        
        # Check if both DataFrames are not empty
        if financials_df.empty:
            self.logger.warning("Financial data is empty, cannot merge")
            return financials_df
        
        if sector_df.empty:
            self.logger.warning("Sector data is empty, returning financial data as is")
            return financials_df
        
        # Make a copy of the DataFrames to avoid modifying the originals
        financials_copy = financials_df.copy()
        sector_copy = sector_df.copy()
        
        # Normalize column names for consistency
        financials_cols = {col: col.upper() for col in financials_copy.columns}
        financials_copy = financials_copy.rename(columns=financials_cols)
        
        # Ensure ID column is properly named in both dataframes
        if 'id' in sector_copy.columns and 'ID' not in financials_copy.columns:
            financials_copy = financials_copy.rename(columns={'id': 'ID'})
        elif 'ID' in financials_copy.columns and 'id' in sector_copy.columns:
            sector_copy = sector_copy.rename(columns={'id': 'ID'})
        
        # If sector_df has ID as index, reset it to make it a column
        if sector_copy.index.name == 'id' or sector_copy.index.name == 'ID':
            sector_copy = sector_copy.reset_index()
            if 'id' in sector_copy.columns:
                sector_copy = sector_copy.rename(columns={'id': 'ID'})
        
        # Ensure financials_df has ID column
        if 'ID' not in financials_copy.columns:
            self.logger.warning("Financial data does not have ID column, cannot merge")
            return financials_copy
        
        # Ensure sector_df has ID column
        if 'ID' not in sector_copy.columns:
            self.logger.warning("Sector data does not have ID column, cannot merge")
            return financials_copy
        
        # Log sample IDs from both dataframes to verify matching
        fin_sample_ids = financials_copy['ID'].head(5).tolist()
        sec_sample_ids = sector_copy['ID'].head(5).tolist()
        self.logger.info(f"Sample financial IDs: {fin_sample_ids}")
        self.logger.info(f"Sample sector IDs: {sec_sample_ids}")
        
        # Normalize sector column names to uppercase and prefix with SECTOR_
        sector_cols = [col for col in sector_copy.columns if 'sector' in col.lower() and col.lower() != 'id']
        sector_cols_mapping = {col: f"SECTOR_{col.upper()}" for col in sector_cols}
        sector_copy = sector_copy.rename(columns=sector_cols_mapping)
        
        # Log the column names before merging
        self.logger.info(f"Financial data columns: {financials_copy.columns.tolist()}")
        self.logger.info(f"Sector data columns after renaming: {sector_copy.columns.tolist()}")
        
        # Merge the DataFrames
        try:
            merged_df = pd.merge(
                financials_copy,
                sector_copy[['ID'] + list(sector_cols_mapping.values())],
                on='ID',
                how='left'
            )
            
            self.logger.info(f"Merged data shape: {merged_df.shape}")
            
            # Check if sector columns were added
            sector_cols = [col for col in merged_df.columns if 'SECTOR_' in col]
            if sector_cols:
                self.logger.info(f"Added sector columns: {sector_cols}")
                
                # Check for null values in sector columns
                null_counts = merged_df[sector_cols].isnull().sum()
                self.logger.info(f"Null counts in sector columns: {null_counts.to_dict()}")
                
                # Calculate percentage of rows with sector data
                total_rows = len(merged_df)
                non_null_rows = total_rows - merged_df[sector_cols[0]].isnull().sum()
                coverage_pct = (non_null_rows / total_rows) * 100 if total_rows > 0 else 0
                self.logger.info(f"Sector data coverage: {coverage_pct:.2f}% ({non_null_rows}/{total_rows} rows)")
                
                # Sample of merged data to verify
                sample_size = min(5, len(merged_df))
                sample_df = merged_df.sample(sample_size)[['ID'] + sector_cols]
                self.logger.info(f"Sample of merged data with sector columns:\n{sample_df}")
            else:
                self.logger.warning("No sector columns were added in the merge")
            
            return merged_df
        except Exception as e:
            self.logger.error(f"Error merging financial and sector data: {e}")
            return financials_copy
    
    def store_fundamentals_data(self, df: pd.DataFrame, year: int, month: int) -> str:
        """
        Store fundamentals data (financials + sector).
        
        Args:
            df: DataFrame containing fundamentals data
            year: Year of the data
            month: Month of the data
            
        Returns:
            Path to the saved file
        """
        self.logger.info(f"Storing fundamentals data for {year}-{month:02d}")
        
        # Add year and month columns if not present
        if 'YEAR' not in df.columns:
            df['YEAR'] = year
        if 'MONTH' not in df.columns:
            df['MONTH'] = month
        
        # Define the file path
        file_path = os.path.join(self.fundamentals_dir, f"fundamentals_{year}_{month:02d}.pq")
        
        # Store the data
        df.to_parquet(file_path)
        self.logger.info(f"Stored fundamentals data to {file_path}")
        
        # Log some statistics about the stored data
        self.logger.info(f"Stored data shape: {df.shape}")
        
        # Log sector columns specifically
        sector_cols = [col for col in df.columns if 'SECTOR_' in col]
        if sector_cols:
            self.logger.info(f"Sector columns in stored data: {sector_cols}")
            
            # Sample of sector data
            sample_size = min(5, len(df))
            sector_sample = df.sample(sample_size)[['ID'] + sector_cols]
            self.logger.info(f"Sample of sector data:\n{sector_sample}")
            
            # Count of unique values in each sector column
            for col in sector_cols:
                unique_values = df[col].nunique()
                self.logger.info(f"Column {col} has {unique_values} unique values")
        
        return file_path
    
    def store_returns_data(self, returns_data: Dict[str, pd.DataFrame]) -> str:
        """
        Store returns data (price + total).
        
        Args:
            returns_data: Dictionary containing price and total returns DataFrames
            
        Returns:
            Path to the saved file
        """
        self.logger.info("Storing returns data")
        
        # Check if returns data is available
        if not returns_data:
            self.logger.warning("No returns data to store")
            return ""
        
        # Store price returns
        if 'price_returns' in returns_data:
            price_df = returns_data['price_returns'].copy()
            
            # Normalize column names to uppercase for consistency
            price_df.columns = [col.upper() for col in price_df.columns]
            
            # Ensure ID column is uppercase
            if 'id' in price_df.columns:
                price_df = price_df.rename(columns={'id': 'ID'})
            
            # Rename return column if it exists
            if 'RET' in price_df.columns:
                price_df = price_df.rename(columns={'RET': 'PRICE_RET'})
            
            price_returns_path = os.path.join(self.returns_dir, "price_returns.pq")
            price_df.to_parquet(price_returns_path)
            self.logger.info(f"Stored price returns data to {price_returns_path}")
        
        # Store total returns
        if 'total_returns' in returns_data:
            total_df = returns_data['total_returns'].copy()
            
            # Normalize column names to uppercase for consistency
            total_df.columns = [col.upper() for col in total_df.columns]
            
            # Ensure ID column is uppercase
            if 'id' in total_df.columns:
                total_df = total_df.rename(columns={'id': 'ID'})
            
            # Rename return column if it exists
            if 'RET' in total_df.columns:
                total_df = total_df.rename(columns={'RET': 'TOTAL_RET'})
            
            total_returns_path = os.path.join(self.returns_dir, "total_returns.pq")
            total_df.to_parquet(total_returns_path)
            self.logger.info(f"Stored total returns data to {total_returns_path}")
        
        # Store combined returns
        if 'price_returns' in returns_data and 'total_returns' in returns_data:
            # Get the normalized dataframes
            price_df = returns_data['price_returns'].copy()
            total_df = returns_data['total_returns'].copy()
            
            # Normalize column names to uppercase for consistency
            price_df.columns = [col.upper() for col in price_df.columns]
            total_df.columns = [col.upper() for col in total_df.columns]
            
            # Ensure ID column is uppercase
            if 'id' in price_df.columns:
                price_df = price_df.rename(columns={'id': 'ID'})
            if 'id' in total_df.columns:
                total_df = total_df.rename(columns={'id': 'ID'})
            
            # Rename return columns to avoid conflicts
            if 'RET' in price_df.columns:
                price_df = price_df.rename(columns={'RET': 'PRICE_RET'})
            if 'RET' in total_df.columns:
                total_df = total_df.rename(columns={'RET': 'TOTAL_RET'})
            
            # Identify common columns for merging
            common_cols = [col for col in price_df.columns if col in total_df.columns and col not in ['PRICE_RET', 'TOTAL_RET']]
            
            if common_cols:
                self.logger.info(f"Merging returns data on columns: {common_cols}")
                
                # Merge the DataFrames
                merged_df = pd.merge(
                    price_df,
                    total_df,
                    on=common_cols,
                    how='outer'
                )
                
                combined_returns_path = os.path.join(self.returns_dir, "combined_returns.pq")
                merged_df.to_parquet(combined_returns_path)
                self.logger.info(f"Stored combined returns data to {combined_returns_path}")
                
                # Log some statistics about the combined data
                self.logger.info(f"Combined returns data shape: {merged_df.shape}")
                self.logger.info(f"Combined returns data columns: {merged_df.columns.tolist()}")
                
                # Sample of combined data
                sample_size = min(5, len(merged_df))
                sample_df = merged_df.sample(sample_size)
                self.logger.info(f"Sample of combined returns data:\n{sample_df}")
        
        return self.returns_dir
    
    def organize_data_storage(self, year: int, month: int) -> Dict[str, str]:
        """
        Organize the data storage structure.
        
        Args:
            year: Year to process data for
            month: Month to process data for
            
        Returns:
            Dictionary containing paths to the stored data
        """
        self.logger.info(f"Organizing data storage for {year}-{month:02d}")
        
        # Load one month of financial data
        financials_df = self.load_one_month_financials(year, month)
        if financials_df.empty:
            self.logger.error(f"No financial data found for {year}-{month:02d}")
            return {}
        
        # Load sector mapping
        sector_df = self.load_sector_mapping()
        
        # Merge financials and sector data
        fundamentals_df = self.merge_financials_and_sector(financials_df, sector_df)
        
        # Store fundamentals data
        fundamentals_path = self.store_fundamentals_data(fundamentals_df, year, month)
        
        # Load returns data
        returns_data = self.load_returns_data()
        
        # Store returns data
        returns_path = self.store_returns_data(returns_data)
        
        return {
            'fundamentals_path': fundamentals_path,
            'returns_path': returns_path
        }

def load_config() -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Returns:
        Configuration dictionary
    """
    import yaml
    
    config_path = "config/config.yaml"
    
    # Check if the file exists
    if not os.path.exists(config_path):
        logger.error(f"Configuration file not found: {config_path}")
        return {}
    
    # Load the configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Loaded configuration from {config_path}")
    
    return config

def main():
    """
    Main function to organize the data storage structure.
    """
    logger.info("Starting to organize the data storage structure...")
    
    # Load configuration
    config = load_config()
    if not config:
        logger.error("Failed to load configuration")
        return
    
    # Initialize the data storage organizer
    organizer = DataStorageOrganizer(config, logger)
    
    # Process one month of data (September 2024 as specified in the config)
    year = 2024
    month = 9
    
    # Organize the data storage
    result = organizer.organize_data_storage(year, month)
    
    if result:
        logger.info("Data storage organization completed successfully")
        logger.info(f"Fundamentals data stored at: {result.get('fundamentals_path', 'N/A')}")
        logger.info(f"Returns data stored at: {result.get('returns_path', 'N/A')}")
    else:
        logger.error("Data storage organization failed")

if __name__ == "__main__":
    main() 
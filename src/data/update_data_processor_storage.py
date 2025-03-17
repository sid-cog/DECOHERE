#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to update the DataProcessor class to use the new storage structure:
1. Financials and sector data stored together in fundamentals folder
2. Price and total returns data stored together in returns folder
"""

import os
import sys
import logging
import shutil
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger('update_data_processor_storage')

def main():
    """
    Main function to update the DataProcessor class.
    """
    logger.info("Starting to update the DataProcessor class to use the new storage structure...")
    
    # Define paths
    data_processor_path = "src/data/data_processor.py"
    backup_path = f"{data_processor_path}.bak_new_storage"
    
    # Create backup
    logger.info(f"Creating backup at {backup_path}")
    shutil.copy2(data_processor_path, backup_path)
    
    # Read current implementation
    logger.info("Reading current data processor implementation")
    with open(data_processor_path, 'r') as f:
        content = f.read()
    
    # Update the __init__ method to include new storage paths
    init_pattern = """    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        \"\"\"
        Initialize the data processor.
        
        Args:
            config: Configuration dictionary
            logger: Logger to use (if None, create a new logger)
        \"\"\"
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
        
        # Create processed data directory if it doesn't exist
        os.makedirs(self.processed_data_dir, exist_ok=True)
        
        # Initialize efficient data storage
        self.data_storage = EfficientDataStorage(config, logger)"""
    
    updated_init = """    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        \"\"\"
        Initialize the data processor.
        
        Args:
            config: Configuration dictionary
            logger: Logger to use (if None, create a new logger)
        \"\"\"
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Extract configuration parameters
        self.raw_data_path = config['data']['raw_data']
        self.processed_data_dir = config['data']['processed_data']
        self.enable_filling = config['processing'].get('enable_filling', True)
        self.winsorize_threshold = config['processing'].get('winsorize_threshold', 3.0)
        
        # Original data source paths (for backward compatibility)
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
        self.data_storage = EfficientDataStorage(config, logger)"""
    
    content = content.replace(init_pattern, updated_init)
    
    # Update the load_raw_data method to use the new storage structure
    load_raw_data_pattern = """    def load_raw_data(self, date: Optional[str] = None) -> pd.DataFrame:
        \"\"\"
        Load raw financial data.
        
        Args:
            date: Date to filter data for (if None, load all data)
            
        Returns:
            DataFrame containing raw financial data
        \"\"\"
        # Determine the file path based on the date
        if date:
            year, month, _ = date.split('-')
            financials_file = os.path.join(self.financials_dir, f"financials_{year}_{month}.pq")
            self.logger.info(f"Loading raw data from {financials_file}")
            
            if os.path.exists(financials_file):
                # Load raw data for the specific month
                df = pd.read_parquet(financials_file)
            else:
                # Fall back to the consolidated file if available
                consolidated_file = os.path.join(self.financials_dir, "processed_financials.pq")
                if os.path.exists(consolidated_file):
                    self.logger.info(f"File {financials_file} not found, using consolidated file {consolidated_file}")
                    df = pd.read_parquet(consolidated_file)
                else:
                    # Fall back to the original raw data path
                    self.logger.info(f"Using default raw data path: {self.raw_data_path}")
                    df = pd.read_parquet(self.raw_data_path)
        else:
            # If no date is specified, use the consolidated file if available
            consolidated_file = os.path.join(self.financials_dir, "processed_financials.pq")
            if os.path.exists(consolidated_file):
                self.logger.info(f"Loading raw data from consolidated file {consolidated_file}")
                df = pd.read_parquet(consolidated_file)
            else:
                # Fall back to the original raw data path
                self.logger.info(f"Using default raw data path: {self.raw_data_path}")
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
        
        return df"""
    
    updated_load_raw_data = """    def load_raw_data(self, date: Optional[str] = None) -> pd.DataFrame:
        \"\"\"
        Load raw financial data.
        
        Args:
            date: Date to filter data for (if None, load all data)
            
        Returns:
            DataFrame containing raw financial data
        \"\"\"
        # Determine the file path based on the date
        if date:
            year, month, _ = date.split('-')
            
            # Try loading from the new fundamentals directory first
            fundamentals_file = os.path.join(self.fundamentals_dir, f"fundamentals_{year}_{month}.pq")
            if os.path.exists(fundamentals_file):
                self.logger.info(f"Loading raw data from fundamentals file: {fundamentals_file}")
                df = pd.read_parquet(fundamentals_file)
                # Filter out sector columns if needed
                sector_cols = [col for col in df.columns if 'sector' in col.lower()]
                if sector_cols:
                    self.logger.info(f"Found sector columns in fundamentals data: {sector_cols}")
            else:
                # Fall back to the original financials directory
                financials_file = os.path.join(self.financials_dir, f"financials_{year}_{month}.pq")
                self.logger.info(f"Fundamentals file not found, trying financials file: {financials_file}")
                
                if os.path.exists(financials_file):
                    # Load raw data for the specific month
                    df = pd.read_parquet(financials_file)
                else:
                    # Fall back to the consolidated file if available
                    consolidated_file = os.path.join(self.financials_dir, "processed_financials.pq")
                    if os.path.exists(consolidated_file):
                        self.logger.info(f"File {financials_file} not found, using consolidated file {consolidated_file}")
                        df = pd.read_parquet(consolidated_file)
                    else:
                        # Fall back to the original raw data path
                        self.logger.info(f"Using default raw data path: {self.raw_data_path}")
                        df = pd.read_parquet(self.raw_data_path)
        else:
            # If no date is specified, try to load from the fundamentals directory
            fundamentals_files = [f for f in os.listdir(self.fundamentals_dir) 
                                if f.startswith('fundamentals_') and f.endswith('.pq')]
            
            if fundamentals_files:
                # Use the most recent fundamentals file
                latest_file = sorted(fundamentals_files)[-1]
                fundamentals_file = os.path.join(self.fundamentals_dir, latest_file)
                self.logger.info(f"Loading raw data from latest fundamentals file: {fundamentals_file}")
                df = pd.read_parquet(fundamentals_file)
            else:
                # Fall back to the consolidated file if available
                consolidated_file = os.path.join(self.financials_dir, "processed_financials.pq")
                if os.path.exists(consolidated_file):
                    self.logger.info(f"Loading raw data from consolidated file {consolidated_file}")
                    df = pd.read_parquet(consolidated_file)
                else:
                    # Fall back to the original raw data path
                    self.logger.info(f"Using default raw data path: {self.raw_data_path}")
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
        
        return df"""
    
    content = content.replace(load_raw_data_pattern, updated_load_raw_data)
    
    # Update the methods for loading sector mapping and returns data
    load_sector_mapping_pattern = """    def load_sector_mapping(self) -> pd.DataFrame:
        \"\"\"
        Load sector mapping data.
        
        Returns:
            DataFrame containing sector mapping data
        \"\"\"
        self.logger.info(f"Loading sector mapping data from {self.sector_mapping_path}")
        
        # Load sector mapping data
        df = pd.read_parquet(self.sector_mapping_path)
        
        self.logger.info(f"Loaded sector mapping data with shape: {df.shape}")
        
        return df"""
    
    updated_load_sector_mapping = """    def load_sector_mapping(self) -> pd.DataFrame:
        \"\"\"
        Load sector mapping data.
        
        Returns:
            DataFrame containing sector mapping data
        \"\"\"
        # Try to extract sector data from fundamentals first
        fundamentals_files = [f for f in os.listdir(self.fundamentals_dir) 
                            if f.startswith('fundamentals_') and f.endswith('.pq')]
        
        if fundamentals_files:
            # Use the most recent fundamentals file
            latest_file = sorted(fundamentals_files)[-1]
            fundamentals_file = os.path.join(self.fundamentals_dir, latest_file)
            self.logger.info(f"Extracting sector data from fundamentals file: {fundamentals_file}")
            
            try:
                df = pd.read_parquet(fundamentals_file)
                
                # Extract sector columns
                sector_cols = [col for col in df.columns if 'sector' in col.lower()]
                
                if sector_cols:
                    self.logger.info(f"Found sector columns: {sector_cols}")
                    sector_df = df[['ID'] + sector_cols].drop_duplicates()
                    self.logger.info(f"Extracted sector data with shape: {sector_df.shape}")
                    return sector_df
                else:
                    self.logger.warning("No sector columns found in fundamentals data")
            except Exception as e:
                self.logger.warning(f"Error extracting sector data from fundamentals: {e}")
        
        # Fall back to the original sector mapping file
        self.logger.info(f"Loading sector mapping data from {self.sector_mapping_path}")
        
        try:
            # Load sector mapping data
            df = pd.read_parquet(self.sector_mapping_path)
            self.logger.info(f"Loaded sector mapping data with shape: {df.shape}")
            return df
        except Exception as e:
            self.logger.error(f"Error loading sector mapping data: {e}")
            return pd.DataFrame()"""
    
    content = content.replace(load_sector_mapping_pattern, updated_load_sector_mapping)
    
    load_price_returns_pattern = """    def load_price_returns(self) -> pd.DataFrame:
        \"\"\"
        Load price returns data.
        
        Returns:
            DataFrame containing price returns data
        \"\"\"
        self.logger.info(f"Loading price returns data from {self.price_returns_path}")
        
        # Load price returns data
        df = pd.read_parquet(self.price_returns_path)
        
        self.logger.info(f"Loaded price returns data with shape: {df.shape}")
        
        return df"""
    
    updated_load_price_returns = """    def load_price_returns(self) -> pd.DataFrame:
        \"\"\"
        Load price returns data.
        
        Returns:
            DataFrame containing price returns data
        \"\"\"
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
        
        # Fall back to the original price returns file
        self.logger.info(f"Loading price returns data from original path: {self.price_returns_path}")
        
        try:
            # Load price returns data
            df = pd.read_parquet(self.price_returns_path)
            self.logger.info(f"Loaded price returns data with shape: {df.shape}")
            return df
        except Exception as e:
            self.logger.error(f"Error loading price returns data: {e}")
            return pd.DataFrame()"""
    
    content = content.replace(load_price_returns_pattern, updated_load_price_returns)
    
    load_total_returns_pattern = """    def load_total_returns(self) -> pd.DataFrame:
        \"\"\"
        Load total returns data.
        
        Returns:
            DataFrame containing total returns data
        \"\"\"
        self.logger.info(f"Loading total returns data from {self.total_returns_path}")
        
        # Load total returns data
        df = pd.read_parquet(self.total_returns_path)
        
        self.logger.info(f"Loaded total returns data with shape: {df.shape}")
        
        return df"""
    
    updated_load_total_returns = """    def load_total_returns(self) -> pd.DataFrame:
        \"\"\"
        Load total returns data.
        
        Returns:
            DataFrame containing total returns data
        \"\"\"
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
        
        # Fall back to the original total returns file
        self.logger.info(f"Loading total returns data from original path: {self.total_returns_path}")
        
        try:
            # Load total returns data
            df = pd.read_parquet(self.total_returns_path)
            self.logger.info(f"Loaded total returns data with shape: {df.shape}")
            return df
        except Exception as e:
            self.logger.error(f"Error loading total returns data: {e}")
            return pd.DataFrame()"""
    
    content = content.replace(load_total_returns_pattern, updated_load_total_returns)
    
    # Write the updated content
    logger.info("Writing updated data processor implementation")
    with open(data_processor_path, 'w') as f:
        f.write(content)
    
    logger.info("DataProcessor class has been updated to use the new storage structure")

if __name__ == "__main__":
    main() 
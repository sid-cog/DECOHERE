#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to update the DataProcessor class to use all three data sources:
1. Financials data from /home/siddharth_johri/projects/data/financials/
2. Sector mapping data from /home/siddharth_johri/projects/data/sector/
3. Returns data from /home/siddharth_johri/projects/data/returns/
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
logger = logging.getLogger('update_data_sources')

def main():
    """
    Main function to update the DataProcessor class.
    """
    logger.info("Starting to update the DataProcessor class to use all three data sources...")
    
    # Define paths
    data_processor_path = "src/data/data_processor.py"
    backup_path = f"{data_processor_path}.bak_data_sources"
    
    # Create backup
    logger.info(f"Creating backup at {backup_path}")
    shutil.copy2(data_processor_path, backup_path)
    
    # Read current implementation
    logger.info("Reading current data processor implementation")
    with open(data_processor_path, 'r') as f:
        content = f.read()
    
    # Update the __init__ method to include new data paths
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
        
        # Create processed data directory if it doesn't exist
        os.makedirs(self.processed_data_dir, exist_ok=True)
        
        # Initialize efficient data storage
        self.data_storage = EfficientDataStorage(config, logger)
        
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
        
        # Data source paths
        self.financials_dir = config['data'].get('financials_dir', '/home/siddharth_johri/projects/data/financials')
        self.sector_mapping_path = config['data'].get('sector_mapping', '/home/siddharth_johri/projects/data/sector/sector.pq')
        self.price_returns_path = config['data'].get('price_returns', '/home/siddharth_johri/projects/data/returns/px_df.pq')
        self.total_returns_path = config['data'].get('total_returns', '/home/siddharth_johri/projects/data/returns/tr_df.pq')
        
        # Create processed data directory if it doesn't exist
        os.makedirs(self.processed_data_dir, exist_ok=True)
        
        # Initialize efficient data storage
        self.data_storage = EfficientDataStorage(config, logger)"""
    
    content = content.replace(init_pattern, updated_init)
    
    # Update the load_raw_data method to handle different data sources
    load_raw_data_pattern = """    def load_raw_data(self, date: Optional[str] = None) -> pd.DataFrame:
        \"\"\"
        Load raw financial data.
        
        Args:
            date: Date to filter data for (if None, load all data)
            
        Returns:
            DataFrame containing raw financial data
        \"\"\"
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
    
    content = content.replace(load_raw_data_pattern, updated_load_raw_data)
    
    # Add new methods for loading sector mapping and returns data
    # Find the position to add the new methods (after the transform_raw_data method)
    transform_raw_data_end = """        return transformed_df"""
    
    new_methods = """        return transformed_df
    
    def load_sector_mapping(self) -> pd.DataFrame:
        \"\"\"
        Load sector mapping data.
        
        Returns:
            DataFrame containing sector mapping data
        \"\"\"
        self.logger.info(f"Loading sector mapping data from {self.sector_mapping_path}")
        
        # Load sector mapping data
        df = pd.read_parquet(self.sector_mapping_path)
        
        self.logger.info(f"Loaded sector mapping data with shape: {df.shape}")
        
        return df
    
    def load_price_returns(self) -> pd.DataFrame:
        \"\"\"
        Load price returns data.
        
        Returns:
            DataFrame containing price returns data
        \"\"\"
        self.logger.info(f"Loading price returns data from {self.price_returns_path}")
        
        # Load price returns data
        df = pd.read_parquet(self.price_returns_path)
        
        self.logger.info(f"Loaded price returns data with shape: {df.shape}")
        
        return df
    
    def load_total_returns(self) -> pd.DataFrame:
        \"\"\"
        Load total returns data.
        
        Returns:
            DataFrame containing total returns data
        \"\"\"
        self.logger.info(f"Loading total returns data from {self.total_returns_path}")
        
        # Load total returns data
        df = pd.read_parquet(self.total_returns_path)
        
        self.logger.info(f"Loaded total returns data with shape: {df.shape}")
        
        return df
    
    def load_all_data_sources(self, date: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        \"\"\"
        Load all data sources.
        
        Args:
            date: Date to filter financial data for (if None, load all data)
            
        Returns:
            Dictionary containing all data sources
        \"\"\"
        self.logger.info("Loading all data sources...")
        
        data = {}
        
        # Load financial data
        data['financials'] = self.load_raw_data(date)
        
        # Load sector mapping data
        data['sector_mapping'] = self.load_sector_mapping()
        
        # Load returns data
        data['price_returns'] = self.load_price_returns()
        data['total_returns'] = self.load_total_returns()
        
        return data"""
    
    content = content.replace(transform_raw_data_end, new_methods)
    
    # Write the updated content
    logger.info("Writing updated data processor implementation")
    with open(data_processor_path, 'w') as f:
        f.write(content)
    
    logger.info("DataProcessor class has been updated to use all three data sources")

if __name__ == "__main__":
    main() 
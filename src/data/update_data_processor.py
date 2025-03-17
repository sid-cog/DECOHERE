#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to update the DataProcessor class to load sector and returns data and pass them to the FeatureGenerator.
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
logger = logging.getLogger('update_data_processor')

def main():
    """
    Main function to update the DataProcessor class.
    """
    logger.info("Starting to update the DataProcessor class...")
    
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
    
    # Update the imports
    imports_pattern = """import os
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union"""
    
    updated_imports = """import os
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import pyarrow.parquet as pq"""
    
    content = content.replace(imports_pattern, updated_imports)
    
    # Update the __init__ method to include new data source paths
    init_pattern = """    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        \"\"\"
        Initialize the DataProcessor.
        
        Args:
            config: Configuration dictionary
            logger: Logger instance
        \"\"\"
        self.config = config
        
        # Set up logging
        if logger is None:
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.INFO)
            
            # Create console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            
            # Create formatter
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            
            # Add handler to logger
            self.logger.addHandler(console_handler)
        else:
            self.logger = logger
            
        # Set up paths
        self.raw_data_dir = self.config['data']['raw_data_dir']
        self.processed_data_dir = self.config['data']['processed_data_dir']
        
        # Create directories if they don't exist
        os.makedirs(self.raw_data_dir, exist_ok=True)
        os.makedirs(self.processed_data_dir, exist_ok=True)"""
    
    updated_init = """    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        \"\"\"
        Initialize the DataProcessor.
        
        Args:
            config: Configuration dictionary
            logger: Logger instance
        \"\"\"
        self.config = config
        
        # Set up logging
        if logger is None:
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.INFO)
            
            # Create console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            
            # Create formatter
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            
            # Add handler to logger
            self.logger.addHandler(console_handler)
        else:
            self.logger = logger
            
        # Set up paths
        self.raw_data_dir = self.config['data']['raw_data_dir']
        self.processed_data_dir = self.config['data']['processed_data_dir']
        
        # Set up paths for additional data sources
        self.financials_dir = self.config['data'].get('financials_dir', None)
        self.sector_mapping_path = self.config['data'].get('sector_mapping', None)
        self.price_returns_path = self.config['data'].get('price_returns', None)
        self.total_returns_path = self.config['data'].get('total_returns', None)
        
        # Create directories if they don't exist
        os.makedirs(self.raw_data_dir, exist_ok=True)
        os.makedirs(self.processed_data_dir, exist_ok=True)"""
    
    content = content.replace(init_pattern, updated_init)
    
    # Add methods to load sector and returns data
    process_features_pattern = """    def process_features(self, date: str) -> str:
        \"\"\"
        Process features for a specific date.
        
        Args:
            date: Date to process features for
            
        Returns:
            Path to the saved features file
        \"\"\"
        self.logger.info(f"Processing features for date: {date}")
        
        # Initialize the FeatureGenerator
        from src.features.feature_generator import FeatureGenerator
        feature_generator = FeatureGenerator(self.config, self.logger)
        
        # Process features
        file_path = feature_generator.process_features(date)
        
        return file_path"""
    
    updated_process_features = """    def load_sector_data(self) -> Optional[pd.DataFrame]:
        \"\"\"
        Load sector mapping data.
        
        Returns:
            DataFrame containing sector mapping data, or None if the file doesn't exist
        \"\"\"
        if self.sector_mapping_path is None:
            self.logger.warning("No sector mapping path specified in config")
            return None
            
        if not os.path.exists(self.sector_mapping_path):
            self.logger.warning(f"Sector mapping file not found at: {self.sector_mapping_path}")
            return None
            
        try:
            self.logger.info(f"Loading sector mapping data from: {self.sector_mapping_path}")
            
            # Check file extension
            if self.sector_mapping_path.endswith('.pq'):
                # Load Parquet file
                sector_data = pd.read_parquet(self.sector_mapping_path)
            elif self.sector_mapping_path.endswith('.csv'):
                # Load CSV file
                sector_data = pd.read_csv(self.sector_mapping_path)
            else:
                self.logger.warning(f"Unsupported file format for sector mapping: {self.sector_mapping_path}")
                return None
                
            self.logger.info(f"Loaded sector mapping data with shape: {sector_data.shape}")
            return sector_data
        except Exception as e:
            self.logger.error(f"Error loading sector mapping data: {str(e)}")
            return None
    
    def load_returns_data(self, returns_type: str = 'price') -> Optional[pd.DataFrame]:
        \"\"\"
        Load returns data.
        
        Args:
            returns_type: Type of returns to load ('price' or 'total')
            
        Returns:
            DataFrame containing returns data, or None if the file doesn't exist
        \"\"\"
        if returns_type == 'price':
            returns_path = self.price_returns_path
        elif returns_type == 'total':
            returns_path = self.total_returns_path
        else:
            self.logger.warning(f"Invalid returns type: {returns_type}")
            return None
            
        if returns_path is None:
            self.logger.warning(f"No {returns_type} returns path specified in config")
            return None
            
        if not os.path.exists(returns_path):
            self.logger.warning(f"{returns_type.capitalize()} returns file not found at: {returns_path}")
            return None
            
        try:
            self.logger.info(f"Loading {returns_type} returns data from: {returns_path}")
            
            # Check file extension
            if returns_path.endswith('.pq'):
                # Load Parquet file
                returns_data = pd.read_parquet(returns_path)
            elif returns_path.endswith('.csv'):
                # Load CSV file
                returns_data = pd.read_csv(returns_path)
            else:
                self.logger.warning(f"Unsupported file format for {returns_type} returns: {returns_path}")
                return None
                
            self.logger.info(f"Loaded {returns_type} returns data with shape: {returns_data.shape}")
            return returns_data
        except Exception as e:
            self.logger.error(f"Error loading {returns_type} returns data: {str(e)}")
            return None
    
    def process_features(self, date: str) -> str:
        \"\"\"
        Process features for a specific date.
        
        Args:
            date: Date to process features for
            
        Returns:
            Path to the saved features file
        \"\"\"
        self.logger.info(f"Processing features for date: {date}")
        
        # Load additional data sources
        sector_data = self.load_sector_data()
        price_returns = self.load_returns_data('price')
        total_returns = self.load_returns_data('total')
        
        # Initialize the FeatureGenerator
        from src.features.feature_generator import FeatureGenerator
        feature_generator = FeatureGenerator(self.config, self.logger)
        
        # Process features
        file_path = feature_generator.process_features(date, sector_data, price_returns, total_returns)
        
        return file_path"""
    
    content = content.replace(process_features_pattern, updated_process_features)
    
    # Write the updated content
    logger.info("Writing updated data processor implementation")
    with open(data_processor_path, 'w') as f:
        f.write(content)
    
    logger.info("DataProcessor class has been updated to load sector and returns data")

if __name__ == "__main__":
    main() 
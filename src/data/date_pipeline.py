import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Tuple, Dict, Optional

from .data_processor import DataProcessor
from .data_storage import DataStorage, DataType, DataStage

class DatePipeline:
    """Class to handle date-by-date pipeline execution."""
    
    def __init__(self, config: dict, logger: logging.Logger):
        """
        Initialize the date pipeline.
        
        Args:
            config: Configuration dictionary
            logger: Logger instance
        """
        self.config = config
        self.logger = logger
        self.data_processor = DataProcessor(config, logger)
        self.data_storage = DataStorage(config, logger)
    
    def process_single_date(self, date_str: str) -> Tuple[bool, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        Process data for a single date.
        
        Args:
            date_str: Date string in YYYY-MM-DD format
            
        Returns:
            Tuple of (success, processed_df, feature_df)
        """
        try:
            self.logger.info(f"Processing data for date: {date_str}")
            
            # 1. Load raw data
            raw_data = self.data_processor.load_raw_data(date=date_str)
            if raw_data.empty:
                self.logger.error(f"No raw data found for date: {date_str}")
                return False, None, None
                
            self.logger.info(f"Loaded raw data with shape: {raw_data.shape}")
            
            # 2. Process the data
            processed_files = self.data_processor.process_data(start_date=date_str, end_date=date_str)
            
            if date_str not in processed_files:
                self.logger.error(f"Failed to process data for date: {date_str}")
                return False, None, None
                
            # 3. Load processed data
            processed_df = self.data_processor.load_processed_data_by_mode(
                mode='day',
                date=date_str
            )
            
            # 4. Generate features
            feature_df = self.data_processor.processed_data_feat_gen(processed_df)
            
            self.logger.info(f"Successfully processed data for {date_str}")
            return True, processed_df, feature_df
            
        except Exception as e:
            self.logger.error(f"Error processing date {date_str}: {str(e)}")
            return False, None, None
    
    def process_date_range(self, start_date: str, end_date: str) -> Dict[str, Tuple[bool, Optional[pd.DataFrame], Optional[pd.DataFrame]]]:
        """
        Process data for a range of dates.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            Dictionary mapping dates to (success, processed_df, feature_df) tuples
        """
        results = {}
        current_date = datetime.strptime(start_date, '%Y-%m-%d')
        end_date = datetime.strptime(end_date, '%Y-%m-%d')
        
        while current_date <= end_date:
            date_str = current_date.strftime('%Y-%m-%d')
            success, processed_df, feature_df = self.process_single_date(date_str)
            results[date_str] = (success, processed_df, feature_df)
            current_date += timedelta(days=1)
            
        return results 
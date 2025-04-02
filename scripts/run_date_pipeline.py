import os
import sys
import pandas as pd
from datetime import datetime, timedelta
import logging
from pathlib import Path

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from src.data.data_processor import DataProcessor
from src.data.data_storage import DataStorage, DataType, DataStage
from src.utils.config import load_config
from src.utils.logging import setup_logging

def process_single_date(date_str: str, config: dict, logger: logging.Logger) -> bool:
    """
    Process data for a single date.
    
    Args:
        date_str: Date string in YYYY-MM-DD format
        config: Configuration dictionary
        logger: Logger instance
        
    Returns:
        bool: True if processing was successful, False otherwise
    """
    try:
        # Initialize data processor and storage
        data_processor = DataProcessor(config, logger)
        data_storage = DataStorage(config, logger)
        
        logger.info(f"Processing data for date: {date_str}")
        
        # 1. Load raw data
        raw_data = data_processor.load_raw_data(date=date_str)
        if raw_data.empty:
            logger.error(f"No raw data found for date: {date_str}")
            return False
            
        logger.info(f"Loaded raw data with shape: {raw_data.shape}")
        
        # 2. Process the data
        processed_files = data_processor.process_data(start_date=date_str, end_date=date_str)
        
        if date_str not in processed_files:
            logger.error(f"Failed to process data for date: {date_str}")
            return False
            
        logger.info(f"Successfully processed data for {date_str}")
        logger.info(f"Output files: {processed_files}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error processing date {date_str}: {str(e)}")
        return False

def process_date_range(start_date: str, end_date: str, config: dict, logger: logging.Logger) -> dict:
    """
    Process data for a range of dates.
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        config: Configuration dictionary
        logger: Logger instance
        
    Returns:
        dict: Dictionary with processing results for each date
    """
    results = {}
    current_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.strptime(end_date, '%Y-%m-%d')
    
    while current_date <= end_date:
        date_str = current_date.strftime('%Y-%m-%d')
        success = process_single_date(date_str, config, logger)
        results[date_str] = success
        current_date += timedelta(days=1)
        
    return results

def main():
    """Main function to run the date pipeline."""
    # Set up logging
    logger = setup_logging()
    logger.info("Starting date pipeline execution")
    
    # Load configuration
    config = load_config()
    
    # Get command line arguments
    if len(sys.argv) < 2:
        logger.error("Please provide at least one date in YYYY-MM-DD format")
        sys.exit(1)
        
    # Process single date or date range
    if len(sys.argv) == 2:
        date_str = sys.argv[1]
        success = process_single_date(date_str, config, logger)
        if success:
            logger.info(f"Successfully processed data for {date_str}")
        else:
            logger.error(f"Failed to process data for {date_str}")
            sys.exit(1)
    else:
        start_date = sys.argv[1]
        end_date = sys.argv[2]
        results = process_date_range(start_date, end_date, config, logger)
        
        # Print summary
        successful_dates = sum(1 for success in results.values() if success)
        total_dates = len(results)
        logger.info(f"\nProcessing Summary:")
        logger.info(f"Total dates processed: {total_dates}")
        logger.info(f"Successful dates: {successful_dates}")
        logger.info(f"Failed dates: {total_dates - successful_dates}")
        
        if successful_dates < total_dates:
            logger.error("Some dates failed to process")
            sys.exit(1)

if __name__ == "__main__":
    main() 
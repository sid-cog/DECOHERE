#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to update the pipeline.ipynb notebook to use the new EfficientDataStorage system.
"""

import os
import sys
import logging
import json
import re
import shutil
from pathlib import Path
import nbformat as nbf

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger('update_pipeline')

def main():
    """
    Main function to update the pipeline notebook.
    """
    logger.info("Starting to update the pipeline notebook...")
    
    # Define paths
    pipeline_path = "notebooks/pipeline.ipynb"
    backup_path = f"{pipeline_path}.bak_before_efficient_storage"
    
    # Create backup
    logger.info(f"Creating backup at {backup_path}")
    shutil.copy2(pipeline_path, backup_path)
    
    # Read the notebook
    logger.info("Reading pipeline notebook")
    with open(pipeline_path, 'r', encoding='utf-8') as f:
        notebook = nbf.read(f, as_version=4)
    
    # Update the mode selection cell to include ALL YEARS mode
    for i, cell in enumerate(notebook.cells):
        if cell.cell_type == 'code' and 'mode_dropdown = widgets.Dropdown(' in cell.source:
            logger.info("Updating mode selection cell")
            updated_source = cell.source.replace(
                "options=['day', 'week', 'year'],",
                "options=['day', 'week', 'year', 'all_years'],"
            )
            notebook.cells[i].source = updated_source
    
    # Update the process_data function to use the efficient storage system
    for i, cell in enumerate(notebook.cells):
        if cell.cell_type == 'code' and 'def process_data(raw_data, data_processor, logger):' in cell.source:
            logger.info("Updating process_data function")
            
            # Define the new process_data function
            new_process_data = """# Process the raw data
def process_data(raw_data, data_processor, logger):
    logger.info("Processing raw data...")
    
    # Get the selected mode and date
    mode, date = get_mode_and_date()
    date_str = date.strftime('%Y-%m-%d')
    
    # Determine date range based on mode
    if mode == 'day':
        start_date = date_str
        end_date = date_str
    elif mode == 'week':
        start_date = date_str
        end_date = (date + timedelta(days=6)).strftime('%Y-%m-%d')
    elif mode == 'year':
        start_date = date_str
        end_date = (date + timedelta(days=364)).strftime('%Y-%m-%d')
    elif mode == 'all_years':
        # For all_years mode, we don't need date range
        logger.info("Loading all available processed data")
        processed_data = data_processor.load_processed_data_by_mode(mode='all_years')
        logger.info(f"Processed data shape: {processed_data.shape}")
        return processed_data
    
    logger.info(f"Processing data from {start_date} to {end_date}")
    
    # For day, week, and year modes, we can either process raw data or load already processed data
    if mode in ['day', 'week', 'year']:
        # Check if we should process raw data or load already processed data
        if config['processing'].get('reprocess_data', False):
            # Process the data with date range
            processed_files = data_processor.process_data(start_date, end_date)
            
            # Load the processed data
            processed_data = pd.DataFrame()
            for date, file_path in processed_files.items():
                logger.info(f"Loading processed data for {date} from {file_path}")
                
                # Read the file based on its extension
                if file_path.endswith('.parquet') or file_path.endswith('.pq'):
                    logger.info(f"Reading parquet file: {file_path}")
                    date_data = pd.read_parquet(file_path)
                elif file_path.endswith('.csv'):
                    logger.info(f"Reading CSV file: {file_path}")
                    date_data = pd.read_csv(file_path)
                elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
                    logger.info(f"Reading Excel file: {file_path}")
                    date_data = pd.read_excel(file_path)
                else:
                    logger.error(f"Unsupported file format: {file_path}")
                    raise ValueError(f"Unsupported file format: {file_path}")
                
                processed_data = pd.concat([processed_data, date_data], ignore_index=True)
        else:
            # Load already processed data using the efficient storage system
            processed_data = data_processor.load_processed_data_by_mode(
                mode=mode,
                date=date_str if mode == 'day' else None,
                start_date=start_date if mode in ['week', 'year'] else None,
                end_date=end_date if mode in ['week', 'year'] else None
            )
    
    logger.info(f"Processed data shape: {processed_data.shape}")
    return processed_data

# Execute data processing
processed_data = process_data(raw_data, data_processor, logger)

# Display a sample of the processed data
display(processed_data.head())
print(f"Processed data shape: {processed_data.shape}")"""
            
            notebook.cells[i].source = new_process_data
    
    # Write the updated notebook
    logger.info("Writing updated pipeline notebook")
    with open(pipeline_path, 'w', encoding='utf-8') as f:
        nbf.write(notebook, f)
    
    logger.info("Pipeline notebook has been updated to use the EfficientDataStorage system")

if __name__ == "__main__":
    main() 
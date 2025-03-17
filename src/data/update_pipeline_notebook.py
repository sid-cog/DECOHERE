#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to update the pipeline notebook to use all three data sources:
1. Financials data from /home/siddharth_johri/projects/data/financials/
2. Sector mapping data from /home/siddharth_johri/projects/data/sector/
3. Returns data from /home/siddharth_johri/projects/data/returns/
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
logger = logging.getLogger('update_pipeline_notebook')

def main():
    """
    Main function to update the pipeline notebook.
    """
    logger.info("Starting to update the pipeline notebook...")
    
    # Define paths
    pipeline_path = "notebooks/pipeline.ipynb"
    backup_path = f"{pipeline_path}.bak_data_sources"
    
    # Create backup
    logger.info(f"Creating backup at {backup_path}")
    shutil.copy2(pipeline_path, backup_path)
    
    # Read the notebook
    logger.info("Reading pipeline notebook")
    with open(pipeline_path, 'r', encoding='utf-8') as f:
        notebook = nbf.read(f, as_version=4)
    
    # Update the data loading section to include all three data sources
    for i, cell in enumerate(notebook.cells):
        if cell.cell_type == 'code' and 'def load_data():' in cell.source:
            logger.info("Updating data loading function")
            
            # Define the new load_data function
            new_load_data = """# Load the raw data
def load_data():
    # Get the selected mode and date
    mode, date = get_mode_and_date()
    date_str = date.strftime('%Y-%m-%d')
    
    # Load the configuration for the selected mode
    config = load_mode_config(mode)
    
    # Setup logging
    logger = setup_logging(config)
    logger.info(f"Running pipeline in {mode} mode for date {date_str}")
    
    # Initialize the data processor
    data_processor = DataProcessor(config, logger)
    
    # Load all data sources
    logger.info("Loading all data sources...")
    data_sources = data_processor.load_all_data_sources(date=date_str)
    
    # Extract the financial data as raw_data for backward compatibility
    raw_data = data_sources['financials']
    logger.info(f"Loaded {len(raw_data)} rows of financial data")
    
    # Log information about other data sources
    logger.info(f"Loaded sector mapping data with shape: {data_sources['sector_mapping'].shape}")
    logger.info(f"Loaded price returns data with shape: {data_sources['price_returns'].shape}")
    logger.info(f"Loaded total returns data with shape: {data_sources['total_returns'].shape}")
    
    return raw_data, data_processor, config, logger, data_sources

# Execute data loading
raw_data, data_processor, config, logger, data_sources = load_data()

# Display a sample of the raw data
display(raw_data.head())
print(f"Raw data shape: {raw_data.shape}")

# Display samples of other data sources
print("\\nSector Mapping Data Sample:")
display(data_sources['sector_mapping'].head(3))

print("\\nPrice Returns Data Sample:")
display(data_sources['price_returns'].head(3))

print("\\nTotal Returns Data Sample:")
display(data_sources['total_returns'].head(3))"""
            
            notebook.cells[i].source = new_load_data
    
    # Update the feature generation section to use sector data
    for i, cell in enumerate(notebook.cells):
        if cell.cell_type == 'code' and 'def generate_features(' in cell.source:
            logger.info("Updating feature generation function")
            
            # Define the new generate_features function
            new_generate_features = """# Generate features
def generate_features(processed_data, config, logger, data_sources):
    logger.info("Generating features...")
    feature_generator = FeatureGenerator(config, logger)
    
    # Pass all data sources to the feature generator
    features = feature_generator.generate_features(
        processed_data, 
        sector_data=data_sources['sector_mapping'],
        price_returns=data_sources['price_returns'],
        total_returns=data_sources['total_returns']
    )
    
    logger.info(f"Generated features shape: {features.shape}")
    return features, feature_generator

# Execute feature generation
features, feature_generator = generate_features(processed_data, config, logger, data_sources)

# Display a sample of the generated features
display(features.head())
print(f"Features shape: {features.shape}")"""
            
            notebook.cells[i].source = new_generate_features
    
    # Write the updated notebook
    logger.info("Writing updated pipeline notebook")
    with open(pipeline_path, 'w', encoding='utf-8') as f:
        nbf.write(notebook, f)
    
    logger.info("Pipeline notebook has been updated to use all three data sources")

if __name__ == "__main__":
    main() 
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example script demonstrating best practices for processing financial data
following the DECOHERE Terminal Command Guidelines.
"""

import os
import pandas as pd
import numpy as np
import sys
import traceback
import logging
import argparse
from pathlib import Path
import time

# Ensure logs directory exists
os.makedirs('logs', exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/process_financial_data.log')
    ]
)
logger = logging.getLogger('process_financial_data')

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Process financial data with best practices.')
    parser.add_argument('--input', type=str, required=True, 
                        help='Path to input parquet file')
    parser.add_argument('--output', type=str, default='data/processed/output.csv',
                        help='Path to output CSV file')
    parser.add_argument('--ticker', type=str, 
                        help='Filter data for a specific ticker')
    parser.add_argument('--chunk-size', type=int, default=10000,
                        help='Chunk size for processing large files')
    return parser.parse_args()

def process_chunk(chunk, ticker=None):
    """
    Process a chunk of financial data.
    
    Parameters:
    -----------
    chunk : pandas.DataFrame
        Chunk of data to process
    ticker : str, optional
        Filter data for a specific ticker
        
    Returns:
    --------
    pandas.DataFrame
        Processed data
    """
    # Filter by ticker if specified
    if ticker:
        chunk = chunk[chunk['ID'] == ticker].copy()
        if chunk.empty:
            return chunk
    
    # Example processing steps
    # 1. Normalize column names
    chunk.columns = [col.upper() for col in chunk.columns]
    
    # 2. Calculate financial metrics (example)
    if 'NET_INCOME' in chunk.columns and 'SALES' in chunk.columns:
        # Use .loc to avoid SettingWithCopyWarning
        mask = chunk['SALES'] != 0
        chunk.loc[mask, 'PROFIT_MARGIN'] = chunk.loc[mask, 'NET_INCOME'] / chunk.loc[mask, 'SALES']
    
    # 3. Apply transformations
    numeric_cols = chunk.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col in chunk.columns and not chunk[col].isna().all():
            # Use .loc to avoid SettingWithCopyWarning
            chunk.loc[:, f"{col}_SIGNED_LOG"] = np.sign(chunk[col]) * np.log1p(np.abs(chunk[col]))
    
    return chunk

def main():
    """
    Main function to process financial data following best practices.
    """
    start_time = time.time()
    logger.info("Starting financial data processing...")
    
    try:
        # Parse arguments
        args = parse_arguments()
        logger.info(f"Processing file: {args.input}")
        
        # Check if input file exists
        if not os.path.exists(args.input):
            logger.error(f"Input file does not exist: {args.input}")
            return 1
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Process data in chunks
        chunk_count = 0
        total_rows = 0
        processed_chunks = []
        
        logger.info(f"Processing data in chunks of {args.chunk_size} rows...")
        
        # Read and process data in chunks
        for chunk in pd.read_parquet(args.input, engine='pyarrow', chunksize=args.chunk_size):
            chunk_count += 1
            logger.info(f"Processing chunk {chunk_count} with {len(chunk)} rows...")
            
            processed_chunk = process_chunk(chunk, args.ticker)
            total_rows += len(processed_chunk)
            
            if not processed_chunk.empty:
                processed_chunks.append(processed_chunk)
            
            # Log memory usage periodically
            if chunk_count % 5 == 0:
                logger.info(f"Memory usage: {processed_chunk.memory_usage(deep=True).sum() / 1e6:.2f} MB")
        
        # Combine processed chunks
        if processed_chunks:
            result_df = pd.concat(processed_chunks, ignore_index=True)
            logger.info(f"Processed {total_rows} rows across {chunk_count} chunks")
            
            # Print summary statistics instead of full DataFrame
            logger.info(f"Result shape: {result_df.shape}")
            logger.info(f"Columns: {', '.join(result_df.columns[:10])}...")
            
            # Save to file instead of printing to console
            result_df.to_csv(args.output, index=False)
            logger.info(f"Results saved to {args.output}")
            
            # Print only a small sample to console
            if not result_df.empty:
                logger.info(f"Sample data (first 3 rows):")
                sample_str = result_df.head(3).to_string()
                for line in sample_str.split('\n'):
                    logger.info(line)
        else:
            logger.warning("No data processed. Result is empty.")
        
        elapsed_time = time.time() - start_time
        logger.info(f"Processing completed in {elapsed_time:.2f} seconds")
        
    except KeyboardInterrupt:
        logger.warning("Processing interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Error processing data: {e}")
        logger.error("Traceback:")
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 
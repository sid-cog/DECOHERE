#!/usr/bin/env python
import argparse
import logging
from datetime import datetime, timedelta
import multiprocessing as mp
from functools import partial
import pandas as pd
from tqdm import tqdm
import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

from src.inference.daily_residual_generator import DailyResidualGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def process_date(date: str, generator: DailyResidualGenerator = None) -> dict:
    """Process a single date with error handling."""
    if generator is None:
        generator = DailyResidualGenerator()
    
    try:
        generator.generate_residuals(date)
        return {'date': date, 'status': 'success', 'error': None}
    except Exception as e:
        return {'date': date, 'status': 'failed', 'error': str(e)}

def generate_date_range(start_date: str, end_date: str) -> list:
    """Generate list of dates between start and end dates."""
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    
    dates = []
    current_dt = start_dt
    while current_dt <= end_dt:
        dates.append(current_dt.strftime("%Y-%m-%d"))
        current_dt += timedelta(days=1)
    return dates

def parallel_generate_residuals(start_date: str, end_date: str, n_processes: int = None) -> pd.DataFrame:
    """Generate residuals in parallel for date range."""
    if n_processes is None:
        n_processes = max(1, mp.cpu_count() - 2)  # Leave 2 CPUs free
    
    dates = generate_date_range(start_date, end_date)
    logger.info(f"Processing {len(dates)} dates using {n_processes} processes")
    
    # Initialize a generator for each process to avoid recreation
    generators = [DailyResidualGenerator() for _ in range(n_processes)]
    
    # Create pool with initialized generators
    with mp.Pool(n_processes) as pool:
        # Use partial to pass generator to each process
        process_with_generator = [partial(process_date, generator=gen) for gen in generators]
        
        # Process dates in chunks
        chunk_size = len(dates) // len(process_with_generator)
        date_chunks = [dates[i:i + chunk_size] for i in range(0, len(dates), chunk_size)]
        
        # Map chunks to processes
        results = []
        for i, (chunk, processor) in enumerate(zip(date_chunks, process_with_generator)):
            chunk_results = list(tqdm(
                pool.imap(processor, chunk),
                total=len(chunk),
                desc=f"Process {i+1}/{n_processes}"
            ))
            results.extend(chunk_results)
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(results)
    
    # Log summary statistics
    success_count = len(summary_df[summary_df['status'] == 'success'])
    total_count = len(summary_df)
    success_rate = (success_count / total_count) * 100
    
    logger.info(f"\nProcessing Summary:")
    logger.info(f"Total dates processed: {total_count}")
    logger.info(f"Successful: {success_count}")
    logger.info(f"Failed: {total_count - success_count}")
    logger.info(f"Success rate: {success_rate:.2f}%")
    
    return summary_df

def main():
    parser = argparse.ArgumentParser(description='Generate residuals for date range in parallel')
    parser.add_argument('--start-date', required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--processes', type=int, help='Number of processes to use')
    parser.add_argument('--output', help='Output file for summary CSV')
    
    args = parser.parse_args()
    
    summary_df = parallel_generate_residuals(
        args.start_date,
        args.end_date,
        n_processes=args.processes
    )
    
    if args.output:
        summary_df.to_csv(args.output, index=False)
        logger.info(f"Summary saved to {args.output}")

if __name__ == "__main__":
    main() 
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Demonstration script for the efficient data storage system.
This script demonstrates:
1. Loading data in different modes (day, week, year, ALL YEARS)
2. Performance comparison with legacy storage
3. Memory usage comparison
4. Migration of legacy data to the efficient storage format
"""

import os
import sys
import logging
import yaml
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import psutil
import gc

# Add the project root to the Python path
project_root = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.insert(0, project_root)

from src.data.data_processor import DataProcessor
from src.data.efficient_data_storage import EfficientDataStorage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger('demo_efficient_storage')

def load_config():
    """
    Load the configuration file.
    
    Returns:
        Configuration dictionary
    """
    config_path = os.path.join(project_root, 'config', 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def measure_memory_usage(func, *args, **kwargs):
    """
    Measure the memory usage of a function.
    
    Args:
        func: Function to measure
        *args: Arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function
        
    Returns:
        Tuple of (result, memory_usage_mb)
    """
    # Clear memory before measurement
    gc.collect()
    
    # Get initial memory usage
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Run the function and measure time
    start_time = time.time()
    result = func(*args, **kwargs)
    elapsed_time = time.time() - start_time
    
    # Get final memory usage
    gc.collect()
    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Calculate memory usage
    memory_usage = final_memory - initial_memory
    
    return result, memory_usage, elapsed_time

def demo_data_loading(data_processor):
    """
    Demonstrate loading data in different modes.
    
    Args:
        data_processor: DataProcessor instance
    """
    logger.info("Demonstrating data loading in different modes...")
    
    # Get available dates
    available_dates = data_processor.get_available_dates()
    
    if not available_dates:
        logger.warning("No processed data available. Please process some data first.")
        return
    
    # Sort dates
    available_dates.sort()
    
    # Select a date for demonstration
    demo_date = available_dates[0]
    logger.info(f"Using date {demo_date} for demonstration")
    
    # Demonstrate day mode
    logger.info("Loading data in DAY mode...")
    day_data, day_memory, day_time = measure_memory_usage(
        data_processor.load_processed_data_by_mode,
        mode='day',
        date=demo_date
    )
    logger.info(f"DAY mode: Loaded {len(day_data)} rows in {day_time:.2f} seconds, using {day_memory:.2f} MB of memory")
    
    # Demonstrate week mode
    if len(available_dates) >= 7:
        logger.info("Loading data in WEEK mode...")
        end_date = available_dates[min(6, len(available_dates)-1)]
        week_data, week_memory, week_time = measure_memory_usage(
            data_processor.load_processed_data_by_mode,
            mode='week',
            start_date=demo_date,
            end_date=end_date
        )
        logger.info(f"WEEK mode: Loaded {len(week_data)} rows in {week_time:.2f} seconds, using {week_memory:.2f} MB of memory")
    
    # Demonstrate year mode
    if len(available_dates) >= 30:
        logger.info("Loading data in YEAR mode...")
        end_date = available_dates[min(29, len(available_dates)-1)]
        year_data, year_memory, year_time = measure_memory_usage(
            data_processor.load_processed_data_by_mode,
            mode='year',
            start_date=demo_date,
            end_date=end_date
        )
        logger.info(f"YEAR mode: Loaded {len(year_data)} rows in {year_time:.2f} seconds, using {year_memory:.2f} MB of memory")
    
    # Demonstrate ALL YEARS mode
    logger.info("Loading data in ALL YEARS mode...")
    all_years_data, all_years_memory, all_years_time = measure_memory_usage(
        data_processor.load_processed_data_by_mode,
        mode='all_years'
    )
    logger.info(f"ALL YEARS mode: Loaded {len(all_years_data)} rows in {all_years_time:.2f} seconds, using {all_years_memory:.2f} MB of memory")

def demo_migration(data_processor):
    """
    Demonstrate migration of legacy data to the efficient storage format.
    
    Args:
        data_processor: DataProcessor instance
    """
    logger.info("Demonstrating migration of legacy data to the efficient storage format...")
    
    # Migrate legacy data
    success = data_processor.migrate_to_efficient_storage()
    
    if success:
        logger.info("Migration completed successfully")
    else:
        logger.warning("Migration completed with some errors")

def demo_performance_comparison(data_processor, config):
    """
    Demonstrate performance comparison between legacy and efficient storage.
    
    Args:
        data_processor: DataProcessor instance
        config: Configuration dictionary
    """
    logger.info("Demonstrating performance comparison...")
    
    # Get available dates
    available_dates = data_processor.get_available_dates()
    
    if not available_dates:
        logger.warning("No processed data available. Please process some data first.")
        return
    
    # Sort dates
    available_dates.sort()
    
    # Select a date for demonstration
    demo_date = available_dates[0]
    logger.info(f"Using date {demo_date} for performance comparison")
    
    # Legacy storage (reading individual files)
    logger.info("Testing legacy storage performance...")
    
    def load_legacy():
        """Load data using legacy method"""
        processed_file = os.path.join(config['data']['processed_data'], f"processed_{demo_date}.pq")
        return pd.read_parquet(processed_file)
    
    legacy_data, legacy_memory, legacy_time = measure_memory_usage(load_legacy)
    logger.info(f"Legacy storage: Loaded {len(legacy_data)} rows in {legacy_time:.2f} seconds, using {legacy_memory:.2f} MB of memory")
    
    # Efficient storage
    logger.info("Testing efficient storage performance...")
    efficient_data, efficient_memory, efficient_time = measure_memory_usage(
        data_processor.load_processed_data_by_mode,
        mode='day',
        date=demo_date
    )
    logger.info(f"Efficient storage: Loaded {len(efficient_data)} rows in {efficient_time:.2f} seconds, using {efficient_memory:.2f} MB of memory")
    
    # Performance improvement
    time_improvement = (legacy_time - efficient_time) / legacy_time * 100
    memory_improvement = (legacy_memory - efficient_memory) / legacy_memory * 100
    
    logger.info(f"Performance improvement: {time_improvement:.2f}% faster, {memory_improvement:.2f}% less memory")

def main():
    """
    Main function to run the demonstration.
    """
    logger.info("Starting demonstration of the efficient data storage system...")
    
    # Load configuration
    config = load_config()
    
    # Initialize data processor
    data_processor = DataProcessor(config, logger)
    
    # Demonstrate migration
    demo_migration(data_processor)
    
    # Demonstrate data loading
    demo_data_loading(data_processor)
    
    # Demonstrate performance comparison
    demo_performance_comparison(data_processor, config)
    
    logger.info("Demonstration completed successfully")

if __name__ == "__main__":
    main() 
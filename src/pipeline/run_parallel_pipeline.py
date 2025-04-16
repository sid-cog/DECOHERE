#!/usr/bin/env python3
"""
Parallel Pipeline Runner
This script runs the data processing pipeline in parallel for a range of dates.
Usage: python run_parallel_pipeline.py --start-date YYYY-MM-DD --end-date YYYY-MM-DD [--num-workers N]
"""

import os
import sys
import time
import logging
import argparse
import pandas as pd
import psutil
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from typing import List, Dict, Tuple
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.data_processor import DataProcessor
from src.data.efficient_data_storage import EfficientDataStorage, DataType, DataStage
from src.data.feature_generator import FeatureGenerator
import yaml

def get_system_resources():
    """Get available system resources."""
    # Use logical cores (vCPUs) instead of physical cores
    cpu_cores = psutil.cpu_count(logical=True)
    available_memory = psutil.virtual_memory().available / (1024**3)  # Convert to GB
    return cpu_cores, available_memory

def estimate_memory_per_worker() -> float:
    """
    Estimate memory required per worker in GB.
    This is based on typical memory usage patterns in the pipeline.
    """
    # Base memory for Python process
    base_memory = 0.5  # GB
    
    # Memory for data processing
    data_memory = 2.0  # GB
    
    # Memory for feature generation
    feature_memory = 1.5  # GB
    
    return base_memory + data_memory + feature_memory

def determine_optimal_workers() -> int:
    """
    Determine optimal number of workers based on system resources.
    Returns:
        Optimal number of workers
    """
    cpu_cores, available_memory_gb = get_system_resources()
    memory_per_worker = estimate_memory_per_worker()
    
    # Calculate maximum workers based on memory
    max_memory_workers = int(available_memory_gb / memory_per_worker)
    
    # Calculate maximum workers based on CPU
    max_cpu_workers = cpu_cores
    
    # Use the smaller of the two, but ensure at least 1 worker
    optimal_workers = min(max_memory_workers, max_cpu_workers)
    optimal_workers = max(1, optimal_workers)
    
    return optimal_workers

def setup_logging():
    """Configure logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('pipeline.log')
        ]
    )
    return logging.getLogger('Pipeline')

def generate_business_days(start_date_str: str, end_date_str: str) -> List[str]:
    """Generate list of business days between start and end dates"""
    start_date = pd.to_datetime(start_date_str)
    end_date = pd.to_datetime(end_date_str)
    return [d.strftime('%Y-%m-%d') for d in pd.bdate_range(start=start_date, end=end_date)]

def process_single_date(date_str: str, config: Dict) -> Dict:
    """Process a single date and return results"""
    start_time = time.time()
    logger = logging.getLogger(f'DateProcessor_{date_str}')
    
    try:
        # Initialize components
        processor = DataProcessor(config, winsorize=True)
        storage = EfficientDataStorage(config, logger)
        feature_generator = FeatureGenerator(config, logger)
        
        # Step 1: Load raw data
        raw_data = processor.load_raw_data(date_str)
        if raw_data.empty:
            return {
                'date': date_str,
                'success': True,
                'is_missing': True,
                'paths': [None, None, None],  # processed, pre-feature, enhanced
                'duration': time.time() - start_time
            }
        
        # Step 2: Transform data
        transformed_data = processor.transform_raw_data(raw_data)
        
        # Step 3: Store processed data
        processed_path = storage.store_data(
            transformed_data,
            DataType.FUNDAMENTALS,
            DataStage.PROCESSED,
            date_str
        )
        
        # Step 4: Generate pre-feature set
        pre_feature_df = storage.processed_data_feat_gen(transformed_data)
        pre_feature_path = None
        if not pre_feature_df.empty:
            pre_feature_path = storage.store_data(
                pre_feature_df,
                DataType.FUNDAMENTALS,
                DataStage.FEATURES,
                date_str,
                sub_type='pre_feature_set'
            )
            
            # Step 5: Generate enhanced features
            feature_cfg = config.get('features', {})
            # Get sector mapping path from data section of config
            sector_mapping_path = config.get('data', {}).get('sector_mapping')
            if not sector_mapping_path:
                logger.warning("No sector mapping path found in config. Sector features will be skipped.")
            else:
                # Verify the sector mapping file exists
                if not os.path.exists(sector_mapping_path):
                    logger.error(f"Sector mapping file not found at: {sector_mapping_path}")
                    sector_mapping_path = None
                else:
                    logger.info(f"Using sector mapping file: {sector_mapping_path}")
                    # Verify sector columns exist in the mapping file
                    try:
                        sector_df = pd.read_parquet(sector_mapping_path)
                        available_sectors = [col for col in ['sector_1', 'sector_2'] if col in sector_df.columns]
                        if not available_sectors:
                            logger.error("Neither sector_1 nor sector_2 found in sector mapping file")
                            sector_mapping_path = None
                        else:
                            logger.info(f"Found sector columns in mapping file: {available_sectors}")
                    except Exception as e:
                        logger.error(f"Error reading sector mapping file: {e}")
                        sector_mapping_path = None
            
            enhanced_feature_df = feature_generator.generate_enhanced_features(
                df=pre_feature_df,
                hist_window=feature_cfg.get('hist_window', 6),
                fwd_window=feature_cfg.get('fwd_window', 6),
                target_metric=feature_cfg.get('target_metric', 'PE_RATIO_RATIO_SIGNED_LOG'),
                sector_mapping_path=sector_mapping_path,  # Use the correct path from config
                sector_levels_to_include=['sector_1', 'sector_2'],  # Include both sector levels
                include_sector_features=True  # Explicitly enable sector features
            )
            
            # Log the columns in the enhanced features DataFrame
            if not enhanced_feature_df.empty:
                logger.info(f"Enhanced features columns: {enhanced_feature_df.columns.tolist()}")
                logger.info(f"Enhanced features shape: {enhanced_feature_df.shape}")
                # Check if sector columns were included
                sector_cols = [col for col in ['sector_1', 'sector_2'] if col in enhanced_feature_df.columns]
                if sector_cols:
                    logger.info(f"Included sector columns: {sector_cols}")
                else:
                    logger.warning("No sector columns found in enhanced features")
            
            enhanced_feature_path = None
            if not enhanced_feature_df.empty:
                enhanced_feature_path = storage.store_data(
                    enhanced_feature_df,
                    DataType.FUNDAMENTALS,
                    DataStage.FEATURES,
                    date_str,
                    sub_type='enhanced_features'
                )
        
        return {
            'date': date_str,
            'success': True,
            'is_missing': False,
            'paths': [processed_path, pre_feature_path, enhanced_feature_path],
            'duration': time.time() - start_time
        }
        
    except Exception as e:
        logger.error(f"Error processing {date_str}: {str(e)}")
        return {
            'date': date_str,
            'success': False,
            'error': str(e),
            'duration': time.time() - start_time
        }

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run parallel data processing pipeline')
    parser.add_argument('--start-date', required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--num-workers', type=int, help='Number of parallel workers (optional)')
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    
    # Determine optimal number of workers
    if args.num_workers is None:
        optimal_workers = determine_optimal_workers()
        cpu_cores, available_memory_gb = get_system_resources()
        memory_per_worker = estimate_memory_per_worker()
        
        logger.info(f"System Resources:")
        logger.info(f"  CPU Cores: {cpu_cores}")
        logger.info(f"  Available Memory: {available_memory_gb:.2f} GB")
        logger.info(f"  Estimated Memory per Worker: {memory_per_worker:.2f} GB")
        logger.info(f"  Optimal Workers: {optimal_workers}")
        
        num_workers = optimal_workers
    else:
        num_workers = args.num_workers
        logger.info(f"Using specified number of workers: {num_workers}")
    
    # Load configuration
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                              'config', 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Generate business days
    dates = generate_business_days(args.start_date, args.end_date)
    logger.info(f"Processing {len(dates)} dates from {args.start_date} to {args.end_date}")
    logger.info(f"Using {num_workers} parallel workers")
    
    # Process dates in parallel
    start_time = time.time()
    results = []
    
    # Create a pool of workers
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        futures = [executor.submit(process_single_date, date, config) for date in dates]
        
        # Process results as they complete
        for future in tqdm(futures, total=len(dates), desc="Processing dates"):
            try:
                result = future.result()
                results.append(result)
                
                # Log individual date processing time
                if result['success'] and not result['is_missing']:
                    logger.info(f"Processed {result['date']} in {result['duration']:.2f} seconds")
            except Exception as e:
                logger.error(f"Error in future: {str(e)}")
                results.append({
                    'date': 'unknown',
                    'success': False,
                    'error': str(e),
                    'duration': 0
                })
    
    # Generate summary
    total_time = time.time() - start_time
    successful = sum(1 for r in results if r['success'] and not r['is_missing'])
    failed = len(results) - successful
    avg_time = total_time / len(dates)
    
    # Calculate parallelization metrics
    total_processing_time = sum(r['duration'] for r in results if r['success'] and not r['is_missing'])
    theoretical_min_time = total_processing_time / num_workers
    actual_speedup = total_processing_time / total_time
    parallel_efficiency = actual_speedup / num_workers * 100
    
    logger.info("\nPipeline Summary:")
    logger.info(f"Total dates processed: {len(dates)}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Total wall time: {total_time:.2f} seconds")
    logger.info(f"Total processing time: {total_processing_time:.2f} seconds")
    logger.info(f"Average time per date: {avg_time:.2f} seconds")
    logger.info(f"Parallelization Metrics:")
    logger.info(f"  Theoretical minimum time: {theoretical_min_time:.2f} seconds")
    logger.info(f"  Actual speedup: {actual_speedup:.2f}x")
    logger.info(f"  Parallel efficiency: {parallel_efficiency:.1f}%")
    
    # Save detailed results
    results_df = pd.DataFrame(results)
    results_df.to_csv(f'pipeline_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv', index=False)
    
    if failed > 0:
        logger.warning(f"Failed dates: {[r['date'] for r in results if not r['success']]}")
        sys.exit(1)

if __name__ == "__main__":
    main() 
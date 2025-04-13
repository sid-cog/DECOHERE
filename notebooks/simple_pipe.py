# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: venv
#     language: python
#     name: python3
# ---

# %%
#Imports
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

# Add project root to Python path
project_root = str(Path.cwd().parent)
if project_root not in sys.path:
    sys.path.append(project_root)

# Import project modules
from src.data.efficient_data_storage import EfficientDataStorage, DataType, DataStage
from src.data.data_processor import DataProcessor
from src.utils.logging_config import setup_logging
from src.utils.config_loader import load_config

from src.data.feature_generator import FeatureGenerator


# %%
# Cell 2: Setup
def setup_pipeline():
    """Initialize pipeline configuration and logging."""
    # Load configuration with absolute path
    config_path = '/home/siddharth.johri/DECOHERE/config/config.yaml'
    config = load_config(config_path)
    
    # Setup logging
    logger = setup_logging(config)
    logger.info("Pipeline initialization started")
    
    # Initialize storage system
    storage = EfficientDataStorage(config, logger)
    
    # Initialize data processor
    processor = DataProcessor(config, logger)
    
    # Initialize feature generator
    feature_generator = FeatureGenerator(config, logger)
    
    logger.info("Pipeline initialization completed")
    return config, logger, storage, processor, feature_generator

# Initialize pipeline
config, logger, storage, processor, feature_generator = setup_pipeline()

# %%
# Initialize data processor
from typing import Tuple
data_processor = DataProcessor(config, logger)

def load_and_process_data(date: str, mode: str = 'day') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and process data for a specific date and mode.
    
    Args:
        date: Date to process (YYYY-MM-DD)
        mode: Mode to use ('day', 'week', 'year')
        
    Returns:
        Tuple of (processed_df, feature_ready_df)
    """
    try:
        # Load raw data
        raw_df = data_processor.load_raw_data(date)
        
        if raw_df.empty:
            logger.warning(f"No data found for date: {date}")
            return pd.DataFrame(), pd.DataFrame()
            
        # Transform raw data
        transformed_df = data_processor.transform_raw_data(raw_df)
        
        # Fill missing values
        filled_df = data_processor.fill_missing_values(transformed_df)
        
        # Generate feature-ready DataFrame
        feature_ready_df = data_processor.processed_data_feat_gen(filled_df)
        
        # Save processed data
        processed_path = data_processor.save_processed_data(filled_df, date)
        logger.info(f"Saved processed data to: {processed_path}")
        
        # Save feature-ready data
        feature_path = data_processor.save_pre_feature_set(feature_ready_df)
        logger.info(f"Saved feature-ready data to: {feature_path}")
        
        return filled_df, feature_ready_df
        
    except Exception as e:
        logger.error(f"Error processing data for date {date}: {str(e)}")
        return pd.DataFrame(), pd.DataFrame()


# %%
# def load_and_process_data(date_str: str):
#     """Load and process data for a specific date."""
#     try:
#         # Load raw data
#         raw_data = processor.load_raw_data(date_str)
#         if raw_data.empty:
#             logger.warning(f"No raw data found for date {date_str}")
#             return pd.DataFrame(), pd.DataFrame()
        
#         # Transform data
#         transformed_data = processor.transform_data(raw_data)
        
#         # Fill missing values
#         filled_data = processor.fill_missing_values(transformed_data)
        
#         # Generate features
#         feature_df = feature_generator.generate_enhanced_features(
#             filled_data,
#             hist_window=6,
#             fwd_window=6,
#             target_metric='PE_RATIO_RATIO_SIGNED_LOG'
#         )
        
#         if not feature_df.empty:
#             # Save processed data
#             storage.save_processed_data(feature_df, date_str)
#             logger.info(f"Saved processed data for date {date_str}")
            
#             # Save feature-ready data
#             storage.save_feature_ready_data(feature_df, date_str)
#             logger.info(f"Saved feature-ready data for date {date_str}")
            
#             return feature_df, filled_data
#         else:
#             logger.warning(f"No features generated for date {date_str}")
#             return pd.DataFrame(), pd.DataFrame()
            
#     except Exception as e:
#         logger.error(f"Error processing data for date {date_str}: {str(e)}")
#         return pd.DataFrame(), pd.DataFrame()

# %%
def generate_features(processed_df: pd.DataFrame, mode: str = 'day') -> pd.DataFrame:
    """
    Generate features from processed data.
    
    Args:
        processed_df: DataFrame containing processed data
        mode: Mode to use ('day', 'week', 'year')
        
    Returns:
        DataFrame containing generated features
    """
    try:
        # Generate feature set from processed data
        feature_df = data_processor.processed_data_feat_gen(processed_df)
        
        if feature_df.empty:
            logger.warning("No features generated")
            return pd.DataFrame()
            
        # Save pre-feature set
        pre_feature_path = data_processor.save_pre_feature_set(feature_df)
        logger.info(f"Saved pre-feature set to: {pre_feature_path}")
        
        return feature_df
        
    except Exception as e:
        logger.error(f"Error generating features: {str(e)}")
        return pd.DataFrame()


# %%
def load_and_process_data(date_str: str):
    """Load and process data for a specific date."""
    try:
        print(f"\nStarting data processing for {date_str}")
        
        # Load raw data
        print("Loading raw data...")
        raw_data = processor.load_raw_data(date_str)
        if raw_data.empty:
            logger.warning(f"No raw data found for date {date_str}")
            return pd.DataFrame(), pd.DataFrame()
        print(f"Raw data shape: {raw_data.shape}")
        
        # Transform data
        print("Transforming data...")
        transformed_data = processor.transform_raw_data(raw_data)  # Changed from transform_data to transform_raw_data
        print(f"Transformed data shape: {transformed_data.shape}")
        
        # Fill missing values
        print("Filling missing values...")
        filled_data = processor.fill_missing_values(transformed_data)
        print(f"Filled data shape: {filled_data.shape}")
        
        # Generate features
        print("Generating features...")
        feature_df = feature_generator.generate_enhanced_features(
            filled_data,
            hist_window=6,
            fwd_window=6,
            target_metric='PE_RATIO_RATIO_SIGNED_LOG'
        )
        print(f"Feature data shape: {feature_df.shape}")
        
        if not feature_df.empty:
            # Save processed data
            print(f"Saving processed data for {date_str}...")
            storage.store_data(
                df=feature_df,
                data_type=DataType.FUNDAMENTALS,
                stage=DataStage.PROCESSED,
                date=date_str
            )
            
            # Save feature-ready data
            print(f"Saving feature-ready data for {date_str}...")
            storage.store_data(
                df=feature_df,
                data_type=DataType.FUNDAMENTALS,
                stage=DataStage.FEATURES,
                date=date_str,
                sub_type='pre_feature_set'
            )
            
            return feature_df, filled_data
        else:
            logger.warning(f"No features generated for date {date_str}")
            return pd.DataFrame(), pd.DataFrame()
            
    except Exception as e:
        logger.error(f"Error processing data for date {date_str}: {str(e)}")
        print(f"Error details: {str(e)}")
        return pd.DataFrame(), pd.DataFrame()


# %%
def load_and_process_data(date_str: str):
    """Load and process data for a specific date."""
    try:
        print(f"\nStarting data processing for {date_str}")
        
        # Load raw data
        print("Loading raw data...")
        raw_data = processor.load_raw_data(date_str)
        if raw_data.empty:
            logger.warning(f"No raw data found for date {date_str}")
            return pd.DataFrame()
        print(f"Raw data shape: {raw_data.shape}")
        
        # Transform data
        print("Transforming data...")
        transformed_data = processor.transform_raw_data(raw_data)
        print(f"Transformed data shape: {transformed_data.shape}")
        
        # Fill missing values
        print("Filling missing values...")
        filled_data = processor.fill_missing_values(transformed_data)
        print(f"Filled data shape: {filled_data.shape}")
        
        # Print available columns for debugging
        print("\nAvailable columns in processed data:")
        for col in filled_data.columns:
            if any(pattern in col.lower() for pattern in ['signed_log', 'ratio', 'coeff_of_var']):
                print(f"- {col}")
        
        # Save processed data
        print(f"\nSaving processed data for {date_str}...")
        storage.store_data(
            df=filled_data,
            data_type=DataType.FUNDAMENTALS,
            stage=DataStage.PROCESSED,
            date=date_str
        )
        
        # Generate and save pre-feature set data
        print(f"Generating and saving pre-feature set data for {date_str}...")
        pre_feature_data = storage.processed_data_feat_gen(filled_data)
        if not pre_feature_data.empty:
            storage.store_data(
                df=pre_feature_data,
                data_type=DataType.FUNDAMENTALS,
                stage=DataStage.FEATURES,
                date=date_str,
                sub_type='pre_feature_set'
            )
            print(f"Pre-feature set data shape: {pre_feature_data.shape}")
            print("Pre-feature set columns:")
            for col in pre_feature_data.columns:
                print(f"- {col}")
        else:
            print("Warning: No pre-feature set data generated")
        
        return filled_data
            
    except Exception as e:
        logger.error(f"Error processing data for date {date_str}: {str(e)}")
        print(f"Error details: {str(e)}")
        return pd.DataFrame()

# %%
# def generate_enhanced_features(date_str: str, processed_df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Generate enhanced features for a specific date using the FeatureGenerator.
    
#     Args:
#         date_str: Date string in YYYY-MM-DD format
#         processed_df: DataFrame containing processed data
        
#     Returns:
#         DataFrame containing generated features
#     """
#     try:
#         # Generate features
#         feature_df = feature_generator.generate_enhanced_features(
#             processed_df,
#             hist_window=6,
#             fwd_window=6,
#             target_metric='PE_RATIO_RATIO_SIGNED_LOG'
#         )
        
#         if not feature_df.empty:
#             print(f"\nGenerated features for {date_str}:")
#             print(f"Feature DataFrame shape: {feature_df.shape}")
#             print("\nSample of features:")
#             print(feature_df.head())
#             return feature_df
#         else:
#             print(f"\nNo features generated for {date_str}")
#             return pd.DataFrame()
            
#     except Exception as e:
#         print(f"Error generating features for date {date_str}: {str(e)}")
#         return pd.DataFrame()

# # Example usage:
# # processed_df, _ = load_and_process_data("2024-09-02")
# # feature_df = generate_enhanced_features("2024-09-02", processed_df)

# %%
# # Process multiple dates
# dates = ["2024-09-02", "2024-09-03", "2024-09-04"]

# for date in dates:
#     # Load and process data
#     processed_df, _ = load_and_process_data(date)
    
#     if not processed_df.empty:
#         # Generate features
#         feature_df = generate_enhanced_features(date, processed_df)
        
#         if not feature_df.empty:
#             print(f"\nSuccessfully generated features for {date}")
#         else:
#             print(f"\nFailed to generate features for {date}")
#     else:
#         print(f"\nNo data processed for {date}")

# %%
# Print a special message
print("sid sux 😤")
print("sid sux yesterday 😡")

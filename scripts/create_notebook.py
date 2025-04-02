import nbformat as nbf

# Create a new notebook
nb = nbf.v4.new_notebook()

# Title and Introduction
intro_md = """# DECOHERE Quantitative Trading Pipeline - Updated Version

This notebook implements the updated pipeline structure with efficient data storage and processing.

## Pipeline Components
1. Data Loading and Storage
2. Data Processing
3. Feature Generation
4. Feature Selection
5. Model Training and Evaluation

## Pipeline Modes
- Day Mode: Process single day data
- Week Mode: Process weekly data
- Year Mode: Process yearly data"""

nb.cells.append(nbf.v4.new_markdown_cell(intro_md))

# Imports
imports_code = """import os
import sys
import pandas as pd
import numpy as np
import yaml
import logging
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to Python path
project_root = str(Path.cwd().parent)
if project_root not in sys.path:
    sys.path.append(project_root)

# Import project modules
from src.data.efficient_data_storage import EfficientDataStorage, DataType, DataStage
from src.data.data_processor import DataProcessor
from src.features.feature_generator import FeatureGenerator
from src.features.feature_selector import FeatureSelector
from src.models.model_trainer import ModelTrainer
from src.utils.logging_config import setup_logging
from src.utils.config_loader import load_config"""

nb.cells.append(nbf.v4.new_code_cell(imports_code))

# Setup and Configuration
setup_md = "## 1. Setup and Configuration"
nb.cells.append(nbf.v4.new_markdown_cell(setup_md))

setup_code = """def setup_pipeline():
    \"\"\"Initialize pipeline configuration and logging.\"\"\"
    # Load configuration
    config = load_config()
    
    # Setup logging
    logger = setup_logging(config)
    logger.info("Pipeline initialization started")
    
    # Initialize storage system
    storage = EfficientDataStorage(config, logger)
    
    # Initialize data processor
    processor = DataProcessor(config, logger)
    
    logger.info("Pipeline initialization completed")
    return config, logger, storage, processor

# Initialize pipeline
config, logger, storage, processor = setup_pipeline()"""

nb.cells.append(nbf.v4.new_code_cell(setup_code))

# Data Loading and Processing
data_md = "## 2. Data Loading and Processing"
nb.cells.append(nbf.v4.new_markdown_cell(data_md))

data_code = """def load_and_process_data(date_str: str, mode: str = 'day'):
    \"\"\"Load and process data for a specific date and mode.\"\"\"
    logger.info(f"Loading and processing data for date: {date_str} in {mode} mode")
    
    try:
        # Load fundamentals data
        fundamentals_df = storage.load_data(
            data_type=DataType.FUNDAMENTALS,
            stage=DataStage.RAW,
            date=date_str,
            mode=mode
        )
        
        if fundamentals_df.empty:
            logger.error(f"No fundamentals data found for date: {date_str}")
            return None
        
        # Load returns data
        returns_df = storage.load_data(
            data_type=DataType.RETURNS,
            stage=DataStage.RAW,
            date=date_str,
            mode=mode
        )
        
        # Process fundamentals data
        processed_fundamentals = processor.process_fundamentals(fundamentals_df)
        
        # Store processed data
        storage.store_data(
            df=processed_fundamentals,
            data_type=DataType.FUNDAMENTALS,
            stage=DataStage.PROCESSED,
            date=date_str
        )
        
        return processed_fundamentals, returns_df
        
    except Exception as e:
        logger.error(f"Error in data loading and processing: {e}")
        return None

# Example usage
date_str = datetime.now().strftime('%Y-%m-%d')
processed_data, returns_data = load_and_process_data(date_str, mode='day')

if processed_data is not None:
    print(f"Processed data shape: {processed_data.shape}")
    print(f"Returns data shape: {returns_data.shape if not returns_data.empty else 'No returns data'}")"""

nb.cells.append(nbf.v4.new_code_cell(data_code))

# Feature Generation
features_md = "## 3. Feature Generation"
nb.cells.append(nbf.v4.new_markdown_cell(features_md))

features_code = """def generate_features(processed_data: pd.DataFrame, date_str: str):
    \"\"\"Generate features from processed data.\"\"\"
    logger.info(f"Generating features for date: {date_str}")
    
    try:
        # Initialize feature generator
        feature_gen = FeatureGenerator(config, logger)
        
        # Generate features
        features_df = feature_gen.generate_features(processed_data)
        
        # Store features
        storage.store_data(
            df=features_df,
            data_type=DataType.FUNDAMENTALS,
            stage=DataStage.FEATURES,
            date=date_str
        )
        
        return features_df
        
    except Exception as e:
        logger.error(f"Error in feature generation: {e}")
        return None

# Generate features
if processed_data is not None:
    features_df = generate_features(processed_data, date_str)
    if features_df is not None:
        print(f"Generated features shape: {features_df.shape}")"""

nb.cells.append(nbf.v4.new_code_cell(features_code))

# Feature Selection
selection_md = "## 4. Feature Selection"
nb.cells.append(nbf.v4.new_markdown_cell(selection_md))

selection_code = """def select_features(features_df: pd.DataFrame, returns_df: pd.DataFrame):
    \"\"\"Select the most relevant features.\"\"\"
    logger.info("Starting feature selection")
    
    try:
        # Initialize feature selector
        selector = FeatureSelector(config, logger)
        
        # Select features
        selected_features = selector.select_features(features_df, returns_df)
        
        return selected_features
        
    except Exception as e:
        logger.error(f"Error in feature selection: {e}")
        return None

# Select features
if features_df is not None and not returns_df.empty:
    selected_features = select_features(features_df, returns_df)
    if selected_features is not None:
        print(f"Selected {len(selected_features)} features")"""

nb.cells.append(nbf.v4.new_code_cell(selection_code))

# Model Training and Evaluation
model_md = "## 5. Model Training and Evaluation"
nb.cells.append(nbf.v4.new_markdown_cell(model_md))

model_code = """def train_and_evaluate_model(features_df: pd.DataFrame, returns_df: pd.DataFrame, selected_features: list):
    \"\"\"Train and evaluate the model.\"\"\"
    logger.info("Starting model training and evaluation")
    
    try:
        # Initialize model trainer
        trainer = ModelTrainer(config, logger)
        
        # Train model
        model = trainer.train_model(features_df, returns_df, selected_features)
        
        # Evaluate model
        evaluation_results = trainer.evaluate_model(model, features_df, returns_df)
        
        return model, evaluation_results
        
    except Exception as e:
        logger.error(f"Error in model training and evaluation: {e}")
        return None, None

# Train and evaluate model
if selected_features is not None:
    model, results = train_and_evaluate_model(features_df, returns_df, selected_features)
    if model is not None and results is not None:
        print("Model training and evaluation completed successfully")"""

nb.cells.append(nbf.v4.new_code_cell(model_code))

# Complete Pipeline
pipeline_md = "## 6. Run Complete Pipeline"
nb.cells.append(nbf.v4.new_markdown_cell(pipeline_md))

pipeline_code = """def run_pipeline(date_str: str, mode: str = 'day'):
    \"\"\"Run the complete pipeline for a specific date and mode.\"\"\"
    logger.info(f"Starting pipeline run for date: {date_str} in {mode} mode")
    
    try:
        # 1. Load and process data
        processed_data, returns_data = load_and_process_data(date_str, mode)
        if processed_data is None:
            return False
        
        # 2. Generate features
        features_df = generate_features(processed_data, date_str)
        if features_df is None:
            return False
        
        # 3. Select features
        selected_features = select_features(features_df, returns_data)
        if selected_features is None:
            return False
        
        # 4. Train and evaluate model
        model, results = train_and_evaluate_model(features_df, returns_data, selected_features)
        if model is None or results is None:
            return False
        
        logger.info("Pipeline run completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error in pipeline run: {e}")
        return False

# Run pipeline
success = run_pipeline(date_str, mode='day')
if success:
    print("Pipeline completed successfully")
else:
    print("Pipeline failed")"""

nb.cells.append(nbf.v4.new_code_cell(pipeline_code))

# Write the notebook to a file
with open('notebooks/updated_pipe.ipynb', 'w') as f:
    nbf.write(nb, f) 
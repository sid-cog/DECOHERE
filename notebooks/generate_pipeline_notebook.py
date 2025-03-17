#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to generate the pipeline.ipynb notebook for the DECOHERE project.
"""

import nbformat as nbf
import os
from datetime import datetime, timedelta
import pandas as pd

# Create a new notebook
nb = nbf.v4.new_notebook()

# Title and introduction
markdown_intro = """# DECOHERE Pipeline

This notebook provides an interface to run the DECOHERE quantitative trading pipeline. The pipeline consists of the following components:

1. Data Loading: Load raw financial data
2. Data Processing: Clean and transform the data
3. Feature Generation: Create features for machine learning
4. Feature Selection: Select important features using SHAP
5. Regression: Train and evaluate regression models

The pipeline can be run in three modes:
- **Day Mode**: Process a single day of data
- **Week Mode**: Process a week of data
- **Year Mode**: Process a year of data
"""

# Import libraries
code_imports = """# Import necessary libraries
import os
import sys
import yaml
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from pathlib import Path

# Add the project root to the path
sys.path.append('..')

# Import project modules
from src.data.data_processor import DataProcessor
from src.features.feature_generator import FeatureGenerator
from src.features.feature_selector import FeatureSelector
from src.models.model_trainer import ModelTrainer
from src.visualization.visualizer import Visualizer
"""

# Setup logging
markdown_logging = """## Setup Logging"""

code_logging = """# Configure logging
def setup_logging(config):
    log_level = getattr(logging, config['logging']['level'])
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Create logger
    logger = logging.getLogger('decohere')
    logger.setLevel(log_level)
    
    # Clear existing handlers
    logger.handlers = []
    
    # Create console handler if enabled
    if config['logging'].get('console', True):
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(console_handler)
    
    # Create file handler if log file is specified
    if 'file' in config['logging']:
        log_dir = os.path.dirname(config['logging']['file'])
        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(config['logging']['file'])
        file_handler.setLevel(log_level)
        file_handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(file_handler)
    
    return logger
"""

# Load configuration
markdown_config = """## Load Configuration"""

code_config = """# Load the main configuration file
def load_main_config():
    config_path = '../config/config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

# Load mode-specific configuration
def load_mode_config(mode):
    main_config = load_main_config()
    mode_config_path = main_config['modes'][mode]['config_file']
    
    # Convert relative path to absolute path
    if not os.path.isabs(mode_config_path):
        mode_config_path = os.path.join('..', mode_config_path)
    
    with open(mode_config_path, 'r') as f:
        mode_config = yaml.safe_load(f)
    
    # Merge mode config with main config
    merged_config = {**main_config, **mode_config}
    return merged_config

# Load the main configuration
main_config = load_main_config()
print(f"Loaded main configuration from {os.path.abspath('../config/config.yaml')}")
"""

# Select pipeline mode
markdown_mode = """## Select Pipeline Mode

Choose the mode to run the pipeline in:
- **day**: Process a single day of data
- **week**: Process a week of data
- **year**: Process a year of data"""

code_mode = """# Select the mode
import ipywidgets as widgets
from IPython.display import display

mode_dropdown = widgets.Dropdown(
    options=['day', 'week', 'year'],
    value='day',
    description='Mode:',
    disabled=False,
)

# Use '2024-09-11' as the default date
default_date = datetime.strptime('2024-09-11', '%Y-%m-%d').date()
date_picker = widgets.DatePicker(
    description='Date:',
    disabled=False,
    value=default_date
)

display(widgets.VBox([mode_dropdown, date_picker]))

# Function to get the selected mode and date
def get_mode_and_date():
    mode = mode_dropdown.value
    date = date_picker.value
    
    # If date is None, use the default date
    if date is None:
        date = default_date
        
    return mode, date

# Load the configuration for the selected mode
def load_selected_config():
    mode, _ = get_mode_and_date()
    config = load_mode_config(mode)
    return config
"""

# Data loading
markdown_data_loading = """## 1. Data Loading

Load raw financial data based on the selected mode and date."""

code_data_loading = """# Load the raw data
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
    
    # Load the raw data
    logger.info("Loading raw data...")
    raw_data = data_processor.load_raw_data(date=date_str)
    logger.info(f"Loaded {len(raw_data)} rows of raw data")
    
    return raw_data, data_processor, config, logger

# Execute data loading
raw_data, data_processor, config, logger = load_data()

# Display a sample of the raw data
display(raw_data.head())
print(f"Raw data shape: {raw_data.shape}")
"""

# Data processing
markdown_data_processing = """## 2. Data Processing

Clean and transform the raw data."""

code_data_processing = """# Process the raw data
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
    
    logger.info(f"Processing data from {start_date} to {end_date}")
    
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
    
    logger.info(f"Processed data shape: {processed_data.shape}")
    return processed_data

# Execute data processing
processed_data = process_data(raw_data, data_processor, logger)

# Display a sample of the processed data
display(processed_data.head())
print(f"Processed data shape: {processed_data.shape}")
"""

# Feature generation
markdown_feature_generation = """## 3. Feature Generation

Generate features for machine learning."""

code_feature_generation = """# Generate features
def generate_features(processed_data, config, logger):
    logger.info("Generating features...")
    feature_generator = FeatureGenerator(config, logger)
    features = feature_generator.generate_features(processed_data)
    logger.info(f"Generated features shape: {features.shape}")
    return features, feature_generator

# Execute feature generation
features, feature_generator = generate_features(processed_data, config, logger)

# Display a sample of the generated features
display(features.head())
print(f"Features shape: {features.shape}")
"""

# Feature selection
markdown_feature_selection = """## 4. Feature Selection

Select important features using SHAP."""

code_feature_selection = """# Select features
def select_features(features, config, logger):
    logger.info("Selecting features...")
    feature_selector = FeatureSelector(config, logger)
    
    # Get target variables from config
    target_vars = config['features']['targets']
    
    # Prepare X and y
    X = features.drop(columns=target_vars)
    y = features[target_vars[0]]  # Use the first target variable
    
    # Select features
    selected_features = feature_selector.select_features(X, y)
    logger.info(f"Selected {len(selected_features)} features")
    
    # Create dataset with selected features and target
    selected_data = features[selected_features + target_vars]
    
    return selected_data, selected_features, feature_selector

# Execute feature selection
selected_data, selected_features, feature_selector = select_features(features, config, logger)

# Display the selected features
print("Selected features:")
print(selected_features)
print(f"\\nSelected data shape: {selected_data.shape}")
display(selected_data.head())
"""

# Model training and evaluation
markdown_model = """## 5. Model Training and Evaluation

Train and evaluate regression models."""

code_model = """# Train and evaluate models
def train_and_evaluate_model(selected_data, config, logger):
    logger.info("Training and evaluating model...")
    model_trainer = ModelTrainer(config, logger)
    
    # Get target variables from config
    target_vars = config['features']['targets']
    
    # Prepare X and y
    X = selected_data.drop(columns=target_vars)
    y = selected_data[target_vars[0]]  # Use the first target variable
    
    # Train and evaluate model
    model, metrics = model_trainer.train_and_evaluate(X, y)
    
    return model, metrics, model_trainer

# Execute model training and evaluation
model, metrics, model_trainer = train_and_evaluate_model(selected_data, config, logger)

# Display model metrics
print("Model Metrics:")
for metric_name, metric_value in metrics.items():
    print(f"{metric_name}: {metric_value:.4f}")
"""

# Visualization
markdown_visualization = """## 6. Visualization

Visualize the results."""

code_visualization = """# Visualize results
def visualize_results(model, selected_data, selected_features, feature_selector, config, logger):
    logger.info("Visualizing results...")
    visualizer = Visualizer(config, logger)
    
    # Get target variables from config
    target_vars = config['features']['targets']
    
    # Prepare X and y
    X = selected_data.drop(columns=target_vars)
    y = selected_data[target_vars[0]]  # Use the first target variable
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    visualizer.plot_feature_importance(model, X.columns)
    plt.tight_layout()
    plt.show()
    
    # Plot SHAP values
    plt.figure(figsize=(12, 8))
    visualizer.plot_shap_values(model, X)
    plt.tight_layout()
    plt.show()
    
    # Plot actual vs predicted values
    plt.figure(figsize=(10, 6))
    y_pred = model.predict(X)
    visualizer.plot_actual_vs_predicted(y, y_pred)
    plt.tight_layout()
    plt.show()
    
    return visualizer

# Execute visualization
visualizer = visualize_results(model, selected_data, selected_features, feature_selector, config, logger)
"""

# Save results
markdown_save = """## 7. Save Results

Save the processed data, features, model, and results."""

code_save = """# Save results
def save_results(processed_data, features, selected_data, model, metrics, config, logger):
    logger.info("Saving results...")
    
    # Get the selected mode and date
    mode, date = get_mode_and_date()
    date_str = date.strftime('%Y-%m-%d')
    
    # Create output directory
    output_dir = os.path.join('..', 'data', 'results', mode, date_str)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save processed data
    processed_data_path = os.path.join(output_dir, 'processed_data.csv')
    processed_data.to_csv(processed_data_path, index=False)
    logger.info(f"Saved processed data to {processed_data_path}")
    
    # Save features
    features_path = os.path.join(output_dir, 'features.csv')
    features.to_csv(features_path, index=False)
    logger.info(f"Saved features to {features_path}")
    
    # Save selected data
    selected_data_path = os.path.join(output_dir, 'selected_data.csv')
    selected_data.to_csv(selected_data_path, index=False)
    logger.info(f"Saved selected data to {selected_data_path}")
    
    # Save model
    import joblib
    model_path = os.path.join(output_dir, 'model.joblib')
    joblib.dump(model, model_path)
    logger.info(f"Saved model to {model_path}")
    
    # Save metrics
    import json
    metrics_path = os.path.join(output_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Saved metrics to {metrics_path}")
    
    return output_dir

# Execute saving results
output_dir = save_results(processed_data, features, selected_data, model, metrics, config, logger)
print(f"Results saved to {output_dir}")
"""

# Summary
markdown_summary = """## 8. Summary

Display a summary of the pipeline run."""

code_summary = """# Display summary
def display_summary(raw_data, processed_data, features, selected_data, selected_features, metrics, config):
    # Get the selected mode and date
    mode, date = get_mode_and_date()
    date_str = date.strftime('%Y-%m-%d')
    
    print("\\n" + "=" * 80)
    print(f"DECOHERE Pipeline Summary - {mode.upper()} MODE - {date_str}")
    print("=" * 80)
    
    print("\\nData Processing:")
    print(f"  Raw data shape: {raw_data.shape}")
    print(f"  Processed data shape: {processed_data.shape}")
    
    print("\\nFeature Engineering:")
    print(f"  Total features generated: {features.shape[1] - len(config['features']['targets'])}")
    print(f"  Selected features: {len(selected_features)}")
    
    print("\\nModel Performance:")
    for metric_name, metric_value in metrics.items():
        print(f"  {metric_name}: {metric_value:.4f}")
    
    print("\\nTop 10 Important Features:")
    for i, feature in enumerate(selected_features[:10]):
        print(f"  {i+1}. {feature}")
    
    print("\\nResults saved to:")
    print(f"  {os.path.abspath(output_dir)}")
    
    print("\\n" + "=" * 80)

# Execute summary display
display_summary(raw_data, processed_data, features, selected_data, selected_features, metrics, config)
"""

# Add cells to the notebook
cells = [
    nbf.v4.new_markdown_cell(markdown_intro),
    nbf.v4.new_code_cell(code_imports),
    nbf.v4.new_markdown_cell(markdown_logging),
    nbf.v4.new_code_cell(code_logging),
    nbf.v4.new_markdown_cell(markdown_config),
    nbf.v4.new_code_cell(code_config),
    nbf.v4.new_markdown_cell(markdown_mode),
    nbf.v4.new_code_cell(code_mode),
    nbf.v4.new_markdown_cell(markdown_data_loading),
    nbf.v4.new_code_cell(code_data_loading),
    nbf.v4.new_markdown_cell(markdown_data_processing),
    nbf.v4.new_code_cell(code_data_processing),
    nbf.v4.new_markdown_cell(markdown_feature_generation),
    nbf.v4.new_code_cell(code_feature_generation),
    nbf.v4.new_markdown_cell(markdown_feature_selection),
    nbf.v4.new_code_cell(code_feature_selection),
    nbf.v4.new_markdown_cell(markdown_model),
    nbf.v4.new_code_cell(code_model),
    nbf.v4.new_markdown_cell(markdown_visualization),
    nbf.v4.new_code_cell(code_visualization),
    nbf.v4.new_markdown_cell(markdown_save),
    nbf.v4.new_code_cell(code_save),
    nbf.v4.new_markdown_cell(markdown_summary),
    nbf.v4.new_code_cell(code_summary),
]

nb['cells'] = cells

# Set notebook metadata
nb.metadata = {
    'kernelspec': {
        'display_name': 'Python 3',
        'language': 'python',
        'name': 'python3'
    },
    'language_info': {
        'codemirror_mode': {
            'name': 'ipython',
            'version': 3
        },
        'file_extension': '.py',
        'mimetype': 'text/x-python',
        'name': 'python',
        'nbconvert_exporter': 'python',
        'pygments_lexer': 'ipython3',
        'version': '3.8.10'
    }
}

# Write the notebook to a file
with open('pipeline.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Generated pipeline.ipynb successfully!") 
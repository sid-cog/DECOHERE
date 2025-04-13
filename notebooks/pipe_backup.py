# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
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

# %% tags=["always_run", "always run", "core_run", "alqways"]
# --- Cell 1: Setup and Initialization ---

import os
import sys
import logging
import yaml
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime
import pytz  # We'll need to install this

# Add project root to Python path
project_root = str(Path(os.getcwd()))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import project modules
from src.data.data_processor import DataProcessor
from src.data.efficient_data_storage import EfficientDataStorage, DataType, DataStage
from src.features.feature_selector import FeatureSelector
from src.data.feature_generator import FeatureGenerator

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', force=True)

# Configuration loading functions
def deep_merge(base_dict: Dict[str, Any], override_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two dictionaries, with override_dict taking precedence."""
    merged = base_dict.copy()
    for key, value in override_dict.items():
        if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_mode_config(mode: str, base_config_path: str) -> Dict[str, Any]:
    """Load mode-specific configuration and merge with main config."""
    base_config = load_config(base_config_path)
    mode_config_path = base_config['modes'][mode]['config_file']
    mode_config = load_config(mode_config_path)
    return deep_merge(base_config, mode_config)

# Load configuration
config_path = os.path.join(project_root, 'config', 'config.yaml')
config = load_config(config_path)  # For now, using base config
# If you need mode-specific config, use: config = load_mode_config('your_mode', config_path)

# Initialize classes
processor = DataProcessor(config, logging.getLogger('Processor'))
storage = EfficientDataStorage(config, logging.getLogger('Storage'))
feature_generator = FeatureGenerator(config, logging.getLogger('FeatureGenerator'))

# Initialize FeatureSelector with correct config structure
feature_selection_config = {
    'data': {
        'base_dir': config['data']['base_dir'],  # This is what FeatureSelector expects
        'features': {
            'fundamentals': config['data']['features']['fundamentals']
        }
    },
    'feature_selection': config['feature_selection'],
    'output': {
        'results_dir': os.path.join(config['data']['base_dir'], 'results', 'feature_selection')
    }
}

# Generate descriptive run name with Sydney time if no custom name provided
def generate_run_name(config: dict) -> str:
    """Generate a descriptive run name with Sydney timestamp."""
    # Get Sydney time
    sydney_tz = pytz.timezone('Australia/Sydney')
    sydney_time = datetime.now(sydney_tz)
    
    # Extract feature selection parameters
    method = config['feature_selection'].get('method', 'shap_threshold')[:4]
    min_thresh = f"t{int(config['feature_selection'].get('min_threshold', 0.01) * 100)}"
    min_feat = f"min{config['feature_selection'].get('min_features', 10)}"
    max_feat = f"max{config['feature_selection'].get('max_features', 40)}"
    cumul = "cum" if config['feature_selection'].get('use_cumulative', True) else "nocum"
    cumul_thresh = f"c{int(config['feature_selection'].get('cumulative_threshold', 0.95) * 100)}" if cumul == "cum" else ""
    
    # Format timestamp
    timestamp = sydney_time.strftime("%Y%m%d_%H%M_SYD")
    
    # Combine into run name
    run_name = f"{method}_{min_thresh}_{min_feat}_{max_feat}_{cumul}{cumul_thresh}_{timestamp}"
    return run_name

# You can specify a custom run_id here, or leave it as None for automatic generation
run_id = None  # Change this to your desired run name if needed
if run_id is None:
    run_id = generate_run_name(feature_selection_config)
    logging.info(f"Generated run name: {run_id}")

feature_selector = FeatureSelector(feature_selection_config, run_id=run_id, logger=logging.getLogger('FeatureSelector'))


# %%
# --- Cell 2: Processing Function (with Enhanced Features) ---
def run_pipeline_for_date(date_str: str, processor: DataProcessor, storage: EfficientDataStorage, 
                         feature_generator: Optional[FeatureGenerator], config: dict):
    """Runs the full data pipeline for a single date, including enhanced features if available."""
    
    if not processor or not storage:
        logging.error(f"[{date_str}] Processor or Storage not initialized. Aborting.")
        return False, None, None, None

    logging.info(f"--- Starting Pipeline for Date: {date_str} ---")
    processed_file_path = None
    pre_feature_file_path = None
    enhanced_feature_file_path = None
    success = False

    try:
        # 1. Load and process raw data
        raw_data = processor.load_raw_data(date_str)
        if raw_data.empty:
            logging.warning(f"[{date_str}] No raw data found. Skipping remaining steps.")
            return True, None, None, None

        # 2. Process data
        transformed_data = processor.transform_raw_data(raw_data)
        filled_data = processor.fill_missing_values(transformed_data)

        # 3. Store processed data
        processed_file_path = storage.store_data(
            df=filled_data, 
            data_type=DataType.FUNDAMENTALS,
            stage=DataStage.PROCESSED, 
            date=date_str
        )

        # 4. Generate and store pre-feature data
        pre_feature_df = storage.processed_data_feat_gen(filled_data)
        if not pre_feature_df.empty:
            pre_feature_file_path = storage.store_data(
                df=pre_feature_df, 
                data_type=DataType.FUNDAMENTALS,
                stage=DataStage.FEATURES, 
                date=date_str, 
                sub_type='pre_feature_set'
            )

            # 5. Generate enhanced features if generator available
            if feature_generator:
                # Get parameters from config
                feature_cfg = config.get('features', {})
                hist_w = feature_cfg.get('hist_window', 6)
                fwd_w = feature_cfg.get('fwd_window', 6)
                target_m = feature_cfg.get('target_metric', 'PE_RATIO_RATIO_SIGNED_LOG')
                sector_map_rel_path = feature_cfg.get('sector_mapping_path', None)
                sector_map_abs_path = os.path.join(project_root, sector_map_rel_path) if sector_map_rel_path else None
                sector_levels = feature_cfg.get('sector_levels_to_include', ['sector_1'])
                include_sectors = feature_cfg.get('include_sector_features', True)

                enhanced_feature_df = feature_generator.generate_enhanced_features(
                    df=pre_feature_df,
                    hist_window=hist_w,
                    fwd_window=fwd_w,
                    target_metric=target_m,
                    sector_mapping_path=sector_map_abs_path,
                    sector_levels_to_include=sector_levels,
                    include_sector_features=include_sectors
                )
                
                if not enhanced_feature_df.empty:
                    enhanced_feature_file_path = storage.store_data(
                        df=enhanced_feature_df, 
                        data_type=DataType.FUNDAMENTALS,
                        stage=DataStage.FEATURES, 
                        date=date_str, 
                        sub_type='enhanced_features'
                    )
            else:
                logging.info(f"[{date_str}] FeatureGenerator not available. Skipping enhanced features.")

        success = True

    except Exception as e:
        logging.error(f"[{date_str}] Pipeline error: {e}", exc_info=True)
        success = False

    finally:
        logging.info(f"--- Finished Pipeline for Date: {date_str} (Success: {success}) ---")

    return success, processed_file_path, pre_feature_file_path, enhanced_feature_file_path

print("Pipeline function defined.")

# %%
# --- Cell 3: Execution Loop and Verification ---
dates_to_process = ['2024-09-02', '2024-09-03', '2024-09-04']
results = {}

# Process each date
for date_str in dates_to_process:
    logging.info(f"--- Starting Pipeline Run for Date: {date_str} ---")
    success, proc_path, pre_feat_path, enh_feat_path = run_pipeline_for_date(
        date_str=date_str,
        processor=processor,
        storage=storage,
        feature_generator=feature_generator,
        config=config
    )
    results[date_str] = (success, proc_path, pre_feat_path, enh_feat_path)
    logging.info(f"--- Completed Pipeline Run for Date: {date_str} ---")

# Verify results
def verify_pipeline_output(date_str: str, result: tuple, expect_files: bool = True) -> bool:
    """Verify the pipeline output for a specific date."""
    success, proc_path, pre_feat_path, enh_feat_path = result
    
    if not success:
        logging.error(f"[{date_str}] Pipeline run failed")
        return False
        
    if expect_files:
        # Check if all expected files exist
        for path, file_type in [
            (proc_path, "processed data"),
            (pre_feat_path, "pre-feature data"),
            (enh_feat_path, "enhanced features")
        ]:
            if path and not os.path.exists(path):
                logging.error(f"[{date_str}] Missing {file_type} file: {path}")
                return False
            elif path:
                logging.info(f"[{date_str}] Verified {file_type} file: {path}")
    
    return True

# Verify all results
all_success = True
for date_str, result in results.items():
    if not verify_pipeline_output(date_str, result):
        all_success = False
        logging.error(f"[{date_str}] Verification failed")
    else:
        logging.info(f"[{date_str}] Verification successful")

if all_success:
    logging.info("All pipeline runs completed and verified successfully")
else:
    logging.error("Some pipeline runs failed verification")

print("Pipeline execution and verification complete.")

# %%
#Comparison block

# s =pd.read_parquet('/home/siddharth.johri/DECOHERE/data/raw/fundamentals/financials_2024_09.pq')
# a = pd.read_parquet('/home/siddharth.johri/DECOHERE/data/processed/fundamentals/year=2024/month=09/data_2024-09-04.pq')
# b = pd.read_parquet('/home/siddharth.johri/DECOHERE/data/features/fundamentals/pre_feature_set/year=2024/month=09/data_2024-09-04.pq')
c = pd.read_parquet('/home/siddharth.johri/DECOHERE/data/features/fundamentals/enhanced_features/year=2024/month=09/data_2024-09-04.pq')
# a.query('ID == "INFO IB Equity" & PERIOD_END_DATE == "2024-03-31"')['PE_RATIO_RATIO']
c.query('ID == "INFO IB Equity"')['PE_RATIO_RATIO_SIGNED_LOG']
list(c.columns)

# %%
# --- Cell 4: Feature Selection ---
target_date = '2024-09-02'

try:
    # Select features for target date
    print(f"\nSelecting features for date: {target_date}")
    feature_selection_results = feature_selector.select_features_daily(target_date)
    
    if feature_selection_results is None:
        print("No features were selected. Check the data and preprocessing steps.")
    else:
        # Print feature selection results
        print("\nFeature Selection Results:")
        print(f"Total features after preprocessing: {len(feature_selection_results['importance_scores'])}")
        print(f"Selected features: {len(feature_selection_results['selected_features'])}")
        
        if feature_selection_results['selected_features']:
            print("\nSelected Features (with importance scores):")
            importance_df = feature_selection_results['importance_scores']
            for feature in feature_selection_results['selected_features']:
                mean_imp = importance_df.loc[feature, 'mean_importance']
                std_imp = importance_df.loc[feature, 'std_importance']
                print(f"- {feature} (Mean importance: {mean_imp:.4f}, Std: {std_imp:.4f})")
        
        # The visualization will be automatically generated by the FeatureSelector class
        # and saved to the results directory
        
except Exception as e:
    print(f"Error during feature selection: {str(e)}")
    print("Check the logs for more detailed error information.")


# %%
def run_feature_selection(date: str, run_id: Optional[str] = None) -> Dict[str, List[str]]:
    """
    Run feature selection for a specific date.
    
    Args:
        date: Date in YYYY-MM-DD format
        run_id: Optional custom run identifier. If None, an intuitive name will be generated
        
    Returns:
        Dictionary containing selected features for each method
    """
    try:
        logger.info(f"Starting feature selection for date: {date}")
        
        # Initialize feature selector
        feature_selector = FeatureSelector(config, run_id=run_id, logger=logger)
        
        # Select features for the date
        results = feature_selector.select_features_daily(date)
        
        if results:
            logger.info(f"Feature selection completed successfully for {date}")
            logger.info(f"Selected features saved in run directory: {feature_selector.results_dir}")
        else:
            logger.warning(f"No features selected for {date}")
            
        return results
        
    except Exception as e:
        logger.error(f"Error in feature selection for {date}: {str(e)}")
        return None


# %%
def analyze_feature_stability(start_date: str, end_date: str, run_id: str) -> Dict[str, Any]:
    """
    Analyze feature stability across a date range.
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        run_id: Run identifier to analyze
        
    Returns:
        Dictionary containing stability analysis results
    """
    try:
        logger.info(f"Analyzing feature stability from {start_date} to {end_date}")
        
        # Initialize feature selector with the same run_id
        feature_selector = FeatureSelector(config, run_id=run_id, logger=logger)
        
        # Analyze stability
        stability_metrics = feature_selector.analyze_feature_stability(start_date, end_date)
        
        logger.info(f"Stability analysis completed. Results saved in run directory: {feature_selector.results_dir}")
        
        return stability_metrics
        
    except Exception as e:
        logger.error(f"Error in stability analysis: {str(e)}")
        return None


# %%
# Example: Run feature selection for a specific date
date = "2024-03-15"
results = run_feature_selection(date)

# Example: Run feature selection with custom run ID
custom_run_id = "my_experiment_1"
results = run_feature_selection(date, run_id=custom_run_id)

# Example: Analyze feature stability
start_date = "2024-03-01"
end_date = "2024-03-15"
stability = analyze_feature_stability(start_date, end_date, custom_run_id)

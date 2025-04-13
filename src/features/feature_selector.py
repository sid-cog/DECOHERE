#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Feature Selection for the DECOHERE project.
This module contains the FeatureSelector class, which implements various feature selection methods.
"""

import os
import pandas as pd
import numpy as np
import logging
import shap
import xgboost as xgb
import matplotlib.pyplot as plt
import optuna
import json
from typing import Dict, List, Tuple, Optional, Any, Union, Set
from pathlib import Path
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import TimeSeriesSplit, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNetCV, ElasticNet
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import mutual_info_regression
import seaborn as sns
import re


class NumpyEncoder(json.JSONEncoder):
    """Custom encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

class FeatureSelector:
    """
    Feature Selector class for the DECOHERE project.
    Implements various feature selection methods for both Elastic Net and XGBoost models.
    """
    
    def __init__(self, config: Dict[str, Any], run_id: Optional[str] = None, logger: Optional[logging.Logger] = None):
        """
        Initialize the FeatureSelector.
        
        Args:
            config: Configuration dictionary
            run_id: Optional custom run identifier. If None, an intuitive name will be generated
            logger: Logger instance
        """
        self.config = config
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logger or logging.getLogger(__name__)
        self.logger.info("Initializing FeatureSelector")
        
        # Get feature selection parameters from config
        self.missing_threshold = config.get('feature_selection', {}).get('missing_threshold', 0.5)  # Changed to 50%
        self.correlation_threshold = config.get('feature_selection', {}).get('correlation_threshold', 0.95)
        self.min_variance = config.get('feature_selection', {}).get('min_variance', 0.0)
        
        # Get model-specific parameters
        self.elastic_net_params = config.get('feature_selection', {}).get('elastic_net', {})
        self.xgboost_params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'early_stopping_rounds': 10,
            'random_state': 42,
            'enable_categorical': True
        }
        
        # Initialize scaler
        self.scaler = StandardScaler()
        
        # Extract configuration parameters
        self.features_base_dir = config['data']['base_dir']
        self.features_fundamental_dir = os.path.join(
            self.features_base_dir, 
            'features',
            'fundamentals',
            'enhanced_features'
        )
        
        # Feature selection parameters
        self.method = config['feature_selection'].get('method', 'shap_threshold')
        self.min_threshold = config['feature_selection'].get('min_threshold', 0.01)
        self.min_features = config['feature_selection'].get('min_features', 10)
        self.max_features = config['feature_selection'].get('max_features', 40)
        self.cumulative_threshold = config['feature_selection'].get('cumulative_threshold', 0.95)
        self.use_cumulative = config['feature_selection'].get('use_cumulative', True)
        
        # Generate or use provided run_id
        self.run_id = run_id if run_id else self._generate_run_id()
        
        # Create results directory with run_id
        self.results_dir = os.path.join(
            config.get('output', {}).get('results_dir', 'results/feature_selection'),
            f"run_{self.run_id}"
        )
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Save run configuration
        self._save_run_config()
        
        # Feature selection thresholds
        self.importance_threshold = config.get('feature_selection', {}).get('importance_threshold', 0.01)
        self.stability_threshold = config.get('feature_selection', {}).get('stability_threshold', 0.7)
    
    def _generate_run_id(self) -> str:
        """
        Generate an intuitive run ID based on key parameters.
        
        Returns:
            String containing the generated run ID
        """
        # Extract key parameters for naming
        method = self.method[:4]  # First 4 chars of method name
        min_thresh = f"t{int(self.min_threshold * 100)}"  # Threshold as percentage
        min_feat = f"min{self.min_features}"
        max_feat = f"max{self.max_features}"
        cumul = "cum" if self.use_cumulative else "nocum"
        cumul_thresh = f"c{int(self.cumulative_threshold * 100)}" if self.use_cumulative else ""
        
        # Combine into run ID
        run_id = f"{method}_{min_thresh}_{min_feat}_{max_feat}_{cumul}{cumul_thresh}"
        
        # Add timestamp to ensure uniqueness
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        run_id = f"{run_id}_{timestamp}"
        
        return run_id
    
    def _save_run_config(self) -> None:
        """Save the configuration for this run."""
        config_file = os.path.join(self.results_dir, "run_config.json")
        config_to_save = {
            'run_id': self.run_id,
            'feature_selection': {
                'missing_threshold': self.missing_threshold,
                'correlation_threshold': self.correlation_threshold,
                'min_variance': self.min_variance,
                'method': self.method,
                'min_threshold': self.min_threshold,
                'min_features': self.min_features,
                'max_features': self.max_features,
                'cumulative_threshold': self.cumulative_threshold,
                'use_cumulative': self.use_cumulative
            },
            'elastic_net_params': self.elastic_net_params,
            'xgboost_params': self.xgboost_params
        }
        
        with open(config_file, 'w') as f:
            json.dump(config_to_save, f, indent=4)
        
        self.logger.info(f"Saved run configuration to {config_file}")
    
    def _load_processed_data(self, date_str: str) -> Optional[pd.DataFrame]:
        """Loads the processed feature data for a single date."""
        self.logger.debug(f"Attempting to load processed data for date: {date_str}")
        try:
            date_obj = pd.to_datetime(date_str)
            year = date_obj.year
            month = f"{date_obj.month:02d}"
            # Use the specified path structure
            file_path = os.path.join(
                self.features_fundamental_dir, # Assumes this points to enhanced_features dir
                f"year={year}",
                f"month={month}",
                f"data_{date_str}.pq"
            )

            if not os.path.exists(file_path):
                self.logger.warning(f"Processed data file not found: {file_path}")
                return None

            df = pd.read_parquet(file_path)
            self.logger.debug(f"Loaded data from {file_path} with shape {df.shape}")
            return df
        except FileNotFoundError:
            self.logger.warning(f"Processed data file explicitly not found (FileNotFoundError): {file_path}")
            return None
        except Exception as e:
            self.logger.error(f"Error loading processed data from {file_path}: {e}", exc_info=True)
            return None

    def _train_xgboost(self, X_train: pd.DataFrame, y_train: pd.Series, params: Optional[Dict] = None) -> Optional[xgb.XGBRegressor]:
        """Placeholder: Trains a basic XGBoost model with categorical support."""
        self.logger.debug(f"Training XGBoost model on data shape: {X_train.shape}")
        default_params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'n_estimators': 100, 
            'seed': 42,
            'verbosity': 0,
            'enable_categorical': True
        }
        model_params = {**default_params, **(params or {})} 
        # Ensure enable_categorical is True if overridden
        model_params['enable_categorical'] = True 
        try:
            # Convert category columns to pd.Categorical if not already?
            # XGBoost should handle this, but check documentation if errors occur.
            model = xgb.XGBRegressor(**model_params)
            model.fit(X_train, y_train, verbose=False)
            return model
        except Exception as e:
            self.logger.error(f"Error during placeholder XGBoost training: {e}", exc_info=True)
            return None
            
    def _calculate_feature_importance(self, model: xgb.XGBRegressor, feature_names: List[str]) -> Optional[pd.DataFrame]:
         """Placeholder: Calculates feature importance."""
         self.logger.debug("Calculating feature importance (placeholder).")
         try:
             importances = model.feature_importances_
             if importances is None or len(importances) != len(feature_names):
                 self.logger.error("Failed to get valid importances from model.")
                 return None
             # Return DataFrame matching structure expected by select_features_daily
             importance_df = pd.DataFrame({
                 'feature': feature_names,
                 'mean_importance': importances, # Use direct importance as mean
                 'std_importance': np.zeros_like(importances) # Placeholder std dev
             })
             return importance_df.sort_values('mean_importance', ascending=False)
         except Exception as e:
             self.logger.error(f"Error calculating placeholder feature importance: {e}", exc_info=True)
             return None

    def _calculate_shap_values(self, model: xgb.XGBRegressor, X_eval: pd.DataFrame, selected_features: List[str]) -> Tuple[Optional[List[np.ndarray]], Optional[List[np.ndarray]]]:
        """Calculates SHAP values for the given model and evaluation data."""
        self.logger.debug("Calculating SHAP values. This may be slow.")
        try:
            explainer = shap.TreeExplainer(model)
            shap_values_full = explainer.shap_values(X_eval)
            
            # Ensure SHAP values are a numpy array
            if not isinstance(shap_values_full, np.ndarray):
                self.logger.error("SHAP values are not a numpy array, received type: " + str(type(shap_values_full)))
                return [shap_values_full] if shap_values_full is not None else None, None
            
            # Ensure selected_features are in X_eval columns for indexing
            valid_selected_features = [f for f in selected_features if f in X_eval.columns]
            if len(valid_selected_features) != len(selected_features):
                self.logger.warning("Some selected features not found in X_eval for SHAP indexing.")

            # Extract SHAP for selected features if possible
            shap_values_selected = None
            if valid_selected_features:
                try:
                    # Get column indices for the selected features
                    selected_indices = [X_eval.columns.get_loc(f) for f in valid_selected_features]
                    
                    # Check the dimensionality of SHAP values
                    if shap_values_full.ndim == 2:
                        # For a 2D array, we can index directly with the column indices
                        # Convert list to numpy array for proper slicing
                        selected_indices_array = np.array(selected_indices)
                        shap_values_selected = shap_values_full[:, selected_indices_array]
                    else:
                        self.logger.warning(f"SHAP values have unexpected dimensions: {shap_values_full.ndim}. Cannot select subset.")
                except Exception as idx_e:
                    self.logger.error(f"Error indexing SHAP values for selected features: {idx_e}")
            
            # Return in the expected list format (even if only one set)
            return [shap_values_full], [shap_values_selected] if shap_values_selected is not None else None
        except Exception as e:
            self.logger.error(f"Error calculating SHAP values: {e}", exc_info=True)
            return None, None

    def _select_features_with_xgboost(self, X: pd.DataFrame, y: pd.Series) -> Tuple[Optional[List[str]], Optional[pd.DataFrame], Optional[List[np.ndarray]], Optional[List[np.ndarray]]]:
        """Placeholder: Runs the core XGBoost/SHAP feature selection when no subset is provided."""
        # Added assertion for y type
        assert isinstance(y, pd.Series), "Target y is not a pandas Series"
        self.logger.info("Running placeholder feature selection with XGBoost/SHAP.")
        # This should contain your original logic for fitting, getting importance/SHAP, and selecting features
        # For now, just train, get importance/SHAP, and select based on simple importance threshold
        model = self._train_xgboost(X, y)
        if not model: return None, None, None, None
        
        feature_names = list(X.columns)
        importance_df = self._calculate_feature_importance(model, feature_names)
        if importance_df is None: return None, None, None, None
        
        # Simple selection based on mean importance > 0 (or a small threshold)
        selected_features = importance_df[importance_df['mean_importance'] > 1e-6]['feature'].tolist()
        if not selected_features:
            self.logger.warning("Placeholder selection didn't select any features based on importance > 1e-6")
            # Fallback: select top N features? For now, return empty.
            return [], importance_df, None, None 
        
        # Calculate SHAP (potentially slow)
        # Pass X itself as X_eval for this placeholder
        shap_full_list, shap_selected_list = self._calculate_shap_values(model, X, selected_features)

        return selected_features, importance_df, shap_full_list, shap_selected_list

    def preprocess_features(self, X: pd.DataFrame, y: pd.Series, feature_subset: Optional[List[str]] = None) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series], Optional[List[str]]]:
        """
        Preprocesses features: handles missing values, converts categoricals,
        drops remaining non-numeric/non-categorical, optionally filters to subset,
        removes high correlation.
        """
        self.logger.info("Starting feature preprocessing (incl. categorical support)...")
        X_processed = X.copy(); y_processed = y.copy()
        if feature_subset:
            self.logger.info(f"Filtering features to provided subset of {len(feature_subset)} features.")
            valid_subset = [str(col) for col in feature_subset if str(col) in X_processed.columns]
            missing_subset_cols = set(map(str, feature_subset)) - set(valid_subset)
            if missing_subset_cols: self.logger.warning(f"Columns from subset not found: {missing_subset_cols}")
            if not valid_subset: self.logger.error("Subset resulted in no valid columns."); return None, None, None
            X_processed = X_processed[valid_subset]
            self.logger.info(f"Using {len(valid_subset)} valid features from subset.")
            
        initial_rows = len(X_processed)
        na_target_mask = y_processed.isna()
        if na_target_mask.any():
            self.logger.warning(f"Target variable has {na_target_mask.sum()} NaNs. Dropping rows.")
            X_processed = X_processed[~na_target_mask]; y_processed = y_processed[~na_target_mask]
            if X_processed.empty: self.logger.error("No rows after target NA drop."); return None, None, None
        self.logger.info(f"Rows after target NA drop: {len(X_processed)} (dropped {initial_rows - len(X_processed)})")

        # --- Step 2: Identify and Prepare Column Types --- 
        numeric_cols = X_processed.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = X_processed.select_dtypes(include='category').columns.tolist()
        # Identify columns intended to be categorical but aren't yet (e.g., SECTOR_* columns)
        potential_categorical_cols = [col for col in X_processed.columns 
                                       if col.startswith('SECTOR_') and col not in categorical_cols and col not in numeric_cols]
        
        # Convert potential categoricals (e.g., loaded as object/string)
        for col in potential_categorical_cols:
            self.logger.info(f"Converting potential categorical column '{col}' to category dtype.")
            try:
                # Fill NaNs before converting
                if X_processed[col].isnull().any():
                    fill_val = f"Missing_{col}"
                    self.logger.debug(f"Filling {X_processed[col].isnull().sum()} NaNs in '{col}' with '{fill_val}'")
                    X_processed[col] = X_processed[col].fillna(fill_val)
                X_processed[col] = X_processed[col].astype('category')
                categorical_cols.append(col) # Add to our list of known categoricals
            except Exception as e:
                 self.logger.warning(f"Could not convert '{col}' to category: {e}. It might be dropped.")

        # --- Step 3: Drop Remaining Non-Numeric/Non-Categorical Columns --- 
        potential_id_cols = {'ID', 'SECURITY_ID', 'DATE', 'PIT_DATE', 'PERIOD'} # Keep these out of models
        cols_to_keep = set(numeric_cols) | set(categorical_cols) # Union of numeric and categorical
        cols_to_drop = list(set(X_processed.columns) - cols_to_keep - potential_id_cols)

        if cols_to_drop:
            self.logger.info(f"Dropping columns not identified as numeric, categorical, or ID: {cols_to_drop}")
            X_processed = X_processed.drop(columns=cols_to_drop)
        else:
            self.logger.info("No columns dropped based on type (all numeric, categorical, or ID).")

        # At this point, X_processed should only contain numeric and category types
        # Update the list of columns to process further
        current_feature_cols = numeric_cols + categorical_cols 
        X_processed = X_processed[current_feature_cols] # Ensure order and selection

        # --- Step 4: Handle Missing Numeric Feature Values (Imputation) --- 
        # Only impute numeric columns
        numeric_cols_in_processed = list(set(numeric_cols) & set(X_processed.columns))
        if not numeric_cols_in_processed:
             self.logger.info("No numeric columns remaining for imputation.")
        else:
             missing_numeric_counts = X_processed[numeric_cols_in_processed].isna().sum()
             numeric_cols_with_missing = missing_numeric_counts[missing_numeric_counts > 0]
             if not numeric_cols_with_missing.empty:
                 self.logger.info(f"Imputing missing values using median for {len(numeric_cols_with_missing)} numeric columns.")
                 imputer = SimpleImputer(strategy='median')
                 # Important: Use fit_transform only on columns with missing values
                 try:
                     X_processed[numeric_cols_with_missing.index] = imputer.fit_transform(X_processed[numeric_cols_with_missing.index])
                 except Exception as impute_err:
                     self.logger.error(f"Error during median imputation: {impute_err}. Check numeric columns.", exc_info=True)
                     # Depending on severity, might need to return None or drop problematic cols
             else:
                 self.logger.info("No missing values found in remaining numeric features.")

        # --- Step 5: Remove Highly Correlated Features (Only Numeric) ---
        # Only calculate correlation on numeric features
        correlation_threshold = self.config['feature_selection'].get('correlation_threshold', 0.95)
        initial_numeric_count = len(numeric_cols_in_processed)
        numeric_cols_after_impute = list(set(numeric_cols) & set(X_processed.columns))
        
        if correlation_threshold < 1.0 and len(numeric_cols_after_impute) > 1:
             self.logger.info(f"Removing highly correlated numeric features (> {correlation_threshold})...")
             numeric_df = X_processed[numeric_cols_after_impute]
             try:
                 corr_matrix = numeric_df.corr().abs()
                 upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                 numeric_to_drop = {column for column in upper.columns if any(upper[column] > correlation_threshold)}
                 
                 if numeric_to_drop:
                     self.logger.info(f"Dropping {len(numeric_to_drop)} highly correlated numeric features: {numeric_to_drop}")
                     X_processed = X_processed.drop(columns=list(numeric_to_drop))
                     # Update the list of current features
                     current_feature_cols = [col for col in current_feature_cols if col not in numeric_to_drop]
                 else:
                     self.logger.info("No numeric features dropped due to high correlation.")
             except Exception as corr_e:
                  self.logger.error(f"Error during correlation calculation/removal: {corr_e}")
        self.logger.info(f"Numeric features after correlation removal: {len(list(set(numeric_cols_after_impute) - numeric_to_drop))} (removed {len(numeric_to_drop)})")

        # --- Final Checks --- 
        if X_processed.empty or X_processed.shape[1] == 0: self.logger.error("Preprocessing resulted in zero features."); return None, None, None
        
        final_feature_names = list(X_processed.columns) # Features remaining
        self.logger.info(f"Preprocessing complete. Final shape: X={X_processed.shape}, y={y_processed.shape}")
        y_processed = y_processed.loc[X_processed.index] # Final index alignment
        # Add assertions before final return
        assert isinstance(y_processed, pd.Series) or y_processed is None, "y_processed is not a Series or None"
        assert isinstance(X_processed, pd.DataFrame) or X_processed is None, "X_processed is not a DataFrame or None"
        assert isinstance(final_feature_names, list) or final_feature_names is None, "final_feature_names is not a list or None"
        return X_processed, y_processed, final_feature_names

    def select_features_daily(self,
                              target_date_str: str,
                              target_col: str = 'PE_RATIO_RATIO_SIGNED_LOG',
                              feature_subset: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
        """
        Selects features for a specific date OR uses a pre-defined feature subset.
        """
        self.logger.info(f"Starting daily process for {target_date_str}. Feature subset provided: {feature_subset is not None}")

        # Load data for the target date
        daily_data = self._load_processed_data(target_date_str)
        if daily_data is None or daily_data.empty:
            self.logger.error(f"No data loaded for {target_date_str}. Cannot proceed.")
            return None

        if target_col not in daily_data.columns:
            self.logger.error(f"Target column '{target_col}' not found in data for {target_date_str}.")
            return None

        y = daily_data[target_col]
        X = daily_data.drop(columns=[target_col])

        # Call the single, correct preprocess_features method
        preprocess_result = self.preprocess_features(X, y, feature_subset=feature_subset)
        if preprocess_result is None: 
            self.logger.error(f"Preprocessing failed for {target_date_str}")
            return None
        
        X_processed, y_processed, processed_feature_names = preprocess_result
        
        # Explicit null checks before proceeding
        if X_processed is None or y_processed is None or processed_feature_names is None:
            self.logger.error(f"Preprocessing returned None for {target_date_str}")
            return None

        selected_features: Optional[List[str]] = None
        importance_scores: Optional[pd.DataFrame] = None
        shap_values_list: Optional[List[np.ndarray]] = None
        selected_shap_values_list: Optional[List[np.ndarray]] = None
        model_trained_daily: Optional[xgb.XGBRegressor] = None

        if feature_subset:
            self.logger.info(f"Using provided feature subset ({len(processed_feature_names)} features after preprocessing).")
            selected_features = processed_feature_names
            
            # Train model, get importance & SHAP (with null checks)
            model_trained_daily = self._train_xgboost(X_processed, y_processed)
            
            if model_trained_daily:
                importance_scores = self._calculate_feature_importance(model_trained_daily, selected_features)
                shap_values_list, selected_shap_values_list = self._calculate_shap_values(model_trained_daily, X_processed, selected_features)
            else:
                self.logger.error("Failed to train daily model on pre-selected features.")
        else:
            self.logger.info("Performing feature selection process (no subset provided).")
            
            # Use feature selection logic (with null check)
            select_result = self._select_features_with_xgboost(X_processed, y_processed)
            
            if select_result is None:
                self.logger.error("XGBoost feature selection failed.")
                return None
                
            selected_features, importance_scores, shap_values_list, selected_shap_values_list = select_result
            
            if selected_features is None:
                self.logger.error("XGBoost selection returned None/empty.")
                return None

        # Construct results dictionary
        results = {
            'target_date': target_date_str,
            'selected_features': selected_features or [],
            'importance_scores': importance_scores,
            'shap_values': shap_values_list,
            'selected_shap_values': selected_shap_values_list,
            'feature_names': processed_feature_names or [],
            'X_processed': X_processed,
            'y_processed': y_processed
        }
        return results

    def save_performance_metrics(self, date: str, cv_scores: List[Dict[str, float]], importance_scores: pd.DataFrame, 
                               shap_values: List[np.ndarray], selected_features: List[str]) -> None:
        """Save performance metrics to a JSON file."""
        metrics_file = os.path.join(self.results_dir, f"performance_metrics_{date}.json")
        
        # Convert date to string if it's a Timestamp
        if isinstance(date, pd.Timestamp):
            date = date.strftime("%Y-%m-%d")
        
        # Prepare metrics data
        metrics_data = {
            'date': date,
            'cv_scores': cv_scores,
            'average_metrics': {
                'rmse': np.mean([score['rmse'] for score in cv_scores]),
                'r2': np.mean([score['r2'] for score in cv_scores]),
                'n_trees': np.mean([score['n_trees'] for score in cv_scores])
            },
            'selected_features': selected_features,
            'importance_scores': importance_scores.to_dict(),
            'shap_values_summary': {
                'mean': [values.mean() for values in shap_values],
                'std': [values.std() for values in shap_values],
                'min': [values.min() for values in shap_values],
                'max': [values.max() for values in shap_values]
            }
        }
        
        # Save to JSON
        with open(metrics_file, 'w') as f:
            json.dump(metrics_data, f, indent=4, cls=NumpyEncoder)
        
        self.logger.info(f"Saved performance metrics to {metrics_file}")

    def display_results(self, results: Dict[str, Any]) -> None:
        """
        Display feature selection results in a formatted way.
        
        Args:
            results: Dictionary containing feature selection results
        """
        if not results:
            self.logger.info("No results to display")
            return
            
        self.logger.info(f"\nSelected {len(results['selected_features'])} features:")
        for feature in results['selected_features']:
            self.logger.info(f"- {feature}")
        
        self.logger.info("\nFeature Importance Scores:")
        importance_df = results['importance_scores']
        importance_df = importance_df.sort_values('mean_importance', ascending=False)
        self.logger.info(importance_df.head(10))
        
        self.logger.info("\nSHAP Values Summary:")
        # Create DataFrame with selected features' SHAP values
        shap_df = pd.DataFrame(results['selected_shap_values'][0], columns=results['selected_features'])
        self.logger.info(shap_df.describe())

    def visualize_feature_importance(self, results: Dict[str, Any], top_n: int = 20) -> None:
        """
        Visualize feature importance using both SHAP values and feature importance scores.
        
        Args:
            results: Dictionary containing feature selection results
            top_n: Number of top features to display
        """
        if not results:
            self.logger.warning("No results to visualize")
            return
        
        importance_scores = results['importance_scores']
        shap_values = results['shap_values']
        feature_names = results['feature_names']
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # 1. Feature Importance Plot
        # Get top N features by mean importance
        top_features = importance_scores.nlargest(top_n, 'mean_importance')
        
        # Plot mean importance with error bars
        ax1.errorbar(
            top_features['mean_importance'],
            top_features.index,
            xerr=top_features['std_importance'],
            fmt='o',
            capsize=5
        )
        ax1.set_title('Feature Importance (Mean ± Std across folds)')
        ax1.set_xlabel('Mean SHAP Importance')
        ax1.grid(True, alpha=0.3)
        
        # 2. SHAP Value Distribution Plot
        # Calculate mean absolute SHAP values across all folds
        mean_shap = np.abs(np.mean([np.mean(np.abs(shap), axis=0) for shap in shap_values], axis=0))
        top_shap_idx = np.argsort(mean_shap)[-top_n:]
        
        # Create boxplot of SHAP values for top features
        shap_data = []
        for shap in shap_values:
            shap_data.append(pd.DataFrame(
                np.abs(shap)[:, top_shap_idx],
                columns=[feature_names[i] for i in top_shap_idx]
            ))
        
        shap_df = pd.concat(shap_data)
        sns.boxplot(data=shap_df, orient='h', ax=ax2)
        ax2.set_title('Distribution of SHAP Values (Top Features)')
        ax2.set_xlabel('Absolute SHAP Value')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot
        plot_file = os.path.join(self.results_dir, "feature_importance_plot.png")
        plt.savefig(plot_file)
        self.logger.info(f"Saved feature importance plot to {plot_file}")
        
        # Print feature importance table
        self.logger.info("\nTop Features by Importance:")
        self.logger.info("=" * 80)
        self.logger.info(f"{'Feature':<50} {'Mean Importance':<15} {'Std Importance':<15}")
        self.logger.info("-" * 80)
        for feature, row in top_features.iterrows():
            self.logger.info(f"{feature:<50} {row['mean_importance']:<15.4f} {row['std_importance']:<15.4f}")
        
        plt.show()
    
    def create_summary(self, results: Dict[str, Any], target_col: str) -> None:
        """
        Create a summary of the feature selection results.
        
        Args:
            results: Dictionary containing feature selection results
            target_col: Name of the target column
        """
        if not results:
            self.logger.warning("No results to create summary for")
            return
            
        selected_features = results['selected_features']
        importance_scores = results['importance_scores']
        
        # Create summary file
        summary_file = os.path.join(self.results_dir, f"feature_selection_summary_{target_col}.txt")
        
        with open(summary_file, 'w') as f:
            f.write("Feature Selection Summary\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Target Column: {target_col}\n")
            f.write(f"Number of Selected Features: {len(selected_features)}\n")
            f.write(f"Feature Selection Method: {self.method}\n")
            f.write(f"Threshold: {self.min_threshold}\n")
            f.write(f"Cumulative Threshold: {self.cumulative_threshold}\n")
            f.write(f"Use Cumulative: {self.use_cumulative}\n\n")
            
            # Only write cumulative importance if it exists
            if 'cumulative_importance' in results:
                f.write(f"Selected {len(selected_features)} features with cumulative importance: {results['cumulative_importance']:.4f}\n\n")
            else:
                f.write(f"Selected {len(selected_features)} features\n\n")
            
            f.write("Top 20 Features by Importance:\n")
            f.write("-" * 50 + "\n")
            
            # Sort features by importance
            sorted_features = importance_scores.sort_values('mean_importance', ascending=False)
            for i, (feature, row) in enumerate(sorted_features.head(20).iterrows()):
                f.write(f"{i+1}. {feature}: {row['mean_importance']:.4f}\n")
            
            f.write("\nSelected Features:\n")
            f.write("-" * 50 + "\n")
            for feature in selected_features:
                f.write(f"- {feature}\n")
            
            f.write("\nFeature Importance Statistics:\n")
            f.write("-" * 50 + "\n")
            f.write(f"Mean Importance: {importance_scores['mean_importance'].mean():.4f}\n")
            f.write(f"Max Importance: {importance_scores['mean_importance'].max():.4f}\n")
            f.write(f"Min Importance: {importance_scores['mean_importance'].min():.4f}\n")
        
        self.logger.info(f"Created summary file: {summary_file}")
    
    @classmethod
    def from_config(cls, config: Dict[str, Any], run_id: Optional[str] = None) -> 'FeatureSelector':
        """
        Create a FeatureSelector instance from a configuration dictionary.
        
        Args:
            config: Configuration dictionary
            run_id: Optional custom run identifier
            
        Returns:
            FeatureSelector instance
        """
        # Create feature selection config
        feature_selection_config = {
            'data': {
                'base_dir': config['data']['base_dir'],
                'features': {
                    'fundamentals': {
                        'base_dir': os.path.join(config['data']['base_dir'], 'features', 'fundamentals')
                    }
                }
            },
            'feature_selection': config['feature_selection'],
            'output': {
                'results_dir': os.path.join(config['data']['base_dir'], 'results', 'feature_selection')
            }
        }
        
        # Create logger
        logger = logging.getLogger('FeatureSelector')
        logger.setLevel(logging.INFO)
        
        # Create console handler if not exists
        if not logger.handlers:
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            ch.setFormatter(formatter)
            logger.addHandler(ch)
        
        return cls(feature_selection_config, run_id=run_id, logger=logger)

    # --- Place _load_data_for_tuning definition here, before it's called --- 
    def _load_data_for_tuning(self, target_date_str: str, max_days: int = 20, target_col: str = 'PE_RATIO_RATIO_SIGNED_LOG') -> Optional[Tuple[pd.DataFrame, pd.Series, List[str]]]:
        self.logger.info(f"Loading data for tuning up to {target_date_str}, max {max_days} days.")
        target_date = pd.to_datetime(target_date_str)
        all_features_list = []; all_target_list = []; feature_names_set = set(); loaded_days_count = 0
        for i in range(max_days):
            current_date = target_date - pd.Timedelta(days=i); date_str = current_date.strftime('%Y-%m-%d')
            self.logger.debug(f"Attempting load for {date_str}")
            try:
                daily_data = self._load_processed_data(date_str)
                if daily_data is None or daily_data.empty: continue
                if target_col not in daily_data.columns: self.logger.warning(f"Target '{target_col}' not found for {date_str}. Skipping."); continue
                if 'PIT_DATE' in daily_data.columns:
                    daily_data['PIT_DATE'] = pd.to_datetime(daily_data['PIT_DATE'])
                    daily_data = daily_data.set_index('PIT_DATE')
                elif not isinstance(daily_data.index, pd.DatetimeIndex):
                    try: daily_data.index = pd.to_datetime(daily_data.index)
                    except Exception: self.logger.error(f"Could not set DatetimeIndex for {date_str}. Skipping."); continue
                y_daily = daily_data[target_col]; X_daily = daily_data.drop(columns=[target_col], errors='ignore')
                all_features_list.append(X_daily); all_target_list.append(y_daily)
                feature_names_set.update(X_daily.columns); loaded_days_count += 1
            except Exception as e: self.logger.error(f"Error loading/processing {date_str}: {e}", exc_info=True)
        if not all_features_list: self.logger.error("No data loaded for tuning."); return None
        final_feature_names = sorted(list(feature_names_set))
        combined_features_list = [df.reindex(columns=final_feature_names) for df in all_features_list]
        X_combined = pd.concat(combined_features_list[::-1], axis=0); y_combined = pd.concat(all_target_list[::-1], axis=0)
        target_na_mask = y_combined.isna()
        if target_na_mask.any(): X_combined = X_combined[~target_na_mask]; y_combined = y_combined[~target_na_mask]
        if X_combined.empty: self.logger.error("Combined tuning data empty."); return None
        # Ensure y_combined is Series
        if not isinstance(y_combined, pd.Series):
             self.logger.error("y_combined is not a Series after processing.")
             return None
        y_combined = y_combined.loc[X_combined.index] 
        X_combined = X_combined.sort_index(); y_combined = y_combined.sort_index()
        final_feature_names_str = [str(f) for f in final_feature_names]; X_combined.columns = final_feature_names_str
        self.logger.info(f"Loaded tuning data ({loaded_days_count} days). Shape: X={X_combined.shape}")
        # Final type assertion before return
        assert isinstance(y_combined, pd.Series)
        return X_combined, y_combined, final_feature_names_str
        
    # --- Other helper methods like _load_processed_data, _train_xgboost etc. --- 
    # ... (Definitions as before) ...

    # --- Tuning Methods --- 
    def tune_for_stable_features(self, target_date_str: str, k_features: int = 50, n_trials: int = 50,
                                 n_splits: int = 5, early_stopping_rounds: int = 20, max_days_lookback: int = 20,
                                 target_col: str = 'PE_RATIO_RATIO_SIGNED_LOG', study_name_prefix: str = "xgboost_stability") -> Optional[Tuple[List[str], Dict[str, Any]]]:
        """
        Tune XGBoost hyperparameters focusing on feature stability across time periods.
        Returns stable features and optimal hyperparameters for feature selection.
        
        Always includes sector features in the returned feature set regardless of their importance.
        """
        self.logger.info(f"Starting hyperparameter tuning for feature stability up to {target_date_str}.")
        
        # Load data for tuning
        data_load_result = self._load_data_for_tuning(target_date_str, max_days_lookback, target_col)
        if data_load_result is None:
            self.logger.error("Failed to load data for stability tuning.")
            return None
            
        X_data, y_data, feature_names = data_load_result
        
        # Identify sector features that should always be included
        sector_pattern = r'^(SECTOR_|sector_)'
        sector_features = [col for col in X_data.columns if re.match(sector_pattern, col)]
        
        if sector_features:
            self.logger.info(f"Found {len(sector_features)} sector features that will be always included: {sector_features}")
            
            # Ensure sector features are properly set as categorical
            for feature in sector_features:
                if X_data[feature].dtype.name != 'category':
                    self.logger.info(f"Converting sector feature '{feature}' to category dtype")
                    try:
                        # Handle missing values before conversion
                        if X_data[feature].isna().any():
                            X_data[feature] = X_data[feature].fillna(f"Missing_{feature}")
                        X_data[feature] = X_data[feature].astype('category')
                    except Exception as e:
                        self.logger.warning(f"Could not convert sector feature '{feature}' to category: {e}")
        else:
            self.logger.warning("No sector features found in the data. Sector features will not be forced into selection.")
        
        # Create Optuna study for hyperparameter tuning
        study_name = f"{study_name_prefix}_{target_date_str}"
        self.logger.info(f"Creating Optuna study '{study_name}' for stability tuning.")
        
        study = optuna.create_study(
            direction="maximize",  # Maximize stability
            study_name=study_name,
            load_if_exists=True
        )
        
        # Define stability objective function
        def objective_stability(trial):
            # Define hyperparameter search space
            params = {
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
                'n_estimators': 1000,  # Will use early stopping
                'learning_rate': trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                'max_depth': trial.suggest_int("max_depth", 3, 10),
                'subsample': trial.suggest_float("subsample", 0.6, 1.0),
                'colsample_bytree': trial.suggest_float("colsample_bytree", 0.6, 1.0),
                'gamma': trial.suggest_float("gamma", 0, 5),
                'reg_alpha': trial.suggest_float("reg_alpha", 1e-5, 1.0, log=True),
                'reg_lambda': trial.suggest_float("reg_lambda", 1e-5, 1.0, log=True),
                'min_child_weight': trial.suggest_int("min_child_weight", 1, 10),
                'verbosity': 0,
                'seed': 42,
                'enable_categorical': True
            }
            
            # Use TimeSeriesSplit for cross-validation
            tscv = TimeSeriesSplit(n_splits=n_splits)
            
            # For each fold, track selected features
            fold_selected_features = []
            fold_validation_rmses = []
            
            for fold, (train_idx, val_idx) in enumerate(tscv.split(X_data)):
                X_train, X_val = X_data.iloc[train_idx], X_data.iloc[val_idx]
                y_train, y_val = y_data.iloc[train_idx], y_data.iloc[val_idx]
                
                # Train model with early stopping
                model = xgb.XGBRegressor(**params)
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    early_stopping_rounds=early_stopping_rounds,
                    verbose=False
                )
                
                # Get validation RMSE
                y_pred = model.predict(X_val)
                rmse = np.sqrt(mean_squared_error(y_val, y_pred))
                fold_validation_rmses.append(rmse)
                
                # Get top k features by importance
                importance = model.feature_importances_
                feature_indices = np.argsort(importance)[::-1][:k_features]
                top_features = [X_data.columns[i] for i in feature_indices]
                fold_selected_features.append(set(top_features))
            
            # Calculate average pairwise Jaccard index (stability metric)
            jaccard_indices = []
            n_folds = len(fold_selected_features)
            
            for i in range(n_folds):
                for j in range(i+1, n_folds):
                    set_i = fold_selected_features[i]
                    set_j = fold_selected_features[j]
                    jaccard = len(set_i.intersection(set_j)) / len(set_i.union(set_j))
                    jaccard_indices.append(jaccard)
            
            avg_jaccard = np.mean(jaccard_indices) if jaccard_indices else 0
            avg_rmse = np.mean(fold_validation_rmses)
            
            # Store validation metrics
            trial.set_user_attr("avg_rmse", float(avg_rmse))
            
            self.logger.info(f"Trial {trial.number}: Avg Jaccard={avg_jaccard:.4f}, Avg RMSE={avg_rmse:.4f}")
            return avg_jaccard
        
        # Run Optuna optimization
        try:
            study.optimize(objective_stability, n_trials=n_trials, timeout=None, gc_after_trial=True)
        except Exception as e:
            self.logger.error(f"Optuna stability study failed: {e}", exc_info=True)
            return None
            
        # Get best trial and parameters
        best_params = study.best_trial.params
        self.logger.info(f"Best stability score: {study.best_value:.4f}")
        self.logger.info(f"Best parameters: {best_params}")
        
        # Train final model with best parameters and extract robust feature set
        try:
            # Combine best params with other required parameters
            final_params = {
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
                'n_estimators': 1000,  # We'll use the learned n_estimators later
                'verbosity': 0,
                'seed': 42,
                'enable_categorical': True,
                **best_params  # Add the tuned parameters
            }
            
            # Train on all data
            final_model = xgb.XGBRegressor(**final_params)
            final_model.fit(X_data, y_data)
            
            # Extract top k features by importance (minus the number of sector features to make room)
            importance = final_model.feature_importances_
            
            # Calculate how many non-sector features to include
            num_non_sector_features = min(k_features - len(sector_features), len(X_data.columns) - len(sector_features))
            
            # Get indices of non-sector features by importance
            non_sector_indices = []
            feature_importances_with_index = list(zip(importance, range(len(importance))))
            feature_importances_with_index.sort(reverse=True)  # Sort by importance (descending)
            
            for imp, idx in feature_importances_with_index:
                feature_name = X_data.columns[idx]
                if not re.match(sector_pattern, feature_name) and len(non_sector_indices) < num_non_sector_features:
                    non_sector_indices.append(idx)
            
            # Get feature names for selected non-sector features
            selected_non_sector_features = [X_data.columns[i] for i in non_sector_indices]
            
            # Combine with sector features
            robust_feature_set = selected_non_sector_features + sector_features
            
            # Log information about the selected features
            self.logger.info(f"Selected {len(robust_feature_set)} stable features:")
            self.logger.info(f"  - {len(selected_non_sector_features)} features based on importance")
            self.logger.info(f"  - {len(sector_features)} sector features always included")
            
            return robust_feature_set, best_params
            
        except Exception as e:
            self.logger.error(f"Error in final stability model training: {e}", exc_info=True)
            return None

    def _objective_rmse(self, trial: optuna.trial.Trial, X: pd.DataFrame, y: pd.Series, 
                       stable_features: List[str], n_splits: int, 
                       early_stopping_rounds: int) -> float:
        """Optuna objective function to minimize RMSE using stable features."""
        self.logger.info(f"--- Starting RMSE Trial {trial.number} using {len(stable_features)} stable features ---")
        
        # Filter X to only include stable features
        if not all(f in X.columns for f in stable_features):
            missing = set(stable_features) - set(X.columns)
            self.logger.error(f"Trial {trial.number}: Stable features missing from input data: {missing}.")
            return float('inf')
        
        X_stable = X[stable_features]
        
        # Define hyperparameter search space
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'n_estimators': 1000,
            'learning_rate': trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int("max_depth", 3, 10),
            'subsample': trial.suggest_float("subsample", 0.6, 1.0),
            'colsample_bytree': trial.suggest_float("colsample_bytree", 0.6, 1.0),
            'gamma': trial.suggest_float("gamma", 0, 5),
            'reg_alpha': trial.suggest_float("reg_alpha", 1e-5, 1.0, log=True),
            'reg_lambda': trial.suggest_float("reg_lambda", 1e-5, 1.0, log=True),
            'min_child_weight': trial.suggest_int("min_child_weight", 1, 10),
            'verbosity': 0,
            'seed': 42,
            'enable_categorical': True
        }
        
        # TimeSeriesSplit Cross-Validation
        tscv = TimeSeriesSplit(n_splits=n_splits)
        validation_rmses = []
        best_iterations = []
        
        for fold, (train_index, val_index) in enumerate(tscv.split(X_stable)):
            self.logger.debug(f"RMSE Trial {trial.number}, Fold {fold+1}/{n_splits} starting...")
            X_train, X_val = X_stable.iloc[train_index], X_stable.iloc[val_index]
            y_train, y_val = y.iloc[train_index], y.iloc[val_index]
            
            # Align indices just in case
            if not X_train.index.equals(y_train.index):
                y_train = y_train.loc[X_train.index]
            if not X_val.index.equals(y_val.index):
                y_val = y_val.loc[X_val.index]
            
            # Train XGBoost with early stopping
            model = xgb.XGBRegressor(**params)
            try:
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    early_stopping_rounds=early_stopping_rounds,
                    verbose=False
                )
                
                # Get validation RMSE and best iteration
                y_pred = model.predict(X_val)
                rmse = np.sqrt(mean_squared_error(y_val, y_pred))
                validation_rmses.append(rmse)
                best_iterations.append(model.best_iteration)
                
                self.logger.debug(f"RMSE Trial {trial.number}, Fold {fold+1} fitted. Best Iter: {model.best_iteration}, Val RMSE: {rmse:.4f}")
                
            except Exception as e:
                self.logger.error(f"Error during RMSE model fitting Trial {trial.number}, Fold {fold+1}: {e}", exc_info=True)
                return float('inf')
        
        # Calculate average RMSE & best iteration
        if not validation_rmses or not best_iterations:
            self.logger.warning(f"RMSE Trial {trial.number}: No valid results collected.")
            return float('inf')
        
        average_rmse = float(np.mean(validation_rmses))
        average_best_iteration = float(np.mean(best_iterations))
        
        # Store average best iteration for later use
        trial.set_user_attr("avg_best_iteration", average_best_iteration)
        
        self.logger.info(f"--- Finished RMSE Trial {trial.number} --- Avg Val RMSE: {average_rmse:.4f}, Avg Best Iter: {average_best_iteration:.1f}")
        return average_rmse

    def tune_daily_predictor_params(self, target_date_str: str, stable_features: List[str], 
                                   n_trials: int = 50, n_splits: int = 5, 
                                   early_stopping_rounds: int = 15, max_days_lookback: int = 20,
                                   target_col: str = 'PE_RATIO_RATIO_SIGNED_LOG',
                                   study_name_prefix: str = "xgboost_daily_rmse",
                                   max_cap_daily_estimators: int = 150) -> Optional[Tuple[Dict[str, Any], int]]:
        """
        Tunes XGBoost hyperparameters using Optuna to minimize RMSE on recent data,
        using stable features. Returns best params AND calculated n_estimators for daily use.
        """
        self.logger.info(f"Starting hyperparameter tuning for daily predictor RMSE using {len(stable_features)} features.")
        
        if not stable_features:
            self.logger.error("Cannot tune daily predictor: No stable features provided.")
            return None
        
        # Load data for tuning
        data_load_result = self._load_data_for_tuning(target_date_str, max_days_lookback, target_col)
        if data_load_result is None:
            self.logger.error("Failed to load data for daily predictor tuning.")
            return None
            
        X_data, y_data, _ = data_load_result
        
        # Ensure all stable features are present in the loaded data
        if not all(f in X_data.columns for f in stable_features):
            missing = set(stable_features) - set(X_data.columns)
            self.logger.error(f"Stable features missing from data: {missing}. Cannot tune.")
            return None
        
        # Create Optuna study for RMSE minimization
        study_name = f"{study_name_prefix}_{target_date_str}"
        self.logger.info(f"Creating Optuna study '{study_name}' for RMSE minimization.")
        
        study = optuna.create_study(
            direction="minimize",  # Minimize RMSE
            study_name=study_name,
            load_if_exists=True
        )
        
        # Adjust n_splits if the time window is short
        actual_n_splits = min(n_splits, max(2, max_days_lookback - 1))
        if actual_n_splits != n_splits:
            self.logger.warning(f"Adjusting n_splits from {n_splits} to {actual_n_splits} due to short lookback.")
        
        # Define objective function
        objective_func = lambda trial: self._objective_rmse(
            trial, X_data, y_data, stable_features,
            n_splits=actual_n_splits,
            early_stopping_rounds=early_stopping_rounds
        )
        
        # Run Optuna optimization
        try:
            study.optimize(objective_func, n_trials=n_trials, timeout=None, gc_after_trial=True)
        except Exception as e:
            self.logger.error(f"Optuna RMSE study failed: {e}", exc_info=True)
            return None
        
        # Get best trial and parameters
        best_params = study.best_trial.params
        self.logger.info(f"Best RMSE: {study.best_value:.4f}")
        self.logger.info(f"Best parameters: {best_params}")
        
        # Calculate optimal n_estimators for daily model
        n_estimators_20day_avg = study.best_trial.user_attrs.get("avg_best_iteration")
        
        if n_estimators_20day_avg is None or n_estimators_20day_avg <= 0:
            self.logger.warning("Could not retrieve valid avg_best_iteration. Using max cap.")
            n_estimators_daily = max_cap_daily_estimators
        else:
            # Scale down estimators for daily prediction
            scaling_factor = 1.0 / np.sqrt(max_days_lookback)
            n_estimators_daily_raw = n_estimators_20day_avg * scaling_factor
            n_estimators_daily = min(int(round(n_estimators_daily_raw)), max_cap_daily_estimators)
            n_estimators_daily = max(1, n_estimators_daily)
            
            self.logger.info(f"Calculated n_estimators: {n_estimators_20day_avg:.1f} → {n_estimators_daily}")
        
        return best_params, n_estimators_daily

# Ensure class definition ends correctly 
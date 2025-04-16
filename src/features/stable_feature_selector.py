import os
import pandas as pd
import numpy as np
import logging
import lightgbm as lgb
import optuna
import json
from typing import Dict, List, Tuple, Optional, Any, Set, Union, Callable
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta
import scipy.stats
import itertools
import joblib # For potentially saving scaler or model if needed later
import pickle
import time
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn.feature_selection import VarianceThreshold
from xgboost.callback import EarlyStopping
from sklearn.ensemble import RandomForestRegressor
from optuna.trial import Trial
import re

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StableFeatureSelector:
    """
    Selects stable features using Optuna, XGBoost, and Spearman rank correlation.

    Optimizes XGBoost hyperparameters to maximize feature importance rank stability
    across time-series folds, subject to a minimum RMSE performance threshold.
    Handles compulsory features and saves the selected stable feature set.
    """

    # Add class-level constants for hyperparameter search space
    HYPERPARAM_SEARCH_SPACE = {
        'num_leaves': {'min': 31, 'max': 127, 'type': 'int'},
        'learning_rate': {'min': 0.005, 'max': 0.1, 'type': 'float', 'log': True},
        'feature_fraction': {'min': 0.7, 'max': 1.0, 'type': 'float'},
        'bagging_fraction': {'min': 0.7, 'max': 1.0, 'type': 'float'},
        'bagging_freq': {'min': 1, 'max': 10, 'type': 'int'},
        'min_child_samples': {'min': 20, 'max': 100, 'type': 'int'},
        'lambda_l1': {'min': 1e-8, 'max': 10.0, 'type': 'float', 'log': True},
        'lambda_l2': {'min': 1e-8, 'max': 10.0, 'type': 'float', 'log': True},
        'max_depth': {'min': 3, 'max': 12, 'type': 'int'},
        'min_data_in_leaf': {'min': 20, 'max': 100, 'type': 'int'}
    }

    def __init__(
        self,
        k_features: int = 50,
        lookback_days: int = 20,
        compulsory_features: Optional[List[str]] = None,
        system_config: Optional[Dict[str, Any]] = None,
        n_splits: int = 3,
        n_trials: int = 50,
        optuna_n_jobs: int = 15,
        random_seed: int = 42,
        results_base_dir: Optional[str] = None,
        early_stopping_rounds: int = 50,
        rmse_threshold: float = 2.0,
        stability_candidate_pool: int = 250
    ):
        """Initialize the StableFeatureSelector.
        
        Args:
            k_features: Number of features to select
            lookback_days: Number of days to look back for stability analysis
            compulsory_features: List of features that must be included in the selection
            system_config: System configuration dictionary with cpu_cores, memory_gb, and num_threads
            n_splits: Number of cross-validation splits
            n_trials: Number of Optuna trials to run
            optuna_n_jobs: Number of parallel jobs for Optuna
            random_seed: Random seed for reproducibility
            results_base_dir: Base directory for saving results
            early_stopping_rounds: Number of rounds for early stopping
            rmse_threshold: RMSE threshold for early stopping
            stability_candidate_pool: Number of top features to consider for stability
        """
        self.k_features = k_features
        self.lookback_days = lookback_days
        self.compulsory_features = compulsory_features
        self.system_config = system_config or {
            'cpu_cores': optuna_n_jobs,  # Use optuna_n_jobs for cpu_cores
            'memory_gb': 64,  # Default to 64GB RAM
            'num_threads': optuna_n_jobs  # Use optuna_n_jobs for num_threads
        }
        self.n_splits = n_splits
        self.n_trials = n_trials
        self.optuna_n_jobs = optuna_n_jobs
        self.random_seed = random_seed
        self.results_base_dir = results_base_dir
        self.early_stopping_rounds = early_stopping_rounds
        self.rmse_threshold = rmse_threshold
        self.stability_candidate_pool = stability_candidate_pool
        
        self.best_rmse = None
        self.best_iteration = None
        self.best_importance = None
        
        # Initialize other attributes
        self.X = None
        self.y = None
        self.feature_importance_scores = None
        self.cv_results = None
        self.best_params = None
        self.stability_metrics = None

        # Internal state set during execution
        self.study: Optional[optuna.Study] = None
        self.feature_names: List[str] = []
        self.compulsory_features_present: List[str] = []
        self.target_date: Optional[pd.Timestamp] = None
        self.target_column: Optional[str] = None
        self.numeric_feature_names: List[str] = []
        self.categorical_feature_names: List[str] = [] # Or non-numeric
        self.cv_splitter = TimeSeriesSplit(n_splits=n_splits)

        logger.info("StableFeatureSelector initialized.")
        logger.info(f"Parameters: k_features={k_features}, lookback_days={lookback_days}")
        logger.info(f"Compulsory features configured: {compulsory_features}")
        logger.info(f"CV splits: {n_splits}, Trials: {n_trials}, Optuna jobs: {optuna_n_jobs}")
        logger.info(f"Early stopping rounds: {early_stopping_rounds}, RMSE threshold: {rmse_threshold}")
        logger.info(f"Stability candidate pool: {stability_candidate_pool}")

    def _load_and_prepare_data(self, panel_data_base_path: str, target_date_dt: pd.Timestamp, target_column_name: str) -> bool:
        """Loads and prepares data from a partitioned parquet structure for the lookback period."""
        logger.info(f"Loading data from base path: {panel_data_base_path}")
        date_col = "PIT_DATE"  # Use the correct date column name
        try:
            # Determine date range
            # TODO: Add proper trading day calendar logic if needed for precise lookback
            start_date = target_date_dt - timedelta(days=self.lookback_days * (7/5)) # Heuristic for calendar days
            start_date = start_date.normalize()
            target_date_dt = target_date_dt.normalize()
            self.target_date = target_date_dt # Store for results path

            logger.info(f"Required date range: {start_date.date()} to {target_date_dt.date()}")

            # Generate expected file paths within the date range
            required_files = []
            current_date = start_date
            while current_date <= target_date_dt:
                year = current_date.year
                month = f"{current_date.month:02d}"
                day_str = current_date.strftime('%Y-%m-%d')
                file_path = Path(panel_data_base_path) / f"year={year}" / f"month={month}" / f"data_{day_str}.pq"
                if file_path.exists():
                     required_files.append(file_path)
                else:
                     logger.debug(f"Optional file not found, skipping: {file_path}")
                current_date += timedelta(days=1)

            if not required_files:
                logger.error(f"No data files found within the range {start_date.date()} to {target_date_dt.date()} in {panel_data_base_path}")
                return False

            logger.info(f"Found {len(required_files)} potential data files to load.")

            # Load data using pd.concat
            df_list = [pd.read_parquet(f) for f in required_files]
            data = pd.concat(df_list, ignore_index=True)
            logger.info(f"Loaded and concatenated data shape: {data.shape}")

            # --- Resume previous preparation steps ---

            # Ensure **PIT_DATE** column is datetime and filter again
            if date_col not in data.columns:
                 logger.error(f"'{date_col}' column not found in loaded panel data.")
                 return False
            data[date_col] = pd.to_datetime(data[date_col])
            data = data[(data[date_col] >= start_date) & (data[date_col] <= target_date_dt)].copy()

            if data.empty:
                logger.error(f"No data remains after precise date filtering {start_date.date()} to {target_date_dt.date()}.")
                return False
            logger.info(f"Lookback data shape after precise filtering: {data.shape}")

            # Store dates for CV splits
            self.dates = data[date_col].copy()
            logger.info(f"Stored dates from {self.dates.min()} to {self.dates.max()}")

            # Verify target column
            if target_column_name not in data.columns:
                logger.error(f"Target column '{target_column_name}' not found.")
                return False
            if data[target_column_name].isnull().any():
                logger.warning(f"NaNs found in target column '{target_column_name}'. Dropping rows with NaN target.")
                data.dropna(subset=[target_column_name], inplace=True)
                if data.empty:
                    logger.error("All rows dropped due to NaN target. Cannot proceed.")
                    return False

            # --- Type Identification --- 
            # Use date_col from init or define here
            date_col = "PIT_DATE"
            id_col = "ID"
            potential_feature_cols = [col for col in data.columns if col not in [id_col, date_col, target_column_name]]
            temp_X = data[potential_feature_cols]
            
            self.numeric_feature_names = temp_X.select_dtypes(include=np.number).columns.tolist()
            self.categorical_feature_names = temp_X.select_dtypes(exclude=np.number).columns.tolist()
            
            logger.info(f"Identified {len(self.numeric_feature_names)} numeric features.")
            logger.info(f"Identified {len(self.categorical_feature_names)} non-numeric features: {self.categorical_feature_names}")
            
            # Update the main feature list used for X
            self.feature_names = self.numeric_feature_names + self.categorical_feature_names 
            
            # Verify compulsory features against ALL identified features
            self.compulsory_features_present = []
            for feature in self.compulsory_features:
                if feature in self.feature_names:
                    # Check for NaNs only if it's numeric (imputer handles) or categorical (might need different handling)
                    if data[feature].isnull().any():
                        logger.warning(f"Compulsory feature '{feature}' (type: {data[feature].dtype}) contains NaNs.")
                    self.compulsory_features_present.append(feature)
                else:
                    logger.warning(f"Configured compulsory feature '{feature}' not found in data columns. Skipping.")
            
            num_compulsory = len(self.compulsory_features_present)
            logger.info(f"Found {num_compulsory} compulsory features present in data: {self.compulsory_features_present}")

            if self.k_features < num_compulsory:
                 logger.error(f"k_features ({self.k_features}) is less than the number of present compulsory features ({num_compulsory}).")
                 return False

            # Separate features and target
            self.X = data[self.feature_names].copy()
            self.y = data[target_column_name].copy()

            # Check feature types
            numeric_cols = self.X.select_dtypes(include=np.number).columns.tolist()
            if len(numeric_cols) != len(self.feature_names):
                 non_numeric = list(set(self.feature_names) - set(numeric_cols))
                 logger.warning(f"Non-numeric columns detected: {non_numeric}. Ensure these are handled (e.g., categorical).")

            logger.info("Data loading and preparation complete.")
            return True

        except FileNotFoundError:
             logger.error(f"Panel data base path not found or inaccessible: {panel_data_base_path}")
             return False
        except Exception as e:
            logger.error(f"Error during data loading/preparation from partitions: {e}", exc_info=True)
            return False

    def _pre_filter_features(
        self,
        df: pd.DataFrame,
        missing_threshold: float = 0.2,
        variance_threshold: float = 0.05,
        univariate_corr_threshold: float = 0.01
    ) -> tuple[list[str], dict[str, list[str]]]:
        """Pre-filter features based on basic criteria before optimization."""
        logger.info("Starting feature pre-filtering...")
        
        # Identify numeric columns (excluding target and ID columns)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [col for col in numeric_cols 
                       if col not in ['PERIOD_END_DATE', 'TICKER']]
        
        dropped_log = {
            'missing_values': [],
            'variance': [],
            'univariate_correlation': []
        }
        
        # 1. Missing values filter
        missing_ratios = df[numeric_cols].isnull().mean()
        to_drop = missing_ratios[missing_ratios > missing_threshold].index.tolist()
        dropped_log['missing_values'] = to_drop
        valid_cols = [col for col in numeric_cols if col not in to_drop]
        logger.info(f"After missing values filter: {len(valid_cols)} features")
        
        # 2. Variance filter
        variances = df[valid_cols].var()
        to_drop = variances[variances < variance_threshold].index.tolist()
        dropped_log['variance'] = to_drop
        valid_cols = [col for col in valid_cols if col not in to_drop]
        logger.info(f"After variance filter: {len(valid_cols)} features")
        
        # 3. Univariate correlation with target filter
        if len(valid_cols) > 0 and self.y is not None:
            target_correlations = df[valid_cols].corrwith(self.y).abs()
            to_drop = target_correlations[target_correlations < univariate_corr_threshold].index.tolist()
            dropped_log['univariate_correlation'] = to_drop
            valid_cols = [col for col in valid_cols if col not in to_drop]
            logger.info(f"After univariate correlation filter: {len(valid_cols)} features")
        
        # Ensure compulsory features are kept
        compulsory_present = [f for f in self.compulsory_features if f in valid_cols]
        if compulsory_present:
            logger.info(f"Keeping {len(compulsory_present)} compulsory features: {compulsory_present}")
            valid_cols = list(set(valid_cols + compulsory_present))
        
        return valid_cols, dropped_log

    def _create_study(self) -> optuna.Study:
        """Create and configure the Optuna study."""
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=2,
                interval_steps=1
            )
        )
        return study

    def _objective_stability_filtered(self, trial: Trial, n_splits: int = 3, random_state: Optional[int] = None) -> float:
        """
        Objective function for Optuna that calculates both RMSE and stability.
        """
        if self.X is None or self.y is None:
            raise ValueError("X and y must be set before calling optimize")

        # Define parameter search space for LightGBM
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'verbose': -1,
            'num_threads': self.system_config.get('num_threads', 12),
            'max_depth': trial.suggest_int('max_depth', self.HYPERPARAM_SEARCH_SPACE['max_depth']['min'], self.HYPERPARAM_SEARCH_SPACE['max_depth']['max']),
            'num_leaves': trial.suggest_int('num_leaves', self.HYPERPARAM_SEARCH_SPACE['num_leaves']['min'], self.HYPERPARAM_SEARCH_SPACE['num_leaves']['max']),
            'learning_rate': trial.suggest_float('learning_rate', self.HYPERPARAM_SEARCH_SPACE['learning_rate']['min'], self.HYPERPARAM_SEARCH_SPACE['learning_rate']['max'], log=self.HYPERPARAM_SEARCH_SPACE['learning_rate'].get('log', False)),
            'feature_fraction': trial.suggest_float('feature_fraction', self.HYPERPARAM_SEARCH_SPACE['feature_fraction']['min'], self.HYPERPARAM_SEARCH_SPACE['feature_fraction']['max']),
            'bagging_fraction': trial.suggest_float('bagging_fraction', self.HYPERPARAM_SEARCH_SPACE['bagging_fraction']['min'], self.HYPERPARAM_SEARCH_SPACE['bagging_fraction']['max']),
            'bagging_freq': trial.suggest_int('bagging_freq', self.HYPERPARAM_SEARCH_SPACE['bagging_freq']['min'], self.HYPERPARAM_SEARCH_SPACE['bagging_freq']['max']),
            'lambda_l1': trial.suggest_float('lambda_l1', self.HYPERPARAM_SEARCH_SPACE['lambda_l1']['min'], self.HYPERPARAM_SEARCH_SPACE['lambda_l1']['max'], log=self.HYPERPARAM_SEARCH_SPACE['lambda_l1'].get('log', False)),
            'lambda_l2': trial.suggest_float('lambda_l2', self.HYPERPARAM_SEARCH_SPACE['lambda_l2']['min'], self.HYPERPARAM_SEARCH_SPACE['lambda_l2']['max'], log=self.HYPERPARAM_SEARCH_SPACE['lambda_l2'].get('log', False)),
            'max_bin': 255,
            'max_bin_by_feature': [255] * len(self.X.columns),  # Set max_bin for each feature
            'histogram_pool_size': 64 * 1024 * 1024,  # 64MB per thread
            'gpu_platform_id': -1,  # Disable GPU
            'gpu_device_id': -1,    # Disable GPU
            'gpu_use_dp': False,    # Disable GPU
            'force_col_wise': True,  # Force column-wise histogram building
            'force_row_wise': False, # Disable row-wise histogram building
            'categorical_feature': self.categorical_feature_names
        }
        
        # Create time-series aware CV splits
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        # Initialize lists to store results
        rmse_scores = []
        best_num_boost_rounds = []
        fold_importances = []
        
        # Log the split dates if available (only once)
        if hasattr(self, 'dates') and self.dates is not None and trial.number == 0:
            logger.info("Time Series CV Splits:")
            for fold, (train_idx, val_idx) in enumerate(tscv.split(self.X)):
                train_dates = self.dates.iloc[train_idx]
                val_dates = self.dates.iloc[val_idx]
                logger.info(f"Fold {fold}: Train={train_dates.min()} to {train_dates.max()}, "
                           f"Validate={val_dates.min()} to {val_dates.max()}")
        
        # Perform cross-validation
        for fold, (train_idx, val_idx) in enumerate(tscv.split(self.X)):
            X_train, X_val = self.X.iloc[train_idx], self.X.iloc[val_idx]
            y_train, y_val = self.y.iloc[train_idx], self.y.iloc[val_idx]
            
            try:
                # Create LightGBM datasets with categorical features
                categorical_features = self.categorical_feature_names if hasattr(self, 'categorical_feature_names') else []
                train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_features)
                val_data = lgb.Dataset(X_val, label=y_val, reference=train_data, categorical_feature=categorical_features)
                
                # Train model with early stopping
                model = lgb.train(
                    params,
                    train_data,
                    num_boost_round=2000,
                    valid_sets=[val_data],
                    callbacks=[
                        lgb.early_stopping(stopping_rounds=20, verbose=False),
                        lgb.log_evaluation(period=100, show_stdv=False)
                    ]
                )
                
                # Get best iteration
                best_iteration = model.best_iteration
                best_num_boost_rounds.append(best_iteration)
                
                # Get predictions and calculate RMSE
                val_pred = model.predict(X_val)
                mse = mean_squared_error(y_val, val_pred)
                fold_rmse = np.sqrt(mse)  # Calculate RMSE manually
                rmse_scores.append(fold_rmse)
                
                # Get feature importances for this fold
                importance = pd.Series(
                    model.feature_importance(importance_type='gain'),
                    index=self.X.columns
                )
                fold_importances.append(importance)
                
                # Early stopping based on RMSE threshold
                if fold_rmse > 2.0:
                    logger.warning(f"Fold {fold} RMSE {fold_rmse:.4f} exceeds threshold 2.0")
                    trial.set_user_attr("early_stopped", True)
                    return float("inf")
                
            except Exception as e:
                logger.error(f"Error in fold {fold}: {str(e)}")
                return float("inf")
        
        # Calculate average RMSE and Spearman correlation
        avg_rmse = np.mean(rmse_scores)
        avg_spearman = self._calculate_average_spearman(fold_importances)
        
        # Store trial information
        trial.set_user_attr("average_rmse", avg_rmse)
        trial.set_user_attr("average_fold_importance", pd.concat(fold_importances, axis=1).mean(axis=1).to_dict())
        trial.set_user_attr("fold_importances", [imp.to_dict() for imp in fold_importances])
        trial.set_user_attr("rmse_scores", rmse_scores)
        trial.set_user_attr("best_iterations", best_num_boost_rounds)
        
        # Log results
        logger.info(f"Trial {trial.number}: Avg Spearman={avg_spearman:.4f}, Avg RMSE={avg_rmse:.4f}")
        return avg_spearman

    def _filter_and_select_best_trial(self, study: optuna.study.Study) -> Optional[optuna.trial.FrozenTrial]:
        """Filters completed trials by RMSE threshold and selects the best trial based on stability."""
        # Get all completed trials
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        
        if not completed_trials:
            logger.warning("No completed trials found")
            return None

        # Extract RMSE values and stability scores
        trial_data = []
        for t in completed_trials:
            if t.value is not None:  # Check if trial has a valid value (stability score)
                trial_data.append({
                    'trial': t,
                    'stability': t.value,
                    'rmse': t.user_attrs.get('average_rmse', float('inf'))
                })
        
        if not trial_data:
            logger.warning("No trials with valid stability scores found")
            return None

        # Sort trials by RMSE (lower is better)
        sorted_by_rmse = sorted(trial_data, key=lambda x: x['rmse'])
        
        # Calculate RMSE threshold (75th percentile)
        rmse_threshold = sorted_by_rmse[int(len(sorted_by_rmse) * 0.75)]['rmse']
        logger.info(f"RMSE threshold (75th percentile): {rmse_threshold:.4f}")
        
        # Filter out trials with RMSE above threshold
        qualified_trials = [t for t in trial_data if t['rmse'] <= rmse_threshold]
        
        if not qualified_trials:
            logger.warning(f"No trials met the RMSE threshold <= {rmse_threshold:.4f}")
            return None
        
        # Sort qualified trials by stability score (higher is better)
        sorted_trials = sorted(qualified_trials, key=lambda x: x['stability'], reverse=True)
        
        # Select the best trial (highest stability among qualified trials)
        best_trial_info = sorted_trials[0]
        best_trial = best_trial_info['trial']
        
        logger.info(f"Selected Best Trial {best_trial.number}:")
        logger.info(f"  Stability Score: {best_trial_info['stability']:.4f}")
        logger.info(f"  Average RMSE: {best_trial_info['rmse']:.4f}")
        
        return best_trial

    def _aggregate_importance_and_select(self, best_trial: optuna.trial.FrozenTrial) -> Tuple[Optional[List[str]], Optional[pd.Series]]:
        """Aggregates importance from the best trial (using avg importance) and selects the final feature set."""
        try:
            # Retrieve the pre-aggregated average importance dictionary
            avg_importance_dict = best_trial.user_attrs.get("average_fold_importance")
            if not avg_importance_dict or not isinstance(avg_importance_dict, dict):
                logger.error(f"Invalid or missing 'average_fold_importance' dict in best trial {best_trial.number}.")
                logger.error(f"Available user attributes: {best_trial.user_attrs.keys()}")
                return None, None

            # Convert dictionary back to Series
            aggregated_importance = pd.Series(avg_importance_dict).sort_values(ascending=False)
            logger.info(f"Total features with importance scores: {len(aggregated_importance)}")

            # Separate compulsory features
            compulsory_present = [f for f in self.compulsory_features if f in aggregated_importance.index]
            compulsory_missing = [f for f in self.compulsory_features if f not in aggregated_importance.index]
            
            if compulsory_missing:
                logger.warning(f"Compulsory features missing from importance data: {compulsory_missing}. They cannot be selected.")
            
            # Get top K candidates excluding compulsory ones that are present
            k_remaining = self.k_features - len(compulsory_present)
            if k_remaining < 0:
                logger.warning(f"More compulsory features ({len(compulsory_present)}) than requested total features ({self.k_features}). Selecting only compulsory features.")
                k_remaining = 0  # Select only compulsory

            # Get top K non-compulsory features
            top_k_candidates = aggregated_importance.drop(compulsory_present, errors='ignore').head(k_remaining)
            logger.info(f"Selected {len(top_k_candidates)} non-compulsory features from top importance scores")

            # Combine compulsory (present) and top K candidates
            final_selected_features = compulsory_present + top_k_candidates.index.tolist()
            logger.info(f"Final selection: {len(compulsory_present)} compulsory + {len(top_k_candidates)} top importance = {len(final_selected_features)} total features")

            # Verify we have enough features
            if len(final_selected_features) < self.k_features:
                logger.warning(f"Selected only {len(final_selected_features)} features, less than requested {self.k_features}")
                logger.warning("This might be due to missing compulsory features or insufficient features with importance scores")
            elif len(final_selected_features) > self.k_features:
                logger.warning(f"Selected {len(final_selected_features)} features, more than requested {self.k_features}")
                final_selected_features = final_selected_features[:self.k_features]  # Trim if needed

            if not final_selected_features:
                logger.error("No features selected after aggregation and filtering")
                return None, None

            # Log the top 10 features by importance
            logger.info("\nTop 10 Features by Importance:")
            for i, (feature, importance) in enumerate(aggregated_importance.head(10).items(), 1):
                logger.info(f"{i}. {feature}: {importance:.4f}")

            return final_selected_features, aggregated_importance

        except Exception as e:
            logger.error(f"Error during feature aggregation and selection: {str(e)}", exc_info=True)
            return None, None

    def _save_results(self, stable_feature_set: List[str], best_trial: optuna.trial.FrozenTrial, date_str: str) -> None:
        """Saves the stable feature selection results to a partitioned structure."""
        # Convert trial parameters to serializable format
        serializable_params = {k: float(v) if isinstance(v, (np.float32, np.float64)) else v 
                             for k, v in best_trial.params.items()}
        
        # Get stability score and RMSE
        stability_score_save = best_trial.value  # Use the trial's value which is the stability score
        rmse_save = best_trial.user_attrs.get('average_rmse', None)
        
        # Get the full importance ranking (top 200)
        avg_importance = pd.Series(best_trial.user_attrs["average_fold_importance"])
        full_importance_ranking = avg_importance.sort_values(ascending=False).head(200)
        
        # Convert target date string to datetime for partitioning
        target_date = pd.to_datetime(date_str)
        year = target_date.year
        month = f"{target_date.month:02d}"
        
        # Create partitioned directory structure
        base_dir = "/home/siddharth.johri/DECOHERE/data/features/fundamentals/stable_features"
        partitioned_dir = os.path.join(base_dir, f"year={year}", f"month={month}")
        os.makedirs(partitioned_dir, exist_ok=True)
        
        # Save three different feature sets
        for k_multiplier in [1, 1.5, 2]:
            k = int(self.k_features * k_multiplier)
            
            # Get the aggregated importance for selected features
            selected_features = full_importance_ranking.head(k).index.tolist()
            aggregated_importance_ranking = pd.Series(
                {feature: avg_importance[feature] for feature in selected_features}
            ).sort_values(ascending=False)
            
            # Create DataFrame with all feature information
            feature_data = pd.DataFrame({
                'feature_name': full_importance_ranking.index,
                'importance_score': full_importance_ranking.values,
                'is_selected': full_importance_ranking.index.isin(selected_features),
                'is_compulsory': full_importance_ranking.index.isin(self.compulsory_features),
                'target_date': target_date
            })
            
            # Save feature data to parquet
            feature_file = os.path.join(partitioned_dir, f"data_{date_str}_k{k}.pq")
            feature_data.to_parquet(feature_file)
            
            # Save metadata to JSON
            metadata = {
                'target_date': date_str,
                'k_features_requested': k,
                'num_compulsory_features': len(self.compulsory_features),
                'num_selected_features': len(selected_features),
                'stable_feature_set': selected_features,
                'best_trial_params': serializable_params,
                'best_trial_stability_score (avg_spearman)': stability_score_save,
                'best_trial_rmse': rmse_save,
                'aggregated_importance_ranking': aggregated_importance_ranking.astype(float).to_dict()
            }
            
            metadata_file = os.path.join(partitioned_dir, f"metadata_{date_str}_k{k}.json")
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=4)
            
            logger.info(f"Successfully saved stable feature data to: {feature_file}")
            logger.info(f"Successfully saved metadata to: {metadata_file}")
            
            # Log the top 10 features from the full ranking
            logger.info(f"\nTop 10 Features from Full Ranking (k={k}):")
            for i, (feature, importance) in enumerate(full_importance_ranking.head(10).items(), 1):
                logger.info(f"{i}. {feature}: {importance:.4f}")
            
            # Log date information
            logger.info(f"\nDate Information (k={k}):")
            logger.info(f"Target Date: {date_str}")
            logger.info(f"Data Analysis Period: {target_date - pd.Timedelta(days=self.lookback_days)} to {target_date}")
            logger.info(f"Note: The target date {date_str} is included in the analysis period")

    def optimize_lightgbm_params(
        self, 
        stable_feature_set: List[str], 
        date_str: str,
        search_space: Dict[str, Dict[str, Any]],
        n_trials: int = 5,    # Reduced to 5 trials
        n_folds: int = 3,     # Reduced to 3 folds
        num_threads: int = 16  # Using all 16 vCPUs
    ) -> Dict[str, Any]:
        """Optimizes LightGBM hyperparameters using Optuna for the selected stable features."""
        start_time = time.time()
        logger.info("\n=== Starting LightGBM Hyperparameter Optimization ===")
        logger.info(f"Number of trials: {n_trials}")
        logger.info(f"Number of CV folds: {n_folds}")
        logger.info(f"Number of threads: {num_threads}")
        logger.info(f"Expected runtime: ~{n_trials * 1} minutes")
        
        # Configure system resources
        system_config = {
            'num_threads': num_threads,
            'max_bin': 255,  # Increased from default
            'histogram_pool_size': 1024 * 1024 * 64,  # 64MB per thread
            'gpu_platform_id': -1,  # Ensure CPU usage
            'gpu_device_id': -1,    # Ensure CPU usage
            'gpu_use_dp': False,    # Disable double precision
            'device_type': 'cpu',
            'verbose': -1
        }
        
        # Load data for the stable features
        data_load_start = time.time()
        X, y = self._load_data_for_features(stable_feature_set, date_str)
        data_load_time = time.time() - data_load_start
        logger.info(f"Data loading completed in {data_load_time:.2f} seconds")
        
        # Identify categorical features
        categorical_features = ['sector_1', 'sector_2']
        categorical_features = [f for f in categorical_features if f in X.columns]
        if categorical_features:
            logger.info(f"Using categorical features: {categorical_features}")
            # Convert to category type
            for col in categorical_features:
                X[col] = X[col].astype('category')
        else:
            logger.warning("No categorical features found in the data")
        
        # Create time-series aware CV splits
        tscv = TimeSeriesSplit(n_splits=n_folds)
        
        # Define the objective function for Optuna
        def objective(trial: optuna.trial.Trial) -> float:
            trial_start = time.time()
            logger.info(f"\nStarting trial {trial.number + 1}/{n_trials}")
            
            # Suggest hyperparameters with dependencies
            max_depth = trial.suggest_int('max_depth', 3, 6)
            num_leaves = trial.suggest_int('num_leaves', 
                                         min(31, 2**max_depth),
                                         min(63, 2**max_depth))
            
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                **system_config,  # Add system configuration
                'max_depth': max_depth,
                'num_leaves': num_leaves,
                'min_data_in_leaf': 5,
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.7, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.7, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 5),
                'lambda_l1': trial.suggest_float('lambda_l1', 1e-5, 0.1, log=True),
                'lambda_l2': trial.suggest_float('lambda_l2', 1e-5, 0.1, log=True),
                'min_gain_to_split': 0.0,
                'min_data_per_group': 5
            }
            
            # Add early stopping callback with reduced rounds
            early_stopping = lgb.early_stopping(
                stopping_rounds=20,
                verbose=False
            )
            
            # Perform time-series aware cross-validation
            fold_scores = []
            fold_importances = []
            fold_best_iterations = []
            
            for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
                fold_start = time.time()
                logger.info(f"Starting fold {fold + 1}/{n_folds} for trial {trial.number + 1}")
                
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                try:
                    # Create LightGBM datasets with categorical features
                    train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_features)
                    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data, categorical_feature=categorical_features)
                    
                    # Train model with early stopping
                    model = lgb.train(
                        params,
                        train_data,
                        num_boost_round=1000,
                        valid_sets=[val_data],
                        callbacks=[
                            early_stopping,
                            lgb.log_evaluation(period=50, show_stdv=False)
                        ]
                    )
                    
                    # Get validation score
                    val_pred = model.predict(X_val)
                    mse = mean_squared_error(y_val, val_pred)
                    val_rmse = np.sqrt(mse)
                    fold_scores.append(val_rmse)
                    
                    # Get feature importance
                    importance = pd.Series(model.feature_importance(), index=X.columns)
                    fold_importances.append(importance)
                    fold_best_iterations.append(model.best_iteration)
                    
                    # Log sector feature importances
                    if categorical_features:
                        sector_importances = importance[categorical_features]
                        logger.info(f"Fold {fold + 1} - Sector feature importances: {sector_importances.to_dict()}")
                    
                    fold_time = time.time() - fold_start
                    logger.info(f"Fold {fold + 1} completed in {fold_time:.2f} seconds")
                    logger.info(f"Fold {fold + 1} RMSE: {val_rmse:.4f}")
                    logger.info(f"Fold {fold + 1} best iteration: {model.best_iteration}")
                    
                except Exception as e:
                    logger.error(f"Error in fold {fold + 1}: {str(e)}")
                    return float('inf')  # Return infinity for failed folds
            
            # Calculate average score and importance
            avg_rmse = np.mean(fold_scores)
            avg_importance = pd.concat(fold_importances, axis=1).mean(axis=1)
            avg_best_iteration = np.mean(fold_best_iterations)
            
            # Log average sector importances
            if categorical_features:
                sector_avg_importances = avg_importance[categorical_features]
                logger.info(f"Average sector feature importances: {sector_avg_importances.to_dict()}")
            
            # Store trial information
            trial.set_user_attr('fold_scores', fold_scores)
            trial.set_user_attr('avg_importance', avg_importance.to_dict())
            trial.set_user_attr('best_iteration', avg_best_iteration)
            
            trial_time = time.time() - trial_start
            logger.info(f"Trial {trial.number + 1} completed in {trial_time:.2f} seconds")
            logger.info(f"Average RMSE: {avg_rmse:.4f}")
            logger.info(f"Average best iteration: {avg_best_iteration:.1f}")
            
            return float(avg_rmse)
        
        # Create Optuna study with pruning
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=2,
            n_warmup_steps=3,
            interval_steps=1
        )
        
        study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=pruner
        )
        
        # Optimize with parallel processing
        study.optimize(
            objective,
            n_trials=n_trials,
            n_jobs=num_threads,  # Use all available threads
            show_progress_bar=True
        )
        
        # Get best parameters if any trials completed successfully
        if study.best_trial is None:
            logger.error("No trials completed successfully. Check the logs for errors.")
            raise ValueError("No trials completed successfully. Check the logs for errors.")
            
        best_params = study.best_params
        best_params.update({
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'verbose': -1,
            'num_threads': num_threads,
            'max_bin': 255,
            'histogram_pool_size': 64 * 1024 * 1024,
            'gpu_platform_id': -1,
            'gpu_device_id': -1,
            'gpu_use_dp': False,
            'force_col_wise': True,
            'force_row_wise': False
        })
        
        # Store best metrics and trial history
        self.best_rmse = study.best_value
        self.best_iteration = study.best_trial.user_attrs['best_iteration']
        self.best_importance = pd.Series(study.best_trial.user_attrs['avg_importance'])
        
        # Get parameter importance
        param_importance = optuna.importance.get_param_importances(study)
        
        # Save results with appropriate k value in filename
        k_value = len(X.columns) - len(categorical_features)  # Exclude sector features
        self._save_hyperparam_results(
            stable_feature_set=X.columns.tolist(),  # Use all features including sectors
            date_str=date_str,
            search_space=search_space,
            study=study,
            k_value=k_value
        )
        
        return best_params

    def _save_hyperparam_results(
        self,
        stable_feature_set: List[str],
        date_str: str,
        search_space: Dict[str, Dict[str, Any]],
        study: optuna.study.Study,
        k_value: int
    ) -> None:
        """Saves the hyperparameter optimization results to a JSON file.
        
        Args:
            stable_feature_set: List of stable features used
            date_str: Target date string
            search_space: Search space used for optimization
            study: Optuna study object containing trial results
            k_value: Number of non-sector features used
        """
        # Convert target date string to datetime for partitioning
        target_date = pd.to_datetime(date_str)
        year = target_date.year
        month = f"{target_date.month:02d}"
        
        # Create output directory structure
        base_dir = "/home/siddharth.johri/DECOHERE/data/features/fundamentals/stable_hyperparams"
        output_dir = os.path.join(base_dir, f"year={year}", f"month={month}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Get best parameters and importance
        best_params = study.best_params
        best_trial = study.best_trial
        
        # Calculate parameter importance
        param_importance = optuna.importance.get_param_importances(study)
        
        # Prepare trial history with serializable values
        trial_history = {
            'params': [t.params for t in study.trials],
            'values': [float(t.value) if t.value is not None else None for t in study.trials],
            'durations': [t.duration.total_seconds() if t.duration is not None else None for t in study.trials],
            'states': [str(t.state) for t in study.trials]
        }
        
        # Prepare results dictionary
        results = {
            'target_date': date_str,
            'num_features': len(stable_feature_set),
            'feature_set': stable_feature_set,
            'search_space': search_space,
            'optimized_params': best_params,
            'best_rmse': float(self.best_rmse) if self.best_rmse is not None else None,
            'best_iteration': int(self.best_iteration) if self.best_iteration is not None else None,
            'feature_importance': self.best_importance.to_dict() if self.best_importance is not None else None,
            'param_importance': param_importance,
            'trial_history': trial_history,
            'cv_folds': 5,  # Updated to match notebook
            'num_trials': 5,  # Updated to match notebook
            'lookback_days': self.lookback_days,
            'system_config': self.system_config  # Use the configured system settings
        }
        
        # Save results to JSON file with k value from metadata
        output_file = os.path.join(output_dir, f"hyperparams_{date_str}_k{k_value}.json")
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)
            
        logger.info(f"Hyperparameter optimization results saved to: {output_file}")

    def tune_and_select_stable_features(
        self,
        panel_data_path: str,
        target_date_str: str,
        target_column_name: str,
        missing_threshold: float = 0.50,
        variance_threshold_value: float = 1e-4,
        univariate_corr_threshold: float = 0.01
    ) -> List[str]:
        """Main method to tune and select stable features.
        
        Args:
            panel_data_path: Path to the panel data directory
            target_date_str: Target date string in YYYY-MM-DD format
            target_column_name: Name of the target column
            missing_threshold: Threshold for missing values (default: 0.50)
            variance_threshold_value: Threshold for variance (default: 1e-4)
            univariate_corr_threshold: Threshold for univariate correlation (default: 0.01)
            
        Returns:
            List of selected stable feature names
        """
        # Convert target date string to datetime
        target_date = pd.to_datetime(target_date_str)
        
        # Load and prepare data
        if not self._load_and_prepare_data(panel_data_path, target_date, target_column_name):
            raise ValueError("Failed to load and prepare data")
        
        # Pre-filter features
        valid_features, dropped_log = self._pre_filter_features(
            self.X,
            missing_threshold=missing_threshold,
            variance_threshold=variance_threshold_value,
            univariate_corr_threshold=univariate_corr_threshold
        )
        
        # Create Optuna study
        study = self._create_study()
        
        # Optimize feature selection
        study.optimize(
            lambda trial: self._objective_stability_filtered(trial, self.n_splits, self.random_seed),
            n_trials=self.n_trials,
            n_jobs=self.optuna_n_jobs,
            show_progress_bar=True
        )
        
        # Get best trial and select stable features
        best_trial = self._filter_and_select_best_trial(study)
        if best_trial is None:
            raise ValueError("No valid trial found after filtering")
        
        # Get stable features
        stable_features, importance_ranking = self._aggregate_importance_and_select(best_trial)
        if stable_features is None:
            raise ValueError("Failed to select stable features")
        
        # Save results
        self._save_results(stable_features, best_trial, target_date_str)
        
        return stable_features

    def _save_optimized_params(self, params: Dict[str, Any], date_str: str) -> None:
        """Saves the optimized LightGBM parameters to a JSON file."""
        # Convert target date string to datetime for partitioning
        target_date = pd.to_datetime(date_str)
        year = target_date.year
        month = f"{target_date.month:02d}"
        
        # Create partitioned directory structure
        base_dir = "/home/siddharth.johri/DECOHERE/data/features/fundamentals/stable_features"
        partitioned_dir = os.path.join(base_dir, f"year={year}", f"month={month}")
        os.makedirs(partitioned_dir, exist_ok=True)
        
        # Save parameters to JSON
        params_file = os.path.join(partitioned_dir, f"optimized_params_{date_str}.json")
        with open(params_file, 'w') as f:
            json.dump(params, f, indent=4)
        
        logger.info(f"Successfully saved optimized LightGBM parameters to: {params_file}")

    def _get_stable_features_from_study(self) -> List[str]:
        """Get stable features from the study results."""
        if self.study is None:
            logger.error("No study available")
            return []

        best_trial = self.study.best_trial
        if best_trial is None:
            logger.error("No best trial found")
            return []

        # Retrain the best model on all available numeric data using best hyperparameters
        best_params = best_trial.params
        final_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'random_state': 42,
            **best_params
        }

        logger.info("Retraining final model on full dataset with best hyperparameters...")
        
        if self.X is None or self.y is None or self.numeric_feature_names is None:
            raise ValueError("Data (X, y, numeric_feature_names) not available for final model training.")

        # Prepare full numeric data
        X_numeric_full = self.X[self.numeric_feature_names]
        y_full = self.y.to_numpy()

        # Add check for infinite values in the full dataset
        if not np.all(np.isfinite(X_numeric_full.to_numpy()[np.isfinite(X_numeric_full.to_numpy())])):
            raise ValueError("Infinite values detected in the full feature set (X_numeric_full). Cannot train final model.")
        if not np.all(np.isfinite(y_full)):
            raise ValueError("Infinite values detected in the full target variable (y_full). Cannot train final model.")

        try:
            # Create LightGBM dataset
            train_data = lgb.Dataset(X_numeric_full, label=y_full)
            
            # Train final model
            final_model = lgb.train(
                final_params,
                train_data,
                num_boost_round=1000,
                callbacks=[lgb.log_evaluation(period=0)]
            )
            
            logger.info("Final model training complete.")
            
            # Get feature importances from the final model
            final_importances = pd.Series(final_model.feature_importance(), index=self.numeric_feature_names)

            # Select top features based on importance
            selected_features = final_importances.nlargest(self.k_features).index.tolist()
            return selected_features

        except Exception as e:
            logger.error(f"Error training final model: {e}")
            raise RuntimeError(f"Failed to train final model: {e}")

    def _calculate_average_spearman(self, fold_importances: List[pd.Series]) -> float:
        """
        Calculate the average Spearman correlation between feature importance rankings across folds.
        Uses a pool of candidate features (default 500) for efficient stability calculation.
        """
        # Get candidate pool of features from each fold
        candidate_features = set()
        for imp in fold_importances:
            # Get top N features by importance from each fold
            candidate_features.update(imp.nlargest(self.stability_candidate_pool).index)
        
        # Create DataFrame with candidate features
        ranks_df = pd.DataFrame(index=sorted(candidate_features))
        for i, imp in enumerate(fold_importances):
            # Only include candidate features
            imp_candidates = imp[imp.index.isin(candidate_features)]
            imp_filled = imp_candidates.reindex(ranks_df.index, fill_value=0.0)
            ranks_df[f'fold_{i}'] = imp_filled.rank(ascending=False)
        
        # Calculate correlations between folds
        correlations = []
        for i, j in itertools.combinations(range(len(fold_importances)), 2):
            x = np.asarray(ranks_df[f'fold_{i}'].values, dtype=float)
            y = np.asarray(ranks_df[f'fold_{j}'].values, dtype=float)
            corr_result = spearmanr(x, y)
            corr = float(corr_result[0]) if isinstance(corr_result, tuple) else float(corr_result)
            if not np.isnan(corr):
                correlations.append(corr)
        
        return float(np.mean(correlations)) if correlations else 0.0

    def _select_stable_features(self, best_trial: optuna.trial.FrozenTrial) -> List[str]:
        """
        Selects stable features based on the best trial's feature importances.
        Ensures exactly k_features (50) plus compulsory features (2) are selected.
        """
        if not best_trial.user_attrs.get("average_fold_importance"):
            logger.error(f"Invalid or missing 'average_fold_importance' dict in best trial {best_trial.number}.")
            raise ValueError("Missing feature importance data in best trial")

        # Get average feature importances
        avg_importance = pd.Series(best_trial.user_attrs["average_fold_importance"])
        
        # First, ensure compulsory features are included
        compulsory_features = []
        for feature in self.compulsory_features:
            if feature in avg_importance.index:
                compulsory_features.append(feature)
            else:
                logger.warning(f"Compulsory feature {feature} not found in importance data. It cannot be selected.")
        
        if len(compulsory_features) != 2:
            logger.error(f"Expected 2 compulsory features, but found {len(compulsory_features)}")
            raise ValueError(f"Expected 2 compulsory features, but found {len(compulsory_features)}")
        
        # Sort features by importance (descending), excluding compulsory features
        sorted_features = avg_importance.drop(compulsory_features).sort_values(ascending=False)
        
        # Select exactly k_features (50) non-compulsory features
        selected_non_compulsory = sorted_features.head(self.k_features).index.tolist()
        
        # Combine compulsory and selected features
        selected_features = compulsory_features + selected_non_compulsory
        
        # Verify the total number of features
        if len(selected_features) != self.k_features + len(compulsory_features):
            logger.error(f"Expected {self.k_features + len(compulsory_features)} features, but selected {len(selected_features)}")
            raise ValueError(f"Expected {self.k_features + len(compulsory_features)} features, but selected {len(selected_features)}")
        
        # Log the selection process
        logger.info(f"Selected {len(selected_features)} features:")
        logger.info(f"  - Compulsory features: {len(compulsory_features)}")
        logger.info(f"  - Top importance features: {len(selected_non_compulsory)}")
        logger.info(f"  - Total features (k + compulsory): {self.k_features + len(compulsory_features)}")
        
        # Log top 10 feature importances
        top_10 = avg_importance.head(10)
        logger.info("\nTop 10 Feature Importances (Aggregated):")
        logger.info(top_10)
        
        return selected_features

    def _load_data_for_features(
        self,
        features: List[str],
        date_str: str,
        lookback_days: int = 20
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Load data for the given features from the enhanced features directory.
        
        Args:
            features: List of feature names to load (if None, will load from metadata)
            date_str: Target date in YYYY-MM-DD format
            lookback_days: Number of days to look back for data
            
        Returns:
            Tuple of (feature matrix, target variable)
        """
        # Validate date_str
        if not date_str or not isinstance(date_str, str):
            raise ValueError("date_str must be a non-empty string in YYYY-MM-DD format")
            
        # Convert date string to datetime
        try:
            target_date = pd.to_datetime(date_str)
        except Exception as e:
            raise ValueError(f"Invalid date format for date_str: {date_str}. Expected YYYY-MM-DD format. Error: {str(e)}")
            
        year = target_date.year
        month = f"{target_date.month:02d}"
        
        # Log lookback period information
        logger.info(f"\n=== Data Loading Verification ===")
        logger.info(f"Target date: {target_date.strftime('%Y-%m-%d')}")
        logger.info(f"Lookback period: {lookback_days} days")
        
        # If features not provided, load from metadata
        if features is None:
            # Load metadata for k=50 by default
            metadata_path = f"/home/siddharth.johri/DECOHERE/data/features/fundamentals/stable_features/year={year}/month={month}/metadata_{date_str}_k50.json"
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                features = metadata['stable_feature_set']
                logger.info(f"Loaded {len(features)} features from metadata file: {metadata_path}")
                
                # Extract k value from filename
                k_match = re.search(r'_k(\d+)\.json$', metadata_path)
                if k_match:
                    self.k_value = int(k_match.group(1))
                    logger.info(f"Extracted k value from metadata filename: {self.k_value}")
                else:
                    logger.warning("Could not extract k value from metadata filename, defaulting to 50")
                    self.k_value = 50
                
                # Add sector features to the feature set if they're not already present
                sector_features = ['sector_1', 'sector_2']
                for sector_feature in sector_features:
                    if sector_feature not in features:
                        features.append(sector_feature)
                logger.info(f"Added sector features to feature set. Total features: {len(features)}")
                
            except FileNotFoundError:
                logger.warning(f"Metadata file not found at {metadata_path}. Using provided features list.")
                if not features:
                    raise ValueError("No features provided and metadata file not found")
        
        # Base directory for enhanced features
        base_dir = "/home/siddharth.johri/DECOHERE/data/features/fundamentals/enhanced_features"
        partitioned_dir = os.path.join(base_dir, f"year={year}", f"month={month}")
        
        # Find available data files
        available_dates = []
        current_date = target_date
        while len(available_dates) < lookback_days:
            date_str = current_date.strftime("%Y-%m-%d")
            data_file = os.path.join(partitioned_dir, f"data_{date_str}.pq")
            if os.path.exists(data_file):
                available_dates.append(current_date)
            current_date -= pd.Timedelta(days=1)
            
            # If we've gone back too far without finding enough data, raise an error
            if (target_date - current_date).days > 100:  # Arbitrary large number to prevent infinite loop
                raise ValueError(f"Could not find {lookback_days} days of data within reasonable lookback period")
        
        # Sort dates to ensure chronological order
        available_dates.sort()
        logger.info(f"Selected {len(available_dates)} days from {available_dates[0].strftime('%Y-%m-%d')} to {available_dates[-1].strftime('%Y-%m-%d')}")
        
        # Load data for the selected days
        all_data = []
        for current_date in available_dates:
            date_str = current_date.strftime("%Y-%m-%d")
            data_file = os.path.join(partitioned_dir, f"data_{date_str}.pq")
            
            try:
                # Load data with proper handling of categorical columns
                df = pd.read_parquet(data_file)
                logger.info(f"Loaded data from {date_str}")
                
                # Verify all required features are present
                missing_features = [f for f in features if f not in df.columns]
                if missing_features:
                    raise ValueError(f"Missing features in data: {missing_features}")
                
                # Verify target column exists
                target_col = 'PE_RATIO_RATIO_SIGNED_LOG'
                if target_col not in df.columns:
                    raise ValueError(f"Target column {target_col} not found in data")
                
                # Select only required features and target
                df = df[features + [target_col, 'PIT_DATE']].copy()
                
                # Handle NaN values differently for numeric and categorical features
                numeric_features = [f for f in features if f not in ['sector_1', 'sector_2']]
                categorical_features = [f for f in features if f in ['sector_1', 'sector_2']]
                
                # Fill NaN values for numeric features with 0
                if numeric_features:
                    df[numeric_features] = df[numeric_features].fillna(0)
                
                # Fill NaN values for categorical features with 'Missing_Sector'
                if categorical_features:
                    for col in categorical_features:
                        df[col] = df[col].fillna('Missing_Sector')
                        df[col] = df[col].astype('category')
                        df[col] = df[col].cat.set_categories(sorted(df[col].cat.categories))
                
                # Fill NaN values for target
                df[target_col] = df[target_col].fillna(0)
                
                # Store categorical feature names for LightGBM
                self.categorical_feature_names = categorical_features
                if self.categorical_feature_names:
                    logger.info(f"Identified categorical features: {self.categorical_feature_names}")
                else:
                    logger.warning("No categorical features (sector_1, sector_2) found in the data")
                
                all_data.append(df)
                
            except FileNotFoundError:
                logger.warning(f"Data file not found for date {date_str}: {data_file}")
                continue
        
        if not all_data:
            raise ValueError(f"No data found for any date in lookback period of {lookback_days} days")
        
        # Combine all data while preserving temporal order
        combined_df = pd.concat(all_data, axis=0)
        combined_df = combined_df.sort_values('PIT_DATE')  # Sort by PIT_DATE to ensure temporal order
        
        # Store dates and data in class attributes
        self.dates = combined_df['PIT_DATE']
        self.X = combined_df[features]
        self.y = combined_df[target_col]
        
        # Log data loading summary
        logger.info("\n=== Data Loading Summary ===")
        logger.info(f"Total days of data loaded: {len(available_dates)}")
        logger.info(f"Final dataframe dimensions: {combined_df.shape[0]} rows x {combined_df.shape[1]} columns")
        logger.info(f"Number of features: {len(features)}")
        logger.info(f"Number of categorical features: {len(self.categorical_feature_names)}")
        
        # Print summary for notebook visibility
        print("\n=== Data Loading Summary ===")
        print(f"Total days of data loaded: {len(available_dates)}")
        print(f"Final dataframe dimensions: {combined_df.shape[0]} rows x {combined_df.shape[1]} columns")
        print(f"Number of features: {len(features)}")
        print(f"Number of categorical features: {len(self.categorical_feature_names)}")
        
        return self.X, self.y

# Example usage (intended to be called from a runner script)
# if __name__ == '__main__':
#     # This part is better placed in a separate runner script
#     # to handle command-line arguments for path, date, etc.
#     pass 
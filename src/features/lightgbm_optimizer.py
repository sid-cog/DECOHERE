import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import logging
from datetime import datetime, timedelta
import os
from typing import Dict, Tuple, List, Optional
import multiprocessing

logger = logging.getLogger(__name__)

class LightGBMOptimizer:
    def __init__(self, n_trials: int = 50, n_splits: int = 3, lookback_days: int = 20):
        self.n_trials = n_trials
        self.n_splits = n_splits
        self.lookback_days = lookback_days
        self.best_params: Optional[Dict] = None
        self.best_rmse: Optional[float] = None
        
        # Optimize for 16 vCPUs
        self.num_threads = min(16, multiprocessing.cpu_count())
        logger.info(f"Using {self.num_threads} threads for LightGBM")

    def load_data(self, date_str: str, lookback_days: int = 20) -> None:
        """Load data for the given date and lookback period.
        
        Args:
            date_str: Target date in YYYY-MM-DD format
            lookback_days: Number of days to look back for data
        """
        # Convert date string to datetime
        target_date = pd.to_datetime(date_str)
        year = target_date.year
        month = f"{target_date.month:02d}"
        
        # Base directory for enhanced features
        base_dir = "/home/siddharth.johri/DECOHERE/data/features/fundamentals/enhanced_features"
        partitioned_dir = os.path.join(base_dir, f"year={year}", f"month={month}")
        
        # Load data for each day in lookback period
        all_data = []
        for i in range(lookback_days):
            current_date = target_date - pd.Timedelta(days=i)
            date_str = current_date.strftime("%Y-%m-%d")
            data_file = os.path.join(partitioned_dir, f"data_{date_str}.pq")
            
            try:
                # Load data with proper handling of categorical columns
                df = pd.read_parquet(data_file)
                logger.info(f"Loaded data from {date_str}")
                
                # Verify all required features are present
                missing_features = [f for f in self.stable_features if f not in df.columns]
                if missing_features:
                    raise ValueError(f"Missing features in data: {missing_features}")
                
                # Verify target column exists
                target_col = 'PE_RATIO_RATIO_SIGNED_LOG'
                if target_col not in df.columns:
                    raise ValueError(f"Target column {target_col} not found in data")
                
                # Select only required features and target
                df = df[self.stable_features + [target_col, 'PIT_DATE']].copy()
                
                # Handle NaN values
                df[self.stable_features] = df[self.stable_features].fillna(0)  # Fill feature NaNs with 0
                df[target_col] = df[target_col].fillna(0)  # Fill target NaNs with 0
                
                # Convert categorical columns to proper type
                categorical_cols = ['sector_1', 'sector_2']
                for col in categorical_cols:
                    if col in df.columns:
                        # Fill NaN values with a specific category
                        df[col] = df[col].fillna('Missing_Sector')
                        # Convert to category type
                        df[col] = df[col].astype('category')
                        # Ensure categories are properly ordered
                        df[col] = df[col].cat.set_categories(sorted(df[col].cat.categories))
                
                # Store categorical feature names for LightGBM
                self.categorical_feature_names = [col for col in categorical_cols if col in df.columns]
                logger.info(f"Identified categorical features: {self.categorical_feature_names}")
                
                all_data.append(df)
                
            except FileNotFoundError:
                logger.warning(f"Data file not found for date {date_str}: {data_file}")
                continue
        
        if not all_data:
            raise ValueError(f"No data found for any date in lookback period of {lookback_days} days")
        
        # Combine all data while preserving temporal order
        combined_df = pd.concat(all_data, axis=0)
        combined_df = combined_df.sort_values('PIT_DATE')  # Sort by PIT_DATE to ensure temporal order
        
        # Store dates for logging
        self.dates = combined_df['PIT_DATE']
        logger.info(f"Data loaded from {combined_df['PIT_DATE'].min()} to {combined_df['PIT_DATE'].max()}")
        
        # Split into features and target
        self.X = combined_df[self.stable_features]
        self.y = combined_df[target_col]

    def objective(self, trial: optuna.trial.Trial) -> float:
        """Optuna objective function for LightGBM hyperparameter optimization."""
        if self.X is None or self.y is None:
            raise ValueError("Data not loaded. Call load_data() first.")
            
        # Define hyperparameter search space
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': trial.suggest_int('num_leaves', 30, 50),
            'max_depth': trial.suggest_int('max_depth', 4, 6),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05, log=True),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.8, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.8, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 3, 5),
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-4, 0.01, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-4, 0.01, log=True),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 20),
            'min_child_weight': trial.suggest_float('min_child_weight', 1e-4, 0.01, log=True),
            'min_split_gain': trial.suggest_float('min_split_gain', 0.01, 0.1),
            'verbose': -1,
            'seed': 42
        }
        
        # Create time-series aware CV splits
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        
        # Track validation metrics
        fold_rmses = []
        fold_iterations = []
        
        # Get categorical feature indices
        categorical_feature_indices = []
        for i, col in enumerate(self.X.columns):
            if col in ['sector_1', 'sector_2']:
                categorical_feature_indices.append(i)
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(self.X)):
            X_train, X_val = self.X.iloc[train_idx], self.X.iloc[val_idx]
            y_train, y_val = self.y.iloc[train_idx], self.y.iloc[val_idx]
            
            # Log date ranges for each fold
            train_dates = self.dates.iloc[train_idx]
            val_dates = self.dates.iloc[val_idx]
            logger.info(f"Fold {fold + 1}/{self.n_splits} - Train dates: {train_dates.min()} to {train_dates.max()}")
            logger.info(f"Fold {fold + 1}/{self.n_splits} - Val dates: {val_dates.min()} to {val_dates.max()}")
            
            # Create LightGBM datasets
            train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_feature_indices)
            val_data = lgb.Dataset(X_val, label=y_val, categorical_feature=categorical_feature_indices, reference=train_data)
            
            # Train model with fixed number of iterations
            model = lgb.train(
                params,
                train_data,
                num_boost_round=500,  # Fixed number of iterations
                valid_sets=[val_data],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=30),
                    lgb.log_evaluation(period=0)
                ]
            )
            
            # Get validation predictions
            y_pred = model.predict(X_val)
            
            # Calculate RMSE
            rmse = float(np.sqrt(mean_squared_error(y_val, y_pred)))
            fold_rmses.append(rmse)
            fold_iterations.append(model.best_iteration)
            
            logger.info(f"Fold {fold + 1}/{self.n_splits} RMSE: {rmse:.4f} (iterations: {model.best_iteration})")
        
        # Calculate average RMSE across folds
        avg_rmse = float(np.mean(fold_rmses))
        
        # Add stability check with adjusted threshold
        fold_std = np.std(fold_rmses)
        if fold_std > 0.2:  # Increased threshold to 0.2
            logger.warning(f"Trial {trial.number} rejected due to high fold variance: {fold_std:.4f}")
            return float('inf')
            
        # Check for consistent early stopping
        iterations_std = np.std(fold_iterations)
        if iterations_std > 100:  # If iterations vary too much
            logger.warning(f"Trial {trial.number} rejected due to inconsistent early stopping: {iterations_std:.1f}")
            return float('inf')
            
        logger.info(f"Trial {trial.number} completed. Average RMSE: {avg_rmse:.4f}")
        
        return avg_rmse

    def optimize(self, date_str: str, stable_features: List[str]) -> Tuple[Dict, float]:
        """Run the optimization process."""
        logger.info(f"Starting optimization for date {date_str}")
        
        self.stable_features = stable_features
        self.load_data(date_str)
        logger.info(f"Loaded data shape: {self.X.shape}")
        
        # Configure Optuna for parallel optimization
        study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=3,
                interval_steps=1
            )
        )
        
        # Run optimization with parallel trials
        study.optimize(
            lambda trial: self.objective(trial),
            n_trials=self.n_trials,
            show_progress_bar=True,
            n_jobs=self.num_threads
        )
        
        self.best_params = study.best_params
        self.best_params.update({
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_threads': self.num_threads,
            'device_type': 'cpu',
            'gpu_platform_id': -1,
            'gpu_device_id': -1,
            'gpu_use_dp': False
        })
        
        self.best_rmse = study.best_value
        logger.info(f"Best parameters: {self.best_params}")
        logger.info(f"Best RMSE: {self.best_rmse:.4f}")
        
        return self.best_params, self.best_rmse 
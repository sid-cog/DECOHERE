import os
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import lightgbm as lgb
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DailyResidualGenerator:
    """Class to generate daily prediction residuals using LightGBM."""
    
    def __init__(
        self,
        enhanced_features_dir: str = "/home/siddharth.johri/DECOHERE/data/features/fundamentals/enhanced_features",
        hyperparams_dir: str = "/home/siddharth.johri/DECOHERE/data/features/fundamentals/stable_hyperparams",
        results_dir: str = "/home/siddharth.johri/DECOHERE/data/features/fundamentals/daily_residuals"
    ):
        """
        Initialize the DailyResidualGenerator.
        
        Args:
            enhanced_features_dir: Directory containing enhanced features data
            hyperparams_dir: Directory containing hyperparameter JSON files
            results_dir: Directory to save residual results
        """
        self.enhanced_features_dir = Path(enhanced_features_dir)
        self.hyperparams_dir = Path(hyperparams_dir)
        self.results_dir = Path(results_dir)
        
        # Create results directory if it doesn't exist
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized DailyResidualGenerator with:")
        logger.info(f"Enhanced features directory: {self.enhanced_features_dir}")
        logger.info(f"Hyperparameters directory: {self.hyperparams_dir}")
        logger.info(f"Results directory: {self.results_dir}")
    
    def _find_latest_hyperparams(self, target_date: str) -> Tuple[str, str]:
        """
        Find the most recent hyperparameters file before the target date.
        
        Args:
            target_date: Target date in YYYY-MM-DD format
            
        Returns:
            Tuple of (filepath, k_value) for the most recent hyperparameters file
        """
        target_dt = datetime.strptime(target_date, "%Y-%m-%d")
        max_lookback = 90  # Maximum days to look back for hyperparameters
        
        logger.info(f"Looking for hyperparameter files up to {max_lookback} days before {target_date}")
        
        for days_back in range(max_lookback):
            check_date = target_dt - timedelta(days=days_back)
            year = check_date.strftime("%Y")
            month = check_date.strftime("%m")
            
            # Look for hyperparameter files in year/month directory
            hyperparams_path = self.hyperparams_dir / f"year={year}" / f"month={month}"
            logger.info(f"Checking directory: {hyperparams_path}")
            
            if not hyperparams_path.exists():
                logger.info(f"Directory does not exist, continuing to next date")
                continue
                
            # Find the most recent k-value file
            files = list(hyperparams_path.glob("hyperparams_*_k*.json"))
            logger.info(f"Found {len(files)} files in {hyperparams_path}")
            
            if files:
                # Get the most recent file
                latest_file = max(files, key=lambda x: x.stat().st_mtime)
                logger.info(f"Found suitable file: {latest_file}")
                
                # Extract k value from filename (e.g., hyperparams_2024-08-30_k50.json)
                k_value = latest_file.stem.split("_k")[1]
                logger.info(f"Using k value: {k_value}")
                return str(latest_file), k_value
        
        raise ValueError(f"No hyperparameter files found within {max_lookback} days before {target_date}")
    
    def _load_hyperparams(self, hyperparams_file: str) -> Tuple[Dict, List[str]]:
        """
        Load hyperparameters and features from JSON file.
        
        Args:
            hyperparams_file: Path to hyperparameters JSON file
            
        Returns:
            Tuple of (hyperparameters, features)
        """
        with open(hyperparams_file, 'r') as f:
            data = json.load(f)
            
        # Extract hyperparameters and features
        hyperparams = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'verbose': -1,
            'max_depth': data['optimized_params']['max_depth'],
            'num_leaves': data['optimized_params']['num_leaves'],
            'learning_rate': data['optimized_params']['learning_rate'],
            'feature_fraction': data['optimized_params']['feature_fraction'],
            'bagging_fraction': data['optimized_params']['bagging_fraction'],
            'bagging_freq': data['optimized_params']['bagging_freq'],
            'lambda_l1': data['optimized_params']['lambda_l1'],
            'lambda_l2': data['optimized_params']['lambda_l2']
        }
        features = data['feature_set']  # Use feature_set instead of features
        
        logger.info(f"Loaded hyperparameters from {hyperparams_file}")
        logger.info(f"Number of features: {len(features)}")
        logger.info(f"Features being used: {features}")
        logger.info(f"Hyperparameters: {hyperparams}")
        
        return hyperparams, features
    
    def _load_daily_data(self, date: str, features: List[str]) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load data for a single day.
        
        Args:
            date: Date in YYYY-MM-DD format
            features: List of feature names to load
            
        Returns:
            Tuple of (X, y) where X is features and y is target
        """
        # Construct path to daily data
        date_dt = datetime.strptime(date, "%Y-%m-%d")
        year = date_dt.strftime("%Y")
        month = date_dt.strftime("%m")
        day = date_dt.strftime("%d")
        
        # Use the correct file naming pattern
        data_path = self.enhanced_features_dir / f"year={year}" / f"month={month}" / f"data_{date}.pq"
        
        if not data_path.exists():
            raise FileNotFoundError(f"No data found for date {date} at path: {data_path}")
            
        # Load data
        df = pd.read_parquet(data_path)
        
        # Print column names for debugging
        logger.info(f"Available columns: {list(df.columns)}")
        
        # Check if all required columns exist
        missing_features = [f for f in features if f not in df.columns]
        if missing_features:
            raise KeyError(f"Missing features in data: {missing_features}")
            
        # Select features and target
        X = df[features]
        y = df['PE_RATIO_RATIO_SIGNED_LOG']  # Using PE_RATIO_RATIO_SIGNED_LOG as the target column
        
        # Handle missing values - treat categorical columns differently
        for col in X.columns:
            if pd.api.types.is_categorical_dtype(X[col]):
                # For categorical columns, fill with the most frequent category
                X[col] = X[col].fillna(X[col].mode()[0])
            else:
                # For numerical columns, fill with 0
                X[col] = X[col].fillna(0)
        
        # Fill target missing values with 0
        y = y.fillna(0)
        
        logger.info(f"Loaded data for {date}:")
        logger.info(f"Shape: {X.shape[0]} samples, {X.shape[1]} features")
        logger.info(f"Features used: {list(X.columns)}")
        logger.info(f"Target range: [{y.min():.4f}, {y.max():.4f}]")
        
        return X, y
    
    def _run_lightgbm(self, X: pd.DataFrame, y: pd.Series, params: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run LightGBM and calculate residuals.
        
        Args:
            X: Feature matrix
            y: Target vector
            params: LightGBM parameters
            
        Returns:
            Tuple of (predictions, residuals)
        """
        # Create dataset
        train_data = lgb.Dataset(X, label=y)
        
        # Train model
        model = lgb.train(
            params,
            train_data,
            num_boost_round=100,
            valid_sets=[train_data],
            callbacks=[lgb.early_stopping(10)]
        )
        
        # Get predictions and calculate residuals
        predictions = model.predict(X)
        residuals = y - predictions
        
        logger.info(f"Model training complete:")
        logger.info(f"Predictions range: [{predictions.min():.4f}, {predictions.max():.4f}]")
        logger.info(f"Residuals range: [{residuals.min():.4f}, {residuals.max():.4f}]")
        
        return predictions, residuals
    
    def _save_results(
        self,
        date: str,
        k_value: str,
        predictions: np.ndarray,
        residuals: np.ndarray,
        features: List[str],
        params: Dict
    ) -> None:
        """
        Save results and metadata.
        
        Args:
            date: Date in YYYY-MM-DD format
            k_value: Number of features used
            predictions: Model predictions
            residuals: Prediction residuals
            features: List of features used
            params: LightGBM parameters used
        """
        # Create date directory structure
        date_dt = datetime.strptime(date, "%Y-%m-%d")
        year = date_dt.strftime("%Y")
        month = date_dt.strftime("%m")
        day = date_dt.strftime("%d")
        
        results_path = self.results_dir / f"year={year}" / f"month={month}" / f"day={day}"
        results_path.mkdir(parents=True, exist_ok=True)
        
        # Convert to numpy arrays for consistent handling
        predictions = np.array(predictions)
        residuals = np.array(residuals)
        
        # Calculate basic statistics
        residual_stats = {
            'mean': float(residuals.mean()),
            'std': float(residuals.std()),
            'min': float(residuals.min()),
            'max': float(residuals.max()),
            'median': float(np.median(residuals)),
            'skewness': float(pd.Series(residuals).skew()),
            'kurtosis': float(pd.Series(residuals).kurtosis()),
            'percentiles': {
                '1%': float(np.percentile(residuals, 1)),
                '5%': float(np.percentile(residuals, 5)),
                '25%': float(np.percentile(residuals, 25)),
                '75%': float(np.percentile(residuals, 75)),
                '95%': float(np.percentile(residuals, 95)),
                '99%': float(np.percentile(residuals, 99))
            }
        }
        
        # Save residuals and predictions
        results_df = pd.DataFrame({
            'predictions': predictions,
            'residuals': residuals
        })
        results_df.to_parquet(results_path / f"k_{k_value}_residuals.parquet")
        
        # Save metadata with detailed statistics
        metadata = {
            'date': date,
            'k_value': k_value,
            'features': features,
            'params': params,
            'num_samples': len(predictions),
            'prediction_stats': {
                'mean': float(predictions.mean()),
                'std': float(predictions.std()),
                'min': float(predictions.min()),
                'max': float(predictions.max())
            },
            'residual_stats': residual_stats
        }
        
        with open(results_path / f"k_{k_value}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved results to {results_path}")
        logger.info(f"Residual Statistics:")
        logger.info(f"Mean: {residual_stats['mean']:.4f}")
        logger.info(f"Std: {residual_stats['std']:.4f}")
        logger.info(f"Min: {residual_stats['min']:.4f}")
        logger.info(f"Max: {residual_stats['max']:.4f}")
        logger.info(f"Skewness: {residual_stats['skewness']:.4f}")
        logger.info(f"Kurtosis: {residual_stats['kurtosis']:.4f}")
    
    def generate_residuals(self, date: str) -> None:
        """
        Generate residuals for a given date.
        
        Args:
            date: Date in YYYY-MM-DD format
        """
        try:
            logger.info(f"Starting residual generation for {date}")
            
            # Find and load hyperparameters
            hyperparams_file, k_value = self._find_latest_hyperparams(date)
            params, features = self._load_hyperparams(hyperparams_file)
            
            # Load data
            X, y = self._load_daily_data(date, features)
            
            # Run LightGBM
            predictions, residuals = self._run_lightgbm(X, y, params)
            
            # Save results
            self._save_results(date, k_value, predictions, residuals, features, params)
            
            logger.info(f"Successfully completed residual generation for {date}")
            
        except Exception as e:
            logger.error(f"Error generating residuals for {date}: {str(e)}")
            raise 
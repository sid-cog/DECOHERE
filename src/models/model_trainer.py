#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Model trainer for training regression models.
"""

import os
import pandas as pd
import numpy as np
import logging
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path


class ModelTrainer:
    """
    Model trainer for training regression models.
    """
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Initialize the model trainer.
        
        Args:
            config: Configuration dictionary
            logger: Logger to use (if None, create a new logger)
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Extract configuration parameters
        self.model_type = config['model'].get('type', 'elasticnet')
        self.cv_folds = config['model'].get('cv_folds', 5)
        self.test_size = config['model'].get('test_size', 0.2)
        self.random_state = config['model'].get('random_state', 42)
        
        # Model parameters
        self.model_params = config['model'].get('params', {})
        
        # Create results directory
        self.results_dir = config.get('output', {}).get('results_dir', 'results/model')
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Initialize model
        self.model = self._initialize_model()
    
    def _initialize_model(self):
        """
        Initialize the regression model based on the configuration.
        
        Returns:
            Initialized model
        """
        self.logger.info(f"Initializing {self.model_type} model")
        
        if self.model_type == 'elasticnet':
            # Get ElasticNet parameters
            alpha = self.model_params.get('alpha', 0.1)
            l1_ratio = self.model_params.get('l1_ratio', 0.5)
            
            # Initialize ElasticNet model
            model = ElasticNet(
                alpha=alpha,
                l1_ratio=l1_ratio,
                random_state=self.random_state
            )
        elif self.model_type == 'linear':
            # Initialize Linear Regression model
            model = LinearRegression()
        elif self.model_type == 'randomforest':
            # Get RandomForest parameters
            n_estimators = self.model_params.get('n_estimators', 100)
            max_depth = self.model_params.get('max_depth', 10)
            
            # Initialize RandomForest model
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=self.random_state
            )
        else:
            self.logger.error(f"Unsupported model type: {self.model_type}")
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        return model
    
    def load_features(self, features_file: str) -> pd.DataFrame:
        """
        Load features from a file.
        
        Args:
            features_file: Path to the features file
            
        Returns:
            DataFrame containing features
        """
        # Check if file exists
        if not os.path.exists(features_file):
            self.logger.error(f"Features file not found: {features_file}")
            raise FileNotFoundError(f"Features file not found: {features_file}")
        
        # Load features
        self.logger.info(f"Loading features from {features_file}")
        features_df = pd.read_parquet(features_file)
        
        self.logger.info(f"Loaded features with shape: {features_df.shape}")
        
        return features_df
    
    def prepare_data(self, features_df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Prepare data for model training.
        
        Args:
            features_df: DataFrame containing features
            target_col: Target column name
            
        Returns:
            Tuple of (X_train, y_train, X_test, y_test)
        """
        self.logger.info(f"Preparing data for model training with target: {target_col}")
        
        # Check if target column exists
        if target_col not in features_df.columns:
            self.logger.error(f"Target column not found: {target_col}")
            raise ValueError(f"Target column not found: {target_col}")
        
        # Get feature columns (exclude id, pit_date, and target columns)
        feature_cols = [col for col in features_df.columns 
                      if col not in ['id', 'pit_date'] 
                      and not col.endswith('_TARGET')]
        
        # Create X and y
        X = features_df[feature_cols].copy()
        y = features_df[target_col].copy()
        
        # Drop rows with NaN in target
        mask = ~y.isna()
        X = X.loc[mask]
        y = y.loc[mask]
        
        # Fill remaining NaN values in features with 0
        X = X.fillna(0)
        
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        
        self.logger.info(f"Prepared data with {X_train.shape[1]} features")
        self.logger.info(f"Training set: {X_train.shape[0]} samples")
        self.logger.info(f"Test set: {X_test.shape[0]} samples")
        
        return X_train, y_train, X_test, y_test
    
    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training target
        """
        self.logger.info(f"Training {self.model_type} model")
        
        # Fit the model
        self.model.fit(X_train, y_train)
        
        # Log training score
        train_score = self.model.score(X_train, y_train)
        self.logger.info(f"Training R² score: {train_score:.4f}")
    
    def evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """
        Evaluate the model on the test set.
        
        Args:
            X_test: Test features
            y_test: Test target
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.logger.info("Evaluating model on test set")
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Log metrics
        self.logger.info(f"Test MSE: {mse:.4f}")
        self.logger.info(f"Test RMSE: {rmse:.4f}")
        self.logger.info(f"Test MAE: {mae:.4f}")
        self.logger.info(f"Test R²: {r2:.4f}")
        
        # Return metrics
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
    
    def cross_validate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, List[float]]:
        """
        Perform cross-validation.
        
        Args:
            X: Features
            y: Target
            
        Returns:
            Dictionary of cross-validation scores
        """
        self.logger.info(f"Performing {self.cv_folds}-fold cross-validation")
        
        # Initialize KFold
        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        
        # Initialize lists to store scores
        r2_scores = []
        mse_scores = []
        
        # Perform cross-validation
        for train_index, test_index in kf.split(X):
            # Split data
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            # Train model
            model = self._initialize_model()
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate scores
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            
            # Append scores
            r2_scores.append(r2)
            mse_scores.append(mse)
        
        # Calculate mean and std of scores
        mean_r2 = np.mean(r2_scores)
        std_r2 = np.std(r2_scores)
        mean_mse = np.mean(mse_scores)
        std_mse = np.std(mse_scores)
        
        # Log scores
        self.logger.info(f"CV R² score: {mean_r2:.4f} ± {std_r2:.4f}")
        self.logger.info(f"CV MSE: {mean_mse:.4f} ± {std_mse:.4f}")
        
        # Return scores
        return {
            'r2_scores': r2_scores,
            'mse_scores': mse_scores,
            'mean_r2': mean_r2,
            'std_r2': std_r2,
            'mean_mse': mean_mse,
            'std_mse': std_mse
        }
    
    def get_feature_importance(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Get feature importance.
        
        Args:
            X: Features
            
        Returns:
            DataFrame containing feature importance
        """
        self.logger.info("Calculating feature importance")
        
        # Get feature importance based on model type
        if self.model_type == 'elasticnet':
            # Get coefficients
            importance = np.abs(self.model.coef_)
        elif self.model_type == 'linear':
            # Get coefficients
            importance = np.abs(self.model.coef_)
        elif self.model_type == 'randomforest':
            # Get feature importance
            importance = self.model.feature_importances_
        else:
            self.logger.warning(f"Feature importance not available for model type: {self.model_type}")
            return pd.DataFrame()
        
        # Create importance DataFrame
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': importance
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        # Calculate normalized importance
        importance_df['normalized_importance'] = importance_df['importance'] / importance_df['importance'].sum()
        
        # Calculate cumulative importance
        importance_df['cumulative_importance'] = importance_df['normalized_importance'].cumsum()
        
        return importance_df
    
    def visualize_results(self, X_test: pd.DataFrame, y_test: pd.Series, importance_df: pd.DataFrame) -> None:
        """
        Visualize model results.
        
        Args:
            X_test: Test features
            y_test: Test target
            importance_df: Feature importance DataFrame
        """
        self.logger.info("Visualizing model results")
        
        # Create visualizations directory
        vis_dir = os.path.join(self.results_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Plot actual vs predicted
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title('Actual vs Predicted')
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, "actual_vs_predicted.png"))
        plt.close()
        
        # Plot residuals
        residuals = y_test - y_pred
        plt.figure(figsize=(10, 6))
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted')
        plt.ylabel('Residuals')
        plt.title('Residuals vs Predicted')
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, "residuals.png"))
        plt.close()
        
        # Plot residuals distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(residuals, kde=True)
        plt.xlabel('Residuals')
        plt.ylabel('Frequency')
        plt.title('Residuals Distribution')
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, "residuals_distribution.png"))
        plt.close()
        
        # Plot feature importance
        if not importance_df.empty:
            plt.figure(figsize=(12, 8))
            plt.barh(
                importance_df.head(20)['feature'],
                importance_df.head(20)['importance']
            )
            plt.xlabel('Importance')
            plt.ylabel('Feature')
            plt.title('Top 20 Features by Importance')
            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, "feature_importance.png"))
            plt.close()
    
    def save_model(self, model_name: str) -> str:
        """
        Save the trained model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Path to the saved model
        """
        self.logger.info(f"Saving {self.model_type} model")
        
        # Create models directory
        models_dir = os.path.join(self.results_dir, "models")
        os.makedirs(models_dir, exist_ok=True)
        
        # Define model file path
        model_file = os.path.join(models_dir, f"{model_name}.joblib")
        
        # Save model
        joblib.dump(self.model, model_file)
        
        self.logger.info(f"Saved model to {model_file}")
        
        return model_file
    
    def create_summary(self, metrics: Dict[str, float], cv_scores: Dict[str, Any], 
                     importance_df: pd.DataFrame, target_col: str) -> None:
        """
        Create a summary of the model results.
        
        Args:
            metrics: Dictionary of evaluation metrics
            cv_scores: Dictionary of cross-validation scores
            importance_df: Feature importance DataFrame
            target_col: Target column name
        """
        self.logger.info("Creating model summary")
        
        # Create summary file
        summary_file = os.path.join(self.results_dir, "summary.txt")
        
        with open(summary_file, 'w') as f:
            f.write(f"Model Training Summary for {target_col}\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Model Type: {self.model_type}\n")
            f.write(f"Model Parameters: {self.model_params}\n")
            f.write(f"CV Folds: {self.cv_folds}\n")
            f.write(f"Test Size: {self.test_size}\n")
            f.write(f"Random State: {self.random_state}\n\n")
            
            f.write("Cross-Validation Results:\n")
            f.write("-" * 50 + "\n")
            f.write(f"Mean R²: {cv_scores['mean_r2']:.4f} ± {cv_scores['std_r2']:.4f}\n")
            f.write(f"Mean MSE: {cv_scores['mean_mse']:.4f} ± {cv_scores['std_mse']:.4f}\n\n")
            
            f.write("Test Set Results:\n")
            f.write("-" * 50 + "\n")
            f.write(f"MSE: {metrics['mse']:.4f}\n")
            f.write(f"RMSE: {metrics['rmse']:.4f}\n")
            f.write(f"MAE: {metrics['mae']:.4f}\n")
            f.write(f"R²: {metrics['r2']:.4f}\n\n")
            
            if not importance_df.empty:
                f.write("Top 20 Features by Importance:\n")
                f.write("-" * 50 + "\n")
                
                for i, (feature, importance) in enumerate(zip(importance_df.head(20)['feature'], importance_df.head(20)['importance']), 1):
                    f.write(f"{i:2d}. {feature:40s} | {importance:.6f}\n")
        
        self.logger.info(f"Saved summary to {summary_file}")
    
    def train_and_evaluate(self, features_file: str, target_col: str, model_name: str) -> Dict[str, Any]:
        """
        Train and evaluate the model.
        
        Args:
            features_file: Path to the features file
            target_col: Target column name
            model_name: Name of the model
            
        Returns:
            Dictionary of results
        """
        self.logger.info(f"Training and evaluating model for target: {target_col}")
        
        # Load features
        features_df = self.load_features(features_file)
        
        # Prepare data
        X_train, y_train, X_test, y_test = self.prepare_data(features_df, target_col)
        
        # Train model
        self.train_model(X_train, y_train)
        
        # Evaluate model
        metrics = self.evaluate_model(X_test, y_test)
        
        # Perform cross-validation
        X = pd.concat([X_train, X_test])
        y = pd.concat([y_train, y_test])
        cv_scores = self.cross_validate(X, y)
        
        # Get feature importance
        importance_df = self.get_feature_importance(X)
        
        # Visualize results
        self.visualize_results(X_test, y_test, importance_df)
        
        # Save model
        model_file = self.save_model(model_name)
        
        # Create summary
        self.create_summary(metrics, cv_scores, importance_df, target_col)
        
        # Return results
        results = {
            'metrics': metrics,
            'cv_scores': cv_scores,
            'importance_df': importance_df,
            'model_file': model_file
        }
        
        return results 
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Regression Model for the DECOHERE project.
This module contains the RegressionModel class, which handles model training and evaluation.
"""

import os
import pandas as pd
import numpy as np
import logging
import optuna
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import ElasticNet
import xgboost as xgb


class RegressionModel:
    """
    Regression Model class for the DECOHERE project.
    Handles model training and evaluation for both Elastic Net and XGBoost models.
    """
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Initialize the RegressionModel.
        
        Args:
            config: Configuration dictionary
            logger: Logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Get model parameters from config
        self.elastic_net_params = config.get('regression', {}).get('elastic_net', {})
        self.xgboost_params = config.get('regression', {}).get('xgboost', {})
        
        # Initialize models
        self.elastic_net_model = None
        self.xgboost_model = None
    
    def objective_elastic_net(self, trial, X: pd.DataFrame, y: pd.Series, cv_splits: int = 5) -> float:
        """
        Objective function for Elastic Net parameter optimization using Optuna.
        
        Args:
            trial: Optuna trial object
            X: Feature matrix
            y: Target variable
            cv_splits: Number of cross-validation splits
            
        Returns:
            Mean cross-validation score
        """
        # Define parameter space
        alpha = trial.suggest_float('alpha', 1e-4, 1.0, log=True)
        l1_ratio = trial.suggest_float('l1_ratio', 0.0, 1.0)
        
        # Initialize model
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        
        # Cross-validation
        cv = KFold(n_splits=cv_splits, shuffle=True, random_state=42)
        scores = []
        
        for train_idx, val_idx in cv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            score = mean_squared_error(y_val, y_pred)
            scores.append(score)
        
        return np.mean(scores)
    
    def optimize_elastic_net(self, X: pd.DataFrame, y: pd.Series, n_trials: int = 100) -> Dict[str, Any]:
        """
        Optimize Elastic Net parameters using Optuna.
        
        Args:
            X: Feature matrix
            y: Target variable
            n_trials: Number of optimization trials
            
        Returns:
            Dictionary containing best parameters and model
        """
        self.logger.info("Optimizing Elastic Net parameters...")
        
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda trial: self.objective_elastic_net(trial, X, y), n_trials=n_trials)
        
        best_params = study.best_params
        best_model = ElasticNet(**best_params, random_state=42)
        best_model.fit(X, y)
        
        self.logger.info(f"Best parameters: {best_params}")
        
        return {
            'params': best_params,
            'model': best_model
        }
    
    def objective_xgboost(self, trial, X: pd.DataFrame, y: pd.Series, cv_splits: int = 5) -> float:
        """
        Objective function for XGBoost parameter optimization using Optuna.
        
        Args:
            trial: Optuna trial object
            X: Feature matrix
            y: Target variable
            cv_splits: Number of cross-validation splits
            
        Returns:
            Mean cross-validation score
        """
        # Define parameter space
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
            'gamma': trial.suggest_float('gamma', 0.0, 0.5),
            'random_state': 42
        }
        
        # Initialize model
        model = xgb.XGBRegressor(**params)
        
        # Cross-validation
        cv = KFold(n_splits=cv_splits, shuffle=True, random_state=42)
        scores = []
        
        for train_idx, val_idx in cv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            score = mean_squared_error(y_val, y_pred)
            scores.append(score)
        
        return np.mean(scores)
    
    def optimize_xgboost(self, X: pd.DataFrame, y: pd.Series, n_trials: int = 100) -> Dict[str, Any]:
        """
        Optimize XGBoost parameters using Optuna.
        
        Args:
            X: Feature matrix
            y: Target variable
            n_trials: Number of optimization trials
            
        Returns:
            Dictionary containing best parameters and model
        """
        self.logger.info("Optimizing XGBoost parameters...")
        
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda trial: self.objective_xgboost(trial, X, y), n_trials=n_trials)
        
        best_params = study.best_params
        best_model = xgb.XGBRegressor(**best_params, random_state=42)
        best_model.fit(X, y)
        
        self.logger.info(f"Best parameters: {best_params}")
        
        return {
            'params': best_params,
            'model': best_model
        }
    
    def train_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Train both Elastic Net and XGBoost models.
        
        Args:
            X: Feature matrix
            y: Target variable
            
        Returns:
            Dictionary containing trained models and their parameters
        """
        self.logger.info("Training models...")
        
        # Train Elastic Net
        elastic_net_result = self.optimize_elastic_net(X, y)
        self.elastic_net_model = elastic_net_result['model']
        
        # Train XGBoost
        xgboost_result = self.optimize_xgboost(X, y)
        self.xgboost_model = xgboost_result['model']
        
        return {
            'elastic_net': elastic_net_result,
            'xgboost': xgboost_result
        }
    
    def predict(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Make predictions using both models.
        
        Args:
            X: Feature matrix
            
        Returns:
            Dictionary containing predictions from each model
        """
        if self.elastic_net_model is None or self.xgboost_model is None:
            raise ValueError("Models have not been trained yet")
        
        return {
            'elastic_net': self.elastic_net_model.predict(X),
            'xgboost': self.xgboost_model.predict(X)
        }
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            X: Feature matrix
            y: Target variable
            
        Returns:
            Dictionary containing evaluation metrics for each model
        """
        if self.elastic_net_model is None or self.xgboost_model is None:
            raise ValueError("Models have not been trained yet")
        
        predictions = self.predict(X)
        
        return {
            'elastic_net': mean_squared_error(y, predictions['elastic_net']),
            'xgboost': mean_squared_error(y, predictions['xgboost'])
        } 
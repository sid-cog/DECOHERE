#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Feature selector for selecting important features using SHAP.
"""

import os
import pandas as pd
import numpy as np
import logging
import shap
import xgboost as xgb
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path


class FeatureSelector:
    """
    Feature selector for selecting important features using SHAP.
    """
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Initialize the feature selector.
        
        Args:
            config: Configuration dictionary
            logger: Logger to use (if None, create a new logger)
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Extract configuration parameters
        self.features_base_dir = config['data']['features']['base_dir']
        self.features_fundamental_dir = os.path.join(
            self.features_base_dir, 
            config['data']['features']['fundamental']
        )
        
        # Feature selection parameters
        self.method = config['feature_selection'].get('method', 'shap_threshold')
        self.min_threshold = config['feature_selection'].get('min_threshold', 0.01)
        self.min_features = config['feature_selection'].get('min_features', 10)
        self.max_features = config['feature_selection'].get('max_features', 40)
        self.cumulative_threshold = config['feature_selection'].get('cumulative_threshold', 0.95)
        self.use_cumulative = config['feature_selection'].get('use_cumulative', True)
        
        # Create results directory
        self.results_dir = config.get('output', {}).get('results_dir', 'results/feature_selection')
        os.makedirs(self.results_dir, exist_ok=True)
    
    def load_features(self, date: Optional[str] = None) -> pd.DataFrame:
        """
        Load features from a file.
        
        Args:
            date: Date to load features for (if None, load combined features)
            
        Returns:
            DataFrame containing features
        """
        if date:
            # Load features for a specific date
            file_path = os.path.join(self.features_fundamental_dir, f"fundamental_features_{date}.pq")
        else:
            # Load combined features
            file_path = os.path.join(self.features_fundamental_dir, "fundamental_features.pq")
        
        # Check if file exists
        if not os.path.exists(file_path):
            self.logger.warning(f"Features file not found: {file_path}")
            return pd.DataFrame()
        
        # Load features
        self.logger.info(f"Loading features from {file_path}")
        features_df = pd.read_parquet(file_path)
        
        self.logger.info(f"Loaded features with shape: {features_df.shape}")
        
        return features_df
    
    def prepare_data(self, features_df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare data for feature selection.
        
        Args:
            features_df: DataFrame containing features
            target_col: Target column name
            
        Returns:
            Tuple of (X, y) for feature selection
        """
        self.logger.info(f"Preparing data for feature selection with target: {target_col}")
        
        # Check if target column exists
        if target_col not in features_df.columns:
            self.logger.error(f"Target column not found: {target_col}")
            raise ValueError(f"Target column not found: {target_col}")
        
        # Get feature columns (exclude id, pit_date, and target columns)
        feature_cols = [col for col in features_df.columns 
                      if col not in ['id', 'pit_date'] 
                      and not col.endswith('_TARGET')]
        
        # Remove columns with all NaN values
        valid_cols = []
        for col in feature_cols:
            if not features_df[col].isna().all():
                valid_cols.append(col)
            else:
                self.logger.warning(f"Dropping column with all NaN values: {col}")
        
        # Create X and y
        X = features_df[valid_cols].copy()
        y = features_df[target_col].copy()
        
        # Drop rows with NaN in target
        mask = ~y.isna()
        X = X.loc[mask]
        y = y.loc[mask]
        
        # Fill remaining NaN values in features with 0
        X = X.fillna(0)
        
        self.logger.info(f"Prepared data with {X.shape[1]} features and {X.shape[0]} samples")
        
        return X, y
    
    def shap_threshold_selection(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Select features using SHAP importance threshold.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            
        Returns:
            Tuple of (selected_features_df, results_dict)
        """
        self.logger.info("Selecting features using SHAP threshold method")
        
        # Train an XGBoost model
        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42
        )
        
        # Fit the model
        self.logger.info("Training XGBoost model for SHAP values")
        model.fit(X, y)
        
        # Calculate SHAP values
        self.logger.info("Calculating SHAP values")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        
        # Calculate feature importance
        feature_importance = np.abs(shap_values).mean(0)
        
        # Create importance DataFrame
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': feature_importance
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        # Calculate cumulative importance
        importance_df['normalized_importance'] = importance_df['importance'] / importance_df['importance'].sum()
        importance_df['cumulative_importance'] = importance_df['normalized_importance'].cumsum()
        
        # Save importance DataFrame
        importance_file = os.path.join(self.results_dir, "shap_threshold_importance.csv")
        importance_df.to_csv(importance_file, index=False)
        
        # Select features based on threshold and cumulative importance
        if self.use_cumulative:
            # Select features with importance above threshold
            above_threshold = importance_df[importance_df['importance'] >= self.min_threshold]
            
            # Ensure we have at least min_features
            if len(above_threshold) < self.min_features:
                selected_features = importance_df.head(self.min_features)['feature'].tolist()
            else:
                # Select features based on cumulative importance
                cumulative_selected = importance_df[importance_df['cumulative_importance'] <= self.cumulative_threshold]
                
                # Ensure we have at least min_features
                if len(cumulative_selected) < self.min_features:
                    selected_features = importance_df.head(self.min_features)['feature'].tolist()
                else:
                    selected_features = cumulative_selected['feature'].tolist()
        else:
            # Select top max_features features
            selected_features = importance_df.head(self.max_features)['feature'].tolist()
        
        # Create selected features DataFrame
        selected_df = X[selected_features].copy()
        
        # Create results dictionary
        results = {
            'importance_df': importance_df,
            'selected_features': selected_features,
            'n_features': len(selected_features),
            'cumulative_importance': importance_df.loc[importance_df['feature'].isin(selected_features), 'cumulative_importance'].max()
        }
        
        self.logger.info(f"Selected {len(selected_features)} features with cumulative importance: {results['cumulative_importance']:.4f}")
        
        return selected_df, results
    
    def visualize_feature_importance(self, importance_df: pd.DataFrame, selected_features: List[str]) -> None:
        """
        Visualize feature importance.
        
        Args:
            importance_df: DataFrame containing feature importance
            selected_features: List of selected features
        """
        self.logger.info("Visualizing feature importance")
        
        # Create visualizations directory
        vis_dir = os.path.join(self.results_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        
        # Plot feature importance
        plt.figure(figsize=(12, 8))
        plt.barh(
            importance_df.head(30)['feature'],
            importance_df.head(30)['importance'],
            color=['#1f77b4' if f in selected_features else '#d62728' for f in importance_df.head(30)['feature']]
        )
        plt.xlabel('SHAP Importance')
        plt.ylabel('Feature')
        plt.title('Top 30 Features by SHAP Importance')
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, "feature_importance.png"))
        plt.close()
        
        # Plot cumulative importance
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(importance_df) + 1), importance_df['cumulative_importance'], 'b-')
        plt.axhline(y=self.cumulative_threshold, color='r', linestyle='--', label=f'Threshold: {self.cumulative_threshold}')
        plt.axvline(x=len(selected_features), color='g', linestyle='--', label=f'Selected: {len(selected_features)}')
        plt.xlabel('Number of Features')
        plt.ylabel('Cumulative Importance')
        plt.title('Cumulative Feature Importance')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, "cumulative_importance.png"))
        plt.close()
    
    def create_summary(self, results: Dict[str, Any], target_col: str) -> None:
        """
        Create a summary of the feature selection results.
        
        Args:
            results: Dictionary containing feature selection results
            target_col: Target column name
        """
        self.logger.info("Creating feature selection summary")
        
        # Get importance DataFrame
        importance_df = results['importance_df']
        selected_features = results['selected_features']
        
        # Create summary file
        summary_file = os.path.join(self.results_dir, "summary.txt")
        
        with open(summary_file, 'w') as f:
            f.write(f"SHAP Feature Selection Summary for {target_col}\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Method: {self.method}\n")
            f.write(f"Minimum Threshold: {self.min_threshold}\n")
            f.write(f"Minimum Features: {self.min_features}\n")
            f.write(f"Maximum Features: {self.max_features}\n")
            f.write(f"Cumulative Threshold: {self.cumulative_threshold}\n")
            f.write(f"Use Cumulative: {self.use_cumulative}\n\n")
            
            f.write(f"Selected {len(selected_features)} features with cumulative importance: {results['cumulative_importance']:.4f}\n\n")
            
            f.write("Top 20 Features by Importance:\n")
            f.write("-" * 50 + "\n")
            
            for i, (feature, importance) in enumerate(zip(importance_df.head(20)['feature'], importance_df.head(20)['importance']), 1):
                selected = "✓" if feature in selected_features else " "
                f.write(f"{i:2d}. [{selected}] {feature:40s} | {importance:.6f}\n")
            
            f.write("\n")
            
            # Group selected features by type
            feature_types = {}
            for feature in selected_features:
                if "_DIFF_" in feature:
                    feature_type = "Difference"
                elif "_RAW_" in feature:
                    feature_type = "Raw Ratio"
                elif "_STDDEV_" in feature:
                    feature_type = "Standard Deviation"
                else:
                    feature_type = "Other"
                
                if feature_type not in feature_types:
                    feature_types[feature_type] = []
                
                feature_types[feature_type].append(feature)
            
            f.write("Selected Features by Type:\n")
            f.write("-" * 50 + "\n")
            
            for feature_type, features in feature_types.items():
                f.write(f"{feature_type}: {len(features)} features\n")
                
                for feature in features:
                    importance = importance_df.loc[importance_df['feature'] == feature, 'importance'].values[0]
                    f.write(f"  - {feature:40s} | {importance:.6f}\n")
                
                f.write("\n")
        
        self.logger.info(f"Saved summary to {summary_file}")
    
    def save_selected_features(self, selected_df: pd.DataFrame, features_df: pd.DataFrame, 
                             target_col: str, date: Optional[str] = None) -> str:
        """
        Save selected features to a file.
        
        Args:
            selected_df: DataFrame containing selected features
            features_df: Original features DataFrame
            target_col: Target column name
            date: Date of the features (if None, use combined features)
            
        Returns:
            Path to the saved file
        """
        self.logger.info("Saving selected features")
        
        # Create a copy of the selected features DataFrame
        result_df = selected_df.copy()
        
        # Add id, pit_date, and target columns
        result_df['id'] = features_df['id']
        result_df['pit_date'] = features_df['pit_date']
        result_df[target_col] = features_df[target_col]
        
        # Define output file path
        if date:
            output_file = os.path.join(self.results_dir, f"selected_features_{date}.pq")
        else:
            output_file = os.path.join(self.results_dir, "selected_features.pq")
        
        # Save to parquet file
        result_df.to_parquet(output_file)
        
        self.logger.info(f"Saved selected features to {output_file}")
        
        return output_file
    
    def select_features(self, date: Optional[str] = None, target_col: str = 'PX_TO_BOOK_RATIO_TARGET') -> str:
        """
        Select features using the specified method.
        
        Args:
            date: Date to select features for (if None, use combined features)
            target_col: Target column name
            
        Returns:
            Path to the saved selected features file
        """
        self.logger.info(f"Selecting features for target: {target_col}")
        
        # Load features
        features_df = self.load_features(date)
        
        # Prepare data
        X, y = self.prepare_data(features_df, target_col)
        
        # Select features using the specified method
        if self.method == 'shap_threshold':
            selected_df, results = self.shap_threshold_selection(X, y)
        else:
            self.logger.error(f"Unsupported feature selection method: {self.method}")
            raise ValueError(f"Unsupported feature selection method: {self.method}")
        
        # Visualize feature importance
        self.visualize_feature_importance(results['importance_df'], results['selected_features'])
        
        # Create summary
        self.create_summary(results, target_col)
        
        # Save selected features
        output_file = self.save_selected_features(selected_df, features_df, target_col, date)
        
        return output_file 
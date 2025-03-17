#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Visualizer for the DECOHERE pipeline results.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from typing import Dict, List, Tuple, Optional, Any, Union
import logging


class Visualizer:
    """
    Visualizer for the DECOHERE pipeline results.
    """
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Initialize the visualizer.
        
        Args:
            config: Configuration dictionary
            logger: Logger to use (if None, create a new logger)
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Set default style
        sns.set_style('whitegrid')
        plt.rcParams['figure.figsize'] = (12, 8)
    
    def plot_feature_importance(self, model: Any, feature_names: List[str], top_n: int = 20) -> None:
        """
        Plot feature importance for a trained model.
        
        Args:
            model: Trained model
            feature_names: List of feature names
            top_n: Number of top features to display
        """
        self.logger.info("Plotting feature importance")
        
        # Get feature importance
        if hasattr(model, 'feature_importances_'):
            # Tree-based models
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            # Linear models
            importance = np.abs(model.coef_)
        else:
            self.logger.warning("Model does not have feature_importances_ or coef_ attribute")
            return
        
        # Create DataFrame for plotting
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        })
        
        # Sort by importance and take top N
        importance_df = importance_df.sort_values('Importance', ascending=False).head(top_n)
        
        # Plot
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=importance_df)
        plt.title('Feature Importance')
        plt.tight_layout()
    
    def plot_shap_values(self, model: Any, X: pd.DataFrame, max_display: int = 20) -> None:
        """
        Plot SHAP values for a trained model.
        
        Args:
            model: Trained model
            X: Feature data
            max_display: Maximum number of features to display
        """
        self.logger.info("Plotting SHAP values")
        
        try:
            # Create SHAP explainer
            if hasattr(model, 'predict_proba'):
                explainer = shap.Explainer(model)
            else:
                explainer = shap.Explainer(model)
            
            # Calculate SHAP values
            shap_values = explainer(X)
            
            # Plot summary
            shap.summary_plot(shap_values, X, max_display=max_display, show=False)
            plt.title('SHAP Feature Importance')
        except Exception as e:
            self.logger.error(f"Error plotting SHAP values: {e}")
            plt.text(0.5, 0.5, f"Error plotting SHAP values: {e}", 
                     horizontalalignment='center', verticalalignment='center')
    
    def plot_actual_vs_predicted(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """
        Plot actual vs predicted values.
        
        Args:
            y_true: True target values
            y_pred: Predicted target values
        """
        self.logger.info("Plotting actual vs predicted values")
        
        # Create DataFrame for plotting
        results_df = pd.DataFrame({
            'Actual': y_true,
            'Predicted': y_pred
        })
        
        # Plot
        plt.figure(figsize=(10, 6))
        plt.scatter(results_df['Actual'], results_df['Predicted'], alpha=0.5)
        
        # Add perfect prediction line
        min_val = min(results_df['Actual'].min(), results_df['Predicted'].min())
        max_val = max(results_df['Actual'].max(), results_df['Predicted'].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title('Actual vs Predicted Values')
        plt.tight_layout()
    
    def plot_residuals(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """
        Plot residuals.
        
        Args:
            y_true: True target values
            y_pred: Predicted target values
        """
        self.logger.info("Plotting residuals")
        
        # Calculate residuals
        residuals = y_true - y_pred
        
        # Create DataFrame for plotting
        results_df = pd.DataFrame({
            'Predicted': y_pred,
            'Residuals': residuals
        })
        
        # Plot
        plt.figure(figsize=(10, 6))
        plt.scatter(results_df['Predicted'], results_df['Residuals'], alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        
        plt.xlabel('Predicted')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')
        plt.tight_layout()
    
    def plot_feature_distribution(self, data: pd.DataFrame, features: List[str], n_cols: int = 3) -> None:
        """
        Plot distribution of features.
        
        Args:
            data: DataFrame containing the features
            features: List of features to plot
            n_cols: Number of columns in the subplot grid
        """
        self.logger.info("Plotting feature distributions")
        
        # Calculate number of rows needed
        n_features = len(features)
        n_rows = (n_features + n_cols - 1) // n_cols
        
        # Create subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows))
        axes = axes.flatten()
        
        # Plot each feature
        for i, feature in enumerate(features):
            if i < len(axes):
                sns.histplot(data[feature], kde=True, ax=axes[i])
                axes[i].set_title(feature)
        
        # Hide unused subplots
        for i in range(n_features, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
    
    def plot_correlation_matrix(self, data: pd.DataFrame, features: List[str]) -> None:
        """
        Plot correlation matrix of features.
        
        Args:
            data: DataFrame containing the features
            features: List of features to include in the correlation matrix
        """
        self.logger.info("Plotting correlation matrix")
        
        # Calculate correlation matrix
        corr = data[features].corr()
        
        # Plot
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, cmap='coolwarm', annot=True, fmt='.2f', 
                    square=True, linewidths=0.5)
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
    
    def save_plots(self, output_dir: str, formats: List[str] = ['png', 'pdf']) -> None:
        """
        Save all open plots to files.
        
        Args:
            output_dir: Directory to save plots to
            formats: List of formats to save plots in
        """
        self.logger.info(f"Saving plots to {output_dir}")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save all open figures
        figures = [plt.figure(i) for i in plt.get_fignums()]
        for i, fig in enumerate(figures):
            for fmt in formats:
                filename = os.path.join(output_dir, f'plot_{i}.{fmt}')
                fig.savefig(filename, bbox_inches='tight')
                self.logger.info(f"Saved plot to {filename}") 
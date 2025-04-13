import optuna
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
import numpy as np
import pandas as pd
import logging
from typing import List, Set, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Helper function for pairwise Jaccard ---
def calculate_average_pairwise_jaccard(list_of_sets: List[Set[str]]) -> float:
    """Calculates the average Jaccard similarity between all pairs of sets in a list."""
    if len(list_of_sets) < 2:
        logger.warning("Cannot calculate pairwise Jaccard with less than 2 sets. Returning 0.0")
        return 0.0

    total_jaccard = 0
    num_pairs = 0
    for i in range(len(list_of_sets)):
        for j in range(i + 1, len(list_of_sets)):
            set1 = list_of_sets[i]
            set2 = list_of_sets[j]
            
            # Handle potential empty sets
            if not set1 and not set2:
                jaccard = 1.0 # Treat two empty sets as perfectly similar
            elif not set1 or not set2:
                jaccard = 0.0 # Treat one empty set vs non-empty as dissimilar
            else:
                intersection = len(set1.intersection(set2))
                union = len(set1.union(set2))
                jaccard = intersection / union if union > 0 else 0.0 # Avoid division by zero

            total_jaccard += jaccard
            num_pairs += 1

    average_jaccard = total_jaccard / num_pairs if num_pairs > 0 else 0.0
    logger.debug(f"Calculated average Jaccard for {len(list_of_sets)} sets: {average_jaccard:.4f}")
    return average_jaccard

# --- Optuna Objective Function ---
def objective_stability(trial: optuna.trial.Trial, 
                        X: pd.DataFrame, 
                        y: pd.Series, 
                        feature_names: List[str], 
                        n_splits: int = 5, 
                        k_features: int = 50, 
                        early_stopping_rounds: int = 20) -> float:
    """Optuna objective function to maximize feature set stability."""
    
    logger.info(f"--- Starting Trial {trial.number} ---")

    # 1. Define Hyperparameter Search Space
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'booster': 'gbtree',
        'n_estimators': 1000, # Fixed large number, early stopping handles it
        'learning_rate': trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int("max_depth", 3, 10),
        'subsample': trial.suggest_float("subsample", 0.6, 1.0), # Start slightly higher maybe
        'colsample_bytree': trial.suggest_float("colsample_bytree", 0.6, 1.0), # Start slightly higher
        'gamma': trial.suggest_float("gamma", 0, 5),
        'reg_alpha': trial.suggest_float("reg_alpha", 1e-5, 1.0, log=True), # Wider range
        'reg_lambda': trial.suggest_float("reg_lambda", 1e-5, 1.0, log=True), # Wider range
        'min_child_weight': trial.suggest_int("min_child_weight", 1, 10),
        'verbosity': 0, # Suppress XGBoost output during tuning
        'seed': 42 # For reproducibility within a trial run
    }
    logger.debug(f"Trial {trial.number} Params: {params}")

    # 2. TimeSeriesSplit Cross-Validation
    tscv = TimeSeriesSplit(n_splits=n_splits)
    selected_feature_sets: List[Set[str]] = []
    validation_rmses: List[float] = []

    for fold, (train_index, val_index) in enumerate(tscv.split(X)):
        logger.debug(f"Trial {trial.number}, Fold {fold+1}/{n_splits} starting...")
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        # Ensure indices are aligned if needed (usually fine with iloc)
        if not X_train.index.equals(y_train.index) or not X_val.index.equals(y_val.index):
             logger.warning(f"Index mismatch detected in Trial {trial.number}, Fold {fold+1}. Realigning.")
             y_train = y_train.loc[X_train.index]
             y_val = y_val.loc[X_val.index]

        # 3. Train XGBoost with Early Stopping
        model = xgb.XGBRegressor(**params)
        try:
            model.fit(X_train, y_train,
                      eval_set=[(X_val, y_val)],
                      early_stopping_rounds=early_stopping_rounds,
                      verbose=False)
            
            # Store validation performance (optional, but good for monitoring)
            results = model.evals_result()
            best_rmse = results['validation_0']['rmse'][model.best_iteration]
            validation_rmses.append(best_rmse)
            logger.debug(f"Trial {trial.number}, Fold {fold+1} fitted. Best Iter: {model.best_iteration}, Val RMSE: {best_rmse:.4f}")

        except Exception as e:
            logger.error(f"Error during model fitting in Trial {trial.number}, Fold {fold+1}: {e}", exc_info=True)
            # Decide how to handle errors: return a very low stability score? Skip trial?
            return -1.0 # Indicate failure to Optuna

        # 4. Extract Top K Features (using 'gain' importance)
        try:
             # Use property directly if available and correct type
             importances = model.feature_importances_
             if importances is None or len(importances) != len(feature_names):
                 logger.warning(f"Trial {trial.number}, Fold {fold+1}: Invalid importances detected (None or length mismatch). Skipping fold for stability.")
                 continue # Skip this fold for stability calculation
                 
             feature_importance_map = dict(zip(feature_names, importances))

             # Sort features by importance (descending)
             sorted_features = sorted(feature_importance_map.items(), key=lambda item: item[1], reverse=True)

             # Select top k features
             # Handle cases where k_features > actual number of features with importance > 0
             top_k_features = set([
                 feature for feature, importance in sorted_features 
                 if importance > 1e-9 # Only consider features with non-trivial importance
             ][:k_features])
             
             if not top_k_features:
                 logger.warning(f"Trial {trial.number}, Fold {fold+1}: No features selected with importance > 0. Adding empty set.")
             
             selected_feature_sets.append(top_k_features)
             logger.debug(f"Trial {trial.number}, Fold {fold+1}: Selected {len(top_k_features)} features.")

        except Exception as e:
            logger.error(f"Error extracting features in Trial {trial.number}, Fold {fold+1}: {e}", exc_info=True)
            # Skip this fold if feature extraction fails
            continue

    # 5. Calculate Average Pairwise Jaccard Index
    if not selected_feature_sets:
        logger.warning(f"Trial {trial.number}: No feature sets collected across folds. Returning low stability.")
        return 0.0 # Or handle as failure (-1.0)

    average_jaccard = calculate_average_pairwise_jaccard(selected_feature_sets)
    
    # Log average validation RMSE for info (optional)
    avg_rmse = np.mean(validation_rmses) if validation_rmses else float('inf')
    logger.info(f"--- Finished Trial {trial.number} --- Avg Jaccard: {average_jaccard:.4f}, Avg Val RMSE: {avg_rmse:.4f}")

    return average_jaccard # Optuna will maximize this

# --- Main Execution Function ---
def run_tuning(X_20day: pd.DataFrame, y_20day: pd.Series, 
               feature_names: List[str], 
               n_trials: int = 50, 
               n_splits: int = 5, 
               k_features: int = 50,
               study_name: str = "xgboost_feature_stability",
               final_model_early_stopping_split: float = 0.1):
    """Runs the Optuna study and extracts the final robust feature set."""

    logger.info(f"Starting Optuna study '{study_name}' with {n_trials} trials.")
    study = optuna.create_study(direction="maximize", study_name=study_name, load_if_exists=True)
    
    # Define the objective function with fixed data arguments
    objective_func = lambda trial: objective_stability(
        trial, X_20day, y_20day, feature_names, n_splits=n_splits, k_features=k_features
    )
    
    study.optimize(objective_func, n_trials=n_trials, timeout=None, gc_after_trial=True) # Add timeout if needed

    logger.info(f"Optuna study finished. Best trial: {study.best_trial.number}")
    logger.info(f"  Best Value (Avg Jaccard): {study.best_trial.value:.4f}")
    logger.info("  Best Params: ")
    best_params = study.best_trial.params
    for key, value in best_params.items():
        logger.info(f"    {key}: {value}")

    # --- Rerun with best params on full data to get the final feature set ---
    logger.info("\nTraining final model on full 20-day data with best hyperparameters...")
    
    final_params = {**best_params, 'n_estimators': 2000} # Use larger n_estimators for final model
    
    # Option 2: Use a time-based split of the 20-day data for final early stopping
    split_index = int(len(X_20day) * (1 - final_model_early_stopping_split))
    X_train_final, X_val_final = X_20day.iloc[:split_index], X_20day.iloc[split_index:]
    y_train_final, y_val_final = y_20day.iloc[:split_index], y_20day.iloc[split_index:]
    
    logger.info(f"Using final {final_model_early_stopping_split*100:.0f}% of data for early stopping in final model.")

    final_model = xgb.XGBRegressor(**final_params, verbosity=1, seed=42) # Show verbosity for final fit
    
    try:
        final_model.fit(X_train_final, y_train_final,
                        eval_set=[(X_val_final, y_val_final)],
                        early_stopping_rounds=50, # Use a larger patience for final model
                        verbose=True) # See the training progress
                        
        logger.info(f"Final model trained. Best iteration: {final_model.best_iteration}")
        
        # Use the model at the best iteration
        best_iteration = final_model.best_iteration 
        # Re-instantiate model and fit up to best_iteration on FULL data? 
        # Or just use the importances from the early-stopped model?
        # Using importances from early-stopped model is simpler and often sufficient.
        final_importances = final_model.feature_importances_

        if final_importances is None or len(final_importances) != len(feature_names):
             logger.error("Failed to get valid feature importances from the final model.")
             return None, None # Indicate failure

        final_feature_importance_map = dict(zip(feature_names, final_importances))
        sorted_final_features = sorted(final_feature_importance_map.items(), key=lambda item: item[1], reverse=True)
        
        robust_feature_set = [
            feature for feature, importance in sorted_final_features 
            if importance > 1e-9 # Filter potentially zero importance features
        ][:k_features] # Apply final k cut-off

        logger.info(f"\nFinal robust feature set (Top {len(robust_feature_set)}):")
        for i, feature in enumerate(robust_feature_set):
             logger.info(f"{i+1}. {feature}")
             
        return robust_feature_set, best_params

    except Exception as e:
        logger.error(f"Error during final model training or feature extraction: {e}", exc_info=True)
        return None, None # Indicate failure


if __name__ == '__main__':
    # --- Example Usage ---
    # This section needs to be adapted to load your actual data
    logger.info("Setting up example data (replace with actual data loading)")
    
    # Create dummy data for demonstration
    n_samples = 1000
    n_features = 200
    n_days = 20
    samples_per_day = n_samples // n_days
    
    # Generate features (replace with your actual feature matrix)
    X_20day = pd.DataFrame(np.random.rand(n_samples, n_features), 
                         columns=[f'feature_{i+1}' for i in range(n_features)])
    
    # Generate time index (e.g., representing days)
    time_index = pd.to_datetime(np.repeat(pd.date_range(end='2024-01-20', periods=n_days, freq='D'), samples_per_day))
    # Ensure the index length matches the number of samples
    if len(time_index) > n_samples:
        time_index = time_index[:n_samples]
    elif len(time_index) < n_samples:
        # Adjust if necessary, e.g., repeat the last day's index
        last_day_index = time_index[-1]
        time_index = time_index.append(pd.Index([last_day_index] * (n_samples - len(time_index))))
        
    X_20day.index = time_index # Assign time-based index for TimeSeriesSplit
    
    # Generate target variable (replace with your actual target)
    # Example: target depends on a few features + noise
    relevant_features = np.random.choice(n_features, 10, replace=False)
    y_20day = X_20day.iloc[:, relevant_features].sum(axis=1) + np.random.randn(n_samples) * 0.5
    y_20day.index = X_20day.index # Ensure target index matches features
    
    feature_names = list(X_20day.columns)

    logger.info(f"Example Data Shape: X={X_20day.shape}, y={y_20day.shape}")
    logger.info(f"Example Index Range: {X_20day.index.min()} to {X_20day.index.max()}")

    # --- Run the tuning ---
    robust_features, best_hyperparams = run_tuning(
        X_20day=X_20day, 
        y_20day=y_20day, 
        feature_names=feature_names,
        n_trials=10,       # Low number for quick demo, increase significantly (e.g., 50-100+)
        n_splits=3,        # Low number for demo, increase (e.g., 5)
        k_features=30      # Target number of robust features
    )

    if robust_features:
        logger.info("\n--- Tuning Process Completed Successfully ---")
        logger.info(f"Found {len(robust_features)} robust features.")
        # You can now save/use 'robust_features' and 'best_hyperparams'
    else:
        logger.error("\n--- Tuning Process Failed ---") 
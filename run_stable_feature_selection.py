import argparse
import logging
import sys
import os
from pathlib import Path

# Add project root to sys.path to allow importing src modules
project_root = Path(__file__).resolve().parent.parent # Assumes this script is in the project root
sys.path.insert(0, str(project_root))

from src.features.stable_feature_selector import StableFeatureSelector

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Run Stable Feature Selection using Optuna.")

    # Required Arguments
    parser.add_argument("--panel-data-path", type=str, required=True,
                        help="Path to the panel data Parquet file/directory.")
    parser.add_argument("--target-date", type=str, required=True,
                        help="Target date (end of lookback window) in YYYY-MM-DD format.")
    parser.add_argument("--target-column", type=str, required=True,
                        help="Name of the target variable column.")
    parser.add_argument("--k-features", type=int, required=True,
                        help="Total number of stable features desired (including compulsory).")
    parser.add_argument("--compulsory-features", type=str, nargs='+', required=True,
                        help="List of compulsory feature names (e.g., sector_1 sector_2_OHE_A). Separate by space.")

    # Optional Arguments with Defaults from Spec
    parser.add_argument("--lookback-days", type=int, default=20,
                        help="Number of trading days for the lookback window.")
    parser.add_argument("--n-splits", type=int, default=5,
                        help="Number of TimeSeriesSplit folds.")
    parser.add_argument("--n-trials", type=int, default=50,
                        help="Number of Optuna trials.")
    parser.add_argument("--rmse-quantile", type=float, default=0.75,
                        help="RMSE quantile threshold for filtering trials.")
    parser.add_argument("--optuna-jobs", type=int, default=12,
                        help="Number of parallel jobs for Optuna study.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")
    parser.add_argument("--results-dir", type=str, default="results/stable_features",
                        help="Base directory to save selection results.")

    args = parser.parse_args()

    logger.info("--- Starting Stable Feature Selection Run ---")
    logger.info(f"Command line arguments: {vars(args)}")

    # Instantiate the selector
    selector = StableFeatureSelector(
        k_features=args.k_features,
        compulsory_features=args.compulsory_features,
        lookback_days=args.lookback_days,
        n_splits=args.n_splits,
        n_trials=args.n_trials,
        rmse_threshold_quantile=args.rmse_quantile,
        optuna_n_jobs=args.optuna_jobs,
        random_seed=args.seed,
        results_base_dir=args.results_dir
    )

    # Run the main process
    results = selector.tune_and_select_stable_features(
        panel_data_path=args.panel_data_path,
        target_date_str=args.target_date,
        target_column_name=args.target_column
    )

    if results:
        stable_set, best_params, stability_score, rmse_score, ranking = results
        logger.info("--- Stable Feature Selection Completed Successfully ---")
        logger.info(f"Selected {len(stable_set)} features. Stability (Spearman): {stability_score:.4f}, RMSE: {rmse_score:.4f}")
        # logger.info(f"Best Params: {best_params}") # Can be verbose
        # logger.info(f"Feature Set: {stable_set}")
        sys.exit(0) # Indicate success
    else:
        logger.error("--- Stable Feature Selection Failed ---")
        sys.exit(1) # Indicate failure

if __name__ == "__main__":
    main() 
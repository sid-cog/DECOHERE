import logging
import json
from datetime import datetime
from pathlib import Path
from src.features.lightgbm_optimizer import LightGBMOptimizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_stable_features(date_str: str) -> list:
    """Load stable features for the given date."""
    date = datetime.strptime(date_str, "%Y-%m-%d")
    year = date.year
    month = f"{date.month:02d}"
    
    features_file = Path(f"/home/siddharth.johri/DECOHERE/data/features/fundamentals/stable_features/year={year}/month={month}/stable_features_{date_str}.json")
    
    if not features_file.exists():
        raise FileNotFoundError(f"Stable features file not found for date {date_str}")
    
    with open(features_file, 'r') as f:
        stable_features = json.load(f)
    
    return stable_features

def save_results(date_str: str, best_params: dict, best_rmse: float):
    """Save optimization results to a JSON file."""
    results = {
        'date': date_str,
        'best_params': best_params,
        'best_rmse': best_rmse
    }
    
    output_dir = Path("/home/siddharth.johri/DECOHERE/data/models/lightgbm_optimization")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"lightgbm_optimization_results_{date_str}.json"
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    logger.info(f"Results saved to {output_file}")

def main():
    # Set target date
    target_date = "2024-03-01"
    
    try:
        # Load stable features
        logger.info(f"Loading stable features for date {target_date}")
        stable_features = load_stable_features(target_date)
        logger.info(f"Loaded {len(stable_features)} stable features")
        
        # Run optimization
        optimizer = LightGBMOptimizer(n_trials=100, n_splits=5, lookback_days=20)
        best_params, best_rmse = optimizer.optimize(target_date, stable_features)
        
        # Save results
        save_results(target_date, best_params, best_rmse)
        
    except Exception as e:
        logger.error(f"Error in optimization process: {str(e)}")
        raise

if __name__ == "__main__":
    main() 
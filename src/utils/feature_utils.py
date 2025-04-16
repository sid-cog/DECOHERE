"""Utility functions for feature operations."""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

def load_stable_features(date_str: str) -> List[str]:
    """Load stable features for the given date from metadata JSON file.
    
    Args:
        date_str: Date string in format 'YYYY-MM-DD'
        
    Returns:
        List of stable feature names
        
    Raises:
        FileNotFoundError: If metadata file doesn't exist
        Exception: For other loading errors
    """
    date = datetime.strptime(date_str, "%Y-%m-%d")
    year = date.year
    month = f"{date.month:02d}"
    
    metadata_file = Path(f"/home/siddharth.johri/DECOHERE/data/features/fundamentals/stable_features/year={year}/month={month}/metadata_{date_str}.json")
    
    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
    
    try:
        # Load the metadata JSON file
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Extract stable feature set
        stable_features = metadata['stable_feature_set']
        logger.info(f"Loaded {len(stable_features)} stable features from {metadata_file}")
        
        return stable_features
        
    except Exception as e:
        logger.error(f"Error loading stable features: {str(e)}")
        raise

def save_optimization_results(date_str: str, best_params: Dict[str, Any], best_rmse: float) -> Path:
    """Save optimization results to a JSON file.
    
    Args:
        date_str: Date string in format 'YYYY-MM-DD'
        best_params: Dictionary of best parameters
        best_rmse: Best RMSE value
        
    Returns:
        Path to the saved results file
    """
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
    return output_file 
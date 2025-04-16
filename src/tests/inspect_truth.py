"""
Simple script to inspect the source of truth file.
"""

import pandas as pd
import logging

def setup_logging():
    """Configure logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger('InspectTruth')

def main():
    """Main function"""
    logger = setup_logging()
    
    # Test date
    test_date = "2024-09-02"
    year = test_date[:4]
    month = test_date[5:7]
    truth_path = f"/home/siddharth.johri/DECOHERE/data/features/fundamentals/enhanced_features/year={year}/month={month}/data_{test_date} copy.pq"
    
    try:
        # Load source of truth
        logger.info(f"Loading source of truth from {truth_path}")
        df = pd.read_parquet(truth_path)
        
        # Display basic information
        logger.info(f"\nDataFrame shape: {df.shape}")
        logger.info(f"\nColumns: {df.columns.tolist()}")
        logger.info(f"\nData types:\n{df.dtypes}")
        
        # Display sample of numerical columns
        numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
        logger.info(f"\nSample values from numerical columns:")
        for col in numerical_cols[:5]:  # Show first 5 numerical columns
            logger.info(f"\n{col}:")
            logger.info(df[col].describe())
        
    except Exception as e:
        logger.error(f"Error loading file: {str(e)}")

if __name__ == "__main__":
    main() 
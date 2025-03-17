#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to update the FeatureGenerator class to use sector and returns data.
"""

import os
import sys
import logging
import shutil
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger('update_feature_generator')

def main():
    """
    Main function to update the FeatureGenerator class.
    """
    logger.info("Starting to update the FeatureGenerator class...")
    
    # Define paths
    feature_generator_path = "src/features/feature_generator.py"
    backup_path = f"{feature_generator_path}.bak_data_sources"
    
    # Create backup
    logger.info(f"Creating backup at {backup_path}")
    shutil.copy2(feature_generator_path, backup_path)
    
    # Read current implementation
    logger.info("Reading current feature generator implementation")
    with open(feature_generator_path, 'r') as f:
        content = f.read()
    
    # Update the generate_features method to accept sector and returns data
    generate_features_pattern = """    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        \"\"\"
        Generate features from processed data.
        
        Args:
            df: DataFrame containing processed data
            
        Returns:
            DataFrame containing generated features
        \"\"\"
        self.logger.info(f"Generating features from data with shape: {df.shape}")
        
        # Create a copy of the DataFrame to avoid modifying the original
        df_copy = df.copy()"""
    
    updated_generate_features = """    def generate_features(self, df: pd.DataFrame, sector_data: Optional[pd.DataFrame] = None, 
                       price_returns: Optional[pd.DataFrame] = None, 
                       total_returns: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        \"\"\"
        Generate features from processed data.
        
        Args:
            df: DataFrame containing processed data
            sector_data: DataFrame containing sector mapping data
            price_returns: DataFrame containing price returns data
            total_returns: DataFrame containing total returns data
            
        Returns:
            DataFrame containing generated features
        \"\"\"
        self.logger.info(f"Generating features from data with shape: {df.shape}")
        
        # Create a copy of the DataFrame to avoid modifying the original
        df_copy = df.copy()
        
        # Log information about additional data sources
        if sector_data is not None:
            self.logger.info(f"Using sector data with shape: {sector_data.shape}")
        if price_returns is not None:
            self.logger.info(f"Using price returns data with shape: {price_returns.shape}")
        if total_returns is not None:
            self.logger.info(f"Using total returns data with shape: {total_returns.shape}")"""
    
    content = content.replace(generate_features_pattern, updated_generate_features)
    
    # Add methods for generating sector-based features
    process_features_end = """        # Save features
        file_path = self.save_features(features_df, date)
        
        return file_path"""
    
    new_methods = """        # Save features
        file_path = self.save_features(features_df, date)
        
        return file_path
        
    def generate_sector_features(self, df: pd.DataFrame, sector_data: pd.DataFrame) -> pd.DataFrame:
        \"\"\"
        Generate sector-based features.
        
        Args:
            df: DataFrame containing processed data
            sector_data: DataFrame containing sector mapping data
            
        Returns:
            DataFrame containing sector-based features
        \"\"\"
        self.logger.info("Generating sector-based features")
        
        # Create a copy of the DataFrame to avoid modifying the original
        df_copy = df.copy()
        
        # Create a dictionary to store features
        features = {}
        
        # Ensure sector_data has ID as index
        if not sector_data.index.name == 'ID':
            if 'ID' in sector_data.columns:
                sector_data = sector_data.set_index('ID')
            else:
                self.logger.warning("Sector data does not have ID column, cannot generate sector features")
                return pd.DataFrame(features)
        
        # Get unique IDs from the processed data
        unique_ids = df_copy['ID'].unique()
        
        # Filter sector data to include only IDs in the processed data
        sector_data_filtered = sector_data.loc[sector_data.index.isin(unique_ids)]
        
        # Create sector features
        for id_val in unique_ids:
            if id_val in sector_data_filtered.index:
                # Get sector information for this ID
                sector_info = sector_data_filtered.loc[id_val]
                
                # Add sector information to features
                features[id_val] = {
                    'SECTOR_1': sector_info.get('sector_1', None),
                    'SECTOR_2': sector_info.get('sector_2', None),
                    'SECTOR_3': sector_info.get('sector_3', None),
                    'SECTOR_4': sector_info.get('sector_4', None)
                }
        
        # Convert features to DataFrame
        sector_features_df = pd.DataFrame.from_dict(features, orient='index')
        sector_features_df.index.name = 'ID'
        sector_features_df = sector_features_df.reset_index()
        
        self.logger.info(f"Generated {len(sector_features_df.columns) - 1} sector-based features")
        
        return sector_features_df
    
    def generate_returns_features(self, df: pd.DataFrame, price_returns: pd.DataFrame, total_returns: pd.DataFrame) -> pd.DataFrame:
        \"\"\"
        Generate returns-based features.
        
        Args:
            df: DataFrame containing processed data
            price_returns: DataFrame containing price returns data
            total_returns: DataFrame containing total returns data
            
        Returns:
            DataFrame containing returns-based features
        \"\"\"
        self.logger.info("Generating returns-based features")
        
        # Create a copy of the DataFrame to avoid modifying the original
        df_copy = df.copy()
        
        # Create a dictionary to store features
        features = {}
        
        # Get unique IDs from the processed data
        unique_ids = df_copy['ID'].unique()
        
        # Get unique dates from the processed data
        unique_dates = df_copy['PIT_DATE'].unique()
        
        # Process price returns
        if price_returns is not None:
            self.logger.info("Processing price returns data")
            
            # Ensure price_returns has the necessary columns
            if 'ID' in price_returns.columns and any(col for col in price_returns.columns if 'RETURN' in col or 'RET' in col):
                # Filter price returns to include only IDs in the processed data
                price_returns_filtered = price_returns[price_returns['ID'].isin(unique_ids)]
                
                # Find the return column
                return_cols = [col for col in price_returns.columns if 'RETURN' in col or 'RET' in col]
                
                if return_cols:
                    return_col = return_cols[0]
                    self.logger.info(f"Using price returns column: {return_col}")
                    
                    # Create features for each ID
                    for id_val in unique_ids:
                        id_returns = price_returns_filtered[price_returns_filtered['ID'] == id_val]
                        
                        if not id_returns.empty:
                            # Calculate return statistics
                            mean_return = id_returns[return_col].mean()
                            std_return = id_returns[return_col].std()
                            max_return = id_returns[return_col].max()
                            min_return = id_returns[return_col].min()
                            
                            # Add return statistics to features
                            if id_val not in features:
                                features[id_val] = {}
                            
                            features[id_val].update({
                                'PRICE_RETURN_MEAN': mean_return,
                                'PRICE_RETURN_STD': std_return,
                                'PRICE_RETURN_MAX': max_return,
                                'PRICE_RETURN_MIN': min_return
                            })
        
        # Process total returns
        if total_returns is not None:
            self.logger.info("Processing total returns data")
            
            # Ensure total_returns has the necessary columns
            if 'ID' in total_returns.columns and any(col for col in total_returns.columns if 'RETURN' in col or 'RET' in col):
                # Filter total returns to include only IDs in the processed data
                total_returns_filtered = total_returns[total_returns['ID'].isin(unique_ids)]
                
                # Find the return column
                return_cols = [col for col in total_returns.columns if 'RETURN' in col or 'RET' in col]
                
                if return_cols:
                    return_col = return_cols[0]
                    self.logger.info(f"Using total returns column: {return_col}")
                    
                    # Create features for each ID
                    for id_val in unique_ids:
                        id_returns = total_returns_filtered[total_returns_filtered['ID'] == id_val]
                        
                        if not id_returns.empty:
                            # Calculate return statistics
                            mean_return = id_returns[return_col].mean()
                            std_return = id_returns[return_col].std()
                            max_return = id_returns[return_col].max()
                            min_return = id_returns[return_col].min()
                            
                            # Add return statistics to features
                            if id_val not in features:
                                features[id_val] = {}
                            
                            features[id_val].update({
                                'TOTAL_RETURN_MEAN': mean_return,
                                'TOTAL_RETURN_STD': std_return,
                                'TOTAL_RETURN_MAX': max_return,
                                'TOTAL_RETURN_MIN': min_return
                            })
        
        # Convert features to DataFrame
        returns_features_df = pd.DataFrame.from_dict(features, orient='index')
        returns_features_df.index.name = 'ID'
        returns_features_df = returns_features_df.reset_index()
        
        self.logger.info(f"Generated {len(returns_features_df.columns) - 1} returns-based features")
        
        return returns_features_df"""
    
    content = content.replace(process_features_end, new_methods)
    
    # Update the generate_features method to include sector and returns features
    features_df_pattern = """        # Create a DataFrame from the features dictionary
        features_df = pd.DataFrame.from_dict(features)
        
        # Transpose the DataFrame to have features as columns
        features_df = features_df.T
        
        # Reset the index to have ID as a column
        features_df = features_df.reset_index()
        features_df = features_df.rename(columns={'index': 'ID'})
        
        # Add target variables
        features_df = self.add_target_variables(features_df, df_copy)
        
        self.logger.info(f"Generated features with shape: {features_df.shape}")
        
        return features_df"""
    
    updated_features_df = """        # Create a DataFrame from the features dictionary
        features_df = pd.DataFrame.from_dict(features)
        
        # Transpose the DataFrame to have features as columns
        features_df = features_df.T
        
        # Reset the index to have ID as a column
        features_df = features_df.reset_index()
        features_df = features_df.rename(columns={'index': 'ID'})
        
        # Add target variables
        features_df = self.add_target_variables(features_df, df_copy)
        
        # Add sector features if sector data is provided
        if sector_data is not None:
            sector_features_df = self.generate_sector_features(df_copy, sector_data)
            if not sector_features_df.empty:
                features_df = pd.merge(features_df, sector_features_df, on='ID', how='left')
                self.logger.info(f"Added sector features, new shape: {features_df.shape}")
        
        # Add returns features if returns data is provided
        if price_returns is not None or total_returns is not None:
            returns_features_df = self.generate_returns_features(df_copy, price_returns, total_returns)
            if not returns_features_df.empty:
                features_df = pd.merge(features_df, returns_features_df, on='ID', how='left')
                self.logger.info(f"Added returns features, new shape: {features_df.shape}")
        
        self.logger.info(f"Generated features with shape: {features_df.shape}")
        
        return features_df"""
    
    content = content.replace(features_df_pattern, updated_features_df)
    
    # Write the updated content
    logger.info("Writing updated feature generator implementation")
    with open(feature_generator_path, 'w') as f:
        f.write(content)
    
    logger.info("FeatureGenerator class has been updated to use sector and returns data")

if __name__ == "__main__":
    main() 
import nbformat as nbf

# Read the existing notebook
with open('notebooks/updated_pipe.ipynb', 'r') as f:
    nb = nbf.read(f, as_version=4)

# Update the data loading and processing cell
data_loading_code = """def load_and_process_data(date_str: str, mode: str = 'day'):
    \"\"\"Load and process data for a specific date and mode.\"\"\"
    logger.info(f"Loading and processing data for date: {date_str} in {mode} mode")
    
    try:
        # Load fundamentals data
        fundamentals_df = storage.load_data(
            data_type=DataType.FUNDAMENTALS,
            stage=DataStage.RAW,
            date=date_str,
            mode=mode
        )
        
        if fundamentals_df.empty:
            logger.error(f"No fundamentals data found for date: {date_str}")
            return None
        
        # Load returns data
        returns_df = storage.load_data(
            data_type=DataType.RETURNS,
            stage=DataStage.RAW,
            date=date_str,
            mode=mode
        )
        
        # Process fundamentals data - store intermediate results
        intermediate_data = processor.process_fundamentals(fundamentals_df)
        
        # Store intermediate data
        storage.store_data(
            df=intermediate_data,
            data_type=DataType.FUNDAMENTALS,
            stage=DataStage.INTERMEDIATE,
            date=date_str
        )
        
        # Further process for final processed data
        processed_fundamentals = processor.prepare_for_features(intermediate_data)
        
        # Store processed data
        storage.store_data(
            df=processed_fundamentals,
            data_type=DataType.FUNDAMENTALS,
            stage=DataStage.PROCESSED,
            date=date_str
        )
        
        return processed_fundamentals, intermediate_data, returns_df
        
    except Exception as e:
        logger.error(f"Error in data loading and processing: {e}")
        return None

# Example usage
date_str = datetime.now().strftime('%Y-%m-%d')
processed_data, intermediate_data, returns_data = load_and_process_data(date_str, mode='day')

if processed_data is not None:
    print(f"Processed data shape: {processed_data.shape}")
    print(f"Intermediate data shape: {intermediate_data.shape}")
    print(f"Returns data shape: {returns_data.shape if not returns_data.empty else 'No returns data'}")"""

# Update the feature generation cell
feature_gen_code = """def generate_features(intermediate_data: pd.DataFrame, processed_data: pd.DataFrame, date_str: str):
    \"\"\"Generate features from intermediate and processed data.\"\"\"
    logger.info(f"Generating features for date: {date_str}")
    
    try:
        # Initialize feature generator
        feature_gen = FeatureGenerator(config, logger)
        
        # Generate features using both intermediate and processed data
        features_df = feature_gen.generate_features(intermediate_data, processed_data)
        
        # Store features
        storage.store_data(
            df=features_df,
            data_type=DataType.FUNDAMENTALS,
            stage=DataStage.FEATURES,
            date=date_str
        )
        
        return features_df
        
    except Exception as e:
        logger.error(f"Error in feature generation: {e}")
        return None

# Generate features
if processed_data is not None and intermediate_data is not None:
    features_df = generate_features(intermediate_data, processed_data, date_str)
    if features_df is not None:
        print(f"Generated features shape: {features_df.shape}")"""

# Update the cells in the notebook
for cell in nb.cells:
    if cell.cell_type == 'code':
        if 'def load_and_process_data' in cell.source:
            cell.source = data_loading_code
        elif 'def generate_features' in cell.source:
            cell.source = feature_gen_code

# Write the updated notebook
with open('notebooks/updated_pipe.ipynb', 'w') as f:
    nbf.write(nb, f) 
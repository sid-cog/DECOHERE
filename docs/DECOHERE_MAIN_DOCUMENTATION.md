# DECOHERE

A clean, organized implementation of a quantitative trading pipeline for financial data analysis.

## Overview

DECOHERE (Data Extraction and COmprehensive Handling for Enhanced REsults) is a comprehensive pipeline for processing financial data, generating features, and analyzing financial time series. The pipeline is designed to be modular, efficient, and easy to use through a Jupyter notebook interface.

## Project Structure

```
DECOHERE/
├── config/              # Configuration files
├── data/                # Data storage
│   ├── raw/             # Raw financial data
│   ├── processed/       # Processed data
│   └── features/        # Generated features
├── docs/                # Documentation
├── notebooks/           # Jupyter notebooks
├── src/                 # Source code
│   ├── data/            # Data processing modules
│   │   ├── data_processor.py
│   │   ├── efficient_data_storage.py
│   │   └── load_data.py
│   ├── features/        # Feature engineering modules
│   │   └── feature_generator.py
│   ├── models/          # Model training and evaluation
│   └── utils/           # Utility functions
│       └── logging_utils.py
├── tests/               # Unit tests
├── output/              # Output files
└── scripts/             # Utility scripts
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/DECOHERE.git
cd DECOHERE
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Key Components

### DataProcessor
The `DataProcessor` class in `src/data/data_processor.py` is responsible for processing raw financial data. It performs operations such as:
- Normalizing column names
- Calculating periods based on fiscal months
- Filling missing values using sophisticated interpolation techniques
- Handling outliers through winsorization

### EfficientDataStorage
The `EfficientDataStorage` class in `src/data/efficient_data_storage.py` handles the storage and retrieval of processed data. It:
- Saves processed data in an efficient format (Parquet)
- Loads processed data for feature generation
- Handles data partitioning for efficient access

### FeatureGenerator
The `FeatureGenerator` class in `src/features/feature_generator.py` generates features from processed data. The current implementation:
- Generates a minimal set of features
- Creates random values for demonstration purposes
- Includes a celebratory success message

## Pipeline Modes

The pipeline can be run through the Jupyter notebook interface:

```bash
jupyter notebook notebooks/pipeline.ipynb
```

The pipeline can be run in different modes:
- **Day Mode**: Process data for a single day
- **Week Mode**: Process data for a week
- **Month Mode**: Process data for a month
- **Year Mode**: Process data for a year
- **All Years Mode**: Process all available years of data

## Data Flow
1. **Load Data**: Raw financial data is loaded from the specified source.
2. **Process Data**: The raw data is processed to normalize column names, calculate periods, fill missing values, and handle outliers.
3. **Generate Features**: Features are generated from the processed data.
4. **Store Data**: The processed data and generated features are stored efficiently.

## Configuration

The pipeline can be configured through YAML files in the `config/` directory:
- `config.yaml`: Main configuration file

## Testing

The `test_pipeline.py` script provides a way to test the pipeline functionality. It:
- Loads a sample of raw financial data
- Processes the data
- Generates features
- Verifies the results

To test the pipeline with a specific ticker and date:

```bash
python test_pipeline.py
```

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
# DECOHERE Data Directory

This directory contains the data structure for the DECOHERE project. The actual data files are not included in the Git repository due to their size and potentially sensitive nature.

## Directory Structure

```
data/
├── raw/             # Raw financial data
├── processed/       # Processed data
├── features/        # Generated features
└── sample/          # Small sample datasets for testing (can be committed to Git)
```

## How to Obtain the Data

To use the DECOHERE pipeline, you'll need to obtain the financial data from the appropriate sources:

1. **Internal Data Source**: If you're part of the organization, contact the data team to get access to the financial datasets.

2. **External Data Source**: If you're using publicly available data, you can download it from:
   - [Financial Data Source Name] - [URL]
   - [Alternative Data Source] - [URL]

## Data Setup

Once you have obtained the data files, place them in the appropriate directories:

1. Raw data files should be placed in the `data/raw/` directory
2. If you have pre-processed data, place it in the `data/processed/` directory

## Sample Data

A small sample dataset is provided in the `data/sample/` directory for testing and demonstration purposes. This sample data is included in the Git repository.

To use the sample data:

```bash
# Copy sample data to the raw directory
cp -r data/sample/* data/raw/

# Run the pipeline with sample data
python test_pipeline.py --use-sample-data
```

## Data Handling Guidelines

1. **Large Files**: Do not commit large data files to the Git repository
2. **Sensitive Information**: Ensure any sensitive information is properly anonymized
3. **Data Formats**: The pipeline expects data in the following formats:
   - Raw data: Parquet files (.pq or .parquet)
   - Processed data: Parquet files with specific schema
   - Features: CSV or Parquet files

4. **Data Backup**: Regularly back up your data to a secure location outside the repository

## Creating Sample Data

If you need to create a sample dataset from your full dataset:

```python
import pandas as pd

# Load full dataset
df = pd.read_parquet('data/raw/full_dataset.pq')

# Create a small sample (e.g., 100 rows)
sample_df = df.sample(100, random_state=42)

# Save to sample directory
sample_df.to_parquet('data/sample/sample_dataset.pq')
``` 
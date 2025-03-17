# DECOHERE Technical Documentation

This document provides detailed technical information about specific components and implementations in the DECOHERE project.

## Table of Contents
1. [NaN Handling](#nan-handling)
2. [Feature Generator Updates](#feature-generator-updates)
3. [PIT_DATE Handling Changes](#pit_date-handling-changes)
4. [Efficient Data Storage System](#efficient-data-storage-system)

## NaN Handling

### Overview

The DECOHERE project includes a sophisticated approach to handling missing values (NaNs) in financial time series data. This is implemented in the `_fill_between_valid_values` method of the `DataProcessor` class.

### The Problem

Financial time series data often contains missing values due to:
- Irregular reporting periods
- Data collection issues
- Mergers, acquisitions, or other corporate events

Simply dropping rows with missing values would result in a significant loss of data, while using simple imputation methods (like mean or median) might not preserve the temporal patterns in the data.

### The Solution

The `_fill_between_valid_values` method fills missing values between valid values with equally spaced values, preserving the trend between known data points. This approach is particularly valuable for financial data where maintaining the trend is important.

## Feature Generator Updates

### Overview
This section documents the changes made to the feature generator in the DECOHERE project.

### Original Implementation

The original feature generator implementation:
1. **Complex Feature Generation**: Generated a large number of features based on financial metrics
2. **Period-Based Features**: Created features spanning from negative to positive periods
3. **Statistical Features**: Calculated standard deviations and differences between periods

### Current Implementation

The current feature generator implementation:
1. **Simplified Feature Generation**: Generates a minimal set of features
2. **Random Features**: Creates random values for each metric for demonstration purposes
3. **Success Message**: Includes a success message to celebrate the completion of the cleanup

### Issues Fixed
1. **Missing Periods**: The feature generator was only using periods 0 and 10, missing the negative periods and other positive periods.
2. **Pivot Table Handling**: Fixed issues with missing periods in the pivot table causing KeyErrors.
3. **Column Name Case Sensitivity**: Updated the feature generator to handle uppercase column names correctly.
4. **Raw Financial Ratios**: Fixed the feature generator to use the signed_log versions of metrics for raw financial ratios.

### Changes Made

#### Data Processor
- Updated the `calculate_periods` method to include negative periods, ensuring proper data handling.
- Enhanced the `_fill_between_valid_values` method to accommodate the new period structure, ensuring proper filling of missing values.

#### Feature Generator
- Fixed the `calculate_feature_set` method to handle missing periods in the pivot table.
- Updated the time-series feature generation to check if periods exist in the pivot table before creating features.
- Modified the standard deviation feature generation to filter available periods before calculating statistics.
- Updated the feature generator to use the original uppercase metric names.
- Changed the raw financial ratios to use the signed_log versions of metrics.

## PIT_DATE Handling Changes

### Background

Analysis showed that there are **0 instances** of missing PIT_DATE values in the raw data. All 78,336 rows in the dataset have valid PIT_DATE values. Therefore, the logic to fill in default values for missing PIT_DATE was unnecessary and has been removed.

### Changes Made

1. **examine_raw_data.py**:
   - Removed the conditional check `if 'PIT_DATE' not in ticker_data.columns`
   - Removed the code that added a default PIT_DATE value of '2024-09-11'
   - Updated the step numbering in the processing pipeline
   
2. **src/data/efficient_data_storage.py**:
   - Removed the conditional check `if 'PIT_DATE' not in df.columns`
   - Removed the code that added a default PIT_DATE value based on the date parameter

### Rationale

The changes were made for the following reasons:
1. **Data Integrity**: The raw data already contains valid PIT_DATE values for all rows, so there's no need to add default values.
2. **Code Simplification**: Removing unnecessary conditional checks simplifies the code and makes it easier to maintain.
3. **Consistency**: Using the actual PIT_DATE values from the raw data ensures consistency throughout the data processing pipeline.

## Efficient Data Storage System

### Overview

The efficient data storage system is designed to improve the performance, scalability, and flexibility of data storage and retrieval in the DECOHERE project. It uses partitioned Parquet datasets to efficiently store and query processed financial data.

### Key Features

1. **Partitioned Storage**: Data is partitioned by year and month, allowing for efficient querying of specific time periods.
2. **Multiple Access Modes**: Supports day, week, year, and ALL YEARS modes for flexible data access.
3. **Backward Compatibility**: Maintains compatibility with the existing codebase and legacy storage format.
4. **Efficient Filtering**: Uses PyArrow for efficient filtering of data based on date ranges.
5. **Memory Efficiency**: Reduces memory usage by loading only the required partitions.
6. **Automatic Date Handling**: Automatically handles date ranges based on the selected mode.
7. **Migration Utilities**: Provides utilities to migrate legacy data to the new storage format.

### Components

The efficient storage system consists of the following components:

1. **EfficientDataStorage Class** (`src/data/efficient_data_storage.py`): Core class that implements the efficient storage and retrieval functionality.
2. **DataProcessor Update** (`src/data/update_data_processor.py`): Script to update the DataProcessor class to use the efficient storage system.
3. **Pipeline Update** (`src/data/update_pipeline.py`): Script to update the pipeline notebook to support the ALL YEARS mode.
4. **Configuration Update** (`src/data/update_config.py`): Script to update the configuration file to include the ALL YEARS mode.
5. **Demonstration Script** (`src/data/demo_efficient_storage.py`): Script to demonstrate the features and benefits of the efficient storage system.
6. **Setup Script** (`setup_efficient_storage.py`): Script to set up the efficient storage system.

### Usage

#### Loading Data in Different Modes

You can load data in different modes using the `load_processed_data_by_mode` method of the `DataProcessor` class:

```python
# Initialize the data processor
data_processor = DataProcessor(config, logger)

# Load data in day mode
day_data = data_processor.load_processed_data_by_mode(
    mode='day',
    date='2024-09-11'
)

# Load data in week mode
week_data = data_processor.load_processed_data_by_mode(
    mode='week',
    start_date='2024-09-11',
    end_date='2024-09-17'
)

# Load data in year mode
year_data = data_processor.load_processed_data_by_mode(
    mode='year',
    start_date='2024-01-01',
    end_date='2024-12-31'
)

# Load data in ALL YEARS mode
all_years_data = data_processor.load_processed_data_by_mode(
    mode='all_years'
)
```

### Performance Considerations

The efficient storage system offers several performance benefits:

1. **Faster Queries**: Partitioning allows for faster queries by reading only the relevant partitions.
2. **Reduced Memory Usage**: Loading only the required partitions reduces memory usage.
3. **Scalability**: The system can handle larger datasets by partitioning the data.
4. **Efficient Filtering**: PyArrow provides efficient filtering capabilities. 
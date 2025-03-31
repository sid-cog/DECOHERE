# Efficient Data Storage System

## Overview

The efficient data storage system in DECOHERE is designed to improve the performance, scalability, and flexibility of data storage and retrieval. It uses partitioned Parquet datasets to efficiently store and query processed financial data.

## Storage Structure

```
processed/
├── fundamentals/           # Financial and sector data
│   ├── fundamentals_YYYY_MM.pq  # Monthly partitioned files
│   └── processed_financials.pq  # Consolidated file (legacy)
└── returns/               # Price and total returns data
    ├── price_returns.pq   # Price returns data
    ├── total_returns.pq   # Total returns data
    └── combined_returns.pq # Combined returns data
```

## Key Features

1. **Partitioned Storage**: Data is partitioned by year and month, allowing for efficient querying of specific time periods.
2. **Multiple Access Modes**: Supports day, week, year, and ALL YEARS modes for flexible data access.
3. **Backward Compatibility**: Maintains compatibility with the existing codebase and legacy storage format.
4. **Efficient Filtering**: Uses PyArrow for efficient filtering of data based on date ranges.
5. **Memory Efficiency**: Reduces memory usage by loading only the required partitions.
6. **Automatic Date Handling**: Automatically handles date ranges based on the selected mode.

## Usage

### Loading Data in Different Modes

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

### Data Sources

The system supports multiple data sources:

1. **Financial Data**:
   - Stored in the fundamentals directory
   - Includes financial metrics and sector information
   - Partitioned by year and month

2. **Sector Data**:
   - Integrated with financial data in the fundamentals directory
   - Can be extracted from financial data or loaded from separate files
   - Maintains backward compatibility with legacy sector mapping

3. **Returns Data**:
   - Stored in the returns directory
   - Includes price returns and total returns
   - Available as separate files or combined returns file

## Performance Considerations

The efficient storage system offers several performance benefits:

1. **Faster Queries**: Partitioning allows for faster queries by reading only the relevant partitions.
2. **Reduced Memory Usage**: Loading only the required partitions reduces memory usage.
3. **Scalability**: The system can handle larger datasets by partitioning the data.
4. **Efficient Filtering**: PyArrow provides efficient filtering capabilities.

## Configuration

The storage system can be configured through the `config.yaml` file:

```yaml
data:
  processed_data: "path/to/processed/data"
  fundamentals_dir: "path/to/fundamentals"
  returns_dir: "path/to/returns"
  sector_mapping: "path/to/sector/mapping"
  price_returns: "path/to/price/returns"
  total_returns: "path/to/total/returns"

processing:
  enable_filling: true
  winsorize_threshold: 3.0
```

## Best Practices

1. **Data Organization**:
   - Keep related data together (e.g., financials and sector data)
   - Use consistent naming conventions for files
   - Maintain clear directory structure

2. **Performance Optimization**:
   - Use appropriate partition sizes
   - Enable efficient filtering
   - Monitor memory usage

3. **Data Maintenance**:
   - Regularly clean up temporary files
   - Archive old data when needed
   - Keep backup copies of critical data

## Troubleshooting

Common issues and solutions:

1. **Memory Issues**:
   - Use appropriate mode for data loading
   - Enable efficient filtering
   - Monitor partition sizes

2. **Performance Problems**:
   - Check partition structure
   - Verify file formats
   - Monitor system resources

3. **Data Access Errors**:
   - Verify file permissions
   - Check file paths
   - Ensure data consistency 
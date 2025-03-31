# DECOHERE

A clean, organized implementation of a quantitative trading pipeline for financial data analysis.

## Overview

DECOHERE is a comprehensive pipeline for processing financial data, generating features, and analyzing financial time series. The pipeline is designed to be modular, efficient, and easy to use through a Jupyter notebook interface.

## Project Structure

```
DECOHERE/
├── config/              # Configuration files
├── data/                # Data storage
│   ├── raw/             # Raw financial data
│   ├── processed/       # Processed data
│   │   ├── fundamentals/ # Financial and sector data
│   │   └── returns/     # Price and total returns data
│   └── features/        # Generated features
├── docs/                # Documentation
├── notebooks/           # Jupyter notebooks
├── src/                 # Source code
│   ├── data/            # Data processing modules
│   │   ├── data_processor.py      # Main data processing class
│   │   └── efficient_data_storage.py # Efficient storage implementation
│   ├── features/        # Feature engineering modules
│   ├── models/          # Model training and evaluation
│   └── utils/           # Utility functions
├── tests/               # Unit tests
└── cursor.rules         # Guidelines for Cursor IDE users
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

## Usage

The pipeline can be run through the Jupyter notebook interface:

```bash
jupyter notebook notebooks/pipeline.ipynb
```

The notebook provides four modes of operation:
- **Day Mode**: Process a single day of data
- **Week Mode**: Process a week of data
- **Year Mode**: Process a year of data
- **All Years Mode**: Process all available years of data

## Pipeline Components

1. **Data Loading**: Load raw financial data from various sources:
   - Financial data from fundamentals directory
   - Sector mapping data
   - Price and total returns data
2. **Data Processing**: Clean, transform, and normalize the data
3. **Feature Generation**: Create features for analysis
4. **Data Storage**: Efficiently store and retrieve processed data using partitioned Parquet datasets

## Documentation

For detailed documentation, please refer to:

- [Project Documentation](docs/DECOHERE_DOCUMENTATION.md): Comprehensive documentation of the entire project
- [Pipeline Documentation](notebooks/README.md): Detailed documentation of the pipeline notebook
- [Project Rules and Guidelines](docs/PROJECT_RULES.md): Rules for contributing to and maintaining the project
- [Technical Documentation](docs/TECHNICAL_DOCUMENTATION.md): Detailed technical information about specific components
- [Efficient Storage](docs/README_EFFICIENT_STORAGE.md): Documentation of the efficient data storage system

## Configuration

The pipeline can be configured through YAML files in the `config/` directory:
- `config.yaml`: Main configuration file containing:
  - Data source paths
  - Processing parameters
  - Feature generation settings
  - Storage configuration

## Maintenance

### Data Storage

The project uses an efficient data storage system that:
1. Partitions data by year and month for faster queries
2. Supports multiple access modes (day, week, year, all years)
3. Uses Parquet format for efficient storage and retrieval
4. Maintains backward compatibility with legacy formats

For more details, see the [Efficient Storage Documentation](docs/README_EFFICIENT_STORAGE.md).

### Cleaning Up the Project

To clean up unnecessary files generated during development and troubleshooting, use the cleanup script:

```bash
python cleanup.py
```

This script will remove:
- Backup files (*.bak*)
- Fix files (*.fix*)
- Temporary scripts (fix_*.py)
- One-time migration scripts

All removed files are backed up to the `backup_before_cleanup` directory for reference.

### Testing the Pipeline

To test the pipeline with a specific ticker and date:

```bash
python test_pipeline.py
```

This script will:
1. Load the raw data
2. Filter for a specific ticker and date
3. Process the data
4. Generate features
5. Print a success message

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
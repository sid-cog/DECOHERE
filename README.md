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
│   └── features/        # Generated features
├── docs/                # Documentation
├── notebooks/           # Jupyter notebooks
├── src/                 # Source code
│   ├── data/            # Data processing modules
│   ├── features/        # Feature engineering modules
│   ├── models/          # Model training and evaluation
│   └── utils/           # Utility functions
├── temp_execution_files/ # Temporary files for testing and development
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

The notebook provides three modes of operation:
- **Day Mode**: Process a single day of data
- **Week Mode**: Process a week of data
- **Year Mode**: Process a year of data
- **All Years Mode**: Process all available years of data

## Pipeline Components

1. **Data Loading**: Load raw financial data from various sources
2. **Data Processing**: Clean, transform, and normalize the data
3. **Feature Generation**: Create features for analysis
4. **Data Storage**: Efficiently store and retrieve processed data

## Documentation

For detailed documentation, please refer to:

- [Project Documentation](docs/DECOHERE_DOCUMENTATION.md): Comprehensive documentation of the entire project
- [Pipeline Documentation](notebooks/README.md): Detailed documentation of the pipeline notebook
- [Project Rules and Guidelines](docs/PROJECT_RULES.md): Rules for contributing to and maintaining the project
- [Terminal Command Guidelines](docs/TERMINAL_COMMAND_GUIDELINES.md): Best practices for running terminal commands and scripts with large datasets
- [PIT_DATE Changes](README_PIT_DATE_CHANGES.md): Documentation of changes made to PIT_DATE handling
- [Cleanup Summary](CLEANUP_SUMMARY.md): Summary of the cleanup process performed on the project
- [Efficient Storage](README_EFFICIENT_STORAGE.md): Documentation of the efficient data storage system

## Configuration

The pipeline can be configured through YAML files in the `config/` directory:
- `config.yaml`: Main configuration file

## Maintenance

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
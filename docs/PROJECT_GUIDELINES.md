# DECOHERE Project Guidelines

This document outlines the rules, guidelines, and best practices for contributing to and maintaining the DECOHERE project.

## Table of Contents
1. [Code Organization](#code-organization)
2. [Documentation Standards](#documentation-standards)
3. [Terminal Command Guidelines](#terminal-command-guidelines)
4. [Project Maintenance](#project-maintenance)

## Code Organization

1. **Module Structure**: Maintain the existing module structure with clear separation of concerns:
   - `data`: Data loading and processing
   - `features`: Feature engineering and selection
   - `models`: Model training and evaluation
   - `utils`: Utility functions
   - `visualization`: Visualization tools

2. **File Naming**: Use snake_case for file names and follow the existing naming conventions.

3. **Class and Function Naming**: Use CamelCase for class names and snake_case for function names.

4. **Temporary Files**: 
   - Store all temporary, test, and backup files in the `temp_execution_files` directory
   - This includes:
     - Test output files
     - Temporary data files
     - Backup files created during testing
     - Log files from test runs
     - Any other files that are not part of the main codebase
   - Using this directory makes it easier to clean up temporary files later
   - Example:
     ```python
     # Good practice
     output_path = os.path.join('temp_execution_files', 'test_output.csv')
     
     # Avoid creating files in the root directory
     # output_path = 'test_output.csv'  # Not recommended
     ```

## Documentation Standards

1. **Code Documentation**: All classes and functions must have docstrings following the existing format.
2. **Project Documentation**: Keep the documentation in the `docs/` directory up-to-date with any changes to the codebase.
3. **README Files**: Each major component should have a README file explaining its purpose and usage.

## Terminal Command Guidelines

### General Guidelines

1. **Limit Output Volume**: 
   - Avoid printing large DataFrames or extensive logs directly to the console
   - Use summary statistics and limited sample data instead
   - For large outputs, write to files rather than printing to console

2. **Progress Reporting**:
   - Add progress indicators for long-running operations
   - Use logging with appropriate log levels (INFO for general progress, DEBUG for details)
   - Implement checkpoints to save intermediate results

3. **Error Handling**:
   - Always include proper try/except blocks
   - Log specific error messages and stack traces
   - Provide clear instructions on how to resolve common errors

4. **Memory Management**:
   - Process large datasets in chunks
   - Use memory-efficient data structures
   - Release memory when no longer needed
   - Monitor memory usage during execution

5. **Command-Line Arguments**:
   - Implement command-line arguments for flexibility
   - Provide sensible defaults
   - Include help text for each argument
   - Support common options like `--verbose`, `--quiet`, `--output`, etc.

### Example Usage

To run scripts with the new guidelines:

```bash
# Basic usage
python scripts/process_financial_data_example.py --input /path/to/data.pq

# Save output to a specific location
python scripts/process_financial_data_example.py --input /path/to/data.pq --output temp_execution_files/results.csv

# Filter for a specific ticker
python scripts/process_financial_data_example.py --input /path/to/data.pq --ticker "AAPL IB Equity"

# Process in smaller chunks for memory efficiency
python scripts/process_financial_data_example.py --input /path/to/data.pq --chunk-size 5000
```

## Project Maintenance

### Cleaning Up the Project

To maintain a clean and organized codebase, periodically remove unnecessary files:

1. **Types of Files to Remove**:
   - Backup files (*.bak*)
   - Fix files (*.fix*)
   - Temporary scripts (fix_*.py)
   - One-time migration scripts

2. **Backup Before Removal**:
   - Always back up files before removing them
   - Store backups in a designated directory (e.g., `backup_before_cleanup`)

3. **Documentation**:
   - Document the cleanup process
   - Create a summary of removed files
   - Explain the rationale for removal

4. **Temporary Files Cleanup**:
   - Periodically clean up the `temp_execution_files` directory
   - Before cleanup, verify that no important files are stored there
   - Consider creating a cleanup script that preserves the directory structure but removes the contents

### Testing

1. **Test Scripts**:
   - Create test scripts to verify functionality
   - Test with specific inputs and expected outputs
   - Verify that changes don't break existing functionality
   - Store test outputs in the `temp_execution_files` directory

2. **Verification**:
   - Verify that the pipeline works correctly after changes
   - Check that data processing produces the expected results
   - Ensure that feature generation is working properly 
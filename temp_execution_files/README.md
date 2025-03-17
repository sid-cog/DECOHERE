# Temporary Execution Files Directory

This directory is designated for storing all temporary, test, and backup files created during the development and testing of the DECOHERE project.

## Purpose

The purpose of this directory is to:
1. Keep the project root and other directories clean
2. Make it easier to identify and clean up temporary files
3. Provide a consistent location for all non-essential files

## What to Store Here

- Test output files
- Temporary data files
- Backup files created during testing
- Log files from test runs
- Any other files that are not part of the main codebase

## Usage Guidelines

When writing code that generates temporary files, always use this directory:

```python
import os

# Good practice
output_path = os.path.join('temp_execution_files', 'test_output.csv')
df.to_csv(output_path)

# When running from scripts directory
output_path = os.path.join('..', 'temp_execution_files', 'test_output.csv')
```

## Cleanup

This directory should be periodically cleaned up to remove unnecessary files. Before cleanup, verify that no important files are stored here.

A simple cleanup command:

```bash
# Remove all files but keep the README
find temp_execution_files -type f -not -name "README.md" -delete
```

## Note

Files in this directory are not tracked by version control and should not be relied upon for long-term storage of important data. 
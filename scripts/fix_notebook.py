import nbformat as nbf

# Read the existing notebook
with open('notebooks/updated_pipe.ipynb', 'r') as f:
    nb = nbf.read(f, as_version=4)

# Update the imports cell
imports_code = """import os
import sys
import pandas as pd
import numpy as np
import yaml
import logging
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to Python path
project_root = str(Path.cwd().parent)
if project_root not in sys.path:
    sys.path.append(project_root)

# Import project modules
from src.data.efficient_data_storage import EfficientDataStorage, DataType, DataStage
from src.data.data_processor import DataProcessor
from src.features.feature_generator import FeatureGenerator
from src.features.feature_selector import FeatureSelector
from src.models.model_trainer import ModelTrainer
from src.utils.logging_config import setup_logging
from src.utils.config_loader import load_config, load_mode_config"""

# Update the cells in the notebook
for cell in nb.cells:
    if cell.cell_type == 'code':
        if 'import os' in cell.source:
            cell.source = imports_code

# Write the updated notebook
with open('notebooks/updated_pipe.ipynb', 'w') as f:
    nbf.write(nb, f) 
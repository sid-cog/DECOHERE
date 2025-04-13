#!/bin/bash

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Install Jupyter extensions
echo "Installing Jupyter extensions..."
pip install jupyter-contrib-nbextensions
jupyter contrib nbextension install --user

# Backup and convert existing notebooks
echo "Backing up and converting existing notebooks..."
find . -name "*.ipynb" -not -path "*/venv/*" -not -path "*/.*/*" | while read -r notebook; do
    # Get directory and filename
    dir=$(dirname "$notebook")
    filename=$(basename "$notebook")
    
    # Create backup with orig_ prefix
    backup_path="$dir/orig_$filename"
    if [ ! -f "$backup_path" ]; then
        echo "Creating backup: $backup_path"
        cp "$notebook" "$backup_path"
    fi
    
    # Convert the original notebook
    echo "Converting: $notebook"
    jupytext --sync "$notebook"
done

# Create .gitignore if it doesn't exist
if [ ! -f ".gitignore" ]; then
    echo "Creating .gitignore..."
    cat > .gitignore << EOL
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
ENV/

# IDE
.idea/
.vscode/
*.swp
*.swo

# Jupyter Notebook
.ipynb_checkpoints
*/.ipynb_checkpoints/*

# Jupytext
*.py:percent

# Project specific
data/
results/
logs/
*.log

# Original notebook backups
orig_*.ipynb
EOL
fi

echo "Setup complete! Your environment is ready to use Jupytext."
echo "Remember to:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. Open VSCode in this directory"
echo "3. Install the Jupytext extension in VSCode"
echo ""
echo "Note: Original notebook files have been preserved with 'orig_' prefix" 
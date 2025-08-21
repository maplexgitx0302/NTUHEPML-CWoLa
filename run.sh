#!/bin/zsh

# Ensure conda is initialized before trying to activate the environment
source ~/miniconda3/etc/profile.d/conda.sh

# Activate the conda environment
conda activate cwola

# Convert the notebook to a Python script
ipynb-py-convert notebooks/main.ipynb tmp/main.py

# Run the training script
python tmp/main.py

# Remove the Python script regardless of success or failure
rm tmp/main.py

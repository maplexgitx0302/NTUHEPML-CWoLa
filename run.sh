#!/bin/zsh

# Ensure conda is initialized before trying to activate the environment
source ~/miniconda3/etc/profile.d/conda.sh

# Activate the conda environment
conda activate cwola

# Create a temporary directory if it doesn't exist
mkdir -p tmp

# Convert the notebook to a Python script
ipynb-py-convert notebooks/main_tf.ipynb tmp/main_tf.py

# Run the training script
python tmp/main_tf.py

# Remove the Python script regardless of success or failure
rm tmp/main_tf.py

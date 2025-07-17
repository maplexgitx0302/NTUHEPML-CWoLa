#!/bin/zsh

# Ensure conda is initialized before trying to activate the environment
source ~/miniconda3/etc/profile.d/conda.sh

# Activate the conda environment
conda activate cwola

# Convert the notebook to a Python script
ipynb-py-convert notebooks/training.ipynb notebooks/training.py

# Run the Python script
current_time=$(date '+%Y%m%d_%H%M%S')
# python notebooks/training.py -d './config/data_zz4l.yml' -e './config/exp.yml' -i 'test1,test2' -r 42 -t $current_time
python notebooks/training.py -d './config/data_diphoton.yml' -e './config/exp.yml' -i 'test1,test2' -r 42 -t $current_time

# Remove the Python script regardless of success or failure
rm notebooks/training.py
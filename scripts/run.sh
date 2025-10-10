#!/bin/zsh

# Ensure conda is initialized before trying to activate the environment
source ~/miniconda3/etc/profile.d/conda.sh

# Activate the conda environment
conda activate cwola

# Uncomment the following lines if you want to run

##################################################
# Run the training script
# python ./training.py
# python ./training.py --num_phi_augmentation 10
##################################################

##################################################
# Run the inference script
# python ./inference.py
################################################

##################################################
# python ./part_tuning.py
##################################################

#!/bin/zsh

# Ensure conda is initialized before trying to activate the environment
source ~/miniconda3/etc/profile.d/conda.sh

# Activate the conda environment
conda activate cwola

# Uncomment the following lines if you want to run

##################################################
# Run the training script
python ./training.py --channel zz4l --data_mode supervised
# python ./training.py --channel diphoton --data_mode jet_flavor
# python ./training.py --channel diphoton --data_mode jet_flavor --num_phi_augmentation 5
# python ./training.py --channel diphoton --data_mode jet_flavor --num_phi_augmentation 10
##################################################

##################################################
# Run the inference script
# python ./inference.py
##################################################

##################################################
# python ./part_tuning.py
##################################################

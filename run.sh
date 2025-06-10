#!/bin/bash

# Define arrays
rotations=(1 2 4)
models=("CNN_Light" "CNN_Baseline")

# Loop over all combinations
for rot in "${rotations[@]}"; do
  for model in "${models[@]}"; do
    echo "Running: num_rot=$rot, model=$model"
    python a.py num_rot=$rot model=$model
  done
done

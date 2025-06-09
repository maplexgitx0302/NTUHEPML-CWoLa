#!/bin/bash

# Define arrays
ratios=(0.0)
rotations=(1)
models=("CNN_Light" "CNN_Baseline")

# Loop over all combinations
for ratio in "${ratios[@]}"; do
  for rot in "${rotations[@]}"; do
    for model in "${models[@]}"; do
      echo "Running: CWoLa_ratio=$ratio, num_rot=$rot, model=$model"
      python a.py CWoLa_ratio=$ratio num_rot=$rot model=$model
    done
  done
done

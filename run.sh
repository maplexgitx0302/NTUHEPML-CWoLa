#!/bin/zsh

# Ensure conda is initialized before trying to activate the environment
source ~/miniconda3/etc/profile.d/conda.sh

# Activate the conda environment
conda activate cwola

# Convert the notebook to a Python script
ipynb-py-convert notebooks/training.ipynb notebooks/training.py

# # >>> specific runs >>>
# data_yml='./config/data_diphoton.yml'
# current_time='20250721_121840'
# rnd_seed=2
# python notebooks/training.py -e './config/exp_jet.yml' -i 'jet_flavor' -d $data_yml -r $rnd_seed -t $current_time
# # <<< specific runs <<<

# # >>> batch runs >>>
# data_yml='./config/data_diphoton.yml'
# current_time=$(date '+%Y%m%d_%H%M%S')
# include_decay=False  # Either 'True' or 'False'
# for rnd_seed in {1..5}
# do  
#     python notebooks/training.py -e './config/exp_jet.yml' -i 'jet_flavor' -d $data_yml -r $rnd_seed -t $current_time -x $include_decay
#     python notebooks/training.py -e './config/exp_jet_uni5.yml' -i 'jet_flavor,uni5' -d $data_yml -r $rnd_seed -t $current_time -x $include_decay
#     python notebooks/training.py -e './config/exp_jet_uni10.yml' -i 'jet_flavor,uni10' -d $data_yml -r $rnd_seed -t $current_time -x $include_decay
#     python notebooks/training.py -e './config/exp_jet_uni15.yml' -i 'jet_flavor,uni15' -d $data_yml -r $rnd_seed -t $current_time -x $include_decay
# done
# # <<< batch runs <<<

# # >>> batch runs with concatenated datasets >>>
# current_time=$(date '+%Y%m%d_%H%M%S')
# include_decay=True  # Either 'True' or 'False'
# for rnd_seed in {1..5}
# do  
#     python notebooks/training.py -e './config/exp_jet.yml' -i 'jet_flavor' -d './config/data_diphoton.yml' './config/data_zz4l.yml' -r $rnd_seed -t $current_time -x $include_decay
#     python notebooks/training.py -e './config/exp_jet_uni5.yml' -i 'jet_flavor,uni5' -d './config/data_diphoton.yml' './config/data_zz4l.yml' -r $rnd_seed -t $current_time -x $include_decay
#     python notebooks/training.py -e './config/exp_jet_uni10.yml' -i 'jet_flavor,uni10' -d './config/data_diphoton.yml' './config/data_zz4l.yml' -r $rnd_seed -t $current_time -x $include_decay
#     python notebooks/training.py -e './config/exp_jet_uni15.yml' -i 'jet_flavor,uni15' -d './config/data_diphoton.yml' './config/data_zz4l.yml' -r $rnd_seed -t $current_time -x $include_decay
# done
# # <<< batch runs with concatenated datasets <<<

# >>> batch runs with preprocessed datasets >>>
current_time=$(date '+%Y%m%d_%H%M%S')

for include_decay in True False
do
    for rnd_seed in {1..10}
    do  
        python notebooks/training.py -e './config/exp_sv.yml' \
                                     -i 'supervised' \
                                     -d './config/data_diphoton.yml' \
                                     -r $rnd_seed \
                                     -t $current_time \
                                     -x $include_decay
    done
done
# <<< batch runs with preprocessed datasets <<<

# Remove the Python script regardless of success or failure
rm notebooks/training.py

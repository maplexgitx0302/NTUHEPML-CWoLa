# Deep Learning + CWoLa for VBF vs. GGF Classification

This project applies deep learning to distinguish between VBF and GGF Higgs production modes using the **Classification Without Labels (CWoLa)** framework. The approach is inspired by the paper [*Classification without labels: Learning from mixed samples in high energy physics*](https://arxiv.org/abs/1708.02949), which introduces CWoLa as a viable strategy for learning directly from mixed real data samples.

---

## Environment Setup

1. Download miniconda through:
   ```bash
   # Assuming Linux system
   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

   # Install thorugh sh. Type 'yes' when asking automatically initialization.
   sh Miniconda3-latest-Linux-x86_64.sh
   ```

2. Create a virtual environment:
   ```bash
   # Initialization
   conda env create -f environment.yml

   # Update when the environment.yml changed
   conda env update -f environment.yml
   ```

3. Directly activate in Jupyter, or activate/exit with:
   ```bash
   conda activate cwola
   conda deactivate
   ```

4. (Optional) Create a `.env` such that VSCode can fetch the packages.
   ```
   PYTHONPATH=~/miniconda3/envs/cwola/lib/python3.12/site-packages
   ```


## Models

#### Convolutional Neural Networks (CNN)

- `CNN_Baseline`: A reference implementation based on [*VBF vs. GGF Higgs with Full-Event Deep Learning: Towards a Decay-Agnostic Tagger*](https://arxiv.org/abs/2209.05518). This model utilizes event-level CNNs to perform classification.

- `CNN_Light`: A simplified and more lightweight version of `CNN_Baseline`, designed to reduce model complexity and training time while retaining performance.

#### Particle Transformers (ParT)

- `ParT_Baseline`: A transformer-based architecture based on [*Particle Transformer for Jet Tagging*](https://arxiv.org/abs/2202.03772). This model captures particle-level features using attention mechanisms tailored for jet tagging tasks.

- `ParT_*`: A family of lighter variants derived from `ParT_Baseline`, offering faster training and inference with reduced computational cost.

## Usage

#### Data Preprocessing & Augmentation

The data preprocessings can be implemented by the following steps:

1. Check the supported methods:
   - **data preprocessing:** Check the methods provided in the class `src.data_preprocess.MCSimData`
   - **data augmentation:** Supported functions can be found in `src.data_augment`.
2. Give abbreviations for the preprocessing/augmentation methods in the class `LitDataModule`.
    `./notebooks/training.ipynb` &rarr; `LitDataModule` &rarr; `__init__`
3. Determine which preprocessings/augmentations to be used through YAML files in `./config` with file named `exp_*.yml`.
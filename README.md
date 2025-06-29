# Deep Learning + CWoLa for VBF vs. GGF Classification

This project applies deep learning to distinguish between VBF and GGF Higgs production modes using the **Classification Without Labels (CWoLa)** framework. The approach is inspired by the paper [*Classification without labels: Learning from mixed samples in high energy physics*](https://arxiv.org/abs/1708.02949), which introduces CWoLa as a viable strategy for learning directly from mixed real data samples.

---

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
   - **data preprocessing:** Check the methods provided in the class `./source/data_preprocess.py` &rarr; `MCSimData`
   - **data augmentation:** Supported functions can be found in `./source/data_augment.py`.
2. Give abbreviations for the preprocessing methods in the class `LitDataModule`.
    `./main.ipynb` &rarr; `LitDataModule` &rarr; `__init__`
3. Determine which preprocessings to be used in `config.yaml` by their abbreviations.
   `./config.yaml` &rarr; `dataset` &rarr; `preprocessings`
4. To distinguish each training result, we recommend to add tags for each training in `config.yaml`.
   `./config.yaml` &rarr; `tags`
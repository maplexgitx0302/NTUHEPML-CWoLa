import argparse
from itertools import product
import os
from pathlib import Path
import sys
import time
from typing import List

import lightning
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import CSVLogger
import torch
import torch.nn as nn
import yaml

ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT))

from src import CNN_EventCNN, ParT_Light, utils
from src.lightning import LitDataModule, BinaryLitModel


def keras_like_init(module: nn.Module):
    # Dense / Conv weights = Glorot-uniform, biases = zeros
    if isinstance(module, (nn.Linear,
                           nn.Conv1d, nn.Conv2d, nn.Conv3d,
                           nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
        if getattr(module, "weight", None) is not None:
            nn.init.xavier_uniform_(module.weight)     # = Keras glorot_uniform
        if getattr(module, "bias", None) is not None:
            nn.init.zeros_(module.bias)                # = Keras zeros

    # BatchNorm: gamma=1, beta=0 (running stats untouched here)
    if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        if getattr(module, "weight", None) is not None:
            nn.init.ones_(module.weight)               # gamma
        if getattr(module, "bias", None) is not None:
            nn.init.zeros_(module.bias)                # beta

    # LayerNorm: gamma=1, beta=0 (match Keras LayerNormalization scale/beta)
    if isinstance(module, nn.LayerNorm):
        if getattr(module, "weight", None) is not None:
            nn.init.ones_(module.weight)
        if getattr(module, "bias", None) is not None:
            nn.init.zeros_(module.bias)


def training(
    data_mode: str, data_format: str, data_info: dict, include_decay: bool,
    model_cls: nn.Module, lr: float, keras_init: bool,
    tags: List[str], rnd_seed: int, date_time: str, **kwargs
):
    """Train a model with the given configuration."""

    lightning.seed_everything(rnd_seed)

    num_channels = 3 if include_decay else 2
    model = model_cls(num_channels=num_channels, keras_init=keras_init)
    if keras_init:
        model.apply(keras_like_init)

    # Output and log directories
    save_dir = ROOT / 'output' / (('' if include_decay else 'ex-') + data_info['decay_channel']) / ('_'.join(tags))
    if data_mode == 'jet_flavor':
        name = f"{model.__class__.__name__}-{date_time}-L{kwargs['luminosity']}"
    elif data_mode == 'supervised':
        name = f"{model.__class__.__name__}-{date_time}-SV"
    version = f"rnd_seed-{rnd_seed}"
    output_dir = save_dir / name / version
    if os.path.exists(output_dir):
        print(f"[Warning] Output directory {output_dir} already exists.")
        return

    # Lightning data setup
    BATCH_SIZE = 512
    lit_data_module = LitDataModule(
        batch_size=BATCH_SIZE,
        data_mode=data_mode,
        data_format=data_format,
        data_info=data_info,
        include_decay=include_decay,
        **kwargs
    )

    # Lightning model setup
    with open(ROOT / 'config' / 'training.yml', 'r') as f:
        training_config = yaml.safe_load(f)
    lit_model = BinaryLitModel(
        model=model,
        lr=lr,
        pos_weight=lit_data_module.pos_weight,
        optimizer_settings=training_config['optimizer_settings']
    )

    # Lightning loggers
    logger = CSVLogger(save_dir=save_dir, name=name, version=version)
    hparams = {}
    hparams.update(lit_data_module.hparams)
    hparams.update(training_config)
    logger.log_hyperparams(hparams)

    # Lightning trainer
    trainer = lightning.Trainer(
        max_epochs=500,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        logger=logger,
        callbacks=[
            ModelCheckpoint(**training_config['ModelCheckpoint']),
            EarlyStopping(**training_config['EarlyStopping']),
        ],
    )

    # Lightning trainning and testing
    trainer.fit(lit_model, lit_data_module)
    trainer.test(lit_model, datamodule=lit_data_module, ckpt_path='best')
    os.makedirs(output_dir, exist_ok=True)
    utils.count_number_of_data(lit_data_module, output_dir)
    utils.count_model_parameters(lit_model, output_dir)
    utils.plot_metrics(output_dir)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Run experiment with config and options.")
    parser.add_argument('--channel', type=str, required=True, choices=['diphoton', 'zz4l'], help="Decay channel to use.")
    parser.add_argument('--data_mode', type=str, required=True, choices=['jet_flavor', 'supervised'], help="Data mode to use.")
    parser.add_argument('--keras_init', action='store_true', help="Use Keras-like initialization.")
    parser.add_argument('--num_phi_augmentation', type=int, default=0, help="Number of phi augmentations.")
    parser.add_argument('--datetime', type=str, default=None, help="Datetime string for naming.")
    args = parser.parse_args()
    CHANNEL = args.channel
    DATA_MODE = args.data_mode
    KERAS_INIT = args.keras_init
    NUM_PHI_AUGMENTATION = args.num_phi_augmentation
    DATETIME = time.strftime("%Y%m%d_%H%M%S", time.localtime()) if args.datetime is None else args.datetime

    # Random seeds and configurations
    rnd_seeds = [123 + 100 * i for i in range(10)]
    for rnd_seed, include_decay in product(rnd_seeds, [True, False]):

        with open(ROOT / 'config' / f"data_{CHANNEL}.yml", 'r') as f:
            data_info = yaml.safe_load(f)

        for data_format, model_cls, lr in [
            ('image', CNN_EventCNN, 1e-4),
            ('sequence', ParT_Light, 4e-4),
        ]:

            for luminosity in [100, 300, 900, 1800, 3000]:
                training(
                    data_mode=DATA_MODE,
                    data_format=data_format,
                    data_info=data_info,
                    include_decay=include_decay,
                    model_cls=model_cls,
                    lr=lr,
                    keras_init=KERAS_INIT,
                    tags=[DATA_MODE],
                    rnd_seed=rnd_seed,
                    date_time=DATETIME,
                    luminosity=luminosity,
                    num_phi_augmentation=NUM_PHI_AUGMENTATION,
                )

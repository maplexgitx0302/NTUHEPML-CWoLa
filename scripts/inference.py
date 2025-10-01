from itertools import product
import os
from pathlib import Path
import sys
from typing import List

import lightning
import pandas as pd
import torch
import torch.nn as nn
import yaml

ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT))

from src import CNN_EventCNN, ParT_Light
from src.lightning import LitDataModule, BinaryLitModel


def inference(
    data_mode: str, data_format: str, train_channel: str, test_channel: str, include_decay: bool,
    model_cls: nn.Module, tags: List[str], rnd_seed: int, date_time: str, **kwargs
):
    """Inference with a trained model with the given configuration."""

    lightning.seed_everything(rnd_seed)

    with open(ROOT / 'config' / f"data_{test_channel}.yml", 'r') as f:
        data_info = yaml.safe_load(f)

    # Output and log directories
    save_dir = ROOT / 'output' / (('' if include_decay else 'ex-') + train_channel) / ('_'.join(tags))
    if data_mode == 'jet_flavor':
        name = f"{model_cls.__name__}-{date_time}-L{kwargs['luminosity']}"
    elif data_mode == 'supervised':
        name = f"{model_cls.__name__}-{date_time}-SV"
    version = f"rnd_seed-{rnd_seed}"
    ckpt_dir = save_dir / name / version / 'checkpoints'
    ckpt_path = ckpt_dir / os.listdir(ckpt_dir)[0]

    # Lightning data setup
    BATCH_SIZE = 64
    lit_data_module = LitDataModule(
        batch_size=BATCH_SIZE,
        data_mode=data_mode,
        data_format=data_format,
        data_info=data_info,
        include_decay=include_decay,
        **kwargs
    )

    # Lightning model setup
    lit_model = BinaryLitModel.load_from_checkpoint(ckpt_path)

    # Lightning trainer
    trainer = lightning.Trainer(accelerator='gpu' if torch.cuda.is_available() else 'cpu', logger=False)

    # Lightning testing
    metrics = trainer.test(lit_model, dataloaders=lit_data_module.test_dataloader())

    return metrics


if __name__ == '__main__':

    TRAIN_CHANNEL = 'diphoton'
    TEST_CHANNEL = 'zz4l'
    DATETIME = '20250930_105915'

    df = pd.DataFrame()
    rnd_seeds = [123 + 100 * i for i in range(10)]
    luminosities = [100, 300, 900, 1800, 3000]

    for rnd_seed, data_mode in product(rnd_seeds, ['jet_flavor']):

        for data_format, model_cls in [('image', CNN_EventCNN), ('sequence', ParT_Light)]:

            for luminosity in luminosities:
                metrics = inference(
                    data_mode=data_mode,
                    data_format=data_format,
                    train_channel=TRAIN_CHANNEL,
                    test_channel=TEST_CHANNEL,
                    include_decay=False,
                    model_cls=model_cls,
                    tags=[data_mode],
                    rnd_seed=rnd_seed,
                    date_time=DATETIME,
                    luminosity=luminosity,
                )

                df = pd.concat([df, pd.DataFrame({
                    'rnd_seed': rnd_seed,
                    'model_name': model_cls.__name__,
                    'luminosity': luminosity,
                    'test_accuracy': metrics[0]['test_accuracy'],
                    'test_auc': metrics[0]['test_auc'],
                }, index=[0])], ignore_index=True)

    os.makedirs(ROOT / 'output' / 'inference', exist_ok=True)
    df.to_csv(ROOT / 'output' / 'inference' / f'inference_{TRAIN_CHANNEL}_to_{TEST_CHANNEL}_{DATETIME}.csv', index=False)
import argparse
from itertools import product
import os
from pathlib import Path
import sys

import lightning
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT))

from src.lightning import LitDataModule, BinaryLitModel


def inference(data_format: str, model: str, rnd_seed: int, date_time: str, luminosity: float, channel: str):
    """Inference with a trained model with the given configuration."""

    lightning.seed_everything(rnd_seed)

    with open(ROOT / 'config' / f"data_{channel}.yml", 'r') as f:
        data_info = yaml.safe_load(f)

    # Output and log directories
    save_dir = ROOT / 'output' / 'ex-diphoton' / 'jet_flavor'
    name = f"{model}-{date_time}-L{luminosity}"
    version = f"rnd_seed-{rnd_seed}"
    ckpt_dir = save_dir / name / version / 'checkpoints'
    ckpt_path = ckpt_dir / os.listdir(ckpt_dir)[0]

    # Lightning data setup
    lit_data_module = LitDataModule(
        batch_size=512,
        data_mode='jet_flavor',
        data_format=data_format,
        data_info=data_info,
        include_decay=False,
        num_train=10000,
        num_valid=10000,
        num_test=10000,
        luminosity=luminosity,
    )

    # Lightning model setup
    lit_model = BinaryLitModel.load_from_checkpoint(ckpt_path)

    # Lightning trainer
    trainer = lightning.Trainer(accelerator='gpu' if torch.cuda.is_available() else 'cpu', logger=False)

    # Lightning testing
    metrics = trainer.test(lit_model, dataloaders=lit_data_module.test_dataloader())

    return metrics


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Run experiment with config and options.")
    parser.add_argument('--channel', type=str, required=True, choices=['diphoton', 'zz4l', 'za2l'], help="Decay channel to be inferenced.")
    args = parser.parse_args()
    CHANNEL = args.channel

    inference_info_list = [
        # # Removing decay product only
        # ('image', 'CNN_EventCNN', '20250930_105915', '+0'),
        # ('image', 'CNN_EventCNN', '20250923_232355', '+5'),
        # ('image', 'CNN_EventCNN', '20250924_111848', '+10'),
        # ('sequence', 'ParT_Light', '20250930_105915', '+0'),
        # ('sequence', 'ParT_Light', '20250923_232355', '+5'),
        # ('sequence', 'ParT_Light', '20250924_111848', '+10'),

        # Removing also neighbors near decay product
        ('image', 'CNN_EventCNN', '20251005_154731', '+0'),
        ('image', 'CNN_EventCNN', '20251006_114628', '+5'),
        ('image', 'CNN_EventCNN', '20251007_015709', '+10'),
        ('sequence', 'ParT_Light', '20251005_154731', '+0'),
        ('sequence', 'ParT_Light', '20251006_114628', '+5'),
        ('sequence', 'ParT_Light', '20251007_015709', '+10'),
    ]

    rnd_seed_list = [123 + 100 * i for i in range(10)]
    luminosity_list = [100, 300, 900, 1800, 3000]

    for (data_format, model, date_time, num_rot_aug) in inference_info_list:

        df = pd.DataFrame()

        for rnd_seed, luminosity in product(rnd_seed_list, luminosity_list):

            metrics = inference(
                data_format=data_format,
                model=model,
                rnd_seed=rnd_seed,
                date_time=date_time,
                luminosity=luminosity,
                channel=CHANNEL,
            )

            df = pd.concat([df, pd.DataFrame({
                'rnd_seed': rnd_seed,
                'model': model,
                'luminosity': luminosity,
                'test_accuracy': metrics[0]['test_accuracy'],
                'test_auc': metrics[0]['test_auc'],
            }, index=[0])], ignore_index=True)

        output_dir = ROOT / 'output' / 'inference' / CHANNEL
        os.makedirs(output_dir, exist_ok=True)
        df.to_csv(output_dir / f'{model}_{date_time}.csv', index=False)
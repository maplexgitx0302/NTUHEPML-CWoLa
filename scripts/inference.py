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

from src import CNN_EventCNN, ParT_Light
from src.lightning import LitDataModule, BinaryLitModel


def inference(data_format: str, model_name: str, rnd_seed: int, date_time: str, **kwargs):
    """Inference with a trained model with the given configuration."""

    lightning.seed_everything(rnd_seed)

    with open(ROOT / 'config' / f"data_zz4l.yml", 'r') as f:
        data_info = yaml.safe_load(f)

    # Output and log directories
    save_dir = ROOT / 'output' / 'ex-diphoton' / 'jet_flavor'
    name = f"{model_name}-{date_time}-L{kwargs['luminosity']}"
    version = f"rnd_seed-{rnd_seed}"
    ckpt_dir = save_dir / name / version / 'checkpoints'
    ckpt_path = ckpt_dir / os.listdir(ckpt_dir)[0]

    # Lightning data setup
    lit_data_module = LitDataModule(
        batch_size=256,
        data_mode='jet_flavor',
        data_format=data_format,
        data_info=data_info,
        include_decay=False,
        num_train=10000,
        num_valid=10000,
        num_test=10000,
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

    for DATE_TIME in ['20250930_105915', '20250923_232355', '20250924_111848']:

        df = pd.DataFrame()
        rnd_seeds = [123 + 100 * i for i in range(10)]
        luminosities = [100, 300, 900, 1800, 3000]

        for rnd_seed, luminosity in product(rnd_seeds, luminosities):

            for data_format, model_cls in [('image', CNN_EventCNN), ('sequence', ParT_Light)]:
                
                metrics = inference(
                    data_format=data_format,
                    model_name=model_cls.__name__,
                    rnd_seed=rnd_seed,
                    date_time=DATE_TIME,
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
        df.to_csv(ROOT / 'output' / 'inference' / f'inference_{DATE_TIME}.csv', index=False)
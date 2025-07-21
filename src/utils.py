import os

import lightning
from lightning.pytorch.utilities.model_summary import ModelSummary
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme()


def count_model_parameters(lit_model: lightning.LightningModule, output_dir: str):
    """Count the number of parameters in the model and save the summary."""

    with open(os.path.join(output_dir, 'num_params.txt'), 'w') as file:
        # Recursively count to 3 layers is enough
        for depth in range(1, 4):
            print(f"Model Summary (max_depth={depth}):", file=file)
            print(ModelSummary(lit_model, max_depth=depth), file=file)
            print(f"\n{'='*100}\n", file=file)


def plot_metrics(output_dir: str):
    """Plot training and validation metrics from the CSV log file."""

    df = pd.read_csv(os.path.join(output_dir, 'metrics.csv'))

    # 1st row: train | 2nd row: valid
    fig, ax = plt.subplots(2, 3, figsize=(10, 6))
    metrics = [
        'train_loss_epoch', 'train_accuracy', 'train_auc',
        'valid_loss', 'valid_accuracy', 'valid_auc'
    ]

    # Plot epoch vs metrics
    for i, metric in enumerate(metrics):
        data = df[df[metric].notna()]
        plot = sns.lineplot(data=data, x='epoch', y=metric, ax=ax.flat[i])
        plot.set_title(metric)

    # Save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics.png'))

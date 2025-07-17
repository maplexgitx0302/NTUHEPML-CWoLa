# %%
import argparse
from pathlib import Path
import sys
import yaml

import lightning
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import CSVLogger
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC

try:
    project_root = Path(__file__).parent.parent
except:
    project_root = Path.cwd().parent
sys.path.append(project_root.as_posix())

from src.data_augment import aug_phi_shift
from src.data_preprocess import MCSimData
from src.data_cwola import split_by_sv, split_by_jet_flavor
from src.model_cnn import CNN_Baseline, CNN_Light, CNN_EventCNN
from src.model_part import ParT_Baseline, ParT_Medium, ParT_Light, ParT_SuperLight, ParT_ExtremeLight
from src import utils

# %%
# Argument parser for training configurations
parser = argparse.ArgumentParser(description='Parser for training configurations.')

# Define the arguments
parser.add_argument('-d', '--data_yaml', type=str, required=True, help='YAML configuration file of data in config directory')
parser.add_argument('-e', '--exp_yaml', type=str, required=True, help='YAML configuration file of experiment in config directory')
parser.add_argument('-i', '--tags', type=str, required=True, help="Comma-separated list of tags")
parser.add_argument('-r', '--rnd_seed', type=int, required=True, help='Random seed for initialization')
parser.add_argument('-t', '--time', type=str, required=True, help="Datetime for the experiment")

# Parse the arguments
args = parser.parse_args()

# Configurations for the whole training
config = {}

# Access the arguments
config['rnd_seed'] = args.rnd_seed
config['tags'] = args.tags.split(',')
config['time'] = args.time
with open(project_root / Path(args.data_yaml), 'r') as f:
    config['data'] = yaml.safe_load(f)
with open(project_root / Path(args.exp_yaml), 'r') as f:
    config['exp'] = yaml.safe_load(f)
with open(project_root / Path('config') / Path('training.yml'), 'r') as f:
    config['training'] = yaml.safe_load(f)

# %%
class LitDataModule(lightning.LightningDataModule):
    def __init__(self, batch_size: int, mode: str, data_format: str, data_info: dict, include_decay: bool,
                 train_size: float = None, num_train: int = None, num_valid: int = None, num_test: int = None,
                 preprocessings: list[str] = [], augmentations: dict = {'functions': []},
                 **kwargs):
        super().__init__()

        self.data_format = data_format
        self.data_info = data_info
        self.batch_size = batch_size
        self.preprocessings = preprocessings
        self.augmentations = augmentations

        # Information of signal and background datasets
        sig_info = data_info['signal']
        bkg_info = data_info['background']

        # Monte Carlo simulation data
        SIG = MCSimData(sig_info['path'], include_decay=include_decay)
        BKG = MCSimData(bkg_info['path'], include_decay=include_decay)

        ''' ***** Preprocessing ***** '''
        SIG = self._data_preprocessings(SIG)
        BKG = self._data_preprocessings(BKG)

        # Choose the representation of the dataset
        if data_format == 'image':
            sig_tensor = SIG.to_image()
            bkg_tensor = BKG.to_image()
        elif data_format == 'sequence':
            sig_tensor = SIG.to_sequence()
            bkg_tensor = BKG.to_sequence()

        # Create mixed dataset for implementing CWoLa
        if mode == 'jet_flavor':
            train_sig, train_bkg, valid_sig, valid_bkg, test_sig, test_bkg = split_by_jet_flavor(
                sig_tensor=sig_tensor, bkg_tensor=bkg_tensor,
                sig_flavor=SIG.jet_flavor, bkg_flavor=BKG.jet_flavor,
                branching_ratio=data_info['branching_ratio'], luminosity=data_info['luminosity'],
                sig_cross_section=sig_info['cross_section'], bkg_cross_section=bkg_info['cross_section'],
                sig_preselection_rate=sig_info['preselection_rate'], bkg_preselection_rate=bkg_info['preselection_rate'],
                train_size=train_size, num_test=num_test,
            )
        elif mode == 'sv':
            train_sig, train_bkg, valid_sig, valid_bkg, test_sig, test_bkg = split_by_sv(
                sig_tensor=sig_tensor, bkg_tensor=bkg_tensor,
                num_train=num_train, num_valid=num_valid, num_test=num_test,
            )

        ''' ***** Augmentation ***** '''
        train_sig = self._data_augmentations(train_sig)
        train_bkg = self._data_augmentations(train_bkg)

        # Create torch datasets
        self.train_dataset = TensorDataset(torch.cat([train_sig, train_bkg], dim=0), torch.cat([torch.ones(len(train_sig)), torch.zeros(len(train_bkg))], dim=0))
        self.valid_dataset = TensorDataset(torch.cat([valid_sig, valid_bkg], dim=0), torch.cat([torch.ones(len(valid_sig)), torch.zeros(len(valid_bkg))], dim=0))
        self.test_dataset = TensorDataset(torch.cat([test_sig, test_bkg], dim=0), torch.cat([torch.ones(len(test_sig)), torch.zeros(len(test_bkg))], dim=0))

        # Calculate positive weight for loss function
        num_pos = len(train_sig)  # y == 1
        num_neg = len(train_bkg)  # y == 0
        self.pos_weight = torch.tensor([num_neg / num_pos], dtype=torch.float32)
    
    def _data_preprocessings(self, Data: MCSimData) -> MCSimData:
        if 'cop' in self.preprocessings:
            Data.preprocess_center_of_phi()
        return Data

    def _data_augmentations(self, data: torch.Tensor) -> torch.Tensor:
        aug_dict = self.augmentations
        for func in aug_dict['functions']:
            if func == 'phi_uni':
                data = aug_phi_shift(data, mode='uniform', format=self.data_format, rotations=aug_dict['rotations'])
            elif func == 'phi_rand':
                data = aug_phi_shift(data, mode='random', format=self.data_format, rotations=aug_dict['rotations'])
        return data

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

# %%
class BinaryLitModel(lightning.LightningModule):
    def __init__(self, model: nn.Module, lr: float, pos_weight: torch.Tensor, optimizer_settings: dict = None):
        super().__init__()

        self.model = model
        self.lr = lr
        self.optimizer_settings = optimizer_settings
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        self.train_accuracy = BinaryAccuracy()
        self.valid_accuracy = BinaryAccuracy()
        self.test_accuracy = BinaryAccuracy()

        self.train_auc = BinaryAUROC()
        self.valid_auc = BinaryAUROC()
        self.test_auc = BinaryAUROC()

    def forward(self, x: torch.Tensor) -> torch.Tensor:        
        return self.model(x)

    def configure_optimizers(self):
        optimizer_settings = self.optimizer_settings
        optimizer = getattr(torch.optim, self.optimizer_settings['optimizer'])
        optimizer = optimizer(self.parameters(), lr=self.lr)
        if optimizer_settings['lr_scheduler'] is None:
            return optimizer
        else:
            scheduler = getattr(torch.optim.lr_scheduler, optimizer_settings['lr_scheduler'])
            scheduler = scheduler(optimizer, **optimizer_settings[scheduler.__name__])
            lr_scheduler: dict = {'scheduler': scheduler}
            lr_scheduler.update(optimizer_settings['lightning_monitor'])
            return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}

    def _shared_step(self, batch: tuple[torch.Tensor, torch.Tensor], mode: str):
        x, y_true = batch
        logits: torch.Tensor = self(x)
        loss = self.loss_fn(logits.view(-1), y_true.float())
        y_pred = torch.sigmoid(logits.view(-1))

        if mode == 'train':
            self.train_auc.update(y_pred, y_true)
            self.train_accuracy.update(y_pred, y_true)
        elif mode == 'valid':
            self.valid_auc.update(y_pred, y_true)
            self.valid_accuracy.update(y_pred, y_true)
        elif mode == 'test':
            self.test_auc.update(y_pred, y_true)
            self.test_accuracy.update(y_pred, y_true)

        self.log(f"{mode}_loss", loss, on_epoch=True, prog_bar=(mode == 'train'))

        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, mode='train')

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, mode='valid')

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, mode='test')

    def on_train_epoch_end(self):
        self.log('train_auc', self.train_auc.compute(), prog_bar=True)
        self.log('train_accuracy', self.train_accuracy.compute(), prog_bar=True)
        self.train_auc.reset()
        self.train_accuracy.reset()

    def on_validation_epoch_end(self):
        self.log('valid_auc', self.valid_auc.compute(), prog_bar=True)
        self.log('valid_accuracy', self.valid_accuracy.compute(), prog_bar=True)
        self.valid_auc.reset()
        self.valid_accuracy.reset()

    def on_test_epoch_end(self):
        self.log('test_auc', self.test_auc.compute(), prog_bar=True)
        self.log('test_accuracy', self.test_accuracy.compute(), prog_bar=True)
        self.test_auc.reset()
        self.test_accuracy.reset()

# %%
num_channels = config['exp']['LitDataModule']['num_channels']

for data_format, model, lr, batch_size_step, batch_accumulate in [
    # ('image', CNN_EventCNN(num_channels=num_channels), 2e-4, 64, 8),
    ('image', CNN_Baseline(num_channels=num_channels), 1e-5, 64, 8),
    ('image', CNN_Light(num_channels=num_channels), 5e-4, 64, 8),
    # ('sequence', ParT_Baseline(num_channels=num_channels), 5e-5, 64, 8),
    # ('sequence', ParT_Medium(num_channels=num_channels), 1e-4, 64, 8),
    # ('sequence', ParT_Light(num_channels=num_channels), 5e-4, 64, 8),
    # ('sequence', ParT_SuperLight(num_channels=num_channels), 1e-3, 64, 8),
    ('sequence', ParT_ExtremeLight(num_channels=num_channels), 5e-3, 64, 8),
]:
    # Set the random seed for reproducibility
    lightning.seed_everything(config['rnd_seed'])
    
    # Path for saving training results and logs
    save_dir = project_root / Path('output') / Path(config['data']['decay_channel']) / Path('_'.join(config['tags']))
    name = model.__class__.__name__
    version = f"{config['time']}-rnd_seed{config['rnd_seed']}"
    output_dir = save_dir / Path(name) / Path(version)

    # Lightning DataModule
    lit_data_module = LitDataModule(
        data_format=data_format,
        data_info=config['data'],
        **config['exp']['LitDataModule'],
    )

    # Lightning Model
    lit_model = BinaryLitModel(
        model=model,
        lr=lr,
        pos_weight=lit_data_module.pos_weight,
        optimizer_settings=config['training']['optimizer_settings']
    )

    # Lightning Logger
    logger = CSVLogger(save_dir=save_dir, name=name, version=version)
    logger.log_hyperparams(config)

    # Lightning Trainer & Callbacks
    trainer = lightning.Trainer(
        logger=logger,
        callbacks=[
            ModelCheckpoint(**config['training']['ModelCheckpoint']),
            EarlyStopping(**config['training']['EarlyStopping']),
        ],
        **config['exp']['Trainer'],
    )

    # Train and test the model
    trainer.fit(lit_model, lit_data_module)
    trainer.test(lit_model, datamodule=lit_data_module, ckpt_path='best')

    # Summary of the training
    utils.count_model_parameters(lit_model, output_dir)
    utils.plot_metrics(output_dir)
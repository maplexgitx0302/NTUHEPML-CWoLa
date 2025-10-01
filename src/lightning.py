from .common import ROOT, wrap_pi
from .data import MCSimData

from typing import Tuple

import lightning
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC


class LitDataModule(lightning.LightningDataModule):
    def __init__(self, batch_size: int, data_mode: str, data_format: str, data_info: dict,
                 include_decay: bool, luminosity: float = None, num_phi_augmentation: int = 0,
                 **kwargs):
        super().__init__()

        self.save_hyperparameters()
        self.batch_size = batch_size

        # Monte Carlo simulation data
        sig_data = MCSimData(ROOT / data_info['signal']['path'])
        bkg_data = MCSimData(ROOT / data_info['background']['path'])

        # Create mixed dataset for implementing CWoLa
        if data_mode == 'jet_flavor':
            train_sig, train_bkg, valid_sig, valid_bkg, test_sig, test_bkg = self.split_by_jet_flavor(luminosity, data_info, sig_data, bkg_data)
        elif data_mode == 'supervised':
            train_sig, train_bkg, valid_sig, valid_bkg, test_sig, test_bkg = self.split_by_supervised(sig_data, bkg_data)
        else:
            raise ValueError(f"Unsupported data mode: {data_mode}. Supported data modes are 'jet_flavor' and 'supervised'.")

        # Data augmentation by random phi rotation
        if num_phi_augmentation > 0:
            train_sig = self.augment_phi_per_event(train_sig, num_phi_augmentation)
            train_bkg = self.augment_phi_per_event(train_bkg, num_phi_augmentation)

        # Transform to desired data format
        if data_format == 'image':
            train_sig = sig_data.to_image(train_sig, include_decay)
            train_bkg = bkg_data.to_image(train_bkg, include_decay)
            valid_sig = sig_data.to_image(valid_sig, include_decay)
            valid_bkg = bkg_data.to_image(valid_bkg, include_decay)
            test_sig = sig_data.to_image(test_sig, include_decay)
            test_bkg = bkg_data.to_image(test_bkg, include_decay)
        elif data_format == 'sequence':
            train_sig = sig_data.to_sequence(train_sig, include_decay)
            train_bkg = bkg_data.to_sequence(train_bkg, include_decay)
            valid_sig = sig_data.to_sequence(valid_sig, include_decay)
            valid_bkg = bkg_data.to_sequence(valid_bkg, include_decay)
            test_sig = sig_data.to_sequence(test_sig, include_decay)
            test_bkg = bkg_data.to_sequence(test_bkg, include_decay)

        # For tracking number of data samples
        self.train_sig, self.train_bkg = train_sig, train_bkg
        self.valid_sig, self.valid_bkg = valid_sig, valid_bkg
        self.test_sig, self.test_bkg = test_sig, test_bkg

        # Create torch datasets
        self.train_dataset = TensorDataset(torch.cat([train_sig, train_bkg], dim=0), torch.cat([torch.ones(len(train_sig)), torch.zeros(len(train_bkg))], dim=0))
        self.valid_dataset = TensorDataset(torch.cat([valid_sig, valid_bkg], dim=0), torch.cat([torch.ones(len(valid_sig)), torch.zeros(len(valid_bkg))], dim=0))
        self.test_dataset = TensorDataset(torch.cat([test_sig, test_bkg], dim=0), torch.cat([torch.ones(len(test_sig)), torch.zeros(len(test_bkg))], dim=0))

        # Calculate positive weight for loss function
        num_pos = len(train_sig)  # y == 1
        num_neg = len(train_bkg)  # y == 0
        self.pos_weight = torch.tensor([num_neg / num_pos], dtype=torch.float32)

    def split_by_supervised(self, sig_data: MCSimData, bkg_data: MCSimData):
        """Split data for supervised training."""

        NUM_TRAIN, NUM_VALID, NUM_TEST = 100000, 25000, 25000

        perm_sig = np.random.permutation(len(sig_data.particle_flow))
        perm_bkg = np.random.permutation(len(bkg_data.particle_flow))
        sig_array = sig_data.particle_flow[perm_sig]
        bkg_array = bkg_data.particle_flow[perm_bkg]

        train_sig = sig_array[:NUM_TRAIN]
        train_bkg = bkg_array[:NUM_TRAIN]
        valid_sig = sig_array[NUM_TRAIN: NUM_TRAIN + NUM_VALID]
        valid_bkg = bkg_array[NUM_TRAIN: NUM_TRAIN + NUM_VALID]
        test_sig = sig_array[NUM_TRAIN + NUM_VALID: NUM_TRAIN + NUM_VALID + NUM_TEST]
        test_bkg = bkg_array[NUM_TRAIN + NUM_VALID: NUM_TRAIN + NUM_VALID + NUM_TEST]

        return train_sig, train_bkg, valid_sig, valid_bkg, test_sig, test_bkg

    def split_by_jet_flavor(self, luminosity: float, data_info: dict, sig_data: MCSimData, bkg_data: MCSimData):
        """Split data by jet flavor composition for CWoLa training."""

        def get_event_counts(data_type: str, cut_info_key: str):
            cut_info_path = ROOT / data_info[data_type]['cut_info']
            cut_info_npy = np.load(cut_info_path, allow_pickle=True)
            cut_info = cut_info_npy.item()['cutflow_number']
            N = int(data_info[data_type]['cross_section'] * cut_info[cut_info_key] / cut_info['Total'] * data_info['branching_ratio'] * luminosity)
            print(f"[CWoLa-Log] [{data_type}] {cut_info_key}: {N} events")
            return N

        NUM_TEST = 10000
        num_sig_in_sig = get_event_counts('signal', 'two quark jet: sig region')
        num_sig_in_bkg = get_event_counts('signal', 'two quark jet: bkg region')
        num_bkg_in_sig = get_event_counts('background', 'two quark jet: sig region')
        num_bkg_in_bkg = get_event_counts('background', 'two quark jet: bkg region')

        sig_in_sig_mask = sig_data.jet_flavor['2q0g']
        sig_in_bkg_mask = sig_data.jet_flavor['1q1g'] | sig_data.jet_flavor['0q2g']
        bkg_in_sig_mask = bkg_data.jet_flavor['2q0g']
        bkg_in_bkg_mask = bkg_data.jet_flavor['1q1g'] | bkg_data.jet_flavor['0q2g']
        num_test_sig_in_sig = int(NUM_TEST * np.sum(sig_in_sig_mask) / (np.sum(sig_in_sig_mask) + np.sum(sig_in_bkg_mask)))
        num_test_sig_in_bkg = NUM_TEST - num_test_sig_in_sig
        num_test_bkg_in_sig = int(NUM_TEST * np.sum(bkg_in_sig_mask) / (np.sum(bkg_in_sig_mask) + np.sum(bkg_in_bkg_mask)))
        num_test_bkg_in_bkg = NUM_TEST - num_test_bkg_in_sig

        sig_in_sig = sig_data.particle_flow[sig_in_sig_mask]
        sig_in_bkg = sig_data.particle_flow[sig_in_bkg_mask]
        bkg_in_sig = bkg_data.particle_flow[bkg_in_sig_mask]
        bkg_in_bkg = bkg_data.particle_flow[bkg_in_bkg_mask]

        idx_sig_in_sig = np.random.choice(len(sig_in_sig), num_sig_in_sig + num_test_sig_in_sig, replace=False)
        idx_sig_in_bkg = np.random.choice(len(sig_in_bkg), num_sig_in_bkg + num_test_sig_in_bkg, replace=False)
        idx_bkg_in_sig = np.random.choice(len(bkg_in_sig), num_bkg_in_sig + num_test_bkg_in_sig, replace=False)
        idx_bkg_in_bkg = np.random.choice(len(bkg_in_bkg), num_bkg_in_bkg + num_test_bkg_in_bkg, replace=False)

        sig_in_sig = sig_in_sig[idx_sig_in_sig]
        sig_in_bkg = sig_in_bkg[idx_sig_in_bkg]
        bkg_in_sig = bkg_in_sig[idx_bkg_in_sig]
        bkg_in_bkg = bkg_in_bkg[idx_bkg_in_bkg]

        def split_data(data: np.array, num_samples: int, num_test: int):
            TRAIN_SIZE_RATIO = 0.8
            train_size = int(num_samples * TRAIN_SIZE_RATIO)
            return (
                data[:train_size],
                data[train_size:num_samples],
                data[num_samples:num_samples + num_test]
            )

        train_sig_in_sig, valid_sig_in_sig, test_sig_in_sig = split_data(sig_in_sig, num_sig_in_sig, num_test_sig_in_sig)
        train_sig_in_bkg, valid_sig_in_bkg, test_sig_in_bkg = split_data(sig_in_bkg, num_sig_in_bkg, num_test_sig_in_bkg)
        train_bkg_in_sig, valid_bkg_in_sig, test_bkg_in_sig = split_data(bkg_in_sig, num_bkg_in_sig, num_test_bkg_in_sig)
        train_bkg_in_bkg, valid_bkg_in_bkg, test_bkg_in_bkg = split_data(bkg_in_bkg, num_bkg_in_bkg, num_test_bkg_in_bkg)

        train_sig = np.concatenate([train_sig_in_sig, train_bkg_in_sig], axis=0)
        train_bkg = np.concatenate([train_sig_in_bkg, train_bkg_in_bkg], axis=0)
        valid_sig = np.concatenate([valid_sig_in_sig, valid_bkg_in_sig], axis=0)
        valid_bkg = np.concatenate([valid_sig_in_bkg, valid_bkg_in_bkg], axis=0)
        test_sig = np.concatenate([test_sig_in_sig, test_sig_in_bkg], axis=0)
        test_bkg = np.concatenate([test_bkg_in_sig, test_bkg_in_bkg], axis=0)

        return train_sig, train_bkg, valid_sig, valid_bkg, test_sig, test_bkg

    def augment_phi_per_event(self, data: np.ndarray, k: int) -> np.ndarray:
        """Augment data by random phi rotation per event, implemented in a memory-efficient way."""

        N, M, C = data.shape
        out = np.empty(((k + 1) * N, M, C), dtype=data.dtype)
        out[:N] = data
        for i in range(k):
            s = slice((i + 1) * N, (i + 2) * N)
            out[s] = data
            shift = np.random.uniform(-np.pi, np.pi, size=(N, 1)).astype(data.dtype)
            out[s, :, 2] = wrap_pi(out[s, :, 2] + shift)
        return out

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)


class BinaryLitModel(lightning.LightningModule):
    def __init__(self, model: nn.Module, lr: float, pos_weight: torch.Tensor = None, optimizer_settings: dict = None):
        super().__init__()
        self.save_hyperparameters()

        self.model = model
        self.lr = lr
        self.optimizer_settings = optimizer_settings
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        base_metrics = MetricCollection({"auc": BinaryAUROC(), "accuracy": BinaryAccuracy()})
        self.metrics = {
            "train": base_metrics.clone(prefix="train_"),
            "valid": base_metrics.clone(prefix="valid_"),
            "test": base_metrics.clone(prefix="test_"),
        }

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

    def on_train_start(self):
        self.metrics['train'] = self.metrics['train'].to(self.device)

    def on_validation_start(self):
        self.metrics['valid'] = self.metrics['valid'].to(self.device)

    def on_test_start(self):
        self.metrics['test'] = self.metrics['test'].to(self.device)

    def _shared_step(self, batch: Tuple[torch.Tensor, torch.Tensor], split: str):
        x, y_true = batch
        logits = self(x).squeeze(-1)
        loss = self.loss_fn(logits, y_true.float())
        y_pred = torch.sigmoid(logits)
        self.metrics[split].update(y_pred, y_true.int())
        self.log(f"{split}_loss", loss, on_epoch=True, on_step=False, prog_bar=(split == "train"), batch_size=y_true.size(0))
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "valid")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    def _compute_and_log_split(self, split: str, prog_bar: bool = False):
        computed = self.metrics[split].compute()
        self.log_dict(computed, on_epoch=True, on_step=False, prog_bar=prog_bar)
        self.metrics[split].reset()

    def on_train_epoch_end(self):
        self._compute_and_log_split("train", prog_bar=True)

    def on_validation_epoch_end(self):
        self._compute_and_log_split("valid", prog_bar=True)

    def on_test_epoch_end(self):
        self._compute_and_log_split("test", prog_bar=False)

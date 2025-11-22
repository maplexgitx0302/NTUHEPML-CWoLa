from .common import ROOT, wrap_pi

from pathlib import Path
from typing import Dict, List

import h5py
import numpy as np
import torch

from .common import ROOT


class MCSimData:
    def __init__(self, path: str):
        self.path = str(path)

        # -------- Channels --------
        self.detector_channels = ['TOWER', 'TRACK']
        self.slices = [slice(0, 250), slice(250, 400)]
        if 'diphoton' in self.path:
            self.decay_channel = ['PHOTON']
            self.slices.append(slice(400, 402))
        elif 'zz4l' in self.path or 'ZZ4l' in self.path:
            self.decay_channel = ['LEPTON']
            self.slices.append(slice(400, 404))
        elif 'za2l' in self.path or 'Za2l' in self.path:
            self.decay_channel = ['LEPTON', 'PHOTON']
            self.slices.append(slice(400, 402))
            self.slices.append(slice(402, 403))
        else:
            raise ValueError(f"Unsupported dataset: {self.path}. Supported datasets are 'diphoton', 'zz4l', and 'za2l'.")
        self.channels = self.detector_channels + self.decay_channel

        with h5py.File(str(path), 'r') as hdf5_file:
            # -------- Jet flavor --------
            self.jet_flavor = self._extract_jet_flavor(hdf5_file)

            # -------- Particle flow information (pt, eta, phi) --------
            particle_flow = self._extract_particle_flow(hdf5_file, self.channels)

            # -------- Preprocessing --------
            particle_flow = self._preprocess_phi_transformation(particle_flow)
            particle_flow = self._preprocess_center_of_phi(particle_flow)
            particle_flow = self._preprocess_flipping(particle_flow)

        self.particle_flow = particle_flow

    def _extract_jet_flavor(self, hdf5_file: h5py.File) -> Dict[str, np.ndarray | int]:
        """Build gluon/quark composition masks from J1/J2 flavors."""

        J1 = np.asarray(hdf5_file["J1"]["flavor"][:])
        J2 = np.asarray(hdf5_file["J2"]["flavor"][:])
        g1, g2 = (J1 == 21), (J2 == 21)  # 21 == gluon

        mask_2q0g = (~g1) & (~g2)
        mask_1q1g = ((~g1) & g2) | (g1 & (~g2))
        mask_0q2g = g1 & g2

        return {
            "2q0g": mask_2q0g,
            "1q1g": mask_1q1g,
            "0q2g": mask_0q2g,
            "total": len(J1),
        }

    def _extract_particle_flow(self, hdf5_file: h5py.File, channels: List[str]) -> np.ndarray:
        """Load pt/eta/phi/(mask) for each channel."""

        # -------- Particle flow array (N, ΣM, 4) --------
        pts = np.concatenate([np.asarray(hdf5_file[channel]['pt'][:],  dtype=np.float32) for channel in channels], axis=1)  # (N, ΣM)
        etas = np.concatenate([np.asarray(hdf5_file[channel]['eta'][:], dtype=np.float32) for channel in channels], axis=1)  # (N, ΣM)
        phis = np.concatenate([np.asarray(hdf5_file[channel]['phi'][:], dtype=np.float32) for channel in channels], axis=1)  # (N, ΣM)
        phis = wrap_pi(phis)
        particle_flow = np.stack([pts, etas, phis], axis=-1)  # (N, ΣM, 3)

        # -------- Mask array (N, ΣM) --------
        mask = []
        for channel in channels:
            if 'mask' in hdf5_file[channel]:
                _mask = np.asarray(hdf5_file[channel]['mask'][:], dtype=bool)  # (N, M)
            else:
                _mask = np.ones_like(hdf5_file[channel]['pt'][:], dtype=bool)
            mask.append(_mask)
        mask = np.concatenate(mask, axis=1)  # (N, ΣM)

        # -------- Apply mask to particle flow --------
        particle_flow = np.where(mask[..., None], particle_flow, np.nan)

        return particle_flow

    def _preprocess_phi_transformation(self, particle_flow: np.ndarray) -> np.ndarray:
        """Transform phi to reduce variance (if var(phi) > 0.5, phi -> phi + pi)."""

        phi = particle_flow[..., 2]
        phi_var = np.var(np.nan_to_num(phi, nan=0.0), axis=-1, keepdims=True)
        phi = np.where(phi_var > 0.5, phi + np.pi, phi)
        phi = wrap_pi(phi)
        particle_flow[..., 2] = phi

        return particle_flow

    def _preprocess_center_of_phi(self, particle_flow: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        """Shift phi to the center of pt frame."""

        pt = particle_flow[..., 0]  # (N, ΣM)
        phi = particle_flow[..., 2]  # (N, ΣM)

        pt_phi = np.nansum(pt * phi, axis=-1, keepdims=True)  # (N, 1)
        center_of_phi = pt_phi / (np.nansum(pt, axis=-1, keepdims=True) + eps)  # (N, 1)
        phi = wrap_pi(phi - center_of_phi)
        particle_flow[..., 2] = phi

        return particle_flow

    def _preprocess_flipping(self, particle_flow: np.ndarray) -> np.ndarray:
        """Flip quadrant with highest pt to the first quadrant (phi > 0, eta > 0)."""

        pt = particle_flow[..., 0]  # (N, ΣM)
        eta = particle_flow[..., 1]  # (N, ΣM)
        phi = particle_flow[..., 2]  # (N, ΣM)

        # -------- Quadrant pT sums (0: ++, 1: +-, 2: --, 3: -+) --------
        cond0 = (eta > 0) & (phi > 0)
        cond1 = (eta > 0) & (phi < 0)
        cond2 = (eta < 0) & (phi < 0)
        cond3 = (eta < 0) & (phi > 0)
        conds = np.stack([cond0, cond1, cond2, cond3], axis=-1)  # (N, ΣM, 4)
        pt_quadrants = np.nansum(pt[..., None] * conds, axis=1)  # (N, 4)

        # -------- Decide flips per event --------
        q_argmax = np.argmax(pt_quadrants, axis=1)  # (N,)
        phi_flip = np.where((q_argmax == 1) | (q_argmax == 2), -1.0, 1.0)[:, None]  # (N, 1)
        eta_flip = np.where((q_argmax == 2) | (q_argmax == 3), -1.0, 1.0)[:, None]  # (N, 1)

        # -------- Apply flips to particle flow --------
        eta = eta * eta_flip
        phi = wrap_pi(phi * phi_flip)
        particle_flow[..., 1] = eta
        particle_flow[..., 2] = phi

        return particle_flow

    def to_image(self, particle_flow: np.ndarray, include_decay: bool, norm: bool = True, grid_size: int = 40, eps: float = 1e-8) -> torch.Tensor:
        """Convert the particle flow data to images (N, C, H, W)."""

        particle_flow = np.where(np.isnan(particle_flow), 0.0, particle_flow)  # (N, ΣM, 3)

        phi_bins = np.linspace(-np.pi, np.pi, grid_size + 1, dtype=np.float32)
        eta_bins = np.linspace(-5.0, 5.0, grid_size + 1, dtype=np.float32)

        def array_to_image(array) -> np.ndarray:
            """Convert one channel to image (N, H, W)."""

            pt, eta, phi = array[..., 0], array[..., 1], array[..., 2]  # (N, M)

            N, M = pt.shape
            image = np.zeros((N, grid_size, grid_size), dtype=np.float32)

            phi_idx = np.digitize(phi, phi_bins, right=False) - 1
            eta_idx = np.digitize(eta, eta_bins, right=False) - 1
            phi_idx = np.clip(phi_idx, 0, grid_size - 1)
            eta_idx = np.clip(eta_idx, 0, grid_size - 1)

            event_idx = np.repeat(np.arange(N, dtype=np.int64), M)
            np.add.at(image, (event_idx, eta_idx.ravel(), phi_idx.ravel()), pt.ravel())

            return image

        images = []
        if include_decay:
            for _slice in self.slices:
                array = particle_flow[:, _slice, :]  # (N, M, 3)
                images.append(array_to_image(array))
        else:
            if 'diphoton' in self.path or 'zz4l' in self.path:
                decay_image = array_to_image(particle_flow[:, self.slices[-1], :])
            elif 'za2l' in self.path:
                decay_image_lepton = array_to_image(particle_flow[:, self.slices[-2], :])
                decay_image_photon = array_to_image(particle_flow[:, self.slices[-1], :])
                decay_image = decay_image_lepton + decay_image_photon
            decay_mask = decay_image > 0.0
            for i, channel in enumerate(self.detector_channels):
                array = particle_flow[:, self.slices[i], :]  # (N, M, 3)
                image = array_to_image(array)
                image = np.where(decay_mask, 0.0, image)
                images.append(image)
        images = np.stack(images, axis=1)  # (N, C, H, W)

        # --- pt normalisation per (N, C) across H*W ---
        if norm:
            N, C, H, W = images.shape
            flat = images.reshape(N, C, -1)
            mean = flat.mean(axis=-1, keepdims=True)
            std = flat.std(axis=-1, keepdims=True)
            std = np.clip(std, a_min=eps, a_max=None)
            images = (flat - mean) / std
            images = images.reshape(N, C, H, W)

        return torch.from_numpy(images).float()

    def to_sequence(self, particle_flow: np.ndarray, include_decay: bool, eps: float = 0.0) -> torch.Tensor:
        """Convert particle flow features to sequences (N, ΣM_selected, 3+C)."""

        # Choose which channel names / spans to emit
        if include_decay:
            channel_slices = self.slices
        else:
            channel_slices = self.slices[:len(self.detector_channels)]

            # --- Remove detector hits that match decay objects (like _exclude_decay_information) ---
            decay_eta = np.concatenate([particle_flow[:, s, 1] for s in self.slices[len(self.detector_channels):]], axis=-1)
            decay_phi = np.concatenate([particle_flow[:, s, 2] for s in self.slices[len(self.detector_channels):]], axis=-1)

            # NaNs compare False in <= / ==, so NaN decay entries are ignored automatically
            for detector_slice in channel_slices:
                detector_eta = particle_flow[:, detector_slice, 1]  # (N, M_det)
                detector_phi = particle_flow[:, detector_slice, 2]  # (N, M_det)

                # roadcast (N, M_det, 1) vs (N, 1, M_dec) -> (N, M_det, M_dec)
                eta_diff2 = (detector_eta[:, :, None] - decay_eta[:, None, :]) ** 2  # (N, M_det, M_dec)
                phi_diff2 = (detector_phi[:, :, None] - decay_phi[:, None, :]) ** 2  # (N, M_det, M_dec)

                # Set the threshold inside the image grid (eps = 1 / grid_size)
                matched = (((phi_diff2 / np.pi ** 2) < (eps ** 2)) & ((eta_diff2 / 5 ** 2) < (eps ** 2))).any(axis=-1)  # (N, M_det)
                matched_mean = matched.sum(axis=-1).mean()
                matched_std = matched.sum(axis=-1).std()
                print(f"[Sequence] {matched_mean:.1f} +- {matched_std:.1f} detector hits matched to decay objects over {len(matched)} events (eps={eps})")

                # Set matched detector hits to NaN across [pt, eta, phi]
                detector_view = particle_flow[:, detector_slice, :]
                detector_view[matched] = np.nan

        # --- Build sequences with per-event pt normalization and one-hot channel indicator ---
        C = len(self.channels) if include_decay else len(self.detector_channels)
        sequences = []

        for one_hot_index, detector_slice in enumerate(channel_slices):
            pt = particle_flow[:, detector_slice, 0]  # (N, M)
            eta = particle_flow[:, detector_slice, 1]  # (N, M)
            phi = particle_flow[:, detector_slice, 2]  # (N, M)

            # mask of valid hits for this channel
            valid = ~np.isnan(pt)  # (N, M)

            # per-event normalization (ignore NaNs)
            pt_mean = np.nanmean(pt, axis=-1, keepdims=True)  # (N,1)
            pt_std = np.nanstd(pt,  axis=-1, keepdims=True)  # (N,1)
            pt = (pt - pt_mean) / pt_std

            feat = np.stack([pt, eta, phi], axis=-1)  # (N, M, 3)

            # one-hot channel id
            one_hot = np.zeros((1, 1, C), dtype=feat.dtype)  # (1,1,C)
            one_hot[..., one_hot_index] = 1.0
            one_hot = np.broadcast_to(one_hot, (feat.shape[0], feat.shape[1], C))
            feat_oh = np.concatenate([feat, one_hot], axis=-1)  # (N, M, 3+C)

            # keep NaNs where invalid
            feat_oh = np.where(valid[..., None], feat_oh, np.nan)
            sequences.append(feat_oh)

        # concat channels along the sequence axis
        sequences = np.concatenate(sequences, axis=1)  # (N, ΣM_selected, 3+C)

        return torch.from_numpy(sequences).float()

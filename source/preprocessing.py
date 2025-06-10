import h5py
import os

import numpy as np
import torch

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def hdf5_jet_flavor(filename):
    """Read *.h5 files and return the jet flavor."""

    with h5py.File(os.path.join(root_dir, 'dataset', filename), 'r') as f:
        # 1 : d, 2 : u, 3 : s, 4 : c, 5 : b, 21 : g

        J1 = np.array(f['J1']['flavor'][:])
        J1 = torch.from_numpy(J1).long()

        J2 = np.array(f['J2']['flavor'][:])
        J2 = torch.from_numpy(J2).long()

        J1_q = (J1 != 0) & (J1 != 21)  # Non-gluon and non-unknown
        J1_g = (J1 == 21)  # Gluon

        J2_q = (J2 != 0) & (J2 != 21)
        J2_g = (J2 == 21)

        J_2q0g = (J1_q & J2_q)
        J_1q1g = (J1_q & J2_g) | (J1_g & J2_q)
        J_0q2g = (J1_g & J2_g)

    return {'2q0g': J_2q0g, '1q1g': J_1q1g, '0q2g': J_0q2g, 'total': len(J1)}


def hdf5_to_image(filename, grid_size=40):
    """Read *.h5 files and turn it into images."""

    with h5py.File(os.path.join(root_dir, 'dataset', filename), 'r') as f:

        channels = ['PHOTON', 'TOWER', 'TRACK']
        images = []

        phi_bins = np.linspace(-np.pi, np.pi, grid_size + 1)
        eta_bins = np.linspace(-5, 5, grid_size + 1)

        for channel in channels:
            pt = f[channel]['pt'][:]
            eta = f[channel]['eta'][:]
            phi = f[channel]['phi'][:]

            # Apply mask if available
            if 'mask' in f[channel]:
                mask = f[channel]['mask'][:]
                pt = pt * mask

            # Get shapes: N events, M indices per event.
            N, M = pt.shape

            # Initialize the image array for the current channel
            image = np.zeros((N, grid_size, grid_size))  # (N, L, L)

            # Compute indices for phi and eta
            phi_indices = np.digitize(phi, phi_bins) - 1  # (N, M)
            eta_indices = np.digitize(eta, eta_bins) - 1  # (N, M)

            # Ensure indices are within bounds
            phi_indices = np.clip(phi_indices, 0, grid_size - 1)
            eta_indices = np.clip(eta_indices, 0, grid_size - 1)

            # Create indexing arrays by flattening the 2D arrays.
            event_idx = np.repeat(np.arange(N), M)  # (N * M,)
            phi_idx = phi_indices.reshape(-1)   # (N * M,)
            eta_idx = eta_indices.reshape(-1)   # (N * M,)
            pt_flat = pt.reshape(-1)            # (N * M,)

            # Use np.add.at to accumulate the pt values at the corresponding indices.
            np.add.at(image, (event_idx, phi_idx, eta_idx), pt_flat)

            images.append(image)

    # Concatenate images along the channel axis
    images = np.stack(images, axis=1)

    return torch.from_numpy(images).float()


def hdf5_to_seq(filename):
    """Read *.h5 files and turn it into sequences."""

    with h5py.File(os.path.join(root_dir, 'dataset', filename), 'r') as f:

        channels = ['PHOTON', 'TOWER', 'TRACK']
        preprocessed_features = []

        for one_hot_index, channel in enumerate(channels):

            # Particle flow features.
            pt = f[channel]['pt'][:]    # (N, M)
            eta = f[channel]['eta'][:]  # (N, M)
            phi = f[channel]['phi'][:]  # (N, M)
            feature = np.stack([pt, eta, phi], axis=-1)  # (N, M, 3)

            # Create one-hot vector and broadcast using `np.tile`.
            one_hot = np.zeros((1, 1, len(channels)), dtype=feature.dtype)
            one_hot[0, 0, one_hot_index] = 1.0
            one_hot_broadcast = np.tile(one_hot, (feature.shape[0], feature.shape[1], 1))  # (N, M, 3)

            # Concatenate one-hot vector with the feature.
            feature_with_onehot = np.concatenate([feature, one_hot_broadcast], axis=-1)  # (N, M, 6)

            if 'mask' in f[channel]:
                mask = f[channel]['mask'][:]  # (N, M)
                mask_expanded = np.expand_dims(mask, axis=-1)  # (N, M, 1)
                feature_with_onehot = np.where(mask_expanded, feature_with_onehot, np.nan)

            preprocessed_features.append(feature_with_onehot)  # (N, M, 6)

        x = np.concatenate(preprocessed_features, axis=1)  # (N, M1+M2+M3, 6)
        x = torch.from_numpy(x).float()

        return x

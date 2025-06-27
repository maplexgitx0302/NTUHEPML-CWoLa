import h5py

import numpy as np
import torch


class MCSimData:
    def __init__(self, path: str, including_unknown=True):
        """Class for the Monte Carlo simulation data.

            Args:
                path (str): Path to the HDF5 file.
                including_unknown (bool): Whether to include unknown jet flavors (default: True).
        """

        # Read the HDF5 file and store it in the class
        with h5py.File(path, 'r') as hdf5_file:

            # Extract the jet flavor information
            J1 = torch.from_numpy(np.array(hdf5_file['J1']['flavor'][:])).long()
            J2 = torch.from_numpy(np.array(hdf5_file['J2']['flavor'][:])).long()

            J1_q = (J1 != 21) if including_unknown else ((J1 != 21) & (J1 != 0))
            J1_g = (J1 == 21)

            J2_q = (J2 != 21) if including_unknown else ((J2 != 21) & (J2 != 0))
            J2_g = (J2 == 21)

            J_2q0g = (J1_q & J2_q)
            J_1q1g = (J1_q & J2_g) | (J1_g & J2_q)
            J_0q2g = (J1_g & J2_g)

            self.jet_flavor = {'2q0g': J_2q0g, '1q1g': J_1q1g, '0q2g': J_0q2g, 'total': len(J1)}

            # Extract the particle flow information for each channel
            self.channels = ['PHOTON', 'TOWER', 'TRACK']
            self.data: dict[dict[str, np.ndarray]] = {}
            for channel in self.channels:
                self.data[channel] = {
                    'pt': hdf5_file[channel]['pt'][:],
                    'eta': hdf5_file[channel]['eta'][:],
                    'phi': hdf5_file[channel]['phi'][:],
                }
                if 'mask' in hdf5_file[channel]:
                    self.data[channel]['mask'] = hdf5_file[channel]['mask'][:]

    def preprocess_center_of_phi(self, eps=1e-8):
        """Shift phi to the center of pt frame."""
        for channel in self.channels:
            pt = self.data[channel]['pt']
            phi = self.data[channel]['phi']
            phi = np.mod(phi + np.pi, 2 * np.pi) - np.pi
            phi = phi - np.sum(pt * phi, axis=-1, keepdims=True) / (np.sum(pt, axis=-1, keepdims=True) + eps)
            phi = np.mod(phi + np.pi, 2 * np.pi) - np.pi
            self.data[channel]['phi'] = phi

    def to_image(self, grid_size: int = 40) -> torch.Tensor:
        """Convert the particle flow data to images."""

        images = []

        phi_bins = np.linspace(-np.pi, np.pi, grid_size + 1)
        eta_bins = np.linspace(-5, 5, grid_size + 1)

        for channel in self.channels:
            pt = self.data[channel]['pt']
            eta = self.data[channel]['eta']
            phi = self.data[channel]['phi']

            # Apply mask if available
            if 'mask' in self.data[channel]:
                mask = self.data[channel]['mask']
                pt = pt * mask

            # Get shapes: N events, M indices per event.
            N, M = pt.shape

            # Initialize the image array for the current channel
            image = np.zeros((N, grid_size, grid_size))

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

    def to_sequence(self) -> torch.Tensor:
        """Convert the particle flow data to sequences."""

        sequences = []

        for one_hot_index, channel in enumerate(self.channels):

            # Particle flow features.
            pt = self.data[channel]['pt']
            eta = self.data[channel]['eta']
            phi = self.data[channel]['phi']
            feature = np.stack([pt, eta, phi], axis=-1)

            # Create one-hot vector and broadcast using `np.tile`.
            one_hot = np.zeros((1, 1, len(self.channels)), dtype=feature.dtype)
            one_hot[0, 0, one_hot_index] = 1.0
            one_hot_broadcast = np.tile(one_hot, (feature.shape[0], feature.shape[1], 1))  # (N, M, 3)

            # Concatenate one-hot vector with the feature.
            feature_with_onehot = np.concatenate([feature, one_hot_broadcast], axis=-1)  # (N, M, 6)

            if 'mask' in self.data[channel]:
                mask = self.data[channel]['mask'][:]  # (N, M)
                mask_expanded = np.expand_dims(mask, axis=-1)  # (N, M, 1)
                feature_with_onehot = np.where(mask_expanded, feature_with_onehot, np.nan)

            sequences.append(feature_with_onehot)  # (N, M, 6)

        sequences = np.concatenate(sequences, axis=1)  # (N, M1+M2+M3, 6)
        sequences = torch.from_numpy(sequences).float()

        return sequences

import numpy as np
import torch


def aug_phi_shift(data: torch.Tensor, format: str, rotations: int, mode: str) -> torch.Tensor:
    """Augment the data by shifting the phi angle.

        Args:
            data (torch.Tensor): Input data tensor.
            format (str): Format of the data, either 'image' or 'sequence'.
            rotations (int): Number of rotations for augmentation.
            phi_column (int): Index of the phi column in the data.
            mode (str): Mode of augmentation, either 'uniform' or 'random'.
    """

    augmented = [data]

    if format == 'image':
        grid_size = data.shape[-1]
        perm = np.random.choice(grid_size - 1, (grid_size - 1,), replace=False) + 1  # Avoid shifting by 0

    for r in range(1, rotations + 1):

        new_data = data.clone()

        if format == 'image':
            if mode == 'uniform':
                shift = int(grid_size * r / (rotations + 1))
            elif mode == 'random':
                shift = perm[r]
            phi_column = -2  # See `to_image` function in data_preprocess.py
            new_data = torch.roll(new_data, shifts=shift, dims=phi_column)

        elif format == 'sequence':
            if mode == 'uniform':
                shift = r * 2 * np.pi / (rotations + 1)
            elif mode == 'random':
                shift = 2 * np.pi * np.random.rand()
            phi_column = 2  # See `to_sequence` function in data_preprocess.py
            new_data[:, phi_column] = (new_data[:, phi_column] + shift) % (2 * np.pi)

        else:
            raise ValueError(f"Unsupported format: {format}")

        augmented.append(new_data)

    return torch.cat(augmented, dim=0)

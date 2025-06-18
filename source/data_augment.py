import torch


def aug_seq_uniform_phi_shift(data: torch.Tensor, rotation_divisions: int, phi_column: int) -> torch.Tensor:
    """Augment the sequence data by uniformly shifting the phi angle."""
    angle_diff = 2 * torch.pi / rotation_divisions
    augmented = [data]
    for r in range(1, rotation_divisions):
        new_data = data.clone()
        new_data[:, phi_column] = (new_data[:, phi_column] + angle_diff * r) % (2 * torch.pi)
        augmented.append(new_data)
    return torch.cat(augmented, dim=0)

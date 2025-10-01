from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parent.parent


def wrap_pi(phi: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """Wrap angles to [-pi, pi)."""
    return (phi + np.pi) % (2 * np.pi) - np.pi

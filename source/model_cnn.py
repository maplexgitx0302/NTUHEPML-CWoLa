import torch
import torch.nn as nn


class CNN_Baseline(nn.Module):
    def __init__(self, num_rot=1, num_channels=3):
        """Assuming input images are of size (3, 40, 40)."""

        super().__init__()

        self.num_rot = num_rot  # rotation-equivariance, 1 means no rotation

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=num_channels, out_channels=64, kernel_size=5, padding='same'),  # 40 * 40
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # 20 * 20

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding='same'),  # 20 * 20
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # 10 * 10

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding='same'),  # 10 * 10
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # 5 * 5

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding='same'),  # 5 * 5
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Flatten()
        )

        # Fully connected layers
        self.fnn = nn.Sequential(
            nn.Linear(3200, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        grid_size = x.shape[-1]

        if self.num_rot == 1:
            # No rotation, just apply CNN
            x = self.cnn(x)
        else:
            # Translate respect to `phi`
            shifted_features = []
            for i in range(self.num_rot):
                x_shifted = torch.roll(x, shifts=int(grid_size * i / self.num_rot), dims=-2)
                shifted_features.append(self.cnn(x_shifted))
            x = sum(shifted_features) / self.num_rot

        x = self.fnn(x)
        return x


class CNN_Light(CNN_Baseline):
    def __init__(self, num_rot=1, num_channels=3):
        """Assuming input images are of size (3, 40, 40)."""

        super().__init__(num_rot)

        self.cnn = nn.Sequential(
            nn.Conv2d(num_channels, 8, kernel_size=3, padding=1),   # (3, 40, 40) → (8, 40, 40)
            nn.ReLU(),
            nn.MaxPool2d(2),                             # → (8, 20, 20)

            nn.Conv2d(8, 16, kernel_size=3, padding=1),  # → (16, 20, 20)
            nn.ReLU(),
            nn.MaxPool2d(2),                             # → (16, 10, 10)
        )

        self.fnn = nn.Sequential(
            nn.Flatten(),                                # → 16 * 10 * 10 = 1600
            nn.Linear(1600, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )


class CNN_ExtremeLight(CNN_Baseline):
    def __init__(self, num_rot=1, num_channels=3):
        """Minimal CNN for (3, 40, 40) input with low capacity."""
        super().__init__(num_rot)

        self.cnn = nn.Sequential(
            nn.Conv2d(num_channels, 4, kernel_size=3, padding=1),   # (3, 40, 40) → (4, 40, 40)
            nn.ReLU(),
            nn.MaxPool2d(2),                             # → (4, 20, 20)

            nn.Conv2d(4, 8, kernel_size=3, padding=1),   # → (8, 20, 20)
            nn.ReLU(),
            nn.MaxPool2d(2),                             # → (8, 10, 10)
        )

        self.fnn = nn.Sequential(
            nn.Flatten(),                                # → 8 * 10 * 10 = 800
            nn.Linear(800, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

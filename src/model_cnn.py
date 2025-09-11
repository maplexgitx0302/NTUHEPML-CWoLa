import torch
import torch.nn as nn


class CNN_EventCNN(nn.Module):
    def __init__(self, num_channels: int = 3, keras_init: bool = False):
        """Assuming input images are of size (3, 40, 40)."""
        super().__init__()

        if keras_init:
            self.in_bn = nn.BatchNorm2d(num_channels, eps=1e-3, momentum=0.01)
        else:
            self.in_bn = nn.BatchNorm2d(num_channels)

        self.conv_1 = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=3, padding=0),  # (3, 40, 40) → (32, 38, 38)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=0),  # → (64, 36, 36)
            nn.ReLU(),
            nn.MaxPool2d(3),  # → (64, 12, 12)
        )

        self.conv_2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # (64, 12, 12) → (64, 12, 12)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # (64, 12, 12) → (64, 12, 12)
            nn.ReLU(),
        )

        self.conv_3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # (64, 12, 12) → (64, 12, 12)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # (64, 12, 12) → (64, 12, 12)
            nn.ReLU(),
        )

        self.conv_4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=0),  # (64, 12, 12) → (64, 10, 10)
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),  # → (64, 1, 1)
            nn.Flatten(),  # → 64
        )

        self.fnn = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_bn(x)
        x = self.conv_1(x)
        x = self.conv_2(x) + x  # residual connection
        x = self.conv_3(x) + x  # residual connection
        x = self.conv_4(x)
        x = self.fnn(x)
        return x

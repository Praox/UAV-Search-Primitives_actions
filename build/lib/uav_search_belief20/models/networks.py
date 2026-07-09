from __future__ import annotations

import torch
from torch import nn


class GridFeatureNet(nn.Module):
    """Small CNN feature extractor for memory maps.

    Input shape must be `[batch, channels, grid, grid]`.
    """

    def __init__(self, in_channels: int, grid_size: int, feature_dim: int = 128):
        super().__init__()
        self.in_channels = int(in_channels)
        self.grid_size = int(grid_size)
        self.feature_dim = int(feature_dim)
        self.conv = nn.Sequential(
            nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        conv_out = 64 * self.grid_size * self.grid_size
        self.mlp = nn.Sequential(
            nn.Linear(conv_out, 256),
            nn.ReLU(),
            nn.Linear(256, self.feature_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.conv(x))


class QNetwork(nn.Module):
    """DQN/DDQN network: shared feature extractor + deterministic linear Q head."""

    def __init__(self, obs_shape: tuple[int, int, int], action_dim: int, feature_dim: int = 128):
        super().__init__()
        c, h, w = obs_shape
        if h != w:
            raise ValueError("GridFeatureNet assumes a square grid.")
        self.feature_net = GridFeatureNet(c, h, feature_dim)
        self.head = nn.Linear(feature_dim, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.feature_net(x))

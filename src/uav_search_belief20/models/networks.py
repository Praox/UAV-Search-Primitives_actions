from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class GridFeatureNet(nn.Module):
    """Compact spatial feature extractor for UAV memory maps.

    Compared with the original 6.6M-parameter flattening network, this network
    uses strided convolutions and adaptive pooling. For a 20x20 grid and
    ``feature_dim=128`` it has roughly half a million parameters.

    L2-normalized output is especially useful for a Bayesian linear head because
    the prior scale then has a stable interpretation:

        Var[Q(o,a)] = phi(o)^T Sigma_a phi(o)

    is not dominated merely by an uncontrolled feature norm.
    """

    def __init__(
        self,
        in_channels: int,
        grid_size: int,
        feature_dim: int = 128,
        *,
        normalize_output: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.grid_size = int(grid_size)
        self.feature_dim = int(feature_dim)
        self.normalize_output = bool(normalize_output)

        self.conv = nn.Sequential(
            nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )
        self.mlp = nn.Sequential(
            nn.Linear(96 * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, self.feature_dim),
            nn.LayerNorm(self.feature_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.mlp(self.conv(x))
        if self.normalize_output:
            features = F.normalize(features, p=2.0, dim=-1, eps=1e-6)
        return features


class QNetwork(nn.Module):
    """DQN/DDQN network: compact feature extractor plus linear Q head."""

    def __init__(
        self,
        obs_shape: tuple[int, int, int],
        action_dim: int,
        feature_dim: int = 128,
        *,
        normalize_features: bool = True,
    ) -> None:
        super().__init__()
        c, h, w = obs_shape
        if h != w:
            raise ValueError("GridFeatureNet assumes a square grid.")
        self.feature_net = GridFeatureNet(
            c,
            h,
            feature_dim,
            normalize_output=normalize_features,
        )
        self.head = nn.Linear(feature_dim, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.feature_net(x))

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from uav_search_belief20.models.networks import QNetwork
from uav_search_belief20.marl.qmix_mixer import QMixer


@dataclass
class QMIXConfig:
    obs_shape: tuple[int, int, int]
    state_dim: int
    n_agents: int = 3
    action_dim: int = 5
    feature_dim: int = 128
    mixing_embed_dim: int = 32
    lr: float = 1e-4
    gamma: float = 0.99
    device: str = "cpu"


class QMIXDDQNSkeleton(nn.Module):
    """Skeleton for phase 4: QMIX-DDQN.

    This file intentionally does not hide the complexity behind a framework. It gives you
    the two core modules that will be trained together later:
    - a shared per-agent DDQN utility network;
    - a monotonic QMixer using the global state.
    """

    def __init__(self, cfg: QMIXConfig):
        super().__init__()
        self.cfg = cfg
        self.agent_net = QNetwork(cfg.obs_shape, cfg.action_dim, cfg.feature_dim)
        self.target_agent_net = QNetwork(cfg.obs_shape, cfg.action_dim, cfg.feature_dim)
        self.mixer = QMixer(cfg.n_agents, cfg.state_dim, cfg.mixing_embed_dim)
        self.target_mixer = QMixer(cfg.n_agents, cfg.state_dim, cfg.mixing_embed_dim)
        self.target_agent_net.load_state_dict(self.agent_net.state_dict())
        self.target_mixer.load_state_dict(self.mixer.state_dict())

    def agent_qs(self, obs_batch: torch.Tensor) -> torch.Tensor:
        """obs_batch: [batch, n_agents, C, H, W] -> [batch, n_agents, action_dim]."""
        b, n, c, h, w = obs_batch.shape
        flat = obs_batch.reshape(b * n, c, h, w)
        q = self.agent_net(flat)
        return q.reshape(b, n, self.cfg.action_dim)

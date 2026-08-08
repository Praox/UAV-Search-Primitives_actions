from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class QMixer(nn.Module):
    """Monotonic QMIX mixing network."""

    def __init__(
        self,
        n_agents: int,
        state_dim: int,
        mixing_embed_dim: int = 32,
        hypernet_embed: int = 64,
    ) -> None:
        super().__init__()
        self.n_agents = int(n_agents)
        self.state_dim = int(state_dim)
        self.embed_dim = int(mixing_embed_dim)
        self.hypernet_embed = int(hypernet_embed)
        self.hyper_w1 = nn.Sequential(
            nn.Linear(self.state_dim, self.hypernet_embed),
            nn.ReLU(),
            nn.Linear(self.hypernet_embed, self.n_agents * self.embed_dim),
        )
        self.hyper_b1 = nn.Linear(self.state_dim, self.embed_dim)
        self.hyper_w_final = nn.Sequential(
            nn.Linear(self.state_dim, self.hypernet_embed),
            nn.ReLU(),
            nn.Linear(self.hypernet_embed, self.embed_dim),
        )
        self.v = nn.Sequential(
            nn.Linear(self.state_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, 1),
        )

    def forward(
        self, agent_qs: torch.Tensor, states: torch.Tensor
    ) -> torch.Tensor:
        if agent_qs.dim() != 2 or agent_qs.size(1) != self.n_agents:
            raise ValueError("agent_qs must be [batch, n_agents]")
        if states.dim() != 2 or states.size(1) != self.state_dim:
            raise ValueError("states must be [batch, state_dim]")
        batch_size = agent_qs.size(0)
        agent_qs = agent_qs.view(batch_size, 1, self.n_agents)
        w1 = torch.abs(self.hyper_w1(states)).view(
            batch_size, self.n_agents, self.embed_dim
        )
        b1 = self.hyper_b1(states).view(batch_size, 1, self.embed_dim)
        hidden = F.elu(torch.bmm(agent_qs, w1) + b1)
        w_final = torch.abs(self.hyper_w_final(states)).view(
            batch_size, self.embed_dim, 1
        )
        value = self.v(states).view(batch_size, 1, 1)
        return (torch.bmm(hidden, w_final) + value).view(batch_size)

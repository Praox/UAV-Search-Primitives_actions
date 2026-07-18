from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class QMixer(nn.Module):
    """QMIX monotonic mixing network.

    Inputs:
        agent_qs: Tensor of shape [batch, n_agents]
        states: Tensor of shape [batch, state_dim]

    Output:
        q_tot: Tensor of shape [batch]

    The absolute value on hypernetwork-generated weights enforces the QMIX
    monotonicity constraint: dQ_tot / dQ_i >= 0.
    """

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

        # State-dependent scalar bias V(s).
        self.v = nn.Sequential(
            nn.Linear(self.state_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, 1),
        )

    def forward(self, agent_qs: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        """Mix per-agent utilities into Q_tot.

        Args:
            agent_qs: [batch, n_agents]
            states: [batch, state_dim]

        Returns:
            q_tot: [batch]
        """
        if agent_qs.dim() != 2:
            raise ValueError(f"agent_qs must be [batch, n_agents], got {tuple(agent_qs.shape)}")
        if states.dim() != 2:
            raise ValueError(f"states must be [batch, state_dim], got {tuple(states.shape)}")
        if agent_qs.size(1) != self.n_agents:
            raise ValueError(f"Expected {self.n_agents} agent Qs, got {agent_qs.size(1)}")
        if states.size(1) != self.state_dim:
            raise ValueError(f"Expected state_dim={self.state_dim}, got {states.size(1)}")

        batch_size = agent_qs.size(0)

        agent_qs = agent_qs.view(batch_size, 1, self.n_agents)

        # First mixing layer.
        w1 = torch.abs(self.hyper_w1(states)).view(batch_size, self.n_agents, self.embed_dim)
        b1 = self.hyper_b1(states).view(batch_size, 1, self.embed_dim)
        hidden = F.elu(torch.bmm(agent_qs, w1) + b1)

        # Final mixing layer.
        w_final = torch.abs(self.hyper_w_final(states)).view(batch_size, self.embed_dim, 1)
        v = self.v(states).view(batch_size, 1, 1)
        y = torch.bmm(hidden, w_final) + v

        return y.view(batch_size)

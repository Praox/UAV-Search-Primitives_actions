from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class QMixer(nn.Module):
    """Minimal QMIX monotonic mixer.

    Inputs:
        agent_qs: [batch, n_agents]
        states: [batch, state_dim]
    Output:
        q_tot: [batch]

    This is included as the next-phase building block. The single-agent and shared-BDQN
    experiments do not depend on it.
    """

    def __init__(self, n_agents: int, state_dim: int, mixing_embed_dim: int = 32, hypernet_embed: int = 64):
        super().__init__()
        self.n_agents = int(n_agents)
        self.state_dim = int(state_dim)
        self.embed_dim = int(mixing_embed_dim)

        self.hyper_w1 = nn.Sequential(
            nn.Linear(self.state_dim, hypernet_embed),
            nn.ReLU(),
            nn.Linear(hypernet_embed, self.n_agents * self.embed_dim),
        )
        self.hyper_b1 = nn.Linear(self.state_dim, self.embed_dim)
        self.hyper_w_final = nn.Sequential(
            nn.Linear(self.state_dim, hypernet_embed),
            nn.ReLU(),
            nn.Linear(hypernet_embed, self.embed_dim),
        )
        self.v = nn.Sequential(
            nn.Linear(self.state_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, 1),
        )

    def forward(self, agent_qs: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        bs = agent_qs.size(0)
        agent_qs = agent_qs.view(bs, 1, self.n_agents)
        w1 = torch.abs(self.hyper_w1(states)).view(bs, self.n_agents, self.embed_dim)
        b1 = self.hyper_b1(states).view(bs, 1, self.embed_dim)
        hidden = F.elu(torch.bmm(agent_qs, w1) + b1)
        w_final = torch.abs(self.hyper_w_final(states)).view(bs, self.embed_dim, 1)
        v = self.v(states).view(bs, 1, 1)
        y = torch.bmm(hidden, w_final) + v
        return y.view(bs)

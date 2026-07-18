from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from uav_search_belief20.agents.replay_buffer import Batch, ReplayBuffer
from uav_search_belief20.models.networks import QNetwork


@dataclass
class DQNConfig:
    obs_shape: tuple[int, int, int] = (6, 20, 20)
    action_dim: int = 5
    feature_dim: int = 128
    gamma: float = 0.99
    lr: float = 1e-4
    batch_size: int = 64
    replay_capacity: int = 50_000
    target_update_period: int = 500
    double_dqn: bool = False
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 20_000
    grad_clip_norm: float = 10.0
    device: str = "cpu"
    seed: int = 42


class DQNAgent:
    def __init__(self, cfg: DQNConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.rng = np.random.default_rng(cfg.seed)
        self.q_net = QNetwork(cfg.obs_shape, cfg.action_dim, cfg.feature_dim).to(self.device)
        self.target_q_net = QNetwork(cfg.obs_shape, cfg.action_dim, cfg.feature_dim).to(self.device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.optim = torch.optim.Adam(self.q_net.parameters(), lr=cfg.lr)
        self.replay = ReplayBuffer(cfg.replay_capacity, cfg.obs_shape, seed=cfg.seed)
        self.train_steps = 0
        self.env_steps = 0

    def epsilon(self) -> float:
        frac = min(1.0, self.env_steps / max(1, self.cfg.epsilon_decay_steps))
        return float(self.cfg.epsilon_start + frac * (self.cfg.epsilon_end - self.cfg.epsilon_start))

    @torch.no_grad()
    def act(
        self,
        obs: np.ndarray,
        explore: bool = True,
        action_mask: np.ndarray | None = None,
    ) -> int:
        allowed = np.ones(self.cfg.action_dim, dtype=bool) if action_mask is None else action_mask.astype(bool)
        allowed_idx = np.flatnonzero(allowed)
        if len(allowed_idx) == 0:
            raise ValueError("No allowed action in action_mask.")

        if explore and self.rng.random() < self.epsilon():
            action = int(self.rng.choice(allowed_idx))
        else:
            x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            q = self.q_net(x).cpu().numpy()[0]
            q = q.copy()
            q[~allowed] = -1e9
            action = int(np.argmax(q))

        if explore:
            self.env_steps += 1
        return action

    def train_step(self) -> dict:
        if len(self.replay) < self.cfg.batch_size:
            return {"loss": 0.0, "q_mean": 0.0}
        batch = self.replay.sample(self.cfg.batch_size)
        loss, q_mean = self._gradient_update(batch)
        self.train_steps += 1
        if self.train_steps % self.cfg.target_update_period == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        return {"loss": float(loss), "q_mean": float(q_mean), "epsilon": self.epsilon()}

    def _gradient_update(self, batch: Batch):
        obs = torch.as_tensor(batch.obs, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch.actions, dtype=torch.long, device=self.device)
        rewards = torch.as_tensor(batch.rewards, dtype=torch.float32, device=self.device)
        next_obs = torch.as_tensor(batch.next_obs, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(batch.dones, dtype=torch.float32, device=self.device)

        q_all = self.q_net(obs)
        q = q_all.gather(1, actions.view(-1, 1)).squeeze(1)

        with torch.no_grad():
            if self.cfg.double_dqn:
                next_actions = self.q_net(next_obs).argmax(dim=1)
                next_q_all = self.target_q_net(next_obs)
                next_q = next_q_all.gather(1, next_actions.view(-1, 1)).squeeze(1)
            else:
                next_q = self.target_q_net(next_obs).max(dim=1).values
            y = rewards + self.cfg.gamma * (1.0 - dones) * next_q

        loss = F.mse_loss(q, y)
        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), self.cfg.grad_clip_norm)
        self.optim.step()
        return loss.item(), q.mean().item()

    def save(self, path: str) -> None:
        torch.save(
            {
                "cfg": self.cfg.__dict__,
                "q_net": self.q_net.state_dict(),
                "target_q_net": self.target_q_net.state_dict(),
                "optim": self.optim.state_dict(),
                "train_steps": self.train_steps,
                "env_steps": self.env_steps,
            },
            path,
        )

    def load(self, path: str) -> None:
        try:
            ckpt = torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:
            ckpt = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(ckpt["q_net"])
        self.target_q_net.load_state_dict(ckpt.get("target_q_net", ckpt["q_net"]))
        if "optim" in ckpt:
            self.optim.load_state_dict(ckpt["optim"])
        self.train_steps = int(ckpt.get("train_steps", 0))
        self.env_steps = int(ckpt.get("env_steps", 0))

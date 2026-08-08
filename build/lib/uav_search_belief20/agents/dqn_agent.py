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
    n_step: int = 3
    lr: float = 1e-4
    batch_size: int = 64
    replay_capacity: int = 50_000

    # If target_tau > 0, Polyak updates are used after every gradient step.
    # Set target_tau=0 to recover periodic hard copies.
    target_tau: float = 0.005
    target_update_period: int = 500

    double_dqn: bool = False
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 20_000
    huber_delta: float = 1.0
    normalize_features: bool = True
    grad_clip_norm: float = 10.0
    device: str = "cpu"
    seed: int = 42


class DQNAgent:
    def __init__(self, cfg: DQNConfig):
        if cfg.n_step <= 0:
            raise ValueError("n_step must be positive")
        if cfg.huber_delta <= 0:
            raise ValueError("huber_delta must be positive")
        if not 0.0 <= cfg.target_tau <= 1.0:
            raise ValueError("target_tau must be in [0, 1]")

        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.rng = np.random.default_rng(cfg.seed)
        self.q_net = QNetwork(
            cfg.obs_shape,
            cfg.action_dim,
            cfg.feature_dim,
            normalize_features=cfg.normalize_features,
        ).to(self.device)
        self.target_q_net = QNetwork(
            cfg.obs_shape,
            cfg.action_dim,
            cfg.feature_dim,
            normalize_features=cfg.normalize_features,
        ).to(self.device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.optim = torch.optim.Adam(self.q_net.parameters(), lr=cfg.lr)
        self.replay = ReplayBuffer(
            cfg.replay_capacity,
            cfg.obs_shape,
            seed=cfg.seed,
            action_dim=cfg.action_dim,
            n_step=cfg.n_step,
            gamma=cfg.gamma,
        )
        self.train_steps = 0
        self.env_steps = 0

    def epsilon(self) -> float:
        frac = min(1.0, self.env_steps / max(1, self.cfg.epsilon_decay_steps))
        return float(
            self.cfg.epsilon_start
            + frac * (self.cfg.epsilon_end - self.cfg.epsilon_start)
        )

    def advance_env_step(self, count: int = 1) -> None:
        """Advance epsilon time explicitly.

        In shared-policy multi-UAV training, call ``act(...,
        advance_env_step=False)`` for every UAV, then call this method once for
        the joint environment transition. This makes epsilon schedules directly
        comparable to QMIX.
        """

        self.env_steps += int(count)

    @staticmethod
    def _masked_argmax(q_values: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        masks = masks.bool()
        if q_values.shape != masks.shape:
            raise ValueError(f"q shape {q_values.shape} != mask shape {masks.shape}")
        if not torch.all(masks.any(dim=-1)):
            raise ValueError("At least one transition has no valid next action.")
        masked = q_values.masked_fill(~masks, torch.finfo(q_values.dtype).min)
        return masked.argmax(dim=-1)

    @torch.no_grad()
    def act(
        self,
        obs: np.ndarray,
        explore: bool = True,
        action_mask: np.ndarray | None = None,
        *,
        advance_env_step: bool = True,
    ) -> int:
        allowed = (
            np.ones(self.cfg.action_dim, dtype=bool)
            if action_mask is None
            else np.asarray(action_mask, dtype=bool)
        )
        allowed_idx = np.flatnonzero(allowed)
        if len(allowed_idx) == 0:
            raise ValueError("No allowed action in action_mask.")

        if explore and self.rng.random() < self.epsilon():
            action = int(self.rng.choice(allowed_idx))
        else:
            x = torch.as_tensor(
                obs, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            q = self.q_net(x).cpu().numpy()[0].copy()
            q[~allowed] = -np.inf
            action = int(np.argmax(q))

        if explore and advance_env_step:
            self.advance_env_step()
        return action

    def train_step(self) -> dict[str, float]:
        if len(self.replay) < self.cfg.batch_size:
            return {
                "loss": 0.0,
                "q_mean": 0.0,
                "target_mean": 0.0,
                "epsilon": self.epsilon(),
            }

        batch = self.replay.sample(self.cfg.batch_size)
        loss, q_mean, target_mean = self._gradient_update(batch)
        self.train_steps += 1
        self._update_target_network()
        return {
            "loss": float(loss),
            "q_mean": float(q_mean),
            "target_mean": float(target_mean),
            "epsilon": self.epsilon(),
        }

    def _update_target_network(self) -> None:
        tau = float(self.cfg.target_tau)
        if tau > 0.0:
            with torch.no_grad():
                for target, source in zip(
                    self.target_q_net.parameters(), self.q_net.parameters()
                ):
                    target.mul_(1.0 - tau).add_(source, alpha=tau)
        elif self.train_steps % self.cfg.target_update_period == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())

    def _gradient_update(self, batch: Batch) -> tuple[float, float, float]:
        obs = torch.as_tensor(batch.obs, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(
            batch.actions, dtype=torch.long, device=self.device
        )
        rewards = torch.as_tensor(
            batch.rewards, dtype=torch.float32, device=self.device
        )
        next_obs = torch.as_tensor(
            batch.next_obs, dtype=torch.float32, device=self.device
        )
        dones = torch.as_tensor(
            batch.dones, dtype=torch.float32, device=self.device
        )
        discounts = torch.as_tensor(
            batch.discounts, dtype=torch.float32, device=self.device
        )
        next_masks = torch.as_tensor(
            batch.next_action_masks, dtype=torch.bool, device=self.device
        )

        q_all = self.q_net(obs)
        q = q_all.gather(1, actions.view(-1, 1)).squeeze(1)

        with torch.no_grad():
            if self.cfg.double_dqn:
                next_online = self.q_net(next_obs)
                next_actions = self._masked_argmax(next_online, next_masks)
                next_target = self.target_q_net(next_obs)
                next_q = next_target.gather(
                    1, next_actions.view(-1, 1)
                ).squeeze(1)
            else:
                next_target = self.target_q_net(next_obs).masked_fill(
                    ~next_masks, torch.finfo(torch.float32).min
                )
                next_q = next_target.max(dim=1).values
            targets = rewards + discounts * (1.0 - dones) * next_q

        loss = F.smooth_l1_loss(
            q,
            targets,
            beta=float(self.cfg.huber_delta),
        )
        self.optim.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(
            self.q_net.parameters(), self.cfg.grad_clip_norm
        )
        self.optim.step()
        return loss.item(), q.mean().item(), targets.mean().item()

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
            checkpoint = torch.load(
                path, map_location=self.device, weights_only=False
            )
        except TypeError:
            checkpoint = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(checkpoint["q_net"])
        self.target_q_net.load_state_dict(
            checkpoint.get("target_q_net", checkpoint["q_net"])
        )
        if "optim" in checkpoint:
            self.optim.load_state_dict(checkpoint["optim"])
        self.train_steps = int(checkpoint.get("train_steps", 0))
        self.env_steps = int(checkpoint.get("env_steps", 0))

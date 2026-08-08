from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from uav_search_belief20.marl.qmix_mixer import QMixer
from uav_search_belief20.models.networks import QNetwork


@dataclass
class LocalQMIXConfig:
    obs_shape: tuple[int, int, int]
    state_dim: int
    n_agents: int = 3
    action_dim: int = 5
    feature_dim: int = 128
    mixing_embed_dim: int = 32
    mixing_hypernet_embed: int = 64
    gamma: float = 0.99
    lr: float = 1e-4
    batch_size: int = 64
    replay_capacity: int = 100_000
    target_update_period: int = 500
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 20_000
    grad_clip_norm: float = 10.0
    device: str = "cpu"
    seed: int = 42


class LocalJointBatch(NamedTuple):
    obs: np.ndarray
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_obs: np.ndarray
    next_states: np.ndarray
    dones: np.ndarray
    action_masks: np.ndarray
    next_action_masks: np.ndarray


class LocalJointReplayBuffer:
    def __init__(
        self,
        capacity: int,
        n_agents: int,
        obs_shape: tuple[int, int, int],
        state_dim: int,
        action_dim: int,
        seed: int = 42,
    ) -> None:
        self.capacity = int(capacity)
        self.n_agents = int(n_agents)
        self.obs_shape = tuple(obs_shape)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.rng = np.random.default_rng(seed)
        self.obs = np.zeros((capacity, n_agents, *obs_shape), dtype=np.float32)
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, n_agents), dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_obs = np.zeros_like(self.obs)
        self.next_states = np.zeros_like(self.states)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.action_masks = np.ones((capacity, n_agents, action_dim), dtype=bool)
        self.next_action_masks = np.ones_like(self.action_masks)
        self.pos = 0
        self.size = 0

    def __len__(self) -> int:
        return self.size

    def add(
        self,
        *,
        obs_all: np.ndarray,
        state: np.ndarray,
        actions: np.ndarray,
        reward: float,
        next_obs_all: np.ndarray,
        next_state: np.ndarray,
        done: bool,
        action_masks: np.ndarray,
        next_action_masks: np.ndarray,
    ) -> None:
        index = self.pos
        self.obs[index] = np.asarray(obs_all, dtype=np.float32)
        self.states[index] = np.asarray(state, dtype=np.float32)
        self.actions[index] = np.asarray(actions, dtype=np.int64)
        self.rewards[index] = float(reward)
        self.next_obs[index] = np.asarray(next_obs_all, dtype=np.float32)
        self.next_states[index] = np.asarray(next_state, dtype=np.float32)
        self.dones[index] = float(done)
        self.action_masks[index] = np.asarray(action_masks, dtype=bool)
        self.next_action_masks[index] = np.asarray(next_action_masks, dtype=bool)
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> LocalJointBatch:
        if self.size <= 0:
            raise RuntimeError("Cannot sample from an empty joint replay buffer")
        indices = self.rng.integers(0, self.size, size=int(batch_size))
        return LocalJointBatch(
            self.obs[indices],
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_obs[indices],
            self.next_states[indices],
            self.dones[indices],
            self.action_masks[indices],
            self.next_action_masks[indices],
        )


class LocalQMIXAgent:
    """QMIX-DDQN with local maps and no centralized input in ``act``."""

    def __init__(self, cfg: LocalQMIXConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.rng = np.random.default_rng(cfg.seed)
        self.agent_net = QNetwork(cfg.obs_shape, cfg.action_dim, cfg.feature_dim).to(self.device)
        self.target_agent_net = QNetwork(cfg.obs_shape, cfg.action_dim, cfg.feature_dim).to(self.device)
        self.target_agent_net.load_state_dict(self.agent_net.state_dict())
        self.mixer = QMixer(
            cfg.n_agents, cfg.state_dim, cfg.mixing_embed_dim, cfg.mixing_hypernet_embed
        ).to(self.device)
        self.target_mixer = QMixer(
            cfg.n_agents, cfg.state_dim, cfg.mixing_embed_dim, cfg.mixing_hypernet_embed
        ).to(self.device)
        self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.optim = torch.optim.Adam(
            list(self.agent_net.parameters()) + list(self.mixer.parameters()), lr=cfg.lr
        )
        self.replay = LocalJointReplayBuffer(
            cfg.replay_capacity,
            cfg.n_agents,
            cfg.obs_shape,
            cfg.state_dim,
            cfg.action_dim,
            cfg.seed,
        )
        self.train_steps = 0
        self.env_steps = 0

    def epsilon(self) -> float:
        fraction = min(1.0, self.env_steps / max(1, self.cfg.epsilon_decay_steps))
        return float(
            self.cfg.epsilon_start
            + fraction * (self.cfg.epsilon_end - self.cfg.epsilon_start)
        )

    @staticmethod
    def masked_argmax(q_values: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        masks = masks.bool()
        if q_values.shape != masks.shape:
            raise ValueError(f"q shape {q_values.shape} != mask shape {masks.shape}")
        if not torch.all(masks.any(dim=-1)):
            raise ValueError("At least one agent has no valid action")
        masked = q_values.masked_fill(~masks, torch.finfo(q_values.dtype).min)
        return masked.argmax(dim=-1)

    @torch.no_grad()
    def q_values(self, obs_all: np.ndarray) -> np.ndarray:
        obs = torch.as_tensor(obs_all, dtype=torch.float32, device=self.device)
        if obs.dim() != 4:
            raise ValueError(f"obs_all must be [N,C,H,W], got {tuple(obs.shape)}")
        return self.agent_net(obs).cpu().numpy()

    @torch.no_grad()
    def act(
        self,
        obs_all: np.ndarray,
        *,
        action_masks: np.ndarray,
        explore: bool = True,
    ) -> np.ndarray:
        q_values = self.q_values(obs_all)
        masks = np.asarray(action_masks, dtype=bool)
        expected = (self.cfg.n_agents, self.cfg.action_dim)
        if masks.shape != expected:
            raise ValueError(f"Expected masks {expected}, got {masks.shape}")
        actions = np.zeros(self.cfg.n_agents, dtype=np.int64)
        for agent_id in range(self.cfg.n_agents):
            allowed = np.flatnonzero(masks[agent_id])
            if allowed.size == 0:
                raise ValueError(f"Agent {agent_id} has no valid action")
            if explore and self.rng.random() < self.epsilon():
                actions[agent_id] = int(self.rng.choice(allowed))
            else:
                local_q = q_values[agent_id].copy()
                local_q[~masks[agent_id]] = -np.inf
                actions[agent_id] = int(np.argmax(local_q))
        if explore:
            self.env_steps += 1
        return actions

    def train_step(self) -> dict[str, float]:
        if len(self.replay) < self.cfg.batch_size:
            return {
                "loss": 0.0,
                "q_tot_mean": 0.0,
                "target_mean": 0.0,
                "epsilon": self.epsilon(),
            }
        batch = self.replay.sample(self.cfg.batch_size)
        loss, q_tot_mean, target_mean = self._gradient_update(batch)
        self.train_steps += 1
        if self.train_steps % self.cfg.target_update_period == 0:
            self.target_agent_net.load_state_dict(self.agent_net.state_dict())
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        return {
            "loss": float(loss),
            "q_tot_mean": float(q_tot_mean),
            "target_mean": float(target_mean),
            "epsilon": self.epsilon(),
        }

    def _gradient_update(self, batch: LocalJointBatch):
        cfg = self.cfg
        batch_size = int(batch.obs.shape[0])
        n_agents = cfg.n_agents
        obs = torch.as_tensor(batch.obs, dtype=torch.float32, device=self.device)
        states = torch.as_tensor(batch.states, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch.actions, dtype=torch.long, device=self.device)
        rewards = torch.as_tensor(batch.rewards, dtype=torch.float32, device=self.device)
        next_obs = torch.as_tensor(batch.next_obs, dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(batch.next_states, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(batch.dones, dtype=torch.float32, device=self.device)
        next_masks = torch.as_tensor(
            batch.next_action_masks, dtype=torch.bool, device=self.device
        )

        flat_obs = obs.reshape(batch_size * n_agents, *cfg.obs_shape)
        q_all = self.agent_net(flat_obs).reshape(batch_size, n_agents, cfg.action_dim)
        chosen_q = q_all.gather(2, actions.unsqueeze(-1)).squeeze(-1)
        q_tot = self.mixer(chosen_q, states)

        with torch.no_grad():
            flat_next = next_obs.reshape(batch_size * n_agents, *cfg.obs_shape)
            next_online = self.agent_net(flat_next).reshape(
                batch_size, n_agents, cfg.action_dim
            )
            next_actions = self.masked_argmax(next_online, next_masks)
            next_target = self.target_agent_net(flat_next).reshape(
                batch_size, n_agents, cfg.action_dim
            )
            next_chosen = next_target.gather(
                2, next_actions.unsqueeze(-1)
            ).squeeze(-1)
            target_q_tot = self.target_mixer(next_chosen, next_states)
            targets = rewards + cfg.gamma * (1.0 - dones) * target_q_tot

        loss = F.mse_loss(q_tot, targets)
        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.agent_net.parameters()) + list(self.mixer.parameters()),
            cfg.grad_clip_norm,
        )
        self.optim.step()
        return loss.item(), q_tot.mean().item(), targets.mean().item()

    def save(self, path: str) -> None:
        torch.save(
            {
                "cfg": self.cfg.__dict__,
                "agent_net": self.agent_net.state_dict(),
                "target_agent_net": self.target_agent_net.state_dict(),
                "mixer": self.mixer.state_dict(),
                "target_mixer": self.target_mixer.state_dict(),
                "optim": self.optim.state_dict(),
                "train_steps": self.train_steps,
                "env_steps": self.env_steps,
            },
            path,
        )

    def load(self, path: str) -> None:
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(path, map_location=self.device)
        self.agent_net.load_state_dict(checkpoint["agent_net"])
        self.target_agent_net.load_state_dict(
            checkpoint.get("target_agent_net", checkpoint["agent_net"])
        )
        self.mixer.load_state_dict(checkpoint["mixer"])
        self.target_mixer.load_state_dict(
            checkpoint.get("target_mixer", checkpoint["mixer"])
        )
        if "optim" in checkpoint:
            self.optim.load_state_dict(checkpoint["optim"])
        self.train_steps = int(checkpoint.get("train_steps", 0))
        self.env_steps = int(checkpoint.get("env_steps", 0))

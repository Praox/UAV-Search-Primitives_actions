from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import torch
import torch.nn.functional as F
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


class JointBatch(NamedTuple):
    obs: np.ndarray               # [B, N, C, H, W]
    states: np.ndarray            # [B, S]
    actions: np.ndarray           # [B, N]
    rewards: np.ndarray           # [B]
    next_obs: np.ndarray          # [B, N, C, H, W]
    next_states: np.ndarray       # [B, S]
    dones: np.ndarray             # [B]


class JointReplayBuffer:
    """Replay buffer for one full multi-agent transition per environment step.

    This is the key difference from independent/shared DDQN: QMIX must store the
    joint transition, not one transition per UAV.
    """

    def __init__(
        self,
        capacity: int,
        n_agents: int,
        obs_shape: tuple[int, int, int],
        state_dim: int,
        seed: int = 42,
    ) -> None:
        self.capacity = int(capacity)
        self.n_agents = int(n_agents)
        self.obs_shape = tuple(obs_shape)
        self.state_dim = int(state_dim)
        self.rng = np.random.default_rng(seed)

        self.obs = np.zeros((self.capacity, self.n_agents, *self.obs_shape), dtype=np.float32)
        self.states = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        self.actions = np.zeros((self.capacity, self.n_agents), dtype=np.int64)
        self.rewards = np.zeros((self.capacity,), dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, self.n_agents, *self.obs_shape), dtype=np.float32)
        self.next_states = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        self.dones = np.zeros((self.capacity,), dtype=np.float32)

        self.pos = 0
        self.size = 0

    def __len__(self) -> int:
        return self.size

    def add(
        self,
        obs_all: np.ndarray,
        state: np.ndarray,
        actions: np.ndarray,
        reward: float,
        next_obs_all: np.ndarray,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        idx = self.pos
        self.obs[idx] = np.asarray(obs_all, dtype=np.float32)
        self.states[idx] = np.asarray(state, dtype=np.float32)
        self.actions[idx] = np.asarray(actions, dtype=np.int64)
        self.rewards[idx] = float(reward)
        self.next_obs[idx] = np.asarray(next_obs_all, dtype=np.float32)
        self.next_states[idx] = np.asarray(next_state, dtype=np.float32)
        self.dones[idx] = float(done)

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> JointBatch:
        if self.size <= 0:
            raise ValueError("Cannot sample from an empty replay buffer.")
        idx = self.rng.integers(0, self.size, size=int(batch_size))
        return JointBatch(
            obs=self.obs[idx],
            states=self.states[idx],
            actions=self.actions[idx],
            rewards=self.rewards[idx],
            next_obs=self.next_obs[idx],
            next_states=self.next_states[idx],
            dones=self.dones[idx],
        )


class QMIXAgent:
    """QMIX-DDQN agent with parameter-shared per-agent utility network.

    - One shared DDQN utility network Q_i(o_i, a_i) is used by all UAVs.
    - The mixer learns Q_tot(Q_1, ..., Q_N, s_global).
    - Action selection is decentralized at execution time: each UAV greedily selects
      argmax_a Q_i(o_i, a). The mixer is used only during centralized training.
    """

    def __init__(self, cfg: QMIXConfig) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.rng = np.random.default_rng(cfg.seed)

        self.agent_net = QNetwork(cfg.obs_shape, cfg.action_dim, cfg.feature_dim).to(self.device)
        self.target_agent_net = QNetwork(cfg.obs_shape, cfg.action_dim, cfg.feature_dim).to(self.device)
        self.target_agent_net.load_state_dict(self.agent_net.state_dict())

        self.mixer = QMixer(
            n_agents=cfg.n_agents,
            state_dim=cfg.state_dim,
            mixing_embed_dim=cfg.mixing_embed_dim,
            hypernet_embed=cfg.mixing_hypernet_embed,
        ).to(self.device)
        self.target_mixer = QMixer(
            n_agents=cfg.n_agents,
            state_dim=cfg.state_dim,
            mixing_embed_dim=cfg.mixing_embed_dim,
            hypernet_embed=cfg.mixing_hypernet_embed,
        ).to(self.device)
        self.target_mixer.load_state_dict(self.mixer.state_dict())

        self.optim = torch.optim.Adam(
            list(self.agent_net.parameters()) + list(self.mixer.parameters()),
            lr=cfg.lr,
        )

        self.replay = JointReplayBuffer(
            capacity=cfg.replay_capacity,
            n_agents=cfg.n_agents,
            obs_shape=cfg.obs_shape,
            state_dim=cfg.state_dim,
            seed=cfg.seed,
        )

        self.train_steps = 0
        self.env_steps = 0

    def epsilon(self) -> float:
        frac = min(1.0, self.env_steps / max(1, self.cfg.epsilon_decay_steps))
        return float(self.cfg.epsilon_start + frac * (self.cfg.epsilon_end - self.cfg.epsilon_start))

    @torch.no_grad()
    def q_values(self, obs_all: np.ndarray) -> np.ndarray:
        obs = torch.as_tensor(obs_all, dtype=torch.float32, device=self.device)
        if obs.dim() != 4:
            raise ValueError(f"obs_all must be [n_agents, C, H, W], got {tuple(obs.shape)}")
        q = self.agent_net(obs)
        return q.detach().cpu().numpy()

    @torch.no_grad()
    def act(
        self,
        obs_all: np.ndarray,
        action_masks: np.ndarray | None = None,
        explore: bool = True,
    ) -> np.ndarray:
        q = self.q_values(obs_all)
        actions = np.zeros((self.cfg.n_agents,), dtype=np.int64)

        if action_masks is None:
            masks = np.ones((self.cfg.n_agents, self.cfg.action_dim), dtype=bool)
        else:
            masks = np.asarray(action_masks, dtype=bool)

        for i in range(self.cfg.n_agents):
            allowed = masks[i]
            allowed_idx = np.flatnonzero(allowed)
            if allowed_idx.size == 0:
                raise ValueError(f"No allowed action for agent {i}.")

            if explore and self.rng.random() < self.epsilon():
                actions[i] = int(self.rng.choice(allowed_idx))
            else:
                q_i = q[i].copy()
                q_i[~allowed] = -1e9
                actions[i] = int(np.argmax(q_i))

        if explore:
            # One joint action corresponds to one environment step.
            self.env_steps += 1

        return actions

    def train_step(self) -> dict:
        if len(self.replay) < self.cfg.batch_size:
            return {"loss": 0.0, "q_tot_mean": 0.0, "target_mean": 0.0, "epsilon": self.epsilon()}

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

    def _gradient_update(self, batch: JointBatch):
        cfg = self.cfg
        bsz = int(batch.obs.shape[0])
        n = cfg.n_agents

        obs = torch.as_tensor(batch.obs, dtype=torch.float32, device=self.device)
        states = torch.as_tensor(batch.states, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch.actions, dtype=torch.long, device=self.device)
        rewards = torch.as_tensor(batch.rewards, dtype=torch.float32, device=self.device)
        next_obs = torch.as_tensor(batch.next_obs, dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(batch.next_states, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(batch.dones, dtype=torch.float32, device=self.device)

        # Current Q_i(o_i, a_i).
        flat_obs = obs.reshape(bsz * n, *cfg.obs_shape)
        q_all = self.agent_net(flat_obs).reshape(bsz, n, cfg.action_dim)
        chosen_qs = q_all.gather(2, actions.unsqueeze(-1)).squeeze(-1)  # [B, N]
        q_tot = self.mixer(chosen_qs, states)  # [B]

        with torch.no_grad():
            flat_next_obs = next_obs.reshape(bsz * n, *cfg.obs_shape)

            # DDQN-style target: online net selects next actions, target net evaluates them.
            next_q_online = self.agent_net(flat_next_obs).reshape(bsz, n, cfg.action_dim)
            next_actions = next_q_online.argmax(dim=2, keepdim=True)  # [B, N, 1]

            next_q_target_all = self.target_agent_net(flat_next_obs).reshape(bsz, n, cfg.action_dim)
            next_chosen_qs = next_q_target_all.gather(2, next_actions).squeeze(-1)  # [B, N]

            target_q_tot = self.target_mixer(next_chosen_qs, next_states)
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
            ckpt = torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:
            ckpt = torch.load(path, map_location=self.device)

        self.agent_net.load_state_dict(ckpt["agent_net"])
        self.target_agent_net.load_state_dict(ckpt.get("target_agent_net", ckpt["agent_net"]))
        self.mixer.load_state_dict(ckpt["mixer"])
        self.target_mixer.load_state_dict(ckpt.get("target_mixer", ckpt["mixer"]))
        if "optim" in ckpt:
            self.optim.load_state_dict(ckpt["optim"])
        self.train_steps = int(ckpt.get("train_steps", 0))
        self.env_steps = int(ckpt.get("env_steps", 0))

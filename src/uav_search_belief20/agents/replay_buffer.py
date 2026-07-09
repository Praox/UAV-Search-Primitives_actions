from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Batch:
    obs: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_obs: np.ndarray
    dones: np.ndarray


class ReplayBuffer:
    def __init__(self, capacity: int, obs_shape: tuple[int, int, int], seed: int = 42):
        self.capacity = int(capacity)
        self.obs_shape = tuple(obs_shape)
        self.rng = np.random.default_rng(seed)
        self.obs = np.zeros((self.capacity, *self.obs_shape), dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, *self.obs_shape), dtype=np.float32)
        self.actions = np.zeros((self.capacity,), dtype=np.int64)
        self.rewards = np.zeros((self.capacity,), dtype=np.float32)
        self.dones = np.zeros((self.capacity,), dtype=np.float32)
        self.idx = 0
        self.full = False

    def __len__(self) -> int:
        return self.capacity if self.full else self.idx

    def add(self, obs, action: int, reward: float, next_obs, done: bool) -> None:
        self.obs[self.idx] = np.asarray(obs, dtype=np.float32)
        self.actions[self.idx] = int(action)
        self.rewards[self.idx] = float(reward)
        self.next_obs[self.idx] = np.asarray(next_obs, dtype=np.float32)
        self.dones[self.idx] = float(done)
        self.idx = (self.idx + 1) % self.capacity
        if self.idx == 0:
            self.full = True

    def sample(self, batch_size: int) -> Batch:
        n = len(self)
        if n <= 0:
            raise RuntimeError("Cannot sample from an empty replay buffer.")
        indices = self.rng.integers(0, n, size=int(batch_size))
        return Batch(
            obs=self.obs[indices],
            actions=self.actions[indices],
            rewards=self.rewards[indices],
            next_obs=self.next_obs[indices],
            dones=self.dones[indices],
        )

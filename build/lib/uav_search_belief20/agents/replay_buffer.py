from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque

import numpy as np


@dataclass
class Batch:
    """A sampled batch of already-aggregated n-step transitions.

    ``discounts[i]`` is :math:`\\gamma^{k_i}`, where ``k_i`` is the actual
    number of rewards accumulated in transition ``i``. It can be smaller than
    ``n_step`` near the end of an episode.
    """

    obs: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_obs: np.ndarray
    dones: np.ndarray
    discounts: np.ndarray
    action_masks: np.ndarray
    next_action_masks: np.ndarray


@dataclass
class _RawTransition:
    obs: np.ndarray
    action: int
    reward: float
    next_obs: np.ndarray
    done: bool
    action_mask: np.ndarray
    next_action_mask: np.ndarray


class ReplayBuffer:
    """Replay buffer with valid-action masks and per-stream n-step returns.

    The ``stream_id`` argument is important for parameter-shared multi-agent
    learning. Each UAV has its own temporal trajectory, so their transitions
    must not be interleaved in one n-step queue.

    Existing one-step calls remain valid because all new arguments have safe
    defaults.
    """

    def __init__(
        self,
        capacity: int,
        obs_shape: tuple[int, int, int],
        seed: int = 42,
        *,
        action_dim: int = 5,
        n_step: int = 1,
        gamma: float = 0.99,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if action_dim <= 0:
            raise ValueError("action_dim must be positive")
        if n_step <= 0:
            raise ValueError("n_step must be positive")
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1]")

        self.capacity = int(capacity)
        self.obs_shape = tuple(obs_shape)
        self.action_dim = int(action_dim)
        self.n_step = int(n_step)
        self.gamma = float(gamma)
        self.rng = np.random.default_rng(seed)

        self.obs = np.zeros((self.capacity, *self.obs_shape), dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, *self.obs_shape), dtype=np.float32)
        self.actions = np.zeros((self.capacity,), dtype=np.int64)
        self.rewards = np.zeros((self.capacity,), dtype=np.float32)
        self.dones = np.zeros((self.capacity,), dtype=np.float32)
        self.discounts = np.zeros((self.capacity,), dtype=np.float32)
        self.action_masks = np.ones(
            (self.capacity, self.action_dim), dtype=bool
        )
        self.next_action_masks = np.ones_like(self.action_masks)

        self.idx = 0
        self.full = False
        self._queues: dict[int, Deque[_RawTransition]] = {}

    def __len__(self) -> int:
        return self.capacity if self.full else self.idx

    def _normalize_mask(self, mask: np.ndarray | None) -> np.ndarray:
        if mask is None:
            return np.ones(self.action_dim, dtype=bool)
        out = np.asarray(mask, dtype=bool).reshape(-1)
        if out.shape != (self.action_dim,):
            raise ValueError(
                f"Expected action mask {(self.action_dim,)}, got {out.shape}."
            )
        if not np.any(out):
            raise ValueError("An action mask must allow at least one action.")
        return out.copy()

    def add(
        self,
        obs,
        action: int,
        reward: float,
        next_obs,
        done: bool,
        action_mask: np.ndarray | None = None,
        next_action_mask: np.ndarray | None = None,
        *,
        stream_id: int = 0,
    ) -> None:
        """Add one raw transition.

        The buffer converts raw transitions into n-step transitions before they
        become sampleable. On episode termination, the remaining shorter
        transitions in the corresponding stream are flushed automatically.
        """

        stream_id = int(stream_id)
        queue = self._queues.setdefault(stream_id, deque())
        queue.append(
            _RawTransition(
                obs=np.asarray(obs, dtype=np.float32).copy(),
                action=int(action),
                reward=float(reward),
                next_obs=np.asarray(next_obs, dtype=np.float32).copy(),
                done=bool(done),
                action_mask=self._normalize_mask(action_mask),
                next_action_mask=self._normalize_mask(next_action_mask),
            )
        )

        if done:
            while queue:
                self._commit_oldest(queue)
        elif len(queue) >= self.n_step:
            self._commit_oldest(queue)

    def _commit_oldest(self, queue: Deque[_RawTransition]) -> None:
        if not queue:
            return

        first = queue[0]
        discounted_reward = 0.0
        steps = 0
        final = first

        for transition in list(queue)[: self.n_step]:
            discounted_reward += (self.gamma**steps) * transition.reward
            steps += 1
            final = transition
            if transition.done:
                break

        index = self.idx
        self.obs[index] = first.obs
        self.actions[index] = first.action
        self.rewards[index] = float(discounted_reward)
        self.next_obs[index] = final.next_obs
        self.dones[index] = float(final.done)
        self.discounts[index] = float(self.gamma**steps)
        self.action_masks[index] = first.action_mask
        self.next_action_masks[index] = final.next_action_mask

        self.idx = (self.idx + 1) % self.capacity
        if self.idx == 0:
            self.full = True
        queue.popleft()

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
            discounts=self.discounts[indices],
            action_masks=self.action_masks[indices],
            next_action_masks=self.next_action_masks[indices],
        )

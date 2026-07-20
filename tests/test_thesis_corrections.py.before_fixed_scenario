from __future__ import annotations

import numpy as np
import torch

from uav_search_belief20.agents.dqn_agent import DQNAgent, DQNConfig
from uav_search_belief20.agents.replay_buffer import ReplayBuffer
from uav_search_belief20.models.networks import QNetwork


def test_n_step_return_and_masks() -> None:
    buffer = ReplayBuffer(
        32,
        (2, 4, 4),
        action_dim=3,
        n_step=3,
        gamma=0.9,
        seed=0,
    )
    obs = np.zeros((2, 4, 4), dtype=np.float32)
    current = np.array([True, True, False])
    next_mask = np.array([True, False, True])
    for step in range(3):
        buffer.add(
            obs,
            0,
            1.0,
            obs,
            step == 2,
            current,
            next_mask,
        )
    assert len(buffer) == 3
    assert np.isclose(buffer.rewards[0], 1.0 + 0.9 + 0.9**2)
    assert np.isclose(buffer.discounts[0], 0.9**3)
    assert np.array_equal(buffer.next_action_masks[0], next_mask)


def test_shared_epsilon_clock_can_advance_once() -> None:
    agent = DQNAgent(
        DQNConfig(
            obs_shape=(2, 4, 4),
            action_dim=3,
            feature_dim=8,
            epsilon_decay_steps=100,
        )
    )
    obs = np.zeros((2, 4, 4), dtype=np.float32)
    mask = np.ones(3, dtype=bool)
    for _ in range(3):
        agent.act(
            obs,
            explore=True,
            action_mask=mask,
            advance_env_step=False,
        )
    assert agent.env_steps == 0
    agent.advance_env_step()
    assert agent.env_steps == 1


def test_network_is_compact() -> None:
    network = QNetwork((8, 20, 20), 5, feature_dim=128)
    parameters = sum(parameter.numel() for parameter in network.parameters())
    assert parameters < 800_000
    obs = torch.zeros(2, 8, 20, 20)
    q = network(obs)
    assert q.shape == (2, 5)


def test_masked_argmax_excludes_invalid_action() -> None:
    q_values = torch.tensor([[1.0, 100.0, 2.0]])
    masks = torch.tensor([[True, False, True]])
    selected = DQNAgent._masked_argmax(q_values, masks)
    assert selected.item() == 2

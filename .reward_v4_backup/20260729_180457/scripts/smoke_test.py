from __future__ import annotations

import numpy as np

from uav_search_belief20.agents.bdqn_agent import BDQNAgent, BDQNConfig
from uav_search_belief20.agents.dqn_agent import DQNAgent, DQNConfig
from uav_search_belief20.envs.multi_drone_env import MultiDroneEnvConfig, MultiDronePrimitiveSearchEnv
from uav_search_belief20.envs.primitive_search_env import EnvConfig, PrimitiveSearchEnv


def single_env_smoke() -> None:
    env = PrimitiveSearchEnv(EnvConfig(seed=0, max_steps=10))
    obs, info = env.reset()
    assert obs.shape == env.observation_shape
    total = 0.0
    for _ in range(10):
        a = int(np.random.randint(env.action_dim))
        obs, r, terminated, truncated, info = env.step(a)
        total += r
        if terminated or truncated:
            break
    print("single env OK", obs.shape, info["t"], round(total, 3))


def agents_smoke() -> None:
    env = PrimitiveSearchEnv(EnvConfig(seed=1, max_steps=5))
    obs, _ = env.reset()
    dqn = DQNAgent(DQNConfig(obs_shape=env.observation_shape, action_dim=env.action_dim, batch_size=2))
    bdqn = BDQNAgent(BDQNConfig(obs_shape=env.observation_shape, action_dim=env.action_dim, batch_size=2))
    for agent in [dqn, bdqn]:
        next_obs, r, terminated, truncated, _ = env.step(0)
        agent.replay.add(obs, 0, r, next_obs, terminated or truncated)
        agent.replay.add(next_obs, 1, r, obs, False)
        out = agent.train_step()
        print(agent.__class__.__name__, "OK", out)


def multi_env_smoke() -> None:
    env = MultiDronePrimitiveSearchEnv(MultiDroneEnvConfig(seed=2, max_steps=5, n_agents=3))
    obs, info = env.reset()
    assert obs.shape == (3, *env.observation_shape)
    obs, r, terminated, truncated, info = env.step(np.array([0, 1, 4]))
    print("multi env OK", obs.shape, round(r, 3), info["t"])


if __name__ == "__main__":
    single_env_smoke()
    agents_smoke()
    multi_env_smoke()

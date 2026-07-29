from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from uav_search_belief20.agents.dqn_agent import DQNAgent, DQNConfig
from uav_search_belief20.baselines import make_baseline
from uav_search_belief20.envs.primitive_search_env import PrimitiveSearchEnv
from uav_search_belief20.experiments.single_ablation import ABLATIONS, build_env_config


def args_for(ablation: str) -> SimpleNamespace:
    return SimpleNamespace(
        ablation=ablation,
        grid_size=20,
        n_value1_targets=3,
        n_value2_targets=1,
        sensor_radius=2,
        detection_probability=1.0,
        track_radius=1,
        track_required=3,
        max_steps=20,
        reward_version="v3_frontier",
        track_progress_scale=None,
    )


def make_env(ablation: str, seed: int = 42) -> PrimitiveSearchEnv:
    return PrimitiveSearchEnv(build_env_config(args_for(ablation), seed=seed))


def main() -> None:
    for variant in ABLATIONS:
        env = make_env(variant)
        obs, _ = env.reset()
        expected_channels = 7 if variant in {"B", "C", "D"} else 6
        assert obs.shape == (expected_channels, 20, 20), (variant, obs.shape)
        assert env.observation_shape == obs.shape

        agent = DQNAgent(
            DQNConfig(
                obs_shape=env.observation_shape,
                action_dim=env.action_dim,
                feature_dim=16,
                replay_capacity=32,
                batch_size=4,
                device="cpu",
                seed=1,
            )
        )
        action = agent.act(obs, explore=False, action_mask=env.action_mask())
        assert env.action_mask()[action]

    # Boundary mask: UP and LEFT must be invalid in the top-left corner.
    masked_env = make_env("A")
    masked_env.drone_pos[:] = (0, 0)
    mask = masked_env.action_mask()
    assert mask.tolist() == [True, False, True, False, True], mask

    # Historical v3 keeps every action available.
    historical_env = make_env("v3")
    historical_env.drone_pos[:] = (0, 0)
    assert historical_env.action_mask().all()

    # Tracking progress must be observable and normalized in B/C/D.
    progress_env = make_env("C")
    progress_env.drone_pos[:] = (5, 5)
    progress_env.memory.add_or_update_target(0, (5, 5), value=1, step=0)
    progress_env.track_progress[0] = 2
    progress_env.memory.update_target_progress(0, progress=2, completed=False)
    obs = progress_env._obs()
    assert np.isclose(obs[3, 5, 5], 2.0 / 3.0), obs[3, 5, 5]

    # Reward decomposition must exactly reconstruct the scalar reward.
    env = make_env("C")
    obs, _ = env.reset()
    action = int(np.flatnonzero(env.action_mask())[0])
    _, reward, _, _, info = env.step(action)
    assert np.isclose(reward, sum(info["last_reward_parts"].values()))

    # Every heuristic must obey action masks for multiple steps.
    for baseline_name in ("random", "frontier", "oracle"):
        env = make_env("C", seed=7)
        policy = make_baseline(baseline_name, seed=9)
        obs, _ = env.reset()
        for step in range(20):
            action = policy.act(env, obs, step)
            assert env.action_mask()[action], (baseline_name, action, env.drone_pos)
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break

    print("single-UAV v4 smoke test: OK")


if __name__ == "__main__":
    main()

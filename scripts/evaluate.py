from __future__ import annotations

import argparse
from collections import Counter

import numpy as np
import torch
  
from uav_search_belief20.actions import ACTION_NAMES
from uav_search_belief20.agents.bdqn_agent import BDQNAgent, BDQNConfig
from uav_search_belief20.agents.dqn_agent import DQNAgent, DQNConfig
from uav_search_belief20.envs.primitive_search_env import EnvConfig, PrimitiveSearchEnv
from uav_search_belief20.utils import pick_device, seed_everything


def make_env(args, seed_offset: int = 0) -> PrimitiveSearchEnv:
    return PrimitiveSearchEnv(
        EnvConfig(
            grid_size=args.grid_size,
            n_value1_targets=args.n_value1_targets,
            n_value2_targets=args.n_value2_targets,
            sensor_radius=args.sensor_radius,
            detection_probability=args.detection_probability,
            track_radius=args.track_radius,
            track_required=args.track_required,
            max_steps=args.max_steps,
            seed=args.seed + seed_offset,
        )
    )


def make_agent(args, env: PrimitiveSearchEnv, device: str):
    if args.algo in {"dqn", "ddqn"}:
        return DQNAgent(
            DQNConfig(
                obs_shape=env.observation_shape,
                action_dim=env.action_dim,
                double_dqn=args.algo == "ddqn",
                device=device,
                seed=args.seed,
            )
        )
    return BDQNAgent(
        BDQNConfig(
            obs_shape=env.observation_shape,
            action_dim=env.action_dim,
            device=device,
            seed=args.seed,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["dqn", "ddqn", "bdqn"], required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--grid-size", type=int, default=20)
    parser.add_argument("--n-value1-targets", type=int, default=3)
    parser.add_argument("--n-value2-targets", type=int, default=1)
    parser.add_argument("--sensor-radius", type=int, default=2)
    parser.add_argument("--detection-probability", type=float, default=1.0)
    parser.add_argument("--track-radius", type=int, default=1)
    parser.add_argument("--track-required", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--torch-threads", type=int, default=1)
    args = parser.parse_args()

    torch.set_num_threads(max(1, args.torch_threads))
    seed_everything(args.seed)
    device = pick_device()
    env0 = make_env(args)
    agent = make_agent(args, env0, device)
    agent.load(args.checkpoint)

    rewards, detected, completed, coverage = [], [], [], []
    detected_value, completed_value = [], []
    action_counts = Counter()
    boundary_hits = 0
    revisit_steps = 0
    decisions = 0

    for ep in range(args.episodes):
        env = make_env(args, seed_offset=1000 + ep)
        obs, info = env.reset()
        done = False
        total = 0.0
        while not done:
            if isinstance(agent, BDQNAgent):
                action = agent.act(obs, use_sample=False, action_mask=env.action_mask())
            else:
                action = agent.act(obs, explore=False, action_mask=env.action_mask())
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total += reward
            action_counts[ACTION_NAMES[action]] += 1
            boundary_hits += int(info["last_boundary_hit"])
            revisit_steps += int(not info["last_new_cell"])
            decisions += 1
        rewards.append(total)
        detected.append(info["detected"])
        completed.append(info["completed"])
        detected_value.append(info["detected_value"])
        completed_value.append(info["completed_value"])
        coverage.append(info["visited_ratio"])

    metrics = {
        "reward_mean": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
        "detected_mean": float(np.mean(detected)),
        "completed_mean": float(np.mean(completed)),
        "detected_value_mean": float(np.mean(detected_value)),
        "completed_value_mean": float(np.mean(completed_value)),
        "sensor_coverage_ratio_mean": float(np.mean(coverage)),
        "stay_ratio": action_counts["stay"] / max(1, decisions),
        "boundary_hit_ratio": boundary_hits / max(1, decisions),
        "revisit_ratio": revisit_steps / max(1, decisions),
        "action_counts": dict(action_counts),
    }
    for k, v in metrics.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()

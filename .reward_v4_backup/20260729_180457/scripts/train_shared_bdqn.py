from __future__ import annotations

import argparse
import csv
from collections import deque
from pathlib import Path

import numpy as np
import torch
from tqdm import trange

from uav_search_belief20.agents.bdqn_agent import BDQNAgent, BDQNConfig
from uav_search_belief20.envs.multi_drone_env import MultiDroneEnvConfig, MultiDronePrimitiveSearchEnv
from uav_search_belief20.utils import pick_device, seed_everything


def make_env(args, seed_offset: int = 0) -> MultiDronePrimitiveSearchEnv:
    return MultiDronePrimitiveSearchEnv(
        MultiDroneEnvConfig(
            grid_size=args.grid_size,
            n_agents=args.n_agents,
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


def evaluate(agent: BDQNAgent, args, episodes: int = 20) -> dict:
    rewards, detected, completed, coverage = [], [], [], []
    detected_value, completed_value = [], []
    for ep in range(episodes):
        env = make_env(args, seed_offset=10_000 + ep)
        obs_all, info = env.reset()
        done = False
        total = 0.0
        while not done:
            masks = env.action_mask()
            actions = np.array([
                agent.act(obs_all[i], use_sample=False, action_mask=masks[i])
                for i in range(args.n_agents)
            ], dtype=np.int64)
            next_obs_all, reward, terminated, truncated, info = env.step(actions)
            done = terminated or truncated
            total += reward
            obs_all = next_obs_all
        rewards.append(total)
        detected.append(info["detected"])
        completed.append(info["completed"])
        detected_value.append(info["detected_value"])
        completed_value.append(info["completed_value"])
        coverage.append(info["visited_ratio"])
    return {
        "eval_reward": float(np.mean(rewards)),
        "eval_detected": float(np.mean(detected)),
        "eval_completed": float(np.mean(completed)),
        "eval_detected_value": float(np.mean(detected_value)),
        "eval_completed_value": float(np.mean(completed_value)),
        "eval_sensor_coverage": float(np.mean(coverage)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--n-agents", type=int, default=3)
    parser.add_argument("--grid-size", type=int, default=20)
    parser.add_argument("--n-value1-targets", type=int, default=3)
    parser.add_argument("--n-value2-targets", type=int, default=1)
    parser.add_argument("--sensor-radius", type=int, default=2)
    parser.add_argument("--detection-probability", type=float, default=1.0)
    parser.add_argument("--track-radius", type=int, default=1)
    parser.add_argument("--track-required", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-dir", type=str, default="runs/shared_bdqn_3uav")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--replay-capacity", type=int, default=100_000)
    parser.add_argument("--target-update-period", type=int, default=500)
    parser.add_argument("--posterior-update-period", type=int, default=100)
    parser.add_argument("--torch-threads", type=int, default=1)
    args = parser.parse_args()

    torch.set_num_threads(max(1, args.torch_threads))
    seed_everything(args.seed)
    device = pick_device()
    env = make_env(args)
    agent = BDQNAgent(
        BDQNConfig(
            obs_shape=env.observation_shape,
            action_dim=env.action_dim,
            device=device,
            seed=args.seed,
            lr=args.lr,
            batch_size=args.batch_size,
            replay_capacity=args.replay_capacity,
            target_update_period=args.target_update_period,
            posterior_update_period=args.posterior_update_period,
        )
    )

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.csv"
    with metrics_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "episode", "train_reward", "train_detected", "train_completed",
            "eval_reward", "eval_detected", "eval_completed", "eval_detected_value",
            "eval_completed_value", "eval_sensor_coverage", "best_score",
        ])
        writer.writeheader()

    recent_reward = deque(maxlen=50)
    recent_completed = deque(maxlen=50)
    best_score = -1e18
    best_ep = 0
    pbar = trange(args.episodes, desc="Shared BDQN independent")
    for ep in pbar:
        obs_all, info = env.reset()
        agent.resample_policy()
        done = False
        ep_reward = 0.0
        while not done:
            masks = env.action_mask()
            actions = np.array([
                agent.act(obs_all[i], use_sample=True, action_mask=masks[i])
                for i in range(args.n_agents)
            ], dtype=np.int64)
            next_obs_all, reward, terminated, truncated, info = env.step(actions)
            done = terminated or truncated
            # Shared independent baseline: each UAV transition receives the team reward.
            for i in range(args.n_agents):
                agent.replay.add(obs_all[i], int(actions[i]), reward, next_obs_all[i], done)
            agent.train_step()
            obs_all = next_obs_all
            ep_reward += reward

        recent_reward.append(ep_reward)
        recent_completed.append(info["completed"])
        pbar.set_postfix(reward=f"{np.mean(recent_reward):.2f}", comp=f"{np.mean(recent_completed):.2f}", best=f"{best_score:.2f}")

        if (ep + 1) % args.eval_every == 0 or (ep + 1) == args.episodes:
            metrics = evaluate(agent, args, episodes=args.eval_episodes)
            score = metrics["eval_reward"] + 2.0 * metrics["eval_completed"] + metrics["eval_detected"]
            if score > best_score:
                best_score = score
                best_ep = ep + 1
                agent.save(str(run_dir / "best.pt"))
                print(f"\n[Best] episode={best_ep} score={best_score:.3f} metrics={metrics}")
            agent.save(str(run_dir / "latest.pt"))
            with metrics_path.open("a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "episode", "train_reward", "train_detected", "train_completed",
                    "eval_reward", "eval_detected", "eval_completed", "eval_detected_value",
                    "eval_completed_value", "eval_sensor_coverage", "best_score",
                ])
                writer.writerow({
                    "episode": ep + 1,
                    "train_reward": ep_reward,
                    "train_detected": info["detected"],
                    "train_completed": info["completed"],
                    **metrics,
                    "best_score": best_score,
                })
            print(f"\n[Eval {ep + 1}] {metrics}")

    print(f"Done. Best checkpoint: {run_dir / 'best.pt'} episode={best_ep} score={best_score:.3f}")


if __name__ == "__main__":
    main()

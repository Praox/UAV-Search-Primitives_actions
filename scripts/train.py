from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter, deque
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from tqdm import trange

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
            reward_version=args.reward_version,
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
                lr=args.lr,
                batch_size=args.batch_size,
                replay_capacity=args.replay_capacity,
                target_update_period=args.target_update_period,
                epsilon_start=args.epsilon_start,
                epsilon_end=args.epsilon_end,
                epsilon_decay_steps=args.epsilon_decay_steps,
            )
        )
    if args.algo == "bdqn":
        return BDQNAgent(
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
                blr_lambda=args.blr_lambda,
                blr_noise_var=args.blr_noise_var,
            )
        )
    raise ValueError(f"Unknown algo: {args.algo}")


def evaluate(agent, args, episodes: int = 20) -> dict:
    rewards, detected, completed, coverage = [], [], [], []
    detected_value, completed_value = [], []
    action_counts = Counter()
    boundary_hits = 0
    position_revisit_steps = 0
    sensor_revisit_steps = 0
    tracking_progress_steps = 0
    new_observed_total = 0
    decisions = 0

    for ep in range(episodes):
        env = make_env(args, seed_offset=10_000 + ep)
        obs, info = env.reset()
        done = False
        total = 0.0
        while not done:
            if isinstance(agent, BDQNAgent):
                action = agent.act(obs, use_sample=False, action_mask=env.action_mask())
            else:
                action = agent.act(obs, explore=False, action_mask=env.action_mask())
            next_obs, reward, terminated, truncated, info = env.step(action)
            total += reward
            done = terminated or truncated
            obs = next_obs

            action_counts[ACTION_NAMES[action]] += 1
            boundary_hits += int(info["last_boundary_hit"])
            position_revisit_steps += int(not info["last_new_cell"])
            sensor_revisit_steps += int(info.get("last_new_observed_cells", 0) == 0)
            new_observed_total += int(info.get("last_new_observed_cells", 0))
            tracking_progress_steps += int(info.get("last_tracking_progress", False))
            decisions += 1

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
        "stay_ratio": action_counts["stay"] / max(1, decisions),
        "boundary_hit_ratio": boundary_hits / max(1, decisions),
        # keep old metric for backward compatibility
        "revisit_ratio": position_revisit_steps / max(1, decisions),
        # V2 metric: this is the meaningful revisit metric for sensor-based exploration
        "sensor_revisit_ratio": sensor_revisit_steps / max(1, decisions),
        "new_observed_cells_per_step": new_observed_total / max(1, decisions),
        "tracking_progress_ratio": tracking_progress_steps / max(1, decisions),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["dqn", "ddqn", "bdqn"], default="bdqn")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--grid-size", type=int, default=20)
    parser.add_argument("--n-value1-targets", type=int, default=3)
    parser.add_argument("--n-value2-targets", type=int, default=1)
    parser.add_argument("--sensor-radius", type=int, default=2)
    parser.add_argument("--detection-probability", type=float, default=1.0)
    parser.add_argument("--track-radius", type=int, default=1)
    parser.add_argument("--track-required", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--reward-version", type=str, default="v2_frontier")
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-dir", type=str, default="runs/debug")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--replay-capacity", type=int, default=50_000)
    parser.add_argument("--target-update-period", type=int, default=500)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay-steps", type=int, default=20_000)
    parser.add_argument("--posterior-update-period", type=int, default=500)
    parser.add_argument("--blr-lambda", type=float, default=1.0)
    parser.add_argument("--blr-noise-var", type=float, default=1.0)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--train-every", type=int, default=4)
    parser.add_argument("--learning-starts", type=int, default=1000)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    torch.set_num_threads(max(1, args.torch_threads))
    seed_everything(args.seed)
    device = pick_device() if args.device == "auto" else args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA demandé avec --device cuda, mais torch.cuda.is_available() == False")
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS demandé avec --device mps, mais torch.backends.mps.is_available() == False")

    env = make_env(args)
    agent = make_agent(args, env, device)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {device}")
    print(f"torch threads: {args.torch_threads}")
    print(f"train_every: {args.train_every}, learning_starts: {args.learning_starts}")
    print(f"reward_version: {env.cfg.reward_version}")
    print("reward_config:")
    for k, v in env.cfg.reward_dict().items():
        if "bonus" in k or "penalty" in k or "reward_version" in k:
            print(f"  {k}: {v}")

    run_config = {
        "args": vars(args),
        "device": device,
        "env_config": env.cfg.reward_dict(),
        "reward_version": env.cfg.reward_version,
    }
    with (run_dir / "run_config.json").open("w") as f:
        json.dump(run_config, f, indent=2)

    metrics_path = run_dir / "metrics.csv"
    fieldnames = [
        "episode", "reward_version", "train_reward", "train_detected", "train_completed",
        "eval_reward", "eval_detected", "eval_completed", "eval_detected_value",
        "eval_completed_value", "eval_sensor_coverage", "stay_ratio", "boundary_hit_ratio",
        "revisit_ratio", "sensor_revisit_ratio", "new_observed_cells_per_step",
        "tracking_progress_ratio", "best_score",
    ]
    with metrics_path.open("w", newline="") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    recent_reward = deque(maxlen=50)
    recent_detected = deque(maxlen=50)
    recent_completed = deque(maxlen=50)
    best_score = -1e18
    best_ep = 0
    global_steps = 0
    prof = {"act": 0.0, "env": 0.0, "replay": 0.0, "train": 0.0, "eval": 0.0, "save": 0.0, "steps": 0}

    #pbar = trange(args.episodes, desc=f"{args.algo.upper()} {args.reward_version}")
    pbar = range(args.episodes)
    for ep in pbar:
        obs, info = env.reset()
        if isinstance(agent, BDQNAgent):
            agent.resample_policy()
        done = False
        ep_reward = 0.0

        while not done:
            if args.profile:
                t0 = time.perf_counter()
            if isinstance(agent, BDQNAgent):
                action = agent.act(obs, use_sample=True, action_mask=env.action_mask())
            else:
                action = agent.act(obs, explore=True, action_mask=env.action_mask())
            if args.profile:
                prof["act"] += time.perf_counter() - t0

            if args.profile:
                t0 = time.perf_counter()
            next_obs, reward, terminated, truncated, info = env.step(action)
            if args.profile:
                prof["env"] += time.perf_counter() - t0

            done = terminated or truncated

            if args.profile:
                t0 = time.perf_counter()
            agent.replay.add(obs, action, reward, next_obs, done)
            if args.profile:
                prof["replay"] += time.perf_counter() - t0

            if args.profile:
                t0 = time.perf_counter()
            if global_steps >= args.learning_starts and global_steps % args.train_every == 0:
                agent.train_step()
            if args.profile:
                prof["train"] += time.perf_counter() - t0

            obs = next_obs
            ep_reward += reward
            global_steps += 1
            prof["steps"] += 1

        recent_reward.append(ep_reward)
        recent_detected.append(info["detected"])
        recent_completed.append(info["completed"])
        """
        postfix = {
            "reward": f"{np.mean(recent_reward):.2f}",
            "det": f"{np.mean(recent_detected):.2f}",
            "comp": f"{np.mean(recent_completed):.2f}",
            "best": f"{best_score:.2f}",
        }
        if isinstance(agent, DQNAgent):
            postfix["eps"] = f"{agent.epsilon():.2f}"
        pbar.set_postfix(postfix)

"""

        if (ep + 1) % args.eval_every == 0 or (ep + 1) == args.episodes:
            if args.profile:
                t0 = time.perf_counter()
            metrics = evaluate(agent, args, episodes=args.eval_episodes)
            if args.profile:
                prof["eval"] += time.perf_counter() - t0

            # Score used only for checkpoint selection. Real comparison should use evaluate.py.
            score = metrics["eval_reward"] + 2.0 * metrics["eval_completed"] + metrics["eval_detected"]

            if args.profile:
                t0 = time.perf_counter()
            if score > best_score:
                best_score = score
                best_ep = ep + 1
                agent.save(str(run_dir / "best.pt"))
                print(f"\n[Best] episode={best_ep} score={best_score:.3f} metrics={metrics}")
            if (ep + 1) % args.save_every == 0 or (ep + 1) == args.episodes:
                agent.save(str(run_dir / "latest.pt"))
            if args.profile:
                prof["save"] += time.perf_counter() - t0

            with metrics_path.open("a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writerow({
                    "episode": ep + 1,
                    "reward_version": env.cfg.reward_version,
                    "train_reward": ep_reward,
                    "train_detected": info["detected"],
                    "train_completed": info["completed"],
                    **metrics,
                    "best_score": best_score,
                })
            print(f"\n[Eval {ep + 1}] {metrics}")

    if args.profile:
        total = sum(prof[k] for k in ["act", "env", "replay", "train", "eval", "save"])
        print("\n=== PROFILE ===")
        print(f"steps: {prof['steps']}")
        print(f"total_profiled_time: {total:.2f}s")
        for key in ["act", "env", "replay", "train", "eval", "save"]:
            pct = 100.0 * prof[key] / max(total, 1e-9)
            ms_per_step = 1000.0 * prof[key] / max(prof["steps"], 1)
            print(f"{key:>8}: {prof[key]:8.2f}s | {pct:6.2f}% | {ms_per_step:8.3f} ms/step")
        if prof["steps"] > 0 and total > 0:
            print(f"throughput: {prof['steps'] / total:.2f} profiled steps/s")

    print("Training complete.")
    print(f"Best checkpoint: {run_dir / 'best.pt'} at episode {best_ep}, score={best_score:.3f}")


if __name__ == "__main__":
    main()

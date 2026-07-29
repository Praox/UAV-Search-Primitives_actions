from __future__ import annotations

import argparse
import contextlib
import csv
import json
import sys
import time
from collections import Counter, deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from tqdm import trange

from uav_search_belief20.actions import ACTION_NAMES
from uav_search_belief20.agents.bdqn_agent import BDQNAgent, BDQNConfig
from uav_search_belief20.agents.dqn_agent import DQNAgent, DQNConfig
from uav_search_belief20.envs.multi_drone_env import (
    MultiDroneEnvConfig,
    MultiDronePrimitiveSearchEnv,
)
from uav_search_belief20.utils import pick_device, seed_everything


class Tee:
    """Duplicate stdout into the terminal and a log file."""

    def __init__(self, *files):
        self.files = files

    def write(self, data: str) -> None:
        for f in self.files:
            f.write(data)

    def flush(self) -> None:
        for f in self.files:
            f.flush()


def make_env(args, seed: int, seed_offset: int = 0) -> MultiDronePrimitiveSearchEnv:
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
            seed=int(seed) + int(seed_offset),
            reward_version=args.reward_version,
        )
    )


def make_agent(args, env: MultiDronePrimitiveSearchEnv, algo: str, seed: int, device: str):
    if algo == "ddqn":
        return DQNAgent(
            DQNConfig(
                obs_shape=env.observation_shape,
                action_dim=env.action_dim,
                double_dqn=True,
                device=device,
                seed=int(seed),
                lr=args.lr,
                batch_size=args.batch_size,
                replay_capacity=args.replay_capacity,
                target_update_period=args.target_update_period,
                epsilon_start=args.epsilon_start,
                epsilon_end=args.epsilon_end,
                epsilon_decay_steps=args.epsilon_decay_steps,
                posterior_replay_size=args.posterior_replay_size,
                posterior_chunk_size=args.posterior_chunk_size,
                posterior_min_samples=args.posterior_min_samples,
                posterior_mode=args.posterior_mode,
                freeze_feature_after_steps=args.freeze_feature_after_steps,
                posterior_jitter=args.posterior_jitter,
            )
        )

    if algo == "bdqn":
        return BDQNAgent(
            BDQNConfig(
                obs_shape=env.observation_shape,
                action_dim=env.action_dim,
                device=device,
                seed=int(seed),
                lr=args.lr,
                batch_size=args.batch_size,
                replay_capacity=args.replay_capacity,
                target_update_period=args.target_update_period,
                posterior_update_period=args.posterior_update_period,
                blr_lambda=args.blr_lambda,
                blr_noise_var=args.blr_noise_var,
            )
        )

    raise ValueError(f"Unknown algo: {algo}")


def act_one(agent, obs: np.ndarray, mask: np.ndarray, train: bool) -> int:
    if isinstance(agent, BDQNAgent):
        return agent.act(obs, use_sample=train, action_mask=mask)
    return agent.act(obs, explore=train, action_mask=mask)


def evaluate(agent, args, algo: str, seed: int, episodes: int = 1000) -> dict:
    rewards, detected, completed, coverage = [], [], [], []
    detected_value, completed_value = [], []
    action_counts = Counter()

    boundary_hits = 0
    collision_count = 0
    position_revisit_steps = 0
    sensor_revisit_steps = 0
    tracking_progress_steps = 0
    new_observed_total = 0
    decisions = 0
    env_steps = 0

    for ep in range(episodes):
        env = make_env(args, seed=seed, seed_offset=10_000 + ep)
        obs_all, info = env.reset()
        done = False
        total = 0.0

        while not done:
            masks = env.action_mask()
            actions = np.array(
                [
                    act_one(agent, obs_all[i], masks[i], train=False)
                    for i in range(args.n_agents)
                ],
                dtype=np.int64,
            )

            next_obs_all, reward, terminated, truncated, info = env.step(actions)
            done = terminated or truncated
            total += reward
            obs_all = next_obs_all
            env_steps += 1

            for i, a in enumerate(actions):
                action_counts[ACTION_NAMES[int(a)]] += 1
                boundary_hits += int(info["last_boundary_hits"][i])
                position_revisit_steps += int(not info["last_new_cells"][i])
                sensor_revisit_steps += int(info["last_new_observed_cells"][i] == 0)
                new_observed_total += int(info["last_new_observed_cells"][i])
                tracking_progress_steps += int(info["last_tracking_progress"][i])
                decisions += 1

            collision_count += int(info.get("last_collision_count", 0))

        rewards.append(total)
        detected.append(info["detected"])
        completed.append(info["completed"])
        detected_value.append(info["detected_value"])
        completed_value.append(info["completed_value"])
        coverage.append(info["visited_ratio"])

    return {
        "reward_version": args.reward_version,
        "setting": f"shared_{algo}_{args.n_agents}uav",
        "algo": algo,
        "n_agents": int(args.n_agents),
        "episodes": int(episodes),
        "reward_mean": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
        "detected_mean": float(np.mean(detected)),
        "completed_mean": float(np.mean(completed)),
        "detected_value_mean": float(np.mean(detected_value)),
        "completed_value_mean": float(np.mean(completed_value)),
        "sensor_coverage_ratio_mean": float(np.mean(coverage)),
        "stay_ratio": action_counts["stay"] / max(1, decisions),
        "boundary_hit_ratio": boundary_hits / max(1, decisions),
        "collision_ratio": collision_count / max(1, env_steps),
        # Backward-compatible position metric.
        "revisit_ratio": position_revisit_steps / max(1, decisions),
        # V3 sensor-exploration metrics.
        "sensor_revisit_ratio": sensor_revisit_steps / max(1, decisions),
        "new_observed_cells_per_step": new_observed_total / max(1, decisions),
        "tracking_progress_ratio": tracking_progress_steps / max(1, decisions),
        "action_counts": dict(action_counts),
    }


def checkpoint_score(metrics: dict) -> float:
    """Checkpoint selection score.

    The thesis comparison should use the final evaluate metrics, not this score alone.
    This score intentionally favors actually completing targets.
    """
    return (
        metrics["reward_mean"]
        + 3.0 * metrics["completed_mean"]
        + 1.5 * metrics["completed_value_mean"]
        + 0.5 * metrics["detected_mean"]
        + 0.5 * metrics["sensor_coverage_ratio_mean"]
    )


def train_one_job(args, algo: str, seed: int, run_dir: Path) -> None:
    seed_everything(int(seed))
    env = make_env(args, seed=seed)
    agent = make_agent(args, env, algo=algo, seed=seed, device=args.resolved_device)

    print(f"Using device: {args.resolved_device}")
    print(f"algo: {algo}")
    print(f"n_agents: {args.n_agents}")
    print(f"observation_shape: {env.observation_shape}")
    print(f"train_every: {args.train_every}, learning_starts: {args.learning_starts}")
    print(f"reward_version: {env.cfg.reward_version}")
    print("reward_config:")
    for k, v in env.cfg.reward_dict().items():
        if (
            "bonus" in k
            or "penalty" in k
            or "reward_version" in k
            or k in {"track_required", "n_agents"}
        ):
            print(f"  {k}: {v}")

    run_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        "algo": algo,
        "seed": int(seed),
        "args": vars(args),
        "device": args.resolved_device,
        "env_config": env.cfg.reward_dict(),
        "reward_version": env.cfg.reward_version,
    }
    with (run_dir / "run_config.json").open("w") as f:
        json.dump(run_config, f, indent=2)

    metrics_path = run_dir / "metrics.csv"
    fieldnames = [
        "episode",
        "reward_version",
        "algo",
        "n_agents",
        "train_reward",
        "train_detected",
        "train_completed",
        "train_detected_value",
        "train_completed_value",
        "eval_reward",
        "eval_detected",
        "eval_completed",
        "eval_detected_value",
        "eval_completed_value",
        "eval_sensor_coverage",
        "stay_ratio",
        "boundary_hit_ratio",
        "collision_ratio",
        "revisit_ratio",
        "sensor_revisit_ratio",
        "new_observed_cells_per_step",
        "tracking_progress_ratio",
        "best_score",
    ]
    with metrics_path.open("w", newline="") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    recent_reward = deque(maxlen=50)
    recent_completed = deque(maxlen=50)
    recent_detected = deque(maxlen=50)

    best_score = -1e18
    best_ep = 0
    global_env_steps = 0

    prof = {
        "act": 0.0,
        "env": 0.0,
        "replay": 0.0,
        "train": 0.0,
        "eval": 0.0,
        "save": 0.0,
        "env_steps": 0,
    }

    pbar = trange(
        args.episodes,
        desc=f"Shared {algo.upper()} {args.n_agents}UAV {args.reward_version} seed={seed}",
    )

    for ep in pbar:
        obs_all, info = env.reset()
        if isinstance(agent, BDQNAgent):
            agent.resample_policy()

        done = False
        ep_reward = 0.0

        while not done:
            masks = env.action_mask()

            if args.profile:
                t0 = time.perf_counter()
            actions = np.array(
                [
                    act_one(agent, obs_all[i], masks[i], train=True)
                    for i in range(args.n_agents)
                ],
                dtype=np.int64,
            )
            if args.profile:
                prof["act"] += time.perf_counter() - t0

            if args.profile:
                t0 = time.perf_counter()
            next_obs_all, reward, terminated, truncated, info = env.step(actions)
            if args.profile:
                prof["env"] += time.perf_counter() - t0

            done = terminated or truncated

            # Independent/shared Q-learning baseline:
            # every UAV transition receives the same team reward.
            if args.profile:
                t0 = time.perf_counter()
            stored_reward = float(reward) * float(args.reward_scale)
            for i in range(args.n_agents):
                agent.replay.add(
                    obs_all[i],
                    int(actions[i]),
                    stored_reward,
                    next_obs_all[i],
                    done,
                )
            if args.profile:
                prof["replay"] += time.perf_counter() - t0

            if args.profile:
                t0 = time.perf_counter()
            if (
                global_env_steps >= args.learning_starts
                and global_env_steps % args.train_every == 0
            ):
                agent.train_step()
            if args.profile:
                prof["train"] += time.perf_counter() - t0

            obs_all = next_obs_all
            ep_reward += reward
            global_env_steps += 1
            prof["env_steps"] += 1

        recent_reward.append(ep_reward)
        recent_completed.append(info["completed"])
        recent_detected.append(info["detected"])

        postfix = {
            "reward": f"{np.mean(recent_reward):.2f}",
            "det": f"{np.mean(recent_detected):.2f}",
            "comp": f"{np.mean(recent_completed):.2f}",
            "best": f"{best_score:.2f}",
        }
        if isinstance(agent, DQNAgent):
            postfix["eps"] = f"{agent.epsilon():.2f}"
        pbar.set_postfix(postfix)

        if (ep + 1) % args.eval_every == 0 or (ep + 1) == args.episodes:
            if args.profile:
                t0 = time.perf_counter()
            metrics = evaluate(agent, args, algo=algo, seed=seed, episodes=args.eval_episodes)
            if args.profile:
                prof["eval"] += time.perf_counter() - t0

            score = checkpoint_score(metrics)

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
                writer.writerow(
                    {
                        "episode": ep + 1,
                        "reward_version": args.reward_version,
                        "algo": algo,
                        "n_agents": args.n_agents,
                        "train_reward": ep_reward,
                        "train_detected": info["detected"],
                        "train_completed": info["completed"],
                        "train_detected_value": info["detected_value"],
                        "train_completed_value": info["completed_value"],
                        "eval_reward": metrics["reward_mean"],
                        "eval_detected": metrics["detected_mean"],
                        "eval_completed": metrics["completed_mean"],
                        "eval_detected_value": metrics["detected_value_mean"],
                        "eval_completed_value": metrics["completed_value_mean"],
                        "eval_sensor_coverage": metrics["sensor_coverage_ratio_mean"],
                        "stay_ratio": metrics["stay_ratio"],
                        "boundary_hit_ratio": metrics["boundary_hit_ratio"],
                        "collision_ratio": metrics["collision_ratio"],
                        "revisit_ratio": metrics["revisit_ratio"],
                        "sensor_revisit_ratio": metrics["sensor_revisit_ratio"],
                        "new_observed_cells_per_step": metrics["new_observed_cells_per_step"],
                        "tracking_progress_ratio": metrics["tracking_progress_ratio"],
                        "best_score": best_score,
                    }
                )

            print(f"\n[Eval {ep + 1}] {metrics}")

    if best_ep == 0:
        agent.save(str(run_dir / "best.pt"))
        best_ep = args.episodes
        best_score = float("nan")

    if args.profile:
        total = sum(prof[k] for k in ["act", "env", "replay", "train", "eval", "save"])
        print("\n=== PROFILE ===")
        print(f"env_steps: {prof['env_steps']}")
        print(f"agent_decisions: {prof['env_steps'] * args.n_agents}")
        print(f"total_profiled_time: {total:.2f}s")
        for key in ["act", "env", "replay", "train", "eval", "save"]:
            pct = 100.0 * prof[key] / max(total, 1e-9)
            ms_per_env_step = 1000.0 * prof[key] / max(prof["env_steps"], 1)
            print(f"{key:>8}: {prof[key]:8.2f}s | {pct:6.2f}% | {ms_per_env_step:8.3f} ms/env-step")
        if prof["env_steps"] > 0 and total > 0:
            print(f"throughput: {prof['env_steps'] / total:.2f} env steps/s")

    print("Training complete.")
    print(f"Best checkpoint: {run_dir / 'best.pt'} at episode {best_ep}, score={best_score:.3f}")


def evaluate_checkpoint(args, algo: str, seed: int, run_dir: Path, eval_json_path: Path) -> dict:
    env = make_env(args, seed=seed)
    agent = make_agent(args, env, algo=algo, seed=seed, device=args.resolved_device)
    checkpoint_path = run_dir / "best.pt"
    agent.load(str(checkpoint_path))

    print(f"Using device: {args.resolved_device}")
    print(f"algo: {algo}")
    print(f"n_agents: {args.n_agents}")
    print(f"checkpoint: {checkpoint_path}")
    print(f"reward_version: {args.reward_version}")

    metrics = evaluate(
        agent,
        args,
        algo=algo,
        seed=seed,
        episodes=args.final_eval_episodes,
    )

    eval_json_path.parent.mkdir(parents=True, exist_ok=True)
    with eval_json_path.open("w") as f:
        json.dump(metrics, f, indent=2)

    for k, v in metrics.items():
        print(f"{k}: {v}")

    return metrics


def parse_algos(algo_arg: str) -> list[str]:
    if algo_arg == "all":
        return ["ddqn", "bdqn"]
    return [algo_arg]


def parse_seeds(seed_arg: str) -> list[int]:
    if seed_arg == "all":
        return [42, 43, 44]
    return [int(s.strip()) for s in seed_arg.split(",") if s.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()

    # Main experimental controls.
    parser.add_argument("--algo", choices=["ddqn", "bdqn", "all"], default="ddqn")
    parser.add_argument("--seed", type=str, default="42", help="'42', '43', '44', '42,43,44', or 'all'")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--final-eval-episodes", type=int, default=1000)
    parser.add_argument("--reward-version", type=str, default="v3_frontier")
    parser.add_argument("--run-root", type=str, default="runs")
    parser.add_argument("--log-root", type=str, default="logs")

    # Environment.
    parser.add_argument("--n-agents", type=int, default=3)
    parser.add_argument("--grid-size", type=int, default=20)
    parser.add_argument("--n-value1-targets", type=int, default=3)
    parser.add_argument("--n-value2-targets", type=int, default=1)
    parser.add_argument("--sensor-radius", type=int, default=2)
    parser.add_argument("--detection-probability", type=float, default=1.0)
    parser.add_argument("--track-radius", type=int, default=1)
    parser.add_argument("--track-required", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=150)

    # Training/evaluation schedule.
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--train-every", type=int, default=2)
    parser.add_argument("--learning-starts", type=int, default=1000)

    # Optimization.
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--replay-capacity", type=int, default=100_000)
    parser.add_argument("--target-update-period", type=int, default=500)
    parser.add_argument("--reward-scale", type=float, default=1.0)

    # DDQN.
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay-steps", type=int, default=20_000)

    # BDQN.
    parser.add_argument("--posterior-update-period", type=int, default=500)
    parser.add_argument("--blr-lambda", type=float, default=1.0)
    parser.add_argument("--blr-noise-var", type=float, default=1.0)

    # Runtime.
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--skip-final-eval", action="store_true")

    
    parser.add_argument(
    "--posterior-replay-size",
    type=int,
    default=8192,
    )
    parser.add_argument(
        "--posterior-chunk-size",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--posterior-min-samples",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--posterior-mode",
        choices=["rebuild", "cumulative"],
        default="rebuild",
    )
    parser.add_argument(
        "--freeze-feature-after-steps",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--posterior-jitter",
        type=float,
        default=1e-6,
    )
    args = parser.parse_args()

    torch.set_num_threads(max(1, args.torch_threads))
    if args.device == "auto":
        resolved_device = pick_device()
    else:
        resolved_device = args.device

    if resolved_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA demandé avec --device cuda, mais torch.cuda.is_available() == False")
    if resolved_device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS demandé avec --device mps, mais torch.backends.mps.is_available() == False")

    # Store resolved device inside a mutable namespace so helper functions can use it.
    args = SimpleNamespace(**vars(args), resolved_device=resolved_device)

    algos = parse_algos(args.algo)
    seeds = parse_seeds(args.seed)

    log_dir = Path(args.log_root) / args.reward_version
    run_root = Path(args.run_root) / args.reward_version
    log_dir.mkdir(parents=True, exist_ok=True)
    run_root.mkdir(parents=True, exist_ok=True)

    summary_path = log_dir / f"shared_{args.n_agents}uav_summary.csv"
    summary_exists = summary_path.exists()
    summary_fieldnames = [
        "reward_version",
        "setting",
        "algo",
        "seed",
        "n_agents",
        "episodes",
        "reward_mean",
        "reward_std",
        "detected_mean",
        "completed_mean",
        "detected_value_mean",
        "completed_value_mean",
        "sensor_coverage_ratio_mean",
        "stay_ratio",
        "boundary_hit_ratio",
        "collision_ratio",
        "revisit_ratio",
        "sensor_revisit_ratio",
        "new_observed_cells_per_step",
        "tracking_progress_ratio",
    ]

    with summary_path.open("a", newline="") as sf:
        summary_writer = csv.DictWriter(sf, fieldnames=summary_fieldnames)
        if not summary_exists:
            summary_writer.writeheader()

        for algo in algos:
            for seed in seeds:
                tag = f"shared_{algo}_{args.n_agents}uav_seed{seed}_{args.episodes}"
                run_dir = run_root / tag
                train_log = log_dir / f"{tag}_train.log"
                eval_log = log_dir / f"{tag}_eval{args.final_eval_episodes}.log"
                eval_json = log_dir / f"{tag}_eval{args.final_eval_episodes}.json"

                print(f"\n=== TRAIN {args.reward_version} {tag} ===")
                with train_log.open("w") as lf:
                    with contextlib.redirect_stdout(Tee(sys.__stdout__, lf)):
                        train_one_job(args, algo=algo, seed=seed, run_dir=run_dir)

                final_metrics = None
                if not args.skip_final_eval:
                    print(f"\n=== EVAL {args.reward_version} {tag} ===")
                    with eval_log.open("w") as lf:
                        with contextlib.redirect_stdout(Tee(sys.__stdout__, lf)):
                            final_metrics = evaluate_checkpoint(
                                args,
                                algo=algo,
                                seed=seed,
                                run_dir=run_dir,
                                eval_json_path=eval_json,
                            )

                    row = {k: final_metrics.get(k) for k in summary_fieldnames}
                    row["seed"] = seed
                    summary_writer.writerow(row)
                    sf.flush()

                # Release GPU memory before the next algo/seed.
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                print(f"Done: {args.reward_version} {tag}")

    print("\nAll requested jobs complete.")
    print(f"Summary CSV: {summary_path}")


if __name__ == "__main__":
    main()
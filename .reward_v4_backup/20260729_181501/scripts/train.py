from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter, deque
from pathlib import Path

import numpy as np
import torch
from tqdm import trange

from uav_search_belief20.agents.bdqn_agent import BDQNAgent, BDQNConfig
from uav_search_belief20.agents.dqn_agent import DQNAgent, DQNConfig
from uav_search_belief20.envs.primitive_search_env import REWARD_PART_KEYS, PrimitiveSearchEnv
from uav_search_belief20.evaluation import evaluate_policy, training_metric_view
from uav_search_belief20.experiments.single_ablation import (
    ABLATIONS,
    build_env_config,
    describe_ablation,
)
from uav_search_belief20.utils import pick_device, seed_everything


UNCERTAINTY_KEYS = (
    "posterior_rebuilds",
    "predictive_epistemic_std_mean",
    "predictive_selected_std_mean",
    "predictive_max_std_mean",
)


def make_env(args, *, seed: int) -> PrimitiveSearchEnv:
    return PrimitiveSearchEnv(build_env_config(args, seed=seed))


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
                posterior_replay_size=args.posterior_replay_size,
                posterior_chunk_size=args.posterior_chunk_size,
                posterior_min_samples=args.posterior_min_samples,
                posterior_mode=args.posterior_mode,
                freeze_feature_after_steps=args.freeze_feature_after_steps,
                posterior_jitter=args.posterior_jitter,
            )
        )
    raise ValueError(f"Unknown algo: {args.algo}")


def evaluate_agent(agent, args, episodes: int) -> dict[str, object]:
    def env_factory(seed: int) -> PrimitiveSearchEnv:
        return make_env(args, seed=seed)

    def policy(env: PrimitiveSearchEnv, obs: np.ndarray, episode_index: int) -> int:
        del episode_index
        if isinstance(agent, BDQNAgent):
            return agent.act(obs, use_sample=False, action_mask=env.action_mask())
        return agent.act(obs, explore=False, action_mask=env.action_mask())

    return evaluate_policy(
        env_factory,
        policy,
        episodes=episodes,
        seed_base=args.periodic_eval_seed_base,
    )


def checkpoint_score(metrics: dict[str, object]) -> float:
    return float(
        metrics["reward_mean"]
        + 3.0 * metrics["completed_mean"]
        + 1.5 * metrics["completed_value_mean"]
        + 0.5 * metrics["detected_mean"]
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["dqn", "ddqn", "bdqn"], default="bdqn")
    parser.add_argument("--ablation", choices=list(ABLATIONS), default="v3")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--grid-size", type=int, default=20)
    parser.add_argument("--n-value1-targets", type=int, default=3)
    parser.add_argument("--n-value2-targets", type=int, default=1)
    parser.add_argument("--sensor-radius", type=int, default=2)
    parser.add_argument("--detection-probability", type=float, default=1.0)
    parser.add_argument("--track-radius", type=int, default=1)
    parser.add_argument("--track-required", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--reward-version", type=str, default="v3_frontier")
    parser.add_argument(
        "--track-progress-scale",
        type=float,
        default=None,
        help="Override the ablation preset's tracking-progress reward scale.",
    )

    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--eval-episodes", type=int, default=30)
    parser.add_argument("--periodic-eval-seed-base", type=int, default=100_000)
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
    parser.add_argument("--train-every", type=int, default=4)
    parser.add_argument("--learning-starts", type=int, default=1000)

    parser.add_argument("--posterior-update-period", type=int, default=500)
    parser.add_argument("--posterior-replay-size", type=int, default=8192)
    parser.add_argument("--posterior-chunk-size", type=int, default=512)
    parser.add_argument("--posterior-min-samples", type=int, default=1000)
    parser.add_argument("--posterior-mode", choices=["rebuild", "cumulative"], default="rebuild")
    parser.add_argument("--freeze-feature-after-steps", type=int, default=None)
    parser.add_argument("--blr-lambda", type=float, default=1.0)
    parser.add_argument("--blr-noise-var", type=float, default=1.0)
    parser.add_argument("--posterior-jitter", type=float, default=1e-6)

    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--profile", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    torch.set_num_threads(max(1, args.torch_threads))
    seed_everything(args.seed)

    device = pick_device() if args.device == "auto" else args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but torch.backends.mps.is_available() is False.")

    env = make_env(args, seed=args.seed)
    agent = make_agent(args, env, device)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {device}")
    print(f"algo: {args.algo}")
    print(f"ablation: {args.ablation} — {describe_ablation(args.ablation)}")
    print(f"observation_shape: {env.observation_shape}")
    print(f"train_every: {args.train_every}, learning_starts: {args.learning_starts}")
    print("environment_config:")
    for key, value in env.cfg.reward_dict().items():
        print(f"  {key}: {value}")

    run_config = {
        "algo": args.algo,
        "ablation": args.ablation,
        "args": vars(args),
        "device": device,
        "env_config": env.cfg.reward_dict(),
        "agent_config": agent.cfg.__dict__,
    }
    with (run_dir / "run_config.json").open("w") as file:
        json.dump(run_config, file, indent=2)

    fieldnames = [
        "episode",
        "algo",
        "ablation",
        "reward_version",
        "train_reward",
        "train_detected",
        "train_completed",
        "train_detected_value",
        "train_completed_value",
        "eval_reward",
        "eval_reward_std",
        "eval_detected",
        "eval_completed",
        "eval_detected_value",
        "eval_completed_value",
        "eval_sensor_coverage",
        "eval_episode_length",
        "eval_detected_to_completed_ratio",
        "eval_first_detection_step",
        "eval_first_completion_step",
        "stay_ratio",
        "boundary_hit_ratio",
        "revisit_ratio",
        "sensor_revisit_ratio",
        "new_observed_cells_per_step",
        "tracking_progress_ratio",
        "loss",
        "q_mean",
        "epsilon",
        *UNCERTAINTY_KEYS,
        *[f"train_reward_part_{key}" for key in REWARD_PART_KEYS],
        *[f"eval_reward_part_{key}" for key in REWARD_PART_KEYS],
        "best_score",
    ]
    metrics_path = run_dir / "metrics.csv"
    with metrics_path.open("w", newline="") as file:
        csv.DictWriter(file, fieldnames=fieldnames).writeheader()

    recent_reward = deque(maxlen=50)
    recent_detected = deque(maxlen=50)
    recent_completed = deque(maxlen=50)
    best_score = -float("inf")
    best_episode = 0
    global_steps = 0
    last_train_stats: dict[str, float] = {"loss": 0.0, "q_mean": 0.0}
    profile = Counter()

    progress = trange(args.episodes, desc=f"{args.algo.upper()} {args.ablation} seed={args.seed}")
    for episode in progress:
        obs, info = env.reset()
        if isinstance(agent, BDQNAgent):
            agent.resample_policy()
        done = False
        episode_reward = 0.0
        episode_reward_parts: Counter[str] = Counter()

        while not done:
            start = time.perf_counter()
            if isinstance(agent, BDQNAgent):
                action = agent.act(obs, use_sample=True, action_mask=env.action_mask())
            else:
                action = agent.act(obs, explore=True, action_mask=env.action_mask())
            profile["act"] += time.perf_counter() - start

            start = time.perf_counter()
            next_obs, reward, terminated, truncated, info = env.step(action)
            profile["env"] += time.perf_counter() - start
            done = bool(terminated or truncated)

            for key, value in info.get("last_reward_parts", {}).items():
                episode_reward_parts[str(key)] += float(value)

            start = time.perf_counter()
            agent.replay.add(obs, action, reward, next_obs, done)
            profile["replay"] += time.perf_counter() - start

            if global_steps >= args.learning_starts and global_steps % args.train_every == 0:
                start = time.perf_counter()
                last_train_stats = agent.train_step()
                profile["train"] += time.perf_counter() - start

            obs = next_obs
            episode_reward += float(reward)
            global_steps += 1
            profile["steps"] += 1

        recent_reward.append(episode_reward)
        recent_detected.append(info["detected"])
        recent_completed.append(info["completed"])
        progress.set_postfix(
            reward=f"{np.mean(recent_reward):.2f}",
            detected=f"{np.mean(recent_detected):.2f}",
            completed=f"{np.mean(recent_completed):.2f}",
        )

        if (episode + 1) % args.eval_every != 0 and (episode + 1) != args.episodes:
            continue

        start = time.perf_counter()
        canonical_metrics = evaluate_agent(agent, args, episodes=args.eval_episodes)
        profile["eval"] += time.perf_counter() - start
        eval_metrics = training_metric_view(canonical_metrics)
        score = checkpoint_score(canonical_metrics)

        if score > best_score:
            best_score = score
            best_episode = episode + 1
            agent.save(str(run_dir / "best.pt"))
            print(f"\n[Best] episode={best_episode} score={best_score:.3f}")

        if (episode + 1) % args.save_every == 0 or (episode + 1) == args.episodes:
            agent.save(str(run_dir / "latest.pt"))

        row: dict[str, object] = {
            "episode": episode + 1,
            "algo": args.algo,
            "ablation": args.ablation,
            "reward_version": env.cfg.reward_version,
            "train_reward": episode_reward,
            "train_detected": info["detected"],
            "train_completed": info["completed"],
            "train_detected_value": info["detected_value"],
            "train_completed_value": info["completed_value"],
            **eval_metrics,
            "loss": last_train_stats.get("loss", float("nan")),
            "q_mean": last_train_stats.get("q_mean", float("nan")),
            "epsilon": last_train_stats.get("epsilon", getattr(agent, "epsilon", lambda: float("nan"))()),
            "best_score": best_score,
        }
        for key in UNCERTAINTY_KEYS:
            row[key] = last_train_stats.get(key, float("nan"))
        for key in REWARD_PART_KEYS:
            row[f"train_reward_part_{key}"] = float(episode_reward_parts.get(key, 0.0))

        with metrics_path.open("a", newline="") as file:
            csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore").writerow(row)
        print(f"\n[Eval {episode + 1}] {canonical_metrics}")

    if best_episode == 0:
        agent.save(str(run_dir / "best.pt"))
        best_episode = args.episodes

    if args.profile:
        total_time = sum(profile[key] for key in ("act", "env", "replay", "train", "eval"))
        print("\n=== PROFILE ===")
        print(f"steps: {int(profile['steps'])}")
        for key in ("act", "env", "replay", "train", "eval"):
            percentage = 100.0 * profile[key] / max(total_time, 1e-9)
            milliseconds = 1000.0 * profile[key] / max(profile["steps"], 1)
            print(f"{key:>8}: {profile[key]:8.2f}s | {percentage:6.2f}% | {milliseconds:8.3f} ms/step")

    print("Training complete.")
    print(f"Best checkpoint: {run_dir / 'best.pt'} at episode {best_episode}, score={best_score:.3f}")


if __name__ == "__main__":
    main()

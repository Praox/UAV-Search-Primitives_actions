from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from uav_search_belief20.agents.bdqn_agent import BDQNAgent, BDQNConfig
from uav_search_belief20.agents.dqn_agent import DQNAgent, DQNConfig
from uav_search_belief20.envs.multi_drone_local_env import (
    MultiDroneLocalEnvConfig,
    MultiDroneLocalMemoryEnv,
)
from uav_search_belief20.evaluation_multi_local import evaluate_multi_local_policy
from uav_search_belief20.marl.qmix_local_agent import LocalQMIXAgent, LocalQMIXConfig
from uav_search_belief20.utils import pick_device, seed_everything

ALIASES = {
    "shared_ddqn": "shared_ddqn",
    "ddqn": "shared_ddqn",
    "shared_bdqn": "shared_bdqn",
    "bdqn": "shared_bdqn",
    "qmix_ddqn": "qmix_ddqn",
    "qmix": "qmix_ddqn",
}


def normalize_algo(name: str) -> str:
    normalized = str(name).strip().lower().replace("-", "_")
    if normalized not in ALIASES:
        raise ValueError(f"Unknown algorithm {name!r}; expected one of {sorted(ALIASES)}")
    return ALIASES[normalized]


def make_env(args, seed: int) -> MultiDroneLocalMemoryEnv:
    return MultiDroneLocalMemoryEnv(
        MultiDroneLocalEnvConfig(
            grid_size=args.grid_size,
            n_agents=args.n_agents,
            n_value1_targets=args.n_value1_targets,
            n_value2_targets=args.n_value2_targets,
            sensor_radius=args.sensor_radius,
            teammate_visibility_radius=args.teammate_visibility_radius,
            detection_probability=args.detection_probability,
            track_radius=args.track_radius,
            track_required=args.track_required,
            max_steps=args.max_steps,
            seed=int(seed),
            reward_version=args.reward_version,
            include_agent_id_map=args.include_agent_id_map,
            collision_penalty=args.collision_penalty,
        )
    )


def make_learner(args, env: MultiDroneLocalMemoryEnv, algo: str, seed: int):
    if algo == "shared_ddqn":
        return DQNAgent(
            DQNConfig(
                obs_shape=env.observation_shape,
                action_dim=env.action_dim,
                feature_dim=args.feature_dim,
                gamma=args.gamma,
                lr=args.lr,
                batch_size=args.batch_size,
                replay_capacity=args.replay_capacity,
                target_update_period=args.target_update_period,
                double_dqn=True,
                epsilon_start=args.epsilon_start,
                epsilon_end=args.epsilon_end,
                epsilon_decay_steps=args.epsilon_decay_steps,
                grad_clip_norm=args.grad_clip_norm,
                device=args.resolved_device,
                seed=int(seed),
            )
        )
    if algo == "shared_bdqn":
        return BDQNAgent(
            BDQNConfig(
                obs_shape=env.observation_shape,
                action_dim=env.action_dim,
                feature_dim=args.feature_dim,
                gamma=args.gamma,
                lr=args.lr,
                batch_size=args.batch_size,
                replay_capacity=args.replay_capacity,
                target_update_period=args.target_update_period,
                posterior_update_period=args.posterior_update_period,
                posterior_replay_size=args.posterior_replay_size,
                posterior_chunk_size=args.posterior_chunk_size,
                posterior_min_samples=args.posterior_min_samples,
                blr_lambda=args.blr_lambda,
                blr_noise_var=args.blr_noise_var,
                posterior_jitter=args.posterior_jitter,
                posterior_mode=args.posterior_mode,
                freeze_feature_after_steps=args.freeze_feature_after_steps,
                grad_clip_norm=args.grad_clip_norm,
                device=args.resolved_device,
                seed=int(seed),
            )
        )
    if algo == "qmix_ddqn":
        return LocalQMIXAgent(
            LocalQMIXConfig(
                obs_shape=env.observation_shape,
                state_dim=env.state_dim,
                n_agents=env.cfg.n_agents,
                action_dim=env.action_dim,
                feature_dim=args.feature_dim,
                mixing_embed_dim=args.mixing_embed_dim,
                mixing_hypernet_embed=args.mixing_hypernet_embed,
                gamma=args.gamma,
                lr=args.lr,
                batch_size=args.batch_size,
                replay_capacity=args.replay_capacity,
                target_update_period=args.target_update_period,
                epsilon_start=args.epsilon_start,
                epsilon_end=args.epsilon_end,
                epsilon_decay_steps=args.epsilon_decay_steps,
                grad_clip_norm=args.grad_clip_norm,
                device=args.resolved_device,
                seed=int(seed),
            )
        )
    raise AssertionError(algo)


def decentralized_actions(learner, algo: str, obs_all, masks, train: bool):
    if algo == "qmix_ddqn":
        return learner.act(obs_all, action_masks=masks, explore=train)
    if algo == "shared_bdqn":
        return np.asarray(
            [
                learner.act(obs_all[i], use_sample=train, action_mask=masks[i])
                for i in range(obs_all.shape[0])
            ],
            dtype=np.int64,
        )
    return np.asarray(
        [
            learner.act(obs_all[i], explore=train, action_mask=masks[i])
            for i in range(obs_all.shape[0])
        ],
        dtype=np.int64,
    )


def evaluation_policy(learner, algo: str):
    return lambda obs_all, masks: decentralized_actions(
        learner, algo, obs_all, masks, train=False
    )


def checkpoint_score(metrics: dict) -> float:
    return float(
        metrics["reward_mean"]
        + 4.0 * metrics["completed_mean"]
        + 2.0 * metrics["completed_value_mean"]
        + 0.5 * metrics["detected_mean"]
        + metrics["team_coverage_ratio_mean"]
        - 2.0 * metrics["coverage_overlap_ratio_mean"]
        - 2.0 * metrics["collision_agent_ratio"]
    )


def evaluate(learner, args, algo: str, episodes: int) -> dict:
    metrics = evaluate_multi_local_policy(
        policy=evaluation_policy(learner, algo),
        env_factory=lambda world_seed: make_env(args, world_seed),
        episodes=episodes,
        eval_seed_base=args.eval_seed_base,
    )
    metrics.update(
        {
            "algo": algo,
            "seed": int(args.seed),
            "n_agents": int(args.n_agents),
            "reward_version": args.reward_version,
        }
    )
    return metrics


def train(args, algo: str, run_dir: Path) -> None:
    seed_everything(int(args.seed))
    env = make_env(args, int(args.seed))
    learner = make_learner(args, env, algo, int(args.seed))
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "run_config.json").open("w") as handle:
        json.dump(
            {
                "algo": algo,
                "args": vars(args),
                "env_config": env.cfg.reward_dict(),
                "observation_shape": env.observation_shape,
                "state_dim": env.state_dim,
            },
            handle,
            indent=2,
            default=str,
        )

    csv_path = run_dir / "metrics.csv"
    fields = [
        "episode", "train_reward", "train_detected", "train_completed",
        "eval_reward", "eval_detected", "eval_completed",
        "eval_completed_value", "eval_team_coverage", "eval_overlap",
        "eval_collision", "best_score",
    ]
    with csv_path.open("w", newline="") as handle:
        csv.DictWriter(handle, fieldnames=fields).writeheader()

    best_score = -float("inf")
    best_episode = 0
    global_env_steps = 0
    for episode in range(1, int(args.episodes) + 1):
        obs_all, info = env.reset()
        state = env.global_state()
        if algo == "shared_bdqn":
            # One coherent posterior sample shared by the whole team for one episode.
            learner.resample_policy()
        done = False
        episode_reward = 0.0
        while not done:
            masks = env.action_mask()
            actions = decentralized_actions(learner, algo, obs_all, masks, train=True)
            next_obs, reward, terminated, truncated, info = env.step(actions)
            done = bool(terminated or truncated)
            next_state = env.global_state()
            next_masks = env.action_mask()
            stored_reward = float(reward) * float(args.reward_scale)
            if algo == "qmix_ddqn":
                learner.replay.add(
                    obs_all=obs_all,
                    state=state,
                    actions=actions,
                    reward=stored_reward,
                    next_obs_all=next_obs,
                    next_state=next_state,
                    done=done,
                    action_masks=masks,
                    next_action_masks=next_masks,
                )
            else:
                for agent_id in range(args.n_agents):
                    learner.replay.add(
                        obs_all[agent_id],
                        int(actions[agent_id]),
                        stored_reward,
                        next_obs[agent_id],
                        done,
                    )
            if (
                global_env_steps >= args.learning_starts
                and global_env_steps % args.train_every == 0
            ):
                learner.train_step()
            obs_all, state = next_obs, next_state
            episode_reward += float(reward)
            global_env_steps += 1

        if episode % args.eval_every == 0 or episode == args.episodes:
            metrics = evaluate(learner, args, algo, args.eval_episodes)
            score = checkpoint_score(metrics)
            if score > best_score:
                best_score = score
                best_episode = episode
                learner.save(str(run_dir / "best.pt"))
            learner.save(str(run_dir / "latest.pt"))
            with csv_path.open("a", newline="") as handle:
                csv.DictWriter(handle, fieldnames=fields).writerow(
                    {
                        "episode": episode,
                        "train_reward": episode_reward,
                        "train_detected": info["detected"],
                        "train_completed": info["completed"],
                        "eval_reward": metrics["reward_mean"],
                        "eval_detected": metrics["detected_mean"],
                        "eval_completed": metrics["completed_mean"],
                        "eval_completed_value": metrics["completed_value_mean"],
                        "eval_team_coverage": metrics["team_coverage_ratio_mean"],
                        "eval_overlap": metrics["coverage_overlap_ratio_mean"],
                        "eval_collision": metrics["collision_agent_ratio"],
                        "best_score": best_score,
                    }
                )
            print(
                f"[episode {episode}] reward={episode_reward:.3f} "
                f"completed={metrics['completed_mean']:.3f} "
                f"coverage={metrics['team_coverage_ratio_mean']:.3f} "
                f"overlap={metrics['coverage_overlap_ratio_mean']:.3f}"
            )
    if best_episode == 0:
        learner.save(str(run_dir / "best.pt"))
    print(f"Best checkpoint: {run_dir / 'best.pt'} at episode {best_episode}")


def final_evaluation(args, algo: str, run_dir: Path, output_json: Path) -> dict:
    env = make_env(args, int(args.seed))
    learner = make_learner(args, env, algo, int(args.seed))
    learner.load(str(run_dir / "best.pt"))
    metrics = evaluate(learner, args, algo, args.final_eval_episodes)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w") as handle:
        json.dump(metrics, handle, indent=2, allow_nan=True)
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=list(ALIASES), default="shared_ddqn")
    parser.add_argument("--mode", choices=["train_eval", "eval_only"], default="train_eval")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--eval-episodes", type=int, default=30)
    parser.add_argument("--final-eval-episodes", type=int, default=1000)
    parser.add_argument("--eval-seed-base", type=int, default=100_000)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--eval-json", required=True)

    parser.add_argument("--reward-version", default="multi_local_v1_from_single_D")
    parser.add_argument("--n-agents", type=int, default=3)
    parser.add_argument("--grid-size", type=int, default=20)
    parser.add_argument("--n-value1-targets", type=int, default=3)
    parser.add_argument("--n-value2-targets", type=int, default=1)
    parser.add_argument("--sensor-radius", type=int, default=2)
    parser.add_argument("--teammate-visibility-radius", type=int, default=2)
    parser.add_argument("--detection-probability", type=float, default=1.0)
    parser.add_argument("--track-radius", type=int, default=1)
    parser.add_argument("--track-required", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--include-agent-id-map", action="store_true")
    parser.add_argument("--collision-penalty", type=float, default=-0.02)

    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--replay-capacity", type=int, default=100_000)
    parser.add_argument("--target-update-period", type=int, default=500)
    parser.add_argument("--grad-clip-norm", type=float, default=10.0)
    parser.add_argument("--feature-dim", type=int, default=128)
    parser.add_argument("--reward-scale", type=float, default=1.0)
    parser.add_argument("--train-every", type=int, default=1)
    parser.add_argument("--learning-starts", type=int, default=1000)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay-steps", type=int, default=20_000)
    parser.add_argument("--mixing-embed-dim", type=int, default=32)
    parser.add_argument("--mixing-hypernet-embed", type=int, default=64)

    parser.add_argument("--posterior-update-period", type=int, default=500)
    parser.add_argument("--posterior-replay-size", type=int, default=8192)
    parser.add_argument("--posterior-chunk-size", type=int, default=512)
    parser.add_argument("--posterior-min-samples", type=int, default=1000)
    parser.add_argument("--blr-lambda", type=float, default=1.0)
    parser.add_argument("--blr-noise-var", type=float, default=1.0)
    parser.add_argument("--posterior-jitter", type=float, default=1e-6)
    parser.add_argument("--posterior-mode", choices=["rebuild", "cumulative"], default="rebuild")
    parser.add_argument("--freeze-feature-after-steps", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--torch-threads", type=int, default=1)
    return parser


def main() -> None:
    parsed = build_parser().parse_args()
    torch.set_num_threads(max(1, parsed.torch_threads))
    device = pick_device() if parsed.device == "auto" else parsed.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    args = SimpleNamespace(**vars(parsed), resolved_device=device)
    algo = normalize_algo(args.algo)
    run_dir = Path(args.run_dir)
    output_json = Path(args.eval_json)
    if args.mode == "train_eval":
        train(args, algo, run_dir)
    elif not (run_dir / "best.pt").exists():
        raise FileNotFoundError(run_dir / "best.pt")
    final_evaluation(args, algo, run_dir, output_json)


if __name__ == "__main__":
    main()

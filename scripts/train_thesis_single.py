from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from uav_search_belief20.agents.bdqn_agent import BDQNAgent, BDQNConfig
from uav_search_belief20.agents.dqn_agent import DQNAgent, DQNConfig
from uav_search_belief20.envs.thesis_envs import (
    ThesisEnvConfig,
    ThesisPrimitiveSearchEnv,
)
from uav_search_belief20.evaluation import evaluate_policy
from uav_search_belief20.utils import pick_device, seed_everything


def make_env(args, seed: int) -> ThesisPrimitiveSearchEnv:
    return ThesisPrimitiveSearchEnv(
        ThesisEnvConfig(
            grid_size=args.grid_size,
            n_value1_targets=args.n_value1_targets,
            n_value2_targets=args.n_value2_targets,
            sensor_radius=args.sensor_radius,
            detection_probability=args.detection_probability,
            track_radius=args.track_radius,
            track_required=args.track_required,
            max_steps=args.max_steps,
            seed=int(seed),
            reward_version=f"thesis_{args.reward_mode}",
            ablation_name="thesis_v2",
            use_boundary_action_mask=True,
            include_track_progress_map=True,
            reward_mode=args.reward_mode,
            track_progress_decay=args.track_progress_decay,
            shaping_gamma=args.gamma,
            coverage_potential_scale=args.coverage_potential_scale,
            detection_potential_scale=args.detection_potential_scale,
            progress_potential_scale=args.progress_potential_scale,
        )
    )


def make_agent(args, env, device: str):
    common = dict(
        obs_shape=env.observation_shape,
        action_dim=env.action_dim,
        feature_dim=args.feature_dim,
        gamma=args.gamma,
        n_step=args.n_step,
        lr=args.lr,
        batch_size=args.batch_size,
        replay_capacity=args.replay_capacity,
        target_tau=args.target_tau,
        target_update_period=args.target_update_period,
        huber_delta=args.huber_delta,
        device=device,
        seed=args.seed,
    )
    if args.algo == "ddqn":
        return DQNAgent(
            DQNConfig(
                **common,
                double_dqn=True,
                epsilon_start=args.epsilon_start,
                epsilon_end=args.epsilon_end,
                epsilon_decay_steps=args.epsilon_decay_steps,
            )
        )
    agent = BDQNAgent(
        BDQNConfig(
            **common,
            posterior_update_period=args.posterior_update_period,
            posterior_replay_size=args.posterior_replay_size,
            posterior_min_samples=args.posterior_min_samples,
            blr_lambda=args.blr_lambda,
            blr_noise_var=args.blr_noise_var,
            posterior_mode="rebuild",
        )
    )
    if args.warmstart_ddqn:
        agent.load_feature_extractor_from_dqn_checkpoint(
            args.warmstart_ddqn,
            freeze=not args.adapt_bdqn_features,
        )
    return agent


def evaluate(agent, args, seed_base: int, episodes: int, *, sampled: bool = False):
    def env_factory(seed: int):
        return make_env(args, seed)

    def policy(env, obs, episode_index):
        del episode_index
        if isinstance(agent, BDQNAgent):
            return agent.act(
                obs,
                use_sample=sampled,
                action_mask=env.action_mask(),
            )
        return agent.act(obs, explore=False, action_mask=env.action_mask())

    return evaluate_policy(
        env_factory,
        policy,
        episodes=episodes,
        seed_base=seed_base,
        on_episode_start=(
            (lambda _: agent.resample_policy())
            if sampled and isinstance(agent, BDQNAgent)
            else None
        ),
    )


def score(metrics: dict) -> float:
    return float(
        10.0 * metrics["completed_mean"]
        + 3.0 * metrics["completed_value_mean"]
        - 0.02 * metrics["first_completion_step_mean"]
        + 0.1 * metrics["reward_mean"]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["ddqn", "bdqn"], required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])

    parser.add_argument("--grid-size", type=int, default=20)
    parser.add_argument("--n-value1-targets", type=int, default=3)
    parser.add_argument("--n-value2-targets", type=int, default=1)
    parser.add_argument("--sensor-radius", type=int, default=2)
    parser.add_argument("--detection-probability", type=float, default=1.0)
    parser.add_argument("--track-radius", type=int, default=1)
    parser.add_argument("--track-required", type=int, default=3)
    parser.add_argument("--track-progress-decay", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--reward-mode", choices=["legacy", "task_potential"], default="task_potential")
    parser.add_argument("--coverage-potential-scale", type=float, default=5.0)
    parser.add_argument("--detection-potential-scale", type=float, default=1.0)
    parser.add_argument("--progress-potential-scale", type=float, default=1.0)

    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--n-step", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--replay-capacity", type=int, default=50_000)
    parser.add_argument("--target-tau", type=float, default=0.005)
    parser.add_argument("--target-update-period", type=int, default=500)
    parser.add_argument("--huber-delta", type=float, default=1.0)
    parser.add_argument("--feature-dim", type=int, default=128)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay-steps", type=int, default=20_000)
    parser.add_argument("--learning-starts", type=int, default=1000)
    parser.add_argument("--train-every", type=int, default=4)

    parser.add_argument("--posterior-update-period", type=int, default=500)
    parser.add_argument("--posterior-replay-size", type=int, default=8192)
    parser.add_argument("--posterior-min-samples", type=int, default=1000)
    parser.add_argument("--blr-lambda", type=float, default=1.0)
    parser.add_argument("--blr-noise-var", type=float, default=1.0)
    parser.add_argument("--warmstart-ddqn", default="")
    parser.add_argument("--adapt-bdqn-features", action="store_true")

    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--validation-episodes", type=int, default=100)
    parser.add_argument("--final-test-episodes", type=int, default=1000)
    parser.add_argument("--validation-seed-base", type=int, default=100_000)
    parser.add_argument("--final-test-seed-base", type=int, default=200_000)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = pick_device() if args.device == "auto" else args.device
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    env = make_env(args, args.seed)
    agent = make_agent(args, env, device)

    with (run_dir / "run_config.json").open("w") as handle:
        json.dump(vars(args), handle, indent=2)
    with (run_dir / "metrics.csv").open("w", newline="") as handle:
        csv.DictWriter(
            handle,
            fieldnames=[
                "episode", "train_reward", "validation_reward",
                "validation_completed", "validation_completed_value",
                "best_score", "loss", "td_residual_variance",
            ],
        ).writeheader()

    global_step = 0
    best_score = -float("inf")
    for episode in range(1, args.episodes + 1):
        obs, info = env.reset()
        if isinstance(agent, BDQNAgent):
            agent.resample_policy()
        done = False
        episode_reward = 0.0
        last_stats: dict[str, float] = {}
        while not done:
            mask = env.action_mask()
            if isinstance(agent, BDQNAgent):
                action = agent.act(obs, use_sample=True, action_mask=mask)
            else:
                action = agent.act(obs, explore=True, action_mask=mask)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            next_mask = env.action_mask()
            agent.replay.add(
                obs,
                action,
                reward,
                next_obs,
                done,
                action_mask=mask,
                next_action_mask=next_mask,
            )
            if global_step >= args.learning_starts and global_step % args.train_every == 0:
                last_stats = agent.train_step()
            obs = next_obs
            episode_reward += float(reward)
            global_step += 1

        if episode % args.eval_every != 0 and episode != args.episodes:
            continue
        validation = evaluate(
            agent,
            args,
            args.validation_seed_base,
            args.validation_episodes,
            sampled=False,
        )
        current_score = score(validation)
        if current_score > best_score:
            best_score = current_score
            agent.save(str(run_dir / "best.pt"))
        agent.save(str(run_dir / "latest.pt"))
        with (run_dir / "metrics.csv").open("a", newline="") as handle:
            csv.DictWriter(
                handle,
                fieldnames=[
                    "episode", "train_reward", "validation_reward",
                    "validation_completed", "validation_completed_value",
                    "best_score", "loss", "td_residual_variance",
                ],
            ).writerow(
                {
                    "episode": episode,
                    "train_reward": episode_reward,
                    "validation_reward": validation["reward_mean"],
                    "validation_completed": validation["completed_mean"],
                    "validation_completed_value": validation["completed_value_mean"],
                    "best_score": best_score,
                    "loss": last_stats.get("loss", np.nan),
                    "td_residual_variance": last_stats.get("td_residual_variance", np.nan),
                }
            )
        print(
            f"episode={episode} reward={episode_reward:.3f} "
            f"val_completed={validation['completed_mean']:.3f}"
        )

    agent.load(str(run_dir / "best.pt"))
    final_mean = evaluate(
        agent,
        args,
        args.final_test_seed_base,
        args.final_test_episodes,
        sampled=False,
    )
    output = {"posterior_mean": final_mean}
    if isinstance(agent, BDQNAgent):
        output["posterior_sample"] = evaluate(
            agent,
            args,
            args.final_test_seed_base,
            args.final_test_episodes,
            sampled=True,
        )
    with (run_dir / "final_test.json").open("w") as handle:
        json.dump(output, handle, indent=2, allow_nan=True)


if __name__ == "__main__":
    main()

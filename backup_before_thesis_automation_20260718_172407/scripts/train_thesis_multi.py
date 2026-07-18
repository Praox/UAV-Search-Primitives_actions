from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from uav_search_belief20.agents.bdqn_agent import BDQNAgent, BDQNConfig
from uav_search_belief20.agents.dqn_agent import DQNAgent, DQNConfig
from uav_search_belief20.envs.thesis_envs import (
    ThesisMultiDroneLocalMemoryEnv,
    ThesisMultiEnvConfig,
)
from uav_search_belief20.evaluation_multi_local import evaluate_multi_local_policy
from uav_search_belief20.marl.thesis_qmix import (
    ThesisBayesianQMIXAgent,
    ThesisBayesianQMIXConfig,
    ThesisQMIXAgent,
    ThesisQMIXConfig,
)
from uav_search_belief20.utils import pick_device, seed_everything


JOINT_ALGOS = {"qmix_ddqn", "bayes_qmix_shared", "bayes_qmix_independent"}


def make_env(args, seed: int) -> ThesisMultiDroneLocalMemoryEnv:
    return ThesisMultiDroneLocalMemoryEnv(
        ThesisMultiEnvConfig(
            grid_size=args.grid_size,
            n_agents=args.n_agents,
            n_value1_targets=args.n_value1_targets,
            n_value2_targets=args.n_value2_targets,
            sensor_radius=args.sensor_radius,
            teammate_visibility_radius=args.teammate_visibility_radius,
            detection_probability=args.detection_probability,
            track_radius=args.track_radius,
            track_required=args.track_required,
            track_progress_decay=args.track_progress_decay,
            max_steps=args.max_steps,
            seed=int(seed),
            reward_version=f"thesis_multi_{args.reward_mode}",
            use_boundary_action_mask=True,
            include_agent_id_map=args.include_agent_id_map,
            global_state_mode=args.global_state_mode,
            reward_mode=args.reward_mode,
            shaping_gamma=args.gamma,
            coverage_potential_scale=args.coverage_potential_scale,
            detection_potential_scale=args.detection_potential_scale,
            progress_potential_scale=args.progress_potential_scale,
        )
    )


def make_agent(args, env, device: str):
    # A QMIX batch of B joint transitions contains B*N local observations.
    # The shared learner therefore uses B*N local transitions per update.
    shared_batch = args.batch_size * args.n_agents
    common_local = dict(
        obs_shape=env.observation_shape,
        action_dim=env.action_dim,
        feature_dim=args.feature_dim,
        gamma=args.gamma,
        n_step=args.n_step,
        lr=args.lr,
        batch_size=shared_batch,
        replay_capacity=args.replay_capacity,
        target_tau=args.target_tau,
        target_update_period=args.target_update_period,
        huber_delta=args.huber_delta,
        device=device,
        seed=args.seed,
    )
    if args.algo == "shared_ddqn":
        return DQNAgent(
            DQNConfig(
                **common_local,
                double_dqn=True,
                epsilon_start=args.epsilon_start,
                epsilon_end=args.epsilon_end,
                epsilon_decay_steps=args.epsilon_decay_steps,
            )
        )
    if args.algo == "shared_bdqn":
        agent = BDQNAgent(
            BDQNConfig(
                **common_local,
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

    common_joint = dict(
        obs_shape=env.observation_shape,
        state_dim=env.state_dim,
        n_agents=args.n_agents,
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
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay_steps=args.epsilon_decay_steps,
        device=device,
        seed=args.seed,
    )
    if args.algo == "qmix_ddqn":
        return ThesisQMIXAgent(ThesisQMIXConfig(**common_joint))

    sampling = "shared" if args.algo == "bayes_qmix_shared" else "independent"
    return ThesisBayesianQMIXAgent(
        ThesisBayesianQMIXConfig(
            **common_joint,
            posterior_sampling=sampling,
            prior_std=args.bayes_prior_std,
            initial_posterior_std=args.bayes_initial_std,
            kl_weight=args.bayes_kl_weight,
            epsilon_start=0.0,
            epsilon_end=0.0,
        )
    )


def actions_for(agent, algo: str, obs_all, masks, *, train: bool, sampled: bool = False):
    if algo == "qmix_ddqn":
        return agent.act(obs_all, action_masks=masks, explore=train)
    if algo.startswith("bayes_qmix"):
        return agent.act(
            obs_all,
            action_masks=masks,
            use_sample=(train or sampled),
            explore=train,
        )
    if algo == "shared_bdqn":
        return np.asarray(
            [
                agent.act(obs_all[index], use_sample=(train or sampled), action_mask=masks[index])
                for index in range(obs_all.shape[0])
            ],
            dtype=np.int64,
        )
    # Do not advance epsilon once per UAV. Advance it once after the joint action.
    actions = np.asarray(
        [
            agent.act(
                obs_all[index],
                explore=train,
                action_mask=masks[index],
                advance_env_step=False,
            )
            for index in range(obs_all.shape[0])
        ],
        dtype=np.int64,
    )
    if train:
        agent.advance_env_step()
    return actions


def evaluate(agent, args, seed_base: int, episodes: int, *, sampled: bool = False):
    def policy(obs_all, masks):
        return actions_for(
            agent,
            args.algo,
            obs_all,
            masks,
            train=False,
            sampled=sampled,
        )

    episode_start = None
    if sampled and (args.algo == "shared_bdqn" or args.algo.startswith("bayes_qmix")):
        episode_start = agent.resample_policy
    return evaluate_multi_local_policy(
        policy=policy,
        env_factory=lambda seed: make_env(args, seed),
        episodes=episodes,
        eval_seed_base=seed_base,
        episode_start_fn=episode_start,
    )


def score(metrics: dict) -> float:
    return float(
        10.0 * metrics["completed_mean"]
        + 3.0 * metrics["completed_value_mean"]
        + metrics["team_coverage_ratio_mean"]
        - 2.0 * metrics["coverage_overlap_ratio_mean"]
        - 3.0 * metrics["collision_agent_ratio"]
        + 0.1 * metrics["reward_mean"]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algo",
        choices=[
            "shared_ddqn",
            "shared_bdqn",
            "qmix_ddqn",
            "bayes_qmix_shared",
            "bayes_qmix_independent",
        ],
        required=True,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])

    parser.add_argument("--n-agents", type=int, default=3)
    parser.add_argument("--grid-size", type=int, default=20)
    parser.add_argument("--n-value1-targets", type=int, default=3)
    parser.add_argument("--n-value2-targets", type=int, default=1)
    parser.add_argument("--sensor-radius", type=int, default=2)
    parser.add_argument("--teammate-visibility-radius", type=int, default=2)
    parser.add_argument("--detection-probability", type=float, default=1.0)
    parser.add_argument("--track-radius", type=int, default=1)
    parser.add_argument("--track-required", type=int, default=3)
    parser.add_argument("--track-progress-decay", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--include-agent-id-map", action="store_true")
    parser.add_argument("--global-state-mode", choices=["privileged_truth", "memory_union"], default="memory_union")
    parser.add_argument("--reward-mode", choices=["legacy", "task_potential"], default="task_potential")
    parser.add_argument("--coverage-potential-scale", type=float, default=5.0)
    parser.add_argument("--detection-potential-scale", type=float, default=1.0)
    parser.add_argument("--progress-potential-scale", type=float, default=1.0)

    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--n-step", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--replay-capacity", type=int, default=100_000)
    parser.add_argument("--target-tau", type=float, default=0.005)
    parser.add_argument("--target-update-period", type=int, default=500)
    parser.add_argument("--huber-delta", type=float, default=1.0)
    parser.add_argument("--feature-dim", type=int, default=128)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay-steps", type=int, default=20_000)
    parser.add_argument("--learning-starts", type=int, default=1000)
    parser.add_argument("--train-every", type=int, default=1)

    parser.add_argument("--posterior-update-period", type=int, default=500)
    parser.add_argument("--posterior-replay-size", type=int, default=8192)
    parser.add_argument("--posterior-min-samples", type=int, default=1000)
    parser.add_argument("--blr-lambda", type=float, default=1.0)
    parser.add_argument("--blr-noise-var", type=float, default=1.0)
    parser.add_argument("--warmstart-ddqn", default="")
    parser.add_argument("--adapt-bdqn-features", action="store_true")
    parser.add_argument("--bayes-prior-std", type=float, default=1.0)
    parser.add_argument("--bayes-initial-std", type=float, default=0.05)
    parser.add_argument("--bayes-kl-weight", type=float, default=1e-3)

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
    fields = [
        "episode", "train_reward", "validation_reward", "validation_completed",
        "validation_completed_value", "validation_coverage", "validation_overlap",
        "validation_collision", "best_score", "loss",
    ]
    with (run_dir / "metrics.csv").open("w", newline="") as handle:
        csv.DictWriter(handle, fieldnames=fields).writeheader()

    global_step = 0
    best_score = -float("inf")
    for episode in range(1, args.episodes + 1):
        obs_all, info = env.reset()
        state = env.global_state()
        if args.algo == "shared_bdqn" or args.algo.startswith("bayes_qmix"):
            agent.resample_policy()
        done = False
        episode_reward = 0.0
        last_stats: dict[str, float] = {}
        while not done:
            masks = env.action_mask()
            actions = actions_for(agent, args.algo, obs_all, masks, train=True)
            next_obs, reward, terminated, truncated, info = env.step(actions)
            done = bool(terminated or truncated)
            next_state = env.global_state()
            next_masks = env.action_mask()
            if args.algo in JOINT_ALGOS:
                agent.replay.add(
                    obs_all=obs_all,
                    state=state,
                    actions=actions,
                    reward=reward,
                    next_obs_all=next_obs,
                    next_state=next_state,
                    done=done,
                    action_masks=masks,
                    next_action_masks=next_masks,
                )
            else:
                for agent_id in range(args.n_agents):
                    agent.replay.add(
                        obs_all[agent_id],
                        int(actions[agent_id]),
                        reward,
                        next_obs[agent_id],
                        done,
                        action_mask=masks[agent_id],
                        next_action_mask=next_masks[agent_id],
                        stream_id=agent_id,
                    )
            if global_step >= args.learning_starts and global_step % args.train_every == 0:
                last_stats = agent.train_step()
            obs_all, state = next_obs, next_state
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
            csv.DictWriter(handle, fieldnames=fields).writerow(
                {
                    "episode": episode,
                    "train_reward": episode_reward,
                    "validation_reward": validation["reward_mean"],
                    "validation_completed": validation["completed_mean"],
                    "validation_completed_value": validation["completed_value_mean"],
                    "validation_coverage": validation["team_coverage_ratio_mean"],
                    "validation_overlap": validation["coverage_overlap_ratio_mean"],
                    "validation_collision": validation["collision_agent_ratio"],
                    "best_score": best_score,
                    "loss": last_stats.get("loss", np.nan),
                }
            )
        print(
            f"episode={episode} reward={episode_reward:.3f} "
            f"val_completed={validation['completed_mean']:.3f} "
            f"coverage={validation['team_coverage_ratio_mean']:.3f}"
        )

    agent.load(str(run_dir / "best.pt"))
    final_mean = evaluate(
        agent,
        args,
        args.final_test_seed_base,
        args.final_test_episodes,
        sampled=False,
    )
    output = {"deterministic_or_posterior_mean": final_mean}
    if args.algo == "shared_bdqn" or args.algo.startswith("bayes_qmix"):
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

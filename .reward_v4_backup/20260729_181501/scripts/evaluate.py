from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from uav_search_belief20.agents.bdqn_agent import BDQNAgent, BDQNConfig
from uav_search_belief20.agents.dqn_agent import DQNAgent, DQNConfig
from uav_search_belief20.envs.primitive_search_env import PrimitiveSearchEnv
from uav_search_belief20.evaluation import evaluate_policy
from uav_search_belief20.experiments.single_ablation import ABLATIONS, build_env_config
from uav_search_belief20.utils import pick_device, seed_everything


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["dqn", "ddqn", "bdqn"], required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--ablation", choices=list(ABLATIONS), default="v3")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--eval-seed-base", type=int, default=200_000)
    parser.add_argument("--grid-size", type=int, default=20)
    parser.add_argument("--n-value1-targets", type=int, default=3)
    parser.add_argument("--n-value2-targets", type=int, default=1)
    parser.add_argument("--sensor-radius", type=int, default=2)
    parser.add_argument("--detection-probability", type=float, default=1.0)
    parser.add_argument("--track-radius", type=int, default=1)
    parser.add_argument("--track-required", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--reward-version", type=str, default="v3_frontier")
    parser.add_argument("--track-progress-scale", type=float, default=None)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--bdqn-sampled-eval", action="store_true")
    parser.add_argument("--json-out", type=str, default="")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    torch.set_num_threads(max(1, args.torch_threads))
    seed_everything(args.seed)
    device = pick_device() if args.device == "auto" else args.device

    def env_factory(seed: int) -> PrimitiveSearchEnv:
        return PrimitiveSearchEnv(build_env_config(args, seed=seed))

    env0 = env_factory(args.eval_seed_base)
    if args.algo in {"dqn", "ddqn"}:
        agent = DQNAgent(
            DQNConfig(
                obs_shape=env0.observation_shape,
                action_dim=env0.action_dim,
                double_dqn=args.algo == "ddqn",
                device=device,
                seed=args.seed,
            )
        )
    else:
        agent = BDQNAgent(
            BDQNConfig(
                obs_shape=env0.observation_shape,
                action_dim=env0.action_dim,
                device=device,
                seed=args.seed,
            )
        )
    agent.load(args.checkpoint)

    def policy(env: PrimitiveSearchEnv, obs: np.ndarray, episode_index: int) -> int:
        del episode_index
        if isinstance(agent, BDQNAgent):
            return agent.act(
                obs,
                use_sample=args.bdqn_sampled_eval,
                action_mask=env.action_mask(),
            )
        return agent.act(obs, explore=False, action_mask=env.action_mask())

    on_start = agent.resample_policy if args.bdqn_sampled_eval and isinstance(agent, BDQNAgent) else None
    metrics = evaluate_policy(
        env_factory,
        policy,
        episodes=args.episodes,
        seed_base=args.eval_seed_base,
        on_episode_start=(lambda _: on_start()) if on_start is not None else None,
    )
    metrics.update(
        {
            "algo": args.algo,
            "seed": args.seed,
            "checkpoint": str(args.checkpoint),
            "evaluation_mode": "posterior_sample" if args.bdqn_sampled_eval else "posterior_mean",
            "eval_seed_base": args.eval_seed_base,
        }
    )

    for key, value in metrics.items():
        print(f"{key}: {value}")

    if args.json_out:
        output = Path(args.json_out)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w") as file:
            json.dump(metrics, file, indent=2, allow_nan=True)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from types import SimpleNamespace

from scripts.train_thesis_multi import actions_for, make_agent, make_env
from uav_search_belief20.agents.bdqn_agent import BDQNAgent
from uav_search_belief20.experiments.thesis_automation import (
    load_json,
    resolve_checkpoint,
    write_csv,
    write_json,
)
from uav_search_belief20.experiments.thesis_evaluation import evaluate_multi_detailed
from uav_search_belief20.marl.thesis_qmix import ThesisBayesianQMIXAgent
from uav_search_belief20.utils import pick_device, seed_everything


DETERMINISTIC_ALGOS = {"shared_ddqn", "qmix_ddqn"}
BAYES_QMIX_ALGOS = {"bayes_qmix_shared", "bayes_qmix_independent"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detailed, reproducible evaluation for corrected multi-UAV checkpoints."
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", default="best", help="best, latest, or a path")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--eval-seed-base", type=int, default=200_000)
    parser.add_argument(
        "--policy-modes",
        default="auto",
        help=(
            "auto, deterministic, posterior_mean, posterior_sample_shared, "
            "posterior_sample_independent, or a comma-separated list"
        ),
    )
    parser.add_argument("--tag", default="final_test")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--posterior-eval-seed-base", type=int, default=900_000)
    return parser


def _modes(algo: str, requested: str) -> list[str]:
    if requested.strip().lower() == "auto":
        if algo in DETERMINISTIC_ALGOS:
            return ["deterministic"]
        if algo == "shared_bdqn":
            return ["posterior_mean", "posterior_sample_shared"]
        return [
            "posterior_mean",
            "posterior_sample_shared",
            "posterior_sample_independent",
        ]

    modes = [item.strip() for item in requested.split(",") if item.strip()]
    allowed = {
        "deterministic",
        "posterior_mean",
        "posterior_sample_shared",
        "posterior_sample_independent",
    }
    unknown = sorted(set(modes) - allowed)
    if unknown:
        raise ValueError(f"Unknown policy modes: {unknown}")
    if algo in DETERMINISTIC_ALGOS and modes != ["deterministic"]:
        raise ValueError(f"{algo} supports only deterministic evaluation.")
    if algo == "shared_bdqn" and "posterior_sample_independent" in modes:
        raise ValueError("shared_bdqn has one shared posterior sample, not independent samples.")
    return modes


def _fresh_agent(config: dict, checkpoint: Path, device: str):
    values = dict(config)
    values["device"] = device
    values["warmstart_ddqn"] = ""
    args = SimpleNamespace(**values)
    env = make_env(args, int(args.seed))
    agent = make_agent(args, env, device)
    agent.load(str(checkpoint))
    return args, agent


def main() -> None:
    cli = build_parser().parse_args()
    run_dir = Path(cli.run_dir)
    config = load_json(run_dir / "run_config.json")
    algo = str(config.get("algo", ""))
    allowed_algos = DETERMINISTIC_ALGOS | BAYES_QMIX_ALGOS | {"shared_bdqn"}
    if algo not in allowed_algos:
        raise ValueError(f"run_config.json does not describe a thesis multi-UAV run: {algo!r}")

    checkpoint = resolve_checkpoint(run_dir, cli.checkpoint)
    device = pick_device() if cli.device == "auto" else cli.device
    output_dir = Path(cli.output_dir) if cli.output_dir else run_dir / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    index = {
        "schema_version": 1,
        "scope": "multi",
        "algo": algo,
        "training_seed": int(config["seed"]),
        "detection_probability": float(config["detection_probability"]),
        "checkpoint": str(checkpoint),
        "evaluations": [],
    }

    for mode_index, mode in enumerate(_modes(algo, cli.policy_modes)):
        seed_everything(cli.posterior_eval_seed_base + int(config["seed"]) + mode_index)
        args, agent = _fresh_agent(config, checkpoint, device)

        sampled = mode.startswith("posterior_sample")
        execution_sampling: str | None = None
        if isinstance(agent, ThesisBayesianQMIXAgent):
            if mode == "posterior_sample_shared":
                agent.cfg.posterior_sampling = "shared"
                execution_sampling = "shared"
            elif mode == "posterior_sample_independent":
                agent.cfg.posterior_sampling = "independent"
                execution_sampling = "independent"
            else:
                execution_sampling = "mean"
            agent.resample_policy()
        elif isinstance(agent, BDQNAgent):
            execution_sampling = "shared" if sampled else "mean"

        def policy(env, obs_all, masks, episode_index):
            del env, episode_index
            return actions_for(
                agent,
                algo,
                obs_all,
                masks,
                train=False,
                sampled=sampled,
            )

        def on_episode_start(episode_index: int, world_seed: int) -> None:
            del episode_index, world_seed
            if sampled:
                agent.resample_policy()

        rows, summary = evaluate_multi_detailed(
            env_factory=lambda seed: make_env(args, seed),
            policy=policy,
            episodes=cli.episodes,
            seed_base=cli.eval_seed_base,
            on_episode_start=on_episode_start if sampled else None,
        )

        stem = f"{cli.tag}_{mode}"
        episode_path = output_dir / f"{stem}_episodes.csv"
        summary_path = output_dir / f"{stem}_summary.json"
        write_csv(episode_path, rows)
        payload = {
            "schema_version": 1,
            "scope": "multi",
            "algo": algo,
            "training_seed": int(config["seed"]),
            "detection_probability": float(config["detection_probability"]),
            "reward_mode": config.get("reward_mode", "unknown"),
            "global_state_mode": config.get("global_state_mode", "unknown"),
            "n_agents": int(config.get("n_agents", 3)),
            "policy_mode": mode,
            "checkpoint": str(checkpoint),
            "eval_seed_base": int(cli.eval_seed_base),
            "episodes": int(cli.episodes),
            "episode_csv": str(episode_path),
            "summary": summary,
        }
        if isinstance(agent, ThesisBayesianQMIXAgent):
            payload.update(
                {
                    "trained_posterior_sampling": str(config.get("algo", "")).replace(
                        "bayes_qmix_", ""
                    ),
                    "execution_posterior_sampling": execution_sampling,
                    **agent.head.diagnostics(),
                }
            )
        elif isinstance(agent, BDQNAgent):
            payload.update(
                {
                    "execution_posterior_sampling": execution_sampling,
                    "blr_lambda": float(agent.cfg.blr_lambda),
                    "blr_noise_var": float(agent.cfg.blr_noise_var),
                    "features_frozen": bool(agent.features_frozen),
                    "posterior_rebuilds": int(agent.posterior_rebuilds),
                }
            )
        write_json(summary_path, payload)
        index["evaluations"].append(str(summary_path))
        print(
            f"[{mode}] reward={summary.get('reward_mean', float('nan')):.3f} "
            f"completed={summary.get('completed_mean', float('nan')):.3f} "
            f"coverage={summary.get('team_coverage_ratio_mean', float('nan')):.3f} "
            f"collision={summary.get('collision_agent_ratio', float('nan')):.3f}"
        )
        print(f"Saved {summary_path}")

    write_json(output_dir / f"{cli.tag}_evaluation_index.json", index)


if __name__ == "__main__":
    main()

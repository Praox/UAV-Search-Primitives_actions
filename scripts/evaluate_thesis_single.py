from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from types import SimpleNamespace

from scripts.train_thesis_single import make_agent, make_env
from uav_search_belief20.agents.bdqn_agent import BDQNAgent
from uav_search_belief20.experiments.thesis_automation import (
    load_json,
    resolve_checkpoint,
    write_csv,
    write_json,
)
from uav_search_belief20.experiments.thesis_evaluation import (
    evaluate_single_detailed,
)
from uav_search_belief20.utils import pick_device, seed_everything


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detailed, reproducible evaluation for thesis single-UAV checkpoints."
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", default="best", help="best, latest, or a path")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--eval-seed-base", type=int, default=200_000)
    parser.add_argument(
        "--policy-modes",
        default="auto",
        help="auto, deterministic, posterior_mean, posterior_sample, or comma-separated modes",
    )
    parser.add_argument("--tag", default="final_test")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--posterior-eval-seed-base", type=int, default=900_000)
    return parser


def _modes(algo: str, requested: str) -> list[str]:
    if requested.strip().lower() == "auto":
        return ["deterministic"] if algo == "ddqn" else ["posterior_mean", "posterior_sample"]
    modes = [item.strip() for item in requested.split(",") if item.strip()]
    allowed = {"deterministic", "posterior_mean", "posterior_sample"}
    unknown = sorted(set(modes) - allowed)
    if unknown:
        raise ValueError(f"Unknown policy modes: {unknown}")
    if algo == "ddqn" and any(mode != "deterministic" for mode in modes):
        raise ValueError("DDQN supports only deterministic evaluation.")
    return modes


def _fresh_agent(config: dict, checkpoint: Path, device: str):
    values = dict(config)
    values["device"] = device
    values["warmstart_ddqn"] = ""  # The checkpoint already contains the learned features.
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
    if algo not in {"ddqn", "bdqn"}:
        raise ValueError(f"run_config.json does not describe a thesis single-UAV run: algo={algo!r}")

    checkpoint = resolve_checkpoint(run_dir, cli.checkpoint)
    device = pick_device() if cli.device == "auto" else cli.device
    output_dir = Path(cli.output_dir) if cli.output_dir else run_dir / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    index = {
        "schema_version": 1,
        "scope": "single",
        "algo": algo,
        "training_seed": int(config["seed"]),
        "detection_probability": float(config["detection_probability"]),
        "checkpoint": str(checkpoint),
        "evaluations": [],
    }

    for mode_index, mode in enumerate(_modes(algo, cli.policy_modes)):
        # Construct a fresh agent per mode so posterior RNG consumption from one
        # evaluation cannot affect another evaluation.
        seed_everything(cli.posterior_eval_seed_base + int(config["seed"]) + mode_index)
        args, agent = _fresh_agent(config, checkpoint, device)
        sampled = mode == "posterior_sample"

        def policy(env, obs, episode_index):
            del episode_index
            if isinstance(agent, BDQNAgent):
                return agent.act(obs, use_sample=sampled, action_mask=env.action_mask())
            return agent.act(obs, explore=False, action_mask=env.action_mask())

        def on_episode_start(episode_index: int, world_seed: int) -> None:
            del episode_index, world_seed
            if sampled:
                agent.resample_policy()

        rows, summary = evaluate_single_detailed(
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
            "scope": "single",
            "algo": algo,
            "training_seed": int(config["seed"]),
            "detection_probability": float(config["detection_probability"]),
            "reward_mode": config.get("reward_mode", "unknown"),
            "policy_mode": mode,
            "checkpoint": str(checkpoint),
            "eval_seed_base": int(cli.eval_seed_base),
            "episodes": int(cli.episodes),
            "episode_csv": str(episode_path),
            "summary": summary,
        }
        if isinstance(agent, BDQNAgent):
            payload.update(
                {
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
            f"coverage={summary.get('sensor_coverage_ratio_mean', float('nan')):.3f}"
        )
        print(f"Saved {summary_path}")

    write_json(output_dir / f"{cli.tag}_evaluation_index.json", index)


if __name__ == "__main__":
    main()

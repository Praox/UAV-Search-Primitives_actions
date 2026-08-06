from __future__ import annotations

import argparse
import json
from pathlib import Path

import train_thesis_multi as multi
import train_thesis_single as single

from uav_search_belief20.utils import pick_device, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scope",
        choices=["single", "multi"],
        required=True,
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--seed-base", type=int, default=200_000)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "mps", "cuda"],
        default="auto",
    )
    args_cli = parser.parse_args()

    run_dir = args_cli.run_dir
    config_path = run_dir / "run_config.json"
    checkpoint_path = run_dir / "best.pt"

    if not config_path.exists():
        raise FileNotFoundError(f"Missing configuration: {config_path}")

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    with config_path.open() as handle:
        config = json.load(handle)

    # The full checkpoint is loaded below. There is no need to reload
    # the original DDQN warm-start before loading best.pt.
    config["warmstart_ddqn"] = ""
    config["run_dir"] = str(run_dir)
    config["final_test_episodes"] = args_cli.episodes
    config["final_test_seed_base"] = args_cli.seed_base
    config["device"] = args_cli.device

    saved_args = argparse.Namespace(**config)

    seed_everything(saved_args.seed)

    device = (
        pick_device()
        if args_cli.device == "auto"
        else args_cli.device
    )

    if args_cli.scope == "single":
        env = single.make_env(saved_args, saved_args.seed)
        agent = single.make_agent(saved_args, env, device)
        agent.load(str(checkpoint_path))

        output = {
            "posterior_mean": single.evaluate(
                agent,
                saved_args,
                args_cli.seed_base,
                args_cli.episodes,
                sampled=False,
            )
        }

        if saved_args.algo == "bdqn":
            output["posterior_sample"] = single.evaluate(
                agent,
                saved_args,
                args_cli.seed_base,
                args_cli.episodes,
                sampled=True,
            )

    else:
        env = multi.make_env(saved_args, saved_args.seed)
        agent = multi.make_agent(saved_args, env, device)
        agent.load(str(checkpoint_path))

        output = {
            "deterministic_or_posterior_mean": multi.evaluate(
                agent,
                saved_args,
                args_cli.seed_base,
                args_cli.episodes,
                sampled=False,
            )
        }

        if (
            saved_args.algo == "shared_bdqn"
            or saved_args.algo.startswith("bayes_qmix")
        ):
            output["posterior_sample"] = multi.evaluate(
                agent,
                saved_args,
                args_cli.seed_base,
                args_cli.episodes,
                sampled=True,
            )

    output_path = run_dir / "final_test.json"
    with output_path.open("w") as handle:
        json.dump(output, handle, indent=2, allow_nan=True)

    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Evaluated worlds: {args_cli.seed_base}–"
          f"{args_cli.seed_base + args_cli.episodes - 1}")
    print(f"Saved results: {output_path}")


if __name__ == "__main__":
    main()
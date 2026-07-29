from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from uav_search_belief20.baselines import make_baseline
from uav_search_belief20.envs.primitive_search_env import PrimitiveSearchEnv
from uav_search_belief20.evaluation import evaluate_policy
from uav_search_belief20.experiments.single_ablation import ABLATIONS, build_env_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation", choices=[*ABLATIONS, "all"], default="all")
    parser.add_argument("--baselines", nargs="+", default=["random", "frontier", "oracle"])
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--eval-seed-base", type=int, default=200_000)
    parser.add_argument("--policy-seed", type=int, default=999)
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
    parser.add_argument("--output-dir", type=str, default="logs/single_baselines")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    variants = list(ABLATIONS) if args.ablation == "all" else [args.ablation]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, object]] = []

    for variant in variants:
        args.ablation = variant
        for baseline_name in args.baselines:
            policy_object = make_baseline(baseline_name, seed=args.policy_seed)

            def env_factory(seed: int) -> PrimitiveSearchEnv:
                return PrimitiveSearchEnv(build_env_config(args, seed=seed))

            metrics = evaluate_policy(
                env_factory,
                policy_object.act,
                episodes=args.episodes,
                seed_base=args.eval_seed_base,
            )
            metrics.update(
                {
                    "algo": f"baseline_{baseline_name}",
                    "baseline": baseline_name,
                    "seed": args.policy_seed,
                    "eval_seed_base": args.eval_seed_base,
                }
            )
            all_rows.append(metrics)

            variant_dir = output_dir / variant
            variant_dir.mkdir(parents=True, exist_ok=True)
            json_path = variant_dir / f"baseline_{baseline_name}_eval{args.episodes}.json"
            with json_path.open("w") as file:
                json.dump(metrics, file, indent=2, allow_nan=True)
            print(f"Saved {json_path}")

    # Rebuild the global summary from every per-variant JSON already present.
    # This keeps resume/partial runs correct instead of overwriting the CSV with
    # only the last variant evaluated.
    summary_rows: list[dict[str, object]] = []
    for json_path in sorted(output_dir.glob("*/baseline_*_eval*.json")):
        try:
            row = json.loads(json_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(row, dict):
            summary_rows.append(row)

    csv_path = output_dir / "baseline_summary.csv"
    fieldnames = sorted({key for row in summary_rows for key in row if key != "action_counts"})
    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Saved {csv_path}")


if __name__ == "__main__":
    main()

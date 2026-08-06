from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path

from uav_search_belief20.experiments.thesis_automation import (
    mean_std_ci95,
)


METRICS = [
    "reward_mean",
    "detected_mean",
    "completed_mean",
    "detected_value_mean",
    "completed_value_mean",
    "sensor_coverage_ratio_mean",
    "team_coverage_ratio_mean",
    "mean_local_coverage_ratio_mean",
    "coverage_overlap_ratio_mean",
    "knowledge_overlap_ratio_mean",
    "collision_agent_ratio",
    "first_detection_step_mean",
    "first_completion_step_mean",
    "had_detection_mean",
    "had_completion_mean",
    "detected_to_completed_ratio",
    "stay_ratio",
    "tracking_progress_ratio",
]


def select_result_block(
    payload: dict,
    mode: str,
) -> dict:
    if mode == "sample":
        block = payload.get("posterior_sample")
        if not isinstance(block, dict):
            raise KeyError("No posterior_sample result")
        return block

    for key in (
        "posterior_mean",
        "deterministic_or_posterior_mean",
    ):
        block = payload.get(key)
        if isinstance(block, dict):
            return block

    raise KeyError("No posterior-mean or deterministic result")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--glob",
        required=True,
        help='Example: "runs/single/ddqn/seed*/final_test.json"',
    )
    parser.add_argument(
        "--mode",
        choices=["mean", "sample"],
        default="mean",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    paths = sorted(Path(path) for path in glob.glob(args.glob))

    if not paths:
        raise FileNotFoundError(f"No file found for {args.glob}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    per_seed_rows: list[dict[str, object]] = []

    for path in paths:
        config_path = path.parent / "run_config.json"

        with path.open() as handle:
            payload = json.load(handle)

        with config_path.open() as handle:
            config = json.load(handle)

        try:
            result = select_result_block(payload, args.mode)
        except KeyError as error:
            print(f"[skip] {path}: {error}")
            continue

        row: dict[str, object] = {
            "seed": int(config["seed"]),
            "algo": str(config["algo"]),
            "run_dir": str(path.parent),
        }

        for metric in METRICS:
            value = result.get(metric)
            if isinstance(value, (int, float)):
                row[metric] = float(value)

        per_seed_rows.append(row)

    if not per_seed_rows:
        raise RuntimeError("No usable final-test result")

    per_seed_path = args.output_dir / "per_seed.csv"
    fieldnames = sorted(
        {key for row in per_seed_rows for key in row}
    )

    with per_seed_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_seed_rows)

    summary_rows: list[dict[str, object]] = []

    for metric in METRICS:
        values = [
            float(row[metric])
            for row in per_seed_rows
            if metric in row
        ]

        if not values:
            continue

        stats = mean_std_ci95(values)

        summary_rows.append(
            {
                "metric": metric,
                "n_training_seeds": stats["n"],
                "mean": stats["mean"],
                "std": stats["std"],
                "ci95_low": stats["ci95_low"],
                "ci95_high": stats["ci95_high"],
            }
        )

    summary_path = args.output_dir / "summary_across_seeds.csv"

    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "metric",
                "n_training_seeds",
                "mean",
                "std",
                "ci95_low",
                "ci95_high",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Per-seed results: {per_seed_path}")
    print(f"Across-seed summary: {summary_path}")


if __name__ == "__main__":
    main()
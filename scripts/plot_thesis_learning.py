#!/usr/bin/env python3
"""Plot thesis learning curves from one or more training-seed folders.

Examples
--------
Single group:
    python scripts/plot_thesis_learning.py \
        --group "DDQN ego=runs/single_ddqn_ego/seed*" \
        --output-dir figures/single_ddqn

Ablation comparison:
    python scripts/plot_thesis_learning.py \
        --group "global 5-1-1=runs/ablation/A/seed*" \
        --group "global 10-1-2=runs/ablation/B/seed*" \
        --group "ego 5-1-1=runs/ablation/C/seed*" \
        --group "ego 10-1-2=runs/ablation/D/seed*" \
        --output-dir figures/ablation_single \
        --smooth 3 \
        --show-seeds

Optional random baseline:
    python scripts/plot_thesis_learning.py \
        --group "DDQN ego=runs/single_ddqn_ego/seed*" \
        --baseline "Random=runs/random_single/summary.json" \
        --output-dir figures/single_vs_random
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_METRICS = [
    "validation_completed",
    "validation_completed_value",
    "validation_coverage",
    "validation_reward",
    "validation_first_completion",
    "validation_first_detection",
    "validation_tracking_progress",
    "validation_stay",
    "validation_collision",
    "validation_overlap",
    "loss",
    "q_mean",
    "target_mean",
    "epsilon",
]

METRIC_LABELS = {
    "validation_completed": "Completed target",
    "validation_completed_value": "Completed value",
    "validation_coverage": "Coverage",
    "validation_reward": "Validation reward",
    "validation_first_completion": "First completion (step)",
    "validation_first_detection": "First detection (step)",
    "validation_tracking_progress": "Tracking progress ratio",
    "validation_stay": "STAY Ratio ",
    "validation_collision": "collision ratio",
    "validation_overlap": "Coverage overlap ratio",
    "loss": "Loss",
    "q_mean": "Q mean",
    "target_mean": "Target Bellman mean",
    "epsilon": "Epsilon",
}

LOWER_IS_BETTER = {
    "validation_first_completion",
    "validation_first_detection",
    "validation_collision",
    "validation_overlap",
    "loss",
}

BASELINE_KEY_MAP = {
    "validation_completed": [
        "validation_completed",
        "completed_mean",
    ],
    "validation_completed_value": [
        "validation_completed_value",
        "completed_value_mean",
    ],
    "validation_coverage": [
        "validation_coverage",
        "sensor_coverage_ratio_mean",
        "team_coverage_ratio_mean",
    ],
    "validation_reward": [
        "validation_reward",
        "reward_mean",
    ],
    "validation_first_completion": [
        "validation_first_completion",
        "first_completion_step_mean",
    ],
    "validation_first_detection": [
        "validation_first_detection",
        "first_detection_step_mean",
    ],
    "validation_tracking_progress": [
        "validation_tracking_progress",
        "tracking_progress_ratio",
    ],
    "validation_stay": [
        "validation_stay",
        "stay_ratio",
    ],
    "validation_collision": [
        "validation_collision",
        "collision_agent_ratio",
    ],
    "validation_overlap": [
        "validation_overlap",
        "coverage_overlap_ratio_mean",
    ],
}


def parse_named_value(text: str) -> tuple[str, str]:
    if "=" not in text:
        raise argparse.ArgumentTypeError(
            f"Expected LABEL=PATH_OR_GLOB, got: {text!r}"
        )
    label, value = text.split("=", 1)
    label = label.strip()
    value = value.strip()
    if not label or not value:
        raise argparse.ArgumentTypeError(
            f"Expected non-empty LABEL=PATH_OR_GLOB, got: {text!r}"
        )
    return label, value


def read_metrics_csv(path: Path) -> dict[str, np.ndarray]:
    columns: dict[str, list[float]] = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"No CSV header in {path}")
        for row in reader:
            for key, raw in row.items():
                if raw is None or raw == "":
                    continue
                try:
                    value = float(raw)
                except ValueError:
                    continue
                columns.setdefault(key, []).append(value)

    return {
        key: np.asarray(values, dtype=np.float64)
        for key, values in columns.items()
    }


def discover_metric_files(pattern: str) -> list[Path]:
    raw_matches = [Path(item) for item in glob.glob(pattern)]
    if not raw_matches:
        direct = Path(pattern)
        if direct.exists():
            raw_matches = [direct]

    output: list[Path] = []
    for match in raw_matches:
        if match.is_file() and match.name == "metrics.csv":
            output.append(match)
        elif match.is_dir():
            direct_metrics = match / "metrics.csv"
            if direct_metrics.exists():
                output.append(direct_metrics)
            else:
                output.extend(match.rglob("metrics.csv"))

    return sorted(set(path.resolve() for path in output))


def trailing_mean(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or values.size <= 1:
        return values.copy()

    output = np.empty_like(values, dtype=np.float64)
    for index in range(values.size):
        start = max(0, index - window + 1)
        current = values[start : index + 1]
        finite = current[np.isfinite(current)]
        output[index] = (
            float(finite.mean()) if finite.size else float("nan")
        )
    return output


def t_critical_95(n: int) -> float:
    table = {
        2: 12.706,
        3: 4.303,
        4: 3.182,
        5: 2.776,
        6: 2.571,
        7: 2.447,
        8: 2.365,
        9: 2.306,
        10: 2.262,
        11: 2.228,
        12: 2.201,
        13: 2.179,
        14: 2.160,
        15: 2.145,
        16: 2.131,
        17: 2.120,
        18: 2.110,
        19: 2.101,
        20: 2.093,
        21: 2.086,
        22: 2.080,
        23: 2.074,
        24: 2.069,
        25: 2.064,
        26: 2.060,
        27: 2.056,
        28: 2.052,
        29: 2.048,
        30: 2.045,
        31: 2.042,
    }
    if n <= 1:
        return float("nan")
    return table.get(n, 1.96)


def align_runs(
    runs: list[dict[str, np.ndarray]],
    metric: str,
    x_key: str,
    smooth: int,
) -> tuple[np.ndarray, np.ndarray]:
    x_union: set[float] = set()
    valid_runs: list[tuple[np.ndarray, np.ndarray]] = []

    for run in runs:
        if x_key not in run or metric not in run:
            continue
        length = min(run[x_key].size, run[metric].size)
        x = run[x_key][:length]
        y = trailing_mean(run[metric][:length], smooth)
        finite = np.isfinite(x) & np.isfinite(y)
        x, y = x[finite], y[finite]
        if x.size == 0:
            continue
        valid_runs.append((x, y))
        x_union.update(float(value) for value in x)

    if not valid_runs:
        return np.empty(0), np.empty((0, 0))

    x_values = np.asarray(sorted(x_union), dtype=np.float64)
    matrix = np.full(
        (len(valid_runs), x_values.size),
        np.nan,
        dtype=np.float64,
    )
    index_by_x = {float(value): index for index, value in enumerate(x_values)}

    for run_index, (x, y) in enumerate(valid_runs):
        for x_value, y_value in zip(x, y):
            matrix[run_index, index_by_x[float(x_value)]] = y_value

    return x_values, matrix


def group_statistics(
    matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    count = np.sum(np.isfinite(matrix), axis=0)
    mean = np.full(matrix.shape[1], np.nan, dtype=np.float64)
    low = np.full_like(mean, np.nan)
    high = np.full_like(mean, np.nan)

    for index in range(matrix.shape[1]):
        values = matrix[:, index]
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue

        mean[index] = float(values.mean())
        if values.size == 1:
            low[index] = high[index] = mean[index]
            continue

        std = float(values.std(ddof=1))
        half_width = (
            t_critical_95(int(values.size))
            * std
            / math.sqrt(float(values.size))
        )
        low[index] = mean[index] - half_width
        high[index] = mean[index] + half_width

    return count, mean, low, high


def flatten_baseline_payload(payload: object) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise ValueError("Baseline JSON must contain an object.")

    # Common outputs produced by thesis evaluation scripts.
    for key in (
        "summary",
        "posterior_mean",
        "deterministic_or_posterior_mean",
    ):
        candidate = payload.get(key)
        if isinstance(candidate, dict):
            return candidate

    return payload


def read_baseline(path: Path) -> dict[str, float]:
    if path.suffix.lower() == ".json":
        with path.open() as handle:
            payload = flatten_baseline_payload(json.load(handle))
        output: dict[str, float] = {}
        for metric, candidate_keys in BASELINE_KEY_MAP.items():
            for key in candidate_keys:
                value = payload.get(key)
                try:
                    output[metric] = float(value)
                    break
                except (TypeError, ValueError):
                    continue
        return output

    if path.suffix.lower() == ".csv":
        data = read_metrics_csv(path)
        output = {}
        for metric in BASELINE_KEY_MAP:
            if metric in data and data[metric].size:
                output[metric] = float(np.nanmean(data[metric]))
        return output

    raise ValueError(
        f"Unsupported baseline format for {path}; use JSON or CSV."
    )


def safe_filename(metric: str) -> str:
    return "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in metric
    )


def early_late_summary(
    label: str,
    metric: str,
    mean: np.ndarray,
    count: np.ndarray,
    points: int,
) -> dict[str, object] | None:
    finite_indices = np.flatnonzero(
        np.isfinite(mean) & (count > 0)
    )
    if finite_indices.size == 0:
        return None

    early_indices = finite_indices[:points]
    late_indices = finite_indices[-points:]
    early = float(np.mean(mean[early_indices]))
    late = float(np.mean(mean[late_indices]))
    raw_delta = late - early
    improvement = early - late if metric in LOWER_IS_BETTER else raw_delta

    return {
        "group": label,
        "metric": metric,
        "early": early,
        "late": late,
        "raw_delta_late_minus_early": raw_delta,
        "improvement_positive_is_better": improvement,
        "last_available_seed_count": int(count[finite_indices[-1]]),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--group",
        action="append",
        required=True,
        type=parse_named_value,
        metavar="LABEL=PATH_OR_GLOB",
        help=(
            "Training group. The path may be a metrics.csv, one run "
            "directory, or a glob such as runs/experiment/seed*."
        ),
    )
    parser.add_argument(
        "--baseline",
        action="append",
        default=[],
        type=parse_named_value,
        metavar="LABEL=JSON_OR_CSV",
        help="Optional horizontal baseline from an evaluation summary.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--x-axis",
        choices=["episode", "global_step"],
        default="episode",
    )
    parser.add_argument(
        "--metrics",
        default=",".join(DEFAULT_METRICS),
        help="Comma-separated metric names.",
    )
    parser.add_argument(
        "--smooth",
        type=int,
        default=1,
        help="Trailing moving-average window over validation points.",
    )
    parser.add_argument(
        "--summary-points",
        type=int,
        default=3,
        help="Number of early/late evaluation points used in summary.csv.",
    )
    parser.add_argument(
        "--show-seeds",
        action="store_true",
        help="Show each individual training seed in addition to the mean.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
    )
    parser.add_argument(
        "--title-prefix",
        default="",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.smooth <= 0:
        raise ValueError("--smooth must be positive.")
    if args.summary_points <= 0:
        raise ValueError("--summary-points must be positive.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics = [
        metric.strip()
        for metric in args.metrics.split(",")
        if metric.strip()
    ]

    groups: dict[str, list[dict[str, np.ndarray]]] = {}
    for label, pattern in args.group:
        files = discover_metric_files(pattern)
        if not files:
            raise FileNotFoundError(
                f"No metrics.csv found for group {label!r}: {pattern}"
            )

        runs = [read_metrics_csv(path) for path in files]
        groups[label] = runs
        print(f"[{label}] {len(files)} seed/run file(s)")
        for path in files:
            print(f"  - {path}")

    baselines: dict[str, dict[str, float]] = {}
    for label, raw_path in args.baseline:
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(path)
        baselines[label] = read_baseline(path)
        print(f"[baseline {label}] {path}")

    summary_rows: list[dict[str, object]] = []
    plotted_metrics = 0

    for metric in metrics:
        available = any(
            any(metric in run for run in runs)
            for runs in groups.values()
        )
        if not available:
            print(f"[skip] metric absent: {metric}")
            continue

        plt.figure(figsize=(8.5, 5.2))
        group_was_plotted = False

        for label, runs in groups.items():
            x_values, matrix = align_runs(
                runs,
                metric,
                args.x_axis,
                args.smooth,
            )
            if matrix.size == 0:
                continue

            count, mean, low, high = group_statistics(matrix)
            finite = np.isfinite(mean)
            if not np.any(finite):
                continue

            if args.show_seeds:
                for seed_values in matrix:
                    seed_finite = np.isfinite(seed_values)
                    plt.plot(
                        x_values[seed_finite],
                        seed_values[seed_finite],
                        linewidth=0.8,
                        alpha=0.22,
                    )

            line = plt.plot(
                x_values[finite],
                mean[finite],
                linewidth=2.2,
                #Avoir le nombre de seeds ds le label
                ##label=f"{label} (n={matrix.shape[0]})",
                label=f"{label}",
            )[0]

            # Use the automatically selected line color for its own CI band.
            ci_finite = finite & np.isfinite(low) & np.isfinite(high)
            if np.any(ci_finite) and matrix.shape[0] >= 2:
                plt.fill_between(
                    x_values[ci_finite],
                    low[ci_finite],
                    high[ci_finite],
                    alpha=0.15,
                    color=line.get_color(),
                )

            summary = early_late_summary(
                label,
                metric,
                mean,
                count,
                args.summary_points,
            )
            if summary is not None:
                summary_rows.append(summary)

            group_was_plotted = True

        for baseline_label, values in baselines.items():
            if metric not in values:
                continue
            plt.axhline(
                values[metric],
                linestyle="--",
                linewidth=1.7,
                label=f"{baseline_label}: {values[metric]:.3f}",
            )

        if not group_was_plotted:
            plt.close()
            continue

        title = METRIC_LABELS.get(metric, metric)
        if args.title_prefix:
            title = f"{args.title_prefix} — {title}"

        plt.title(title)
        plt.xlabel(
            "Épisode d'entraînement"
            if args.x_axis == "episode"
            else "Étapes environnement"
        )
        plt.ylabel(METRIC_LABELS.get(metric, metric))
        plt.grid(True, alpha=0.25)
        plt.legend()
        plt.tight_layout()

        output_path = (
            args.output_dir
            / f"{safe_filename(metric)}.png"
        )
        plt.savefig(output_path, dpi=args.dpi)
        plt.close()
        plotted_metrics += 1
        print(f"[plot] {output_path}")

    summary_path = args.output_dir / "learning_summary.csv"
    fieldnames = [
        "group",
        "metric",
        "early",
        "late",
        "raw_delta_late_minus_early",
        "improvement_positive_is_better",
        "last_available_seed_count",
    ]
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"[summary] {summary_path}")
    print(f"Generated {plotted_metrics} plot(s).")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

PRIMARY_METRICS = (
    "reward_mean",
    "detected_mean",
    "completed_mean",
    "detected_value_mean",
    "completed_value_mean",
    "detected_to_completed_ratio",
    "team_coverage_ratio_mean",
    "mean_local_coverage_ratio_mean",
    "coverage_overlap_ratio_mean",
    "knowledge_overlap_ratio_mean",
    "collision_agent_ratio",
    "local_sensor_revisit_ratio",
    "team_new_observed_cells_per_env_step",
    "tracking_progress_ratio",
)


def ci95(values: list[float]) -> tuple[float, float]:
    array = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if array.size == 0:
        return float("nan"), float("nan")
    mean = float(array.mean())
    if array.size == 1:
        return mean, mean
    half = 1.96 * float(array.std(ddof=1)) / math.sqrt(array.size)
    return mean - half, mean + half


def load_runs(root: Path) -> list[dict]:
    rows = []
    for path in sorted(root.rglob("*_eval.json")):
        try:
            with path.open() as handle:
                row = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        if "algo" in row and "seed" in row:
            row["source_file"] = str(path)
            rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[str(row["algo"])].append(row)
    output = []
    for algo, group in sorted(grouped.items()):
        summary: dict[str, object] = {
            "algo": algo,
            "n_seeds": len(group),
            "seeds": ",".join(
                str(int(row["seed"]))
                for row in sorted(group, key=lambda item: int(item["seed"]))
            ),
        }
        for metric in PRIMARY_METRICS:
            values = [float(row[metric]) for row in group if metric in row]
            if not values:
                continue
            low, high = ci95(values)
            summary[f"{metric}_mean"] = float(np.mean(values))
            summary[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            summary[f"{metric}_ci95_low"] = low
            summary[f"{metric}_ci95_high"] = high
        output.append(summary)
    return output


def paired(rows: list[dict], first: str, second: str) -> list[dict]:
    index = {(str(row["algo"]), int(row["seed"])): row for row in rows}
    seeds = sorted(
        {seed for algo, seed in index if algo == first}
        & {seed for algo, seed in index if algo == second}
    )
    if not seeds:
        return []
    result: dict[str, object] = {
        "comparison": f"{first}_minus_{second}",
        "n_pairs": len(seeds),
        "seeds": ",".join(map(str, seeds)),
    }
    for metric in PRIMARY_METRICS:
        differences = []
        for seed in seeds:
            left, right = index[(first, seed)], index[(second, seed)]
            if metric in left and metric in right:
                differences.append(float(left[metric]) - float(right[metric]))
        if differences:
            low, high = ci95(differences)
            result[f"{metric}_difference_mean"] = float(np.mean(differences))
            result[f"{metric}_difference_ci95_low"] = low
            result[f"{metric}_difference_ci95_high"] = high
    return [result]


def write_markdown(path: Path, summary: list[dict]) -> None:
    columns = [
        "algo",
        "n_seeds",
        "completed_mean_mean",
        "completed_value_mean_mean",
        "detected_mean_mean",
        "team_coverage_ratio_mean_mean",
        "coverage_overlap_ratio_mean_mean",
        "collision_agent_ratio_mean",
        "reward_mean_mean",
    ]
    lines = [
        "# Multi-UAV local-memory summary",
        "",
        "| " + " | ".join(columns) + " |",
        "|" + "|".join(["---"] * len(columns)) + "|",
    ]
    for row in summary:
        cells = []
        for column in columns:
            value = row.get(column, "")
            cells.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    rows = load_runs(Path(args.input_root))
    if not rows:
        raise RuntimeError(f"No evaluation JSON below {args.input_root}")
    output = Path(args.output_dir)
    summary = aggregate(rows)
    write_csv(output / "multi_runs.csv", rows)
    write_csv(output / "multi_summary.csv", summary)
    write_csv(
        output / "paired_qmix_vs_shared_ddqn.csv",
        paired(rows, "qmix_ddqn", "shared_ddqn"),
    )
    write_csv(
        output / "paired_shared_bdqn_vs_ddqn.csv",
        paired(rows, "shared_bdqn", "shared_ddqn"),
    )
    write_markdown(output / "multi_summary.md", summary)
    print(f"Aggregated {len(rows)} runs into {output}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Iterable


METRICS = (
    "reward_mean",
    "detected_mean",
    "completed_mean",
    "detected_value_mean",
    "completed_value_mean",
    "sensor_coverage_ratio_mean",
    "detected_to_completed_ratio",
    "stay_ratio",
    "boundary_hit_ratio",
    "sensor_revisit_ratio",
    "new_observed_cells_per_step",
    "tracking_progress_ratio",
)

# Two-sided Student-t critical values for a 95% confidence interval.
T_975 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    24: 2.064,
    30: 2.042,
}


def _t_critical(df: int) -> float:
    if df <= 0:
        return float("nan")
    if df in T_975:
        return T_975[df]
    smaller = [key for key in T_975 if key <= df]
    if smaller:
        return T_975[max(smaller)]
    return 1.96


def _infer_algo(path: Path, data: dict) -> str:
    if data.get("algo"):
        return str(data["algo"])
    match = re.search(r"(?:^|/)(dqn|ddqn|bdqn)_seed", path.as_posix())
    if match:
        return match.group(1)
    match = re.search(r"baseline_(random|frontier|oracle)", path.name)
    if match:
        return f"baseline_{match.group(1)}"
    return "unknown"


def _infer_seed(path: Path, data: dict) -> int:
    if data.get("seed") is not None:
        return int(data["seed"])
    match = re.search(r"seed(\d+)", path.name)
    return int(match.group(1)) if match else -1


def _infer_ablation(path: Path, data: dict) -> str:
    if data.get("ablation") not in (None, "", "unknown"):
        return str(data["ablation"])
    parts = set(path.parts)
    for variant in ("v3", "A", "B", "C", "D"):
        if variant in parts:
            return variant
    reward_version = str(data.get("reward_version", ""))
    if reward_version.startswith("v3_frontier"):
        return "v3"
    return path.parent.name


def _normalize_metrics(data: dict) -> dict[str, float]:
    aliases = {
        "reward_mean": ("reward_mean", "eval_reward"),
        "detected_mean": ("detected_mean", "eval_detected"),
        "completed_mean": ("completed_mean", "eval_completed"),
        "detected_value_mean": ("detected_value_mean", "eval_detected_value"),
        "completed_value_mean": ("completed_value_mean", "eval_completed_value"),
        "sensor_coverage_ratio_mean": (
            "sensor_coverage_ratio_mean",
            "eval_sensor_coverage",
        ),
        "detected_to_completed_ratio": (
            "detected_to_completed_ratio",
            "eval_detected_to_completed_ratio",
        ),
        "stay_ratio": ("stay_ratio",),
        "boundary_hit_ratio": ("boundary_hit_ratio",),
        "sensor_revisit_ratio": ("sensor_revisit_ratio",),
        "new_observed_cells_per_step": ("new_observed_cells_per_step",),
        "tracking_progress_ratio": ("tracking_progress_ratio",),
    }
    output: dict[str, float] = {}
    for target, candidates in aliases.items():
        value = float("nan")
        for candidate in candidates:
            if candidate in data and data[candidate] is not None:
                try:
                    value = float(data[candidate])
                except (TypeError, ValueError):
                    pass
                break
        output[target] = value
    if math.isnan(output["detected_to_completed_ratio"]):
        detected = output["detected_mean"]
        completed = output["completed_mean"]
        output["detected_to_completed_ratio"] = completed / max(detected, 1e-12)
    return output


def load_rows(roots: Iterable[Path]) -> list[dict[str, object]]:
    # Later roots override earlier roots for the same (ablation, algo, seed), which
    # lets corrected BDQN logs replace historical BDQN logs cleanly.
    rows_by_key: dict[tuple[str, str, int], dict[str, object]] = {}
    for root in roots:
        if not root.exists():
            print(f"Warning: missing root {root}")
            continue
        for path in sorted(root.rglob("*.json")):
            try:
                data = json.loads(path.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            if not isinstance(data, dict):
                continue
            algo = _infer_algo(path, data)
            seed = _infer_seed(path, data)
            ablation = _infer_ablation(path, data)
            metrics = _normalize_metrics(data)
            row: dict[str, object] = {
                "ablation": ablation,
                "algo": algo,
                "seed": seed,
                "source": str(path),
                **metrics,
            }
            rows_by_key[(ablation, algo, seed)] = row
    return sorted(
        rows_by_key.values(),
        key=lambda row: (str(row["ablation"]), str(row["algo"]), int(row["seed"])),
    )


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: defaultdict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["ablation"]), str(row["algo"]))].append(row)

    summary_rows: list[dict[str, object]] = []
    for (ablation, algo), group in sorted(grouped.items()):
        output: dict[str, object] = {
            "ablation": ablation,
            "algo": algo,
            "n_seeds": len(group),
            "seeds": ",".join(str(row["seed"]) for row in group),
        }
        for metric in METRICS:
            values = [float(row[metric]) for row in group if not math.isnan(float(row[metric]))]
            if not values:
                output[f"{metric}_mean"] = float("nan")
                output[f"{metric}_std"] = float("nan")
                output[f"{metric}_ci95_low"] = float("nan")
                output[f"{metric}_ci95_high"] = float("nan")
                continue
            mean = statistics.fmean(values)
            std = statistics.stdev(values) if len(values) >= 2 else 0.0
            half_width = (
                _t_critical(len(values) - 1) * std / math.sqrt(len(values))
                if len(values) >= 2
                else 0.0
            )
            output[f"{metric}_mean"] = mean
            output[f"{metric}_std"] = std
            output[f"{metric}_ci95_low"] = mean - half_width
            output[f"{metric}_ci95_high"] = mean + half_width
        summary_rows.append(output)
    return summary_rows


def paired_comparisons(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_variant_algo_seed: dict[tuple[str, str, int], dict[str, object]] = {
        (str(row["ablation"]), str(row["algo"]), int(row["seed"])): row
        for row in rows
    }
    variants = sorted({str(row["ablation"]) for row in rows})
    output_rows: list[dict[str, object]] = []
    for variant in variants:
        ddqn_seeds = {
            int(seed)
            for ablation, algo, seed in by_variant_algo_seed
            if ablation == variant and algo == "ddqn"
        }
        bdqn_seeds = {
            int(seed)
            for ablation, algo, seed in by_variant_algo_seed
            if ablation == variant and algo == "bdqn"
        }
        common = sorted(ddqn_seeds & bdqn_seeds)
        if not common:
            continue
        row: dict[str, object] = {
            "ablation": variant,
            "comparison": "bdqn_minus_ddqn",
            "n_pairs": len(common),
            "seeds": ",".join(map(str, common)),
        }
        for metric in METRICS:
            differences = [
                float(by_variant_algo_seed[(variant, "bdqn", seed)][metric])
                - float(by_variant_algo_seed[(variant, "ddqn", seed)][metric])
                for seed in common
            ]
            differences = [value for value in differences if not math.isnan(value)]
            if not differences:
                continue
            mean = statistics.fmean(differences)
            std = statistics.stdev(differences) if len(differences) >= 2 else 0.0
            half_width = (
                _t_critical(len(differences) - 1) * std / math.sqrt(len(differences))
                if len(differences) >= 2
                else 0.0
            )
            row[f"{metric}_difference_mean"] = mean
            row[f"{metric}_difference_ci95_low"] = mean - half_width
            row[f"{metric}_difference_ci95_high"] = mean + half_width
        output_rows.append(row)
    return output_rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, summary_rows: list[dict[str, object]]) -> None:
    lines = [
        "# Single-UAV summary",
        "",
        "| Ablation | Algorithm | Seeds | Reward | Completed | Coverage | Detect→complete | Boundary |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| {ablation} | {algo} | {n_seeds} | {reward:.3f} | {completed:.3f} | {coverage:.3f} | {ratio:.3f} | {boundary:.3f} |".format(
                ablation=row["ablation"],
                algo=row["algo"],
                n_seeds=row["n_seeds"],
                reward=float(row["reward_mean_mean"]),
                completed=float(row["completed_mean_mean"]),
                coverage=float(row["sensor_coverage_ratio_mean_mean"]),
                ratio=float(row["detected_to_completed_ratio_mean"]),
                boundary=float(row["boundary_hit_ratio_mean"]),
            )
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", nargs="+", required=True)
    parser.add_argument("--output-dir", default="logs/single_aggregate")
    args = parser.parse_args()

    rows = load_rows(Path(root) for root in args.roots)
    summary_rows = summarize(rows)
    paired_rows = paired_comparisons(rows)
    output_dir = Path(args.output_dir)
    write_csv(output_dir / "single_runs.csv", rows)
    write_csv(output_dir / "single_summary.csv", summary_rows)
    write_csv(output_dir / "single_paired_bdqn_vs_ddqn.csv", paired_rows)
    write_markdown(output_dir / "single_summary.md", summary_rows)

    print(f"Loaded {len(rows)} unique runs.")
    print(f"Summary: {output_dir / 'single_summary.md'}")


if __name__ == "__main__":
    main()

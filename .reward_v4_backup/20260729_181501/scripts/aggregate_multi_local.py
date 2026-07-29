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
    "simultaneous_sensor_overlap_ratio_mean",
    "knowledge_overlap_ratio_mean",
    "collision_agent_ratio",
    "local_sensor_revisit_ratio",
    "team_new_observed_cells_per_env_step",
    "tracking_progress_ratio",
    "episode_sample_distance_mean",
    "posterior_std_mean",
    "posterior_std_max",
    "posterior_kl_per_parameter",
    "mean_policy_reward_mean",
    "mean_policy_completed_mean",
    "mean_policy_completed_value_mean",
    "mean_policy_team_coverage_ratio_mean",
    "mean_policy_collision_agent_ratio",
    "sampled_shared_reward_mean",
    "sampled_shared_completed_mean",
    "sampled_shared_completed_value_mean",
    "sampled_shared_team_coverage_ratio_mean",
    "sampled_shared_collision_agent_ratio",
    "sampled_independent_reward_mean",
    "sampled_independent_completed_mean",
    "sampled_independent_completed_value_mean",
    "sampled_independent_team_coverage_ratio_mean",
    "sampled_independent_collision_agent_ratio",
    "execution_independent_minus_shared_reward_mean",
    "execution_independent_minus_shared_completed_mean",
    "execution_independent_minus_shared_completed_value_mean",
    "execution_independent_minus_shared_team_coverage_ratio_mean",
    "execution_independent_minus_shared_collision_agent_ratio",
)

# Two-sided Student-t critical values t_(0.975, df).  The experiment uses at most
# a few dozen seeds, so a compact lookup avoids adding scipy to the project.
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
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


def _t_critical(df: int) -> float:
    if df <= 0:
        return float("nan")
    if df in T_975:
        return T_975[df]
    return 1.96


def confidence_intervals(values: list[float]) -> dict[str, float]:
    array = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if array.size == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
            "ci95_z_low": float("nan"),
            "ci95_z_high": float("nan"),
        }
    mean = float(array.mean())
    if array.size == 1:
        return {
            "mean": mean,
            "std": 0.0,
            "ci95_low": mean,
            "ci95_high": mean,
            "ci95_z_low": mean,
            "ci95_z_high": mean,
        }
    std = float(array.std(ddof=1))
    standard_error = std / math.sqrt(array.size)
    t_half = _t_critical(array.size - 1) * standard_error
    z_half = 1.96 * standard_error
    return {
        "mean": mean,
        "std": std,
        "ci95_low": mean - t_half,
        "ci95_high": mean + t_half,
        "ci95_z_low": mean - z_half,
        "ci95_z_high": mean + z_half,
    }


def normalized_metadata(row: dict) -> dict:
    row = dict(row)
    row.setdefault("scenario_label", "deterministic_privileged")
    row.setdefault("global_state_mode", "privileged_truth")
    row.setdefault("detection_probability", 1.0)
    row.setdefault("evaluation_policy", "deterministic")
    if str(row.get("algo", "")).startswith("bayes_qmix"):
        row.setdefault(
            "posterior_sampling",
            "shared" if str(row["algo"]).endswith("shared") else "independent",
        )
    else:
        row.setdefault("posterior_sampling", "none")
    return row


def load_runs(roots: list[Path]) -> list[dict]:
    rows: list[dict] = []
    seen: set[tuple] = set()
    for root in roots:
        for path in sorted(root.rglob("*_eval.json")):
            try:
                with path.open() as handle:
                    row = normalized_metadata(json.load(handle))
            except (OSError, json.JSONDecodeError):
                continue
            if "algo" not in row or "seed" not in row:
                continue
            key = (
                str(row["scenario_label"]),
                str(row["global_state_mode"]),
                str(row["algo"]),
                int(row["seed"]),
            )
            if key in seen:
                continue
            seen.add(key)
            row["source_file"] = str(path)
            rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def group_identity(row: dict) -> tuple[str, str, str]:
    return (
        str(row["scenario_label"]),
        str(row["global_state_mode"]),
        str(row["algo"]),
    )


def aggregate(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[group_identity(row)].append(row)
    output = []
    for (scenario, state_mode, algo), group in sorted(grouped.items()):
        summary: dict[str, object] = {
            "scenario_label": scenario,
            "global_state_mode": state_mode,
            "algo": algo,
            "posterior_sampling": str(group[0].get("posterior_sampling", "none")),
            "detection_probability": float(group[0].get("detection_probability", 1.0)),
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
            statistics = confidence_intervals(values)
            for name, value in statistics.items():
                summary[f"{metric}_{name}"] = value
        output.append(summary)
    return output


def paired(rows: list[dict], first: str, second: str) -> list[dict]:
    by_context: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        by_context[(str(row["scenario_label"]), str(row["global_state_mode"]))].append(row)

    results: list[dict] = []
    for (scenario, state_mode), context_rows in sorted(by_context.items()):
        index = {
            (str(row["algo"]), int(row["seed"])): row for row in context_rows
        }
        seeds = sorted(
            {seed for algo, seed in index if algo == first}
            & {seed for algo, seed in index if algo == second}
        )
        if not seeds:
            continue
        result: dict[str, object] = {
            "scenario_label": scenario,
            "global_state_mode": state_mode,
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
            if not differences:
                continue
            statistics = confidence_intervals(differences)
            for name, value in statistics.items():
                result[f"{metric}_difference_{name}"] = value
        results.append(result)
    return results


def paired_long(rows: list[dict], comparisons: list[tuple[str, str]]) -> list[dict]:
    output: list[dict] = []
    for first, second in comparisons:
        for wide in paired(rows, first, second):
            for metric in PRIMARY_METRICS:
                key = f"{metric}_difference_mean"
                if key not in wide:
                    continue
                output.append(
                    {
                        "scenario_label": wide["scenario_label"],
                        "global_state_mode": wide["global_state_mode"],
                        "comparison": wide["comparison"],
                        "metric": metric,
                        "difference_mean": wide[key],
                        "difference_std": wide.get(f"{metric}_difference_std"),
                        "ci95_t_low": wide.get(f"{metric}_difference_ci95_low"),
                        "ci95_t_high": wide.get(f"{metric}_difference_ci95_high"),
                        "ci95_z_low": wide.get(f"{metric}_difference_ci95_z_low"),
                        "ci95_z_high": wide.get(f"{metric}_difference_ci95_z_high"),
                        "n_pairs": wide["n_pairs"],
                        "seeds": wide["seeds"],
                    }
                )
    return output


def write_markdown(path: Path, summary: list[dict]) -> None:
    columns = [
        "scenario_label",
        "global_state_mode",
        "algo",
        "posterior_sampling",
        "n_seeds",
        "completed_mean_mean",
        "completed_value_mean_mean",
        "detected_mean_mean",
        "team_coverage_ratio_mean_mean",
        "coverage_overlap_ratio_mean_mean",
        "simultaneous_sensor_overlap_ratio_mean_mean",
        "collision_agent_ratio_mean",
        "reward_mean_mean",
    ]
    lines = [
        "# Multi-UAV local-memory and Bayesian-QMIX summary",
        "",
        "Intervals in CSV use Student-t as the primary 95% interval; z=1.96 intervals are retained separately.",
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
    parser.add_argument("--input-root", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    roots = [Path(root) for root in args.input_root]
    rows = load_runs(roots)
    if not rows:
        raise RuntimeError(f"No evaluation JSON below {roots}")

    output = Path(args.output_dir)
    summary = aggregate(rows)
    comparisons = [
        ("qmix_ddqn", "shared_ddqn"),
        ("shared_bdqn", "shared_ddqn"),
        ("bayes_qmix_shared", "bayes_qmix_independent"),
        ("bayes_qmix_shared", "qmix_ddqn"),
        ("bayes_qmix_independent", "qmix_ddqn"),
    ]

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
    write_csv(
        output / "paired_bayes_shared_vs_independent.csv",
        paired(rows, "bayes_qmix_shared", "bayes_qmix_independent"),
    )
    write_csv(
        output / "paired_bayes_shared_vs_qmix.csv",
        paired(rows, "bayes_qmix_shared", "qmix_ddqn"),
    )
    write_csv(
        output / "paired_bayes_independent_vs_qmix.csv",
        paired(rows, "bayes_qmix_independent", "qmix_ddqn"),
    )
    write_csv(output / "paired_all_long.csv", paired_long(rows, comparisons))
    write_markdown(output / "multi_summary.md", summary)
    print(f"Aggregated {len(rows)} runs from {len(roots)} root(s) into {output}")


if __name__ == "__main__":
    main()

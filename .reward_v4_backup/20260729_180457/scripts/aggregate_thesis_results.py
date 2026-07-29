from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from uav_search_belief20.experiments.thesis_automation import (
    flatten_summary_record,
    is_finite_number,
    load_json,
    mean_std_ci95,
    write_csv,
)


PRIMARY_METRICS = [
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
    "episode_length_mean",
    "first_detection_step_mean",
    "first_completion_step_mean",
    "had_detection_mean",
    "had_completion_mean",
    "detected_to_completed_ratio",
    "stay_ratio",
    "boundary_hit_ratio",
    "revisit_ratio",
    "sensor_revisit_ratio",
    "new_observed_cells_per_step",
    "tracking_progress_ratio",
    "collision_agent_ratio",
    "local_sensor_revisit_ratio",
    "local_new_observed_cells_per_decision",
    "team_new_observed_cells_per_env_step",
    "simultaneous_sensor_overlap_ratio_mean",
]


# (label, scope, left_algo, left_mode, right_algo, right_mode)
COMPARISONS = [
    ("bdqn_mean_minus_ddqn", "single", "bdqn", "posterior_mean", "ddqn", "deterministic"),
    ("bdqn_sample_minus_ddqn", "single", "bdqn", "posterior_sample", "ddqn", "deterministic"),
    (
        "shared_bdqn_mean_minus_shared_ddqn",
        "multi",
        "shared_bdqn",
        "posterior_mean",
        "shared_ddqn",
        "deterministic",
    ),
    (
        "shared_bdqn_sample_minus_shared_ddqn",
        "multi",
        "shared_bdqn",
        "posterior_sample_shared",
        "shared_ddqn",
        "deterministic",
    ),
    (
        "qmix_minus_shared_ddqn",
        "multi",
        "qmix_ddqn",
        "deterministic",
        "shared_ddqn",
        "deterministic",
    ),
    (
        "bayes_qmix_shared_mean_minus_qmix",
        "multi",
        "bayes_qmix_shared",
        "posterior_mean",
        "qmix_ddqn",
        "deterministic",
    ),
    (
        "bayes_qmix_shared_sample_minus_qmix",
        "multi",
        "bayes_qmix_shared",
        "posterior_sample_shared",
        "qmix_ddqn",
        "deterministic",
    ),
    (
        "bayes_qmix_independent_mean_minus_qmix",
        "multi",
        "bayes_qmix_independent",
        "posterior_mean",
        "qmix_ddqn",
        "deterministic",
    ),
    (
        "bayes_qmix_independent_sample_minus_qmix",
        "multi",
        "bayes_qmix_independent",
        "posterior_sample_independent",
        "qmix_ddqn",
        "deterministic",
    ),
    (
        "independent_sampling_minus_shared_sampling",
        "multi",
        "bayes_qmix_independent",
        "posterior_sample_independent",
        "bayes_qmix_shared",
        "posterior_sample_shared",
    ),
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate corrected thesis evaluations and learning curves.")
    parser.add_argument("--run-root", default="runs/thesis_v2")
    parser.add_argument("--baseline-root", default="logs/thesis_v2/baselines")
    parser.add_argument("--output-dir", default="logs/thesis_v2/aggregate")
    parser.add_argument("--did-low-probability", type=float, default=0.7)
    parser.add_argument("--did-high-probability", type=float, default=1.0)
    return parser


def _discover_summaries(root: Path) -> list[dict]:
    records: list[dict] = []
    if not root.exists():
        return records
    for path in sorted(root.glob("**/*_summary.json")):
        try:
            payload = load_json(path)
        except Exception:
            continue
        if not {"scope", "algo", "policy_mode", "summary"}.issubset(payload):
            continue
        payload["summary_path"] = str(path)
        records.append(payload)
    return records


def _method_summary(records: list[dict]) -> list[dict[str, object]]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for record in records:
        key = (
            record.get("scope"),
            record.get("algo"),
            record.get("policy_mode"),
            float(record.get("detection_probability", float("nan"))),
            record.get("reward_mode"),
            record.get("global_state_mode"),
            bool(record.get("is_baseline", False)),
        )
        groups[key].append(record)

    output: list[dict[str, object]] = []
    for key, group in sorted(groups.items(), key=lambda item: str(item[0])):
        scope, algo, policy_mode, probability, reward_mode, state_mode, is_baseline = key
        row: dict[str, object] = {
            "scope": scope,
            "algo": algo,
            "policy_mode": policy_mode,
            "detection_probability": probability,
            "reward_mode": reward_mode,
            "global_state_mode": state_mode,
            "is_baseline": is_baseline,
            "n_training_seeds": len({record.get("training_seed") for record in group}),
            "training_seeds": ",".join(
                str(seed) for seed in sorted({int(record.get("training_seed", -1)) for record in group})
            ),
            "uncertainty_unit": "evaluation_episode" if is_baseline else "training_seed",
        }
        for metric in PRIMARY_METRICS:
            values = [record.get("summary", {}).get(metric) for record in group]
            finite = [float(value) for value in values if is_finite_number(value)]
            if not finite:
                continue
            stats = mean_std_ci95(finite)
            # A baseline generally has one policy seed. Reuse its episode-level CI
            # instead of pretending there is training-seed uncertainty.
            if is_baseline and len(group) == 1:
                summary = group[0]["summary"]
                row[f"{metric}_mean"] = finite[0]
                row[f"{metric}_std"] = summary.get(
                    f"{metric.removesuffix('_mean')}_std", float("nan")
                )
                row[f"{metric}_ci95_low"] = summary.get(
                    f"{metric.removesuffix('_mean')}_ci95_low", float("nan")
                )
                row[f"{metric}_ci95_high"] = summary.get(
                    f"{metric.removesuffix('_mean')}_ci95_high", float("nan")
                )
            else:
                row[f"{metric}_mean"] = stats["mean"]
                row[f"{metric}_std"] = stats["std"]
                row[f"{metric}_ci95_low"] = stats["ci95_low"]
                row[f"{metric}_ci95_high"] = stats["ci95_high"]
        output.append(row)
    return output


def _record_index(records: list[dict]) -> dict[tuple, dict]:
    output = {}
    for record in records:
        if record.get("is_baseline"):
            continue
        key = (
            record.get("scope"),
            record.get("algo"),
            record.get("policy_mode"),
            float(record.get("detection_probability")),
            int(record.get("training_seed")),
        )
        output[key] = record
    return output


def _paired(records: list[dict]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    index = _record_index(records)
    by_seed: list[dict[str, object]] = []

    probabilities = sorted({key[3] for key in index})
    seeds = sorted({key[4] for key in index})
    for label, scope, left_algo, left_mode, right_algo, right_mode in COMPARISONS:
        for probability in probabilities:
            for seed in seeds:
                left = index.get((scope, left_algo, left_mode, probability, seed))
                right = index.get((scope, right_algo, right_mode, probability, seed))
                if left is None or right is None:
                    continue
                row: dict[str, object] = {
                    "comparison": label,
                    "scope": scope,
                    "detection_probability": probability,
                    "training_seed": seed,
                    "left_algo": left_algo,
                    "left_policy_mode": left_mode,
                    "right_algo": right_algo,
                    "right_policy_mode": right_mode,
                }
                for metric in PRIMARY_METRICS:
                    left_value = left["summary"].get(metric)
                    right_value = right["summary"].get(metric)
                    if is_finite_number(left_value) and is_finite_number(right_value):
                        row[f"{metric}_difference"] = float(left_value) - float(right_value)
                by_seed.append(row)

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in by_seed:
        groups[(row["comparison"], row["scope"], row["detection_probability"])].append(row)

    summary_rows: list[dict[str, object]] = []
    for (label, scope, probability), group in sorted(groups.items()):
        row: dict[str, object] = {
            "comparison": label,
            "scope": scope,
            "detection_probability": probability,
            "n_pairs": len(group),
            "training_seeds": ",".join(str(int(item["training_seed"])) for item in group),
        }
        for metric in PRIMARY_METRICS:
            key = f"{metric}_difference"
            values = [item.get(key) for item in group if is_finite_number(item.get(key))]
            if not values:
                continue
            stats = mean_std_ci95(values)
            row[f"{key}_mean"] = stats["mean"]
            row[f"{key}_std"] = stats["std"]
            row[f"{key}_ci95_low"] = stats["ci95_low"]
            row[f"{key}_ci95_high"] = stats["ci95_high"]
        summary_rows.append(row)
    return by_seed, summary_rows


def _difference_in_differences(
    paired_by_seed: list[dict[str, object]],
    *,
    low_probability: float,
    high_probability: float,
) -> list[dict[str, object]]:
    index = {
        (
            row["comparison"],
            int(row["training_seed"]),
            float(row["detection_probability"]),
        ): row
        for row in paired_by_seed
    }
    comparisons = sorted({str(row["comparison"]) for row in paired_by_seed})
    seeds = sorted({int(row["training_seed"]) for row in paired_by_seed})
    output: list[dict[str, object]] = []

    for comparison in comparisons:
        per_seed: list[dict[str, object]] = []
        for seed in seeds:
            low = index.get((comparison, seed, float(low_probability)))
            high = index.get((comparison, seed, float(high_probability)))
            if low is None or high is None:
                continue
            row = {"comparison": comparison, "training_seed": seed}
            for metric in PRIMARY_METRICS:
                key = f"{metric}_difference"
                if is_finite_number(low.get(key)) and is_finite_number(high.get(key)):
                    row[f"{metric}_did"] = float(low[key]) - float(high[key])
            per_seed.append(row)
        if not per_seed:
            continue
        summary: dict[str, object] = {
            "comparison": comparison,
            "low_probability": low_probability,
            "high_probability": high_probability,
            "n_pairs": len(per_seed),
            "training_seeds": ",".join(str(int(row["training_seed"])) for row in per_seed),
        }
        for metric in PRIMARY_METRICS:
            key = f"{metric}_did"
            values = [row.get(key) for row in per_seed if is_finite_number(row.get(key))]
            if not values:
                continue
            stats = mean_std_ci95(values)
            summary[f"{key}_mean"] = stats["mean"]
            summary[f"{key}_std"] = stats["std"]
            summary[f"{key}_ci95_low"] = stats["ci95_low"]
            summary[f"{key}_ci95_high"] = stats["ci95_high"]
        output.append(summary)
    return output


def _learning_curves(run_root: Path) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    if not run_root.exists():
        return output
    for metrics_path in sorted(run_root.glob("**/metrics.csv")):
        config_path = metrics_path.parent / "run_config.json"
        if not config_path.exists():
            continue
        try:
            config = load_json(config_path)
        except Exception:
            continue
        algo = str(config.get("algo", ""))
        scope = "multi" if algo.startswith(("shared_", "qmix_", "bayes_qmix_")) else "single"
        with metrics_path.open(newline="") as handle:
            for raw in csv.DictReader(handle):
                row: dict[str, object] = {
                    "scope": scope,
                    "algo": algo,
                    "training_seed": int(config.get("seed", -1)),
                    "detection_probability": float(config.get("detection_probability", float("nan"))),
                    "reward_mode": config.get("reward_mode", "unknown"),
                    "global_state_mode": config.get("global_state_mode", ""),
                    "run_dir": str(metrics_path.parent),
                }
                for key, value in raw.items():
                    if value is None or value == "":
                        row[key] = value
                    elif is_finite_number(value):
                        row[key] = float(value)
                    else:
                        row[key] = value
                output.append(row)
    return output


def _write_markdown(path: Path, method_rows: list[dict], paired_rows: list[dict]) -> None:
    lines = [
        "# Résumé agrégé de l'étude thesis-v2",
        "",
        "Les intervalles des méthodes apprises sont des IC Student-t à 95 % sur les seeds d'entraînement.",
        "Les intervalles des baselines sont calculés sur les mondes d'évaluation.",
        "",
        "## Méthodes",
        "",
        "| Scope | pD | Méthode | Politique | Seeds | Reward | Completed | Coverage | Collision |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in method_rows:
        coverage = row.get("team_coverage_ratio_mean_mean", row.get("sensor_coverage_ratio_mean_mean", float("nan")))
        lines.append(
            "| {scope} | {p:.2f} | {algo} | {mode} | {n} | {reward:.3f} | {completed:.3f} | {coverage:.3f} | {collision:.3f} |".format(
                scope=row.get("scope", ""),
                p=float(row.get("detection_probability", float("nan"))),
                algo=row.get("algo", ""),
                mode=row.get("policy_mode", ""),
                n=int(row.get("n_training_seeds", 0)),
                reward=float(row.get("reward_mean_mean", float("nan"))),
                completed=float(row.get("completed_mean_mean", float("nan"))),
                coverage=float(coverage),
                collision=float(row.get("collision_agent_ratio_mean", float("nan"))),
            )
        )
    lines.extend(
        [
            "",
            "## Comparaisons appariées",
            "",
            "| pD | Comparaison | Paires | Δ reward | Δ completed | Δ coverage | Δ collision |",
            "|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in paired_rows:
        coverage = row.get(
            "team_coverage_ratio_mean_difference_mean",
            row.get("sensor_coverage_ratio_mean_difference_mean", float("nan")),
        )
        lines.append(
            "| {p:.2f} | {comparison} | {n} | {reward:.3f} | {completed:.3f} | {coverage:.3f} | {collision:.3f} |".format(
                p=float(row.get("detection_probability", float("nan"))),
                comparison=row.get("comparison", ""),
                n=int(row.get("n_pairs", 0)),
                reward=float(row.get("reward_mean_difference_mean", float("nan"))),
                completed=float(row.get("completed_mean_difference_mean", float("nan"))),
                coverage=float(coverage),
                collision=float(row.get("collision_agent_ratio_difference_mean", float("nan"))),
            )
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = build_parser().parse_args()
    run_root = Path(args.run_root)
    baseline_root = Path(args.baseline_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = _discover_summaries(run_root) + _discover_summaries(baseline_root)
    flat_records = [flatten_summary_record(record) for record in records]
    write_csv(output_dir / "all_runs.csv", flat_records)

    method_rows = _method_summary(records)
    write_csv(output_dir / "summary_by_method.csv", method_rows)

    paired_by_seed, paired_summary = _paired(records)
    write_csv(output_dir / "paired_by_seed.csv", paired_by_seed)
    write_csv(output_dir / "paired_summary.csv", paired_summary)

    did = _difference_in_differences(
        paired_by_seed,
        low_probability=args.did_low_probability,
        high_probability=args.did_high_probability,
    )
    write_csv(output_dir / "difference_in_differences.csv", did)

    learning_rows = _learning_curves(run_root)
    write_csv(output_dir / "learning_curves.csv", learning_rows)
    _write_markdown(output_dir / "summary.md", method_rows, paired_summary)

    print(f"Discovered {len(records)} evaluation summaries.")
    print(f"Saved aggregate files in {output_dir}")


if __name__ == "__main__":
    main()

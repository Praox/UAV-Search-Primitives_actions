from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


METRICS = (
    "reward_mean",
    "detected_mean",
    "completed_mean",
    "detected_value_mean",
    "completed_value_mean",
    "detected_to_completed_ratio",
    "sensor_coverage_ratio_mean",
    "team_coverage_ratio_mean",
    "mean_local_coverage_ratio_mean",
    "coverage_overlap_ratio_mean",
    "simultaneous_sensor_overlap_ratio_mean",
    "knowledge_overlap_ratio_mean",
    "collision_agent_ratio",
    "stay_ratio",
    "sensor_revisit_ratio",
    "local_sensor_revisit_ratio",
    "new_observed_cells_per_step",
    "team_new_observed_cells_per_env_step",
    "tracking_progress_ratio",
    "first_detection_step_mean",
    "first_completion_step_mean",
    "posterior_std_mean",
    "posterior_std_max",
    "posterior_kl_per_parameter",
)

T_975 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
    7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
    13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110,
    18: 2.101, 19: 2.093, 20: 2.086, 24: 2.064, 30: 2.042,
}

SINGLE_COMPARISONS = (
    ("bdqn", "posterior_mean", "ddqn", "deterministic", "single_bdqn_mean_minus_ddqn"),
    ("bdqn", "posterior_sample_per_episode", "ddqn", "deterministic", "single_bdqn_sample_minus_ddqn"),
)
MULTI_COMPARISONS = (
    ("shared_bdqn", None, "shared_ddqn", None, "shared_bdqn_minus_shared_ddqn"),
    ("qmix_ddqn", None, "shared_ddqn", None, "qmix_ddqn_minus_shared_ddqn"),
    ("bayes_qmix_shared", None, "qmix_ddqn", None, "bayes_qmix_shared_minus_qmix_ddqn"),
    ("bayes_qmix_independent", None, "qmix_ddqn", None, "bayes_qmix_independent_minus_qmix_ddqn"),
    ("bayes_qmix_shared", None, "bayes_qmix_independent", None, "bayes_shared_minus_independent"),
)


def t_critical(df: int) -> float:
    if df <= 0:
        return float("nan")
    if df in T_975:
        return T_975[df]
    smaller = [value for value in T_975 if value <= df]
    return T_975[max(smaller)] if smaller else 1.96


def stats(values: list[float]) -> dict[str, float]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {
            "mean": float("nan"), "std": float("nan"),
            "ci95_low": float("nan"), "ci95_high": float("nan"),
        }
    mean = statistics.fmean(finite)
    if len(finite) == 1:
        return {"mean": mean, "std": 0.0, "ci95_low": mean, "ci95_high": mean}
    std = statistics.stdev(finite)
    half = t_critical(len(finite) - 1) * std / math.sqrt(len(finite))
    return {"mean": mean, "std": std, "ci95_low": mean - half, "ci95_high": mean + half}


def parse_probability_from_path(path: Path) -> float:
    for part in path.parts:
        if part.startswith("pdet_"):
            text = part.split("__", 1)[0].removeprefix("pdet_").replace("p", ".")
            try:
                return float(text)
            except ValueError:
                pass
    return 1.0


def infer_scope(path: Path, row: dict) -> str:
    if row.get("scope") in {"single", "multi"}:
        return str(row["scope"])
    parts = set(path.parts)
    return "multi" if "multi" in parts or "multi_local" in parts else "single"


def normalize(path: Path, payload: dict) -> dict:
    row = dict(payload)
    scope = infer_scope(path, row)
    probability = float(row.get("detection_probability", parse_probability_from_path(path)))
    row["scope"] = scope
    row["detection_probability"] = probability
    row.setdefault("scenario_label", f"pdet_{probability:.2f}".replace(".", "p"))
    row.setdefault("global_state_mode", "not_applicable" if scope == "single" else "privileged_truth")
    row.setdefault("evaluation_mode", row.get("evaluation_policy", "deterministic"))
    row.setdefault("evaluation_policy", row.get("evaluation_mode", "deterministic"))
    row.setdefault("seed", -1)
    row.setdefault("algo", "unknown")
    row["source_file"] = str(path)
    return row


def load_rows(roots: list[Path]) -> list[dict]:
    rows: list[dict] = []
    seen: set[tuple] = set()
    for root in roots:
        if not root.exists():
            print(f"Warning: missing root {root}")
            continue
        for path in sorted(root.rglob("*.json")):
            if "eval" not in path.name:
                continue
            try:
                payload = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(payload, dict) or "completed_mean" not in payload:
                continue
            row = normalize(path, payload)
            key = (
                row["scope"], row["scenario_label"], row["global_state_mode"],
                row["algo"], row["evaluation_policy"], int(row["seed"]),
            )
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields = sorted({key for row in rows for key in row if key != "action_counts"})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict]) -> list[dict]:
    grouped: defaultdict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(
            str(row["scope"]), str(row["scenario_label"]),
            str(row["global_state_mode"]), str(row["algo"]),
            str(row["evaluation_policy"]), float(row["detection_probability"]),
        )].append(row)

    output: list[dict] = []
    for key, group in sorted(grouped.items()):
        scope, scenario, state_mode, algo, evaluation_policy, probability = key
        result: dict[str, object] = {
            "scope": scope,
            "scenario_label": scenario,
            "global_state_mode": state_mode,
            "algo": algo,
            "evaluation_policy": evaluation_policy,
            "detection_probability": probability,
            "n_runs": len(group),
            "seeds": ",".join(str(int(row["seed"])) for row in sorted(group, key=lambda item: int(item["seed"]))),
        }
        for metric in METRICS:
            values = [float(row[metric]) for row in group if metric in row and row[metric] is not None]
            if not values:
                continue
            for name, value in stats(values).items():
                result[f"{metric}_{name}"] = value
        output.append(result)
    return output


def _select_row(
    context_rows: list[dict],
    *,
    algo: str,
    seed: int,
    evaluation_policy: str | None,
) -> dict | None:
    candidates = [
        row for row in context_rows
        if str(row["algo"]) == algo and int(row["seed"]) == int(seed)
    ]
    if evaluation_policy is not None:
        candidates = [
            row for row in candidates
            if str(row["evaluation_policy"]) == evaluation_policy
        ]
    if not candidates:
        return None
    if len(candidates) > 1 and evaluation_policy is None:
        # Multi-agent files have one primary row per method. Should an extra file be
        # present, prefer the non-mean primary sampled/deterministic evaluation.
        candidates.sort(
            key=lambda row: (
                str(row["evaluation_policy"]) == "posterior_mean",
                str(row["source_file"]),
            )
        )
    return candidates[0]


def paired(rows: list[dict]) -> list[dict]:
    output: list[dict] = []
    contexts: defaultdict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        contexts[(
            str(row["scope"]), str(row["scenario_label"]),
            str(row["global_state_mode"]), float(row["detection_probability"]),
        )].append(row)

    for context, context_rows in sorted(contexts.items()):
        scope, scenario, state_mode, probability = context
        comparisons = SINGLE_COMPARISONS if scope == "single" else MULTI_COMPARISONS
        all_seeds = sorted({int(row["seed"]) for row in context_rows})
        for first, first_policy, second, second_policy, label in comparisons:
            common = [
                seed for seed in all_seeds
                if _select_row(context_rows, algo=first, seed=seed, evaluation_policy=first_policy) is not None
                and _select_row(context_rows, algo=second, seed=seed, evaluation_policy=second_policy) is not None
            ]
            if not common:
                continue
            result: dict[str, object] = {
                "scope": scope,
                "scenario_label": scenario,
                "global_state_mode": state_mode,
                "detection_probability": probability,
                "first_evaluation_policy": first_policy or "primary",
                "second_evaluation_policy": second_policy or "primary",
                "comparison": label,
                "first": first,
                "second": second,
                "n_pairs": len(common),
                "seeds": ",".join(map(str, common)),
            }
            for metric in METRICS:
                differences = []
                for seed in common:
                    left = _select_row(context_rows, algo=first, seed=seed, evaluation_policy=first_policy)
                    right = _select_row(context_rows, algo=second, seed=seed, evaluation_policy=second_policy)
                    if left is not None and right is not None and metric in left and metric in right:
                        differences.append(float(left[metric]) - float(right[metric]))
                if not differences:
                    continue
                for name, value in stats(differences).items():
                    result[f"{metric}_difference_{name}"] = value
            output.append(result)
    return output


def difference_in_differences(rows: list[dict], reference_probability: float = 1.0) -> list[dict]:
    output: list[dict] = []
    by_scope_state: defaultdict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        by_scope_state[(str(row["scope"]), str(row["global_state_mode"]))].append(row)

    for context, context_rows in sorted(by_scope_state.items()):
        scope, state_mode = context
        comparisons = SINGLE_COMPARISONS if scope == "single" else MULTI_COMPARISONS
        probabilities = sorted({float(row["detection_probability"]) for row in context_rows})
        all_seeds = sorted({int(row["seed"]) for row in context_rows})
        for target_probability in probabilities:
            if math.isclose(target_probability, reference_probability):
                continue
            noisy_rows = [row for row in context_rows if math.isclose(float(row["detection_probability"]), target_probability)]
            reference_rows = [row for row in context_rows if math.isclose(float(row["detection_probability"]), reference_probability)]
            for first, first_policy, second, second_policy, label in comparisons:
                seeds = [
                    seed for seed in all_seeds
                    if _select_row(noisy_rows, algo=first, seed=seed, evaluation_policy=first_policy) is not None
                    and _select_row(noisy_rows, algo=second, seed=seed, evaluation_policy=second_policy) is not None
                    and _select_row(reference_rows, algo=first, seed=seed, evaluation_policy=first_policy) is not None
                    and _select_row(reference_rows, algo=second, seed=seed, evaluation_policy=second_policy) is not None
                ]
                if not seeds:
                    continue
                result: dict[str, object] = {
                    "scope": scope,
                    "global_state_mode": state_mode,
                    "comparison": label,
                    "first_evaluation_policy": first_policy or "primary",
                    "second_evaluation_policy": second_policy or "primary",
                    "reference_detection_probability": reference_probability,
                    "target_detection_probability": target_probability,
                    "n_pairs": len(seeds),
                    "seeds": ",".join(map(str, seeds)),
                }
                for metric in METRICS:
                    values = []
                    for seed in seeds:
                        noisy_first = _select_row(noisy_rows, algo=first, seed=seed, evaluation_policy=first_policy)
                        noisy_second = _select_row(noisy_rows, algo=second, seed=seed, evaluation_policy=second_policy)
                        ref_first = _select_row(reference_rows, algo=first, seed=seed, evaluation_policy=first_policy)
                        ref_second = _select_row(reference_rows, algo=second, seed=seed, evaluation_policy=second_policy)
                        if not all(item is not None and metric in item for item in (noisy_first, noisy_second, ref_first, ref_second)):
                            continue
                        noisy_advantage = float(noisy_first[metric]) - float(noisy_second[metric])
                        reference_advantage = float(ref_first[metric]) - float(ref_second[metric])
                        values.append(noisy_advantage - reference_advantage)
                    if not values:
                        continue
                    for name, value in stats(values).items():
                        result[f"{metric}_interaction_{name}"] = value
                output.append(result)
    return output


def write_markdown(path: Path, summary: list[dict], paired_rows: list[dict], interaction_rows: list[dict]) -> None:
    lines = [
        "# Uncertainty study summary",
        "",
        "Primary intervals are two-sided Student-t 95% intervals across training seeds.",
        "Heuristic baselines usually have one policy seed and should be treated as absolute references, not training-seed confidence intervals.",
        "",
        "## Method summaries",
        "",
        "| Scope | p(det) | Algorithm | Evaluation | Runs | Completed | Coverage | Collisions | Reward |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        coverage = row.get("team_coverage_ratio_mean_mean", row.get("sensor_coverage_ratio_mean_mean", float("nan")))
        lines.append(
            f"| {row['scope']} | {float(row['detection_probability']):.2f} | {row['algo']} | {row['evaluation_policy']} | "
            f"{row['n_runs']} | {float(row.get('completed_mean_mean', float('nan'))):.3f} | "
            f"{float(coverage):.3f} | {float(row.get('collision_agent_ratio_mean', float('nan'))):.3f} | "
            f"{float(row.get('reward_mean_mean', float('nan'))):.3f} |"
        )
    lines.extend(["", "## Paired comparisons", "", f"Rows: {len(paired_rows)}", "", "## Difference-in-differences", "", f"Rows: {len(interaction_rows)}"])
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reference-detection-probability", type=float, default=1.0)
    args = parser.parse_args()

    roots = [Path(root) for root in args.roots]
    rows = load_rows(roots)
    if not rows:
        raise RuntimeError(f"No evaluation JSON found below {roots}")

    output = Path(args.output_dir)
    summary_rows = summarize(rows)
    paired_rows = paired(rows)
    interaction_rows = difference_in_differences(
        rows, reference_probability=float(args.reference_detection_probability)
    )

    write_csv(output / "uncertainty_runs.csv", rows)
    write_csv(output / "uncertainty_summary.csv", summary_rows)
    write_csv(output / "uncertainty_paired.csv", paired_rows)
    write_csv(output / "uncertainty_difference_in_differences.csv", interaction_rows)
    write_markdown(output / "uncertainty_summary.md", summary_rows, paired_rows, interaction_rows)
    print(f"Aggregated {len(rows)} evaluation files into {output}")


if __name__ == "__main__":
    main()

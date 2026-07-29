from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - dependency error path
    raise SystemExit("matplotlib is required. Run: pip install -r requirements-thesis.txt") from exc
import numpy as np

from uav_search_belief20.experiments.thesis_automation import (
    is_finite_number,
    mean_std_ci95,
)


LEARNING_METRICS = {
    "single": [
        ("validation_reward", "Validation reward"),
        ("validation_completed", "Completed targets"),
        ("validation_completed_value", "Completed target value"),
        ("validation_coverage", "Sensor coverage ratio"),
        ("loss", "Training loss"),
        ("td_residual_variance", "TD residual variance"),
    ],
    "multi": [
        ("validation_reward", "Validation reward"),
        ("validation_completed", "Completed targets"),
        ("validation_completed_value", "Completed target value"),
        ("validation_coverage", "Team coverage ratio"),
        ("validation_overlap", "Coverage overlap ratio"),
        ("validation_collision", "Collision-agent ratio"),
        ("loss", "Training loss"),
        ("posterior_std_mean", "Posterior std mean"),
    ],
}

FINAL_METRICS = {
    "single": [
        ("reward_mean_mean", "Evaluation reward"),
        ("completed_mean_mean", "Completed targets"),
        ("completed_value_mean_mean", "Completed target value"),
        ("sensor_coverage_ratio_mean_mean", "Sensor coverage ratio"),
    ],
    "multi": [
        ("reward_mean_mean", "Evaluation reward"),
        ("completed_mean_mean", "Completed targets"),
        ("completed_value_mean_mean", "Completed target value"),
        ("team_coverage_ratio_mean_mean", "Team coverage ratio"),
        ("collision_agent_ratio_mean", "Collision-agent ratio"),
        ("coverage_overlap_ratio_mean_mean", "Coverage overlap ratio"),
    ],
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create thesis learning and final-performance plots.")
    parser.add_argument("--aggregate-dir", default="logs/thesis_v2/aggregate")
    parser.add_argument("--output-dir", default="logs/thesis_v2/plots")
    parser.add_argument("--dpi", type=int, default=180)
    return parser


def _read(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _slug(value: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in value).strip("_").lower()


def _save(fig, output_dir: Path, stem: str, dpi: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_dir / f"{stem}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def _learning_plots(rows: list[dict[str, str]], output_dir: Path, dpi: int) -> None:
    if not rows:
        return
    scopes = sorted({row.get("scope", "") for row in rows})
    probabilities = sorted(
        {float(row["detection_probability"]) for row in rows if is_finite_number(row.get("detection_probability"))}
    )

    for scope in scopes:
        if scope not in LEARNING_METRICS:
            continue
        for probability in probabilities:
            selected = [
                row
                for row in rows
                if row.get("scope") == scope
                and is_finite_number(row.get("detection_probability"))
                and math.isclose(float(row["detection_probability"]), probability)
            ]
            algorithms = sorted({row.get("algo", "") for row in selected})
            for metric, ylabel in LEARNING_METRICS[scope]:
                if not any(is_finite_number(row.get(metric)) for row in selected):
                    continue
                fig, ax = plt.subplots(figsize=(10, 6))
                for algo in algorithms:
                    algo_rows = [row for row in selected if row.get("algo") == algo]
                    by_seed: dict[int, list[tuple[float, float]]] = defaultdict(list)
                    for row in algo_rows:
                        if not is_finite_number(row.get("episode")) or not is_finite_number(row.get(metric)):
                            continue
                        seed = int(float(row.get("training_seed", -1)))
                        by_seed[seed].append((float(row["episode"]), float(row[metric])))
                    if not by_seed:
                        continue

                    for points in by_seed.values():
                        points.sort()

                    episodes = sorted({episode for points in by_seed.values() for episode, _ in points})
                    means, lows, highs, valid_x = [], [], [], []
                    for episode in episodes:
                        values = []
                        for points in by_seed.values():
                            mapping = {x: y for x, y in points}
                            if episode in mapping:
                                values.append(mapping[episode])
                        if not values:
                            continue
                        stats = mean_std_ci95(values)
                        valid_x.append(episode)
                        means.append(float(stats["mean"]))
                        lows.append(float(stats["ci95_low"]))
                        highs.append(float(stats["ci95_high"]))
                    line, = ax.plot(
                        valid_x, means, linewidth=2.8, label=algo, zorder=3
                    )
                    for seed, points in sorted(by_seed.items()):
                        x = [item[0] for item in points]
                        y = [item[1] for item in points]
                        ax.plot(
                            x,
                            y,
                            linewidth=0.8,
                            alpha=0.20,
                            color=line.get_color(),
                            zorder=1,
                        )
                    if len(by_seed) >= 2 and all(np.isfinite(lows)) and all(np.isfinite(highs)):
                        ax.fill_between(
                            valid_x,
                            lows,
                            highs,
                            alpha=0.18,
                            color=line.get_color(),
                            zorder=2,
                        )

                ax.set_title(f"{scope} — pD={probability:.2f} — {ylabel}")
                ax.set_xlabel("Training episode")
                ax.set_ylabel(ylabel)
                ax.grid(alpha=0.25)
                ax.legend()
                _save(
                    fig,
                    output_dir / "learning_curves" / scope / f"pdet_{probability:.2f}".replace(".", "p"),
                    _slug(metric),
                    dpi,
                )


def _final_plots(rows: list[dict[str, str]], output_dir: Path, dpi: int) -> None:
    if not rows:
        return
    for scope, metrics in FINAL_METRICS.items():
        scope_rows = [row for row in rows if row.get("scope") == scope]
        if not scope_rows:
            continue
        for metric, ylabel in metrics:
            available = [row for row in scope_rows if is_finite_number(row.get(metric))]
            if not available:
                continue
            fig, ax = plt.subplots(figsize=(10, 6))
            methods = sorted(
                {
                    (row.get("algo", ""), row.get("policy_mode", ""))
                    for row in available
                }
            )
            for algo, mode in methods:
                method_rows = [
                    row
                    for row in available
                    if row.get("algo") == algo and row.get("policy_mode") == mode
                ]
                method_rows.sort(key=lambda row: float(row["detection_probability"]))
                x = [float(row["detection_probability"]) for row in method_rows]
                y = [float(row[metric]) for row in method_rows]
                low_key = metric.removesuffix("_mean") + "_ci95_low"
                high_key = metric.removesuffix("_mean") + "_ci95_high"
                yerr_low, yerr_high = [], []
                valid_error = True
                for row, value in zip(method_rows, y):
                    low = row.get(low_key)
                    high = row.get(high_key)
                    if not is_finite_number(low) or not is_finite_number(high):
                        valid_error = False
                        break
                    yerr_low.append(max(0.0, value - float(low)))
                    yerr_high.append(max(0.0, float(high) - value))
                label = f"{algo}:{mode}"
                if valid_error:
                    ax.errorbar(x, y, yerr=[yerr_low, yerr_high], marker="o", capsize=4, label=label)
                else:
                    ax.plot(x, y, marker="o", label=label)

            ax.set_title(f"{scope} — final {ylabel}")
            ax.set_xlabel("Detection probability pD")
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.25)
            ax.legend(fontsize="small")
            _save(fig, output_dir / "final_performance" / scope, _slug(metric), dpi)


def _paired_plots(rows: list[dict[str, str]], output_dir: Path, dpi: int) -> None:
    if not rows:
        return
    metrics = [
        ("completed_mean_difference_mean", "Δ completed targets"),
        ("reward_mean_difference_mean", "Δ evaluation reward"),
        ("team_coverage_ratio_mean_difference_mean", "Δ team coverage"),
        ("collision_agent_ratio_difference_mean", "Δ collision ratio"),
    ]
    for metric, xlabel in metrics:
        selected = [row for row in rows if is_finite_number(row.get(metric))]
        if not selected:
            continue
        selected.sort(key=lambda row: (float(row["detection_probability"]), row["comparison"]))
        labels = [f"pD={float(row['detection_probability']):.2f}  {row['comparison']}" for row in selected]
        values = [float(row[metric]) for row in selected]
        low_key = metric.removesuffix("_mean") + "_ci95_low"
        high_key = metric.removesuffix("_mean") + "_ci95_high"
        lower = [
            max(0.0, value - float(row[low_key])) if is_finite_number(row.get(low_key)) else 0.0
            for row, value in zip(selected, values)
        ]
        upper = [
            max(0.0, float(row[high_key]) - value) if is_finite_number(row.get(high_key)) else 0.0
            for row, value in zip(selected, values)
        ]
        y = np.arange(len(selected))
        fig_height = max(5.0, 0.38 * len(selected) + 2.0)
        fig, ax = plt.subplots(figsize=(11, fig_height))
        ax.errorbar(values, y, xerr=[lower, upper], fmt="o", capsize=3)
        ax.axvline(0.0, linewidth=1.0)
        ax.set_yticks(y, labels)
        ax.set_xlabel(xlabel)
        ax.set_title(f"Paired comparisons — {xlabel}")
        ax.grid(axis="x", alpha=0.25)
        _save(fig, output_dir / "paired_comparisons", _slug(metric), dpi)


def main() -> None:
    args = build_parser().parse_args()
    aggregate_dir = Path(args.aggregate_dir)
    output_dir = Path(args.output_dir)
    learning_rows = _read(aggregate_dir / "learning_curves.csv")
    method_rows = _read(aggregate_dir / "summary_by_method.csv")
    paired_rows = _read(aggregate_dir / "paired_summary.csv")

    _learning_plots(learning_rows, output_dir, args.dpi)
    _final_plots(method_rows, output_dir, args.dpi)
    _paired_plots(paired_rows, output_dir, args.dpi)
    print(f"Saved plots in {output_dir}")


if __name__ == "__main__":
    main()

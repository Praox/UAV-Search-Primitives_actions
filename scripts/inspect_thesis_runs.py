from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

from uav_search_belief20.experiments.thesis_automation import (
    is_finite_number,
    load_json,
    tail,
    write_csv,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect thesis runs, logs and common failure signatures.")
    parser.add_argument("--run-root", default="runs/thesis_v2")
    parser.add_argument("--log-root", default="logs/thesis_v2")
    parser.add_argument("--output-dir", default="logs/thesis_v2/diagnostics")
    return parser


def _metrics(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _finite(row: dict[str, str], key: str) -> float | None:
    value = row.get(key)
    return float(value) if is_finite_number(value) else None


def _matching_logs(log_root: Path, algo: str, seed: int) -> list[Path]:
    candidates = list(log_root.glob(f"**/{algo}/seed{seed}_*.log"))
    if not candidates:
        candidates = list(log_root.glob(f"**/*{algo}*seed{seed}*.log"))
    return sorted(set(candidates))


def _evaluation_modes(run_dir: Path) -> dict[str, dict]:
    output = {}
    for path in sorted((run_dir / "evaluation").glob("*_summary.json")):
        try:
            payload = load_json(path)
        except Exception:
            continue
        mode = str(payload.get("policy_mode", path.stem))
        output[mode] = payload
    return output


def main() -> None:
    args = build_parser().parse_args()
    run_root = Path(args.run_root)
    log_root = Path(args.log_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    warning_lines: list[str] = []

    for config_path in sorted(run_root.glob("**/run_config.json")):
        run_dir = config_path.parent
        try:
            config = load_json(config_path)
        except Exception as exc:
            rows.append({"run_dir": str(run_dir), "status": "bad_config", "warnings": str(exc)})
            continue

        algo = str(config.get("algo", "unknown"))
        seed = int(config.get("seed", -1))
        metrics = _metrics(run_dir / "metrics.csv")
        best_exists = (run_dir / "best.pt").exists()
        latest_exists = (run_dir / "latest.pt").exists()
        eval_index_exists = (run_dir / "evaluation" / "final_test_evaluation_index.json").exists()
        status = "complete" if best_exists and eval_index_exists else "trained" if best_exists else "incomplete"
        warnings: list[str] = []

        if not metrics:
            warnings.append("metrics.csv absent ou vide")
        if not best_exists:
            warnings.append("best.pt absent")
        if not latest_exists:
            warnings.append("latest.pt absent")
        if best_exists and not eval_index_exists:
            warnings.append("évaluation détaillée finale absente")

        last = metrics[-1] if metrics else {}
        episodes = [_finite(row, "episode") for row in metrics]
        completed = [
            value for row in metrics if (value := _finite(row, "validation_completed")) is not None
        ]
        rewards = [
            value for row in metrics if (value := _finite(row, "validation_reward")) is not None
        ]
        collisions = [
            value for row in metrics if (value := _finite(row, "validation_collision")) is not None
        ]
        overlaps = [
            value for row in metrics if (value := _finite(row, "validation_overlap")) is not None
        ]

        if completed and completed[-1] < max(completed) - 0.25:
            warnings.append(
                f"régression tardive: completed final={completed[-1]:.3f}, max={max(completed):.3f}"
            )
        if collisions and collisions[-1] > 0.20:
            warnings.append(f"collision finale élevée: {collisions[-1]:.3f}")
        if overlaps and overlaps[-1] > 0.60:
            warnings.append(f"overlap final élevé: {overlaps[-1]:.3f}")

        last_loss = _finite(last, "loss")
        if metrics and last_loss is None:
            warnings.append("loss finale non finie ou absente")

        posterior_rebuilds = _finite(last, "posterior_rebuilds")
        if algo in {"bdqn", "shared_bdqn"} and metrics and (posterior_rebuilds is None or posterior_rebuilds < 1):
            warnings.append("aucun posterior rebuild observé")

        posterior_std = _finite(last, "posterior_std_mean")
        if posterior_std is not None:
            if posterior_std <= 1.2e-4:
                warnings.append(f"posterior std proche du minimum: {posterior_std:.2e}")
            if posterior_std >= 0.95:
                warnings.append(f"posterior std proche du maximum: {posterior_std:.3f}")

        tracebacks = []
        for log_path in _matching_logs(log_root, algo, seed):
            text = tail(log_path, 80)
            if "Traceback (most recent call last)" in text or "ERROR:" in text:
                tracebacks.append(str(log_path))
        if tracebacks:
            warnings.append("erreur détectée dans: " + ", ".join(tracebacks))
            if status == "incomplete":
                status = "failed"

        evaluations = _evaluation_modes(run_dir)
        mean_payload = evaluations.get("posterior_mean")
        sample_payload = (
            evaluations.get("posterior_sample")
            or evaluations.get("posterior_sample_shared")
            or evaluations.get("posterior_sample_independent")
        )
        sampled_minus_mean_completed = float("nan")
        sampled_minus_mean_coverage = float("nan")
        if mean_payload and sample_payload:
            mean_summary = mean_payload.get("summary", {})
            sample_summary = sample_payload.get("summary", {})
            if is_finite_number(mean_summary.get("completed_mean")) and is_finite_number(sample_summary.get("completed_mean")):
                sampled_minus_mean_completed = float(sample_summary["completed_mean"]) - float(mean_summary["completed_mean"])
                if abs(sampled_minus_mean_completed) > 0.50:
                    warnings.append(
                        f"fort écart sampled-mean sur completion: {sampled_minus_mean_completed:+.3f}"
                    )
            coverage_key = "team_coverage_ratio_mean" if algo.startswith(("shared_", "qmix_", "bayes_qmix_")) else "sensor_coverage_ratio_mean"
            if is_finite_number(mean_summary.get(coverage_key)) and is_finite_number(sample_summary.get(coverage_key)):
                sampled_minus_mean_coverage = float(sample_summary[coverage_key]) - float(mean_summary[coverage_key])
                if abs(sampled_minus_mean_coverage) > 0.15:
                    warnings.append(
                        f"fort écart sampled-mean sur coverage: {sampled_minus_mean_coverage:+.3f}"
                    )

        row = {
            "run_dir": str(run_dir),
            "status": status,
            "algo": algo,
            "seed": seed,
            "detection_probability": config.get("detection_probability"),
            "reward_mode": config.get("reward_mode"),
            "global_state_mode": config.get("global_state_mode"),
            "metric_rows": len(metrics),
            "last_episode": max([value for value in episodes if value is not None], default=float("nan")),
            "validation_reward_final": rewards[-1] if rewards else float("nan"),
            "validation_reward_best": max(rewards) if rewards else float("nan"),
            "validation_completed_final": completed[-1] if completed else float("nan"),
            "validation_completed_best": max(completed) if completed else float("nan"),
            "validation_collision_final": collisions[-1] if collisions else float("nan"),
            "validation_overlap_final": overlaps[-1] if overlaps else float("nan"),
            "loss_final": last_loss if last_loss is not None else float("nan"),
            "posterior_rebuilds_final": posterior_rebuilds if posterior_rebuilds is not None else float("nan"),
            "posterior_std_mean_final": posterior_std if posterior_std is not None else float("nan"),
            "sampled_minus_mean_completed": sampled_minus_mean_completed,
            "sampled_minus_mean_coverage": sampled_minus_mean_coverage,
            "warnings": " | ".join(warnings),
        }
        rows.append(row)
        if warnings:
            warning_lines.append(f"- **{algo} seed {seed} pD={config.get('detection_probability')}**: " + "; ".join(warnings))

    write_csv(output_dir / "run_diagnostics.csv", rows)

    counts: dict[str, int] = {}
    for row in rows:
        counts[str(row["status"])] = counts.get(str(row["status"]), 0) + 1
    markdown = [
        "# Diagnostic automatique des runs thesis-v2",
        "",
        "## État global",
        "",
    ]
    for status, count in sorted(counts.items()):
        markdown.append(f"- `{status}`: {count}")
    markdown.extend(["", "## Alertes", ""])
    markdown.extend(warning_lines or ["Aucune alerte automatique détectée."])
    markdown.extend(
        [
            "",
            "## Interprétation",
            "",
            "Une alerte n'implique pas automatiquement qu'un run est invalide. Elle indique un point à vérifier dans les courbes, le log d'entraînement et l'évaluation appariée.",
        ]
    )
    (output_dir / "run_diagnostics.md").write_text("\n".join(markdown) + "\n")
    print(f"Inspected {len(rows)} runs. Saved diagnostics in {output_dir}")


if __name__ == "__main__":
    main()

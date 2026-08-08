from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


# Two-sided 95% Student-t critical values, indexed by degrees of freedom.
# Values above 30 use the normal approximation, which is adequate for the
# repository's intended 3-14 training seeds.
_T95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
    26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}


def parse_csv_strings(value: str | Sequence[str]) -> list[str]:
    if isinstance(value, str):
        raw = value.split(",")
    else:
        raw = list(value)
    return [str(item).strip() for item in raw if str(item).strip()]


def parse_probabilities(value: str | Sequence[float]) -> list[float]:
    if isinstance(value, str):
        items = parse_csv_strings(value)
    else:
        items = [str(item) for item in value]
    probabilities = [float(item) for item in items]
    for probability in probabilities:
        if not 0.0 < probability <= 1.0:
            raise ValueError("Detection probabilities must be in (0, 1].")
    return probabilities


def parse_seeds(value: str | Sequence[int]) -> list[int]:
    """Parse ``42,43,44`` and inclusive ranges such as ``42-48``."""

    if not isinstance(value, str):
        seeds = [int(item) for item in value]
    else:
        seeds: list[int] = []
        for token in parse_csv_strings(value):
            if "-" in token:
                left, right = token.split("-", 1)
                start, stop = int(left), int(right)
                step = 1 if stop >= start else -1
                seeds.extend(range(start, stop + step, step))
            else:
                seeds.append(int(token))
    # Preserve order while removing accidental duplicates.
    return list(dict.fromkeys(seeds))


def probability_label(probability: float) -> str:
    return f"pdet_{float(probability):.2f}".replace(".", "p")


def safe_float(value, default: float = float("nan")) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result


def is_finite_number(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def t_critical_95(n: int) -> float:
    if n <= 1:
        return float("nan")
    return _T95.get(n - 1, 1.96)


def mean_std_ci95(values: Iterable[float]) -> dict[str, float | int]:
    array = np.asarray(
        [float(value) for value in values if is_finite_number(value)],
        dtype=np.float64,
    )
    n = int(array.size)
    if n == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "sem": float("nan"),
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
        }
    mean = float(array.mean())
    if n == 1:
        return {
            "n": 1,
            "mean": mean,
            "std": 0.0,
            "sem": float("nan"),
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
        }
    std = float(array.std(ddof=1))
    sem = std / math.sqrt(n)
    half_width = t_critical_95(n) * sem
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "sem": sem,
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
    }


def summarize_episode_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    weighted_ratios: Mapping[str, tuple[str, str]] | None = None,
) -> dict[str, object]:
    """Summarize numeric episode columns and selected weighted ratios.

    Every ordinary numeric column receives ``mean``, population ``std`` and a
    Student-t confidence interval over evaluation episodes. Ratio metrics that
    should be pooled over decisions, rather than averaged episode-wise, can be
    supplied as ``output_name -> (numerator_column, denominator_column)``.
    """

    if not rows:
        raise ValueError("Cannot summarize an empty episode table.")

    summary: dict[str, object] = {"episodes": len(rows)}
    keys = sorted({key for row in rows for key in row})
    excluded = {"episode", "world_seed", "scope", "algo", "policy_mode"}
    for key in keys:
        if key in excluded:
            continue
        values = [row.get(key) for row in rows]
        finite = [float(value) for value in values if is_finite_number(value)]
        if not finite:
            continue
        array = np.asarray(finite, dtype=np.float64)
        stats = mean_std_ci95(array)
        summary[f"{key}_mean"] = stats["mean"]
        summary[f"{key}_std"] = float(array.std(ddof=0))
        summary[f"{key}_ci95_low"] = stats["ci95_low"]
        summary[f"{key}_ci95_high"] = stats["ci95_high"]

    for output_name, (numerator_key, denominator_key) in (weighted_ratios or {}).items():
        numerator = sum(safe_float(row.get(numerator_key), 0.0) for row in rows)
        denominator = sum(safe_float(row.get(denominator_key), 0.0) for row in rows)
        summary[output_name] = numerator / max(1.0, denominator)

    return summary


def flatten_summary_record(record: Mapping[str, object]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in record.items():
        if key == "summary" and isinstance(value, Mapping):
            output.update(value)
        elif isinstance(value, (str, int, float, bool)) or value is None:
            output[key] = value
    return output


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row}) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def load_json(path: Path) -> dict:
    with path.open() as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def write_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(value, handle, indent=2, allow_nan=True)


def resolve_checkpoint(run_dir: Path, checkpoint: str) -> Path:
    if checkpoint in {"best", "latest"}:
        path = run_dir / f"{checkpoint}.pt"
    else:
        path = Path(checkpoint)
        if not path.is_absolute():
            path = run_dir / path
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def valid_evaluation_summary(path: Path) -> bool:
    try:
        payload = load_json(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    required = {"scope", "algo", "policy_mode", "summary"}
    return required.issubset(payload) and isinstance(payload.get("summary"), dict)


def tail(path: Path, lines: int = 40) -> str:
    try:
        content = path.read_text(errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(content[-int(lines):])

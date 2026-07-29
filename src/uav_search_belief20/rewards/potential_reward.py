from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class PotentialRewardConfig:
    """Weights of Phi(s) = w_C C(s) + w_D D(s) + w_P P(s)."""

    gamma: float = 0.99
    coverage_weight: float = 10.0
    detection_weight: float = 1.0
    progress_weight: float = 2.0

    def validate(self) -> None:
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError(f"gamma must be in [0, 1], got {self.gamma}")
        for name in ("coverage_weight", "detection_weight", "progress_weight"):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be non-negative")

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class PotentialSnapshot:
    """Only the observable/memorized quantities used by the potential."""

    coverage_ratio: float
    detected: tuple[bool, ...]
    track_progress: tuple[int, ...]
    target_values: tuple[int, ...]
    track_required: int

    @classmethod
    def from_arrays(
        cls,
        *,
        coverage_ratio: float,
        detected: Sequence[bool] | np.ndarray,
        track_progress: Sequence[int] | np.ndarray,
        target_values: Sequence[int] | np.ndarray,
        track_required: int,
    ) -> "PotentialSnapshot":
        result = cls(
            coverage_ratio=float(coverage_ratio),
            detected=tuple(bool(x) for x in detected),
            track_progress=tuple(int(x) for x in track_progress),
            target_values=tuple(int(x) for x in target_values),
            track_required=int(track_required),
        )
        result.validate()
        return result

    def validate(self) -> None:
        n_targets = len(self.target_values)
        if not 0.0 <= self.coverage_ratio <= 1.0 + 1e-7:
            raise ValueError(f"coverage_ratio must be in [0, 1], got {self.coverage_ratio}")
        if self.track_required <= 0:
            raise ValueError("track_required must be positive")
        if len(self.detected) != n_targets or len(self.track_progress) != n_targets:
            raise ValueError("All target vectors must have the same length")
        if any(value <= 0 for value in self.target_values):
            raise ValueError("target values must be positive")
        if any(progress < 0 for progress in self.track_progress):
            raise ValueError("track progress must be non-negative")


@dataclass(frozen=True)
class ShapingBreakdown:
    coverage: float
    detection: float
    progress: float
    total: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


class PotentialReward:
    """Pure potential-based shaping shared by single-UAV and multi-UAV envs."""

    def __init__(self, config: PotentialRewardConfig):
        config.validate()
        self.cfg = config

    @staticmethod
    def _denominator(snapshot: PotentialSnapshot) -> float:
        return float(max(1, sum(snapshot.target_values)))

    def normalized_components(self, snapshot: PotentialSnapshot) -> tuple[float, float, float]:
        snapshot.validate()
        denominator = self._denominator(snapshot)

        coverage = float(np.clip(snapshot.coverage_ratio, 0.0, 1.0))
        detection = sum(
            value
            for value, is_detected in zip(snapshot.target_values, snapshot.detected)
            if is_detected
        ) / denominator
        progress = sum(
            value * min(progress, snapshot.track_required) / snapshot.track_required
            for value, progress in zip(snapshot.target_values, snapshot.track_progress)
        ) / denominator

        return float(coverage), float(detection), float(progress)

    def weighted_components(self, snapshot: PotentialSnapshot) -> tuple[float, float, float]:
        coverage, detection, progress = self.normalized_components(snapshot)
        return (
            self.cfg.coverage_weight * coverage,
            self.cfg.detection_weight * detection,
            self.cfg.progress_weight * progress,
        )

    def potential(self, snapshot: PotentialSnapshot) -> float:
        return float(sum(self.weighted_components(snapshot)))

    def shaping(self, before: PotentialSnapshot, after: PotentialSnapshot) -> ShapingBreakdown:
        """Return gamma*Phi(after) - Phi(before), split by component."""
        if before.target_values != after.target_values:
            raise ValueError("target_values changed during a transition")
        if before.track_required != after.track_required:
            raise ValueError("track_required changed during a transition")

        previous = self.weighted_components(before)
        following = self.weighted_components(after)
        components = tuple(
            self.cfg.gamma * next_value - previous_value
            for previous_value, next_value in zip(previous, following)
        )
        return ShapingBreakdown(
            coverage=float(components[0]),
            detection=float(components[1]),
            progress=float(components[2]),
            total=float(sum(components)),
        )

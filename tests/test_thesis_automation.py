from __future__ import annotations

from uav_search_belief20.experiments.thesis_automation import (
    mean_std_ci95,
    parse_probabilities,
    parse_seeds,
    probability_label,
    summarize_episode_rows,
)


def test_seed_and_probability_parsing() -> None:
    assert parse_seeds("42-44,48") == [42, 43, 44, 48]
    assert parse_probabilities("1.0,0.7") == [1.0, 0.7]
    assert probability_label(0.7) == "pdet_0p70"


def test_student_interval_and_weighted_ratios() -> None:
    stats = mean_std_ci95([1.0, 2.0, 3.0])
    assert stats["n"] == 3
    assert stats["ci95_low"] < stats["mean"] < stats["ci95_high"]

    rows = [
        {"reward": 1.0, "completed": 1.0, "detected": 2.0, "decisions": 10, "action_stay": 2},
        {"reward": 3.0, "completed": 2.0, "detected": 2.0, "decisions": 20, "action_stay": 3},
    ]
    summary = summarize_episode_rows(
        rows,
        weighted_ratios={
            "detected_to_completed_ratio": ("completed", "detected"),
            "stay_ratio": ("action_stay", "decisions"),
        },
    )
    assert summary["reward_mean"] == 2.0
    assert summary["detected_to_completed_ratio"] == 0.75
    assert summary["stay_ratio"] == 5.0 / 30.0

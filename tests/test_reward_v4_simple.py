from __future__ import annotations

import math

from uav_search_belief20.rewards.potential_reward import (
    PotentialReward,
    PotentialRewardConfig,
    PotentialSnapshot,
)


def snap(*, coverage=0.0, detected=(False, False, False, False), progress=(0, 0, 0, 0)):
    return PotentialSnapshot.from_arrays(
        coverage_ratio=coverage,
        detected=detected,
        track_progress=progress,
        target_values=(1, 1, 1, 2),
        track_required=3,
    )


def test_empty_potential_is_zero():
    reward = PotentialReward(PotentialRewardConfig())
    assert reward.potential(snap()) == 0.0


def test_maximum_potential_is_thirteen():
    reward = PotentialReward(PotentialRewardConfig())
    state = snap(coverage=1.0, detected=(True, True, True, True), progress=(3, 3, 3, 3))
    assert math.isclose(reward.potential(state), 13.0, rel_tol=1e-9)


def test_ten_new_cells_give_positive_coverage_signal():
    reward = PotentialReward(PotentialRewardConfig(gamma=0.99))
    before = snap(coverage=0.20)
    after = snap(coverage=0.225)  # 10 / 400 new cells
    parts = reward.shaping(before, after)
    assert parts.coverage > 0.0
    assert math.isclose(parts.total, parts.coverage, rel_tol=1e-9)


def test_no_progress_has_small_negative_discount_drift():
    reward = PotentialReward(PotentialRewardConfig(gamma=0.99))
    state = snap(coverage=0.20)
    parts = reward.shaping(state, state)
    assert parts.total < 0.0


def test_value_two_detection_is_twice_value_one_detection():
    reward = PotentialReward(PotentialRewardConfig(gamma=1.0))
    base = snap()
    value_one = reward.shaping(base, snap(detected=(True, False, False, False))).detection
    value_two = reward.shaping(base, snap(detected=(False, False, False, True))).detection
    assert math.isclose(value_two, 2.0 * value_one, rel_tol=1e-9)


def test_tracking_progress_is_monotone_and_capped():
    reward = PotentialReward(PotentialRewardConfig(gamma=1.0))
    p0 = reward.potential(snap(progress=(0, 0, 0, 0)))
    p1 = reward.potential(snap(progress=(1, 0, 0, 0)))
    p3 = reward.potential(snap(progress=(3, 0, 0, 0)))
    p9 = reward.potential(snap(progress=(9, 0, 0, 0)))
    assert p0 < p1 < p3
    assert math.isclose(p3, p9, rel_tol=1e-9)


def test_breakdown_sums_to_total():
    reward = PotentialReward(PotentialRewardConfig())
    parts = reward.shaping(
        snap(coverage=0.1),
        snap(coverage=0.2, detected=(True, False, False, False), progress=(1, 0, 0, 0)),
    )
    assert math.isclose(
        parts.total,
        parts.coverage + parts.detection + parts.progress,
        rel_tol=1e-9,
    )

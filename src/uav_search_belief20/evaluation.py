from __future__ import annotations

from collections import Counter
from typing import Callable

import numpy as np

from uav_search_belief20.actions import ACTION_NAMES
from uav_search_belief20.envs.primitive_search_env import REWARD_PART_KEYS, PrimitiveSearchEnv


PolicyFn = Callable[[PrimitiveSearchEnv, np.ndarray, int], int]
EpisodeStartFn = Callable[[int], None]


def evaluate_policy(
    env_factory: Callable[[int], PrimitiveSearchEnv],
    policy: PolicyFn,
    *,
    episodes: int,
    seed_base: int,
    on_episode_start: EpisodeStartFn | None = None,
) -> dict[str, float | int | dict[str, int] | str]:
    """Evaluate one policy on deterministic, reproducible environment seeds."""

    rewards: list[float] = []
    detected: list[float] = []
    completed: list[float] = []
    detected_value: list[float] = []
    completed_value: list[float] = []
    coverage: list[float] = []
    episode_lengths: list[float] = []
    first_detection_steps: list[float] = []
    first_completion_steps: list[float] = []

    action_counts: Counter[str] = Counter()
    reward_parts: Counter[str] = Counter()
    boundary_hits = 0
    position_revisit_steps = 0
    sensor_revisit_steps = 0
    tracking_progress_steps = 0
    new_observed_total = 0
    decisions = 0

    final_info: dict = {}
    for episode_index in range(int(episodes)):
        if on_episode_start is not None:
            on_episode_start(episode_index)

        env = env_factory(int(seed_base) + episode_index)
        obs, info = env.reset()
        done = False
        total_reward = 0.0
        previous_detected = 0
        previous_completed = 0
        first_detection: int | None = None
        first_completion: int | None = None

        while not done:
            action = int(policy(env, obs, episode_index))
            if not env.action_mask()[action]:
                raise RuntimeError(
                    f"Policy selected masked action {action} at position {env.drone_pos}."
                )

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            total_reward += float(reward)
            obs = next_obs

            action_counts[ACTION_NAMES[action]] += 1
            boundary_hits += int(info["last_boundary_hit"])
            position_revisit_steps += int(not info["last_new_cell"])
            sensor_revisit_steps += int(info["last_new_observed_cells"] == 0)
            new_observed_total += int(info["last_new_observed_cells"])
            tracking_progress_steps += int(info["last_tracking_progress"])
            decisions += 1

            for key, value in info.get("last_reward_parts", {}).items():
                reward_parts[str(key)] += float(value)

            current_detected = int(info["detected"])
            current_completed = int(info["completed"])
            if first_detection is None and current_detected > previous_detected:
                first_detection = int(info["t"])
            if first_completion is None and current_completed > previous_completed:
                first_completion = int(info["t"])
            previous_detected = current_detected
            previous_completed = current_completed

        rewards.append(total_reward)
        detected.append(float(info["detected"]))
        completed.append(float(info["completed"]))
        detected_value.append(float(info["detected_value"]))
        completed_value.append(float(info["completed_value"]))
        coverage.append(float(info["visited_ratio"]))
        episode_lengths.append(float(info["t"]))
        if first_detection is not None:
            first_detection_steps.append(float(first_detection))
        if first_completion is not None:
            first_completion_steps.append(float(first_completion))
        final_info = info

    episode_count = max(1, int(episodes))
    decision_count = max(1, decisions)
    detected_total = float(np.sum(detected))
    completed_total = float(np.sum(completed))

    result: dict[str, float | int | dict[str, int] | str] = {
        "reward_version": str(final_info.get("reward_version", "unknown")),
        "ablation": str(final_info.get("ablation", "unknown")),
        "episodes": int(episodes),
        "reward_mean": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
        "detected_mean": float(np.mean(detected)),
        "completed_mean": float(np.mean(completed)),
        "detected_value_mean": float(np.mean(detected_value)),
        "completed_value_mean": float(np.mean(completed_value)),
        "sensor_coverage_ratio_mean": float(np.mean(coverage)),
        "episode_length_mean": float(np.mean(episode_lengths)),
        "detected_to_completed_ratio": completed_total / max(1.0, detected_total),
        "first_detection_step_mean": (
            float(np.mean(first_detection_steps))
            if first_detection_steps
            else float("nan")
        ),
        "first_completion_step_mean": (
            float(np.mean(first_completion_steps))
            if first_completion_steps
            else float("nan")
        ),
        "stay_ratio": action_counts["stay"] / decision_count,
        "boundary_hit_ratio": boundary_hits / decision_count,
        "revisit_ratio": position_revisit_steps / decision_count,
        "sensor_revisit_ratio": sensor_revisit_steps / decision_count,
        "new_observed_cells_per_step": new_observed_total / decision_count,
        "tracking_progress_ratio": tracking_progress_steps / decision_count,
        "action_counts": dict(action_counts),
    }

    for key in REWARD_PART_KEYS:
        total = float(reward_parts.get(key, 0.0))
        result[f"reward_part_{key}_mean_per_episode"] = total / episode_count
        result[f"reward_part_{key}_mean_per_step"] = total / decision_count

    return result


def training_metric_view(metrics: dict[str, object]) -> dict[str, object]:
    """Map canonical evaluation metrics to the historical training CSV names."""

    output: dict[str, object] = {
        "eval_reward": metrics["reward_mean"],
        "eval_reward_std": metrics["reward_std"],
        "eval_detected": metrics["detected_mean"],
        "eval_completed": metrics["completed_mean"],
        "eval_detected_value": metrics["detected_value_mean"],
        "eval_completed_value": metrics["completed_value_mean"],
        "eval_sensor_coverage": metrics["sensor_coverage_ratio_mean"],
        "eval_episode_length": metrics["episode_length_mean"],
        "eval_detected_to_completed_ratio": metrics["detected_to_completed_ratio"],
        "eval_first_detection_step": metrics["first_detection_step_mean"],
        "eval_first_completion_step": metrics["first_completion_step_mean"],
        "stay_ratio": metrics["stay_ratio"],
        "boundary_hit_ratio": metrics["boundary_hit_ratio"],
        "revisit_ratio": metrics["revisit_ratio"],
        "sensor_revisit_ratio": metrics["sensor_revisit_ratio"],
        "new_observed_cells_per_step": metrics["new_observed_cells_per_step"],
        "tracking_progress_ratio": metrics["tracking_progress_ratio"],
    }
    for key in REWARD_PART_KEYS:
        output[f"eval_reward_part_{key}"] = metrics[
            f"reward_part_{key}_mean_per_episode"
        ]
    return output

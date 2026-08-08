from __future__ import annotations

from collections import Counter
from typing import Callable, Mapping

import numpy as np

from uav_search_belief20.actions import ACTION_NAMES
from uav_search_belief20.experiments.thesis_automation import summarize_episode_rows


SinglePolicy = Callable[[object, np.ndarray, int], int]
MultiPolicy = Callable[[object, np.ndarray, np.ndarray, int], np.ndarray]


def evaluate_single_detailed(
    *,
    env_factory: Callable[[int], object],
    policy: SinglePolicy,
    episodes: int,
    seed_base: int,
    on_episode_start: Callable[[int, int], None] | None = None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    rows: list[dict[str, object]] = []

    for episode_index in range(int(episodes)):
        world_seed = int(seed_base) + episode_index
        env = env_factory(world_seed)
        obs, info = env.reset()
        if on_episode_start is not None:
            on_episode_start(episode_index, world_seed)

        done = False
        total_reward = 0.0
        action_counts: Counter[str] = Counter()
        reward_parts: Counter[str] = Counter()
        boundary_hits = position_revisits = sensor_revisits = 0
        tracking_steps = new_observed_total = decisions = 0
        first_detection: int | None = None
        first_completion: int | None = None
        previous_detected = int(info.get("detected", 0))
        previous_completed = int(info.get("completed", 0))

        while not done:
            action = int(policy(env, obs, episode_index))
            mask = np.asarray(env.action_mask(), dtype=bool)
            if not bool(mask[action]):
                raise RuntimeError(
                    f"Policy selected invalid action {action} in world {world_seed}."
                )

            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            total_reward += float(reward)
            decisions += 1
            action_counts[ACTION_NAMES[action]] += 1
            boundary_hits += int(info.get("last_boundary_hit", False))
            position_revisits += int(not info.get("last_new_cell", False))
            sensor_revisits += int(info.get("last_new_observed_cells", 0) == 0)
            new_observed_total += int(info.get("last_new_observed_cells", 0))
            tracking_steps += int(info.get("last_tracking_progress", False))
            for key, value in info.get("last_reward_parts", {}).items():
                reward_parts[str(key)] += float(value)

            detected = int(info.get("detected", 0))
            completed = int(info.get("completed", 0))
            if first_detection is None and detected > previous_detected:
                first_detection = int(info.get("t", decisions))
            if first_completion is None and completed > previous_completed:
                first_completion = int(info.get("t", decisions))
            previous_detected = detected
            previous_completed = completed

        row: dict[str, object] = {
            "episode": episode_index,
            "world_seed": world_seed,
            "reward": total_reward,
            "detected": float(info.get("detected", 0)),
            "completed": float(info.get("completed", 0)),
            "detected_value": float(info.get("detected_value", 0)),
            "completed_value": float(info.get("completed_value", 0)),
            "sensor_coverage_ratio": float(info.get("visited_ratio", 0.0)),
            "episode_length": float(info.get("t", decisions)),
            "first_detection_step": (
                float(first_detection) if first_detection is not None else float("nan")
            ),
            "first_completion_step": (
                float(first_completion) if first_completion is not None else float("nan")
            ),
            "had_detection": float(first_detection is not None),
            "had_completion": float(first_completion is not None),
            "decisions": decisions,
            "boundary_hits": boundary_hits,
            "position_revisit_steps": position_revisits,
            "sensor_revisit_steps": sensor_revisits,
            "new_observed_cells": new_observed_total,
            "tracking_progress_steps": tracking_steps,
        }
        for name in ACTION_NAMES:
            row[f"action_{name}"] = int(action_counts[name])
        for key, value in reward_parts.items():
            row[f"reward_part_{key}"] = float(value)
        rows.append(row)

    summary = summarize_episode_rows(
        rows,
        weighted_ratios={
            "detected_to_completed_ratio": ("completed", "detected"),
            "stay_ratio": ("action_stay", "decisions"),
            "boundary_hit_ratio": ("boundary_hits", "decisions"),
            "revisit_ratio": ("position_revisit_steps", "decisions"),
            "sensor_revisit_ratio": ("sensor_revisit_steps", "decisions"),
            "new_observed_cells_per_step": ("new_observed_cells", "decisions"),
            "tracking_progress_ratio": ("tracking_progress_steps", "decisions"),
        },
    )
    summary["eval_seed_base"] = int(seed_base)
    return rows, summary


def evaluate_multi_detailed(
    *,
    env_factory: Callable[[int], object],
    policy: MultiPolicy,
    episodes: int,
    seed_base: int,
    on_episode_start: Callable[[int, int], None] | None = None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    rows: list[dict[str, object]] = []

    for episode_index in range(int(episodes)):
        world_seed = int(seed_base) + episode_index
        env = env_factory(world_seed)
        obs_all, info = env.reset()
        if on_episode_start is not None:
            on_episode_start(episode_index, world_seed)

        done = False
        total_reward = 0.0
        reward_parts: Counter[str] = Counter()
        action_counts: Counter[str] = Counter()
        boundary_hits = collision_agents = local_sensor_revisits = 0
        local_new_observed = team_new_observed = tracking_steps = 0
        simultaneous_overlap_sum = 0.0
        decisions = env_steps = 0
        first_detection: int | None = None
        first_completion: int | None = None
        previous_detected = int(info.get("detected", 0))
        previous_completed = int(info.get("completed", 0))

        while not done:
            masks = np.asarray(env.action_mask(), dtype=bool)
            actions = np.asarray(
                policy(env, obs_all, masks, episode_index), dtype=np.int64
            )
            expected = (env.cfg.n_agents,)
            if actions.shape != expected:
                raise RuntimeError(
                    f"Policy returned actions with shape {actions.shape}, expected {expected}."
                )
            for agent_id, action in enumerate(actions):
                if not bool(masks[agent_id, int(action)]):
                    raise RuntimeError(
                        f"Agent {agent_id} selected invalid action {action} in world {world_seed}."
                    )

            obs_all, reward, terminated, truncated, info = env.step(actions)
            done = bool(terminated or truncated)
            total_reward += float(reward)
            env_steps += 1
            simultaneous_overlap_sum += float(
                info.get("last_simultaneous_sensor_overlap_ratio", 0.0)
            )
            team_new_observed += int(info.get("last_new_team_observed_cells", 0))
            for key, value in info.get("last_reward_parts", {}).items():
                reward_parts[str(key)] += float(value)

            last_boundary = np.asarray(
                info.get("last_boundary_hits", np.zeros(env.cfg.n_agents)), dtype=bool
            )
            last_collisions = np.asarray(
                info.get("last_collisions", np.zeros(env.cfg.n_agents)), dtype=bool
            )
            last_new_observed = np.asarray(
                info.get("last_new_observed_cells", np.zeros(env.cfg.n_agents)),
                dtype=np.int64,
            )
            last_tracking = np.asarray(
                info.get("last_tracking_progress", np.zeros(env.cfg.n_agents)),
                dtype=bool,
            )
            for agent_id, action in enumerate(actions):
                action_counts[ACTION_NAMES[int(action)]] += 1
                boundary_hits += int(last_boundary[agent_id])
                collision_agents += int(last_collisions[agent_id])
                local_sensor_revisits += int(last_new_observed[agent_id] == 0)
                local_new_observed += int(last_new_observed[agent_id])
                tracking_steps += int(last_tracking[agent_id])
                decisions += 1

            detected = int(info.get("detected", 0))
            completed = int(info.get("completed", 0))
            if first_detection is None and detected > previous_detected:
                first_detection = int(info.get("t", env_steps))
            if first_completion is None and completed > previous_completed:
                first_completion = int(info.get("t", env_steps))
            previous_detected = detected
            previous_completed = completed

        row: dict[str, object] = {
            "episode": episode_index,
            "world_seed": world_seed,
            "reward": total_reward,
            "detected": float(info.get("detected", 0)),
            "completed": float(info.get("completed", 0)),
            "detected_value": float(info.get("detected_value", 0)),
            "completed_value": float(info.get("completed_value", 0)),
            "team_coverage_ratio": float(info.get("team_coverage_ratio", 0.0)),
            "mean_local_coverage_ratio": float(
                info.get("mean_local_coverage_ratio", 0.0)
            ),
            "coverage_overlap_ratio": float(
                info.get("coverage_overlap_ratio", 0.0)
            ),
            "knowledge_overlap_ratio": float(
                info.get("knowledge_overlap_ratio", 0.0)
            ),
            "episode_length": float(info.get("t", env_steps)),
            "first_detection_step": (
                float(first_detection) if first_detection is not None else float("nan")
            ),
            "first_completion_step": (
                float(first_completion) if first_completion is not None else float("nan")
            ),
            "had_detection": float(first_detection is not None),
            "had_completion": float(first_completion is not None),
            "decisions": decisions,
            "env_steps": env_steps,
            "boundary_hits": boundary_hits,
            "collision_agents": collision_agents,
            "local_sensor_revisit_steps": local_sensor_revisits,
            "local_new_observed_cells": local_new_observed,
            "team_new_observed_cells": team_new_observed,
            "tracking_progress_steps": tracking_steps,
            "simultaneous_overlap_sum": simultaneous_overlap_sum,
        }
        for name in ACTION_NAMES:
            row[f"action_{name}"] = int(action_counts[name])
        for key, value in reward_parts.items():
            row[f"reward_part_{key}"] = float(value)
        rows.append(row)

    summary = summarize_episode_rows(
        rows,
        weighted_ratios={
            "detected_to_completed_ratio": ("completed", "detected"),
            "stay_ratio": ("action_stay", "decisions"),
            "boundary_hit_ratio": ("boundary_hits", "decisions"),
            "collision_agent_ratio": ("collision_agents", "decisions"),
            "local_sensor_revisit_ratio": (
                "local_sensor_revisit_steps",
                "decisions",
            ),
            "local_new_observed_cells_per_decision": (
                "local_new_observed_cells",
                "decisions",
            ),
            "team_new_observed_cells_per_env_step": (
                "team_new_observed_cells",
                "env_steps",
            ),
            "tracking_progress_ratio": ("tracking_progress_steps", "decisions"),
            "simultaneous_sensor_overlap_ratio_mean": (
                "simultaneous_overlap_sum",
                "env_steps",
            ),
        },
    )
    summary["eval_seed_base"] = int(seed_base)
    return rows, summary

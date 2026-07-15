from __future__ import annotations

from collections import Counter, defaultdict
from typing import Callable

import numpy as np

from uav_search_belief20.actions import ACTION_NAMES
from uav_search_belief20.envs.multi_drone_local_env import (
    MULTI_REWARD_PART_KEYS,
    MultiDroneLocalMemoryEnv,
)

PolicyFn = Callable[[np.ndarray, np.ndarray], np.ndarray]
EnvFactory = Callable[[int], MultiDroneLocalMemoryEnv]


def evaluate_multi_local_policy(
    *,
    policy: PolicyFn,
    env_factory: EnvFactory,
    episodes: int,
    eval_seed_base: int = 100_000,
    episode_start_fn: Callable[[], None] | None = None,
) -> dict:
    rewards, detected, completed = [], [], []
    detected_value, completed_value = [], []
    team_coverage, local_coverage, coverage_overlap, knowledge_overlap = [], [], [], []
    episode_lengths, first_detection_steps, first_completion_steps = [], [], []
    action_counts = Counter()
    reward_parts = defaultdict(list)
    boundary_hits = collision_agents = local_sensor_revisits = 0
    local_new_observed = team_new_observed = tracking_progress = 0
    simultaneous_sensor_overlap_sum = 0.0
    decisions = env_steps = 0

    for episode in range(int(episodes)):
        env = env_factory(int(eval_seed_base) + episode)
        obs_all, info = env.reset()
        if episode_start_fn is not None:
            episode_start_fn()
        done = False
        total_reward = 0.0
        episode_parts = {key: 0.0 for key in MULTI_REWARD_PART_KEYS}
        first_detection = first_completion = None
        while not done:
            masks = env.action_mask()
            actions = np.asarray(policy(obs_all, masks), dtype=np.int64)
            obs_all, reward, terminated, truncated, info = env.step(actions)
            done = bool(terminated or truncated)
            total_reward += float(reward)
            env_steps += 1
            for key, value in info["last_reward_parts"].items():
                episode_parts[key] = episode_parts.get(key, 0.0) + float(value)
            for agent_id, action in enumerate(actions):
                action_counts[ACTION_NAMES[int(action)]] += 1
                boundary_hits += int(info["last_boundary_hits"][agent_id])
                collision_agents += int(info["last_collisions"][agent_id])
                local_sensor_revisits += int(
                    info["last_new_observed_cells"][agent_id] == 0
                )
                local_new_observed += int(info["last_new_observed_cells"][agent_id])
                tracking_progress += int(info["last_tracking_progress"][agent_id])
                decisions += 1
            team_new_observed += int(info["last_new_team_observed_cells"])
            simultaneous_sensor_overlap_sum += float(
                info.get("last_simultaneous_sensor_overlap_ratio", 0.0)
            )
            if first_detection is None and info["detected"] > 0:
                first_detection = int(info["t"])
            if first_completion is None and info["completed"] > 0:
                first_completion = int(info["t"])

        rewards.append(total_reward)
        detected.append(float(info["detected"]))
        completed.append(float(info["completed"]))
        detected_value.append(float(info["detected_value"]))
        completed_value.append(float(info["completed_value"]))
        team_coverage.append(float(info["team_coverage_ratio"]))
        local_coverage.append(float(info["mean_local_coverage_ratio"]))
        coverage_overlap.append(float(info["coverage_overlap_ratio"]))
        knowledge_overlap.append(float(info["knowledge_overlap_ratio"]))
        episode_lengths.append(float(info["t"]))
        if first_detection is not None:
            first_detection_steps.append(float(first_detection))
        if first_completion is not None:
            first_completion_steps.append(float(first_completion))
        for key in MULTI_REWARD_PART_KEYS:
            reward_parts[key].append(float(episode_parts.get(key, 0.0)))

    detected_mean = float(np.mean(detected))
    completed_mean = float(np.mean(completed))
    metrics = {
        "episodes": int(episodes),
        "reward_mean": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
        "detected_mean": detected_mean,
        "completed_mean": completed_mean,
        "detected_value_mean": float(np.mean(detected_value)),
        "completed_value_mean": float(np.mean(completed_value)),
        "detected_to_completed_ratio": completed_mean / max(detected_mean, 1e-9),
        "team_coverage_ratio_mean": float(np.mean(team_coverage)),
        "mean_local_coverage_ratio_mean": float(np.mean(local_coverage)),
        "coverage_overlap_ratio_mean": float(np.mean(coverage_overlap)),
        "knowledge_overlap_ratio_mean": float(np.mean(knowledge_overlap)),
        "episode_length_mean": float(np.mean(episode_lengths)),
        "first_detection_step_mean": (
            float(np.mean(first_detection_steps)) if first_detection_steps else float("nan")
        ),
        "first_completion_step_mean": (
            float(np.mean(first_completion_steps)) if first_completion_steps else float("nan")
        ),
        "stay_ratio": action_counts["stay"] / max(1, decisions),
        "boundary_hit_ratio": boundary_hits / max(1, decisions),
        "collision_agent_ratio": collision_agents / max(1, decisions),
        "local_sensor_revisit_ratio": local_sensor_revisits / max(1, decisions),
        "local_new_observed_cells_per_decision": local_new_observed / max(1, decisions),
        "team_new_observed_cells_per_env_step": team_new_observed / max(1, env_steps),
        "tracking_progress_ratio": tracking_progress / max(1, decisions),
        "simultaneous_sensor_overlap_ratio_mean": (
            simultaneous_sensor_overlap_sum / max(1, env_steps)
        ),
        "action_counts": dict(action_counts),
        "eval_seed_base": int(eval_seed_base),
    }
    for key in MULTI_REWARD_PART_KEYS:
        values = reward_parts[key]
        metrics[f"reward_part_{key}_mean_per_episode"] = float(np.mean(values))
        metrics[f"reward_part_{key}_mean_per_env_step"] = float(
            np.sum(values) / max(1, env_steps)
        )
    return metrics

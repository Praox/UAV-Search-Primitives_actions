from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable

import numpy as np

from uav_search_belief20.actions import ACTION_DIM, ACTION_NAMES, MOVES, STAY
from uav_search_belief20.envs.drone_memory import DroneMemory


MULTI_REWARD_PART_KEYS: tuple[str, ...] = (
    "step", "boundary", "collision", "new_cell", "revisit",
    "new_observed", "detect", "track_progress", "complete",
    "idle_stay", "all_targets",
)


@dataclass
class MultiDroneLocalEnvConfig:
    grid_size: int = 20
    n_agents: int = 3
    n_value1_targets: int = 3
    n_value2_targets: int = 1
    sensor_radius: int = 2
    teammate_visibility_radius: int = 2
    detection_probability: float = 1.0
    track_radius: int = 1
    track_required: int = 3
    max_steps: int = 150
    seed: int | None = None

    reward_version: str = "multi_local_v1_from_single_D"
    use_boundary_action_mask: bool = True
    include_agent_id_map: bool = False
    global_state_mode: str = "privileged_truth"

    step_penalty: float = -0.005
    collision_penalty: float = -0.02
    new_cell_bonus: float = 0.002
    revisit_penalty: float = 0.0
    new_observed_cell_bonus: float = 0.015
    new_observed_cell_bonus_cap_per_agent: float = 0.15
    boundary_penalty: float = -0.20
    idle_stay_penalty: float = -0.03

    detect_value1_bonus: float = 0.60
    detect_value2_bonus: float = 1.50
    track_progress_value1_bonus: float = 0.30
    track_progress_value2_bonus: float = 0.90
    complete_value1_bonus: float = 4.0
    complete_value2_bonus: float = 12.0
    all_targets_bonus: float = 5.0
    max_trackers_per_target: int = 1

    def reward_dict(self) -> dict:
        return asdict(self)


class MultiDroneLocalMemoryEnv:
    """Multi-UAV environment with one distinct DroneMemory per UAV.

    Local channels: own position, locally visible teammates, local belief,
    local known target values, local tracking progress, local completed map,
    local visited map, time remaining, and optional agent-id map.
    """

    action_dim = ACTION_DIM
    action_names = ACTION_NAMES

    def __init__(self, config: MultiDroneLocalEnvConfig | None = None):
        self.cfg = config or MultiDroneLocalEnvConfig()
        if self.cfg.n_agents <= 0:
            raise ValueError("n_agents must be positive")
        if self.cfg.max_trackers_per_target != 1:
            raise ValueError("v1 supports exactly one tracker per target and step")
        if self.cfg.global_state_mode not in {"privileged_truth", "memory_union"}:
            raise ValueError("global_state_mode must be privileged_truth or memory_union")
        self.rng = np.random.default_rng(self.cfg.seed)
        self.observation_channels = 8 + int(self.cfg.include_agent_id_map)
        self.observation_shape = (
            self.observation_channels,
            self.cfg.grid_size,
            self.cfg.grid_size,
        )
        self.reset()

    @property
    def n_targets(self) -> int:
        return self.cfg.n_value1_targets + self.cfg.n_value2_targets

    @property
    def state_dim(self) -> int:
        return 2 * self.cfg.n_agents + 6 * self.n_targets + 1

    def reset(self):
        grid = self.cfg.grid_size
        self.t = 0
        self.target_values = np.array(
            [1] * self.cfg.n_value1_targets + [2] * self.cfg.n_value2_targets,
            dtype=np.int64,
        )
        forbidden: set[tuple[int, int]] = set()
        self.drone_pos = np.asarray(
            self._sample_unique_positions(self.cfg.n_agents, forbidden), dtype=np.int64
        )
        self.target_pos = np.asarray(
            self._sample_unique_positions(self.n_targets, forbidden), dtype=np.int64
        )
        self.detected = np.zeros(self.n_targets, dtype=bool)
        self.completed = np.zeros(self.n_targets, dtype=bool)
        self.track_progress = np.zeros(self.n_targets, dtype=np.int64)
        self.memories = [
            DroneMemory(grid_size=grid, n_targets=self.n_targets)
            for _ in range(self.cfg.n_agents)
        ]
        for agent_id, memory in enumerate(self.memories):
            memory.mark_visited([tuple(self.drone_pos[agent_id])])

        self.team_visited = np.zeros((grid, grid), dtype=np.float32)
        for position in self.drone_pos:
            self.team_visited[tuple(position)] = 1.0

        self.last_actions = np.full(self.cfg.n_agents, STAY, dtype=np.int64)
        self.last_boundary_hits = np.zeros(self.cfg.n_agents, dtype=bool)
        self.last_collisions = np.zeros(self.cfg.n_agents, dtype=bool)
        self.last_new_cells = np.zeros(self.cfg.n_agents, dtype=bool)
        self.last_new_observed_cells = np.zeros(self.cfg.n_agents, dtype=np.int64)
        self.last_new_team_observed_cells = 0
        self.last_simultaneous_sensor_overlap_ratio = 0.0
        self.last_tracking_progress = np.zeros(self.cfg.n_agents, dtype=bool)
        self.last_track_progress_targets: list[int | None] = [
            None for _ in range(self.cfg.n_agents)
        ]
        self.last_collision_count = 0
        self.last_reward_parts = self._zero_reward_parts()
        return self._obs_all(), self._info()

    def action_mask(self) -> np.ndarray:
        masks = np.ones((self.cfg.n_agents, self.action_dim), dtype=bool)
        if not self.cfg.use_boundary_action_mask:
            return masks
        for agent_id, position in enumerate(self.drone_pos):
            row, col = int(position[0]), int(position[1])
            for action, (dr, dc) in MOVES.items():
                nr, nc = row + int(dr), col + int(dc)
                masks[agent_id, int(action)] = (
                    0 <= nr < self.cfg.grid_size and 0 <= nc < self.cfg.grid_size
                )
            masks[agent_id, STAY] = True
        return masks

    def step(self, actions):
        actions = np.asarray(actions, dtype=np.int64)
        if actions.shape != (self.cfg.n_agents,):
            raise ValueError(f"Expected {(self.cfg.n_agents,)}, got {actions.shape}")
        if np.any(actions < 0) or np.any(actions >= self.action_dim):
            raise ValueError(f"Invalid actions: {actions}")

        self.t += 1
        self.last_actions = actions.copy()
        self.last_boundary_hits[:] = False
        self.last_collisions[:] = False
        self.last_new_cells[:] = False
        self.last_new_observed_cells[:] = 0
        self.last_new_team_observed_cells = 0
        self.last_simultaneous_sensor_overlap_ratio = 0.0
        self.last_tracking_progress[:] = False
        self.last_track_progress_targets = [None for _ in range(self.cfg.n_agents)]
        self.last_collision_count = 0
        self.last_reward_parts = self._zero_reward_parts()

        reward = 0.0
        for _ in range(self.cfg.n_agents):
            reward += self._add_part("step", self.cfg.step_penalty)

        old_positions = self.drone_pos.copy()
        proposed = old_positions.copy()
        masks = self.action_mask()
        for agent_id, action in enumerate(actions):
            if not bool(masks[agent_id, int(action)]):
                self.last_boundary_hits[agent_id] = True
                reward += self._add_part("boundary", self.cfg.boundary_penalty)
                continue
            dr, dc = MOVES[int(action)]
            proposed[agent_id] = old_positions[agent_id] + np.array([dr, dc])

        collision_agents = self._resolve_collision_agents(old_positions, proposed)
        for agent_id in collision_agents:
            proposed[agent_id] = old_positions[agent_id]
            self.last_collisions[agent_id] = True
            reward += self._add_part("collision", self.cfg.collision_penalty)
        self.last_collision_count = len(collision_agents)
        self.drone_pos = proposed

        visible_sets = [
            set(self._cells_in_radius(position, self.cfg.sensor_radius))
            for position in self.drone_pos
        ]
        visible_union: set[tuple[int, int]] = set().union(*visible_sets)
        visibility_counts: dict[tuple[int, int], int] = {}
        for visible in visible_sets:
            for cell in visible:
                visibility_counts[cell] = visibility_counts.get(cell, 0) + 1
        simultaneous_overlap_cells = sum(
            1 for count in visibility_counts.values() if count > 1
        )
        self.last_simultaneous_sensor_overlap_ratio = (
            float(simultaneous_overlap_cells) / float(max(1, len(visible_union)))
        )
        new_team_cells = [cell for cell in visible_union if self.team_visited[cell] < 0.5]
        self.last_new_team_observed_cells = len(new_team_cells)
        if new_team_cells:
            reward += self._add_part(
                "new_observed",
                min(
                    self.cfg.new_observed_cell_bonus_cap_per_agent * self.cfg.n_agents,
                    self.cfg.new_observed_cell_bonus * float(len(new_team_cells)),
                ),
            )
        for cell in visible_union:
            self.team_visited[cell] = 1.0

        for agent_id, visible in enumerate(visible_sets):
            memory = self.memories[agent_id]
            position = tuple(self.drone_pos[agent_id])
            self.last_new_cells[agent_id] = bool(memory.visited[position] < 0.5)
            reward += self._add_part(
                "new_cell" if self.last_new_cells[agent_id] else "revisit",
                self.cfg.new_cell_bonus if self.last_new_cells[agent_id] else self.cfg.revisit_penalty,
            )
            self.last_new_observed_cells[agent_id] = sum(
                1 for cell in visible if memory.visited[cell] < 0.5
            )
            reward += self._observe_from_agent(agent_id, visible)

        reward += self._assign_and_apply_tracking()
        for agent_id, action in enumerate(actions):
            if int(action) == STAY and not self.last_tracking_progress[agent_id]:
                reward += self._add_part("idle_stay", self.cfg.idle_stay_penalty)

        terminated = bool(np.all(self.completed))
        if terminated:
            reward += self._add_part("all_targets", self.cfg.all_targets_bonus)
        truncated = bool(self.t >= self.cfg.max_steps)
        return self._obs_all(), float(reward), terminated, truncated, self._info()

    def global_state(self) -> np.ndarray:
        """Return the centralized mixer state selected by ``global_state_mode``.

        ``privileged_truth`` preserves the validated 31-value state used by the
        current QMIX baseline. ``memory_union`` has exactly the same dimension but
        replaces hidden target truth by the union of facts present in local UAV
        memories. This makes the later state-information ablation architecture-fair.
        """
        scale = float(max(1, self.cfg.grid_size - 1))
        values: list[float] = []
        for position in self.drone_pos:
            values.extend([float(position[0]) / scale, float(position[1]) / scale])
        denom = float(max(1, self.cfg.track_required))

        if self.cfg.global_state_mode == "privileged_truth":
            for target_id, position in enumerate(self.target_pos):
                values.extend([
                    float(position[0]) / scale,
                    float(position[1]) / scale,
                    float(self.target_values[target_id]) / 2.0,
                    float(self.detected[target_id]),
                    float(self.completed[target_id]),
                    min(1.0, float(self.track_progress[target_id]) / denom),
                ])
        else:
            for target_id in range(self.n_targets):
                records = [
                    memory.known_targets[target_id]
                    for memory in self.memories
                    if target_id in memory.known_targets
                ]
                if not records:
                    values.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                    continue
                most_recent = max(records, key=lambda target: int(target.last_seen_step))
                max_progress = max(int(target.progress) for target in records)
                known_completed = any(bool(target.completed) for target in records)
                values.extend([
                    float(most_recent.pos[0]) / scale,
                    float(most_recent.pos[1]) / scale,
                    float(most_recent.value) / 2.0,
                    1.0,
                    float(known_completed),
                    min(1.0, float(max_progress) / denom),
                ])

        values.append(1.0 - float(self.t) / float(self.cfg.max_steps))
        state = np.asarray(values, dtype=np.float32)
        if state.shape != (self.state_dim,):
            raise RuntimeError(f"state shape {state.shape} != {(self.state_dim,)}")
        return state

    def _sample_unique_positions(self, count: int, forbidden: set[tuple[int, int]]):
        output = []
        while len(output) < int(count):
            p = (
                int(self.rng.integers(self.cfg.grid_size)),
                int(self.rng.integers(self.cfg.grid_size)),
            )
            if p in forbidden:
                continue
            forbidden.add(p)
            output.append(p)
        return output

    def _resolve_collision_agents(self, old: np.ndarray, proposed: np.ndarray) -> set[int]:
        collisions: set[int] = set()
        destinations: dict[tuple[int, int], list[int]] = {}
        for agent_id, destination in enumerate(proposed):
            destinations.setdefault(tuple(destination), []).append(agent_id)
        for agents in destinations.values():
            if len(agents) > 1:
                collisions.update(agents)
        for first in range(self.cfg.n_agents):
            for second in range(first + 1, self.cfg.n_agents):
                if (
                    np.array_equal(proposed[first], old[second])
                    and np.array_equal(proposed[second], old[first])
                    and not np.array_equal(old[first], old[second])
                ):
                    collisions.update((first, second))
        changed = True
        while changed:
            changed = False
            stationary = {
                i for i in range(self.cfg.n_agents)
                if i in collisions or np.array_equal(proposed[i], old[i])
            }
            occupied = {tuple(old[i]): i for i in stationary}
            for i, destination in enumerate(proposed):
                if i in stationary:
                    continue
                if tuple(destination) in occupied:
                    collisions.add(i)
                    changed = True
        return collisions

    def _observe_from_agent(self, agent_id: int, visible: Iterable[tuple[int, int]]) -> float:
        memory = self.memories[agent_id]
        visible = list(visible)
        memory.mark_visited(visible)
        reward = 0.0
        empty_cells: list[tuple[int, int]] = []
        for cell in visible:
            target_ids = [
                i for i, position in enumerate(self.target_pos)
                if tuple(position) == cell
            ]
            if not [i for i in target_ids if not self.completed[i]]:
                empty_cells.append(cell)
            for target_id in target_ids:
                if self.completed[target_id]:
                    memory.add_or_update_target(
                        target_id, cell, int(self.target_values[target_id]), self.t
                    )
                    memory.update_target_progress(
                        target_id, int(self.track_progress[target_id]), True
                    )
                    continue
                if self.rng.random() >= self.cfg.detection_probability:
                    continue
                if not self.detected[target_id]:
                    self.detected[target_id] = True
                    reward += self._add_part(
                        "detect", self._detect_reward(int(self.target_values[target_id]))
                    )
                memory.add_or_update_target(
                    target_id, cell, int(self.target_values[target_id]), self.t
                )
                memory.update_target_progress(
                    target_id,
                    int(self.track_progress[target_id]),
                    bool(self.completed[target_id]),
                )
        memory.suppress_empty_cells(empty_cells, factor=0.2)
        return float(reward)

    def _assign_and_apply_tracking(self) -> float:
        candidates = []
        for agent_id, memory in enumerate(self.memories):
            for target_id, target in memory.known_targets.items():
                if self.completed[target_id] or target.completed:
                    continue
                distance = self._dist(self.drone_pos[agent_id], target.pos)
                if distance <= self.cfg.track_radius:
                    priority = (
                        -int(target.value),
                        -int(self.track_progress[target_id]),
                        int(distance),
                        int(agent_id),
                    )
                    candidates.append((priority, agent_id, int(target_id)))
        candidates.sort(key=lambda item: item[0])
        used_agents: set[int] = set()
        used_targets: set[int] = set()
        reward = 0.0
        for _, agent_id, target_id in candidates:
            if agent_id in used_agents or target_id in used_targets:
                continue
            used_agents.add(agent_id)
            used_targets.add(target_id)
            self.last_tracking_progress[agent_id] = True
            self.last_track_progress_targets[agent_id] = target_id
            self.track_progress[target_id] += 1
            value = int(self.target_values[target_id])
            reward += self._add_part("track_progress", self._track_progress_reward(value))
            if self.track_progress[target_id] >= self.cfg.track_required:
                self.completed[target_id] = True
                reward += self._add_part("complete", self._complete_reward(value))
            self.memories[agent_id].update_target_progress(
                target_id,
                int(self.track_progress[target_id]),
                bool(self.completed[target_id]),
            )
        return float(reward)

    def _obs_all(self) -> np.ndarray:
        return np.stack(
            [self._obs_agent(i) for i in range(self.cfg.n_agents)], axis=0
        ).astype(np.float32)

    def _obs_agent(self, agent_id: int) -> np.ndarray:
        grid = self.cfg.grid_size
        memory = self.memories[agent_id]
        own = np.zeros((grid, grid), dtype=np.float32)
        own[tuple(self.drone_pos[agent_id])] = 1.0
        teammates = np.zeros((grid, grid), dtype=np.float32)
        for other_id, position in enumerate(self.drone_pos):
            if other_id != agent_id and self._dist(
                self.drone_pos[agent_id], position
            ) <= self.cfg.teammate_visibility_radius:
                teammates[tuple(position)] = 1.0
        time = np.full(
            (grid, grid),
            1.0 - float(self.t) / float(self.cfg.max_steps),
            dtype=np.float32,
        )
        channels = [
            own,
            teammates,
            memory.belief,
            memory.known_target_value_map(),
            memory.track_progress_map(self.cfg.track_required),
            memory.completed_map,
            memory.visited,
            time,
        ]
        if self.cfg.include_agent_id_map:
            channels.append(
                np.full(
                    (grid, grid),
                    float(agent_id) / float(max(1, self.cfg.n_agents - 1)),
                    dtype=np.float32,
                )
            )
        return np.stack(channels, axis=0).astype(np.float32)

    def _info(self) -> Dict:
        local_visited = np.stack([m.visited for m in self.memories], axis=0)
        visit_counts = local_visited.sum(axis=0)
        team_cells = max(1, int(np.count_nonzero(visit_counts > 0)))
        overlap_cells = int(np.count_nonzero(visit_counts > 1))
        known_sets = [set(m.known_targets) for m in self.memories]
        union = set().union(*known_sets) if known_sets else set()
        intersection = set.intersection(*known_sets) if known_sets else set()
        knowledge_overlap = float(len(intersection)) / float(len(union)) if union else 1.0
        return {
            "reward_version": self.cfg.reward_version,
            "t": int(self.t),
            "drone_pos": self.drone_pos.copy(),
            "detected": int(self.detected.sum()),
            "completed": int(self.completed.sum()),
            "detected_value": int((self.detected.astype(np.int64) * self.target_values).sum()),
            "completed_value": int((self.completed.astype(np.int64) * self.target_values).sum()),
            "team_coverage_ratio": float(self.team_visited.mean()),
            "local_coverage_ratios": np.asarray([m.visited.mean() for m in self.memories]),
            "mean_local_coverage_ratio": float(np.mean([m.visited.mean() for m in self.memories])),
            "coverage_overlap_ratio": float(overlap_cells) / float(team_cells),
            "known_target_counts": np.asarray([len(m.known_targets) for m in self.memories]),
            "knowledge_overlap_ratio": knowledge_overlap,
            "target_values": self.target_values.copy(),
            "target_pos": self.target_pos.copy(),
            "track_progress": self.track_progress.copy(),
            "completed_flags": self.completed.copy(),
            "detected_flags": self.detected.copy(),
            "last_actions": self.last_actions.copy(),
            "last_action_names": [ACTION_NAMES[int(a)] for a in self.last_actions],
            "last_boundary_hits": self.last_boundary_hits.copy(),
            "last_collisions": self.last_collisions.copy(),
            "last_collision_count": int(self.last_collision_count),
            "last_new_cells": self.last_new_cells.copy(),
            "last_new_observed_cells": self.last_new_observed_cells.copy(),
            "last_new_team_observed_cells": int(self.last_new_team_observed_cells),
            "last_simultaneous_sensor_overlap_ratio": float(
                self.last_simultaneous_sensor_overlap_ratio
            ),
            "global_state_mode": self.cfg.global_state_mode,
            "last_tracking_progress": self.last_tracking_progress.copy(),
            "last_track_progress_targets": list(self.last_track_progress_targets),
            "last_reward_parts": dict(self.last_reward_parts),
        }

    def _zero_reward_parts(self) -> dict[str, float]:
        return {key: 0.0 for key in MULTI_REWARD_PART_KEYS}

    def _add_part(self, key: str, value: float) -> float:
        self.last_reward_parts[key] = self.last_reward_parts.get(key, 0.0) + float(value)
        return float(value)

    def _detect_reward(self, value: int) -> float:
        return self.cfg.detect_value1_bonus if value == 1 else self.cfg.detect_value2_bonus

    def _track_progress_reward(self, value: int) -> float:
        return self.cfg.track_progress_value1_bonus if value == 1 else self.cfg.track_progress_value2_bonus

    def _complete_reward(self, value: int) -> float:
        return self.cfg.complete_value1_bonus if value == 1 else self.cfg.complete_value2_bonus

    def _cells_in_radius(self, center: np.ndarray, radius: int):
        grid = self.cfg.grid_size
        cr, cc = int(center[0]), int(center[1])
        output = []
        for row in range(max(0, cr - radius), min(grid, cr + radius + 1)):
            for col in range(max(0, cc - radius), min(grid, cc + radius + 1)):
                if abs(row - cr) + abs(col - cc) <= radius:
                    output.append((row, col))
        return output

    @staticmethod
    def _dist(first: np.ndarray, second: np.ndarray) -> int:
        return int(abs(int(first[0]) - int(second[0])) + abs(int(first[1]) - int(second[1])))

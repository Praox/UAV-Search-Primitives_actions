from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Tuple

import numpy as np

from uav_search_belief20.actions import ACTION_DIM, ACTION_NAMES, MOVES, STAY
from uav_search_belief20.envs.drone_memory import DroneMemory


REWARD_PART_KEYS: tuple[str, ...] = (
    "step",
    "boundary",
    "new_cell",
    "revisit",
    "new_observed",
    "detect",
    "track_progress",
    "complete",
    "idle_stay",
    "all_targets",
)


@dataclass
class EnvConfig:
    grid_size: int = 20
    n_value1_targets: int = 3
    n_value2_targets: int = 1
    sensor_radius: int = 2
    detection_probability: float = 1.0
    track_radius: int = 1
    track_required: int = 3
    max_steps: int = 150
    seed: int | None = None

    # Experiment metadata.
    reward_version: str = "v3_frontier"
    ablation_name: str = "v3"

    # Observation/action switches used by the single-UAV ablation study.
    use_boundary_action_mask: bool = False
    include_track_progress_map: bool = False

    # Base movement shaping.
    step_penalty: float = -0.005
    new_cell_bonus: float = 0.002
    revisit_penalty: float = 0.0
    new_observed_cell_bonus: float = 0.015
    new_observed_cell_bonus_cap: float = 0.15
    boundary_penalty: float = -0.20
    idle_stay_penalty: float = -0.03

    # Task rewards.
    detect_value1_bonus: float = 0.60
    detect_value2_bonus: float = 1.50
    track_progress_value1_bonus: float = 0.20
    track_progress_value2_bonus: float = 0.60
    complete_value1_bonus: float = 4.0
    complete_value2_bonus: float = 12.0
    all_targets_bonus: float = 5.0

    def reward_dict(self) -> dict:
        return asdict(self)


class PrimitiveSearchEnv:
    """Single-UAV primitive-action search-and-track environment.

    Hidden truth remains inside the environment. The policy receives only the
    UAV's memory maps. Optional switches make the boundary mask and the tracking
    progress channel independently ablatable without duplicating environments.
    """

    action_dim = ACTION_DIM
    action_names = ACTION_NAMES

    def __init__(self, config: EnvConfig | None = None):
        self.cfg = config or EnvConfig()
        self.rng = np.random.default_rng(self.cfg.seed)
        self.observation_channels = 7 if self.cfg.include_track_progress_map else 6
        self.observation_shape = (
            self.observation_channels,
            self.cfg.grid_size,
            self.cfg.grid_size,
        )
        self.reset()

    def reset(self) -> Tuple[np.ndarray, Dict]:
        grid = self.cfg.grid_size
        self.t = 0
        self.drone_pos = np.array(
            [self.rng.integers(grid), self.rng.integers(grid)],
            dtype=np.int64,
        )
        self.target_values = np.array(
            [1] * self.cfg.n_value1_targets + [2] * self.cfg.n_value2_targets,
            dtype=np.int64,
        )
        n_targets = len(self.target_values)

        forbidden = {tuple(self.drone_pos)}
        positions: list[tuple[int, int]] = []
        while len(positions) < n_targets:
            position = (
                int(self.rng.integers(grid)),
                int(self.rng.integers(grid)),
            )
            if position not in forbidden:
                positions.append(position)
                forbidden.add(position)

        self.target_pos = np.array(positions, dtype=np.int64)
        self.detected = np.zeros(n_targets, dtype=bool)
        self.completed = np.zeros(n_targets, dtype=bool)
        self.track_progress = np.zeros(n_targets, dtype=np.int64)
        self.memory = DroneMemory(grid_size=grid, n_targets=n_targets)
        self.memory.mark_visited([tuple(self.drone_pos)])

        self.last_action = STAY
        self.last_boundary_hit = False
        self.last_new_cell = False
        self.last_new_observed_cells = 0
        self.last_track_progress_target: int | None = None
        self.last_tracking_progress = False
        self.last_reward_parts: dict[str, float] = {}
        return self._obs(), self._info()

    def action_mask(self) -> np.ndarray:
        """Return valid primitive actions for the current UAV position.

        In the historical v3 setup every action remains available. For the mask
        ablations, movements that would leave the grid are excluded while STAY is
        always valid.
        """

        mask = np.ones(self.action_dim, dtype=bool)
        if not self.cfg.use_boundary_action_mask:
            return mask

        row, col = int(self.drone_pos[0]), int(self.drone_pos[1])
        for action, (delta_row, delta_col) in MOVES.items():
            next_row = row + int(delta_row)
            next_col = col + int(delta_col)
            mask[int(action)] = (
                0 <= next_row < self.cfg.grid_size
                and 0 <= next_col < self.cfg.grid_size
            )
        mask[STAY] = True
        return mask

    def step(self, action: int):
        action = int(action)
        if not 0 <= action < self.action_dim:
            raise ValueError(f"Invalid action {action}.")

        self.t += 1
        self.last_action = action
        self.last_reward_parts = {}
        self.last_new_observed_cells = 0
        self.last_tracking_progress = False
        self.last_track_progress_target = None

        reward = self._add_part("step", self.cfg.step_penalty)

        delta_row, delta_col = MOVES[action]
        new_position = self.drone_pos + np.array(
            [delta_row, delta_col],
            dtype=np.int64,
        )
        clipped = np.clip(new_position, 0, self.cfg.grid_size - 1)
        self.last_boundary_hit = bool(np.any(clipped != new_position))
        if self.last_boundary_hit:
            # Safety fallback. A correctly masked policy should never reach this
            # branch, but keeping it makes environment failures visible.
            reward += self._add_part("boundary", self.cfg.boundary_penalty)
        self.drone_pos = clipped

        position = tuple(self.drone_pos)
        self.last_new_cell = bool(self.memory.visited[position] < 0.5)
        if self.last_new_cell:
            reward += self._add_part("new_cell", self.cfg.new_cell_bonus)
        else:
            reward += self._add_part("revisit", self.cfg.revisit_penalty)

        reward += self._observe_and_update_memory()
        reward += self._auto_track_update_if_possible()

        if action == STAY and not self.last_tracking_progress:
            reward += self._add_part("idle_stay", self.cfg.idle_stay_penalty)

        terminated = bool(np.all(self.completed))
        if terminated:
            reward += self._add_part("all_targets", self.cfg.all_targets_bonus)
        truncated = bool(self.t >= self.cfg.max_steps)
        return self._obs(), float(reward), terminated, truncated, self._info()

    def _add_part(self, key: str, value: float) -> float:
        self.last_reward_parts[key] = self.last_reward_parts.get(key, 0.0) + float(value)
        return float(value)

    def _detect_reward(self, value: int) -> float:
        return (
            self.cfg.detect_value1_bonus
            if int(value) == 1
            else self.cfg.detect_value2_bonus
        )

    def _track_progress_reward(self, value: int) -> float:
        return (
            self.cfg.track_progress_value1_bonus
            if int(value) == 1
            else self.cfg.track_progress_value2_bonus
        )

    def _complete_reward(self, value: int) -> float:
        return (
            self.cfg.complete_value1_bonus
            if int(value) == 1
            else self.cfg.complete_value2_bonus
        )

    def _observe_and_update_memory(self) -> float:
        reward = 0.0
        visible = self._cells_in_radius(self.drone_pos, self.cfg.sensor_radius)

        newly_observed = sum(
            1 for cell in visible if self.memory.visited[cell] < 0.5
        )
        self.last_new_observed_cells = int(newly_observed)
        if newly_observed > 0:
            exploration_bonus = min(
                self.cfg.new_observed_cell_bonus_cap,
                self.cfg.new_observed_cell_bonus * float(newly_observed),
            )
            reward += self._add_part("new_observed", exploration_bonus)

        self.memory.mark_visited(visible)
        empty_cells: list[tuple[int, int]] = []

        for cell in visible:
            target_ids_here = [
                target_id
                for target_id, target_position in enumerate(self.target_pos)
                if tuple(target_position) == cell and not self.completed[target_id]
            ]
            if not target_ids_here:
                empty_cells.append(cell)
                continue

            for target_id in target_ids_here:
                detected_now = self.rng.random() < self.cfg.detection_probability
                if not detected_now:
                    continue
                if not self.detected[target_id]:
                    self.detected[target_id] = True
                    reward += self._add_part(
                        "detect",
                        self._detect_reward(int(self.target_values[target_id])),
                    )
                self.memory.add_or_update_target(
                    target_id=target_id,
                    pos=cell,
                    value=int(self.target_values[target_id]),
                    step=self.t,
                )

        self.memory.suppress_empty_cells(empty_cells, factor=0.2)
        return float(reward)

    def _auto_track_update_if_possible(self) -> float:
        candidates: list[int] = []
        for target_id, target in self.memory.known_targets.items():
            if target.completed:
                continue
            if self._dist(self.drone_pos, target.pos) <= self.cfg.track_radius:
                candidates.append(int(target_id))

        if not candidates:
            return 0.0

        candidates.sort(
            key=lambda target_id: (
                -int(self.memory.known_targets[target_id].value),
                self._dist(
                    self.drone_pos,
                    self.memory.known_targets[target_id].pos,
                ),
            )
        )
        target_id = int(candidates[0])
        self.last_track_progress_target = target_id
        self.last_tracking_progress = True
        self.track_progress[target_id] += 1

        value = int(self.target_values[target_id])
        reward = self._add_part(
            "track_progress",
            self._track_progress_reward(value),
        )

        if (
            self.track_progress[target_id] >= self.cfg.track_required
            and not self.completed[target_id]
        ):
            self.completed[target_id] = True
            reward += self._add_part("complete", self._complete_reward(value))

        self.memory.update_target_progress(
            target_id,
            int(self.track_progress[target_id]),
            bool(self.completed[target_id]),
        )
        return float(reward)

    def _obs(self) -> np.ndarray:
        grid = self.cfg.grid_size
        drone = np.zeros((grid, grid), dtype=np.float32)
        drone[tuple(self.drone_pos)] = 1.0
        known_target_value = self.memory.known_target_value_map()
        time_remaining = np.full(
            (grid, grid),
            1.0 - float(self.t) / float(self.cfg.max_steps),
            dtype=np.float32,
        )

        channels = [
            drone,
            self.memory.belief,
            known_target_value,
        ]
        if self.cfg.include_track_progress_map:
            channels.append(
                self.memory.track_progress_map(self.cfg.track_required)
            )
        channels.extend(
            [
                self.memory.completed_map,
                self.memory.visited,
                time_remaining,
            ]
        )
        return np.stack(channels, axis=0).astype(np.float32)

    def _info(self) -> Dict:
        completed_value = int(
            (self.completed.astype(np.int64) * self.target_values).sum()
        )
        detected_value = int(
            (self.detected.astype(np.int64) * self.target_values).sum()
        )
        return {
            "reward_version": self.cfg.reward_version,
            "ablation": self.cfg.ablation_name,
            "t": int(self.t),
            "drone_pos": self.drone_pos.copy(),
            "detected": int(self.detected.sum()),
            "completed": int(self.completed.sum()),
            "known_targets": len(self.memory.known_targets),
            "known_uncompleted": sum(
                1 for target in self.memory.known_targets.values()
                if not target.completed
            ),
            "visited_ratio": float(self.memory.visited.mean()),
            "target_values": self.target_values.copy(),
            # Debug/evaluation only; never included in the policy observation.
            "target_pos": self.target_pos.copy(),
            "track_progress": self.track_progress.copy(),
            "last_action": int(self.last_action),
            "last_action_name": ACTION_NAMES[int(self.last_action)],
            "last_boundary_hit": bool(self.last_boundary_hit),
            "last_new_cell": bool(self.last_new_cell),
            "last_new_observed_cells": int(self.last_new_observed_cells),
            "last_sensor_revisit": int(self.last_new_observed_cells) == 0,
            "last_track_progress_target": self.last_track_progress_target,
            "last_tracking_progress": bool(self.last_tracking_progress),
            "last_reward_parts": dict(self.last_reward_parts),
            "completed_value": completed_value,
            "detected_value": detected_value,
        }

    def _cells_in_radius(
        self,
        center: np.ndarray,
        radius: int,
    ) -> list[tuple[int, int]]:
        grid = self.cfg.grid_size
        center_row, center_col = int(center[0]), int(center[1])
        output: list[tuple[int, int]] = []
        for row in range(
            max(0, center_row - radius),
            min(grid, center_row + radius + 1),
        ):
            for col in range(
                max(0, center_col - radius),
                min(grid, center_col + radius + 1),
            ):
                if abs(row - center_row) + abs(col - center_col) <= radius:
                    output.append((row, col))
        return output

    @staticmethod
    def _dist(first: np.ndarray, second: np.ndarray) -> int:
        return int(
            abs(int(first[0]) - int(second[0]))
            + abs(int(first[1]) - int(second[1]))
        )

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from uav_search_belief20.actions import ACTION_DIM, ACTION_NAMES, MOVES, STAY
from uav_search_belief20.envs.drone_memory import DroneMemory


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

    # --- Base movement shaping ---
    step_penalty: float = -0.005

    # Old position-based exploration signal.
    # Keep it very small because sensor coverage is more important than standing on a new cell.
    new_cell_bonus: float = 0.002
    revisit_penalty: float = 0.0

    # New sensor-based exploration signal.
    new_observed_cell_bonus: float = 0.015
    new_observed_cell_bonus_cap: float = 0.15

    # Avoid degenerate wall-hitting policies.
    boundary_penalty: float = -0.20

    # Penalize staying only when it does not help tracking.
    idle_stay_penalty: float = -0.03

    # --- Task rewards ---
    detect_value1_bonus: float = 0.60
    detect_value2_bonus: float = 1.50

    track_progress_value1_bonus: float = 0.15
    track_progress_value2_bonus: float = 0.45

    complete_value1_bonus: float = 3.0
    complete_value2_bonus: float = 10.0
    all_targets_bonus: float = 5.0

class PrimitiveSearchEnv:
    """Single-UAV primitive-action search/track environment.

    Hidden truth remains in the environment. The agent only receives the UAV memory.
    The learned policy directly selects one of: stay, up, down, left, right.
    """

    action_dim = ACTION_DIM
    action_names = ACTION_NAMES
    observation_channels = 6

    def __init__(self, config: EnvConfig | None = None):
        self.cfg = config or EnvConfig()
        self.rng = np.random.default_rng(self.cfg.seed)
        self.observation_shape = (self.observation_channels, self.cfg.grid_size, self.cfg.grid_size)
        self.reset()

    def reset(self) -> Tuple[np.ndarray, Dict]:
        g = self.cfg.grid_size
        self.t = 0
        self.drone_pos = np.array([self.rng.integers(g), self.rng.integers(g)], dtype=np.int64)
        self.target_values = np.array(
            [1] * self.cfg.n_value1_targets + [2] * self.cfg.n_value2_targets,
            dtype=np.int64,
        )
        n_targets = len(self.target_values)

        forbidden = {tuple(self.drone_pos)}
        positions: list[tuple[int, int]] = []
        while len(positions) < n_targets:
            p = (int(self.rng.integers(g)), int(self.rng.integers(g)))
            if p not in forbidden:
                positions.append(p)
                forbidden.add(p)

        self.target_pos = np.array(positions, dtype=np.int64)
        self.detected = np.zeros(n_targets, dtype=bool)
        self.completed = np.zeros(n_targets, dtype=bool)
        self.track_progress = np.zeros(n_targets, dtype=np.int64)
        self.memory = DroneMemory(grid_size=g, n_targets=n_targets)
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
        """All primitive actions are allowed; boundary hits receive a penalty.

        Keeping all actions available is useful because the baseline can learn the cost of
        trying to leave the map. If desired, this can be changed into a strict valid-action
        mask later.
        """
        return np.ones(self.action_dim, dtype=bool)

    def step(self, action: int):
        assert 0 <= int(action) < self.action_dim
        self.t += 1
        self.last_action = int(action)
        self.last_reward_parts = {}
        reward = self._add_part("step", self.cfg.step_penalty)

        dr, dc = MOVES[int(action)]
        new_pos = self.drone_pos + np.array([dr, dc], dtype=np.int64)
        clipped = np.clip(new_pos, 0, self.cfg.grid_size - 1)
        self.last_boundary_hit = bool(np.any(clipped != new_pos))
        if self.last_boundary_hit:
            reward += self._add_part("boundary", self.cfg.boundary_penalty)
        self.drone_pos = clipped

        pos = tuple(self.drone_pos)
        self.last_new_cell = bool(self.memory.visited[pos] < 0.5)

        # Very small position-based shaping.
        # The stronger exploration reward is now based on newly observed sensor cells.
        if self.last_new_cell:
            reward += self._add_part("new_cell", self.cfg.new_cell_bonus)
        else:
            reward += self._add_part("revisit", self.cfg.revisit_penalty)

        reward += self._observe_and_update_memory()
        reward += self._auto_track_update_if_possible()

        # Penalize stay only when it did not advance tracking.
        # This avoids killing the useful "stay near target to complete track" behavior.
        if int(action) == STAY and not self.last_tracking_progress:
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
        return self.cfg.detect_value1_bonus if int(value) == 1 else self.cfg.detect_value2_bonus

    def _track_progress_reward(self, value: int) -> float:
        return self.cfg.track_progress_value1_bonus if int(value) == 1 else self.cfg.track_progress_value2_bonus

    def _complete_reward(self, value: int) -> float:
        return self.cfg.complete_value1_bonus if int(value) == 1 else self.cfg.complete_value2_bonus

    def _observe_and_update_memory(self) -> float:
        reward = 0.0
        visible = self._cells_in_radius(self.drone_pos, self.cfg.sensor_radius)

        # Reward true sensor exploration: how many cells become observed now?
        newly_observed = sum(1 for cell in visible if self.memory.visited[cell] < 0.5)
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
                i for i, p in enumerate(self.target_pos)
                if tuple(p) == cell and not self.completed[i]
            ]
            if not target_ids_here:
                empty_cells.append(cell)
                continue

            for i in target_ids_here:
                detected_now = self.rng.random() < self.cfg.detection_probability
                if detected_now:
                    if not self.detected[i]:
                        self.detected[i] = True
                        r = self._detect_reward(int(self.target_values[i]))
                        reward += self._add_part("detect", r)
                    self.memory.add_or_update_target(
                        target_id=i,
                        pos=cell,
                        value=int(self.target_values[i]),
                        step=self.t,
                    )

        self.memory.suppress_empty_cells(empty_cells, factor=0.2)
        return float(reward)

    def _auto_track_update_if_possible(self) -> float:
        self.last_track_progress_target = None
        self.last_tracking_progress = False
        candidates = []
        for target_id, target in self.memory.known_targets.items():
            if target.completed:
                continue
            if self._dist(self.drone_pos, target.pos) <= self.cfg.track_radius:
                candidates.append(int(target_id))

        if not candidates:
            return 0.0

        candidates.sort(
            key=lambda i: (
                -int(self.memory.known_targets[i].value),
                self._dist(self.drone_pos, self.memory.known_targets[i].pos),
            )
        )
        i = int(candidates[0])
        self.last_track_progress_target = i
        self.track_progress[i] += 1
        self.last_tracking_progress = True
        value = int(self.target_values[i])
        reward = self._add_part("track_progress", self._track_progress_reward(value))

        if self.track_progress[i] >= self.cfg.track_required and not self.completed[i]:
            self.completed[i] = True
            reward += self._add_part("complete", self._complete_reward(value))
        self.memory.update_target_progress(i, int(self.track_progress[i]), bool(self.completed[i]))
        return float(reward)

    def _obs(self) -> np.ndarray:
        g = self.cfg.grid_size
        drone = np.zeros((g, g), dtype=np.float32)
        drone[tuple(self.drone_pos)] = 1.0
        known_target_value = self.memory.known_target_value_map()
        time_remaining = np.full(
            (g, g),
            1.0 - float(self.t) / float(self.cfg.max_steps),
            dtype=np.float32,
        )
        return np.stack(
            [
                drone,
                self.memory.belief,
                known_target_value,
                self.memory.completed_map,
                self.memory.visited,
                time_remaining,
            ],
            axis=0,
        ).astype(np.float32)

    def _info(self) -> Dict:
        completed_value = int((self.completed.astype(np.int64) * self.target_values).sum())
        detected_value = int((self.detected.astype(np.int64) * self.target_values).sum())
        return {
            "t": int(self.t),
            "drone_pos": self.drone_pos.copy(),
            "detected": int(self.detected.sum()),
            "completed": int(self.completed.sum()),
            "known_targets": len(self.memory.known_targets),
            "known_uncompleted": sum(1 for t in self.memory.known_targets.values() if not t.completed),
            "visited_ratio": float(self.memory.visited.mean()),
            "target_values": self.target_values.copy(),
            "target_pos": self.target_pos.copy(),  # debug/evaluation only; not part of obs.
            "track_progress": self.track_progress.copy(),
            "last_action": int(self.last_action),
            "last_action_name": ACTION_NAMES[int(self.last_action)],
            "last_boundary_hit": bool(self.last_boundary_hit),
            "last_new_cell": bool(self.last_new_cell),
            "last_new_observed_cells": int(self.last_new_observed_cells),
            "last_track_progress_target": self.last_track_progress_target,
            "last_tracking_progress": bool(self.last_tracking_progress),
            "last_reward_parts": dict(self.last_reward_parts),
            "completed_value": completed_value,
            "detected_value": detected_value,
        }

    def _cells_in_radius(self, center: np.ndarray, radius: int) -> list[tuple[int, int]]:
        g = self.cfg.grid_size
        cr, cc = int(center[0]), int(center[1])
        out: list[tuple[int, int]] = []
        for r in range(max(0, cr - radius), min(g, cr + radius + 1)):
            for c in range(max(0, cc - radius), min(g, cc + radius + 1)):
                if abs(r - cr) + abs(c - cc) <= radius:
                    out.append((r, c))
        return out

    @staticmethod
    def _dist(a: np.ndarray, b: np.ndarray) -> int:
        return int(abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1])))

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict

import numpy as np

from uav_search_belief20.actions import ACTION_DIM, ACTION_NAMES, MOVES, STAY
from uav_search_belief20.envs.drone_memory import DroneMemory


@dataclass
class MultiDroneEnvConfig:
    """Configuration for the shared-memory multi-UAV primitive-action environment.

    This file is aligned with the validated single-UAV reward v3_frontier:
    - sensor/frontier exploration reward through newly observed cells,
    - stronger tracking/completion rewards,
    - anti-boundary and anti-idle-stay shaping.

    The reward is a TEAM reward. In independent/shared training, every UAV transition
    can receive this same team reward.
    """

    grid_size: int = 20
    n_agents: int = 3
    n_value1_targets: int = 3
    n_value2_targets: int = 1
    sensor_radius: int = 2
    detection_probability: float = 1.0
    track_radius: int = 1
    track_required: int = 3
    max_steps: int = 150
    seed: int | None = None

    # Explicit label so logs/checkpoints cannot be confused with single-agent v1/v2.
    reward_version: str = "v3_frontier"

    # --- Base movement shaping ---
    step_penalty: float = -0.005
    collision_penalty: float = -0.02

    # Position-based shaping is kept tiny. The real exploration signal is sensor-based.
    new_cell_bonus: float = 0.002
    revisit_penalty: float = 0.0

    # Frontier/sensor-based exploration signal.
    # Reward = min(cap, bonus * number_of_newly_observed_sensor_cells)
    new_observed_cell_bonus: float = 0.015
    new_observed_cell_bonus_cap: float = 0.15

    # Avoid degenerate wall-hitting and useless STAY policies.
    boundary_penalty: float = -0.20
    idle_stay_penalty: float = -0.03

    # --- Task rewards, v3_frontier ---
    detect_value1_bonus: float = 0.60
    detect_value2_bonus: float = 1.50

    track_progress_value1_bonus: float = 0.20
    track_progress_value2_bonus: float = 0.60

    complete_value1_bonus: float = 4.0
    complete_value2_bonus: float = 12.0
    all_targets_bonus: float = 5.0

    def reward_dict(self) -> dict:
        """Helper used by training/evaluation scripts to log the exact reward config."""
        return asdict(self)


class MultiDronePrimitiveSearchEnv:
    """Shared-memory multi-UAV primitive-action search-and-track environment.

    This environment is the baseline before QMIX:
    - each UAV has its own local observation,
    - UAVs use primitive actions: stay, up, down, left, right,
    - all UAVs share one team memory/belief map,
    - the environment returns a single TEAM reward,
    - a future QMIX learner can use `global_state()`.

    Observation shape for each UAV:
        (7, grid_size, grid_size)

    Observation channels:
        0. own UAV position
        1. other UAV positions
        2. shared belief map
        3. known target value map
        4. completed target map
        5. shared visited map
        6. time remaining
    """

    action_dim = ACTION_DIM
    action_names = ACTION_NAMES
    observation_channels = 7

    def __init__(self, config: MultiDroneEnvConfig | None = None):
        self.cfg = config or MultiDroneEnvConfig()
        self.rng = np.random.default_rng(self.cfg.seed)
        self.observation_shape = (
            self.observation_channels,
            self.cfg.grid_size,
            self.cfg.grid_size,
        )
        self.reset()

    def reset(self):
        g = self.cfg.grid_size
        self.t = 0

        self.target_values = np.array(
            [1] * self.cfg.n_value1_targets + [2] * self.cfg.n_value2_targets,
            dtype=np.int64,
        )
        n_targets = len(self.target_values)

        forbidden: set[tuple[int, int]] = set()

        drone_positions: list[tuple[int, int]] = []
        while len(drone_positions) < self.cfg.n_agents:
            p = (int(self.rng.integers(g)), int(self.rng.integers(g)))
            if p not in forbidden:
                drone_positions.append(p)
                forbidden.add(p)
        self.drone_pos = np.array(drone_positions, dtype=np.int64)

        target_positions: list[tuple[int, int]] = []
        while len(target_positions) < n_targets:
            p = (int(self.rng.integers(g)), int(self.rng.integers(g)))
            if p not in forbidden:
                target_positions.append(p)
                forbidden.add(p)
        self.target_pos = np.array(target_positions, dtype=np.int64)

        self.detected = np.zeros(n_targets, dtype=bool)
        self.completed = np.zeros(n_targets, dtype=bool)
        self.track_progress = np.zeros(n_targets, dtype=np.int64)

        self.memory = DroneMemory(grid_size=g, n_targets=n_targets)
        self.memory.mark_visited([tuple(p) for p in self.drone_pos])

        self.last_actions = np.full((self.cfg.n_agents,), STAY, dtype=np.int64)
        self.last_boundary_hits = np.zeros((self.cfg.n_agents,), dtype=bool)
        self.last_new_cells = np.zeros((self.cfg.n_agents,), dtype=bool)
        self.last_new_observed_cells = np.zeros((self.cfg.n_agents,), dtype=np.int64)
        self.last_tracking_progress = np.zeros((self.cfg.n_agents,), dtype=bool)
        self.last_track_progress_targets: list[int | None] = [
            None for _ in range(self.cfg.n_agents)
        ]
        self.last_collision_count = 0
        self.last_reward_parts: dict[str, float] = {}

        return self._obs_all(), self._info()

    def action_mask(self) -> np.ndarray:
        """All primitive actions are allowed; bad actions are discouraged by penalties."""
        return np.ones((self.cfg.n_agents, self.action_dim), dtype=bool)

    def step(self, actions):
        actions = np.asarray(actions, dtype=np.int64)
        if actions.shape != (self.cfg.n_agents,):
            raise ValueError(f"Expected actions shape {(self.cfg.n_agents,)}, got {actions.shape}.")

        self.t += 1
        self.last_actions = actions.copy()
        self.last_boundary_hits[:] = False
        self.last_new_cells[:] = False
        self.last_new_observed_cells[:] = 0
        self.last_tracking_progress[:] = False
        self.last_track_progress_targets = [None for _ in range(self.cfg.n_agents)]
        self.last_collision_count = 0
        self.last_reward_parts = {}

        reward = 0.0

        # Move all UAVs first.
        for i, action in enumerate(actions):
            reward += self._add_part("step", self.cfg.step_penalty)

            dr, dc = MOVES[int(action)]
            new_pos = self.drone_pos[i] + np.array([dr, dc], dtype=np.int64)
            clipped = np.clip(new_pos, 0, self.cfg.grid_size - 1)

            if np.any(clipped != new_pos):
                self.last_boundary_hits[i] = True
                reward += self._add_part("boundary", self.cfg.boundary_penalty)

            self.drone_pos[i] = clipped

        # Same-cell collision penalty. We do not block movement.
        # Count every extra UAV landing on an already occupied cell.
        seen: set[tuple[int, int]] = set()
        for p in map(tuple, self.drone_pos):
            if p in seen:
                self.last_collision_count += 1
                reward += self._add_part("collision", self.cfg.collision_penalty)
            seen.add(p)

        # Then observe and auto-track from each UAV.
        # Sequential update is intentional: if UAV 0 observes a cell first, UAV 1 will
        # not receive a new-observed bonus for observing the exact same already-marked cell.
        for i in range(self.cfg.n_agents):
            pos = tuple(self.drone_pos[i])
            self.last_new_cells[i] = bool(self.memory.visited[pos] < 0.5)

            if self.last_new_cells[i]:
                reward += self._add_part("new_cell", self.cfg.new_cell_bonus)
            else:
                reward += self._add_part("revisit", self.cfg.revisit_penalty)

            reward += self._observe_from_agent(i)
            reward += self._auto_track_from_agent(i)

            if int(actions[i]) == STAY and not self.last_tracking_progress[i]:
                reward += self._add_part("idle_stay", self.cfg.idle_stay_penalty)

        terminated = bool(np.all(self.completed))
        if terminated:
            reward += self._add_part("all_targets", self.cfg.all_targets_bonus)

        truncated = bool(self.t >= self.cfg.max_steps)
        return self._obs_all(), float(reward), terminated, truncated, self._info()

    def _add_part(self, key: str, value: float) -> float:
        self.last_reward_parts[key] = self.last_reward_parts.get(key, 0.0) + float(value)
        return float(value)

    def _detect_reward(self, value: int) -> float:
        return self.cfg.detect_value1_bonus if int(value) == 1 else self.cfg.detect_value2_bonus

    def _track_progress_reward(self, value: int) -> float:
        return self.cfg.track_progress_value1_bonus if int(value) == 1 else self.cfg.track_progress_value2_bonus

    def _complete_reward(self, value: int) -> float:
        return self.cfg.complete_value1_bonus if int(value) == 1 else self.cfg.complete_value2_bonus

    def _observe_from_agent(self, agent_id: int) -> float:
        reward = 0.0
        visible = self._cells_in_radius(self.drone_pos[agent_id], self.cfg.sensor_radius)

        newly_observed = sum(1 for cell in visible if self.memory.visited[cell] < 0.5)
        self.last_new_observed_cells[agent_id] = int(newly_observed)

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

            for target_id in target_ids_here:
                detected_now = self.rng.random() < self.cfg.detection_probability
                if detected_now:
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

    def _auto_track_from_agent(self, agent_id: int) -> float:
        candidates: list[int] = []

        for target_id, target in self.memory.known_targets.items():
            if target.completed:
                continue
            if self._dist(self.drone_pos[agent_id], target.pos) <= self.cfg.track_radius:
                candidates.append(int(target_id))

        if not candidates:
            return 0.0

        candidates.sort(
            key=lambda i: (
                -int(self.memory.known_targets[i].value),
                self._dist(self.drone_pos[agent_id], self.memory.known_targets[i].pos),
            )
        )

        i = int(candidates[0])
        self.last_track_progress_targets[agent_id] = i
        self.last_tracking_progress[agent_id] = True
        self.track_progress[i] += 1

        value = int(self.target_values[i])
        reward = self._add_part("track_progress", self._track_progress_reward(value))

        if self.track_progress[i] >= self.cfg.track_required and not self.completed[i]:
            self.completed[i] = True
            reward += self._add_part("complete", self._complete_reward(value))

        self.memory.update_target_progress(
            i,
            int(self.track_progress[i]),
            bool(self.completed[i]),
        )
        return float(reward)

    def _obs_all(self) -> np.ndarray:
        return np.stack(
            [self._obs_agent(i) for i in range(self.cfg.n_agents)],
            axis=0,
        ).astype(np.float32)

    def _obs_agent(self, agent_id: int) -> np.ndarray:
        g = self.cfg.grid_size

        own = np.zeros((g, g), dtype=np.float32)
        own[tuple(self.drone_pos[agent_id])] = 1.0

        others = np.zeros((g, g), dtype=np.float32)
        for j, p in enumerate(self.drone_pos):
            if j != agent_id:
                others[tuple(p)] = 1.0

        time_remaining = np.full(
            (g, g),
            1.0 - float(self.t) / float(self.cfg.max_steps),
            dtype=np.float32,
        )

        return np.stack(
            [
                own,
                others,
                self.memory.belief,
                self.memory.known_target_value_map(),
                self.memory.completed_map,
                self.memory.visited,
                time_remaining,
            ],
            axis=0,
        ).astype(np.float32)

    def global_state(self) -> np.ndarray:
        """Compact global state for a future QMIX mixer input."""
        return self._obs_all().reshape(-1).astype(np.float32)

    def _info(self) -> Dict:
        completed_value = int((self.completed.astype(np.int64) * self.target_values).sum())
        detected_value = int((self.detected.astype(np.int64) * self.target_values).sum())

        return {
            "reward_version": self.cfg.reward_version,
            "t": int(self.t),
            "drone_pos": self.drone_pos.copy(),
            "detected": int(self.detected.sum()),
            "completed": int(self.completed.sum()),
            "known_targets": len(self.memory.known_targets),
            "known_uncompleted": sum(
                1 for t in self.memory.known_targets.values() if not t.completed
            ),
            "visited_ratio": float(self.memory.visited.mean()),
            "target_values": self.target_values.copy(),
            "target_pos": self.target_pos.copy(),  # debug/evaluation only; not part of obs.
            "track_progress": self.track_progress.copy(),
            "last_actions": self.last_actions.copy(),
            "last_action_names": [ACTION_NAMES[int(a)] for a in self.last_actions],
            "last_boundary_hits": self.last_boundary_hits.copy(),
            "last_new_cells": self.last_new_cells.copy(),
            "last_new_observed_cells": self.last_new_observed_cells.copy(),
            "last_sensor_revisits": self.last_new_observed_cells == 0,
            "last_tracking_progress": self.last_tracking_progress.copy(),
            "last_track_progress_targets": list(self.last_track_progress_targets),
            "last_collision_count": int(self.last_collision_count),
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
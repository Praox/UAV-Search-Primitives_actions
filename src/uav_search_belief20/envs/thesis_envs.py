from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from uav_search_belief20.actions import STAY
from uav_search_belief20.envs.multi_drone_local_env import (
    MultiDroneLocalEnvConfig,
    MultiDroneLocalMemoryEnv,
)
from uav_search_belief20.envs.primitive_search_env import (
    EnvConfig,
    PrimitiveSearchEnv,
)


@dataclass
class ThesisEnvConfig(EnvConfig):
    """Single-UAV thesis environment settings.

    ``legacy`` reproduces the original reward. ``task_potential`` keeps only
    mission/safety terms and adds potential-based shaping.

    ``observation_frame='egocentric'`` preserves the original observation
    dimensions while expressing every spatial channel relative to the UAV.
    """

    reward_mode: str = "task_potential"
    tracking_requires_stay: bool = True
    track_progress_decay: int = 1

    # Phi(s) = 10 C(s) + D(s) + 2 P(s)
    coverage_potential_scale: float = 10.0
    detection_potential_scale: float = 1.0
    progress_potential_scale: float = 2.0
    shaping_gamma: float = 0.99

    observation_frame: str = "egocentric"
    strict_action_mask: bool = True

    # Diagnostic overfitting mode. Every reset recreates the exact same
    # spawn, target layout and simulator random-number sequence.
    fixed_scenario: bool = False
    scenario_seed: int = 12_345


@dataclass
class ThesisMultiEnvConfig(MultiDroneLocalEnvConfig):
    """Multi-UAV thesis environment settings."""

    reward_mode: str = "task_potential"
    tracking_requires_stay: bool = True
    track_progress_decay: int = 1

    # Phi(s) = 10 C(s) + D(s) + 2 P(s)
    coverage_potential_scale: float = 10.0
    detection_potential_scale: float = 1.0
    progress_potential_scale: float = 2.0
    shaping_gamma: float = 0.99

    observation_frame: str = "egocentric"
    strict_action_mask: bool = True

    # Diagnostic overfitting mode. Every reset recreates the exact same
    # multi-UAV spawns, target layout and simulator RNG sequence.
    fixed_scenario: bool = False
    scenario_seed: int = 12_345


def _egocentric_map(
    source: np.ndarray,
    center: np.ndarray,
    output_size: int,
    *,
    fill_value: float = 0.0,
) -> np.ndarray:
    """Translate a global 2-D map into a UAV-centred map.

    No circular wrapping is used. Cells outside the physical environment are
    filled with ``fill_value`` and identified separately by the ego/boundary
    channel.
    """

    source = np.asarray(source, dtype=np.float32)
    if source.ndim != 2:
        raise ValueError(f"Expected a 2-D map, got {source.shape}.")
    if output_size <= 0:
        raise ValueError("output_size must be positive.")

    source_height, source_width = source.shape
    anchor = int(output_size) // 2
    center_row, center_col = int(center[0]), int(center[1])

    global_top = center_row - anchor
    global_left = center_col - anchor
    global_bottom = global_top + int(output_size)
    global_right = global_left + int(output_size)

    source_row_start = max(0, global_top)
    source_row_end = min(source_height, global_bottom)
    source_col_start = max(0, global_left)
    source_col_end = min(source_width, global_right)

    output = np.full(
        (output_size, output_size),
        float(fill_value),
        dtype=np.float32,
    )

    if (
        source_row_start >= source_row_end
        or source_col_start >= source_col_end
    ):
        return output

    output_row_start = source_row_start - global_top
    output_col_start = source_col_start - global_left
    output_row_end = output_row_start + (
        source_row_end - source_row_start
    )
    output_col_end = output_col_start + (
        source_col_end - source_col_start
    )

    output[
        output_row_start:output_row_end,
        output_col_start:output_col_end,
    ] = source[
        source_row_start:source_row_end,
        source_col_start:source_col_end,
    ]
    return output


def _ego_boundary_map(center: np.ndarray, grid_size: int) -> np.ndarray:
    """Return +1 at the UAV, -1 outside the world and 0 on valid cells."""

    valid_global = np.ones((grid_size, grid_size), dtype=np.float32)
    valid_local = _egocentric_map(
        valid_global,
        center,
        grid_size,
        fill_value=0.0,
    )
    output = np.where(valid_local > 0.5, 0.0, -1.0).astype(np.float32)
    anchor = int(grid_size) // 2
    output[anchor, anchor] = 1.0
    return output


class _TaskPotentialMixin:
    """Shared reward transformation for the corrected thesis environments."""

    cfg: ThesisEnvConfig | ThesisMultiEnvConfig
    last_reward_parts: dict[str, float]

    def _potential(self) -> float:  # pragma: no cover - subclass contract
        raise NotImplementedError

    def _task_reward_from_legacy_parts(
        self,
        parts: dict[str, float],
    ) -> float:
        # Detection, progress and coverage are represented by the potential.
        # Completion and all-target completion remain true task rewards.
        keep = (
            "step",
            "boundary",
            "collision",
            "complete",
            "all_targets",
        )
        return float(sum(float(parts.get(key, 0.0)) for key in keep))

    def _replace_reward_with_task_potential(
        self,
        *,
        potential_before: float,
        terminated: bool,
        truncated: bool,
    ) -> float:
        """Apply r' = r_task + gamma Phi(s') - Phi(s).

        Terminal and time-limit states use Phi=0. This preserves the standard
        episodic potential-shaping contract and makes the shaping terms
        telescope over a complete episode.
        """

        legacy_parts = dict(self.last_reward_parts)
        task_reward = self._task_reward_from_legacy_parts(legacy_parts)
        potential_after = 0.0 if (terminated or truncated) else self._potential()
        shaping = (
            float(self.cfg.shaping_gamma) * potential_after
            - float(potential_before)
        )

        kept_keys = (
            "step",
            "boundary",
            "collision",
            "complete",
            "all_targets",
        )
        corrected_parts = {
            key: float(legacy_parts.get(key, 0.0))
            for key in kept_keys
            if key in legacy_parts
        }
        corrected_parts["potential"] = float(shaping)
        self.last_reward_parts = corrected_parts
        return float(task_reward + shaping)


class ThesisPrimitiveSearchEnv(_TaskPotentialMixin, PrimitiveSearchEnv):
    """Corrected single-UAV search-and-track environment.

    Scientific contracts:
    1. negative sensing uses the configured detection probability;
    2. tracking requires STAY and otherwise decays;
    3. reward uses task terms plus potential shaping;
    4. spatial observations may be UAV-centred without changing their shape;
    5. invalid masked actions fail fast in thesis experiments.
    """

    cfg: ThesisEnvConfig

    def __init__(self, config: ThesisEnvConfig | None = None):
        config = config or ThesisEnvConfig(
            use_boundary_action_mask=True,
            include_track_progress_map=True,
        )
        if config.reward_mode not in {"legacy", "task_potential"}:
            raise ValueError(
                "reward_mode must be 'legacy' or 'task_potential'."
            )
        if config.track_progress_decay < 0:
            raise ValueError("track_progress_decay must be non-negative.")
        if config.observation_frame not in {"global", "egocentric"}:
            raise ValueError(
                "observation_frame must be 'global' or 'egocentric'."
            )
        super().__init__(config)

    def reset(self):
        if self.cfg.fixed_scenario:
            # PrimitiveSearchEnv.reset() draws the spawn and targets from rng.
            self.rng = np.random.default_rng(int(self.cfg.scenario_seed))
        return super().reset()

    def step(self, action: int):
        action = int(action)
        if not 0 <= action < self.action_dim:
            raise ValueError(f"Invalid action index {action}.")

        if self.cfg.strict_action_mask:
            mask = np.asarray(self.action_mask(), dtype=bool)
            if not bool(mask[action]):
                raise RuntimeError(
                    f"Masked action {action} selected at "
                    f"position {tuple(self.drone_pos)}."
                )

        potential_before = self._potential()
        obs, reward, terminated, truncated, _ = super().step(action)
        if self.cfg.reward_mode == "task_potential":
            reward = self._replace_reward_with_task_potential(
                potential_before=potential_before,
                terminated=bool(terminated),
                truncated=bool(truncated),
            )
        return obs, float(reward), terminated, truncated, self._info()

    def _obs(self) -> np.ndarray:
        if self.cfg.observation_frame == "global":
            return super()._obs()

        grid = int(self.cfg.grid_size)
        center = self.drone_pos

        channels = [
            _ego_boundary_map(center, grid),
            _egocentric_map(self.memory.belief, center, grid),
            _egocentric_map(
                self.memory.known_target_value_map(),
                center,
                grid,
            ),
        ]

        if self.cfg.include_track_progress_map:
            channels.append(
                _egocentric_map(
                    self.memory.track_progress_map(
                        self.cfg.track_required
                    ),
                    center,
                    grid,
                )
            )

        channels.extend(
            [
                _egocentric_map(
                    self.memory.completed_map,
                    center,
                    grid,
                ),
                _egocentric_map(
                    self.memory.visited,
                    center,
                    grid,
                ),
                np.full(
                    (grid, grid),
                    1.0 - float(self.t) / float(self.cfg.max_steps),
                    dtype=np.float32,
                ),
            ]
        )

        observation = np.stack(channels, axis=0).astype(np.float32)
        if observation.shape != self.observation_shape:
            raise RuntimeError(
                f"Observation shape {observation.shape} "
                f"!= expected {self.observation_shape}."
            )
        return observation

    def _potential(self) -> float:
        coverage = float(self.memory.visited.mean())
        detected_value = float(
            (
                self.detected.astype(np.float32)
                * self.target_values
            ).sum()
        )
        progress = float(
            np.sum(
                self.target_values
                * np.minimum(
                    1.0,
                    self.track_progress
                    / float(max(1, self.cfg.track_required)),
                )
            )
        )
        return float(
            self.cfg.coverage_potential_scale * coverage
            + self.cfg.detection_potential_scale * detected_value
            + self.cfg.progress_potential_scale * progress
        )

    def _observe_and_update_memory(self) -> float:
        """Bayes-consistent negative sensing for a no-false-positive sensor."""

        reward = 0.0
        visible = self._cells_in_radius(
            self.drone_pos,
            self.cfg.sensor_radius,
        )
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
                if tuple(target_position) == cell
                and not self.completed[target_id]
            ]
            if not target_ids_here:
                empty_cells.append(cell)
                continue

            any_detection = False
            for target_id in target_ids_here:
                detected_now = (
                    self.rng.random()
                    < self.cfg.detection_probability
                )
                if not detected_now:
                    continue
                any_detection = True
                if not self.detected[target_id]:
                    self.detected[target_id] = True
                    reward += self._add_part(
                        "detect",
                        self._detect_reward(
                            int(self.target_values[target_id])
                        ),
                    )
                self.memory.add_or_update_target(
                    target_id=target_id,
                    pos=cell,
                    value=int(self.target_values[target_id]),
                    step=self.t,
                )

            # "Nothing detected" is evidence with likelihood 1 - p_D.
            if not any_detection:
                empty_cells.append(cell)

        likelihood_no_detection = max(
            1e-6,
            1.0 - float(self.cfg.detection_probability),
        )
        self.memory.suppress_empty_cells(
            empty_cells,
            factor=likelihood_no_detection,
        )
        return float(reward)

    def _decay_untracked_targets(
        self,
        tracked_target: int | None,
    ) -> None:
        decay = int(self.cfg.track_progress_decay)
        if decay <= 0:
            return

        for target_id in range(len(self.track_progress)):
            if self.completed[target_id] or target_id == tracked_target:
                continue
            old = int(self.track_progress[target_id])
            new = max(0, old - decay)
            if new == old:
                continue
            self.track_progress[target_id] = new
            self.memory.update_target_progress(target_id, new, False)

    def _auto_track_update_if_possible(self) -> float:
        if (
            self.cfg.tracking_requires_stay
            and int(self.last_action) != STAY
        ):
            self._decay_untracked_targets(None)
            return 0.0

        candidates: list[int] = []
        for target_id, target in self.memory.known_targets.items():
            if target.completed:
                continue
            if (
                self._dist(self.drone_pos, target.pos)
                <= self.cfg.track_radius
            ):
                candidates.append(int(target_id))

        if not candidates:
            self._decay_untracked_targets(None)
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
            reward += self._add_part(
                "complete",
                self._complete_reward(value),
            )

        self.memory.update_target_progress(
            target_id,
            int(self.track_progress[target_id]),
            bool(self.completed[target_id]),
        )
        self._decay_untracked_targets(target_id)
        return float(reward)


class ThesisMultiDroneLocalMemoryEnv(
    _TaskPotentialMixin,
    MultiDroneLocalMemoryEnv,
):
    """Corrected local-memory multi-UAV thesis environment."""

    cfg: ThesisMultiEnvConfig

    def __init__(self, config: ThesisMultiEnvConfig | None = None):
        config = config or ThesisMultiEnvConfig()
        if config.reward_mode not in {"legacy", "task_potential"}:
            raise ValueError(
                "reward_mode must be 'legacy' or 'task_potential'."
            )
        if config.track_progress_decay < 0:
            raise ValueError("track_progress_decay must be non-negative.")
        if config.observation_frame not in {"global", "egocentric"}:
            raise ValueError(
                "observation_frame must be 'global' or 'egocentric'."
            )
        super().__init__(config)

    def reset(self):
        if self.cfg.fixed_scenario:
            # MultiDroneLocalMemoryEnv.reset() draws all positions from rng.
            self.rng = np.random.default_rng(int(self.cfg.scenario_seed))
        return super().reset()

    def step(self, actions):
        actions = np.asarray(actions, dtype=np.int64)
        expected = (self.cfg.n_agents,)
        if actions.shape != expected:
            raise ValueError(
                f"Expected actions shape {expected}, got {actions.shape}."
            )
        if np.any(actions < 0) or np.any(actions >= self.action_dim):
            raise ValueError(f"Invalid actions: {actions}.")

        if self.cfg.strict_action_mask:
            masks = np.asarray(self.action_mask(), dtype=bool)
            invalid = [
                (agent_id, int(action))
                for agent_id, action in enumerate(actions)
                if not bool(masks[agent_id, int(action)])
            ]
            if invalid:
                raise RuntimeError(
                    f"Masked multi-agent actions selected: {invalid}."
                )

        potential_before = self._potential()
        obs, reward, terminated, truncated, _ = super().step(actions)
        if self.cfg.reward_mode == "task_potential":
            reward = self._replace_reward_with_task_potential(
                potential_before=potential_before,
                terminated=bool(terminated),
                truncated=bool(truncated),
            )
        return obs, float(reward), terminated, truncated, self._info()

    def _obs_agent(self, agent_id: int) -> np.ndarray:
        if self.cfg.observation_frame == "global":
            return super()._obs_agent(agent_id)

        grid = int(self.cfg.grid_size)
        center = self.drone_pos[agent_id]
        memory = self.memories[agent_id]

        teammates_global = np.zeros(
            (grid, grid),
            dtype=np.float32,
        )
        for other_id, position in enumerate(self.drone_pos):
            if other_id == agent_id:
                continue
            if (
                self._dist(center, position)
                <= self.cfg.teammate_visibility_radius
            ):
                teammates_global[tuple(position)] = 1.0

        channels = [
            _ego_boundary_map(center, grid),
            _egocentric_map(
                teammates_global,
                center,
                grid,
            ),
            _egocentric_map(memory.belief, center, grid),
            _egocentric_map(
                memory.known_target_value_map(),
                center,
                grid,
            ),
            _egocentric_map(
                memory.track_progress_map(
                    self.cfg.track_required
                ),
                center,
                grid,
            ),
            _egocentric_map(
                memory.completed_map,
                center,
                grid,
            ),
            _egocentric_map(
                memory.visited,
                center,
                grid,
            ),
            np.full(
                (grid, grid),
                1.0 - float(self.t) / float(self.cfg.max_steps),
                dtype=np.float32,
            ),
        ]

        if self.cfg.include_agent_id_map:
            channels.append(
                np.full(
                    (grid, grid),
                    float(agent_id)
                    / float(max(1, self.cfg.n_agents - 1)),
                    dtype=np.float32,
                )
            )

        observation = np.stack(channels, axis=0).astype(np.float32)
        if observation.shape != self.observation_shape:
            raise RuntimeError(
                f"Observation shape {observation.shape} "
                f"!= expected {self.observation_shape}."
            )
        return observation

    def _potential(self) -> float:
        coverage = float(self.team_visited.mean())
        detected_value = float(
            (
                self.detected.astype(np.float32)
                * self.target_values
            ).sum()
        )
        progress = float(
            np.sum(
                self.target_values
                * np.minimum(
                    1.0,
                    self.track_progress
                    / float(max(1, self.cfg.track_required)),
                )
            )
        )
        return float(
            self.cfg.coverage_potential_scale * coverage
            + self.cfg.detection_potential_scale * detected_value
            + self.cfg.progress_potential_scale * progress
        )

    def _observe_from_agent(
        self,
        agent_id: int,
        visible: Iterable[tuple[int, int]],
    ) -> float:
        memory = self.memories[agent_id]
        visible = list(visible)
        memory.mark_visited(visible)
        reward = 0.0
        negative_cells: list[tuple[int, int]] = []

        for cell in visible:
            target_ids = [
                index
                for index, position in enumerate(self.target_pos)
                if tuple(position) == cell
            ]
            unfinished = [
                index
                for index in target_ids
                if not self.completed[index]
            ]
            if not unfinished:
                negative_cells.append(cell)

            any_detection = False
            for target_id in target_ids:
                if self.completed[target_id]:
                    memory.add_or_update_target(
                        target_id,
                        cell,
                        int(self.target_values[target_id]),
                        self.t,
                    )
                    memory.update_target_progress(
                        target_id,
                        int(self.track_progress[target_id]),
                        True,
                    )
                    continue

                if self.rng.random() >= self.cfg.detection_probability:
                    continue

                any_detection = True
                if not self.detected[target_id]:
                    self.detected[target_id] = True
                    reward += self._add_part(
                        "detect",
                        self._detect_reward(
                            int(self.target_values[target_id])
                        ),
                    )

                memory.add_or_update_target(
                    target_id,
                    cell,
                    int(self.target_values[target_id]),
                    self.t,
                )
                memory.update_target_progress(
                    target_id,
                    int(self.track_progress[target_id]),
                    bool(self.completed[target_id]),
                )

            if unfinished and not any_detection:
                negative_cells.append(cell)

        likelihood_no_detection = max(
            1e-6,
            1.0 - float(self.cfg.detection_probability),
        )
        memory.suppress_empty_cells(
            negative_cells,
            factor=likelihood_no_detection,
        )
        return float(reward)

    def _update_all_known_progress(
        self,
        target_id: int,
        progress: int,
        completed: bool,
    ) -> None:
        for memory in self.memories:
            memory.update_target_progress(
                target_id,
                progress,
                completed,
            )

    def _decay_untracked_targets(
        self,
        tracked_targets: set[int],
    ) -> None:
        decay = int(self.cfg.track_progress_decay)
        if decay <= 0:
            return

        for target_id in range(self.n_targets):
            if (
                self.completed[target_id]
                or target_id in tracked_targets
            ):
                continue
            old = int(self.track_progress[target_id])
            new = max(0, old - decay)
            if new == old:
                continue
            self.track_progress[target_id] = new
            self._update_all_known_progress(
                target_id,
                new,
                False,
            )

    def _assign_and_apply_tracking(self) -> float:
        candidates = []

        for agent_id, memory in enumerate(self.memories):
            if (
                self.cfg.tracking_requires_stay
                and int(self.last_actions[agent_id]) != STAY
            ):
                continue

            for target_id, target in memory.known_targets.items():
                if self.completed[target_id] or target.completed:
                    continue

                distance = self._dist(
                    self.drone_pos[agent_id],
                    target.pos,
                )
                if distance <= self.cfg.track_radius:
                    priority = (
                        -int(target.value),
                        -int(self.track_progress[target_id]),
                        int(distance),
                        int(agent_id),
                    )
                    candidates.append(
                        (priority, agent_id, int(target_id))
                    )

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
            reward += self._add_part(
                "track_progress",
                self._track_progress_reward(value),
            )

            if self.track_progress[target_id] >= self.cfg.track_required:
                self.completed[target_id] = True
                reward += self._add_part(
                    "complete",
                    self._complete_reward(value),
                )

            self._update_all_known_progress(
                target_id,
                int(self.track_progress[target_id]),
                bool(self.completed[target_id]),
            )

        self._decay_untracked_targets(used_targets)
        return float(reward)
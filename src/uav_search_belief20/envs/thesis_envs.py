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
    mission/safety terms and adds policy-invariant potential shaping.
    """

    reward_mode: str = "task_potential"
    tracking_requires_stay: bool = True
    track_progress_decay: int = 1
    coverage_potential_scale: float = 5.0
    detection_potential_scale: float = 1.0
    progress_potential_scale: float = 1.0
    shaping_gamma: float = 0.99


@dataclass
class ThesisMultiEnvConfig(MultiDroneLocalEnvConfig):
    reward_mode: str = "task_potential"
    tracking_requires_stay: bool = True
    track_progress_decay: int = 1
    coverage_potential_scale: float = 5.0
    detection_potential_scale: float = 1.0
    progress_potential_scale: float = 1.0
    shaping_gamma: float = 0.99


class _TaskPotentialMixin:
    """Shared reward transformation for the corrected environments."""

    cfg: ThesisEnvConfig | ThesisMultiEnvConfig
    last_reward_parts: dict[str, float]

    def _potential(self) -> float:  # pragma: no cover - implemented by subclasses
        raise NotImplementedError

    def _task_reward_from_legacy_parts(self, parts: dict[str, float]) -> float:
        # Completion is the mission objective. Step cost approximates energy/time;
        # boundary and collision terms are safety costs.
        keep = ("step", "boundary", "collision", "complete")
        return float(sum(float(parts.get(key, 0.0)) for key in keep))

    def _replace_reward_with_task_potential(
        self,
        *,
        potential_before: float,
        terminated: bool,
        truncated: bool,
    ) -> float:
        legacy_parts = dict(self.last_reward_parts)
        task_reward = self._task_reward_from_legacy_parts(legacy_parts)
        potential_after = 0.0 if (terminated or truncated) else self._potential()
        shaping = (
            float(self.cfg.shaping_gamma) * potential_after
            - potential_before
        )

        corrected_parts = {
            key: float(legacy_parts.get(key, 0.0))
            for key in ("step", "boundary", "collision", "complete")
            if key in legacy_parts
        }
        corrected_parts["potential"] = float(shaping)
        self.last_reward_parts = corrected_parts
        return float(task_reward + shaping)


class ThesisPrimitiveSearchEnv(_TaskPotentialMixin, PrimitiveSearchEnv):
    """Thin correction layer over ``PrimitiveSearchEnv``.

    Changes only three scientific contracts:

    1. negative observations use the configured detection probability;
    2. tracking requires STAY and otherwise decays;
    3. an optional completion-centric potential-shaped reward is available.
    """

    cfg: ThesisEnvConfig

    def __init__(self, config: ThesisEnvConfig | None = None):
        config = config or ThesisEnvConfig(
            use_boundary_action_mask=True,
            include_track_progress_map=True,
        )
        if config.reward_mode not in {"legacy", "task_potential"}:
            raise ValueError("reward_mode must be 'legacy' or 'task_potential'")
        if config.track_progress_decay < 0:
            raise ValueError("track_progress_decay must be non-negative")
        super().__init__(config)

    def step(self, action: int):
        potential_before = self._potential()
        obs, reward, terminated, truncated, _ = super().step(action)
        if self.cfg.reward_mode == "task_potential":
            reward = self._replace_reward_with_task_potential(
                potential_before=potential_before,
                terminated=bool(terminated),
                truncated=bool(truncated),
            )
        return obs, float(reward), terminated, truncated, self._info()

    def _potential(self) -> float:
        coverage = float(self.memory.visited.mean())
        detected_value = float(
            (self.detected.astype(np.float32) * self.target_values).sum()
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
        """Bayes-consistent negative sensing for a no-false-positive sensor.

        After no detection in a visible cell, its target probability is
        multiplied by P(no detection | target there) = 1 - p_D.
        """

        reward = 0.0
        visible = self._cells_in_radius(
            self.drone_pos, self.cfg.sensor_radius
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
                    self.rng.random() < self.cfg.detection_probability
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
            # A missed detection is evidence, but weaker than observing a truly
            # empty cell. Both use the same likelihood factor under this sensor
            # model because the cell-level observation is "nothing detected".
            if not any_detection:
                empty_cells.append(cell)

        likelihood_no_detection = max(
            1e-6, 1.0 - float(self.cfg.detection_probability)
        )
        self.memory.suppress_empty_cells(
            empty_cells,
            factor=likelihood_no_detection,
        )
        return float(reward)

    def _decay_untracked_targets(self, tracked_target: int | None) -> None:
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
        if self.cfg.tracking_requires_stay and int(self.last_action) != STAY:
            self._decay_untracked_targets(None)
            return 0.0

        candidates: list[int] = []
        for target_id, target in self.memory.known_targets.items():
            if target.completed:
                continue
            if self._dist(self.drone_pos, target.pos) <= self.cfg.track_radius:
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
            "track_progress", self._track_progress_reward(value)
        )
        if (
            self.track_progress[target_id] >= self.cfg.track_required
            and not self.completed[target_id]
        ):
            self.completed[target_id] = True
            reward += self._add_part(
                "complete", self._complete_reward(value)
            )
        self.memory.update_target_progress(
            target_id,
            int(self.track_progress[target_id]),
            bool(self.completed[target_id]),
        )
        self._decay_untracked_targets(target_id)
        return float(reward)


class ThesisMultiDroneLocalMemoryEnv(
    _TaskPotentialMixin, MultiDroneLocalMemoryEnv
):
    """Corrected local-memory multi-UAV environment."""

    cfg: ThesisMultiEnvConfig

    def __init__(self, config: ThesisMultiEnvConfig | None = None):
        config = config or ThesisMultiEnvConfig()
        if config.reward_mode not in {"legacy", "task_potential"}:
            raise ValueError("reward_mode must be 'legacy' or 'task_potential'")
        if config.track_progress_decay < 0:
            raise ValueError("track_progress_decay must be non-negative")
        super().__init__(config)

    def step(self, actions):
        potential_before = self._potential()
        obs, reward, terminated, truncated, _ = super().step(actions)
        if self.cfg.reward_mode == "task_potential":
            reward = self._replace_reward_with_task_potential(
                potential_before=potential_before,
                terminated=bool(terminated),
                truncated=bool(truncated),
            )
        return obs, float(reward), terminated, truncated, self._info()

    def _potential(self) -> float:
        coverage = float(self.team_visited.mean())
        detected_value = float(
            (self.detected.astype(np.float32) * self.target_values).sum()
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
                index for index in target_ids if not self.completed[index]
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
            1e-6, 1.0 - float(self.cfg.detection_probability)
        )
        memory.suppress_empty_cells(
            negative_cells,
            factor=likelihood_no_detection,
        )
        return float(reward)

    def _update_all_known_progress(
        self, target_id: int, progress: int, completed: bool
    ) -> None:
        for memory in self.memories:
            memory.update_target_progress(
                target_id, progress, completed
            )

    def _decay_untracked_targets(self, tracked_targets: set[int]) -> None:
        decay = int(self.cfg.track_progress_decay)
        if decay <= 0:
            return
        for target_id in range(self.n_targets):
            if self.completed[target_id] or target_id in tracked_targets:
                continue
            old = int(self.track_progress[target_id])
            new = max(0, old - decay)
            if new == old:
                continue
            self.track_progress[target_id] = new
            self._update_all_known_progress(target_id, new, False)

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
                    self.drone_pos[agent_id], target.pos
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
                "track_progress", self._track_progress_reward(value)
            )
            if self.track_progress[target_id] >= self.cfg.track_required:
                self.completed[target_id] = True
                reward += self._add_part(
                    "complete", self._complete_reward(value)
                )
            self._update_all_known_progress(
                target_id,
                int(self.track_progress[target_id]),
                bool(self.completed[target_id]),
            )

        self._decay_untracked_targets(used_targets)
        return float(reward)

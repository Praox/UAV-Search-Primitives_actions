from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class KnownTarget:
    target_id: int
    pos: np.ndarray
    value: int
    progress: int = 0
    completed: bool = False
    last_seen_step: int = 0


@dataclass
class DroneMemory:
    """What the UAV currently knows about the world.

    The memory has the same spatial resolution as the hidden world, but it is not
    privileged truth. It is updated only through local sensing and tracking.
    """

    grid_size: int
    n_targets: int
    belief_total: float | None = None
    belief: np.ndarray = field(init=False)
    visited: np.ndarray = field(init=False)
    completed_map: np.ndarray = field(init=False)
    known_targets: dict[int, KnownTarget] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        total = float(self.n_targets if self.belief_total is None else self.belief_total)
        self.belief = np.ones((self.grid_size, self.grid_size), dtype=np.float32)
        self.belief *= total / float(self.grid_size * self.grid_size)
        self.visited = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        self.completed_map = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        self.known_targets = {}

    def mark_visited(self, cells: list[tuple[int, int]]) -> None:
        for r, c in cells:
            self.visited[int(r), int(c)] = 1.0

    def suppress_empty_cells(self, cells: list[tuple[int, int]], factor: float = 0.2) -> None:
        for r, c in cells:
            self.belief[int(r), int(c)] *= factor
        self.normalize_belief()

    def add_or_update_target(self, target_id: int, pos: tuple[int, int], value: int, step: int) -> None:
        old = self.known_targets.get(int(target_id))
        progress = 0 if old is None else old.progress
        completed = False if old is None else old.completed
        self.known_targets[int(target_id)] = KnownTarget(
            target_id=int(target_id),
            pos=np.array(pos, dtype=np.int64),
            value=int(value),
            progress=int(progress),
            completed=bool(completed),
            last_seen_step=int(step),
        )
        self.belief[pos] = max(float(self.belief[pos]), 1.0)
        self.normalize_belief()

    def update_target_progress(self, target_id: int, progress: int, completed: bool) -> None:
        target_id = int(target_id)
        if target_id not in self.known_targets:
            return
        target = self.known_targets[target_id]
        target.progress = int(progress)
        target.completed = bool(completed)
        if completed:
            r, c = target.pos
            self.completed_map[int(r), int(c)] = 1.0

    def known_target_value_map(self) -> np.ndarray:
        out = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        for target in self.known_targets.values():
            if not target.completed:
                r, c = target.pos
                out[int(r), int(c)] = float(target.value) / 2.0
        return out

    def track_progress_map(self, track_required: int) -> np.ndarray:
        """Return the normalized progress of every known, unfinished target.

        A value of 0.0 means no progress and 1.0 means the target has reached the
        required tracking duration. Completed targets remain represented in the
        dedicated completed map and are therefore omitted here.
        """

        denominator = max(1, int(track_required))
        out = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        for target in self.known_targets.values():
            if target.completed:
                continue
            r, c = target.pos
            out[int(r), int(c)] = min(1.0, max(0.0, float(target.progress) / denominator))
        return out

    def normalize_belief(self) -> None:
        target_total = float(self.n_targets if self.belief_total is None else self.belief_total)
        self.belief = np.clip(self.belief, 1e-6, None)
        total = float(self.belief.sum())
        if total <= 1e-8:
            self.belief[:] = target_total / float(self.grid_size * self.grid_size)
        else:
            self.belief *= target_total / total

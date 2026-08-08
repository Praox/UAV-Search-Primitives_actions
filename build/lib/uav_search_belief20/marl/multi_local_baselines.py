from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from uav_search_belief20.actions import MOVES, STAY
from uav_search_belief20.envs.multi_drone_local_env import MultiDroneLocalMemoryEnv


def _manhattan(first: np.ndarray, second: np.ndarray) -> int:
    return int(np.abs(np.asarray(first, dtype=np.int64) - np.asarray(second, dtype=np.int64)).sum())


def _candidate(position: np.ndarray, action: int) -> np.ndarray:
    dr, dc = MOVES[int(action)]
    return np.asarray(position, dtype=np.int64) + np.asarray([dr, dc], dtype=np.int64)


@dataclass
class RandomMultiLocalPolicy:
    """Independent valid random actions, reproducible per evaluation world."""

    seed: int = 999

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(int(self.seed))

    def begin_episode(self, world_seed: int) -> None:
        self.rng = np.random.default_rng(int(self.seed) + 1_000_003 * int(world_seed))

    def act(
        self,
        env: MultiDroneLocalMemoryEnv,
        obs_all: np.ndarray,
        action_masks: np.ndarray,
    ) -> np.ndarray:
        del env, obs_all
        actions = []
        for mask in np.asarray(action_masks, dtype=bool):
            valid = np.flatnonzero(mask)
            if valid.size == 0:
                raise RuntimeError("Random baseline received an empty action mask.")
            actions.append(int(self.rng.choice(valid)))
        return np.asarray(actions, dtype=np.int64)


@dataclass
class LocalFrontierMultiPolicy:
    """No-communication local search-and-track heuristic.

    Every UAV uses only its own current position and its own ``DroneMemory``. Known,
    incomplete targets are pursued before exploration. Otherwise, a one-step greedy
    frontier score rewards unobserved cells in the next sensor footprint and uses
    local belief mass as a tie-breaker.

    The policy intentionally does not read hidden target truth, another UAV's memory,
    the centralized mixer state, or the future joint action. It is therefore a fair
    absolute baseline for the repository's local-memory decentralized execution.
    """

    seed: int = 999
    new_cell_weight: float = 10.0
    belief_weight: float = 1.0
    stay_penalty: float = 1.0

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(int(self.seed))

    def begin_episode(self, world_seed: int) -> None:
        self.rng = np.random.default_rng(int(self.seed) + 1_000_003 * int(world_seed))

    def act(
        self,
        env: MultiDroneLocalMemoryEnv,
        obs_all: np.ndarray,
        action_masks: np.ndarray,
    ) -> np.ndarray:
        del obs_all
        actions = [
            self._agent_action(env, agent_id, np.asarray(action_masks[agent_id], dtype=bool))
            for agent_id in range(env.cfg.n_agents)
        ]
        return np.asarray(actions, dtype=np.int64)

    def _agent_action(
        self,
        env: MultiDroneLocalMemoryEnv,
        agent_id: int,
        action_mask: np.ndarray,
    ) -> int:
        valid_actions = np.flatnonzero(action_mask)
        if valid_actions.size == 0:
            raise RuntimeError(f"Agent {agent_id} received an empty action mask.")

        memory = env.memories[int(agent_id)]
        position = np.asarray(env.drone_pos[int(agent_id)], dtype=np.int64)
        unfinished = [target for target in memory.known_targets.values() if not target.completed]

        if unfinished:
            # Value first, then preserve existing tracking commitment, then proximity.
            target = min(
                unfinished,
                key=lambda item: (
                    -int(item.value),
                    -int(item.progress),
                    _manhattan(position, item.pos),
                    int(item.target_id),
                ),
            )
            if _manhattan(position, target.pos) <= int(env.cfg.track_radius):
                if bool(action_mask[STAY]):
                    return int(STAY)
            scores = [
                (-float(_manhattan(_candidate(position, int(action)), target.pos)), int(action))
                for action in valid_actions
            ]
            return self._random_argmax(scores)

        scores: list[tuple[float, int]] = []
        for action in valid_actions:
            candidate = _candidate(position, int(action))
            visible = env._cells_in_radius(candidate, int(env.cfg.sensor_radius))
            new_cells = sum(1 for cell in visible if memory.visited[cell] < 0.5)
            belief_mass = float(sum(float(memory.belief[cell]) for cell in visible))
            score = self.new_cell_weight * float(new_cells)
            score += self.belief_weight * belief_mass
            if int(action) == STAY:
                score -= self.stay_penalty
            scores.append((score, int(action)))
        return self._random_argmax(scores)

    def _random_argmax(self, scores: list[tuple[float, int]]) -> int:
        best = max(score for score, _ in scores)
        candidates = [action for score, action in scores if np.isclose(score, best)]
        return int(self.rng.choice(candidates))


def make_multi_local_baseline(name: str, seed: int = 999):
    normalized = str(name).strip().lower().replace("-", "_")
    if normalized == "random":
        return RandomMultiLocalPolicy(seed=seed)
    if normalized in {"frontier", "local_frontier"}:
        return LocalFrontierMultiPolicy(seed=seed)
    raise ValueError(
        f"Unknown multi-local baseline {name!r}; expected random or local_frontier."
    )

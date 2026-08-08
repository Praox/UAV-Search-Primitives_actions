from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from uav_search_belief20.actions import MOVES, STAY
from uav_search_belief20.envs.primitive_search_env import PrimitiveSearchEnv


def _candidate_position(env: PrimitiveSearchEnv, action: int) -> np.ndarray:
    delta_row, delta_col = MOVES[int(action)]
    candidate = env.drone_pos + np.array([delta_row, delta_col], dtype=np.int64)
    # Match the environment's historical clipping semantics when the v3 mask is off.
    return np.clip(candidate, 0, env.cfg.grid_size - 1)


def _valid_actions(env: PrimitiveSearchEnv) -> np.ndarray:
    return np.flatnonzero(env.action_mask())


@dataclass
class RandomPolicy:
    seed: int = 0

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def act(self, env: PrimitiveSearchEnv, obs: np.ndarray, episode_index: int) -> int:
        del obs, episode_index
        return int(self.rng.choice(_valid_actions(env)))


@dataclass
class FrontierPolicy:
    """Greedy, non-learning search-and-track heuristic.

    Known unfinished targets are pursued first. Otherwise the policy selects the
    action whose next sensor footprint contains the most unobserved cells, with
    belief mass used as a secondary tie-breaker.
    """

    seed: int = 0

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def act(self, env: PrimitiveSearchEnv, obs: np.ndarray, episode_index: int) -> int:
        del obs, episode_index
        allowed = _valid_actions(env)

        known_targets = [
            target
            for target in env.memory.known_targets.values()
            if not target.completed
        ]
        if known_targets:
            known_targets.sort(
                key=lambda target: (
                    -int(target.value),
                    env._dist(env.drone_pos, target.pos),
                )
            )
            target = known_targets[0]
            if env._dist(env.drone_pos, target.pos) <= env.cfg.track_radius:
                return STAY

            scores: list[tuple[float, int]] = []
            for action in allowed:
                candidate = _candidate_position(env, int(action))
                distance = env._dist(candidate, target.pos)
                scores.append((-float(distance), int(action)))
            return self._choose_best(scores)

        scores = []
        for action in allowed:
            candidate = _candidate_position(env, int(action))
            visible = env._cells_in_radius(candidate, env.cfg.sensor_radius)
            newly_observed = sum(
                1 for cell in visible if env.memory.visited[cell] < 0.5
            )
            belief_mass = float(sum(env.memory.belief[cell] for cell in visible))
            stay_penalty = 1.0 if int(action) == STAY else 0.0
            score = 10.0 * newly_observed + belief_mass - stay_penalty
            scores.append((score, int(action)))
        return self._choose_best(scores)

    def _choose_best(self, scores: list[tuple[float, int]]) -> int:
        best_score = max(score for score, _ in scores)
        candidates = [
            action for score, action in scores if np.isclose(score, best_score)
        ]
        return int(self.rng.choice(candidates))


@dataclass
class OraclePolicy:
    """Approximate upper-bound heuristic with access to hidden target positions."""

    seed: int = 0

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def act(self, env: PrimitiveSearchEnv, obs: np.ndarray, episode_index: int) -> int:
        del obs, episode_index
        unfinished = np.flatnonzero(~env.completed)
        if unfinished.size == 0:
            return STAY

        # Prioritize high-value targets, then short distance.
        target_id = min(
            (int(index) for index in unfinished),
            key=lambda index: (
                -int(env.target_values[index]),
                env._dist(env.drone_pos, env.target_pos[index]),
            ),
        )
        target_position = env.target_pos[target_id]
        if env._dist(env.drone_pos, target_position) <= env.cfg.track_radius:
            return STAY

        allowed = _valid_actions(env)
        scored: list[tuple[int, int]] = []
        for action in allowed:
            candidate = _candidate_position(env, int(action))
            distance = env._dist(candidate, target_position)
            scored.append((distance, int(action)))
        best_distance = min(distance for distance, _ in scored)
        candidates = [
            action for distance, action in scored if distance == best_distance
        ]
        return int(self.rng.choice(candidates))


def make_baseline(name: str, seed: int):
    normalized = str(name).strip().lower()
    if normalized == "random":
        return RandomPolicy(seed=seed)
    if normalized == "frontier":
        return FrontierPolicy(seed=seed)
    if normalized == "oracle":
        return OraclePolicy(seed=seed)
    raise ValueError(f"Unknown baseline {name!r}; expected random, frontier, or oracle.")

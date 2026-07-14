from __future__ import annotations

import numpy as np

from uav_search_belief20.actions import DOWN, LEFT, RIGHT, STAY, UP
from uav_search_belief20.envs.multi_drone_local_env import (
    MultiDroneLocalEnvConfig,
    MultiDroneLocalMemoryEnv,
)


def reset_memories(env: MultiDroneLocalMemoryEnv) -> None:
    for agent_id, memory in enumerate(env.memories):
        memory.reset()
        memory.mark_visited([tuple(env.drone_pos[agent_id])])
    env.team_visited[:] = 0.0
    for position in env.drone_pos:
        env.team_visited[tuple(position)] = 1.0
    env.detected[:] = False
    env.completed[:] = False
    env.track_progress[:] = 0


def main() -> None:
    env = MultiDroneLocalMemoryEnv(MultiDroneLocalEnvConfig(seed=1))
    obs, _ = env.reset()
    assert obs.shape == (3, 8, 20, 20)
    assert env.global_state().shape == (env.state_dim,)

    env.drone_pos[0] = np.array([0, 0])
    mask = env.action_mask()[0]
    assert mask[STAY] and mask[DOWN] and mask[RIGHT]
    assert not mask[UP] and not mask[LEFT]

    env.drone_pos[:] = np.array([[2, 2], [10, 10], [18, 18]])
    env.target_pos[:] = np.array([[2, 2], [5, 5], [12, 12], [17, 17]])
    reset_memories(env)
    obs, reward, _, _, info = env.step(np.array([STAY, STAY, STAY]))
    assert 0 in env.memories[0].known_targets
    assert 0 not in env.memories[1].known_targets
    assert 0 not in env.memories[2].known_targets
    assert obs[0, 3, 2, 2] > 0
    assert obs[1, 3, 2, 2] == 0 and obs[2, 3, 2, 2] == 0
    assert abs(reward - sum(info["last_reward_parts"].values())) < 1e-6

    env.drone_pos[:] = np.array([[10, 9], [10, 11], [2, 2]])
    env.target_pos[:] = np.array([[10, 10], [5, 5], [12, 12], [17, 17]])
    reset_memories(env)
    for agent_id in (0, 1):
        env.memories[agent_id].add_or_update_target(0, (10, 10), 1, 0)
    _, _, _, _, info = env.step(np.array([STAY, STAY, STAY]))
    assert env.track_progress[0] == 1
    assert int(np.sum(info["last_tracking_progress"])) == 1
    print("multi-local environment smoke test: OK")


if __name__ == "__main__":
    main()

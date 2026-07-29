from __future__ import annotations

import inspect
from pathlib import Path
import tempfile

import numpy as np
import torch

from uav_search_belief20.envs.multi_drone_local_env import (
    MultiDroneLocalEnvConfig,
    MultiDroneLocalMemoryEnv,
)
from uav_search_belief20.marl.bayesian_qmix_agent import (
    BayesianLocalQMIXAgent,
    BayesianLocalQMIXConfig,
)


def changed(before: list[torch.Tensor], module: torch.nn.Module) -> bool:
    after = [parameter.detach().cpu() for parameter in module.parameters()]
    return any(not torch.allclose(first, second) for first, second in zip(before, after))


def make_env(state_mode: str = "privileged_truth") -> MultiDroneLocalMemoryEnv:
    return MultiDroneLocalMemoryEnv(
        MultiDroneLocalEnvConfig(
            grid_size=4,
            n_agents=2,
            n_value1_targets=1,
            n_value2_targets=1,
            max_steps=6,
            seed=7,
            global_state_mode=state_mode,
        )
    )


def make_agent(env: MultiDroneLocalMemoryEnv, sampling: str) -> BayesianLocalQMIXAgent:
    return BayesianLocalQMIXAgent(
        BayesianLocalQMIXConfig(
            obs_shape=env.observation_shape,
            state_dim=env.state_dim,
            n_agents=env.cfg.n_agents,
            action_dim=env.action_dim,
            feature_dim=8,
            mixing_embed_dim=4,
            mixing_hypernet_embed=8,
            batch_size=2,
            replay_capacity=8,
            target_update_period=2,
            posterior_sampling=sampling,
            uncertainty_mc_samples=4,
            device="cpu",
            seed=7,
        )
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


def assert_local_environment_contract() -> None:
    env = make_env()
    obs, _ = env.reset()
    assert obs.shape == (2, 8, 4, 4)
    assert env.global_state().shape == (env.state_dim,)

    env.drone_pos[0] = np.asarray([0, 0], dtype=np.int64)
    mask = env.action_mask()[0]
    assert bool(mask[0]) and bool(mask[2]) and bool(mask[4])
    assert not bool(mask[1]) and not bool(mask[3])

    env.drone_pos[:] = np.asarray([[1, 1], [3, 3]], dtype=np.int64)
    env.target_pos[:] = np.asarray([[1, 1], [0, 3]], dtype=np.int64)
    reset_memories(env)
    obs, reward, _, _, info = env.step(np.asarray([0, 0], dtype=np.int64))
    assert 0 in env.memories[0].known_targets
    assert 0 not in env.memories[1].known_targets
    assert obs[0, 3, 1, 1] > 0.0
    assert obs[1, 3, 1, 1] == 0.0
    assert abs(float(reward) - sum(info["last_reward_parts"].values())) < 1e-6

    env.drone_pos[:] = np.asarray([[2, 1], [2, 3]], dtype=np.int64)
    env.target_pos[:] = np.asarray([[2, 2], [0, 3]], dtype=np.int64)
    reset_memories(env)
    for agent_id in range(2):
        env.memories[agent_id].add_or_update_target(0, (2, 2), 1, 0)
    _, _, _, _, info = env.step(np.asarray([0, 0], dtype=np.int64))
    assert int(env.track_progress[0]) == 1
    assert int(np.sum(info["last_tracking_progress"])) == 1

def assert_sampling_semantics(env: MultiDroneLocalMemoryEnv) -> None:
    shared = make_agent(env, "shared")
    shared.resample_policy()
    assert shared.episode_sample_distance() == 0.0
    assert torch.equal(shared._episode_weight[0], shared._episode_weight[1])

    independent = make_agent(env, "independent")
    independent.resample_policy()
    assert independent.episode_sample_distance() > 0.0
    assert not torch.equal(independent._episode_weight[0], independent._episode_weight[1])


def assert_ctde_and_training(env: MultiDroneLocalMemoryEnv) -> None:
    obs_all, _ = env.reset()
    agent = make_agent(env, "independent")

    signature = inspect.signature(agent.act)
    assert "state" not in signature.parameters and "states" not in signature.parameters

    original_forward = agent.mixer.forward

    def forbidden(*args, **kwargs):
        raise AssertionError("Mixer leaked into decentralized execution")

    agent.mixer.forward = forbidden  # type: ignore[method-assign]
    actions = agent.act(
        obs_all,
        action_masks=env.action_mask(),
        use_sample=True,
        explore=False,
    )
    agent.mixer.forward = original_forward  # type: ignore[method-assign]
    assert actions.shape == (env.cfg.n_agents,)

    state = env.global_state()
    for _ in range(2):
        masks = env.action_mask()
        actions = agent.act(
            obs_all,
            action_masks=masks,
            use_sample=True,
            explore=True,
        )
        next_obs, reward, terminated, truncated, _ = env.step(actions)
        next_state = env.global_state()
        next_masks = env.action_mask()
        agent.replay.add(
            obs_all=obs_all,
            state=state,
            actions=actions,
            reward=reward,
            next_obs_all=next_obs,
            next_state=next_state,
            done=bool(terminated or truncated),
            action_masks=masks,
            next_action_masks=next_masks,
        )
        obs_all, state = next_obs, next_state

    feature_before = [parameter.detach().clone() for parameter in agent.feature_net.parameters()]
    head_before = [parameter.detach().clone() for parameter in agent.head.parameters()]
    mixer_before = [parameter.detach().clone() for parameter in agent.mixer.parameters()]
    result = agent.train_step()
    assert np.isfinite(result["loss"])
    assert result["kl_loss"] >= 0.0
    assert result["posterior_std_mean"] > 0.0
    assert changed(feature_before, agent.feature_net)
    assert changed(head_before, agent.head)
    assert changed(mixer_before, agent.mixer)

    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "bayes_qmix.pt"
        agent.save(str(checkpoint))
        restored = make_agent(env, "independent")
        restored.load(str(checkpoint))
        restored_actions = restored.act(
            obs_all,
            action_masks=env.action_mask(),
            use_sample=False,
            explore=False,
        )
        assert restored_actions.shape == actions.shape


def assert_state_modes() -> None:
    privileged = make_env("privileged_truth")
    memory_union = make_env("memory_union")
    privileged_state = privileged.global_state()
    memory_state = memory_union.global_state()
    assert privileged_state.shape == memory_state.shape == (privileged.state_dim,)
    # Before any detection, the target-information block of memory_union is zero.
    target_start = 2 * memory_union.cfg.n_agents
    target_end = target_start + 6 * memory_union.n_targets
    assert np.allclose(memory_state[target_start:target_end], 0.0)
    assert not np.allclose(privileged_state[target_start:target_end], 0.0)


def assert_simultaneous_overlap_metric() -> None:
    env = make_env()
    env.drone_pos[:] = np.asarray([[1, 1], [1, 2]], dtype=np.int64)
    _, _, _, _, info = env.step(np.asarray([0, 0], dtype=np.int64))
    ratio = float(info["last_simultaneous_sensor_overlap_ratio"])
    assert 0.0 < ratio <= 1.0


def main() -> None:
    torch.set_num_threads(1)
    torch.manual_seed(7)
    np.random.seed(7)
    assert_local_environment_contract()
    env = make_env()
    assert_sampling_semantics(env)
    assert_ctde_and_training(env)
    assert_state_modes()
    assert_simultaneous_overlap_metric()
    print("Bayesian-QMIX smoke and CTDE tests: OK")


if __name__ == "__main__":
    main()

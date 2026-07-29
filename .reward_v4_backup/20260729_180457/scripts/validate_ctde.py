from __future__ import annotations

import inspect

import numpy as np
import torch

from uav_search_belief20.envs.multi_drone_local_env import (
    MultiDroneLocalEnvConfig,
    MultiDroneLocalMemoryEnv,
)
from uav_search_belief20.marl.qmix_local_agent import LocalQMIXAgent, LocalQMIXConfig


def changed(before, module: torch.nn.Module) -> bool:
    after = [p.detach().cpu() for p in module.parameters()]
    return any(not torch.allclose(a, b) for a, b in zip(before, after))


def main() -> None:
    torch.manual_seed(7)
    env = MultiDroneLocalMemoryEnv(MultiDroneLocalEnvConfig(seed=7))
    obs, _ = env.reset()
    agent = LocalQMIXAgent(
        LocalQMIXConfig(
            obs_shape=env.observation_shape,
            state_dim=env.state_dim,
            n_agents=env.cfg.n_agents,
            action_dim=env.action_dim,
            batch_size=4,
            replay_capacity=32,
            device="cpu",
            seed=7,
        )
    )
    signature = inspect.signature(agent.act)
    assert "state" not in signature.parameters and "states" not in signature.parameters

    original_forward = agent.mixer.forward
    def forbidden(*args, **kwargs):
        raise AssertionError("Mixer leaked into decentralized execution")
    agent.mixer.forward = forbidden  # type: ignore[method-assign]
    _ = agent.act(obs, action_masks=env.action_mask(), explore=False)
    agent.mixer.forward = original_forward  # type: ignore[method-assign]

    q = torch.tensor([[[1000.0, 1.0, 2.0, 3.0, 4.0]]])
    mask = torch.tensor([[[False, True, True, True, True]]])
    assert int(LocalQMIXAgent.masked_argmax(q, mask).item()) != 0

    state = torch.as_tensor(env.global_state()[None, :], dtype=torch.float32)
    base_q = torch.zeros((1, env.cfg.n_agents))
    base_total = agent.mixer(base_q, state)
    for agent_id in range(env.cfg.n_agents):
        increased = base_q.clone()
        increased[0, agent_id] += 1.0
        assert torch.all(agent.mixer(increased, state) >= base_total - 1e-6)

    obs_all, current_state = obs, env.global_state()
    for _ in range(4):
        masks = env.action_mask()
        actions = agent.act(obs_all, action_masks=masks, explore=True)
        next_obs, reward, terminated, truncated, _ = env.step(actions)
        next_state, next_masks = env.global_state(), env.action_mask()
        agent.replay.add(
            obs_all=obs_all,
            state=current_state,
            actions=actions,
            reward=reward,
            next_obs_all=next_obs,
            next_state=next_state,
            done=bool(terminated or truncated),
            action_masks=masks,
            next_action_masks=next_masks,
        )
        obs_all, current_state = next_obs, next_state

    utility_before = [p.detach().clone() for p in agent.agent_net.parameters()]
    mixer_before = [p.detach().clone() for p in agent.mixer.parameters()]
    result = agent.train_step()
    assert np.isfinite(result["loss"])
    assert changed(utility_before, agent.agent_net)
    assert changed(mixer_before, agent.mixer)
    print("CTDE validation: OK")


if __name__ == "__main__":
    main()

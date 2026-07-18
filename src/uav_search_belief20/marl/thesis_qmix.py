from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Deque, NamedTuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from uav_search_belief20.marl.qmix_mixer import QMixer
from uav_search_belief20.models.networks import GridFeatureNet, QNetwork


class VariationalLinearQHead(nn.Module):
    """Mean-field Gaussian posterior over a linear action-value head."""

    def __init__(
        self,
        feature_dim: int,
        action_dim: int,
        *,
        prior_std: float,
        initial_std: float,
        min_std: float,
        max_std: float,
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.action_dim = int(action_dim)
        self.prior_std = float(prior_std)
        self.min_std = float(min_std)
        self.max_std = float(max_std)
        self.weight_mu = nn.Parameter(torch.empty(action_dim, feature_dim))
        self.bias_mu = nn.Parameter(torch.empty(action_dim))
        initial_rho = self._inverse_softplus(initial_std)
        self.weight_rho = nn.Parameter(
            torch.full((action_dim, feature_dim), initial_rho)
        )
        self.bias_rho = nn.Parameter(torch.full((action_dim,), initial_rho))
        nn.init.kaiming_uniform_(self.weight_mu, a=math.sqrt(5.0))
        bound = 1.0 / math.sqrt(max(1, feature_dim))
        nn.init.uniform_(self.bias_mu, -bound, bound)

    @staticmethod
    def _inverse_softplus(value: float) -> float:
        return float(math.log(math.expm1(max(float(value), 1e-8))))

    def weight_std(self) -> torch.Tensor:
        return F.softplus(self.weight_rho).clamp(
            self.min_std, self.max_std
        )

    def bias_std(self) -> torch.Tensor:
        return F.softplus(self.bias_rho).clamp(
            self.min_std, self.max_std
        )

    def sample_parameters(self, n_samples: int) -> tuple[torch.Tensor, torch.Tensor]:
        weight_epsilon = torch.randn(
            int(n_samples),
            self.action_dim,
            self.feature_dim,
            device=self.weight_mu.device,
            dtype=self.weight_mu.dtype,
        )
        bias_epsilon = torch.randn(
            int(n_samples),
            self.action_dim,
            device=self.bias_mu.device,
            dtype=self.bias_mu.dtype,
        )
        weight = (
            self.weight_mu.unsqueeze(0)
            + self.weight_std().unsqueeze(0) * weight_epsilon
        )
        bias = (
            self.bias_mu.unsqueeze(0)
            + self.bias_std().unsqueeze(0) * bias_epsilon
        )
        return weight, bias

    def mean_q(self, features: torch.Tensor) -> torch.Tensor:
        return F.linear(features, self.weight_mu, self.bias_mu)

    @staticmethod
    def sampled_q(
        features: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
    ) -> torch.Tensor:
        if features.dim() == 2:
            return torch.einsum("nf,naf->na", features, weight) + bias
        if features.dim() == 3:
            return (
                torch.einsum("bnf,naf->bna", features, weight)
                + bias.unsqueeze(0)
            )
        raise ValueError(f"Unsupported feature shape {features.shape}")

    def kl_divergence(self) -> torch.Tensor:
        prior_variance = self.prior_std**2

        def gaussian_kl(mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
            variance = std.square()
            return 0.5 * torch.sum(
                (variance + mean.square()) / prior_variance
                - 1.0
                + math.log(prior_variance)
                - torch.log(variance)
            )

        return gaussian_kl(self.weight_mu, self.weight_std()) + gaussian_kl(
            self.bias_mu, self.bias_std()
        )

    @property
    def posterior_parameter_count(self) -> int:
        return int(self.weight_mu.numel() + self.bias_mu.numel())

    @torch.no_grad()
    def clamp_rho_(self) -> None:
        self.weight_rho.clamp_(
            self._inverse_softplus(self.min_std),
            self._inverse_softplus(self.max_std),
        )
        self.bias_rho.clamp_(
            self._inverse_softplus(self.min_std),
            self._inverse_softplus(self.max_std),
        )

    @torch.no_grad()
    def diagnostics(self) -> dict[str, float]:
        std = torch.cat((self.weight_std().flatten(), self.bias_std().flatten()))
        return {
            "posterior_std_mean": float(std.mean().item()),
            "posterior_std_min": float(std.min().item()),
            "posterior_std_max": float(std.max().item()),
            "posterior_kl_per_parameter": float(
                (self.kl_divergence() / max(1, self.posterior_parameter_count)).item()
            ),
        }


class ThesisJointBatch(NamedTuple):
    obs: np.ndarray
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_obs: np.ndarray
    next_states: np.ndarray
    dones: np.ndarray
    discounts: np.ndarray
    action_masks: np.ndarray
    next_action_masks: np.ndarray


@dataclass
class _JointRawTransition:
    obs: np.ndarray
    state: np.ndarray
    actions: np.ndarray
    reward: float
    next_obs: np.ndarray
    next_state: np.ndarray
    done: bool
    action_masks: np.ndarray
    next_action_masks: np.ndarray


class ThesisJointReplayBuffer:
    """Joint n-step replay used by deterministic and Bayesian QMIX."""

    def __init__(
        self,
        capacity: int,
        n_agents: int,
        obs_shape: tuple[int, int, int],
        state_dim: int,
        action_dim: int,
        *,
        n_step: int = 3,
        gamma: float = 0.99,
        seed: int = 42,
    ) -> None:
        self.capacity = int(capacity)
        self.n_agents = int(n_agents)
        self.obs_shape = tuple(obs_shape)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.n_step = int(n_step)
        self.gamma = float(gamma)
        self.rng = np.random.default_rng(seed)

        self.obs = np.zeros(
            (capacity, n_agents, *obs_shape), dtype=np.float32
        )
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, n_agents), dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_obs = np.zeros_like(self.obs)
        self.next_states = np.zeros_like(self.states)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.discounts = np.zeros(capacity, dtype=np.float32)
        self.action_masks = np.ones(
            (capacity, n_agents, action_dim), dtype=bool
        )
        self.next_action_masks = np.ones_like(self.action_masks)
        self.pos = 0
        self.size = 0
        self._queue: Deque[_JointRawTransition] = deque()

    def __len__(self) -> int:
        return self.size

    def add(
        self,
        *,
        obs_all: np.ndarray,
        state: np.ndarray,
        actions: np.ndarray,
        reward: float,
        next_obs_all: np.ndarray,
        next_state: np.ndarray,
        done: bool,
        action_masks: np.ndarray,
        next_action_masks: np.ndarray,
    ) -> None:
        self._queue.append(
            _JointRawTransition(
                obs=np.asarray(obs_all, dtype=np.float32).copy(),
                state=np.asarray(state, dtype=np.float32).copy(),
                actions=np.asarray(actions, dtype=np.int64).copy(),
                reward=float(reward),
                next_obs=np.asarray(next_obs_all, dtype=np.float32).copy(),
                next_state=np.asarray(next_state, dtype=np.float32).copy(),
                done=bool(done),
                action_masks=np.asarray(action_masks, dtype=bool).copy(),
                next_action_masks=np.asarray(
                    next_action_masks, dtype=bool
                ).copy(),
            )
        )
        if done:
            while self._queue:
                self._commit_oldest()
        elif len(self._queue) >= self.n_step:
            self._commit_oldest()

    def _commit_oldest(self) -> None:
        first = self._queue[0]
        reward = 0.0
        steps = 0
        final = first
        for transition in list(self._queue)[: self.n_step]:
            reward += (self.gamma**steps) * transition.reward
            steps += 1
            final = transition
            if transition.done:
                break

        index = self.pos
        self.obs[index] = first.obs
        self.states[index] = first.state
        self.actions[index] = first.actions
        self.rewards[index] = reward
        self.next_obs[index] = final.next_obs
        self.next_states[index] = final.next_state
        self.dones[index] = float(final.done)
        self.discounts[index] = self.gamma**steps
        self.action_masks[index] = first.action_masks
        self.next_action_masks[index] = final.next_action_masks
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        self._queue.popleft()

    def sample(self, batch_size: int) -> ThesisJointBatch:
        if self.size <= 0:
            raise RuntimeError("Cannot sample from empty joint replay.")
        indices = self.rng.integers(0, self.size, size=int(batch_size))
        return ThesisJointBatch(
            self.obs[indices],
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_obs[indices],
            self.next_states[indices],
            self.dones[indices],
            self.discounts[indices],
            self.action_masks[indices],
            self.next_action_masks[indices],
        )


@dataclass
class ThesisQMIXConfig:
    obs_shape: tuple[int, int, int]
    state_dim: int
    n_agents: int = 3
    action_dim: int = 5
    feature_dim: int = 128
    mixing_embed_dim: int = 32
    mixing_hypernet_embed: int = 64
    gamma: float = 0.99
    n_step: int = 3
    lr: float = 1e-4
    batch_size: int = 64
    replay_capacity: int = 100_000
    target_tau: float = 0.005
    target_update_period: int = 500
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 20_000
    huber_delta: float = 1.0
    normalize_features: bool = True
    grad_clip_norm: float = 10.0
    device: str = "cpu"
    seed: int = 42


class ThesisQMIXAgent:
    """Masked n-step DDQN-QMIX with Polyak target updates."""

    def __init__(self, cfg: ThesisQMIXConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.rng = np.random.default_rng(cfg.seed)
        self.agent_net = QNetwork(
            cfg.obs_shape,
            cfg.action_dim,
            cfg.feature_dim,
            normalize_features=cfg.normalize_features,
        ).to(self.device)
        self.target_agent_net = QNetwork(
            cfg.obs_shape,
            cfg.action_dim,
            cfg.feature_dim,
            normalize_features=cfg.normalize_features,
        ).to(self.device)
        self.target_agent_net.load_state_dict(self.agent_net.state_dict())
        self.mixer = QMixer(
            cfg.n_agents,
            cfg.state_dim,
            cfg.mixing_embed_dim,
            cfg.mixing_hypernet_embed,
        ).to(self.device)
        self.target_mixer = QMixer(
            cfg.n_agents,
            cfg.state_dim,
            cfg.mixing_embed_dim,
            cfg.mixing_hypernet_embed,
        ).to(self.device)
        self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.optim = torch.optim.Adam(
            list(self.agent_net.parameters()) + list(self.mixer.parameters()),
            lr=cfg.lr,
        )
        self.replay = ThesisJointReplayBuffer(
            cfg.replay_capacity,
            cfg.n_agents,
            cfg.obs_shape,
            cfg.state_dim,
            cfg.action_dim,
            n_step=cfg.n_step,
            gamma=cfg.gamma,
            seed=cfg.seed,
        )
        self.train_steps = 0
        self.env_steps = 0

    def epsilon(self) -> float:
        fraction = min(
            1.0, self.env_steps / max(1, self.cfg.epsilon_decay_steps)
        )
        return float(
            self.cfg.epsilon_start
            + fraction * (self.cfg.epsilon_end - self.cfg.epsilon_start)
        )

    @staticmethod
    def masked_argmax(
        q_values: torch.Tensor, masks: torch.Tensor
    ) -> torch.Tensor:
        if q_values.shape != masks.shape:
            raise ValueError("Q values and masks must have identical shapes.")
        if not torch.all(masks.any(dim=-1)):
            raise ValueError("At least one agent has no valid action.")
        return q_values.masked_fill(
            ~masks.bool(), torch.finfo(q_values.dtype).min
        ).argmax(dim=-1)

    @torch.no_grad()
    def act(
        self,
        obs_all: np.ndarray,
        *,
        action_masks: np.ndarray,
        explore: bool = True,
    ) -> np.ndarray:
        obs = torch.as_tensor(
            obs_all, dtype=torch.float32, device=self.device
        )
        q_values = self.agent_net(obs).cpu().numpy()
        masks = np.asarray(action_masks, dtype=bool)
        actions = np.zeros(self.cfg.n_agents, dtype=np.int64)
        for agent_id in range(self.cfg.n_agents):
            allowed = np.flatnonzero(masks[agent_id])
            if explore and self.rng.random() < self.epsilon():
                actions[agent_id] = int(self.rng.choice(allowed))
            else:
                local_q = q_values[agent_id].copy()
                local_q[~masks[agent_id]] = -np.inf
                actions[agent_id] = int(np.argmax(local_q))
        if explore:
            self.env_steps += 1
        return actions

    def _soft_update(self) -> None:
        tau = float(self.cfg.target_tau)
        if tau > 0.0:
            with torch.no_grad():
                for target, source in zip(
                    self.target_agent_net.parameters(),
                    self.agent_net.parameters(),
                ):
                    target.mul_(1.0 - tau).add_(source, alpha=tau)
                for target, source in zip(
                    self.target_mixer.parameters(), self.mixer.parameters()
                ):
                    target.mul_(1.0 - tau).add_(source, alpha=tau)
        elif self.train_steps % self.cfg.target_update_period == 0:
            self.target_agent_net.load_state_dict(
                self.agent_net.state_dict()
            )
            self.target_mixer.load_state_dict(self.mixer.state_dict())

    def train_step(self) -> dict[str, float]:
        if len(self.replay) < self.cfg.batch_size:
            return {"loss": 0.0, "epsilon": self.epsilon()}
        batch = self.replay.sample(self.cfg.batch_size)
        stats = self._gradient_update(batch)
        self.train_steps += 1
        self._soft_update()
        return {**stats, "epsilon": self.epsilon()}

    def _gradient_update(self, batch: ThesisJointBatch) -> dict[str, float]:
        cfg = self.cfg
        batch_size = int(batch.obs.shape[0])
        obs = torch.as_tensor(batch.obs, dtype=torch.float32, device=self.device)
        states = torch.as_tensor(
            batch.states, dtype=torch.float32, device=self.device
        )
        actions = torch.as_tensor(
            batch.actions, dtype=torch.long, device=self.device
        )
        rewards = torch.as_tensor(
            batch.rewards, dtype=torch.float32, device=self.device
        )
        next_obs = torch.as_tensor(
            batch.next_obs, dtype=torch.float32, device=self.device
        )
        next_states = torch.as_tensor(
            batch.next_states, dtype=torch.float32, device=self.device
        )
        dones = torch.as_tensor(
            batch.dones, dtype=torch.float32, device=self.device
        )
        discounts = torch.as_tensor(
            batch.discounts, dtype=torch.float32, device=self.device
        )
        next_masks = torch.as_tensor(
            batch.next_action_masks, dtype=torch.bool, device=self.device
        )

        flat_obs = obs.reshape(batch_size * cfg.n_agents, *cfg.obs_shape)
        q_all = self.agent_net(flat_obs).reshape(
            batch_size, cfg.n_agents, cfg.action_dim
        )
        chosen = q_all.gather(2, actions.unsqueeze(-1)).squeeze(-1)
        q_tot = self.mixer(chosen, states)

        with torch.no_grad():
            flat_next = next_obs.reshape(
                batch_size * cfg.n_agents, *cfg.obs_shape
            )
            next_online = self.agent_net(flat_next).reshape(
                batch_size, cfg.n_agents, cfg.action_dim
            )
            next_actions = self.masked_argmax(next_online, next_masks)
            next_target = self.target_agent_net(flat_next).reshape(
                batch_size, cfg.n_agents, cfg.action_dim
            )
            next_chosen = next_target.gather(
                2, next_actions.unsqueeze(-1)
            ).squeeze(-1)
            target_q_tot = self.target_mixer(next_chosen, next_states)
            targets = rewards + discounts * (1.0 - dones) * target_q_tot

        loss = F.smooth_l1_loss(
            q_tot, targets, beta=float(cfg.huber_delta)
        )
        self.optim.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.agent_net.parameters()) + list(self.mixer.parameters()),
            cfg.grad_clip_norm,
        )
        self.optim.step()
        return {
            "loss": float(loss.item()),
            "q_tot_mean": float(q_tot.mean().item()),
            "target_mean": float(targets.mean().item()),
        }

    def save(self, path: str) -> None:
        torch.save(
            {
                "cfg": self.cfg.__dict__,
                "agent_net": self.agent_net.state_dict(),
                "target_agent_net": self.target_agent_net.state_dict(),
                "mixer": self.mixer.state_dict(),
                "target_mixer": self.target_mixer.state_dict(),
                "optim": self.optim.state_dict(),
                "train_steps": self.train_steps,
                "env_steps": self.env_steps,
            },
            path,
        )

    def load(self, path: str) -> None:
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(path, map_location=self.device)
        self.agent_net.load_state_dict(checkpoint["agent_net"])
        self.target_agent_net.load_state_dict(
            checkpoint.get("target_agent_net", checkpoint["agent_net"])
        )
        self.mixer.load_state_dict(checkpoint["mixer"])
        self.target_mixer.load_state_dict(
            checkpoint.get("target_mixer", checkpoint["mixer"])
        )
        if "optim" in checkpoint:
            self.optim.load_state_dict(checkpoint["optim"])
        self.train_steps = int(checkpoint.get("train_steps", 0))
        self.env_steps = int(checkpoint.get("env_steps", 0))


@dataclass
class ThesisBayesianQMIXConfig(ThesisQMIXConfig):
    posterior_sampling: str = "shared"
    prior_std: float = 1.0
    initial_posterior_std: float = 0.05
    min_posterior_std: float = 1e-4
    max_posterior_std: float = 1.0
    kl_weight: float = 1e-3
    epsilon_start: float = 0.0
    epsilon_end: float = 0.0


class ThesisBayesianQMIXAgent:
    """Variational Bayesian-QMIX with the same corrected training contract."""

    def __init__(self, cfg: ThesisBayesianQMIXConfig):
        if cfg.posterior_sampling not in {"shared", "independent"}:
            raise ValueError("posterior_sampling must be shared or independent")
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.rng = np.random.default_rng(cfg.seed)
        channels, height, width = cfg.obs_shape
        self.feature_net = GridFeatureNet(
            channels,
            height,
            cfg.feature_dim,
            normalize_output=cfg.normalize_features,
        ).to(self.device)
        self.target_feature_net = GridFeatureNet(
            channels,
            height,
            cfg.feature_dim,
            normalize_output=cfg.normalize_features,
        ).to(self.device)
        self.target_feature_net.load_state_dict(self.feature_net.state_dict())
        self.head = VariationalLinearQHead(
            cfg.feature_dim,
            cfg.action_dim,
            prior_std=cfg.prior_std,
            initial_std=cfg.initial_posterior_std,
            min_std=cfg.min_posterior_std,
            max_std=cfg.max_posterior_std,
        ).to(self.device)
        self.target_head = VariationalLinearQHead(
            cfg.feature_dim,
            cfg.action_dim,
            prior_std=cfg.prior_std,
            initial_std=cfg.initial_posterior_std,
            min_std=cfg.min_posterior_std,
            max_std=cfg.max_posterior_std,
        ).to(self.device)
        self.target_head.load_state_dict(self.head.state_dict())
        self.mixer = QMixer(
            cfg.n_agents,
            cfg.state_dim,
            cfg.mixing_embed_dim,
            cfg.mixing_hypernet_embed,
        ).to(self.device)
        self.target_mixer = QMixer(
            cfg.n_agents,
            cfg.state_dim,
            cfg.mixing_embed_dim,
            cfg.mixing_hypernet_embed,
        ).to(self.device)
        self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.optim = torch.optim.Adam(
            list(self.feature_net.parameters())
            + list(self.head.parameters())
            + list(self.mixer.parameters()),
            lr=cfg.lr,
        )
        self.replay = ThesisJointReplayBuffer(
            cfg.replay_capacity,
            cfg.n_agents,
            cfg.obs_shape,
            cfg.state_dim,
            cfg.action_dim,
            n_step=cfg.n_step,
            gamma=cfg.gamma,
            seed=cfg.seed,
        )
        self.train_steps = 0
        self.env_steps = 0
        self._episode_weight: torch.Tensor | None = None
        self._episode_bias: torch.Tensor | None = None
        self.resample_policy()

    @property
    def posterior_sampling(self) -> str:
        return self.cfg.posterior_sampling

    def epsilon(self) -> float:
        fraction = min(
            1.0, self.env_steps / max(1, self.cfg.epsilon_decay_steps)
        )
        return float(
            self.cfg.epsilon_start
            + fraction * (self.cfg.epsilon_end - self.cfg.epsilon_start)
        )

    @staticmethod
    def masked_argmax(
        q_values: torch.Tensor, masks: torch.Tensor
    ) -> torch.Tensor:
        return ThesisQMIXAgent.masked_argmax(q_values, masks)

    @torch.no_grad()
    def resample_policy(self) -> None:
        draws = 1 if self.cfg.posterior_sampling == "shared" else self.cfg.n_agents
        weight, bias = self.head.sample_parameters(draws)
        if draws == 1:
            weight = weight.expand(self.cfg.n_agents, -1, -1).clone()
            bias = bias.expand(self.cfg.n_agents, -1).clone()
        self._episode_weight = weight
        self._episode_bias = bias

    @torch.no_grad()
    def act(
        self,
        obs_all: np.ndarray,
        *,
        action_masks: np.ndarray,
        use_sample: bool = True,
        explore: bool = True,
    ) -> np.ndarray:
        obs = torch.as_tensor(obs_all, dtype=torch.float32, device=self.device)
        features = self.feature_net(obs)
        if use_sample:
            q_values = self.head.sampled_q(
                features, self._episode_weight, self._episode_bias
            )
        else:
            q_values = self.head.mean_q(features)
        q_values_np = q_values.cpu().numpy()
        masks = np.asarray(action_masks, dtype=bool)
        actions = np.zeros(self.cfg.n_agents, dtype=np.int64)
        for agent_id in range(self.cfg.n_agents):
            allowed = np.flatnonzero(masks[agent_id])
            if explore and self.rng.random() < self.epsilon():
                actions[agent_id] = int(self.rng.choice(allowed))
            else:
                local_q = q_values_np[agent_id].copy()
                local_q[~masks[agent_id]] = -np.inf
                actions[agent_id] = int(np.argmax(local_q))
        if explore:
            self.env_steps += 1
        return actions

    def _sample_training_head(self) -> tuple[torch.Tensor, torch.Tensor]:
        draws = 1 if self.cfg.posterior_sampling == "shared" else self.cfg.n_agents
        weight, bias = self.head.sample_parameters(draws)
        if draws == 1:
            weight = weight.expand(self.cfg.n_agents, -1, -1)
            bias = bias.expand(self.cfg.n_agents, -1)
        return weight, bias

    def _soft_update(self) -> None:
        tau = float(self.cfg.target_tau)
        if tau > 0.0:
            with torch.no_grad():
                pairs = (
                    (self.target_feature_net, self.feature_net),
                    (self.target_head, self.head),
                    (self.target_mixer, self.mixer),
                )
                for target_module, source_module in pairs:
                    for target, source in zip(
                        target_module.parameters(), source_module.parameters()
                    ):
                        target.mul_(1.0 - tau).add_(source, alpha=tau)
        elif self.train_steps % self.cfg.target_update_period == 0:
            self.target_feature_net.load_state_dict(self.feature_net.state_dict())
            self.target_head.load_state_dict(self.head.state_dict())
            self.target_mixer.load_state_dict(self.mixer.state_dict())

    def train_step(self) -> dict[str, float]:
        if len(self.replay) < self.cfg.batch_size:
            return {"loss": 0.0, **self.head.diagnostics()}
        batch = self.replay.sample(self.cfg.batch_size)
        stats = self._gradient_update(batch)
        self.train_steps += 1
        self._soft_update()
        return {**stats, **self.head.diagnostics()}

    def _gradient_update(self, batch: ThesisJointBatch) -> dict[str, float]:
        cfg = self.cfg
        batch_size = int(batch.obs.shape[0])
        obs = torch.as_tensor(batch.obs, dtype=torch.float32, device=self.device)
        states = torch.as_tensor(batch.states, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch.actions, dtype=torch.long, device=self.device)
        rewards = torch.as_tensor(batch.rewards, dtype=torch.float32, device=self.device)
        next_obs = torch.as_tensor(batch.next_obs, dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(batch.next_states, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(batch.dones, dtype=torch.float32, device=self.device)
        discounts = torch.as_tensor(batch.discounts, dtype=torch.float32, device=self.device)
        next_masks = torch.as_tensor(batch.next_action_masks, dtype=torch.bool, device=self.device)

        flat_obs = obs.reshape(batch_size * cfg.n_agents, *cfg.obs_shape)
        features = self.feature_net(flat_obs).reshape(
            batch_size, cfg.n_agents, cfg.feature_dim
        )
        sampled_weight, sampled_bias = self._sample_training_head()
        q_all = self.head.sampled_q(features, sampled_weight, sampled_bias)
        chosen = q_all.gather(2, actions.unsqueeze(-1)).squeeze(-1)
        q_tot = self.mixer(chosen, states)

        with torch.no_grad():
            flat_next = next_obs.reshape(batch_size * cfg.n_agents, *cfg.obs_shape)
            next_online_features = self.feature_net(flat_next).reshape(
                batch_size, cfg.n_agents, cfg.feature_dim
            )
            next_online_q = self.head.mean_q(next_online_features)
            next_actions = self.masked_argmax(next_online_q, next_masks)
            next_target_features = self.target_feature_net(flat_next).reshape(
                batch_size, cfg.n_agents, cfg.feature_dim
            )
            next_target_q = self.target_head.mean_q(next_target_features)
            next_chosen = next_target_q.gather(
                2, next_actions.unsqueeze(-1)
            ).squeeze(-1)
            target_q_tot = self.target_mixer(next_chosen, next_states)
            targets = rewards + discounts * (1.0 - dones) * target_q_tot

        td_loss = F.smooth_l1_loss(
            q_tot, targets, beta=float(cfg.huber_delta)
        )
        kl_per_parameter = self.head.kl_divergence() / max(
            1, self.head.posterior_parameter_count
        )
        kl_loss = cfg.kl_weight * kl_per_parameter
        loss = td_loss + kl_loss
        self.optim.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.feature_net.parameters())
            + list(self.head.parameters())
            + list(self.mixer.parameters()),
            cfg.grad_clip_norm,
        )
        self.optim.step()
        self.head.clamp_rho_()
        return {
            "loss": float(loss.item()),
            "td_loss": float(td_loss.item()),
            "kl_loss": float(kl_loss.item()),
            "q_tot_mean": float(q_tot.mean().item()),
            "target_mean": float(targets.mean().item()),
        }

    def save(self, path: str) -> None:
        torch.save(
            {
                "cfg": self.cfg.__dict__,
                "feature_net": self.feature_net.state_dict(),
                "target_feature_net": self.target_feature_net.state_dict(),
                "head": self.head.state_dict(),
                "target_head": self.target_head.state_dict(),
                "mixer": self.mixer.state_dict(),
                "target_mixer": self.target_mixer.state_dict(),
                "optim": self.optim.state_dict(),
                "train_steps": self.train_steps,
                "env_steps": self.env_steps,
            },
            path,
        )

    def load(self, path: str) -> None:
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(path, map_location=self.device)
        self.feature_net.load_state_dict(checkpoint["feature_net"])
        self.target_feature_net.load_state_dict(
            checkpoint.get("target_feature_net", checkpoint["feature_net"])
        )
        self.head.load_state_dict(checkpoint["head"])
        self.target_head.load_state_dict(
            checkpoint.get("target_head", checkpoint["head"])
        )
        self.mixer.load_state_dict(checkpoint["mixer"])
        self.target_mixer.load_state_dict(
            checkpoint.get("target_mixer", checkpoint["mixer"])
        )
        if "optim" in checkpoint:
            self.optim.load_state_dict(checkpoint["optim"])
        self.train_steps = int(checkpoint.get("train_steps", 0))
        self.env_steps = int(checkpoint.get("env_steps", 0))
        self.resample_policy()

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from uav_search_belief20.marl.qmix_local_agent import (
    LocalJointBatch,
    LocalJointReplayBuffer,
)
from uav_search_belief20.marl.qmix_mixer import QMixer
from uav_search_belief20.models.networks import GridFeatureNet


@dataclass
class BayesianLocalQMIXConfig:
    """Configuration for Bayesian-QMIX with a variational last utility layer.

    The feature extractor and mixer are deterministic.  Epistemic uncertainty is
    represented by a diagonal Gaussian posterior over the final per-action utility
    weights.  The posterior is optimized end-to-end through the team TD loss; no
    artificial per-agent Bellman targets are constructed.
    """

    obs_shape: tuple[int, int, int]
    state_dim: int
    n_agents: int = 3
    action_dim: int = 5
    feature_dim: int = 128
    mixing_embed_dim: int = 32
    mixing_hypernet_embed: int = 64
    gamma: float = 0.99
    lr: float = 1e-4
    batch_size: int = 64
    replay_capacity: int = 100_000
    target_update_period: int = 500
    grad_clip_norm: float = 10.0

    # Thompson-style execution.  Both modes use one shared posterior distribution.
    # "shared" draws one utility head and uses it for every UAV during an episode.
    # "independent" draws one utility head per UAV from that same posterior.
    posterior_sampling: str = "shared"

    prior_std: float = 1.0
    initial_posterior_std: float = 0.05
    min_posterior_std: float = 1e-4
    max_posterior_std: float = 1.0
    kl_weight: float = 1e-3

    # Optional small epsilon safety net.  Defaults to pure posterior sampling.
    epsilon_start: float = 0.0
    epsilon_end: float = 0.0
    epsilon_decay_steps: int = 20_000

    uncertainty_mc_samples: int = 16
    device: str = "cpu"
    seed: int = 42

    def __post_init__(self) -> None:
        if self.posterior_sampling not in {"shared", "independent"}:
            raise ValueError("posterior_sampling must be 'shared' or 'independent'.")
        if self.prior_std <= 0.0:
            raise ValueError("prior_std must be > 0.")
        if not 0.0 < self.min_posterior_std <= self.initial_posterior_std:
            raise ValueError("Require 0 < min_posterior_std <= initial_posterior_std.")
        if self.max_posterior_std < self.initial_posterior_std:
            raise ValueError("max_posterior_std must be >= initial_posterior_std.")
        if self.kl_weight < 0.0:
            raise ValueError("kl_weight must be >= 0.")


class VariationalLinearQHead(nn.Module):
    """Mean-field Gaussian posterior over a linear Q-value head."""

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

        self.weight_mu = nn.Parameter(torch.empty(self.action_dim, self.feature_dim))
        self.bias_mu = nn.Parameter(torch.empty(self.action_dim))
        self.weight_rho = nn.Parameter(
            torch.full((self.action_dim, self.feature_dim), self._inverse_softplus(initial_std))
        )
        self.bias_rho = nn.Parameter(
            torch.full((self.action_dim,), self._inverse_softplus(initial_std))
        )
        nn.init.kaiming_uniform_(self.weight_mu, a=math.sqrt(5.0))
        fan_in = max(1, self.feature_dim)
        bound = 1.0 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias_mu, -bound, bound)

    @staticmethod
    def _inverse_softplus(value: float) -> float:
        value = max(float(value), 1e-8)
        return float(math.log(math.expm1(value)))

    def weight_std(self) -> torch.Tensor:
        return F.softplus(self.weight_rho).clamp(self.min_std, self.max_std)

    def bias_std(self) -> torch.Tensor:
        return F.softplus(self.bias_rho).clamp(self.min_std, self.max_std)

    def sample_parameters(self, n_samples: int) -> tuple[torch.Tensor, torch.Tensor]:
        n_samples = int(n_samples)
        if n_samples <= 0:
            raise ValueError("n_samples must be positive.")
        weight_eps = torch.randn(
            n_samples,
            self.action_dim,
            self.feature_dim,
            dtype=self.weight_mu.dtype,
            device=self.weight_mu.device,
        )
        bias_eps = torch.randn(
            n_samples,
            self.action_dim,
            dtype=self.bias_mu.dtype,
            device=self.bias_mu.device,
        )
        weight = self.weight_mu.unsqueeze(0) + self.weight_std().unsqueeze(0) * weight_eps
        bias = self.bias_mu.unsqueeze(0) + self.bias_std().unsqueeze(0) * bias_eps
        return weight, bias

    def mean_q(self, features: torch.Tensor) -> torch.Tensor:
        return F.linear(features, self.weight_mu, self.bias_mu)

    @staticmethod
    def sampled_q(
        features: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate one sampled head per agent.

        Supported shapes:
          features [N,F], weight [N,A,F], bias [N,A] -> [N,A]
          features [B,N,F], weight [N,A,F], bias [N,A] -> [B,N,A]
        """
        if features.dim() == 2:
            return torch.einsum("nf,naf->na", features, weight) + bias
        if features.dim() == 3:
            return torch.einsum("bnf,naf->bna", features, weight) + bias.unsqueeze(0)
        raise ValueError(f"Unsupported feature shape {tuple(features.shape)}")

    def kl_divergence(self) -> torch.Tensor:
        prior_var = self.prior_std ** 2

        def gaussian_kl(mu: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
            var = std.square()
            return 0.5 * torch.sum(
                (var + mu.square()) / prior_var
                - 1.0
                + math.log(prior_var)
                - torch.log(var)
            )

        return gaussian_kl(self.weight_mu, self.weight_std()) + gaussian_kl(
            self.bias_mu, self.bias_std()
        )

    @property
    def posterior_parameter_count(self) -> int:
        return int(self.weight_mu.numel() + self.bias_mu.numel())

    @torch.no_grad()
    def clamp_rho_(self) -> None:
        min_rho = self._inverse_softplus(self.min_std)
        max_rho = self._inverse_softplus(self.max_std)
        self.weight_rho.clamp_(min_rho, max_rho)
        self.bias_rho.clamp_(min_rho, max_rho)

    @torch.no_grad()
    def diagnostics(self) -> dict[str, float]:
        std = torch.cat((self.weight_std().flatten(), self.bias_std().flatten()))
        kl_per_parameter = self.kl_divergence() / max(1, self.posterior_parameter_count)
        return {
            "posterior_std_mean": float(std.mean().item()),
            "posterior_std_min": float(std.min().item()),
            "posterior_std_max": float(std.max().item()),
            "posterior_mu_abs_mean": float(
                torch.cat((self.weight_mu.abs().flatten(), self.bias_mu.abs().flatten())).mean().item()
            ),
            "posterior_kl_per_parameter": float(kl_per_parameter.item()),
        }


class BayesianLocalQMIXAgent:
    """Bayesian-QMIX with decentralized Thompson-style action selection.

    Centralized training:
      sampled local utilities -> monotonic mixer -> team TD loss + variational KL.

    Decentralized execution:
      each UAV selects argmax_a Q_i(o_i, a) from a posterior sample.  ``act`` has no
      global-state argument and never invokes the mixer.
    """

    def __init__(self, cfg: BayesianLocalQMIXConfig) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.rng = np.random.default_rng(cfg.seed)
        c, h, w = cfg.obs_shape
        if h != w:
            raise ValueError("GridFeatureNet assumes a square grid.")

        self.feature_net = GridFeatureNet(c, h, cfg.feature_dim).to(self.device)
        self.target_feature_net = GridFeatureNet(c, h, cfg.feature_dim).to(self.device)
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
        self.replay = LocalJointReplayBuffer(
            cfg.replay_capacity,
            cfg.n_agents,
            cfg.obs_shape,
            cfg.state_dim,
            cfg.action_dim,
            cfg.seed,
        )
        self.train_steps = 0
        self.env_steps = 0
        self._episode_weight: torch.Tensor | None = None
        self._episode_bias: torch.Tensor | None = None
        self.resample_policy()
        self.last_train_stats: dict[str, float] = {
            "loss": 0.0,
            "td_loss": 0.0,
            "kl_loss": 0.0,
            "q_tot_mean": 0.0,
            "target_mean": 0.0,
        }

    @property
    def posterior_sampling(self) -> str:
        return self.cfg.posterior_sampling

    def epsilon(self) -> float:
        fraction = min(1.0, self.env_steps / max(1, self.cfg.epsilon_decay_steps))
        return float(
            self.cfg.epsilon_start
            + fraction * (self.cfg.epsilon_end - self.cfg.epsilon_start)
        )

    @staticmethod
    def masked_argmax(q_values: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        masks = masks.bool()
        if q_values.shape != masks.shape:
            raise ValueError(f"q shape {q_values.shape} != mask shape {masks.shape}")
        if not torch.all(masks.any(dim=-1)):
            raise ValueError("At least one agent has no valid action")
        masked = q_values.masked_fill(~masks, torch.finfo(q_values.dtype).min)
        return masked.argmax(dim=-1)

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
    def episode_sample_distance(self) -> float:
        if self._episode_weight is None or self.cfg.n_agents < 2:
            return 0.0
        flat = torch.cat(
            (self._episode_weight.flatten(1), self._episode_bias.flatten(1)), dim=1
        )
        distances = []
        for first in range(self.cfg.n_agents):
            for second in range(first + 1, self.cfg.n_agents):
                denominator = 0.5 * (
                    flat[first].norm(p=2) + flat[second].norm(p=2)
                ).clamp_min(1e-8)
                distances.append((flat[first] - flat[second]).norm(p=2) / denominator)
        return float(torch.stack(distances).mean().item()) if distances else 0.0

    @torch.no_grad()
    def q_values(self, obs_all: np.ndarray, *, use_sample: bool) -> np.ndarray:
        obs = torch.as_tensor(obs_all, dtype=torch.float32, device=self.device)
        if obs.shape != (self.cfg.n_agents, *self.cfg.obs_shape):
            raise ValueError(
                f"Expected obs {(self.cfg.n_agents, *self.cfg.obs_shape)}, got {tuple(obs.shape)}"
            )
        features = self.feature_net(obs)
        if use_sample:
            if self._episode_weight is None or self._episode_bias is None:
                self.resample_policy()
            q_values = self.head.sampled_q(
                features,
                self._episode_weight,
                self._episode_bias,
            )
        else:
            q_values = self.head.mean_q(features)
        return q_values.cpu().numpy()

    @torch.no_grad()
    def act(
        self,
        obs_all: np.ndarray,
        *,
        action_masks: np.ndarray,
        use_sample: bool = True,
        explore: bool = True,
    ) -> np.ndarray:
        q_values = self.q_values(obs_all, use_sample=use_sample)
        masks = np.asarray(action_masks, dtype=bool)
        expected = (self.cfg.n_agents, self.cfg.action_dim)
        if masks.shape != expected:
            raise ValueError(f"Expected masks {expected}, got {masks.shape}")

        actions = np.zeros(self.cfg.n_agents, dtype=np.int64)
        for agent_id in range(self.cfg.n_agents):
            allowed = np.flatnonzero(masks[agent_id])
            if allowed.size == 0:
                raise ValueError(f"Agent {agent_id} has no valid action")
            if explore and self.rng.random() < self.epsilon():
                actions[agent_id] = int(self.rng.choice(allowed))
            else:
                local_q = q_values[agent_id].copy()
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

    def train_step(self) -> dict[str, float]:
        if len(self.replay) < self.cfg.batch_size:
            return {
                **self.last_train_stats,
                **self.head.diagnostics(),
                "epsilon": self.epsilon(),
                "episode_sample_distance": self.episode_sample_distance(),
            }
        batch = self.replay.sample(self.cfg.batch_size)
        stats = self._gradient_update(batch)
        self.train_steps += 1
        if self.train_steps % self.cfg.target_update_period == 0:
            self.target_feature_net.load_state_dict(self.feature_net.state_dict())
            self.target_head.load_state_dict(self.head.state_dict())
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.last_train_stats = stats
        return {
            **stats,
            **self.head.diagnostics(),
            "epsilon": self.epsilon(),
            "episode_sample_distance": self.episode_sample_distance(),
        }

    def _gradient_update(self, batch: LocalJointBatch) -> dict[str, float]:
        cfg = self.cfg
        batch_size = int(batch.obs.shape[0])
        n_agents = cfg.n_agents

        obs = torch.as_tensor(batch.obs, dtype=torch.float32, device=self.device)
        states = torch.as_tensor(batch.states, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch.actions, dtype=torch.long, device=self.device)
        rewards = torch.as_tensor(batch.rewards, dtype=torch.float32, device=self.device)
        next_obs = torch.as_tensor(batch.next_obs, dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(batch.next_states, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(batch.dones, dtype=torch.float32, device=self.device)
        next_masks = torch.as_tensor(
            batch.next_action_masks, dtype=torch.bool, device=self.device
        )

        flat_obs = obs.reshape(batch_size * n_agents, *cfg.obs_shape)
        features = self.feature_net(flat_obs).reshape(batch_size, n_agents, cfg.feature_dim)
        sampled_weight, sampled_bias = self._sample_training_head()
        q_all = self.head.sampled_q(features, sampled_weight, sampled_bias)
        chosen_q = q_all.gather(2, actions.unsqueeze(-1)).squeeze(-1)
        q_tot = self.mixer(chosen_q, states)

        with torch.no_grad():
            flat_next = next_obs.reshape(batch_size * n_agents, *cfg.obs_shape)
            next_online_features = self.feature_net(flat_next).reshape(
                batch_size, n_agents, cfg.feature_dim
            )
            next_online_q = self.head.mean_q(next_online_features)
            next_actions = self.masked_argmax(next_online_q, next_masks)

            next_target_features = self.target_feature_net(flat_next).reshape(
                batch_size, n_agents, cfg.feature_dim
            )
            next_target_q = self.target_head.mean_q(next_target_features)
            next_chosen = next_target_q.gather(
                2, next_actions.unsqueeze(-1)
            ).squeeze(-1)
            target_q_tot = self.target_mixer(next_chosen, next_states)
            targets = rewards + cfg.gamma * (1.0 - dones) * target_q_tot

        td_loss = F.mse_loss(q_tot, targets)
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

    @torch.no_grad()
    def uncertainty_metrics(self, sample_size: int = 64) -> dict[str, float]:
        if len(self.replay) == 0:
            return {
                "predictive_epistemic_std_mean": float("nan"),
                "predictive_epistemic_std_max_action_mean": float("nan"),
            }
        count = min(int(sample_size), len(self.replay))
        indices = self.rng.choice(len(self.replay), size=count, replace=False)
        obs = torch.as_tensor(
            self.replay.obs[indices], dtype=torch.float32, device=self.device
        )
        flat_obs = obs.reshape(count * self.cfg.n_agents, *self.cfg.obs_shape)
        features = self.feature_net(flat_obs)
        mc = max(2, int(self.cfg.uncertainty_mc_samples))
        weight, bias = self.head.sample_parameters(mc)
        # [MC, BN, A]
        q_samples = torch.einsum("bf,maf->mba", features, weight) + bias[:, None, :]
        std = q_samples.std(dim=0, unbiased=True)
        return {
            "predictive_epistemic_std_mean": float(std.mean().item()),
            "predictive_epistemic_std_max_action_mean": float(
                std.max(dim=1).values.mean().item()
            ),
        }

    def save(self, path: str) -> None:
        torch.save(
            {
                "cfg": asdict(self.cfg),
                "feature_net": self.feature_net.state_dict(),
                "target_feature_net": self.target_feature_net.state_dict(),
                "head": self.head.state_dict(),
                "target_head": self.target_head.state_dict(),
                "mixer": self.mixer.state_dict(),
                "target_mixer": self.target_mixer.state_dict(),
                "optim": self.optim.state_dict(),
                "train_steps": self.train_steps,
                "env_steps": self.env_steps,
                "last_train_stats": self.last_train_stats,
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
        self.last_train_stats = checkpoint.get("last_train_stats", self.last_train_stats)
        self.resample_policy()

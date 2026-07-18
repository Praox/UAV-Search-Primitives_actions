from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from uav_search_belief20.agents.replay_buffer import Batch, ReplayBuffer
from uav_search_belief20.models.networks import GridFeatureNet


@dataclass
class BDQNConfig:
    obs_shape: tuple[int, int, int] = (6, 20, 20)
    action_dim: int = 5
    feature_dim: int = 128
    gamma: float = 0.99
    n_step: int = 3
    lr: float = 1e-4
    batch_size: int = 64
    replay_capacity: int = 50_000

    target_tau: float = 0.005
    target_update_period: int = 500

    posterior_update_period: int = 500
    posterior_replay_size: int = 8192
    posterior_chunk_size: int = 512
    posterior_min_samples: int = 1000

    blr_lambda: float = 1.0
    blr_noise_var: float = 1.0
    posterior_jitter: float = 1e-6
    posterior_mode: str = "rebuild"
    freeze_feature_after_steps: int | None = None

    huber_delta: float = 1.0
    normalize_features: bool = True
    grad_clip_norm: float = 10.0
    device: str = "cpu"
    seed: int = 42


class BayesianLinearHead:
    """Per-action Bayesian linear regression.

    For action ``a`` the model is

        y = phi^T w_a + epsilon,
        w_a ~ N(0, lambda^{-1} I),
        epsilon ~ N(0, noise_var).

    Sufficient statistics are

        A_a = lambda I + noise_var^{-1} sum phi phi^T,
        b_a = noise_var^{-1} sum phi y.
    """

    def __init__(
        self,
        action_dim: int,
        feature_dim: int,
        lam: float = 1.0,
        noise_var: float = 1.0,
        jitter: float = 1e-6,
        seed: int = 42,
    ) -> None:
        if lam <= 0 or noise_var <= 0 or jitter <= 0:
            raise ValueError("lam, noise_var and jitter must be > 0.")
        self.action_dim = int(action_dim)
        self.feature_dim = int(feature_dim)
        self.lam = float(lam)
        self.noise_var = float(noise_var)
        self.jitter = float(jitter)
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self) -> None:
        eye = np.eye(self.feature_dim, dtype=np.float64)
        self.A = np.stack(
            [self.lam * eye.copy() for _ in range(self.action_dim)]
        )
        self.b = np.zeros(
            (self.action_dim, self.feature_dim), dtype=np.float64
        )
        self.mu = np.zeros(
            (self.action_dim, self.feature_dim), dtype=np.float32
        )
        self.cov = np.stack(
            [
                (eye / self.lam).astype(np.float32)
                for _ in range(self.action_dim)
            ]
        )
        self._chol_cov = np.stack(
            [
                np.linalg.cholesky(
                    self.cov[action].astype(np.float64)
                    + self.jitter * eye
                ).astype(np.float32)
                for action in range(self.action_dim)
            ]
        )
        self.sampled_w = self.mu.copy()
        self.n_obs = np.zeros((self.action_dim,), dtype=np.int64)

    def accumulate(
        self,
        phi: np.ndarray,
        actions: np.ndarray,
        targets: np.ndarray,
    ) -> None:
        phi = np.asarray(phi, dtype=np.float64)
        actions = np.asarray(actions, dtype=np.int64).reshape(-1)
        targets = np.asarray(targets, dtype=np.float64).reshape(-1)
        if phi.ndim != 2 or phi.shape[1] != self.feature_dim:
            raise ValueError(
                f"phi must be [N, {self.feature_dim}], got {phi.shape}."
            )
        if not (len(phi) == len(actions) == len(targets)):
            raise ValueError(
                "phi, actions and targets must have matching length."
            )

        inverse_noise = 1.0 / self.noise_var
        for action in range(self.action_dim):
            selected = actions == action
            if not np.any(selected):
                continue
            x = phi[selected]
            y = targets[selected]
            self.A[action] += inverse_noise * (x.T @ x)
            self.b[action] += inverse_noise * (x.T @ y)
            self.n_obs[action] += int(selected.sum())

    def finalize(self) -> None:
        eye = np.eye(self.feature_dim, dtype=np.float64)
        for action in range(self.action_dim):
            precision = (
                0.5 * (self.A[action] + self.A[action].T)
                + self.jitter * eye
            )
            try:
                chol_precision = np.linalg.cholesky(precision)
            except np.linalg.LinAlgError:
                chol_precision = np.linalg.cholesky(
                    precision + 1e-4 * eye
                )

            temporary = np.linalg.solve(chol_precision, self.b[action])
            mean = np.linalg.solve(chol_precision.T, temporary)
            inverse_l = np.linalg.solve(chol_precision, eye)
            covariance = inverse_l.T @ inverse_l
            covariance = 0.5 * (covariance + covariance.T)

            self.mu[action] = mean.astype(np.float32)
            self.cov[action] = covariance.astype(np.float32)
            self._chol_cov[action] = np.linalg.cholesky(
                covariance + self.jitter * eye
            ).astype(np.float32)

    def update(
        self,
        phi: np.ndarray,
        actions: np.ndarray,
        targets: np.ndarray,
        *,
        reset: bool = False,
    ) -> None:
        if reset:
            self.reset()
        self.accumulate(phi, actions, targets)
        self.finalize()

    def sample(self) -> None:
        samples = []
        for action in range(self.action_dim):
            z = self.rng.standard_normal(self.feature_dim).astype(np.float32)
            samples.append(
                self.mu[action] + self._chol_cov[action] @ z
            )
        self.sampled_w = np.stack(samples, axis=0).astype(np.float32)

    def predictive_variance(
        self,
        phi: np.ndarray,
        *,
        include_noise: bool = False,
    ) -> np.ndarray:
        x = np.asarray(phi, dtype=np.float64)
        single = x.ndim == 1
        if single:
            x = x[None, :]
        if x.ndim != 2 or x.shape[1] != self.feature_dim:
            raise ValueError(
                f"phi must be [N, {self.feature_dim}], got {x.shape}."
            )

        output = np.empty(
            (x.shape[0], self.action_dim), dtype=np.float64
        )
        for action in range(self.action_dim):
            covariance = self.cov[action].astype(np.float64)
            output[:, action] = np.einsum(
                "bi,ij,bj->b", x, covariance, x
            )
        output = np.maximum(output, 0.0)
        if include_noise:
            output += self.noise_var
        result = output.astype(np.float32)
        return result[0] if single else result

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {
            "A": torch.as_tensor(self.A),
            "b": torch.as_tensor(self.b),
            "mu": torch.as_tensor(self.mu),
            "cov": torch.as_tensor(self.cov),
            "sampled_w": torch.as_tensor(self.sampled_w),
            "n_obs": torch.as_tensor(self.n_obs),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if "A" in state and "b" in state:
            self.A = state["A"].detach().cpu().numpy().astype(np.float64)
            self.b = state["b"].detach().cpu().numpy().astype(np.float64)
            self.n_obs = (
                state.get(
                    "n_obs",
                    torch.zeros(self.action_dim, dtype=torch.int64),
                )
                .detach()
                .cpu()
                .numpy()
                .astype(np.int64)
            )
            self.finalize()
            self.sampled_w = (
                state.get("sampled_w", state["mu"])
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            return

        self.mu = state["mu"].detach().cpu().numpy().astype(np.float32)
        self.cov = state["cov"].detach().cpu().numpy().astype(np.float32)
        self.sampled_w = (
            state.get("sampled_w", state["mu"])
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        eye = np.eye(self.feature_dim, dtype=np.float64)
        self.A = np.empty(
            (self.action_dim, self.feature_dim, self.feature_dim),
            dtype=np.float64,
        )
        self.b = np.empty(
            (self.action_dim, self.feature_dim), dtype=np.float64
        )
        for action in range(self.action_dim):
            covariance = self.cov[action].astype(np.float64)
            precision = np.linalg.inv(covariance + self.jitter * eye)
            self.A[action] = precision
            self.b[action] = precision @ self.mu[action].astype(
                np.float64
            )
        self.n_obs = np.zeros((self.action_dim,), dtype=np.int64)
        self._chol_cov = np.stack(
            [
                np.linalg.cholesky(
                    self.cov[action].astype(np.float64)
                    + self.jitter * eye
                ).astype(np.float32)
                for action in range(self.action_dim)
            ]
        )


class BDQNAgent:
    """Approximate BDQN with masked n-step Bellman targets.

    For the thesis comparison, warm-start this feature extractor from the
    corrected DDQN and freeze it. That removes the zero-gradient initial phase
    and isolates posterior action selection from representation learning.
    """

    def __init__(self, cfg: BDQNConfig):
        if cfg.posterior_mode not in {"rebuild", "cumulative"}:
            raise ValueError(
                "posterior_mode must be 'rebuild' or 'cumulative'."
            )
        if (
            cfg.posterior_mode == "cumulative"
            and cfg.freeze_feature_after_steps is None
        ):
            raise ValueError(
                "posterior_mode='cumulative' requires frozen features."
            )
        if not 0.0 <= cfg.target_tau <= 1.0:
            raise ValueError("target_tau must be in [0, 1]")

        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.rng = np.random.default_rng(cfg.seed)
        channels, height, width = cfg.obs_shape
        if height != width:
            raise ValueError("GridFeatureNet assumes a square grid.")

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
        self.target_feature_net.load_state_dict(
            self.feature_net.state_dict()
        )
        self.optim = torch.optim.Adam(
            self.feature_net.parameters(), lr=cfg.lr
        )

        self.blr = BayesianLinearHead(
            cfg.action_dim,
            cfg.feature_dim,
            cfg.blr_lambda,
            cfg.blr_noise_var,
            cfg.posterior_jitter,
            cfg.seed,
        )
        self.blr.sample()
        self.replay = ReplayBuffer(
            cfg.replay_capacity,
            cfg.obs_shape,
            seed=cfg.seed,
            action_dim=cfg.action_dim,
            n_step=cfg.n_step,
            gamma=cfg.gamma,
        )
        self.train_steps = 0
        self.posterior_rebuilds = 0
        self.features_frozen = False
        self.last_predictive_stats = {
            "predictive_epistemic_std_mean": float("nan"),
            "predictive_selected_std_mean": float("nan"),
            "predictive_max_std_mean": float("nan"),
            "td_residual_variance": float("nan"),
        }

    @staticmethod
    def _masked_argmax(
        q_values: torch.Tensor, masks: torch.Tensor
    ) -> torch.Tensor:
        masks = masks.bool()
        if q_values.shape != masks.shape:
            raise ValueError(
                f"q shape {q_values.shape} != mask shape {masks.shape}"
            )
        if not torch.all(masks.any(dim=-1)):
            raise ValueError("At least one sample has no valid next action.")
        return q_values.masked_fill(
            ~masks, torch.finfo(q_values.dtype).min
        ).argmax(dim=-1)

    def load_feature_extractor_from_dqn_checkpoint(
        self,
        checkpoint_path: str,
        *,
        freeze: bool = True,
    ) -> None:
        """Load the corrected DDQN representation into BDQN.

        The source DDQN and this BDQN must use the same observation shape,
        feature dimension and compact network architecture.
        """

        try:
            checkpoint = torch.load(
                checkpoint_path,
                map_location=self.device,
                weights_only=False,
            )
        except TypeError:
            checkpoint = torch.load(
                checkpoint_path, map_location=self.device
            )
        q_state = checkpoint["q_net"]
        prefix = "feature_net."
        feature_state = {
            key[len(prefix) :]: value
            for key, value in q_state.items()
            if key.startswith(prefix)
        }
        if not feature_state:
            raise ValueError(
                "Checkpoint has no q_net.feature_net parameters."
            )
        self.feature_net.load_state_dict(feature_state, strict=True)
        self.target_feature_net.load_state_dict(feature_state, strict=True)
        if freeze:
            self.freeze_features()

    def freeze_features(self) -> None:
        self.features_frozen = True
        for parameter in self.feature_net.parameters():
            parameter.requires_grad_(False)

    def resample_policy(self) -> None:
        self.blr.sample()

    def _maybe_freeze_features(self) -> None:
        freeze_at = self.cfg.freeze_feature_after_steps
        if (
            freeze_at is not None
            and not self.features_frozen
            and self.train_steps >= freeze_at
        ):
            self.freeze_features()

    @torch.no_grad()
    def _features(self, obs: np.ndarray) -> np.ndarray:
        tensor = torch.as_tensor(
            obs, dtype=torch.float32, device=self.device
        )
        if tensor.ndim == len(self.cfg.obs_shape):
            tensor = tensor.unsqueeze(0)
        return self.feature_net(tensor).cpu().numpy()

    @torch.no_grad()
    def act(
        self,
        obs: np.ndarray,
        use_sample: bool = True,
        action_mask: np.ndarray | None = None,
    ) -> int:
        phi = self._features(obs)[0]
        weights = self.blr.sampled_w if use_sample else self.blr.mu
        q_values = weights @ phi
        if action_mask is not None:
            allowed = np.asarray(action_mask, dtype=bool)
            if not np.any(allowed):
                raise ValueError("No allowed action in action_mask.")
            q_values = q_values.copy()
            q_values[~allowed] = -np.inf
        return int(np.argmax(q_values))

    @torch.no_grad()
    def q_mean_and_variance(
        self,
        obs: np.ndarray,
        include_noise: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        phi = self._features(obs)
        means = phi @ self.blr.mu.T
        variances = self.blr.predictive_variance(
            phi, include_noise=include_noise
        )
        return means, variances

    def train_step(self) -> dict[str, float]:
        if len(self.replay) < self.cfg.batch_size:
            return {
                "loss": 0.0,
                "q_mean": 0.0,
                "posterior_rebuilds": float(self.posterior_rebuilds),
                **self.last_predictive_stats,
            }

        self._maybe_freeze_features()
        batch = self.replay.sample(self.cfg.batch_size)
        if self.features_frozen:
            loss = 0.0
            q_mean = self._batch_q_mean(batch)
        else:
            loss, q_mean = self._gradient_update(batch)

        self.train_steps += 1
        self._update_target_features()

        if (
            len(self.replay) >= self.cfg.posterior_min_samples
            and self.train_steps % self.cfg.posterior_update_period == 0
        ):
            if self.cfg.posterior_mode == "rebuild":
                self.rebuild_posterior_from_replay()
            else:
                posterior_batch = self.replay.sample(
                    min(self.cfg.posterior_replay_size, len(self.replay))
                )
                self.update_posterior_cumulative(posterior_batch)
            self.resample_policy()
            self.last_predictive_stats = self.uncertainty_metrics()

        return {
            "loss": float(loss),
            "q_mean": float(q_mean),
            "posterior_rebuilds": float(self.posterior_rebuilds),
            **self.last_predictive_stats,
        }

    def _update_target_features(self) -> None:
        tau = float(self.cfg.target_tau)
        if tau > 0.0:
            with torch.no_grad():
                for target, source in zip(
                    self.target_feature_net.parameters(),
                    self.feature_net.parameters(),
                ):
                    target.mul_(1.0 - tau).add_(source, alpha=tau)
        elif self.train_steps % self.cfg.target_update_period == 0:
            self.target_feature_net.load_state_dict(
                self.feature_net.state_dict()
            )

    @torch.no_grad()
    def _batch_q_mean(self, batch: Batch) -> float:
        obs = torch.as_tensor(
            batch.obs, dtype=torch.float32, device=self.device
        )
        phi = self.feature_net(obs)
        mean_weights = torch.as_tensor(
            self.blr.mu, dtype=torch.float32, device=self.device
        )
        q_all = phi @ mean_weights.t()
        actions = torch.as_tensor(
            batch.actions, dtype=torch.long, device=self.device
        )
        q = q_all.gather(1, actions[:, None]).squeeze(1)
        return float(q.mean().item())

    def _gradient_update(self, batch: Batch) -> tuple[float, float]:
        obs = torch.as_tensor(
            batch.obs, dtype=torch.float32, device=self.device
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
        dones = torch.as_tensor(
            batch.dones, dtype=torch.float32, device=self.device
        )
        discounts = torch.as_tensor(
            batch.discounts, dtype=torch.float32, device=self.device
        )
        next_masks = torch.as_tensor(
            batch.next_action_masks, dtype=torch.bool, device=self.device
        )
        mean_weights = torch.as_tensor(
            self.blr.mu, dtype=torch.float32, device=self.device
        )

        phi = self.feature_net(obs)
        q_all = phi @ mean_weights.t()
        q = q_all.gather(1, actions[:, None]).squeeze(1)
        with torch.no_grad():
            next_online_phi = self.feature_net(next_obs)
            next_online_q = next_online_phi @ mean_weights.t()
            next_actions = self._masked_argmax(
                next_online_q, next_masks
            )
            next_target_phi = self.target_feature_net(next_obs)
            next_target_q = next_target_phi @ mean_weights.t()
            next_q = next_target_q.gather(
                1, next_actions[:, None]
            ).squeeze(1)
            targets = rewards + discounts * (1.0 - dones) * next_q

        loss = F.smooth_l1_loss(
            q, targets, beta=float(self.cfg.huber_delta)
        )
        self.optim.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(
            self.feature_net.parameters(), self.cfg.grad_clip_norm
        )
        self.optim.step()
        return float(loss.item()), float(q.mean().item())

    def _posterior_replay_indices(self) -> np.ndarray:
        size = len(self.replay)
        sample_size = min(self.cfg.posterior_replay_size, size)
        if sample_size == size:
            return np.arange(size, dtype=np.int64)
        return self.rng.choice(
            size, size=sample_size, replace=False
        ).astype(np.int64)

    @torch.no_grad()
    def _compute_targets_and_features(
        self,
        obs_np,
        actions_np,
        rewards_np,
        next_obs_np,
        dones_np,
        discounts_np,
        next_masks_np,
        *,
        mean_w_np=None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        obs = torch.as_tensor(
            obs_np, dtype=torch.float32, device=self.device
        )
        next_obs = torch.as_tensor(
            next_obs_np, dtype=torch.float32, device=self.device
        )
        if mean_w_np is None:
            mean_w_np = self.blr.mu.copy()
        mean_weights = torch.as_tensor(
            mean_w_np, dtype=torch.float32, device=self.device
        )

        phi = self.feature_net(obs)
        next_online_phi = self.feature_net(next_obs)
        next_online_q = next_online_phi @ mean_weights.t()
        next_masks = torch.as_tensor(
            next_masks_np, dtype=torch.bool, device=self.device
        )
        next_actions = self._masked_argmax(next_online_q, next_masks)

        next_target_phi = self.target_feature_net(next_obs)
        next_target_q = next_target_phi @ mean_weights.t()
        next_q = next_target_q.gather(
            1, next_actions[:, None]
        ).squeeze(1)

        rewards = torch.as_tensor(
            rewards_np, dtype=torch.float32, device=self.device
        )
        dones = torch.as_tensor(
            dones_np, dtype=torch.float32, device=self.device
        )
        discounts = torch.as_tensor(
            discounts_np, dtype=torch.float32, device=self.device
        )
        targets = rewards + discounts * (1.0 - dones) * next_q
        return (
            phi.cpu().numpy().astype(np.float32),
            np.asarray(actions_np, dtype=np.int64),
            targets.cpu().numpy().astype(np.float32),
        )

    @torch.no_grad()
    def rebuild_posterior_from_replay(self) -> None:
        indices = self._posterior_replay_indices()
        target_mean_weights = self.blr.mu.copy()
        self.blr.reset()
        chunk_size = max(1, int(self.cfg.posterior_chunk_size))
        for start in range(0, len(indices), chunk_size):
            index = indices[start : start + chunk_size]
            phi, actions, targets = self._compute_targets_and_features(
                self.replay.obs[index],
                self.replay.actions[index],
                self.replay.rewards[index],
                self.replay.next_obs[index],
                self.replay.dones[index],
                self.replay.discounts[index],
                self.replay.next_action_masks[index],
                mean_w_np=target_mean_weights,
            )
            self.blr.accumulate(phi, actions, targets)
        self.blr.finalize()
        self.posterior_rebuilds += 1

    @torch.no_grad()
    def update_posterior_cumulative(self, batch: Batch) -> None:
        if not self.features_frozen:
            raise RuntimeError(
                "Cumulative posterior update requires fixed features."
            )
        phi, actions, targets = self._compute_targets_and_features(
            batch.obs,
            batch.actions,
            batch.rewards,
            batch.next_obs,
            batch.dones,
            batch.discounts,
            batch.next_action_masks,
        )
        self.blr.update(phi, actions, targets, reset=False)

    @torch.no_grad()
    def estimate_td_residual_variance(
        self, sample_size: int = 512
    ) -> float:
        if len(self.replay) == 0:
            return float("nan")
        count = min(int(sample_size), len(self.replay))
        indices = self.rng.choice(
            len(self.replay), size=count, replace=False
        )
        phi, actions, targets = self._compute_targets_and_features(
            self.replay.obs[indices],
            self.replay.actions[indices],
            self.replay.rewards[indices],
            self.replay.next_obs[indices],
            self.replay.dones[indices],
            self.replay.discounts[indices],
            self.replay.next_action_masks[indices],
        )
        predictions = np.einsum(
            "nf,nf->n", phi, self.blr.mu[actions]
        )
        return float(np.mean(np.square(targets - predictions)))

    @torch.no_grad()
    def uncertainty_metrics(
        self, sample_size: int = 256
    ) -> dict[str, float]:
        if len(self.replay) == 0:
            return {
                "predictive_epistemic_std_mean": float("nan"),
                "predictive_selected_std_mean": float("nan"),
                "predictive_max_std_mean": float("nan"),
                "td_residual_variance": float("nan"),
            }
        count = min(int(sample_size), len(self.replay))
        indices = self.rng.choice(
            len(self.replay), size=count, replace=False
        )
        phi = self._features(self.replay.obs[indices])
        variances = self.blr.predictive_variance(
            phi, include_noise=False
        )
        standard_deviation = np.sqrt(np.maximum(variances, 0.0))
        selected = standard_deviation[
            np.arange(count), self.replay.actions[indices]
        ]
        return {
            "predictive_epistemic_std_mean": float(
                standard_deviation.mean()
            ),
            "predictive_selected_std_mean": float(selected.mean()),
            "predictive_max_std_mean": float(
                standard_deviation.max(axis=1).mean()
            ),
            "td_residual_variance": self.estimate_td_residual_variance(),
        }

    def save(self, path: str) -> None:
        torch.save(
            {
                "cfg": self.cfg.__dict__,
                "feature_net": self.feature_net.state_dict(),
                "target_feature_net": self.target_feature_net.state_dict(),
                "optim": self.optim.state_dict(),
                "blr": self.blr.state_dict(),
                "train_steps": self.train_steps,
                "posterior_rebuilds": self.posterior_rebuilds,
                "features_frozen": self.features_frozen,
                "last_predictive_stats": self.last_predictive_stats,
            },
            path,
        )

    def load(self, path: str) -> None:
        try:
            checkpoint = torch.load(
                path, map_location=self.device, weights_only=False
            )
        except TypeError:
            checkpoint = torch.load(path, map_location=self.device)
        self.feature_net.load_state_dict(checkpoint["feature_net"])
        self.target_feature_net.load_state_dict(
            checkpoint.get(
                "target_feature_net", checkpoint["feature_net"]
            )
        )
        if "optim" in checkpoint:
            self.optim.load_state_dict(checkpoint["optim"])
        self.blr.load_state_dict(checkpoint["blr"])
        self.train_steps = int(checkpoint.get("train_steps", 0))
        self.posterior_rebuilds = int(
            checkpoint.get("posterior_rebuilds", 0)
        )
        self.features_frozen = bool(
            checkpoint.get("features_frozen", False)
        )
        self.last_predictive_stats = checkpoint.get(
            "last_predictive_stats", self.last_predictive_stats
        )
        if self.features_frozen:
            self.freeze_features()

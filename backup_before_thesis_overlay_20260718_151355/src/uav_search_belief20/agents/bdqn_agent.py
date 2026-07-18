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
    lr: float = 1e-4
    batch_size: int = 64
    replay_capacity: int = 50_000
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

    grad_clip_norm: float = 10.0
    device: str = "cpu"
    seed: int = 42


class BayesianLinearHead:
    """Per-action Bayesian linear regression with cumulative sufficient statistics."""

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
        self.A = np.stack([self.lam * eye.copy() for _ in range(self.action_dim)])
        self.b = np.zeros((self.action_dim, self.feature_dim), dtype=np.float64)
        self.mu = np.zeros((self.action_dim, self.feature_dim), dtype=np.float32)
        self.cov = np.stack([(eye / self.lam).astype(np.float32) for _ in range(self.action_dim)])
        self._chol_cov = np.stack([
            np.linalg.cholesky(self.cov[a].astype(np.float64) + self.jitter * eye).astype(np.float32)
            for a in range(self.action_dim)
        ])
        self.sampled_w = self.mu.copy()
        self.n_obs = np.zeros((self.action_dim,), dtype=np.int64)

    def accumulate(self, phi: np.ndarray, actions: np.ndarray, targets: np.ndarray) -> None:
        phi = np.asarray(phi, dtype=np.float64)
        actions = np.asarray(actions, dtype=np.int64).reshape(-1)
        targets = np.asarray(targets, dtype=np.float64).reshape(-1)
        if phi.ndim != 2 or phi.shape[1] != self.feature_dim:
            raise ValueError(f"phi must be [N, {self.feature_dim}], got {phi.shape}.")
        if not (len(phi) == len(actions) == len(targets)):
            raise ValueError("phi, actions and targets must have matching length.")

        inv_noise = 1.0 / self.noise_var
        for action in range(self.action_dim):
            mask = actions == action
            if not np.any(mask):
                continue
            x = phi[mask]
            y = targets[mask]
            self.A[action] += inv_noise * (x.T @ x)
            self.b[action] += inv_noise * (x.T @ y)
            self.n_obs[action] += int(mask.sum())

    def finalize(self) -> None:
        eye = np.eye(self.feature_dim, dtype=np.float64)
        for action in range(self.action_dim):
            precision = 0.5 * (self.A[action] + self.A[action].T) + self.jitter * eye
            try:
                chol_precision = np.linalg.cholesky(precision)
            except np.linalg.LinAlgError:
                chol_precision = np.linalg.cholesky(precision + 1e-4 * eye)

            tmp = np.linalg.solve(chol_precision, self.b[action])
            mu = np.linalg.solve(chol_precision.T, tmp)
            inv_l = np.linalg.solve(chol_precision, eye)
            cov = inv_l.T @ inv_l
            cov = 0.5 * (cov + cov.T)

            self.mu[action] = mu.astype(np.float32)
            self.cov[action] = cov.astype(np.float32)
            self._chol_cov[action] = np.linalg.cholesky(cov + self.jitter * eye).astype(np.float32)

    def update(self, phi: np.ndarray, actions: np.ndarray, targets: np.ndarray, reset: bool = False) -> None:
        if reset:
            self.reset()
        self.accumulate(phi, actions, targets)
        self.finalize()

    def sample(self) -> None:
        ws = []
        for action in range(self.action_dim):
            z = self.rng.standard_normal(self.feature_dim).astype(np.float32)
            ws.append(self.mu[action] + self._chol_cov[action] @ z)
        self.sampled_w = np.stack(ws, axis=0).astype(np.float32)

    def predictive_variance(self, phi: np.ndarray, include_noise: bool = False) -> np.ndarray:
        x = np.asarray(phi, dtype=np.float64)
        single = x.ndim == 1
        if single:
            x = x[None, :]
        if x.ndim != 2 or x.shape[1] != self.feature_dim:
            raise ValueError(f"phi must be [N, {self.feature_dim}], got {x.shape}.")

        out = np.empty((x.shape[0], self.action_dim), dtype=np.float64)
        for action in range(self.action_dim):
            sigma = self.cov[action].astype(np.float64)
            out[:, action] = np.einsum("bi,ij,bj->b", x, sigma, x)
        out = np.maximum(out, 0.0)
        if include_noise:
            out += self.noise_var
        result = out.astype(np.float32)
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
            self.n_obs = state.get("n_obs", torch.zeros(self.action_dim, dtype=torch.int64)).detach().cpu().numpy().astype(np.int64)
            self.finalize()
            if "sampled_w" in state:
                self.sampled_w = state["sampled_w"].detach().cpu().numpy().astype(np.float32)
            else:
                self.sample()
            return

        self.mu = state["mu"].detach().cpu().numpy().astype(np.float32)
        self.cov = state["cov"].detach().cpu().numpy().astype(np.float32)
        self.sampled_w = state.get("sampled_w", state["mu"]).detach().cpu().numpy().astype(np.float32)
        self.A = np.empty((self.action_dim, self.feature_dim, self.feature_dim), dtype=np.float64)
        self.b = np.empty((self.action_dim, self.feature_dim), dtype=np.float64)
        eye = np.eye(self.feature_dim, dtype=np.float64)
        for action in range(self.action_dim):
            sigma = self.cov[action].astype(np.float64)
            precision = np.linalg.inv(sigma + self.jitter * eye)
            self.A[action] = precision
            self.b[action] = precision @ self.mu[action].astype(np.float64)
        self.n_obs = np.zeros((self.action_dim,), dtype=np.int64)
        self._chol_cov = np.stack([
            np.linalg.cholesky(self.cov[a].astype(np.float64) + self.jitter * eye).astype(np.float32)
            for a in range(self.action_dim)
        ])


class BDQNAgent:
    """BDQN with current-feature replay posterior rebuilds and uncertainty diagnostics."""

    def __init__(self, cfg: BDQNConfig):
        if cfg.posterior_mode not in {"rebuild", "cumulative"}:
            raise ValueError("posterior_mode must be 'rebuild' or 'cumulative'.")
        if cfg.posterior_mode == "cumulative" and cfg.freeze_feature_after_steps is None:
            raise ValueError("posterior_mode='cumulative' requires freeze_feature_after_steps.")

        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.rng = np.random.default_rng(cfg.seed)
        c, h, w = cfg.obs_shape
        if h != w:
            raise ValueError("GridFeatureNet assumes a square grid.")

        self.feature_net = GridFeatureNet(c, h, cfg.feature_dim).to(self.device)
        self.target_feature_net = GridFeatureNet(c, h, cfg.feature_dim).to(self.device)
        self.target_feature_net.load_state_dict(self.feature_net.state_dict())
        self.optim = torch.optim.Adam(self.feature_net.parameters(), lr=cfg.lr)

        self.blr = BayesianLinearHead(
            cfg.action_dim,
            cfg.feature_dim,
            cfg.blr_lambda,
            cfg.blr_noise_var,
            cfg.posterior_jitter,
            cfg.seed,
        )
        self.blr.sample()
        self.replay = ReplayBuffer(cfg.replay_capacity, cfg.obs_shape, seed=cfg.seed)
        self.train_steps = 0
        self.posterior_rebuilds = 0
        self.features_frozen = False
        self.last_predictive_stats = {
            "predictive_epistemic_std_mean": float("nan"),
            "predictive_selected_std_mean": float("nan"),
            "predictive_max_std_mean": float("nan"),
        }

    def resample_policy(self) -> None:
        self.blr.sample()

    def _maybe_freeze_features(self) -> None:
        freeze_at = self.cfg.freeze_feature_after_steps
        if freeze_at is not None and not self.features_frozen and self.train_steps >= freeze_at:
            self.features_frozen = True
            for parameter in self.feature_net.parameters():
                parameter.requires_grad_(False)

    @torch.no_grad()
    def _features(self, obs: np.ndarray) -> np.ndarray:
        x = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        if x.ndim == len(self.cfg.obs_shape):
            x = x.unsqueeze(0)
        return self.feature_net(x).cpu().numpy()

    @torch.no_grad()
    def act(self, obs: np.ndarray, use_sample: bool = True, action_mask: np.ndarray | None = None) -> int:
        phi = self._features(obs)[0]
        w = self.blr.sampled_w if use_sample else self.blr.mu
        q = w @ phi
        if action_mask is not None:
            allowed = np.asarray(action_mask, dtype=bool)
            if not np.any(allowed):
                raise ValueError("No allowed action in action_mask.")
            q = q.copy()
            q[~allowed] = -1e9
        return int(np.argmax(q))

    @torch.no_grad()
    def q_mean_and_variance(self, obs: np.ndarray, include_noise: bool = False) -> tuple[np.ndarray, np.ndarray]:
        phi = self._features(obs)
        means = phi @ self.blr.mu.T
        variances = self.blr.predictive_variance(phi, include_noise=include_noise)
        return means, variances

    def train_step(self) -> dict[str, float]:
        if len(self.replay) < self.cfg.batch_size:
            return {"loss": 0.0, "q_mean": 0.0, "posterior_rebuilds": float(self.posterior_rebuilds), **self.last_predictive_stats}

        self._maybe_freeze_features()
        batch = self.replay.sample(self.cfg.batch_size)
        if self.features_frozen:
            loss = 0.0
            q_mean = self._batch_q_mean(batch)
        else:
            loss, q_mean = self._gradient_update(batch)

        self.train_steps += 1
        if self.train_steps % self.cfg.target_update_period == 0:
            self.target_feature_net.load_state_dict(self.feature_net.state_dict())

        if len(self.replay) >= self.cfg.posterior_min_samples and self.train_steps % self.cfg.posterior_update_period == 0:
            if self.cfg.posterior_mode == "rebuild":
                self.rebuild_posterior_from_replay()
            else:
                posterior_batch = self.replay.sample(min(self.cfg.posterior_replay_size, len(self.replay)))
                self.update_posterior_cumulative(posterior_batch)
            self.resample_policy()
            self.last_predictive_stats = self.uncertainty_metrics()

        return {"loss": float(loss), "q_mean": float(q_mean), "posterior_rebuilds": float(self.posterior_rebuilds), **self.last_predictive_stats}

    @torch.no_grad()
    def _batch_q_mean(self, batch: Batch) -> float:
        obs = torch.as_tensor(batch.obs, dtype=torch.float32, device=self.device)
        phi = self.feature_net(obs)
        mean_w = torch.as_tensor(self.blr.mu, dtype=torch.float32, device=self.device)
        q_all = phi @ mean_w.t()
        actions = torch.as_tensor(batch.actions, dtype=torch.long, device=self.device)
        q = q_all.gather(1, actions[:, None]).squeeze(1)
        return float(q.mean().item())

    def _gradient_update(self, batch: Batch) -> tuple[float, float]:
        obs = torch.as_tensor(batch.obs, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch.actions, dtype=torch.long, device=self.device)
        rewards = torch.as_tensor(batch.rewards, dtype=torch.float32, device=self.device)
        next_obs = torch.as_tensor(batch.next_obs, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(batch.dones, dtype=torch.float32, device=self.device)
        mean_w = torch.as_tensor(self.blr.mu, dtype=torch.float32, device=self.device)

        phi = self.feature_net(obs)
        q_all = phi @ mean_w.t()
        q = q_all.gather(1, actions[:, None]).squeeze(1)
        with torch.no_grad():
            next_phi_online = self.feature_net(next_obs)
            next_q_online = next_phi_online @ mean_w.t()
            next_actions = next_q_online.argmax(dim=1)
            next_phi_target = self.target_feature_net(next_obs)
            next_q_target = next_phi_target @ mean_w.t()
            next_q = next_q_target.gather(1, next_actions[:, None]).squeeze(1)
            targets = rewards + self.cfg.gamma * (1.0 - dones) * next_q

        loss = F.mse_loss(q, targets)
        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.feature_net.parameters(), self.cfg.grad_clip_norm)
        self.optim.step()
        return float(loss.item()), float(q.mean().item())

    def _posterior_replay_indices(self) -> np.ndarray:
        n = len(self.replay)
        sample_size = min(self.cfg.posterior_replay_size, n)
        if sample_size == n:
            return np.arange(n, dtype=np.int64)
        return self.rng.choice(n, size=sample_size, replace=False).astype(np.int64)

    @torch.no_grad()
    def _compute_targets_and_features(self, obs_np, actions_np, rewards_np, next_obs_np, dones_np, mean_w_np=None):
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device)
        next_obs = torch.as_tensor(next_obs_np, dtype=torch.float32, device=self.device)
        if mean_w_np is None:
            mean_w_np = self.blr.mu.copy()
        mean_w = torch.as_tensor(mean_w_np, dtype=torch.float32, device=self.device)
        phi = self.feature_net(obs)
        next_phi_online = self.feature_net(next_obs)
        next_q_online = next_phi_online @ mean_w.t()
        next_actions = next_q_online.argmax(dim=1)
        next_phi_target = self.target_feature_net(next_obs)
        next_q_target = next_phi_target @ mean_w.t()
        next_q = next_q_target.gather(1, next_actions[:, None]).squeeze(1)
        rewards = torch.as_tensor(rewards_np, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(dones_np, dtype=torch.float32, device=self.device)
        targets = rewards + self.cfg.gamma * (1.0 - dones) * next_q
        return phi.cpu().numpy().astype(np.float32), np.asarray(actions_np, dtype=np.int64), targets.cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def rebuild_posterior_from_replay(self) -> None:
        indices = self._posterior_replay_indices()
        # Freeze the old posterior mean while constructing all Bellman targets.
        # Resetting A/b must not accidentally turn every bootstrap target into zero.
        target_mean_w = self.blr.mu.copy()
        self.blr.reset()
        chunk_size = max(1, int(self.cfg.posterior_chunk_size))
        for start in range(0, len(indices), chunk_size):
            idx = indices[start:start + chunk_size]
            phi, actions, targets = self._compute_targets_and_features(
                self.replay.obs[idx],
                self.replay.actions[idx],
                self.replay.rewards[idx],
                self.replay.next_obs[idx],
                self.replay.dones[idx],
                mean_w_np=target_mean_w,
            )
            self.blr.accumulate(phi, actions, targets)
        self.blr.finalize()
        self.posterior_rebuilds += 1

    @torch.no_grad()
    def update_posterior_cumulative(self, batch: Batch) -> None:
        if not self.features_frozen:
            raise RuntimeError("Cumulative posterior update requested while features are changing.")
        phi, actions, targets = self._compute_targets_and_features(batch.obs, batch.actions, batch.rewards, batch.next_obs, batch.dones)
        self.blr.update(phi, actions, targets, reset=False)

    @torch.no_grad()
    def uncertainty_metrics(self, sample_size: int = 256) -> dict[str, float]:
        if len(self.replay) == 0:
            return {
                "predictive_epistemic_std_mean": float("nan"),
                "predictive_selected_std_mean": float("nan"),
                "predictive_max_std_mean": float("nan"),
            }
        n = len(self.replay)
        k = min(int(sample_size), n)
        indices = self.rng.choice(n, size=k, replace=False)
        phi = self._features(self.replay.obs[indices])
        variances = self.blr.predictive_variance(phi, include_noise=False)
        std = np.sqrt(np.maximum(variances, 0.0))
        selected_std = std[np.arange(k), self.replay.actions[indices]]
        return {
            "predictive_epistemic_std_mean": float(std.mean()),
            "predictive_selected_std_mean": float(selected_std.mean()),
            "predictive_max_std_mean": float(std.max(axis=1).mean()),
        }

    def save(self, path: str) -> None:
        torch.save({
            "cfg": self.cfg.__dict__,
            "feature_net": self.feature_net.state_dict(),
            "target_feature_net": self.target_feature_net.state_dict(),
            "optim": self.optim.state_dict(),
            "blr": self.blr.state_dict(),
            "train_steps": self.train_steps,
            "posterior_rebuilds": self.posterior_rebuilds,
            "features_frozen": self.features_frozen,
            "last_predictive_stats": self.last_predictive_stats,
        }, path)

    def load(self, path: str) -> None:
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(path, map_location=self.device)
        self.feature_net.load_state_dict(checkpoint["feature_net"])
        self.target_feature_net.load_state_dict(checkpoint.get("target_feature_net", checkpoint["feature_net"]))
        if "optim" in checkpoint:
            self.optim.load_state_dict(checkpoint["optim"])
        self.blr.load_state_dict(checkpoint["blr"])
        self.train_steps = int(checkpoint.get("train_steps", 0))
        self.posterior_rebuilds = int(checkpoint.get("posterior_rebuilds", 0))
        self.features_frozen = bool(checkpoint.get("features_frozen", False))
        self.last_predictive_stats = checkpoint.get("last_predictive_stats", self.last_predictive_stats)
        if self.features_frozen:
            for parameter in self.feature_net.parameters():
                parameter.requires_grad_(False)

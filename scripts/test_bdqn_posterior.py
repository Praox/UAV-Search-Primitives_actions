from __future__ import annotations

import argparse
import numpy as np

from uav_search_belief20.agents.bdqn_agent import BayesianLinearHead


def test_sequential_equals_batch() -> None:
    rng = np.random.default_rng(7)
    n, feature_dim, action_dim = 800, 5, 3
    phi = rng.normal(size=(n, feature_dim)).astype(np.float32)
    actions = rng.integers(0, action_dim, size=n)
    true_w = rng.normal(size=(action_dim, feature_dim))
    targets = np.sum(phi * true_w[actions], axis=1) + 0.1 * rng.normal(size=n)

    batch_head = BayesianLinearHead(action_dim, feature_dim, lam=1.0, noise_var=0.01, seed=1)
    batch_head.update(phi, actions, targets, reset=True)

    seq_head = BayesianLinearHead(action_dim, feature_dim, lam=1.0, noise_var=0.01, seed=1)
    for start in range(0, n, 37):
        seq_head.accumulate(phi[start:start + 37], actions[start:start + 37], targets[start:start + 37])
    seq_head.finalize()

    mu_error = np.max(np.abs(batch_head.mu - seq_head.mu))
    cov_error = np.max(np.abs(batch_head.cov - seq_head.cov))
    print(f"[sequential=batch] max mu error  = {mu_error:.3e}")
    print(f"[sequential=batch] max cov error = {cov_error:.3e}")
    assert mu_error < 1e-4
    assert cov_error < 1e-4


def test_uncertainty_contracts() -> None:
    rng = np.random.default_rng(11)
    head = BayesianLinearHead(1, 2, lam=1.0, noise_var=0.1, seed=11)
    query = np.array([1.0, 0.5], dtype=np.float32)
    before = float(head.predictive_variance(query)[0])
    phi = np.repeat(query[None, :], repeats=500, axis=0)
    targets = (2.0 * phi[:, 0] - 0.5 * phi[:, 1] + 0.1 * rng.normal(size=500)).astype(np.float32)
    actions = np.zeros((500,), dtype=np.int64)
    head.update(phi, actions, targets, reset=False)
    after = float(head.predictive_variance(query)[0])
    print(f"[uncertainty] before = {before:.6f}")
    print(f"[uncertainty] after  = {after:.6f}")
    assert after < before * 0.05


def run_two_arm_bandit(seed: int, thompson: bool, steps: int) -> float:
    rng = np.random.default_rng(seed)
    head = BayesianLinearHead(2, 1, lam=1.0, noise_var=0.05**2, seed=seed)
    phi = np.ones((1, 1), dtype=np.float32)
    total = 0.0
    for _ in range(steps):
        if thompson:
            head.sample()
            q = head.sampled_w[:, 0]
        else:
            q = head.mu[:, 0]
        action = int(np.argmax(q))
        mean_reward = 0.2 if action == 0 else 1.0
        reward = float(mean_reward + 0.05 * rng.normal())
        total += reward
        head.update(phi, np.array([action]), np.array([reward], dtype=np.float32), reset=False)
    return total


def test_thompson_helps(seeds: int = 100, steps: int = 100) -> None:
    ts_returns = [run_two_arm_bandit(seed, True, steps) for seed in range(seeds)]
    greedy_returns = [run_two_arm_bandit(seed, False, steps) for seed in range(seeds)]
    ts_mean = float(np.mean(ts_returns))
    greedy_mean = float(np.mean(greedy_returns))
    print(f"[bandit] Thompson mean return = {ts_mean:.2f}")
    print(f"[bandit] Greedy mean return   = {greedy_mean:.2f}")
    print(f"[bandit] Improvement          = {ts_mean - greedy_mean:.2f}")
    assert ts_mean > greedy_mean + 20.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bandit-seeds", type=int, default=100)
    parser.add_argument("--bandit-steps", type=int, default=100)
    args = parser.parse_args()
    test_sequential_equals_batch()
    test_uncertainty_contracts()
    test_thompson_helps(args.bandit_seeds, args.bandit_steps)
    print("All corrected-BDQN posterior tests passed.")


if __name__ == "__main__":
    main()

# Automated uncertainty study

Target repository commit:

```text
2de9426fb6340be701223c980c1b027485b8cf4f  full BDQN-Qmix
```

## Scientific question

The study separates four effects:

1. **Single-agent Bayesian effect**: `BDQN - DDQN`.
2. **Shared-policy Bayesian effect**: `shared_bdqn - shared_ddqn`.
3. **Centralized credit-assignment effect**: `qmix_ddqn - shared_ddqn`.
4. **Bayesian effect inside QMIX**: Bayesian-QMIX minus `qmix_ddqn`.
5. **Posterior coordination effect**: shared minus independent Bayesian-QMIX sampling.

The main noisy condition should initially be `p_detect=0.7`. `p_detect=0.6` is a later stress test, not a mandatory second full grid.

## Why retraining is necessary

Changing the detection probability changes the observation distribution, reward frequency and useful revisit behaviour. A checkpoint trained at `p_detect=1.0` and only evaluated at `0.7` measures **zero-shot robustness**. Training and evaluating at `0.7` measures whether an algorithm can **learn under uncertainty**. These are different questions.

## Files added

```text
scripts/run_uncertainty_study.py
scripts/evaluate_multi_local_baselines.py
scripts/aggregate_uncertainty_study.py
src/uav_search_belief20/marl/multi_local_baselines.py
docs/UNCERTAINTY_STUDY.md
```

No existing source file is overwritten.

## 1. Smoke tests

```bash
python scripts/run_uncertainty_study.py smoke --device cpu
```

This runs the existing single-UAV and Bayesian-QMIX smoke tests, then evaluates the new multi-UAV random and local-frontier baselines for two worlds.

## 2. Recommended screening experiment

### Single UAV

```bash
python scripts/run_uncertainty_study.py single-screen \
  --device cuda \
  --probabilities 1.0,0.7 \
  --single-algos ddqn,bdqn
```

This runs:

- DDQN;
- BDQN posterior-mean evaluation;
- BDQN posterior-sampled evaluation;
- random, frontier and oracle baselines;
- seeds 42, 43 and 44;
- 400 training episodes;
- 300 paired evaluation worlds.

DQN is deliberately omitted because it does not isolate the Bayesian question. Add it only for completeness:

```bash
--single-algos dqn,ddqn,bdqn
```

### Multi UAV

```bash
python scripts/run_uncertainty_study.py multi-screen \
  --device cuda \
  --probabilities 1.0,0.7 \
  --multi-algos shared_ddqn,shared_bdqn,qmix_ddqn,bayes_qmix_shared,bayes_qmix_independent \
  --bayes-kl-weight 0.001
```

This also evaluates no-learning multi-UAV random and local-frontier policies.

### Everything with one command

```bash
python scripts/run_uncertainty_study.py full-screen \
  --device cuda \
  --probabilities 1.0,0.7 \
  --bayes-kl-weight 0.001
```

The runner skips valid JSON results and reuses existing checkpoints inside its own run root.

## 3. Interpretation

Do not interpret only the raw drop between `p=1.0` and `p=0.7`. Every method should deteriorate when observations become less reliable.

The central statistic is the relative Bayesian advantage:

```text
BDQN - DDQN
shared-BDQN - shared-DDQN
Bayesian-QMIX - QMIX-DDQN
```

The aggregator also calculates a difference-in-differences:

```text
(Bayesian - deterministic at p=0.7)
-
(Bayesian - deterministic at p=1.0)
```

A positive value for completion means that Bayesian uncertainty becomes relatively more useful under noisy sensing, even when both algorithms lose absolute performance.

Outputs:

```text
logs/uncertainty_study/aggregate/
├── uncertainty_runs.csv
├── uncertainty_summary.csv
├── uncertainty_paired.csv
├── uncertainty_difference_in_differences.csv
└── uncertainty_summary.md
```

Student-t 95% intervals are used across training seeds.

## 4. Confirmation

Do not confirm all algorithms. Select one Bayesian-QMIX sampling mode from the screen, then run only the thesis-critical comparisons at `p=0.7`.

Example with shared Bayesian-QMIX selected:

```bash
python scripts/run_uncertainty_study.py full-confirm \
  --device cuda \
  --probabilities 0.7 \
  --single-algos ddqn,bdqn \
  --multi-algos qmix_ddqn,bayes_qmix_shared \
  --bayes-kl-weight 0.001
```

Confirmation defaults:

- seeds 42--48;
- 1000 training episodes;
- 1000 final evaluation worlds.

## 5. Optional p=0.6 stress test

Run `p=0.6` only after the `0.7` screen:

```bash
python scripts/run_uncertainty_study.py full-screen \
  --device cuda \
  --probabilities 0.6 \
  --single-algos ddqn,bdqn \
  --multi-algos qmix_ddqn,bayes_qmix_shared,bayes_qmix_independent
```

Use it when:

- the `0.7` Bayesian effect is promising and needs a stronger stress test;
- the `0.7` result is ambiguous;
- the thesis explicitly studies uncertainty intensity.

Do not automatically confirm both `0.6` and `0.7` with seven seeds.

## 6. What the new baselines answer

### Random multi-UAV

Answers whether learned coordination beats valid but uncontrolled movement.

### Local-frontier multi-UAV

Each UAV uses only its own memory and position. It pursues locally known incomplete targets; otherwise it greedily maximizes new cells in its next sensor footprint. It uses no hidden target truth, no other UAV memory and no mixer state.

This answers whether QMIX and Bayesian-QMIX improve over a simple domain-structured decentralized controller, not only over another neural model.

## Thesis decisions produced by the study

| Comparison | Thesis question |
|---|---|
| BDQN vs DDQN at p=1 and p=.7 | Does single-agent Bayesian exploration benefit from sensor uncertainty? |
| Shared BDQN vs shared DDQN | Does posterior exploration help without centralized credit assignment? |
| QMIX-DDQN vs shared DDQN | Does centralized factorisation improve coordination? |
| Bayesian-QMIX vs QMIX-DDQN | Does Bayesian uncertainty add value after credit assignment is solved? |
| Shared vs independent Bayesian-QMIX | Is common commitment or exploration diversity more useful? |
| Learned methods vs frontier | Is the contribution operational performance or primarily algorithmic analysis? |

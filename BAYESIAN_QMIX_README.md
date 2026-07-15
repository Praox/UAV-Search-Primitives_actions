# Bayesian-QMIX + posterior-sampling ablation patch

This patch targets repository commit:

```text
c559cf7665c63fb0bd43d1ffe8cf062801885b64  qmix tests
```

It keeps the validated local-memory environment and existing algorithms, then adds:

- `bayes_qmix_shared`: one posterior draw shared by all UAVs for one episode;
- `bayes_qmix_independent`: one independent draw per UAV from the same posterior;
- variational Bayesian last utility layer trained through the QMIX team TD loss;
- posterior-mean checkpoint selection and both sampled/mean final evaluations;
- simultaneous sensor-overlap logging;
- `privileged_truth` versus `memory_union` mixer-state modes;
- automated KL, sampling, noise, and state-information experiment suites;
- Student-t and z=1.96 confidence intervals in the aggregator.

## Install

Copy the archive over the repository root and reinstall editable mode:

```bash
unzip -o bayesian_qmix_posterior_sampling_patch.zip \
  -d UAV-Search-Primitives_actions
cd UAV-Search-Primitives_actions
pip install -e .
```

## First command

```bash
python scripts/run_multi_local_suite.py bayes-smoke --device cpu
```

Expected output:

```text
Bayesian-QMIX smoke and CTDE tests: OK
```

## Recommended experiment order

### 1. Cheap KL calibration

```bash
python scripts/run_multi_local_suite.py bayes-kl-screen --device cuda
```

Defaults:

- shared posterior sampling only;
- KL weights `1e-4, 1e-3, 1e-2`;
- seeds `42,43,44`;
- 250 training episodes;
- 150 final evaluation worlds.

Choose one KL weight using completion, completed value, posterior standard deviation,
seed stability, and collision rate. Do not choose from scalar reward alone.

### 2. Shared versus independent posterior screening

```bash
python scripts/run_multi_local_suite.py bayes-screen \
  --device cuda \
  --bayes-kl-weight 0.001
```

The existing deterministic QMIX screen in `logs/multi_local/screen` is reused by the
aggregator, so it is not retrained.

### 3. Seven-seed confirmation

```bash
python scripts/run_multi_local_suite.py bayes-confirm \
  --device cuda \
  --bayes-kl-weight 0.001
```

Defaults become 7 seeds, 1000 training episodes, and 1000 paired evaluation worlds.
The existing deterministic QMIX confirmation is reused from `logs/multi_local/confirm`.

### 4. Extend seeds only when necessary

```bash
python scripts/run_multi_local_suite.py bayes-confirm \
  --device cuda \
  --seeds 49,50,51,52,53,54,55 \
  --bayes-kl-weight 0.001
```

This appends new runs to the same experiment directory. Aggregation then covers seeds
42--55 without rerunning completed jobs.

### 5. Noisy-detection screen

```bash
python scripts/run_multi_local_suite.py bayes-noise-screen \
  --device cuda \
  --detection-probabilities 1.0,0.8,0.6
```

This runs deterministic QMIX and both Bayesian-QMIX sampling modes on every noise
level, using identical world seeds.

### 6. Mixer-state information ablation

```bash
python scripts/run_multi_local_suite.py bayes-state-screen \
  --device cuda \
  --state-modes privileged_truth,memory_union
```

Both state modes have the same dimension. `memory_union` uses only target facts present
in at least one local UAV memory and therefore avoids a network-capacity confound.

## Main outputs

```text
logs/multi_local/bayesian_qmix/
├── kl_screen/
├── screen/deterministic_privileged/
├── confirm/deterministic_privileged/
├── noise_screen/
└── state_screen/
```

Every experiment folder contains:

```text
<algo>_seed<seed>_train<episodes>.log
<algo>_seed<seed>_train<episodes>_eval.json
aggregate/
  multi_runs.csv
  multi_summary.csv
  multi_summary.md
  paired_bayes_shared_vs_independent.csv
  paired_bayes_shared_vs_qmix.csv
  paired_bayes_independent_vs_qmix.csv
  paired_all_long.csv
```

Training directories retain `metrics.csv`, `best.pt`, `latest.pt`, and
`run_config.json`.

See `docs/BAYESIAN_QMIX_PLAN.md` for the scientific rationale and interpretation.

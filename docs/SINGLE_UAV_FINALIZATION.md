# Single-UAV definitive study

## Experimental variants

| Variant | Boundary mask | Track-progress map | Tracking reward |
|---|---:|---:|---:|
| `v3` | no | no | 1.0x |
| `A` | yes | no | 1.0x |
| `B` | no | yes | 1.0x |
| `C` | yes | yes | 1.0x |
| `D` | yes | yes | 1.5x |

This layout isolates both main changes independently before testing their combination and a modest reward adjustment.

## Files changed

- `src/uav_search_belief20/envs/drone_memory.py`
  - adds a normalized tracking-progress map;
- `src/uav_search_belief20/envs/primitive_search_env.py`
  - adds boundary masking, optional progress observation, and stable reward-part keys;
- `src/uav_search_belief20/experiments/single_ablation.py`
  - centralizes all five presets;
- `src/uav_search_belief20/evaluation.py`
  - centralizes paired evaluation and reward decomposition;
- `src/uav_search_belief20/baselines.py`
  - implements random, frontier, and target-aware oracle policies;
- `scripts/train.py`
  - trains all variants and logs reward decomposition plus BDQN uncertainty;
- `scripts/evaluate.py`
  - performs final paired evaluation on a fixed world set;
- `scripts/evaluate_baselines.py`
  - evaluates the three non-learning baselines;
- `scripts/aggregate_single_results.py`
  - aggregates means, standard deviations, 95% confidence intervals, and paired BDQN–DDQN differences;
- `scripts/run_single_suite.py`
  - launches the experiment pipeline with resume support;
- `scripts/smoke_test_single_v4.py`
  - validates observation shapes, masks, progress maps, reward sums, networks, and baselines.

## Recommended evening sequence

### 1. Validate

```bash
python scripts/run_single_suite.py smoke
```

### 2. Aggregate the current seven-seed benchmark

```bash
python scripts/run_single_suite.py existing
```

Read:

```text
logs/single_suite/existing_v3_aggregate/single_summary.md
logs/single_suite/existing_v3_aggregate/single_paired_bdqn_vs_ddqn.csv
```

### 3. Screen all environment changes

```bash
python scripts/run_single_suite.py screen --device cuda
```

Default screening budget:

- variants: `v3,A,B,C,D`;
- algorithm: DDQN;
- training seeds: `42,43,44`;
- 400 training episodes;
- 300 final evaluation episodes;
- random/frontier/oracle baselines on 500 episodes.

Faster sanity run:

```bash
python scripts/run_single_suite.py screen \
  --device cuda \
  --train-episodes 150 \
  --final-eval-episodes 100 \
  --baseline-episodes 100
```

### 4. Select the environment

Prefer the variant that improves, in order:

1. completed targets;
2. completed target value;
3. completion/detection ratio;
4. stability across seeds;
5. coverage and first-detection time;
6. low boundary or idle behavior.

Do not select a variant only because its scalar reward is larger.

### 5. Confirm DDQN versus BDQN

The default confirmation tests `v3,C,D` with both algorithms and seven seeds:

```bash
python scripts/run_single_suite.py confirm --device cuda
```

To confirm only the chosen variant:

```bash
python scripts/run_single_suite.py confirm \
  --device cuda \
  --variants C
```

## Output layout

```text
runs/single_suite/
  screen/<variant>/<algo>_seed<seed>_train<episodes>/
  confirm/<variant>/<algo>_seed<seed>_train<episodes>/

logs/single_suite/
  baselines/<variant>/
  screen/<variant>/
  screen/aggregate/
  confirm/<variant>/
  confirm/aggregate/
```

## Scientific interpretation

- `A - v3` estimates the effect of removing physically invalid boundary choices.
- `B - v3` estimates the effect of making tracking progress observable.
- `C` tests whether both corrections are complementary.
- `D - C` isolates the effect of stronger tracking shaping after observability is fixed.
- Random/frontier/oracle establish a lower baseline, a classical heuristic reference, and an approximate upper bound.

The final single-UAV environment should be frozen after confirmation. Multi-UAV work should then reuse its observation semantics, reward accounting, paired test worlds, and aggregation pipeline.

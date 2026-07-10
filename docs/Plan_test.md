# UAV Search-and-Track — Reward experiment plan

## Goal
Compare reward versions for the primitive-action UAV search-and-track repo.

Do not compare only `reward_mean` across reward versions, because the reward scale changes. Compare these task metrics first:

1. `completed_mean`
2. `completed_value_mean`
3. `detected_mean`
4. `detected_value_mean`
5. `sensor_coverage_ratio_mean`
6. `boundary_hit_ratio`
7. `stay_ratio`
8. `revisit_ratio`

## Reward versions

### v1_baseline
Current reward in `primitive_search_env.py`.
Already completed for DDQN and BDQN seeds 42, 43, 44.

### v2_frontier
Patch idea:
- reward newly observed sensor cells, not only new drone position;
- stronger boundary penalty;
- penalize `stay` only when it does not produce tracking progress;
- increase detect / track / completion rewards moderately.

First run only DDQN seed 43. If it improves the task metrics, run all seeds and BDQN.

### v3_frontier_tracking
Only use if v2 is not enough.
- If coverage still low: increase `new_observed_cell_bonus` and cap.
- If boundary hits still high: increase `boundary_penalty`.
- If coverage improves but completion drops: increase tracking and completion rewards.

## Main commands

### Smoke test after patch
```bash
python scripts/smoke_test.py
```

### Debug v2 DDQN
```bash
python scripts/train.py \
  --algo ddqn \
  --episodes 20 \
  --device cuda \
  --train-every 4 \
  --learning-starts 1000 \
  --eval-every 20 \
  --eval-episodes 5 \
  --seed 42 \
  --run-dir runs/v2_frontier/debug_ddqn_seed42
```

### First serious v2 test: DDQN seed 43
```bash
python scripts/train.py \
  --algo ddqn \
  --episodes 1000 \
  --device cuda \
  --train-every 4 \
  --learning-starts 1000 \
  --eval-every 50 \
  --eval-episodes 10 \
  --seed 43 \
  --run-dir runs/v2_frontier/ddqn_seed43_1000
```

```bash
python scripts/evaluate.py \
  --algo ddqn \
  --checkpoint runs/v2_frontier/ddqn_seed43_1000/best.pt \
  --episodes 1000
```

## Full test grid

For each reward version that you validate:
- algorithms: `dqn`, `ddqn`, `bdqn`
- seeds: `42`, `43`, `44`
- episodes: `1000`
- final evaluation episodes: `1000`

Use this standard train config:

```bash
--device cuda --train-every 4 --learning-starts 1000 --eval-every 50 --eval-episodes 10
```

For BDQN add:

```bash
--posterior-update-period 500
```

## Decision rules

Reward v2 is better if, relative to v1:

- `completed_mean` is stable or higher;
- `completed_value_mean` is stable or higher;
- `sensor_coverage_ratio_mean` improves;
- `boundary_hit_ratio` decreases or stays acceptable;
- `stay_ratio` is not degenerate, roughly 0.15 to 0.40 depending on tracking behavior.

Use v3 only if v2 still fails one of these:

- `completed_mean < 0.45`
- `sensor_coverage_ratio_mean < 0.30`
- `boundary_hit_ratio > 0.20`
- `stay_ratio > 0.45` without higher completion
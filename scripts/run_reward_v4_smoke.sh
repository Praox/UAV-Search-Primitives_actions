#!/usr/bin/env bash
set -euo pipefail

DEVICE="${DEVICE:-cpu}"
EPISODES="${EPISODES:-300}"

python scripts/set_reward_v4_weights.py \
  --coverage 10 \
  --detection 1 \
  --progress 2 \
  --completion-scale 5

python scripts/train.py \
  --algo ddqn \
  --episodes "$EPISODES" \
  --device "$DEVICE" \
  --seed 42 \
  --reward-version v4_potential_simple \
  --eval-every 25 \
  --eval-episodes 30 \
  --run-dir runs/reward_v4_smoke/ddqn_c10_p2_seed42
  
#exple
DEVICE=mps EPISODES=100 \
bash scripts/run_reward_v4_smoke.sh
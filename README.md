# UAV Search Belief20 — primitive actions, DQN/DDQN/BDQN

```text
0 = STAY
1 = UP
2 = DOWN
3 = LEFT
4 = RIGHT
```

The goal is to keep the mental model of the original repo while making the experiments easier to compare:

- same hidden 20x20 world idea;
- same local sensor idea;
- same drone memory / belief map idea;
- same CNN feature extractor idea;
- shared replay-buffer interface;
- one deterministic branch: `DQN` / `DDQN`;
- one Bayesian branch: `BDQN`.

## Main conceptual change

In the original repo, the learned BDQN selected only:

```text
0 = SEARCH
1 = TRACK
```

Then a heuristic `MissionController` converted the mission into primitive movement.

Here the learned agent directly chooses:

```text
STAY, UP, DOWN, LEFT, RIGHT
```

There is no `MissionController` in the learning loop. Completion is handled by the environment: once a target is detected and the UAV remains within `track_radius`, tracking progress increases automatically. This makes the action `STAY` meaningful: after detection, the agent can choose to stay near the target to complete it, or leave to search more.

## Observation

The agent receives a memory state, not hidden truth:

```text
obs shape = (6, grid_size, grid_size)

0. drone position map
1. belief map
2. known target value map
3. completed target map
4. visited map
5. time remaining map
```

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Smoke test

```bash
python scripts/smoke_test.py
```

## Train single-drone baselines

DQN:

```bash
python scripts/train.py --algo dqn --episodes 500 --run-dir runs/dqn
```

DDQN:

```bash
python scripts/train.py --algo ddqn --episodes 500 --run-dir runs/ddqn
```
```bash
python scripts/train.py \
  --algo ddqn \
  --episodes 300 \
  --device mps \
  --train-every 4 \
  --learning-starts 1000 \
  --eval-every 50 \
  --eval-episodes 10 \
  --seed 0 \
  --run-dir runs/fast/ddqn_seed0_300
 ```
BDQN:

```bash
python scripts/train.py --algo bdqn --episodes 500 --run-dir runs/bdqn
```
```bash
python scripts/train.py \
  --algo ddqn \
  --episodes 300 \
  --device mps \
  --train-every 4 \
  --learning-starts 1000 \
  --eval-every 50 \
  --eval-episodes 10 \
  --seed 0 \
  --run-dir runs/fast/ddqn_seed0_300
 ```

 
## Evaluate

```bash
python scripts/evaluate.py --algo dqn --checkpoint runs/dqn/best.pt --episodes 200
python scripts/evaluate.py --algo ddqn --checkpoint runs/ddqn/best.pt --episodes 200
python scripts/evaluate.py --algo bdqn --checkpoint runs/bdqn/best.pt --episodes 200
```

## First MARL baseline: 3 drones shared BDQN independent

A minimal shared-parameter multi-drone baseline is included:

```bash
python scripts/train_shared_bdqn.py --episodes 500 --n-agents 3 --run-dir runs/shared_bdqn_3uav
```

This is not QMIX. It is the quick baseline: one shared BDQN policy, three UAVs, shared team memory, and team reward.

## Repo layout

```text
scripts/
  train.py                 # single-drone DQN / DDQN / BDQN
  evaluate.py              # single-drone evaluation
  smoke_test.py            # quick sanity test
  train_shared_bdqn.py     # simple 3-drone shared BDQN independent baseline
src/uav_search_belief20/
  envs/
    drone_memory.py
    primitive_search_env.py
    multi_drone_env.py
  agents/
    replay_buffer.py
    dqn_agent.py
    bdqn_agent.py
  models/
    networks.py
  marl/
    qmix_mixer.py          # QMIX building block, for next phase
    qmix_agent.py          # minimal skeleton, not the priority path yet
  actions.py
  utils.py
docs/
  ROADMAP.md
  GANTT.md
```

## Thesis path

1. DQN / DDQN single drone with primitive actions and belief map
2. BDQN single drone on exactly the same setup
3. 3 drones shared BDQN independent, no mixer
4. QMIX-DDQN
5. QMIX-BDQN

Example launch code :
```bash
python run_tests.py v2_frontier ddqn 43
```